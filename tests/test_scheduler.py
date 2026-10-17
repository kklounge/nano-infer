"""Scheduler: continuous batching, mixed prefill/decode steps, chunked prefill, budgets, preemption."""

import pytest
import torch

from nanoinfer import LLM, SamplingParams
from nanoinfer.config import EngineConfig
from nanoinfer.engine.scheduler import Scheduler
from nanoinfer.engine.sequence import Sequence, SequenceStatus

BS = 4
EOS = 0


def make_scheduler(num_blocks=100, **kw):
    cfg = EngineConfig(model="unused", block_size=BS, max_model_len=kw.pop("max_model_len", 64),
                       kv_watermark=kw.pop("kv_watermark", 0.0), **kw)
    return Scheduler(cfg, num_blocks, eos_token_ids=[EOS])


def seq(n, max_tokens=8, start=1, **kw):
    return Sequence(list(range(start, start + n)), SamplingParams(max_tokens=max_tokens, **kw), BS)


def step(sch, token=7):
    """schedule + pretend-run: every sampling sequence gets `token`."""
    out = sch.schedule()
    out.sampled = list(out.sampling_seqs)        # (must be read before postprocess changes the counters)
    finished = sch.postprocess(out, [token] * len(out.sampled))
    return out, finished


def test_new_requests_are_prefilled_together_then_decoded():
    sch = make_scheduler()
    a, b = seq(5), seq(3)
    sch.add(a)
    sch.add(b)
    out = sch.schedule()
    assert out.seqs == [a, b] and out.num_decode == 0
    assert [s.num_scheduled_tokens for s in out.seqs] == [5, 3]
    assert a.status == SequenceStatus.RUNNING
    sch.postprocess(out, [7, 7])
    out = sch.schedule()
    assert out.seqs == [a, b] and out.num_decode == 2 and out.num_batched_tokens == 2


def test_new_request_joins_running_batch_in_a_mixed_step():
    """Continuous batching: a request that arrives later is prefilled IN THE SAME STEP as the
    running requests' decode tokens (decode never stalls)."""
    sch = make_scheduler()
    a = seq(5)
    sch.add(a)
    step(sch)
    step(sch)                                   # a decodes alone
    b = seq(3)
    sch.add(b)
    out = sch.schedule()
    assert out.seqs == [a, b]                   # decode tokens first, then the new prompt
    assert out.num_decode == 1 and out.num_prefill == 1
    assert out.num_batched_tokens == 1 + 3
    sch.postprocess(out, [7, 7])
    out = sch.schedule()
    assert out.num_decode == 2


def test_max_num_seqs_limits_batch():
    sch = make_scheduler(max_num_seqs=2)
    for _ in range(5):
        sch.add(seq(3))
    out = sch.schedule()
    assert len(out.seqs) == 2
    assert len(sch.waiting) == 3


def test_token_budget_without_chunking_waits_for_next_step():
    sch = make_scheduler(max_num_batched_tokens=64, max_model_len=64, enable_chunked_prefill=False)
    a, b = seq(40), seq(30, start=100)
    sch.add(a)
    sch.add(b)
    out = sch.schedule()
    assert out.seqs == [a]                      # 40 + 30 > 64 -> b waits for the next step


def test_chunked_prefill_splits_long_prompt():
    sch = make_scheduler(max_num_batched_tokens=16, max_model_len=64, max_num_seqs=4)
    a = seq(40)
    sch.add(a)
    sizes = []
    for _ in range(3):
        out, _ = step(sch)
        sizes.append(a.num_computed_tokens)
        assert out.sampled == ([] if a.num_computed_tokens < 40 else [a])
    assert sizes == [16, 32, 40]
    assert a.num_completion_tokens == 1         # sampled only after the LAST chunk
    out = sch.schedule()
    assert out.num_decode == 1


def test_chunked_prefill_shares_budget_with_decodes():
    sch = make_scheduler(max_num_batched_tokens=16, max_model_len=64)
    a = seq(4)
    sch.add(a)
    step(sch)                                   # a is decoding now
    b = seq(30, start=100)
    sch.add(b)
    out = sch.schedule()
    assert out.seqs == [a, b]
    assert a.num_scheduled_tokens == 1 and b.num_scheduled_tokens == 15   # 16 - 1 decode token
    sch.postprocess(out, [7])                   # only a samples; b is mid-prompt
    assert b.num_computed_tokens == 15 and b.num_completion_tokens == 0
    out = sch.schedule()
    assert a.num_scheduled_tokens == 1 and b.num_scheduled_tokens == 15
    sch.postprocess(out, [7, 7])                # b's last chunk -> b samples too
    assert b.num_completion_tokens == 1


def test_finish_on_eos_and_max_tokens_frees_blocks():
    sch = make_scheduler(num_blocks=10)
    a, b = seq(3, max_tokens=2), seq(3, max_tokens=5)
    sch.add(a)
    sch.add(b)
    out = sch.schedule()
    finished = sch.postprocess(out, [5, EOS])          # b hits EOS
    assert finished == [b] and b.finish_reason == "stop"
    out = sch.schedule()
    finished = sch.postprocess(out, [5])               # a reaches max_tokens=2
    assert finished == [a] and a.finish_reason == "length"
    assert sch.is_finished()
    assert sch.block_manager.num_free_blocks == 10


def test_ignore_eos():
    sch = make_scheduler()
    a = seq(3, max_tokens=3, ignore_eos=True)
    sch.add(a)
    for _ in range(2):
        _, fin = step(sch, EOS)
        assert fin == []
    _, fin = step(sch, EOS)
    assert fin == [a] and a.finish_reason == "length"


def test_stop_token_ids():
    sch = make_scheduler()
    a = seq(3, stop_token_ids=[42])
    sch.add(a)
    _, fin = step(sch, 42)
    assert fin == [a] and a.stop_reason == 42


def test_min_tokens_delays_stop():
    sch = make_scheduler()
    a = seq(3, max_tokens=5, min_tokens=2)
    sch.add(a)
    _, fin = step(sch, EOS)
    assert fin == []                            # only 1 token generated, min is 2
    _, fin = step(sch, EOS)
    assert fin == [a] and a.finish_reason == "stop"


def test_max_model_len_stops_generation():
    sch = make_scheduler(max_model_len=8)
    a = seq(6, max_tokens=100)
    sch.add(a)
    step(sch)
    _, fin = step(sch)
    assert fin == [a] and a.finish_reason == "length" and a.num_tokens == 8


def test_preemption_when_out_of_blocks():
    # 4 blocks of 4 tokens. Two sequences of 8 tokens use all 4 blocks after prefill.
    sch = make_scheduler(num_blocks=4, max_model_len=16, enable_prefix_caching=False)
    a, b = seq(8, max_tokens=8), seq(8, max_tokens=8, start=50)
    sch.add(a)
    sch.add(b)
    out = sch.schedule()
    assert out.seqs == [a, b]
    sch.postprocess(out, [9, 9])               # both now have 9 tokens -> next token needs a new block
    out = sch.schedule()
    assert out.seqs == [a]                     # a got the blocks ...
    assert out.num_preempted == 1
    assert b.status == SequenceStatus.WAITING  # ... b (newest) was preempted
    assert sch.waiting[0] is b and b.block_table == [] and b.num_computed_tokens == 0
    assert sch.num_preemptions == 1


def test_no_admission_in_a_step_that_preempted():
    sch = make_scheduler(num_blocks=4, max_model_len=16, enable_prefix_caching=False)
    a, b = seq(8), seq(8, start=50)
    sch.add(a)
    sch.add(b)
    step(sch)
    sch.add(seq(2, start=90))
    out = sch.schedule()
    assert out.num_preempted == 1 and out.num_prefill == 0


def test_watermark_keeps_room_for_running_requests():
    sch = make_scheduler(num_blocks=10, kv_watermark=0.2)       # keep 2 blocks free for growth
    a = seq(11)                                                  # 3 blocks (its next token fits too)
    sch.add(a)
    step(sch)
    b = seq(20, start=40)                                        # 5 blocks; 7 free - 2 reserved = 5 OK
    c = seq(4, start=80)                                         # 1 more block would eat the reserve
    sch.add(b)
    sch.add(c)
    out = sch.schedule()
    assert b in out.seqs and c not in out.seqs


def test_abort_releases_blocks():
    sch = make_scheduler(num_blocks=10)
    a = seq(5)
    sch.add(a)
    sch.schedule()
    assert sch.abort(a.seq_id) is a
    assert a.finish_reason == "abort"
    assert sch.is_finished() and sch.block_manager.num_free_blocks == 10


def test_engine_preemption_gives_same_greedy_output(tiny_qwen2):
    """With a tiny KV cache the engine must preempt and recompute, but greedy outputs stay identical."""
    torch.manual_seed(1)
    prompts = [torch.randint(0, 500, (n,)).tolist() for n in (20, 25, 30, 35, 18, 22)]
    sp = SamplingParams(temperature=0, max_tokens=40, ignore_eos=True)

    big = LLM(str(tiny_qwen2), dtype="float32", device="cpu", block_size=4, max_model_len=128)
    ref = big.generate(prompts, sp)
    assert big.stats.preemptions == 0

    for prefix in (False, True):    # with prefix caching, recomputation re-uses the blocks it had filled
        small = LLM(str(tiny_qwen2), dtype="float32", device="cpu", block_size=4, max_model_len=128,
                    num_kvcache_blocks=40, enable_prefix_caching=prefix, max_num_batched_tokens=48)
        out = small.generate(prompts, sp)
        assert small.stats.preemptions > 0
        assert [o.token_ids for o in out] == [o.token_ids for o in ref]


def test_kv_cache_smaller_than_one_sequence_is_rejected(tiny_qwen2):
    with pytest.raises(ValueError):
        LLM(str(tiny_qwen2), dtype="float32", device="cpu", block_size=4, max_model_len=128, num_kvcache_blocks=8)
