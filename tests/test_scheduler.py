"""Scheduler: continuous batching, prefill/decode alternation, budgets, preemption, stopping."""

import pytest
import torch

from nanoinfer import LLM, SamplingParams
from nanoinfer.config import EngineConfig
from nanoinfer.engine.scheduler import Scheduler
from nanoinfer.engine.sequence import Sequence, SequenceStatus

BS = 4
EOS = 0


def make_scheduler(num_blocks=100, **kw):
    cfg = EngineConfig(model="unused", block_size=BS, max_model_len=kw.pop("max_model_len", 64), **kw)
    return Scheduler(cfg, num_blocks, eos_token_ids=[EOS])


def seq(n, max_tokens=8, **kw):
    return Sequence(list(range(1, n + 1)), SamplingParams(max_tokens=max_tokens, **kw), BS)


def test_prefill_first_then_decode():
    sch = make_scheduler()
    a, b = seq(5), seq(3)
    sch.add(a)
    sch.add(b)
    batch, is_prefill = sch.schedule()
    assert is_prefill and batch == [a, b]
    assert a.status == SequenceStatus.RUNNING
    sch.postprocess(batch, [7, 7])
    batch, is_prefill = sch.schedule()
    assert not is_prefill and batch == [a, b]


def test_new_request_joins_running_batch():
    """Continuous batching: a request that arrives later is prefilled, then decoded together."""
    sch = make_scheduler()
    a = seq(5)
    sch.add(a)
    sch.postprocess(*_run(sch, [7]))
    sch.postprocess(*_run(sch, [7]))           # a decodes alone
    b = seq(3)
    sch.add(b)
    batch, is_prefill = sch.schedule()
    assert is_prefill and batch == [b]          # new arrival gets prefilled right away
    sch.postprocess(batch, [7])
    batch, is_prefill = sch.schedule()
    assert not is_prefill and set(batch) == {a, b}


def _run(sch, toks):
    batch, _ = sch.schedule()
    return batch, toks[:len(batch)]


def test_max_num_seqs_limits_batch():
    sch = make_scheduler(max_num_seqs=2)
    seqs = [seq(3) for _ in range(5)]
    for s in seqs:
        sch.add(s)
    batch, is_prefill = sch.schedule()
    assert is_prefill and len(batch) == 2
    assert len(sch.waiting) == 3


def test_token_budget_limits_prefill():
    sch = make_scheduler(max_num_batched_tokens=64, max_model_len=64)
    a, b = seq(40), seq(30)
    sch.add(a)
    sch.add(b)
    batch, _ = sch.schedule()
    assert batch == [a]                         # 40 + 30 > 64 -> b waits for the next step


def test_finish_on_eos_and_max_tokens_frees_blocks():
    sch = make_scheduler(num_blocks=10)
    a, b = seq(3, max_tokens=2), seq(3, max_tokens=5)
    sch.add(a)
    sch.add(b)
    batch, _ = sch.schedule()
    finished = sch.postprocess(batch, [5, EOS])        # b hits EOS
    assert finished == [b] and b.finish_reason == "stop"
    batch, _ = sch.schedule()
    finished = sch.postprocess(batch, [6])             # a reaches max_tokens=2
    assert finished == [a] and a.finish_reason == "length"
    assert sch.is_finished()
    assert sch.block_manager.num_free_blocks == 10


def test_ignore_eos():
    sch = make_scheduler()
    a = seq(3, max_tokens=3, ignore_eos=True)
    sch.add(a)
    batch, _ = sch.schedule()
    assert sch.postprocess(batch, [EOS]) == []


def test_stop_token_ids():
    sch = make_scheduler()
    a = seq(3, stop_token_ids=[42])
    sch.add(a)
    batch, _ = sch.schedule()
    assert sch.postprocess(batch, [42]) == [a]


def test_preemption_when_out_of_blocks():
    # 4 blocks of 4 tokens. Two sequences of 8 tokens use all 4 blocks after prefill.
    sch = make_scheduler(num_blocks=4, max_model_len=16, enable_prefix_caching=False)
    a, b = seq(8, max_tokens=8), seq(8, max_tokens=8)
    a.token_ids[0] = 100                       # different prompts
    sch.add(a)
    sch.add(b)
    batch, is_prefill = sch.schedule()
    assert is_prefill and batch == [a, b]
    sch.postprocess(batch, [9, 9])             # both now have 9 tokens -> next token needs a new block
    batch, is_prefill = sch.schedule()
    assert not is_prefill
    assert batch == [a]                        # a got the blocks ...
    assert b.status == SequenceStatus.WAITING  # ... b (newest) was preempted
    assert sch.waiting[0] is b and b.block_table == []
    assert sch.num_preemptions == 1


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

    small = LLM(str(tiny_qwen2), dtype="float32", device="cpu", block_size=4, max_model_len=128,
                num_kvcache_blocks=40, enable_prefix_caching=False)
    out = small.generate(prompts, sp)
    assert small.stats.preemptions > 0
    assert [o.token_ids for o in out] == [o.token_ids for o in ref]


def test_kv_cache_smaller_than_one_sequence_is_rejected(tiny_qwen2):
    with pytest.raises(ValueError):
        LLM(str(tiny_qwen2), dtype="float32", device="cpu", block_size=4, max_model_len=128, num_kvcache_blocks=8)
