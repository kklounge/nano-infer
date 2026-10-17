"""Our model implementation vs Hugging Face transformers on tiny random models (float32).

Checks logits (prefill and every decode step) and greedy tokens, for both Qwen2 and Qwen3
architectures, both attention backends, several block sizes, batched and with prefix-cache hits.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from nanoinfer import LLM, SamplingParams
from nanoinfer.engine.sequence import Sequence

from conftest import hf_greedy

ATOL = 1e-4


def random_prompts(seed=0, lengths=(1, 5, 17, 33, 64)):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 500, (n,), generator=g).tolist() for n in lengths]


@pytest.fixture(scope="module", params=["qwen2", "qwen3"])
def tiny(request, tiny_qwen2, tiny_qwen3):
    path = tiny_qwen2 if request.param == "qwen2" else tiny_qwen3
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    return path, hf


def test_weights_loaded_exactly(tiny):
    path, hf = tiny
    llm = LLM(str(path), dtype="float32", device="cpu", max_model_len=256)
    ours = dict(llm.runner.model.named_parameters(remove_duplicate=False))
    from nanoinfer.model.loader import map_name
    for name, p in hf.named_parameters(remove_duplicate=False):
        packed = map_name(name, llm.model_config)       # q/k/v and gate/up live in merged matrices
        if packed is None:
            assert torch.equal(ours[name], p), name
        else:
            target, start, n = packed
            assert torch.equal(ours[target][start:start + n], p), name


@pytest.mark.parametrize("backend", ["sdpa", "naive"])
@pytest.mark.parametrize("block_size", [1, 4, 16])
def test_logits_match_hf_prefill_and_decode(tiny, backend, block_size):
    path, hf = tiny
    llm = LLM(str(path), dtype="float32", device="cpu", max_model_len=256, block_size=block_size,
              attn_backend=backend)
    prompts = random_prompts()
    seqs = [Sequence(p, SamplingParams(temperature=0, max_tokens=8), block_size) for p in prompts]
    for s in seqs:
        llm.scheduler.add(s)

    out = llm.scheduler.schedule()
    assert len(out.seqs) == len(prompts)            # all prompts prefilled in ONE packed batch
    for step in range(6):
        logits = llm.runner.compute_logits(out)
        with torch.no_grad():
            for i, s in enumerate(out.sampling_seqs):
                ref = hf(torch.tensor([s.token_ids])).logits[0, -1]
                torch.testing.assert_close(logits[i], ref, atol=ATOL, rtol=1e-4)
        llm.scheduler.postprocess(out, logits.argmax(-1).tolist())
        out = llm.scheduler.schedule()
        assert out.num_prefill == 0


@pytest.mark.parametrize("backend", ["sdpa", "naive"])
def test_logits_match_hf_in_mixed_and_chunked_steps(tiny, backend):
    """Decode tokens and prompt chunks in the same forward pass, chunks that continue a cached prefix:
    every logit row must still equal HF's."""
    path, hf = tiny
    llm = LLM(str(path), dtype="float32", device="cpu", max_model_len=256, block_size=4,
              attn_backend=backend, max_num_batched_tokens=24)
    prompts = random_prompts(seed=21, lengths=(6, 50, 9, 31))
    for p in prompts:
        llm.scheduler.add(Sequence(p, SamplingParams(temperature=0, max_tokens=12), 4))
    mixed = chunked = 0
    while llm.has_unfinished_requests():
        out = llm.scheduler.schedule()
        mixed += out.num_decode > 0 and out.num_prefill > 0
        chunked += any(s.num_computed_tokens + s.num_scheduled_tokens < s.num_tokens for s in out.seqs)
        logits = llm.runner.compute_logits(out)
        with torch.no_grad():
            for i, s in enumerate(out.sampling_seqs):
                ref = hf(torch.tensor([s.token_ids])).logits[0, -1]
                torch.testing.assert_close(logits[i], ref, atol=ATOL, rtol=1e-4)
        llm.scheduler.postprocess(out, logits.argmax(-1).tolist())
    assert mixed > 0 and chunked > 0


@pytest.mark.parametrize("budget", [7, 16, 33])
def test_greedy_tokens_match_hf_with_chunked_prefill(tiny, budget):
    path, hf = tiny
    prompts = random_prompts(seed=4, lengths=(3, 45, 29, 70, 1, 12))
    ref = [hf_greedy(hf, p, 20) for p in prompts]
    llm = LLM(str(path), dtype="float32", device="cpu", max_model_len=256, block_size=4,
              max_num_batched_tokens=budget, max_num_seqs=4)
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=20, ignore_eos=True))
    assert llm.stats.mixed_steps > 0
    assert [o.token_ids for o in outs] == ref


@pytest.mark.parametrize("backend", ["sdpa", "naive"])
def test_greedy_tokens_match_hf_batched(tiny, backend):
    path, hf = tiny
    prompts = random_prompts(seed=3, lengths=(3, 11, 29, 50, 7, 64, 2, 40))
    ref = [hf_greedy(hf, p, 32) for p in prompts]
    llm = LLM(str(path), dtype="float32", device="cpu", max_model_len=256, block_size=8, attn_backend=backend)
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=32, ignore_eos=True))
    assert [o.token_ids for o in outs] == ref


def test_greedy_matches_hf_with_prefix_cache_hits(tiny):
    path, hf = tiny
    system = random_prompts(seed=5, lengths=(37,))[0]
    prompts = [system + p for p in random_prompts(seed=6, lengths=(4, 9, 1, 20))]
    ref = [hf_greedy(hf, p, 16) for p in prompts]
    llm = LLM(str(path), dtype="float32", device="cpu", max_model_len=256, block_size=4)
    out0 = llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=16, ignore_eos=True))
    out = llm.generate(prompts[1:], SamplingParams(temperature=0, max_tokens=16, ignore_eos=True))
    assert all(o.num_cached_tokens == 36 for o in out)       # 9 full blocks of the 37-token system prompt
    assert [o.token_ids for o in out0 + out] == ref


def test_stop_token_ends_generation(tiny):
    path, hf = tiny
    llm = LLM(str(path), dtype="float32", device="cpu", max_model_len=256)
    prompt = random_prompts(seed=9, lengths=(10,))[0]
    first = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=5, ignore_eos=True))[0].token_ids
    # declare the 3rd generated token a stop token: generation must end at its first occurrence
    out = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=5, stop_token_ids=[first[2]]))[0]
    assert out.token_ids == first[:first.index(first[2]) + 1]
    assert out.finish_reason == "stop"


def test_static_padded_decode_matches_eager(tiny):
    """The CUDA-graph code path (static, padded buffers) is plain PyTorch: verify it on CPU."""
    path, _ = tiny
    llm = LLM(str(path), dtype="float32", device="cpu", max_model_len=128, block_size=4)
    runner = llm.runner
    prompts = random_prompts(seed=11, lengths=(5, 9, 14))
    seqs = [Sequence(p, SamplingParams(temperature=0), 4) for p in prompts]
    for s in seqs:
        llm.scheduler.add(s)
    out = llm.scheduler.schedule()
    llm.scheduler.postprocess(out, runner.run(out)[0])
    out = llm.scheduler.schedule()
    assert out.num_prefill == 0
    snapshot = runner.kv_cache.clone()
    eager = runner.compute_logits(out)

    runner.alloc_static_buffers(max_bs=8)
    for bucket in (16, 32, 128):                             # context-length buckets (sdpa graphs)
        runner.kv_cache.copy_(snapshot)
        runner.fill_static_inputs(out.seqs, b=8)             # 3 real rows + 5 padding rows
        with torch.inference_mode():
            hidden = runner.model(*runner.static_decode_args(8, bucket))
            static = runner.model.compute_logits(hidden[:3])
        torch.testing.assert_close(static, eager, atol=ATOL, rtol=1e-4)
