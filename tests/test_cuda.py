"""GPU-only checks (skipped automatically on machines without CUDA).

Run these on a PC with an NVIDIA GPU: they exercise the CUDA code paths that the CPU-only
development machine could not test: SDPA on GPU, KV-cache sizing from free GPU memory,
and CUDA-graph capture/replay for decode.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from nanoinfer import LLM, SamplingParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


def _prompts():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(0, 500, (n,), generator=g).tolist() for n in (3, 17, 40, 9)]


def test_gpu_greedy_matches_hf_cpu_reference(tiny_qwen2):
    hf = AutoModelForCausalLM.from_pretrained(tiny_qwen2, dtype=torch.float32).eval()
    llm = LLM(str(tiny_qwen2), device="cuda", dtype="float32", max_model_len=256, block_size=8)
    outs = llm.generate(_prompts(), SamplingParams(temperature=0, max_tokens=16, ignore_eos=True))
    from conftest import hf_greedy
    assert [o.token_ids for o in outs] == [hf_greedy(hf, p, 16) for p in _prompts()]


def test_auto_sized_kv_cache_on_gpu(tiny_qwen2):
    llm = LLM(str(tiny_qwen2), device="cuda", dtype="float32", max_model_len=256)
    assert llm.runner.num_blocks * llm.config.block_size >= 256


def test_cuda_graph_decode_matches_eager(tiny_qwen2):
    sp = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    eager = LLM(str(tiny_qwen2), device="cuda", dtype="float32", max_model_len=256, max_num_seqs=8)
    graph = LLM(str(tiny_qwen2), device="cuda", dtype="float32", max_model_len=256, max_num_seqs=8,
                use_cuda_graph=True)
    assert graph.runner.graphs, "CUDA graph capture failed (see the warning in the log)"
    a = eager.generate(_prompts(), sp)
    b = graph.generate(_prompts(), sp)
    assert [o.token_ids for o in a] == [o.token_ids for o in b]


def test_real_model_bf16_runs_on_gpu(real_model):
    llm = LLM(str(real_model), device="cuda", max_model_len=1024)
    out = llm.chat([{"role": "user", "content": "What is 2 + 3? Reply with just the number."}],
                   SamplingParams(temperature=0, max_tokens=8))[0]
    assert "5" in out.text
