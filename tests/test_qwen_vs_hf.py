"""End-to-end check on the real Qwen2.5-0.5B-Instruct: our engine's greedy output must equal
Hugging Face transformers token for token.

Runs in float32 by default because bfloat16 matrix multiplications on a multi-threaded CPU are
not bit-for-bit deterministic (even HF disagrees with itself between two identical runs).
Override with NANOINFER_TEST_DTYPE / NANOINFER_TEST_DEVICE (see below).
Memory: the HF model and ours are never loaded at the same time (~2 GB each in float32).
"""

import gc
import os

import pytest
import torch

from nanoinfer import LLM, SamplingParams

from conftest import hf_greedy

DTYPE = os.environ.get("NANOINFER_TEST_DTYPE", "float32")
# HF runs on CPU; ours too by default, so both sides use the same kernels. Set to "cuda" to try the GPU
# (tiny float differences between GPU and CPU kernels may then legitimately flip a near-tie token).
DEVICE = os.environ.get("NANOINFER_TEST_DEVICE", "cpu")
MAX_NEW = 32

CHATS = [
    "Give me a short introduction to large language models.",
    "What is 17 * 23? Answer briefly.",
    "Write a haiku about winter in Ithaca.",
    "用一句话解释什么是KV缓存。",
    "List three famous investment banks.",
]
RAW = [
    "The capital of France is",
    "def fibonacci(n):",
    "In finance, a leveraged buyout (LBO) is",
]


@pytest.fixture(scope="module")
def prompts_and_refs(real_model):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(real_model)
    prompts = [tok.encode(tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                                  add_generation_prompt=True)) for c in CHATS]
    prompts += [tok.encode(r) for r in RAW]
    hf = AutoModelForCausalLM.from_pretrained(real_model, dtype=getattr(torch, DTYPE)).eval()
    eos = hf.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    refs = [hf_greedy(hf, p, MAX_NEW, eos) for p in prompts]
    del hf
    gc.collect()
    return prompts, refs


@pytest.fixture(scope="module")
def llm(real_model, prompts_and_refs):
    engine = LLM(str(real_model), dtype=DTYPE, device=DEVICE, max_model_len=1024, block_size=16)
    yield engine
    del engine
    gc.collect()


def test_greedy_matches_hf_one_by_one(llm, prompts_and_refs):
    prompts, refs = prompts_and_refs
    for p, ref in zip(prompts, refs):
        out = llm.generate([p], SamplingParams(temperature=0, max_tokens=MAX_NEW))[0]
        assert out.token_ids == ref


def test_greedy_matches_hf_batched(llm, prompts_and_refs):
    prompts, refs = prompts_and_refs
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=MAX_NEW))
    assert [o.token_ids for o in outs] == refs


def test_greedy_matches_hf_with_prefix_cache(llm, prompts_and_refs):
    # same prompts again: their full blocks are now served from the prefix cache
    prompts, refs = prompts_and_refs
    before = llm.stats.prefix_cache_hit_tokens
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=MAX_NEW))
    assert llm.stats.prefix_cache_hit_tokens > before
    assert [o.token_ids for o in outs] == refs


def test_naive_attention_backend_matches_hf(real_model, prompts_and_refs, llm):
    prompts, refs = prompts_and_refs
    llm.runner.config.attn_backend = "naive"     # read when building each step's metadata
    try:
        outs = llm.generate(prompts[:3], SamplingParams(temperature=0, max_tokens=MAX_NEW))
    finally:
        llm.runner.config.attn_backend = "sdpa"
    assert [o.token_ids for o in outs] == refs[:3]


def test_chat_text(llm):
    out = llm.chat([{"role": "user", "content": "What is 2 + 3? Reply with just the number."}],
                   SamplingParams(temperature=0, max_tokens=8))[0]
    assert "5" in out.text
    assert out.finish_reason == "stop"
