"""Triton paged-attention kernels vs the PyTorch reference path.

On a GPU these run the real kernels; on a CPU-only machine conftest.py turns on TRITON_INTERPRET=1,
which executes the same kernels in Triton's interpreter (float32 only, slow but exact enough).
"""

import pytest
import torch

from nanoinfer import LLM, SamplingParams
from nanoinfer.model.attention import AttentionMetadata, paged_attention
from nanoinfer.model.triton_attention import HAS_TRITON, decode_split_plan, triton_usable

from conftest import hf_greedy

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPES = [torch.float32] + ([torch.bfloat16, torch.float16] if DEV.type == "cuda" else [])


def make_batch(dec_ctx, pre, block_size, heads, kvh, dim, dtype, seed=0):
    """dec_ctx: context lengths of decode sequences; pre: (cached, new) per prefill sequence.
    Returns q, k, v, caches, and metadata for both backends."""
    g = torch.Generator().manual_seed(seed)
    lens = list(dec_ctx) + [c + n for c, n in pre]
    num_blocks = sum(-(-n // block_size) for n in lens) + 3
    kc = torch.randn(num_blocks + 1, block_size, kvh, dim, generator=g).to(DEV, dtype)
    vc = torch.randn(num_blocks + 1, block_size, kvh, dim, generator=g).to(DEV, dtype)
    perm = torch.randperm(num_blocks, generator=g).tolist()       # pages scattered in random order
    tables, i = [], 0
    for n in lens:
        nb = -(-n // block_size)
        tables.append(perm[i:i + nb])
        i += nb
    width = max(len(t) for t in tables)
    bt = torch.tensor([t + [0] * (width - len(t)) for t in tables], device=DEV)
    new_tokens = [1] * len(dec_ctx) + [n for _, n in pre]
    t = sum(new_tokens)
    q = torch.randn(t, heads, dim, generator=g).to(DEV, dtype)
    k = torch.randn(t, kvh, dim, generator=g).to(DEV, dtype)
    v = torch.randn(t, kvh, dim, generator=g).to(DEV, dtype)
    slots = []
    for row, (n_new, n) in enumerate(zip(new_tokens, lens)):
        for p in range(n - n_new, n):
            slots.append(tables[row][p // block_size] * block_size + p % block_size)
    nd = len(dec_ctx)
    cu = [0]
    for _, n in pre:
        cu.append(cu[-1] + n)

    def meta(backend):
        m = AttentionMetadata(slot_mapping=torch.tensor(slots, device=DEV), block_size=block_size,
                              backend=backend, num_decode=nd)
        if nd:
            m.decode_context_lens = torch.tensor(dec_ctx, device=DEV)
            m.decode_block_tables = bt[:nd]
            m.decode_max_len = max(dec_ctx)
            m.decode_all_same_len = len(set(dec_ctx)) == 1
        if pre:
            m.cu_seqlens_q, m.seqlens_k = cu, lens[nd:]
            m.cu_seqlens_q_t = torch.tensor(cu, device=DEV)
            m.seqlens_k_t = torch.tensor(lens[nd:], device=DEV)
            m.prefill_block_tables = bt[nd:]
            m.max_q_len = max(n for _, n in pre)
        m.prepare(True)
        return m
    return q, k, v, kc, vc, meta


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads,kvh,dim,block_size", [(4, 2, 16, 4), (14, 2, 64, 16), (8, 8, 128, 16), (16, 8, 80, 8)])
def test_mixed_batch_matches_reference(dtype, heads, kvh, dim, block_size):
    dec_ctx = [1, 7, 33, 64, 100]
    pre = [(0, 5), (12, 9), (0, 70), (48, 1 + 16), (3, 2)]       # (already cached, new) tokens
    q, k, v, kc, vc, meta = make_batch(dec_ctx, pre, block_size, heads, kvh, dim, dtype)
    scale = dim ** -0.5
    kc2, vc2 = kc.clone(), vc.clone()
    ref = paged_attention(q.float(), k.float(), v.float(), kc2.float(), vc2.float(), meta("sdpa"), scale)
    out = paged_attention(q, k, v, kc, vc, meta("triton"), scale)
    tol = 2e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("ctx", [600, 2000, 5000])
def test_flash_decoding_splits_long_contexts(ctx):
    splits, part = decode_split_plan(ctx)
    assert splits > 1 and splits * part >= ctx
    q, k, v, kc, vc, meta = make_batch([ctx, ctx // 3], [], 16, 4, 2, 32, torch.float32, seed=1)
    ref = paged_attention(q, k, v, kc.clone(), vc.clone(), meta("sdpa"), 0.2)
    out = paged_attention(q, k, v, kc, vc, meta("triton"), 0.2)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


def test_decode_bound_may_exceed_real_context():
    """CUDA graphs pass max_model_len as the bound; empty partitions must not change the result."""
    q, k, v, kc, vc, meta = make_batch([5, 40], [], 4, 4, 2, 16, torch.float32, seed=2)
    ref = paged_attention(q, k, v, kc.clone(), vc.clone(), meta("sdpa"), 0.25)
    m = meta("triton")
    m.decode_max_len = 4096
    out = paged_attention(q, k, v, kc, vc, m, 0.25)
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not triton_usable(DEV), reason="triton cannot run here")
def test_engine_with_triton_backend_matches_hf(tiny_qwen2):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(tiny_qwen2, dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(5)
    system = torch.randint(0, 500, (21,), generator=g).tolist()
    prompts = [system + torch.randint(0, 500, (n,), generator=g).tolist() for n in (2, 13, 30)]
    ref = [hf_greedy(hf, p, 10) for p in prompts]
    llm = LLM(str(tiny_qwen2), dtype="float32", device=DEV.type, max_model_len=128, block_size=4,
              attn_backend="triton", max_num_batched_tokens=20, enforce_eager=True)
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=10, ignore_eos=True))
    assert llm.stats.mixed_steps > 0 and llm.stats.prefix_cache_hit_tokens > 0
    assert [o.token_ids for o in outs] == ref
