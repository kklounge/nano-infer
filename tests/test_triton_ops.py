"""Fused Triton element-wise kernels vs the plain PyTorch reference ops.

GPU: real kernels in bf16/fp16/fp32. CPU-only machine: Triton interpreter, fp32 (see conftest.py).
"""

import pytest
import torch
import torch.nn.functional as F

from nanoinfer.model.layers import RMSNorm, RotaryEmbedding, apply_rotary
from nanoinfer.model.triton_attention import HAS_TRITON

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPES = [torch.float32] + ([torch.bfloat16, torch.float16] if DEV.type == "cuda" else [])


def tol(dtype):
    # reductions are summed in a different order than torch -> allow one rounding step of the dtype
    return {torch.float32: 1e-5, torch.bfloat16: 1.6e-2, torch.float16: 2e-3}[dtype]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", [64, 896, 1000])
def test_rmsnorm_and_fused_add(dtype, n):
    from nanoinfer.model.triton_ops import fused_add_rmsnorm, rmsnorm
    g = torch.Generator().manual_seed(n)
    x = torch.randn(7, n, generator=g).to(DEV, dtype)
    r = torch.randn(7, n, generator=g).to(DEV, dtype)
    norm = RMSNorm(n, 1e-6).to(DEV, dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(n, generator=g))
        ref = norm(x)
        torch.testing.assert_close(rmsnorm(x, norm.weight, 1e-6), ref, atol=tol(dtype), rtol=tol(dtype))
        out, res = fused_add_rmsnorm(x, r, norm.weight, 1e-6)
        assert torch.equal(res, r + x)                       # the residual add itself is exact
        torch.testing.assert_close(out, norm(r + x), atol=tol(dtype), rtol=tol(dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads,kvh,dim", [(4, 2, 16), (14, 2, 64), (16, 8, 128)])
def test_rope_and_store_kv(dtype, heads, kvh, dim):
    from nanoinfer.model.triton_ops import rope_and_store_kv
    g = torch.Generator().manual_seed(dim)
    t, block_size, num_blocks = 9, 4, 8
    qkv = torch.randn(t, (heads + 2 * kvh) * dim, generator=g).to(DEV, dtype)
    q, k, v = qkv.split([heads * dim, kvh * dim, kvh * dim], dim=-1)          # strided views, like the model
    q, k, v = q.view(t, heads, dim), k.view(t, kvh, dim), v.view(t, kvh, dim)
    positions = torch.tensor([0, 1, 2, 3, 4, 17, 40, 41, 99], device=DEV)
    slots = torch.randperm(num_blocks * block_size, generator=g)[:t].to(DEV)
    rope = RotaryEmbedding(dim, 128, 10000.0)
    cos, sin = rope(positions, dtype)
    ref_q, ref_k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    ref_v = v.clone()
    kc = torch.zeros(num_blocks, block_size, kvh, dim, device=DEV, dtype=dtype)
    vc = torch.zeros_like(kc)
    rope_and_store_kv(q, k, v, *rope.tables(DEV, dtype), positions, slots, kc, vc)
    torch.testing.assert_close(q, ref_q, atol=tol(dtype), rtol=0)       # usually bit-identical
    torch.testing.assert_close(k, ref_k, atol=tol(dtype), rtol=0)
    assert torch.equal(kc.view(-1, kvh, dim)[slots], k)
    assert torch.equal(vc.view(-1, kvh, dim)[slots], ref_v)


@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_and_mul(dtype):
    from nanoinfer.model.triton_ops import silu_and_mul
    x = torch.randn(5, 2 * 1500, generator=torch.Generator().manual_seed(0)).to(DEV, dtype)
    gate, up = x.split(1500, dim=-1)
    torch.testing.assert_close(silu_and_mul(x), F.silu(gate) * up, atol=tol(dtype), rtol=tol(dtype))
