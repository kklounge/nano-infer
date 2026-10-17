"""Fused element-wise GPU kernels (Triton) for the non-attention parts of a decoder layer.

In eager PyTorch every small op is its own GPU kernel: one RMSNorm is ~7 kernels (cast, square,
mean, rsqrt, multiply, cast, multiply by weight), RoPE on q and k is ~12, writing K/V into the cache
is 2 more. During decode those kernels each process only a few kilobytes, so the GPU spends its
time launching kernels, not computing. These kernels do the same math in ONE pass each:

  fused_add_rmsnorm : residual += x ; out = RMSNorm(residual)          (1 kernel instead of ~8)
  rmsnorm           : out = RMSNorm(x)                                 (also used for Qwen3 q/k norm)
  rope_and_store_kv : rotate q and k in place (RoPE) and write k, v into their KV-cache slots
  silu_and_mul      : out = silu(gate) * up  on the merged [gate | up] projection

The rounding steps mirror the reference (Hugging Face) implementation: e.g. RMSNorm normalises in
float32, rounds to the model dtype, then multiplies by the weight in the model dtype, so results
agree with the PyTorch path to within float rounding of the reductions.
"""

import torch

from nanoinfer.model.triton_attention import HAS_TRITON

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_kernel(X, R, W, O, RO, stride_x, stride_r, stride_o, stride_ro, n_cols, eps,
                        HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
        if HAS_RESIDUAL:
            r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0)
            x = (x.to(tl.float32) + r.to(tl.float32)).to(r.dtype)        # residual add in the model dtype
            tl.store(RO + row * stride_ro + cols, x, mask=mask)
        x32 = x.to(tl.float32)
        var = tl.sum(x32 * x32, axis=0) / n_cols
        normed = (x32 * tl.math.rsqrt(var + eps)).to(x.dtype)            # round like the reference ...
        w = tl.load(W + cols, mask=mask, other=0.0)
        out = (w.to(tl.float32) * normed.to(tl.float32)).to(O.dtype.element_ty)   # ... then weight * x
        tl.store(O + row * stride_o + cols, out, mask=mask)

    @triton.jit
    def _rope_store_kernel(Q, K, V, COS, SIN, POS, SLOT, KC, VC,
                           stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh,
                           stride_cos, stride_cslot, stride_ch,
                           NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
                           HALF: tl.constexpr, HALF_PAD: tl.constexpr, STORE: tl.constexpr):
        t = tl.program_id(0)
        h = tl.program_id(1)                     # 0..NUM_Q_HEADS-1 -> q heads, then k heads
        d = tl.arange(0, HALF_PAD)
        dm = d < HALF
        pos = tl.load(POS + t)
        cos = tl.load(COS + pos * stride_cos + d, mask=dm, other=0.0).to(tl.float32)
        sin = tl.load(SIN + pos * stride_cos + d, mask=dm, other=0.0).to(tl.float32)
        if h < NUM_Q_HEADS:
            base = Q + t * stride_qt + h * stride_qh
        else:
            base = K + t * stride_kt + (h - NUM_Q_HEADS) * stride_kh
        x1 = tl.load(base + d, mask=dm, other=0.0)
        x2 = tl.load(base + HALF + d, mask=dm, other=0.0)
        dt = x1.dtype
        f1, f2 = x1.to(tl.float32), x2.to(tl.float32)
        # reference: x * cos + rotate_half(x) * sin, every multiply / add rounded to the model dtype
        o1 = ((f1 * cos).to(dt).to(tl.float32) + (-f2 * sin).to(dt).to(tl.float32)).to(dt)
        o2 = ((f2 * cos).to(dt).to(tl.float32) + (f1 * sin).to(dt).to(tl.float32)).to(dt)
        tl.store(base + d, o1, mask=dm)
        tl.store(base + HALF + d, o2, mask=dm)
        if STORE:
            if h >= NUM_Q_HEADS:
                kvh = h - NUM_Q_HEADS
                slot = tl.load(SLOT + t)
                dst = slot * stride_cslot + kvh * stride_ch
                tl.store(KC + dst + d, o1, mask=dm)
                tl.store(KC + dst + HALF + d, o2, mask=dm)
                v1 = tl.load(V + t * stride_vt + kvh * stride_vh + d, mask=dm, other=0.0)
                v2 = tl.load(V + t * stride_vt + kvh * stride_vh + HALF + d, mask=dm, other=0.0)
                tl.store(VC + dst + d, v1, mask=dm)
                tl.store(VC + dst + HALF + d, v2, mask=dm)

    @triton.jit
    def _silu_mul_kernel(X, O, stride_x, stride_o, n_cols, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < n_cols
        g = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
        u = tl.load(X + row * stride_x + n_cols + cols, mask=mask, other=0.0)
        g32 = g.to(tl.float32)
        act = (g32 / (1.0 + tl.exp(-g32))).to(g.dtype)                  # silu, rounded like torch
        out = (act.to(tl.float32) * u.to(tl.float32)).to(g.dtype)
        tl.store(O + row * stride_o + cols, out, mask=mask)


def _block(n: int) -> int:
    return max(16, 1 << (n - 1).bit_length())


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x [..., N] -> RMSNorm(x) * weight (rows = all leading dims)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    out = torch.empty_like(x2)
    _rmsnorm_kernel[(x2.shape[0],)](x2, x2, weight, out, out, x2.stride(0), 0, out.stride(0), 0, shape[-1], eps,
                                    HAS_RESIDUAL=False, BLOCK=_block(shape[-1]), num_warps=4)
    return out.view(shape)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (RMSNorm(x + residual) * weight, x + residual) for x, residual [T, N]."""
    out = torch.empty_like(x)
    new_res = torch.empty_like(residual)
    _rmsnorm_kernel[(x.shape[0],)](x, residual, weight, out, new_res, x.stride(0), residual.stride(0),
                                   out.stride(0), new_res.stride(0), x.shape[1], eps,
                                   HAS_RESIDUAL=True, BLOCK=_block(x.shape[1]), num_warps=4)
    return out, new_res


def rope_and_store_kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      positions: torch.Tensor, slot_mapping: torch.Tensor | None,
                      k_cache: torch.Tensor | None, v_cache: torch.Tensor | None):
    """In place: q [T, H, D], k [T, KVH, D] get RoPE; then k, v are written to the cache slots.
    cos/sin are the full tables [max_positions, D] (first half == second half)."""
    t, h, d = q.shape
    kvh = k.shape[1]
    store = k_cache is not None
    kc = k_cache if store else k
    vc = v_cache if store else v
    _rope_store_kernel[(t, h + kvh)](
        q, k, v, cos, sin, positions, slot_mapping if store else positions, kc, vc,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        cos.stride(0), kc.stride(1) if store else 0, kc.stride(2) if store else 0,
        NUM_Q_HEADS=h, NUM_KV_HEADS=kvh, HALF=d // 2, HALF_PAD=_block(d // 2), STORE=store, num_warps=1)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """x [T, 2I] = [gate | up] -> silu(gate) * up  [T, I]."""
    t, two_i = x.shape
    n = two_i // 2
    out = torch.empty(t, n, dtype=x.dtype, device=x.device)
    block = 1024
    _silu_mul_kernel[(t, triton.cdiv(n, block))](x, out, x.stride(0), out.stride(0), n, BLOCK=block, num_warps=4)
    return out
