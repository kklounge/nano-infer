"""Paged-attention GPU kernels written in Triton (https://triton-lang.org).

Why: the portable path in attention.py first COPIES each sequence's K/V out of its pages into a
contiguous tensor (index_select), then calls PyTorch attention. These kernels instead read K/V
directly from the pages through the block table, one tile at a time, with an online softmax
(FlashAttention style), so nothing is copied and nothing of size [len x len] is materialised.

Two kernels:
  * paged_decode_attention : one query token per sequence. "Flash-decoding": the context is split
    into partitions processed by different GPU programs in parallel (important when the batch is
    small: batch 1 x 14 heads would only use 14 of the GPU's ~48 SMs), then a tiny second kernel
    merges the partial softmax results. Runtime loop bounds come from context_lens, so the same
    launch works for any context length -> safe to capture in a CUDA graph.
  * paged_prefill_attention: many query tokens per sequence (a prompt or a prompt chunk) that also
    attend to earlier tokens already in the cache (prefix-cache hits, earlier chunks). Causal mask
    on absolute positions. Variable lengths in one launch (no Python loop over sequences).

Tested against the PyTorch reference on CPU with TRITON_INTERPRET=1 and on a real GPU.
"""

import os

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:          # not installed (e.g. CPU-only box) -> attention.py uses the PyTorch path
    HAS_TRITON = False


def triton_usable(device: torch.device) -> bool:
    if not HAS_TRITON:
        return False
    if os.environ.get("TRITON_INTERPRET") == "1":
        return True                       # interpreter runs kernels on CPU (slow, for tests)
    return device.type == "cuda"


if HAS_TRITON:

    @triton.jit
    def _decode_partial_kernel(
        Q, K, V, BT, CTX, O, MP, LP, OP,
        scale,
        stride_qt, stride_qh, stride_kslot, stride_kh, stride_bt, stride_ot, stride_oh,
        N_REP: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, D_PAD: tl.constexpr,
        BLOCK_N: tl.constexpr, PARTITION: tl.constexpr, NUM_SPLITS: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        split = tl.program_id(2)
        kvh = h // N_REP
        ctx = tl.load(CTX + b)
        start = split * PARTITION
        end = tl.minimum(start + PARTITION, ctx)
        d = tl.arange(0, D_PAD)
        dmask = d < HEAD_DIM
        q = tl.load(Q + b * stride_qt + h * stride_qh + d, mask=dmask, other=0.0).to(tl.float32) * scale
        m_i = tl.max(tl.full([BLOCK_N], -float("inf"), dtype=tl.float32), axis=0)   # fp32 scalar
        l_i = tl.sum(tl.zeros([BLOCK_N], dtype=tl.float32), axis=0)
        acc = tl.zeros([D_PAD], dtype=tl.float32)
        for t0 in range(start, end, BLOCK_N):
            offs = t0 + tl.arange(0, BLOCK_N)
            tmask = offs < end
            blk = tl.load(BT + b * stride_bt + offs // BLOCK_SIZE, mask=tmask, other=0)
            slot = blk.to(tl.int64) * BLOCK_SIZE + offs % BLOCK_SIZE
            kv_off = slot[:, None] * stride_kslot + kvh * stride_kh + d[None, :]
            kvmask = tmask[:, None] & dmask[None, :]
            k = tl.load(K + kv_off, mask=kvmask, other=0.0).to(tl.float32)
            s = tl.sum(k * q[None, :], axis=1)
            s = tl.where(tmask, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            p = tl.exp(s - m_new)
            alpha = tl.exp(m_i - m_new)
            v = tl.load(V + kv_off, mask=kvmask, other=0.0).to(tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new
        if NUM_SPLITS == 1:
            out = acc / l_i
            tl.store(O + b * stride_ot + h * stride_oh + d, out.to(O.dtype.element_ty), mask=dmask)
        else:
            idx = (b * tl.num_programs(1) + h) * NUM_SPLITS + split
            tl.store(MP + idx, m_i)
            tl.store(LP + idx, l_i)
            tl.store(OP + idx * D_PAD + d, acc)

    @triton.jit
    def _decode_reduce_kernel(MP, LP, OP, O, stride_ot, stride_oh,
                              HEAD_DIM: tl.constexpr, D_PAD: tl.constexpr, NUM_SPLITS: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        base = (b * tl.num_programs(1) + h) * NUM_SPLITS
        sp = tl.arange(0, NUM_SPLITS)
        d = tl.arange(0, D_PAD)
        m = tl.load(MP + base + sp)
        l = tl.load(LP + base + sp)
        acc = tl.load(OP + (base + sp)[:, None] * D_PAD + d[None, :])
        m_max = tl.max(m, axis=0)
        w = tl.exp(m - m_max)                       # empty partitions have m = -inf -> weight 0
        out = tl.sum(acc * w[:, None], axis=0) / tl.sum(l * w, axis=0)
        tl.store(O + b * stride_ot + h * stride_oh + d, out.to(O.dtype.element_ty), mask=d < HEAD_DIM)

    @triton.jit
    def _prefill_kernel(
        Q, K, V, BT, CU_Q, CTX, O,
        scale,
        stride_qt, stride_qh, stride_kslot, stride_kh, stride_bt, stride_ot, stride_oh,
        N_REP: tl.constexpr, BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, D_PAD: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, PRECISION: tl.constexpr,
    ):
        i = tl.program_id(0)            # sequence
        h = tl.program_id(1)            # query head
        mb = tl.program_id(2)           # tile of BLOCK_M query rows
        kvh = h // N_REP
        q_start = tl.load(CU_Q + i)
        q_len = tl.load(CU_Q + i + 1) - q_start
        ctx = tl.load(CTX + i)                       # total tokens of this sequence incl. the new ones
        ctx_start = ctx - q_len                      # absolute position of the first query row
        rows = mb * BLOCK_M + tl.arange(0, BLOCK_M)
        rmask = rows < q_len
        d = tl.arange(0, D_PAD)
        dmask = d < HEAD_DIM
        q = tl.load(Q + (q_start + rows)[:, None] * stride_qt + h * stride_qh + d[None, :],
                    mask=rmask[:, None] & dmask[None, :], other=0.0)
        q_pos = ctx_start + rows
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D_PAD], dtype=tl.float32)
        # causal: the last row of this tile sees keys up to its own position
        k_end = tl.minimum(ctx, ctx_start + (mb + 1) * BLOCK_M)
        k_end = tl.where(mb * BLOCK_M < q_len, k_end, 0)          # tiles past the end do nothing
        for n0 in range(0, k_end, BLOCK_N):
            offs = n0 + tl.arange(0, BLOCK_N)
            nmask = offs < ctx
            blk = tl.load(BT + i * stride_bt + offs // BLOCK_SIZE, mask=nmask, other=0)
            slot = blk.to(tl.int64) * BLOCK_SIZE + offs % BLOCK_SIZE
            kv_off = slot[:, None] * stride_kslot + kvh * stride_kh + d[None, :]
            kvmask = nmask[:, None] & dmask[None, :]
            k = tl.load(K + kv_off, mask=kvmask, other=0.0)
            s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
            allowed = (offs[None, :] <= q_pos[:, None]) & nmask[None, :]
            s = tl.where(allowed, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            v = tl.load(V + kv_off, mask=kvmask, other=0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision=PRECISION)
            m_i = m_new
        out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]          # (rows past the end: l = 0)
        tl.store(O + (q_start + rows)[:, None] * stride_ot + h * stride_oh + d[None, :],
                 out.to(O.dtype.element_ty), mask=rmask[:, None] & dmask[None, :])


def _pad_dim(d: int) -> int:
    return max(16, 1 << (d - 1).bit_length())


def decode_split_plan(max_context: int) -> tuple[int, int]:
    """(number of partitions, tokens per partition) for flash-decoding. Only a handful of distinct
    plans exist, so Triton compiles only a handful of kernel variants."""
    if max_context <= 512:
        return 1, 1 << 30                              # one program walks the whole context
    splits = min(16, 1 << (-(-max_context // 512) - 1).bit_length())
    partition = max(512, -(-max_context // splits))
    return splits, -(-partition // 512) * 512


def paged_decode_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                           block_tables: torch.Tensor, context_lens: torch.Tensor, scale: float,
                           max_context: int) -> torch.Tensor:
    """q [B, H, D]; caches [num_blocks, block_size, KVH, D]; block_tables [B, W]; context_lens [B].
    max_context only chooses the number of partitions (an upper bound is fine, e.g. max_model_len)."""
    bsz, heads, dim = q.shape
    _, block_size, kvh, _ = k_cache.shape
    out = torch.empty_like(q)
    d_pad = _pad_dim(dim)
    splits, partition = decode_split_plan(max_context)
    if splits > 1:
        mp = torch.empty(bsz * heads * splits, dtype=torch.float32, device=q.device)
        lp = torch.empty_like(mp)
        op = torch.empty(bsz * heads * splits, d_pad, dtype=torch.float32, device=q.device)
    else:
        mp = lp = op = out                              # unused
    _decode_partial_kernel[(bsz, heads, splits)](
        q, k_cache, v_cache, block_tables, context_lens, out, mp, lp, op, scale,
        q.stride(0), q.stride(1), k_cache.stride(1), k_cache.stride(2), block_tables.stride(0),
        out.stride(0), out.stride(1),
        N_REP=heads // kvh, BLOCK_SIZE=block_size, HEAD_DIM=dim, D_PAD=d_pad, BLOCK_N=64,
        PARTITION=partition, NUM_SPLITS=splits, num_warps=4)
    if splits > 1:
        _decode_reduce_kernel[(bsz, heads)](mp, lp, op, out, out.stride(0), out.stride(1),
                                            HEAD_DIM=dim, D_PAD=d_pad, NUM_SPLITS=splits, num_warps=1)
    return out


def paged_prefill_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                            block_tables: torch.Tensor, cu_seqlens_q: torch.Tensor, context_lens: torch.Tensor,
                            max_q_len: int, scale: float) -> torch.Tensor:
    """q [T, H, D] new tokens of P sequences packed back to back (cu_seqlens_q [P+1]);
    context_lens [P] = cached + new tokens per sequence; block_tables [P, W]. Returns [T, H, D]."""
    t, heads, dim = q.shape
    _, block_size, kvh, _ = k_cache.shape
    out = torch.empty_like(q)
    d_pad = _pad_dim(dim)
    block_m = 64
    block_n = 64 if d_pad <= 64 else 32
    precision = "ieee" if q.dtype == torch.float32 else "tf32"
    grid = (cu_seqlens_q.shape[0] - 1, heads, triton.cdiv(max_q_len, block_m))
    _prefill_kernel[grid](
        q, k_cache, v_cache, block_tables, cu_seqlens_q, context_lens, out, scale,
        q.stride(0), q.stride(1), k_cache.stride(1), k_cache.stride(2), block_tables.stride(0),
        out.stride(0), out.stride(1),
        N_REP=heads // kvh, BLOCK_SIZE=block_size, HEAD_DIM=dim, D_PAD=d_pad,
        BLOCK_M=block_m, BLOCK_N=block_n, PRECISION=precision, num_warps=4)
    return out
