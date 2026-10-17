"""Paged attention.

The KV cache of one layer is two big tensors:
    k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
Think of them as a notebook with `num_blocks` pages; each page holds `block_size` tokens.
A sequence owns a list of pages (its *block table*), not one contiguous region.

A token at position p of a sequence lives in "slot"
    slot = block_table[p // block_size] * block_size + p % block_size
of the flattened view k_cache.view(-1, num_kv_heads, head_dim).

One step's batch is packed into a flat list of T tokens, decode tokens first:
    [ d_0, d_1, ..., d_{D-1} | prompt chunk of seq P0 | prompt chunk of seq P1 | ... ]
      one token per decoding    many tokens per prefilling sequence; they also see earlier
      sequence                  tokens of their sequence that are already in the cache
Per layer:
  1. store_kv : write the new tokens' k/v into their slots (ALL tokens, before any attention).
  2. decode attention  for the first D tokens (one query each, vectorised over the batch)
  3. prefill attention for the rest (causal on absolute positions)

Backends:
  "triton": kernels that read K/V straight from the pages (model/triton_attention.py) - GPU
  "sdpa"  : gather the pages into a contiguous tensor, then torch's fused attention - everywhere
  "naive" : gather + the textbook formula softmax(q·kᵀ/sqrt(d))·v - for learning
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    """Everything attention needs to know about the current (packed) batch."""

    slot_mapping: torch.Tensor                     # [T] where each new token's k/v is written
    block_size: int
    backend: str = "sdpa"
    # ---- decode part: the first num_decode tokens, one per sequence
    num_decode: int = 0
    decode_context_lens: torch.Tensor | None = None    # [D] tokens per sequence including the new one
    decode_block_tables: torch.Tensor | None = None    # [D, W] page ids, padded with 0
    decode_max_len: int = 0                            # python int: longest context (gather width / bound)
    decode_all_same_len: bool = False                  # every context == decode_max_len -> no mask
    # ---- prefill part: the remaining tokens
    cu_seqlens_q: list[int] | None = None              # [P+1] offsets of each sequence's new tokens
    seqlens_k: list[int] | None = None                 # [P] cached + new tokens per sequence
    cu_seqlens_q_t: torch.Tensor | None = None         # same on the device (triton)
    seqlens_k_t: torch.Tensor | None = None
    prefill_block_tables: torch.Tensor | None = None   # [P, W]
    max_q_len: int = 0
    # filled by prepare() once per forward pass and shared by all layers
    decode_slots: torch.Tensor | None = None
    decode_mask: torch.Tensor | None = None
    prefill_slots: list | None = None

    @property
    def num_prefill_seqs(self) -> int:
        return 0 if self.cu_seqlens_q is None else len(self.cu_seqlens_q) - 1

    @property
    def prefill_has_context(self) -> bool:
        """Does any prefill sequence attend to tokens computed in an earlier step?"""
        if not self.num_prefill_seqs:
            return False
        return any(k != self.cu_seqlens_q[i + 1] - self.cu_seqlens_q[i] for i, k in enumerate(self.seqlens_k))

    def prepare(self, has_cache: bool):
        """Compute the slot ids to read back from the cache ONCE per forward (not once per layer)."""
        if self.backend == "triton" and has_cache:
            return                                  # the kernels walk the block tables themselves
        if self.num_decode:
            pos = torch.arange(self.decode_max_len, device=self.decode_block_tables.device)
            bt = self.decode_block_tables
            self.decode_slots = bt[:, pos // self.block_size] * self.block_size + pos % self.block_size
            if not self.decode_all_same_len:
                self.decode_mask = (pos[None, :] < self.decode_context_lens[:, None])[:, None, None, :]
        if has_cache and self.prefill_has_context:
            self.prefill_slots = [
                slots_for(self.prefill_block_tables[i], self.seqlens_k[i], self.block_size)
                if self.seqlens_k[i] != self.cu_seqlens_q[i + 1] - self.cu_seqlens_q[i] else None
                for i in range(self.num_prefill_seqs)
            ]


def slots_for(block_table: torch.Tensor, length: int, block_size: int) -> torch.Tensor:
    """Slot ids of positions 0..length-1 for one sequence."""
    pos = torch.arange(length, device=block_table.device)
    return block_table[pos // block_size] * block_size + pos % block_size


def store_kv(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """Write k/v of the new tokens ([T, KVH, D]) into their cache slots."""
    kvh, d = k.shape[1], k.shape[2]
    k_cache.view(-1, kvh, d).index_copy_(0, slot_mapping, k)
    v_cache.view(-1, kvh, d).index_copy_(0, slot_mapping, v)


def gather_kv(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Read tokens back from their pages. slots: any shape S -> result S + [KVH, D]."""
    kvh, d = cache.shape[-2], cache.shape[-1]
    return cache.view(-1, kvh * d).index_select(0, slots.reshape(-1)).view(*slots.shape, kvh, d)


def _attend(q, k, v, scale, mask, is_causal, backend):
    """q: [B, H, Lq, D]; k, v: [B, KVH, Lk, D]; mask: bool broadcastable to [B, H, Lq, Lk] (True = may attend)."""
    n_rep = q.shape[1] // k.shape[1]
    if backend != "naive":
        if mask is None:
            # torch's fused kernel; grouped-query attention handled inside
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale, enable_gqa=n_rep > 1)
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    # "naive": the textbook formula, easiest to read
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)).float() * scale       # [B, H, Lq, Lk]
    if is_causal:
        lq, lk = q.shape[2], k.shape[2]
        mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(lk - lq)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def paged_attention(q, k, v, k_cache, v_cache, meta: AttentionMetadata, scale: float,
                    store: bool = True) -> torch.Tensor:
    """q: [T, H, D]; k, v: [T, KVH, D] for the new tokens. Returns [T, H, D].
    store=False: the caller already wrote k/v into the cache (fused RoPE + store kernel)."""
    if k_cache is not None and store:
        store_kv(k, v, k_cache, v_cache, meta.slot_mapping)
    nd = meta.num_decode
    if not meta.num_prefill_seqs:
        return _decode_attention(q, k_cache, v_cache, meta, scale)
    if nd == 0:
        return _prefill_attention(q, k, v, k_cache, v_cache, meta, scale)
    out = torch.empty_like(q)
    out[:nd] = _decode_attention(q[:nd], k_cache, v_cache, meta, scale)
    out[nd:] = _prefill_attention(q[nd:], k[nd:], v[nd:], k_cache, v_cache, meta, scale)
    return out


def _prefill_attention(q, k, v, k_cache, v_cache, meta, scale):
    """Prefill: each sequence feeds many tokens."""
    if meta.backend == "triton" and k_cache is not None:
        from nanoinfer.model.triton_attention import paged_prefill_attention
        return paged_prefill_attention(q, k_cache, v_cache, meta.prefill_block_tables, meta.cu_seqlens_q_t,
                                       meta.seqlens_k_t, meta.max_q_len, scale)
    # portable path: loop over sequences (prompts have different lengths)
    out = torch.empty_like(q)
    for i in range(meta.num_prefill_seqs):
        qs, qe = meta.cu_seqlens_q[i], meta.cu_seqlens_q[i + 1]
        lq = qe - qs
        lk = meta.seqlens_k[i]
        if lk == lq:
            # nothing computed earlier: keys/values are exactly the new tokens
            k_i, v_i = k[qs:qe], v[qs:qe]
        else:
            # earlier tokens (prefix-cache hit / earlier chunk) are in the cache -> read them from the pages
            k_i = gather_kv(k_cache, meta.prefill_slots[i])
            v_i = gather_kv(v_cache, meta.prefill_slots[i])
        q_i = q[qs:qe].transpose(0, 1).unsqueeze(0)       # [1, H, Lq, D]
        k_i = k_i.transpose(0, 1).unsqueeze(0)            # [1, KVH, Lk, D]
        v_i = v_i.transpose(0, 1).unsqueeze(0)
        if lk == lq:
            o = _attend(q_i, k_i, v_i, scale, None, True, meta.backend)
        else:
            # query j sits at absolute position (lk - lq + j) and may see keys 0..(lk - lq + j)
            mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(lk - lq)
            o = _attend(q_i, k_i, v_i, scale, mask, False, meta.backend)
        out[qs:qe] = o.squeeze(0).transpose(0, 1)
    return out


def _decode_attention(q, k_cache, v_cache, meta, scale):
    """Decode: every sequence feeds exactly ONE new token."""
    if meta.backend == "triton":
        from nanoinfer.model.triton_attention import paged_decode_attention
        return paged_decode_attention(q, k_cache, v_cache, meta.decode_block_tables, meta.decode_context_lens,
                                      scale, meta.decode_max_len)
    # portable path, vectorised over the batch: gather each sequence's keys from its pages into a
    # padded [B, max_len] tensor and mask the padding.
    bsz, num_heads, d = q.shape
    kvh = k_cache.shape[-2]
    k = gather_kv(k_cache, meta.decode_slots).transpose(1, 2)    # [B, KVH, max_len, D]
    v = gather_kv(v_cache, meta.decode_slots).transpose(1, 2)
    if meta.decode_mask is None:
        # no padding at all (e.g. batch of 1): plain attention, GQA handled by the kernel
        return _attend(q.unsqueeze(2), k, v, scale, None, False, meta.backend).squeeze(2)
    # Grouped-query trick: the n_rep query heads that share one k/v head are laid out as n_rep
    # "query rows" of that k/v head. Same math, but k/v never have to be copied n_rep times.
    n_rep = num_heads // kvh
    q = q.view(bsz, kvh, n_rep, d)                                    # [B, KVH, n_rep, D]
    o = _attend(q, k, v, scale, meta.decode_mask, False, meta.backend)   # [B, KVH, n_rep, D]
    return o.reshape(bsz, num_heads, d)
