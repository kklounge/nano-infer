"""Paged attention in plain PyTorch.

The KV cache of one layer is two big tensors:
    k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
Think of them as a notebook with `num_blocks` pages; each page holds `block_size` tokens.
A sequence owns a list of pages (its *block table*), not one contiguous region.

A token at position p of a sequence lives in "slot"
    slot = block_table[p // block_size] * block_size + p % block_size
of the flattened view k_cache.view(-1, num_kv_heads, head_dim).

Per layer:
  1. store_kv : write the new tokens' k/v into their slots.
  2. attention: for each query, gather that sequence's keys/values from its pages and
     compute softmax(q·kᵀ / sqrt(d)) · v  (causal: a token never looks at the future).

Note on prefix caching inside one batch: store_kv writes the k/v of ALL tokens in the batch
before any attention is computed, so a sequence may read blocks that another sequence of the
same batch is filling in this very step.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    """Everything attention needs to know about the current (packed) batch.

    All sequences of a step are packed into one flat list of tokens (no padding):
        tokens = [seq0 new tokens..., seq1 new tokens..., ...]
    Prefill: cu_seqlens_q[i]:cu_seqlens_q[i+1] are the new (query) tokens of sequence i and
             cu_seqlens_k gives its total length (cached prefix + new tokens).
    Decode:  exactly one new token per sequence.
    """

    is_prefill: bool
    slot_mapping: torch.Tensor                # [T] where each new token's k/v is written
    block_size: int
    context_lens: torch.Tensor | None = None  # [B] decode: tokens per sequence including the new one
    block_tables: torch.Tensor | None = None  # [B, max_blocks] page ids, padded with 0
    cu_seqlens_q: list[int] | None = None     # prefill only (python ints: we loop over sequences)
    cu_seqlens_k: list[int] | None = None     # prefill only
    max_context_len: int = 0                  # decode: longest context in the batch (gather width)
    all_same_len: bool = False                # decode: every context is max_context_len long -> no mask
    backend: str = "sdpa"                     # "sdpa" | "naive"
    # filled by prepare() once per forward pass and shared by all layers
    gather_slots: "torch.Tensor | list | None" = None
    decode_mask: torch.Tensor | None = None

    def prepare(self):
        """Compute the slot ids to read back from the cache ONCE per forward (not once per layer)."""
        if self.is_prefill:
            if self.block_tables is not None:
                self.gather_slots = [
                    slots_for(self.block_tables[i], self.cu_seqlens_k[i + 1] - self.cu_seqlens_k[i], self.block_size)
                    for i in range(len(self.cu_seqlens_q) - 1)
                ]
        else:
            pos = torch.arange(self.max_context_len, device=self.block_tables.device)
            slots = self.block_tables[:, pos // self.block_size] * self.block_size + pos % self.block_size
            self.gather_slots = slots                                          # [B, max_len]
            if not self.all_same_len:
                self.decode_mask = (pos[None, :] < self.context_lens[:, None])[:, None, None, :]   # [B,1,1,max_len]


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
    if backend == "sdpa":
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


def paged_attention(q, k, v, k_cache, v_cache, meta: AttentionMetadata, scale: float) -> torch.Tensor:
    """q: [T, H, D]; k, v: [T, KVH, D] for the new tokens. Returns [T, H, D]."""
    if k_cache is not None:
        store_kv(k, v, k_cache, v_cache, meta.slot_mapping)
    if meta.is_prefill:
        return _prefill_attention(q, k, v, k_cache, v_cache, meta, scale)
    return _decode_attention(q, k_cache, v_cache, meta, scale)


def _prefill_attention(q, k, v, k_cache, v_cache, meta, scale):
    """Prefill: each sequence feeds many tokens. Loop over sequences (prompts have different lengths)."""
    out = torch.empty_like(q)
    for i in range(len(meta.cu_seqlens_q) - 1):
        qs, qe = meta.cu_seqlens_q[i], meta.cu_seqlens_q[i + 1]
        lq = qe - qs
        lk = meta.cu_seqlens_k[i + 1] - meta.cu_seqlens_k[i]
        if lk == lq:
            # no cached prefix: keys/values are exactly the new tokens
            k_i, v_i = k[qs:qe], v[qs:qe]
        else:
            # prefix-cache hit: the first (lk - lq) tokens are already in the cache -> read them from the pages
            k_i = gather_kv(k_cache, meta.gather_slots[i])
            v_i = gather_kv(v_cache, meta.gather_slots[i])
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
    """Decode: every sequence feeds exactly ONE new token. Vectorised over the batch:
    gather each sequence's keys from its pages into a padded [B, max_len] tensor and mask the padding."""
    bsz, num_heads, d = q.shape
    kvh = k_cache.shape[-2]
    k = gather_kv(k_cache, meta.gather_slots).transpose(1, 2)    # [B, KVH, max_len, D]
    v = gather_kv(v_cache, meta.gather_slots).transpose(1, 2)
    if meta.decode_mask is None:
        # no padding at all (e.g. batch of 1): plain attention, GQA handled by the kernel
        return _attend(q.unsqueeze(2), k, v, scale, None, False, meta.backend).squeeze(2)
    # Grouped-query trick: the n_rep query heads that share one k/v head are laid out as n_rep
    # "query rows" of that k/v head. Same math, but k/v never have to be copied n_rep times.
    n_rep = num_heads // kvh
    q = q.view(bsz, kvh, n_rep, d)                                    # [B, KVH, n_rep, D]
    o = _attend(q, k, v, scale, meta.decode_mask, False, meta.backend)   # [B, KVH, n_rep, D]
    return o.reshape(bsz, num_heads, d)
