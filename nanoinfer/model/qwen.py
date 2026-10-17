"""Our own implementation of the Qwen2 / Qwen3 decoder-only transformer.

Shapes use a *flat* token layout: all tokens of all sequences in the batch are concatenated
into one dimension T (no padding). Attention uses AttentionMetadata to know which token
belongs to which sequence.

Two small deviations from the Hugging Face module layout, both for speed:
  * q/k/v projections are ONE matrix multiply (qkv_proj) and gate/up are ONE (gate_up_proj):
    fewer, bigger GPU kernels. The loader slices the checkpoint's q_proj/k_proj/v_proj (and
    gate_proj/up_proj) tensors into the merged weights (see loader.PACKED).
  * the residual stream is passed along explicitly, so "residual add + RMSNorm" can be one fused
    kernel. The math is identical: h = h + Attn(Norm(h)); h = h + MLP(Norm(h)).
With `use_fused_ops` (CUDA + Triton) the element-wise parts run as fused Triton kernels
(model/triton_ops.py); otherwise everything is plain PyTorch, op for op like the reference.
"""

import torch
import torch.nn.functional as F
from torch import nn

from nanoinfer.config import ModelConfig
from nanoinfer.model.attention import AttentionMetadata, paged_attention
from nanoinfer.model.layers import RMSNorm, RotaryEmbedding, apply_rotary


class QwenAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads      # GQA: several query heads share one k/v head
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=False)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.use_fused_ops = False

    def forward(self, rope, x, kv_cache, meta: AttentionMetadata):
        t = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)
        k_cache, v_cache = kv_cache if kv_cache is not None else (None, None)
        if self.use_fused_ops and k_cache is not None:
            from nanoinfer.model.triton_ops import rmsnorm, rope_and_store_kv
            if self.qk_norm:
                q = rmsnorm(q, self.q_norm.weight, self.q_norm.eps)
                k = rmsnorm(k, self.k_norm.weight, self.k_norm.eps)
            cos_t, sin_t, positions = rope
            # one kernel: RoPE on q and k (in place, strided views of qkv are fine) + write k, v to the cache
            rope_and_store_kv(q, k, v, cos_t, sin_t, positions, meta.slot_mapping, k_cache, v_cache)
            o = paged_attention(q, k, v, k_cache, v_cache, meta, self.scale, store=False)
        else:
            if self.qk_norm:
                q, k = self.q_norm(q), self.k_norm(k)
            cos, sin = rope
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            o = paged_attention(q, k, v, k_cache, v_cache, meta, self.scale)
        return self.o_proj(o.reshape(t, self.q_size))


class QwenMLP(nn.Module):
    """Feed-forward block: down( silu(gate(x)) * up(x) ), with gate and up merged into one matmul."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.use_fused_ops = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        if self.use_fused_ops:
            from nanoinfer.model.triton_ops import silu_and_mul
            return self.down_proj(silu_and_mul(gate_up))
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class QwenDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = QwenAttention(cfg)
        self.mlp = QwenMLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, rope, h, residual, kv_cache, meta):
        """h = output of the previous layer (not yet added to the residual stream)."""
        if residual is None:
            residual, x = h, self.input_layernorm(h)
        else:
            x, residual = self.input_layernorm.add_and_norm(h, residual)
        h = self.self_attn(rope, x, kv_cache, meta)
        x, residual = self.post_attention_layernorm.add_and_norm(h, residual)
        return self.mlp(x), residual


class QwenModel(nn.Module):
    def __init__(self, cfg: ModelConfig, max_positions: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([QwenDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, max_positions, cfg.rope_theta)
        self.use_fused_ops = False

    def forward(self, input_ids, positions, kv_caches, meta):
        h = self.embed_tokens(input_ids)
        if self.use_fused_ops and kv_caches is not None:
            rope = (*self.rotary.tables(positions.device, h.dtype), positions)   # kernel indexes the tables
        else:
            rope = self.rotary(positions, h.dtype)   # computed once, shared by every layer
        meta.prepare(kv_caches is not None)          # cache-read slots, also shared by every layer
        residual = None
        for i, layer in enumerate(self.layers):
            h, residual = layer(rope, h, residual, kv_caches[i] if kv_caches is not None else None, meta)
        return self.norm.add_and_norm(h, residual)[0]


class QwenForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig, max_positions: int | None = None):
        super().__init__()
        self.config = cfg
        max_positions = max_positions or cfg.max_position_embeddings
        self.model = QwenModel(cfg, max_positions)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def tie_weights(self):
        """Small models reuse the input embedding matrix as the output layer."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def set_fused_ops(self, enabled: bool):
        """Switch the element-wise parts to the fused Triton kernels (CUDA only)."""
        for m in self.modules():
            if hasattr(m, "use_fused_ops"):
                m.use_fused_ops = enabled

    def forward(self, input_ids, positions, kv_caches, meta) -> torch.Tensor:
        """Returns final hidden states [T, hidden]."""
        return self.model(input_ids, positions, kv_caches, meta)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [N, hidden] -> logits [N, vocab] in float32."""
        return self.lm_head(hidden).float()
