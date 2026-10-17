"""Our own implementation of the Qwen2 / Qwen3 decoder-only transformer.

Shapes use a *flat* token layout: all tokens of all sequences in the batch are concatenated
into one dimension T (no padding). Attention uses AttentionMetadata to know which token
belongs to which sequence.

Parameter names match the Hugging Face checkpoint exactly (model.layers.0.self_attn.q_proj.weight, ...)
so weights can be loaded by name.
"""

import torch
from torch import nn

from nanoinfer.config import ModelConfig
from nanoinfer.model.attention import AttentionMetadata, paged_attention
from nanoinfer.model.layers import RMSNorm, RotaryEmbedding, SwiGLUMLP, apply_rotary


class QwenAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads      # GQA: several query heads share one k/v head
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        bias = cfg.attention_bias
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, rope, x, kv_cache, meta: AttentionMetadata):
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        k_cache, v_cache = kv_cache if kv_cache is not None else (None, None)
        o = paged_attention(q, k, v, k_cache, v_cache, meta, self.scale)
        return self.o_proj(o.reshape(t, self.num_heads * self.head_dim))


class QwenDecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.self_attn = QwenAttention(cfg)
        self.mlp = SwiGLUMLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, rope, h, kv_cache, meta):
        # pre-norm residual blocks: h = h + Attn(Norm(h)); h = h + MLP(Norm(h))
        h = h + self.self_attn(rope, self.input_layernorm(h), kv_cache, meta)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class QwenModel(nn.Module):
    def __init__(self, cfg: ModelConfig, max_positions: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([QwenDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = RotaryEmbedding(cfg.head_dim, max_positions, cfg.rope_theta)

    def forward(self, input_ids, positions, kv_caches, meta):
        h = self.embed_tokens(input_ids)
        rope = self.rotary(positions, h.dtype)       # computed once, shared by every layer
        meta.prepare()                               # cache-read slots, also shared by every layer
        for i, layer in enumerate(self.layers):
            h = layer(rope, h, kv_caches[i] if kv_caches is not None else None, meta)
        return self.norm(h)


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

    def forward(self, input_ids, positions, kv_caches, meta) -> torch.Tensor:
        """Returns final hidden states [T, hidden]."""
        return self.model(input_ids, positions, kv_caches, meta)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [N, hidden] -> logits [N, vocab] in float32."""
        return self.lm_head(hidden).float()
