"""Configuration objects.

ModelConfig  = the shape of the neural network (read from the model's config.json).
EngineConfig = how the inference engine runs it (block size, batch limits, KV cache size...).
"""

import json
import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool       # Qwen2 has bias on q/k/v projections, Qwen3 does not
    qk_norm: bool              # Qwen3 normalises q and k per head before RoPE
    eos_token_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, cfg: dict) -> "ModelConfig":
        model_type = cfg.get("model_type", "qwen2")
        if model_type not in ("qwen2", "qwen3"):
            raise ValueError(f"Only qwen2 / qwen3 architectures are supported, got {model_type!r}")
        if cfg.get("rope_scaling") not in (None, {}) and cfg["rope_scaling"].get("rope_type", "default") != "default":
            raise ValueError("RoPE scaling variants are not implemented")
        heads = cfg["num_attention_heads"]
        eos = cfg.get("eos_token_id")
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        rope_theta = cfg.get("rope_theta")
        if rope_theta is None:  # newer transformers nest it under rope_parameters
            rope_theta = (cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0)
        return cls(
            model_type=model_type,
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=cfg.get("num_key_value_heads", heads),
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // heads,
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=float(rope_theta),
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            attention_bias=cfg.get("attention_bias", model_type == "qwen2"),
            qk_norm=(model_type == "qwen3"),
            eos_token_ids=eos,
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelConfig":
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        mc = cls.from_dict(cfg)
        # generation_config.json often lists extra stop tokens (e.g. <|im_end|> and <|endoftext|>)
        gen_path = os.path.join(path, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path, encoding="utf-8") as f:
                eos = json.load(f).get("eos_token_id")
            if eos is not None:
                for e in ([eos] if isinstance(eos, int) else eos):
                    if e not in mc.eos_token_ids:
                        mc.eos_token_ids.append(e)
        return mc


@dataclass
class EngineConfig:
    model: str                              # local folder with config.json + *.safetensors
    device: str = "auto"                    # "auto" | "cuda" | "cpu"
    dtype: str = "auto"                     # "auto" | "bfloat16" | "float16" | "float32"
    block_size: int = 16                    # tokens per KV-cache block (page)
    max_num_seqs: int = 64                  # max sequences in one batch
    max_num_batched_tokens: int | None = None   # token budget of one forward pass (default: 2048 with
                                            # chunked prefill, else max_model_len)
    max_model_len: int = 4096               # max prompt + output length per sequence
    enable_chunked_prefill: bool = True     # split long prompts across steps and mix them with decodes
    num_kvcache_blocks: int | None = None   # set explicitly, or let the engine size it
    cpu_kvcache_gb: float = 0.5             # KV cache budget on CPU
    gpu_memory_utilization: float = 0.85    # fraction of GPU memory the engine may use
    kv_watermark: float = 0.01              # fraction of blocks kept free when admitting new requests
    enable_prefix_caching: bool = True
    # "auto" -> "triton" on CUDA when Triton is installed, else "sdpa".
    # "triton": our paged-attention kernels (read K/V straight from the pages, no copies)
    # "sdpa"  : gather pages into a contiguous tensor + torch's fused attention (works everywhere)
    # "naive" : plain matmul + softmax, the textbook formula (for learning)
    attn_backend: str = "auto"
    use_fused_ops: bool | None = None       # None -> fused Triton RMSNorm/RoPE/SiLU kernels with the triton backend
    use_cuda_graph: bool | None = None      # None -> on for CUDA (decode steps replay captured graphs)
    enforce_eager: bool = False             # True -> never use CUDA graphs
    max_cuda_graph_batch: int = 128         # largest decode batch that is captured
    seed: int = 0

    def __post_init__(self):
        if self.attn_backend not in ("auto", "triton", "sdpa", "naive"):
            raise ValueError("attn_backend must be 'auto', 'triton', 'sdpa' or 'naive'")
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.max_num_batched_tokens is None:
            self.max_num_batched_tokens = 2048 if self.enable_chunked_prefill else self.max_model_len
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            # without chunked prefill a whole prompt must fit in one forward pass
            self.max_num_batched_tokens = self.max_model_len
