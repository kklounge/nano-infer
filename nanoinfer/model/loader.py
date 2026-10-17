"""Load Hugging Face *.safetensors weights into our own model, by parameter name."""

import glob
import os

import torch
from safetensors import safe_open

from nanoinfer.config import ModelConfig
from nanoinfer.model.qwen import QwenForCausalLM

# checkpoint tensor -> (our merged parameter, which shard). Shards are stacked along dim 0 in this order.
PACKED = {
    "q_proj": ("qkv_proj", 0), "k_proj": ("qkv_proj", 1), "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1),
}


def shard_rows(cfg: ModelConfig, merged: str) -> list[int]:
    """Rows of each shard inside a merged weight."""
    if merged == "qkv_proj":
        q, kv = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
        return [q, kv, kv]
    return [cfg.intermediate_size, cfg.intermediate_size]


def map_name(name: str, cfg: ModelConfig) -> tuple[str, int, int] | None:
    """'model.layers.3.self_attn.k_proj.weight' -> ('model.layers.3.self_attn.qkv_proj.weight', start_row, rows)."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[-2] in PACKED:
        merged, idx = PACKED[parts[-2]]
        rows = shard_rows(cfg, merged)
        return ".".join(parts[:-2] + [merged, parts[-1]]), sum(rows[:idx]), rows[idx]
    return None


def load_model(path: str, device: torch.device, dtype: torch.dtype, max_positions: int | None = None,
               cfg: ModelConfig | None = None) -> QwenForCausalLM:
    cfg = cfg or ModelConfig.from_pretrained(path)
    # 1) build the module skeleton on the "meta" device: shapes only, no memory, no random init
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            model = QwenForCausalLM(cfg, max_positions)
    finally:
        torch.set_default_dtype(prev)
    # 2) allocate real (uninitialised) memory on the target device, then tie embeddings
    model = model.to_empty(device=device)
    model.tie_weights()

    # 3) copy every tensor from the checkpoint into the parameter with the same name
    #    (or into its rows of a merged parameter); track coverage row by row
    params = dict(model.named_parameters(remove_duplicate=False))
    rows_loaded = {name: 0 for name in params}
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {path}")
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                packed = map_name(name, cfg)
                with torch.no_grad():
                    if packed is not None:
                        target, start, n = packed
                        if target not in params:
                            continue
                        params[target][start:start + n].copy_(f.get_tensor(name))
                        rows_loaded[target] += n
                    elif name in params:
                        params[name].copy_(f.get_tensor(name))
                        rows_loaded[name] = params[name].shape[0] if params[name].dim() else 1
                    # anything else (e.g. rotary inv_freq buffers in old checkpoints) is ignored
    if cfg.tie_word_embeddings:
        rows_loaded["lm_head.weight"] = params["lm_head.weight"].shape[0]
    missing = [n for n, p in params.items() if rows_loaded[n] != (p.shape[0] if p.dim() else 1)]
    if missing:
        raise RuntimeError(f"weights missing from checkpoint: {sorted(missing)[:10]}")
    return model.eval()
