"""Load Hugging Face *.safetensors weights into our own model, by parameter name."""

import glob
import os

import torch
from safetensors import safe_open

from nanoinfer.config import ModelConfig
from nanoinfer.model.qwen import QwenForCausalLM


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
    params = dict(model.named_parameters(remove_duplicate=False))
    loaded = set()
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors files in {path}")
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name not in params:
                    continue  # e.g. rotary inv_freq buffers in old checkpoints
                with torch.no_grad():
                    params[name].copy_(f.get_tensor(name))
                loaded.add(name)
    if cfg.tie_word_embeddings:
        loaded.add("lm_head.weight")
    missing = set(params) - loaded
    if missing:
        raise RuntimeError(f"weights missing from checkpoint: {sorted(missing)[:10]}")
    return model.eval()
