"""Small helpers: choose device (GPU/CPU) and dtype automatically."""

import torch

_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
}


def pick_device(device: str = "auto") -> torch.device:
    """'auto' -> CUDA if an NVIDIA GPU is usable, otherwise CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def pick_dtype(dtype: "str | torch.dtype", device: torch.device) -> torch.dtype:
    """'auto' -> bfloat16 where it is supported, float16 on old GPUs.

    Qwen weights are stored in bfloat16. Modern CPUs (AVX512-BF16 / AMX, e.g. Ryzen 7000/9000,
    recent Xeons) run bf16 matmuls fast; on older CPUs pass dtype="float32" if bf16 feels slow.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype != "auto":
        return _DTYPES[dtype]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


def dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()
