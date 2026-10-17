"""Building blocks of a Qwen-style transformer: RMSNorm, RoPE, SwiGLU MLP.

Every op is written to mirror the reference Hugging Face implementation step by step
(same dtype casts, same order), so our greedy outputs can match it token for token.
"""

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    """Root-Mean-Square normalisation: x / sqrt(mean(x^2) + eps) * weight.

    Computed in float32 for stability, then cast back to the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        variance = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return self.weight * x32.to(in_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE).

    Instead of adding a position vector, RoPE *rotates* pairs of numbers in q and k by an
    angle that depends on the token position. The dot product q·k then only depends on the
    distance between the two tokens. cos/sin for every position are precomputed once, and
    looked up once per forward pass (all layers share them).
    """

    def __init__(self, head_dim: int, max_positions: int, base: float):
        super().__init__()
        self.head_dim = head_dim
        self.max_positions = max_positions
        self.base = base
        self._tables: dict = {}   # (device, dtype) -> (cos, sin), each [max_positions, head_dim]

    def _build(self, device: torch.device, dtype: torch.dtype):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=device).float() / self.head_dim))
        pos = torch.arange(self.max_positions, device=device, dtype=torch.float32)
        freqs = pos[:, None] * inv_freq[None, :]          # [P, D/2] angle = position * frequency
        emb = torch.cat((freqs, freqs), dim=-1)           # [P, D]
        # computed in float32, then rounded to the model dtype (same as the reference implementation)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        """positions [T] -> (cos, sin), each [T, 1, D] (the 1 broadcasts over heads)."""
        key = (positions.device, dtype)
        if key not in self._tables:
            self._tables[key] = self._build(positions.device, dtype)
        cos, sin = self._tables[key]
        return cos[positions].unsqueeze(1), sin[positions].unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, heads, D]. Rotates every (x_i, x_{i+D/2}) pair by the position's angle."""
    return (x * cos) + (rotate_half(x) * sin)


class SwiGLUMLP(nn.Module):
    """Feed-forward block: down( silu(gate(x)) * up(x) )."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
