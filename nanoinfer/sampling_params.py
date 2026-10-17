"""How to pick the next token (采样参数)."""

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """Per-request generation settings.

    temperature == 0  -> greedy decoding (always pick the most likely token).
    top_k <= 0        -> top-k disabled.
    top_p == 1.0      -> top-p (nucleus) disabled.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 64
    ignore_eos: bool = False          # keep generating even after the end-of-sequence token (used by benchmarks)
    stop_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def is_greedy(self) -> bool:
        return self.temperature < 1e-5
