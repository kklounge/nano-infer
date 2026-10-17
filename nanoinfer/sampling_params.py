"""How to pick the next token (采样参数), and when to stop.

The fields follow the OpenAI / vLLM conventions so the HTTP server can pass them through 1:1.
"""

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """Per-request generation settings.

    temperature == 0        -> greedy decoding (always pick the most likely token).
    top_k <= 0              -> top-k disabled.        top_p == 1.0 -> top-p (nucleus) disabled.
    min_p == 0              -> min-p disabled (drop tokens whose prob < min_p * max prob).
    repetition_penalty 1.0  -> off. HF semantics: every token already in prompt+output gets its logit
                               divided by the penalty (if > 0) or multiplied by it (if < 0).
    presence_penalty 0.0    -> off. OpenAI semantics, output tokens only: logit -= presence * [count > 0]
    frequency_penalty 0.0   -> off. OpenAI semantics, output tokens only: logit -= frequency * count
    stop                    -> stop strings; the returned text is cut BEFORE the stop string.
    stop_token_ids          -> extra token ids that end generation (the token is kept in token_ids).
    min_tokens              -> eos / stop tokens are forbidden until this many tokens were generated.
    seed                    -> per-request random seed: the same request gives the same text no matter
                               which other requests share its batch.
    logprobs                -> None = off; k >= 0 = return the sampled token's log-probability plus the
                               k most likely alternatives at every step (k <= 20).
    n                       -> number of independent samples for the same prompt.
    """

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    max_tokens: int = 64
    min_tokens: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False          # keep generating even after the end-of-sequence token (used by benchmarks)
    seed: int | None = None
    logprobs: int | None = None
    n: int = 1
    skip_special_tokens: bool = True
    # produce incremental text (StepOutput.text) inside the engine; turned on automatically for stop strings
    detokenize: bool = False

    def __post_init__(self):
        if isinstance(self.stop, str):
            self.stop = [self.stop]
        self.stop = [s for s in (self.stop or []) if s]
        self.stop_token_ids = list(self.stop_token_ids or [])
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0 <= self.min_tokens <= self.max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if not -2.0 <= self.presence_penalty <= 2.0 or not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("presence_penalty / frequency_penalty must be in [-2, 2]")
        if self.logprobs is not None and not 0 <= self.logprobs <= 20:
            raise ValueError("logprobs must be in [0, 20]")
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if self.stop:
            self.detokenize = True

    @property
    def is_greedy(self) -> bool:
        return self.temperature < 1e-5

    @property
    def has_penalties(self) -> bool:
        return (self.repetition_penalty != 1.0 or self.presence_penalty != 0.0
                or self.frequency_penalty != 0.0)

    def clone(self, **changes) -> "SamplingParams":
        from dataclasses import replace
        return replace(self, **changes)
