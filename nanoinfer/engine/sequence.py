"""A Sequence = one request moving through the engine (prompt tokens + generated tokens)."""

import itertools
from enum import Enum, auto

from nanoinfer.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()    # in the waiting queue (new, or preempted and waiting to be recomputed)
    RUNNING = auto()    # owns KV-cache blocks; being prefilled (maybe in chunks) or decoded
    FINISHED = auto()


class Sequence:
    """Token bookkeeping. The key counter is `num_computed_tokens`: how many of `token_ids` already
    have their K/V in the cache. Every step the scheduler gives the sequence `num_scheduled_tokens`
    more tokens to compute:
        prefill        : a whole prompt (or a chunk of it, with chunked prefill)
        decode         : exactly 1 token (the one sampled in the previous step)
    When the computed tokens reach the end of token_ids, the model's output for the last token is
    sampled and appended, so the sequence grows by one."""

    _counter = itertools.count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams, block_size: int):
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.sampling_params = sampling_params
        self.block_size = block_size
        self.block_table: list[int] = []       # page ids that hold this sequence's KV cache
        self.num_computed_tokens = 0           # tokens whose K/V are in the cache
        self.num_scheduled_tokens = 0          # tokens to compute in the current step
        self.num_cached_tokens = 0             # tokens served by the prefix cache at the latest admission
        self.finish_reason: str | None = None  # "stop" | "length" | "abort"
        self.stop_reason: "str | int | None" = None   # which stop string / stop token ended it
        self.num_preemptions = 0
        self.num_prefix_hit_tokens = 0         # prefix-cache hits at FIRST admission (stats)
        self.generator = None                  # per-request torch.Generator (seeded requests only)
        self.logprobs: list = []               # TokenLogprob per generated token (if requested)
        # incremental detokenization (only when sampling_params.detokenize)
        self.output_text = ""                  # all text decoded so far (stop strings already cut)
        self.num_emitted_chars = 0             # how much of output_text was already streamed
        self.prefix_offset = 0                 # detokenizer window (see engine/detokenizer.py)
        self.read_offset = 0
        # timestamps (time.perf_counter) for latency metrics
        self.arrival_time: float | None = None
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.finish_time: float | None = None

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def num_uncomputed_tokens(self) -> int:
        return len(self.token_ids) - self.num_computed_tokens

    @property
    def is_prefill(self) -> bool:
        """Still working through prompt (or recomputation) tokens: more than one token left to compute."""
        return self.num_uncomputed_tokens > 1

    @property
    def num_blocks(self) -> int:
        """Blocks needed to hold all current tokens (ceil division)."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i: int) -> list[int]:
        """Token ids stored in the i-th block of this sequence."""
        return self.token_ids[i * self.block_size:(i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def __repr__(self):
        return (f"Sequence(id={self.seq_id}, status={self.status.name}, len={self.num_tokens}, "
                f"computed={self.num_computed_tokens}, blocks={self.block_table})")
