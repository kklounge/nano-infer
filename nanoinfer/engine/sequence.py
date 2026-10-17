"""A Sequence = one request moving through the engine (prompt tokens + generated tokens)."""

import itertools
from enum import Enum, auto

from nanoinfer.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()    # in the waiting queue (new, or preempted and waiting to be recomputed)
    RUNNING = auto()    # has KV-cache blocks and is being decoded
    FINISHED = auto()


class Sequence:
    _counter = itertools.count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams, block_size: int):
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.sampling_params = sampling_params
        self.block_size = block_size
        self.block_table: list[int] = []       # page ids that hold this sequence's KV cache
        self.num_cached_tokens = 0             # prompt tokens whose KV came from the prefix cache (skip compute)
        self.finish_reason: str | None = None  # "stop" | "length" | "abort"
        self.num_preemptions = 0
        self.num_prefix_hit_tokens = 0         # tokens served from the prefix cache at first admission (stats)
        # timestamps (time.perf_counter) for latency metrics
        self.arrival_time: float | None = None
        self.first_token_time: float | None = None
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
    def num_blocks(self) -> int:
        """Blocks needed to hold all current tokens (ceil division)."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    def block(self, i: int) -> list[int]:
        """Token ids stored in the i-th block of this sequence."""
        return self.token_ids[i * self.block_size:(i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def __repr__(self):
        return f"Sequence(id={self.seq_id}, status={self.status.name}, len={self.num_tokens}, blocks={self.block_table})"
