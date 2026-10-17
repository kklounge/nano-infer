"""Scheduler: decides WHICH sequences run in the next forward pass (continuous batching).

Two queues:
  waiting : requests that have no KV cache yet (new, or preempted)
  running : requests that are generating tokens

Each call to schedule() returns one batch, either
  * a PREFILL batch  - new requests whose whole prompt is processed at once, or
  * a DECODE batch   - every running request generates exactly one token.
Prefill has priority (it gets new requests going quickly).

Continuous batching = after EVERY step, finished requests leave and waiting requests join.
Nobody waits for the slowest request in a fixed batch.

Preemption: if the KV cache is full and a running request needs a new block, we kick out the
most recently added running request: free its blocks and put it back at the FRONT of the
waiting queue. Later it is recomputed from its tokens (prompt + what it already generated).
"""

from collections import deque

from nanoinfer.config import EngineConfig
from nanoinfer.engine.block_manager import BlockManager
from nanoinfer.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: EngineConfig, num_blocks: int, eos_token_ids: list[int]):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.eos_token_ids = set(eos_token_ids)
        self.block_manager = BlockManager(num_blocks, config.block_size, config.enable_prefix_caching)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # ---- 1) prefill: admit waiting requests while budget and KV blocks allow
        scheduled: list[Sequence] = []
        num_batched_tokens = 0
        while self.waiting and len(self.running) + len(scheduled) < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            scheduled.append(seq)
        if scheduled:
            self.running.extend(scheduled)
            return scheduled, True

        # ---- 2) decode: one token for every running request
        while self.running and len(scheduled) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self._preempt(self.running.pop())   # victim: the newest running request
                else:
                    self._preempt(seq)                  # nobody else to evict: wait ourselves
                    break
            else:
                self.block_manager.may_append(seq)
                scheduled.append(seq)
        if not scheduled:
            raise RuntimeError("KV cache too small to run even one sequence; increase num_kvcache_blocks")
        self.running.extendleft(reversed(scheduled))
        return scheduled, False

    def _preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[Sequence]:
        """Append sampled tokens; finish sequences that hit a stop condition. Returns finished ones."""
        finished = []
        for seq, tok in zip(seqs, token_ids):
            seq.append_token(tok)
            sp = seq.sampling_params
            if not sp.ignore_eos and (tok in self.eos_token_ids or tok in sp.stop_token_ids):
                seq.finish_reason = "stop"
            elif seq.num_completion_tokens >= sp.max_tokens or seq.num_tokens >= self.max_model_len:
                seq.finish_reason = "length"
            if seq.finish_reason is not None:
                self._finish(seq)
                finished.append(seq)
        return finished

    def _finish(self, seq: Sequence):
        seq.status = SequenceStatus.FINISHED
        self.block_manager.deallocate(seq)
        self.running.remove(seq)

    def abort(self, seq_id: int) -> Sequence | None:
        """Cancel a request (e.g. the HTTP client disconnected)."""
        for queue in (self.waiting, self.running):
            for seq in queue:
                if seq.seq_id == seq_id:
                    queue.remove(seq)
                    if seq.block_table:
                        self.block_manager.deallocate(seq)
                    seq.status = SequenceStatus.FINISHED
                    seq.finish_reason = "abort"
                    return seq
        return None
