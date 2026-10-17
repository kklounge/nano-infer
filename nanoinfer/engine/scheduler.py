"""Scheduler: decides WHICH sequences run in the next forward pass, and HOW MANY tokens each computes.

Two queues:
  waiting : requests that have no KV cache yet (new, or preempted)
  running : requests that own KV-cache blocks (being prefilled or decoded)

Every step has a token budget (max_num_batched_tokens). schedule() fills it in this order:
  1. running requests first, oldest first: a decoding request needs 1 token; a request in the
     middle of a long prompt (chunked prefill) gets the next chunk of its prompt;
  2. then waiting requests, first come first served: look up the prefix cache, then schedule
     as much of the prompt as the remaining budget allows (the rest continues next step).

So one step can MIX decode tokens and prefill chunks (like vLLM v1 / Sarathi-Serve). Decoding
never stalls behind a long prompt, and a 10k-token prompt cannot blow up the activation memory.

Continuous batching = this happens after EVERY step: finished requests leave and waiting requests join.

Preemption: if the KV cache is full and a running request needs a new block, we kick out the
most recently admitted running request: free its blocks and put it back at the FRONT of the
waiting queue. Later it is recomputed from its tokens (prompt + what it already generated); the
blocks it had filled stay in the prefix cache, so recomputation is usually mostly cache hits.
"""

from collections import deque
from dataclasses import dataclass, field

from nanoinfer.config import EngineConfig
from nanoinfer.engine.block_manager import BlockManager
from nanoinfer.engine.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    seqs: list[Sequence] = field(default_factory=list)   # 1-token (decode) sequences first, then prefill chunks
    num_decode: int = 0                                   # how many of seqs compute exactly one token
    num_batched_tokens: int = 0
    num_preempted: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.seqs

    @property
    def num_prefill(self) -> int:
        return len(self.seqs) - self.num_decode

    @property
    def sampling_seqs(self) -> list[Sequence]:
        """Sequences whose scheduled tokens reach the end -> a new token is sampled for them."""
        return [s for s in self.seqs if s.num_computed_tokens + s.num_scheduled_tokens == s.num_tokens]


class Scheduler:
    def __init__(self, config: EngineConfig, num_blocks: int, eos_token_ids: list[int]):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_model_len = config.max_model_len
        self.chunked_prefill = config.enable_chunked_prefill
        self.eos_token_ids = set(eos_token_ids)
        self.block_size = config.block_size
        self.block_manager = BlockManager(num_blocks, config.block_size, config.enable_prefix_caching)
        # keep a few blocks free when admitting NEW requests, so running ones can still grow
        self.watermark_blocks = int(num_blocks * config.kv_watermark)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0
        self.num_admitted_tokens = 0          # prompt (+ recomputed) tokens of admitted sequences
        self.num_prefix_hit_tokens = 0        # ... of which were served by the prefix cache

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # ------------------------------------------------------------------ the core
    def schedule(self) -> SchedulerOutput:
        bm = self.block_manager
        budget = self.max_num_batched_tokens
        scheduled: list[Sequence] = []
        not_scheduled: list[Sequence] = []
        preempted = 0

        # ---- 1) running sequences (oldest first)
        while self.running:
            seq = self.running.popleft()
            n = min(seq.num_uncomputed_tokens, budget)
            if n == 0:
                not_scheduled.append(seq)
                continue
            while not bm.can_allocate_slots(seq, n):
                if self.running:
                    self._preempt(self.running.pop())     # victim: the newest running request
                    preempted += 1
                else:
                    self._preempt(seq)                    # nobody left to evict: wait ourselves
                    preempted += 1
                    seq = None
                    break
            if seq is None:
                break
            bm.allocate_slots(seq, n)
            seq.num_scheduled_tokens = n
            budget -= n
            scheduled.append(seq)
        self.running.extend(scheduled)
        self.running.extend(not_scheduled)

        # ---- 2) waiting sequences (FCFS); never right after a preemption (memory is tight)
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs and not preempted:
            seq = self.waiting[0]
            hit_ids, _ = bm.get_cached_prefix(seq)
            num_cached = len(hit_ids) * self.block_size
            remaining = seq.num_tokens - num_cached
            n = min(remaining, budget)
            if n < remaining and not self.chunked_prefill:
                break
            need = -(-(num_cached + n) // self.block_size) - len(hit_ids)
            reserve = self.watermark_blocks if self.running else 0
            if bm.num_free_after_revive(hit_ids) - reserve < need:
                break
            self.waiting.popleft()
            bm.allocate_prefix(seq, hit_ids)
            bm.allocate_slots(seq, n)
            seq.num_scheduled_tokens = n
            seq.status = SequenceStatus.RUNNING
            if seq.num_preemptions == 0:
                seq.num_prefix_hit_tokens = num_cached
            self.num_admitted_tokens += seq.num_tokens
            self.num_prefix_hit_tokens += num_cached
            budget -= n
            self.running.append(seq)
            scheduled.append(seq)

        if not scheduled and not preempted and not self.is_finished():
            raise RuntimeError("KV cache too small to run even one sequence; increase num_kvcache_blocks")
        decode = [s for s in scheduled if s.num_scheduled_tokens == 1]
        prefill = [s for s in scheduled if s.num_scheduled_tokens > 1]
        return SchedulerOutput(decode + prefill, len(decode),
                               self.max_num_batched_tokens - budget, preempted)

    def _preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.block_manager.deallocate(seq)
        seq.num_scheduled_tokens = 0
        self.waiting.appendleft(seq)

    # ------------------------------------------------------------------ after the forward pass
    def postprocess(self, out: SchedulerOutput, token_ids: list[int]) -> list[Sequence]:
        """Advance computed counters; append sampled tokens (one per sampling sequence, in order);
        finish sequences that hit a stop token or a length limit. Returns finished sequences."""
        finished = []
        it = iter(token_ids)
        for seq in out.seqs:
            seq.num_computed_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.num_computed_tokens < seq.num_tokens:
                continue                                    # mid-prompt chunk: nothing sampled yet
            tok = next(it)
            seq.append_token(tok)
            sp = seq.sampling_params
            can_stop = seq.num_completion_tokens >= sp.min_tokens
            if can_stop and tok in sp.stop_token_ids:
                seq.finish_reason, seq.stop_reason = "stop", tok
            elif can_stop and not sp.ignore_eos and tok in self.eos_token_ids:
                seq.finish_reason = "stop"
            elif seq.num_completion_tokens >= sp.max_tokens or seq.num_tokens >= self.max_model_len:
                seq.finish_reason = "length"
            if seq.finish_reason is not None:
                self.finish(seq)
                finished.append(seq)
        return finished

    def finish(self, seq: Sequence, reason: str | None = None):
        if reason is not None:
            seq.finish_reason = reason
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
