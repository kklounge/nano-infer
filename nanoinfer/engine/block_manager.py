"""Block manager: the "librarian" of the paged KV cache.

The KV cache is cut into `num_blocks` fixed-size blocks (pages) of `block_size` tokens.
The block manager decides which physical block each sequence uses:

* get_cached_prefix(seq) - find blocks of the prompt that are already in the cache (no side effects)
* allocate_prefix(seq)   - take those blocks for a newly admitted sequence
* allocate_slots(seq, n) - make sure the sequence has blocks for its next n tokens
                           (a prompt chunk during prefill, or 1 token during decode)
* deallocate(seq)        - return a finished/preempted sequence's blocks

Prefix caching (前缀缓存):
  Every FULL block gets a hash computed from (hash of the previous block, token ids in this block).
  Because the hash chains through all previous blocks, equal hashes mean equal *prefixes*.
  If a new request's block hash is already known, we reuse that physical block instead of
  recomputing it. Blocks are shared with a reference count (ref_count); a block is only
  truly free when nobody uses it. Freed blocks keep their content and hash, so they can
  still be "revived" by a later request until they are overwritten (LRU eviction).

When is a block's hash published? Only when its K/V will really be in the cache by the end of the
current step: at allocate_slots() time for blocks that this step's tokens fill completely. (With
chunked prefill, the blocks of later chunks are not computed yet, so they must not be findable.)
Sharing inside one step is safe because every layer writes the K/V of ALL tokens in the batch
before it computes any attention.
"""

import hashlib
from collections import OrderedDict

from nanoinfer.engine.sequence import Sequence


def compute_block_hash(token_ids: list[int], prefix_hash: int | None) -> int:
    """Deterministic hash of a block's tokens chained with the previous block's hash."""
    h = hashlib.blake2b(digest_size=8)
    if prefix_hash is not None:
        h.update(prefix_hash.to_bytes(8, "little"))
    h.update(b"".join(t.to_bytes(4, "little") for t in token_ids))
    return int.from_bytes(h.digest(), "little")


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash: int | None = None          # set only when the block is full (and prefix caching on)
        self.token_ids: list[int] = []        # kept to double-check hash hits (guards against collisions)

    def reset(self):
        self.ref_count = 1
        self.hash = None
        self.token_ids = []

    def __repr__(self):
        return f"Block(id={self.block_id}, ref={self.ref_count}, hash={self.hash})"


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        # free blocks in LRU order: the front is evicted first. OrderedDict gives O(1) removal from the middle,
        # which we need when a freed-but-cached block is revived by a prefix-cache hit.
        self.free_block_ids: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.hash_to_block_id: dict[int, int] = {}
        # statistics
        self.num_cache_queries = 0     # full prompt blocks looked up at admission
        self.num_cache_hits = 0        # ... of which were reused

    # ------------------------------------------------------------------ helpers
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    @property
    def usage(self) -> float:
        return len(self.used_block_ids) / max(self.num_blocks, 1)

    def _take_free_block(self, block_id: int | None = None) -> Block:
        """Take a specific free block (revive) or the least-recently-freed one (evict)."""
        if block_id is None:
            block_id, _ = self.free_block_ids.popitem(last=False)
            block = self.blocks[block_id]
            # its old content is about to be overwritten: forget its hash
            if block.hash is not None and self.hash_to_block_id.get(block.hash) == block_id:
                del self.hash_to_block_id[block.hash]
            block.reset()
        else:
            del self.free_block_ids[block_id]
            block = self.blocks[block_id]
            block.ref_count = 1      # revived: keep hash and token_ids
        self.used_block_ids.add(block_id)
        return block

    def _release(self, block_id: int):
        block = self.blocks[block_id]
        block.ref_count -= 1
        assert block.ref_count >= 0
        if block.ref_count == 0:
            self.used_block_ids.remove(block_id)
            self.free_block_ids[block_id] = None    # most recently freed -> evicted last

    def _register(self, block: Block, block_hash: int, token_ids: list[int]):
        block.hash = block_hash
        block.token_ids = list(token_ids)
        self.hash_to_block_id[block_hash] = block.block_id

    def blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        """Extra blocks the sequence needs so that its next num_new_tokens fit."""
        total = seq.num_computed_tokens + num_new_tokens
        return max(0, -(-total // self.block_size) - len(seq.block_table))

    # ------------------------------------------------------------------ admission (prefix cache lookup)
    def get_cached_prefix(self, seq: Sequence) -> tuple[list[int], list[int]]:
        """Longest run of leading full blocks already in the cache. Returns (block ids, hashes).
        The last prompt token is never served from the cache: its logits are needed for sampling."""
        if not self.enable_prefix_caching:
            return [], []
        ids, hashes = [], []
        prefix_hash = None
        for i in range((seq.num_tokens - 1) // self.block_size):
            token_ids = seq.block(i)
            h = compute_block_hash(token_ids, prefix_hash)
            block_id = self.hash_to_block_id.get(h)
            if block_id is None or self.blocks[block_id].token_ids != token_ids:
                break
            ids.append(block_id)
            hashes.append(h)
            prefix_hash = h
        return ids, hashes

    def num_free_after_revive(self, block_ids: list[int]) -> int:
        """Free blocks left once the given cached blocks are taken (free-but-cached ones count)."""
        return self.num_free_blocks - sum(1 for b in block_ids if b not in self.used_block_ids)

    def allocate_prefix(self, seq: Sequence, block_ids: list[int]):
        """Admit `seq`: share the cached blocks found by get_cached_prefix()."""
        assert not seq.block_table
        if self.enable_prefix_caching:
            self.num_cache_queries += (seq.num_tokens - 1) // self.block_size
            self.num_cache_hits += len(block_ids)
        for block_id in block_ids:
            if block_id in self.used_block_ids:
                self.blocks[block_id].ref_count += 1
            else:
                self._take_free_block(block_id)
            seq.block_table.append(block_id)
        seq.num_cached_tokens = len(block_ids) * self.block_size
        seq.num_computed_tokens = seq.num_cached_tokens

    # ------------------------------------------------------------------ every step
    def can_allocate_slots(self, seq: Sequence, num_new_tokens: int, reserve: int = 0) -> bool:
        return self.num_free_blocks - reserve >= self.blocks_needed(seq, num_new_tokens)

    def allocate_slots(self, seq: Sequence, num_new_tokens: int):
        """Give `seq` blocks for tokens [num_computed, num_computed + n) and publish the hashes of the
        blocks those tokens complete."""
        for _ in range(self.blocks_needed(seq, num_new_tokens)):
            seq.block_table.append(self._take_free_block().block_id)
        if not self.enable_prefix_caching:
            return
        bs = self.block_size
        first, end = seq.num_computed_tokens // bs, (seq.num_computed_tokens + num_new_tokens) // bs
        for idx in range(first, end):               # blocks that become full in this step
            block = self.blocks[seq.block_table[idx]]
            if block.hash is not None:
                continue
            prev_hash = self.blocks[seq.block_table[idx - 1]].hash if idx > 0 else None
            if idx > 0 and prev_hash is None:
                return                              # chain broken (should not happen); stop publishing
            token_ids = seq.block(idx)
            self._register(block, compute_block_hash(token_ids, prev_hash), token_ids)

    def deallocate(self, seq: Sequence):
        # free in reverse so tail blocks are evicted before shared prefix blocks
        for block_id in reversed(seq.block_table):
            self._release(block_id)
        seq.block_table = []
        seq.num_computed_tokens = 0
