"""Block manager: the "librarian" of the paged KV cache.

The KV cache is cut into `num_blocks` fixed-size blocks (pages) of `block_size` tokens.
The block manager decides which physical block each sequence uses:

* allocate(seq)   - give a new sequence enough blocks for its prompt
* may_append(seq) - give a growing sequence a new block when its last block is full
* deallocate(seq) - return a finished/preempted sequence's blocks

Prefix caching (前缀缓存):
  Every FULL block gets a hash computed from (hash of the previous block, token ids in this block).
  Because the hash chains through all previous blocks, equal hashes mean equal *prefixes*.
  If a new request's block hash is already known, we reuse that physical block instead of
  recomputing it. Blocks are shared with a reference count (ref_count); a block is only
  truly free when nobody uses it. Freed blocks keep their content and hash, so they can
  still be "revived" by a later request until they are overwritten (LRU eviction).
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
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        # free blocks in LRU order: the front is evicted first. OrderedDict gives O(1) removal from the middle,
        # which we need when a freed-but-cached block is revived by a prefix-cache hit.
        self.free_block_ids: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.hash_to_block_id: dict[int, int] = {}
        # statistics
        self.num_cache_queries = 0     # full prompt blocks looked up
        self.num_cache_hits = 0        # ... of which were reused

    # ------------------------------------------------------------------ helpers
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

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

    # ------------------------------------------------------------------ prefill
    def can_allocate(self, seq: Sequence) -> bool:
        # conservative: pretend nothing is cached
        return self.num_free_blocks >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Assign blocks to every token of `seq` (used before its prefill)."""
        assert not seq.block_table
        seq.num_cached_tokens = 0
        prefix_hash = None
        cache_miss = False
        n = seq.num_blocks
        for i in range(n):
            token_ids = seq.block(i)
            is_full = len(token_ids) == self.block_size
            block_hash = compute_block_hash(token_ids, prefix_hash) if (is_full and self.enable_prefix_caching) else None
            if block_hash is not None:
                self.num_cache_queries += 1
            # The last token must always be computed (we need its logits to sample the next token),
            # so a hit on the final block of a fully-cached prompt is treated as a miss.
            is_last_full_block = (i == n - 1) and is_full
            block_id = self.hash_to_block_id.get(block_hash) if block_hash is not None else None
            if (block_id is None or cache_miss or is_last_full_block
                    or self.blocks[block_id].token_ids != token_ids):
                cache_miss = True
                block = self._take_free_block()
            else:
                # prefix cache hit: share the existing block, skip computing these tokens
                self.num_cache_hits += 1
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._take_free_block(block_id)
            if block_hash is not None:
                self._register(block, block_hash, token_ids)
            seq.block_table.append(block.block_id)
            prefix_hash = block_hash

    def deallocate(self, seq: Sequence):
        # free in reverse so tail blocks are evicted before shared prefix blocks
        for block_id in reversed(seq.block_table):
            self._release(block_id)
        seq.block_table = []
        seq.num_cached_tokens = 0

    # ------------------------------------------------------------------ decode
    def can_append(self, seq: Sequence) -> bool:
        """The token fed in this decode step sits at position len(seq)-1.
        It needs a fresh block exactly when that position starts a new block."""
        needs_new_block = (seq.num_tokens - 1) % self.block_size == 0
        return self.num_free_blocks >= int(needs_new_block)

    def may_append(self, seq: Sequence):
        pos = seq.num_tokens - 1
        if pos % self.block_size == 0 and len(seq.block_table) * self.block_size <= pos:
            seq.block_table.append(self._take_free_block().block_id)
        if self.enable_prefix_caching and (pos + 1) % self.block_size == 0:
            # this step fills the last block -> register its hash so future requests can reuse it
            idx = pos // self.block_size
            block = self.blocks[seq.block_table[idx]]
            if block.hash is None:
                prev_hash = self.blocks[seq.block_table[idx - 1]].hash if idx > 0 else None
                if idx == 0 or prev_hash is not None:
                    token_ids = seq.block(idx)
                    self._register(block, compute_block_hash(token_ids, prev_hash), token_ids)
