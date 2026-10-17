"""Unit tests for the paged KV-cache block manager (no model needed)."""

import pytest

from nanoinfer.engine.block_manager import BlockManager, compute_block_hash
from nanoinfer.engine.sequence import Sequence
from nanoinfer.sampling_params import SamplingParams

BS = 4


def seq(tokens):
    return Sequence(tokens, SamplingParams(), BS)


def admit(bm, s, n=None):
    """What the scheduler does for a new request: prefix-cache lookup, then slots for n tokens."""
    hits, _ = bm.get_cached_prefix(s)
    bm.allocate_prefix(s, hits)
    n = s.num_tokens - s.num_computed_tokens if n is None else n
    bm.allocate_slots(s, n)
    return n


def compute(s, n):
    """Pretend the model computed n more tokens."""
    s.num_computed_tokens += n


def decode_one(bm, s, token):
    """Sample a token, then give the sequence room to compute it in the next step."""
    s.append_token(token)
    bm.allocate_slots(s, 1)
    compute(s, 1)


def test_allocate_assigns_ceil_blocks_and_free_returns_them():
    bm = BlockManager(num_blocks=10, block_size=BS, enable_prefix_caching=False)
    s = seq(list(range(10)))             # 10 tokens -> 3 blocks
    assert bm.can_allocate_slots(s, 10)
    admit(bm, s)
    assert len(s.block_table) == 3
    assert len(set(s.block_table)) == 3
    assert bm.num_free_blocks == 7
    bm.deallocate(s)
    assert s.block_table == []
    assert bm.num_free_blocks == 10
    assert all(b.ref_count == 0 for b in bm.blocks)


def test_can_allocate_false_when_full():
    bm = BlockManager(num_blocks=2, block_size=BS, enable_prefix_caching=False)
    assert not bm.can_allocate_slots(seq(list(range(9))), 9)   # needs 3 blocks
    assert bm.can_allocate_slots(seq(list(range(8))), 8)       # needs exactly 2


def test_chunked_allocation_is_lazy():
    """With chunked prefill, blocks are only taken for the chunk being computed."""
    bm = BlockManager(num_blocks=10, block_size=BS, enable_prefix_caching=False)
    s = seq(list(range(10)))
    admit(bm, s, n=5)                    # first chunk: tokens 0..4 -> 2 blocks
    assert len(s.block_table) == 2
    compute(s, 5)
    bm.allocate_slots(s, 3)              # tokens 5..7 fit in block #2
    assert len(s.block_table) == 2
    compute(s, 3)
    bm.allocate_slots(s, 2)              # tokens 8..9 -> block #3
    assert len(s.block_table) == 3


def test_decode_appends_new_block_exactly_at_block_boundary():
    bm = BlockManager(num_blocks=10, block_size=BS, enable_prefix_caching=False)
    s = seq([1, 2, 3, 4])                # exactly one full block
    compute(s, admit(bm, s))
    assert len(s.block_table) == 1
    decode_one(bm, s, 5)                 # token at position 4 -> needs block #2
    assert len(s.block_table) == 2
    for t in (6, 7, 8):                  # positions 5,6,7 fit in block #2
        decode_one(bm, s, t)
        assert len(s.block_table) == 2
    decode_one(bm, s, 9)                 # position 8 -> block #3
    assert len(s.block_table) == 3
    assert bm.num_free_blocks == 7


def test_can_append_false_when_no_free_block_at_boundary():
    bm = BlockManager(num_blocks=1, block_size=BS, enable_prefix_caching=False)
    s = seq([1, 2, 3, 4])
    compute(s, admit(bm, s))
    s.append_token(5)
    assert not bm.can_allocate_slots(s, 1)
    bm2 = BlockManager(num_blocks=1, block_size=BS, enable_prefix_caching=False)
    s2 = seq([1, 2])
    compute(s2, admit(bm2, s2))
    s2.append_token(3)                   # position 2 still inside block #1 -> no new block needed
    assert bm2.can_allocate_slots(s2, 1)


def test_block_hash_chains_prefix():
    h1 = compute_block_hash([1, 2, 3, 4], None)
    assert h1 == compute_block_hash([1, 2, 3, 4], None)           # deterministic
    assert compute_block_hash([5, 6, 7, 8], h1) != compute_block_hash([5, 6, 7, 8], None)
    assert compute_block_hash([1, 2, 3, 5], None) != h1


def test_lru_eviction_order():
    bm = BlockManager(num_blocks=3, block_size=BS, enable_prefix_caching=False)
    a, b = seq([1]), seq([2])
    admit(bm, a)
    admit(bm, b)
    first_free = next(iter(bm.free_block_ids))
    bm.deallocate(a)                    # a's block becomes the most recently freed
    c = seq([3])
    admit(bm, c)
    assert c.block_table == [first_free]   # least-recently freed block is reused first


def test_double_free_is_detected():
    bm = BlockManager(num_blocks=2, block_size=BS)
    s = seq([1, 2])
    admit(bm, s)
    table = list(s.block_table)
    bm.deallocate(s)
    s.block_table = table
    with pytest.raises(AssertionError):
        bm.deallocate(s)


def test_usage_fraction():
    bm = BlockManager(num_blocks=4, block_size=BS)
    s = seq(list(range(6)))
    admit(bm, s)
    assert bm.usage == 0.5
