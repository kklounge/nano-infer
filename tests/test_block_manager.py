"""Unit tests for the paged KV-cache block manager (no model needed)."""

import pytest

from nanoinfer.engine.block_manager import BlockManager, compute_block_hash
from nanoinfer.engine.sequence import Sequence
from nanoinfer.sampling_params import SamplingParams

BS = 4


def seq(tokens):
    return Sequence(tokens, SamplingParams(), BS)


def test_allocate_assigns_ceil_blocks_and_free_returns_them():
    bm = BlockManager(num_blocks=10, block_size=BS, enable_prefix_caching=False)
    s = seq(list(range(10)))             # 10 tokens -> 3 blocks
    assert bm.can_allocate(s)
    bm.allocate(s)
    assert len(s.block_table) == 3
    assert len(set(s.block_table)) == 3
    assert bm.num_free_blocks == 7
    bm.deallocate(s)
    assert s.block_table == []
    assert bm.num_free_blocks == 10
    assert all(b.ref_count == 0 for b in bm.blocks)


def test_can_allocate_false_when_full():
    bm = BlockManager(num_blocks=2, block_size=BS, enable_prefix_caching=False)
    assert not bm.can_allocate(seq(list(range(9))))   # needs 3 blocks
    assert bm.can_allocate(seq(list(range(8))))       # needs exactly 2


def test_decode_appends_new_block_exactly_at_block_boundary():
    bm = BlockManager(num_blocks=10, block_size=BS, enable_prefix_caching=False)
    s = seq([1, 2, 3, 4])                # exactly one full block
    bm.allocate(s)
    assert len(s.block_table) == 1
    s.append_token(5)                    # token at position 4 -> needs block #2
    assert bm.can_append(s)
    bm.may_append(s)
    assert len(s.block_table) == 2
    for t in (6, 7, 8):                  # positions 5,6,7 fit in block #2
        s.append_token(t)
        bm.may_append(s)
        assert len(s.block_table) == 2
    s.append_token(9)                    # position 8 -> block #3
    bm.may_append(s)
    assert len(s.block_table) == 3
    assert bm.num_free_blocks == 7


def test_can_append_false_when_no_free_block_at_boundary():
    bm = BlockManager(num_blocks=1, block_size=BS, enable_prefix_caching=False)
    s = seq([1, 2, 3, 4])
    bm.allocate(s)
    s.append_token(5)
    assert not bm.can_append(s)
    s2 = seq([1, 2])
    bm2 = BlockManager(num_blocks=1, block_size=BS, enable_prefix_caching=False)
    bm2.allocate(s2)
    s2.append_token(3)                   # position 2 still inside block #1 -> no new block needed
    assert bm2.can_append(s2)


def test_block_hash_chains_prefix():
    h1 = compute_block_hash([1, 2, 3, 4], None)
    assert h1 == compute_block_hash([1, 2, 3, 4], None)           # deterministic
    assert compute_block_hash([5, 6, 7, 8], h1) != compute_block_hash([5, 6, 7, 8], None)
    assert compute_block_hash([1, 2, 3, 5], None) != h1


def test_lru_eviction_order():
    bm = BlockManager(num_blocks=3, block_size=BS, enable_prefix_caching=False)
    a, b = seq([1]), seq([2])
    bm.allocate(a)
    bm.allocate(b)
    first_free = next(iter(bm.free_block_ids))
    bm.deallocate(a)                    # a's block becomes the most recently freed
    c = seq([3])
    bm.allocate(c)
    assert c.block_table == [first_free]   # least-recently freed block is reused first


def test_double_free_is_detected():
    bm = BlockManager(num_blocks=2, block_size=BS)
    s = seq([1, 2])
    bm.allocate(s)
    table = list(s.block_table)
    bm.deallocate(s)
    s.block_table = table
    with pytest.raises(AssertionError):
        bm.deallocate(s)
