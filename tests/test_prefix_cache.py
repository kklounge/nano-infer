"""Prefix caching: block hashing, sharing with reference counts, revival, eviction."""

import torch

from nanoinfer import LLM, SamplingParams
from nanoinfer.engine.block_manager import BlockManager
from nanoinfer.engine.sequence import Sequence

BS = 4


def seq(tokens):
    return Sequence(tokens, SamplingParams(), BS)


def test_shared_prefix_reuses_blocks_with_refcount():
    bm = BlockManager(num_blocks=20, block_size=BS)
    system = list(range(100, 108))                   # 2 full blocks shared by both requests
    a = seq(system + [1, 2, 3])
    b = seq(system + [7, 8])
    bm.allocate(a)
    bm.allocate(b)
    assert a.num_cached_tokens == 0
    assert b.num_cached_tokens == 8
    assert a.block_table[:2] == b.block_table[:2]    # same physical blocks
    assert a.block_table[2] != b.block_table[2]
    for bid in a.block_table[:2]:
        assert bm.blocks[bid].ref_count == 2
    bm.deallocate(a)
    for bid in b.block_table[:2]:
        assert bm.blocks[bid].ref_count == 1         # still used by b -> not free
        assert bid not in bm.free_block_ids
    bm.deallocate(b)
    assert bm.num_free_blocks == 20


def test_partial_last_block_is_never_shared():
    bm = BlockManager(num_blocks=20, block_size=BS)
    a, b = seq([1, 2, 3, 4, 5, 6]), seq([1, 2, 3, 4, 5, 6])
    bm.allocate(a)
    bm.allocate(b)
    assert b.num_cached_tokens == 4                   # only the full first block
    assert a.block_table[1] != b.block_table[1]       # partial blocks get written by decode -> private


def test_fully_cached_prompt_still_computes_last_block():
    bm = BlockManager(num_blocks=20, block_size=BS)
    tokens = list(range(8))                           # exactly two full blocks
    a, b = seq(tokens), seq(tokens)
    bm.allocate(a)
    bm.allocate(b)
    # we need logits for the last token, so at least the last block is recomputed
    assert b.num_cached_tokens == 4
    assert b.num_cached_tokens < b.num_tokens


def test_freed_blocks_can_be_revived():
    bm = BlockManager(num_blocks=20, block_size=BS)
    prompt = list(range(12))
    a = seq(prompt + [99])
    bm.allocate(a)
    old = list(a.block_table[:3])
    bm.deallocate(a)                                  # blocks are free but keep their content + hash
    b = seq(prompt + [42])
    bm.allocate(b)
    assert b.num_cached_tokens == 12
    assert b.block_table[:3] == old
    assert bm.num_cache_hits == 3


def test_evicted_block_forgets_hash():
    bm = BlockManager(num_blocks=3, block_size=BS)
    a = seq([1, 2, 3, 4, 9])
    bm.allocate(a)
    h = bm.blocks[a.block_table[0]].hash
    assert h in bm.hash_to_block_id
    bm.deallocate(a)
    filler = seq(list(range(50, 62)))                 # needs all 3 blocks -> overwrites a's blocks
    bm.allocate(filler)
    assert h not in bm.hash_to_block_id
    bm.deallocate(filler)
    c = seq([1, 2, 3, 4, 9])
    bm.allocate(c)
    assert c.num_cached_tokens == 0


def test_hash_collision_is_caught_by_token_check():
    bm = BlockManager(num_blocks=10, block_size=BS)
    a = seq([1, 2, 3, 4, 5])
    bm.allocate(a)
    # simulate a collision: pretend the block holds different tokens
    bm.blocks[a.block_table[0]].token_ids = [9, 9, 9, 9]
    b = seq([1, 2, 3, 4, 6])
    bm.allocate(b)
    assert b.num_cached_tokens == 0


def test_decode_tokens_register_hash_for_future_requests():
    bm = BlockManager(num_blocks=10, block_size=BS)
    a = seq([1, 2, 3])
    bm.allocate(a)
    a.append_token(4)                                 # generated token fills block 0
    bm.may_append(a)
    assert bm.blocks[a.block_table[0]].hash is not None
    b = seq([1, 2, 3, 4, 5])                          # e.g. the next chat turn
    bm.allocate(b)
    assert b.num_cached_tokens == 4


def test_disabled_prefix_caching_never_hits():
    bm = BlockManager(num_blocks=10, block_size=BS, enable_prefix_caching=False)
    a, b = seq(list(range(9))), seq(list(range(9)))
    bm.allocate(a)
    bm.allocate(b)
    assert b.num_cached_tokens == 0
    assert not set(a.block_table) & set(b.block_table)


def test_engine_outputs_identical_with_and_without_prefix_cache(tiny_qwen2):
    torch.manual_seed(0)
    system = torch.randint(0, 500, (40,)).tolist()
    prompts = [system + torch.randint(0, 500, (n,)).tolist() for n in (3, 9, 17, 1)]
    sp = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True)

    on = LLM(str(tiny_qwen2), dtype="float32", device="cpu", block_size=8, max_model_len=256)
    first = on.generate(prompts[:1], sp)                # warm the cache with the shared system prompt
    rest = on.generate(prompts[1:], sp)
    assert on.stats.prefix_cache_hit_tokens >= 3 * 40 // 8 * 8
    assert all(o.num_cached_tokens == 40 for o in rest)

    off = LLM(str(tiny_qwen2), dtype="float32", device="cpu", block_size=8, max_model_len=256,
              enable_prefix_caching=False)
    ref = off.generate(prompts, sp)
    assert off.stats.prefix_cache_hit_tokens == 0
    assert [o.token_ids for o in first + rest] == [o.token_ids for o in ref]
