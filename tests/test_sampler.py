"""Sampling: greedy, temperature, top-k, top-p, mixed settings in one batch."""

import torch

from nanoinfer.engine.sampler import sample


def _params(b, t=1.0, p=1.0, k=0):
    return (torch.full((b,), t), torch.full((b,), p), torch.full((b,), k, dtype=torch.long))


def test_greedy_is_argmax():
    logits = torch.randn(5, 100)
    assert torch.equal(sample(logits, *_params(5, t=0.0)), logits.argmax(-1))


def test_top_k_1_equals_greedy():
    logits = torch.randn(8, 50)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(sample(logits, *_params(8, t=1.0, k=1), g), logits.argmax(-1))


def test_tiny_top_p_equals_greedy():
    logits = torch.randn(8, 50)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(sample(logits, *_params(8, t=1.0, p=1e-6), g), logits.argmax(-1))


def test_top_k_only_samples_top_tokens():
    logits = torch.tensor([[5.0, 4.0, 3.0, -1.0, -2.0]]).repeat(2000, 1)
    g = torch.Generator().manual_seed(0)
    out = sample(logits, *_params(2000, t=1.0, k=2), g)
    assert set(out.tolist()) == {0, 1}


def test_top_p_keeps_smallest_set_reaching_p():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = probs.log().repeat(3000, 1)
    g = torch.Generator().manual_seed(0)
    out = sample(logits, *_params(3000, t=1.0, p=0.75), g)
    assert set(out.tolist()) == {0, 1}          # 0.5 + 0.3 >= 0.75 -> token 2 and 3 are dropped


def test_temperature_matches_softmax_distribution():
    logits = torch.tensor([[2.0, 1.0, 0.0]]).repeat(20000, 1)
    g = torch.Generator().manual_seed(0)
    for t in (0.5, 1.0, 2.0):
        out = sample(logits, *_params(20000, t=t), g)
        freq = torch.bincount(out, minlength=3).float() / 20000
        expected = torch.softmax(logits[0] / t, -1)
        assert torch.allclose(freq, expected, atol=0.02)


def test_mixed_rows_in_one_batch():
    logits = torch.tensor([[1.0, 3.0, 2.0]]).repeat(4, 1)
    temps = torch.tensor([0.0, 1.0, 0.0, 1.0])
    top_ps = torch.tensor([1.0, 1.0, 1.0, 1.0])
    top_ks = torch.tensor([0, 1, 0, 1])
    out = sample(logits, temps, top_ps, top_ks, torch.Generator().manual_seed(0))
    assert out.tolist() == [1, 1, 1, 1]


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(4, 1000)
    a = sample(logits, *_params(4), torch.Generator().manual_seed(123))
    b = sample(logits, *_params(4), torch.Generator().manual_seed(123))
    assert torch.equal(a, b)


# ----------------------------------------------------------------------------- engine-level Sampler
from nanoinfer import LLM, SamplingParams                      # noqa: E402
from nanoinfer.engine.sampler import Sampler, apply_penalties   # noqa: E402
from nanoinfer.engine.sequence import Sequence                  # noqa: E402

from conftest import hf_greedy                                  # noqa: E402


def _seq(tokens, n_prompt=None, **kw):
    s = Sequence(tokens[:n_prompt] if n_prompt else tokens, SamplingParams(**kw), 4)
    for t in tokens[len(s.token_ids):]:
        s.append_token(t)
    return s


def test_penalty_formulas():
    logits = torch.tensor([[2.0, -2.0, 1.0, 0.5]])
    ids = torch.tensor([[0, 1, 2, 2, 4]])            # 4 = padding (vocab size)
    out_mask = torch.tensor([[False, False, True, True, False]])   # tokens 0,1 are prompt; 2,2 generated
    rep = apply_penalties(logits, ids, out_mask, torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    assert rep.tolist() == [[1.0, -4.0, 0.5, 0.5]]   # seen tokens: positive / 2, negative * 2
    pf = apply_penalties(logits, ids, out_mask, torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.25]))
    assert pf.tolist() == [[2.0, -2.0, 1.0 - 0.5 - 0.25 * 2, 0.5]]   # only generated tokens, count = 2


def test_min_tokens_bans_eos():
    sampler = Sampler(torch.device("cpu"), 10, eos_token_ids=[3])
    logits = torch.zeros(2, 10)
    logits[:, 3] = 10.0                              # eos is by far the most likely
    a = _seq([1, 2], temperature=0, min_tokens=2)
    b = _seq([1, 2], temperature=0)
    toks, _ = sampler(logits, [a, b])
    assert toks[0] != 3 and toks[1] == 3


def test_logprobs_are_log_softmax_of_raw_logits():
    sampler = Sampler(torch.device("cpu"), 50, eos_token_ids=[])
    logits = torch.randn(3, 50)
    seqs = [_seq([1], temperature=0, logprobs=3), _seq([1], temperature=0), _seq([1], temperature=0, logprobs=0)]
    toks, lps = sampler(logits, seqs)
    ref = torch.log_softmax(logits, -1)
    assert lps[1] is None and lps[2].top == []
    assert lps[0].token_id == toks[0] == int(logits[0].argmax())
    assert abs(lps[0].logprob - ref[0, toks[0]].item()) < 1e-5
    assert [t for t, _ in lps[0].top] == ref[0].topk(3).indices.tolist()


def test_seeded_request_is_independent_of_batch():
    sampler = Sampler(torch.device("cpu"), 1000, eos_token_ids=[])
    logits = torch.randn(1, 1000)
    alone = [sampler(logits, [_seq([1], seed=42)])[0][0] for _ in range(1)]
    s = _seq([1], seed=42)
    other = [_seq([1]) for _ in range(3)]
    batched = sampler(logits.repeat(4, 1), other[:2] + [s] + other[2:])[0][2]
    assert alone[0] == batched


def test_min_p_filters_unlikely_tokens():
    logits = torch.tensor([[0.0, -0.1, -5.0, -6.0]]).repeat(2000, 1)
    g = torch.Generator().manual_seed(0)
    out = sample(logits, *_params(2000), g, min_ps=torch.full((2000,), 0.1))
    assert set(out.tolist()) == {0, 1}


def test_repetition_penalty_matches_hf(tiny_qwen2):
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(tiny_qwen2, dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(8)
    prompts = [torch.randint(0, 50, (n,), generator=g).tolist() for n in (5, 12)]   # small id range -> repeats
    refs = [hf.generate(torch.tensor([p]), max_new_tokens=16, do_sample=False, repetition_penalty=1.5,
                        temperature=None, top_p=None, top_k=None)[0][len(p):].tolist() for p in prompts]
    llm = LLM(str(tiny_qwen2), dtype="float32", device="cpu", max_model_len=128)
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True,
                                                repetition_penalty=1.5))
    assert [o.token_ids for o in outs] == refs
    plain = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=16, ignore_eos=True))
    assert [o.token_ids for o in plain] != refs      # the penalty really changed something


def test_engine_seed_and_n(tiny_qwen2):
    llm = LLM(str(tiny_qwen2), dtype="float32", device="cpu", max_model_len=128)
    p = list(range(10, 20))
    sp = SamplingParams(temperature=1.0, max_tokens=12, ignore_eos=True, seed=7)
    a = llm.generate([p], sp)[0].token_ids
    b = llm.generate([p, list(range(30, 45))], sp)[0].token_ids     # different batch, same seed
    assert a == b
    many = llm.generate([p], sp.clone(n=4))[0]
    assert len(many.outputs) == 4 and many.outputs[0].token_ids == a
    assert len({tuple(o.token_ids) for o in many.outputs}) > 1        # seeds 7, 8, 9, 10 differ


def test_stop_strings_cut_text(real_model):
    llm = LLM(str(real_model), dtype="float32", device="cpu", max_model_len=256)
    prompt = llm.apply_chat_template([{"role": "user", "content": "Count from 1 to 10, separated by commas."}])
    full = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=40))[0]
    assert "4" in full.text
    cut = llm.generate([prompt], SamplingParams(temperature=0, max_tokens=40, stop=["4"]))[0]
    assert cut.finish_reason == "stop" and cut.stop_reason == "4"
    assert cut.text == full.text[:full.text.index("4")]
    # streaming deltas (held back while they could be the start of a stop string) add up to the text
    rid = llm.add_request(prompt, SamplingParams(temperature=0, max_tokens=40, stop=[", 5"]))
    pieces = []
    while llm.has_unfinished_requests():
        pieces += [o.text for o in llm.step() if o.request_id == rid]
    out = llm.get_output(rid)
    assert "".join(pieces) == out.text == full.text[:full.text.index(", 5")]


# ----------------------------------------------------------------------------- candidate window
# The engine does steps 4-7 inside a narrow "candidate window" instead of over the whole vocabulary
# (sampler._candidate_window). These tests pin that down against a row-by-row reference.
import pytest                                                          # noqa: E402
from nanoinfer.engine.sampler import _candidate_window, filter_logits   # noqa: E402


def _reference_filter(logits, top_ks, top_ps, min_ps):
    """The documented pipeline (top-k, then top-p, then min-p), written out one row at a time.
    Each filter is off at its neutral value, exactly as SamplingParams documents: top_k <= 0,
    top_p == 1.0, min_p == 0. (Running top-p at p == 1.0 anyway would drop the last token now and
    then, because a float32 cumsum of the whole vocabulary reaches 1.0 before the end.)"""
    vocab = logits.shape[-1]
    out = torch.full_like(logits, float("-inf"))
    for r in range(logits.shape[0]):
        row = logits[r]
        order = torch.argsort(row, descending=True)
        k = vocab if top_ks[r] <= 0 else min(int(top_ks[r]), vocab)
        cand = order[:k]                                            # top-k
        if float(top_ps[r]) < 1.0:                                  # top-p
            probs = torch.softmax(row[cand], dim=-1)
            mass_before = probs.cumsum(0) - probs
            cand = cand[mass_before <= float(top_ps[r])]
        if float(min_ps[r]) > 0:                                    # min-p
            probs = torch.softmax(row[cand], dim=-1)
            cand = cand[probs >= float(min_ps[r]) * float(probs.max())]
        out[r, cand] = row[cand]
    return out


@pytest.mark.parametrize("vocab", [37, 3000])
@pytest.mark.parametrize("top_k", [0, 1, 8, 10**6, "mixed"])
@pytest.mark.parametrize("top_p", [1.0, 0.7])
@pytest.mark.parametrize("min_p", [0.0, 0.1])
def test_candidate_window_matches_row_by_row_reference(vocab, top_k, top_p, min_p):
    b = 6
    logits = torch.randn(b, vocab) * 4                       # random floats: no ties to break
    ks = [0 if i % 2 else 8 for i in range(b)] if top_k == "mixed" else [top_k] * b
    top_ks = torch.tensor(ks, dtype=torch.long) if any(ks) else None
    top_ps = torch.full((b,), top_p) if top_p < 1.0 else None
    min_ps = torch.full((b,), min_p) if min_p > 0 else None
    k_max = max((vocab if k <= 0 else k) for k in ks)
    got = filter_logits(logits, top_ks, top_ps, min_ps, k_max)
    want = _reference_filter(logits, ks, [top_p] * b, [min_p] * b)
    # the kept tokens, and the distribution they imply, must be the same
    assert torch.equal(got > float("-inf"), want > float("-inf"))
    torch.testing.assert_close(torch.softmax(got, -1), torch.softmax(want, -1), atol=1e-6, rtol=1e-6)


def test_window_is_narrow_when_top_k_bounds_it():
    """The whole point: no vocabulary-wide sort when every row caps its candidates."""
    logits = torch.randn(4, 50_000)
    _, idx = _candidate_window(logits, torch.full((4,), 20, dtype=torch.long), torch.full((4,), 0.9), None, 20)
    assert idx.shape == (4, 20)
    # one row with top-k off needs the whole row, and says so
    ks = torch.tensor([20, 0, 20, 20])
    _, idx = _candidate_window(logits, ks, torch.full((4,), 0.9), None, 50_000)
    assert idx.shape == (4, 50_000)


def test_greedy_row_does_not_disturb_the_others():
    """A greedy row must not change what a sampling row draws (it used to widen the window)."""
    sampler = Sampler(torch.device("cpu"), 500, eos_token_ids=[])
    logits = torch.randn(1, 500) * 3
    sp = dict(temperature=0.8, top_p=0.8, top_k=20, seed=11)
    alone = sampler(logits, [_seq([1], **sp)])[0][0]
    with_greedy = sampler(logits.repeat(3, 1),
                          [_seq([1], temperature=0), _seq([1], **sp), _seq([1], temperature=0)])[0]
    assert with_greedy[1] == alone
    assert with_greedy[0] == with_greedy[2] == int(logits[0].argmax())


def test_penalties_without_counts_match_the_full_path():
    """need_counts=False is the repetition-penalty-only shortcut: it must agree with the full form."""
    logits = torch.randn(3, 200) * 2
    ids = torch.randint(0, 60, (3, 12))
    ids[:, 9:] = 200                                         # tail padding (= vocab size)
    mask = (ids < 200) & (torch.arange(12)[None, :] >= 4)
    rep, zero = torch.full((3,), 1.3), torch.zeros(3)
    full = apply_penalties(logits.clone(), ids, mask, rep, zero, zero, need_counts=True)
    quick = apply_penalties(logits.clone(), ids, mask, rep, zero, zero, need_counts=False)
    torch.testing.assert_close(full, quick)
    assert not torch.equal(full, logits)                     # the penalty did something
