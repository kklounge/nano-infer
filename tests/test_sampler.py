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
