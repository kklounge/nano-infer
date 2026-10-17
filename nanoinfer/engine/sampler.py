"""Turn logits (a score for every word in the vocabulary) into the next token.

greedy      : take the highest score (temperature = 0)
temperature : divide scores by T; T<1 sharper, T>1 more random
top-k       : only keep the k highest-scoring tokens
top-p       : keep the smallest set of tokens whose probabilities add up to p
Every row in the batch can have different settings; everything is vectorised.
"""

import torch


def sample(logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V] float32; temperatures/top_ps [B] float; top_ks [B] long (<=0 = off). Returns [B] long."""
    greedy_tokens = logits.argmax(dim=-1)
    is_greedy = temperatures < 1e-5
    if bool(is_greedy.all()):
        return greedy_tokens

    vocab = logits.shape[-1]
    temps = torch.where(is_greedy, torch.ones_like(temperatures), temperatures)
    logits = logits / temps[:, None]

    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    ranks = torch.arange(vocab, device=logits.device)[None, :]
    # top-k: drop everything ranked k or worse
    k = torch.where(top_ks <= 0, torch.full_like(top_ks, vocab), top_ks.clamp(max=vocab))
    sorted_logits = sorted_logits.masked_fill(ranks >= k[:, None], float("-inf"))
    # top-p: drop a token if the tokens ranked above it already reach probability p
    probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = probs.cumsum(dim=-1) - probs
    sorted_logits = sorted_logits.masked_fill(mass_before > top_ps[:, None], float("-inf"))

    probs = torch.softmax(sorted_logits, dim=-1)
    choice = torch.multinomial(probs, num_samples=1, generator=generator)   # index into the sorted order
    sampled = sorted_idx.gather(1, choice).squeeze(1)
    return torch.where(is_greedy, greedy_tokens, sampled)
