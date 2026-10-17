"""Turn logits (a score for every word in the vocabulary) into the next token.

Pipeline for one decode step, applied to the whole batch at once (every row can use different settings):

  1. penalties   : repetition (HF style), presence / frequency (OpenAI style)
  2. min_tokens  : forbid end-of-sequence until enough tokens were generated
  3. temperature : divide scores by T; T<1 sharper, T>1 more random
  4. top-k       : only keep the k highest-scoring tokens
  5. top-p       : keep the smallest set of tokens whose probabilities add up to p
  6. min-p       : drop tokens whose probability is below min_p x (probability of the best token)
  7. draw        : greedy rows take the argmax; random rows draw from the softmax with the
                   "exponential race" trick: argmax(p_i / E_i), E_i ~ Exp(1) picks token i with
                   probability exactly p_i. Each row can have its own random generator (per-request
                   seed), so a seeded request is reproducible whatever else is in the batch.

Everything decided from Python-side SamplingParams (no GPU->CPU syncs except the final .tolist()).
"""

from dataclasses import dataclass

import torch

from nanoinfer.engine.sequence import Sequence


@dataclass
class TokenLogprob:
    token_id: int
    logprob: float
    top: list[tuple[int, float]]      # the k most likely tokens at this step (token_id, logprob)


# ----------------------------------------------------------------------------- pure tensor functions
def apply_penalties(logits: torch.Tensor, all_ids: torch.Tensor, out_mask: torch.Tensor,
                    rep: torch.Tensor, pres: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
    """logits [R, V]; all_ids [R, L] token history padded with V; out_mask [R, L] True for generated
    tokens (False for prompt and padding); rep/pres/freq [R]."""
    r, vocab = logits.shape
    ones = torch.ones_like(all_ids, dtype=logits.dtype)
    seen = torch.zeros(r, vocab + 1, dtype=torch.bool, device=logits.device).scatter_(1, all_ids, True)[:, :vocab]
    counts = torch.zeros(r, vocab + 1, dtype=logits.dtype, device=logits.device)
    counts.scatter_add_(1, all_ids, ones * out_mask)
    counts = counts[:, :vocab]
    rep = rep[:, None]
    logits = torch.where(seen, torch.where(logits > 0, logits / rep, logits * rep), logits)
    return logits - freq[:, None] * counts - pres[:, None] * (counts > 0).to(logits.dtype)


def filter_logits(logits: torch.Tensor, top_ks: torch.Tensor | None, top_ps: torch.Tensor | None,
                  min_ps: torch.Tensor | None, k_max: int | None = None) -> torch.Tensor:
    """Set the logits of dropped tokens to -inf. Each filter is skipped when its tensor is None.
    k_max (largest top_k in the batch, known on the Python side) avoids a GPU->CPU sync."""
    vocab = logits.shape[-1]
    neg_inf = float("-inf")
    if top_ps is not None:
        # full sort needed for top-p; top-k is applied on the sorted order at the same time
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        if top_ks is not None:
            ranks = torch.arange(vocab, device=logits.device)[None, :]
            k = torch.where(top_ks <= 0, torch.full_like(top_ks, vocab), top_ks.clamp(max=vocab))
            sorted_logits = sorted_logits.masked_fill(ranks >= k[:, None], neg_inf)
        probs = torch.softmax(sorted_logits, dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs          # drop a token once the ones above it reach p
        sorted_logits = sorted_logits.masked_fill(mass_before > top_ps[:, None], neg_inf)
        logits = torch.full_like(logits, neg_inf).scatter_(1, sorted_idx, sorted_logits)
    elif top_ks is not None:
        k = torch.where(top_ks <= 0, torch.full_like(top_ks, vocab), top_ks.clamp(max=vocab))
        k_max = min(int(k.max()) if k_max is None else k_max, vocab)
        top_vals = logits.topk(k_max, dim=-1).values                # [B, k_max] sorted descending
        thresh = top_vals.gather(1, (k.clamp(max=k_max) - 1)[:, None])
        logits = logits.masked_fill(logits < thresh, neg_inf)
    if min_ps is not None:
        probs = torch.softmax(logits, dim=-1)
        logits = logits.masked_fill(probs < min_ps[:, None] * probs.amax(dim=-1, keepdim=True), neg_inf)
    return logits


def sample(logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor,
           generator: torch.Generator | None = None, min_ps: torch.Tensor | None = None) -> torch.Tensor:
    """Functional form: logits [B, V] float32; temperatures/top_ps [B]; top_ks [B] long (<=0 = off).
    Returns [B] long. (Used by unit tests; the engine uses Sampler, which skips unused filters.)"""
    greedy_tokens = logits.argmax(dim=-1)
    is_greedy = temperatures < 1e-5
    temps = torch.where(is_greedy, torch.ones_like(temperatures), temperatures)
    filtered = filter_logits(logits / temps[:, None], top_ks, top_ps, min_ps)
    probs = torch.softmax(filtered, dim=-1)
    noise = torch.empty_like(probs).exponential_(generator=generator)
    sampled = (probs / noise).argmax(dim=-1)
    return torch.where(is_greedy, greedy_tokens, sampled)


# ----------------------------------------------------------------------------- engine-facing sampler
class Sampler:
    def __init__(self, device: torch.device, vocab_size: int, eos_token_ids: list[int], seed: int = 0):
        self.device = device
        self.vocab_size = vocab_size
        self.eos_token_ids = list(eos_token_ids)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def _f32(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32).to(self.device, non_blocking=True)

    def _long(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long).to(self.device, non_blocking=True)

    def _row_generator(self, seq: Sequence) -> torch.Generator:
        if seq.generator is None:
            seq.generator = torch.Generator(device=self.device)
            seq.generator.manual_seed(seq.sampling_params.seed)
        return seq.generator

    @torch.inference_mode()
    def __call__(self, logits: torch.Tensor, seqs: list[Sequence]) -> tuple[list[int], list[TokenLogprob | None] | None]:
        """logits [B, V] float32 for the sequences that need a new token. Returns (token ids, logprobs)."""
        sps = [s.sampling_params for s in seqs]
        raw_logits = logits
        vocab = logits.shape[-1]

        # 1) penalties (only rows that use them)
        pen_rows = [i for i, sp in enumerate(sps) if sp.has_penalties]
        if pen_rows:
            hist = [seqs[i].token_ids for i in pen_rows]
            width = max(len(h) for h in hist)
            ids = [h + [vocab] * (width - len(h)) for h in hist]
            mask = [[j >= seqs[i].num_prompt_tokens and j < len(h) for j in range(width)]
                    for i, h in zip(pen_rows, hist)]
            rows = self._long(pen_rows)
            sub = apply_penalties(logits[rows], self._long(ids), torch.tensor(mask).to(self.device),
                                  self._f32([sps[i].repetition_penalty for i in pen_rows]),
                                  self._f32([sps[i].presence_penalty for i in pen_rows]),
                                  self._f32([sps[i].frequency_penalty for i in pen_rows]))
            logits = logits.index_copy(0, rows, sub)

        # 2) min_tokens: ban end-of-sequence tokens for rows that have not generated enough yet
        ban = [(i, t) for i, (s, sp) in enumerate(zip(seqs, sps))
               if s.num_completion_tokens < sp.min_tokens
               for t in (([] if sp.ignore_eos else self.eos_token_ids) + sp.stop_token_ids) if t < vocab]
        if ban:
            if logits is raw_logits:
                logits = logits.clone()
            r, c = zip(*ban)
            logits[self._long(list(r)), self._long(list(c))] = float("-inf")

        # 3) greedy fast path
        greedy = [sp.is_greedy for sp in sps]
        tokens = logits.argmax(dim=-1)
        if not all(greedy):
            temps = self._f32([1.0 if g else sp.temperature for g, sp in zip(greedy, sps)])
            top_ks = self._long([sp.top_k for sp in sps]) if any(sp.top_k > 0 for sp in sps) else None
            top_ps = self._f32([sp.top_p for sp in sps]) if any(sp.top_p < 1.0 for sp in sps) else None
            min_ps = self._f32([sp.min_p for sp in sps]) if any(sp.min_p > 0 for sp in sps) else None
            k_max = max((vocab if sp.top_k <= 0 else sp.top_k) for sp in sps)
            filtered = filter_logits(logits / temps[:, None], top_ks, top_ps, min_ps, k_max)
            probs = torch.softmax(filtered, dim=-1)
            noise = torch.empty_like(probs).exponential_(generator=self.generator)
            for i, (s, sp) in enumerate(zip(seqs, sps)):
                if sp.seed is not None and not greedy[i]:
                    noise[i].exponential_(generator=self._row_generator(s))
            sampled = (probs / noise).argmax(dim=-1)
            tokens = torch.where(self._long(greedy).bool(), tokens, sampled)

        # 4) log-probabilities of the raw model distribution (only rows that ask for them)
        lp_rows = [i for i, sp in enumerate(sps) if sp.logprobs is not None]
        if not lp_rows:
            return tokens.tolist(), None
        rows = self._long(lp_rows)
        logp = torch.log_softmax(raw_logits[rows], dim=-1)
        chosen = logp.gather(1, tokens[rows][:, None]).squeeze(1)
        k = max(sps[i].logprobs for i in lp_rows)
        top_v, top_i = logp.topk(k, dim=-1) if k > 0 else (logp[:, :0], logp[:, :0].long())
        token_list = tokens.tolist()
        chosen, top_v, top_i = chosen.tolist(), top_v.tolist(), top_i.tolist()
        out: list[TokenLogprob | None] = [None] * len(seqs)
        for j, i in enumerate(lp_rows):
            kk = sps[i].logprobs
            out[i] = TokenLogprob(token_list[i], chosen[j], list(zip(top_i[j][:kk], top_v[j][:kk])))
        return token_list, out
