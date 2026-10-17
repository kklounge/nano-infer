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

Steps 4-7 run inside a *candidate window*: the few tokens each row could still pick, best first
(see _candidate_window). A vocabulary is ~152k wide, so touching all of it every step is what makes
sampling cost more than the model's own forward pass; top-k usually bounds the candidates to a
couple of dozen, and every token outside the window ends up with probability 0 anyway.
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
                    rep: torch.Tensor, pres: torch.Tensor, freq: torch.Tensor,
                    need_counts: bool = True) -> torch.Tensor:
    """logits [R, V]; all_ids [R, L] token history padded with V; out_mask [R, L] True for generated
    tokens (False for prompt and padding); rep/pres/freq [R].

    A penalty can only change the logit of a token that occurs in the history, so the work happens
    on the [R, L] history (a few thousand entries) instead of the [R, V] vocabulary (~10M at batch
    64). need_counts=False also skips the one vocabulary-sized buffer that presence / frequency
    penalties need for their occurrence counts; a repetition penalty on its own does not.
    """
    r, vocab = logits.shape
    # padding positions are redirected to the row's own first token: gathering and then scattering
    # it again writes the value it already had, and out_mask keeps padding out of the counts
    ids = torch.where(all_ids < vocab, all_ids, all_ids[:, :1])
    counted = None
    if need_counts:
        counts = torch.zeros(r, vocab, dtype=logits.dtype, device=logits.device)
        counts.scatter_add_(1, ids, out_mask.to(logits.dtype))
        counted = counts.gather(1, ids)       # how often ids[r, j] occurs among generated tokens
    rep = rep[:, None]
    vals = logits.gather(1, ids)
    vals = torch.where(vals > 0, vals / rep, vals * rep)
    if counted is not None:
        vals = vals - freq[:, None] * counted - pres[:, None] * (counted > 0).to(logits.dtype)
    # repeated token ids all carry the same value here, so which duplicate wins does not matter
    return logits.scatter(1, ids, vals)


def _apply_min_p(vals: torch.Tensor, min_ps: torch.Tensor) -> torch.Tensor:
    """Drop candidates whose probability is below min_p x (probability of the best candidate)."""
    probs = torch.softmax(vals, dim=-1)
    return vals.masked_fill(probs < min_ps[:, None] * probs.amax(dim=-1, keepdim=True), float("-inf"))


def _candidate_window(logits: torch.Tensor, top_ks: torch.Tensor | None, top_ps: torch.Tensor | None,
                      min_ps: torch.Tensor | None, k_max: int | None = None,
                      temps: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The tokens each row may still sample, best first: (vals, idx), both [R, W].

    vals = logits / temperature with dropped tokens at -inf; idx says which column of `logits` each
    one came from (None means "the window is the whole row, already in place"). Each filter is
    skipped when its tensor is None.

    W is the width of the window. When *every* row caps its candidates with top-k, only the
    W = max(top_k) best tokens of each row can ever be picked, so topk() is enough; otherwise the
    whole row has to be sorted. The difference dominates a decode step: at batch 64 over a 152k
    vocabulary a full sort costs ~6 ms and topk(20) ~0.7 ms, against ~8 ms for the model itself.
    k_max (the largest effective top_k, vocab when some row has top-k off) is known on the Python
    side from SamplingParams, so choosing W costs no GPU->CPU sync.
    """
    vocab = logits.shape[-1]
    neg_inf = float("-inf")
    if top_ks is None and top_ps is None:
        # nothing ranks the candidates, so no ordering is needed: min-p (if any) works in place
        vals = logits if temps is None else logits / temps[:, None]
        return (vals if min_ps is None else _apply_min_p(vals, min_ps)), None
    if top_ks is None:
        width = vocab
    else:
        if k_max is None:
            k_max = int(torch.where(top_ks <= 0, torch.full_like(top_ks, vocab), top_ks).max())
        width = min(max(k_max, 1), vocab)
    if width < vocab:
        vals, idx = logits.topk(width, dim=-1)              # topk returns them sorted, best first
    else:
        vals, idx = logits.sort(dim=-1, descending=True)
    if temps is not None:
        # dividing by a temperature > 0 keeps the order, so the window was picked correctly above
        vals = vals / temps[:, None]
    if top_ks is not None:
        ranks = torch.arange(width, device=logits.device)[None, :]
        k = torch.where(top_ks <= 0, torch.full_like(top_ks, width), top_ks.clamp(max=width))
        vals = vals.masked_fill(ranks >= k[:, None], neg_inf)
    if top_ps is not None:
        probs = torch.softmax(vals, dim=-1)
        mass_before = probs.cumsum(dim=-1) - probs          # drop a token once the ones above it reach p
        vals = vals.masked_fill(mass_before > top_ps[:, None], neg_inf)
    if min_ps is not None:
        vals = _apply_min_p(vals, min_ps)
    return vals, idx


def _draw(probs: torch.Tensor, idx: torch.Tensor | None, noise: torch.Tensor) -> torch.Tensor:
    """Exponential race: argmax(p_i / E_i) with E_i ~ Exp(1) picks candidate i with probability
    exactly p_i. probs / noise [R, W]; returns token ids [R]."""
    choice = (probs / noise).argmax(dim=-1)
    return choice if idx is None else idx.gather(1, choice[:, None]).squeeze(1)


def filter_logits(logits: torch.Tensor, top_ks: torch.Tensor | None, top_ps: torch.Tensor | None,
                  min_ps: torch.Tensor | None, k_max: int | None = None) -> torch.Tensor:
    """Set the logits of dropped tokens to -inf, full width. The readable reference form; the engine
    stays inside the candidate window instead and never materialises this."""
    vals, idx = _candidate_window(logits, top_ks, top_ps, min_ps, k_max)
    if idx is None:
        return vals
    return torch.full_like(logits, float("-inf")).scatter_(1, idx, vals)


def sample(logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor, top_ks: torch.Tensor,
           generator: torch.Generator | None = None, min_ps: torch.Tensor | None = None) -> torch.Tensor:
    """Functional form: logits [B, V] float32; temperatures/top_ps [B]; top_ks [B] long (<=0 = off).
    Returns [B] long. (Used by unit tests; the engine uses Sampler, which skips unused filters.)"""
    greedy_tokens = logits.argmax(dim=-1)
    is_greedy = temperatures < 1e-5
    temps = torch.where(is_greedy, torch.ones_like(temperatures), temperatures)
    vals, idx = _candidate_window(logits, top_ks, top_ps, min_ps, temps=temps)
    probs = torch.softmax(vals, dim=-1)
    noise = torch.empty_like(probs).exponential_(generator=generator)
    return torch.where(is_greedy, greedy_tokens, _draw(probs, idx, noise))


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
            # occurrence counts are only needed by the presence / frequency penalties
            need_counts = any(sps[i].presence_penalty or sps[i].frequency_penalty for i in pen_rows)
            sub = apply_penalties(logits[rows], self._long(ids), torch.tensor(mask).to(self.device),
                                  self._f32([sps[i].repetition_penalty for i in pen_rows]),
                                  self._f32([sps[i].presence_penalty for i in pen_rows]),
                                  self._f32([sps[i].frequency_penalty for i in pen_rows]),
                                  need_counts=need_counts)
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
            # A greedy row's drawn token is thrown away below, so give it the settings that cost
            # least: a one-candidate window and no filters. Left as they are, a single greedy
            # request (top_k is 0 by default) would widen the window to the whole vocabulary -- and
            # pull every other row into a full sort with it.
            eff_k = [1 if g else max(sp.top_k, 0) for g, sp in zip(greedy, sps)]
            sampled_sps = [sp for g, sp in zip(greedy, sps) if not g]
            temps = self._f32([1.0 if g else sp.temperature for g, sp in zip(greedy, sps)])
            top_ks = self._long(eff_k) if any(eff_k) else None
            top_ps = (self._f32([1.0 if g else sp.top_p for g, sp in zip(greedy, sps)])
                      if any(sp.top_p < 1.0 for sp in sampled_sps) else None)
            min_ps = (self._f32([0.0 if g else sp.min_p for g, sp in zip(greedy, sps)])
                      if any(sp.min_p > 0 for sp in sampled_sps) else None)
            k_max = max((vocab if k <= 0 else k) for k in eff_k)
            vals, idx = _candidate_window(logits, top_ks, top_ps, min_ps, k_max, temps=temps)
            probs = torch.softmax(vals, dim=-1)
            width = probs.shape[-1]
            noise = torch.empty_like(probs).exponential_(generator=self.generator)
            for i, (s, sp) in enumerate(zip(seqs, sps)):
                if sp.seed is not None and not greedy[i]:
                    # Draw this row's noise at ITS OWN candidate count, not the batch's window
                    # width, so a seeded request keeps getting the same token whatever else shares
                    # the batch. The slots past it hold candidates this row already dropped
                    # (probability 0), so any positive noise value will do for them.
                    w = width if eff_k[i] <= 0 else min(eff_k[i], width)
                    noise[i, :w].exponential_(generator=self._row_generator(s))
                    noise[i, w:] = 1.0
            tokens = torch.where(self._long(greedy).bool(), tokens, _draw(probs, idx, noise))

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
