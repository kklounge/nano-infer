"""The user-facing engine: LLM.generate(), plus the step() loop that ties everything together.

One engine step =
    scheduler.schedule()    -> which sequences run now and how many tokens each computes
    model_runner.run()      -> one forward pass + sampling -> one new token per finished prompt/decode
    scheduler.postprocess() -> append tokens, finish sequences, free their KV blocks
    (+ incremental detokenization and stop strings, for requests that need text while generating)
"""

import logging
import os
import time
from dataclasses import dataclass, field

from nanoinfer.config import EngineConfig, ModelConfig
from nanoinfer.engine.detokenizer import check_stop_strings, detokenize_incrementally, emittable_text
from nanoinfer.engine.model_runner import ModelRunner
from nanoinfer.engine.sampler import TokenLogprob
from nanoinfer.engine.scheduler import Scheduler
from nanoinfer.engine.sequence import Sequence
from nanoinfer.sampling_params import SamplingParams
from nanoinfer.utils import pick_device, pick_dtype

logger = logging.getLogger(__name__)


@dataclass
class CompletionOutput:
    index: int                           # which of the n samples
    token_ids: list[int]                 # generated tokens only
    text: str | None                     # decoded text (None if no tokenizer is available)
    finish_reason: str | None            # "stop" | "length" | "abort"
    stop_reason: "str | int | None" = None
    logprobs: list[TokenLogprob] | None = None
    ttft: float | None = None
    latency: float | None = None


@dataclass
class RequestOutput:
    request_id: int
    prompt_token_ids: list[int]
    token_ids: list[int]                 # generated tokens of the first sample
    text: str | None
    finish_reason: str | None
    num_cached_tokens: int = 0           # prompt tokens reused from the prefix cache
    ttft: float | None = None            # time to first token (seconds)
    latency: float | None = None         # arrival -> finished (seconds)
    logprobs: list[TokenLogprob] | None = None
    stop_reason: "str | int | None" = None
    outputs: list[CompletionOutput] = field(default_factory=list)   # all n samples


@dataclass
class StepOutput:
    request_id: int
    new_token_id: int
    finished: bool
    finish_reason: str | None = None
    text: str = ""                       # new text (only when sampling_params.detokenize)
    logprob: TokenLogprob | None = None
    stop_reason: "str | int | None" = None


@dataclass
class EngineStats:
    steps: int = 0
    prefill_steps: int = 0               # steps containing at least one prompt chunk
    decode_steps: int = 0                # decode-only steps
    mixed_steps: int = 0                 # steps with both decode tokens and prompt chunks
    cuda_graph_steps: int = 0
    prompt_tokens: int = 0               # prompt tokens admitted (including recomputation after preemption)
    prefill_tokens_computed: int = 0     # ... actually run through the model
    prefix_cache_hit_tokens: int = 0     # ... skipped thanks to the prefix cache
    decode_tokens: int = 0
    preemptions: int = 0
    requests_finished: int = 0
    requests_aborted: int = 0
    step_seconds: float = 0.0            # wall time spent inside step()


class LLM:
    """Example:
        llm = LLM("path/to/Qwen2.5-0.5B-Instruct")
        outs = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=32))
        print(outs[0].text)
    """

    def __init__(self, model: str, **kwargs):
        self.config = EngineConfig(model=model, **kwargs)
        self.device = pick_device(self.config.device)
        self.dtype = pick_dtype(self.config.dtype, self.device)
        self.model_config = ModelConfig.from_pretrained(model)
        if self.config.max_model_len > self.model_config.max_position_embeddings:
            logger.warning("max_model_len %d > model limit %d; clamping", self.config.max_model_len,
                           self.model_config.max_position_embeddings)
            self.config.max_model_len = self.model_config.max_position_embeddings
        self.tokenizer = self._load_tokenizer(model)
        self.runner = ModelRunner(self.config, self.model_config, self.device, self.dtype)
        self.scheduler = Scheduler(self.config, self.runner.num_blocks, self.model_config.eos_token_ids)
        self._seqs: dict[int, Sequence] = {}
        self.stats = EngineStats()

    @staticmethod
    def _load_tokenizer(path: str):
        if not os.path.exists(os.path.join(path, "tokenizer.json")) and \
                not os.path.exists(os.path.join(path, "tokenizer_config.json")):
            return None   # e.g. the tiny random test model: works with token ids only
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path)

    @property
    def attn_backend(self) -> str:
        return self.runner.attn_backend

    @property
    def uses_cuda_graph(self) -> bool:
        return bool(self.runner.graphs)

    # ------------------------------------------------------------------ request handling
    def encode(self, prompt: "str | list[int]") -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("this model folder has no tokenizer; pass token ids instead of text")
            return self.tokenizer.encode(prompt)
        return list(prompt)

    def apply_chat_template(self, messages: list[dict], **kwargs) -> list[int]:
        """Chat messages [{'role': 'user', 'content': '...'}] -> prompt token ids."""
        if self.tokenizer is None:
            raise ValueError("chat needs a tokenizer")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        # the template already contains every special token it needs
        return self.tokenizer.encode(text, add_special_tokens=False)

    def add_request(self, prompt: "str | list[int]", sampling_params: SamplingParams | None = None) -> int:
        """Queue ONE sequence (use generate() for n > 1)."""
        token_ids = self.encode(prompt)
        sp = sampling_params or SamplingParams()
        if not token_ids:
            raise ValueError("empty prompt")
        if len(token_ids) >= self.config.max_model_len:
            raise ValueError(f"prompt has {len(token_ids)} tokens; max_model_len is {self.config.max_model_len}")
        if sp.detokenize and self.tokenizer is None:
            raise ValueError("stop strings / detokenize need a tokenizer")
        seq = Sequence(token_ids, sp, self.config.block_size)
        seq.arrival_time = time.perf_counter()
        self._seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    def abort(self, request_id: int):
        seq = self.scheduler.abort(request_id)
        if seq is not None:
            seq.finish_time = time.perf_counter()
            self.stats.requests_aborted += 1

    def release(self, request_id: int):
        """Forget a finished request (generate() does this for you via get_output())."""
        self._seqs.pop(request_id, None)

    def has_unfinished_requests(self) -> bool:
        return not self.scheduler.is_finished()

    @property
    def num_running(self) -> int:
        return len(self.scheduler.running)

    @property
    def num_waiting(self) -> int:
        return len(self.scheduler.waiting)

    def step(self) -> list[StepOutput]:
        t0 = time.perf_counter()
        out = self.scheduler.schedule()
        st = self.stats
        st.preemptions = self.scheduler.num_preemptions
        if out.is_empty:                      # everything was just preempted; try again next step
            return []
        st.steps += 1
        if out.num_prefill and out.num_decode:
            st.mixed_steps += 1
        if out.num_prefill:
            st.prefill_steps += 1
        else:
            st.decode_steps += 1
        for s in out.seqs:
            if s.first_scheduled_time is None:
                s.first_scheduled_time = t0
            is_decode = (s.num_scheduled_tokens == 1 and s.num_completion_tokens > 0
                         and s.num_computed_tokens == s.num_tokens - 1)
            if is_decode:
                st.decode_tokens += 1
            else:
                st.prefill_tokens_computed += s.num_scheduled_tokens
        st.prompt_tokens = self.scheduler.num_admitted_tokens
        st.prefix_cache_hit_tokens = self.scheduler.num_prefix_hit_tokens
        sampling = out.sampling_seqs
        if out.num_prefill == 0 and self.runner._graph_key(out.seqs) is not None:
            st.cuda_graph_steps += 1
        token_ids, logprobs = self.runner.run(out, sampling)
        self.scheduler.postprocess(out, token_ids)
        now = time.perf_counter()
        outputs = []
        for i, (s, tok) in enumerate(zip(sampling, token_ids)):
            lp = logprobs[i] if logprobs is not None else None
            if lp is not None:
                s.logprobs.append(lp)
            if s.first_token_time is None:
                s.first_token_time = now
            s.last_token_time = now
            text = ""
            if s.sampling_params.detokenize:
                text = self._detokenize(s)
            if s.is_finished:
                s.finish_time = now
                st.requests_finished += 1
            outputs.append(StepOutput(s.seq_id, tok, s.is_finished, s.finish_reason, text, lp, s.stop_reason))
        st.step_seconds += time.perf_counter() - t0
        return outputs

    def _detokenize(self, s: Sequence) -> str:
        delta = detokenize_incrementally(self.tokenizer, s)
        s.output_text += delta
        if delta and s.sampling_params.stop and not isinstance(s.stop_reason, str):
            stop = check_stop_strings(s, len(delta))
            if stop is not None:
                s.stop_reason = stop
                if s.is_finished:              # also hit eos / length in this very step
                    s.finish_reason = "stop"
                else:
                    self.scheduler.finish(s, "stop")
        if s.is_finished and not isinstance(s.stop_reason, str):
            # flush: the final text is exactly the decode of all generated tokens
            s.output_text = self.tokenizer.decode(s.completion_token_ids,
                                                  skip_special_tokens=s.sampling_params.skip_special_tokens)
        return emittable_text(s, s.is_finished)

    def _completion(self, seq: Sequence, index: int) -> CompletionOutput:
        text = None
        if self.tokenizer is not None:
            if seq.sampling_params.detokenize and isinstance(seq.stop_reason, str):
                text = seq.output_text                  # cut before the stop string
            else:
                text = self.tokenizer.decode(seq.completion_token_ids,
                                             skip_special_tokens=seq.sampling_params.skip_special_tokens)
        return CompletionOutput(
            index=index, token_ids=seq.completion_token_ids, text=text, finish_reason=seq.finish_reason,
            stop_reason=seq.stop_reason, logprobs=seq.logprobs if seq.sampling_params.logprobs is not None else None,
            ttft=(seq.first_token_time - seq.arrival_time) if seq.first_token_time else None,
            latency=(seq.finish_time - seq.arrival_time) if seq.finish_time else None)

    def get_output(self, request_id: "int | list[int]", release: bool = True) -> RequestOutput:
        ids = request_id if isinstance(request_id, list) else [request_id]
        seqs = [self._seqs.pop(i) if release else self._seqs[i] for i in ids]
        comps = [self._completion(s, k) for k, s in enumerate(seqs)]
        first, c0 = seqs[0], comps[0]
        return RequestOutput(
            request_id=first.seq_id, prompt_token_ids=first.token_ids[:first.num_prompt_tokens],
            token_ids=c0.token_ids, text=c0.text, finish_reason=c0.finish_reason,
            num_cached_tokens=first.num_prefix_hit_tokens, ttft=c0.ttft, latency=c0.latency,
            logprobs=c0.logprobs, stop_reason=c0.stop_reason, outputs=comps)

    # ------------------------------------------------------------------ batch API
    def add_request_group(self, prompt: "str | list[int]", sp: SamplingParams) -> list[int]:
        """Queue sp.n independent samples of one prompt (seeds seed, seed+1, ... if a seed is set)."""
        if sp.n == 1:
            return [self.add_request(prompt, sp)]
        token_ids = self.encode(prompt)
        return [self.add_request(token_ids, sp.clone(n=1, seed=None if sp.seed is None else sp.seed + i))
                for i in range(sp.n)]

    def generate(self, prompts: "list[str] | list[list[int]] | str",
                 sampling_params: "SamplingParams | list[SamplingParams] | None" = None,
                 use_tqdm: bool = False) -> list[RequestOutput]:
        """Run all prompts to completion with continuous batching; outputs keep the input order."""
        if isinstance(prompts, str) or (prompts and isinstance(prompts[0], int)):
            prompts = [prompts]
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params or SamplingParams()] * len(prompts)
        if len(sampling_params) != len(prompts):
            raise ValueError("need one SamplingParams per prompt")
        groups = [self.add_request_group(p, sp) for p, sp in zip(prompts, sampling_params)]
        pending = {i for g in groups for i in g}
        bar = None
        if use_tqdm:
            from tqdm import tqdm
            bar = tqdm(total=len(pending), desc="generating")
        while pending:
            for out in self.step():
                if out.finished and out.request_id in pending:
                    pending.discard(out.request_id)
                    if bar:
                        bar.update(1)
        if bar:
            bar.close()
        return [self.get_output(g) for g in groups]

    def chat(self, conversations: "list[dict] | list[list[dict]]",
             sampling_params: SamplingParams | None = None) -> list[RequestOutput]:
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]
        return self.generate([self.apply_chat_template(c) for c in conversations], sampling_params)

    # ------------------------------------------------------------------ monitoring
    def metrics(self) -> dict:
        st, bm = self.stats, self.scheduler.block_manager
        return {
            "num_requests_running": self.num_running,
            "num_requests_waiting": self.num_waiting,
            "kv_cache_usage": bm.usage,
            "kv_cache_blocks_total": bm.num_blocks,
            "prefix_cache_hit_rate": (self.scheduler.num_prefix_hit_tokens / self.scheduler.num_admitted_tokens
                                      if self.scheduler.num_admitted_tokens else 0.0),
            **{k: v for k, v in vars(st).items()},
        }
