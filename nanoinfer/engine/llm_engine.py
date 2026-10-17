"""The user-facing engine: LLM.generate(), plus the step() loop that ties everything together.

One engine step =
    scheduler.schedule()   -> which sequences run now (prefill batch or decode batch)
    model_runner.run()     -> one forward pass + sampling -> one new token per sequence
    scheduler.postprocess()-> append tokens, finish sequences, free their KV blocks
"""

import logging
import os
import time
from dataclasses import dataclass

from nanoinfer.config import EngineConfig, ModelConfig
from nanoinfer.engine.model_runner import ModelRunner
from nanoinfer.engine.scheduler import Scheduler
from nanoinfer.engine.sequence import Sequence
from nanoinfer.sampling_params import SamplingParams
from nanoinfer.utils import pick_device, pick_dtype

logger = logging.getLogger(__name__)


@dataclass
class RequestOutput:
    request_id: int
    prompt_token_ids: list[int]
    token_ids: list[int]                 # generated tokens only
    text: str | None                     # decoded text (None if no tokenizer is available)
    finish_reason: str | None
    num_cached_tokens: int = 0           # prompt tokens reused from the prefix cache
    ttft: float | None = None            # time to first token (seconds)
    latency: float | None = None         # arrival -> finished (seconds)


@dataclass
class StepOutput:
    request_id: int
    new_token_id: int
    finished: bool
    finish_reason: str | None = None


@dataclass
class EngineStats:
    steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    prompt_tokens: int = 0               # prompt tokens admitted (including recomputation after preemption)
    prefill_tokens_computed: int = 0     # ... actually run through the model
    prefix_cache_hit_tokens: int = 0     # ... skipped thanks to the prefix cache
    decode_tokens: int = 0
    preemptions: int = 0


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

    # ------------------------------------------------------------------ request handling
    def encode(self, prompt: "str | list[int]") -> list[int]:
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("this model folder has no tokenizer; pass token ids instead of text")
            return self.tokenizer.encode(prompt)
        return list(prompt)

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        """Chat messages [{'role': 'user', 'content': '...'}] -> prompt token ids."""
        if self.tokenizer is None:
            raise ValueError("chat needs a tokenizer")
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # the template already contains every special token it needs
        return self.tokenizer.encode(text, add_special_tokens=False)

    def add_request(self, prompt: "str | list[int]", sampling_params: SamplingParams | None = None) -> int:
        token_ids = self.encode(prompt)
        sp = sampling_params or SamplingParams()
        if not token_ids:
            raise ValueError("empty prompt")
        if len(token_ids) >= self.config.max_model_len:
            raise ValueError(f"prompt has {len(token_ids)} tokens; max_model_len is {self.config.max_model_len}")
        seq = Sequence(token_ids, sp, self.config.block_size)
        seq.arrival_time = time.perf_counter()
        self._seqs[seq.seq_id] = seq
        self.scheduler.add(seq)
        return seq.seq_id

    def abort(self, request_id: int):
        seq = self.scheduler.abort(request_id)
        if seq is not None:
            seq.finish_time = time.perf_counter()

    def release(self, request_id: int):
        """Forget a finished request (generate() does this for you via get_output())."""
        self._seqs.pop(request_id, None)

    def has_unfinished_requests(self) -> bool:
        return not self.scheduler.is_finished()

    def step(self) -> list[StepOutput]:
        seqs, is_prefill = self.scheduler.schedule()
        st = self.stats
        st.steps += 1
        if is_prefill:
            st.prefill_steps += 1
            for s in seqs:
                st.prompt_tokens += s.num_tokens
                st.prefill_tokens_computed += s.num_tokens - s.num_cached_tokens
                st.prefix_cache_hit_tokens += s.num_cached_tokens
                if s.num_preemptions == 0:
                    s.num_prefix_hit_tokens = s.num_cached_tokens
        else:
            st.decode_steps += 1
            st.decode_tokens += len(seqs)
        token_ids = self.runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        st.preemptions = self.scheduler.num_preemptions
        now = time.perf_counter()
        outputs = []
        for s, tok in zip(seqs, token_ids):
            if s.first_token_time is None:
                s.first_token_time = now
            if s.is_finished:
                s.finish_time = now
            outputs.append(StepOutput(s.seq_id, tok, s.is_finished, s.finish_reason))
        return outputs

    def get_output(self, request_id: int, release: bool = True) -> RequestOutput:
        seq = self._seqs.pop(request_id) if release else self._seqs[request_id]
        text = None
        if self.tokenizer is not None:
            text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True)
        return RequestOutput(
            request_id=seq.seq_id,
            prompt_token_ids=seq.token_ids[:seq.num_prompt_tokens],
            token_ids=seq.completion_token_ids,
            text=text,
            finish_reason=seq.finish_reason,
            num_cached_tokens=seq.num_prefix_hit_tokens,
            ttft=(seq.first_token_time - seq.arrival_time) if seq.first_token_time else None,
            latency=(seq.finish_time - seq.arrival_time) if seq.finish_time else None,
        )

    # ------------------------------------------------------------------ batch API
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
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, sampling_params)]
        pending = set(ids)
        bar = None
        if use_tqdm:
            from tqdm import tqdm
            bar = tqdm(total=len(ids), desc="generating")
        while pending:
            for out in self.step():
                if out.finished and out.request_id in pending:
                    pending.discard(out.request_id)
                    if bar:
                        bar.update(1)
        if bar:
            bar.close()
        return [self.get_output(i) for i in ids]

    def chat(self, conversations: "list[dict] | list[list[dict]]",
             sampling_params: SamplingParams | None = None) -> list[RequestOutput]:
        if conversations and isinstance(conversations[0], dict):
            conversations = [conversations]
        return self.generate([self.apply_chat_template(c) for c in conversations], sampling_params)
