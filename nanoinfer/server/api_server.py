"""OpenAI-compatible HTTP server with token streaming.

    python -m nanoinfer serve --model models/Qwen2.5-0.5B-Instruct --port 8000

Endpoints:
  GET  /health                 liveness probe
  GET  /metrics                Prometheus text format (running/waiting requests, KV usage, token counters...)
  GET  /v1/models
  POST /v1/completions         prompt (text or token ids), stream, n, stop, logprobs, seed, penalties...
  POST /v1/chat/completions    messages, stream (+ stream_options.include_usage), n, stop, logprobs/top_logprobs...

Design: the engine is not thread-safe, so ONE background thread owns it and runs step() in a loop.
HTTP handlers (asyncio) talk to that thread through a queue ("inbox") and receive outputs back
through one asyncio.Queue per HTTP request. Many HTTP requests in flight = one continuous batch.
If a streaming client disconnects, its sequences are aborted and their KV blocks freed at once.
"""

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import AsyncIterator

from nanoinfer.engine.llm_engine import LLM, StepOutput
from nanoinfer.sampling_params import SamplingParams

logger = logging.getLogger(__name__)


class AsyncEngine:
    def __init__(self, llm: LLM):
        self.llm = llm
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue, int]] = {}
        self.num_http_requests = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._loop, name="nanoinfer-engine", daemon=True)
        self._thread.start()

    # ---------------- engine thread
    def _handle(self, cmd):
        kind = cmd[0]
        if kind == "add":
            _, token_ids, sps, loop, q, fut = cmd
            rids = []
            try:
                for sp in sps:
                    rids.append(self.llm.add_request(token_ids, sp))
            except Exception as e:  # e.g. prompt too long
                for rid in rids:
                    self.llm.abort(rid)
                    self.llm.release(rid)
                loop.call_soon_threadsafe(_set_exception, fut, e)
                return
            for index, rid in enumerate(rids):
                self._streams[rid] = (loop, q, index)
            loop.call_soon_threadsafe(_set_result, fut, rids)
        elif kind == "abort":
            for rid in cmd[1]:
                if rid in self._streams:
                    self.llm.abort(rid)
                    self.llm.release(rid)
                    self._streams.pop(rid, None)

    def _loop(self):
        while True:
            idle = not self.llm.has_unfinished_requests()
            try:
                self._handle(self._inbox.get(block=idle))   # sleep here when there is nothing to do
                while True:
                    self._handle(self._inbox.get_nowait())
            except queue.Empty:
                pass
            if not self.llm.has_unfinished_requests():
                continue
            try:
                outputs = self.llm.step()
            except Exception as e:   # report the failure to every waiting client, then reset
                logger.exception("engine step failed")
                self.errors += 1
                for rid, (loop, q, _) in list(self._streams.items()):
                    loop.call_soon_threadsafe(q.put_nowait, e)
                    self.llm.abort(rid)
                    self.llm.release(rid)
                self._streams.clear()
                continue
            for out in outputs:
                stream = self._streams.get(out.request_id)
                if stream is None:
                    continue
                loop, q, index = stream
                if out.finished:
                    self._streams.pop(out.request_id, None)
                    res = self.llm.get_output(out.request_id)   # releases the sequence
                    loop.call_soon_threadsafe(q.put_nowait, (index, out, res))
                else:
                    loop.call_soon_threadsafe(q.put_nowait, (index, out, None))

    # ---------------- asyncio side
    async def generate(self, token_ids: list[int], sps: list[SamplingParams]) -> AsyncIterator[tuple]:
        """Yields (choice index, StepOutput, RequestOutput or None) until all len(sps) sequences finish."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        fut = loop.create_future()
        self.num_http_requests += 1
        self._inbox.put(("add", token_ids, sps, loop, q, fut))
        rids = await fut
        unfinished = set(range(len(rids)))
        try:
            while unfinished:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
                if item[1].finished:
                    unfinished.discard(item[0])
                yield item
        finally:
            if unfinished:          # client went away: free the KV cache of what is still running
                self._inbox.put(("abort", [rids[i] for i in unfinished]))


def _set_result(fut, value):
    if not fut.done():
        fut.set_result(value)


def _set_exception(fut, exc):
    if not fut.done():
        fut.set_exception(exc)


class IncrementalDetokenizer:
    """Turn a growing list of token ids into text deltas (for scripts that call llm.step() directly).
    Uses the same O(1)-per-token sliding window as the engine (engine/detokenizer.py)."""

    def __init__(self, tokenizer, skip_special_tokens: bool = True):
        self.tokenizer = tokenizer
        self.skip = skip_special_tokens
        self.ids: list[int] = []
        self.prefix_offset = 0
        self.read_offset = 0
        self.text = ""

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        prefix = self.tokenizer.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=self.skip)
        new = self.tokenizer.decode(self.ids[self.prefix_offset:], skip_special_tokens=self.skip)
        if len(new) > len(prefix) and not new.endswith("\ufffd"):
            delta = new[len(prefix):]
            self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
            self.text += delta
            return delta
        return ""


def prometheus_text(llm: LLM, engine: AsyncEngine) -> str:
    lines = []
    m = llm.metrics()
    m["http_requests_total"] = engine.num_http_requests
    m["engine_errors_total"] = engine.errors
    for k, v in m.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        lines.append(f"nanoinfer_{k} {v}")
    return "\n".join(lines) + "\n"


def create_app(llm: LLM, model_name: str = "nano-infer"):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    from pydantic import BaseModel, Field

    if llm.tokenizer is None:
        raise ValueError("the server needs a model folder with a tokenizer")
    engine = AsyncEngine(llm)
    app = FastAPI(title="nano-infer")
    tok = llm.tokenizer

    class StreamOptions(BaseModel):
        include_usage: bool = False

    class CommonParams(BaseModel):
        model: str | None = None
        max_tokens: int | None = None            # None -> as many as fit in max_model_len
        max_completion_tokens: int | None = None
        min_tokens: int = 0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = 0
        min_p: float = 0.0
        n: int = Field(1, ge=1, le=16)
        stop: str | list[str] | None = None
        stop_token_ids: list[int] | None = None
        ignore_eos: bool = False
        seed: int | None = None
        presence_penalty: float = 0.0
        frequency_penalty: float = 0.0
        repetition_penalty: float = 1.0
        stream: bool = False
        stream_options: StreamOptions | None = None
        user: str | None = None

    class CompletionRequest(CommonParams):
        prompt: str | list[int]
        logprobs: int | None = None              # legacy completions API: number of top alternatives
        echo: bool = False

    class ChatMessage(BaseModel):
        role: str
        content: str | list[dict] | None = ""

    class ChatRequest(CommonParams):
        messages: list[ChatMessage]
        logprobs: bool = False
        top_logprobs: int | None = None
        chat_template_kwargs: dict | None = None

    def sampling_params(req, prompt_len: int, logprobs: int | None) -> list[SamplingParams]:
        room = llm.config.max_model_len - prompt_len
        max_tokens = req.max_completion_tokens or req.max_tokens or room
        try:
            sp = SamplingParams(
                temperature=req.temperature, top_p=req.top_p, top_k=req.top_k, min_p=req.min_p,
                max_tokens=max(1, min(max_tokens, room)), min_tokens=req.min_tokens,
                stop=req.stop or [], stop_token_ids=req.stop_token_ids or [], ignore_eos=req.ignore_eos,
                seed=req.seed, presence_penalty=req.presence_penalty, frequency_penalty=req.frequency_penalty,
                repetition_penalty=req.repetition_penalty, logprobs=logprobs, detokenize=True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return [sp] + [sp.clone(seed=None if sp.seed is None else sp.seed + i) for i in range(1, req.n)]

    def check_len(token_ids):
        if not token_ids:
            raise HTTPException(status_code=400, detail="empty prompt")
        if len(token_ids) >= llm.config.max_model_len:
            raise HTTPException(status_code=400, detail=f"prompt has {len(token_ids)} tokens; the limit "
                                                        f"(max_model_len) is {llm.config.max_model_len}")

    def tok_str(t: int) -> str:
        return tok.decode([t])

    def logprob_entry(lp, chat: bool):
        if chat:
            return {"token": tok_str(lp.token_id), "logprob": lp.logprob,
                    "bytes": list(tok_str(lp.token_id).encode("utf-8")),
                    "top_logprobs": [{"token": tok_str(t), "logprob": v, "bytes": list(tok_str(t).encode("utf-8"))}
                                     for t, v in lp.top]}
        return lp

    def legacy_logprobs(lps):
        return {"tokens": [tok_str(lp.token_id) for lp in lps], "token_logprobs": [lp.logprob for lp in lps],
                "top_logprobs": [{tok_str(t): v for t, v in lp.top} for lp in lps], "text_offset": []}

    def usage(prompt_len, completion, cached):
        return {"prompt_tokens": prompt_len, "completion_tokens": completion,
                "total_tokens": prompt_len + completion, "prompt_tokens_details": {"cached_tokens": cached}}

    async def run(req, token_ids, chat: bool, logprobs: int | None, prefix_text: str = ""):
        check_len(token_ids)
        sps = sampling_params(req, len(token_ids), logprobs)
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        gen = engine.generate(token_ids, sps)

        if req.stream:
            include_usage = bool(req.stream_options and req.stream_options.include_usage)
            try:
                first = await gen.__anext__()      # fail with a proper HTTP error if the prompt is rejected
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            def chunk(index, delta_text, finish_reason, lp=None, role=False):
                if chat:
                    delta = {"role": "assistant", "content": ""} if role else ({"content": delta_text} if delta_text else {})
                    choice = {"index": index, "delta": delta, "finish_reason": finish_reason}
                    if lp is not None:
                        choice["logprobs"] = {"content": [logprob_entry(lp, True)]}
                    body = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                            "choices": [choice]}
                else:
                    choice = {"index": index, "text": delta_text, "finish_reason": finish_reason}
                    if lp is not None:
                        choice["logprobs"] = legacy_logprobs([lp])
                    body = {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                            "choices": [choice]}
                if include_usage:
                    body["usage"] = None
                return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"

            async def sse():
                completion, cached = 0, 0
                if chat:
                    for i in range(len(sps)):
                        yield chunk(i, None, None, role=True)
                elif prefix_text:
                    for i in range(len(sps)):
                        yield chunk(i, prefix_text, None)
                pending = [first]
                try:
                    while True:
                        while pending:
                            index, out, res = pending.pop()
                            completion += 1
                            lp = out.logprob if logprobs is not None else None
                            if out.text or out.finished or lp is not None:
                                yield chunk(index, out.text, out.finish_reason if out.finished else None, lp)
                            if res is not None:
                                cached = res.num_cached_tokens
                        pending.append(await gen.__anext__())
                except StopAsyncIteration:
                    pass
                except Exception as e:      # engine failure mid-stream
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
                if include_usage:
                    body = {"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
                            "created": created, "model": model_name, "choices": [],
                            "usage": usage(len(token_ids), completion, cached)}
                    yield f"data: {json.dumps(body)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        results = {}
        try:
            async for index, out, res in gen:
                if res is not None:
                    results[index] = res
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        choices, completion = [], 0
        for i in range(len(sps)):
            res = results[i]
            completion += len(res.token_ids)
            text = prefix_text + res.text
            if chat:
                choice = {"index": i, "message": {"role": "assistant", "content": text},
                          "finish_reason": res.finish_reason}
                if logprobs is not None:
                    choice["logprobs"] = {"content": [logprob_entry(lp, True) for lp in res.logprobs]}
            else:
                choice = {"index": i, "text": text, "finish_reason": res.finish_reason}
                if logprobs is not None:
                    choice["logprobs"] = legacy_logprobs(res.logprobs)
            if isinstance(res.stop_reason, (str, int)):
                choice["stop_reason"] = res.stop_reason
            choices.append(choice)
        cached = results[0].num_cached_tokens
        return {"id": rid, "object": "chat.completion" if chat else "text_completion", "created": created,
                "model": model_name, "choices": choices, "usage": usage(len(token_ids), completion, cached)}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(prometheus_text(llm, engine))

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "nano-infer",
                                            "max_model_len": llm.config.max_model_len}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        token_ids = llm.encode(req.prompt)
        prefix = (req.prompt if isinstance(req.prompt, str) else tok.decode(token_ids)) if req.echo else ""
        return await run(req, token_ids, chat=False, logprobs=req.logprobs, prefix_text=prefix)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        messages = []
        for m in req.messages:
            content = m.content
            if isinstance(content, list):     # [{"type": "text", "text": "..."}, ...]
                content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
            messages.append({"role": m.role, "content": content or ""})
        try:
            token_ids = llm.apply_chat_template(messages, **(req.chat_template_kwargs or {}))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"chat template failed: {e}")
        logprobs = (req.top_logprobs or 0) if req.logprobs else None
        return await run(req, token_ids, chat=True, logprobs=logprobs)

    @app.exception_handler(HTTPException)
    async def http_error(_, exc: HTTPException):
        # OpenAI-style error body
        return JSONResponse(status_code=exc.status_code,
                            content={"error": {"message": exc.detail, "type": "invalid_request_error",
                                               "code": exc.status_code}})

    app.state.engine = engine
    return app


def main(argv=None):
    from nanoinfer.cli_args import add_engine_args, describe, engine_kwargs
    ap = argparse.ArgumentParser(description="nano-infer OpenAI-compatible server")
    add_engine_args(ap)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--served-model-name", default=None, help="name reported by /v1/models (default: folder name)")
    args = ap.parse_args(argv)

    import os

    import uvicorn
    logging.basicConfig(level=logging.INFO)
    llm = LLM(args.model, **engine_kwargs(args))
    print("loaded " + describe(llm))
    name = args.served_model_name or os.path.basename(os.path.normpath(args.model))
    uvicorn.run(create_app(llm, model_name=name), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
