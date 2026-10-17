"""OpenAI-compatible HTTP server with token streaming.

    python -m nanoinfer.server.api_server --model models/Qwen2.5-0.5B-Instruct --port 8000

Endpoints: GET /v1/models, POST /v1/completions, POST /v1/chat/completions (stream=true supported).

Design: the engine is not thread-safe, so ONE background thread owns it and runs step() in a loop.
HTTP handlers (asyncio) talk to that thread through a queue ("inbox") and receive tokens back
through one asyncio.Queue per request. Many HTTP requests in flight = one continuous batch.
"""

import argparse
import asyncio
import json
import queue
import threading
import time
import uuid
from typing import AsyncIterator

from nanoinfer.engine.llm_engine import LLM, StepOutput
from nanoinfer.sampling_params import SamplingParams


class AsyncEngine:
    def __init__(self, llm: LLM):
        self.llm = llm
        self._inbox: queue.Queue = queue.Queue()
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._thread = threading.Thread(target=self._loop, name="nanoinfer-engine", daemon=True)
        self._thread.start()

    # ---------------- engine thread
    def _handle(self, cmd):
        kind = cmd[0]
        if kind == "add":
            _, token_ids, sp, loop, q, fut = cmd
            try:
                rid = self.llm.add_request(token_ids, sp)
            except Exception as e:  # e.g. prompt too long
                loop.call_soon_threadsafe(_set_exception, fut, e)
                return
            self._streams[rid] = (loop, q)
            loop.call_soon_threadsafe(_set_result, fut, rid)
        elif kind == "abort":
            rid = cmd[1]
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
                for rid, (loop, q) in list(self._streams.items()):
                    loop.call_soon_threadsafe(q.put_nowait, e)
                    self.llm.abort(rid)
                    self.llm.release(rid)
                self._streams.clear()
                continue
            for out in outputs:
                stream = self._streams.get(out.request_id)
                if stream is None:
                    continue
                loop, q = stream
                loop.call_soon_threadsafe(q.put_nowait, out)
                if out.finished:
                    self._streams.pop(out.request_id, None)
                    self.llm.release(out.request_id)

    # ---------------- asyncio side
    async def generate(self, token_ids: list[int], sp: SamplingParams) -> AsyncIterator[StepOutput]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        fut = loop.create_future()
        self._inbox.put(("add", token_ids, sp, loop, q, fut))
        rid = await fut
        finished = False
        try:
            while not finished:
                item = await q.get()
                if isinstance(item, Exception):
                    raise item
                finished = item.finished
                yield item
        finally:
            if not finished:          # client went away: free its KV cache
                self._inbox.put(("abort", rid))


def _set_result(fut, value):
    if not fut.done():
        fut.set_result(value)


def _set_exception(fut, exc):
    if not fut.done():
        fut.set_exception(exc)


class IncrementalDetokenizer:
    """Turn a growing list of token ids into text deltas. A multi-byte character (e.g. Chinese or emoji)
    can be split across tokens; while the text ends with the replacement char we wait for more tokens."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.text = ""

    def add(self, token_id: int) -> str:
        self.ids.append(token_id)
        full = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if full.endswith("\ufffd"):
            return ""
        delta = full[len(self.text):] if full.startswith(self.text) else full
        self.text = full
        return delta


def create_app(llm: LLM, model_name: str = "nano-infer"):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    if llm.tokenizer is None:
        raise ValueError("the server needs a model folder with a tokenizer")
    engine = AsyncEngine(llm)
    app = FastAPI(title="nano-infer")

    class CompletionRequest(BaseModel):
        model: str | None = None
        prompt: str
        max_tokens: int = 128
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = 0
        stream: bool = False

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str | None = None
        messages: list[ChatMessage]
        max_tokens: int = 128
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = 0
        stream: bool = False

    def params(req) -> SamplingParams:
        try:
            return SamplingParams(temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                                  max_tokens=req.max_tokens)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def run(token_ids, sp, chat: bool, stream: bool):
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"

        def chunk(delta: str | None, finish_reason, first=False):
            if chat:
                d = {"role": "assistant", "content": ""} if first else ({"content": delta} if delta else {})
                choice = {"index": 0, "delta": d, "finish_reason": finish_reason}
                return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                        "choices": [choice]}
            return {"id": rid, "object": "text_completion", "created": created, "model": model_name,
                    "choices": [{"index": 0, "text": delta or "", "finish_reason": finish_reason}]}

        detok = IncrementalDetokenizer(llm.tokenizer)
        gen = engine.generate(token_ids, sp)
        if stream:
            async def sse():
                if chat:
                    yield f"data: {json.dumps(chunk(None, None, first=True))}\n\n"
                async for out in gen:
                    delta = detok.add(out.new_token_id)
                    if delta or out.finished:
                        yield f"data: {json.dumps(chunk(delta, out.finish_reason if out.finished else None), ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        try:
            finish_reason, n = None, 0
            async for out in gen:
                detok.add(out.new_token_id)
                n += 1
                finish_reason = out.finish_reason
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        usage = {"prompt_tokens": len(token_ids), "completion_tokens": n, "total_tokens": len(token_ids) + n}
        if chat:
            choice = {"index": 0, "message": {"role": "assistant", "content": detok.text}, "finish_reason": finish_reason}
        else:
            choice = {"index": 0, "text": detok.text, "finish_reason": finish_reason}
        return {"id": rid, "object": obj, "created": created, "model": model_name, "choices": [choice], "usage": usage}

    def check_len(token_ids):
        if len(token_ids) >= llm.config.max_model_len:
            raise HTTPException(status_code=400, detail=f"prompt too long ({len(token_ids)} tokens)")

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "nano-infer"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        token_ids = llm.encode(req.prompt)
        check_len(token_ids)
        return await run(token_ids, params(req), chat=False, stream=req.stream)

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        token_ids = llm.apply_chat_template([m.model_dump() for m in req.messages])
        check_len(token_ids)
        return await run(token_ids, params(req), chat=True, stream=req.stream)

    app.state.engine = engine
    return app


def main():
    ap = argparse.ArgumentParser(description="nano-infer OpenAI-compatible server")
    ap.add_argument("--model", required=True, help="local model folder (config.json + safetensors + tokenizer)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--cuda-graph", action="store_true", help="experimental: capture CUDA graphs for decode")
    args = ap.parse_args()

    import uvicorn
    llm = LLM(args.model, device=args.device, dtype=args.dtype, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, enable_prefix_caching=not args.no_prefix_caching,
              use_cuda_graph=args.cuda_graph)
    print(f"loaded {args.model} on {llm.device} ({llm.dtype}), KV cache: {llm.runner.num_blocks} blocks")
    uvicorn.run(create_app(llm), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
