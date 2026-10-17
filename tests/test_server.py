"""OpenAI-style HTTP server: non-streaming, streaming (SSE), concurrency, validation.

Uses a tiny random model that shares the real Qwen tokenizer, so it is fast; the generated
text is gibberish, but the plumbing (batching, streaming, detokenisation) is exercised for real.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from nanoinfer import LLM
from nanoinfer.server.api_server import IncrementalDetokenizer, create_app


@pytest.fixture(scope="module")
def client(tiny_with_tokenizer):
    from fastapi.testclient import TestClient
    llm = LLM(str(tiny_with_tokenizer), dtype="float32", device="cpu", max_model_len=256)
    with TestClient(create_app(llm, model_name="tiny")) as c:
        c.llm = llm
        yield c


def test_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200 and r.json()["data"][0]["id"] == "tiny"


def test_completion_is_deterministic_with_greedy(client):
    body = {"prompt": "Hello, my name is", "max_tokens": 12, "temperature": 0}
    r = client.post("/v1/completions", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "text_completion"
    assert data["usage"]["completion_tokens"] == data["usage"]["total_tokens"] - data["usage"]["prompt_tokens"]
    assert data["usage"]["completion_tokens"] <= 12
    r2 = client.post("/v1/completions", json=body)
    assert r2.json()["choices"][0]["text"] == data["choices"][0]["text"]   # greedy is deterministic


def test_chat_streaming_equals_non_streaming(client):
    msgs = [{"role": "user", "content": "Tell me a joke"}]
    full = client.post("/v1/chat/completions", json={"messages": msgs, "max_tokens": 16, "temperature": 0}).json()
    text = full["choices"][0]["message"]["content"]

    chunks, done = [], False
    with client.stream("POST", "/v1/chat/completions",
                       json={"messages": msgs, "max_tokens": 16, "temperature": 0, "stream": True}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                done = True
                break
            chunks.append(json.loads(payload))
    assert done
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == text
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_concurrent_requests_are_batched(client):
    before = client.llm.stats.decode_tokens, client.llm.stats.decode_steps
    def call(i):
        return client.post("/v1/completions", json={"prompt": f"Request number {i}", "max_tokens": 20,
                                                    "temperature": 0.8, "top_p": 0.9}).status_code
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(call, range(8)))
    assert codes == [200] * 8
    tokens = client.llm.stats.decode_tokens - before[0]
    steps = client.llm.stats.decode_steps - before[1]
    assert tokens / steps > 1.0      # on average more than one sequence per decode step


def test_bad_requests(client):
    assert client.post("/v1/completions", json={"prompt": "x", "temperature": -1}).status_code == 400
    assert client.post("/v1/completions", json={"prompt": "hi " * 400}).status_code == 400


def test_incremental_detokenizer_handles_split_characters(client):
    tok = client.llm.tokenizer
    text = "你好，世界 🌍 nano-infer"
    detok = IncrementalDetokenizer(tok)
    assert "".join(detok.add(t) for t in tok.encode(text)) == text
