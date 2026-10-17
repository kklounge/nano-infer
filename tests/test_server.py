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


def _sse(client, path, body):
    chunks = []
    with client.stream("POST", path, json={**body, "stream": True}) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    return chunks
                chunks.append(json.loads(payload))
    raise AssertionError("stream ended without [DONE]")


def test_health_and_metrics(client):
    assert client.get("/health").json() == {"status": "ok"}
    client.post("/v1/completions", json={"prompt": "metrics please", "max_tokens": 3})
    text = client.get("/metrics").text
    values = dict(line.split(" ") for line in text.strip().splitlines())
    assert float(values["nanoinfer_requests_finished"]) >= 1
    assert "nanoinfer_kv_cache_usage" in values and "nanoinfer_num_requests_running" in values


def test_n_samples_and_seed(client):
    body = {"prompt": "Once upon a time", "max_tokens": 10, "temperature": 1.0, "n": 3, "seed": 123,
            "ignore_eos": True}
    a = client.post("/v1/completions", json=body).json()
    b = client.post("/v1/completions", json=body).json()
    assert [c["index"] for c in a["choices"]] == [0, 1, 2]
    assert [c["text"] for c in a["choices"]] == [c["text"] for c in b["choices"]]     # seeded -> reproducible
    assert len({c["text"] for c in a["choices"]}) > 1                               # different samples
    assert a["usage"]["completion_tokens"] == 30
    streamed = _sse(client, "/v1/completions", body)
    for i in range(3):
        text = "".join(c["choices"][0]["text"] for c in streamed if c["choices"] and c["choices"][0]["index"] == i)
        assert text == a["choices"][i]["text"]


def test_stop_string_and_usage_in_stream(client):
    base = {"prompt": "The quick brown fox", "max_tokens": 24, "temperature": 0, "ignore_eos": True}
    full = client.post("/v1/completions", json=base).json()["choices"][0]["text"]
    stop = full[len(full) // 2: len(full) // 2 + 2]
    assert stop.strip()
    cut = client.post("/v1/completions", json={**base, "stop": [stop]}).json()["choices"][0]
    assert cut["finish_reason"] == "stop" and cut["stop_reason"] == stop
    assert cut["text"] == full[:full.index(stop)]
    chunks = _sse(client, "/v1/completions", {**base, "stop": [stop], "stream_options": {"include_usage": True}})
    assert "".join(c["choices"][0]["text"] for c in chunks if c["choices"]) == cut["text"]
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0


def test_chat_logprobs(client):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}], "max_tokens": 5,
            "temperature": 0, "logprobs": True, "top_logprobs": 2}
    r = client.post("/v1/chat/completions", json=body).json()
    content = r["choices"][0]["logprobs"]["content"]
    assert len(content) == r["usage"]["completion_tokens"]
    for entry in content:
        assert entry["logprob"] <= 0 and len(entry["top_logprobs"]) == 2
        assert entry["top_logprobs"][0]["logprob"] >= entry["top_logprobs"][1]["logprob"]
        assert entry["top_logprobs"][0]["logprob"] == entry["logprob"]      # greedy picks the top token


def test_prompt_as_token_ids_and_echo(client):
    ids = client.llm.tokenizer.encode("Hello world")
    r = client.post("/v1/completions", json={"prompt": ids, "max_tokens": 3, "temperature": 0, "echo": True}).json()
    assert r["choices"][0]["text"].startswith("Hello world")
    assert r["usage"]["prompt_tokens"] == len(ids)


def test_error_body_is_openai_style(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "top_p": 0})
    assert r.status_code == 400 and "top_p" in r.json()["error"]["message"]


def test_client_disconnect_aborts_request(tiny_with_tokenizer):
    """Real uvicorn server + real HTTP client: hanging up mid-stream must abort the request and free
    its KV-cache blocks (the TestClient cannot simulate a disconnect faithfully)."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    llm = LLM(str(tiny_with_tokenizer), dtype="float32", device="cpu", max_model_len=2048)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(llm), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        body = {"prompt": "Write a very long story", "max_tokens": 2000, "temperature": 0, "ignore_eos": True,
                "stream": True}
        with httpx.stream("POST", f"http://127.0.0.1:{port}/v1/completions", json=body, timeout=30) as r:
            for i, _ in enumerate(r.iter_lines()):
                if i > 5:
                    break                   # hang up after a few tokens
        for _ in range(200):
            if not llm.has_unfinished_requests():
                break
            time.sleep(0.05)
        assert not llm.has_unfinished_requests()
        assert llm.stats.requests_aborted == 1
        assert llm.stats.decode_tokens < 1500               # it did not run to the end
        assert llm.scheduler.block_manager.usage == 0.0
    finally:
        server.should_exit = True
        thread.join(timeout=10)
