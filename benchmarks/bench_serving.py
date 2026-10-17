"""Online serving benchmark: latency percentiles under load (TTFT, TPOT, ITL, end-to-end).

    # starts its own server (same engine flags as `python -m nanoinfer serve`) and measures it:
    python benchmarks/bench_serving.py --model models/Qwen2.5-0.5B-Instruct --num-requests 200 --request-rate 20
    # or measure a server that is already running:
    python benchmarks/bench_serving.py --url http://127.0.0.1:8000 --num-requests 200 --request-rate 20
    # compare engine configurations (one server each), e.g. chunked prefill on vs off:
    python benchmarks/bench_serving.py --model ... --configs="default;--no-chunked-prefill"

Workload: synthetic, seeded. Each request has a random prompt of `--input-len` tokens (uniform in
[input_len * (1 - range_ratio), input_len]) sent as token ids, and must generate exactly a random
number of tokens in the same way (ignore_eos). Requests arrive as a Poisson process with
`--request-rate` requests/s (inf = all at once). Every request streams, so we see each token arrive.

Metrics (per request, then mean / median / p90 / p99 over requests):
  TTFT  time to first token      (queueing + prefill)
  TPOT  time per output token    (end-to-end - TTFT) / (output tokens - 1)
  ITL   inter-token latency      gap between consecutive streamed chunks (all gaps of all requests)
  E2E   end-to-end latency
"""

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def percentile(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def make_workload(args):
    rng = random.Random(args.seed)
    reqs = []
    for _ in range(args.num_requests):
        n_in = rng.randint(max(1, int(args.input_len * (1 - args.range_ratio))), args.input_len)
        n_out = rng.randint(max(1, int(args.output_len * (1 - args.range_ratio))), args.output_len)
        prompt = [rng.randint(1000, 30000) for _ in range(n_in)]
        reqs.append((prompt, n_out))
    if args.shared_prefix_len:
        prefix = [rng.randint(1000, 30000) for _ in range(args.shared_prefix_len)]
        reqs = [(prefix + p, n) for p, n in reqs]
    return reqs


async def one_request(client, url, prompt, n_out, model):
    body = {"model": model, "prompt": prompt, "max_tokens": n_out, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    times, n_tokens, error = [], 0, None
    try:
        async with client.stream("POST", url + "/v1/completions", json=body) as r:
            if r.status_code != 200:
                return {"error": f"HTTP {r.status_code}: {(await r.aread())[:200]!r}"}
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                data = json.loads(line[6:])
                if data.get("usage"):
                    n_tokens = data["usage"]["completion_tokens"]
                if data.get("choices"):
                    times.append(time.perf_counter())
    except Exception as e:     # noqa: BLE001
        error = repr(e)
    end = time.perf_counter()
    if error or not times:
        return {"error": error or "no tokens"}
    ttft = times[0] - t0
    return {"ttft": ttft, "e2e": end - t0, "itl": [b - a for a, b in zip(times, times[1:])],
            "tpot": (times[-1] - times[0]) / max(n_tokens - 1, 1), "output_tokens": n_tokens,
            "prompt_tokens": len(prompt)}


async def run_load(args, url):
    import httpx
    reqs = make_workload(args)
    rng = random.Random(args.seed + 1)
    limits = httpx.Limits(max_connections=args.max_concurrency or 1000, max_keepalive_connections=100)
    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=limits) as client:
        model = (await client.get(url + "/v1/models")).json()["data"][0]["id"]
        # warm-up so first-request costs (kernel compilation, allocator growth) are not measured
        await asyncio.gather(*[one_request(client, url, p, 8, model) for p, _ in reqs[:4]])
        sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None

        async def limited(p, n):
            if sem is None:
                return await one_request(client, url, p, n, model)
            async with sem:
                return await one_request(client, url, p, n, model)

        tasks = []
        t_start = time.perf_counter()
        for prompt, n_out in reqs:
            tasks.append(asyncio.create_task(limited(prompt, n_out)))
            if args.request_rate != float("inf"):
                await asyncio.sleep(rng.expovariate(args.request_rate))
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - t_start
        # server-side view: how long engine steps took (decode-only vs steps containing prefill)
        steps = {}
        try:
            for line in (await client.get(url + "/metrics")).text.splitlines():
                name, _, value = line.partition(" ")
                if "_step_ms_" in name or name.endswith(("_preemptions", "_mixed_steps", "_steps")):
                    steps[name.removeprefix("nanoinfer_")] = float(value)
        except Exception:      # noqa: BLE001  (not a nano-infer server)
            pass
    return results, duration, steps


def summarize(results, duration, steps=None):
    ok = [r for r in results if "error" not in r]
    errors = [r["error"] for r in results if "error" in r]
    out_tok = sum(r["output_tokens"] for r in ok)
    s = {"completed": len(ok), "failed": len(errors), "duration_s": duration,
         "request_throughput": len(ok) / duration, "output_throughput": out_tok / duration,
         "total_token_throughput": (out_tok + sum(r["prompt_tokens"] for r in ok)) / duration}
    for name in ("ttft", "tpot", "e2e"):
        xs = [r[name] * 1000 for r in ok]
        s[name + "_ms"] = {"mean": statistics.mean(xs) if xs else float("nan"), "p50": percentile(xs, 50),
                           "p90": percentile(xs, 90), "p99": percentile(xs, 99)}
    itl = [x * 1000 for r in ok for x in r["itl"]]
    s["itl_ms"] = {"mean": statistics.mean(itl) if itl else float("nan"), "p50": percentile(itl, 50),
                   "p90": percentile(itl, 90), "p99": percentile(itl, 99), "max": max(itl) if itl else float("nan")}
    if steps:
        s["server_steps"] = steps
    if errors:
        s["first_errors"] = errors[:3]
    return s


def print_summary(s, title):
    print(f"\n=== {title} ===")
    print(f"completed {s['completed']}  failed {s['failed']}  in {s['duration_s']:.1f}s | "
          f"{s['request_throughput']:.2f} req/s | output {s['output_throughput']:.0f} tok/s | "
          f"total {s['total_token_throughput']:.0f} tok/s")
    print("| metric (ms) | mean | p50 | p90 | p99 |")
    print("|---|---|---|---|---|")
    for k in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
        m = s[k]
        print(f"| {k[:-3].upper()} | {m['mean']:.1f} | {m['p50']:.1f} | {m['p90']:.1f} | {m['p99']:.1f} |")
    st = s.get("server_steps")
    if st:
        print(f"server: {st.get('steps', 0):.0f} steps ({st.get('mixed_steps', 0):.0f} mixed), "
              f"decode-only step p50/p99/max {st.get('decode_step_ms_p50', 0):.1f}/"
              f"{st.get('decode_step_ms_p99', 0):.1f}/{st.get('decode_step_ms_max', 0):.1f} ms, "
              f"steps with prefill p50/p99/max {st.get('with_prefill_step_ms_p50', 0):.1f}/"
              f"{st.get('with_prefill_step_ms_p99', 0):.1f}/{st.get('with_prefill_step_ms_max', 0):.1f} ms, "
              f"preemptions {st.get('preemptions', 0):.0f}")
    if s.get("first_errors"):
        print("errors:", s["first_errors"])


def start_server(args, extra):
    import httpx
    port = args.port
    cmd = [sys.executable, "-m", "nanoinfer", "serve", "--model", args.model, "--port", str(port)] + extra
    # server output goes to a file: an unread pipe would eventually fill up and block the server
    log = tempfile.NamedTemporaryFile("w+", suffix="-nanoinfer-server.log", delete=False, encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    t0 = time.time()
    while time.time() - t0 < 600:
        if proc.poll() is not None:
            log.seek(0)
            raise RuntimeError("server exited:\n" + log.read()[-3000:])
        try:
            if httpx.get(url + "/health", timeout=1).status_code == 200:
                return proc, url
        except Exception:      # noqa: BLE001
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not start in time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="existing server; otherwise one is started with --model")
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--request-rate", type=float, default=float("inf"), help="requests per second (Poisson)")
    ap.add_argument("--max-concurrency", type=int, default=None)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--range-ratio", type=float, default=0.5, help="lengths uniform in [len*(1-r), len]")
    ap.add_argument("--shared-prefix-len", type=int, default=0, help="common prefix for every prompt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--configs", default=None,
                    help='compare server configs separated by ";" (each starts a server), e.g. '
                         '--configs="default;--no-chunked-prefill"')
    ap.add_argument("--out", default=None, help="save results as JSON")
    args, extra = ap.parse_known_args()

    runs = {}
    if args.url:
        results, duration, steps = asyncio.run(run_load(args, args.url))
        runs["server"] = summarize(results, duration, steps)
        print_summary(runs["server"], args.url)
    else:
        if not args.model:
            ap.error("give --url or --model")
        configs = [c.strip() for c in args.configs.split(";")] if args.configs else [" ".join(extra) or "default"]
        for cfg in configs:
            flags = [] if cfg == "default" else cfg.split() + extra
            if cfg == "default":
                flags = list(extra)
            proc, url = start_server(args, flags)
            try:
                results, duration, steps = asyncio.run(run_load(args, url))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            name = cfg
            runs[name] = summarize(results, duration, steps)
            print_summary(runs[name], name)
    if args.out:
        workload = {k: v for k, v in vars(args).items() if k not in ("url", "configs", "out")}
        workload["request_rate"] = str(args.request_rate)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"workload": workload, "runs": runs}, f, indent=2)
        print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
