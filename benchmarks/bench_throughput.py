"""Benchmark nano-infer against a naive Hugging Face `model.generate()` baseline.

    python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct

Three scenarios (every engine runs in its own subprocess so only one model is in memory):
  1. static     : B requests, same prompt length, same output length (fair to HF's padded batching)
  2. mixed      : 32 requests with different output lengths. HF processes them in fixed batches
                  and must wait for the longest request; nano-infer uses continuous batching.
  3. prefix     : 16 requests that share a long system prompt; nano-infer with/without prefix caching.
Numbers are printed as a Markdown table and saved to benchmarks/results/<device>.json.
Never copy numbers from one machine to another: run this on YOUR hardware.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ----------------------------------------------------------------------------- workloads
def make_prompts(n, length, seed, vocab_lo=1000, vocab_hi=30000):
    import torch
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(vocab_lo, vocab_hi, (length,), generator=g).tolist() for _ in range(n)]


def mixed_output_lens(n, seed):
    import random
    rng = random.Random(seed)
    return [rng.randint(16, 256) for _ in range(n)]


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- nano-infer worker
def run_nano(args):
    from nanoinfer import LLM, SamplingParams
    llm = LLM(args.model, device=args.device, dtype=args.dtype, max_model_len=2048,
              max_num_seqs=max(64, max(args.batch_sizes)), use_cuda_graph=args.cuda_graph,
              enable_prefix_caching=not args.no_prefix)
    res = {"engine": "nano-infer", "device": str(llm.device), "dtype": str(llm.dtype).replace("torch.", ""),
           "kv_blocks": llm.runner.num_blocks, "static": [], "mixed": None, "prefix": None}
    # warm-up (first calls allocate memory / pick kernels)
    llm.generate(make_prompts(2, 32, 99), SamplingParams(temperature=0, max_tokens=8, ignore_eos=True))

    if "static" in args.scenarios:
        for b in args.batch_sizes:
            times, ttfts = [], []
            for r in range(args.repeats):
                # a different seed per repeat so the prefix cache can't help
                prompts = make_prompts(b, args.input_len, seed=1000 * b + r)
                sp = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
                sync(llm.device)
                t0 = time.perf_counter()
                outs = llm.generate(prompts, sp)
                sync(llm.device)
                times.append(time.perf_counter() - t0)
                ttfts.append(statistics.mean(o.ttft for o in outs))
                assert all(len(o.token_ids) == args.output_len for o in outs)
            t = statistics.median(times)
            res["static"].append({"batch": b, "seconds": t, "tok_per_s": b * args.output_len / t,
                                  "ttft_s": statistics.median(ttfts), "all_seconds": times})
            print(f"[nano] static B={b}: {b * args.output_len / t:.1f} tok/s", flush=True)

    if "mixed" in args.scenarios:
        n = 32
        prompts = make_prompts(n, args.input_len, seed=7)
        lens = mixed_output_lens(n, seed=7)
        sps = [SamplingParams(temperature=0, max_tokens=m, ignore_eos=True) for m in lens]
        llm.config.enable_prefix_caching = False
        sync(llm.device)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sps)
        sync(llm.device)
        t = time.perf_counter() - t0
        res["mixed"] = {"requests": n, "useful_tokens": sum(lens), "seconds": t, "tok_per_s": sum(lens) / t,
                        "mean_latency_s": statistics.mean(o.latency for o in outs)}
        print(f"[nano] mixed: {sum(lens) / t:.1f} tok/s", flush=True)

    if "prefix" in args.scenarios:
        res["prefix"] = bench_prefix(args)
    return res


def bench_prefix(args):
    """Shared 512-token system prompt + 32 unique tokens, 32 output tokens, 16 requests."""
    import gc
    from nanoinfer import LLM, SamplingParams
    system = make_prompts(1, 512, seed=42)[0]
    questions = make_prompts(16, 32, seed=43)
    prompts = [system + q for q in questions]
    sp = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    out = {}
    for enabled in (False, True):
        llm = LLM(args.model, device=args.device, dtype=args.dtype, max_model_len=2048, max_num_seqs=64,
                  enable_prefix_caching=enabled)
        llm.generate([system + [5]], SamplingParams(temperature=0, max_tokens=1))   # warm-up (+ fills the cache)
        before = llm.stats.prefill_tokens_computed
        sync(llm.device)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp)
        sync(llm.device)
        t = time.perf_counter() - t0
        key = "with_prefix_cache" if enabled else "without_prefix_cache"
        out[key] = {"seconds": t, "prefill_tokens_computed": llm.stats.prefill_tokens_computed - before,
                    "mean_ttft_s": statistics.mean(o.ttft for o in outs)}
        print(f"[nano] prefix cache={enabled}: {t:.2f}s", flush=True)
        del llm
        gc.collect()
    return out


# ----------------------------------------------------------------------------- HF baseline worker
def run_hf(args):
    import torch
    from transformers import AutoModelForCausalLM
    from nanoinfer.utils import pick_device, pick_dtype
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    res = {"engine": "hf-generate", "device": str(device), "dtype": str(dtype).replace("torch.", ""),
           "static": [], "mixed": None}

    def gen(prompts, max_new):
        # equal-length prompts -> no padding needed; force exactly max_new tokens
        x = torch.tensor(prompts, device=device)
        with torch.inference_mode():
            return model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                                  min_new_tokens=max_new, do_sample=False, repetition_penalty=1.0,
                                  temperature=None, top_p=None, top_k=None, pad_token_id=0)

    gen(make_prompts(2, 32, 99), 8)   # warm-up
    if "static" in args.scenarios:
        for b in args.batch_sizes:
            times, ttfts = [], []
            for r in range(args.repeats):
                prompts = make_prompts(b, args.input_len, seed=1000 * b + r)
                sync(device)
                t0 = time.perf_counter()
                gen(prompts, 1)                       # time to first token = prefill + 1 sample
                sync(device)
                ttfts.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                out = gen(prompts, args.output_len)
                sync(device)
                times.append(time.perf_counter() - t0)
                assert out.shape[1] == args.input_len + args.output_len
            t = statistics.median(times)
            res["static"].append({"batch": b, "seconds": t, "tok_per_s": b * args.output_len / t,
                                  "ttft_s": statistics.median(ttfts), "all_seconds": times})
            print(f"[hf] static B={b}: {b * args.output_len / t:.1f} tok/s", flush=True)

    if "mixed" in args.scenarios:
        n, bsz = 32, args.mixed_hf_batch
        prompts = make_prompts(n, args.input_len, seed=7)
        lens = mixed_output_lens(n, seed=7)
        latencies = []
        sync(device)
        t0 = time.perf_counter()
        for i in range(0, n, bsz):
            # static batching: the whole batch runs until its LONGEST request is done
            gen(prompts[i:i + bsz], max(lens[i:i + bsz]))
            sync(device)
            latencies += [time.perf_counter() - t0] * len(lens[i:i + bsz])
        t = time.perf_counter() - t0
        res["mixed"] = {"requests": n, "useful_tokens": sum(lens), "seconds": t, "tok_per_s": sum(lens) / t,
                        "mean_latency_s": statistics.mean(latencies), "hf_batch_size": bsz}
        print(f"[hf] mixed: {sum(lens) / t:.1f} tok/s", flush=True)
    return res


# ----------------------------------------------------------------------------- driver
def hardware_info():
    import torch
    info = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "cpu": platform.processor()}
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                info["cpu"] = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mixed-hf-batch", type=int, default=16)
    ap.add_argument("--scenarios", nargs="+", default=["static", "mixed", "prefix"])
    ap.add_argument("--engines", nargs="+", default=["hf", "nano"])
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--no-prefix", action="store_true", help="disable prefix caching in the static/mixed runs")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for PyTorch (default: all cores). On a busy machine, fewer threads than "
                         "cores is often much faster because all threads wait for the slowest one")
    ap.add_argument("--out", default=None)
    ap.add_argument("--_worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    if args._worker:
        res = run_nano(args) if args._worker == "nano" else run_hf(args)
        print("@@RESULT@@" + json.dumps(res))
        return

    results = {"hardware": hardware_info(), "config": {k: v for k, v in vars(args).items() if k != "_worker"},
               "engines": {}}
    for engine in args.engines:
        if engine == "hf":
            scen = [s for s in args.scenarios if s != "prefix"]
            if not scen:
                continue
        cmd = [sys.executable, os.path.abspath(__file__), "--_worker", engine] + sys.argv[1:]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        sys.stderr.write(proc.stderr[-2000:] if proc.returncode else "")
        line = next((l for l in proc.stdout.splitlines() if l.startswith("@@RESULT@@")), None)
        if line is None:
            raise RuntimeError(f"{engine} worker failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
        print("\n".join(l for l in proc.stdout.splitlines() if not l.startswith("@@RESULT@@")))
        results["engines"][engine] = json.loads(line[len("@@RESULT@@"):])

    device = results["engines"].get("nano", results["engines"].get("hf", {})).get("device", "run")
    suffix = f"-{args.threads}threads" if args.threads and device == "cpu" else ""
    out = args.out or os.path.join(ROOT, "benchmarks", "results", f"{device}{suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print_tables(results)
    print(f"\nsaved to {out}")


def print_tables(results):
    hw = results["hardware"]
    print(f"\nHardware: {hw.get('gpu') or hw['cpu']} | {hw['platform']} | torch {hw['torch']} | threads {hw['torch_threads']}")
    eng = results["engines"]
    nano, hf = eng.get("nano"), eng.get("hf")
    any_e = nano or hf
    cfg = results["config"]
    print(f"dtype: {any_e['dtype']}, input {cfg['input_len']} tokens, output {cfg['output_len']} tokens, "
          f"median of {cfg['repeats']} runs\n")
    if nano and nano["static"] or hf and hf["static"]:
        print("| batch | HF tok/s | nano tok/s | speed-up | HF batch time (s) | nano batch time (s) | HF TTFT (s) | nano TTFT (s) |")
        print("|---|---|---|---|---|---|---|---|")
        rows = {r["batch"]: r for r in (nano or {}).get("static", [])}
        hrows = {r["batch"]: r for r in (hf or {}).get("static", [])}
        for b in sorted(set(rows) | set(hrows)):
            n, h = rows.get(b), hrows.get(b)
            f = lambda r, k, fmt="{:.1f}": fmt.format(r[k]) if r else "-"
            sp = f"{n['tok_per_s'] / h['tok_per_s']:.2f}x" if n and h else "-"
            print(f"| {b} | {f(h, 'tok_per_s')} | {f(n, 'tok_per_s')} | {sp} | {f(h, 'seconds', '{:.2f}')} | "
                  f"{f(n, 'seconds', '{:.2f}')} | {f(h, 'ttft_s', '{:.3f}')} | {f(n, 'ttft_s', '{:.3f}')} |")
    if (nano and nano.get("mixed")) or (hf and hf.get("mixed")):
        print("\nMixed output lengths (32 requests, 16-256 new tokens each):\n")
        print("| engine | useful tok/s | total time (s) | mean request latency (s) |")
        print("|---|---|---|---|")
        for name, e in (("HF generate, static batches of %d" % cfg["mixed_hf_batch"], hf), ("nano-infer, continuous batching", nano)):
            if e and e.get("mixed"):
                m = e["mixed"]
                print(f"| {name} | {m['tok_per_s']:.1f} | {m['seconds']:.2f} | {m['mean_latency_s']:.2f} |")
    if nano and nano.get("prefix"):
        p = nano["prefix"]
        print("\nShared 512-token system prompt, 16 requests (32 own tokens + 32 output tokens each):\n")
        print("| prefix caching | total time (s) | prefill tokens computed | mean TTFT (s) |")
        print("|---|---|---|---|")
        for k, label in (("without_prefix_cache", "off"), ("with_prefix_cache", "on")):
            r = p[k]
            print(f"| {label} | {r['seconds']:.2f} | {r['prefill_tokens_computed']} | {r['mean_ttft_s']:.3f} |")


if __name__ == "__main__":
    main()
