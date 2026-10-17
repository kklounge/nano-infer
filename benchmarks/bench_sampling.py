"""What does a sampling configuration cost per decode step? Old sampler vs new, interleaved A/B.

    python benchmarks/bench_sampling.py --model models/Qwen2.5-0.5B-Instruct --batch 64

The other three benchmarks (`bench_throughput.py`, `bench_serving.py`, `profile_step.py`) all
hardcode `temperature=0`, so none of them touches the top-k / top-p / min-p filters, the
exponential-race draw, or the repetition / presence / frequency penalties. That makes them a useful
control group when the sampler changes -- but it also means the sampler's cost is invisible in every
table they produce. This script fills that hole: it measures one decode step under the sampling
configurations people actually use, and compares the current sampler against the pre-rewrite one.

Six configurations (one row per request, so a batch can mix them):
  1. greedy                        : temperature=0 -- the CONTROL. Goes through the `argmax` fast
                                     path and touches none of the rewritten code, so if this one
                                     moves, something other than the sampler changed.
  2. top_p=0.9, top_k=0            : nucleus only. With no top-k there is no bound on the candidate
                                     count, so the whole vocabulary has to be sorted either way.
  3. top_p=0.8, top_k=20           : Qwen2.5's own recommended sampling -- the case that matters.
  4. the same + repetition_penalty : adds the penalty path, which walks the token history.
  5. min_p=0.05                    : min-p compares against the maximum only, needs no ranking, and
                                     is done in place over the full row before and after.
  6. B-1 greedy + 1 sampling       : one sampling request must not drag a mostly-greedy batch into a
                                     full-vocabulary sort with it. (Skipped at --batch 1.)

Both arms run in ONE process against ONE loaded model, and the rounds interleave them (round 1: all
configurations on arm A, then all on arm B; round 2: the same again; ...), so background load that
drifts over minutes lands on both arms about equally. Per (arm, configuration) we report the BEST
round -- the least-contended observation -- plus the spread over rounds, so contention stays visible
instead of being averaged in. On a machine with busy background apps the absolute milliseconds run
high; the before/after RATIO is what reproduces. Never copy numbers from one machine to another: run
this on YOUR hardware.

Numbers are printed as two Markdown tables (the whole decode step, then the sampler call on its own)
and saved to benchmarks/results/<device>-sampling-<model size>-b<batch>.json.

Why the old arm comes from a commit and not from HEAD: the candidate-window rewrite is *committed*
(`fbfcde5`), so `git show HEAD:nanoinfer/engine/sampler.py` hands back the NEW sampler and the A/B
would silently compare the new sampler with itself. The last commit that still has the old
implementation is `4aaf1f1`, which this script extracts by default -- no manual setup needed. The
check is `grep -c _candidate_window`: 0 means it really is the old one (5 at HEAD). Only the sampler
differs between the arms -- model, KV cache, CUDA graphs and scheduler are shared, and the graphs
cover the model forward only (the sampler runs outside them), so hot-swapping `runner.sampler` is
sound. Where the commit is not reachable (shallow clone, source tarball, no git) the old arm is
dropped with a warning and the new arm is measured on its own.
"""

import argparse
import importlib.util
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from nanoinfer import LLM, SamplingParams  # noqa: E402

# The last commit before the candidate-window rewrite (fbfcde5). See the module docstring.
OLD_SAMPLER_REV = "4aaf1f1"
SAMPLER_PATH_IN_REPO = "nanoinfer/engine/sampler.py"


# ----------------------------------------------------------------------------- the configurations
def build_configs(batch, max_tok):
    """(name, description, [SamplingParams per row]) -- one entry per row so a batch can mix them.

    max_tok has to cover every step a round takes (warm-up + the timed pass + the phase pass), or a
    row would finish mid-measurement and shrink the batch under us."""
    base = dict(max_tokens=max_tok, ignore_eos=True)
    greedy = dict(temperature=0.0)
    qwen = dict(temperature=0.7, top_p=0.8, top_k=20)           # Qwen2.5's own recommendation
    out = [
        ("greedy", "temperature=0 (control: touches none of the changed code)",
         [SamplingParams(**base, **greedy)] * batch),
        ("top_p=0.9, top_k=0", "nucleus only -- top_k off, so the whole vocabulary must be sorted",
         [SamplingParams(**base, temperature=0.7, top_p=0.9, top_k=0)] * batch),
        ("top_p=0.8, top_k=20", "Qwen2.5's recommended sampling",
         [SamplingParams(**base, **qwen)] * batch),
        ("top_p=0.8, top_k=20, rep=1.05", "the same plus a repetition penalty over the token history",
         [SamplingParams(**base, repetition_penalty=1.05, **qwen)] * batch),
        ("min_p=0.05", "min-p only -- no ranking filter, so it works in place over the full row",
         [SamplingParams(**base, temperature=0.7, min_p=0.05)] * batch),
    ]
    if batch > 1:
        # the mixed batch: the point is that ONE sampling row must not drag the greedy ones into a
        # full-vocabulary sort with it
        rows = [SamplingParams(**base, **greedy) for _ in range(batch - 1)]
        rows.append(SamplingParams(**base, **qwen))
        out.append((f"{batch - 1} greedy + 1 sampling", "a mostly-greedy batch with one sampling row", rows))
    return out


# ----------------------------------------------------------------------------- arms
def extract_old_sampler(rev, tmpdir):
    """`git show <rev>:nanoinfer/engine/sampler.py` into a temp file; None if that is not possible.

    Returns None (with a printed reason) rather than raising: a source tarball or a shallow clone
    has no `4aaf1f1`, and measuring the new arm alone is still useful there."""
    try:
        blob = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{SAMPLER_PATH_IN_REPO}"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as e:                                 # git not installed
        print(f"[old arm] git unavailable ({e}); measuring the new sampler only.", flush=True)
        return None
    if blob.returncode != 0:
        print(f"[old arm] cannot read {rev}:{SAMPLER_PATH_IN_REPO} "
              f"({blob.stderr.strip().splitlines()[:1]}); measuring the new sampler only.", flush=True)
        return None
    path = os.path.join(tmpdir, f"old_sampler_{rev}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(blob.stdout)
    return path


def load_old_sampler_class(path):
    """Import the extracted file as its own module and hand back its Sampler class.

    The old file imports only from `nanoinfer.*` and torch, both already importable, so loading a
    second copy of the module is harmless -- it just defines a second Sampler class."""
    spec = importlib.util.spec_from_file_location("old_sampler_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["old_sampler_mod"] = mod
    spec.loader.exec_module(mod)
    return mod.Sampler


def make_arm_samplers(llm, old_path, seed):
    """{"new": the engine's own sampler, "old": one built from `old_path`} -- "old" only if we have it.

    The old Sampler takes the same constructor arguments as the current one
    (device, vocab_size, eos_token_ids, seed), which is why it can simply be assigned to
    `runner.sampler`; if that ever stops being true, this is where it breaks loudly."""
    cur = llm.runner.sampler
    arms = {"new": cur}
    if old_path:
        with open(old_path, encoding="utf-8") as f:
            src = f.read()
        # the sanity check from the docstring: the old implementation has no candidate window
        if "_candidate_window" in src:
            print(f"[old arm] WARNING: {old_path} already contains _candidate_window, i.e. it is a "
                  f"POST-rewrite sampler -- the A/B would compare the new sampler with itself.", flush=True)
        Old = load_old_sampler_class(old_path)
        arms["old"] = Old(cur.device, cur.vocab_size, cur.eos_token_ids, seed)
    return arms


# ----------------------------------------------------------------------------- measurement
def prompts_for(batch, prompt_len, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1000, 30000, (prompt_len,), generator=g).tolist() for _ in range(batch)]


def setup_rows(llm, sps, prompt_len, seed):
    """Add one request per SamplingParams and drive the prefill to completion.

    Timing must not include prefill steps, so we step until every row has produced its first token."""
    ids = [llm.add_request(p, sp) for p, sp in zip(prompts_for(len(sps), prompt_len, seed), sps)]
    guard = 0
    while any(llm._seqs[i].num_completion_tokens == 0 for i in ids):
        llm.step()
        guard += 1
        if guard > 10_000:
            raise RuntimeError("prefill did not finish")
    return ids


def teardown(llm, ids):
    """Drop the rows so the next configuration starts from an empty scheduler and a free KV cache."""
    for i in ids:
        llm.abort(i)
        llm.release(i)


def time_steps(llm, steps, warmup):
    """Median ms per decode step, synchronising only around whole steps.

    One sync per step, not per phase: syncing inside the step would serialise CPU bookkeeping
    against GPU work and inflate every number (that is what `profile_step.py` is for)."""
    sync = torch.cuda.synchronize if llm.device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        llm.step()
    out = []
    for _ in range(steps):
        sync()
        t0 = time.perf_counter()
        llm.step()
        sync()
        out.append((time.perf_counter() - t0) * 1000)
    return statistics.median(out)


def time_sampler_phase(llm, ids, steps):
    """Just the sampler call, in isolation: median ms.

    This needs a sync immediately before and after the sampler, which slows the surrounding step
    down -- hence a separate pass from `time_steps` above, instead of one pass reporting both. It
    reimplements the body of `LLM.step()` (schedule -> forward -> logits -> sample -> postprocess)
    because the sampler call has to be bracketed from the outside."""
    runner, sch = llm.runner, llm.scheduler
    sync = torch.cuda.synchronize if llm.device.type == "cuda" else (lambda: None)
    out = []
    for _ in range(steps):
        sched = sch.schedule()
        sampling = sched.sampling_seqs
        with torch.inference_mode():
            if sched.num_prefill == 0 and runner._graph_key(sched.seqs) is not None:
                hidden = runner._replay_graph(sched.seqs)
            else:
                input_ids, positions, meta, lidx = runner.prepare(sched)
                hidden = runner.model(input_ids, positions, runner.kv_caches, meta)[lidx]
            logits = runner.model.compute_logits(hidden)
            sync()
            t0 = time.perf_counter()
            toks, _ = runner.sampler(logits, sampling)
            sync()
            out.append((time.perf_counter() - t0) * 1000)
        sch.postprocess(sched, toks)
    return statistics.median(out)


def run(args, old_path, old_desc):
    llm = LLM(args.model, device=args.device, dtype=args.dtype, max_model_len=args.max_model_len,
              max_num_seqs=max(64, args.batch), enforce_eager=args.enforce_eager,
              attn_backend=args.attn_backend)
    arms = make_arm_samplers(llm, old_path, args.seed)
    # warm-up + timed pass + (optional) phase pass, with room to spare
    configs = build_configs(args.batch, args.warmup + args.steps * (2 if args.phases else 1) + 64)
    print(f"device={llm.device} dtype={llm.dtype} backend={llm.attn_backend} "
          f"cuda_graphs={len(llm.runner.graphs)} batch={args.batch} prompt_len={args.prompt_len}", flush=True)
    print(f"arms: {', '.join(arms)}  rounds={args.rounds} steps={args.steps} warmup={args.warmup}\n", flush=True)

    # step[arm][config] = list of per-round medians
    step = {a: {c[0]: [] for c in configs} for a in arms}
    phase = {a: {c[0]: [] for c in configs} for a in arms}
    hist_len = {}

    for rnd in range(args.rounds):
        for arm_name, sampler in arms.items():
            llm.runner.sampler = sampler                 # the only thing that differs between arms
            for name, _desc, sps in configs:
                ids = setup_rows(llm, sps, args.prompt_len, seed=1000 * rnd + 7)
                try:
                    step[arm_name][name].append(time_steps(llm, args.steps, args.warmup))
                    # record how long the token history is -- the new penalty path scales with it
                    hist_len[name] = max(len(llm._seqs[i].token_ids) for i in ids)
                    if args.phases:
                        phase[arm_name][name].append(time_sampler_phase(llm, ids, args.steps))
                finally:
                    teardown(llm, ids)
            print(f"  round {rnd + 1}/{args.rounds} {arm_name}: " +
                  "  ".join(f"{n}={step[arm_name][n][-1]:.2f}" for n, _, _ in configs), flush=True)
    llm.runner.sampler = arms["new"]                     # leave the engine as we found it

    res = {
        "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "platform": platform.platform(), "torch": torch.__version__,
                     "python": platform.python_version(), "torch_threads": torch.get_num_threads()},
        "config": {k: v for k, v in vars(args).items()} | {"old_arm": old_desc},
        "engine": {"attn_backend": llm.attn_backend, "cuda_graphs": len(llm.runner.graphs),
                   "dtype": str(llm.dtype).replace("torch.", ""), "kv_blocks": llm.runner.num_blocks,
                   "chunked_prefill": llm.config.enable_chunked_prefill},
        "descriptions": {n: d for n, d, _ in configs},
        "history_tokens": hist_len,
        "runs": {},
    }
    for arm in arms:
        res["runs"][arm] = {}
        for name, _d, _s in configs:
            xs = step[arm][name]
            e = {"step_ms_best": min(xs), "step_ms_median": statistics.median(xs),
                 "step_ms_worst": max(xs), "rounds": xs}
            if args.phases and phase[arm][name]:
                p = phase[arm][name]
                e |= {"sampler_ms_best": min(p), "sampler_ms_median": statistics.median(p), "sampler_rounds": p}
            res["runs"][arm][name] = e
    return res, configs


def print_table(res, configs, args):
    runs = res["runs"]
    has_old = "old" in runs
    b = args.batch
    print(f"\n=== best-of-{args.rounds} decode step, batch {b}, "
          f"{res['engine']['dtype']}, {res['hardware']['gpu']} ===\n")
    head = "| sampling configuration | " + ("old sampler (ms) | new sampler (ms) | change | " if has_old
                                            else "step (ms) | ") + "tok/s | spread of rounds (new) |"
    print(head)
    print("|---" * (6 if has_old else 4) + "|")
    for name, _d, _s in configs:
        n = runs["new"][name]
        new_ms = n["step_ms_best"]
        spread = f"{n['step_ms_best']:.2f}-{n['step_ms_worst']:.2f}"
        toks = b / new_ms * 1000
        if has_old:
            o = runs["old"][name]["step_ms_best"]
            chg = f"{(new_ms / o - 1) * 100:+.0f}%" if o else "-"
            print(f"| {name} | {o:.2f} | **{new_ms:.2f}** | {chg} | {toks:.0f} | {spread} |")
        else:
            print(f"| {name} | **{new_ms:.2f}** | {toks:.0f} | {spread} |")
    if args.phases:
        print(f"\nsampler phase alone (median ms, measured with a sync around the sampler call only):\n")
        print("| sampling configuration | " + ("old | new | change |" if has_old else "sampler (ms) |"))
        print("|---" * (4 if has_old else 2) + "|")
        for name, _d, _s in configs:
            n = runs["new"][name].get("sampler_ms_best")
            if n is None:
                continue
            if has_old:
                o = runs["old"][name].get("sampler_ms_best")
                chg = f"{(n / o - 1) * 100:+.0f}%" if o else "-"
                print(f"| {name} | {o:.3f} | **{n:.3f}** | {chg} |")
            else:
                print(f"| {name} | **{n:.3f}** |")
    print(f"\ntoken history at measurement time: {res['history_tokens']}")


# ----------------------------------------------------------------------------- default output path
def slug(text):
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def default_out(args):
    """benchmarks/results/<device>-sampling-<model size>-b<batch>.json

    Same idea as `bench_throughput.py`'s default: name the file after the hardware it was measured
    on, so results from two machines cannot overwrite each other. "rtx5070-sampling-0.5b-b64.json"
    on this card; the model part is the size in the model's directory name (0.5b, 1.5b, 3b) and
    falls back to the whole name when there is no size in it."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    # "NVIDIA GeForce RTX 5070" -> "rtx5070" (the vendor words say nothing; the model number does)
    device = re.sub(r"[^a-z0-9]", "", gpu.lower().replace("nvidia", "").replace("geforce", "")) if gpu else "cpu"
    name = os.path.basename(os.path.normpath(args.model))
    m = re.search(r"\d+(?:\.\d+)?b", name, re.I)
    return os.path.join(ROOT, "benchmarks", "results",
                        f"{device}-sampling-{slug(m.group(0) if m else name)}-b{args.batch}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=64,
                    help="the sampler's work is [batch, vocab], so the batch size is the point")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--steps", type=int, default=60, help="timed decode steps per round")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=5, help="N for best-of-N; arms interleave per round")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--old-sampler", default=None,
                    help=f"path to a pre-rewrite sampler.py; default: extracted from {OLD_SAMPLER_REV}")
    ap.add_argument("--old-sampler-rev", default=OLD_SAMPLER_REV,
                    help="commit to take the old arm from (must predate the rewrite in fbfcde5)")
    ap.add_argument("--no-old", action="store_true", help="measure the current sampler only, no A/B")
    ap.add_argument("--attn-backend", default="auto", choices=["auto", "triton", "sdpa", "naive"])
    ap.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs")
    ap.add_argument("--no-phases", dest="phases", action="store_false",
                    help="skip the extra pass that times the sampler call on its own")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory(prefix="nanoinfer-bench-sampling-") as tmpdir:
        if args.no_old:
            old_path, old_desc = None, None
        elif args.old_sampler:
            old_path, old_desc = args.old_sampler, args.old_sampler
        else:
            # the temp copy is gone by the time the JSON is written, so record the commit, not the path
            old_path = extract_old_sampler(args.old_sampler_rev, tmpdir)
            old_desc = f"git {args.old_sampler_rev}:{SAMPLER_PATH_IN_REPO}" if old_path else None
        res, configs = run(args, old_path, old_desc)
    print_table(res, configs, args)

    out = args.out or default_out(args)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
