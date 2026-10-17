"""Where does the time of one decode step go? (CPU bookkeeping vs GPU work)

    python benchmarks/profile_step.py --model models/Qwen2.5-0.5B-Instruct --batch 1 16

For each phase of LLM.step() it reports the average wall time, synchronising the GPU after every
phase so the numbers add up (this makes the whole step a little slower than in normal runs, where
CPU work of a phase can overlap with GPU work of the previous one).
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from nanoinfer import LLM, SamplingParams  # noqa: E402


def profile(llm: LLM, batch: int, steps: int, prompt_len: int):
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(1000, 30000, (prompt_len,), generator=g).tolist() for _ in range(batch)]
    sp = SamplingParams(temperature=0, max_tokens=steps + 20, ignore_eos=True)
    ids = [llm.add_request(p, sp) for p in prompts]
    while any(llm._seqs[i].num_completion_tokens == 0 for i in ids):
        llm.step()                                       # finish the prefill first
    sync = torch.cuda.synchronize if llm.device.type == "cuda" else (lambda: None)
    phases = {k: [] for k in ("schedule", "prepare+forward", "lm_head", "sample", "postprocess", "total")}
    sch, runner = llm.scheduler, llm.runner
    for _ in range(steps):
        sync()
        t0 = time.perf_counter()
        out = sch.schedule()
        t1 = time.perf_counter()
        sampling = out.sampling_seqs
        key = runner._graph_key(out.seqs) if out.num_prefill == 0 else None
        with torch.inference_mode():
            if key is not None:
                hidden = runner._replay_graph(out.seqs)
            else:
                input_ids, positions, meta, lidx = runner.prepare(out)
                hidden = runner.model(input_ids, positions, runner.kv_caches, meta)[lidx]
            sync()
            t2 = time.perf_counter()
            logits = runner.model.compute_logits(hidden)
            sync()
            t3 = time.perf_counter()
            toks, _ = runner.sampler(logits, sampling)
            t4 = time.perf_counter()
        sch.postprocess(out, toks)
        t5 = time.perf_counter()
        for k, v in zip(phases, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)):
            phases[k].append(v * 1000)
    for i in ids:
        llm.abort(i)
        llm.release(i)
    return {k: statistics.median(v) for k, v in phases.items()}, key is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--attn-backend", default="auto")
    ap.add_argument("--no-fused-ops", action="store_true")
    args = ap.parse_args()
    llm = LLM(args.model, max_model_len=2048, enforce_eager=args.enforce_eager, attn_backend=args.attn_backend,
              use_fused_ops=False if args.no_fused_ops else None)
    print(f"device={llm.device} backend={llm.attn_backend} fused_ops={llm.runner.use_fused_ops} "
          f"cuda_graphs={len(llm.runner.graphs)}")
    for b in args.batch:
        res, graph = profile(llm, b, args.steps, args.prompt_len)
        print(f"batch {b:3d} ({'graph' if graph else 'eager'}): " +
              "  ".join(f"{k} {v:.3f}ms" for k, v in res.items()) + f"  -> {1000 / res['total'] * b:.0f} tok/s")


if __name__ == "__main__":
    main()
