"""Check that nano-infer computes the same thing as Hugging Face transformers for a given model.

    python scripts/verify_model.py --model models/Qwen2.5-1.5B-Instruct                 # bf16 on GPU
    python scripts/verify_model.py --model models/Qwen2.5-1.5B-Instruct --dtype float32 # exact check

Two measurements over a set of chat prompts (English + Chinese):
  1. greedy generation: are our tokens identical to HF's? (prefix length until the first difference)
  2. teacher-forced next-token agreement: we feed HF's greedy tokens to OUR engine one by one
     (real paged decode path, CUDA graphs, Triton kernels) and check that our argmax equals HF's at
     every position. In bfloat16 two correct implementations still disagree on near-ties now and
     then (different kernels round differently), and one flip changes everything after it in (1);
     (2) does not cascade, so it is the fairer bf16 metric. In float32, (1) should match exactly.
  3. noise floor: the same teacher-forced agreement between HF and HF itself with its other
     attention implementation (eager vs sdpa). Our agreement should be about as high as this.
"""

import argparse
import gc
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

PROMPTS = [
    "Give me a short introduction to large language models.",
    "What is 17 * 23? Answer briefly.",
    "Write a haiku about winter in Ithaca.",
    "用一句话解释什么是KV缓存。",
    "List three famous investment banks and one fact about each.",
    "Explain the difference between a DCF and an LBO model in two sentences.",
    "写一首关于秋天的四行诗。",
    "def quicksort(arr):",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [tok.encode(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                  add_generation_prompt=True), add_special_tokens=False)
               for p in PROMPTS]
    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    refs = []
    with torch.inference_mode():
        # plain greedy loop on the HF model (no generation_config: no repetition penalty, no EOS
        # suppression; EOS is just another token, like ignore_eos=True on our side)
        for p in prompts:
            ids = torch.tensor([p], device=args.device)
            out = hf(ids, use_cache=True)
            past, ref = out.past_key_values, []
            for _ in range(args.max_new):
                nxt = int(out.logits[0, -1].float().argmax())
                ref.append(nxt)
                out = hf(torch.tensor([[nxt]], device=args.device), past_key_values=past, use_cache=True)
                past = out.past_key_values
            refs.append(ref)
        # Noise floor: how often do two CORRECT implementations disagree at this precision?
        # Re-run HF itself with its other attention implementation (eager instead of sdpa),
        # teacher-forced on the same reference tokens.
        hf_agree = hf_total = 0
        try:
            hf.set_attn_implementation("eager")
            for p, ref in zip(prompts, refs):
                logits = hf(torch.tensor([p + ref[:-1]], device=args.device)).logits[0, len(p) - 1:]
                hf_agree += int((logits.float().argmax(-1).cpu() == torch.tensor(ref)).sum())
                hf_total += len(ref)
        except Exception as e:          # older transformers without set_attn_implementation
            print("noise-floor check skipped:", e)
    del hf
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    from nanoinfer import LLM, SamplingParams
    from nanoinfer.engine.sequence import Sequence
    llm = LLM(args.model, dtype=args.dtype, device=args.device, max_model_len=1024, max_num_seqs=16)
    # 1) free-running greedy
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=args.max_new, ignore_eos=True))
    prefix = []
    for o, r in zip(outs, refs):
        n = next((i for i, (a, b) in enumerate(zip(o.token_ids, r)) if a != b), len(r))
        prefix.append(n)
    # 2) teacher forcing through the engine: append HF's token instead of ours, record our argmax
    seqs = [Sequence(p, SamplingParams(temperature=0, max_tokens=args.max_new, ignore_eos=True), llm.config.block_size)
            for p in prompts]
    for s in seqs:
        llm.scheduler.add(s)
    agree = total = 0
    while llm.has_unfinished_requests():
        out = llm.scheduler.schedule()
        sampling = out.sampling_seqs
        logits = llm.runner.compute_logits(out)
        ours = logits.argmax(-1).tolist()
        forced = []
        for s, mine in zip(sampling, ours):
            want = refs[seqs.index(s)][s.num_completion_tokens]
            agree += mine == want
            total += 1
            forced.append(want)
        llm.scheduler.postprocess(out, forced)
    res = {"model": os.path.basename(os.path.normpath(args.model)), "dtype": args.dtype, "device": args.device,
           "attn_backend": llm.attn_backend, "cuda_graphs": len(llm.runner.graphs),
           "greedy_exact_matches": f"{sum(n == len(r) for n, r in zip(prefix, refs))}/{len(refs)}",
           "greedy_matching_prefix_tokens": prefix, "teacher_forced_top1_agreement": agree / total,
           "positions_checked": total,
           "hf_eager_vs_hf_sdpa_top1_agreement": hf_agree / hf_total if hf_total else None}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    chat = llm.chat([{"role": "user", "content": "What is 2 + 3? Reply with just the number."}],
                    SamplingParams(temperature=0, max_tokens=8))[0].text
    print("sanity:", repr(chat))
    res["sanity_2_plus_3"] = chat
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
