"""Smallest possible example: batch generation with the offline API.

    python examples/basic_generate.py --model models/Qwen2.5-0.5B-Instruct
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoinfer import LLM, SamplingParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    llm = LLM(args.model, device=args.device, max_model_len=2048)
    print(f"device={llm.device} dtype={llm.dtype} kv_blocks={llm.runner.num_blocks}")

    questions = [
        "用三句话解释什么是大语言模型推理。",
        "What is a leveraged buyout? Explain in two sentences.",
        "Write a haiku about debugging.",
        "Give me three tips for a finance interview.",
    ]
    prompts = [llm.apply_chat_template([{"role": "user", "content": q}]) for q in questions]

    t0 = time.perf_counter()
    outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=128))
    dt = time.perf_counter() - t0
    for q, o in zip(questions, outs):
        print(f"\n### {q}\n{o.text}")
    n = sum(len(o.token_ids) for o in outs)
    print(f"\n{n} tokens in {dt:.2f}s -> {n / dt:.1f} tokens/s | stats: {llm.stats}")


if __name__ == "__main__":
    main()
