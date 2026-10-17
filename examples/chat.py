"""Interactive multi-turn chat in the terminal, printing tokens as they are generated.

    python examples/chat.py --model models/Qwen2.5-0.5B-Instruct

Each new turn re-sends the whole conversation, so the prefix cache kicks in: watch the
"reused N prompt tokens" line grow.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoinfer import LLM, SamplingParams  # noqa: E402
from nanoinfer.server.api_server import IncrementalDetokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    llm = LLM(args.model, device=args.device, max_model_len=4096)
    print(f"nano-infer chat on {llm.device} ({llm.dtype}). Type 'exit' to quit.\n")
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    sp = SamplingParams(temperature=args.temperature, top_p=0.9, max_tokens=512)
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in ("exit", "quit"):
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        rid = llm.add_request(llm.apply_chat_template(history), sp)
        detok = IncrementalDetokenizer(llm.tokenizer)
        print("bot> ", end="", flush=True)
        done = False
        while not done:
            for out in llm.step():
                if out.request_id == rid:
                    print(detok.add(out.new_token_id), end="", flush=True)
                    done = out.finished
        result = llm.get_output(rid)
        print(f"\n   [reused {result.num_cached_tokens} prompt tokens from the prefix cache]\n")
        history.append({"role": "assistant", "content": detok.text})


if __name__ == "__main__":
    main()
