"""Command-line interface.

    python -m nanoinfer serve    --model models/Qwen2.5-0.5B-Instruct --port 8000   # OpenAI-compatible server
    python -m nanoinfer chat     --model models/Qwen2.5-0.5B-Instruct               # chat in the terminal
    python -m nanoinfer generate --model models/Qwen2.5-0.5B-Instruct --prompt "Hello" --max-tokens 64
    python -m nanoinfer info     --model models/Qwen2.5-0.5B-Instruct               # what the engine picked

(After `pip install -e .` the same commands are available as `nano-infer serve ...`.)
"""

import argparse
import sys
import time

DEFAULT_MODEL = "models/Qwen2.5-0.5B-Instruct"


def _sampling_args(ap):
    g = ap.add_argument_group("sampling")
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.9)
    g.add_argument("--top-k", type=int, default=0)
    g.add_argument("--min-p", type=float, default=0.0)
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--stop", action="append", default=[], help="stop string (repeatable)")
    g.add_argument("--sample-seed", type=int, default=None, help="per-request seed (reproducible sampling)")


def _sp(args, **kw):
    from nanoinfer import SamplingParams
    return SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
                          repetition_penalty=args.repetition_penalty, max_tokens=args.max_tokens, stop=args.stop,
                          seed=args.sample_seed, **kw)


def _load(args):
    from nanoinfer import LLM
    from nanoinfer.cli_args import describe, engine_kwargs
    t0 = time.perf_counter()
    llm = LLM(args.model, **engine_kwargs(args))
    print(f"[nano-infer] {describe(llm)} | loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return llm


def cmd_chat(args):
    llm = _load(args)
    history = [{"role": "system", "content": args.system}] if args.system else []
    print("Type your message ('/reset' clears the history, 'exit' quits).\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in ("exit", "quit"):
            break
        if user == "/reset":
            history = history[:1] if args.system else []
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        rid = llm.add_request(llm.apply_chat_template(history), _sp(args, detokenize=True))
        print("bot> ", end="", flush=True)
        t0 = time.perf_counter()
        done = False
        while not done:
            for out in llm.step():
                if out.request_id == rid:
                    print(out.text, end="", flush=True)
                    done = out.finished
        res = llm.get_output(rid)
        dt = time.perf_counter() - t0
        print(f"\n   [{len(res.token_ids)} tokens, {len(res.token_ids) / dt:.1f} tok/s, first token "
              f"{res.ttft * 1000:.0f} ms, {res.num_cached_tokens} prompt tokens from the prefix cache]\n")
        history.append({"role": "assistant", "content": res.text})


def cmd_generate(args):
    llm = _load(args)
    prompts = args.prompt or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    if not args.raw:
        prompts = [llm.apply_chat_template([{"role": "user", "content": p}]) for p in prompts]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, _sp(args, n=args.n))
    dt = time.perf_counter() - t0
    for i, o in enumerate(outs):
        for c in o.outputs:
            tag = f"[{i}]" if len(o.outputs) == 1 else f"[{i}.{c.index}]"
            print(f"{tag} {c.text}\n")
    n = sum(len(c.token_ids) for o in outs for c in o.outputs)
    print(f"[nano-infer] {n} tokens in {dt:.2f}s ({n / dt:.1f} tok/s)", file=sys.stderr)


def cmd_info(args):
    llm = _load(args)
    mc = llm.model_config
    print(f"layers={mc.num_hidden_layers} hidden={mc.hidden_size} heads={mc.num_attention_heads} "
          f"kv_heads={mc.num_key_value_heads} head_dim={mc.head_dim} vocab={mc.vocab_size}")
    print(f"KV cache: {llm.runner.num_blocks} blocks = {llm.runner.num_blocks * llm.config.block_size} tokens "
          f"({llm.runner.num_blocks * llm.runner.block_bytes() / 1024 ** 3:.2f} GiB)")
    if llm.runner.graphs:
        print(f"CUDA graphs: batch sizes {llm.runner.graph_batch_sizes}, context buckets "
              f"{llm.runner.graph_ctx_buckets}, captured in {llm.runner.graph_capture_seconds:.2f}s")


def main(argv=None):
    from nanoinfer.cli_args import add_engine_args
    ap = argparse.ArgumentParser(prog="nano-infer", description="nano-infer: a small, fast LLM inference engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="OpenAI-compatible HTTP server", add_help=False)
    p = sub.add_parser("chat", help="interactive chat in the terminal")
    add_engine_args(p, DEFAULT_MODEL)
    _sampling_args(p)
    p.add_argument("--system", default="You are a helpful assistant.")
    p = sub.add_parser("generate", help="generate for prompts given as arguments or on stdin")
    add_engine_args(p, DEFAULT_MODEL)
    _sampling_args(p)
    p.add_argument("--prompt", action="append", help="prompt (repeatable); default: one per stdin line")
    p.add_argument("--raw", action="store_true", help="do not apply the chat template")
    p.add_argument("-n", type=int, default=1, help="samples per prompt")
    p = sub.add_parser("info", help="load the model and print what the engine chose")
    add_engine_args(p, DEFAULT_MODEL)

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "serve":
        from nanoinfer.server.api_server import main as serve
        return serve(argv[1:])
    args = ap.parse_args(argv)
    {"chat": cmd_chat, "generate": cmd_generate, "info": cmd_info}[args.cmd](args)


if __name__ == "__main__":
    main()
