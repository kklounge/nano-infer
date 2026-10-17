"""Shared command-line options for every entry point (server, chat, generate, benchmarks)."""

import argparse


def add_engine_args(ap: argparse.ArgumentParser, default_model: str | None = None):
    g = ap.add_argument_group("engine")
    g.add_argument("--model", required=default_model is None, default=default_model,
                   help="local model folder (config.json + *.safetensors + tokenizer)")
    g.add_argument("--device", default="auto", help="auto | cuda | cpu")
    g.add_argument("--dtype", default="auto", help="auto | bfloat16 | float16 | float32")
    g.add_argument("--max-model-len", type=int, default=4096, help="max prompt + output tokens per request")
    g.add_argument("--max-num-seqs", type=int, default=64, help="max requests in one batch")
    g.add_argument("--max-num-batched-tokens", type=int, default=None, help="token budget per step")
    g.add_argument("--block-size", type=int, default=16)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    g.add_argument("--attn-backend", default="auto", choices=["auto", "triton", "sdpa", "naive"])
    g.add_argument("--no-prefix-caching", action="store_true")
    g.add_argument("--no-chunked-prefill", action="store_true")
    g.add_argument("--no-fused-ops", action="store_true", help="plain PyTorch element-wise ops")
    g.add_argument("--enforce-eager", action="store_true", help="never use CUDA graphs")
    g.add_argument("--cuda-graph", action="store_true", help=argparse.SUPPRESS)   # old flag; graphs are default now
    g.add_argument("--seed", type=int, default=0)
    return ap


def engine_kwargs(args) -> dict:
    kw = dict(device=args.device, dtype=args.dtype, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, block_size=args.block_size,
              gpu_memory_utilization=args.gpu_memory_utilization, attn_backend=args.attn_backend,
              enable_prefix_caching=not args.no_prefix_caching,
              enable_chunked_prefill=not args.no_chunked_prefill, enforce_eager=args.enforce_eager,
              seed=args.seed)
    if args.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.no_fused_ops:
        kw["use_fused_ops"] = False
    return kw


def describe(llm) -> str:
    r = llm.runner
    graphs = f"{len(r.graphs)} CUDA graphs" if r.graphs else "eager"
    return (f"{llm.model_config.model_type} on {llm.device} ({str(llm.dtype).replace('torch.', '')}), "
            f"attention={r.attn_backend}, fused_ops={r.use_fused_ops}, {graphs}, "
            f"KV cache {r.num_blocks} blocks x {llm.config.block_size} tokens, "
            f"chunked_prefill={llm.config.enable_chunked_prefill}, prefix_caching={llm.config.enable_prefix_caching}")
