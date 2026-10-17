"""ModelRunner: owns the model weights and the KV cache, and runs one forward pass per step.

It translates the scheduler's batch (Python objects) into flat tensors the model understands:
  input_ids / positions  - which tokens to compute and where they sit in their sequence
  slot_mapping           - where to write each new token's K/V in the paged cache
  block_tables           - which pages each sequence owns (to read old K/V back)
All integer inputs of a step are packed into ONE pinned CPU tensor and copied to the GPU with a
single asynchronous transfer (each separate small copy costs ~10 us of launch overhead).

Decode-only steps can replay a captured CUDA graph (see the bottom of this file).
"""

import logging
import math
import time

import torch

from nanoinfer.config import EngineConfig, ModelConfig
from nanoinfer.engine.sampler import Sampler
from nanoinfer.engine.scheduler import SchedulerOutput
from nanoinfer.engine.sequence import Sequence
from nanoinfer.model.attention import AttentionMetadata
from nanoinfer.model.loader import load_model
from nanoinfer.model.triton_attention import HAS_TRITON, triton_usable
from nanoinfer.utils import dtype_size

logger = logging.getLogger(__name__)


def resolve_attn_backend(requested: str, device: torch.device) -> str:
    if requested == "auto":
        return "triton" if (device.type == "cuda" and HAS_TRITON) else "sdpa"
    if requested == "triton" and not triton_usable(device):
        raise ValueError("attn_backend='triton' needs an NVIDIA GPU and the triton package "
                         "(pip install triton, or triton-windows on Windows)")
    return requested


class ModelRunner:
    def __init__(self, config: EngineConfig, model_config: ModelConfig, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.model_config = model_config
        self.device = device
        self.dtype = dtype
        self.block_size = config.block_size
        self.attn_backend = resolve_attn_backend(config.attn_backend, device)
        self.pin = device.type == "cuda"
        torch.manual_seed(config.seed)
        self.model = load_model(config.model, device, dtype, max_positions=config.max_model_len, cfg=model_config)

        self.num_blocks = self._decide_num_blocks()
        mc = model_config
        # One tensor for the whole cache: [K/V, layer, block, slot-in-block, kv_head, head_dim].
        # One extra "scratch" block at the end absorbs writes from padding rows (CUDA graphs) and warm-up.
        self.kv_cache = torch.zeros(2, mc.num_hidden_layers, self.num_blocks + 1, self.block_size,
                                    mc.num_key_value_heads, mc.head_dim, device=device, dtype=dtype)
        self.kv_caches = [(self.kv_cache[0, i], self.kv_cache[1, i]) for i in range(mc.num_hidden_layers)]
        self.scratch_block = self.num_blocks
        self.scratch_slot = self.num_blocks * self.block_size
        self.max_blocks_per_seq = math.ceil(config.max_model_len / self.block_size)
        self.sampler = Sampler(device, mc.vocab_size, mc.eos_token_ids, config.seed)

        self.graphs: dict[tuple[int, int], "torch.cuda.CUDAGraph"] = {}
        self.graph_batch_sizes: list[int] = []
        self.graph_ctx_buckets: list[int] = []
        self.graph_capture_seconds = 0.0
        use_graph = config.use_cuda_graph
        if use_graph is None:
            use_graph = device.type == "cuda"
        if config.enforce_eager:
            use_graph = False
        if self.attn_backend == "triton":
            self._warmup()
        if use_graph:
            if device.type != "cuda":
                logger.warning("use_cuda_graph=True ignored: CUDA graphs need an NVIDIA GPU")
            else:
                t0 = time.perf_counter()
                try:
                    self._capture_cuda_graphs()
                except Exception as e:   # never let an optional optimisation break generation
                    logger.warning("CUDA graph capture failed (%s); falling back to eager mode", e)
                    self.graphs = {}
                self.graph_capture_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ KV cache sizing
    def block_bytes(self) -> int:
        mc = self.model_config
        return 2 * mc.num_hidden_layers * self.block_size * mc.num_key_value_heads * mc.head_dim * dtype_size(self.dtype)

    def _decide_num_blocks(self) -> int:
        cfg = self.config
        # never useful to have more blocks than max_num_seqs full-length sequences
        max_useful = cfg.max_num_seqs * math.ceil(cfg.max_model_len / self.block_size)
        if cfg.num_kvcache_blocks is not None:
            n = cfg.num_kvcache_blocks
        elif self.device.type == "cuda":
            n = min(self._profile_gpu_blocks(), max_useful)
        else:
            n = min(int(cfg.cpu_kvcache_gb * 1024 ** 3) // self.block_bytes(), max_useful)
        if n * self.block_size < cfg.max_model_len:
            raise ValueError(f"KV cache of {n} blocks x {self.block_size} tokens cannot hold one sequence of "
                             f"max_model_len={cfg.max_model_len}; give it more memory or lower max_model_len")
        return n

    @torch.inference_mode()
    def _profile_gpu_blocks(self) -> int:
        """Run a worst-case prefill without a cache to measure activation memory, then give the rest to the cache."""
        cfg = self.config
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        n_tok = cfg.max_num_batched_tokens
        seq_len = min(cfg.max_model_len, n_tok)
        lens = [seq_len] * (n_tok // seq_len)
        cu = [0]
        for length in lens:
            cu.append(cu[-1] + length)
        ids = torch.zeros(cu[-1], dtype=torch.long, device=self.device)
        pos = torch.cat([torch.arange(length, device=self.device) for length in lens])
        meta = AttentionMetadata(slot_mapping=pos, block_size=self.block_size, backend=self.attn_backend,
                                 cu_seqlens_q=cu, seqlens_k=lens, max_q_len=seq_len)
        hidden = self.model(ids, pos, None, meta)
        logits = self.model.compute_logits(hidden[-cfg.max_num_seqs:])
        # the sampler needs a few [batch, vocab] float32 temporaries (sort, softmax, noise...)
        tmp = [torch.empty_like(logits) for _ in range(4)]
        torch.cuda.synchronize()
        # extra memory the forward pass needed on top of the weights
        activation_peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
        del hidden, logits, tmp
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free          # weights + CUDA context + anything else running on the GPU
        budget = total * cfg.gpu_memory_utilization - used - activation_peak
        return max(int(budget) // self.block_bytes(), 0)

    # ------------------------------------------------------------------ batch preparation
    def _to_device(self, *arrays: list[int]) -> list[torch.Tensor]:
        """Copy several int lists to the device with ONE pinned, non-blocking transfer."""
        flat = [x for a in arrays for x in a]
        cpu = torch.tensor(flat, dtype=torch.long)
        if self.device.type == "cpu":
            dev = cpu
        else:
            dev = cpu.pin_memory().to(self.device, non_blocking=True)
        out, i = [], 0
        for a in arrays:
            out.append(dev[i:i + len(a)])
            i += len(a)
        return out

    def _table_width(self, seqs: list[Sequence]) -> int:
        w = max(len(s.block_table) for s in seqs)
        return min(-(-w // 16) * 16, max(self.max_blocks_per_seq, w))   # pad -> fewer kernel variants

    def prepare(self, out: SchedulerOutput):
        """Returns input_ids, positions, metadata, logits_indices (rows whose next token is sampled)."""
        seqs, nd = out.seqs, out.num_decode
        bs = self.block_size
        input_ids, positions, seq_index, ctx_lens, logits_idx = [], [], [], [], []
        cu_q = [0]
        for i, s in enumerate(seqs):
            start, n = s.num_computed_tokens, s.num_scheduled_tokens
            end = start + n
            input_ids.extend(s.token_ids[start:end])
            positions.extend(range(start, end))
            seq_index.extend([i] * n)
            ctx_lens.append(end)
            if i >= nd:
                cu_q.append(cu_q[-1] + n)
            if end == s.num_tokens:
                logits_idx.append(len(input_ids) - 1)
        width = self._table_width(seqs)
        tables = []
        for s in seqs:
            tables.extend(s.block_table)
            tables.extend([0] * (width - len(s.block_table)))
        (ids_t, pos_t, sidx_t, ctx_t, cu_t, lidx_t, bt_t) = self._to_device(
            input_ids, positions, seq_index, ctx_lens, cu_q, logits_idx, tables)
        bt_t = bt_t.view(len(seqs), width)
        slot_mapping = bt_t[sidx_t, pos_t // bs] * bs + pos_t % bs
        nprefill = len(seqs) - nd
        meta = AttentionMetadata(slot_mapping=slot_mapping, block_size=bs, backend=self.attn_backend,
                                 num_decode=nd)
        if nd:
            dec_lens = ctx_lens[:nd]
            meta.decode_context_lens = ctx_t[:nd]
            meta.decode_block_tables = bt_t[:nd]
            meta.decode_max_len = max(dec_lens)
            meta.decode_all_same_len = min(dec_lens) == meta.decode_max_len
        if nprefill:
            meta.cu_seqlens_q = cu_q
            meta.seqlens_k = ctx_lens[nd:]
            meta.cu_seqlens_q_t = cu_t
            meta.seqlens_k_t = ctx_t[nd:]
            meta.prefill_block_tables = bt_t[nd:]
            meta.max_q_len = max(s.num_scheduled_tokens for s in seqs[nd:])
        return ids_t, pos_t, meta, lidx_t

    # ------------------------------------------------------------------ forward + sampling
    @torch.inference_mode()
    def compute_logits(self, out: SchedulerOutput) -> torch.Tensor:
        """One forward pass; returns next-token logits [num sampling seqs, vocab] (float32)."""
        if out.num_prefill == 0 and self._graph_key(out.seqs) is not None:
            hidden = self._replay_graph(out.seqs)
            rows = [i for i, s in enumerate(out.seqs) if s.num_computed_tokens + 1 == s.num_tokens]
            if len(rows) != len(out.seqs):        # some rows were 1-token prompt chunks: nothing to sample
                hidden = hidden[torch.tensor(rows, dtype=torch.long, device=self.device)]
            return self.model.compute_logits(hidden)
        input_ids, positions, meta, logits_idx = self.prepare(out)
        hidden = self.model(input_ids, positions, self.kv_caches, meta)
        return self.model.compute_logits(hidden[logits_idx])

    @torch.inference_mode()
    def run(self, out: SchedulerOutput, sampling: list[Sequence] | None = None):
        """Forward + sample. Returns (token ids, logprobs) for out.sampling_seqs, in order."""
        logits = self.compute_logits(out)
        sampling = out.sampling_seqs if sampling is None else sampling
        if not sampling:
            return [], None
        return self.sampler(logits, sampling)

    @torch.inference_mode()
    def _warmup(self):
        """Compile the Triton kernels now (a few seconds the first time, cached on disk afterwards)
        instead of during the first request. Writes only into the scratch block."""
        n = self.block_size
        seq_tables = torch.full((1, 16), self.scratch_block, dtype=torch.long, device=self.device)
        slots = torch.arange(n, device=self.device) + self.scratch_slot
        for q_len in (n, 1):
            meta = AttentionMetadata(slot_mapping=slots[:q_len], block_size=self.block_size, backend=self.attn_backend)
            if q_len == 1:
                meta.num_decode = 1
                meta.decode_context_lens = torch.tensor([n], device=self.device)
                meta.decode_block_tables = seq_tables
                meta.decode_max_len = n
            else:
                meta.cu_seqlens_q, meta.seqlens_k, meta.max_q_len = [0, n], [n], n
                meta.cu_seqlens_q_t = torch.tensor([0, n], device=self.device)
                meta.seqlens_k_t = torch.tensor([n], device=self.device)
                meta.prefill_block_tables = seq_tables
            self.model(torch.zeros(q_len, dtype=torch.long, device=self.device),
                       torch.arange(n - q_len, n, device=self.device), self.kv_caches, meta)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ------------------------------------------------------------------ CUDA graphs
    # A CUDA graph records the hundreds of small GPU kernel launches of one decode step once and
    # replays them with a single launch. Shapes must be static, so we capture a few batch sizes and
    # pad the batch up to the next captured size (padding rows write into the scratch block).
    #   triton backend: the kernels loop up to each sequence's real context length at runtime,
    #                   so one graph per batch size covers every context length;
    #   sdpa backend  : attention gathers a fixed number of keys, so graphs are also captured for a few
    #                   context-length buckets (256, 512, ... max_model_len) and the smallest bucket
    #                   that fits is replayed (instead of always gathering max_model_len keys).
    # The "static decode" helpers below are plain PyTorch and are unit-tested on CPU; only the
    # capture/replay calls themselves need a GPU.
    def _ctx_buckets(self) -> list[int]:
        max_len = self.max_blocks_per_seq * self.block_size
        if self.attn_backend == "triton":
            return [max_len]
        buckets, b = [], 256
        while b < max_len:
            buckets.append(b)
            b *= 2
        return buckets + [max_len]

    def alloc_static_buffers(self, max_bs: int):
        dev, L = self.device, torch.long
        self._g = dict(
            vec=torch.zeros(4, max_bs, dtype=L, device=dev),      # rows: input_ids, positions, slots, ctx lens
            block_tables=torch.zeros(max_bs, self.max_blocks_per_seq, dtype=L, device=dev),
            hidden=torch.zeros(max_bs, self.model_config.hidden_size, dtype=self.dtype, device=dev),
        )
        self._g["vec"][2].fill_(self.scratch_slot)
        self._g["vec"][3].fill_(1)

    def static_decode_args(self, b: int, ctx_bucket: int | None = None):
        """Model inputs that are views into the static buffers, for a padded batch of size b."""
        g = self._g
        ctx_bucket = ctx_bucket or self.max_blocks_per_seq * self.block_size
        vec = g["vec"]
        meta = AttentionMetadata(
            slot_mapping=vec[2, :b], block_size=self.block_size, backend=self.attn_backend, num_decode=b,
            decode_context_lens=vec[3, :b], decode_block_tables=g["block_tables"][:b, :-(-ctx_bucket // self.block_size)],
            decode_max_len=ctx_bucket, decode_all_same_len=False)
        return vec[0, :b], vec[1, :b], self.kv_caches, meta

    def fill_static_inputs(self, seqs: list[Sequence], b: int):
        """Copy a real decode batch into the static buffers; rows n..b-1 become harmless padding
        (they write into the scratch block and attend to a single key)."""
        n, bs = len(seqs), self.block_size
        ids, pos, slots, ctx = [], [], [], []
        for s in seqs:
            p = s.num_computed_tokens
            ids.append(s.token_ids[p])
            pos.append(p)
            slots.append(s.block_table[p // bs] * bs + p % bs)
            ctx.append(p + 1)
        pad = b - n
        ids += [0] * pad
        pos += [0] * pad
        slots += [self.scratch_slot] * pad
        ctx += [1] * pad
        width = max(len(s.block_table) for s in seqs)
        tables = [x for s in seqs for x in s.block_table + [0] * (width - len(s.block_table))]
        vec_t, bt_t = self._to_device(ids + pos + slots + ctx, tables)
        self._g["vec"][:, :b].copy_(vec_t.view(4, b))
        self._g["block_tables"][:n, :width].copy_(bt_t.view(n, width))
        # Columns >= width of rows < n keep stale (valid) page ids: they lie beyond the context and
        # are never read (triton) or masked out (sdpa).

    def _graph_key(self, seqs: list[Sequence]) -> tuple[int, int] | None:
        if not self.graphs or len(seqs) > self.graph_batch_sizes[-1]:
            return None
        n = len(seqs)
        b = next(size for size in self.graph_batch_sizes if size >= n)
        need = max(s.num_computed_tokens for s in seqs) + 1
        bucket = next(c for c in self.graph_ctx_buckets if c >= need)
        return b, bucket

    def _capture_cuda_graphs(self):
        max_bs = min(self.config.max_num_seqs, self.config.max_cuda_graph_batch)
        sizes = sorted({b for b in [1, 2, 4, 8, *range(16, 33, 8), *range(48, max_bs + 1, 16)] if b <= max_bs} | {max_bs})
        buckets = self._ctx_buckets()
        self.alloc_static_buffers(max_bs)
        hidden = self._g["hidden"]
        pool = None
        with torch.inference_mode():
            for c in reversed(buckets):
                for b in reversed(sizes):        # largest first so smaller graphs can share its memory pool
                    args = self.static_decode_args(b, c)
                    hidden[:b] = self.model(*args)          # warm-up (also builds RoPE tables, compiles kernels)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        hidden[:b] = self.model(*args)
                    if pool is None:
                        pool = graph.pool()
                    torch.cuda.synchronize()
                    self.graphs[(b, c)] = graph
        self.graph_batch_sizes = sizes
        self.graph_ctx_buckets = buckets
        logger.info("captured %d CUDA graphs (batch sizes %s, context buckets %s)", len(self.graphs), sizes, buckets)

    def _replay_graph(self, seqs: list[Sequence]) -> torch.Tensor:
        b, c = self._graph_key(seqs)
        self.fill_static_inputs(seqs, b)
        self.graphs[(b, c)].replay()
        return self._g["hidden"][:len(seqs)]
