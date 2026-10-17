"""ModelRunner: owns the model weights and the KV cache, and runs one forward pass per step.

It translates a list of Sequences (Python objects) into flat tensors the model understands:
  input_ids / positions  - which tokens to compute and where they sit in their sequence
  slot_mapping           - where to write each new token's K/V in the paged cache
  block_tables           - which pages each sequence owns (to read old K/V back)
"""

import logging
import math

import torch

from nanoinfer.config import EngineConfig, ModelConfig
from nanoinfer.engine.sampler import sample
from nanoinfer.engine.sequence import Sequence
from nanoinfer.model.attention import AttentionMetadata
from nanoinfer.model.loader import load_model
from nanoinfer.utils import dtype_size

logger = logging.getLogger(__name__)


class ModelRunner:
    def __init__(self, config: EngineConfig, model_config: ModelConfig, device: torch.device, dtype: torch.dtype):
        self.config = config
        self.model_config = model_config
        self.device = device
        self.dtype = dtype
        self.block_size = config.block_size
        torch.manual_seed(config.seed)
        self.model = load_model(config.model, device, dtype, max_positions=config.max_model_len, cfg=model_config)

        self.num_blocks = self._decide_num_blocks()
        mc = model_config
        # One tensor for the whole cache: [K/V, layer, block, slot-in-block, kv_head, head_dim].
        # One extra "scratch" block at the end absorbs writes from padding rows (CUDA graphs).
        self.kv_cache = torch.zeros(2, mc.num_hidden_layers, self.num_blocks + 1, self.block_size,
                                    mc.num_key_value_heads, mc.head_dim, device=device, dtype=dtype)
        self.kv_caches = [(self.kv_cache[0, i], self.kv_cache[1, i]) for i in range(mc.num_hidden_layers)]
        self.scratch_slot = self.num_blocks * self.block_size
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(config.seed)

        self.graphs: dict[int, "torch.cuda.CUDAGraph"] = {}
        if config.use_cuda_graph:
            if device.type != "cuda":
                logger.warning("use_cuda_graph=True ignored: CUDA graphs need an NVIDIA GPU")
            else:
                try:
                    self._capture_cuda_graphs()
                except Exception as e:   # never let an optional optimisation break generation
                    logger.warning("CUDA graph capture failed (%s); falling back to eager mode", e)
                    self.graphs = {}

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
        meta = AttentionMetadata(is_prefill=True, slot_mapping=pos, block_size=self.block_size,
                                 cu_seqlens_q=cu, cu_seqlens_k=cu, backend=cfg.attn_backend)
        hidden = self.model(ids, pos, None, meta)
        self.model.compute_logits(hidden[-cfg.max_num_seqs:])
        torch.cuda.synchronize()
        # extra memory the forward pass needed on top of the weights
        activation_peak = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
        del hidden
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        used = total - free          # weights + CUDA context + anything else running on the GPU
        budget = total * cfg.gpu_memory_utilization - used - activation_peak
        return max(int(budget) // self.block_bytes(), 0)

    # ------------------------------------------------------------------ batch preparation
    def _slot(self, seq: Sequence, pos: int) -> int:
        return seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size

    def _tensor(self, values) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _block_table_tensor(self, seqs: list[Sequence]) -> torch.Tensor:
        """[B, max_blocks] page ids; short rows are padded with 0 (never read: masked or out of range)."""
        width = max(len(s.block_table) for s in seqs)
        return self._tensor([s.block_table + [0] * (width - len(s.block_table)) for s in seqs])

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, slot_mapping = [], [], []
        cu_q, cu_k = [0], [0]
        for seq in seqs:
            start, end = seq.num_cached_tokens, seq.num_tokens   # skip tokens already in the prefix cache
            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            slot_mapping.extend(self._slot(seq, p) for p in range(start, end))
            cu_q.append(cu_q[-1] + end - start)
            cu_k.append(cu_k[-1] + end)
        has_prefix = any(s.num_cached_tokens > 0 for s in seqs)
        t = self._tensor
        meta = AttentionMetadata(
            is_prefill=True, slot_mapping=t(slot_mapping), block_size=self.block_size,
            block_tables=self._block_table_tensor(seqs) if has_prefix else None,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, backend=self.config.attn_backend)
        last_token_index = t([c - 1 for c in cu_q[1:]])   # we only need logits of each prompt's last token
        return t(input_ids), t(positions), meta, last_token_index

    def prepare_decode(self, seqs: list[Sequence]):
        t = self._tensor
        lens = [s.num_tokens for s in seqs]
        input_ids = t([s.last_token for s in seqs])
        positions = t([n - 1 for n in lens])
        slot_mapping = t([self._slot(s, n - 1) for s, n in zip(seqs, lens)])
        max_len = max(lens)
        meta = AttentionMetadata(
            is_prefill=False, slot_mapping=slot_mapping, block_size=self.block_size,
            context_lens=t(lens), block_tables=self._block_table_tensor(seqs),
            max_context_len=max_len, all_same_len=all(n == max_len for n in lens),
            backend=self.config.attn_backend)
        return input_ids, positions, meta

    # ------------------------------------------------------------------ forward + sampling
    @torch.inference_mode()
    def compute_logits(self, seqs: list[Sequence], is_prefill: bool) -> torch.Tensor:
        """One forward pass; returns next-token logits [len(seqs), vocab] (float32)."""
        if is_prefill:
            input_ids, positions, meta, last_idx = self.prepare_prefill(seqs)
            hidden = self.model(input_ids, positions, self.kv_caches, meta)[last_idx]
        elif self.graphs and len(seqs) <= max(self.graphs):
            hidden = self._replay_graph(seqs)
        else:
            input_ids, positions, meta = self.prepare_decode(seqs)
            hidden = self.model(input_ids, positions, self.kv_caches, meta)
        return self.model.compute_logits(hidden)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        logits = self.compute_logits(seqs, is_prefill)
        sps = [s.sampling_params for s in seqs]
        temps = torch.tensor([sp.temperature for sp in sps], dtype=torch.float32, device=self.device)
        top_ps = torch.tensor([sp.top_p for sp in sps], dtype=torch.float32, device=self.device)
        top_ks = torch.tensor([sp.top_k for sp in sps], dtype=torch.long, device=self.device)
        return sample(logits, temps, top_ps, top_ks, self.generator).tolist()

    # ------------------------------------------------------------------ CUDA graphs (experimental)
    # A CUDA graph records the hundreds of small GPU kernel launches of one decode step once and
    # replays them with a single launch. Shapes must be static, so we capture a few batch sizes,
    # pad the batch up to the next captured size, and always gather max_model_len keys (masked).
    # The "static decode" helpers below are plain PyTorch and are unit-tested on CPU; only the
    # capture/replay calls themselves need a GPU.
    def alloc_static_buffers(self, max_bs: int):
        max_blocks = math.ceil(self.config.max_model_len / self.block_size)
        dev, L = self.device, torch.long
        self._g = dict(
            input_ids=torch.zeros(max_bs, dtype=L, device=dev),
            positions=torch.zeros(max_bs, dtype=L, device=dev),
            slot_mapping=torch.full((max_bs,), self.scratch_slot, dtype=L, device=dev),
            context_lens=torch.ones(max_bs, dtype=L, device=dev),
            block_tables=torch.zeros(max_bs, max_blocks, dtype=L, device=dev),
            hidden=torch.zeros(max_bs, self.model_config.hidden_size, dtype=self.dtype, device=dev),
        )

    def static_decode_args(self, b: int):
        """Model inputs that are views into the static buffers, for a padded batch of size b."""
        g = self._g
        max_len = g["block_tables"].shape[1] * self.block_size
        meta = AttentionMetadata(
            is_prefill=False, slot_mapping=g["slot_mapping"][:b], block_size=self.block_size,
            context_lens=g["context_lens"][:b], block_tables=g["block_tables"][:b],
            max_context_len=max_len, all_same_len=False, backend=self.config.attn_backend)
        return g["input_ids"][:b], g["positions"][:b], self.kv_caches, meta

    def fill_static_inputs(self, seqs: list[Sequence], b: int):
        """Copy a real decode batch into the static buffers; rows n..b-1 become harmless padding
        (they write into the scratch block and attend to a single dummy key)."""
        n, g = len(seqs), self._g
        input_ids, positions, meta = self.prepare_decode(seqs)
        g["input_ids"][:b].zero_()
        g["positions"][:b].zero_()
        g["slot_mapping"][:b].fill_(self.scratch_slot)
        g["context_lens"][:b].fill_(1)
        g["block_tables"][:b].zero_()
        g["input_ids"][:n] = input_ids
        g["positions"][:n] = positions
        g["slot_mapping"][:n] = meta.slot_mapping
        g["context_lens"][:n] = meta.context_lens
        g["block_tables"][:n, :meta.block_tables.shape[1]] = meta.block_tables

    def _capture_cuda_graphs(self):
        max_bs = min(self.config.max_num_seqs, 256)
        sizes = sorted({b for b in [1, 2, 4, 8, *range(16, max_bs + 1, 16)] if b <= max_bs} | {max_bs})
        self.alloc_static_buffers(max_bs)
        hidden = self._g["hidden"]
        pool = None
        with torch.inference_mode():
            for b in reversed(sizes):        # largest first so smaller graphs can share its memory pool
                args = self.static_decode_args(b)
                hidden[:b] = self.model(*args)          # warm-up (also builds RoPE tables)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    hidden[:b] = self.model(*args)
                if pool is None:
                    pool = graph.pool()
                torch.cuda.synchronize()
                self.graphs[b] = graph
        logger.info("captured CUDA graphs for decode batch sizes %s", sizes)

    def _replay_graph(self, seqs: list[Sequence]) -> torch.Tensor:
        n = len(seqs)
        b = min(size for size in self.graphs if size >= n)
        self.fill_static_inputs(seqs, b)
        self.graphs[b].replay()
        return self._g["hidden"][:n].clone()
