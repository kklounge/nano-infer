# 06 · 代码地图：每个文件对应哪个概念

```
nano-infer/
├── nanoinfer/
│   ├── __init__.py            对外接口：LLM, SamplingParams
│   ├── __main__.py            命令行：python -m nanoinfer {serve, chat, generate, info}
│   ├── cli_args.py            serve / chat / benchmark 共用的引擎参数
│   ├── config.py              ModelConfig（读 config.json）+ EngineConfig（引擎参数）
│   ├── sampling_params.py     每个请求的采样设置（温度、top-p/k、min-p、惩罚、stop、seed、logprobs、n…）
│   ├── utils.py               自动选设备（有 CUDA 用 GPU，否则 CPU）和数据类型
│   ├── model/
│   │   ├── layers.py          RMSNorm（含残差融合）、RoPE                    → 05 篇
│   │   ├── attention.py       注意力元数据、写缓存、decode/prefill 分派、PyTorch 后端 → 02、05 篇
│   │   ├── triton_attention.py  Triton paged attention：flash-decoding、varlen prefill → 05 篇
│   │   ├── triton_ops.py      Triton 融合算子：add+RMSNorm、RoPE+写缓存、SiLU×mul → 05 篇
│   │   ├── qwen.py            Qwen2/Qwen3 网络结构（合并的 qkv_proj / gate_up_proj） → 05 篇
│   │   └── loader.py          safetensors 权重加载（含合并矩阵的映射）          → 05 篇
│   ├── engine/
│   │   ├── sequence.py        Sequence：一个请求的全部状态（已算/待算 token 数、时间戳…）
│   │   ├── block_manager.py   块管理器 + 前缀缓存                             → 02、04 篇
│   │   ├── scheduler.py       调度器：token 预算、分块预填充、混合批、抢占      → 03 篇
│   │   ├── model_runner.py    把调度结果变成张量、跑模型、管 KV 缓存、CUDA Graph
│   │   ├── sampler.py         惩罚、温度、top-k/top-p/min-p、种子、logprobs     → 05 篇第 8 节
│   │   ├── detokenizer.py     增量解码（流式输出不乱码）、停止字符串
│   │   └── llm_engine.py      LLM 类：add_request / step / generate / chat / metrics
│   └── server/
│       └── api_server.py      OpenAI 兼容 HTTP 服务（FastAPI + SSE 流式、/metrics、/health）
├── tests/                     pytest 测试（见 07 篇）
├── benchmarks/
│   ├── bench_throughput.py    离线吞吐/延迟：和 HF generate 对比（静态、混合长度、共享前缀）
│   ├── bench_serving.py       在线服务：泊松到达的 HTTP 流式请求，TTFT/TPOT/ITL 分位数
│   ├── bench_sampling.py      采样配置的开销：候选窗口改写前后的交替 A/B（旧实现从 git 取）
│   ├── profile_step.py        一步 decode 里各阶段（准备输入、前向、lm_head、采样）各花多少时间
│   └── results/               实测结果 JSON
├── examples/                  basic_generate.py（批量生成）、chat.py（终端多轮对话）
└── scripts/
    ├── download_model.py      下载模型权重
    └── verify_model.py        任意 Qwen 模型和 HF 对比（贪心 + teacher forcing 一致率）
```

核心代码（`nanoinfer/`，不含测试）约 3,100 行 Python，其中很大一部分是注释。

## 一个请求的完整旅程

以 `llm.generate(["你好"], SamplingParams(temperature=0, max_tokens=20))` 为例：

1. **`LLM.add_request`**（llm_engine.py）
   用分词器把文字变成 token id，创建一个 `Sequence`（状态 WAITING，`num_computed_tokens = 0`），放进调度器的 `waiting` 队列。

2. **`LLM.step` → `Scheduler.schedule`**（scheduler.py）
   先排 running 里的序列，再看 waiting：`BlockManager.get_cached_prefix` 查前缀缓存 → 剩下的 prompt 在 token 预算内能放多少放多少 → `allocate_prefix` + `allocate_slots` 分配块 → 状态变 RUNNING → 返回 `SchedulerOutput`（序列列表 + 每个序列这一步算几个 token）。

3. **`ModelRunner.compute_logits` → `prepare`**（model_runner.py）
   把每个序列这一步要算的 token 拼成一维 `input_ids`，算好 `positions`、块表、`cu_seqlens`，一次拷到 GPU，在 GPU 上算出 `slot_mapping`。

4. **`QwenForCausalLM.forward`**（qwen.py）
   Embedding → 24 层（RMSNorm → qkv_proj → RoPE + 写 KV 缓存 → paged attention → o_proj → add+RMSNorm → gate_up_proj → SiLU×mul → down_proj）→ RMSNorm。只取需要采样的 token 的输出去乘 lm_head，得到 logits。

5. **`Sampler`**（sampler.py）
   temperature=0 → 取 argmax，得到第一个新 token。

6. **`Scheduler.postprocess`**
   `num_computed_tokens` 前进；把 token 追加到序列末尾；检查 EOS、stop token、`max_tokens`、`min_tokens`；结束的序列释放块。`LLM.step` 再做增量解码、检查停止字符串。

7. **下一次 `step`**：这个序列只差 1 个 token → decode。
   `allocate_slots(seq, 1)` 需要时分配新块 → 如果这一步全是 decode、batch 大小有录好的 CUDA Graph，就把输入拷进静态缓冲区直接 `replay()` → 采样 → postprocess。

8. 重复第 7 步直到结束 → `deallocate` 释放块（引用计数减一；块内容和哈希保留，供前缀缓存复用）→ `generate` 返回 `RequestOutput`（文本、token、首字延迟 TTFT、总延迟、finish_reason）。

## 建议的阅读顺序

1. `sampling_params.py`、`sequence.py`（最简单，热身）
2. `block_manager.py` + `tests/test_block_manager.py` + `tests/test_prefix_cache.py`（对着测试读，最好懂）
3. `scheduler.py` + `tests/test_scheduler.py`
4. `model/layers.py` → `model/attention.py` → `model/qwen.py` → `model/triton_attention.py` → `model/triton_ops.py`
5. `engine/sampler.py` → `engine/model_runner.py` → `engine/detokenizer.py` → `engine/llm_engine.py`
6. `server/api_server.py`

每读完一个文件，试着不看代码，在纸上画出它的数据流。能画出来，才算真的懂了。
