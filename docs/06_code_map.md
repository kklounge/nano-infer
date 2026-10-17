# 06 · 代码地图：每个文件对应哪个概念

```
nano-infer/
├── nanoinfer/
│   ├── __init__.py            对外接口：LLM, SamplingParams
│   ├── config.py              ModelConfig（读 config.json）+ EngineConfig（引擎参数）
│   ├── sampling_params.py     每个请求的采样设置（温度、top-p、top-k、最大长度…）
│   ├── utils.py               自动选设备（有 CUDA 用 GPU，否则 CPU）和数据类型
│   ├── model/
│   │   ├── layers.py          RMSNorm、RoPE、SwiGLU MLP           → 05 篇
│   │   ├── attention.py       分页注意力：写缓存、gather、SDPA    → 02、05 篇
│   │   ├── qwen.py            Qwen2/Qwen3 网络结构                → 05 篇
│   │   └── loader.py          safetensors 权重加载                → 05 篇
│   ├── engine/
│   │   ├── sequence.py        Sequence：一个请求的全部状态
│   │   ├── block_manager.py   块管理器 + 前缀缓存                 → 02、04 篇
│   │   ├── scheduler.py       调度器：连续批处理 + 抢占           → 03 篇
│   │   ├── model_runner.py    把 Sequence 变成张量、跑模型、管 KV 缓存、CUDA Graph
│   │   ├── sampler.py         greedy / temperature / top-k / top-p → 01 篇
│   │   └── llm_engine.py      LLM 类：add_request / step / generate / chat
│   └── server/
│       └── api_server.py      OpenAI 兼容的 HTTP 服务（FastAPI，支持流式输出）
├── tests/                     pytest 测试（见 07 篇）
├── benchmarks/bench_throughput.py   与 HF generate 的吞吐/延迟对比
├── examples/                  basic_generate.py（批量生成）、chat.py（终端多轮对话）
└── scripts/download_model.py  下载模型权重
```

核心代码（不含测试）约 1,700 行 Python，其中很大一部分是注释，一两个下午可以通读。

## 一个请求的完整旅程

以 `llm.generate(["你好"], SamplingParams(temperature=0, max_tokens=20))` 为例：

1. **`LLM.add_request`**（llm_engine.py）
   用分词器把文字变成 token id，创建一个 `Sequence`（状态 WAITING），放进调度器的 `waiting` 队列。

2. **`LLM.step` → `Scheduler.schedule`**（scheduler.py）
   发现 waiting 里有请求、空块也够 → 调用 `BlockManager.allocate`，给 prompt 分配块（顺便查前缀缓存，命中的部分记在 `seq.num_cached_tokens`）→ 状态变 RUNNING → 返回 `(batch, is_prefill=True)`。

3. **`ModelRunner.run` → `prepare_prefill`**（model_runner.py）
   把 batch 里每个序列**没命中缓存的** token 拼成一维 `input_ids`，算好 `positions`、`slot_mapping`、`cu_seqlens`。

4. **`QwenForCausalLM.forward`**（qwen.py）
   Embedding → 24 层（每层的注意力先 `store_kv` 写缓存，再算注意力）→ RMSNorm。只取每个序列**最后一个 token** 的输出去乘 lm_head，得到 logits。

5. **`sample`**（sampler.py）
   temperature=0 → 取 argmax，得到第一个新 token。

6. **`Scheduler.postprocess`**
   把 token 追加到序列末尾；检查是否遇到结束符（EOS）或达到 `max_tokens`。

7. **下一次 `step`**：waiting 空了 → decode 步。
   `BlockManager.can_append / may_append` 检查是否要新块 → `prepare_decode` 只喂最后 1 个 token → 模型前向（注意力从缓存读出所有旧 K/V）→ 采样 → postprocess。

8. 重复第 7 步直到结束 → `deallocate` 释放块（引用计数减一；块内容和哈希保留，供前缀缓存复用）→ `generate` 返回 `RequestOutput`（文本、token、首字延迟 TTFT、总延迟）。

## 建议的阅读顺序

1. `sampling_params.py`、`sequence.py`（最简单，热身）
2. `block_manager.py` + `tests/test_block_manager.py` + `tests/test_prefix_cache.py`（对着测试读，最好懂）
3. `scheduler.py` + `tests/test_scheduler.py`
4. `model/layers.py` → `model/attention.py` → `model/qwen.py`
5. `engine/model_runner.py` → `engine/llm_engine.py`
6. `server/api_server.py`

每读完一个文件，试着不看代码，在纸上画出它的数据流。能画出来，才算真的懂了。
