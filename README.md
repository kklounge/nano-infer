# nano-infer 🐣

**一个从零手写、能读懂、在消费级显卡上也很快的大模型推理引擎（LLM inference engine）。**
约 3,100 行带详细注释的 Python + Triton 代码，实现了 vLLM 的核心技术，支持 Qwen2 / Qwen2.5 / Qwen3：

- 🧠 **自己实现的 Qwen2 / Qwen3 模型**：RMSNorm、RoPE、GQA、SwiGLU、Qwen3 QK-Norm，合并的 qkv / gate_up 投影，直接加载 Hugging Face 官方 safetensors 权重
- 📄 **分页 KV 缓存（Paged KV Cache）**：固定大小的块 + 块表，按需分配
- 🔄 **连续批处理 + 分块预填充（Chunked Prefill）**：按 token 预算调度，decode 和 prompt 分段**混在同一个 batch**（vLLM v1 / Sarathi 风格）；缓存不够时抢占并重算
- ♻️ **前缀缓存（Prefix Caching）**：块哈希链 + 引用计数 + LRU，相同开头只算一次（同一个 batch 内也能共享）
- ⚡ **自己写的 Triton GPU kernel**：分页注意力（flash-decoding 的 decode kernel + varlen 分页 prefill kernel），以及融合的 add+RMSNorm、RoPE+写 KV 缓存、SiLU×mul
- 🎬 **CUDA Graph**（GPU 默认开启）：一整步 decode 录制成一张图回放
- 🎲 **完整的采样**：greedy、temperature、top-k、top-p、min-p、repetition / presence / frequency penalty、min_tokens、stop 字符串、stop token、每个请求自己的随机种子、logprobs、n 个候选
- 🌐 **OpenAI 兼容 HTTP 服务**：`/v1/completions`、`/v1/chat/completions`，流式输出（SSE）、usage（含缓存命中 token 数）、`/metrics`（Prometheus）、`/health`
- 🖥️ **命令行**：`python -m nanoinfer serve / chat / generate / info`
- ✅ **正确性**：float32 下贪心输出和 Hugging Face transformers **逐 token 一致**（含 RTX 5070 上 Qwen2.5-0.5B / 1.5B、Qwen3-1.7B）；125 个测试（开发机 CPU：121 通过 + 4 个 GPU 测试跳过；RTX 5070：147 通过，GPU 上额外跑 bf16/fp16 的 kernel 测试）

**在 RTX 5070 上（Qwen2.5-0.5B，bf16）：吞吐量是 Hugging Face `generate()` 的 5.9–7.1 倍（静态批处理）和 11.4 倍（长度不一的并发请求）；比本项目自己的 v0.1 快 2.4–3.1 倍。** 详见[性能测试](#性能测试结果rtx-5070-实测)。

> 这是一个**学习项目**：设计思路参考了 vLLM / Sarathi-Serve / FlashAttention / flash-decoding 的论文和 nano-vllm 等开源项目，代码是重新独立编写的。

---

## 目录
- [它是什么？（给完全的新手）](#它是什么给完全的新手)
- [在 Windows 上安装和运行](#在-windows-上安装和运行)
- [怎么用](#怎么用)
- [测试结果](#测试结果)
- [性能测试结果（RTX 5070 实测）](#性能测试结果rtx-5070-实测)
- [文档：一步一步学懂推理](#文档一步一步学懂推理)
- [局限和可以继续做的事](#局限和可以继续做的事)

---

## 它是什么？（给完全的新手）

ChatGPT 这类大模型，"训练"是教会它说话，"**推理（inference）**"是让训练好的模型真正回答问题。推理的时候，模型一次只能写出一个字（token），要一个字一个字地循环。

当很多人同时提问时，推理引擎要解决这些问题：

1. **记忆放哪？** 每个对话都要存一份"已经读过的内容的笔记"（KV 缓存），很占显存。→ 像操作系统管理内存一样切成一页一页（**分页**）。
2. **谁先算？** 有人问得短，有人问得长。→ 每写一个字就重新排一次队，写完的马上走、新来的马上加入（**连续批处理**）；很长的问题切成几段，不让别人干等（**分块预填充**）。
3. **重复的能不能少算？** 很多请求开头一样（比如同一段系统提示词）。→ 相同的开头只算一次（**前缀缓存**）。
4. **GPU 有没有在空转？** 小模型每一步的计算很少，时间都花在"发指令"上了。→ 自己写 GPU kernel 把小操作合并（**Triton 融合算子**），把一整步录成一张图一次发出去（**CUDA Graph**）。

nano-infer 就是这些事的一个小而完整、能跑、经过测试的实现。

---

## 在 Windows 上安装和运行

> 以下命令在 **PowerShell** 里运行。

### 第 1 步：装 Python
从 <https://www.python.org/downloads/> 安装 Python 3.10 – 3.13，安装时**勾选 "Add python.exe to PATH"**。

### 第 2 步：解压项目，建虚拟环境
```powershell
cd $HOME\Desktop
Expand-Archive nano-infer.zip -DestinationPath .
cd nano-infer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
如果提示"禁止运行脚本"，先执行一次：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

### 第 3 步：装 PyTorch
```powershell
nvidia-smi
```
- **能看到显卡** → 装 CUDA 版（RTX 50 系列需要 cu128 或更新）：
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```
- **没有 NVIDIA 显卡** → `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 第 4 步：装其余依赖、下载模型
```powershell
pip install -r requirements.txt          # Windows 上会自动装 triton-windows（GPU kernel 需要）
python scripts/download_model.py         # 默认 Qwen2.5-0.5B-Instruct，约 1 GB
python scripts/download_model.py --repo Qwen/Qwen2.5-1.5B-Instruct   # 可选：更大的模型
```
国内网络慢的话，先设置镜像：`$env:HF_ENDPOINT="https://hf-mirror.com"`。

### 第 5 步：看看引擎选了什么
```powershell
python -m nanoinfer info --model models/Qwen2.5-0.5B-Instruct
```
有显卡时应该看到 `on cuda (bfloat16), attention=triton, fused_ops=True, 9 CUDA graphs`。没装 Triton 时会自动退回 PyTorch 的 `sdpa` 后端（仍然能用 CUDA Graph）。

### 第 6 步：跑起来
```powershell
python -m nanoinfer chat --model models/Qwen2.5-0.5B-Instruct          # 终端里多轮对话
python examples/basic_generate.py                                         # 批量生成
python -m pytest tests -q                                                 # 全部测试
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct   # 性能测试
```

---

## 怎么用

### Python 接口
```python
from nanoinfer import LLM, SamplingParams

llm = LLM("models/Qwen2.5-0.5B-Instruct")          # 自动选 GPU / CPU、注意力后端、CUDA Graph
outs = llm.generate(
    ["The capital of France is", "用一句话介绍康奈尔大学"],
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64, stop=["\n\n"], seed=1),
)
for o in outs:
    print(o.text, o.finish_reason, o.ttft)

print(llm.chat([{"role": "user", "content": "What is an LBO?"}])[0].text)   # 自动套用 chat template
print(llm.metrics())   # 运行/等待的请求数、KV 使用率、前缀命中率、各类 step 数、step 耗时分位数……
```

`SamplingParams` 支持：`temperature, top_p, top_k, min_p, repetition_penalty, presence_penalty, frequency_penalty, max_tokens, min_tokens, stop, stop_token_ids, ignore_eos, seed, logprobs, n, skip_special_tokens`。

`LLM(...)` 的常用参数（见 `nanoinfer/config.py`）：`device`、`dtype`、`max_num_seqs`、`max_model_len`、`max_num_batched_tokens`、`enable_chunked_prefill`、`enable_prefix_caching`、`attn_backend`（"auto"/"triton"/"sdpa"/"naive"）、`use_fused_ops`、`enforce_eager`（关闭 CUDA Graph）、`gpu_memory_utilization`。

### HTTP 服务（OpenAI 兼容）
```powershell
python -m nanoinfer serve --model models/Qwen2.5-0.5B-Instruct --port 8000
```
可以直接用 `openai` 这个 Python 包：
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
stream = client.chat.completions.create(model="Qwen2.5-0.5B-Instruct", stream=True,
                                        messages=[{"role": "user", "content": "讲个笑话"}])
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```
支持 `n`、`seed`、`stop`、`logprobs` / `top_logprobs`、`echo`、`stream_options.include_usage`、prompt 直接传 token id；客户端断开时自动取消请求、释放 KV 块。`GET /metrics` 返回 Prometheus 格式的指标。

---

## 测试结果

```powershell
python -m pytest tests -q
```

| 环境 | 结果 |
|---|---|
| 开发服务器（Linux，8 vCPU Xeon，**无 GPU**；Triton kernel 用解释器 `TRITON_INTERPRET=1` 在 CPU 上以 fp32 测） | **121 passed, 4 skipped**（跳过的是只能在 GPU 上跑的 `test_cuda.py`） |
| Patrick 的 PC（Windows 11，**RTX 5070**，torch 2.11.0+cu128，triton-windows 3.6） | **147 passed**（GPU 上 Triton 测试额外跑 bf16 / fp16） |

| 测试文件 | 内容 |
|---|---|
| `test_block_manager.py` | 块分配/释放、按 token 数懒分配、块边界、LRU、重复释放检测 |
| `test_prefix_cache.py` | 哈希链、引用计数、复活、淘汰、碰撞检查、分块预填充时未算完的块不可命中、同 batch 共享；开/关缓存输出相同 |
| `test_scheduler.py` | 混合批、分块预填充、token 预算、min_tokens / stop token、水位线、抢占（开/关前缀缓存），**强制抢占后输出不变** |
| `test_sampler.py` | greedy/top-k/top-p/min-p、温度分布、惩罚公式、repetition penalty 和 HF 一致、种子可复现且与 batch 无关、logprobs、stop 字符串 |
| `test_tiny_model_vs_hf.py` | 随机小 Qwen2/Qwen3：权重逐个相等；prefill、decode、**混合批和分块**时 logits 与 HF 误差 < 1e-4；不同分块预算下贪心逐 token 相同；CUDA Graph 补齐路径 |
| `test_triton_attention.py` / `test_triton_ops.py` | Triton kernel 和 PyTorch 参考实现逐元素对比（多种头数/维度/块大小；flash-decoding 长上下文） |
| `test_qwen_vs_hf.py` | 真实 Qwen2.5-0.5B，8 个中英文 prompt：逐个、批量、命中前缀缓存、naive 后端，都和 HF **逐 token 相同**（fp32） |
| `test_server.py` | 流式/非流式一致、并发合并成 batch、n/seed、stop + usage、logprobs、echo、错误格式、/metrics、**真实 uvicorn 下客户端断开会取消请求** |
| `test_cuda.py` | （仅 GPU）GPU 输出与 HF 一致、自动分配 KV 缓存、CUDA Graph 与普通模式逐 token 一致、真实模型 bf16 |

### 更大的模型：和 HF 对比（RTX 5070，`scripts/verify_model.py`）
8 个中英文 prompt × 48 个 token = 384 个位置。"teacher forcing 一致率"：把 HF 的贪心 token 一个个喂给本引擎（真实的分页 decode、CUDA Graph、Triton kernel），看每个位置的 argmax 是否和 HF 相同。

| 模型 | fp32：贪心完全一致 | fp32：一致率 | bf16：本引擎 vs HF | bf16 噪声基线：HF eager vs HF sdpa |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 8/8 | 100% | 98.2% | 96.6% |
| Qwen2.5-1.5B-Instruct | 8/8 | 100% | 99.2% | 96.1% |
| Qwen3-1.7B | 8/8 | 100% | 98.7% | 99.2% |
| Qwen2.5-3B-Instruct | 没测（fp32 权重 12 GB，放不进 12 GB 显存） | — | 98.2% | 99.2% |

**怎么读**：fp32 下完全一致，说明实现是对的；bf16 下偶尔不一致是舍入误差造成的"近似平局翻转"——HF 自己换一种注意力实现，一致率也只有 96–99%，本引擎落在同一个范围里。原始数据：`benchmarks/results/verify-*.json`。

---

## 性能测试结果（RTX 5070 实测）

**硬件和设置**：NVIDIA GeForce RTX 5070（12 GB），Windows 11，torch 2.11.0+cu128，triton-windows 3.6，**bfloat16**。基线是 Hugging Face `model.generate()`（贪心、静态批处理、SDPA 注意力）。测试前 `nvidia-smi` 显示 GPU 只被桌面程序占用约 2 GB、利用率 0%。测试日期 2026-10-17。原始数据在 `benchmarks/results/rtx5070-*.json`，完整输出在 `benchmarks/results/logs/`。

### 1. Qwen2.5-0.5B-Instruct：HF vs v0.1 vs v0.2

静态批处理：B 个请求，每个 128 token 输入、128 token 输出，3 次取中位数。

| 场景 | HF generate | nano v0.1（eager） | nano v0.1（`--cuda-graph`） | **nano v0.2（默认）** | v0.2 / HF | v0.2 / v0.1 最好 |
|---|---|---|---|---|---|---|
| B=1（tok/s） | 57.2 | 52.4 | 143.7 | **339.5** | 5.9x | 2.4x |
| B=4（tok/s） | 220.0 | 203.8 | 563.4 | **1571.3** | 7.1x | 2.8x |
| B=16（tok/s） | 877.5 | 791.8 | 1827.8 | **5615.4** | 6.4x | 3.1x |
| 混合长度：32 个请求各生成 16–256 token（有效 tok/s） | 409.2 | 690.0 | 1510.1 | **4656.3** | 11.4x | 3.1x |
| 共享 512 token 系统提示词，16 个请求：前缀缓存关 / 开（总耗时 s） | — | 0.88 / 0.68 | — | **0.25 / 0.11** | — | — |

- v0.1 的数字是同一天、同一块显卡上、在 v0.1 代码（commit `fd0355d`）上测的；HF 和 v0.2 的数字来自最终这次运行（`rtx5070-qwen2.5-0.5b.json`）。
- 混合长度场景：平均请求延迟 HF 6.50 s → v0.2 0.37 s。
- 共享前缀场景：v0.2 开前缀缓存后，实际计算的 prefill token 从 8,704 降到 512，平均首字延迟 0.092 s → 0.012 s。
- 首字延迟（TTFT，128 token 的 prompt）：B=1 时 HF 0.021 s，v0.2 0.006 s。

### 2. 每项优化贡献多少（v0.2，Qwen2.5-0.5B，tok/s）

| 配置 | B=1 | B=4 | B=16 | 混合长度 |
|---|---|---|---|---|
| 完整 v0.2（Triton 注意力 + 融合算子 + CUDA Graph） | 341.6 | 1556.1 | 5573.7 | 4621.9 |
| 关掉 CUDA Graph（`--enforce-eager`） | 207.8 | 789.1 | 3023.7 | 2656.3 |
| 注意力换回 PyTorch SDPA（`--attn-backend sdpa`，融合算子随之关闭，仍有 CUDA Graph） | 244.2 | 997.2 | 3270.0 | 2699.7 |
| 加融合算子之前（Triton 注意力 + CUDA Graph，上一个 commit） | 237.6 | 1000.7 | 3674.7 | 2164.0 |

（前三行是同一次运行的消融；第四行是同一天早些时候在融合算子 commit 之前测的。）用 `benchmarks/profile_step.py` 看单步 decode：B=1 时一步从 3.83 ms（无融合算子）降到 2.95 ms（其中前向 2.36 ms、lm_head 0.48 ms、采样 0.10 ms）。

### 3. 更大的模型（静态 128 进 / 128 出，3 次中位数；混合长度同上）

| 模型 | HF B=1 | nano B=1 | HF B=16 | nano B=16 | B=16 倍数 | HF 混合 | nano 混合 | 混合倍数 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 49.9 | 159.8 | 696.9 | 2204.6 | 3.2x | 320.0 | 1829.2 | 5.7x |
| Qwen3-1.7B | 40.0 | 152.0 | 583.4 | 1882.5 | 3.2x | 245.2 | 1596.8 | 6.5x |
| Qwen2.5-3B-Instruct | 37.1 | 88.8 | 501.4 | 1172.2 | 2.3x | 216.5 | 996.5 | 4.6x |

**为什么大模型倍数变小？** batch=1 的 decode 每生成一个 token 要把全部权重从显存读一遍。粗算"权重文件大小 × tok/s"：0.5B 约 1.0 GB × 339.5 ≈ 340 GB/s，1.5B 约 3.1 GB × 159.8 ≈ 500 GB/s，3B 约 6.2 GB × 88.8 ≈ 550 GB/s，而 RTX 5070 的显存带宽是 672 GB/s。也就是说：小模型时 HF 主要输在 CPU 发 kernel 的开销上（我们用 CUDA Graph 和融合算子消掉了），大模型时两边都逐渐接近显存带宽这个物理上限，能省的开销占比变小。（这是用文件大小估算的，忽略了 KV 缓存的读取。）

### 4. 在线服务：泊松到达的 HTTP 流式请求（`benchmarks/bench_serving.py`）

**分块预填充**：Qwen2.5-1.5B，150 个请求，平均每秒 2.5 个，prompt 1,900–3,800 token，输出 96–192 token。

| 配置 | TTFT 平均 / p99 (ms) | TPOT p99 (ms) | **ITL p99 (ms)** | 端到端平均 (ms) | 含 prefill 的 step 耗时 p50 (ms) |
|---|---|---|---|---|---|
| 不分块（`--no-chunked-prefill`） | 228 / 594 | 48.7 | **190.7** | 2870 | 155.0 |
| 分块，预算 2048（默认） | 251 / 580 | 44.6 | **116.5** | 2806 | 104.0 |
| 分块，预算 512 | 306 / 856 | 33.2 | **37.9** | 2576 | 31.0 |

**怎么读**：不分块时，一个 3,800 token 的 prompt 一进来，所有正在输出的请求都要等它整个算完，表现为 token 间隔（ITL）的长尾。分块把每一步的工作量限制在预算内：预算 512 时 ITL p99 从 190.7 ms 降到 37.9 ms（**-80%**），代价是新请求的首字延迟变长（p99 594 → 856 ms）。这就是 `max_num_batched_tokens` 要调的取舍。

**前缀缓存**：Qwen2.5-0.5B，200 个请求，平均每秒 16 个，每个 prompt = 768 token 共享前缀 + 512–1024 token 自己的内容，输出 64–128 token。

| 前缀缓存 | TTFT 平均 / p99 (ms) | TPOT 平均 (ms) | 端到端平均 (ms) |
|---|---|---|---|
| 关 | 61.7 / 144.6 | 10.9 | 1085 |
| **开** | **34.0 / 74.5** | **6.9** | **680** |

### 5. 开发机 CPU 上的数字（v0.1，仅供参考）
在没有 GPU 的开发服务器（Intel Xeon 8 vCPU，6 线程，bf16）上测过 v0.1：混合长度场景 HF 115.6 → nano 233.1 tok/s，共享前缀场景首字延迟 2.822 s → 0.245 s（`benchmarks/results/cpu-6threads.json`）。**这些是 CPU 数字，不能和上面的 GPU 数字混用。**

### 自己测
```powershell
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct              # HF vs nano，全部场景
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct --engines nano --enforce-eager   # 消融
python benchmarks/bench_serving.py --model models/Qwen2.5-1.5B-Instruct --num-requests 150 --request-rate 2.5 --input-len 3800 --output-len 192 --configs="default;--max-num-batched-tokens 512;--no-chunked-prefill"
python scripts/verify_model.py --model models/Qwen2.5-1.5B-Instruct --dtype float32
python benchmarks/profile_step.py --model models/Qwen2.5-0.5B-Instruct
```

---

## 文档：一步一步学懂推理

按顺序读（全部是中文，专业名词附英文）：

1. [大模型推理是怎么回事：token、自回归、采样、prefill vs decode](docs/01_llm_inference_basics.md)
2. [KV 缓存与分页：为什么要分页、块表、槽位](docs/02_kv_cache_and_paging.md)
3. [连续批处理与调度器：token 预算、分块预填充、混合批、抢占](docs/03_continuous_batching.md)
4. [前缀缓存：哈希链、引用计数、LRU](docs/04_prefix_caching.md)
5. [模型、注意力与 GPU 优化：Triton kernel、融合算子、CUDA Graph、采样器](docs/05_model_and_attention.md)
6. [代码地图：每个文件对应哪个概念，一个请求的完整旅程](docs/06_code_map.md)
7. [怎么证明它是对的、有多快：测试与性能测试](docs/07_testing_and_benchmark.md)
8. [面试指南：怎么讲这个项目、面试题、简历写法](docs/08_interview_guide.md)

---

## 局限和可以继续做的事

- **单卡**：没有张量并行（tensor parallelism）/ 流水线并行。
- **抢占只有重算（recompute）**，没有换出到 CPU（swap）。因为有前缀缓存，重算通常大部分是缓存命中。
- **没有量化**（int8 / int4 / FP8）、投机解码（speculative decoding）、LoRA、结构化输出（JSON schema）。
- **Triton kernel 是"够用"的版本**：prefill kernel 的块大小固定、没有 autotune；fp32 时用 IEEE 精度的 `tl.dot`。没有和 FlashAttention / FlashInfer 做过对比。
- 只支持 Qwen2 / Qwen3 架构（不支持 RoPE scaling、滑动窗口注意力、MoE）。
- 只在 RTX 5070（Windows）和 CPU（Linux）上测过。

## License
MIT
