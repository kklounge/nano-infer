# nano-infer 🐣

**一个从零手写、可以读懂的大模型推理引擎（LLM inference engine）。**
用大约 1,700 行带详细注释的 PyTorch 代码，实现了 vLLM 的几项核心技术：

- 🧠 **自己实现的 Qwen2 / Qwen3 模型**：RMSNorm、RoPE（旋转位置编码）、GQA（分组查询注意力）、SwiGLU MLP，直接加载 Hugging Face 官方的 safetensors 权重
- 📄 **分页 KV 缓存（Paged KV Cache）**：固定大小的块、块表（block table），按需分配和释放
- 🔄 **连续批处理（Continuous Batching）**：等待队列和运行队列，prefill 和 decode 两种步骤，缓存不够时**抢占（preemption）**并重算
- ♻️ **前缀缓存（Prefix Caching）**：块哈希链 + 引用计数 + LRU 淘汰，相同开头只算一次
- 🎲 **采样（Sampling）**：greedy、temperature、top-k、top-p，同一个 batch 里每个请求可以用不同设置
- 🌐 **OpenAI 兼容的 HTTP 服务**：`/v1/completions` 和 `/v1/chat/completions`，支持流式输出（streaming）
- ⚡ 有 NVIDIA 显卡就自动用 CUDA，没有就用 CPU；可选 **CUDA Graph**（实验性）
- ✅ **贪心输出和 Hugging Face transformers 逐 token 一致**，共 74 个测试（CPU 上 70 个通过，4 个 GPU 测试在没有显卡时自动跳过）

> 这是一个**学习项目**：设计思路参考了 vLLM 论文和 nano-vllm 等开源项目，代码是重新独立编写的。目标是**看得懂**，而不是和 vLLM 比速度。

---

## 目录
- [它是什么？（给完全的新手）](#它是什么给完全的新手)
- [在 Windows 上安装和运行](#在-windows-上安装和运行)
- [怎么用](#怎么用)
- [测试结果](#测试结果)
- [性能测试结果（实测）](#性能测试结果实测)
- [文档：一步一步学懂推理](#文档一步一步学懂推理)
- [局限和可以继续做的事](#局限和可以继续做的事)

---

## 它是什么？（给完全的新手）

ChatGPT 这类大模型，"训练"是教会它说话，"**推理（inference）**"是让训练好的模型真正回答问题。推理的时候，模型一次只能写出一个字（token），要一个字一个字地循环。

当很多人同时提问时，推理引擎要解决三个问题：

1. **记忆放哪？** 每个对话都要存一份"已经读过的内容的笔记"（KV 缓存），很占内存。→ 我们像操作系统管理内存一样，把它切成一页一页（**分页**）。
2. **谁先算？** 有人问得短，有人问得长。→ 每写一个字就重新排一次队，写完的马上走、新来的马上加入（**连续批处理**）。
3. **重复的能不能少算？** 很多请求开头一样（比如同一段系统提示词）。→ 相同的开头只算一次，大家共用（**前缀缓存**）。

nano-infer 就是这三件事的一个最小、完整、能跑、经过测试的实现。

---

## 在 Windows 上安装和运行

> 以下命令在 **PowerShell** 里运行。你的电脑（Ryzen 5 9600X + 32GB 内存）完全够用；有 NVIDIA 显卡会快很多。

### 第 1 步：装 Python
从 <https://www.python.org/downloads/> 安装 Python 3.10 – 3.13。安装时**勾选 "Add python.exe to PATH"**。

### 第 2 步：解压项目，建虚拟环境
```powershell
cd $HOME\Desktop
Expand-Archive nano-infer.zip -DestinationPath .
cd nano-infer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
如果提示"禁止运行脚本"，先执行一次：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，再激活。

### 第 3 步：装 PyTorch（先确认有没有 NVIDIA 显卡）
```powershell
nvidia-smi
```
- **能看到显卡信息** → 装 CUDA 版。去 <https://pytorch.org/get-started/locally/> 选 Windows / Pip / CUDA，复制它给的命令。例如：
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  ```
  （RTX 50 系列需要 cu128 或更新的版本。）
- **提示找不到命令** → 装 CPU 版：
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  ```

### 第 4 步：装其余依赖、下载模型
```powershell
pip install -r requirements.txt
python scripts/download_model.py
```
模型约 1 GB，会下载到 `models\Qwen2.5-0.5B-Instruct`。国内网络慢的话，先设置镜像：`$env:HF_ENDPOINT="https://hf-mirror.com"`。

### 第 5 步：确认能用 GPU（可选）
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 第 6 步：跑起来
```powershell
python examples/basic_generate.py                     # 批量生成 4 个回答
python examples/chat.py                               # 终端里多轮对话（会显示前缀缓存复用了多少 token）
python -m pytest tests -v                             # 跑全部测试
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct   # 性能测试，测你自己的数字
```
有显卡的话，`tests/test_cuda.py` 里的 4 个 GPU 测试会自动运行（包括 CUDA Graph），请留意它们是否通过。

---

## 怎么用

### Python 接口
```python
from nanoinfer import LLM, SamplingParams

llm = LLM("models/Qwen2.5-0.5B-Instruct")          # 自动选 GPU / CPU
outs = llm.generate(
    ["The capital of France is", "用一句话介绍康奈尔大学"],
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64),
)
for o in outs:
    print(o.text, o.finish_reason, o.ttft)

# 对话格式（自动套用 chat template）
print(llm.chat([{"role": "user", "content": "What is an LBO?"}])[0].text)
print(llm.stats)   # 计算了多少 prefill token、前缀缓存命中多少、抢占了几次……
```

常用参数（`LLM(...)` 的关键字参数，见 `nanoinfer/config.py`）：
`device`（"auto"/"cuda"/"cpu"）、`dtype`（"auto"/"bfloat16"/"float16"/"float32"）、`block_size`、`max_num_seqs`、`max_model_len`、`num_kvcache_blocks`、`enable_prefix_caching`、`attn_backend`（"sdpa"/"naive"）、`use_cuda_graph`。

在 CPU 上，如果电脑同时在做别的事，**少用几个线程反而更快**（见下面的性能测试）：`import torch; torch.set_num_threads(6)`。

### HTTP 服务（OpenAI 兼容）
```powershell
python -m nanoinfer.server.api_server --model models/Qwen2.5-0.5B-Instruct --port 8000
```
另开一个 PowerShell，用 Python 测试流式输出：
```python
import httpx, json
body = {"messages": [{"role": "user", "content": "讲个笑话"}], "max_tokens": 100, "stream": True}
with httpx.stream("POST", "http://127.0.0.1:8000/v1/chat/completions", json=body, timeout=None) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            print(json.loads(line[6:])["choices"][0]["delta"].get("content", ""), end="", flush=True)
```
因为接口和 OpenAI 一样，也可以直接用 `openai` 这个 Python 包，把 `base_url` 设成 `http://127.0.0.1:8000/v1`。

---

## 测试结果

在开发服务器上（Linux，8 核 Intel Xeon 云服务器 CPU，**没有 GPU**，Python 3.13，PyTorch 2.14.1 CPU 版，transformers 5.20.1）运行 `python -m pytest tests -v`：

```
======================== 70 passed, 4 skipped in 33.48s ========================
```

4 个跳过的是 `tests/test_cuda.py` 里只能在 NVIDIA 显卡上跑的测试。

| 测试文件 | 数量 | 内容 |
|---|---|---|
| `test_block_manager.py` | 7 | 块的分配/释放、块边界、LRU 淘汰、重复释放检测 |
| `test_prefix_cache.py` | 9 | 哈希链、引用计数共享、复活、淘汰、碰撞检查；开/关前缀缓存输出完全相同 |
| `test_scheduler.py` | 11 | prefill 优先、中途加入、预算限制、停止条件、抢占、取消；**强制抢占后输出不变** |
| `test_sampler.py` | 8 | greedy、top-k、top-p、温度分布、混合设置、随机种子 |
| `test_tiny_model_vs_hf.py` | 24 | 随机小 Qwen2/Qwen3：权重逐个相等，每步 logits 和 HF 误差 < 1e-4，批量贪心逐 token 相同，前缀命中后仍相同，CUDA Graph 补齐路径正确 |
| `test_qwen_vs_hf.py` | 5 | **真实 Qwen2.5-0.5B-Instruct**，8 个中英文 prompt：逐个、批量、命中前缀缓存、naive 注意力，都和 HF **逐 token 相同**（float32） |
| `test_server.py` | 6 | HTTP 普通/流式输出一致、并发请求被合并成 batch、参数校验、中文和 emoji 增量解码 |
| `test_cuda.py` | 4 | （仅 GPU）GPU 上贪心输出和 HF 一致、显存自动分配 KV 缓存、CUDA Graph 和普通模式输出一致、真实模型 bf16 在 GPU 上运行 |

为什么逐 token 对比用 float32？因为实测发现 **bf16 在多线程 CPU 上不是确定性的**：HF 自己对同一个输入算四次，logits 最大相差约 1.19，贪心输出会分叉（详见 [docs/07](docs/07_testing_and_benchmark.md)）。

---

## 性能测试结果（实测）

> ⚠️ 下面所有数字都是在**开发用的云服务器 CPU** 上实测的，**不是你电脑上的数字**。GPU 上的数字必须在你自己的机器上用 `benchmarks/bench_throughput.py` 测，简历上只能写你自己测到、能解释的数字。

**硬件和设置**：Intel Xeon 云服务器（8 个 vCPU，Linux，无 GPU），PyTorch 2.14.1+cpu，transformers 5.20.1，模型 Qwen2.5-0.5B-Instruct，**bfloat16**，PyTorch 使用 **6 个线程**，两边都用 SDPA 注意力。基线是 Hugging Face `model.generate()`（贪心、静态批处理）。原始数据：`benchmarks/results/cpu-6threads.json`。测试日期 2026-10-17。

### 1. 静态批处理（B 个请求，每个 128 token 输入、128 token 输出，3 次取中位数）

| batch | HF tok/s | nano-infer tok/s | 倍数 | HF 整批耗时 (s) | nano 整批耗时 (s) | HF 首字延迟 (s) | nano 首字延迟 (s) |
|---|---|---|---|---|---|---|---|
| 1 | 23.4 | 27.0 | 1.16x | 5.48 | 4.73 | 0.097 | 0.097 |
| 4 | 77.5 | 85.2 | 1.10x | 6.61 | 6.01 | 0.273 | 0.175 |
| 16 | 251.9 | 268.0 | 1.06x | 8.13 | 7.64 | 0.897 | 0.890 |

**解读**：这个场景对 HF 最有利（所有请求一样长，没有 padding，同时开始同时结束），两边基本持平（快 6%–16%）。批处理本身的威力两边都看得到：batch 从 1 到 16，吞吐量涨了约 10 倍。

### 2. 混合输出长度（32 个请求，各自要生成 16–256 个 token）

| 引擎 | 有效吞吐 tok/s | 总耗时 (s) | 平均请求延迟 (s) |
|---|---|---|---|
| HF generate，每 16 个一批（静态批处理） | 115.6 | 30.25 | 22.70 |
| **nano-infer，连续批处理** | **233.1** | **15.01** | **8.10** |

**解读**：这才是真实服务的样子。连续批处理让吞吐量提高到 **2.0 倍**，平均延迟降到原来的 **36%**：短请求写完就走，不用陪长请求等到最后。

### 3. 共享前缀（16 个请求共用一段 512 token 的系统提示词，各自再加 32 token 问题，生成 32 token）

| 前缀缓存 | 总耗时 (s) | 实际计算的 prefill token | 平均首字延迟 (s) |
|---|---|---|---|
| 关 | 5.95 | 8,704 | 2.822 |
| **开** | **2.14** | **512** | **0.245** |

**解读**：prefill 计算量减少 **94%**，首字延迟缩短到约 **1/11.5**，总耗时缩短到 **36%**。

### 一个意外发现：线程不是越多越好
同一台机器上用全部 8 个线程时，batch=16 只有 HF 92.3 / nano 123.4 tok/s（`benchmarks/results/cpu-8threads.json`），比 6 线程慢一半多。原因是这台云服务器上还有别的程序在用 CPU，PyTorch 每个并行操作都要等最慢的那个线程，只要有一个核被占用，所有线程都会被拖慢。所以 CPU 上推荐留一两个核给系统：`--threads 6` 或 `torch.set_num_threads(...)`。

### 在你的电脑上测
```powershell
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct                  # 有 GPU 会自动用 GPU
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct --device cpu --threads 5
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct --cuda-graph     # 试试实验性的 CUDA Graph
```
结果保存在 `benchmarks/results/` 里。

---

## 文档：一步一步学懂推理

按顺序读（全部是中文，专业名词附英文）：

1. [大模型推理是怎么回事：token、自回归、采样、prefill vs decode](docs/01_llm_inference_basics.md)
2. [KV 缓存与分页：为什么要分页、块表、槽位](docs/02_kv_cache_and_paging.md)
3. [连续批处理与调度器：等待/运行队列、抢占](docs/03_continuous_batching.md)
4. [前缀缓存：哈希链、引用计数、LRU](docs/04_prefix_caching.md)
5. [模型结构与注意力实现：RMSNorm、RoPE、GQA、SwiGLU、CUDA Graph](docs/05_model_and_attention.md)
6. [代码地图：每个文件对应哪个概念，一个请求的完整旅程](docs/06_code_map.md)
7. [怎么证明它是对的、有多快：测试与性能测试](docs/07_testing_and_benchmark.md)
8. [面试指南：怎么讲这个项目、20+ 道面试题、简历写法](docs/08_interview_guide.md)

---

## 局限和可以继续做的事

诚实地说，和 vLLM 相比它缺很多东西，这些也是很好的"下一步"：

- **注意力没有自定义 kernel**：先用 `index_select` 把分页的 K/V 捡成连续张量，再调用 PyTorch SDPA。可读、能在 CPU 跑，但多一次拷贝。→ 可以写一个 Triton 版的 paged attention。
- **prefill 是逐序列循环**，没有 varlen FlashAttention。
- **没有分块预填充（chunked prefill）**：一个 prompt 必须一次算完；prefill 步会让 decode 暂停一下。
- **抢占只有重算（recompute）**，没有换出到 CPU（swap）。
- 没有张量并行、量化、投机解码（speculative decoding）、LoRA、重复惩罚（repetition penalty）、停止字符串（只支持停止 token）。
- **CUDA Graph 没有在真实 GPU 上验证过**（开发机没有显卡），默认关闭；`tests/test_cuda.py` 会在有显卡的机器上测它。
- 只支持 Qwen2 / Qwen3 架构（不支持 RoPE scaling、滑动窗口注意力）。

## License
MIT
