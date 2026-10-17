# 07 · 怎么证明它是对的、有多快（Testing & Benchmark）

## 1. 正确性：和 Hugging Face 逐 token 对比

推理引擎最重要的是**算得对**。我们用 Hugging Face transformers 当"标准答案"：同一个 prompt，都用贪心解码（greedy），比较生成的 token 序列是否**每一个都相同**。

运行全部测试：

```bash
python -m pytest tests -q
```

实测：RTX 5070（Windows）**198 passed**（2026-10-17）；开发服务器（Linux CPU，无 GPU）**121 passed, 4 skipped**（加采样器测试之前测的，没有复测）。GPU 上多出来的是 Triton kernel 在 bf16 / fp16 下的测试；CPU 上 Triton kernel 用解释器（`TRITON_INTERPRET=1`，`tests/conftest.py` 自动设置）以 fp32 运行——解释器跑 bf16 结果不可靠，所以 CPU 上只测 fp32。

| 测试文件 | 测什么 | 需要下载模型吗 |
|---|---|---|
| `test_block_manager.py` | 分配、释放、按 token 数懒分配（分块预填充）、块边界、LRU、重复释放检测 | 否 |
| `test_prefix_cache.py` | 哈希链、引用计数、复活、淘汰、碰撞检查、decode 块登记、**分块预填充时还没算的块不能被命中**、同 batch 共享；开/关缓存输出相同 | 否 |
| `test_scheduler.py` | 混合批（新请求在 decode 中途加入）、分块预填充、和 decode 共享预算、EOS/最大长度/stop token/min_tokens、抢占、抢占那一步不接纳新请求、水位线、取消；**强制抢占后输出不变** | 否 |
| `test_sampler.py` | greedy、top-k、top-p、min-p、温度分布（统计检验）、惩罚公式、repetition penalty 和 HF 的 processor 一致、种子可复现且不受 batch 里其他请求影响、logprobs = log_softmax、stop 字符串截断 | 部分用真实模型 |
| `test_tiny_model_vs_hf.py` | **随机初始化的小 Qwen2 和 Qwen3**：权重逐个相等；prefill、decode、**混合批、分块**时每步 logits 和 HF 误差 < 1e-4；预算 7/16/33 的分块下贪心逐 token 相同；命中前缀缓存后仍相同；合并投影权重加载正确；CUDA Graph 静态补齐路径（多个上下文桶）与普通路径一致 | 否（现场生成） |
| `test_triton_attention.py` | Triton decode/prefill kernel 和 PyTorch 参考实现对比：多种头数（含 GQA）、head_dim（16/64/80/128）、块大小；flash-decoding 600/2000/5000 长度；用 Triton 后端的整个引擎和 HF 一致 | 否 |
| `test_triton_ops.py` | 融合 RMSNorm / add+RMSNorm、RoPE+写缓存、SiLU×mul 和 PyTorch 逐元素对比 | 否 |
| `test_qwen_vs_hf.py` | **真实 Qwen2.5-0.5B-Instruct**：8 个 prompt，逐个、批量、命中前缀缓存、naive 后端，都和 HF 逐 token 相同 | 是（找不到时跳过） |
| `test_server.py` | HTTP：流式/非流式一致、并发合并、n + seed、stop + usage、logprobs、echo、token id 作为 prompt、OpenAI 格式的错误、/health 和 /metrics、**真实 uvicorn 服务器上客户端断开会取消请求** | 是（只用分词器） |
| `test_cuda.py` | （仅 GPU）GPU 上贪心和 HF 一致、显存自动分配 KV 缓存、CUDA Graph 和普通模式逐 token 一致、真实模型 bf16 | 是 |

### 更大的模型：`scripts/verify_model.py`
测试只用 0.5B；更大的模型用这个脚本单独验证，三个指标：
1. **贪心生成是否完全一致**（8 个 prompt × 48 token）；
2. **teacher forcing 一致率**：把 HF 的 token 一个个喂给本引擎（真实的调度器、分页 decode、CUDA Graph），检查每个位置的 argmax 是否和 HF 相同。和 (1) 不同，一次翻转不会连锁影响后面；
3. **噪声基线**：HF 自己换一种注意力实现（eager vs sdpa）的一致率。

RTX 5070 实测：fp32 下 Qwen2.5-0.5B、Qwen2.5-1.5B、Qwen3-1.7B 都是 8/8 完全一致、100%；bf16 下本引擎 98.2–99.2%，HF 自己的噪声基线 96.1–99.2%（Qwen2.5-3B 的 fp32 放不进 12 GB 显存，只测了 bf16：98.2%，基线 99.2%）。表格见 README。

踩过的坑：一开始参考答案用 HF `generate(min_new_tokens=48)` 生成，结果有一条 fp32 也对不上——原来 `min_new_tokens` 会把 EOS 的分数压成 -inf，而我们这边用的是 `ignore_eos`（EOS 照常参与 argmax，只是不停止）。两边"规则"不一样。改成手写的纯贪心循环后就 100% 一致了。**对比之前，先确认两边算的是同一件事。**

### 为什么用随机小模型？
- 几 MB 大，几秒钟跑完，不用下载。
- 可以覆盖 Qwen3 等没有下载的架构。
- 随机初始化时故意把权重设得大一些（`initializer_range=0.2`），让 logits "尖"一点，避免两个 token 分数几乎相同、浮点误差导致 argmax 翻转。

### 一个真实踩过的坑：bf16 在多线程 CPU 上不是确定性的
开发时发现：**Hugging Face 自己**用 bfloat16 在 8 线程 CPU 上对同一个 prompt 做四次前向计算，最后一个位置的 logits 最大相差约 1.19；四次贪心生成里，有一次从第 7 个 token 开始和另外三次分叉。换成单线程 bf16，或者多线程 float32，四次结果完全相同。原因是多线程矩阵乘法里浮点加法的顺序不固定，bf16 精度又低（只有 8 位有效位），误差会被放大。

所以：**逐 token 对比的测试默认用 float32**（`NANOINFER_TEST_DTYPE` 可以改）。这是面试时很好的一个故事："怎么确认结果对不对，以及为什么低精度下不能要求逐位一致"。

另一个坑：Qwen 的 `generation_config.json` 默认带 `repetition_penalty=1.1`，HF `generate()` 会自动使用。对比时必须显式设成 1.0，否则"标准答案"本身就不是纯贪心。

## 2. 性能：离线吞吐（`benchmarks/bench_throughput.py`）

```bash
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct
```

三个场景（每个引擎在独立子进程里运行，显存里同时只有一个模型；计时前先用**同样形状**的输入预热，避免把 Triton 编译、cuBLAS 选算法的时间算进去——没预热时，第一次跑新模型的 B=1 首字延迟会多出约 0.4 s）：

1. **static（静态）**：B 个请求（B = 1 / 4 / 16），每个 128 token 的 prompt，生成 128 个 token（忽略 EOS）。所有 prompt 一样长，对 HF 最公平。3 次取中位数。
2. **mixed（混合长度）**：32 个请求，各自要生成 16–256 个不等的 token。HF 按 16 个一批静态处理；nano-infer 用连续批处理。只统计"有用的" token。
3. **prefix（共享前缀）**：16 个请求共享 512 token 的系统提示词，各自再加 32 token，生成 32 个 token；比较前缀缓存开/关。

消融（ablation）用命令行开关：`--enforce-eager`（关 CUDA Graph）、`--attn-backend sdpa`、`--no-chunked-prefill`、`--max-num-batched-tokens N`。`benchmarks/profile_step.py` 把一步 decode 拆成准备输入 / 前向 / lm_head / 采样四段计时。

**这三个场景（以及 `bench_serving.py`、`profile_step.py`）都写死了 `temperature=0`**，也就是全部走贪心。好处是和 HF 对比时"标准答案"唯一、可复现；代价是它们**完全不覆盖采样路径**——top-k / top-p / min-p、指数竞赛抽样、repetition / presence / frequency penalty 一个都不碰。所以改采样器时，这些 benchmark 不会有任何变化（这反过来是个有用的对照组），采样的开销要单独测，见 README 第 5 节。

## 3. 性能：在线服务（`benchmarks/bench_serving.py`）

离线吞吐只回答"一共多快"，线上用户感受到的是延迟。这个脚本自己启动服务器（或连接 `--url`），按泊松过程（Poisson，随机间隔）发出流式 HTTP 请求，统计：

- **TTFT**（time to first token）：发出请求 → 收到第一个 token
- **TPOT**（time per output token）：(最后一个 token 的时间 - 第一个 token 的时间) / (输出 token 数 - 1)，每个请求一个值
- **ITL**（inter-token latency）：相邻两个 token 之间的间隔，所有请求的所有间隔放在一起
- **E2E**：发出请求 → 收到最后一个 token

每项给出平均值和 p50 / p90 / p99，还会从服务器的 `/metrics` 读出"纯 decode 的 step"和"含 prefill 的 step"各自的耗时分位数。`--configs="default;--no-chunked-prefill"` 一次对比多种服务器配置。

**为什么看 p99 而不是平均值？** 分块预填充对平均 ITL 几乎没影响（大部分 step 都是纯 decode），它改善的是长尾：长 prompt 进来的那一步。实测见 README 第 4 节。

## 4. 注意
- **数字只对测量它的那台机器有效**。README 的 GPU 数字来自 RTX 5070，CPU 数字来自开发服务器，两者不能混用。
- 测之前看一眼 `nvidia-smi`，确认没有别的程序在用 GPU（本项目测试时只有桌面程序占约 2 GB、利用率 0%）。一边开着游戏一边测，实测拷贝带宽会从 528 GB/s 掉到约 211 GB/s，单次结果能差 ±13%。
- 和 HF 对比时两边用同样的数据类型（bf16）。

### 一个真实踩过的坑：显卡空闲不等于可以测了
2026-10-17 复测时遇到的情况：关掉游戏后显卡确实空闲了（839 MiB、利用率 2–3%、带宽 528 GB/s = 规格的 79%、无降频），但机器上还开着一批后台程序，**CPU 占用约 36%**。结果所有数字一致地低了约 9%。

两个教训：

1. **要看重复之间的离散度，不只看中位数。** 更早那次安静环境下的三次重复是 0.377 / 0.378 / 0.376 s（差 0.5%）；复测时同一个点是 0.410 / 0.411 / 0.411 s——同样很稳，但稳在一个更慢的值上。而第一次只跑 3 次重复时是 0.416 / 0.408 / 0.378 s（差 10%），最快的那次恰好等于安静环境的值。**离散度大说明被抢占了，此时"最快的一次"比中位数更接近真实值**；离散度小也不代表环境干净，只代表干扰稳定。
2. **CPU 被抢占会让倍数变好看，这是个陷阱。** batch=1 的 decode 瓶颈是 CPU 发 kernel 的开销（这正是 CUDA Graph 要消掉的东西），而 HF `generate()` 在这上面花的时间比本引擎多得多，所以 CPU 一紧张，**HF 掉得比本引擎更多**：复测时 Qwen2.5-1.5B 的 B=1 倍数从 3.20x "涨"到 3.75x。看到倍数无缘无故变好，先怀疑基线被拖慢了，而不是自己变快了。

判断瓶颈在 CPU 还是显存带宽，可以粗算"权重大小 × tok/s"：0.5B 在 B=1 时约 310 GB/s，而实测可用带宽是 528 GB/s，离上限还远，所以那一档是 CPU 开销主导的。
- 用随机 token 当 prompt（固定种子），所以输出内容没有意义，只看速度；`ignore_eos` 保证输出长度固定。
