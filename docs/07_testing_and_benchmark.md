# 07 · 怎么证明它是对的、有多快（Testing & Benchmark）

## 1. 正确性：和 Hugging Face 逐 token 对比

推理引擎最重要的是**算得对**。我们用 Hugging Face transformers 当"标准答案"：同一个 prompt，都用贪心解码（greedy），比较生成的 token 序列是否**每一个都相同**。

运行全部测试：

```bash
python -m pytest tests -v
```

| 测试文件 | 测什么 | 需要下载模型吗 |
|---|---|---|
| `test_block_manager.py` | 分配、释放、在块边界追加新块、LRU 淘汰顺序、重复释放检测 | 否 |
| `test_prefix_cache.py` | 哈希链、引用计数共享、复活、淘汰后忘记哈希、碰撞检查、decode 块登记；开/关前缀缓存输出相同 | 否 |
| `test_scheduler.py` | prefill 优先、新请求中途加入、序列数和 token 预算、EOS/最大长度/停止词、抢占、取消；**强制抢占后输出不变** | 否 |
| `test_sampler.py` | greedy、top-k、top-p、温度分布（统计检验）、batch 内混合设置、随机种子可复现 | 否 |
| `test_tiny_model_vs_hf.py` | 用**随机初始化的小 Qwen2 和 Qwen3**：权重逐个相等；prefill 和每一步 decode 的 logits 和 HF 误差 < 1e-4；两种注意力后端 × 三种块大小；批量贪心输出逐 token 相同；命中前缀缓存后仍相同；CUDA Graph 的静态补齐路径与普通路径一致 | 否（现场生成） |
| `test_qwen_vs_hf.py` | **真实 Qwen2.5-0.5B-Instruct**：8 个 prompt（中英文、对话和续写），逐个跑、批量跑、命中前缀缓存后跑、naive 后端跑，都和 HF 逐 token 相同 | 是（找不到时自动跳过） |
| `test_server.py` | HTTP 服务：普通/流式输出结果一致、8 个并发请求确实被合并成 batch、参数校验、中文和 emoji 的增量解码 | 是（只用分词器） |

### 为什么用随机小模型？
- 几 MB 大，几秒钟跑完，不用下载。
- 可以覆盖 Qwen3 等没有下载的架构。
- 随机初始化时故意把权重设得大一些（`initializer_range=0.2`），让 logits "尖"一点，避免两个 token 分数几乎相同、浮点误差导致 argmax 翻转。

### 一个真实踩过的坑：bf16 在多线程 CPU 上不是确定性的
开发时发现：**Hugging Face 自己**用 bfloat16 在 8 线程 CPU 上对同一个 prompt 做四次前向计算，最后一个位置的 logits 最大相差约 1.19；四次贪心生成里，有一次从第 7 个 token 开始和另外三次分叉。换成单线程 bf16，或者多线程 float32，四次结果完全相同。原因是多线程矩阵乘法里浮点加法的顺序不固定，bf16 精度又低（只有 8 位有效位），误差会被放大。

所以：**逐 token 对比的测试默认用 float32**（`NANOINFER_TEST_DTYPE` 可以改）。这是面试时很好的一个故事："怎么确认结果对不对，以及为什么低精度下不能要求逐位一致"。

另一个坑：Qwen 的 `generation_config.json` 默认带 `repetition_penalty=1.1`，HF `generate()` 会自动使用。对比时必须显式设成 1.0，否则"标准答案"本身就不是纯贪心。

## 2. 性能：和朴素的 HF generate 对比

```bash
python benchmarks/bench_throughput.py --model models/Qwen2.5-0.5B-Instruct
```

三个场景（每个引擎在独立子进程里运行，内存里同时只有一个模型）：

1. **static（静态）**：B 个请求（B = 1 / 4 / 16），每个 128 token 的 prompt，生成 128 个 token（忽略 EOS，保证长度一样）。这对 HF 最公平：所有 prompt 一样长，没有 padding 浪费。重复 3 次取中位数。
2. **mixed（混合长度）**：32 个请求，各自要生成 16–256 个不等的 token。HF 按 16 个一批静态处理（每批要等最长的那个）；nano-infer 用连续批处理。只统计"有用的" token。
3. **prefix（共享前缀）**：16 个请求共享一段 512 token 的系统提示词，各自再加 32 token 的问题，生成 32 个 token；比较前缀缓存开/关。

指标：
- **tok/s（吞吐量）**：生成的 token 总数 ÷ 总时间。
- **TTFT（首字延迟）**：请求到达 → 第一个 token 生成的时间。
- **latency（延迟）**：请求到达 → 完成的时间。

结果会打印成 Markdown 表格，并保存到 `benchmarks/results/<设备>.json`。

### 注意
- **数字只对测量它的那台机器有效**。README 里的数字是在开发用的云服务器 CPU 上测的；你的 Windows 电脑（尤其有 NVIDIA 显卡时）必须自己跑一遍，用你自己测到的数字。
- 开发机是共享的，其他程序也在用 CPU，所以数字有一定波动。
- 和 HF 对比时两边用同样的数据类型（bf16）、同样的注意力算子（SDPA）。
