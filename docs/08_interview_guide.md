# 08 · 面试指南：怎么讲这个项目（Interview Guide）

> ⚠️ **最重要的一条**：简历上写了，面试官就会往深了问。**你必须真的读懂每一个文件，能脱稿画出数据流、能现场改代码**，才可以把它写上简历。建议的标准：不看代码，能在白板上写出 `BlockManager.allocate` 的逻辑和 decode 时 slot 的计算公式；能解释每个测试在验证什么。做不到之前，先别投。
>
> 另外要诚实：这是一个**学习项目**，参考了 vLLM / nano-vllm 的设计思想（论文和公开代码），但代码是自己写的。被问到时就这么说，不要夸大。

## 1. 电梯演讲（30 秒）

> "我从零写了一个大模型推理引擎 nano-infer，大约 3,000 行 Python 加 Triton。它自己实现了 Qwen2/Qwen3 的模型结构，实现了 vLLM 的核心机制：分页 KV 缓存、按 token 预算调度的连续批处理（支持分块预填充，decode 和 prefill 混在一个 batch 里）、基于块哈希的前缀缓存，还有 OpenAI 兼容的流式服务。GPU 这边我自己用 Triton 写了分页注意力 kernel（decode 用 flash-decoding）和几个融合算子，再加上 CUDA Graph。在 RTX 5070 上，Qwen2.5-0.5B 的吞吐量是 Hugging Face generate 的 6 到 7 倍，长度不一的并发请求下是 11 倍；正确性上，fp32 下和 HF 逐 token 一致，从 0.5B 到 1.7B 都验证过。"

## 2. 两分钟版本（按这个顺序讲）

1. **动机**：decode 阶段是访存瓶颈，提高吞吐的关键是把更多请求凑进一个 batch，而 batch 大小受 KV 缓存显存限制；小模型时还有 CPU 发 kernel 的开销。
2. **内存**：分页 KV 缓存（固定大小的块 + 块表），前缀缓存（哈希链 + 引用计数 + LRU）。
3. **调度**：每一步在 token 预算内决定每个序列算几个 token——decode 1 个、长 prompt 一段。所以 decode 永远不会被长 prompt 卡住；缓存不够时抢占最新的请求、之后重算（前缀缓存让重算很便宜）。
4. **GPU**：Triton 分页注意力（不需要先把 K/V 拷成连续张量）、融合算子（减少 kernel 数）、CUDA Graph（一步一次发射）、每步只有一次主机到显卡的拷贝、采样器中间没有 GPU 同步。
5. **验证**：三层——单元测试、随机小模型逐步 logits 对比、真实模型逐 token 对比；更大的模型用 teacher forcing 一致率，并用"HF 自己换注意力实现"的一致率做噪声基线。
6. **结果**：RTX 5070 的数字（见第 4 节），能说清楚每一项优化贡献多少（消融）。
7. **取舍和下一步**：单卡、没有量化和投机解码、Triton kernel 没有 autotune。

## 3. 高频面试题与参考答案

### 基础概念

**Q1：prefill 和 decode 有什么区别？各自的瓶颈是什么？**
prefill 一次处理整个 prompt，大量 token 并行做矩阵乘法，是计算密集（compute-bound），决定首字延迟 TTFT。decode 每步每个请求只处理 1 个 token，但要读一遍全部权重和该请求的 KV 缓存，是访存密集（memory-bound），决定每个 token 的间隔 TPOT。所以 decode 要靠批处理把读权重的成本分摊给多个请求。

**Q2：KV 缓存为什么能用？存的是什么？**
因果注意力里，旧 token 的 K、V 不受新 token 影响，所以算一次存起来，以后每步只算新 token 的 Q/K/V。存的是每一层、每个 KV 头的 K 和 V 向量。Qwen2.5-0.5B 每个 token 是 2×24×2×64×2 字节 = 12 KB。

**Q3：为什么要分页？解决了什么问题？**
按最大长度连续预留会造成内部浪费（预留了没用）和外部碎片（空洞塞不进新请求），还不能共享。分页后每个请求最多浪费不到一个块，所有块等大没有外部碎片，块表天然支持共享。代价是注意力要按块表间接寻址。

**Q4：block_size 怎么选？大了小了各有什么问题？**
小：浪费更少、前缀共享粒度更细，但块表更长、间接寻址和 kernel 效率变差，哈希计算更频繁。大：反之，且最后一块的平均浪费是 block_size/2。vLLM 常用 16。本项目的测试用了 1、4、8、16 几种大小都验证正确。

**Q5：GQA 是什么，对推理有什么好处？**
多个 Q 头共享一组 KV 头。Qwen2.5-0.5B 是 14 个 Q 头对 2 个 KV 头，KV 缓存缩小 7 倍，同样内存能装更多请求或更长上下文。我在 decode 里把共享 KV 的 Q 头重排成"多行查询"，不需要 repeat K/V。

### 调度与批处理

**Q6：连续批处理和静态批处理的区别？**
静态批处理一整批一起开始一起结束，短请求要等长请求，空位浪费。连续批处理在每一步（每生成一个 token）重新组 batch，完成的立即释放资源，等待的立即加入。我的 benchmark 里，32 个输出长度 16–256 不等的请求，就是在测这个差别：RTX 5070 上 0.5B 模型 HF 409.2 tok/s、平均延迟 6.50 s，nano-infer 4656.3 tok/s、0.37 s。

**Q7：KV 缓存满了怎么办？你怎么实现抢占？**
调度时给每个 running 序列分配这一步需要的块；不够就抢占 running 队列里最后加入的请求：释放它所有块，放回 waiting 队首，以后重新 prefill（prompt + 已生成的 token）。它的块内容和哈希还留在前缀缓存里，所以重算大多是缓存命中。另外两条规则防止抢占风暴：发生抢占的那一步不接纳新请求；接纳新请求时保留 1% 的空闲块（水位线）。测试里强制抢占，验证输出和不抢占时逐 token 相同。另一种方案是 swap 到 CPU 内存（没做）。

**Q8：为什么选最后加入的请求作为牺牲者？**
它通常已生成的 token 最少，重算代价最小；也符合先来先服务（FCFS）的公平性，老请求先完成。

**Q9：prefill 优先有什么问题？分块预填充怎么解决？**
v0.1 是"prefill 优先"：有新请求就整步做 prefill，正在 decode 的请求全部暂停，一个长 prompt 会造成明显的 token 间隔长尾。v0.2 改成按 token 预算调度：每一步先给 running 的序列排（decode 1 个 token，没算完的 prompt 拿下一段），剩下的预算再给新请求，放不下的 prompt 下一步接着算。实测（RTX 5070，Qwen2.5-1.5B，1,900–3,800 token 的 prompt，每秒 2.5 个请求）：不分块 ITL p99 是 190.7 ms，预算 512 时降到 37.9 ms；代价是首字延迟 p99 从 594 ms 升到 856 ms。所以预算是 TTFT 和 ITL 之间的取舍。

**Q10：max_num_batched_tokens 和 max_num_seqs 分别控制什么？**
前者是每一步的 token 预算：限制一次前向的计算量和激活内存，也决定了长 prompt 被切成多大的段（默认 2048）。后者限制并发序列数，控制 decode batch 大小和 KV 缓存占用（CUDA Graph 也按它录制）。

**Q10b：分块预填充和前缀缓存一起用，有什么要小心的？**
块的哈希只能在"这一步确实会被写满"时登记，否则另一个请求可能命中一个 K/V 还没写进去的块。我的 `allocate_slots` 只登记这一步算完的块，有专门的测试。同一步里共享是安全的，因为每一层都是先把整个 batch 的 K/V 写进缓存再算注意力。

### 前缀缓存

**Q11：前缀缓存怎么判断两个块可以共享？**
对写满的块算哈希，哈希里包含前一个块的哈希，形成哈希链，所以哈希相同意味着整个前缀相同。命中后再比较 token 防止碰撞。共享用引用计数，计数归零才进入空闲队列。

**Q12：为什么不能只哈希当前块的 token？**
因为 K/V 依赖前面所有 token（注意力看了整个上文），同样 16 个 token 在不同上文里 K/V 不同。

**Q13：如果整个 prompt 都命中缓存会怎样？**
那就没有 token 需要计算，也就拿不到 logits 来采样第一个输出 token。我的处理是：查缓存时最多只查前 `(num_tokens - 1) // block_size` 个块，保证最后一个 token 一定会被重新计算。

**Q14：释放的块里的缓存还能用吗？**
能。引用计数归零的块进入 LRU 空闲队列，但保留内容和哈希；被新请求命中就复活。只有被分配给其他内容时才删掉它的哈希。释放时倒序释放，让共享的开头留得更久。

### 实现与工程

**Q15：你的注意力怎么实现的？和 vLLM 有什么不同？**
GPU 上用我自己写的两个 Triton kernel。decode kernel：每个程序负责（序列，Q 头），按块表一块块读 K/V，用 online softmax；batch 小的时候程序太少、GPU 吃不饱，所以用 flash-decoding 把长上下文切成最多 16 段并行算，再用一个小 kernel 按 `exp(m_i - M)` 加权合并。循环上界来自运行时的 context_lens，所以同一次启动能被 CUDA Graph 录制。prefill kernel：varlen，一次启动处理所有长度不同的序列，因果掩码按绝对位置，所以普通 prefill、前缀命中、分块的后续段是同一个 kernel。CPU 上退回 PyTorch：按块表 index_select 拷成连续张量再调 SDPA。和 vLLM 比：思路一样，但没有 autotune、块大小写死，也没和 FlashAttention/FlashInfer 比过。消融：把注意力换回 SDPA，B=16 从 5573.7 降到 3270.0 tok/s。

**Q16：怎么保证你的实现是对的？**
四层：(1) 单元测试覆盖块管理器、调度器（混合批、分块、抢占、水位线）、采样器；(2) 随机初始化的小 Qwen2/Qwen3 上，prefill、decode、混合批、分块时每一步的 logits 都和 HF 比对（误差 < 1e-4）；(3) Triton kernel 和 PyTorch 参考实现逐元素对比（CPU 上用 Triton 解释器，GPU 上跑真 kernel，含 bf16/fp16）；(4) 真实模型上贪心输出和 HF 逐 token 一致，包括批量、前缀命中、各种注意力后端，更大的模型用 `verify_model.py`。RTX 5070 上 198 个测试全部通过（开发机 CPU 上是 121 通过 + 4 个 GPU 测试跳过，那一轮在加采样器测试之前）。

**Q17：为什么测试用 float32？bf16 下怎么证明是对的？**
bf16 只有 8 位有效位，不同 kernel 的加法顺序不同，舍入就不同，遇到两个候选分数几乎一样时 argmax 会翻转，之后整段生成都不同。所以逐 token 对比用 fp32：RTX 5070 上 0.5B、1.5B、Qwen3-1.7B 都是 8/8 完全一致。bf16 下我用 teacher forcing 一致率（每个位置喂 HF 的 token，比较 argmax，一次翻转不会连锁）：98.2–99.2%；再拿 HF 自己 eager 和 sdpa 两种注意力实现互相比，也只有 96.1–99.2%，说明这个差距是精度噪声，不是 bug。

**Q18：CUDA Graph 是什么？为什么能加速？有什么限制？**
把一步 decode 的几百个 kernel 启动录制成一张图，一次回放，省掉 CPU 端的启动开销；小模型、小 batch 最明显。要求形状和地址固定：所以录制 1、2、4、8、16、24、32、48、64 这些 batch 大小，实际 batch 向上补齐，补齐的行写进一个草稿块；所有输入放在静态缓冲区，每步拷一次。混合批（含 prefill）不走图。实测：关掉 CUDA Graph，0.5B B=1 从 341.6 降到 207.8 tok/s，B=16 从 5573.7 降到 3023.7。

**Q18b：融合算子做了什么？为什么不用 torch.compile？**
四个 Triton kernel：残差加法 + RMSNorm、RMSNorm、RoPE + 写 KV 缓存、SiLU×mul；配合合并的 qkv 和 gate_up 投影，每层的 kernel 数大大减少。B=1 一步 decode 从 3.83 ms 降到 2.95 ms。舍入顺序刻意和 HF 一致（fp32 归一化 → 转回 bf16 → 乘权重）。torch.compile 在这块卡上试过能用，但我想要每一行都能解释清楚的代码。

**Q19：HTTP 服务怎么和引擎配合？**
引擎不是线程安全的，所以一个后台线程独占引擎、循环调用 step()；FastAPI 的异步处理函数通过队列提交请求，每个请求一个 asyncio.Queue 接收 token。多个 HTTP 请求同时在飞，就自然组成一个连续批。客户端断开时取消请求、释放 KV 块（用真实 uvicorn 服务器测过）。增量解码用 prefix_offset / read_offset 窗口：只解码最近几个 token，如果结尾是替换字符（多字节字符被拆开）就等下一个 token；stop 字符串可能跨 token，所以流式输出时会扣住可能是 stop 开头的那一小段文字。/metrics 输出 Prometheus 指标，包括 step 耗时分位数。

**Q20：如果要支持更大的模型，你会怎么做？**
张量并行（按头切分注意力、按列/行切分 MLP，用 all-reduce 合并）；权重量化（int8/int4/FP8，decode 是带宽瓶颈，权重字节数减半吞吐几乎翻倍）；投机解码（小模型猜、大模型一次验证多个 token）；swap 式抢占。

### 关于数字的追问

**Q21：吞吐量提升来自哪里？**
三类：(1) 调度——连续批处理让短请求做完就走，混合长度场景 0.5B 下 HF 409.2 vs 4656.3 tok/s；(2) 少算——前缀缓存把共享 prompt 的 prefill 从 8,704 token 降到 512；采样只在"候选窗口"里做，不扫整个 152k 词表，batch 64 下按 Qwen 推荐的 `top_p=0.8, top_k=20` 采样时一步 decode 从 8.30 ms 降到 3.95 ms（-52%），惩罚改到 `[rows, history]` 上算之后再加 `repetition_penalty` 是 10.62 → 4.74 ms（见 README 第 5 节）；(3) 少开销——CUDA Graph、融合算子、Triton 注意力，消融里每项都能看到（见 README 第 2 节）。

**Q22：为什么模型越大，比 HF 快得越少？**
batch=1 的 decode 每个 token 要读一遍全部权重。用"权重大小 × tok/s"估算：0.5B 约 340 GB/s，1.5B 约 500 GB/s，3B 约 550 GB/s，RTX 5070 的带宽是 672 GB/s。小模型时 HF 主要输在 kernel 启动开销上，这部分我们省掉了；大模型两边都接近带宽上限，能省的比例就小了（B=16：0.5B 6.4x，1.5B 3.2x，3B 2.3x）。要再快就得减少读的字节数（量化）或每次读权重多产出 token（更大的 batch、投机解码）。

**Q23：你的数字可信吗？怎么测的？**
每个引擎在独立进程里跑，计时前用同样形状预热（我发现不预热的话，新模型第一次的 B=1 首字延迟会多出约 0.4 s，是 kernel 编译/选算法），静态场景 3 次取中位数，测前看 nvidia-smi 确认 GPU 空闲。原始 JSON 和完整日志都在仓库里。

## 4. 简历写法（英文，全部是 RTX 5070 上实测的数字）

> 只写你**能解释是怎么测的**数字。下面每个数字都能在 README 和 `benchmarks/results/` 里找到出处。三选二即可。

**选项 A（系统 + 正确性）**
> Built **nano-infer**, a from-scratch LLM inference engine (~3k lines of Python/Triton) for Qwen2/Qwen3 with a paged KV cache, a token-budget continuous-batching scheduler with chunked prefill and preemption, and hash-based prefix caching; greedy outputs match Hugging Face Transformers token for token in fp32 on 0.5B–1.7B models, covered by 125 pytest tests.

**选项 B（GPU 性能）**
> Wrote Triton GPU kernels for paged attention (flash-decoding decode, varlen paged prefill) and fused RMSNorm/RoPE/SwiGLU ops and added CUDA-graph decoding, reaching **5.9–7.1x the throughput of Hugging Face `generate()`** on Qwen2.5-0.5B (5,615 vs 878 tokens/s at batch 16) and **11.4x on a mixed-length workload** on an RTX 5070.

**选项 C（服务 + 延迟）**
> Served the engine through an OpenAI-compatible streaming API (FastAPI/SSE) and built a Poisson load generator; chunked prefill cut p99 inter-token latency by 80% (190.7 → 37.9 ms) under long-prompt traffic on Qwen2.5-1.5B, and prefix caching cut mean time-to-first-token by 45% (61.7 → 34.0 ms) for requests sharing a 768-token system prompt.

注意：
- 一定写清楚 **"on an RTX 5070"**、模型大小和负载。不要写"比 vLLM 快"或"生产级"。
- 被问到大模型：1.5B / 3B 上 B=16 是 HF 的 3.2x / 2.3x，能用带宽上限解释为什么变小（Q22）。
- 被问"这些是你自己写的吗"：是自己写的，设计参考了 vLLM、Sarathi-Serve、FlashAttention / flash-decoding 的论文和 nano-vllm。

## 5. 面试前的自检清单

- [ ] 能画出 prefill 和 decode 时 `input_ids / positions / slot_mapping` 的具体内容（拿 2 个序列举例）
- [ ] 能手算：block_size=16，块表 [7, 3]，位置 20 的 slot 是多少（答案：52）
- [ ] 能讲清 `get_cached_prefix` 为什么只查前 `(num_tokens - 1) // block_size` 个块
- [ ] 能讲清 `allocate_slots` 为什么只登记"这一步会被写满"的块的哈希
- [ ] 能手推 flash-decoding 的合并公式 `Σ exp(m_i - M)·acc_i / Σ exp(m_i - M)·l_i`
- [ ] 能讲清 CUDA Graph 为什么要补齐 batch、补齐的行写到哪里
- [ ] 能解释 PyTorch 后端 decode attention 里 mask 的形状 `[B, 1, 1, max_len]` 为什么能广播
- [ ] 在自己电脑上跑通所有测试和 benchmark，记下自己的数字
- [ ] 至少自己改一个功能（比如给 Triton prefill kernel 加 autotune，或加一个新的采样参数），证明你能动这份代码
