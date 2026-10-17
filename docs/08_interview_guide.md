# 08 · 面试指南：怎么讲这个项目（Interview Guide）

> ⚠️ **最重要的一条**：简历上写了，面试官就会往深了问。**你必须真的读懂每一个文件，能脱稿画出数据流、能现场改代码**，才可以把它写上简历。建议的标准：不看代码，能在白板上写出 `BlockManager.allocate` 的逻辑和 decode 时 slot 的计算公式；能解释每个测试在验证什么。做不到之前，先别投。
>
> 另外要诚实：这是一个**学习项目**，参考了 vLLM / nano-vllm 的设计思想（论文和公开代码），但代码是自己写的。被问到时就这么说，不要夸大。

## 1. 电梯演讲（30 秒）

> "我从零用 PyTorch 写了一个小型的大模型推理引擎，叫 nano-infer。它自己实现了 Qwen 的模型结构，从官方权重加载，然后实现了 vLLM 的几个核心机制：分页 KV 缓存、连续批处理的调度器（包括缓存不够时的抢占）、基于块哈希的前缀缓存，还有一个兼容 OpenAI 接口的流式 HTTP 服务。正确性上，它的贪心输出和 Hugging Face transformers 逐 token 一致，有 70 个测试覆盖。性能上，在长度不一的并发请求场景下，吞吐量是朴素的 HF generate 的 2 倍；共享系统提示词时，首字延迟缩短到约十分之一（CPU 实测，换成你自己机器上的数字）。"

## 2. 两分钟版本（按这个顺序讲）

1. **动机**：推理成本是大模型落地的主要成本。decode 阶段是访存瓶颈，提高吞吐的关键是把更多请求凑进一个 batch，而 batch 大小被 KV 缓存的内存限制。
2. **分页 KV 缓存**：固定大小的块 + 块表，解决预留式分配的内存浪费和碎片。
3. **连续批处理**：每一步重新调度，完成的立即离开、新的立即加入；缓存不够时抢占最新的请求，之后重算。
4. **前缀缓存**：哈希链识别相同前缀，引用计数共享块，LRU 淘汰。
5. **验证**：和 HF 逐 token 对比；单元测试覆盖块管理器、调度器、采样；发现了 bf16 多线程不确定性的问题，于是用 fp32 做精确比对。
6. **结果**：报你**自己机器上**测到的数字。
7. **取舍和下一步**：注意力用 PyTorch gather + SDPA 而不是自定义 kernel（可读、能在 CPU 跑，但多了一次拷贝）；没有做 chunked prefill、张量并行、量化；下一步可以写一个 Triton paged-attention kernel。

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
静态批处理一整批一起开始一起结束，短请求要等长请求，空位浪费。连续批处理在每一步（每生成一个 token）重新组 batch，完成的立即释放资源，等待的立即加入。我的 benchmark 里，32 个输出长度 16–256 不等的请求，就是在测这个差别。

**Q7：KV 缓存满了怎么办？你怎么实现抢占？**
decode 前检查每个请求是否需要新块。如果没有空块，就抢占 running 队列里最后加入的请求：释放它所有块，放回 waiting 队首，以后重新 prefill（prompt + 已生成的 token）。我用一个测试强制发生抢占，验证输出和不抢占时逐 token 相同。另一种方案是 swap 到 CPU 内存。

**Q8：为什么选最后加入的请求作为牺牲者？**
它通常已生成的 token 最少，重算代价最小；也符合先来先服务（FCFS）的公平性，老请求先完成。

**Q9：调度为什么 prefill 优先？有什么缺点？**
优先让新请求拿到第一个 token，降低 TTFT。缺点是长 prompt 的 prefill 步会让正在 decode 的请求卡一下（TPOT 抖动）。改进是 chunked prefill：把长 prompt 切成小段，和 decode 放在同一个 batch 里。

**Q10：max_num_batched_tokens 和 max_num_seqs 分别控制什么？**
前者限制一次前向的 token 总数（控制 prefill 的计算量和激活内存），后者限制并发序列数（控制 decode batch 大小和 KV 缓存占用）。

### 前缀缓存

**Q11：前缀缓存怎么判断两个块可以共享？**
对写满的块算哈希，哈希里包含前一个块的哈希，形成哈希链，所以哈希相同意味着整个前缀相同。命中后再比较 token 防止碰撞。共享用引用计数，计数归零才进入空闲队列。

**Q12：为什么不能只哈希当前块的 token？**
因为 K/V 依赖前面所有 token（注意力看了整个上文），同样 16 个 token 在不同上文里 K/V 不同。

**Q13：如果整个 prompt 都命中缓存会怎样？**
那就没有 token 需要计算，也就拿不到 logits 来采样第一个输出 token。我的处理是：命中 prompt 的最后一块时视为未命中，重算这一块。

**Q14：释放的块里的缓存还能用吗？**
能。引用计数归零的块进入 LRU 空闲队列，但保留内容和哈希；被新请求命中就复活。只有被分配给其他内容时才删掉它的哈希。释放时倒序释放，让共享的开头留得更久。

### 实现与工程

**Q15：你的注意力怎么实现的？和 vLLM 有什么不同？**
每层先用 `index_copy_` 把新 token 的 K/V 写进槽位，然后按块表用 `index_select` 把需要的 K/V gather 成连续张量，再调用 PyTorch 的 SDPA。vLLM 用自定义 CUDA/Triton kernel 直接在分页内存上计算，省掉 gather 这次拷贝。我这样做是为了可读、能在 CPU 和任何 GPU 上跑；代价是多一次内存拷贝，长上下文时更明显。

**Q16：怎么保证你的实现是对的？**
三层：(1) 单元测试覆盖块管理器、调度器、采样器；(2) 随机初始化的小 Qwen2/Qwen3 模型上，prefill 和每一步 decode 的 logits 都和 HF 比对（误差 < 1e-4）；(3) 真实 Qwen2.5-0.5B 上贪心输出和 HF 逐 token 一致，包括批量、前缀命中、两种注意力后端。还做过"变异测试"：故意改坏 mask 或前缀逻辑，确认测试会失败。

**Q17：为什么测试用 float32？**
我发现 bf16 在多线程 CPU 上不是确定性的：HF 自己同一个输入跑四次，logits 最大相差约 1.19，贪心输出会分叉。原因是并行归约的加法顺序不固定、bf16 有效位很少。所以逐 token 对比用 fp32，bf16 只用来测速度。

**Q18：CUDA Graph 是什么？为什么能加速？**
把一步 decode 里几百次小 kernel 启动录制成一个图，一次回放，省掉 CPU 端的启动开销；小模型、小 batch 时效果最明显。要求形状固定，所以要按几个 batch 大小分别录制、补齐，注意力 gather 固定长度。我实现了但默认关闭，因为开发机没有 GPU 没测过（可以诚实说），它依赖的静态补齐逻辑在 CPU 上测过。

**Q19：HTTP 服务怎么和引擎配合？**
引擎不是线程安全的，所以一个后台线程独占引擎、循环调用 step()；FastAPI 的异步处理函数通过队列提交请求，每个请求一个 asyncio.Queue 接收 token。多个 HTTP 请求同时在飞，就自然组成一个连续批。客户端断开时取消请求、释放 KV 块。增量解码时，如果文本以替换字符结尾（多字节字符被拆开），就等下一个 token。

**Q20：如果要支持更大的模型，你会怎么做？**
张量并行（tensor parallelism，把每层的矩阵按头切到多张卡上，用 all-reduce 合并）；量化（int8/int4 权重，减少显存和带宽）；自定义 paged attention kernel；chunked prefill；投机解码（speculative decoding）。

### 关于数字的追问

**Q21：吞吐量提升来自哪里？**
（结合你自己的数字回答）batch=1 时两边差不多，因为都卡在读权重上；batch 变大后，我的引擎打包（无 padding）、decode 完全向量化。混合长度场景主要来自连续批处理；共享前缀场景主要来自前缀缓存跳过了重复的 prefill。

**Q22：为什么 batch=1 时你没有比 HF 快多少？**
batch=1 的 decode 瓶颈是每步读一遍约 1GB 权重，两边都一样；我的引擎还多了调度和 gather 的开销。推理引擎的优势本来就在并发场景。

## 4. 简历写法（英文，按实际测量结果填写）

> 只写你**自己机器上真实测到**的数字，并能解释它是怎么测的。下面的数字来自开发服务器 CPU（见 README），在你的电脑上重新跑一遍后替换。

开发服务器 CPU 上的实测结果（Intel Xeon 8 vCPU、bf16、6 线程，详见 README）可以支撑下面这些写法。三选二即可：

**选项 A（重正确性和工程）**
> Built **nano-infer**, a from-scratch LLM inference engine in PyTorch (~1.7k lines) for Qwen2.5-0.5B, implementing a paged KV cache, a continuous-batching scheduler with preemption, and hash-based prefix caching; greedy outputs match Hugging Face Transformers token for token, verified by 70 pytest tests.

**选项 B（重吞吐）**
> Doubled serving throughput (115.6 → 233.1 tokens/s) and cut mean request latency by 64% versus Hugging Face `generate()` on a 32-request mixed-length CPU benchmark by replacing static batching with iteration-level continuous batching over a paged KV cache.

**选项 C（重前缀缓存和服务）**
> Implemented prefix caching with chained block hashes, reference counting and LRU eviction, cutting prefill compute by 94% and time-to-first-token by 11x for requests sharing a 512-token system prompt; served the engine through an OpenAI-compatible streaming API (FastAPI).

注意：
- 这些数字是 **CPU** 上测的，写的时候要能说清楚是 CPU、什么模型、什么负载。在你自己的 GPU 上重测后，**用你自己的数字替换**，并相应改成 "on an RTX xxxx GPU"。
- **不要**写"比 vLLM 快"或者"生产级"。静态批处理场景下它和 HF 基本持平（快 6%–16%），如果被问到要如实说。
- 被问"这些是你自己写的吗"：是自己写的，设计思路参考了 vLLM 论文和 nano-vllm。

## 5. 面试前的自检清单

- [ ] 能画出 prefill 和 decode 时 `input_ids / positions / slot_mapping` 的具体内容（拿 2 个序列举例）
- [ ] 能手算：block_size=16，块表 [7, 3]，位置 20 的 slot 是多少（答案：52）
- [ ] 能讲清 `allocate` 里 `cache_miss` 为什么一旦为真就一直为真
- [ ] 能讲清 `can_append` 里 `(num_tokens - 1) % block_size == 0` 的含义
- [ ] 能解释 decode attention 里 mask 的形状 `[B, 1, 1, max_len]` 为什么能广播
- [ ] 在自己电脑上跑通所有测试和 benchmark，记下自己的数字
- [ ] 至少自己改一个功能（比如加 repetition penalty 或 stop 字符串），证明你能动这份代码
