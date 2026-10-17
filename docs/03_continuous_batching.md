# 03 · 连续批处理与调度器（Continuous Batching & Scheduler）

## 1. 为什么要批处理（Batching）

上一篇说过：decode 每一步都要把整个模型的权重（约 1 GB）从内存读一遍，而真正的计算很少。如果一次同时给 16 个请求各算一个 token，权重还是只读一遍，**吞吐量（throughput）几乎翻 16 倍**，每一步只慢一点点。

## 2. 静态批处理（Static Batching）的问题

Hugging Face `model.generate(batch)` 是**静态批处理**：

- 凑齐一批（比如 16 个），一起开始。
- **必须等这一批里最长的那个写完**，整批才结束，下一批才能开始。

**比喻：公交车**。一车 16 个人，有人坐 1 站，有人坐 20 站，可车要开到最后一个人下车才能回去接新乘客。中途下车的人留下的空座位一路空着。

更糟的是 padding（填充）：长短不一的 prompt 要补齐成一样长，补的部分也在白白计算。

## 3. 连续批处理（Continuous Batching）

也叫 **iteration-level scheduling（按步调度）**，出自 Orca 论文（OSDI 2022），vLLM 等系统都在用：

> **每生成一个 token（一步，step），就重新决定一次 batch 里有谁。**

- 有请求写完了？立刻离场，释放它的 KV 块。
- 有新请求在排队？只要有空位和空闲块，下一步就让它进场。

**比喻：旋转寿司 / 出租车车队**。谁吃完谁走，空位马上有新客人坐下，座位一直是满的。

## 4. 本项目的调度器怎么工作

代码：`nanoinfer/engine/scheduler.py`，核心就是 `schedule()` 一个函数。

### 两个队列
- `waiting`（等待队列）：还没有 KV 缓存的请求（新来的，或被抢占的）。
- `running`（运行队列）：已经有 KV 块的请求——可能正在 decode，也可能长 prompt 还没算完（分块预填充中）。

### 一个统一的视角：每个序列"还差几个 token 没算"
v0.2 的调度器不再区分"prefill 步"和"decode 步"。每个序列只记两个数：

- `num_tokens`：它现在一共有多少 token（prompt + 已生成的）
- `num_computed_tokens`：其中多少个的 K/V 已经在缓存里了

两者之差就是"还没算的 token"。刚进来的请求差一整个 prompt；正在 decode 的请求差 1 个（刚采样出来的那个）。调度器每一步做的事情只有一件：**在 token 预算 `max_num_batched_tokens`（默认 2048）之内，决定这一步给每个序列算几个 token**。

```
schedule():
  budget = max_num_batched_tokens
  1. 先排 running 里的序列（最老的先）：
       n = min(还差的 token 数, budget)      # decode 的序列 n = 1；没算完的长 prompt 拿下一段
       块不够？→ 抢占 running 里最新的那个，直到够为止
       分配块，budget -= n
  2. 再排 waiting 里的序列（先来先服务 FCFS），条件：
       - 预算还有剩、序列数 < max_num_seqs、这一步没有发生抢占
       - 查前缀缓存：开头命中的块不用算
       - 剩下的 prompt 能放多少放多少（放不下的部分下一步继续——这就是分块预填充）
       - 进场后至少还要剩 kv_watermark（1%）的空闲块，给正在跑的序列留成长空间
  返回：这一步的序列列表（decode 的排前面），每个序列这一步要算的 token 数
```

### 分块预填充（Chunked Prefill）和混合批（mixed batch）
旧版本（v0.1）是"prefill 优先"：只要有新请求就整步做 prefill，正在 decode 的请求全部暂停。一个 4000 token 的 prompt 进来，所有人都要卡一下——表现为 **token 间隔（ITL）的长尾**。

现在一个 step 里可以**同时**有 decode 的 token 和 prefill 的一段（chunk）。长 prompt 被切成每段最多"剩余预算"大小，分几步算完；decode 的请求每一步都照常前进。这就是 vLLM v1 和 Sarathi-Serve 论文的做法。好处：

1. decode 永远不会被长 prompt 卡住，ITL 稳定；
2. 每步的 token 数有上限，激活内存（activation memory）可控，一个 10 万 token 的 prompt 也不会撑爆显存；
3. 线性层在一次前向里同时处理 decode 和 prefill 的 token，GPU 吃得更饱。

实现上的关键：**只有最后一段算完的那一步，才需要采样**（`SchedulerOutput.sampling_seqs`）。中间段只是把 K/V 写进缓存。注意力也必须支持"query 是 prompt 的中间一段，key 是从开头到这一段末尾"——这和"前缀缓存命中后只算剩下的部分"是同一个问题，所以两者共用同一条代码路径（varlen prefill，见 05 篇）。

`enable_chunked_prefill=False`（命令行 `--no-chunked-prefill`）可以关掉它做对比：此时预算自动放大到 `max_model_len`，整个 prompt 必须一次算完。

### 一个 step 的完整流程
`nanoinfer/engine/llm_engine.py` 的 `LLM.step()`：

```python
out = scheduler.schedule()                         # 1. 选人 + 决定每人算几个 token
token_ids, logprobs = runner.run(out)              # 2. 一次前向 + 采样（只为算到末尾的序列采样）
scheduler.postprocess(out, token_ids)              # 3. 追加 token、检查停止条件、结束的释放块
```

`LLM.generate()` 就是不停地调用 `step()` 直到所有请求结束。

### 打包（packing），不是填充（padding）
一个 batch 里的所有新 token 被首尾相接排成一维 `[T]`，用 `cu_seqlens`（累计长度）记录每个序列的起止位置。例如 3 个 prompt 长度 5、3、7：

```
tokens:      [a a a a a | b b b | c c c c c c c]
cu_seqlens:  [0,         5,      8,             15]
```

线性层（矩阵乘法）不在乎 token 属于哪个序列，直接一次算完 15 个；只有注意力需要按 `cu_seqlens` 分开算。混合批里，decode 的序列（每个 1 个 token）排在最前面，后面是各个 prefill 段，注意力分两条路径分别处理。这样一个 padding 都没有。

## 5. 抢占（Preemption）：缓存满了怎么办

请求越写越长，需要的块越来越多。某一步发现空块用完了，怎么办？

本项目的策略（和 vLLM 默认一致）是**重算式抢占（recompute preemption）**：

1. 选**最后进场**的请求作为"牺牲者"（victim）。
2. 释放它的所有块，把它放回 `waiting` 队列的**最前面**。
3. 以后有空间了，它重新做一次 prefill——这次的"prompt"是它原来的 prompt **加上已经生成的所有 token**，然后接着往下写。

因为贪心解码的结果是确定的，被抢占的请求最终输出和没被抢占时**一模一样**。测试 `tests/test_scheduler.py::test_engine_preemption_gives_same_greedy_output` 专门验证了这一点：故意把 KV 缓存设得很小，强制发生抢占，输出仍和大缓存时逐 token 相同。

另一种策略是**换出（swap）**：把被抢占请求的 KV 块拷到 CPU 内存，回来时再拷回去。省计算但费带宽，实现也更复杂。

还有一个巧妙之处：被抢占的请求释放的块里内容还在，并且带着哈希。如果在被覆盖之前它就重新进场，**前缀缓存**能把这些块直接"复活"，重算几乎不花时间（`test_scheduler.py` 里分别测了开/关前缀缓存时的抢占）。

另外两条防止"抢占风暴"的规则：发生抢占的那一步不再接纳新请求（内存已经很紧）；接纳新请求时保留 `kv_watermark` 比例的空闲块。

## 6. 可调参数（`EngineConfig`）

| 参数 | 默认 | 含义 |
|---|---|---|
| `max_num_seqs` | 64 | 一个 batch 最多几个序列 |
| `max_num_batched_tokens` | 2048（关闭分块时 = `max_model_len`） | 一次前向最多处理多少 token |
| `enable_chunked_prefill` | True | 长 prompt 分段算、和 decode 混在一个 batch |
| `kv_watermark` | 0.01 | 新请求进场后至少保留多少比例的空闲块 |
| `max_model_len` | 4096 | 单个请求 prompt + 输出的最大长度 |
| `block_size` | 16 | 每块装几个 token |
| `num_kvcache_blocks` | 自动 | 块总数；GPU 上按剩余显存自动算，CPU 上默认用 0.5 GB |

## 7. 实测效果
- `benchmarks/bench_throughput.py` 的 "mixed" 场景：32 个请求，每个要生成 16 到 256 个不等的 token。HF 按 16 个一批静态处理，必须等每批最长的那个；本项目用连续批处理。
- `benchmarks/bench_serving.py`：模拟真实线上流量（泊松到达的 HTTP 流式请求），测 TTFT / TPOT / ITL 的 p50/p90/p99。用 `--configs="default;--no-chunked-prefill"` 可以直接对比分块预填充开和关。

实测数字（RTX 5070）见 README。
