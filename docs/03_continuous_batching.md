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
- `running`（运行队列）：已经有 KV 块、正在生成的请求。

### 每一步的决策
```
schedule():
  1. 先看能不能做 prefill：
       从 waiting 队头开始，只要
         - batch 里序列数 < max_num_seqs
         - 本步 token 总数 ≤ max_num_batched_tokens
         - 块管理器还有足够空块
       就给它分配块，挪到 running。
     如果挪进来了至少一个 → 这一步就是 prefill 步，返回。
  2. 否则做 decode：running 里每个请求各生成 1 个 token。
       每个请求先问块管理器 can_append()（需要新块吗？有空块吗？）
       没有空块 → 抢占（preempt）
```

**prefill 优先**：新请求尽快拿到第一个 token（降低 TTFT）。代价是 prefill 那一步，正在 decode 的请求要等一下。更高级的系统会用**分块预填充（chunked prefill）**把长 prompt 切开，和 decode 混在同一个 batch 里，本项目没有实现（见"可以继续做的事"）。

### 一个 step 的完整流程
`nanoinfer/engine/llm_engine.py` 的 `LLM.step()`：

```python
seqs, is_prefill = scheduler.schedule()        # 1. 选人
token_ids = model_runner.run(seqs, is_prefill)  # 2. 一次前向计算 + 采样，每人得到 1 个新 token
scheduler.postprocess(seqs, token_ids)          # 3. 追加 token，检查是否结束，结束的释放块
```

`LLM.generate()` 就是不停地调用 `step()` 直到所有请求结束。

### 打包（packing），不是填充（padding）
一个 batch 里的所有新 token 被首尾相接排成一维 `[T]`，用 `cu_seqlens`（累计长度）记录每个序列的起止位置。例如 3 个 prompt 长度 5、3、7：

```
tokens:      [a a a a a | b b b | c c c c c c c]
cu_seqlens:  [0,         5,      8,             15]
```

线性层（矩阵乘法）不在乎 token 属于哪个序列，直接一次算完 15 个；只有注意力需要按 `cu_seqlens` 分开算。这样一个 padding 都没有。

## 5. 抢占（Preemption）：缓存满了怎么办

请求越写越长，需要的块越来越多。某一步发现空块用完了，怎么办？

本项目的策略（和 vLLM 默认一致）是**重算式抢占（recompute preemption）**：

1. 选**最后进场**的请求作为"牺牲者"（victim）。
2. 释放它的所有块，把它放回 `waiting` 队列的**最前面**。
3. 以后有空间了，它重新做一次 prefill——这次的"prompt"是它原来的 prompt **加上已经生成的所有 token**，然后接着往下写。

因为贪心解码的结果是确定的，被抢占的请求最终输出和没被抢占时**一模一样**。测试 `tests/test_scheduler.py::test_engine_preemption_gives_same_greedy_output` 专门验证了这一点：故意把 KV 缓存设得很小，强制发生抢占，输出仍和大缓存时逐 token 相同。

另一种策略是**换出（swap）**：把被抢占请求的 KV 块拷到 CPU 内存，回来时再拷回去。省计算但费带宽，实现也更复杂。

还有一个巧妙之处：被抢占的请求释放的块里内容还在，并且带着哈希。如果在被覆盖之前它就重新进场，**前缀缓存**能把这些块直接"复活"，重算几乎不花时间。

## 6. 可调参数（`EngineConfig`）

| 参数 | 默认 | 含义 |
|---|---|---|
| `max_num_seqs` | 64 | 一个 batch 最多几个序列 |
| `max_num_batched_tokens` | 4096 | 一次前向最多处理多少 token（限制 prefill 的大小） |
| `max_model_len` | 4096 | 单个请求 prompt + 输出的最大长度 |
| `block_size` | 16 | 每块装几个 token |
| `num_kvcache_blocks` | 自动 | 块总数；GPU 上按剩余显存自动算，CPU 上默认用 0.5 GB |

## 7. 实测效果
`benchmarks/bench_throughput.py` 的 "mixed" 场景：32 个请求，每个要生成 16 到 256 个不等的 token。HF 按 16 个一批静态处理，必须等每批最长的那个；本项目用连续批处理。实测数字见 README。
