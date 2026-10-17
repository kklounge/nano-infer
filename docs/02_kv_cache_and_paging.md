# 02 · KV 缓存与分页（KV Cache & Paged Attention）

## 1. 什么是 KV 缓存（KV Cache）

在注意力（attention）里，每个 token 会算出三个向量：

- **Q（query，查询）**："我想找什么信息？"
- **K（key，键）**："我身上有什么标签？"
- **V（value，值）**："我真正携带的内容。"

新 token 用自己的 Q 去和前面所有 token 的 K 做匹配（打分），再按分数把它们的 V 加权平均。

关键观察：**旧 token 的 K 和 V 永远不会变**（因为 token 只能看前面，不能看后面，前面的东西不受新 token 影响）。所以：

> 算过一次的 K、V 存起来，下次直接用。这就是 KV 缓存。

有了它，decode 每一步只需要算**新的那 1 个 token** 的 Q、K、V，然后去缓存里读旧的 K、V。计算量从"整个序列"降到"1 个 token"。

### KV 缓存有多大？
以 Qwen2.5-0.5B 为例，每个 token 要存：

```
2 (K 和 V) × 24 层 × 2 个 KV 头 × 64 维 × 2 字节 (bf16) = 12,288 字节 ≈ 12 KB
```

一个 4000 token 的对话约 48 MB，64 个用户同时聊就是 3 GB。**KV 缓存是推理时最主要的显存消耗之一**，怎么管好它，就是推理引擎的核心问题。

## 2. 老办法的问题：按最大长度预留

最朴素的做法（Hugging Face `generate` 默认就是这样）：给每个请求分配一块**连续**内存，大小按"最长可能长度"预留。

问题：
1. **内部浪费（internal fragmentation）**：预留 4096 个位置，用户只聊了 200 个 token，剩下 95% 闲置。
2. **外部碎片（external fragmentation）**：请求有长有短，结束后留下大小不一的空洞，新的大请求塞不进去。
3. **不能共享**：两个请求开头一样（比如同一个系统提示词），也得各存一份。

vLLM 论文测出这种方式实际只用到了 20%–40% 的 KV 内存。

## 3. 新办法：像操作系统一样分页（Paging）

操作系统管理内存的经典办法是**分页（paging）**：把内存切成固定大小的"页"，程序看到的是连续的地址，实际可以散落在各处，靠一张"页表"翻译。

**PagedAttention**（vLLM 提出）把这个想法搬到了 KV 缓存上：

- 把整块 KV 缓存切成很多**块（block）**，每块固定装 `block_size` 个 token（本项目默认 16）。
- 每个请求有一张**块表（block table）**，记录"我的第 0 块在物理第 7 号块，第 1 块在第 3 号块……"
- 需要时一块一块地分配，用完立刻还回去。

**比喻**：一本公用笔记本（KV 缓存），每页能写 16 个字。图书管理员（block manager）给每个学生发页码。学生的作文不需要写在连续的页上，只要记住自己的页码清单（block table）就行。写满一页再领一页，写完了把页还回来。

好处：
- 浪费最多就是每个请求最后一块没写满的部分（平均半块）。
- 没有外部碎片：所有块一样大，任何空块都能用。
- 可以共享：两个请求开头相同，块表里指向同一个物理块就行（见 [04_prefix_caching.md](04_prefix_caching.md)）。

## 4. 本项目里的具体做法

### 4.1 缓存的形状
`nanoinfer/engine/model_runner.py`：

```python
kv_cache = torch.zeros(2, num_layers, num_blocks + 1, block_size, num_kv_heads, head_dim)
#                      K/V  层数      块数(+1个草稿块)  每块token数  KV头数       每头维度
```

最后多出的 1 块是"草稿块（scratch block）"，只给 CUDA Graph 的填充行写垃圾数据用，永远不会分给真实请求。

### 4.2 槽位（slot）：token 存在哪
把一层的缓存摊平成 `[num_blocks × block_size, kv_heads, head_dim]`，每一行叫一个**槽位（slot）**。位置 p 的 token 存在：

```
slot = block_table[p // block_size] * block_size + p % block_size
```

例：block_size=16，block_table=[7, 3]，位置 20 → 第 20//16=1 块 → 物理块 3 → 3×16 + 20%16 = 52 号槽。

### 4.3 每层注意力做两件事
`nanoinfer/model/attention.py`：

1. **写（store_kv）**：把这一步新 token 的 K、V 写进它们的槽位：`k_cache.view(-1, H, D).index_copy_(0, slot_mapping, k)`
2. **读（gather）+ 计算**：按块表把这个序列所有 token 的 K、V 从各个块里"捡"回来，拼成连续张量，再算注意力。

本项目有两种实现，用 `attn_backend` 切换：
- **`triton`（GPU 默认）**：自己写的 Triton kernel **直接在块上算注意力**，按块表一块一块读 K/V，不需要先拼成连续张量（和 vLLM 的 PagedAttention 是同一个思路）。见 [05 篇](05_model_and_attention.md) 第 4 节。
- **`sdpa`（CPU 默认）/ `naive`**：用 PyTorch 的 `index_select` 先按块表把 K/V gather 成连续张量，再调用 `torch.nn.functional.scaled_dot_product_attention`（SDPA）或教科书公式。好读、能在 CPU 上跑，但多了一次拷贝。测试保证三种后端结果一致。

### 4.4 Decode 时怎么批量读？
batch 里每个序列长度不同。PyTorch 后端把每个序列的槽位号排成一个 `[B, 最长长度]` 的矩阵，短的用 0 填充，再用一个 **mask（掩码）** 把填充位置屏蔽掉，一次 `index_select` 读出所有序列的 K/V，完全向量化。Triton 后端不需要这个矩阵：kernel 的每个程序（program）负责一个序列的一个 KV 头，自己读块表、按真实长度循环。

### 4.5 块管理器（Block Manager）
`nanoinfer/engine/block_manager.py`，主要方法（v0.2 起按"这一步要算几个 token"来分配，支持分块预填充）：

| 方法 | 什么时候调用 | 做什么 |
|---|---|---|
| `get_cached_prefix(seq)` | 新请求想进场 | 查前缀缓存：开头有几个写满的块已经算过（只查，不改任何状态） |
| `allocate_prefix(seq, ids)` | 新请求进场 | 共享这些命中的块（引用计数 +1，或从空闲队列"复活"） |
| `can_allocate_slots(seq, n)` | 每一步调度时 | 这一步要再算 n 个 token，空闲块够不够 |
| `allocate_slots(seq, n)` | 每一步调度时 | 按需分配新块；这一步会被写满的块，立刻算哈希登记 |
| `deallocate(seq)` | 请求结束 / 被抢占 | 引用计数减一，归零的块回到空闲队列 |

decode 就是 n = 1 的特例：只有当新 token 正好落在新块的第一个位置时，才需要 1 个空块。

空闲块用 `OrderedDict` 按"最近释放"排序（**LRU，least recently used**），最久没用的先被覆盖。这个顺序对前缀缓存很重要，下一篇讲。

## 5. 小结

- KV 缓存：存下旧 token 的 K、V，避免重复计算。
- 分页：把缓存切成固定大小的块，用块表映射，浪费少、无碎片、能共享。
- 下一篇：有了块，调度器怎么决定谁先跑？→ [03_continuous_batching.md](03_continuous_batching.md)
