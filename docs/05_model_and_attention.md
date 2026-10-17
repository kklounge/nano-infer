# 05 · 模型结构与注意力实现（Qwen Architecture & Attention）

本项目**没有调用** Hugging Face 的模型代码来做推理，而是自己用 PyTorch 写了一遍 Qwen2 / Qwen3 的网络结构（`nanoinfer/model/`），再把官方的 safetensors 权重文件按名字加载进来。Hugging Face transformers 只在测试里当"标准答案"，以及用它的分词器（tokenizer）。

## 1. 整体结构

```
input_ids ──► Embedding（查表：编号 → 896 维向量）
          ──► 24 × DecoderLayer:
                 h = h + Attention( RMSNorm(h) )
                 h = h + MLP( RMSNorm(h) )
          ──► RMSNorm
          ──► lm_head（896 → 151,936 个 logits）
```

`h = h + ...` 叫**残差连接（residual connection）**：每层只学"在原来基础上改多少"，让很深的网络也能训练。归一化放在子层前面，叫 **pre-norm**。

Qwen2.5-0.5B 的关键数字（来自 `config.json`）：

| 名称 | 值 |
|---|---|
| hidden_size（向量维度） | 896 |
| num_hidden_layers（层数） | 24 |
| num_attention_heads（Q 头数） | 14 |
| num_key_value_heads（KV 头数） | 2 |
| head_dim（每头维度） | 64 |
| intermediate_size（MLP 中间维度） | 4864 |
| vocab_size（词表大小） | 151,936 |
| tie_word_embeddings | true（lm_head 和 embedding 共用一个矩阵） |

## 2. 四个核心组件（`nanoinfer/model/layers.py`）

### RMSNorm（均方根归一化）
把向量"缩放到差不多的大小"，防止数值越算越大或越算越小：

```
y = x / sqrt(mean(x²) + eps) * weight
```

比 LayerNorm 少了"减均值"这一步，更便宜。我们先转成 float32 计算再转回来，和官方实现完全一致。

### RoPE（旋转位置编码，Rotary Position Embedding）
注意力本身不知道 token 的先后顺序，需要告诉它位置。RoPE 的做法：把 Q 和 K 的每一对数字，按"位置 × 某个频率"的角度**旋转**一下。两个向量都转过之后再做点积，结果只和它们的**相对距离**有关。

实现上：所有位置的 cos、sin 预先算好成一张表；每次前向只查一次表，24 层共享。

### GQA（分组查询注意力，Grouped-Query Attention）
Qwen2.5-0.5B 有 14 个 Q 头，但只有 2 个 KV 头：每 7 个 Q 头共用一组 K、V。

好处：**KV 缓存缩小到 1/7**。KV 缓存是推理的内存瓶颈，所以几乎所有新模型都用 GQA。

本项目在 decode 时用了一个小技巧：把共用同一组 KV 的 7 个 Q 头，看成这组 KV 的"7 行查询"，`q.view(B, 2, 7, 64)`。数学完全一样，但不需要把 K、V 复制 7 遍，省内存（见 `attention.py` 的 `_decode_attention`）。

### SwiGLU MLP
```
MLP(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
```
两条路：一条经过 SiLU 激活当"门"，一条是普通线性变换，相乘后再投影回 896 维。

### 合并投影（merged projections）
官方权重里 q_proj、k_proj、v_proj 是三个矩阵，gate_proj、up_proj 是两个。它们的输入完全一样（同一个 x），所以可以把权重**竖着拼成一个大矩阵**：一次矩阵乘法算出 `[q | k | v]`，再切开。本项目的模型里只有 `qkv_proj` 和 `gate_up_proj` 两个参数，加载权重时 `loader.py` 的 `PACKED` 表负责把官方的 `q_proj.weight` 拷到 `qkv_proj.weight` 的前 896 行、`k_proj` 拷到接下来的 128 行……（并检查每一行都被填过）。decode 时每层少了 3 次 kernel 启动，大矩阵乘法也更能吃满 GPU。

### Qwen3 的区别
Qwen3 多了 **QK-Norm**（在 RoPE 之前对每个头的 q、k 做 RMSNorm），q/k/v 投影没有偏置（bias）。`ModelConfig` 会根据 `model_type` 自动切换，测试里用随机初始化的小 Qwen3 验证过和 HF 一致。

## 3. 扁平 token 布局（flat layout）

模型的输入不是常见的 `[batch, seq_len]`，而是所有序列首尾相接的一维 `[T]`（见 03 篇的"打包"）。线性层、RMSNorm、MLP 都是逐 token 的，不在乎 token 属于谁；只有注意力需要知道边界，它从 `AttentionMetadata` 里读取。一个 batch 里 **decode 的序列排在前面，prefill 段排在后面**：

| 字段 | 含义 |
|---|---|
| `slot_mapping [T]` | 每个新 token 的 K/V 写到哪个槽位（在 GPU 上由块表和位置算出来，不在 Python 里循环） |
| `num_decode` | 前多少个序列是 decode（每个 1 个 token） |
| `decode_block_tables / decode_context_lens` | decode 部分：每个序列的块表和总长度 |
| `cu_seqlens_q` | prefill 部分：每段新 token 的起止位置 |
| `seqlens_k` | prefill 部分：每个序列"已缓存 + 这一段"的总长度 |
| `prefill_block_tables` | prefill 部分的块表 |

所有整数输入（token id、位置、块表……）在 CPU 上拼成**一个**张量，用锁页内存（pinned memory）一次性异步拷到 GPU（`ModelRunner._to_device`），每步只有一次主机→显卡的传输。

## 4. 注意力（`nanoinfer/model/attention.py` + `triton_attention.py`）

每一层的注意力分两步：先把这一步所有新 token 的 K/V 写进缓存（store），再算注意力。`paged_attention()` 把 batch 切成 decode 部分和 prefill 部分，各走各的路径。

### 三个后端（`attn_backend`）
| 后端 | 什么时候用 | 怎么算 |
|---|---|---|
| `triton` | 有 NVIDIA GPU 且装了 Triton 时自动选 | 自己写的 Triton kernel，**直接在分页的缓存上算**，不拷贝 |
| `sdpa` | CPU 默认 | 先按块表 `index_select` 把 K/V 拼成连续张量，再调用 PyTorch 的 SDPA |
| `naive` | 学习用 | 教科书公式 `softmax(QKᵀ/√d)·V` |

测试保证三者结果一致（`test_triton_attention.py` 在 CPU 上用 Triton 解释器 `TRITON_INTERPRET=1` 跑，在 GPU 上跑真的 kernel）。

### Triton decode kernel：flash-decoding
decode 时每个序列只有 1 个 query。kernel 的每个程序（program）负责一个（序列，Q 头），按块表一块一块读 K、V，用 **online softmax**（边读边维护当前最大值 m、分母 l 和加权和 acc，FlashAttention 的核心技巧）算完，不需要把 `[长度]` 的分数存下来。

问题：batch=1 时只有 14 个 Q 头 = 14 个程序，而 RTX 5070 有 48 个 SM，大部分 GPU 闲着。**flash-decoding** 的做法：把长上下文切成几段（partition），每段一个程序并行算出局部的 (m, l, acc)，再用一个很小的 reduce kernel 合并：

```
out = Σ_i exp(m_i - M) · acc_i  /  Σ_i exp(m_i - M) · l_i      (M = max_i m_i)
```

`decode_split_plan()` 决定切几段：上下文 ≤ 512 不切，否则每段 512 的倍数、最多 16 段。只有少数几种方案，Triton 只需要编译少数几个版本。循环的上界来自 `context_lens` 张量（运行时数据），所以**同一次 kernel 启动适用于任何长度——可以被 CUDA Graph 录制**。

GQA：`kvh = h // N_REP`，7 个共享同一组 KV 的 Q 头读的是同一块显存（会命中 L2 缓存），不需要复制 K/V。

### Triton prefill kernel：varlen + 分页 + 因果
prefill 段（完整 prompt、prompt 的某一段、或前缀缓存命中后剩下的部分）的 query 是这一段的新 token，key 是从序列开头到这一段末尾的所有 token（前面的已经在缓存里）。kernel 的每个程序负责（序列，Q 头，64 行 query），用 `tl.dot` 做分块矩阵乘法 + online softmax。

因果掩码按**绝对位置**：第 j 个新 token 的位置是 `ctx_start + j`，只能看位置 ≤ 它的 key。这样"普通 prefill"、"前缀命中"和"分块预填充的后续段"是**同一个 kernel**，所有长度不同的序列一次启动算完（不再按序列做 Python 循环）。

几个踩过的坑：fp32 下 `tl.dot` 默认用 TF32（精度只有 10 位），和 PyTorch 对不上，要指定 `input_precision="ieee"`；超出序列长度的 tile 整行都是 -inf，`acc / l` 会变成 NaN，要先用 `tl.where` 保护；循环变量的类型在 Triton 里必须从第一次迭代起就固定。

### PyTorch 后端（sdpa / naive）
- **prefill**：对每个序列循环；没有已缓存部分时直接 `is_causal=True`，否则从缓存 gather 出完整 K/V，用"右下对齐"的因果掩码。
- **decode**：完全向量化：算出 `[B, 最长长度]` 的槽位矩阵（每次前向只算一次，所有层共用），一次 `index_select` 读出所有 K、V，用 mask 屏蔽填充位置，调用 SDPA。

## 5. 融合算子（fused ops，`nanoinfer/model/triton_ops.py`）

PyTorch eager 模式下每个小运算都是一个独立的 GPU kernel：一次 RMSNorm 大约 7 个 kernel（转 fp32、平方、求均值、rsqrt、相乘、转回、乘权重），q、k 的 RoPE 大约 12 个，写 KV 缓存 2 个。decode 时每个 kernel 只处理几 KB 数据，GPU 的时间都花在"启动 kernel"上了。四个融合 kernel 把它们各自合成一个：

| kernel | 做什么 |
|---|---|
| `fused_add_rmsnorm` | `residual += x; out = RMSNorm(residual)`（残差加法和下一层的归一化合并） |
| `rmsnorm` | 普通 RMSNorm（也用于 Qwen3 的 q/k norm） |
| `rope_and_store_kv` | 原地旋转 q、k（RoPE），并把 k、v 写进 KV 缓存的槽位 |
| `silu_and_mul` | 在合并的 `[gate | up]` 上算 `silu(gate) * up` |

为了让残差加法能和归一化融合，模型的每一层显式地传递残差流（`hidden, residual = layer(hidden, residual)`），和 vLLM 的写法一样。舍入步骤刻意和 HF 一致（先在 fp32 里归一化，转回模型精度，再乘权重），测试里和 PyTorch 路径逐元素对比。GPU 上默认开启，`use_fused_ops=False` 可关闭。

为什么不用 `torch.compile`？在 RTX 5070 上试过，能用，但首次编译要等、生成的代码是黑盒，不好讲清楚；手写的四个 kernel 每个只有二三十行，面试时能逐行解释。

## 6. 加载权重（`nanoinfer/model/loader.py`）

1. 在 `meta` 设备上搭建模型骨架：只有形状，不占内存，不做随机初始化。
2. `to_empty()` 在目标设备（CPU/GPU）上分配未初始化的内存。
3. 用 `safetensors.safe_open` 逐个读取权重，按**名字**拷进对应参数（名字和官方 checkpoint 一致，比如 `model.layers.0.input_layernorm.weight`；q/k/v、gate/up 按 `PACKED` 表拷进合并矩阵的对应行）。
4. 检查是否有缺失的权重；绑定 `lm_head` 和 embedding（tied embeddings）。

这样峰值内存约等于模型本身大小，不会出现"先在 float32 里随机初始化一遍再覆盖"的浪费。

## 7. CUDA Graph（GPU 默认开启）

decode 每一步要启动几百个 GPU kernel（24 层 × 每层十几个），在小模型上，CPU 发射 kernel 的时间比 GPU 计算本身还长。CUDA Graph 把一整步 decode 的 kernel 序列"录制"一次，以后一次性"回放"，CPU 只需要发一条命令。

限制：录制时张量的形状和地址必须固定。所以：
- 预先录制几个 batch 大小（1, 2, 4, 8, 16, 24, 32, 48, 64, …，最大 `max_cuda_graph_batch`），实际 batch 向上补齐到最近的那个；补齐的行写进一个专门的"草稿块"（scratch block），注意力只看 1 个 key，不会污染真实数据。
- 所有输入（token id、位置、槽位、长度、块表）都放在固定的"静态缓冲区"里，每步只把新数据拷进去（一次 H2D 拷贝），然后 `graph.replay()`。
- Triton 后端的循环上界来自运行时的 `context_lens`，一张图适用于任何上下文长度；sdpa 后端要 gather 固定长度，所以额外按上下文长度分桶（256, 512, …, max_model_len）各录一张，用最小的够用的那张。
- 混合批（有 prefill 段）的那一步不走图，走普通（eager）路径。

`enforce_eager=True`（命令行 `--enforce-eager`）关闭它做对比。录制失败会自动退回普通模式。"静态补齐"的 decode 逻辑是纯 PyTorch，`test_tiny_model_vs_hf.py::test_static_padded_decode_matches_eager` 在 CPU 上验证了它和普通路径结果一致；`test_cuda.py` 在真实 GPU 上验证录制/回放的输出和普通模式逐 token 相同。

## 8. 采样器（`nanoinfer/engine/sampler.py`）

一个 batch 里每个请求可以有不同的采样设置，采样器全部用批量张量运算处理，除了最后把 token 取回 CPU，中间没有 GPU 同步：
1. 惩罚：repetition penalty（HF 的定义：正 logit 除以 p，负 logit 乘以 p）、presence / frequency penalty（OpenAI 的定义），只对出现过的 token 生效；
2. 温度，然后 top-k、top-p、min-p 过滤；
3. 抽样用"指数竞赛"（exponential race）：`argmax(p / E)`，E 是独立的指数分布随机数，和按概率抽样数学上等价，但不需要 `torch.multinomial` 的同步；设置了 `seed` 的请求用自己的 `torch.Generator`，结果可复现；
4. 需要时返回 logprobs（所选 token 和前 k 个候选）。

### 候选窗口（candidate window）

第 2、3 步不在整个词表上做，而是在一个"候选窗口"里做（`_candidate_window`）：词表有 15 万多列，每步都整列扫一遍，采样会比模型本身的前向还慢。

- 只要 batch 里**每一行**都用 top-k 限制了候选数，那么每行最多只有 W = max(top_k) 个 token 可能被选中，用 `topk(W)` 就够了（`topk` 返回的结果本来就是降序的），不需要整列排序；有一行关掉了 top-k，才退回整列 `sort`。
- W 由 Python 侧的 `SamplingParams` 算出（关掉 top-k 的行按词表大小算），所以选 W 不需要 GPU→CPU 同步。
- 贪心（temperature=0）的行最后会被 argmax 的结果覆盖，所以给它们 top_k=1、关掉其它过滤器：否则 batch 里一个贪心请求（`top_k` 默认 0）就会把窗口撑到整个词表，把其它请求一起拖进整列排序。
- 温度除法放在取窗口之后：除以一个正温度不改变顺序，所以窗口照样是对的，但少写一遍 [B, V]。
- 惩罚同理，在 [R, L] 的历史上做 gather/scatter（几千个元素），而不是开 [R, V] 的词表缓冲区（batch 64 时约一千万个元素）；只有 presence / frequency penalty 需要计数，才开那一个词表大小的缓冲区。

实测（RTX 5070，Qwen2.5-0.5B，batch 64，每步）：按 Qwen 推荐的 `top_p=0.8, top_k=20` 采样，21.2 ms → 8.2 ms；再加 `repetition_penalty=1.05`，22.6 ms → 11.3 ms；63 个贪心请求 + 1 个采样请求，20.3 ms → 9.7 ms。单路（batch 1）没有明显变化——那时瓶颈是读权重，不是采样。`top_k=0` 配 `top_p<1` 仍然要整列排序，这是没办法省的。
