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

### Qwen3 的区别
Qwen3 多了 **QK-Norm**（在 RoPE 之前对每个头的 q、k 做 RMSNorm），q/k/v 投影没有偏置（bias）。`ModelConfig` 会根据 `model_type` 自动切换，测试里用随机初始化的小 Qwen3 验证过和 HF 一致。

## 3. 扁平 token 布局（flat layout）

模型的输入不是常见的 `[batch, seq_len]`，而是所有序列首尾相接的一维 `[T]`（见 03 篇的"打包"）。线性层、RMSNorm、MLP 都是逐 token 的，不在乎 token 属于谁；只有注意力需要知道边界，它从 `AttentionMetadata` 里读取：

| 字段 | 含义 |
|---|---|
| `slot_mapping [T]` | 每个新 token 的 K/V 写到哪个槽位 |
| `cu_seqlens_q / cu_seqlens_k` | prefill：每个序列新 token / 总 token 的起止位置 |
| `block_tables [B, max_blocks]` | 每个序列的块表 |
| `context_lens [B]` | decode：每个序列当前总长度 |

## 4. 注意力的两条路径（`nanoinfer/model/attention.py`）

### Prefill
对每个序列循环（prompt 长度不同，循环最直观）：
- **没有前缀命中**：K、V 就是这一步刚算的，直接用 SDPA 的 `is_causal=True`（因果掩码：只能看自己和前面）。
- **有前缀命中**：从缓存里按块表 gather 出完整的 K、V，用"右下对齐"的因果掩码。

### Decode
完全向量化：
1. 为 batch 中每个序列算出 `[B, 最长长度]` 的槽位矩阵（每次前向只算一次，所有层共用：`AttentionMetadata.prepare()`）。
2. 一次 `index_select` 读出所有 K、V。
3. 用 mask 屏蔽填充位置，调用 SDPA。
4. 如果所有序列一样长（比如 batch=1），就不需要 mask。

### 两个后端（backend）
- `sdpa`（默认）：`torch.nn.functional.scaled_dot_product_attention`，PyTorch 的融合算子，在 GPU 上会自动用 FlashAttention 类的 kernel。
- `naive`：教科书公式 `softmax(QKᵀ/√d)·V`，一行一行写出来，方便学习。测试保证两者结果一致。

## 5. 加载权重（`nanoinfer/model/loader.py`）

1. 在 `meta` 设备上搭建模型骨架：只有形状，不占内存，不做随机初始化。
2. `to_empty()` 在目标设备（CPU/GPU）上分配未初始化的内存。
3. 用 `safetensors.safe_open` 逐个读取权重，按**名字**拷进对应参数（我们的参数名和官方 checkpoint 完全一致，比如 `model.layers.0.self_attn.q_proj.weight`）。
4. 检查是否有缺失的权重；绑定 `lm_head` 和 embedding（tied embeddings）。

这样峰值内存约等于模型本身大小，不会出现"先在 float32 里随机初始化一遍再覆盖"的浪费。

## 6. CUDA Graph（实验性，仅 GPU）

decode 每一步要启动几百个小 GPU kernel，在小模型上，**启动开销（launch overhead）**甚至比计算本身还大。CUDA Graph 把一整步的 kernel 序列"录制"一次，以后一次性"回放"。

限制：录制时张量形状必须固定。所以我们：
- 录制几个固定的 batch 大小（1, 2, 4, 8, 16, 32, …），实际 batch 向上补齐到最近的那个；
- 注意力总是 gather `max_model_len` 长度并用 mask；
- 补齐的行写入"草稿块"，不会污染真实数据。

默认关闭（`--cuda-graph` 打开），因为**它没有在真实 GPU 上测试过**（开发机没有 GPU）。录制失败会自动退回普通模式。但它用到的"静态补齐"的 decode 逻辑是纯 PyTorch，`tests/test_tiny_model_vs_hf.py::test_static_padded_decode_matches_eager` 在 CPU 上验证了它和普通路径结果一致。
