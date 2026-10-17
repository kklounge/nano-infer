# 01 · 大模型推理是怎么回事（LLM Inference Basics）

> 这一篇不需要任何 AI 背景。读完你应该能用自己的话讲清楚：模型是怎么"一个字一个字"写出回答的，以及为什么推理分成 prefill 和 decode 两个阶段。

## 1. 先把文字变成数字：Token

模型不认识汉字和字母，只认识数字。所以第一步是**分词（tokenization）**：把一句话切成小块，每块叫一个 **token（词元）**，每个 token 对应词表（vocabulary）里的一个编号。

```
"The capital of France is"  →  [785, 6722, 315, 9625, 374]
```

Qwen2.5 的词表有 151,936 个 token。一个常见英文单词通常是 1 个 token；中文里常见的词也会合成一个 token，比如"用一句话介绍康奈尔大学"11 个字只切成 7 个 token（"一句话"和"大学"各是 1 个）。

## 2. 模型做的唯一一件事：猜下一个 token

把一串 token 喂给模型，它只输出一样东西：**下一个 token 是词表里每个词的可能性分数**，这串分数叫 **logits**（151,936 个数字）。

就像手机输入法的联想：你打"今天天气"，它猜下一个字是"很"的可能性最大。

## 3. 一个字一个字地写：自回归生成（Autoregressive Generation）

模型一次只能猜一个 token，那怎么写出一整段话？很简单，**循环**：

```
输入: [今天, 天气]           → 猜出 "很"
输入: [今天, 天气, 很]       → 猜出 "好"
输入: [今天, 天气, 很, 好]   → 猜出 "。"
输入: [..., 。]              → 猜出 <结束符 EOS>，停止
```

每次把新猜出来的 token 接到末尾，再喂回去。这就叫**自回归（autoregressive）**。

## 4. 从分数里挑一个：采样（Sampling）

拿到 logits 后，怎么选出下一个 token？最基本的四种方法（`nanoinfer/engine/sampler.py`）：

| 方法 | 英文 | 做法 | 效果 |
|---|---|---|---|
| 贪心 | greedy (temperature=0) | 永远选分数最高的 | 稳定、可复现，适合做测试 |
| 温度 | temperature | 分数除以 T 再变成概率；T<1 更保守，T>1 更放飞 | 控制"创造力" |
| Top-k | top-k | 只在分数前 k 名里抽 | 防止抽到离谱的词 |
| Top-p | top-p / nucleus | 只在"累计概率刚好达到 p"的最小集合里抽 | 候选数量自动伸缩 |
| Min-p | min-p | 去掉概率小于"最高概率 × p"的候选 | 比 top-p 更适应"有把握/没把握"的情况 |

此外还支持：**重复惩罚**（repetition / presence / frequency penalty，压低已经出现过的 token 的分数，防止车轱辘话）、`min_tokens`（至少写够 N 个 token 才允许结束）、**stop 字符串**（生成的文字里一出现就停）、每个请求自己的 `seed`（同样的种子 → 同样的输出）、`logprobs`（返回每个 token 的对数概率）、`n`（一个问题生成 n 个不同的回答）。实现细节见 [05 篇](05_model_and_attention.md) 第 8 节。

## 5. 模型里面长什么样（只讲直觉）

Qwen 这类模型叫 **decoder-only Transformer**。一个 token 进来后：

1. **嵌入（embedding）**：编号 → 一个 896 维的向量（可以理解为这个词的"坐标"）。
2. 经过 **24 层**一模一样结构的"楼层"（decoder layer），每层两个房间：
   - **注意力（attention）**：这个 token 回头看前面所有 token，决定"该关注谁"。比如"它"要回头找"它"指代的是谁。
   - **前馈网络（MLP / feed-forward）**：对每个 token 自己做一次复杂的加工。
3. 最后一层输出再乘以一个大矩阵（**lm_head**），得到 151,936 个 logits。

细节见 [05_model_and_attention.md](05_model_and_attention.md)。

## 6. 关键：Prefill 和 Decode 两个阶段

一次生成其实分两个很不一样的阶段：

### Prefill（预填充）
- 把**整个提示词（prompt）**一次性喂进模型，比如 500 个 token **同时**计算。
- 输出第一个新 token。
- 特点：计算量大，但可以并行，GPU 很"吃饱"。属于**计算密集（compute-bound）**。
- 用户感受到的指标：**TTFT（time to first token，首字延迟）**——按下回车到看到第一个字的时间。

### Decode（解码）
- 之后每一步只喂 **1 个**新 token，输出下一个 token。重复几百次。
- 特点：每一步计算很少，但每一步都要把**全部模型权重**（0.5B 模型约 1GB）从内存读一遍。瓶颈在"搬数据"而不是"算"，属于**访存密集（memory-bound）**。
- 指标：**TPOT（time per output token，每个字的间隔）** 和 **吞吐量（throughput，tokens/s）**。

**一个比喻**：prefill 像是老师第一次通读整篇作文（一次读完很多字）；decode 像是老师接着往下一个字一个字地续写，每写一个字都要翻一遍整本字典（权重）。

### 为什么这个区分这么重要？
因为 decode 的瓶颈是"每步都要读一遍权重"，那么**一次同时帮 16 个用户各写一个字**，权重也只读一遍！这就是**批处理（batching）**能大幅提高吞吐量的原因。本项目的调度器（scheduler）就是为了把尽可能多的请求凑在一起 decode。详见 [03_continuous_batching.md](03_continuous_batching.md)。

## 7. 但是有个大浪费……

按第 3 节的循环，每一步都把**整个序列**重新喂一遍，前面的 token 被重复计算了无数次。第 100 步要算 100 个 token，第 101 步又要算 101 个……

解决办法就是 **KV 缓存（KV cache）**：把以前算过的东西存起来。下一篇讲：[02_kv_cache_and_paging.md](02_kv_cache_and_paging.md)。
