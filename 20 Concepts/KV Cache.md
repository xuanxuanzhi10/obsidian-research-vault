---
type: concept
topic: inference
status: learning-guide
aliases: [Key-Value Cache, 键值缓存]
---

# KV Cache

> [!summary] 一句话定义
> 缓存不会变化的 attention Key/Value，使后续生成步骤不必重复计算同一前缀。

## 直觉理解

回答同一篇文章上的十个问题时，不必每次重新扫描并抄写全文；可以保留文章的索引，只重新处理新问题。KV Cache 保存的就是 attention 查找所需的“索引和值”。

## 为什么需要它

Transformer 每层都会从 hidden state 计算 K/V。如果图像、语言和状态在多次生成迭代中不变，重复计算它们只增加延迟。

## 工作原理：第一次与后续迭代

```text
第一次：P/S → K_PS,V_PS ┐
        A_0 → K_A0,V_A0 ├→ attention

第二次：复用 K_PS,V_PS  ┐
        A_1 → K_A1,V_A1 ├→ attention
```

在 Flow Matching VLA 中，`A_τ` 每个 solver step 都变化，因此 action 部分不能直接沿用旧 cache；固定条件可以复用。

## 成立的三个前提

1. prefix 内容在各次迭代中不变；
2. mask 阻止 prefix 读取会变化的 action block；
3. position encoding 与缓存位置保持一致。

若 perception 能反向读取 noisy action，它自己的 hidden/K/V 也会随 `τ` 变化，旧缓存立即失效。

## 与相近概念的边界

- Action Chunking 减少需要重规划的频率；KV Cache 减少一次多步生成内的重复计算。
- 缓存不是压缩：它通常以额外显存换取更低计算延迟。
- 训练 checkpoint cache 与 attention KV cache 不是一回事。

## 在论文生态中的位置

- [[Hy-Embodied-0.5-VLA]]：P/S 作为稳定条件前缀，配合 [[Block-wise Causal Attention]]；已核对机制。
- [[π₀]]：P/S prefix KV 缓存，十次只重算 action suffix；三路相机在 RTX 4090 上 image 14 ms、observation 32 ms、十次 action 27 ms，已核对 Appendix D。
- [[π₀.5]]：相似条件生成结构仍以其自身原文/代码为准，不能直接继承 π₀ 的时间表。

## 优势、代价与失败边界

**优势：** 不改变模型输出语义即可减少重复投影和前缀计算。

**代价：** 占用显存，并增加缓存生命周期、位置和 batch 管理复杂度。

> [!question] KV Cache 会让模型更聪明吗？
> 不会。它改变的是计算复用和延迟，不会增加训练权重中不存在的能力。
