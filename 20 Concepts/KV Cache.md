---
type: concept
topic: inference
status: checked-seed
---

# KV Cache

一句话：缓存不变 prefix 的 attention keys/values，避免后续生成步骤重复计算。

## 在 Flow Matching VLA 中

如果 [[Block-wise Causal Attention]] 保证 perception/state 不依赖 noisy action，那么多次 solver iteration 中只有 action tokens 需要更新，固定条件的 K/V 可以复用。

## 成立的前提

- prefix 在各次生成迭代中不变
- attention mask 阻止 prefix 读取会变化的 action block
- position encoding 与缓存位置一致

如果 perception token 会反向读取 noisy action，旧缓存就不再代表本轮计算。

> [!question] KV Cache 会让模型更聪明吗？
> 不会。它只消除重复计算，主要改变 latency 与显存占用，不改变已训练权重表达的 policy。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
