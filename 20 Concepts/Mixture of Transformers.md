---
type: concept
aliases: [MoT]
topic: architecture
status: seed
---

# Mixture of Transformers

一句话：不同模态使用独立的 QKV/FFN 参数，在 joint attention 中交换信息。

## 为什么需要？

视觉、语言与连续动作具有不同统计特性，但动作决策又必须读取视觉语言上下文。MoT 用“参数专门化 + 注意力共享”处理这个矛盾。

## 不要和 MoE 混淆

- MoT：按模态固定选择计算参数。
- MoE：通常由 router 为 token 动态选择 experts。

## 出现于

- [[Hy-Embodied-0.5-VLA]]

