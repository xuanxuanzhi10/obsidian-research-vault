---
type: concept
topic: attention
status: deep-explained
aliases: [块级因果注意力]
---

# Block-wise Causal Attention

> [!summary] 一句话
> 把 token 分成语义块；块内部按任务需要双向交流，块之间保持“条件 → 动作”的因果方向。

![[90 Attachments/HyVLA/hyvla-blockwise-mask.svg]]

## 它在解决什么矛盾？

严格 token-wise causal mask 会迫使同一 action chunk 逐点生成，图像内部也不能完整交流；完全双向又会让 perception 表征读取 noisy action，破坏稳定条件前缀。于是粒度从 token 提升到 block：

```text
P = perception（images + language）
S = robot state
A = noisy action chunk
顺序：P → S → A
```

| Query | 能读取 | 不能读取 |
|---|---|---|
| P | 完整 P | S、A |
| S | 完整 P、S | A |
| A | 完整 P、S、A | — |

## 玩具矩阵怎么手工构造？

假设 5 个 P、2 个 S、3 个 A。对任意 query `i`、key `j`：

```python
allow(i, j) = block_index(j) <= block_index(i)
```

注意比较的是 block index，不是 token index，所以 A₁ 能看 A₂/A₃，P₁ 也能看 P₅。

## 为什么 A 块内部双向不算作弊？

Flow Matching 的训练输入本就是完整的加噪动作块 `A_τ`，网络联合预测整块速度，不是自回归地预测“下一个动作 token”。块内双向是模型目标的一部分。

## 与 KV Cache 的联系

在多步 Flow solver 中，P/S 不随 `τ` 变化，又不能读取 A，因此它们的隐藏状态和 K/V 是稳定前缀；A 每步更新，主要重算 A。mask 不只控制因果语义，也为 [[KV Cache]] 创造条件。

## 不要混淆

- **Block-wise mask**：规定谁能看谁，是信息流拓扑；
- **[[Mixture of Transformers]]**：规定不同模态用哪套参数，是计算路径；
- 两者互补，但不是同一设计。

## 出现于

- [[Hy-Embodied-0.5-VLA]]

