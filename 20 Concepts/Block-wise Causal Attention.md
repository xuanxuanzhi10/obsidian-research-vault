---
type: concept
topic: attention
status: seed
---

# Block-wise Causal Attention

一句话：block 内允许适合该模态的双向注意力，block 间保持条件到动作的单向信息流。

典型顺序：`Perception → State → Action`。

Action 可以读取完整条件；Perception 不读取 State/Action，因此固定 prefix 可以复用 [[KV Cache]]。

## 出现于

- [[Hy-Embodied-0.5-VLA]]

