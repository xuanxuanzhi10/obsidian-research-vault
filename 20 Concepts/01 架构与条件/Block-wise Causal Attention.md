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

## 直觉理解

把多模态序列想成一栋分层办公楼：感知、状态、动作各占一层。每层内部可以开会讨论，所以是双向注意力；跨层的信息只能按“感知 → 状态 → 动作”向下传递。动作层能询问感知层“杯子在哪里”，感知层却不能拿尚在生成的动作当作视觉证据。

## 它在解决什么矛盾？

严格 token-wise causal mask 会迫使同一 action chunk 逐点生成，图像内部也不能完整交流；完全双向又会让 perception 表征读取 noisy action，破坏稳定条件前缀。于是粒度从 token 提升到 block：P 是 perception，S 是 robot state，A 是 noisy action chunk，顺序为 `P → S → A`。

| Query | 能读取 | 不能读取 |
|---|---|---|
| P | 完整 P | S、A |
| S | 完整 P、S | A |
| A | 完整 P、S、A | — |

## 玩具矩阵怎么手工构造？

假设 5 个 P、2 个 S、3 个 A。下面用接近实现的伪代码构造 mask：

```python
# 教学伪代码：Block-wise Causal Attention Mask
# 不是论文官方实现，但每一行都对应图中的一个区域。

def build_blockwise_mask(num_p=5, num_s=2, num_a=3):
    block_sizes = [num_p, num_s, num_a]
    block_names = ["P: perception", "S: state", "A: noisy action"]

    # token_block = [0,0,0,0,0, 1,1, 2,2,2]
    #                └──── P ────┘ └S┘ └─ A ─┘
    token_block = repeat_each_block_id(block_sizes)
    total_tokens = sum(block_sizes)
    mask = zeros(total_tokens, total_tokens)  # [query, key]

    for query in range(total_tokens):
        for key in range(total_tokens):
            query_block = token_block[query]
            key_block = token_block[key]

            # 核心规则：只能读取自己所在块或更早的条件块
            if key_block <= query_block:
                mask[query, key] = ALLOW
            else:
                mask[query, key] = BLOCK

    return mask, block_names
```

注意比较的是 `block index`，不是 `token index`：

- `query=A₁, key=A₃`：二者 block id 都是 2，所以允许；
- `query=P₁, key=P₅`：二者 block id 都是 0，所以允许；
- `query=P₁, key=A₁`：`2 > 0`，因此禁止，动作不能反向进入感知。

## 为什么 A 块内部双向不算作弊？

Flow Matching 的训练输入本就是完整的加噪动作块 `A_τ`，网络联合预测整块速度，不是自回归地预测“下一个动作 token”。块内双向是模型目标的一部分。

## 与 KV Cache 的联系

在多步 Flow solver 中，P/S 不随 `τ` 变化，又不能读取 A，因此它们的隐藏状态和 K/V 是稳定前缀；A 每步更新，主要重算 A。mask 不只控制因果语义，也为 [[KV Cache]] 创造条件。

## 不要混淆

- **Block-wise mask**：规定谁能看谁，是信息流拓扑；
- **[[Mixture of Transformers]]**：规定不同模态用哪套参数，是计算路径；
- 两者互补，但不是同一设计。

## 在论文生态中的位置

| 论文 | 概念在当前笔记中的角色 | 核实状态 |
|---|---|---|
| [[Hy-Embodied-0.5-VLA]] | P/S/A 三块；为 Flow solver 提供稳定条件前缀 | 已核对全文 |
| [[π₀]] | P/S/A 三块；P→S→A 单向，块内双向；P/S 可缓存 | 已核对 Appendix B |
| [[π₀.5]] | 当前笔记记录为进一步隔离不同动作表示 | 待用原文 PDF 复核 |

## 优势、代价与失败边界

**优势：** 同时保留块内联合建模和块间单向条件关系；固定前缀还可以配合 KV Cache。

**代价：** block 划分与 mask 需要手工设计，实现比标准 causal/full attention 更复杂。

**边界：** mask 只约束前向可见性；它不自动冻结参数，也不保证 action loss 无法更新 VLM。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
- [[π₀]]
