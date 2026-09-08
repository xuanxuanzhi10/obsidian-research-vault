---
type: concept
topic: conditioning
status: learning-guide-needs-source-check
aliases: [自适应 RMSNorm, AdaRMSNorm]
---

# Adaptive RMSNorm

> [!summary] 一句话定义
> 先用 RMSNorm 稳定隐藏状态，再由 timestep、语言或其他条件动态调节每层的尺度、偏移或门控。

## 直觉理解

同一个动作生成网络在高噪声阶段应先决定大方向，在低噪声阶段应精修细节。Adaptive RMSNorm 像给每一层一个“当前工作阶段”旋钮，让网络随条件改变处理方式。

## 为什么需要它

只在输入处加入 time token，条件信号必须经过许多层才能影响深层计算，而且每层收到的调节方式相同。自适应归一化把条件直接注入每层残差路径。

## 工作原理：最小公式

```math
RMSNorm(x)=x / sqrt(mean(x²)+ε)
y = scale(c) ⊙ RMSNorm(x) + shift(c)
```

有些实现还加入 residual gate：

```math
out = x + gate(c) ⊙ F(y)
```

其中 `c` 可以是 flow/diffusion timestep，也可以混入语言或状态。不同论文未必同时使用 scale、shift、gate。

## 逐步数据流

```text
condition c → 小型 MLP/Linear → scale, shift, gate
hidden x → RMSNorm ───────────→ 调制后的 hidden → Attention/FFN
```

## 与相近方法的边界

| 方法 | 条件注入位置 | 特点 |
|---|---|---|
| 拼接 time token | 序列输入 | 简单，但影响路径间接 |
| 加到 hidden | 输入或每层 | 直接，但调节形式固定 |
| Adaptive RMSNorm | 每层归一化/残差 | 可逐层改变处理策略 |

## 阅读论文时必须检查

- 条件到底来自 timestep、语言还是 state？
- 调节 scale、shift、residual gate 中的哪些？
- 参数每层独立还是共享？
- 是否 zero-init，插入 Attention 还是 FFN 两处？

## 在论文生态中的位置

- [[π₀.5]]：当前知识库记录其使用 Adaptive RMSNorm；具体公式和层级仍需用原文 PDF 复核。

## 优势、代价与失败边界

**优势：** 条件能直接影响深层网络；适合生成阶段随 timestep 改变行为。

**代价：** 增加条件投影参数与实现分支；强调制可能破坏预训练分布。

**边界：** Adaptive RMSNorm 是一个家族名，不能看到名称就假定具体实现完全相同。

