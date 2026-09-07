---
type: concept
topic: conditioning
status: seed
---

# Adaptive RMSNorm

一句话：先用 RMSNorm 标准化隐藏状态，再由条件向量动态调节其尺度（有些实现还包含偏移或门控）。

## 为什么在生成模型里常见？

Flow/diffusion 模型的行为依赖时间步：高噪声阶段和低噪声阶段应采用不同去噪策略。把时间嵌入转成归一化参数，比只在输入处拼接一个 time token 更直接地影响每层计算。

## 因果链

固定归一化无法表达当前噪声阶段 → 用 timestep/condition 生成调制参数 → 同一网络随生成时间改变处理方式。

## 阅读时要检查

- 条件来自 timestep、语言，还是 robot state？
- 调节的是 scale、shift，还是 residual gate？
- 每层独立生成参数，还是共享？

