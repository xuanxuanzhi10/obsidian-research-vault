---
type: concept
topic: action-generation
status: checked-seed
---

# Flow Matching

一句话：学习一个随时间变化的 velocity field，把简单噪声分布连续搬运到目标数据分布。

## 在 VLA 中为什么需要？

连续 robot actions 可能是多峰分布；相比离散 action tokens 或单次 MSE 回归，Flow Matching 能直接生成连续 action chunks。

## 用最小例子跑通

训练时取真实动作 `A` 和噪声 `ε`，在随机时间 `τ` 构造中间状态。网络看到 observation、state、`A_τ` 与 `τ`，学习“此刻应该往哪个方向移动”。

推理时从噪声出发，多次积分预测的 velocity field，最后得到完整动作块。

```text
noise --vθ(·, τ | condition)--> action chunk
```

## 主动回答读者疑问

> [!question] 它就是 diffusion 吗？
> 两者都是从简单分布生成数据，但训练目标和连续路径的构造不同。阅读论文时不要只凭“多步去噪”把二者等同。

> [!question] 为什么不直接回归平均动作？
> 当多条轨迹都合理时，平均动作可能一条都不像；生成模型的目标是描述条件分布，而不只是条件均值。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
- [[π₀]]
- [[π₀.5]]
