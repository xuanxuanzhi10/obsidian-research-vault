---
type: concept
topic: architecture
status: checked-seed
---

# Action Expert

一句话：在保留 VLM 语义能力的同时，给连续动作生成一条专门的计算路径。

## 为什么需要？

视觉语言 token 与机器人动作的统计结构不同：前者偏语义和离散，后者要求连续、精确并具有强时间相关性。完全共享参数会要求同一组权重同时适应两种任务。

## 它做什么，不做什么？

- 读取视觉、语言和 robot state 条件。
- 预测动作生成过程中的向量场或动作块。
- 不等同于独立 low-level controller；IK、servo 和 safety layer 仍可位于模型之外。

## 常见误读

> [!warning]
> “独立参数路径”不等于“动作损失不会更新 VLM”。是否冻结、梯度如何传播，要看具体训练配置，而不是从 expert 这个名字推断。

## 出现于

- [[π₀]]
- [[π₀.5]]
- [[Hy-Embodied-0.5-VLA]]

