---
type: concept
topic: architecture
status: learning-guide
aliases: [动作专家]
---

# Action Expert

> [!summary] 一句话定义
> 在保留 VLM 语义表征的同时，为连续动作生成提供专门的参数和计算路径。

![[90 Attachments/HyVLA/hyvla-mot-flow.svg]]

## 直觉理解

VLM 像理解任务的“观察员”，Action Expert 像把观察转成精细控制的“运动员”。观察员知道“把 USB 插进去”是什么意思，运动员需要处理毫米级位置、姿态和时间协调。

## 为什么需要它

语言 token 偏符号与语义，动作 tensor 是连续、精确、强时间相关的信号。全部共享一套参数，会要求同一 FFN 同时服务两种差异很大的统计分布；完全分离又会让动作读不到视觉语言条件。

## 工作原理

```text
视觉/语言 backbone ──条件信息──┐
robot state ────────────────────┼→ Action Expert → action velocity/chunk
noisy action + flow time τ ─────┘
```

具体模型中，它可能通过 joint attention 读取 backbone，也可能通过 cross-attention 或显式条件向量连接。名字相同不代表接口完全相同。

## 它做什么，不做什么？

**通常做：** 融合视觉、语言、状态条件；生成动作 token、连续动作或速度场。

**通常不做：** IK、servo、碰撞保护、力矩闭环和安全限制；这些仍可位于模型外。

## 在论文生态中的位置

| 论文 | 当前知识库记录 | 核实状态 |
|---|---|---|
| [[Hy-Embodied-0.5-VLA]] | 370M action tower，输出 Flow Matching velocity | 已核对全文 |
| [[π₀]] | VLM 与连续动作 expert 双路建模 | 待原文 PDF 复核细节 |
| [[π₀.5]] | 后训练阶段使用连续动作 expert | 待原文 PDF 复核细节 |

## 与相近概念的边界

- [[Mixture of Transformers]] 是更广的多模态参数分工方式；Action Expert 是动作一侧的专门路径。
- low-level controller 把目标转成真实 actuator command；Action Expert 通常仍属于学习策略。

## 优势、代价与失败边界

**优势：** 动作容量可单独调整，连续控制不必完全复用语言 FFN。

**代价：** 增加参数管理、mask、宽度对齐和训练稳定性问题。

> [!warning] 常见误读
> “独立参数路径”不等于“动作损失不会更新 VLM”。是否冻结、梯度如何传播，必须看论文训练配置。

