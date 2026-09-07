---
type: concept
aliases: [MoT, 混合 Transformer]
topic: architecture
status: deep-explained
---

# Mixture of Transformers

> [!summary] 一句话
> 不同模态使用独立 QKV/FFN 学习自己的统计规律，但在 joint attention 中交换上下文。

![[90 Attachments/HyVLA/hyvla-mot-flow.svg]]

## 为什么普通“全部共享”不理想？

视觉 patch 关心空间结构，语言 token 关心符号关系，连续 action token 关心运动几何和精细数值。让它们完整共享一套变换，会让同一参数同时适配差异很大的分布；完全独立又会切断“看懂场景后做动作”的链路。

MoT 的折中：

```text
Vision:   Q_v,K_v,V_v, FFN_v
Language: Q_l,K_l,V_l, FFN_l
Action:   Q_a,K_a,V_a, FFN_a
                 ↓
        masked joint attention
```

每种 token 用自己的投影生成 Q/K/V，随后按统一序列与 mask 做注意力。输出再回各自 FFN。因此“共享”的是交互空间/注意力运算，“不共享”的是投影和非线性计算参数。

## 与 MoE 的差异

| | MoT | MoE |
|---|---|---|
| 选择方式 | 按模态固定 | router 动态选择 expert |
| 目标 | 模态专门化并跨模态交互 | 扩大容量、稀疏计算 |
| token 路径 | 通常确定 | 依路由结果变化 |

## 梯度会不会被隔断？

不会自然隔断。只要 joint attention 与 backbone 参数参与训练，action loss 可以反向影响其读取的视觉/语言路径。[[Hy-Embodied-0.5-VLA]] 的 pre-training 与 SFT 都写明 all parameters trainable。

## 论文证据边界

HyVLA 展示完整系统有效，但没有给出“共享 Transformer vs MoT”的干净独立消融。因此 MoT 在这篇论文里是合理架构选择，不是被单独证明的性能来源。

## 出现于

- [[Hy-Embodied-0.5-VLA]]

