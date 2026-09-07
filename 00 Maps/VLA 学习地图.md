---
type: map
topic: VLA
status: active
created: 2026-09-07
---

# VLA 学习地图

## 核心问题链

1. 视觉、语言、动作为什么要统一建模？
2. 不同模态为什么不能简单共用同一套计算？
3. 动作应该离散化，还是在连续空间生成？
4. 怎样利用异构数据学习可迁移的动作先验？
5. 怎样把 policy 适配到不同 embodiment？
6. SFT 后的长尾失败怎样继续修正？
7. 慢模型怎样驱动高频真实机器人？

## 已读论文

- [[Hy-Embodied-0.5-VLA]] — 完整 robot learning stack
- [[π₀]] — VLM + continuous Action Expert
- [[π₀.5]] — 异构 co-training 与 open-world generalization
- [[OpenVLA]] — 开放的 autoregressive VLA 基线

## 核心概念

- [[Mixture of Transformers]]
- [[Block-wise Causal Attention]]
- [[Flow Matching]]
- [[Action Chunking]]
- [[Action Expert]]
- [[FAST Tokenizer]]
- [[Adaptive RMSNorm]]
- [[Relative EEF Action]]
- [[Compact Memory Encoder]]
- [[FlowPRO]]
- [[UMI]]

## 横向入口

- [[VLA 方法比较]]
- [[训练范式地图]]
- [[数据策略对比]]
- [[动作表示方法对比]]
- [[模型架构设计]]
- [[部署与推理优化]]
- [[待核实事实]]
