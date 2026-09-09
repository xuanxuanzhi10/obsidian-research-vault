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
8. 视觉看不见接触内部状态时，触觉怎样进入表示、动作与闭环？

## 已读论文

- [[Hy-Embodied-0.5-VLA]] — 完整 robot learning stack
- [[π₀]] — VLM + continuous Action Expert
- [[π₀.5]] — 异构 co-training 与 open-world generalization
- [[OpenVLA]] — 开放的 autoregressive VLA 基线
- [[T-Rex]] — 慢速视觉规划 + 高频触觉细化的灵巧操作 VLA
- [[FTP-1]] — MTTS 统一异构触觉并跨传感器预训练
- [[N0-TWAM]] — 联合预测未来视频/触觉，并用当前触觉闭环纠错
- [[N0-VTLA]] — 用 10 个预测性触觉 latent 条件化 π0.5，并以 ALTER 利用部署数据

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
- [[触觉信号表示]]
- [[异步触觉动作细化]]
- [[Morphology-Aware Tactile Token Space]]
- [[预测触觉与观测触觉]]
- [[触觉标点]]
- [[Latent Tactile Token]]
- [[ALTER]]
- [[异构协同训练]]
- [[层级 VLA 推理]]

## 横向入口

- [[VLA 方法比较]]
- [[训练范式地图]]
- [[数据策略对比]]
- [[动作表示方法对比]]
- [[模型架构设计]]
- [[部署与推理优化]]
- [[待核实事实]]
- [[触觉机器人学习]]
- [[触觉 VLA 方法比较]]
- [[状态条件注入方式比较]]
