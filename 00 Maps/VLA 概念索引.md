---
type: map
topic: VLA-concepts
status: active
created: 2026-09-07
---

# VLA 概念索引

## 数据与动作标签

- [[UMI]]：怎样低成本采集自然人类操作
- [[Relative EEF Action]]：怎样用末端相对运动减轻 embodiment 差异

## 动作的模型表示

- [[Action Chunking]]：为什么一次生成多步动作
- [[FAST Tokenizer]]：怎样把动作块压缩成离散 token
- [[Flow Matching]]：怎样直接生成连续、多峰动作分布
- [[Action Expert]]：为什么动作需要专门计算路径

## 多模态架构

- [[Mixture of Transformers]]：参数专门化，注意力中交流
- [[Block-wise Causal Attention]]：怎样控制条件到动作的信息方向
- [[Adaptive RMSNorm]]：怎样把生成时间或其他条件注入每层
- [[Compact Memory Encoder]]：怎样保留历史而不让 VLM token 暴涨

## 训练与适配

- [[LoRA]]：低成本微调大 VLA
- [[FlowPRO]]：用 winning/losing action pairs 修复长尾失败

## 推理与部署

- [[KV Cache]]：复用不变条件的 attention 计算
- [[Action Chunking]]：协调慢推理与快执行
- [[部署与推理优化]]：从模型输出到实时控制系统

## 横向理解

- [[数据策略对比]]
- [[动作表示方法对比]]
- [[模型架构设计]]
- [[训练范式地图]]

