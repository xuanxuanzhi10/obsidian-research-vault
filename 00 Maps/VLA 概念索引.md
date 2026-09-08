---
type: map
topic: VLA-concepts
status: active
created: 2026-09-07
---

# VLA 概念索引

> [!tip] 建议阅读方式
> 第一次进入概念页，先读“一句话定义 → 直觉理解 → 为什么需要”；第二遍跑通最小例子和数据流；最后再看跨论文差异、优势代价与证据边界。链接用于建立生态，不代替页面内的完整解释。

## 概念页统一结构

```text
第一层：直觉入口——它是什么，为什么必须出现
第二层：机制跑通——最小例子、tensor/data flow、伪代码或教学图
第三层：论文生态——不同论文怎么使用，与相近概念怎样区分
第四层：研究核查——优势、代价、失败边界、证据和待核实项
```

> [!warning] 核实状态
> “待原文 PDF 复核”表示当前只建立了知识链接，不能把具体配置当成已验证事实。现阶段 HyVLA 已完成全文与附录核查；π₀、π₀.5、OpenVLA 仍需各自 PDF 精读。

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
- [[Transformer 参数量估算]]：从 Q/K/V/O、MQA/GQA 与 gated FFN 的矩阵 shape 估算模型规模
- [[Block-wise Causal Attention]]：怎样控制条件到动作的信息方向
- [[Adaptive RMSNorm]]：怎样用 scale、shift、gate 把 timestep 与本体状态直接注入每层
- [[状态条件注入方式比较]]：HyVLA、π0、π0.5、FTP-1 怎样分别处理 robot state 与 flow timestep
- [[Compact Memory Encoder]]：怎样保留历史而不让 VLM token 暴涨

## 训练与适配

- [[LoRA]]：低成本微调大 VLA
- [[FlowPRO]]：用 winning/losing action pairs 修复长尾失败
- [[异构多域分布式训练]]：同域组成本地 batch，专属参数本域更新，共享参数跨域汇聚

## 推理与部署

- [[KV Cache]]：复用不变条件的 attention 计算
- [[Action Chunking]]：协调慢推理与快执行
- [[部署与推理优化]]：从模型输出到实时控制系统

## 触觉感知与闭环

- [[触觉信号表示]]：怎样同时保留力的历史、瞬时冲击与空间形变
- [[Morphology-Aware Tactile Token Space]]：怎样按身体功能区对齐不同触觉传感器
- [[预测触觉与观测触觉]]：怎样把接触前的前瞻与接触后的纠错分开建模
- [[触觉标点]]：怎样用接触事件切分长任务并切换子目标
- [[异步触觉动作细化]]：怎样让慢视觉规划接受高频触觉纠偏
- [[触觉机器人学习]]：按用途、表示、动作接口、融合和验证组织触觉论文
- [[触觉 VLA 方法比较]]：T-Rex、FTP-1、N0-TWAM 的互补关系与 HyVLA 组合路线

## 横向理解

- [[数据策略对比]]
- [[动作表示方法对比]]
- [[模型架构设计]]
- [[训练范式地图]]
