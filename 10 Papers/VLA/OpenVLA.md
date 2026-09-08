---
type: paper
title: OpenVLA
year: 2024
status: imported-seed
verification: core-claims-checked
topics: [VLA, open-source, action-tokenization]
source: https://arxiv.org/abs/2406.09246
---

# OpenVLA

上级地图：[[VLA 学习地图]] · 相关比较：[[VLA 方法比较]]

> [!summary] 一句话结论
> OpenVLA 首先解决的不是“动作生成最精细吗”，而是：能否把一个有竞争力的通用 VLA 做成开放、可微调、可部署的研究基线。

## 01 为什么当时需要 OpenVLA？

强 VLA 多为闭源 → 研究者无法检查训练与适配细节 → 新任务往往只能重新训练小 policy → 因而需要一个开放的 generalist VLA 和完整微调路径。

## 02 它怎样把 VLM 变成 policy？

图像编码器融合 DINOv2 与 SigLIP 特征，语言主干使用 Llama 2。连续动作被逐维离散化为 token，模型便能沿用语言模型的 next-token prediction 来输出动作。

这一步的好处是工程统一；代价是时间结构与连续几何被拆成 token 分类问题。

## 03 真正应该记住的贡献

- 7B 参数、基于 970k 条真实机器人 demonstrations 训练。
- 检查点、训练代码和微调路径公开。
- 证明 LoRA 与量化可以降低适配和部署门槛。

## 04 主动回答读者疑问

> [!question] 离散 token 的每一格是否对应固定毫米数？
> 不能笼统这样说。分箱边界依赖动作归一化和训练数据分布，跨机器人、跨维度并没有一个通用的“毫米/格”。

> [!question] 它和 [[π₀]] 的核心分歧是什么？
> OpenVLA 把动作纳入自回归 token 接口；π₀ 增加独立 [[Action Expert]]，用 [[Flow Matching]] 直接生成连续 action chunk。

## 05 证据边界

核心架构、数据规模与开放微调主张已按论文摘要核对；具体训练耗时、单卡频率和量化数值暂不写入正文，见 [[待核实事实]]。

