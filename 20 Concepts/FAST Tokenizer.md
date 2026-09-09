---
type: concept
topic: action-tokenization
status: learning-guide-source-linked
aliases: [FAST, Frequency-space Action Sequence Tokenization]
source: https://arxiv.org/abs/2501.09747
---

# FAST Tokenizer

> [!summary] 一句话定义
> 先用频率表示压缩一段动作轨迹，再量化并序列化成适合 autoregressive 模型预测的离散 token。

## 直觉理解

一条平滑动作轨迹像一段声音：逐时刻记录每个数值很冗余，记录“整体趋势 + 少量快速变化”更紧凑。DCT 把时间轨迹从逐点坐标变成不同频率的系数。

## 为什么逐维、逐时刻分箱不够？

相邻动作高度相关。逐点分箱会重复描述平滑变化，token 序列很长；每一步、每一维的独立误差还可能破坏整段轨迹结构。

## 工作原理

```text
action chunk [H,D]
→ 沿时间维做 DCT
→ 高频/低频系数量化
→ 按约定顺序序列化
→ 离散 action tokens
```

玩具例子：若 8 步动作几乎匀速，DCT 能量主要集中在直流和少数低频系数；无需用 8 个完全独立 token 重复表达相近数值。

> [!note] 这是机制示意，不是官方实现
> 系数排序、量化尺度、熵编码和词表构造必须以 FAST 原文/代码为准。

## 与 Flow Matching 的分歧

| | FAST | [[Flow Matching]] |
|---|---|---|
| 最终表示 | 离散 tokens | 连续 action tensor |
| 训练接口 | next-token prediction | velocity-field regression |
| 推理 | autoregressive token decoding | 多步 ODE solver |
| 主要优势 | 复用 LLM 离散建模能力 | 保留连续、多峰生成 |

## 在论文生态中的位置

- FAST 原论文：概念来源，source 已记录但尚未在本 Vault 做全文精读。
- [[π₀.5]]：280k 离散预训练使用 FAST，80k 后训练加入连续 Flow Expert；PDF 已复核。
- [[OpenVLA]] 不使用 FAST：它只对单个时间步的各动作维度分别做 256-bin quantization，见 [[逐维动作分箱]]。

## 优势、代价与失败边界

**优势：** 利用时间频率冗余缩短动作 token 序列，并兼容成熟的 next-token 训练框架。

**代价：** 量化引入精度损失；序列化和码率配置会影响跨数据集泛化。

**边界：** FAST 的核心不是 VQ-VAE learned codebook；FAST 与 FAST+ 的训练规模和通用性也不能混为一谈。
