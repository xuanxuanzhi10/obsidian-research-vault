---
type: concept
topic: action-tokenization
status: checked-seed
source: https://arxiv.org/abs/2501.09747
---

# FAST Tokenizer

一句话：先在频率空间压缩一段动作，再把它表示成适合自回归模型预测的离散 token。

## 为什么逐维、逐时刻分箱不够？

高频动作相邻时刻高度相关。逐点 tokenization 重复描述平滑变化，序列很长，也难以表达灵巧动作的时间结构。

## 工作机制

```text
action chunk
→ 离散余弦变换（DCT）
→ 频率系数的量化与序列化
→ 离散 action tokens
```

平滑轨迹的能量集中在少量低频系数，因此能以更短序列保留主要形状。

## 与 Flow Matching 的分歧

- FAST：把连续动作转换成离散序列，适配 autoregressive next-token prediction。
- [[Flow Matching]]：不把最终动作离散化，通过连续动力系统生成 action chunk。

> [!warning] 不要混淆
> FAST 的核心是 DCT-based compression，不是一个 VQ-VAE 式“学习出来的向量量化码本”。FAST+ 才是在大规模轨迹上形成的通用 tokenizer 配置。

