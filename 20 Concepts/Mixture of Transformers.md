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

## 直觉理解

想象视觉、语言和动作三组研究员共用一张会议桌：所有人能听见被允许的信息，但各自使用不同的专业工具和笔记本。会议桌对应 joint attention，专业工具对应每种模态独立的 QKV 与 FFN。

## 为什么普通“全部共享”不理想？

视觉 patch 关心空间结构，语言 token 关心符号关系，连续 action token 关心运动几何和精细数值。让它们完整共享一套变换，会让同一参数同时适配差异很大的分布；完全独立又会切断“看懂场景后做动作”的链路。

MoT 的折中：

```python
# 教学伪代码：一个 HyVLA MoT layer
# 不是官方代码；目的是看清“哪里独立、哪里发生交互”。

def mot_layer(vision, language, action, attention_mask):
    # 1. 每种模态使用自己的投影参数
    q_v, k_v, v_v = vision_qkv(vision)       # [B, Nv, heads, Dh]
    q_l, k_l, v_l = language_qkv(language)   # [B, Nl, heads, Dh]
    q_a, k_a, v_a = action_qkv(action)       # [B, Na, heads, Dh]

    # 2. 拼在同一个注意力空间里交换信息
    q = concat([q_v, q_l, q_a], dim="token")
    k = concat([k_v, k_l, k_a], dim="token")
    v = concat([v_v, v_l, v_a], dim="token")
    mixed = attention(q, k, v, mask=attention_mask)

    # 3. 按 token 边界拆回各模态
    mixed_v, mixed_l, mixed_a = split(mixed, [Nv, Nl, Na])

    # 4. 每种模态再走自己的 FFN
    vision_out = vision_ffn(mixed_v)
    language_out = language_ffn(mixed_l)
    action_out = action_ffn(mixed_a)
    return vision_out, language_out, action_out
```

逐行看，跨模态交互只发生在 `attention(...)`；`vision_qkv/language_qkv/action_qkv` 和三个 FFN 都使用各自参数。因此“共享注意力”更准确地说是共享一次联合注意力运算及交互空间，不是共享同一套 QKV 权重。

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

## 在论文生态中的位置

| 论文 | 当前知识库中的架构位置 | 核实状态 |
|---|---|---|
| [[Hy-Embodied-0.5-VLA]] | Vision / Language / Action 三路专属参数 + joint attention | 已核对全文 |
| [[π₀]] | VLM 主干与 Action Expert 的双路计算 | 待用原文 PDF 复核细节 |
| [[π₀.5]] | 延续双路结构，并引入新的条件与动作表示 | 待用原文 PDF 复核细节 |

## 优势、代价与失败边界

**优势：** 模态可以专门化，同时保留跨模态信息交换；不同塔还可使用不同宽度。

**代价：** 多套参数、切片和 mask 增加实现复杂度；joint attention 本身并没有缩短序列。

**边界：** “参数独立”不等于“梯度隔离”，也不等于 MoE 式动态路由。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
