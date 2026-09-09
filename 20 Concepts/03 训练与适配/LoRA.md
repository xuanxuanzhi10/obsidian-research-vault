---
type: concept
topic: parameter-efficient-finetuning
status: learning-guide
aliases: [Low-Rank Adaptation, 低秩适配]
---

# LoRA

> [!summary] 一句话定义
> 冻结原线性层，只训练一个低秩权重增量，以更少可训练参数和优化器显存适配大模型。

## 直觉理解

完整微调像重写一本厚字典；LoRA 假设新任务只需要一组结构化修订，于是在原书旁增加一小叠勘误页。推理时把原权重与修订共同使用。

## 为什么低秩可能够用？

虽然权重矩阵很大，单一任务需要的参数变化可能集中在较低维子空间。于是将增量限制为两个窄矩阵的乘积：

```math
W' = W + sBA
```

若 `W∈R[d_out,d_in]`，可令 `A∈R[r,d_in]`、`B∈R[d_out,r]`，且 `r` 远小于输入输出维度。

## 最小参数量比较

```text
完整矩阵可训练参数：d_out × d_in
LoRA 可训练参数：   r × (d_in + d_out)
```

例如 4096×4096 线性层约 1680 万参数；`r=16` 的 A/B 合计约 13 万，不到前者 1%。这只是算术例子，不代表所有任务都适合 r=16。

## 工作流程

```text
加载并冻结 pretrained W
→ 在选定层挂载 A/B
→ 只优化 A/B
→ 推理时现场相加或 merge 进 W
```

## 在 VLA 中为什么有用？

VLA backbone 往往数十亿参数；LoRA 让单任务或单机器人适配降低训练显存和 checkpoint 体积。

## OpenVLA 的实证坐标

在 33 次 Franka-Tabletop rollouts 的较小 SigLIP-only OpenVLA variant 上：

| 方法 | 成功率 | 可训练参数 | 显存（batch 16） |
|---|---:|---:|---:|
| Full FT | 69.7±7.2% | 7,188.1M | 163.3GB，2卡 FSDP |
| LoRA r=32 | 68.2±7.5% | 97.6M（1.4%） | 59.7GB |
| LoRA r=64 | 68.2±7.8% | 195.2M | 60.5GB |

因此论文推荐默认 `r=32`；在单张 A100 上约 10-15 小时完成适配。注意：1.4% 指可训练参数，不是推理时只加载 1.4% 基座权重；这组实验也不是完整 DINOv2+SigLIP final checkpoint。

## 与相近方法的边界

- Full fine-tuning：容量最大，但训练成本高、每个任务都保存完整模型。
- Adapter：通常插入新的小网络；LoRA 直接参数化线性权重增量。
- Prompt tuning：主要学习输入侧参数，不直接修改中间线性映射。

## 优势、代价与失败边界

**优势：** 少量可训练参数、较小 checkpoint、同一 backbone 可挂载多个任务适配器。

**代价：** rank 和插入层选择敏感；强域迁移时低秩容量可能不足。

**边界：** 可训练参数少不自动意味着推理更快，也不自动带来跨 embodiment 泛化。
