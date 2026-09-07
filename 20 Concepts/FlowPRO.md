---
type: concept
topic: post-training
status: deep-explained
aliases: [RPRO, Flow Preference Optimization]
---

# FlowPRO

> [!summary] 一句话
> 面向 flow policy 的偏好后训练：把 rollout 中的失败动作和人工回滚后的纠正动作组成对比信号，不另训 reward/critic。

![[90 Attachments/HyVLA/hyvla-flowpro-loop.svg]]

## 为什么 SFT 之后还需要它？

SFT 擅长复现示范覆盖的主分布，但长尾失败通常位于狭窄决策边界：USB 差 2 mm、瓶盖角度略偏、拉链卡住。继续加入大量普通成功示范，信号会被主分布淹没。

FlowPRO 主动采集“模型恰好会犯错的位置”。

## 数据闭环

```text
rollout → operator 发现失败 → intervention
        → rollback 到失败前 → 人给 correction
        → rejected / preferred action pair
        → RPRO 更新 → 新策略再次 rollout
```

若某状态只记录到一侧动作，论文用平滑插值生成缺失 counterpart，使偏好比较尽量发生在相近状态。

## RPRO 直觉

若 `ℓθ(a|s)` 是 flow-matching loss，`ℓref` 是冻结参考策略的 loss：

```math
r_θ(s,a)=β²(ℓ_ref-ℓ_θ)
```

当前模型相对 reference 越能拟合动作，隐式分数越高。对 preferred/rejected 做对比，同时用 symmetric proximal regularizer 约束偏移，并混合 SFT 防止基础能力遗忘。

## HyVLA 训练配方

- 第 1 轮：80% 新偏好、20% SFT；
- 第 2 轮起：70% 新偏好、15% 历史偏好、15% SFT；
- 3 轮，每轮 25K steps；global batch 20；
- 每任务不超过 `O(10²)` preference pairs。

## “reward-free”准确是什么意思？

没有显式 reward model、value network 或 critic；不是没有监督，也不是无人参与。人仍负责判断失败、回滚、纠正。它更准确的优势是减少“学一个可靠标量奖励函数”的负担。

## 证据边界

HyVLA 在 4 个长尾任务、3 seeds 上优于 DAgger 和 π0.6*，三轮后达 99/99/98/94%。但任务数量有限，operator skill、回滚精度与 pair 质量可能影响结果。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
