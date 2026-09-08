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

## 直觉理解

它像驾校教练：平时让学员自己开，只有在即将犯错时接管；随后退回到错误发生前，示范一次正确操作。这样得到的不是泛泛的“这一趟失败了”，而是同一个局部情景下“刚才那样做不好，应该这样做”。

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

## 与相近方法的边界

| 方法 | 监督从哪里来 | 是否显式训练 reward/critic |
|---|---|---|
| 继续 SFT | 更多成功示范 | 否 |
| DAgger | policy 访问状态上的 expert label | 否 |
| 常见 RLHF/RL | 偏好或环境反馈 | 通常需要 reward/value 中至少一部分 |
| FlowPRO | 失败分支 + rollback 后 correction | 否 |

## 优势、代价与失败边界

**优势：** 数据集中在模型真实失败边界；不必先拟合一个可靠的标量 reward model。

**代价：** 需要熟练 operator 在线观察、回滚和纠正；状态无法精确复现时，pair 可能不完全可比。

**边界：** 当前证据来自少量精细操作任务，不能直接外推到长时程自主任务。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
