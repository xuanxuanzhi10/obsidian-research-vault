---
type: concept
topic: post-training
status: seed
---

# FlowPRO

一句话：面向 flow-matching policy 的 preference post-training，用失败动作与人工纠正形成对比信号，不训练 reward/critic model。

## 因果链

SFT 遇到长尾失败 → rollout 触发 intervention-and-rollback → 形成 winning/losing trajectories → RPRO 拉近正确动作并推远失败动作 → proximal regularizer 锚定 reference policy。

## 容易误解

Reward-free 不代表 human-feedback-free。

## 出现于

- [[Hy-Embodied-0.5-VLA]]

