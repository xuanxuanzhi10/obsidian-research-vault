---
type: comparison
title: 触觉 VLA 方法比较
topics: [tactile, VLA, comparison]
---

# 触觉 VLA 方法比较

原版补充答疑：[[tactile-vla-qa2.html|触觉 VLA 深度答疑（二）HTML]]

其中三个可复用问题已独立整理并校正：[[Transformer 参数量估算]] · [[Adaptive RMSNorm]] · [[异构多域分布式训练]]。

> [!abstract] 先给选择结论
> T-Rex 解决“反应速度”，FTP-1 解决“跨传感器迁移”，N0-TWAM 解决“未来接触预测 + 当前接触纠错”。它们不是三选一，而是触觉系统的三个互补层。

| 维度 | [[T-Rex]] | [[FTP-1]] | [[N0-TWAM]] |
|---|---|---|---|
| 核心问题 | 高频触觉反应 | 异构传感器统一与迁移 | 未来触觉前瞻 + 当前触觉反射 |
| 触觉角色 | 当前观测，后半段 flow 细化 | 当前观测，条件动作 | 未来生成目标 + 当前条件 |
| 表示 | 16 帧 wrench VQ + 当前力 + 形变图 | MTTS 24 功能槽；image/array/state encoder | 预测用 VAE residual；观测用 force space |
| 模型接口 | 独立快 Tactile Expert + cache | 独立 300M Tactile Expert | Video/Tactile/Action MoT + observed cross-attn |
| 时间结构 | 约 5 Hz 慢规划 / 20 Hz 快纠偏 | 常规 action chunk，无专门快慢伺服 | sensor/chunk/event 三时间尺度 |
| 数据重点 | 100 h 触觉接触原语 | 约 3000 h、26 sources、21 sensors | 30,000+ h、6 embodiments、450 tasks |
| 最强证据 | 12 真机任务、快慢消融 | seen/unseen sensors 与 NTP-1 对照 | predicted/observed 双路径消融 |
| 最明显局限 | 固定硬件、复杂调度 | 未解决预测式低层触觉控制 | 7B 规模、体系内数据/表征依赖强 |

## 对 HyVLA 的组合顺序

```text
第一步：当前触觉 encoder + zero-init cross-attention
  └─ 验证 vision-only vs current tactile
第二步：若跨硬件，加入 MTTS 与多 sensor adapters（FTP-1）
第三步：若控制延迟是瓶颈，做 cache + fast tactile refinement（T-Rex）
第四步：若需要接触前避错，加入 future tactile residual loss（N0-TWAM）
```

不要一开始把三套机制全部叠加，否则消融无法回答性能来自哪里。最重要的统一实验矩阵是：

```python
variants = {
  "V":              vision_policy,
  "V+O":            with_observed_touch,
  "V+P":            with_predicted_touch,
  "V+O+P":          with_both_touch_paths,
  "V+O+fast":       with_async_refinement,
  "V+O+P+cross_hw": with_MTTS_unseen_sensor,
}
```

## 关于 HTML 的证据级别

三个 HTML 是学习指南/二手解释，保留其原貌用于回顾；论文事实以对应 PDF 为准。通用答疑页并非某一篇论文的附件，因此放在本比较页与专题页，而不是伪装成论文原文。
