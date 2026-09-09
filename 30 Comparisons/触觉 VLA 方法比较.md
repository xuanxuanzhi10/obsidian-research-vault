---
type: comparison
title: 触觉 VLA 方法比较
topics: [tactile, VLA, comparison]
---

# 触觉 VLA 方法比较

原版补充答疑：[[tactile-vla-qa2.html|触觉 VLA 深度答疑（二）HTML]]

其中三个可复用问题已独立整理并校正：[[Transformer 参数量估算]] · [[Adaptive RMSNorm]] · [[异构多域分布式训练]]。

> [!abstract] 先给选择结论
> T-Rex 解决“反应速度”，FTP-1 解决“跨传感器迁移”，N0-VTLA 解决“以最小接口预测一个动作块的接触后果”，N0-TWAM 解决“完整未来生成 + 当前接触纠错”。它们不是四选一，而是不同研究问题。

| 维度 | [[T-Rex]] | [[FTP-1]] | [[N0-VTLA]] | [[N0-TWAM]] |
|---|---|---|---|---|
| 核心问题 | 高频触觉反应 | 异构传感器统一与迁移 | 紧凑预测未来接触 | 未来前瞻 + 当前反射 |
| 触觉角色 | 当前观测，后半段 flow 细化 | 当前观测，条件动作 | 未来 H 步净变化的 latent condition | 未来生成目标 + 当前条件 |
| 表示 | 16 帧 wrench VQ + 当前力 + 形变图 | MTTS 24 功能槽；多类 encoder | DINOv2 差分特征；10 latent tokens | 预测 VAE residual；观测 force space |
| 模型接口 | 独立快 Tactile Expert + cache | 独立 300M Tactile Expert | Predictor → π0.5 Action Expert | Video/Tactile/Action MoT + observed cross-attn |
| 时间结构 | 约 5 Hz 慢规划 / 20 Hz 快纠偏 | 常规 action chunk | 预测 H=50 chunk 的净变化 | sensor/chunk/event 三尺度 |
| 数据重点 | 100 h 接触原语 | 约 3000 h、26 sources、21 sensors | NeoData 多平台视触觉；本报告未列精确小时数 | 30,000+ h、6 embodiments、450 tasks |
| 最强证据 | 12 真机任务、快慢消融 | seen/unseen sensor 对照 | 表示检索、反事实触觉、29 个任务 | predicted/observed 双路径消融 |
| 最明显局限 | 固定硬件、复杂调度 | 未解决预测式低层控制 | 仅预测净变化、无当前触觉直达、传感器范围窄 | 7B 规模、体系内依赖强 |

## 对 HyVLA 的组合顺序

```text
第一步：当前触觉 encoder + zero-init cross-attention
  └─ 验证 vision-only vs current tactile
第二步：若跨硬件，加入 MTTS 与多 sensor adapters（FTP-1）
第三步：若控制延迟是瓶颈，做 cache + fast tactile refinement（T-Rex）
第四步：若需要接触前避错，先试 10-token future latent（N0-VTLA）
第五步：若紧凑 latent 不足，再升级成 predicted + observed 双路径（N0-TWAM）
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

## N0-VTLA 与 N0-TWAM 最容易混淆在哪里？

```text
N0-VTLA：当前差分 g → Predictor → 未来净变化 z → 普通 Action Expert
N0-TWAM：历史视频/触觉 → 共同生成未来视频/触觉 → Action Expert
                                ↑ 同时读取当前真实 force map
```

前者是贴近 π0.5 的小型预测接口，不生成完整未来 tactile video，当前 g 也不直达动作；后者是完整 WAM，明确证明 predicted 与 observed 两条路径互补。若目标是低成本改造 HyVLA，先做 N0-VTLA 式模块更容易获得可解释消融。

## 关于 HTML 的证据级别

三个 HTML 是学习指南/二手解释，保留其原貌用于回顾；论文事实以对应 PDF 为准。通用答疑页并非某一篇论文的附件，因此放在本比较页与专题页，而不是伪装成论文原文。
