---
type: comparison
topic: VLA
status: seed
created: 2026-09-07
---

# VLA 方法比较

| 方法 | 动作表示 | 感知底座 | 数据/训练侧重点 | 最鲜明的问题意识 |
|---|---|---|---|---|
| [[OpenVLA]] | 离散 action tokens | 通用 VLM | OXE、开放微调 | 开源 generalist policy |
| [[π₀]] | 连续 Flow Matching | PaliGemma + Action Expert | 高质量机器人数据 | 高频灵巧控制 |
| [[π₀.5]] | 连续 Flow Matching | VLM + Action Expert | 异构数据 co-training | open-world generalization |
| [[Hy-Embodied-0.5-VLA]] | relative-EEF delta chunk | 4B embodied MoT | 10K h UMI + FlowPRO | 完整 real-world learning stack |

## 后续比较问题

- 连续动作表示是否始终优于离散 action tokens？
- embodied-native backbone 的收益能否与数据规模解耦？
- cross-embodiment transfer 中，表示、数据和 deployment mapper 各贡献多少？

