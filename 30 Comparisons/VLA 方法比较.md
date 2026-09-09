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
| [[π₀]] | H=50 连续 Flow Matching | PaliGemma 3B + 300M Action Expert | >10k h、多样预训练→高质量 post-training | 语义预训练怎样兼容高频灵巧控制 |
| [[π₀.5]] | 连续 Flow Matching | VLM + Action Expert | 异构数据 co-training | open-world generalization |
| [[Hy-Embodied-0.5-VLA]] | relative-EEF delta chunk | 4B embodied MoT | 10K h UMI + FlowPRO | 完整 real-world learning stack |

## 后续比较问题

- 连续动作表示是否始终优于离散 action tokens？
- embodied-native backbone 的收益能否与数据规模解耦？
- cross-embodiment transfer 中，表示、数据和 deployment mapper 各贡献多少？

## π₀ 已核实的基准坐标

```text
PaliGemma expert：image + language，width 2048
Action Expert：state + noisy actions，width 1024
共享位置：每层 joint self-attention；FFN 与 residual 分开
动作：50-step chunk，10 次 Euler；训练/推理容器 pad 到 18D
执行：20 Hz 机器人每 16 步重规划；50 Hz 机器人每 25 步重规划
```

因此比较后续 π₀.5、HyVLA 时，必须分别核对 state/timestep 注入、动作坐标和 chunk 消费方式，不能仅因都叫 Action Expert 就视为同一实现。
