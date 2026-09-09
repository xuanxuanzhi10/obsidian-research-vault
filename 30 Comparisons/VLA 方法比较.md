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
| [[π₀.5]] | 预训练 FAST；后训练/推理为 H=50 连续 Flow | PaliGemma 2.6B + 300M Action Expert | MM/ME/CE/HL/WD，后训练加 VI；280k+80k | 未见家庭中的开放世界长任务 |
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

## π₀ → π₀.5：不要漏掉的结构变化

| 维度 | π₀ | π₀.5 |
|---|---|---|
| 第一阶段动作监督 | 连续 Flow Matching | FAST 离散 next-token |
| state | continuous S block | 离散化为 prefix text tokens |
| timestep | 与 noisy action embedding 融合 | timestep MLP → 每层 AdaRMSNorm |
| 高层 | 无核心显式子任务阶段 | 同一模型先生成子任务，再生成 action |
| 数据问题 | 多 embodiment 的通用控制 | 环境、语义、技能与高层的异构协同 |

π₀.5 论文中的 `π₀-FAST+Flow` 是专门构造的增强基线，不是原始 π₀ 的训练方式。
