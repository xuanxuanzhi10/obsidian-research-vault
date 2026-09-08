---
type: concept
topic: action-representation
status: learning-guide
aliases: [动作分块, 动作块]
---

# Action Chunking

> [!summary] 一句话定义
> 一次预测未来 H 步动作，而不是每次只预测下一个 servo step。

## 直觉理解

逐点动作像开车时每隔很久才收到一句“方向盘向左 1°”；action chunk 则一次给出未来一小段轨迹。即使模型还在计算下一段，机器人也能继续执行手里的轨迹。

## 为什么需要它

高容量 VLA 的一次推理通常慢于机器人底层 servo。若每个控制点都等待完整 forward，机器人会走走停停；逐点预测也缺少显式的短期轨迹一致性。

## 工作原理：最小时间线

假设执行频率 10 Hz、`H=4`：

```text
一次模型输出：[a_t, a_t+1, a_t+2, a_t+3]
servo 时刻：     0     100ms   200ms   300ms
下一次推理可与这些动作的执行并行
```

输出 tensor 通常是 `[B,H,D_action]`。`H` 决定开放环轨迹长度；replanning rate 决定多久用新观察替换旧计划。

## 它把问题转移到了哪里？

- 新旧 chunk 在边界可能不连续；
- 推理期间前几个动作已经过期，形成 stale prefix；
- chunk 越长，越容易忽略执行中出现的新变化；
- chunk 越短，模型 forward 压力越大。

[[Hy-Embodied-0.5-VLA]] 用异步 buffer 和 Bézier stitching 处理前两项。

## 在论文生态中的位置

| 论文 | 当前知识库记录的用途 | 核实状态 |
|---|---|---|
| [[Hy-Embodied-0.5-VLA]] | Flow Matching 一次生成整段 relative-EEF action | 已核对全文 |
| [[π₀]] | 连续动作 expert 输出动作块 | 待原文 PDF 复核配置 |
| [[π₀.5]] | 与 FAST/连续动作生成共同使用 | 待原文 PDF 复核配置 |

## 与相近概念的边界

- [[Flow Matching]]：决定动作块怎样生成；Action Chunking 决定一次生成多少步。
- [[FAST Tokenizer]]：决定动作块怎样压缩成离散 token。
- trajectory planning：往往显式使用动力学/代价；action chunk 可完全由策略网络生成。

## 优势、代价与失败边界

**优势：** 减少高层推理频率、提高短期一致性、支持推理与执行并行。

**代价：** 牺牲部分闭环反应速度，并引入边界拼接和过期动作问题。

> [!question] chunk 越长越好吗？
> 不是。它是在计算效率、反应速度和轨迹稳定性之间的折中，不存在脱离任务与延迟的最佳 H。

