---
type: paper
title: "Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack"
alias: HyVLA-0.5
year: 2026
arxiv: "2606.14409v2"
status: first-pass
topics: [VLA, cross-embodiment, robot-learning-stack]
backbone: Hy-Embodied-0.5-MoT
parameters: "4B backbone + 370M action expert"
action_representation: relative-EEF delta chunk
action_generation: Flow Matching
dataset: Hy-UMI-10K
created: 2026-09-07
---

# Hy-Embodied-0.5-VLA

上级地图：[[VLA 学习地图]] · 相关比较：[[VLA 方法比较]]

> [!summary] 一句话结论
> 这篇论文的核心贡献不是一个孤立模型模块，而是把 **高精度 UMI 数据 → embodied VLM → continuous action expert → failure-driven post-training → asynchronous deployment** 连成一套真实机器人学习栈。

## 01 为什么强 VLA 仍不等于可部署机器人？

- **Data**：机器人 teleoperation 昂贵，人类数据的动作标签又常常不够精确。
- **Architecture**：视觉语言理解与连续动作生成需要不同计算路径。
- **Training**：SFT 容易卡在少量、关键的长尾失败。
- **Deployment**：4B 模型的推理速度与 50 Hz servo rate 不匹配。

→ 下一问：怎样让视觉、语言和动作交流，却不强迫它们使用完全相同的参数？

## 02 架构：分参数，但在注意力中交流

[[Mixture of Transformers]] 让 vision、language、action 使用各自 QKV/FFN，再进入 shared joint-attention。Action Expert 约 370M 参数，负责根据视觉语言上下文和 robot state 预测连续动作速度场。

> [!warning] 容易误读
> MoT 不等于阻断 action loss 对 VLM 的全部梯度。论文明确说明 pre-training 与 SFT 阶段 all parameters trainable；更准确的说法是不同模态拥有独立的计算参数路径。

→ 下一问：参数分开以后，怎样保证 perception 不依赖 noisy action，同时保留图像内部的双向注意力？

## 03 注意力：Block-wise Causal Attention

| Query block | 可以读取 |
|---|---|
| Perception（image + language） | Perception |
| State | Perception + State |
| Noisy Action | Perception + State + Action |

block 内部 bidirectional，block 之间 causal。固定 observation prefix 因而可以使用 [[KV Cache]]，Flow Matching 的后续 solver steps 主要重算 action tokens。

→ 下一问：机器人需要历史，但直接送 K 帧会让 VLM token 数量乘 K，怎么办？

## 04 记忆：把历史压进当前帧

[[Compact Memory Encoder]] 每隔 4 个 ViT layers 插入 temporal attention：同一 patch 位置先跨时间交流，再做 frame 内 spatial attention。上层只保留当前帧 tokens，因此送入 VLM 的 token 数仍等于单帧。

- temporal pass 复用原 ViT 的 QKV 与输出投影
- fixed sinusoidal temporal encoding，且 `e(0)=0`
- pre-training 用 `K=1`，SFT 再切到 `K=6`

**证据检查**：RoboTwin full 为 90.9/90.1；去掉 memory 后为 88.8/88.6。

→ 下一问：视觉接口统一以后，不同机器人的关节结构仍不同，动作怎样跨 embodiment？

## 05 动作接口：Relative EEF delta chunk

模型不直接预测 joint angles，而预测相对于当前 end-effector frame 的未来位姿增量：

```text
3D translation + 6D continuous rotation + 1D gripper
= 10D / arm
= 20D / step for two arms
```

这把“末端要去哪里”与“某台机器人怎样弯关节”分离。后者交给 deployment mapper 和 IK。

> [!note] 表示统一不等于物理能力统一
> 新机器人仍有 reachability、collision 和 torso pose 问题。JAKA 数据经过 IK feasibility filtering；humanoid 任务先限制在 reachable shell 内。

关联概念：[[Relative EEF Action]] · [[Action Chunking]]

→ 下一问：一段连续动作存在多种合理解，为什么不直接 MSE 回归？

## 06 动作生成：Flow Matching

[[Flow Matching]] 学习把 Gaussian noise 搬运到真实 action distribution 的 velocity field。推理从噪声出发，用 10 次 Euler integration 生成整个 chunk。

```text
A_tau = tau * A + (1-tau) * epsilon
noise -- learned velocity field --> action chunk [H, 20]
```

> [!danger] 复现疑点
> 由插值式可得速度应为 `A - epsilon`，Sec. 4.2 也如此定义；但 Eq. (3) 写成 `epsilon - A`，同时又从 tau=0 向 1 正向积分。正文存在符号不一致，应以官方代码为准。

→ 下一问：模型能生成动作后，能力怎样从 10K 小时数据迁移到具体任务，并继续修复失败？

## 07 训练：Pre-training → SFT → FlowPRO

### Stage 1：UMI pre-training

- Hy-UMI-10K：10K+ hours、1M+ episodes、70 tasks
- `K=1`，3 views，224×320，`H=50 @ 10 Hz`
- 200K steps，global batch 1,024，peak LR 5e-5，bf16

### Stage 2：task SFT

- `K=6`，all parameters trainable
- 真实机器人：`H=50 @ 50 Hz`
- 60K steps，batch 32，LR 2.5e-5

### Stage 3：[[FlowPRO]]

operator 在 rollout 失败时 intervention-and-rollback：失败段作为 losing trajectory，人工纠正段作为 winning trajectory。RPRO 拉近 preferred action、推远 failed action，并用 proximal regularizer 锚定 reference policy。

> [!warning] Reward-free 不是 feedback-free
> 它不训练 reward model、value network 或 critic，但仍依赖人发现错误、回滚、复位并提供 correction。

## 08 部署：模型慢，机器人不能停

推理线程不断产生 action chunk；执行线程以 50 Hz 从 buffer 取动作。新 chunk 到达时丢掉 stale prefix，并用 cubic Bézier 把当前轨迹接到新 chunk 的内部点：position 保持 C1 continuity，orientation 用 SLERP，gripper 线性插值。

> [!warning] 50 Hz 不等于模型每秒 forward 50 次
> 论文报告的是 action execution rate，没有给出 backbone latency、replanning frequency 或端到端 control lag。

## 09 结果与证据边界

- RoboTwin 2.0：90.9% Clean / 90.1% Randomized
- Track-B：JAKA 90%，Astribot 89%，无需目标机器人 teleoperation
- RPRO 三轮后：Bottle/Cap/USB/Zip 为 99/99/98/94%

### 证据边界

- **强证据**：Compact Memory removal、FlowPRO baseline comparison
- **中等证据**：UMI pre-training 的 simulation 与 real-robot 对照
- **有限演示**：跨 embodiment 只有两个平台各一个任务，并经过 reachability filtering
- **缺少消融**：MoT、relative EEF、async runtime、Bézier stitching 的独立贡献
- **缺少部署指标**：GPU、显存、量化、latency、实际 replanning rate

## 自测

- [ ] MoT 的“分开”与“共享”分别发生在哪里？
- [ ] Block-wise mask 和 MoT 各自保护什么？
- [ ] Compact Memory 为什么不会让 VLM token 数乘 K？
- [ ] relative EEF 为什么仍不能消除 reachability 问题？
- [ ] reward-free、50 Hz、cross-embodiment 最容易分别被误读成什么？

