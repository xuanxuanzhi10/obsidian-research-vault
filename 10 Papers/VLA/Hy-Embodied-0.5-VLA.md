---
type: paper
title: "Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack"
aliases: [HyVLA-0.5, Hy-Embodied-0.5-VLA]
year: 2026
arxiv: "2606.14409v2"
status: deep-read
verification: full-paper-and-appendix-checked
topics: [VLA, flow-matching, cross-embodiment, UMI, post-training, deployment]
backbone: Hy-Embodied-0.5-MoT
parameters: "4B backbone + 370M action expert"
action_representation: relative-EEF delta chunk
action_generation: conditional-flow-matching
dataset: Hy-UMI-10K
source_pdf: "/home/ubuntu/Desktop/paper/vla/Hy-Embodied-0.5-VLA.pdf"
created: 2026-09-07
updated: 2026-09-07
---

# Hy-Embodied-0.5-VLA

上级地图：[[VLA 学习地图]] · 横向比较：[[VLA 方法比较]]

> [!summary] 一句话结论
> HyVLA 的真正贡献不是“给 VLM 接了一个动作头”，而是把 **高精度人类示范 → 具身表征 → 连续动作生成 → 长尾失败纠正 → 不断档部署** 连成了一条完整链路。它问的是：一个 VLA 怎样从“离线指标不错”走到“真实机器人能持续干活”？

![[90 Attachments/HyVLA/hyvla-original-pipeline.png]]

> [!info] 原论文 Fig. 1 怎么读
> 从左到右是数据、模型和后训练，从中间向下是部署。左边保证动作标签“准”，中间负责“看懂并生成”，右边专门修复 rollout 暴露的失败，下面把同一策略落到多种机器人。它不是四个并列卖点，而是一条前后依赖的因果链。

![[90 Attachments/HyVLA/hyvla-causal-stack.svg]]

---

## 1. 先建立矛盾：为什么一个强 VLM 还不是机器人策略？

最容易带着一个错误直觉读这篇论文：既然 VLM 已经能看图、听指令，只要在末尾预测动作即可。真实系统里至少有五个缺口：

| 缺口 | 表面现象 | 根因 | HyVLA 的设计 |
|---|---|---|---|
| 数据 | 人类视频很多，却不能直接监督机器人 | 缺少同步、精确、可执行的 6DoF 动作标签 | Hy-UMI-10K |
| 表征 | 语义理解强，但精细时空控制弱 | 视觉、语言、动作统计不同；历史又昂贵 | MoT + Compact Memory |
| 生成 | MSE 动作容易平均化 | 合理动作可能多峰且是连续轨迹 | Flow Matching action expert |
| 纠错 | SFT 在罕见失败上反复犯错 | 成功演示很少覆盖“差一点失败”的边界 | FlowPRO / RPRO |
| 部署 | 大模型推理时机器人会停顿 | inference 频率和 servo 频率不同 | 异步 buffer + Bézier stitching |

所以文章的逻辑不是“提出一个模块，然后刷榜”，而是：**上一环节留下的瓶颈，逼出下一环节。**

---

## 2. 数据：为什么一定要定制 UMI？

### 2.1 作者到底想同时得到什么？

数据采集存在一个三角矛盾：人直接操作物体最自然，且手指能感知接触；机器人 teleoperation 能记录动作，却慢、贵、受工作空间限制；普通人类视频规模大，但相机轨迹不等于精确末端轨迹。

HyVLA 使用 finger-attached gripper、头戴 RGB-D 相机和外部光学 motion-capture cage。人的手负责自然操作，光学系统给出全局坐标中的亚毫米级 6DoF 轨迹，旋转编码器记录夹爪开合。RGB-D 中当前训练只使用 RGB。

> [!question] 戴在手上的夹具只是“跟踪器”吗？
> 不是。它让人保持直接的力与接触反馈；部分夹具在末端还集成了 6D 力/力矩传感器。不过论文当前策略并没有把这等同于完整触觉学习。

### 2.2 Hy-UMI-10K 的量级与构成

- 10K+ 小时、1M+ episodes、70 个任务；
- 第一人称头部视角配合双腕部视角；

| 类别 | 小时 | 占比 |
|---|---:|---:|
| Laundry | 3,025 | 28.5% |
| Kitchen | 2,040 | 19.2% |
| Personal care / miscellaneous | 1,465 | 13.8% |
| Dexterous / tool use | 1,110 | 10.4% |
| Storage / organization | 1,065 | 10.0% |
| Cleaning | 610 | 5.7% |
| Other | 1,315 | 12.4% |

### 2.3 为什么有机会跨机器人？

标签不是某台机器人的 joint angle，而是人手夹具在空间中的相对末端运动。部署时不同机器人各自用 IK 把末端目标映射到关节。因此数据先学习“末端应该怎样移动”，而不是“JAKA 的第 3 关节转多少度”。

> [!warning] 代价没有消失
> 光学 motion-capture cage 提供了精度，也限制了 in-the-wild 扩展。作者明确把“摆脱 mocap”列为未来方向。

关联：[[UMI]] · [[Relative EEF Action]]

---

## 3. 整体架构：哪些共享，哪些必须分开？

![[90 Attachments/HyVLA/hyvla-original-architecture.png]]

原论文 Fig. 2 同时画了三个关键设计：左侧是多帧视觉记忆，中间是 Vision / Language / Action 三塔，右侧是 block-wise attention mask。

### 3.1 输入在模型里被分成三个 block

```text
Perception block P = [多视角图像 I_t, 语言指令 l]
State block      S = [机器人当前本体状态 s_t]
Action block     A = [加噪后的未来动作块 a_τ]
```

动作 expert 不是看一张图片就直接吐控制量；它在每个 Flow Matching 时间点 `τ` 都接收固定条件 `P,S` 和变化中的 `a_τ`，预测此刻的速度场。

### 3.2 MoT：参数专门化，但让信息在注意力里相遇

![[90 Attachments/HyVLA/hyvla-mot-flow.svg]]

Vision、Language、Action 各自拥有非共享 QKV 和 FFN，因为像素 patch、文本 token、连续动作 token 的统计结构差异很大。但三者产生的 Q/K/V 会进入 joint-attention，按照 mask 交换信息。

> [!question] HyVLA 是否像 π0.5 一样用 Adaptive RMSNorm 注入 state？
> 原文没有报告 AdaRMSNorm/AdaLN。它明确把 projected robot state 作为独立 `[s_t]` block，位于 perception block 与 noisy-action block 之间。不要由“HyVLA 借鉴 π0.5”推断 state 注入实现相同。详见 [[状态条件注入方式比较]]。

```text
完全共享参数 → 动作控制被迫适应语言模型的计算方式
完全隔离模块 → 动作又读不到语言和视觉语义
MoT          → 计算路径专门化，交互位置共享
```

> [!warning] “分塔”不等于冻结 VLM
> 论文明确写了 pre-training 和 SFT 阶段 all parameters trainable。Action loss 可以经 joint attention 影响 backbone；不能把 MoT 讲成完全阻断动作梯度。

> [!question] 它和 MoE 是一回事吗？
> 不是。MoE 常由 router 为 token 动态选择 expert；MoT 在这里按模态固定走对应参数塔。

关联：[[Mixture of Transformers]] · [[Action Expert]]

---

## 4. Block-wise Causal Attention：谁能看谁？

普通自回归 Transformer 使用 token 级下三角 mask；这里三个 block 的内部需求不同：图像与同一动作块内部需要双向交流，同时又不能让感知表征偷看 noisy action。

![[90 Attachments/HyVLA/hyvla-blockwise-mask.svg]]

| Query 来自 | 可读取的 Key / Value | 设计原因 |
|---|---|---|
| Perception `P` | `P` | 感知前缀不依赖状态/噪声动作 |
| State `S` | `P + S` | 状态需要与场景语义对齐 |
| Action `A` | `P + S + A` | 动作读取全部条件；chunk 内联合生成 |

以 `5P + 2S + 3A` 为例：任意 `P_i` 可看所有 P，但看不到 S/A；任意 `S_i` 可看所有 P/S；任意 `A_i` 可看 P/S 和整个 A 块，包括未来动作位置。因此它叫 **block-wise causal**，而不是严格 token-wise causal。

### 4.1 为什么这个 mask 同时服务于速度？

Flow solver 的 10 步中，图像、语言和当前状态不变，变化的只是 noisy action block。于是 `P/S` 的 Key、Value 可以缓存，后续步主要重算 A。这就是 mask 与 [[KV Cache]] 的联系。

> [!question] Action block 内看到“未来”不会泄漏答案吗？
> 训练输入是整段加噪动作 `a_τ`，任务本来就是联合恢复整个 chunk，而非逐 token 自回归预测；所以块内双向符合生成目标。

关联：[[Block-wise Causal Attention]]

---

## 5. Compact Memory：K 帧历史怎样不把 VLM 撑爆？

直接拼 K 帧，会让上层 VLM 一直处理 `K·n` 个视觉 token。HyVLA 的答案是：**在图像编码器内把历史压进当前帧，不让上层 VLM保存全部历史。**

![[90 Attachments/HyVLA/hyvla-compact-memory.svg]]

### 5.1 一次 memory block 的数据流

```text
输入：X ∈ R[K, n, d]
1. 对每个 patch 位置 p：沿 K 个时刻做 causal temporal attention
2. 对每个时刻 k：在该帧 n 个 patch 间做 spatial attention
3. 每隔 4 个 ViT layer 插入上述结构
4. 上层只保留当前帧 X[K-1, :, :]
输出给 VLM：R[n, d]，不是 R[K·n, d]
```

时间注意力复用原 ViT 的 QKV 与输出投影 `W_O`，使用固定 sinusoidal temporal encoding，且令当前帧位置编码 `e(0)=0`。论文强调没有新增可学习参数。

### 5.2 复杂度省在哪里？

- 每帧空间注意力：`O(K·n²)`；
- 每个 patch 跨时间：`O(n·K²)`；
- 合计：`O(Kn² + nK²)`；
- 避免把全部时空 token 摊平后的 `O(K²n²)`。

> [!question] K=1 预训练，K=6 SFT，不会接口突变吗？
> 论文把 K=1 设计成原图像编码器的精确特例。于是可以先用海量单帧数据预训练，再在任务 SFT 注入 6 帧历史。

RoboTwin full 为 `90.9/90.1`（Clean/Randomized），去掉 memory 后为 `88.8/88.6`，支持“历史有帮助”；但没有拆开验证 temporal mask、插入频率和位置编码各自的贡献。

关联：[[Compact Memory Encoder]]

---

## 6. Relative EEF Action：跨 embodiment 的接口究竟是什么？

双臂动作每步 20D，每只手 10D：

```text
Δposition 3D + Δrotation 6D（SO(3) 前两行）+ gripper 1D
= 每臂 10D；双臂 20D；整个动作块形状约为 [H,20]
```

每个位姿相对于 action chunk 起点的末端坐标表达。同一个“向杯子左侧移动 3 cm”的末端意图，可由不同机器人的 IK 解成不同关节动作，因此 policy 不必记住每台机器人的 joint topology。

### 它没有解决什么？

- 新机器人是否可达、IK 是否有解；
- 自碰撞和环境碰撞；
- humanoid 的躯干与头部怎样配合。

论文做了工程补丁：JAKA 轨迹经过 IK feasibility filtering；humanoid 数据被限制在 reachable shell；额外 24D torso/head 控制由确定性启发式生成，而不是 policy 预测。

> [!warning] 正确表述
> 这是 **embodiment-agnostic action interface**，不是“任意机器人零样本通用”。作者明确不主张 zero-shot generalization。

关联：[[Relative EEF Action]] · [[Action Chunking]]

---

## 7. Flow Matching：为什么从噪声出发？

面对障碍物，左绕和右绕都可能正确。若单次 MSE 回归条件均值，平均轨迹可能撞上正中间。Flow Matching 学习条件动作分布的运输速度场，可以保留多种合理轨迹。

![[90 Attachments/HyVLA/hyvla-flow-and-deploy.svg]]

### 7.1 训练构造

取真实动作块 `A`、高斯噪声 `ε`、随机时间 `τ∈[0,1]`：

```math
A_τ = τA + (1-τ)ε,  因而 dA_τ/dτ = A-ε
```

网络输入 `P,S,A_τ,τ`，预测从噪声指向真实动作的速度。

### 7.2 推理过程

```text
a₀ ~ N(0,I)
repeat 10 times:
    v ← action_expert(P,S,aτ,τ)
    aτ+0.1 ← aτ + 0.1·v
return a₁
```

论文用 10 个 Euler steps，`Δτ=0.1`，一次生成完整 chunk。

> [!danger] 论文内部符号冲突
> Sec. 2.3 的 Eq. (3) 把监督目标写作 `ε-A`，但同节插值和从 `τ=0` 到 `1` 的正向积分要求 `A-ε`；Sec. 4.2 又明确使用 `u=a-ε`。复现时应检查官方代码或勘误，不能静默选一个符号。

> [!question] 这就是 diffusion 吗？
> 都从简单噪声生成数据，也都多步更新；但 Flow Matching 直接监督连续概率路径上的速度场，与常见噪声预测 diffusion 的训练目标不同。

关联：[[Flow Matching]] · [[Action Expert]]

---

## 8. 三阶段训练：每个阶段分别学什么？

### Stage 1 — Hy-UMI-10K continued pre-training

| 项目 | 配置 |
|---|---|
| 历史 / 视觉 | K=1；3 views；224×320 |
| 动作块 | H=50 @ 10 Hz，覆盖 5 秒 |
| 步数 / batch | 200K / global 1,024 |
| 优化 | AdamW，bf16，all parameters trainable |
| 学习率 | peak 5e-5，warmup 1K；160K 衰减到 1/10，再训练 40K |
| 归一化 | state/action 用全数据集 mean/std |

这一阶段学习广覆盖的人类操作先验，不是某台机器人的任务成功率。

### Stage 2 — Task-specific SFT

真实机器人：`K=6`，`H=50 @ 50 Hz`（覆盖 1 秒），历史间隔 1 秒，60K steps，batch 32，LR 2.5e-5 并在 40K steps 衰减，all parameters trainable。

- Track A：同 embodiment 适配，使用目标机器人 demonstrations；
- Track B：跨 embodiment，只用 UMI 数据，不采目标机器人 teleoperation。

RoboTwin：50 tasks，每任务 50 clean + 500 randomized，共 27.5K episodes、超过 6M frames；stride 3、`H=20`、历史间隔 `5×stride`、batch 128。

> [!note] 模拟数据还做了清洗
> 作者用 HDBSCAN 聚类 episode length，删除 noise、少于 100 样本的小模式，以及最长有效模式 top-5% 尾部。模拟动作同时预测 relative 与 absolute EEF，沿 chunk 维拼接并在部署时用 SLERP 融合。

### Stage 3 — FlowPRO 长尾纠错

SFT 已会大多数情况，接下来最值钱的数据是“策略刚失败，人把它救回来”的决策边界：

```text
policy rollout
  ├─ 正常 → 不干预
  └─ 即将失败 → 人接管 → rollback → correction
                         ↓
                losing / winning pair
                         ↓
                    RPRO + SFT
```

关联：[[FlowPRO]]

---

## 9. FlowPRO：没有 reward model，偏好从哪里来？

### 9.1 Intervention-and-rollback 如何构造 pair？

1. operator 发现错误动作并记录失败分支；
2. 回滚到错误前的近似状态；
3. 人工执行正确动作，记录 preferred 分支；
4. 对某状态缺失的一侧，用平滑插值补出对应动作；
5. 形成相同/相近状态下的 winning vs losing 轨迹。

这比只有 episode 成败标签更密集：它告诉模型“就在这个局部状态，哪段动作更好”。

### 9.2 RPRO 优化

论文将 flow loss 记为 `ℓ`，构造隐式偏好分数：

```math
r_θ = β²(ℓ_ref - ℓ_θ)
```

preferred action 要比 reference 拟合得更好，rejected action 不应继续被高概率拟合；symmetric proximal regularizer 限制策略偏离 reference，同时仍混入 SFT batch 防遗忘。

数据混合：第 1 轮 `80% 新 preference + 20% SFT`；第 2 轮起 `70% 新 + 15% 历史 preference + 15% SFT`。

- 3 轮 × 25K steps = 75K；global batch 20（5/GPU）；
- LR 1e-5，warmup 1K，cosine 15K 后到 2.5e-6；
- 每任务 preference pairs 不超过 `O(10²)`。

> [!warning] Reward-free ≠ feedback-free
> 它不训练 reward model、value model 或 critic，但仍需要人判断失败、回滚并给出 correction。

---

## 10. 异步部署：4B 模型怎样驱动 50 Hz 控制？

### 10.1 Producer–consumer 而不是同步停等

- inference 线程根据最新 observation/history 生成 chunk 并覆盖 buffer；
- execution 线程以 servo rate 逐点弹出动作并记录执行历史；
- 两者并行，所以模型推理期间机器人继续执行上一 chunk。

> [!warning] 最容易误报的数字
> `50 Hz` 是真实机器人 action execution rate，不表示 4B backbone 每秒 forward 50 次。论文没有报告模型 latency、GPU/显存、replanning frequency 或端到端 lag。

### 10.2 新 chunk 到达时为什么不能直接切换？

生成期间机器人已经前进，新 chunk 的开头已过期。直接从第 1 点开始会回跳或产生速度不连续。论文：

1. 估算 stale prefix `K=ceil(N/α)`，且 `K≤N-3`；
2. 丢弃过期前缀，保留未来段 F；
3. 选连接点 `c=clip(floor(γM),1,M-2)`；
4. 以当前点 `P₀`、未来点 `P₃`，从历史/未来估计两端切向；
5. 控制柄长度 `λ=σ·distance`；位置用 cubic Bézier，姿态用 SLERP，夹爪线性插值。

因此位置轨迹具有 `C¹` 连续性：位置连续，连接处速度方向也连续。

---

## 11. 实验：哪些数字回答了哪些问题？

### 11.1 RoboTwin：整体与消融

每个任务 100 rollouts，50 个任务平均：

| 方法 | Clean | Randomized | 能说明什么 |
|---|---:|---:|---|
| HyVLA full | **90.9** | **90.1** | 完整系统表现 |
| w/o Compact Memory | 88.8 | 88.6 | 历史视觉约贡献 1.5–2.1 点 |
| w/o Memory + UMI pretrain | 88.1 | 87.9 | 组合移除进一步下降约 0.7 点 |

最后一行同时去掉两个因素，不能把差值全部归因于 UMI；最干净的 UMI 独立对照并不充分。

### 11.2 真实机器人迁移

- Dobot Track A：4 tasks，每任务 300 robot demos，共约 18 h；
- JAKA Track B：300 条 UMI、1.2 h，成功率 90%；
- Astribot Track B：200 条 UMI、1.5 h，成功率 89%；
- Unitree 力控任务：400 条 UMI、2.2 h。

Track B 没有目标机器人 teleoperation，但仍需要机器人侧适配、IK、可达域筛选和部署工程。

### 11.3 FlowPRO 三轮后

每项 3 seeds、每 seed 100 rollouts：

| 任务 | Success rate | 平均完成时间 |
|---|---:|---:|
| Bottle | 99±0.6% | 16 s |
| Cap | 99±0.7% | 21 s |
| USB | 98±0.9% | 22 s |
| Zip | 94±1.1% | 37 s |

DAgger 为 93/88/86/83%，π0.6* 为 95/95/95/89%。这支持 FlowPRO 修复局部长尾失败，但任务数仍少，且依赖人工介入质量。

---

## 12. 证据地图：什么被证明，什么只是系统选择？

| 主张 | 证据强度 | 理由 |
|---|---|---|
| 历史视觉提升 RoboTwin | 较强 | 有直接 w/o memory 消融 |
| FlowPRO 优于所列 baseline | 较强 | 多任务、多轮、3 seeds 对照 |
| UMI 预训练带来泛化 | 中等 | 有组合消融和真实迁移，独立因果隔离有限 |
| relative EEF 支持跨 embodiment | 中等 | 两个平台演示，但有可达域/IK 筛选 |
| MoT 优于共享 Transformer | 未充分证明 | 缺少独立架构消融 |
| 异步 + Bézier 优于同步部署 | 未充分证明 | 缺少 latency、jerk、失败率对照 |
| 任意机器人零样本泛化 | 没有证明 | 作者明确不作此主张 |

---

## 13. 最值得学的三个设计思想

1. **不让一种表示包办一切。** MoT 解决参数如何分工，block-wise mask 解决信息如何流动，两者不能混为一谈。
2. **跨 embodiment 靠接口抽象，也靠工程约束。** Relative EEF 抽掉 joint topology，但可达性、IK、躯干控制仍需系统层补齐。
3. **部署不是推理后的附录。** Action chunking 解决未来一段动作，异步 buffer 解决等待，Bézier 解决 chunk 交界。

## 14. 仍然想追问作者的问题

1. Eq. (3) 的 Flow Matching 符号究竟是排版错误还是时间定义不同？官方代码如何实现？
2. MoT、block-wise mask、relative EEF 各自的独立消融在哪里？
3. 实际使用什么 GPU、显存和精度？一次 10-step solver 延迟是多少？
4. stale-prefix 估计误差在高速接触任务中产生多大 jerk？
5. 去掉 mocap、加入动作标签噪声后，10K 小时优势还剩多少？
6. Track B 扩大到更多形态后，reachable-shell filtering 会不会成为瓶颈？

---

## 15. 30 秒复述模板

> HyVLA 把 VLA 当成系统工程。它用 mocap UMI 获取大规模精确的人类末端轨迹；用 MoT 让视觉、语言、动作拥有专属参数但共享 joint attention；用 block-wise mask 保证条件到动作的单向信息流；用 Compact Memory 把多帧历史压入当前视觉 token；再用 Flow Matching 生成 relative-EEF action chunks。SFT 后，FlowPRO 用 intervention-and-rollback 形成偏好对，专门修复长尾失败。部署端用异步 buffer 和 Bézier stitching，让慢速大模型输出仍可驱动 50 Hz servo。它展示了跨 embodiment 迁移，但不等于任意机器人零样本泛化。

## 16. 自测：答不出来就回看对应图

- [ ] MoT 中“独立”和“共享”分别在哪个算子？
- [ ] 为什么 Action block 内可双向看完整 chunk？
- [ ] 为什么 P/S 能做 KV cache，而 A 每步要重算？
- [ ] Compact Memory 为何从 `[K,n,d]` 回到 `[n,d]`？
- [ ] `O(Kn²+nK²)` 来自哪两次注意力？
- [ ] Relative EEF 为什么没有消除 reachability？
- [ ] 按插值式求导，Flow Matching 的速度符号是什么？
- [ ] Reward-free 为什么仍需要人类反馈？
- [ ] 50 Hz 描述 inference 还是 execution？
- [ ] 哪些模块有独立消融，哪些主要依赖系统演示？

## 关联概念

[[UMI]] · [[Mixture of Transformers]] · [[Block-wise Causal Attention]] · [[Compact Memory Encoder]] · [[Flow Matching]] · [[FlowPRO]] · [[Relative EEF Action]] · [[Action Chunking]] · [[KV Cache]]
