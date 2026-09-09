---
type: paper
title: "N0-VTLA: Scaling Vision–Tactile–Language–Action Model with Latent Tactile Tokens"
year: 2026
status: read
verification: full-paper-and-appendix-checked
topics: [VLA, tactile, latent-tactile-token, future-prediction, flow-matching, offline-RL, ALTER]
source: "[[N0-VTLA.pdf]]"
code: https://github.com/neoteai/N0-VTLA
---

# N0-VTLA：不等碰撞发生，先预测这段动作将带来什么接触

上级地图：[[VLA 学习地图]] · 主题：[[触觉机器人学习]]  
核心概念：[[Latent Tactile Token]] · [[预测触觉与观测触觉]] · [[ALTER]]  
横向对比：[[触觉 VLA 方法比较]]

原文：[[N0-VTLA.pdf|论文 PDF]] · [[n0-vtla-guide.html|HTML 原版讲解（系统浏览器打开）]] · [代码仓库](https://github.com/neoteai/N0-VTLA)

> [!abstract] 一句话结论
> N0-VTLA 不把触觉当成额外图像塞入 VLM，而把“未来 H=50 步动作导致的净触觉变化”压成 10 个 latent tokens；动作专家据此生成 action chunk。ALTER 再把部署轨迹中的进步或退步变成 `Advantage: positive/negative` 文本条件，用固定数据离线改进策略。

## 先看原论文 pipeline

![[n0-vtla-fig1-overview.png]]

```text
当前 RGB + 指令 + robot state → π0.5 的 Vision-Language Prefix
当前触觉差分 → 冻结 DINOv2 → tokens g
                 g + VL Prefix → Predictor → 未来触觉潜变量 z
                                       ↓
                      z + noisy action → Flow Action Expert → H=50 动作

部署数据 → 阶段/掉落/HIL 事件 → Pairwise Progress Model
        → 阶段内 advantage 标签 → 作为文字条件离线训练策略
```

## 它到底在修补 π0.5 的哪个缺口？

π0.5 能凭视觉与语言判断“插头大概对齐了”，却看不到接触面的真实约束：究竟进入孔内，还是顶在孔沿；夹瓶口时，究竟刚好夹稳，还是已经把软瓶压瘪。若仅输入当前触觉，机器人只能在接触发生后反应。

N0-VTLA 的设计链条是：

```text
动作会改变未来接触
→ 未来接触变化比静态触觉更接近动作后果
→ 先预测未来 chunk 的净触觉变化
→ 用该预测条件化 action generation
→ 动作模型在接触发生前就能选择更合理的轨迹
```

注意：这并不证明实时当前触觉没有价值。N0-VTLA 选择一条紧凑的预测路径；[[N0-TWAM]] 后续把未来预测与当前观测拆成互补双路径。

## 完整前向过程：每个 tensor 在哪里流动？

### 1. π0.5 主干

- PaliGemma 编码多视角 RGB 与语言；离散化 robot state 也进入 prefix；
- flow-matching Action Expert 从噪声去噪出动作块；
- 统一容器为 `32D state/action × H=50`。前 20 维是两个 10D 手臂槽：`3D position + 6D rotation + 1D gripper`；余下 12 维为零；单臂只占第一个槽；
- 原始 absolute EEF action 在训练前转成以 chunk 起点为锚的 relative action，部署时做逆变换；归一化统计按机器人与 action schema 单独计算。

### 2. 当前触觉怎样变成 g？

对第 k 个触觉视角，在 episode 起点记录静止张开至少 0.5 秒的基线 `tᵏ₀`：

$$d^k_\tau=t^k_\tau-t^k_0.$$

差分减掉固定 gel 外观和安装压痕，但只能提高鲁棒性，不能保证对温漂、基线移动或永久形变严格不变。每幅差分图经冻结 DINOv2 与可训练投影后，取一个 class token，加上把 `16×16` patch grid 自适应池化成 `3×3` 的九个空间 token，所以每视角得到 10 个 token；n 个有效视角组成：

$$g\in\mathbb{R}^{10n\times d}.$$

### 3. z 不是当前触觉的压缩，而是未来触觉的预测

![[n0-vtla-teaching-latent.svg]]

预测器读取 `g`、已经语境化的 VL prefix 和 10 个 learned queries，输出：

$$z\in\mathbb{R}^{10\times d}.$$

训练目标来自同一动作块末端与当前帧之间的触觉变化：

$$z^*=\frac{1}{n}\sum_{k=1}^{n}f_{enc}(t^k_{\tau+H}-t^k_\tau),\qquad H=50.$$

因此：

- `g` 回答“此刻相对 episode 起点发生了什么接触”；
- `z*` 回答“执行接下来 50 步后，触觉相对现在净改变多少”；
- `z` 在不知道未来帧的情况下预测 `z*`；推理时当然不需要未来触觉。

多视角的目标在 view 维平均，因此 z 是紧凑的共享后果摘要，而非保留每根手指完整几何的高保真未来触觉视频。

### 4. 动作专家实际看见什么？

z 投影到 Action Expert 的宽度后，放在 noisy action tokens 之前。当前触觉 tokens `g` **从不直接进入 Action Expert**，只能先经 Predictor 蒸馏成 z：

```python
def n0_vtla_forward(rgb, prompt, state, tactile, tau):
    prefix = paligemma(rgb, prompt, discretize(state))

    # 每个 tactile view：当前帧减 episode 基线；每视角 1+9 tokens
    g = concat([
        tactile_projection(frozen_dinov2(view.current - view.episode_start))
        for view in tactile.active_views
    ])
    if tactile.has_no_active_view:
        g = learned_null_tactile_token

    z = tactile_predictor(current_touch=g,
                          vl_context=prefix,
                          queries=learned_queries(10))

    noisy_action = gaussian_noise(shape=[50, 32])
    for t in euler_schedule(steps=10):
        # action 能读 prefix 和 z；prefix 不反向读取 z
        noisy_action = flow_action_expert(prefix, z, noisy_action, t)
    return denormalize_and_unanchor(noisy_action)
```

若整条 episode 没有触觉，learned null token 代替 g，z 仍会产生，模型退化成 VL 控制而不是接口崩溃。

## 为什么必须分三阶段训练？

![[n0-vtla-teaching-stages.svg]]

### Stage 1：先固定 z 的语义

![[n0-vtla-fig2-stage1.png]]

只训练 Predictor，使 z 对齐 z*。损失由 mean-pooled、L2-normalized 表示上的对称 InfoNCE 与未来触觉粗差分场的 L1 重建构成：

$$\mathcal L_1=\mathcal L_{NCE}+\lambda_{rec}\mathcal L_{rec}.$$

因果目的：若一开始只给 action loss，z 可以抄 VL prefix、被动作专家绕开，或编码任何能偶然降损的东西；future-tactile target 先规定“这个瓶颈必须代表未来接触”。

### Stage 2：强迫旧 Action Expert 学会读 z

冻结 Stage 1 的触觉感知栈，只训练 latent-to-expert projection 和 Action Expert。对 action queries 屏蔽 VL prefix 的 key/value，使其只能看 z 与 noisy action：

```python
def stage2(batch):
    z = frozen_tactile_path(batch)
    action = action_expert(
        noisy_action=batch.noise,
        latent_touch=z,
        vl_prefix=batch.prefix,
        mask_prefix_for_action_queries=True,
    )
    return flow_matching_loss(action, batch.target_action)
```

这不是部署时永久丢掉视觉，而是一段接口课程：不给模型走原来的视觉捷径，它才会认真学会如何消费全新的 z。

### Stage 3：恢复视觉路径，端到端协同

解除 mask；除了始终冻结的 DINOv2 backbone，其余模块在 action objective 下联合训练。此时不再施加 Stage 1 的 contrastive/reconstruction targets，避免固定的辅助目标妨碍最终动作最优。

> [!warning] 三阶段的证据边界
> 论文给出了 Stage 1 表示检索和扰动探针，说明 z 确实预测且偏向触觉；但没有给出“完整三阶段 vs 一步到位”在全部下游任务上的定量消融。因此训练课程的稳定性逻辑合理，尚不能当成已被完整消融证明的结论。

## ALTER：为什么不是简单筛掉失败轨迹？

失败轨迹也可能包含一段正确操作、一次失败和一段有效恢复；整条删除会浪费数据。ALTER 的问题改写是：在同一任务阶段内，`t → t+H` 是在进步还是退步？

![[n0-vtla-fig6-alter.png]]

![[n0-vtla-teaching-alter.svg]]

### 监督从哪里来？

1. **干净示范的稠密进度对**：先用触觉接触变化、EEF 运动、夹爪状态与视觉 GEBD 候选定位边界，再由 VLM 把预切分区间映射到人工审核过一次的 L3目标→L2阶段→L1步骤模板；
2. **不完美部署的稀疏事件对**：触觉接触丢失定位 object drop，HIL 日志给出纠正起止；掉落前应高于掉落后，纠正末端应高于纠正起点；
3. 事件信号只构造标签，Pairwise Progress Model 的输入本身是成对的多视角 RGB 与任务 prompt，不输入触觉/运动学事件。

阶段 k 按干净示范的平均耗时分配任务进度宽度：

$$w_k=\frac{\bar d_k}{\sum_j\bar d_j},\qquad
\phi_t=\sum_{j<k}w_j+w_ku_t.$$

这样短暂过渡不会和长时间操作被硬分成一样宽。进度模型同时拟合稠密差值，并用 margin ranking 学稀疏事件：

$$\mathcal L_{prog}=\mathbb E[ A_\theta(x_a,x_b)-(\phi(x_a)-\phi(x_b))]^2
+\lambda_{event}\mathbb E[\max(0,m-yA_\theta(x_a,x_b))].$$

论文设置 `m=0.02`；事件 pair 约占采样的 5%，loss multiplier 为 0.1，并随机交换 pair 顺序防止模型记住输入槽位。

### 怎样生成 advantage 标签？

```python
def label_deployment_episode(frames, episode_start, H=50):
    for t, x_t in enumerate(frames):
        global_progress[t] = recovery_offset + A_theta(x_t, episode_start)
        local_change[t] = A_theta(frames[min(t + H, len(frames)-1)], x_t)

    smooth_progress = median_filter(global_progress, window=5)
    # 附录用累计最大值避免阶段编号倒退；局部变化仍允许为负
    stage = duration_calibrated_stage(cumulative_max(smooth_progress))
    for k in unique(stage):
        threshold = quantile(local_change[stage == k], 0.70)
        label[stage == k] = where(local_change >= threshold,
                                  "Advantage: positive",
                                  "Advantage: negative")
    return label
```

- 每个阶段内局部变化最高的 30% 标 positive，而不是跨阶段比较；
- staged-recovery episode 用三个干净示范起点估算起始 progress offset；短于 H 的尾段做 horizon normalization；
- 训练时 0.3 概率省略 advantage tag，避免模型完全依赖标签；部署时始终用 `Advantage: positive`；
- 策略结构和 flow-matching loss 不变，且不新增环境交互，所以它是 advantage-conditioned offline RL，而非在线 RL。

## 数据怎样跨机器人对齐？

NeoData 含 ARX X5、UR5e、Flexiv、Franka、Piper 与 Neo TacUMI，单臂/双臂、机器人/UMI 式手持数据均转为 LeRobot 格式。每条样本含三路 RGB、语言、本体状态，以及单臂两路或双臂四路触觉流；各模态以 30 Hz 同步，模型输入统一 resize 到 `224×224`。

> [!note] 不应填假的规模数字
> 本报告把精确 NeoData 规模放在 companion data report，并在附录明确省略硬件型号、设备数、显存和吞吐。因此这里只记已核实的多平台结构，不补猜测 GPU 或总小时数。

## 部署时最容易踩的坑

- 一次请求执行一次 prefix encoding，再做 10 个 Euler steps，返回完整 `H=50` chunk；
- 论文要求完整执行整个 chunk 后再请求。提前截断会丢掉常集中在 chunk 尾端的闭爪命令；
- reset 必须清空触觉基线，reset 后第一帧成为新 baseline；缺失视角按自己的 baseline/no-contact 处理；
- transport protocol 未公开，不能从模型结构推断 ROS topic、线程或 RPC 细节。

## 实验到底证明了什么？

### NeoReal：9 个真机任务

- N0-VTLA 9/9 任务胜出，平均成功率 `47.2%`，π0.5 为 `29.4%`；平均 progress score `56.8` vs `42.3`；
- Socket Plugging：`85% vs 60%`，触边后会抬起、重对齐、再插；
- Board Insertion：`25% vs 0%`，两个视觉基线均为 0；
- Bottle Standing：`30% vs 0%`，通过小幅调节夹爪开度避免捏软瓶并把它提起。

### 20 个仿真任务

- UniVTAC 8 任务：`83.1%`，最强外部 baseline InternVLA-A1 为 `67.1%`；
- NeoSim 12 任务：`50.8%`，π0.5 为 `45.8%`；单臂均值 `73.8%`，双臂均值 `39.4%`；
- 合并 20 任务：`63.8%`，最强总体 baseline π0.5 为 `44.0%`。

### ALTER：同一离线方法是否还能叠加在触觉预训练上？

| 方法 | Towel Folding | Bag Packing | Cardboard Box Folding |
|---|---:|---:|---:|
| π0.5-SFT | 40 | 20 | 5 |
| N0-VTLA-SFT | 50 | 35 | 20 |
| π0.5+χ0 | 80 | 65 | 55 |
| π0.5+ALTER | 90 | 75 | 60 |
| **N0-VTLA+ALTER** | **95** | **80** | **75** |

N0-VTLA+ALTER 每小时成功执行量分别为 `43 / 15 / 30`。这说明触觉预训练优势没有被 ALTER 抹平；但 ALTER 是 task-specific progress model，尚非零样本通用 reward model。

### z 真的是预测未来触觉吗？

- 378 个 held-out query、约 32 个候选池中，z 检索对应 z* 的 top-1 为 `92.3%`，chance `3.2%`；这是 pool 内检索，不是全库检索；
- 只用当前 g 检索为 `57%`；候选池扩大到 128 时，g 为 `40%`、predictor 为 `81%`，说明 z 不只是复制当前触觉；
- 替换触觉输入使 z 的 centered-cosine distance 约变 `0.9`，替换 RGB+prompt 不超过约 `0.2`；触觉/视觉语言敏感度比 Stage 1 为 `4.3`，端到端后约 `1.4`；
- 固定 observation 与 sampling noise、移除触觉的反事实探针显示：接触/抓取时动作路径明显分离，自由空间中近乎重合。

这些证据支持“z 由触觉主导、携带超过当前触觉的未来信息、并在接触关键时刻改变动作”。它仍不能单独证明提升全部来自“预测”而非更多数据、传感器体系或训练课程；更严格的 current-only / predicted-only / both 消融仍值得做。

## 与已有三篇触觉路线怎么区分？

| 方法 | 它把触觉当成什么 | 时间方向 | 最适合回答的问题 |
|---|---|---|---|
| [[T-Rex]] | 快速专家的当前触觉输入 | 事后高频纠偏 | 大 VLA 太慢，接触后如何迅速改动作？ |
| [[FTP-1]] | 统一功能槽中的当前触觉 | 当前、跨硬件 | 不同传感器怎样共享预训练？ |
| **N0-VTLA** | 未来 chunk 净触觉变化的紧凑 latent | 前瞻 | 怎样用很小接口给 π0.5 增加接触后果预测？ |
| [[N0-TWAM]] | 未来生成 + 当前力场双路径 | 前瞻 + 反射 | 怎样同时预测、纠错，并扩展成长时序 WAM？ |

N0-VTLA 与 N0-TWAM 名字相近但不能混用：前者的 `g → z → Action Expert` 是轻量条件旁路，不生成完整未来触觉，也没有当前触觉直达动作的 reactive port；后者在 MoT 内联合生成未来视频/触觉，再让 action 读取预测结果与当前真实力场。

## 对 HyVLA 最有价值的可复现路线

```python
experiments = {
    "HyVLA":                 base,
    "+ current_touch":       reactive_cross_attention,
    "+ predicted_latent":    n0_vtla_predictor,
    "+ current + predicted": dual_path,
}

# 所有变体固定：数据、参数预算、action horizon、训练步数、控制频率
# 额外测：自由空间/接触阶段的 action divergence、延迟、掉传感器退化
```

若目标是最小改造，N0-VTLA 比复制 N0-TWAM 更贴近 HyVLA：保留 VLM + Action Expert，只外挂 DINOv2、Predictor 和 latent conditioning block。若实验证明接触后的快速误差仍无法处理，再叠加 T-Rex 式高频端口；若换传感器性能崩溃，再引入 FTP-1 的统一 slot 表示。顺序化消融比一次堆齐更能回答性能来源。

## 局限与待验证问题

- 论文触觉范围是自研 vision-based tactile sensors，尚不能外推到任意 taxel、六维力或视触觉硬件；
- 多视角平均成 10 个 z token 很紧凑，也可能丢失“哪根手指、哪个位置”发生接触的细粒度身份；
- z* 是 chunk 两端的净变化，可能抵消 chunk 中间发生又释放的短暂接触；
- 没有 current-touch 直达动作的快速通路，闭环上限仍受 chunk 请求/执行节奏影响；
- ALTER 依赖每任务模板、一次人工审核与 task-specific progress model；阶段内 top 30% 是相对标签，不代表物理意义上的正回报；
- 作者明确提出未来要探索 free-latent 与 supervised 之间更广的触觉表征空间，并在更多任务上验证 ALTER。

## 回顾时先问这六个问题

1. g、z、z* 各自对应哪个时间点？
2. 为什么当前 g 不直接进入 Action Expert？
3. Stage 2 为什么暂时切断 VL prefix→Action？
4. ALTER 为什么必须“阶段内”比较 local change？
5. 92.3% 检索结果证明什么，又没有证明什么？
6. 如果 HyVLA 加这个模块，必须怎样做 current/predicted/both 消融？
