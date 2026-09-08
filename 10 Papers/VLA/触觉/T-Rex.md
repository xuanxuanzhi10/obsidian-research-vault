---
type: paper
title: "T-Rex: Tactile-Reactive Dexterous Manipulation"
year: 2026
status: read
verification: full-paper-and-appendix-checked
topics: [VLA, tactile, dexterous-manipulation, flow-matching, MoT]
source: "[[T-Rex.pdf]]"
---

# T-Rex：让 VLA 不只“看见动作”，还能在接触后及时改动作

上级地图：[[VLA 学习地图]] · 主题：[[触觉机器人学习]]  
核心概念：[[触觉信号表示]] · [[异步触觉动作细化]] · [[Flow Matching]] · [[Mixture of Transformers]]

原文：[[T-Rex.pdf|在 Obsidian 中打开论文 PDF]]

> [!abstract] 一句话结论
> T-Rex 的关键不是给 VLA 多塞一个触觉输入，而是把**慢速视觉规划**和**快速触觉纠偏**分成两个 expert：视觉先把动作从噪声去到中间状态，触觉再用最新接触信号完成最后一段去噪。

## 先看原论文总图

![[trex-fig1-overview.png]]

这张图给出完整因果链：

```text
22,889 h 人类第一视角视频
    ↓ 学到物体、手部运动和语言语义先验
T-Rex Model：Latent / Action / Tactile 三个 expert
    ↑ 用 100 h、7,755 条带触觉机器人轨迹把先验落到接触世界
约 100 条/任务的 post-training
    ↓
12 个需要精细力控、形变感知和滑移恢复的真实任务
```

## 读者首先会问：普通 VLA 到底缺了什么？

相机适合回答“目标在哪里、整体该怎么做”，但接触发生后，决定成败的信号往往不可见：夹得是否过紧、纸页是否已经分开、杯子是否正在滑、钥匙是否卡在锁芯边缘。此时仍按几百毫秒前的画面执行整段动作，就会形成开环误差。

最直接的想法是把触觉 token 拼进 VLA。论文用 `π₀.5 + tactile` 做了这个反例：平均成功率从 17% 降到 6%。这说明问题不是“有没有输入”，而是：

1. 视觉和触觉的有效频率不同；
2. 触觉的时间动态和空间形变不能用一个静态向量概括；
3. 大 VLA 每次重跑太慢，无法让触觉形成高频闭环；
4. 新模态若缺少专门训练阶段，可能干扰已有视觉—动作能力。

因此 T-Rex 同时改了**数据、表示、模型分工和运行时调度**。

## 方法总览：四个问题，四个回答

| 障碍 | T-Rex 的回答 | 为什么可能有效 |
|---|---|---|
| 没有规模化触觉数据 | 100 h、7,755 episodes、22 motor primitives、207 objects | 先学可复用接触原语，再少量适配任务 |
| 触觉同时有历史与空间结构 | 力历史 VQ-VAE + 当前力旁路 + 形变 CNN | 同时保留动态、瞬时冲击和接触几何 |
| 视觉慢、触觉应快 | Action Expert 约 5 Hz；Tactile Expert 约 20 Hz | 规划与反射按频率和职责拆开 |
| 快速触觉不应重算视觉 | 缓存视觉语言 KV 与中间动作状态 | 快 tick 只跑小型 tactile expert |

## 原论文架构图：动作是怎样生成的

![[trex-fig3-architecture.png]]

### 三个 expert 各自负责什么？

- **Latent Expert**：处理图像和语言，并学习预测未来视觉 latent；作用是提供带时间感的语义上下文。
- **Action Expert**：低频生成全局动作计划；它看视觉—语言上下文，把纯噪声动作去噪到边界 `τ_split=0.4`。
- **Tactile Expert**：高频读取最新触觉，不看原始图像；复用缓存，从边界状态继续去噪到可执行动作。

它们使用 Mixture-of-Transformer-Experts：参数路径分开，但在共同注意力空间交换信息。不是传统稀疏 MoE 的“选一个专家”，而是按模态和职责分参数。

## 触觉信号怎样表示？

![[trex-teaching-tactile-token.svg]]

每个指尖有两类原始观测：

- 最近 16 帧六维 wrench：三维力 + 三维力矩；
- 当前的单通道皮肤形变深度图。

论文没有把它们粗暴压成同一个标量，而是走三条互补路径：

```python
class SpatialTemporalTactileEncoder:
    def encode_fingertip(self, force_history, force_now, deform_map, finger_id):
        # force_history: [T=16, 6]，回答“接触怎样变化”
        h = temporal_conv1d(force_history)       # 两个 stride block
        h = temporal_mean_pool(h)                # [256]
        force_code = nearest_code(h, K=64)       # 1 个离散动态 token
        force_code += finger_identity(finger_id) # 共享卷积，但保留指头身份

        # force_now: [6]，防止历史压缩丢掉这一刻的冲击
        instant = linear_project(force_now)

        # deform_map: [1, H, W]，回答接触在哪里、是否滑移/剪切
        spatial = frozen_resnet18_stage1_to_3(deform_map)
        spatial = flatten_and_project(spatial)

        return concat(force_code, instant, spatial)
```

为什么对力历史做 VQ-VAE？连续传感器会漂移，而且大多数帧都是弱接触/无接触，直接回归容易让表示被背景主导。作者把 256D 连续表示量化到 64 个 code，并用：

- EMA 更新 codebook；
- 定期重置低使用率 code；
- magnitude-weighted MSE，让大力接触帧承担更高重建权重；
- 五指共享卷积权重，并注入 finger identity embedding。

最终得到“每只手每根手指一个动态 token”。形变编码器则由自监督卷积 autoencoder 预训练，进入 policy 训练后冻结，以稳定接触几何表示。

## 触觉怎样真正改写动作？

![[trex-teaching-slow-fast.svg]]

### 先理解论文的 Flow Matching 记号

论文定义：干净动作 `x₀=A`，高斯噪声 `x₁=ε`，训练目标 `v*=ε-A`。推理从 `τ=1` 向 `0` 积分，所以 Euler 步长为负数。看似目标指向噪声，但负步长会把状态推回干净动作；符号是自洽的。

```python
def flow_train_sample(action_demo, context):
    eps = randn_like(action_demo)
    tau = sample_beta(1.5, 1.0)
    x_tau = (1 - tau) * action_demo + tau * eps
    target_velocity = eps - action_demo
    return mse(expert(x_tau, tau, context), target_velocity)
```

### 推理不是两次独立预测，而是同一条轨迹的接力

```python
class TReXCascadedPolicy:
    def slow_tick(self, rgb, language):
        # x: [action_chunk=16, action_dim=62]
        x = gaussian_noise(shape=[16, 62])
        vl_context = latent_expert(rgb, language)

        # 6 个 Euler step：tau 1.0 -> 0.4
        for tau in [1.0, .9, .8, .7, .6, .5]:
            v = action_expert(x, tau, vl_context)
            x = x - 0.1 * v

        # 缓存不仅含视觉语言 KV，也含 tau=.4 的动作位置重编码
        with execution_lock:
            shared.x_boundary = x.detach()
            shared.kv = concat(KV_lat(vl_context), KV_action_at_boundary(x))

    def fast_tick(self, tactile_now, offset):
        # 在 action chunk 的 offset = 0, 4, 8, 12 触发
        with execution_lock:
            x = shared.x_boundary.clone()
            kv = shared.kv.clone()

        tactile_tokens = tactile_encoder(tactile_now)
        # 4 个 Euler step：tau .4 -> 0
        for tau in [.4, .3, .2, .1]:
            v = tactile_expert(x, tau, tactile_tokens, kv)
            x = x - 0.1 * v
        executor.replace_future_chunk(offset, x)
```

快流不重跑 visual tower、latent expert 或 action expert。它复用边界动作和 KV cache，因此可以更频繁地依据接触状态替换未来 action chunk。运行时用单线程请求 socket 和 execution lock，防止快流读到尚未提交完整的缓存。

### 为什么训练时还要人为制造“不同步”？

部署时快触觉和冻结的视觉缓存天然有时间差。如果训练总给完美同步数据，模型上线会遇到分布偏移。因此作者采样：

```python
delay = random_choice([0, 4, 8, 12])
tactile_context = tactile_frames_shifted_from_visual(delay)
```

这不是普通数据增强，而是在训练中复现运行时的 **staleness distribution**。

## 数据为什么按 motor primitive 组织？

![[trex-fig2-dataset.png]]

若直接为每个下游任务采集海量触觉轨迹，数据会被任务组合数量拖垮。T-Rex 把操作拆成 close、peel、wrap、fold、wipe、squeeze、insert、extract 等 22 个接触原语，再与 207 种物体组合。候选 `207×22=4554` 先做可行性筛选，留下 502 个 object–primitive 组合，最终采集 7,755 episodes、100 h。

每条轨迹包含三路单目 RGB、双臂与双手状态、两个 wrist pose、十个指尖的六轴 wrench 和形变深度图，以及语言标注。采集频率为 30 Hz；真实机器人底层控制线程运行在 300 Hz。

数据还主动随机化桌面背景、0–5 个干扰物、物体位置和朝向，并过滤触觉损坏、遥操作失败及极端速度轨迹。核心思想是：**让 mid-training 学“接触动作词汇”，让 post-training 学“如何把词汇组成任务”。**

## 三阶段训练各自解决什么？

```text
Stage 1 — 22,889 h human egocentric pre-training
语义、物体、手部运动与未来视觉先验；此时没有 tactile expert
        ↓
Stage 2 — 100 h tactile-grounded robot mid-training
把视觉动作先验接地到真实接触，学习触觉表示与快慢 expert 协作
        ↓
Stage 3 — 每个任务约 100 条示范 post-training
将通用接触原语组合成具体任务策略
```

总损失为：

$$L=L_{act}+1.0L_{tac}+0.5L_{future}.$$

值得注意：Action Expert 在完整 `τ∈(0,1]` 上训练，而不是只训练慢段，以保留其独立生成能力；Tactile Expert 训练时接收 detached 的慢流边界与缓存，迫使职责真正分开。

## 实验结果到底证明了什么？

### 主结果

12 个任务、每任务 16 次随机位姿试验。平均成功率：

| 方法 | 平均成功率 |
|---|---:|
| ViTacFormer | 3% |
| RDP | 6% |
| Tactile-VLA | 15% |
| EgoScale | 35% |
| π₀.5 | 17% |
| π₀.5 + tactile | 6% |
| **T-Rex** | **65%** |

“比最强基线高 30%”在表中实际是 **65%−35%=30 个百分点**，不要误读为相对提升 30%。

### 消融把因果链拆开了

| 六任务消融 | 平均 | 相对完整模型 |
|---|---:|---:|
| Full | 65% | — |
| 无触觉 | 42% | -23 pp |
| MLP Force + Deform | 58% | -7 pp |
| 仅 Deform | 54% | -11 pp |
| MLP Force + VQ-VAE Force | 59% | -6 pp |
| 无异步机制 | 60% | -5 pp |

可支持的结论是：

1. 触觉本身很重要（去掉后 -23 pp）；
2. 力与形变互补，单一路径不够；
3. 动态 VQ-VAE 比只有 MLP 当前力更好；
4. 异步结构有额外收益，但小于“是否有触觉”的收益；
5. `π₀.5 + tactile` 反例说明**融合方法和训练配方**比“加了传感器”更关键。

但这些是同一平台、同一组任务上的关联/消融证据，不能自动外推到所有机器人和触觉硬件。

## 12 个任务在验证什么？

- **精细接触/分离**：Flip Page、Extract Card、Split Cup；
- **稳定抓持与滑移**：Transfer Egg、Deal Poker；
- **连续力控制**：Wipe Plate、Apply Toothpaste；
- **插入与约束接触**：Open Lock、Refill Tablet、Screw Lightbulb；
- **表面/结构辨别与多阶段操作**：Sort Mahjong、Acid-Base Neutralization。

更好的触觉 benchmark 不应只问最终成功率，还应测：接触检测延迟、滑移恢复率、峰值力、物体损伤、动作平滑性、视觉遮挡下退化、传感器漂移、异步延迟，以及跨材质/跨物体泛化。

## 优势、限制与可迁移启示

### 优势

- 把触觉从静态附加特征升级为高频闭环控制信号；
- 表示层明确区分动态力、瞬时力和空间形变；
- 用缓存和小 expert 把系统频率问题纳入模型架构；
- motor primitive 数据组织适合构建可复用接触能力库。

### 限制

- 长时序、极窄容差任务仍可能需要 RL 或在线交互；
- 传感器形变失真、校准漂移、缺少掌面密集触觉仍是硬件瓶颈；
- 当前触觉表示依赖特定传感器，跨传感器统一表示尚未解决；
- 失败仍包括碰撞、滑移、定位偏差、多指摩擦耦合、过大用力和滑动错位；
- 论文给出约 5/20 Hz 设计频率，但没有报告端到端毫秒级 latency。

## 复现时最容易踩的坑

1. 论文报告 action dimension 为 62、chunk length 为 16，但正文没有逐维列出 62D 的确切坐标拆分；不要擅自写成确定的 `14+44+4`。
2. 手臂动作是 relative EEF delta，手指动作是 absolute joint control；二者不能用同一归一化直觉处理。
3. 快流缓存必须包含边界动作位置的重新编码，不只是视觉 KV。
4. 慢流生成缓存和快流读取缓存之间需要锁，否则会出现上下文与边界状态错配。
5. 训练要模拟视觉缓存变旧，而不是只用严格同步传感器帧。
6. VQ-VAE 若不用大力加权和 code 重置，容易塌缩到常见的无接触状态。

## 与现有 VLA 知识的连接

- 与 [[π₀]]：都用 continuous action flow matching；T-Rex 把去噪轨迹按频率分给两个 expert。
- 与 [[Hy-Embodied-0.5-VLA]]：都强调专门 Action Expert 与部署闭环；T-Rex 的新轴是触觉的异步 fast expert。
- 与 [[Block-wise Causal Attention]]：前者控制 token 间信息方向；T-Rex 更进一步控制不同模态在**什么频率、哪段生成轨迹**参与。
- 与 [[KV Cache]]：缓存不只是提速技巧，而是让慢视觉语境能够被多次快速触觉更新复用。
- 与 [[Action Chunking]]：chunk 提供未来动作缓冲，触觉可以在执行过程中替换尚未执行的部分。

## 原版 HTML 讲解入口

- [[trex-study-guide.html|打开学习指南（原始 HTML）]]
- [[trex-qa-guide.html|打开问答指南（原始 HTML）]]

> [!warning] HTML 与论文证据边界
> 两份 HTML 适合帮助形成直觉，但不是论文原文。其毫秒级延迟数字未在 PDF 中报告；`62=14+44+4` 的动作拆分也无法由正文与表格确认，且与正文“手臂使用 relative EEF delta”存在表述张力。因此正式知识页不把它们作为已核实事实。

## 读完后应该能回答

1. 为什么把触觉 token 拼到普通 VLA 里可能反而变差？
2. 为什么既要力历史 VQ-VAE，又要当前力直接投影？
3. 形变图提供了六轴 wrench 缺少的什么信息？
4. 两个 expert 为什么必须共享同一条 flow trajectory？
5. `τ_split` 在计算量、视觉先验与触觉修正容量之间如何权衡？
6. delay augmentation 为什么属于训练—部署一致性，而不只是鲁棒性增强？

## 来源与核实范围

- PDF：全文 30 页（正文与附录 A–H）已核查，版本 arXiv:2606.17055v2，2026-06-18。
- 原图：Figure 1、Figure 2、Figure 3 均从用户提供 PDF 截取。
- 两份 HTML：作为辅助解释保留，事实以 PDF 为准。
