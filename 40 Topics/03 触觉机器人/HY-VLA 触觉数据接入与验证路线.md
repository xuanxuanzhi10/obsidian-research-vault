---
type: research
title: "HY-VLA 触觉数据接入与验证路线"
aliases: [HY-VLA 触觉研究路线, 指尖触觉接入 HyVLA]
status: active
updated: 2026-09-11
topics: [tactile, HY-VLA, VLA, force, contact-rich, experiment-design]
data: "/home/ubuntu/Desktop/test_single"
verification: "local-data-audited-and-primary-sources-checked"
---

# HY-VLA 触觉数据接入与验证路线

关联：[[Hy-Embodied-0.5-VLA]] · [[触觉机器人学习]] · [[触觉 VLA 方法比较]] · [[触觉信号表示]]

> [!abstract] 先给结论
> 这段 `test_single` 数据已经证明采集链路基本可用：相机、位姿、动作与左右指尖的 3×3×3 三轴力阵列已对齐，触觉并非一条无结构的标量，而是有明显左右差异和空间接触图样。它目前只能做**数据管线冒烟测试与 encoder 预实验**，不能证明触觉给 HY‑VLA 带来任务增益，因为只有 1 个 episode，且任务语义没有被可靠记录。
>
> 对 HY‑VLA，第一版最值得做的是：**每指独立的时空触觉编码器 + 接触门控 + 动作专家中的零初始化 cross-attention 残差**。先保持视觉语言主干不变，在 30 Hz 同步数据上完成可解释消融；证明确有增益以后，再使用原始约 80 Hz 触觉做快分支。不要一开始同时加入未来触觉、异步调度、跨传感器统一和混合力位控制，否则失败时无法定位原因。
>
> 首选验证任务不是气球，而是 **同外观、不同重量/摩擦的滑移恢复**；第二选择是标准化易损物体；第三是末端遮挡的 1–2 mm 精密插入。它们都能制造“视觉相同、物理状态不同”的反事实，因而更能证明增益真的来自触觉。

![[90 Attachments/HyVLA-Tactile-Integration/tactile-causal-chain.png]]

---

## 0. 给触觉小白的三个坐标系

### 0.1 视觉告诉你“东西在哪里”，触觉告诉你“接触正在怎样发生”

视觉擅长物体类别、位置、姿态和长程规划，但以下状态常常看不见：

- 夹爪是否真的夹紧，还是只有外观上闭合；
- 物体是否已经开始微滑，但还没有在图像中产生明显位移；
- 接触力是否集中在一个角点，马上会卡住或旋转；
- 软物体是否已经进入损伤区；
- 插入失败时，约束力来自左、右还是某个角点。

你们的传感器不是“每指一个力”。每个手指有 9 个 taxel，每个 taxel 测量局部 `x/y/z` 三轴力。粗略地说：

- `Fz` 更像压紧程度；
- `Fx/Fy` 更像切向拖拽、偏心和潜在滑移；
- 9 个点的分布告诉模型“力落在哪里”；
- 一段时间的变化告诉模型“接触是在建立、稳定、滑动还是释放”。

这些解释依赖传感器安装方向。由于当前数据没有记录 finger-to-robot 外参，本文只能把 `x/y/z` 当作**每指传感器局部坐标**，不能把左右指的向量直接相加成机器人坐标系中的力。

### 0.2 “把触觉加进 VLA”其实有三种不同问题

| 问题 | 触觉扮演的角色 | 典型输出 |
|---|---|---|
| 感知 | 判断接触、滑移、材质、卡住方向 | latent / contact probability |
| 策略 | 触觉作为条件，改变下一段位置或夹爪动作 | EEF pose / gripper chunk |
| 控制 | 直接调节目标力、阻抗或位置—力掩码 | force target / stiffness / control mask |

你们当前的数据只有位置/夹爪动作标签，没有示范者的目标力、阻抗和位置—力控制掩码。因此当前最合理的是第二种：**让触觉改变 HY‑VLA 的位置与夹爪动作**。ForceVLA2 那种混合力—位控制可以作为第三阶段，但不能从这一个数据格式直接监督出来。[^forcevla2]

### 0.3 “用了触觉”不等于“证明触觉有用”

如果给触觉版增加更多参数、更多示范或不同初始条件，即使成功率变高，也无法知道提升来自触觉、容量还是数据。强证据必须满足：

1. 同一模型骨架、相同示范、相同训练步数；
2. 初始 RGB/状态尽量匹配，只切换触觉输入；
3. 任务中存在视觉不可辨识的隐藏物理状态；
4. 真实同步触觉优于时间错位、打乱或常数触觉；
5. 除成功率外，还能看到峰值力、滑移恢复延迟等过程指标改善。

---

## 1. 对 `test_single` 的数据审计

### 1.1 数据格式：已经很接近可训练数据集

本地目录：`/home/ubuntu/Desktop/test_single`

| 项目 | 实际内容 | 对训练的含义 |
|---|---:|---|
| 格式 | LeRobot v3.0 | 容易接数据加载器 |
| episode | 1 | 只能调通，不能训练/评测结论 |
| 主频率 | 30 Hz | 830 帧，约 27.63 s |
| 图像 | hand camera，480×640 H.264 | 有第一视角视觉条件 |
| state / action | 各 10D | xyz + rotation 6D + gripper |
| IMU | 6D | 当前研究可先不用 |
| touch | 60D | 左右各 `(total + P1…P9) × xyz` |
| validity | `touch_record_present` | 能区分采样记录是否存在 |
| 对齐质量 | touch age p95 ≈ 11.04 ms，最大约 25 ms | 小于 schema 的 50 ms 上限，当前对齐良好 |
| 原始触觉 | 2,786 条 / 34.66 s，约 80 Hz | 可用于后续快分支，而不仅是 30 Hz 对齐帧 |

![[90 Attachments/HyVLA-Tactile-Integration/test-single-tactile-overview.png]]

> [!info] 图怎么读
> 橙色阴影是“左右任一指总力模长 > 1 N”的启发式接触段，不是人工真值。样本中主要有约 `10.97–15.40 s` 和 `20.43–22.83 s` 两段接触。第一段夹爪停在中间开度并保持约 8–10 N，第二段更接近完全闭合。视频中能看到夹爪与桌面物体交互，但任务元数据只有字符串 `23`，因此不能把它正式标注成某个语义任务。

### 1.2 信号里确实有可利用结构

- 每帧触觉记录的 present mask 都为 1；约 52.89% 帧是全零触觉，符合“长非接触 + 短接触”的机器人分布。
- 以任一指合力模长 > 1 N 作为粗略接触阈值，接触帧约占 25.06%。这意味着按普通随机帧训练会严重偏向“无接触”。
- 左指峰值合力约 10.91 N，右指约 10.41 N。
- `total` 与 9 个 taxel 的逐轴求和高度一致（本样本各轴相关性约 0.99，平均绝对误差 < 0.008 N），说明 `total` 可作数据完整性校验，9 点分布则保留空间信息。
- 全部 60 个通道中约 90.59% 数值恰为 0；编码器必须做接触平衡采样，否则最容易学会“永远忽略触觉”。

![[90 Attachments/HyVLA-Tactile-Integration/test-single-taxel-contact-maps.png]]

> [!info] 空间图怎么读
> 颜色是各 taxel 的局部法向 `Fz`，箭头是 `Fx/Fy`。第一接触段主要集中在左 P9、右 P2；第二段主要集中在左 P5/P8、右 P4。即使左右总力相近，空间分布也不同。这正是“只给 6D 左右合力”和“给完整 54D taxel”应该做消融的理由。

### 1.3 当前最关键的数据问题：action 是下一帧的绝对 state

逐帧检查得到：

```text
action[t] == observation.state[t+1]
平均绝对误差 = 0
```

这说明当前动作标签是“下一帧绝对 EEF 状态”，不是 HY‑VLA 使用的、以当前时刻为锚点的相对 EEF action chunk。不能直接把这 10D 喂给 HY‑VLA 训练，否则模型会混淆动作坐标系。

正确转换应是：

```text
给定 chunk 起点 (p0, R0)
第 i 步平移：Δp_i = R0ᵀ (p_i - p0)       # 若 HY‑VLA 代码使用起点局部 EEF 系
第 i 步旋转：ΔR_i = R0ᵀ R_i
再把 ΔR_i 转成与 HY‑VLA 实现一致的 6D rotation
gripper 单独归一化
```

> [!warning] 不能做的事
> 不要对 6D rotation 向量逐元素相减。6D rotation 是旋转矩阵的连续表示，正确相对旋转必须先还原 `R`，做矩阵乘法，再编码回 6D。平移究竟是在 world、base 还是起点 EEF 坐标中，也必须以实际 HY‑VLA 数据预处理代码为准。

### 1.4 当前数据还缺什么

| 缺口 | 为什么重要 | 采集时应补上 |
|---|---|---|
| task 文本 | VLA 需要语言语义；`23` 不可解释 | 明确任务、阶段、物体 ID |
| finger-to-EEF 外参 | 不能把左右局部力变换到共同坐标 | 每次装配的旋转/位移与 sensor serial |
| 空载基线/漂移 | 0 N 附近偏置会伪造接触 | 每 episode 前后 1–2 s 无接触段 |
| 传感器饱和/有效位 | 全零可能是无载荷，也可能是局部掉线 | 每指 validity、饱和 flag、错误码 |
| 物体物性 | 无法按重量/摩擦/刚度分层评测 | object、mass、surface、stiffness、damage limit |
| 失败原因 | 只看 success 难解释触觉作用 | drop、slip、crush、jam、overforce、timeout |
| 控制状态 | 触觉变化可能来自底层控制器 | servo mode、gain、command/actual pose、latency |

> [!tip] 对当前样本的定位
> 它非常适合做：解析器测试、时间同步测试、contact label 原型、taxel encoder 的 shape test、HY‑VLA batch 拼装测试。它不适合做：train/test split、任务成功率、跨物体泛化、触觉贡献结论。

---

## 2. 最新且可复用的开源路线：什么真的能拿来改

截至 2026-09-11，代码开放程度必须单独判断。`Awesome-Touch` 很适合发现论文，但列表本身不是实验结论；论文数字仍以原文为准。[^awesome]

| 工作 | 它真正解决什么 | 触觉/力输入 | 开源可用性 | 对本项目的价值 |
|---|---|---|---|---|
| RDP (RSS 2025) | 慢视觉规划 + 快触觉修正 | 光学触觉或腕部 F/T | 主仓库仍有训练/数据/checkpoint TODO；论文方法完整 | slow–fast 设计原点；用 ImplicitRDP 代码落地更现实 |
| FoAR (2025) | 未来接触门控 + 100 Hz 力历史 + 手工反应修正 | 6D 腕部 F/T | 训练、数据结构、评测代码开放 | 门控、历史 encoder、时间对齐代码最直接 |
| ImplicitRDP (RA-L 2026) | 单网络结构化 slow–fast + 虚拟目标正则 | 6D 腕部 F/T | 训练/推理/数据/checkpoint 均开放 | 最强的因果 mask、快循环与物理辅助任务模板 |
| AT‑VLA (CVPR 2026 Oral) | 触觉门控 + VLA 动作专家 + 快慢流 | 6D resultant force | 只开放推理导向代码；无完整训练/真机 harness | 与 HY‑VLA 结构最接近，但需自行实现训练 |
| FE‑VLA (2026 repo) | LeRobot/π0 中的轻量 force prefix | 30 点标量力窗，200 Hz sensor | 训练、记录、评测、模型/数据开放 | 最贴近今天能改的 LeRobot 工程模板；论文状态与仓库数字需谨慎 |
| UniTac‑NV (IROS 2025) | 非视觉触觉跨传感器共同 latent | 3×3×3 PapillArray、4×6×3 uSkin | 数据、预处理、对齐 notebook 开放；不是完整 policy | 与你们 3×3×3 taxel 形态最接近，适合 encoder 预训练/硬件迁移 |
| ForceVLA2 (CVPR 2026) | VLA 直接输出混合力—位控制 | 300 Hz 腕部 6D F/T | 论文/项目页；本次未发现官方训练仓库 | 远期 action-space 方向，不是当前数据的 MVP |

### 2.1 RDP：为什么“动作块”会让触觉失去反应速度

![[90 Attachments/VLA-Tactile-Survey-2026/rdp-fig6-pipeline.png]]

> [!info] 原论文 Fig. 6 阅读路径
> 左侧先用动作 encoder 把完整 action chunk 压成 latent；慢策略以约 1–2 Hz 结合视觉预测 latent。右侧快 decoder 在执行过程中不断读取 20–30 Hz 触觉，用最新触觉把同一个 latent 解码成动作。最关键的非对称约束是：encoder 只看 action，decoder 看 latent + tactile，这迫使 decoder 不能绕开触觉。[^rdp-paper]

RDP 的矛盾是：普通 Diffusion Policy 每次先预测一整段动作，再开环执行这段；触觉即使在中途发生变化，也要等下一次大模型规划。RDP 将“想做什么”放在慢 latent，将“这一刻如何修正”放在快 decoder。

它在剥皮、擦拭和双臂搬运上测试，论文中 RDP(force) 的剥皮综合分数为 0.95，视觉 Diffusion Policy 为 0.48；只保留 normal force 时降到 0.48，说明切向信息和非对称 tokenizer 都重要（原文 Table II、VI）。但其主仓库截至本次核对仍将训练代码、数据与 checkpoints 标为 TODO，因此不要把“仓库存在”当作“可直接复现”。[^rdp-repo]

对你们的启发：**先证明同步触觉有信息，再做快 decoder**。否则 slow–fast 会把数据、模型和部署问题绑在一起。

### 2.2 FoAR：门控的理由不是漂亮，而是避免非接触噪声破坏策略

![[90 Attachments/HyVLA-Tactile-Integration/foar-original-fig2.png]]

> [!info] 原论文 Fig. 2 阅读路径
> 上路把 RGB-D 变成点云 scene feature；下路把过去约 2 s、100 Hz 的 6D F/T 序列经 MLP + Transformer 变成 force feature。未来接触预测器输出 `φ`：非接触时用学习到的 neutral embedding，接触将要发生时才放大 force feature。融合结果进入 diffusion action head。部署还有一个基于阈值和 6 mm 位移的手工 reactive refinement。[^foar-paper]

FoAR 的价值是把两个经常混淆的东西分开：

1. **学习式门控**：现在是否应该信任力信息；
2. **部署时规则**：力不足时沿预测方向追加一小段位移。

在擦白板消融中，完整 100 Hz + contact predictor + reactive control 得分 0.875；10 Hz 为 0.800，2 Hz 为 0.625；去掉 predictor 或 reactive refinement 的配置约 0.65（原文 Table III）。这支持“历史频率、门控与闭环修正都重要”，但它仍是腕部 F/T + 点云 Diffusion Policy，不是 VLA，也不是你们的分布式指尖阵列。官方仓库确实包含 `policy/tokenizer.py` 的 ForceEncoder、`dataset/realworld.py` 的高频对齐、`train.py` 与 `eval.py`，可直接借鉴数据组织。[^foar-repo]

### 2.3 ImplicitRDP：把快慢流塞进一个有因果约束的网络

![[90 Attachments/HyVLA-Tactile-Integration/implicitrdp-original-fig2.png]]

> [!info] 原论文 Fig. 2 阅读路径
> 黄色是慢条件（图像、proprioception、diffusion step），绿色是与动作时间轴对齐的 force 序列，紫色是 noisy action。GRU 保持力编码因果；交叉注意力的三角 mask 保证第 i 个 action 只能看当下及过去的 force，不能在训练时偷看未来。[^implicit-paper]

它解决 RDP 两阶段训练与信息隔离的问题：慢图像 token、初始 diffusion noise 在一个 chunk 内缓存；每个动作时刻追加最新力 token，用确定性 DDIM 重算增长中的序列，并执行最后一个动作。虚拟目标正则 VRR 用近似关系

```text
x_virtual = x_real + K(F)^(-1) · F_external
```

把力映射到动作同一空间：小力时刚度高，虚拟目标≈真实动作，抑制噪声；大力时刚度低，接触修正被放大。这比“预测下一时刻力”给出的控制方向更直接。

三项任务、每项 40 demos、20 次测试中，DP / RDP / ImplicitRDP 分别得到：

| 任务 | DP | RDP | ImplicitRDP |
|---|---:|---:|---:|
| 翻薄盒 | 0/20 | 16/20 | 18/20 |
| 拨断路器 | 8/20 | 10/20 | 18/20 |
| 1 mm 裕量插孔 | 0/20 | 5/20 | 8/20 |

VRR 辅助任务在翻盒/开关上为 18/20、18/20；普通 force prediction 只有 8/20、10/20（原文 Table I、III）。这说明“辅助任务是否和动作因果相关”比“多预测一个模态”更重要。官方仓库开放训练脚本、真实机器人管线、数据与 checkpoint；`transformer_for_diffusion.py` 的 causal memory mask 和 `post_process_data.py` 的 stiffness/virtual-target 计算尤其值得参考。[^implicit-repo]

### 2.4 AT‑VLA：最接近我们要改的结构，也给出一个重要反例

![[90 Attachments/HyVLA-Tactile-Integration/atvla-original-fig2.png]]

> [!info] 原论文 Fig. 2 阅读路径
> 慢流用图像与语言产生 key/value；状态或触觉 token 作为动作专家 Adaptive Attention 的 query。触觉 gate 决定是否使用触觉；激活后，快触觉流在慢视觉上下文之间多次运行。损失包括 action、gate 和未来触觉生成。[^atvla-paper]

AT‑VLA 最重要的不是“多一条输入”，而是两个证据：

- 四个接触任务的平均成功率：vanilla 0.22，naive 直接加入触觉反而降到 0.13；加入 gate 为 0.39，再加未来触觉生成为 0.43，再加 dual-stream 为 0.50（原文 Table 3）。
- 30–50 demos/task、15 trials；快慢推理比为 3:1，论文报告快触觉闭环约 0.04 s。

这直接反驳“只要 concat 触觉就会变好”。非接触区中，触觉是稀疏、偏置和硬件相关的新模态，会扰动预训练 VLA 的视觉定位。遗憾是官方 README 明确写明：当前 release 只面向 inference，不开放完整 training scripts 与 real-robot evaluation harness。因此它是我们的**结构证据**，不是可直接跑通的训练基线。[^atvla-repo]

### 2.5 UniTac‑NV：为什么它和你们的 3×3×3 阵列特别相关

![[90 Attachments/HyVLA-Tactile-Integration/unitac-original-fig1.png]]

> [!info] 原论文 Fig. 1 阅读路径
> 每种传感器有自己的 encoder，但共享 decoder。配对的相同接触被分别编码；decoder 同时做自重建和交叉重建，因此共同 latent 被迫保留两种硬件都能表达的接触属性。下游接触几何 MLP 可以在一个传感器 latent 上训练，再直接用于另一个。[^unitac-paper]

PapillArray 输入正好是 `3×3×3=27D`，与你们每指 9 个三轴 taxel 同形；论文把它与 `4×6×3=72D` uSkin 在 100 Hz 下对齐，encoder 投影到 16D latent。对未见不规则物体，PapillArray→uSkin 的 SSIM 降到 0.581，接触几何跨传感器误差约 0.637/0.666 mm，相比同传感器 0.353/0.397 mm 更差（原文 Table I 与 Sec. III-C）。这说明统一表示能迁移共同物理量，但不能凭空补出目标硬件的空间分辨率。

对本项目最实际的使用方式不是立刻做跨传感器，而是：用**每指独立 encoder + 共享下游接口**，并把 sensor ID、taxel 坐标和 validity 保存好。未来换指尖时，只训练新的 sensor adapter。官方仓库主要开放数据、CAD、预处理和对齐 notebook，并非完整策略训练仓库。[^unitac-repo]

### 2.6 FE‑VLA 与 ForceVLA2：一个是工程捷径，一个是远期上限

FE‑VLA 仓库把 200 Hz 单点力传感器的最近 30 个样本异步放入 deque，用 1D CNN 编成默认 4 个 prefix tokens，并在 LeRobot/π0 训练脚本中增加 `observation.force` 与 `observation.force_max`。它还有一个预测 action window 峰值力的 head 和部署安全 clamp。这个实现与当前 LeRobot 数据最接近，尤其值得参考 `force_sensor.py`、`modeling_fevla_pi0.py` 和修改后的 record/train/eval 脚本。仓库声称对 5 类易损物体相对 π0/π0.5 降低约 50% 峰值力；在没有正式同行评审论文可核验时，应把它标为**工程报告证据**，不能和 CVPR/RSS 数字同等级引用。[^fevla]

ForceVLA2 则把力同时作为 VLM 的阶段 prompt、动作专家的实时条件和输出 action 的目标，学习混合 force-position control。它有 1,000 条轨迹、5 个任务，平均成功率 66%，论文报告相对 π0、π0.5 分别高 48、35 个百分点（原文 Table 1/2）。这条路线更强，但需要：腕部 6D F/T、目标力/位置标签、控制掩码和能执行力控制的底层控制器。你们当前只有指尖局部三轴力与位置动作，所以应先把 ForceVLA2 当作**第二篇论文的方向**，不是第一版实现。[^forcevla2]

---

## 3. HY‑VLA 应该怎么改：推荐的最小架构

先回到 HY‑VLA 原始结构：

![[90 Attachments/HyVLA/hyvla-original-architecture.png]]

HY‑VLA 用 MoT 组织视觉/语言 `P`、当前 state `S` 和 noisy action chunk `A`，action expert 通过 conditional flow matching 生成连续动作。它已经有 compact memory、相对 EEF action 与异步 buffer/stitching，因此不需要重新发明一套完整 VLA。[^hyvla]

### 3.1 建议增加第四类 block：Tactile `T`

![[90 Attachments/HyVLA-Tactile-Integration/hyvla-tactile-integration.png]]

建议的最小接口：

```text
P = image history + language tokens              # 原 HY-VLA，慢、可缓存
S = current EEF pose + gripper state             # 原 HY-VLA
T = 2–4 tactile tokens                           # 新增，来自最近一段触觉
A = noisy relative-EEF action chunk              # 原 HY-VLA action expert

h_A <- h_A + g_contact * gamma * CrossAttn(Q=h_A, KV=T)
gamma initialized to 0
```

为什么不是直接把 60D concat 到 state：

1. `state` 是密集、稳定、每帧有意义的运动学量；触觉是 75% 非接触、硬件相关、带漂移的稀疏量；
2. cross-attention 让 action token 主动查询触觉，信息路径清楚；
3. `gamma=0` 初始化时网络严格退化为原 HY‑VLA，能逐步学会使用新模态；
4. tactile encoder 和 adapter 可单独冻结/训练，便于消融；
5. 未来做高频时，只刷新 `T/A`，不必重跑全部视觉语言主干。

这是结合 AT‑VLA 的 gate、FoAR 的 neutral path 与现有 Obsidian 中 T‑Rex/N0 系列经验得到的**设计推断**，不是某篇论文已经在 HY‑VLA 上验证过的结论。

### 3.2 触觉 encoder：既保留空间，也保留时间

推荐输入张量：

```text
touch_raw:  [B, W, 2 fingers, 9 taxels, 3 axes]
present:    [B, W, 2]
age_s:      [B, W, 2]
taxel_xyz:  [2, 9, 3]              # 安装几何；当前只有 sensor-local 点坐标
finger_id:  [2]
```

第一版 `W` 可取最近 16 个 30 Hz 对齐帧（约 0.53 s）；采集系统稳定后，再从原始约 80 Hz ring buffer 查询最近 0.5 s。窗口长度是超参数，不是越长越好：滑移需要短时差分，阶段/持续力需要较长历史。

建议 encoder 分三层：

```text
每 taxel 三轴力 + ΔF + mask
  -> 共享 MLP（所有 taxel 共用）
  -> 每时刻在 3×3 几何上做小型 spatial attention / graph / 2D conv
  -> 每指用 causal 1D CNN 或 GRU 聚合时间
  -> left_global, right_global, dynamic_event（2–4 tokens）
```

不要只输入模型学习后的 token。额外保留这些可解释 summary，既可做辅助监督，也可在部署日志中检查：

- 左/右合力 `Σ_j F_j`；
- 法向总量、切向模长；
- center of pressure 与 contact spread；
- `dF/dt`、短窗方差和高频能量；
- 激活 taxel 数量；
- present、age 与饱和位。

### 3.3 标定与归一化：先处理物理，再谈大模型

每个传感器实例、每个轴分别处理：

1. 用 episode 开头无接触窗估计 baseline，`F ← F_raw - median(F_free)`；
2. 用训练集 robust scale（如 MAD 或 1–99% 分位）归一化，禁止使用测试集统计量；
3. 对明显尖峰做物理范围裁剪，同时保留 `clipped` flag；
4. 训练时加入小幅 bias drift、时间抖动、dropout，模拟真机；
5. 缺失必须用 mask 表示，不能把缺失悄悄当成 0 N。

当前 `TCH0 v1` 没有 per-finger validity，所以“整指全零”在协议层可能是卸载，也可能是局部缺失。短期用 record-level mask，长期应修改协议。

### 3.4 接触门控：不要让 75% 的无接触帧淹没学习

可先用可审计的软标签：

```text
contact_on  if max_f ||Σtaxel F|| > τ_on for k frames
contact_off if max_f ||Σtaxel F|| < τ_off for m frames
τ_off < τ_on        # hysteresis，防止门控抖动
```

阈值不能永远写死 1 N；应从空载噪声和最小有意义接触分布中标定。随后训练小 gate 网络，输入触觉统计 + state + 可选视觉上下文，目标是当前/临近接触。训练采样需提高接触 onset、slip 和 release 附近帧的权重。

建议把 gate 当**连续置信度**而不是硬开关；部署安全限制则必须是模型之外的硬规则。

### 3.5 辅助任务的选择

第一版总损失：

```text
L = L_flow(action)
  + λ_contact · BCE(gate, contact_label)
  + λ_phys · Huber(physical_target)
```

`physical_target` 优先级：

1. action window 内的峰值法向力/接触释放；
2. 下一小窗触觉残差，而不是绝对原始值；
3. 若有 robot-frame 外参与控制刚度，再做 ImplicitRDP 式 virtual target；
4. 未来才考虑 N0‑VTLA/N0‑TWAM 式长程 future tactile latent。

只预测下一帧原始触觉容易学到“持续不变”，不一定迫使 token 保留会改变动作的信息。ImplicitRDP 的 VRR 消融正好提醒了这一点。

### 3.6 为什么先按 30 Hz 训练，而不是强行对齐 HY‑VLA 的 50 Hz

HY‑VLA 真机阶段使用约 50 Hz action horizon，但当前位置/动作标签只有 30 Hz。把它插值到 50 Hz 会增加帧数，却不会制造新的独立行为信息，还可能把接触跃迁抹平。

MVP 建议：

- 数据 loader 配成 30 Hz，1 s chunk 即 `H=30`；
- 触觉先用对应的 30 Hz 窗；
- 原始 80 Hz 仅用于同一时刻之前的 history，不生成虚假 80 Hz action label；
- 当机器人控制栈能记录 50 Hz 实际/命令位姿时，再训练 50 Hz 版本。

---

## 4. 分阶段训练路线：每一步只回答一个问题

### Phase 0：数据与标签 QA

通过标准：

- 每 episode 生成视觉—gripper—force 时间线；
- `total ≈ Σ taxels`，异常率有报告；
- touch age、dropout、饱和、baseline drift 可视化；
- action 坐标转换有单元测试：恒定位姿→零 delta、已知旋转→正确 delta；
- task/object/material/success/failure 元数据齐全。

### Phase 1：触觉 probe，不碰 HY‑VLA

用轻量 encoder 做四个 probe：接触检测、左右接触位置、峰值力预测、已知滑移事件/方向。目的不是刷分，而是确认 60D 中哪些结构可学、哪些只是硬件偏置。

比较：

```text
total-6D vs full-taxel-54D
current-frame vs history
30Hz-aligned vs raw-80Hz history
with vs without baseline/mask/age
```

如果 full taxel 连 probe 都不优于 total，就不要急着把 60D 全塞进 4B VLA；先检查传感器安装、标签和任务是否真的含空间差异。

### Phase 2：HY‑VLA 同步触觉 MVP

冻结或低学习率训练视觉语言主干，训练：

- tactile encoder；
- contact gate；
- zero-init tactile adapter；
- action expert 的小部分参数或 LoRA。

先使用相同动作频率与相同 chunk 执行方式，回答唯一问题：**真实同步触觉是否比无触觉与错位触觉更好？**

### Phase 3：快触觉闭环

只有当 Phase 2 证明触觉有信息、且日志显示失败发生在视觉/VLA 刷新间隙，才做：

- 缓存 `P/S` slow tokens；
- ring buffer 取最新 raw touch；
- 仅刷新 `T` 与 action refinement；
- 参考 ImplicitRDP 的 causal mask 和 deterministic-noise consistency；
- 记录 sensor→encoder→policy→command 的端到端延迟及 deadline miss。

### Phase 4：跨物体、跨传感器与未来触觉

- 先做 sensor-instance-held-out；
- 再做 UniTac‑NV 式 sensor adapter/shared latent；
- 再做 N0‑VTLA 式 compact future tactile latent；
- 只有当前/未来两路分别有消融增益，才做 N0‑TWAM 式双路径。

### Phase 5：混合力—位控制

这一步需要重采数据：目标力、实际力、position/force mask、底层控制模式与刚度。否则模型只能“看力、改位置”，不能被称为 ForceVLA2 式 force-control policy。

---

## 5. 用什么任务证明触觉真的有用？

![[90 Attachments/HyVLA-Tactile-Integration/tactile-benchmark-ladder.png]]

### 5.1 第一优先：隐藏负载/摩擦的滑移恢复

**装置**：外观完全相同的盒子或圆柱，内部随机放不同砝码；表面可更换高/低摩擦套。抓起后由装置施加可复现侧向扰动，或在视觉不变的情况下追加负载。

**为什么强**：视觉和当前 gripper width 相同，策略只能在接触后从切向力、空间力迁移和微滑移中发现变化。

**指标**：

- 持续 5–10 s 不掉落率；
- 最小稳定夹持力；
- 峰值法向力与积分力；
- 扰动到增力/重抓的恢复延迟；
- slip 次数、恢复率、物体位移。

**核心对照**：no touch、同步 touch、touch 延迟 100/300/500 ms、时间打乱、total-only、full-taxel。

### 5.2 第二优先：易损/柔顺物体的窄力窗口

不要先用气球，先使用可重复测量的对象：

- 不同刚度、同外观外壳的标准泡棉块；
- 带形变量标尺的薄壁杯；
- 3D 打印弹性蜂窝/弹簧测试件；
- 内置脆断片或压力指示膜的统一样件。

成功不是“抓起来”一个二元条件，而是：

```text
抓持力 >= 防滑下界
抓持力 <= 损伤上界
```

触觉的价值正是把动作维持在这个窄窗口。报告抓取成功、损坏率、最大形变、峰值力和稳态力波动。

> [!question] 气球到底能不能用？
> 可以作为最终演示或安全边界测试，但不适合首个论文主 benchmark：不同气球的橡胶厚度、预充气量和爆破阈值波动大；视觉也能看到明显形变；一次爆破后样本不可复用。若一定要做，需按批次记录直径/压力，先用材料测试机标定爆破分布，并把“峰值力/形变曲线”作为主要指标，而不只是爆/没爆。

### 5.3 第三优先：遮挡下的精密插入

**装置**：1–2 mm 间隙的方/圆 peg；末端接触区域用挡板遮住；孔相对视觉估计有随机小偏置。

**为什么强**：卡住时，某些 taxel 的切向/法向力会上升，空间分布指示纠偏方向。总力只能告诉你“卡了”，full taxel 可能告诉你“往哪边改”。

**指标**：插入成功、卡死率、重试次数、时间、峰值力、孔/peg 磨损。必须分别比较 total-only 与 full-taxel。

### 5.4 第四优先：曲面擦拭与恒力跟随

让曲面高度/法向随机，或在执行中小幅移动工件。目标是维持接触但不过载。

指标：目标力 RMSE、接触丢失时长、覆盖率、过载率、完成时间。这个任务与 FoAR/ForceVLA2 文献对应最直接，但要注意：如果底层位置控制很硬且模型动作频率太低，触觉再好也无法产生平滑恒力。

### 5.5 第五优先：按钮/卡扣/阈值事件

要求达到阈值后立即停止，或按固定次数、跨多个阶段。它适合验证 history、contact gate 和峰值/事件辅助任务，但容易被简单手写阈值解决。论文中应加入随机位置/阈值/顺序，避免只证明一个 safety rule。

### 5.6 鼠标抓取的正确位置

当前样本中的桌面物体交互适合作为“相机、动作、触觉一起跑通”的冒烟测试。它通常不适合作为主要触觉论据，因为视觉、几何夹爪宽度和位置轨迹就可能解决大部分任务。除非引入同外观不同重量/摩擦、遮挡与外扰，否则 vision-only 很可能没有原则性困难。

---

## 6. 实验矩阵：怎样把触觉贡献钉死

### 6.1 最小主表

| ID | 视觉/状态 | 触觉 | 门控 | 频率 | 回答的问题 |
|---|---|---|---|---|---|
| V | ✓ | — | — | 30 Hz | HY‑VLA 基线 |
| V+T | ✓ | 同步 full taxel | ✓ | 30 Hz | 触觉总体是否有益 |
| V+T-current | ✓ | 当前帧 | ✓ | 30 Hz | 历史是否必要 |
| V+T-total | ✓ | 左右合力 6D | ✓ | 30 Hz | 空间 taxel 是否必要 |
| V+T-no-gate | ✓ | 同步 full taxel | — | 30 Hz | gate 是否避免负迁移 |
| V+T-shift | ✓ | 时间错位 | ✓ | 30 Hz | 模型是否真的用因果触觉 |
| V+T-fast | ✓ | raw history | ✓ | ~80 Hz T / 30 Hz action | 高频是否独立贡献 |

`V+T-shift` 至少做 `+100/+300/+500 ms` 或 episode 内循环移位。若错位触觉也同样提升，模型可能只把它当作任务阶段标签，或增益来自额外参数而非实时反馈。

### 6.2 训练公平性

- 所有主对照用相同 episode、相同 optimizer steps、相同 seed 列表；
- V 基线可放同参数量的 dummy adapter，排除容量差异；
- train/val/test 按 episode、物体实例和物理属性分组，禁止随机拆帧；
- contact oversampling 只用于 train，评测保持真实时间分布；
- 模型选择不能看 test；
- 至少报告 3 个训练 seed 与每 seed 的真机 trial 分布。

### 6.3 真机试验随机化

每次 trial 随机化：

```text
policy condition
object instance
hidden weight / friction / stiffness
initial pose perturbation
disturbance magnitude and time
```

操作者在 trial 结束前不知道隐藏属性；能自动触发的扰动不要手动触发。相同物体实例尽量做 paired comparison，但要随机化顺序，避免电机温度、传感器漂移和操作者熟练度偏差。

### 6.4 不只报成功率

推荐统一日志：

| 层级 | 指标 |
|---|---|
| 任务 | success、damage、drop、jam、timeout |
| 接触 | onset latency、contact loss duration、slip count/recovery |
| 力 | peak、impulse、steady-state variance、target RMSE |
| 动作 | jerk、replan/retry、gripper oscillation |
| 系统 | sensor age、policy latency、deadline miss、buffer staleness |

二元成功率用 Wilson 95% CI；连续指标用按 trial bootstrap CI。20 trials/condition 与文献相当但区间仍较宽，若主结论接近，应扩到 30–50 而不是只看均值。预先写清主指标，避免事后挑最好看的指标。

### 6.5 论文最关键的反事实图

建议最终论文至少画三种曲线：

1. 同一扰动下，`V` 与 `V+T` 的切向力、夹爪动作、物体位移时间线；
2. 触觉延迟从 0→500 ms 时 success/恢复延迟如何退化；
3. full taxel 与 total-only 在不同孔偏置方向上的纠偏向量与成功率。

这些图比“触觉 attention 很亮”更接近因果证据。

---

## 7. 代码落地清单

### 7.1 数据 adapter

目标 batch：

```python
batch = {
    "images": ...,                       # HY-VLA 原输入
    "language": ...,
    "state": ...,                        # [B, state_dim]
    "actions_rel": ...,                  # [B, H, action_dim]
    "touch": ...,                        # [B, W, 2, 9, 3]
    "touch_present": ...,                # [B, W, 2]
    "touch_age_s": ...,
    "contact_target": ...,
    "future_peak_force": ...,
}
```

需要新增测试：

- schema names 到 `[finger,taxel,axis]` 的映射；
- `total` 与 taxel sum；
- episode 边界窗口 padding；
- 所有 history 时间戳 `<= action anchor`，防止未来泄漏；
- absolute state→relative chunk 的 SE(3) 转换；
- missing 与 true zero 分开。

### 7.2 模型 adapter

建议接口：

```python
tactile_tokens, tactile_mask, gate = tactile_encoder(
    touch, present=touch_present, age=touch_age_s
)
action_hidden = action_hidden + gate * tactile_scale * tactile_cross_attn(
    query=action_hidden,
    key=tactile_tokens,
    value=tactile_tokens,
    key_padding_mask=~tactile_mask,
)
```

`tactile_scale` 参数初始化为 0。第一轮实验不要改 P/S 的注意力可见性；让 `A` 看 `T` 即可。若将来做 action-time-aligned fast force，要采用类似 ImplicitRDP 的三角 mask，确保第 i 个 action 看不到 `i+1` 之后的触觉。

### 7.3 可复用代码位置

| 需求 | 首选参考 |
|---|---|
| 高频 F/T 窗对齐与 padding | FoAR `dataset/realworld.py` |
| history MLP + Transformer | FoAR `policy/tokenizer.py` |
| future contact label / gate | FoAR dataset + policy |
| causal force/action mask | ImplicitRDP `transformer_for_diffusion.py` |
| virtual target/stiffness | ImplicitRDP `post_process_data.py` |
| LeRobot force key、ring buffer、CNN token | FE‑VLA `force_sensor.py`, `modeling_fevla_pi0.py` |
| VLA 自适应触觉 query | AT‑VLA inference model；训练需自建 |
| 3×3×3 非视觉触觉表示 | UniTac‑NV data/alignment notebook |

### 7.4 部署 safety layer

无论模型多强，都应在策略外保留：

- 绝对最大力/夹爪命令 clamp；
- 力上升率阈值；
- sensor stale/dropout 时降级到安全动作；
- action workspace、速度与加速度限制；
- 硬件急停。

安全层不能被计作“触觉策略学会了”。实验需分别记录 safety intervention 次数；如果触觉模型成功率高但全靠 clamp，结论应是 safety rule 有效，而不是策略理解了触觉。

---

## 8. 研究假设与可证伪条件

### H1：当前 60D 指尖触觉包含超出 gripper state 的接触信息

支持：full-taxel probe 在 held-out objects 上优于 state/total-only，尤其是接触位置与滑移方向。

证伪：触觉 probe 只在随机帧 split 上好，按 object/episode split 后失效；或打乱 taxel 仍不掉性能。

### H2：门控的 tactile adapter 能提高 HY‑VLA 接触任务而不损伤非接触任务

支持：V+T 在接触任务提升，同时 pick/place 等非接触阶段不低于 V；no-gate 出现 AT‑VLA 式负迁移。

证伪：dummy adapter 与真实触觉同样提升，或同步/错位触觉没有差异。

### H3：空间 taxel 比左右总力更适合方向性纠偏

支持：在遮挡插入/偏心抓取中 full-taxel 优于 total-only，且纠偏方向随接触分布改变。

证伪：差异只来自参数量；随机旋转/置换 taxel 不影响策略。

### H4：高频触觉的价值来自低延迟，而不是更多样本点

支持：控制视觉与 action rate 后，raw ~80 Hz 快分支降低恢复延迟；人为增加 100–500 ms 触觉延迟会单调退化。

证伪：将 80 Hz 降采样到 30 Hz 性能不变，或失败发生在策略无法改变的底层控制限制上。

---

## 9. 推荐的实际排期

### 第 1–2 周：把“数据能信”做扎实

- 录制至少 3 类接触原语：正压、切向滑动、偏心夹持；
- 每类覆盖多个力级、速度、物体与两侧手指；
- 加入空载 baseline、sensor serial、外参与物体物性；
- 自动生成本文这种时间线与 taxel map；
- 完成 SE(3) action chunk 转换测试。

### 第 3–4 周：轻量 probe + 第一批基线

- total/current、total/history、full/current、full/history 四组；
- 选择 2–4 tactile tokens；
- 确认接触不平衡采样、漂移增强和 mask 有效；
- 建立 mouse grasp 冒烟测试，但不作为论文主结果。

### 第 5–8 周：HY‑VLA 同步触觉与主消融

- V、V+T、V+T-no-gate、V+T-shift、V+T-total；
- 先选隐藏负载滑移 + 标准易损物抓取；
- 每条件先做小规模 pilot，确认 effect direction 后预注册正式 trial 数；
- 同时记录任务与过程指标。

### 第 9 周以后：根据失败类型升级

```text
如果“看不出接触”       -> encoder / 标定 / task 设计
如果“看出了但动作不变” -> adapter / auxiliary target / action authority
如果“动作变了但太慢”   -> fast tactile branch / cache / async
如果“换传感器就坏”     -> sensor adapter / UniTac-NV / MTTS
如果“位置动作做不到恒力” -> 重采 hybrid force-position labels
```

---

## 10. 最终建议：把第一篇研究收敛成一句话

> 在视觉不可辨识的隐藏重量、摩擦或遮挡接触条件下，一个带接触门控、保留 taxel 空间—时间结构的 HY‑VLA tactile adapter，能否在不损伤非接触能力的情况下，降低滑移/峰值力并提高任务成功率？

这句话同时限定了：

- **信息来源**：指尖 taxel，而不是泛称“多模态”；
- **模型贡献**：gate + zero-init action-expert adapter；
- **任务因果性**：视觉不可辨识；
- **成功标准**：不仅成功率，还包括力和恢复；
- **负面约束**：不能损伤原 HY‑VLA 非接触能力。

如果这一版成立，再扩展到 raw-80Hz fast branch、未来触觉预测、跨传感器和混合力—位控制，研究链条会很自然；如果不成立，也能通过 probe、错位触觉和 total/full 消融知道问题在传感器、数据、模型还是任务，而不是得到一个无法解释的“大系统没跑起来”。

---

## Sources

[^hyvla]: [Hy-Embodied-0.5-VLA, arXiv:2606.14409](https://arxiv.org/abs/2606.14409). 本文关于 HY‑VLA 的 MoT、相对 EEF action chunk、flow matching 与部署频率来自原论文；本库已有全文核对笔记 [[Hy-Embodied-0.5-VLA]]。
[^awesome]: [Awesome-Touch: A curated list of touch sensing and touch perception](https://github.com/linchangyi1/Awesome-Touch#non-vision-based). 仅作为工作发现索引，不作为实验数字来源。
[^rdp-paper]: [Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation, arXiv:2503.02881](https://arxiv.org/abs/2503.02881), RSS 2025. 数字见原文 Table II、VI，结构见 Fig. 6。
[^rdp-repo]: [Reactive Diffusion Policy official GitHub](https://github.com/xiaoxiaoxh/reactive_diffusion_policy). 本次核对 README 的 release/TODO 状态。
[^foar-paper]: [FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation, arXiv:2411.15753](https://arxiv.org/abs/2411.15753). 结构见 Fig. 2，频率消融见 Table III。
[^foar-repo]: [FoAR official GitHub](https://github.com/Alan-Heoooh/FoAR). 本次核对了 ForceEncoder、100 Hz 对齐、训练与部署代码。
[^implicit-paper]: [ImplicitRDP: An End-to-End Visual-Force Diffusion Policy with Structural Slow-Fast Learning, arXiv:2512.10946](https://arxiv.org/abs/2512.10946), accepted by IEEE RA-L in June 2026. 结构见 Fig. 2，结果见 Table I–IV。
[^implicit-repo]: [ImplicitRDP official GitHub](https://github.com/Chen-Wendi/ImplicitRDP). README 提供 DP/RDP/DPT/ImplicitRDP 训练脚本、真实机器人推理、数据与 checkpoint 链接。
[^atvla-paper]: [AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in Vision-Language-Action Models, arXiv:2605.07308](https://arxiv.org/abs/2605.07308), CVPR 2026 Oral. 结构见 Fig. 2，消融见 Table 3。
[^atvla-repo]: [AT-VLA official GitHub](https://github.com/clorislili/AT-VLA). README 明确说明 release 面向 inference，未开放完整训练与真机评测 harness。
[^unitac-paper]: [UniTac-NV: A Unified Tactile Representation For Non-Vision-Based Tactile Sensors, arXiv:2506.19699](https://arxiv.org/abs/2506.19699), IROS 2025. 结构见 Fig. 1，跨传感器结果见 Table I 与 Sec. III-C。
[^unitac-repo]: [UniTac-NV official dataset GitHub](https://github.com/JiannnH/UniTac-NV). 开放数据说明、CAD、预处理与对齐 notebook。
[^fevla]: [FE-VLA official GitHub](https://github.com/fevla2026/fe-vla). 本文只把其数字视为仓库自述；工程接口来自实际代码与 README。
[^forcevla2]: [ForceVLA2: Unleashing Hybrid Force-Position Control with Force Awareness for Contact-Rich Manipulation, arXiv:2603.15169](https://arxiv.org/abs/2603.15169), CVPR 2026. 数据规模与结果见原文 Table 1/2；项目页见论文链接。
