---
type: paper
title: "OpenVLA: An Open-Source Vision-Language-Action Model"
aliases: [OpenVLA]
year: 2024
status: deeply-explained
verification: full-paper-and-appendix-checked
topics: [VLA, open-source, action-tokenization, Open-X-Embodiment, LoRA, quantization]
source: https://arxiv.org/abs/2406.09246
project: https://openvla.github.io
---

# OpenVLA：把 VLM 直接改造成可复现、可微调的机器人策略

上级地图：[[VLA 学习地图]] · 相关比较：[[VLA 方法比较]] · [[数据策略对比]]

核心概念：[[逐维动作分箱]] · [[双视觉编码器融合]] · [[非阻塞控制与推理延迟]] · [[LoRA]]

> [!info] 原始材料
> - [[90 Attachments/OpenVLA/openvla.pdf|论文 PDF]]
> - [[90 Attachments/OpenVLA/openvla-study-guide.html|OpenVLA 原版 HTML 学习指南]]
> - HTML 用于帮助组织问题；精确事实、图号和证据边界均重新按 PDF 正文与 Appendix 核对。

> [!summary] 一句话结论
> OpenVLA 的核心贡献不是发明新的动作专家，而是证明一个开放的 Prismatic-7B VLM 可以通过“连续动作逐维分箱 → 复用 Llama token → 只对动作 token 做 next-token prediction”，在 970k 条多机器人轨迹上端到端微调成有竞争力、可继续 LoRA 适配和量化部署的通用策略。

## 00 先建立因果链：它为什么必须出现？

```text
RT-2-X 已证明：Internet-pretrained VLM 可以迁移到机器人控制
→ 但模型、训练配方和数据混合 largely closed
→ 外部研究者无法系统研究 VLA 的数据、架构与微调

开放 VLM 会看图、会回答文字
→ 但机器人需要连续控制量
→ 必须把 action 接入原有 LLM 输出接口

多机器人数据的相机、动作和场景不同
→ 直接混训会导致 input/output space 不一致或大数据集垄断
→ 需要筛选、标准化和 mixture weighting

7B policy 能训练出来，但新实验室不可能每次用 64 张 A100 重训
→ 还要证明 LoRA、量化与远程推理可用
```

所以 OpenVLA 同时回答两个问题：

1. **科学问题**：通用 VLM 是否能被直接微调为多机器人 policy？
2. **生态问题**：普通实验室能否下载、适配、部署并继续研究它？

## 01 先看论文原始总览

![[90 Attachments/OpenVLA/openvla-fig1-overview.png]]

*论文 Figure 1 · PDF 第 1 页 · 原论文图片。*

### 看图顺序

1. 左侧：来自 Open X-Embodiment 的 970k robot episodes；
2. 中间：把 base VLM 用 robot actions 端到端微调成 OpenVLA；
3. 右侧：用户语言指令 + 当前图像 → 单步机器人动作；
4. 下方：同一模型跨机器人开箱控制，并提供数据、权重和代码；
5. 部署到新机器人时，不要求从头预训练，可走 parameter-efficient fine-tuning。

> [!question] “完全开源”是否包括所有祖先模型的训练数据和代码？
> 不包括。论文脚注明确说 SigLIP、DINOv2、Llama 2 的权重开放，但它们各自的训练数据或训练代码并不全部开放。OpenVLA 开放的是其机器人训练数据来源/处理、模型权重、VLA 训练与微调代码。因此它是当时高度开放的 VLA 系统，但不能理解成整个 pretrained stack 从原始互联网数据起完全可复现。

## 02 真正的方法 pipeline：VLM 的“回答”变成 action

![[90 Attachments/OpenVLA/openvla-fig2-architecture.png]]

*论文 Figure 2 · PDF 第 4 页 · 原论文图片。*

从图中依次读：

```text
第三人称 RGB image
→ 同时经过 DINOv2 与 SigLIP
→ 同一 patch 的两种 feature 沿 channel 拼接
→ 2-layer MLP projector 映射到 Llama embedding space

language instruction
→ Llama tokenizer
→ 与 visual tokens 一起进入 Llama 2 7B

Llama 自回归输出 action tokens
→ Action De-Tokenizer 反量化
→ [Δx, Δθ, ΔGrip]，共 7D
```

这就是 OpenVLA 的极简处：没有 [[Action Expert]]、没有 [[Flow Matching]]、没有新设计的 cross-attention。它把 robot control 改写成 VLM 熟悉的 token prediction。

## 03 双视觉编码器：一个负责“像什么”，一个补空间结构

![[90 Attachments/OpenVLA/openvla-teaching-vision-fusion.svg]]

*教学示意图（根据 Sec. 3.1 与 Appendix D.2 绘制，并非论文原图）。*

### 3.1 最小数据流

```python
class FusedVisionEncoder:
    def forward(self, image):
        # image: [B, 3, 224, 224]
        # 两个 encoder 看同一组图像 patch
        f_sig = siglip(image)       # [B, N, D_sig]，偏图文语义
        f_dino = dinov2(image)      # [B, N, D_dino]，偏局部视觉结构

        # 沿 feature/channel 轴拼接，不是把 token 数 N 翻倍
        fused = concat([f_sig, f_dino], dim=-1)
        # [B, N, D_sig + D_dino]

        visual_tokens = projector_2layer_mlp(fused)
        return visual_tokens        # [B, N, D_llama]
```

Prismatic-7B 的视觉 encoder 总计约 600M 参数，后接小型 2-layer MLP，再连接 Llama 2 7B。

### 3.2 为什么不只用 SigLIP？

SigLIP 的图文预训练适合回答“这是杯子还是盘子”；DINOv2 的自监督视觉特征被作者用于补充低层空间信息。允许的解释是：两类 feature 互补，使 LLM 同时获得语义与空间线索。

但不能把 DINOv2 夸成直接输出精确像素坐标或 3D pose：论文没有这样的监督或定位头。Appendix D.2 的同设置消融是：

```text
OpenVLA-Bridge（双 encoder）           45.6 ± 5.6%
OpenVLA-Bridge-SigLIP（去 DINOv2）    40.6 ± 5.5%
```

约 5 个百分点的差距支持“双 encoder 有帮助”，但远小于去掉 OpenX 多样数据造成的约 30 个百分点差距。

### 3.3 为什么最终只用 224×224？

作者在小规模选择实验中比较 224 与 384：真实机器人表现未见差异，而 384 训练约慢 3 倍，因为 patch 数增大导致序列和 attention 成本上升。

证据边界：这是论文所测任务和 backbone 下的结论，不表示高分辨率对穿线、精密插接等任务普遍无用。

## 04 动作 token 化：关键不是“256”，而是统一 LLM 接口

![[90 Attachments/OpenVLA/openvla-teaching-action-token.svg]]

*教学示意图（根据 Sec. 3.2 绘制，并非论文原图）。*

### 4.1 为什么需要分箱？

Llama 的输出是 vocabulary 上的 logits；机器人动作是连续向量。若不修改 Llama 输出头，最直接的桥接办法是先把每个连续维度变成分类标签。

对动作维度 $d$，取训练数据的 1% 与 99% 分位数 $q_d^{1\%},q_d^{99\%}$，把区间均匀切成 256 个 bins：

$$
z_d=\operatorname{clip}\left(
\left\lfloor 256\frac{a_d-q_d^{1\%}}{q_d^{99\%}-q_d^{1\%}}\right\rfloor,
0,255\right).
$$

分位数而不是 min/max 的原因是：极少量 outlier 不会把区间拉得很宽、导致大多数正常动作挤在很少的 bins 中。

```python
def tokenize_one_action(action_7d, q01, q99):
    tokens = []
    for d, value in enumerate(action_7d):
        value = clip(value, q01[d], q99[d])
        bin_id = uniform_bin(value, q01[d], q99[d], num_bins=256)

        # Llama 只预留 100 个 special tokens，不够 256 个动作类别
        # 因而复用 tokenizer 词表最后 256 个、使用最少的 token
        token_id = llama_vocab_last_256[bin_id]
        tokens.append(token_id)
    return tokens                  # 7D action → 7 个 autoregressive tokens
```

训练时仅在这些 action target tokens 上计算 cross-entropy；图像、提示模板和任务指令是条件，不作为目标语言继续生成。

### 4.2 它怎样学到维度耦合？

第 2 个 action token 可以看第 1 个，第 7 个可以看前 6 个，因此模型学习的是：

$$p(z_1,\ldots,z_7\mid I,\ell)=\prod_d p(z_d\mid I,\ell,z_{<d}).$$

所以 HTML 中“每维独立分箱，因此完全忽略维度相关性”说得太重。**独立的是量化器，不是联合概率模型。** 自回归条件可以学习维度相关性。

同理，“离散自回归不能表达多模态”也不准确：categorical distribution 本身可以多峰，连续 7 维的自回归联合分布也能多峰。真正的限制是：

- 常用 greedy/argmax 部署会选一个模式；
- 每维均匀量化造成离散误差；
- 单步动作没有显式建模跨时间的 chunk 结构；
- 7 次 token 解码叠加延迟。

### 4.3 一格等于多少毫米？

不能统一回答。每维的物理单位、1%/99% 范围和机器人数据统计不同；旋转、平移、gripper 也不是同一单位。bin 宽度是 `(q99-q01)/256`，必须读取对应动作统计后才能换算。

### 4.4 和 FAST 不是一回事

| | OpenVLA 逐维分箱 | [[FAST Tokenizer]] |
|---|---|---|
| 输入单位 | 一个时间步、每维独立量化 | 整段 action chunk |
| 压缩 | 7D 通常变成 7 tokens | 利用时间频率冗余压缩序列 |
| 时间耦合 | 没有显式 chunk | 在 chunk 表示中保留时间结构 |
| 使用论文 | OpenVLA、RT-2 风格 | [[π₀.5]] 预训练 |

## 05 训练数据：970k 不是把 OpenX 全部倒进去

Open X-Embodiment 当时包含 70 多个数据集、超过 2M trajectories。OpenVLA 为了让输入输出一致并平衡数据分布，做了筛选：

```python
def curate_openx(dataset):
    if not dataset.is_manipulation:
        return DROP
    if not dataset.has_third_person_camera:
        return DROP
    if not dataset.supports_single_arm_end_effector_control:
        return DROP

    # 通过第一轮后，沿用 Octo 的 heuristic mixture weights
    # 多样任务/场景上调，重复或低多样数据下调/移除
    return sample_with_octo_weight(dataset)
```

最终 mixture 约 970k robot trajectories/episodes，覆盖多个机器人、任务和场景。它主要是**真实机器人 action data**，不是像 π₀.5 那样在 VLA 训练阶段继续混入网页 caption/VQA 数据。

### DROID 的有趣失败

DROID 以保守的 10% 权重加入，但 action token accuracy 一直较低；作者在最后三分之一训练中把它移除，并把权重重新分配给其他数据。

这说明：更丰富的数据不一定立刻有利。如果一个固定容量模型在给定采样权重和训练预算下无法拟合该域，把它加入 mixture 可能拖累最终 checkpoint。

### No-op 清洗为什么能决定机器人会不会“冻住”？

BridgeData V2 每条 demonstration 的第一个 transition 都记录全零 action。高容量单步 policy 很容易学到一个看似安全的高概率模式：输出零并保持不动。

```text
训练数据中 no-op 比例偏高
→ CE 奖励模型给 zero-action 更高概率
→ 部署时某个状态选择 zero-action
→ 下一帧几乎不变
→ 模型再次看到相同状态，再输出 zero-action
→ 形成冻结吸收态
```

OpenVLA 删除每条 Bridge demonstration 的首 transition，便大幅缓解冻结。LIBERO 适配时也过滤 translation/rotation 近零且 gripper 不变的 no-op actions。

## 06 训练：为什么要端到端，而不是冻结 vision？

OpenVLA 从 Prismatic-7B 开始，**包括视觉 encoder 在内全部微调**：

```python
def train_openvla(batch):
    image = resize(batch.third_person_image, [224, 224])
    prompt = f"What should the robot do to {batch.task}? A:"
    action_tokens = discretize_to_256_bins(batch.action)

    logits = prismatic_vlm(image, prompt, teacher_force=action_tokens)
    loss = cross_entropy(logits[action_positions], action_tokens)
    loss.backward()                 # gradients reach Llama, projector, SigLIP, DINOv2
```

### 关键训练配置

| 项目 | 论文配置 | 证据位置 |
|---|---:|---|
| epochs | 27 | Sec. 3.4 |
| learning rate | 固定 `2e-5`，无 warmup | Sec. 3.4 |
| batch size | 2048 | Sec. 3.5 |
| 集群 | 64×A100 | Sec. 3.5 |
| 时间 | 14 天，约 21,500 A100-hours | Sec. 3.5 |
| 训练停止观察 | action-token accuracy 超过 95% 后表现仍提升到最终训练 | Sec. 3.4 |

Appendix D.3 显示，在两个早期 VLA 设置中 fine-tuned vision 平均约 80.0%，frozen vision 约 46.7%，部分 frozen policy 甚至不稳定。因此这里视觉 backbone 不是只提供通用语义，还必须适配控制所需的细粒度场景特征。

但“端到端更新有利于动作”留下另一个问题：只用 robot data 会不会遗忘 VLM 的开放语义？论文没有 co-training 网页数据，且明确把这个问题留作未来研究；RT-2-X 在 semantic generalization 上仍略强。

## 07 推理：一次只预测一个动作，延迟就是系统动力学

OpenVLA 的输入限制非常明确：

- 单张第三人称图像；
- 一条语言指令；
- 没有 proprioceptive state；
- 没有 observation history；
- 每次只输出一个 7D relative action；
- 没有 action chunking。

最终 OpenVLA 以 bfloat16 加载约需 15GB 显存，在 RTX 4090 上约 6Hz（未使用 compilation、speculative decoding 等加速）。

![[90 Attachments/OpenVLA/openvla-teaching-latency.svg]]

*教学示意图（根据 Sec. 5.4 与 Appendix D.4 绘制，并非论文原图）。*

### 为什么慢一点不只是吞吐少一点？

Franka/Bridge 使用 non-blocking controller：模型计算下一步时，机器人仍执行上一动作。因此推理从训练时约 5Hz 降到 1.2Hz，会让同一个动作保持更久，改变闭环系统动态。

这解释了一个反直觉实验：int8 的数值位数比 int4 多，但当时 kernel/量化开销使它更慢，于是真机更差。换成 blocking control、消除不同方法的时序差异后，三种精度的成功率误差条重叠。

## 08 实验一：开箱即用的多机器人泛化

![[90 Attachments/OpenVLA/openvla-fig3-generalization.png]]

*论文 Figure 3 · PDF 第 7 页 · 原论文图片。每种方法在 BridgeData V2 上共 170 rollouts。*

BridgeData V2 WidowX：17 tasks × 10 trials，覆盖 visual、motion、physical、semantic generalization 和 language grounding。

| 模型 | 平均成功率 |
|---|---:|
| RT-1-X | 18.5% |
| Octo | 20.0% |
| RT-2-X | 50.6% |
| OpenVLA | **70.6%** |

Google robot：12 tasks × 5 trials，共 60 rollouts；OpenVLA 85.0%，RT-2-X 78.3%，两者在误差范围内更接近。

### 16.5 个百分点该怎样读？

论文汇总 29 个任务，报告 OpenVLA 比 RT-2-X 高 16.5 个绝对百分点，同时参数从 55B 降到 7B。但这不是“架构效率”的纯净实验：

- OpenVLA 用 970k trajectories，RT-2-X 约 350k；
- OpenVLA 对 Bridge zero-actions 做了更谨慎清洗；
- OpenVLA 使用 SigLIP+DINOv2；
- RT-2-X 闭源，无法在相同清洗数据和训练预算下重训。

所以实验支持“OpenVLA 这个完整系统更强”，不能把全部差距归因于更小的参数量或双 encoder。

### 它在哪一项没有赢？

Bridge semantic generalization：RT-2-X 38.8%，OpenVLA 36.3%。RT-2-X 在机器人训练时还 co-fine-tune Internet pretraining data，且 base VLM 更大；OpenVLA 只用机器人数据微调。作者据此认为保留互联网共训可能有利，但这不是严格隔离变量的消融。

## 09 实验二：新机器人上，通用预训练何时最有用？

![[90 Attachments/OpenVLA/openvla-fig5-adaptation.png]]

*论文 Figure 5 · PDF 第 9 页 · 原论文图片。7 个任务各 10-150 条 demonstrations，共 129 rollouts/method。*

测试包含 Franka-Tabletop 5Hz 与 Franka-DROID 15Hz：

- 窄、单指令、要求精细轨迹的任务：Diffusion Policy 常更强、更平滑；
- 多物体、多指令、含 distractors 的任务：OpenVLA/Octo 的通用预训练更有优势；
- OpenVLA 总平均最好，也是唯一所有所测任务都至少达到 50% 的方法；
- `OpenVLA (scratch)` 更弱，支持 OpenX robot pretraining 对下游适配有用。

这不是“foundation model 总比从头训练好”。更准确的分界是：语言/对象组合越多，通用视觉语言先验越值钱；单一精密任务中，带 history、proprioception、action chunking 的专用 Diffusion Policy 仍可能占优。

## 10 LoRA：可训练参数少，不等于模型本身变小

![[90 Attachments/OpenVLA/openvla-table1-lora.png]]

*论文 Table 1 · PDF 第 10 页 · 原论文图片。每种方法 33 rollouts。*

| 方法 | 成功率 | 可训练参数 | batch 16 显存 |
|---|---:|---:|---:|
| Full FT | 69.7±7.2% | 7,188.1M | 163.3GB，2卡 FSDP |
| Last layer | 30.3±6.1% | 465.1M | 51.4GB |
| Frozen vision | 47.0±6.9% | 6,760.4M | 156.2GB，2卡 |
| Sandwich | 62.1±7.9% | 914.2M | 64.0GB |
| LoRA r=32 | **68.2±7.5%** | **97.6M / 1.4%** | 59.7GB |
| LoRA r=64 | 68.2±7.8% | 195.2M | 60.5GB |

LoRA r=32 在这些任务上接近 full fine-tuning，可在单张 A100 上约 10-15 小时完成；full FT 使用 8 张 A100、每任务 5-15 小时。

> [!warning] 不要忽略脚注
> Sec. 5.3/5.4 的 LoRA 与量化实验使用了**较小训练 mixture、只有 SigLIP 的稍小模型版本**，不是 Figure 2 的完整 DINOv2+SigLIP final OpenVLA。它证明方法可行，但具体显存和成功率不能无条件外推到完整 checkpoint。

LoRA 只减少反向传播与 optimizer state 的成本；基础 7B 权重仍需加载，所以 1.4% trainable 不意味着推理模型只有原来的 1.4%。详见 [[LoRA]]。

## 11 量化：4-bit 没降分，不等于 4-bit 普遍更准

![[90 Attachments/OpenVLA/openvla-fig6-table2-quantization.png]]

*论文 Figure 6 + Table 2 · PDF 第 10 页 · 原论文图片。Bridge 8 个任务、80 rollouts/method。*

| 精度 | 非阻塞 Bridge 成功率 | 显存 |
|---|---:|---:|
| bfloat16 | 71.3±4.8% | 16.8GB |
| int8 | 58.1±5.1% | 10.2GB |
| int4 | 71.9±4.7% | 7.0GB |

不能说 int4 比 bfloat16 “更好”：71.9 与 71.3 的误差条大幅重叠。结论是 int4 在该设置中**没有可测的明显退化**，同时显存减半以上。

Appendix D.4 的 blocking control 结果进一步是：bf16 70.0±5.1%、int8 74.4±4.9%、int4 68.8±5.2%，均有重叠误差条。这支持 int8 在非阻塞实验中的掉分主要来自低吞吐造成的控制时序变化。

## 12 Appendix 里还告诉我们什么？

### 12.1 OpenX 与双 encoder 谁更重要？

8 个 Bridge tasks、80 trials/method：

```text
OpenVLA                                  76.3 ± 4.8%
OpenVLA-Bridge（无 OpenX mixture）       45.6 ± 5.6%
OpenVLA-Bridge-SigLIP（再去 DINOv2）     40.6 ± 5.5%
```

大规模跨机器人数据的约 30 点收益远大于这组实验里 DINOv2 的约 5 点收益。

### 12.2 能否从真实机器人预训练迁移到模拟器？

LIBERO 四个 suite，每 suite 10 tasks、每 task 50 demonstrations；每个统计量 500 trials × 3 seeds = 1500 trials：

| 方法 | 平均成功率 | 平均 rank |
|---|---:|---:|
| Diffusion Policy scratch | 72.4±0.7% | 2.5 |
| Octo fine-tuned | 75.1±0.6% | 2.0 |
| OpenVLA LoRA | **76.5±0.6%** | **1.5** |

OpenVLA 平均最好，但优势小于真实机器人适配；作者推测这是因为预训练只含真实机器人数据，存在 sim-real domain gap。

数据处理同样关键：重新渲染 256×256 图像、过滤 no-op、回放并删除失败 demonstrations，而且只用静态第三人称相机保证比较公平。

## 13 主动回答最容易产生的疑问

> [!question] OpenVLA 是不是“一个 VLM 换了输出头”？
> 比这更直接：它尽量不换结构，动作 bin 复用原 Llama vocabulary，仍由语言 softmax 与 next-token loss 学习。真正的大改动发生在数据分布和端到端参数更新，而不是新建动作网络。

> [!question] 为什么训练只在 action tokens 上算 loss，模型还不会忘记语言吗？
> 预训练语言/视觉知识作为初始化保留，但机器人微调期间没有语言或网页 token loss 主动约束它。论文没有证明语言生成能力被完整保留，并把 robot+Internet co-training 列为未来问题；这正是 [[π₀.5]] 加入 WD 的重要背景。

> [!question] 27 epochs 会不会过拟合？
> 论文观察真机表现随 action-token accuracy 提升而继续改善，最终跑 27 epochs；但没有完整 epoch-by-epoch 泛化曲线，也没有证明 27 对其他 mixture/模型最优。机器人 trajectory 的有效多样性和常规语言 corpus 不同，不能照搬 LLM 的 1-2 epoch 经验。

> [!question] 970k 是 970k 个独立技能吗？
> 不是，是 trajectories/episodes；同一任务、场景或机器人会有许多轨迹。数据规模必须与 task diversity、scene diversity 和 mixture weights 一起理解。

> [!question] 为什么不输入 state？只看图真的够吗？
> 在 WidowX/Google/Franka 所测任务中，图像可提供较多末端和物体状态；但遮挡、关节极限、速度和接触力无法可靠从单帧恢复。论文明确把 multi-image、proprioception 和 observation history 留作未来工作。

> [!question] OpenVLA 可以 50Hz 灵巧控制吗？
> 不可以从论文证据推出。未加速 RTX 4090 约 6Hz，作者也把 ALOHA 50Hz 列为尚未满足的场景，并建议 action chunking 或 speculative decoding。

## 14 局限：它建立了基线，也把下一代问题暴露出来

| 局限 | 直接后果 | 后续路线 |
|---|---|---|
| 单图、无 state/history | 看不到速度、遮挡前状态和关节内部信息 | π₀、HyVLA 的 state/history；触觉 VLA |
| 单步 7D action | 7B 推理频率等于控制频率，时间一致性弱 | [[Action Chunking]]、Flow/Diffusion |
| 逐维均匀量化 | 分辨率固定、outlier clip、几何结构不显式 | FAST、continuous Action Expert |
| 只用 robot data 微调 | 可能损失开放语义，semantic gen 未领先 | π₀.5 异构 co-training |
| 单臂 EEF 数据筛选 | 难直接覆盖双臂、移动底盘和异构传感器 | richer action/state schemas |
| 通常 <90% 成功率 | 尚不足以直接用于高可靠部署 | recovery、safety、post-training |

> [!danger] HTML 指南中需要纠正的三句话
> 1. 256 bins 的物理精度不是固定 `0.4 mm`，由每维 1%/99% 范围决定。
> 2. 自回归离散模型不是数学上“不能多模态”；问题在量化、单步时序和部署 decoding。
> 3. DINOv2 没有在这里直接给像素坐标；论文只证明融合 feature 在特定消融中约多 5 个百分点。

## 15 与 π₀ / π₀.5 的最小坐标

| 维度 | OpenVLA | [[π₀]] | [[π₀.5]] |
|---|---|---|---|
| backbone | Prismatic/Llama 2 7B | PaliGemma + Action Expert | PaliGemma + Action Expert |
| 视觉 | 单图，SigLIP+DINOv2 | 多图 | 多图 |
| state | 无 | continuous state block | tokenized prefix |
| 动作 | 单步逐维 256-bin tokens | H=50 continuous Flow | FAST 预训练 + H=50 Flow 后训练 |
| 推理 | 7 action tokens 自回归 | 10-step flow | 子任务文本 + 10-step flow |
| 核心问题 | 开放、通用、可适配的 VLA baseline | 高频连续灵巧动作 | 未见家庭的长任务泛化 |

OpenVLA 不是“落后的 π₀”。它优化的是另一个目标：最少架构改动、最大生态兼容性。π₀/π₀.5 则用专用动作计算换取连续 chunk、并行生成和更丰富条件输入。

## 16 复现清单

```python
reproduction_checks = [
    "只纳入 manipulation + third-person camera + single-arm EEF datasets",
    "逐 dataset/动作维统计 1% 与 99% quantiles",
    "确认 256 bin 到 Llama 最后 256 tokens 的映射方向",
    "loss 只覆盖 action target positions",
    "过滤 Bridge/LIBERO no-op，避免 freeze absorbing state",
    "完整微调时允许 gradient 进入 vision encoder",
    "评测要固定 initial states 并报告 rollouts/sample size",
    "非阻塞部署必须同时记录 inference latency 与 action hold time",
    "LoRA/量化数字注明使用的是较小 SigLIP-only variant",
]
```

## 17 复习卡片

> [!tip] 只记五句话
> 1. OpenVLA = Prismatic-7B 被端到端微调成动作语言模型。<br>
> 2. 单步 7D 动作逐维做 256-bin quantization，再复用 Llama 最后 256 tokens。<br>
> 3. SigLIP+DINOv2 沿 channel 融合；数据多样性贡献大于这项架构增益。<br>
> 4. 970k OpenX 数据、27 epochs、全量 vision+LLM 更新；清洗 no-op 至关重要。<br>
> 5. 它最强的贡献是开放与可适配；最大短板是单图、无 state/history、无 action chunking、约 6Hz。

## 18 下一步研究问题

1. 若保留 OpenVLA 的简单 AR 接口，加入 action chunk 后应使用 FAST、VQ 还是连续 head？
2. robot data 与 web data co-training 怎样防止语言遗忘，又不稀释动作梯度？
3. state/history 是作为 tokens、cross-attention memory，还是 adaptive normalization 更合适？
4. 数据 mixture 权重能否根据 per-domain loss/gradient 自动学习，而不是手工沿用 Octo？
5. 延迟感知训练能否让 policy 对 3-15Hz 的变化更稳健，而不只依赖更快 GPU？
