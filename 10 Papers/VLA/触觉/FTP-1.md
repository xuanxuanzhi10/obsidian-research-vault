---
type: paper
title: "FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation"
year: 2026
status: read
verification: full-paper-and-appendix-checked
topics: [VLA, tactile, cross-sensor, foundation-policy, MTTS]
source: "[[FTP-1.pdf]]"
---

# FTP-1：先统一“触觉来自哪里”，再跨传感器预训练

上级地图：[[VLA 学习地图]] · 主题：[[触觉机器人学习]]  
核心概念：[[Morphology-Aware Tactile Token Space]] · [[Adaptive RMSNorm]] · [[Transformer 参数量估算]] · [[异构多域分布式训练]]
对比入口：[[触觉 VLA 方法比较]]

原文：[[FTP-1.pdf|论文 PDF]] · [[ftp1-study-guide.html|HTML 原版讲解（系统浏览器打开）]]

> [!abstract] 一句话结论
> FTP-1 的贡献不是高频力控，而是把 image / array / state 三类、21 种触觉传感器按身体功能区映射到统一 token 空间，用约 3000 小时异构数据预训练一个可迁移的 Tactile Expert。

## 先看原论文 pipeline

![[ftp1-fig2-architecture.png]]

```text
异构触觉（图像 / 阵列 / 状态）
  → 传感器专属 encoder
  → 24 个身体功能区槽位 MTTS + functional-area embedding
  → 独立 300M Tactile Expert
  → Action Expert 读取触觉、视觉语言和本体状态
  → Flow Matching action chunk
```

## 它到底在解决什么？

触觉硬件不像 RGB 相机：GelSight 输出图像，Contactile 输出空间力阵列，腕部 F/T 传感器只输出低维向量；它们还装在不同手指或手腕。若直接拼接，模型既不知道“值是什么”，也不知道“来自身体哪里”。因此 FTP-1 把问题拆成两层：

1. **传感器层**：不同原始格式先由各自 encoder 变成 token；
2. **形态层**：token 再放进共同的身体功能区坐标系，使跨硬件的“拇指接触”“腕部受力”对齐。

这和 T-Rex 是正交方向：[[T-Rex]] 追求固定硬件上的快触觉纠偏；FTP-1 追求跨硬件的广泛迁移。

## 核心一：MTTS 怎样统一 21 种传感器？

![[ftp1-teaching-mtts.svg]]

MTTS 规定 24 个槽位：`0–14` 是手部功能区，`15–20` 是腕部/手指力矩，`21–23` 保留。平行夹爪两侧分别映射到拇指尖 `slot 0` 和食指尖 `slot 1`；左右手使用不同的功能区 embedding。

```python
class MTTSTokenizer:
    def encode(self, sensor, raw, body_region, hand_side):
        # 第一步回答“信号长什么样”
        if sensor.kind == "image":
            local = sensor_specific_vit(raw)
            token = shared_T3_transformer(local).cls
        elif sensor.kind == "array":
            token = cnn(raw)                 # 保留 taxel 空间结构
        elif sensor.kind == "state":
            token = mlp(fourier_encode(raw)) # 低维力/力矩

        # 第二步回答“信号来自身体哪里”
        slot = map_region_to_24_slots(body_region)
        token += functional_area_embed(hand_side, slot)
        return slot, token
```

关键不是强迫所有原始信号同构，而是只统一**模型接口与身体语义**。新传感器仍需训练新的前端；真正复用的是共享 T3 chunk、功能区 embedding 和 Tactile Expert。

## 核心二：为什么用独立 Tactile Expert？

![[ftp1-teaching-attention.svg]]

基座沿用 π0.5 多专家架构：VLM Expert 处理图像和语言；Flow-Matching Action Expert 生成动作；本体状态通过 AdaRMSNorm 调制动作分支。新增约 300M 参数的 Tactile Expert。

```python
def ftp1_forward(rgb, language, state, tactile, noisy_action, tau):
    vl = vl_expert(rgb, language)          # 保留预训练视觉语言能力
    tac = tactile_expert(mtts(tactile))    # 跨传感器共享触觉能力

    # 非对称：action 可读 tactile；tactile 不读 action，也不读 text
    action = action_expert(
        noisy_action,
        attend_to=[vl, tac],
        modulation=ada_rms_norm(state, tau),
    )
    return action
```

如果把触觉 token 注入 VLM，稀疏且分布特殊的接触信号可能干扰视觉语言先验。独立 expert 让能力和参数路径隔离；动作读取触觉，但触觉不被动作噪声反向“污染”。论文还报告更复杂 MoE 融合没有稳定增益，因此采用简单设计。

## 数据为什么能支撑跨传感器迁移？

FTP-1 汇总 26 个数据源、21 种传感器（7 image、5 array、9 state），总计约 3000 小时，并额外采集 4000 条长时程灵巧操作。重采样后，人类 / 灵巧手机器人 / 平行夹爪机器人约占 `20% / 30% / 50%`，避免大数据源吞没小传感器。

```python
for sample in heterogeneous_sources:
    tactile_tokens = mtts_tokenizer(sample.sensor, sample.tactile)
    action_target = map_to_UAS(sample.robot_action)
    loss = flow_matching(action_target, tactile_tokens, sample.rgb, sample.text)
    loss *= dataset_sampling_weight(sample.source)
```

UAS 用统一动作维度承接不同 embodiment；缺失维度通过 mask 不计入损失。MTTS 统一“感觉”，UAS 统一“动作”，两端共同让多硬件数据能进同一策略。

## 实验回答了哪些疑问？

### 见过的传感器：触觉预训练是否比只加分支更好？

UniVTAC 六任务：FTP-1 平均 `66.66%`，不含两项几乎可由视觉解决的 Lift 任务为 `59.5%`；相对次优分别高约 17.5 个百分点。真机 seen-sensor 六项平均 `62.5%`，π0.5 为 `45.3%`。

### 没见过的传感器：是不是只记住了硬件？

Xense 与 Contactile 三任务中，FTP-1 为 `46.6%`，从零加入同架构触觉分支的 FTP-π0.5 为 `15.0%`。新 sensor encoder 从零训练，但共享触觉组件复用，说明迁移不只来自输入格式。

### 会不会只是预训练数据分布更接近下游？

NTP-1 使用同一数据与设置预训练，但预训练时移除触觉。FTP-1 在 FlexivXense 上比 NTP-1 高 `37.5` 个百分点，支持“触觉分支确实学到可迁移知识”；但这仍是有限任务/机构上的证据，不等于任意新传感器都能零样本使用。

## 它没有解决什么？

> [!warning] 论文明确的证据边界
> 作者明确写道：FTP-1 主要研究通用触觉感知，尚未解决基于触觉/力的 servoing 与控制；未来方向才包括未来触觉预测和预测式低层控制。

- 新传感器不是即插即用，仍需适配 encoder 和下游示范；
- 数据规模虽大，但传感器/任务覆盖仍有限；
- 论文观察到插入减速、维持压力等反应行为，不等于证明了独立高频闭环；
- 对 HyVLA 的价值首先是 **MTTS + tactile expert 预训练初始化**，不是直接替代 [[异步触觉动作细化]]。

## 对我们的 HyVLA 最直接的启示

```text
FTP-1 提供“通用触觉词表与预训练触觉大脑”
        ↓
HyVLA 保留视觉语言 + Flow Matching 主干
        ↓
若要高频接触修正，再接 T-Rex 式快速触觉细化分支
        ↓
若要行动前预判接触，再加入 N0-TWAM 式未来触觉目标
```

因此它最像**触觉基础层**，而不是完整的高频控制答案。
