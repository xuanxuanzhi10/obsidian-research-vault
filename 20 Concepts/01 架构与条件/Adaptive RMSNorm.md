---
type: concept
topic: conditioning
status: deep-explained
verification: FTP-1-appendix-B3-and-pi05-appendix-E-checked
aliases: [自适应 RMSNorm, AdaRMSNorm]
source_guide: "[[tactile-vla-qa2.html]]"
---

# Adaptive RMSNorm：让条件变成每一层的“控制旋钮”

原版概念答疑：[[tactile-vla-qa2.html|触觉 VLA 深度答疑（二）HTML]]  
论文实例：[[FTP-1]] · 相关：[[Flow Matching]] · [[Action Expert]]

> [!summary] 一句话定义
> RMSNorm 先把 hidden state 的整体尺度归一化；Adaptive RMSNorm 再让 timestep、本体状态等条件动态生成缩放、偏移或门控，使同一层在不同状态下采用不同的特征处理方式。

![[adarmsnorm-conditioning.svg]]

*教学示意图（根据通用机制与 FTP-1 Appendix B.3 绘制，并非论文原图）。*

## 第一层：为什么需要这个“控制旋钮”？

Flow Matching 网络在高噪声时要决定动作大方向，在低噪声时要修细节；机器人伸直、弯曲或靠近关节极限时，同一个视觉目标对应的可行动作也不同。因此 timestep 和 proprioception 不只是“序列里又一个事实”，而是应该改变**每一层怎样处理所有 action tokens**。

把 condition 做成普通 token 当然也可能学会这些关系，但它依赖 attention 主动读取；adaptive normalization 则提供直接的逐层调制路径。这是不同 inductive bias，不应说普通 token 必然被“淹没”。

类比：普通 condition token 像会议中的一名参会者，其他 token 可以向它提问；AdaRMSNorm 像按机器人状态调整整间会议室的控制面板。类比的边界是：调制不是推理主体，它只改变特征通道的尺度/偏移/门控。

## 第二层：从 RMSNorm 跑到 Adaptive RMSNorm

### 1. RMSNorm 做了什么？

对单个 token `x∈R^d`：

$$\operatorname{rms}(x)=\sqrt{\frac1d\sum_i x_i^2+\epsilon},\qquad
\hat x=\frac{x}{\operatorname{rms}(x)}.$$

标准 RMSNorm 通常再乘固定可学习权重 `w∈R^d`。它与 LayerNorm 的主要区别是通常不减均值；但具体库是否带 bias，要看实现。

### 2. Adaptive 在哪里？

```python
class AdaptiveRMSNorm:
    def forward(self, x, condition):
        # x: [B, N, d]，N 个 action tokens
        # condition: [B, c]，例如 flow timestep + proprioception
        x_hat = x / sqrt(mean(x**2, dim=-1, keepdim=True) + eps)

        # FTP-1/OpenPI 官方实现生成 scale、shift、residual gate
        scale, shift, gate = condition_mlp(condition).chunk(3, dim=-1)
        y = (1 + scale[:, None, :]) * x_hat + shift[:, None, :]
        return x + gate[:, None, :] * attention_or_ffn(y)
```

这段是教学伪代码，不是逐行复制，但它的 `scale + shift + gate`、zero-initialized condition projection，以及对 Attention/FFN 两条 residual branch 的调制，已经由 FTP-1 官方仓库实现核对。

### 3. FTP-1 的条件怎样形成？

```python
def ftp1_condition(proprio, flow_timestep):
    # Appendix B.3 已核对
    state = concat(proprio, fourier_encode(proprio))
    state = layer_norm(relu_mlp_3_layers(state))

    time = layer_norm(timestep_features(flow_timestep))
    return concat(state, time)  # 随后注入 attention blocks
```

允许路径：`proprio/timestep → condition MLP → 每层归一化调制 → action attention`。  
失败路径：若 condition 只在输入端注入而深层没有稳定读取，它对后续动作细化可能变弱；但是否真的失败必须由同模型消融证明。

## 第三层：和普通 token 有什么差别？

| 方法 | 条件怎样影响 hidden | 优点 | 代价/边界 |
|---|---|---|---|
| condition token | 通过 attention 被其他 token 读取 | 表达灵活、可形成显式交互 | 是否被读取由模型学习 |
| 加法 embedding | 直接加到 token | 简单、便宜 | 所有通道的作用形式较固定 |
| Adaptive RMSNorm | 每层动态调制通道 | 全局、逐层、几乎不增加序列长度 | 强 modulation 可能扰动预训练分布 |
| Cross-attention | condition 作独立 K/V | 路径明确、容量大 | 多一套 attention 计算与参数 |

增加一个 token 的确会让 attention 长度从 `N` 变为 `N+1`，但通常不是显著算力瓶颈；AdaRMSNorm 的核心优势是**调制形式和逐层直达路径**，而不是省掉一个 token 的计算。

## 重要澄清：用了 AdaRMSNorm，不等于用它注入 state

同一个 adaptive norm 可以接收 timestep、class label、text condition 或 robot state。必须检查传给 condition MLP 的究竟是什么，不能从“模型用了 AdaRMSNorm”推断“state 没有 token”。详见 [[状态条件注入方式比较]]。

| 模型 | Adaptive norm 接收什么 | State 走什么路径 | 已核对结论 |
|---|---|---|---|
| π0 | 不使用 AdaRMSNorm | continuous state 作为 suffix state token | 官方 OpenPI 代码 |
| π0.5 | Flow Matching timestep | state 离散化后进入 tokenized prompt | π₀.5 Appendix E + 官方 OpenPI 代码 |
| HyVLA | 论文未报告使用 AdaRMS/AdaLN | projected state 是独立 `[s_t]` block | HyVLA Sec. 2.3 |
| FTP-1 | timestep + proprioception | final design 用二者共同调制 Action Expert | FTP-1 Appendix B.3 + 官方代码 |

所以“π0.5 只用 AdaRMSNorm 注入 proprioception，不再使用 state token”是错误的；它把 **timestep** 放进 AdaRMSNorm，而 state 已在 prompt token 中。HyVLA 也不能按 π0.5 猜测：其论文明确画出了 state block。

## 第四层：证据、边界与待核实项

- **FTP-1 明确声称**：preliminary experiments 中，AdaRMSNorm 注入 proprioception 比独立 proprioceptive token 有更好 generalization/robustness（Appendix B.3）。
- **官方代码确认**：condition 经过 zero-initialized Dense 投影为 `3d`，切分成 scale/shift/gate；pre-attention 和 pre-FFN 都执行 adaptive RMSNorm，gate 控制对应 residual branch。zero-init 使初始调制与残差增量为零，训练再逐渐打开条件作用。
- **尚未报告**：正文没有给独立数值表、任务拆分或统计显著性，因此不能量化这项设计贡献。
- **已核对配置**：proprioception 经 Fourier encoding、3-layer ReLU MLP、LayerNorm；与归一化后的 flow-timestep features 拼接；注入 attention blocks。
- **边界**：上面的实现可作为 π0.5/FTP-1 代码族的具体例子，但 T-Rex、N0-TWAM 或其他 DiT 的 AdaLN/AdaRMS 仍应分别核对，不能由名称推断完全相同。
