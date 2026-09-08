---
type: concept
topic: conditioning
status: deep-explained
verification: FTP-1-appendix-B3-checked
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

        # 不同实现可只生成 scale，也可生成 shift / residual gate
        scale, shift, gate = condition_mlp(condition).chunk(3, dim=-1)
        y = (1 + scale[:, None, :]) * x_hat + shift[:, None, :]
        return x + gate[:, None, :] * attention_or_ffn(y)
```

不要把这段伪代码当成 FTP-1 官方实现。论文只明确说明“经 adaptive RMSNorm 注入 attention blocks”，没有在文中展开其究竟生成 scale、shift、gate 中的哪些量。

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

## 第四层：证据、边界与待核实项

- **FTP-1 明确声称**：preliminary experiments 中，AdaRMSNorm 注入 proprioception 比独立 proprioceptive token 有更好 generalization/robustness（Appendix B.3）。
- **尚未报告**：正文没有给独立数值表、任务拆分或统计显著性，因此不能量化这项设计贡献。
- **已核对配置**：proprioception 经 Fourier encoding、3-layer ReLU MLP、LayerNorm；与归一化后的 flow-timestep features 拼接；注入 attention blocks。
- **待核实**：π0/π0.5、T-Rex、N0-TWAM 各自究竟采用 AdaRMSNorm 还是 AdaLN、是否含 zero-init/gate，必须分别依据原文或代码，不能从家族名称推断。

