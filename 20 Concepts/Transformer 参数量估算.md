---
type: concept
topic: model-scaling
status: deep-explained
verification: FTP-1-appendix-B4-checked
aliases: [Transformer Parameter Estimation, 参数量速算]
source_guide: "[[tactile-vla-qa2.html]]"
---

# Transformer 参数量估算：先看矩阵形状，不要只背 `12Ld²`

原版概念答疑：[[tactile-vla-qa2.html|触觉 VLA 深度答疑（二）HTML]]  
论文实例：[[FTP-1]] · 相关：[[Mixture of Transformers]]

> [!summary] 一句话定义
> 参数量估算就是把每层的 Q/K/V/O 与 FFN 权重按输入、输出维度逐项相乘；`12Ld²` 只适用于特定的标准配置，不是普适定律。

![[transformer-parameter-ledger.svg]]

*教学示意图（根据矩阵形状绘制，并非论文原图）。*

## 第一层：为什么 `层数 × 宽度²` 能速算？

Transformer 大多数参数都在二维线性矩阵中。如果 attention 的内部总宽度等于 hidden size `d`，且普通 FFN 宽度 `d_ff=4d`：

$$P_{attn}\approx4d^2,\qquad P_{ffn}\approx2d\,d_{ff}=8d^2,$$

所以 `L` 层约为：

$$P\approx L(4d^2+8d^2)=12Ld^2.$$

这个公式省略 bias、norm、embedding、input/output heads，也假设 FFN 只有两个矩阵。

## 第二层：用最小例子逐矩阵计算

```python
def transformer_block_params(d_model, n_q_heads, n_kv_heads,
                             head_dim, d_ff, gated_ffn=False):
    # Q 可以比 K/V 拥有更多 heads：MHA / GQA / MQA 都能计算
    q_width = n_q_heads * head_dim
    kv_width = n_kv_heads * head_dim

    # Wq + Wk + Wv + Wo
    attention = (
        d_model * q_width
        + 2 * d_model * kv_width
        + q_width * d_model
    )

    # 普通 FFN 两个矩阵；SwiGLU/GeGLU 是 gate/up/down 三个矩阵
    ffn_matrices = 3 if gated_ffn else 2
    ffn = ffn_matrices * d_model * d_ff

    # bias、norm 通常是 O(d)，先作为小项另计
    return attention + ffn
```

### FTP-1 Tactile Expert 为什么约 300M？

论文 Appendix B.4 给出：`d_model=1024`、`L=18`、`d_ff=4096`、`8 query heads`、`head_dim=256`。官方代码进一步确认 `num_kv_heads=1`，因此这里是 MQA，并使用 gate/up/down 三矩阵 FFN：

```python
q_width  = 8 * 256                       # 2048
kv_width = 1 * 256                       # 256（MQA）
P_attn = (1024*q_width                   # Wq
          + 2*1024*kv_width              # Wk, Wv
          + q_width*1024)                # Wo
        # = 4.72M / layer
P_ffn = 3 * 1024 * 4096                  # gate/up/down = 12.58M / layer
P_core = 18 * (P_attn + P_ffn)           # ≈ 311.4M
```

这与官方配置注释的 `311M params` 一致，也解释了论文为何把它称为约 300M。HTML 里的 `12×18×1024²≈226M` 同时假定 `head_dim=d/heads=128`、普通 MHA 和两矩阵 FFN；FTP-1 三个条件都不同，所以不能直接套用。

### 注意力头数到底影响参数量吗？

- 固定 `d_model` 且固定 `d_attn=d_model` 时，改变 head 数只会反向改变 `head_dim`，参数量近似不变。
- 固定 `head_dim` 时，增加 head 数会增大 `d_attn=n_heads×head_dim`，QKV/O 参数随之增加。

所以“头数不影响参数量”只有在**总 attention 投影宽度固定**时成立。

## 第三层：常见架构为什么让速算失灵？

| 变化 | 参数式如何变 | 必须检查什么 |
|---|---|---|
| SwiGLU / GeGLU | FFN 常由 2 个矩阵变 3 个 | `d_ff` 是否已按门控缩小 |
| GQA / MQA | K/V 头少于 Q 头，减少 K/V 参数与 cache | `n_kv_heads` 与各自 head_dim |
| 非标准 head_dim | `d_attn≠d_model` | 不能使用 `4d²` |
| MoT | 每个 expert 有独立投影/FFN | 哪些参数共享，哪些分模态 |
| embedding / LM head | 可能占大量参数 | vocab、是否 weight tying |
| LoRA | 只增加低秩增量 | 区分总参数和 trainable 参数 |

```python
def estimate_model(config):
    total = config.layers * exact_block_matrices(config)
    total += embedding_and_output_heads(config)
    total += modality_adapters_and_projectors(config)
    return total
```

## 第四层：参数量代表什么，又不代表什么？

参数量主要限制表示容量与显存占用，但不能单独推出：推理速度、有效能力、是否过拟合或是否“会推理”。这些还取决于序列长度、激活量、数据质量、优化、稀疏性、计算精度和预训练来源。

### 证据边界

- FTP-1 论文明确报告主要配置和约 300M 规模；官方仓库 `gemma_300m` 配置进一步给出 `num_kv_heads=1` 并标注 `311M params`。
- 官方实现的 FeedForward 使用 gating/up/linear 三个主权重，因此上面的约 311.4M 核心层推算与代码结构一致；小量 norm 等参数不会改变量级。
- HTML 中按模型量级列出的“能力范围”是教学性经验，不是可验证的硬阈值，因此不作为知识库事实表保留。
