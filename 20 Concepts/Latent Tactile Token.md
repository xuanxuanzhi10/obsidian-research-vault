---
type: concept
title: Latent Tactile Token
topics: [tactile, representation, future-prediction, VLA]
---

# Latent Tactile Token

相关论文：[[N0-VTLA]] · 相关概念：[[预测触觉与观测触觉]] · [[Action Expert]]

> [!abstract] 一句话定义
> Latent Tactile Token 是把“接下来一段动作会造成怎样的净接触变化”压成少量 token，供动作模型读取的预测性中间表示；它不是原始触觉，也不是未来触觉图像本身。

## 第一层：直觉理解

想象把插头插进插座：当前触觉只告诉你“现在是否碰到了”；真正有助于选动作的问题是“沿这条轨迹继续 50 步，会顺利滑入还是撞上边缘”。Latent tactile token 就像动作前的一句接触预报。

为什么不直接预测高分辨率触觉图？控制不需要复原所有纹理细节，只需一个足以区分“滑入、卡住、抓稳、打滑”的行动相关摘要。压缩也让它能低成本接到已有 VLA 的 Action Expert。

## 第二层：机制跑通

![[n0-vtla-teaching-latent.svg]]

```python
def make_latent_touch(current_views, start_views, rgb_language_context):
    g = []
    for current, start in zip(current_views, start_views):
        feature_map = frozen_dinov2(current - start)
        # class token + 3×3 pooled spatial tokens
        g.extend(project([feature_map.cls, adaptive_pool(feature_map.patch, 3, 3)]))

    # learned queries 将任意数量视角压成固定 10 tokens
    z = predictor(g, rgb_language_context, learned_queries=10)
    return z
```

N0-VTLA 的监督目标为：

$$z^*=\frac1n\sum_k f_{enc}(t^k_{\tau+H}-t^k_\tau).$$

最小例子：如果夹爪当前刚触碰海绵，计划动作会继续闭合，那么 `g` 编码“相对 episode 起点已经产生的形变”，z 则应靠近“50 步后新增压缩形变”的 z*；若计划是松开，未来净变化方向应不同。

## 第三层：论文生态

| 方法 | 表示对象 | 是否生成未来 | 动作怎样使用 |
|---|---|---:|---|
| [[N0-VTLA]] | H 步后的净触觉变化 latent | 是，紧凑 token | z 作为 Action Expert 条件 |
| [[N0-TWAM]] | 未来触觉 residual video latent | 是，完整生成分支 | 预测 KV + 当前力场双路径 |
| [[T-Rex]] | 当前/历史 wrench 与形变 | 否 | 高频 Tactile Expert 细化动作 |
| [[FTP-1]] | 跨传感器功能槽 | 否 | 统一 Tactile Expert 条件 |

latent tactile token 与“tactile token”只差一个词，却可能有根本区别：普通 token 常是传感器当前读数的编码；这里的 latent 由当前触觉和 VL 场景共同推断，语义由未来目标规定。

## 第四层：优势、代价与证据

优势：固定 10 token、与视角数解耦、不会污染 VLM prefix、可作为小接口嫁接已有 VLA，并天然要求动作模型考虑未来接触后果。

代价与失败边界：

- 多视角平均可能丢失手指身份和局部接触位置；
- 只拟合 chunk 两端净变化，会漏掉中途发生又恢复的瞬态接触；
- baseline differencing 会受漂移与错误 reset 影响；
- 预测不等于实测，若接触发展偏离预测，仍需要高频观测通路纠偏。

N0-VTLA 的 32-candidate top-1 检索为 92.3%，当前 g 为 57%；扰动触觉比扰动 RGB+语言更显著。这支持 z 确实携带预测性触觉信息。论文没有完整报告 current-only / predicted-only / both 的公平消融，所以不能据此断言 latent prediction 必然优于所有实时触觉接口。
