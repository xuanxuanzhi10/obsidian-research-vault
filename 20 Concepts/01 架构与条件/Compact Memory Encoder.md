---
type: concept
topic: temporal-modeling
status: deep-explained
aliases: [紧凑记忆编码器]
---

# Compact Memory Encoder

> [!summary] 一句话
> 在 image encoder 内把 K 帧历史压缩进当前帧 token，上层 VLM 仍只接收单帧数量的 token。

![[90 Attachments/HyVLA/hyvla-compact-memory.svg]]

## 直觉理解

人看操作视频时，不会把过去六帧逐像素背下来，而会让当前画面携带“手刚从左边移过来”“杯子已经被抬起”的历史。Compact Memory 做的不是另写一份视频摘要，而是在图像编码器内部让当前 patch 吸收对应位置的过去信息。

## 因果链

机器人要判断速度、遮挡和操作阶段 → 单帧不够 → 直接拼 K 帧让 VLM token 数乘 K → 将时空注意力分解并尽早压缩 → 当前帧 token 带着历史进入 VLM。

## 张量怎样变化？

设每帧 `n` 个 patch、隐藏维 `d`：

```python
# 教学伪代码：HyVLA Compact Memory Encoder
# 关键：没有 learnable query token，也不是 Perceiver-style resampler。

class CompactMemoryEncoder:
    def __init__(self, vit):
        # 复用原 ViT 已有的 QKV 和 output projection
        # 论文强调：不增加新的可学习参数
        self.qkv = vit.qkv
        self.out_proj = vit.out_proj

    def memory_block(self, frame_tokens):
        # frame_tokens: [B, K, N, D]
        # B=batch, K=历史帧数, N=每帧 patch 数, D=hidden dim

        x = add_fixed_temporal_encoding(frame_tokens)
        # temporal encoding 采用固定 sinusoidal 编码；当前帧 e(0)=0

        # 1. 交换 K 与 N 的观察方式：固定 patch，沿时间看历史
        x_by_patch = rearrange(x, "B K N D -> B N K D")
        # 每个 patch 位置形成一条长度 K 的时间序列

        x_by_patch = causal_temporal_attention(
            x_by_patch,
            qkv=self.qkv,
            out_proj=self.out_proj,
        )  # [B, N, K, D]；过去可影响当前，未来不能影响过去

        # 2. 换回逐帧表示，在每一帧内部理解空间关系
        x = rearrange(x_by_patch, "B N K D -> B K N D")
        x = spatial_attention_per_frame(x)  # [B, K, N, D]

        # 3. 上层 VLM 不保留全部历史，只接收“吸收过历史”的当前帧
        current_tokens = x[:, -1, :, :]     # [B, N, D]
        return current_tokens
```

逐行理解：`N` 条时间序列先回答“同一个 patch 位置刚才发生了什么”，随后每帧的 `N` 个 patch 再回答“手、物体和背景如何组合”。最后的 `[:, -1]` 是 compact 的关键：历史信息留下，历史 token 不继续进入上层 VLM。

> [!warning] 为什么图一的伪代码风格可取、内容却不能照搬？
> `class + comments + shape` 很适合学习；但图一把机制写成 learnable query cross-attention，实际对应的是另一类 resampler。这里保留呈现方式，替换为 HyVLA 原文机制。

## 为什么复杂度更低？

- K 次帧内空间注意力：`O(Kn²)`；
- n 次跨 K 帧时间注意力：`O(nK²)`；
- 总计 `O(Kn²+nK²)`；
- 全部时空 token 一次注意力为 `O((Kn)²)=O(K²n²)`。

更重要的是，上层 VLM 只继续处理 n 个 token。

## HyVLA 的具体实现

- 每 4 个 ViT layers 插入 temporal attention；
- temporal pass 复用原 ViT QKV 与输出投影；
- fixed sinusoidal temporal encoding，当前帧 `e(0)=0`；
- 不增加可学习参数；
- pre-training `K=1`，SFT `K=6`。

## K=1 为什么重要？

它被设计为原图像编码器的精确特例，允许先利用单帧大数据训练，再切换到多帧任务。如果 K=1 时模型行为完全不同，预训练权重就很难平滑复用。

## 证据与未证实部分

RoboTwin 中 full 为 90.9/90.1，w/o memory 为 88.8/88.6。历史有效得到支持，但“每 4 层”“共享 QKV”“固定编码”等选择没有逐项消融。

## 与相近方法的边界

| 方法 | 历史怎样进入模型 | 上层 token 数 |
|---|---|---:|
| 直接拼接 K 帧 | 所有帧 token 一起送入 VLM | `K·n` |
| Learned-query resampler | 用额外 query cross-attention 压缩 | 由 query 数决定 |
| HyVLA Compact Memory | 同 patch 跨时间，再做帧内空间注意力 | `n` |

> [!warning] 不要照搬下载版中的错误
> HyVLA 这里不是 learned query token resampler。论文描述的是 temporal/spatial attention，并强调无新增可学习参数。

## 优势、代价与失败边界

**优势：** 上层 VLM token 数不随 K 增长；K=1 与单帧预训练兼容。

**代价：** 历史最终被压进当前帧 token，细粒度旧信息可能丢失；计算仍随 K 增长，只是不是最昂贵的全时空平方增长。

**边界：** 只在 HyVLA 当前实验中得到验证，不能直接推出任意视频模型都有效。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
