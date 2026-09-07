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

## 因果链

机器人要判断速度、遮挡和操作阶段 → 单帧不够 → 直接拼 K 帧让 VLM token 数乘 K → 将时空注意力分解并尽早压缩 → 当前帧 token 带着历史进入 VLM。

## 张量怎样变化？

设每帧 `n` 个 patch、隐藏维 `d`：

```text
X: [K,n,d]
transpose conceptual axes → 对每个 patch p 得到 [K,d]
causal temporal attention → [K,n,d]
每帧 spatial attention   → [K,n,d]
只取当前帧               → [n,d]
```

“同位置跨时间”先回答这个局部区域发生了什么，“同帧跨空间”再把手、物体和背景组合起来。

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

## 出现于

- [[Hy-Embodied-0.5-VLA]]

