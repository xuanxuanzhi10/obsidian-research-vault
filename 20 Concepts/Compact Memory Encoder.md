---
type: concept
topic: temporal-modeling
status: seed
---

# Compact Memory Encoder

一句话：在 image encoder 内把历史帧信息压入当前帧 tokens，而不是把所有历史 tokens 交给 VLM。

## 因果链

多帧很重要 → 直接拼接会让 token 数乘 K → 在 ViT 内做 temporal/spatial factorization → 只输出 current-frame tokens。

## 出现于

- [[Hy-Embodied-0.5-VLA]]

