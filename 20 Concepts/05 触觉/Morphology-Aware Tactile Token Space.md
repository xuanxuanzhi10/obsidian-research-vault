---
type: concept
title: Morphology-Aware Tactile Token Space
aliases: [MTTS, 形态感知触觉 Token 空间]
topics: [tactile, representation, cross-sensor]
---

# Morphology-Aware Tactile Token Space

> [!abstract] 一句话定义
> MTTS 不统一传感器的原始格式，而是把各传感器编码后的 token 对齐到共同的身体功能区槽位，让“哪里受力”成为跨硬件共享的语义。

![[ftp1-teaching-mtts.svg]]

## 为什么需要它？

同一接触可以来自 GelSight 图像、taxel 阵列或六轴 F/T 向量；只统一维度会丢掉安装位置，只记录传感器 ID 又难以迁移。MTTS 用“身体功能”作为中间坐标系。

```python
token = encoder_by_signal_type(raw_sensor)
slot = canonical_body_region(sensor.mount)
token = token + shared_region_embedding(slot)
```

## 在 FTP-1 中怎样实现？

- 24 槽：手部 `0–14`，腕/指力矩 `15–20`，保留 `21–23`；
- image → sensor-specific ViT + shared T3 Transformer；
- array → CNN；state → Fourier features + MLP；
- 同形状、同传感器的多功能区可共享 encoder；左右手 embedding 分开。

## 优势与局限

优势是新传感器只需接入前端，后续共享模块可复用。局限是槽位是人工定义的硬路由，依赖正确的安装/形态标注；不同接触面积、分辨率和标定误差不会因“同槽位”自动消失，也不是零样本硬件兼容。

出现于：[[FTP-1]] · 对比：[[触觉信号表示]] · [[触觉 VLA 方法比较]]

