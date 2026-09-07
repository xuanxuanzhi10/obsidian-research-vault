---
type: concept
topic: parameter-efficient-finetuning
status: checked-seed
---

# LoRA

一句话：冻结原权重，只学习一个低秩增量，以较少显存和可训练参数适配大模型。

## 工作机制

对原线性层 `W`，不直接更新整个矩阵，而使用：

```text
W' = W + B A
```

其中秩 `r` 远小于输入、输出维度。

## 在 VLA 中为什么有用？

完整微调数十亿参数的 VLA 成本高；LoRA 让单任务、单机器人适配更容易。但节省训练参数不自动等于推理更快，也不保证跨 embodiment 泛化。

## 出现于

- [[OpenVLA]] 的高效微调实验

