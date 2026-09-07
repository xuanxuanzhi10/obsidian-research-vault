---
type: concept
topic: action-generation
status: deep-explained
aliases: [流匹配]
---

# Flow Matching

> [!summary] 一句话
> 学习一个随时间变化的速度场，把简单噪声分布沿连续路径运输成目标数据分布。

![[90 Attachments/HyVLA/hyvla-flow-and-deploy.svg]]

## 为什么动作不直接做 MSE？

同一场景可能存在多条正确轨迹。单次 MSE 倾向于学条件均值；两条绕障路径的平均可能恰好穿过障碍。生成式策略要表达的是 `p(action | observation, language, state)`，而不只是均值。

## 最小数学例子

真实标量动作 `A=3`，噪声 `ε=-1`：

```math
A_τ = τA+(1-τ)ε = -1+4τ
```

当 `τ=0.25`，中间状态是 0；沿路径的恒定速度为 `A-ε=4`。网络训练时看到中间点、条件和 τ，学习该往哪个方向走。

多维 action chunk 只是把标量换成 `[H,D]` tensor：

```text
condition: image/language/state
A, ε, Aτ: [batch,H,D]
τ: [batch,1,1]，广播到整个 chunk
vθ output: [batch,H,D]
```

## 训练与推理

```python
# training
A_tau = tau * A + (1-tau) * eps
target = A - eps
loss = mse(v_theta(condition, A_tau, tau), target)

# inference
a = gaussian_noise()
for tau in solver_grid:
    a = a + delta_tau * v_theta(condition, a, tau)
```

[[Hy-Embodied-0.5-VLA]] 使用 10 个 Euler steps、`Δτ=0.1`。

## 它和 diffusion 的关系

两者都从简单分布生成数据，也可能多步迭代；Flow Matching 直接回归概率路径的速度场，常见 diffusion 则从前向加噪过程推导噪声、score 或其他参数化目标。不要只因“多步去噪”就说完全相同。

## HyVLA 的符号警报

Sec. 2.3 Eq. (3) 写 `ε-A`，但它给出的插值式从噪声 `τ=0` 到数据 `τ=1`，求导是 `A-ε`；Sec. 4.2 也使用 `u=a-ε`。复现应查官方代码或勘误。

## 常见误解

- 多峰能力不等于每次采样一定给出语义上不同的模式；还取决于条件、数据与训练。
- solver steps 越多通常越精细，但延迟也越大。
- 一次生成 chunk 不等于一次动作；chunk 还要经执行与重规划机制消费。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
- [[π₀]]
- [[π₀.5]]
