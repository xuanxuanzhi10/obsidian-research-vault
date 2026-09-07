---
type: concept
topic: action-representation
status: deep-explained
aliases: [相对末端执行器动作, Relative End-Effector Action]
---

# Relative EEF Action

> [!summary] 一句话
> 在末端执行器坐标中描述相对位姿变化，把“末端要怎样运动”和“具体机器人的关节怎样实现”分开。

![[90 Attachments/HyVLA/hyvla-relative-eef.svg]]

## 为什么不用 joint action？

同一个空间目标，在 6-DoF 机械臂、双臂机器人和 humanoid 上对应完全不同的关节维度与数值。joint action 把任务意图和 embodiment kinematics 绑死，难以复用人类示范。

Relative EEF 先统一上层接口：

```text
policy: observation → 末端相对目标
robot-specific mapper: 相对目标 → 当前坐标系的绝对目标
IK/controller: 绝对目标 → joints / torques
```

## HyVLA 的具体编码

每只手每步 10D：位置 3D、旋转 6D continuous representation、夹爪 1D；双臂共 20D。整个 chunk 的形状是 `[H,20]`，所有增量相对 chunk 起点末端位姿。

6D rotation 不是 6 个独立旋转角，而是以旋转矩阵前两行/列形成的连续表示，随后正交化恢复 SO(3)，避免欧拉角不连续。

## 为什么用 chunk 起点作参考？

它让一整段轨迹处于一致局部坐标中，减少全局工作空间差异；但长 chunk 的相对误差可能累计，部署端仍需频繁重规划与状态更新。

## 它解决与不解决的边界

| 解决 | 不解决 |
|---|---|
| 不同 joint topology | reachable workspace |
| 动作维度统一 | IK singularity / 无解 |
| UMI → robot 标签接口 | 碰撞与动力学差异 |
| 局部坐标泛化 | humanoid torso/head 协调 |

HyVLA 对 JAKA 做 IK feasibility filter，对 humanoid 限制 reachable shell，并用启发式生成额外 24D torso/head 控制。因此“action space 统一”不能被夸大成“任意机器人直接通用”。

## 与 Action Chunking 的关系

Relative EEF 定义每一步“表示什么”；[[Action Chunking]] 定义一次预测多少步。二者是坐标表示与时间打包两个不同轴。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
