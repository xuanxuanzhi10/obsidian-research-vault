---
type: concept
topic: data-collection
status: deep-explained
aliases: [Universal Manipulation Interface]
---

# UMI

> [!summary] 一句话
> 让人用接近自然操作的装置完成任务，同时记录可转换为机器人末端动作的视觉、位姿和夹爪信号。

![[90 Attachments/HyVLA/hyvla-umi-data.svg]]

## 它填补哪条数据鸿沟？

```text
普通人类视频：自然、便宜、规模大，但缺精确可执行动作
机器人遥操作：动作可执行，但昂贵、慢、受机器人形态限制
UMI：尽量保留人的自然性，同时获得 6DoF EEF trajectory
```

## 一个可用 UMI 数据管线需要什么？

1. 第一人称/腕部视觉；
2. 末端 6DoF 位姿；
3. gripper 开合状态；
4. 相机、位姿、夹爪的时间同步；
5. 坐标系标定；
6. 从人体装置轨迹到机器人 action representation 的转换；
7. 对不可达、碰撞、异常轨迹做过滤。

最难的往往不是“拍视频”，而是让第 t 帧视觉与第 t 个动作标签精确对应，并且能被不同机器人解释。

## HyVLA 版本的特点

- finger-attached gripper，保持人手接触与施力直觉；
- head-mounted ego RGB-D + 双腕视角；当前训练只用 RGB；
- 外部 optical mocap 给全局 6DoF，声称亚毫米精度；
- rotary encoder 记录夹爪开合；
- 部分夹具含 6D force/torque sensor；
- Hy-UMI-10K：10K+ h、1M+ episodes、70 tasks。

## 为什么 UMI 数据还不能无条件上机器人？

人的可达域、碰撞习惯、速度与动力学都不同。HyVLA 仍需 relative EEF 转换、IK feasibility filtering、reachable-shell 限制及 robot-specific controller。

## 精度与扩展性的交换

HyVLA 用 mocap cage 换高精度标签，但牺牲场地自由度和扩展便利。SLAM-only 方案更灵活，却可能在精细插接任务中产生累计漂移。论文没有消除这个 trade-off，只是选择了精度端。

## 不要误解触觉

finger-attached 设计让采集者本人有自然触觉，并不自动代表策略获得了触觉输入。部分夹具的 F/T sensing 也需确认是否进入当前训练数据字段。

## 出现于

- [[Hy-Embodied-0.5-VLA]]
