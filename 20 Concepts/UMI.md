---
type: concept
topic: data-collection
status: paper-specific-seed
---

# UMI

一句话：用手持或穿戴式装置记录人类操作，把自然的人类示范转成可供机器人学习的视觉—动作数据。

## 为什么需要？

传统 teleoperation 受机器人工作空间和操纵界面限制，采集速度慢；直接观察人类视频又缺少精确、可执行的末端动作标签。

UMI 类方案尝试同时获得自然操作行为与 6-DoF end-effector trajectory。

## 真正困难的部分

- 位姿追踪与视觉时间同步
- 坐标系标定
- gripper 状态与接触信息
- 人手轨迹到具体机器人可达动作的映射

> [!warning] 不要一概而论“没有触觉”
> 不同 UMI 实现不同。[[Hy-Embodied-0.5-VLA]] 的 finger-attached 采集机构强调保留人的接触感受，部分夹爪还集成了末端 6D 力/力矩传感器。

