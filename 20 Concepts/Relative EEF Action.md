---
type: concept
topic: action-representation
status: seed
---

# Relative EEF Action

一句话：在 end-effector frame 中描述相对位姿变化，把任务运动意图与具体机器人 joint kinematics 分离。

## 优势

- 减少 embodiment-specific 参数化
- 适合 [[Action Chunking]]
- 便于 UMI-to-robot transfer

## 不能解决

- reachability
- self-collision
- IK instability
- humanoid torso/head inference

## 出现于

- [[Hy-Embodied-0.5-VLA]]

