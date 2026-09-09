---
type: concept
title: ALTER
topics: [offline-RL, advantage-conditioning, progress-model, tactile]
---

# ALTER

全称：Advantage Labeling from Trajectory Events and Relative Progress  
来源：[[N0-VTLA]] · 相关：[[FlowPRO]]

> [!abstract] 一句话定义
> ALTER 用干净示范的阶段进度和失败/HIL 事件训练成对进度模型，再把固定部署数据中的每个片段标成阶段内 positive 或 negative，作为文字条件训练原策略。

## 第一层：为什么需要它？

部署轨迹不是“成功全好、失败全坏”。一次失败 rollout 前半段可能正确，HIL rollout 中还包含宝贵的恢复动作。若只保留成功 episode，会丢数据；若全部行为克隆，又会把退步动作一起学进去。

ALTER 不问“这一帧绝对值多少分”，而问两个更容易的问题：当前在任务哪个阶段？从当前到一个 action chunk 后，在这个阶段内是在进步还是退步？

## 第二层：工作机制

![[n0-vtla-teaching-alter.svg]]

```python
# 1. 用两类 pair 训练 Aθ(xa, xb)
dense_pairs = clean_demo_stage_progress_pairs()
event_pairs = object_drop_pairs() + hil_correction_pairs()
A_theta = train_pairwise_progress(dense_pairs, event_pairs)
freeze(A_theta)

# 2. 在每条离线轨迹上打标签
global_phase = A_theta(current, episode_start) + recovery_offset
local_change = A_theta(frame_t_plus_H, current)
stage = assign_duration_calibrated_stage(global_phase)
positive = local_change >= percentile(local_change_in_same_stage, 70)

# 3. 不改策略结构，只把标签追加给指令
prompt = task_prompt + ("Advantage: positive" if positive else "Advantage: negative")
policy.train_with_original_flow_matching_loss(prompt, observation, action)
```

必须在同阶段比较，因为“放下物体”的位移大小不能和“精细插入”的位移大小直接排序。部署时始终提示 `Advantage: positive`，等价于要求条件策略从离线数据中选择更像阶段内高进展样本的行为。

## 第三层：与相近方法区别

- 与成功轨迹过滤：ALTER 保留失败与恢复轨迹中的局部好片段；
- 与普通 reward model：它学两帧相对进度，并转成二值文字条件，不直接把标量奖励送入在线优化；
- 与在线 RL：使用固定 deployment corpus，不再进行环境交互；
- 与 χ0 Stage Advantage：ALTER 用触觉掉落、HIL 与不等时长阶段构造相对监督，而不是依赖均匀阶段间隔；
- 与 [[FlowPRO]]：FlowPRO 关注 winning/losing action pairs 对 flow policy 的直接优化；ALTER 先标注 observation transition，再保持原策略目标做条件学习。

## 第四层：优势、局限与证据

N0-VTLA 报告 ALTER 在 Towel/Bag/Cardboard 上把 π0.5 提到 `90/75/60%`，配合 N0-VTLA 为 `95/80/75%`，均优于相应 SFT；进度曲线也会在掉落时下降、恢复后回升。

局限：

- 每个任务要建立并人工审核一次阶段模板，progress model 也是 task-specific；
- top 30% 是数据集内部的相对定义，不保证绝对安全或最优；
- 触觉事件参与造标签，却不进入 progress model，纯视觉无法辨认的失败可能仍被漏判；
- 标签质量依赖边界检测、VLM 区间映射与 recovery offset，误差会级联；
- 三个任务上的成功不能证明其对任意长任务或跨任务 reward 泛化。

