---
type: concept
topic: long-horizon-control
status: deep-explained
verification: pi05-pdf-checked
aliases: [hierarchical VLA inference, 高低层 VLA]
---

# 层级 VLA 推理：先缩短语义距离，再生成动作

出现于：[[π₀.5]] · 相关：[[Action Chunking]] · [[Action Expert]]

> [!summary] 一句话定义
> 层级 VLA 先把长时程总任务转成当前可执行的语义子任务，再让低层 policy 以该子任务为条件生成短时程动作，从而避免一步跨越“整理房间”到“关节该转多少度”的巨大抽象距离。

## 直觉理解

“收拾卧室”像项目目标，“拿起枕头”像当前工单，action chunk 像工人的几秒钟动作。项目目标不能直接告诉每个关节怎样移动；当前工单把搜索范围收窄到可见对象和局部技能。

## π₀.5 的概率分解

$$
\pi(a,\hat\ell\mid o,\ell)
=\pi(a\mid o,\hat\ell)\pi(\hat\ell\mid o,\ell).
$$

```python
def hierarchical_vla(observation, global_task):
    subtask = model.generate_text(observation, global_task)
    action_chunk = model.flow_action_expert(observation, subtask)
    return action_chunk
```

这里最重要的不是“多生成一句话”，而是这句话进入动作分布的条件。

## 为什么 human high-level 不一定最好？

高层命令存在接口分布：粒度太大，低层不会；粒度太小，频繁切换；用词或时机不在低层训练分布内，也可能失败。π₀.5 完整模型超过 human-HL baseline，合理解释是联合训练让高层命令与低层可执行分布匹配；它不是“AI 普遍比人更会规划”的证明。

## 显式高层与隐式高层

| 方式 | 运行时生成子任务？ | 训练见过 HL 数据？ | 含义 |
|---|---:|---:|---|
| explicit HL | 是 | 是 | 子任务既塑造表示，也直接条件化动作 |
| implicit HL | 否 | 是 | 高层监督只通过共享参数影响 policy |
| no HL | 否 | 否 | 直接从总任务到动作 |

π₀.5 中 implicit HL 是第二强的变体，说明 HL 数据本身能塑造共享表示；显式推理则在此基础上进一步缩短当前动作的条件距离。

## 它不等于什么？

- 不等于显式符号 planner：未维护任务树或形式化前置条件；
- 不等于两个完全分离的模型：π₀.5 在同一模型内共享 prefix 与 attention；
- 不等于长期记忆：只预测当前子任务仍可能重复、遗忘或受遮挡干扰；
- 不等于安全层：子任务合理不能保证轨迹无碰撞。

## 设计与评测清单

```python
checks = {
    "granularity": "子任务是否正好落在低层 policy 的技能范围？",
    "timing": "何时重算高层，何时继续当前 chunk？",
    "grounding": "子任务是否绑定到当前可见对象/位置？",
    "memory": "怎样避免重复与跨房间遗忘？",
    "recovery": "低层失败后，高层是否能重新分解？",
    "interface": "human/LLM 产生的语言是否落在训练分布？",
}
```

评测必须同时看子任务正确率、低层条件成功率、最终 task progress、循环/重复率和错误恢复，不能只评文本是否“听起来合理”。

