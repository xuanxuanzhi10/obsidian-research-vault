---
type: concept
topic: action-representation
status: checked-seed
---

# Action Chunking

一句话：一次预测未来 H 步动作，而不是每次只预测下一步。

## 为什么需要？

逐步预测容易受到推理延迟影响，也缺少明确的短期轨迹结构。一次预测一个 chunk 可以：

- 让高容量 backbone 不必为每个 servo step 完整运行一次
- 提供短期时间一致性
- 让执行线程在下一次推理完成前持续运动

## 它把问题转移到了哪里？

- chunk boundary discontinuity
- observation-to-action latency
- stale prefix

这些问题在 [[Hy-Embodied-0.5-VLA]] 中由异步 buffer 与 Bézier stitching 处理。

> [!question] chunk 越长越好吗？
> 不是。长 chunk 计算利用率更高，却更容易因环境变化而过时；短 chunk 更闭环，但推理压力更大。实际系统需要在 horizon、replanning rate 和延迟之间折中。

## 出现于

- [[π₀]]
- [[π₀.5]]
- [[Hy-Embodied-0.5-VLA]]
