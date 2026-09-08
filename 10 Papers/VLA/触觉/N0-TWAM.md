---
type: paper
title: "N0-TWAM: Tactile World-Action Model"
year: 2026
status: read
verification: full-paper-and-appendix-checked
topics: [VLA, WAM, tactile, future-prediction, MoT, flow-matching]
source: "[[N0-TWAM.pdf]]"
---

# N0-TWAM：既预见下一次接触，也读取此刻接触

上级地图：[[VLA 学习地图]] · 主题：[[触觉机器人学习]]  
核心概念：[[预测触觉与观测触觉]] · [[触觉标点]] · [[Mixture of Transformers]]  
对比入口：[[触觉 VLA 方法比较]]

原文：[[N0-TWAM.pdf|论文 PDF]] · [[n0-twam-study-guide.html|HTML 原版讲解（系统浏览器打开）]]

> [!abstract] 一句话结论
> N0-TWAM 把触觉拆成两个不同职责：生成式分支预测未来接触以提前避错，观测分支读取当前力场以快速纠偏；Action Expert 在因果级联中同时使用两者。

## 先看原论文 pipeline

![[n0-twam-fig2-overview.png]]

```text
语言 + 历史视频 + 历史触觉
  → Video Expert ↔ Tactile Expert：先联合去噪未来画面与未来接触
  → 固定预测侧 KV
  → Action Expert：读取刚预测的未来 + 当前观测触觉
  → 去噪 action chunk 并执行
  → 真实新画面/触觉替换预测，滚动到下一 chunk
```

## 为什么“只把当前触觉喂给策略”还不够？

当前触觉只能在接触发生后告诉机器人滑了、卡了或压重了。若模型能先预测“这条动作会产生怎样的接触变化”，就可以在错误真正发生前改变动作。反过来，预测总会有误差，所以仍需真实当前触觉纠偏。两条路径解决不同时间方向的问题：

| 路径 | 问题 | 作用 |
|---|---|---|
| predicted tactile | 下一 chunk 会摸到什么？ | 前瞻、规划、避免不合理接触 |
| observed tactile | 这一刻实际摸到什么？ | 反射、恢复、修正模型误差 |

## 三 Expert 如何实现“先预测，再行动”？

![[n0-twam-teaching-cascade.svg]]

模型约 7.16B：Video Expert 约 5.00B、宽度 3072；Action/Tactile Expert 各约 1.13B/1.03B、宽度 1024。每层只有 self-attention 空间共享，FFN、归一化和残差流按模态分开。

```python
def generate_one_chunk(history, prompt, observed_touch):
    # Phase A：未来视觉和未来触觉同一步表内共同去噪
    v = noise_video_latents()
    t = noise_tactile_latents()
    for sigma in prediction_schedule:
        v, t = shared_attention_mot.denoise_prediction(
            v, t, committed_history=history, text=prompt
        )

    # 因果约束：v/t 不读当前 action；所以预测完成后 KV 不再变化
    frozen_future_kv = cache(v, t)

    # Phase B：动作读取已预测未来 + 当前真实触觉
    a = noise_action_chunk(horizon=24, dim=20)
    obs_tokens = observed_tactile_encoder(observed_touch)
    for sigma in shorter_action_schedule:
        a = action_expert.denoise(a, attend_to=[frozen_future_kv, obs_tokens])
    return a, v, t
```

这不是“先跑完整世界模型，再跑一个独立 policy”的两套模型。三种 token 在同一 MoT 注意力中交互，但用 attention mask 固定信息方向。预测端不看当前 action，因此 action 多步去噪时可以缓存 5B 视频分支的 KV，只重复运行较窄的 Action Expert。

## 未来触觉具体预测什么？

视觉与触觉都经同一个冻结 causal video VAE 编成 latent；触觉被当作小视频。模型不直接预测绝对触觉，而预测相对 chunk 初始触觉的残差：

$$\Delta t_i=x^t_i-x^t_0,\qquad x^t_i=x^t_0+\Delta t_i.$$

远离接触时触觉近似常量，真正有信息的是接触开始、释放和滑移处的变化；残差目标因此减少“全是静态背景”的学习负担。视频与触觉共享同一 frame position 和噪声步表，在同帧互相注意，迫使“画面中的接触结果”和“摸到的接触结果”保持一致。

## 当前观测触觉为什么不用同一种表示？

![[n0-twam-teaching-two-paths.svg]]

预测分支必须生成，所以使用与视频一致的 VAE latent；观测分支只需读懂真实接触，因此在真机上使用物理意义更直接的 force space：

```python
class ObservedTouchPort:
    def forward(self, raw_tactile_image, action_hidden):
        # 标定好的 sensor→physics 转换器冻结
        force = frozen_estimator(raw_tactile_image) # [fx, fy, fz] + contact mask
        obs = neoforce_encoder(force)                # ViT-B，post-train 时微调

        # 输出投影零初始化：刚接入时严格等于原策略，随后再学会依赖触觉
        correction = cross_attention(action_hidden, obs)
        return action_hidden + zero_init_out(correction)
```

真机 `fx,fy` 表示剪切，`fz` 表示法向压力；平行夹爪两侧合计六个力通道并带 contact mask。仿真传感器不在 NeoForce 域内，因此不用伪装成物理力场，而训练轻量 latent encoder 接入同一个零初始化端口。

## 为什么训练分两阶段？

预训练只启用 predicted pathway；如果一开始就把真实当前触觉给模型，它可能走捷径，只依赖事后信号而不学习预测未来接触。后训练才打开 observed pathway。

```python
def pretrain(batch):
    # 约 30,000+ h，六种 embodiment、450 个接触任务
    # video / predicted tactile / action 的 flow loss 权重 1:1:1
    return flow_loss(video=True, future_touch=True, action=True,
                     observed_touch=False)

def posttrain(task_demo):
    # horizon=24，约每个 latent frame 对齐 12 个 action step
    # 同时保留未来触觉目标，并让当前触觉 cross-attend 到 action
    return flow_loss(video=True, future_touch=True, action=True,
                     observed_touch=True)
```

预训练：Wan2.2-TI2V-5B 系视频 expert 从 LingBot-VA 热启动，action/tactile expert 从零训练；30k steps、batch 512、128×H800。真实 NeoForce encoder 另用 20k demonstrations、30 tasks 训练 100k steps。语言和触觉条件各以 0.1 概率 dropout；触觉 dropout 是彻底移除 token，使无触觉数据、传感器故障和 vision-only fallback 使用同一接口。

## “触觉标点”怎样帮助长任务？

![[n0-twam-teaching-punctuation.svg]]

长任务的相邻阶段往往视觉相似，而接触开始/释放是天然边界。训练时用触觉变化为主、夹爪开度为后备切分 demonstrations，再给每段短指令；推理时 scheduler 维护子任务队列：预测触觉提前触发候选切换，观测触觉确认后再提交。

```python
if predicted_touch.detect(next_contact_event):
    pending_transition = True
if pending_transition and observed_touch.confirms(contact_or_release):
    prompt = subtask_queue.pop_next()
    reseed_streaming_context(prompt)
```

这部分是显式外部 scheduler，不应误解为 7B 模型自己端到端生成完整任务计划。

## 数据和动作怎样对齐？

- NeoData：超过 30,000 小时、六种 embodiment、450 个接触任务；大量 episode 有同步多视角 RGB 与逐指触觉。
- 每个窗口 33 个 latent frames；原始 30 fps 的 387 帧先降到 10 fps，再经 VAE 4× 时序压缩；窗口 stride 24，约 27% 重叠。
- 统一 20D 双臂 EEF 动作，每臂 10D；位置与 6D rotation 用 chunk-anchor delta，gripper 保持 absolute；单臂缺失维度 mask。

## 实验真正证明了什么？

- UniVTAC 8 任务平均 `84.5%`，优于表内各 VLA/WAM 基线；
- NeoSim 12 任务平均 `49.4%`，π0.5 为 `45.8%`，优势较小；
- 8 个真机接触任务平均 `46.3%`，π0.5 `30.0%`，每任务 20 次；论文也提醒单任务二项标准误可达约 11%，应看跨任务一致性而非夸大单列差距；
- 消融：去掉 predicted pathway 后 UniVTAC/NeoSim 为 `71.8/41.1`；去掉 observed pathway 为 `70.5/29.6`，完整模型为 `84.5/49.4`。两条路径都有效，NeoSim 更依赖当前观测；
- 仅用 20% 预训练数据时 UniVTAC 从 `84.5` 降到 `65.4`，说明规模是重要因素，但该变体只在 UniVTAC 报告。

## 读者容易误解的地方

> [!warning] 证据边界
> “预测触觉有效”由消融支持；但大规模 NeoData 与 NeoForce 是同一研究体系，尚不能据此断言跨任意触觉硬件都能复现。论文也没有证明预测触觉可以替代实时观测。

- 仿真 observed encoder 与真机 NeoForce 不同，不能把全部结果归因于统一物理力表示；
- 模型规模约 7B，实时性依赖 KV cache、异步执行和硬件，架构可行不等于部署廉价；
- delta EEF 并非总是最好：论文对齐任务中 absolute 平均 `82.5%`，delta `50.0%`；
- 长任务“触觉标点”含事件检测、子任务文本与 scheduler，人为结构贡献不能忽略。

## 对 HyVLA 的组合建议

最小可行路线不是立即复制整个 7B WAM，而是先验证一条附加预测头：

```text
HyVLA 主干 + 当前触觉条件
  → 加 future tactile residual prediction auxiliary loss
  → Action Expert cross-attend predicted tactile tokens
  → 再对比 current-only / predicted-only / both
```

若后续需要跨硬件，用 FTP-1 的 MTTS；需要更快纠偏，用 T-Rex 式异步触觉 expert；需要提前推断接触后果，再逐步引入 N0-TWAM 的预测目标与因果级联。

