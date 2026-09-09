---
type: topic
aliases: [VLA 经典论文图鉴, VLA 主干论文粗读]
created: 2026-09-09
checked: 2026-09-09
status: rough-reading-atlas
topics: [VLA, robot-learning, classics, paper-atlas]
verification: primary-pdf-navigation-checked
---

# VLA 经典论文粗读图鉴：每篇究竟改变了哪一层？

上级地图：[[VLA 全景综述：技术脉络、前沿与研究地图]] · 后续：[[VLA 前沿论文粗读雷达（2025-2026）]]

> [!abstract] 这不是 17 个模型名字
> 这些工作分别改写了五层：**数据怎样来、观测怎样表征、动作怎样生成、VLM 知识怎样进入控制、策略怎样实时执行**。所谓“经典”，不是今天仍在所有榜单第一，而是它提出的接口已经变成后续论文绕不开的比较坐标。

> [!warning] 核验口径
> 本页是基于原论文全文、method figure、abstract 与结论做的**粗读图鉴**，不是逐表复现。图片均来自原论文；精确训练配置与所有实验数字仍应进入单篇精读后再引用。

## 0. 一眼选择阅读路线

| 你现在想解决的问题 | 优先顺序 |
|---|---|
| 为什么一次生成一段动作？ | ACT → Diffusion Policy → π₀ → OpenVLA-OFT |
| 网页/VLM 知识怎样进入控制？ | RT-1 → RT-2 → OpenVLA → π₀.5 |
| 跨机器人数据如何成立？ | Open X-Embodiment → Octo / RDT-1B |
| 真机数据为什么不只看小时数？ | DROID → UMI |
| 3D 对控制有什么直接价值？ | DP3 → iDP3 |
| 高频动作如何进入 token 模型？ | OpenVLA → FAST → π₀.5 |

---

## 1. RT-1：把“规模化真机多任务 policy”变成可信研究对象

![[90 Attachments/VLA-Paper-Atlas/rt1-pipeline.png]]

*RT-1 Figure 3 · PDF 第 6 页 · 原论文架构图。*

**Pipeline**：语言指令先变成 embedding；连续图像由 EfficientNet 编码并用 FiLM 接收语言条件；TokenLearner 压缩视觉 token；decoder-only Transformer 输出离散化动作 token。

**为什么经典**：它没有借助后来的巨型 VLM，而是先证明了一个更基础的命题——同一套 real-robot policy 可以随数据量、任务多样性和模型容量一起扩展，并且仍满足真机控制频率。后续 VLA 都必须回答“语义更强后，是否牺牲了 RT-1 式的控制效率”。

**值得关注的洞察**：TokenLearner 不是小工程细节。真实控制的约束不是只把 loss 降低，而是把多帧视觉压到足够少的 token，使 policy 能按时给动作。它把 architecture 与 deployment 放在同一个目标函数外共同设计。

**不要误读**：RT-1 主要证明多任务 robot data 的扩展性，不等于互联网知识已经被迁移进控制；这一步由 RT-2 接上。[原文](https://arxiv.org/abs/2212.06817)

## 2. RT-2：第一次把机器人动作正式当成 VLM 的另一种“语言”

![[90 Attachments/VLA-Paper-Atlas/rt2-pipeline.png]]

*RT-2 Figure 1 · PDF 第 2 页 · 原论文总览图。*

**Pipeline**：互联网 image-text data 与 robot trajectories 共同 fine-tune VLM；机器人动作被序列化成文本 token；推理时图像与指令进入同一模型，token 再反序列化为 closed-loop robot action。

**为什么经典**：RT-2 给出了 “Vision-Language-Action model” 最有影响力的早期范式：不另建完全独立的 planner，而是把网页语义知识与动作监督放入一个自回归接口。它让“认识垃圾、数字、图标”有机会改变抓哪个物体。

**真正的洞察**：迁移的是**选择和语义 grounding**，不是凭网页图片学会新的接触动力学。VLM 可以告诉机器人石头能当锤子，但敲击力度、轨迹与失败恢复仍来自 robot data 和控制闭环。

**不要误读**：动作 token 像文字只是计算格式统一；它们没有天然语言含义。co-fine-tuning 还承担了保护互联网知识的作用，不能只比较参数量。[原文](https://arxiv.org/abs/2307.15818)

## 3. ACT / ALOHA：先修复时间结构，再谈大模型

![[90 Attachments/VLA-Paper-Atlas/act-pipeline.png]]

*ACT Figure 4 · PDF 第 4 页 · 原论文架构图。*

**Pipeline**：多视角图像、关节状态和一段示范动作进入 CVAE；Transformer encoder 学 latent style，decoder 一次预测未来 action chunk；部署时对重叠预测做 temporal ensemble。

**为什么经典**：ACT 把模仿学习中两个顽固问题放在一起解决：人类 demonstration 有多种风格，单步 BC 又会不断累积误差。chunk 把高频动作的局部结构一次建模，temporal ensemble 则减轻相邻 chunk 接缝。

**值得关注的洞察**：Action Chunking 首先是**时间抽象**，其次才是加速。它把策略从“每帧猜下一步”改成“当前证据下承诺一小段轨迹”。后来 π₀、OFT、RTC 都在处理这个承诺应该多长、何时允许重写。

**边界**：chunk 越长，推理次数越少，但环境新变化进入动作的时间越晚；平滑接缝也不等于看到了新的触觉或视觉反馈。[原文](https://arxiv.org/abs/2304.13705)

## 4. Diffusion Policy：把动作平均值换成动作分布

![[90 Attachments/VLA-Paper-Atlas/diffusion-policy-pipeline.png]]

*Diffusion Policy Figure 2 · PDF 第 3 页 · 原论文总览图。*

**Pipeline**：最近若干观测条件化一个 diffusion model；从 noisy action sequence 开始多步去噪，生成预测时域内的动作轨迹；只执行前一段，再用新观测滚动重规划。

**为什么经典**：传统 MSE regression 在“两条路都合理”时容易输出两者平均，而平均轨迹可能撞上障碍。Diffusion Policy 把 policy 看成条件生成分布，能表达多峰、高维、时间相关的动作。

**值得关注的洞察**：它的强点不是“噪声使探索更强”，而是**训练目标与多模态动作分布更匹配**；receding horizon 又把开放环生成重新接回闭环。

**边界**：多次 denoising 会占据真实延迟预算；采样出一条看似连贯的轨迹，不代表满足碰撞、力或动力学约束。它应作为强 visuomotor baseline，而不是被 VLA 名字自动淘汰。[原文](https://arxiv.org/abs/2303.04137)

## 5. Open X-Embodiment / RT-X：经典贡献首先是“公共接口”

![[90 Attachments/VLA-Paper-Atlas/oxe-pipeline.png]]

*Open X-Embodiment Figure 1 · PDF 第 1 页 · 原论文数据总览图。*

**Pipeline**：不同实验室的机器人、相机、任务和 action schema 被转换为统一数据格式；在跨 embodiment mixture 上训练 RT-1-X / RT-2-X；再回到多个机器人评测迁移。

**为什么经典**：它把多机构、异构机器人数据从“各自保存的视频”提升为可联合训练的公共研究对象。后续 OpenVLA、Octo、π 系列与大量开源 VLA 都建立在这一基础设施上。

**真正的洞察**：跨机器人学习的瓶颈不是文件能否拼起来，而是 action semantics 是否可比、采样权重是否合理、相机和语言标签是否一致。统一 RLDS 容器只解决第一步。

**边界**：正迁移是统计结果，不意味着每个源机器人都帮助每个目标机器人；dataset 数量也不能代替有效状态—动作覆盖。[原文](https://arxiv.org/abs/2310.08864)

## 6. DROID：数据规模之外，独立环境数本身是一条泛化轴

![[90 Attachments/VLA-Paper-Atlas/droid-pipeline.png]]

*DROID Figure 1 · PDF 第 1 页 · 原论文数据采集总览图。*

**Pipeline**：标准化 Franka workstation 被分发到多地，由多位采集者在大量真实场景中遥操作；记录多相机、状态、动作与语言元数据；最终用于预训练和下游 generalization 测试。

**为什么经典**：早期大数据往往仍来自少数实验室布景。DROID 把“in-the-wild”具体化为采集者、房间、背景、物体和任务的联合变化，并公开硬件复现方案。

**值得关注的洞察**：350 小时在同一桌面与 350 小时分布在数百场景，不是同一份训练信号。环境熵、相机放置和操作者差异，可能比单纯 trajectory count 更决定 OOD 表现。

**边界**：更多自然场景也会带来动作标签噪声与长尾不均衡。OpenVLA 训练时曾因 DROID action-token 学习较慢而调整 mixture，说明“多样”与“易学”并不自动一致。[原文](https://arxiv.org/abs/2403.12945)

## 7. UMI：采集工具不是前处理，而是在定义 policy

![[90 Attachments/VLA-Paper-Atlas/umi-pipeline.png]]

*UMI Figure 5 · PDF 第 5 页 · 原论文 policy interface 图。*

**Pipeline**：手持 gripper 的 GoPro 与 SLAM/标定恢复相对轨迹；训练时将图像、相对 EEF pose 与动作时间对齐后交给 Diffusion Policy；部署时显式补偿 observation 和 execution latency，再映射到机器人。

**为什么经典**：UMI 让人在真实环境中直接采集高质量操作，而无需把机器人搬到每个现场。更重要的是，它把 relative trajectory、镜头设计和 latency matching 一起定义为可迁移接口。

**真正的洞察**：human demonstration 能转移，不是因为“手和机器人都能抓”，而是因为表示层主动消除了部分硬件差异，并在部署时重建同样的时间语义。

**边界**：portable 不等于零标定；SLAM 误差、双手同步、夹爪状态、IK 可达性与相机延迟都会进入 action label。对 HyVLA，UMI 的价值首先是数据接口，而不只是更多小时数。[原文](https://arxiv.org/abs/2402.10329)

## 8. Octo：通用 robot policy 不一定要从超大 VLM 开始

![[90 Attachments/VLA-Paper-Atlas/octo-pipeline.png]]

*Octo Figure 2 · PDF 第 3 页 · 原论文架构图。*

**Pipeline**：语言或 goal image 被 token 化为 task tokens，多种 observation 进入对应 tokenizer；Octo Transformer 融合条件；可替换 action head 输出动作。新传感器或 action space 可通过新 tokenizer/head 适配，而不必重做整个 backbone。

**为什么经典**：它把“通用策略”拆成一个更工程化的问题：怎样让预训练 backbone 在新机器人上快速接入不同观测与动作接口。其开放代码、checkpoint 和系统消融让它长期是有价值的 baseline。

**值得关注的洞察**：泛化不只有 zero-shot。一个模型能否在消费级 GPU 上几小时内适配新 action space，本身就是 foundation policy 的核心能力。

**边界**：模块可插拔不代表新模态天然对齐；新 tokenizer/head 仍需要目标域数据。Octo 的语义能力也不应与更大 Internet-VLM 初始化的模型直接等同。[原文](https://arxiv.org/abs/2405.12213)

## 9. DP3：当任务主要缺几何时，3D bias 比更大语义模型更直接

![[90 Attachments/VLA-Paper-Atlas/dp3-pipeline.png]]

*DP3 Figure 2 · PDF 第 4 页 · 原论文方法图。*

**Pipeline**：RGB-D 转为稀疏 point cloud；紧凑 3D encoder 提取几何 feature；与 robot state、diffusion timestep 一起条件化 action diffusion；滚动生成 action chunk 并闭环执行。

**为什么经典**：DP3 用非常清晰的控制变量表明，机器人 manipulation 的 visual representation 不必永远围绕 2D image backbone 展开。少量 3D 点就能提供相机视角和空间位置上的有效归纳偏置。

**真正的洞察**：它不是“点云信息更多”，而是让 policy 更容易学习与相机外观变化无关的几何关系。对有 3D vision 背景的研究者，这是一条比从头训练大 VLA 更可验证的入口。

**边界**：收益依赖 depth 质量、裁剪、坐标系和 point sampling。若 baseline 使用的信息量、augmentation 或 encoder capacity 不同，就不能把全部增益归因于 3D。[原文](https://arxiv.org/abs/2403.03954)

## 10. iDP3：把 3D policy 推到 humanoid egocentric deployment

![[90 Attachments/VLA-Paper-Atlas/idp3-pipeline.png]]

*iDP3 Figure 2 · PDF 第 4 页 · 原论文系统总览图。*

**Pipeline**：Vision Pro 遥操作 humanoid 收集 demonstrations；头部 LiDAR/depth 形成 egocentric 3D observation；iDP3 学 action chunks；最终在同一 humanoid 上部署到新场景。

**为什么值得放入主干**：它不是另一个更大的 diffusion model，而是把 DP3 的 3D 表征放进最难的系统约束之一：头部视角不断移动、全身/双手 action 维度高、机载算力与标定都有限。

**值得关注的洞察**：egocentric 3D 表示的价值在于减少外参依赖，而不是完全消灭几何误差；相机随头运动后，时序对齐和自遮挡反而更重要。

**边界**：论文中的“无需 camera calibration/segmentation”是相对特定 pipeline 的工程简化，不应外推为任意机器人即插即用。[原文](https://arxiv.org/abs/2410.10803)

## 11. OpenVLA：开放 VLM-action baseline 的坐标原点

![[90 Attachments/OpenVLA/openvla-fig2-architecture.png]]

*OpenVLA Figure 2 · PDF 第 4 页 · 原论文架构图。*

**Pipeline**：SigLIP 与 DINOv2 feature 沿 channel 融合，经 projector 进入 Llama 2；7D 单步动作逐维分成 256 bins 并映射到 LLM tokens；模型自回归生成后反量化执行。

**为什么经典**：RT-2 证明路线，OpenVLA 则把代码、权重、数据处理和 fine-tuning recipe 变成社区能真正修改的基线。今天很多论文仍以“在 OpenVLA 上换动作头/视觉监督/训练配方”定义贡献。

**真正的洞察**：它最重要的不只是 7B，而是把 VLA 研究的接口暴露出来；代价是单步离散动作、无 proprioception/history、推理频率直接限制闭环。

**边界**：逐维分箱不等于概率维度独立；4-bit 结果也不等于普遍更准。详见 [[OpenVLA]]。[原文](https://arxiv.org/abs/2406.09246)

## 12. RDT-1B：双臂 foundation policy 的难点是动作语义，不只是参数量

![[90 Attachments/VLA-Paper-Atlas/rdt1b-pipeline.png]]

*RDT-1B Figure 3 · PDF 第 4 页 · 原论文架构图。*

**Pipeline**：不同机器人的 heterogeneous action 被映射进 Physically Interpretable Unified Action Space；图像、语言、本体输入条件化 diffusion Transformer；模型生成高维双臂 action chunk，再映射回目标机器人。

**为什么经典**：它把 diffusion foundation model 扩到 1B 量级和多机器人双臂数据，同时明确承认跨 embodiment 的核心难题是 action representation。如果没有物理含义一致的接口，参数再大也只是在拟合混合编号。

**值得关注的洞察**：统一动作空间是一种人为 inductive bias：它告诉模型哪些维度在机器人之间可共享。真正应消融的是统一表示、数据规模与模型规模各自贡献。

**边界**：高维统一向量会产生大量不存在或无效的 DoF；mask、归一化和目标机器人映射决定所谓“跨形态”到底学到了什么。[原文](https://arxiv.org/abs/2410.07864)

## 13. π₀：语义主干与连续动作专家开始明确分工

![[90 Attachments/Pi0/pi0-fig3-framework.png]]

*π₀ Figure 3 · PDF 第 4 页 · 原论文总览图。*

**Pipeline**：图像/语言由预训练 VLM 处理，robot state、flow timestep 与 noisy action chunk 进入较小 Action Expert；两者通过联合 attention 交换信息；多步 flow integration 输出连续 action chunk。

**为什么经典**：它改变了 RT-2/OpenVLA 的单一 token 接口：语义知识仍由 VLM 提供，但高频连续控制交给专门参数路径。模型容量、输出表示和采样计算因此可以分开扩展。

**真正的洞察**：Action Expert 不是简单外挂 MLP；attention mask 与 KV cache 决定固定 observation 能否在多次 denoising 中复用。架构设计同时服务表示能力与推理成本。

**边界**：Flow 生成连续 chunk 不代表有力反馈或安全规划；离开训练数据的接触变化仍不可见。详见 [[π₀]]。[原文](https://arxiv.org/abs/2410.24164)

## 14. FAST：动作 tokenizer 应利用时间频率结构

![[90 Attachments/VLA-Paper-Atlas/fast-pipeline.png]]

*FAST Figure 4 · PDF 第 5 页 · 原论文方法图。*

**Pipeline**：action chunk 先按维归一化并做 DCT；高频系数通常接近零，按频率顺序展平；数值经 byte-pair encoding 压成离散 token；decoder 逆过程恢复连续动作。

**为什么经典**：它指出逐时间、逐维分箱浪费 token，因为高频机器人轨迹在时间上高度冗余。FAST 把 tokenizer 从“数值量化”提升为“利用信号结构的压缩问题”。

**值得关注的洞察**：token 数与物理时间长度不是同一个轴。一段 1 秒动作可以由少量频域 token 表示；这使大规模 next-token pretraining 更高效，但部署仍输出连续轨迹。

**边界**：DCT 假设一定平滑性；突发接触、脉冲动作和不同控制频率可能改变压缩误差。FAST+ 的通用性来自额外百万轨迹训练，不能与原始公式混为一谈。[[FAST Tokenizer]] · [原文](https://arxiv.org/abs/2501.09747)

## 15. OpenVLA-OFT：原版 OpenVLA 的弱点并不必然属于 VLM backbone

![[90 Attachments/VLA-Paper-Atlas/openvla-oft-pipeline.png]]

*OpenVLA-OFT Figure 2 · PDF 第 3 页 · 原论文方法对比图。*

**Pipeline**：保留 OpenVLA 预训练表示，但下游 fine-tuning 改为 parallel decoding、continuous action、action chunking 与 L1 regression；一次并行生成整段动作，而非串行生成离散 action tokens。

**为什么重要**：它把 base model 与 adaptation recipe 分开。原 OpenVLA 在 LIBERO 或高频双臂上的弱表现，可能主要来自输出头和时序接口，而非 VLM 表示本身；OFT 报告了显著成功率与吞吐提升。

**真正的洞察**：很多“新 VLA 胜过旧 VLA”的结论，其实同时更换了动作表示、loss、chunk 长度和 decoder。OFT 提醒我们先优化接口，再宣称 backbone 被淘汰。

**边界**：L1 parallel head 简单高效，但不显式建模多峰生成分布；优势是否保留取决于目标任务与数据歧义。[原文](https://arxiv.org/abs/2502.19645)

## 16. UniVLA：没有 robot action label 的视频，也能先学“任务变化”

![[90 Attachments/VLA-Paper-Atlas/univla-pipeline.png]]

*UniVLA Figure 2 · PDF 第 4 页 · 原论文两阶段 pipeline。*

**Pipeline**：第一阶段在视频中用 DINO feature、语言与 spatial-temporal Transformer 学 task-centric latent action；第二阶段把 latent dynamics 作为跨视频的中间监督，再用少量 robot data 学 latent-to-action decoder。

**为什么重要**：它试图绕开机器人动作标签稀缺：互联网视频没有共同的关节空间，却包含“世界怎样因行为改变”。latent action 把可共享行为先验与 embodiment-specific control 分开。

**值得关注的洞察**：好的 latent action 不应只重建像素变化，而应过滤镜头运动、背景和与任务无关的人体动态；语言条件在这里承担“哪些变化算任务相关”的筛选器。

**边界**：latent 到真实 action 的 grounding 仍需机器人数据；更强视频预训练不能自动解决接触力、可达性与控制延迟。[原文](https://arxiv.org/abs/2505.06111)

## 17. π₀.5：foundation policy 的核心从 architecture 转向 heterogeneous training

![[90 Attachments/Pi05/pi05-fig3-pipeline.png]]

*π₀.5 Figure 3 · PDF 第 4 页 · 原论文两阶段 pipeline。*

**Pipeline**：预训练把多机器人、mobile manipulation、vision-language、高层 subtask 与网页数据统一到 FAST/text next-token 接口；post-training 再加入连续 Flow Expert；推理先产生当前子任务，再条件化低层 action chunk。

**为什么经典**：它把“会一个动作”与“在陌生家庭持续完成任务”之间的缺口拆成数据覆盖、层级决策和连续控制三层。贡献不再是单个新 head，而是一套异构数据怎样协同的训练协议。

**真正的洞察**：不同数据源补不同能力，不能只用 aggregate success 判断价值；网页数据可能主要保护 OOD 语义，高层数据负责阶段切换，robot data 负责可执行动作。

**边界**：长任务能力不等于长期记忆；π₀.5 仍会在遮挡、失败重试和跨十分钟状态追踪上暴露缺口，后续 MEM、RECAP、π₀.7 正是在补这些。详见 [[π₀.5]]。[原文](https://arxiv.org/abs/2504.16054)

---

## 18. 从经典论文提炼出的五个研究判断

1. **动作接口往往比 backbone 名字更决定真机表现**：OpenVLA → OFT 是最直接证据。
2. **数据“多”至少要拆成环境、任务、形态和失败状态四种覆盖**：OXE、DROID、UMI 分别补不同轴。
3. **chunk 是承诺，不只是加速**：ACT/DP/π₀ 都在交换推理频率与反馈新鲜度。
4. **3D、语言、触觉解决的是不同不可观测性**：DP3 的几何增益不能替代接触反馈，VLM 语义也不能替代坐标对齐。
5. **所谓 foundation，必须同时评价 zero-shot、适配效率与部署成本**：Octo、OpenVLA、RDT、π 系列强调的并非同一维度。

下一篇：[[VLA 前沿论文粗读雷达（2025-2026）]]。
