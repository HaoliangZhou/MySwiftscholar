# 1. 论文基本信息
## 1.1. 标题
论文标题为 **VTAM: Video-Tactile-Action Models for Complex Physical Interaction Beyond VLAs**，核心主题是提出一种融合视频、触觉感知的动作模型，解决现有视觉-语言-动作（VLA）模型在接触密集的复杂物理交互任务中性能不足的问题，实现鲁棒的力感知机器人操作。
## 1.2. 作者与隶属机构
作者包括：Haoran Yuan、Weigang Yi（通讯作者）、Zhenyu Zhang、Wendi Chen、Yuchen Mo、Jiashi Yin、Xinzhuo Li、Xiangyu Zeng等，隶属机构覆盖卡内基梅隆大学、伊利诺伊大学厄巴纳-香槟分校、斯坦福大学、上海交通大学等机器人与人工智能领域顶尖科研机构。
## 1.3. 发表状态
当前为预印本状态，发布于计算机领域预印本平台arXiv，尚未正式发表在学术期刊或会议。
## 1.4. 发表年份
2026年
## 1.5. 摘要
- **研究目的**：解决现有视频-动作模型（VAM）、视觉-语言-动作模型（VLA）在接触密集场景下的局限性：仅靠视觉无法捕捉细粒度力调制、接触状态遮挡导致的操作不稳定、精度不足问题。
- **核心方法**：提出视频-触觉动作模型（VTAM），在预训练视频Transformer上新增触觉流，通过轻量模态迁移微调实现跨模态表示学习，无需触觉-语言配对数据或独立触觉预训练；同时引入触觉正则化损失，平衡跨模态注意力，避免视觉隐特征主导模型。
- **主要结果**：接触密集操作任务平均成功率达90%；在需要高保真力感知的薯片搬运任务中，性能比$\pi_{0.5}$基线高出80%。
- **关键结论**：整合触觉反馈可有效修正世界动作模型的视觉估计误差，为物理接地的具身基础模型提供了可扩展的实现路径。
## 1.6. 原文链接
- 预印本链接：https://arxiv.org/abs/2603.23481v1
- PDF链接：https://arxiv.org/pdf/2603.23481v1
- 发布状态：arXiv预印本，发布时间为2026-03-24 UTC。

  ---

# 2. 整体概括
## 2.1. 研究背景与动机
### 核心问题与重要性
当前<strong>视觉-语言-动作模型（VLA，解释：将视觉观测、自然语言指令映射为机器人控制动作的大模型，是通用机器人控制的主流范式）</strong> 已实现跨任务、跨场景的泛化能力，但在接触密集的精细操作任务（如抓取易碎品、削皮、擦拭）中几乎失效。这类任务的核心状态（如接触力、滑动、表面变形）发生在视觉被遮挡的接触界面，仅靠视觉无法感知，是通用机器人走向实际应用必须解决的核心瓶颈。
### 现有研究的空白
现有将触觉融入VLA的方法存在两类核心缺陷：
1.  把触觉作为补充语义token投影到预训练的视觉-语言隐空间，或仅在下游策略层拼接视觉-触觉特征：这类方法的隐空间是为语义对齐优化的，无法有效编码接触动态的物理规律，且缺乏时序建模，难以学习动作、触觉、视觉之间的因果关系。
2.  训练时普遍存在<strong>模态坍塌（Modality Collapse，解释：多模态模型训练中，梯度更强的模态压制其他模态的信号，最终模型仅依赖强模态的现象）</strong>问题：视觉信号梯度远强于高频局部触觉信号，模型最终仍只依赖视觉，完全忽略触觉输入。
### 本文创新思路
跳出“触觉是视觉补充输入”的固有框架，将触觉作为核心感知模态融入预测式视频世界模型，联合预测未来视觉与触觉流的演化，让模型自主学习时序一致的接触动态；同时从触觉变形信号中导出自监督的虚拟力正则化，无需额外硬件即可解决模态坍塌问题。
## 2.2. 核心贡献/主要发现
### 核心贡献
本文共有四项核心贡献：
1.  提出VTAM：首个将高分辨率触觉感知融入预测视频主干网络的视觉-触觉世界动作模型，可实现鲁棒的接触密集机器人操作。
2.  提出联合视觉-触觉预测框架：在共享隐空间同步预测未来视觉、触觉流，无需显式接触事件标注，即可学习时序一致的接触动态。
3.  提出虚拟力预测目标：从触觉传感器的变形信号中导出虚拟力作为正则化，有效缓解多模态训练的模态坍塌问题，无需额外力传感器硬件。
4.  在三类挑战性接触密集任务上完成验证：性能远优于仅视觉、朴素触觉融合的基线模型。
### 主要发现
1.  接触密集任务中仅依赖视觉的模型几乎完全失效，即使是最先进的VLA也无法完成需要力控制的操作。
2.  朴素将触觉作为额外输入加入VLA无法提升性能，模态坍塌问题会导致触觉信号完全被视觉压制。
3.  预测式视觉-触觉联合建模+虚拟力正则化可让模型有效利用触觉信号，将接触密集任务的成功率从接近0提升至90%左右。

    ---

# 3. 预备知识与相关工作
## 3.1. 基础概念
理解本文需要掌握以下核心基础概念：
1.  <strong>具身智能（Embodied Intelligence）</strong>：指可与物理环境交互的智能体（如机器人），通过感知环境、执行动作完成任务，区别于仅处理文本、图像的被动智能。
2.  <strong>世界模型（World Model）</strong>：可预测环境未来状态的模型，输入当前观测与动作即可输出未来的环境状态，让智能体可在模型内“预演”动作后果，提升策略稳定性与样本效率。
3.  <strong>流匹配（Flow Matching）</strong>：一种生成式建模方法，通过学习从随机噪声分布到目标数据分布的连续向量场实现数据生成，相比扩散模型训练更稳定、推理速度更快，本文用于隐空间动态建模与动作生成。
4.  **GelSight触觉传感器**：一种高分辨率视觉触觉传感器，表面为可变形弹性体，内部通过摄像头捕捉弹性体的变形即可得到接触力分布、滑动、纹理等信息，是当前机器人操作领域的主流触觉传感器。
5.  <strong>自注意力机制（Self-Attention）</strong>：Transformer架构的核心组件，通过计算序列中每个token与其他所有token的相关性，加权融合得到更新后的特征表示，本文用于模态内与模态间的特征融合，核心公式为：
    $$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    其中$Q$（查询）、$K$（键）、$V$（值）是输入token经过三个线性层投影得到的矩阵，$d_k$为$K$的维度，除以$\sqrt{d_k}$用于防止内积过大导致softmax饱和。
## 3.2. 前人工作
### 3.2.1 视觉-语言-动作模型
现有VLA（如RT-2、Octo、$\pi_0$）依赖大规模视觉-语言预训练实现跨任务泛化，后续改进包括加入3D几何先验、分层规划、世界知识等，但所有仅视觉的VLA都无法处理接触状态被遮挡、需要力感知的操作任务。
### 3.2.2 机器人生成式世界模型
现有世界模型（如DreamZero、UWM、DreamVLA）基于视频扩散模型，通过预测未来视觉状态指导动作生成，但这类模型仅预测视觉信息，无法捕捉接触界面的滑动、变形、力变化，在精细操作中极易失败。
### 3.2.3 触觉集成的机器人学习
现有触觉融合方法（如Tactile-VLA、ForceVLA）大多将触觉作为反应式补充输入，仅在下游层与视觉特征拼接，缺乏预测式联合建模，且普遍存在模态坍塌问题，需要额外的力传感器或混合控制器，硬件要求高、通用性差。
## 3.3. 技术演进
该领域的技术发展脉络如下：
1.  早期机器人操作：依赖人工编程或传统视觉伺服，仅能完成特定任务，泛化性极差。
2.  深度学习时代：端到端视觉操作策略，通过监督学习、强化学习训练，仍存在泛化性差、数据需求量大的问题。
3.  VLA时代：基于大规模互联网视觉-语言预训练，实现跨任务、跨场景泛化，但接触密集任务完全失效。
4.  多模态VLA：尝试加入触觉、力觉信号，但均采用反应式融合，无法解决模态坍塌问题。
5.  本文VTAM：首次将触觉融入预测式世界模型，联合建模视觉-触觉动态，通过自监督正则化解决模态坍塌问题，填补了接触密集通用操作的技术空白。
## 3.4. 差异化分析
本文方法与现有方案的核心区别如下：
1.  与仅视觉VLA/世界模型相比：新增触觉模态，可捕捉视觉不可见的接触动态，适配接触密集任务。
2.  与朴素触觉融合VLA相比：不将触觉作为下游补充输入，而是将其融入视频主干做预测式联合建模，可学习时序一致的接触动态，无需显式标注。
3.  与其他触觉融合方法相比：无需额外力传感器，通过触觉变形导出虚拟力做正则化即可解决模态坍塌问题，硬件要求低、可扩展性强。
4.  无需触觉-语言配对数据，也无需单独预训练触觉模型，仅需轻量微调预训练视频主干，训练成本极低。

    ---

# 4. 方法论
VTAM的整体架构如下图（原文Figure 2）所示：

![Figure 2: VTAM Overview. A pretrained video backbone jointly models multi-view visual and tactile latents via alternating intra-view and cross-view attention. The resulting multimodal representation is injected into a conditional action diffusion head to predict action, virtual force, and proprioceptive state.](images/2.jpg)
*该图像是示意图，展示了VTAM模型的结构。图中左侧为视频基础模型，整合来自不同视角的自注意力模块；右侧为动作扩散模型，采用多视角交叉注意力机制。该模型输入包括过往的动作和状态信息，预测未来的动作、状态及力的变化。*

## 4.1. 方法原理
### 核心思想
将触觉作为与视觉同等重要的感知模态融入预训练视频世界模型，联合预测未来视觉、触觉流的演化，让模型学习接触动态的时序规律；同时引入从触觉变形导出的虚拟力作为正则化，防止训练时视觉模态主导，保证触觉信号有效影响动作生成。
### 设计直觉
接触密集任务的失败（如捏碎薯片、削皮脱离接触、擦拭飘起）均源于视觉无法感知接触界面的状态，而触觉可直接捕捉这类信息；仅通过联合预测视觉与触觉的未来变化，模型才能提前预判接触状态的变化，做出正确的动作调整；同时必须通过专门的正则化避免模型依赖更强的视觉信号忽略触觉。
## 4.2. 核心方法详解
### 4.2.1 多视角扩散的视觉-触觉隐空间世界建模
本文采用预训练的<strong>变分自编码器（Variational Autoencoder, VAE，解释：一种生成模型，由编码器将输入压缩到低维隐空间，解码器将隐向量还原为输入，训练目标为最小化重建误差，可保留输入的细粒度细节）</strong>提取视觉、触觉的隐特征，其重建导向的训练目标天然适合保留触觉的微小变形细节，无需专门设计触觉主干网络。
#### 步骤1：隐特征提取
对于时刻$t$第$v$个视角的输入帧$\mathbf{I}_t^v$，通过预训练视频VAE编码器$E$提取连续隐表示$\mathbf{z}_t^v$：
$$
\mathbf{z}_t^v = E\big(\mathbf{I}_t^v\big), \quad v \in \{1,2,3\}
$$
其中：
- $v=1,2$分别对应第三人称、第一人称视觉摄像头视角
- $v=3$对应GelSight触觉流视角
- $\mathbf{z}_t^v$为对应输入的隐特征向量
#### 步骤2：多模态时空动态建模
通过$B$个交替的视角内自注意力、跨视角自注意力块建模多模态的时空动态：
第$b$个块的输入为$\mathbf{Z}_{b-1} = \{\mathbf{z}_{t,b-1}^1, \mathbf{z}_{t,b-1}^2, \mathbf{z}_{t,b-1}^3\}$，初始输入$\mathbf{Z}_0$为VAE提取的初始隐特征。
首先对每个模态单独执行视角内自注意力，捕捉单模态内部的空间结构：
$$
\tilde{\mathbf{z}}_{t,b}^v = \mathrm{SelfAttention}(\mathbf{z}_{t,b-1}^v) \quad \forall v \in \{1,2,3\}
$$
其中$\tilde{\mathbf{z}}_{t,b}^v$为第$v$个模态更新后的隐特征。
随后将三个模态的更新token拼接，执行跨视角自注意力，建模模态间的交互关系：
$$
\mathbf{Z}_b = \mathrm{CrossViewAttention}\bigl(\mathrm{Concat}\bigl(\tilde{\mathbf{z}}_{t,b}^1, \tilde{\mathbf{z}}_{t,b}^2, \tilde{\mathbf{z}}_{t,b}^3\bigr)\bigr)
$$
其中$\mathbf{Z}_b$为第$b$个块的输出，包含融合后的视觉-触觉时空特征。
上述交替结构重复$B$次，最终得到稠密的联合视觉-触觉表示。
---
### 4.2.2 变形感知的虚拟力预测正则化
为解决模态坍塌问题，本文无需额外力传感器，直接从GelSight的变形图像导出虚拟力作为监督信号。
#### 步骤1：虚拟力计算
给定无接触的参考触觉帧$\mathbf{I}_0$与当前时刻的触觉帧$\mathbf{I}_t$，计算两者的稠密光流$\mathbf{u}_t = (u_x, u_y)$（表示触觉表面每个点的位移），随后从光流场导出3D虚拟力代理$\boldsymbol{F}_t^v = [f_x, f_y, f_z]^T$：
$$
f_x = \mathbb{E}\big[u_x\big], \quad f_y = \mathbb{E}\big[u_y\big], \quad f_z = \mathbb{E}\big[\boldsymbol{\nabla} \cdot \boldsymbol{u}_t\big]
$$
其中：
- $\mathbb{E}[\cdot]$为空间期望，即对整个触觉帧所有点取平均值
- $f_x$为x方向切向力估计，来自x方向光流的平均值
- $f_y$为y方向切向力估计，来自y方向光流的平均值
- $f_z$为法向压力估计，来自光流散度$\boldsymbol{\nabla} \cdot \boldsymbol{u}_t$的平均值：按压弹性体时接触点周围表面会向外扩张，光流散度可直接反映扩张程度，与法向压力正相关
  该虚拟力为几何接地的代理信号，无需力传感器校准即可反映接触力的大小与方向。
#### 步骤2：虚拟力正则化损失
将虚拟力作为辅助监督信号加入动作生成的流匹配目标，虚拟力的正则化损失为：
$$
\mathcal{L}_{\mathrm{force}} = \mathbb{E}\left[ \left\| v_{\theta}^f(z_t, t \vert c) - v^{*f} \right\|^2 \right]
$$
其中：
- $v_{\theta}^f$为模型预测的虚拟力对应的速度场
- $v^{*f}$为真实虚拟力对应的最优速度场
- $z_t$为加噪后的联合隐向量，$t$为扩散时间步，$c$为条件信号（当前状态）
- $\mathbb{E}[\cdot]$为对训练样本的期望
  该损失保证触觉通路有稳定的梯度回传，不会被视觉梯度压制，从根源避免模态坍塌。
---
### 4.2.3 两阶段训练策略
为稳定训练，本文将训练拆分为两个独立阶段：
#### 阶段1：多视角视觉-触觉隐流匹配
本阶段仅微调预训练的视频主干，不加入动作监督，目标是让主干学习视觉与触觉的联合动态，建立一致的多模态隐空间。损失函数为隐空间的流匹配损失：
$$
\mathcal{L}_{\mathrm{stage1}} = \mathbb{E}\left[ \left. \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) - \mathbf{v}^* \right.^2 \right]
$$
其中：
- $\mathbf{z}_0$为未来多视角观测（两个摄像头+触觉）的VAE编码隐序列
- $\mathbf{v}_{\boldsymbol{\theta}}$为模型预测的隐向量速度场
- $\mathbf{v}^*$为真实的最优速度场
  该损失仅作用于未来预测帧，初始条件帧不参与优化，可让主干学会从当前状态预测未来的视觉-触觉动态，无需任何接触标注。
#### 阶段2：条件联合动作-状态-力去噪
阶段1训练好的视觉-触觉世界模型固定，单独训练动作头生成条件动作。
联合去噪目标为动作、虚拟力、本体觉状态的拼接：
$$
\mathbf{z}_0 = \big[ \mathbf{a}; \mathbf{f}; \mathbf{s} \big]
$$
其中：
- $\mathbf{a} \in \mathbb{R}^7$为7维动作：6维末端执行器姿态 + 1维夹爪宽度
- $\mathbf{f} \in \mathbb{R}^3$为变形导出的3维虚拟力
- $\mathbf{s} \in \mathbb{R}^{16}$为16维机器人本体觉状态
  条件信号$c = [\mathbf{0}_{10}; \mathbf{s}_t]$，其中动作与力的维度用0填充，仅保留当前本体觉状态作为条件。
分别计算动作、状态的流匹配损失：
$$
\mathcal{L}_{\mathrm{action}} = \mathbb{E}\left[ \left. \mathbf{v}_{\boldsymbol{\theta}}^{\mathbf{a}}(\mathbf{z}_t, t \mid \mathbf{c}) - \mathbf{v}^{*\mathbf{a}} \right.^2 \right]
$$
$$
\mathcal{L}_{\mathrm{state}} = \mathbb{E}\left[ \left. \mathbf{v}_{\theta}^{\mathbf{s}}(\mathbf{z}_t, t \mid \mathbf{c}) - \mathbf{v}^{*s} \right.^2 \right]
$$
其中$\mathbf{v}_{\boldsymbol{\theta}}^{\mathbf{a}}$、$\mathbf{v}_{\boldsymbol{\theta}}^{\mathbf{s}}$分别为模型预测的动作、状态对应的速度场，$\mathbf{v}^{*\mathbf{a}}$、$\mathbf{v}^{*s}$为对应的真实最优速度场。
阶段2的总损失为三个损失的加权和：
$$
\mathcal{L}_{\mathrm{stage2}} = \mathcal{L}_{\mathrm{action}} + \lambda_1 \mathcal{L}_{\mathrm{state}} + \lambda_2 \mathcal{L}_{\mathrm{force}}
$$
本文所有实验取$\lambda_1 = \lambda_2 = 1$，由于流匹配回归的是归一化速度场，三个损失的量级天然一致，无需复杂的超参数调整。该联合训练可保证动作生成与状态、力预测一致，确保动作符合物理规律。

---

# 5. 实验设置
实验硬件环境为6自由度xArm6机械臂，配备平行夹爪，夹爪手指上安装GelSight Mini触觉传感器，视觉观测来自两个Intel RealSense D455 RGB-D摄像头，数据采集与动作执行频率为30Hz，实验场景如下图（原文Figure 3）所示：

![Figure 3: Experiment setup and data acquisition. We collect demonstrations through manual teleoperatio using a visuotactile sensing setup for contact-rich manipulation tasks such as chip pick-and-place.](images/3.jpg)
*该图像是实验设置的示意图，展示了用于接触丰富的操作任务（如零食夹取）的机器人实验装置和手动数据收集过程。机器人手臂正准备对盘中的物体进行操作，周围布置了视觉和触觉传感器以获取数据。*

## 5.1. 数据集
实验采用作者自行收集的真实世界视觉-触觉数据集，包含三类接触密集任务：
1.  **薯片搬运**：100条演示轨迹，任务为夹取易碎薯片放到目标盘子中，要求不能捏碎、不能掉落，抓取阶段手部遮挡严重，视觉无法观测接触状态。
2.  **黄瓜削皮**：61条演示轨迹，任务为用削皮刀给黄瓜削皮，要求保持稳定接触与切向力控制，黄瓜为可变形物体，厚度会随削皮动态变化。
3.  **白板擦拭**：105条演示轨迹，包含平面白板、45度倾斜白板的擦拭任务，要求保持稳定的法向压力，不能飘起也不能压力过大。
    所有演示通过手动遥操作收集，包含同步的多视角RGB流、GelSight触觉变形图像、机器人本体觉状态。
## 5.2. 评估指标
本文采用<strong>任务成功率（Task Success Rate）</strong>作为核心评估指标，说明如下：
1.  **概念定义**：量化智能体在多次独立测试中成功完成指定任务的比例，反映模型完成目标任务的可靠性，是机器人操作任务最核心的评估指标。三类任务的成功判定标准为：
    - 薯片搬运：成功夹取薯片，运输过程无掉落、无碎裂，成功放置到目标盘子中。
    - 黄瓜削皮：连续20次削皮动作，每次削皮条长度超过10cm。
    - 白板擦拭：最多5次擦拭动作，完全清除随机绘制的黑色污渍，无明显残留。
2.  **数学公式**：
    $$
\text{Success Rate} = \frac{\text{Number of Successful Trials}}{\text{Total Number of Test Trials}} \times 100\%
$$
3.  **符号解释**：
    - $\text{Number of Successful Trials}$：满足任务成功标准的测试次数
    - $\text{Total Number of Test Trials}$：总的独立测试次数
## 5.3. 对比基线
本文选择三类代表性基线：
1.  <strong>Genie Envisioner（GE）</strong>：最先进的仅视觉视频-动作模型，将指令条件的视频扩散主干与流匹配动作解码器结合，代表现有仅视觉世界模型的最高水平。
2.  <strong>$\pi_{0.5}$（仅视觉）</strong>：通用VLA模型$\pi_0$的改进版，具备极强的开放世界泛化能力，代表仅视觉VLA的最高水平，用于测试仅靠语义视觉表示在力敏感任务中的性能上限。
3.  **$\pi_{0.5}$ + 朴素触觉注入**：将GelSight触觉流作为额外视觉视图直接加入$\pi_{0.5}$的输入，无专门融合与正则化，用于测试朴素触觉融合的效果，验证模态坍塌问题的存在。

    ---

# 6. 实验结果与分析
每个任务执行20次独立测试，推理频率为1Hz。
## 6.1. 核心结果分析
整体性能对比结果如下（原文Table 1）：

<table>
<thead>
<tr>
<th>Model</th>
<th>Chip（薯片搬运）</th>
<th>Peel（黄瓜削皮）</th>
<th>Wipe（白板擦拭）</th>
</tr>
</thead>
<tbody>
<tr>
<td>Genie Envisioner</td>
<td>0%</td>
<td>0%</td>
<td>2.5%</td>
</tr>
<tr>
<td>$\pi_{0.5}$ (Vision)</td>
<td>10%</td>
<td>0%</td>
<td>0%</td>
</tr>
<tr>
<td>$\pi_{0.5}$ + Tactile</td>
<td>5%</td>
<td>0%</td>
<td>0%</td>
</tr>
<tr>
<td>VTAM (Ours)</td>
<td>90%</td>
<td>85%</td>
<td>95%</td>
</tr>
</tbody>
</table>

结果分析：
1.  所有基线在黄瓜削皮、白板擦拭任务中成功率均为0，薯片搬运最高仅10%，说明仅靠视觉或朴素触觉融合完全无法处理接触密集任务：基线无法判断是否成功抓取、仅靠视觉跟随物体轮廓导致脱离接触、压力控制完全失效。
2.  VTAM在三类任务中成功率分别达到90%、85%、95%，远高于所有基线，证明联合视觉-触觉预测+虚拟力正则化可让模型有效利用触觉信号，稳定控制接触力。
    定性对比结果如下图（原文Figure 4）所示，红框为基线的典型失败案例：

    ![Figure 4: Qualitative comparison between VTAM and baseline methods on real-world manipulation tassTop:Chippick-an-plaVisin-y baselne l detere whethehe hias beeuy raspe and proc the placeent tage even when he rasp fails.Mid:ucmber peelig int s. Baselines tend to follow a vision-driven trajectory that approaches the center of the cucumber but fail to maintain consistent contact with the surfae, indicating poor forc regulation and lackof contac awareness. Bot:Whiteboard wipng under varying heights and tilt angles. Baselines exhibit unstable wiping behaviors, oe applyieithesuient xcesivel largorce, particularyntilteurfacs. In cntrast, VTAM maintains stable contact and appropriate force regulation across all tasks, enabling robust manipulation behaviors. Red boxes highlight representative failure cases of baselines.](images/4.jpg)
    *该图像是VTAM与基线方法在真实世界操作任务中的定性比较，包括薯片拾取、黄瓜削皮以及白板擦拭。图中展示了VTAM在各种操作场景中维持稳定接触和适当力调节的表现，而基线方法则显示不稳定的操控行为，红框突出基线的失败案例。*

本文还可视化了世界模型的预测结果（原文Figure 5），模型可准确预测未来两个视觉视角与触觉流的变化，仅与操作无关的细节存在轻微模糊，证明视觉-触觉世界建模的可靠性：

![Figure 5: Prediction visualization of the backbone video model. From top to bottom: Camera-1 viev Camera-2 view, Tactile stream prediction. Ground-truth (top rows) and VTAM predictions (bottom rows).](images/5.jpg)
*该图像是VTAM模型的预测可视化，展示了从不同相机视角和触觉流的预测。包括真实数据（顶部行）和VTAM预测（底部行），展示了与物理交互相关的复杂动态。*

## 6.2. 消融实验分析
作者在最具挑战性的薯片搬运任务上开展消融实验，每个变体执行10次测试，结果如下（原文Table 2）：

<table>
<thead>
<tr>
<th>Model Variant</th>
<th>Tactile Integration</th>
<th>Success Rate</th>
</tr>
</thead>
<tbody>
<tr>
<td>Vision-only (No Tactile)</td>
<td>无触觉输入</td>
<td>0%</td>
</tr>
<tr>
<td>Late-Fusion Tactile</td>
<td>仅下游动作头融合触觉</td>
<td>0%</td>
</tr>
<tr>
<td>No Virtual-Force Reg.</td>
<td>联合隐空间建模，无虚拟力正则化</td>
<td>10%</td>
</tr>
<tr>
<td>VTAM (Ours)</td>
<td>分层世界模型+虚拟力正则化</td>
<td>90%</td>
</tr>
</tbody>
</table>

消融结果分析：
1.  仅视觉变体成功率为0%，验证了抓取遮挡场景下视觉无法感知接触状态，完全无法完成任务。
2.  下游 late fusion 触觉的变体成功率为0%，说明不将触觉融入世界模型做联合预测，仅靠下游反应式融合无法学习接触动态。
3.  移除虚拟力正则化的变体成功率仅10%，证明模态坍塌问题确实存在：无正则化时视觉梯度主导，模型几乎忽略触觉信号，性能与仅视觉基线接近。
4.  完整VTAM成功率达90%，证明联合视觉-触觉世界建模、虚拟力正则化两个组件缺一不可。

    ---

# 7. 总结与思考
## 7.1. 结论总结
本文提出的VTAM是首个将触觉感知融入预测式视频世界模型的视觉-触觉动作模型，有效解决了现有VLA在接触密集任务中失效的核心问题。其核心创新为：1. 将触觉作为核心模态融入预训练视频主干，联合预测未来视觉、触觉流，无需显式接触标注与触觉-语言配对数据；2. 提出从触觉变形导出的虚拟力正则化，无需额外硬件即可解决多模态训练的模态坍塌问题。实验表明VTAM在三类接触密集任务上的成功率远优于现有基线，证明了预测式多模态建模对物理接地具身智能的重要性，为通用机器人操作提供了可扩展的实现路径。
## 7.2. 局限性与未来工作
### 现有局限性
1.  目前仅在三类特定操作任务上验证，未测试跨更多任务、跨不同机器人的泛化能力。
2.  虚拟力为几何代理信号，未校准为真实物理力，力控制精度仍有提升空间。
3.  世界模型仅能预测几秒内的短时序状态，无法支持更复杂的长时序接触任务。
4.  目前未接入语言指令，还不能实现自然语言控制的多任务接触操作。
### 未来研究方向
1.  扩大数据集规模，用大规模多模态机器人演示数据预训练VTAM，提升跨任务、跨机器人的泛化能力。
2.  结合少量力传感器数据校准虚拟力，提升力控制精度。
3.  扩展长时序预测能力，支持更复杂的长周期接触任务。
4.  加入语言条件，扩展为真正的多模态VLA，支持自然语言指令控制。
## 7.3. 个人启发与批判
### 启发
1.  本文打破了“触觉是视觉补充输入”的固有框架，将触觉融入世界模型做预测式联合建模的思路极具启发性：其他模态（如力觉、听觉）均可采用类似思路融入世界模型，大幅提升具身智能的物理交互能力。
2.  利用传感器本身的信号导出自监督正则化信号（如本文从触觉变形导出虚拟力），无需额外标注与硬件即可解决模态坍塌问题，该思路可迁移到各类多模态学习场景。
3.  两阶段训练策略（先对齐多模态世界表示，再训练动作头）大幅提升了训练稳定性，可推广到其他多模态具身模型的训练。
### 潜在改进空间
1.  当前推理频率仅为1Hz，远低于真实机器人控制所需的30Hz，未来需优化模型推理速度，或结合高频反馈控制器才能落地部署。
2.  目前正则化权重为固定值，未来可设计动态权重调整机制，根据训练过程中模态的贡献自动调整权重，进一步缓解模态坍塌问题。
3.  尚未在动态干扰环境中测试鲁棒性，未来可验证模型在环境突变、外力干扰下的稳定性。