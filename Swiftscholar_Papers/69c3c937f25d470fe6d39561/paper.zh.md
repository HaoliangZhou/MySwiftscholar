# VTAM：用于复杂物理交互的视频-触觉-动作模型，超越视觉-语言架构

霍然·袁1,\*\*, 因冠 $\mathbf { Y _ { i } ^ { \bullet } } ^ { 1 , \ast }$ , 张振宇, 陈文迪3, 莫宇辰1, 颜佳士 $\mathrm { Y i n } ^ { 1 }$ , 李欣卓 $\mathrm { L i } ^ { 1 }$ , 曾向宇1 C We C ², 伊利诺伊大学厄本-香槟分校 tanord Univy3上海交通大学

![](images/1.jpg)  

Figure 1: We introduce VTAM, a generalist VideoTactile Action Model that integrates tactile sensing into a predictive video world model. By grounding control in visuotactile dynamics and force-aware reasoning, VTAM enables stable control in diverse contact-rich scenarios.

摘要：视频动作模型（VAMs）作为一种有前途的具身智能框架，从原始视频流中学习隐含的世界动态，以产生时间一致的动作预测。尽管此类模型在长时间任务中的视觉推理表现强劲，但它们在仅依赖视觉信息时存在不足。特别是，精细的力反馈调制和接触过渡在视觉词元中并未得到可靠编码，导致动作行为不稳定和不精确。为了解决这一问题，我们引入了视频触觉模型（VTAM），这是一个多模态世界建模框架，结合了触觉感知作为补充的基础信号。VTAM通过轻量化的模态转换，将触觉流增强到预训练的视频变换器中，而无需触觉独立预训练。为稳定多模态融合，我们引入了一种触觉正则化损失，强制平衡跨模态注意力，防止视觉信息在动作模型中主导。VTAM在接触丰富的操控任务中表现优越，在平均悬挂物体的成功率达到了90%，超出$\pi _ { 0.5 }$基线80%。我们的研究表明，整合触觉反馈对于纠正视觉估计错误在工作动作模型中至关重要，提供了一种可扩展的方法以实现物理基础的具身基础模型。

# 1. 引言

最近在视觉语言-动作（VLA）模型方面的进展使得通过大规模多模态对齐实现通用机器人控制成为可能。通过将视觉观察和语言指令嵌入到共享的语义潜在空间中，这些模型能够跨多种操作任务和环境进行泛化。然而，尽管视觉支持高水平的语义理解，语言却在战术上限制了模型的能力，触觉则直接编码机器人与其环境之间瞬时的接触动态。触觉感知对于细致、富含接触的操作特别关键，例如处理易损、可变形或滑腻的物体。与从远处捕捉相对稳定物体几何形状的视觉不同，触觉需要不仅对力分布进行空间推理，还要对这些分布在动态交互下如何演变进行时间推理。

大多数现有的触觉增强视觉-语言架构通过两种方式整合触觉信息：（1）将触觉嵌入投影到预训练的视觉语言潜在空间中，将其视为额外的语义标记[37]，或（2）在下游策略中将触觉特征与语言条件的视觉表示进行连接[2, 11, 16]。虽然这些方法使模型能够接触触觉输入，但在表征学习上却带来了沉重的负担：模型必须在为视觉对齐和静态场景描述而优化的语义嵌入空间中隐式推断接触物理，而非物理预测。学习特定的触觉模式与滑动、变形或不稳定性相关，需要通过静态相关性间接发现这些概念。这通常需要大规模标注的数据集来保证这些动态高频特征能被忠实捕捉。没有明确的时间建模，这些学习到的表征难以编码视觉输入、触觉输入之间的因果关系[22, 32, 42]。此外，由于许多视觉-语言架构优先考虑语义对齐，而非预测性物理建模，因此在细粒度的空间和时间推理方面进一步低估了触觉信号的利用[2, 16]。

我们通过引入VTAM（一种通用视频触觉动作模型）来解决这些局限，该模型将触觉感知整合进面向接触丰富操作的预测世界模型框架中，如图1所示。在表示层面，我们设计了一个基于预训练视频主干网络的视觉触觉预测模块。与将触觉信号映射到语言对齐语义空间不同，VTAM将触觉视为一种主要感官模态，并联合预测未来视觉和触觉流的演变，条件为机器人末端执行器的状态。该预测式设计使主干网络能够学习时间上连续一致的视觉触觉特征，无需对接触事件进行显式语义标注。此外，在动作学习层面，我们针对集成触觉输入进行训练时常见的模态崩塌问题，通过引入虚拟力预测目标对多模态融合进行正则化，从而稳定训练。该设计鼓励策略在动作优化过程中保持对触觉信号的敏感性，有效防止视觉特征的主导地位。我们在三项不同的接触丰富操作任务上验证了VTAM：薯片拾取放置、削皮和擦拭。在薯片拾取放置任务中，VTAM达到了90%的成功率，而仅使用视觉的基线为0%，去除虚拟力正则化时成功率为10%。未经预测的视觉触觉模型的下游力信号简单集成完全失败（成功率0%）。在削皮和擦拭任务中观察到了类似趋势，表明预测视觉触觉表示学习结合动作层正则化显著提升了系统的稳定性和任务成功率。总结我们的主要贡献如下： - 我们提出VTAM，一种将高分辨率触觉感知与视觉观察结合在预测视频主干网络中的视觉触觉世界动作模型，以实现鲁棒的接触丰富机器人操作。 - 我们设计了联合视觉触觉预测框架，在共享的潜在空间中预测未来的视觉与触觉流，使模型无需显式语义标注即可学习时间连续一致的接触动态。 - 我们引入虚拟力预测目标，有效缓解训练过程中的模态崩塌问题，相较仅视觉和简单集成基线带来了实证性能提升。 - 我们在包括薯片拾取放置、土豆削皮和具有不同高度及倾斜角度的白板擦拭等具有挑战性的接触丰富机器人任务上验证了VTAM，展示了较仅视觉和简单触觉基线显著提升的成功率。

# 2.相关工作

视觉-语言-动作模型。VLA 模型已成为通用机器人控制的主流范式，利用互联网规模的视觉语言预训练将自然语言指令与视觉观察相结合，并通过统一架构解码运动命令。随后的工作在多个方向上扩展了这一范式，融入了 3D 几何先验、层级任务规划和预测世界知识，持续改善了模型的泛化能力和样本效率。现有的视听语言 VLA 在视觉线索被遮挡时在物理交互方面表现不佳，尤其是对脆弱物体。VTAM 针对这一不足，通过将高分辨率触觉观察直接融入生成世界模型的主干网络来解决这一问题：模型学习联合的视触觉动态，并利用这些表示指导动作生成，从而在交互过程中用触觉线索纠正视觉误估，提高在脆弱和对力敏感任务上的鲁棒性。机器人的生成世界模型。生成世界模型预测未来环境状态以支持规划和策略学习。最近的研究通过联合扩散视频和动作轨迹扩展了这一理念。DreamZero 在预训练的视频扩散主干上构建了世界动作模型，通过从异构机器人数据中学习物理动态，实现了零样本泛化和跨身体转移。UWM 引入了特定模态的扩散时间步，解耦了视频与动作噪声调度，支持在包括无动作视频的大规模数据集上进行预训练。DreamVLA 通过未来视觉词元预测增强了 VLA，而 RDP 以层级方式应用扩散进行接触感知的动作细化。尽管取得了这些进展，但大多数现有的世界模型几乎完全通过视觉预测编码环境动态。尽管视觉线索能够捕捉物体运动和场景演变，但也会限制对物理交互目标的直接访问，这些目标在接触敏感的操作中至关重要。关键现象如变形和扭转往往出现在接触界面，并且在摄像机视野之外几乎完全不可观察。因此，依靠视觉动态的模型可能会在细致或对力敏感的交互中预测失败模式时遇到困难。受到这一限制的启发，VTAM 在预测世界模型中引入了触觉变形动态，并通过虚拟力目标锚定控制学习，使得策略在接触变得视觉模糊时仍能保持响应。

触觉整合在机器人学习中的应用。触觉传感为接触物理提供了直接的访问，且在涉及可变形、易碎或被遮挡物体的操作中至关重要。在表征层面，已有对比目标用于对齐视觉和触觉嵌入，或用于学习传感器-混合专家路由、双层反馈融合或触觉偏好优化。然而，这些方法将触觉视为与视觉反应融合的补充输入通道，而不是以预测方式建模。此外，一个实际挑战是模态崩溃：在训练过程中，视觉梯度占主导地位并抑制触觉或力信号。现有的缓解措施依赖于显式的力-扭矩传感器或混合位置-力控制器，从而施加硬件限制，限制了通用性。VTAM在两个方面偏离了这种反应性范式：它将触觉感知嵌入生成视频主干中，以实现联合视觉-触觉动态预测，而不是静态融合，并在动作头中引入考虑变形的虚拟力正则化，以在整个训练过程中保持触觉梯度的影响，而无需外部力-扭矩硬件。

![](images/2.jpg)  

Figure 2: VTAM Overview. A pretrained video backbone jointly models multi-view visual and tactile latents via alternating intra-view and cross-view attention. The resulting multimodal representation is injected into a conditional action diffusion head to predict action, virtual force, and proprioceptive state.

# 3. 方法

我们提出了视频触觉动作模型（VTAM），这是一个统一的视觉-触觉世界动作模型，旨在处理接触丰富的操作。如图2所示，VTAM通过同时投影多视角视觉观察和高输出触觉信息来工作，利用预训练的变分自编码器（VAE）。在这个空间内，采用交替的视内和跨视注意力的多视角扩散过程共同建模视觉场景的时间动态以及触觉传感器所捕捉的细微物理变形。这些结果多模态表征编码了预测接触卷积。这些表征随后通过交叉注意力注入条件扩散基础动作头，从而产生时间一致且具有物理基础的控制动作。在共享的主干网络中共同优化视觉和触觉模态往往会导致模态崩溃，支配性的视觉梯度会抑制局部高频触觉信号。为了解决这一基本的优化挑战，我们在动作头引入了一种感知变形的虚拟力正则化机制。该机制对触觉路径提供了有针对性的监督，从而稳定多模态融合，并确保策略在下游任务中对关键接触过渡保持敏感。

# 3.1. 通过多视图扩散进行视觉-触觉潜在世界建模

在视觉触觉建模中，一个基本挑战是保留高频空间细节，例如微妙的表面变形和纹理变化，这些细节在如GelSight的触觉传感器中编码了剪切、滑动和压力。标准的语义视觉编码器通常会为了粗略的对象级特征而丢弃这些细节。因此，我们基于预训练的视频变分自编码器（VAE）。VAE的重建导向目标提供了一个自然的归纳偏置，能够保留细粒度的空间和运动模式。这使我们能够高效地传递模态，而无需设计专门的触觉主干。除了空间细节，有效的接触操控还需要理解力的时间演变。我们并不通过轻量级的反应性下游分支处理触觉信号，而是将触觉流直接嵌入高容量的视频变压器中。这种架构同时捕捉了帧内信息结构和瞬时接触状态。因此，该模型对力趋势进行预测推理，使其能够预见关键的转变——这对于处理在运动中几毫毫米内发生失效的脆弱物体至关重要。正式地，给定来自视图$v$的时间步$t$的输入帧$\mathbf { I } _ { t } ^ { v }$，我们使用预训练的视频VAE编码器$E$提取连续潜在表示${ \mathbf { z } } _ { t } ^ { v }$：

$$
\begin{array} { r } { \mathbf { z } _ { t } ^ { v } = E \big ( \mathbf { I } _ { t } ^ { v } \big ) , \quad v \in \{ 1 , 2 , 3 \} , } \end{array}
$$

其中 $v = 1 , 2$ 表示第三人称和第一人称视觉相机视角，$v = 3$ 表示 GelSight 触觉流。为了建模复杂的空间和跨模态动态，我们通过一系列 $B$ 个块对这些潜在变量进行处理。$\mathbf{Z}_{b} = \big\{ \mathbf{z}_{t,b}^{1}, \mathbf{z}_{t,b}^{2}, \mathbf{z}_{t,b}^{3} \big\}$ 代表第 $b$ 个块，$\mathbf{Z}_{0}$ 是初始的变分自编码器编码。对于每个块 $b \in \left\{ 1, \ldots, B \right\}$，我们首先独立地对每个模态应用视内自注意力，以捕捉空间结构：

$$
\tilde { \mathbf { z } } _ { t , b } ^ { v } = \mathrm { S e l f A t t e n t i o n } ( \mathbf { z } _ { t , b - 1 } ^ { v } ) \quad \forall v \in \{ 1 , 2 , 3 \} .
$$

接下来，我们将所有视图中的更新词元进行拼接，并应用跨视图自注意力操作以建模模态间的交互：

$$
\mathbf { Z } _ { b } = \mathrm { C r o s s V i e w A t t e n t i o n } \bigl ( \mathrm { C o n c a t } \bigl ( \tilde { \mathbf { z } } _ { t , b } ^ { 1 } , \tilde { \mathbf { z } } _ { t , b } ^ { 2 } , \tilde { \mathbf { z } } _ { t , b } ^ { 3 } \bigr ) \bigr ) .
$$

这种交替结构在所有 $B$ 个块中重复，逐渐构建出关节的密集视觉触觉表示。

# 3.2. 考虑形变的正则化通过虚拟力预测

尽管预测主干网络能够实现视觉与触觉的联合表征学习，但我们在动作训练过程中观察到一种关键的模态崩溃现象。具体来说，当任务损失足够依赖视觉线索进行最小化时，流经触觉分支的梯度会减弱。因此，策略变得过于依赖视觉，忽视触觉反馈，导致在对力敏感的操控任务中控制不稳定。为了解决这一问题，我们引入了一种辅助目标，使得触觉通路能够提供更好的信息。以往的研究常常依赖于安装在机器人手腕或抓手上的外部力-扭矩传感器以获得真实的三维力监督。相反，我们观察到基于视觉的触觉传感器本质上编码了与接触力相关的丰富变形模式。通过强制预测一个紧凑的、与变形相关的信号，我们确保触觉表征在不增加重构高维触觉图像的计算开销的同时仍然保持信息性。形式上，给定一个没有接触的参考帧 $I _ { 0 }$ 和当前触觉帧 $I _ { t }$ ，我们计算出稠密光流 $u _ { t } = \left( u _ { x } , u _ { y } \right)$ 。我们直接从这个变形场中推导出一个三维虚拟力代理 $\boldsymbol { F } _ { t } ^ { v } = \left[ f _ { x } , f _ { y } , f _ { z } \right] ^ { \intercal }$ ：

$$
f _ { x } = \mathbb { E } \big [ u _ { x } \big ] , \quad f _ { y } = \mathbb { E } \big [ u _ { y } \big ] , \quad f _ { z } = \mathbb { E } \big [ \boldsymbol { \nabla } \cdot \boldsymbol { u } _ { t } \big ] .
$$

在这里，流动分量 $f _ { x }$ 和 $f _ { y }$ 的空间期望编码了切向剪切。关键是，$f _ { z }$ 通过流动散度近似法向压缩，利用了这一特性：将可变形弹性体压紧于物体上会导致表面图案向外扩展。该信号作为几何基础的代理，而不是经过校准的物理力。推导出的虚拟力 $\boldsymbol { F } _ { t } ^ { v } \in \mathbb { R } ^ { 3 }$ 在动作训练期间作为辅助监督信号。我们并未添加一个孤立的下游预测头，而是将这个紧凑的力代理作为条件流匹配目标的联合去噪目标中的一个附加组成部分。具体而言，网络的任务是共同预测未来的动作和虚拟力，有效地将控制梯度与触觉表示绑定在一起。显式的力正则化项评估力分量的矢量场速度匹配：

$$
\mathcal { L } _ { \mathrm { f o r c e } } = \mathbb { E } \left[ \left\| v _ { \theta } ^ { f } ( z _ { t } , t \vert c ) - v ^ { * f } \right\| ^ { 2 } \right] .
$$

该公式在潜在空间中保留了对变形敏感的信息，并在优化过程中保持平衡的多模态梯度。

# 3.3. 优化目标

为了将预训练的视觉主干网络适应于多模态视觉触觉建模，我们采用了两阶段训练策略。该主干网络最初仅在视觉数据集上进行了预训练，因此缺乏对触觉信号的高频局部变形模式的先前接触。引入动作监督和虚拟力正则化，与模态对齐同时进行，迫使网络在优化控制策略的同时调整其内部表示。触觉潜在质量不佳，导致不稳定的收敛。为了解决这个问题，我们解耦了 pSa e-ne he aconexcusive odjoinisa ate yami blih a coerent multi-modal world representation。阶段 I 随即利用这种对齐表示引入规则化的动作预测。阶段 I：多视角视觉触觉潜在流匹配。令 $\mathbf { z } _ { 0 }$ 表示未来多视角观察的 VAE 编码潜在序列，包括两个摄像头视角和 GelSight 流。我们应用流匹配公式建模这些视觉-触觉潜在的前向动力学：

$$
\mathcal { L } _ { \mathrm { s t a g e 1 } } = \mathbb { E } \left[ \left. \mathbf { v } _ { \boldsymbol { \theta } } ( \mathbf { z } _ { t } , t ) - \mathbf { v } ^ { * } \right. ^ { 2 } \right] .
$$

至关重要的是，该损失仅应用于未来预测帧，而初始条件帧被排除在优化目标之外。这个阶段的目的是使预训练的主干网络适应宏观视觉动态与微观触觉变形之间的物理相互作用，确保在引入任何控制信号之前，获得合理的多模态潜在空间。第二阶段：条件联合动作状态力去噪。在第一阶段训练出强健的视触觉世界模型之后，我们优化控制策略。我们将动作生成公式化为条件匹配过程。联合去噪目标是通过连接动作、虚拟力和状态构建的：

$$
{ \bf z } _ { 0 } = \big [ { \bf a } ; { \bf f } ; { \bf s } \big ] ,
$$

其中 $\mathbf{a} \in \mathbb{R}^{7}$ 表示6自由度末端执行器的位姿和1维夹爪宽度，$\mathbf{f} \in \mathbb{R}^{3}$ 是基于变形得到的虚拟力，$\mathbf{s} \in \mathbb{R}^{16}$ 是感知数据。网络预测关节速度，条件是当前状态标记 $\mathbf{c} = \left[ \mathbf{0}_{10}; \mathbf{s}_{t} \right]$，其中动作和力的维度在条件化过程中用零填充。我们为动作和状态分量定义流匹配目标，以追踪各自子空间的最优去噪轨迹：

$$
\mathcal { L } _ { \mathrm { a c t i o n } } = \mathbb { E } \left[ \left. \mathbf { v } _ { \boldsymbol { \theta } } ^ { \mathbf { a } } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) - \mathbf { v } ^ { * { \mathbf { a } } } \right. ^ { 2 } \right] ,
$$

$$
\mathcal { L } _ { \mathrm { s t a t e } } = \mathbb { E } \left[ \left. \mathbf { v } _ { \theta } ^ { \mathbf { s } } ( \mathbf { z } _ { t } , t \mid \mathbf { c } ) - \mathbf { v } ^ { * s } \right. ^ { 2 } \right] .
$$

然后，我们将其与虚拟力正则化 ${ \mathcal { L } } _ { \mathrm { f o r c e } }$（在公式 5 中定义）结合，以形成 cope tag 目标。总损失最小化各个组件之间的速度匹配误差之和：

$$
\mathcal { L } _ { \mathrm { s t a g e 2 } } = \mathcal { L } _ { \mathrm { a c t i o n } } + \lambda _ { 1 } \mathcal { L } _ { \mathrm { s t a t e } } + \lambda _ { 2 } \mathcal { L } _ { \mathrm { f o r c e } } .
$$

由于流匹配回归的是一个归一化的速度场 $( \epsilon - { \bf z } _ { 0 } )$，而不是原始数据值，因此在动作、状态和力维度上的目标方差自然保持了尺度一致。这避免了在标准均方误差回归中通常需要进行的激进超参数平衡。此外，联合状态预测引入了一个重要的动态一致性约束，确保模型将其控制预测基于一致的物理状态转移，而不是记忆孤立的动作轨迹。

# 4. 实验

我们在真实的接触丰富的操作任务中评估 VAM，以研究视觉-触觉世界动作建模的有效性。我们的实验旨在回答以下关键问题：Q1：视觉-触觉世界动作建模的有效性。在需要精细力调节的场景中，VTAM 是否优于仅基于视觉和多模态的基线？ Q2：潜在视频融合与后期注入。将视觉-触觉动态建模于共享的视频潜在空间，是否相较于后期触觉注入具有性能优势？ Q3：虚拟力正则化的影响。接触感知的虚拟目标正则化在多大程度上减轻模态崩溃并稳定多模态训练？

# 4.1. 实验设置

所有实验均在配备并联系抓手的6自由度xArm6机器人 manipulator上进行（3a）。我们使用安装在抓手手指上的GelSight Mini触觉传感器捕捉高分辨率的表面形变。视觉观测数据来自两个以双头配置安装的英特尔RealSense D455 RGB-D摄像头。数据采集和动作执行均在$30 \mathrm{Hz}$下进行。为了评估视觉触觉世界动作建模的有效性，我们将VTAM与若干强基准进行比较。

视频动作模型（Genie Envisioner）[19]。一种最先进的视频基础模型，结合了指令条件的视频扩散主干与流匹配动作解码器。 • $\pi _ { 0 . 5 }$（仅视觉）[24]。$\pi _ { 0 . 5 }$通用视觉-语言-动作（VLA）策略的官方实现，该策略针对开放世界泛化对$\pi _ { 0 }$架构[3]进行了扩展。该基线隔离了在力敏感场景中，关键接触状态被视觉遮挡时语义重的仅视觉表示的性能极限。b $\pi _ { 0 . 5 } ~ +$ 初始触觉注入 [24]。$\pi _ { 0 . 5 }$架构的多模态扩展，其中高维的GelSight触觉流仅作为额外的视觉视图注入。该设置专门用于展示模态崩溃现象，即在无正则化的联合训练过程中，主导的视觉梯度抑制局部触觉信号。

![](images/3.jpg)  

Figure 3: Experiment setup and data acquisition. We collect demonstrations through manual teleoperatio using a visuotactile sensing setup for contact-rich manipulation tasks such as chip pick-and-place.

# 4.2. 真实世界任务与数据收集

我们在三个接触丰富的操控任务上评估了VTAM： • 薯片拾取与放置：抓取和运输易碎的薯片而不造成破损，要求细致的力调节。成功依赖于精准调节抓取力以及在严重的手引起的遮挡下检测接触开始。策略必须避免抓取不足（滑落/掉落）和过度抓取（薯片破裂），同时进行提起和放置。 • 黄瓜削皮：在削去可变形蔬菜的皮时保持稳定接触，要求持续的剪切力控制。该任务要求对工具滑动过程中摩擦和变形的微小变化具备敏感性。 • 白板擦拭：使用刚性白板擦擦拭平坦或倾斜的表面，要求持续接触和精确的法向力调节，以防止抖动和 lifted-off。 为了评估，我们使用双摄像头设置和GelSight传感器收集了这些任务的真实世界视触觉数据集（图3b）。该数据集包含100个薯片拾取与放置轨迹、105个白板擦拭轨迹以及61个削皮轨迹。所有示范通过手动遥控收集，包含同步的多视图RGB流、触觉变形图像和机器人状态信息。

# 4.3. 定量结果 (Q1)

为了评估，我们在每个任务上进行20次试验。任务包括芯片抓取与放置、使用刚性橡皮擦进行平面白板擦拭、倾斜白板擦拭和黄瓜剥皮。总的来说，每个模型在80次真实世界试验中进行评估，推理频率为 $1 \ \mathrm { H z }$。表1报告了接触丰富的操作任务的性能比较。总体而言，VTAM显著优于基线，在芯片抓取与放置、黄瓜剥皮和白板擦拭任务上的成功率分别达到了 $9 0 \%$、$8 5 \%$ 和 $9 5 \%$。

Table 1: Overall performance comparison.   

<table><tr><td>Model</td><td>Chip</td><td>Peel</td><td>Wipe</td></tr><tr><td>Genie Envisioner</td><td>0%</td><td>0%</td><td>2.5%</td></tr><tr><td>π0.5 (Vision)</td><td>10%</td><td>0%</td><td>0%</td></tr><tr><td>π0.5 + Tactile</td><td>5%</td><td>0%</td><td>0%</td></tr><tr><td>VTAM (Ours)</td><td>90%</td><td>85%</td><td>95%</td></tr></table>

在芯片取放任务中，VTAM的成功率达到了90%，展现了在脆弱物体操作中的强大鲁棒性，尤其是在需要精确抓取验证和力控制时。相比之下，基线方法经常无法检测到不成功的抓取而直接进入放置阶段。在黄瓜剥皮任务中，VTAM的成功率达到85%，而所有基线方法均未能完成任务， confirming 触觉反馈在与可变形物体接触时，对维持稳定接触和调节剪切力至关重要。在白板擦拭任务中，VTAM在平面和倾斜表面上的成功率达到了95%，而基线方法要么施加不稳定的接触力，要么未能保持一致的表面跟随。这些结果强调了视觉触觉世界动作建模在接触丰富的操作中对强力调节的重要性。

# 4.4. 质性示例和失败模式分析

图4展示了三个任务的定性比较。我们分析不同方法的表现，以理解VTAM是如何应对接触丰富任务中的挑战的。有关更多示例，请参见附录。芯片拾取和放置。在仅使用视觉的G基线中，主要失败源于无法验证成功的抓取。机器人经常在芯片上方关闭夹持器，即使没有有效的触觉信号整合，也无法进行正确的基于力量的抓取。相比之下，VTAM展现出触觉感知行为。机器人仅在触觉变形确认成功接触时才抓取芯片，并保持稳定的夹持器宽度以防止在提起时掉落。如果抓取失败，策略还能够在提起期间检测到触觉信号的缺失，并立即返回芯片重新尝试抓取，而不是继续移动到盘子上并释放夹持器。黄瓜剥皮。GE基线和两个Pi变体展现出类似的运动模式。从黄瓜的左侧开始，工具首先朝中心线移动，然后在沿表面滑动时远离它。这条轨迹类似于一种视觉驱动的策略，试图根据物体的曲率进行跟随，而不是调节接触力。因此，工具频繁与黄瓜表面失去接触。相比之下，我们的VTAM策略在沿表面移动时建立了稳定的接触并保持适当的力量。即使黄瓜厚度变化，机器人也能够在同一位置进行重复的剥皮动作，准确感知接触状态。白板擦拭。在平坦和倾斜的白板上，这两个Pi变体施加了过大的接触力，有时甚至会将用于支撑倾斜白板的书推得乱七八糟。这种行为可能源于训练数据同时包含平坦和倾斜的表面，使得模型仅从视觉观察中推断出正确的末端执行器高度变得困难。因此，策略无法可靠地判断夹持器何时应向下移动以跟随较低的表面，或者保持较高的位置以适应倾斜的平面，而是通过施加过大的力量来维持接触。在倾斜板上，行为变得不稳定，末端执行器的运动变得不规则。策略倾向于跟随平坦或倾斜轨迹，导致末端执行器过度接触倾斜板的高区域，而在表面高度变化时未能保持稳定接触。相比之下，VTAM在平坦和倾斜表面上保持适度且稳定的接触力量，实现一致的擦拭和有效的污渍去除。这表明VTAM能够有效利用触觉信息来处理视觉上模糊的接触丰富任务。

![](images/4.jpg)  

Figure 4: Qualitative comparison between VTAM and baseline methods on real-world manipulation tassTop:Chippick-an-plaVisin-y baselne l detere whethehe hias beeuy raspe and proc the placeent tage even when he rasp fails.Mid:ucmber peelig int s. Baselines tend to follow a vision-driven trajectory that approaches the center of the cucumber but fail to maintain consistent contact with the surfae, indicating poor forc regulation and lackof contac awareness. Bot:Whiteboard wipng under varying heights and tilt angles. Baselines exhibit unstable wiping behaviors, oe applyieithesuient  xcesivel largorce, particularyntilteurfacs. In cntrast, VTAM maintains stable contact and appropriate force regulation across all tasks, enabling robust manipulation behaviors. Red boxes highlight representative failure cases of baselines.

![](images/5.jpg)  

Figure 5: Prediction visualization of the backbone video model. From top to bottom: Camera-1 viev Camera-2 view, Tactile stream prediction. Ground-truth (top rows) and VTAM predictions (bottom rows).

# 4.5. 预测可视化

我们在图5中可视化了主干视频模型的预测结果。对于每个视角和触觉预测，顶部行展示了真实帧，而底部行展示了模型预测。模型保持了跨视角的时间一致性和视内动态，仅在与操作无关的细节上有轻微模糊，表明其在动作生成方面具备可靠的视觉—触觉世界建模能力。

Table 2: Ablation study on the Chip Pick-and-Place task (10 trials per variant).   

<table><tr><td>Model Variant</td><td>Tactile Integration</td><td>Success Rate</td></tr><tr><td>Vision-only (No Tactile)</td><td>None</td><td>0%</td></tr><tr><td>Late-Fusion Tactile</td><td>Downstream Only</td><td>0%</td></tr><tr><td>No Virtual-Force Reg.</td><td>Joint Latent</td><td>10%</td></tr><tr><td>VTAM (Ours)</td><td>Hierarchical World Model</td><td>90%</td></tr></table>

# 4.6. 芯片拾取与放置任务的消融研究（Q2 & Q3）

为了评估 VTAM 的架构组件，我们在受限的 $1 \ \mathrm{H z}$ 推理频率下对接触敏感的芯片拾取与放置任务进行消融实验（见表 2）。完整的 VTAM 模型实现了 $90 \%$ 的成功率，通过预测世界建模来预判低频传感器数据之间的交互状态。相比之下，消融实验未能可靠地捕捉细微的接触变化。视觉基线模型的成功率为 $0 \%$，因为在最终靠近过程中视觉深度估计受到严重遮挡，无法感知微妙的接触转换。仅在动作头引入触觉信号（晚融合）同样导致 $0 \%$ 的成功率（Q2），表明没有我们的层次化视觉触觉世界建模，单纯的力输入是不够的。最后，去除虚拟力正则化（无正则化）使性能下降至 $10 \%$，这是由于“视觉模态主导”（Q3），确认了该辅助损失对防止表示崩溃和确保触觉信号影响整个去噪过程至关重要。

# 5. 结论

我们介绍了VTAM，一种用于接触丰富操控的视觉触觉世界动作模型。VTAM训练一个预测主干网络，以建模多视角视频和高分辨率触觉信号的联合演变，从而使策略能够使用在视觉上观察较弱或被遮挡的接触动态。这种预测形式学习时间上连续的接触特征，而不需要显式标签或接触事件，并且避免单纯依赖下游反应性的触觉融合。为了防止行动训练默认于视觉线索并压制触觉信息的常见失败模式，我们添加了一个基于形变预测的虚拟力预测目标，通过控制路径保持触觉监督。在要求精确力调节的真实机器人任务中，包括芯片的拾取和放置以及黄瓜的剥皮，VTAM显著提高了成功率和交互稳定性，相比于仅依赖视觉的 naive 触觉基线，突显了直接建模接触动态对于可靠物理交互的重要性。最终，我们的框架提供了一种可扩展的、基于物理的具身智能方法，证明了预测联合建模对于复杂物理交互中的可靠执行至关重要。

# References

[16] Jialei Huang, Shuo Wang, Fanqi Lin, Yihang Hu, Chuan Wen, and Yang Gao. Tactile-VLA: Unlocking vision-language-action model's physical knowledge for tactile generalization. arXiv preprint arXiv:2507.09160, 2025.

[17] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. OpenVLA: An open-source vision-languageaction model. arXiv preprint arXiv:2406.09246, 2024.

[18] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 2013.

[19] Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang, Shengcong Chen, Yuxin Jiang, Yue Hu, Jingbin Cai, Si Liu, Jianlan Luo, et al. Genie envisioner: A unified world foundation platform for robotic manipulation. arXiv preprint arXiv:2508.05635, 2025.

[20 Jason Jingzhou Lu, Yulong Li, Kennh Shaw, Tony To, Ruslan Salakhudinov, and Deepak Pathak. acr: Force-attending curriculum training for contact-rich policy learning. arXiv preprint arXiv:2502.17432, 2025.

[21] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprin arXiv:1711.05101, 2017.

[22] Amit Parag, Edward H Adelson, and Ekrem Misimi. Learning incipient slip with gelsight sensors: Attention classification with video vision transformers. In International Conference on Intelligent Robots and Systems (IROS), pages 1396013966, 2024.

[23] Jeongeun Park, Jihwan Yoon, Byungwoo Jeon, Juhan Park, Jinwoo Shin, Namhoon Cho, Kyungjae Lee, Sangdoo Yun, and Sungjoon Choi. Hierarchical vision language action model using success and failure demonstrations. arXiv preprint arXiv:2512.03913, 2025.

[24] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. pi0.5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

[25] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 116. IEEE, 2020.

[26] Younggyo Seo, Danijar Hafner, Hao Liu, Fangchen Liu, Stephen James, Kimin Lee, and Pieter Abbeel. Masked world models for visual control. In Conference on Robot Learning (CoRL), pages 13321344, 2023.

[27] Zilin Si, Gu Zhang, Qingwei Ben, Branden Romero, Zhou Xian, Chao Liu, and Chuang Gan. Difftactile: A physics-based differentiable tactile simulator for contact-rich robotic manipulation. arXiv preprint arXiv:2403.08716, 2024.

[28] Lin Sun, Bin Xie, Yingfei Liu, Hao Shi, Tiancai Wang, and Jiale Cao. Geovla: Empowering 3d represer tations in vision-language-action models. arXiv preprint arXiv:2508.09071, 2025.

[29] Balakumar Sundaralingam, Alexander Sasha Lambert, Ankur Handa, Byron Boots, Tucker Hermans, Stan Birchfield, Nathan Ratliff, and Dieter Fox. Robust learning of tactile force estimation through robot interaction. In International Conference on Robotics and Automation (ICRA), pages 90359042, 2019.

[30] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.

[31] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In International Conference on Machine Learning (ICML), pages 10961103, 2008.

[32] Qiang Wang, Pablo Martinez Ulloa, Robert Burke, David Cordova Bulens, and Stephen J Redmond. Robust learning-based incipient slip detection using the papillarray optical tactile sensor for improved robotic gripping. IEEE Robotics and Automation Letters, 9(2):18271834, 2023.

[33] Weiyao Wang, Du Tran, and Matt Feiszli. What makes training multi-modal classification networks hard? In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1269512705, 2020.

[34] Nan Wu, Stanislaw Jastrzebski, Kyunghyun Cho, and Krzysztof J Geras. Characterizing and overcoming the greedy nature of learning in multi-modal deep neural networks. In International Conference on Machine Learning (ICML), pages 2404324055, 2022.

[35] Zhengtong Xu, Raghava Uppuluri, Xinwei Zhang, Cael Fitch, Philip Glen Crandall, Wan Shou, Dongyi Wang, and Yu She. Unit: Data efficient tactile representation with generalization to unseen objects. IEEE Robotics and Automation Letters, 2025.

[36] Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, and Cewu Lu. Reactive diffusion policy: Slow-fast visual-tactile policy learning for contact-rich manipulation. arXiv preprint arXiv:2503.02881, 2025.

[37] Fengyu Yang, Chao Feng, Ziyang Chen, Hyoungseob Park, Daniel Wang, Yiming Dou, Ziyao Zeng, Xien Chen, Rit Gangopadhyay, Andrew Owens, et al. Binding touch to everything: Learning unified multimodal tactile representations. In IEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2634026353, 2024.

[38] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel Urtasun. Unisim: A neural closed-loop sensor simulator. In IEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 13891399, 2023.

[39] Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

[40] Jiawen Yu, Hairuo Liu, Qiaojun Yu, Jieji Ren, Ce Hao, Haitong Ding, Guangyu Huang, Guofan Huang, Yan Song, Panpan Cai, et al. Forcevla: Enhancing vla models with a force-aware moe for contact-rich manipulation. arXiv preprint arXiv:2505.22159, 2025.

[41] Wenzhen Yuan, Siyuan Dong, and Edward H Adelson. Gelsight: High-resolution tactile sensors fo: perceiving physical properties. IEEE Robotics & Automation Magazine, 24(3):6677, 2017.

[42] Brayan S Zapata-Impata, Pablo Gil, and Fernando Torres. Learning spatio temporal tactile feature with a convlstm for the direction of slip detection. Sensors, 19(3):523, 2019.

[43] Chaofan Zhang, Peng Hao, Xiaoge Cao, Xiaoshuai Hao, Shaowei Cui, and Shuo Wang. VTLA: Visiontactile-language-action model with preference learning for insertion manipulation. arXiv preprint arXiv:2505.09577, 2025.

[44] Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, Fan Lu, He Wang, et al. DreamVLA: a vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447, 2025.

[45] Zongzheng Zhang, Haobo Xu, Zhuo Yang, Chenghao Yue, Zehao Lin, Huan-ang Gao, Ziwei Wang, and Hao Zhao. Ta-vla: Elucidating the design space of torque-aware vision-language-action models. arXiv preprint arXiv:2509.07962, 2025.

[46] Haoyu Zhen, Xiaowen Qiu, Peihao Chen, Jincheng Yang, Xin Yan, Yilun Du, Yining Hong, and Chuang Gan. 3d-vla: A 3d vision-language-action generative world model. arXiv preprint arXiv:2403.09631, 2024.

[47] Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, and Abhishek Gupta. Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets. arXiv preprint arXiv:2504.02792, 2025.

[48] Yifan Zhu, Mei Hao, Xupeng Zhu, Quentin Bateux, Alex Wong, and Aaron M Dollar. Forces for free: Vision-based contact force estimation with a compliant hand. Science Robotics, 10(103):eadq5046, 2025.

[49] Briana Zitkovich, Tianhe u, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohhart, Stefan Welker, Ayzaan Wahid, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), pages 21652183, 2023.

# A. Training Details

VTAM World Model. The VTAM model is trained in two stages on $4 \times$ NVIDIA A100 GPUs (40 GB VRAM each) using DeepSpeed ZeRO Stage 2 [25] with bf16 mixed precision. In Stage 1: Video-only pre-training, the video prediction backbone is initialized from a pre-trained Genie Envisioner (GE-base) checkpoint [19], an LTX-Video transformer [10] with 28 layers, 32 attention heads, and hidden dimension 2048, and fine-tuned for 50,000 steps using the video-only objective (train_mode $=$ video_only). We set $B \ = \ 2 8$ following the default configuration of the pretrained LTx-Video backbone. This choice ensures compatibility with the pretrained architecture and maintains stable training behavior. We employ AdamW [21] $( \beta _ { 1 } = 0 . 9$ $\beta _ { 2 } = 0 . 9 5$ weight decay ${ { 1 0 } ^ { - 5 } } \cdot$ a cnstn learng a $3 \times { 1 0 } ^ { - 4 }$ after 1,000 warmup steps, gradient clipping $( \lVert \nabla \rVert = 1 . 0 )$ , and a per-GPU batch size of 16. In Stage 2: Action head training, an action expert head is appended to the frozen video backbone. The action expert is implemented as a parallel Transformer branch consisting of 28 layers, mirroring the depth of the video backbone. Each layer contains (i) a selfattention module over action-state tokens, (ii) a cross-attention module attending to the corresponding layer' videohidden states, and ii a feed-forwardnetwork. All modules are modulated using adaptive ayer noralization (AdaL) conditioned on the diffusion timestep. This stage is tained for 20,000 steps using the action-full objective with a lower learning rate of $5 \times { 1 0 } ^ { - 5 }$ (constant with 1,000 warmup steps), while keeping all other hyperparameters identical to Stage 1. Training is performed using Flow Matching with the Euler Discrete Scheduler, achieving approximately 3.4s per optimization step. For the chip pick-and-place, cucumber peeling, and whiteboard wiping tasks, video inputs are resized to $1 9 2 \times 2 5 6$ with a temporal chunk frames and an actin chunk siz .Actins arerepresent in bsolute joit space . te joint positions rather than deltas) and normalized using pre-computed per-dimension statistics. We apply caption dropout $( p = 0 . 0 6 )$ and first-frame noise injection (scale 0.1) for regularization.

Optimization Details. We set the total loss coefficients to $\lambda _ { 1 } = \lambda _ { 2 } = 1$ for all experiments. Since all three objectives share the same flow-matching formulation, implemented as mean squared error on predicted velocity fields in normalized latent space, their magnitudes remain on comparable scales. Therefore, equal weighting provides a stable and straightforward choice without introducing additional hyperparameters.

GE-Act Baseline. The Vision-only GE-Act baseline [19] follows the same two-stage training protocol as VTAM. In Stage 1, the LTX-Video transformer backbone is initialized from the GE-base checkpoint and fine-tuned for 50,000 steps using the video-only objective. We employ AdamW $\beta _ { 1 } = 0 . 9$ $\beta _ { 2 } = 0 . 9 5$ , weight decay ${ { 1 0 } ^ { - 5 } } )$ with a constant learning rate of $3 \times { 1 0 } ^ { - 4 }$ after 1,000 warmup steps, gradient clipping $( \lVert \nabla \rVert = 1 . 0 )$ , and a per-GPU batch size of 16. In Stage 2, a randomly initialized action expert head is appended and trained for 20,000 steps using the action-full objective with a lower learning rate of $5 \times { 1 0 } ^ { - 5 }$ (constant with 1,000 warmup steps). All remaining settings, including video resolution $( 1 9 2 \times 2 5 6 )$ , temporal chunk size (9 frames), action chunk size (54), bf16 precision, DeepSpeed ZeRO Stage 2, and Flow Matching with the Euler Discrete Scheduler, are identical to those used in VTAM training.

$\pi _ { 0 . 5 }$ Policy. The $\pi _ { 0 . 5 }$ policies [24] are fine-tuned from a pre-trained $\pi _ { 0 . 5 }$ base checkpoint on $4 \times$ NVIDIA A100 GPUs (40 GB VRAM each) using Fully Sharded Data Parallel (FSDP) with bfloat16 mixed precision. We optimize the models using AdamW $( \beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } = 0 . 9 5 )$ with a peak learning rate of $2 . 5 \times { { 1 0 } ^ { - 5 } }$ following a cosne decy hedule wih 1,000 wrmup seps, deyn o $2 . 5 \times { { 1 0 } ^ { - 6 } }$ over 30,000 steps. Gradient clipping is applied with $\Vert \nabla \Vert = 1 . 0$ . The global batch size is set to 64, and we use an exponential moving average (EMA) with decay 0.999. The action dimension is 32 with an action horizon of 50. Input images are resized to $2 2 4 \times 2 2 4$ .States and actions are normalized using quantile normalization with pre-computed dataset statistics. All task-specific models are trained for 10,000 optimization steps.

# B. Experimental Details

Evltn roWe valuat a poic usigl-oruc rat over muliplndpenent ils. For all tasks, the robot executes the policy under randomized initial conditions, and success is determined according to task-specific criteria.

Chip Pick-and-Place.The polic is evaluated over 20 consecutive trials. In each trial, the potato chip is pl t a rndm nitl poitn. Therobot mus mov abov he potao hi, gras it ihout dme, p drops it during transport.

Whiteboard Wiping. We evaluate the wiping task under two board inclination settings: $0 ^ { \circ }$ (flat) and $4 5 ^ { \circ }$ icn.Aranom black stain irawat he tar  ac t.The robot is aloweat most vep mos e e leolh a ioser . Each setting is evaluated over 20 trials.

Cucuber Peeling. For the peeling task, the robot performs 20 consecutive motions at a fixed cutting posi As the cucumber is peeled, its height decreases, requiring the robot to dynamically adjust contact forc

As shown in Fig. 6, 17 out of 20 trials ( $8 5 \%$ success rate) produced peel strips longer than $_ { 1 0 \mathrm { c m } }$ , demonstrating the VTAM's ability to maintain stable contact despite changing geometry.

# C. Video Prediction Examples

We visualize qualitative video prediction results for two contactrich manipulation tasks: cucumber peeling and whiteboard wiping. To evaluate visual fidelity, we compare predicted frames against ground truth for the rear camera (Fig. 7 and Fig. 10) and front camera (Fig. 8 and Fig. 11). Furthermore, we assess the model's ability to anticipate contact dynamics by predicting tactile streams (Fig. 9 and Fig. 12). Yellow arrows in the tactile plots visualize estimated contact force magnitude and direction. Note that these are for visualization purposes only; the model processes raw tactile tokens without explicit force inputs. Results demonstrate that VTAM effectively captures both visual motion and fine-grained contact dynamics across modalities.

![](images/6.jpg)  
Figure 6: Qualitative peeling results. VTAM achieves an $8 5 \%$ success rate (17/20 trials), producing peel strips longer than $1 0 \mathrm { c m }$ in successful runs.

![](images/7.jpg)  
Figure 7: Cucumber peeling video prediction (rear camera view). Top row: ground-truth; Bottom row model predictions. The predicted frames closely match the ground-truth observations.

![](images/8.jpg)  
Figure 8: Cucumber peeling video prediction (front camera view). The model maintains consistenc: with the ground truth across the manipulation sequence.

![](images/9.jpg)  
Figure 9: Cucumber peeling tactile prediction. Tactile frames are predicted accurately. Yellow arrow visualize estimated forces for interpretation only.

![](images/10.jpg)  
Figure 10: Whiteboard wiping video prediction (rear camera view). Top row: ground-truth; Bottom row model predictions. Predictions match the ground truth across the wiping motion.

![](images/11.jpg)  
Figure 11: Whiteboard wiping video prediction (front camera view). The model successfull reproduce the visual dynamics of the wiping interaction.

![](images/12.jpg)  
Figure 12: Whiteboard wiping tactile prediction. The predicted tactile frames capture the interactio oetween the wiper and the board surface.