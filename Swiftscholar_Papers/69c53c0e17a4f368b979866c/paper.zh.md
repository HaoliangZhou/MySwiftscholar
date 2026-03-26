# 从完美环境到嘈杂上下文：缓解语音大语言模型中的上下文曝光偏差

小勇 郭1，南杰 李1，子杰 曾3，凯 王1，浩 黄*1,2，海华 许3，伟 施3 1 新疆大学计算机科学与技术学院，中国 乌鲁木齐 2 丝绸之路多语种认知计算联合国际研究实验室，中国 乌鲁木齐 3 Timekettle 通信联系人：hwanghao@gmail.com

# 摘要

基于对话上下文的自动语音识别（ASR）与语音大型语言模型（Speech-LLMs）通常使用真实的对话历史进行训练，但在推断时依赖易出错的历史信息，造成我们称之为上下文曝光偏差的训练与测试不匹配。我们提出一个统一的训练框架，以提高在真实历史下的鲁棒性：(i) 使用Whisper大模型v3假设作为训练期间历史的教师错误知识，(ii) 通过上下文丢弃来规范对历史的过度依赖，(iii) 在精心挑选的失败案例上进行直接偏好优化（DPO）。在TED-LIUM 3（领域内）和零-shot LibriSpeech（领域外）上的实验表明，在预测历史解码下取得了一致的提升。在以两个发言历史作为上下文的情况下，Whisper假设的SFT将字错误率（WER）从$5.59\%$（使用真实历史训练）降低至$5.47\%$，而DPO进一步改善至$5.17\%$。在无关上下文攻击下，DPO导致的降幅最小（$5.17\% \to 5.63\%$），表明对误导性上下文的鲁棒性得到了改善。我们的代码和模型已发布在https://github.com/XYGuo1996/Contextual_Speech_LLMs。

# 1 引言

自动语音识别（ASR）迅速发展（Ma et al., 2025），从传统架构如CTC（Graves et al., 2006）、AED（Chan et al., 2016）和RNN-T（Graves et al., 2013）演变到强大的预训练模型，如HuBert（Hsu et al., 2021）、Whisper（Radford et al., 2023）和Whistle（Yusuyin et al., 2025；Li et al., 2022）。尽管最近出现的Speech-LLMs（Cui et al., 2024）进一步扩展了多模态能力，有效地融入了上下文线索——这一问题被称为上下文ASR（Aleksic et al., 2015；Hall et al., 2015）——在这些范式中仍然是一个关键挑战，以弥补模糊的音频证据。

在传统的自动语音识别（ASR）系统中，利用上下文信息主要遵循两种核心范式：浅融合和深融合（Fang et al., 2025）。浅融合本质上作为一种推理阶段的集成策略，通过利用外部语言模型对声学模型生成的假设进行评分，从而提高特定上下文中的识别准确性（McDermott et al., 2019；Ravi et al., 2020；Guo et al., 2023）。相对而言，深融合涉及更根本的架构创新。这种方法将上下文信息编码为向量嵌入，并直接集成到端到端模型的内部组件中。因此，这使得模型能够在训练阶段学习声学特征和上下文线索的联合表示（Toshniwal et al., 2018；Wang et al., 2024；Tang et al., 2024；Huang et al., 2024b,a；Kolokolov et al., 2024；Shi et al., 2024；Sudo et al., 2024）。例如，上下文 RNN-T（Jain et al., 2020）展示了关注元数据嵌入在稀有词识别中的有效性，而Hou et al.（Hou et al., 2022）则通过将对话历史整合到流式 RNN-T 编码器中进行了扩展。Hori et al.（Hori et al., 2020）探讨了基于变换器的架构用于长上下文建模，强调了跨话语依赖的重要性。

得益于大型语言模型（LLMs）强大的上下文推理能力，最近的研究优先将上下文线索嵌入到语音LLMs的提示中（Chen et al., 2024; Yang et al., 2024; Shen et al. 2025; Lakomkin et al., 2024; Cheng, 2024; Lei et al., 2025; Koshkin et al., 2024; Fang et al., 2025; Gong et al., 2024; Zhou et al., 2025）。方法多样，从使用文本元数据如标题和描述（Lakomkin et al., 2024）到结合特定实体列表（Chen et al., 2024）和多模态辅助输入（Yang et al., 2024）不等。值得注意的是，Lakomkin等人（Lakomkin et al., 2024）还检查了模型对上下文干扰的抗扰动能力。基于这一趋势，检索增强生成（RAG）被提出作为一种新方法，以集成语音LLMs（Shen et al., 2025; Li et al., 2024; Mu et al., 2025; Gourav et al., 2024）。另一方面，Step-Audio采用基于文本的上下文管理器来保持对话历史并支持多轮交互（Huang et al., 2025）。本文未单独分析历史文本对系统性能的影响。我们研究逐句上下文自动语音识别（ASR），其中每个发言都使用先前轮次的转录文本作为文本历史顺序解码。在训练阶段，模型可以依赖于oracle历史，但在实际应用中，oracle历史不可用，模型必须基于易出错的ASR假设进行条件化，这在训练时和推理时上下文之间产生了分布差异。我们称这种上下文通道的不匹配为上下文暴露偏差。尽管已经针对利用不完美上下文的挑战进行了探讨——例如，通过语音翻译中的上下文丢失（context dropout）（Hussein et al., 2024）或对话ASR中的噪声表示学习（Lee et al., 2024），但这些方法往往依赖于隐式正则化或辅助模块，缺乏直接的机制将模型的生成偏好与显式拒绝上下文错误对齐。为了减轻连续发言ASR中的暴露偏差，我们提出了一种统一的训练框架，结合三种互补策略。教师错误知识：我们不使用真实历史，而是将Whisper large-v3解码假设作为训练上下文，使模型接触到真实的“教师错误”，更好地匹配推理时的条件。上下文丢失：我们随机以固定概率屏蔽历史上下文，减少对文本历史的过度依赖，鼓励以声学为中心的转录，从而缓解历史过拟合。直接偏好优化（DPO）：我们通过从选择的困难负样本构建偏好对，进一步优化上下文生成，明确使模型避免负面行为，提高输出质量。本文的主要贡献总结如下：• 利用上下文暴露偏差改善语音LLMs ASR：我们将上下文暴露偏差识别为上下文条件语音LLMs基础ASR的关键失败模式，并明确利用这种训练与测试不匹配作为设计训练和对齐策略的指导原则，以增强在不完美历史上下文下的识别性能。• 针对不完美上下文的噪声感知训练：我们提出统一的训练框架，通过(i) 使用强大的教师ASR系统生成现实的、易出错的历史上下文来对齐训练与推理（在我们的实验中以Whisper large-v3为实例），(ii) 施用上下文丢失以正则化对历史的依赖，以及(iii) 使用基于上下文失败案例构建的偏好对进行DPO，以减少错误放大。• 在现实和跨领域评估下的稳健性：使用强大教师生成的历史上下文进行训练大幅缩小了oracle推理差距，DPO通过降低模型对不相关或错误上下文的敏感性进一步提高了稳健性，在现实解码条件下始终获得最佳的领域内和跨领域性能。

# 2 方法

# 2.1 发话级上下文自适应语音识别

我们考虑一个基于话语级别的上下文自动语音识别（ASR）设置，其中每个话语是顺序识别的，前面话语的转录作为当前解码的文本上下文。形式上，输入流被划分为一系列话语 $\{ X_{t} \}_{t=1}^{T}$，其中 $X_{t}$ 表示第 $t$ 个话语的声学信号，而 $Y_{t}$ 表示其参考转录。对于每个话语 $t$，识别器在当前声学 $X_{t}$ 和可用上下文 $C_{t}$ 的条件下生成一个假设 $\hat{Y}_{t}$：

$$
\hat { Y } _ { t } \sim p _ { \theta } ( Y | X _ { t } , C _ { t } )
$$

我们将文本上下文 $C _ { t }$ 定义为先前发言转录的函数。在最简单的情况下，$C _ { t }$ 是最近 $N$ 个发言级别转录的串联：

$$
C _ { t } = c o n c a t ( S _ { t - N } , \cdot \cdot \cdot , S _ { t - 1 } )
$$

其中 $S _ { i }$ 是用于语句 $i$ 的历史记录的文字记录，$N$ 控制上下文窗口大小（$N = 0$ 对应于无上下文基线）。

![](images/1.jpg)  

Figure 1: Model architecture

这个设置的核心方面是，在推断时可用的历史转录本的质量可能与训练期间使用的转录本不同。在许多实验设置中，$S _ { i }$ 被视为预言转录本 $Y _ { i }$，从而产生了预言上下文条件。然而，在实际部署中，系统无法访问 $Y _ { i }$，而必须依赖于上游对先前回合的自动生成假设。我们通过设定 $S _ { i } = \hat { Y } _ { i }$ 来表示这种预测上下文条件，其中 $\hat { Y _ { i } }$ 可能包含相对于参考 $Y _ { i }$ 的识别错误。这种差异在训练和推断之间引起了条件上下文的分布不匹配：训练通常基于预言历史进行条件化，而推断则基于不完美的、模型生成的历史进行条件化。在本工作中，我们将这种上下文通道中的训练—测试不匹配称为上下文暴露偏差。重要的是，这是一个上下文层面的不匹配：即使模型使用标准的教师强制进行当前话语的训练，在推断时提供给模型的历史上下文可能会出错，并且可能系统性地与训练中使用的预言上下文不同。在论文的其余部分，我们研究这种上下文暴露偏差如何影响语音大语言模型中的上下文自动语音识别，并开发更好地将训练时上下文与推断时条件对齐的训练策略。

# 2.2 模型架构

我们采用一个训练框架，包括一个现成的语音编码器、一个冻结的LLM主干和一个可训练的MLP投影器用于模态适配。我们使用低秩适配（LoRA）（Hu et al., 2021）对LLM进行微调，分为两个阶段：(i) 一个专门用于SFT的SFT LoRA模块，以及(ii) 一个单独的用于偏好对齐的DPO LoRA模块。根据(Jiang et al., 2025)，DPO阶段引入了一个额外的独立LoRA模块，而非重用SFT适配器，从而实现偏好优化的独立控制。整个架构如图1所示。

# 2.3 训练方法论

我们研究了在现实的训练—测试差异下的上下文自动语音识别（ASR）：在训练过程中，通常提供的上下文历史是作为标准答案的转录文本，而在推理时，历史必须通过上游ASR输出获得，因此包含识别错误。为了弥补这一差距，我们提出了一个训练框架，(i) 用教师生成的假设替换标准答案历史，以模拟训练过程中的不完美上下文，其中教师是一种强大的现成ASR系统（在我们的实验中实现为Whisper large-v3），以及(ii) 对模型对历史的依赖进行正则化，以防止在上下文嘈杂或误导时的错误放大。

# 2.3.1 教师错误知识

为了近似推理时的条件，我们引入了教师错误知识，其中历史转录 $\hat { Y }$ 取自 Whisper large-v3 解码的假设，而不是始终使用理想历史。具体来说，我们使用 Whisper 对训练集进行预解码，并存储假设 $\hat { Y } _ { i } ^ { W h i s p e r }$ 。在训练过程中，我们使用 Whisper large-v3 解码的假设作为历史来源构建 $C _ { t }$ 。这样使模型暴露于真实的上下文噪声，并促进对历史错误的鲁棒性。我们优化基于构建的上下文的标准序列负对数似然：

$$
\mathcal { L } _ { S F T } = - \sum _ { t } \log p _ { \theta } ( Y _ { t } | X _ { t } , C _ { t } , P )
$$

其中 $P$ 是任务提示，$\theta$ 表示可训练参数。

# 2.3.2 上下文丢弃

即使在嘈杂的历史数据中，模型仍可能过度信任上下文文本，并在历史数据误导时变得脆弱。为了进一步规范上下文的使用，我们在训练过程中应用上下文丢弃（Context Dropout）：以概率 $p _ { d r o p }$ 掩盖文本历史 $C _ { t }$，同时保持当前语句 $X _ { t }$ 不变：

$$
\tilde { C } _ { t } = \left\{ \begin{array} { l l } { { \emptyset , } } & { { \mathrm { w i t h ~ } p _ { \mathrm { d r o p } } , } } \\ { { C _ { t } , } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right.
$$

然后根据 $\tilde { C } _ { t }$ 计算 SFT 目标。该机制防止模型依赖历史作为捷径，并鼓励模型保留强有力的语音信息，同时在有上下文时仍然能够受益于上下文。

# 2.3.3 基于 DPO 的困难负样本偏好优化

虽然带有噪声历史的SFT提升了整体鲁棒性，但我们观察到仍然可以触发一小部分失败案例。为了明确抑制这种不良行为，我们进一步在精心挑选的困难负样本上应用DPO。对于每个选择的发话，我们在相同输入下构造一个偏好对 $( Y ^ { + } , Y ^ { - } )$，其中 $\mathbf { X } =$ $( X _ { t } , C _ { t } , P )$，$Y ^ { + }$ 是真实标注，$Y ^ { - }$ 是模型的推理转录。DPO 优化策略，使得 $Y ^ { + }$ 的可能性高于 $Y ^ { - }$，而无需明确的奖励模型。根据标准 DPO 公式，我们最小化：

$$
\Delta _ { \theta } = \log \pi _ { \theta } ( Y ^ { + } \mid \mathbf { X } ) - \log \pi _ { \theta } ( Y ^ { - } \mid \mathbf { X } )
$$

$$
\Delta _ { r } = \log \pi _ { r } ( Y ^ { + } \mid \mathbf { X } ) - \log \pi _ { r } ( Y ^ { - } \mid \mathbf { X } )
$$

$$
m = \beta \left( \Delta _ { \theta } - \Delta _ { \mathrm { r } } \right)
$$

$$
\mathcal { L } _ { \mathrm { D P O } } = - \log \sigma ( m )
$$

在我们的实验中，我们将参考策略 $\pi _ { r }$ 设置为 SFT 检查点（冻结），$\beta$ 是一个温度系数，用于控制偏好增强的强度，在本研究中，我们使用 $\beta = 0 . 1$，而 $\sigma ( \cdot )$ 是 Sigmoid 函数。这里，$\Delta _ { \theta }$ 和 $\Delta _ { r }$ 分别衡量在当前策略 $\pi _ { \theta }$ 和参考策略 $\pi _ { r }$ 下优选和劣选响应之间的对数似然差异，而 $m$ 缩放它们的差异以形成 DPO 目标（Ouyang 等，2022；Rafailov 等，2023）。在我们的实现中，我们为 DPO 使用一个独立的 LoRA 模块，以将偏好优化与主要的 SFT 适应解耦（Jiang 等，2025），如图 1 所示。

# 3 实验设置

# 3.1 模型与模块

我们在内部开发的语音大语言模型训练框架上构建我们的系统。我们采用Whisper large-v3（Radford等，2023），丢弃解码器，仅使用编码器作为特征提取器。基于之前在语音大语言模型上的研究发现，经过微调的LLM在语音识别任务中显著优于普通预训练模型（Ma等，2025），我们选择vicuna-7Bv1.5（Zheng等，2023）作为系统中的大语言模型，并使用LoRA适配器。除非另有说明，我们保持Whisper large-v3编码器的冻结状态，并与LoRA参数一起训练一个轻量级多层感知机投影器。我们报告的主要指标是词错误率（WER）。

# 3.2 数据集

我们在TED-LIUM 3（Hernandez等，2018）数据集上进行utterance级别的分割训练和领域内评估。每个utterance与其参考转录文本配对，并从同一会话中的前一utterance构建上下文历史。我们使用官方划分进行训练、开发和测试。为了评估跨领域泛化能力，我们在LibriSpeech（Panayotov等，2015）上进行评估，而不进行任何额外训练。我们在testclean和test-other上报告结果。

# 3.3 训练协议

我们将系统训练分为三个阶段。所有实验均在四个 NVIDIA A100 GPU 上进行。在第一阶段，我们通过冻结 Whisper large-v3 编码器和 Vicuna v1.5-7B，训练一个无上下文的基础模型，仅优化 MLP 投影器，并采用标准的语音识别（ASR）目标。这一阶段建立了语音与文本之间的可靠对齐，为后续的上下文训练提供了稳定的初始化。此阶段的批量大小设为 6，学习率设为 1e-4，预热步骤设为 1000。在第二阶段，我们在话语级上下文 ASR 设置下，使用配对的语音文本数据对 Speech-LLMs 进行示范微调（SFT）。在 SFT 期间，我们用这些 Whisper 假设替换了 oracle 历史，以便将模型暴露于历史噪声中。重要的是，Whisper 仅在离线状态下用于构建教师错误知识以进行训练；我们的推理不需要 Whisper 作为辅助组件。为了进一步减少对历史通道的过度依赖，我们应用了上下文 dropout，随机以概率 $p = 0.5$ 屏蔽提供的文本历史（其余的训练流程保持不变）。这一正则化器迫使模型在历史缺失或不可靠时仍然依赖音频信息。在这一阶段，批量大小设为 6，学习率为 1e-4，预热步骤为 1000。在第三阶段，我们在一组具有挑战性的示例上优化模型，称为“硬负样本”，这些样本是通过使用第二阶段的最佳模型对训练集进行解码并选择字错误率（WER）超过 $20\%$ 的实例来识别的。在这一特定子集上，我们对两种优化策略进行了比较分析：一个额外的 SFT 过程和 DPO，其中后者利用偏好对显式减轻在嘈杂历史条件下的错误传播。为了实现这些策略，我们引入了一个专用的 LoRA 模块，与之前的 SFT 阶段不同，同时保持模型主干不变。在这一阶段，对于额外的 SFT，批量大小为 6；对于 DPO，批量大小为 2，梯度累积为 16。两种策略均使用 0 个预热步骤，学习率为 1e-5。

# 3.4 推理细节

在推理阶段，我们采用束搜索解码来平衡生成质量和多样性。具体而言，我们将束宽设置为4，最大生成长度设置为200个词元。为确保结果的确定性和可复现性，我们禁用采样策略（将do_sample设置为False，top_p设置为1.0，温度设置为1.0）。我们还保持中性的惩罚设置，重复惩罚和长度惩罚均设置为1.0。推理时LoRA强度的调整。我们引入一个显式的推理时强度因子$\gamma$来控制DPO LoRA适配器的贡献，同时在训练和推理过程中保持LoRA的秩和缩放超参数不变。LoRA的公式。设$W_{\mathrm{LLM}}$为冻结的基础LLM权重。我们附加两个LoRA适配器：一个SFT适配器和一个DPO适配器。前向传播中使用的有效权重矩阵为：

$$
W = W _ { \mathrm { L L M } } + { \frac { \alpha } { r } } \Delta W _ { \mathrm { S F T } } + \gamma { \frac { \alpha ^ { \prime } } { r ^ { \prime } } } \Delta W _ { \mathrm { D P O } }
$$

其中 $\Delta W _ { \mathrm { S F T } }$ 和 $\Delta W _ { \mathrm { D P O } }$ 是在 SFT 和 DPO 阶段学习的低秩更新（例如，$\Delta W = B A$），而 $\frac { \alpha } { r }$（分别为 $\frac { \alpha ^ { \prime } } { r ^ { \prime } }$）控制注入的 LoRA 特征的幅度（Hu et al. 2021）。在我们的实现中，我们对两个适配器使用相同的秩和缩放，即具体地，$r = 8$ 和 $\alpha = 32$，因此 DPO 适配器在推理时唯一的额外自由度是 $\gamma$。

$$
r = r ^ { \prime } , \qquad \alpha = \alpha ^ { \prime }
$$

训练与推理。在 DPO 训练过程中，我们设置 $\gamma = 1$，因此 DPO 适配器以其最大强度被应用：

$$
W _ { \mathrm { t r a i n } } = W _ { \mathrm { L L M } } + \frac { \alpha } { r } \Delta W _ { \mathrm { S F T } } + \frac { \alpha } { r } \Delta W _ { \mathrm { D P O } }
$$

在推理时，我们保持 $r$ 和 $\alpha$ 不变，而是调整 $\gamma < 1$ 以减弱 DPO 适配器的影响：

$$
W _ { \mathrm { i n f e r } } = W _ { \mathrm { L L M } } + \frac { \alpha } { r } \Delta W _ { \mathrm { S F T } } + \gamma \frac { \alpha } { r } \Delta W _ { \mathrm { D P O } }
$$

在我们的实验中，我们设置 $\gamma = 0.25$，这有效地将 DPO 适配器的强度与训练相比降低了 $4 \times$，同时保留了学习到的参数和所有其他 LoRA 超参数。为什么在推理时调整 $\gamma$？经验上，我们观察到 DPO 调优的适配器在解码过程中可能表现得过于激进，偶尔出现奖励过度优化（即模型在保持连贯性和忠实生成的代价下过度利用偏好信号）（Gao et al., 2023）。引入 $\gamma$ 提供了一种简单而稳定的方式来控制 DPO 在测试时的影响：降低 $\gamma$ 可以减轻过度优化，并有助于保持基础模型的泛化能力和流畅性，同时保留大部分来自偏好对齐的鲁棒性提升。

# 4 结果与分析

# 4.1 主要结果

我们在TED-LIUM 3数据集上验证了所提出框架的有效性。表1展示了基线方法与我们提出的方法之间的字符错误率（WER）比较。为了确保在实际部署条件下的公平比较，我们主要关注于$\mathrm { C o n } _ { \mathrm { i n f } }$为假设的设置，这意味着模型必须依赖于预测历史，而非oracle转录。

上下文和训练策略的影响：与没有上下文的基线 $( 7 . 8 9 \% )$ 进行比较，显示出引入上下文并不一定保证性能提升；这在很大程度上依赖于训练策略。如表1所示，配置为 $\mathrm { { C o n } _ { \mathrm { { t r a i n } } } }$ 的 Whisper large-v3 生成的上下文，在没有上下文丢弃的情况下 (0 Dropout) 达到 $8 . 1 5 \%$ 的字错误率 (WER) 在 $N = 2$ 时，表现不如无上下文基线。这一劣化表明，如果没有正则化，模型可能会过拟合到嘈杂的历史数据或过于依赖它。相比之下，以 0.5 丢弃率训练的模型有效利用了上下文。在相同的配置 $\mathrm { ~ N ~ } = \mathrm { ~ 2 ~ }$ 下，该模型（Whisper历史）取得了 $5 . 4 7 \%$ 的字错误率，显著超过无上下文基线。因此，我们将以 Whisper large-v3 生成的上下文和 0.5 丢弃率训练的模型识别为我们表现最佳的 SFT 模型，它为后续优化提供了稳健的基线。

Table 1: WER comparison on TED-LIUM 3 and out-of-domain Librispeech dataset across different context window sizes $( N )$ .The column $\mathbf { C o n } _ { \mathrm { i n f } } / \mathbf { C o n } _ { \mathrm { t r a i n } }$ specifies the source of history used during inference and training, respectively. hyp denotes using the model's own predictions as history during inference. Regarding training conguration, GT uses ground-truth history, while Whisper indicates the model was trained using context decoded by Whisper to simulate historical errors. $\mathbf { + D P O }$ and $+ \mathbf { S F T } 2$ are additional fine-tuning stages applied to the SFT model.   

<table><tr><td rowspan="2">N</td><td rowspan="2">Coninf/Contrain</td><td colspan="4">0 Dropout WER (%)↓</td><td colspan="4">0.5 Dropout WER (%)↓</td></tr><tr><td>TED</td><td>Test-clean</td><td>Test-other LS-Ave.</td><td></td><td>TED</td><td>Test-clean Test-other LS-Ave.</td><td></td><td></td></tr><tr><td rowspan="6">0</td><td>-1-</td><td>7.89</td><td>4.79</td><td>9.83</td><td>7.310</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GT / GT</td><td>5.6</td><td>4.49</td><td>10.36</td><td>7.425</td><td>7.89</td><td>4.31</td><td>9.68</td><td>6.995</td></tr><tr><td>hyp / GT</td><td>5.85</td><td>4.54</td><td>10.63</td><td>7.585</td><td>7.47</td><td>4.74</td><td>9.94</td><td>7.340</td></tr><tr><td>hyp / Whisper</td><td>5.62</td><td>4.67</td><td>9.46</td><td>7.065</td><td>7.21</td><td>5.37</td><td>9.96</td><td>7.665</td></tr><tr><td>+ DPO</td><td>5.69</td><td>4.71</td><td>9.57</td><td>7.140</td><td>5.32</td><td>4.56</td><td>9.38</td><td>6.970</td></tr><tr><td>+ SFT2</td><td>5.76</td><td>4.67</td><td>9.49</td><td>7.080</td><td>7.26</td><td>5.14</td><td>9.30</td><td>7.220</td></tr><tr><td rowspan="6">2</td><td>GT / GT</td><td>6.73</td><td>4.10</td><td>8.36</td><td>6.230</td><td>5.66</td><td>4.10</td><td>8.37</td><td>6.235</td></tr><tr><td>hyp / GT</td><td>6.89</td><td>4.85</td><td>9.88</td><td>7.365</td><td>5.59</td><td>5.15</td><td>9.10</td><td>7.130</td></tr><tr><td>hyp / Whisper</td><td>8.15</td><td>5.57</td><td>12.00</td><td>8.785</td><td>5.47</td><td>5.14</td><td>9.50</td><td>7.320</td></tr><tr><td>+ DPO</td><td>5.07</td><td>4.87</td><td>9.51</td><td>7.190</td><td>5.17</td><td>4.84</td><td>9.19</td><td>7.015</td></tr><tr><td>+ SFT2</td><td>6.90</td><td>4.55</td><td>11.17</td><td>7.860</td><td>6.10</td><td>5.43</td><td>9.66</td><td>7.545</td></tr><tr><td>GT / GT</td><td>7.35</td><td>4.24</td><td>8.29</td><td>6.265</td><td>10.42</td><td>4.89</td><td>10.36</td><td>7.625</td></tr><tr><td rowspan="4">3</td><td>hyp / GT</td><td>7.05</td><td>5.03</td><td>10.68</td><td>7.855</td><td>12.62</td><td>5.28</td><td>10.93</td><td>8.105</td></tr><tr><td>hyp / Whisper</td><td>10.06</td><td>5.36</td><td>10.69</td><td>8.025</td><td>7.87</td><td>5.93</td><td>10.39</td><td>8.160</td></tr><tr><td>+ DPO</td><td>5.98</td><td>5.60</td><td>9.96</td><td>7.780</td><td>5.18</td><td>4.73</td><td>9.36</td><td>7.045</td></tr><tr><td>+ SFT2</td><td>9.30</td><td>5.22</td><td>10.49</td><td>7.855</td><td>8.01</td><td>6.11</td><td>10.20</td><td>8.155</td></tr><tr><td rowspan="6">4</td><td>GT / GT</td><td>8.54</td><td>4.26</td><td>9.01</td><td>6.635</td><td>9.22</td><td>4.75</td><td>10.23</td><td>7.490</td></tr><tr><td>hyp / GT</td><td>7.74</td><td>4.87</td><td>11.07</td><td>7.970</td><td>10.87</td><td>4.75</td><td>10.23</td><td>7.490</td></tr><tr><td>hyp / Whisper</td><td>87.37</td><td>4.66</td><td>10.81</td><td>7.735</td><td>7.81</td><td>4.75</td><td>9.82</td><td>7.285</td></tr><tr><td>+ DPO</td><td>4.93</td><td>4.79</td><td>9.97</td><td>7.380</td><td>5.69</td><td>4.79</td><td>9.25</td><td>7.020</td></tr><tr><td>+ SFT2</td><td>113.95</td><td>4.90</td><td>11.34</td><td>8.120</td><td>9.16</td><td>4.75</td><td>9.83</td><td>7.290</td></tr><tr><td>GT / GT</td><td>8.72</td><td>5.46</td><td>9.49</td><td>7.475</td><td>9.57</td><td>4.90</td><td></td><td></td></tr><tr><td rowspan="6">5</td><td rowspan="3">hyp / GT hyp / Whisper</td><td></td><td></td><td></td><td></td><td></td><td></td><td>10.04</td><td>7.470</td></tr><tr><td>10.34</td><td>5.08</td><td>10.76</td><td>7.920</td><td>8.19</td><td>5.36</td><td>11.29</td><td>8.325</td></tr><tr><td>135.57</td><td>4.59</td><td>10.87</td><td>7.730</td><td>8.5</td><td>4.95</td><td>9.33</td><td>7.140</td></tr><tr><td>+ DPO</td><td>5.34</td><td>4.67</td><td>9.85</td><td>7.260</td><td>4.96</td><td>4.55</td><td>9.24</td><td>6.895</td></tr><tr><td>+ SFT2</td><td>72.55</td><td>4.90</td><td>10.57</td><td>7.735</td><td>8.51</td><td>5.23</td><td>9.33</td><td>7.280</td></tr></table>

Table 2: Impact of DPO LoRA scaling factor $( \gamma )$ during inference. TED-LIUM 3 Gap denotes the WER degradation cabyeevacntacksAttacks/" ee eleva tewhiAttacks/we irrelevent context randomly selected from the test set.   

<table><tr><td>γ</td><td colspan="2">TED-LIUM 3 (WER %)↓ Attacks/o</td><td colspan="3">LibriSpeech (WER %) ↓</td></tr><tr><td>0</td><td>5.47</td><td>Attacks/w 7.93</td><td>Gap 2.46</td><td>Test-clean 5.14</td><td>Test-other Ave. 9.50</td></tr><tr><td>0.0625</td><td>5.37</td><td>7.13 1.76</td><td>5.12</td><td>9.31</td><td>7.320 7.215</td></tr><tr><td>0.125 0.1875</td><td>5.11</td><td>5.76 0.65</td><td>5.02</td><td>9.53</td><td>7.275</td></tr><tr><td></td><td>5.06</td><td>5.69</td><td>0.63</td><td>4.70 9.08</td><td>6.890</td></tr><tr><td>0.25</td><td>5.17</td><td>5.63</td><td>0.46</td><td>4.84 9.19</td><td>7.015</td></tr><tr><td>0.375</td><td>5.55</td><td>5.73</td><td>0.18</td><td>4.85 9.63</td><td>7.240</td></tr><tr><td>0.5</td><td>8.39</td><td>8.67</td><td>0.28</td><td>6.44 12.14</td><td>9.290</td></tr><tr><td>0.625</td><td>53.26</td><td>57.15</td><td>3.89</td><td>27.11</td><td>28.96 28.035</td></tr></table>

对正则化和暴露偏差的敏感性：我们分析了不同训练数据来源如何影响模型对上下文丢弃的敏感性。如表1所示，使用真实标注上下文$\mathbf { \bar { C } o n } _ { \mathrm { t r a i n } } = \mathbf { G T }$训练的基线模型对于丢弃率表现出不一致的偏好，具体取决于上下文长度。特别地，当$N = 1 , 3 , 4$时，模型在不使用丢弃（0丢弃）的情况下表现更好，而在$N = 2 , 5$时，则需要0.5丢弃才能达到最佳效果。这种波动表明，使用$\mathrm { C o n } _ { \mathrm { t r a i n } } = \mathrm{ GT}$的模型在结构上不稳定，需进行特定上下文长度的超参数调优以避免性能下降（例如，在$N = 3$时使用错误的0.5丢弃会导致高错误率为$12.62\%$）。相比之下，我们的模型使用Whisper训练上下文$( \mathbf { C o n } _ { \mathrm { t r a i n } } = \mathbf { W } \mathbf { h i s p e }$r)在顺序任务上表现出清晰且一致的模式。在上下文$N = 1$时偏爱0丢弃，但对于所有多轮场景$( N \ge 2 )$，0.5丢弃始终提供更优性能，并对稳定性至关重要。比较最佳配置，我们的方法$\mathbf { C o n } _ { \mathrm { t r a i n } } = \mathrm{ Whisper}$与0.5丢弃在$N = 2$时实现最佳整体性能，$5.47\%$对比GT最佳的$5.59\%$。此外，与在需要和拒绝正则化之间波动的真实标注基线不同，我们的方法提供了一个可靠的策略${ p = 0.5 }$，确保在不同上下文长度下的鲁棒性，避免了无正则化Whisper模型中出现的灾难性失败。

DPO的改进：为了进一步抑制识别错误，我们使用相同策略挖掘的“硬负样本”精炼每个SFT检查点，将额外的SFT（SFT2）与DPO进行比较。表1显示SFT2并不是一种可靠的精炼策略：它通常提供微乎其微的收益，并且经常导致性能下降（例如，在$N = 2$且0.5的dropout时，TED WER从$5.47\%$增加到$6.10\%$）。相比之下，DPO在几乎所有配置中（10个中的9个）都能稳定提高WER，尤其是在较长上下文窗口下，错误积累更为严重时，收益尤为显著（例如，在$N = 3$且0.5的dropout时，$7.87\%$与$5.18\%$；在$N = 5$且0.5的dropout时，$8.50\%$与$4.96\%$）。我们将这一现象归因于SFT仅仅迫使模型模仿真实标注数据，而DPO则明确优化模型，使其更倾向于真实标注数据而非其错误假设，从而实现更稳定的改进。

# 4.2 跨领域泛化

为了评估域外泛化，我们在LibriSpeech（test-clean/test-other）数据集上，以零-shot的设置评估所有模型，使用话语级解码，其中推理历史由模型自身的假设形成。如表1所示，使用噪声历史的SFT并未可靠地改善跨域迁移：在${ \bf N } = 2$和0.5 dropout下，hyp/Whisper SFT模型的LS-Ave.达到$7 . 3 2 \%$，这与无上下文基线$( 7 . 3 1 \% )$基本持平，而增加上下文窗口甚至可能导致性能下降（例如，在${ \mathrm { N } } { = } 3$时，0.5 dropout的表现为$8 . 1 6 \%$）。

相比之下，DPO在多个上下文窗口尺寸上始终提高了鲁棒性。例如，在${ \bf N } = 2$（0.5丢弃率）时，DPO将LS-Ave.从$7.32\%$降至$7.015\%$，而在${ \mathrm { N } } { = } 3$（0.5丢弃率）时，从$8.16\%$降至$7.045\%$。在${ \mathrm { N } } { = } 5$和0.5丢弃率下，获得的最佳域外结果为$6.895\%$ LS-Ave.，优于无上下文基线和相应的SFT基线。这一发现表明，DPO有效地减轻了领域特定的过拟合。通过惩罚历史错误和幻觉的重复，模型通过偏好学习学会更好地平衡上下文线索与声学证据，从而在未见的域外数据上实现更优秀的泛化能力。

# 4.3 DPO 推理扩展的影响

为了确定 DPO 模块的最佳推理策略，我们分析了 LoRA 缩放因子 $\gamma$ 的敏感性。如表 2 所示，DPO 信号强度衍生出一个关键的权衡。尽管 $\gamma = 0.1875$ 可实现最佳的干净准确率，但将 $\gamma$ 增加到 0.25 显著增强了模型的鲁棒性，使攻击下的性能下降（TED Gap）降至 $0.46\%$。这证实了更强的 DPO 权重有助于模型拒绝误导性的上下文。然而，过度的缩放（$ \gamma > 0.375 $）会导致奖励过度优化。因此，我们选择 $\gamma = 0.25$ 以优先考虑稳定性和错误抑制。

<table><tr><td rowspan="2">N</td><td rowspan="2">Coninf / Contrain</td><td colspan="3">0 Dropout WER (%)↓</td><td colspan="3">0.5 Dropout WER (%)↓</td></tr><tr><td>Attacks/o</td><td>Attacks/w</td><td>Gap</td><td>Attacks/o</td><td>Attacks/w</td><td>Gap </td></tr><tr><td rowspan="3">1</td><td>hyp / GT</td><td>5.85</td><td>8.32</td><td>2.47</td><td>7.47</td><td>8.82</td><td>1.35</td></tr><tr><td>hyp / Whisper</td><td>5.62</td><td>6.27</td><td>0.65</td><td>7.21</td><td>7.09</td><td>-0.12</td></tr><tr><td>+ DPO</td><td>5.69</td><td>5.5</td><td>-0.19</td><td>5.32</td><td>5.23</td><td>-0.09</td></tr><tr><td rowspan="3">2</td><td>hyp / GT</td><td>6.89</td><td>8.43</td><td>1.54</td><td>5.59</td><td>9.23</td><td>3.46</td></tr><tr><td>hyp / Whisper</td><td>8.15</td><td>10.37</td><td>2.22</td><td>5.47</td><td>7.93</td><td>2.46</td></tr><tr><td>+ DPO</td><td>5.07</td><td>6.59</td><td>1.52</td><td>5.17</td><td>5.63</td><td>0.46</td></tr><tr><td rowspan="3">3</td><td>hyp / GT</td><td>7.05</td><td>7.64</td><td>0.59</td><td>12.62</td><td>10.19</td><td>-2.43</td></tr><tr><td>hyp / Whisper</td><td>10.06</td><td>11.18</td><td>1.12</td><td>7.87</td><td>9.53</td><td>1.66</td></tr><tr><td>+ DPO</td><td>5.98</td><td>6.24</td><td>0.26</td><td>5.18</td><td>5.31</td><td>0.13</td></tr><tr><td rowspan="3">4</td><td>hyp / GT</td><td>7.74</td><td>9.75</td><td>2.01</td><td>10.87</td><td>13.15</td><td>2.28</td></tr><tr><td>hyp / Whisper</td><td>87.37</td><td>8.8</td><td>-78.57</td><td>7.81</td><td>8.82</td><td>1.01</td></tr><tr><td>+ DPO</td><td>4.93</td><td>5.44</td><td>0.51</td><td>5.69</td><td>7.58</td><td>1.89</td></tr><tr><td rowspan="3">5</td><td>hyp / GT</td><td>10.34</td><td>11.83</td><td>1.49</td><td>8.19</td><td>10.41</td><td>2.22</td></tr><tr><td>hyp / Whisper</td><td>135.57</td><td>10.74</td><td>-124.83</td><td>8.5</td><td>11.34</td><td>2.84</td></tr><tr><td>+ DPO</td><td>5.34</td><td>6.20</td><td>0.86</td><td>4.96</td><td>5.51</td><td>0.55</td></tr></table>

在TED-LIUM上进行与无关上下文攻击的故事稳健性分析。我们通过用随机抽样的无关上下文替换历史上下文来评估模型的稳健性。

# 4.4 对无关上下文的鲁棒性

为了验证模型是否真正理解上下文信息，我们进行了稳健性攻击测试。在该实验中，我们在推理过程中用从TED数据集中随机采样的语义无关上下文替换真实历史。一个稳健的上下文自动语音识别模型应能够识别上下文的不相关性，并回退到声学信号，从而最小化性能下降。我们将所提出方法的性能下降与标准的oracle训练基线进行比较。如表3所示，DPO优化模型显示出显著的鲁棒性。在所有上下文窗口大小 $N = [1, 5]$ 和两个dropout设置（0和0.5）中，尽管DPO的下降差距（Attacks/w - Attacks/o）并不总是最小，DPO优化模型始终实现了最低的攻击字错误率（WER）。总体而言，这些结果表明，结合噪声教师历史、上下文dropout以及DPO优化有助于在提供的历史不可靠时抑制性能下降。

# 4.5 DPO的数据选择与推断扩展

我们对WER选择阈值进行了消融研究（基于$N = 2$，0.5弃权SFT模型）。如附录A中的表4所示，我们的方法对数据严苛性表现出显著的鲁棒性，在所有阈值下都能获得一致的增益，而无需精确调优。此外，所有这些配置中的最优推理缩放$\gamma$保持稳定，这与第4.3节一致。这表明，最优推理策略与数据策划过程有效解耦，显著简化了部署。

# 5 结论

在本研究中，我们通过提出一个统一的训练框架，解决了语音大语言模型中的上下文暴露偏差，该框架整合了教师错误知识、上下文随机失活和DPO。我们证明了使用教师错误知识进行训练有效地缩小了训练-测试差距，显著优于基于oracle的基线。重要的是，上下文随机失活被证明是稳定性的决定性因素；它防止模型过度依赖文本历史。此外，DPO明确抑制错误传播，产生了优越的领域内性能和稳健的跨领域迁移能力。总体而言，我们的方法为在现实的、不完美条件下的长篇语音识别建立了一个可靠的范式。

# 局限性

尽管我们提出的框架有效缓解了上下文暴露偏差并增强了语音大语言模型的鲁棒性，仍然存在多个限制需要在未来的工作中解决。首先，多说话者重叠。我们当前的实验设置假设为顺序的轮流讲话（如在TED-LIUM 3和LibriSpeech中所见）。我们尚未在涉及重叠语音或“鸡尾酒会”环境的场景中评估模型的性能。由于语音大语言模型将音频处理为一个单一序列，处理重叠严重的同时发言者可能需要专门的架构修改或数据策划策略，这超出了本研究的范围。其次，教师错误源的多样性有限。虽然教师错误知识是一个模型无关的概念，原则上可以通过任意自动语音识别系统的假设进行实例化，但我们当前的实现依赖于单一的教师模型（Whisper large-v3）来生成错误的上下文知识。因此，在训练中暴露的模拟错误偏向于该教师的错误特征，可能无法全面代表语音大语言模型的多样化失效模式或在不受限制环境中遇到的广泛声学失真。

# References

Petar Aleksic, Mohammadreza Ghodsi, Assaf Michaely, Cyril Allauzen, Keith Hall, Brian Roark, David Rybach, and Pedro Moreno. 2015. Bringing contextual information to google speech recognition. In Proc. Interspeech 2015, pages 468472.

William Chan, Navdeep Jaitly, Quoc Le, and Oriol Vinyals. 2016. Listen, attend and spell: A neural network for large vocabulary conversational speech recognition. In 2016 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 49604964. IEEE.

Zhehuai Chen, He Huang, Andrei Andrusenko, Oleksii Hrinchuk, Krishna C Puvvada, Jason Li, Subhankar Ghosh, Jagadeesh Balam, and Boris Ginsburg. 2024. Salm: Speech-augmented language model with incontext learning for speech recognition and translation. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1352113525. IEEE.

Jian Cheng. 2024. Context-aware speech recognition using prompts for language learners. Interspeech 2024, pages 40094013.

Wenqian Cui, Dianzhi Yu, Xiaoqi Jiao, Ziqiao Meng, Guangyan Zhang, Qichao Wang, Yiwen Guo, and Irwin King. 2024. Recent advances in speech language models: A survey. arXiv preprint arXiv:2410.03751.

Yangui Fang, Jing Peng, Yu Xi, Xu Li, Haoyu Li, Chengwei Zhang, Guohui Zhong, and Kai Yu. 2025. Joint decoding method for controllable contextual speech recognition based on speech llm. arXiv preprint arXiv:2508.08585.

Leo Gao, John Schulman, and Jacob Hilton. 2023. Scaling laws for reward model overoptimization. In International Conference on Machine Learning, pages 1083510866. PMLR.

Xun Gong, Anqi Lv, Zhiming Wang, and Yanmin Qian. 2024. Contextual biasing speech recognition in speech-enhanced large language model. Proc. Interspeech. ISCA, pages 257261.

Aditya Gourav, Jari Kolehmainen, Prashanth Shivakumar, Yile Gu, Grant Strimel, Ankur Gandhe, Ariya Rastrow, and Ivan Bulyko. 2024. Multi-modal retrieval for large language model based speech recognition. In Findings of the Association for Computational Linguistics ACL 2024, pages 44354446.

Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. 2006. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In Proceedings of the 23rd international conference on Machine learning, pages 369376.

Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. 2013. Speech recognition with deep recurrent neural networks. In 2013 IEEE international conference on acoustics, speech and signal processing, pages 66456649. Ieee.

Yachao Guo, Zhibin Qiu, Hao Huang, and Chng Eng Siong. 2023. Improved keyword recognition based on aho-corasick automaton. In 2023 International Joint Conference on Neural Networks (IJCNN), pages 1-7.IEEE.

Keith Hall, Eunjoon Cho, Cyril Allauzen, Françoise Beaufays, Noah Coccaro, Kaisuke Nakajima, Michael Riley, Brian Roark, David Rybach, and Linda Zhang. 2015. Composition-based on-the-fly rescoring for salient n-gram biasing. In Proc. Interspeech 2015, pages 14181422.

François Hernandez, Vincent Nguyen, Sahar Ghannay, Natalia Tomashenko, and Yannick Estève. 2018. Tedlium 3: Twice as much data and corpus repartition for experiments on speaker adaptation. In Speech and Computer, pages 198208. Springer International Publishing.

Takaaki Hori, Niko Moritz, Chiori Hori, and Jonathan Le Roux. 2020. Transformer-based long-context endto-end speech recognition. In Interspeech, pages 50115015.

Junfeng Hou, Jinkun Chen, Wanyu Li, Yufeng Tang, Jun Zhang, and Zejun Ma. 2022. Bring dialogue-context into rnn-t for streaming asr. In INTERSPEECH, pages 20482052.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing, 29:34513460.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2021. Lora: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, and 1 others. 2025. Step-audio: Unified understanding and generation in intelligent speech interaction. CoRR.

Ruizhe Huang, Mahsa Yarmohammadi, Sanjeev Khudanpur, and Daniel Povey. 2024a. Improving neural biasing for contextual speech recognition by early context injection and text perturbation. In Proc. Interspeech 2024, pages 752756.

Zhiqi Huang, Diamantino Caseiro, Kandarp Joshi, Christopher Li, Pat Rondon, Zelin Wu, Petr Zadrazil, and Lillian Zhou. 2024b. Optimizing large-scale context retrieval for end-to-end asr. Proc. Interspeech. ISCA, pages 45734577.

Amir Hussein, Brian Yan, Antonios Anastasopoulos, Shinji Watanabe, and Sanjeev Khudanpur. 2024. Enhancing end-to-end conversational speech translation through target language context utilization. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1197111975. IEEE.

Mahaveer Jain, Gil Keren, Jay Mahadeokar, Geoffrey Zweig, Florian Metze, and Yatharth Saraf. 2020. Contextual rnn-t for open domain asr. In Proc. Interspeech 2020, pages 1115.

Lifan Jiang, Boxi Wu, Jiahui Zhang, Xiaotong Guan, and Shuang Chen. 2025. Huvidpo: Enhancing video generation through direct preference optimization for human-centric alignment. arXiv preprint arXiv:2502.01690.

Konstantin Kolokolov, Pavel Pekichev, and Karthik Raghunathan. 2024. Self-consistent context aware conformer transducer for speech recognition. arXiv preprint arXiv:2402.06592.

Roman Koshkin, Katsuhito Sudoh, and Satoshi Nakamura. 2024. Llms are zero-shot context-aware simultaneous translators. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 11921207.

Egor Lakomkin, Chunyang Wu, Yassir Fathullah, Ozlem Kalinli, Michael L Seltzer, and Christian Fuegen. 2024. End-to-end speech recognition contextualization with large language models. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1240612410. IEEE.

Wonjun Lee, San Kim, and Gary Geunbae Lee. 2024. Enhancing dialogue speech recognition with robust contextual awareness via noise representation learning. In Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 333343.

Zhihong Lei, Xingyu Na, Mingbin Xu, Ernest Pusateri, Christophe Van Gysel, Yuanyuan Zhang, Shiyi Han, and Zhen Huang. 2025. Contextualization of asr with llm using phonetic retrieval-based augmentation. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 15. IEEE.

Jinyu Li and 1 others. 2022. Recent advances in end-toend automatic speech recognition. APSIPA Transactions on Signal and Information Processing, 11.

Shaojun Li, Hengchao Shang, Daimeng Wei, Jiaxin Guo, Zongyao Li, Xianghui He, Min Zhang, and Hao Yang. 2024. La-rag: Enhancing llm-based asr accuracy with retrieval-augmented generation. CoRR.

Ziyang Ma, Guanrou Yang, Yifan Yang, Zhifu Gao, Jiaming Wang, Zhihao Du, Fan Yu, Qian Chen, Siqi Zheng, Shiliang Zhang, and 1 others. 2025. Speech recognition meets large language model: Benchmarking, models, and exploration. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 2484024848.

Erik McDermott, Hasim Sak, and Ehsan Variani. 2019. A density ratio approach to language model fusion in end-to-end automatic speech recognition. In 2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 434441. IEEE.

Bingshen Mu, Hexin Liu, Hongfei Xue, Kun Wei, and Lei Xie. 2025. Hearing more with less: Multi-modal retrieval-and-selection augmented conversational llmbased asr. arXiv preprint arXiv:2508.01166.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroli Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:2773027744.

Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. 2015. Librispeech: an asr corpus based on public domain audio books. In 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 52065210. IEEE.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 2849228518. PMLR.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:5372853741.

Vijay Ravi, Yile Gu, Ankur Gandhe, Ariya Rastrow, Linda Liu, Denis Filimonov, Scott Novotney, and Ivan Bulyko. 2020. Improving accuracy of rare words for rnn-transducer through unigram shallow fusion. arXiv preprint arXiv:2012.00133.

Peng Shen, Xugang Lu, and Hisashi Kawai. 2025. Retrieval-augmented speech recognition approach for domain challenges. arXiv preprint arXiv:2502.15264.

Xian Shi, Yexin Yang, Zerui Li, Yanni Chen, Zhifu Gao, and Shiliang Zhang. 2024. Seaco-paraformer: A non-autoregressive asr system with flexible and effective hotword customization ability. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1034610350. IEEE.

Yui Sudo, Yosuke Fukumoto, Muhammad Shakeel, Yifan Peng, and Shinji Watanabe. 2024. Contextualized automatic speech recognition with dynamic vocabulary. In 2024 IEEE Spoken Language Technology Workshop (SLT), pages 7885. IEEE.

Jiyang Tang, Kwangyoun Kim, Suwon Shon, Felix Wu, and Prashant Sridhar. 2024. Improving asr contextual biasing with guided attention. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1209612100. IEEE.

Shubham Toshniwal, Anjuli Kannan, Chung-Cheng Chiu, Yonghui Wu, Tara N Sainath, and Karen Livescu. 2018. A comparison of techniques for language model integration in encoder-decoder speech recognition. In 2018 IEEE spoken language technology workshop (SLT), pages 369375. IEEE.

Mengzhi Wang, Shifu Xiong, Genshun Wan, Hang Chen, Jianqing Gao, and Lirong Dai. 2024. Deep clas: Deep contextual listen, attend and spell. arXiv preprint arXiv:2409.17603.

Guanrou Yang, Ziyang Ma, Fan Yu, Zhifu Gao, Shiliang Zhang, and Xie Chen. 2024. Mala-asr: Multimediaassisted llm-based asr. CoRR.

Saierdaer Yusuyin, Te Ma, Hao Huang, Wenbo Zhao, and Zhijian Ou. 2025. Whistle: Data-efficient multilingual and crosslingual speech recognition via weakly phonetic supervision. IEEE Transactions on Audio, Speech and Language Processing.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, and 1 others. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:4659546623.

Yue Zhou, Yuxuan Yuan, Chengwei Zhang, and Xiaodong Shi. 2025. Boosting context-aware speech translation with large language models. IEEE Signal Processing Letters.

# A Data Selection and Inference Scaling for DPO

<table><tr><td rowspan="2">WER (%) Threshold</td><td rowspan="2">γ</td><td colspan="3">TED-LIUM 3 (WER %) ↓</td><td colspan="3">LibriSpeech (WER %) ↓</td></tr><tr><td></td><td>Attacks/o Attacks/w</td><td>Gap↓</td><td>Test-clean Test-other</td><td></td><td>Ave.</td></tr><tr><td rowspan="9">5</td><td>0</td><td>5.47</td><td>7.93</td><td>2.46</td><td>5.14</td><td>9.50</td><td>7.320</td></tr><tr><td>0.0625</td><td>5.40</td><td>7.02</td><td>1.62</td><td>5.11</td><td>9.32</td><td>7.215</td></tr><tr><td>0.125</td><td>5.43</td><td>7.34</td><td>1.91</td><td>5.18</td><td>9.41</td><td>7.295</td></tr><tr><td>0.1875</td><td>5.04</td><td>6.76</td><td>1.72</td><td>5.09</td><td>9.70</td><td>7.395</td></tr><tr><td>0.25</td><td>5.23</td><td>5.99</td><td>0.76</td><td>5.18</td><td>9.51</td><td>7.345</td></tr><tr><td>0.375</td><td>6.21</td><td>6.42</td><td>0.21</td><td>5.49</td><td>10.19</td><td>7.840</td></tr><tr><td>0.5</td><td>9.18</td><td>9.63</td><td>0.45</td><td>8.58</td><td>14.65</td><td>11.615</td></tr><tr><td>0.625</td><td>50.40</td><td>50.97</td><td>0.57</td><td>69.98</td><td>70.66</td><td>70.320</td></tr><tr><td>0</td><td>5.47</td><td>7.93</td><td>2.46</td><td>5.14</td><td>9.50</td><td>7.320</td></tr><tr><td rowspan="8">10</td><td>0.0625</td><td>5.40</td><td>7.10</td><td>1.70</td><td>5.11</td><td>9.30</td><td>7.205</td></tr><tr><td>0.125</td><td>5.32</td><td>6.91</td><td>1.59</td><td>5.19</td><td>9.61</td><td>7.400</td></tr><tr><td>0.1875</td><td>5.04</td><td>6.63</td><td>1.59</td><td>5.04</td><td>9.67</td><td>7.355</td></tr><tr><td>0.25</td><td>5.27</td><td>6.00</td><td>0.73</td><td>4.62</td><td>9.50</td><td>7.060</td></tr><tr><td>0.375</td><td>6.29</td><td>6.38</td><td>0.09</td><td>5.06</td><td>10.22</td><td>7.640</td></tr><tr><td>0.5</td><td>10.22</td><td>10.87</td><td>0.65</td><td>8.65</td><td>14.96</td><td>11.805</td></tr><tr><td>0.625</td><td>85.55</td><td>89.97</td><td>4.42</td><td>88.93</td><td>85.49</td><td>87.210</td></tr><tr><td>15</td><td>5.47</td><td>7.93</td><td>2.46</td><td></td><td>5.14</td><td>9.50</td><td>7.320</td></tr><tr><td rowspan="8"></td><td>0 0.0625</td><td>5.40</td><td>7.43</td><td>2.03</td><td>5.12</td><td>9.30</td><td>7.210</td></tr><tr><td>0.125</td><td>5.15</td><td>6.63</td><td>1.48</td><td>5.21</td><td>9.58</td><td>7.395</td></tr><tr><td>0.1875</td><td>5.12</td><td>5.91</td><td>0.79</td><td>4.98</td><td>9.46</td><td>7.220</td></tr><tr><td></td><td>5.07</td><td>5.67</td><td>0.60</td><td>4.77</td><td>9.27</td><td>7.020</td></tr><tr><td>0.25</td><td>5.79</td><td>6.14</td><td>0.35</td><td>5.04</td><td>9.84</td><td></td></tr><tr><td>0.375</td><td>8.27</td><td>8.84</td><td>0.57</td><td>6.99</td><td>12.89</td><td>7.440</td></tr><tr><td>0.5 0.625</td><td>33.58</td><td>35.59</td><td>2.01</td><td>22.72</td><td>27.96</td><td>9.940 25.340</td></tr><tr><td></td><td>5.47</td><td></td><td></td><td>5.14</td><td></td><td></td></tr><tr><td rowspan="11">20</td><td>0</td><td></td><td>7.93</td><td>2.46</td><td></td><td>9.50</td><td>7.320</td></tr><tr><td>0.0625</td><td>5.37</td><td>7.13</td><td>1.76</td><td>5.12</td><td>9.31</td><td>7.215</td></tr><tr><td>0.125</td><td>5.11</td><td>5.76</td><td>0.65</td><td>5.02</td><td>9.53</td><td>7.275</td></tr><tr><td>0.1875</td><td>5.06</td><td>5.69</td><td>0.63</td><td>4.70</td><td>9.08</td><td>6.890</td></tr><tr><td>0.25</td><td>5.17</td><td>5.63</td><td>0.46</td><td>4.84</td><td>9.19</td><td>7.015</td></tr><tr><td>0.375</td><td>5.55</td><td>5.73</td><td>0.18</td><td>4.85</td><td>9.63</td><td>7.240</td></tr><tr><td>0.5 0.625</td><td>8.39</td><td>8.67</td><td>0.28</td><td>6.44</td><td>12.14</td><td>9.290</td></tr><tr><td></td><td>53.26 5.47</td><td>57.15</td><td>3.89</td><td>27.11</td><td>28.96</td><td>28.035</td></tr><tr><td rowspan="7">25</td><td>0</td><td></td><td>7.93</td><td>2.46</td><td>5.14</td><td>9.50</td><td>7.320</td></tr><tr><td>0.0625</td><td>5.39</td><td>7.40</td><td>2.01</td><td>5.13</td><td>9.33</td><td>7.230</td></tr><tr><td>0.125</td><td>5.28</td><td>6.00</td><td>0.72</td><td>5.21</td><td>9.57</td><td>7.390</td></tr><tr><td>0.1875</td><td>5.08</td><td>5.68</td><td>0.60</td><td>4.70</td><td>9.18</td><td>6.940</td></tr><tr><td>0.25</td><td>5.28</td><td>5.34</td><td>0.06</td><td>4.84</td><td>9.33</td><td>7.085</td></tr><tr><td>0.375</td><td>5.74</td><td>6.04</td><td>0.30</td><td>5.12</td><td>10.01</td><td>7.565</td></tr><tr><td>0.5</td><td>10.26</td><td>10.50</td><td>0.24</td><td>8.25</td><td>14.64</td><td>11.445</td></tr><tr><td>0.625</td><td>31.01</td><td>30.48</td><td>-0.53</td><td>28.03</td><td>35.84</td><td>31.935</td></tr><tr><td rowspan="8">30</td><td>0</td><td>5.47</td><td>7.93</td><td>2.46</td><td>5.14</td><td>9.50</td><td>7.320</td></tr><tr><td>0.0625</td><td>5.32</td><td>7.38</td><td>2.06</td><td>5.14</td><td>9.28</td><td>7.210</td></tr><tr><td>0.125</td><td>5.12</td><td>6.06</td><td>0.94</td><td>5.14</td><td>9.48</td><td>7.310</td></tr><tr><td>0.1875</td><td>5.00</td><td>5.51</td><td>0.51</td><td>4.70</td><td>9.54</td><td>7.120</td></tr><tr><td>0.25</td><td>4.97</td><td>5.39</td><td>0.42</td><td>4.74</td><td>9.28</td><td>7.010</td></tr><tr><td>0.375</td><td>4.97</td><td>5.20</td><td>0.23</td><td>4.83</td><td>9.33</td><td>7.080</td></tr><tr><td>0.5</td><td>5.49</td><td>5.78</td><td>0.29</td><td>5.11</td><td>10.00</td><td>7.555</td></tr><tr><td>0.625</td><td>8.77</td><td>9.11</td><td>0.34</td><td>8.16</td><td>14.73</td><td>11.445</td></tr></table>

Tabl : Impact of Hard Negatives threshold and DPO LoRA scaling fctor $( \gamma )$ during inference. TED-LIUM 3 Gap denotes the WER degradation caused by irrelevant context attacks.