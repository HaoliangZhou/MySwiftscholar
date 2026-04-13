# FiRE : 通过细粒度上下文学习增强多模态大语言模型以进行复杂图像检索

博翰·侯 山东大学 中国山东青岛 bohanhou@foxmail.com 浩强·林 山东大学 中国山东青岛 zichaohq@gmail.com 雪梦·宋* 香港城市大学 中国香港 sxmustc@gmail.com 浩琨·温 哈尔滨工业大学（深圳） 中国广东深圳 whenhaokun@gmail.com 孟·刘 山东建筑大学 中国山东济南 mengliu.sdu@gmail.com 宇鹏·胡 山东大学 中国山东济南 huyupeng@sdu.edu.cn 翔豫·赵 香港城市大学 中国香港 xy.zhao@cityu.edu.hk

# 摘要

由于其强大的可泛化多模态处理和推理能力，多模态大语言模型（MLLMs）在作为通用图像检索器方面展现了显著的潜力，有效应对各种现实世界的图像检索任务。然而，首批研究虽然前景可期，却忽视了细粒度上下文建模和解耦微调目标在提升MLLM检索性能中的潜力，特别是在诸如长文本到图像检索、视觉对话检索和复合图像检索（CIR）等复杂任务中。因此，在本研究中，我们提出了一种自动化的细粒度多模态五元组数据集构建流程以及一种新颖的两阶段细粒度多模态微调策略。数据集生成流程产生了一个全面的CIR数据集，包含细粒度的图像描述和修改文本，促进了细粒度上下文建模。超越之前的纠缠微调范式，我们的方法将微调过程分为两个不同的阶段：（1）细粒度上下文推理导向的微调和（2）细粒度检索导向的微调。这些阶段旨在顺序提升模型的上下文理解和查询-目标对齐能力，从而提高检索性能。针对五个包含多样复杂图像检索任务的数据集进行的大量实验表明，我们的方法在零样本检索设置中，与现有方法相比展现了显著的优越性，即使与那些方法相比，采用了更轻量的MLLM主干网络。

# CCS 概念

信息系统 图片搜索。

# 关键词

多模态大语言模型；图像检索；复杂图像检索；细粒度上下文建模；

# ACM 参考格式：

Bohan Hou, Haoqiang Lin, Xuemeng Song\*, Haokun Wen, Meng Liu, Yupeng Hu, 和 Xiangyu Zhao. 2025. FiRE : 通过细粒度上下文学习增强大规模语言模型以进行复杂图像检索. 载于第48届国际ACM SIGIR信息检索研究与开发大会论文集 (SIGIR '25), 2025年7月13-18日, 意大利帕多瓦. ACM, 纽约, NY, 美国, 10页. https://doi.org/10.1145/3726302.3729979

# 1 引言

为了满足用户在实际应用中的多样化需求[30, 33, 35, 36]，提出了多种图像检索范式。这些范式包括标准的短文本到图像检索[18, 26, 41]，其中简短的标题足以表达用户的搜索意图，以及更复杂的范式，如长文本到图像检索[39]、基于对话的图像检索[5]和组合图像检索（CIR）[10, 32]。在这些更加具有挑战性的场景中，用户复杂的搜索意图往往需要通过长而详细的查询或组合的多模态输入（即带有修改文本的参考图像）来传达，这显著增加了检索的复杂性。现有的方法通常独立处理这些任务，导致高昂的训练成本和碎片化的解决方案。为了提高效率和可扩展性，近期研究已转向开发能够在单一框架内处理多样化任务和查询类型的统一检索模型。

![](images/1.jpg)  

Figure 1: Illustration of complex image retrieval tasks: (a) Composed Image Retrieval, (b) Long-Text-to-Image Retrieval, and (c) Dialog-based Image Retrieval.

先锋工作 [2, 40] 通常依赖于视觉-语言预训练模型 (VLPs) [15, 27]，利用其强大的多模态嵌入能力来解决各种检索任务。然而，由于模型规模相对较小，VLPs 在理解复杂查询方面存在困难 [8]，尤其是涉及推理的查询。最近的研究转向大型语言模型 (LLMs)，这些模型在语言理解和推理能力上表现更佳 [11]，以应对多样的图像检索任务。由于 LLMs 作为生成模型，本质上缺乏进行检索任务所需的判别查询-目标对齐能力 [13, 22, 23]，这些研究专注于设计有效的微调策略来弥补这一差距。例如，MCL [16] 在两项任务——多模态上下文标注和多模态上下文检索上微调一个与 CLIP 视觉编码器和适配器集成的 LLM，使用了一个定制的大规模多模态组合数据集。相对而言，E5-V [11] 在纯句子对上微调一个预训练 MLLM 的核心 LLM 组件，旨在增强其查询-目标对齐能力。尽管这些方法取得了可喜的结果，但现有方法面临两个主要限制，妨碍其在复杂图像检索任务中的表现。1) 缺乏细粒度上下文建模。现有方法依赖句子对或 $<$ 参考图像、短修改文本、短目标标题 $>$ 三元组来微调 MLLMs。然而，这些数据类型仅提供粗粒度信息，缺乏发展模型细粒度上下文理解能力所需的详细描述。事实上，许多现实世界检索任务中的查询往往涉及长或复合上下文（见图 1），这本质上需要细粒度理解以便进行准确解读。此外，现代 MLLMs 通常将每个图像表示为大量词元嵌入，与处理长上下文相似，进一步强调了细粒度上下文建模的重要性。2) 次优的微调目标。E5-V 的微调策略仅专注于提高 MLLM 的同质查询-目标对齐能力，而忽视了其多模态上下文理解能力的优化。尽管 MCL 融入了多模态上下文学习，但它在多模态检索和生成任务上同时微调 MLLMs。这种纠缠的微调目标可能会妨碍增强上下文理解和提高查询-目标对齐之间的平衡，最终导致次优的检索性能。

因此，我们的目标是通过增强 MLLM 的细粒度上下文学习能力来解决这些限制，将其转变为一个强大的通用图像检索器，能够处理各种图像检索任务，特别是复杂任务。具体而言，针对第一个限制，我们提出了一个完全自动化的流程，用于生成大规模细粒度多模态组合数据集。该流程分为三个阶段：（1）基于 CoT 的细粒度标题生成，其中 LLM 在细粒度推理步骤的指导下生成详细标题，包括以主题为导向、以属性为导向和以上下文为导向的推理；（2）基于 MLLM 的图像对识别，利用细粒度语义相似性（而非传统的视觉或粗粒度相似性）来识别潜在的参考-目标图像对。此阶段通过与 $< i m \cdot$ age, 细粒度标题> 对进行微调，增强其长文本编码能力；（3）类人细粒度修改生成，其中设计了一个模糊性引导的指令，以弥合 LLM 输出与人类标注之间的差异。通过这个流程，我们创建了一个名为 FiGMaQ 的大规模细粒度多模态五元组数据集，包含 87K 个样本，每个五元组样本遵循格式 $<$ 参考图像, 参考标题, 修改文本, 目标图像, 目标标题 >。与现有的 LLM 生成的多模态三元组数据集相比，我们的数据集提供了三个显著特征。1）图像标题是细粒度的，包含详细信息，平均超过 100 个词元。2）修改文本捕获了更多的具体细节，且更具类人特征，能够容纳模糊术语。3）每个样本包括五个组成部分，便于多种微调任务，如多模态上下文标题生成和检索。

为了应对第二个限制，我们提出了一种两阶段的精细化多模态微调策略，称为FiRE，旨在通过解耦的微调目标，依次增强大规模语言模型（MLLMs）的上下文理解和查询-目标对齐能力。具体而言，受到基于CIR的微调在提升MLLM在各种零样本检索任务上表现成功的启发，我们采用CIR——这一需要复杂上下文理解和细粒度推理的任务，作为MLLM微调的代表任务。在第一阶段，我们通过指示MLLM基于参考图像和修改文本生成目标图像的细粒度标题，进行针对细粒度上下文推理的微调。在第二阶段，基于第一阶段所发展起来的提升的多模态推理能力，我们进行针对细粒度检索的微调，其中同时使用InfoNCE损失和Recall@k代理损失来提升MLLM的查询-目标对齐能力。总结而言，我们的贡献可以概括如下：•我们提出了一种自动化流程，用于生成细粒度多模态组合数据集，并贡献了一个大规模数据集FiGMaQ，以支持未来在MLLM上的细粒度上下文学习研究。

![](images/2.jpg)  
(c) provides inferencing with fine-tuned MLLM.

我们提出了一种两阶段的细粒度微调方法，分别增强多模态语言模型在复杂上下文学习和查询-目标对齐中的能力，促进其对多样化图像检索任务的适应。通过解耦的微调目标，我们的微调方法减少了计算资源的需求。我们在七个涵盖多样化图像检索任务的数据集上进行了广泛的实验，证明了我们方法的显著优越性——在零-shot单检查点设置下，在各种复杂图像检索任务中实现了最先进的性能，即使在采用更轻量的多模态语言模型主干时也是如此。

# 2 相关工作

多模态组合数据集。现有研究[16]已展示多模态组合数据能够增强大语言模型（MLLMs）对多模态输入的理解。这类数据通常类似于CIR样本，采用三元组形式 $<$ 参考图像，修改文本，目标图像 >。早期的CIR数据集[1, 20, 35]主要是人工标注的，由于标注成本过高，因此在规模上受到限制。为了解决这个问题，研究人员提出了几种自动化三元组生成方法，这些方法分为两类。1) 半自动标注。这种策略[14]使用大语言模型对图像对的修改文本进行标注，但依赖于人工干预来选择图像和评估三元组。例如，LaSCo[14]基于VQA2.0[7]数据集构建其CIR数据集。最初，它为每张图像提供24张视觉相似的图像，允许志愿者选择相关图像以形成图像对。随后，它将相关图像对的问答信息输入大语言模型，生成修改文本，并使用人工审查者评估生成的三元组质量。虽然这种方法降低了人工标注成本，但由于涉及人为因素，仍然成本高昂且难以扩展。2) 完全自动标注。这种策略[16]旨在完全自动化三元组数据的生成，而无需人工干预。例如，MCL[16]通过使用大语言模型根据参考图像及其衍生文本生成修改和目标标题，从而生成其MMC数据集。然而，缺乏真实目标图像导致三元组质量较低。相比之下，MagicLens[40]通过首先将同一网页中的图像进行分组，自动识别潜在的参考-目标图像对，然后为每个图像生成各种标注工具的元数据，最后结合图像的CLIP视觉相似性和文本相似性进行潜在配对过滤。然后，这些图像对的元数据被大语言模型处理以生成修改文本。该方法的一个主要限制是生成的元数据仍然是粗粒度的，只捕捉到图像中主要对象的一般和常见属性。为了解决这一问题，我们提出了一种全自动流水线，用于生成细粒度的多模态组合数据集。

# 3 精细化多模态数据集生成

在本节中，我们展示了我们的细粒度多模态五元组数据集生成管道，如图2(a)所示，该管道包含三个阶段：基于CoT的细粒度图像描述生成、基于MLLM的图像对识别、类人细粒度修改生成。

# 3.1 基于链式推理的细粒度图像描述生成。

在这一阶段，我们旨在生成图像的细粒度描述，以提供详细的上下文作为后续基于大语言模型的潜在查询-目标图像配对识别的输入。为此，我们利用ImageNet1K的未标记测试集[6]，该数据集包含10万张未标记的开放领域真实世界图像，涵盖各种主题，作为初始数据集。

![](images/3.jpg)  

Figure 3: Illustration of instructions involved in: (a) CoTbased Instruction and (b) Vagueness-guided Instruction.

为了避免要求多语言大模型（MLLM）在单一推理步骤中生成细粒度的图像描述，我们设计了一种链式思维（CoT）指令，鼓励通过一系列细粒度的推理步骤生成详细的描述，从而确保描述的丰富性和特异性。直观上，人在处理视觉信息时是分阶段进行的——首先识别主要对象，其次注意其细微特征，最后考虑背景信息。因此，如图3(a)所示，我们将链式思维引导的指令结构化为四个关键推理步骤：面向主体的推理、面向属性的推理、面向背景的推理和面向总结的推理。第一步引导MLLM识别主体及其数量，而第二步则专注于输出这些主体的详细属性。在本研究中，我们定义了六种类型的属性——外观、颜色、图案、特征、动作和互动，以指导MLLM捕捉主体的细致特性。接下来，在面向背景的推理步骤中，我们引导模型提供上下文信息，包括场景及其他重要元素。最后，面向总结的推理步骤将前面三步的输出综合起来，形成一个连贯且全面的细粒度图像描述。让 $D _ { I }$ 表示为每个原始图像 $I$ 生成的细粒度描述，该描述为相对较长的文本，平均超过100个词元。

# 3.2 基于多模态大语言模型的图像配对识别

在获得细粒度图像-标题对后，我们开始创建五元组样本。首先，我们需要识别相关的图像对以形成参考-目标对，然后为每对生成修改文本。与之前的CIR数据集生成方法主要依赖于视觉或粗粒度相似性不同，我们使用细粒度语义相似性来选择相关图像对。这种方法的动机是我们观察到，视觉上相似的图像对往往在语义上是不相关的（例如，具有相似视觉风格但内容无关的图像），这与实际用户的检索需求不符。考虑到现有的VLP编码器在处理过于细粒度的文本（例如，我们的细粒度标题）时存在困难，我们转向大型语言模型（LLM）来利用其强大的上下文理解能力。然而，由于LLM是生成模型，并非最初为检索任务设计的，我们提议对其进行微调，以提升其处理细粒度长文本的编码能力。为此，我们利用在3.1小节中获得的细粒度图像-标题对来优化一个MLLM，以完成图像-文本对齐任务。接着，我们利用经过微调的MLLM的LLM部分进行长文本编码，从而实现相关图像对识别的语义相似性评估。具体来说，受到基于解码器的LLM查询-文档检索模型的启发，该模型在给定的词元序列末尾附加一个序列结束（EOS）标记以总结其语义内容，我们为每个图像词元序列及其对应的标题词元序列附加$M$个EOS标记。然后，我们从MLLM的最后隐层状态中平均$M$个EOS标记的嵌入，以获得输入图像/标题的最终表示。这个过程可以表示为：

$$
\mathbf { f } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathrm { L L M } \big ( \mathrm { I n p u t } ; M \cdot [ \mathrm { E O S } ] \big ) \big [ - m \big ] .
$$

随后，我们采用常用的图像-文本对比损失进行微调，其形式化如下：

$$
\mathcal { L } _ { a l i g n } = - \frac { 1 } { 2 B } \sum _ { i = 1 } ^ { B } \left[ \log \frac { \exp \left( s _ { i i } / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( s _ { i j } / \tau \right) } + \log \frac { \exp \left( s _ { i i } / \tau \right) } { \sum _ { j = 1 } ^ { B } \exp \left( s _ { j i } / \tau \right) } \right] .
$$

这里，$s _ { i j } = \cos \bigl \langle \mathbf { f } _ { i } ^ { v } , \mathbf { f } _ { j } ^ { t } \bigr \rangle$ 表示图像 $i$ 和文本 $j$ 之间的相似度。$B$ 是批量大小，$\tau$ 是温度参数。一旦 MLLM 使用上述损失函数得到了充分的训练，就可以用于识别潜在的图像对。为了排除过于相似或不相关的图像对，这些图像对对于实际的修改任务没有用处，我们采用余弦相似度阈值，如 [20, 40] 中所建议的。具体而言，我们定义了一个上限阈值 $\theta _ { h }$ 和一个下限阈值 $\theta _ { l }$，仅保留余弦相似度在此范围内的图像对，如下所示：

$$
\{ \langle I _ { r } , I _ { t } \rangle \ : | \ : \cos \langle \mathbf { f } _ { I _ { r } } , \mathbf { f } _ { I _ { t } } \rangle \in [ \theta _ { l } , \theta _ { h } ] \} .
$$

# 3.3 类人细粒度修改生成

在获得相关的图像对后，我们可以进行自动化的修改文本生成。现有的研究通常直接提示大型语言模型(LLMs)基于图像对的粗粒度信息（例如，全球标题）生成修改文本。然而，这种方法存在两个主要局限性：1）由于缺乏详细输入，影响了对细粒度属性的修改；2）生成的修改文本往往过于精确，例如“将十五人改为七人”和“将深红色替换为浅红色”，这主要是由于LLM强大的推理能力。这种精确度偏离了实际用户在提出修改请求时的模糊表达（例如，“减少人数”和“更浅的颜色”）。为了促进细粒度修改并确保自然表达，我们提出了一种类人的细粒度修改生成方案。不同于以前的研究，我们将图像对的细粒度标题输入到LLM中，使其能够生成更准确的修改文本。此外，我们设计了一种模糊性引导的指令，以鼓励LLM生成类人的修改。如图3(b)所示，我们的指令模拟现实场景，引导模型生成模糊的修改。值得注意的是，我们并不强制LLM始终产生模糊的修改，而是允许其在修改需求高度具体时提供精确的变化（例如，“将猫变为$\deg ^ { \prime \prime }$”）。为了进一步增强LLM对这一指令的理解，我们提供了三个人工标注的示例，涵盖了模糊修改和直接比较，利用了LLM强大的少量学习能力。为了确保生成的修改文本的质量，我们使用了一种强大的LLM，具体为LLaMA $3 . 1 7 0 \mathrm { B } ^ { 1 }$。最终，通过这一数据生成管道，我们构建了一个约87K可扩展五元组的数据集，每个五元组包含<$参考图像，细粒度参考标题，修改文本，目标图像，细粒度目标标题>。

# 4 细粒度多模态微调

在本节中，我们介绍我们的两阶段微调策略，如图 2(b) 所示，包括细粒度上下文推理导向的微调和细粒度检索导向的微调。

# 4.1 面向细粒度上下文推理的微调

MLLMs 本质上缺乏足够的细粒度上下文建模能力，导致在细粒度上下文推理方面存在不足，限制了它们在复杂任务中的有效性。为了解决这一局限，我们通过指令微调执行以细粒度上下文推理为导向的微调。具体而言，类似于 [16]，我们基于生成的多模态五元组构建指令-答案对。指令格式为：“<参考图像> 但经过修改文本修改。请描述新图像”，而答案则对应于细粒度目标标题。值得注意的是，不同于 [16] 简单使用单一的基于 CLIP 的视觉嵌入，我们使用 MLLMs 产生的长序列词元嵌入来表示 <参考图像>。这种方法基于将参考图像视为类似于长文本的想法，提供了两个关键好处：1）鼓励 MLLM 发展细粒度上下文推理；2）增强 MLLM 在其他复杂图像检索任务中的泛化能力，例如长文本到图像和基于对话的图像检索。我们通过生成损失对 MLLM 进行微调，其形式化如下：

$$
\mathcal { L } _ { g e n } = - \sum _ { i = 1 } ^ { N } \log P _ { L L M } ( x _ { i } | \Phi , x _ { 1 : i - 1 } ) ,
$$

其中 $\Phi$ 代表指令的嵌入，$N$ 表示答案的词元嵌入的长度，$x_{i}$ 表示答案的第 $i$ 个词元嵌入。

# 4.2 细粒度检索导向微调

本阶段旨在优化多模态大语言模型（MLLM）的查询-目标对齐能力，在第一阶段增强了其多模态推理能力的基础上进行。具体而言，类似于MCL [16]，我们采用组合图像检索任务进行微调。然而，与MCL旨在将多模态查询特征与单模态目标描述特征对齐不同，我们提出将多模态查询特征与多模态目标特征对齐。特别地，我们将查询格式化为“（参考图像但进行修改，请描述新图像）”而目标格式化为“（目标图像）描述图像”。添加提示“描述图像”形成统一的多模态输入格式，便于跨模态查询-目标对齐。具体而言，我们按照公式(1)推导多模态查询和目标特征，并使用基于批量的InfoNCE损失进行查询-目标对齐优化，可形式化为：

$$
\mathcal { L } _ { I n f o N C E } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp { \left( \cos \langle \mathbf { f } _ { i } ^ { \mathrm { q u e r y } } , \mathbf { f } _ { i } ^ { \mathrm { t a r g e t } } \rangle / \tau ^ { \prime } \right) } } { \sum _ { j = 1 } ^ { B } \exp { \left( \cos \langle \mathbf { f } _ { i } ^ { \mathrm { q u e r y } } , \mathbf { f } _ { j } ^ { \mathrm { t a r g e t } } \rangle / \tau ^ { \prime } \right) } } ,
$$

其中 $B$ 是批量大小，$\tau ^ { \prime }$ 是温度参数。为了进一步增强多模态大语言模型（MLLM）在实现查询与目标的区分对齐能力，我们引入了 Recall@k 替代损失[25]。该损失函数直接约束同一批次中真实目标的排名，有效提升目标检索性能。其公式如下：

$$
\begin{array}{c} \begin{array} { r } { \left\{ \tilde { R } _ { \Omega } ^ { k } ( q ) = \frac { \displaystyle \sum _ { x \in \mathcal { P } _ { q } } \sigma _ { \tau _ { 1 } } \left( k - 1 - \displaystyle \sum _ { z \in \Omega , z \neq x } \sigma _ { \tau _ { 2 } } ( s _ { q z } - s _ { q x } ) \right) } { \displaystyle | \mathcal { P } _ { q } | } , \right.} \\ { \displaystyle \mathcal { L } _ { r e c a l l } ^ { k } = \frac { 1 } { B } \displaystyle \sum _ { i = 1 } ^ { B } ( 1 - \tilde { R } _ { \Omega } ^ { k } ( q _ { i } ) ) , } \end{array}   \end{array}
$$

其中 $s _ { q z }$ 表示查询 $q$ 与候选项 $z$ 之间的相似度得分，$\mathcal { P } _ { q }$ 表示查询 $q$ 的真实匹配（正例）集合，$\Omega$ 表示所有候选项的集合。$\sigma _ { \tau _ { 1 } }$ 和 $\sigma _ { \tau _ { 2 }$ 分别表示带温度参数 $\tau _ { 1 }$ 和 $\tau _ { 2 }$ 的 sigmoid 函数。$\tilde { R } _ { \Omega } ^ { k } ( q )$ 是给定查询 $q$ 的可微分 Recall@k 替代量。总之，我们的总损失可以写为：

$$
\mathcal { L } _ { T o t a l } = \mathcal { L } _ { I n f o N C E } + \sum _ { k \in \mathcal { R } _ { s } } \beta _ { k } \mathcal { L } _ { R e c a l l } ^ { k } ,
$$

其中 $\mathcal { R } s$ 是用于召回优化的 $k$ 值集合。$\{ \beta _ { k } \}$ 是控制已采用召回代理损失贡献的超参数。

# 4.3 使用微调后的大语言模型进行推理

在推理过程中，经过微调的多模态语言模型（MLLM）分别编码查询和候选目标图像。然后，根据目标图像与查询的相似性进行检索。如图 2(c) 所示，经过微调的 MLLM 支持多种查询类型，包括纯文本、纯图像（使用指令描述图像 <image>）以及图像与文本的组合（使用指令描述图像 <image> 并附加 <text>），其中 <image> 和 <text> 由相应的图像和文本词元嵌入替代。

# 5 实验

在本节中，我们首先介绍实验设置，然后提供实验结果。

# 5.1 实验设置

5.1.1 评估数据集。为了全面评估我们方法在各种图像检索任务中的有效性，我们选择了三个复杂的检索任务：CIR、长文本到图像检索和对话式图像检索，以及一个较简单的任务，即短文本到图像检索。对于CIR任务，我们采用了三个常用的数据集，包括两个开放领域的数据集：

Table 1: Performance comparison on CIRR respect to $\mathbf { R } @ k ( \% )$ and $\mathbf { R _ { s u b s e t } } @ k ( \% )$ and CIRCO respect to $\mathbf { m } \mathbf { A } \mathbf { P } @ k ( \% )$ The best model and both dedicated ZS-CIR models and universal retrieval models.   

<table><tr><td rowspan="3">Method</td><td colspan="7">CIRR</td><td colspan="4">CIRCO</td></tr><tr><td colspan="4">R@k</td><td colspan="3">Rsubset@</td><td colspan="4">mAP@k</td></tr><tr><td>k = 1</td><td> = 5</td><td>k = 10</td><td>k = 50</td><td> = 1</td><td> = 2</td><td> = 3</td><td>k = 5</td><td>k = 10</td><td>k = 25</td><td>k = 50</td></tr><tr><td>Pic2Word [29](CVPR&#x27;23)</td><td>23.90</td><td>51.70</td><td>65.30</td><td>87.80</td><td></td><td>−</td><td>−</td><td>8.72</td><td>9.51</td><td>10.46</td><td>11.29</td></tr><tr><td>LinCIR [8](CVPR&#x27;24)</td><td>25.04</td><td>53.25</td><td>66.68</td><td>—</td><td>57.11</td><td>77.37</td><td>88.89</td><td>12.59</td><td>13.58</td><td>15.00</td><td>15.85</td></tr><tr><td>SEARLE-XL-OTI [1](ICCV&#x27;23)</td><td>24.87</td><td>52.31</td><td>66.29</td><td>88.58</td><td>53.80</td><td>74.31</td><td>86.94</td><td>10.18</td><td>11.03</td><td>12.72</td><td>13.67</td></tr><tr><td>SEARLE-XL [1](ICCV&#x27;23)</td><td>24.24</td><td>52.48</td><td>66.29</td><td>88.84</td><td>53.76</td><td>75.01</td><td>88.19</td><td>11.68</td><td>12.73</td><td>14.33</td><td>15.12</td></tr><tr><td>ContextI2W [31](AAAI&#x27;24)</td><td>25.60</td><td>55.10</td><td>68.50</td><td>89.80</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FTI4CIR [17](SIGIR&#x27;24)</td><td>25.90</td><td>55.61</td><td>67.66</td><td>89.66</td><td>55.21</td><td>75.88</td><td>87.98</td><td>15.05</td><td>16.32</td><td>18.06</td><td>19.05</td></tr><tr><td>MagicLens [40](ICML 24)</td><td>30.10</td><td>61.70</td><td>74.40</td><td>92.60</td><td>68.10</td><td>84.80</td><td>93.20</td><td>29.60</td><td>30.80</td><td>33.40</td><td>34.40</td></tr><tr><td>CIReVL(GPT-3.5-turbo) [12]ICLR&#x27;24</td><td>24.55</td><td>52.31</td><td>64.92</td><td>86.34</td><td>59.54</td><td>79.88</td><td>89.69</td><td>18.57</td><td>19.01</td><td>20.9</td><td>21.80</td></tr><tr><td>LDRE(GPT-3.5-turbo) [38](SIGIR&#x27;24)</td><td>26.53</td><td>55.57</td><td>67.54</td><td>88.50</td><td>66.43</td><td>80.31</td><td>90.05</td><td>23.35</td><td>24.03</td><td>26.44</td><td>27.50</td></tr><tr><td>MCL(OPT-2.7B) [16](ICML&#x27;24)</td><td>23.28</td><td>54.17</td><td>67.16</td><td>90.05</td><td>58.24</td><td>79.37</td><td>90.51</td><td>14.55</td><td>15.79</td><td>17.38</td><td>18.27</td></tr><tr><td>MCL(OPT-6.7B) [16](ICML&#x27;24)</td><td>24.15</td><td>55.98</td><td>69.21</td><td>90.82</td><td>59.52</td><td>80.34</td><td>91.13</td><td>14.14</td><td>16.13</td><td>17.88</td><td>18.82</td></tr><tr><td>MCL(LLaMA2-7B) [16](ICML&#x27;24)</td><td>26.22</td><td>56.84</td><td>70.00</td><td>91.35</td><td>61.45</td><td>81.61</td><td>91.93</td><td>17.67</td><td>18.86</td><td>20.80</td><td>21.68</td></tr><tr><td>E5-V(LLaVA-NeXT-8B) [11](Arxiv&#x27;24)</td><td>33.90</td><td>64.12</td><td>75.88</td><td>93.54</td><td>67.48</td><td>81.20</td><td>92.48</td><td>18.48</td><td>19.21</td><td>20.95</td><td>21.83</td></tr><tr><td>FiRE(BLIP-3-4B) (Ours)</td><td>43.33</td><td>74.02</td><td>83.51</td><td>95.83</td><td>73.01</td><td>88.38</td><td>94.94</td><td>31.03</td><td>32.08</td><td>34.40</td><td>35.50</td></tr><tr><td>Ours vs. Dedicated ZS-CIR Model</td><td>↑ 13.23</td><td>↑ 12.32</td><td>↑9.11</td><td>↑3.23</td><td>↑4.91</td><td>↑3.58</td><td>↑ 1.74</td><td>↑ 1.43</td><td>↑ 1.28</td><td>↑ 1.00</td><td>↑ 1.10</td></tr><tr><td>Ours vs. Universal Retrieval Model</td><td>↑ 9.43</td><td>↑9.90</td><td>↑7.63</td><td>↑ 2.29</td><td>↑ 5.53</td><td>↑7.18</td><td>↑2.46</td><td>↑ 12.55</td><td>↑ 12.87</td><td>↑ 13.45</td><td>↑ 13.77</td></tr></table>

CIRR 数据集 [20] 和 CIRCO 数据集 [1]，以及一个时尚领域数据集：FashionIQ，进一步划分为三个子集：裙子、衬衫和上衣&T恤。对于长文本到图像的检索，我们采用了公开可用的 Urban1K 数据集 [39]，该数据集包含 1000 张具有细微差异的城市风景图像，每张图像附带详细的描述。对于基于对话的图像检索，我们使用了 Visual Dialog 数据集 [5]。对于短文本到图像的检索，我们选择了两个经典数据集：COCO [18] 和 Flickr30K [26]。5.1.2 实现细节。多模态大语言模型（MLLMs）固有地消耗大量内存 [11]，而在复杂的图像检索任务中涉及的长输入嵌入进一步加剧了这个问题。因此，为了尽可能减小内存消耗，我们使用了 BLIP-3 [37]，一个只有 40 亿参数的更轻量级 MLLM，作为细粒度数据集生成和通用图像检索的模型主干。

关于超参数，在细粒度数据集生成中，我们在公式(1)中设置结束符数量 $M = 5$，在公式(2)中将图像-文本对比损失的温度系数 $\tau$ 设置为0.01。在公式(3)中，将下限和上限阈值 $\theta _ { l }$ 和 $\theta _ { h }$ 分别设置为0.6和0.83。对于两阶段微调，我们在公式(5)中将 $\tau ^ { \prime }$ 设置为0.01，在公式(6)中将召回替代损失的温度系数设置为 $\tau _ { 1 } = 1$ 和 $\tau _ { 2 } = 0 . 0 1$。此外，在公式(7)中，我们采用了召回率 $@ 1$ 和召回率 $@ 5$ 来增强模型的区分检索能力。参数设置如下：$\mathcal { R } _ { s } = [ 1 , 5 ]$，其中 $\beta _ { 1 } = 0 . 4$ 和 $\beta _ { 2 } = 0 . 1 5$。在所有阶段，我们使用AdamW [21]进行优化。

对于数据生成和模型微调，我们冻结了MLLM的视觉编码器和投影层，仅使用LoRA [9]对LLM进行微调，这是一种轻量级的参数高效微调方法。特别地，我们将LoRA近似的秩设置为64，lora_alpha参数设置为128，lora_dropout参数设置为0.1。为了在我们的数据生成流程中推导出用于相关图像对识别的MLLM编码器，我们将批量大小设置为16，学习率设置为1e-4，并训练模型2个epoch。在第一阶段微调中，我们将学习率设置为$1 \mathrm { e } { - 4 }$，训练模型1个epoch；而在第二阶段微调中，我们将批量大小设置为16，学习率设置为1e-4，并训练2个epoch。此外，我们利用了DeepSpeed ZeRO-2 [28]进行分布式训练。所有训练均使用4个NVIDIA A100-40G GPU进行。值得注意的是，一旦模型训练完成，我们会保留相同的检查点，以便在零-shot设置下对不同任务进行评估。

5.1.3 评估。参考之前的研究 [11, 16, 17]，我们采用标准评估协议对每个数据集验证我们的方法。对于 CIRR，我们计算了排名 $k$ 的召回率 $ (\mathbf { R } @ k )$ $( k \ = \ 1 , 5 , 10 , 50 )$，以及测试集上的 $\mathrm { R } _ { S u b s e t } @ k$ $( k \ : = \ : 1 , 2 , 3 )$。对于 FashionIQ， 我们采用了每个类别的 $\mathrm { R } @ k ( k = 10 , 50 )$ 并报告了平均指标。对于 CIRCO，我们采用平均精度均值 (mAP) 作为指标，具体为 $\mathrm { m A P } @ k$ $( k = 5, 10, 25, 50 )$。对于 Urban1K、Visual Dialog、COCO 和 Flickr，我们使用了 $\mathrm { R @ } k$ $( k = 1 , 5 , 10 )$ 作为评估指标。

# 5.2 关于 CIR 比较

为了进行全面评估，我们不仅将我们的方法与基于MLLM的通用图像检索模型进行了比较，包括E5-V [11]和MCL [16]，还比较了几种专门的零样本CIR方法，分为三类。1) 基于文本反演的方法，包括Pic2Word [29]、LinCIR [8]、SEARLE [1]、Context I2W [31]和FTI4CIR [17]。这些方法旨在预训练一个模型，将图像映射为伪词元，从而将多模态查询统一为一个词元序列，该序列可以由预训练的VLP编码器处理以进行目标图像检索。2) 基于三元组生成的方法，即MagicLens [40]，引入了一个数据生成管道，用于生成大量三元组样本，以训练基于双VLP编码器的CIR模型。值得注意的是，通用图像检索模型MCL和我们提出的模型也属于这一类别。3) 基于LLM的无训练方法，包括CIReVL [12]和LDRE [38]，直接利用强大的LLM根据输入参考图像和修改文本生成目标图像标题，从而将CIR转化为目标文本到图像检索，这可以通过VLP编码器解决。值得一提的是，遵循E5-V [11]，我们在提示中包含了“描述<服装类别>”的子集服装类别，以提升模型在FashionIQ数据集上的表现。

Table 2: Performance comparison on FashionIQ with respect to $\mathbf { R } @ k ( \% )$ The best results are in boldface. We also reported the absoluteperormancprovementbetweurmodeland bothdedicateZ-IRmodel anniveralrerivalels.   

<table><tr><td rowspan="2">Method</td><td colspan="2">Dresses</td><td colspan="2">Shirts</td><td colspan="2">Tops&amp;Tees</td><td colspan="2">Avg</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td>Pic2Word [29](CVPR&#x27;23)</td><td>20.00</td><td>40.20</td><td>26.20</td><td>43.60</td><td>27.90</td><td>47.40</td><td>24.70</td><td>43.70</td></tr><tr><td>LinCIR [8](CVPR&#x27;24)</td><td>20.92</td><td>42.44</td><td>29.10</td><td>46.81</td><td>28.81</td><td>50.18</td><td>26.28</td><td>46.48</td></tr><tr><td>SEARLE-XL-OTI [1](ICCV&#x27;23]</td><td>21.57</td><td>44.47</td><td>30.37</td><td>47.49</td><td>30.90</td><td>51.76</td><td>27.61</td><td>47.90</td></tr><tr><td>SEARLE-XL [1](ICCV&#x27;23)</td><td>20.48</td><td>43.13</td><td>26.89</td><td>45.58</td><td>29.32</td><td>49.97</td><td>25.56</td><td>46.23</td></tr><tr><td>Context-I2W [31](AAAI&#x27;24)</td><td>23.10</td><td>45.30</td><td>29.70</td><td>48.60</td><td>30.60</td><td>52.90</td><td>27.80</td><td>48.93</td></tr><tr><td>FTI4CIR [17](SIGIR&#x27;24)</td><td>24.39</td><td>47.84</td><td>31.35</td><td>50.59</td><td>32.43</td><td>54.21</td><td>29.39</td><td>50.88</td></tr><tr><td>MagicLens [40]CML2</td><td>25.50</td><td>46.10</td><td>32.70</td><td>53.80</td><td>34.00</td><td>57.70</td><td>30.73</td><td>52.53</td></tr><tr><td>CIReVL(GPT-3.5-turbo) [12](ICLR&#x27;24)</td><td>24.79</td><td>44.76</td><td>29.49</td><td>47.40</td><td>31.36</td><td>53.65</td><td>28.55</td><td>48.57</td></tr><tr><td>LDRE(GPT-3.5-turbo) [38](SIGIR&#x27;24)</td><td>22.93</td><td>46.76</td><td>31.04</td><td>51.22</td><td>31.57</td><td>53.64</td><td>28.51</td><td>50.54</td></tr><tr><td>E5-V(LLaVA-NeXT-8B) [11](Arxiv&#x27;24)</td><td>23.75</td><td>47.45</td><td>36.36</td><td>56.43</td><td>35.29</td><td>57.47</td><td>31.80</td><td>53.78</td></tr><tr><td>FiRE(BLIP-3-4B) (Ours)</td><td>29.60</td><td>50.87</td><td>39.84</td><td>60.06</td><td>35.64</td><td>57.83</td><td>35.02</td><td>56.25</td></tr><tr><td>Ours vs. Dedicated ZS-CIR Model</td><td>↑4.10</td><td>↑3.03</td><td>↑7.14</td><td>↑6.26</td><td>↑ 1.64</td><td>↑0.13</td><td>↑4.29</td><td>↑3.72</td></tr><tr><td>Ours vs. Universal Retrieval Model</td><td>↑5.85</td><td>↑3.42</td><td>↑ 3.48</td><td>↑ 3.63</td><td>↑ 0.35</td><td>↑ 0.36</td><td>↑3.22</td><td>↑ 2.47</td></tr></table>

Table 3: Performance comparison on Visual Dialog and Urban1K with $\mathbf { R } @ k ( \% )$ .The best results are in boldface.   

<table><tr><td rowspan="2">Method</td><td colspan="3">Visual Dialog</td><td colspan="3">Urban1K</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>CLIP [27]</td><td>17.7</td><td>38.9</td><td>50.2</td><td>55.8</td><td>79.6</td><td>86.5</td></tr><tr><td>MCL(OPT-2.7B) [16]</td><td>25.6</td><td>51.9</td><td>65.2</td><td>−</td><td></td><td>−</td></tr><tr><td>MCL(OPT-6.7B) [16]</td><td>27.2</td><td>51.0</td><td>64.0</td><td></td><td>−</td><td></td></tr><tr><td>MCL(LLaMA2-7B) [16]</td><td>29.8</td><td>57.1</td><td>69.4</td><td>−</td><td></td><td></td></tr><tr><td>E5-V(LLaVA-NeXT-8B) [11]</td><td>48.1</td><td>74.8</td><td>83.7</td><td>80.6</td><td>93.9</td><td>96.6</td></tr><tr><td>Long-CLIP [39]</td><td>35.4</td><td>62.0</td><td>72.7</td><td>86.1</td><td>96.4</td><td>98.1</td></tr><tr><td>FiRE(BLIP-3-4B) (Ours)</td><td>54.9</td><td>79.9</td><td>88.0</td><td>91.4</td><td>98.0</td><td>99.2</td></tr></table>

表1和表2展示了我们在CIRR、CIRCO和FashionIQ上的结果。我们直接使用了基线原始论文中报告的结果，同时特别复现了最佳的通用图像检索基线E5-V及其公开参数，将其作为主要基准。值得注意的是，所有基线报告的结果均基于ViT-L/14视觉编码器，与我们MLLM中使用的编码器一致。我们还报告了我们的结果与现有最佳专用零样本计算机视觉模型及领先的基于MLLM的通用图像检索模型（即E5-V）相比的改进。从这两张表中，我们得出以下观察结果。1）与基于大型语言模型的通用图像检索器（如MCL和E5-V）相比，我们的方法即使在使用更轻量的MLLM主干网络时，仍在所有指标上持续表现出性能提升。具体而言，CIRR上的$\mathrm { R @ 1 }$平均值、CIRCO上的$\mathrm { m A P } @ 5$以及FashionIQ上的$\mathrm { R @ 10 }$显示，我们的方法与最佳通用基线E5-V相比提升了$8.4\%$。这表明我们模型在各种CIR背景下具有优越的泛化能力。

Table 4: Performance comparison on COCO and Flickr with respect to $\mathbf { R } @ k ( \% )$ The best results are in boldface.   

<table><tr><td rowspan="2">Method</td><td colspan="3">COCO</td><td colspan="2">Flickr</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1 R@5</td><td>R@10</td></tr><tr><td>CLIP [27]</td><td>35.4</td><td>60.1</td><td>70.2</td><td>68.7 90.6</td><td>95.2</td></tr><tr><td>MagicLens [40]</td><td>44.3</td><td>69.4</td><td>78.3</td><td>72.5 91.5</td><td>95.2</td></tr><tr><td>Long-CLIP [39]</td><td>46.3</td><td>70.8</td><td>79.8</td><td>76.1 93.5</td><td>95.2</td></tr><tr><td>E5-V [11]</td><td>52.0</td><td>76.5</td><td>84.7</td><td>79.5 95.0</td><td>97.6</td></tr><tr><td>FiRE (Ours)</td><td>52.3</td><td>76.7</td><td>82.6</td><td>76.2 93.0</td><td>95.5</td></tr></table>

与那些专用的CIR模型相比，我们的通用模型在所有指标上都表现出持续的显著改善，展示了其卓越的多模态上下文理解能力。值得注意的是，尽管基于LLM的无训练方法（即$\mathrm { C I R e V L }$和LDRE）在推理时使用了更大的LLM，其检索性能仍低于我们的。这表明，为了增强其多模态上下文理解能力，进行恰当的LLM微调是必要的。此外，与其他基于三元组生成的方法（即MCL和MagicLens）相比，我们的方法在模型微调中使用的生成三元组数量要少得多（仅87K），而MCL和MagicLens分别利用了$2 . 7 M$和$3 6 . 7 M$生成的三元组。这表明我们生成的数据集质量更高，具有更细致的差异和更接近人工标注的修改文本。 在涉及更细致的服装细节修改的FashionIQ上，我们的方法在裙子和衬衫子集上表现出更卓越的性能，相较于上衣和T恤子集。一种可能的解释是，与涉及多种上衣类别的上衣和T恤相比，裙子和衬衫中的服装更集中于单一服装类别，这要求模型具备更强的细粒度推理能力。在这些情况下，我们模型的优势得以凸显。

Table 5: Ablation study on FashionIQ, CIRR, Visual Dialog, and Urban1K towards for key components of our method.   

<table><tr><td rowspan="2">Method</td><td>FashionIQ-Avg</td><td colspan="2">CIRR</td><td colspan="2">Visual Dialog</td><td colspan="2">Urban1K</td></tr><tr><td>R@10</td><td>R@50 R@1</td><td>R@5</td><td>R@1</td><td>R@5</td><td>R@1</td><td>R@5</td></tr><tr><td>w/-Img-LongCap</td><td>11.24 23.56</td><td>11.33</td><td>33.61</td><td>53.10</td><td>77.82</td><td>91.70</td><td>98.40</td></tr><tr><td>w/-OneStage</td><td>32.72 53.60</td><td>41.67</td><td>72.35</td><td>53.21</td><td>78.99</td><td>87.90</td><td>96.20</td></tr><tr><td>w/o-FirstStage</td><td>32.54 52.06</td><td>41.79</td><td>71.59</td><td>54.12</td><td>79.31</td><td>88.20</td><td>97.40</td></tr><tr><td>w/-ShortCap</td><td>32.21 52.63</td><td>42.33</td><td>72.72</td><td>53.45</td><td>78.97</td><td>88.90</td><td>97.20</td></tr><tr><td>w/o-RecallLoss</td><td>33.14</td><td>53.69 41.21</td><td>72.25</td><td>53.88</td><td>79.12</td><td>87.90</td><td>97.20</td></tr><tr><td>FiRE (Ours)</td><td>35.02</td><td>56.25</td><td>43.33</td><td>74.02</td><td>54.88</td><td>79.89</td><td>91.40 98.00</td></tr></table>

Table 6: Performance comparison of dataset on CIRR. The best zero-shot results are in boldface.   

<table><tr><td rowspan="2">Supervision</td><td rowspan="2">Dataset</td><td rowspan="2">Scale</td><td colspan="3">R@k</td><td rowspan="2">RSubset@1</td></tr><tr><td> = 1</td><td>k = 5</td><td> = 10</td></tr><tr><td rowspan="3">Zero-Shot</td><td>MMC [16]</td><td>2.7M</td><td>21.74</td><td>51.54</td><td>65.33</td><td>49.28</td></tr><tr><td>LaSCo [14]</td><td>359.2K</td><td>23.98</td><td>53.68</td><td>67.40</td><td>51.06</td></tr><tr><td>FiGMaQ(Ours)</td><td>87K</td><td>26.96</td><td>55.52</td><td>70.24</td><td>55.66</td></tr><tr><td>Supervised</td><td>CIRR</td><td>28.2K</td><td>28.17</td><td>57.51</td><td>71.74</td><td>58.77</td></tr></table>

# 5.3 关于跨模态检索的比较

除了CIR，我们还使用三个跨模态图像检索任务对我们的模型进行了评估，包括两个相对复杂的任务（即长文本到图像检索和基于对话的图像检索）以及一个标准的短文本到图像检索任务。

对于这两个复杂的检索任务，据我们所知，目前没有强有力的零样本基线。因此，除了两个通用图像检索模型（即 MCL 和 E5-V）外，我们引入了 Long-CLIP [39]，一个专用的零样本长文本检索模型，作为基线。表 3 显示了在基于对话的图像检索数据集 Visual Dialog 和长文本到图像检索数据集 Urban1K 上的性能比较。如所示，尽管没有针对这两个任务进行专门训练，我们的方法仍然表现出卓越的性能。一方面，这验证了进行细粒度上下文建模本质上增强了 MLLM 对复杂查询的理解能力。另一方面，这表明通过 CIR 任务微调 MLLM 对改进复杂图像检索任务有显著贡献。我们将此归因于 CIR 任务本身具有多模态复合查询的复杂性，这可能提升了模型对复杂查询的理解能力。

表4展示了不同模型在标准短文本到图像检索数据集（即COCO和Flickr）上的性能比较，我们排除了MCL，因为其在这些数据集上的结果缺失，而是纳入了已报告相应结果的CLIP和MagicLens。由于Long-CLIP在零-shot跨模态检索任务中表现出色，我们也将其纳入此次比较。可以看出，包括E5-V和我们的模型在内的基于LLM的模型，均优于所有基于VLP的模型。这表明使用LLM作为编码器相比于VLP模型具有优势。关于我们的模型在Flickr上稍微落后于E5-V的观察，我们归因于两个关键因素：1) E5-V利用了一个具有80亿参数的大型LLM作为主干，而我们的模型仅有40亿参数；2) E5-V专门针对短文本对进行了训练，使其更适合短文本到图像检索场景，而我们的模型则是在长文本、细粒度文本对上训练，以增强其对复杂上下文的理解能力。尽管如此，我们的模型在COCO数据集上的表现仍与E5-V相当，证明其在较简单图像检索任务中的有效性。值得再次提及的是，我们的模型在五个数据集上的各种复杂检索任务中，明显优于E5-V。

![](images/4.jpg)  

Figure 4: Modification text generated by our method and its two variants: w/-ShortCap and w/o-VagueInstruct.

# 5.4 消融研究

为了验证我们方法中每个组件的重要性，我们将其与以下派生方法进行了比较。

• w/-Img-LongCap. 为了探索使用 CIR 任务进行 MLLM 微调的影响，我们用标准的图像-文本对齐任务对 MLLM 进行了微调，而不是使用两个多模态推理和检索任务。具体而言，仅使用了 $< i m$ - age, 生成的细粒度标题 $>$ 对。 • w/-OneStage. 为了探讨进行两阶段微调的好处，我们通过在一个阶段同时优化多模态推理和检索任务，模拟了 MCL。 • w/o-FirstStage. 为了探讨注重细粒度上下文推理的微调阶段的作用，我们禁用了该阶段。 • w/-ShortCap. 为了验证在第一阶段微调中使用细粒度标题进行多模态上下文推理的必要性，我们用由 BLIP-2 [15] 生成的粗粒度标题 [34] 替换了细粒度标题。 • w/o-RecallLoss. 为了探索召回替代损失的影响，我们在微调模型时未使用这些损失。

根据表5，我们有以下观察结果。1）w/-Img-LongCap在CIR和基于对话的图像检索任务上的性能明显弱于我们的方法，但在长文本到图像检索任务（Urban1K）上略显优势。这验证了利用CIR任务及其衍生任务（即目标描述生成任务）进行微调，由于其复杂性和多模态特性，更好地促进了MLLM的推理和上下文理解能力的提升，相较于简单的跨模态图像文本对齐。在Urban1K上，w/-Img-LongCap表现更好也合理，因为其微调目标与长文本到图像检索任务完全一致。2）我们的方法在性能上优于w/-OneStage和w/o-FirstStage。这表明顺序进行针对细粒度上下文推理的微调和针对检索的微调的重要性，以确保模型在各种图像检索任务上的通用性能。3）w/-ShortCap在所有任务上的表现都逊色于我们的模型。这突显了使用细粒度描述来提高模型复杂上下文推理能力的好处。4）与我们的方法相比，w/o-RecallLoss表现不佳。这证实了召回替代损失在增强模型的判别查询-目标对齐能力方面的作用。

# 5.5 关于数据集比较

定量比较。为了验证我们数据集FiGMaQ的质量，参考文献[16]，我们使用它来训练经典的基于CLIP编码器的CIR模型Combiner [2]。为了对比，我们还采用了两个大型公开可用的CIR数据集：MMC（MCL使用的数据集）和LaSCo [14]，其修改文本也通过LLM自动标注，以训练Combiner模型。表6展示了在开放领域CIRR数据集的测试集上，使用不同数据集训练的Combiner模型的零样本性能。值得注意的是，我们还包含了在监督设置下Combiner的性能，其中Combiner使用CIRR的训练集进行训练。可以看出，尽管我们的数据集规模最小，但它实现了最佳的零样本性能。同时，我们观察到，使用我们数据集训练的Combiner的零样本性能接近于在传统监督设置下训练的表现。这验证了我们数据集的高质量，原因如下：1）我们提出的细粒度语义过滤显著减少了不相关配对的包含。2）我们类人化的细粒度修改生成方法有效模拟了真实人类的修改，使生成的数据更符合人类标注。

定性比较。为了深入了解我们生成的数据集，我们比较了我们模型及其两个变体生成的修改文本。1) w/-ShortCap。基于 BLIP-2 [15] 生成的粗粒度图像标题，使用 LLaMA 3.1 生成修改文本。2) w/0-VagueInstruct。使用 LLaMA 3.1 在给定的细粒度图像标题对上生成基于一般指令的修改文本。图 4 通过一个示例展示了修改文本的比较，同时提供了生成的粗粒度和细粒度图像标题以供参考。如图所示，与细粒度标题相比，BLIP-2 生成的粗粒度图像标题在某些细节上存在信息丢失，导致 LLM 生成的修改文本“针对整个浴室场景”过于笼统，无法有效检索目标图像。此外，我们发现由 w/o-VagueInstruct 生成的修改文本过于细致，几乎是直接描述目标图像。这可能会产生偏差三元组，阻碍模型的微调，并与现实场景相悖，用户通常提供模糊的修改，关注于图像的关键方面，而不是逐一详细描述图像的每一个元素。

# 5.6 案例研究

检索结果。图5展示了我们在CIR和长文本到图像检索任务中的检索结果，并与最佳表现的通用图像检索模型E5-V [11]进行了比较。如图5(a)所示，给定的复合查询要求对参考图像进行细致的修改，涉及多个方面，例如主题数量、空间关系和细微细节（例如包含一个绳子）。在这种情况下，我们的方法准确地首次检索到了真实标注数据，而E5-V则未能做到，其检索到的图像无法完全满足修改需求，例如数量和空间的变化未得到满足。至于图5(b)所示的长文本到图像检索案例，E5-V的表现明显不如我们的模型。具体而言，E5-V主要检索与查询的一般描述相符的图像，这不足以检索到正确的图像。相比之下，我们的方法检索到的图像不仅与整体描述匹配，还满足查询中的细粒度锚点，例如黄色出租车、雨天或穿着特定服装的行人。这两个案例展示了我们的方法在细粒度上下文推理中的有效性。

![](images/5.jpg)  

Figure 5: Illustration of CIR and Long-Text-to-Image Retrieval results, with ground-truth images highlighted in green boxes.

# 6 结论与未来工作

在本研究中，我们提出了一种自动化流程，用于构建细粒度多模态五元组数据集，并针对复杂图像检索任务提出了一种新颖的两阶段微调策略。通过该流程，我们创建了一个大规模数据集FiGMaQ，以增强细粒度上下文建模。我们的策略将微调划分为两个阶段：（1）面向上下文推理的微调和（2）面向检索的微调，逐步提升上下文理解和查询-目标对齐。在五个复杂和两个简单图像检索任务中的广泛实验验证了我们方法的有效性。消融实验和案例研究进一步证明了细粒度微调的价值。未来的工作将集中于扩展数据集，并探索通用多模态重新排序算法以提升检索精度。

# 7 致谢

本研究得到了中国国家自然科学基金的支持（编号：62376137、624B2047、62376140、62276155和U23A20315）、山东省自然科学基金的支持（编号：ZR2022YQ59）。本研究还得到了研究影响基金（编号：R1015-23）、协作研究基金（编号：C1043-24GF）、华为（华为创新研究计划、华为奖学金）、腾讯（CCF-腾讯开放基金、腾讯犀牛鸟专项研究计划）、阿里巴巴（CCF-阿里妈妈科技袋鼠基金编号：2024002）、蚂蚁集团（CCF-蚂蚁研究基金）和快手的支持。

# References

[1] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. 2023. Zero-Shot Composed Image Retrieval with Textual Inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE, 1533815347.   
[2] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. 2023. Composed image retrieval using contrastive learning and task-oriented clipbased features. ACM Transactions on Multimedia Computing, Communications and Applications 20, 3 (2023), 124.   
[3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. In Advances in neural information processing systems. 18771901.   
[4] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2024. Research 25, 70 (2024), 153.   
5  S   u A gh, D Y, Jos IEEE conference on computer vision and pattern recognition. 326335.   
[6] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imaenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition. Ieee, 248255.   
[7] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. [n. .]. Making he  in va matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition. IEEE, 6904-6913.   
[8] Geonmo Gu, Sanghyuk Chun, Wonjae Kim, Yoohoon Kang, and Sangdoo Yun. 2024. Language-only training of zero-shot composed image retrieval. In Proceedi f the  Conference on Computer ision and Patter Reconition. IEEE, 1322513234.   
[9 wd Hu, Yeo She, hi Was, yleZu, azi, en W   Wi h   Lo p language models. arXiv preprint arXiv:2106.09685 (2021).   
[10] Surgan Jandial, Pinkesh Badjatiya, Pranit Chawla, Ayush Chopra, Mausoom Sarkar, and Balaji Krishnamurthy. 2022. SAC: Semantic Attention Composition for Text-Conditioned Image Retrieval. In Proceedings of the IEEE Winter Conference on Applications of Computer Vision. IEEE, 40214030.   
[11] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. 2024. E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580 (2024).   
[12] Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. 2023. Vision-by-language for training-free compositional image retrieval. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 1-15.   
[1] Jing Yu Koh, Ruslan Salakhutdinov, and Daniel Fried. 03. Grounding lan-International Conference on Machine Learning. PMLR, 1728317300.   
[14] Matan Levy, Rami Ben-Ari, Nir Darshan, and Dani Lischinski. 2024. Data roaming and quality assessment for composed image retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 29912999.   
language-image pre-training with frozen image encoders and large language models. In Proceedings of the International Conference on Machine Learning. PMLR, 1973019742.   
[16] Wei Li, Hehe Fan, Yongkang Wong, Yi Yang, and Mohan Kankanhalli. 2024. Improving Context Understanding in Multimodal Large Language Models via Multimodal Composition Learning. In Proceedings of the International Conference on Machine Learning. PMLR, 121.   
[17] Haoqiang Lin, Haokun Wen, Xuemeng Song, Meng Liu, Yupeng Hu, and Liqiang Nie. 2024. Fine-grained Textual Inversion Network for Zero-Shot Composed Image Retrieval. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 240250.   
[18] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In Proceedings of the European Conference on Computer Vision. Springer, 740755.   
, J o,  Z W n W Xe. 0. - shot composed text-image retrieval. arXiv preprint arXiv:2306.07272 (2023).   
[20] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. 2021. Image retrieval on real-life images with pre-trained vision-and-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. IEEE, 21252134.   
[21] I Loshchilov. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).   
[22] Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024. Fine-Tuning LLaMA for Multi-Stage Text Retrieval. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 24212425.   
[23] Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. 2024. Generative representational instruction tuning. arXiv preprint arXiv:2402.09906 (2024).   
[24 Lo Og, J , X J, o m,  Wih, mea Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. (2022), 2773027744.   
[25] Yash Patel, Giorgos Tolias, and Jirí Matas. 2022. Recall@k Surrogate Loss with Large Batches and Similarity Mixup. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 74927501.   
[26] Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2016. Flickr30k Entities: Collecting Regionto-Phrase Correspondences for Richer Image-to-Sentence Models. (2016).   
[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. In Proceedings of the International Conference on Machine Learning. PMLR, 87488763.   
S   , O, n x He.  Z: Memory Optimization Towards Training A Trillion Parameter Models. arXiv preprint arXiv: 1910.02054 (2019).   
[29] Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate S, nd Toms iser. 0.Wor Mapp us  Wors o Zero-shot Composed Image Retrieval. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 1930519314.   
[30] Xuemeng Song, Fuli Feng, Jinhuan Liu, Zekun Li, Liqiang Nie, and Jun Ma. 2017. of the ACM international conference on Multimedia. 753761.   
[31] Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Yue Hu, and Qi Wu. 2024. Context-I2W: Mapping Images to Context-dependent Words for Accurate Zero-Shot Composed Image Retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence. AAAI, 5180-5188.   
[32] Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. 2019. Composing Text and Image for Image Retrieval - An Empirical Odyssey. In Proceedings of the IEEE conference on computer vision and pattern recognition. IEEE, 64396448.   
[33] Junyan Wang, Peng Zhang, Cheng Zhang, and Dawei Song. 2019. Scss-lie: A novel synchronous collaborative search system with a live interactive engine. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 13091312.   
[34] Haokun Wen, Xuemeng Song, Xiaolin Chen, Yinwei Wei, Liqiang Nie, and Tat-Seng Chua. 2024. Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 229239.   
[35] Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Iae y Natural ngg Feebac. In rocing  the  Cone Computer Vision and Pattern Recognition. IEEE, 1130711317.   
[36] Xiaohui Xie, Jiaxin Mao, Yiqun Liu, and Maarten de Rijke. 2020. Modeling user behavior for vertical search: images, apps and products. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 24402443.   
[37] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S. Ryoo, Shrikant Kendre, Jieyu Zhang, Can Qin, Shu Zhang, Chia-Chih Chen, Ning Yu, Juntao Tan, Tulika Manoj Awalgaonkar, Shelby Heinecke, Huan Wang, Yejin Choi, Ludwig Schmidt, Zeyuan Chen, Silvio Savarese, Juan Carlos Niebles, Caiming Xiong, and Ran Xu. 2024. xGen-MM (BLIP-3): A Family of Open Large Multimodal Models. arXiv preprint arXiv:2408.08872 (2024).   
[38] Zhenyu Yang, Dizhan Xue, Shengsheng Qian, Weiming Dong, and Changsheng Xu. 2024. LDRE: LLM-based Divergent Reasoning and Ensemble for Zero-Shot Composed Image Retrieval. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 8090.   
[39] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. 2024. Long-CLIP: Unlocking the Long-Text Capability of CLIP. In Proceedings of the European Conference on Computer Vision. Springer, 310325.   
[40] Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu Su, and Ming-Wei Chang. 2024. Magiclens: Self-supervised image retrieval with open-ended instructions. In Proceedings of the International Conference on Machine Learning. PMLR, 118.   
[41] Liangli Zhen, Peng Hu, Xu Wang, and Dezhong Peng. 2019. Deep supervised cross-modal retrieval. In Proceedings of the IEEE conference on computer vision and pattern recognition. IEEE, 10394-10403.