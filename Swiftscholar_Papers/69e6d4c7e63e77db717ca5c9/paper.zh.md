# Magic-MM-Embedding：面向视觉-令牌高效的通用多模态嵌入与MLLMs

齐利 颜哲 赵永新 周亚蒙 王言冬 杨员佳 周绝 王佐建 王金香 刘* 荣誉设备有限公司

# 摘要

多模态大型语言模型（MLLMs）在通用多模态检索中表现出巨大的潜力，该目标是为给定查询找到相关的各种模态项目。然而，由于处理大量视觉输入时所产生的高计算成本，它们的实际应用常常受到限制。本文提出了Magic-MM-Embedding，这是一系列新颖的模型，在通用多模态嵌入中实现了高效率和最先进的性能。我们的方法建立在两个协同支柱上：（1）一个高效的MLLM架构，结合视觉词元压缩，显著减少推理延迟和内存占用；（2）一种多阶段渐进式训练策略，旨在不仅恢复而且显著提升性能。这种粗到细的训练范式始于广泛的持续预训练，以恢复多模态理解和生成能力，接着进行大规模对比预训练和难负例挖掘，以增强区分能力，最终在由MLLM作为评测者指导的数据策划下进行任务感知的微调阶段。综合实验表明，我们的模型在推理效率更高的同时，显著超越现有方法。

![](images/1.jpg)  
Figure 1Breaking the eficiency-performance trade-off for MLLM embedders for universal multimodal retrieval. (a) Standard MLLM-based embedders suffer from high computational costs due to processing redundant, dense visual token sequences. b) We propose a visual token compression model paired with a robust three-stage progressive training strategy. (c) Comparisons on MMEB [35] demonstrate that our approach establishes a new k .

# 1 引言

多模态嵌入模型旨在将异构数据模态（如文本、图像）投射到一个语义空间中。这些模型广泛应用于多个领域，如多模态检索、推荐系统和检索增强生成。最近，这一领域经历了显著的范式转变，从双塔架构（如 CLIP 和 UnilR）向具有更强多模态理解能力的多模态大型语言模型（MLLMs）发展。这一转变是由于双塔架构的内在局限性：（1）模态无关的编码架构和特征后处理缺乏跨模态交互，限制了进行精细多模态推理的能力；（2）有限的语言理解能力，刚性的上下文约束和有限的先前知识限制了对复杂语义的理解。相反，基于 MLLM 的方法在文本和视觉 tokens 上共同处理。这促进了深层次 token 级别的跨模态融合，而不是浅层的全局结合。借助于大型语言模型的广泛世界知识和强大的指令跟随能力，这些模型能够在多种场景中执行复杂的多模态检索。在这些优势的基础上，近期研究通过增强数据规模和质量、梯度放大负样本、负样本策略的优化、专家知识蒸馏、超参数调优、比率思维和模型学习，以及在推理过程中与重排序器的协调，迅速推动了基于 MLLM 的通用多模态嵌入方法的进展。然而，这些进展忽视了一个关键瓶颈：长视觉 token 序列的高昂成本。标准的 MLLM 架构通常采用全序列集成策略，其中密集的输入流被处理。以广泛使用的 LLaVA-1.5 为例，它将标准 $336 \times 336$ 图像分割成 576 个视觉 token，所有这些 token 都直接输入到语言模型中。尽管这种全序列注入对 OCR 等精细生成任务有益，但它却提高了处理这些冗余视觉 tokens 的计算成本，随着序列长度的增加，成本呈二次增长，而这些 token 对最终嵌入语义质量的贡献往往是微不足道的。因此，这一过程实际上成为了部署基于 MLLM 的大规模应用系统的主要障碍。

为了应对在提供高性能的同时计算效率低下的挑战，我们提出了一个新颖的框架，该框架在架构设计上实现了创新。我们采用了一种无参数的空间插值模块，将长视觉序列投影为压缩形式，降低了 $75\%$ 的词元开销，同时避免了可学习抽象器的优化困难。为了减轻激进压缩可能导致的性能下降，并学习多模态基础能力恢复，我们首先在通用多模态指令数据集上进行生成性继续训练。这一阶段与语言模型紧密对接，确保基础多模态理解和生成能力的保持。其次进行多模态对比预训练。我们利用 1600 万多模态样本构建了一个通用嵌入器，并通过对比训练进行训练。这一阶段旨在通过从对比热身演进到基于检索的困难负样本挖掘的自我提炼阶段，培养强大的通用表示能力。最后进行任务感知微调。我们通过检索和策划策略在准确的高质量多任务数据集上进一步细化模型。利用前一阶段的模型，我们为训练集的每个查询检索候选样本，并利用一个多语言大型模型作为评审构建候选集。最终嵌入模型处理多样且复杂的场景。通过大量实验，我们的方法实现了接近最先进的效果。这种卓越的性能在显著的词元效率下取得，仅使用四分之一的视觉词元，验证了我们所提炼的压缩和迁移方法的强大能力。我们的贡献有三个：我们提出了一个成功调和效率与性能的框架，用于基于多语言大型模型的通用嵌入。我们证明，当有专门的高级训练管道支持时，具有激进视觉词元压缩的模型可以显著超越其未压缩的对应物。

![](images/2.jpg)  
Figure:Overview the proposed visual-token-cient architecture orniversalmultimodal retrieval.) The proposed MLLM architecture with Visual Token Compression, InternVL3-VTC. (b, c) The proposed inferenceefficient, universal multimodal embedder and reranker, both of which are built upon InternVL3-VTC.

我们提出了一种专门针对压缩大语言模型的粗到细训练策略。该流程提供了系统有效的方法论，用于恢复基础能力，构建强大的判别能力，并通过来自大语言模型作为评判者的精心策划的数据实现强大的多任务泛化。通过广泛的实验，我们展示了我们提出的模型建立了新的最佳状态结果，验证了其优越性，显示出高度的有效性。

# 2 相关工作

多模态表示学习最初是通过CLIP风格的模型而受到广泛关注。这些模型采用图像-文本双编码器架构，因此仅支持双向的文本-图像检索。在此基础上，UniR和MagicLens等方法融合了两个塔的特征，将模型输入从单一模态扩展到交错的图像-文本内容。然而，这些方法本质上仍然遵循旧有的范式：它们首先独立处理每种模态，然后再利用表示，这限制了它们捕捉细粒度跨模态关系的能力。此外，这些模型使用BERT风格的文本编码器，缺乏足够的现实世界知识，并且输入长度限制严格，导致在复杂文本理解上表现不佳。

与CLIP风格的主干网络相比，多模态大语言模型（MLLMs）本质上支持交错的文本和图像输入，展现出更强的多模态理解和指令跟随能力。得益于MLLMs的快速进展，基于MLLM的多模态嵌入范式迅速出现。E5-V以文本到文本的方式训练MLLM的语言组件，实现了零样本多模态检索。然而，针对多模态检索任务的评估仍然有限。VLM2Vec引入了MMEB，这是第一个全面的多任务多模态嵌入训练和评估基准。VLM2Vec将MLLM带入对比学习框架，利用指令跟随和多模态推理能力。通过对ME的训练，达到强大的多任务实现，跨越广泛的检索任务。为了进一步提高区分能力，后续工作系统地进行了广泛的优化，包括提高数据规模和质量，放大困难负样本的梯度，细化困难负样本挖掘策略，提取专家知识，采用多阶段渐进训练，引入思考和强化学习，并在推理期间与重排序器协调。通过这些社区努力，基于MLLM的多模态嵌入模型的区分能力显著提高。然而，由于这些模型直接采用通用的MLLM架构，由视觉词元冗余引起的高推理成本问题尚未引起关注或找到解决方案。

# 3 方法论

我们的目标是开发一个基于多语言大模型（MLLM）的高效且有效的通用多模态嵌入模型用于检索。所提议架构的概述如图所示。我们首先正式定义通用多模态检索任务，然后详细介绍我们的框架，包括架构修改和渐进式训练流程。

# 3.1 准备工作

我们将通用多模态嵌入的学习表述为在共享语义空间内的统一映射函数。令 $\mathcal{X}$ 中的 $x \in \mathcal{X}$ 为查询 $q$ 或候选项 $c$，它由任务指令、视觉上下文和文本上下文组成。查询和候选项的相应输入模板如下，针对不同数据集的任务指令在表 11 和表 12 中显示。

<table><tr><td>Query Template:</td><td>I Candidate Template:</td></tr><tr><td>Instruct: {query instruction}</td><td>I Instruct: {target instruction} I</td></tr><tr><td>&lt;image&gt;</td><td>I &lt;image&gt;</td></tr><tr><td></td><td>I Text: {target text}</td></tr><tr><td>Query: {query text}</td><td>I</td></tr></table>

为了获得多模态嵌入，我们采用一种具有视觉词元压缩的多语言模型（MLLM）作为编码器 $f : \mathcal { X } $ $\mathbb { R } ^ { L \times D }$，以将输入 $x \in \mathcal { X }$ 映射为 $\mathbf { h } _ { 1 } , \mathbf { h } _ { 2 } , \ldots , \mathbf { h } _ { L }$，其中每个隐藏状态 $\mathbf { h } _ { i } \in \mathbb { R } ^ { D }$。我们对最后一个词元的隐藏表示 $\mathbf { h } _ { L }$ 应用 $\ell _ { 2 }$ 归一化，以获得最终的嵌入 $\mathbf { z } _ { x }$：

$$
\mathbf { z } _ { x } = \frac { \mathbf { h } _ { L } } { \| \mathbf { h } _ { L } \| _ { 2 } } .
$$

为了学习判别性嵌入空间，我们采用了 InfoNCE 损失 [81] 进行模型训练。对于给定的查询 $q$，我们定义候选集合 $\mathcal { C } _ { q } = \overset { \cdot } { \{ c _ { q } ^ { + } \} } \cup \mathcal { C } _ { q } ^ { - }$ 进行损失计算，其中 $c _ { q } ^ { + }$ 表示与 $q$ 相关联的真实正目标，而 $\mathcal { C } _ { q } ^ { - }$ 是负样本的集合。每个 $c _ { q } ^ { - } \in \mathcal { C } _ { q } ^ { - }$ 代表通过批次内采样或困难负样本挖掘获得的负样本。模型的训练目标是通过最小化以下目标，从而最大化查询与正目标之间的语义对齐，同时抑制负样本：

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \log \frac { \exp ( \mathbf { z } _ { q } ^ { \top } \mathbf { z } _ { c _ { q } ^ { + } } / \tau ) } { \exp ( \mathbf { z } _ { q } ^ { \top } \mathbf { z } _ { c _ { q } ^ { + } } / \tau ) + \sum _ { c _ { q } ^ { - } \in \mathcal { C } _ { q } ^ { - } } \exp ( \mathbf { z } _ { q } ^ { \top } \mathbf { z } _ { c _ { q } ^ { - } } / \tau ) } ,
$$

其中 $\tau$ 为温度，$( \cdot ) ^ { \top } ( \cdot )$ 表示点积相似度。

# 3.2 无参数视觉词元压缩

标准多模态语言模型范式。设 $\mathcal{T}$ 表示输入图像的空间。标准多模态语言模型通常依赖于视觉编码器 $e_{v} : \mathcal{T} \rightarrow \mathbb{R}^{H \times \widetilde{W} \times C}$，将图像 $\mathbf{I} \in \mathcal{Z}$ 编码为特征图 $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$，其中 $H \times W \times C$ 表示图像的高度、宽度和通道数。在传统方法中，$\mathbf{F}$ 被展平为 $N = H \times W$ 个词元的序列，并投影到语言模型的输入连接器中。然而，长视觉词元序列由于语言模型注意力机制的平方复杂度而引入了显著的计算瓶颈。视觉词元压缩通过插值策略缓解了这一词元超载问题，我们引入了一个在视觉编码器与连接器之间插入的参数自由视觉压缩模块。与复杂的学习压缩方案不同，我们在空间维度上采用直接的双线性插值策略[44, 103]，通过双线性下采样操作 $\Phi(\cdot)$ 将空间分辨率降低 $s$ 倍。压缩后的特征图 $\mathbf{F^{\prime}}$ 的计算公式为：

$$
\mathbf { F } ^ { \prime } = \Phi ( \mathbf { F } ; H ^ { \prime } , W ^ { \prime } ) \in \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times C } ,
$$

目标空间维度为 $H ^ { \prime } = H / s$ 和 $W ^ { \prime } = W / s$。压缩后的映射 $\mathbf { F ^ { \prime } }$ 被展平为视觉词元序列，并输入到连接器中进行投影。此操作将视觉词元的总数从 $N$ 降低至 $N / s ^ { 2 }$，同时保留了空间布局。通过减少等效的二次项，可以显著降低推理延迟和内存消耗，而无需引入任何参数。

# 3.3 渐进式粗到细训练流程

尽管我们的模型架构通过减少视觉词元显著提高了效率，但直接训练这种共压缩模式无法达到最佳性能，原因在于潜在的方向性。因此，我们设计了一个包括三个独立阶段的训练管道：生成恢复、对比重训练和任务感知微调。阶段1：多模态基础能力恢复。插值模块的引入改变了空间结构和密度，使得预训练的LLM主干网络的预期发生变化。因此，第一阶段的主要目标不是检索，而是对齐。为此，我们重新对齐压缩的视觉表征与LLM的语义空间。通过在通用多模态指令跟随数据集上的生成训练，我们恢复了基本的多模态理解和生成能力。对于文本响应的词元序列 $y _ { 1 } , y _ { 2 } , \dotsc , y _ { T }$，模型使用标准的自回归下一个词预测（NTP）损失进行优化：

$$
\mathcal { L } _ { \mathrm { N T P } } = - \sum _ { t = 1 } ^ { T } \log P ( y _ { t } | y _ { < t } , x ) ,
$$

其中 $y_{t}$ 是真实的下一个词元，$x \in \mathcal{X}$ 表示由视觉和文本上下文组成的多模态输入。这个步骤对于弥合因词元压缩造成的分布差距至关重要，确保大语言模型在过渡到嵌入学习之前保持其推理能力。

阶段二：多模态对比预训练。在恢复了多模态基础能力后，我们将模型的重点转向多模态表示学习。该阶段在一个大规模多模态检索语料库上进行，并分为两个步骤逐步增加难度。我们首先通过使用标准的InfoNCE损失和批内负样本对模型进行热身，如公式（2）所示。随后，为了鼓励模型学习更强的区分能力，我们引入全局硬负样本挖掘策略，在训练集中注入硬负样本，并进行新一轮训练。与使用随机批内负样本的热身阶段不同，对于每个子数据集，我们从所有候选样本中为每个查询挖掘负样本。具体而言，对于每个查询 $q$ ，我们检索一个候选样本的排名列表。我们排除真实正样本 $c_{g}^{+}$ ，以帮助避免在排名前列的结果中常见的假阴性（例如排名前10），同时保持负样本比随机批负样本更具挑战性。

阶段三：任务感知微调，使用大型语言模型作为评判者。最后阶段专注于增强模型以处理多样化场景和复杂任务。标准训练数据集通常存在假负例，并且缺乏足够具有挑战性的负例。为了解决这个问题，我们进一步利用专家级的大型语言模型作为评判者来执行数据整理并生成高保真度的困难负例。具体而言，对于目标训练集中每个查询 $q$，我们执行一个“检索-评判”过程。首先，我们使用阶段二的模型检索前 $K$（$K = 20$）个候选项 $\mathcal{C}_{\mathrm{ret}}$。我们将每对 $(q, c_{i})$，其中 $c_{i} \in \mathcal{C}_{\mathrm{ret}}$，输入到 Qwen3-VL [1] 中，使用以下判断模板来评估它们的相关性。每个数据集使用的判断指令如表 13 所示。{判断指令} 查询: {query} 目标: {target} 请做出您的判断。如果相关，请输出“yes”。如果不相关，请输出“no”。您只能输出“yes”或“no”。请严格按照要求输出。

我们接着检查“yes”和${ \mathfrak { c } } _ { \mathtt { n 0 } }$词元的输出logits，以确定其相关性。如果$\log \mathrm { i t } ( \mathbf { y } \mathbf { e } \mathbf { s } ) > \log \mathrm { i t } ( \mathbf { n } \mathbf { o } )$，则该候选被视为相关。这有助于我们发现之前未标注的真实正例，从而扩展正例集超出原始的真实标注正例；如果$\mathrm { l o g i t } ( \mathbf { n } \mathbf { 0 } ) > \mathrm { l o g i t } ( \mathbf { y } \mathbf { e } \mathbf { s } )$，则该候选被视为负例。为了计算对比损失，我们保留原始真实标注$c _ { q } ^ { + }$作为唯一正例，以保持一致性。负样本集$\mathcal { C } _ { q } ^ { - }$与评审确定的困难负例进行增强，这些负例作为具有挑战性的干扰项，迫使模型区分更具混淆性的样本。

# 3.4 协同重排序器

为了构建一个全面的检索系统，参考之前的工作 [59, 23]，我们基于模型阶段训练了一个重排序器，以利用其保留的多模态理解和生成能力。得益于丰富的 MLM 模式，我们采用了点对点重排序方法进行培训。与标准的方法不同，我们训练的是针对查询 $q$ 的评审集合 $\mathcal { C } _ { \mathrm { a u g } } ^ { + } = \{ c _ { q } ^ { + } \} \cup \mathcal { C } _ { \mathrm { j u d g e } } ^ { + }$，其中 $c _ { q } ^ { + }$ 表示真实的正例，而 $\mathcal { C } _ { \mathrm { j u d g e } } ^ { + }$ 表示类似的正例。同时，负例集合为 $\mathcal { C } _ { \mathrm { j u d g e } } ^ { - }$。在点对点重排序的公式下，模型独立评估查询-候选对。我们构建了 $c ^ { + } \in \mathcal { C } _ { \mathrm { a u g } } ^ { + }$ 和 $c ^ { - } \in \mathcal { C } _ { \mathrm { j u d g e } } ^ { - }$。我们指示模型使用以下模板输出正向对的标记 $\cos \cdot$ 和负向对的标记 $^ { \mathfrak { c } } \mathrm { N o } ^ { \mathfrak { s } }$。我将为您提供一个查询和一个候选项。请评估候选项是否满足查询的要求。如果满足，请回复“是”；如果不满足，请回复“否”。查询：{query} 候选项：{target} 点对点损失使用标准的交叉熵 (CE) 损失进行最小化：

$$
\mathcal { L } _ { \mathrm { p o i n t } } = \mathcal { L } _ { \mathrm { C E } } ( \Upsilon \mathbf { e s } , r ( q , c ^ { + } ) ) + \mathcal { L } _ { \mathrm { C E } } ( \mathbb { N } \mathbf { o } , r ( q , c ^ { - } ) ) ,
$$

其中 $r ( \cdot )$ 表示重新排序器的自回归输出过程。从 $\mathcal { C } _ { \mathrm { j u d g e } } ^ { - }$ 中获取 $M$ 个硬负样本 $c _ { 1 } ^ { - } , \ldots , c _ { M } ^ { - }$ （其中 $M \in [ 2 , 5 ]$，正样本 $c ^ { + }$ 来自 $\mathcal { C } _ { \mathrm { a u g } } ^ { + }$）。我们随机将 $c ^ { + }$ 插入列表中的位置 $k$，并提示模型识别出最相关的候选项。列表重排序的输入模板如下：我将向您提供一个查询，后面跟着多个候选项，格式为：(1) cand1 (2) cand2 等。每个候选项相对于其他候选项是独立的。根据查询评估每个候选项，并响应符合查询要求最好的候选项的数字。查询：{query} 候选项：{candidate set} 模型被训练为直接生成正样本候选项的位置索引 $k$。列表损失的公式为：

$$
\mathcal { L } _ { \mathrm { l i s t } } = \mathcal { L } _ { \mathrm { C E } } ( k , r ( q , c _ { 1 } ^ { - } , \ldots , c ^ { + } , \ldots , c _ { M } ^ { - } ) )
$$

最终目标是两个任务的加权和：${ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { p o i n t } } + { \mathcal { L } } _ { \mathrm { l i s t } }$

# 4 数据集构建

第一阶段。在这一阶段，为了恢复令牌压缩模型的多模态理解和生成能力，我们构建了一个包含3200万示例的多模态指令跟随数据集。该语料库由开源数据和内部数据组成，涵盖了广泛的任务类型，包括多模态和仅文本的指令数据、图像描述、定位和分类。数据集组成的详细信息见表1。我们执行了基于规则的去重，并将所有注释标准化为统一格式。在本阶段，为了增强模型在多模态表示学习中的能力，我们构建了一个包含M个样本的多模态检索数据集。该数据集包括以下三类数据：单模态：包括文本对文本 $\mathrm{T} \mathrm{T}$ 和图像对图像 $(\mathrm{I} \mathrm{I})$ 的配对。跨模态：查询或候选为单一模态，查询与候选对跨越不同模态，例如文本到图像检索 $(\mathrm{T} \to \mathrm{I})$ 或文本到视觉文档检索 $\mathrm{(T \to VD)}$。

<table><tr><td>Task</td><td>#Samples</td><td>Datasets</td></tr><tr><td>Multimodal Instruction Data</td><td>12.8M</td><td>Infinity-MM [21], Bunny-v1.1 [25], VLFeedback [48], RLHF-V [93], RLAIF-V [94], DT-VQA [101], LLaVA Visual Instruct 150K [56], Monkey [50], LVIS-Instruct4V [83], LRV-Instruction [54]</td></tr><tr><td>Pure Text Instruction Data</td><td>8.8M</td><td>Infinity Instruct [45], ShareGPT-Chinese-English-90k [77], firefly-train-1.1M [91], COIG-CQIA [2]</td></tr><tr><td>Captioning</td><td>1.7M</td><td>ShareGPT4V [9], In-house RefCOCO [36], RefCOCO+ [36], RefCOCOg [66], Objects365 v2 [76],</td></tr><tr><td>Grounding</td><td>5.7M</td><td>Visual Genome [38], gRefCOCO [53], Open Images V6 [39], V3Det [82], In-house</td></tr><tr><td>Classification</td><td>2.8M</td><td>In-house</td></tr></table>

• 融合模态：查询和/或候选项同时包含图像和文本。例如，在MegaPairs数据集中，查询是这种混合的图像-文本查询（图像-文本到图像，$\mathrm { I T \to I } \setminus$）。训练数据采样自MegaPairs [106]、Colpali训练集 [18]、VisRAG [92]、Docmatix [43]、BAAI-MTP [4]、ImageNet-1K [13]、BLIP自启动图像-文本对 [47]、MMEB-train [35]和mmE5-合成 [8]。第二阶段训练数据的详细组成见表2。所有这些数据集均来源于MMEB-train [35]。

<table><tr><td>Class</td><td>Task</td><td>#Samples</td><td>Datasets</td></tr><tr><td rowspan="2">Single-Modal</td><td>T→T (1)</td><td>1M</td><td>BAAI-MTP [4]</td></tr><tr><td>I→I (2)</td><td>1.3M</td><td>ImageNet-1K [13], NIGHTS* [19]</td></tr><tr><td rowspan="3">Cross-Modal</td><td>T→I (5)</td><td>5.3M</td><td>VisualNews* [55], MSCOCO* [52], mmE5-synthetic [8], VisDial* [12], BLIP Bootstrapped Image-Text Pairs [47]</td></tr><tr><td>T→VD (3)</td><td>1.6M</td><td>Docmatix [43], Colpali train set [18], VisRAG [92]</td></tr><tr><td>I→T (7)</td><td>0.5M</td><td>ImageNet-1K* [13], HatefulMemes* [37], VOC2007* [17], SUN397* [88], VisualNews* [55], MSCOCO* [52], mmE5-synthetic [8]</td></tr><tr><td rowspan="4">Fused-Modal</td><td>IT→I (5)</td><td>5.3M</td><td>MegaPairs [106], mmE5-synthetic [8], CIRR* [60], N24News* [85], MSCOCO* [52]</td></tr><tr><td>IT→T (8)</td><td>1.6M</td><td>Docmatix [43], mmE5-synthetic [8], OK-VQA* [67], A-OKVQA* [75], DocVQA* [70], InfographicVQA* [69], ChartQA* [68], Visual7W* [109]</td></tr><tr><td>T→IT (2)</td><td>3.2K</td><td>WebQA* [7], mmE5-synthetic [8]</td></tr><tr><td>IT→IT (1)</td><td>3.1K</td><td>mmE5-synthetic [8]</td></tr></table>

Seanv 包含 150 万个高质量的多任务样本。这些数据旨在用于基于图像的视觉任务和视觉文档检索任务。对于基于图像的视觉任务，我们使用 MMEB-train 作为训练集，而对于视觉文档检索任务，我们采用 Colpali 训练集和 VisRAG 作为数据。

# 5 实验

# 5.1 评估设置与基准。

我们首先评估Magic-MM-Embedding在自然图像检索和视觉文档检索任务上的性能。对于自然图像检索，我们使用MMEB [35]，这是一个包含36个子数据集和4个元任务的综合基准，来评估并报告精确率 $@ 1$ 。对于视觉文档检索（VisDoc），我们遵循VLM2Vec-V2 [71] 的设置，并使用ViDoRe v1 (VDRv1) [18]、ViDoRe v2 (VDRv2) [65]、VisRAG (VR) [92] 和ViDoSeek [84]+MMLongBench-Doc (OOD) [64] 来评估并报告NDCG $\textcircled { a } 5$ 。为了评估Magic-MM-Embedding在跨模态检索上的性能，我们遵循UniME-V2的设置 [23]，进一步在Flickr30K [73]、MSCOCO [52]、ShareGPT4V [9]、Urban1K [97] 和SugarCrepe [28]上进行评估并报告精确率 $@ 1$ 。表格：MMEB基准 [35] 的结果。分数以每个元任务平均。每个区块中的最佳性能以粗体显示。“E”指仅使用嵌入器的单阶段检索性能；“E + R”指从重排序器进行排名的结果。

<table><tr><td rowspan="2">Model</td><td rowspan="2">Backbone (Model Size)</td><td colspan="4">Per Meta-Task Score</td><td colspan="3">Average Score</td></tr><tr><td>Classification</td><td>VQA</td><td>Retrieval</td><td>Grounding</td><td>IND</td><td>OOD</td><td>Overall</td></tr><tr><td># of datasets →</td><td></td><td>10</td><td>10</td><td>12</td><td>4</td><td>20</td><td>16</td><td>36</td></tr><tr><td colspan="9">Zero-shot Results</td></tr><tr><td>CLIP [74]</td><td>-(0.4B)</td><td>42.8</td><td>9.1</td><td>53.0</td><td>51.8</td><td>37.1</td><td>38.7</td><td>37.8</td></tr><tr><td>SigLIP [96]</td><td>-(0.9B)</td><td>40.3</td><td>8.4</td><td>31.6</td><td>59.5</td><td>32.3</td><td>38.0</td><td>34.8</td></tr><tr><td>EVA-CLIP [79]</td><td>-(8.1B)</td><td>56.0</td><td>10.4</td><td>49.2</td><td>58.9</td><td>38.1</td><td>45.6</td><td>43.7</td></tr><tr><td>MagicLens [99]</td><td>-(0.4B)</td><td>38.8</td><td>8.3</td><td>35.4</td><td>26.0</td><td>31.0</td><td>23.7</td><td>27.8</td></tr><tr><td>E5-V [34] E5-V [34]</td><td>Phi3.5-V (4.2B) LLaVA-1.6 (8.4B)</td><td>39.1 39.7</td><td>9.6 10.8</td><td>38.0 39.4</td><td>57.6 60.2</td><td>33.1 34.2</td><td>31.9 33.4</td><td>36.1 37.5</td></tr><tr><td colspan="9"></td></tr><tr><td></td><td></td><td>Trained with MMEB</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">VLM2Vec-V1 [35]</td></tr><tr><td>UniME [22]</td><td>Qwen2VL (2.2B) Phi3.5-V (4.2B)</td><td>59.0 54.8</td><td>49.4 55.9</td><td>65.4 64.5</td><td>73.4 81.8</td><td>66.0 68.2</td><td>52.6 52.7</td><td>59.3 64.2</td></tr><tr><td>LLaVE [41]</td><td>Aquila-VL (2.0B)</td><td>62.1</td><td>60.2</td><td>65.2</td><td>84.9</td><td>69.4</td><td>59.8</td><td>65.2</td></tr><tr><td>UniME-V2 (E) [23]</td><td>Qwen2VL (2.2B)</td><td>62.1</td><td>56.3</td><td>68.0</td><td>72.7</td><td>67.4</td><td>58.9</td><td>63.6</td></tr><tr><td>UniME-V2 (E+) [23]</td><td>Qwen2VL (2.2B)</td><td>64.1</td><td>64.3</td><td>71.6</td><td>70.6</td><td>69.8</td><td>64.3</td><td>67.4</td></tr><tr><td>Magic-MM-Embedding (E)</td><td>InternVL3-VTC (1.9B)</td><td>60.9</td><td>63.3</td><td>72.2</td><td>84.6</td><td>74.7</td><td>59.5</td><td>68.0</td></tr><tr><td>Magic-MM-Embedding (E+R)</td><td>InternVL3-VTC (1.9B)</td><td>61.3</td><td>67.2</td><td>73.5</td><td>89.8</td><td>75.2</td><td>63.9</td><td>70.2</td></tr><tr><td>VLM2Vec-V1 [35]</td><td>Qwen2VL (8.3B</td><td>62.6</td><td>57.8</td><td>69.9</td><td>81.7</td><td>65.2</td><td>556.3</td><td>65.8</td></tr><tr><td>UniME [22]</td><td>LLaVA-OV (8.0B)</td><td>66.8</td><td>66.6</td><td>70.5</td><td>90.9</td><td>74.6</td><td>65.8</td><td>70.7</td></tr><tr><td>LLaVE [41]</td><td>LLaVA-OV (8.0B)</td><td>65.7</td><td>65.4</td><td>70.9</td><td>91.9</td><td>75.0</td><td>64.4</td><td>70.3</td></tr><tr><td>QQMM [89]</td><td>LLaVA-OV (8.0B)</td><td>66.8</td><td>66.8</td><td>70.5</td><td>90.4</td><td>74.7</td><td>65.6</td><td>70.7</td></tr><tr><td>UniME-V2 [23]</td><td>LLaVA-OV (8.0B)</td><td>65.3</td><td>67.6</td><td>72.9</td><td>90.2</td><td>74.8</td><td>66.7</td><td>71.2</td></tr><tr><td>UniME-V2 (E) [23]</td><td>Qwen2VL (8.3B)</td><td>64.0</td><td>60.1</td><td>73.1</td><td>82.8</td><td>72.0</td><td>63.0</td><td>68.0</td></tr><tr><td>UniME-V2 (E+R) [23]</td><td>Qwen2VL (8.3B)</td><td>63.8</td><td>66.3</td><td>73.5</td><td>75.0</td><td>71.7</td><td>65.6</td><td>69.0</td></tr><tr><td>Magic-MM-Embedding (E)</td><td>InternVL3-VTC (8.1B)</td><td>64.8</td><td>68.1</td><td>75.0</td><td>88.7</td><td>78.3</td><td>63.6</td><td>71.8</td></tr><tr><td>Magic-MM-Embedding (E+R)</td><td>InternVL3-VTC (8.1B)</td><td>64.3</td><td>70.9</td><td>75.7</td><td>90.4</td><td>78.4</td><td>65.9</td><td>72.8</td></tr></table>

# 5.2 实现细节

模型架构。我们使用 PyTorch 和 ms-swift [104] 库实现我们的框架。我们采用 InternVL3 [107] 作为我们的主干 MLLM，并将提议的视觉词元压缩变体命名为 InternVL3-VTC。对于我们的无参数空间压缩设计，我们利用双线性插值方法通过空间维度下采样视觉特征图，从而保留视觉词元。 图像切片策略。对于图像切片策略，我们遵循 InternVL3 [107] 中使用的方法。然而，为了确保计算效率，我们在训练和推理过程中减少最大图像切片数量（MAX_NUM）。具体而言，在训练阶段，MAX_NUM 设定为 。在下游嵌入器和重排序模型的推理阶段，我们采用数据依赖的策略：对于包含视觉文档图像的数据，MAX_NUM 设置为 4，而对于所有其他自然图像数据，MAX_NUM 统一设置为 1。

嵌入模型实现。在阶段 1，我们在 $4 8 \times$ NVIDIA A800 (80GB) GPU 上训练完整模型参数，学习率为 $1 \times 1 0 ^ { - 5 }$，全局批量大小为 48。我们将梯度累积步数设置为 8，并应用数据打包训练模型或 0,0 步以恢复生成能力。在阶段 n 中，我们使用低秩适配（LoRA）在同样的 $4 8 \times$ NVIDIA A800 (80GB) GPU 上进行对比预训练和任务感知微调。两个阶段均使用统一的最大学习率 $2 \times 1 0 ^ { - 4 }$，LoRA 秩为 16；详细超参数见表 14。为了在阶段 3 进行困难负例判断，我们采用 Qwen3-VL-7B [49] 作为鉴别器，并为每个训练实例插入 12 个困难负样本。重新排序器实现。重新排序器从阶段检查点初始化。我们使用相同的训练数据在 $2 4 \times$ NVIDIA A800 (80GB) GPU 上训练，学习率为 $4 \times 1 0 ^ { - 5 }$，每个设备的批量大小分别设置为 16 和 12（用于 8B 和 2B 模型）。模型训练 2 个 epoch。损失权重对于点式和列表式目标都设置为 1。在推理过程中，重新排序器用于获得两阶段检索结果，首先使用嵌入模型检索候选，然后通过重新排序器对嵌入模型的前 5 个结果进行点式重新排序。表：VisDoc [1] 的结果。每个区块中的最佳表现用粗体显示，$E ^ { \ast }$ 指仅使用嵌入模型的单阶段检索表现；$E { + } R ^ { \prime }$ 指嵌入模型检索候选集，然后由重新排序器进行最终排序。

<table><tr><td rowspan="2">Model</td><td rowspan="2">Backbone (Model Size)</td><td colspan="5">VisDoc</td></tr><tr><td>VDRv1</td><td>VDRv2</td><td>VR</td><td>OOD</td><td>Overall</td></tr><tr><td># of Datasets →</td><td></td><td>10</td><td>4</td><td>6</td><td>4</td><td>24</td></tr><tr><td>GME [102]</td><td>Qwen2VL (2.2B)</td><td>86.1</td><td>54.0</td><td>82.5</td><td>43.1</td><td>72.7</td></tr><tr><td>ColPali [18]</td><td>Paligemma (2.9B)</td><td>83.6</td><td>52.0</td><td>81.1</td><td>43.1</td><td>71.0</td></tr><tr><td>Ops-MM-embedding-v1 [72]</td><td>Qwen2VL (8.3B)</td><td>80.1</td><td>59.6</td><td>79.3</td><td>43.3</td><td>70.3</td></tr><tr><td>VLM2Vec-V2 [71]</td><td>Qwen2VL (2.2B)</td><td>75.5</td><td>44.9</td><td>79.4</td><td>39.4</td><td>65.4</td></tr><tr><td>Magic-MM-Embedding (E)</td><td>InternVL3-VTC (1.9B)</td><td>83.4</td><td>53.3</td><td>85.6</td><td>42.2</td><td>72.1</td></tr><tr><td>Magic-MM-Embedding (E+R)</td><td>InternVL3-VTC (1.9B)</td><td>84.4</td><td>56.1</td><td>87.4</td><td>41.8</td><td>73.3</td></tr><tr><td>Ops-MM-embedding-v1 [72]</td><td>Qwen2VL (8.3B)</td><td>80.1</td><td>59.6</td><td>79.3</td><td>43.3</td><td>70.3</td></tr><tr><td>GME [102]</td><td>Qwen2VL (8.3B)</td><td>89.4</td><td>55.6</td><td>85.0</td><td>44.4</td><td>75.2</td></tr><tr><td>LamRA-Qwen2 [59]</td><td>Qwen2VL (8.3B)</td><td>22.0</td><td>11.5</td><td>37.4</td><td>21.0</td><td>23.9</td></tr><tr><td>LamRA-Qwen2.5 [59]</td><td>Qwen2.5VL (8.3B)</td><td>56.3</td><td>33.3</td><td>58.2</td><td>40.1</td><td>50.2</td></tr><tr><td>VLM2Vec-V2 [71]</td><td>Qwen2VL (8.3B)</td><td>78.8</td><td>52.6</td><td>82.7</td><td>42.1</td><td>69.3</td></tr><tr><td>Magic-MM-Embedding (E)</td><td>InternVL3-VTC (8.1B)</td><td>86.1</td><td>59.9</td><td>87.6</td><td>43.4</td><td>75.0</td></tr><tr><td>Magic-MM-Embedding (E+R)</td><td>InternVL3-VTC (8.1B)</td><td>86.9</td><td>60.4</td><td>89.2</td><td>43.1</td><td>75.8</td></tr></table>

# 5.3 实验结果

多模态检索在MMEB上。在表3中，我们展示了与代表性基线模型的性能比较。结果显示，传统的双塔模型如CLIP在MLLM基础方法上明显落后，反映出其架构的固有限制。在MLLM方法中，我们提出的嵌入器建立了新的最先进水平，超越了像UniME-V2 [23] 和 QQMM [89]这样的强基线，证明了我们的渐进式训练策略成功克服了词元压缩可能导致的信息损失。关于两阶段检索范式，我们的方法和UniME-V2 [23]均表明，加入重排器进一步提升了准确性。然而，直接比较揭示了我们的一致优势：在独立嵌入器设置和完整的“嵌入器 $^ +$ 重排器”设置中，我们的模型均优于UniME-V2。这证实了我们的框架提供了更强大的基础检索器和更有效的整体管道，超过了先前表现最佳的方法。 多模态检索在VisD上。除了通用多模态检索外，我们在具有挑战性的领域——视觉文档检索（VisDoc）上评估了我们的模型，这是一个理论上要求高分辨率输入以保留文本细节的细粒度任务。在表中，我们报告了分解结果。令人意外的是，尽管视觉特征压缩了 $75\%$，Uroken-ient仍然达到了最先进的结果，这挑战了高冗余在细粒度检索中是严格必要的假设。此外，我们观察到GME [102]作为一个强大基线，超越了我们的独立嵌入器；我们将此归因于GME使用了大量专有的任务感知数据集，而我们使用的是更小的公开训练源。然而，我们的“嵌入器 $^ +$ 重排器”管道成功超越了GME，建立了新的最先进水平。这表明仅使用公共数据和显著更少的视觉词元也能取得成功。

文本-图像交叉模态检索。基于之前的研究，我们进一步评估了文本-图像的交叉模态检索能力。我们的嵌入方法在多个数据集上始终交付新的最先进结果。在8B规模上，即便在强基线UniME-V接近饱和水平的基准上，我们的方法也取得了显著提升，达到$9 5 \%$。具体而言，我们在ShareGPT4V（文本到图像）上的准确率$@ 1$从$9 5 . 1 \%$提升到$9 8 . 5 \%$，在Urban1K（图像到文本）上的准确率从$9 6 . 7 \%$提升到$9 8 . 7 \%$。在2B规模上，我们的模型在具有挑战性的SugarCrepe基准上以巨大的优势超越了UniME-V2——在其三个子设置中得分分别为$9 1 . 6 \%$、$8 2 . 6 \%$和$9 4 . 2 \%$，而UniME-V2的得分则为$7 0 . 9 \%$、$5 1 . 2 \%$和$7 0 . 2 \%$。至关重要的是，我们的方法仅使用64个视觉词元，远低于其他标准方法。这种优势的实现与我们的渐进式训练流程相结合，显著提高了推理效率，同时在跨模态对齐方面取得了更好的效果。推理成本比较。我们比较了所提的Magic-MM-Embedding与目前流行的基于MLLM的嵌入模型的推理效率，如表6所示。对于每个MLLM主干，我们选取了一个嵌入模型进行比较。我们随机从MMEB和VisDoc训练集中抽取了5000个查询和候选项。我们测量了MMEB和VisDoc上查询和候选项的平均推理延迟以及平均视觉词元数量。为了确保公平性，我们将视觉文档图像的分辨率调整为$8 9 6 \times 8 9 6$，自然图像的分辨率调整为$4 4 8 \times 4 4 8$。所有结果均使用NVIDIA L20（48GB）GPU，在批量大小为1且BF16精度下获得。测试过程中未使用加速技术。

Table 5: Cross-modal retrieval results on Flickr30K [73], MSCOCO [52], ShareGPT4V [9], Urban1K [97] and SugarCrepe [28].   

<table><tr><td rowspan="3">Models</td><td rowspan="3">Backbone (Model Size) Flickr30K</td><td colspan="3">Short Caption</td><td colspan="4">Long Caption</td><td colspan="3">Compositional</td></tr><tr><td colspan="2"></td><td colspan="2">MSCOCO</td><td colspan="2">ShareGPT4V</td><td colspan="2">Urban1K</td><td colspan="2">SugarCrepe</td></tr><tr><td>T → II → T T → II → TT</td><td></td><td></td><td></td><td>→ II → T T </td><td></td><td>→ II</td><td></td><td>| → T Replace Swap Add</td><td></td></tr><tr><td>OpenCLIP [74]</td><td>- (0.4B)</td><td>67.3</td><td>87.2</td><td>37.0</td><td>58.1</td><td>81.8</td><td>84.0</td><td>47.0</td><td>47.0</td><td>79.5</td><td>62.7 74.9</td></tr><tr><td>CLIP [10]</td><td>- (2.5B)</td><td>79.5</td><td>92.9</td><td>51.3</td><td>67.3</td><td>90.1</td><td>93.6</td><td>77.8</td><td>80.7</td><td>86.5</td><td>68.9 88.4</td></tr><tr><td>EVA-CLIP [79]</td><td>-(8.1B)</td><td>80.3</td><td>94.5</td><td>52.0</td><td>70.1</td><td>93.1</td><td>91.2</td><td>80.4</td><td>77.8</td><td>85.9</td><td>70.3 86.7</td></tr><tr><td>E5-V [34]</td><td>Phi3.5-V (4.2B)</td><td>72.2</td><td>79.6</td><td>44.7</td><td>53.4</td><td>86.0</td><td>88.5</td><td>83.8</td><td>83.6</td><td>88.2</td><td>66.675.3</td></tr><tr><td>VLM2Vec [35]</td><td>Qwen2-VL (2.2B)</td><td>69.3</td><td>89.6</td><td>40.0</td><td>62.5</td><td>78.1</td><td>88.2</td><td>78.7</td><td>83.9</td><td>67.2</td><td>46.5 66.4</td></tr><tr><td>UniME [22]</td><td>Qwen2-VL (2.2B)</td><td>74.9</td><td>90.6</td><td>44.0</td><td>63.5</td><td>83.6</td><td>88.6</td><td>83.3</td><td>83.2</td><td>65.6</td><td>45.2 65.7</td></tr><tr><td>UniME-V2 [23]</td><td>Qwen2-VL (2.2B)</td><td>79.8</td><td>89.9</td><td>53.7</td><td>65.1</td><td>91.6</td><td>94.2</td><td>95.6</td><td>92.2</td><td>70.9</td><td>51.2 70.2</td></tr><tr><td>Magic-MM-Embedding_InternVL3-VTC (1.9B)</td><td></td><td>84.4</td><td>93.0</td><td>61.4</td><td>75.8</td><td>97.2</td><td>97.3</td><td>98.4</td><td>97.8</td><td>91.6</td><td>82.6 94.2</td></tr><tr><td>E5-V [34]</td><td>LLaVA-1.6 (8.4B)</td><td>77.3</td><td>85.7</td><td>49.1</td><td>57.6</td><td>85.1</td><td>82.1</td><td>88.9</td><td>83.2</td><td>86.3</td><td>68.7 66.9</td></tr><tr><td>VLM2Vec [35]</td><td>Qwen2-VL (8.3B)</td><td>80.0</td><td>94.2</td><td>49.2</td><td>68.5</td><td>78.5</td><td>90.4</td><td>94.0</td><td>94.2</td><td>70.0</td><td>51.7 72.2</td></tr><tr><td>UniME [22]</td><td>Qwen2-VL (8.3B)</td><td>80.8</td><td>92.7</td><td>50.9</td><td>69.8</td><td>86.5</td><td>93.8</td><td>95.3</td><td>94.0</td><td>68.8</td><td>53.0 69.8</td></tr><tr><td>UniME [22]</td><td>LLaVA-OV (8.0B)</td><td>83.3</td><td>94.4</td><td>54.8</td><td>74.0</td><td>93.9</td><td>89.3</td><td>94.3</td><td>95.5</td><td>80.5</td><td>65.5 82.2</td></tr><tr><td>UniME-V2 [23]</td><td>Qwen2-VL (8.3B)</td><td>84.6</td><td>93.5</td><td>57.3</td><td>70.3</td><td>94.3</td><td>95.2</td><td>97.2</td><td>96.3</td><td>77.8</td><td>62.2 79.0</td></tr><tr><td>UniME-V2 [23]</td><td>LLaVA-OV (8.0B)</td><td>85.5</td><td>93.7</td><td>60.9</td><td>74.1</td><td>95.1</td><td>94.1</td><td>96.3</td><td>96.7</td><td>88.6</td><td>73.7 90.5</td></tr><tr><td>Magic-MM-Embedding InternVL3-VTC (8.1B)</td><td></td><td>82.9</td><td>93.1</td><td>63.2</td><td>79.3</td><td>98.5</td><td>98.3</td><td>98.5</td><td>98.7</td><td>92.6</td><td>86.9 95.1</td></tr></table>

在具有相似参数规模的模型下，Magic-MM-Embedding 显示出显著降低的推理延迟，例如，相比基于 Aquila-VL 的 LLaVE-2B，Magic-MM-Embedding-2B 将 MMEB 查询的推理延迟从 162.8 毫秒降至 $29.9 ~ \mathrm{ms}$，并将 VisDoc 候选的推理延迟从 $233.6 ~ \mathrm{ms}$ 降至 $57.3 ~ \mathrm{ms}$。我们观察到，在 VisDoc 查询的推理延迟中，Magic-MM-Embedding(2B/8B) 的延迟略高于 GME(2B/8B)。这是因为，在 $\mathrm{T} \to \mathrm{VD}$ 任务中，Inte 的系统提示长度相比于 Qwe-L 有所增加，考虑到参数规模的变化，新模型的延迟有所不同。但这种延迟差异可以通过预缓存技术来缓解 [40]。我们还对原生的 InternVL3 架构进行了比较。Magic-MM-Embedding 唯一的不同之处在于引入了一个无参数的视觉令牌压缩模块。我们发现，与原生架构相比，将视觉令牌数量减少 $75\%$ 显著提高了推理效率。

Table 6: Inference efficiency comparison. $\# V T _ { q }$ and $\# V T _ { c }$ refer to the average number of visual tokens in queries and candidates containing images, respectively. $l _ { q }$ and $l _ { c }$ mean the average latency (millisecond) of query inference and candidate inference, respectively. The best performance in each block is in bold.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Backbone (Model Size)</td><td colspan="4">MMEB</td><td colspan="4">VisDoc</td></tr><tr><td>#V Tq</td><td>lq</td><td>#V Tc</td><td>lc</td><td>#V Tq</td><td>lq</td><td>#V Tc</td><td>lc</td></tr><tr><td>VLM2Vec [35]</td><td>Phi3.5-V (4.2B)</td><td>757.0</td><td>99.4</td><td>757.0</td><td>85.9</td><td>0</td><td>34.0</td><td>757.0</td><td>128.6</td></tr><tr><td>GME [102]</td><td>Qwen2VL (2.2B)</td><td>362.8</td><td>46.8</td><td>256.0</td><td>34.5</td><td>0</td><td>19.3</td><td>1024.0</td><td>153.8</td></tr><tr><td>LLaVE [41]</td><td>Aquila-VL (2.0B)</td><td>3699.0</td><td>162.8</td><td>3699.0</td><td>143.0</td><td>0</td><td>18.5</td><td>3699.0</td><td>233.6</td></tr><tr><td>InternVL3 [107]</td><td>InternVL3 (1.9B)</td><td>398.4</td><td>37.1</td><td>256.0</td><td>29.2</td><td>0</td><td>19.8</td><td>1280.0</td><td>103.6</td></tr><tr><td>Magic-MM-Embedding _ InternVL3-VTC (1.9B)</td><td></td><td>99.6</td><td>29.9</td><td>64.0</td><td>26.1</td><td>0</td><td>19.7</td><td>320.0</td><td>57.3</td></tr><tr><td>VLM2Vec [35]</td><td>LLaVA-1.6 (8.4B)</td><td>2928.0</td><td>332.3</td><td>2928.0</td><td>278.9</td><td>0</td><td>32.4</td><td>2928.0</td><td>458.1</td></tr><tr><td>GME [102]</td><td>Qwen2VL (8.3B)</td><td>362.8</td><td>82.2</td><td>256.0</td><td>56.7</td><td>0</td><td>26.6</td><td>1024.0</td><td>268.2</td></tr><tr><td>LamRA [59]</td><td>Qwen2.5VL (8.3B)</td><td>362.8</td><td>83.4</td><td>256.0</td><td>61.6</td><td>0</td><td>28.9</td><td>1024.0</td><td>251.7</td></tr><tr><td>UniME-V2 [23]</td><td>LLaVA-OV (8.0B)</td><td>7371.0</td><td>906.9</td><td>7371.0</td><td>788.1</td><td>0</td><td>32.1</td><td>7371.0</td><td>1341.1</td></tr><tr><td>InternVL3 [107]</td><td>InternVL3 (8.1B)</td><td>398.4</td><td>76.7</td><td>256.0</td><td>55.9</td><td>0</td><td>33.8</td><td>1280.0</td><td>260.4</td></tr><tr><td>Magic-MM-Embedding</td><td>InternVL3-VTC (8.1B)</td><td>99.6</td><td>50.9</td><td>64.0</td><td>40.6</td><td>0</td><td>33.8</td><td>320.0</td><td>94.8</td></tr></table>

# 5.4 消融研究

对渐进训练流程及重排序器的消融研究。我们分析了渐进粗到细训练流程中每个组件的贡献。结果如表7所示。所有实验均使用 Magi-MM-Embedding-2B 进行。我们在阶段中使用的模式为对比预热，作为基线，其中学习仅使用批内负样本。我们发现引入全局困难负样本挖掘策略在 MMEB 和 VisDoc 上分别带来了 2.5 和 2.3 的绝对增益。这表明引入困难负样本显著增强了模型的区分能力。基于此，我们进一步使用经过 MLLM 判断过滤的高质量多任务数据对模型进行了微调。这分别在 MMEB 和 VisDoc 上带来了 2.6 和 1.4 的额外提升，显示出使用 MLLM 作为评判者的任务感知微调有助于模型有效适应复杂多样的下游任务。我们还研究了协同重排序器对模型性能的影响。如表 7 所示，添加重排序器分别在 MMEB 和 VisDoc 上带来了 2.2 和 1.2 的进一步改善。这证实了在推理流程中引入重排序可以进一步提升检索性能。表中关于渐进训练流程和重排序器的消融研究。我们报告了 MMEB 和 ViD 的平均得分。每一行代表一个累积版组件。“Warm-Up” 表示在阶段中仅使用批内负样本的对比预热；“Global-H” 指在使用全局困难负样本挖掘的阶段进行预训练；“MLLM-Judge-FT” 表示使用 MLLM 作为评判者进行微调；“Reranker” 代表使用协同重排序器的两阶段检索。

<table><tr><td>Stage 2 (Warm-Up)</td><td>Stage 2 (Global-HNM)</td><td>Stage 3 (MLLM-Judge-FT)</td><td>Inference (Reranker)</td><td>MMEB</td><td>VisDoc</td></tr><tr><td></td><td></td><td></td><td>X</td><td>62.9</td><td>68.4</td></tr><tr><td>:</td><td>×</td><td>\x$</td><td>×</td><td>65.4</td><td>70.7</td></tr><tr><td>:</td><td>;</td><td>:</td><td>X</td><td>68.0</td><td>72.1</td></tr><tr><td></td><td></td><td></td><td>;</td><td>70.2</td><td>73.3</td></tr></table>

针对困难负样本数量和类型的消融研究。我们调查了模型在第三阶段训练中对困难负样本数量 $n$ 的敏感性，如表 8 所示。所有实验在 Magic-MM-Embedding-2B 上进行，其中 $n$ 变化范围为 0 到某个值。结果表明，与仅使用标准的批内负样本采样策略相比，引入任何数量的基于 MLLM 的困难负样本始终能显著提升性能。随着 $n$ 的增加，模型性能先是提升，随后略微下降。例如，在 MMEB 上，性能在 $n = 16$ 时达到峰值，进一步增加 $n$ 会导致性能下降。我们进一步将我们的方法与基于规则的困难负样本采样策略进行比较，如表中所述。该策略从检索到的前 $K$ 个候选样本中移除真实标注样本，并将剩余样本视为可用负样本。理论上，使用基于 MLLM 的困难负样本训练的模型相比于使用相同数量的基于规则的困难负样本训练的模型，性能始终显著更优。表 8：困难负样本数量和类型的影响。“Avg.”表示 MMEB 分数和 VisDoc 分数的平均值。

<table><tr><td rowspan="2">#HN (n)</td><td colspan="3">MLLM-based HN</td><td colspan="3">Rule-based HN</td></tr><tr><td>MMEB</td><td>VisDoc</td><td>Avg.</td><td>MMEB</td><td>VisDoc</td><td>Avg.</td></tr><tr><td>0</td><td>65.5</td><td>70.6</td><td>68.1</td><td>65.5</td><td>70.6</td><td>68.1</td></tr><tr><td>4</td><td>67.4</td><td>71.7</td><td>69.5</td><td>65.7</td><td>69.5</td><td>67.6</td></tr><tr><td>8</td><td>67.8</td><td>71.9</td><td>69.9</td><td>66.5</td><td>70.6</td><td>68.6</td></tr><tr><td>12</td><td>68.0</td><td>72.1</td><td>70.0</td><td>67.0</td><td>70.1</td><td>68.5</td></tr><tr><td>16</td><td>67.9</td><td>72.1</td><td>70.0</td><td>65.9</td><td>70.7</td><td>68.3</td></tr><tr><td>20</td><td>67.6</td><td>71.9</td><td>69.8</td><td>67.1</td><td>70.5</td><td>68.8</td></tr></table>

关于视觉令牌压缩对训练效率的消融研究。我们使用 InternVL3-VTC-2B 和原始的 InternL3-2B 作为主干网络，调查有无令牌压缩模块对训练效率的影响。Bot 模型在 16M 数据集上通过对比学习进行训练，训练期间的热身阶段采用此方法。在训练过程中，批次大小增加到 GPU 内存允许的最大值，以确保模型充分训练。所有其他训练超参数在两个模型之间保持一致。实验结果显示，尽管在模型性能上几乎没有下降，所提议的视觉令牌压缩方法显著提高了训练效率。例如，对于一个 2B 规模的模型，训练 2 个时期的时间从大约 53 小时减少到 23 小时。此外，视觉令牌压缩显著减少了 GPU 内存消耗，使模型能够处理批内负样本。

Table 9: Ablation of visual token compression for training efficiency.   

<table><tr><td>Backbone</td><td>Training Duration</td><td>MMEB</td><td>VisDoc</td><td>Global Batch Size</td></tr><tr><td>InternVL3 (vanilla)</td><td>52h 43m 35s</td><td>62.9</td><td>68.4</td><td>6144</td></tr><tr><td>InternVL3-VTC (ours)</td><td>22h 57m 6s</td><td>63.7</td><td>68.5</td><td>3456</td></tr></table>

LoRA秩的消融研究。表10展示了LoRA秩的消融结果。我们在第二阶段对比学习的预热阶段使用Magic-MM-Embedding-2B进行了实验。我们将LoRA秩设置为8、16和32进行实验。我们发现，当LoRA秩设置为16时，MMEB和VisDoc上的平均指标最佳。进一步增加LoRA秩会导致整体性能下降。因此，在嵌入器的所有训练中，LoRA秩均设置为16。

Table 10: Ablation analysis of LoRA rank. "Avg." means the average of the MMEB score and the VisDoc score   

<table><tr><td>LoRA Rank</td><td>MMEB</td><td>VisDoc</td><td>Avg.</td></tr><tr><td>8</td><td>62.9</td><td>68.0</td><td>65.5</td></tr><tr><td>16</td><td>62.9</td><td>68.4</td><td>65.7</td></tr><tr><td>32</td><td>62.6</td><td>67.6</td><td>65.1</td></tr></table>

# 6 结论

在本研究中，我们识别出了当前基于多语言大模型（MLLM）的通用嵌入模型中一个关键的计算瓶颈：处理冗余视觉词元的高昂成本。为了解决这个问题，我们提出了一种简单但强大的基线，仅使用基线视觉词元的 $25\%$，显著降低了推理延迟和内存占用，同时达到了全新的最先进性能。关键是，我们展示了性能通过重构方式进行渐进式改进—从基本生成到任务驱动的细化，由一个MLLM作为评判者引导。该策略有效地将重要语义信息提炼到压缩表示中。此外，通过为这一高效嵌入器配备一个协同训练的重排序模型，我们建立了一个综合性的检索系统。大量实验表明，我们的系统在竞争对手面前表现出色，尽管后者使用了更大规模的专有数据集，证明高效性和卓越有效性确实可以同时实现。

# References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.   
[2] Yuelin Bai, Xinrun Du, Yiming Liang, Yonggang Jin, Ziqiang Liu, Junting Zhou, Tianyu Zheng, Xincheng Zhang, Nuo Ma, Zekun Wang, et al. Coig-cqia: Quality is all you need or chinee instruction fne-tuning, 2024.   
[3] Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. Advances in neural information processing systems, 32, 2019.   
[4] BeijAcademyofArtificial IntellenceBaai-tdataset.https:/ata.bi.acc/atadetail/BAAI-MTP. Accessed: 2026-01-24.   
[5] Anjia Cao, Xing Wei, and Zhiheng Ma. Flame: Frozen large language models enable data-effcient language-image pre-training. arXiv:2411.11927, 2024.

[6] Junbum Cha, Wooyoung Kang, Jonghwan Mun, and Byungseok Roh. Honeybee: Locality-enhanced projector for multimodal llm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1381713827, 2024.

[7] Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1649516504, 2022.

[8] Haonan Chen, Liang Wang, Nan Yang, Yutao Zhu, Ziliang Zhao, Furu Wei, and Zhicheng Dou. mme5: Improving multimodal multilingual embeddings via high-quality synthetic data. arXiv preprint arXiv:2502.08468, 2025.

[9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370387. Springer, 2024.

[10] Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. arXiv:2212.07143, 2022.

[1] Xuanming Cui, Jianpeng Cheng, Hong-you Chen, Satya Narayan Shukla, Abhijeet Awasthi, Xichen Pan, Chaitanya Ahuja, Shlok Kumar Mishra, Yonghuan Yang, Jun Xiao, et al. Think then embed: Generative context improves multimodal embedding. arXiv preprint arXiv:2510.05014, 2025.

[12] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José MF Moura, Devi Parikh, and Dhruv Batra. Visual dialog. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 326335, 2017.

[13]Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet:A large-scale hierarchial image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248255. Ieee, 2009.

[14] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 41714186, 2019.

[15] Mohamed Dhouib, Davide Buscaldi, Sonia Vanier, and Aymen Shabou. Pact: Pruning and clustering-based token reduction for faster visual language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1458214592, 2025.

[16] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021.

[17] Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. International journal of computer vision, 111(1):98136, 2015.

[18] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. Colpali: Effcient document retrieval with vision language models. arXiv preprint arXiv:2407.01449, 2024.

[19] Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023.

[20] Ramin Giahi, Kehui Yao, Sriram Kollipara, Kai Zhao, Vahid Mirjalili, Jianpeng Xu, Topojoy Biswas, Evren Korpeoglu, and Kannan Achan. Vl-clip: Enhancing multimodal recommendations via visual grounding and llm-augmented clip embeddings. In Proceedings of the Nineteenth ACM Conference on Recommender Systems, pages 482491, 2025.

[21] Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, et al. Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data. arXiv preprint arXiv:2410.18558, 2024.

[22] Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, and Jiankang Deng. Breaking the modality barrier: Universal embedding learning with multimodal llms. In Proceedings of the 33rd ACM International Conference on Multimedia, MM '25, page 28602869, New York, NY, USA, 2025. Association for Computing Machinery.

[23] Tiancheng Gu, Kaicheng Yang, Kaichen Zhang, Xiang An, Ziyong Feng, Yueyi Zhang, Tom Weidong Cai, Jiankang Deng, and Lidong Bing. Unime-v2: Mllm-as-a-judge for universal multimodal embedding learning. AAAI, 2026.

[24] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffry P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 36083617, 2018.

[25] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024.

[26] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-ofdistribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 83408349, 2021.

[27] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1526215271, 2021.

[28] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in neural information processing systems, 36:3109631116, 2023.

[29] Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-Wei Chang. Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 12065 12075, 2023.

[30] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 67006709, 2019.

[31] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju Hwang. Videorag: Retrieval-augmented generation over video corpus. arXiv preprint arXiv:2501.05874, 2025.

[32] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 49044916. PMLR, 2021.

[33] Weijan Jian, Yajun Zhang, Dawei Liang, Chunyu Xie, Yixiao He, Dawei Leng, and Yuhui Yin. Rzenembed: Towards comprehensive multimodal retrieval. CoRR, abs/2510.27350, 2025.

[34] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. E5-v: Universal embeddings with multimodal large language models. arXiv:2407.12580, 2024.

[35] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. ICLR, 2025.

[36] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787798, 2014.

[37] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj Goswami, Amanpreet Singh, Pratik Ringshia, and Davide Testuggine. The hateful memes challenge: Detecting hate speech in multimodal memes. Advances in neural information processing systems, 33:26112624, 2020.

[38] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, DavidA Shamma, et al.Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):3273, 2017.

[39] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International journal of computer vision, 128(7):19561981, 2020.

[40] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

[41] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Llave: Large language and vision embedding models with hardness-weighted contrastive learning. CoRR, abs/2503.04812, 2025.

[42] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Ume-r1: Exploring reasoning-driven generative multimodal embeddings. arXiv preprint arXiv:2511.00405, 2025.

[43] Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding vision-language models: insights and future directions. arXiv preprint arXiv:2408.12637, 2024.

[44] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

[45] Jijie Li, Li Du, Hanyu Zhao, Bo-wen Zhang, Liangdong Wang, Boyan Gao, Guang Liu, and Yonghua Lin. Infinity instruct: Scaling instruction selection and synthesis to enhance language models. arXiv preprint arXiv:2506.11116, 2025.

[46] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 1973019742. PMLR, 2023.

[47] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 1288812900. PMLR, 2022.

[48] Lei Li, Zhihui Xie, Mukai Li, Shunian Chen, Peiyi Wang, Liang Chen, Yazheng Yang, Benyou Wang, and Lingpeng Kong. Silkie: Preference distillation for large visual language models. arXiv preprint arXiv:2312.10665, 2023.

[49] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

[50] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2676326773, 2024.

[51] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. Mmembed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571, 2024.

[52] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740755. Springer, 2014.

[53] Chang Liu, Henghui Ding, and Xudong Jiang. Gres: Generalized referring expression segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2359223601, 2023.   
[54] FuxioLiu, Kevin Lin, Linje Li, Jianfeg Wang, Yaser Yacob, and Lijuan Wang.Mitigati halluaion in large multi-modal models via robust instruction tuning. arXiv preprint arXiv:2306.14565, 2023.   
[55] Fuxiao Liu, Yinghan Wang, Tianlu Wang, and Vicente Ordonez. Visual news: Benchmark and challenges in news image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 67616771, 2021.   
[56] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2629626306, 2024.   
[57] Jinxiang Liu, Chen Ju, Weidi Xie, and Ya Zhang. Exploiting transformation invariance and equivariance for self-supervised sound localisation. In Proceedings of the 30th ACM International Conference on Multimedia, pages 37423753, 2022.   
[58] Siqi Liu, Weixi Feng, Tsu-Jui Fu, Wenhu Chen, and William Wang. Edis: Entity-driven image search over multimodal web content. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 48774894, 2023.   
[59] Yikun Liu, Pingan Chen, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. Lamra: Large multimodal model as your advanced retrieval assistant. CVPR, 2024.   
[60] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. Image retrieval on real-life images with pre-trained vision-and-language models. In Procedings of the IEEE/CVF international conference on computer vision, pages 21252134, 2021.   
[61] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:25072521, 2022.   
[62] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. Unifying multimodal retrieval via document screenshot embedding. arXiv preprint arXiv:2406.11251, 2024.   
[63] Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Wenhu Chen, and Jimmy Lin. Visa: Retrieval augmented generation with visual source attribution. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3015430169, 2025.   
[64] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. Advances in Neural Information Processing Systems, 37:9596396010, 2024.   
[65] Quentin Macé, António Loison, and Manuel Faysse. Vidore benchmark v2: Raising the bar for visual retrieval. arXiv preprint arXiv:2505.17166, 2025.   
[66] Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1120, 2016.   
[67] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 31953204, 2019.   
[68] Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 22632279, 2022.   
[69] Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 16971706, 2022.   
[70] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 22002209, 2021.   
[71] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, et al. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590, 2025.   
[72] OpenSearch-AI. Ops-mm-embedding-v1. https://huggingface.co/OpenSearch-AI/ Ops-MM-embedding-v1-2B. Accessed: 2026-01-24.   
[73] Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In Proceedings of the IEEE international conference on computer vision, pages 26412649, 2015.   
[74] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askel Pamela Mishkin, Jack Clark, Gretchen Krueger, and ya Sutskever. Learning transferable visual models from natural language supervision. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 87488763. PMLR, 2021.   
[75] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. Aokvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146162. Springer, 2022.   
[76] Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 84308439, 2019.   
[77ShareAI Lab. Sharegpt-chinese-english-90k: A bilingual chinese-english human-machine dialogue dataset. https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k, 2023. Hugging Face dataset repository.   
[78] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 83178326, 2019.   
[79] Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, and Xinlong Wang. Eva-clip-18b: Scaling clip to 18 billion parameters. arXiv:2402.04252, 2023.   
[80] Raghuveer Thirukovalluru, Rui Meng, Ye Liu, Mingyi Su, Ping Nie, Semih Yavuz, Yingbo Zhou, Wenhu . arXiv preprint arXiv:2505.11293, 2025.   
[81] Aäron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. CoRR, abs/1807.03748, 2018.   
[82] Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. V3det: Vast vocabulary visual detection dataset. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1984419854, 2023.   
[83] Junke Wang, Lingchen Meng, Zejia Weng, Bo He, Zuxuan Wu, and Yu-Gang Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. arXiv preprint arXiv:2311.07574, 2023.   
[84] Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. Vidorag: Visual document retrieval-augmented generation via dynamic iterative reasoning agents. arXiv preprint arXiv:2502.18017, 2025.   
[85] Zhen Wang, Xu Shan, Xiangxie Zhang, and Jie Yang. N24news: A new dataset for multimodal news classification. In Proceedings of the thirteenth language resources and evaluation conference, pages 67686775, 2022.   
[86] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers. In European Conference on Computer Vision, pages 387404. Springer, 2024.

[87] Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris. Fashion iq: A new dataset towards retrieving images by natural language feedback. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 1130711317, 2021.

[88] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Toralba. Sun database: Largescale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 34853492. IEEE, 2010.

[89] Youze Xue, Dian Li, and Gang Liu. Improve multi-modal embedding learning via explicit hard negative gradient amplifying. arXiv preprint arXiv:2506.02020, 2025.

[90] Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, et al. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1980319813, 2025.

[91] Jianxin Yang. Firefly. https://github.com/yangjianxin1/Firefly, 2023.

[92] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. Visrag: Vision-based retrieval-augmented generation on multimodality documents. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025.

[93] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. arXiv preprint arXiv:2312.00849, 2023.

[94] Tianyu Yu, Haoye Zhang, Yuan Yao, Yunkai Dang, Da Chen, Xiaoman Lu, Ganqu Cui, Taiwen He, Zhiyuan Liu, Tat-Seng Chua, and Maosong Sun. Rlaif-v: Aligning mllms through open-source ai feedback for super gpt-4v trustworthiness. arXiv preprint arXiv:2405.17220, 2024.

[95] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? arXiv preprint arXiv:2210.01936, 2022.

[96] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pretraining. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1197511986, 2023.

[97] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. In ECV, 2024.

[98] Chao Zhang, Haoxin Zhang, Shiwei Wu, Di Wu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, and Enhong Chen. Notellm-2: Multimodal large representation models for recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 28152826, 2025.

[99] Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu Su, and Ming-Wei Chang. Magiclens: Self-supervised image retrieval with open-ended instructions. arXiv:2403.19651, 2024.

[100] Shaolei Zhang, Qingkai Fang, Zhe Yang, and Yang Feng. Llava-mini: Effcient image and video large multimodal models with one vision token. arXiv preprint arXiv:2501.03895, 2025.

[101] Shuo Zhang, Biao Yang, Zhang Li, Zhiyin Ma, Yuliang Liu, and Xiang Bai. Exploring the capabilities of large multimodal models on dense text. In International Conference on Document Analysis and Recognition, pages 281298. Springer, 2024.

[102] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Gme: Improving universal multimodal retrieval by multimodal llms. arXiv preprint arXiv:2412.16855, 2024.

[103] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024.

[104] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. Swift:a scalable lightweight infrastructure for fine-tuning, 2024.   
[105] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence, 40(6):14521464, 2017.   
[106] Junjie Zhou, Yongping Xiong, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, and Defu Lian. Megapairs: Massive data synthesis for universal multimodal retrieval. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1907619095, 2025.   
[107] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.   
[108] Lanyun Zhu, Deyi Ji, Tianrun Chen, Haiyang Wu, and Shiqi Wang. Retrv-r1: A reasoning-driven mllm framework for universal and efficient multimodal retrieval. NeurIPS, 2025.   
[109] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 49955004, 2016.

# A Task Instruction for Embedding

Table11 Quey andtaretnstrutins or different datasets (art o .For the queri i mE5-ynthe [8, we use the original instructions from the dataset.   

<table><tr><td>Task</td><td>Dataset</td><td>Query Instruction</td><td>Target Instruction</td></tr><tr><td>T→T</td><td>BAAI-MTP [4]</td><td>Retrieve relevant texts based on a given query.</td><td>Represent the given text.</td></tr><tr><td>I→I</td><td>ImageNet-1K [13]</td><td>Find a image that looks similar to the provided image.</td><td>Represent the given image.</td></tr><tr><td rowspan="7">T→I</td><td>NIGHTS [19] BLIP Bootstrapped</td><td>Find dy-y hat ok miar e roi</td><td>Represent the given image.</td></tr><tr><td>Image-Text Pairs [47]</td><td>Retrieve relevant images based on a given query.</td><td>Represent the given image.</td></tr><tr><td>VisDial [12]</td><td>Repreent  vendialoge bout  age hicis us retrieval.</td><td>Represent the given image.</td></tr><tr><td>VisualNews [55]</td><td>Retrieve nagef te ven query.</td><td>Represent the given image.</td></tr><tr><td>MSCOCO [52]</td><td>Find me an everyday image that matches the given query.</td><td>Represent the given image.</td></tr><tr><td>Flickr30K [73]</td><td>Find me an everyday image that matches the given query.</td><td>Represent the given image.</td></tr><tr><td>ShareGPT4V [9]</td><td>Find me an everyday image that matches the given query.</td><td>Represent the given image.</td></tr><tr><td>Urban1K [97]</td><td></td><td>Find men veryayge hatathe heiv qy.</td><td>Represent the given image.</td></tr><tr><td rowspan="5">T→VD</td><td>mmE5-synthetic [8]</td><td></td><td>Represent the given image.</td></tr><tr><td>Docmatix [43]</td><td>Rerivevantsl ents bas qy.</td><td>Represent the given visual documents. Represent the given visual</td></tr><tr><td>Colpali [18]</td><td>Retrieve relevant visual documents based on a given query.</td><td>documents. Represent the given visual</td></tr><tr><td>VisRAG [92]</td><td>Reriveevantsl ents bas qy.</td><td>documents. Represent the given visual</td></tr><tr><td>ViDoSeek [84]</td><td>Retrieve elevantvisal ocments bas iv quy.</td><td>documents.</td></tr><tr><td></td><td>MMLongBench [64]</td><td>Retrieve relevant visual documents based on a given query.</td><td>Represent the given visual documents.</td></tr><tr><td rowspan="8">I→T</td><td>Wiki-SS-NQ [62]</td><td>Find the dcment image that can answer the given query.</td><td>Represent the given visual documents.</td></tr><tr><td>ImageNet-1K [13] HatefulMemes [37]</td><td>Represent the given image for classification. Represent the given image for binary classification to determine</td><td>Represent the given text.</td></tr><tr><td></td><td>whether it constitutes hateful speech or not.</td><td>Represent the given text.</td></tr><tr><td>VOC2007 [17]</td><td>Identify the object shown in the image.</td><td>Represent the given text.</td></tr><tr><td>SUN397 [88]</td><td>Identify the scene shown  theage.</td><td>Represent the given text.</td></tr><tr><td>Place365 [105]</td><td>Identify the scene shown  theage.</td><td>Represent the given text.</td></tr><tr><td>ImageNet-A [27]</td><td>Represent the given image for classification.</td><td>Represent the given text</td></tr><tr><td>ImageNet-R [26]</td><td>Represent the given image for classification.</td><td>Represent the given text</td></tr><tr><td></td><td>ObjectNet [3]</td><td>Identi ow</td><td>Represent the given text</td></tr><tr><td></td><td>Country-211 [74]</td><td>Identify the cuntry dpicted in thee.</td><td>Represent the given text</td></tr><tr><td></td><td>VisualNews [55]</td><td>Find a caption for the news in the given photo.</td><td>Represent the given text.</td></tr><tr><td></td><td>MSCOCO [52]</td><td>Find an image caption describing the given everyday image.</td><td>Represent the given text.</td></tr><tr><td></td><td>Flickr30K [73]</td><td>Find an image caption describing the given image.</td><td>Represent the given text.</td></tr><tr><td></td><td>ShareGPT4V [9]</td><td>Find an image caption describing the given image.</td><td>Represent the given text.</td></tr><tr><td></td><td>Urban1K [97]</td><td>Find an image caption describing the given image.</td><td>Represent the given text.</td></tr><tr><td></td><td>SugarCrepe [28]</td><td>Find an image caption describing the given image.</td><td>Represent the given text.</td></tr><tr><td></td><td>mmE5-synthetic [8]</td><td></td><td>Represent the given text.</td></tr></table>

Tabl1: Quey and taret instrutions ordifferent datasets art 2. Fo the queri i mmE5-ynthe [8], we use the original instructions from the dataset.   

<table><tr><td>Task</td><td>Dataset</td><td>Query Instruction</td><td>Target Instruction</td></tr><tr><td rowspan="6">IT→I</td><td>MegaPairs [106]</td><td>Repreent the iven mage with the iven query and rere the related images.</td><td>Represent the given image.</td></tr><tr><td>CIRR [60]</td><td>Given an image, find a similar everyday image with the de- scribed changes as the given query.</td><td>Represent the given image.</td></tr><tr><td>N24News [85]</td><td>Represent the given news image with the given query for domain classification.</td><td>Represent the given text.</td></tr><tr><td>MSCOCO [52]</td><td>Sl sented by the given query.</td><td>Represent the cropped image.</td></tr><tr><td>FashionIQ [87]</td><td>Find an image to match the fashion image and style note.</td><td>Represent the given image.</td></tr><tr><td>Visual7W-Pointing [109]</td><td>Select the portion of the image that answers the given query.</td><td>Represent the cropped image.</td></tr><tr><td></td><td>RefCOCO [36]</td><td>S sented by the given query.</td><td>Represent the cropped image.</td></tr><tr><td rowspan="10"></td><td>mmE5-synthetic [8]</td><td></td><td>Represent the given image.</td></tr><tr><td>Docmatix [43]</td><td>Repreent the iven mage with the iven query and rive the answer.</td><td>Represent the given text.</td></tr><tr><td>OK-VQA [67]</td><td>Represent the given image with the given query and retrieve the answer.</td><td>Represent the given text.</td></tr><tr><td>A-OKVQA [75]</td><td>Repreent the iven age with the ivn query and rive the answer. Represent the given image with the given query and retrieve</td><td>Represent the given text.</td></tr><tr><td>DocVQA [70] InfographicVQA [69]</td><td>the answer. Represent the given image with the given query and rtrive</td><td>Represent the given text.</td></tr><tr><td>ChartQA [68]</td><td>the answer. Represent the given image with the given query and retrieve</td><td>Represent the given text.</td></tr><tr><td>Visual7W [109]</td><td>the answer. Represent the given image with the given query and retrieve</td><td>Represent the given text.</td></tr><tr><td>ScienceQA [61]</td><td>the answer. Represent the given image with the given query and retrieve</td><td>Represent the given text.</td></tr><tr><td>VizWiz [24]</td><td>the answer. Repreent the iven mage with the iven query and rive</td><td>Represent the given text.</td></tr><tr><td></td><td>the answer. Represent the given image with the given query and rtrive</td><td>Represent the given text.</td></tr><tr><td>GQA [30]</td><td>the answer.</td><td>Represent the given image with the given query and retreve</td><td>Represent the given text.</td></tr><tr><td></td><td>TextVQA [78]</td><td>the answer.</td><td>Represent the given text.</td></tr><tr><td rowspan="3">T→IT</td><td>mmE5-synthetic [8] WebQA [7]</td><td>Find a related image and text content from Wikipedia that</td><td>Represent the given text. Represent the given Wikipedia im</td></tr><tr><td>EDIS [58]</td><td>answers the given query. Find a related image and text content from a news that matches</td><td>age with related text information. Represent the given image with re</td></tr><tr><td>mmE5-synthetic [8]</td><td>the provided query.</td><td>lated text information. Represent the given image with re</td></tr><tr><td rowspan="3">IT→IT</td><td>OVEN [29]</td><td>Retrieve a Wikipedia image-description pair that provides</td><td>lated text information. Represent the given image with re</td></tr><tr><td></td><td>evidence for the given query. Select the identical object in the image that follows the given</td><td>lated text information. Represent the object in the image</td></tr><tr><td>RefCOCO-Matching [36]</td><td>query.</td><td>that follows the given text.</td></tr></table>

# B Judgment Instruction for Hard Negatives Filtering Using MLLM in Stage 3

Table 13: Instructions for MLLM judgment in the stage 3. For the HatefulMemes dataset, because it has only Yes/No labels, removing the ground truth leaves only correct negative samples. Therefore, we did not use MLLMs to judge this dataset.   

<table><tr><td>Domain</td><td>Dataset</td><td>Judgment Instruction</td></tr><tr><td rowspan="10">MMEB</td><td>ImageNet-1K [13]</td><td>Determine whether a given image contains an object specified by a class label. Given news containing both image and text content, determine whether the category of the news matches the</td></tr><tr><td>N24News [85]</td><td>given category.</td></tr><tr><td>VOC2007 [17]</td><td>Determine whether a given image contains an object specified by a class label.</td></tr><tr><td>SUN397 [88]</td><td>Determine whether the given image matches the given scene description.</td></tr><tr><td>OK-VQA [67]</td><td>Given a reference image and a question, determine whether the provided answer is correct.</td></tr><tr><td>A-OKVQA [75]</td><td>Given a reference image and a question, determine whether the provided answer is correct.</td></tr><tr><td>DocVQA [70]</td><td>Given a reference image and a question, determine whether the provided answer is correct.</td></tr><tr><td>InfographicsVQA [69]</td><td>Given a reference image and a question, determine whether the provided answer is correct.</td></tr><tr><td>ChartQA [68]</td><td>Given a reference image and a question, determine whether the provided answer is correct.</td></tr><tr><td>Visual7W [109]</td><td>Given a reference image and a question, determine whether the provided answer is correct.</td></tr><tr><td>VisDial [12]</td><td>Determine whether the given dialogue text is relevant to the given target image.</td></tr><tr><td>CIRR [60]</td><td>Gvxahe by the text instruction is relevant to the target image.</td></tr><tr><td>VisualNews (T→I) [55]</td><td>Determine whether the given query text is relevant to the given image.</td></tr><tr><td>VisualNews (I→T) [55]</td><td>Determine whether the given text can serve as a caption for the given image.</td></tr><tr><td>MSCOCO (T→I) [52]</td><td>Determine whether the given query text is relevant to the given image.</td></tr><tr><td>MSCOCO (I→T) [52]</td><td>Determine whether the given text can serve as a caption for the given image.</td></tr><tr><td>NIGHTS [19]</td><td>Determine whether the two given images are similar.</td></tr><tr><td>WebQA [7]</td><td>Determine whether the query text is relevant to the given image-text mixed content.</td></tr><tr><td>MSCOCO (IT→I) [52]</td><td>Givna efea  jec label pre in tex point   je  theeemage an  extracted from an image, determine whether the text label points to the given crop.</td></tr><tr><td rowspan="2">VisDoc</td><td>Colpali [18]</td><td>Determine whether the given query text is relevant to the given visual document image.</td></tr><tr><td>VisRAG [92]</td><td>Determine whether the given query text is relevant to the given visual document image.</td></tr></table>

# C Hyperparameters for Embedder Training

Table14Hyperparameters or stage and stage  training "Warm-Up" denotes a contrastive warm-up phas st  using only in-batch negatives; "Global-HNM" refers to pretraining in stage 2 with Global Hard Negative Mining; "MLLM-Judge-FT" indicates finetuning with an MLLM as a judge.   

<table><tr><td rowspan="2">Hyperparameter</td><td colspan="2">Stage 2 (Warm-Up)</td><td colspan="2">Stage 2 (Global-HNM)</td><td colspan="2">Stage 3 (MLLM-Judge-FT)</td></tr><tr><td>2B</td><td>8B</td><td>2B</td><td>8B</td><td>2B</td><td>8B</td></tr><tr><td>#Samples</td><td>16M</td><td>16M</td><td>16M</td><td>16M</td><td>1.5M</td><td>1.5M</td></tr><tr><td>#Hard Negatives</td><td colspan="2">0 2</td><td colspan="2">2</td><td>12</td><td>12</td></tr><tr><td>#GPUs</td><td colspan="6">48</td></tr><tr><td>Maximum learning rate</td><td colspan="6">2 × 10−4</td></tr><tr><td>Temperature</td><td colspan="6">0.03</td></tr><tr><td>LoRA rank</td><td colspan="6">16</td></tr><tr><td>Training epochs</td><td colspan="6">2</td></tr><tr><td>Batch size per device</td><td>128</td><td>72</td><td>64</td><td>48</td><td>12</td><td>10</td></tr></table>