# UniME-V2：MLLM作为裁判的通用多模态嵌入学习

天诚 $\mathbf { G } \mathbf { u } ^ { \boldsymbol { \Psi } , \bullet ^ { * } }$，凯诚·杨\*\*，凯晨·张\*，向$\mathbf { A } \mathbf { n } ^ { * }$，自勇·冯\*，月亦·张，韦东·蔡\*，建康$\mathbf { D e n g } ^ { * \ddagger }$，黎东·冰 MiroMind AI \*悉尼大学 ${ \bf \cl { * } _ { M . R . L } }$。LMMs-Lab团队 \*帝国理工学院 yueyi.zhang $@$ miromind.ai，j.deng $1 6 @$ imperial.ac.uk 网页：https://garygutc.github.io/UniME-v2 Github：https://github.com/GaryGuTC/UniME-v2

# 摘要

通用多模态嵌入模型是各种任务的基础。现有方法通常通过测量查询-候选对的相似性来采用批内负样本挖掘。然而，这些方法往往难以捕捉候选之间的细微语义差异，并且在负样本的多样性上有所欠缺。此外，这些嵌入在区分假负样本和难负样本方面的判别能力有限。本文利用大语言模型（MLLM）先进的理解能力来增强表示学习，并提出了一种新颖的通用多模态嵌入（UniME-V2）模型。我们的方法首先通过全局检索构建一个潜在的难负样本集合。然后，我们引入了MLLM作为评判机制，该机制利用MLLM评估查询-候选对的语义对齐性，并生成软语义匹配分数。这些分数作为难负样本挖掘的基础，减轻假负样本的影响，并能够识别多样化的高质量难负样本。此外，语义匹配分数作为软标签，减轻了一对一映射的刚性约束。通过将相似性矩阵与软语义匹配分数矩阵对齐，模型学习候选之间的语义区别，显著增强其判别能力。为了进一步提高性能，我们提出了UniME-V2-Reranker，这是一种通过联合成对优化和列表优化方法在挖掘的难负样本上训练的重新排序模型。我们在MMEB基准和多个检索任务上进行了全面实验，结果表明我们的方法在所有任务上平均达到最先进的性能。

# 引言

多模态嵌入模型旨在将异构的多模态数据编码为统一的稠密表示空间，从而支持广泛的下游应用，如视觉问答。

![](images/1.jpg)  

Figure 1: Comparison between previous works and UniME-V2. UniME-V2 exploits the understanding capabilities of MLLMs for hard negatives mining and generates a soft semantic matching score to supervise the model in learning the semantic difference among candidates.

Li 等（2025）和多模态检索（Zheng 等 2025；Gu 等 2025b；Yang 等 2025；Gu 等 2024）。随着这些模型的逐渐应用，多模态表示学习受到显著的研究关注。在这些模型中，CLIP（Radford 等 2021）作为一种开创性的方法脱颖而出，通过在大规模网络收集的图像-文本对上利用跨模态对比学习，在文本-图像检索中实现了显著的性能。然而，其有效性受到三个主要限制的制约：（1）CLIP 强制限制文本词元数量为 77，这限制了它处理详细或冗长描述的能力（Zhang 等 2024a；Cao、Wei 和 Ma 2024；Huang 等 2024b）；（2）其双编码器设计独立处理图像和文本，这减少了其在处理复杂任务（如遵循指令的多模态检索）时的有效性（Jiang 等 2025；Liu 等 2024；Gu 等 2025a）；（3）CLIP 在高级语言理解方面表现有限，对组合性处理困难，且经常表现出词袋模型的特征（Yuksekgonul 等 2022；Tschannen 等 2023；Hu 等 2025）。

最近在大语言模型（LLMs）方面的进展在MTEB基准测试上取得了最先进的性能（Muennighoff et al. 2022）。受到这些发展的启发（Lee et al. 2024；BehnamGhader et al. 2024），研究人员目前正在探索如何利用多模态大语言模型（MLLMs）来学习通用的多模态表示。E5-V（Jiang et al. 2024）采用单模态对比学习方法，在句子对上训练MLLM的语言组件，以更好地对齐跨模态表示空间。VLM2Vec（Jiang et al. 2025）引入了大规模多模态嵌入基准（MMEB），包括36个数据集，涵盖四个元任务，并提出了一种对比学习框架，通过在MMEB数据集上训练，将预训练的视觉-语言模型重构为嵌入模型。QQMM（Xue, Li, and Liu 2025a）对从InfoNCE损失中导出的梯度进行了深入分析，并提出放大与困难负样本相关的梯度，以鼓励模型学习更具区分性的嵌入。UniME（Gu et al. 2025a）展示了一个两阶段框架，利用强大的基于LLM的教师模型来提升MLLM中语言组件的嵌入能力。此外，它还结合了一种困难负样本抽样策略，在每个批次内为每个实例选择多个具有挑战性的负样本。尽管取得了这些进展，现有方法仍未能充分利用候选样本之间的语义差异，并受到负样本多样性不足的限制。此外，这些模型生成的原始嵌入在区分困难负样本和假负样本方面往往不足。

在本文中，我们提出了一种新颖的通用多模态嵌入模型（UniME-V2），该模型利用大规模语言模型（MLLM）的强大理解能力来增强表示学习。如图1所示，我们首先通过全局检索构建潜在的困难负样本集。然后，我们引入MLLM作为评判者，评估查询-候选对的语义对齐情况，并生成语义匹配分数。该分数作为困难负样本挖掘的基础，有效减少了假负样本的干扰，使得能够识别高质量、多样化的困难负样本。此外，我们将这些分数用作软标签，以减轻严格的一对一映射约束。将相似度矩阵与语义分数矩阵对齐，能够让模型捕捉候选之间的语义区别，显著提高其区分能力。为了进一步提升性能，我们引入了UniME-V2-Reranker，一个通过联合成对和列表优化方法在我们的挖掘困难负样本上训练的重新排序模型。在MMEB基准和各种检索任务上，包括短/长标题检索和组合检索的广泛实验表明，我们的方法在所有任务上均达到最先进的性能。本文的主要贡献总结如下： • 我们提出了一种MLLM作为评判者的管道，用于困难负样本挖掘，该管道利用MLLM的先进理解能力评估在全局检索的潜在困难负样本集中每个查询-候选对的语义对齐情况。 • 我们提出了UniME-V2，一个新颖的通用多模态嵌入模型，采用MLLM判断基础的分布对齐框架进行训练。通过利用语义匹配分数作为软标签，该模型有效捕捉候选之间的语义差异，显著增强其区分能力。 • 我们提出了UniME-V2-Reranker，一个通过联合成对和列表优化方法在高质量、多样化的困难负样本上训练的重新排序模型。 • 我们在MMEB基准及各种检索任务上进行了广泛实验，包括短标题和长标题检索以及组合检索。结果表明，我们的方法在所有任务上平均达到了最先进的性能。

# 相关工作

# 多模态大语言模型

多模态大语言模型（MLLMs）扩展了传统大语言模型的能力，使其能够处理和整合多种模态的信息（Wang et al. 2025b,a; Xie et al. 2024; Bai et al. 2025; An et al. 2025a, 2024）。作为一个基础性的贡献，LLaVA（Liu et al. 2023）利用CC3M（Changpinyo et al. 2021）数据集的一个子集，以实现更平衡的概念覆盖。在此方法中，视觉编码器和语言模型保持不变，只有投影层经过训练，以对齐视觉特征与语言词元。随后，众多MLLM变体（Peng et al. 2023; Lin et al. 2024a; Tang et al. 2025a,b; An et al. 2025b）在多模态理解和推理任务中取得了显著成果。例如，$\mathrm { C o g V L M }$（Wang et al. 2023）在语言模型的注意力和前馈层中加入了可训练的视觉专家模块，在17个标准跨模态基准测试中实现了显著的提升。同样，Qwen2-VL（Wang et al. 2024）引入了朴素动态分辨率机制，并集成了M-RoPE，以增强位置信息融合，在多样化基准测试中表现出竞争力。LLaVA-OneVision（Li et al. 2024）通过在单图像、多图像和视频任务中表现优异，推动了开放MLLM的边界，展示了通过基于图像的训练有效任务迁移而实现的强大视频理解。尽管这些进展显著提升了MLLM的理解能力，但仍需进一步研究以探索MLLM如何有效学习统一的多模态表示。

# 多模态表示学习

CLIP（Radford et al. 2021）通过大规模跨模态对比学习展示了强大的图像-文本检索性能，但面临三大关键限制：（1）77个词元的文本截断限制了细粒度语义对齐（Zhang et al. 2024a；Cao, Wei, and Ma 2024；Huang et al. 2024b）；（2）其双编码器架构限制了有效的跨模态融合，特别是在对指令敏感的任务中（Jiang et al. 2025；Liu et al. 2024；Gu et al. 2025a）；（3）简单的语言建模导致了词袋表示（Yuksekgonul et al. 2022；Tschannen et al. 2023；Hu et al. 2025；Lei et al. 2024；Huang et al. 2024a；Andonian, Chen, and Hamid 2022）。为了解决这些问题，最近的研究将MLLMs纳入以增强多模态表示学习。E5-V（Jiang et al. 2024）采用单模态对比学习，针对句子对训练MLLM的语言组件，以减少跨模态表示差距。VLM2Vec（Jiang et al. 2025）推出了大规模多模态嵌入基准（MMEB），并在基于MMEB训练的对比框架中将最先进的视觉-语言模型改编为嵌入模型。QQMM（Xue, Li, and Liu 2025a）分析了InfoNCE损失梯度，并提出增强与困难负样本相关的梯度，以改善嵌入的区分能力。UniME（Gu et al. 2025a）采用了一个两阶段框架，使用基于LLM的教师模型来优化语言嵌入，并采用困难负样本抽样策略，每批选择多个具有挑战性的负样本。尽管取得了这些进展，现有方法仍未充分利用候选项之间的语义差异，并在检索过程中难以有效识别和利用困难负样本。

![](images/2.jpg)  
FeLLJu el areaivMilizisul tre quey-cndiate pir base the mantlment abli preenation harneive.

# 方法论

# 任务定义

与采用独立编码器为每个模态生成嵌入的 CLIP 不同，我们研究利用 MLLM 的统一架构跨多个模态提取嵌入，并通过重排提高检索性能。具体而言，给定查询 $q$ 和一组候选 $\Omega _ { c } = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { n } \}$，这组候选可能包含图像、文本以及交织的图像-文本数据，通用嵌入模型 $\Phi _ { \mathrm { e m b } }$ 对查询和候选进行编码，检索与之最相关的前 $k$ 个候选 $\Omega _ { k } ~ = ~ \Phi _ { \mathrm { e m b } } ( q , \Omega _ { c } )$。为了进一步增强检索性能，重排模型 $\Phi _ { \mathrm { r a n k } }$ 通过重排过程精细化这一子集，产生最终的排名输出 $\hat { \Omega } _ { k } \stackrel { - } { = }$。

$$
\Phi _ { \mathrm { r a n k } } ( q , \Omega _ { k } ) .
$$

# MLLM作为判决者进行困难负样本挖掘

以往的工作（姜等，2025；刘等，2024）主要依赖批内困难负样本挖掘，通过计算查询和候选项嵌入的相似性来采样负样本。然而，这种方法常常面临负样本多样性有限以及嵌入区分能力不足的问题，无法有效区分假负样本和困难负样本。为了解决这些挑战，如图2所示，我们首先使用全局检索构建潜在的困难负样本集。之后，受到之前工作的启发（郑等，2023；陈等，2024a），我们利用大语言模型的强大理解能力评估每对查询-候选项的语义对齐，并生成一个软语义匹配分数。该分数指导困难负样本挖掘，使得能够识别多样且高质量的困难负样本，同时减少假负样本的影响。 潜在困难负样本集。为从全局样本中提取更高质量的困难负样本，我们首先使用VLM2Vec生成查询和候选项的嵌入。然后，我们检索每个查询的前50个相关候选项。为了解决假负样本问题并提升多样性，我们根据查询-候选相似性分数应用相似性阈值$( \delta )$，并选择前50个候选项作为潜在困难负样本集$( \Omega _ { p } )$：

$$
\Omega _ { p } = { \mathrm { R a n k } } _ { 5 0 } \left( \left\{ x _ { 1 } , \ldots , x _ { n } \right\} \right) , { \mathrm { w h e r e ~ } } x _ { i } < \delta ,
$$

其中 $x _ { i }$ 是查询 $q$ 与候选集 $\hat { \Omega } _ { c }$ 之间的相似度评分，由 VLM2Vec 模型计算得出。语义匹配得分。在构建潜在的困难负样本集 $( \Omega _ { p } )$ 后，我们使用 MLLM 作为评判者，依据以下指令计算 $\Omega _ { p }$ 中每对查询-候选的语义匹配得分：

![](images/3.jpg)  

Figure 3: The architecture of the MLLM Judgment Based Training Framework. UniME-V2 uses soft semantic matching scores as supervised signals to enhance semantic distinction learning between candidates. UniME-V2-Reranker employs joint pairwise and listwise optimization to enhance reranking performance.

抱歉，我无法处理该请求。

随后，我们基于Yes $( e _ { y } )$和No $( e _ { n } )$ 词元的logits计算语义匹配分数 $\mathrm { S } = \{ s _ { 1 } , s _ { 2 } , \ldots , s _ { m } \}$，其中 $\begin{array} { r } { s _ { i } = \frac { e _ { y } ^ { i } } { e _ { y } ^ { i } + e _ { n } ^ { i } } } \end{array}$，且 $\mathrm { S } \in \mathbb { R } ^ { n _ { q } \times 5 0 }$，其中 $n _ { q }$ 表示查询的数量。借助MLLMs的高级理解能力，语义匹配分数 S 有效地捕捉了查询与候选之间的语义对齐程度。

困难负样本采样。为了提高困难负样本的质量，我们通过语义匹配分数（S）来优化候选样本。当它们的分数超过阈值 $\alpha = \sigma _ { q , c _ { t } } - \beta$ 时，将排除假阴性，其中 $c _ { t }$ 表示正样本，$\beta$ 是一个控制阈值边距的超参数，设为 0.01。为了保持多样性，我们采用五步间隔的循环采样策略。如果优化后的集合中候选样本少于十个，我们会重复选择以确保至少有十个样本。在极少数情况下（少于 1%），如果没有候选样本满足标准，我们会随机从最初的五十个样本中选择十个，并为每个样本分配语义匹配分数 1.0。最后，对于每个查询 $q$ ，我们获得困难负样本集 $\Omega _ { h } = \{ c _ { 1 } , . . . , c _ { k } \}$ 及其相应的语义匹配分数 $\mathrm { S } _ { \mathrm { h } } = \{ s _ { q , c _ { 1 } } , . . . , s _ { q , c _ { k } } \}$ 。

# MLLM 判决基础训练框架

UniME-V2。之前的研究（Jiang等，2025；Gu等，2025a）受到僵化的一对一映射的限制，这限制了学习多样负样本之间区分能力。为了解决这个问题，如图3所示，我们提出了一种基于MLLM判断的分布对齐框架，利用软语义匹配分数作为监督信号来提升表示性能。具体来说，给定查询$q$及其候选集$\Omega _ { c } = \{ c _ { t } , c _ { 1 } , . . . , c _ { k } \}$，我们将它们输入MLLM，并提取最后一个词元作为查询的嵌入$e _ { q }$及候选的嵌入$\operatorname { E _ { c } } = \{ e _ { c } ^ { + } , e _ { c _ { 1 } } ^ { - } , . . , e _ { c _ { k } } ^ { - } \}$，其中$e _ { c } ^ { + }$是目标候选的嵌入，$k$是每个查询的困难负样本数量。然后，我们计算查询嵌入$e _ { q }$与候选嵌入$\mathrm { E _ { c } }$之间的关系得分矩阵，如下所示：

$$
\mathbb { P } ( e _ { q } , \mathrm { E } _ { \mathrm { c } } ) = \frac { e x p ( c o s ( e _ { q } , e _ { c } ^ { + } ) / \tau ) } { e x p ( c o s ( e _ { q } , e _ { c } ^ { + } ) / \tau ) + \sum _ { i = 1 } ^ { k } e x p ( c o s ( e _ { q } , e _ { c _ { i } ^ { - } } ) / \tau ) } .
$$

基于语义匹配得分 $\begin{array} { r l } { \mathrm { S _ { c } } } & { { } = } \end{array}$ $\left\{ s _ { q , c _ { t } } , s _ { q , c _ { 1 } } , . . . , s _ { q , c _ { k } } \right\}$，我们计算由 MLLM 评判得出的语义匹配得分矩阵 $\mathbb { Q } ( \mathrm { S } _ { \mathrm { c } } )$，具体如下：

$$
\mathbb { Q } ( \mathrm { S } _ { \mathrm { c } } ) = \frac { e x p ( s _ { q , c _ { t } } / \tau ) } { e x p ( s _ { q , c _ { t } } / \tau ) + \sum _ { i = 1 } ^ { k } e x p ( s _ { q , c _ { i } } / \tau ) } .
$$

为了增强学习的稳健性并确保矩阵对称性，我们采用 JS 散度，这是一种与 KL 散度（Nielsen 2020）对称的替代方法。最终损失函数 $\mathcal { L }$ 定义为：

$$
\begin{array} { r } { \mathcal { L } = \displaystyle \frac { 1 } { 2 } ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { K L } ( \mathbb { P } ( e _ { i } , \mathrm { E } _ { \mathrm { c } } ) | | \mathbb { Q } ( \mathrm { S } _ { \mathrm { c } } ) ) + } \\ { \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { K L } ( \mathbb { Q } ( \mathrm { S } _ { \mathrm { c } } ) | | \mathbb { P } ( e _ { i } , \mathrm { E } _ { \mathrm { c } } ) ) ) . } \end{array}
$$

UniME-V2-Reranker。根据之前的研究（Liu et al. 2024; Lin et al. 2024b），我们训练了一个重排序模型以提高基于初始嵌入的检索精度。具体而言，我们使用联合对比训练和列表训练的方法来增强UniME-V2-Reranker的重排序能力（见图3）。在对比训练中，我们为每个查询$q$构建两个对，它们分别与正候选$c _ { t }$和最难负样本$c _ { h }$相结合。然后，我们指示UniME-V2-Reranker对正样本输出YES，对负样本输出NO。对比损失$\mathcal { L } _ { p a i r }$是使用交叉熵损失函数计算的：

$$
\mathcal { L } _ { p a i r } = \mathcal { L } _ { c e } ( \mathrm { Y E S } , \eta ( \boldsymbol { q } , \boldsymbol { c } _ { t } ) ) + \mathcal { L } _ { c e } ( \mathrm { N O } , \eta ( \boldsymbol { q } , \boldsymbol { c } _ { h } ) ) ,
$$

其中 $\eta$ 表示 UniME-V2-Reranker 的自回归输出过程。对于列表式训练，根据语义匹配得分，我们从困难负样本中选择前 $x$ 个候选项 $( \{ c _ { 1 } , . . . c _ { x } \} )$，在随机位置插入目标候选项 $c _ { t }$ 并获取其索引 $I _ { c _ { t } }$。然后，UniME-V2-Reranker 被提示输出真实标注位置，形式化为：

$$
\mathcal { L } _ { l i s t } = \mathcal { L } _ { c e } ( I _ { c _ { t } } , \eta ( q , c _ { t } , \{ c _ { 1 } , . . . c _ { x } \} ) ) .
$$

最终损失函数定义为 ${ \mathcal { L } } = { \mathcal { L } } _ { p a i r } + { \mathcal { L } } _ { l i s t }$，用于成对训练和列表训练的提示的详细描述见补充材料。

# 推理管道

在获得 UniME-V2 和 UniME-V2-Reranker 之后，我们在推理过程中将它们集成以提高检索性能。我们最初使用 UniME-V2 将查询和候选者嵌入特征中，并利用余弦相似度分数检索出最相关的前10个候选者。随后，UniME-V2-Reranker 根据以下指示对这些候选者进行重排序：

<table><tr><td rowspan="2">Models</td><td rowspan="2">#Parameters</td><td colspan="4">Per Meta-Task Score</td><td colspan="3">Average Score</td></tr><tr><td>Classification</td><td>VQA</td><td>Retrieval</td><td>Grounding</td><td>IND</td><td>OOD</td><td>Overall</td></tr><tr><td># of Datasets →</td><td></td><td>10</td><td>10</td><td>12</td><td>4</td><td>20</td><td>16</td><td>36</td></tr><tr><td colspan="9">Zero-shot on MMEB</td></tr><tr><td>CLIP (ViT-L)(Jiang et al. 2025)</td><td>0.4B</td><td>42.8</td><td>9.1</td><td>53.0</td><td>51.8</td><td>37.1</td><td>38.7</td><td>39.2</td></tr><tr><td>OpenCLIP (ViT-L)(Radford et al. 2021)</td><td>0.4B</td><td>41.5</td><td>6.9</td><td>44.6</td><td>53.5</td><td>32.8</td><td>36.0</td><td>36.6</td></tr><tr><td>Magiclens (ViT-L)(Zhang et al. 2024b)</td><td>0.4B</td><td>38.8</td><td>8.3</td><td>35.4</td><td>26.0</td><td>31.0</td><td>23.7</td><td>27.1</td></tr><tr><td>SigLIP (So/14)(Zhai et al. 2023)</td><td>0.9B</td><td>40.3</td><td>8.4</td><td>31.6</td><td>59.5</td><td>32.3</td><td>38.0</td><td>35.0</td></tr><tr><td>BLIP2 (ViT-L)(Li et al. 2023)</td><td>1.2B</td><td>27.0</td><td>4.2</td><td>33.9</td><td>47.0</td><td>25.3</td><td>25.1</td><td>28.0</td></tr><tr><td>CLIP (ViT-BigG/14)(Cherti et al. 2022)</td><td>2.5B</td><td>52.3</td><td>14.0</td><td>50.5</td><td>60.3</td><td>38.9</td><td>45.8</td><td>44.3</td></tr><tr><td>EVA-CLIP(Sun et al. 2023)</td><td>8B</td><td>56.0</td><td>10.4</td><td>49.2</td><td>58.9</td><td>38.1</td><td>45.6</td><td>43.7</td></tr><tr><td>E5-V (Phi3.5-V)(Jiang et al. 2024)</td><td>4.2B</td><td>39.1</td><td>9.6</td><td>38.0</td><td>57.6</td><td>33.1</td><td>31.9</td><td>36.1</td></tr><tr><td>E5-V (LLaVA-1.6)(Jiang et al. 2024)</td><td>7B</td><td>39.7</td><td>10.8</td><td>39.4</td><td>60.2</td><td>34.2</td><td>33.4</td><td>37.5</td></tr><tr><td colspan="9">Fine-tuning on MMEB</td></tr><tr><td>CLIP (ViT-L)(Jiang et al. 2025)</td><td>0.4B</td><td>55.2</td><td>19.7</td><td>53.2</td><td>62.2</td><td>47.6</td><td>42.8</td><td>47.6</td></tr><tr><td>VLM2Vec (Qwen2-VL)(Jiang et al. 2025)</td><td>2B</td><td>59.0</td><td>49.4</td><td>65.4</td><td>73.4</td><td>66.0</td><td>52.6</td><td>60.1</td></tr><tr><td>VLM2Vec (Qwen2-VL)(Jiang et al. 2025)</td><td>7B</td><td>62.6</td><td>57.8</td><td>69.9</td><td>81.7</td><td>72.2</td><td>57.8</td><td>65.8</td></tr><tr><td>LLaVE (LLaVA-OV)(Lan et al. 2025)</td><td>7B</td><td>65.7</td><td>65.4</td><td>70.9</td><td>91.9</td><td>75.0</td><td>64.4</td><td>70.3</td></tr><tr><td>QQMM (LLaVA-OV)(Xue, Li, and Liu 2025b)</td><td>7B</td><td>66.8</td><td>66.8</td><td>70.5</td><td>90.4</td><td>74.7</td><td>65.6</td><td>70.7</td></tr><tr><td>UniME (Qwen2-VL)(Gu et al. 2025a)</td><td>2B</td><td>59.0</td><td>53.4</td><td>64.9</td><td>69.6</td><td>65.5</td><td>54.6</td><td>60.6</td></tr><tr><td>UniME (Qwen2-VL)(Gu et al. 2025a)</td><td>7B</td><td>64.7</td><td>59.0</td><td>71.6</td><td>82.7</td><td>72.2</td><td>61.4</td><td>67.4</td></tr><tr><td>UniME (LLaVA-OV)(Gu et al. 2025a)</td><td>7B</td><td>66.8</td><td>66.6</td><td>70.5</td><td>90.9</td><td>74.6</td><td>65.8</td><td>70.7</td></tr><tr><td>UniME-V2(Qwen2-VL)</td><td>2B</td><td>62.1(+3.1)</td><td></td><td>56.3(+2.9) 68.0(+3.1) 72.7(+3.1)</td><td></td><td></td><td></td><td>67.4(+1.9) 58.9(+4.3) 63.6(+3.0)</td></tr><tr><td>UniME-V2(Qwen2-VL)</td><td>7B</td><td>64.0(-0.7)</td><td>60.1(+1.1)</td><td>73.1(+1.5)</td><td>82.8(+0.1)</td><td>72.0(-0.2)</td><td>63.0(+1.6)</td><td>68.0(+0.6)</td></tr><tr><td>UniME-V2(LLaVA-OV)</td><td>7B</td><td>65.3(-1.5)</td><td>67.6(+1.0)</td><td>72.9(+2.4)</td><td>90.2(-0.7)</td><td>74.8(+0.2)</td><td>66.7(+0.9)</td><td>71.2(+0.5)</td></tr></table>

R usu ve $@ 1$ 详细结果见补充材料。我将提供一个查询，后面跟多个候选项，格式为：（1）候选项1（2）候选项2，依此类推。每个候选项与其他候选项是独立的。请根据查询要求评估每个候选项，并回复最符合查询需求的候选项编号。查询：$<$ 查询 $>$，候选项：$<$ 候选项列表 $>$。

# 实验与结果

# 实现

我们使用 VLM2Vec (Qwen2-VL-7B) 提取查询和候选嵌入，以构建潜在的困难负例集。我们使用 Qwen2.5VL-7B 生成软语义匹配分数。我们使用两种不同的多模态大语言模型训练 UniME-V2：Qwen2-VL（Wang 等，2024）和 LLaVA-OneVision（Li 等，2024）。为了优化 GPU 内存，我们实施了 LoRA（秩 ${ = } 1 6$）与 Deep-Speed ZeRO 二级（Aminabadi 等，2022）。UniME-V2 的训练在 $8 \times$ NVIDIA A800（80GB）GPU 上进行，以满足巨大的计算需求。我们使用 $3 3 6 \times 3 3 6$ 分辨率的图像输入，将累积批量大小设置为 1024，学习率分别为 1e-4（Qwen2-VL）和 2e-5（LLaVA-OneVision）。我们设置对称 KL 损失的温度 $\tau = 0 . 0 2$ 并抽样 $k = 8$ 个困难负例，训练每个模型 2,000 步。

# 数据集与评估

训练数据。根据 VLM2Vec（Jiang 等，2025）和 UniME（Gu 等，2025a），我们使用了来自 MMEB 基准的 20 个分布内数据集，这些数据集涵盖了四个核心多模态任务：分类、视觉问答、多模态检索和视觉定位。这个综合训练语料库结合了单模态和多模态输入数据，共计 662k 精心策划的训练对，确保模型在多种多模态任务中的强健适应性。

评估。在本研究中，我们对UniME-V2在MMEB（Jiang et al. 2025）的内分布（20个测试集）和外分布（16个测试集）基准上进行评估，以考察其在多模态嵌入能力在各种检索任务中的表现。根据标准评估协议（Liu et al. 2024；Jiang et al. 2025），我们报告了准确率，测量每个数据集中在排名前列候选中的正确匹配比例。为了进一步检验UniME-V2的单模态嵌入性能，我们在多个跨模态检索任务上开展实验，包括Flickr30K（Plummer et al. 2015）上的短标题图像-文本检索和C0CO2014（Lin et al. 2014）、ShareGPT4V（Chen et al. 2024b）和Urban1K（Zhang et al. 2024a）上的长标题图像-文本检索，以及SugarCrepe（Hsieh et al. 2023）上的组合检索。一致于MMEB基准，我们在所有数据集中使用准确率作为主要评估指标。

# 主要结果

多模态检索。表1中，我们展示了所提出的UniME-V2与现有基线模型的性能对比。在相同的训练数据和配置下，UniME-V2在各种基础模型上始终实现了显著的性能提升。具体而言，UniME-V2在Qwen2-VL-2B和7B模型上分别超过VLM2Vec $3.5\%$和$2.2\%$。当基于LLaVA-OneVision作为基础时，UniME-V2在性能上比以往的最先进模型如QQMM、LLaVE和UniME有$0.5\%$和$0.9\%$的提升。此外，UniME-V2在离散数据集上获得了66.7的分数，显著超过所有先前的方法，突显了其鲁棒性和优越的可迁移性。TZhoF0K和组合（SugarCrepe）数据集。分数是Recall $@ 1$。

<table><tr><td rowspan="3">Models</td><td rowspan="3">#Parameters</td><td colspan="4">Short Caption</td><td colspan="4">Long Caption</td><td colspan="3">Compositional</td></tr><tr><td colspan="2">Flickr30K</td><td colspan="2">COCO</td><td colspan="2">ShareGPT4V</td><td colspan="2">Urban1K</td><td colspan="3">SugarCrepe</td></tr><tr><td>→ ci</td><td>→ ct qi</td><td>→ ci</td><td>q i → ct</td><td>qt → ci</td><td>qi → ct</td><td>qt → ci</td><td>→ ct g2</td><td>Replace</td><td>Swap</td><td>Add</td></tr><tr><td>OpenCLIP (ViT-L) (Radford et al. 2021)</td><td>0.4B</td><td>67.3</td><td>87.2</td><td>37.0</td><td>58.1</td><td>81.8</td><td>84.0</td><td>47.0</td><td>47.0</td><td>79.5</td><td>62.7</td><td>74.9</td></tr><tr><td>CLIP (ViT-BigG/14) (Cherti et al. 2022)</td><td>2.5B</td><td>79.5</td><td>92.9</td><td>51.3</td><td>67.3</td><td>90.1</td><td>93.6</td><td>77.8</td><td>80.7</td><td>86.5</td><td>68.9</td><td>88.4</td></tr><tr><td>EVA-CLIP (Sun et al. 2023)</td><td>8B</td><td>80.3</td><td>94.5</td><td>52.0</td><td>70.1</td><td>93.1</td><td>91.2</td><td>80.4</td><td>77.8</td><td>85.9</td><td>70.3</td><td>86.7</td></tr><tr><td>E5-V (Phi3.5-V) (Jiang et al. 2024)</td><td>4.2B</td><td>72.2</td><td>79.6</td><td>44.7</td><td>53.4</td><td>86.0</td><td>88.5</td><td>83.8</td><td>83.6</td><td>88.2</td><td>66.6</td><td>75.3</td></tr><tr><td>E5-V (LLaVA-1.6) (Jiang et al. 2024)</td><td>7B</td><td>77.3</td><td>85.7</td><td>49.1</td><td>57.6</td><td>85.1</td><td>82.1</td><td>88.9</td><td>83.2</td><td>86.3</td><td>68.7</td><td>66.9</td></tr><tr><td>VLM2Vec (Qwen2-VL) (Jiang et al. 2025)</td><td>2B</td><td>69.3</td><td>89.6</td><td>40.0</td><td>62.5</td><td>78.1</td><td>88.2</td><td>78.7</td><td>83.9</td><td>67.2</td><td>46.5</td><td>66.4</td></tr><tr><td>VLM2Vec (Qwen2-VL) (Jiang et al. 2025)</td><td>7B</td><td>80.0</td><td>94.2</td><td>49.2</td><td>68.5</td><td>78.5</td><td>90.4</td><td>94.0</td><td>94.2</td><td>70.0</td><td>51.7</td><td>72.2</td></tr><tr><td>UniME (Qwen2-VL) (Gu et al. 2025a)</td><td>2B</td><td>74.9</td><td>90.6</td><td>44.0</td><td>63.5</td><td>83.6</td><td>88.6</td><td>83.3</td><td>83.2</td><td>65.6</td><td>45.2</td><td>65.7</td></tr><tr><td>UniME (Qwen2-VL) (Gu et al. 2025a)</td><td>7B</td><td>80.8</td><td>92.7</td><td>50.9</td><td>69.8</td><td>86.5</td><td>93.8</td><td>95.3</td><td>94.0</td><td>68.8</td><td>53.0</td><td>69.8</td></tr><tr><td>UniME (LLaVA-OV) (Gu et al. 2025a)</td><td>7B</td><td>83.3</td><td>94.4</td><td>54.8</td><td>74.0</td><td>93.9</td><td>89.3</td><td>94.3</td><td>95.5</td><td>80.5</td><td>65.5</td><td>82.2</td></tr><tr><td>UniME-V2 (Qwen2-VL)</td><td>2B</td><td>79.84.9.90..7(.)65.11.91.8.)94..)95.61.)9.(+9.70.9(+5.1.2(6.)0.2+.)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UniME-V2 (Qwen2-VL)</td><td>7B</td><td colspan="9">8.+3.8)93.5(0.)57.3(+6.)0.3(+0.594.3(0.)95.197.2(196.3+2.7.8(+9.0)62.2(9.79.0+.)</td><td></td></tr><tr><td>UniME-V2 (LLaVA-OV)</td><td>7B</td><td colspan="9">8.5(+2.93.70.760.9(+6.174.1(+0.195.1(+1.)94.1(+4.96.3(+2.96.7(+1.)88.6(+8.173.7(8.2)90.5(+8.3)</td></tr></table>

![](images/4.jpg)  

Figure 4: Comparison of representation distributions between EVA-CLIP-8B and UniME-V2 (LLaVA-OneVision-7B).

短标题与长标题跨模态检索。我们在零样本跨模态检索任务中评估了 UniME-V2。在短标题数据集上，包括 Flickr30K 和 MS-COCO，UniME-V2 在图像到文本的检索中相较于 UniME 提升了 $2.2\% - 9.7\%$ 的性能。在文本到图像的检索中，其性能与 UniME 相当，主要由于两个因素：(1) MMEB 训练集中文本到图像数据的比例有限，(2) 短标题中语义信息不足。在长标题跨模态检索任务中，UniME-V2 在 ShareGPT4V 和 Urban1K 上取得了显著提升，这得益于其增强的区分能力和详细标题所提供的丰富语义内容。值得注意的是，与 EVA-CLIP-8B 相比，UniME-V2 展现了更强的检索性能。这主要得益于其通用多模态嵌入能够显著减小模态间的差距（如图 4 所示）。

Table 3: Comparison of reranking performance between LamRA and UniME-V2-Reranker using UniME-V2 (Qwen2- VL-7B) and UniME-V2 (Qwen2-VL-2B).   

<table><tr><td>Embedding Model</td><td>Reranker</td><td>#Data</td><td>MMEB</td><td>RShort</td><td>RLong</td><td>RCompos</td></tr><tr><td>UniME(2B)</td><td></td><td></td><td>60.6</td><td>68.3</td><td>84.7</td><td>58.8</td></tr><tr><td>UniME-V2(2B)</td><td></td><td></td><td>63.6</td><td>72.1</td><td>93.4</td><td>64.1</td></tr><tr><td>UniME-V2(2B)</td><td>LamRA(7B)</td><td>1.1M</td><td>67.3</td><td>76.4</td><td>96.4</td><td>87.4</td></tr><tr><td>UniME-V2(2B)</td><td>UniME-V2(7B)</td><td>0.6M</td><td>67.6</td><td>76.4</td><td>96.9</td><td>94.8</td></tr><tr><td>UniME(7B)</td><td></td><td></td><td>67.4</td><td>73.6</td><td>92.4</td><td>63.9</td></tr><tr><td>UniME-V2(7B)</td><td></td><td></td><td>68.0</td><td>76.4</td><td>95.8</td><td>73.0</td></tr><tr><td>UniME-V2(7B)</td><td>LamRA(7B)</td><td>1.1M</td><td>69.1</td><td>78.3</td><td>97.2</td><td>87.4</td></tr><tr><td>UniME-V2(7B)</td><td>UniME-V2(7B)</td><td>0.6M</td><td>69.6</td><td>78.7</td><td>97.5</td><td>94.8</td></tr></table>

组合跨模态检索。我们评估了UniME-V2模型区分困难负样本的能力，使用的基准是组合基准SugarCrepe。如表2所示，UniME-V2在所有评估指标上均持续提供优越的性能。与UniME相比，我们的模型在使用Qwen2-VL-2B时实现了$5.3\%$、$6.0\%$、$4.5\%$的性能提升。在将模型从2B扩展到7B后，我们的模型也实现了$9.0\%$、$9.2\%$和$9.2\%$的性能提升。此外，与EVA-CLIP-8B相比，UniME-V2在性能上分别提高了$2.7\%$、$3.4\%$和$3.8\%$，突显了其在区分困难负样本方面的强大能力。

重排比较。在表3中，我们比较了LamRA与UniME-V2-Reranker在前5条检索结果上使用列表重排的性能。为了确保公平，我们使用与LamRA相同的训练参数和基础模型（Qwen2.5-VL-7B）。当使用UniME-V2（Qwen2-VL-2B）进行检索时，LamRA和UniME-V2-Reranker在四个下游任务中均提升了性能，UniME-V2-Reranker始终取得了更优的结果，而只使用了一半的数据。同样，当使用UniME-V2（Qwen2-VL-7B）进行检索时，UniME-V2-Reranker超过了LamRA，在四个任务中的性能提升分别为$0.5\%$、$0.4\%$、$0.3\%$和$7.4\%$。值得注意的是，

Table 4: Ablation study on our proposed MLLM-as-a-Judge hard negatives mining method and MLLM judgment based training framework.   

<table><tr><td>Hard Negatives</td><td>Soft Score</td><td>MMEB</td><td>RShort</td><td>RLong</td><td>RCompos</td></tr><tr><td>x</td><td>×</td><td>60.1</td><td>63.4</td><td>82.2</td><td>60.0</td></tr><tr><td></td><td></td><td>61.6</td><td>68.9</td><td>89.8</td><td>63.7</td></tr><tr><td>v</td><td>v</td><td>63.6</td><td>72.1</td><td>93.4</td><td>64.1</td></tr></table>

Table 5: Ablation study on different MLLM-based judges.   

<table><tr><td>Judge Model</td><td>MMEB</td><td>RShort</td><td>RLong</td><td>RCompos</td></tr><tr><td>Qwen2.5VL-7B</td><td>63.6</td><td>72.1</td><td>93.4</td><td>64.1</td></tr><tr><td>InternVL3-8B</td><td>58.5</td><td>70.2</td><td>91.3</td><td>64.1</td></tr><tr><td>InternVL3-14B</td><td>63.2</td><td>72.9</td><td>93.1</td><td>63.2</td></tr></table>

<table><tr><td>#Negatives</td><td>MMEB</td><td>RShort</td><td>RLong</td><td>RCompos</td></tr><tr><td>4</td><td>61.3</td><td>69.2</td><td>91.0</td><td>62.4</td></tr><tr><td>6</td><td>61.8</td><td>70.8</td><td>91.7</td><td>61.2</td></tr><tr><td>8</td><td>63.6</td><td>72.1</td><td>93.4</td><td>64.1</td></tr><tr><td>10</td><td>63.0</td><td>72.0</td><td>93.4</td><td>63.4</td></tr></table>

Table 6: Ablation study on the number of hard negatives.

UniME-V2-Reranker 在组合理解检索任务中相较于 LamRA 展示了显著的优势，这归因于其利用 MLLM 的理解能力提取多样且高质量的困难样本，从而有效提升了模型的区分能力。

# 分析

不同组件的消融实验。我们通过对提出的基于MLLM判断的硬负样本挖掘方法和MLLM判断基础训练框架进行消融研究，以评估UniME-V2的有效性，使用Qwen2-VL-2B。正如表4所示，我们提出的硬负样本挖掘方法在MMEB、短检索、长检索和复合检索任务上，相较于直接对比学习（例如VLM2Vec）分别实现了$1 . 5 \%$、$5 . 5 \%$、$7 . 6 \%$和$3 . 7 \%$的性能提升。在此基础上，MLLM判断基础训练框架的引入进一步提升了模型的区分能力，通过捕捉候选样本间更细的语义差异，导致对应任务的额外性能增益分别为$2 . 0 \%$、$3 . 2 \%$、$3 . 6 \%$和$0 . 4 \%$。

在不同的基于大语言模型的判断者的消融研究中，作为判断者的MLLM的理解能力直接影响生成的语义匹配分数的准确性，从而影响最终模型的性能。因此，我们基于Qwen2-VL-2B，比较当前开源社区中两个具有影响力的MLLM：Qwen2.5-VL-7B、InternVL3-8B和InternVL3-14B。如表5所示，在相同的推理设置下，Qwen2.5-VL生成的语义匹配分数质量显著优于InternVL3-8B，尤其在MMEB上（63.6对58.5）。使用InternVL3-14B时，与InternVL3-8B相比，下游性能显著提升，但仍略低于Qwen2.5-7B。其主要原因可以归因于在它们的SFT阶段使用的指令数据分布的差异。

![](images/5.jpg)  

Figure 5: Qualitative examples. We present the retrieval and reranking results of our method across different tasks.

对困难负样本数量的消融实验。表6展示了基于Qwen2-VL-2B的困难负样本数量变化的影响。当困难负样本数量从4增加到8时，UniME-V2在所有评估指标上均表现出一致的提升：MMEB提高了$+ 2 . 3 \%$，短检索提高了$+ 2 . 9 \%$，长检索提高了$+ 2 . 4 \%$，组合检索提高了$+ 1 . 7 \%$。这些提升可以归因于模型在训练过程中增强了对候选样本的区分能力。然而，进一步增加到10时引入了较简单的负样本，降低了区分学习能力，并略微降低了性能。定性结果。图5展示了我们方法在各项任务中的定性结果。显示了UniME-V2的检索结果，UniME-V2-Reranker优化后的top-1候选项用红色虚线框突出显示。UniME-V2有效地检索到与查询相关的候选项，例如第一个例子中的“黑熊”和“棕熊”，而UniME-V2-Reranker进一步优化了检索结果的排名，优先考虑“棕熊”而不是“黑熊”。

# 结论

在本文中，我们探讨如何利用大语言模型（MLLMs）先进的理解能力来增强表征学习，并提出了一种新颖的通用多模态嵌入模型（UniME-V2）。具体而言，我们首先构建一个潜在的硬负样本集，采用全局检索。接着，我们引入MLLM-as-a-Judge，该方法利用MLLMs的强语义理解能力评估查询-候选对的对齐情况，并生成软语义匹配分数。这些分数通过降低假阴性干扰并识别高质量、多样化的硬负样本来指导硬负样本挖掘。此外，这些分数作为软标签，放宽了严格的一对一映射约束。通过将相似度矩阵与软语义匹配分数矩阵对齐，模型学习候选者之间更细粒度的语义区分，从而增强其区分能力。为了进一步提升性能，我们提出了UniME-V2-Reranker，该方法结合基于挖掘出的硬负样本的联合成对和列表重排序优化。我们在MMEB基准测试和各种检索任务上进行了广泛实验，我们的方法在所有任务中均实现了最先进的平均性能。我们希望我们的研究能够为通用多模态表征学习提供启示。

# References

Aminabadi, R. Y.; Rajbhandari, S.; Awan, A. A.; Li, C.; Li, D.; Zheng, E.; Ruwase, O.; Smith, S.; Zhang, M.; Rasley, J.; et al. 2022. Deepspeed-inference: enabling efficient inference of transformer models at unprecedented scale. In SC22: International Conference for High Performance Computing, Networking, Storage and Analysis, 115. IEEE.   
An, R.; Yang, S.; Lu, M.; Zhang, R.; Zeng, K.; Luo, Y.; Cao, J.; Liang, H.; Chen, Y.; She, Q.; et al. 2024. Mc-llava: Multiconcept personalized vision-language model. arXiv preprint arXiv:2411.11706.   
R. S Zh R he Z. L M.Dai  , H.Guo, Z.; Yan, S.; Luo, Y.; et al. 2025a.UniCTokens: Boosting Personalized Understanding and Generation via Unified Concept Tokens. arXiv preprint arXiv:2505.14671. An, X.; Xie, Y.; Yang, K.; Zhang, W.; Zhao, X.; Cheng, Z.; Wang, Y.; Xu, S.; Chen, C.; Wu, C.; et al. 2025b. LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training. arXiv preprint arXiv:2509.23661. Andonian, A.; Chen, S.; and Hamid, R. 2022. Robust cross-modal representation learning with progressive selfdistillation. In CVPR.   
Bi S.; hen, K. Liu, X.; Wang, J. Ge, .Song S. a, K.; Wang, P.; Wang, S.; Tang, J.; et al. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.   
BehnamGhader, P.; Adlakha, V.; Mosbach, M.; Bahdanau, D.; Chapados, N.; and Reddy, S. 2024. Llm2vec: Large language models are secretly powerful text encoders. COLM.   
Cao, A.; Wei, X.; and Ma, Z. 2024. FLAME: Frozen Large Language Models Enable Data-Efficient Language-Image Pre-training. arXiv:2411.11927.   
Changpinyo, S.; Sharma, P.; Ding, N.; and Soricut, R. 2021. Conceptual $1 2 \mathrm { m }$ : Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In CVPR, 35583568. Chen, D.; Chen, R.; Zhang, S.; Wang, Y.; Liu, Y.; Zhou, H.; Zhang, Q.; Wan, Y.; Zhou, P.; and Sun, L. 2024a. Mllm-asa-judge: Assessing multimodal llm-as-a-judge with visionlanguage benchmark. In ICML.   
Chen, L.; Li, J.; Dong, X.; Zhang, P.; He, C.; Wang, J.; Zhao, F.; and Lin, D. 2024b. Sharegpt4v: Improving large multimodal models with better captions. In ECCV.   
Cherti, M.; Beaumont, R.; Wightman, R.; Wortsman, M.; Ilharco, G.; Gordon, C.; Schuhmann, C.; Schmidt, L.; and Jitsev, J. 2022. Reproducible scaling laws for contrastive language-image learning. arXiv:2212.07143.   
Dong, G.; Song, X.; Zhu, Y.; Qiao, R.; Dou, Z.; and Wen, J.-R. 205. Towar general instruction-following alignment for retrieval-augmented generation. In AAAI. G LZha Y.; Han, J.; anCallan, J.201.Sci deep contrastive learning batch size under memory limited setup. arXiv preprint arXiv:2101.06983.   
TYag,K.An, X.Fe Z. Liu, D. Cai W, J. 2024. Rwkv-clip: A robust vision-language representation learner. In EMNLP.   
Gu, T.; Yang, K.; Feng, Z.; Wang, X.; Zhang, Y.; Long, D.; Chen, Y.; Cai, W.; and Deng, J. 2025a. Breakig the Modality Barrier: Universal Embedding Learning with Multimodal LLMs. In ACM MM.   
Gu T.; Yang, K.; Zhang, C.; Xie, Y.; An, X.; Feng, Z.; Liu, D.; Cai, W.; and Deng, J. 2025b. RealSyn: An Effective and Scalable Multimodal Interleaved Document Transformation Paradigm. In ACM MM.   
Hamza, A.; Ahn, Y. H.; Lee, S.; Kim, S. T.; et al. 2025. Llava needs more knowledge: Retrieval augmented natural language generation with knowledge graph for explaining thoracic pathologies. In AAAI.   
Hsieh, C.-Y.; Zhang, J.; Ma, Z.; Kembhavi, A.; and Krishna, R. 2023. Sugarcrepe: Fixing hackable benchmarks for visionlanguage compositionality. NeurIPS.   
Hu, X.; Yang, K.; Wang, J.; Xu, H.; Feng, Z.; and Wang, Y. 2025. Decoupled Global-Local Alignment for Improving Compositional Understanding. In ACM MM.   
Huang, H.; Nie, Z.; Wang, Z.; and Shang, Z. 2024a. Crossmodal and uni-modal soft-label alignment for image-text retrieval. In AAAI.   
Huang, W.; Wu, A.; Yang, Y.; Luo, X.; Yang, Y.; Hu, L.; Dai, Q.; Dai, X.; Chen, D.; Luo, C.; et al. 2024b. Llm2clip: Powerful language model unlock richer visual representation. arXiv:2411.04997.   
Jiang, T.; Song, M.; Zhang, Z.; Huang, H.; Deng, W.; Sun, F.; Zhang, Q.; Wang, D.; and Zhuang, F. 2024. E5-v: Universal embeddings with multimodal large language models. arXiv:2407.12580.   
Jiang, Z.; Meng, R.; Yang, X.; Yavuz, S.; Zhou, Y.; and Chen, W. 2025. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. ICLR.   
Lan, Z.; Niu, L.; Meng, F.; Zhou, J.; and Su, J. 2025. Llave: Large language and vision embedding models with hardness-weighted contrastive learning. arXiv preprint arXiv:2503.04812.   
Lee, C.; Roy, R.; Xu, M.; Raiman, J.; Shoeybi, M.; Catanzaro, B.; and Ping, W. 2024. Nv-embed: Improved techniques for training llms as generalist embedding models. ICLR.   
Lei, Y.; He, F.; Chen, C.; Mo, Y.; Li, S. J.; Xie, D.; and Lu, H. 2024. MCAD: Multi-teacher Cross-modal Alignment Distillation for efficient image-text retrieval. In NAACL. Li B.; Zhang Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; et al. 2024. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.   
Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML. L Y Cao,Y.; He, H; he, Q.; Fu, X. Xiao, X T.; and Tang, R. 2025. M2IV: Towards Efficient and Finegrained Multimodal In-Context Learning via Representation Engineering. In Second Conference on Language Modeling. Lin, J.; Yin, H.; Ping, W.; Molchanov, P.; Shoeybi, M.; and Han, S. 0a. Vil: On pre-raii r visual ange models. In CVPR.   
Lin, S.-C.; Lee, C.; Shoeybi, M.; Lin, J.; Catanzaro, B.; and Ping, W. 2024b. Mm-embed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571. Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Dollár, P.; and Zitnick, C. L. 2014. Microsoft coco: Common objects in context. In ECCV.   
Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual instruction tuning. NeurIPS.   
Y.Ch  Cai, J.; J X.Hu . ao, J. Y.; and Xie, W. 2024. LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant. CVPR.   
Muennighoff, N.; Tazi, N.; Magne, L.; and Reimers, N. 2022. MTEB: Massive Text Embedding Benchmark. arXiv:2210.07316.   
Nielsen, F. 2020. On a generalization of the JensenShannon divergence and the JensenShannon centroid. Entropy. Peng, Z.; Wang, W.; Dong, L.; Hao, Y.; Huang, S.; Ma, S.; and Wei, F. 2023. Kosmos-2: Grounding multimodal large language models to the world. arXiv:2306.14824.   
Plummer, B. A.; Wang, L.; Cervantes, C. M.; Caicedo, J. C.; Hockenmaier, J.; and Lazebnik, S. 2015. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In ICCV.   
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In ICML.   
Schuhmann, C.; Vencu, R.; Beaumont, R.; Kaczmarczyk, R.Mullis, C.;KattaA.Coobes, T.; Jitsev, J.;and Komatsuzaki, A. 2021. Laion- $4 0 0 \mathrm { m }$ : Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114.   
Sun, Q.; Wang, J.; Yu, Q.; Cui, Y.; Zhang, F.; Zhang, X.; and Wang, X. 2023. EVA-CLIP-18B: Scaling CLIP to 18 Billion Parameters. arXiv:2402.04252.   
Tan, F.; Huang, Z.; Liu, C.; Sun, Q.; Yang, H.; and Lim, S.-N.0. Inteveni nhor toke Decdi at n alleviating hallucinations for MLLMs. In ICLR.   
Tang, F.; Liu, C.; Xu, Z.; Hu, M.; Huang, Z.; Xue, H.; Chen, Z.; Peng, Z.; Yang, Z.; Zhou, S.; et al. 2025b. Seeing Far and Clearly: Mitigating Hallucinations in MLLMs with Attention Causal Decoding. In CVPR, 2614726159.   
Tn, M.; Kuar, M.; Steiner, A.; Zhai, X.; Houly, N.; and Beyer, L. 2023. Image captioners are scalable vision learners too. NeurIPS.   
W H.; Li, L.; Qu, C.; Zhu, F; Xu, W.; Chu W.; ad Lin, F. 2025a. To code or not to code? adaptive tool integration for math language models via expectation-maximization. arXiv preprint arXiv:2502.00691. Wang, H.; Qu, C.; Huang, Z.; Chu, W.; Lin, F.; and Chen, W. 2025b. V1-rethinker: Incentivizing self-reflection of visionlanguage models with reinforcement learning. arXiv preprint arXiv:2504.08837.   
Wang, P.; Bai, S.; Tan, S.; Wang, S.; Fan, Z.; Bai, J.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; et al. 2024. Qwen2-vl: Enhancig ision-nguge mode's percetion  he world t ay resolution. arXiv preprint arXiv:2409.12191.   
Wan, W.; Lv, Q.; Yu, W.; Ho, W.; Qi, J.; Wa, Y.; Ji, J.; Yang, Z.; Zhao, L.; Song, X.; Xu, J.; Xu, B.; Li, J.; Dong, Y.; Ding, M.; and Tang, J. 2023. CogVLM: Visual Expert for Pretrained Language Models. arXiv:2311.03079.   
Xi Y.; Yag, K.;Yag, N.; Deg W.; ai, X.; Gu, T.; Wa, YAn, X.ZY.Fe .. .C large multimodal models with cross-modal comprehension. arXiv preprint arXiv:2410.14332.   
Xue, Y.; Li, D.; and Liu, G. 2025a. Improve Multi-Modal Embedding Learning via Explicit Hard Negative Gradient Amplifying. arXiv preprint arXiv:2506.02020.   
Xue, Y.; Li, D.; and Liu, G. 2025b. Improve Multi-Modal Embedding Learning via Explicit Hard Negative Gradient Amplifying. arXiv preprint arXiv:2506.02020.   
Yang, K.; Gu, T.; An, X.; Jiang, H.; Dai, X.; Feng, Z.; Cai, W.; and Deng, J. 2025. Clip-cid: Efficient clip distillation via cluster-instance discrimination. In AAAI, volume 39, 2197421982.   
Yuksekgonul, M.; Bianchi, F.; Kalluri, P.; Jurafsky, D.; and Zou, J. 2022. When and why vision-language models behave like bags-of-words, and what to do about it? arXiv:2210.01936.   
Zhai, X.; Mustafa, B.; Kolesnikov, A.; and Beyer, L. 2023. Sigmoid loss for language image pre-training. In ICCV. Zhang, B.; Zhang, P.; Dong, X.; Zang, Y.; and Wang, J. 2024a. Long-clip: Unlocking the long-text capability of clip. In ECCV.   
Zhang, K.; Luan, Y.; Hu, H.; Lee, K.; Qiao, S.; Chen, W.; Su, Y.; and Chang, M.-W. 2024b. Magiclens: Selfsupervised image retrieval with open-ended instructions. arXiv:2403.19651.   
Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E.; et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. NeuriPS, 36: 4659546623.   
Zh T; Zhg, Y.; An, X.; Feng, Z.; Yag, K.; an Dig, Q. 2025. Gradient-Attention Guided Dual-Masking Synergetic Framework for Robust Text-based Person Retrieval. In EMNLP.

This supplementary material elaborates on our experimental setup, covering training configurations, instruction prompts for UniME-V2-Reranker and evaluation benchmarks for retrieval tasks. It also includes extended results such as an ablation study on temperature and a detailed performance analysis on the MMEB benchmark. Additionally, we provide supplementary visualizations of training data samples, retrieval outputs, and reranking results.

# Detail Experiment Setting

# Training Details

We provide the training configurations of UniME-V2 in Tab. 7 and UniME-V2-Reranker in Tab. 8.

UniME-V2: Following UniME (Gu et al. 2025a), we adopt identical experimental settings for training UniME-V2. We configure LoRA rank as 16 and employ GradCache (Gao et al. 2021) for efficient training over 2,000 steps using $8 \times \mathbf { A } 8 0 0$ GPUs (80GB memory each). The learning rate is set to $1 \times 1 0 ^ { - 4 }$ for the Qwen series and $2 \times 1 0 ^ { - 5 }$ for LLaVA-OneVision. Due to memory constraints, the input resolution is fixed at 336 for LLaVA-OneVision and 672 for the Qwen series.

Table 7: Training hyperparameters and computational requirements for UniME-V2 (Qwen2-VL-2B/7B) and UniME-V2 (LLaVA-OneVision-7B).   

<table><tr><td>Hyperparameter</td><td>Qwen2-VL-2B/7B</td><td>LLaVA-OV-7B</td></tr><tr><td>Training samples</td><td>662K</td><td></td></tr><tr><td>Batch size</td><td>1024</td><td></td></tr><tr><td>Learning rate</td><td>1×10−4</td><td>2×10−5</td></tr><tr><td>LoRA rank</td><td>16</td><td></td></tr><tr><td>Training steps</td><td>2000</td><td></td></tr><tr><td>Optimizer</td><td>AdamW</td><td></td></tr><tr><td>Infra</td><td>GradCache</td><td></td></tr><tr><td>Max length</td><td>4096</td><td></td></tr><tr><td>temperature</td><td>0.02</td><td></td></tr><tr><td>#Hard Negatives</td><td>8</td><td></td></tr><tr><td>Image Resolution</td><td>672</td><td>336</td></tr><tr><td>Precision</td><td>BF16</td><td></td></tr><tr><td>GPU configuration</td><td>8×A800</td><td></td></tr><tr><td>Random Seed</td><td>42</td><td></td></tr></table>

UniME-V2-Reranker: Following LamRA's experimental setup (Liu et al. 2024), we adopt Qwen2.5-VL-7B as the backbone for UniME-V2-Reranker. The model is trained using LoRA with a rank of 128 for 1 epoch almost 2,000 steps. All experiments are conducted using the lmms-finetune infrastructure with a maximum sequence length of 4096 tokens.

# Detail Instruction Prompt for UniME-V2-Reranker

The prompt template employed for pairwise training of UniME-V2-Reranker is presented below:

Table 8: Training hyperparameters and computational requirements for UniME-V2-Reranker (Qwen2.5-VL-7B).   

<table><tr><td>Hyperparameter</td><td>Qwen2.5-VL-7B</td></tr><tr><td>Training samples</td><td>662K</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>2×10-5</td></tr><tr><td>LoRA rank</td><td>128</td></tr><tr><td>Training epochs</td><td>1</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Infra</td><td>lmms-finetune</td></tr><tr><td>Max length</td><td>4096</td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>DeepSpeed Stage</td><td>2</td></tr><tr><td>GPU configuration</td><td>8×A800</td></tr><tr><td>Random Seed</td><td>42</td></tr></table>

I will provide you with a query and a candidate. Please evaluate whether the candidate meets the requirements of the query. If it does, respond with 'Yes'; if it doesn't, respond with 'No'. uery: $<$ Query>, Candidate: $<$ Candidate>.

The prompt used for listwise training of UniME-V2- Reranker is shown below:

I will provide you with a query followed by multiple candidates in the format: (1) cand1 (2) cand2, etc. Each candidate is independent of the others. Evaluate each candidate against the query, and respond with the number corresponding to the candidate that best meets the requirements of the query. Query: $<$ Query>, Candidates: $<$ Candidate list>.

# Retrieval Task Evaluation Benchmarks

We evaluate UniME-V2 and UniME-V2-Reranker on diverse retrieval benchmarks, including short-caption, long-caption, and compositional image-text tasks (Tab. 9). For each benchmark, we follow the standard evaluation protocol. In retrieval tasks, we primarily report Recall $@ 1$ as the evaluation metric, using the prompt "Represent the image/text" for both image and text instructions.

<table><tr><td>Benchmark</td><td>Zero-shot #Queries #Cands</td><td></td><td></td></tr><tr><td>Flickr30K (Plummer et al. 2015)</td><td></td><td>1K</td><td>5K</td></tr><tr><td>COCO (Lin et al. 2014)</td><td>E</td><td>5K</td><td>25K</td></tr><tr><td>ShareGPT4V (Chen et al. 2024b)</td><td></td><td>1K</td><td>1K</td></tr><tr><td>Urban1K (Zhang et al. 2024a)</td><td>:</td><td>1K</td><td>1K</td></tr><tr><td>SugarCrepe (Hsieh et al. 2023)</td><td></td><td>7.5K</td><td>2</td></tr></table>

Table 9: Summary of the evaluation benchmarks. # Queries represents the number of test queries, and # Cands denotes the number of test candidates per query.

<table><tr><td>Temperature</td><td>MMEB</td><td>RShort</td><td>RLong</td><td>RCompos</td></tr><tr><td>0.03</td><td>61.9</td><td>70.6</td><td>92.0</td><td>65.8</td></tr><tr><td>0.02</td><td>63.6</td><td>72.1</td><td>93.4</td><td>64.1</td></tr><tr><td>0.01</td><td>62.1</td><td>70.1</td><td>91.1</td><td>66.5</td></tr></table>

Table 10: Ablation study on the temperature. We report the mean scores on the MMEB benchmark, short and long crossmodal retrieval, as well as compositional cross-modal retrieval.

![](images/6.jpg)  
Figure 6: Qualitative examples. We present examples showing queries and their corresponding hard negative candidates processed after our hard negative mining pipeline.

# External Results

# Ablation on the Temperature

We conduct additional experiments with UniME-V2 (Qwen2- VL-2B) to analyze the impact of temperature in the final loss function. As evidenced by Tab. 10, a temperature value of 0.02 yields optimal performance across all evaluation metrics, including MMEB, short & long retrieval, and compositional retrieval tasks.

# Specific Results on the MMEB Benchmark

Tab. 11 presents comprehensive results on the MMEB benchmark across eight models: BLIP2, MagicLens, EVA-CLIP, E5-V, VLM2Vec, UniME, UniME-V2, and UniME- $\mathbf { V } 2 ^ { \dagger }$ .Results for BLIP2 through UniME are reproduced directly from UniME (Gu et al. 2025a). Our implementation details specify that UniME-V2 employs the Qwen2-VL-7B backbone, while UniME-V2† represents uses LLaVA-OneVision-7B as its backbone.

# Further Analysis

# Visualization of the Training Data

Fig. 6 presents training examples from the MMEB dataset annotated with their semantic matching scores, which obtained after our proposed pipeline. The visualization demonstrates: (1) target candidates achieving the highest match scores (nearly 1.0), (2) partially relevant candidates with intermediate scores (between 0.0 and 1.0), and (3) irrelevant candidates receiving near-zero scores.

![](images/7.jpg)  
Figure 7: Qualitative examples. We present the additional retrieval and reranking results of our method across different tasks.

# Visualization of the Retrieval and Rerank Results

Fig. 7 presents additional qualitative results demonstrating our method's performance across multiple tasks. The visualization reveals that while UniME-V2 successfully retrieves query-matched candidates, UniME-V2-Reranker further refines these results by selecting the optimally matched candidate as the final output.

<table><tr><td></td><td>BLIP2</td><td>MagicLens</td><td>EVA-CLIP</td><td>E5-V</td><td>VLM2Vec</td><td>UniME</td><td>UniME-V2</td><td>UniME-V2†</td></tr><tr><td colspan="9">Classification (10 tasks)</td></tr><tr><td>ImageNet-1K</td><td>10.3</td><td>48.0</td><td>75.0</td><td>40.5</td><td>66.5</td><td>71.3</td><td>80.3</td><td>78.8</td></tr><tr><td> 24News</td><td>36.0</td><td>33.7</td><td>33.8</td><td>31.5</td><td>76.4</td><td>79.5</td><td>66.9</td><td>66.6</td></tr><tr><td>HatefulMemes</td><td>49.6</td><td>49.0</td><td>49.3</td><td>49.3</td><td>60.9</td><td>64.6</td><td>65.9</td><td>65.3</td></tr><tr><td>VOC2007</td><td>52.1</td><td>51.6</td><td>44.3</td><td>76.7</td><td>84.0</td><td>90.4</td><td>84.9</td><td>92.0</td></tr><tr><td>SUN397</td><td>34.5</td><td>57.0</td><td>62.7</td><td>52.3</td><td>73.2</td><td>75.9</td><td>78.9</td><td>78.7</td></tr><tr><td>Place365</td><td>21.5</td><td>31.5</td><td>38.7</td><td>32.0</td><td>42.1</td><td>45.6</td><td>42.5</td><td>42.9</td></tr><tr><td> mageNet-A</td><td>3.2</td><td>8.0</td><td>54.8</td><td>18.2</td><td>39.9</td><td>45.5</td><td>53.7</td><td>48.0</td></tr><tr><td> mageNet-R</td><td>39.7</td><td>70.9</td><td>95.4</td><td>56.7</td><td>746</td><td>78.4</td><td>87.9</td><td>89.3</td></tr><tr><td> ObjectNet</td><td>20.6</td><td>31.6</td><td>67.8</td><td>34.2</td><td>34.3</td><td>36.4</td><td>35.0</td><td>73.1</td></tr><tr><td> Cuntry-211</td><td>2.5</td><td>6.2</td><td>38.7</td><td>5.9</td><td>16.1</td><td>18.7</td><td>32.3</td><td>19.8</td></tr><tr><td>All Classification</td><td>27.0</td><td>38.8</td><td>56.0</td><td>39.7</td><td>56.8</td><td>60.6</td><td>64.0</td><td>65.3</td></tr><tr><td colspan="9">VQA (10 tasks)</td></tr><tr><td>OK-VQA</td><td>8.7</td><td>12.7</td><td>9.9</td><td>15.1</td><td>66.5</td><td>68.3</td><td>59.3</td><td>71.9</td></tr><tr><td> -OKQA</td><td>3.2</td><td>2.9</td><td>2.8</td><td>4.7</td><td>54.9</td><td>58.7</td><td>32.3</td><td>71.4</td></tr><tr><td>T DocVGQA</td><td>2.6</td><td>3.0</td><td>7.4</td><td>9.1</td><td>64.4</td><td>6.6</td><td>91.2</td><td>92.6</td></tr><tr><td>InfographicsVQA</td><td>2.0</td><td>5.9</td><td>6.0</td><td>8.7</td><td>34.8</td><td>37.0</td><td>63.9</td><td>63.5</td></tr><tr><td>ChartA</td><td>0.5</td><td>0.9</td><td>1.5</td><td>4.2</td><td>33.1</td><td>33.4</td><td>56.9</td><td>55.8</td></tr><tr><td> Visual7w</td><td>1.3</td><td>2.5</td><td>2.2</td><td>4.5</td><td>49.8</td><td>51.7</td><td>600.1</td><td>62.5</td></tr><tr><td>SScienceQA</td><td>6.8</td><td>5.2</td><td>14.1</td><td>9.6</td><td>37.3</td><td>40.5</td><td>44.5</td><td>54.0</td></tr><tr><td> ViizWiz</td><td>4.0</td><td>1.7</td><td>4.3</td><td>8.6</td><td>39.9</td><td>42.7</td><td>47.4</td><td>53.7</td></tr><tr><td>NAIQAL GQA</td><td>9.7</td><td>43.5</td><td>44.7</td><td>34.1</td><td>557.3</td><td>63.6</td><td>55.8</td><td>669.5</td></tr><tr><td>T TextiVA</td><td>3.3</td><td>4.6</td><td>10.8</td><td>9.5</td><td>65.7</td><td>65.2</td><td>78.4</td><td>84.5</td></tr><tr><td>A A VGA</td><td>4.2</td><td>8.3</td><td>10.4</td><td>10.8</td><td>50.4</td><td>52.9</td><td>60.1</td><td>67.6</td></tr><tr><td colspan="9">Retrieval (12 tasks)</td></tr><tr><td>VisDial</td><td>18.0</td><td>24.8</td><td>20.4</td><td>57.6</td><td>75.3</td><td>79.7</td><td>83.4</td><td>84.2</td></tr><tr><td>CIRR</td><td>9.8</td><td>39.1</td><td>36.0</td><td>41.0</td><td>51.3</td><td>52.2</td><td>64.0</td><td>65.5</td></tr><tr><td>VisualNews_t2i</td><td>48.1</td><td>50.7</td><td>82.4</td><td>43.9</td><td>70.7</td><td>74.8</td><td>79.9</td><td>77.3</td></tr><tr><td>VisualNews_i2t</td><td>13.5</td><td>21.1</td><td>88.2</td><td>46.8</td><td>75.2</td><td>78.8</td><td>83.5</td><td>79.2</td></tr><tr><td>MSCOCO_t2i</td><td>53.7</td><td>54.1</td><td>65.3</td><td>68.6</td><td>69.9</td><td>74.9</td><td>77.7</td><td>79.1</td></tr><tr><td>MSCO i2t</td><td>20.3</td><td>40.0</td><td>67.2</td><td>54.8</td><td>67.7</td><td>73.8</td><td>73.0</td><td>75.2</td></tr><tr><td> NIGHTS</td><td>56.5</td><td>58.1</td><td>0.2</td><td>0.1</td><td>63.3</td><td>66.2</td><td>69.3</td><td>68.1</td></tr><tr><td>WWebQA</td><td>55.4</td><td>43.0</td><td>70.9</td><td>33.7</td><td>83.6</td><td>89.8</td><td>91.5</td><td>90.6</td></tr><tr><td>FashionIQ</td><td>9.3</td><td>11.2</td><td>16.1</td><td>11.2</td><td>15.2</td><td>16.5</td><td>28.5</td><td>26.4</td></tr><tr><td>Wi-S-NQ</td><td>28.7</td><td>18.7</td><td>46.7</td><td>61.0</td><td>63.4</td><td>66.6</td><td>688</td><td>71.2</td></tr><tr><td> VEN</td><td>39.5</td><td>1.6</td><td>1.8</td><td>0.5</td><td>49.6</td><td>5.7</td><td>71.2</td><td>68.0</td></tr><tr><td>EDIS</td><td>54.4</td><td>62.6</td><td>95.6</td><td>553.8</td><td>73.7</td><td>86.2</td><td>84.4</td><td>88.2</td></tr><tr><td>All Retrieval</td><td>33.9</td><td>35.4</td><td>49.2</td><td>39.4</td><td>63.3</td><td>67.9</td><td>73.1</td><td>72.9</td></tr><tr><td colspan="9">Visual Grounding (4 tasks)</td></tr><tr><td>MSCOCO</td><td>28.9</td><td>22.1</td><td>35.8</td><td>41.7</td><td>77.0</td><td>76.5</td><td>69.3</td><td>78.2</td></tr><tr><td>RefCoco</td><td>47.4</td><td>22.8</td><td>59.9</td><td>62.2</td><td>85.9</td><td>89.3</td><td>88.4</td><td>94.6</td></tr><tr><td>RefCoCO-matching</td><td>59.5</td><td>35.6</td><td>70.0</td><td>74.9</td><td>83.8</td><td>90.6</td><td>89.7</td><td>91.4</td></tr><tr><td>isual7W-pointing</td><td>552.0</td><td>23.4</td><td>70.2</td><td>61.8</td><td>83.6</td><td>84.1</td><td>78.7</td><td>93.8</td></tr><tr><td>All Visual Grounding</td><td>47.0</td><td>26.0</td><td>58.9</td><td>60.2</td><td>82.6</td><td>85.1</td><td>82.8</td><td>90.2</td></tr><tr><td colspan="9">Final Score (36 tasks)</td></tr><tr><td>All</td><td>28.0</td><td>27.1</td><td>43.7</td><td>37.5</td><td>63.3 64.9</td><td>66.6 68.4</td><td>68.0 72.0</td><td>71.2 74.8</td></tr><tr><td>All IND All OOD</td><td>25.3 25.1</td><td>31.0 23.7</td><td>38.1 45.6</td><td>34.2 33.4</td></table>

Ta Thesalu eul araseet rnM , VL-7B as its backbone, and UniME-V2† denotes using LLaVA-OneVision-7B as its backbone.