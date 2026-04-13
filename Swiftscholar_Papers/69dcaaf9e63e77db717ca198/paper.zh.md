# ReTrack：基于证据驱动的双流方向锚点校准网络用于组合视频检索

Zixu $\mathbf { L i } ^ { 1 }$, Yupeng $\mathbf { H } \mathbf { u } ^ { 1 * }$, Zhiwei Chen, Qinlei Huang, Guozhi $\mathbf { Q } \mathbf { i } \mathbf { u } ^ { 1 }$, Zhiheng $\mathbf { F u } ^ { 1 }$, Meng Liu 1软件学院，山东大学 2计算机科学与技术学院，山东建筑大学 {lizixu.cs, zivczw, fuzhiheng8, mengliu.sdu}@gmail.com, {hql, qiugz}@mail.sdu.edu.cn, huyupeng@sdu.edu.cn,

# 摘要

随着视频数据的快速增长，复合视频检索（Composed Video Retrieval, CVR）作为一种新兴的检索范式逐渐受到研究者的关注。与单模态视频检索方法不同，CVR任务以由参考视频和修改文本组成的多模态查询作为输入。修改文本传达了用户对参考视频的预期更改。基于这一输入，模型旨在检索最相关的目标视频。在CVR任务中，视频和文本模态之间的信息密度存在显著差异。传统的组合方法往往使得组合特征偏向于参考视频，导致检索性能不佳。这个局限性主要体现在三个核心挑战上：（1）模态贡献纠缠，（2）组合特征的显式优化，以及（3）检索的不确定性。为了解决这些挑战，我们提出了证据驱动的双流方向锚定校准网络（ReTrack）。ReTrack是第一个通过校准组合特征中的方向偏差来提升多模态查询理解的CVR框架。它由三个关键模块组成：语义贡献解耦、组合几何校准和可靠的证据驱动对齐。具体而言，ReTrack估计每个模态的语义贡献，以校准组合特征的方向偏差。然后，利用校准后的方向锚计算双向证据，推动可靠的组合与目标相似性估计。此外，ReTrack在复合图像检索（Composed Image Retrieval, CIR）任务中也表现出很强的泛化能力，在CVR和CIR场景下，在三个基准数据集上实现了最先进的性能。

# 1 引言

随着视频数据的迅速扩张（刘等，2018a；胡等，2021a；张等，2025；刘等，2018b；胡等，2023；刘等，2025a），视频检索已成为多模态检索领域的核心研究重点（孔等，2025；普等，2025a；刘等，2024b；普等，2025b；孙等，2023a；张等，2023；穆等，2025）。为了满足日益增长的灵活查询需求，Ventura等（Ventura et al. 2024b）提出了组合视频检索（Composed Video Retrieval, CVR），从而引起了显著关注（Ventura et al. 2024a；Thawakar等，2024；岳等，2025）。如图1(a)所示，与传统单模态视频检索（天等，2025b,a；魏等，2019, 2020）不同，CVR通过包含参考视频和修改文本的多模态查询，从大规模数据库中检索最相关的目标视频。作为多模态交互中的基础任务，CVR支持多种现实世界应用，如多模态推理（孙等，2023b；胡等，2021b；李等，2025a, 2023b；王、张和Dodgson，2024, 2025；Yi-fan，2016；许等，2025）和智能交互系统（马等，2025b,a；姜等，2025；欧、德布伦和舒尔茨，2025；刘等，2021a）。

![](images/1.jpg)  

Figure 1: (a) illustrates a typical CVR example. (b) highlights the directional bias issue in existing methods, where the similarity between the composed feature and the target video becomes indistinguishable from that of certain negative candidates, degrading retrieval performance. (c) demonstrates that our method effectively mitigates directional bias, producing a clear separation between the composed feature's similarity to the target and all negative samples.

然而，由于在组合特征中被忽视的方向偏差，CVR仍处于早期阶段。具体而言，视频模态通常捕获丰富的时间和视觉信息，而文本模态简洁地传达语义，从而导致信息密度的显著差异。因此，现有的CVR方法（Ventura等，2024b,a；Thawakar等，2024）利用统一编码器（例如BLIP、BLIP-2）对视频和文本数据进行编码，往往表现出语义偏差。如图1(b)所示，现有方法生成的组合特征往往与参考视频（黄色区域）具有过高的相似性，而与修改文本（绿色区域）则表现出较低的相似性。如右侧的相似性矩阵所示，这导致组合特征与正目标视频的相似性接近某些负候选者，最终导致检索精度下降。

为了应对组合特征中的方向性偏差，我们提出了一种基于双流方向锚的策略，以显式校准组合特征，使跨模态语义的准确集成成为可能。如图1(c)所示，我们的方法生成的组合特征对参考语义和修改语义表现出相似性，并在候选目标视频中实现了更好的区分性。然而，实现该策略涉及三个主要挑战。(1) 模态贡献纠缠。纠正方向性偏差需要识别每种模态的语义贡献。然而，由于这些语义的纠缠特性和缺乏显式监督，从组合特征中解开不同模态的语义贡献构成了第一个挑战。(2) 组合特征的显式优化。一旦识别出语义贡献，第二个挑战在于评估组合特征是否基于当前的语义贡献表现出语义方向性偏差，并据此进行显式校准。(3) 检索不确定性。类似于组合图像检索（CIR），CVR任务也依赖于三元数据，这些数据的标注成本高，且往往包含大量视觉或语义相似的候选视频（Yue et al. 2025）。这些视频给检索正确目标视频带来了相当大的不确定性。因此，仅依赖组合特征与候选视频之间的相似性可能不足以实现准确检索。因此，第三个挑战是如何评估相似性估计的可靠性以实现精确检索。为了应对上述挑战，我们提出了证据驱动的双流方向锚校准网络（ReTrack），该网络校准组合特征的方向性偏差，并利用校准的方向锚计算双向证据以实现可靠的组合到目标的相似性估计。具体而言，(1) 为了解决模态贡献纠缠问题，我们引入了语义贡献解纠缠技术，该技术分离组合特征中的视觉和文本语义贡献，以支持后续的偏差修正；(2) 为了应对组合特征显式优化挑战，我们提出了组合几何校准，该方法基于模态语义贡献构建方向锚，并重构组合特征以消除方向性偏差；(3) 为了缓解检索不确定性，我们设计了可靠证据驱动对齐方法，从锚点和目标特征之间的交互中推导双向证据，实现高可信样本的自适应加权和组合特征与目标特征之间的稳健对齐。总之，我们的贡献包括：• 我们提出了一种新型组合视频检索（CVR）框架，命名为ReTrack。根据我们所知，这是第一个通过纠正组合特征中的方向性偏差来提高多模态查询理解的CVR模型。ReTrack通过解开语义贡献实现组合特征的二次构建，允许对其空间位置和方向偏差进行精细调整。它还通过证据学习进行相似性可靠性估计，实现精确的组合特征优化。在三个广泛使用的基准数据集上进行的大量实验，覆盖CVR和CIR任务，证明了我们提出的ReTrack的优越性。

# 2 相关工作

我们的工作与组合视频检索（CVR）和不确定性估计密切相关。

组合视频检索。与CIR类似（Li et al. 2025b,c; Huang et al. 2025c; Fu et al. 2025; Wen et al. 2023b; Chen et al. 2025b,a），CVR任务专注于开发能够解读用户修改后的描述和参考视频的模型，以实现多模态视频检索。Ventura等人首次形式化了CVR，并展示了像BLIP（Li et al. 2022）和BLIP-2（Li et al. 2023a）这样的预训练视觉-语言模型在多模态查询理解中的有效性（Lu et al. 2024; Lu, Liu, and Kong 2023），并通过简单的组合机制将它们适应于CVR。Thawakar等人随后通过丰富的字幕增强了查询语义。然而，之前的方法忽略了组合特征中的方向性偏差以及多个相似候选的挑战，导致检索不准确。ReTrack通过校准特征偏差和使用方向性锚点来计算双向证据，解决了这些问题，提高了相似性估计和检索准确性。

不确定性估计 为了量化深度神经网络中的预测不确定性（Huang et al. 2025a; Liu et al. 2025d; Huang et al. 2024a; Yifan 2018; Ting 和 Listening 2024），许多研究（Liu et al. 2024a; Huang et al. 2024b; Liu et al. 2025c; Lia0 et al. 2024, 2025; Liu et al. 2025b; Wu et al. 2025）集中于不确定性估计。早期方法使用贝叶斯理论，近似后验预测分布（Huang et al. 2025b; Kingma, Salimans 和 Welling 2015; Huang et al. 2023），从而产生贝叶斯神经网络（BNNs）。然而，BNNs面临着高计算成本和推理速度缓慢的问题。证据深度学习（EDL）通过通过网络输出建模不确定性来解决这些局限性，在视觉（Liu et al. 2025e; Wang et al. 2025; Liang et al. 2024; Li et al. 2024）和多模态任务中取得了成功。

![](images/2.jpg)  
Geometry Calibration, and (c) Reliable Evidence-driven Alignment.

Kandemir（2018年）。Sensoy等（Sensoy, Kaplan, 和Kandemir 2018年）引入了主观逻辑以改善不确定性估计和鲁棒性，而Han等（Han等，2022年）则将这些思想扩展到具有动态证据融合的多视角分类，提高了可靠性。受EDL启发，ReTrack利用方向锚和目标特征之间的双向证据交互。通过自适应地加权高置信度样本，ReTrack更准确地对齐复合特征和目标特征，从而减少在检索过程中相似候选视频的影响。

# 3 ReTrack

作为一个关键创新，我们引入了ReTrack模型，该模型校准了组合特征中的方向性偏差，并利用经过校准的方向锚点提供的证据实现可靠的组合到目标的相似性计算。如图2所示，ReTrack包含三个关键模块：（a）语义贡献解耦，解耦组合特征中的视觉和文本贡献，以支持有效的偏差校准（第3.2节）；（b）组合几何校准，基于模态语义贡献构建方向锚点并重构组合特征以校准方向性偏差（第3.3节）；（c）可靠的证据驱动对齐，利用方向锚点与目标特征之间的双向证据对高可信样本加权，并可靠地将组合特征与目标对齐（第3.4节）。在本节中，我们首先形式化CVR任务，然后详细说明ReTrack的每个模块。

# 3.1 问题表述

组合视频检索（CVR）任务旨在检索满足多模态查询的目标视频。令 $\mathcal { T } = \{ ( x _ { r } , x _ { m } , x _ { t } ) _ { n } \} _ { n = 1 } ^ { N }$ 表示一个包含 $N$ 个三元组的集合，其中 $x _ { r } , x _ { m }$ 和 $x _ { t }$ 分别表示参考视频、修改文本和目标视频。我们的目标是优化一个度量空间，使得多模态查询的嵌入 $( x _ { r } , x _ { m } )$ 应尽可能接近相应的目标视频 $x _ { t }$，形式化为 $\mathcal { G } ( x _ { r } , x _ { m } ) \mathcal { G } ( x _ { t } )$，其中 $\mathcal { G }$ 表示待优化的多模态查询和目标视频的嵌入函数。

# 3.2 语义贡献解耦

为了校准组合特征中的方向性偏差，我们首先解构每种模态的语义贡献。为此，我们引入了语义贡献解构（Semantic Contribution Disentanglement），该方法首先提取参考视频、修改文本及其组合特征的特征。然后，它分别将组合特征与参考视频和修改文本分支进行交互，从而解构各自的语义贡献。解构形成后续方向性校准的基础，具体如下。双模态提取与组合。在前期工作（Ventura et al. 2024a; Xu et al. 2024; Li et al. 2025c）的基础上，我们首先利用 Q-Former 提取参考视频和修改文本的特征，以及它们的跨模态组合特征，具体公式如下，其中 $\mathbf { F } _ { r } , \mathbf { F } _ { m } , \mathbf { F } _ { c } \in \mathbb { R } ^ { Q \times D }$ 分别代表参考特征、修改特征和组合特征。$Q$ 是针对 $N _ { f }$ 采样帧的可学习查询数，$N _ { f }$ 是帧数，$D$ 是嵌入维度。$\varPhi _ { \mathbb { I } }$ 和 $\varPhi _ { \mathbb { T } }$ 分别表示视觉编码器和文本分词器。随后，对于目标视频，我们以相同方式获取目标特征 Ft R×

$$
\left\{ \begin{array} { l l } { \mathbf { F } _ { r } = \operatorname { Q - F o r m e r } ( \varPhi _ { \mathbb { I } } ( x _ { r } ) ) , \mathbf { F } _ { m } = \operatorname { Q - F o r m e r } ( \varPhi _ { \mathbb { T } } ( x _ { m } ) ) , } \\ { \mathbf { F } _ { c } = \operatorname { Q - F o r m e r } ( \varPhi _ { \mathbb { I } } ( x _ { r } ) , \varPhi _ { \mathbb { T } } ( x _ { m } ) ) , } \end{array} \right.
$$

贡献解耦。随后，我们分别解耦组合特征中的参考视频和修改文本的语义贡献。下面以参考视频为例。为了解耦参考视频的语义贡献，我们可能自然考虑从组合特征中减去修改文本特征。然而，这种简单的减法未能捕捉到真实的视觉语义贡献，因为修改语义的复杂性。因此，我们引入了 Transformer 解码器以更准确地提取参考视频的语义贡献。具体而言，参考视频特征 $\mathbf { F } _ { r }$ 被用作查询（Query），组合特征 $\mathbf { F } _ { c }$ 则作为键（Key）和值（Value），从而得到参考语义贡献 $\mathbf { P } _ { r }$，其中 $ { \mathbf { P } } _ { r } \in \mathbb { R } ^ { Q \times D }$ 是参考贡献。同样，我们获得修改贡献 $\mathbf { P } _ { m } \in \mathbb { R } ^ { Q \times D }$。

$$
\mathbf { P } _ { r } = \operatorname { D e c o d e r } ( Q = \mathbf { F } _ { r } , \{ K , V \} = \mathbf { F } _ { c } ) ,
$$

# 3.3 组合几何标定

为了校准组合特征中的潜在方向偏差（即过度偏向视觉或文本模态而牺牲另一方），我们引入了组合几何校准模块，确保校准后的特征与目标视频保持接近。该模块首先利用模态语义贡献为组合特征的每个通道生成双模态方向锚点。然后，它采用距离导向的对齐方法来最小化与目标特征的距离，并通过方向导向的校准方法从这些锚点重构组合特征，优化其相对于目标的方向。这种方法使得多模态特征组合更为精确，具体细节如下。 锚点生成。首先，由于双模态特征中的并非所有通道对组合方向的影响是相同的，因此与组合特征具有更大组合关系的通道在方向校准中应被赋予更高的权重。为此，我们基于语义贡献引入了面向组合的双模态方向锚点。参考锚点的计算如下所述，作为示例。具体来说，我们首先引入点权重 ${ \bf W } _ { p } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { Q \times D }$，该权重根据参考特征与组合特征之间的相似性自适应学习每个通道特征对方向性的影响权重，公式如下。

$$
\mathbf { W } _ { p } = \mathrm { M L P } ( \mathbf { F } _ { c } \cdot \mathbf { F } _ { r } ^ { \top } ) .
$$

随后，我们利用点权重 $\mathbf { W } _ { p }$ 来调整不同特征通道在合成方向上的语义贡献，从而生成参考锚点 $\mathbf { A } _ { r }$，其公式如下，其中 $\mathbf { A } _ { r } \in \mathbb { R } ^ { Q \times D }$。同样，我们得到修改锚点 $\mathbf { A } _ { m } \in \mathbb { R } ^ { Q \times D }$。

$$
\mathbf { A } _ { r } = \mathbf { F } _ { c } + \mathbf { W } _ { p } \odot \mathbf { P } _ { r } ,
$$

距离导向对齐。其次，为了为后续的标定提供更准确的距离基础，我们执行距离导向对齐。在这一部分，我们利用一种批次分类损失，这在CVR/CIR任务中被广泛应用（Ventura等，$2 0 2 4 \mathrm { a }$；Xu等，2024），旨在将组成特征的位置拉近至目标特征的位置，其公式如下，其中$ { \boldsymbol { S } } ( \cdot , \cdot )$是相似度函数，$B$是批次大小，$\tau$是温度系数。${ \bf F } _ { c i }$和$\mathbf { F } _ { t i }$分别表示批次中的第$i$个组成特征和目标特征。

$$
\mathcal { L } _ { d i s } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } - \log \left\{ \frac { \exp \left\{ S \left( \mathbf { F } _ { c i } , \mathbf { F } _ { t i } \right) / \tau \right\} } { \sum _ { j = 1 } ^ { B } \exp \left\{ S \left( \mathbf { F } _ { c i } , \mathbf { F } _ { t j } \right) / \tau \right\} } \right\} ,
$$

方向导向的校准最终，我们从方向锚点出发，将它们的语义贡献施加到组合特征上，从而得出组合方向锚点。然后，我们使用这个组合方向锚点作为中介，将其拉近目标特征，从而确保每种模态在组合特征中的语义贡献的准确性。具体而言，我们基于平行四边形构建组合方向锚点 $\mathbf { A } _ { c } \in \mathbb { R } ^ { \sum Q \times D }$。

$$
\mathbf { A } _ { c } = ( \mathbf { A } _ { r } - \mathbf { F } _ { c } ) + \left( \mathbf { A } _ { m } - \mathbf { F } _ { c } \right) .
$$

随后，我们计算从原始合成特征到目标特征的真实方向向量 $\mathbf { A } _ { t } = ( \mathbf { F } _ { t } - \mathbf { \bar { F } } _ { c } ) \in \mathbb { R } ^ { \mathbf { \bar { Q } } \times D }$，该向量用于指导合成方向锚点 $\mathbf { A } _ { c }$ 朝向目标特征，从而消除方向偏差，确保合成过程在空间方向上更精确地指向目标特征，具体公式如下，其中 $ { \boldsymbol { S } } ( \cdot , \cdot )$ 是相似度函数，$B$ 是批量大小，$\tau$ 是温度系数。$\mathbf { A } _ { c _ { i } }$ 和 $\mathbf { A } _ { t _ { i } }$ 分别表示批次中的第 $i$ 个合成方向锚点和真实方向向量。

$$
\mathcal { L } _ { d i r } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } - \log \left\{ \frac { \exp \left\{ S \left( \mathbf { A } _ { c i } , \mathbf { A } _ { t i } \right) / \tau \right\} } { \sum _ { j = 1 } ^ { B } \exp \left\{ S \left( \mathbf { A } _ { c i } , \mathbf { A } _ { t j } \right) / \tau \right\} } \right\} ,
$$

# 3.4 可靠的证据驱动对齐

为了减少 ReTrack 在遇到类似候选视频时的不确定性，我们提出了基于可靠证据的对齐方法。该方法通过将方向性锚点与目标特征进行交互计算双向证据，自动加权高可信样本，并可靠地将组合特征与目标特征对齐。证据建模。为了降低组合特征与目标特征之间对齐的不确定性，我们采用了 Dempster-Shafer 证据理论 $( D S T )$（Zadeh 1986）。该理论广泛应用于处理来自不同来源的可用证据，以量化给定假设的可靠性。在我们的 ReTrack 模型中，我们利用 DST 测量两个方向性锚点集与目标特征之间的相关性可靠性，从而进一步增强对齐过程中相似性矩阵的可靠性。接下来，我们将以参考锚点为例，说明证据计算过程。

具体来说，我们首先在DST中定义证据向量为 $\dot { \bf E } = [ e _ { 1 } , \dot { \bf \Xi } , \epsilon , e _ { Q } ] \in \mathbb { R } ^ { Q }$ ，它表示参考锚点 $\mathbf { A } _ { r }$ 的每个通道与目标特征 $\mathbf { F } _ { t }$ 之间的匹配证据。根据证据深度学习（Evidence Deep Learning, EDL）（Sensoy, Kaplan, 和 Kandemir 2018），我们利用主观逻辑将证据表述为如下，其中 $Q$ 是 Q-Former 中可学习查询的数量，而 $\mathbf { A } _ { r ( q ) }$ 表示参考锚点的第 $q$ 个通道。$e _ { q }$ 是参考锚点第 $q$ 个通道与目标特征之间的匹配证据。基于所有通道的匹配证据，我们进一步计算每个通道的信念质量，以衡量每个通道对其自身决策的信心，公式如下：

Table 1: Performance comparison on the test set of the CVR dataset, WebVid-CoVR, relative to $\mathrm { { R @ } } k ( \% )$ . The overall best results are in bold, while the best results over baselines are underlined.   

<table><tr><td rowspan="3">Method</td><td colspan="5">WebVid-CoVR-Test</td></tr><tr><td></td><td colspan="3">R@k</td><td rowspan="2">Avg.</td></tr><tr><td>k=1</td><td>k=5</td><td>k=10</td><td>k=50</td></tr><tr><td></td><td>Pre-trianed Models</td><td></td><td></td><td></td><td></td></tr><tr><td>CLIP (Radford et al. 2021)</td><td>44.37</td><td>69.13</td><td>77.62</td><td>93.00</td><td>71.03</td></tr><tr><td>BLIP (Li et al. 2022)</td><td>45.46</td><td>70.46</td><td>79.54</td><td>93.27</td><td>72.18</td></tr><tr><td></td><td>CVR Models</td><td></td><td></td><td></td><td></td></tr><tr><td>CoVR (Ventura et al. 2024b)</td><td>53.13</td><td>79.93</td><td>86.85</td><td>97.69</td><td>79.40</td></tr><tr><td>CoVR Enrich (Thawakar et al. 2024)</td><td>60.12</td><td>84.32</td><td>91.27</td><td>98.72</td><td>83.61</td></tr><tr><td>CoVR-2 (Ventura et al. 2024a)</td><td>59.82</td><td>83.84</td><td>91.28</td><td>98.24</td><td>83.30</td></tr><tr><td>FDCA (Yue et al. 2025)</td><td>54.80</td><td>82.27</td><td>89.84</td><td>97.70</td><td>81.15</td></tr><tr><td>ReTrack (Ours)</td><td>63.85</td><td>87.05</td><td>92.80</td><td>99.10</td><td>85.70</td></tr></table>

$$
\begin{array} { r } { e _ { q } = \exp ( \underset { \ b { \hat { q } } = 1 } { \overset { Q } { \operatorname* { m a x } } } \left( \mathbf A _ { r ( q ) } \cdot \mathbf F _ { t } ^ { \top } \right) _ { \ b { \hat { q } } } / \tau ) , } \end{array}
$$

$$
b _ { q } = \frac { e _ { q } } { \sum _ { \hat { q } = 1 } ^ { Q } \left( e _ { \hat { q } } + 1 \right) } .
$$

基于每个通道对自身决策的信任度，可以推导出参考锚点的总体关联可靠性，该可靠性在组合过程中表示方向性的语义信息，公式化为：

$$
\mathbb { E } _ { r } = \sum _ { q = 1 } ^ { Q } b _ { q } = 1 - \frac { Q } { \sum _ { \hat { q } = 1 } ^ { Q } \left( e _ { \hat { q } } + 1 \right) } .
$$

同样，我们可以获得修改文本的方向语义信息与目标特征之间的相关性可靠性，用 $\mathbb { E } _ { m }$ 表示。优化。随后，基于EDL（Sensoy, Kaplan和Kandemir 2018），我们认为相关性可靠性 $\mathbb { E } _ { r } , \mathbb { E } _ { m }$ 应该与批次中组合特征与目标特征之间的相似性正相关，以便于更好的理解。因此，基于这两组相关性可靠性，我们设计了一种证据驱动的正则化损失，以确保相似性度量与相关性可靠性之间的一致性，从而增强组合特征与目标特征之间相似性的可靠性，公式如下，其中 $B$ 是批次大小，$\mathbf { F } _ { c b } , \mathbf { F } _ { t b }$ 分别表示批次中的第 $b$ 个组合特征和目标特征。

$$
\mathcal { L } _ { e v i } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } { ( \mathbb { E } _ { r b } - S \left( \mathbf { F } _ { c b } , \mathbf { F } _ { t b } \right) ) ^ { 2 } } + \left( \mathbb { E } _ { m b } - S \left( \mathbf { F } _ { c b } , \mathbf { F } _ { t b } \right) \right) ^ { 2 } ,
$$

最终，我们得到 ReTrack 的最终损失函数，其中 $\Theta$ 是需学习的 ReTrack 参数，$\kappa$ 和 $\lambda$ 是权衡超参数。

$$
\Theta ^ { * } = \underset { \Theta } { \arg \operatorname* { m i n } } \left( \mathcal { L } _ { d i s } + \kappa \mathcal { L } _ { d i r } + \lambda \mathcal { L } _ { e v i } \right) ,
$$

# 4 实验

本节将深入探讨我们对 ReTrack 进行的全面实验及其相关分析。

# 4.1 实验设置

数据集。为了全面评估所提的 ReTrack 的有效性和普适性，我们在 CVR 和 CIR 任务上进行实验。对于 CVR 任务，我们采用大规模的开放域 WebVid-CoVR 数据集（Ventura 等，2024b）。对于 CIR 任务，我们使用广泛应用的时尚领域 FashionIQ 数据集（Wu 等，2021），以及开放域 CIRR 数据集（Liu 等，2021b）。评估指标。为了确保公平比较，我们遵循每个数据集的标准评估协议，并报告 $\operatorname { R e c a l l } @ k \left( \operatorname { R } @ k \right)$ 作为主要指标：1) WebVid-CoVR: ${ \mathrm { R } @ \left\{ 1 , 5 , 1 0 , 5 0 \right\} }$ ，以及它们的均值。2) FashionIQ: $\mathrm { R } @ \{ 1 0 , 5 0 \}$ 针对每个类别。3) CIRR: ${ \mathrm { R } @ \{ 1 , 5 , 1 0 , 5 0 \} }$ ，以及基于子集的指标 $\mathbf { R } _ { \mathrm { s u b } } @ \{ 1 , 2 , 3 \}$ 。

实施细节。根据之前的研究（Ventura et al. 2024a），我们采用基于COCO数据集微调的BLIP-2模型（Li et al. 2023a），输入分辨率为364像素，作为ReTrack的主干模型，并在训练过程中冻结ViT。帧数$N _ { f } ~ = ~ 4$，学习查询数量$Q \ = \ 3 2 N _ { f }$。对于公式(12)中的折衷超参数，我们通过网格搜索设置$\lambda = 1 . 0$和$\kappa = 0 . 5$，温度系数$\tau = 0 . 1$。ReTrack使用批量大小为64的AdamW优化器进行训练，学习率为$2 e \mathrm { ~ - ~ } 5$。在CVR和CIR数据集上进行5和10个周期的训练。所有实验均在配备32GB显存的NVIDIA V100 GPU上进行。

# 4.2 性能比较

为了验证 ReTrack 的性能和泛化能力，我们对 CVR 和 CIR 任务进行了广泛的比较。

<table><tr><td rowspan="3">Method</td><td colspan="5">FashionIQ</td><td colspan="6">CIRR</td></tr><tr><td>Dresses</td><td></td><td>Shirts</td><td></td><td>Tops&amp;Tees</td><td></td><td>R@k</td><td></td><td></td><td>Rsub @k</td><td></td></tr><tr><td>R@10 R@50</td><td>R @ 10 R @50</td><td>R@ 10 R@ 50</td><td></td><td></td><td>k=1</td><td></td><td>k=5 k=10</td><td>k=50</td><td>k=1</td><td>k=2</td><td>k=3</td></tr><tr><td colspan="9">CIR Models</td><td></td><td>89.25</td><td></td></tr><tr><td>TG-CIR (Wen et al. 2023b)</td><td>45.22</td><td>69.66 52.60</td><td>72.52</td><td>56.14</td><td>77.10</td><td>45.25</td><td>78.29</td><td>87.16 97.30</td><td></td><td>72.84</td><td>95.13</td></tr><tr><td>SSN (Yang et al. 2024)</td><td>34.36</td><td>60.78 38.13</td><td>61.83</td><td>44.26</td><td>69.05</td><td>43.91</td><td>77.25</td><td>86.48</td><td>97.45</td><td>71.76 88.63</td><td>95.54</td></tr><tr><td>SADN (Wang et al. 2024)</td><td>40.01</td><td>65.10 43.67</td><td>66.05</td><td>48.04</td><td>70.93</td><td>44.27</td><td>78.10</td><td>87.71</td><td>97.89</td><td>72.34 88.70</td><td>95.23</td></tr><tr><td>SPRC (Xu et al. 2024)</td><td>49.18</td><td>72.43 55.64</td><td>73.89</td><td>59.35</td><td>78.58</td><td>51.96</td><td>82.12</td><td>89.74</td><td>97.69</td><td>80.65 92.31</td><td>96.60</td></tr><tr><td>LIMN (Wen et al. 2023a) LIMN+ (Wen et al. 2023a)</td><td>50.72 74.52 52.11</td><td>56.08</td><td>77.09</td><td>60.94</td><td>81.85</td><td>43.64</td><td>75.37</td><td>85.42</td><td>97.04</td><td>69.01 86.22</td><td>94.19</td></tr><tr><td>IUDC (Ge et al. 2025)</td><td>75.21 35.22</td><td>57.51</td><td>77.92</td><td>62.67</td><td>82.66</td><td>43.33</td><td>75.41</td><td>85.81 97.21</td><td></td><td>69.28 86.43 -</td><td>94.26</td></tr><tr><td>ENCODER (Li et al. 2025b)</td><td>61.90 51.51 76.95</td><td>41.86 54.86</td><td>63.52 74.93</td><td>42.19 62.01</td><td>69.23 80.88</td><td></td><td>- 46.10 77.98 87.16 97.64</td><td>-</td><td>-</td><td>76.92 90.41</td><td>95.95</td></tr><tr><td colspan="10">CVR Models</td><td></td></tr><tr><td colspan="10">69.03</td></tr><tr><td>CoVR (Ventura et al. 2024b) CoVR _Enrich (Thawakar et al. 2024)</td><td>44.55</td><td>48.43</td><td>67.42</td><td>52.60</td><td>74.31</td><td></td><td>49.69 78.60 86.77 94.31</td><td></td><td>75.01</td><td></td><td>88.12 93.16</td></tr><tr><td>CoVR-2 (Ventura et al. 2024a)</td><td>46.12 69.52</td><td>49.61</td><td>68.88</td><td>53.79</td><td>74.74</td><td>51.03</td><td></td><td>88.93 97.53</td><td>76.51</td><td>-</td><td>95.76</td></tr><tr><td>ReTrack (Ours)</td><td>46.53</td><td>69.60 51.23</td><td>70.64</td><td>52.14 63.22</td><td>73.27</td><td></td><td></td><td>50.43 81.08 88.89 98.05</td><td></td><td>76.75</td><td>90.34 95.78</td></tr><tr><td></td><td>52.91</td><td>77.54</td><td>61.91 81.26</td><td></td><td>83.36</td><td></td><td>52.34 82.53 90.34 98.13</td><td></td><td></td><td>79.64</td><td>92.58 96.99</td></tr></table>

Table 2: Performance comparison on the CIR dataset, FashionIQ and CIRR, relative to $R @ k ( \% )$ The overall best results are in bold, while the best results over baselines are underlined.

在CVR任务中。如表1所示，我们比较了两类基线模型：预训练模型和CVR模型。结果揭示了以下观察结果：1）ReTrack在两个CVR数据集上在所有评估指标中均表现最佳。具体而言，在WebVid-CoVR上，ReTrack的平均指标提高了$2 . 5 0 \%$。而R1指标显著提升。这表明，通过校准组合特征中的方向性偏差并增强组合特征与目标特征之间相似性的可靠性，ReTrack有效提升了其对多模态查询的理解。2）CoVR_Enrich在WebVid-CoVR上表现不佳，这可能与其使用额外生成的标题以改善跨模态感知有关。相比之下，ReTrack在没有额外输入的情况下超越了它，仅依赖于组合几何校准和可靠证据驱动的对齐。

关于CIR任务。如表2所示，我们比较了CIR模型和CVR模型。结果得出以下关键见解：1）ReTrack在所有CIR数据集的所有指标上均表现最佳。与第二名方法相比，ReTrack在FashionIQ不同类别上的$\mathrm{R@10}$上相对提升了$1.54\%$、$7.7\%$和$0.88\%$，在CIRR上的$\mathbf{R@1}$上提升了$0.73\%$。这表明ReTrack的多模态语义解耦和基于校准的特征建模提供了强大的领域泛化能力。2）大多数CVR模型在CIR任务上落后于专业化的CIR模型，这可能是由于它们专注于全球视觉视角以及依赖于跨帧的重复关键目标，从而忽略了单帧视觉细节并引入语义偏差。相比之下，ReTrack有效地关注多模态细节并执行跨模态校准，实现了CVR和CIR的精确语义组合。这凸显了ReTrack在视觉模态语义理解方面的强大泛化能力。

# 4.3 消融研究

为了评估每个 ReTrack 模块的效果，我们对以下不同变体组进行了详细的消融研究：

G[A]: 消融实验关于语义贡献解耦 D#(1) wo_C_ref, D#(2) wo_C_mod：分别移除来自参考视频或修改文本的语义贡献，仅使用单一模态的贡献。$\mathbf { D } \# ( 3 )$ wo_SCD：移除语义贡献解耦，改用原始特征。 G[B]: 消融实验关于组合几何校准 D#(4) wo_Ldis：移除基于距离的对齐，以测试其在校准中的位置作用。D#(5) wo_A_ref, D#(6) wo_A_mod：分别移除公式(6)中的参考或修改锚点。$ { \mathbf { D } } \# ( 7 )$ WO $\mathcal { L } _ { d i r }$ ：在公式(7)中移除面向方向的校准 $\mathcal { L } _ { d i r }$ 。 G[C]: 消融实验关于可靠证据驱动对齐 D#(8) wo_Evi_ref, D#(9) wo_Evi_mod：移除常规化损失中的参考或修改证据项。D#(10) wo. $\mathcal { L } _ { e v i }$ ：移除整个证据驱动的常规化损失。 G[D]: 消融实验关于 $E \nu \mathrm { . }$ . 证据计算 D#(11) w_RELU，$ { \mathbf { D } } \# ( \mathbf { 1 } 2 )$ w Softplus：将公式(8)中的指数证据计算替换为 ReLU 和 Softplus，以测试功能选择。

从表3中，我们获得以下观察结果。1) 与完整的ReTrack模型相比，$ { \mathbf { D } } \# ( { \mathbf { 1 } } )$ 和 $\mathbf { D } \# ( 2 )$ 轻微下降，表明有效的校准和检索需要将视觉和文本贡献分离。2) 在G[A]中，$\mathbf { D } \# ( 3 )$ 显示出最大的下降，表明联合多模态语义偏差对校准特定模态的语义偏差至关重要，从而增强了多模态理解。3) D#(4) 产生了显著下降，强调了距离引导在方向校准中的重要性。$ { \mathbf { D } } \# ( 5 )$ 和 $\bf { D } \# ( \bf { 6 } )$ 的性能降低，确认参考锚点和修改锚点各自提供了重要的方向线索。D#(7) 显示出更大的下降，进一步强化了在测量每个模态贡献时距离的作用。4) D#(8) 和 $\mathbf { D } \# ( \mathbf { 9 } )$ 也导致了性能下降，表明从两个模态的量化不确定性对于可靠对齐至关重要。$\mathbf { D } \# ( \mathbf { 1 0 } )$ 在G[C]中导致了最大的下降，显示了基于证据的正则化对强大检索的重要性。5) D#(11) 和 $ { \mathbf { D } } \# ( \mathbf { 1 } 2 )$ 检查证据计算方法，揭示了符合证据理论的方法能够估计数据的不确定性，其中采用的指数方法被认为是最佳选择。

Table 3: Ablation study on three CVR and CIR datasets.   

<table><tr><td rowspan="2">D#</td><td rowspan="2">Derivatives</td><td colspan="2">FIQ-Avg.</td><td>CIRR</td><td>WebVid</td></tr><tr><td>R@10</td><td>R@50</td><td>Avg.</td><td>Avg.</td></tr><tr><td colspan="6">G[A]: Semantic Contribution Disentanglement</td></tr><tr><td>1</td><td>wo_C ref</td><td>58.84</td><td>80.25</td><td>79.86</td><td>83.90</td></tr><tr><td>2</td><td>wo_C_mod</td><td>58.68</td><td>79.94</td><td>79.54</td><td>84.20</td></tr><tr><td>3</td><td>Wo_SCD</td><td>57.69</td><td>78.48</td><td>78.49</td><td>83.37</td></tr><tr><td colspan="6">G[B]: Composition Geometry Calibration</td></tr><tr><td>4</td><td>wo_Ldis</td><td>3.78</td><td>9.12</td><td>16.08</td><td>27.59</td></tr><tr><td>5</td><td>wo_A_ref</td><td>58.21</td><td>79.48</td><td>79.73</td><td>84.33</td></tr><tr><td>6</td><td>wo_A_mod</td><td>58.31</td><td>79.87</td><td>79.54</td><td>84.27</td></tr><tr><td>7</td><td>woLdir</td><td>57.64</td><td>78.82</td><td>79.68</td><td>83.66</td></tr><tr><td colspan="6">G[C]: Reliable Evidence-driven Alignment</td></tr><tr><td>8</td><td>wo_Evi_ref</td><td>59.11</td><td>80.09</td><td>80.03</td><td>84.27</td></tr><tr><td>9</td><td>wo_Evi_mod</td><td>59.03</td><td>80.08</td><td>80.59</td><td>84.19</td></tr><tr><td>10</td><td>wo_Levi</td><td>56.93</td><td>78.19</td><td>78.94</td><td>83.02</td></tr><tr><td colspan="6">G[D]: Calculation of Evidence</td></tr><tr><td>11</td><td>w_ReLU</td><td>59.02</td><td>80.07</td><td>80.68</td><td>84.65</td></tr><tr><td>12</td><td>w_Softplus</td><td>59.11</td><td>80.19</td><td>81.01</td><td>84.54</td></tr><tr><td colspan="2">ReTrack (Ours)</td><td>59.35</td><td>80.72</td><td>81.09</td><td>85.70</td></tr></table>

![](images/3.jpg)  

Figure 3: Sensitivity to the hyper-parameters (a) $\kappa$ , and (b) $\lambda$ on WebVid-CoVR and CIRR datasets.

为了分析 ReTrack 对超参数 $\kappa$ 和 $\lambda$ 在公式 (12) 中的敏感性，我们在图 3 中展示了 WebVid-CoVR 和 CIRR 上的结果。我们观察到，对于这两个数据集，随着 $\kappa$ 和 $\lambda$ 的增加，性能先增加后下降。这种行为是合理的，因为需要校准的组成几何体并不会表现出无界偏差，而是处于一个有限范围内，因此需要平衡的超参数以限制校准的程度。此外，更大的 $\lambda$ 有效地将可靠的证据应用于相应的通道；然而，并非所有通道都需要高证据支持，因为某些通道可能固有地缺乏可靠的语义信息。因此，过大的值会导致性能下降。

# 4.4 案例研究

如图4所示，我们比较了我们的ReTrack模型与代表性的CVR模型CoVR-2在WebVid-CoVR和CIRR上的检索结果，得出了以下观察结论：1) 在图4(a)中，ReTrack将目标视频检索为第一名，而CoVR-2的前两个结果为两个“海天”视频。这一参考视频中“天空”和“海洋”的普遍性引入了视频语义的高度不确定性，减少了文本的贡献，导致CoVR-2的检索不准确。此外，由于修饰文本中对“人”的强调，CoVR-2的组合特征变得过于偏向文本。通过利用基于证据的不确定性量化，ReTrack有效减轻了背景语义干扰，实现了更高质量的结果，体现了其偏置校准和可靠相似性计算的价值。2) 在图4(b)中，ReTrack将目标图像排在首位，而CoVR-2则排在第三位。修饰文本中包含了几个低不确定性的要求，ReTrack准确捕捉到这些要求，产生了更完整的匹配。与此相比，CoVR-2仅检索到一个满足部分要求的图像作为第一名。这突显了在形成组合特征和可靠相似性计算中平衡模态贡献的必要性。

![](images/4.jpg)  

Figure 4: Case study on (a) WebVid-CoVR and (b) CIRR.

# 5 结论

在这项工作中，我们研究了新颖的CVR任务。尽管之前的方法取得了显著进展，但它们忽视了组合特征中潜在的方向性偏差，这可能导致次优的检索性能。为了应对这一局限性，我们提出了ReTrack，这是第一个通过修正组合特征中的方向性偏差来提升多模态查询理解的CVR框架。ReTrack通过计算模态特定的语义贡献来校准方向性偏差，并利用校准后的方向锚点生成双向证据，从而实现可靠的组合到目标相似性估计。此外，ReTrack还与CIR兼容，并在涵盖CVR和CIR任务的三个基准数据集上达到了最先进的性能。在未来的工作中，我们计划将我们的方法扩展到多轮交互组合多模态检索。

# 致谢

本研究部分得到了中国国家自然科学基金的支持，项目编号：62276155、62576195、62376140 和 U23A20315；同时得到了山东省泰山学者工程专项基金的支持；还得到了中国国家大学生创新创业发展计划的部分资助，项目编号：2025282 和 2025283。

# References

Chen, Z.; Hu, Y.; Li, Z.; Fu, Z.; Song, X.; and Nie, L. 2025a. OFF-SET: Segmentation-based Focus Shift Revision for Composed Image Retrieval. In ACM MM, 61136122. ACM.   
Chen, Z.; Hu, Y.; Li, Z.; Fu, Z.; Wen, H.; and Guan, W. 2025b. HUD: Hierarchical Uncertainty-Aware Disambiguation Network for Composed Video Retrieval. In ACM MM, 61436152. ACM. Fu, Z.; Li, Z.; Chen, Z.; Wang, C.; Song, X.; Hu, Y.; and Nie, L. 2025. PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval. In ICASSP, 15. IEEE.   
Ge, H.; Jiang, Y.; Sun, J.; Yuan, K.; and Liu, Y. 2025. LLM-Enhanced Composed Image Retrieval: An Intent Uncertainty-Aware Linguistic-Visual Dual Channel Matching Model. ACM TOIS, 43(2): 130.   
Han, Z.; Zhang, C.; Fu, H.; and Zhou, J. T. 2022. Trusted multiview classification with dynamic evidential fusion. IEEE TPAMI, 45(2): 25512566.   
Hu, Y.; Liu, M.; Su, X.; Gao, Z.; and Nie, L. 2021a. Video moment localization via deep cross-modal hashing. IEEE TIP, 30: 4667 4677.   
Hu, Y.; Nie, L.; Liu, M.; Wang, K.; Wang, Y.; and Hua, X.-S. 2021b. Coarse-to-fine semantic alignment for cross-modal moment localization. IEEE TIP, 30: 59335943.   
Hu, Y.; Wang, K.; Liu, M.; Tang, H.; and Nie, L. 2023. Semantic collaborative learning for cross-modal moment localization. ACM TOIS, 42(2): 126.   
Huang, J.; Du, L.; Chen, X.; Fu, Q.; Han, S.; and Zhang, D. 2023. Robust mid-pass filtering graph convolutional networks. In ACM WWW, 328338.   
Huang, J.; Mo, Y.; Hu, P.; Shi, X.; Yuan, S.; Zhang, Z.; and Zhu, X. 2024a. Exploring the Role of Node Diversity in Directed Graph Representation Learning. In IJCAI.   
Huang, J.; Mo, Y.; Shi, X.; Feng, L.; and Zhu, X. 2025a. Enhancing the Influence of Labels on Unlabeled Nodes in Graph Convolutional Networks. In ICML.   
Huang, J.; Shen, J.; Shi, X.; and Zhu, X. 2024b. On Which Nodes Does GCN Fail? Enhancing GCN From the Node Perspective. In ICML.   
Huang, J.; Xu, J.; Shi, X.; Hu, P.; Feng, L.; and Zhu, X. 2025b. The Final Layer Holds the Key: A Unified and Efficient GNN Calibration Framework. arXiv preprint arXiv:2505.11335.   
Huang, Q.; Chen, Z.; Li, Z.; Wang, C.; Song, X.; Hu, Y.; and Nie, L. 2025c. MEDIAN: Adaptive Intermediate-grained Aggregation Network for Composed Image Retrieval. In ICASSP, 15. IEEE. Jiang, W.; Zhang, S.; You, S.; Feng, P.; and Lu, Z. 2025. Traditional Chinese Painting Completion via Hierarchical Optimal Transport. IEEE Access.   
Kingma, D. P.; Salimans, T.; and Welling, M. 2015. Variational dropout and the local reparameterization trick. NeurIPS, 28. Kong, F.; Zhang, J.; Liu, Y.; Zhang, H.; Feng, S.; Yang, X.; Wang, D. Tian, Y.; Zhang, F.; Zhou, G.; et al. 2025. Modality curation: Building universal embeddings for advanced multimodal information retrieval. arXiv preprint arXiv:2505.19650.   
Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023a. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML, 1973019742. PMLR.   
Li, J.; Li, D.; Xiong, C.; and Hoi, S. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 1288812900. PMLR. L Y.; Chen, C.; Zhag Y.; Liu, W.;Lyu, L.; Zheg, X.; MeD.; and Wang, J. 2023b. Ultrare: Enhancing receraser for recommendation unlearning via error decomposition. NeurIPS, 36: 12611 12625.   
Li, Y.; Zhang, Y.; Liu, W.; Feng, X.; Han, Z.; Chen, C.; and Yan, C. 2025a. Multi-Objective Unlearning in Recommender Systems via Preference Guided Pareto Exploration. IEEE TSC.   
Li, Z.; Chen, Z.; Wen, H.; Fu, Z.; Hu, Y.; and Guan, W. 2025b. ENCODER: Entity Mining and Modification Relation Binding for Composed Image Retrieval. In AAAI.   
Li, Z.; Fu, Z.; Hu, Y.; Chen, Z.; Wen, H.; and Nie, L. 2025c. FineCIR: Explicit Parsing of Fine-Grained Modification Semantics for Composed Image Retrieval. https://arxiv.org/abs/2503.21309. Li, Z.; He, Y.; He, L.; Wang, J.; Shi, T.; Lei, B.; Li, Y.; and Chen, Q. 2024. FALCÓN: Feedback-driven Adaptive Long/short-term memory reinforced Coding Optimization system. arXiv preprint arXiv:2410.21349.   
Liang, X.; He, Y.; Xia, Y.; Song, X.; Wang, J.; Tao, M.; Sun, uan X.Su, J. LiK. et al 0Sel-evoliAts with reflective and memory-augmented abilities. arXiv preprint arXiv:2409.00872.   
Liao, B.; Zhao, Z.; Chen, L.; Li, H.; Cremers, D.; and Liu, P. 2024. GlobalPointer: Large-Scale Plane Adjustment with Bi-Convex Relaxation. In ECCV, 360376. Springer.   
Liao, B.; Zhao, Z.; Li, H.; Zhou, Y.; Zeng, Y.; Li, H.; and Liu, P. 2025. Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World. In CVPR, 1582315832.   
Liu, F.; Cheng, Z.; Zhu, L.; Gao, Z.; and Nie, L. 2021a. Interestaware message-passing GCN for recommendation. In ACM WWW, 12961305.   
Liu, F.; Liu, Y.; Chen, H.; Cheng, Z.; Nie, L.; and Kankanhalli, M. 2025a. Understanding Before Recommendation: Semantic Aspect-Aware Review Exploitation via Large Language Models. ACM TOIS, 43(2).   
Liu, H.; Li, X.; Zhang, X.; Liu, G.; and Lu, M. 2025b. In-Pipe Navigation Development Environment and a Smooth Path Planning Method on Pipeline Surface. In ICRA, 128084128090. IEEE. Liu, J.; Liu, Y.; Shang, F.; Liu, H.; Liu, J.; and Feng, W. 2025c. Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging. In ICML.   
Liu, J.; Shang, F.; Liu, Y.; Liu, H.; Li, Y.; and Gong, Y. 2024a. Fedbcgd: Communication-efficient accelerated block coordinate gradient descent for federated learning. In ACM MM, 29552963. Liu, J.; Shang, F.; Tian, Y.; Liu, H.; and Liu, Y. 2025d. Consistency of local and global flatness for federated learning. In ACM MM, 38753883.   
Liu, K.; Gong, Y.; Cao, Y.; Ren, Z.; Peng, D.; and Sun, Y. 2024b. Dual semantic fusion hashing for multi-label cross-modal retrieval. In IJCAI, 45694577. Liu, M.; Wang, X.; Nie, L.; He, X.; Chen, B.; and Chua, T.-S. 2018a. Attentive moment retrieval in videos. In ACM SIGIR, 15 24.   
Liu, M.; Wang, X.; Nie, L.; Tian, Q.; Chen, B.; and Chua, T.-S. 2018b. Cross-modal moment localization in videos. In ACM MM, 843851.   
Liu, X.; Lu, Y.; Wang, X.; and Wu, X. 2025e. Training-Free Multi-Style Fusion Through Reference-Based Adaptive Modulation. arXiv:2509.18602.   
Liu, Z.; Opazo, C. R.; Teney, D.; and Gould, S. 2021b. Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models. In ICCV, 21052114. IEEE.   
Lu, S.; Liu, Y.; and Kong, A. W.-K. 2023. Tf-icon: Diffusion-based training-free cross-domain image composition. In ICCV, 2294 2305.   
Lu, S.; Wang, Z.; Li, L.; Liu, Y.; and Kong, A. W.-K. 2024. Mace: Mass concept erasure in diffusion models. In CVPR, 64306440. Ma, Z.; Luo, Y.; Zhang, Z.; Sun, A.; Yang, Y.; and Liu, H. 2025a. Reinforcement Learning Approach for Highway Lane-Changing: PPO-Based Strategy Design.   
Ma, Z.; Zhang, Z.; Gao, Z.; Sun, A.; Yang, Y.; and Liu, H. 2025b. Energy-Constrained Motion Planning and Scheduling for Autonomous Robots in Complex Environments. preprints.   
Mu, X.; Tang, H.; Jiang, H.; Liang, T.; Zheng, Q.; and Zhu, J. 2025. FACE: A Dual-Template and Adaptive Curriculum Framework for Unsupervised Text-Based Person Search. In ACM MM, 41074116.   
Ou, Y.; de Bruijn, G.-J.; and Schulz, P. J. 2025. Social Media as an Emotional Barometer: Bidirectional Encoder Representations From TransformersLong Short-Term Memory Sentiment Analysis on the Evolution of Public Sentiments During Influenza A on Sina Weibo. JMIR, 27: e68205.   
Pu, R.; Qin, Y.; Song, X.; Peng, D.; Ren, Z.; and Sun, Y. 2025a. SHE: Streaming-media Hashing Retrieval. In ICML.   
R.Sun, Y.; Qin, Y.; Ren, Z.; Song, X.; Zh, H.; and P, D. 2025b. Robust Self-Paced Hashing for Cross-Modal Retrieval with Noisy Labels. In AAAI, volume 39, 1996919977.   
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; et al. 2021. Learning transferable visual models from natural language supervision. In ICML, 87488763. PMLR.   
Sensoy, M.; Kaplan, L.; and Kandemir, M. 2018. Evidential deep learning to quantify classification uncertainty. NeurIPS, 31.   
Sun, Y.; Peng, D.; Dai, J.; and Ren, Z. 2023a. Stepwise refinement short hashing for image retrieval. In ACM MM, 65016509.   
Sun, Y.; Ren, Z.; Hu, P.; Peng, D.; and Wang, X. 2023b. Hierarchical consensus hashing for cross-modal retrieval. IEEE TMM, 26: 824836.   
Thawakar, O.; Naseer, M.; Anwer, R. M.; Khan, S.; Felsberg, M.; Shah, M.; and Khan, F. S. 2024. Composed video retrieval via enriched context and discriminative embeddings. In CVPR, 26896 26906.   
Tian, Y.; Liu, F.; Zhang, J.; Bi, W.; Hu, Y.; and Nie, L. 2025a. Open Multimodal Retrieval-Augmented Factual Image Generation. arXiv preprint arXiv:2510.22521.   
Tian, Y.; Liu, F.; Zhang, J.; W., V.; Hu, Y.; and Nie, L. 2025b. CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG. In ACL, 3296732982.   
Ting, Y.; and Listening, C. 2024. When Radio Become a Broadcasting Application. Ventura, L.; Yang, A.; Schmid, C.; and Varol, G. 2024a. CoVR-2: Automatic Data Construction for Composed Video Retrieval. IEEE TPAMI.   
Ventura, L.; Yang, A.; Schmid, C.; and Varol, G. 2024b. CoVR: Learning composed video retrieval from web video captions. In AAAI, volume 38, 52705279.   
Wang, R.; He, Y.; Sun, T.; Li, X.; and Shi, T. 2025. UniTMGE: Uniform Text-Motion Generation and Editing Model via Diffusion. In WACV, 61046114. IEEE.   
Wang, Y.; Huang, W.; Li, L.; and Yuan, C. 2024. Semantic Distillation from Neighborhood for Composed Image Retrieval. In ACM MM.   
Wang, Y.; Zhang, F.-L.; and Dodgson, N. A. 2024. Scantd: $3 6 0 ^ { \circ }$ scanpath prediction based on time-series diffusion. In ACM MM, 77647773.   
Wang, Y.; Zhang, F.-L.; and Dodgson, N. A. 2025. Target Scanpath-Guided 360-Degree Image Enhancement. In AAAI, volume 39, 81698177.   
Wei, Y.; Wang, X.; Nie, L.; He, X.; and Chua, T.-S. 2020. Graphrefined convolutional network for multimedia recommendation with implicit feedback. In ACM MM, 35413549.   
Wei, Y.; Wang, X.; Nie, L.; He, X.; Hong, R.; and Chua, T.-S. 2019. MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video. In ACM MM, 14371445. Wen, H.; Song, X.; Yin, J.; Wu, J.; Guan, W.; and Nie, L. 2023a. Self-Training Boosted Multi-Factor Matching Network for Composed Image Retrieval. IEEE TPAMI.   
H. h X. S Xiie, L b.T guided composed image retrieval. In ACM MM, 915923.   
Wu, H.; Gao, Y.; Guo, X.; Al-Halah, Z.; Rennie, S.; Grauman, K.; and Feris, R. 2021. Fashion iq: A new dataset towards retrieving images by natural language feedback. In CVPR, 1130711317. Wu, Y.; Liu, X.; Zhao, C.; and Wu, X. 2025. Prompt-Guided Dual Latent Steering for Inversion Problems. arXiv:2509.18619.   
Xu, M.; Yu, C.; Li, Z.; Tang, H.; Hu, Y.; and Nie, L. 2025. Hdnet: A hybrid domain network with multi-scale high-frequency information enhancement for infrared small target detection. IEEE Transactions on Geoscience and Remote Sensing.   
XuX.uY. S F Zuo Wo R..M., C.-M.; et al. 2024. Sentence-level Prompts Benefit Composed Image Retrieval. In ICLR.   
Yang, X.; Liu, D.; Zhang, H.; Luo, Y.; Wang, C.; and Zhang, J. 2024. Decomposing Semantic Shifts for Composed Image Retrieval. In AAAI, volume 38, 65766584.   
Yi-fan, O. 2016. Communication and operation of TV WeChat official account. Journalism and Mass Communication, 6(12): 730 736.   
Yifan, O. 2018. Participating in Chinese Social Question and Answer Communities: A Case Study of Zhihu. com.   
Yue, W.; Qi, Z.; Wu, Y.; Sun, J.; Wang, Y.; and Wang, S. 2025. Learning Fine-Grained Representations through Textual Token Disentanglement in Composed Video Retrieval. In ICLR.   
Zadeh, L. A. 1986. A simple view of the Dempster-Shafer theory of evidence and its implication for the rule of combination. AI magazine, 7(2): 8585.   
Zhang, H.; Liu, M.; Li, Y.; Yan, M.; Gao, Z.; Chang, X.; and Nie, .Attrbue-guie colaiv earor aral r re-identification. IEEE TPAMI, 45(12): 1414414160.   
Zhang, H.; Liu, M.; Li, Z.; Wen, H.; Guan, W.; Wang, Y.; and Nie, L. 5. Spatial Understanding from Videos: Structured Prompts Meet Simulation Data. In NeurIPS, 116.