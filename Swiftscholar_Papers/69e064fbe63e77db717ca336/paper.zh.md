# PDV: 用于零-shot组合图像检索的提示方向向量

奥斯曼·图尔孙1，西南·卡尔坎2，西蒙·丹曼1，克林顿·福克斯1 1昆士兰科技大学 2中东技术大学 {osman.tursun,s.denman,c.fookes}@qut.edu.au, skalkan@metu.edu.tr

# 摘要

零-shot组合图像检索（ZS-CIR）通过参考图像和文本提示实现图像搜索，而无需依赖于在大规模配对数据上训练的专用文本-图像组合网络。然而，当前的ZS-CIR方法在依赖组合文本嵌入时存在三个关键限制：静态查询嵌入表示、对图像嵌入的利用不足以及在融合文本和图像嵌入时性能欠佳。为了解决这些挑战，我们引入了提示方向向量（PDV），这是一种简单而有效的无训练增强，能够捕捉用户提示所引起的语义修改。PDV实现了三个关键改进：（1）动态组合文本嵌入，提示调整可通过缩放因子进行控制，（2）通过从文本提示到图像特征的语义转移生成组合图像嵌入，以及（3）对组合文本和图像嵌入进行加权融合，通过平衡视觉和语义相似性来增强检索性能。我们的方法作为一种即插即用的增强技术，可与现有的ZS-CIR方法结合，计算开销极小。在多个基准测试中的广泛实验证明，当与最先进的ZS-CIR方法结合时，PDV始终能够提高检索性能，尤其是对生成准确组合嵌入的方法而言。代码将在发表时发布。

# 1. 引言

组合图像检索（CIR）涉及使用参考图像和描述目标图像应如何与参考图像不同的提示进行图像搜索 [3, 6, 22, 26]。与传统的基于内容的图像检索（CBIR）系统相比，CIR 通过允许用户表达复杂的多模态查询，结合视觉和语义信息，从而提供了更高的灵活性和精确度 [8, 14, 22]。CIR 的核心挑战在于有效整合来自两种不同模态的信息：图像和文本。随着视觉和语言模型（VLMs）的快速发展，CIR 在计算机视觉社区中引起了显著关注 [3, 6, 14, 18, 22]。早期的 CIR 方法主要是监督性质的 [1, 7, 15, 16, 26, 28]。然而，正如 Saito 等人所强调的 [22]，该领域监督数据集的标注成本高昂，促使研究人员探索更高效的替代方案，即零样本组合图像检索（ZS-CIR）。在这项工作中，我们提供了一种简单且无需训练的方法，以提高现有 ZS-CIR 方法的可控性和准确性。

ZS-CIR 利用 VLMs，记作 $\Psi$，其通过双通道架构进行操作。第一个通道由视觉分支组成，$\Psi _ { I }$，从目标图像 $I _ { t a r g e t }$ 中提取特征表示。第二个通道采用语言分支 $\Psi _ { T }$，处理参考图像 $I _ { r e f }$ 的文本组合和用户提供的文本提示 $P$。这一组合由 $\mathcal { F } ( I _ { r e f } , P )$ 表示，可以通过两种主要方法实现：(1) 标题生成，通过 VLM 为参考图像生成标题，并使用大语言模型（LLMs）将此标题与文本提示合并，如 CIReVL [14] 所示；(2) 伪标记化，使用 CLIP 的 [20] 视觉分支处理 $I _ { r e f }$，并通过一个映射网络（由轻量级多层感知机组成）对图像进行标记化，如 Pic2Word [22] 所示。得到的 $\mathcal { F } ( I _ { r e f } , P )$ 是一个文本查询表示，包含提供的视觉信息和文本信息，并促进零-shot 检索。上述管道在图 1a 中进行了说明。尽管结果令人鼓舞 [3, 11, 14, 22]，我们在文献中发现三个主要缺口：缺口 1：由于静态组合嵌入导致的低效迭代搜索 现有 CIR 研究主要强调初始检索成功，但忽视了用户在初始搜索失败时需要迭代细化和引导结果的需求。这一限制主要源于组合文本嵌入的静态特性。现有的 ZS-CIR 方法无法直接调整组合文本嵌入 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) )$，使其更接近目标图像嵌入 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) )$。在现有的 ZS-CIR 方法中，用户需要细化他们的提示以迭代改进检索结果。然而，这种方法需要额外的手动工作来构建提示，并且由于昂贵的特征提取而带来计算开销。

![](images/1.jpg)  
PDV-modified composed image embedding combined with composed text embedding.

缺口 2：参考图像嵌入的未充分利用。目前的方法通常不直接利用参考图像的嵌入 $\Psi _ { I } ( I _ { r e f } )$ 进行检索；相反，$\Psi _ { I } ( I _ { r e f } )$ 仅用于组合。这一遗漏源于将这些嵌入纳入时的检索性能始终较差（参见表 S8 中的图像单一结果），这一点在多项研究中都有记录 [3, 14, 22]。缺口 3：图像-文本嵌入融合的性能不理想。虽然图像和文本嵌入的融合优于单一模态方法（仅图像或仅文本）[3, 14, 22]，但相比于组合文本嵌入，其性能依然较差（参见补充材料中的 "图像 $^ +$ 文本" 结果，表 S5, S6）。

提示方向向量（PDV）：一种即插即用的解决方案。我们提出提示方向向量（PDV），作为一种简单的、无训练的方法来解决上述问题。用 $\Delta _ { P D V }$ 表示，PDV 代表两个文本嵌入之间的残差向量：组合文本嵌入 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) )$ 和参考图像文本嵌入 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } ) )$。后者等价于 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P _ { E m p t y } ) )$，其中 $P _ { E m p t y }$ 代表空输入字符串，对应于未提示的基线。如图 1b 所示，并通过红色箭头显示，这一 PDV 捕获了提示所引起的语义修改。接下来，我们总结 PDV 如何有效地解决这三个上述挑战。

PDV：解决差距1。为了改变组合文本嵌入的静态特性，并增加用户的灵活性和实用性，我们对组合文本嵌入的合成进行了概括，$\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) )$。我们将$\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P )$视为从无提示的参考图像文本嵌入$\Psi _ { T } ( \mathcal { F } ( I ) )$的一个偏移，偏移量为向量$\Delta _ { P D V }$。在这种表述下，基线的ZS-CIR方法可以看作是一个特例，其中$\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) = \Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) + \alpha \Delta _ { P D V }$且$\alpha =$ 1。我们假设，当$\Delta _ { P D V }$捕捉到所需的修改但没有精确的幅度（特别是在描述性较弱的提示下）时，调整$\alpha$可以提升检索性能和可控性。如图1c所示，将$\alpha$增加到1.3时，得到的结果与目标更为一致，相较于默认的$\alpha = 1$。PDV：解决差距2。尽管图像嵌入$\Psi _ { I } ( I _ { r e f } )$包含了关于参考图像的有价值的视觉内容，但它们缺乏与提示相关的语义信息，导致在ZS-CIR中表现不佳。通过利用视觉-语言模型所学习的共享语义空间，我们可以通过添加提示向量$\Delta$将提示语义转移到图像嵌入中，从而得到$\Psi _ { I } ( I _ { r e f } ) + \alpha \Delta _ { P D V }$，如图1d所示。我们将这种增强表示称为组合图像嵌入。与动态组合文本嵌入类似，该表示也可以通过缩放因子$\alpha$进行调整，以提供可控性以增强检索。

PDV：解决差距 3。最后，几项研究表明，图像和文本嵌入的直接融合优于单独使用任一输入特征（图像或文本提示）[3, 14, 22]。然而，这种融合方法仍然无法与使用组合文本嵌入的效果相比。这一性能差距存在的原因在于，通过结合参考图像的上下文，提示嵌入发生了显著变化。具体而言，$\Delta _ { P D V }$ 并不等同于 $\Psi _ { T } ( P )$ 。为了解决这一问题，我们提议融合组合文本和组合图像嵌入，如图1d所示。通过变化融合权重因子 $\beta$ ，我们可以动态调控与参考图像的视觉相似性和与提示的语义对齐之间的平衡，而无需制作新的提示或修改参考图像。较低的 $\beta$ 值优先考虑视觉保真度，而较高的值则强调在提示中指定的语义修改。先前的研究 CompoDiff [12] 也通过调整视觉和语义权重展示了类似的可控性。然而，他们的方法不能像我们的方法那样应用于其他方法，因为我们的方法是附加的，而他们的方法是特定于其框架的。PDV 作为大多数零样本类别识别（ZS-CIR）方法的即插即用增强，为其提供了一种简单且无需训练的解决方案。计算开销很小，仅需计算来自参考图像的文本和图像嵌入。我们通过将 PDV 与四种不同的 ZS-CIR 方法整合，在多个 CIR 基准上对其进行了评估。我们的实验结果表明，PDV 的所有三种应用在基准方法的基础上均有稳定的提升，尤其是在基准方法已经生成准确的组合嵌入时。贡献。我们的主要贡献如下：(i) 我们提出了提示方向向量（PDV），这是一种简单且无需训练的增强方法，克服了当前零样本类别识别方法的局限性。(ii) 我们提出了 PDV 的三种新应用：(1) 通过 PDV 缩放实现动态组合文本嵌入合成，增强对检索结果的控制，而无需繁琐的提示修改；(2) 通过 PDV 添加入将提示的语义转移到视觉特征的组合图像嵌入合成，优先考虑视觉相似性；(3) 有效融合组合文本和图像嵌入，提高整体性能并实现视觉和语义相似性的可控平衡。(iii) 通过在多个基准上对四种 ZS-CIR 方法进行广泛实验，表明 PDV 在计算开销最小的情况下持续改善检索性能。

# 2. 相关工作

视觉-语言（VL）模型通过有效地桥接视觉与文本模态，彻底改变了计算机视觉领域。强大的模型如CLIP、ALIGN和Florence的出现，使得多模态理解取得了显著进展。这些模型通过对比学习在大规模图像-文本对上进行训练，学习到富有的视觉-语义表示，能够跨领域和任务进行泛化。在这些进展的基础上，组合图像检索（CIR）也显示出显著的进展。早期方法利用VL模型，或者训练组合网络以将文本和图像特征组合起来，或者微调文本编码器以提取任务特定的文本特征。然而，这些方法需要昂贵的领域特定三元组（参考图像、修改图像和文本描述），且必须手动验证。最近的研究探索了减少数据收集负担的替代方法，如使用合成三元组或从大规模图像-文本数据集中挖掘三元组。然而，这些方法在训练过程中仍然耗费大量计算资源。

零-shot CIR与文本反转 最近的研究集中在零-shot方法上以解决这些挑战。许多方法采用文本反转，这是一种最初为个性化图像生成提出的技术，将图像映射到伪词元或单词。Pic2Word引入了一种自监督文本反转网络，通过循环对比损失进行训练，尽管它需要大规模的图像数据集。SEARLE降低了训练Pic2Word的成本，并提高了文本反转网络的效率。KEDs通过整合数据库隐式建模参考图像的属性；因此，通过反转获取的词元包含颜色、物体数量和布局等属性。为了进一步提高可扩展性，LinCIR提出了一种仅依赖语言的方法，降低了训练成本并增加了可扩展性。最近，CIReVL引入了一种更直接的方法，利用图像标题生成模型生成参考图像的自然语言描述，然后与指定所需修改的文本结合形成查询。随后提出的方法，例如LDRE和SEIZE，利用多个标题而非单个标题来增加多样性，并在组合时考虑语义增量。 残差组合 与ZS-CIR不同，早期的监督CIR方法通过对有标签的三元组数据（参考图像、提示和目标图像）进行训练来学习提示诱导的修改。Vo等人率先通过引入基于LSTM网络的残差学习模块提出了这种方法。随后，几种方法采用了类似的残差学习策略进行文本图像组合。Baldrati等人进一步推进了这一方法，通过微调CLIP的文本编码器来学习残差嵌入。虽然这些先前的工作探索了基于残差的方法，但它们都依赖于监督训练。相比之下，我们提出的PDV通过直接利用预训练的视觉-语言模型实现了类似的能力，消除了任务特定训练的需求。

# 3. 方法论

# 3.1. 基线 ZS-CIR 框架

组合图像检索（CIR）通过提供参考图像 $I _ { r e f }$ 和描述所需修改的文本提示 $P$，使用户能够搜索目标图像 $I _ { t a r g e t }$。零样本组合图像检索（ZS-CIR）利用视觉-语言（VL）模型 $\Psi$，例如 CLIP [20]，其视觉分支 $\Psi _ { I }$ 和文本分支 $\Psi _ { T }$ 被训练以学习一个共享的嵌入空间，在该空间中，语义相似的图像和文本对被映射到彼此的邻近。在此框架中，如图 1a 所示，目标图像通过视觉分支 $\Phi _ { I }$ 进行编码，而查询则通过文本分支 $\Psi _ { T }$ 同时处理 $I _ { r e f }$ 和 $P$ 进行组合，因为组合操作在文本模态中处理得更为自然。近期的 ZS-CIR 方法通过两种方法之一生成来自 $I _ { r e f }$ 和 $P$ 的组合文本嵌入：直接图像描述（CIReVL、LDRE 和 SEIZE）或伪标记化（Pic2Word、LinCIR、SEARLE 和 KEDs）。我们将这个组合过程表示为 $\mathcal { F }$，其结果是组合文本嵌入 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) )$。在理想的 ZS-CIR 情景下，目标图像 $I _ { t a r g e t }$ 应在从图库 $\mathcal { D }$ 检索到的前 $ \mathbf { \nabla } \cdot \mathbf { k } $ 个结果中出现。这一检索过程形式化为：

$$
\mathbb { I } _ { t o p - k } = \underset { I \in \mathcal { D } } { \arg \operatorname* { m a x } } _ { \boldsymbol { \mathsf { k } } } \frac { \Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) ^ { T } \cdot \Psi _ { I } ( I ) } { \| \Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) \| \cdot \| \Psi _ { I } ( I ) \| } .
$$

如果 $I_{target} \notin \mathbb{I}_{top-k}$，用户必须重新构造提示并重复特征提取过程以获得替代的检索结果，这会导致时间和计算资源的消耗。值得注意的是，如公式 1 所示，只有组合特征嵌入 $\Psi_{T}(\mathcal{F}(I_{ref}, P))$ 直接影响 $\mathbb{I}_{top-k}$ 的计算。尽管图库图像是通过其图像嵌入表示的，参考图像的图像嵌入 $\Psi_{I}(I_{ref})$ 并未对检索过程贡献作用。

# 3.2. 我们的方法：提示方向向量

我们提出了一种组合文本嵌入的广义公式，而不仅仅是简单地使用组合嵌入 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) )$ ，如图 1b 所示。通过考虑嵌入修改方向 $\delta _ { P D V }$，该方向源自提供的提示 $P$ 和参考图像 $I _ { r e f }$ 之间的差异。正式地，我们定义 $\delta _ { P D V }$ 为：

$$
\Delta _ { P D V } = \Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) - \Psi _ { T } ( \mathcal { F } ( I _ { r e f } ) ) .
$$

我们随后按如下方式形成组合文本嵌入，其中 $\alpha$ 控制沿提示向量 $\Delta _ { P D V }$ 的移动，$\Psi _ { T } ( \mathcal { F } ( I _ { r e f } ) )$ 是原始文本嵌入。

$$
\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) = \Psi _ { T } ( \mathcal { F } ( I _ { r e f } ) ) + \alpha _ { T } \Delta _ { P D V } ,
$$

# 3.3. 使用PDV的策略

我们探讨了使用 $\Delta _ { P D V }$ 的三种策略：（1）文本的提示方向向量（PDV-T），增强了零样本类别识别（ZS-CIR）的可控性。尽管基线 ZS-CIR 方法代表了一个特殊情况，其中 $\alpha = 1$，但调整 $\alpha$ 为用户提供了对检索过程的额外控制（参见图 1c）。设置 $\alpha > 1$ 会放大提示所指定的修改，而 $\alpha < 1$ 则会减少其效果。这种方法提供了一种更高效的替代方案，无需直接修改提示，因为它既不需要新的特征提取，也不需要重新表述提示。请注意，我们使用符号 $\Phi _ { P D V - T }$ 来表示组合文本嵌入。

PDV-T 在何时有效？在 CIR 中，更好的检索性能与目标嵌入向量 $\Psi _ { I } ( I _ { t a r g e t } )$ 与组合嵌入向量 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) )$ 之间的夹角 $\theta$ 成反比（参见图 2a）。当计算得到的提示方向向量 $\Delta _ { \mathrm { P D V } }$ 与真实提示方向向量 $\Delta _ { \mathrm { G T } }$ 之间的角度 $\phi$ 较小时，调整参数 $\alpha$ 可以有效减少 $\theta$。为了证明这一关系，我们进行了一个基于常见现实场景的模拟，在该场景中，由于用户编写的提示描述性较少，导致 $\Delta _ { \mathrm { P D V } }$ 的幅度小于 $\Delta _ { \mathrm { G T } }$（见图 2a）。根据模拟结果（图 2b），当 $\phi$ 小于 70 度时，将 $\alpha$ 增加到 3 可以减少 $\theta$；然而，当 $\phi$ 在 70 到 90 度之间时，降低 $\alpha$ 在减少 $\theta$ 上更为有效。这些发现表明，PDV 的性能高度依赖于基线 CIR 方法的准确性。在一个强大的基线系统下，调整 $\alpha$ 成为调优 CIR 结果的一种直接有效的方法。

(2) 图像的提示方向向量（PDV-I），将修改原理扩展到视觉嵌入。尽管之前的方法主要依赖组合文本嵌入，实验结果表明，图像和文本特征的直接融合表现不如组合特征。我们假设这种性能差距的出现是因为直接文本嵌入 $\Phi _ { T } ( P )$ 与提示向量 $\Delta _ { P D V }$ 存在显著差异，如图3所示。这种差异发生是因为自然语言的语义在上下文中是敏感的，在我们的案例中，上下文由参考图像嵌入 $\Psi _ { T } ( \mathcal { F } ( I _ { r e f } ) )$ 提供。为了克服这一局限性，我们建议将 $\Delta _ { P D V }$ 与视觉嵌入结合。具体而言，我们计算组合的视觉嵌入 $\Phi _ { P D V - I }$ 为 $\Psi _ { I } ( I _ { r e f } ) + \alpha _ { I } \Delta _ { P D V }$，其中 $\Psi _ { I } ( I _ { r e f } )$ 表示从参考图像中获得的原始视觉嵌入，而通过公式2获得的相同提示向量用于修改这一视觉表示。

![](images/2.jpg)  
re   isualiation  o $\alpha$ affects the $\theta$ a e beinh a

（3）提示方向向量融合（PDV-F），其计算查询与目标图像之间的最终相似度得分，结合了两个组成的嵌入。该融合嵌入 $\Phi _ { P D V - F }$ 可以定义为，其中 $\beta$ 是一个权重参数，用于平衡组成的视觉和文本嵌入的贡献。

$$
\Phi _ { P D V - F } = ( 1 - \beta ) \Phi _ { P D V - I } + \beta \Phi _ { P D V - T } ,
$$

![](images/3.jpg)  
Figure 3. Comparison of Image $^ +$ Text (a) vs PDV-I (b).

# 3.4. 参数调优与高效搜索

PDV 引入了三个超参数：$\alpha _ { T }$, $\alpha _ { I }$ 和 $\beta$。其中，$\alpha _ { T }$ 是最关键的，单独调整它通常就足够了。微调 $\alpha _ { I }$ 和 $\beta$ 可以进一步提升检索性能和可控性。详细的调优指南见第 $\mathrm { S } 1 . 1 ^ { 1 }$ 节，我们的自动调优方法针对 $\alpha _ { I }$ 在第 S1.1.1 节中有所描述。PDV 主要在迭代搜索中降低特征提取成本。我们还发现，通过先前排名结果过滤图库图像可以降低排名成本，这是 CIR 检索中的另一个关键因素。第 S1.2 节提供了讨论和支持实验。

# 4. 实验

实现细节。我们利用四种零样本图像语义关联（ZS-CIR）基准方法的官方实现：$\mathrm { C I R e V L } ^ { 2 }$ 和 $\mathrm { L D R E ^ { 3 }$ 作为代表性的基于标题的特征提取方法，以及 Pic2Word 和 SEARLE 作为代表性的基于伪标记化的方法。所有特征提取过程遵循基准方法提供的原始实现。然而，为了计算 $\Delta _ { P D V }$，我们需要没有提示的文本嵌入，这些嵌入在原始实现中未提供。对于 CIReVL 和 LDRE，我们通过将生成的图像标题直接传递给 CLIP 来获得这些嵌入。对于 Pic2Word 和 SEARL，我们通过将短语 "a photo of {token}" 传递给 CLIP 来构建基础文本嵌入，其中 $\langle \mathrm { t o k e n } \rangle$ 表示通过文本反演获得的提取图像标记。数据集和基础视觉-语言模型。跟随之前的工作，我们在一系列数据集上评估了我们的方法，包括 Fashion-IQ [28]、CIRR [18] 和 CIRCO [3]。我们提出的方法是一种即插即用的方式，无需额外训练，仅利用预训练模型。对于特征提取，我们使用三种 CLIP 变体：ViT-B/32、ViT-L/14 和 ViT-G/14，并使用与基准方法相同的预训练权重。对于图像标记化，我们采用预训练的 Pic2Word 模型。

# 4.1. 使用PDV的影响

我们现在探讨所提议的三种 PDV 使用方式的影响：使用 PDV 增强文本查询（PDV-T，见第 4.1 节），使用 PDV 增强图像查询（PDV-I，见第 4.1 节），以及在融合图像和文本数据的查询中使用 PDV（PDV-F，见第 4.1 节）。分析文本的 PDV（PDV-T） 为了研究提示向量 $\Delta _ { P D V }$ 的缩放如何影响组合文本嵌入的检索性能，我们进行了实验，采用两种零样本方法（CIReVL 和 Pic2Word），并使用不同的主干网络，在三个数据集上进行测试。我们通过将缩放参数 $\alpha$（方程 3）从 -0.5 变更到 3，步长为 0.1，来评估性能。

结果如图4a所示。为了考虑不同实验之间的尺度变化，我们报告相对召回值，其中以$\alpha \ = \ 1$建立了零基线。如图4a所示，变化$\alpha$会显著影响相对召回性能。我们的分析揭示了数据集间方法特定的模式。使用CIReVL时，增加$\alpha$可以改善FashionIQ和CIRCO数据集上的相对召回表现。相比之下，Pic2Word在变化$\alpha$时对FashionIQ和CIRR并没有显著改善，而在将$\alpha$降低到0.8-1.0时，CIRCO的表现有所提高。这种不同的表现与每种方法生成准确$\Delta _ { P D V }$的能力根本相关。如表1和表2所示，CIReVL在各种基准测试中始终优于Pic2Word，表明其生成更准确的组合查询及因此更准确的$\Delta _ { P D V }$的优越能力。因此，对于CIReVL来说，增加$\alpha$带来的益处大于Pic2Word。我们使用CIReVL和ViT-B-32主干在三种数据集（每种一张参考图像）下，可视化了在不同$\alpha$值下的前五条检索结果，如图5a所示。随着$\alpha$的增加，检索结果与提示的匹配度增强。相反，当$\alpha$超过1时，结果包含语义相关但未见的变体，而$\alpha$值低于0.5时，结果与提示意图相反。例如，“更亮的蓝色和无袖”检索到“深蓝色带袖子”，“单色背景”得到“自然/深色背景”，“年轻男孩”返回“成人”图像。

分析图像的 PDV（PDV-I） 为了评估 $\Delta _ { P D V }$ 是否增强了图像嵌入的检索性能，我们根据第 4.1 节中描述的协议进行了实验。我们通过添加与 $\alpha$ 值相乘的 $\Delta _ { P D V }$ 修改了图像嵌入，$\alpha$ 的范围从 -0.5 到 2.0，其中 $\alpha = 0$ 代表原始的仅图像嵌入。如图 4b 所示，Recall $@ \mathrm { K }$ 在 $\alpha$ 小于 1 时与 $\alpha$ 存在正相关关系。这个上升趋势持续到 $\alpha = 2.0$ 对于 CIReVL，而 Pic2Word 的性能在 $\alpha$ 达到 1.4 时达到峰值。PDV-I 的性能在 FashionIQ、CIRR 和 CIRCO 数据集上进行评估，与其他基于视觉嵌入的方法进行了比较，详细信息见补充材料中的表 S5、S6。结果表明，PDV-I 在现有方法中取得了显著改进。根据第 4.1 节中的方法，我们进行了类似的可视化，结果如图 5b 所示。与 PDV-T 一样，增加 $\alpha$ 会导致检索结果与提示之间的更强对齐。当 $\alpha$ 超过 0.5 时，结果与查询之间呈现语义关系，而 $\alpha$ 值低于 0.5 的结果与提示的意图相反。值得注意的是，PDV-I 的最佳检索结果与参考图像相比，其视觉相似性高于 PDV-F，这一点通过服装项目（左）和笔记本电脑（中）的设计元素得以保留可以得到证实。这一特性对于时尚搜索 [28] 和徽标检索 [25] 等应用尤为有价值，在这些应用中，视觉相似性起着至关重要的作用。分析 PDV 融合（PDV-F） 最后，我们通过将融合参数 $\beta$ 从 0 调整到 1，同时保持 PDV-I 和 PDV-F 的 $\alpha = 1$ 来评估融合图像和文本构成的嵌入的有效性。当 $\beta = 0$ 时，模型仅依赖于构成的图像嵌入，而当 $\beta = 1$ 时，仅使用构成的文本嵌入。如图 4c 所示，两个嵌入的融合始终优于单独使用任一种嵌入类型。最佳检索性能通常在 $\beta$ 介于 0.4 和 0.8 之间时实现。我们同样可视化了不同 $\beta$ 值下的前 5 个检索结果。如图 5c 所示，当 $\beta$ 较小时，检索结果与参考图像保持较高的视觉相似性。相反，当 $\beta$ 超过 0.5 时，结果与提示之间的语义对齐更强。

# 4.2. ZS-CIR 基准比较

我们在三个基准上评估了 PDV-F 以及四种基线方法（CIReVL、LDRE、Pic2Word 和 SEARLE）。值得注意的是，CIReVL 在三个数据集上使用了三种不同的主干网络进行测试，因为其模型和中间结果是公开的。然而，对于其余方法，我们使用公开可用的模型进行了评估。

数值结果见于表1和表2。这些表中显示的超参数是根据图4中确定的有效范围选取的。由于这些参数是供用户调整的，并且用户需要根据初始检索结果进行调优，我们展示了三次实验中的最佳超参数。在FashionIQ基准测试中，PDV-F对所有基线方法都产生了显著的改进，其中CIReVL在主干网络规模增加时表现出特别强劲的增益。同样，所有方法在CIRCO和CIRR数据集上均显示了显著的性能提升。值得注意的是，CIReVL相比其他方法实现了更大的改进，在使用小型和中型主干架构时，观察到最显著的增益。我们在CIReVL框架内实现的PDV-F在大多数评估指标上始终优于其他最先进的方法，包括LinCIR和SEIZE。与SEIZE类似，PDV-F的优势在于完全无训练；然而，与SEIZE不同，它并没有显著增加特征提取的计算成本。虽然LinCIR表现出卓越的推理速度，但缺乏我们方法的无训练特性，部署前需要专门的模型训练。基线。橙色 $= { \mathrm { P D V } }$ 改进，粗体 $=$ 最佳，下划线 $=$ 第二佳。 $^ \dagger$ 来自原始论文的数字。

<table><tr><td colspan="5">Fashion-IQ</td><td colspan="2">Shirt</td><td colspan="2">Dress</td><td colspan="2">Toptee</td><td colspan="2">Average</td></tr><tr><td>Backbone</td><td>Method</td><td>β</td><td>αI</td><td>αT</td><td>R@10</td><td>R@50</td><td>R @ 10</td><td>R@50</td><td>R @ 10</td><td>R@50</td><td>R@ 10</td><td>R@50</td></tr><tr><td rowspan="6">ViT-B/32</td><td>SEARLE</td><td>-</td><td>-</td><td>-</td><td>24.14</td><td>41.81</td><td>18.39</td><td>38.08</td><td>25.91</td><td>47.02</td><td>22.81</td><td>42.30</td></tr><tr><td>SEARLE + PDV-F</td><td>0.9</td><td>1.1</td><td>0.9</td><td>24.83</td><td>41.71</td><td>20.13</td><td>41.40</td><td>25.96</td><td>47.17</td><td>23.64</td><td>43.43</td></tr><tr><td>CIReVL</td><td>-</td><td>-</td><td>-</td><td>28.36</td><td>47.84</td><td>25.29</td><td>46.36</td><td>31.21</td><td>53.85</td><td>28.29</td><td>49.35</td></tr><tr><td>CIReVL + PDV-F</td><td>0.75</td><td>1.4</td><td>1.4</td><td>32.88</td><td>52.80</td><td>32.67</td><td>54.49</td><td>38.91</td><td>61.81</td><td>34.82</td><td>56.37</td></tr><tr><td>LDRE</td><td>-</td><td>-</td><td>-</td><td>27.38</td><td>46.27</td><td>19.97</td><td>41.84</td><td>27.07</td><td>48.78</td><td>24.81</td><td>45.63</td></tr><tr><td>SEIZE</td><td>-</td><td>-</td><td>-</td><td>29.38</td><td>47.97</td><td>25.37</td><td>46.84</td><td>32.07</td><td>54.78</td><td>28.94</td><td>49.86</td></tr><tr><td rowspan="8">ViT-L/14</td><td>Pic2Word</td><td></td><td></td><td></td><td>25.96</td><td>43.52</td><td>19.63</td><td>40.90</td><td>27.28</td><td>47.83</td><td>24.29</td><td>44.08</td></tr><tr><td>Pic2Word + PV-F</td><td>0.8</td><td>1.0</td><td>1.0</td><td>28.21</td><td>44.55</td><td>20.92</td><td>42.24</td><td>29.02</td><td>48.90</td><td>26.05</td><td>45.23</td></tr><tr><td>SEARLE</td><td>-</td><td>-</td><td>-</td><td>26.84</td><td>45.19</td><td>20.08</td><td>42.19</td><td>28.40</td><td>49.62</td><td>25.11</td><td>45.67</td></tr><tr><td>SEARLE +PDV-F</td><td>0.8</td><td>1.2</td><td>1.0</td><td>28.66</td><td>46.76</td><td>23.60</td><td>46.41</td><td>31.00</td><td>52.32</td><td>27.75</td><td>48.50</td></tr><tr><td>CIReVL</td><td></td><td></td><td></td><td>29.49</td><td>47.40</td><td>24.79</td><td>44.76</td><td>31.36</td><td>53.65</td><td>28.55</td><td>48.57</td></tr><tr><td>CIReVL + PDV-F</td><td>0.55</td><td>1</td><td>1.3</td><td>37.78</td><td>54.22</td><td>33.61</td><td>56.07</td><td>41.61</td><td>62.16</td><td>37.67</td><td>57.48</td></tr><tr><td>LinCIR</td><td>-</td><td>-</td><td>-</td><td>29.10</td><td>46.81</td><td>20.92</td><td>42.44</td><td>28.81</td><td>50.18</td><td>26.82</td><td>46.49</td></tr><tr><td>SEIZE</td><td>-</td><td>-</td><td>-</td><td>33.04</td><td>53.22</td><td>30.93</td><td>50.76</td><td>35.57</td><td>58.64</td><td>33.18</td><td>54.21</td></tr><tr><td rowspan="6">ViT-G/14</td><td>Pic2Word</td><td>-</td><td>-</td><td>-</td><td>33.17</td><td>50.39</td><td>25.43</td><td>47.65</td><td>35.24</td><td>57.62</td><td>31.28</td><td>51.89</td></tr><tr><td>SEARLE</td><td>-</td><td>-</td><td>-</td><td>36.46</td><td>55.35</td><td>28.16</td><td>50.32</td><td>39.83</td><td>61.45</td><td>34.81</td><td>55.71</td></tr><tr><td>CIReVL</td><td>-</td><td>-</td><td>-</td><td>33.71</td><td>51.42</td><td>27.07</td><td>49.53</td><td>35.80</td><td>56.14</td><td>32.19</td><td>52.36</td></tr><tr><td>CIReVL + PV-F</td><td>0.6</td><td>1.4</td><td>1.4</td><td>41.90</td><td>58.19</td><td>40.70</td><td>62.82</td><td>48.09</td><td>67.77</td><td>43.56</td><td>62.93</td></tr><tr><td>LinCIR</td><td>:</td><td>-</td><td></td><td>46.76</td><td>65.11</td><td>38.08</td><td>60.88</td><td>50.48</td><td>71.09</td><td>45.11</td><td>65.69</td></tr><tr><td>SEIZE</td><td></td><td>-</td><td>:</td><td>43.60</td><td>65.42</td><td>39.61</td><td>61.02</td><td>45.94</td><td>71.12</td><td>43.05</td><td>65.5</td></tr></table>

<table><tr><td colspan="5">Dataset</td><td colspan="4">CIRCO</td><td colspan="8">CIRR</td></tr><tr><td colspan="5">Metric</td><td colspan="4">mAP@k</td><td colspan="5">Recall@k</td><td colspan="3">Rs@k</td></tr><tr><td>Arch</td><td>Method</td><td>β</td><td>αI</td><td>αT</td><td>k=5</td><td>k=10</td><td>k=25</td><td>k=50</td><td>k=1</td><td></td><td>k=5</td><td>k=10</td><td>k=50</td><td>k=1</td><td>k=2</td><td>k=3</td></tr><tr><td rowspan="8">ViT-B/32</td><td>PALAVRA [8] </td><td>-</td><td>-</td><td>-</td><td></td><td>4.61</td><td>5.32</td><td>6.33</td><td>6.80</td><td>16.62</td><td>43.49</td><td>58.51</td><td>83.95</td><td>41.61</td><td>65.30</td><td>80.94</td></tr><tr><td>SEARLE</td><td>-</td><td>-</td><td>-</td><td></td><td>9.35</td><td>9.94</td><td>11.13</td><td>11.84</td><td>24.00</td><td>53.42</td><td>66.82</td><td>89.78</td><td>54.89</td><td>76.60</td><td>88.19</td></tr><tr><td>SEARLE + PDV-F</td><td>0.9</td><td>1.4</td><td>1.2</td><td></td><td>9.99</td><td>10.50</td><td>11.70 12.40</td><td></td><td>24.53</td><td>53.71</td><td>67.33</td><td>89.81</td><td>56.94</td><td>78.05</td><td>88.99</td></tr><tr><td>CIReVL</td><td></td><td>- -</td><td>-</td><td></td><td>14.94</td><td>15.42</td><td>17.00</td><td>17.82</td><td>23.94</td><td>52.51</td><td>66.00</td><td>86.95</td><td>60.17</td><td>80.05</td><td>90.19</td></tr><tr><td>CIReVL + PDV-F</td><td></td><td>0.75</td><td>1.4</td><td>1.2</td><td>19.90</td><td>20.61</td><td>22.64</td><td>23.52</td><td>33.25</td><td>64.15</td><td>75.23</td><td>92.43</td><td>65.81</td><td>83.76</td><td>92.10</td></tr><tr><td>LDRE</td><td>-</td><td>-</td><td>-</td><td></td><td>17.81</td><td>18.04</td><td>19.73</td><td>20.67</td><td>25.69</td><td>55.52</td><td>68.77</td><td>89.86</td><td>60.10</td><td>80.58</td><td>91.04</td></tr><tr><td>LDRE + PDV-F</td><td>0.75</td><td>1.4</td><td>1.4</td><td></td><td>17.80</td><td>18.78</td><td>20.61</td><td>21.56</td><td>29.30</td><td>60.39</td><td>72.51</td><td>91.42</td><td>63.06</td><td>82.36</td><td>91.54</td></tr><tr><td rowspan="13"></td><td>SEIZE</td><td>-</td><td>- -</td><td>-</td><td>19.04</td><td>19.64</td><td>21.55</td><td>22.49</td><td>27.47</td><td>57.42</td><td>70.17</td><td></td><td>-</td><td>65.59</td><td>84.48</td><td>92.77</td></tr><tr><td>Pic2Word Pic2Word + PDV-F</td><td>-</td><td></td><td>- 1.0</td><td></td><td>6.81</td><td>7.49</td><td>8.51</td><td>9.07</td><td>23.69</td><td>51.32</td><td>63.66</td><td>86.21</td><td>53.61</td><td>74.34</td><td>87.28</td></tr><tr><td>SEARLE</td><td></td><td>0.85 -</td><td>1.2 -</td><td></td><td>7.74</td><td>8.67 12.73</td><td>9.77</td><td>10.37</td><td>23.90</td><td>51.95</td><td>64.63</td><td>87.04</td><td>53.16</td><td>74.07</td><td>87.08</td></tr><tr><td>SEARLE + PDV-F</td><td></td><td></td><td>-</td><td></td><td>11.68 12.58</td><td>13.57</td><td>14.33</td><td>15.12</td><td>24.24</td><td>52.48</td><td>66.29</td><td>88.84</td><td>53.76</td><td>75.01</td><td>88.19</td></tr><tr><td></td><td>0.85</td><td>1.4</td><td>1.2</td><td></td><td></td><td></td><td>15.30</td><td>16.07</td><td>25.64</td><td>53.61</td><td>66.58</td><td>88.55</td><td>55.83</td><td>76.48</td><td>88.53</td></tr><tr><td>CIReVL</td><td>-</td><td>-</td><td>- 1.2</td><td></td><td>18.57 25.67</td><td>19.01 26.61</td><td>20.89 28.81</td><td>21.80</td><td>24.55</td><td>52.31</td><td>64.92</td><td>86.34</td><td>59.54</td><td>79.88</td><td>89.69</td></tr><tr><td>ViT-L/14 CIReVL + PDV-F LDRE</td><td>0.75</td><td>1.4 -</td><td>-</td><td></td><td></td><td></td><td></td><td>29.95</td><td>36.24</td><td>66.17</td><td>76.96</td><td>92.29</td><td>68.07</td><td>85.35</td><td>93.47</td></tr><tr><td>LDRE + PDV-F</td><td></td><td>- 0.75</td><td></td><td>1.4</td><td>22.32 25.23</td><td>23.75 26.52</td><td>25.97 28.94</td><td>27.03 29.95</td><td>26.68</td><td>55.45 59.98</td><td>67.49 71.90</td><td>88.65</td><td>60.39</td><td>80.53</td><td>90.15</td></tr><tr><td>LinCIR</td><td></td><td></td><td>1.4 -</td><td>-</td><td>12.59</td><td>13.58</td><td>15.00</td><td>15.85</td><td>30.16 25.04</td><td>53.25</td><td>66.68</td><td>90.87</td><td>63.66</td><td>82.87</td><td>91.57</td></tr><tr><td>SEIZE</td><td></td><td>- -</td><td>-</td><td>-</td><td>24.98</td><td>25.82</td><td>28.24</td><td>29.35</td><td>28.65</td><td>57.16</td><td>69.23</td><td>- -</td><td>57.11 66.22</td><td>77.37 84.05</td><td>88.89 92.34</td></tr><tr><td rowspan="7">ViT-G/14</td><td>CIReVL</td><td>-</td><td>-</td><td>-</td><td></td><td>26.77 27.59</td><td>29.96</td><td>31.03</td><td>34.65</td><td>64.29</td><td>75.06</td><td>91.66</td><td>67.95</td><td>84.87</td><td></td></tr><tr><td>CIReVL + PDV-F</td><td>0.75</td><td>1.4</td><td>1.2</td><td>30.02</td><td>31.46</td><td>34.01</td><td>35.08</td><td>38.15</td><td>67.93</td><td>77.90</td><td>92.77</td><td></td><td>69.37 85.37</td><td>93.21 93.45</td></tr><tr><td>LDRE</td><td>-</td><td>-</td><td>-</td><td>33.30</td><td>34.32</td><td>37.17</td><td>38.27</td><td>37.40</td><td>66.96</td><td>78.17</td><td>93.66</td><td>68.84</td><td>85.64</td><td>93.90</td></tr><tr><td>LDRE + PDV-F</td><td>0.75</td><td>1.4</td><td>1.4</td><td>34.88</td><td>36.41</td><td>39.12</td><td>40.23</td><td>42.51</td><td>72.22</td><td>81.71</td><td>94.94</td><td>72.39</td><td>88.34</td><td>94.80</td></tr><tr><td>SEARLE</td><td>-</td><td></td><td>-</td><td>13.20</td><td>13.85</td><td>15.32</td><td>16.04</td><td>34.80</td><td>64.07</td><td>75.11</td><td>,</td><td>68.72</td><td>84.70</td><td>93.23</td></tr><tr><td>LinCIR</td><td>-</td><td>-</td><td>-</td><td>19.71</td><td>21.01</td><td>23.13</td><td>24.18</td><td>35.25</td><td>64.72</td><td>76.05</td><td>-</td><td>63.35</td><td>82.22</td><td>91.98</td></tr><tr><td>SEIZE</td><td>-</td><td>-</td><td>-</td><td>32.46</td><td>33.77</td><td>36.46</td><td>37.55</td><td>38.87</td><td>69.42</td><td>79.42</td><td>-</td><td>74.15</td><td>89.23</td><td>95.71</td></tr></table>

Table 2. Performance comparison on CIRCO and CIRR test datasets. As in previous works, for CIRCO, $\mathrm { m A P } @ \mathrm { k }$ is reported, while for CIRR both Recall $@ \mathbf { k }$ and $R _ { s } \ @ \mathbf { k }$ metrics are used. Orange $=$ PDV improvements, bold $=$ best, underline $=$ second." ‡Numbers from original paper.

![](images/4.jpg)  
(a) PDV-T: Impact of $_ \alpha$ scaling on composed text mbedings

![](images/5.jpg)  
(b) PDV-I: Impact of $_ \alpha$ scaling on composed image embeddings

![](images/6.jpg)  
(c) PDV-F: Impact of varying $\beta$ with on composed fused embeddings   
Figure 4. Impact of changing $\alpha / \beta$ on Recall $\textcircled { \omega } 5$ performance across different PDV applications. For each row, results are shown for the CIReVL (left) and Pic2Word (right) baseline methods.

![](images/7.jpg)  
Figure 5. Visualisation of the impact of $\alpha / \beta$ scaling on top-5 retrieval results. CIReVL with ViT-B-32 Clip model is the baseline method top. Green and blue bounding boxes indicate true positives and near-true positives, respectively.

# 5. 结论与未来应用

我们引入了提示方向向量（PDV），这是一种简单但有效的方法，旨在增强零样本组合图像检索。PDV 捕捉由用户提示引起的语义修改，无需额外的训练或昂贵的数据收集。通过在多个基准上进行广泛的实验，我们展示了 PDV 的三种成功应用：动态文本嵌入合成、通过语义转移进行组合图像嵌入以及有效的多模态融合。我们的方法不仅始终如一地提高了检索性能，还通过使用缩放因子提供了更高的可控性。PDV 作为一种即插即用的增强方法，可以与现有的零样本组合图像检索方法轻松集成，同时所需的计算开销极少。我们注意到，PDV 的有效性与基础方法生成准确组合嵌入的能力具有很强的关联。这一洞察表明了有前景的未来研究方向，包括开发更强大的组合嵌入技术以及探索 PDV 的自适应缩放策略。PDV 的简单性和有效性还为其在多提示组合图像检索（即基于对话的搜索）以及其他语义修改发挥重要作用的多模态任务中的应用开辟了可能性。

# 附录 A. 额外实验与结果

# A.1. 关于超参数调优

在本研究中，我们引入了三个参数：$\alpha _ { I } , ~ \alpha _ { T }$ 和 $\beta$。一旦理解了它们的语义角色，手动调整它们就变得直观且简单： $\alpha _ { T } $ 表示文本提示对参考图像语义内容的影响 $\alpha _ { I } $ 表示文本提示对参考图像视觉内容的影响 $\beta $ 控制视觉内容和语义内容之间的权衡 我们还考察了自动调优的潜力。然而，由于多个不确定因素——如提示质量、在目标数据集上的基线表现和用户期望——自动优化这三个参数仍然非常具有挑战性。尽管如此，我们发现可以自动调节 $\alpha _ { I }$ 以减少 PDV-I 和 PDV-T 之间的差距。在以下各小节中，我们将讨论每个参数的手动调整和 $\alpha _ { I }$ 的自动调优。

# A.1.1 调整 $\alpha _ { T }$ : 文本提示的影响

在这些参数中，最重要的是 $\alpha _ { T }$，它主要控制文本提示的影响。如果用户观察到检索到的顶级结果中的语义变化不足，则应增加 $\alpha _ { T }$；反之，如果变化过强，则应减少其值。例如，在图 S6 的顶端案例中，当 $\alpha _ { T } = 1$ 时，裙子尚未显示出明显的白色条纹。增加 $\alpha _ { T }$ 会产生更加明显的白色条纹的结果。相反，在图 S6 的中间和底部示例中，当 $\alpha _ { T } = 1$ 时，检索到的结果已经有效。在这些情况下进一步增加 $\alpha _ { T }$ 会使语义变化过于强烈，导致该方法返回无效结果。

# A.1.2 手动调节 $\alpha _ { I }$ : 文本提示的影响

$\alpha_{I}$ 的角色与 $\alpha_{T}$ 相似。它同样控制提示的强度，但在这种情况下，组成部分是原始的视觉嵌入 $\Psi_{I}(I_{ref})$。当 $\alpha_{I} = 0$ 时，PDV-I 简化为基于内容的图像检索。从图 S13 中显示的消融结果来看，我们观察到将 $\alpha_{I} = 1$ 通常是安全的，因大多数方法在将 $\alpha_{I}$ 从 $-0.5$ 增加时能够获得一致的提升。然而，进一步增加 $\alpha_{I}$ 也可能是有益的。当检索到的顶级结果与参考图像过于相似且未能结合提示中指定的语义概念时，用户应该继续增加 $\alpha_{I}$。例如，在图 S7 的顶部和底部示例中，当 $\alpha_{I} > 1$ 时，top-1 检索结果成功呈现了用户提示中描述的语义元素。

# A.1.3 调整 $\beta$ ：融合因子

参数 $\beta$ 在 PDV-F 中被使用，它融合了 PDV-I 和 PDV-T。其值范围从 0 到 1。当 $\beta = 1$ 时，PDV-F 相当于 PDV-T；当 $\beta = 0$ 时，降为 PDV-I。从图 S14 中的消融结果来看，我们观察到大多数方法在 $\beta$ 位于 0.6 到 0.9 之间时，性能有所提高。除了性能优化，$\beta$ 在平衡检索特性方面也起着关键作用。如图 S8 所示，较低的 $\beta$ 值 $( \beta < 0 . 5 )$ 强调视觉相似性，产生在外观上与参考图像 $I_{ref}$ 密切相似的最佳结果。相反，较高的 $\beta$ 值 $( \beta > 0 . 5 )$ 则优先考虑语义一致性，融入文本提示中描述的概念元素。这提供了对检索系统在视觉真实性与语义相关性之间的偏好进行细粒度控制的能力。

# A.1.4 自动调整 $\alpha _ { I }$

根据实验结果，我们观察到PDV-T的性能始终优于PDV-I。如果PDV-I的特征$\Phi _ { \mathrm { P D V - I } }$与PDV-T的特征$\Phi _ { \mathrm { P D V - T } }$更为接近，则PDV-I的性能可以接近PDV-T。受到这一观察的启发，我们调整参数$\alpha _ { I }$（同时保持$\Phi _ { \mathrm { P D V - T } }$不变），以最小化$\Phi _ { \mathrm { P D V - I } }$与$\Phi _ { \mathrm { P D V - T } }$之间的$\ell _ { 2 }$距离，如方程S5所示。为了确定$\alpha _ { I }$的最优值，我们采用Nelder-Mead优化方法[19]，这是一种无导数且简单的方法，实施起来特别方便。

$$
\alpha _ { I } = \underset { \alpha } { \arg \operatorname* { m i n } } \ : \mathcal { L } ( \Phi _ { \mathrm { P D V - T } } , \Phi _ { \mathrm { P D V - I } } ( \alpha ) ) ,
$$

为了评估所提方法的有效性，我们将 $\alpha _ { T }$ 固定为 1，使 PDV-T 等价于基线，并比较调优后的 $\alpha _ { I }$ 与固定设置 $\alpha _ { I } = 1$ 的性能。我们在 FashionIQ 数据集上使用三种不同的方法和多种主干架构进行了实验。如表 S3 所示，我们的方法成功为每个设置确定了定制的 $\alpha _ { I }$ 值。在所有情况下，$\mathrm { R @ 5 0 }$ 都显示出相对于基线的一致改进，其中 CIReVL 方法在 Toptee 子集中使用 ViT-B/32 主干时取得了最大 $23 \%$ 的增益。$\mathrm { R @ 1 0 }$ 在大多数场景中也稳步提高，特别是在 CIReVL 方法下。然而，对于 Pic2Word，Dress 子集上的 $\textrm { R @ 1 0 }$ 降低了 $3 . 4 2 \%$。

# A.2. 采用PDV进行高效检索

PDV旨在提高基线方法在随后搜索中的检索性能，该搜索在某个条件触发时进行。

# 提示

穿着条纹裙子的女孩身材较矮，裙子上有更多的白色，周围显示出较少的人，还有一辆停放的摩托车。

![](images/8.jpg)  
Figure S6. Qualitative results of PDV-T showing the effect of different $\alpha _ { T }$ values. For each query, we display the top-1 retrieval result for three different $\alpha _ { T }$ settings. The middle result uses $\alpha _ { T } = 1$ (baseline), the left result uses a smaller $\alpha _ { T }$ value, and the right result uses a larger $\alpha _ { T }$ value. All $\alpha _ { T }$ values are within the range $[ - 0 . 5 , 2 ]$ .

小狗坐在酒红色的沙发上。表 S3。与固定设置 $\alpha _ { I } = 1$ 相比，自动调节 $\alpha _ { I }$ 在 FashionIQ 数据集上的性能差异。$\alpha _ { I }$ 列报告了 Shirt、Dress 和 Toptee 子集的调节值。

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td rowspan="2">αI</td><td colspan="2">Shirt</td><td colspan="2">Dress</td><td colspan="2">Toptee</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td rowspan="2">ViT-B/32</td><td>SEARLE</td><td>1.57/1.65/1.58</td><td>15.68%</td><td>15.62%</td><td>20.04%</td><td>20.51%</td><td>14.52%</td><td>11.18%</td></tr><tr><td>CIReVL</td><td>1.92/2.24/2.02</td><td>25.65%</td><td>18.82%</td><td>24.95%</td><td>16.86%</td><td>27.81%</td><td>23.11%</td></tr><tr><td rowspan="3">ViT-L/14</td><td>CIReVL</td><td>1.90/2.16/1.95</td><td>12.96%</td><td>9.58%</td><td>15.12%</td><td>10.50%</td><td>17.70%</td><td>12.80%</td></tr><tr><td>Pic2Word</td><td>1.47/1.46/1.48</td><td>0.00%</td><td>3.09%</td><td>-3.42%</td><td>2.07%</td><td>0.00%</td><td>0.74%</td></tr><tr><td>SEARLE</td><td>1.67/1.82/1.73</td><td>5.84%</td><td>10.62%</td><td>16.57%</td><td>9.41%</td><td>6.15%</td><td>9.86%</td></tr><tr><td>ViT-G/14</td><td>CIReVL</td><td>1.50/1.63/1.53</td><td>10.43%</td><td>6.61%</td><td>19.15%</td><td>14.30%</td><td>17.51%</td><td>12.14%</td></tr></table>

初始查询失败，而不需承担通常与迭代搜索过程相关的高计算成本。在零次学习组合图像检索（ZS-CIR）系统中，计算瓶颈主要源于两个基本操作：特征提取和相似度排序。在特征提取方面，成本由所使用的模型决定。近期的ZS-CIR方法依赖于大型视觉-语言模型，其特征提取的开销显著，如表S4所详述。相比之下，PDV通过基于初始检索中的嵌入构建新的特征来进行后续试验。这个过程仅涉及高效的标量乘法和矩阵加法，使得每次试验的特征提取成本几乎可以忽略不计。唯一的显著计算开销是参考图像嵌入的一次性初始计算，$\Psi _ { I } ( I _ { r e f } )$。另一方面，相似度排序的成本主要取决于特征维度和图库大小。尽管特征维度是固定的基础模型所决定，但图库大小可以在后续搜索中减少。为了提高效率，我们将简单的过滤策略与PDV结合：对超出预定义阈值的查询距离的项目进行移除，以便于后续排名。虽然这并非PDV所独有，但我们认为这是在ZS-CIR背景下首次讨论这种优化。我们在FashionIQ数据集上评估了这种方法，使用了两个基线方法CIReVL和Pic2Word。如表S5所示，对于$\mathrm { C I R e V L }$，阈值为0.8时可过滤掉超过$80 \%$的图库项目，同时$\mathrm { R @ 50 }$指标最多降低$1.31 \%$。对于Pic2Word，阈值为0.75时过滤掉超过$68 \%$的图库，最大$\mathrm { R @ 50 }$下降为$3.73 \%$。这些结果表明，结合图库过滤的PDV提供了一个高效的权衡方案，显著加速了检索速度，同时保持了竞争力的准确性。

![](images/9.jpg)  
Figure S7. Qualitative results of PDV-I showing the effect of different $\alpha \boldsymbol { I }$ values. For each query, we display the top-1 retrieval result for three different $\alpha _ { I }$ settings. The middle result uses $\alpha _ { I } = 1$ (baseline), the left result uses a smaller $\alpha _ { I }$ value, and the right result uses a larger $\alpha _ { I }$ value. All $\alpha _ { I }$ values are within the range $[ - 0 . 5 , 2 ]$ .

![](images/10.jpg)  
Figure S8. Qualitative results of PDV-F illustrating the effect of different $\beta$ values. For each query, we show the top-1 retrieval result unde three settings: $\beta = 0$ (left), $\beta = 0 . 5$ (middle), and $\beta = 1$ (right).

表 S4. 基线 ZS-CIR 方法在 NVIDIA A100 GPU 上计算效率的比较。

<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=2>Feature Extraction Time (Sec.)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Initial</td><td rowspan=1 colspan=1>Retrial</td></tr><tr><td rowspan=1 colspan=1>Pic2Word</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>+ PDV</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>LinCIR</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.02</td></tr><tr><td rowspan=1 colspan=1>+ PDV</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=2 colspan=1>KEDsCIReVL (2 Captions)</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td rowspan=1 colspan=1>1.23</td><td rowspan=1 colspan=1>1.23</td></tr><tr><td rowspan=3 colspan=1>+ PDVLDRE (20 Captions)+ PDV</td><td rowspan=1 colspan=1>1.24</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=1>17.30</td><td rowspan=1 colspan=1>17.30</td></tr><tr><td rowspan=1 colspan=1>17.31</td><td rowspan=1 colspan=1>0.00</td></tr></table>

# A.3. 基线的 $\phi$ 角

在主论文的第3.3节中，我们讨论了PDV的性能与基线性能之间的高度相关性，并通过如图2所示的仿真结果提供了理论依据。我们引入了一个新参数$\phi$，表示计算得到的提示方向向量$\Delta _ { \mathrm { P D V } }$与真实提示方向向量$\Delta _ { \mathrm { G T } }$之间的夹角。当$\phi$较小时，调整参数$\alpha$可以有效降低$\theta$，后者是目标嵌入向量$\Psi _ { I } ( I _ { t a r g e t } )$与组成嵌入向量$\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) )$之间的夹角。

在这里，我们展示了三种基线方法的实际 $\phi$ 值——CIReVL、Pic2Word 和 SEARLE——在 FashionIQ 数据集的三个子数据集上使用不同的主干网络。图 S9 展示了这三种基线方法在 FashionIQ 子数据集上的 $\phi$ 值。尽管我们观察到预期的趋势，即更强的模型表现出更小的 $\phi$ 值，但我们惊讶地发现，最先进的 CIReVL 基线仍维持着约 $6 5 ^ { \circ }$ 的较大 $\phi$ 角度。这个发现表明，尽管已在 CIReVL 上取得了相当大的收益，PDV 仍未与最优基线模型进行评估。因此，我们预期，当 PDV 与未来能够取得更小 $\phi$（理想情况下 $< 6 0 ^ { \circ }$ ）的基线配对时，将会更加有效，这是根据图 2b 所示，我们的方法影响最大的一种情况。

# A.4. 额外的定量结果

在本补充部分，我们展示了由于篇幅限制而未包含在主论文中的额外定量结果。

# A.4.1 消融分析

主论文中的图 3 说明了缩放因子 $\alpha$ 和融合因子 $\beta$ 对各种 PDV 应用中 Recall $\textcircled { a } 5$ 性能的影响，图 S12、S13 和 S14 则呈现了 Recall $@ 1 0$ 和 Recall $@ 5 0$ 指标的互补结果。Recall $@ 1 0$ 和 Recall $@ 5 0$ 的结果与主论文中 Recall $\textcircled { \alpha } 5$ 的发现表现出一致的趋势，从而验证了我们在多个评估指标上得出的结论。

# A.4.2 PDV-I 结果

我们还提供了在FashionIQ数据集验证集上取得的额外PDV-I结果，如表S7和S8所示。PDV-I在直接利用图像嵌入进行检索的现有方法上也取得了显著改善。最后，我们详细展示了$\alpha / \beta$缩放对前五个检索结果的影响。图S11展示了使用ViT-B-32 CLIP模型的CIReVL在三个不同数据集上的表现。

# A.4.3 使用PDV的零-shot方法与监督方法

我们比较了使用PDV增强的最先进的零-shot复合图像检索（ZS-CIR）方法与在FashionIQ和CIRR数据集上的监督方法。我们的评估包括早期的监督方法，如TIRG和ARTEMIS，以及最近的最先进的方法，如CCIN和SPRC。表S6中的比较结果表明，基于PDV的方法在与监督方法竞争时表现出显著的竞争力。基于PDV的方法显著超越了早期的监督基线，如在FashionIQ Dress上的TIRG $41.90 \%$ 对比 $14.13 \%$ $\mathbb{R} @ 10$ 和ARTEMIS $41.90 \%$ 对比 $25.68 \%$。虽然PDV方法的表现尚未能与最近的最先进方法如CCIN和SPRC相匹敌，但性能差距仍然相对较小——通常在FashionIQ上为7-9个百分点，在CIRR的平均分数上约为8-9个百分点（$72.85 \%$对比$81.66 \%$，针对CCIN）。考虑到使用PDV的ZS-CIR方法在没有人工标注训练数据的情况下运作，这一狭窄的性能差距尤其值得注意。这些结果表明，无监督方法正在达到一种有效性水平，使其成为监督方法的可行替代方案。

<table><tr><td>Backbone</td><td>Method</td><td>Dataset</td><td>Threshold</td><td>Filtered Ratio</td><td>Change in R@10</td><td>Change in R@50</td></tr><tr><td rowspan="3">ViT-B/32</td><td rowspan="3">CIReVL + PDV-F</td><td>Toptee</td><td>0.8</td><td>90.85%</td><td>-2.88%</td><td>-1.31%</td></tr><tr><td>Dress</td><td>0.8</td><td>82.31%</td><td>0.00%</td><td>-0.73%</td></tr><tr><td>Shirt</td><td>0.8</td><td>87.96%</td><td>-1.22%</td><td>-1.31%</td></tr><tr><td rowspan="3">ViT-L/14</td><td rowspan="3">Pic2Word + PDV-F</td><td>Toptee</td><td>0.75</td><td>85.62%</td><td>-1.24%</td><td>-0.94%</td></tr><tr><td>Dress</td><td>0.75</td><td>68.79%</td><td>0.00%</td><td>0.12%</td></tr><tr><td>Shirt</td><td>0.75</td><td>88.36%</td><td>-1.91%</td><td>-3.73%</td></tr></table>

T /ti:/ [计] 表, 终端, 测试, 磁道, 树, 真, 太, 万亿 [医] 温度, 胸的, 胸廓的, 眼球内压, 眼压, 垓, 千京, 兆兆 Definition: n. the 20th letter of the Roman alphabet

![](images/11.jpg)  
Figure S9. Performance Comparison: $\phi$ Angles by Method and Model Across FashionIQ Datasets

<table><tr><td rowspan="2">Method</td><td colspan="6">FashionIQ</td><td rowspan="2"></td><td colspan="5">CIRR</td></tr><tr><td>Dress</td><td></td><td>Shirt</td><td></td><td>Toptee</td><td>Mean</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td></td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>Mean</td></tr><tr><td>TIRG [26]</td><td>14.13</td><td>34.61</td><td>13.10</td><td>30.91</td><td>14.79</td><td>34.37</td><td>23.66</td><td>14.61</td><td>48.37</td><td>64.08</td><td>90.03</td><td>54.27</td></tr><tr><td>ARTEMIS S [9]</td><td>25.68</td><td>51.05</td><td>21.57</td><td>44.13</td><td>25.89</td><td>55.06</td><td>37.68</td><td>16.96</td><td>46.10</td><td>61.31</td><td>87.73</td><td>53.03</td></tr><tr><td>CLIP4CIR [5]</td><td>33.81</td><td>59.40</td><td>39.99</td><td>60.45</td><td>41.41</td><td>65.37</td><td>50.03</td><td>38.53</td><td>69.98</td><td>81.86</td><td>95.93</td><td>71.58</td></tr><tr><td>CompoDiff [12]</td><td>40.65</td><td>57.14</td><td>36.87</td><td>57.39</td><td>43.93</td><td>61.17</td><td>49.53</td><td>22.35</td><td>54.36</td><td>73.41</td><td>91.77</td><td>60.47</td></tr><tr><td>TG-CIR [27]</td><td>45.22</td><td>69.66</td><td>52.60</td><td>72.52</td><td>56.14</td><td>77.10</td><td>58.05</td><td>45.25</td><td>78.29</td><td>87.16</td><td>97.30</td><td>77.00</td></tr><tr><td>SPRC [2]</td><td>48.83</td><td>72.09</td><td>53.83</td><td>74.14</td><td>58.13</td><td>78.58</td><td>64.27</td><td>51.96</td><td>82.12</td><td>89.74</td><td>97.69</td><td>80.37</td></tr><tr><td>CCIN [24]</td><td>49.38</td><td>72.58</td><td>55.93</td><td>74.14</td><td>57.93</td><td>77.56</td><td>64.59</td><td>53.41</td><td>84.05</td><td>91.17</td><td>98.00</td><td>81.66</td></tr><tr><td>CIReVL + PDV LDRE + PDV</td><td>41.90 -</td><td>58.19 -</td><td>40.70 -</td><td>62.82 -</td><td>48.09 -</td><td>67.77 -</td><td>53.25 -</td><td>38.15 42.51</td><td>67.93 72.22</td><td>77.90 81.71</td><td>92.77 94.94</td><td>69.19 72.85</td></tr></table>

表 S6. PDV 与监督学习方法在 FashionIQ 和 CIRR 数据集上的比较。

# 补充材料 B. PDV 算法与代码

PDV算法在算法1中给出，代码显示在图S10中。PDV的实现非常直观，能够轻松与任何零样本分类（ZS-CIR）方法集成。

# 算法 1 计算 PDV 特征 1. 函数 CALCULATEPDVFEATURES(ftext, ftext.composed, fimage, $\alpha _ { i }$ $\alpha _ { t } , \beta )$ 2: ftext ← 归一化(ftext) 3: ftext_.composed ← 归一化(ftext.composed) 4: fimage ← 归一化(fimage) 5: pdv ← ftext.composed − ftext 6: fpDVI ← fimage + αi · pdv 7: fpDVT ← ftext + αt · pdv 8: fpDVF ← (1 − β) · fpDVI + β · fpDVT 9: 返回 归一化(fpDVF) 10: 结束函数

<table><tr><td>Fashion-IQ</td><td></td><td></td><td colspan="2">Shirt</td><td colspan="2">Dress</td><td colspan="2">Toptee</td><td colspan="2">Average</td></tr><tr><td>Backbone</td><td>Method</td><td>αI</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td rowspan="5">ViT-B/32</td><td>Image-only </td><td>-</td><td>6.92</td><td>14.23</td><td>4.46</td><td>12.19</td><td>6.32</td><td>13.77</td><td>5.90</td><td>13.37</td></tr><tr><td>Text-only</td><td>-</td><td>19.87</td><td>34.99</td><td>15.42</td><td>35.05</td><td>20.81</td><td>40.49</td><td>18.70</td><td>36.84</td></tr><tr><td>Image + Text </td><td>-</td><td>13.44</td><td>26.25</td><td>13.83</td><td>30.88</td><td>17.08</td><td>31.67</td><td>14.78</td><td>29.60</td></tr><tr><td>SEARLE + PDV-I</td><td>2</td><td>18.25</td><td>31.84</td><td>18.49</td><td>39.17</td><td>21.32</td><td>37.74</td><td>19.35</td><td>36.25</td></tr><tr><td>CIReVL + PDV-I</td><td>2</td><td>28.95</td><td>45.88</td><td>29.00</td><td>49.13</td><td>34.22</td><td>56.09</td><td>30.72</td><td>50.37</td></tr></table>

表 S7. PDV-I 在 FashionIQ 验证数据集上的表现。$^ \dagger$ 表示数据来源于原始论文。

<table><tr><td colspan="2">Dataset</td><td></td><td colspan="4">CIRCO</td><td colspan="6">CIRR</td></tr><tr><td></td><td>Metric</td><td></td><td colspan="4">mAP@k</td><td colspan="4">Recall@k</td><td colspan="2">Rs@k</td></tr><tr><td>Arch</td><td>Method</td><td>αI</td><td>k=5</td><td>k=10 k=25</td><td></td><td>k=50</td><td>k=1 k=5</td><td>k=10</td><td>k=50</td><td>k=1</td><td>k=2</td><td>k=3</td></tr><tr><td rowspan="6">ViT-B/32</td><td>Image-only </td><td>-</td><td>1.34</td><td>1.60 2.12</td><td>2.41</td><td>6.89</td><td>22.99</td><td>33.68</td><td>59.23</td><td>21.04</td><td>41.04</td><td>60.31</td></tr><tr><td>Text-only</td><td>-</td><td>2.56</td><td>2.67 2.98</td><td>3.18</td><td>21.81</td><td>45.22</td><td>57.42</td><td>81.01</td><td>62.24</td><td>81.13</td><td>90.70</td></tr><tr><td>Image + Text </td><td>-</td><td>2.65</td><td>3.25 4.14</td><td>4.54</td><td>11.71</td><td>35.06</td><td>48.94</td><td>77.49</td><td>32.77</td><td>56.89</td><td>74.96</td></tr><tr><td>SEARLE + PDV-I</td><td>1.5</td><td>4.77</td><td>5.23</td><td>6.31 6.82</td><td></td><td>16.65 42.53</td><td>55.16</td><td>81.42</td><td>44.68</td><td>67.78</td><td>82.94</td></tr><tr><td>CIReVL + PDV-I</td><td>2.0</td><td>10.29</td><td>10.80</td><td>12.23 12.93</td><td></td><td>27.18 56.53</td><td>67.76</td><td>87.64</td><td>59.81</td><td>79.59</td><td>90.15</td></tr><tr><td>LDRE + PDV-I</td><td>2.0</td><td>8.00</td><td>8.88 10.06</td><td>10.72</td><td>23.37</td><td>51.21</td><td>63.69</td><td>85.57</td><td>55.57</td><td>76.63</td><td>88.15</td></tr></table>

T 参考图像，而仅使用文本的方法则采用提示的文本嵌入。$\mathbf { B o l d } =$ 最佳结果，underline $=$ 次佳结果。数字来源于原始论文。

![](images/12.jpg)  
Figure S10. Python function for calculating PDV features.

![](images/13.jpg)  
Figure S11. Visualisation of the impact of $\alpha / \beta$ scaling on top-5 retrieval results. $\mathrm { C I R e V L }$ with ViT-B-32 Clip model is the baseline method top. Green and blue bounding boxes indicate true positives and near-true positives, respectively.

![](images/14.jpg)  
Figure S12. PDV-T: Impact of $\alpha$ scaling on Recall $@ 1 0$ (left) and Recall $\textcircled { \omega } 5 0$ (right) performance. Results shown for three baseline methods: CIReVL (top), Pic2Word (middle) and SEARLE (bottom).

![](images/15.jpg)  
Figure S13. PDV-I: Impact of $\alpha$ scaling on Recall $@ 1 0$ (left) and Recall $\textcircled { \omega } 5 0$ (right) performance. Results shown for three baseline methods CIReVL (top), Pic2Word (middle) and SEARLE (bottom).

![](images/16.jpg)  
Figure S14. PDV-F: Impact of $\beta$ scaling on Recall $@ 1 0$ (left) and Recall $\textcircled { a } 5 0$ (right) performance. Results shown for three baseline methods: CIReVL (top), Pic2Word (middle) and SEARLE (bottom).

# References

[1] Muhammad Umer Anwaar, Egor Labintcev, and Martin Kleinsteuber. Compositional learning of image-text query for image retrieval. In Proceedings of the IEEE/CVF Winter conference on Applications of Computer Vision, pages 1140 1149, 2021. 1   
[2] Yang Bai, Xinxing Xu, Yong Liu, Salman Khan, Fahad Khan, Wangmeng Zuo, Rick Siow Mong Goh, and Chun-Mei Feng. Sentence-level prompts benefit composed image retrieval. arXiv preprint arXiv:2310.05473, 2023. 13, 14   
[3] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. Zero-shot composed image retrieval with textual inversion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1533815347, 2023. 1, 2, 3, 6   
[4] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Conditioned and composed image retrieval combining and partially fine-tuning clip-based features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 49594968, 2022. 3   
[5] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Conditioned and composed image retrieval combining and partially fine-tuning clip-based features. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 49594968, June 2022. 14   
[6] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Effective conditioned and composed image retrieval combining clip-based features. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2146621474, 2022. 1, 3   
[7] Yanbei Chen and Loris Bazzani. Learning joint visual semantic matching embeddings for language-guided retrieval. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 2328, 2020, Proceedings, Part XXII 16, pages 136152. Springer, 2020. 1, 3   
[8] Niv Cohen, Rinon Gal, Eli A Meirom, Gal Chechik, and Yuval Atzmon. this is my unicorn, fluffy": Personalizing frozen vision-language representations. In European conference on computer vision, pages 558577. Springer, 2022. 1, 7   
[9] Ginger Delmas, Rafael Sampaio de Rezende, Gabriela Csurka, and Diane Larlus. Artemis: Attention-based retrieval with text-explicit matching and implicit similarity. arXiv preprint arXiv:2203.08101, 2022. 13, 14   
[10] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022. 3   
[11] Geonmo Gu, Sanghyuk Chun, Wonjae Kim, , Yoohoon Kang, and Sangdoo Yun. Language-only training of zeroshot composed image retrieval. In Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 1, 3   
[12] Geonmo Gu, Sanghyuk Chun, Wonjae Kim, HeeJae Jun, Yoohoon Kang, and Sangdoo Yun. Compodiff: Versatile composed image retrieval with latent diffusion. arXiv preprint arXiv:2303.11916, 2023. 3, 14   
[13] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 49044916. PMLR, 2021. 3   
[14] Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. Vision-by-language for training-free compositional image retrieval. International Conference on Learning Representations (ICLR), 2024. 1, 2, 3   
[15] Jongseok Kim, Youngjae Yu, Hoeseong Kim, and Gunhee Kim. Dual compositional learning in interactive image retrieval. In Proceedings of the AAAI Conference on Artificial Intelligence, 2021. 1   
[16] Seungmin Lee, Dongwan Kim, and Bohyung Han. Cosmo: Content-style modulation for image retrieval with text feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 802812, 2021. 1   
[17] Yikun Liu, Jiangchao Yao, Ya Zhang, Yanfeng Wang, and Weidi Xie. Zero-shot composed text-image retrieval. arXiv preprint arXiv:2306.07272, 2023. 3   
[18] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. Image retrieval on real-life images with pretrained vision-and-language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21252134, 2021. 1, 5   
[19] John A Nelder and Roger Mead. A simplex method for function minimization. The computer journal, 7(4):308313, 1965. 10   
[20] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 87488763. PMLR, 2021. 1, 3, 4   
[21] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500 22510, 2023. 3   
[22] Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister. Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19305 19314, 2023. 1, 2, 3   
[23] Yucheng Suo, Fan Ma, Linchao Zhu, and Yi Yang. Knowledge-enhanced dual-stream zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26951- 26962, 2024. 3   
[24] Likai Tian, Jian Zhao, Zechao Hu, Zhengwei Yang, Hao Li, Lei Jin, Zheng Wang, and Xuelong Li. Ccin: Compositional conflict identification and neutralization for composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 39743983, June 2025. 13, 14   
[25] Osman Tursun, Simon Denman, Sabesan Sivapalan, Sridha Sridharan, Clinton Fookes, and Sandra Mau. Componentbased attention for large-scale trademark retrieval. IEEE Transactions on Information Forensics and Security, 17:23502363, 2019. 6   
[26] Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, and James Hays. Composing text and image for image retrieval-an empirical odyssey. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 64396448, 2019. 1, 3, 13, 14   
[27] Haokun Wen, Xian Zhang, Xuemeng Song, Yinwei Wei, and Liqiang Nie. Target-guided composed image retrieval. In Proceedings of the 31st ACM international conference on multimedia, pages 915923, 2023. 14   
[28] Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris. Fashion iq: A new dataset towards retrieving images by natural language feedback. In Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pages 11307 11317, 2021. 1, 3, 5, 6   
[29] Zhenyu Yang, Shengsheng Qian, Dizhan Xue, Jiahong Wu, Fan Yang, Weiming Dong, and Changsheng Xu. Semantic editing increment benefits zero-shot composed image retrieval. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 12451254, 2024. 3   
[30] Zhenyu Yang, Dizhan Xue, Shengsheng Qian, Weiming Dong, and Changsheng Xu. Ldre: Llm-based divergent reasoning and ensemble for zero-shot composed image retrieval. In Proceedings of the 47th International ACM SI-GIR Conference on Research and Development in Information Retrieval, pages 8090, 2024. 3   
[31] Youngjae Yu, Seunghwan Lee, Yuncheol Choi, and Gunhee Kim. Curlingnet: Compositional learning between images and text for fashion iq data. arXiv preprint arXiv:2003.12299, 2020. 3   
[32] Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432, 2021. 3