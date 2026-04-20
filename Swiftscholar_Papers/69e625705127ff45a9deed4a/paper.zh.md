# V-Retrver：基于证据的主体性推理用于通用多模态检索

# 陈东阳 \* 1 王朝阳 \*2 苏德钊 3 肖夕 1 张泽宇 4 熊晶 5 李青 6 尚宇璋 2 阚世超 7

QHome: https://github.com/chendy25/V-Retrver HF: https://huggingface.co/V-Retrver

# 摘要

多模态大语言模型（MLLMs）最近被应用于通用多模态检索，其中链式思维（CoT）推理改善了候选项的重新排序。然而，现有的方法在很大程度上仍然以语言为驱动，依赖于静态的视觉编码，缺乏主动验证细粒度视觉证据的能力，这往往导致在视觉模糊的情况下进行推测性推理。我们提出了V-Retrver，这是一种基于证据的检索框架，将多模态检索重新构造成一个基于视觉检查的智能推理过程。V-Retrver使得MLLM能够在推理过程中通过外部视觉工具选择性地获取视觉证据，执行一种多模态交替推理过程，交替进行假设生成和针对性的视觉验证。为了训练这样的证据收集检索智能体，我们采用了一种课程学习策略，结合了监督推理激活、基于拒绝的精炼以及与证据对齐的强化学习目标。多项多模态检索基准实验表明检索准确性（平均提高$23.0\%$）、基于感知的推理可靠性和泛化能力均有一致改进，使得单一模型能够支持文本到图像、图像到文本和交替多模态查询等多样化的检索场景。近期的研究进一步表明，结合链式思维（CoT）推理可以通过增强可解释性和候选项区分度来提高检索性能。然而，尽管取得了这些进展，现有的基于CoT的检索系统仍然在根本上以语言为驱动，即使在检索决策关键依赖于视觉证据的情况下也是如此。这一限制在视觉模糊的检索场景中尤为明显，在这些情况下，候选图像共享相似的语义内容，但在细粒度视觉属性（如物体外观、风格或局部上下文）上有所不同。大多数当前基于MLLM的检索方法将视觉输入压缩为固定的嵌入或文本描述，迫使推理过程仅依赖语言来推断视觉差异。因此，当所需证据在视觉模态中时，模型往往会产生推测性或虚构的推理。即使是最近增强推理的检索框架，如Retrv-R1和MM-R5，也提高了文本推理的深度，但依然依赖于单次视觉编码，缺乏在推理过程中主动验证视觉假设的能力。

# 1. 引言

多模态大型语言模型（MLLMs）的快速发展显著推动了通用多模态检索的发展。为了解决这一差距，我们提出了V-Retrver，一个基于证据驱动的检索框架，它将多模态检索重新构建为一个基于视觉检查的智能推理过程。V-Retrver并不是将视觉表征视为静态输入，而是通过调用外部视觉工具，使MLLM能够在推理过程中选择性地获取视觉证据。通过多模态交错的思维链过程，该模型在假设生成和针对性的视觉验证之间交替进行，使其能够动态解决视觉歧义并逐步优化排名决策，如图1所示。

![](images/1.jpg)

白色沙发配有斑驳的靠垫，而不是深棕色沙发搭配白色靠垫和放置书籍。

![](images/2.jpg)  
Candidate 1

![](images/3.jpg)  
Candidate 2

![](images/4.jpg)  
Candidate 3

![](images/5.jpg)  
Candidate 4

![](images/6.jpg)  
Candidate 5

# 基于文本的思维链 目标是识别一个候选项，反映查询中的特定视觉转变：将深棕色沙发和白色枕头以及底下的书籍替换成白色沙发和斑点枕头。最初，我们可以进行预筛选来过滤候选项。候选项 2 完全符合文本中描述的负面例子——深棕色沙发与白色枕头和书籍在下面——因此它的相关性最低……候选项 3 是一款简单的沙发，缺乏斑点纹理和储物主题，使其不太合适。候选项 4 展示了一款传统的白色沙发，上面放着五颜六色的斑点枕头，地上有一本书。候选项 4 在其对白色沙发的明确呈现和斑点枕头的突出展示上表现突出，与文本请求高度一致。虽然候选项 1 也符合……候选项 4 对白色布料和生动枕头图案的美学强调使其成为这一特定转变的高度相关选择。接下来，候选项 3 的相关性高于其他选项，其次是候选项 5 的床型布置，最后是候选项 2 作为明确的反例。 <answer>[4, 1, 3, 5, 2]</answer>

# 多模态交错思维 > 根据您的要求，我将分析查询和候选图像，以提供一个排名列表……我将使用选择工具对它们进行更仔细的比较。 <tool_call {"name": "select_images", "arguments": {"target_images": [1,} </tool_call> <think> 从比较中：特征包括一款现代白色沙发和独特的斑驳/斑马图案的靠垫……我将对架子区域进行裁剪。</think> {"name": "crop_image", "arguments": {"bbox_2d": [290, 80, 380, 200], "target_image": } </tool_call>

![](images/7.jpg)  
Candidate 1

![](images/8.jpg)  
Candidate 4

![](images/9.jpg)  
Candidate 1   
xta u uTe reasoning and more reliable ranking decisions.

基于这一分析，最终排名如下 <answer>[1, 4, 3, 5, 2]/answer> 训练这样一个证据收集检索智能体不仅需要强大的推理能力，还需要检索性能与视觉工具使用之间的有效对齐。因此，我们采用了一种基于课程的训练策略，包括三个阶段。首先，冷启动监督阶段通过合成高质量的Chain of Thought (CoT) 数据来初始化模型，使其具备基本的推理能力和操作格式。其次，拒绝采样微调阶段巩固高质量的推理轨迹，并提高结构合规性。最后，我们引入了证据对齐策略优化（EAPO），通过群体相对策略优化（GRPO）实现（Guo等，2025），该策略强化了正确的排名决策，同时鼓励信息丰富的视觉验证，并抑制冗余工具使用。在通用多模态检索基准M-BEIR以及多个超出领域的数据集上的大量实验表明，V-Retrver在不同的检索设置中始终优于强基线。结果显示，V-Retrver达到了更高的检索准确率、更可靠的感知基础推理和更强的泛化能力，验证了交替视觉推理用于多模态检索的有效性。总之，我们的贡献主要有三点： • 我们提出了V-Retrver，一个基于证据驱动的智能检索框架，使多模态大语言模型（MLLMs）能够在多模态推理过程中主动获取视觉证据。 • 我们引入了一种基于课程的训练策略，具有证据对齐的强化学习目标，联合改善推理质量、排名准确性和高效的视觉工具使用。 • 在多个基准上的大量实验表明，V-Retrver始终优于现有方法，并能够很好地泛化到不同的多模态检索场景。

# 2. 相关工作

多模态大型语言模型。近年来，多模态大型语言模型（MLLMs）的快速发展推动了视觉感知与语言推理的深度融合，导致了一系列高性能开源模型的出现，特别是 LLaVA（刘等，2024；郭等，2024；张等，2025c；林等，2023a；李等，2023a）、Qwen-VL（白等，2023；王等，2024a；杨等，2024）和 InternVL（陈等，2024b；Ga0等，2024；陆等，2025；王等，2025a；b；c；d）系列。同时，一些大规模模型如 Flamingo（Alayrac 等，2022）、mPLUG-Owl（叶等，2023；2024b；a）和 GPT-4V（杨等，2023）追求更整体的视觉-语言建模范式，结合了包括专家混合架构（舒等，2024；李等，2025b；申等，2024）和图像生成组件（谢等，2024；徐等，2025a）在内的先进机制。然而，这些模型普遍缺乏诸如思维链和测试时可扩展性等推理能力（Muennighoff 等，2025；张等，2025b；陈等，2024a），在很大程度上依然将视觉感知与文本推理过程解耦。

多模态检索。最近深度学习的进展（Zhu et al., 2021; 2024; 2025a;c;b; Ji et al., 2024）显著推动了广泛的检索任务的进展，包括文本-图像跨模态检索（Pham et al., 2024; Fu et al., 2024; Zhang et al., 2020; Chun et al., 2021; Kim et al., 2023b;a）、组合图像检索（Baldrati et al., 2022; Saito et al., 2023; Gu et al., 2024; Suo et al., 2024; Baldrati et al., 2023）、多模态文档检索（Chen et al., 2023; Hu et al., 2023; Liu et al., 2023）和基于指令的图像检索（Wu et al., 2021; Zhang et al., 2024a; Asai et al., 2023）。在这些方法中，视觉语言模型（VLMs），特别是CLIP（Radford et al., 2021），在多模态检索场景中展现出了强大的有效性和可扩展性（Baldrati et al., 2022; Wei et al., 2024b; Sain et al., 2023; Pei et al., 2023; Jin et al., 2024）。例如，Kim等（Kim et al., 2023a）通过提示调优改进了CLIP，从而增强了其在多样化检索设置中的泛化能力。最近，多模态大语言模型（MLLMs）被引入以进一步提升检索性能（Liu et al., 2025; Jiang et al., 2024; Lin et al., 2024a; Zhou et al., 2024）。一些方法（Zhou et al., 2024; Lan et al., 2025; Lin et al., 2024a; Zhang et al., 2024b; Jian et al., 2025; Gu et al., 2025）利用从MLLMs提取的嵌入进行基于相似性的检索。其他方法，例如LamRA（Liu et al., 2025; Li et al., 2025a），将MLLMs作为重排序代理来优化候选列表并选择最相关的结果。Retrv-R1（Zhu et al., 2025d）通过强化学习为多模态检索任务增强了模型的文本推理能力。与以前的工作相比，我们引入了V-Retrver，一个基于证据的检索框架，它可以在推理过程中通过调用视觉工具自适应地调整其视觉探索策略，从而实现更灵活和有效的推理过程，并显著提升检索性能。

# 3. 方法

# 3.1. 问题描述

我们研究普遍多模态检索的问题。给定一个任意模态的查询 $q$（文本、图像或交织的多模态输入）和候选池 $\Omega = \{ c_{n} \}_{n=1}^{N}$，数据 $\hat{c} \in \Omega$。传统的多模态检索方法通常将此问题表述为静态相似性匹配或仅基于语言的重排序，依赖于固定的视觉表示。这种表述隐含地假设在推理之前，所有必要的视觉证据已经完全编码到嵌入或文本描述中。然而，这一假设在细粒度或视觉模糊的检索场景中是成立不了的，在这些场景中，细微的局部细节决定了相关性，无法仅从压缩的表示中可靠推断。为了应对这一限制，我们将多模态检索重构为基于证据的推理问题。在这一框架下，检索不再是单次推理过程，而是一个迭代决策过程，在此过程中，模型需在排名期间主动获取和验证视觉证据。具体来说，检索过程由三个紧密相关的步骤组成：（i）根据可用信息生成关于候选相关性的假设，（ii）选择性地检查视觉证据以解决不确定性，以及（iii）根据已验证的观察结果细化排名决策。这一视角自然引出了一种代理重排序范式，其中检索模型被赋予推理、检查和修正决策的能力，而不是被动地使用固定表示对候选进行评分。

# 3.2. V-Retrver 概述

基于上述概念，我们提出了 V-Retrver，一个基于证据的推理框架，用于通用多模态检索。如图 2 所示，V-Retrver 遵循从粗到细的检索流程，将高效的候选提议与计算密集型的基于证据的推理解耦。在第一阶段，嵌入模型 $\phi$ 将查询 $q$ 和每个候选 $c_{n}$ 编码到共享表示空间，根据相似度检索前 $K$ 个候选。我们采用与 LamRA (Liu 等，2025) 相同的方法构建嵌入模型 $\phi$。此阶段充当高效的候选提议机制，并大幅减少搜索空间：

$$
\begin{array} { r } { \mathcal { C } = \{ c _ { k } \} _ { k = 1 } ^ { K } , \quad K \ll N . } \end{array}
$$

在第二阶段，V-Retrver 采用推理智能体 $\theta$ 对减少的候选集 $\mathcal{C}$ 进行细粒度的重排序。重要的是，$\theta$ 不同于传统的重排序器，它不是基于静态特征运行的。而是作为一种智能证据收集模型设计，可以迭代推理、调用视觉检查工具，并根据新获取的视觉观察修正其排序决策。最终预测的结果为：

$$
{ \hat { c } } = \theta ( q , { \mathcal { C } } ) .
$$

![](images/10.jpg)  
Rejection sampling Fine-Tuning, and EAPO.

本节剩余部分将详细介绍在 V-Retrver 中实现基于证据的推理的核心机制，包括多模态交替推理、视觉工具和基于课程的训练策略。

# 3.3. 多模态交错证据推理

我们引入多模态交错证据推理（MIER），一种将文本假设生成与针对性视觉证据获取紧密交错的推理范式。与仅基于语言的思维链推理不同，MIER 允许中间推理步骤明确地基于按需获取的视觉观察进行基础性支撑。形式上，给定初始文本查询 $T_{0}$ 和候选图像集 $I_{0}$，推理智能体迭代产生输出：

$$
O _ { k } = f _ { \mathrm { M L L M } } \big ( \{ T _ { i } , C _ { i } , V _ { i } \} _ { i = 0 } ^ { k } \big ) ,
$$

其中 $T _ { i }$ 表示文本推理步骤， $C _ { i }$ 表示工具调用请求， $V _ { i }$ 代表工具返回的视觉证据。然后，解析器决定是否提取下一个推理步骤和工具请求 $( T _ { k + 1 } , C _ { k + 1 } )$，或者终止该过程并输出最终排名。如果调用了工具，则相应的视觉工具被执行并返回新的视觉证据 $V _ { k + 1 }$，该证据将附加到推理上下文中。这个过程产生了一个多模态推理轨迹：

$$
\tau = \{ T _ { 1 } , C _ { 1 } , V _ { 1 } , T _ { 2 } , C _ { 2 } , V _ { 2 } , . . . , T _ { n } , A _ { n } \} ,
$$

其中 $A _ { n }$ 表示候选者的最终排名列表。通过明确将中间推理步骤与动态获取的视觉证据相结合，MIER 减少了推测性推理和幻觉的发生，从而在视觉模糊情况下实现更可靠的排名决策。

# 3.4. 可视化工具

为支持MIER，我们为推理智能体配备了一套视觉工具，作为选择性视觉检查的外部感知接口。这些工具使模型能够控制在推理过程中观察的内容和关注的焦点。具体而言，我们实现了两个工具： （1）SELECT-IMAGE，使智能体能够在多个候选图像表现出高语义相似性时，选择一部分候选图像进行更仔细的检查。 （2）ZOOM-IN，对图像指定区域进行局部放大操作，允许对显著视觉属性（如物体、纹理或空间配置）进行细致分析。这些工具促进了检索过程中的选择性感知。智能体并不是一开始就编码所有视觉信息，而是根据需要动态扩展其视觉感受野，紧密模拟人类在检索过程中通过“再看一眼”关键细节来解决模糊候选项的行为。

# 3.5. 通过基于课程的代理学习训练 V-Retrver

训练 V-Retrver 需要将通用 MLLM 转变为能够进行稳定、基于证据推理和战略工具使用的智能体。为此，我们设计了一个三阶段的课程，逐步构建推理结构、可靠性和决策最优性。第一阶段：通过监督微调激活推理。我们从冷启动的监督微调阶段开始，以激活基本的推理和工具使用行为。由于现有的检索数据集缺乏注释的推理轨迹，我们使用 Qwen2.5-VL-72B-Instruct 合成多模态的链式思维数据。这些轨迹包括结构化的推理步骤和有效的工具调用模式。在应用基于规则的过滤以删除逻辑不一致或格式错误的样本后，基础模型使用标准的 SFT 损失进行微调。此阶段建立了基础的推理语法和工具意识，但尚未保证稳健性或最优的工具使用策略。第二阶段：推理可靠性的拒绝微调。尽管第一阶段激活了工具使用行为，但所得到的策略表现出高方差，并产生大量低质量的轨迹。为了提高推理可靠性，我们执行拒绝抽样微调（RSFT）。对于每个训练实例，我们采样多个推理轨迹，并仅保留那些严格满足格式约束并产生正确检索排名的轨迹。对这个过滤后的数据集进行微调显著提高了逻辑一致性和格式合规性，为强化学习提供了稳定的初始化。第三阶段：证据对齐的策略优化。虽然前两个阶段激活了结构化推理并改善了轨迹可靠性，但它们并未明确优化在检索过程中如何获取视觉证据。实际上，模型可能要么未充分利用视觉检查，要么冗余调用工具而未能改善排名决策。为了解决这一限制，我们引入了证据对齐的策略优化（EAPO），这是一个强化学习目标，明确将检索性能与有效且经济的视觉验证行为对齐。EAPO 将多模态检索公式化为轨迹级决策问题，其中每个推理轨迹 $o _ { i }$ 基于排名质量和证据利用进行评估。具体而言，我们定义了复合奖励：

$$
R _ { i } = \alpha r _ { \mathrm { f o r m a t } } ( o _ { i } ) + \beta r _ { \mathrm { r a n k } } ( o _ { i } ) + r _ { \mathrm { t o o l } } ( o _ { i } ) ,
$$

三个组成部分分别促进结构正确性、准确排序和信息丰富的可视检查。下面，我们详细说明每个奖励项。格式合规奖励。格式合规奖励 $r _ { \mathrm { f o r m a t } }$ 确保模型遵循所需的推理和输出协议，这对于具有结构化多模态输出的稳定策略优化至关重要。设 $\Omega _ { \mathrm { t a g } }$ 为输出正确被预定义的 <think $>$ 和 <answer $>$ 标签包围的轨迹集合，设 $\Omega _ { \mathrm { l i s t } }$ 为最终答案严格遵循所需整数排序列表格式的轨迹集合。我们定义：

$$
r _ { \mathrm { f o r m a t } } ( o _ { i } ) = \frac { 1 } { 2 } \mathbb { I } _ { \{ o _ { i } \in \Omega _ { \mathrm { t a g } } \} } + \frac { 1 } { 2 } \mathbb { I } _ { \{ o _ { i } \in \Omega _ { \mathrm { l i s t } } \} } ,
$$

其中 $\mathbb { I } _ { \{ \cdot \} }$ 是指示函数。此项主要作为稳定信号，防止畸形轨迹主导策略更新。软排序奖励。为了缓解检索任务中二元正确性信号的稀疏性，我们引入了软排序奖励 $r _ { \mathrm { r a n k } }$，该奖励根据正确候选的相对位置提供密集反馈。设 $k$ 表示在预测轨迹列表 $o _ { i }$ 中真实候选的 1-索引排名。如果正确候选未出现在前 $K _ { r }$ 个位置或输出无效，则奖励设为零。否则，奖励定义为：

$$
r _ { \mathrm { r a n k } } ( o _ { i } ) = \exp \left( - \frac { ( k - 1 ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,
$$

其中 $\sigma$ 控制对排名错误的敏感度。该公式鼓励智能体持续改善排名质量，而不是仅优化稀疏的 top-1 信号。工具使用奖励。工具使用奖励 $r _ { \mathrm { t o o l } }$ 直接影响智能体的证据获取行为，鼓励仅在其有助于正确决策时进行视觉检查，同时抑制冗余或过度的工具使用。设 $N _ { \mathrm { t o o l } }$ 表示轨迹 $o _ { i }$ 中有效视觉工具调用的数量，$k$ 为正确候选者的排名位置。我们定义：

$$
\begin{array} { r l } & { r _ { \mathrm { t o o l } } ( o _ { i } ) = \eta \cdot \mathbb { I } _ { \{ k = 1 \} } \cdot \mathbb { I } _ { \{ N _ { \mathrm { t o o l } } > 0 \} } } \\ & { ~ - ~ \rho \cdot \operatorname* { m a x } ( 0 , N _ { \mathrm { t o o l } } - \tau ) , } \end{array}
$$

其中 $\eta$ 鼓励成功的基于证据的验证，$\rho$ 处罚过多的工具调用，而 $\tau$ 则指定了容忍阈值。此设计明确编码了有效工具使用而非频繁使用应当获得奖励的原则。政策优化。我们使用群体相对策略优化（GRPO） (Guo et al., 2025) 实现 EAPO。给定一组为同一查询采样的 $G$ 条轨迹，我们计算标准化优势：

$$
A _ { i } = { \frac { R _ { i } - \operatorname* { m e a n } ( R ) } { \operatorname* { s t d } ( R ) } } .
$$

最终的优化目标是：

$$
\mathcal { I } _ { \mathrm { E A P O } } ( \theta ) = \mathbb { E } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { \pi _ { \theta } ( o _ { i } | q ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { i } | q ) } A _ { i } - \lambda \mathrm { K L } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) \Bigg ] .
$$

通过EAPO，模型不仅学习如何进行排名，还学习如何以及何时获取视觉证据，以支持可靠和高效的检索决策。

# 4. 实验

# 4.1. 实验设置

Table 1. Summary of the evaluation benchmarks. The benchmarks are categorized into Supervised and Zero-shot settings. # Queries represents the number of test queries, and # Candidates denotes the number of test candidates per query.   

<table><tr><td>Benchmark</td><td># Queries</td><td># Candidates</td></tr><tr><td>Supervised</td><td></td><td></td></tr><tr><td>M-BEIR (Wei et al., 2024a)</td><td>190K</td><td>5.6M</td></tr><tr><td>Zero-shot</td><td></td><td></td></tr><tr><td>CIRCO (Baldrati et al., 2023)</td><td>800</td><td>120K</td></tr><tr><td>GeneCIS (Vaze et al., 2023)</td><td>8K</td><td>10 ∼ 15</td></tr><tr><td>Visual Storytelling (Huang et al., 2016)</td><td>5K</td><td>8K</td></tr><tr><td>Visual Dialog (Das et al., 2017)</td><td>2K</td><td>2K</td></tr><tr><td>Multi-round FashionIQ (Yuan &amp; Lam, 2021)</td><td>2.4K</td><td>6.2K</td></tr></table>

数据集和指标。我们使用 M-BEIR（Wei 等，2024a）数据集进行训练。M-BEIR 数据集涵盖了 10 个不同检索数据集中的 8 个独特检索任务，包含总共 110 万个训练样本。如表 1 所示，为了评估 V-Retrver 在各种检索任务中的多样性，我们在 M-BEIR 测试集上进行评估。此外，我们还研究了 V-Retrver 在其他未见过的数据集上的泛化能力，包括 CIRCO（Baldrati 等，2023）、GeneCIS（Vaze 等，2023）、视觉讲故事（Huang 等，2016）、视觉对话（Das 等，2017）等。我们遵循为每个数据集建立的标准评估指标。我们主要使用 Recall $@ \mathrm { K }$ 作为检索任务的评估指标。此外，对于像 CIRCO 这样的特定数据集，我们报告 $\mathbf { M A P @ 5 }$ 以提供对排名质量的更细致评估。

实验设置与基线。我们建立了三个不同的实验设置：(i) 为了验证我们方法在一系列检索任务中的通用性，我们在 M-BEIR 基准中的所有 8 个任务上训练 V-Retrver，并评估其在测试集上的性能。基线方面，我们将我们的模型与以下内容进行比较：(1) 基础 VLM（如 Qwen2.5-VL、CLIP、BLIP）；(2) 微调的通用检索器，如 UniIR ${\mathrm{BLIP}}_{\mathrm{FF}}$ 和 UniIR ${\mathrm{CLIP}}_{\mathrm{SF}}$；(3) 最近的增强推理模型和通用检索器，包括 Vision-R1（Huang 等，2025），VLM-R1（Shen 等，2025），MM-Embed（Lin 等，2024a），LamRA（Liu 等，2025）和 U-MARVEL（Li 等，2025a），以展示我们视觉 CoT 框架的优势。(ii) 为了评估在之前未见检索数据集上的泛化能力，我们在 5 个训练期间未遇到的数据集上进行零样本实验。在这种情况下，基线包括一系列通用检索器，如 E5-V、MagicLens 和 MM-Embed。(iii) 为了研究在未见检索任务上的泛化能力，我们故意排除三个检索任务的数据：图像到图像检索、文本图像到文本检索以及文本图像到文本图像检索。然后在剩余的五个任务上进行训练，并评估这些被排除任务的表现。 滑动窗口重排序。遵循粗到细的范式，V-Retrver 采用滑动窗口策略来重新排序初始检索结果。具体而言，我们首先使用 MLLM 基嵌入模型 $\phi$ 检索前 $K$ 个候选项，如第 3.1 节所述。受到（Zhang 等，2025a）中迭代重排序方法的启发，我们将窗口大小设置为 $K=20$，步长为 10，以有效识别最相关的项。这使得每个查询在逐步细化结果到最终排名时，进行四次 MLLM 推理调用。该滑动窗口方法使我们的模型能够在大型候选池中进行细粒度的多模态推理，同时保持可管理的计算开销。

实施细节。我们的模型基于 Qwen2.5-VL-7B-Instruct（Bai 等，2025）进行初始化。在 SFT 和拒绝微调阶段，我们利用 LLaMA-Factory（Zheng 等，2024）框架，在 8 个 A800 GPU 上进行训练，批次大小为 64，学习率为 $1 \times 1 0 ^ { - 5 }$，训练两个周期。RL 训练基于 verl-tool（Jiang 等，2025）框架，该框架扩展了 verl（Sheng 等，2024）和 vLLM（Kwon 等，2023）的功能，提供了多模态工具增强的多轮训练和评估的专门支持。在 RL 阶段，模型训练 1 个周期，学习率为 $1 \times 1 0 ^ { - 6 }$，每个查询使用 8 次推演。在所有训练阶段中，视觉编码器保持冻结状态，而语言模型进行微调。输入 MLLM $\theta$ 的候选数量 $K$ 设置为 20。在 M-BEIR 评估中，实验在本地池中进行，V-Retrver 对前 50 个结果进行重排序。对于未见数据集的实验，对前 10 个结果应用重排序。软排名敏感度 $\sigma$ 设置为 1.0，排名奖励阈值 $K _ { r }$ 设置为 5。奖励加权因子 $\alpha$ 和 $\beta$ 固定为 0.2 和 0.8。关于工具使用机制，方程 (4) 中的超参数配置为 $\eta = 0 . 2$，$\rho = 0 . 1$ 和 $\tau = 1$。此外，我们在 EAPO 目标中使用 KL 惩罚系数 $\lambda = 0$。

# 4.2. 主要结果

在 M-BEIR 上的表现。如表 2 所示，V-Retrver-7B 在 M-BEIR 基准上建立了新的最先进水平，平均召回率为 $69.7\%$。这比最强基线 U-MARVEL-7B 的 $64.8\%$ 提升了 $+4.9\%$。我们的方法在需要细粒度视觉细节的场景中尤其明显，如在 FIQ 和 CIRR 上的 $(q^{i}, q^{t}) \to c^{i}$。相比之下，V-Retrver 在 FIQ 上的得分为 $51.2\%$，在 CIRR 上为 $73.5\%$。这些得分大幅超越了 U-MARVEL-7B，后者在这两项上的得分分别为 $38.2\%$ 和 $63.2\%$。这些结果证实了多模态交错链式推理方法可以有效提升模型的信息检索能力。

Table 2. Comparison with other methods on M-BEIR test set. $\operatorname { R @ K }$ refers to the Recall $@ \mathrm { K }$ metric. $q ^ { t } , q ^ { i }$ $c ^ { t }$ and $c ^ { i }$ denote the text Fashion200K, InfoS for InfoSeek, and FIQ for FashionIQ. The best results are highlighted in bold.   

<table><tr><td rowspan="3">Models</td><td colspan="3">qt → ci</td><td colspan="2">qt → ct</td><td colspan="2">qt → (ci, ct)</td><td colspan="2">qi → ct</td><td>qi → ci</td><td colspan="2">(qi, qt) → ct</td><td>(qi, qt) → ci</td><td colspan="2">(qi, qt) → (ci, ct)</td><td></td><td rowspan="3">Avg.</td></tr><tr><td>VN</td><td>COCO F200K WebQA EDIS WebQA VN</td><td></td><td></td><td></td><td></td><td></td><td></td><td>COCO F200K NIGHTS OVEN InfoS</td><td></td><td></td><td></td><td>FIQ</td><td></td><td>CIRR OVEN</td><td>InfoS</td></tr><tr><td>R@5</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@5</td><td>R@5</td><td></td><td>R@10 R@5</td><td>R@5</td><td>R@5</td><td></td></tr><tr><td>CLIP-L (Radford et al., 2021)</td><td>43.3</td><td>61.1</td><td>6.6</td><td>36.2</td><td>43.3</td><td>45.1</td><td>41.3</td><td>79.0</td><td>7.7</td><td>26.1</td><td>24.2</td><td>20.5</td><td>7.0</td><td>13.2</td><td>38.8</td><td>26.4</td><td>32.5</td></tr><tr><td>SigLIP (Zhai et al., 2023)</td><td>30.1</td><td>75.7</td><td>36.5</td><td>39.8</td><td>27.0</td><td>43.5</td><td>30.8</td><td>88.2</td><td>34.2</td><td>28.9</td><td>29.7</td><td>25.1</td><td>14.4</td><td>22.7</td><td>41.7</td><td>27.4</td><td>37.2</td></tr><tr><td>BLIP (Li et al., 2022)</td><td>16.4</td><td>74.4</td><td>15.9</td><td>44.9</td><td>26.8</td><td>20.3</td><td>17.2</td><td>83.2</td><td>19.9</td><td>27.4</td><td>16.1</td><td>10.2</td><td>2.3</td><td>10.6</td><td>27.4</td><td>16.6</td><td>26.8</td></tr><tr><td>BLIP2 (Li et al., 2023b)</td><td>16.7</td><td>63.8</td><td>14.0</td><td>38.6</td><td>26.9</td><td>24.5</td><td>15.0</td><td>80.0</td><td>14.2</td><td>25.4</td><td>12.2</td><td>5.5</td><td>4.4</td><td>11.8</td><td>27.3</td><td>15.8</td><td>24.8</td></tr><tr><td>UnilR-BLIPF (Wei et al., 2024b)</td><td>23.4</td><td>79.7</td><td>26.1</td><td>80.0</td><td>50.9</td><td>79.8</td><td>22.8</td><td>89.9</td><td>28.9</td><td>33.0</td><td>41.0</td><td>22.4</td><td>29.2</td><td>52.2</td><td>55.8</td><td>33.0</td><td>46.8</td></tr><tr><td>UnilR-CLIPsF (Wei et al., 202</td><td>42.6</td><td>81.1</td><td>18.0</td><td>84.7</td><td>59.4</td><td>78.7</td><td>43.1</td><td>92.3</td><td>18.3</td><td>32.0</td><td>45.5</td><td>27.9</td><td>24.4</td><td>44.6</td><td>67.6</td><td>48.9</td><td>50.6</td></tr><tr><td>Qwen2.5-VL-3B (Bai et al., 2025)</td><td>36.0</td><td>67.8</td><td>16.1</td><td>69.5</td><td>45.2</td><td>61.7</td><td>23.3</td><td>82.3</td><td>12.0</td><td>20.9</td><td>36.7</td><td>22.3</td><td>24.3</td><td>53.5</td><td>56.4</td><td>49.8</td><td>42.4</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>40.2</td><td>71.9</td><td>20.3</td><td>71.9</td><td>49.4</td><td>64.5</td><td>29.3</td><td>84.6</td><td>19.4</td><td>25.5</td><td>42.4</td><td>32.1</td><td>25.0</td><td>55.1</td><td>60.8</td><td>54.9</td><td>46.7</td></tr><tr><td>Vision-R1-7B (Huang et al., 20251.9</td><td></td><td>75.0</td><td>22.0</td><td>70.6</td><td>51.3</td><td>69.1</td><td>35.4</td><td>85.1</td><td>22.4</td><td>25.9</td><td>48.8</td><td>44.0</td><td>29.2</td><td>57.7</td><td>66.2</td><td>59.0</td><td>50.2</td></tr><tr><td>VLM-R1-7B (Shen et al., 2025)</td><td>40.5</td><td>77.2</td><td>22.5</td><td>72.3</td><td>50.0</td><td>67.9</td><td>36.2</td><td>86.3</td><td>20.9</td><td>26.4</td><td>48.8</td><td>37.5</td><td>29.9</td><td>57.4</td><td>64.0</td><td>62.3</td><td>50.0</td></tr><tr><td>MM-Embed-7B (Lin et al., 2024a)41.0</td><td></td><td>71.3</td><td>17.1</td><td>95.9</td><td>68.8</td><td>85.0</td><td>41.3</td><td>90.1</td><td>18.4</td><td>32.4</td><td>42.1</td><td>42.3</td><td>25.7</td><td>50.0</td><td>64.1</td><td>57.7</td><td>52.7</td></tr><tr><td>LamRA-7B (Liu et al., 2025)</td><td>48.0 49.4</td><td>85.2 85.6</td><td>32.9</td><td>96.7</td><td>75.8</td><td>87.7</td><td>48.6</td><td>92.3</td><td>36.1</td><td>33.5</td><td>59.2</td><td>64.1</td><td>37.8</td><td>63.3</td><td>79.2</td><td>78.3</td><td>63.7</td></tr><tr><td>U-MARVEL-7B (Li et al., 2025a)</td><td></td><td></td><td>34.2</td><td>98.5</td><td>81.4</td><td>89.4</td><td>50.5</td><td>88.4</td><td>37.7</td><td>34.7</td><td>63.7</td><td>62.9</td><td>38.2</td><td>63.2</td><td>80.8</td><td>78.9</td><td>64.8</td></tr><tr><td>V-Retrver-7B</td><td>51.8 87.5</td><td></td><td>40.3</td><td>96.9</td><td>82.9</td><td>90.2</td><td>52.2</td><td>94.8</td><td>37.8</td><td>39.8</td><td>69.8</td><td>73.2 51.2</td><td></td><td>73.5</td><td>87.8</td><td>85.0</td><td>69.7</td></tr></table>

Table 3. Experimental results on unseen datasets. $q ^ { \mathrm { d i a l o g } }$ and $( q ^ { i } \oplus q ^ { t } )$ refer to the dialog queries and multi-interleaved imagetext queries, respectively.   

<table><tr><td rowspan="3">Models</td><td colspan="2">(qi,qt) → ci</td><td colspan="3">qdialog → ci (qi  qt) → ci</td></tr><tr><td>CIRCO</td><td>GeneCIS</td><td>VisD</td><td></td><td>VIST MT-FIQ</td></tr><tr><td>MAP@5</td><td>R@1</td><td>R@1</td><td>R@1</td><td>R@5</td></tr><tr><td>CLIP-L (Radford et al., 2021)</td><td>4.0</td><td>13.3</td><td>23.7</td><td>0.6</td><td>17.7</td></tr><tr><td>UnilR-CLIP (Wei et al., 2024b)</td><td>12.5</td><td>16.8</td><td>26.8</td><td>0.6</td><td>39.4</td></tr><tr><td>E5-V (Jiang et al., 2024)</td><td>24.8</td><td>18.5</td><td>54.6</td><td>10.0</td><td>19.2</td></tr><tr><td>MagicLens-L (Zhang et al., 2024a)</td><td>29.6</td><td>16.3</td><td>28.0</td><td>3.3</td><td>22.6</td></tr><tr><td>MM-Embed-7B (Lin et al., 2024a)</td><td>35.5</td><td>22.9</td><td>64.7</td><td>25.7</td><td>59.0</td></tr><tr><td>LamRA-7B (Liu et al., 2025)</td><td>42.8</td><td>24.8</td><td>70.9</td><td>28.6</td><td>63.9</td></tr><tr><td>V-Retrver-7B</td><td>48.2</td><td>30.7</td><td>75.1</td><td>31.2</td><td>68.3</td></tr></table>

未见数据集的泛化能力。表3中的零-shot评估结果突显了我们推理框架在训练过程中未接触的数据集上的稳健性。V-Retrver的表现始终优于专业模型和通用MLLM。在具有明显领域转变的CIRCO上，V-Retrver达到了 $\mathbf { M A P @ 5 }$ 的48.2。这显著超过了专业的MM-Embed-7B（35.5）和LamRA-7B（42.8）。同样，在GeneCIS上，我们的模型获得了 $\mathbf { R } \ @ 1$ 为30.7，而LamRA-7B为24.8。我们将这种泛化能力归因于强化学习。对保留任务的稳健性。为验证任务级适应性，我们评估了V-Retrver在特定模态组合在训练期间被严格排除的检索任务上的表现。如表4所示，即使没有接触过这些格式，该模型的平均召回率达到了 $6 1 . 1 \%$ ，显著优于LamRA-7B的 $5 0 . 9 \% $，超出幅度为 $1 0 . 2 \% $。这些结果实证表明，MIER框架有效地将推理过程与特定输入类型解耦，使模型能够利用交错证据进行准确检索，即使在具有挑战性的零-shot场景中。

Table 4. Experimental results on held-out tasks. \* indicates that training is performed on the remaining tasks, w/o any exposure to the three held-out tasks.   

<table><tr><td rowspan="3">Models</td><td colspan="5">qi → ci (qi, qt) → ct (qi, qt) → (ci, ct)</td><td rowspan="3">Avg.</td></tr><tr><td></td><td>NIGHTS OVEN InfoS</td><td></td><td>OVEN</td><td>InfoS</td></tr><tr><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td></tr><tr><td>Supervised</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UnilR-BLIPFF (Wei et al., 2024b)</td><td>33.0</td><td>41.0</td><td>22.4</td><td>55.8</td><td>33.0</td><td>37.0</td></tr><tr><td>UnilR-CLIPsF (Wei et al., 2024b)</td><td>32.0</td><td>45.5</td><td>27.9</td><td>67.6</td><td>48.9</td><td>44.4</td></tr><tr><td>Zero-shot</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>20.3</td><td>38.5</td><td>40.4</td><td>53.6</td><td>44.9</td><td>39.5</td></tr><tr><td>Vision-R1-7B (Huang et al., 2025)</td><td>22.9</td><td>39.8</td><td>42.9</td><td>57.4</td><td>46.5</td><td>41.9</td></tr><tr><td>LamRA-7B* (Liu et al., 2025)</td><td>29.2</td><td>46.9</td><td>54.2</td><td>65.1</td><td>59.1</td><td>50.9</td></tr><tr><td>V-Retrver-7B*</td><td>36.2</td><td>57.8</td><td>65.9</td><td>75.3</td><td>70.3</td><td>61.1</td></tr></table>

# 4.3. 消融研究与分析

训练阶段的影响。表6展示了各训练阶段的消融结果。无 SFT、RSFT 和 RL 的行表示直接提示未训练的主干网络进行工具使用，导致性能骤降至 $4 5 . 8 \%$，甚至低于 Qwen2.5-VL-7B 基线 $( 4 7 . 2 \% )$，这表明没有对齐的零-shot 工具调用是无效的。无 RSFT 和 RL 的设置仅包含 SFT 阶段，激活了基本的工具使用能力，使平均召回率提升至 $5 9 . 4 \%$。仅去掉 RSFT（无 RSFT）意味着模型采用 SFT 和 RL 进行训练，跳过拒绝采样阶段，达到 $6 6 . 3 \%$。无 RL 配置使用 SFT 和 RSFT，但省略强化学习，结果为 $6 0 . 9 \%$。最后，完整的流程达到最高性能 $6 7 . 2 \%$。这些结果突显了结构化课程学习的重要性，因为每个阶段都针对前一个阶段的具体不足。

![](images/11.jpg)  
Figure 3. RL Training curves.   
Table 5. Ablation study on visual tool-use mechanism. We compare the proposed multimodal interleaved CoT (with Visual Tool) against a text-only reasoning baseline (w/o Visual Tool) under the same RL training framework.

视觉工具的有效性。为了隔离工具使用的影响，我们使用基于文本的链式推理，采用端到端强化学习训练 Qwen2.5-VL-7B-Instruct 的一个变体，使用相同的训练数据集（无工具的强化学习）。如表5所示，仅使用文本的变体平均召回率为 $61.8\%$，而 V-Retrver 达到 $67.2\%$。研究结果确认，结合视觉工具能够提供补充性的、高保真的见解，而仅靠文本推理无法从静态表示中捕捉到这些见解。具体而言，主动放大或选择图像的能力使模型能够解决细粒度的模糊，这种模糊通常在压缩的视觉嵌入中丢失，证明对于真正精确的多模态检索是不可或缺的。

# 4.4. 训练曲线

Table 6. Ablation study on training stages and components. We investigate the impact of Cold Start (SFT), Rejection Sampling Fine-Tuning (RSFT), and Reinforcement Learning (RL) using Qwen2.5-VL-7B as the backbone.   

<table><tr><td rowspan="3">Variants</td><td>qt → ci</td><td>qi → ct</td><td>(qi, qt) → ci</td><td>(qi, qt) → ct</td><td rowspan="3">Avg.</td></tr><tr><td>COCO</td><td>F200K</td><td>CIRR</td><td>OVEN</td></tr><tr><td>R@5</td><td>R@10</td><td>R@5</td><td>R@5</td></tr><tr><td rowspan="2">Qwen2.5-VL-7B (Bai et al., 2025) RL w/o tool</td><td>71.9</td><td>19.4</td><td>55.1</td><td>42.4</td><td>47.2</td></tr><tr><td>84.1</td><td>33.2</td><td>66.5</td><td>63.2</td><td>61.8</td></tr><tr><td>V-Retrver-7B</td><td>87.5</td><td>37.8</td><td>73.5</td><td>69.8</td><td>67.2</td></tr></table>

图3展示了在强化学习训练过程中的排名准确性、推理密度和工具使用效率的演变。随着训练的推进，模型的检索准确性总体呈上升趋势，表明EAPO有效提升了模型的感知驱动推理能力。关于工具使用行为，我们观察到有效工具调用的数量在初始阶段略低于总调用次数。这表明尽管模型在SFT和RSFT阶段获得了基本的工具使用能力，但仍然偶尔会出现格式不一致或逻辑错误。随着训练的继续，这两条曲线逐渐收敛，表明强化学习进一步增强了工具使用的稳健性，并消除了错误调用。这种收敛意味着策略优化过程成功惩罚了幻觉工具行为，引导智能体更加严谨地执行工具协议。此外，平均响应长度和工具频率在稳定之前有所减少；这表明模型学会了自主判断视觉证据的必要性，有效抑制了冗余推理，并将注意力集中在通过更有根据和有目的的多模态轨迹解决关键视觉歧义上。

<table><tr><td rowspan="3">Training Stage</td><td>qt → ci</td><td>qi → ct</td><td>(qi, qt) → ci</td><td>(qi,qt) → ct</td><td rowspan="3">Avg.</td></tr><tr><td>COCO</td><td>F200K</td><td>CIRR</td><td>OVEN</td></tr><tr><td>R@5</td><td>R@10</td><td>R@5</td><td>R@5</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>71.9</td><td>19.4</td><td>55.1</td><td>42.4</td><td>47.2</td></tr><tr><td>w/o SFT &amp; RSFT &amp; RL</td><td>71.5</td><td>18.1</td><td>53.4</td><td>40.2</td><td>45.8</td></tr><tr><td>w/o RSFT &amp; RL</td><td>83.2</td><td>31.6</td><td>63.7</td><td>59.0</td><td>59.4</td></tr><tr><td>w/o RSFT</td><td>87.2</td><td>37.3</td><td>72.4</td><td>68.3</td><td>66.3</td></tr><tr><td>w/o RL</td><td>83.9</td><td>32.8</td><td>65.3</td><td>61.5</td><td>60.9</td></tr><tr><td>V-Retrver-7B</td><td>87.5</td><td>37.8</td><td>73.5</td><td>69.8</td><td>67.2</td></tr></table>

# 5. 结论

在本文中，我们提出了 V-Retrver，这是一个针对通用多模态检索的证据驱动 MLLM 框架。V-Retrver 采用多模态交错的推理链（CoT）机制，使模型能够通过视觉工具调用动态检查和验证候选图像，从而实现候选结果列表的更细粒度排名。我们采用了一个三阶段的训练流程来提升多模态交错 CoT 推理能力。大量实验结果表明，V-Retrver 在模型效能和任务普适性方面都取得了显著的提升。我们认为 V-Retrver 是有效引入代理 MLLM 以增强下游多模态任务的重要一步，为构建具有先进推理能力的通用代理 MLLM 打下了坚实的基础。

# 影响声明

本文提出的工作旨在推动机器学习领域的发展。我们的工作可能带来许多社会影响，但我们认为在此处并不需要特别强调。

# References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for fewshot learning. Advances in neural information processing systems, 35:2371623736, 2022.

Asai, A., Schick, T., Lewis, P., Chen, X., Izacard, G., Riedel, S., Hajishirzi, H., and Yih, W.-t. Task-aware retrieval with instructions. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 36503675, 2023.

Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Baldrati, A., Bertini, M., Uricchio, T., and Del Bimbo, A. Effective conditioned and composed image retrieval combining clip-based features. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2146621474, 2022.

Baldrati, A., Agnolucci, L., Bertini, M., and Del Bimbo, A. Zero-shot composed image retrieval with textual inversion. In Proceedings of the International Conference on Computer Vision, 2023.

Chen, Y., Hu, H., Luan, Y., Sun, H., Changpinyo, S., Ritter, A., and Chang, M.-W. Can pre-trained vision and language models answer visual information-seeking questions? In Proceedings of the Conference on Empirical Methods in Natural Language Processinng, 2023.

Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024a.

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2418524198, 2024b.

Chen, Z., Xu, C., Qi, Y., and Guo, J. Mllm is a strong reranker: Advancing multimodal retrieval-augmented generation via knowledge-enhanced reranking and noiseinjected training. arXiv preprint arXiv:2407.21439, 2024c.

Chun, S., Oh, S. J., De Rezende, R. S., Kalantidis, Y., and Larlus, D. Probabilistic embeddings for cross-modal retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 84158424, 2021.

Das, A., Kottur, S., Gupta, K., Singh, A., Yadav, D., Moura, J. M., Parikh, D., and Batra, D. Visual dialog. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.

Fu, Z., Zhang, L., Xia, H., and Mao, Z. Linguistic-aware patch slimming framework for fine-grained cross-modal alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26307 26316, 2024.

Gao, Z., Chen, Z., Cui, E., Ren, Y., Wang, W., Zhu, J., Tian, H., Ye, S., He, J., Zhu, X., et al. Mini-internvl: a flexibletransfer pocket multi-modal model with $5 \%$ parameters and $90 \%$ performance. Visual Intelligence, 2(1):117, 2024.

Gu, G., Chun, S., Kim, W., Kang, Y., and Yun, S. Languageonly training of zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1322513234, 2024.

Gu, T., Yang, K., Feng, Z., Wang, X., Zhang, Y., Long, D., Chen, Y., Cai, W., and Deng, J. Breaking the modality barrier: Universal embedding learning with multimodal llms. In Proceedings of the 33rd ACM International Conference on Multimedia, pp. 28602869, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Guo, Z., Xu, R., Yao, Y., Cui, J., Ni, Z., Ge, C., Chua, T.-S., Liu, Z., and Huang, G. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. In European Conference on Computer Vision, pp. 390406. Springer, 2024.

Hu, H., Luan, Y., Chen, Y., Khandelwal, U., Joshi, M., Lee, K., Toutanova, K., and Chang, M.-W. Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities. In Proceedings of the International Conference on Computer Vision, 2023.

Huang, T.-H., Ferraro, F., Mostafazadeh, N., Misra, I., Agrawal, A., Devlin, J., Girshick, R., He, X., Kohli, P., Batra, D., et al. Visual storytelling. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics, 2016.

Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Hu, Y., and Lin, S. Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749, 2025.

Ji, D., Zhao, F., Zhu, L., Jin, W., Lu, H., and Ye, J. Discrete latent perspective learning for segmentation and detection. arXiv preprint arXiv:2406.10475, 2024.

Jian, W., Zhang, Y., Liang, D., Xie, C., He, Y., Leng, D., and Yin, Y. Rzenembed: Towards comprehensive multimodal retrieval. arXiv preprint arXiv:2510.27350, 2025.

Jiang, D., Lu, Y., Li, Z., Lyu, Z., Nie, P., Wang, H., Su, A., Chen, H., Zou, K., Du, C., et al. Verltool: Towards holistic agentic reinforcement learning with tool use. arXiv preprint arXiv:2509.01055, 2025.

Jiang, T., Song, M., Zhang, Z., Huang, H., Deng, W., Sun, F., Zhang, Q., Wang, D., and Zhuang, F. E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580, 2024.

Jin, H., Zhang, Y., Shi, L., Zhang, S., Kou, F., Yang, J., Zhu, C., and Luo, J. An end-to-end graph attention network hashing for cross-modal retrieval. Advances in Neural Information Processing Systems, 37:21062126, 2024.

Kim, D., Kim, N., and Kwak, S. Improving cross-modal retrieval with set of diverse embeddings. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2342223431, 2023a.

Kim, J. M., Koepke, A., Schmid, C., and Akata, Z. Exposing and mitigating spurious correlations for cross-modal retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 25852595, 2023b.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS Symposium on Operating Systems Principles, 2023.

Lan, Z., Niu, L., Meng, F., Zhou, J., and Su, J. Llave: Large language and vision embedding models with hardness-weighted contrastive learning. arXiv preprint arXiv:2503.04812, 2025.

Li, C., Wong, C., Zhang, S., Usuyama, N.., Liu, H., Yang, J. Naumann, T., Poon, H., and Gao, J. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36:2854128564, 2023a.

Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Proceedings of the International Conference on Machine Learning, 2022.

Li J Li . Se, S. and Hoi S.Bli- Bt language-image pre-training with frozen image encoders and large language models. In Proceedings of the International Conference on Machine Learning, 2023b.

Li, X., Li, C., Chen, S.-Z., and Chen, X. U-marvel: Unveiling key factors for universal multimodal retrieval via embedding learning with mllms. arXiv preprint arXiv:2507.14902, 2025a.

Li, Y., Jiang, S., Hu, B., Wang, L., Zhong, W., Luo, W., Ma, L., and Zhang, M. Uni-moe: Scaling unified multimodal llms with mixture of experts. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025b.

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122, 2023a.

Lin, S.-C., Lee, C., Shoeybi, M., Lin, J., Catanzaro, B., and Ping, W. Mm-embed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571, 2024a.

Lin, W., Chen, J., Mei, J., Coca, A., and Byrne, B. Finegrained late-interaction multi-modal retrieval for retrieval augmented visual question answering. In Advances in Neural Information Processing Systems, 2023b.

Lin, W., Mei, J., Chen, J., and Byrne, B. PreFLMR: Scaling up fine-grained late-interaction multi-modal retrievers. In Association for Computational Linguistics, 2024b.

Liu, S., Cheng, H., Liu, H., Zhang, H., Li, F., Ren, T., Zou, X., Yang, J., Su, H. Zhu, J., et al. Llava-plus: Learning to use tools for creating multimodal agents. In European Conference on Computer Vision, pp. 126142. Springer, 2024.

Liu, Y., Chen, P., Cai, J., Jiang, X., Hu, Y., Yao, J., Wang, Y., and Xie, W. Lamra: Large multimodal model as your advanced retrieval assistant. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Liu, Z., Xiong, C., Lv, Y., Liu, Z., and Yu, G. Universal vision-language dense retrieval: Learning a unified representation space for multi-modal retrieval. In The Eleventh

International Conference on Learning Representations, 2023.

Lu, D., Sun, Y., Zhang, Z., Huang, L., Zeng, J., Shu, M., and Cao, H. Internvl-x: Advancing and accelerating internvl series with efficient visual token compression. arXiv preprint arXiv:2503.21307, 2025.

Marino, K., Rastegari, M., Farhadi, A., and Mottaghi, R. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.

Mensink, T., Uijlings, J., Castrejon, L., Goel, A., Cadar, F., Zhou, H., Sha, F., Araujo, A., and Ferrari, V. Encyclopedic vqa: Visual questions about detailed properties of fine-grained categories. In Proceedings of the International Conference on Computer Vision, 2023.

Muennighoff, N., Yang, Z., Shi, W., Li, X. L., Fei-Fei, L., Hajishirzi, H., Zettlemoyer, L., Liang, P., Candès, E., and Hashimoto, T. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Narayan, K., Xu, Y., Cao, T., Nerella, K., Patel, V. M., Shiee, N., Grasch, P., Jia, C., Yang, Y., and Gan, Z. Deepmmsearch-r1: Empowering multimodal llms in multimodal web search. arXiv preprint arXiv:2510.12801, 2025.

Pei, R., Liu, J., Li, W., Shao, B., Xu, S., Dai, P., Lu, J., and Yan, Y. Clipping: Distilling clip-based models with a student base for video-language retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1898318992, 2023.

Pham, K., Huynh, C., Lim, S.-N., and Shrivastava, A. Composing object relations and attributes for image-text matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14354 14363, 2024.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, 2021.

Sain, A., Bhunia, A. K., Chowdhury, P. N., Koley, S., Xiang, T., and Song, Y.-Z. Clip for all things zero-shot sketchbased image retrieval, fine-grained or not. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 27652775, 2023.

Saito, K., Sohn, K., Zhang, X., Li, C.-L., Lee, C.-Y., Saenko, K., and Pfister, T. Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1930519314, 2023.

Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q. Zhang, Z., Zhao, K., Zhang, Q., et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

Shen, L., Chen, G., Shao, R., Guan, W., and Nie, L. Mome: Mixture of multimodal experts for generalist multimodal large language models. arXiv preprint arXiv:2407.12709, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Shu, F., Liao, Y., Zhuo, L., Xu, C., Zhang, L., Zhang, G., Shi, H., Chen, L., Zhong, T., He, W., et al. Llava-mod: Making llava tiny via moe knowledge distillation. arXiv preprint arXiv:2408.15881, 2024.

Sun, N., Tang, J., Sun, L., Chen, R., Lu, Y., Chu, X., and Ling, H. Reflection from retrieval: Mllm-guided iterative reasoning for zero-shot composed image retrieval.

Suo, Y., Ma, F., Zhu, L., and Yang, Y. Knowledge-enhanced dual-stream zero-shot composed image retrieval. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2695126962, 2024.

Vaze, S., Carion, N., and Misra, I. Genecis: A benchmark for general conditional image similarity. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023.

Wang, C., Feng, K., Chen, D., Wang, Z., Li, Z., Gao, S., Meng, M., Zhou, X., Zhang, M., Shang, Y., et al. Adatooler-v: Adaptive tool-use for images and videos. arXiv preprint arXiv:2512.16918, 2025a.

Wang, C., He, Y., Zhou, Y., Wang, Y., Liu, J., Xia, P., Tu, Z., Bansal, M., and Yao, H. Knowing the answer isn't enough: Fixing reasoning path failures in lvlms. arXiv preprint arXiv:2512.06258, 2025b.

Wang, C., Zhang, Z., Meng, M., Zhou, X., and Jiang, H. Vision-ekipl: External knowledge-infused policy learning for visual reasoning. arXiv preprint arXiv:2506.06856, 2025c.

Wang, C., Zhang, Z., Teng, L., Li, Z., and Kan, S. Tmcir: Token merge benefits composed image retrieval. arXiv preprint arXiv:2504.10995, 2025d.

Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Wang, Y., Wang, L., Zhou, Q., Wang, Z., Li, H., Hua, G., and Tang, W. Multimodal llm enhanced cross-lingual cross-modal retrieval. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 82968305, 2024b.

Wei, C., Chen, Y., Chen, H., Hu, H., Zhang, G., Fu, J., Riter, A., and Chen, W. Uniir: Training and benchmarking universal multimodal information retrievers. In Proceedings of the European Conference on Computer Vision, 2024a.

Wu, H., Gao, Y., Guo, X., Al-Halah, Z., Rennie, S., Grauman, K., and Feris, R. Fashion iq: A new dataset towards retrieving images by natural language feedback. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2021.

Xie, J., Mao, W., Bai, Z., Zhang, D. J., Wang, W., Lin, K. Q., Gu, Y., Chen, Z., Yang, Z., and Shou, M. Z. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Xu, C., Wang, X., Liao, Z., Li, Y., Hou, T., and Deng, Z. Show-o turbo: Towards accelerated unified multimodal understanding and generation. arXiv preprint arXiv:2502.05415, 2025a.

Xu, M., Dong, J., Hou, J., Wang, Z., Li, S., Gao, Z., Zhong, R., and Cai, H. MM-R5: Multimodal reasoning-enhanced reranker via reinforcement learning for document retrieval. arXiv preprint arXiv:2506.12364, 2025b.

Xu, M., Dong, J., Hou, J., Wang, Z., Li, S., Gao, Z., Zhong, R., and Cai, H. Mm-r5: Multimodal reasoning-enhanced reranker via reinforcement learning for document retrieval. arXiv preprint arXiv:2506.12364, 2025c.

Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Yang, Z., Li, L., Lin, K., Wang, J., Lin, C.-C., Liu, Z., and Wang, L. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 9 (1):1, 2023.

Ye, J., Xu, H., Liu, H., Hu, A., Yan, M., Qian, Q., Zhang, J., Huang, F., and Zhou, J. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. arXiv preprint arXiv:2408.04840, 2024a.

Ye, Q., Xu, H., Xu, G., Ye, J., Yan, M., Zhou, Y., Wang, J. Hu, A., Shi, P., Shi, Y., et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.

Ye, Q., Xu, H., Ye, J., Yan, M., Hu, A., Liu, H., Qian, Q., Zhang, J., and Huang, F. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition, pp. 13040 13051, 2024b.

Yuan, Y. and Lam, W. Conversational fashion image retrieval via multiturn natural language feedback. In Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval, 2021.

Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In Proceedings of the International Conference on Computer Vision, 2023.

Zhang, K., Luan, Y., Hu, H., Lee, K., Qiao, S., Chen, W., Su, Y., and Chang, M.-W. Magiclens: self-supervised image retrieval with open-ended instructions. In Proceedings of the 41st International Conference on Machine Learning, pp. 5940359420, 2024a.

Zhang, L., Wang, B., Qiu, X., Reddy, S., and Agrawal, A. REARANK: Reasoning re-ranking agent via reinforcement learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 24582471, 2025a.

Zhang, Q., Lei, Z., Zhang, Z., and Li, S. Z. Context-aware attention network for image-text retrieval. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 35363545, 2020.

Zhang, Q., Lyu, F., Sun, Z., Wang, L., Zhang, W., Guo, Z., Wang, Y., King, I., Liu, X., and Ma, C. What, how, where, and how well? a survey on test-time scaling in large language models. arXiv preprint arXiv:2503.24235, 2025b.

Zhang, S., Fang, Q., Yang, Z., and Feng, Y. Llava-mini: Efficient image and video large multimodal models with one vision token. arXiv preprint arXiv:2501.03895, 2025c.

Zhang, X., Zhang, Y., Xie, W., Li, M., Dai, Z., Long, D., Xie, P., Zhang, M., Li, W., and Zhang, M. Gme: Improving universal multimodal retrieval by multimodal llms. arXiv preprint arXiv:2412.16855, 2024b.

Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z., and Ma, Y. Llamafactory: Unified efficient finetuning of $1 0 0 +$ language models. In Proceedings of the

62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics. URL http://arxiv.org/abs/2403. 13372.

Zhou, J., Liu, Z., Liu, Z., Xiao, S., Wang, Y., Zhao, B., Zhang, C. J., Lian, D., and Xiong, Y. Megapairs: Massive data synthesis for universal multimodal retrieval. arXiv preprint arXiv:2412.14475, 2024.

Zhu, L., Ji, D., Zhu, S., Gan, W., Wu, W., and Yan, J. Learning statistical texture for semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1253712546, 2021.

Zhu, L., Chen, T., Ji, D., Ye, J., and Liu, J. Llafs: When large language models meet few-shot segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 30653075, 2024.

Zhu, L., Chen, T., Ji, D., Xu, P., Ye, J., and Liu, J. Llafs $^ { + + }$ : Few-shot image segmentation with large language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025a.

Zhu, L., Chen, T., Ji, D., Ye, J., and Liu, J. Not every patch is needed: Towards a more efficient and effective backbone for video-based person re-identification. IEEE Transactions on Image Processing, 2025b.

Zhu, L., Chen, T., Yin, J., See, S., Soh, D. W., and Liu, J. Replay master: Automatic sample selection and effective memory utilization for continual semantic segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025c.

Zhu, L., Ji, D., Chen, T., Wu, H., and Wang, S. Retrvr1: A reasoning-driven mllm framework for universal and efficient multimodal retrieval. arXiv preprint arXiv:2510.02745, 2025d.

# A. Prompt Template

# A.1. System Prompt

Fig. 4 illustrate the system prompt for both training and inference.

images."}, "minItems": 1, "maxItems": 4}, "required: ["target_images"]}}

<tool_call>

{"name"<function-name>, "arguments": <args-json-object} </tool_call>

# A.2. User Prompt

Fig. 5 illustrate the user prompt for both training and inference.

<table><tr><td>User Prompt for Training and Inference</td></tr><tr><td>Query: &lt;question&gt; Candidates: (1) &lt;candidate1&gt;(2) &lt;candidate&gt;..(K) &lt;candidate&gt;</td></tr><tr><td>task is  rank all candidates from most relevant o least relevant basedon how well they match the query.</td></tr><tr><td>You can use the following tools to assist with analysis:</td></tr><tr><td>crop_image: Zoom in on speciic areas of the query image or a candidate image for detailed analysis. selaeSelec 4 queryagescaniaage romhe provide isor uec.</td></tr><tr><td></td></tr><tr><td>If you DO need to use tools: </td></tr><tr><td></td></tr><tr><td>If you do NOT need to use tools: &lt;&gt;oudetaiea oehereexpaiyouanalys teps./hn[ranke_is]/a</td></tr></table>

<Yoas ocsshere/hikolcal"name "selectmges", "retsargetmages",1, 15, 12, 4, 8]/answer>

# A.3. Annotation Prompt

F into the user prompt to guide the generation.

<table><tr><td>Annotation Prompt</td></tr><tr><td>a    . </td></tr><tr><td>First, analyze the key features and requirements of the query (both visual and textual). </td></tr><tr><td>F</td></tr><tr><td>match the query (e.g., taxonomy, visual characteristics, context, or any other relevant attributes). </td></tr><tr><td>small subset of the most relevant images.</td></tr><tr><td></td></tr><tr><td>promising candidates and to resolve any ambiguities. </td></tr></table>

# B. Details about M-BEIR Dataset

W  b I haainal siheaas orate hi yeul ifencomeanrval c alatnlyparehata.oroecehensivdetan these differences, we refer the readers to the original UniR (Wei et al., 2024a) paper.

Table 7. Summary of the M-BEIR benchmarks.   

<table><tr><td>Task</td><td>Dataset</td><td>Domain</td><td># Train</td><td># Dev</td><td># Test</td><td># Pool</td></tr><tr><td rowspan="3">qt → ci</td><td>VisualNews</td><td>News</td><td>99K</td><td>20K</td><td>20K</td><td>542K</td></tr><tr><td>MSCOCO</td><td>Misc.</td><td>100K</td><td>24.8K</td><td>24.8K</td><td>5K</td></tr><tr><td>Fashion200K</td><td>Fashion</td><td>15K</td><td>1.7K</td><td>1.7K</td><td>201K</td></tr><tr><td>qt → ct</td><td>WebQA</td><td>Wiki</td><td>16K</td><td>1.7K</td><td>2.4K</td><td>544K</td></tr><tr><td rowspan="2">qt → (ci, ct)</td><td>EDIS</td><td>News</td><td>26K</td><td>3.2K</td><td>3.2K</td><td>1M</td></tr><tr><td>WebQA</td><td>Wiki</td><td>17K</td><td>1.7K</td><td>2.5K</td><td>403K</td></tr><tr><td rowspan="3">qi → ct</td><td>VisualNews</td><td>News</td><td>100K</td><td>20K</td><td>20K</td><td>537K</td></tr><tr><td>MSCOCO</td><td>Misc.</td><td>113K</td><td>5K</td><td>5K</td><td>25K</td></tr><tr><td>Fashion200K</td><td>Fashion</td><td>15K</td><td>4.8K</td><td>4.8K</td><td>61K</td></tr><tr><td>qi → ci</td><td>NIGHTS</td><td>Misc.</td><td>16K</td><td>2K</td><td>2K</td><td>40K</td></tr><tr><td rowspan="2">(qi, qt) → ct</td><td>OVEN</td><td>Wiki</td><td>150K</td><td>50K</td><td>50K</td><td>676K</td></tr><tr><td>InfoSeek</td><td>Wiki</td><td>141K</td><td>11K</td><td>11K</td><td>611K</td></tr><tr><td rowspan="2">(qi, qt) → ci</td><td>FashionIQ</td><td>Fashion</td><td>16K</td><td>2K</td><td>6K</td><td>74K</td></tr><tr><td>CIRR</td><td>Misc.</td><td>26K</td><td>2K</td><td>4K</td><td>21K</td></tr><tr><td rowspan="2">(qi, qt) → (ci, ct)</td><td>OVEN</td><td>Wiki</td><td>157K</td><td>14.7K</td><td>14.7K</td><td>335K</td></tr><tr><td>InfoSeek</td><td>Wiki</td><td>143K</td><td>17.6K</td><td>17.6K</td><td>481K</td></tr><tr><td>8 tasks</td><td>10 datasets</td><td>4 domains</td><td>1.1M</td><td>182K</td><td>190K</td><td>5.6M</td></tr></table>

# C. Details about Unseen Dataset

Here we preent h detail  thense Dataset  Tabl Many  them eactually dapted om MC a T These differences create a substantial disparity compared to the original COCO dataset.

Table 8. Summary of the Unseen Dataset.   

<table><tr><td>Dataset</td><td>Image Source</td><td>Task</td><td>Query Format</td><td>Candidate Format</td></tr><tr><td>CIRCO</td><td>MSCOCO unlabeled set</td><td>(qi, qt) → ci</td><td>&lt;image&gt;&lt;relative caption&gt;</td><td>&lt;image&gt;</td></tr><tr><td>GeneCIS</td><td>MSCOCO</td><td>(qi, qt) → ci</td><td>&lt;image&gt;&lt;relative caption&gt;</td><td>&lt;image&gt;</td></tr><tr><td>Visual Dialog</td><td>MSCOCO</td><td>qdialog → ci</td><td>&lt;Q1&gt;&lt;A1&gt;.…&lt;Qj&gt;&lt;Aj&gt;</td><td>&lt;image&gt;</td></tr><tr><td>Visual Storytelling</td><td>Flickr</td><td>(qi  qt) → ci</td><td>&lt;text1&gt;&lt;image1&gt;..&lt;textj&gt;</td><td>&lt;image&gt;</td></tr><tr><td>MT-FIQ</td><td>FashionIQ</td><td>(qi  qt) → ci</td><td>&lt;image1&gt;&lt;relative caption1&gt;.. &lt;imagej&gt;&lt;relative captionj&gt;</td><td>&lt;image&gt;</td></tr></table>

# D. Exploration of RAG Applications

Visal Quesse (KA)berk. eical ae eval and A sk us the a prhodelheenisl  te neseail Ta Rerve hive pe pc  trival presonan , demonstrating that our Multimodal Interleaved EvidenceReasoning significantly enhances MLLM capabilities in RAG settings.

Table 9. Comparison of RAG capabilities on KVQA tasks.   

<table><tr><td>Method</td><td>OKVQA (Marino et al., 2019)</td><td>Infoseek (Chen et al., 2023)</td><td>E-VQA (Mensink et al., 2023)</td></tr><tr><td colspan="4">Retrieval (PR@5)</td></tr><tr><td>PreFLMR (Lin et al., 2024b)</td><td>70.9</td><td>62.1</td><td>73.7</td></tr><tr><td>LamRA-7B (Liu et al., 2025)</td><td>89.0</td><td>73.4</td><td>75.0</td></tr><tr><td>V-Retrver-7B</td><td>90.9</td><td>78.3</td><td>78.1</td></tr><tr><td colspan="4">VQA (ACC)</td></tr><tr><td>RA-VQAv2 w/ PreFLMR (Lin et al., 2023b)</td><td>61.9</td><td>32.1</td><td>54.5</td></tr><tr><td>LamRA-7B (Liu et al., 2025)</td><td>64.3</td><td>28.8</td><td>56.2</td></tr><tr><td>V-Retrver-7B</td><td>65.7</td><td>31.9</td><td>58.0</td></tr></table>

# E. Algorithms and Detailed Analysis

I      R complexity analysis.

# E.1. Inference Process

T in Algorithm 1.

# E.2. Training Pipeline

Treaay oesivghe ri objectives, is presented in Algorithm 2.

Algorithm 1 V-Retrver Inference Pipeline   

<table><tr><td>I: Query q, andidate Pool Ω = {cn}=1, Embedig Model , Reasonig Agent π, Top-ize , Windo ize W, Stride S</td></tr><tr><td>Output: Ranked Candidate List </td></tr><tr><td>{// Stage 1: Coarse Retrieval (Embedding-based)}</td></tr><tr><td>Compute similarity scores sn = cos(Φ(q), Φ(cn)) for all cn  Ω</td></tr><tr><td>Select top-K candidates: Ctop ← Top-K(Ω, {sn})</td></tr><tr><td>{// Stage 2: Agentic Reranking (Reasoning-based)}</td></tr><tr><td>Initialize global ranking list Lglobal ← Ø</td></tr><tr><td>Split Ctop into windows {w1, w2, . . . , wm} with size W and stride S</td></tr><tr><td>for each window wj  {w1, . . . , wm} do</td></tr><tr><td>Initialize context H0 ← (q, ωj, Instruction)</td></tr><tr><td>t ← 0 while True do</td></tr><tr><td>Generate output: 0 ∼ πθ(Ht)</td></tr><tr><td>if ot contains &lt;tool_call&gt; then</td></tr><tr><td>Parse action at and arguments from ot</td></tr><tr><td>Execute visual tool: vobs ← ftool(at, wj)</td></tr><tr><td>Update context: Ht+1 ← Ht  ot  vobs</td></tr><tr><td>else if ot contains &lt;answer&gt; then</td></tr><tr><td>Parse local rank list j from ot</td></tr><tr><td>Update Lglobal with local rank rj</td></tr><tr><td>break</td></tr><tr><td>end if</td></tr><tr><td>t ← t + 1</td></tr><tr><td>end while</td></tr><tr><td>end for</td></tr><tr><td>L ← AggregateRanks(Lglobal)</td></tr></table>

# Algorithm 2 Curriculum-Based Agentic Training

<table><tr><td colspan="2">Iorithm 2 Curriculum-Based Agentic Training</td></tr><tr><td>Input: Pretrained MLLM θinit, Retrieval Dataset D, Synth Model Msyn Output: Optimized Policy πθ*</td><td></td></tr><tr><td>{// Stage 1: Reasoning Activation (SFT)}</td><td></td></tr><tr><td>Synthesize CoT data: Dsft ← {(q, c, Tcot)} using Msyn on D</td><td></td></tr><tr><td>Filter Dsft for format compliance Update θsf t ← Minimize LSFT (θinit, Dsf t)</td><td></td></tr><tr><td>{// Stage 2: Reliability Refinement (Rejection Sampling)}</td><td></td></tr><tr><td>Initialize Drsft ← </td><td></td></tr><tr><td>for each (q, c)  D do</td><td></td></tr><tr><td>Sample k trajectories {τ1, · . . , τk} ∼ πθsf t(q, c)</td><td></td></tr><tr><td>if IsFormatValid(τi)  IsRankCorrect(τi) then Add valid Ti to Drsf t</td><td></td></tr><tr><td>end if</td><td></td></tr><tr><td>end for</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Update θrsf t ← Minimize LSFT (θsf t, Drsf t)</td><td></td></tr><tr><td>{// Stage 3: Evidence-Aligned Policy Optimization (EAPO)}</td><td></td></tr><tr><td>Initialize θ ← θrsft, Reference policy πref ← θrsft</td><td></td></tr><tr><td>while not converged do</td><td></td></tr><tr><td>Sample batch of queries Bq ∼ D</td><td></td></tr><tr><td></td><td></td></tr><tr><td>for each query q  Bq do</td><td></td></tr><tr><td>Sample group of trajectories G = {o1, . . . , 0G} ∼ πθ(q)</td><td></td></tr><tr><td>Compute rewards R(oi) = αr f mt(oi) + βrrank(oi) + rtool(oi) end for</td><td></td></tr><tr><td>Compute advantages Ai via Group Normalization over G</td><td></td></tr><tr><td>Compute GRPO loss TEAPO(θ)</td><td></td></tr><tr><td>Update θ ← Optimize TEAPO(θ)</td><td></td></tr><tr><td>end while</td><td></td></tr><tr><td></td><td></td></tr><tr><td>return πθ</td><td></td></tr></table>

# F. Qualitative Examples

u the proposed method.

# G. Limitations and Future Works

selcnoeas my n  c ha qv recommendation and retrieval-augmented generation, further advancing general-purpose agentic MLLMs.

11

Has more buttons and is blue and is lighter

1

![](images/12.jpg)  
Candidate 2

![](images/13.jpg)  
Candidate 3

![](images/14.jpg)  
Candidate 4

# Candidate 1

# Candidate 5

3 and 4</think>

![](images/15.jpg)  
Figure 7. A qualitative example of the retrieval result generated from V-Retrver.

<tool_call{"name": "select_images", "arguments": {"target_images": [3,4]} </tool_call>

![](images/16.jpg)  
Candidate 3

![](images/17.jpg)  
Candidate 4

remaining candidates are less relevant. The final ranking is: 4, 3, 2, 1, 5 </think>   
<answer>[4,3,2,1,5]/answer>

Find a smaller monitor and an additional laptop

![](images/18.jpg)  
Candidate 1

![](images/19.jpg)  
Candidate 2

![](images/20.jpg)  
Candidate 3

![](images/21.jpg)  
Candidate 4

![](images/22.jpg)  
Candidate 5   
Figure 8. A qualitative example of the retrieval result generated from V-Retrver.

les relevant. To further refine the ranking, let's select images 2 and 5 for detailed analysis./think <tool_call{"name": "select_images", "arguments": {"target_images": [2, 5]}}</tool_call>

![](images/23.jpg)  
Candidate 2

![](images/24.jpg)  
Candidate 5

ki Cta eqea 3</think>

<answer>[2, 5, 1, 4, 3]/answer>

I

![](images/25.jpg)  
Green metallic knitted dress   
Candidate 3

![](images/26.jpg)  
Candidate 2

![](images/27.jpg)

# Candidate 1

![](images/28.jpg)  
Candidate 4

Candidate 5

i y confirm.</think>

<ol_cal"name" "crop_image", "arguments": {"bbox_2d: [0, 0,180, 250], "target_image": 3}/tool_call>

![](images/29.jpg)  
Figure 9. A qualitative example of the retrieval result generated from V-Retrver.

remaining candidates are less relevant. The final ranking is: 4, 3, 2, 1, 5 </think>   
<answer>[4,3,2,1,5]/answer>

![](images/30.jpg)

What is this plant named after?

# Candidate 1

Li Car . orange to brown stamens. The plant flowers in June, with flowers.

# Candidate 2

L Cal

# Candidate 3

named by.

Candidate 4 Liuoaal

# Candidate 5

parry" (lemon lily) Photos gallery: "Lilium parryi.

<ool_call"name "cro_image", "argents"{"bbox_2d: [125,88,145,108],{"target_iage:0}</tool_ca

![](images/31.jpg)  
Figure 10. A qualitative example of the retrieval result generated from V-Retrver.

nc   
  
flowers n the ageh   
<answer>[1, 2, 4, 3, 5]/answer>

![](images/32.jpg)

![](images/33.jpg)

![](images/34.jpg)  
Candidate 2

![](images/35.jpg)  
Candidate 3

![](images/36.jpg)  
Candidate 4

# Candidate 1

Candidate 5

dyue  .Ca promising. Let's use the select_images'tool to focus on these two for a detailed comparison./think <tool_call{"name": "select_images", "arguments": {"target_images": [2,5]}}</tool_call/tool_call>

![](images/37.jpg)  
Figure 11. A qualitative example of the retrieval result generated from V-Retrver.

![](images/38.jpg)  
Candidate 2

![](images/39.jpg)  
Candidate 5

e nnCoy <answer>[5, 2, 1, 3, 4]/answer>