# Embed-RL：用于推理驱动的多模态嵌入的强化学习

江浩男 $^ { 1 , 2 }$ \*, 王宇吉 $^ { 1 , 2 \colon }$ , 朱永杰 $^ { 2 \dagger }$ , 陆欣 $^ 2$ , 秦文宇 $^ 2$ , 王猛 $^ 2$ , 万鹏飞 $^ 2$ , 唐延松 $^ 1 ‡ \bot$ 清华大学深圳国际研究生院 $^ 2$ 快手科技 Kling 团队 {jiang-hn24@mails, yuji-wan24@mails, tang.yansong@sz}.tsinghua.edu.cn {zhuyongjie, luxin09, qinwenyu, wangmeng46, wanpengfei}@kuaishou.com

摘要。利用多模态大语言模型（MLLM）已成为推动通用多模态嵌入（UME）在解决多样化跨模态任务中的关键。近期研究表明，结合生成性思维链（CoT）推理相比于区分性方法可以显著增强任务特定的表征。然而，现有生成嵌入方法生成的推理CoTs局限于对查询的文本分析，与目标检索无关。为了解决这些局限性，我们提出了一种基于推理驱动的UME框架，集成了嵌入者引导的强化学习（EG-RL），以优化推理器生成证据可追溯思维链（T-CoT）。我们的主要贡献有三点：（1）我们设计了一个EG-RL框架，其中嵌入者为推理器提供明确的监督，确保生成的CoT轨迹与嵌入任务对齐。（2）我们引入了T-CoT，它提取关键的多模态线索，以聚焦于与检索相关的元素，并为嵌入者提供多模态输入。（3）在有限的计算资源下，我们的框架在MMEB-V2和UVRB基准测试中超越了开创性的嵌入模型。结构化推理中的多模态证据与检索导向的对齐有效强化了跨模态的语义一致性，并提升了模型在复杂场景下的细粒度匹配能力以及泛化能力。我们的工作表明，针对性的推理优化可以显著提高多模态嵌入质量，为推理驱动的UME开发提供了实用且高效的解决方案。 项目页面。关键词：多模态嵌入 $\cdot$ 生成性推理 $\cdot$ 强化学习

# 1 引言

多模态嵌入作为跨模态任务的核心支持技术，已经广泛应用于多个重要方向，如图像-文本检索、视频时刻定位和视觉文档理解。传统的多模态嵌入方法采用双编码器架构，如CLIP、BLIP和SigLIP。这些方法在弥合不同模态之间的差距方面的能力较弱，相比于多模态大型语言模型（MLLMs）。此外，MLLMs凭借其强大的多模态理解和遵循指令的能力，能够适应多样且复杂的任务需求。因此，越来越多的文献证明，MLLMs可以用于学习通用多模态嵌入（UME），从而捕捉通用内容相似性。同时，像多模态嵌入基准（MMEB）及其升级版MMEB-V2等评估基准满足了UME研究的学术需求，涵盖了78项跨三种模态的指令感知任务。

![](images/1.jpg)  
Fig. 1: Multimodal embedding optimization via Embedder-Guided Reinforcement Learning (EG-RL). (a) Frameworks evolution. (b) Reasoning enhancement with RL-optimized evidential Traceability CoT (T-CoT). (c) Comparison of multi-task performance.

目前，大多数基于大语言模型（MLLM）的嵌入方法都是判别嵌入模型。这些模型通常直接从输入词元的最终隐状态中提取嵌入特征，未能充分利用MLLM固有的生成能力和推理潜力。因此，最近的研究探索将生成推理整合到统一多模态嵌入（UME）任务中，如图1所示。例如，UME-R1等方法通过由MLLM嵌入器生成的文本链式思维（CoTs）统一了判别和生成嵌入。然而，同时优化对比损失和下一个词预测目标会导致梯度冲突，从而导致性能不理想。相比之下，TTE提出的解耦推理-嵌入器范式依赖于大量计算资源和从大模型中提取的数据精炼，旨在通过解耦这两个过程来缓解该问题，使用预训练的MLLM生成离线CoT推理，仅对嵌入器进行训练，以提升嵌入质量。然而，TTE的推理器生成的CoT并没有专门为嵌入设计，因为它们没有与嵌入器一起训练。这种不一致可能引入噪声，甚至导致幻觉。此外，仅依赖文本推理无法充分利用MLLM嵌入器处理多模态信号的潜力，这些丰富的表征可能显著提升检索性能。多模态线索的不足导致显著的嵌入对齐偏差，使得关键的视觉空间线索和视频时序信号未能在检索中充分捕获，结果造成跨模态匹配效果较差以及在复杂的现实世界多模态任务中泛化能力受限。

为了解决上述问题，本文提出了一种基于推理驱动的解耦式UME框架。该框架利用嵌入引导强化学习（EG-RL）算法优化由推理器生成的CoT，采用我们新颖的过程奖励，旨在对齐查询与目标之间的关系以及可验证的检索结果奖励。首先，我们构建了一个数据集，最初用于训练嵌入器，以生成基于前序输入和CoT词元序列的高质量嵌入。训练后的嵌入器作为奖励模型，提供稳定且可靠的奖励信号。其次，受区域感知范式的启发，我们提出了证据追溯CoT（T-CoT），明确引导嵌入器专注于与任务相关的信息，有效过滤冗余的视觉元素，并整合特定模态的关键信息，以适应长文本检索、粗粒度语义匹配和细粒度对齐，从而在异构任务中实现鲁棒性能。本文的主要贡献总结如下：1. 嵌入引导强化学习。我们提出了一种新颖的解耦强化学习框架，其中嵌入器引导推理器优化针对特定嵌入任务的CoT轨迹。该方法解决了生成目标与嵌入目标之间的关键冲突，确保推理器的输出显著提高检索质量，并应对将通用CoT适配到嵌入任务的核心挑战。2. 用于嵌入的证据追溯CoT。我们进一步将CoT推理扩展到复杂的多模态场景，整合显式的视觉定位信息、视频关键帧和文本关键词到详细的推理轨迹中。该设计使模型能够专注于核心检索相关信息，有效缓解冗余多模态和文本数据对整体嵌入对齐性能的负面影响。3. 多基准高效性能提升。在计算受限的设置下，本文提出的框架在MMEB-V2和视频检索UVRB基准数据集上超越了最先进的生成嵌入模型，并在多样的组合场景中取得了卓越的表现。

# 2 相关工作

# 2.1 通用多模态嵌入

构建稳健的多模态表示是多模态学习中的一个核心挑战。像 CLIP 和 ALIGN 这样的开创性模型采用了双编码器架构，通过在丰富的大规模图像-文本配对数据上进行对比学习来学习有效的表示。然而，它们在处理交错的图像-文本输入方面存在困难，并且其文本编码器缺乏足够的能力来理解真正复杂的文本内容。为了解决这个问题，研究人员利用多模态大语言模型（MLLMs）来构建嵌入模型，利用其强大的多模态理解能力来提升整体学习性能。现有的研究集中在不同的方面：VLM2Vec 将 MLLMs 通过对比学习转化为嵌入模型，并在视觉问答和定位等非常规检索任务上取得了卓越的表现；MM-Embed 探索将现成的 MLLMs 作为零-shot 重新排序器来优化检索结果；LamRA 通过两阶段检索训练和联合重新排序统一了多模态检索范式。Mega-Pairs 和 GME 通过自动化流水线解决模态不平衡问题；LLaVE 和 Unite 则专注于困难负样本的挖掘。近期的研究关注于与指令相关的表示：MMEB 和 MMEB-V2 构建了一个涵盖 78 个任务的综合评估基准。UME-R1 首先引入推理机制，但通过强化学习（RL）同时优化两个组件会导致冲突，冗余的思维链（CoT）轨迹稀释了表示。TTE 采用了一个解耦但计算成本高昂的架构，其推理器与检索任务不匹配，导致任务无关的输出。本文提出了一种解耦的强化学习框架，能够分别优化两个组件，并通过双重奖励机制生成与检索相关的推理轨迹，以解决上述挑战。

# 2.2 使用强化学习的多模态推理

MLLMs [2,31,38, 51] 扩展并丰富了大型语言模型 (LLMs) 在多模态领域的能力，在包括视觉问答 [3,17,38,50]、视觉定位 [12,28, 33] 和关键帧提取 [35, 52, 68, 79] 等多样任务中取得了令人鼓舞的结果。早期的工作 [23, 53, 58, 60, 73] 大多使用标准化的链式推理 (CoT) 提示完成推理任务。自从 DeepSeek-R1 [19] 提出了群体相对策略优化 (GRPO) 强化学习算法以来，许多近期的研究优化了先进的强化学习算法 [64, 74]，并增强了 MLLMs 的推理能力 [6, 14, 44, 56, 59]。

GRIT [15] 交替使用边界框坐标和文本推理链，并基于 GRPO 算法设计了一种强化学习方案，实现了高效训练，同时具有双重稳健奖励且无需额外标注数据。Ground-R1 [6] 提出了一个强化学习框架，以在没有额外注释的情况下实现有根视觉推理，通过双重奖励指导响应生成，提高推理的可靠性和可解释性。BRPO [11] 使用基于交并比（IoU）的奖励来指导模型自主生成视觉-文本反映，并结合视觉词元机制以缓解视觉注意力稀释和幻觉问题。DeepEyes [75] 采用端到端强化学习促使模型发展“用图像思考”的能力，提高在各种推理任务上的性能。TreeVGR [47] 提出了 TreeBench 基准和 TreeVGR 训练范式，通过强化学习共同监督定位和推理，增强视觉定位推理能力。受到有根推理的启发，本文进一步提出了证据追溯链（T-CoT），通过提取图像的边界框、视频的关键帧和文本的关键词构建结构化的多模态推理链。该方法使模型能够专注于检索任务的核心区域，从而提高嵌入质量。

# 3 方法论

# 3.1 基础知识

W $q$ $\varOmega = \{ c _ { n } \} _ { n = 1 } ^ { N }$ 的目标是从 $\varOmega$ 中检索出最相关的候选。为了学习辨别性多模态嵌入，我们采用对比学习与 InfoNCE 损失 [42]，如图 2(a) 所示。对于查询 $q _ { i }$，其正目标 $t _ { i } ^ { + }$，以及其负目标集 $\mathcal { T } ^ { - } = \{ t _ { j } ^ { - } \} _ { j \neq i }$，InfoNCE 损失优化模型，以最大化 $q _ { i }$ 和 $t _ { i } ^ { + }$ 之间的相似度，同时最小化与所有 $t _ { j } ^ { - } \in \mathcal { T } ^ { - }$ 之间的相似度。损失定义为：

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { I n f o N C E } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \log \frac { \exp \Bigl ( \cos ( h _ { q _ { i } } , h _ { t _ { i } ^ { + } } ) / \tau \Bigr ) } { \exp \Bigl ( \cos ( h _ { q _ { i } } , h _ { t _ { i } ^ { + } } ) / \tau \Bigr ) + \sum _ { t ^ { - } \in \mathcal { T } ^ { - } } \exp \bigl ( \cos ( h _ { q _ { i } } , h _ { t ^ { - } } ) / \tau \bigr ) } \ , } \end{array}
$$

其中 $h _ { q _ { i } }$ 和 $\boldsymbol { h } _ { t }$ 是从视觉语言模型中提取的 $q _ { i }$ 和目标 $t$ 的嵌入（作为最后一个 token 的最后一层隐藏状态），$\cos ( \cdot , \cdot )$ 表示余弦相似度，$\tau$ 是温度超参数。

# 3.2 数据构建

为了支持基于推理的通用多模态嵌入的训练，我们按照“采样-标注-过滤-划分”的流程构建了高质量的多模态数据集，如图2(a)所示。该数据集整合了多种模态（图像、视频、视觉文档），并通过严格的质量控制确保推理轨迹与检索目标之间的对齐。我们首先通过采用分层抽样策略从三个核心来源中策划初始数据池，并参考了VLM2Vec-V2的数据范式[41]：

![](images/2.jpg)  
Fig. 2: Overview of the proposed data synthesis and EG-RL framework. (a) Data Construction generates T-CoT annotations for query-positive pairs, filters and splits the dataset to enable contrastive and reinforcement learning, laying the groundwork for reasoning-aware embedding. (b) Embedder-Guided Reinforcement Learning finetunes the MLLM with a process-outcome reward function, encouraging T-CoT trajectories that yield more discriminative and beneficial generative embeddings.

（1）来自MMEB-train [26] 的以图像为中心的任务，涵盖图像分类、问答、检索和定位；（2）来自LLaVA-Hound [70] 的视频语言指令数据，包括视频描述、问答和检索；（3）来自ViDoRe [16] 和VisRAG [65] 的视觉文档检索数据。接下来，我们对所有查询-正对样本执行证据追溯思维链（T-CoT）注释。每个T-CoT遵循结构化的三部分格式：（1）<thinking> 提取特定模态的线索（通过text_keywords提取文本关键词，通过bbox_2d提取图像空间位置，通过key_frames提取视频关键时刻）；（2）<rethink> 精炼推理逻辑，专注于关键的检索相关方面；（3）最终答案总结核心检索相关信息。我们为注释设计了特定任务的提示，确保T-CoT与多样的多模态检索场景对齐。在注释之后，我们执行严格的CoT引导相关性过滤，以消除噪声样本。通过定制设计的判断提示，我们评估查询的T-CoT及其正样本是否明显与任务描述不相关或矛盾。我们仅保留标记为“No”的样本，意为相关且不矛盾的样本，以用于对比学习。此过滤步骤有效减少了对比学习中的噪声干扰。初始数据集包含222万样本，过滤后保留183万样本，保留率约为80%。被过滤掉的样本中，约20%被均匀抽样用于强化学习阶段，因为这些困难示例对模型在强化学习中的探索具有价值。此外，我们根据不同数据集的任务重要性和数据质量分配训练权重。构建的数据集展现了三个特征：（1）模态多样性，涵盖文本、图像和视频三种模态；（2）推理一致性，T-CoT明确整合多模态线索和检索相关逻辑，避免信息冗余；（3）质量保证，严格的过滤和加权抽样确保数据集没有显著噪声且在任务间平衡。该数据集为两阶段训练奠定了坚实基础，使嵌入器能够学习具有推理意识的表示，而推理器则通过强化学习优化检索中心T-CoT的生成。

# 3.3 嵌入引导的强化学习

为了解决生成推理与嵌入目标之间的不一致，我们提出了一个解耦的强化学习框架，其中预训练的嵌入模型为推理器提供监督。该框架优化 T-CoT 的生成，优先考虑与检索相关的多模态线索，利用双重指导奖励机制和群体相对策略优化（GRPO）[19]。图 2(b) 说明了该阶段的工作流程。

EG-RL框架设计。首先，我们使用InfoNCE损失对嵌入器进行全面训练，以赋予其强大的嵌入能力。我们的强化学习框架严格分离两个组件：推理器，负责生成T-CoT；以及嵌入器，在对比训练后被冻结。这种解耦确保了三个关键好处：（1）针对推理的优化，而不干扰嵌入器所学习的区分能力；（2）来自冻结嵌入器的稳定奖励信号，基于嵌入对齐持续评估T-CoT质量；（3）灵活整合多源奖励，以内化检索与重排序的知识。推理器将多模态查询作为输入，并输出结构化的T-CoT，它整合了三个关键线索：文本关键词、图像边界框和视频关键帧。此外，我们根据T-CoT重新裁剪边界框和关键帧内的内容，以实现多模态推理感知的嵌入。这种结构化的推理随后与原始输入连接形成嵌入器的输入，记作$\mathcal { L }$：

$$
\mathcal { T } = [ x _ { \mathrm { t e x t } } , x _ { \mathrm { i m g } } , x _ { \mathrm { v i d } } , \mathrm { T } \mathrm { - } \mathrm { C o T } ( x ) , \mathrm { < e m b > } ] .
$$

在这个方程中，$<$ emb $>$ 是一个特殊的标记，其隐状态被提取为最终的嵌入，并且该嵌入的评估由嵌入器直接指导推理器的策略更新。具有过程和结果指导的奖励函数。我们设计了一个由三个组成部分构成的奖励函数，以使 T-CoT 生成与嵌入质量对齐，结合格式合规性、结果级检索有效性和过程级 T-CoT 对齐：格式奖励 $( \mathcal { R } _ { f o r m a t } )$：该奖励确保 T-CoT 严格遵循预定义模板（<thinking> → <rethink> <answer>），并包含所有必要的多模态线索。完全合规时奖励值为 1，否则为 0，确保 T-CoT 输出的可解释性与嵌入模块的兼容性。

嵌入引导的结果奖励 (Routcome)：该奖励度量 T-CoT 通过联合评估正样本的排名准确度和正样本与困难负样本之间的相似性边际，从而提高嵌入对齐的程度。该边际是通过温度参数缩放的负相似度的软最大加权平均。对于查询 $q _ { i }$，其正目标为 $t _ { i } ^ { + }$，批内负样本为 $\{ t _ { j } ^ { - } \} _ { j \neq i }$，嵌入表示为 $e _ { q _ { i } } = \pi _ { e } ( q _ { i } , o _ { i } ^ { q } )$ 和 $e _ { t _ { j } } = \pi _ { e } ( t _ { j } , o _ { j } ^ { t } )$，其中 $o _ { i } ^ { q }$ 和 $o _ { j } ^ { t }$ 是查询和目标的 T-CoT 输出。$o _ { i } ^ { q }$ 的奖励定义为：

$$
\begin{array} { r } { \mathcal { R } _ { \mathrm { o u t c o m e } } ( o _ { i } ^ { q } ) = \operatorname { A c c } _ { k } ( e _ { q _ { i } } , t _ { i } ^ { + } ) \cdot \Big ( \sin ( e _ { q } , e _ { t _ { i } ^ { + } } ) - \mathbb { E } _ { \tau } \big [ \sin ( e _ { q _ { i } } , e _ { t _ { j } ^ { - } } ) \big ] \Big ) , } \end{array}
$$

其中 $\mathrm{Acc}_{k}(e_{q_{i}}, t_{i}^{+})$ 表示前 $k$ 的检索准确率，衡量的是当按照余弦相似度对目标进行排序时，$t_{i}^{+}$ 是否位于前 $k$ 名中；$e_{q_{i}}$ 表示归一化嵌入之间的余弦相似度 $\mathrm{sim}(\cdot, \cdot)$；而 $\mathbb{E}_{\tau}[\cdot]$ 代表批次内负目标嵌入与 $e_{q_{i}}$ 之间余弦相似度的 softmax 加权平均。另外，我们对正目标对称地计算 $\mathcal{R}_{\mathrm{outcome}}$：以 $t_{i}^{+}$ 作为锚点，$q_{i}$ 作为其正查询，其他批次内查询作为负样本来计算 $o_{i}^{t}$ 的 $\mathcal{R}_{\mathrm{outcome}}$。这种对称计算在查询到目标和目标到查询的方向上强制了一致的嵌入对齐。该奖励通过嵌入学习优化 T-CoT 作为核心目标，增强其在样本间的区分能力。

T-CoT 过程奖励 $( \mathcal { R } _ { p r o c e s s } )$：我们采用一个独立的预训练视觉-语言模型判别器 $\mathcal { D }$ 进行逐项比较，以对齐查询和目标的 T-CoT 输出。令 $q _ { \mathrm { c o t } }$ 为查询的 T-CoT 输出，$\{ c _ { \mathrm { c o t } } ^ { j } \} _ { j = 1 } ^ { m }$ 为来自查询数据对和负样本多次推演的 $m$ 个样本的 T-CoT 输出。在洗牌后，真实正样本的索引集记作 $\mathcal { P }$ 。为了减轻位置偏差，我们将 $q _ { \mathrm { c o t } }$ 和洗牌后的 $\{ c _ { \mathrm { c o t } } ^ { j } \} _ { j = 1 } ^ { m }$ 作为成对输入馈送给 $\mathcal { D }$。奖励通过 $\mathcal { D }$ 的选择正确性来量化对齐，正式定义为：

$$
\begin{array} { r } { \mathcal { R } _ { \mathrm { p r o c e s s } } ( o _ { i } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f } \ \mathcal { D } \big ( q _ { \mathrm { c o t } } , \{ c _ { \mathrm { c o t } } ^ { j } \} _ { j = 1 } ^ { m } \big ) \in \mathcal { P } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}
$$

其中 $o_{i}$ 表示第 $i$ 个样本的 T-CoT 生成结果，$\mathcal{D}(\cdot, \cdot)$ 输出与 $q_{\mathrm{c o t}}$ 在打乱候选集中的对齐程度最高的候选 T-CoT 的索引。奖励值为 1 表示 $\mathcal{D}$ 正确选择了来自真实标注数据集的正向 T-CoT，标志着查询-目标 T-CoT 对的良好对齐；奖励值为 0 则意味着 $\mathcal{D}$ 未能选择任何正向 T-CoT，表明查询与目标 T-CoT 输出之间的不对齐。我们在反向方向上对 $\mathcal{R}_{\mathrm{p r o c e s s}}$ 进行对称计算，从正向目标移动至其对应查询，以确保两个方向上的嵌入对齐一致性。该奖励指导推理器对齐查询和目标 T-CoT 输出。由于 T-CoT 是嵌入生成的中间过程，因此由 $\mathcal{D}$ 的选择正确性量化的 T-CoT 对齐直接提高了嵌入质量。总奖励为这三个组成部分的加权组合：

$$
\mathcal { R } _ { \mathrm { t o t a l } } = \alpha \mathcal { R } _ { \mathrm { f o r m a t } } + \beta \mathcal { R } _ { \mathrm { p r o c e s s } } + \gamma \mathcal { R } _ { \mathrm { o u t c o m e } } ,
$$

其中 $\alpha, \beta, \gamma$ 是非负的加权系数，用以平衡三个奖励组成部分在总奖励优化中的相对贡献。使用 GRPO 进行策略优化。我们采用 GRPO 来优化推理者的策略，并使用基于组的奖励来稳定训练过程。对于每个查询-目标对 $q \sim s$，其中 $\boldsymbol { S }$ 表示查询-目标对的训练样本集。我们根据旧策略 $\pi _ { \theta _ { \mathrm { o l d } } }$ 采样 $G = 8$ 个候选 T-CoT 序列 $\{ o _ { i } \} _ { i = 1 } ^ { G }$。优化目标定义为：

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { g r p o } } = \mathbb { E } \underset { \{ o _ { i } \} \sim \pi _ { \theta _ { \mathrm { o l d } } } } { \mathbb { E } } \Bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \bigg ( } & { \operatorname* { m i n } ( r _ { \theta } ( o _ { i } ) A _ { i } , \mathrm { c l i p } ( r _ { \theta } ( o _ { i } ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } ) } \\ & { - \beta \mathbb { D } _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) \bigg ) \Bigg ] , } \end{array}
$$

其中 $r _ { \theta } ( o _ { i } ) = \pi _ { \theta } ( o _ { i } | q ) / \pi _ { \theta _ { \mathrm { o l d } } } ( o _ { i } | q )$ 表示重要性比率，$\epsilon$ 是重要性比率的剪切阈值，$\beta$ 是加权 Kullback-Leibler 散度项的超参数，$\pi _ { \mathrm { r e f } }$ 表示优化前的参考策略模型，$A _ { i } = ( r _ { i } - \mu _ { r } ) / \sigma _ { r }$ 表示优势，其中 $\mu _ { r } = \operatorname { m e a n } ( \{ r _ { 1 } , . . . , r _ { G } \} )$ 和 $\sigma _ { r } = \mathrm { s t d } ( \{ r _ { 1 } , . . . , r _ { G } \} )$ 分别是组奖励的均值和标准差。

# 4 实验

# 4.1 实现细节

我们使用 Deep-Speed Zero2 优化框架训练 Qwen3-VL-2B [2] 和 Qwen3-VL-4B [2] 作为嵌入模型，并采用 VLM2Vec [26] 中的子批次策略。这些模型训练了 2 个 epoch，学习率为 1e-4，权重衰减为 0.01；批次大小计算轻量：2B 模型为 512，4B 模型为 256，同时我们使用低秩适应 (LoRA) [21] 进行微调。针对强化学习，Qwen3-VL-8B [2] 通过 GRPO 算法 [19] 作为推理模型训练 1 个 epoch，批次大小同样计算轻量，为 256，学习率为 3e-6，采用标准 GRPO 超参数。

# 4.2 基线模型与数据集

我们与具有多样架构、模态和规模的代表性多模态嵌入模型进行了比较。这些基线涵盖了图像、视频和视觉文档检索，确保了全面和公正的评估。我们对比了 GME [72]、ColPali [16]、VLM2Vec [26]、LamRA [39]、CAFe [63]、VLM2Vec-V2 [41]、TTE [13]、UME-R1 [30]、InternVideo2 [55]、Unite [18] 和 GVE [20]。在训练阶段，我们遵循 VLM2Vec-V2 [41] 的方法，从三个关键来源构建了一个综合数据集：来自 LLaVA-Hound [70] 的视频-语言指令数据、来自 ViDoRe [16] 和 VisRAG [65] 的视觉文档检索数据，以及来自 MMEB-train [26] 的图像基础视觉-语言任务数据。详细的训练过程、超参数和数据集构建的信息可在补充材料中找到。我们在两个综合基准上进行了评估：MMEB-V2（大规模多模态嵌入基准）[41]：这是一个全面且稳健的基准，包含78个多样化任务，涵盖三种核心视觉模态（图像、视频和视觉文档）。MMEB-V2 在原始 MMEB [26] 的基础上，通过引入五个额外的元任务，专注于视频和视觉文档理解，使元任务总数达到九个。我们采用 Hit@1 作为图像和视频任务的评估指标，对于视觉文档则采用归一化折扣累积增益（NDCG@5）[22]。UVRB（通用视频检索基准）[20]：这是一个包含16个数据集的套件，旨在识别视频检索在任务和领域中的能力差距。UVRB 衡量了文本、复合和视觉检索任务的多维泛化能力，包括粗粒度、细粒度和长上下文场景。我们报告了所有 UVRB 任务的平均精度均值（mAP）。

Table 1: Comparison of performance between baselines and our method on MMEB-V2. CLS: classification, QA: question answering, RET: retrieval, GD: grounding, MRET: moment retrieval, VDR: ViDoRe, VR: VisRAG, OOD: out-of-domain. The highest and second-highest values are highlighted in bold and underline.   

<table><tr><td rowspan="2">Model</td><td colspan="4">Image</td><td colspan="6">Video</td><td colspan="4">VisDoc</td><td rowspan="2">All</td></tr><tr><td>CLS</td><td>QA</td><td>RET</td><td>GD</td><td>Overall CLS</td><td></td><td></td><td>QA RET MRET Overall VDRv1 VDRv2</td><td></td><td></td><td></td><td>VR</td><td>OOD Overall</td><td></td></tr><tr><td># of Datasets</td><td>10</td><td>10</td><td>12</td><td>4 36</td><td>5</td><td>5</td><td>5</td><td>3</td><td>18</td><td>10</td><td>4</td><td>6</td><td>4</td><td>24</td><td>78</td></tr><tr><td colspan="10">Baseline Models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ColPali-V1.3-3B [16]</td><td>40.3 11.5 48.1 40.3</td><td></td><td></td><td></td><td>34.9</td><td>26.7 37.8 21.6</td><td></td><td>25.5</td><td>28.2</td><td>83.6</td><td>52.0</td><td>81.1</td><td>43.1</td><td>71.0</td><td>44.4</td></tr><tr><td>GME-2B [72]</td><td>54.4 29.9</td><td></td><td>66.9</td><td>55.5</td><td>51.9</td><td>34.9 42.0 25.6</td><td></td><td>32.4</td><td>33.9</td><td>86.1</td><td>54.0</td><td>82.5</td><td>43.1</td><td>72.7</td><td>54.1</td></tr><tr><td>GME-7B [72]</td><td>57.7 34.7 71.2 59.3</td><td></td><td></td><td></td><td>56.0</td><td>37.4 50.4 28.4</td><td></td><td>38.2</td><td>38.6</td><td>89.4</td><td>55.6</td><td></td><td>85.0 44.4</td><td>75.2</td><td>57.8</td></tr><tr><td>LamRA-2-7B [39]</td><td>59.2 26.5</td><td></td><td>70.0</td><td>62.7</td><td>54.1</td><td>39.3 42.6 24.3</td><td></td><td>34.6</td><td>35.2</td><td>22.0</td><td>11.5</td><td>37.4 58.2</td><td>21.0</td><td>23.9</td><td>40.4</td></tr><tr><td>LamRA-2.5-7B [39]</td><td>51.7 34.1</td><td></td><td>66.9</td><td>56.7</td><td>52.4</td><td>32.9 42.6 23.2</td><td></td><td>37.6</td><td>33.7</td><td>56.3</td><td>33.3</td><td>51.8</td><td>40.1 33.5</td><td>50.2</td><td>47.4</td></tr><tr><td>VLM2Vec-2B [26]</td><td>58.7</td><td>49.3</td><td>65.0</td><td>72.9</td><td>59.7</td><td>33.4 30.5 39.1 30.0 29.0</td><td>20.6</td><td>33.0 40.6</td><td>29.0 34.0</td><td>49.8 56.9</td><td>13.5 9.4</td><td></td><td>59.1 38.1</td><td>41.6</td><td>47.0 52.3</td></tr><tr><td>VLM2Vec-7B [26]</td><td>62.7 56.9</td><td></td><td>69.4 82.2</td><td></td><td>65.5</td><td></td><td></td><td>38.5</td><td>34.9</td><td>75.5</td><td>44.9</td><td>79.4</td><td>39.4</td><td>46.4</td><td>58.0</td></tr><tr><td>VLM2Vec-V2-2B [41]</td><td>62.9 56.3 69.5</td><td></td><td></td><td>77.3</td><td>64.9 68.1</td><td>39.3 34.3 28.8 45.9 33.9 27.6</td><td></td><td>39.3</td><td>36.4</td><td>78.8</td><td>52.6</td><td>82.7</td><td>42.1</td><td>65.4</td><td>61.2</td></tr><tr><td>VLM2Vec-V2-7B [41]</td><td>65.7 61.5 70.0</td><td></td><td></td><td>85.2</td><td>67.6</td><td>35.8 58.7 34.4</td><td></td><td>39.5</td><td>42.4</td><td>70.7</td><td>49.6</td><td>79.5 38.1</td><td></td><td>69.3 63.9</td><td>60.6</td></tr><tr><td>CAFe-7B [63] TTEs-2B [13]</td><td>63.6 61.7 69.1 87.6 67.9 66.6 70.2</td><td></td><td></td><td>84.1</td><td>70.1</td><td>47.3 49.1</td><td>34.4</td><td>33.2</td><td>32.1</td><td>77.5</td><td>53.2</td><td>83.2</td><td>41.1</td><td>68.8</td><td>63.1</td></tr><tr><td>UME-R1-2B [30]</td><td>64.8 62.8 67.6 77.2</td><td></td><td></td><td></td><td>66.6</td><td>44.3 51.2 32.9</td><td></td><td>39.7</td><td>42.2</td><td>72.4</td><td>46.2</td><td>79.2</td><td>37.2</td><td></td><td></td></tr><tr><td>UME-R1-7B [30]</td><td>67.1</td><td>69.2 71.9 84.9</td><td></td><td></td><td>71.3</td><td>48.6 60.7</td><td>38.2</td><td>39.3</td><td>47.5</td><td>75.7</td><td>50.5</td><td>83.7</td><td>37.6</td><td>63.9</td><td>60.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td>67.1</td><td>64.5</td></tr><tr><td colspan="10"></td><td colspan="7"></td></tr><tr><td>Embed-RL-2B</td><td></td><td>62.8 67.9 68.6 90.4</td><td></td><td>69.2</td><td></td><td>57.0 55.9 45.1</td><td></td><td>49.4</td><td>52.1</td><td>79.9</td><td>52.0 53.4</td><td></td><td>84.6 65.7</td><td>74.1</td><td>66.8</td></tr><tr><td>Embed-RL-4B</td><td></td><td>63.7 70.5 71.3 91.4</td><td></td><td></td><td>70.1</td><td>57.6 58.4 45.1</td><td></td><td>49.5</td><td>53.0</td><td>80.2</td><td></td><td></td><td>84.9 67.1</td><td>74.7</td><td>68.1</td></tr></table>

# 4.3 主要结果

表1比较了我们所提出的Embed-RL模型与MMEB-V2基准上各种基线方法的综合性能。我们的Embed-RL模型与所有基线模型直接相比，始终表现出更优的性能。具体来说，Embed-RL-4B获得了68.1的最佳整体得分，比强有力的下一个基线模型UME-R1-7B高出3.6分。Embed-RL-2B紧随其后，整体得分为66.8，同样超过了所有基线变体。在不同模态下，我们的模型展现出明显且显著的优势。在图像模态中，Embed-RL-4B令人印象深刻地获得了91.4的最佳定位(GD)性能，Embed-RL-2B以90.4排名第二。对于视频模态，Embed-RL-2B和Embed-RL-4B的整体得分均优于所有基线，Embed-RL-2B得分为52.1，Embed-RL-4B得分为53.0，并实现了45.1的视频检索(RET)得分。在视觉文档模态中，我们观察到领域外(OOD)性能的显著提升，其中Embed-RL-4B达到了67.1，Embed-RL-2B为65.7，远超以前的基线结果。这些结果展示了我们所提出的方法在多样化视觉模态和任务类型中的有效性。

Table 2: Video retrieval performance on UVRB. Domain dimensions: Coarse-grained (CG), Finegrained (FG), Long-context (LC). The best and second-best results are marked in bold and underline.   

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>CGFGLC</td></tr><tr><td rowspan=1 colspan=1>InternVideo2-6B [55]</td><td rowspan=1 colspan=1>| 50.441.742.3</td></tr><tr><td rowspan=1 colspan=1>VLM2Vec-V2 [41]</td><td rowspan=1 colspan=1>49.850.276.2</td></tr><tr><td rowspan=1 colspan=1>GME-7B [72]</td><td rowspan=1 colspan=1>51.850.778.8</td></tr><tr><td rowspan=1 colspan=1>Unite-7B [</td><td rowspan=1 colspan=1>54.153.974.6</td></tr><tr><td rowspan=1 colspan=1>GVE-3B [20]</td><td rowspan=1 colspan=1>55.254.176.4</td></tr><tr><td rowspan=1 colspan=1>Embed-RL-2BEmbed-RL-4B</td><td rowspan=1 colspan=1>| 59.154.686.960.755.686.1</td></tr></table>

在视频检索的广泛领域中，得益于我们 T-CoT 有效地准确定位关键词或关键帧，我们的模型在粗粒度（CG）、细粒度（FG）和长上下文（LC）检索任务中展现出显著优势。如表 2 所示，详细展示了不同模型在 UVRB 数据集上三个领域的视频检索性能，我们的 Embed-RL 模型始终表现出色：4B 在 CG 上以 60.7 的分数领先，在 FG 上以 55.6 的分数领先，而 2B 在 LC 上以 86.9 的分数领先，均明显优于所有现有基线。此外，图 3 展示了我们 T-CoT 在文本、图像和视频上的可视化；我们裁剪了边界框和关键帧以实现多模态 CoT 输入，且我们的 T-CoT 准确定位检索需求以提高检索性能。更多的评分和可视化结果见补充材料。

# 4.4 消融研究

为了剖析我们框架中每个组件的贡献，我们对 MMEB-V2 进行了消融实验，其中 Embed-RL-2B 作为模型。

在强化学习阶段的奖励模型分析中，如表3所示，去除强化学习阶段导致整体性能下降1.5分，分数从66.8降至65.3。这个结果验证了强化学习微调对嵌入对齐的重要性。省去加权负样本抽样，我们对比奖励机制的核心模块，导致性能下降0.3分，从66.8降至66.5。这一结果强调了该组件在优先考虑困难负样本以增强区分性嵌入学习中的关键作用。过程奖励旨在奖励逻辑推理步骤并将查询T-CoT与目标T-CoT对齐。它为整体性能贡献了0.8分，因为其去除使得分数从66.8降低至66.0。该组件在视频任务中影响最为显著，性能从52.1降至51.3。这一趋势表明，视频理解在很大程度上依赖逐步推理，并进一步反映了过程奖励在T-CoT对齐中的重要作用。此外，结果奖励旨在奖励最终预测。它为整体性能贡献了1.0分，去除后总分从66.8降至65.8。该奖励确保推理过程与目标任务的一致性。

![](images/3.jpg)  
Fig. 3: Example visualization of our reasoning-driven embedding framework on multimodal retrieval tasks. The figure shows the evidential Traceability CoT reasoning process for video and visual document retrieval.

推理组件对 T-CoT 的影响。如表 4 所示，去掉推理过程而保留答案部分会导致性能下降 1.3 分，从 66.8 降至 65.5。图像定位和视频时刻检索的性能显著下降，分别从 69.2 降至 67.9 和 52.1 降至 50.5，这突显了多模态证据追踪在精细对齐任务中的重要性。去掉多模态线索使得性能降低 1.0 分，从 66.8 降至 65.8，验证了通过边界框和关键帧提取多模态线索以增强多模态表示与检索目标之间对齐的必要性。最关键的是，仅使用原始输入而不应用 T-CoT 会导致整体性能灾难性下降 6.6 分，从 66.8 降至 60.2。视频任务受到的影响最大，性能从 52.1 降至 43.7。这一剧烈下降表明，高质量证据可追溯性 CoT 对于检索准确性的重要性，因为它使模型能够将复杂的检索和理解任务分解为可管理的步骤，并改善跨模态嵌入的质量。

Table 3: Ablation Study on Reward Components in EG-RL stage.   

<table><tr><td>Model</td><td>Image</td><td>Video</td><td>VisDoc All</td><td></td></tr><tr><td>Embed-RL-2B</td><td>69.2</td><td>52.1</td><td>74.1</td><td>66.8</td></tr><tr><td>w/o EG-RL</td><td>68.0</td><td>50.1</td><td>72.7</td><td>65.3</td></tr><tr><td>w/o weighted negative</td><td>68.9</td><td>51.7</td><td>73.9</td><td>66.5</td></tr><tr><td>w/o process reward</td><td>68.3</td><td>51.3</td><td>73.5</td><td>66.0</td></tr><tr><td>w/o outcome reward</td><td>68.1</td><td>51.2</td><td>73.1</td><td>65.8</td></tr></table>

Table 4: Ablation Study on Reasoning Components in T-CoT.   

<table><tr><td>Model</td><td colspan="4">Image Video VisDoc All</td></tr><tr><td>Embed-RL-2B</td><td>69.2</td><td>52.1</td><td>74.1</td><td>66.8</td></tr><tr><td>w/o reasoning</td><td>67.9</td><td>50.5</td><td>73.1</td><td>65.5</td></tr><tr><td>w/o multimodal cues</td><td>68.1</td><td>51.4</td><td>73.3</td><td>65.8</td></tr><tr><td>w/ raw input</td><td>60.4</td><td>43.7</td><td>72.4</td><td>60.2</td></tr></table>

模型对候选者的区分能力的消融研究。我们将具有最高相似度（排除正样本）的排名靠前候选者定义为高度相似候选样本。在此基础上，我们研究通过EG-RL优化推理器如何提高模型区分相似候选者的能力。具体而言，我们计算每个数据集中查询与最相似候选者的相似度与与第二相似候选者的相似度之间的差异，并在RL之前和之后进行对比。这个差异衡量模型是否为正样本分配了显著高于其他高度相似候选者的相似度。如图4所示，我们观察到在不同数据集的三种模态中，经过RL得到的雷达图明显包围了经过RL之前的雷达图。这表明经过RL后计算出的相似度差异增大，拓宽了查询与排名第一候选者和排名第二候选者之间的相似度差距。它表明模型对相似候选者的区分能力得到了有效提升。同时，柱状图显示模型在三种模态数据集上实现了一致的整体提升。这验证了通过RL优化T-CoT增强了模型区分不同候选者的整体能力。

![](images/4.jpg)  
Fig. 4: Similarity difference $\varDelta s = \sin ( \mathrm { q u e r y } , \mathrm { t o p 1 } ) - \sin ( \mathrm { q u e r y } , \mathrm { t o p 2 } )$ before and after EG-RL. Here, $\mathrm { s i m } ( \cdot , \cdot )$ denotes cosine similarity of normalized embeddings, top1 is the most similar positive candidate and top2 the second-most similar. This metric quantifies the model's discriminative ability over similar candidates on multimodal datasets.

对可追踪证据数量和检索指标的消融分析。我们还系统地分析了在强化学习前后，各数据集可追踪证据数量与核心检索指标之间的关系。对于图像和视觉文档数据，我们统计了边界框数量的变化。对于视频数据，我们同样统计了关键帧数量的变化。我们观察到，经过强化学习后，推理器生成的T-CoT倾向于生成更多的边界框。对于视频模态，模型倾向于关注更少的关键帧。检索指标显示出一致且显著的改善，经过强化学习后的曲线完全高于强化学习之前的曲线。对于图像模态，模型捕捉到更多视觉证据，以提高推理准确性和召回率。对于视频模态，模型专注于关键帧，并进行精确的关键帧提取和时序定位，以识别关键内容。这些变化在涉及多目标定位和多人物关系推理的复杂样本上尤为明显。

![](images/5.jpg)  
Fig. 5: Relationship between traceable evidence counts and retrieval metrics across datasets. $\mathrm { H i t @ 1 }$ is employed for Image and Video; NDCG $\ @ 5$ is used for VisDoc. Bounding box counts are shown for Image and VisDoc, while keyframe counts for Video.

# 5 结论

本工作针对生成性通用多模态嵌入（UME）方法的关键局限性进行探讨：链式思维（CoT）仍然是文本专用的，导致检索相关性差，而生成和嵌入目标的联合优化则引发了梯度冲突，妨碍了跨模态匹配。我们提出了Embed-RL，一种基于嵌入引导强化学习（EG-RL）的推理驱动的UME模型，它作为一个解耦的强化学习框架，结合了多模态证据可追溯性链式思维（T-CoT）和面向检索的双重奖励机制，以实现精准的推理与嵌入对齐。在MMEB-V2和UVRB基准上的大量实验表明，Embed-RL在计算约束内优于最先进的同行，显著改善了跨模态检索和领域外泛化任务。本工作表明，针对性推理优化能够显著增强多模态嵌入，为推理驱动的UME提供了一种高效解决方案，并为生成性推理与多模态表示学习的结合提供了宝贵的见解。

# 文本-图像 → 文本 #

空箱子右边的水果是什么？

# 在强化学习之前查询 T-CoT：

空箱子位于图像的左侧，而红苹果直接位于其右侧，确认了所讨论的水果。具体来说，空箱子左侧是一个空箱，而红苹果的坐标为 [560, 480, 800, 580]。因此，空箱子的右侧是苹果。

# 纠正错误信息。

![](images/6.jpg)  

相似度：0.3613

![](images/7.jpg)

# 在强化学习后查询 T-CoT：

图中显示了一个市场摊位，上面摆放着多串香蕉和一堆红苹果。左下角可见一个空的木箱。在这个空箱子的右侧，桌子上展示了更多的香蕉。空箱子位于左下角，右侧的香蕉在桌子上清晰可见，确认该水果为香蕉。

![](images/8.jpg)

# 转变为准确的信息。

相似度：0.4883 目标：该水果是一根香蕉。 <思考：该任务要求识别积极文本输出中的关键概念。文本“该水果是一根香蕉。”明确识别了一种水果及其名称。关键概念包括“水果”和“香蕉”，分别作为主语和宾语。{"text_keywords\": [\"水果\", \"香蕉\"]}</思考>\n<再思考>关键字确认了输出识别出特定类型的水果，符合任务预期内容。</再思考> \n<答案> 文本输出确认香蕉是一种水果。</答案> 图6：T-CoT在EG-RL之前和之后的示例1。

# 文本 → 图像 #

关于支架孔径对组织再生影响的洞察可以总结如下： 1. 支架孔径大小会直接影响细胞的迁移和增殖。较大的孔径能够促进细胞的渗透与扩散，而适当的孔径则有助于细胞在支架内部的生长。 2. 支架孔隙结构对血管化过程至关重要，适宜的孔径可以促进新血管的形成，进而增强组织的营养供应。 3. 孔径过小可能导致细胞聚集和缺氧，而孔径过大则可能降低支架的机械强度和支持能力。 4. 不同类型的细胞对孔径的响应不同。如成骨细胞可能更倾向于在较大孔径的支架中生长，而软骨细胞可能更适合较小孔径的环境。 5. 支架的孔径不仅影响细胞行为，还会对细胞与支架材料的交互作用产生影响，从而影响组织再生的整体效果。 6. 孔径的形状和分布同样重要，不均匀的孔径分布可能会造成力学性能和生物相容性的差异。 综合来看，支架孔径的选择应根据目标组织类型及再生所需的特性进行优化。

# 查询 T-CoT：

一幅展示多孔支架的图像，具有明显的孔隙以及生长的组织或细胞，说明支架孔径与组织再生之间的关系。

# 在强化学习之前的目标 T-CoT:

这张幻灯片的标题是“外周神经的支架”，表明这是一个关于神经再生和支架相互作用的生物医学主题。

# 丢失的基础数据。

# 目标 T-CoT 在强化学习后的表现：

该图像是一个生物医学幻灯片，标题为“外周神经支架”，关键发现是“再生神经的传导速度在支架孔径大约为5微米时恢复正常”。

![](images/9.jpg)  
Fig. 7: Example 2 of T-CoT Before and After EG-RL.

# 文本视频 #

抱歉，我无法协助处理该请求。

# 查询 T-CoT：

一段视频展示了一名男子在户外场地进行标枪投掷的动作。

# 在强化学习之前的目标 T-CoT:

视频展示了一位网球运动员在红黏土场上，运动员在不同的帧中处于挥拍的各个阶段，背景中可见绿色草坪和树木。

相似度：0.3164

# 误导性和与查询无关。

视频展示了一位运动员在红色跑道上进行标枪投掷，跑道旁是绿色的草坪，背景中能看到树木和建筑。运动过程包括投掷中的关键动作和投掷后的随后的动作。

# 相似度：0.5742

查询对齐且以关键动作为重点。

![](images/10.jpg)  
Fig. 8: Example 3 of T-CoT Before and After EG-RL.

# References

1. Anne Hendricks, L., Wang, O., Shechtman, E., Sivic, J., Darrell, T., Russell, B.: Localizing moments in video with natural language. In: Proceedings of the IEEE international conference on computer vision. pp. 58035812 (2017)   
2. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Z. Li M. Li, M. LiK.Ln Z. i J. Lu X. Liu J. Liu C. Liu Y. D. Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Su Y., Tag, J., Tu, J. Wan, J., Wa, P. Wan, P. Wa, Q. ag,Y. Xie, T. XuY. Xu H. Xu, J  Z. M.  J A.Yu B. , F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., Zhu, K.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025)   
3.Bai, S., Chen, K., Liu X. Wng, J. Ge, W. Song, S. Dag, K., Wa, P. Wan, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025)   
4.Bolya, D., Huang, P.Y., Sun, P., Cho, J.H., Madotto, A., Wei, C., Ma, T., Zhi, J., Rajasegaran, J., Rasheed, H., et al.: Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181 (2025)   
5.Cai, Q., Liang, H., Dong, H., Qiang, M., An, R., Han, Z., Zhu, Z., Cui, B., Zhang, W.: Lovr: A benchmark for long video retrieval in multimodal contexts. arXiv preprint arXiv:2505.13928 (2025)   
6. Cao, M., Zhao, H., Zhang, C., Chang, X., Reid, I., Liang, X.: Ground-r1: Incentivizing grounded visual reasoning via reinforcement learning. arXiv preprint arXiv:2505.20272 (2025)   
7. Chai, W., Song, E., Du, Y., Meng, C., Madhavan, V., Bar-Tal, O., Hwang, J.N., Xie, S., Manning, C.D.: Auroracap: Effient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051 (2024)   
8. Chen, H., Liu, H., Luo, Y., Wang, L., Yang, N., Wei, F., Dou, Z.: Moca: Modalityaware continual pre-training makes better bidirectional multimodal embeddings. arXiv preprint arXiv:2506.23115 (2025)   
Chen, H., Wang, L., Yang, N., Zhu, Y., Zhao, Z., Wei, F., Dou, Z.: mme5: Improving multimodal multilingual embeddings via high-quality synthetic data. arXiv preprint arXiv:2502.08468 (2025)   
0. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 28182829 (2023)   
1. Chu, X., Chen, X., Wang, G., Tan, Z., Huang, K., Lv, W., Mo, T., Li, W.: Qwen look again: Guiding vision-language reasoning models to re-attention visual information. arXiv preprint arXiv:2505.23558 (2025)   
Chung, J., Kim, J., Kim, S., Lee J., Kim, M.S., Yu, Y.: Don't look only once: Towards multimodal interactive reasoning with selective visual revisitation. arXiv preprint arXiv:2505.18842 (2025)   
3. Cui, X., Cheng, J., Chen, H.y., Shukla, S.N., Awasthi, A., Pan, X., Ahuja, C., Mishra, S.K., Yang, Y., Xiao, J., et al.: Think then embed: Generative context improves multimodal embedding. arXiv preprint arXiv:2510.05014 (2025)   
4. Duan, C., Fang, R., Wang, Y., Wang, K., Huang, L., Zeng, X., Li, H., Liu, X.: Gotr1: Unleashing reasoning capability of mllm for visual generation with reinforcement learning. arXiv preprint arXiv:2505.17022 (2025)   
15. Fan, Y., He, X., Yang, D., Zheng, K., Kuo, C.C., Zheng, Y., Narayanaraju, S.J., Guan, X., Wang, X.E.: Grit: Teaching mllms to think with images. arXiv preprint arXiv:2505.15879 (2025)   
16. Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., Colombo, P.: Colpali: Effcient document retrieval with vision language models. arXiv preprint arXiv:2407.01449 (2024)   
17. Geng, X., Xia, P., Zhang, Z., Wang, X., Wang, Q., Ding, R., Wang, C., Wu, J., Zo Y. Li K. al. Wee: ren  s-aee research agent. arXiv preprint arXiv:2508.05748 (2025)   
18Gu, T. Yag, K., Feng, Z., Wag, X., Zhag, Y. Lng, D., Chen, Y., Cai, W., Deng, J.: Breaking the modality barrier: Universal embedding learning with multimodal llms. arXiv preprint arXiv:2504.17432 (2025)   
19. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)   
20. Guo, Z., Li, M., Zhang, Y., Long, D., Xie, P., Chu, X.: Towards universal video retrieval: Generalizing video embedding via synthesized multimodal pyramid curriculum. arXiv preprint arXiv:2510.27571 (2025)   
21. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., e a Lora: Low-rank adaptation of large language models. ICLR 1(2), 3 (2022)   
22. Järvelin, K., Kekäläinen, J.: Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems (TOIS) 20(4), 422446 (2002)   
D. Zhu L oS. Xu P. Lu H.Ye J. ZhaoF.Treean the power of llms for enhanced large-scale table understanding. arXiv preprint arXiv:2411.08516 (2024)   
24. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning x uei  Inal ca er pp. 49044916. PMLR (2021)   
Jiang, T. Song, M. Zhang, Z., Huang, H. Deg, W., Sun, F., Zhang, Q., Wang, D., Zhuang, F.: E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580 (2024)   
26. Jiang, Z., Meng, R., Yang, X., Yavuz, S., Zhou, Y., Chen, W.: Vlm2vec: Training vision-language models for massive multimodal embedding tasks. arXiv preprint arXiv:2410.05160 (2024)   
27. Kong, F., Zhang, J., Liu, Y., Zhang, H., Feng, S., Yang, X., Wang, D., Tian, Y., Zhang, F., Zhou, G., et al.: Modality curation: Building universal embeddings for advanced multimodal information retrieval. arXiv preprint arXiv:2505.19650 (2025)   
Lai X. Tin Z.henY. LiY.uanY. Liu S. Jia . iementation via large language model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 95799589 (2024)   
29. Lan, Z., Niu, L., Meng, F., Zhou, J., Su, J.: Llave: Large language and vision embedding models with hardness-weighted contrastive learning. arXiv preprint arXiv:2503.04812 (2025)   
30. Lan, Z., Niu, L., Meng, F., Zhou, J., Su, J.: Ume-r1: Exploring reasoning-driven generative multimodal embeddings. arXiv preprint arXiv:2511.00405 (2025)   
31. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024)   
32. Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., Li, C.: Llava-nextinterleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895 (2024)   
33. Li, G., Xu, J., Zhao, Y., Peng, Y.: Dyfo: A training-free dynamic focus visual search for enhancing lmms in fine-grained visual understanding. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 90989108 (2025)   
3. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrappig language-image pretraining with frozen image encoders and large language models. In: International conference on machine learning. pp. 1973019742. PMLR (2023)   
35. Liao, Z., Xie, Q., Zhang, Y., Kong, Z., Lu, H., Yang, Z., Deng, Z.: Improved visual-spatial reasoning via r1-zero-like training. arXiv preprint arXiv:2504.00883 (2025)   
36. Lin, S.C., Lee, C., Shoeybi, M., Lin, J., Catanzaro, B., Ping, W.: Mm-embed: Universal multimodal retrieval with multimodalllms. arXiv preprint arXiv:2411.02571 (2024)   
37. Lin, Z., Cen, S., Jiang, D., Karhade, J., Wang, H., Mitra, C., Ling, T., Huang, Y., Liu, S., Chen, M., et al.: Towards understanding camera motions in any video. arXiv preprint arXiv:2504.15376 (2025)   
38. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 3489234916 (2023)   
Lu Y. Za Y. ai J. J X. Hu,Y.Yo, J. WaY. Xie, W. : Large multimodal model as your advanced retrieval assistant. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 40154025 (2025)   
40. Luo, H., Ji, L., Zhong, M., Chen, Y., Lei, W., Duan, N., Li, T.: Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing 508, 293304 (2022)   
41. Meng, R., Jiang, Z., Liu, Y., Su, M., Yang, X., Fu, Y., Qin, C., Chen, Z., Xu, R., Xiog C. et al Vlm2vec-v2: Advancig multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590 (2025)   
42.Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018)   
43. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from nal language upervision. In: International conference nmachie learning. pp. 87488763. PmLR (2021)   
4.Su, Z., Li, L., Song, M., Hao, Y., Yang, Z., Zhang, J., Chen, G., Gu, J., Li, J., Qu, X., et al.: Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617 (2025)   
45. Su, Z., Xia, P., Guo, H., Liu, Z., Ma, Y., Qu, X., Liu, J., Li, Y., Zeng, K., Yang, Z., et al.: Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918 (2025)   
46. Thirukovalluru, R., Meng, R., Liu, Y., Su, M., Nie, P., Yavuz, S., Zhou, Y., Chen, . Dhi B.  l Brekg he ba barr cvle smart batch mining. arXiv preprint arXiv:2505.11293 (2025)   
47. Wang, H., Li, X., Huang, Z., Wang, A., Wang, J., Zhang, T., Zheng, J., Bai, S., Z. Fe J.  lTrb vvis : Evaluation and methodology. arXiv preprint arXiv:2507.07999 (2025)   
8Wang J., Wang C. Huang, K. Huang, J., Jin, L: Videocli-x: Advancig ong description understanding for video clip models. arXiv preprint arXiv:2410.00741 (2024)   
Wang, J. uan, L. Zhang Y. Sun, H. Tars Recis or ain al large video description models. arXiv preprint arXiv:2407.00634 (2024)   
50. Wang, P., Ling, H.: Svqa-r1: Reinforcing spatial reasoning in mllms via viewconsistent reward optimization. arXiv preprint arXiv:2506.01371 (2025)   
51. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)   
Wg, Q., Lu, J., Liag, J. Jiag, Y., Zhag, Y., Chen, J. Zhe,Y. n, X., Wan, P., Yue, X., et al.: Vr-thinker: Boosting video reward models through thinking-with-image reasoning. arXiv preprint arXiv:2510.10518 (2025)   
53. Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., Zhou, D.: Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022)   
5Wa,Y. He, Y., Li Y., Li, K. u, J., Ma, X., Li, X., e, G. Chen, X. Wa, Y.  Inteiaio-x ata rulalntan and generation. arXiv preprint arXiv:2307.06942 (2023)   
55. Wng, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheg, R., Wang, Z., Shi, Y., et al.: Internvideo2: Scaling foundation models for multimodal video understanding. In: European Conference on Computer Vision. pp. 396416. Springer (2024)   
Wang, Y., Liu, W., Niu, J., Zhang, H., Tang, Y.: Vg-refiner: Towards tool-refined referring grounded reasoning via agentic reinforcement learning. arXiv preprint arXiv:2512.06373 (2025)   
57. Wei, C., Chen, Y., Chen, H., Hu, H., Zhang, G., Fu, J., Ritter, A., Chen, W.: Uniir: Training and benchmarking universal multimodal information retrievers. In: European Conference on Computer Vision. pp. 387404. Springer (2024)   
58. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q.V., Zhou, D., et al.: Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35, 2482424837 (2022)   
59. Wu, M., Yang, J., Jiang, J., Li, M., Yan, K., Yu, H., Zhang, M., Zhai, C., Nahrstedt, K.: Vtool-r1: Vlms learn to think with images via reinforcement learning on multimodal tool use. arXiv preprint arXiv:2505.19255 (2025)   
60 Xu, G., Jin, P., Wu, Z., Li, H., Song, Y., Sun, L., Yuan, L.: Llava-cot: Let vision language models reason step-by-step. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 20872098 (2025)   
61. Xu, J., Mei, T., Yao, T., Rui, Y.: Msr-vtt: A large video description dataset for bridging video and language. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 52885296 (2016)   
62. Xu, Y., Li, X., Yang, Y., Meng, D., Huang, R., Wang, L.: Carebench: A fine-grained benchmark for video captioning and retrieval (2025), https://arxiv.org/abs/ 2501.00513   
63.Yu, H., Zhao, Z., Yan, S., Korycki, L., Wang, J., He, B., Liu, J., Zhang, L., Fan, X., Yu, H.: Cafe: Unifying representation and generation with contrastive-autoregressive finetuning. arXiv preprint arXiv:2503.19900 (2025)   
64. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., L, L., et al.: Dapo: An open-source lm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476 (2025)   
6Yu, S., Tang, C., Xu, B., Cui, J., Ran, J., Yan, Y., Liu, Z., Wang, S., Han, X., Liu, Z., et al.: Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594 (2024)   
66. Yuan, H., Ni, J., Liu, Z., Wang, Y., Zhou, J., Liang, Z., Zhao, B., Cao, Z., Dou, Z., Wen, J.R.: Momentseeker: A task-oriented benchmark for long-video moment retrieval. arXiv preprint arXiv:2502.12558 (2025)   
Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pretraining. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1197511986 (2023)   
68. Zhang, H., Gu, X., Li, J., Ma, C., Bai, S., Zhang, C., Zhang, B., Zhou, Z., He, D., Tag, Y. Thinking with video: Multimodal too-augmente reinorcement learing for long video reasoning. arXiv preprint arXiv:2508.04416 (2025)   
69. Zhang, K., Luan, Y., Hu, H., Lee, K., Qiao, S., Chen, W., Su, Y., Chang, M.W.: Magiclens: Self-supervised image retrieval with open-ended instructions. arXiv preprint arXiv:2403.19651 (2024)   
Za R. Gui L. Sun, Z. Feng Y. Xu,K. ZY. Fu, . Li C. H, A.G., Bisk, Y., et al.: Direct preference optimization of video large multimodal models from language model reward. In: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). pp. 694717 (2025)   
71. Zhang, X., Zhang, Y., Xie, W., Li, M., Dai, Z., Long, D., Xie, P., Zhang, M., Li, W., Zhang, M.: Gme: Improving universal multimodal retrieval by multimodal llms. arXiv preprint arXiv:2412.16855 (2024)   
7. Zhang, X., Zhang, Y., Xie, W., Li, M., Dai, Z., Long, D., Xie, P., Zhang, M., Li, W., Zhang, M.: Bridging modalities: Improving universal multimodal retrieval by multimodal large language models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 92749285 (2025)   
73. Zhang, X., Du, C., Pang, T., Liu, Q., Gao, W., Lin, M.: Chain of preference optimization: Improving chain-of-thought reasoning in llms. Advances in Neural Information Processing Systems 37, 333356 (2024)   
74. Zheng, C., Liu, S., Li, M., Chen, X.H., Yu, B., Gao, C., Dang, K., Liu, Y., Men, R., Yang, A., et al.: Group sequence policy optimization. arXiv preprint arXiv:2507.18071 (2025)   
75. Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., Yu, X.: Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362 (2025)   
76. Zhou, J., Liu, Z., Liu, Z., Xiao, S., Wang, Y., Zhao, B., Zhang, C.J., Lian, D., Xiong, Y.: Megapairs: Massive data synthesis for universal multimodal retrieval. arXiv preprint arXiv:2412.14475 (2024)   
7Zhou, J., Xiong, Y., Liu, Z., Liu, Z., Xiao, S., Wang, Y., Zhao, B., Zhang, C.J., Lian, D.: Megapairs: Massive data synthesis for universal multimodal retrieval. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 1907619095 (2025)   
78. Zhu, B., Lin, B., Ning, M., Yan, Y., Cui, J., HongFa, W., Pang, Y., Jiang, W., Zhang, J., Li, Z., Zhang, C.W., Li, Z., Liu, W., Yuan, L.: Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. In: The Twelfth International Conference on Learning Representations (2024)   
79. Zhu, L., Chen, Q., Shen, X., Cun, X.: Vau-r1: Advancing video anomaly understanding via reinforcement fine-tuning. arXiv preprint arXiv:2505.23504 (2025)

# Embed-RL: Reinforcement Learning for Reasoning-Driven Multimodal Embeddings

Supplementary Material

# A Additional details

In this supplementary material, we elaborate on further insights, provide detailed derivations, and include additional qualitative results to enhance the comprehensiveness of our work.

# B Training Details

# B.1 Contrastive Learning

We trained the Qwen3-VL-2B-Instruct [2] and Qwen3-VL-4B-Instruct [2] models using the DeepSpeed Zero2 optimization framework. Key training hyperparameters were summarized as follows: The training process was conducted for 2 epochs with a batch size of 512 for Qwen3-VL-2B-Instruct and 256 for Qwen3- VL-4B-Instruct. Following the sub-batch training scheme in VLM2Vec [26], we adopted a sub-batch training strategy that ensured samples in each sub-batch are drawn from the same dataset, where the sub-batch size was set to 256 for Qwen3-VL-2B-Instruct and 128 for Qwen3-VL-4B-Instruct. We set the initial learning rate to 1e-4, using a cosine learning rate scheduler with 10 warm-up steps and a weight decay of 0.01. We employed Low-Rank Adaptation (LoRA) [21] for fine-tuning: the rank $r$ and scaling factor $\alpha$ were set to 64 and 128 for Qwen3- VL-2B-Instruct, and 96 and 192 for Qwen3-VL-4B-Instruct, respectively. For comparison, UME-R1 [30] was trained with a batch size of 1024 and TTE [13] was trained with 8192, indicating that our training scale was significantly smaller than both methods. Notably, our experimental results could be further scaled up with more abundant computational resources.

# B.2 Embedder-Guided Reinforcement Learning

In the reinforcement learning stage, we trained the Qwen3-VL-8B-Instruct [2] Reasoner with the GRPO [19]. We adopted in-batch negative contrastive rewards for optimization, with GRPO hyperparameters set as group size $G = 8$ , clipping parameter $\varepsilon = 0 . 2$ , and KL-divergence coefficient $\beta = 0 . 0 1$ . We set batch size to 256, learning rate to 3e-6, and trained the model for one epoch. We also restricted each step to samples from the same dataset to avoid overly simple negative samples affecting optimization.

For the embedder-guided outcome reward $\mathcal { R } _ { \mathrm { o u t c o m e } }$ , two core hyperparameters balance retrieval accuracy and similarity margin calculation: the top- $k$ parameter for retrieval accuracy $\operatorname { A c c } _ { k }$ and the temperature parameter $\tau$ for softmax-weighted negative sampling.

Top- $k$ Retrieval Accuracy $k$ : We set the parameter $k$ in $\mathrm { A c c } _ { k } ( e _ { q _ { i } } , t _ { i } ^ { + } )$ to 8. Specifically, $\mathrm { A c c } _ { 8 } ( e _ { q _ { i } } , t _ { i } ^ { + } )$ checks whether all 8 T-CoT rollouts of a query-target pair rank in the top-8 by cosine similarity to $e _ { q _ { i } }$ , which captures the embedding alignment consistency critical for stable GRPO reward optimization.

Temperature Parameter ( $\tau$ ): The temperature $\tau$ scales the softmax weights for hard negative sampling in the similarity margin calculation, which is defined as:

$$
\mathbb { E } _ { \tau } \big [ \mathrm { s i m } ( e _ { q _ { i } } , e _ { t _ { j } ^ { - } } ) \big ] = \frac { \sum _ { j \neq i } \mathrm { e x p } \left( \frac { \mathrm { s i m } ( e _ { q _ { i } } , e _ { t _ { j } ^ { - } } ) } { \tau } \right) \cdot \mathrm { s i m } ( e _ { q _ { i } } , e _ { t _ { j } ^ { - } } ) } { \sum _ { j \neq i } \mathrm { e x p } \left( \frac { \mathrm { s i m } ( e _ { q _ { i } } , e _ { t _ { j } ^ { - } } ) } { \tau } \right) } .
$$

We set $\tau = 0 . 5$ for all experiments, a value chosen to emphasize hard negatives while avoiding overfitting to noisy negative samples.

For process reward, we employ an independent Qwen3-VL-8B-Instruct [2] as the pretrained VLM Discriminator $\mathcal { D }$ . It performs listwise comparison to align query T-CoT outputs with corresponding target outputs. The discriminator selects the candidate T-CoT that best matches the query. Selection correctness quantifies alignment quality and forms the process reward signal.

We set $\alpha = 0 . 0 5$ , $\beta = 0 . 8$ , and $\gamma = 0 . 2$ for the total reward $\mathcal { R } _ { \mathrm { t o t a l } }$ to balance the contributions of format, process, and outcome rewards respectively.

# B.3 Multimodel Vision Processing

For visual input processing, we set specific pixel constraints for images and videos to balance computational efficiency and feature fidelity:

For images: MIN $\mathrm { P I X E L S } = 1 2 8 { \times } 3 2 { \times } 3 2$ and MAX $\mathrm { P I X E L S } = 7 6 8 \times 3 2 \times 3 2$

For videos: VIDEO_MIN_PIXELS $= 1 2 8 \times 3 2 \times 3 2$ , VIDEO_MAX_PIXELS $= 3 0 0 \times 3 2 \times 3 2$ , and VIDEO_TOTAL_PIXELS $= 3 0 0 \times 3 2 \times 3 2 \times 8$ .

Video frame sampling hyperparameters were fixed as FRAME _FACTOR = 2, FPS = 2.0, FPS_MIN_FRAMES = 8 and FPS_MAX_FRAMES = 8.

For multimodal cropping, we first convert the relative coordinates of bounding boxes, which are scaled to the range 0-1000, into the original image coordinates, then conduct cropping on the raw image. Keyframes corresponding to the sampled frames are re-extracted and concatenated between the </thinking $>$ and <rethink $>$ tokens.

# C Detailed Dataset Construction

# C.1 Data Sources and Initial Sampling Strategy

To enable effective multi-modal task training, we adopt the training data paradigm of VLM2Vec-V2 [41] and build a comprehensive dataset from three core sources: (1) video-language instruction data (LLaVA-Hound [70]), (2) visual document retrieval data (ViDoRe [16] and VisRAG [65]), and (3) image-based vision task data (MMEB-train [26]). We apply a stratified sampling strategy across data modalities to ensure balanced coverage:

Image-based datasets: Maximum 50,000 samples per dataset Document-based datasets: Maximum 100,000 samples per dataset Video-based datasets: Maximum 300,000 samples per dataset

Full sampling is used if the original dataset size is smaller than the above maximum. As shown in Table 5, we presented the exact number of samples selected for each dataset in our experiments.

Table 5: Statistics of Initial Sampling and CoT-guided Filtering.   

<table><tr><td>Dataset</td><td colspan="4">Initial Samples Filtered Samples Retention Ratio Weight</td></tr><tr><td colspan="6">Modality Image-based (MMEB-train)</td></tr><tr><td>A-OKVQA</td><td>50,000</td><td>37,929</td><td>75.86%</td><td>0.26</td><td>Text-Image → Text</td></tr><tr><td>CIRR</td><td>50,000</td><td>35,085</td><td>70.17%</td><td>0.43</td><td>Text-Image → Text-Image</td></tr><tr><td>ChartQA</td><td>50,000</td><td>39,512</td><td>79.02%</td><td>0.35</td><td>Text-Image → Text</td></tr><tr><td>DocVQA</td><td>50,000</td><td>47,401</td><td>94.80%</td><td>0.84</td><td>Text-Image → Text</td></tr><tr><td>HatefulMemes</td><td>25,500</td><td>16,572</td><td>64.99%</td><td>0.30</td><td>Text-Image → Text</td></tr><tr><td>ImageNet-1K</td><td>50,000</td><td>44,409</td><td>88.82%</td><td>2.25</td><td>Text-Image → Text</td></tr><tr><td>InfographicsVQA</td><td>50,000</td><td>40,746</td><td>81.49%</td><td>0.31</td><td>Text-Image → Text</td></tr><tr><td>MSCOCO</td><td>50,000</td><td>26,429</td><td>52.86%</td><td>3.78</td><td>Text-Image → Text-Image</td></tr><tr><td>MSCOCO-i2t</td><td>50,000</td><td>46,596</td><td>93.19%</td><td>2.58</td><td>Text-Image → Text</td></tr><tr><td>MSCOCO-t2i</td><td>50,000</td><td>43,173</td><td>86.35%</td><td>2.32</td><td>Text → Text-Image</td></tr><tr><td>N24News</td><td>50,000</td><td>30,320</td><td>60.64%</td><td>1.65</td><td>Text-Image → Text</td></tr><tr><td>NIGHTS</td><td>47,823</td><td>43,167</td><td>90.26%</td><td>0.23</td><td>Text-Image → Text-Image</td></tr><tr><td>OK-VQA</td><td>27,027</td><td>19,900</td><td>73.63%</td><td>0.25</td><td>Text-Image → Text</td></tr><tr><td>SUN397</td><td>50,000</td><td>45,864</td><td>91.73%</td><td>0.22</td><td>Text-Image → Text</td></tr><tr><td>VOC2007</td><td>23,532</td><td>20,454</td><td>86.92%</td><td>0.24</td><td>Text-Image → Text</td></tr><tr><td>Visual7W</td><td>50,000</td><td>41,677</td><td>83.35%</td><td>1.68</td><td>Text-Image → Text</td></tr><tr><td>VisDial</td><td>50,000</td><td>34,652</td><td>69.30%</td><td>3.75</td><td>Text → Text-Image</td></tr><tr><td>VisualNews-i2t</td><td>50,000</td><td>34,364</td><td>68.73%</td><td>2.91</td><td>Text-Image → Text</td></tr><tr><td>VisualNews-t2i WebQA</td><td>50,000 50,000</td><td>28,684 43,910</td><td>57.37% 87.82%</td><td>3.49 0.23</td><td>Text → Text-Image Text → Text-Image</td></tr><tr><td colspan="6">Video-based (LLaVA-Hound)</td></tr><tr><td>Caption Retrieval</td><td>300,000</td><td>283,721</td><td>94.57%</td><td>5.27</td><td>Video → Text</td></tr><tr><td>Video QA</td><td>300,000</td><td>273,906</td><td>91.30%</td><td>4.38</td><td>Video-Text → Text</td></tr><tr><td>Video Retrieval</td><td>300,000</td><td>260,410</td><td>86.80%</td><td>5.76</td><td>Text → Video</td></tr><tr><td colspan="6">Document-based</td></tr><tr><td>ViDoRe</td><td>100,000</td><td>83,964</td><td>83.96%</td><td>5.0</td><td>Text-Image → Text</td></tr><tr><td>VisRAG</td><td>100,000</td><td>60,266</td><td>60.27%</td><td>6.0</td><td>Text → Image</td></tr><tr><td>Image-based</td><td>1,123,882</td><td>865,074</td><td>76.97%</td><td></td><td>Image-centric</td></tr><tr><td>Video-based</td><td>900,000</td><td>819,037</td><td>91.02%</td><td>-</td><td>Video-centric</td></tr><tr><td>Document-based</td><td>200,000</td><td>144,230</td><td>72.12%</td><td></td><td>Document-centric</td></tr><tr><td>Total</td><td>2,223,882</td><td>1,828,341</td><td>82.21%</td><td></td><td>Multimodal</td></tr></table>

We exclude three classification datasets (HatefulMemes, N24News, VOC2007) from the first-stage contrastive learning. This stage only employs contrastive loss under a single-dataset sub-batch constraint. Their limited number of categories results in false negatives and noise with large sub-batches, which impairs representation learning. This explains the model's weak performance on image classification, as it never encounters these samples during training.

# C.2 CoT-guided Relevance Filtering

We generate chain-of-thought (CoT) annotations for queries and positive samples using Qwen3-VL-8B [2], and further conduct strict relevance filtering via a custom prompt with the same model to discard annotations that are irrelevant to or conflict with the query task. To mitigate noise in contrastive learning, only samples labeled "No" are retained. Table 5 lists the sample size, retention ratio, training weight, and modality of each dataset.

# k/> CoT Relevance and Conflict Judgment Prompt

Your task is to determine whether the content  'querycot and 'poscotare \*\*obviously ielevant\*\* or \*\*obviously conflicting\*\* based on the given qry' (task description).   
\*\*Rules\*\*:   
Ony outut a sige wor: "Ye" obviusly irrelevant/conflicing)or "No" relevantand not conficting)   
Oi ene nen  queyct nsct have  oglc  ec or to the qry task   
Oviusy conlcigThe o conclusions/key oati queycot nd scot  ualy contradictory   
Only judge "obvious" cases — if the relevance is ambiguous, output "No"

# C.3 RL Dataset Sampling

Using the high-quality CoT-fltered dataset, we apply equidistant sampling to construct the reinforcement learning (RL) training set, ensuring uniform distribution across challenging sub-datasets, as shown in Table 6.

Table 6: Reinforcement Learning Dataset Construction and Sampling Strategy.   

<table><tr><td>Sampled Dataset</td><td>RL Samples</td><td>Modality</td></tr><tr><td>A-OKVQA</td><td>1,000</td><td>Text-Image → Text</td></tr><tr><td>llavahound video retrieval</td><td>2,000</td><td>Text → Video</td></tr><tr><td>ViDoRe colpali train set</td><td>2,000</td><td>Text-Image → Text</td></tr><tr><td>VisualNews-t2i</td><td>1,000</td><td>Text → Text-Image</td></tr><tr><td>VisualNews-i2t</td><td>1,000</td><td>Text-Image → Text</td></tr><tr><td>VisRAG-Ret-Train-In-domain-data</td><td>2,000</td><td>Text → Image</td></tr><tr><td>CIRR</td><td>2,000</td><td>Text-Image → Text-Image</td></tr><tr><td>ChartQA</td><td>1,000</td><td>Text-Image → Text</td></tr><tr><td>OK-VQA</td><td>1,000</td><td>Text-Image → Text</td></tr><tr><td>llavahound qa</td><td>2,000</td><td>Video-Text → Text</td></tr><tr><td>llavahound caption retrieval</td><td>2,000</td><td>Video → Text</td></tr><tr><td>Visual7W</td><td>1,000</td><td>Text-Image → Text</td></tr><tr><td>N24News</td><td>1,000</td><td>Text-Image → Text</td></tr><tr><td>Total</td><td>19,000</td><td>Multimodal</td></tr></table>

# D Detailed Scores of MMEB-V2

We report the detailed metrics for our MMEB-V2 dataset [41], as shown in Table 7 and Table 8. The highest and second-highest values are highlighted in bold and underlined, respectively.

Table 7: Detailed results of baselines and Embed-RL on full MMEB-v2 benchmark. Video and Visual Doc results are shown on the next table.   

<table><tr><td></td><td colspan="9">v1MVVAF1- 0E1RB</td></tr><tr><td>Avg - All (78 tasks)</td><td>44.4</td><td>57.8</td><td>52.3</td><td>58.0</td><td>60.6</td><td>60.1</td><td>64.5</td><td>66.8</td><td>68.1</td></tr><tr><td>Avg - Image (36 tasks, Hit@1)</td><td>34.9</td><td>56.0</td><td>65.5</td><td>64.9</td><td>67.6</td><td>66.6</td><td>71.3</td><td>69.2</td><td>71.2</td></tr><tr><td>Avg - Video (18 tasks, Hit@1)</td><td>28.2</td><td>38.4</td><td>33.7</td><td>34.6</td><td>42.4</td><td>42.2</td><td>47.5</td><td>52.1</td><td>53.0</td></tr><tr><td>Avg - Visdoc (24 tasks, NDCG@5)</td><td>71.0</td><td>75.2</td><td>46.4</td><td>65.4</td><td>63.9</td><td>63.9</td><td>67.1</td><td>74.1</td><td>74.7</td></tr><tr><td>I-CLS (10)</td><td>40.3</td><td>57.7</td><td>62.7</td><td>62.9</td><td>63.6</td><td>64.8</td><td>67.1</td><td>62.8</td><td>63.7</td></tr><tr><td>I-QA (10)</td><td>11.5</td><td>34.7</td><td>56.9</td><td>56.3</td><td>61.7</td><td>62.8</td><td>69.2</td><td>67.9</td><td>70.5</td></tr><tr><td>I-RET (12)</td><td>48.1</td><td>71.2</td><td>69.4</td><td>69.5</td><td>69.1</td><td>67.6</td><td>71.9</td><td>68.6</td><td>71.3</td></tr><tr><td>I-VG (4)</td><td>40.3</td><td>59.3</td><td>82.2</td><td>77.3</td><td>87.6</td><td>77.2</td><td>84.9</td><td>90.4</td><td>91.4</td></tr><tr><td>V-CLS (5)</td><td>26.7</td><td>37.4</td><td>39.1</td><td>39.3</td><td>35.8</td><td>44.3</td><td>48.6</td><td>57.0</td><td>57.6</td></tr><tr><td>V-QA (5)</td><td>37.8</td><td>50.4</td><td>30.0</td><td>34.3</td><td>58.7</td><td>51.0</td><td>60.7</td><td>55.9</td><td>58.4</td></tr><tr><td>V-RET (5)</td><td>21.6</td><td>28.4</td><td>29.0</td><td>28.8</td><td>34.4</td><td>32.9</td><td>38.2</td><td>45.1</td><td>45.1</td></tr><tr><td>V-MR (3)</td><td>25.5</td><td>37.0</td><td>38.9</td><td>36.8</td><td>39.5</td><td>39.7</td><td>39.3</td><td>49.4</td><td>49.5</td></tr><tr><td>VD-Vidore-V1 (10)</td><td>83.6</td><td>89.4</td><td>56.9</td><td>75.7</td><td>70.7</td><td>72.4</td><td>75.7</td><td>79.9</td><td>80.2</td></tr><tr><td>VD-Vidore-V2 (4)</td><td>52.0</td><td>55.6</td><td>9.4</td><td>45.1</td><td>49.6</td><td>46.2</td><td>50.5</td><td>52.0</td><td>53.4</td></tr><tr><td>VD-VisRAG (6)</td><td>81.1</td><td>85.0</td><td>59.1</td><td>79.6</td><td>79.5</td><td>79.2</td><td>83.7</td><td>84.6</td><td>84.9</td></tr><tr><td>VD-OOD (4)</td><td>43.1</td><td>44.4</td><td>38.1</td><td>39.6</td><td>38.1</td><td>37.2</td><td>37.6</td><td>65.7</td><td>67.1</td></tr><tr><td>ImageNet-1K</td><td>42.4</td><td>64.6</td><td>80.1</td><td>80.8</td><td>77.3</td><td>75.3</td><td>80.4</td><td>78.0</td><td>79.5</td></tr><tr><td>N24News</td><td>25.5</td><td>50.5</td><td>79.7</td><td>72.9</td><td>83.2</td><td>81.1</td><td>82.3</td><td>44.9</td><td>48.3</td></tr><tr><td>HatefulMemes</td><td>50.6</td><td>53.6</td><td>69.7</td><td>56.3</td><td>78.7</td><td>75.2</td><td>79.0</td><td>65.0</td><td>66.2</td></tr><tr><td>VOC2007</td><td>69.8</td><td>80.3</td><td>80.7</td><td>85.0</td><td>89.8</td><td>80.0</td><td>90.8</td><td>78.7</td><td>79.5</td></tr><tr><td>SUN397</td><td>56.1</td><td>69.5</td><td>77.4</td><td>71.0</td><td>79.9</td><td>79.4</td><td>80.3</td><td>75.4</td><td>79.2</td></tr><tr><td>Place365</td><td>27.5</td><td>39.1</td><td>37.4</td><td>35.9</td><td>45.0</td><td>42.6</td><td>46.8</td><td>43.9</td><td>43.1</td></tr><tr><td>ImageNet-A</td><td>14.9</td><td>41.2</td><td>58.1</td><td>47.4</td><td>55.2</td><td>50.4</td><td>53.9</td><td>59.2</td><td>58.1</td></tr><tr><td>ImageNet-R</td><td>64.6</td><td>83.9</td><td>73.9</td><td>89.3</td><td>88.0</td><td>88.7</td><td>90.1</td><td>88.5</td><td>88.2</td></tr><tr><td>ObjectNet</td><td>45.6</td><td>69.0</td><td>40.1</td><td>65.2</td><td>22.5</td><td>52.0</td><td>42.3</td><td>74.8</td><td>75.4</td></tr><tr><td>Country211</td><td>6.0</td><td>24.8</td><td>29.8</td><td>25.2</td><td>16.7</td><td>23.4</td><td>25.0</td><td>20.0</td><td>19.4</td></tr><tr><td>OK-VQA</td><td>9.4</td><td>33.2</td><td>56.8</td><td>51.5</td><td>67.3</td><td>62.4</td><td>71.7</td><td>61.4</td><td>67.3</td></tr><tr><td>A-OKVQA</td><td>6.6</td><td>21.0</td><td>47.3</td><td>43.6</td><td>63.8</td><td>51.1</td><td>58.7</td><td>54.7</td><td>59.3</td></tr><tr><td>DocVQA</td><td>11.3</td><td>41.4</td><td>89.7</td><td>90.1</td><td>79.2</td><td>92.2</td><td>93.8</td><td>92.4</td><td>94.3</td></tr><tr><td>InfographicsVQA</td><td>5.0</td><td>20.3</td><td>60.0</td><td>58.8</td><td>53.3</td><td>67.7</td><td>79.2</td><td>76.7</td><td>77.5</td></tr><tr><td>ChartQA</td><td>5.7</td><td>17.8</td><td>56.9</td><td>47.4</td><td>48.8</td><td>64.9</td><td>75.1</td><td>80.7</td><td>80.9</td></tr><tr><td>Visual7W</td><td>6.1</td><td>22.2</td><td>52.7</td><td>52.9</td><td>52.5</td><td>54.1</td><td>55.2</td><td>52.7</td><td>55.3</td></tr><tr><td>ScienceQA</td><td>16.3</td><td>28.0</td><td>38.5</td><td>38.2</td><td>65.4</td><td>42.7</td><td>53.7</td><td>57.3</td><td>61.6</td></tr><tr><td>VizWiz</td><td>27.6</td><td>39.0</td><td>39.9</td><td>43.3</td><td>43.8</td><td>46.8</td><td>51.6</td><td>54.5</td><td>56.2</td></tr><tr><td>GQA</td><td>8.3</td><td>76.9</td><td>55.1</td><td>64.9</td><td>65.7</td><td>67.3</td><td>69.3</td><td>64.9</td><td>68.5</td></tr><tr><td>TextVQA</td><td>18.8</td><td>46.8</td><td>71.6</td><td>72.2</td><td>76.8</td><td>78.6</td><td>83.5</td><td>83.8</td><td>84.3</td></tr><tr><td>VisDial</td><td>41.2</td><td>60.8</td><td>81.9</td><td>82.7</td><td>82.7</td><td>76.6</td><td>80.7</td><td>81.5</td><td>84.9</td></tr><tr><td>CIRR</td><td>8.2</td><td>54.9</td><td>51.1</td><td>57.5</td><td>60.4</td><td>53.7</td><td>55.3</td><td>47.6</td><td>61.2</td></tr><tr><td>VisualNews_t2i</td><td>50.1</td><td>79.7</td><td>80.5</td><td>74.5</td><td>69.5</td><td>71.7</td><td>76.8</td><td>71.9</td><td>73.7</td></tr><tr><td>VisualNews_i2t</td><td>47.6</td><td>83.6</td><td>81.2</td><td>78.2</td><td>79.4</td><td>74.2</td><td>82.0</td><td>73.6</td><td>73.9</td></tr><tr><td>MSCOCO_t2i MSCOCO_i2t</td><td>59.2 49.9</td><td>71.2 57.7</td><td>77.2 73.9</td><td>75.3 71.4</td><td>75.4 73.1</td><td>75.1 68.9</td><td>78.3 71.4</td><td>79.4 75.3</td><td>78.9 76.3</td></table>

Table 8: Detailed results of baselines and Embed-RL on Video and Visual Doc of MMEB-v2 benchmark.   

<table><tr><td></td><td colspan="8">1V-F1-R10</td></tr><tr><td>K700</td><td>23.4</td><td>39.7</td><td>35.5</td><td>38.0</td><td>40.1</td><td>35.8</td><td>42.8</td><td>55.8</td><td>56.8</td></tr><tr><td>SmthSmthV2</td><td>25.1</td><td>30.6</td><td>32.1</td><td>42.8</td><td>35.8</td><td>44.1</td><td>50.4</td><td>56.7</td><td>59.5</td></tr><tr><td>HMDB51</td><td>24.8</td><td>47.9</td><td>42.2</td><td>40.9</td><td>46.9</td><td>54.4</td><td>58.3</td><td>56.7</td><td>60.1</td></tr><tr><td>UCF101</td><td>49.4</td><td>54.7</td><td>61.8</td><td>60.0</td><td>39.6</td><td>67.2</td><td>70.0</td><td>79.3</td><td>78.5</td></tr><tr><td>Breakfast</td><td>10.9</td><td>14.3</td><td>23.8</td><td>14.8</td><td>16.6</td><td>20.1</td><td>21.5</td><td>36.7</td><td>33.0</td></tr><tr><td>MVBench</td><td>33.7</td><td>46.6</td><td>28.5</td><td>33.7</td><td>48.9</td><td>49.9</td><td>58.2</td><td>50.8</td><td>55.9</td></tr><tr><td>Video-MME</td><td>30.6</td><td>39.2</td><td>27.8</td><td>30.7</td><td>46.0</td><td>41.7</td><td>47.3</td><td>47.1</td><td>50.5</td></tr><tr><td>NExTQA</td><td>35.2</td><td>53.6</td><td>20.3</td><td>20.9</td><td>62.4</td><td>59.9</td><td>69.6</td><td>53.9</td><td>58.2</td></tr><tr><td>EgoSchema</td><td>38.4</td><td>46.8</td><td>21.8</td><td>34.0</td><td>60.0</td><td>45.4</td><td>52.4</td><td>53.0</td><td>52.8</td></tr><tr><td>ActivityNetQA</td><td>51.3</td><td>65.6</td><td>51.4</td><td>52.3</td><td>76.0</td><td>57.8</td><td>76.0</td><td>74.8</td><td>74.4</td></tr><tr><td>DiDeMo</td><td>22.8</td><td>26.4</td><td>29.3</td><td>30.4</td><td>37.8</td><td>32.4</td><td>40.0</td><td>45.3</td><td>46.8</td></tr><tr><td>MSR-VTT</td><td>17.6</td><td>31.8</td><td>34.5</td><td>28.3</td><td>36.5</td><td>34.3</td><td>38.9</td><td>45.7</td><td>46.2</td></tr><tr><td>MSVD</td><td>45.4</td><td>49.7</td><td>46.7</td><td>48.1</td><td>56.4</td><td>55.4</td><td>60.8</td><td>67.2</td><td>65.8</td></tr><tr><td>VATEX</td><td>16.7</td><td>24.9</td><td>25.5</td><td>26.5</td><td>32.0</td><td>29.9</td><td>32.6</td><td>43.6</td><td>43.4</td></tr><tr><td>YouCook2</td><td>5.3</td><td>9.1</td><td>9.0</td><td>10.6</td><td>9.5</td><td>12.7</td><td>18.5</td><td>23.5</td><td>23.3</td></tr><tr><td>QVHighlight</td><td>19.9</td><td>59.5</td><td>57.7</td><td>49.4</td><td>58.4</td><td>57.5</td><td>54.9</td><td>70.7</td><td>73.6</td></tr><tr><td>Charades-STA</td><td>29.0</td><td>14.0 37.4</td><td>19.8 39.3</td><td>20.2 40.8</td><td>18.7 41.4</td><td>20.4 41.2</td><td>21.9 41.1</td><td>26.4 50.9</td><td>25.0 49.9</td></tr><tr><td>MomentSeeker</td><td>27.6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ViDoRe_arxivqa</td><td>81.7</td><td>86.9</td><td>60.2</td><td>80.6</td><td>73.3</td><td>73.9</td><td>73.6</td><td>86.1</td><td>88.7</td></tr><tr><td>ViDoRe_docvqa ViDoRe_infovqa</td><td>56.6</td><td>57.5</td><td>34.7</td><td>44.9</td><td>38.3</td><td>37.9</td><td>41.1</td><td>45.7</td><td>47.5</td></tr><tr><td></td><td>84.9</td><td>91.6</td><td>70.4</td><td>83.7</td><td>80.6</td><td>76.2</td><td>80.8</td><td>86.8</td><td>86.9</td></tr><tr><td>ViDoRe_tabfquad</td><td>86.9</td><td>94.6 74.1</td><td>78.2</td><td>89.2</td><td>80.7</td><td>86.1</td><td>90.2</td><td>94.5</td><td>94.7</td></tr><tr><td>ViDoRe_tatdqa</td><td>70.9</td><td>96.8</td><td>27.6</td><td>43.8</td><td>37.8</td><td>40.6</td><td>46.7</td><td>54.6</td><td>54.8</td></tr><tr><td>ViDoRe_shiftproject</td><td>75.1</td><td>99.6</td><td>38.6</td><td>60.8 88.5</td><td>52.0</td><td>66.8</td><td>65.0</td><td>70.7</td><td>69.0</td></tr><tr><td>ViDoRe−artificial_intelligence</td><td>95.7</td><td>95.3</td><td>67.7</td><td>86.5</td><td>86.0 84.8</td><td>85.9 83.3</td><td>89.5</td><td>94.0 86.7</td><td>91.6</td></tr><tr><td>ViDoRe_energy</td><td>94.7</td><td>98.8</td><td>60.4</td><td></td><td>85.0</td><td></td><td>85.7</td><td></td><td>88.1</td></tr><tr><td>ViDoRe_government_reports</td><td>93.6</td><td>99.3</td><td>61.8 69.9</td><td>85.0 92.2</td><td>88.4</td><td>82.6 90.8</td><td>89.8 94.3</td><td>89.0 91.1</td><td>90.7 90.4</td></tr><tr><td>ViDoRe_healthcare_industry</td><td>95.9</td><td>63.4</td><td>6.8</td><td>45.6</td><td>50.7</td><td>50.2</td><td>50.4</td><td>56.9</td><td>59.8</td></tr><tr><td>ViDoRe_esg_reports_human_labeled_v2</td><td>51.3</td><td>49.5</td><td>5.1</td><td>44.3</td><td>50.9</td><td>46.2</td><td>50.7</td><td></td><td>50.1</td></tr><tr><td>ViDoRe_biomedical_lectures_v2_multilingual</td><td>54.7</td><td>54.2</td><td></td><td></td><td></td><td></td><td></td><td>51.0</td><td></td></tr><tr><td>ViDoRe_economics_reports_v2_multilingual</td><td>49.0</td><td></td><td>13.9</td><td>43.0</td><td>54.3</td><td>45.7</td><td>57.8</td><td>53.0</td><td>53.9</td></tr><tr><td>ViDoRe_esg_reports_v2_multilingual</td><td>52.9</td><td>55.4</td><td>11.9</td><td>46.6</td><td>42.3</td><td>42.7</td><td>43.2</td><td>46.9</td><td>49.7</td></tr><tr><td>VisRAG_ArxivQA</td><td>80.9</td><td>87.4</td><td>52.6</td><td>76.9</td><td>74.0</td><td>74.3</td><td>80.5</td><td>84.9</td><td>86.9</td></tr><tr><td>VisRAG_ChartQA</td><td>72.3</td><td>86.1</td><td>57.7</td><td>83.7</td><td>82.7</td><td>86.0</td><td>85.0</td><td>88.3</td><td>88.5</td></tr><tr><td>VisRAG_MP-DocVQA</td><td>82.0</td><td>89.7</td><td>60.6</td><td>88.1</td><td>75.1</td><td>75.6</td><td>83.4</td><td>79.1</td><td>79.3</td></tr><tr><td>VisRAG_SlideVQA</td><td>85.1</td><td>92.6</td><td>54.7</td><td>84.1</td><td>87.6</td><td>87.1</td><td>91.5</td><td>92.3</td><td>92.6</td></tr><tr><td>VisRAG_InfoVQA</td><td>83.5</td><td>88.6</td><td>66.0</td><td>82.3</td><td>87.9</td><td>84.4</td><td>89.2</td><td>90.0</td><td>89.6</td></tr><tr><td>VisRAG_PlotQA</td><td>79.3</td><td>76.5 32.6</td><td>62.7</td><td>75.9</td><td>69.4</td><td>68.0</td><td>72.7</td><td>73.0</td><td>72.4</td></tr><tr><td>ViDoSeek-page</td><td>38.1</td><td></td><td>16.3</td><td>29.1</td><td>22.5</td><td>21.2</td><td>21.3</td><td>82.0</td><td>84.4</td></tr><tr><td>ViDoSeek-doc</td><td>87.5</td><td>90.3</td><td>69.4</td><td>79.0</td><td>73.8</td><td>75.9</td><td>75.3</td><td>82.6</td><td>82.4</td></tr><tr><td>MMLongBench-page MMLongBench-doc</td><td>27.1 80.4</td><td>36.9 85.2</td><td>0.4 28.8</td><td>15.8 63.0</td><td>13.3 42.6</td><td>11.9 39.7</td><td>12.3 41.3</td><td>47.7 50.3</td><td>51.0 50.7</td></table>

# E Detailed Scores of MMEB-V1

We also report our performance on MMEB-V1 [26], including both in-domain and out-of-domain performance, as shown in Table 9.

Table 9: Results on the MMEB-V1 benchmark (consisting of 36 image embedding tasks). IND and OOD denote the in-distribution and out-of-distribution datasets, respectively. The highest and second-highest values are highlighted in bold and underline.   

<table><tr><td rowspan="2">Model</td><td colspan="4">Per Meta-Task Score</td><td colspan="3">Average Score</td></tr><tr><td>Classification VQA Retrieval Grounding IND </td><td></td><td></td><td></td><td></td><td>OOD Overall</td><td></td></tr><tr><td># of Datasets</td><td colspan="7">10 10 12</td></tr><tr><td colspan="7">Baseline I Models</td><td></td><td></td></tr><tr><td>CLIP [43]</td><td>42.8</td><td>9.1</td><td>53.0</td><td>51.8</td><td>37.1</td><td>38.7</td><td>37.8</td></tr><tr><td>BLIP2 [34]</td><td>27.0</td><td>4.2</td><td>33.9</td><td>47.0</td><td>25.3</td><td>25.1</td><td>25.2</td></tr><tr><td>SigLIP [67]</td><td>40.3</td><td>8.4</td><td>31.6</td><td>59.5</td><td>32.3</td><td>38.0</td><td>34.8</td></tr><tr><td>OpenCLIP [10]</td><td>47.8</td><td>10.9</td><td>52.3</td><td>53.3</td><td>39.3</td><td>40.2</td><td>39.7</td></tr><tr><td>UniIR (BLIPFF) [57]</td><td>42.1</td><td>15.0</td><td>60.1</td><td>62.2</td><td>44.7</td><td>40.4</td><td>42.8</td></tr><tr><td>UniIR (CLIPSF) [57]</td><td>44.3</td><td>16.2</td><td>61.8</td><td>65.3</td><td>47.1</td><td>41.7</td><td>44.7</td></tr><tr><td>Magiclens [69]</td><td>38.8</td><td>8.3</td><td>35.4</td><td>26.0</td><td>31.0</td><td>23.7</td><td>27.8</td></tr><tr><td colspan="8">MLLM-based Baseline Models</td></tr><tr><td>E5-V [25]</td><td>21.8</td><td>4.9</td><td>11.5</td><td>19.0</td><td>14.9</td><td>11.5</td><td>13.3</td></tr><tr><td>VLM2Vec-2B [26]</td><td>59.0</td><td>49.4</td><td>65.4</td><td>73.4</td><td>66.0</td><td>52.6</td><td>60.1</td></tr><tr><td>VLM2Vec-7B [26]</td><td>62.6</td><td>57.8</td><td>69.9</td><td>81.7</td><td>72.2</td><td>57.8</td><td>65.8</td></tr><tr><td>VLM2Vec-V2 [41]</td><td>62.9</td><td>56.3</td><td>69.5</td><td>77.3</td><td>68.8</td><td>59.9</td><td>64.9</td></tr><tr><td>MMRet-7B [77]</td><td>56.0</td><td>57.4</td><td>69.9</td><td>83.6</td><td>68.0</td><td>59.1</td><td>64.1</td></tr><tr><td>CAFe-V1-7B [63]</td><td>65.2</td><td>65.6</td><td>70.0</td><td>91.2</td><td>75.8</td><td>62.4</td><td>69.8</td></tr><tr><td>CAFe-V2-7B [63]</td><td>63.6</td><td>61.7</td><td>69.1</td><td>87.6</td><td>72.8</td><td>61.1</td><td>67.6</td></tr><tr><td>mmE5-11B [9]</td><td>67.6</td><td>62.8</td><td>70.9</td><td>89.7</td><td>72.3</td><td>66.7</td><td>69.8</td></tr><tr><td>LLaVE-2B [29]</td><td>62.1</td><td>60.2</td><td>65.2</td><td>84.9</td><td>69.4</td><td>59.8</td><td>65.2</td></tr><tr><td>LLaVE-7B [29]</td><td>65.7</td><td>65.4</td><td>70.9</td><td>91.9</td><td>75.0</td><td>64.4</td><td>70.3</td></tr><tr><td>UniME-4B [18]</td><td>54.8</td><td>55.9</td><td>64.5</td><td>81.8</td><td>68.2</td><td>52.7</td><td>64.2</td></tr><tr><td>UniME-7B [18]</td><td>66.8</td><td>66.6</td><td>70.6</td><td>90.9</td><td>74.6</td><td>65.8</td><td>70.7</td></tr><tr><td>UME-R1-2B [30]</td><td>64.8</td><td>62.8</td><td>67.6</td><td>77.2</td><td>71.5</td><td>60.4</td><td>66.6</td></tr><tr><td>UME-R1-7B b [30]</td><td>67.1</td><td>69.2</td><td>71.9</td><td>84.9</td><td>76.1</td><td>65.1</td><td>71.3</td></tr><tr><td colspan="8">Ours</td></tr><tr><td>Embed-RL-2B</td><td>62.8</td><td>67.9</td><td>68.6</td><td>90.4</td><td>71.9 65.9</td><td></td><td>69.2</td></tr><tr><td>Embed-RL-4B</td><td>63.7</td><td>70.5</td><td>71.3</td><td>91.4</td><td>74.3 67.3</td><td></td><td>71.2</td></tr></table>

# F Prompt for synthesizing multimodal chain of thought

To enable precise guidance for visual reasoning and retrieval tasks, we design a hierarchically structured prompting system that instructs models to execute visual analysis tasks across text, image, and video modalities. This system consists of two scenario-specialized core modules, which impose constraints on reasoning logic, output formatting, and evidence anchoring, and adheres to a framework of two-round reasoning with fixed-format output to guarantee the consistency and accuracy of the generated results.

# F.1 Basic Visual Reasoning Prompts

Such prompts guide the model to complete basic reasoning based on inputs, supporting various visual tasks, and are divided into 4 items according to inputs and objectives:

Text-to-Image Retrieval Reasoning Prompts: Focus on text-to-image retrieval, extract key visual concepts, anchor textual evidence to output a JSON keyword list, adapt to subtasks, and output results in a fixed format. Image Reasoning Prompts: For image-based tasks, anchor visual evidence to output the 2D bounding box coordinates of key elements, locate core features, and complete reasoning and answer output in accordance with the process.   
Video Reasoning Prompts: Adapt to video sequence tasks, output 1-based key frame indices based on frame visual evidence, identify core frames, and generate results following a fixed process.   
Text-to-Video Retrieval Reasoning Prompts: For text-to-video retrieval, extract visual concepts containing temporal dynamics, output a JSON keyword list, and standardize result output according to subtasks.

# </> Text-to-Image Retrieval Visual Reasoning Prompt

You are avisual easoning asstant specialized in text—to—mage etrieval.Given  text desciption n a alyz  nn een e ia images.

# \*\*Rules:\*\*

Keep reasoning concise and grounded in textual evidence. Limit each step to 1-2 sentences. Base your reasonig solely  he textual content and the task desiptin

Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if needed.   
In your thinkng process, you must extract and output key visual concepts from the text descption.

Use JSON format with key 'text keywords' to specify the important keywords as a list.

5.First, think between ' <thinking>' and $\dot { \iota } _ { < / \mathrm { t h i n k i n g } > } \cdot$ while output necessary keywords from the text in JON with key 'text keywords' Then, based on the thinking contents, rethink between '<rethink and $\cdot _ { < } /$ rethink Finally, output the answer within <answer</answer

6.Your thought process should adapt to the task type:

$^ *$ For \*\*caption—based retrieval\*\* (e.g., find image from caption): Extract key visual elements, objects, scenes, and relationships.   
\* F\*ne etrival\*  d news mage from headline: Identy key people, ocatons,nts, and contextual elements. Fo\*\*dialogue—based retrieval\* . find image fromconversation): Summarize visual attriuts, actions, and scene details.   
\* For \*\*question—based retrieval\*\* (e.g., find factual image): Identify the key concepts and relationships.

Now, process the following input: TASK/QUESTION: {question}

Y al e i gn  adi un aly step—by—step to produce the required output. The task may involve image retrieval, classification, question answering, or object identification.

# \*\*Rules:\*\*

1Keep reasoning concise and grounded in visual evidence. Limit each step to 1—2 sentences.

Rephrase the final answer to preserve its exact meaning, changing only wording/phrasing if ne   
In your thinking process, you must output coordinates for the key visual element(s) relevant to anwering the queston. Use JSON format with key 'bbox d to speciy the bounding box as [x1, y1, $_ { \times 2 }$ ,y2]. For multiple elements, use a list of bboxes: [x1, y1, $\times 2$ y2], [x1, y1, $\times 2$ y2]].

5. First, think between '<thinking>' and $^ { \circ } < \iota$ 'thinking>' while output necessary coordinates needed to answer the question in JSON with key 'bbox _2d'. Then, based on the thinking contents and coordinates, rethink between '<rethink>' and $^ { \cdot } < /$ rethink>'. Finally, output the answer within $^ { \prime } <$ answer>...</answer>'.

6.Your thought process should adapt to the task:

For \*\*retrieval\*\* (e.g., find similar image): Identify and locate key visual elements that define the match.   
$^ *$ F\*classificatio\*  sceejecdomai: Locat distinguishinvisal features that be to the class.   
$^ *$ For \*\*question answering\*\*: Locate the visual or textual clues in the image that lead to the answer.   
$^ *$ For \*\*object identification/segmentation\*\*: Provide the object's location and boundaries.

Now, process the following input IMAGE: {image} TASK/QUESTION: {question}

# < Video Sequence Visual Reasoning Prompt

ly   u ut yo captioning, video question answering, or video retrieval.

# \*\*Rules:\*\*

Keep reasoning concise and grounded in visual evidence from the video frames. Limit each step to 1—2 sentences.

.Base your reasoning solely on the visual content of the video frames and the task description.

In your thinkng pocess you must entiyand output rames fom the de seuenc tha most relevant to answering the question. Use JSON format with key 'key_frames' to specify the frame indices as a list (using 1—based indexing).

5. First, think between ' <thinking>' and $\cdot < / \mathrm { t h i n k i n g } > ^ { \prime }$ while output necessary key frame indices in JSON with key 'yframes' Then, based on the thinking contents, rethink between'rethinkand $^ { \ast } < /$ rethink> Finally, output the answer within ' <answer>..</answer>'.

Your thought process should adapt to the task type:

\* For \*\*video captioning/description\*\*: Identify frames that show main events, transitions, or key moments in the video sequence.   
\* For \*\*video question answering\*\*: Locate frames that contain the visual evidence needed to answer the specific question.   
\* For \*\*video retrieval\*\*: Identify frames that represent the core content or distinguishing features o the video.

Now, process the following input: VIDEO FRAMES: {video} TASK/QUESTION: {question}

# </> Text-to-Video Retrieval Visual Reasoning Prompt

You ar  visual reasoning assistant specialized in tex—to—videretrival. Given avideo desciption na alyz t connt  dte ial cnt eed evi a video clips or keyframes.

# \*\*Rules:\*\*

Keep reasoning concise and grounded in textual evidence. Limit each step to $^ { 1 - 2 }$ sentences. Rephras th final answer preserve  exact meaningchanging only wording/phrasing  needed.

In your thinking process, you must extract and output key visual concepts from the text description. Use JSON format with key 'text _keywords' to specify the important keywords as a list.   
5 First, think between ' <thinking $> ^ { \prime }$ and $^ { \circ } < \iota$ thinking>' while output necessary keywords from the text in JON with key 'text  keywords' Then, based on the thinking contents, rethink between '<rethink> and $\scriptstyle \cdot < / r e t h i n k > ^ { \prime }$ .Finally, output the answer within ' <answer>. $< /$ answer>'.   
Your thought process should adapt to the task type: $^ *$ For \*\*video—based retrieval\*\* (e.g., find video from description): Extract key visual elements, objects, scenes, actions, temporal sequences, and relationships. \* For \*\*scene—based retrieval\*\* (e.g., find video from scene description): Identify key people, locations, events, contextual elements, and temporal progression. \* For \*\*action—based retrieval\*\* (e.g., find video from action sequence): Summarize visual attributes, actions, scene details, and temporal dynamics.

Now, process the following input TASK/QUESTION: {question}

# F.2 Positive Sample Verification Reasoning Prompts

These prompts analyze the rationale behind positive samples to support model training and verification, and are categorized into three types according to the positive sample modality:

Text Positive Sample Analysis Prompts: Anchor textual evidence to output a JSON-formatted keyword list, generate a text summary through two rounds of reasoning, and clarify the rationale for labeling the sample as positive.   
Image Positive Sample Analysis Prompts: Infer image content in conjunction with the target task, output bounding box coordinates of key elements, generate an image summary through two rounds of reasoning, and clarify the core features and grounding basis.   
Video Positive Sample Analysis Prompts: Anchor video evidence to output 1-indexed key frame indices, generate a structured video analysis through two rounds of reasoning, and clarify the core rationale for positive sample validity.

# </> Positive Text Output Visual Reasoning Analysis Prompt

to determine the key concepts that make it the correct output.

\*\*Rules:\*\*   
1. Keep reasoning concise and grounded in the textual evidence. Limit each step to $^ { 1 - 2 }$ sentences.   
Use the task description to understand the context of the output.   
In your thinking proess, you must extract and output key concts rom the text. Use JSON forat with key 'text_keywords' to specify the important keywords as a list.   
First, think between<thinkingand </thinkingwhile output necessary keywords from the text in JSON with key 'textkeywords' Then, based on the thinking contents, rethink between <rethink and '</rethink Finally, output a brief description of the text within '<answer></answer   
Now, process the following input:   
POSITIVE TEXT OUTPUT: {pos_text}

# </> Positive Image Output Visual Reasoning Analysis Prompt

Y al . s de  uu l x templat analyze the ask  determi what the mag hould contand locat key rons.

\*\*Rules:\*\*

1Keep reasoning concise and grounded in the task description. Limit each step to $^ { 1 - 2 }$ sentences.   
Use the task description to infer what the target image should look like.

In your thinking process, you must output coordinates for the key visual element(s) relevant to answering the question. If no question is specified, output the most important element. Use JSON format with key 'bbox 2d' to specify the bounding box as [x1, y1, $\times 2$ y2] For multiple elements, use a lst of oes: [[1, , $\times 2$ y2], [x1, y1, $\times 2 ,$ y22}].

4. First, think between '<thinking>' and $^ i < \bar { / }$ thinking>' while output necessary coordinates in JSON with key bboxd Then based on the thinking contents, rethink between <rethinkand $\cdot _ { < } /$ rethink $> ^ { i }$ . Finally, output a brief description of the image within ' <answer>..</answer>'.

Now, process the following input: POSITIVE TEXT OUTPUT: {pos_text} POSITIVE IMAGE OUTPUT:{pos_image_description}

# </> Positive Video Output Visual Reasoning Analysis Prompt

Y the output to determine the key moments or concepts that make it correct for the task.

# \*\*Rules:\*\*

Keep reasoning concise and grounded in the provided video output evidence. Limit each step to 1—2 sentences.   
Use the task description to understand the context of the output.   
In yohi p y mus en ouut hd  he os elevn  re frames from the described video sequence. Use JSON format with key 'key_frames' to specify the frame indices as a list (using 1—based indexing).   
4. First, think between '<thinking>'and $^ { \circ } < _ { \prime }$ thinking>' while outputting necessary key frame indices in JSON with key 'key_frames'. Then, based on the thinking contents, rethink between '<rethink>' and '</rethink>' Finally, output a brief analysis of the video output within '<answer>..</answer $> ^ { \prime }$ .

Now, process the following input: POSITIVE TEXT OUTPUT: {pos_text} POSITIVE VIDEO OUTPUT:{pos_video_output}

# G Comprehensive Performance Characterization on Video Retrieval

To fully validate the cross-dimensional generalization capability of our proposed model in complex and diverse retrieval scenarios, we conduct a systematic performance evaluation on the Universal Video Retrieval Benchmark (UVRB) [20]. This benchmark consists of 16 datasets targeting distinct core capabilities, comprehensively covering multiple retrieval paradigms (textual, composed, and visual retrieval) and diverse semantic scenarios (coarse-grained, fine-grained, and longcontext retrieval). It thus enables accurate quantification of the model's universal adaptation capacity across heterogeneous retrieval tasks. Experimental results demonstrate that our model outperforms counterparts with equivalent parameter scales, maintains a consistent performance advantage, and achieves the optimal comprehensive capability among models of the same parameter scale. Specifically, in core retrieval dimensions: (1) it attains state-of-the-art performance in coarse-grained semantic retrieval tasks and the second-best result in fine-grained semantic understanding scenarios; (2) it secures the optimal and second-best performances in spatial fine-grained perception (object/appearance recognition) and temporal fine-grained perception (motion/dynamics capture) subtasks, respectively. These results fully highlight the model's robust capability in multi-dimensional semantic understanding and spatiotemporal feature extraction.

# G.1 Video Retrieval Performance on UVRB Datasets

Baselines. Following the experimental settings of GVE [20], we evaluate 16 representative baselines spanning diverse architectures, parameter scales, and training data compositions. These baselines are categorized into two groups: (1) traditional CLIP-based embedding models, including CLIP4Clip [40], ViCLIP [54], VideoCLIP-XL [48], LanguageBind [78], and the InternVideo2 (1B/6B) [55]; (2) recent MLLM-based embedding models, including GVE-2B/7B [20], GME-2B/7B [72], Unite-2B/7B [27], VLM2Vec-V2 [41], BGE-VL [76], UniME-7B [18], and B3-7B [46].

Datasets. We adopt the Universal Video Retrieval Benchmark (UVRB) [20], which assesses model universality via 16 test datasets targeting distinct core abilities. UVRB covers diverse retrieval scenarios, with datasets categorized as: coarse-grained retrieval (MSRVTT [61], DiDeMo [1], CRB-G [62]); fine-grained retrieval (spatial: CRB-S [62]/VDC-O [7], temporal: CRB-T [62]/CMRB [37], partially relevant: DREAM-E [49]/LoVR-Theme2Clip [5]/PEV-K [4]); longcontext retrieval (LoVR-V [5], VDC-D [7]); and composed query retrieval (MS-TI/MS-TV adapted from MomentSeeker [66], MSRVTT-I2V [61], LoVR-C2V [5]).

Metrics. Following GVE [20], we adopt Recall@1 (R@1) as the primary metric, which measures the accuracy of identifying the most relevant item. For challenging datasets with ambiguous queries (e.g., CMRB, LoVR-TH), we additionally report Recall@10 (R@10) to reflect performance on top-k retrieval. For

Table 10: Performance of video retrieval on UVRB datasets: AVG values represent the average across 16 datasets, with the highest score in each column bolded and the second-highest underlined. Metrics include R@1 (Recall@1), R@10 (Recall@10) and P@1 (Precision $@ 1$ ). The highest and second-highest values are highlighted in bold and underline.   

<table><tr><td>Model</td><td>AVG</td><td>MSRVTT DiDeMo R@1</td><td>R@1</td><td>CRB-G R@1</td><td>CRB-S R@1</td><td>VDC-O R@1</td><td>CRB-T R@1</td><td>R@10</td><td>CMRB DREAM-E R@1</td></tr><tr><td>CLIP4Clip [40]</td><td>39.0</td><td>33.3</td><td>29.7</td><td>51.1</td><td>49.7</td><td>62.0</td><td>28.9</td><td>28.0</td><td>19.1</td></tr><tr><td>ViCLIP [54]</td><td>35.2</td><td>38.6</td><td>30.6</td><td>44.7</td><td>43.7</td><td>53.0</td><td>34.9</td><td>22.9</td><td>23.5</td></tr><tr><td>VideoCLIP-XL [48]</td><td>49.1</td><td>44.3</td><td>40.3</td><td>82.8</td><td>83.9</td><td>73.5</td><td>48.7</td><td>27.4</td><td>26.3</td></tr><tr><td>LanguageBind [78]</td><td>48.7</td><td>47.9</td><td>42.1</td><td>71.6</td><td>68.7</td><td>75.9</td><td>46.6</td><td>29.0</td><td>28.0</td></tr><tr><td>InternVideo2-1B [55]</td><td>40.4</td><td>44.9</td><td>40.4</td><td>58.6</td><td>56.8</td><td>64.4</td><td>47.0</td><td>35.5</td><td>24.2</td></tr><tr><td>InternVideo2-6B [55]</td><td>42.7</td><td>48.5</td><td>41.8</td><td>60.8</td><td>61.2</td><td>65.0</td><td>45.5</td><td>34.6</td><td>27.1</td></tr><tr><td>GME-2B [72]</td><td>48.8</td><td>39.0</td><td>30.3</td><td>69.0</td><td>71.8</td><td>71.5</td><td>40.0</td><td>29.8</td><td>24.0</td></tr><tr><td>Unite-2B [27]</td><td>48.0</td><td>36.7</td><td>29.8</td><td>69.9</td><td>72.3</td><td>72.7</td><td>40.9</td><td>28.4</td><td>22.3</td></tr><tr><td>VLM2Vec-V2 [41]</td><td>50.8</td><td>33.0</td><td>29.9</td><td>82.8</td><td>84.3</td><td>77.5</td><td>41.0</td><td>28.6</td><td>22.8</td></tr><tr><td>BGE-VL [76]</td><td>44.3</td><td>33.7</td><td>31.8</td><td>69.0</td><td>68.8</td><td>63.9</td><td>35.9</td><td>22.5</td><td>21.2</td></tr><tr><td>UniME-7B [18]</td><td>52.1</td><td>35.1</td><td>33.5</td><td>81.5</td><td>82.7</td><td>74.3</td><td>47.6</td><td>31.7</td><td>29.3</td></tr><tr><td>B3-7B [46]</td><td>51.1</td><td>28.2</td><td>35.0</td><td>81.5</td><td>82.5</td><td>76.8</td><td>41.5</td><td>31.2</td><td>21.6</td></tr><tr><td>GME-7B [72]</td><td>53.0</td><td>43.6</td><td>37.7</td><td>74.0</td><td>76.7</td><td>73.1</td><td>44.2</td><td>30.4</td><td>27.4</td></tr><tr><td>Unite-7B [27]</td><td>53.8</td><td>43.9</td><td>38.6</td><td>79.8</td><td>80.4</td><td>75.3</td><td>47.2</td><td>35.1</td><td>27.9</td></tr><tr><td>GVE-3B [20]</td><td>54.4</td><td>43.1</td><td>37.6</td><td>85.0</td><td>84.6</td><td>78.6</td><td>49.6</td><td>36.3</td><td>28.0</td></tr><tr><td>GVE-7B [20]</td><td>57.3</td><td>46.4</td><td>43.3</td><td>86.5</td><td>84.7</td><td>79.4</td><td>53.9</td><td>39.8</td><td>30.2</td></tr><tr><td>Embed-RL-2B</td><td></td><td>43.7</td><td>42.2</td><td>91.4</td><td>89.7</td><td>84.6</td><td>49.4</td><td>36.5</td><td></td></tr><tr><td>Embed-RL-4B</td><td>58.7 60.2</td><td>44.0</td><td>46.1</td><td>92.1</td><td>89.9</td><td>85.9</td><td>53.8</td><td>38.2</td><td>31.8 32.0</td></tr><tr><td colspan="10">Model</td></tr><tr><td></td><td colspan="3">R@10 R@1</td><td>R@1</td><td>R@1 P@1</td><td>P@1</td><td colspan="3">LoVR-TH PEV-K LoVR-V VDC-D MS-TI MS-TV MSRVTT-I2V LoVR-C2V R@1 R@1</td></tr><tr><td>CLIP4Clip [40]</td><td colspan="3">33.8 17.9</td><td>36.0</td><td>56.6 17.3</td><td>18.3</td><td colspan="3">92.4</td></tr><tr><td>ViCLIP [54]</td><td colspan="3">20.2 7.5</td><td>23.0</td><td>39.5 28.3</td><td>24.3</td><td colspan="2">84.6</td><td>50.3 43.3</td></tr><tr><td>VideoCLIP-XL [48]</td><td colspan="3">43.9 22.9</td><td>38.0 82.0</td><td>23.0</td><td>22.3</td><td colspan="2">86.1</td><td>40.3</td></tr><tr><td>LanguageBind [78]</td><td colspan="3">42.5 30.3</td><td>54.0 67.9</td><td>22.8</td><td>23.3</td><td colspan="2">82.7</td><td>46.3</td></tr><tr><td>InternVideo2-1B [55]</td><td colspan="3">29.8 2.6</td><td>28.0 48.5</td><td>26.5</td><td>23.0</td><td colspan="2">79.4</td><td>36.8 45.2</td></tr><tr><td>InternVideo2-6B [55]</td><td colspan="3">30.2 8.6</td><td>33.0 51.6</td><td>23.5</td><td>20.5</td><td colspan="3">86.8</td></tr><tr><td>GME-2B [72]</td><td colspan="3">44.6 35.4</td><td>53.0</td><td>83.9 35.0</td><td>34.0</td><td colspan="3">82.7</td></tr><tr><td>Unite-2B [27</td><td colspan="3">44.5 35.5</td><td>57.0</td><td>79.2 25.0</td><td>23.3</td><td colspan="3">86.3</td></tr><tr><td>VLM2Vec-V2 [41]</td><td colspan="3">49.2 32.4</td><td>61.0</td><td>91.3 27.5</td><td>25.0</td><td colspan="3">84.1</td></tr><tr><td>BGE-VL [76]</td><td colspan="3">38.7 18.4</td><td>55.0</td><td>72.2 30.3</td><td>23.3</td><td colspan="3">77.9 86.7</td></tr><tr><td>UniME-7B [18]</td><td colspan="3">50.4 32.3</td><td>48.0</td><td>84.7 31.0</td><td colspan="3">30.5 88.4</td><td>46.5 53.7</td></tr><tr><td>B3-7B [46]</td><td colspan="3">46.2 38.7</td><td>59.0 85.3</td><td>27.5</td><td colspan="3">26.5</td><td>47.1</td></tr><tr><td>GME-7B [72]</td><td colspan="3">52.3 39.6</td><td>71.0 86.5</td><td>34.8</td><td colspan="3">33.3</td><td>37.0</td></tr><tr><td></td><td colspan="3">55.5 44.0</td><td>62.0 87.1</td><td>27.8</td><td colspan="3">23.0</td><td>44.8</td></tr><tr><td>Unite-7B [</td><td colspan="3">52.2 33.0</td><td>61.0 91.8</td><td></td><td></td><td colspan="3">88.3 89.1</td></tr><tr><td>GVE-3B [20]</td><td colspan="3">54.2</td><td>68.0</td><td>34.0 34.3</td><td>26.8</td><td colspan="3">40.3 89.9</td></tr><tr><td>GVE-7B [20]</td><td colspan="3">41.3</td><td></td><td>94.8</td><td>28.0</td><td colspan="3">41.5 89.1 51.4</td></tr><tr><td>Embed-RL-2B Embed-RL-4B</td><td colspan="2">56.5 57.9</td><td>33.6 31.9</td><td>80.0 77.0</td><td>93.8 19.3 95.2 15.8</td><td>21.0 21.0</td></table>

MS-TI and MS-TV (containing multiple positive candidates), we use Precision@1 (P@1) as the key metric.

Performance. Table 10 presents the video retrieval performance of diverse models on the Universal Video Retrieval Benchmark (UVRB) [20] datasets. Our Embed-RL-2B and Embed-RL-4B models outperform all baselines in the average score (AVG), with Embed-RL-4B achieving the highest AVG (0.602). Both models secure the first or second place on key datasets across different retrieval paradigms, including coarse-grained retrieval (DiDeMo [1], CRB-G [62]] and fine-grained/long-context retrieval (CRB-S [62], VDC-O/VDC-D [7]), which verifies their superior multi-dimensional video retrieval capability.

# G.2 Capability Characterization of the UVRB Evaluation Metrics

The Universal Video Retrieval Benchmark (UVRB) adopts unweighted arithmetic means for all metric calculations to ensure fair comparison across heterogeneous datasets, and comprehensively evaluates video retrieval models from three orthogonal dimensions (Tasks, Domains, Sub-domains).

UVRB covers 16 datasets that are exhaustively partitioned into non-overlapping categories according to Tasks, Domains and Sub-domains, as detailed in Table 11. This partition is the foundation for quantifying distinct model capabilities in different retrieval scenarios.

Table 11: Detailed Partition of Datasets in the Universal Video Retrieval Benchmark (UVRB) Across Tasks, Domains, and Sub-domains   

<table><tr><td>Partition</td><td>Content</td></tr><tr><td>DTXT</td><td>{MSRVTT, DiDeMo, CRB-G, CRB-S, VDC-O, CRB-T, CMRB, DREAM-E, LoVR-TH, PEV-K, LoVR-V, VDC-D}</td></tr><tr><td>Dcmp</td><td>{MS-TI, MS-TV}</td></tr><tr><td>DvIs</td><td>{MSRVTT-I2V, LoVR-C2V}</td></tr><tr><td>DcG</td><td>{MSRVTT, DiDeMo, CRB-G}</td></tr><tr><td>DFG</td><td>{CRB-S, VDC-O, CRB-T, CMRB, DREAM-E, LoVR-TH, PEV-K}</td></tr><tr><td>DLC</td><td>{LoVR-V, VDC-D}</td></tr><tr><td>Ds</td><td>{CRB-S, VDC-O}</td></tr><tr><td>DT</td><td>{CRB-T, CMRB}</td></tr><tr><td>DPR</td><td>{DREAM-E, LoVR-TH, PEV-K}</td></tr></table>

Based on the above dataset partition, Table 12 defines the calculation rules and corresponding capability characterization for each evaluation dimension. All metrics are computed as unweighted arithmetic means of corresponding datasets (denoted as $\mathcal { D }$ ), with results rounded to three decimal places. The three core dimensions are defined as follows:

1. Task dimension: Distinguishes retrieval paradigms by query formats (textual, composed, visual), reflecting cross-modal alignment ability for different query types;   
2. Domain dimension: Assesses model performance across different levels of semantic granularity (coarse-grained, fine-grained, long-context), measuring generalization on retrieval tasks involving short/long context and high/lowlevel semantics;   
3. Sub-domain dimension: Further decomposes fine-grained retrieval into three sub-tasks (spatial, temporal, partially relevant), pinpointing model strengths and weaknesses in fine-grained understanding.

The overall AVG score is the arithmetic mean of the three task columns (TXT, CMP, VIS) and three domain columns (CG, FG, LC). This score aggregates model performance across core retrieval paradigms, rather than taking raw averages over datasets.

Table 12: Calculation Rules and Capability Characterization for Model Evaluation on UVRB (Table 13). All Metrics Are Computed as Unweighted Arithmetic Means of Corresponding Datasets (Denoted as $_ { \mathcal { D } }$ ), with Results Rounded to Three Decimal Places.   

<table><tr><td colspan="3">Dimension Level Column Name Capability Characterization</td></tr><tr><td>Tasks</td><td>TXT</td><td>Text-to-video retrieval (12 datasets)</td></tr><tr><td>Tasks</td><td>CMP</td><td>Composed retrieval (text+image/video, 2 datasets)</td></tr><tr><td>Tasks</td><td>VIS</td><td>Visual-to-video retrieval (image/video, 2 datasets)</td></tr><tr><td>Domains</td><td>CG</td><td>Coarse-grained semantic retrieval (3 datasets)</td></tr><tr><td>Domains</td><td>FG</td><td>Fine-grained semantic retrieval (7 datasets)</td></tr><tr><td>Domains</td><td>LC</td><td>Long-context retrieval (long text/video, 2 datasets)</td></tr><tr><td>Sub-domains</td><td>S</td><td>Spatial fine-grained (object/appearance, 2 datasets)</td></tr><tr><td>Sub-domains</td><td>T</td><td>Temporal fine-grained (motion/dynamics, 2 datasets)</td></tr><tr><td>Sub-domains</td><td>PR</td><td>Partially relevant retrieval (3 datasets)</td></tr><tr><td>Overall</td><td>AVG</td><td>Aggregated task and domains performance (core retrieval paradigms)</td></tr></table>

Table 13 presents the video retrieval performance of mainstream models on UVRB, characterized by the above-defined abilities (Tasks, Domains, and Sub-domains). The AVG score is the arithmetic mean of performance across textual (TXT), composed (CMP), and visual (VIS) retrieval tasks, aggregating model performance across core retrieval paradigms (rather than raw averages over datasets).

Table 13: Video Retrieval Performance of Models on UVRB by Ability Dimensions (Tasks, Domains, Sub-domains). The AVG score denotes the mean performance across textual (TXT), composed (CMP), and visual (VIS) retrieval tasks. Domains involve coarse-grained (CG), fine-grained (FG), and long-context (LC) retrieval, while subdomains include spatial (S), temporal (T), and partially relevant (PR) retrieval. The highest and second-highest values are highlighted in bold and underline.   

<table><tr><td rowspan="2">Model</td><td></td><td>Tasks</td><td></td><td></td><td colspan="3">Domains</td><td colspan="3">Sub-domains</td></tr><tr><td>|AVG|</td><td> TXT</td><td>CMP</td><td>VIS|</td><td>CG</td><td>FG</td><td>LC</td><td>S</td><td>T</td><td>PR</td></tr><tr><td>CLIP4Clip [40]</td><td>41.6</td><td>40.1</td><td>17.8</td><td>71.4</td><td>|38.0</td><td>36.0</td><td>46.3</td><td>55.9</td><td>28.5</td><td>23.6</td></tr><tr><td>ViCLIP [54</td><td>37.5</td><td>33.6</td><td>26.3</td><td>64.0</td><td>38.0</td><td>31.5</td><td>31.3</td><td>48.4</td><td>28.9</td><td>17.1</td></tr><tr><td>VideoCLIP-XL [48]</td><td>51.0</td><td>55.0</td><td>22.7</td><td>63.2</td><td>55.8</td><td>49.3</td><td>60.0</td><td>78.7</td><td>38.1</td><td>31.0</td></tr><tr><td>LanguageBind [78</td><td>50.8</td><td>54.3</td><td>23.1</td><td>64.5</td><td>53.9</td><td>47.9</td><td>61.0</td><td>72.3</td><td>37.8</td><td>33.6</td></tr><tr><td>InternVideo2-1B [55]</td><td>42.0</td><td>42.2</td><td>24.8</td><td>58.1</td><td>48.0</td><td>40.3</td><td>38.3</td><td>60.6</td><td>41.3</td><td>18.9</td></tr><tr><td>InternVideo2-6B [55]</td><td>44.5</td><td>44.8</td><td>22.0</td><td>66.0</td><td>50.4</td><td>41.7</td><td>42.3</td><td>63.1</td><td>40.0</td><td>22.0</td></tr><tr><td>GME-2B [72]</td><td>41.6</td><td>53.9</td><td>34.5</td><td>59.7</td><td>46.1</td><td>47.1</td><td>68.5</td><td>71.6</td><td>34.9</td><td>34.7</td></tr><tr><td>Unite-2B [27]</td><td>50.7</td><td>53.6</td><td>24.2</td><td>65.4</td><td>45.5</td><td>47.1</td><td>68.1</td><td>72.5</td><td>34.7</td><td>34.1</td></tr><tr><td>VLM2Vec-V2 [41]</td><td>53.8</td><td>58.7</td><td>26.3</td><td>61.3</td><td>49.8</td><td>50.2</td><td>76.2</td><td>80.9</td><td>34.8</td><td>34.8</td></tr><tr><td>BGE-VL [76]</td><td>48.0</td><td>49.7</td><td>26.8</td><td>62.2</td><td>44.8</td><td>40.6</td><td>63.6</td><td>66.4</td><td>29.2</td><td>26.1</td></tr><tr><td>UniME-7B [18]</td><td>54.2</td><td>56.1</td><td>30.8</td><td>70.2</td><td>50.0</td><td>51.8</td><td>66.4</td><td>78.5</td><td>39.6</td><td>37.3</td></tr><tr><td>B3-7B [46]</td><td>53.8</td><td>57.0</td><td>27.0</td><td>67.8</td><td>48.2</td><td>50.5</td><td>72.2</td><td>79.7</td><td>36.4</td><td>35.5</td></tr><tr><td>GME-7B [72]</td><td>56.2</td><td>60.4</td><td>34.1</td><td>61.5</td><td>51.8</td><td>50.7</td><td>78.8</td><td>74.9</td><td>37.3</td><td>39.8</td></tr><tr><td>Unite-7B [27]</td><td>55.9</td><td>60.9</td><td>25.4</td><td>66.6</td><td>54.1</td><td>53.9</td><td>74.6</td><td>77.9</td><td>41.2</td><td>42.5</td></tr><tr><td>GVE-3B [0]</td><td>57.1</td><td>61.9</td><td>30.4</td><td>64.7</td><td>55.2</td><td>54.1</td><td>76.4</td><td>81.6</td><td>43.0</td><td>37.7</td></tr><tr><td>Embed-RL-2B</td><td>58.7</td><td>61.1</td><td>20.1</td><td>70.3</td><td>59.1</td><td>54.6</td><td>86.9|</td><td>| 87.2</td><td>43.0</td><td>40.6</td></tr><tr><td>Embed-RL-4B</td><td>58.5</td><td>62.0</td><td>18.4</td><td>70.3</td><td></td><td>60.7 55.6</td><td>86.1</td><td></td><td>87.9 46.0</td><td>40.6</td></tr></table>

# H Training Trajectory Dynamics

This section details dynamic training trajectories of 2B and 4B Embed-RL models via core metrics.

# H.1 Training Metrics

We track key metrics throughout the reinforcement learning phase, as shown in Figure 9. We observe that the entropy declines gradually and then plateaus, while the response length increases steadily with ongoing training. Meanwhile, the reward exhibits a fluctuating upward trend, which is attributed to the effects of in-batch reward sample sampling as the discrepancies between individual samples far outweigh the inherent growth of the reward itself.

![](images/11.jpg)  
Fig. 9: Key RL-phase metrics of Embedder-Guided RL (entropy, response length, reward).

Additionally, we track two core training metrics for 2B and 4B-scale Embed-RL models during contrastive learning: contrastive training loss and gradient norm, as presented in Figure 10. Based on our training experience, neither an excessively large nor an overly small converged loss is favorable.An overly large converged loss suggests that the model fails to correctly discriminate positive and negative samples, whereas an excessively small one indicates that in-batch negative samples are too easily distinguished, leaving the model unable to learn effective discriminative information.Proper adjustment of the sampling ratio and sub-batch size is thus required to ensure the model converges correctly.

![](images/12.jpg)  
Fig. 10: Contrastive training loss and gradient norm of 2B and 4B scale Embed-RL models.

# I Efficiency and Latency

The proposed evidential Traceability CoT (T-CoT) demonstrates prominent efficiency advantages and negligible latency overhead compared with traditional generative embedding methods. For any multimodal retrieval target, T-CoT only needs to be generated once offline, and the derived multimodal embedding vector can be directly stored in the retrieval database. This is different from generative embedding approaches that require on-the-fly reasoning chain and embedding generation for each query, which leads to repeated computational costs. Designed to be targeted and concise, T-CoT only extracts core retrievalrelated multimodal cues such as text keywords, image bounding boxes and video keyframes while abandoning redundant content. This ensures it does not significantly increase single embedding inference latency. Additionally, the stable semantic representation of T-CoT-based embeddings allows for long-term caching and reuse in subsequent tasks without frequent re-generation or updates. This further reduces inference latency and computational consumption in largescale scenarios and makes the framework more suitable for practical industrial deployment.

# J Limitations

This work has several notable limitations. First, the weight coefficients of the multi-component reward function are empirically set for simplicity, lacking an adaptive optimization mechanism for diverse multimodal tasks, which may lead to suboptimal performance in specific scenarios. Second, the constructed dataset excludes partial classification tasks, resulting in relatively weak performance on image classification subtasks; we recommend designing additional loss for classification tasks to avoid false negatives while adapting to large-batch contrastive loss. Finally, we have not applied any hard negative sample mining or curriculum learning strategies, which are expected to further enhance the model's discriminative capability and training stability if incorporated.

# K Exploratory Perspectives

In numerous practical systems such as Multimodal Content Understanding, Recommendation Systems (RS), and Retrieval-Augmented Generation (RAG), high-quality Universal Multimodal Embeddings serve as the core foundation supporting downstream tasks. The quality of embeddings directly determines the performance ceiling of downstream tasks—whether it is the matching accuracy of recommendation systems, the answer correctness of RAG systems, or the semantic alignment performance of cross-modal tasks, all are closely related to it.

In existing solutions, to extract richer semantic information, multi-dimensional feature fusion is often performed through techniques such as semantic segmentation, image and video fine-grained understanding, and user profile analysis. However, this approach typically incurs additional computational and storage overhead, and the feature fusion module suffers from high design complexity.

To address this, we strive to deeply internalize key information capture capabilities, including core visual region localization, video keyframe extraction, and core semantic keyword mining, into the embedding large model itself. This exploration aims to simplify the model structure of feature fusion, reduce the computational and storage costs of semantic vectors, and simultaneously enable embedding results to naturally incorporate task-relevant core semantic information.

# L Comparative Examples of T-CoT Before and After Embedder-Guided RL

Figures 11 to 8 present additional comparative examples of T-CoT before and after the application of Embedder-Guided Reinforcement Learning. These examples fully demonstrate that after RL optimization, the Reasoner achieves more accurate localization of key regions and exhibits improved embedding quality, which further validates the effectiveness of the proposed EG-RL approach in enhancing the performance of T-CoT.

# Text-Image → Text

# Query:

Represent the given image with the following question: Is the plastic bowl to the left of the bread on the right?

# Query T-CoT Before RL:

<thinking> The image shows a meal in bento boxes on a checkered tablecloth. The bread is not visible; instead, there are apple slices in a clear plastic container on the top left and rice in a blue container on the bottom left. The \"bread\" mentioned in the question is not present in the image. The plastic bowl (containing apple slices) is on the left side of the image, but there is no bread to its right. {\"bbox_2d\": [100, 0, 450, 250]]}</thinking>\n<rethink> The plastic bowl with apple slices is located on the left, but there is no bread visible anywhere in the image, so the condition \"to the left of the bread on the right\" cannot be evaluated as true or false./rethink? \n<answer>No, there is no bread visible in the image, so the plastic bowl is not to the left of the bread on the right.</answer>

![](images/13.jpg)

# Failed to ground.

Similarity: 0.5742

# Query T-CoT After RL:

<thinking> The image shows a meal in bento-style containers on a checkered tablecloth. On the right side, there is a container with bread (specifically, sliced bread with spinach) and vegetables. To the left of this container, there is a plastic bowl containing sliced apples. The question asks if the plastic bowl is to the left of the bread on the right. The plastic bowl is indeed located to the left of the bread container. {\"bbox_2d\": [[105, 0, 437, 280], [437, 105, 870, 720]]}</thinking>\n<rethink>The plastic bowl with apples is positioned to the left of the container with bread and vegetables, which is on the right side of the image. This confirms the spatial relationship described in the question.</rethink: \n<answer>Yes, the plastic bowl is to the left of the bread on the right.</answer"

![](images/14.jpg)

Similarity: 0.7070

# Ground to accurate information.

# Target:

Yes, the bowl is to the left of the bread.

Target T-CoT:   
<thinking? The task involves spatial reasoning about object placement. The output confirms a relative position: \"bowl" is to the \"left of the bread.\" Key concepts include object names and directional relationship. {\"text_keywords\": '[\"bowl\", \"bread\", \"left\", \"Yes\"]}/thinking>\n<rethink>The keywords indicate a spatial relationship between two objects, with a clear affirmative confirmation of their relative positioning.</rethink>\n<answer The text confirms that the bowl is positioned to the left of the bread.</answer>

Fig. 11: Example 4 of T-CoT Before and After EG-RL.

# Text-Image → Text

#

Represent the given image with the following question: What is needed to use this transportation?

# Query T-CoT Before RL:

<thinking The image shows a black car being lifted by a car lift, with a person in a blue shirt and red cap operating the lift. The car is on a platform, and the lift mechanism is visible. The key visual element is the car lift, which requires electricity to operate. {\bbox_2d\": [550, 340, 997, 997]}/thinking>\n<rethink>The car lift in the image is a mechanical device that requires power to function, so electricity is needed to use this transportation-related equipment.</rethink>\n<answer>Electricity is needed to use this transportation.</answer>

![](images/15.jpg)

# Missed grounding.

Similarity: 0.4141

# Query T-CoT After RL:

<thinking> The image shows a black car parked on a ramp, with a man in a red cap and blue shirt appearing to work on or inspect it. The car is positioned on a loading ramp, suggesting it is being loaded or unloaded, likely for transport. To use this type of transportation (a car), one typically needs a'driver and a vehicle in operational condition. The man's activity implies he may be a mechanic or technician, indicating that maintenance or preparation is needed. The key visual elements are the car and the man interacting with it. {\"bbox_2d\": [[540, 350, 997, 930], [700, 610, 950, 997]]}/thinking>\n<rethink> The car is the primary transportation vehicle, and the man's presence suggests that maintenance or preparation is required to use it. The ramp indicates it may be part of a transport or loading process, reinforcing the need for operational readiness.</rethink>\n<answer To use this transportation, a driver and a properly maintained vehicle are needed.</answer>

![](images/16.jpg)

Similarity: 0.6055

# Complete grounding.

Target: driver

Target T-CoT: <thinking? The task requires identifying key concepts from the positive text output "driver". The word "driver" is a noun and likely refers to a person who operates a vehicle or a software component that controls hardware. Key concepts include the role or function implied by the term. {\"text_keywords\" [\"driver\", \"person\", \"vehicle\", \"software", \"hardware\"]}/thinking\n<rethinkThe term \"driver\" is contextually ambiguous but commonly refers to either a human operator or a software component; both interpretations are valid based on common usage.</rethink> \n<answer> The text output \"driver\" refers to either a person who operates a vehicle or a software component that interfaces with hardware.</answer>

Fig. 12: Example 5 of T-CoT Before and After EG-RL.

# Text-Video → Text

#

Who did Chandler think was "the other guy" after Joey said he had a feeling Katy was into some other guy?

# Query T-CoT Before RL:

<thinking>The video frames show two men, Chandler and Joey, engaged in a conversation. Their expressions and body language suggest a comedic exchange. The question references a specific line from' the show \"Friends\" where Chandler jokes that Joey is \"the other guy\" after Joey expresses concern about Kathy. The visual content aligns with this scene, showing the characters interacting in a way consistent with their dynamic. {\"key_frames\": [1, 2, 3, 4, 5, 6, 7, 8]}/thinking>\n<rethink>The frames depict Chandler and Joey in a familiar setting, with their expressions and gestures indicating a humorous dialogue. This matches the context of the question, which references a well-known joke from the show.</rethink>\n<answer> Chandler/answer>

Similarity: 0.6719

# Excessive attention to non-key points.

# Query T-CoT After RL:

<thinking>The video frames show two men, Joey and Chandler, engaged in a conversation. In frames 1, 3, and 7, Chandler appears to be reacting with surprise or disbelief, which aligns with his character's comedic response to Joey's statement about Kathy. The context of their interaction and Chandler's facial expressions suggest he is reacting to Joey's comment about \"the other guy." {\"key_frames" [1, 3, 7]}</thinking\n<rethink>Chandler'ssurprised reaction in frames 1, 3, and 7 indicates he is responding to Joey's claim about Kathy being interested in someone else, which is consistent with his character's typical behavior in this scenario.</rethink \n<answer>Chandler thought it was himself.</answer>

# Similarity: 0.7773

Focus on key points.

Target:

Chandler thinks the \"other guy" is himself.

![](images/17.jpg)  
Target T-CoT : <thinking? The output states a belief held by Chandler regarding \"the other guy.\" The key elements are Chandler's perspective, the phrase \"the other guy,\" and the selfreferential conclusion that it is himself. {\"text_keywords\": [""Chandler\", \"the other guy\", \"himself\", \"thinks\"]}/thinking\n<rethink The keywords indicate that the output is a statement of Chandler's belief that \"the other guy\" refers to himself, implying self-awareness or confusion.</rethink>\n<answer>Chandler believes that \"the other guy\" refers to himself./answer

Fig. 13: Example 6 of T-CoT Before and After EG-RL.

# Text → Video

#

A young girl does gymnastics outside in the middle of grass in a sunlight.

# Query T-CoT:

<thinking The description specifies a young girl performing gymnastics outdoors on grass under sunlight. Key visual elements include the subject (young girl), activity (gymnastics), location (grass), and environmental condition (sunlight). These are critical for video retrieval to match the correct visual content. {"text_keywords\": [\"young gir|\", "gymnastics\", \"outside\", \"grass\", \"sunlight\"]}/thinking>\n<rethink>The task requires identifying a video of a young girl doing gymnastics on grass in sunlight, so the retrieval should prioritize these core visual concepts to ensure accurate match.</rethink \n<answerA video of a young girl performing gymnastics on grass outdoors under sunlight.</answer>

# Target T-CoT Before RL:

<thinking?The video shows a young girl in a green top and pink skirt playing energetically on a grassy lawn in front of buildings. Frame 1 shows her posing, Frame shows her smiling and moving, Frame 3-5 capture her performing a cartwheel, and Framhos arem il another child. These frames represent the main actions and the conclusion. {\"key_frames\": [1, 2, 3, 5, 7, 8]}/thinking>\n<rethinkThe key frames (1, 2, 3, 5, 7, 8] capture the girl's initial pose, her joyful movement, the cartwheel action, her running toward the camera, and the final embrace with another child, showing the full sequence of play and interaction./rethink \n<answer The video shows a young girl in a green top and pink skirt playing on a grassy lawn in front of buildings. She poses, runs, performs a cartwheel, and then runs toward the camera before hugging another child in a final photo.</answer>

# Similarity: 0.4961

# Overly detailed and off-topic.

Target T-CoT After RL :   
<thinking> The video shows a young girl playing energetically on a grassy lawn in front of buildings. Frame 1 establishes the setting and the girl's initial pose. Frames 2-5 capture her dynamic movements, including a jump and a handstand. Frame 6 shows her running toward the camera. Frame 7 shows her running closer while smiling. Frame 8 ends with her hugging another child. These frames represent the key actions and transitions.{\"key_frames\": [1, 3, 5, 6, $\angle 1$ thinking\n<rethink The key frames (1, 3, 8 capture the sequene inial pos, acrobac moves, runnig toward camera, and the final embrace, showing the progression of playful activity.</rethink>\n<answer> The video shows a young girl in a turquoise top and pink skirt playing on a grassy lawn in front of buildings. She performs a jump and a handstand, then runs toward the camera ending with her hugging another child./answer>

# Similarity: 0.5664

Concise and query-aligned.

![](images/18.jpg)  
Fig. 14: Example 7 of T-CoT Before and After EG-RL.