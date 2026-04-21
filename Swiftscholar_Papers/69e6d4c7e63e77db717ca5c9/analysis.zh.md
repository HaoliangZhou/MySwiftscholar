# 1. 论文基本信息

## 1.1. 标题
**Magic-MM-Embedding: Towards Visual-Token-Efficient Universal Multimodal Embedding with MLLMs**

## 1.2. 作者
Qi Li, Yanzhe Zhao, Yongxin Zhou, Yameng Wang, Yandong Yang, Yuanjia Zhou, Jue Wang, Zuojian Wang, Jinxiang Liu*

## 1.3. 发表期刊/会议
arXiv (预印本)

## 1.4. 发表年份
2026

## 1.5. 摘要
多模态大语言模型在通用多模态检索中展现出巨大潜力，但其实际应用常受限于处理大量视觉输入词元所带来的高昂计算成本。本文提出了 Magic-MM-Embedding，这是一系列新颖的模型，在通用多模态嵌入领域实现了高效率和最先进的性能。该方法建立在两个协同的支柱之上：(1) 一种高效的 MLLM 架构，结合了视觉词元压缩技术，以大幅降低推理延迟和内存占用；(2) 一种多阶段渐进式训练策略，旨在不仅恢复而且显著提升性能。这种由粗到细的训练范式始于广泛的继续预训练以恢复多模态理解和生成能力，接着进行大规模对比预训练和难负样本挖掘以增强判别能力，最终在 MLLM-as-a-Judge 的指导下进行任务感知微调阶段，以实现精确的数据筛选。综合实验表明，我们的模型在推理效率更高的同时，以显著优势超越了现有方法。

## 1.6. 原文链接
https://arxiv.org/abs/2602.05275 (预印本)

# 2. 整体概括

## 2.1. 研究背景与动机
随着多模态大语言模型的发展，通用多模态检索领域正从传统的双塔架构（如 CLIP）向基于 MLLM 的方法转变。MLLM 具有更强的多模态理解和推理能力，能够处理交错的图像-文本输入。然而，现有的 MLLM 嵌入方法存在一个关键的瓶颈：**高昂的计算成本**。

标准的 MLLM（如 LLaVA-1.5）通常将图像划分为密集的视觉词元序列（例如 576 个词元）并直接注入到语言模型中。虽然这种全序列注入策略有利于 OCR 等细粒度生成任务，但在检索任务中，这些冗余的视觉词元对最终嵌入语义质量的贡献往往有限，却带来了巨大的计算开销（注意力机制的复杂度随序列长度呈二次方增长）。这种低效性成为了在大规模关键任务系统中部署 MLLM 嵌入模型的主要障碍。

本文的切入点在于：**如何在大幅减少视觉词元数量的同时，不牺牲甚至提升检索性能？** 作者认为，通过架构创新（视觉词元压缩）配合专门的训练策略，可以打破效率与性能之间的权衡。

## 2.2. 核心贡献/主要发现
1.  **高效的架构设计：** 提出了 InternVL3-VTC 架构，引入了一种**无参数的视觉词元压缩模块**。通过双线性插值将视觉特征图的空间分辨率降低，从而将视觉词元数量减少了 75%，显著降低了推理延迟和内存消耗。
2.  **渐进式训练策略：** 设计了一套专门针对压缩 MLLM 的由粗到细的三阶段训练流程：
    *   **阶段 1：多模态基础能力恢复**（生成式继续训练）。
    *   **阶段 2：多模态对比预训练**（包含全局难负样本挖掘）。
    *   **阶段 3：任务感知微调**（利用 MLLM-as-a-Judge 进行数据筛选）。
3.  **协同重排序器：** 构建了一个综合检索系统，包括基于压缩架构的高效嵌入模型和协同训练的重排序器，进一步提升了检索精度。
4.  **SOTA 性能：** 在 MMEB、VisDoc 等多个基准测试中取得了最先进的结果。实验证明，即使仅使用 25% 的视觉词元，配合先进的训练管道，模型性能仍能显著超越未压缩的基线模型。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，我们需要掌握以下核心概念：

*   **多模态嵌入：** 将不同模态的数据（如文本、图像、文档）映射到一个统一的语义空间中，使得在这个空间中，语义相似的数据距离较近。这是检索任务的基础。
*   **双塔架构：** 传统的多模态检索模型（如 CLIP）通常包含两个独立的编码器，分别处理图像和文本。这种架构难以捕捉深层的跨模态交互，且文本编码器（如 BERT）的上下文长度和知识储备有限。
*   **多模态大语言模型：** 这类模型（如 LLaVA, Qwen-VL）将视觉编码器与大语言模型（LLM）连接，能够理解并生成包含图像和文本的交错内容。它们具备强大的指令遵循能力和世界知识。
*   **视觉词元：** 在 MLLM 中，图像经过视觉编码器处理后，被转化为一系列向量，每个向量被视为一个“词元”输入到 LLM 中。
*   **对比学习：** 一种训练方法，旨在拉近正样本对（相关的查询和候选）在特征空间中的距离，推远负样本对。常用的损失函数是 InfoNCE。
*   **难负样本挖掘：** 在对比学习中，随机采样的负样本往往太容易区分。难负样本挖掘旨在找出那些与查询相似但实际上不相关的样本，迫使模型学习更细微的语义差异。

## 3.2. 前人工作
作者在相关工作部分回顾了多模态表示学习的演进：

*   **CLIP 风格模型：** 如 CLIP [74], SigLIP [96] 等。它们开创了图像-文本对比学习的范式，但受限于双塔架构。
*   **通用检索模型：** 如 UniR [86], MagicLens [99]。它们扩展了输入模态（支持交错内容），但本质上仍是独立编码后融合，缺乏深层交互。
*   **MLLM 嵌入模型：** 如 E5-V [34], VLM2Vec [35], UniME [22]。这些工作利用 MLLM 的强大能力，通过指令微调、数据增强、难负样本挖掘、梯度放大等技术显著提升了检索性能。然而，这些工作大多直接通用的 MLLM 架构，**忽略了视觉词元冗余带来的高推理成本问题**。

## 3.3. 技术演进
该领域经历了从“简单的双塔对比学习”到“基于 MLLM 的深度语义理解”的演变。早期的 CLIP 模型虽然高效，但理解能力有限。后来的 VLM2Vec 等工作通过引入 MLLM 解决了理解能力的问题，却引入了计算复杂度的新问题。本文的工作正是处于这一技术脉络的最新阶段，旨在解决 MLLM 落地部署时的效率瓶颈。

## 3.4. 差异化分析
本文与现有工作的核心区别在于**对效率的关注**。
*   **现有工作：** 主要关注如何通过增加数据量、优化损失函数、引入复杂的负样本挖掘策略来提升检索准确率，通常假设使用标准的、未经压缩的 MLLM 架构。
*   **本文方法：** 直面计算成本问题，提出通过架构层面的**视觉词元压缩**来从根本上减少计算量。为了弥补压缩可能带来的信息损失，作者设计了独特的**多阶段渐进训练策略**，证明了“少即是多”——更少的词元配合更好的训练，可以超越更多的词元配合标准训练。

# 4. 方法论

本章将详细拆解 Magic-MM-Embedding 的技术方案，包括其架构设计、训练流程和重排序机制。

## 4.1. 方法原理
本文的核心思想是<strong>“架构压缩 + 训练补偿”</strong>。
首先，通过在视觉编码器和 LLM 连接器之间插入一个无参数的空间插值模块，将视觉特征图的尺寸缩小，从而大幅减少进入 LLM 的词元数量。这直接降低了注意力机制的二次方计算复杂度。
其次，由于压缩操作改变了视觉特征的分布和密度，直接使用对比学习训练会导致性能下降。因此，作者设计了一个由粗到细的三阶段训练策略：
1.  **恢复：** 先通过生成式训练，让模型适应压缩后的特征，恢复其基础的多模态理解能力。
2.  **判别：** 通过大规模对比学习，特别是引入难负样本，训练模型生成具有判别力的嵌入。
3.  **精炼：** 利用 MLLM 作为裁判，筛选高质量数据，进行任务感知的微调，进一步提升特定任务的表现。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 任务定义与基础公式
首先，作者定义了通用多模态检索任务。给定一个查询 $q$ 或候选 $c$，它们由任务指令、视觉上下文和文本上下文组成。模型的目标是将这些输入映射到共享的语义空间。

**步骤 1：嵌入生成**
模型使用一个带有视觉词元压缩功能的 MLLM 作为编码器 $f$，将输入 $x$ 映射为隐藏状态序列 $\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_L$。为了得到最终的嵌入向量 $\mathbf{z}_x$，作者取最后一个隐藏状态 $\mathbf{h}_L$ 并进行 $\ell_2$ 归一化。公式如下：

$$
\mathbf{z}_x = \frac{\mathbf{h}_L}{\|\mathbf{h}_L\|_2}.
$$

*   **符号解释：**
    *   $\mathbf{z}_x$：输入样本 $x$ 的最终嵌入向量。
    *   $\mathbf{h}_L$：MLLM 输出的最后一个词元的隐藏状态。
        表示向量的 $\ell_2$ 范数（即向量长度的平方和开根号）。归一化操作使得所有嵌入向量都在单位超球面上，便于计算余弦相似度。

**步骤 2：对比学习目标**
为了学习具有判别力的嵌入空间，作者使用 InfoNCE 损失进行训练。对于每个查询 $q$，定义一个候选集合 $\mathcal{C}_q$，包含一个正样本 $c_q^+$ 和一组负样本 $\mathcal{C}_q^-$。模型的目标是最大化查询与正样本的相似度，同时抑制负样本。损失函数公式如下：

$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\log \frac{\exp(\mathbf{z}_q^{\top} \mathbf{z}_{c_q^+} / \tau)}{\exp(\mathbf{z}_q^{\top} \mathbf{z}_{c_q^+} / \tau) + \sum_{c_q^- \in \mathcal{C}_q^-} \exp(\mathbf{z}_q^{\top} \mathbf{z}_{c_q^-} / \tau)},
$$

*   **符号解释：**
    *   $\mathcal{L}_{\mathrm{InfoNCE}}$：InfoNCE 损失值。
    *   $\mathbf{z}_q$：查询 $q$ 的嵌入向量。
    *   $\mathbf{z}_{c_q^+}$：正样本候选的嵌入向量。
    *   $\mathbf{z}_{c_q^-}$：负样本候选的嵌入向量。
    *   $\top$：向量转置操作，$\mathbf{z}_q^{\top} \mathbf{z}_{c_q^+}$ 表示两个向量的点积（即相似度）。
    *   $\tau$：温度参数，用于控制 softmax 分布的平滑程度。
    *   $\exp(\cdot)$：以自然常数 $e$ 为底的指数函数。

### 4.2.2. 无参数视觉词元压缩
这是本文架构创新的核心。

**步骤 1：标准 MLLM 的瓶颈**
在标准 MLLM 中，视觉编码器输出一个特征图 $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$，其中 $H \times W$ 是空间维度，$C$ 是通道数。这个特征图被展平为 $N = H \times W$ 个词元并输入 LLM。当 $N$ 很大时（如 576），计算成本极高。

**步骤 2：引入压缩模块**
为了解决“词元过载”问题，作者在视觉编码器和连接器之间插入了一个压缩模块。不同于需要学习的复杂压缩器，本文采用了一种直接的双线性插值策略。设 $\Phi(\cdot)$ 为双线性下采样操作，目标是将空间分辨率缩小 $s$ 倍。压缩后的特征图 $\mathbf{F}'$ 计算公式为：

$$
\mathbf{F}' = \Phi(\mathbf{F}; H', W') \in \mathbb{R}^{H' \times W' \times C},
$$

其中，目标空间维度为 $H' = H / s$ 和 $W' = W / s$。

*   **符号解释：**
    *   $\mathbf{F}$：原始视觉特征图。
    *   $\mathbf{F}'$：压缩后的视觉特征图。
    *   $\Phi$：双线性插值下采样函数。
    *   `H, W`：原始特征图的高度和宽度。
    *   `H', W'`：压缩后特征图的高度和宽度。
    *   $s$：下采样倍率（压缩因子）。

**步骤 3：输入 LLM**
压缩后的特征图 $\mathbf{F}'$ 被展平为词元序列，词元数量从 $N$ 减少到 $N / s^2$。这直接降低了 LLM 的注意力计算复杂度，且不引入任何额外的可训练参数。

下图（原文 Figure 2）展示了提议的视觉令牌压缩架构 InternVL3-VTC 及其在通用多模态检索中的应用。

![Figure:Overview the proposed visual-token-cient architecture orniversalmultimodal retrieval.) The proposed MLLM architecture with Visual Token Compression, InternVL3-VTC. (b, c) The proposed inferenceefficient, universal multimodal embedder and reranker, both of which are built upon InternVL3-VTC.](images/2.jpg)
*该图像是示意图，展示了提议的视觉令牌压缩架构 InternVL3-VTC 及其在通用多模态检索中的应用。图中包括了各个组件，如MLP投影器、文本标记器和视觉编码器，以及魔法多模态嵌入和重排方法的示意。*

### 4.2.3. 渐进式由粗到细训练管道
为了克服压缩带来的信息损失，作者设计了三阶段训练策略。

**阶段 1：多模态基础能力恢复**
由于插值模块改变了视觉特征的空间结构，预训练的 LLM 主干可能无法直接适应。因此，第一阶段的目标不是检索，而是**对齐**。通过在通用的多模态指令跟随数据集上进行生成式训练，恢复模型的基础多模态理解和生成能力。这里使用标准的自回归下一个词元预测损失：

$$
\mathcal{L}_{\mathrm{NTP}} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, x),
$$

*   **符号解释：**
    *   $\mathcal{L}_{\mathrm{NTP}}$：Next Token Prediction 损失。
    *   $y_t$：真实的第 $t$ 个文本词元。
    *   $y_{<t}$：第 $t$ 个词元之前的所有文本词元序列。
    *   $x$：多模态输入（包含视觉和文本上下文）。
    *   $P(\cdot)$：模型预测的概率分布。

        这一步至关重要，它弥合了压缩带来的分布差异，确保 LLM 在进入嵌入学习前仍具备推理能力。

**阶段 2：多模态对比预训练**
在恢复基础能力后，模型转向多模态表示学习。这一阶段在大规模检索语料库上进行，包含两个步骤：
1.  **热身：** 使用标准的 InfoNCE 损失和批内负样本进行训练。
2.  **难负样本挖掘：** 为了增强判别力，引入“全局难负样本挖掘”。对于每个查询，从整个数据集中检索 Top-K 候选，排除 Ground Truth 后，剩下的作为难负样本。这比随机批内负样本更具挑战性。

<strong>阶段 3：任务感知微调 (MLLM-as-a-Judge)</strong>
标准训练数据常包含“假负样本”（即被标记为负但实际上相关的样本）且缺乏足够难的负样本。为此，作者利用一个专家 MLLM（如 Qwen3-VL）作为裁判。
1.  **检索与判断：** 对于训练集中的每个查询 $q$，用阶段 2 的模型检索 Top-K 候选。
2.  **裁判打分：** 将 $(q, c_i)$ 对输入裁判 MLLM，判断其相关性。如果裁判认为相关，则视为“漏掉的正样本”（扩充正样本集）；如果不相关，则视为“高质量的难负样本”。
3.  **训练：** 使用这些经过裁判筛选的高质量数据对模型进行微调，进一步提升其在复杂任务上的泛化能力。

### 4.2.4. 协同重排序器
为了构建完整的检索系统，作者基于阶段 3 的模型训练了一个重排序器。重排序器利用模型保留的生成能力，对嵌入模型检索出的 Top-K 结果进行精细重排。

**Pointwise 重排序：**
模型独立评估查询-候选对。对于正样本对，模型输出 "Yes"；对于负样本对，输出 "No"。损失函数为标准的交叉熵损失：

$$
\mathcal{L}_{\mathrm{point}} = \mathcal{L}_{\mathrm{CE}}(\mathrm{Yes}, r(q, c^+)) + \mathcal{L}_{\mathrm{CE}}(\mathrm{No}, r(q, c^-)),
$$

*   **符号解释：**
    *   $\mathcal{L}_{\mathrm{point}}$：Pointwise 排序损失。
    *   $\mathcal{L}_{\mathrm{CE}}$：交叉熵损失函数。
    *   `r(q, c)`：模型对输入对 `(q, c)` 的自回归生成过程。
    *   $c^+$：正样本候选。
    *   $c^-$：负样本候选。

**Listwise 重排序：**
模型一次性评估一个查询和多个候选（包含 1 个正样本和多个负样本），任务是输出正样本在列表中的位置索引 $k$。损失函数为：

$$
\mathcal{L}_{\mathrm{list}} = \mathcal{L}_{\mathrm{CE}}(k, r(q, c_1^-, \ldots, c^+, \ldots, c_M^-)).
$$

*   **符号解释：**
    *   $\mathcal{L}_{\mathrm{list}}$：Listwise 排序损失。
    *   $k$：正样本在候选列表中的真实位置索引。
    *   $c_1^-, \ldots, c_M^-$：$M$ 个负样本候选。

        最终的总损失是两者的加权和：$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{point}} + \mathcal{L}_{\mathrm{list}}$。

# 5. 实验设置

## 5.1. 数据集
实验使用了以下几类数据集来全面评估模型性能：

*   <strong>自然图像检索 (MMEB)：</strong> 一个包含 36 个子数据集和 4 个元任务（分类、VQA、检索、定位）的综合基准。用于评估通用多模态检索能力。
*   **视觉文档检索：** 包括 ViDoRe v1/v2, VisRAG, ViDoSeek 等。这是一项细粒度任务，理论上需要高分辨率输入来保留文本细节，用于测试模型在处理密集文本图像时的能力。
*   **跨模态检索：** 包括 Flickr30K, MSCOCO, ShareGPT4V, Urban1K, SugarCrepe。用于评估标准的文本-图像双向检索能力。

    这些数据集涵盖了从通用图像到复杂文档、从简单检索到组合推理的广泛场景，能够有效验证方法的通用性和鲁棒性。

## 5.2. 评估指标
论文中主要使用了以下指标：

### 5.2.1. Precision@1 (P@1)
*   **概念定义：** 这是一个广泛用于检索系统的指标。它计算在检索结果列表中，排名第一位的结果是否是正确的（即是否为 Ground Truth）。如果是，得分为 1，否则为 0。它直观地反映了模型“一击即中”的能力。
*   **数学公式：**
    $$
    P@1 = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{rank}(q_i, \text{gt}_i) = 1)
    $$
*   **符号解释：**
    *   $N$：查询的总数。
    *   $q_i$：第 $i$ 个查询。
    *   $\text{gt}_i$：第 $i$ 个查询对应的真实标注数据。
    *   $\text{rank}(q_i, \text{gt}_i)$：真实标注数据在查询 $q_i$ 的检索结果列表中的排名。
    *   $\mathbb{I}(\cdot)$：指示函数，当条件为真时为 1，否则为 0。

### 5.2.2. NDCG@5
*   **概念定义：** 归一化折损累计增益。它不仅考虑检索结果是否相关，还考虑相关项在排名列表中的位置（排名越靠前得分越高）。@5 表示只考虑前 5 个结果。这对于评估重排序器的效果非常有用，因为重排序器主要优化 Top-K 结果的顺序。
*   **数学公式：**
    $$
    \text{NDCG}@5 = \frac{1}{\text{IDCG}} \sum_{j=1}^{5} \frac{2^{rel_j} - 1}{\log_2(j + 1)}
    $$
*   **符号解释：**
    *   $rel_j$：第 $j$ 个检索结果的相关性等级（通常二值化：相关为 1，不相关为 0，或分级评分）。
    *   $\text{IDCG}$：理想折损累计增益，即当所有最相关的结果都排在最前面时能达到的最大 DCG 值。用于归一化，使分数在 0 到 1 之间。

## 5.3. 对比基线
论文将 Magic-MM-Embedding 与以下代表性基线模型进行了比较：
*   **传统双塔模型：** CLIP, SigLIP, EVA-CLIP, MagicLens。作为性能下限参考。
*   **MLLM 嵌入模型：** E5-V, VLM2Vec, UniME, LLaVE, QQMM, UniME-V2。这些是当前 SOTA 的 MLLM 检索方法，用于验证本文方法的优越性。
*   **视觉文档检索专用模型：** ColPali, GME, Ops-MM-embedding。用于在 VisDoc 任务上进行针对性对比。

# 6. 实验结果与分析

## 6.1. 核心结果分析
实验结果强有力地验证了 Magic-MM-Embedding 的有效性。即使在视觉词元减少 75% 的情况下，模型依然在多个基准上取得了最先进的结果。

### 6.1.1. MMEB 基准测试结果
在 MMEB 基准上，Magic-MM-Embedding 在 2B 和 8B 参数规模下均显著优于基线模型。例如，在 2B 规模下，Magic-MM-Embedding (E+R) 的 Overall 得分为 70.2，超过了 UniME-V2 (E+R) 的 67.4。这证明了渐进式训练策略成功克服了词元压缩带来的潜在信息损失。

以下是原文 [Table 3] 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Backbone (Model Size)</th>
<th colspan="4">Per Meta-Task Score</th>
<th colspan="3">Average Score</th>
</tr>
<tr>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>Grounding</th>
<th>IND</th>
<th>OOD</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9">Zero-shot Results</td>
</tr>
<tr>
<td>CLIP [74]</td>
<td>-(0.4B)</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.1</td>
<td>38.7</td>
<td>37.8</td>
</tr>
<tr>
<td>SigLIP [96]</td>
<td>-(0.9B)</td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>32.3</td>
<td>38.0</td>
<td>34.8</td>
</tr>
<tr>
<td>EVA-CLIP [79]</td>
<td>-(8.1B)</td>
<td>56.0</td>
<td>10.4</td>
<td>49.2</td>
<td>58.9</td>
<td>38.1</td>
<td>45.6</td>
<td>43.7</td>
</tr>
<tr>
<td>MagicLens [99]</td>
<td>-(0.4B)</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td>31.0</td>
<td>23.7</td>
<td>27.8</td>
</tr>
<tr>
<td>E5-V [34]</td>
<td>Phi3.5-V (4.2B)</td>
<td>39.1</td>
<td>9.6</td>
<td>38.0</td>
<td>57.6</td>
<td>33.1</a>
<td>31.9</td>
<td>36.1</td>
</tr>
<tr>
<td>E5-V [34]</td>
<td>LLaVA-1.6 (8.4B)</td>
<td>39.7</td>
<td>10.8</td>
<td>39.4</td>
<td>60.2</td>
<td>34.2</td>
<td>33.4</td>
<td>37.5</td>
</tr>
<tr>
<td colspan="9">Trained with MMEB</td>
</tr>
<tr>
<td>VLM2Vec-V1 [35]</td>
<td>Qwen2VL (2.2B)</td>
<td>59.0</td>
<td>49.4</td>
<td>65.4</td>
<td>73.4</td>
<td>66.0</td>
<td>52.6</td>
<td>59.3</td>
</tr>
<tr>
<td>UniME [22]</td>
<td>Phi3.5-V (4.2B)</td>
<td>54.8</td>
<td>55.9</td>
<td>64.5</td>
<td>81.8</td>
<td>68.2</td>
<td>52.7</td>
<td>64.2</td>
</tr>
<tr>
<td>LLaVE [41]</td>
<td>Aquila-VL (2.0B)</td>
<td>62.1</td>
<td>60.2</td>
<td>65.2</td>
<td>84.9</td>
<td>69.4</td>
<td>59.8</td>
<td>65.2</td>
</tr>
<tr>
<td>UniME-V2 (E) [23]</td>
<td>Qwen2VL (2.2B)</td>
<td>62.1</td>
<td>56.3</td>
<td>68.0</td>
<td>72.7</td>
<td>67.4</td>
<td>58.9</td>
<td>63.6</td>
</tr>
<tr>
<td>UniME-V2 (E+) [23]</td>
<td>Qwen2VL (2.2B)</td>
<td>64.1</td>
<td>64.3</td>
<td>71.6</td>
<td>70.6</td>
<td>69.8</td>
<td>64.3</td>
<td>67.4</td>
</tr>
<tr>
<td><b>Magic-MM-Embedding (E)</b></td>
<td>InternVL3-VTC (1.9B)</td>
<td>60.9</td>
<td>63.3</td>
<td>72.2</td>
<td>84.6</td>
<td>74.7</td>
<td>59.5</td>
<td><b>68.0</b></td>
</tr>
<tr>
<td><b>Magic-MM-Embedding (E+R)</b></td>
<td>InternVL3-VTC (1.9B)</td>
<td>61.3</td>
<td>67.2</td>
<td>73.5</td>
<td>89.8</td>
<td>75.2</td>
<td>63.9</td>
<td><b>70.2</b></td>
</tr>
<tr>
<td>VLM2Vec-V1 [35]</td>
<td>Qwen2VL (8.3B)</td>
<td>62.6</td>
<td>57.8</td>
<td>69.9</td>
<td>81.7</td>
<td>65.2</td>
<td>56.3</td>
<td>65.8</td>
</tr>
<tr>
<td>UniME [22]</td>
<td>LLaVA-OV (8.0B)</td>
<td>66.8</td>
<td>66.6</td>
<td>70.5</td>
<td>90.9</td>
<td>74.6</td>
<td>65.8</td>
<td>70.7</td>
</tr>
<tr>
<td>LLaVE [41]</td>
<td>LLaVA-OV (8.0B)</td>
<td>65.7</td>
<td>65.4</td>
<td>70.9</td>
<td>91.9</td>
<td>75.0</td>
<td>64.4</td>
<td>70.3</td>
</tr>
<tr>
<td>QQMM [89]</td>
<td>LLaVA-OV (8.0B)</td>
<td>66.8</td>
<td>66.8</td>
<td>70.5</td>
<td>90.4</td>
<td>74.7</td>
<td>65.6</td>
<td>70.7</td>
</tr>
<tr>
<td>UniME-V2 [23]</td>
<td>LLaVA-OV (8.0B)</td>
<td>65.3</td>
<td>67.6</td>
<td>72.9</td>
<td>90.2</td>
<td>74.8</td>
<td>66.7</td>
<td>71.2</td>
</tr>
<tr>
<td>UniME-V2 (E) [23]</td>
<td>Qwen2VL (8.3B)</td>
<td>64.0</td>
<td>60.1</td>
<td>73.1</td>
<td>82.8</td>
<td>72.0</td>
<td>63.0</td>
<td>68.0</td>
</tr>
<tr>
<td>UniME-V2 (E+R) [23]</td>
<td>Qwen2VL (8.3B)</td>
<td>63.8</td>
<td>66.3</td>
<td>73.5</td>
<td>75.0</td>
<td>71.7</td>
<td>65.6</td>
<td>69.0</td>
</tr>
<tr>
<td><b>Magic-MM-Embedding (E)</b></td>
<td>InternVL3-VTC (8.1B)</td>
<td>64.8</td>
<td>68.1</td>
<td>75.0</td>
<td>88.7</td>
<td>78.3</td>
<td>63.6</td>
<td><b>71.8</b></td>
</tr>
<tr>
<td><b>Magic-MM-Embedding (E+R)</b></td>
<td>InternVL3-VTC (8.1B)</td>
<td>64.3</td>
<td>70.9</td>
<td>75.7</td>
<td>90.4</td>
<td>78.4</td>
<td>65.9</td>
<td><b>72.8</b></td>
</tr>
</tbody>
</table>

### 6.1.2. 视觉文档检索结果
在 VisDoc 任务中，Magic-MM-Embedding 同样表现出色。这是一个极具挑战性的领域，通常认为需要高分辨率输入。然而，本文模型在压缩 75% 词元的情况下，依然取得了 SOTA 结果。特别是结合重排序器 (E+R) 后，在 8B 模型上达到了 75.8 的 Overall 得分，超过了使用私有数据的强力基线 GME。

以下是原文 [Table 4] 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Backbone (Model Size)</th>
<th colspan="5">VisDoc</th>
</tr>
<tr>
<th>VDRv1</th>
<th>VDRv2</th>
<th>VR</th>
<th>OOD</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td>GME [102]</td>
<td>Qwen2VL (2.2B)</td>
<td>86.1</td>
<td>54.0</td>
<td>82.5</td>
<td>43.1</td>
<td>72.7</td>
</tr>
<tr>
<td>ColPali [18]</td>
<td>Paligemma (2.9B)</td>
<td>83.6</td>
<td>52.0</td>
<td>81.1</td>
<td>43.1</td>
<td>71.0</td>
</tr>
<tr>
<td>Ops-MM-embedding-v1 [72]</td>
<td>Qwen2VL (8.3B)</td>
<td>80.1</td>
<td>59.6</td>
<td>79.3</td>
<td>43.3</td>
<td>70.3</td>
</tr>
<tr>
<td>VLM2Vec-V2 [71]</td>
<td>Qwen2VL (2.2B)</td>
<td>75.5</td>
<td>44.9</td>
<td>79.4</td>
<td>39.4</td>
<td>65.4</td>
</tr>
<tr>
<td><b>Magic-MM-Embedding (E)</b></td>
<td>InternVL3-VTC (1.9B)</td>
<td>83.4</td>
<td>53.3</td>
<td>85.6</td>
<td>42.2</td>
<td><b>72.1</b></td>
</tr>
<tr>
<td><b>Magic-MM-Embedding (E+R)</b></td>
<td>InternVL3-VTC (1.9B)</td>
<td>84.4</td>
<td>56.1</td>
<td>87.4</td>
<td>41.8</td>
<td><b>73.3</b></td>
</tr>
<tr>
<td>Ops-MM-embedding-v1 [72]</td>
<td>Qwen2VL (8.3B)</td>
<td>80.1</td>
<td>59.6</td>
<td>79.3</td>
<td>43.3</td>
<td>70.3</td>
</tr>
<tr>
<td>GME [102]</td>
<td>Qwen2VL (8.3B)</td>
<td>89.4</td>
<td>55.6</td>
<td>85.0</td>
<td>44.4</td>
<td>75.2</td>
</tr>
<tr>
<td>LamRA-Qwen2 [59]</td>
<td>Qwen2VL (8.3B)</td>
<td>22.0</td>
<td>11.5</td>
<td>37.4</td>
<td>21.0</td>
<td>23.9</td>
</tr>
<tr>
<td>LamRA-Qwen2.5 [59]</td>
<td>Qwen2.5VL (8.3B)</td>
<td>56.3</td>
<td>33.3</td>
<td>58.2</td>
<td>40.1</td>
<td>50.2</td>
</tr>
<tr>
<td>VLM2Vec-V2 [71]</td>
<td>Qwen2VL (8.3B)</td>
<td>78.8</td>
<td>52.6</td>
<td>82.7</td>
<td>42.1</td>
<td>69.3</td>
</tr>
<tr>
<td><b>Magic-MM-Embedding (E)</b></td>
<td>InternVL3-VTC (8.1B)</td>
<td>86.1</td>
<td>59.9</td>
<td>87.6</td>
<td>43.4</td>
<td><b>75.0</b></td>
</tr>
<tr>
<td><b>Magic-MM-Embedding (E+R)</b></td>
<td>InternVL3-VTC (8.1B)</td>
<td>86.9</td>
<td>60.4</td>
<td>89.2</td>
<td>43.1</td>
<td><b>75.8</b></td>
</tr>
</tbody>
</table>

### 6.1.3. 推理效率对比
Table 6 展示了推理效率的显著提升。在相似参数规模下，Magic-MM-Embedding 的平均视觉词元数和推理延迟远低于基线模型。例如，与 LLaVE-2B 相比，Magic-MM-Embedding-2B 将 MMEB 查询的推理延迟从 162.8 ms 降低到了 29.9 ms，提速约 5.4 倍。这直接证明了视觉词元压缩的有效性。

以下是原文 [Table 6] 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Backbone (Model Size)</th>
<th colspan="4">MMEB</th>
<th colspan="4">VisDoc</th>
</tr>
<tr>
<th>#VTq</th>
<th>lq</th>
<th>#VTc</th>
<th>lc</th>
<th>#VTq</th>
<th>lq</th>
<th>#VTc</th>
<th>lc</th>
</tr>
</thead>
<tbody>
<tr>
<td>VLM2Vec [35]</td>
<td>Phi3.5-V (4.2B)</td>
<td>757.0</td>
<td>99.4</td>
<td>757.0</td>
<td>85.9</td>
<td>0</td>
<td>34.0</td>
<td>757.0</td>
<td>128.6</td>
</tr>
<tr>
<td>GME [102]</td>
<td>Qwen2VL (2.2B)</td>
<td>362.8</td>
<td>46.8</td>
<td>256.0</td>
<td>34.5</td>
<td>0</td>
<td>19.3</td>
<td>1024.0</td>
<td>153.8</td>
</tr>
<tr>
<td>LLaVE [41]</td>
<td>Aquila-VL (2.0B)</td>
<td>3699.0</td>
<td>162.8</td>
<td>3699.0</td>
<td>143.0</td>
<td>0</td>
<td>18.5</td>
<td>3699.0</td>
<td>233.6</td>
</tr>
<tr>
<td>InternVL3 [107]</td>
<td>InternVL3 (1.9B)</td>
<td>398.4</td>
<td>37.1</td>
<td>256.0</td>
<td>29.2</td>
<td>0</td>
<td>19.8</td>
<td>1280.0</td>
<td>103.6</td>
</tr>
<tr>
<td><b>Magic-MM-Embedding</b></td>
<td><b>InternVL3-VTC (1.9B)</b></td>
<td><b>99.6</b></td>
<td><b>29.9</b></td>
<td><b>64.0</b></td>
<td><b>26.1</b></td>
<td>0</td>
<td>19.7</td>
<td><b>320.0</b></td>
<td><b>57.3</b></td>
</tr>
<tr>
<td>VLM2Vec [35]</td>
<td>LLaVA-1.6 (8.4B)</td>
<td>2928.0</td>
<td>332.3</td>
<td>2928.0</td>
<td>278.9</td>
<td>0</td>
<td>32.4</td>
<td>2928.0</td>
<td>458.1</td>
</tr>
<tr>
<td>GME [102]</td>
<td>Qwen2VL (8.3B)</td>
<td>362.8</td>
<td>82.2</td>
<td>256.0</td>
<td>56.7</td>
<td>0</td>
<td>26.6</td>
<td>1024.0</td>
<td>268.2</td>
</tr>
<tr>
<td>LamRA [59]</td>
<td>Qwen2.5VL (8.3B)</td>
<td>362.8</td>
<td>83.4</td>
<td>256.0</td>
<td>61.6</td>
<td>0</td>
<td>28.9</td>
<td>1024.0</td>
<td>251.7</td>
</tr>
<tr>
<td>UniME-V2 [23]</td>
<td>LLaVA-OV (8.0B)</td>
<td>7371.0</td>
<td>906.9</td>
<td>7371.0</td>
<td>788.1</td>
<td>0</td>
<td>32.1</td>
<td>7371.0</td>
<td>1341.1</td>
</tr>
<tr>
<td>InternVL3 [107]</td>
<td>InternVL3 (8.1B)</td>
<td>398.4</td>
<td>76.7</td>
<td>256.0</td>
<td>55.9</td>
<td>0</td>
<td>33.8</td>
<td>1280.0</td>
<td>260.4</td>
</tr>
<tr>
<td><b>Magic-MM-Embedding</b></td>
<td><b>InternVL3-VTC (8.1B)</b></td>
<td><b>99.6</b></td>
<td><b>50.9</b></td>
<td><b>64.0</b></td>
<td><b>40.6</b></td>
<td>0</td>
<td>33.8</td>
<td><b>320.0</b></td>
<td><b>94.8</b></td>
</tr>
</tbody>
</table>

## 6.2. 消融实验/参数分析
作者进行了详细的消融实验来验证各组件的有效性。

### 6.2.1. 渐进式训练管道与重排序器
Table 7 的结果表明，训练管道的每个阶段都带来了性能提升。加入全局难负样本挖掘 显著提高了判别能力。而使用 MLLM-as-a-Judge 进行微调进一步提升了性能。最后，加入重排序器带来了额外的增益。

以下是原文 [Table 7] 的结果：

| Stage 2 (Warm-Up) | Stage 2 (Global-HNM) | Stage 3 (MLLM-Judge-FT) | Inference (Reranker) | MMEB | VisDoc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | | | X | 62.9 | 68.4 |
| X | | | X | 65.4 | 70.7 |
| X | X | | X | 68.0 | 72.1 |
| X | X | X | | 68.0 | 72.1 |
| X | X | X | X | 70.2 | 73.3 |

### 6.2.2. 难负样本的数量与类型
Table 8 分析了难负样本数量 $n$ 的影响。结果显示，引入任何数量的 MLLM-based 难负样本都能带来显著提升。性能在 $n=12$ 或 `16` 时达到峰值。此外，对比 MLLM-based 和 Rule-based 的难负样本，前者始终优于后者，证明了利用 MLLM 进行数据筛选的有效性。

以下是原文 [Table 8] 的结果：

<table>
<thead>
<tr>
<th rowspan="2">#HN (n)</th>
<th colspan="3">MLLM-based HN</th>
<th colspan="3">Rule-based HN</th>
</tr>
<tr>
<th>MMEB</th>
<th>VisDoc</th>
<th>Avg.</th>
<th>MMEB</th>
<th>VisDoc</th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>65.5</td>
<td>70.6</td>
<td>68.1</td>
<td>65.5</td>
<td>70.6</td>
<td>68.1</td>
</tr>
<tr>
<td>4</td>
<td>67.4</td>
<td>71.7</td>
<td>69.5</td>
<td>65.7</td>
<td>69.5</td>
<td>67.6</td>
</tr>
<tr>
<td>8</td>
<td>67.8</td>
<td>71.9</td>
<td>69.9</td>
<td>66.5</td>
<td>70.6</td>
<td>68.6</td>
</tr>
<tr>
<td>12</td>
<td>68.0</td>
<td>72.1</td>
<td>70.0</td>
<td>67.0</td>
<td>70.1</td>
<td>68.5</td>
</tr>
<tr>
<td>16</td>
<td>67.9</td>
<td>72.1</td>
<td>70.0</td>
<td>65.9</td>
<td>70.7</td>
<td>68.3</td>
</tr>
<tr>
<td>20</td>
<td>67.6</td>
<td>71.9</td>
<td>69.8</td>
<td>67.1</td>
<td>70.5</td>
<td>68.8</td>

    </r>
  </tbody>
</table>

### 6.2.3. 视觉词元压缩对训练效率的影响
Table 9 展示了视觉词元压缩对训练速度的提升。在模型性能几乎不变的情况下，压缩模型的训练时间从约 53 小时减少到了 23 小时，并且显著降低了 GPU 显存占用，允许更大的批大小。

以下是原文 [Table 9] 的结果：

| Backbone | Training Duration | MMEB | VisDoc | Global Batch Size |
| :--- | :--- | :--- | :--- | :--- |
| InternVL3 (vanilla) | 52h 43m 35s | 62.9 | 68.4 | 6144 |
| InternVL3-VTC (ours) | 22h 57m 6s | 63.7 | 68.5 | 3456 |

# 7. 总结与思考

## 7.1. 结论总结
本文成功解决了 MLLM 嵌入模型在通用多模态检索中的计算效率瓶颈。通过提出 InternVL3-VTC 架构，利用无参数的双线性插值将视觉词元压缩了 75%，并结合一套精心设计的由粗到细的三阶段训练策略（基础恢复、对比预训练、任务感知微调），作者证明了高效率和高性能可以兼得。实验结果表明，Magic-MM-Embedding 在多个基准测试中取得了 SOTA 结果，同时在推理延迟和内存占用上显著优于现有方法。

## 7.2. 局限性与未来工作
尽管本文取得了显著成果，但仍存在一些潜在的局限性：
*   **压缩的通用性：** 虽然本文在 InternVL3 上取得了成功，但这种压缩策略在其他架构（如基于 CNN 的视觉编码器或不同连接器设计的 MLLM）上的效果如何尚需验证。
*   **极端细节的丢失：** 虽然训练策略补偿了大部分信息损失，但对于某些极度依赖像素级细节的任务（如非常细小的文字识别），压缩可能仍会带来不可逆的性能下降。
*   **训练成本：** 三阶段训练策略虽然有效，但引入了额外的训练复杂度和计算开销（特别是阶段 3 使用 MLLM-as-a-Judge 进行数据筛选）。

    未来工作可以探索：
1.  **自适应压缩：** 根据图像内容的复杂度动态调整压缩率，而不是使用固定的下采样倍率。
2.  **更轻量的裁判：** 阶段 3 中的 MLLM-as-a-Judge 本身计算量较大，可以探索蒸馏出更小的裁判模型。
3.  **端到端优化：** 探索可学习的压缩模块，与训练过程联合优化，可能比固定的插值更优。

## 7.3. 个人启发与批判
这篇论文给我最大的启发是<strong>“架构减法，训练加法”</strong>的设计哲学。通常我们倾向于通过增加模型容量或数据量来提升性能，而本文反其道而行之，通过大幅减少输入信息（视觉词元），倒逼模型学习更紧凑、更本质的语义表示，并通过复杂的训练策略来弥补信息损失。这为解决大模型落地部署的效率问题提供了一个极具价值的思路。

批判性地看，虽然论文展示了在公开数据集上的 SOTA 结果，但与 GME（使用了私有数据）的对比表明，数据质量依然是决定性能上限的关键因素。本文的 MLLM-as-a-Judge 本质上也是一种数据增强和清洗技术，这再次印证了“数据质量 > 模型规模”的趋势。此外，对于工业界应用，这种三阶段训练的复杂性可能是一个门槛，如何简化训练流程或提供更易用的预训练模型将是未来的关键。