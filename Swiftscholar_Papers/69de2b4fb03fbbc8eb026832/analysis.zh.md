# 1. 论文基本信息

## 1.1. 标题
UniME-V2: MLLM-as-a-Judge for Universal Multimodal Embedding Learning

## 1.2. 作者
Tiancheng Gu, Kaicheng Yang, Kaichen Zhang, Xiang An, Ziyong Feng, Yueyi Zhang, Weidong Cai, Jiankang Deng, Lidong Bing

**研究背景与隶属机构：**
*   **MiroMind AI**
*   **The University of Sydney**
*   **Team LMMs-Lab**
*   **Imperial College London**

## 1.3. 发表期刊/会议
arXiv (预印本)

## 1.4. 发表年份
2025年 (发布于 2025-10-15)

## 1.5. 摘要
通用多模态嵌入模型是各种任务的基础。现有方法通常采用批次内负样本挖掘，通过测量查询-候选对的相似度来实现。然而，这些方法往往难以捕捉候选之间细微的语义差异，且负样本缺乏多样性。此外，嵌入在区分假负样本和困难负样本方面的判别能力有限。在本文中，我们利用多模态大语言模型（MLLM）先进的理解能力来增强表征学习，并提出了一种新颖的通用多模态嵌入（UniME-V2）模型。我们的方法首先通过全局检索构建一个潜在的困难负样本集。然后，我们引入 MLLM-as-a-Judge 机制，利用 MLLM 评估查询-候选对的语义对齐并生成软语义匹配分数。这些分数作为困难负样本挖掘的基础，减轻了假负样本的影响，并能够识别多样化、高质量的困难负样本。此外，语义匹配分数被用作软标签，以缓解刚性的一对一映射约束。通过将相似度矩阵与软语义匹配分数矩阵对齐，模型学习候选之间的语义区别，显著增强了其判别能力。为了进一步提高性能，我们提出了 UniME-V2-Reranker，这是一个在我们挖掘的困难负样本上通过联合成对和列表优化方法训练的重排序模型。我们在 MMEB 基准测试和多个检索任务上进行了全面的实验，证明我们的方法在所有任务中平均取得了最先进的性能。

## 1.6. 原文链接
https://arxiv.org/abs/2510.13515 (预印本)

# 2. 整体概括

## 2.1. 研究背景与动机
**核心问题：**
现有的通用多模态嵌入模型（如 CLIP 及其变体）在训练过程中主要依赖“批次内负样本挖掘”。这种方法通过计算查询与候选之间的相似度来选取负样本，存在两个主要缺陷：
1.  **语义区分能力弱：** 难以捕捉候选样本之间细微的语义差异，导致模型在处理语义相近但实际不匹配的样本时表现不佳。
2.  **负样本质量与多样性不足：** 简单的相似度度量容易筛选出“假负样本”（即实际上与查询相关但被错误标记为负样本的样本），同时也缺乏样本的多样性，限制了模型学习到更丰富的表征。

**现有挑战：**
尽管已有工作（如 VLM2Vec, UniME）尝试引入困难负样本或利用 MLLM 进行蒸馏，但它们往往未能充分利用候选之间的语义差异，且在有效区分困难负样本和假负样本方面仍有不足。

**创新思路：**
本文提出利用多模态大语言模型（MLLM）强大的语义理解能力作为“评判者”，来指导嵌入模型的训练。具体而言，通过 MLLM 生成“软语义匹配分数”，以此挖掘更高质量的困难负样本，并利用这些分数作为软标签来监督模型学习更精细的语义对齐。

## 2.2. 核心贡献/主要发现
1.  **MLLM-as-a-Judge 挖掘机制：** 提出了一种利用 MLLM 评估查询-候选对语义对齐的流程，生成软语义匹配分数，从而指导困难负样本的挖掘，有效过滤假负样本并提高负样本的多样性和质量。
2.  **UniME-V2 模型：** 提出了一种基于 MLLM 判定分布对齐的通用多模态嵌入模型。通过将模型预测的相似度矩阵与 MLLM 生成的软语义分数矩阵进行对齐（使用对称 KL 散度损失），模型能够学习候选之间更细微的语义区别，显著提升了判别能力。
3.  **UniME-V2-Reranker：** 提出了一个重排序模型，结合了成对和列表优化策略，进一步提升了检索精度。
4.  **实验验证：** 在 MMEB 基准测试及多个检索任务（包括短/长字幕检索和组合检索）上，该方法取得了最先进的平均性能，证明了其有效性和鲁棒性。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，读者需要掌握以下核心概念：

*   **多模态嵌入：** 将来自不同模态（如文本、图像）的数据映射到同一高维向量空间的技术。在这个空间中，语义相关的数据应当距离较近。例如，一张“猫”的图片和“一只可爱的猫”这段文本的向量应当具有很高的相似度。
*   <strong>CLIP (Contrastive Language-Image Pre-training)：</strong> 一种开创性的视觉-语言模型，通过在大规模图像-文本对上进行对比学习来学习多模态表征。它通常使用双编码器结构，分别处理图像和文本。
*   **困难负样本：** 在对比学习中，那些与正样本非常相似（容易被模型混淆）但实际上不匹配的样本。挖掘困难负样本对于提升模型的判别能力至关重要。
*   **假负样本：** 在数据集中，某些样本虽然被标记为负样本（即与查询不匹配），但实际上可能包含与查询相关的语义内容。例如，对于查询“红色的车”，一张“红色的跑车”图片如果被标记为负样本可能就是假负样本。直接将假负样本作为负例训练会误导模型。
*   <strong>MLLM (Multimodal Large Language Model)：</strong> 具有视觉和语言理解能力的大语言模型，如 LLaVA, Qwen-VL 等。它们不仅能识别物体，还能理解复杂的指令和场景关系。
*   **KL 散度：** 用于衡量两个概率分布之间差异的指标。在本文中，它用于对齐模型预测的相似度分布和 MLLM 判定的语义分数分布。

## 3.2. 前人工作
作者在文中提到了以下关键前人研究：

*   <strong>CLIP (Radford et al. 2021)：</strong> 奠基性工作，证明了大规模对比学习的有效性。但其存在 77 token 限制、双编码器限制以及词袋行为等缺陷。
*   <strong>E5-V (Jiang et al. 2024)：</strong> 采用单模态对比学习，训练 MLLM 的语言组件以对齐跨模态表征空间。
*   <strong>VLM2Vec (Jiang et al. 2025)：</strong> 提出了 MMEB 基准测试，并提出了将预训练视觉模型转换为嵌入模型的对比学习框架。
*   <strong>QQMM (Xue, Li, and Liu 2025a)：</strong> 分析了 InfoNCE 损失的梯度，提出放大与困难负样本相关的梯度。
*   <strong>UniME (Gu et al. 2025a)：</strong> 提作者的前序工作，利用基于 LLM 的教师模型改进语言嵌入，并采用了困难负样本采样策略。

**补充背景知识：InfoNCE Loss**
虽然本文未直接展开 InfoNCE 的公式，但它是对比学习的核心。为了理解本文的改进，有必要了解其标准形式。InfoNCE 损失旨在最大化正样本对的相似度，同时最小化负样本对的相似度。其公式通常表示为：
$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{K} \exp(\text{sim}(z_i, z_k)/\tau)}
$$
其中 $z_i$ 和 $z_j$ 是正样本对，$z_k$ 包含正样本和负样本，$\tau$ 是温度参数。本文提出的损失函数是对这一范式的扩展，引入了软标签。

## 3.3. 技术演进
该领域从 CLIP 的简单双编码器对比学习，发展到利用 MLLM 的强大理解能力进行表征学习（如 E5-V, VLM2Vec）。随后，研究重点转向如何更有效地挖掘困难负样本（如 QQMM, UniME）。本文的工作进一步演进，不仅利用 MLLM 进行特征提取，更将其作为“评判者”来生成高质量的软标签和挖掘困难负样本，实现了从硬标签训练到软标签分布对齐的跨越。

## 3.4. 差异化分析
与现有的 VLM2Vec 和 UniME 相比，本文的核心区别在于：
1.  **引入 MLLM-as-a-Judge：** 不仅仅是使用 MLLM 作为编码器，而是利用其生成语义匹配分数，用于更精细的困难负样本挖掘。
2.  **软标签分布对齐：** 传统的对比学习通常隐含地假设正样本相似度为 1，负样本为 0（硬标签）。本文利用 MLLM 生成的连续分数作为软标签，通过 KL 散度损失对齐模型预测分布与软标签分布，从而捕捉更细微的语义差异。

# 4. 方法论

## 4.1. 方法原理
UniME-V2 的核心思想是利用 MLLM 强大的语义理解能力来解决传统对比学习中负样本质量不高和标签过于刚性的问题。
1.  **困难负样本挖掘：** 首先通过全局检索（使用现成的 VLM2Vec）构建一个潜在的困难负样本集。然后，利用 MLLM（作为 Judge）评估查询与这些潜在负样本的语义对齐程度，生成一个 0 到 1 之间的软分数。基于这个分数，过滤掉假负样本（分数过高的），并选出高质量的困难负样本。
2.  **分布对齐训练：** 在训练阶段，不再仅仅区分“正”与“负”，而是让模型预测的相似度分布去拟合 MLLM 生成的语义分数分布。这使得模型能够学习到，某些负样本可能比其他负样本“更接近”查询，从而学习到更精细的语义边界。

    下图（原文 Figure 1）展示了 UniME-V2 与传统方法的对比，直观地呈现了 MLLM-as-a-Judge 机制在困难负样本挖掘和软标签生成中的作用。

    ![Figure 1: Comparison between previous works and UniME-V2. UniME-V2 exploits the understanding capabilities of MLLMs for hard negatives mining and generates a soft semantic matching score to supervise the model in learning the semantic difference among candidates.](images/1.jpg)
    *该图像是示意图，展示了UniME-V2模型与传统方法的对比。上半部分为传统方法，展示了查询及候选框架和训练目标；下半部分为UniME-V2，展示了MLLM-as-a-Judge机制进行困难负样本挖掘的流程和语义匹配分数的生成。通过这些机制，UniME-V2提升了模型的语义区分能力。*

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 任务定义
首先，我们需要明确模型的目标。给定一个查询 $q$ 和一个候选集 $\Omega_c = \{c_1, c_2, \ldots, c_n\}$（其中候选可能包含图像、文本或图文交错数据），通用嵌入模型 $\Phi_{emb}$ 的目标是编码查询和候选，检索出最相关的前 $k$ 个候选 $\Omega_k$：
$$
\Omega_k = \Phi_{emb}(q, \Omega_c)
$$
为了进一步提高性能，引入一个重排序模型 $\Phi_{rank}$，对初步检索结果进行精排，得到最终输出 $\hat{\Omega}_k$：
$$
\hat{\Omega}_k = \Phi_{rank}(q, \Omega_k)
$$

### 4.2.2. MLLM-as-a-Judge for Hard Negatives Mining
这一步的核心是利用 MLLM 找到高质量的困难负样本。

**步骤 1：构建潜在困难负样本集**
为了从全局样本中提取更高质量的困难负样本，作者首先使用 VLM2Vec 生成查询和候选的嵌入，并检索出相似度最高的前 50 个候选。为了处理假负样本并增加多样性，应用了一个基于相似度分数的阈值 $\delta$，选出相似度分数低于 $\delta$ 的前 50 个候选作为潜在困难负样本集 $\Omega_p$：
$$
\Omega_p = \text{Rank}_{50}(\{x_1, \ldots, x_n\}), \text{where } x_i < \delta
$$
其中，$x_i$ 是通过 VLM2Vec 模型计算的查询 $q$ 和候选 $\hat{\Omega}_c$ 之间的相似度分数。

**步骤 2：计算语义匹配分数**
构建好 $\Omega_p$ 后，利用 MLLM 作为评判者，计算其中每个查询-候选对的语义匹配分数。MLLM 会根据特定的指令输出 "Yes" 或 "No"。基于输出词元的 logits，计算语义匹配分数 $S = \{s_1, s_2, \ldots, s_m\}$。对于第 $i$ 个样本，其分数 $s_i$ 计算如下：
$$
s_i = \frac{e_y^i}{e_y^i + e_n^i}
$$
其中，$e_y^i$ 是 "Yes" 词元的 logit 值，$e_n^i$ 是 "No" 词元的 logit 值。这个分数 $S \in \mathbb{R}^{n_q \times 50}$ ($n_q$ 为查询数量) 有效捕捉了查询与候选之间语义对齐的程度。

**步骤 3：困难负样本采样**
基于语义匹配分数 $S$ 对候选进行精炼。为了排除假负样本，如果候选的分数超过阈值 $\alpha = \sigma_{q, c_t} - \beta$（其中 $c_t$ 是正样本，$\beta$ 是控制阈值边界的超参数，设为 0.01），则将其排除。为了保持多样性，采用间隔为 5 的循环采样策略。最终，对于每个查询 $q$，得到困难负样本集 $\Omega_h = \{c_1, . . ., c_k\}$ 及其对应的语义匹配分数 $S_h = \{s_{q, c_1}, . . ., s_{q, c_k}\}$。

下图（原文 Figure 2）展示了 MLLM-as-a-Judge 机制如何评估查询与候选对的语义对齐，并生成软语义匹配分数。

![该图像是一个示意图，展示了UniME-V2模型在多模态嵌入学习中的过程。图中包含查询、潜在的困难负样本以及候选样本的结构，体现了MLLM-as-a-Judge机制如何用于评估查询与候选对的语义对齐，并生成软语义匹配分数，助力挖掘高质量的困难负样本。](images/2.jpg)
*该图像是一个示意图，展示了UniME-V2模型在多模态嵌入学习中的过程。图中包含查询、潜在的困难负样本以及候选样本的结构，体现了MLLM-as-a-Judge机制如何用于评估查询与候选对的语义对齐，并生成软语义匹配分数，助力挖掘高质量的困难负样本。*

### 4.2.3. MLLM Judgment Based Training Framework
这一步是模型训练的核心，旨在利用 MLLM 生成的软分数来指导模型学习。

**步骤 1：UniME-V2 的分布对齐**
给定查询 $q$ 和候选集 $\Omega_c = \{c_t, c_1, . . ., c_k\}$（包含正样本 $c_t$ 和 $k$ 个困难负样本），将它们输入 MLLM 并提取最后一个词元的嵌入，得到查询嵌入 $e_q$ 和候选嵌入 $E_c = \{e_c^+, e_{c_1}^-, . ., e_{c_k}^-\}$。

首先，计算查询嵌入 $e_q$ 与所有候选嵌入 $E_c$ 之间的关系分数矩阵 $\mathbb{P}(e_q, E_c)$。这里使用的是标准的对比学习 softmax 形式：
$$
\mathbb{P}(e_q, E_c) = \frac{\exp(\cos(e_q, e_c^+)/\tau)}{\exp(\cos(e_q, e_c^+)/\tau) + \sum_{i=1}^{k} \exp(\cos(e_q, e_{c_i}^-)/\tau)}
$$
其中，$\cos(\cdot)$ 表示余弦相似度，$\tau$ 是温度参数。这个矩阵 $\mathbb{P}$ 代表了模型当前的预测分布。

接下来，基于 MLLM 判定得到的语义匹配分数 $S_c = \{s_{q, c_t}, s_{q, c_1}, . . ., s_{q, c_k}\}$，计算目标语义分数矩阵 $\mathbb{Q}(S_c)$。这个矩阵代表了 MLLM 认为的理想分布：
$$
\mathbb{Q}(S_c) = \frac{\exp(s_{q, c_t}/\tau)}{\exp(s_{q{,} c_t}/\tau) + \sum_{i=1}^{k} \exp(s_{q, c_i}/\tau)}
$$

为了增强学习的鲁棒性并确保矩阵对称性，作者采用对称 KL 散度（即 Jensen-Shannon 散度的一种形式）作为最终的损失函数 $\mathcal{L}$：
$$
\mathcal{L} = \frac{1}{2} \left( \frac{1}{N} \sum_{i=1}^{N} \text{KL}(\mathbb{P}(e_i, E_c) || \mathbb{Q}(S_c)) + \frac{1}{N} \sum_{i=1}^{N} \text{KL}(\mathbb{Q}(S_c) || \mathbb{P}(e_i, E_c)) \right)
$$
这个损失函数强制模型预测的相似度分布 $\mathbb{P}$ 尽可能接近 MLLM 判定的语义分数分布 $\mathbb{Q}$，从而使模型学习到更精细的语义区别。

下图（原文 Figure 3）展示了基于 MLLM 判定的训练框架架构，包括 UniME-V2 使用软语义匹配分数作为监督信号，以及 UniME-V2-Reranker 的联合优化。

![Figure 3: The architecture of the MLLM Judgment Based Training Framework. UniME-V2 uses soft semantic matching scores as supervised signals to enhance semantic distinction learning between candidates. UniME-V2-Reranker employs joint pairwise and listwise optimization to enhance reranking performance.](images/3.jpg)

**步骤 2：UniME-V2-Reranker 的联合优化**
为了进一步提升检索性能，作者训练了一个重排序模型。该模型结合了成对和列表优化。

*   **成对训练：** 对于每个查询 $q$，构建两个对：$(q, c_t)$（正样本对）和 $(q, c_h)$（最困难负样本对）。指示模型对正样本输出 "YES"，对负样本输出 "NO"。成对损失 $\mathcal{L}_{pair}$ 使用交叉熵损失计算：
    $$
    \mathcal{L}_{pair} = \mathcal{L}_{ce}(\text{YES}, \eta(q, c_t)) + \mathcal{L}_{ce}(\text{NO}, \eta(q, c_h))
    $$
    其中 $\eta$ 表示 UniME-V2-Reranker 的自回归输出过程。

*   **列表训练：** 基于语义匹配分数，从困难负样本候选中选择前 $x$ 个 $\{c_1, . . ., c_x\}$，将目标候选 $c_t$ 随机插入其中，并记录其位置索引 $I_{c_t}$。指示模型输出正确位置的索引。列表损失 $\mathcal{L}_{list}$ 定义为：
    $$
    \mathcal{L}_{list} = \mathcal{L}_{ce}(I_{c_t}, \eta(q, c_t, \{c_1, . . ., c_x\}))
    $$

最终的重排序器损失函数是两者的加和：
$$
\mathcal{L} = \mathcal{L}_{pair} + \mathcal{L}_{list}
$$

# 5. 实验设置

## 5.1. 数据集
实验使用了以下数据集：
*   **MMEB Benchmark：** 包含 36 个数据集，涵盖四个元任务：分类、视觉问答、检索和视觉定位。训练使用了其中的 20 个分布内数据集，共 662k 训练对。
*   **检索任务数据集：**
    *   **短字幕检索：** Flickr30K, MS-COCO。
    *   **长字幕检索：** ShareGPT4V, Urban1K。
    *   **组合检索：** SugarCrepe（用于测试模型对组合性语义的理解能力）。

        **选择原因：** 这些数据集涵盖了从简单到复杂的各种多模态理解任务，能够全面评估模型的泛化能力和判别能力。特别是 SugarCrepe，专门用于测试模型区分细微语义差异的能力，非常适合验证本文方法的有效性。

## 5.2. 评估指标
论文主要使用了以下评估指标：

1.  <strong>Precision (精确度)</strong>
    *   **概念定义：** 在检索任务中，精确度通常指在返回的前 $k$ 个结果中，正确匹配的结果所占的比例。它衡量的是检索结果的准确性。
    *   **数学公式：** 对于查询 $q$，返回结果集 $R$，真实相关集 $G$，Top-$k$ 精确度定义为：
        $$
        \text{Precision}@k = \frac{|R \cap G|}{k}
        $$
    *   **符号解释：** $|R \cap G|$ 表示返回结果中与真实结果交集的数量（即正确命中的数量），$k$ 为返回结果的总数。

2.  <strong>Recall@1 (召回率@1)</strong>
    *   **概念定义：** 在返回的 Top-1 结果中，是否包含正确答案。如果是则为 1，否则为 0。它衡量的是模型把最相关的结果排在第一位的概率。
    *   **数学公式：**
        $$
        \text{Recall}@1 = \begin{cases} 1, & \text{if } \text{rank}(\text{correct\_item}) = 1 \\ 0, & \text{otherwise} \end{cases}
        $$
    *   **符号解释：** $\text{rank}(\text{correct\_item})$ 表示正确答案在排序结果中的位置。

## 5.3. 对比基线
论文将 UniME-V2 与以下基线模型进行了比较：
*   **CLIP 系列：** CLIP (ViT-L, ViT-BigG/14), OpenCLIP, EVA-CLIP。
*   **基于 MLLM 的嵌入模型：** E5-V, VLM2Vec, QQMM, UniME, LLaVE。
    这些基线代表了当前多模态嵌入领域的不同技术路线，具有很强的代表性。

# 6. 实验结果与分析

## 6.1. 核心结果分析
实验结果表明，UniME-V2 在多个基准测试上均取得了最先进的性能。

**多模态检索：**
在 MMEB（Fine-tuning）基准测试中，UniME-V2 (Qwen2-VL-7B) 达到了 68.0 的平均分，优于 VLM2Vec (65.8) 和 UniME (67.4)。特别是在分布外（OOD）数据集上，UniME-V2 达到了 63.0 分，显著优于所有先前方法，显示了其强大的泛化能力。

**短/长字幕检索：**
在长字幕检索任务（ShareGPT4V, Urban1K）上，UniME-V2 取得了显著提升，这得益于其增强的判别能力和对丰富语义内容的捕捉。下图（原文 Figure 4）展示了 EVA-CLIP-8B 与 UniME-V2 在表示分布上的对比，直观地说明了 UniME-V2 如何更好地减少模态差异。

![Figure 4: Comparison of representation distributions between EVA-CLIP-8B and UniME-V2 (LLaVA-OneVision-7B).](images/4.jpg)

**组合检索：**
在 SugarCrepe 数据集上，UniME-V2 一致优于基线模型。例如，使用 Qwen2-VL-2B 时，相比 UniME 有 5.3% - 6.0% 的性能提升。这证明了模型在区分困难负样本方面的优势。

**重排序比较：**
UniME-V2-Reranker 在使用更少数据（0.6M vs 1.1M）的情况下，性能优于 LamRA 重排序器，特别是在组合理解检索任务上优势明显。

以下是原文 Table 1 的结果，展示了在 MMEB 基准测试上的详细对比：

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th rowspan="2">#Parameters</th>
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
<td colspan="9">Zero-shot on MMEB</td>
</tr>
<tr>
<td>CLIP (ViT-L)</td>
<td>0.4B</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.1</td>
<td>38.7</td>
<td>39.2</td>
</tr>
<tr>
<td>OpenCLIP (ViT-L)</td>
<td>0.4B</td>
<td>41.5</td>
<td>6.9</td>
<td>44.6</td>
<td>53.5</td>
<td>32.8</td>
<td>36.0</td>
<td>36.6</td>
</tr>
<tr>
<td>Magiclens (ViT-L)</td>
<td>0.4B</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td>31.0</td>
<td>23.7</td>
<td>27.1</td>
</tr>
<tr>
<td>SigLIP (So/14)</td>
<td>0.9B</td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>32.3</td>
<td>38.0</td>
<td>35.0</td>
</tr>
<tr>
<td>BLIP2 (ViT-L)</td>
<td>1.2B</td>
<td>27.0</td>
<td>4.2</td>
<td>33.9</td>
<td>47.0</td>
<td>25.3</td>
<td>25.1</td>
<td>28.0</td>
</tr>
<tr>
<td>CLIP (ViT-BigG/14)</td>
<td>2.5B</td>
<td>52.3</td>
<td>14.0</td>
<td>50.5</td>
<td>60.3</td>
<td>38.9</td>
<td>45.8</td>
<td>44.3</td>
</tr>
<tr>
<td>EVA-CLIP</td>
<td>8B</td>
<td>56.0</td>
<td>10.4</td>
<td>49.2</td>
<td>58.9</td>
<td>38.1</td>
<td>45.6</td>
<td>43.7</td>
</tr>
<tr>
<td>E5-V (Phi3.5-V)</td>
<td>4.2B</td>
<td>39.1</td>
<td>9.6</td>
<td>38.0</td>
<td>57.6</td>
<td>33.1</td>
<td>31.9</td>
<td>36.1</td>
</tr>
<tr>
<td>E5-V (LLaVA-1.6)</td>
<td>7B</td>
<td>39.7</td>
<td>10.8</td>
<td>39.4</td>
<td>60.2</td>
<td>34.2</td>
<td>33.4</td>
<td>37.5</td>
</tr>
<tr>
<td colspan="9">Fine-tuning on MMEB</td>
</tr>
<tr>
<td>CLIP (ViT-L)</td>
<td>0.4B</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>47.6</td>
</tr>
<tr>
<td>VLM2Vec (Qwen2-VL)</td>
<td>2B</td>
<td>59.0</td>
<td>49.4</td>
<td>65.4</td>
<td>73.4</td>
<td>66.0</td>
<td>52.6</td>
<td>60.1</td>
</tr>
<tr>
<td>VLM2Vec (Qwen2-VL)</td>
<td>7B</td>
<td>62.6</td>
<td>57.8</td>
<td>69.9</td>
<td>81.7</td>
<td>72.2</td>
<td>57.8</td>
<td>65.8</td>
</tr>
<tr>
<td>LLaVE (LLaVA-OV)</td>
<td>7B</td>
<td>65.7</td>
<td>65.4</td>
<td>70.9</td>
<td>91.9</td>
<td>75.0</td>
<td>64.4</td>
<td>70.3</td>
</tr>
<tr>
<td>QQMM (LLaLa-OV)</td>
<td>7B</td>
<td>66.8</td>
<td>66.8</td>
<td>70.5</td>
<td>90.4</td>
<td>74.7</td>
<td>65.6</td>
<td>70.7</td>
</tr>
<tr>
<td>UniME (Qwen2-VL)</td>
<td>2B</td>
<td>59.0</td>
<td>53.4</td>
<td>64.9</td>
<td>69.6</td>
<td>65.5</td>
<td>54.6</td>
<td>60.6</td>
</tr>
<tr>
<td>UniME (Qwen2-VL)</td>
<td>7B</td>
<td>64.7</td>
<td>59.0</td>
<td>71.6</td>
<td>82.7</td>
<td>72.2</td>
<td>61.4</td>
<td>67.4</td>
</tr>
<tr>
<td>UniME (LLaVA-OV)</td>
<td>7B</td>
<td>66.8</td>
<td>66.6</td>
<td>70.5</td>
<td>90.9</td>
<td>74.6</td>
<td>65.8</td>
<td>70.7</td>
</tr>
<tr>
<td>UniME-V2(Qwen2-VL)</td>
<td>2B</td>
<td>62.1</td>
<td>56.3</td>
<td>68.0</td>
<td>72.7</td>
<td>67.4</td>
<td>58.9</td>
<td>63.6</td>
</tr>
<tr>
<td>UniME-V2(Qwen2-VL)</td>
<td>7B</td>
<td>64.0</td>
<td>60.1</td>
<td>73.1</td>
<td>82.8</td>
<td>72.0</td>
<td>63.0</td>
<td>68.0</td>
</tr>
<tr>
<td>UniME-V2(LLaVA-OV)</td>
<td>7B</td>
<td>65.3</td>
<td>67.6</td>
<td>72.9</td>
<td>90.2</td>
<td>74.8</td>
<td>66.7</td>
<td>71.2</td>
</tr>
</tbody>
</table>

以下是原文 Table 2 的结果，展示了在短/长字幕及组合检索任务上的详细对比：

<table>
<thead>
<tr>
<th rowspan="3">Models</th>
<th rowspan="3">#Parameters</th>
<th colspan="4">Short Caption</th>
<th colspan="4">Long Caption</th>
<th colspan="3">Compositional</th>
</tr>
<tr>
<th colspan="2">Flickr30K</th>
<th colspan="2">COCO</th>
<th colspan="2">ShareGPT4V</th>
<th colspan="2">Urban1K</th>
<th colspan="3">SugarCrepe</th>
</tr>
<tr>
<th>I2T</th>
<th>T2I</th>
<th>I2T</th>
<th>T2I</th>
<th>I2T</th>
<th>T2I</th>
<th>I2T</th>
<th>T2I</th>
<th>Replace</th>
<th>Swap</th>
<th>Add</th>
</tr>
</thead>
<tbody>
<tr>
<td>OpenCLIP (ViT-L)</td>
<td>0.4B</td>
<td>67.3</td>
<td>87.2</td>
<td>37.0</td>
<td>58.1</td>
<td>81.8</td>
<td>84.0</td>
<td>47.0</td>
<td>47.0</td>
<td>79.5</td>
<td>62.7</td>
<td>74.9</td>
</tr>
<tr>
<td>CLIP (ViT-BigG/14)</td>
<td>2.5B</td>
<td>79.5</td>
<td>92.9</td>
<td>51.3</td>
<td>67.3</td>
<td>90.1</td>
<td>93.6</td>
<td>77.8</td>
<td>80.7</td>
<td>86.5</td>
<td>68.9</td>
<td>88.4</td>
</tr>
<tr>
<td>EVA-CLIP</td>
<td>8B</td>
<td>80.3</td>
<td>94.5</td>
<td>52.0</td>
<td>70.1</td>
<td>93.1</td>
<td>91.2</td>
<td>80.4</td>
<td>77.8</td>
<td>85.9</td>
<td>70.3</td>
<td>86.7</td>
</tr>
<tr>
<td>E5-V (Phi3.5-V)</td>
<td>4.2B</td>
<td>72.2</td>
<td>79.6</td>
<td>44.7</td>
<td>53.4</td>
<td>86.0</td>
<td>88.5</td>
<td>83.8</td>
<td>83.6</td>
<td>88.2</td>
<td>66.6</td>
<td>75.3</td>
</tr>
<tr>
<td>E5-V (LLaVA-1.6)</td>
<td>7B</td>
<td>77.3</td>
<td>85.7</td>
<td>49.1</td>
<td>57.6</td>
<td>85.1</td>
<td>82.1</td>
<td>88.9</td>
<td>83.2</td>
<td>86.3</td>
<td>68.7</td>
<td>66.9</td>
</tr>
<tr>
<td>VLM2Vec (Qwen2-VL)</td>
<td>2B</td>
<td>69.3</td>
<td>89.6</td>
<td>40.0</td>
<td>62.5</td>
<td>78.1</td>
<td>88.2</td>
<td>78.7</td>
<td>83.9</td>
<td>67.2</td>
<td>46.5</td>
<td>66.4</td>
</tr>
<tr>
<td>VLM2Vec (Qwen2-VL)</td>
<td>7B</td>
<td>80.0</td>
<td>94.2</td>
<td>49.2</td>
<td>68.5</td>
<td>78.5</td>
<td>90.4</td>
<td>94.0</td>
<td>94.2</td>
<td>70.0</td>
<td>51.7</td>
<td>72.2</td>
</tr>
<tr>
<td>UniME (Qwen2-VL)</td>
<td>2B</td>
<td>74.9</td>
<td>90.6</td>
<td>44.0</td>
<td>63.5</td>
<td>83.6</td>
<td>88.6</td>
<td>83.3</td>
<td>83.2</td>
<td>65.6</td>
<td>45.2</td>
<td>65.7</td>
</tr>
<tr>
<td>UniME (Qwen2-VL)</td>
<td>7B</td>
<td>80.8</td>
<td>92.7</td>
<td>50.9</td>
<td>69.8</td>
<td>86.5</td>
<td>93.8</td>
<td>95.3</td>
<td>94.0</td>
<td>68.8</td>
<td>53.0</td>
<td>69.8</td>
</tr>
<tr>
<td>UniME (LLaVA-OV)</td>
<td>7B</td>
<td>83.3</td>
<td>94.4</td>
<td>54.8</td>
<td>74.0</td>
<td>93.9</td>
<td>89.3</td>
<td>94.3</td>
<td>95.5</td>
<td>80.5</td>
<td>65.5</td>
<td>82.2</td>
</tr>
<tr>
<td>UniME-V2 (Qwen2-VL)</td>
<td>2B</td>
<td>79.8</td>
<td>90.7</td>
<td>49.1</td>
<td>65.1</td>
<td>91.8</td>
<td>94.5</td>
<td>95.6</td>
<td>95.1</td>
<td>70.9</td>
<td>51.2</td>
<td>70.2</td>
</tr>
<tr>
<td>UniME-V2 (Qwen2-VL)</td>
<td>7B</td>
<td>83.8</td>
<td>93.5</td>
<td>57.3</td>
<td>70.3</td>
<td>94.3</td>
<td>95.1</td>
<td>97.2</td>
<td>96.3</td>
<td>77.8</td>
<td>62.2</td>
<td>79.0</td>
</tr>
<tr>
<td>UniME-V2 (LLaVA-OV)</td>
<td>7B</td>
<td>85.7</td>
<td>94.7</td>
<td>60.9</td>
<td>74.1</td>
<td>95.1</td>
<td>96.4</td>
<td>96.3</td>
<td>97.5</td>
<td>88.6</td>
<td>73.7</td>
<td>90.5</td>
</tr>
</tbody>
</table>

## 6.2. 消融实验/参数分析
作者进行了详细的消融实验来验证各组件的有效性。

**组件消融：**
Table 4 显示了移除“困难负样本挖掘”和“软分数”组件后的性能下降。结果表明，仅使用困难负样本挖掘（不使用软分数）相比基线（VLM2Vec）有显著提升，而加入基于 MLLM 判定的软分数训练框架后，性能进一步提升。这证明了两个组件的有效性。

**MLLM 评判者选择：**
Table 5 比较了使用不同 MLLM（Qwen2.5-VL-7B, InternVL3-8B, InternVL3-14B）作为评判者时的性能。结果显示，Qwen2.5-VL-7B 生成的语义分数质量最高，带来的下游性能也最好。这表明评判者的理解能力直接影响最终效果。

**困难负样本数量：**
Table 6 分析了困难负样本数量 $k$ 的影响。当 $k$ 从 4 增加到 8 时，各项指标均提升；但增加到 10 时，性能略有下降。这可能是因为引入了过于简单的负样本，反而干扰了判别学习。

**温度参数：**
Table 10 分析了损失函数中温度 $\tau$ 的影响。实验发现 $\tau=0.02$ 时性能最优。

下图（原文 Figure 5）展示了定性结果，包括检索和重排序的结果。

![Figure 5: Qualitative examples. We present the retrieval and reranking results of our method across different tasks.](images/5.jpg)
*该图像是示意图，展示了不同任务中我们方法的检索和重排序结果。图中包含分类、视觉问答和视觉定位等任务的示例，以及对应的指示和语义标签。*

下图（原文 Figure 6）展示了经过困难负样本挖掘流程处理后的查询和候选示例。

![Figure 6: Qualitative examples. We present examples showing queries and their corresponding hard negative candidates processed after our hard negative mining pipeline.](images/6.jpg)
*该图像是插图，展示了查询和对应的经过我们硬负样本挖掘流程处理后的候选图像。在每个查询下，列出了与之匹配的候选图像及其相似度分数，展示了不同场景的多模态嵌入学习实例。*

下图（原文 Figure 7）展示了额外的检索和重排序结果。

![Figure 7: Qualitative examples. We present the additional retrieval and reranking results of our method across different tasks.](images/7.jpg)
*该图像是示意图，展示了多模态嵌入学习中的不同任务示例，包括分类、视觉问答和视觉定位。每个任务都有相应的指导说明和示例答案，体现了如何通过示例图像得到分类结果、回答问题以及识别目标对象。*

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 UniME-V2，一种新颖的通用多模态嵌入模型。其核心创新在于引入了 MLLM-as-a-Judge 机制，利用 MLLM 强大的语义理解能力来生成软语义匹配分数。这些分数不仅用于挖掘高质量、多样化的困难负样本，还作为软标签指导模型进行分布对齐训练。此外，结合 UniME-V2-Reranker，该方法在 MMEB 基准测试和多个检索任务上取得了最先进的性能，证明了利用 MLLM 作为训练信号源的有效性。

## 7.2. 局限性与未来工作
**局限性：**
*   **计算开销：** 引入 MLLM-as-a-Judge 机制意味着在训练前需要额外的 MLLM 推理来生成语义分数，这增加了训练 pipeline 的复杂度和计算成本。
*   **依赖评判者质量：** 方法的性能高度依赖于作为评判者的 MLLM 的理解能力。如果评判者本身存在偏见或理解错误，可能会误导嵌入模型的训练。

**未来工作：**
*   探索更高效的分数生成方法，例如蒸馏或使用更小的模型作为评判者。
*   将该方法扩展到更多模态（如音频、视频）或更复杂的任务中。

## 7.3. 个人启发与批判
**启发：**
*   **软标签的价值：** 本文展示了在对比学习中引入细粒度的软标签（而非简单的 0/1 硬标签）可以显著提升模型的判别能力。这对于其他需要精细区分的任务具有借鉴意义。
*   **利用大模型的知识：** 利用大模型（MLLM）的生成能力作为“教师”来指导“学生”（嵌入模型）的训练，是一条非常有前景的技术路线，特别是在数据标注稀缺或模糊的场景下。

**批判性思考：**
*   **效率权衡：** 虽然性能提升显著，但计算开销的增加是值得关注的。在实际工业应用中，可能需要评估这种性能提升是否足以抵消额外的 MLLM 推理成本。
*   **泛化性：** 该方法在 MMEB 和特定检索任务上表现优异，但在其他类型的任务（如生成、推理）上的效果如何，尚需进一步验证。
*   **分数的可靠性：** MLLM 输出的 "Yes/No" logits 转化为分数的公式 $s_i = \frac{e_y^i}{e_y^i + e_n^i}$ 虽然简单有效，但其物理意义和最优性是否可以进一步探讨？例如，是否可以直接利用 MLLM 输出的概率值？