# 1. 论文基本信息

## 1.1. 标题
MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval (MegaPairs：面向通用多模态检索的大规模数据合成)

## 1.2. 作者
Junjie Zhou, Zheng Liu, Ze Liu, Shitao Xiao, Yueze Wang, Bo Zhao, Chen Jason Zhang, Defu Lian, Yongping Xiong

## 1.3. 发表期刊/会议
arXiv (预印本)

## 1.4. 发表年份
2024年

## 1.5. 摘要
尽管多模态检索的需求迅速增长，但该领域的进展严重受限于训练数据的匮乏。本文介绍了 MegaPairs，一种新颖的数据合成方法，它利用视觉语言模型（VLMs）和开放域图像，生成大规模的合成数据集。实证分析表明，MegaPairs 能够生成高质量的数据，使得多模态检索器在仅使用现有数据集 70 倍少的数据进行训练的情况下，显著优于基线模型。此外，由于 MegaPairs 仅依赖于通用图像语料库和开源 VLMs，因此易于扩展，能够持续提升检索性能。在此阶段，我们生成了超过 2600 万个训练实例，并使用该数据训练了多个不同规模的模型。这些新模型在 4 个流行的组合图像检索（CIR）基准测试中实现了最先进的零样本性能，并在 MMEB 提供的 36 个数据集上取得了最高的整体性能。它们在经过额外的下游微调后也显示出显著的性能提升。我们将发布生成的数据集、训练良好的模型以及数据合成管道，以促进该领域的未来发展。

## 1.6. 原文链接
https://arxiv.org/abs/2412.14475

# 2. 整体概括

## 2.1. 研究背景与动机
多模态检索旨在满足人们跨不同数据模态（特别是文本和图像）的信息需求，广泛应用于图像搜索、视觉问答（VQA）和检索增强生成（RAG）等场景。为了支持各种任务和领域，开发通用的多模态检索器至关重要。

现有的多模态检索器主要基于预训练的视觉语言模型（如 CLIP、ALIGN、SigLIP），这些模型通过文本-图像匹配任务进行预训练。虽然它们具备了初步的文本到图像检索能力，但在组合图像检索（CIR）和多模态文档检索等其他任务上表现不足。为了增强多任务能力，指令微调成为了一种流行策略，即利用包含多种检索指令的数据对预训练模型进行微调。

然而，当前多模态检索的指令微调数据极其稀缺。虽然 MagicLens 等工作尝试通过挖掘同一网页内共存的图像来合成开放式的搜索指令，但仍面临显著限制：
1.  **可扩展性**：互联网上包含多张图像的网页比例很小。
2.  **质量**：许多共存的图像要么不相关，要么是近似重复的。
3.  **多样性**：相关的图像通常表现出单调的关系（如同一物体的不同角度）。
4.  **可用性**：大规模数据集通常被实验室私有持有，不对外开放。

    因此，本文的切入点是提出一种不依赖网页内图像共存关系，而是能够从大规模开放域图像语料库中挖掘多样化、高质量图像对，并自动生成检索指令的数据合成方法。

## 2.2. 核心贡献/主要发现
1.  **MegaPairs 方法**：提出了一种新颖的大规模多模态指令数据合成方法。该方法通过构建异构的 KNN（K-Nearest Neighbors）三元组，利用三种不同的相似度模型（CLIP 视觉编码器、DINO 视觉编码器、CLIP 文本编码器）从开放域图像中采样相关的图像对。
2.  **自动化标注**：利用开源的多模态大语言模型（MLLM，如 InternVL2-26B）和大语言模型（LLM，如 Llama-3-8B）自动生成描述图像对关系的详细文本指令。
3.  **大规模数据集**：生成了包含超过 2600 万个训练实例的大规模合成数据集。实验证明，仅使用该数据集的 50 万个样本（不到 MagicLens 数据集的 2%），训练效果就优于使用 3670 万个 MagicLens 样本的模型，证明了数据的高质量。
4.  **MMRet 模型**：基于 MegaPairs 数据集训练了 MMRet 系列模型（包括 CLIP-based 和 MLLM-based）。这些模型在 4 个 CIR 基准测试和 MMEB 的 36 个数据集上实现了最先进的零样本性能，并在下游微调后表现出强大的泛化能力。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，需要掌握以下核心概念：

*   **多模态检索**：指查询或候选包含图像和文本两种模态的检索任务。例如，给定一张图片和一段文本（如“红色的鞋子”），去检索数据库中符合该文本描述的图片。
*   **组合图像检索**：是多模态检索的一种特定形式，通常指查询由一个参考图像和一个修饰文本组成，目标是从图库中找到符合修饰文本修改后的图像。例如，查询是一张“蓝色连衣裙”的图片，文本是“换成红色”，目标是检索“红色连衣裙”的图片。
*   **指令微调**：一种模型训练策略，通过在训练数据中加入自然语言指令（如“检索一张与这张图片相似但背景是室外的图片”），使模型能够理解并执行特定的任务指令，从而提高模型的泛化能力和多任务适应性。
*   **视觉语言模型**：能够同时处理和理解图像与文本输入的深度学习模型，通常通过对比学习将图像和文本映射到同一特征空间。
*   **困难负样本**：在训练检索模型时，与查询样本相似度较高但并非正确结果的样本。使用困难负样本进行训练可以帮助模型学习更细微的特征差异，从而提高检索精度。

## 3.2. 前人工作
*   **CLIP (Radford et al., 2021)**：通过在大规模图像-文本对上进行对比学习，学习了强大的视觉和语言表示。它是现代多模态检索模型的基石。
*   **MagicLens (Zhang et al., 2024)**：此前的一项重要工作，通过挖掘同一网页内共存的图像来构建大规模的开放式搜索指令数据集。虽然取得了进展，但受限于网页多图共存的自然稀缺性。
*   **UniIR (Wei et al., 2024)**：一种通用的多模态信息检索器，旨在统一处理各种检索任务，展示了指令微调在多模态嵌入中的有效性。

## 3.3. 技术演进
多模态检索技术经历了从简单的跨模态匹配（如 CLIP）到更复杂的组合检索（如 CIR）的演变。早期模型主要依赖预训练的表示，缺乏对复杂指令的理解。随着指令微调在 NLP 领域的成功，这一技术被引入多模态领域。然而，数据瓶颈始终是限制多模态检索发展的关键因素。数据合成技术因此成为研究热点，从早期的规则合成发展到利用生成模型（如 VLMs）进行自动化合成。

## 3.4. 差异化分析
与 MagicLens 等前人工作相比，本文的核心区别在于数据来源和挖掘策略：
*   **MagicLens**：依赖于网页的 HTML 结构，仅挖掘同一页面内的图像对。这限制了数据的规模和多样性。
*   **MegaPairs**：不依赖网页结构，而是利用大规模开放域图像语料库（如 Datacomp），通过多种相似度模型（视觉语义、视觉模式、文本语义）主动检索和配对图像。这使得数据来源更广泛，能够挖掘出跨页面、跨语义关系的丰富图像对，极大地提升了数据的可扩展性和多样性。

# 4. 方法论

## 4.1. 方法原理
MegaPairs 的核心思想是：大规模开放域图像语料库中蕴含着丰富多样的相关图像对。通过利用强大的预训练模型（如 CLIP、DINO）作为“检索器”，可以从海量图像中挖掘出具有不同关联类型（如视觉相似、语义相关）的图像对。然后，利用多模态大语言模型（MLLM）和大语言模型（LLM）自动生成描述这些图像对关系的自然语言指令。最终，这些合成的三元组（查询图像、指令、目标图像）用于训练通用的多模态检索器。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. MegaPairs 数据构建
MegaPairs 的构建包含两个主要步骤：挖掘相关图像对和生成开放式指令。

下图（原文 Figure 1）展示了 MegaPairs 的整体构建管道，包括从多模态三元组挖掘到开放式指令生成的全过程。

![该图像是示意图，展示了如何利用多个相似性模型从大规模图像-文本数据集中检索与2020年丰田Yaris概念车相关的图像。上半部分突出了查询项和检索的图像，底部则展示了挖掘的图像对及如何利用大语言模型生成相关指令。](images/1.jpg)
*该图像是示意图，展示了如何利用多个相似性模型从大规模图像-文本数据集中检索与2020年丰田Yaris概念车相关的图像。上半部分突出了查询项和检索的图像，底部则展示了挖掘的图像对及如何利用大语言模型生成相关指令。*

#### 步骤 1：挖掘相关图像对
首先，从大规模图像语料库中采样相关图像对。对于每个查询图像 $(I_q, C_q)$（其中 $I_q$ 为图像，$C_q$ 为对应的标题），利用多个相似度模型搜索一组多样化的相关目标图像 $\{I_{t_1}, I_{t_2}, ..., I_{t_n}\}$。

为了引入异构的关联性，本文使用了三种类型的相似度模型：
1.  **视觉-语义关联**：使用 CLIP 视觉编码器。这衡量两张图像在语义上的相关性，而不论视觉上的相似度（例如，同一款车的不同视角）。
2.  **视觉-模式关联**：使用 DINOv2 编码器。这衡量两张图像在视觉模式上的相似度，而不论语义相关性（例如，不同款车在相似背景下）。
3.  **标题关联**：使用 CLIP 文本编码器。这衡量两张图像对应的文本标题之间的相似度。

    对于挖掘到的每一对图像 $(I_q, I_{t_i})$，除了正样本外，还将检索集中的其他图像 $\{I_{t_j} | t_j \neq t_i\}$ 作为**困难负样本**加入训练数据。这有助于模型学习区分细微差异。

#### 步骤 2：生成开放式指令
利用开源 MLLM 和 LLM 对挖掘到的图像对进行自动标注。
1.  **描述生成**：首先，将图像对 $(I_q, I_{t_i})$ 输入到 MLLM（如 InternVL2-26B）中，生成一段详细的描述 $D_i$，阐述两张图像之间的共同概念和差异。
2.  **指令生成**：接着，将描述 $D_i$ 输入到 LLM（如 Llama-3-8B），提示 LLM 基于该描述生成多个文本指令 $T_{qt_i}$。这些指令旨在描述如何从查询图像 $I_q$ 过渡到目标图像 $I_{t_i}$。

    下图（原文 Figure 3）展示了用于 MLLM 生成描述的具体提示词，要求模型描述源图像和目标图像之间的连接与差异。

    ![Figure 3: The specific prompts for MLLM. The value of WORD_NUM ranges from 60 to 100 in our practical data generation to enhance the diversity of the generated description.](images/3.jpg)
    *该图像是插图，展示了观察源图像和目标图像之间的连接与差异的具体提示。两个图像都展示了关键共性。然而，源图像包含特定的细节，而目标图像则展示其他不同的特征，旨在引导比较分析。*

最终，构建一个多模态三元组 $(I_q, T_{qt_i}, I_{t_i})$，其中 $(I_q, T_{qt_i})$ 作为查询，用于检索 $I_{t_i}$。

### 4.2.2. MMRet 模型架构
本文提出了 MMRet 模型系列，基于预训练的 VLMs 构建通用多模态嵌入。

#### CLIP-based MMRet
对于基于 CLIP 的模型，采用双编码器架构。给定图像 $I$ 和文本 $T$，其嵌入计算如下：

$$
\begin{array} { l } { \mathbf { e } _ { i } = \Phi _ { I } ( I ) } \\ { \mathbf { e } _ { t } = \Phi _ { T } ( T ) } \end{array}
$$

其中，$\Phi_I$ 和 $\Phi_T$ 分别是图像和文本编码器。为了产生组合图像-文本样本 `(I, T)` 的多模态嵌入，采用**分数融合**策略，即直接对双编码器的输出进行逐元素相加：

$$
{ \bf e } _ { i t } = \Phi _ { I } ( I ) + \Phi _ { T } ( T )
$$

这里，$\mathbf{e}_{it}$ 表示融合后的查询嵌入向量。这种简单的加法操作使得模型能够将图像和文本信息在同一向量空间中进行对齐和组合。

#### MLLM-based MMRet
对于基于 MLLM 的模型，本文构建在 LLaVA-1.6 架构之上。MLLM 能够将图像 Token 直接输入到 LLM 中进行处理。在训练和推理阶段，使用任务特定的指令作为查询输入。

典型的多模态查询输入结构如下：
`instruct {task_inst} {query_text} {query_image} [EOS]`

其中，`{task_inst}` 是任务特定指令，`{query_text}` 是查询文本，`{query_image}` 是查询图像。模型使用 `[EOS]` Token（序列结束符）归一化后的最后隐藏状态作为给定输入序列的嵌入。这种利用特定 Token 嵌入的方法是 LLM-based 嵌入模型的常见做法。

### 4.2.3. 多模态对比学习
使用多模态对比学习将原始的 CLIP 和 MLLM 转化为 MMRet 模型。训练目标采用标准的 InfoNCE 损失函数：

$$
\mathcal { L } = - \frac { 1 } { \vert \boldsymbol { \mathcal { Q } } \vert } } \sum _ { q _ { i } \in \mathcal { Q } } \log \frac { \exp ( \mathbf { e } _ { q _ { i } } \cdot \mathbf { e } _ { c _ { i } ^ { + } } / \tau ) } { \sum _ { c _ { j } \in \mathcal { C } } \exp ( \mathbf { e } _ { q _ { i } } \cdot \mathbf { e } _ { c _ { j } } / \tau ) }
$$

*   **符号解释**：
    *   $\mathcal{Q}$：批次中所有查询样本 $q_i$ 的集合。
    *   $\mathbf{e}_{q_i}$：查询 $q_i$ 的嵌入向量。
    *   $\mathbf{e}_{c_i^+}$：查询 $q_i$ 对应的正候选（正确答案）的嵌入向量。
    *   $\mathcal{C}$：批次中所有候选样本的集合（包括正样本和负样本）。
    *   $\mathbf{e}_{c_j}$：候选 $c_j$ 的嵌入向量。
    *   $\tau$：温度参数，用于调节负样本的惩罚力度，本文中设置为 0.02。

        该损失函数旨在拉近查询与其正候选在特征空间中的距离，同时推远查询与所有其他候选（负样本）的距离。值得注意的是，查询 $q$ 和候选 $c$ 可以是图像、文本或组合的图像-文本数据。

# 5. 实验设置

## 5.1. 数据集
*   <strong>MegaPairs（训练集）</strong>：本文合成的数据集，包含 26,235,105 个图像对。基于 Recap-DataComp-1B 的子集构建，包含 2000 万张带标题的图像。
*   **CIRCO**：一个具有挑战性的零样本 CIR 基准测试，包含 123123 个候选自然图像。测试集包含 800 个组合图像-文本查询。
*   **CIRR**：第一个使用自然图像进行 CIR 任务的基准测试。测试集包含 4,148 个查询和 2,315 张图像的语料库。
*   **FashionIQ**：专注于时尚产品的 CIR 任务。验证集包含 6,016 个查询和 15,536 张图像。
*   **GeneCIS**：通用条件图像相似性基准测试，包含 4 个子任务。
*   **MMEB (Massive Multimodal Embedding Benchmark)**：包含 36 个数据集的综合基准测试，涵盖分类、VQA、检索和视觉定位 4 个元任务类别。

## 5.2. 评估指标
论文中使用了以下主要评估指标：

1.  **平均精度均值**
    *   **概念定义**：在信息检索中，mAP 衡量的是系统在所有查询上的平均精度。对于每个查询，计算其检索结果的精度平均值，然后对所有查询的这些值取平均。它综合反映了排序结果的优劣，既关注查准率也关注排序位置。
    *   **数学公式**：
        $$ \text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{n_i} \sum_{k=1}^{M} P(k) \times \text{rel}(k) \right) $$
    *   **符号解释**：
        *   $N$：查询的总数。
        *   $n_i$：第 $i$ 个查询对应的真实标注数据总数。
        *   $M$：检索列表的长度。
        *   `P(k)`：前 $k$ 个结果的查准率。
        *   $\text{rel}(k)$：指示函数，如果第 $k$ 个结果是相关的则为 1，否则为 0。

2.  <strong>召回率@K (Recall@K)</strong>
    *   **概念定义**：Recall@K 衡量的是在前 K 个检索结果中，相关结果占所有相关结果的比例。它关注系统是否能够找到尽可能多的相关信息。
    *   **数学公式**：
        $$ \text{Recall@K} = \frac{|\{ \text{relevant items} \} \cap \{ \text{top K items} \}|}{|\{ \text{relevant items} \}|} $$
    *   **符号解释**：
        *   分子：前 K 个结果中与真实标注数据相关的项目数量。
        *   分母：数据集中所有与该查询相关的项目总数。

3.  <strong>查准率@1 (Precision@1)</strong>
    *   **概念定义**：Precision@1（或 Hit@1）衡量的是排名第一的检索结果是否正确。这是一个严格的指标，常用于评估系统在最佳情况下的准确性。
    *   **数学公式**：
        $$ \text{Precision@1} = \begin{cases} 1 & \text{if } \text{item}_1 \in \{ \text{relevant items} \} \\ 0 & \text{otherwise} \end{cases} $$
    *   **符号解释**：
        *   $\text{item}_1$：检索列表中的第一个项目。

## 5.3. 对比基线
论文将 MMRet 与多个基线模型进行了比较，包括：
*   **SEARLE**：基于文本反转的零样本 CIR 方法。
*   **CIReVL**：利用视觉语言推理进行 CIR 的方法。
*   **LDRE**：基于 LLM 的发散推理和集成方法。
*   **MagicLens**：利用网页共现图像合成数据训练的模型，是本文的主要对比对象。
*   **E5-V**：使用 LLaVA-1.6 主干网络的通用多模态嵌入模型。
*   **VLM2Vec**：训练视觉语言模型以处理大规模多模态嵌入任务的模型。

# 6. 实验结果与分析

## 6.1. 核心结果分析
### 6.1.1. 零样本组合图像检索 (CIR) 性能
在 CIRCO、CIRR、FashionIQ 和 GeneCIS 四个基准测试上，MMRet 模型展现了卓越的性能。

以下是原文 [Table 1] 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th rowspan="2">Backbone</th>
<th rowspan="2"># Params</th>
<th>CIRCO</th>
<th colspan="2">CIRR</th>
<th>FashionIQ</th>
<th>GeneCIS</th>
</tr>
<tr>
<th>mAP@5</th>
<th>R@1</th>
<th>R_@1</th>
<th>R@10</th>
<th>R@1</th>
</tr>
</thead>
<tbody>
<tr>
<td>SEARLE (Baldrati et al., 2023)</td>
<td>CLIP-B</td>
<td>165M</td>
<td>9.4</td>
<td>24.0</td>
<td>54.9</td>
<td>22.9</td>
<td>-</td>
</tr>
<tr>
<td>CIReVL (Karthik et al., 2023)</td>
<td>CLIP-B</td>
<td>12.3B†</td>
<td>14.9</td>
<td>23.9</td>
<td>60.2</td>
<td>28.3</td>
<td>15.9</td>
</tr>
<tr>
<td>LDRE (Yang et al., 2024)</td>
<td>CLIP-B</td>
<td>7.9B†</td>
<td>18.0</td>
<td>25.7</td>
<td>60.5</td>
<td>24.8</td>
<td>-</td>
</tr>
<tr>
<td>MagicLens-B (Zhang et al., 2024)</td>
<td>CLIP-B</td>
<td>166M</td>
<td>23.1</td>
<td>27.0</td>
<td>66.7</td>
<td>26.3</td>
<td>15.0</td>
</tr>
<tr>
<td>MagicLens-B‡ (Zhang et al., 2024)</td>
<td>CoCa-B</td>
<td>267M</td>
<td>30.8</td>
<td>31.6</td>
<td>69.3</td>
<td>35.2</td>
<td>17.4*</td>
</tr>
<tr>
<td><strong>MMRet-Base</strong></td>
<td><CLIP-B</td>
<td>149M</td>
<td><strong>34.3</strong></td>
<td><strong>36.1</strong></td>
<td><strong>71.6</strong></td>
<td><strong>31.9</strong></td>
<td><strong>18.0</strong></td>
</tr>
<tr>
<td>Pic2Word (Saito et al., 2023)</td>
<td>CLIP-L</td>
<td>429M</td>
<td>8.7</td>
<td>23.9</td>
<td>-</td>
<td>24.7</td>
<td>11.2</td>
</tr>
<tr>
<td>PLI (Chen and Lai, 2023)</td>
<td>CLIP-L</td>
<td>428M</td>
<td>10.4</td>
<td>25.5</td>
<td>55.6</td>
<td>35.4</td>
<td>-</td>
</tr>
<tr>
<td>SEARLE (Baldrati et al., 2023)</td>
<td>CLIP-L</td>
<td>442M</td>
<td>11.7</td>
<td>24.2</td>
<td>53.8</td>
<td>25.6</td>
<td>12.3</td>
</tr>
<tr>
<td>CompoDiff (Gu et al., 2024a)</td>
<td>CLIP-L</td>
<td>568M</td>
<td>12.6</td>
<td>18.2</td>
<td>57.4</td>
<td>36.0</td>
<td>14.9</td>
</tr>
<tr>
<td>CIReVL (Karthik et al., 2023)</td>
<td>CLIP-L</td>
<td>12.5B†</td>
<td>18.6</td>
<td>24.6</td>
<td>59.5</td>
<td>28.6</td>
<td>15.9</td>
</tr>
<tr>
<td>LDRE (Yang et al., 2024)</td>
<td>CLIP-L</td>
<td>8.2B†</td>
<td>23.4</td>
<td>26.5</td>
<td>60.4</td>
<td>28.5</td>
<td>-</td>
</tr>
<tr>
<td>MagicLens-L (Zhang et al., 2024)</td>
<td>CLIP-L</td>
<td>465M</td>
<td>29.6</td>
<td>30.1</td>
<td>68.1</td>
<td>30.7</td>
<td>16.3</td>
</tr>
<tr>
<td>MagicLens-L‡ (Zhang et al., 2024)</td>
<td>CoCa-L</td>
<td>613M</td>
<td>34.1*</td>
<td>33.3*</td>
<td>70.9*</td>
<td>38.0</td>
<td>16.7</td>
</tr>
<tr>
<td><strong>MMRet-Large</strong></td>
<td><CLIP-L</td>
<td>428M</td>
<td><strong>39.2</strong></td>
<td><strong>38.0</strong></td>
<td><strong>73.2</strong></td>
<td><strong>34.6</strong></td>
<td><strong>18.1</strong></td>
</tr>
<tr>
<td>LDRE (Yang et al., 2024)</td>
<td>CLIP-G</td>
<td>10.3B†</td>
<td>31.1</td>
<td>36.2</td>
<td>68.8</td>
<td>32.5</td>
<td>-</td>
</tr>
<tr>
<td>CIReVL (Karthik et al., 2023)</td>
<td>CLIP-G</td>
<td>14.6B†</td>
<td>26.8</td>
<td>34.7</td>
<td>68.0</td>
<td>32.2</td>
<td>17.4*</td>
</tr>
<tr>
<td>IP-CIR (Li et al., 2024c)</td>
<td>CLIP-G</td>
<td>43.8B†</td>
<td>32.8</td>
<td>39.3</td>
<td>70.0</td>
<td>45.7*</td>
<td>-</td>
</tr>
<tr>
<td>E5-V (Jiang et al., 2024a)</td>
<td>LLaVA-1.6</td>
<td>8.35B</td>
<td>19.1</td>
<td>33.9</td>
<td>-</td>
<td>31.8</td>
<td>-</td>
</tr>
<tr>
<td>MM-Emded (Lin et al., 2024)</td>
<td>LLaVA-1.6</td>
<td>7.57B</td>
<td>32.3</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td><strong>MMRet-MLLM</strong></td>
<td><LLaVA-1.6</td>
<td>7.57B</td>
<td><strong>42.2</strong></td>
<td><strong>46.7</strong></td>
<td><strong>75.4</strong></td>
<td><strong>35.6</strong></td>
<td><strong>21.1</strong></td>
</tr>
</tbody>
</table>

**分析**：
1.  **SOTA 性能**：MMRet-MLLM 在 CIRCO 上达到了 42.2% mAP@5，比之前的 SOTA（MagicLens-L, 34.1%）高出 8.1%。在 CIRR 和 GeneCIS 上也取得了显著提升。
2.  **模型规模优势**：即使是较小的 MMRet-Base（149M 参数），在 CIRCO 上的表现（34.3%）也优于更大的 MagicLens-L（465M 参数，29.6%），证明了 MegaPairs 数据的高质量使得小模型也能发挥出色性能。
3.  **架构优势**：基于 LLaVA-1.6 的 MMRet-MLLM 模型在大多数基准测试中表现最佳，表明 MLLM 架构在理解复杂多模态指令方面具有优势。

### 6.1.2. MMEB 零样本与微调性能
在 MMEB 基准测试上，MMRet 同样表现出色。

以下是原文 [Table 2] 的零样本性能结果：

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th colspan="4">Per Meta-Task Score</th>
<th rowspan="2">Overall</th>
</tr>
<tr>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>Grounding</th>
</tr>
</tr>
<tr>
<th>number of datasets</th>
<th>10</th>
<th>10</th>
<th>12</th>
<th>4</th>
<th>36</th>
</tr>
</thead>
<tbody>
<tr>
<td>BLIP2 (Li et al., 2023)</td>
<td>27.0</td>
<td>4.2</td>
<td>33.9</td>
<td>47.0</td>
<td>25.2</td>
</tr>
<tr>
<td>SigLIP (Zhai et al., 2023)</td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>34.8</td>
</tr>
<tr>
<td>CLIP (Radford et al., 2021)</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.8.8</td>
</tr>
<tr>
<td>OpenCLIP (Cherti et al., 2023)</td>
<td>47.8</td>
<td>10.9</td>
<td>52.3</td>
<td>53.3</td>
<td>39.7</td>
</tr>
<tr>
<td>UniIR (Wei et al., 2024)</td>
<td>42.1</td>
<td>15.0</td>
<td>60.1†</td>
<td>62.2</td>
<td>42.8</td>
</tr>
<tr>
<td>MagicLens (Zhang et al., 2024)</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td>27.8</td>
</tr>
<tr>
<td>E5-V (LLaVA-1.6) (Jiang et al., 2024a)</td>
<td>21.8</td>
<td>4.9</td>
<td>11.5</td>
<td>19.0</td>
<td>13.3</td>
</tr>
<tr>
<td><strong>MMRet-MLLM (LLaVA-1.6)</strong></td>
<td><strong>47.2</strong></td>
<td><strong>18.4</strong></td>
<td><strong>56.5</strong></td>
<td><strong>62.2</strong></td>
<td><strong>44.0</strong></td>
</tr>
</tbody>
</table>

**分析**：MMRet-MLLM 在整体得分上达到了 44.0%，优于所有基线模型。特别是在 VQA 和 Grounding 任务上表现突出。

以下是原文 [Table 3] 的监督微调性能结果：

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
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
<tr>
<th>number of datasets</th>
<th>10</th>
<th>10</th>
<th>12</th>
<th>4</th>
<th>20</th>
<th>16</th>
<th>36</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>45.4</td>
</tr>
<tr>
<td>OpenCLIP</td>
<td>56.0</td>
<td>21.9</td>
<td>55.4</td>
<td>64.1</td>
<td>50.5</td>
<td>43.1</td>
<td>47.2</td>
</tr>
<tr>
<td>VLM2Vec (LLaVA-1.6)</td>
<td>54.7</td>
<td>50.3</td>
<td>56.2</td>
<td>64.0</td>
<td>61.0</td>
<td>47.5</td>
<td>55.0</td>
</tr>
<tr>
<td>VLM2Vec (Phi-3.5-V)</td>
<td>54.8</td>
<td>54.9</td>
<td>62.3</td>
<td>79.5</td>
<td>66.5</td>
<td>52.0</td>
<td>60.1</td>
</tr>
<tr>
<td><strong>MMRet-MLLM</strong></td>
<td><strong>56.0</strong></td>
<td><strong>57.4</strong></td>
<td><strong>69.9</strong></td>
<td><strong>83.6</strong></td>
<td><strong>68.0</strong></td>
<td><strong>59.1</strong></td>
<td><strong>64.1</strong></td>
</tr>
</tbody>
</table>

**分析**：在 MMEB 上进行微调后，MMRet-MLLM 的整体得分达到 64.1%，显著优于 VLM2Vec 等基线。特别是在 OOD（分布外）数据集上，MMRet-MLLM 得分为 59.1%，远高于 VLM2Vec (Phi-3.5-V) 的 52.0%，证明了其强大的泛化能力。

## 6.2. 消融实验/参数分析

### 6.2.1. 数据可扩展性与质量
下图（原文 Figure 2）展示了 MMRet-base 在 MegaPairs 数据集不同规模子集上的性能变化趋势。

![Figure 2: Performance scaling of MMRet-base on the MegaPairs as data size increases. The dashed lines indicate the performance of MagicLens-B (CLIP) trained on their dataset of $3 6 . 7 \\mathbf { M }$ data pairs.](images/2.jpg)
*该图像是一个图表，展示了在 MegaPairs 数据集上，MMRet-base 随着训练数据对数的增加而性能的变化情况。不同的曲线代表了 CIRCO、CIRR (val)、FashionIQ、GeneCIS 和平均性能。图中显示，随着训练数据数量的增加，性能逐渐提升，以达成最佳效果。*

**分析**：随着训练数据量的增加，模型在各个基准测试上的性能持续提升，证明了 MegaPairs 的可扩展性。值得注意的是，图中虚线表示 MagicLens-B 在其 36.7M 数据集上的性能。MMRet 仅使用 0.5M MegaPairs 数据（不到 MagicLens 的 2%）就超越了 MagicLens 的性能，这有力地证明了 MegaPairs 数据的高质量。

### 6.2.2. 困难负样本的影响
以下是原文 [Table 4] 关于不同负样本策略的性能对比结果：

以下是原文 [Table 4] 的结果：

| Negatives | | CIRCO | CIRR† | FIQ | CIS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Qry | HN | mAP@5 | R@1 | R@10 | R@1 |
| × | X | 10.1 | 0.2 | 25.3 | 14.4 |
| √ | X | 29.7 | 32.1 | 27.6 | 16.6 |
| √ | - | 32.3 | 33.7 | 30.1 | 17.0 |

**分析**：
*   “Qry”表示使用查询图像本身作为负样本。
*   “HN”表示使用挖掘到的困难负样本。
*   结果表明，不使用负样本或仅使用查询图像作为负样本时，性能较差。而使用挖掘到的困难负样本（√ -）能显著提升模型在所有基准测试上的性能。

### 6.2.3. 数据对搜索策略
以下是原文 [Table 5] 关于不同图像对搜索策略的性能对比结果：

以下是原文 [Table 5] 的结果：

| Strategy | | | CIRCO | CIRR† | FIQ | CIS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| D | I | T | mAP@5 | R@1 | R@10 | R@1 |
| ✓ | × | × | 29.0 | 31.5 | 30.0 | 24.7 |
| × | ✓ | × | 31.6 | 32.2 | 29.6 | 15.3 |
| × | × | ✓ | 32.4 | 33.3 | 28.7 | 17.3 |
| ✓ | ✓ | × | 31.0 | 32.1 | 28.5 | 17.1 |
| ✓ | × | ✓ | 32.2 | 33.3 | 29.7 | 16.4 |
| × | ✓ | ✓ | 32.3 | 33.3 | 28.9 | 17.5 |
| ✓ | ✓ | ✓ | 32.3 | 33.7 | 30.1 | 17.0 |

**分析**：
*   D: DINOv2 Encoder (Visual-Pattern)
*   I: CLIP Image Encoder (Visual-Semantic)
*   T: CLIP Text Encoder (Caption)
*   单独使用文本相似度（T）策略表现最好，可能是因为文本相似度捕捉了更多样的关系。
*   结合任意两种策略通常优于单一策略，因为增加了数据多样性。
*   同时使用三种策略（✓ ✓ ✓）提供了最稳健的性能，因此被选用于构建 MegaPairs。

## 6.3. 数据呈现 (表格)
为了完整性，以下展示附录中的详细结果表格。

以下是原文 [Table 6] 关于 CIRCO 基准测试的完整结果：

| Methods | Backbone | # Params | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PALAVRA (Cohen et al., 2022) | CLIP-B | 176M | 4.6 | 5.3 | 6.3 | 6.8 |
| PLI (Chen and Lai, 2023) | BLIP-B | 224M | 7.1 | 8.0 | 9.2 | 9.7 |
| SEARLE (Baldrati et al., 2023) | CLIP-B | 165M | 9.4 | 9.9 | 11.1 | 11.8 |
| CIReVL (Karthik et al., 2023) | CLIP-B | 12.3B† | 14.9 | 15.4 | 17.0 | 17.8 |
| LDRE (Yang et al., 2024) | CLIP-B | 7.9B† | 18.0 | 18.3 | 20.2 | 21.1 |
| MagicLens-B (Zhang et al., 2024) | CLIP-B | 166M | 23.1 | 23.8 | 25.8 | 26.7 |
| MagicLens-B‡ (Zhang et al., 2024) | CoCa-B | 267M | 30.8 | 32.0 | 34.5 | 35.6 |
| MMRet-Base | CLIP-B | 149M | 34.3 | 35.0 | 37.6 | 38.7 |
| Pic2Word (Saito et al., 2023) | CLIP-L | 429M | 8.7 | 9.5 | 10.6 | 11.3 |
| PLI (Chen and Lai, 2023) | CLIP-L | 428M | 10.4 | 11.6 | 13.0 | 13.7 |
| SEARLE (Baldrati et al., 2023) | CLIP-L | 442M | 11.7 | 12.7 | 14.3 | 15.1 |
| CIReVL (Karthik et al., 2023) | CLIP-L | 12.5B† | 18.6 | 19.0 | 20.9 | 21.8 |
| LinCIR (Gu et al., 2024b) | CLIP-L | 442M | 12.6 | 13.6 | 15.0 | 15.9 |
| CompoDiff (Gu et al., 2024a) | CLIP-L | 568M | 12.6 | 13.4 | 15.8 | 16.4 |
| MagicLens-L (Zhang et al. 2024) | CLIP-L | 465M | 29.6 | 30.8 | 33.4 | 34.4 |
| MagicLens-L‡ (Zhang et al., 2024) | CoCa-L | 613M | 34.1 | 35.4 | 38.1 | 39.2 |
| MMRet-Large | CLIP-L | 428M | 39.2 | 40.2 | 42.9 | 44.0 |
| Pic2Word (Saito et al., 2023) | CLIP-H | 987M | 11.7 | 12.3 | 13.7 | 14.4 |
| SEARLE (Baldrati et al., 2023) | CLIP-H | 1.0B | 16.1 | 16.9 | 18.8 | 19.7 |
| LinCIR (Gu et al., 2024b) | CLIP-H | 1.0B | 17.6 | 18.5 | 20.5 | 21.4 |
| Pic2Word (Saito et al., 2023) | CLIP-G | 2.5B | 5.5 | 5.6 | 6.7 | 7.1 |
| SEARLE (Baldrati et al., 2023) | CLIP-G | 2.6B | 13.2 | 13.9 | 15.3 | 16.0 |
| CompoDiff (Gu et al., 2024a) | CLIP-G | 2.9B | 15.3 | 17.7 | 19.4 | 20.7 |
| CIReVL (Karthik et al., 2023) | CLIP-G | 14.6B† | 26.8 | 27.6. | 30.0 | 31.0 |
| LinCIR (Gu et al., 2024b) | CLIP-G | 2.6B | 19.7 | 21.0. | 23.1 | 24.2 |
| LDRE (Yang et al., 2024) | CLIP-G | 10.3B† | 31.1 | 32.2 | 35.0 | 36.0 |
| IP-CIR (Li et al., 2024c) | CLIP-G | 43.8B† | 32.8 | - | - | - |
| E5-V (Jiang et al., 2024a) | LLaVA-1.6 | 8.35B | 19.1 | 34.3 | 36.9 | 38.0 |
| MM-Emded (Lin et al., 2024) | LLaVA-1.6 | 7.57B | 32.3 | - | - | - |
| MMRet-MLLM | LLaVA-1.6 | 7.57B | 42.2 | 43.4 | 46.5 | 47.6 |

以下是原文 [Table 10] 关于 MMEB 基准测试的详细结果：

<table>
<thead>
<tr>
<th colspan="10"></th>
</tr>
<tr>
<th>Task</th>
<th colspan="7"></th>
<th colspan="2">Fine-Tune</th>
</tr>
<tr>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th>CLIP</th>
<th>OpenCLIP</th>
<th>SigLIP</th>
<th>BLIP2</th>
<th>MagicLens</th>
<th>E5-V</th>
<th>UniIR</th>
<th>MMRet</th>
<th>VLM2Vec</th>
<th>MMRet</th>
</tr>
<tr>
<th colspan="10">Classification (10 tasks)</th>
</tr>
<tr>
<td>ImageNet-1K</td>
<td>55.8</td>
<td>63.5</td>
<td>45.4</td>
<td>10.3</td>
<td>48.0</td>
<td></td>
<td>9.6</td>
<td>53.7</td>
<td>49.1</td>
<td></td>
<td></td>
</tr>
<tr>
<td>N24News</td>
<td>34.7</td>
<td>38.6</td>
<td>13.9</td>
<td>36.0</td>
<td>33.7</td>
<td></td>
<td>23.4</td>
<td>33.9</td>
<td>45.8</td>
<td>65.6</td>
<td>79.5</td>
<td>58.8</td>
<td>71.3</td>
</tr>
<tr>
<td>HatefulMemes</td>
<td>51.1</td>
<td>51.7</td>
<td>47.2</td>
<td>49.6</td>
<td>49.0</td>
<td>49.7</td>
<td>51.0</td>
<td>51.0</td>
<td>67.1</td>
<td>53.7</td>
</tr>
<tr>
<td>VOC2007</td>
<td>50.7</td>
<td>52.4</td>
<td>64.3</td>
<td>52.1</td>
<td></td>
<td>51.6</td>
<td>49.9</td>
<td>62.7</td>
<td>74.6</td>
<td>88.6</td>
<td>85.0</td>
</tr>
<tr>
<td>SUN397</td>
<td>43.4</td>
<td>68.8</td>
<td>39.6</td>
<td>34.5</td>
<td></td>
<td>57.0</td>
<td>33.1</td>
<td>61.7</td>
<td>60.1</td>
<td>72.7</td>
<td>70.0</td>
</tr>
<tr>
<td>Place365</td>
<td>28.5</td>
<td>37.8</td>
<td>20.0</td>
<td>21.5</td>
<td></td>
<td>31.5</td>
<td>8.6</td>
<td>38.0</td>
<td>35.3</td>
<td>42.6</td>
<td>43.0</td>
</tr>
<tr>
<td>ImageNet-A</td>
<td>25.5</td>
<td>14.2</td>
<td>42.6</td>
<td>3.2</td>
<td>8.0</td>
<td></td>
<td>2.0</td>
<td>12.9</td>
<td>31.6</td>
<td>19.3</td>
<td>36.1</td>
</tr>
<tr>
<td>ImageNet-R</td>
<td>75.6</td>
<td>83.0</td>
<td>75.0</td>
<td>39.7</td>
<td></td>
<td>70.9</td>
<td>30.8</td>
<td>61.6</td>
<td>66.2</td>
<td>70.2</td>
<td>71.6</td>
</tr>
<tr>
<td>ObejectNet</td>
<td>43.4</td>
<td>51.4</td>
<td>40.3</td>
<td>20.6</td>
<td></td>
<td>31.6</td>
<td>7.5</td>
<td>37.1</td>
<td>49.2</td>
<td>29.5</td>
<td>55.8</td>
</tr>
<tr>
<td>Country-211</td>
<td>19.2</td>
<td>16.8</td>
<td>14.2</td>
<td>2.5</td>
<td></td>
<td>6.2</td>
<td>3.1</td>
<td>8.8</td>
<td>9.3</td>
<td>13.0</td>
<td>14.7</td>
</tr>
<tr>
<td>All Classification</td>
<td>42.8</td>
<td>47.8</td>
<td>40.3</td>
<td>27.0</td>
<td>38.8</td>
<td>21.8</td>
<td>42.1</td>
<td>47.2</td>
<td>54.8</td>
<td>56.0</td>
</tr>
<tr>
<th colspan="10">VQA (10 tasks)</th>
</tr>
<tr>
<td>OK-QA</td>
<td>7.5</td>
<td>11.5</td>
<td>2.4</td>
<td>8.7</td>
<td></td>
<td>12.7</td>
<td>8.9</td>
<td>24.5</td>
<td>28.0</td>
<td>63.2</td>
<td>73.3</td>
</tr>
<tr>
<td>A-KQA</td>
<td>3.8</td>
<td>3.3</td>
<td>1.5</td>
<td>3.2</td>
<td>2.9</td>
<td>5.9</td>
<td>10.6</td>
<td>11.6</td>
<td>50.2</td>
<td>56.7</td>
</tr>
<tr>
<td>DocVQA</td>
<td>4.0</td>
<td>5.3</td>
<td>4.2</td>
<td>2.6</td>
<td></td>
<td>3.0</td>
<td>1.7</td>
<td>5.6</td>
<td>12.6</td>
<td>78.4</td>
<td>78.5</td>
</tr>
<tr>
<td>InfographicsVQA</td>
<td>4.6</td>
<td>4.6</td>
<td>2.7</td>
<td>2.0</td>
<td></td>
<td>5.9</td2>
<td>2.3</td>
<td>5.0</td>
<td>10.6</td>
<td>40.8</td>
<td>39.3</td>
</tr>
<tr>
<td>ChartQA</td>
<td>1.4</td>
<td>1.5</td>
<td>3.0</td>
<td>0.5</td>
<td></td>
<td>0.9</td>
<td>2.4</td>
<td>1.8</td>
<td>2.4</td>
<td>59.0</td>
<td>41.7</td>
</tr>
<tr>
<td>Visual7W</td>
<td>4.0</td>
<td>2.6</td>
<td>1.2</td>
<td>1.3</td>
<td>2.5</td>
<td>5.8</td>
<td>12.3</td>
<td>9.0</td>
<td>47.7</td>
<td>49.5</td>
</tr>
<tr>
<td>ScienceQA</td>
<td>9.4</td>
<td>10.2</td>
<td>7.9</td>
<td></td>
<td>5.2</td>
<td>3.6</td>
<td>11.6</td>
<td>23.3</td>
<td>43.4</td>
<td>45.2</td>
</tr>
<tr>
<td>VizWiz</td>
<td>8.2</td>
<td>6.6</td>
<td>2.3</td>
<td></td>
<td>1.7</td>
<td>2.6</td>
<td>19.2</td>
<td>25.9</td>
<td>39.2</td>
<td>51.7</td>
</tr>
<tr>
<td>GQA</td>
<td>41.3</td>
<td>52.5</td>
<td>4.0</td>
<td>57.5</td>
<td>9.7</td>
<td></td>
<td>43.5</td>
<td>7.8</td>
<td>49.3</td>
<td>41.3</td>
<td>60.7</td>
<td>59.0</td>
</tr>
<tr>
<td>TextTextVQA</td>
<td>7.0</td>
<td>10.9</td>
<td>1.0</td>
<td>3.3</td>
<td></td>
<td></td>
<td>3.2</td>
<td>10.6</td>
<td>18.9</td>
<td>66.1</td>
<td>79.0</td>
</tr>
<tr>
<td>All VQA</td>
<td>9.1</td>
<td>10.9</td>
<td>8.4</td>
<td>4.2</td>
<td></td>
<td>4.6.8</td>
<td>3</td>
<td>15.0</td>
<td>18.4</td>
<td>54.9</td>
<td>57.4</td>
</tr>
<tr>
<th colspan="10">Retrieval (12 tasks)</th>
</tr>
<tr>
<td>VisDial</td>
<td>30.7</td>
<td>25.4</td>
<td>21.5</td>
<td>18.0</td>
<td>24.8</td>
<td>9.2</td>
<td>37.6</td>
<td>62.6</td>
<td>73.3</td>
<td>83.0</td>
</tr>
<tr>
<td>CIRR</td>
<td>12.6</td>
<td>15.4</td>
<td>15.1</td>
<td>9.8</td>
<td></td>
<td>39.1</td>
<td>6.1</td>
<td>53.2</td>
<td>65.7</td>
<td>47.8</td>
<td>61.4</td>
</tr>
<tr>
<td>VisualNews_t2i</td>
<td>78.9</td>
<td>74.0</td>
<td>51.0</td>
<td>48.1</td>
<td></td>
<td>50.7</td>
<td>13.5</td>
<td>63.6</td>
<td>45.7</td>
<td>67.2</td>
<td>74.2</td>
</tr>
<tr>
<td>VisualNews_i2t</td>
<td>79.6</td>
<td>78.0</td>
<td>52.4</td>
<td>13.5</td>
<td>21.1</td>
<td>8.1</td>
<td>68.8</td>
<td>53.4</td>
<td>70.7</td>
<td>78.1</td>
</tr>
<tr>
<td>MSCOCO_t2i</td>
<td>59.5</td>
<td>63.6</td>
<td>58.3</td>
<td>53.7</td>
<td></td>
<td>54.1</td>
<td>20.7</td>
<td>72.0</td>
<td>68.7</td>
<td>70.6</td>
<td>78.6</td>
</tr>
<tr>
<td>MSCOCO_i2t</td>
<td>57.7</td>
<td>62.1</td>
<td>55.0</td>
<td>20.3</td>
<td>40.0</td>
<td>14.0</td>
<td>74.1</td>
<td>56.7</td>
<td>66.5</td>
<td>72.4</td>
<td>68.3</td>
</tr>
<tr>
<td>NIGHTS</td>
<td>60.4</td>
<td>66.1</td>
<td>62.9</td>
<td>56.5</td>
<td>58.1</td>
<td>4.2</td>
<td>69.7</td>
<td>59.4</td>
<td>66.1</td>
<td></td>
</tr>
<tr>
<td>WebQA</td>
<td>67.5</td>
<td>62.1</td>
<td>58.1</td>
<td>55.4</td>
<td></td>
<td>43.0</td>
<td>17.7</td>
<td>86.3</td>
<td>76.3</td>
<td></td>
<td></td>
</tr>
<tr>
<td>FashionIQ</td>
<td>11.4</td>
<td>13.8</td>
<td>20.1</td>
<td>9.3</td>
<td></td>
<td>11.2</td>
<td>2.8</td>
<td>39.3</td>
<td>31.5</td>
<td>88.1</td>
<td>12.9</td>
</tr>
<tr>
<td>Wiki-SS-NQ</td>
<td>55.0</td>
<td>44.6</td>
<td>55.1</td>
<td>28.7</td>
<td></td>
<td>18.7</td>
<td>8.6</td>
<td>11.3</td>
<td>25.4</td>
<td>56.6</td>
<td>24.9</td>
</tr>
<tr>
<td>OVEN</td>
<td>41.1</td>
<td>45.0</td>
<td>56.0</td>
<td>39.5</td>
<td></td>
<td>1.6</td>
<td>5.9</td>
<td>66.6</td>
<td>73.0</td>
<td>47.3</td>
<td>87.5</td>
</tr>
<tr>
<td>EDIS</td>
<td>81.0</td>
<td>77.5</td>
<td>23.6</td>
<td>54.4</td>
<td>62.6</td>
<td></td>
<td>26.8</td>
<td>78.2</td>
<td>59.9</td>
<td>79.9</td>
<td>65.6</td>
</tr>
<tr>
<td>All Retrieval</td>
<td>53.0</td>
<td>52.3</td>
<td>31.6</td>
<td>33.9</td>
<td>35.4</td>
<td></td>
<td>11.5</td>
<td>60.1</td>
<td>56.5</td>
<td>62.3</td>
<td>69.9</td>
</tr>
<tr>
<th colspan="10">Visual Grounding (4 tasks)</th>
</tr>
<tr>
<td>MSCOCO</td>
<td>33.8</td>
<td>34.5</td>
<td>46.4</td>
<td>28.9</td>
<td>22.1</td>
<td></td>
<td>10.8</td>
<td>46.6</td>
<td>42.7</td>
<td>67.3</td>
<td>76.8</td>
</tr>
<tr>
<td>RefCOCO</td>
<td>56.9</td>
<td>61.3</td>
<td>54.2</td>
<td>68.3</td>
<td>70.8</td>
<td>47.4</td>
<td>22.8</td>
<td>35.6</td>
<td>11