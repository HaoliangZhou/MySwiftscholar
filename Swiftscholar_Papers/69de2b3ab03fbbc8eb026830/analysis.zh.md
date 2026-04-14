# 1. 论文基本信息

## 1.1. 标题
VLM2Vec: Training Vision-Language Models for Massive Multimodal Embedding Tasks

## 1.2. 作者
Ziyan Jiang (University of Waterloo, Salesforce Research), Rui Meng (Salesforce Research), Xinyi Yang (Salesforce Research), Semih Yavuz (Salesforce Research), Yingbo Zhou (Salesforce Research), Wenhu Chen (University of Waterloo).

## 1.3. 发表期刊/会议
arXiv (预印本)

## 1.4. 发表年份
2024

## 1.5. 摘要
论文旨在探索构建能够处理广泛下游任务的通用多模态嵌入模型的潜力。其主要贡献包括两个方面：首先，提出了 MMEB（Massive Multimodal Embedding Benchmark），这是一个涵盖 4 个元任务（分类、视觉问答、多模态检索和视觉定位）和 36 个数据集的基准，包含 20 个训练集和 16 个评估集；其次，提出了 VLM2Vec（Vision-Language Model -> Vector），这是一个对比训练框架，通过在 MMEB 上的训练，将任何最先进的视觉语言模型转换为嵌入模型。与 CLIP 和 BLIP 等独立编码文本或图像且无任务指令的模型不同，VLM2Vec 可以根据任务指令处理图像和文本的任意组合以生成固定维度的向量。实验结果表明，VLM2Vec 在 MMEB 的分布内和分布外数据集上均比现有的多模态嵌入模型有 10% 到 20% 的绝对平均提升，证明了 VLM 实际上是强大的嵌入模型。

## 1.6. 原文链接
https://arxiv.org/abs/2410.05160

# 2. 整体概括

## 2.1. 研究背景与动机
近年来，文本嵌入模型取得了显著进展，出现了如 MTEB (Massive Text Embedding Benchmark) 这样的通用基准和一系列能够跨任务泛化的通用文本嵌入模型。然而，多模态嵌入领域的发展相对滞后。现有的多模态嵌入研究通常局限于孤立的任务（如 ImageNet 分类或简单的图文检索），且大多数模型（如 CLIP、BLIP）采用独立编码器或浅层融合机制，无法充分捕捉文本与图像之间的深层关系，也缺乏处理复杂推理任务和遵循指令的能力。

这篇论文的切入点在于利用现有的、经过大规模预训练的视觉语言模型（VLMs，如 LLaVA、Phi-3.5-V）作为主干网络。VLMs 具备强大的多模态理解能力和指令遵循能力，论文试图通过对比学习将这些生成式或理解式的 VLMs 转化为通用的多模态嵌入模型，以填补该领域缺乏统一基准和通用方法的空白。

## 2.2. 核心贡献/主要发现
1.  **MMEB 基准**：提出了一个大规模多模态嵌入基准，包含 36 个数据集，覆盖分类、视觉问答、检索和视觉定位四大类任务。所有任务被统一重新表述为排序问题，用于全面评估模型的通用性。
2.  **VLM2Vec 框架**：提出了一种对比训练框架，能够将任何 SOTA 的 VLM 转化为嵌入模型。该方法利用 VLM 深度融合视觉和语言特征的优势，通过任务指令引导模型生成上下文相关的嵌入。
3.  **性能提升**：实验证明，VLM2Vec 在 MMEB 上取得了显著优于 CLIP、BLIP、UniR 等基线模型的性能。特别是在分布外的零样本评估中，模型展现了强大的泛化能力。
4.  **关键发现**：研究发现，VLMs 本质上是强大的嵌入模型，只要通过适当的对比学习进行微调，就能在多模态嵌入任务上表现出色。此外，指令对于提升模型在复杂任务上的表现至关重要。

    下图（原文 Figure 1）展示了 VLM2Vec 模型和 MMEB benchmark 的整体框架，体现了模型在不同任务指令下处理图像和文本组合的能力：

    ![Figure 1: We develop a universal multimodal embedding benchmark, MMEB, along with ${ \\tt V I M 2 V E C }$ , an embedding model adapted from vision-language models (VLMs). $\\mathtt { V I M 2 V E C }$ is capable of following instructions and performing various multimodal embedding tasks, accommodating any combination of image and text modalities.](images/1.jpg)
    *该图像是一个示意图，展示了 VLM2Vec 模型和 MMEB benchmark 的适用性，通过不同的任务指令，展示了图像和文本之间的多模态嵌入关系。示意中包含了分类、视觉问答、视觉定位和检索等任务，体现了模型的灵活应用。*

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，读者需要熟悉以下核心概念：

*   **嵌入**：将离散的输入（如文本、图像）映射为连续的固定维度向量表示。这些向量能够捕捉输入之间的语义相似性，常用于检索、聚类等任务。
*   <strong>视觉语言模型 (VLM)</strong>：一种能够同时处理和理解图像与文本输入的深度学习模型。通常基于 Transformer 架构，将图像编码器（如 ViT）与大型语言模型（LLM）连接，通过投影层或注意力机制实现跨模态交互。
*   **对比学习**：一种自监督学习方法，通过拉近正样本对的距离、推远负样本对的距离来学习特征表示。在嵌入学习中，通常使用 InfoNCE 损失函数。
*   **LoRA (Low-Rank Adaptation)**：一种参数高效的微调技术。它冻结预训练模型的权重，并在 Transformer 层中注入可训练的低秩分解矩阵，从而在大幅减少可训练参数量的同时实现模型微调。
*   **GradCache**：一种梯度缓存技术，用于在显存受限的情况下扩大对比学习的批次大小。它将梯度的计算和反向传播解耦，允许模型在处理大批次数据时，分批次计算梯度并进行累加。

## 3.2. 前人工作
*   **文本嵌入**：从 Word2Vec、GloVe 发展到基于 BERT 的模型。近期的研究如 GTR、E5、InstructOR 开始关注通用文本嵌入，利用 LLM 作为主干网络并通过指令微调提升跨任务泛化能力。
*   **多模态嵌入**：早期工作如 CLIP、BLIP、SigLIP 主要通过对比图像-文本对来学习对齐的表示空间。这些模型通常采用双塔架构，分别处理图像和文本。UniR 和 MagicLens 尝试利用 CLIP 特征进行浅层融合以处理更复杂的检索任务。E5-V 是近期的工作，尝试将 VLM 用于嵌入，但其训练数据仅包含文本对。
*   **嵌入基准**：MTEB 是文本领域的权威基准。多模态方面，MBEIR 提供了 8 个检索任务，但覆盖范围和任务多样性仍有限。

## 3.3. 技术演进
该领域经历了从单一模态嵌入到多模态嵌入，从特定任务模型到通用模型的演进。早期多模态模型侧重于图文对齐（如 CLIP），随后出现了具备更强理解和生成能力的 VLMs（如 LLaVA）。本文的工作处于技术演进的前沿，即利用 VLMs 强大的表示能力，将其转化为通用的多模态嵌入模型，并建立了类似于 MTEB 的多模态评估标准。

## 3.4. 差异化分析
与 CLIP、BLIP 等传统双塔模型不同，VLM2Vec 利用 VLM 的单塔架构（或深度融合架构），在 Transformer 内部实现图像和文本特征的深度交互，而非简单的后期特征拼接。与 UniR、MagicLens 等基于 CLIP 特征的浅层融合方法相比，VLM2Vec 能够处理高分辨率图像和长文本，并具备更强的指令遵循能力。与 E5-V 相比，VLM2Vec 直接在多模态数据对上进行训练，而非仅依赖文本对。

# 4. 方法论

## 4.1. 方法原理
VLM2Vec 的核心思想是将多模态嵌入任务统一表述为排序问题。给定一个查询和一个目标，模型的目标是根据查询从候选集中找出正确的目标。VLM2Vec 利用预训练的 VLM 作为主干网络，通过对比学习训练，使模型能够根据任务指令，将任意组合的图像和文本输入映射到统一的向量空间中。在这个空间里，语义相关的查询和目标向量具有更高的相似度。

## 4.2. 核心方法详解

### 4.2.1. 任务重述与输入构建
首先，MMEB 中的所有任务都被重述为排序任务。对于一个查询-目标对 $(q, t^+)$，其中 $q$ 和 $t^+$ 可以是单独的图像、文本或二者的组合。为了增强模型的指令遵循能力，作者将任务指令整合到查询输入中。新的查询 $q_{\text{inst}}$ 构建如下：

$$
q_{\text{inst}} = [\text{IMAGE\_TOKEN}] \text{Instr}; \{\text{task\_definition}\} \backslash n \text{Query}; \{q\}
$$

其中，$[\text{IMAGE\_TOKEN}]$ 是图像占位符，$\text{Instr}$ 是任务指令，$\{\text{task\_definition}\}$ 是任务定义的占位符，$\{q\}$ 是原始查询内容。这种格式强制模型在生成嵌入时考虑任务上下文。

### 4.2.2. 模型架构与嵌入提取
VLM2Vec 直接使用预训练的 VLM（如 Phi-3.5-V 或 LLaVA-1.6）作为编码器。将构建好的查询 $q_{\text{inst}}$ 和目标 $t^+$ 输入 VLM，提取模型最后一层最后一个 token 的词向量作为查询和目标的嵌入表示，分别记为 $\mathbf{h}_{q_{\text{inst}}}$ 和 $\mathbf{h}_{t^+}$。这种利用最后一个 token 向量的做法类似于 LLM 生成句子嵌入的常用方法。

下图（原文 Figure 3）展示了 VLM2Vec 的架构，它使用 VLM 作为主干网络，通过对比损失训练查询和目标：

![Figure 3: VLM2VE c uses a VLM as the backbone to deeply integrate image and text features. It is trained with a contrastive loss between the query and target, following task-specific instructions. The training data consists of diverse combinations of modalities on both the query and target sides, which may include images, text, or image-text pairs.](images/3.jpg)
*该图像是示意图，展示了 VLM2Vec 模型的工作流程。左侧为查询部分，包含图像编码器与投影层，右侧为目标部分，同样包含图像编码器与投影层。两者通过对比损失训练，以实现图像和文本特征的深度融合。*

### 4.2.3. 对比训练损失
为了训练嵌入模型，论文采用了标准的 InfoNCE 损失函数。该损失函数旨在最大化正样本对（正确的查询-目标对）之间的相似度，同时最小化负样本对（查询与错误候选）之间的相似度。损失函数 $\mathcal{L}$ 定义如下：

$$
\operatorname*{min} \mathcal{L} = -\log \frac{\phi(\mathbf{h}_{q_{\text{inst}}}, \mathbf{h}_{t^+})}{\phi(\mathbf{h}_{q_{\text{inst}}}, \mathbf{h}_{t^+}) + \displaystyle \sum_{t^- \in \mathbb{N}} (\phi(\mathbf{h}_{q_{\text{inst}}}, \mathbf{h}_{t^-}))}
$$

公式中各符号的含义如下：
*   $\mathcal{L}$：对比学习的损失值，目标是最小化该值。
*   $\mathbf{h}_{q_{\text{inst}}}$：查询经过 VLM 编码后得到的向量表示。
*   $\mathbf{h}_{t^+}$：正样本目标经过 VLM 编码后得到的向量表示。
*   $\mathbf{h}_{t^-}$：负样本目标经过 VLM 编码后得到的向量表示。
*   $\mathbb{N}$：所有负样本的集合，通常包含批次内的其他样本和难负样本。
*   $\phi(\mathbf{h}_q, \mathbf{h}_t)$：计算两个向量之间匹配分数的函数。在本文中，采用带温度缩放的余弦相似度函数，定义为 $\phi(\mathbf{h}_q, \mathbf{h}_t) = \exp(\frac{1}{\tau} \cos(\mathbf{h}_q, \mathbf{h}_t))$，其中 $\tau$ 是温度超参数，用于控制分布的平滑程度。

### 4.2.4. 大批次训练与 GradCache
由于多模态数据包含图像，显存消耗巨大，限制了批次大小。为了利用大批次数据带来的丰富负样本，论文采用了 GradCache 技术。其核心思想是将梯度的计算分解为两步。

假设有一个大的查询批次 $\mathcal{Q}$，将其划分为若干子批次 $\hat{Q}_1, \hat{Q}_2, \dots$，每个子批次可以放入显存。首先计算损失函数对嵌入向量的梯度 `\mathbf{u}_i = \frac{\partial \mathcal{L}}{\partial f(q_i)}` 并缓存。然后，对每个子批次进行反向传播，计算编码器参数 $\Theta$ 的梯度。最终的梯度是所有子批次梯度的累加：

$$
\frac{\partial \mathcal{L}}{\partial \Theta} = \sum_{\hat{Q}_j \in \mathcal{Q}} \sum_{q_i \in \hat{Q}_j} \mathbf{u}_i \frac{\partial f(q_i)}{\partial \Theta}
$$

这里，$\frac{\partial f(q_i)}{\partial \Theta}$ 是嵌入向量对模型参数的梯度。通过这种方式，模型可以在显存有限的情况下模拟大批次训练的效果。

# 5. 实验设置

## 5.1. 数据集
实验使用了论文提出的 MMEB 基准，包含 36 个数据集，分为 4 个元任务类别：分类、视觉问答 (VQA)、检索和视觉定位。其中 20 个数据集作为分布内训练集，16 个作为分布外评估集。

下图（原文 Figure 2）展示了 MMEB 中四个元任务及其对应的 36 个数据集概览：

![Figure 2: An overview of the tasks and datasets in MMEB. MMEB includes four meta-tasks and 36 datasets: 20 in-distribution datasets (blue) used for training and 16 out-of-distribution (orange) datasets used exclusively for evaluation.](images/2.jpg)
*该图像是一个示意图，展示了MMEB中的四个元任务及其对应的36个数据集。分类、视觉问题解答、检索和视觉定位各自包含不同的数据集，其中蓝色方框代表用于训练的20个同分布数据集，橙色方框则表示用于评估的16个非同分布数据集。*

以下是原文 [Table 1] 的统计数据：

<table>
<thead>
<tr>
<th rowspan="2">Meta-Task</th>
<th rowspan="1" colspan="1">Dataset</th>
<th colspan="2">Query</th>
<th colspan="2">Target</th>
<th rowspan="1">OOD?</th>
<th rowspan="1">#Training</th>
<th rowspan="1">#Eval</th>
<th rowspan="1">#Candidates</th>
</tr>
<tr>
<th></th>
<th>I</th>
<th>T</th>
<th>I</th>
<th>T</th>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="10">Classification(10 Tasks)</td>
<td>ImageNet-1K</td>
<td>✓</td>
<td></td>
<td></td>
<td>✓</td>
<td></td>
<td>100K</td>
<td>1000</td>
<td>1000</td>
</tr>
<tr>
<td>N24News</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td></td>
<td>49K</td>
<td>1000</td>
<td>24</td>
</tr>
<tr>
<td>HatefulMemes</td>
<td>✓</td>
<td></td>
<td></td>
<td>✓</td>
<td></td>
<td>8K</td>
<td>1000</td>
<td>2</td>
</tr>
<tr>
<td>VOC2007</td>
<td>✓</td>
<td></td>
<td></td>
<td>✓</td>
<td></td>
<td>8K</td>
<td>1000</td>
<td>20</td>
</tr>
<tr>
<td>SUN397</td>
<td>✓</td>
<td></td>
<td></td>
<td>✓</td>
<td></td>
<td>20K</td>
<td>1000</td>
<td>397</td>
</tr>
<tr>
<td>Place365</td>
<td>✓</td>
<td></td>
<td></td>
<td>✓</td>
<td>✓</td>
<td>b</td>
<td>1000</td>
<td>365</td>
</tr>
<tr>
<td>ImageNet-A</td>
<td>✓</td>
<td></td>
<td></td>
<td>✓</td>
<td>✓</td>
<td>-</td>
<td>1000</td>
<td>1000</td>
</tr>
<tr>
<td>ImageNet-R</td>
<td>✓</td>
<td></td>
<td></td>
<td>✓</td>
<td>✓</td>
<td>-</td>
<td>1000</td>
<td>200</td>
</tr>
<tr>
<td>ObjectNet</td>
<td>✓</td>
<td></td>
<td></td>
<td>✓</td>
<td>✓</td>
<td>-</td>

      < <td>1000</td>
      <td>313</td>
    </tr>
    <tr>
      <td>Country-211</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td>b</td>
      <td>1000</td>
      <td>211</td>
    </tr>
    <tr>
      <td rowspan="10">VQA(10 Tasks)</td>
      <td>OK-VQA</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>9K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>A-OKVQA</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>17K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>DocVQA</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>40K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>InfographicVQA</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>24K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>ChartQA</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>28K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>Visual7W</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>70K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>ScienceQA</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td>b</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>VizWiz</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td>-</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>GQA</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td>-</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>TextVQA</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td>-</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td rowspan="12">Retrieval(12 Tasks)</td>
      <td>VisDial</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>123K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>CIRR</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>26K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>VisualNews_t2i</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>100K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>VisualNews_i2t</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>100K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>MSCOCO_t2i</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>100K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>MSCOCO i2t</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>113K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>NIGHTS</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>16K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>WebQA</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>17K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>OVEN</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>b</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>FashionIQ</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td>-</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>EDIS</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>b</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>Wiki-SS-NQ</td>
      <td></td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td>-</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td rowspan="4">Visual Grounding(4 Tasks)</td>
      <td>MSCOCO</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td></td>
      <td>100K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>Visual7W-Pointing</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>RefCOCO</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>RefCOCO-Matching</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>✓</td>
      <td>-</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>

## 5.2. 评估指标
论文主要使用 Precision@1 (P@1) 作为评估指标。

1.  **概念定义**：Precision@1 衡量的是在所有查询中，模型预测的排名第一的候选结果是否是真实标注数据的比例。它直观地反映了模型在给定查询下，从众多候选中直接找到正确答案的能力。
2.  **数学公式**：
    $$ P@1 = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{rank}(t_i^+) = 1) $$
3.  **符号解释**：
    *   $N$：查询的总数。
    *   $t_i^+$：第 $i$ 个查询对应的真实标注数据。
    *   $\text{rank}(t_i^+)$：真实标注数据在模型返回的候选列表中的排名。
    *   $\mathbb{I}(\cdot)$：指示函数，当条件为真时返回 1，否则返回 0。

## 5.3. 对比基线
论文选取了四类具有代表性的基线模型：
*   **CLIP-family**：包括 CLIP, Open-CLIP, SigLIP, BLIP2。这些是经典的多模态模型，通常不使用指令。
*   **UniR**：基于 CLIP 和 BLIP 的统一检索模型，使用浅层融合技术。
*   **MagicLens**：自监督图像检索模型，使用多头注意力池化统一多模态输入。
*   **E5-V**：利用 VLM 的多模态嵌入模型，但仅在文本对上训练。

# 6. 实验结果与分析

## 6.1. 核心结果分析
以下是原文 [Table 2] 的主要实验结果，展示了 VLM2Vec 与各基线模型在 MMEB 上的性能对比：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
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
<td colspan="8">Baseline Models (No Fine-tuning on MMEB Training)</td>
</tr>
<tr>
<td>CLIP (Radford et al., 2021)</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.1</td>
<td>38.7</td>
<td>37.8</td>
</tr>
<tr>
<td>BLIP2 (Li et al., 2023a)</td>
<td>27.0</td>
<td>4.2</td>
<td>33.9</td>
<td>47.0</td>
<td>25.3</td>
<td>25.1</td>
<td>25.2</td>
</tr>
<tr>
<td>SigLIP (Zhai et al., 2023)</td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>32.3</td>
<td>38.0</td>
<td>34.8</td>
</tr>
<tr>
<td>OpenCLIP (Cherti et al., 2023)</td>
<td>47.8</td>
<td>10.9</td>
<td>52.3</td>
<td>53.3</td>
<td>39.3</td>
<td>40.2</td>
<td>39.7</td>
</tr>
<tr>
<td>UniR (BLIP_FF) (Wei et al., 2023)</td>
<td>42.1</td>
<td>15.0</td>
<td>60.1</td>
<td>62.2</td>
<td>44.7</td>
<td>40.4</td>
<td>42.8</td>
</tr>
<tr>
<td>UniR (CLIP_SF) (Wei et al., 2023)</td>
<td>44.3</td>
<td>16.2</td>
<td>61.8</td>
<td>65.3</td>
<td>47.1</td>
<td>41.7</td>
<td>44.7</td>
</tr>
<tr>
<td>E5-V (Jiang et al., 2024)</td>
<td>21.8</td>
<td>4.9</td>
<td>11.5</td>
<td>19.0</td>
<td>14.9</td>
<td>11.5</td>
<td>13.3</td>
</tr>
<tr>
<td>Magiclens (Zhang et al., 2024)</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td>31.0</td>
<td>23.7</td>
<td>27.8</td>
</tr>
<tr>
<td colspan="8">Baseline Models (Fine-tuning on MMEB Training)</td>
</tr>
<tr>
<td>CLIP-FFT</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>45.4</td>
</tr>
<tr>
<td>OpenCLIP-FFT</td>
<td>56.0</td>
<td>21.9</td>
<td>55.4</td>
<td>64.1</td>
<td>50.5</td>
<td>43.1</td>
<td>47.2</td>
</tr>
<tr>
<td colspan="8">Ours (VLM2VEC)</td>
</tr>
<tr>
<td>Phi-3.5-V, FFT (bs=1024)</td>
<td>52.8</td>
<td>50.3</td>
<td>57.8</td>
<td>72.3</td>
<td>62.8</td>
<td>47.4</td>
<td>55.9</td>
</tr>
<tr>
<td>Phi-3.5-V, LoRA (bs=1024)</td>
<td>54.8</td>
<td>54.9</td>
<td>62.3</td>
<td>79.5</td>
<td>66.5</td>
<td>52.0</td>
<td>60.1</td>
</tr>
<tr>
<td>LLaVA-1.6, LoRA (bs=1024,res=336x336)</td>
<td>54.7</td>
<td>50.3</td>
<td>56.2</td>
<td>64.0</td>
<td>61.0</td>
<td>47.5</td>
<td>55.0</td>
</tr>
<tr>
<td>LLaVA-1.6, LoRA (bs=1024,res=1344x1344)</td>
<td>61.2</td>
<td>49.9</td>
<td>67.4</td>
<td>86.1</td>
<td>67.5</td>
<td>57.1</td>
<td>62.9</td>
</tr>
<tr>
<td>Δ - Best baseline (No Fine-tuning)</td>
<td>+16.9</td>
<td>+33.7</td>
<td>+5.6</td>
<td>+20.8</td>
<td>+20.4</td>
<td>+15.4</td>
<td>+18.2</td>
</tr>
<tr>
<td>∆ - Best baseline (Fine-tuning)</td>
<td>+5.2</td>
<td>+28.0</td>
<td>+12.0</td>
<td>+22.0</td>
<td>+17.0</td>
<td>+14.0</td>
<td>+15.7</td>
</tr>
</tbody>
</table>

从表中可以看出，VLM2Vec 的最佳变体（基于 LLaVA-1.6，使用 LoRA，高分辨率 1344x1344）在所有 36 个数据集上达到了 62.9% 的平均 P@1，在 16 个分布外数据集上达到了 57.1%。与未微调的最佳基线相比，提升了 18.2 个百分点；与经过微调的最佳基线相比，提升了 15.7 个百分点。这充分证明了 VLM 作为嵌入模型主干的优越性，以及通过指令微调实现跨任务泛化的有效性。

## 6.2. 消融实验/参数分析

### 6.2.1. 全参数微调 vs. LoRA
以下是原文 [Table 3] 的结果，比较了全参数微调与不同秩的 LoRA 微调：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4">Meta-Task Average Score</th>
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
<td>Full Fine-Tuning (bs=256)</td>
<td>50.4</td>
<td>46.4</td>
<td>52.6</td>
<td>68.6</td>
<td>57.9</td>
<td>44.7</td>
<td>52.0</td>
</tr>
<tr>
<td>LoRA r = 4 (bs=256)</td>
<td>52.7</td>
<td>53.6</td>
<td>60.1</td>
<td>80.2</td>
<td>64.9</td>
<td>50.4</td>
<td>58.4</td>
</tr>
<tr>
<td>LoRA r = 8 (bs=256)</td>
<td>52.9</td>
<td>52.5</td>
<td>60.3</td>
<td>80.0</td>
<td>64.2</td>
<td>50.8</td>
<td>58.2</td>
</tr>
<tr>
<td>LoRA r = 16 (bs=256)</td>
<td>51.1</td>
<td>40.5</td>
<td>52.0</td>
<td>72.5</td>
<td>54.9</td>
<td>45.8</td>
<td>50.8</td>
</tr>
<tr>
<td>LoRA r = 32 (bs=256)</td>
<td>50.6</td>
<td>47.8</td>
<td>53.9</td>
<td>72.5</td>
<td>58.9</td>
<td>46.5</td>
<td>53.4</td>
</tr>
</tbody>
</table>

结果显示，适当配置秩的 LoRA（如 r=4 或 r=8）性能优于全参数微调。这可能是因为 LoRA 具有正则化效应，防止了在较小数据集上的过拟合。

### 6.2.2. 训练参数的影响
下图（原文 Figure 4）展示了训练批次大小、子图像裁剪数量和训练步数对 VLM2Vec 性能的影响：

![Figure 4: The figures demonstrate the influence of the training setup on $\\mathtt { V I M 2 V E C }$ 's final performance. Here, we examine the effects of training batch size, the number of sub-image crops, and the number of training steps. All the models utilize $\\mathrm { P h i } { - } 3 . 5 { \\cdot } \\mathrm { V }$ as their backbone.](images/4.jpg)
*该图像是一个图表，展示了训练设置对 VLM2Vec 最终性能的影响，包括批量大小、步长大小和图像裁剪数量的影响。左侧图表显示批量大小增加时性能的提升，中间图表展示步长对性能的影响，右侧图表则反映了裁剪数量与性能之间的关系。所有模型均采用 $\mathrm{Phi-3.5-V}$ 作为基础。*

从图中可以看出，增大批次大小、增加子图像裁剪数量和增加训练步数均能提升模型性能。特别是批次大小，由于缺乏难负样本，大批次带来的丰富随机负样本对于对比学习至关重要。

### 6.2.3. 指令的影响
以下是原文 [Table 4] 的结果，对比了 CLIP 和 VLM2Vec 在有指令和无指令情况下的表现：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4">Meta-Task Average Score</th>
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

      < <td colspan="8">CLIP</td>
    </tr>
    <tr>
      <td>w/o instruction</td>
      <td>42.8</td>
      <td>9.1</td>
      <td>53.0</td>
      <td>51.8</td>
      <td>37.1</td>
      <td>38.7</td>
      <td>37.8</td>
    </tr>
    <tr>
      <td>w/ instruction</td>
      <td>17.4</td>
      <td>8.0</td>
      <td>41.3</td>
      <td>52.9</td>
      <td>23.8</td>
      <td>30.3</td>
      <td>26.7</td>
    </tr>
    <tr>
      <td>∆</td>
      <td>-59.3%</td>
      <td>-12.1%</td>
      <td>--22.1%</td>
      <td>2.1%</td>
      <td>-35.8%</td>
      <td>-21.7%</td>
      <td>-29.4%</td>
    </tr>
    <tr>
      <td colspan="8">Ours (VLM2Vec)</td>
    </tr>
    <tr>
      <td>w/o instruction</td>
      <td>36.7</td>
      <td>33.5</td>
      <td>31.1</td>
      <td>44.3</td>
      <td>37.3</td>
      <td>31.6</td>
      <td>34.8</td>
    </tr>
    <tr>
      <td>w/ instruction</td>
      <td>50.4</td>
      <td>46.4</td>
      <td>52.6</td>
      <td>68.6</td>
      <td>57.9</td>
      <td>44.7</td>
      <td>52.0</td>
    </tr>
    <tr>
      <td>∆</td>
      <td>37.3%</td>
      <td>38.5%</td>
      <td>69.1%</td>
      <td>54.9%</td>
      <td>55.2%</td>
      <td>41.5%</td>
      <td>49.4%</td>
    </tr>
  </tbody>
</table>

对于 CLIP，加入指令导致性能下降了 29.4%，这表明 CLIP 缺乏处理指令的能力。而对于 VLM2Vec，加入指令带来了 49.4% 的显著提升，证明了 VLM 主干网络强大的指令遵循能力对于通用嵌入模型的重要性。

### 6.2.4. 元任务泛化能力
下图（原文 Figure 5）展示了仅在单一元任务上训练的模型在其他未见过的元任务上的泛化能力：

![Figure 5: The figures show the generalization ability of models trained on one meta-task to other unseen meta-tasks. For example, the first subplot compares the performance of ${ \\tt V I M 2 V E C }$ trained exclusively on VQA datasets with ${ \\tt V I M 2 V E C }$ trained exclusively on retrieval datasets across the other two meta-task categories: classification and visual grounding. Overall, ${ \\tt V I M 2 V E C }$ trained on retrieval tasks demonstrate better generalization ability because retrieval tasks involve a more diverse combination of text and visual modalities from both the query and target sides. ${ \\tt V I M 2 V E C }$ utilizes Phi-3.5-V as its backbone.](images/5.jpg)
*该图像是一个条形图，展示了不同任务下 VLM2Vec 模型的性能对比。左侧图显示了 VQA 数据集与检索数据集模型在分类和视觉定位任务上的表现；中间图展示了检索任务的成果；右侧图则对比了检索与分类模型在视觉定位任务上的表现。数据结果表明，检索任务的模型在多项任务上表现优越。*

结果表明，仅在检索任务上训练的模型（VLM2Vec_RET）在其他未见任务（如分类、视觉定位）上表现出更好的泛化能力。这是因为检索任务涉及查询端和目标端更多样的模态组合，有助于模型学习更通用的表示。

# 7. 总结与思考

## 7.1. 结论总结
论文成功构建了首个大规模多模态嵌入框架，包括 MMEB 基准和 VLM2Vec 模型。MMEB 提供了全面的评估环境，而 VLM2Vec 证明了利用 VLM 作为主干网络，通过对比学习和指令微调，可以构建出性能卓越、泛化能力强的通用多模态嵌入模型。研究揭示，VLMs 不仅是强大的生成或理解模型，也是“隐藏的”强大嵌入模型。

## 7.2. 局限性与未来工作
论文指出了当前的局限性，例如在处理高分辨率图像和长文本时的计算成本较高。此外，MMEB 候选集数量的选择（1000个）是在评估成本和难度之间的折衷，未来可能需要更难的评估设置。未来的工作可以探索更高效的训练方法、扩展到更多模态（如视频、音频），以及进一步优化指令设计。

## 7.3. 个人启发与批判
这篇论文对多模态领域具有重要的启发意义。它打破了传统双塔模型（如 CLIP）在嵌入任务上的主导地位，展示了单塔深度融合模型（VLM）的潜力。这提示我们，在追求通用表征时，利用具备强大推理和指令遵循能力的大模型作为基础，往往比专门设计的简单架构更有效。此外，MMEB 基准的提出为该领域提供了统一的标尺，将推动后续研究的发展。一个潜在的批判点是，VLM2Vec 的推理成本可能高于轻量级的双塔模型，因此在资源受限的边缘设备上部署可能面临挑战。