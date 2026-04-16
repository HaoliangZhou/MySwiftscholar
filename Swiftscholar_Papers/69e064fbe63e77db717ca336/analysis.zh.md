# 1. 论文基本信息

## 1.1. 标题
**PDV: Prompt Directional Vectors for Zero-shot Composed Image Retrieval** (PDV：用于零样本组合图像检索的提示方向向量)

## 1.2. 作者
**Osman Tursun** (昆士兰科技大学), **Sinan Kalkan** (中东技术大学), **Simon Denman** (昆士兰科技大学), **Clinton Fookes** (昆士兰科技大学)

## 1.3. 发表期刊/会议
**arXiv** (计算机视觉与模式识别预印本库)。虽然尚未正式发表于顶级会议，但作为该领域最新的研究成果，具有重要的参考价值。

## 1.4. 发表年份
**2025年** (发布于 2025-02-11)

## 1.5. 摘要
零样本组合图像检索 (ZS-CIR) 允许用户使用参考图像和文本提示来搜索图像，而无需在大规模配对数据上训练专门的文本-图像组合网络。然而，当前的 ZS-CIR 方法在依赖组合文本嵌入方面存在三个关键局限性：静态的查询嵌入表示、图像嵌入利用不足以及文本和图像嵌入融合时的次优性能。为了解决这些挑战，本文介绍了 **提示方向向量**，这是一种简单且有效的免训练增强方法，用于捕获用户提示引起的语义修改。PDV 实现了三个关键改进：(1) 动态组合文本嵌入，其中提示调整可通过缩放因子控制；(2) 通过语义转移将文本提示转移到图像特征来组合图像嵌入；(3) 组合文本和图像嵌入的加权融合，通过平衡视觉和语义相似性来增强检索。该方法作为现有 ZS-CIR 方法的即插即用增强功能，计算开销极小。在多个基准测试中的广泛实验表明，当与最先进的 ZS-CIR 方法集成时，PDV 始终提高检索性能，特别是对于能够生成准确组合嵌入的方法。

## 1.6. 原文链接
**PDF 链接:** https://arxiv.org/pdf/2502.07215v3

---

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 核心问题
组合图像检索 (Composed Image Retrieval, CIR) 旨在通过结合参考图像和描述修改内容的文本提示来检索目标图像。传统的 CIR 方法通常需要昂贵的标注数据（参考图、目标图、修改文本）来训练模型。为了降低成本，零样本 CIR (ZS-CIR) 利用预训练的视觉-语言模型 (VLMs) 如 CLIP，通过将图像和文本组合成一个查询嵌入来实现检索。然而，现有的 ZS-CIR 方法面临三大瓶颈：
1.  **静态嵌入表示**：一旦生成组合文本嵌入，它是固定的。如果检索结果不理想，用户必须修改提示并重新提取特征，无法直接微调现有嵌入。
2.  **图像嵌入利用不足**：参考图像的视觉嵌入通常仅用于生成组合文本，而不直接参与检索计算，导致视觉信息的浪费。
3.  **次优的融合性能**：简单的图像和文本嵌入直接融合往往不如生成的组合文本嵌入效果好，因为直接文本嵌入缺乏参考图像的上下文信息。

### 2.1.2. 创新思路
本文提出了一种名为 <strong>提示方向向量 (PDV)</strong> 的免训练方法。其核心思想是将用户提示引起的语义变化视为一个向量（方向向量）。通过计算包含提示的文本嵌入与不包含提示的文本嵌入之间的残差，得到这个方向向量。然后，利用这个向量来动态调整文本嵌入、增强图像嵌入，并进行加权融合。这种方法不需要任何额外的训练，可以直接作为插件应用于现有的 ZS-CIR 方法。

## 2.2. 核心贡献/主要发现
1.  **提出 PDV 概念**：定义了提示方向向量 $\Delta_{PDV}$，用于量化提示引起的语义变化方向。
2.  **三种应用策略**：
    *   **PDV-T**：通过缩放因子 $\alpha$ 动态调整组合文本嵌入，实现无需重写提示的检索结果微调。
    *   **PDV-I**：将 PDV 加到参考图像的视觉嵌入上，实现语义到视觉的转移，生成组合图像嵌入。
    *   **PDV-F**：融合组合文本嵌入和组合图像嵌入，通过权重 $\beta$ 平衡视觉相似性和语义一致性。
3.  **即插即用与高效**：PDV 是一个计算开销极小的后处理模块，广泛适用于多种基线方法（如 CIReVL, Pic2Word 等），并在 Fashion-IQ, CIRR, CIRCO 等数据集上显著提升了性能。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，我们需要掌握以下核心概念：
*   <strong>零样本组合图像检索 (ZS-CIR)</strong>：一种不需要针对特定任务训练数据即可执行的图像检索任务。给定一个参考图像 $I_{ref}$ 和一个文本提示 $P$（例如“把红色变成蓝色”），目标是找到符合修改描述的目标图像 $I_{target}$。
*   <strong>视觉-语言模型 (VLM)</strong>：如 CLIP，这类模型通过在大规模图像-文本对上进行对比学习，将图像和文本映射到同一个共享的嵌入空间。在这个空间中，语义相似的图像和文本距离更近。
*   **嵌入**：将高维数据（如图像或文本）转换为数值向量的过程。在检索任务中，通常通过计算向量之间的余弦相似度来匹配查询和数据库项。
*   **文本反转**：一种将图像映射为文本空间中“伪词元”的技术，常用于 ZS-CIR 中，以便将图像信息融入文本编码器。

## 3.2. 前人工作
*   **监督式 CIR**：早期方法如 TIRG [26] 和 ARTEMIS [9] 依赖于标注的三元组数据来训练组合网络。虽然性能较好，但数据收集成本极高。
*   **零样本 CIR**：
    *   **基于文本反转的方法**：如 Pic2Word [22] 和 SEARLE [3]，通过学习将图像映射为文本词元，从而与提示结合。
    *   **基于图像描述的方法**：如 CIReVL [14]，首先使用大模型生成参考图像的描述，再与提示结合。
    *   这些方法虽然避免了昂贵的标注，但通常只关注初始检索，缺乏对结果的动态控制能力。
*   **残差学习**：在监督学习中，Vo et al. [26] 曾使用 LSTM 学习提示引起的残差变化。本文的 PDV 在概念上与之类似，但完全依赖于预训练 VLM 的固有空间，无需监督训练。

## 3.3. 技术演进
该领域从依赖昂贵标注的监督学习，发展到利用预训练模型和合成数据的零样本学习。早期的零样本方法主要关注如何更好地将图像信息“翻译”成文本信息（如 Pic2Word）。近期的研究（如 CIReVL）开始探索利用大语言模型 (LLM) 的生成能力来处理组合逻辑。本文的工作在此基础上，进一步挖掘了嵌入空间中的几何关系（方向向量），为现有的零样本方法提供了一个通用的性能增强层。

## 3.4. 差异化分析
与现有的 ZS-CIR 方法相比，PDV 不试图发明一种新的图像-文本组合机制，而是专注于**改进**现有方法生成的嵌入。它不改变特征提取过程，而是通过后处理（向量运算）来增强嵌入的表达能力和可控性。与 CompoDiff [12] 等同样提供可控性的方法不同，PDV 是模型无关的，不依赖特定的生成式架构（如 Diffusion），因此适用范围更广。

---

# 4. 方法论

## 4.1. 方法原理
PDV 的核心原理基于向量空间中的几何直觉。在视觉-语言模型的共享嵌入空间中，图像和文本都被表示为向量。作者认为，用户输入的提示 $P$ 引起了一个从“参考图像状态”到“目标图像状态”的语义位移。这个位移可以用一个向量 $\Delta_{PDV}$ 来表示。一旦捕获了这个方向向量，我们就可以：
1.  **缩放**它来控制修改的强度。
2.  将它**加**到图像嵌入上，让图像也“理解”这个修改。
3.  将修改后的图像嵌入和文本嵌入进行**融合**，结合两者的优势。

    下图展示了 PDV 方法在零样本复合图像检索中的整体应用框架，包括参考图像嵌入、复合文本嵌入及目标特征之间的关系，以及通过 PDV 进行的动态调节。

    ![该图像是示意图，展示了提出的“Prompt Directional Vector (PDV)”方法在零-shot复合图像检索中的应用。图中标识了参考图像嵌入、复合文本嵌入及目标特征之间的关系，并说明了通过PDV在视觉空间中进行动态调节的重要性。图(a)至(d)展示了文本提示对图像嵌入的影响及其在检索过程中的权重融合，强调了PDV在提升检索性能方面的作用。](images/1.jpg)
    *该图像是示意图，展示了提出的“Prompt Directional Vector (PDV)”方法在零-shot复合图像检索中的应用。图中标识了参考图像嵌入、复合文本嵌入及目标特征之间的关系，并说明了通过PDV在视觉空间中进行动态调节的重要性。图(a)至(d)展示了文本提示对图像嵌入的影响及其在检索过程中的权重融合，强调了PDV在提升检索性能方面的作用。*

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 基线 ZS-CIR 框架
首先，我们需要理解标准的 ZS-CIR 流程。设 $\Psi$ 为一个预训练的 VLM，包含视觉分支 $\Psi_I$ 和文本分支 $\Psi_T$。给定参考图像 $I_{ref}$ 和提示 $P$，系统通过某种组合函数 $\mathcal{F}$（如描述生成或伪词元化）生成一个文本查询表示。检索的目标是在数据库 $\mathcal{D}$ 中找到与该查询最相似的目标图像。

形式化地，检索过程定义为寻找余弦相似度最高的前 $k$ 个图像：

$$
\mathbb { I } _ { t o p - k } = \underset { I \in \mathcal { D } } { \arg \operatorname* { m a x } } _ { \boldsymbol { \mathsf { k } } } \frac { \Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) ^ { T } \cdot \Psi _ { I } ( I ) } { \| \Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) \| \cdot \| \Psi _ { I } ( I ) \| } .
$$

*   $\mathbb{I}_{top-k}$：检索到的 Top-k 目标图像集合。
*   $\mathcal{D}$：图像数据库。
*   $\Psi_T(\mathcal{F}(I_{ref}, P))$：组合文本嵌入，即查询向量。
*   $\Psi_I(I)$：数据库中图像 $I$ 的视觉嵌入。
*   分子部分表示查询向量与图像向量的点积，分母是它们的模长乘积，整体计算余弦相似度。

    在这个基线框架中，只有组合文本嵌入参与检索，参考图像的原始视觉嵌入 $\Psi_I(I_{ref})$ 在检索阶段被丢弃了。

### 4.2.2. 提示方向向量 (PDV) 的定义
为了捕捉提示引起的语义变化，作者定义了 **提示方向向量** $\Delta_{PDV}$。它是“包含提示的组合文本嵌入”与“不包含提示的参考图像文本嵌入”之间的残差向量。

$$
\Delta _ { P D V } = \Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) - \Psi _ { T } ( \mathcal { F } ( I _ { r e f } ) ) .
$$

*   $\Psi_T(\mathcal{F}(I_{ref}, P))$：加入了提示 $P$ 后的文本嵌入。
*   $\Psi_T(\mathcal{F}(I_{ref}))$：仅基于参考图像 $I_{ref}$ 生成的文本嵌入（相当于 $P$ 为空字符串时的基线）。
*   $\Delta_{PDV}$：表示从参考图像语义移动到目标语义的方向和距离。

### 4.2.3. 动态组合文本嵌入 (PDV-T)
有了 $\Delta_{PDV}$，我们可以重写组合文本嵌入的生成公式。原始的嵌入可以看作是基线向量加上一个单位的 PDV。现在，我们引入一个缩放因子 $\alpha_T$ 来控制这个移动的幅度。

$$
\Psi _ { T } ( \mathcal { F } ( I _ { r e f } , P ) ) = \Psi _ { T } ( \mathcal { F } ( I _ { r e f } ) ) + \alpha _ { T } \Delta _ { P D V } ,
$$

*   $\alpha_T$：文本提示缩放因子。当 $\alpha_T = 1$ 时，等同于原始方法；$\alpha_T > 1$ 会放大提示的语义修改效果；$\alpha_T < 1$ 则减弱效果。

    **原理分析**：如下图所示，PDV 的有效性取决于计算出的 $\Delta_{PDV}$ 与真实语义变化方向（Ground Truth 方向）之间的夹角 $\phi$。如果 $\phi$ 较小（即方向准确），调整 $\alpha$ 可以有效地减小查询向量与目标图像向量之间的夹角 $\theta$，从而提高检索精度。

    ![该图像是示意图，展示了 `heta` 和 `oldsymbol{ ho}` 的可视化 (图(a)) 以及 `heta` 与 `oldsymbol{ ho}` 的关系 (图(b))。左侧图中的绿色点表示调整后的文本嵌入方向，而右侧图为角度 `heta` 与 `oldsymbol{ ho}` 的关系图。](images/2.jpg)
    *该图像是示意图，展示了 `heta` 和 `oldsymbol{ ho}` 的可视化 (图(a)) 以及 `heta` 与 `oldsymbol{ ho}` 的关系 (图(b))。左侧图中的绿色点表示调整后的文本嵌入方向，而右侧图为角度 `heta` 与 `oldsymbol{ ho}` 的关系图。*

### 4.2.4. 组合图像嵌入 (PDV-I)
为了利用被丢弃的参考图像视觉嵌入 $\Psi_I(I_{ref})$，作者提出将 PDV 加到图像嵌入上。这相当于在视觉空间中应用相同的语义修改。

$$
\Phi _ { P D V - I } = \Psi _ { I } ( I _ { r e f } ) + \alpha _ { I } \Delta _ { P D V } ,
$$

*   $\Phi_{PDV-I}$：经过 PDV 增强的组合图像嵌入。
*   $\alpha_I$：图像提示缩放因子，控制视觉特征受提示影响的程度。

    下图对比了传统的“Image + Text”直接拼接与本文提出的 PDV-I 方法。传统方法直接拼接图像特征和独立的文本特征，往往效果不佳；而 PDV-I 使用的是上下文相关的 $\Delta_{PDV}$，能更准确地将语义转移到视觉空间。

    ![Figure 3. Comparison of Image $^ +$ Text (a) vs PDV-I (b).](images/3.jpg)
    *该图像是示意图，比较了图(a)中的“Image + Text”与图(b)中的“PDV”。图(a)展示了图像和文本的组合表示，其中包含了参考图像 $I_{ref}$、目标图像 $I_{target}$ 和文本提示 $P$ 的向量表示。图(b)则展示了Prompt Directional Vector (PDV)的应用，通过引入PDV，文本和图像的组合表示得以优化。绿色和粉色的点分别表示不同的语义修改，展示了如何通过PDV增强检索能力。*

### 4.2.5. 加权融合 (PDV-F)
最后，为了结合视觉和语义的优势，作者将 PDV-T 生成的文本嵌入和 PDV-I 生成的图像嵌入进行加权融合。

$$
\Phi _ { P D V - F } = ( 1 - \beta ) \Phi _ { P D V - I } + \beta \Phi _ { P D V - T } ,
$$

*   $\Phi_{PDV-F}$：最终用于检索的融合嵌入。
*   $\beta$：融合权重因子，范围 `[0, 1]`。
    *   $\beta \to 1$：更依赖文本语义（PDV-T）。
    *   $\beta \to 0$：更依赖视觉相似性（PDV-I）。

### 4.2.6. 算法实现
以下算法总结了计算 PDV 特征的完整流程：

**Algorithm 1 Calculate PDV Features**

1: function CALCULATEPDVFEATURES($f_{text}, f_{text\_composed}, f_{image}, \alpha_i, \alpha_t, \beta$)
2: &nbsp;&nbsp;$f_{text} \leftarrow \text{normalize}(f_{text})$
3: &nbsp;&nbsp;$f_{text\_composed} \leftarrow \text{normalize}(f_{text\_composed})$
4: &nbsp;&nbsp;$f_{image} \leftarrow \text{normalize}(f_{image})$
5: &nbsp;&nbsp;$pdv \leftarrow f_{text\_composed} - f_{text}$
6: &nbsp;&nbsp;$f_{PDV-I} \leftarrow f_{image} + \alpha_i \cdot pdv$
7: &nbsp;&nbsp;$f_{PDV-T} \leftarrow f_{text} + \alpha_t \cdot pdv$
8: &nbsp;&nbsp;$f_{PDV-F} \leftarrow (1 - \beta) \cdot f_{PDV-I} + \beta \cdot f_{PDV-T}$
9: &nbsp;&nbsp;return normalize($f_{PDV-F}$)
10: end function

---

# 5. 实验设置

## 5.1. 数据集
实验在三个标准的 CIR 基准数据集上进行：
*   **Fashion-IQ**：一个时尚图像检索数据集，包含衬衫、连衣裙和上衣三个类别。查询由参考图像和相对修改文本组成（如“更短”、“条纹裙”）。
*   **CIRR**：组合图像检索基准，包含多样化的场景和物体。
*   **CIRCO**：专注于组合图像检索的挑战性数据集，包含复杂的组合关系。

    这些数据集涵盖了从特定领域（时尚）到通用领域的场景，能够全面评估方法的泛化能力。

## 5.2. 评估指标
论文主要使用以下指标：
1.  **Recall@k**
    *   **概念定义**：召回率，指在前 $k$ 个检索结果中出现正确目标图像的概率。它衡量的是检索系统在前几个结果中找到正确项的能力。
    *   **数学公式**：
        $$ \text{Recall@k} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbb{I}(\text{rank}(q, I_{target}) \le k) $$
    *   **符号解释**：$\mathcal{Q}$ 是查询集合，$\text{rank}(q, I_{target})$ 是目标图像在查询 $q$ 的结果列表中的排名，$\mathbb{I}(\cdot)$ 是指示函数，如果条件满足则为 1 否则为 0。
2.  **mAP@k**
    *   **概念定义**：平均精度均值，在 CIRCO 数据集上使用。它综合考虑了排序位置，对排在前面的正确结果给予更高的权重。
    *   **数学公式**：
        $$ \text{mAP} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \left( \frac{1}{m_q} \sum_{i=1}^{m_q} \text{Precision}@i \right) $$
    *   **符号解释**：$m_q$ 是查询 $q$ 相关的图像总数，$\text{Precision}@i$ 是前 $i$ 个结果的精度。

## 5.3. 对比基线
作者选择了四种代表性的 ZS-CIR 方法作为基线：
*   **CIReVL**：基于图像描述生成的零样本方法。
*   **LDRE**：基于 LLM 推理的多样化描述生成方法。
*   **Pic2Word**：基于伪词元化的经典方法。
*   **SEARLE**：Pic2Word 的改进版，降低了训练成本。

    这些基线涵盖了基于描述和基于词元化两大主流技术路线，具有很好的代表性。

---

# 6. 实验结果与分析

## 6.1. 核心结果分析
### 6.1.1. Fashion-IQ 数据集结果
以下是原文 Table 1 的结果，展示了在 Fashion-IQ 数据集上，不同基线方法结合 PDV-F 后的性能提升。表中 Orange 高亮部分表示 PDV 带来的改进。

<table>
<thead>
<tr>
<th colspan="5">Fashion-IQ</th>
<th colspan="2">Shirt</th>
<th colspan="2">Dress</th>
<th colspan="2">Toptee</th>
<th colspan="2">Average</th>
</tr>
<tr>
<th>Backbone</th>
<th>Method</th>
<th>β</th>
<th>αI</th>
<th>αT</th>
<th>R@10</th>
<th>R@50</th>
<th>R @ 10</th>
<th>R@50</th>
<th>R @ 10</th>
<th>R@50</th>
<th>R @ 10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="6">ViT-B/32</td>
<td>SEARLE</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>24.14</td>
<td>41.81</td>
<td>18.39</td>
<td>38.08</td>
<td>25.91</td>
<td>47.02</td>
<td>22.81</td>
<td>42.30</td>
</tr>
<tr>
<td>SEARLE + PDV-F</td>
<td>0.9</td>
<td>1.1</td>
<td>0.9</td>
<td>24.83</td>
<td>41.71</td>
<td>20.13</td>
<td>41.40</td>
<td>25.96</td>
<td>47.17</td>
<td>23.64</td>
<td>43.43</td>
</tr>
<tr>
<td>CIReVL</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>28.36</td>
<td>4747.84</td>
<td>25.29</td>
<td>46.36</td>
<td>31.21</td>
<td>53.85</td>
<td>28.29</td>
<td>49.35</td>
</tr>
<tr>
<td>CIReVL + PDV-F</td>
<td>0.75</td>
<td>1.4</td>
<td>1.4</td>
<td>32.88</td>
<td>52.80</td>
<td>32.67</td>
<td>54.49</td>
<td>38.91</td>
<td>61.81</td>
<td>34.82</td>
<td>56.37</td>
</tr>
<tr>
<td>LDRE</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>27.38</td>
<td>46.27</td>
<td>19.97</td>
<td>41.84</td>
<td>27.07</td>
<td>48.78</td>
<td>24.81</td>
<td>45.63</td>
</tr>
<tr>
<td>SEIZE</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>29.38</td>
<td>47.97</td>
<td>25.37</td>
<td>46.84</td>
<td>32.07</td>
<td>54.78</td>
<td>28.94</td>
<td>49.86</td>
</tr>
<tr>
<td rowspan="8">ViT-L/14</td>
<td>Pic2Word</td>
<td></></td><td></td><td></td>
<td>25.96</td>
<td>43.52</td>
<td>19.63</td>
<td>40.90</td>
<td>27.28</td>
<td>47.83</td>
<td>24.29</td>
<td>44.08</td>
</tr>
<tr>
<td>Pic2Word + PDV-F</td>
<td>0.8</td>
<td>1.0</td>
<td>1.0</td>
<td>28.21</td>
<td>44.55</td>
<td>20.92</td>
<td>42.24</td>
<td>29.02</td>
<td>48.90</td>
<td>26.05</td>
<td>45.23</td>
</tr>
<tr>
<td>SEARLE</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>26.84</td>
<td>45.19</td>
<td>20.08</td>
<td>42.19</td>
<td>28.40</td>
<td>49.62</td>
<td>25.11</td>
<td>45.67</td>
</tr>
<tr>
<td>SEARLE +PDV-F</td>
<td>0.8</td>
<td>1.2</td>
<td>1.0</td>
<td>28.66</td>
<td>46.76</td>
<td>23.60</td>
<td>46.41</td>
<td>31.00</td>
<td>52.32</td>
<td>27.75</td>
<td>48.50</td>
</tr>
<tr>
<td>CIReVL</td>
<td></td><td></td><td></td>
<td>29.49</td>
<td>47.40</td>
<td>24.79</td>
<td>44.76</td>
<td>31.36</td>
<td>53.65</td>
<td>28.55</td>
<td>48.57</td>
</tr>
<tr>
<td>CIReVL + PDV-F</td>
<td>0.55</td>
<td>1</td>
<td>1.3</td>
<td>37.78</td>
<td>54.22</td>
<td>33.61</td>
<td>56.07</td>
<td>41.61</td>
<td>62.16</td>
<td>37.67</td>
<td>57.48</td>
</tr>
<tr>
<td>LinCIR</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>29.10</td>
<td>46.81</td>
<td>20.92</td>
<td>42.44</td>
<td>28.81</td>
<td>50.18</td>
<td>26.82</td>
<td>46.49</td>
</tr>
<tr>
<td>SEIZE</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>33.04</td>
<td>53.22</td>
<td>30.93</td>
<td>50.76</td>
<td>35.57</td>
<td>58.64</td>
<td>33.18</td>
<td>54.21</td>
</tr>
<tr>
<td rowspan="6">ViT-G/14</td>
<td>Pic2Word</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>33.17</td>
<td>50.39</td>
<td>25.43</td>
<td>47.65</td>
<td>35.24</td>
<td>57.62</td>
<td>31.28</td>
<td>51.89</td>
</tr>
<tr>
<td>SEARLE</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>36.46</td>
<td>55.35</td>
<td>28.16</td>
<td>50.32</td>
<td>39.83</td>
<td>61.45</td>
<td>34.81</td>
<td>55.71</td>
</tr>
<tr>
<td>CIReVL</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>33.71</td>
<td>51.42</td>
<td>27.07</td>
<td>49.53</td>
<td>35.80</td>
<td>56.14</td>
<td>32.19</td>
<td>52.36</td>
</tr>
<tr>
<td>CIReVL + PDV-F</td>
<td>0.6</td>
<td>1.4</td>
<td>1.4</td>
<td>41.90</td>
<td>58.19</td>
<td>40.70</td>
<td>62.82</td>
<td>48.09</td>
<td>67.77</td>
<td>43.56</td>
<td>62.93</td>
</tr>
<tr>
<td>LinCIR</td>
<td>:</td>
<td>-</td>
<td></td>
<td>46.76</td>
<td>65.11</td>
<td>38.08</td>
<td>60.88</td>
<td>50.48</td>
<td>71.09</td>
<td>45.11</td>
<td>65.69</td>
</tr>
<tr>
<td>SEIZE</td>
<td></td>
<td>-</td>
<td>:</td>
<td>43.60</td>
<td>65.42</td>
<td>39.61</td>
<td>61.02</td>
<td>45.94</td>
<td>71.12</td>
<td>43.05</td>
<td>65.5</td>
</tr>
</tbody>
</table>

**分析**：
*   PDV-F 在所有基线方法上均带来了显著的性能提升。例如，在 ViT-B/32 主干网络上，CIReVL + PDV-F 在平均 Recall@10 上从 28.29% 提升到了 34.82%。
*   随着主干网络能力的增强（从 ViT-B/32 到 ViT-G/14），PDV 的增益效果依然明显，甚至在 CIReVL 上达到了约 10% 的提升。
*   这表明 PDV 能够有效地修正基线方法生成的嵌入方向，使其更接近真实目标。

### 6.1.2. CIRCO 和 CIRR 数据集结果
以下是原文 Table 2 的结果，展示了在 CIRCO 和 CIRR 数据集上的表现。

<table>
<thead>
<tr>
<th colspan="5">Dataset</th>
<th colspan="4">CIRCO</th>
<th colspan="8">CIRR</th>
</tr>
<tr>
<th colspan="5">Metric</th>
<th colspan="4">mAP@k</th>
<th colspan="5">Recall@k</th>
<th colspan="3">Rs@k</th>
</tr>
<tr>
<th>Arch</th>
<th>Method</th>
<th>β</th>
<th>αI</th>
<th>αT</th>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=50</th>
<th>k=1</th>
<th></th>
<th>k=5</th>
<th>k=10</th>
<th>k=50</th>
<th>k=1</th>
<th>k=2</th>
<th>k=3</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">ViT-B/32</td>
<td>PALAVRA [8] </td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td>4.61</td>
<td>5.32</td>
<td>6.33</td>
<td>6.80</td>
<td>16.62</td>
<td>43.49</td>
<td>58.51</td>
<td>83.95</td>
<td>41.61</td>
<td>65.30</td>
<td>80.94</td>
</tr>
<tr>
<td>SEARLE</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td>9.35</td>
<td>9.94</td>
<td>11.13</td>
<td>11.84</td>
<td>24.00</td>
<td>53.42</td>
<td>66.82</td>
<td>89.78</td>
<td>54.89</td>
<td>76.60</td>
<td>88.19</td>
</tr>
<tr>
<td>SEARLE + PDV-F</td>
<td>0.9</td>
<td>1.4</td>
<td>1.2</td>
<td></td>
<td>9.99</td>
<td>10.50</td>
<td>11.70</td>
<td>12.40</td>
<td></td>
<td>24.53</td>
<td>53.71</td>
<td>67.33</td>
<td>89.81</td>
<td>56.94</td>
<td>78.05</td>
<td>88.99</td>
</tr>
<tr>
<td>CIReVL</td>
<td></td>
<td>- -</td>
<td>-</td>
<td></td>
<td>14.94</td>
<td>15.42</td>
<td>17.00</td>
<td>17.82</td>
<td>23.94</td>
<td>52.51</td>
<td>66.00</td>
<td>86.95</td>
<td>60.17</td>
<td>80.05</td>
<td>90.19</td>
</tr>
<tr>
<td>CIReVL + PDV-F</td>
<td></td>
<td>0.75</td>
<td>1.4</td>
<td>1.2</td>
<td>19.90</td>
<td>20.61</td>
<td>22.64</td>
<td>23.52</td>
<td>33.25</td>
<td>64.15</td>
<td>75.23</td>
<td>92.43</td>
<td>65.81</td>
<td>83.76</td>
<td>92.10</td>
</tr>
<tr>
<td>LDRE</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td></td>
<td>17.81</td>
<td>18.04</td>
<td>19.73</td>
<td>20.67</td>
<td>25.69</td>
<td>55.52</td>
<td>68.77</td>
<td>89.86</td>
<td>60.10</td>
<td>80.58</td>
<td>91.04</td>
</tr>
<tr>
<td>LDRE + PDV-F</td>
<td>0.75</td>
<td>1.4</td>
<td>1.4</td>
<td></td>
<td>17.80</td>
<td>18.78</td>
<td>20.61</td>
<td>21.56</td>
<td>29.30</td>
<td>60.39</td>
<td>72.51</td>
<td>91.42</td>
<td>63.06</td>
<td>82.36</td>
<td>91.54</td>
</tr>
<tr>
<td rowspan="13"></td>
<td>SEIZE</td>
<td>-</td>
<td>- -</td>
<td>-</td>
<td>19.04</td>
<td>19.64</td>
<td>21.55</td>
<td>22.49</td>
<td>27.47</td>
<td>57.42</td>
<td>70.17</td>
<td></td>
<td>-</td>
<td>65.59</td>
<td>84.48</td>
<td>92.77</td>
</tr>
<tr>
<td>Pic2Word Pic2Word + PDV-F</td>
<td>-</td>
<td></td>
<td>- 1.0</td>
<td></td>
<td>6.81</td>
<td>7.49</td>
<td>8.51</td>
<td>9.07</td>
<td>23.69</td>
<td>51.32</td>
<td>63.66</td>
<td>86.21</td>
<td>53.61</td>
<td>74.34</td>
<td>87.28</td>
</tr>
<tr>
<td>SEARLE</td>
<td></td>
<td>0.85 -</td>
<td>1.2 -</td>
<td></td>
<td>7.74</td>
<td>8.67</td>
<td>12.73</td>
<td>9.77</td>
<td>10.37</td>
<td>23.90</td>
<td>51.95</td>
<td>64.63</td>
<td>87.04</td>
<td>53.16</td>
<td>74.07</td>
<td>87.08</td>
</tr>
<tr>
<td>SEARLE + PDV-F</td>
<td></td>
<td></td>
<td>-</td>
<td></td>
<td>11.68</td>
<td>12.58</td>
<td>13.57</td>
<td>14.33</td>
<td>15.12</td>
<td>24.24</td>
<td>52.48</td>
<td>66.29</td>
<td>88.84</td>
<td>53.76</td>
<td>75.01</td>
<td>88.19</td>
</tr>
<tr>
<td></td>
<td>0.85</td>
<td>1.4</</td>
<td>1.2</td>
<td></td>
<td></td>
<td></td>
<td>15.30</td>
<td>16.07</td>
<td>25.64</td>
<td>53.61</td>
<td>66.58</td>
<td>88.55</td>
<td>55.83</td>
<td>76.48</td>
<td>88.53</td>
</tr>
<tr>
<td>CIReVL</td>
<td>-</td>
<td>-</td>
<td>- 1.2</td>
<td></td>
<td>18.57</td>
<td>25.67</td>
<td>19.01</td>
<td>26.61</td>
<td>20.89</td>
<td>28.81</td>
<td>21.80</td>
<td>24.55</td>
<td>52.31</td>
<td>64.92</td>
<td>86.34</td>
<td>59.54</td>
<td>79.88</td>
<td>89.69</td>
</tr>
<tr>
<td>ViT-L/14 CIReVL + PDV-F LDRE</td>
<td>0.75</td>
<td>1.4 -</td>
<td>-</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>29.95</td>
<td>36.24</td>
<td>66.17</td>
<td>76.96</td>
<td>92.29</td>
<td>68.07</td>
<td>85.35</td>
<td>93.47</td>
</tr>
<tr>
<td>LDRE + PDV-F</td>
<td></td>
<td>- 0.75</td>
<td></td>
<td>1.4</td>
<td>22.32</td>

<      <td>25.23</td>
      <td>23.75</td>
      <td>26.52</td>
      <td>25.97</td>
      <td>28.94</td>
      <td>27.03</td>
      <td>29.95</td>
      <td>26.68</td>
      <td>55.45</td>
      <td>59.98</td>
      <td>67.49</td>
      <td>71.90</td>
      <td>88.65</td>
      <td>60.39</td>
      <td>80.53</td>
      <td>90.15</td>
    </tr>
    <tr>
      <td>LinCIR</td>
      <td></td>
      <td></td>
      <td>1.4 -</td>
      <td>-</td>
      <td>12.59</td>
      <td>13.58</td>
      <td>15.00</td>
      <td>15.85</td>
      <td>30.16</td>
      <td>25.04</td>
      <td>53.25</td>
      <td>66.68</td>
      <td>90.87</td>
      <td>63.66</td>
      <td>82.87</td>
      <td>91.57</td>
    </tr>
    <tr>
      <td>SEIZE</td>
      <td></td>
      <td>- -</td>
      <td>-</td>
      <td>-</td>
      <td>24.98</td>
      <td>25.82</td>
      <td>28.24</td>
      <td>29.35</td>
      <td>28.65</td>
      <td>57.16</td>
      <td>69.23</td>
      <td>- -</td>
      <td>57.11</td>
      <td>66.22</td>
      <td>77.37</td>
      <td>84.05</td>
      <td>88.89</td>
      <td>92.34</td>
    </tr>
    <tr>
      <td rowspan="7">ViT-G/14</td>
      <td>CIReVL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td></td>
      <td>26.77</td>
      <td>27.59</td>
      <td>29.96</td>
      <td>31.03</td>
      <td>34.65</td>
      <td>64.29</td>
      <td>75.06</td>
      <td>91.66</td>
      <td>67.95</td>
      <td>84.87</td>
      <td></td>
    </tr>
    <tr>
      <td>CIReVL + PDV-F</td>
      <td>0.75</td>
      <td>1.4</td>
      <td>1.2</td>
      <td>30.02</td>
      <td>31.46</td>
      <td>34.01</td>
      <td>35.08</td>
      <td>38.15</td>
      <td>67.93</td>
      <td>77.90</td>
      <td>92.77</td>
      <td></td>
      <td>69.37</td>
      <td>85.37</td>
      <td>93.21</td>
      <td>93.45</td>
    </tr>
    <tr>
      <td>LDRE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>33.30</td>
      <td>34.32</td>
      <td>37.17</td>
      <td>38.27</td>
      <td>37.40</td>
      <td>66.96</td>
      <td>78.17</td>
      <td>93.66</td>
      <td>68.84</td>
      <td>85.64</td>
      <td>93.90</td>
    </tr>
    <tr>
      <td>LDRE + PDV-F</td>
      <td>0.75</td>
      <td>1.4</td>
      <td>1.4</td>
      <td>34.88</td>
      <td>36.41</td>
      <td>39.12</td>
      <td>40.23</td>
      <td>42.51</td>
      <td>72.22</td>
      <td>81.71</td>
      <td>94.94</td>
      <td>72.39</td>
      <td>88.34</td>
      <td>94.80</td>
    </tr>
    <tr>
      <td>SEARLE</td>
      <td>-</td>
      <td></td>
      <td>-</td>
      <td>13.20</td>
      <td>13.85</td>
      <td>15.32</td>
      <td>16.04</td>
      <td>34.80</td>
      <td>64.07</td>
      <td>75.11</td>
      <td>,</td>
      <td>68.72</td>
      <td>84.70</td>
      <td>93.23</td>
    </tr>
    <tr>
      <td>LinCIR</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>19.71</td>
      <td>21.01</td>
      <td>23.13</td>
      <td>24.18</td>
      <td>35.25</td>
      <td>64.72</td>
      <td>76.05</td>
      <td>-</td>
      <td>63.35</td>
      <td>82.22</td>
      <td>91.98</td>
    </tr>
    <tr>
      <td>SEIZE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>32.46</td>
      <td>33.77</td>
      <td>36.46</td>
      <td>37.55</td>
      <td>38.87</td>
      <td>69.42</td>
      <td>79.42</td>
      <td>-</td>
      <td>74.15</td>
      <td>89.23</td>
      <td>95.71</td>
    </tr>
  </tbody>
</table>

**分析**：
*   在 CIRCO 数据集上，CIReVL + PDV-F 在 mAP@5 上从 14.94 提升到了 19.90，提升显著。
*   在 CIRR 数据集上，Recall@1 也有明显提升，说明 PDV 有助于将正确结果排到更靠前的位置。

## 6.2. 消融实验/参数分析
作者详细分析了缩放因子 $\alpha$ 和融合因子 $\beta$ 对性能的影响。

### 6.2.1. PDV-T (文本缩放) 的影响
下图展示了 PDV-T 中缩放因子 $\alpha$ 对相对召回率的影响。

![该图像是图表，展示了不同方法在相对召回率@5上的表现随参数 `ext{Alpha}` 的变化。左侧图表显示了五种方法的曲线，包括 Cirr、Circo、FashionIQ/TopTee、FashionIQ/Shirt 和 FashionIQ/Dress，右侧为它们在不同 ViT 模型下的相对表现。](images/4.jpg)
*该图像是图表，展示了不同方法在相对召回率@5上的表现随参数 `ext{Alpha}` 的变化。左侧图表显示了五种方法的曲线，包括 Cirr、Circo、FashionIQ/TopTee、FashionIQ/Shirt 和 FashionIQ/Dress，右侧为它们在不同 ViT 模型下的相对表现。*

*   **观察**：对于 CIReVL，增加 $\alpha$ 通常能提高性能，特别是在 FashionIQ 和 CIRCO 上。这意味着 CIReVL 生成的原始 $\Delta_{PDV}$ 幅度可能偏小，需要放大。
*   **对比**：对于 Pic2Word，调整 $\alpha$ 的效果不如 CIReVL 明显，甚至在某些情况下 $\alpha$ 减小才有效。这表明 Pic2Word 生成的方向向量质量不如 CIReVL，或者其幅度已经过大。

### 6.2.2. PDV-I (图像增强) 的影响
下图展示了 PDV-I 中缩放因子 $\alpha$ 的影响。

![该图像是包含两组曲线的图表，展示了不同模型在相对召回率@5与 `ext{Alpha}` 参数之间的关系。左图和右图分别显示了FashionIQ系列与Cirr、Circo模型的比较。](images/5.jpg)
*该图像是包含两组曲线的图表，展示了不同模型在相对召回率@5与 `ext{Alpha}` 参数之间的关系。左图和右图分别显示了FashionIQ系列与Cirr、Circo模型的比较。*

*   **观察**：随着 $\alpha$ 的增加，检索性能呈现正相关趋势。这表明将语义向量加到图像特征上是有效的，且通常需要较大的 $\alpha$（如 1.4 或 2.0）才能达到最佳效果。

### 6.2.3. PDV-F (融合) 的影响
下图展示了融合权重 $\beta$ 的影响。

![该图像是图表，展示了不同模型在各种 `eta` 值下的相对Recall@5表现。以蓝色、红色和绿色分别表示模型Cirr、Circo和FashionIQ系列，示例数据展示了它们在不同超参数设置下的性能变化。这表明参数调整对检索效果的影响。](images/6.jpg)
*该图像是图表，展示了不同模型在各种 `eta` 值下的相对Recall@5表现。以蓝色、红色和绿色分别表示模型Cirr、Circo和FashionIQ系列，示例数据展示了它们在不同超参数设置下的性能变化。这表明参数调整对检索效果的影响。*

*   **观察**：融合策略 (PDV-F) 总是优于单独使用 PDV-I 或 PDV-T。最佳的 $\beta$ 值通常在 0.4 到 0.8 之间，说明在大多数情况下，文本语义（PDV-T）比视觉特征（PDV-I）更重要，但视觉特征的加入能提供额外的增益。

### 6.2.4. 可视化结果分析
下图展示了在不同 $\alpha$ 和 $\beta$ 设置下，Top-5 检索结果的变化。

![Figure 5. Visualisation of the impact of $\\alpha / \\beta$ scaling on top-5 retrieval results. CIReVL with ViT-B-32 Clip model is the baseline method top. Green and blue bounding boxes indicate true positives and near-true positives, respectively.](images/7.jpg)
*该图像是一个示意图，展示了在使用不同的 $\alpha$ 和 $\beta$ 值时，对检索结果的影响。图中包含三个部分：PDV-T、PDV-I 和 PDV-F。每个部分中，绿色框表示真实的正例，蓝色框表示接近真实的正例。不同的 $\alpha$ 值（-0.5 和 2.0）及 $\beta$ 值（0.3 和 0.7）影响了图像检索的结果，展示了各自组合的检索效果。*

*   **PDV-T**：当 $\alpha$ 较小时（如 -0.5），检索结果与提示意图相反（例如提示“更亮”，结果却“更暗”）。当 $\alpha$ 较大时（如 2.0），结果与提示高度一致，但可能引入过度修改。
*   **PDV-I**：检索结果在视觉上更接近参考图像（保留了更多细节，如衣服的纹理），同时融入了提示的语义。
*   **PDV-F**：通过调整 $\beta$，可以在“视觉相似性”和“语义一致性”之间取得平衡。$\beta$ 小时更像原图，$\beta$ 大时更符合文本描述。

    ---

# 7. 总结与思考

## 7.1. 结论总结
本文提出了一种简单而强大的免训练方法 **PDV**，用于提升零样本组合图像检索的性能。通过定义提示方向向量 $\Delta_{PDV}$，作者巧妙地利用了预训练 VLM 嵌入空间中的几何特性。PDV 提供了三种实用的增强策略：动态文本缩放、图像语义转移和加权融合。实验证明，PDV 能够广泛适用于多种基线方法，并在 Fashion-IQ、CIRR 和 CIRCO 数据集上取得了显著的性能提升。更重要的是，它为用户提供了一种无需重新提取特征即可微调检索结果的机制。

## 7.2. 局限性与未来工作
*   **对基线质量的依赖**：PDV 的效果在很大程度上取决于基线方法生成的组合嵌入的质量。如果基线生成的 $\Delta_{PDV}$ 方向（$\phi$ 角）与真实方向偏差较大，单纯调整 $\alpha$ 可能无法带来显著提升，甚至可能产生负面影响。论文中的分析指出，当 $\phi > 70^\circ$ 时，调整 $\alpha$ 的效果有限。
*   **超参数选择**：虽然论文提供了手动调整指南，但如何自动根据查询内容或基线模型特性选择最优的 $\alpha$ 和 $\beta$ 仍是一个开放问题。
*   **未来方向**：作者建议未来可以探索更鲁棒的组合嵌入生成技术，以及自适应的缩放策略。此外，PDV 的思想也可以扩展到多轮对话式检索或其他多模态任务中。

## 7.3. 个人启发与批判
*   **启发性**：PDV 的成功表明，我们往往低估了预训练模型内部嵌入空间的线性结构。简单的向量运算（加法、缩放）在语义空间中往往具有直观且有效的物理意义（如“国王” - “男人” + “女人” = “女王”）。PDV 将这一思想应用到了检索的增量变化上，非常巧妙。
*   **批判性思考**：
    *   **计算开销**：虽然 PDV 声称是免训练且低开销的，但在实际应用中，为了计算 $\Psi_T(\mathcal{F}(I_{ref}))$（无提示的嵌入），可能需要额外的推理步骤（例如重新生成不带提示的描述或词元化）。对于某些基线（如 Pic2Word），这可能需要特殊的处理。
    *   **通用性验证**：论文主要在时尚和通用物体数据集上进行了验证。对于语义差异极其细微或抽象的领域（如艺术风格检索），PDV 的鲁棒性有待进一步验证。
    *   **与监督方法的差距**：虽然 PDV 显著缩小了零样本与监督方法的差距，但在最高性能指标上仍略逊于最新的监督方法（如 CCIN）。这说明虽然嵌入空间的几何结构很有用，但端到端的训练仍能学习到更复杂的非线性组合关系。