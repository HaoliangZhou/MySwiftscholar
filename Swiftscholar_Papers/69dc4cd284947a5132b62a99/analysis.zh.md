# 1. 论文基本信息

## 1.1. 标题
**Generating a Paracosm for Training-Free Zero-Shot Composed Image Retrieval**
(生成用于免训练零样本组合图像检索的“架空世界”)

## 1.2. 作者
**Tong Wang, Yunhan Zhao, Shu Kong**

*   **Tong Wang:** 澳门大学。
*   **Yunhan Zhao:** 加州大学尔湾分校 (UC Irvine)。
*   **Shu Kong:** 澳门大学，澳门大学协同创新研究所。

## 1.3. 发表期刊/会议
<strong>arXiv (预印本)</strong>
*注：根据提供的元数据，该论文目前处于预印本阶段，尚未标注具体的顶级会议或期刊录用信息，但代码和网站已开源。*

## 1.4. 发表年份
**2026年** (根据元数据 UTC 时间 2026-01-31)

## 1.5. 摘要
组合图像检索 (CIR) 旨在利用由参考图像和修改文本组成的多模态查询，从数据库中检索出目标图像。文本指定了如何修改参考图像以形成“心理图像”，CIR 应基于此图像在数据库中找到目标图像。CIR 的根本挑战在于这个“心理图像”在物理上并不存在，仅由查询隐式定义。当代文献通过使用大型多模态模型 (LMM) 为给定的多模态查询生成文本描述，然后使用视觉-语言模型 (VLM) 进行文本-视觉匹配来搜索目标图像，以此追求零样本方法。相比之下，本文从第一性原理出发解决 CIR，通过直接生成“心理图像”以实现更准确的匹配。具体而言，我们提示 LMM 为给定的多模态查询生成一个“心理图像”，并提议使用这个“心理图像”来搜索目标图像。由于“心理图像”与真实图像之间存在合成到真实的域差距，我们还为数据库中的每个真实图像生成一个合成对应物以促进匹配。在这个意义上，我们的方法使用 LMM 构建了一个“架空世界”，在其中匹配多模态查询和数据库图像。因此，我们将该方法命名为 Paracosm。值得注意的是，Paracosm 是一种免训练的零样本 CIR 方法。它在具有挑战性的基准测试中显著优于现有的零样本方法，实现了零样本 CIR 的最先进性能。

## 1.6. 原文链接
https://arxiv.org/abs/2602.00813 (PDF 链接: https://arxiv.org/pdf/2602.00813)

---

# 2. 整体概括

## 2.1. 研究背景与动机
**核心问题：**
组合图像检索 (Composed Image Retrieval, CIR) 旨在通过一个包含“参考图像”和“修改文本”的复合查询，从图像数据库中找到符合描述的目标图像。例如，用户上传一张“红色椅子”的图片，并输入文本“把颜色改成蓝色”，系统需要检索出数据库中的“蓝色椅子”。

**现有挑战与空白：**
1.  <strong>“心理图像”</strong>缺失： CIR 任务中，用户心中想要的那个修改后的图像（即“心理图像”）在现实中往往并不存在，这给直接检索带来了困难。
2.  **现有方法的局限：** 目前的零样本 CIR 方法大多依赖大型多模态模型 (LMM) 将查询转换为一段**文本描述**，然后利用视觉-语言模型 (VLM) 进行“文本到图像”的匹配。然而，文本描述往往无法完全捕捉图像中丰富、细微的视觉信息（如纹理、光照、具体布局），导致检索精度受限。
3.  **域差距：** 即便有研究尝试生成“伪目标图像”，直接将生成的合成图像与数据库中的真实照片进行匹配，往往因为合成图像与真实照片在视觉风格上的差异（即合成到真实的域差距）而导致效果不佳。

**切入点与创新思路：**
本文提出从第一性原理出发，不再将 CIR 视为文本检索问题，而是将其还原为“图像到图像”的匹配问题。核心思路是：**既然“心理图像”不存在，那就直接生成它；既然合成图像与真实图像有差距，那就把数据库里的真实图像也变成合成图像。** 通过构建一个完全由 LMM 生成的虚拟空间（即“架空世界” Paracosm），在这个统一的虚拟空间内进行图像匹配，从而规避域差距问题。

## 2.2. 核心贡献/主要发现
1.  **提出 Paracosm 方法：** 这是一个免训练的零样本 CIR 方法。它利用 LMM 直接为多模态查询生成“心理图像”，而不是生成文本描述。
2.  **解决域差距：** 创新性地提出为数据库中的每一张真实图像也生成一个“合成对应物”。通过在生成的“心理图像”和“合成对应物”之间进行匹配，有效消除了合成与真实域之间的差异。
3.  **性能提升：** 在 CIRR、CIRCO 和 FashionIQ 等标准基准测试中，Paracosm 显著优于现有的零样本方法，甚至可以媲美一些需要训练的有监督方法，达到了零样本 CIR 的最先进水平 (SOTA)。

    下图（原文 Figure 1）展示了 Paracosm 方法的概览及其在基准测试中的性能对比雷达图：

    ![Fig. 1: Overview of our method and benchmarking results. Unlike existing training-free methods \[21, 49, 62\] generating descriptions for multimodal queries, which use an LMM to generate descriptions for multimodal queries, we use it to generate "mental images" for multimodal queries and synthetic counterparts of database images. Matching them effectively mitigates synthetic-to-real domain gaps and boosts CIR performance. Our final training-free zero-shot method Paracosm (Fig. 2) significantly outperforms existing zero-shot CIR methods, as summarized in the radar chart on standard benchmarks. Detailed results are provided in Section E.](images/1.jpg)
    *该图像是一个示意图，展示了Paracosm方法在训练-free零-shot组合图像检索中的应用。左侧雷达图比较了不同方法在CIR性能上的表现，右侧展示了如何生成''心理图像''及其对应的合成图像，并进行匹配以提高性能。*

---

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，我们需要掌握以下核心概念：

*   <strong>组合图像检索 (Composed Image Retrieval, CIR)：</strong> 一种检索任务，输入包括一张参考图像 $I_{ref}$ 和一段修改文本 $t_{mod}$，目标是检索出数据库中符合“将 $I_{ref}$ 按照 $t_{mod}$ 进行修改后”的图像 $I_{target}$。
*   <strong>零样本学习 (Zero-Shot Learning)：</strong> 指模型在没有见过特定类别的标注数据（对于 CIR 来说，是没有见过 `<参考图像, 修改文本, 目标图像>` 这样的三元组训练数据）的情况下，仍能完成该类别的任务。
*   **免训练：** 本文强调的“免训练”比零样本更严格，指不仅不使用三元组数据微调，甚至不进行任何形式的参数更新（如不训练文本反转网络），完全依赖预训练基础模型的推理能力。
*   <strong>大型多模态模型 (Large Multimodal Model, LMM)：</strong> 如 GPT-4V、Qwen-VL 等，这类模型同时具备强大的图像理解和生成能力，能够根据图像和文本指令生成新的图像或描述。
*   <strong>视觉-语言模型 (Vision-Language Model, VLM)：</strong> 如 CLIP，这类模型主要用于将图像和文本映射到同一特征空间，计算它们之间的相似度，用于检索任务。
*   **合成到真实的域差距：** 指由生成模型产生的图像（通常具有某种“AI 味”或特定的风格）与真实拍摄的照片在特征分布上的不一致性。这种不一致性会干扰 VLM 的相似度计算。

## 3.2. 前人工作
作者将相关工作分为三类：

1.  **有监督 CIR：** 早期方法依赖大量人工标注的三元组数据训练神经网络来融合参考图像和修改文本特征。这成本高昂，且泛化能力受限。
2.  **基于训练的零样本 CIR：** 虽然不直接使用三元组训练，但会利用其他数据进行间接训练。例如：
    *   **文本反转：** 训练一个网络将参考图像映射为一个特殊的“伪词元”，然后将其与修改文本结合进行检索。
    *   **伪目标生成：** 利用预训练的生成模型根据描述生成一张伪目标图来辅助检索。
3.  **免训练零样本 CIR：** 这是本文最直接的对标领域。现有方法（如 CIReVL, OSrCIR）主要利用 LMM 为多模态查询生成一段详细的**文本描述**，然后将其视为纯文本检索问题。本文指出，这些方法忽略了图像本身的丰富信息。

## 3.3. 技术演进
CIR 领域的技术演进路径清晰：从依赖昂贵标注数据的**有监督学习**，发展到利用预训练模型进行间接优化的**零样本学习**，再到完全依赖 LMM 推理能力的**免训练方法**。在免训练方法内部，技术路线正从“生成文本描述”向“生成图像（视觉信息）”转变。本文处于这一技术演进的最前沿，通过引入“架空世界”概念，将生成图像的应用范围从单纯的查询端扩展到了数据库端。

## 3.4. 差异化分析
本文方法与现有工作（特别是 OSrCIR 等）的核心区别在于：
*   **现有工作：** 查询端生成文本 -> 数据库端使用真实图像 -> **文本-图像匹配**。
*   **本文方法：** 查询端生成“心理图像” -> 数据库端生成“合成对应物” -> **图像-图像匹配**（在虚拟空间中）。
    这种差异使得本文能够利用图像中比文本更丰富的视觉细节，且通过将数据库图像也“合成化”，解决了匹配时的域不一致问题。

---

# 4. 方法论

## 4.1. 方法原理
Paracosm 的核心原理是利用 LMM 构建一个虚拟的“架空世界”。在这个世界里，所有的查询和数据库图像都表现为由 LMM 生成的合成图像。通过在这个统一的合成特征空间中进行匹配，避免了直接比较“合成查询图像”与“真实数据库图像”时可能出现的域不匹配问题。

## 4.2. 核心方法详解
Paracosm 的流程主要分为三个步骤：处理多模态查询、预处理数据库图像、以及特征提取与匹配。下图（原文 Figure 2）展示了完整的方法流程图：

![Fig. 2: Flowchart of our training-free zero-shot CIR method Paracosm. Given a multimodal query that consists of a reference image and a modification text, we feed it to an LMM to generate a "mental image". We further generate a brief description for it. Both the "mental image" and description, as well as the modification text, are used as feature representations for the query. As the "mental image" is synthetic, we mitigate synthetic-to-real domain gaps by generating synthetic counterparts of database images. To do so, we use the LMM to generate detailed descriptions, which are used as prompts for image generation. For a database image, we use both itself (i.e., the real photo) and its synthetic counterpart as representation for retrieval. In sum, our method uses LMMs to create a virtual paracosm, where it matches the query and database images.](images/2.jpg)
*该图像是示意图，展示了我们的训练-free zero-shot CIR 方法 Paracosm 的流程。左侧处理多模态查询，包括引用图像和文本描述；右侧为数据库图像预处理，生成合成图像和详细描述，以缩小合成与真实图像之间的域差距。*

### 4.2.1. 处理多模态查询
对于输入的多模态查询 $(I_{ref}, t_{mod})$，Paracosm 并不将其转化为文本，而是利用具备图像编辑能力的 LMM（如 Qwen-Image-Edit）直接生成一张“心理图像” $I_{mental}$。这张图像直观地展示了用户修改后的描述。

此外，为了利用文本中的语义信息，Paracosm 还会利用 LMM 为这张生成的“心理图像”生成一段简短的描述 $t_{query}$。这段描述专注于视觉内容，忽略美学细节。

下图（原文 Figure 4）展示了处理多模态查询的具体示例，包括提示词设计和生成结果：

![Fig. 4: Illustration of processing a multimodal query. Two random examples from CIRR and CIRCO datasets are displayed in the two rows, respectively. For a multimodal query consisting of a reference image and modification texts, we design a prompt incorporating the latter to edit the former, generating a mental image representing this query. As a modification text can contain a relative caption and a shared concept (ref. CIRCO in the second row), we design the prompt to incorporate both. Further, for the mental image, we prompt an LMM to generate a a concise, single-sentence description, exclusively focusing on its visual content while minimizing aesthetic details. We use both mental image and short description to retrieve the target image from the database.](images/4.jpg)
*该图像是一个示意图，展示了如何处理一个多模态查询。在左侧是参考图像，中间是修改提示，右侧是生成的心理图像和描述。此例说明了如何根据修改提示编辑参考图像，并生成简短描述，以便检索目标图像。*

### 4.2.2. 预处理数据库图像
为了解决合成图像与真实图像之间的域差距，Paracosm 对数据库中的每一张真实图像 $I^i$ 也进行“合成化”处理：
1.  **生成详细描述：** 使用 LMM 为真实图像 $I^i$ 生成一段极其详细的文本描述，涵盖所有可见物体、属性和空间关系。
2.  **生成合成对应物：** 将这段详细描述作为提示词，输入到文本到图像 (T2I) 生成模型中，生成一张合成图像 $I_{syn}^i$。

    这样，数据库中的每个条目都由“真实图像”和“合成对应物”共同表示。

下图（原文 Figure 5）展示了数据库图像的处理流程：

![Fig. 5: Ilustration of processing database images. For a database image, we first prompt an LMM to generate a comprehensive description about its visual content, capturing all visible objects, attributes, and visual elements. Using this description, we prompt a text-to-image generation model to produce a synthetic counterpart for this database image. For a database image, we use both itself (i.e., the real photo) and its synthetic counterpart as representation for retrieval.](images/5.jpg)
*该图像是示意图，描述了数据库图像的处理过程。首先，提示大型多模态模型生成图像内容的全面描述，然后利用该描述生成数据库图像的合成对应物，以便于检索。*

### 4.2.3. 匹配与检索
在获得了查询端的“心理图像”和数据库端的“合成对应物”后，Paracosm 使用预训练的 VLM（如 CLIP）提取特征进行匹配。

**特征计算公式：**
Paracosm 定义了查询特征 $\mathbf{q}$ 和第 $i$ 个数据库图像特征 $\phi^i$ 的计算方式。这里我们严格遵循原文公式：

$$
\begin{array} { l } { \mathbf { q } = \lambda ( V ( \mathbf { I } _ { m e n t a l } ) + T ( \mathbf { t } _ { q u e r y } ) ) + ( 1 - \lambda ) T ( \mathbf { t } _ { m o d } ) } \\ { \phi ^ { i } = V ( \mathbf { I } ^ { i } ) + V ( \mathbf { I } _ { s y n } ^ { i } ) } \end{array}
$$

**公式符号解释：**
*   $\mathbf{q}$：最终用于检索的查询特征向量。
*   $\phi^i$：数据库中第 $i$ 张图像的特征向量。
*   $V(\cdot)$：VLM 的视觉编码器，用于将图像（无论是真实的还是合成的）映射到特征空间。
*   $T(\cdot)$：VLM 的文本编码器，用于将文本映射到特征空间。
*   $\mathbf{I}_{mental}$：为查询生成的“心理图像”。
*   $\mathbf{t}_{query}$：为“心理图像”生成的简短描述。
*   $\mathbf{t}_{mod}$：原始的修改文本。
*   $\mathbf{I}^i$：数据库中第 $i$ 张真实图像。
*   $\mathbf{I}_{syn}^i$：为数据库中第 $i$ 张图像生成的合成对应物。
*   $\lambda$：超参数，用于控制“心理图像”特征与原始修改文本特征在最终查询特征中的权重比例。

**公式解析：**
1.  **查询特征 $\mathbf{q}$：** 它是两部分特征的加权和。
    *   第一部分 $\lambda ( V ( \mathbf { I } _ { m e n t a l } ) + T ( \mathbf { t } _ { q u e r y } ) )$ 主要关注生成的“心理图像”及其描述，权重由 $\lambda$ 控制。
    *   第二部分 $( 1 - \lambda ) T ( \mathbf { t } _ { m o d } )$ 保留了原始修改文本的信息，权重为 $1 - \lambda$。
    *   这种设计允许模型灵活调整对生成视觉信息和原始文本指令的依赖程度。
2.  **数据库特征 $\phi^i$：** 它是真实图像特征 $V(\mathbf{I}^i)$ 和合成对应物特征 $V(\mathbf{I}_{syn}^i)$ 的直接相加。这意味着数据库图像的表示同时包含了其真实外观和经过 LMM 重新理解后的合成外观，从而增强了特征的鲁棒性。

**最终检索：**
计算出特征后，Paracosm 计算查询特征与所有数据库特征的余弦相似度，并返回相似度最高的图像索引：

$$
i ^ { * } = \underset { i = 1 } { \operatorname { a r g m a x } } \frac { \mathbf { q } ^ { T } \phi ^ { i } } { \| \mathbf { q } \| _ { 2 } \| \phi ^ { i } \| _ { 2 } }
$$

---

# 5. 实验设置

## 5.1. 数据集
实验在三个标准的 CIR 基准数据集上进行：

1.  <strong>CIRR (Compose Image Retrieval on Real-life images)：</strong> 基于真实生活图像的数据集。它通过从 NLVR2 数据集中挖掘视觉相似的图像来构建负样本，并将视觉相似的图像组织成子集，以评估模型的细粒度区分能力。
2.  <strong>CIRCO (Composed Image Retrieval on Common Objects in context)：</strong> 基于 COCO 数据集，包含开放域的真实世界图像。这是首个每个查询具有多个真实标注目标图像且包含细粒度注释的 CIR 数据集。
3.  **Fashion IQ：** 包含来自 Amazon 的时尚图像，分为 Shirt（衬衫）、Dress（连衣裙）、Toptee（T恤）三个类别。由于测试集未公开，实验在其验证集上进行。

## 5.2. 评估指标
论文使用了以下指标进行评估：

1.  <strong>Recall@k (召回率@k)：</strong>
    *   **概念定义：** 衡量检索结果的前 $k$ 个项目中是否包含正确目标图像的概率。它反映了模型在排名靠前的位置中找到正确答案的能力。
    *   **数学公式：**
        $$ \text{Recall@k} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{rank}(i) \le k) $$
    *   **符号解释：**
        *   $N$：查询的总数。
        *   $\text{rank}(i)$：第 $i$ 个查询对应的正确目标图像在检索结果列表中的排名（从1开始）。
        *   $\mathbb{I}(\cdot)$：指示函数，如果条件为真则取1，否则取0。
        *   $k$：设定的截断位置（如 1, 5, 10）。

2.  <strong>mAP@k (Mean Average Precision @k)：</strong>
    *   **概念定义：** 平均精度均值。对于每个查询，计算其前 $k$ 个结果的平均精度 (AP)，然后对所有查询的 AP 取平均。AP 综合考虑了检索结果的排序准确率，对排名越靠前的正确结果给予更高的奖励。
    *   **数学公式：**
        $$ \text{mAP} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{1}{M_i} \sum_{j=1}^{M_i} \text{Precision}@r_j \right) $$
    *   **符号解释：**
        *   $N$：查询的总数。
        *   $M_i$：第 $i$ 个查询对应的正确目标图像的总数（CIRCO 中可能有多个）。
        *   $r_j$：第 $j$ 个正确目标图像在检索结果中的排名。
        *   $\text{Precision}@r_j$：在排名 $r_j$ 处的查准率。

## 5.3. 对比基线
论文将 Paracosm 与以下代表性方法进行了对比：
*   **有监督方法：** Combiner, BLIP4CIR, CLIP-ProbCR。
*   **基于训练的零样本方法：** Pic2Word, SEARLE, LinCIR, IP-CIR, CIG。
*   **免训练零样本方法：** CIReVL, LDRE, AutoCIR, CoTMR, OSrCIR。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
实验结果表明，Paracosm 在所有数据集和主干网络上均显著优于现有的零样本方法，甚至在某些情况下超过了有监督方法。

**定性分析：**
下图（原文 Figure 3）展示了 Paracosm 与 OSrCIR 的定性对比。可以看出，OSrCIR 生成的文本描述往往丢失了关键的视觉细节（如物体的具体状态或背景），导致检索错误。而 Paracosm 生成的“心理图像”保留了丰富的视觉信息，使得检索结果更加准确。

![Fig. 3: Comparison of qualitative results between OSrCIR \[49\] and our Paracosm. We show four examples from the CIRCO dataset \[1\] in the first column, followed by generated descriptions and top-4 retrievals by OSrCIR, and the mental images and top-4 retrievals by Paracosm. For each multimodal query, OSrCIR uses an LMM to generate a description, uses it to match database images, and returns top-ranked ones. Instead, Paracosm uses an LMM to generate a "mental image" for each query, which contains much richer information than a description, allowing image-to-image matching for better retrieval. Consequently, Paracosm yields better retrievals than OSrCIR.](images/3.jpg)
*该图像是一个示意图，展示了多模态查询生成的描述与检索结果的对比。左侧展示了生成的描述和OSrCIR的前四个检索图像，而右侧展示了Paracosm生成的‘心智图像’与相应的前四个检索图像。通过图像到图像的匹配，Paracosm提供了更准确的检索结果。*

## 6.2. 数据呈现 (表格)
以下是原文 [Table 3] 的结果，展示了在 CIRR 和 CIRCO 数据集上的详细对比：

<table>
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Method</th>
<th rowspan="2">venue&amp;year</th>
<th colspan="3">CIRR Recall@k</th>
<th colspan="3">CIRR RecallSubset@k</th>
<th colspan="4">CIRCO mAP@k</th>
</tr>
<tr>
<th>k=1</th>
<th>k=5</th>
<th>k=10</th>
<th>k=1</th>
<th>k=2</th>
<th>k=3</th>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=50</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="12">ViT-L/14</td>
<td>Combiner [5]</td>
<td>CVPR'22</td>
<td>33.59</td>
<td>65.35</td>
<td>77.35</td>
<td>62.39</td>
<td>81.81</td>
<td>92.02</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>BLIP4CIR [33]</td>
<td>WACV'24</td>
<td>40.17</td>
<td>71.81</td>
<td>83.18</td>
<td>72.34</td>
<td>88.70</td>
<td>95.23</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>CLIP-ProbCR [27]</td>
<td>ICMR'24</td>
<td>23.32</td>
<td>54.36</td>
<td>68.64</td>
<td>54.32</td>
<td>76.30</td>
<td>88.88</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>Pic2Word [39]</td>
<td>CVPR'23</td>
<td>23.90</td>
<td>51.70</td>
<td>65.30</td>
<td>53.76</td>
<td>74.46</td>
<td>87.07</td>
<td>8.72</td>
<td>9.51</td>
<td>10.64</td>
<td>11.29</td>
</tr>
<tr>
<td>SEARLE [4]</td>
<td>ICCV'23</td>
<td>24.24</td>
<td>52.48</td>
<td>66.29</td>
<td>53.76</td>
<td>75.01</td>
<td>88.19</td>
<td>11.68</td>
<td>12.73</td>
<td>14.33</td>
<td>15.12</td>
</tr>
<tr>
<td>LinCIR [16]</td>
<td>CVPR'24</td>
<td>25.04</td>
<td>53.25</td>
<td>66.68</td>
<td>57.11</td>
<td>77.37</td>
<td>88.89</td>
<td>12.59</td>
<td>13.58</td>
<td>15.00</td>
<td>15.85</td>
</tr>
<tr>
<td>LDRE [62]</td>
<td>SIGIR'24</td>
<td>26.53</td>
<td>55.57</td>
<td>67.54</td>
<td>60.43</td>
<td>80.31</td>
<td>89.90</td>
<td>23.35</td>
<td>24.03</td>
<td>26.44</td>
<td>27.50</td>
</tr>
<tr>
<td>CIReVL [21]</td>
<td>ICLR'24</td>
<td>24.55</td>
<td>52.31</td>
<td>64.92</td>
<td>59.54</td>
<td>79.88</td>
<td>89.69</td>
<td>18.57</td>
<td>19.01</td>
<td>20.89</td>
<td>21.80</td>
</tr>
<tr>
<td>IP-CIR + LDRE [29]</td>
<td>CVPR'25</td>
<td>29.76</td>
<td>58.82</td>
<td>71.21</td>
<td>62.48</td>
<td>81.64</td>
<td>90.89</td>
<td>26.43</td>
<td>27.41</td>
<td>29.87</td>
<td>31.07</td>
</tr>
<tr>
<td>CIG + SEARLE [57]</td>
<td>CVPR'25</td>
<td>26.72</td>
<td>55.52</tda>
<td>68.10</td>
<td>57.95</td>
<td>77.81</td>
<td>89.45</td>
<td>12.84</td>
<td>13.64</td>
<td>15.32</td>
<td>16.17</td>
</tr>
<tr>
<td><strong>Paracosm</strong></td>
<td>ours</td>
<td><strong>31.95</strong></td>
<td><strong>61.56</strong></td>
<td><strong>72.96</strong></td>
<td><strong>64.68</strong></td>
<td><strong>82.89</strong></td>
<td><strong>91.47</strong></td>
<td><strong>30.24</strong></td>
<td><strong>31.51</strong></td>
<td><strong>34.29</strong></td>
<td><strong>35.42</strong></td>
</tr>
<tr>
<td rowspan="6">ViT-G/14</td>
<td>Pic2Word [39]</td>
<td>CVPR'23</td>
<td>30.41</td>
<td>58.12</td>
<td>69.23</td>
<td>68.92</td>
<td>85.45</td>
<td>93.04</td>
<td>5.54</td>
<td>5.59</td>
<td>6.68</td>
<td>7.12</td>
</tr>
<tr>
<td>SEARLE [4]</td>
<td>ICCV'23</td>
<td>34.80</td>
<td>64.07</td>
<td>75.11</td>
<td>68.72</td>
<td>84.70</td>
<td>93.23</td>
<td>13.20</td>
<td>13.85</td>
<td>15.32</td>
<td>16.04</td>
</tr>
<tr>
<td>LinCIR [16]</td>
<td>CVPR'24</td>
<td>35.25</td>
<td>64.72</td>
<td>76.05</td>
<td>63.35</td>
<td>82.22</td>
<td>91.98</td>
<td>19.71</td>
<td>21.01</td>
<td>23.13</td>
<td>24.18</td>
</tr>
<tr>
<td>LDRE [62]</td>
<td>SIGIR'24</td>
<td>36.15</td>
<td>66.39</td>
<td>77.25</td>
<td>68.82</td>
<td>85.66</td>
<td>93.76</td>
<td>31.12</td>
<td>32.24</td>
<td>34.95</td>
<td>36.03</td>
</tr>
<tr>
<td>CIReVL [21]</td>
<td>ICLR'24</td>
<td>34.65</td>
<td>64.29</td>
<td>75.06</td>
<td>67.95</td>
<td>84.87</td>
<td>93.21</td>
<td>26.77</td>
<td>27.59</td>
<td>29.96</td>
<td>31.03</td>
</tr>
<tr>
<td>CIG + LinIR [57]</td>
<td>CVPR'25</td>
<td>36.05</td>
<td>66.31</td>
<td>76.96</td>
<td>64.94</td>
<td>83.18</td>
<td>91.93</td>
<td>20.64</td>
<td>21.90</td>
<td>24.04</td>
<td>25.20</td>
</tr>
<tr>
<td>CoTMR [44]</td>
<td>ICCV'25</td>
<td>36.36</td>
<td>67.52</td>
<td>77.82</td>
<td>71.19</td>
<td>86.34</td>
<td>93.87</td>
<td>32.23</td>
<td>32.72</td>
<td>35.60</td>
<td>36.83</td>
</tr>
<tr>
<td>OSrCIR [49]</td>
<td>CVPR'25</td>
<td>37.26</td>
<td>67.25</td>
<td>77.33</td>
<td>69.22</td>
<td>85.28</td>
<td>93.55</td>
<td>30.47</td>
<td>31.14</td>
<td>35.03</td>
<td>36.59</td>
</tr>
<tr>
<td><strong>Paracosm</strong></td>
<td>ours</td>
<td><strong>39.30</strong></td>
<td><strong>70.41</strong></td>
<td><strong>80.39</strong></td>
<td><strong>70.82</strong></td>
<td><strong>86.92</strong></td>
<td><strong>94.46</strong></td>
<td><strong>39.82</strong></td>
<td><strong>40.86</strong></td>
<td><strong>43.96</strong></td>
<td><strong>45.05</strong></td>
</tr>
</tbody>
</table>

以下是原文 [Table 4] 的结果，展示了在 Fashion IQ 数据集上的详细对比：

<table>
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Method</th>
<th rowspan="2">venue&amp;year</th>
<th colspan="2">Shirt</th>
<th colspan="2">Dress</th>
<th colspan="2">Toptee</th>
<th colspan="2">Average</th>
</tr>
<tr>
<th>R@10</th>
<th>R@50</th>
<th>R@10</th>
<th>R@50</th>
<th>R@10</th>
<th>R@50</th>
<th>R@10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="10">ViT-L/14</td>
<td>Combiner [5]</td>
<td>CVPR'22</td>
<td>36.36</td>
<td>58.00</td>
<td>31.63</td>
<td>56.67</td>
<td>38.19</td>
<td>62.42</td>
<td>35.39</td>
<td>59.03</td>
</tr>
<tr>
<td>PL4CIR [66]</td>
<td>SIGIR'22</td>
<td>33.22</td>
<td>59.99</td>
<td>46.17</tda>
<td>68.79</td>
<td>46.46</td>
<td>73.84</td>
<td>41.98</td>
<td>67.54</td>
</tr>
<tr>
<td>Uncertainty retrieval [11]</td>
<td>ICLR'24</td>
<td>32.61</td>
<td>61.34</td>
<td>33.23</td>
<td>62.55</td>
<td>41.40</td>
<td>72.51</td>
<td>35.75</td>
<td>65.47</td>
</tr>
<tr>
<td>Pic2Word [39]</td>
<td>CVPR'23</td>
<td>19.14</td>
<td>32.63</td>
<td>14.38</td>
<td>31.09</td>
<td>20.50</td>
<td>36.26</td>
<td>18.01</td>
<td>33.33</td>
</tr>
<tr>
<td>SEARLE [4]</td>
<td>ICCV'23</td>
<td>26.20</td>
<td>43.60</td>
<td>20.00</td>
<td>40.20</td>
<td>27.90</td>
<td>47.4040</td>
<td>24.70</td>
<td>43.70</td>
</tr>
<tr>
<td>LinCIR [16]</td>
<td>CVPR'24</td>
<td>26.89</td>
<td>45.58</td>
<td>20.48</td>
<td>43.13</td>
<td>29.32</td>
<td>49.97</td>
<td>25.56</td>
<td>46.23</td>
</tr>
<tr>
<td>CIReVL [21]</td>
<td>ICLR'24</td>
<td>29.10</td>
<td>46.81</td>
<td>20.92</td>
<td>42.44</td>
<td>28.81</td>
<td>50.18</td>
<td>26.28</td>
<td>46.49</td>
</tr>
<tr>
<td>CIG + LinCIR [57]</td>
<td>CVPR'25</td>
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
<td>Paracosm</td>
<td>ours</td>
<td>31.80</td>
<td>49.51</td>
<td>24.99</td>
<td>47.45</td>
<td>29.78</td>
<td>50.54</td>
<td>28.86</td>
<td>49.17</td>
</tr>
<tr>
<td rowspan="7">ViT-G/14</td>
<td>Pic2Word [39]</td>
<td>CVPR'23</td>
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
<td>SEARLE [4]</td>
<td>ICCV'23</td>
<td>36.46</td>
<td>55.35</td>
<td>28.16</td>
<td>50.32</td>
<td>39.83</td>
<td>61.45</td>
<td>34.82</td>
<td>55.71</td>
</tr>
<tr>
<td>CIReVL [21]</td>
<td>ICLR'24</td>
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
<td>LDRE [62]</td>
<td>SIGIR'24</td>
<td>35.94</td>
<td>58.58</td>
<td>26.11</td>
<td>51.12</td>
<td>35.42</td>
<td>56.67</td>
<td>32.49</td>
<td>55.46</td>
</tr>
<tr>
<td>AutoCIR [12]</td>
<td>KDD'25</td>
<td>36.36</td>
<td>55.84</td>
<td>26.18</td>
<td>47.69</td>
<td>37.28</td>
<td>60.38</td>
<td>33.27</td>
<td>54.63</td>
</tr>
<tr>
<td>OSrCIR [49]</td>
<td>CVPR'25</td>
<td>38.65</td>
<td>54.71</td>
<td>33.02</td>
<td>54.78</td>
<td>41.04</td>
<td>61.83</td>
<td>37.57</td>
<td>57.11</td>
</tr>
<tr>
<td><strong>Paracosm</strong></td>
<td>ours</td>
<td><strong>40.48</strong></td>
<td><strong>57.80</strong></td>
<td><strong>33.17</strong></td>
<td><strong>55.18</strong></td>
<td><strong>42.58</strong></td>
<td><strong>64.20</strong></td>
<td><strong>38.74</strong></td>
<td><strong>59.06</strong></td>
</tr>
</tbody>
</table>

## 6.3. 消融实验/参数分析
作者进行了详细的消融实验来验证各组件的有效性。

**参数 $\lambda$ 分析：**
$\lambda$ 控制了查询特征中“心理图像”部分与原始修改文本部分的权重。下图（原文 Figure 6）展示了 $\lambda$ 对性能的影响。实验发现，在所有数据集上，设置 $\lambda = 0.3$ 时能获得最佳性能，这说明生成的视觉信息虽然重要，但原始文本指令也提供了关键的语义引导。

![Fig. 6: Analysis of $\\lambda$ which controls the importance of incorporating modification text in Eq. 1. We set $\\lambda = 0 . 3$ based on the results on the CIRR validation set. Interestingly, on all datasets, setting $\\lambda =$ 0.3 consistently yields the highest numeric metrics reported for all the datasets.](images/6.jpg)

**组件有效性验证：**
通过移除不同组件（如移除“心理图像” $I_{mental}$、移除数据库“合成对应物” $I_{syn}$、移除修改文本 $t_{mod}$），实验表明：
1.  引入“心理图像”显著提升了性能，证明了视觉信息比单纯的文本描述更有效。
2.  引入数据库“合成对应物”进一步提升了性能，验证了消除域差距的必要性。
3.  加入原始修改文本 $t_{mod}$ 也有正向增益，有助于捕捉特定的修改指令。

**生成方式对比：**
论文还对比了直接编辑参考图像生成“心理图像”与先生成描述再通过 T2I 生成图像的方式。结果显示，直接编辑图像的方式效果更好，因为它更好地保留了参考图像中的上下文和细节。

---

# 7. 总结与思考

## 7.1. 结论总结
Paracosm 提出了一种新颖的免训练零样本组合图像检索方法。它通过利用 LMM 生成查询的“心理图像”和数据库图像的“合成对应物”，构建了一个统一的虚拟特征空间。这种方法不仅克服了文本描述信息不足的问题，还巧妙地解决了合成图像与真实图像之间的域差距问题。实验结果表明，Paracosm 在多个基准测试中取得了最先进的性能，证明了其有效性和鲁棒性。

## 7.2. 局限性与未来工作
作者指出了以下局限性：
1.  **计算成本高：** 由于需要为数据库中的每张图像生成合成对应物，Paracosm 的离线预处理时间成本较高（如表 1 所示，处理 CIRCO 数据集需 12.9 小时）。虽然在线推理效率尚可，但离线开销是一个瓶颈。
2.  **依赖生成质量：** 方法的性能高度依赖于 LMM 生成图像的质量。如果生成的“心理图像”存在幻觉或事实错误（如生成卡通风格的鸭子而非真实毛绒玩具），将导致检索失败。
3.  **提示词敏感性：** 目前针对不同数据集使用了略微不同的提示词模板，缺乏完全自适应的提示生成机制。

    未来工作可能包括优化图像生成效率、提高生成图像的事实保真度，以及开发自适应的提示词生成策略。

## 7.3. 个人启发与批判
**启发：**
1.  <strong>“第一性原理”</strong>思维： 本文最令人印象深刻的地方在于回归问题本质——CIR 本质上是找一张图，而不是找一段话。通过直接生成目标图像，避免了将视觉信息压缩为文本时造成的信息丢失。
2.  <strong>“架空世界”</strong>隐喻： 将生成模型视为构建一个虚拟世界的工具，并在该世界内解决问题，这是一个非常有创意的视角。这为“合成到真实”的迁移学习提供了新的思路：与其强行让合成图像像真实图像，不如让真实图像也进入合成图像的“世界”。
3.  **成本与性能的权衡：** 本文展示了在免训练框架下，通过增加计算资源（生成大量合成图像）来换取性能提升的可能性。这提示我们在算力充足的场景下，生成式方法可以成为传统检索方法的有力补充。

**批判：**
1.  **实用性考量：** 虽然性能优异，但对于大规模动态数据库（如电商实时上架新商品），离线生成所有合成对应物的成本可能过高，难以实时更新。这限制了该方法在动态场景下的应用。
2.  **生成错误的不可控性：** 论文中的失败案例（Figure 7）揭示了生成模型固有的幻觉问题。在检索任务中，一个错误的生成结果可能导致完全错误的检索，且这种错误较难通过后处理修正。如何保证生成内容的可解释性和可控性是未来的关键。