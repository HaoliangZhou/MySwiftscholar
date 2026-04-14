# 1. 论文基本信息

## 1.1. 标题
**VLM2Vec-V2: Advancing Multimodal Embedding for Videos, Images, and Visual Documents** (VLM2Vec-V2：推进视频、图像和视觉文档的多模态嵌入)

## 1.2. 作者
Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, Yingbo Zhou, Wenhu Chen, Semih Yavuz。

作者主要来自 **Salesforce Research**，合作机构包括 **UC Santa Barbara**（加州大学圣塔芭芭拉分校）、**University of Waterloo**（滑铁卢大学）和 **Tsinghua University**（清华大学）。这是一项由工业界研究实验室主导，与学术界紧密合作的成果。

## 1.3. 发表期刊/会议
该论文目前发布于 **arXiv** (Cornell University)，属于计算机科学领域的预印本平台。虽然尚未经过正式的同行评审会议发表，但 arXiv 是人工智能和计算机视觉领域分享最新研究成果的重要渠道，特别是对于 Salesforce Research 这样的顶级工业界实验室，其发布的预印本通常具有很高的技术参考价值。

## 1.4. 发表年份
2025年（根据提供的 UTC 时间戳 2025-07-07）。

## 1.5. 摘要
多模态嵌入模型在实现语义相似度、信息检索和聚类等下游任务中起着至关重要的作用。然而，现有的多模态嵌入模型（如 VLM2Vec, E5-V, GME）主要集中在自然图像上，对视频和视觉文档等其他视觉形式的支持有限。这限制了它们在 AI 智能体、多模态搜索和检索增强生成（RAG）等现实场景中的适用性。为了缩小这一差距，本文提出了 **VLM2Vec-V2**，这是一个用于学习多样化视觉形式嵌入的统一框架。首先，作者介绍了 **MMEB-V2**，这是一个全面的基准测试，扩展了 MMEB，包含五种新的任务类型：视觉文档检索、视频检索、时间定位、视频分类和视频问答——涵盖文本、图像、视频和视觉文档输入。接下来，作者训练了 **VLM2Vec-V2**，这是一个支持文本、图像、视频和视觉文档输入的通用嵌入模型。广泛的实验表明，VLM2Vec-V2 不仅在新引入的视频和文档检索任务上取得了强大的性能，而且在原始图像基准测试上也优于以前的基线模型。通过广泛的评估，本研究提供了关于各种多模态嵌入模型泛化能力的见解，并强调了统一嵌入学习的有效策略，为研究和现实环境中更具可扩展性和适应性的表示学习奠定了基础。

## 1.6. 原文链接
*   **arXiv 链接:** https://arxiv.org/abs/2507.04590
*   **PDF 链接:** https://arxiv.org/pdf/2507.04590v1
*   **发布状态:** 预印本

# 2. 整体概括

## 2.1. 研究背景与动机
### 2.1.1. 核心问题
当前的多模态嵌入模型虽然已经能够很好地处理图像和文本对，但在面对更复杂的现实世界数据时显得力不从心。具体来说，现有的模型（如 VLM2Vec, E5-V, GME）主要针对“自然图像”（如照片）进行优化，缺乏对**视频**（包含时间动态信息）和**视觉文档**（包含结构化布局信息，如 PDF、幻灯片）的有效支持。

### 2.1.2. 重要性与挑战
在现实世界的 AI 应用中，数据形式是极其丰富的。例如：
*   **AI 智能体** 需要理解网页（视觉文档）和操作演示视频。
*   **多模态搜索** 需要能够根据文本描述检索相关的 YouTube 视频。
*   <strong>检索增强生成 (RAG)</strong> 系统需要从长篇 PDF 报告或演示文稿中提取视觉信息。

    如果嵌入模型只能处理静态图片，那么上述应用将无法高效运行，或者需要为每种模态单独维护一个模型，增加了系统复杂度和维护成本。

### 2.1.3. 切入点
本文的切入点是构建一个**统一的框架**，旨在训练一个能够同时处理文本、图像、视频和视觉文档的通用嵌入模型。为了评估这种通用模型的能力，作者首先构建了一个涵盖这四种模态的综合性基准测试。

## 2.2. 核心贡献/主要发现
### 2.2.1. 核心贡献
1.  **MMEB-V2 基准测试：** 提出了一个全面的评估基准，在原有图像-文本任务的基础上，新增了 5 个元任务类别（视频检索、时间定位、视频分类、视频问答、视觉文档检索），共计覆盖 78 个数据集。
2.  **VLM2Vec-V2 模型：** 提出了一个基于 Qwen2-VL 的通用多模态嵌入模型。该模型通过指令引导的对比学习，将不同模态的数据映射到统一的向量空间中。
3.  **有效的训练策略：** 提出了包括交错子批次采样在内的训练策略，以平衡不同模态的数据并优化模型性能。

### 2.2.2. 主要发现
*   VLM2Vec-V2 在 78 个数据集上取得了最高的平均分数，证明了其强大的泛化能力。
*   尽管是一个通用模型，VLM2Vec-V2 在视频和视觉文档任务上表现优异，甚至在原有的图像基准测试上也超越了之前的专用模型。
*   多模态联合训练（Image + Video + VisDoc）能带来最佳的整体性能，说明不同模态之间存在知识迁移和互补效应。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，我们需要掌握以下核心概念：

*   **嵌入模型：** 在机器学习中，嵌入是指将高维、离散的数据（如单词、图像、视频）映射为低维、连续的实数向量。好的嵌入模型能够保留数据之间的语义关系，例如语义相似的文本或图像在向量空间中距离更近。
*   **视觉语言模型：** 这是一种能够同时理解视觉内容（图像/视频）和文本的深度学习模型。通常由一个图像编码器（如 Vision Transformer）和一个文本解码器（如 LLM）组成，能够执行图像描述生成、视觉问答等任务。
*   **对比学习：** 一种自监督学习方法，其核心思想是让模型学会拉近正样本对（相关的样本，如“猫的图片”和文本“一只猫”）在特征空间中的距离，同时推远负样本对（不相关的样本，如“猫的图片”和文本“一辆车”）。
*   <strong>检索增强生成 (RAG)：</strong> 一种结合了检索和生成的 AI 技术。在生成回答之前，模型先从外部知识库中检索相关信息，然后基于这些信息生成更准确、更有依据的答案。
*   **指令微调：** 在微调模型时，不仅输入数据，还输入描述任务的“指令”（例如“请描述这张图片”），使模型能够根据不同的指令执行不同的任务。

## 3.2. 前人工作
作者在相关工作部分回顾了以下几个领域：

*   **多模态嵌入基准：** 早期工作如 MSCOCO, Flickr30K 主要关注静态图像-文本检索。最近的 M-BEIR 和 MMEB 引入了多任务评估，但仍局限于静态图像。
*   **视频表示学习：** 传统的 CLIP 和 BLIP 擅长处理图像，但在处理视频的时间动态信息上存在不足。VideoCLIP 和 InternVideo2 等模型专门针对视频进行了优化。值得注意的是，LamRA 等模型展示了仅用图像-文本数据训练的模型也能零样本泛化到视频任务。
*   **视觉文档表示学习：** ColPali 是该领域的代表工作，它利用 VLM 生成文档页面的多向量嵌入，并结合后期交互机制进行检索，在文档检索任务上表现优异。VisRAG 则提出了基于视觉的 RAG 管道。
*   **统一模态检索：** GME 和 Uni-Retrieval 等方法尝试构建能够处理多种模态的统一检索框架，但它们通常不包含视频和视觉文档的统一处理。

## 3.3. 技术演进
该领域的技术演进路径可以概括为：
1.  **单模态/双模态时代：** 从早期的图像-文本对（CLIP）开始，解决了跨模态对齐的基础问题。
2.  **多任务与指令时代：** 引入指令微调（如 VLM2Vec），使一个模型能够处理分类、检索等多种任务。
3.  <strong>统一多模态时代（本文）：</strong> 尝试打破模态壁垒，将图像、视频、文档等所有视觉形式统一到一个模型和嵌入空间中，以适应复杂的现实应用需求。

## 3.4. 差异化分析
本文的方法与相关工作相比，核心区别在于**统一性**和**全面性**。
*   与 **ColPali** 相比：ColPali 是专门针对视觉文档检索优化的，虽然在该领域表现极佳，但无法处理视频或通用图像任务。VLM2Vec-V2 则是一个“全能选手”，虽然可能在单一任务上略逊于专用模型，但提供了统一的接口和更广的适用范围。
*   与 **GME** 相比：虽然 GME 也追求统一检索，但 VLM2Vec-V2 引入了更全面的视频和视觉文档数据集，并专门针对这些模态设计了训练策略和评估基准。

# 4. 方法论

## 4.1. 方法原理
VLM2Vec-V2 的核心原理是将一个强大的预训练视觉语言模型（VLM）转化为一个通用的嵌入模型。其基本直觉是：现代的 VLM（如 Qwen2-VL）已经具备了理解图像、视频和文档布局的强大能力。通过**对比学习**，我们可以强制模型将这些不同模态的输入，根据它们在语义上的相似性，映射到同一个共享的向量空间中。

为了实现这一点，作者采用了**指令引导**的策略。即不仅仅输入原始数据，还输入一段描述任务关系的文本（指令），以此告诉模型当前是在做“视频检索”还是“文档分类”，从而帮助模型在统一空间中区分不同任务的语义边界。

## 4.2. 核心方法详解

### 4.2.1. 统一表示与主干网络选择
首先，模型需要一个能够处理多种输入形式的主干网络。作者选择了 **Qwen2-VL** 作为基础模型。选择它的原因在于其具备以下三个关键特性，这对于处理多样化的视觉输入至关重要：
1.  **朴素动态分辨率：** 能够高效处理不同分辨率的输入，这对于处理高分辨率文档页面或不同清晰度的视频帧非常有用。
2.  **多模态旋转位置嵌入：** 能够同时捕捉空间（图像/文档）和时间（视频）结构信息。
3.  **统一的 2D 和 3D 卷积架构：** 这使得模型可以用一致的机制处理图像（2D）和视频（3D，即时间+空间）。

### 4.2.2. 对比学习目标函数
在确定了主干网络后，接下来是训练目标。作者采用了标准的 **InfoNCE 损失** 函数来训练模型。

给定一个预训练的 VLM，我们将查询和目标输入模型，得到它们的向量表示 $( \mathbf { h } _ { q _ { \mathrm { i n s t } } } , \mathbf { h } _ { t ^ { + } } )$。这里的 $\mathbf{h}$ 通常取模型最后一层最后一个词元的隐藏状态向量。

为了训练模型拉近正样本对（相关的查询和目标）并推远负样本对，我们最小化以下损失函数 $\mathcal { L }$：

$$
\operatorname* { m i n } \mathcal { L } = - \log \frac { \phi ( \mathbf { h } _ { q _ { \mathrm { i n s t } } } , \mathbf { h } _ { t ^ { + } } ) } { \phi ( \mathbf { h } { q _ { \mathrm { i n s t } } } , \mathbf { h } _ { t ^ { + } } ) + \displaystyle \sum _ { t ^ { - } \in \mathbb { N } } \phi ( \mathbf { h } _ { q _ { \mathrm { i n s t } } } , \mathbf { h } _ { t ^ { - } } ) } ,
$$

**公式详解：**
*   $\mathcal{L}$：损失函数，我们需要通过优化算法（如梯度下降）使其值尽可能小。
*   $\phi(\cdot, \cdot)$：这是一个匹配分数函数，用于计算两个向量之间的相似度。
*   $\mathbf{h}_{q_{\mathrm{inst}}}$：经过指令条件化的查询向量。
*   $\mathbf{h}_{t^{+}}$：正样本目标向量（即与查询语义匹配的目标）。
*   $\mathbb{N}$：负样本集合，包含与当前查询不匹配的所有其他候选目标。
*   $t^{-}$：负样本集合中的一个具体样本。
*   $\mathbf{h}_{t^{-}}$：负样本的向量表示。
*   分子 $\phi(\mathbf{h}_{q_{\mathrm{inst}}}, \mathbf{h}_{t^{+}})$：表示查询与正确答案之间的相似度。
*   分母 $\phi(\mathbf{h}_{q_{\mathrm{inst}}}, \mathbf{h}_{t^{+}}) + \sum_{t^{-} \in \mathbb{N}} \phi(\mathbf{h}_{q_{\mathrm{inst}}}, \mathbf{h}_{t^{-}})$：表示查询与正确答案的相似度加上查询与所有错误答案的相似度之和。
*   $-\log(\cdot)$：对数损失。直观理解是，我们希望分子（正样本相似度）尽可能大，分母中的负样本相似度部分尽可能小，从而使整个分数值趋近于 1，取对数后趋近于 0，损失最小。

    在这个公式中，匹配分数函数 $\phi$ 具体定义为温度缩放的余弦相似度：

$$
\phi ( \mathbf { h } _ { q } , \mathbf { h } _ { t } ) = \exp ( \frac { 1 } { \tau } \cos ( \mathbf { \bar { h } } _ { q } , \mathbf { h } _ { t } ) ) ,
$$

**公式详解：**
*   $\exp(\cdot)$：指数函数，用于将数值映射为正数，且放大差异。
*   $\tau$：温度参数。这是一个超参数，用于控制相似度分布的平滑程度。$\tau$ 越小，模型对正负样本的区分度要求越高（分布越尖锐）。
*   $\cos(\mathbf{\bar{h}}_{q}, \mathbf{h}_{t})$：余弦相似度函数，用于衡量两个向量方向的夹角。值为 1 表示完全同向，-1 表示完全反向。
*   $\mathbf{\bar{h}}_{q}$：查询向量的归一化形式（L2 归一化）。
*   $\mathbf{h}_{t}$：目标向量（通常在计算余弦相似度时也会进行归一化）。

### 4.2.3. 多模态数据格式化
为了使模型能够理解不同模态和任务的数据，作者设计了一种标准化的输入格式。

对于查询部分，模型不仅接收原始查询 $q$，还接收一个任务指令。我们将指令应用到原始查询上，生成指令条件化的查询 $q_{\mathrm{inst}}$：

$$
q _ { \mathrm { i n s t } } = [ \mathsf { V I S U A L _ { - } T O K E N } ] \mathrm { ~ I n s t r u c t : ~ } \{ t a s k \_ i n s t r u c t i o n \} \backslash \mathsf { n } \mathrm { Q u e r y : ~ } \{ q \} ,
$$

**公式详解：**
*   $[ \mathsf{VISUAL\_TOKEN} ]$：这是一个特殊的模态标记，用于告诉模型接下来的视觉输入是什么类型。例如，如果是图像，可能是 `<|image_pad|>`；如果是视频，可能是 `<|video_pad|>`。
*   $\mathrm{Instruct~:}$：固定的提示词前缀。
*   $\{task\_instruction\}$：具体的任务描述，例如“Find a video that contains the following visual content:”（找到一个包含以下视觉内容的视频：）。
*   $\backslash\mathsf{n}$：换行符。
*   $\mathrm{Query~:}$：固定的提示词前缀。
*   $\{q\}$：实际的查询内容，可以是文本、图像或视频帧序列。

    这种格式化方式使得模型能够通过自然语言指令来区分“根据文本检索视频”和“根据视频检索文本”等不同任务。

对于目标部分，也可以应用类似的指令：

$$
t ^ { + } = [ \mathsf { V I S U A L _ { - } T O K E N } ] \ \{ t a r g e t \_ i n s t r u c t i o n \} .
$$

例如，目标指令可能是“Understand the content of the provided video:”（理解所提供视频的内容：）。

### 4.2.4. 数据采样策略
在训练过程中，如何从混合了图像、视频和文档的数据集中取样是一个关键问题。作者提出了**交错子批次**策略。

1.  **批次混合：** 首先根据预定义的权重表，决定每个批次中不同数据源的采样概率，确保不同模态都有曝光。
2.  **交错子批次：** 将一个大的全局批次（例如 1024）划分为若干个子批次（例如 8 个大小为 128 的子批次）。每个子批次独立地从特定的数据源中采样。
    *   这样做的好处是：相比于完全随机采样（每个样本都来自不同源），子批次内样本的同质性增加了对比学习的难度（因为负样本更难区分，即更“硬”），有助于模型学习更精细的特征。相比于整个批次只来自单一数据源，交错不同子批次又保证了批次内的多样性，避免了训练不稳定。

        下图（原文 Figure 1）展示了 MMEB-V2 基准测试的整体架构，包含了本文所涉及的各种任务类型：

        ![Figure 1: An overview of MMEB-V2, which includes 9 meta-tasks and 78 tasks in total. In addition to the original MMEB benchmark, MMEB-v2 introduces five new meta-tasks focused on video and visual documents. Tasks from MMEB are indicated with blue borders, while newly introduced tasks in MMEB-V2 are marked with red borders.](images/1.jpg)
        *该图像是MMEB-V2的示意图，展示了包括9个元任务和总计78个任务的多模态嵌入基准。该基准扩展了原有的任务类型，新增了针对视频和视觉文档的任务标识，蓝色边框表示原有任务，红色边框表示新任务。*

# 5. 实验设置

## 5.1. 数据集
为了训练和评估 VLM2Vec-V2，作者使用了以下几类数据集：

### 5.1.1. 训练数据
*   <strong>视频语言指令数据 (LLaVA-Hound)：</strong> 包含 30 万个视频-字幕对和 24 万个视频问答对。这些数据由 ChatGPT 生成，用于教会模型理解视频内容。
*   **视觉文档检索数据：** 来自 ViDoRe 和 VisRAG。包括 ColPali 训练集（11.8万）、VisRAG 合成数据（23.9万）和域内数据（12.3万）。这些数据用于训练模型处理 PDF、幻灯片等文档。
*   <strong>图像视觉任务 (MMEB-train)：</strong> 来自原始 MMEB 基准的训练数据，涵盖问答、分类、检索等任务，用于保持模型在基础图像任务上的能力。

### 5.1.2. 评估数据 (MMEB-V2)
MMEB-V2 是本文提出的核心评估基准，包含 78 个任务，分为以下几类：
*   **视频检索：** 如 MSR-VTT, MSVD, YouCook2。任务是根据文本描述检索对应的视频。
*   **时间定位：** 如 QVHighlights, Charades-STA。任务是在长视频中根据文本描述找到对应的时间片段。
*   **视频分类：** 如 Kinetics-700, UCF101。任务是识别视频中的动作类别。
*   **视频问答：** 如 MVBench, Video-MME。任务是回答关于视频内容的多项选择题。
*   **视觉文档检索：** 如 ViDoRe, VisRAG。任务是根据文本问题检索相关的文档页面。

    以下是原文 Table 1 的结果，展示了 MMEB-V2 基准测试的详细统计信息：

    <table>
    <thead>
    <tr>
    <th rowspan="2">Task</th>
    <th rowspan="2"></th>
    <th rowspan="2">Query MOD</th>
    <th rowspan="2">Target MOD</th>
    <th rowspan="2">Domain</th>
    <th rowspan="2">#Query</th>
    <th rowspan="2">#Candidates</th>
    </tr>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="6"><strong>Video Retrieval (5 Tasks)</strong></td>
    </tr>
    <tr>
    <td>DiDeMo</td>
    <td></td>
    <td>V</td>
    <td>V</td>
    <td>Open</td>
    <td>1,004</td>
    <td>1,004</td>
    </tr>
    <tr>
    <td>MSR-VTT</td>
    <td>T</td>
    <td>V</td>
    <td>Open</td>
    <td>1,000</td>
    <td>1,000</td>
    </tr>
    <tr>
    <td>MSVD</td>
    <td>T</td>
    <td>V</td>
    <td>Open</td>
    <td>6700</td>
    <td>670</td>
    </tr>
    <tr>
    <td>VATEX</td>
    <td></td>
    <td>V</td>
    <td>Open</td>
    <td>4,468</td>
    <td>4,468</td>
    </tr>
    <tr>
    <td>YouCook2</td>
    <td>T</td>
    <td>V</td>
    <td>Cooking</td>
    <td>3,179</td>
    <td>3,179</td>
    </tr>
    <tr>
    <td colspan="6"><strong>Moment Retrieval (3 Tasks)</strong></td>
    </tr>
    <tr>
    <td>QVHighlights</td>
    <td>T+V</td>
    <td>V</td>
    <td>Vlog/News</td>
    <td>1,083</td>
    <td>10</td>
    </tr>
    <tr>
    <td>Charades-STA</td>
    <td>T +V</td>
    <td>V</td>
    <td>Activity</td>
    <td>727</td>
    <td>10</td>
    </tr>
    <tr>
    <td>MomentSeeker</td>
    <td>I +V</td>
    <td>V</td>
    <td>Open</td>
    <td>1,800</td>
    <td>10</td>
    </tr>
    <tr>
    <td colspan="6"><strong>Video Classification (5 Tasks)</strong></td>
    </tr>
    <tr>
    <td>Kinetics-700</td>
    <td>V</td>
    <td>T</td>
    <td>Open</td>
    <td>1,000</td>
    <td>700</td>
    </tr>
    <tr>
    <td>SSv2</td>
    <td>V</td>
    <td>T</td>
    <td>Human-Object Interaction</td>
    <td>1,000</td>
    <td>174</td>
    </tr>
    <tr>
    <td>HMD51</td>
    <td>V</td>
    <td>T</td>
    <td>Open</td>
    <td>1,000</td>
    <td>51</td>
    </tr>
    <tr>
    <td>UCF101</td>
    <td>V</td>
    <td>T</td>
    <td>Open</td>
    <td>1,000</td>
    <td>101</td>
    </tr>
    <tr>
    <td>Breakfast</td>
    <td>V</td>
    <td>T</td>
    <td>Cooking</td>
    <td>433</td>
    <td>10</td>
    </tr>
    <tr>
    <td colspan="6"><strong>Video QA (5 Tasks)</strong></td>
    </tr>
    <tr>
    <td>MVBench</td>
    <td>V + T</td>
    <td>T</td>
    <td>Spatial/Temporal</td>
    <td>4,000</td>
    <td>3~5</td>
    </tr>
    <tr>
    <td>Video-MME</td>
    <td>V + T</td>

>
<td>T</td>
<td>Real-world</td>
<td>900</td>
<td>4</td>
</tr>
<tr>
<td> NExT-QA</td>
<td>V + T</td>
<td>T</td>
<td>Daily activity</td>
<td>8,564</td>
<td>5</td>
</tr>
<tr>
<td>EgoSchema</td>
<td>V + T</td>
<td>T</td>
<td>Egocentric</td>
<td>500</td>
<td>5</td>
</tr>
<tr>
<td>ActivityNetQA</td>
<td>V + T</td>
<td>T</td>
<td>Activity</td>
<td>1000</td>
<td>2</td>
</tr>
<tr>
<td colspan="6"><strong>Visual Document Retrieval (24 Tasks)</strong></td>
</tr>
<tr>
<td>ViDoRe (10)</td>
<td>T</td>
<td>D</td>
<td>Documents</td>
<td>280 - 1,646</td>
<td>70 - 999</td>
</tr>
<tr>
<td>ViDoRe-V2 (4)</td>
<td>T</td>
<td>D</td>
<td>Documents</td>
<td>52 - 640</td>
<td>452 - 1,538</td>
</tr>
<tr>
<td>VisRAG (6)</td>
<td>T</td>
<td>D</td>
<td>Documents</td>
<td>63-816</td>
<td>500 - 9,590</td>
</tr>
<tr>
<td>ViDoSeek (2)</td>
<td>T</td>
<td>D</td>
<td>Documents</td>
<td>1,142</td>
<td>5,349</td>
</tr>
<tr>
<td>MMLongBench-Doc (2)</td>
<td>T</td>
<td>D</td>
<td>Documents</td>
<td>838</td>
<td>6, 492</td>
</tr>
</tbody>
</table>

## 5.2. 评估指标
论文中使用了两个主要的评估指标，分别针对不同类型的任务。

### 5.2.1. Hit@1 (Recall@1)
*   **概念定义：** 这是一个用于检索和分类任务的指标。它计算的是在所有返回的候选结果中，**排名第一**的结果是正确答案的概率。简单来说，就是“模型认为最像的那个答案，到底是不是真的正确”。Hit@1 越高，说明模型对最相关结果的排序越准确。
*   **数学公式：**
    $$ \text{Hit@1} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{rank}(q_i, \text{gt}_i) = 1) $$
*   **符号解释：**
    *   $N$：查询的总数。
    *   $q_i$：第 $i$ 个查询。
    *   $\text{gt}_i$：第 $i$ 个查询对应的真实标注数据。
    *   $\text{rank}(q_i, \text{gt}_i)$：真实标注数据在模型返回的排序列表中的位置（1 表示排在第一位）。
    *   $\mathbb{I}(\cdot)$：指示函数，如果括号内的条件为真，则值为 1，否则为 0。

### 5.2.2. NDCG@5 (Normalized Discounted Cumulative Gain at 5)
*   **概念定义：** 这是一个常用于推荐系统和文档检索的指标。它不仅考虑检索结果中是否包含正确答案，还考虑正确答案在排序中的位置（位置越靠前得分越高）。NDCG@5 关注的是前 5 个结果的质量。它是对 DCG@5 进行归一化后的值，范围通常在 0 到 1 之间。
*   **数学公式：**
    $$ \text{NDCG@5} = \frac{\text{DCG@5}}{\text{IDCG@5}} $$
    其中，
    $$ \text{DCG@5} = \sum_{j=1}^{5} \frac{2^{rel_j} - 1}{\log_2(j + 1)} $$
*   **符号解释：**
    *   $rel_j$：排在第 $j$ 位的结果的相关性等级（例如，非常相关为 2，相关为 1，不相关为 0）。
    *   $\log_2(j + 1)$：位置惩罚因子，位置越靠后（$j$ 越大），分母越大，贡献越小。
    *   IDCG@5 (Ideal DCG@5)：理想情况下，前 5 个结果按相关性从高到低排列时计算出的 DCG 值，用于归一化。

## 5.3. 对比基线
作者将 VLM2Vec-V2 与以下几类基线模型进行了比较：
*   **通用多模态嵌入模型：** GME (2B/7B), VLM2Vec (2B/7B), LamRA (7B)。这些模型大多基于图像-文本数据训练，虽然可以通过编码多帧图像来处理视频，但并非专门针对视频和文档优化。
*   **专用视觉文档模型：** ColPali v1.3 (3B)。这是一个专门针对视觉文档检索优化的模型，使用了后期交互机制，通常在文档任务上表现很强，但在其他任务上可能较弱。

# 6. 实验结果与分析

## 6.1. 核心结果分析
Table 2 展示了 VLM2Vec-V2 与所有基线模型在 78 个数据集上的综合性能对比。

以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="5">Image</th>
<th colspan="5">Video</th>
<th colspan="5">VisDoc</th>
<th rowspan="2">All</th>
</tr>
<tr>
<th>CLS</th>
<th>QA</th>
<th>RET</th>
<th>GD</th>
<th>Overall</th>
<th>CLS</th>
<th>QA</th>
<th>RET</th>
<th>MRET</th>
<th>Overall</th>
<th>VDRv1</th>
<th>VDRv2</th>
<th>VR</th>
<th>OOD</th>
<th>Overall</th>
</tr>
</tr>
<tr>
<th># of Datasets &rarr;</th>
<th>10</th>
<th>10</th>
<th>12</th>
<th>4</th>
<th>36</th>
<th>5</th>
<th>5</th>
<th>5</th>
<th>3</th>
<th>18</th>
<th>10</th>
<th>4</th>
<th>6</th>
<th>4</th>
<th>24</th>
<th>78</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10"><strong>Baseline Models</strong></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>ColPali v1.3 (3B)</td>
<td>40.3</td>
<td>11.5</td>
<td>48.1</td>
<td>40.3</td>
<td>34.9</td>
<td>26.7</td>
<td>37.8</td>
<td>21.6</td>
<td>25.5</td>
<td>28.2</td>
<td>83.6</td>
<td>52.0</td>
<td>81.1</td>
<td>43.1</td>
<td>71.0</td>
<td>44.4</td>
</tr>
<tr>
<td>GME (2B)</td>
<td>54.4</td>
<td>29.9</td>
<td>66.9</td>
<td>55.5</td>
<td>51.9</td>
<td>34.9</td>
<td>42.0</td>
<td>25.6</td>
<td>32.4</td>
<td>33.9</td>
<td>86.1</td>
<td>54.0</td>
<td>82.5</td>
<td>43.1</td>
<td>72.7</td>
<td>54.1</td>
</tr>
<tr>
<td>GME (7B)</td>
<td>57.7</td>
<td>34.7</td>
<td>71.2</td>
<td>59.3</td>
<td>56.0</td>
<td>37.4</td>
<td>50.4</td>
<td>28.4</td>
<td>38.2</td>
<td>38.6</td>
<td>89.4</td>
<td>55.6</td>
<td>85.0</td>
<td>44.4</td>
<td>75.2</td>
<td>57.8</td>
</tr>
<tr>
<td>LamRA-Qwen2 (7B)</td>
<td>59.2</td>
<td>26.5</td>
<td>70.0</td>
<td>62.7</td>
<td>54.1</td>
<td>39.3</td>
<td>42.6</td>
<td>24.3</td>
<td>34.6</td>
<td>35.2</td>
<td>22.0</td>
<td>11.5</td>
<td>37.4</td>
<td>21.0</td>
<td>23.9</td>
<td>40.4</td>
</tr>
<tr>
<td>LamRA-Qwen2.5 (7B)</td>
<td>51.7</td>
<td>34.1</td>
<td>66.9</td>
<td>56.7</td>
<td>52.4</td>
<td>32.9</td>
<td>42.6</td>
<td>23.2</td>
<td>37.6</td>
<td>33.7</td>
<td>56.3</td>
<td>33.3</td>
<td>58.2</td>
<td>40.1</td>
<td>50.2</td>
<td>47.4</td>
</tr>
<tr>
<td>VLM2Vec-Qwen2VL (2B)</td>
<td>58.7</td>
<td>49.3</td>
<td>65.0</td>
<td>72.9</td>
<td>59.7</td>
<td>33.4</td>
<td>30.5</td>
<td>20.6</td>
<td>33.0</td>
<td>29.0</td>
<td>49.8</td>
<td>13.5</td>
<td>51.8</td>
<td>33.5</td>
<td>41.6</td>
<td>47.0</td>
</tr>
<tr>
<td>VLM2Vec-Qwen2VL (7B)</td>
<td>62.7</td>
<td>56.9</td>
<td>69.4</td>
<td>82.2</td>
<td>65.5</td>
<td>39.1</td>
<td>30.0</td>
<td>29.0</td>
<td>40.6</td>
<td>34.0</td>
<td>56.9</td>
<td>9.4</td>
<td>59.1</td>
<td>38.1</td>
<td>46.4</td>
<td>52.3</td>
</tr>
<tr>
<td colspan="10"><strong>Ours</strong></td>
<td colspan="7"></td>
</tr>
<tr>
<td>VLM2Vec-V2 (2B)</td>
<td>62.9</td>
<td>56.3</td>
<td>69.5</td>
<td>77.3</td>
<td>64.9</td>
<td>39.3</td>
<td>34.3</td>
<td>28.8</td>
<td>38.5</td>
<td>34.9</td>
<td>75.5</td>
<td>44.9</td>
<td>79.4</td>
<td>39.4</td>
<td>65.4</td>
<td><strong>58.0</strong></td>
</tr>
</tbody>
</table>

**结果解读：**
1.  **整体性能：** VLM2Vec-V2 (2B) 在所有 78 个任务上取得了最高的平均分 **58.0**。这证明了统一训练策略的有效性。值得注意的是，这是一个 2B 参数量的模型，它在整体上超越了 7B 参数量的 GME (57.8) 和 VLM2Vec (52.3)，展现了极高的参数效率。
2.  **图像任务：** VLM2Vec-V2 在图像任务上的平均得分为 64.9，优于大多数基线，甚至与 7B 的 VLM2Vec (65.5) 持平。这说明加入视频和文档数据并没有损害模型在图像上的能力，反而可能通过更丰富的数据提升了泛化性。
3.  **视频任务：** VLM2Vec-V2 在视频任务上得分为 34.9，显著优于之前未针对视频训练的 VLM2Vec (29.0)，与 7B 的 GME (38.6) 和 LamRA (35.2) 相当。考虑到视频训练数据相对较少，这是一个不错的结果。
4.  **视觉文档任务：** VLM2Vec-V2 在文档任务上得分为 65.4。虽然略低于专门优化文档的 ColPali (71.0) 和 GME (75.2)，但远优于之前的 VLM2Vec (41.6)。这表明模型成功获得了处理文档的能力。

## 6.2. 消融实验/参数分析

### 6.2.1. 模态组合的影响
Table 3 展示了使用不同模态数据组合训练对模型性能的影响。

以下是原文 Table 3 的结果：

| Modality | Image | VisDoc | Video | Image+Video | Image+VisDoc | Image+Video+VisDoc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Image** | 62.5 | 27.9 | 33.9 | 63.3 | 62.4 | 62.7 |
| **VisDoc** | 41.5 | 42.6 | 47.9 | 51.9 | 47.4 | 52.2 |
| **Video** | 31.5 | 29.1 | 19.9 | 29.7 | 33.3 | 32.4 |
| **AVG** | 45.2 | 33.2 | 33.9 | 48.3 | 47.7 | **49.1** |

**分析：**
*   仅使用图像数据训练时，图像任务得分最高 (62.5)，但在其他模态上得分很低。
*   结合多种模态（Image+Video+VisDoc）训练时，模型在所有模态上的平均得分最高 (49.1)。
*   特别值得注意的是，加入视频数据有助于提升视觉文档任务的表现（Image+Video+VisDoc 在 VisDoc 上得 52.2，高于 Image+VisDoc 的 47.4），这暗示了视频中的动态信息或更丰富的视觉特征可能对理解复杂文档布局有帮助。

### 6.2.2. 数据采样策略的影响
Table 4 展示了不同交错子批次大小对性能的影响。

以下是原文 Table 4 的结果：

| Modality | IB0 | IB32 | IB64 | IB128 | IB1024 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Image** | 61.2 | 62.3 | 63.2 | 62.0 | 60.7 |
| **VisDoc** | 48.6 | 51.0 | 52.1 | 53.9 | 54.3 |
| **Video** | 34.6 | 33.2 | 33.5 | 34.5 | 35.4 |

**分析：**
*   **IB0** 表示完全随机采样。
*   **IB1024** 表示整个批次来自单一数据源（无交错）。
*   对于**图像**任务，适中的子批次大小 (IB64) 效果最好，呈现出倒 U 型趋势。
*   对于**视觉文档**和**视频**任务，更大的子批次（即更单一的数据源）似乎更有利。这表明对于这些较难的任务，模型可能需要更“纯粹”的批次来学习特定模态的特征，避免受到其他模态噪声的干扰。

### 6.2.3. 模型设置与训练步数
Figure 2 展示了 LoRA rank 和训练步数对性能的影响。

![Figure 2: The left figure shows performance across LoRA ranks for different modalities, while the right figure illustrates performance trends across training steps.](images/2.jpg)

**分析：**
*   **LoRA Rank：** 左图显示，Rank 为 16 时整体性能最好。Rank 太小（8）可能限制了模型适应新任务的能力，Rank 太大（32）可能导致过拟合或不必要的参数冗余。
*   **训练步数：** 右图显示，随着训练步数的增加（从 2K 到 5K），所有模态的性能都在提升，且没有明显的饱和迹象。这表明如果继续训练，模型性能可能还会进一步提升。

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 VLM2Vec-V2，这是一个旨在统一处理文本、图像、视频和视觉文档的多模态嵌入模型。通过构建包含 78 个任务的 MMEB-V2 基准测试，作者全面评估了模型的性能。实验结果表明，VLM2Vec-V2 不仅在新增的视频和文档检索任务上表现强劲，而且通过多模态联合训练，在原有的图像任务上也保持了甚至超越了之前的专用模型性能。这项工作证明了构建统一、通用的多模态嵌入模型是可行且有效的，为未来构建更通用的 AI 智能体和 RAG 系统奠定了基础。

## 7.2. 局限性与未来工作
*   **性能权衡：** 虽然 VLM2Vec-V2 是一个通用模型，但在某些特定领域（如纯视觉文档检索），它仍然落后于专门针对该领域优化的模型（如 ColPali）。这是“通才”与“专才”之间的经典权衡。
*   **训练数据量：** 视频数据的训练量相对较小（30万对），这可能限制了模型在复杂视频理解任务上的上限。
*   **计算成本：** 处理视频和高分辨率文档需要大量的计算资源，这可能会限制其在资源受限环境下的部署。
*   **未来方向：** 作者指出，未来可以探索更长周期的训练以观察收敛行为，以及进一步优化长上下文视频和文档的处理效率。

## 7.3. 个人启发与批判
*   **统一性的价值：** 本文最大的启发在于展示了“统一”的力量。在实际的工业应用中，维护多个模型（一个用于图像，一个用于视频，一个用于文档）是极其繁琐的。VLM2Vec-V2 提供了一个“一站式”的解决方案，极大地简化了系统架构。
*   **指令引导的重要性：** 通过自然语言指令来区分不同任务和模态关系，是一个非常优雅且可扩展的设计。它不需要修改模型结构，仅通过输入格式的改变就能引导模型的行为，这与大语言模型的 Prompt 工程理念不谋而合。
*   **批判性思考：** 虽然模型表现优异，但在消融实验中我们发现，对于视觉文档和视频任务，纯粹的单一模态批次（IB1024）效果更好。这可能暗示当前的“交错”策略虽然对图像任务有利，但对其他模态引入了干扰。未来的工作或许可以探索更动态的采样策略，例如根据训练进度动态调整批次混合比例。
*   **数据集构建：** MMEB-V2 基准的构建非常细致，涵盖了从检索到分类再到 QA 的多种任务。这提醒我们，评估一个通用模型不能只看单一指标，而需要一个多维度的、全面的测试集。