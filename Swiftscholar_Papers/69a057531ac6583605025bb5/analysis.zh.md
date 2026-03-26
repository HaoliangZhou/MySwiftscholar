# 1. 论文基本信息

## 1.1. 标题
SDR-CIR: 用于免训练零样本组合图像检索的语义去偏检索框架 (SDR-CIR: Semantic Debias Retrieval Framework for Training-Free Zero-Shot Composed Image Retrieval)。

## 1.2. 作者
*   Yi Sun\* (武汉理工大学，中国)
*   Jinyu Xu\* (武汉理工大学，中国)
*   Qing Xie (武汉理工大学，中国)
*   Jiachen Li (武汉理工大学，中国)
*   Yanchun Ma (武汉职业技术学院，中国)
*   Yongjian Liu (武汉理工大学，中国)
    (\* 表示共同第一作者)

## 1.3. 发表期刊/会议
该论文计划发表于 2026 年 4 月 13-17 日在阿拉伯联合酋长国迪拜举行的 ACM Web Conference 2026 (WWW '26) 会议上。

## 1.4. 发表年份
2026 年。

## 1.5. 摘要
组合图像检索 (Composed Image Retrieval, CIR) 旨在根据由参考图像 (reference image) 和修改文本 (modification text) 组成的查询 (query)，检索目标图像 (target image)。最近的免训练零样本 (training-free zero-shot) 方法通常采用多模态大语言模型 (Multimodal Large Language Models, MLLMs) 结合思维链 (Chain-of-Thought, CoT) 来生成目标图像描述 (target image description) 以进行检索。然而，由于零样本组合图像检索 (ZS-CIR) 任务的模糊匹配 (fuzzy matching) 特性，生成的描述相对于目标图像容易出现语义偏差 (semantic bias)。

本文提出了 SDR-CIR，一种基于思维链 (CoT) 推理的免训练语义去偏排名 (Semantic Debias Ranking) 方法。首先，选择性思维链 (Selective CoT) 在图像理解阶段引导多模态大语言模型 (MLLM) 提取与修改文本相关的视觉内容，从而从源头上减少视觉噪声 (visual noise)。其次，我们引入了一个包含“锚定 (Anchor)”和“去偏 (Debias)”两个步骤的语义去偏排名 (Semantic Debias Ranking) 模块，以缓解语义偏差。在“锚定 (Anchor)”步骤中，我们将参考图像特征 (reference image features) 与目标描述特征 (target description features) 融合，以强化有用语义并补充被遗漏的线索 (omitted cues)。在“去偏 (Debias)”步骤中，我们明确建模了参考图像对描述的视觉语义贡献 (visual semantic contribution)，并将其作为惩罚项 (penalty term) 纳入相似度分数 (similarity score) 中。通过补充被遗漏的线索同时抑制冗余信息，SDR-CIR 有效缓解了语义偏差并提高了检索性能。在三个标准组合图像检索 (CIR) 基准测试上的实验表明，SDR-CIR 在一阶段 (one-stage) 方法中达到了最先进的 (state-of-the-art) 结果，同时保持了高效率。代码已公开。

## 1.6. 原文链接
*   原文链接: https://arxiv.org/abs/2602.04451
*   PDF 链接: https://arxiv.org/pdf/2602.04451v2
    该论文目前作为预印本 (preprint) 发布在 arXiv 上。

# 2. 整体概括

## 2.1. 研究背景与动机

*   **核心问题：** 组合图像检索 (Composed Image Retrieval, CIR) 是一项复杂任务，它要求模型能够联合理解参考图像 (reference image) 的视觉内容和修改文本 (modification text) 所传达的语义变化，以检索目标图像 (target image)。
*   **重要性与挑战：** CIR 在电子商务和网络搜索等领域具有广泛应用。然而，传统的 CIR 方法依赖于耗时且劳动密集型的三元组 (reference image, modification text, target image) 训练数据，这限制了其泛化能力。
*   <strong>现有研究的挑战与空白 (Gap)：</strong>
    1.  <strong>零样本 CIR (Zero-Shot CIR, ZS-CIR) 的需求：</strong> 为了克服传统方法对大规模标注数据的依赖，零样本 CIR 方法利用预训练的视觉-语言模型 (Visual-Language Models, VLMs) 进行无任务特定训练的检索。
    2.  **训练依赖与免训练方法的权衡：** 训练依赖的 ZS-CIR 方法（如文本反演 (textual inversion)）虽然能将图像映射到伪词元 (pseudo-tokens)，但仍需在大量图像-文本对上进行额外训练或微调。相比之下，免训练 (training-free) 方法直接利用大语言模型 (LLMs) 或多模态大语言模型 (MLLMs) 生成组合查询，具有更大的潜力和更广阔的应用前景。
    3.  **免训练 ZS-CIR 的现有问题：**
        *   <strong>两阶段方法 (Two-stage methods) 的局限：</strong> 这类方法先用图像描述模型 (captioning model) 生成参考图像描述，再与修改文本组合。问题在于参考图像描述是独立生成的，可能遗漏与修改文本相关的关键视觉细节，导致语言模型推理不准确。
        *   <strong>一阶段方法 (One-stage methods) 的挑战：</strong> 尽管一阶段方法利用多模态大语言模型 (MLLMs) 同时处理参考图像和修改文本，但在实践中仍面临两大挑战：
            *   <strong>描述生成过程中的视觉噪声 (Visual noise)：</strong> 现有的一阶段思维链 (CoT) 方法通常会先从参考图像中提取几乎所有视觉信息，然后根据修改文本过滤不相关细节。这种方式在初始提取时缺乏语义选择性，导致无关信息干扰后续过滤，从而在生成的描述中引入噪声。
            *   <strong>排名过程中的语义偏差 (Semantic bias)：</strong> ZS-CIR 本质上是一个模糊匹配 (fuzzy matching) 任务。目标图像可能在修改文本未明确提及的隐式语义方面与参考图像有所不同。因此，生成的描述可能只部分对应目标图像，导致不可避免的语义偏差。这种偏差包括<strong>冗余偏差 (redundancy bias)</strong>（描述包含目标图像无关的细节）和<strong>遗漏偏差 (omission bias)</strong>（描述遗漏了目标图像的关键线索）。这种偏差会误导排名过程，降低检索准确性。

*   **论文的切入点/创新思路：** 本文针对免训练一阶段 ZS-CIR 方法的上述两个挑战，提出了 SDR-CIR 框架。通过在描述生成阶段引入`选择性思维链 (Selective CoT)`从源头减少视觉噪声，并在排名阶段设计`语义去偏排名 (Semantic Debias Ranking)`模块来缓解语义偏差，旨在提高检索性能。

## 2.2. 核心贡献/主要发现

1.  **提出 SDR-CIR 框架：** 引入了一个免训练的、一阶段的语义去偏检索框架 SDR-CIR，旨在解决 ZS-CIR 中描述生成过程中的视觉噪声和排名过程中的语义偏差问题。
2.  <strong>设计选择性思维链 (Selective CoT)：</strong> 提出了一种创新的`选择性思维链 (Selective CoT)`提示策略，引导多模态大语言模型 (MLLMs) 在图像理解阶段有选择性地提取与修改文本相关的视觉内容。这有效减少了视觉噪声，并生成了与预期目标图像语义更对齐的描述。
3.  <strong>引入语义去偏排名 (Semantic Debias Ranking) 模块：</strong> 提出一个包含“锚定 (Anchor)”和“去偏 (Debias)”两步的`语义去偏排名 (Semantic Debias Ranking)`模块。该模块通过融合参考图像和描述特征来强化有用语义和补充遗漏线索，并通过明确建模并惩罚参考图像的视觉语义贡献来修正偏差，从而实现更准确和平衡的排名。
4.  <strong>达到最先进 (SOTA) 性能：</strong> 在 CIRR、CIRCO 和 FashionIQ 三个标准 CIR 基准数据集上进行了广泛实验，SDR-CIR 在一阶段方法中取得了最先进的 (state-of-the-art) 性能，并且在保持高效率的同时有效缓解了语义偏差。

# 3. 预备知识与相关工作

## 3.1. 基础概念

*   <strong>组合图像检索 (Composed Image Retrieval, CIR)：</strong> 是一种图像检索任务，其查询 (query) 由两部分组成：一张作为参照的图像（参考图像，`reference image`）和一段描述对该图像进行修改的文本（修改文本，`modification text`）。目标是从一个图像库中检索出与修改后的描述相符的图像（目标图像，`target image`）。
*   <strong>零样本组合图像检索 (Zero-Shot Composed Image Retrieval, ZS-CIR)：</strong> 指在没有针对特定 CIR 任务的标注三元组（参考图像、修改文本、目标图像）进行训练的情况下，执行 CIR 任务。它通常通过利用大规模预训练的视觉-语言模型 (Visual-Language Models, VLMs) 的零样本泛化能力来实现。
*   <strong>多模态大语言模型 (Multimodal Large Language Models, MLLMs)：</strong> 是指能够理解和处理多种模态信息（如文本、图像、音频等）的大型语言模型。它们通过在海量多模态数据上进行预训练，学习到跨模态的对齐和推理能力，能够执行图像描述、视觉问答、图像编辑等任务。
*   <strong>思维链 (Chain-of-Thought, CoT) 提示：</strong> 是一种针对大语言模型 (LLMs) 或多模态大语言模型 (MLLMs) 的提示工程技术。通过设计特定的提示，鼓励模型在给出最终答案之前，生成一系列中间推理步骤。这些中间步骤能够提升模型的复杂推理能力，使其在解决复杂问题时表现更优。
*   <strong>语义偏差 (Semantic bias)：</strong> 在 ZS-CIR 任务中，指生成的关于目标图像的描述与实际目标图像之间存在的语义不一致。这种不一致可能表现为：
    *   <strong>冗余偏差 (Redundancy bias)：</strong> 描述中包含了目标图像中不存在的、但源自参考图像的无关细节。
    *   <strong>遗漏偏差 (Omission bias)：</strong> 描述中遗漏了目标图像中存在的、但生成过程未能捕捉到的关键视觉线索。
        这种偏差会导致检索结果不准确。
*   <strong>视觉-语言模型 (Visual-Language Models, VLMs)：</strong> 是一种能够理解和处理视觉（图像）和语言（文本）两种模态信息的深度学习模型。它们通常通过在大规模图像-文本对上进行预训练来学习跨模态的表示，例如 OpenAI 的 CLIP 模型。
*   <strong>文本反演 (Textual Inversion)：</strong> 一种技术，旨在将特定的视觉概念（例如参考图像中的某个物体或风格）映射到一个或多个“伪词元 (pseudo-tokens)”，这些伪词元可以在文本提示中被大语言模型或扩散模型理解和使用，从而实现对生成内容的精细控制或个性化。

## 3.2. 前人工作

本节根据论文中提及的相关工作，总结了 ZS-CIR 领域的主要方法及其演进，并主动补充了关键背景信息。

### 3.2.1. 零样本组合图像检索 (ZS-CIR) 方法分类

ZS-CIR 方法主要关注如何将参考图像和修改文本组合成一个查询进行检索，大致分为：

*   <strong>训练依赖 (Training-dependent) 方法：</strong>
    *   **代表性方法：** `Pic2Word` (Saito et al., 2023), `SEARLE` (Baldrati et al., 2023), `MLLM-I2W` (Bao et al., 2025)。
    *   **核心思想：** 这些方法通过学习一个投影模块 (projection module)，将参考图像特征映射到一系列“伪词元 (pseudo-tokens)”。这些伪词元随后与修改文本组合，共同构成检索查询。
    *   **局限性：** 尽管能够利用预训练模型，但它们仍然需要在大规模图像-文本对上进行额外的训练或微调，这增加了计算成本和数据依赖性。

*   <strong>免训练 (Training-free) 方法：</strong>
    *   **核心思想：** 直接利用大语言模型 (LLMs) 或多模态大语言模型 (MLLMs) 的零样本能力来生成组合查询，无需额外的训练或微调。随着 GPT、LLaVA 等大模型的快速发展，这类方法展现出巨大的潜力。
    *   **分类：** 根据描述生成过程中是否需要预处理参考图像，可进一步分为两阶段 (two-stage) 和一阶段 (one-stage) 方法。

### 3.2.2. 免训练 ZS-CIR 的两阶段方法

*   **核心思想：**
    1.  <strong>第一阶段：图像描述 (Image Captioning)。</strong> 首先使用一个独立的图像描述模型（如 BLIP-2）生成参考图像的文本描述。
    2.  <strong>第二阶段：文本组合 (Text Composition)。</strong> 然后将生成的图像描述与修改文本一起输入到大语言模型 (LLM) 中，由 LLM 推理并生成最终的目标图像描述。
*   **代表性方法：** `CIReVL` (Karthik et al., 2023), `LDRE` (Yang et al., 2024), `SEIZE` (Yang et al., 2024)。
*   **局限性：** 论文指出，由于参考图像的描述是独立于修改文本生成的，可能无法捕捉到与预期修改相关的关键视觉细节。这会导致 LLMs 在关键信息缺失的情况下难以进行准确推理，从而影响检索性能。

### 3.2.3. 免训练 ZS-CIR 的一阶段方法

*   **核心思想：** 直接将参考图像和修改文本同时作为输入，通过精心设计的提示 (prompt) 引导多模态大语言模型 (MLLM) 联合考虑视觉细节和文本意图，直接生成目标图像描述。
*   **代表性方法：** `OSrCIR` (Tang et al., 2025), `CoTMR` (Sun et al., 2025)。
*   **进步：** 解决了两阶段方法中信息丢失的问题。
*   <strong>新挑战（本文关注的）：</strong>
    1.  **视觉噪声：** 现有一阶段方法在图像理解阶段倾向于提取所有视觉信息，然后过滤。这在初始提取时缺乏选择性，导致无关视觉内容引入噪声。
    2.  **语义偏差：** ZS-CIR 的模糊匹配特性使得生成的描述与目标图像之间存在语义偏差（冗余或遗漏），误导排名过程。

### 3.2.4. 思维链 (Chain-of-Thought, CoT) 在 ZS-CIR 中的应用

*   <strong>背景知识：思维链 (CoT) 提示</strong>
    *   **概念定义：** 思维链提示是一种通过引导大语言模型生成中间推理步骤来提高其复杂推理能力的技术。其核心思想是模仿人类解决问题的过程，将复杂任务分解为一系列更小的、可管理的步骤。
    *   **应用效果：** 已经被证明能够显著增强 LLMs/MLLMs 的推理和理解能力，并在各种领域（如数学、常识推理等）表现出有效性。
*   **在 ZS-CIR 中的应用：**
    *   **代表性方法：** `OSrCIR` 引入了反射性思维链 (reflective CoT) 帮助 MLLMs 理解和执行修改。`CoTMR` 利用思维链推理使 MLLMs 逐步应用修改。
    *   <strong>局限性 (本文指出的)：</strong> 现有基于思维链的 ZS-CIR 方法往往要求 MLLMs 在过滤不相关细节之前提取几乎所有视觉信息。这种方式在语义上缺乏选择性，可能导致生成的描述中包含无关或冗余的视觉内容，引入视觉噪声。

## 3.3. 技术演进与差异化分析

*   **技术演进脉络：**
    *   <strong>传统 CIR (训练依赖)：</strong> 数据标注成本高，泛化能力差。
    *   <strong>ZS-CIR (训练依赖)：</strong> 利用 VLM，但仍需额外训练。
    *   <strong>ZS-CIR (免训练两阶段)：</strong> 利用 MLLM，但图像描述与修改意图脱节，信息丢失。
    *   <strong>ZS-CIR (免训练一阶段)：</strong> MLLM 联合处理图像和文本，但存在视觉噪声和语义偏差。
*   **本文工作的定位：** SDR-CIR 属于免训练一阶段 ZS-CIR 方法，并且针对现有这类方法的核心问题（视觉噪声和语义偏差）提出了解决方案。
*   **与相关工作的核心区别和创新点：**
    1.  **关于视觉噪声：** 现有 CoT 方法先提取所有信息再过滤，而 SDR-CIR 的`选择性思维链 (Selective CoT)`在`图像理解阶段`就引入了修改文本的引导，实现`选择性提取`，从而从源头上减少视觉噪声。
    2.  **关于语义偏差：** 现有方法（如 `SEIZE`）通过语义编辑增量来缓解偏差。本文认为偏差主要源于参考图像与目标图像之间的模糊匹配。SDR-CIR 的`语义去偏排名 (Semantic Debias Ranking)`模块通过`锚定 (Anchor)`（融合参考图像和描述特征来强化有用语义和补充遗漏线索）和`去偏 (Debias)`（明确建模并惩罚参考图像的视觉语义贡献）两步，更直接、更全面地处理了语义偏差，能够同时补充遗漏线索并抑制冗余信息。这种方法是现有方法所缺乏的。

# 4. 方法论

本节将详细介绍 SDR-CIR 框架，其核心在于通过`选择性思维链 (Selective CoT)`减少描述生成过程中的视觉噪声，并通过`语义去偏排名 (Semantic Debias Ranking)`模块缓解排名过程中的语义偏差。

## 4.1. 方法原理

SDR-CIR 框架的整体流程如原文 Figure 2 所示，它包括两个主要阶段：

1.  **描述生成阶段：** 针对现有 CoT 方法在图像理解时提取过多无关视觉内容的问题，引入`选择性思维链 (Selective CoT)`。该机制引导多模态大语言模型 (MLLM) 在理解参考图像时，仅提取与修改文本相关的视觉内容，从而从源头上减少生成的描述中的视觉噪声。
2.  **排名阶段：** 针对 ZS-CIR 任务固有的模糊匹配导致的语义偏差问题，设计了`语义去偏排名 (Semantic Debias Ranking)`模块。该模块分为`锚定 (Anchor)`和`去偏 (Debias)`两步：
    *   <strong>锚定 (Anchor) 步：</strong> 通过融合参考图像特征和生成的描述特征，构建一个鲁棒的组合查询特征。这有助于强化描述中有用的语义，并补充描述中可能遗漏的关键视觉线索。
    *   <strong>去偏 (Debias) 步：</strong> 显式地建模参考图像对目标描述的视觉语义贡献，并将其作为一个惩罚项引入到最终的相似度分数中。这能够有效抑制与目标图像无关的冗余语义，从而降低非目标候选图像的排名。

        通过这两个阶段的协同作用，SDR-CIR 旨在生成更准确的描述，并基于此实现更精确的图像检索。

        ![该图像是示意图，展示了SDR-CIR框架的两个主要步骤：选择性CoT推理和语义去偏排名。第一步通过多模态大语言模型从参考图像和修改文本中提取特征，以减少视觉噪声。第二步通过融合特征强化有用语义，同时使用$S=(1+β)Sq - βSi$的公式对相似性得分进行调整，以减少语义偏差。](images/2.jpg)
        *该图像是示意图，展示了SDR-CIR框架的两个主要步骤：选择性CoT推理和语义去偏排名。第一步通过多模态大语言模型从参考图像和修改文本中提取特征，以减少视觉噪声。第二步通过融合特征强化有用语义，同时使用$S=(1+β)Sq - βSi$的公式对相似性得分进行调整，以减少语义偏差。*

原文 Figure 2: SDR-CIR 框架总览。

## 4.2. 核心方法详解

### 4.2.1. 选择性思维链 (Selective CoT) 推理

**问题背景：** 在组合图像检索 (CIR) 任务中，用户的查询由参考图像 (reference image) 和修改文本 (modification text) 组成。参考图像提供了检索目标图像的视觉内容，但往往包含冗余信息。现有的一阶段思维链 (CoT) 方法在理解参考图像时，倾向于提取几乎所有视觉信息，然后才根据修改文本过滤不相关细节。这种方式在初始提取时缺乏语义选择性，导致无关的视觉信息可能干扰后续过滤，从而在生成的描述中引入噪声，影响检索性能。

**SDR-CIR 的解决方案：** 为解决这一问题，SDR-CIR 设计了一种`选择性思维链 (Selective CoT)`提示。该提示的核心在于：在修改文本的引导下，多模态大语言模型 (MLLM) 在理解参考图像时，有选择性地提取与修改相关的视觉内容。

<strong>具体流程（四阶段结构）：</strong>

1.  <strong>参考图像理解 (Reference image understanding)：</strong>
    *   首先，解析修改文本，推断出**显式修改目标**（直接指定的对象、属性）和**隐式修改目标**（修改文本暗示的对象、属性）。
    *   然后，在这些目标的指导下，理解参考图像并**有选择性地提取**与修改文本相关的视觉内容。例如，如果修改文本是关于人的着装，则只关注人的穿着，忽略背景细节。这种选择性过程有助于从源头上减少生成的描述中的视觉噪声。

2.  <strong>修改文本理解 (Modification text understanding)：</strong>
    *   多模态大语言模型 (MLLM) 推断修改文本中包含的修改意图。
    *   将修改指令分解为独立的修改步骤。
    *   确定需要修改哪些对象或属性，以及如何修改。
    *   关注任何新增、删除或属性变更。

3.  <strong>应用修改 (Applying modification)：</strong>
    *   多模态大语言模型 (MLLM) 逐步将修改应用到已提取的视觉内容上，从而更新内容。

4.  <strong>目标图像描述生成 (Target image description generation)：</strong>
    *   生成一个连贯、简洁的图像描述。
    *   确保描述准确反映所有修改。
    *   描述应尽可能简单。
    *   不提及目标图像中不会出现的内容。

        原文 Figure 3 比较了 OSrCIR 的 CoT 提示和 SDR-CIR 的选择性 CoT 提示，突出了 SDR-CIR 在选择性提取视觉内容方面的优势。

        ![Figure 3: Comparison on CoT prompt between OSrCIR and ours.](images/3.jpg)
        *该图像是一个示意图，比较了OSrCIR和我们提出的选择性CoT提示。左侧为查询图像，描述一只孔雀旁边的两个人和环境信息；右侧为目标图像，显示孔雀在一辆车旁的草地上，且没有人出现。这种比较展示了我们方法在去除不相关信息方面的优势。*

原文 Figure 3: OSrCIR 和我们提出的选择性 CoT 提示的比较。

### 4.2.2. 语义去偏排名 (Semantic Debias Ranking)

为了进一步缓解由参考图像引入的语义偏差，SDR-CIR 提出了一个`语义去偏排名 (Semantic Debias Ranking)`模块，该模块包含`锚定 (Anchor)`和`去偏 (Debias)`两个步骤。

#### 锚定 (Anchor)：强化和补充信息

**问题背景：** 目标图像描述与目标图像之间的语义偏差，源于参考图像与目标图像之间的模糊对应关系。这种偏差表现为**冗余偏差**（描述包含目标图像中不存在的、但源自参考图像的无关细节）和**遗漏偏差**（描述遗漏了目标图像中存在的关键线索）。直接抑制参考图像的视觉贡献可能会导致有用的语义丢失。

**SDR-CIR 的解决方案：** 建立一个语义锚点 (semantic anchor)，通过融合参考图像特征 (reference image feature) 和生成的描述特征 (generated description feature) 来构建一个鲁棒的组合查询特征。这有助于强化描述中有用的视觉语义，并补充描述中可能遗漏的关键视觉线索。

**具体实现：**

1.  **特征编码：**
    *   使用 CLIP 文本编码器 (CLIP text encoder) 将生成的描述 $T_d$ 编码为特征 $F_d$。
    *   使用 CLIP 图像编码器 (CLIP image encoder) 将候选图像集 $\mathcal{D}$ 中的每个图像 $I_t^i$ 和参考图像 $I_r$ 编码为特征 $F(I_t^i)$ 和 $F_r$。所有特征都位于相同的特征空间中。

2.  **组合查询特征融合：**
    将参考图像特征 $F_r$ 与目标图像描述特征 $F_d$ 进行融合，得到组合查询特征 $F_q$：
    $$
    F_q = (1 - \alpha) F_d + \alpha F_r
    $$
    其中：
    *   $F_q$: 组合查询特征。
    *   $F_d$: 目标图像描述特征，由 CLIP 文本编码器编码生成。
    *   $F_r$: 参考图像特征，由 CLIP 图像编码器编码生成。
    *   $\alpha$: 权重因子，用于平衡目标描述特征和参考图像特征的贡献。

        **目的：** 对于具有冗余信息的描述，添加图像特征可以增加有用信息的相对权重，防止在后续惩罚视觉语义贡献时其作用被削弱。对于不完整的描述，图像特征可以补充被遗漏的关键线索。这一`锚定 (Anchor)`步骤能够获得一个鲁棒的组合查询，使有用语义保持突出，同时补充遗漏线索。

#### 去偏 (Debias)：惩罚视觉语义贡献

**问题背景：** 参考图像是描述中语义偏差的主要来源。为了缓解语义偏差，需要明确地建模和纠正参考图像的视觉语义贡献，以防止其在相似度计算中错误地占据主导地位。

**SDR-CIR 的解决方案：** 借鉴 `SEIZE` 的思想，从相似度 (similarity) 的角度来表示描述中的视觉语义贡献。目标图像描述是由参考图像和修改文本共同影响生成的，而修改文本通常包含正确的信息。因此，参考图像的语义贡献可以近似为目标图像描述的贡献与修改文本的贡献之间的差异。

**具体实现：**

1.  **计算相似度：**
    计算组合查询特征 $F_q$、目标图像描述特征 $F_d$ 和修改文本特征 $F_m$（由 CLIP 文本编码器编码）与候选图像集 $\mathcal{D}$ 中每个图像 $I_t^i$ 的特征 $F(I_t^i)$ 之间的余弦相似度 (cosine similarities)：
    $$
    [S_q, S_d, S_m] = \mathrm{sim}([F_q, F_d, F_m], F(I_t^i)), \forall I_t^i \in \mathcal{D}
    $$
    其中：
    *   $S_q$: 组合查询特征 $F_q$ 与候选图像特征 $F(I_t^i)$ 之间的相似度。
    *   $S_d$: 目标图像描述特征 $F_d$ 与候选图像特征 $F(I_t^i)$ 之间的相似度。
    *   $S_m$: 修改文本特征 $F_m$ 与候选图像特征 $F(I_t^i)$ 之间的相似度。
    *   $\mathrm{sim}(\cdot, \cdot)$: 余弦相似度函数。

2.  **计算参考图像的视觉语义贡献相似度：**
    通过计算 $S_d$ 和 $S_m$ 之间的差值，得到候选图像与参考图像的视觉语义贡献之间的余弦相似度 $S_i$：
    $S_i = S_d - S_m$
    *   $S_i$: 候选图像与参考图像的视觉语义贡献之间的相似度。

    **理解 $S_i$：**
    *   对于包含冗余信息的描述，$S_i$ 主要代表候选图像与包含冗余视觉信息的视觉语义贡献之间的相似度。
    *   对于不完整的描述，$S_i$ 主要代表候选图像与遗漏关键线索的视觉语义贡献之间的相似度。
        当 $S_i$ 值较高时，通常意味着候选图像与这种（冗余或不完整）视觉语义贡献高度一致。这样的候选图像更有可能不是真正的目标图像。

3.  **计算最终相似度：**
    将视觉语义贡献的相似度 $S_i$ 作为惩罚项，从组合查询相似度得分 $S_q$ 中减去，得到最终相似度 $S_f$：
    $$
    S_f = (1 + \beta) S_q - \beta S_i
    $$
    其中：
    *   $S_f$: 最终用于排名的相似度分数。
    *   $\beta$: 权重因子，用于抑制来自参考图像的偏差贡献。
    *   $(1 + \beta)S_q$: 为了平衡分数，将 $S_q$ 的权重设置为 $1 + \beta$。

        **目的：** 最终的相似度分数 $S_f$ 用于检索。这一`去偏 (Debias)`步骤通过惩罚与冗余或不完整视觉语义贡献高度相关的候选图像，有效降低了非目标图像的排名，从而实现去偏并提高检索性能。

# 5. 实验设置

## 5.1. 数据集

实验在三个广泛使用的组合图像检索 (CIR) 数据集上进行：

### 5.1.1. CIRR (Composed Image Retrieval on Real-world images)
*   **来源：** 由 Zheyuan Liu 等人于 2021 年提出，是第一个开放域 (open-domain) 的 CIR 数据集。
*   **规模：** 包含 36,554 个查询 (queries)，每个查询都与一个单一的目标图像 (target image) 配对。
*   **特点：** 强调对象级别 (object-level) 的修改，例如添加、删除或替换对象。
*   **样本示例：** (未在原文中提供具体示例，但根据描述可理解为如“一张狗的图片，把狗的颜色改成黑色”)

### 5.1.2. CIRCO (Composed Image Retrieval in the Wild)
*   **来源：** 由 Alberto Baldrati 等人于 2023 年提出，基于 COCO 2017 未标注集 (COCO 2017 unlabeled set) 中的真实图像构建。
*   **规模：** 包含 800 个测试查询 (test queries) 和 220 个验证查询 (validation queries)。
*   **特点：** 每个查询关联多个`真实标注数据 (Ground Truth)`，平均每个查询有 4.53 个目标图像。强调对象级别 (object-level) 的变化。
*   **样本示例：** (未在原文中提供具体示例，但根据描述可理解为如“一张鸟的图片，把鸟的种类改成鹦鹉，背景不变”，且可能有多张满足条件的鹦鹉图片)

### 5.1.3. FashionIQ
*   **来源：** 由 Hui Wu 等人于 2021 年提出，专注于时尚领域。
*   **规模：** 包含 30,135 个查询三元组 (query triplets) 和 77,683 个候选图像。每个查询链接到一个单一的目标图像。
*   **类别：** 涵盖衬衫 (shirt)、连衣裙 (dress) 和 T 恤 (toptee) 三个类别。
*   **特点：** 强调属性级别 (attribute-level) 的修改，例如服装的颜色和款式。
*   **样本示例：** (未在原文中提供具体示例，但根据描述可理解为如“一张红色连衣裙的图片，把颜色改成蓝色，款式变成长袖”)

    **为什么选择这些数据集：** 这些数据集涵盖了 CIR 任务中不同类型的修改需求：CIRR 和 CIRCO 侧重于对象级别的变化（如添加、删除或替换对象），而 FashionIQ 则侧重于属性级别的修改（如服装颜色和样式）。这使得实验能够全面验证方法在不同复杂度和粒度修改任务上的性能。

## 5.2. 评估指标

对论文中使用的每个评估指标，提供其概念定义、数学公式和符号解释。

### 5.2.1. Recall@k (R@k)

*   **概念定义：** Recall@k，通常简写为 R@k，是信息检索和推荐系统领域常用的评估指标。它衡量的是在前 $k$ 个检索结果中，有多少比例的`真实标注数据 (Ground Truth)`被成功检索到。R@k 关注的是模型在检索排名靠前的位置上能否召回所有相关的项目。数值越高表示性能越好。

*   **数学公式：**
    对于单个查询：
    $$
    \mathrm{Recall@k} = \frac{\text{检索结果前 k 项中相关项目的数量}}{\text{所有相关项目的总数量}}
    $$
    对于多个查询的平均 Recall@k：
    $$
    \mathrm{R@k} = \frac{1}{|Q|} \sum_{q \in Q} \frac{\sum_{i=1}^k \mathbb{I}(r_{q,i} \in \mathrm{GT}_q)}{|\mathrm{GT}_q|}
    $$
    其中，当每个查询只有一个`真实标注数据 (Ground Truth)`时（如 CIRR 和 FashionIQ）：
    $$
    \mathrm{R@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{target image is in top k results for query } q)
    $$

*   **符号解释：**
    *   $Q$: 查询集合。
    *   $|Q|$: 查询集合中的查询数量。
    *   $q$: 集合 $Q$ 中的一个查询。
    *   $k$: 检索结果的排名截断点（例如，k=1, 5, 10, 50）。
    *   $\mathrm{GT}_q$: 查询 $q$ 的`真实标注数据 (Ground Truth)`集合（即所有相关的目标图像）。
    *   $| \mathrm{GT}_q |$: 查询 $q$ 的所有相关目标图像的总数量。
    *   $r_{q,i}$: 查询 $q$ 的第 $i$ 个检索结果。
    *   $\mathbb{I}(\cdot)$: 指示函数 (indicator function)，如果括号内的条件为真则返回 1，否则返回 0。

### 5.2.2. Mean Average Precision@k (mAP@k)

*   **概念定义：** Mean Average Precision@k (mAP@k) 是衡量检索系统性能的另一个重要指标，尤其适用于每个查询有多个`真实标注数据 (Ground Truth)`的情况（如 CIRCO 数据集）。它首先计算每个查询的平均精度 (Average Precision, AP)，然后对所有查询的 AP 值取平均。AP 考虑了检索结果的排名顺序，将相关项目的排名位置纳入考量，越相关的项目排在越前面，AP 值越高。mAP@k 是对检索质量和排名顺序的综合评估。数值越高表示性能越好。

*   **数学公式：**
    首先，对于单个查询 $q$，其平均精度 $\mathrm{AP}_q$ 计算为：
    $$
    \mathrm{AP}_q = \sum_{i=1}^k P_q(i) \cdot \Delta r_q(i)
    $$
    其中，$\Delta r_q(i)$ 表示第 $i$ 个位置上的文档是否为相关文档，如果 $r_{q,i}$ 是相关文档，则 $\Delta r_q(i) = 1$，否则为 `0`。$P_q(i)$ 是在第 $i$ 个位置上的精确度 (Precision)：
    $$
    P_q(i) = \frac{\text{检索结果前 i 项中相关项目的数量}}{\text{i}}
    $$
    然后，mAP@k 是所有查询的 $\mathrm{AP}_q$ 的平均值：
    $$
    \mathrm{mAP@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathrm{AP}_q
    $$

*   **符号解释：**
    *   $Q$: 查询集合。
    *   $|Q|$: 查询集合中的查询数量。
    *   $q$: 集合 $Q$ 中的一个查询。
    *   $k$: 检索结果的排名截断点。
    *   $\mathrm{AP}_q$: 查询 $q$ 的平均精度 (Average Precision)。
    *   $P_q(i)$: 对于查询 $q$，在前 $i$ 个检索结果中的精确度。
    *   $\Delta r_q(i)$: 指示函数，当查询 $q$ 的第 $i$ 个检索结果 $r_{q,i}$ 是相关文档时为 1，否则为 0。

### 5.2.3. Recall_sub@k (R_sub@k)

*   **概念定义：** Recall_sub@k (R_sub@k) 是在 CIRR 数据集的一个子集上计算的 Recall@k 指标。这个子集包含了与目标图像高度相似的样本，旨在提供更严格的评估。它特别关注模型在区分细微差异方面的能力，因为这些相似样本对模型提出了更高的挑战。

*   **数学公式：**
    与标准的 Recall@k 公式相同，但仅应用于 CIRR 数据集的特定“高度相似”子集。
    $$
    \mathrm{R_{sub}@k} = \frac{1}{|Q_{sub}|} \sum_{q \in Q_{sub}} \mathbb{I}(\text{target image is in top k results for query } q)
    $$

*   **符号解释：**
    *   $Q_{sub}$: CIRR 数据集的“高度相似”子集中的查询集合。
    *   $|Q_{sub}|$: 子集中的查询数量。
    *   $q$: 子集 $Q_{sub}$ 中的一个查询。
    *   $k$: 检索结果的排名截断点。
    *   $\mathbb{I}(\cdot)$: 指示函数，如果目标图像在前 $k$ 个结果中则返回 1，否则返回 0。

## 5.3. 对比基线

论文将 SDR-CIR 与以下几种代表性的零样本组合图像检索 (ZS-CIR) 方法进行了比较：

### 5.3.1. 训练依赖 (Training-dependent) 的 ZS-CIR 方法
*   **Pic2Word [35]:** 将参考图像特征映射到伪词元 (pseudo-tokens) 的方法。
*   **SEARLE [2]:** 结合伪词元和 GPT 生成的描述进行检索。
*   **MLLM-I2W [5]:** 使用多模态大语言模型 (MLLM) 上下文提示将参考图像特征映射到伪词元。

### 5.3.2. 免训练 (Training-free) 的 ZS-CIR 方法

<strong>两阶段 (Two-stage) 方法：</strong>
*   **CIReVL [18]:** 使用 BLIP-2 和大语言模型 (LLM) 分两阶段生成目标图像描述。
*   **LDRE [47]:** 生成并聚合多个目标图像描述，然后进行检索。
*   **SEIZE [46]:** 生成并聚合多个目标图像描述，并执行语义编辑搜索 (semantic editing search)。

<strong>一阶段 (One-stage) 方法：</strong>
*   **OSrCIR [41]:** 使用基于思维链 (CoT) 的多模态大语言模型 (MLLM) 生成目标图像描述，并进行直接搜索。
*   **CoTMR [37]:** 使用基于思维链 (CoT) 的多模态大语言模型 (MLLM) 生成目标图像描述，并进行多尺度推理。

## 5.4. 实现细节

*   <strong>多模态大语言模型 (MLLM)：</strong>
    *   主要使用 GPT-4.1 [31] 进行目标图像描述生成。
    *   在消融实验中，还使用了 Qwen2.5-VL-72B [1] 和 GPT-4o-mini [30] 进行对比分析。
*   <strong>检索编码器 (Retrieval Encoder)：</strong>
    *   采用来自 OpenCLIP [15] 的预训练 CLIP 模型的三种变体作为主干网络 (backbone)：ViT-B/32, ViT-L/14, 和 ViT-G/14。
*   <strong>超参数 (Hyper-parameters)：</strong>
    *   缩放因子 (scaling factors) $\alpha$ 和 $\beta$ 的设置：
        *   CIRR 数据集: $\alpha = 0.05$, $\beta = 0.5$
        *   CIRCO 数据集: $\alpha = 0.15$, $\beta = 0.35$
        *   FashionIQ 数据集: $\alpha = 0.2$, $\beta = 0.4$
*   **开发环境：**
    *   所有实验均使用 PyTorch [32] 实现。
    *   在单个 NVIDIA RTX 3090 GPU 上进行。

# 6. 实验结果与分析

本节详细分析了 SDR-CIR 在三个标准基准数据集上的实验结果，并对方法中的关键模块进行了消融研究和超参数分析。

## 6.1. ZS-CIR 基准比较

为了公平比较，所有一阶段基线方法都采用相同的多模态大语言模型 (MLLM) (GPT-4.1)，且在思维链 (CoT) 中不使用上下文示例 (in-context examples)。

以下是原文 Table 1，展示了 CIRCO 和 CIRR 数据集上的实验结果：

<table>
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Method</th>
<th rowspan="2">Training-free</th>
<th rowspan="2">Type</th>
<th colspan="4">CIRCO</th>
<th colspan="6">CIRR</th>
</tr>
<tr>
<th colspan="3">mAP@k</th>
<th></th>
<th colspan="3">Recall@k</th>
<th colspan="3">Recall<sub>sub</sub>@k</th>
</tr>
<tr>
<th></th>
<th></th>
<th></th>
<th></th>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=50</th>
<th>k=1</th>
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
<td rowspan="7">ViT-B/32</td>
<td>SEARLE [2]</td>
<td>×</td>
<td>-</td>
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
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>14.95</td>
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
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
<td>17.96</td>
<td>18.32</td>
<td>20.21</td>
<td>21.11</td>
<td>25.69</td>
<td>55.13</td>
<td>69.04</td>
<td>89.90</td>
<td>60.53</td>
<td>80.65</td>
<td>90.70</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
<td>19.04</td>
<td>19.64</td>
<td>21.55</td>
<td>22.49</td>
<td>27.47</td>
<td>57.42</td>
<td>70.17</td>
<td>-</td>
<td>65.59</td>
<td>84.48</td>
<td>92.77</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>16.85</td>
<td>17.39</td>
<td>19.15</td>
<td>20.01</td>
<td>28.07</td>
<td>57.95</td>
<td>69.71</td>
<td>88.94</td>
<td>62.31</td>
<td>81.18</td>
<td>91.04</td>
</tr>
<tr>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>21.16</td>
<td>21.77</td>
<td>23.71</td>
<td>24.70</td>
<td>30.12</td>
<td>60.19</td>
<td>71.71</td>
<td>90.34</td>
<td>67.11</td>
<td>85.13</td>
<td>93.64</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>23.78</td>
<td>24.43</td>
<td>26.58</td>
<td>27.50</td>
<td>34.48</td>
<td>65.74</td>
<td>76.87</td>
<td>93.06</td>
<td>69.90</td>
<td>87.04</td>
<td>94.48</td>
</tr>
<tr>
<td rowspan="7">ViT-L/14</td>
<td>Pic2Word [35]</td>
<td>×</td>
<td>-</td>
<td>8.72</td>
<td>9.51</td>
<td>10.64</td>
<td>11.29</td>
<td>23.90</td>
<td>51.70</td>
<td>65.30</td>
<td>87.80</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>SEARLE [2]</td>
<td>×</td>
<td>-</td>
<td>11.68</td>
<td>12.73</td>
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
<td>MLLM-I2W [5]</td>
<td>×</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>28.30</td>
<td>57.90</td>
<td>70.20</td>
<td>93.90</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>18.57</td>
<td>19.01</td>
<td>20.89</td>
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
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
<td>23.25</td>
<td>24.03</td>
<td>26.44</td>
<td>27.50</td>
<td>26.53</td>
<td>55.57</td>
<td>67.54</td>
<td>88.50</td>
<td>60.43</td>
<td>80.31</td>
<td>89.90</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
<td>24.98</td>
<td>25.82</td>
<td>28.24</td>
<td>29.35</td>
<td>28.65</td>
<td>57.16</td>
<td>69.23</td>
<td>-</td>
<td>66.22</td>
<td>84.05</td>
<td>92.34</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>21.83</td>
<td>22.46</td>
<td>24.49</td>
<td>25.44</td>
<td>30.63</td>
<td>60.34</td>
<td>72.00</td>
<td>89.86</td>
<td>64.31</td>
<td>82.29</td>
<td>91.35</td>
</tr>
<tr>
<td rowspan="7">ViT-G/14</td>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>26.52</td>
<td>27.13</td>
<td>29.51</td>
<td>30.56</td>
<td>33.54</td>
<td>63.25</td>
<td>74.63</td>
<td>91.08</td>
<td>69.88</td>
<td>86.53</td>
<td>94.19</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>35.08</td>
<td>37.61</td>
<td>67.71</td>
<td>79.13</td>
<td>93.81</td>
<td>71.90</td>
<td>88.39</td>
<td>94.63</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
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
<td>93.21</td>
</tr>
<tr>
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
<td>31.12</td>
<td>32.24</td>
<td>34.95</td>
<td>36.03</td>
<td>36.15</td>
<td>66.39</td>
<td>77.25</td>
<td>93.95</td>
<td>68.82</td>
<td>85.66</td>
<td>93.76</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
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
<tr>
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>24.73</td>
<td>25.99</td>
<td>28.09</td>
<td>29.19</td>
<td>33.52</td>
<td>61.66</td>
<td>73.30</td>
<td>90.17</td>
<td>65.42</td>
<td>82.75</td>
<td>91.54</td>
</tr>
<tr>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>29.59</td>
<td>30.74</td>
<td>33.37</td>
<td>34.44</td>
<td>35.93</td>
<td>65.11</td>
<td>75.33</td>
<td>91.45</td>
<td>70.82</td>
<td>87.16</td>
<td>94.48</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>33.05</td>
<td>34.50</td>
<td>37.21</td>
<td>38.42</td>
<td>40.17</td>
<td>69.76</td>
<td>79.88</td>
<td>94.00</td>
<td>73.30</td>
<td>88.89</td>
<td>94.99</td>
</tr>
</tbody>
</table>

以下是原文 Table 2，展示了 FashionIQ 验证集上的实验结果：

<table>
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Method</th>
<th rowspan="2">Training-free</th>
<th rowspan="2">Type</th>
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
<td rowspan="7">ViT-B/32</td>
<td>SEARLE [2]</td>
<td>×</td>
<td>-</td>
<td>24.44</td>
<td>41.61</td>
<td>18.54</td>
<td>39.51</td>
<td>25.70</td>
<td>46.46</td>
<td>22.89</td>
<td>42.53</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>28.36</td>
<td>47.84</td>
<td>25.29</td>
<td>46.36</td>
<td>31.21</td>
<td>53.85</td>
<td>28.29</td>
<td>49.35</td>
</tr>
<tr>
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
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
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
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
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>31.31</td>
<td>50.74</td>
<td>27.07</td>
<td>48.83</td>
<td>33.71</td>
<td>55.23</td>
<td>30.70</td>
<td>51.60</td>
</tr>
<tr>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>30.77</td>
<td>50.54</td>
<td>29.45</td>
<td>50.92</td>
<td>34.93</td>
<td>57.57</td>
<td>31.72</td>
<td>53.01</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>36.41</td>
<td>57.02</td>
<td>36.84</td>
<td>58.85</td>
<td>43.14</td>
<td>64.71</td>
<td>38.80</td>
<td>60.19</td>
</tr>
<tr>
<td rowspan="9">ViT-L/14</td>
<td>Pic2Word [35]</td>
<td>×</td>
<td>-</td>
<td>26.20</td>
<td>43.60</td>
<td>20.00</td>
<td>40.20</td>
<td>27.90</td>
<td>47.40</td>
<td>24.70</td>
<td>43.70</td>
</tr>
<tr>
<td>SEARLE [2]</td>
<td>×</td>
<td>-</td>
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
<td>MLLM-I2W [5]</td>
<td>×</td>
<td>-</td>
<td>27.30</td>
<td>46.50</td>
<td>29.90</td>
<td>48.60</td>
<td>33.80</td>
<td>55.20</td>
<td>30.30</td>
<td>50.10</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
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
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
<td>31.04</td>
<td>51.22</td>
<td>22.93</td>
<td>46.76</td>
<td>31.57</td>
<td>53.64</td>
<td>28.51</td>
<td>50.54</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
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
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>34.00</td>
<td>51.86</td>
<td>27.57</td>
<td>48.34</td>
<td>32.94</td>
<td>54.46</td>
<td>31.50</td>
<td>51.55</td>
</tr>
<tr>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>33.42</td>
<td>52.31</td>
<td>29.70</td>
<td>50.42</td>
<td>34.73</td>
<td>57.47</td>
<td>32.62</td>
<td>53.40</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>41.02</td>
<td>59.27</td>
<td>37.04</td>
<td>59.15</td>
<td>44.47</td>
<td>65.32</td>
<td>40.84</td>
<td>61.25</td>
</tr>
<tr>
<td rowspan="6">ViT-G/14</td>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
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
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
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
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
<td>43.60</td>
<td>65.42</td>
<td>39.61</td>
<td>61.02</td>
<td>45.94</td>
<td>71.12</td>
<td>43.05</td>
<td>65.85</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>35.03</td>
<td>54.02</td>
<td>30.94</td>
<td>53.50</td>
<td>36.66</td>
<td>58.29</td>
<td>34.21</td>
<td>55.27</td>
</tr>
<tr>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>35.03</td>
<td>53.39</td>
<td>32.82</td>
<td>55.48</td>
<td>37.89</td>
<td>59.46</td>
<td>35.25</td>
<td>56.11</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>44.55</td>
<td>62.37</td>
<td>42.74</td>
<td>63.41</td>
<td>48.29</td>
<td>69.71</td>
<td>45.19</td>
<td>65.16</td>
</tr>
</tbody>
</table>

### 6.1.1. CIRCO 数据集结果分析
*   <strong>多`真实标注数据 (Ground Truth)`的挑战：</strong> CIRCO 数据集每个查询有多个`真实标注数据 (Ground Truth)`。过于精确的描述可能只与单一实例匹配，从而缩小检索范围。
*   **SDR-CIR 的优势：** SDR-CIR 通过语义去偏，抑制了描述中由参考图像引入的不相关细节，从而提高了对有效目标图像的覆盖率，在所有比较的基线方法中表现最佳。
*   **具体提升：** 以 ViT-L/14 主干网络 (backbone) 为例，SDR-CIR 的 mAP@5 比 `OSrCIR` 提升了 9.08% (21.83% -> 30.91%)，比 `CoTMR` 提升了 4.39% (26.52% -> 30.91%)。这表明即使不使用上下文示例，SDR-CIR 也能在 CIRCO 上取得显著效果。

### 6.1.2. CIRR 数据集结果分析
*   **修改文本主导：** CIRR 数据集存在许多假阴性 (false negatives)，检索主要由修改文本主导。如果描述中包含冗余的参考图像内容，会严重影响检索性能。
*   **SDR-CIR 的优势：** SDR-CIR 能显著抑制参考图像冗余信息的干扰，在所有基线方法中表现最佳。
*   **具体提升：** 在 ViT-L/14 主干网络上，与最佳免训练方法 `CoTMR` 相比，SDR-CIR 的 Recall@1 提升了 4.07% (33.54% -> 37.61%)。
*   <strong>子集 (Subset) 性能：</strong> 在 CIRR 子集 (CIRR subset) 上，SDR-CIR 优于所有一阶段基线。例如，在 ViT-G/14 上，SDR-CIR 比最佳一阶段方法 `CoTMR` 在 Recall_sub@1 上提升了 2.48% (70.82% -> 73.30%)。
*   **与两阶段方法的差距：** 尽管表现出色，但 SDR-CIR 在 CIRR 子集上的性能仍略低于两阶段方法 `SEIZE`。论文解释这可能是因为子集需要更精细的判别能力，而 `SEIZE` 的语义编辑增量 (semantic editing increment) 提供了更强的线索。SDR-CIR 主要侧重于抑制参考图像的冗余，但总体上仍保持了竞争力。

### 6.1.3. FashionIQ 数据集结果分析
*   **属性修改的适用性：** FashionIQ 侧重于属性级别的修改。
*   **SDR-CIR 的优势：** SDR-CIR 在 ViT-B/32 和 ViT-L/14 两个主干网络上都实现了最佳的平均 R@10 和 R@50 性能，无论与文本反演方法还是免训练方法相比。
*   **具体提升：** 在 ViT-L/14 主干网络上，SDR-CIR 的平均 R@10 比当前最先进的方法 `CoTMR` 提升了 8.22% (32.62% -> 40.84%)。这表明 SDR-CIR 不仅擅长对象修改检索，在属性修改方面也表现出色。
*   **与 `SEIZE` 的对比：** 在 ViT-G/14 上，SDR-CIR 的平均 R@50 略低于 `SEIZE`。这可能表明更大的主干网络会进一步放大 `SEIZE` 中语义编辑增量的细粒度编辑线索的优势。
*   **效率优势：** 论文强调，SDR-CIR 在此数据集上效率更高。
*   **泛化能力总结：** 总体而言，这些结果验证了 SDR-CIR 在不同主干网络和数据集上的强大泛化能力，证明了其在缓解语义偏差的同时保持高检索准确性的能力。

## 6.2. 消融实验/参数分析

消融实验在 CIRCO 测试集和 FashionIQ 验证集上进行，使用 ViT-L/14 作为主干网络，以验证方法中不同模块的效果。

以下是原文 Table 3，展示了消融研究结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">CIRCO</th>
<th colspan="2">FashionIQ</th>
</tr>
<tr>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=10</th>
<th>k=50</th>
</tr>
</thead>
<tbody>
<tr>
<td>SDR-CIR</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>40.84</td>
<td>61.25</td>
</tr>
<tr>
<td colspan="6"><strong>1. Chain-of-Thought</strong></td>
</tr>
<tr>
<td>w/o CoT</td>
<td>24.82</td>
<td>25.61</td>
<td>27.79</td>
<td>38.57</td>
<td>59.26</td>
</tr>
<tr>
<td>w/o Selective extraction</td>
<td>30.65</td>
<td>31.42</td>
<td>33.82</td>
<td>40.51</td>
<td>60.15</td>
</tr>
<tr>
<td colspan="6"><strong>2. Key modules of SDR-CIR</strong></td>
</tr>
<tr>
<td>only description</td>
<td>26.61</td>
<td>27.54</td>
<td>30.01</td>
<td>32.26</td>
<td>52.61</td>
</tr>
<tr>
<td>+Anchor</td>
<td>28.56</td>
<td>29.48</td>
<td>31.39</td>
<td>38.41</td>
<td>59.01</td>
</tr>
<tr>
<td>+Debias</td>
<td>26.88</td>
<td>27.86</td>
<td>30.38</td>
<td>33.63</td>
<td>54.43</td>
</tr>
<tr>
<td>+SDR</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>40.84</td>
<td>61.25</td>
</tr>
<tr>
<td colspan="6"><strong>3. Different MLLMs</strong></td>
</tr>
<tr>
<td>Qwen2.5-VL-72B</td>
<td>27.82</td>
<td>28.46</td>
<td>30.80</td>
<td>38.71</td>
<td>58.60</td>
</tr>
<tr>
<td>GPT-4o-mini</td>
<td>30.67</td>
<td>31.33</td>
<td>33.66</td>
<td>39.26</td>
<td>59.85</td>
</tr>
<tr>
<td>GPT-4.1</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>40.84</td>
<td>61.25</td>
</tr>
</tbody>
</table>

### 6.2.1. 选择性思维链 (Selective CoT) 提示的效果
*   <strong>`w/o CoT` (无思维链推理)：</strong>
    *   在 CIRCO 数据集上，移除思维链 (CoT) 导致 mAP@5 显著下降 6.09% (从 30.91% 到 24.82%)。
    *   在 FashionIQ 数据集上，移除思维链导致平均 R@10 下降 2.27% (从 40.84% 到 38.57%)。
    *   **结论：** 这表明思维链推理对于理解参考图像和修改文本的重要性。FashionIQ 图像相对简单，噪声较少，因此性能下降幅度小于 CIRCO，说明思维链对更复杂图像的数据集效果更显著。
*   <strong>`w/o Selective extraction` (无选择性提取)：</strong>
    *   在 CIRCO 数据集上，CoT 中的全面图像理解导致 mAP@5 下降 0.26% (从 30.91% 到 30.65%)。
    *   在 FashionIQ 数据集上，平均 R@10 下降 0.33% (从 40.84% 到 40.51%)。
    *   **结论：** 这些小但一致的下降表明，仅靠 CoT 策略本身在处理冗余信息方面的能力有限，而`选择性提取`是有效减少视觉噪声的关键。

### 6.2.2. 语义去偏排名 (SDR) 关键步骤的效果
*   <strong>`only description` (仅使用描述)：</strong> 仅使用目标图像描述进行检索，作为基线。
*   <strong>$+Anchor$ (添加锚定步骤)：</strong> 在仅使用描述的基础上添加`锚定 (Anchor)`步骤。
    *   CIRCO 的 mAP@5 提升了 1.95% (从 26.61% 到 28.56%)。
    *   FashionIQ 的平均 Recall@10 提升了 6.15% (从 32.26% 到 38.41%)。
    *   **结论：** 这表明图像特征能够稳定有用的视觉语义并恢复被描述遗漏的线索。FashionIQ 上更大的提升幅度暗示该数据集上遗漏线索的情况更多，融合图像特征尤其有益。
*   <strong>$+Debias$ (添加去偏步骤，不使用锚定)：</strong> 在仅使用描述的基础上添加`去偏 (Debias)`步骤。
    *   CIRCO 和 FashionIQ 分别仅有轻微提升，分别为 0.27% (从 26.61% 到 26.88%) 和 1.37% (从 32.26% 到 33.63%)。
    *   **结论：** 这表明直接抑制描述中的视觉语义贡献可能会削弱有用的视觉信息，从而限制了其效益。
*   <strong>$+SDR$ (完整的 SDR-CIR)：</strong>
    *   与`only description`相比，CIRCO 的 mAP@5 提升了 4.30% (从 26.61% 到 30.91%)。
    *   FashionIQ 的平均 R@10 提升了 8.58% (从 32.26% 到 40.84%)。
    *   与$+Anchor$相比，两个数据集的性能进一步提升，CIRCO 提升 2.35% (从 28.56% 到 30.91%)，FashionIQ 提升 2.43% (从 38.41% 到 40.84%)。
    *   **结论：** 这证明了去偏在锚定之后效果更佳，并且`锚定 (Anchor)`和`去偏 (Debias)`这两个步骤在 SDR 模块内部是互补的。

### 6.2.3. 不同多模态大语言模型 (MLLM) 的比较
*   **GPT-4.1：** 取得了最佳结果 (CIRCO mAP@5 30.91%, FashionIQ R@10 40.84%)。
*   **Qwen2.5-VL-72B：** 在 CIRCO 上比 GPT-4.1 低 3.09%，在 FashionIQ 上低 2.13%。
*   **GPT-4o-mini：** 性能与 GPT-4.1 高度接近，在 CIRCO 上仅低 0.24%，在 FashionIQ 上低 1.58%，但效率更高。
*   **结论：** 这些结果表明 SDR-CIR 方法在不同多模态大语言模型下均表现强劲。

### 6.2.4. 语义去偏排名 (SDR) 对一阶段 ZS-CIR 方法的影响
以下是原文 Table 4，展示了 SDR 对不同一阶段方法的影响：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">SDR</th>
<th colspan="4">CIRCO (mAP@k)</th>
</tr>
<tr>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=50</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">OSrCIR [41]</td>
<td>×</td>
<td>21.83</td>
<td>22.46</td>
<td>24.49</td>
<td>25.44</td>
</tr>
<tr>
<td>√</td>
<td>25.86</td>
<td>26.56</td>
<td>28.73</td>
<td>29.77</td>
</tr>
<tr>
<td>∆</td>
<td>+4.03</td>
<td>+4.10</td>
<td>+4.24</td>
<td>+4.33</td>
</tr>
<tr>
<td rowspan="3">CoTMR [37]</td>
<td>×</td>
<td>26.52</td>
<td>27.13</td>
<td>29.51</td>
<td>30.56</td>
</tr>
<tr>
<td>√</td>
<td>30.12</td>
<td>30.74</td>
<td>33.20</td>
<td>34.28</td>
</tr>
<tr>
<td>∆</td>
<td>+3.60</td>
<td>+3.61</td>
<td>+3.69</td>
<td>+3.72</td>
</tr>
<tr>
<td rowspan="3">Ours</td>
<td>×</td>
<td>26.61</td>
<td>27.54</td>
<td>30.01</td>
<td>31.14</td>
</tr>
<tr>
<td>√</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>35.08</td>
</tr>
<tr>
<td>∆</td>
<td>+4.30</td>
<td>+3.96</td>
<td>+4.02</td>
<td>+3.94</td>
</tr>
</tbody>
</table>

*   **验证 SDR 模块的通用性：** 论文基于 ViT-L/14 主干网络验证了所提出的`语义去偏排名 (SDR)`机制的有效性。在公平比较下，将 SDR 模块应用到 `OSrCIR` 和 `CoTMR` 等一阶段 ZS-CIR 方法上。
*   **显著提升：** `OSrCIR` 和 `CoTMR` 在使用 SDR 模块后，mAP@5 分别实现了 4.03% 和 3.60% 的显著提升。
*   **结论：** 这表明 SDR 模块能够缓解多模态大语言模型 (MLLMs) 在一阶段方法中难以解决的语义偏差问题，验证了其即插即用 (plug-and-play) 的适用性和有效性。

### 6.2.5. 超参数 $\alpha$ 和 $\beta$ 的影响
以下是原文 Figure 4，展示了超参数 $\alpha$ 和 $\beta$ 的分析：

![Figure 4: Hyperparameter analysis of $\\alpha$ and $\\beta$ on CIRCO test set, CIRR test set and FashionIQ val set. All experiments are performed with the ViT-L/14.](images/4.jpg)
*该图像是图表，展示了超参数$\alpha$和$\beta$在CIRCO测试集、CIRR测试集和FashionIQ验证集上的分析结果。左侧（a）显示了不同$\alpha$值对各指标的影响，右侧（b）则展示了不同$\beta$值的效果。所有实验均使用ViT-L/14模型进行。图中不同颜色的曲线代表了不同的评估指标。*

原文 Figure 4: 超参数 $\alpha$ 和 $\beta$ 在 CIRCO 测试集、CIRR 测试集和 FashionIQ 验证集上的分析。所有实验均使用 ViT-L/14 模型进行。

*   <strong>超参数 $\alpha$ (图 4a)：</strong>
    *   随着 $\alpha$ 增加，大多数指标在适度范围内有所改善，之后性能开始下降。
    *   FashionIQ 在 $\alpha$ 增加到 0.2 左右时趋于稳定。
    *   值得注意的是，CIRR 的 Recall_sub@1 指标呈现持续下降趋势，这表明过大的 $\alpha$ 值可能会放大参考图像中的噪声，从而削弱子集检索的准确性（因为子集对细微差异更敏感）。
*   <strong>超参数 $\beta$ (图 4b)：</strong>
    *   随着 $\beta$ 增加，所有数据集的整体性能均有所改善。
    *   CIRCO 和 FashionIQ 在 $\beta$ 分别达到 0.35 和 0.4 时达到峰值。
    *   CIRR 的 Recall@1 和 Recall_sub@1 在 $\beta$ 为 0.5 时表现最佳。
*   **结论：** 适当的 $\alpha$ 可以有效地融合参考图像特征以补充信息，而适当的 $\beta$ 则可以有效惩罚语义偏差，从而提升整体性能。

### 6.2.6. 效率分析
以下是原文 Table 5，展示了训练无关方法在准确性和效率指标上的比较：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">Accuracy metic</th>
<th colspan="2">Efficiency metic (per query)</th>
</tr>
<tr>
<th>CIRCO (k=1)</th>
<th>FashionIQ (k=10)</th>
<th>Infer time (s)</th>
<th>Calls (times)</th>
</tr>
</thead>
<tbody>
<tr>
<td>LDRE [47]</td>
<td>31.12</td>
<td>32.49</td>
<td>19.24</td>
<td>15</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>32.46</td>
<td>43.05</td>
<td>19.24</td>
<td>15</td>
</tr>
<tr>
<td>OSrCIR [41]</td>
<td>24.73</td>
<td>34.21</td>
<td>1.01</td>
<td>1</td>
</tr>
<tr>
<td>CoTMR [37]</td>
<td>29.59</td>
<td>35.25</td>
<td>1.48</td>
<td>2</td>
</tr>
<tr>
<td>Ours</td>
<td>33.05</td>
<td>45.19</td>
<td>0.37</td>
<td>1</td>
</tr>
</tbody>
</table>

*   **效率对比：** 论文比较了常见两阶段和一阶段免训练方法的有效性 (accuracy) 和效率 (efficiency)。
*   **SDR-CIR 的优势：** 在使用相同的 ViT-G/14 主干网络下，SDR-CIR 在现有免训练方法中取得了最佳性能，同时显著减少了推理时间。
*   **与 `SEIZE` 的对比：** `SEIZE` 的准确性与 SDR-CIR 接近，但每个查询需要高达 19.24 秒的推理时间，因为它要生成 15 个描述并调用语言模型 15 次。
*   **结论：** SDR-CIR 在有效性和效率方面都达到了最佳表现，这得益于其一阶段的生成过程和高效的排名机制。

## 6.3. 定性结果

以下是原文 Figure 5，展示了 SDR-CIR 对冗余和缺失信息的鲁棒性：

![Figure 5: Robustness to redundant and missing information on CIRCO and FashionIQ. Top-2 retrieval results of SDR-CIR and a description-only baseline (Base Result) are compared. Red text marks redundant or missing information; green boxes indicate targets.](images/5.jpg)
*该图像是检索示例，展示了SDR-CIR与描述基线在CIRCO和FashionIQ数据集上的检索效果对比。图中红色文字标记了冗余或缺失的信息，而绿色框表示目标对象。*

原文 Figure 5: SDR-CIR 与仅描述基线 (Base Result) 在 CIRCO 和 FashionIQ 上的前 2 个检索结果对比。红色文字标记冗余或缺失信息；绿色框表示目标。

### 6.3.1. 冗余偏差 (Redundant Bias) 示例 (CIRCO)
*   <strong>案例分析 (图 5a)：</strong>
    *   第一个示例中，原始描述包含冗余短语“一顶红色牛仔帽 (a red cowboy hat)”。
    *   第二个示例中，原始描述包含“站在浅水中的人旁边 (near a person standing in shallow water)”，这与目标图像无关。
*   **仅描述检索的失败：** 仅使用描述进行检索时，由于这些冗余细节的引导，检索到了不正确的图像。
*   **SDR-CIR 的成功：** 通过`语义去偏排名 (Semantic Debias Ranking)`机制，SDR-CIR 在保留主要信息的同时，减少了冗余短语的影响，从而成功检索到目标图像。

### 6.3.2. 遗漏偏差 (Omission Bias) 示例 (FashionIQ)
*   <strong>案例分析 (图 5b)：</strong>
    *   这些示例中，描述遗漏了与目标图像相关的关键线索（例如，项链 (necklace)、裙子长度 (skirt length)）。
*   **仅描述检索的失败：** 由于遗漏了这些关键线索，仅使用描述进行检索未能找到目标图像。
*   **SDR-CIR 的成功：** SDR-CIR 方法能够补充描述中遗漏的元素，使其在这两种情况下都成功检索到目标图像。

# 7. 总结与思考

## 7.1. 结论总结

本文提出了 SDR-CIR，一个针对零样本组合图像检索 (ZS-CIR) 任务的一阶段 (one-stage)、免训练 (training-free) 方法。SDR-CIR 旨在顺序解决由参考图像引入的语义偏差问题。其核心创新点包括：

1.  <strong>选择性思维链 (Selective CoT)：</strong> 在描述生成阶段，通过引导多模态大语言模型 (MLLM) 有选择性地提取与修改文本相关的视觉内容，从源头上减少了视觉噪声，使得生成的描述更加精确。
2.  <strong>语义去偏排名 (Semantic Debias Ranking)：</strong> 在排名阶段，该模块采用“锚定 (Anchor)”和“去偏 (Debias)”两步策略。
    *   “锚定 (Anchor)”：通过融合参考图像特征和生成描述特征，构建鲁棒的组合查询，强化有用语义并补充遗漏线索。
    *   “去偏 (Debias)”：显式建模参考图像对描述的视觉语义贡献，并将其作为惩罚项纳入相似度分数，从而抑制冗余语义。

        在 CIRR、CIRCO 和 FashionIQ 三个标准 CIR 基准数据集上的实验结果表明，SDR-CIR 在一阶段方法中达到了最先进的 (state-of-the-art) 性能，并有效缓解了语义偏差，同时保持了高效率。

## 7.2. 局限性与未来工作

论文作者指出了 SDR-CIR 在处理某些复杂情况下的局限性，并暗示了未来的研究方向：

1.  <strong>歧义编辑意图 (Ambiguous editing intentions)：</strong> 当修改文本未明确指定具体的变化，而仅要求“不同于参考图像”时（例如，“照片以不同角度拍摄”、“颜色不同”但未指定具体角度或颜色），多模态大语言模型 (MLLM) 往往难以推断出预期的结果。这导致生成的描述可能具有合理性，但缺乏区分度，无法准确匹配目标图像。
2.  **复杂视觉场景：** 在复杂的视觉场景中，当修改意图不明确时，MLLM 可能无法确定具体的改变方向，或者即使确定了，其推断的改变也可能与实际目标图像不符。
3.  <strong>未充分指定的编辑 (Underspecified edits)：</strong> 这类编辑没有提供足够的约束来唯一确定目标图像，从而导致生成的描述虽然貌似合理，但实际上无法有效区分真实目标。

**未来可能的研究方向：**
*   **增强对模糊修改的理解：** 未来的工作可以集中于开发更强大的 MLLM 推理机制，使其能够更好地处理模糊或不充分指定的修改指令，例如通过引入用户反馈、上下文信息或更复杂的常识推理。
*   **交互式检索：** 结合交互式方法，允许用户在检索过程中提供更多澄清或偏好，以解决歧义性。

## 7.3. 个人启发与批判

### 7.3.1. 个人启发
1.  **对症下药的框架设计：** SDR-CIR 的设计理念非常清晰和有针对性，针对免训练一阶段 ZS-CIR 的两个核心问题（视觉噪声和语义偏差），分别提出了`选择性思维链 (Selective CoT)`和`语义去偏排名 (Semantic Debias Ranking)`。这种分阶段、逐一击破问题的思路值得借鉴。
2.  <strong>“锚定-去偏”</strong>策略的巧妙： `语义去偏排名 (Semantic Debias Ranking)`中的“锚定 (Anchor)”和“去偏 (Debias)”两步是互补且精妙的。`锚定 (Anchor)`通过融合原始图像特征来“保全”有用信息并“补足”遗漏线索，避免了简单“去偏”可能带来的过度惩罚。而`去偏 (Debias)`则专注于消除冗余，两者结合实现了更平衡的检索。这种考虑信息“补充”和“抑制”两方面的设计非常实用。
3.  **MMLM 引导的重要性：** `选择性思维链 (Selective CoT)`在描述生成阶段就介入 MLLM 的推理，通过修改文本引导 MLLM 关注相关视觉内容，这比传统 CoT 后期过滤的效率更高，也更符合人类的“目的性观察”直觉。这提醒我们在利用大模型时，如何在早期阶段就注入任务意图以提升效率和准确性是关键。
4.  **模型通用性与效率：** 实验结果表明 SDR 模块具有良好的即插即用性，能够有效提升其他一阶段方法的性能。同时，SDR-CIR 在保持先进性能的同时，显著提高了效率，这对于实际应用至关重要。

### 7.3.2. 批判与潜在改进
1.  **对 MLLM 初始生成的依赖：** SDR-CIR 的性能在很大程度上仍然依赖于多模态大语言模型 (MLLM) 生成的初始描述质量。如果 MLLM 自身对参考图像或修改文本的理解存在根本性偏差，或生成了质量非常差的描述，那么后续的去偏排名模块可能难以完全弥补。未来可以探索如何提高 MLLM 描述生成的鲁棒性，例如通过多 MLLM 集成或自我修正机制。
2.  **超参数的敏感性与泛化：** 论文中 $\alpha$ 和 $\beta$ 的值针对不同数据集进行了精细调整，这表明这些超参数可能对特定数据集敏感。虽然这是常见做法，但对于真正的零样本 (zero-shot) 场景，如果需要在新数据集上进行推理，可能需要额外的验证或自适应调整机制。未来的工作可以研究如何让这些超参数更具泛化性，例如通过学习或启发式规则进行动态调整。
3.  **处理高度模糊指令的挑战：** 论文也指出了对“不同角度”、“不同颜色”这类高度模糊指令的局限性。这不仅仅是描述生成的问题，更是对修改意图深层理解的挑战。SDR-CIR 在此方面仍有提升空间。或许可以引入外部知识库、用户画像或通过与用户进行多轮交互 (dialogue-based CIR) 来解决这种深层歧义。
4.  <strong>“语义贡献”</strong>建模的近似性： `去偏 (Debias)`步骤中，参考图像的视觉语义贡献 $S_i$ 被近似为 $S_d - S_m$。这种近似可能不总是完全准确，因为 $F_d$ 本身可能已经包含了 MLLM 内部推理后的语义，而不仅仅是简单的 $F_r$ 和 $F_m$ 的线性组合。更精细地建模 MLLM 内部的语义交互和贡献，可能会带来进一步的提升。
5.  **CLIP 特征的局限性：** 尽管 CLIP 作为检索编码器表现出色，但其特征空间可能在某些细粒度修改或特定领域（如时尚）中存在局限。结合更专业的视觉-语言表示或领域适应技术，可能会进一步提升性能。