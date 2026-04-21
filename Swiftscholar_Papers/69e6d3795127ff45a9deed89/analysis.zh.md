# 1. 论文基本信息
## 1.1. 标题
论文标题为《MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model》，核心主题是提出一种<strong>多轮对比学习（MuCo）</strong> 框架，用于优化基于多模态大语言模型的通用多模态嵌入模型的训练效率与表示质量。
## 1.2. 作者与隶属机构
作者团队及隶属机构如下：
*   Geonmo Gu: NAVER AI Lab、高丽大学
*   Byeongho Heo、Jaemyung Yu、Jaehui Hwang、Taekyung Kim、Sangdoo Yun、Dongyoon Han: NAVER AI Lab
*   HeeJae Jun、Yoohoon Kang: NAVER AI Search Platform
*   Sangmin Lee: 高丽大学
    该团队来自韩国顶级互联网公司NAVER的AI实验室，在多模态学习、高效训练领域有深厚的研究积累。
## 1.3. 发表状态
当前为arXiv预印本，尚未正式发表在期刊或会议上。
## 1.4. 发表时间
UTC时间2026年2月6日。
## 1.5. 摘要
论文针对传统单轮对比学习训练多模态嵌入模型时的两个核心缺陷：1）单轮独立处理查询-目标对，忽略同一张图像对应多个查询的上下文关联，学到的表示一致性差；2）训练效率低，每对样本都需要重新编码图像，扩大batch的计算成本极高。提出了对话启发的多轮对比学习框架MuCo，利用多模态大语言模型的对话能力，将同一张图像对应的多个相关查询-目标对拼接为多轮对话序列，在**单次前向传播**中同时提取多组嵌入，图像仅需编码一次，有效batch规模可扩大k倍，而计算量仅增加不到3%。搭配新构建的5M规模多模态多轮数据集M3T，MuCo在MMEB和M-BEIR两个通用多模态嵌入基准上取得了最先进的性能，同时大幅提升了训练效率和跨模态表示一致性。
## 1.6. 原文链接
*   预印本主页：https://arxiv.org/abs/2602.06393
*   PDF链接：https://arxiv.org/pdf/2602.06393v2

    ---
# 2. 整体概括
## 2.1. 研究背景与动机
### 2.1.1. 核心问题
当前基于多模态大语言模型（MLLM）的通用多模态嵌入模型普遍采用**单轮对比学习**范式：每个查询-目标对被视为独立样本，一次前向传播仅处理一对样本，每次都需要重新编码图像。该范式存在两个不可忽视的瓶颈：
1.  **训练效率瓶颈**：图像编码的计算成本远高于文本（平均每张图像的编码开销是平均长度文本的18倍以上），为了提升对比学习的性能需要扩大batch规模，而batch扩大意味着需要处理更多图像，计算开销呈线性增长，严重限制了模型的 scalability。
2.  **表示质量瓶颈**：同一张图像可以对应多个语义相关的查询（比如分类、属性查询、VQA问题等），单轮范式将这些查询独立处理，完全忽略了它们之间的上下文关联，导致学到的跨模态表示语义一致性差，泛化能力不足。
### 2.1.2. 创新切入点
论文巧妙利用了MLLM本身具备的因果对话、多轮上下文理解能力，将同一张图像对应的多个查询-目标对拼接为多轮对话序列，仅需一次前向传播即可处理多对样本，图像仅编码一次，后续轮次均为低开销的文本处理，既解决了效率问题，又能利用多轮上下文关联提升表示的一致性。
## 2.2. 核心贡献
论文的核心贡献可归纳为三点：
1.  **方法创新**：提出了多轮对比学习（MuCo）框架，首次将多轮对话范式引入多模态对比学习，在计算开销仅增加≤3%的前提下，将有效batch规模扩大7倍，同时利用多轮上下文监督提升了表示的语义一致性。
2.  **数据贡献**：构建了5M规模的多模态多轮数据集M3T（Multi-Modal Multi-Turn），包含35M查询-目标对，覆盖分类、检索、VQA等多种任务，为多轮多模态嵌入训练提供了高质量的数据支撑。
3.  **性能突破**：在通用多模态嵌入的两个权威基准MMEB和M-BEIR上取得了新的SOTA性能，零样本性能较之前的SOTA提升3.0%，微调性能提升1.6%~1.7%，同时具备更强的分布外泛化能力。

    ---
# 3. 预备知识与相关工作
## 3.1. 基础概念
为了帮助初学者理解论文，首先对核心基础概念进行解释：
### 3.1.1. 多模态大语言模型（MLLM）
多模态大语言模型是在大语言模型的基础上加入视觉编码器，能够同时处理图像、文本等多种模态的输入，具备强大的多模态语义理解、对话和推理能力的大模型，比如本文用到的Qwen2-VL、开源的LLaVA系列等。它天然支持多轮对话的因果注意力机制（即当前轮的输出可以看到之前所有轮的输入内容），是MuCo框架的骨干基础。
### 3.1.2. 多模态嵌入
多模态嵌入的目标是将文本、图像、图文混合等不同模态的输入，映射到同一个统一的高维向量空间，使得语义相近的内容对应的向量距离更近，语义无关的内容对应的向量距离更远。训练好的多模态嵌入模型可以直接支持分类、检索、视觉问答、视觉定位等多种下游任务，不需要任务特定的微调。
### 3.1.3. 对比学习与InfoNCE损失
对比学习是训练嵌入模型的主流范式，核心目标是拉近**正样本对**（语义相关的内容，比如一张图和它的描述）的向量距离，推远**负样本对**（语义无关的内容）的向量距离。最常用的损失函数是InfoNCE损失，原文给出的公式如下：
$$
\mathcal { L } = \frac { 1 } { | \mathcal { B } | } \sum _ { ( q _ { i } , p _ { i } ) \in \mathcal { B } } - \log \frac { \phi ( q _ { i } , p _ { i } ) } { \sum _ { p \in \mathcal { N } \cup \{ p _ { i } \} } \phi ( q _ { i } , p ) } ,
$$
其中：
*   $\mathcal{B}$ 是当前的小批量样本集合；
*   $q_i$ 是查询样本，$p_i$ 是 $q_i$ 对应的正样本；
*   $\mathcal{N}$ 是当前batch中的负样本集合；
*   相似性函数 $\phi ( q _ { i } , p _ { i } ) = \exp ( f ( q _ { i } ) ^ { \top } f ( p _ { i } ) / \tau )$，其中 $f(\cdot)$ 是嵌入编码器，输出L2归一化的向量，$\tau$ 是温度系数，用于控制相似性得分的区分度。
    对比学习的性能高度依赖batch规模：更大的batch可以提供更多的负样本，让模型学到的表示更有区分度，但传统单轮范式下扩大batch的成本极高。
## 3.2. 关键前人工作
当前基于MLLM的多模态嵌入研究主要分为两个方向：
1.  **数据优化方向**：通过扩大数据规模、提升数据质量来提升嵌入性能，比如mmE5自动生成高质量的多模态对齐监督数据，MMRet构造了26M规模的合成数据集，VLM2Vec统一了36个多模态数据集进行训练。
2.  **训练流程优化方向**：通过改进训练策略提升性能，比如E5-V通过MLLM蒸馏统一表示，UniME、LLaVE通过优化负采样提升监督信号质量，B3通过智能batch构建提升batch利用率，缓解大batch的成本压力。
    但以上所有方法都沿用了单轮对比学习的范式，没有利用MLLM的多轮对话能力，仍然存在效率和表示质量的瓶颈。
## 3.3. 技术演进脉络
通用多模态嵌入的技术演进可以分为三个阶段：
1.  <strong>CLIP时代（2021年起）</strong>：以CLIP为代表，采用图像-文本对的单轮对比学习，首次实现了较强的跨模态对齐能力，但语义理解能力有限，难以处理复杂的指令和推理任务。
2.  <strong>MLLM适配时代（2024年起）</strong>：随着MLLM的成熟，研究者开始用MLLM作为嵌入模型的骨干，利用MLLM强大的语义理解和指令跟随能力，大幅提升了多模态嵌入的性能，代表工作包括VLM2Vec、mmE5等，但仍然沿用单轮对比学习的范式。
3.  <strong>多轮范式时代（2026年，本文）</strong>：本文首次将MLLM的对话能力与对比学习结合，提出多轮对比学习范式，同时解决了单轮范式的效率和表示质量瓶颈。
## 3.4. 本文差异化优势
与之前的单轮方法相比，MuCo的核心优势在于：
1.  **效率优势**：同样的硬件条件下，有效batch规模可以扩大k倍，计算开销仅增加不到3%，远优于传统扩大batch的方式。
2.  **质量优势**：利用多轮上下文的复合监督信号，学到的表示语义一致性更强，分布外泛化能力更优。
3.  **兼容性优势**：提出了单轮数据集的适配微调策略，不需要修改现有下游基准的评估流程，即可无缝适配当前的通用多模态嵌入评估体系。

    ---
# 4. 方法论
MuCo采用两阶段训练流程：第一阶段在多轮数据集M3T上进行预训练，学习多轮上下文的对齐能力；第二阶段适配单轮基准数据集进行微调，对齐下游任务的评估范式。
## 4.1. 预训练阶段：多轮对比学习（MuCo）
### 4.1.1. 核心思想
利用MLLM的因果对话能力，将同一张图像对应的多个查询-目标对拼接为多轮对话序列，单次前向传播即可处理多对样本，图像仅需编码一次，后续轮次均为低开销的文本处理，同时通过多轮上下文的复合监督提升表示质量。
下图（原文Figure1）展示了传统单轮对比学习与MuCo的核心差异：

![该图像是示意图，展示了传统的对比学习与多轮对比学习（MuCo）之间的区别。左侧部分说明了传统对比学习的单轮框架，使用单个图像和多个不同查询进行编码，每个查询独立处理；右侧则展示了多轮对比学习的框架，在同一前向传播中处理多个相关查询，从而提高训练效率和表示一致性。](images/1.jpg)
*该图像是示意图，展示了传统的对比学习与多轮对比学习（MuCo）之间的区别。左侧部分说明了传统对比学习的单轮框架，使用单个图像和多个不同查询进行编码，每个查询独立处理；右侧则展示了多轮对比学习的框架，在同一前向传播中处理多个相关查询，从而提高训练效率和表示一致性。*

### 4.1.2. 多轮对话模板
MuCo设计了专用的多轮对话模板，结构如下：
1.  第一轮：系统提示 + 输入图像 + 第一个用户查询，助手的回复末尾加入特殊的嵌入标记 $<|emb|>$，该标记对应的隐藏状态即为第一轮的嵌入向量。
2.  后续轮次：直接追加新的用户查询和对应的助手回复，每个回复末尾都加入 $<|emb|>$ 标记，提取对应轮次的嵌入向量。
    由于MLLM采用因果注意力机制，每一轮的输入都可以看到之前所有轮的所有内容，因此每一轮的嵌入都包含了之前所有轮的上下文信息。
下图（原文Figure2）展示了多轮对话模板的结构：

![该图像是示意图，展示了多轮对比学习 (MuCo) 框架在多模态大语言模型中的应用。图中显示了用户与助手之间的多轮查询和目标对话，通过正负样本对实现复合监督，从而提高了数据效率和训练效果。每个查询与相应目标相互关联，强调了相同上下文下多个查询的处理方式。](images/2.jpg)
*该图像是示意图，展示了多轮对比学习 (MuCo) 框架在多模态大语言模型中的应用。图中显示了用户与助手之间的多轮查询和目标对话，通过正负样本对实现复合监督，从而提高了数据效率和训练效果。每个查询与相应目标相互关联，强调了相同上下文下多个查询的处理方式。*

### 4.1.3. 损失函数
对于第i张图像 $\mathcal{T}_i$，它对应的第j个查询为 $q_i^j$，对应的正目标为 $p_i^j$，多轮拼接后的第j轮查询和正目标定义为：
$$
\bar { q } _ { i } ^ { j } = ( \mathcal { T } _ { i } , ( q _ { i } ^ { l } ) _ { l \leq j } ) , \quad \bar { p } _ { i } ^ { j } = ( p _ { i } ^ { l } ) _ { l \leq j } .
$$
其中 $\bar{q}_i^j$ 是拼接了图像和前j轮所有查询的第j轮查询输入，$\bar{p}_i^j$ 是拼接了前j轮所有正目标的第j轮正样本。
由于同一张图像对应的不同轮次的正目标语义高度相关，不能作为彼此的负样本，因此MuCo设计了专用的负样本集：
$$
\mathcal { N } _ { i } = \{ \bar { p } _ { k } ^ { l } \ | \ ( \cdot , \bar { p } _ { k } ^ { l } ) \in B , \ k \neq i \}.
$$
即负样本仅包含来自其他图像的所有正样本，排除了同一张图像的所有正样本。最终的MuCo对比损失为：
$$
\mathcal { L } _ { \sf M u C o } = \frac { 1 } { | \boldsymbol { \it B } | } \sum _ { ( \bar { q } _ { i } ^ { j } , \bar { p } _ { i } ^ { j } ) \in \boldsymbol { B } } - \log \frac { \phi ( \bar { q } _ { i } ^ { j } , \bar { p } _ { i } ^ { j } ) } { \sum _ { \bar { p } \in \mathcal { N } _ { i } \cup \{ \bar { p } _ { i } ^ { j } \} } \phi ( \bar { q } _ { i } ^ { j } , \bar { p } ) } .
$$
### 4.1.4. Logit掩码策略
为了实现上述负样本过滤，MuCo采用了logit掩码策略：在计算对比损失的logit矩阵时，将所有来自同一张图像的样本对的logit值设为 $-\infty$，避免它们被当作负样本参与损失计算。
下图（原文Figure3）展示了传统方法与MuCo的logit矩阵差异：

![Figure 3. Logit masking strategy in our MuCo framework. (a) The conventional method with a batch size of $N = 4$ yields a $N \\times N$ (i.e. $4 \\times 4 )$ matrix. In contrast, MuCo (b) uses a batch size of $N = 2$ and $k = 4$ turns to construct a larger $N k \\times N k$ (i.e. $8 \\times 8 )$ matrix. Crucially, our method masks out pairs originating from the same image (gray, $- \\infty )$ to prevent a semantic overlap issue. True positives (blue, `+` ) and true negatives (orange, `-` are used for the loss. Crucially, other pairs originating from the same image (gray, $- \\infty )$ are masked to prevent a semantic overlap issue.](images/3.jpg)
*该图像是示意图，展示了MuCo框架中的Logit掩蔽策略。从左侧的(a)部分可以看到传统方法的logit矩阵，批量大小为$N = 4$，生成一个$4 \times 4$的矩阵。而右侧的(b)部分展示了MuCo方法，使用批量大小$N = 2$和$k = 4$轮次，构建了一个$8 \times 8$的矩阵。此外，图中用灰色表示的$- \infty$对来自同一图像的配对进行了掩蔽，以防止语义重叠。*

例如当batch大小为2、每图4轮时，MuCo的logit矩阵大小为$8\times8$，其中对角线上的蓝色块是正样本对，灰色块是被掩码的同图像样本对，其余橙色块是负样本对。
### 4.1.5. M3T数据集构建
由于公开的多模态多轮对话数据极少，论文专门构建了M3T数据集，构建流程如下：
1.  从DataComp数据集中采样500万张图像，要求图像的最小边长不低于512像素，保证MLLM可以清晰识别图像中的细节。
2.  用先进的MLLM Qwen2.5-VL-75B对每张图像生成密集字幕，提取图像中的所有关键对象、属性、空间关系等视觉信息。
3.  用大语言模型GPT-OSS-120B基于密集字幕，为每张图像生成7个查询-目标对：1个分类任务对、1个检索任务对、5个VQA任务对（2个全局理解、2个局部细节、1个创意推理）。
    最终M3T数据集包含500万张图像、3500万查询-目标对，覆盖了通用多模态嵌入需要的所有核心任务类型。
## 4.2. 微调阶段：单轮数据集适配
当前的通用多模态嵌入基准都是单轮样本结构，为了让预训练好的MuCo适配这些基准，论文提出了基于掩码重建的单轮样本多轮化策略。
### 4.2.1. 核心思想
通过提示引导模型对掩码的配对样本进行重建，模拟多轮交互，让模型充分学习查询与正目标之间的语义关联，同时利用预训练阶段学到的多轮上下文能力，提升单轮嵌入的质量。推理时仅使用第一轮的嵌入，完全兼容现有单轮评估范式。
### 4.2.2. 多轮模拟模板
对于一个单轮样本对 `(q,p)`（q是查询，p是正目标），构造两种增强样本：
1.  查询增强样本 $q'$：第一轮输入q，提取第一轮嵌入（推理时使用的嵌入），第二轮加入提示和被掩码的p，引导模型重建p，第二轮末尾加$<|emb|>$。
2.  正目标增强样本 $p'$：第一轮输入p，提取第一轮嵌入，第二轮加入提示和被掩码的q，引导模型重建q，第二轮末尾加$<|emb|>$。
    最终每个原始样本可以得到4组训练对：$\{(q,p), (q,p'), (q',p), (q',p')\}$，充分利用多轮监督信号优化初始轮的嵌入质量。
下图（原文Figure4）展示了微调阶段的多轮模板结构：

![Figure 4. Multi-turn template for fine-tuning MuCo on singlepair datasets. We illustrate Query (left) and Positive (right) templates. The initial query (cyan) is reused as a masked target on the Positive side, and the positive target (pink) becomes a masked target on the Query side. This process simulates multi-turn interactions from a single pair, guiding the model to reconstruct its counterpart and enrich the learned embeddings.](images/4.jpg)
*该图像是示意图，展示了针对单对数据集的MuCo微调中的多回合模板。左侧为查询，右侧为正例模板，初始查询在正例中作为掩码目标重用，而正例的目标则在查询中成为掩码目标。这一过程模拟了来自单一对的多回合交互，指导模型重构对应内容，丰富学习的嵌入。*

### 4.2.3. 微调损失函数
微调阶段的对比损失与预训练阶段类似，仅需要调整负样本集：排除同一原始样本对应的增强样本（因为它们语义高度相关，不能作为负样本）。负样本集定义为：
$$
\mathcal { N } _ { p _ { i } } = \{ p \vert ( \cdot , p ) \in \bar { B } \} \backslash \{ p _ { i } , p _ { i } ^ { - } \} ,
$$
其中 $p_i^-$ 是 $p_i$ 对应的增强版本（如果 $p_i$ 是原始样本，则 $p_i^-$ 是对应的增强样本，反之亦然）。最终微调损失为：
$$
\mathcal { L } = \frac { 1 } { | \bar { \mathcal { B } } | } { \sum _ { ( q _ { i } , p _ { i } ) \in \bar { \mathcal { B } } } } - \log \frac { \phi ( q _ { i } , p _ { i } ) } { \displaystyle \sum_{p \in \mathcal { N } _ { p _ { i } } \cup \{ p _ { i } \}} \phi ( q _ { i } , p ) } .
$$
其中 $\bar{B}$ 是增强后的样本集合。

---
# 5. 实验设置
## 5.1. 数据集
### 5.1.1. 预训练数据集

| 数据集 | 规模 | 特点 |
| --- | --- | --- |
| M3T（本文） | 5M图像，35M查询-目标对 | 多轮结构，覆盖分类、检索、VQA任务 |
| mmE5 | 0.6M样本 | 单轮结构，高质量合成多模态数据 |
| MegaPairs | 26M样本 | 单轮结构，大规模检索任务数据 |

### 5.1.2. 微调与评估数据集
1.  **MMEB基准**：包含36个多模态数据集，覆盖分类、VQA、检索、视觉定位4类核心任务，其中20个数据集为分布内（ID）训练集，16个为分布外（OOD）测试集，专门用于评估通用多模态嵌入的能力和泛化性。
2.  **M-BEIR基准**：包含16个检索数据集，覆盖8种检索任务（单模→单模、单模→多模、多模→单模、多模→多模），是评估多模态检索能力的权威基准。
### 5.1.3. M3T样本示例
以下是M3T数据集中的一个样本示例，帮助读者直观理解数据结构：
*   图像内容：一枚方形主石的钻石戒指
*   分类查询：What type of jewelry is being described? 答案：Ring
*   全局VQA查询1：Explain the shape contrast between the center stone and the surrounding diamonds. 答案：The center square-cut gemstone contrasts with the round diamonds.
*   局部VQA查询1：Identify the feature that secures the central gemstone. 答案：Four prongs.
*   创意VQA查询：What narrative could one infer about the wearer based on the diamond setting? 答案：Perhaps the wearer values sharp elegance, akin to disciplined thoughts, reflected in the square-cut stone.
*   检索查询：Provide a detailed description of the jewelry item presented. 答案：A ring featuring a prominent square-cut gemstone at its center, surrounded by a halo of smaller round diamonds...
## 5.2. 评估指标
论文采用两种主流的检索评估指标，以下按要求分别解释：
### 5.2.1. Precision@1（P@1）
1.  **概念定义**：衡量所有查询中，返回的排名第一的结果是正样本的比例，关注最相关结果的准确性，适合分类、VQA、匹配类任务的评估。
2.  **数学公式**：
    $$\text{Precision@1} = \frac{N_{\text{top1\_correct}}}{N_{\text{total}}}$$
3.  **符号解释**：$N_{\text{top1\_correct}}$ 是top1结果为正样本的查询数量，$N_{\text{total}}$ 是总查询数量。
### 5.2.2. Recall@k（R@k）
1.  **概念定义**：衡量所有查询中，前k个返回结果中至少包含一个正样本的比例，关注正样本的召回能力，适合检索类任务的评估，本文中k取1、5、10。
2.  **数学公式**：
    $$\text{Recall@k} = \frac{N_{\text{topk\_has\_positive}}}{N_{\text{total}}}$$
3.  **符号解释**：$N_{\text{topk\_has\_positive}}$ 是前k个结果中包含至少一个正样本的查询数量，$N_{\text{total}}$ 是总查询数量。
## 5.3. 对比基线
论文选择了当前最先进的多模态嵌入模型作为对比基线，具有充分的代表性：
*   2B及以下参数量基线：CLIP、VLM2Vec-4B、LLaVE-2B、UniME-4B、B3-2B、MoCa-3B
*   7B及以上参数量基线：VLM2Vec-7B、MMRet-7B、mmE5-11B、LLaVE-7B、UniME-7B、B3-7B、MoCa-7B、LamRA-Ret-7B
## 5.4. 实现细节
*   骨干模型：采用Qwen2-VL作为MLLM骨干，包含2B和7B两种参数量版本。
*   微调策略：使用LoRA（低秩适配）仅微调LLM部分，冻结视觉编码器，LoRA秩为64，缩放因子α=64。
*   训练配置：所有阶段训练1个epoch，使用32张NVIDIA A100 80GB GPU，全局batch大小为1024，对比学习温度τ=0.02，固定学习率为5e-5。
*   微调掩码策略：随机掩码配对样本50%的单词，离线将图像转为文本描述使用Qwen2-VL-7B。

    ---
# 6. 实验结果与分析
## 6.1. 核心主实验结果
### 6.1.1. MMEB基准结果
下表是原文Table1的完整MMEB基准Precision@1结果：

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th rowspan="2"># Params</th>
<th colspan="4">Per Meta-Task Score</th>
<th colspan="3">Average Score</th>
</tr>
<tr>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>Grounding</th>
<th>ID</th>
<th>OOD</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9">Zeroshot setting (pretrained) on MMEB benchmark</td>
</tr>
<tr>
<td>CLIP [53]</td>
<td>0.4B</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>−</td>
<td>−</td>
<td>37.8</td>
</tr>
<tr>
<td>MagicLens [67]</td>
<td>0.6B</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td>−</td>
<td>−</td>
<td>27.8</td>
</tr>
<tr>
<td>E5-V [29]</td>
<td>8B</td>
<td>21.8</td>
<td>4.9</td>
<td>11.5</td>
<td>19.0</td>
<td>−</td>
<td>−</td>
<td>13.3</td>
</tr>
<tr>
<td>MMRet [69]</td>
<td>7B</td>
<td>47.2</td>
<td>18.4</td>
<td>56.5</td>
<td>62.2</td>
<td>−</td>
<td>−</td>
<td>44.0</td>
</tr>
<tr>
<td>mmE5 [8]</td>
<td>11B</td>
<td>60.6</td>
<td>55.7</td>
<td>54.7</td>
<td>72.4</td>
<td>−</td>
<td>−</td>
<td>58.6</td>
</tr>
<tr>
<td>MuCo-2B</td>
<td>2B</td>
<td>53.6</td>
<td>59.9</td>
<td>55.2</td>
<td>74.6</td>
<td>−</td>
<td>−</td>
<td>58.2</td>
</tr>
<tr>
<td>MuCo-7B</td>
<td>7B</td>
<td>56.0</td>
<td>64.7</td>
<td>58.9</td>
<td>75.7</td>
<td>−</td>
<td>−</td>
<td>61.6</td>
</tr>
<tr>
<td colspan="9">Fine-tuning on MMEB benchmark (&lt; 7B Models)</td>
</tr>
<tr>
<td>CLIP [53]</td>
<td>0.4B</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>45.4</td>
</tr>
<tr>
<td>VLM2Vec [30]</td>
<td>4B</td>
<td>54.8</td>
<td>54.9</td>
<td>62.3</td>
<td>79.5</td>
<td>66.5</td>
<td>52.0</td>
<td>60.1</td>
</tr>
<tr>
<td>LLaVE [33]</td>
<td>2B</td>
<td>62.1</td>
<td>60.2</td>
<td>65.2</td>
<td>84.9</td>
<td>69.4</td>
<td>59.8</td>
<td>65.2</td>
</tr>
<tr>
<td>UniME [21]</td>
<td>4B</td>
<td>54.8</td>
<td>55.9</td>
<td>64.5</td>
<td>81.8</td>
<td>68.2</td>
<td>52.7</td>
<td>64.2</td>
</tr>
<tr>
<td>B3-2B [58]</td>
<td>2B</td>
<td>67.0</td>
<td>61.2</td>
<td>70.9</td>
<td>79.9</td>
<td>72.1</td>
<td>63.1</td>
<td>68.1</td>
</tr>
<tr>
<td>MoCa-3B [7]</td>
<td>3B</td>
<td>59.8</td>
<td>62.9</td>
<td>70.6</td>
<td>88.6</td>
<td>72.3</td>
<td>61.5</td>
<td>67.5</td>
</tr>
<tr>
<td>MuCo-2B</td>
<td>2B</td>
<td>66.2</td>
<td>65.6</td>
<td>70.1</td>
<td>85.8</td>
<td>72.9</td>
<td>65.0</td>
<td>69.5</td>
</tr>
<tr>
<td colspan="9">Fine-tuning on MMEB benchmark (≥ 7B Models)</td>
</tr>
<tr>
<td>VLM2Vec [30]</td>
<td>7B</td>
<td>61.2</td>
<td>49.9</td>
<td>67.4</td>
<td>86.1</td>
<td>67.5</td>
<td>57.1</td>
<td>62.9</td>
</tr>
<tr>
<td>MMRet [69]</td>
<td>7B</td>
<td>56.0</td>
<td>57.4</td>
<td>69.9</td>
<td>83.6</td>
<td>68.0</td>
<td>59.1</td>
<td>64.1</td>
</tr>
<tr>
<td>mmE5 [8]</td>
<td>11B</td>
<td>67.6</td>
<td>62.7</td>
<td>71.0</td>
<td>89.7</td>
<td>72.4</td>
<td>66.6</td>
<td>69.8</td>
</tr>
<tr>
<td>LLaVE [33]</td>
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
<td>UniME [21]</td>
<td>7B</td>
<td>66.8</td>
<td>66.6</td>
<td>70.6</td>
<td>90.9</td>
<td>74.6</td>
<td>65.8</td>
<td>70.7</td>
</tr>
<tr>
<td>B3-7B [58]</td>
<td>7B</td>
<td>70.0</td>
<td>66.5</td>
<td>74.1</td>
<td>84.6</td>
<td>75.9</td>
<td>67.1</td>
<td>72.0</td>
</tr>
<tr>
<td>MoCa-7B [7]</td>
<td>7B</td>
<td>65.8</td>
<td>64.7</td>
<td>75.0</td>
<td>92.4</td>
<td>74.7</td>
<td>67.6</td>
<td>71.5</td>
</tr>
<tr>
<td>MuCo-7B</td>
<td>7B</td>
<td>68.3</td>
<td>71.9</td>
<td>73.7</td>
<td>90.9</td>
<td>77.3</td>
<td>69.1</td>
<td>73.6</td>
</tr>
</tbody>
</table>

**结果分析**：
*   零样本场景下：MuCo-7B以61.6%的得分超过了参数量更大的之前SOTA mmE5-11B（58.6%），提升了3.0个百分点，证明预训练的MuCo具备极强的零样本泛化能力。
*   微调场景<7B参数量：MuCo-2B以69.5%的得分超过之前SOTA B3-2B（68.1%），提升1.4个百分点，且OOD得分65.0%为该组最高，说明泛化性更强。
*   微调场景≥7B参数量：MuCo-7B以73.6%的得分超过之前SOTA B3-7B（72.0%），提升1.6个百分点，OOD得分69.1%同样为该组最高，验证了方法的通用性。
### 6.1.2. M-BEIR基准结果
下表是原文Table2的完整M-BEIR基准Recall结果：

<table>
<thead>
<tr>
<th>Models</th>
<th># Params</th>
<th>S → S</th>
<th>S → M</th>
<th>M → S</th>
<th>M → M</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td>UniIR [62]</td>
<td>0.4B</td>
<td>51.0</td>
<td>69.1</td>
<td>32.9</td>
<td>52.4</td>
<td>48.9</td>
</tr>
<tr>
<td>LamRA-Ret [43]</td>
<td>2B</td>
<td>47.6</td>
<td>66.2</td>
<td>41.0</td>
<td>61.0</td>
<td>50.0</td>
</tr>
<tr>
<td>MuCo-2B</td>
<td>2B</td>
<td>49.1</td>
<td>68.2</td>
<td>41.6</td>
<td>64.8</td>
<td>51.6</td>
</tr>
<tr>
<td>MM-Embed [38]</td>
<td>7B</td>
<td>50.9</td>
<td>76.9</td>
<td>40.0</td>
<td>60.9</td>
<td>52.7</td>
</tr>
<tr>
<td>LamRA-Ret [43]</td>
<td>7B</td>
<td>52.9</td>
<td>71.8</td>
<td>45.2</td>
<td>65.0</td>
<td>54.9</td>
</tr>
<tr>
<td>M3Task-UEM [56]</td>
<td>7B</td>
<td>54.0</td>
<td>74.9</td>
<td>41.9</td>
<td>55.7</td>
<td>53.9</td>
</tr>
<tr>
<td>MuCo-7B</td>
<td>7B</td>
<td>54.0</td>
<td>71.6</td>
<td>47.4</td>
<td>70.4</td>
<td>56.6</td>
</tr>
</tbody>
</table>

**结果分析**：
*   MuCo-2B以51.6%的得分超过同规模之前SOTA LamRA-Ret-2B（50.0%），提升1.6个百分点。
*   MuCo-7B以56.6%的得分超过同规模之前SOTA LamRA-Ret-7B（54.9%），提升1.7个百分点，尤其是复杂的多模→多模（M→M）任务得分达到70.4%，远超之前的SOTA，证明MuCo在复杂多模态检索场景下的优势。
### 6.1.3. 效率对比结果
下表是原文Table5的效率对比结果：

<table>
<thead>
<tr>
<th>#turns</th>
<th>#batch</th>
<th>#effective batch</th>
<th>PFLOPs</th>
<th>MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>1024</td>
<td>1024</td>
<td>17.5</td>
<td>57.1</td>
</tr>
<tr>
<td>1</td>
<td>2048</td>
<td>2048</td>
<td>35.1</td>
<td>57.3</td>
</tr>
<tr>
<td>1</td>
<td>4096</td>
<td>4096</td>
<td>70.2</td>
<td>57.4</td>
</tr>
<tr>
<td>1</td>
<td>7168</td>
<td>7168</td>
<td>122.7</td>
<td>57.5</td>
</tr>
<tr>
<td>1</td>
<td>8192</td>
<td>8192</td>
<td>140.4</td>
<td>57.8</td>
</tr>
<tr>
<td>2</td>
<td>1024</td>
<td>2048</td>
<td>17.6</td>
<td>57.4</td>
</tr>
<tr>
<td>4</td>
<td>1024</td>
<td>4096</td>
<td>17.7</td>
<td>57.7</td>
</tr>
<tr>
<td>7</td>
<td>1024</td>
<td>7168</td>
<td>18.0</td>
<td>58.2</td>
</tr>
</tbody>
</table>

**结果分析**：
传统单轮方法将有效batch从1024扩大到7168时，计算开销从17.5PFLOPs增长到122.7PFLOPs，增长了6倍，而性能仅提升0.4个百分点；而MuCo通过将轮数提升到7，在有效batch同样达到7168的情况下，计算开销仅增长到18.0PFLOPs，仅增长2.8%，同时性能提升了1.1个百分点，甚至超过了单轮方法batch扩大到8192的性能（58.2 vs 57.8），充分证明了MuCo的效率优势。
## 6.2. 消融实验分析
### 6.2.1. 预训练数据集的影响
下表是原文Table3的预训练数据集消融结果：

<table>
<thead>
<tr>
<th>Pre-training</th>
<th>Dataset</th>
<th>Samples</th>
<th>ZS MMEB</th>
<th>FT MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Single-turn</td>
<td>None</td>
<td>-</td>
<td>−</td>
<td>68.5</td>
</tr>
<tr>
<td>mmE5 [8]</td>
<td>0.6M</td>
<td>55.6</td>
<td>68.6</td>
</tr>
<tr>
<td>MegaPairs [69]</td>
<td>26M</td>
<td>41.5</td>
<td>68.7</td>
</tr>
<tr>
<td rowspan="4">Multi-turn MuCo</td>
<td>mmE5 [8]</td>
<td>0.6M</td>
<td>57.0</td>
<td>69.0</td>
</tr>
<tr>
<td>M3T (20%)</td>
<td>1M</td>
<td>57.1</td>
<td>69.0</td>
</tr>
<tr>
<td>M3T (60%)</td>
<td>3M</td>
<td>57.7</td>
<td>69.2</td>
</tr>
<tr>
<td>M3T</td>
<td>5M</td>
<td>58.2</td>
<td>69.5</td>
</tr>
</tbody>
</table>

**结果分析**：
1.  多轮预训练优于单轮预训练：同样使用mmE5数据集，多轮预训练的微调性能比单轮高0.4个百分点，证明多轮范式的有效性。
2.  M3T数据集质量高：随着M3T规模从1M增长到5M，性能单调上升，仅用5M样本的效果就超过了26M样本的单轮MegaPairs，证明M3T的高质量。
3.  微调策略本身极强：即使不进行任何预训练，仅使用MuCo的微调策略，性能也达到68.5%，超过了之前的SOTA B3-2B（68.1%），证明微调策略的有效性。
### 6.2.2. 复合监督的影响
下表是原文Table6的复合监督消融结果：

<table>
<thead>
<tr>
<th>Setup</th>
<th>ZS MMEB</th>
<th>FT MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o compounded supervision</td>
<td>57.3</td>
<td>68.4</td>
</tr>
<tr>
<td>w/ compounded supervision</td>
<td>58.2</td>
<td>69.5</td>
</tr>
</tbody>
</table>

**结果分析**：
当禁用多轮的上下文注意力（每轮只能看到自己的内容，看不到之前轮的内容），性能下降了1.1个百分点，证明多轮的复合监督信号确实能有效提升表示质量。
### 6.2.3. Logit掩码的影响
下表是原文Table7的logit掩码消融结果：

<table>
<thead>
<tr>
<th>Pretraining masking</th>
<th>Fine-tuning masking</th>
<th>ZS MMEB</th>
<th>FT MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td>✗</td>
<td>✗</td>
<td>57.7</td>
<td>31.1</td>
</tr>
<tr>
<td>✗</td>
<td>✓</td>
<td>57.7</td>
<td>69.2</td>
</tr>
<tr>
<td>✓</td>
<td>✗</td>
<td>58.2</td>
<td>30.9</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>58.2</td>
<td>69.5</td>
</tr>
</tbody>
</table>

**结果分析**：
微调阶段如果不使用logit掩码，性能会直接崩溃到30%左右，因为模型会把语义高度相关的增强样本当作负样本，导致学习混乱；预训练阶段不使用掩码也会有0.5个百分点的性能下降，证明logit掩码是MuCo的必要组件。

---
# 7. 总结与思考
## 7.1. 结论总结
本文提出的多轮对比学习（MuCo）框架，首次将MLLM的对话能力与多模态对比学习结合，从根本上解决了传统单轮对比学习的效率瓶颈和表示质量瓶颈：
1.  效率层面：在计算开销仅增加≤3%的前提下，有效batch规模可扩大7倍，训练成本远低于传统扩大batch的方式。
2.  性能层面：搭配高质量的M3T多轮数据集，在MMEB和M-BEIR两个权威基准上均取得了新的SOTA性能，同时具备更强的分布外泛化能力。
3.  兼容层面：提出的单轮数据集适配策略，让MuCo可以无缝适配现有的所有多模态嵌入评估体系，不需要修改下游任务的评估流程。
    MuCo重新定义了多模态嵌入训练的效率-性能权衡，为后续的高效多模态表示学习提供了全新的范式。
## 7.2. 局限性与未来工作
论文当前的工作仍然存在一定的局限性，未来可以从以下方向进一步拓展：
1.  **模态拓展**：当前M3T数据集仅覆盖图像+文本到文本的任务结构，未来可以拓展到文本到图像、图文到图文、视频、音频等更多模态的多轮对比学习。
2.  **动态轮数**：当前每幅图像固定使用7轮查询，未来可以根据图像的复杂度动态调整轮数，进一步提升效率和性能。
3.  **真实数据利用**：当前M3T是合成数据集，未来可以探索利用真实的视觉对话数据（如VisDial）进行预训练，进一步提升模型的真实场景泛化能力。
4.  **单模态迁移**：该多轮对比学习的思路可以迁移到单模态文本嵌入领域，将同一段文本对应的多个查询做成多轮序列，提升文本嵌入的训练效率和质量。
## 7.3. 个人启发与批判
这篇论文的创新思路非常值得借鉴：它没有对MLLM的架构进行任何复杂的修改，仅仅利用了MLLM本身就具备的多轮对话能力，通过重新组织训练输入的结构，就同时实现了效率和性能的大幅提升，这种“挖潜现有能力，而非新增复杂模块”的思路在大模型时代极具参考价值。
同时，论文对效率瓶颈的切入点非常精准：之前的研究都在思考如何降低图像编码的开销、如何优化负采样来降低对大batch的依赖，而MuCo跳出了这个思维定式，既然图像编码贵，那就减少图像编码的次数，给每幅图像增加低开销的文本轮次来提升有效batch，这种换维度解决问题的思路非常巧妙。
潜在的可改进点：
1.  多轮查询的顺序可能会对性能产生影响，论文中是随机打乱顺序的，未来可以探索最优的查询顺序，比如从简单到复杂的查询顺序是否能进一步提升表示质量。
2.  当前微调阶段的掩码重建是离线进行的，未来可以探索在线的重建策略，进一步提升训练效率。
3.  可以探索将多轮对比学习与检索增强、负采样优化等现有技术结合，进一步提升多模态嵌入的性能。