# 1. 论文基本信息
## 1.1. 标题
**TRACE: Task-Adaptive Reasoning and Representation Learning for Universal Multimodal Retrieval**
核心主题：提出一种面向通用多模态检索的任务自适应推理与表示学习框架，解决现有静态编码器方法无法处理复杂用户意图的痛点。
## 1.2. 作者
本文作者均来自中国科学院自动化研究所、中国科学院大学：
* 第一作者：郝相兆、王世杰、杨天予
* 其他作者：王天越、郭海云、王金桥（通讯作者）
## 1.3. 发表状态
当前为预印本状态，2026年3月3日发布于arXiv，尚未正式发表在期刊/会议。
## 1.4. 发表年份
2026年
## 1.5. 摘要
通用多模态检索要求统一嵌入模型能够理解从简单关键词到复杂组合指令的各类用户意图。当前多模态大模型（MLLM）虽然具备强大推理能力，但现有适配方法仅将其作为静态编码器使用，未能充分利用其生成潜力，这种仅编码器范式难以处理需要逻辑推理的复杂意图。
为此本文提出TRACE（任务自适应推理与压缩嵌入框架），将生成式推理与判别式表示学习统一：首先生成结构化链式思维（CoT）对查询做显式推理，再通过专用标记将推理轨迹压缩为紧凑嵌入。为训练该框架，本文构建了大规模数据集M-BEIR-CoT，具备难度感知路由策略。
在M-BEIR基准上的实验证明TRACE达到新的最先进水平，其具备隐式路由能力：对复杂查询自动激活推理，对简单查询跳过推理，实现检索精度与推理吞吐量的最优平衡。此外通过内化演绎过程，TRACE在未见域和新约束场景下具备出色的零样本迁移能力。
## 1.6. 原文链接
* 预印本主页：https://arxiv.org/abs/2603.02929
* PDF链接：https://arxiv.org/pdf/2603.02929v2
* 发布状态：arXiv预印本

  ---

# 2. 整体概括
## 2.1. 研究背景与动机
### 2.1.1. 核心问题
通用多模态检索的目标是实现跨模态的统一搜索，支持纯文本、图文混合等多种查询形式，检索目标也可以是文本、图像或多模态内容。当前主流方法将多模态大模型（MLLM）作为静态编码器使用，仅通过单次前向传播就将输入压缩为固定维度嵌入，这种范式在处理需要多步逻辑的复杂用户意图（比如“找和这张图款式相同但颜色是红色的裙子”这类组合修改指令）时存在严重的认知瓶颈，无法充分发挥MLLM的生成推理能力。
### 2.1.2. 领域重要性与现有空白
随着多模态内容的爆发，通用多模态检索是搜索引擎、智能助手等应用的核心技术，现有方法只能处理简单匹配任务，无法满足用户日益复杂的搜索需求。现有研究存在两个核心空白：
1.  没有将MLLM的生成推理能力无缝整合到检索的表示学习过程中，现有带推理的检索方法都是多阶段 pipeline：先用单独模型做查询扩展/改写，再用另一个编码器做嵌入，无法实现视觉感知与逻辑推理的端到端优化。
2.  缺乏适配推理型检索的大规模高质量数据集，现有检索数据集没有标注推理轨迹，无法训练同时具备推理和表示能力的模型。
### 2.1.3. 创新切入点
本文提出“先推理再编码”的新范式：对查询先做显式的生成式推理分解意图，再将推理轨迹压缩为检索嵌入，同时设计隐式自适应路由机制，自动根据查询难度选择是否激活推理，平衡效率与精度。
## 2.2. 核心贡献与主要发现
本文的核心贡献可归纳为三点：
1.  **新框架**：提出TRACE通用检索框架，首次将任务自适应推理显式整合到判别式嵌入过程中，通过端到端训练实现精度与推理吞吐量的平衡，无需显式的架构分支或手动门控网络。
2.  **新数据集**：构建了M-BEIR-CoT大规模高质量数据集，通过难度感知路由、任务特定CoT生成、双重过滤流程，解决了推理型检索的数据稀缺问题，总规模包含51.8万简单样本和57.5万推理样本。
3.  **新发现**：在M-BEIR基准上达到新的最先进水平，同时发现检索推理的基本不对称性：**查询侧的推理可以显著提升语义对齐效果，但强制对候选侧做推理会让模型过拟合生成的文本模式，导致性能灾难性下降**。

    ---

# 3. 预备知识与相关工作
## 3.1. 基础概念
为便于初学者理解，首先解释本文涉及的核心专业概念：
### 3.1.1. 通用多模态检索
指支持任意模态的查询（文本、图像、图文混合、多轮对话等）和任意模态的检索目标（文本、图像、图文对）的统一检索系统，无需为不同模态组合单独训练模型。
### 3.1.2. 多模态大模型（MLLM, Multimodal Large Language Model）
以大语言模型为基座，接入视觉编码器等多模态输入接口，经过预训练后具备跨模态理解、推理和生成能力的大模型，比如Qwen2.5-VL、GPT-4o等。
### 3.1.3. 链式思维（CoT, Chain-of-Thought）
最初是大语言模型领域的技术，通过让模型输出中间推理步骤，显著提升其解决数学题、逻辑推理等复杂任务的能力，后续被扩展到多模态领域用于减少幻觉、提升可解释性。
### 3.1.4. 对比学习与InfoNCE损失
对比学习是表示学习的常用范式，核心目标是让正样本对（语义相似的样本）的嵌入相似度尽可能高，负样本对（语义不相似的样本）的嵌入相似度尽可能低。InfoNCE（Noise Contrastive Estimation）是对比学习最常用的损失函数，公式如下：
$$
\mathcal{L}_{\text{InfoNCE}} = - \frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(\mathbf{e}_i, \mathbf{e}_i^+)/\tau)}{\sum_{j=1}^B \exp(\text{sim}(\mathbf{e}_i, \mathbf{e}_j)/\tau)}
$$
其中$\text{sim}(\cdot)$是相似度函数（通常为余弦相似度），$\mathbf{e}_i$是锚点样本嵌入，$\mathbf{e}_i^+$是正样本嵌入，$\mathbf{e}_j$是batch内的负样本嵌入，$\tau$是温度系数，用于控制相似度分布的尖锐程度，$B$是batch大小。
### 3.1.5. 低秩适配（LoRA, Low-Rank Adaptation）
大模型微调的参数高效方法，通过在Transformer的注意力层插入小的低秩矩阵，仅训练这些低秩矩阵的参数，冻结原有大模型的参数，大幅降低微调的显存开销和计算量，同时性能接近全参数微调。
## 3.2. 前人工作
### 3.2.1. 通用多模态检索的演进
1.  **双编码器阶段**：以CLIP、ALIGN为代表，通过大规模对比学习对齐视觉和文本表示，双编码器分别处理查询和候选，仅在最后做相似度匹配。这类方法只能处理简单的图文匹配，无法捕捉细粒度的组合逻辑，比如需要修改对象属性、保留背景的组合检索任务。
2.  **MLLM静态编码器阶段**：以UniIR、E5-V、LamRA为代表，将MLLM适配为通用检索器，通过添加特定prompt提取最后层的隐状态作为嵌入。这类方法虽然利用了MLLM的广泛知识，但仅将MLLM作为静态编码器，跳过了其固有生成推理能力，处理复杂逻辑查询时存在认知瓶颈。
### 3.2.2. 链式思维在判别任务的应用
CoT在生成任务（比如VQA、图像描述）中已经被广泛验证有效，但在检索这类判别任务中的应用仍然非常有限。现有方法大多采用多阶段pipeline：先用生成模型做外部查询扩展或改写，再将增强后的文本输入单独的编码器，这种分离的架构无法实现视觉感知和逻辑推理的端到端优化，也无法自适应平衡效率与推理深度。
## 3.3. 技术演进脉络
通用多模态检索的技术演进路径可以归纳为：
`双编码器（仅匹配无推理）` → `MLLM静态编码器（有推理能力但未利用）` → `推理增强的检索（本文提出的先推理再编码范式）`
本文的工作是首次将生成式推理与检索表示学习端到端整合，填补了现有技术的空白。
## 3.4. 差异化分析
本文与现有方法的核心区别在于：
1.  与静态编码器方法相比：TRACE主动利用MLLM的生成能力做显式推理，解决复杂意图的语义鸿沟问题，而不是直接映射到嵌入空间。
2.  与多阶段推理检索方法相比：TRACE的推理和编码是在同一个模型里端到端完成的，无需外部模型做查询扩展，还具备自适应路由能力，自动平衡效率与精度。
3.  与其他CoT方法相比：TRACE首次发现检索任务的推理不对称性，明确了推理仅适用于查询侧，候选侧推理会导致性能崩溃，为后续相关研究提供了重要指导。

    ---

# 4. 方法论
TRACE框架的核心设计思路如下图（原文Figure 1）所示：

![Fig. 1: The TRACE Framework. TRACE learns a query-dependent inference strategy. (a) For simple queries, it implicitly bypasses the reasoning stage and directly extracts features to maintain high efficiency. (b) For complex queries, it automatically activates the task-adaptive reasoning process. The model generates an explicit reasoning trace \[44\] to resolve semantic ambiguities before compressing this context into the final representation. (c) Performance comparison on the M-BEIR benchmark \[43\] demonstrates the effectiveness of TRACE, particularly on reasoning-intensive tasks.](images/1.jpg)
*该图像是TRACE框架示意图，展示了模型针对简单和复杂查询的推理策略。对于简单查询，直接提取特征以提高效率；对于复杂查询，则激活任务自适应推理，生成推理轨迹并压缩为最终表示。图(c)展示了TRACE在M-BEIR基准上的性能比较，突出其在推理密集任务中的有效性。*

## 4.1. 方法原理
TRACE的核心直觉是：复杂用户意图需要中间推理步骤才能准确理解，直接将复杂查询压缩为嵌入会丢失关键逻辑信息。因此采用“先推理再编码”的范式：
1.  对简单查询（比如关键词搜索），直接生成嵌入标记，保证效率；
2.  对复杂查询（比如组合修改指令），先生成显式的CoT推理轨迹分解意图，再将推理上下文压缩为检索嵌入；
3.  通过混合数据集的训练，让模型自动学习隐式路由能力，无需手动设计门控网络。
## 4.2. 核心方法详解
### 4.2.1. 问题定义
本文定义的推理感知通用检索目标是学习一个检索函数$f_\theta$，支持任意多模态查询$Q$（文本$T$、图像$I$或混合序列），从候选集$\Omega$中检索目标$C$。与传统检索直接将查询映射为静态向量$Q \to \mathbf{v}_q$不同，TRACE将检索建模为条件生成-压缩过程：
$$
S = [ \mathcal{R} ; <| \mathtt{emb} |> ], \quad \mathrm{where} \ \mathcal{R} = \left\{ \begin{array} { l l } { \emptyset } & { \mathrm{if} \ z = 0 } \\ { \{ r_1 , \dots , r_k \} } & { \mathrm{if} \ z = 1 } \end{array} \right.
$$
其中：
* $S$是模型生成的中间序列，以特殊嵌入标记$<|emb|>$结尾；
* $\mathcal{R}$是生成的推理轨迹序列，$z \in \{0,1\}$是隐式复杂度变量，决定是否激活推理；
* 最终查询表示$\mathbf{e}_q$从预测$<|emb|>$标记的隐状态中提取，目标是最大化与真值目标$\mathbf{e}_c$的相似度$\text{sim}(\mathbf{e}_q, \mathbf{e}_c)$。
### 4.2.2. M-BEIR-CoT数据集构建
为解决推理型检索的数据稀缺问题，本文基于M-BEIR基准构建了M-BEIR-CoT数据集，构建流程分为三个阶段，如下图（原文Figure 2）所示：

![Fig. 2: The construction pipeline of the M-BEIR-CoT dataset. The process operates in three phases: (1) Query Complexity Assessment: An advanced MLLM assesses query difficulty, routing simple queries to a direct path (generating only $< | \\mathsf { e m b } | >$ ) and complex queries to a reasoning path (generating $\\mathtt { C o T } + < | \\mathtt { e m b } | >$ . (2) Task-Specific CoT Generation: We design specialized prompts for diverse tasks (e.g., captioning, text edit, VQA) to generate structured reasoning traces enclosed in <reasoning> tags. (3) Dual Filtering & Curation: To ensure data quality, we apply a coarse-to-fine strategy. We first use rule-based filtering to verify formats and lengths, followed by model-based filtering to ensure semantic consistency between the generated text and ground-truth targets.](images/2.jpg)
*该图像是M-BEIR-CoT数据集的构建流程示意图，展示了查询复杂性评估、任务特定的链式思维生成与双重过滤和策划过程。图中通过示例展示了不同任务的处理方式，包括图像描述、文本编辑和视觉问答（VQA）。*

1.  **阶段1：查询复杂度评估与自适应路由**
    用GPT-4o作为查询复杂度评估器，将查询分为两类：简单模式匹配任务走直接编码流（$z=0$，仅生成$<|emb|>$），包含约束或逻辑的复杂查询走推理增强流（$z=1$，需要生成CoT），让模型学习“什么时候需要推理”。
2.  **阶段2：任务特定CoT生成**
    对路由到推理流的查询，针对不同检索子任务设计专用prompt模板：
    * 视觉推理任务：先描述细粒度细节再总结；
    * 指令遵循任务（比如组合图像检索）：推理轨迹显式定义源状态、目标操作、不变上下文；
    * 逻辑演绎任务：执行多跳推理；
      所有输出都用$<reasoning>$和$<answer>$标签包裹，便于结构化解析。
3.  **阶段3：双重过滤与筛选**
    为减少大模型生成的幻觉，采用粗到细的过滤协议：
    * 规则过滤：验证格式、长度，移除无效标签、过短/过长的生成、重复内容、拒绝类回复；
    * 模型过滤：用强验证模型计算生成内容与真值目标的语义对齐度，仅保留置信度高于阈值的样本。
      最终得到575442个高质量推理样本，与518311个简单样本合并为M-BEIR-CoT数据集，训练时移除辅助标签，仅保留自然语言内容。
### 4.2.3. TRACE架构与自适应机制
TRACE的整体架构如下图（原文Figure 3）所示，基于Qwen2.5-VL基座构建，包含冻结的视觉编码器、可训练投影层、大语言模型（LLM）主干：

![Fig. 3: Illustration of the TRACE architecture. The model processes a multimodal query through a frozen vision encoder and a trainable projector. The LLM acts as a unified reasoner and encoder. It first generates a Chain-of-Thought (CoT) \[44\] to interpret the intent and then compresses the semantics into a learnable $< | \\mathtt { e m b } | >$ token. The final query feature is extracted from the hidden state immediately preceding $< | \\mathtt { e m b } | >$ . During training, the model is optimized jointly using Cross-Entropy (CE) loss for reasoning generation and InfoNCE loss \[32\] for embedding alignment.](images/3.jpg)
*该图像是TRACE架构的示意图，展示了模型如何处理多模态查询。模型通过冻结的视觉编码器和可训练的投影器进行查询，利用大型语言模型生成Chain-of-Thought (CoT) 来解析意图，并将语义压缩为可学习的 $< | ext{emb} | >$ 标记。该过程通过交叉熵损失和InfoNCE损失进行联合优化。*

#### 自适应推理激活机制
由于M-BEIR-CoT数据集混合了直接编码和推理增强的样本，模型会隐式学习评估查询复杂度，推理时通过自回归解码动态选择最优路径，无需显式分支或手动门控：
$$
\mathrm{Output}(Q) = \left\{ \begin{array} { l l } { { [ < | \operatorname { e m b } | > ] } } & { { \mathrm { if } < | \operatorname { e m b } | > = \arg \operatorname* { m a x } _ { y \in \mathcal { V } } P ( y \mid Q ) } } \\ { { [ \mathrm { CoT \ Tokens } , < | \operatorname { e m b } | > ] } } & { { \mathrm { if } \ \arg \operatorname* { m a x } _ { y \in \mathcal { V } } P ( y \mid Q ) \in \mathcal { V } _ { \mathrm { t e x t } } } } \end{array} \right.
$$
其中：
* $\mathcal{V}$是完整词表，$\mathcal{V}_\text{text}$是标准文本词表，不包含$<|emb|>$特殊标记；
* 对简单查询，模型第一步解码的最高概率 token 是$<|emb|>$，直接输出嵌入，跳过推理；
* 对复杂查询，模型第一步解码的最高概率 token 是文本词，会先生成CoT推理轨迹，最后输出$<|emb|>$。
#### 嵌入提取策略
由于LLM采用因果注意力机制，第$t$步的隐状态$\mathbf{h}_t$被优化用于预测第$t+1$步的token$y_{t+1}$，因此$<|emb|>$前一个token的隐状态需要预测这个序列结束标记，聚合了之前所有上下文（原始查询+生成的CoT）的信息，是最优的语义瓶颈。因此本文提取该前导token的隐状态作为最终检索嵌入$\mathbf{e}_q \in \mathbb{R}^d$。
### 4.2.4. 统一单阶段训练
采用混合损失函数端到端同时优化推理能力和表示能力：
1.  **生成式推理损失$\mathcal{L}_\text{gen}$**
    采用标准交叉熵损失监督推理轨迹token的生成，强制模型内化意图分解的逻辑：
$$
\mathcal{L}_{\text{gen}} = - \sum_{t=1}^{|\mathcal{R}|} \log P(y_t \mid y_{<t}, Q)
$$
其中$y$是真值推理token，$|\mathcal{R}|$是推理轨迹的长度。
2.  **判别式对比损失$\mathcal{L}_\text{ret}$**
    采用InfoNCE损失对最终$<|emb|>$的嵌入空间做结构化约束：
$$
\mathcal{L}_{\text{ret}} = - \frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(\mathbf{e}_{q_i}, \mathbf{e}_{c_i})/\tau)}{\sum_{j=1}^B \exp(\text{sim}(\mathbf{e}_{q_i}, \mathbf{e}_{c_j})/\tau)}
$$
其中$B$是batch大小，$\mathbf{e}_{q_i}$是第$i$个查询的嵌入，$\mathbf{e}_{c_i}$是对应真值候选的嵌入，$\tau$是温度系数。
3.  **总损失**
    总损失为两个损失的加权和：
$$
\mathcal{L} = \lambda_{\text{gen}} \mathcal{L}_{\text{gen}} + \lambda_{\text{ret}} \mathcal{L}_{\text{ret}}
$$
其中$\lambda_{\text{gen}}$和$\lambda_{\text{ret}}$是损失权重，本文实验中取$\lambda_{\text{gen}}=1.0, \lambda_{\text{ret}}=1.0$。

---

# 5. 实验设置
## 5.1. 数据集
### 5.1.1. 训练数据集
训练采用本文构建的M-BEIR-CoT数据集，包含：
* 简单子集：51.8万样本，覆盖新闻、日常场景、时尚、维基百科等域的简单匹配任务；
* 推理子集：57.5万高质量样本，覆盖组合检索、视觉问答、实体识别等需要推理的任务。
### 5.1.2. 域内评测数据集
域内评测采用标准M-BEIR测试集，覆盖8类检索任务、10个数据集，包含文本到图像、文本到文本、图像到文本、图像到图像、图文混合查询等各类模态组合。
### 5.1.3. 零样本评测数据集
零样本评测采用13个完全未出现在训练集的数据集，包括ShareGPT4V、Urban-1K、CIRCO、Visual Dialog等，覆盖细粒度识别、组合图像检索、多轮对话等 diverse 场景，严格验证模型的泛化能力。
## 5.2. 评估指标
本文采用三类评估指标，分别解释如下：
### 5.2.1. Recall@K（召回率@K）
1.  **概念定义**：衡量检索系统返回的前K个结果中，包含至少一个正样本的查询占总查询数的比例，数值越高说明检索效果越好，是检索任务最常用的指标。
2.  **数学公式**：
    $$
\text{Recall@K} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}\left( \text{rank}(c_i^+) \leq K \right)
$$
3.  **符号解释**：
    * $N$是总查询数；
    * $\mathbb{1}(\cdot)$是指示函数，条件满足时取值为1，否则为0；
    * $\text{rank}(c_i^+)$是第$i$个查询对应的正样本在检索结果中的排名，排名越小越靠前。
### 5.2.2. MAP@K（平均精度均值@K）
1.  **概念定义**：衡量检索系统在多个查询上的平均精度，既考虑正样本的召回率，也考虑正样本的排序位置，数值越高效果越好，适用于多正样本的检索场景。
2.  **数学公式**：
    $$
\text{MAP@K} = \frac{1}{N} \sum_{i=1}^N \frac{1}{m_i} \sum_{k=1}^K P(k) \cdot \mathbb{1}(k \text{ 位置是正样本})
$$
3.  **符号解释**：
    * $N$是总查询数；
    * $m_i$是第$i$个查询的正样本总数；
    * `P(k)`是前$k$个检索结果的精度（正样本占比）。
### 5.2.3. Accuracy（准确率）
1.  **概念定义**：图文匹配任务中，预测正确的样本占总样本的比例，数值越高效果越好。
2.  **数学公式**：
    $$
\text{Accuracy} = \frac{\text{预测正确的样本数}}{\text{总样本数}}
$$
## 5.3. 对比基线
本文选择两类代表性基线做对比：
1.  **通用视觉语言模型/双编码器**：包括CLIP-L、SigLIP、BLIP、BLIP2、Qwen2.5-VL-7B、Long-CLIP-L、E5-V、MagicLens-L、EVA-CLIP-8B/18B等，这类基线代表基础的视觉语义对齐能力。
2.  **专用通用检索器**：包括UniIR-BLIPFF、UniIR-CLIPSF、LamRA-Ret等，这类基线是当前通用多模态检索的最先进方法，用于验证TRACE在统一检索场景下的优势。

    ---

# 6. 实验结果与分析
## 6.1. 域内核心结果分析
在M-BEIR测试集上的对比结果如下（原文Table 1）：

<table>
<thead>
<tr>
<th rowspan="3">Methods</th>
<th colspan="3">$q^t \to c^i$</th>
<th colspan="2">$q^t \to c^t$</th>
<th colspan="2">$q^t \to (c^i, c^t)$</th>
<th colspan="3">$q^i \to c^t$</th>
<th>$q^i \to c^i$</th>
<th>$(q^i, q^t) \to c^i$</th>
<th>$(q^i, q^t) \to c^t$</th>
<th>$(q^i,q^t) \to (c^i,c^t)$</th>
<th rowspan="3">Avg.</th>
</tr>
<tr>
<th>VN</th>
<th>COCO</th>
<th>F200K</th>
<th>WebQA</th>
<th>EDIS</th>
<th>WebQA</th>
<th>VN</th>
<th>COCO</th>
<th>F200K</th>
<th>NIGHTS</th>
<th>OVEN</th>
<th>InfoS</th>
<th>FIQ</th>
<th>CIRR</th>
<th>OVEN</th>
<th>InfoS</th>
</tr>
<tr>
<th>R@5</th>
<th>R@5</th>
<th>R@10</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
<th>R@10</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
<th>R@10</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP-L [35]</td>
<td>43.3</td>
<td>61.1</td>
<td>6.6</td>
<td>36.2</td>
<td>43.3</td>
<td>45.1</td>
<td>41.3</td>
<td>79.0</td>
<td>7.7</td>
<td>26.1</td>
<td>24.2</td>
<td>20.5</td>
<td>7.0</td>
<td>13.2</td>
<td>38.8</td>
<td>-</td>
<td>26.4</td>
<td>32.5</td>
</tr>
<tr>
<td>SigLIP [50]</td>
<td>30.1</td>
<td>75.7</td>
<td>36.5</td>
<td>39.8</td>
<td>27.0</td>
<td>43.5</td>
<td>30.8</td>
<td>88.2</td>
<td>34.2</td>
<td>28.9</td>
<td>-</td>
<td>29.7</td>
<td>25.1</td>
<td>14.4</td>
<td>22.7</td>
<td>41.7</td>
<td>27.4</td>
<td>37.2</td>
</tr>
<tr>
<td>BLIP[21]</td>
<td>16.4</td>
<td>74.4</td>
<td>15.9</td>
<td>44.9</td>
<td>26.8</td>
<td>20.3</td>
<td>17.2</td>
<td>83.2</td>
<td>19.9</td>
<td>27.4</td>
<td>-</td>
<td>16.1</td>
<td>10.2</td>
<td>2.3</td>
<td>10.6</td>
<td>27.4</td>
<td>16.6</td>
<td>26.8</td>
</tr>
<tr>
<td>BLI2 [20]</td>
<td>16.7</td>
<td>63.8</td>
<td>14.0</td>
<td>38.6</td>
<td>26.9</td>
<td>24.5</td>
<td>15.0</td>
<td>80.0</td>
<td>14.2</td>
<td>25.4</td>
<td>-</td>
<td>12.2</td>
<td>5.5</td>
<td>4.4</td>
<td>11.8</td>
<td>27.3</td>
<td>15.8</td>
<td>24.8</td>
</tr>
<tr>
<td>Qwen2.5-VL-7B [2]</td>
<td>9.3</td>
<td>55.1</td>
<td>5.0</td>
<td>42.0</td>
<td>26.2</td>
<td>9.4</td>
<td>5.4</td>
<td>46.6</td>
<td>4.0</td>
<td>21.3</td>
<td>-</td>
<td>21.4</td>
<td>22.5</td>
<td>4.3</td>
<td>16.3</td>
<td>43.6</td>
<td>36.2</td>
<td>23.0</td>
</tr>
<tr>
<td>UniIR-BLIPFF [43]</td>
<td>23.4</td>
<td>79.7</td>
<td>26.1</td>
<td>80.0</td>
<td>50.9</td>
<td>79.8</td>
<td>22.8</td>
<td>89.9</td>
<td>28.9</td>
<td>33.0</td>
<td>-</td>
<td>41.0</td>
<td>22.4</td>
<td>29.2</td>
<td>52.2</td>
<td>55.8</td>
<td>33.0</td>
<td>46.8</td>
</tr>
<tr>
<td>UniIR-CLIPSF [43]</td>
<td>42.6</td>
<td>81.1</td>
<td>18.0</td>
<td>84.7</td>
<td>59.4</td>
<td>78.7</td>
<td>43.1</td>
<td>92.3</td>
<td>18.3</td>
<td>32.0</td>
<td>45.5</td>
<td>27.9</td>
<td>-</td>
<td>24.4</td>
<td>44.6</td>
<td>67.6</td>
<td>48.9</td>
<td>50.6</td>
</tr>
<tr>
<td>LamRA-Ret [28]</td>
<td>41.6</td>
<td>81.5</td>
<td>28.7</td>
<td>86.0</td>
<td>62.6</td>
<td>81.2</td>
<td>39.6</td>
<td>90.6</td>
<td>30.4</td>
<td>32.1</td>
<td>-</td>
<td>54.1</td>
<td>52.1</td>
<td>33.2</td>
<td>53.1</td>
<td>76.2</td>
<td>63.3</td>
<td>56.6</td>
</tr>
<tr>
<td colspan="19">Our Method</td>
</tr>
<tr>
<td>TRACE</td>
<td>42.1</td>
<td>82.3</td>
<td>30.5</td>
<td>87.8</td>
<td>64.1</td>
<td>82.5</td>
<td>41.2</td>
<td>91.3</td>
<td>33.2</td>
<td>33.6</td>
<td>-</td>
<td>57.4</td>
<td>55.8</td>
<td>36.4</td>
<td>57.3</td>
<td>78.5</td>
<td>67.1</td>
<td>58.8</td>
</tr>
<tr>
<td>Improvement</td>
<td>+0.5</td>
<td>+0.8</td>
<td>+1.8</td>
<td>+1.8</td>
<td>+1.5</td>
<td>+1.3</td>
<td>+1.6</td>
<td>+0.7</td>
<td>+2.8</td>
<td>+1.5</td>
<td>-</td>
<td>+3.3</td>
<td>+3.7</td>
<td>+3.2</td>
<td>+4.2</td>
<td>+2.3</td>
<td>+3.8</td>
<td>+2.2</td>
</tr>
</tbody>
</table>

从结果可以看出：
1.  TRACE在M-BEIR基准上平均得分达到58.8%，比之前的SOTA LamRA-Ret高出2.2%，达到新的最先进水平；
2.  性能提升在推理密集的任务上尤为显著：CIRR（组合检索）R@5提升4.2%，FashionIQ R@10提升3.2%，InfoSeek R@5提升3.8%，说明显式推理有效解决了复杂意图的语义鸿沟问题；
3.  相比基座模型Qwen2.5-VL-7B，TRACE将平均得分从23.0%提升到58.8%，证明了任务自适应推理范式的有效性。
## 6.2. 效率与自适应分析
TRACE的效率与精度平衡结果如下（原文Table 2）：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">MSCOCO (Simple)</th>
<th colspan="2">CIRR (Complex)</th>
</tr>
<tr>
<th>QPS ↑</th>
<th>R@5 ↑</th>
<th>QPS ↑</th>
<th>R@5 ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>Direct Embedding</td>
<td>15.68</td>
<td>87.40</td>
<td>12.15</td>
<td>53.06</td>
</tr>
<tr>
<td>Always CoT (Forced)</td>
<td>4.45</td>
<td>63.90</td>
<td>3.42</td>
<td>54.73</td>
</tr>
<tr>
<td>TRACE (Ours)</td>
<td>8.25</td>
<td>89.10</td>
<td>6.48</td>
<td>57.03</td>
</tr>
</tbody>
</table>

其中QPS（Queries Per Second）是单张H20 GPU上的在线编码吞吐量，数值越高效率越高：
1.  对简单任务MSCOCO，强制生成CoT会导致性能从87.40%暴跌到63.90%，因为简单查询会被过度思考引入幻觉约束，而TRACE自动跳过推理，性能提升到89.10%，吞吐量是强制CoT的近2倍；
2.  对复杂任务CIRR，TRACE自动激活推理，R@5比直接嵌入高3.97%，吞吐量比强制CoT高近1倍；
3.  路由准确率验证：简单查询96%的概率第一步直接输出$<|emb|>$，复杂查询62%的概率自动生成CoT，证明自适应机制准确捕捉了查询的认知需求。
## 6.3. 零样本泛化结果
13个未见数据集的零样本对比结果如下（原文Table 3）：

<table>
<thead>
<tr>
<th rowspan="3">Methods</th>
<th colspan="3">$q^t \to c^i$</th>
<th colspan="3">$q^i \to c^t$</th>
<th colspan="2">$(q^i,q^t) \to c^i$</th>
<th>$q^\text{dialog} \to c^i$</th>
<th colspan="2">$(q^i,q^t) \to c^i$</th>
<th colspan="2">ITM</th>
</tr>
<tr>
<th>Share4V</th>
<th>Urban*</th>
<th>Flickr</th>
<th>Share4V</th>
<th>Urban*</th>
<th>Flickr</th>
<th>CIRCO*</th>
<th>GeneCIS*</th>
<th>VisD*</th>
<th>VIST</th>
<th>MT-FIQ*</th>
<th>CC-Neg</th>
<th>Sugar-Crepe*</th>
</tr>
<tr>
<th>R@1</th>
<th>R@1</th>
<th>R@1</th>
<th>R@1</th>
<th>R@1</th>
<th>R@1</th>
<th>MAP@5</th>
<th>MAP@5</th>
<th>R@1</th>
<th>R@1</th>
<th>R@5</th>
<th>Acc.</th>
<th>Acc.</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP-L [35]</td>
<td>84.0</td>
<td>52.8</td>
<td>67.3</td>
<td>81.8</td>
<td>68.7</td>
<td>87.2</td>
<td>4.0</td>
<td>13.3</td>
<td>23.7</td>
<td>0.6</td>
<td>17.7</td>
<td>66.7</td>
<td>73.0</td>
</tr>
<tr>
<td>Long-CLIP-L [51]</td>
<td>95.6</td>
<td>86.1</td>
<td>76.1</td>
<td>95.8</td>
<td>82.7</td>
<td>89.3</td>
<td>5.7</td>
<td>16.3</td>
<td>37.9</td>
<td>1.1</td>
<td>18.5</td>
<td>76.3</td>
<td>80.9</td>
</tr>
<tr>
<td>UniIR-CLIP [43]</td>
<td>85.8</td>
<td>75.0</td>
<td>78.7</td>
<td>84.1</td>
<td>78.4</td>
<td>94.2</td>
<td>12.5</td>
<td>16.8</td>
<td>26.8</td>
<td>0.6</td>
<td>39.4</td>
<td>79.9</td>
<td>80.3</td>
</tr>
<tr>
<td>E5-V [17]</td>
<td>86.7</td>
<td>84.0</td>
<td>79.5</td>
<td>84.0</td>
<td>82.4</td>
<td>88.2</td>
<td>24.8</td>
<td>18.5</td>
<td>54.6</td>
<td>10.0</td>
<td>19.2</td>
<td>83.2</td>
<td>84.7</td>
</tr>
<tr>
<td>MagicLens-L [52]</td>
<td>85.5</td>
<td>59.3</td>
<td>72.5</td>
<td>60.9</td>
<td>24.2</td>
<td>84.6</td>
<td>29.6</td>
<td>16.3</td>
<td>28.0</td>
<td>3.3</td>
<td>22.6</td>
<td>62.7</td>
<td>75.9</td>
</tr>
<tr>
<td>EVA-CLIP-8B [36]</td>
<td>91.2</td>
<td>77.8</td>
<td>80.8</td>
<td>93.1</td>
<td>80.4</td>
<td>95.6</td>
<td>6.0</td>
<td>13.1</td>
<td>23.2</td>
<td>1.2</td>
<td>22.1</td>
<td>59.4</td>
<td>81.7</td>
</tr>
<tr>
<td>EVA-CLIP-18B [36]</td>
<td>92.1</td>
<td>81.7</td>
<td>83.3</td>
<td>94.0</td>
<td>83.3</td>
<td>96.7</td>
<td>6.1</td>
<td>13.6</td>
<td>24.7</td>
<td>1.0</td>
<td>21.9</td>
<td>63.8</td>
<td>83.1</td>
</tr>
<tr>
<td>LamRA-Ret [28]</td>
<td>93.3</td>
<td>95.1</td>
<td>82.8</td>
<td>88.1</td>
<td>94.3</td>
<td>92.7</td>
<td>33.2</td>
<td>18.9</td>
<td>62.8</td>
<td>23.1</td>
<td>60.9</td>
<td>79.6</td>
<td>85.8</td>
</tr>
<tr>
<td colspan="14">Ours Method</td>
</tr>
<tr>
<td>TRACE</td>
<td>94.9</td>
<td>94.8</td>
<td>84.5</td>
<td>89.1</td>
<td>94.1</td>
<td>94.5</td>
<td>34.8</td>
<td>20.5</td>
<td>65.4</td>
<td>25.8</td>
<td>63.2</td>
<td>84.1</td>
<td>87.5</td>
</tr>
<tr>
<td>Improvement</td>
<td>+1.6</td>
<td>-0.3</td>
<td>+1.7</td>
<td>+1.0</td>
<td>-0.2</td>
<td>+1.8</td>
<td>+1.6</td>
<td>+1.6</td>
<td>+2.6</td>
<td>+2.7</td>
<td>+2.3</td>
<td>+4.5</td>
<td>+1.7</td>
</tr>
</tbody>
</table>

结果显示TRACE在推理密集的零样本任务上优势明显：
1.  在零样本组合检索数据集CIRCO上，MAP@5比LamRA高1.6%；
2.  在多轮交互任务Multi-round FashionIQ和Visual Dialog上，分别提升2.3%和2.6%；
3.  仅在少数简单匹配任务上与基线持平，说明TRACE学习的是通用的意图分解认知能力，而非拟合训练集分布。
## 6.4. 消融实验分析
### 6.4.1. 特征提取位置消融
验证不同嵌入提取位置的效果（原文Table 4）：

<table>
<thead>
<tr>
<th>Pooling Position</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td>Last Token (<|emb|>)</td>
<td>24.46</td>
<td>52.25</td>
<td>63.81</td>
</tr>
<tr>
<td>Structure &lt;/think&gt;</td>
<td>24.36</td>
<td>53.06</td>
<td>63.93</td>
</tr>
<tr>
<td>Pre-Token (Ours)</td>
<td>28.37</td>
<td>57.03</td>
<td>68.30</td>
</tr>
</tbody>
</table>

结果证明$<|emb|>$前导token的隐状态是最优的语义瓶颈，相比其他位置R@5提升近4个百分点。
### 6.4.2. CoT组件消融
验证不同推理数据组件的效果（原文Table 5）：

<table>
<thead>
<tr>
<th>Data Components</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td>Direct Training (None)</td>
<td>24.52</td>
<td>53.15</td>
<td>63.10</td>
</tr>
<tr>
<td>&lt;answer&gt; Only</td>
<td>25.99</td>
<td>53.67</td>
<td>64.48</td>
</tr>
<tr>
<td>&lt;reasoning&gt; Only</td>
<td>26.38</td>
<td>53.41</td>
<td>64.60</td>
</tr>
<tr>
<td>Full CoT</td>
<td>28.37</td>
<td>57.03</td>
<td>68.30</td>
</tr>
</tbody>
</table>

结果显示完整的推理轨迹（推理链+答案）效果最优，说明意图分解的逐步过程加上显式结论能提供最丰富的语义指导。
### 6.4.3. 不对称推理消融
验证查询侧和候选侧加推理的效果（原文Table 6）：

<table>
<thead>
<tr>
<th>Query Side</th>
<th>Candidate Side</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td>Full CoT</td>
<td>Full CoT</td>
<td>7.75</td>
<td>18.90</td>
<td>26.50</td>
</tr>
<tr>
<td>Full CoT</td>
<td>&lt;answer&gt;</td>
<td>8.61</td>
<td>19.78</td>
<td>26.88</td>
</tr>
<tr>
<td>Full CoT</td>
<td>None (Raw)</td>
<td>28.37</td>
<td>57.03</td>
<td>68.30</td>
</tr>
</tbody>
</table>

结果验证了检索推理的不对称性：候选侧加任何生成内容都会导致性能灾难性下降，因为候选需要是稳定的视觉锚点，生成文本会让嵌入过拟合语言模式，破坏与查询空间的对齐。
### 6.4.4. 超参数消融
补充材料中的超参数消融结果：
1.  **LoRA超参数消融**：最优配置为rank=128，alpha=256，平衡性能和参数开销；
2.  **损失权重消融**：最优配置为$\lambda_{\text{gen}}=1.0, \lambda_{\text{ret}}=1.0$，平衡推理生成和检索表示的优化目标。

    ---

# 7. 总结与思考
## 7.1. 结论总结
本文提出了TRACE通用多模态检索框架，将传统的“直接编码”范式升级为“先推理再编码”的新范式，核心结论包括：
1.  显式将生成式推理整合到判别式嵌入流程中，能有效解决复杂用户意图的语义鸿沟问题，大幅提升推理密集型检索任务的性能；
2.  通过混合难度数据集的训练，模型可以自动学习隐式自适应路由能力，无需手动设计门控网络，就能实现检索精度和推理吞吐量的最优平衡；
3.  检索任务存在固有的推理不对称性：仅查询侧的推理是有益的，候选侧的推理会导致性能崩溃，这一发现为后续检索相关研究提供了重要指导；
4.  TRACE在M-BEIR基准上达到新的SOTA，同时具备出色的零样本迁移能力，能适配未见域和新约束场景。
## 7.2. 局限性与未来工作
作者指出的局限性：
1.  复杂查询的自回归生成会引入额外延迟，相比纯前馈编码器吞吐量更低；
2.  M-BEIR-CoT数据集是大模型合成的，嵌入质量受限于教师模型的推理上限，可能存在推理偏见，极端分布外场景下可能出现幻觉。
    未来工作方向：
1.  采用推测解码等技术降低推理延迟，进一步提升吞吐量；
2.  引入人工标注循环优化数据集质量，减少偏见，提升推理鲁棒性；
3.  系统研究旋转位置嵌入（RoPE）在非对称推理模式下的行为，解释候选侧推理崩溃的底层机制。
## 7.3. 个人启发与批判
### 7.3.1. 启发
1.  **自适应推理的思路可迁移**：这种根据输入复杂度动态选择推理深度的思路，不仅适用于检索，还可以迁移到多模态VQA、对话系统等其他场景，平衡效率与性能；
2.  **不对称性发现的指导意义**：检索任务中查询和候选的角色完全不同，后续相关研究不需要浪费资源尝试候选侧推理，同时这一思路也可以扩展到其他匹配类任务；
3.  **端到端整合的优势**：相比多阶段pipeline，端到端训练推理和表示学习能保留更丰富的语义信息，避免离散文本瓶颈的信息损失。
### 7.3.2. 潜在改进方向
1.  目前的自适应路由仅依赖第一个token的概率，可能被对抗样本误导（比如简单查询加入干扰词触发不必要的推理），可以引入更鲁棒的复杂度评估机制；
2.  目前的推理轨迹仅为文本形式，对于纯视觉推理任务，未来可以引入视觉形式的推理轨迹，进一步提升视觉类检索的性能；
3.  目前仅支持单轮查询的自适应推理，未来可以扩展到多轮对话检索场景，利用历史对话的推理上下文。