# 1. 论文基本信息
## 1.1. 标题
论文标题为《ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval》，核心主题是解决生成式多模态大语言模型（MLLM）适配组合图像检索任务时出现的能力退化问题，提出了一套模型无关的校准框架ReCALL。
## 1.2. 作者
核心作者团队及隶属机构如下：
- 第一作者：Tianyu Yang，中国科学院自动化研究所、中国科学院大学
- 共同作者：Chenwei He（东南大学）、Xiangzhao Hao（中科院自动化所、国科大）、Tianyue Wang（中科院自动化所、国科大）、Jiarui Guo（北京邮电大学）等
- 通讯作者：Haiyun Guo（中科院自动化所、国科大）、Leigang Qu（新加坡国立大学）
- 合作机构还包括武汉大学人工智能研究院、广东技术师范大学等。
## 1.3. 发表期刊/会议
当前为2026年2月发布的arXiv预印本，暂未正式发表于期刊或会议。该研究属于多模态检索、大模型应用领域的前沿工作，相关成果达到了领域顶级会议（如CVPR、ICML、ACM MM等）的收录水平。
## 1.4. 发表年份
2026年
## 1.5. 摘要
组合图像检索（CIR）的目标是基于参考图像+修改文本的混合查询检索目标图像。早期双塔视觉语言模型（VLM）难以满足该任务所需的跨模态组合推理能力，而近期将生成式MLLM适配为检索器的方案忽略了一个核心问题：将生成式MLLM压缩为单嵌入判别式检索器会引发范式冲突，导致**能力退化**——即检索适配后模型原生的细粒度推理能力下降。
为解决该问题，论文提出了模型无关的ReCALL框架，遵循「诊断-生成-精炼」的流水线：1. 通过自导向信息实例挖掘诊断检索器的认知盲区；2. 提示基础MLLM生成修正指令和三元组，通过VQA一致性过滤实现质量控制；3. 基于分组对比策略在生成的三元组上持续训练，将细粒度视觉-语义差异内化到检索器的表示中，让检索器的判别嵌入空间与MLLM原生的组合推理能力重新对齐。
在CIRR和FashionIQ两个主流基准上的大量实验表明，ReCALL能持续校准退化的能力，达到最先进的检索性能。
## 1.6. 原文链接
- 预印本主页：https://arxiv.org/abs/2602.01639 （预印本状态）
- PDF链接：https://arxiv.org/pdf/2602.01639
- 代码仓库：https://github.com/RemRico/Recall

  ---

# 2. 整体概括
## 2.1. 研究背景与动机
### 2.1.1. 核心问题与研究价值
组合图像检索（Composed Image Retrieval, CIR）是多模态检索领域的核心任务，支持用户通过「参考图+修改描述」的方式表达复杂检索需求，在电商搜索、创意设计、智能客服等场景有极高的应用价值。例如用户上传一件红色连衣裙的图片，输入修改文本「换成蓝色条纹的半袖款式」，系统需要从海量商品库中找到符合要求的目标连衣裙。
### 2.1.2. 现有研究的空白
早期基于CLIP的双塔VLM方案存在跨模态融合浅、组合推理能力弱的缺陷，难以处理细粒度属性修改、空间关系变化等复杂查询。近年研究开始将生成式MLLM（如Qwen-VL、LLaVA等）适配为检索器，凭借MLLM的深度跨模态融合能力获得了性能提升，但普遍忽略了一个核心矛盾：
- MLLM原生是**生成式范式**：通过分步推理输出文本，擅长细粒度语义理解、逻辑推理；
- 检索器是**判别式范式**：将输入编码为单个固定维度的嵌入向量，通过向量相似度排序，目标是快速匹配而非分步推理。
  两种范式的内在冲突会导致**能力退化**：MLLM微调为检索器后，原生的细粒度推理能力被抑制，原本能通过分步推理正确理解的查询，微调后反而会出错。
论文通过实验直观验证了该现象：

![Figure 1. Empirical illustration of Capability Degradation and the effectiveness of ReCALL $\\left( \\mathcal { R } _ { \\mathrm { r e f i n e } } \\right)$ . (a) We compare the Foundation MLLM $( \\mathcal { F } )$ under its native VQA-based generative paradigm with its fine-tuned retrieval counterpart $( \\mathcal { R } _ { \\mathrm { b a s e } } )$ under a similarity-based discriminative paradigm using a challenging query that requires fine-grained reasoning. The base retriever $\\mathcal { R } _ { \\mathrm { b a s e } }$ fails due to fine-grained grounding errors, while $\\mathcal { F }$ succeeds through step-wise reasoning. b) Quantitative evidence of Capability Degradation and Recalibration. We test $\\mathcal { R } _ { \\mathrm { b a s e } }$ on a subset of 1k instances where $\\mathcal { F }$ successfully retrieves the target (i.e., $\\mathcal { F }$ achieves $1 0 0 \\% \\mathrm { R } @ 1 .$ ). The low $\\mathbb { R } \\ @ 1$ performance of $\\mathcal { R } _ { \\mathrm { b a s e } }$ (only $6 2 . 3 3 \\%$ on CIRR and $5 5 . 8 0 \\%$ on FashionIQ) on this ${ \\mathcal { F } }$ -solvable subset provides quantifiable proof of capability degradation. Our proposed ReCALL framework effectively recovers the lost abilities, elevating $\\mathcal { R } _ { \\mathrm { b a s e } }$ to ${ \\mathcal { R } } _ { \\mathrm { r e f i n e } }$ with significant gains.](images/1.jpg)
*该图像是图表，展示了能力退化的定性说明和 ReCALL 框架带来的定量校准结果。左侧 (a) 通过实例展示了基础模型 $R_{base}$ 在复杂查询下的细粒度错误，而右侧 (b) 则展示了在 CIRR 和 FashionIQ 数据集上的 `R@1` 性能，表明 ReCALL 的有效性。*

如上图所示，左图中基线检索器$\mathcal{R}_{base}$无法区分「半袖」和「无袖」的细粒度差异，而原生MLLM$\mathcal{F}$可以通过VQA推理正确理解查询；右图量化结果显示，在$\mathcal{F}$能100%正确回答的1000个样本子集上，$\mathcal{R}_{base}$的R@1仅为62.33%（CIRR数据集）和55.80%（FashionIQ数据集），充分证明了能力退化的严重性。
### 2.1.3. 论文的创新切入点
论文没有采用常规的「增加数据、调整模型结构」的优化思路，而是从能力退化的根源出发：既然退化是因为检索适配丢失了MLLM原生的推理能力，那么就用原生MLLM的推理信号作为监督，把丢失的能力重新注入到检索器的嵌入空间中，实现能力校准。
## 2.2. 核心贡献/主要发现
论文的核心贡献可以总结为三点：
1. **现象发现**：首次明确提出并验证了MLLM适配检索任务时的「能力退化」现象，指出其根源是生成式到判别式的范式冲突。
2. **方法创新**：提出了模型无关的ReCALL框架，通过「诊断-生成-精炼」的流水线，利用原生MLLM的推理能力校准检索器的嵌入空间，不需要修改模型结构，可适配任意MLLM backbone。
3. **性能验证**：在CIRR（开放域）和FashionIQ（细粒度时尚领域）两个主流CIR基准上达到了最先进（SOTA）性能，相比基线检索器R@1最高提升8.38%，充分验证了方法的有效性。

   ---

# 3. 预备知识与相关工作
本部分为初学者梳理理解论文所需的前置知识，以及该领域的技术演进脉络。
## 3.1. 基础概念
### 3.1.1. 组合图像检索（CIR）
- **定义**：给定参考图像$I_r$和修改文本$T_m$，从大规模图像库中检索符合修改要求的目标图像$I_t$。查询是图像+文本的混合模态，需要模型同时理解图像内容、文本语义，以及两者之间的修改逻辑。
- **难点**：需要处理细粒度属性变化（如颜色、图案、袖子长度）、空间关系变化（如物体朝向、位置）、逻辑组合（如「把左边的猫换成狗，背景换成沙滩」）等复杂推理需求。
### 3.1.2. 多模态大语言模型（MLLM）
- **定义**：在大语言模型（LLM）的基础上扩展了视觉输入能力，能够同时接收图像和文本输入，输出自然语言。代表性模型包括Qwen-VL系列、LLaVA系列、InternVL系列等。
- **核心优势**：通过大规模多模态预训练获得了强大的跨模态理解、细粒度推理、指令遵循能力，原生支持通过分步思维链（CoT）推理解决复杂问题。
### 3.1.3. 对比学习与InfoNCE损失
- **定义**：对比学习是表示学习的核心方法，目标是让正样本对的表示相似度尽可能高，负样本对的相似度尽可能低，从而学习到具有判别性的表示。
- **InfoNCE损失**：对比学习最常用的损失函数，由van den Oord等人在2018年提出，公式如下：
  $$
  \mathcal{L}_{\text{infoNCE}} = -\log \frac{\exp(s(z_q, z_{t^+})/\tau)}{\sum_{z_t \in \mathcal{B}} \exp(s(z_q, z_t)/\tau)}
  $$
  其中$s(u,v)=\frac{u^\top v}{\|u\|\|v\|}$是余弦相似度，$z_q$是查询的表示，$z_{t^+}$是正样本的表示，$\mathcal{B}$是批次内所有样本的表示集合，$\tau$是温度超参数，控制相似度的分布锐度。
### 3.1.4. 思维链（Chain-of-Thought, CoT）
- **定义**：一种提示技术，通过让大模型先输出中间推理步骤，再给出最终答案，显著提升模型的复杂推理能力，尤其适合细粒度语义理解、逻辑推理类任务。
## 3.2. 前人工作
### 3.2.1. 组合图像检索技术演进
CIR技术的发展可以分为三个阶段：
1. **早期手工特征阶段**：基于手工设计的视觉特征和文本特征，通过简单的特征融合实现检索，推理能力极弱，仅能处理非常简单的修改需求。
2. **双塔VLM阶段**：以CLIP为代表的预训练视觉语言模型出现后，主流方案采用双塔结构（图像编码器+文本编码器），通过额外的融合模块或者伪token拼接实现跨模态对齐。但这类方案的跨模态交互仅发生在特征层面，融合程度浅，难以处理需要组合推理的复杂查询。
3. **MLLM驱动阶段**：近年研究开始将生成式MLLM作为编码器适配CIR任务，代表性工作如CIR-LVLM，凭借MLLM的深度跨模态融合能力获得了大幅性能提升，但所有这类工作都忽略了范式冲突导致的能力退化问题。
### 3.2.2. MLLM自改进技术
大模型的自改进是近年的热门研究方向，代表性工作包括：
- STaR：利用模型自身生成的推理路径监督训练，强化正确推理能力；
- Reflexion、Self-Refine：引入自反馈循环，迭代修正模型的输出错误。
  但这类自改进技术主要面向生成任务，此前没有工作将其应用到检索任务的能力校准中。
## 3.3. 技术演进脉络
CIR领域的核心优化目标始终是提升跨模态组合推理能力：从早期的特征拼接，到双塔模型的浅融合，再到MLLM的深度融合，性能逐步提升。但MLLM适配检索时的范式冲突成为了新的性能瓶颈，ReCALL是首个针对该瓶颈提出系统性解决方案的工作，填补了这一研究空白。
## 3.4. 差异化分析
- 与现有MLLM-based CIR方法的区别：现有方法仅关注如何将MLLM适配为检索器，忽略了适配过程中的能力退化；ReCALL则针对退化问题，通过自改进的方式将MLLM原生的推理能力注入检索器，在不增加模型参数的前提下实现性能提升。
- 与现有MLLM自改进方法的区别：现有自改进方法针对生成任务设计，ReCALL是首个面向检索任务的自改进框架，解决了判别式表示空间与生成式推理信号的对齐问题。

  ---

# 4. 方法论
ReCALL是模型无关的框架，可适配任意生成式MLLM backbone，整体流程如下图所示：

![Figure 2. Overview of the ReCALL framework. (1) Stage 1: A baseline retriever $\\mathcal { R } _ { b a s e }$ is adapted from the foundation model $\\mathcal { F }$ via standard fine-tuning. (2) Stage 2 (Diagnose): $\\mathcal { R } _ { b a s e }$ surfaces its own failure cases via self-guided informative instance mining. (3) Stage 3 (Generate): Leveraging native reasoning (CoT), $\\mathcal { F }$ synthesizes minimally edited corrective instructions for the mined informative produce the final $\\mathcal { R } _ { r e f i n e }$ , effectively recalibrating the degraded capabilities.](images/2.jpg)
*该图像是图表，展示了 ReCALL 框架的概述，包括四个阶段的流程：阶段 1 为基线检索模型的适应，阶段 2 为自导向信息实例挖掘，阶段 3 为生成校准，以及阶段 4 的针对性精炼，涉及多个查询和目标实例的调整与反向矢量相似性度量 $L_{InfoNCE}$ 的使用。*

整个框架分为4个阶段：1. 基线检索器适配；2. 自导向信息实例挖掘（诊断阶段）；3. 生成校准（生成阶段）；4. 针对性精炼（精炼阶段）。以下逐层详解每个阶段的原理与实现。
## 4.1. 问题定义与核心模型实体
首先明确任务定义和三个核心模型组件：
1. **任务定义**：给定参考图像$I_r$和修改文本$T_m$，从图像库中检索目标图像$I_t$。
2. <strong>基础模型（$\mathcal{F}$）</strong>：原生的生成式MLLM，具有强组合推理能力，为整个框架提供监督信号来源。
3. <strong>基线检索器（$\mathcal{R}_{base}$）</strong>：将$\mathcal{F}$通过CIR三元组对比学习微调得到的检索器，具备基础检索能力，但存在能力退化问题，是优化的起点。
4. <strong>精炼检索器（$\mathcal{R}_{refine}$）</strong>：经过ReCALL流水线优化后的最终检索器，恢复了退化的细粒度推理能力。
## 4.2. 阶段1：基线检索器适配
该阶段的目标是将生成式MLLM$\mathcal{F}$适配为具备基础检索能力的$\mathcal{R}_{base}$，为后续诊断提供稳定起点。
- 初始化：直接使用$\mathcal{F}$的预训练参数初始化$\mathcal{R}_{base}$，尽可能保留预训练知识。
- 训练目标：在CIR标注三元组$(I_r, T_m, I_t)$上采用InfoNCE损失微调，让查询表示$z_q$（由$I_r$和$T_m$编码得到）与正样本目标$I_t$的表示$z_{t^+}$对齐，同时与批次内的负样本表示拉开距离。损失函数与3.1.3中定义的标准InfoNCE完全一致：
  $$
  \mathcal { L } _ { i n f o N C E } = - \log \frac { \exp ( s ( z _ { q }, z _ { t ^ { + } } ) / \tau ) } { \sum _ { z _ { t } \in \mathcal { B } } \exp ( s ( z _ { q }, z _ { t } ) / \tau ) } ,
  $$
  其中$s(u,v)=\frac{u^\top v}{\|u\|\|v\|}$为余弦相似度，$\mathcal{B}$为批次内所有目标表示的集合，$\tau$为温度超参数。
该阶段完成后得到的$\mathcal{R}_{base}$已经可以实现基础的CIR功能，但因为生成式到判别式的范式冲突，原生的细粒度推理能力被抑制，存在大量认知盲区。
## 4.3. 阶段2：自导向信息实例挖掘（诊断阶段）
该阶段的目标是自动找出$\mathcal{R}_{base}$的认知盲区，也就是它无法正确处理的失败案例，为后续优化提供针对性的目标。
- 推理筛选：使用$\mathcal{R}_{base}$在训练集上做批量检索，筛选出「目标图像$I_t$没有排在检索结果第一位」的查询作为失败案例。
- 信息实例挖掘：对于每个失败案例，取出排在$I_t$之前的Top-K个错误检索结果，构成信息实例集合$\{I_h\}$。这些$I_h$就是$\mathcal{R}_{base}$的认知盲区：它们和真实目标$I_t$仅有非常细微的视觉/语义差异，刚好骗过了能力退化的检索器，是最有价值的优化靶点。
  和随机采样样本相比，该策略将算力完全集中在模型的薄弱点，数据效率提升显著。
## 4.4. 阶段3：生成校准（生成阶段）
该阶段的目标是利用原生MLLM$\mathcal{F}$的推理能力，为每个信息实例$I_h$生成高质量的修正监督信号，构造新的训练三元组。
### 4.4.1. CoT辅助生成
核心思路是生成**最小修改**的文本指令$\tilde{T}_m$，使得新三元组$(I_r, \tilde{T}_m, I_h)$是有效的训练样本，且$\tilde{T}_m$和原$T_m$的差异恰好对应$I_h$和$I_t$的视觉差异，为模型提供最细粒度的监督信号。
生成过程分为两步，通过CoT提示引导$\mathcal{F}$完成：
1. **意图分解与验证**：让$\mathcal{F}$将原修改文本$T_m$拆分为多个原子意图（如「颜色改为蓝色」、「袖子改为半袖」），逐一验证每个意图在$I_r$和$I_h$上是否成立，找出不成立的意图。
2. **最小修改合成**：保留所有符合$I_h$的意图，仅修改不成立的部分，生成和原文本风格、长度一致的修正指令$\tilde{T}_m$。
   例如原指令是「把这件红色无袖连衣裙改成蓝色半袖」，而$I_h$是蓝色无袖连衣裙，那么修正后的$\tilde{T}_m$就是「把这件红色无袖连衣裙改成蓝色无袖」，仅修改了「半袖」相关的描述，最小化语义偏移。
### 4.4.2. VQA辅助质量控制
为了过滤掉生成过程中出现的幻觉、语义不匹配的噪声样本，采用VQA一致性校验：
- 针对$\tilde{T}_m$中的关键属性（如颜色、款式、空间关系）构造VQA问题，提示$\mathcal{F}$回答。
- 仅保留答案置信度高、语义完全一致的三元组进入下一阶段，过滤掉低质量样本。
  论文通过人工验证，该质量控制机制的准确率达到92%，有效保证了监督信号的可靠性。
## 4.5. 阶段4：针对性精炼（精炼阶段）
该阶段的目标是利用生成的修正三元组微调$\mathcal{R}_{base}$，将细粒度差异内化到检索器的表示中，得到最终的$\mathcal{R}_{refine}$。
### 4.5.1. 分组对比精炼
为了最大化修正样本的价值，采用结构化的批次构造策略：
- 每个微批次组包含原正样本三元组$(I_r, T_m, I_t)$和对应的修正三元组$(I_r, \tilde{T}_m, I_h)$。
- 这种分组方式让模型在单次梯度更新中，同时面对仅有细微差异的两个查询、两个目标，强制模型学习区分$I_t$和$I_h$的细粒度语义特征，精准修正决策边界。
### 4.5.2. 双优化目标
采用混合损失函数，平衡全局检索性能和细粒度修正效果：
1. **全局InfoNCE损失**：沿用阶段1的InfoNCE损失，保留模型已经学到的全局检索能力，避免局部修正破坏整体表示空间的结构，公式与阶段1完全一致。
2. **组内三元组边际损失**：针对每个微批次组，强制拉开正样本和对应信息实例的相似度，解决细粒度混淆问题，公式如下：
   $$
   \mathcal { L } _ { t r i p l e t } = \operatorname* { m a x } ( 0, s ( z _ { q }, z _ { t ^ { - } } ) - s ( z _ { q }, z _ { t ^ { + } } ) + m ) ,
   $$
   其中$m$是边际超参数，$z_{t^-}$是信息实例$I_h$的表示，$z_{t^+}$是真实目标$I_t$的表示。该损失要求查询与正样本的相似度至少比查询与对应难负样本$I_h$的相似度高$m$，明确修正模型的混淆点。
最终总损失为两者的加权和：
$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { i n f o N C E } + \lambda \mathcal { L } _ { t r i p l e t } ,
$$
其中$\lambda$是权重超参数，用于平衡全局对齐和针对性修正的权重。

---

# 5. 实验设置
## 5.1. 数据集
实验在两个主流CIR基准上进行，覆盖开放域和细粒度垂直领域两个场景：
### 5.1.1. CIRR数据集
- 来源：基于真实世界数据集NLVR2构建，属于开放域CIR基准。
- 特点：包含复杂的物体交互、空间关系变化、数量变化等查询，比如「把左边站立的猫换成坐着的狗」，用于测试模型的泛化能力和复杂推理能力。
- 规模：包含约17K个三元组，覆盖多种常见物体和场景。
### 5.1.2. FashionIQ数据集
- 来源：电商平台的时尚商品数据集，属于细粒度垂直领域CIR基准。
- 特点：分为连衣裙（Dress）、衬衫（Shirt）、上衣（Top&Tee）三个类别，修改文本都是服装属性的细粒度变化，比如「把这件红色连衣裙的图案改成白色波点，袖子改成半袖」，用于测试模型的细粒度属性理解能力。
- 规模：包含约45K个三元组，是时尚领域CIR的标准评测基准。
## 5.2. 评估指标
论文采用两类标准CIR评估指标，以下按要求逐一解释：
### 5.2.1. Recall@K（R@K）
1. **概念定义**：衡量所有查询中，目标图像排在检索结果前K位的比例，数值越高说明检索准确性越好，核心关注模型能否把正确目标放在靠前的位置。
2. **数学公式**：
   $$
   R@K = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{rank}(I_{t,i}) \leq K \right)
   $$
3. **符号解释**：$N$是总查询数，$\mathbb{I}(\cdot)$是指示函数，括号内条件成立时取值为1，否则为0，$\text{rank}(I_{t,i})$是第$i$个查询的目标图像在全量检索结果中的排名。
### 5.2.2. Recall_subset@K（$R_{\text{subset}}@K$）
1. **概念定义**：CIRR数据集特有的指标，每个查询会提供一个包含6个候选的子集（1个正样本+5个难负样本），衡量目标在该子集的检索结果中排在前K位的比例，专门测试模型区分高度相似的难负样本的能力，更能反映细粒度判别力。
2. **数学公式**：
   $$
   R_{\text{subset}}@K = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( \text{rank}_{\text{subset}}(I_{t,i}) \leq K \right)
   $$
3. **符号解释**：$\text{rank}_{\text{subset}}(I_{t,i})$是第$i$个查询的目标在对应6候选子集中的排名，其余符号与R@K一致。
### 5.2.3. 评测规则
- FashionIQ：报告三个类别的平均R@10、R@50。
- CIRR：报告全库的R@1、R@5、R@10、R@50，以及子集的$R_{\text{subset}}@1$、$R_{\text{subset}}@2$、$R_{\text{subset}}@3$。
## 5.3. 对比基线
论文将ReCALL与两类主流方法对比，保证实验的公平性和说服力：
1. **传统双塔VLM方法**：包括TIRG（CVPR2019）、ARTEMIS（ICLR2022）、SPRC（ICLR2024）等，涵盖近5年的代表性CIR方案。
2. **最新MLLM-based方法**：包括CIR-LVLM（AAAI2025）、QuRe（ICML2025）、CCIN（CVPR2025）、TME（CVPR2025）等，均为2025年之后的前沿SOTA方案。
## 5.4. 实现细节
- 主干网络：采用Qwen2.5-VL-7B作为默认backbone，也验证了Qwen3-VL-8B、LLaVA-NeXT等主干的泛化性。
- 微调策略：采用LoRA（低秩适配）微调，秩r=16，仅调整少量参数，最大化保留预训练知识。
- 硬件：8张NVIDIA H20 GPU。
- 超参数：
  - FashionIQ：学习率4e-5，温度τ=0.03，全局batch size 512，阶段1训练200步，阶段4训练250步，λ=0.3，边际m=0.05。
  - CIRR：学习率2e-5，温度τ=0.02，全局batch size 512，阶段1训练300步，阶段4训练350步，λ=0.25，边际m=0.05。

    ---

# 6. 实验结果与分析
## 6.1. 核心结果分析
### 6.1.1. CIRR数据集结果
以下是原文Table1（CIRR测试集性能对比）的完整结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Venue</th>
<th colspan="4">Recall@K</th>
<th colspan="3">Recall<sub>subset</sub>@K</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>K = 1</th>
<th>K = 5</th>
<th>K = 10</th>
<th>K = 50</th>
<th>K = 1</th>
<th>K = 2</th>
<th>K = 3</th>
</tr>
</thead>
<tbody>
<tr>
<td>TIRG [51]</td>
<td>CVPR'19</td>
<td>14.61</td>
<td>48.37</td>
<td>64.08</td>
<td>90.03</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>ARTEMIS [11]</td>
<td>ICLR'22</td>
<td>16.96</td>
<td>46.10</td>
<td>61.31</td>
<td>87.73</td>
<td>39.99</td>
<td>62.20</td>
<td>75.67</td>
<td>43.05</td>
</tr>
<tr>
<td>TG-CIR [57]</td>
<td>MM'23</td>
<td>45.25</td>
<td>78.29</td>
<td>87.16</td>
<td>97.30</td>
<td>72.84</td>
<td>89.25</td>
<td>95.13</td>
<td>75.57</td>
</tr>
<tr>
<td>SPRC [3]</td>
<td>ICLR'24</td>
<td>51.96</td>
<td>82.12</td>
<td>89.74</td>
<td>97.69</td>
<td>80.65</td>
<td>92.31</td>
<td>96.60</td>
<td>81.39</td>
</tr>
<tr>
<td>LIMN [56]</td>
<td>TPAMI'24</td>
<td>43.64</td>
<td>75.37</td>
<td>85.42</td>
<td>97.04</td>
<td>69.01</td>
<td>86.22</td>
<td>94.19</td>
<td>72.19</td>
</tr>
<tr>
<td>CoVR-2 [50]</td>
<td>TPAMI'24</td>
<td>50.43</td>
<td>81.08</td>
<td>88.89</td>
<td>98.05</td>
<td>76.75</td>
<td>90.34</td>
<td>95.78</td>
<td>79.28</td>
</tr>
<tr>
<td>CaLa [20]</td>
<td>SIGIR'24</td>
<td>49.11</td>
<td>81.21</td>
<td>89.59</td>
<td>98.00</td>
<td>76.27</td>
<td>91.04</td>
<td>96.46</td>
<td>78.74</td>
</tr>
<tr>
<td>ENCODER [26]</td>
<td>AAAI'25</td>
<td>46.10</td>
<td>77.98</td>
<td>87.16</td>
<td>97.64</td>
<td>76.92</td>
<td>90.41</td>
<td>95.95</td>
<td>77.45</td>
</tr>
<tr>
<td>CIR-LVLM [45]</td>
<td>AAAI'25</td>
<td>53.64</td>
<td>83.76</td>
<td>90.60</td>
<td>97.93</td>
<td>79.12</td>
<td>92.33</td>
<td>96.67</td>
<td>81.44</td>
</tr>
<tr>
<td>QuRe [22]</td>
<td>ICML'25</td>
<td>52.22</td>
<td>82.53</td>
<td>90.31</td>
<td>98.17</td>
<td>78.51</td>
<td>91.28</td>
<td>96.48</td>
<td>80.52</td>
</tr>
<tr>
<td>CCIN [48]</td>
<td>CVPR'25</td>
<td>53.41</td>
<td>84.05</td>
<td>91.17</td>
<td>98.00</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>TME [25]</td>
<td>CVPR'25</td>
<td>53.42</td>
<td>82.99</td>
<td>90.24</td>
<td>98.15</td>
<td>81.04</td>
<td>92.58</td>
<td>96.94</td>
<td>82.01</td>
</tr>
<tr>
<td>Baseline (R<sub>base</sub>)</td>
<td>-</td>
<td>51.23</td>
<td>82.15</td>
<td>90.20</td>
<td>98.20</td>
<td>77.57</td>
<td>91.83</td>
<td>96.34</td>
<td>79.86</td>
</tr>
<tr>
<td>ReCALL (R<sub>refine</sub>)</td>
<td>-</td>
<td><b>55.52</b></td>
<td><b>84.07</b></td>
<td><b>91.83</b></td>
<td><b>98.55</b></td>
<td><b>81.49</b></td>
<td><b>93.35</b></td>
<td><b>97.64</b></td>
<td><b>82.81</b></td>
</tr>
<tr>
<td>Improvement (Δ)</td>
<td></td>
<td>+8.38%</td>
<td>+2.34%</td>
<td>+1.81%</td>
<td>+0.36%</td>
<td>+5.06%</td>
<td>+1.65%</td>
<td>+1.35%</td>
<td>+3.70%</td>
</tr>
</tbody>
</table>

**结果分析**：
- ReCALL在所有指标上均达到SOTA，R@1达到55.52%，比之前的最优方法TME高出2.1个百分点，相比基线$\mathcal{R}_{base}$提升8.38%。
- 尤其在衡量细粒度判别力的$R_{\text{subset}}@1$指标上达到81.49%，相比基线提升5.06%，充分验证了ReCALL校准细粒度推理能力的效果，完美解决了基线检索器难以区分难负样本的问题。
### 6.1.2. FashionIQ数据集结果
以下是原文Table2（FashionIQ验证集性能对比）的完整结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Venue</th>
<th colspan="2">Dress</th>
<th colspan="2">Shirt</th>
<th colspan="2">Top&amp;Tee</th>
<th colspan="2">Avg.</th>
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
<td>TIRG [51]</td>
<td>CVPR'19</td>
<td>14.13</td>
<td>34.61</td>
<td>13.10</td>
<td>30.91</td>
<td>14.79</td>
<td>34.37</td>
<td>14.01</td>
<td>33.30</td>
</tr>
<tr>
<td>ARTEMIS [11]</td>
<td>ICLR'22</td>
<td>25.68</td>
<td>51.05</td>
<td>21.57</td>
<td>44.13</td>
<td>28.59</td>
<td>55.06</td>
<td>25.28</td>
<td>50.08</td>
</tr>
<tr>
<td>FashionSAP [15]</td>
<td>CVPR'23</td>
<td>33.71</td>
<td>60.43</td>
<td>41.91</td>
<td>70.93</td>
<td>33.17</td>
<td>61.33</td>
<td>36.26</td>
<td>64.23</td>
</tr>
<tr>
<td>FAME-ViL [14]</td>
<td>CVPR'23</td>
<td>42.19</td>
<td>67.38</td>
<td>47.64</td>
<td>68.79</td>
<td>50.69</td>
<td>73.07</td>
<td>46.84</td>
<td>69.75</td>
</tr>
<tr>
<td>SyncMask [42]</td>
<td>CVPR'24</td>
<td>33.76</td>
<td>61.23</td>
<td>35.82</td>
<td>62.12</td>
<td>44.82</td>
<td>72.06</td>
<td>38.13</td>
<td>65.14</td>
</tr>
<tr>
<td>SADN [55]</td>
<td>MMI'24</td>
<td>40.01</td>
<td>65.10</td>
<td>43.67</td>
<td>66.05</td>
<td>48.04</td>
<td>70.93</td>
<td>43.91</td>
<td>67.36</td>
</tr>
<tr>
<td>CaLa [20]</td>
<td>SIGIR'24</td>
<td>42.38</td>
<td>66.08</td>
<td>46.76</td>
<td>68.16</td>
<td>50.93</td>
<td>73.42</td>
<td>46.69</td>
<td>69.22</td>
</tr>
<tr>
<td>CoVR-2 [50]</td>
<td>TPAMI'24</td>
<td>46.53</td>
<td>69.60</td>
<td>51.23</td>
<td>70.64</td>
<td>52.14</td>
<td>73.27</td>
<td>49.96</td>
<td>71.17</td>
</tr>
<tr>
<td>SPRC [3]</td>
<td>ICLR'24</td>
<td>49.18</td>
<td>72.43</td>
<td>55.64</td>
<td>73.89</td>
<td>59.35</td>
<td>78.58</td>
<td>54.72</td>
<td>74.97</td>
</tr>
<tr>
<td>CIR-LVLM [45]</td>
<td>AAAI'25</td>
<td>50.42</td>
<td>73.60</td>
<td>58.59</td>
<td>75.86</td>
<td>59.61</td>
<td>78.99</td>
<td>56.21</td>
<td>76.14</td>
</tr>
<tr>
<td>CCIN [48]</td>
<td>CVPR'25</td>
<td>49.38</td>
<td>72.58</td>
<td>55.93</td>
<td>74.14</td>
<td>57.93</td>
<td>77.56</td>
<td>54.41</td>
<td>74.76</td>
</tr>
<tr>
<td>TME [25]</td>
<td>CVPR'25</td>
<td>49.73</td>
<td>71.69</td>
<td>56.43</td>
<td>74.44</td>
<td>59.31</td>
<td>78.94</td>
<td>55.15</td>
<td>75.02</td>
</tr>
<tr>
<td>QuRe [22]</td>
<td>ICML'25</td>
<td>46.80</td>
<td>69.81</td>
<td>53.53</td>
<td>72.87</td>
<td>57.47</td>
<td>77.77</td>
<td>52.60</td>
<td>73.48</td>
</tr>
<tr>
<td>Baseline (R<sub>base</sub>)</td>
<td>-</td>
<td>46.80</td>
<td>70.60</td>
<td>55.00</td>
<td>74.39</td>
<td>57.88</td>
<td>78.12</td>
<td>53.23</td>
<td>74.37</td>
</tr>
<tr>
<td>ReCALL (R<sub>refine</sub>)</td>
<td>-</td>
<td><b>51.81</b></td>
<td><b>73.48</b></td>
<td><b>58.49</b></td>
<td><b>76.59</b></td>
<td><b>60.83</b></td>
<td><b>79.19</b></td>
<td><b>57.04</b></td>
<td><b>76.42</b></td>
</tr>
<tr>
<td>Improvement (Δ)</td>
<td></td>
<td>+10.71%</td>
<td>+4.08%</td>
<td>+6.35%</td>
<td>+2.96%</td>
<td>+5.10%</td>
<td>+1.37%</td>
<td>+7.16%</td>
<td>+2.76%</td>
</tr>
</tbody>
</table>

**结果分析**：
- ReCALL的平均R@10达到57.04%，超过之前的SOTA CIR-LVLM 0.83个百分点，相比基线提升7.16%。
- 细粒度的Dress类别R@10提升达10.71%，说明ReCALL对服装颜色、图案、款式等细粒度属性的理解能力提升尤为显著，完美适配电商时尚检索的场景需求。
### 6.1.3. 定性结果分析
如下图所示，基线检索器$\mathcal{R}_{base}$无法捕捉「半袖」、「正对镜头」等细粒度约束，而ReCALL可以准确理解这些修改需求，检索到正确的目标：

![Figure 4. Qualitative comparison between the baseline $R _ { \\mathrm { b a s e } } )$ and our ReCALL $\\left( \\mathcal { R } _ { \\mathrm { r e f i n e } } \\right)$ on FashionIQ (top) and CIRR (bottom). The green dashed boxes indicate the ground-truth targets. $\\mathcal { R } _ { \\mathrm { b a s e } }$ suffers from capability degradation, failing to capture specic details like "half](images/4.jpg)
*该图像是插图，展示了基线模型 $R_{\mathrm{base}}$ 和我们提出的 ReCALL 模型 $\mathcal{R}_{\mathrm{refine}}$ 在 FashionIQ（上方）和 CIRR（下方）数据集上的定性比较。每个部分展示了参考图像、修改说明，以及基线和 ReCALL 模型的检索结果，其中绿框标示了真实目标。ReCALL 模型在细节捕捉上表现更佳，解决了基线模型的能力退化问题。*

## 6.2. 消融实验与参数分析
### 6.2.1. 挖掘策略有效性验证
以下是原文Table3（FashionIQ验证集上挖掘策略的消融实验）的结果：

<table>
<thead>
<tr>
<th>Mining Strategy</th>
<th>R@10</th>
<th>R@50</th>
<th>Mean</th>
</tr>
</thead>
<tbody>
<tr>
<td>R<sub>base</sub></td>
<td>53.23</td>
<td>74.37</td>
<td>63.80</td>
</tr>
<tr>
<td>+ Random Mining</td>
<td>53.80±0.20</td>
<td>74.32±0.06</td>
<td>64.06±0.10</td>
</tr>
<tr>
<td>+ Self-Guided</td>
<td>57.04</td>
<td>76.42</td>
<td>66.73</td>
</tr>
</tbody>
</table>

**分析**：随机采样样本仅能带来可忽略的性能提升，而自导向挖掘（仅针对失败案例的难负样本）带来了3.81个百分点的平均提升，证明该策略将算力集中在模型盲区，数据效率远高于盲目的随机数据合成。
### 6.2.2. 核心组件有效性验证
以下是原文Table4（FashionIQ验证集上核心组件的消融实验）的结果：

<table>
<thead>
<tr>
<th rowspan="2">Baseline</th>
<th colspan="3">Components</th>
<th colspan="3">Metrics (Avg.)</th>
</tr>
<tr>
<th>CG</th>
<th>VC</th>
<th>GR</th>
<th>R@10</th>
<th>R@50</th>
<th>Mean</th>
</tr>
</thead>
<tbody>
<tr>
<td>•</td>
<td>∘</td>
<td>∘</td>
<td>∘</td>
<td>53.23</td>
<td>74.37</td>
<td>63.80</td>
</tr>
<tr>
<td>•</td>
<td>•</td>
<td>∘</td>
<td>∘</td>
<td>55.41</td>
<td>75.17</td>
<td>65.29</td>
</tr>
<tr>
<td>•</td>
<td>•</td>
<td>•</td>
<td>∘</td>
<td>56.13</td>
<td>76.04</td>
<td>66.09</td>
</tr>
<tr>
<td>•</td>
<td>•</td>
<td>•</td>
<td>•</td>
<td>57.04</td>
<td>76.42</td>
<td>66.73</td>
</tr>
</tbody>
</table>

（注：CG=CoT辅助生成，VC=VQA辅助质量控制，GR=分组对比精炼，•表示启用，∘表示禁用）
**分析**：三个组件均带来稳定的性能提升：
1. 启用CoT辅助生成：R@10提升2.18个百分点，验证了原生MLLM的推理信号可以有效补充检索器的能力缺口。
2. 再加VQA质量控制：R@10进一步提升0.72个百分点，证明质量过滤可以有效消除生成噪声，提升监督信号可靠性。
3. 再加分组对比精炼：R@10再提升0.91个百分点，证明结构化的批次构造可以让模型更好地学习细粒度差异，最大化修正样本的价值。
### 6.2.3. 跨主干泛化性验证
ReCALL是模型无关的框架，在不同MLLM主干上均能获得稳定提升：

![Figure 3. Generalizability across backbones. We validate ReCALL on different foundation models (Qwen2.5-VL-7B and Qwen3-VL-8B). Despite higher baselines, ReCALL consistently delivers performance gains on both (a) CIRR and (b) FashionIQ, confirming the strong generalizability of our framework.](images/3.jpg)
*该图像是柱状图，展示了在不同基础模型（Qwen2.5-VL-7B 和 Qwen3-VL-8B）上，ReCALL 在 CIRR 和 FashionIQ 数据集上的表现。图中展示了 R@1 和 R@10 的具体数值，体现了 ReCALL 的强大通用性。*

即使是更强的Qwen3-VL-8B主干，基线性能已经很高的情况下，ReCALL仍然能带来1.16个百分点的CIRR R@1提升，证明能力退化是MLLM适配检索任务的普遍问题，ReCALL的解决方案具有通用性。
### 6.2.4. 其他补充实验
- **超参数分析**：三元组损失的权重$\lambda$在0.3、边际$m$在0.05时性能最优，说明信息实例与正样本的相似度非常高，需要较小的边际来避免破坏全局表示空间。
- **与硬负样本挖掘对比**：仅将$I_h$作为硬负样本训练会导致性能下降，因为没有明确告诉模型$I_h$为什么是负的，反而引入冲突梯度；而ReCALL通过生成修正文本，将负样本转化为新的正样本对，避免了该问题。
- **参数规模消融**：增大LoRA秩甚至全量微调反而会降低性能，证明能力退化不是参数不足导致的，而是范式冲突的结果，更多可训练参数只会让模型更快过拟合到粗粒度检索任务，丢失更多原生推理能力。

  ---

# 7. 总结与思考
## 7.1. 结论总结
该论文首次明确了生成式MLLM适配判别式检索任务时的「能力退化」现象，指出其根源是生成式与判别式范式的内在冲突。为解决该问题，提出了模型无关的ReCALL框架，通过「诊断-生成-精炼」的自改进流水线，将原生MLLM的组合推理能力内化到检索器的嵌入空间中，实现退化能力的校准。在CIRR和FashionIQ两个主流基准上的实验表明，ReCALL不仅达到了SOTA性能，还具有强通用性和高数据效率，为MLLM在检索任务中的应用提供了新的优化思路。
## 7.2. 局限性与未来工作
论文指出的局限性和未来方向包括：
1. **复杂几何推理缺陷**：对于物体旋转、视角变换等复杂几何修改需求，ReCALL仍然存在错误，根源是当前MLLM本身的空间几何推理能力不足，未来可以从提升主干MLLM的空间推理能力入手。
2. **标注歧义问题**：当前CIR数据集普遍仅标注单个正样本，但实际上大量查询存在多个符合要求的样本，导致评估指标低估模型的实际性能，未来可以探索一对多的评估协议。
3. **相对属性歧义**：对于「更少」、「更亮」等主观的相对属性描述，模型难以和标注者的意图对齐，未来可以引入更细粒度的属性标注或者主观偏好建模。
4. **场景扩展**：当前ReCALL仅验证了CIR任务，未来可以扩展到文本-视频检索、3D检索、跨语言检索等更多检索场景，验证框架的普适性。
## 7.3. 个人启发与批判
### 7.3.1. 核心启发
该论文的研究思路极具启发性：没有陷入「堆参数、加数据」的常规优化路径，而是深入挖掘了MLLM适配下游任务时的本质矛盾，从「恢复模型原生能力」的角度切入，用极小的成本获得了极大的性能提升。实际上，能力退化现象不仅存在于CIR任务，所有将生成式MLLM适配为判别式嵌入模型的任务（如通用多模态检索、文本检索、代码检索等）都可能存在该问题，ReCALL的框架可以直接迁移到这些场景，具有极高的应用价值。
此外，论文提出的「诊断-生成-精炼」自改进范式，为大模型下游适配提供了新的通用框架：不需要人工标注额外数据，仅利用大模型自身的能力就可以实现性能提升，极大降低了下游适配的成本。
### 7.3.2. 潜在改进方向
1. **在线闭环优化**：当前生成阶段是离线完成的，未来可以设计在线的诊断-生成-精炼闭环，在训练过程中实时挖掘盲区、实时生成修正样本，进一步提升优化效率，也可以适应动态变化的检索场景。
2. **跨模型知识蒸馏**：当前修正样本由同一个MLLM生成，未来可以用更强的闭源MLLM（如GPT-4V、Gemini Advanced）生成高质量修正样本，监督开源MLLM检索器的校准，进一步提升性能。
3. **多粒度分组对比**：当前每个微批次组仅包含一个修正样本，未来可以扩展为多粒度的分组，包含多个不同差异程度的信息实例，让模型学习更丰富的细粒度语义差异。
4. **效率优化**：当前离线生成阶段需要十余个小时的计算，未来可以通过轻量化生成模型、缓存复用生成样本等方式降低成本，提升工业落地可行性。