# 1. 论文基本信息
## 1.1. 标题
论文标题为《ViCLIP-OT: The First Foundation Vision-Language Model for Vietnamese Image-Text Retrieval with Optimal Transport》，核心主题是**首次提出面向低资源语言越南语的图文检索基础视觉语言模型，通过融合对比学习与最优传输机制提升跨模态对齐效果**。
## 1.2. 作者
三位作者均来自越南芹苴大学（Can Tho University, Vietnam）：Quoc-Khang Tran、Minh-Thien Nguyen、Nguyen-Khang Pham（通讯作者），研究方向聚焦于越南语自然语言处理、多模态学习。
## 1.3. 发表期刊/会议
该论文目前为预印本，发布于arXiv预印本平台，尚未正式发表在期刊或会议上。
## 1.4. 发表年份
预印本发布时间为2026年2月26日（UTC时间）。
## 1.5. 摘要
论文旨在解决现有视觉语言模型大多针对英语等高资源语言优化、在越南语等低资源场景下效果不佳的问题，提出了专门面向越南语文图检索的基础模型ViCLIP-OT。该框架将CLIP风格的对比学习与提出的<strong>相似图正则最优传输（Similarity-Graph Regularized Optimal Transport, SIGROT）</strong> 损失结合，增强全局跨模态一致性、缓解模态间隙问题。在三个越南语基准数据集（UITOpenViIC、KTVIC、Crossmodal-3600）上的实验表明，ViCLIP-OT在域内和零样本场景下均 consistently 超过CLIP、SigLIP基线：在UIT-OpenViIC上平均Recall@K达67.34%，比CLIP提升5.75个百分点；在Crossmodal-3600零样本评估上比CLIP提升11.72个百分点。嵌入空间分析进一步验证了模型的跨模态对齐效果更好、模态间隙更小，证明了SIGROT是低资源语言跨模态检索的有效可扩展方案。
## 1.6. 原文链接
- 预印本原文链接：https://arxiv.org/abs/2602.22678
- PDF链接：https://arxiv.org/pdf/2602.22678v1
- 发布状态：arXiv预印本，未正式发表。

  ---

# 2. 整体概括
## 2.1. 研究背景与动机
### 核心问题与研究意义
跨模态图文检索是智能多媒体系统的核心组件，当前主流的视觉语言模型（如CLIP、SigLIP）大多基于英语大规模图文对训练，在<strong>低资源语言（如越南语）</strong> 场景下存在明显短板：
1.  越南语缺乏大规模公开的图文标注数据集，无法直接复用英语场景下的大规模预训练范式；
2.  现有通用方案是将越南语文本翻译为英语后调用英语模型，会引入翻译噪声，无法保留越南语特有的语义信息；
3.  传统对比学习范式仅关注实例级的配对对齐，忽略批次内样本之间的语义关系，存在模态间隙（不同模态的嵌入在共享空间中分离为两个簇）问题，影响跨模态匹配效果。

### 研究空白与切入点
当前尚未有专门面向越南语的大规模图文检索基础视觉语言模型，同时低资源场景下的跨模态对齐方法也缺乏相关研究。本文的创新切入点是：
1.  采用适配越南语的骨干网络（DINOv3视觉主干、越南语预训练Sentence-BERT文本主干）构建双编码器架构；
2.  提出SIGROT损失，结合对比学习的实例级对齐能力与最优传输的分布级对齐能力，利用批次内样本的内模态、跨模态相似关系增强全局结构一致性，缓解模态间隙。
## 2.2. 核心贡献/主要发现
### 核心贡献
论文的主要贡献可总结为三点：
1.  **首次提出面向越南语的图文检索基础模型ViCLIP-OT**：是目前首个针对越南语场景优化的大规模视觉语言基础模型，填补了低资源语言跨模态检索的研究空白；
2.  **提出SIGROT损失**：通过预计算的批次内样本相似图正则最优传输匹配过程，同时实现实例级对齐与全局语义结构保留，有效减少模态间隙；
3.  **SOTA性能与强泛化性**：在三个越南语基准数据集上取得最优效果，零样本场景下的性能提升尤为显著，证明了该方案在低资源语言场景下的可扩展性。

### 主要发现
1.  在低资源语言场景下，专门针对目标语言训练的小参数模型，效果远超大规模通用多语言模型（ViCLIP-OT参数仅221M，比2B参数的Qwen3-VL-Embedding-2B效果高13个百分点以上）；
2.  对比学习与SIGROT损失的结合能同时提升域内检索效果与零样本泛化能力，且有效降低模态间隙；
3.  相似图构建时同时融合内模态、跨模态相似信息，比仅使用单模态相似信息的效果更优。

    ---

# 3. 预备知识与相关工作
## 3.1. 基础概念
为方便初学者理解，本小节对论文涉及的核心专业术语进行逐一解释：
### 3.1.1. 图文检索（Image-Text Retrieval）
跨模态检索任务的一种，包含两个子方向：
- 文本到图像检索（Text-to-Image Retrieval）：给定文本查询，返回最相关的图像；
- 图像到文本检索（Image-to-Text Retrieval）：给定图像查询，返回最相关的文本描述。
  主流方案采用<strong>双编码器（Dual-Encoder）</strong> 架构：图像和文本各用独立的编码器映射到同一个共享嵌入空间，通过余弦相似度衡量跨模态样本的相关性，推理时可离线预计算所有样本的嵌入，在线查询仅需做向量匹配，适合大规模检索场景。
### 3.1.2. 对比学习（Contrastive Learning）
表示学习的主流范式，核心目标是让语义相似的样本在嵌入空间中距离更近，语义无关的样本距离更远。在跨模态场景下，配对的图文对为正样本，非配对的为负样本，通过损失函数拉进正样本、推开负样本，实现跨模态嵌入对齐。
### 3.1.3. CLIP与SigLIP
- **CLIP**：OpenAI于2021年提出的双编码器视觉语言模型，基于大规模英语图文对训练，采用对称交叉熵对比损失，首次实现了极强的跨模态零样本泛化能力。其损失公式为：
  $$
  \mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left(
  -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\tau \cdot \text{sim}(\mathbf{z}_i^\text{text}, \mathbf{z}_i^\text{image}))}{\sum_{j=1}^N \exp(\tau \cdot \text{sim}(\mathbf{z}_i^\text{text}, \mathbf{z}_j^\text{image}))}
  -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\tau \cdot \text{sim}(\mathbf{z}_i^\text{image}, \mathbf{z}_i^\text{text}))}{\sum_{j=1}^N \exp(\tau \cdot \text{sim}(\mathbf{z}_i^\text{image}, \mathbf{z}_j^\text{text}))}
  \right)
  $$
  其中$\tau$为可学习的温度参数，$\text{sim}(\cdot, \cdot)$为余弦相似度，$N$为批次大小。
- **SigLIP**：CLIP的改进版本，将softmax交叉熵损失替换为sigmoid损失，每个样本对独立计算损失，不需要全局softmax，训练时无需跨设备同步负样本，支持更大的批次大小，训练效率更高。其损失公式为：
  $$
  \mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N \left[ y_{ij} \log \sigma(\tau \cdot \text{sim}(\mathbf{z}_i^\text{image}, \mathbf{z}_j^\text{text}) + b) + (1-y_{ij}) \log (1-\sigma(\tau \cdot \text{sim}(\mathbf{z}_i^\text{image}, \mathbf{z}_j^\text{text}) + b)) \right]
  $$
  其中$y_{ij}=1$当且仅当第$i$个图像和第$j$个文本是配对正样本，否则为0；$\sigma$为sigmoid函数，$b$为可学习的偏置参数。
### 3.1.4. 最优传输（Optimal Transport, OT）
用于衡量两个概率分布之间距离的数学框架，核心思想是寻找将一个分布的质量转移到另一个分布的最小成本方案，广泛用于分布对齐任务。在跨模态学习中，可将图像嵌入和文本嵌入视为两个分布，通过OT实现全局分布级对齐，弥补对比学习仅关注实例级对齐的不足。
### 3.1.5. 模态间隙（Modality Gap）
跨模态表示学习中的常见现象：图像和文本的嵌入在共享空间中会分别聚成两个独立的簇，两个簇质心的距离即为模态间隙，间隙越大说明跨模态对齐效果越差，会直接影响检索性能。
### 3.1.6. 骨干网络（Backbone）
模型的基础特征提取组件，本文中图像骨干采用DINOv3（自监督预训练的视觉Transformer），文本骨干采用越南语预训练的Sentence-BERT（可生成高质量句子级嵌入的BERT变体）。
## 3.2. 前人工作
### 跨模态图文检索的技术演进
1.  早期阶段：采用单模态特征提取+跨模态相似度度量的方案，效果有限；
2.  深度学习阶段：引入注意力机制实现跨模态交互，效果提升但推理速度慢，不适合大规模检索；
3.  CLIP时代：双编码器对比学习成为主流，兼顾效果和推理效率，但大多针对英语等高资源语言优化；
4.  低资源语言适配阶段：现有方案多为翻译适配或多语言模型迁移，存在噪声大、效果差的问题，缺乏专门针对低资源语言的基础VLM。

### 最优传输在跨模态学习中的应用
已有研究将最优传输用于跨模态分布对齐，验证了OT可有效缓解模态间隙问题，但现有方法未利用批次内样本的语义结构信息，也未应用于低资源语言的VLM训练。
## 3.3. 技术演进脉络
本文的工作处在“低资源语言跨模态基础模型”与“最优传输增强跨模态对齐”两个研究方向的交叉点：针对越南语缺乏专用VLM的痛点，在CLIP双编码器架构的基础上，引入带相似图正则的最优传输损失，同时解决低资源场景下的对齐效果差、模态间隙大的问题。
## 3.4. 差异化分析

| 对比对象 | 核心差异 | 本文优势 |
|---------|---------|---------|
| 通用多语言VLM（如Qwen3-VL、Jina CLIP） | 通用模型针对多语言优化，低资源语言的语料占比低，效果差 | 专门针对越南语训练，小参数模型效果远超大规模通用多语言模型 |
| 传统CLIP/SigLIP | 仅做实例级对比对齐，忽略样本间语义结构，模态间隙大 | 加入SIGROT损失，实现实例级+分布级全局对齐，模态间隙更小，效果更优 |
| 普通OT增强的跨模态模型 | 仅做分布对齐，未利用样本间的语义结构信息 | 加入相似图正则，保留批次内样本的语义关系，对齐效果更合理 |

---

# 4. 方法论
## 4.1. 方法原理
### 核心思想
结合对比学习的实例级对齐能力与最优传输的分布级对齐能力，通过预计算的批次内样本相似图正则OT的匹配过程，让跨模态嵌入不仅配对的样本距离近，还能保留样本之间的全局语义结构，有效减少模态间隙，提升低资源语言场景下的图文检索效果与泛化能力。
### 直觉解释
传统CLIP仅拉进配对的图文对，推开非配对样本，但不会考虑“两个描述相似场景的文本对应的图像也应该相似”这类全局语义关系。SIGROT通过预训练的多模态模型提前计算批次内所有样本的语义相似度（包含文本-文本、图像-图像、文本-图像的相似度）构建相似图，再用OT寻找全局最优的跨模态匹配方案，让模型学到的匹配关系与相似图的语义结构一致，从而得到结构更合理的嵌入空间。
## 4.2. 核心方法详解
ViCLIP-OT的整体架构如下图（原文Figure 1）所示：

![Figure 1: ViCLIP-OT architecture overview. The model consists of a DINOv3-based image encoder and a Vietnamese Sentence-BERT text encoder that project images and texts into a shared embedding space. The hybrid training objective combines a CLIPstyle contrastive loss with the proposed SIGROT loss, which uses a similarity graph and optimal transport to enforce global cross-modal alignment.](images/1.jpg)
*该图像是ViCLIP-OT模型架构的示意图。图中展示了如何通过DINOv3图像编码器和越南句子BERT文本编码器，将图像和文本投影到共享嵌入空间。该模型的混合训练目标结合了CLIP风格对比损失与SIGROT损失，通过相似性图和最优传输机制实现全球跨模态对齐。*

### 4.2.1. 双编码器架构
ViCLIP-OT采用标准双编码器设计，包含图像塔和文本塔，分别将图像和文本映射到同一个$d$维共享嵌入空间。
#### 图像塔
采用DINOv3自监督预训练的视觉Transformer（ViT-B/16）作为主干：
1.  输入图像$x_i$经过DINOv3提取patch级特征；
2.  对所有patch特征做平均池化得到全局图像表示；
3.  通过线性投影层映射到$d$维嵌入空间，得到图像嵌入：
    $$\mathbf{z}_i^{\mathrm{image}} = f_{\mathrm{image}}(x_i) \in \mathbb{R}^d$$
4.  对图像嵌入做L2归一化：
    $$\tilde{\mathbf{z}}_i^{\mathrm{image}} = \frac{\mathbf{z}_i^{\mathrm{image}}}{\|\mathbf{z}_i^{\mathrm{image}}\|_2}$$
#### 文本塔
采用越南语预训练的Sentence-BERT作为主干：
1.  输入文本$t_i$经过分词后输入SBERT，得到token级特征；
2.  对非填充token的特征做平均池化得到句子级表示；
3.  若SBERT的隐藏维度与$d$不一致，通过线性投影层映射到$d$维嵌入空间，得到文本嵌入：
    $$\mathbf{z}_i^{\mathrm{text}} = f_{\mathrm{text}}(t_i) \in \mathbb{R}^d$$
4.  对文本嵌入做L2归一化：
    $$\tilde{\mathbf{z}}_i^{\mathrm{text}} = \frac{\mathbf{z}_i^{\mathrm{text}}}{\|\mathbf{z}_i^{\mathrm{text}}\|_2}$$
### 4.2.2. SIGROT损失计算
SIGROT损失的计算分为两个步骤：批次相似图构建、带正则的最优传输求解与损失计算。
#### 步骤1：批次相似图构建
对于每个训练批次，使用预训练的鲁棒多模态嵌入模型（本文采用Qwen3-VL-Embedding-2B）预计算所有样本的L2归一化图像、文本嵌入，堆叠为矩阵$E_{\text{image}}$（批次内所有图像嵌入）和$E_{\text{text}}$（批次内所有文本嵌入），然后计算四个相似矩阵：
1.  文本-文本内模态相似矩阵：$G_{\mathrm{text}} = E_{\mathrm{text}} E_{\mathrm{text}}^\intercal$
2.  图像-图像内模态相似矩阵：$G_{\mathrm{image}} = E_{\mathrm{image}} E_{\mathrm{image}}^\intercal$
3.  文本-图像跨模态相似矩阵：$G_{\mathrm{text-image}} = E_{\mathrm{text}} E_{\mathrm{image}}^\intercal$
4.  图像-文本跨模态相似矩阵：$G_{\mathrm{image-text}} = E_{\mathrm{image}} E_{\mathrm{text}}^\intercal$
    将四个矩阵取平均得到跨模态相似图，同时包含内模态和跨模态的语义关系：
$$G_{\mathrm{cross}} = \frac{1}{4} \left( G_{\mathrm{text}} + G_{\mathrm{image}} + G_{\mathrm{text-image}} + G_{\mathrm{image-text}} \right)$$
#### 步骤2：最优传输求解与损失计算
首先计算模型输出的嵌入的跨模态相似矩阵：
$$S_{\mathrm{image-text}} = \tilde{Z}_{\mathrm{image}} \tilde{Z}_{\mathrm{text}}^\intercal$$
其中$\tilde{Z}_{\mathrm{image}}$是批次内所有归一化图像嵌入的堆叠矩阵，$\tilde{Z}_{\mathrm{text}}$是文本嵌入的堆叠矩阵。定义OT的成本矩阵为：
$$C_{\mathrm{image-text}} = \mathbf{1} - S_{\mathrm{image-text}}$$
相似度越高的样本对转移成本越低，符合OT的最小化转移成本的目标。

本文采用<strong>不平衡最优传输（Unbalanced Optimal Transport, UOT）</strong> 求解传输计划，放松严格的质量守恒约束，可处理带噪声的图文对（如图像包含无关背景、文本包含非视觉词汇）：
##### 图像到文本方向的UOT求解
最优传输计划$\gamma_{\mathrm{i2t}}^*$通过最小化以下目标得到：
$$
\begin{array} { r l }
& { \gamma _ { \mathrm { i 2 t } } ^ { * } = \underset { \gamma \in \mathbb { R } _ { + } ^ { N \times N } } { \arg \operatorname* { m i n } } \left\langle \gamma , C _ { \mathrm { i m a g e - t e x t } } \right\rangle _ { F } - \varepsilon H ( \gamma ) } \\
& { \qquad + \tau _ { m 1 } \mathrm { K L } ( \gamma \mathbf { 1 } _ { N } \| \mu ) + \tau _ { m 2 } \mathrm { K L } ( \gamma ^ { \top } \mathbf { 1 } _ { N } \| \nu ) , }
\end{array}
$$
符号解释：
- $N$：批次大小；
- $\gamma$：传输计划矩阵，$\gamma_{ij}$表示将第$i$个图像的质量转移到第$j$个文本的比例；
- $\langle \cdot, \cdot \rangle_F$：Frobenius内积，即两个矩阵对应元素相乘后求和；
- $\varepsilon$：熵正则系数，$H(\gamma) = -\sum_{i,j} \gamma_{ij}(\log \gamma_{ij} -1)$为熵正则项，让优化问题严格凸，可通过Sinkhorn-Knopp算法高效求解；
- $\tau_{m1}, \tau_{m2}$：边际约束的松弛系数，值越小约束越松；
- $\mathrm{KL}(\cdot \| \cdot)$：KL散度，用于衡量两个分布的差异；
- $\mu = \nu = \frac{1}{N}\mathbf{1}_N$：批次上的均匀分布，$\mathbf{1}_N$是长度为$N$的全1向量。

  得到最优传输计划后，计算图像到文本的SIGROT损失，衡量最优传输计划与相似图的分布差异：
$$\mathcal{L}_{\mathrm{SIGROT}}^{\mathrm{i2t}} = \mathrm{KL}(N \gamma_{\mathrm{i2t}}^* \| \mathrm{softmax}(G_{\mathrm{cross}}))$$
其中$N \gamma_{\mathrm{i2t}}^*$将传输计划缩放为和为1的概率分布，$\mathrm{softmax}(G_{\mathrm{cross}})$将相似图归一化为概率分布，通过KL散度让模型学到的全局匹配与预计算的语义结构一致。
##### 文本到图像方向的UOT求解
与图像到文本方向类似，仅需转置成本矩阵，得到最优传输计划$\gamma_{\mathrm{t2i}}^*$：
$$
\begin{array} { r l }
& { \gamma _ { \mathrm { t 2 i } } ^ { * } = \underset { \gamma \in \mathbb { R } _ { + } ^ { N \times N } } { \arg \operatorname* { m i n } } \left\langle \gamma , C _ { \mathrm { i m a g e - t e x t } } ^ { \intercal } \right\rangle _ { F } - \varepsilon H ( \gamma ) } \\
& { \qquad + \tau _ { m 1 } \mathrm { K L } ( \gamma \mathbf { 1 } _ { N } \| \nu ) + \tau _ { m 2 } \mathrm { K L } ( \gamma ^ { \intercal } \mathbf { 1 } _ { N } \| \mu ) , }
\end{array}
$$
对应的损失为：
$$\mathcal{L}_{\mathrm{SIGROT}}^{\mathrm{t2i}} = \mathrm{KL}(N \gamma_{\mathrm{t2i}}^* \| \mathrm{softmax}(G_{\mathrm{cross}}))$$
##### 总SIGROT损失
取两个方向损失的平均值：
$$\mathcal{L}_{\mathrm{SIGROT}} = \frac{1}{2} \left( \mathcal{L}_{\mathrm{SIGROT}}^{\mathrm{i2t}} + \mathcal{L}_{\mathrm{SIGROT}}^{\mathrm{t2i}} \right)$$
### 4.2.3. 混合训练目标
将对比损失（CLIP或SigLIP）与SIGROT损失加权结合，得到最终的训练目标：
- CLIP+SIGROT混合损失：
  $$\mathcal { L } _ { \mathrm { C L I P - S I G R O T } } = \lambda \mathcal { L } _ { \mathrm { C L I P } } + \mathcal { L } _ { \mathrm { S I G R O T } }$$
- SigLIP+SIGROT混合损失：
  $$\mathcal { L } _ { \mathrm { S i g L I P - S I G R O T } } = \lambda \mathcal { L } _ { \mathrm { S i g L I P } } + \mathcal { L } _ { \mathrm { S I G R O T } }$$
其中$\lambda \geq 0$为平衡两个损失的超参数。为了稳定训练，引入可学习的温度参数`\tau = \exp(\tau')`（保证为正），SigLIP额外引入可学习的偏置参数$b$。

---

# 5. 实验设置
## 5.1. 数据集
实验采用三个越南语图文数据集，覆盖域内评估、零样本泛化评估场景，为防止训练测试污染，采用SSCD（自监督图像拷贝检测描述符）去除测试集与训练集的近重复图像（余弦相似度≥0.8判定为近重复）。

| 数据集 | 类型 | 规模 | 特点 | 用途 |
|-------|------|------|------|------|
| UIT-OpenViIC | 开放域图文数据集 | 13100张图像、61241个越南语caption，训练集9088张、验证集2011张、测试集2001张 | 包含越南真实复杂场景，语义多样 | 主训练集、域内评估 |
| KTVIC | 越南日常生活场景数据集 | 原始4327张图像、21635个caption，去重后训练集1305张、测试集157张 | 聚焦越南本土日常生活场景 | 零样本评估 |
| Crossmodal-3600（XM3600） | 多语言跨模态数据集 | 3600张图像、36种语言的caption，越南语部分共7350个图文对，无近重复 | 地理分布多样，无领域偏向 | 零样本评估 |

## 5.2. 评估指标
### 5.2.1. Recall@K（R@K，召回率@K）
1.  **概念定义**：衡量检索性能的核心指标，指查询时前K个返回结果中包含正确匹配的查询占总查询的比例，值越高说明检索效果越好。
2.  **数学公式**：
    $$\text{Recall@K} = \frac{\text{前K个结果中包含正确匹配的查询数量}}{\text{总查询数量}}$$
3.  **符号解释**：$K$为返回的Top结果数量，本文采用$K=1,5,10$。
### 5.2.2. Alignment（对齐分数）
1.  **概念定义**：衡量配对图文对嵌入的余弦相似度的平均值，值越高说明配对样本的跨模态对齐效果越好。
2.  **数学公式**：
    $$\mathrm { A l i g n m e n t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { s i m } ( \mathbf { z } _ { i } ^ { \mathrm { i m a g e } } , \mathbf { z } _ { i } ^ { \mathrm { t e x t } } )$$
3.  **符号解释**：$N$为样本数量，$\mathrm{sim}(\cdot, \cdot)$为余弦相似度，$\mathbf{z}_i^{\mathrm{image}}$、$\mathbf{z}_i^{\mathrm{text}}$为配对的图像和文本嵌入。
### 5.2.3. Modality Gap（模态间隙）
1.  **概念定义**：衡量图像嵌入质心与文本嵌入质心的L2距离，值越小说明两个模态的嵌入在共享空间中越接近，模态间隙越小。
2.  **数学公式**：
    $$\Delta _ { \mathrm { g a p } } = \left\| \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf z } _ { i } ^ { \mathrm { i m a g e } } - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf z } _ { i } ^ { \mathrm { t e x t } } \right\|$$
3.  **符号解释**：$\|\cdot\|$为L2范数，两个求和项分别为所有图像嵌入的平均值（质心）、所有文本嵌入的平均值（质心）。
## 5.3. 对比基线
论文设置了三类对比基线，全面验证模型效果：
1.  **零样本多语言VLM基线**：包括mSigLIP-base、Jina CLIP v2、Jina Embedding v4、Qwen3-VL-Embedding-2B，均为公开的大规模多语言多模态预训练模型，用于验证专门训练的越南语模型的优势；
2.  **同架构对比基线**：CLIP基线（与ViCLIP-OT相同骨干，仅用CLIP损失训练）、SigLIP基线（相同骨干，仅用SigLIP损失训练），用于验证SIGROT损失的增益；
3.  **消融基线**：CLIP+UOT（仅加普通不平衡OT损失，无相似图正则）、SigLIP+UOT、仅SIGROT损失，用于验证相似图正则、混合损失的有效性。

    ---

# 6. 实验结果与分析
## 6.1. 核心结果分析
### 6.1.1. 域内评估（UIT-OpenViIC测试集）
以下是原文Table 1的实验结果：

<table>
<thead>
<tr>
<th rowspan="2">Method/Model</th>
<th rowspan="2"># Params</th>
<th colspan="3">Text → Image</th>
<th colspan="3">Image → Text</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td>mSigLIP-base* [46]</td>
<td>370M</td>
<td>14.34</td>
<td>28.94</td>
<td>36.21</td>
<td>20.49</td>
<td>32.23</td>
<td>37.43</td>
<td>28.27</td>
</tr>
<tr>
<td>Jina CLIP v2*</td>
<td>865M</td>
<td>30.01</td>
<td>52.09</td>
<td>61.70</td>
<td>40.23</td>
<td>65.02</td>
<td>74.41</td>
<td>53.91</td>
</tr>
<tr>
<td>Jina Embedding v4* [14]</td>
<td>4B</td>
<td>23.97</td>
<td>42.22</td>
<td>50.29</td>
<td>41.48</td>
<td>66.77</td>
<td>75.61</td>
<td>50.06</td>
</tr>
<tr>
<td>Qwen3-VL-Embedding-2B* [21]</td>
<td>2B</td>
<td>32.13</td>
<td>54.00</td>
<td>62.93</td>
<td>39.83</td>
<td>66.52</td>
<td>77.01</td>
<td>55.40</td>
</tr>
<tr>
<td>CLIP</td>
<td>221M</td>
<td>31.19</td>
<td>59.80</td>
<td>71.23</td>
<td>46.60</td>
<td>75.53</td>
<td>85.19</td>
<td>61.59</td>
</tr>
<tr>
<td>SigLIP</td>
<td>221M</td>
<td>34.75</td>
<td>63.01</td>
<td>72.96</td>
<td>50.10</td>
<td>79.78</td>
<td>88.04</td>
<td>64.77</td>
</tr>
<tr>
<td>CLIP + UOT</td>
<td>221M</td>
<td>29.27</td>
<td>57.62</td>
<td>69.07</td>
<td>43.59</td>
<td>75.03</td>
<td>84.03</td>
<td>59.77</td>
</tr>
<tr>
<td>SigLIP + UOT</td>
<td>221M</td>
<td>37.84</td>
<td>65.30</td>
<td>74.98</td>
<td>53.95</td>
<td>80.95</td>
<td>88.81</td>
<td>66.97</td>
</tr>
<tr>
<td>SIGROT</td>
<td>221M</td>
<td>40.75</td>
<td>70.72</td>
<td>80.90</td>
<td>37.99</td>
<td>61.11</td>
<td>71.68</td>
<td>60.53</td>
</tr>
<tr>
<td>ViCLIP-OT (Eq. 19)</td>
<td>221M</td>
<td>37.57</td>
<td>65.65</td>
<td>75.43</td>
<td>54.35</td>
<td>81.83</td>
<td>89.19</td>
<td>67.34</td>
</tr>
<tr>
<td>ViSigLIP-OT (Eq. 20)</td>
<td>221M</td>
<td>39.19</td>
<td>66.71</td>
<td>76.04</td>
<td>57.21</td>
<td>83.83</td>
<td>90.79</td>
<td>68.96</td>
</tr>
</tbody>
</table>

*注：带*的为零样本评估结果。

结果分析：
1.  ViCLIP-OT平均R@K达67.34%，比CLIP基线高5.75个百分点；ViSigLIP-OT平均R@K达68.96%，比SigLIP基线高4.19个百分点，证明SIGROT损失可有效提升检索效果；
2.  仅221M参数的ViCLIP-OT，比参数大10倍的2B参数Qwen3-VL-Embedding-2B的零样本效果高11.94个百分点，证明专门针对低资源语言训练的模型效果远超大参数通用多语言模型；
3.  SigLIP+UOT比SigLIP基线高2.2个百分点，而ViSigLIP-OT比SigLIP基线高4.19个百分点，证明相似图正则的增益远大于普通OT损失；
4.  仅用SIGROT损失时文本到图像的R@K很高，但图像到文本的效果较差，证明对比损失与SIGROT结合是必要的。

    不同模型的R@K对比如下图（原文Figure 2）所示：

    ![Figure 2: R@K comparison on UIT-OpenViIC for text-to-image (left) and image-to-text (right) retrieval tasks. Incorporating the SIGROT loss consistently improves performance over both CLIP and SigLIP baselines across all R@K metrics.](images/2.jpg)
    *该图像是图表，展示了在UIT-OpenViIC上进行文本至图像（左侧）和图像至文本（右侧）检索任务的召回率对比。图表中显示，结合SIGROT损失后，ViCLIP-OT在各个召回率指标上均优于CLIP和SigLIP基线模型。*

### 6.1.2. 零样本评估
以下是原文Table 2的零样本实验结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Text → Image</th>
<th colspan="3">Image → Text</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8">KTVIC-train</td>
</tr>
<tr>
<td>CLIP</td>
<td>21.12</td>
<td>46.99</td>
<td>59.22</td>
<td>31.65</td>
<td>59.46</td>
<td>72.49</td>
<td>48.49</td>
</tr>
<tr>
<td>SigLIP</td>
<td>23.16</td>
<td>48.78</td>
<td>60.57</td>
<td>35.48</td>
<td>62.22</td>
<td>73.64</td>
<td>50.64</td>
</tr>
<tr>
<td>ViCLIP-OT</td>
<td>26.24</td>
<td>52.46</td>
<td>64.14</td>
<td>38.47</td>
<td>64.37</td>
<td>75.48</td>
<td>53.52</td>
</tr>
<tr>
<td>ViSigLIP-OT</td>
<td>26.28</td>
<td>52.58</td>
<td>63.49</td>
<td>39.62</td>
<td>66.44</td>
<td>77.78</td>
<td>54.37</td>
</tr>
<tr>
<td colspan="8">KTVIC-test</td>
</tr>
<tr>
<td>CLIP</td>
<td>50.32</td>
<td>82.80</td>
<td>89.94</td>
<td>63.06</td>
<td>92.36</td>
<td>97.45</td>
<td>79.32</td>
</tr>
<tr>
<td>SigLIP</td>
<td>52.61</td>
<td>83.31</td>
<td>89.94</td>
<td>71.97</td>
<td>94.27</td>
<td>96.18</td>
<td>81.38</td>
</tr>
<tr>
<td>ViCLIP-OT</td>
<td>56.69</td>
<td>85.61</td>
<td>91.97</td>
<td>70.06</td>
<td>93.63</td>
<td>98.09</td>
<td>82.68</td>
</tr>
<tr>
<td>ViSigLIP-OT</td>
<td>56.56</td>
<td>85.99</td>
<td>91.72</td>
<td>71.34</td>
<td>93.63</td>
<td>97.45</td>
<td>82.78</td>
</tr>
<tr>
<td colspan="8">Crossmodal-3600</td>
</tr>
<tr>
<td>CLIP</td>
<td>22.52</td>
<td>45.55</td>
<td>58.01</td>
<td>26.22</td>
<td>53.42</td>
<td>65.06</td>
<td>45.13</td>
</tr>
<tr>
<td>SigLIP</td>
<td>26.67</td>
<td>50.31</td>
<td>61.78</td>
<td>31.17</td>
<td>57.78</td>
<td>69.83</td>
<td>49.59</td>
</tr>
<tr>
<td>ViCLIP-OT</td>
<td>28.90</td>
<td>55.29</td>
<td>66.37</td>
<td>42.56</td>
<td>68.81</td>
<td>79.17</td>
<td>56.85</td>
</tr>
<tr>
<td>ViSigLIP-OT</td>
<td>32.04</td>
<td>57.90</td>
<td>68.95</td>
<td>37.97</td>
<td>64.64</td>
<td>75.53</td>
<td>56.17</td>
</tr>
</tbody>
</table>

结果分析：
1.  在Crossmodal-3600上，ViCLIP-OT比CLIP基线高11.72个百分点，证明SIGROT可大幅提升模型的零样本泛化能力；
2.  在KTVIC的两个拆分上，ViCLIP-OT/ViSigLIP-OT均比对应基线高1.3-5个百分点，验证了模型在不同领域数据上的泛化性。
### 6.1.3. 嵌入空间分析
嵌入空间的UMAP可视化结果如下图（原文Figure 3）所示，圆圈为图像嵌入、三角形为文本嵌入，颜色为K-Means聚类伪标签：

![Figure 3: UMAP visualization of image and text embeddings on the UIT-OpenViIC test set. Each subplot corresponds to a different training objective. Circles represent image embeddings and triangles represent text embeddings, with colors indicating pseudo labels obtained via K-Means clustering ( $k = 2 0$ ). SIGROT-based methods exhibit tighter crossmodal clustering compared to baselines.](images/3.jpg)
*该图像是UMAP可视化图，展示了UIT-OpenViIC测试集上的图像和文本嵌入。每个子图对应不同的训练目标，圆圈代表图像嵌入，三角形代表文本嵌入，颜色表示通过K-Means聚类获得的伪标签（$k=20$）。与基线方法相比，SIGROT方法展现了更紧密的跨模态聚类。*

对齐分数与模态间隙的量化结果如下（原文Table 3）：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">UIT-OpenViIC</th>
<th colspan="2">KTVIC-test</th>
<th colspan="2">Crossmodal-3600</th>
</tr>
<tr>
<th>A↑</th>
<th>∥∆gap∥ ↓</th>
<th>A ↑</th>
<th>∥∆gap∥ ↓</th>
<th>A↑</th>
<th>∥Δgap| ↓</th>
</tr>
</thead>
<tbody>
<tr>
<td>SIGROT</td>
<td>0.8061</td>
<td>0.1323</td>
<td>0.7670</td>
<td>0.2135</td>
<td>0.6976</td>
<td>0.1625</td>
</tr>
<tr>
<td>CLIP</td>
<td>0.5201</td>
<td>0.1952</td>
<td>0.4696</td>
<td>0.2032</td>
<td>0.5329</td>
<td>0.2558</td>
</tr>
<tr>
<td>ViCLIP-OT</td>
<td>0.6624</td>
<td>0.1026</td>
<td>0.6212</td>
<td>0.1636</td>
<td>0.6225</td>
<td>0.1273</td>
</tr>
<tr>
<td>SigLIP</td>
<td>0.3637</td>
<td>0.5843</td>
<td>0.3182</td>
<td>0.5757</td>
<td>0.3790</td>
<td>0.5789</td>
</tr>
<tr>
<td>ViSigLIP-OT</td>
<td>0.3928</td>
<td>0.3177</td>
<td>0.3373</td>
<td>0.3385</td>
<td>0.4142</td>
<td>0.3442</td>
</tr>
</tbody>
</table>

结果分析：
1.  加入SIGROT后，所有数据集上的对齐分数均提升、模态间隙均下降：SigLIP的模态间隙从0.5843降至0.3177，CLIP的模态间隙从0.1952降至0.1026，证明SIGROT有效增强了跨模态对齐、减少了模态间隙；
2.  仅用SIGROT损失的对齐分数最高，验证了OT对分布对齐的强作用。
### 6.1.4. 可视化分析
GradCAM注意力可视化结果如下图（原文Figure 4）所示，对比了SigLIP与ViSigLIP-OT的注意力区域：

![Figure 4: GradCAM visualization comparing the baseline SigLIP and the proposed ViSigLIP-OT on the UIT-OpenViIC test set. Each row shows the original image alongside the GradCAM heatmaps from both models for a given Vietnamese text query. In the first two rows, ViSigLIP-OT focuses more precisely on the query-relevant objects (the girl wearing an Ao dai and the man holding apples in his hands), while SigLIP spreads activations over background regions. In the third row, SigLIP correctly attends to the man standing next to a car, whereas ViSigLIP-OT highlights irrelevant background areas.](images/4.jpg)
*该图像是GradCAM可视化结果，比较了基线模型SigLIP和提出的ViSigLIP-OT在UIT-OpenViIC测试集上的表现。每行展示了原始图像和针对特定越南文本查询的热力图，其中ViSigLIP-OT在与查询相关的对象上表现更突出，而SigLIP则更多地关注背景区域。*

结果显示，ViSigLIP-OT的注意力更集中于文本查询提到的相关对象（如穿奥黛的女孩、拿苹果的男人），而SigLIP的注意力更分散于背景区域，证明SIGROT让模型更关注语义相关的视觉内容。
## 6.2. 消融实验分析
### 6.2.1. 图像编码器部分微调的影响
实验测试了冻结DINOv3主干、解冻最后$k$个Transformer组的效果，结果如下图（原文Figure 5a）所示：

![Figure 5: Effect of (a) the number of last unfrozen groups in the image encoder using ViCLIP-OT, and (b) the hybrid loss weight $\\lambda$ where $\\lambda = 0$ corresponds to SIGROT only. Peak performance occurs at 13 unfrozen groups $( 6 9 . 6 2 \\% )$ and $\\lambda = 0 . 2$ for ViCLIP-OT, $\\lambda = 0 . 1$ for ViSigLIP-OT.](images/5.jpg)
*该图像是图表，展示了使用 ViCLIP-OT 模型的 (a) 最后未冻结组数对平均 Recall@K 的影响，以及 (b) 混合损失权重 `heta` 对模型性能的影响。其中，最高性能出现在 13 个未冻结组时为 69.62%。*

解冻最后13个组时平均R@K最高，达69.62%；解冻所有层（14个组）时效果略有下降，原因是过拟合或预训练特征被破坏；仅解冻最后2个组就有近7个百分点的提升，说明适当微调视觉主干可有效适配越南语场景的图像特征。
### 6.2.2. 混合损失权重$\lambda$的影响
测试了不同$\lambda$值的效果，如上图（原文Figure 5b）所示：
- ViCLIP-OT在$\lambda=0.2$时效果最优，平均R@K达69.20%；
- ViSigLIP-OT在$\lambda=0.1$时效果最优，平均R@K达70.76%；
- $\lambda$过大时对比损失占比过高，掩盖了SIGROT的增益；$\lambda=0$仅用SIGROT时效果不如混合损失，证明两个损失的互补性。
### 6.2.3. 相似图组合策略的影响
测试了不同相似图构建策略的效果，结果如下图（原文Figure 6）所示：

![Figure 6: Average Recall@K for different similarity graph combination strategies. The cross-modality approach achieves the highest performance for both loss configurations.](images/6.jpg)
*该图像是一个图表，展示了不同相似性图组合策略下的平均 Recall@K 结果。ViCLIP-OT 在所有组合策略中，特别是在跨模态策略上表现最佳。*

跨模态组合策略（平均四个相似矩阵）的效果最优，比仅用单模态相似性、双模态算术/调和平均的效果更好，证明同时融合内模态、跨模态相似信息可有效提升SIGROT的效果。
### 6.2.4. 骨干网络泛化性验证
附录实验显示，更换不同的文本骨干（EmbeddingGemma-300M、BGE-M3）、图像骨干（ConvNeXt-base）时，SIGROT均能带来一致的性能提升，证明SIGROT的效果不依赖特定骨干网络，具有强泛化性。

---

# 7. 总结与思考
## 7.1. 结论总结
本文首次提出了面向越南语的图文检索基础视觉语言模型ViCLIP-OT，通过融合CLIP风格对比学习与提出的SIGROT损失，同时实现实例级对齐与全局语义结构保留，有效减少了模态间隙。在三个越南语基准数据集上的实验表明，ViCLIP-OT在域内和零样本场景下均取得SOTA性能，小参数模型效果远超大规模通用多语言模型，证明了该方案在低资源语言跨模态检索场景下的有效性与可扩展性，为越南语智能多媒体系统的落地提供了技术支撑。
## 7.2. 局限性与未来工作
### 局限性
1.  训练数据集规模较小（UIT-OpenViIC仅1.3万张图像），限制了模型的上限；
2.  相似图采用预训练多模态模型预计算，是固定的静态图，无法随训练过程动态调整；
3.  目前仅验证了图文检索任务，未扩展到其他多模态任务。
### 未来工作
1.  构建更大规模的越南语图文数据集，开展大规模预训练；
2.  探索端到端的动态相似图学习，进一步提升对齐效果；
3.  将ViCLIP-OT扩展到视觉问答、多模态推理等其他越南语多模态任务；
4.  验证该方案在其他低资源语言上的迁移效果。
## 7.3. 个人启发与批判
### 启发
1.  低资源语言场景下，专门针对目标语言的小规模适配训练，效果远超大参数通用多语言模型，为其他低资源语言的多模态模型研发提供了参考范式；
2.  SIGROT的思路可迁移到其他跨模态任务（如音频-文本检索、视频-文本检索），利用样本间的语义结构做全局正则，提升对齐效果与泛化能力。
### 潜在问题与改进方向
1.  相似图的质量高度依赖预训练多模态模型在目标低资源语言上的效果，若预训练模型对目标语言的支持差，会影响SIGROT的性能，未来可探索自适应的相似图构建方法；
2.  静态相似图无法适配训练过程中模型表示的动态变化，端到端学习相似图可进一步提升效果；
3.  当前模型参数规模较小，若采用更大的骨干网络、更大规模的训练数据，性能有望进一步提升。