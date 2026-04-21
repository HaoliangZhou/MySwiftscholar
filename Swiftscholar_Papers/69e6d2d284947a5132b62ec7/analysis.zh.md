# 1. 论文基本信息

## 1.1. 标题
**FAM: Fine-Grained Alignment Matters in Multimodal Embedding Learning with Large Vision-Language Models** (FAM：细粒度对齐在大型视觉-语言模型多模态嵌入学习中的重要性)

## 1.2. 作者
<strong>Tianhang Xiang (向天航)</strong>, <strong>Yirui Li (李艺睿)</strong>, <strong>Lizhao Liu (刘礼钊)</strong>, <strong>Hongyan Zhi (智红燕)</strong>, <strong>Chuanshen Chen (陈传深)</strong>, <strong>Qing Du (杜青)</strong>, <strong>Mingkui Tan (谭明奎)</strong>

**研究背景与隶属机构：**
*   <strong>South China University of Technology (华南理工大学)</strong>：主要作者单位，位于中国广州。
*   <strong>Peng Cheng Laboratory (鹏城实验室)</strong>：位于中国深圳的国家级实验室。
*   <strong>Tencent AI Lab (腾讯人工智能实验室)</strong>：位于中国深圳的企业研究机构。

## 1.3. 发表期刊/会议
根据文中引用的 "Jiang et al. 2025. In ICLR" 以及参考文献中包含的 2024、2025 年预印本，该论文目前处于**预印本** 阶段，极有可能投稿或被接收至顶级人工智能会议如 **ICLR (International Conference on Learning Representations)**。

## 1.4. 发表年份
2025年 (基于文中引用的自引文 "Bai et al. 2025" 和 "Jiang et al. 2025" 推断)。

## 1.5. 摘要
该论文旨在解决直接将生成式大型视觉-语言模型 适配为多模态嵌入模型 时存在的两个关键问题：**视觉表示不足** 和 **多模态对齐粗糙**。为了解决这些问题，作者提出了名为 **FAM (Fine-grained Alignment Matters)** 的方法。该方法包含两个核心组件：
1.  **多粒度对齐对比**：通过从粗粒度到细粒度的多个对比损失，显式地学习并对齐图像-文本对在多个粒度上的模态表示。
2.  **视觉嵌入反演**：一种训练策略，通过随机掩码部分视觉特征并利用嵌入向量指导其重建，鼓励提取的嵌入保留细粒度的视觉特征。
    在广泛的下游多模态数据集上的实验表明，该方法取得了优越的性能。

## 1.6. 原文链接
论文 PDF 链接：`/files/papers/69e6d2d284947a5132b62ec7/paper.pdf`
代码库：https://github.com/TianhangXiang/FAM

---

# 2. 整体概括

## 2.1. 研究背景与动机
**核心问题：**
随着大型视觉-语言模型（如 GPT-4V, LLaVA, Qwen-VL）在生成任务上的巨大成功，研究人员开始尝试将这些强大的模型迁移到多模态嵌入学习任务中（如检索、分类）。然而，论文指出，直接将生成式 LVLMs 适配为嵌入模型存在两大缺陷：
1.  **视觉表示不足**：LVLMs 通常训练用于生成高语义的文本，这导致模型偏向语言模态，而忽略了细粒度的视觉细节（例如图像中的局部物体、颜色、纹理等）。在嵌入学习中，这种偏差会导致模型无法区分视觉上相似但语义不同的图像。
2.  **多模态对齐粗糙**：现有的适配方法（如 VLM2Vec）主要关注全局的图像-文本对齐，缺乏对图像局部区域与文本具体词汇之间细粒度对应关系的建模。

**为什么重要：**
多模态嵌入是连接视觉检索、视觉问答 (VQA)、检索增强生成 (RAG) 等下游应用的基础。如果嵌入模型无法捕捉细粒度的视觉特征或无法精确对齐跨模态的细节，那么在需要精确匹配的任务（如“找到一只红色的左脚穿着运动鞋的狗”）中，模型的性能将受到严重限制。

**切入点：**
作者通过实验观察（Figure 1），证实了现有的 VLM2Vec 方法在全局-局部图像相似度和细粒度图像-文本相似度评估上表现不佳。基于此，FAM 提出了“先对齐，后适配” 的训练范式，通过引入细粒度的对比损失和视觉重建机制来弥补上述缺陷。

## 2.2. 核心贡献/主要发现
1.  **问题识别**：首次明确指出并分析了直接将生成式 LVLMs 适配为嵌入模型时存在的“视觉表示不足”和“模态对齐粗糙”问题。
2.  **方法创新 - MAC**：提出了**多粒度对齐对比**，通过设计粗粒度、粗到细、细粒度三个层次的对比损失，强制模型在不同抽象层次上对齐视觉和语言特征。
3.  **方法创新 - VEIN**：提出了**视觉嵌入反演**训练策略，通过掩码重建任务，显式地约束嵌入向量必须包含足够的信息以恢复被掩码的视觉特征，从而增强视觉表示的鲁棒性。
4.  **性能提升**：在 MMEB 基准测试中，FAM 在多个主干网络（Qwen2-VL, Phi-3.5-V）上均超越了现有的最先进方法（SOTA），特别是在多模态检索和视觉定位 任务上提升显著。

    ---

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，读者需要熟悉以下核心概念：

*   **多模态嵌入**：将来自不同模态（如图像、文本）的数据映射到一个共同的向量空间中。在这个空间里，语义相关的数据（如一张狗的图片和“一只狗”的文字）距离较近，不相关的数据距离较远。
*   **对比学习**：一种自监督学习方法，通过拉近正样本对 的距离，推远负样本对 的距离来学习特征表示。本文中广泛使用的 InfoNCE 损失是其典型代表。
*   <strong>大型视觉-语言模型 (LVLM)</strong>：结合了视觉编码器和大语言模型 (LLM) 的模型，通常通过投影层将视觉特征对齐到 LLM 的词表空间，从而实现多模态理解和生成。
*   **指令微调**：通过自然语言指令来引导模型完成特定任务的技术。本文在输入中加入指令（如 "Represent the given image."），使模型能适应不同的下游任务格式。
*   **LoRA (Low-Rank Adaptation)**：一种参数高效微调技术，通过冻结预训练模型权重并在旁路注入低秩矩阵来更新模型，常用于微调大型模型以节省显存。

## 3.2. 前人工作
作者在文中主要讨论了以下几类相关工作：

1.  **双编码器架构**：
    *   代表工作：**CLIP** (Radford et al. 2021), **ALIGN** (Jia et al. 2021)。
    *   原理：使用独立的视觉编码器和文本编码器，通过对比学习对齐全局特征。
    *   局限性：无法处理交错的图文输入，且缺乏复杂推理能力。

2.  **基于 LVLM 的嵌入模型**：
    *   代表工作：**VLM2Vec** (Jiang et al. 2025), **E5-V** (Jiang et al. 2024), **LLaVE** (Lan et al. 2025)。
    *   原理：利用 LVLM 强大的多模态理解能力，通过对比学习将其转化为嵌入模型。
    *   局限性：这些方法通常只进行单阶段的指令微调，往往导致模态对齐不够精细，且容易丢失细粒度的视觉细节。

## 3.3. 技术演进
该领域经历了从**双编码器**（如 CLIP）到**基于生成式 LVLM 的嵌入模型**（如 VLM2Vec）的演变。早期方法虽然高效但语义理解能力有限；后期方法利用了 LVLM 强大的推理能力，但本文发现这种迁移并非完美，引入了新的表示退化问题。

## 3.4. 差异化分析
本文方法与 VLM2Vec 等现有工作的核心区别在于：
*   **对齐粒度**：VLM2Vec 主要关注全局嵌入的对齐。FAM 引入了 MAC，在 Token 级别（细粒度）和 Embedding-to-Token 级别（粗到细）进行显式对齐。
*   **视觉增强**：VLM2Vec 仅依赖任务对比损失。FAM 引入了 VEIN，这是一种类似于 MAE (Masked Autoencoders) 的特征重建机制，作为正则化项来强制保留视觉细节。

    ---

# 4. 方法论

本章将深入解析 FAM 的技术细节。FAM 采用“先对齐，后适配” 的两阶段训练范式。

## 4.1. 方法原理
FAM 的核心思想是：生成式 LVLM 原本是为了生成文本而训练的，直接用于提取嵌入会导致视觉信息压缩过度。为了解决这个问题，FAM 在第一阶段通过 **MAC** 强迫模型在不同粒度上对齐视觉和语言特征；在第二阶段通过 **VEIN** 强迫模型在提取嵌入时保留足够的视觉信息以支持特征重建。

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 符号定义与模型输入
首先，定义输入格式。为了兼容 LVLM，输入遵循指令格式：
$$ x_{\mathrm{ins}} = [ \mathrm{V} ; \mathrm{Instruction} ; \{ \mathrm{instruction} \} ; \mathrm{Text} ; < \mathrm{EOS} > ] $$
其中 $\mathrm{V}$ 表示图像，$<\mathrm{EOS}>$ 是序列结束符。模型 $\mathcal{M}$ 输出隐藏状态 $\mathbf{H}^l$，最终的嵌入向量取自 $<\mathrm{EOS}>$ token 在最后一层的状态：
$$ \mathbf{e}_{\mathrm{ins}} = \mathcal{M}^L(x_{\mathrm{ins}})[<\mathrm{EOS}>] $$
我们还可以通过索引获取中间层 $l$ 的视觉和文本 token 状态：
$$ \mathbf{H}_{\mathrm{img}}^l = \mathbf{H}^l[\mathrm{VIS\_INDEX}], \quad \mathbf{H}_{\mathrm{txt}}^l = \mathbf{H}^l[\mathrm{TXT\_INDEX}] $$

### 4.2.2. 阶段一：多粒度对齐对比 (MAC)
在此阶段，目标是对齐图像-文本对的表示。给定一批 $B$ 个图像-文本对 $\{ p^b = (\mathrm{img}^b, \mathrm{txt}^b) \}_{b=1}^B$，模型分别处理图像和文本，得到全局嵌入 $\{ \mathbf{e}_{\mathrm{img}}^b, \mathbf{e}_{\mathrm{txt}}^b \}$ 以及 token 级特征 $\{ \mathbf{H}_{\mathrm{img}}^b \in \mathbb{R}^{N \times d}, \mathbf{H}_{\mathrm{txt}}^b \in \mathbb{R}^{T \times d} \}$。

MAC 包含三个层次的对比损失：

**1. 粗粒度对比对齐**
这是最基础的实例级对齐，类似于 CLIP 的损失，拉近同一对图像和文本的全局嵌入距离。
$$ \mathcal{L}_{c} = - \frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp \left( g( \mathbf{e}_{\mathrm{img}}^b , \mathbf{e}_{\mathrm{txt}}^b ) \right) }{ \sum_{b'=1}^{B} \exp \left( g( \mathbf{e}_{\mathrm{img}}^b , \mathbf{e}_{\mathrm{txt}}^{b'} ) \right) } $$
其中，$g(\mathbf{u}, \mathbf{v})$ 是带温度的余弦相似度函数：
$$ g(\mathbf{u}, \mathbf{v}) = \frac{1}{\tau} \frac{\mathbf{u}^{\top} \mathbf{v}}{\| \mathbf{u} \|_2 \| \mathbf{v} \|_2} $$
*解释：这里 $\tau$ 是温度参数，分母中的求和表示将当前样本与批次内所有其他样本（包括负样本）进行对比。*

**2. 粗到细对比对齐**
为了捕捉实例级之外的对应关系，我们将一个模态的全局嵌入与另一个模态的 token 级特征进行对齐。
首先计算图像嵌入 $\mathbf{e}_{\mathrm{img}}^i$ 与第 $j$ 个样本的文本 token 特征 $\mathbf{H}_{\mathrm{txt}}^j$ 之间的平均相似度：
$$ \mathbf{s}_{\mathrm{I2T}}^{i, j} = \frac{1}{T} \sum_{t=1}^{T} g \left( \mathbf{e}_{\mathrm{img}}^i , \mathbf{H}_{\mathrm{txt}}^j [t] \right) $$
对称地，计算文本嵌入与图像 patch 特征的相似度：
$$ \mathbf{s}_{\mathrm{T2I}}^{i, j} = \frac{1}{N} \sum_{n=1}^{N} g \left( \mathbf{e}_{\mathrm{txt}}^i , \mathbf{H}_{\mathrm{img}}^j [n] \right) $$
基于这两个相似度分数，定义两个方向的对比损失：
$$ \mathcal{L}_{c2f}^{\mathrm{I2T}} = - \frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp \left( \mathbf{s}_{\mathrm{I2T}}^{b, b} \right) }{ \sum_{b'=1}^{B} \exp \left( \mathbf{s}_{\mathrm{I2T}}^{b, b'} \right) } $$
$$ \mathcal{L}_{c2f}^{\mathrm{T2I}} = - \frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp \left( \mathbf{s}_{\mathrm{T2I}}^{b, b} \right) }{ \sum_{b'=1}^{B} \exp \left( \mathbf{s}_{\mathrm{T2I}}^{b, b'} \right) } $$
总的粗到细损失为两者的平均：
$$ \mathcal{L}_{c2f} = \frac{1}{2} \left( \mathcal{L}_{c2f}^{\mathrm{I2T}} + \mathcal{L}_{c2f}^{\mathrm{T2I}} \right) $$
*解释：这一步强迫图像的全局语义去匹配文本中的每一个词，反之亦然，从而建立跨模态的细粒度关联。*

**3. 细粒度对比对齐**
进一步深入到 token-to-patch 级别的对齐。计算图像 patch $n$ 和文本 token $t$ 之间的相似度矩阵：
$$ \mathbf{S}_{n, t}^{i, j} = g \left( \mathbf{H}_{\mathrm{img}}^i [n] , \mathbf{H}_{\mathrm{txt}}^j [t] \right) $$
对每一对图像-文本样本，聚合所有 patch-token 对的平均相似度：
$$ \mathbf{s}_{\mathrm{f}}^{i, j} = \frac{1}{NT} \sum_{n=1}^{N} \sum_{t=1}^{T} \mathbf{S}_{n, t}^{i, j} $$
细粒度对比损失定义为：
$$ \mathcal{L}_{\mathrm{f}} = - \frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp \left( \mathbf{s}_{\mathrm{f}}^{b, b} \right) }{ \sum_{b'=1}^{B} \exp \left( \mathbf{s}_{\mathrm{f}}^{b, b'} \right) } $$
*解释：这是最细粒度的监督，直接对齐视觉特征图中的每一个 patch 和文本序列中的每一个 token。*

### 4.2.3. 阶段二：视觉增强适配
在此阶段，模型在第一阶段对齐的基础上，适配到下游任务。

**1. 下游任务适配**
对于查询-候选对 $\{ (q, c) \}$，将其转换为指令输入并提取嵌入 $\mathbf{e}_q, \mathbf{e}_c$。应用标准的对比损失：
$$ \mathcal{L}_{\mathrm{task}} = - \frac{1}{B} \sum_{k=1}^{B} \log \frac{\exp \left( g( \mathbf{e}_q^k , \mathbf{e}_{c^+}^k ) \right) }{ \sum_{j=1}^{B} \exp \left( g( \mathbf{e}_q^k , \mathbf{e}_c^j ) \right) } $$

<strong>2. 视觉嵌入反演训练 (VEIN)</strong>
这是 FAM 的核心创新之一。其直觉是：如果提取的嵌入 $\mathbf{e}^l$ 包含了足够的视觉信息，那么它应该能够用来重建被掩码的视觉特征。

*   **步骤 1：掩码**。在指定层 $l$ 提取视觉 token 状态 $\mathbf{H}_{\mathrm{img}}^l$ 和嵌入 $\mathbf{e}^l$。随机掩码比例为 $\gamma$ 的视觉 token。掩码后的序列为：
    $$ \mathbf{H}_{\mathrm{img, mask}}^l [j] = \begin{cases} \mathbf{m}^l, & \text{if } \mathbf{M}_j = 1 \\ \mathbf{H}_{\mathrm{img}}^l [j], & \text{otherwise} \end{cases} $$
    其中 $\mathbf{m}^l$ 是可学习的掩码 token，$\mathbf{M}$ 是二进制掩码向量。

*   **步骤 2：解码重建**。将嵌入 $\mathbf{e}^l$ 与掩码后的视觉 token 拼接，加入位置编码，输入到一个 Transformer 解码器层 $\mathcal{D}$ 中。解码器利用未掩码的 token 作为 key 和 value (通过 Cross-Attention) 来重建掩码部分。
    $$ \mathbf{X}^l = \operatorname{concat} ( \mathbf{e}^l , \mathbf{H}_{\mathrm{img, mask}}^l ) $$
    $$ \mathbf{Z}^l = \mathrm{SelfAttn} ( \mathbf{X}^l_{\mathrm{pos}} ) $$
    $$ \mathbf{Z}_{\mathrm{ca}}^l = \mathrm{CrossAttn} ( \mathbf{Z}^l , \mathbf{H}_{\mathrm{img, unmask}}^l ) $$
    $$ \hat{\mathbf{H}}_{\mathrm{img}}^l = \mathrm{FFN} ( \mathbf{Z}_{\mathrm{ca}}^l ) $$

*   **步骤 3：重建损失**。计算重建特征 $\hat{\mathbf{H}}_{\mathrm{img}}^l$ 与原始特征 $\mathbf{H}_{\mathrm{img}}^l$ 在掩码位置上的余弦相似度损失：
    $$ \mathcal{L}_{\mathrm{rec}} = \frac{1}{|\mathbf{M}|} \sum_{j : \mathbb{I}(\mathbf{M}_j = 1)} \left[ 1 - \cos \left( \hat{\mathbf{H}}_{\mathrm{img}}^l [j] , \mathbf{H}_{\mathrm{img}}^l [j] \right) \right] $$
    *解释：这个损失仅作用于被掩码的 token，强迫模型利用全局嵌入 $\mathbf{e}^l$ 中的信息去恢复局部细节，从而确保 $\mathbf{e}^l$ 不会丢失视觉信息。*

下图（原文 Figure 2）展示了 FAM 的整体架构，包括 MAC 对齐阶段和 VEIN 适配阶段：

![该图像是示意图，展示了多模态对齐（MAC）和下游适应的过程，包括相似性矩阵、图像与文本嵌入的关系以及任务适应的结构。图中涉及多个层次的视觉语言模型和特征重建方法。](images/2.jpg)
*该图像是示意图，展示了多模态对齐（MAC）和下游适应的过程，包括相似性矩阵、图像与文本嵌入的关系以及任务适应的结构。图中涉及多个层次的视觉语言模型和特征重建方法。*

### 4.2.4. 总体优化目标
*   **对齐阶段总损失**：
    $$ \mathcal{L}_{\mathrm{align}} = \mathcal{L}_{c} + \mathcal{L}_{c2f} + \mathcal{L}_{f} $$
*   **适配阶段总损失**：
    $$ \mathcal{L}_{\mathrm{adapt}} = \mathcal{L}_{\mathrm{task}} + \mathcal{L}_{\mathrm{rec}} $$

---

# 5. 实验设置

## 5.1. 数据集
实验分为两个阶段，使用了不同的数据集：
1.  **对齐阶段**：使用 **LLAVA-595K** 数据集。这是一个大规模的图像-文本对数据集，文本描述由 BLIP-2 模型精炼生成。用于训练 MAC 模块。
2.  **适配阶段**：使用 **MMEB (Massive Multimodal Embedding Benchmark)** 数据集。包含 20 个分布内 数据集，涵盖分类、视觉问答 (VQA)、多模态检索 和视觉定位 (VG) 四种元任务。

## 5.2. 评估指标
论文主要使用 **Precision@1 (P@1)** 作为评估指标。

*   **概念定义**：Precision@1 衡量的是在检索任务中，系统返回的第一个结果（即相似度最高的候选项）是正确答案的概率。对于分类任务，它通常指模型预测的 top-1 类别是否正确。它是衡量模型排序能力和检索准确性的最直观指标。
*   **数学公式**：
    $$ P@1 = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{rank}(q_i, \text{correct\_answer}_i) = 1) $$
*   **符号解释**：
    *   $N$：查询样本的总数。
    *   $\text{rank}(q_i, \text{correct\_answer}_i)$：正确答案在第 $i$ 个查询的检索结果列表中的排名。
    *   $\mathbb{I}(\cdot)$：指示函数，如果条件为真则取 1，否则取 0。

## 5.3. 对比基线
论文将 FAM 与以下几类基线模型进行了比较：
1.  **双编码器模型**：CLIP, BLIP2, SigLIP, OpenCLIP, UniIR 等。这些是传统的多模态检索方法。
2.  **LVLM 嵌入模型**：E5-V, VLM2Vec。这些是近期提出的将 LVLM 转化为嵌入模型的方法，其中 VLM2Vec 是本文的主要对比对象。

    ---

# 6. 实验结果与分析

## 6.1. 核心结果分析
以下是原文 Table 1 的结果，展示了 FAM 在 MMEB 基准测试上的性能：

以下是原文 [Table 1] 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Backbone</th>
<th colspan="4">Per Meta-Task Score</th>
<th colspan="3">Average Score</th>
</tr>
<tr>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>VG</th>
<th>IND</th>
<th>OOD</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9"><strong>Dual-Encoder-based Models (Zero-shot)</strong></td>
</tr>
<tr>
<td>CLIP (Radford et al. 2021)</td>
<td></td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.1</td>
<td>38.7</td>
<td>37.8</td>
</tr>
<tr>
<td>BLIP2 (Li et al. 2023)</td>
<td></td>
<td>27.0</td>
<td>4.2</td>
<td>33.9</td>
<td>47.0</td>
<td>25.3</td>
<td>25.1</td>
<td>25.2</td>
</tr>
<tr>
<td>SigLIP (Zhai et al. 2023)</td>
<td></td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>32.3</td>
<td>38.0</td>
<td>34.8</td>
</tr>
<tr>
<td>OpenCLIP (Cherti et al. 2023)</td>
<td></td>
<td>47.8</td>

      < <td>10.9</td>
      <td>52.3</td>
      <td>53.3</td>
      <td>39.3</td>
      <td>40.2</td>
      <td>39.7</td>
    </tr>
    <tr>
      <td>UniR (BLIP_FF) (Wei et al. 2024)</td>
      <td></td>
      <td>42.1</td>
      <td>15.0</td>
      <td>60.1</td>
      <td>62.2</td>
      <td>44.7</td>
      <td>40.4</td>
      <td>42.8</td>
    </tr>
    <tr>
      <td>UniR (CLIP_SF) (Wei et al. 2024)</td>
      <td></td>
      <td>44.3</td>
      <td>16.2</td>
      <td>61.8</td>
      <td>65.3</td>
      <td>47.1</td>
      <td>41.7</td>
      <td>44.7</td>
    </tr>
    <tr>
      <td>MagicLens (Zhang et al. 2024)</td>
      <td></td>
      <td>38.8</td>
      <td>8.3</td>
      <td>35.4</td>
      <td>26.0</td>
      <td>31.0</td>
      <td>23.7</td>
      <td>27.8</td>
    </tr>
    <tr>
      <td colspan="9"><strong>Dual-Encoder-based Models (Fine-tuning on MMEB Training)</strong></td>
    </tr>
    <tr>
      <td>CLIP-FFT (Radford et al. 2021)</td>
      <td></td>
      <td>55.2</td>
      <td>19.7</td>
      <td>53.2</td>
      <td>62.2</td>
      <td>47.6</td>
      <td>42.8</td>
      <td>45.4</td>
    </tr>
    <tr>
      <td>OpenCLIP-FFT (Cherti et al. 2023)</td>
      <td></td>
      <td>56.0</td>
      <td>21.9</td>
      <td>55.4</td>
      <td>64.1</td>
      <td>50.5</td>
      <td>43.1</td>
      <td>47.2</td>
    </tr>
    <tr>
      <td colspan="9"><strong>LVLM-based Models</strong></td>
    </tr>
    <tr>
      <td>E5-V (Jiang et al. 2024)</td>
      <td>LLaVA-NeXT-8B</td>
      <td>21.8</td>
      <td>4.9</td>
      <td>11.5</td>
      <td>19.0</td>
      <td>14.9</td>
      <td>11.5</td>
      <td>13.3</td>
    </tr>
    <tr>
      <td>VLM2Vec* (Jiang et al. 2025)</td>
      <td>Qwen-2VL-2B</td>
      <td>57.8</td>
      <td>40.9</td>
      <td>60.6</td>
      <td>67.1</td>
      <td>59.8</td>
      <td>49.2</td>
      <td>55.0</td>
    </tr>
    <tr>
      <td>FAM (Ours)</td>
      <td>Qwen-2VL-2B</td>
      <td><strong>58.6</strong></td>
      <td><strong>42.2</strong></td>
      <td><strong>64.1</strong></td>
      <td><strong>70.9</strong></td>
      <td><strong>61.7</strong></td>
      <td><strong>51.8</strong></td>
      <td><strong>57.5</strong></td>
    </tr>
    <tr>
      <td>VLM2Vec* (Jiang et al. 2025)</td>
      <td>Phi-3.5-V</td>
      <td>52.7</td>
      <td>50.8</td>
      <td>59.3</td>
      <td>81.0</td>
      <td>63.2</td>
      <td>50.4</td>
      <td>57.3</td>
    </tr>
    <tr>
      <td>FAM (Ours)</td>
      <td>Phi-3.5-V</td>
      <td>53.8</td>
      <td>50.9</td>
      <td>61.2</td>
      <td>83.1</td>
      <td>64.8</td>
      <td>51.1</td>
      <td>58.7</td>
    </tr>
    <tr>
      <td>VLM2Vec* (Jiang et al. 2025)</td>
      <td>Qwen-2VL-7B</td>
      <td>61.3</td>
      <td>47.8</td>
      <td>65.5</td>
      <td>75.8</td>
      <td>65.6</td>
      <td>54.2</td>
      <td>60.5</td>
    </tr>
    <tr>
      <td>FAM (Ours)</td>
      <td>Qwen-2VL-7B</td>
      <td><strong>62.1</strong></td>
      <td>47.5</td>
      <td><strong>68.0</strong></td>
      <td><strong>78.9</strong></td>
      <td><strong>66.5</strong></td>
      <td><strong>56.1</strong></td>
      <td><strong>61.9</strong></td>
    </tr>
  </tbody>
</table>

**分析：**
1.  **LVLM 优势明显**：基于 LVLM 的方法（VLM2Vec, FAM）在各项指标上均显著优于传统的双编码器模型（如 CLIP, OpenCLIP），证明了利用大模型进行嵌入学习的有效性。
2.  **FAM 的提升**：在相同的主干网络下，FAM 均优于 VLM2Vec。
    *   在 **Qwen-2VL-2B** 上，FAM 将总体平均分从 55.0 提升至 57.5 (+2.5)。
    *   在 **Phi-3.5-V** 上，FAM 将总体平均分从 57.3 提升至 58.7 (+1.4)。
    *   在 **Qwen-2VL-7B** 上，FAM 将总体平均分从 60.5 提升至 61.9 (+1.4)。
3.  **特定任务增益**：FAM 在 <strong>Retrieval (检索)</strong> 和 <strong>Visual Grounding (视觉定位)</strong> 任务上的提升尤为明显。这直接验证了 MAC 和 VEIN 在增强细粒度对齐和视觉细节保留方面的有效性，因为这两个任务对细节最敏感。

## 6.2. 消融实验/参数分析

### 6.2.1. 组件有效性消融
以下是原文 [Table 2] 的消融实验结果，展示了 MAC 和 VEIN 各自的贡献：

以下是原文 [Table 2] 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">MAC</th>
<th rowspan="2">VEIN</th>
<th colspan="4">Meta-Task Avg. Score</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>Cls.</th>
<th>VQA</th>
<th>Re.</th>
<th>VG</th>
</tr>
</thead>
<tbody>
<tr>
<td>VLM2Vec</td>
<td>-</td>
<td>-</td>
<td>57.8</td>
<td>40.9</td>
<td>60.6</td>
<td>67.1</td>
<td>55.0</td>
</tr>
<tr>
<td>FAM (Ours)</td>
<td>✓</td>
<td>X</td>
<td>58.5</td>
<td>39.8</td>
<td>64.0</td>
<td>70.3</td>
<td>56.4</td>
</tr>
<tr>
<td>FAM (Ours)</td>
<td>X</td>
<td>✓</td>
<td>58.4</td>
<td>42.9</td>
<td>60.2</td>
<td>69.2</td>
<td>55.9</td>
</tr>
<tr>
<td>FAM (Ours)</td>
<td>✓</td>
<td>✓</td>
<td>58.6</td>
<td>42.2</td>
<td>64.1</td>
<td>70.9</td>
<td>57.3</td>
</tr>
</tbody>
</table>

**分析：**
*   单独使用 MAC 或 VEIN 都能带来性能提升（MAC: +1.4, VEIN: +0.9）。
*   MAC 主要提升了检索 任务 (60.6 -> 64.0)，这得益于多粒度的对齐。
*   VEIN 主要提升了 VQA 任务 (40.9 -> 42.9)，这得益于视觉特征的增强。
*   结合两者达到了最佳性能 (57.3)，证明了互补性。

### 6.2.2. 训练策略分析
以下是原文 [Table 3] 的结果，验证了提升不仅仅是由于数据量增加：

以下是原文 [Table 3] 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Data</th>
<th rowspan="2">MAC</th>
<th colspan="4">Meta-Task Avg. Score</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>Cls.</th>
<th>VQA</th>
<th>Re.</th>
<th>VG</th>
</tr>
</thead>
<tbody>
<tr>
<td>VLM2Vec</td>
<td>D.</td>
<td>✗</td>
<td>57.8</td>
<td>40.9</td>
<td>60.6</td>
<td>67.1</td>
<td>55.0</td>
</tr>
<tr>
<td>VLM2Vec</td>
<td>A.+D.</td>
<td>✗</td>
<td>58.5</td>
<td>41.6</td>
<td>59.6</td>
<td>68.9</td>
<td>55.3</td>
</tr>
<tr>
<td>FAM (Ours)</td>
<td>A.+D.</td>
<td>✓</td>
<td>58.5</td>
<td>39.8</td>
<td>64.0</td>
<td>70.3</td>
<td>56.4</td>
</tr>
</tbody>
</table>

**分析：**
*   "D." 代表下游任务数据，"A." 代表对齐数据。
*   将对齐数据 (A.) 简单加入 VLM2Vec 的训练集仅带来微小提升 (+0.3)。
*   使用 FAM 的策略（即利用 MAC 机制利用这些数据）则带来了显著提升 (+1.4)。这说明 FAM 的收益主要来自于**更有效的训练策略**而非单纯的数据增加。

### 6.2.3. VEIN 掩码比例分析
下图（原文 Figure 4）展示了 VEIN 中不同掩码比例 $\gamma$ 对性能的影响：

![Figure 4: Effect of mask ratios. A moderate ratio (e.g., 0.3) yields the best accuracy.](images/4.jpg)

**分析：**
*   性能随着掩码比例增加先上升，在 **0.3** 左右达到峰值。
*   当比例过高（>0.8）时，性能甚至低于基线。
*   这与 MAE (像素级重建) 不同，MAE 通常在高掩码率（如 75%）下表现更好。作者解释这是因为 VEIN 处理的是**特征级** 重建，特征更具语义和结构，过度的掩码会破坏这种结构，导致无法有效重建。

### 6.2.4. VEIN 插入位置分析
以下是原文 [Table 5] 的结果，展示了在 LVLM 不同层插入 VEIN 的影响：

以下是原文 [Table 5] 的结果：

<table>
<thead>
<tr>
<th>VEIN Conduct Position</th>
<th>Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>N/A</td>
<td>55.0</td>
</tr>
<tr>
<td>All Layers</td>
<td>54.2</td>
</tr>
<tr>
<td>Last Layer</td>
<td>55.3</td>
</tr>
<tr>
<td>Second to Last Layer</td>
<td>55.6</td>
</tr>
<tr>
<td>Third to Last Layer</td>
<td>55.2</td>
</tr>
<tr>
<td>Last Three Layers</td>
<td><strong>55.9</strong></td>
</tr>
</tbody>
</table>

**分析：**
*   在所有层插入 VEIN 反而有害 (54.2 < 55.0)。
*   仅在最后几层（视觉主导层）插入效果最好 (55.9)。
*   这与 Figure 3 的观察一致：LVLM 的浅层可能更关注文本或低级特征，而深层更关注视觉语义。在视觉主导层进行重建约束更有效。

    ---

# 7. 总结与思考

## 7.1. 结论总结
本文提出了一种名为 FAM 的方法，旨在解决将生成式 LVLMs 适配为嵌入模型时存在的视觉表示不足和对齐粗糙问题。通过引入多粒度对齐对比 (MAC) 和视觉嵌入反演 (VEIN) 两种技术，FAM 在多个基准测试中取得了 SOTA 性能。实验证明，显式的细粒度对齐和特征重建约束能显著提升模型在检索和定位任务上的表现。

## 7.2. 局限性与未来工作
*   **计算开销**：虽然 VEIN 仅在训练时使用，不增加推理开销，但引入额外的解码器层和计算重建损失会增加训练时的计算成本和显存占用。
*   **掩码策略**：VEIN 的最佳掩码比例 (0.3) 与 MAE 差异较大，表明特征级重建有其独特的性质，未来可能需要更自适应的掩码策略。
*   **高分辨率扩展**：作者提到未来将在更高分辨率的设置下探索 FAM，因为高分辨率图像通常包含更丰富的细粒度信息，这正是 FAM 旨在捕捉的。

## 7.3. 个人启发与批判
*   **创新的迁移范式**：这篇论文给我最大的启发是“生成式模型到嵌入模型的迁移”并非简单的加个投影层。生成任务关注的是下一个词的概率，而嵌入任务关注的是特征的判别性和完整性。FAM 通过“重建”来约束特征完整性，这是一个非常巧妙的桥接，类似于在判别模型中引入生成式辅助任务。
*   **粒度的重要性**：MAC 的成功再次印证了“细节决定成败”。在多模态大模型时代，仅仅对齐全局语义往往是不够的，Token-to-Patch 的对齐监督对于提升模型对复杂场景的理解至关重要。
*   **潜在问题**：VEIN 依赖于未掩码的 token 来重建掩码 token。如果图像中物体非常密集或遮挡严重，未掩码的部分可能不足以提供上下文来重建被掩码的部分，此时 VEIN 的效果可能会打折扣。此外，对于纯文本任务，VEIN 并不适用，FAM 主要针对的是视觉增强。