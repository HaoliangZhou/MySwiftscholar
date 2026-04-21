# 1. 论文基本信息
## 1.1. 标题
论文标题为 *RETLLM: Training and Data-Free MLLMs for Multimodal Information Retrieval*，核心主题是提出一种完全不需要训练、也不需要额外数据的多模态大语言模型框架，用于解决多模态信息检索任务。

## 1.2. 作者
作者为Dawei Su、Dongsheng Wang，隶属于深圳大学计算机科学与软件工程学院，研究方向为多模态大语言模型、多模态信息检索。

## 1.3. 发表期刊/会议
该论文目前发布在arXiv预印本平台，尚未正式发表在顶会/顶刊，是一篇最新的预印本研究。

## 1.4. 发表年份
预印本发布时间为2026年2月25日（UTC）。

## 1.5. 摘要
本文针对现有基于多模态大语言模型（MLLM）的多模态信息检索方法存在的预训练目标不一致、依赖大规模训练数据的缺陷，提出了全新框架RetLLM，首次实现了完全无训练、无额外数据的MLLM多模态检索。该方法将多模态检索重新形式化为相似性分数生成任务，设计了粗到精的两阶段流程，同时提出无参数视觉增强模块解决MLLM丢失视觉细节的问题，以及基于熵的决策策略解决同分歧义问题。在多个主流多模态检索基准上的实验证明，RetLLM的零-shot性能超过了很多经过微调的SOTA模型，消融实验验证了每个组件的有效性。本文证明了预训练MLLM本身就具备很强的零-shot多模态检索能力，凸显了其固有的多模态推理潜力。

## 1.6. 原文链接
预印本链接：images/1.jpg)
*该图像是示意图，展示了RetLLM框架的工作流程。框架包括粗略阶段的Top $k$ 过滤、视觉增强和基于熵的选择，以实现有效的多模态检索。在细化阶段，输入查询与候选项进行比较，利用MLLM生成相关性评分，最终输出最佳候选。图中包含多个步骤和关键组件，说明了整个检索过程。*

## 4.2. 核心方法详解
### 4.2.1 问题定义
给定用户查询$q$，总共有$N$个候选$\Omega = \{ c_1, c_2, ..., c_N \}$，每个$q$和$c_n$都可以是图像、文本或者交错图文内容，本文关注top-1检索准确率，目标是找到和$q$最匹配的候选$c$。

### 4.2.2 粗选阶段：Top-k语义过滤
如果直接让MLLM给所有$N$个候选打分，当候选库规模很大时，会带来极高的时间成本。因此本文先用轻量级嵌入模型（如CLIP）做快速粗选，只保留最相关的top-k个候选，组成小候选池$\mathcal{C}$，具体计算为：
$$
\mathcal { C } = \mathrm { TopK } ( s ) , \quad s _ { i } = \frac { { \bf q } ^ { \top } { \bf c } _ { i } } { || { \bf q } || \ || {\bf c_i} || } , \quad i = 1 , 2 , \ldots , N ,
$$
符号解释：
- $\mathbf{q}$：CLIP提取的查询$q$的特征向量
- $\mathbf{c}_i$：CLIP提取的第$i$个候选$c_i$的特征向量
- $s_i$：查询$q$和候选$c_i$的余弦相似度，取值范围为`[-1,1]`，越大表示语义越相似
- $\mathrm{TopK}(s)$：选择相似度最高的$k$个候选组成候选池$\mathcal{C}$

  粗选阶段的作用：过滤掉大部分语义不相关的候选，将MLLM需要处理的候选数从$N$（通常很大）降到$k$（本文默认$k=5$），大幅降低计算量，同时让MLLM只需要关注难区分的高相关性候选，提升后续打分的准确率。

### 4.2.3 精选阶段：MLLM相似性分数预测
粗选后的候选池$\mathcal{C}$中的候选都和查询高相关，CLIP的粗粒度相似度已经无法区分，此时用MLLM做精细打分，将检索转化为分数生成任务：
$$
f _ { i } = \mathbf { MLLM } ( q , c _ { i } ) , \quad c _ { i } \in \mathcal { C } .
$$
符号解释：$f_i$是MLLM输出的查询$q$和候选$c_i$的相似性分数，越大表示越匹配。具体实现通过设计提示指令，将查询和候选输入MLLM，让MLLM直接输出0-10之间的相似度分数，解析输出得到$f_i$。

### 4.2.4 视觉增强模块：无参数缓解视觉细节丢失
MLLM普遍存在模态不平衡问题，生成过程中经常丢失细粒度视觉细节，引发幻觉导致打分错误。本文提出无参数的视觉增强方法，将原始视觉特征重新注入到Transformer的前馈网络（FFN）中，不需要训练就能补充视觉信息。

首先，标准Transformer FFN的原始形式为：
$$
\mathrm { FFN } ( \mathbf { x } ) = \phi ( \mathbf { x } \mathbf { W_1 } ) \mathbf { W_2 } ^ { \top } ,
$$
符号解释：
- $\mathbf{x} \in \mathbb{R}^d$：FFN的输入隐藏状态，$d$为隐藏维度
- $\mathbf{W_1}, \mathbf{W_2} \in \mathbb{R}^{d \times D}$：FFN的两个权重矩阵，通常$D=4d$
- $\phi$：激活函数，如ReLU、SiLU

  接下来将FFN重新形式化为键值记忆检索过程，把权重拆分为键向量和值向量：
$$
\mathbf { W_1 } = ( \mathbf { k_1 } , \mathbf { k_2 } , \ldots , \mathbf { k_D } ) , \quad \mathbf { W_2 } = ( \mathbf { v_1 } , \mathbf { v_2 } , \ldots , \mathbf { v_D } ) ,
$$
其中$k_i, v_i \in \mathbb{R}^d$分别为第$i$个键向量和值向量，代入原始公式后FFN可以改写为：
$$
\mathrm { FFN } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { D } \phi ( \langle \mathbf { x } , \mathbf { k } _ { i } \rangle ) \cdot v _ { i } .
$$
这个形式说明FFN本身就是一个记忆模块：用当前隐藏状态$\mathbf{x}$作为查询，检索记忆中相关的值。因此本文将输入的原始视觉词元集合$Z_v = \{ z_{v,1}, ..., z_{v,N_v} \}$（$N_v$为视觉词元数量）作为额外补充记忆加入，计算视觉修正项：
$$
\Delta ( \mathbf { x } \propto \mathbf { Z } _ { v } ) = \sum _ { j = 1 } ^ { N_v } \phi ( \langle \mathbf { x } , \mathbf { z } _ { v , j } \rangle ) \cdot \mathbf { z } _ { v , j } .
$$

最后将原始FFN输出和视觉修正项融合，得到增强后的FFN输出：
$$
\mathrm { FFN } ^ { ( l ) } ( \mathbf { x } \propto \mathbf { Z } _ { \mathbf { v } } ) = \alpha \Delta ( \mathbf { x } \propto \mathbf { Z } _ { v } ) + ( 1 - \alpha ) \mathrm { FFN } ( \mathbf { x } ) ,
$$
符号解释：
- $\alpha \in [0,1]$：视觉注入比例超参数，本文默认取0.3
- $\mathrm { FFN } ^ { ( l ) } ( \dots )$：第$l$层增强后的FFN输出

  该方法的优势：不需要引入任何可训练参数，不改变MLLM的原始权重，完全符合无训练的设定，就能有效补充视觉细节，缓解幻觉。

### 4.2.5 基于熵的决策：解决同分歧义
实际应用中经常出现多个候选获得相同的最高相似性分数，导致无法排序。本文提出基于熵的不确定性校准策略，对于同分的候选集合$\mathcal{P}$，选择模型最确定的候选作为最终结果。

具体步骤：首先设计二分类提示：$"<query>, <candidate>. Does the candidate match the query, True or False."$，输入MLLM后得到输出层的概率分布，计算分布的熵作为不确定性分数：
$$
H _ { i } = - \sum _ { v = 1 } ^ { V } p _ { v } \log p _ { v } ,
$$
符号解释：
- $V$：MLLM的词表大小
- $p_v$：输出层第$v$个词的softmax概率
- $H_i$：该$(q,c_i)$对的熵，熵越低说明模型对自己的判断越确定，熵越高越不确定。

  最后从同分候选中选择熵最小的作为最终结果：
$$
C ^ { * } = \arg \min _ { C _ { i } \in \mathcal { P } } H _ { i } ,
$$
该策略不需要训练，仅需要一次额外前向传播，就能有效解决歧义，提升检索可靠性。

# 5. 实验设置
## 5.1. 数据集
本文在6个主流基准上做零-shot评估，覆盖不同类型的检索任务：
1. **Flickr30K**：经典图文检索基准，包含3万张图像和对应人工标注文本，用于测试短图文双向检索。
2. **COCO**：MS COCO，大规模图文检索基准，包含12万张图像，每个图像对应5个标注文本，是图文检索的通用测试集。
3. **ShareGPT4V**：包含长文本描述的多模态基准，用于测试模型处理长文本查询/候选的能力。
4. **Urban1K**：面向长文本的检索基准，专门测试长文本场景下的检索性能。
5. **SugarCrepe**：组合语义检索基准，测试模型对组合图文语义的理解能力，包含Replace、Swap、Add三个子集。
6. **MMEB**：大规模多模态嵌入基准，包含36个不同的多模态任务，分为分类、VQA、检索、定位四个大类，用于测试零-shot泛化能力。

   所有实验都采用零-shot设置，不使用任何训练数据，完全符合本文无训练的设定，这些数据集覆盖了不同长度、不同类型的多模态检索任务，能全面验证方法的有效性。

## 5.2. 评估指标
本文使用两个核心评估指标，具体说明如下：

### 5.2.1 Recall@1（top-1召回率）
- **概念定义**：Recall@1衡量所有测试查询中，正确候选被模型排在第一位的比例，是检索任务最核心的指标，数值越高表示性能越好。
- **数学公式**：
  $$
\text{Recall@1} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I} \left( \text{rank}(c_q^*, q) = 1 \right)
$$
- **符号解释**：
  - $Q$：所有测试查询的集合，$|Q|$是测试查询总数
  - $c_q^*$：查询$q$对应的正确目标候选
  - $\text{rank}(c_q^*, q)$：$c_q^*$在模型输出排序中的位置
  - $\mathbb{I}(\cdot)$：指示函数，条件为真时输出1，否则输出0。

### 5.2.2 Precision@1（top-1精确率）
- **概念定义**：本文在MMEB基准中使用Precision@1，衡量所有测试查询中，模型输出的top-1结果正确的比例，在top-1检索设定下和Recall@1本质等价，数值越高越好。
- **数学公式**：
  $$
\text{Precision@1} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I} \left( \text{top-1}(q) = c_q^* \right)
$$
- **符号解释**：$\text{top-1}(q)$是模型输出的排名第一的候选，其他符号和Recall@1一致。

## 5.3. 对比基线
本文选择的基线覆盖了两类主流方法，都是当前领域内的代表性SOTA模型：
1. **CLIP系列传统基线**：CLIP(ViT-L)、OpenCLIP(ViT-L)、SigLIP、CLIP(ViT-BigG/14)、EVA-CLIP，代表了传统对比训练多模态检索的最高水平。
2. **基于MLLM的训练式基线**：E5-V、VLM2Vec、UniME，都是近年提出的基于MLLM微调的多模态检索模型，代表了有训练方法的SOTA水平，用于和本文的无训练方法做公平对比。

   所有基线都在相同的零-shot设置下评估，对比公平。

# 6. 实验结果与分析
## 6.1. 核心结果分析
以下是原文 Table 1 在多个图文和组合检索基准上的结果：

<table>
<thead>
<tr>
<th rowspan="3">Method</th>
<th colspan="4">Short Caption Retrieval</th>
<th colspan="4">Long Caption Retrieval</th>
<th colspan="3">Compositional Retrieval</th>
</tr>
<tr>
<th colspan="2">Flickr30K</th>
<th colspan="2">COCO</th>
<th colspan="2">ShareGPT4V</th>
<th colspan="2">Urban1K</th>
<th rowspan="2">Replace</th>
<th rowspan="2">Swap</th>
<th rowspan="2">Add</th>
</tr>
<tr>
<td>qi → ct</td>
<td>qt → ci</td>
<td>qi → ct</td>
<td>qt → ci</td>
<td>qi → ct</td>
<td>qt → ci</td>
<td>qi → ct</td>
<td>qt → ci</td>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP(ViT-L)</td>
<td>87.2</td>
<td>67.3</td>
<td>58.1</td>
<td>37.0</td>
<td>81.8</td>
<td>84.0</td>
<td>47.0</td>
<td>47.0</td>
<td>79.5</td>
<td>62.7</td>
<td>74.9</td>
</tr>
<tr>
<td>EEVA-CLIP</td>
<td>93.9</td>
<td>78.8</td>
<td>68.8</td>
<td>51.1</td>
<td>93.1</td>
<td>81.2</td>
<td>80.0</td>
<td>77.0</td>
<td>85.9</td>
<td>70.3</td>
<td>86.7</td>
</tr>
<tr>
<td>E5V</td>
<td>88.7</td>
<td>79.5</td>
<td>62.0</td>
<td>52.0</td>
<td>85.1</td>
<td>82.1</td>
<td>88.9</td>
<td>83.2</td>
<td>86.3</td>
<td>67.6</td>
<td>76.9</td>
</tr>
<tr>
<td>VLM2Vec</td>
<td>90.6</td>
<td>76.0</td>
<td>66.6</td>
<td>46.0</td>
<td>89.8</td>
<td>86.9</td>
<td>91.0</td>
<td>82.4</td>
<td>85.5</td>
<td>64.8</td>
<td>94.2</td>
</tr>
<tr>
<td>UniME</td>
<td>93.4</td>
<td>81.9</td>
<td>70.1</td>
<td>53.7</td>
<td>97.2</td>
<td>93.9</td>
<td>95.9</td>
<td>95.2</td>
<td>89.0</td>
<td>77.6</td>
<td>94.4</td>
</tr>
<tr>
<td>RetLLM</td>
<td>94.5</td>
<td>82.0</td>
<td>70.4</td>
<td>54.1</td>
<td>97.6</td>
<td>94.2</td>
<td>88.9</td>
<td>78.6</td>
<td>94.8</td>
<td>92.7</td>
<td>96.2</td>
</tr>
</tbody>
</table>

注：qi→ct表示图像查询搜文本候选，qt→ci表示文本查询搜图像候选，最优结果加粗。

从结果可以看出：RetLLM在大部分任务上都超过了所有基线，包括经过微调的MLLM模型。例如Flickr30K图像搜文，RetLLM达到94.5% Recall@1，超过E5-V的88.7%、VLM2Vec的90.6%、UniME的93.4%；在组合检索的Swap任务上，RetLLM达到92.7%，比第二名UniME高了15.1个百分点，证明其细粒度语义理解能力远优于现有方法；在SugarCrepe Add任务上，比VLM2Vec高2个百分点，凸显了零-shot推理的优势。

接下来是MMEB基准的结果，以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<th>Models</th>
<th>#Parameters</th>
<th colspan="4">Per Meta-Task Score</th>
<th colspan="3">Average Score</th>
</tr>
<tr>
<th></th>
<th></th>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>Grounding</th>
<th>IND</th>
<th>OOD</th>
<th>Overall</th>
</tr>
<tr>
<th># of Datasets →</th>
<th></th>
<th>10</th>
<th>10</th>
<th>12</th>
<th>4</th>
<th>20</th>
<th>16</th>
<th>36</th>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP(ViT-L)</td>
<td>0.4B</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.1</td>
<td>38.7</td>
<td>39.2</td>
</tr>
<tr>
<td>OpenCLIP(ViT-L)</td>
<td>0.4B</td>
<td>41.5</td>
<td>6.9</td>
<td>44.6</td>
<td>53.5</td>
<td>32.8</td>
<td>36.0</td>
<td>36.6</td>
</tr>
<tr>
<td>SigLIP(So/14)</td>
<td>0.9B</td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>32.3</td>
<td>38.0</td>
<td>35.0</td>
</tr>
<tr>
<td>CLIP(ViT-BigG/14)</td>
<td>2.5B</td>
<td>52.3</td>
<td>14.0</td>
<td>50.5</td>
<td>60.3</td>
<td>38.9</td>
<td>45.8</td>
<td>44.3</td>
</tr>
<tr>
<td>EVA-CLIP</td>
<td>8B</td>
<td>56.0</td>
<td>10.4</td>
<td>49.2</td>
<td>58.9</td>
<td>38.1</td>
<td>45.6</td>
<td>43.7</td>
</tr>
<tr>
<td>E5-V</td>
<td>7B</td>
<td>39.7</td>
<td>10.8</td>
<td>39.4</td>
<td>60.2</td>
<td>34.2</td>
<td>33.4</td>
<td>37.5</td>
</tr>
<tr>
<td>UniME</td>
<td>7B</td>
<td>43.0</td>
<td>17.7</td>
<td>42.5</td>
<td>63.2</td>
<td>37.6</td>
<td>38.6</td>
<td>41.6</td>
</tr>
<tr>
<td>RetLLM</td>
<td>7B</td>
<td>60.3</td>
<td>27.8</td>
<td>62.4</td>
<td>60.2</td>
<td>52.0</td>
<td>50.2</td>
<td>54.2</td>
</tr>
</tbody>
</table>

从结果可以看出：相同7B参数规模下，RetLLM整体平均Precision@1达到54.2%，比最强的零-shot基线UniME（41.6%）高了12.6个百分点，在分类、VQA、检索三个大类上都远远超过所有基线，证明RetLLM的泛化能力非常强，在不同类型的多模态任务上都有优异的零-shot表现。

## 6.2. 消融实验/参数分析
### 6.2.1 组件有效性验证
以下是原文 Table 3 对两个核心组件的消融结果：

<table>
<thead>
<tr>
<th>Components</th>
<th colspan="2">Flickr30k</th>
<th colspan="2">COCO</th>
</tr>
<tr>
<th></th>
<th>qi → ct</th>
<th>qt → ci</th>
<th>qi → ct</th>
<th>qt → ci</th>
</tr>
</thead>
<tbody>
<tr>
<td>ALL（全组件）</td>
<td>94.5</td>
<td>81.8</td>
<td>69.2</td>
<td>52.1</td>
</tr>
<tr>
<td>entropy only（仅熵决策，无视觉增强）</td>
<td>94.0</td>
<td>81.2</td>
<td>68.7</td>
<td>50.8</td>
</tr>
<tr>
<td>enhancement only（仅视觉增强，无熵决策）</td>
<td>94.0</td>
<td>80.7</td>
<td>68.8</td>
<td>51.5</td>
</tr>
<tr>
<td>MLLM only（两个组件都去掉）</td>
<td>93.6</td>
<td>80.2</td>
<td>66.9</td>
<td>50.3</td>
</tr>
</tbody>
</table>

结果证明：去掉任何一个组件都会带来性能下降，去掉视觉增强后COCO图像搜文下降0.5个百分点，去掉熵决策后Flickr30K文搜图下降1.1个百分点，两个组件都去掉后性能下降更多，证明两个组件都对最终性能有正向贡献，且存在协同增益。

### 6.2.2 Top-k敏感性分析
不同top-k值对性能和效率的影响如下图所示：

![Fig. 2: Ablation studies on the impact of top- $\\mathbf { \\nabla } \\cdot \\mathbf { k }$ values on retrieval performance and inference efficiency.](images/2.jpg)
*该图像是一个示意图，展示了不同 top-$k$ 值对检索性能和推理效率的影响。左侧为平均性能（Accuracy）随 top-$k$ 的变化，右侧为推理时间（Inference Time）随 top-$k$ 的变化，分别显示了从图像到文本（Img→Txt）和从文本到图像（Txt→Img）的表现。*

结果显示：k越大，性能越高，但推理时间也线性增长，k=5时性能和推理时间达到了最佳平衡，验证了本文默认设置的合理性，符合粗到精策略平衡效率和准确率的设计目标。

### 6.2.3 模型可扩展性验证
本文测试了不同CLIP主干和不同MLLM对性能的影响，结果如下：
- 不同CLIP主干：CLIP能力越强，最终检索性能越高，擅长处理长文本的Long-CLIP在Urban1K基准上，文搜图性能从78.6%提升到95.8%，提升非常显著，证明框架能有效受益于更强的粗选模型。
- 不同MLLM：从Phi-3.5-V到Qwen2-VL再到Qwen2.5-VL，随着MLLM本身能力提升，RetLLM的性能稳步提升，证明框架具备很好的可扩展性，是即插即用的，任何新的更强的MLLM都可以直接接入获得性能提升。

# 7. 总结与思考
## 7.1. 结论总结
本文提出了RetLLM，首个完全无训练、无额外数据的多模态信息检索框架，核心思路是将检索任务重新形式化为MLLM的相似性分数生成任务，通过粗到精两阶段策略平衡效率和准确率，加上无参数视觉增强和熵基决策解决MLLM的固有缺陷。实验证明，该方法在多个主流基准上的零-shot性能超过了很多经过微调的SOTA模型，证明了预训练MLLM本身就具备很强的多模态检索能力，不需要微调就能应用，且框架具备很好的可扩展性，能即插即用受益于更强的基础模型，是一种简单、可扩展的检索方案。

## 7.2. 局限性与未来工作
本文作者没有明确指出自身局限性，从方法本身分析，潜在的局限性包括：
1. 粗选阶段依赖外部CLIP模型，粗选的质量直接影响最终结果，如果粗选过滤掉了正确候选，后续精选阶段无法召回。
2. 即使k=5，仍然需要对每个候选做一次MLLM前向传播，推理速度比纯CLIP检索慢，在百万级大规模候选库上的效率还有提升空间。
3. 本文主要测试了top-1检索准确率，没有验证大规模全库检索下的recall@k等指标，在实际大规模场景下的性能还需要进一步验证。

   未来可能的研究方向：
1. 可以用MLLM原生的嵌入做粗选，不需要依赖外部CLIP，实现端到端的无训练检索，进一步提升性能。
2. 优化推理效率，比如批量处理候选、 early exit 等方法，让RetLLM能适配大规模候选库。
3. 将RetLLM集成到多模态RAG系统中，验证其在实际下游任务中的应用价值。

## 7.3. 个人启发与批判
这篇论文的思路非常有启发性，打破了“MLLM做检索必须微调”的固有认知，证明了只要任务形式设计合理，充分利用预训练MLLM本身的知识，完全不需要训练就能超过微调模型，这个思路可以推广到很多其他MLLM的区分性任务（比如分类、匹配、排序），开辟了一个新的研究方向。此外，RetLLM即插即用的特性，不需要修改模型参数，成本很低，非常适合实际部署，有很高的实用价值。

潜在的可改进点：视觉增强需要修改MLLM的前向过程，对于封装好的MLLM部署来说不够方便，未来可以探索通过提示工程实现类似的视觉增强效果，不需要修改前向；熵决策需要额外一次前向传播，增加了推理时间，未来可以探索从原始打分的输出中直接提取不确定性，不需要额外前向，进一步提升效率。