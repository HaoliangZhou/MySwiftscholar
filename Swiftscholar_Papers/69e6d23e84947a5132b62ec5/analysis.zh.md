# 1. 论文基本信息
## 1.1. 标题
论文标题为《U-MARVEL: Unveiling Key Factors for Universal Multimodal Retrieval via Embedding Learning with MLLMs》，核心主题是系统性探索基于多模态大语言模型（MLLM）的通用多模态检索嵌入学习的关键影响因素，并提出高性能的统一检索框架U-MARVEL。
## 1.2. 作者
作者及隶属机构如下：
- 李小杰（Xiaojie Li）：腾讯PCG、南京大学
- 李楚（Chu Li）：字节跳动
- 陈士哲（Shi-Zhe Chen）：腾讯PCG（通讯作者）
- 陈曦（Xi Chen）：腾讯PCG（通讯作者）
## 1.3. 发表期刊/会议
该论文目前为arXiv预印本状态，尚未正式发表于期刊或会议。
## 1.4. 发表年份
2025年（预印本发布时间为2025年7月20日）
## 1.5. 摘要
现有基于MLLM的通用多模态检索（UMR）方法大多采用对比学习框架，但缺乏对训练机制的系统性分析，导致性能次优、泛化能力有限。为解决该问题，论文首先搭建了通用的MLLM嵌入学习流水线，系统分析了高性能通用检索系统的核心影响因素，包括嵌入生成方式、渐进式训练策略、硬负样本挖掘、重排蒸馏等多个维度，发现了多个此前被忽略但对性能影响显著的关键因素。基于上述发现，论文提出了统一框架U-MARVEL，在M-BEIR基准的有监督设置下大幅超越现有最先进（SOTA）方法，同时在组合图像检索、文本到视频检索等任务上表现出极强的零样本性能，验证了框架在不同嵌入检索任务上的泛化潜力。
## 1.6. 原文链接
- 原文链接：images/1.jpg)
*该图像是插图，展示了 U-MARVEL 系统的设计原则和训练策略。在左侧部分，展示了多模态检索的示例，包括如何生成嵌入和使用硬负采样的方法。右侧部分描述了为了将检索和重排模型联合成一个模型的知识蒸馏过程。*

## 4.2. 核心方法详解
### 4.2.1 基线流水线搭建
论文首先构建了通用的基线训练流程：
- 骨干模型：选择Qwen2-VL-7B-Instruct作为基础MLLM，冻结视觉编码器，仅用LoRA微调LLM部分；
- 训练目标：采用InfoNCE损失作为对比学习目标，原文公式如下：
  $$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \log \frac { \exp ( \sin ( e _ { q } , e _ { c } ^ { + } ) / \tau ) } { \sum _ { i } \exp ( \sin ( e _ { q } , e _ { c ^ { \mathrm { i } } } ) / \tau ) }
$$
符号解释：
  - $e_q$：查询的嵌入向量；
  - $e_c^+$：与查询匹配的正候选的嵌入向量；
  - $e_{c^i}$：候选集中任意样本的嵌入向量（包含正样本和所有负样本）；
  - $\tau$：温度系数，用于控制对比损失的分布尖锐程度，值越小分布越尖锐；
  - $\sin(\cdot, \cdot)$：余弦相似度函数，衡量两个向量的语义相似程度，取值范围为[-1, 1]，值越大相似度越高。
    该损失的优化目标是让正样本对的相似度经过温度缩放后的指数值，在所有候选的指数值总和中占比尽可能大，从而实现正、负样本的有效区分。
### 4.2.2 MLLM到嵌入模型的适配
生成式MLLM采用自回归解码，训练目标是预测下一个token，而嵌入模型需要编码整个输入序列的全局语义，二者存在本质差异，因此需要专门的适配策略。论文从三个维度进行了探索：
#### 4.2.2.1 嵌入提取
现有方法的嵌入提取策略分为两类：
1.  **Last Token方案**：在输入末尾加入压缩提示词（如“将上述图像和文本总结为一个词：<emb>”），采用MLLM默认的因果注意力（每个token仅能看到前文的token），提取最后一个token<emb>的隐藏状态作为输入的嵌入，是现有UMR方法的主流方案。
2.  **Mean Token方案**：将MLLM的因果注意力修改为双向注意力（每个token可看到序列中所有位置的token，与BERT的注意力机制一致），对最后一层所有token的隐藏状态进行均值池化，得到输入的嵌入。
    论文通过对比实验得到**Finding 1**：双向注意力+均值池化、且去掉压缩提示词的方案，性能优于主流的Last Token方案。二者的性能对比如下（原文Table 1）：

    <table>
    <thead>
    <tr>
    <th>ID</th>
    <th>Causal or Bidirectional</th>
    <th>Last token or Mean token</th>
    <th>Compression Prompt ("in one word")</th>
    <th>Local Avg.</th>
    <th>Global Avg.</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>0</td>
    <td>Causal</td>
    <td>Last token</td>
    <td>√</td>
    <td>56.6</td>
    <td>54.8</td>
    </tr>
    <tr>
    <td>1</td>
    <td>Bidirectional</td>
    <td>Last token</td>
    <td>√</td>
    <td>55.5</td>
    <td>52.6</td>
    </tr>
    <tr>
    <td>2</td>
    <td>Causal</td>
    <td>Mean token</td>
    <td>√</td>
    <td>33.7</td>
    <td>27.6</td>
    </tr>
    <tr>
    <td>3</td>
    <td>Bidirectional</td>
    <td>Mean token</td>
    <td>√</td>
    <td>51.7</td>
    <td>46.1</td>
    </tr>
    <tr>
    <td>4</td>
    <td>Bidirectional</td>
    <td>Mean token</td>
    <td>×</td>
    <td>57.2</td>
    <td>55.2</td>
    </tr>
    </tbody>
    </table>

原因分析：Last Token方案存在近期偏差，嵌入结果过度依赖末尾token的输出，而双向注意力可让所有token获得全局语义信息，均值池化可整合全序列的特征，去掉压缩提示词后避免了额外提示词对语义的干扰，因此效果更优。
#### 4.2.2.2 指令融合
UMR采用指令引导的任务范式，输入为“指令 + 查询内容”的拼接序列。论文实验发现，在均值池化时掩码掉指令对应的token，仅对查询内容的token进行池化，可进一步提升性能，得到**Finding 2**：掩码指令token可消除指令偏差，提升嵌入性能。二者对比如下（原文Table 2）：

<table>
<thead>
<tr>
<th>ID</th>
<th>Instructions</th>
<th>Masking</th>
<th>Local Avg.</th>
<th>Global Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>√</td>
<td>√</td>
<td>57.3</td>
<td>55.5</td>
</tr>
<tr>
<td>1</td>
<td>√</td>
<td>×</td>
<td>57.2</td>
<td>55.2</td>
</tr>
<tr>
<td>2</td>
<td>×</td>
<td>-</td>
<td>55.3</td>
<td>43.4</td>
</tr>
</tbody>
</table>

原因分析：双向注意力的计算过程中，指令的语义信息已经融入到所有token的隐藏状态中，无需再将指令特征纳入池化计算，这样可避免指令引入的偏差，仅聚焦于查询和候选的内容相似度比较。
#### 4.2.2.3 渐进式过渡
直接在多模态检索数据集上微调MLLM性能不佳，因为MLLM原本采用生成式训练、因果注意力，直接切换为双向注意力的对比学习任务跨度太大。论文提出三步渐进式训练策略，得到**Finding 3**：从简单到复杂的渐进式过渡训练，可有效提升MLLM向检索模型的适配效果。实验结果如下（原文Table 3）：

<table>
<thead>
<tr>
<th>ID</th>
<th>Methods</th>
<th>Local Avg.</th>
<th>Global Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>Instruction Tuning on M-BEIR</td>
<td>56.6</td>
<td>53.9</td>
</tr>
<tr>
<td>1</td>
<td>[0] + text-only retrieval</td>
<td>57.3</td>
<td>55.5</td>
</tr>
<tr>
<td>2</td>
<td>[1] + text-image retrieval</td>
<td>57.7</td>
<td>55.8</td>
</tr>
</tbody>
</table>

渐进式训练的三个步骤为：
1.  纯文本检索适配：在NLI（自然语言推理）数据集上训练文本检索能力，让模型先适应对比学习的优化目标；
2.  跨模态对齐：在CC3M图文对数据集上进行图文对比学习，对齐文本和图像编码器的表征空间；
3.  多模态指令检索：在目标UMR数据集（M-BEIR）上微调，适配多任务、多模态的指令检索场景。
    该策略符合课程学习的思想，逐步提升任务难度，避免模型遗忘预训练知识，更易收敛到最优解。
### 4.2.3 基于InfoNCE的训练策略
论文进一步探索了对比学习训练过程中的关键影响因素：
#### 4.2.3.1 批次大小、学习率、温度的交互作用
现有研究普遍认为对比学习的批次越大性能越好，但论文发现三者存在强交互作用，得到**Finding 4**：批次增大时需要同步缩放学习率才能获得性能收益，且可学习温度的效果远优于固定温度。实验结果如下（原文Table 4）：

<table>
<thead>
<tr>
<th>ID</th>
<th>Batch Size</th>
<th>Temp (τ)</th>
<th>LR (η)</th>
<th>Local Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>480</td>
<td>0.05 (fixed)</td>
<td>1e-4</td>
<td>57.2</td>
</tr>
<tr>
<td>1</td>
<td>1920</td>
<td>0.05 (fixed)</td>
<td>1e-4</td>
<td>57.4</td>
</tr>
<tr>
<td>2</td>
<td>1920</td>
<td>0.05 (fixed)</td>
<td>2e-4</td>
<td>58.3</td>
</tr>
<tr>
<td>3</td>
<td>3840</td>
<td>0.05 (fixed)</td>
<td>2.8e-4</td>
<td>58.5</td>
</tr>
<tr>
<td>4</td>
<td>3840</td>
<td>0.05 (fixed)</td>
<td>4e-4</td>
<td>58.9</td>
</tr>
<tr>
<td>5</td>
<td>5760</td>
<td>0.05 (fixed)</td>
<td>6e-4</td>
<td>58.7</td>
</tr>
<tr>
<td>6</td>
<td>7680</td>
<td>0.05 (fixed)</td>
<td>6e-4</td>
<td>58.9</td>
</tr>
<tr>
<td>7</td>
<td>3840</td>
<td>0.05 (learnable)</td>
<td>4e-4</td>
<td>60.1</td>
</tr>
<tr>
<td>8</td>
<td>7680</td>
<td>0.05 (learnable)</td>
<td>4e-4</td>
<td>60.3</td>
</tr>
</tbody>
</table>

关键结论包括：
1.  批次从480提升到1920时，若学习率保持1e-4不变，性能仅提升0.2；若学习率同步翻倍到2e-4，性能提升1.1；
2.  可学习温度比固定温度带来1.2的性能提升，因为可学习温度可动态调整对比损失的分布尖锐程度，适配不同训练阶段的需求。
#### 4.2.3.2 带硬负采样的持续训练
硬负样本可提升模型的判别能力，但直接使用top-K硬负样本训练易导致模型崩溃，因为大量硬负样本是假负样本（语义与查询相关但标注为负样本的数据集噪声）。论文提出了过滤式硬负采样策略，得到**Finding 5**：过滤假负样本并与批次内随机负样本混合训练，可平衡训练难度和性能提升。实验结果如下（原文Table 5）：

<table>
<thead>
<tr>
<th>ID</th>
<th>Methods</th>
<th>Local Avg.</th>
<th>Global Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="2">Progressive Transition (baseline)</td>
<td>60.6</td>
<td>58.7</td>
</tr>
<tr>
<td>0</td>
<td>only top-k hard neg</td>
<td>failed</td>
<td>failed</td>
</tr>
<tr>
<td>1</td>
<td>in-batch neg and top-k hard neg</td>
<td>57.4</td>
<td>55.4</td>
</tr>
<tr>
<td>2</td>
<td>in-batch neg and filtered top-k hard neg</td>
<td>61.7</td>
<td>59.9</td>
</tr>
</tbody>
</table>

硬负采样的流程为：
1.  用渐进式训练得到的base模型提取所有查询和候选的特征，计算相似度，得到每个查询的top-K硬负样本；
2.  过滤掉相似度超过阈值（论文设为0.7）的硬负样本（这类样本大概率是假负样本）；
3.  将过滤后的硬负样本与批次内随机负样本混合，进行持续微调。
### 4.2.4 重排蒸馏策略
两阶段的召回-重排架构精度高但推理延迟大，论文提出了高效的蒸馏方案，将两阶段的知识迁移到单阶段模型中，得到**Finding 6**：仅对正样本和硬负样本蒸馏的改进方案，可大幅降低蒸馏计算量，同时提升性能。
下图（原文Figure 2）展示了蒸馏的整体流程：

![Figure 2: Distillation Ilustration. It shows how the teacher model generates scores, and compares the traditional and improved distillation methods.](images/2.jpg)
*Figure 2: Distillation Ilustration. It shows how the teacher model generates scores, and compares the traditional and improved distillation methods.*

#### 4.2.4.1 重排器训练
论文首先训练了一个生成式重排器：输入为“查询 + 候选”的拼接序列，让模型输出YES或NO表示候选是否与查询相关，采用next token预测损失训练。推理时，召回模型先返回top-K候选，重排器预测每个候选为YES的概率作为重排分数，将召回相似度分数与重排分数加权融合得到教师模型的综合分数：
$$S_{multi} = \alpha \cdot S_{recall} + (1-\alpha) \cdot S_{rerank}$$
其中$\alpha$为加权系数，论文设为0.5。
#### 4.2.4.2 高效蒸馏方案
传统蒸馏方法需要计算批次内所有查询和候选的重排分数，计算量随批次大小平方增长，当批次为3840时完全无法落地。论文提出的改进方案仅对每个查询的正样本和top-K硬负样本计算重排分数，蒸馏时仅对齐这部分的分布，蒸馏损失采用KL散度：
$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = D _ { K L } ( S _ { m u l t i } \parallel S _ { s i n g l e } ) = \sum _ { i } S _ { m u l t i } ( i ) \log \frac { S _ { m u l t i } ( i ) } { S _ { s i n g l e } ( i ) }
$$
符号解释：
- $S_{multi}$：教师模型的综合分数，经softmax归一化；
- $S_{single}$：学生模型（单阶段召回模型）的分数，经softmax归一化；
- $D_{KL}(\cdot \parallel \cdot)$：KL散度，衡量两个概率分布的差异，训练目标是最小化该值，让学生的输出分布与教师对齐。
  计算量分析：设批次大小为$n$，总查询数为$N$，硬负样本数为$k$，传统方法总计算量为$(n+6)N$，改进方法为$(3k+7)N$。论文中$n=3840$，$k=50$，改进方法的计算量仅为传统方法的4.1%，训练时间从340小时降至14小时，同时训练时模型看到的特征数是传统方法的26倍，效果更优。
蒸馏的性能对比如下（原文Table 6）：

<table>
<thead>
<tr>
<th>Methods</th>
<th>Local Avg.</th>
<th>Global Avg.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Hard Negative Mining (baseline)</td>
<td>61.7</td>
<td>59.9</td>
</tr>
<tr>
<td>Recall-Reranker</td>
<td>64.5</td>
<td>61.7</td>
</tr>
<tr>
<td>Distillation</td>
<td>63.2</td>
<td>60.7</td>
</tr>
<tr>
<td>Continue-hard</td>
<td>62.2</td>
<td>60.0</td>
</tr>
</tbody>
</table>

可以看到蒸馏后的单模型比基线提升1.5，且比仅用硬负样本持续训练的方案高1.0，证明性能提升来自蒸馏而非硬负样本。
### 4.2.5 U-MARVEL整体框架
整合上述所有最优策略，U-MARVEL的训练分为三个阶段：
1.  渐进式过渡训练：通过三步课程学习，将MLLM适配为基础检索模型；
2.  硬负采样与重排器训练：用过滤后的硬负样本微调得到高性能召回模型，同时训练重排器，二者分数融合得到教师模型；
3.  高效蒸馏：将教师模型的知识蒸馏到单阶段学生模型中，得到最终的U-MARVEL单模型，兼顾精度和推理效率。
    此外，U-MARVEL+为两阶段版本，采用U-MARVEL作为召回模型，搭配重排器进行重排，可获得更高精度。
# 5. 实验设置
## 5.1. 数据集
### 5.1.1 有监督训练数据集M-BEIR
M-BEIR是UMR领域的标准基准，包含8类检索任务、10个子数据集、4个领域（新闻、维基、时尚、杂项），具体规模如下：
- 训练集：133万查询，193万候选；
- 测试集：19万查询，560万候选。
  包含两种测试设置：
1.  Local池：每个任务仅在自身的候选池内检索，不会出现模态错误，难度较低；
2.  Global池：所有任务的候选池合并为一个全局候选池，检索可能出现模态错误，难度更高。
### 5.1.2 零样本测试数据集
论文采用12个未见过的数据集验证零样本性能：
- 图文跨模态检索：ShareGPT4V、Urban-1K、Flickr30K；
- 组合图像检索：CIRCO、GeneCIS；
- 对话检索：Visual Dialog；
- 多轮时尚检索：Multi-round FashionIQ；
- 图文匹配：CC-Neg、Sugar-Crepe；
- 文本到视频检索：MSR-VTT、MSVD。
## 5.2. 评估指标
论文核心采用<strong>Recall@k（召回率@k）</strong>作为评估指标，具体说明如下：
1.  **概念定义**：Recall@k衡量的是所有测试查询中，前k个检索结果包含至少一个正样本的查询占比，值越高说明模型的召回能力越强，越容易在前k个结果中找到用户需要的内容。
2.  **数学公式**：
    $$
\text{Recall@k} = \frac{N_{pos@k}}{N_{total}}
$$
3.  **符号解释**：
    - $N_{pos@k}$：前k个检索结果中包含至少一个正样本的查询数量；
    - $N_{total}$：测试集的总查询数量。
      论文中，Fashion200K和FashionIQ任务采用Recall@10，其余任务采用Recall@5，最终计算所有任务的平均值作为综合指标Avg。
## 5.3. 对比基线
论文选择了UMR领域的代表性方法作为对比基线：
1.  单模型基线：CLIP-L、SigLIP、UniIR-BLIP、UniIR-CLIP、LamRA-Ret（此前单模型SOTA）、MM-Embed、VLM2Vec、UniME；
2.  两阶段基线：LamRA（此前两阶段SOTA）。
# 6. 实验结果与分析
## 6.1. 核心结果分析
### 6.1.1 M-BEIR有监督性能
M-BEIR基准Local池的性能对比如下（原文Table 7）：

<table>
<thead>
<tr>
<th rowspan="3">Methods</th>
<th colspan="3">$q^t \to c^i$</th>
<th>$q^t \to c^t$</th>
<th colspan="2">$q^t \to (c^t, c^i)$</th>
<th colspan="3">$q^i \to c^t$</th>
<th>$q^i \to c^i$</th>
<th colspan="2">$(q^i, q^t) \to c^t$</th>
<th colspan="2">$(q^i, q^t) \to c^i$</th>
<th colspan="2">$(q^i, q^t) \to (c^i, c^t)$</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>VN</th>
<th>CO</th>
<th>F200</th>
<th>WQ</th>
<th>ES</th>
<th>WQ</th>
<th>VN</th>
<th>CO</th>
<th>F200</th>
<th>NS</th>
<th>ON</th>
<th>InS</th>
<th>FQ</th>
<th>CR</th>
<th>ON</th>
<th>InS</th>
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
<th>R@10</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
<th>R@10</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
<th>R@5</th>
</tr>
</thead>
<tbody>
<tr>
<th colspan="19">Single model</th>
</tr>
<tr>
<td>CLIP-L</td>
<td>43.3</td>
<td>61.1</td>
<td>6.6</td>
<td>36.2</td>
<td>43.3</td>
<td>45.1</td>
<td>41.3</td>
<td>79</td>
<td>7.7</td>
<td>26.1</td>
<td>24.2</td>
<td>20.5</td>
<td>7</td>
<td>13.2</td>
<td>38.8</td>
<td>26.4</td>
<td>32.5</td>
</tr>
<tr>
<td>SigLIP</td>
<td>30.1</td>
<td>75.7</td>
<td>36.5</td>
<td>39.8</td>
<td>27</td>
<td>43.5</td>
<td>30.8</td>
<td>88.2</td>
<td>34.2</td>
<td>28.9</td>
<td>29.7</td>
<td>25.1</td>
<td>14.4</td>
<td>22.7</td>
<td>41.7</td>
<td>27.4</td>
<td>37.2</td>
</tr>
<tr>
<td>UnilR-BLIP</td>
<td>23.4</td>
<td>79.7</td>
<td>26.1</td>
<td>80</td>
<td>50.9</td>
<td>79.8</td>
<td>22.8</td>
<td>89.9</td>
<td>28.9</td>
<td>33</td>
<td>41</td>
<td>22.4</td>
<td>29.2</td>
<td>52.2</td>
<td>55.8</td>
<td>33</td>
<td>46.8</td>
</tr>
<tr>
<td>UnilR-CLIP</td>
<td>42.1</td>
<td>81.1</td>
<td>18</td>
<td>84.7</td>
<td>59.4</td>
<td>78.7</td>
<td>43.1</td>
<td>92.3</td>
<td>18.3</td>
<td>32</td>
<td>45.5</td>
<td>24.4</td>
<td>33.2</td>
<td>44.6</td>
<td>76.2</td>
<td>48.9</td>
<td>50.6</td>
</tr>
<tr>
<td>LamRA-Ret</td>
<td>41.6</td>
<td>81.5</td>
<td>28.7</td>
<td>95.9</td>
<td>62.6</td>
<td>81.2</td>
<td>39.6</td>
<td>90.6</td>
<td>30.4</td>
<td>32.1</td>
<td>54.1</td>
<td>33.2</td>
<td>53.1</td>
<td>79.4</td>
<td>63.3</td>
<td>56.6</td>
<td>56.6</td>
</tr>
<tr>
<td>U-MARVEL</td>
<td>47.3</td>
<td>84.4</td>
<td>33.6</td>
<td>97.1</td>
<td>78.8</td>
<td>88.5</td>
<td>47.3</td>
<td>93.5</td>
<td>35.1</td>
<td>34.2</td>
<td>62.5</td>
<td>58.3</td>
<td>36.4</td>
<td>60.7</td>
<td>79.4</td>
<td>74.7</td>
<td>63.2</td>
</tr>
<tr>
<th colspan="19">+Reranker</th>
</tr>
<tr>
<td>LamRA</td>
<td>48</td>
<td>85.2</td>
<td>32.9</td>
<td>96.7</td>
<td>75.8</td>
<td>87.7</td>
<td>48.6</td>
<td>92.3</td>
<td>36.1</td>
<td>33.5</td>
<td>59.2</td>
<td>64.1</td>
<td>37.8</td>
<td>63.3</td>
<td>79.2</td>
<td>78.3</td>
<td>63.7</td>
</tr>
<tr>
<td>U-MARVEL+</td>
<td>49.4</td>
<td>85.6</td>
<td>34.2</td>
<td>98.5</td>
<td>81.4</td>
<td>89.4</td>
<td>50.5</td>
<td>88.4</td>
<td>37.7</td>
<td>34.7</td>
<td>63.7</td>
<td>62.9</td>
<td>38.2</td>
<td>63.2</td>
<td>80.8</td>
<td>78.9</td>
<td>64.8</td>
</tr>
</tbody>
</table>

关键结论：
1.  单模型U-MARVEL的Avg为63.2，比此前单模型SOTA LamRA-Ret的56.6高出6.6个百分点，提升幅度显著；
2.  单模型U-MARVEL的性能已经非常接近此前两阶段SOTA LamRA的63.7，证明蒸馏策略有效将重排的知识迁移到了单模型中，兼顾了精度和推理效率；
3.  两阶段U-MARVEL+的Avg为64.8，比LamRA高出1.1个百分点，取得了新的SOTA。
### 6.1.2 零样本性能
零样本图文相关任务的性能对比如下（原文Table 8）：

<table>
<thead>
<tr>
<th rowspan="3">Methods</th>
<th colspan="3">$q^t \to c^i$</th>
<th colspan="3">$q^i \to c^t$</th>
<th colspan="2">$(q^i, q^t) \to c^i$</th>
<th>$q^{dialog} \to c^i$</th>
<th>$(q^i, q^t) \to c^i$</th>
<th colspan="2">ITM</th>
</tr>
<tr>
<th>ST4V</th>
<th>U-1K*</th>
<th>Flickr</th>
<th>ST4V</th>
<th>U-1K*</th>
<th>Flickr</th>
<th>CIRCO*</th>
<th>GCIS*</th>
<th>V-Dia*</th>
<th>M-FIQ*</th>
<th>C-Neg</th>
<th>S-Cre*</th>
</tr>
<tr>
<th>R@1</th>
<th>R@1</th>
<th>R@1</th>
<th>R@1</th>
<th>R@1</th>
<th>R@1</th>
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
<th colspan="13">Single model</th>
</tr>
<tr>
<td>CLIP-L</td>
<td>84</td>
<td>52.8</td>
<td>67.3</td>
<td>81.8</td>
<td>68.7</td>
<td>87.2</td>
<td>4</td>
<td>13.3</td>
<td>23.7</td>
<td>17.7</td>
<td>66.7</td>
<td>73</td>
</tr>
<tr>
<td>UnilR-CLIPS F</td>
<td>85.8</td>
<td>75</td>
<td>78.7</td>
<td>84.1</td>
<td>78.4</td>
<td>94.2</td>
<td>12.5</td>
<td>16.8</td>
<td>26.8</td>
<td>39.4</td>
<td>79.9</td>
<td>80.3</td>
</tr>
<tr>
<td>E5-V</td>
<td>86.7</td>
<td>84</td>
<td>79.5</td>
<td>84</td>
<td>82.4</td>
<td>88.2</td>
<td>24.8</td>
<td>18.5</td>
<td>54.6</td>
<td>19.2</td>
<td>83.2</td>
<td>84.7</td>
</tr>
<tr>
<td>MagicLens-L</td>
<td>85.5</td>
<td>59.3</td>
<td>72.5</td>
<td>60.9</td>
<td>24.2</td>
<td>84.6</td>
<td>29.6</td>
<td>16.3</td>
<td>28</td>
<td>22.6</td>
<td>62.7</td>
<td>75.9</td>
</tr>
<tr>
<td>VLM2Vec</td>
<td>90.7</td>
<td>90.8</td>
<td>76</td>
<td>85.8</td>
<td>84.7</td>
<td>90.6</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>79.5</td>
</tr>
<tr>
<td>UniME</td>
<td>97.2</td>
<td>95.9</td>
<td>81.9</td>
<td>93.9</td>
<td>95.2</td>
<td>93.4</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>85</td>
</tr>
<tr>
<td>LamRA-Ret</td>
<td>93.3</td>
<td>95.1</td>
<td>82.8</td>
<td>88.1</td>
<td>94.3</td>
<td>92.7</td>
<td>33.2</td>
<td>18.9</td>
<td>62.8</td>
<td>60.9</td>
<td>79.6</td>
<td>85.8</td>
</tr>
<tr>
<td>U-MARVEL</td>
<td>96.4</td>
<td>96.7</td>
<td>85.4</td>
<td>97.2</td>
<td>93.5</td>
<td>93.3</td>
<td>36.2</td>
<td>19.1</td>
<td>70.3</td>
<td>65.7</td>
<td>84.5</td>
<td>87.9</td>
</tr>
<tr>
<th colspan="13">+Reranker</th>
</tr>
<tr>
<td>LamRA</td>
<td>97.9</td>
<td>98.8</td>
<td>88.1</td>
<td>96.5</td>
<td>98</td>
<td>97.6</td>
<td>42.8</td>
<td>24.8</td>
<td>70.9</td>
<td>63.9</td>
<td>85.9</td>
<td>93.5</td>
</tr>
<tr>
<td>U-MARVEL+</td>
<td>97.8</td>
<td>97.7</td>
<td>88.5</td>
<td>98.9</td>
<td>98.2</td>
<td>95.1</td>
<td>46.0</td>
<td>22.6</td>
<td>77.6</td>
<td>66.3</td>
<td>86.1</td>
<td>93.4</td>
</tr>
</tbody>
</table>

零样本文本到视频检索的性能对比如下（原文Table 9）：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">MSR-VTT</th>
<th colspan="3">MSVD</th>
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
<th colspan="7">Zero-shot (finetuned only with text-image data)</th>
</tr>
<tr>
<td>VLM2Vec</td>
<td>43.5</td>
<td>69.3</td>
<td>78.9</td>
<td>49.5</td>
<td>77.5</td>
<td>85.7</td>
</tr>
<tr>
<td>LamRA</td>
<td>44.7</td>
<td>68.6</td>
<td>78.6</td>
<td>52.4</td>
<td>79.8</td>
<td>87.0</td>
</tr>
<tr>
<td>LLaVE-7B</td>
<td>46.8</td>
<td>71.1</td>
<td>80.0</td>
<td>52.9</td>
<td>80.1</td>
<td>87.0</td>
</tr>
<tr>
<td>U-MARVEL</td>
<td>47.2</td>
<td>72.0</td>
<td>80.2</td>
<td>54.6</td>
<td>80.9</td>
<td>87.7</td>
</tr>
<tr>
<td>U-MARVEL+</td>
<td>47.5</td>
<td>69.7</td>
<td>78.5</td>
<td>53.1</td>
<td>79.9</td>
<td>87.1</td>
</tr>
</tbody>
</table>

关键结论：
1.  零样本图文相关任务中，U-MARVEL在12个任务里9个取得SOTA，尤其是对话检索任务R@1达到70.3，比LamRA-Ret的62.8高出7.5个百分点；
2.  零样本文本到视频检索任务中，U-MARVEL在MSR-VTT和MSVD数据集上均超过现有SOTA，证明模型的跨域泛化能力极强，即使未见过视频数据也能实现良好的跨模态对齐。
## 6.2. 消融实验分析
论文对U-MARVEL的各个组件进行了全链路消融，结果如下（原文Table 15）：

<table>
<thead>
<tr>
<th rowspan="2">Task</th>
<th rowspan="2">Dataset</th>
<th colspan="3">Local</th>
<th colspan="3">Global</th>
</tr>
<tr>
<th>Progressive transition</th>
<th>Hard-neg mining</th>
<th>Distillation</th>
<th>Progressive transition</th>
<th>Hard-neg mining</th>
<th>Distillation</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">$q^t \to c^i$</td>
<td>VisualNews</td>
<td>46.4</td>
<td>47.5</td>
<td>47.3</td>
<td>46.3</td>
<td>47.5</td>
<td>47.2</td>
</tr>
<tr>
<td>MSCOCO</td>
<td>84.0</td>
<td>83.6</td>
<td>84.8</td>
<td>77.3</td>
<td>79.4</td>
<td>72.8</td>
</tr>
<tr>
<td>Fashion200K</td>
<td>34.4</td>
<td>35.6</td>
<td>33.6</td>
<td>34.3</td>
<td>35.4</td>
<td>33.3</td>
</tr>
<tr>
<td>$q^t \to c^t$</td>
<td>WebQA</td>
<td>92.4</td>
<td>95.2</td>
<td>97.1</td>
<td>92.2</td>
<td>95.0</td>
<td>96.7</td>
</tr>
<tr>
<td rowspan="2">$q^t \to (c^t, c^i)$</td>
<td>EDIS</td>
<td>65.6</td>
<td>72.9</td>
<td>78.8</td>
<td>65.4</td>
<td>72.9</td>
<td>78.7</td>
</tr>
<tr>
<td>WebQA</td>
<td>84.6</td>
<td>86.9</td>
<td>88.5</td>
<td>83.8</td>
<td>86.6</td>
<td>87.7</td>
</tr>
<tr>
<td rowspan="3">$q^i \to c^t$</td>
<td>VisualNews</td>
<td>45.7</td>
<td>47.9</td>
<td>47.3</td>
<td>45.4</td>
<td>47.8</td>
<td>47.2</td>
</tr>
<tr>
<td>MSCOCO</td>
<td>93.3</td>
<td>93.1</td>
<td>93.5</td>
<td>93.2</td>
<td>93.1</td>
<td>93.5</td>
</tr>
<tr>
<td>Fashion200K</td>
<td>35.4</td>
<td>35.6</td>
<td>35.1</td>
<td>35.3</td>
<td>35.6</td>
<td>34.9</td>
</tr>
<tr>
<td>$q^i \to c^i$</td>
<td>NIGHTS</td>
<td>31.4</td>
<td>33.0</td>
<td>34.2</td>
<td>31.4</td>
<td>33.0</td>
<td>34.0</td>
</tr>
<tr>
<td rowspan="2">$(q^i, q^t) \to c^t$</td>
<td>OVEN</td>
<td>56.5</td>
<td>58.9</td>
<td>62.5</td>
<td>51.3</td>
<td>52.2</td>
<td>58.3</td>
</tr>
<tr>
<td>InfoSeek</td>
<td>55.1</td>
<td>52.2</td>
<td>58.3</td>
<td>50.8</td>
<td>47.5</td>
<td>52.2</td>
</tr>
<tr>
<td rowspan="2">$(q^i, q^t) \to c^i$</td>
<td>FashionIQ</td>
<td>35.4</td>
<td>34.8</td>
<td>36.4</td>
<td>35.3</td>
<td>34.7</td>
<td>36.0</td>
</tr>
<tr>
<td>CR</td>
<td>58.7</td>
<td>57.2</td>
<td>60.7</td>
<td>5.9</td>
<td>55.2</td>
<td>56.0</td>
</tr>
<tr>
<td rowspan="2">$(q^i, q^t) \to (c^i, c^t)$</td>
<td>OVEN</td>
<td>78.0</td>
<td>80.8</td>
<td>79.4</td>
<td>71.9</td>
<td>73.8</td>
<td>73.1</td>
</tr>
<tr>
<td>InfoSeek</td>
<td>72.7</td>
<td>72.3</td>
<td>74.7</td>
<td>68.5</td>
<td>68.7</td>
<td>69.2</td>
</tr>
<tr>
<td colspan="2">Avg.</td>
<td>60.6</td>
<td>61.7</td>
<td>63.2</td>
<td>58.7</td>
<td>59.9</td>
<td>60.7</td>
</tr>
</tbody>
</table>

可以看到：
1.  渐进式过渡阶段Avg为60.6，已经超过此前的单模型SOTA LamRA-Ret（56.6）；
2.  硬负采样阶段提升1.1到61.7；
3.  蒸馏阶段再提升1.5到63.2，每个组件都带来了稳定的性能提升，验证了所有设计的有效性。
# 7. 总结与思考
## 7.1. 结论总结
本文首次系统性探索了基于MLLM的通用多模态检索模型的全链路设计空间，挖掘出6个对性能影响显著的关键因素，打破了多个领域默认的认知误区。基于这些发现提出的U-MARVEL框架，在有监督和零样本场景下均大幅超越现有SOTA，且实现了两阶段召回-重排性能向单阶段模型的高效迁移，兼顾了精度和推理效率，为UMR领域的模型设计和训练优化提供了系统性的指导，同时大幅降低了UMR模型的落地门槛。
## 7.2. 局限性与未来工作
作者指出的局限性包括：
1.  目前仅支持文本和图像模态，尚未扩展音频、视频等其他模态；
2.  与检索增强生成（RAG）系统的结合尚未充分探索；
3.  实验仅覆盖了4B和7B尺寸的模型，尚未支持更大尺寸的模型和端侧小模型的适配。
    未来工作方向包括：扩展更多模态支持、探索UMR在RAG中的应用、支持更多尺寸的模型、进一步优化训练和推理效率。
## 7.3. 个人启发与批判
### 7.3.1 启发
1.  系统性的消融实验是挖掘性能优化点的有效手段：很多行业默认的方案（如last token提取嵌入）并非最优，通过控制变量的大规模消融可挖掘出大量易被忽略的优化点；
2.  训练策略的优化比单纯堆参数更有效：本文中4B小模型采用U-MARVEL的训练策略，性能即可超过此前7B的SOTA模型，大幅降低了落地成本；
3.  高效蒸馏是兼顾性能和效率的关键路径：将多阶段复杂系统的知识蒸馏到单阶段模型中，可在不增加推理成本的前提下大幅提升性能，是工业界落地的重要方向。
### 7.3.2 潜在改进点
1.  渐进式训练的数据选择可进一步优化：目前采用的NLI和CC3M数据集并非专门为UMR设计，可探索自动生成适配UMR的渐进式训练数据，进一步提升泛化性；
2.  硬负采样的过滤策略可优化：目前采用固定阈值过滤假负样本，可探索动态阈值或智能假负样本检测方法，进一步提升硬负采样的效果；
3.  可探索视频模态的适配：目前的视频检索为零样本性能，若加入少量视频数据进行适配，可进一步提升多模态支持能力。