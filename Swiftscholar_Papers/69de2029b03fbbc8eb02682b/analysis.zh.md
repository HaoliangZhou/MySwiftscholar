# 1. 论文基本信息
## 1.1. 标题
**OSCAR: Optimization-Steered Agentic Planning for Composed Image Retrieval**
核心主题：提出一种优化引导的智能体规划框架，用于解决组合图像检索任务，首次将智能体组合图像检索从启发式搜索过程重构为有理论支撑的轨迹优化问题。
## 1.2. 作者与隶属机构
本文作者来自两个核心合作单位：
- OPPO研究院：Teng Wang、Junjie Wu、Changwang Zhang、Zhaoxiang Wang、Jun Wang
- 上海交通大学：Rong Shan、Jianghao Lin、Tianyi Xu、Jianping Zhang、Wenteng Chen、Weinan Zhang
## 1.3. 发表状态与来源
当前为**预印本**状态，2026年2月9日发布于arXiv，尚未正式发表于学术会议或期刊。
## 1.4. 发表年份
2026年
## 1.5. 摘要
组合图像检索（Composed Image Retrieval, CIR）需要对异构的视觉和文本约束进行复杂推理，现有方法主要分为两类范式：
1.  统一嵌入检索：存在单模型近视问题，无法适配异构的用户查询意图；
2.  启发式智能体检索：依赖试错式的工具编排，存在次优、冗余的问题。
    为此本文提出OSCAR优化引导的智能体规划框架，首次将智能体CIR从启发式搜索重构为有理论支撑的轨迹优化问题，采用全新的离线-在线范式：
- 离线阶段：将原子检索选择和组合建模为两阶段混合整数规划（Mixed Integer Programming, MIP）问题，通过严格的布尔集合运算推导最大化真值覆盖的最优轨迹，存入黄金库作为上下文示范；
- 在线阶段：利用黄金库中的示范引导VLM规划器完成推理，无需迭代试错。
  在3个公开基准和1个私有工业基准上的实验表明OSCAR consistently超过SOTA基线，仅用10%的训练数据就能取得优异性能，证明其学习的是通用规划逻辑而非数据集特定记忆。
## 1.6. 原文链接
- 预印本主页：https://arxiv.org/abs/2602.08603
- PDF链接：https://arxiv.org/pdf/2602.08603v1

# 2. 整体概括
## 2.1. 研究背景与动机
### 2.1.1. 核心问题与重要性
组合图像检索（CIR）是多模态检索领域的核心任务：给定一张参考图像 + 一段修改文本（例如“参考图是红色连衣裙，检索同款式的蓝色连衣裙”），需要从图像库中返回符合修改要求的目标图像。该任务是电商检索、相册搜索、内容推荐等工业场景的核心技术，随着用户查询复杂度提升，其重要性持续增长。
### 2.1.2. 现有研究的空白与挑战
现有CIR方法存在两类根本性缺陷，如下图（原文Figure 1）所示：

![Figure 1: The illustration of limitations of existing image retrieval methods, i.e., (a) single-model myopia of unified embedding retrieval, and (b) suboptimal orchestration of heuristic agentic retrieval.](images/1.jpg)
*该图像是示意图，展示了现有图像检索方法的局限性：第一部分（a）说明了统一嵌入检索的单模型近视问题，第二部分（b）则阐明了启发式代理检索的次优协同问题。*

1.  <strong>单模型近视问题（统一嵌入范式）</strong>：这类方法将参考图像和修改文本融合为单一嵌入向量，在共享空间中做最近邻检索。但用户查询意图差异极大（从风格修改到细粒度属性约束，跨自然图像、时尚等多个域），单一模型无法适配所有场景，比如面向自然图像优化的模型无法识别时尚领域的细粒度属性，反之亦然。
2.  <strong>次优编排问题（启发式智能体范式）</strong>：这类方法用LLM/VLM编排多个工具（重写器、检索器等）多步完成检索，但依赖ReAct等启发式迭代框架，做贪婪的局部决策，没有全局优化目标，存在工具调用冗余、逻辑顺序错误、无法处理集合约束（包含/排除）的问题，且计算成本高、无最优性保证。
### 2.1.3. 论文的创新切入点
本文首次从全局优化的视角重构智能体CIR任务：将智能体的工具调用轨迹转化为可求解的混合整数规划问题，离线计算出训练样本的最优规划轨迹作为示范，在线阶段用这些示范引导VLM规划器直接生成最优规划，无需迭代试错，同时解决了单模型近视和启发式编排次优的问题。
## 2.2. 核心贡献与主要发现
本文的核心贡献可总结为四点：
1.  **优化视角创新**：首次将智能体CIR建模为混合整数规划问题，从启发式搜索转向全局优化，无需人工标注就能数学推导最大化真值覆盖、最小化计算冗余的最优规划轨迹，提供强监督信号。
2.  **集合论组合逻辑**：引入严格的布尔集合运算（并、交、差）组合检索结果，支持显式的包含和保守排除推理，这是单嵌入模型和启发式智能体方法无法实现的能力。
3.  **OSCAR框架**：提出全新的离线-在线范式，将MIP优化和智能体规划打通，离线求解的最优轨迹存入黄金库作为示范，引导VLM在单次推理中完成复杂组合规划。
4.  **实验验证**：在3个公开基准和1个私有工业基准上全面超过SOTA基线，仅用10%的训练数据构建黄金库就能取得优异性能，证明其学习的是通用规划逻辑而非样本记忆。

# 3. 预备知识与相关工作
## 3.1. 基础概念解释
为方便初学者理解，首先对本文涉及的核心专业术语做统一解释：
- <strong>组合图像检索（Composed Image Retrieval, CIR）</strong>：多模态检索的子任务，输入为「参考图像+修改文本」，输出为图像库中符合修改要求的所有目标图像。
- <strong>混合整数规划（Mixed Integer Programming, MIP）</strong>：一类数学优化问题，其决策变量同时包含连续值和离散整数值，本文中所有决策变量都是二进制0-1变量，属于特殊的0-1整数规划问题，可通过专业求解器得到全局最优解。
- <strong>智能体规划（Agentic Planning）</strong>：让大模型（LLM/VLM）作为智能体，自主决策调用哪些外部工具、以什么顺序调用、如何组合工具输出，完成复杂任务。
- <strong>上下文示范（In-Context Demonstration）</strong>：大模型的能力之一，不需要微调，只需在输入中提供几个任务示例，就能模仿示例的逻辑完成同类新任务。
- **布尔集合运算**：本文指对多个检索结果集合做三种操作：并集（UNION，保留任一集合包含的结果，提升召回）、交集（INTERSECT，保留所有集合共有的结果，提升精度）、差集（DIFFERENCE，从A集合中去掉B集合包含的结果，排除无关内容）。
## 3.2. 相关工作与技术演进
CIR领域的技术演进主要分为两条脉络：
### 3.2.1. 统一嵌入检索范式
这类方法的核心思路是将「参考图像+修改文本」融合为单一嵌入向量，在共享嵌入空间中做最近邻检索，技术演进路径为：
1.  早期方法：设计专门的跨模态融合模块（比如TIRG的门控残差融合、ARTEMIS的注意力匹配），在监督数据下微调融合模型和检索编码器。
2.  零样本方向：借助大规模预训练VLM的能力，将图像映射为文本空间的伪词元（比如Pic2Word、SEARLE），无需微调就能在文本空间完成查询组合。
3.  近期进展：基于大规模多模态嵌入模型（比如QQMM-embed、RzenEmbed）直接做跨模态检索，零样本性能大幅提升，但仍然存在单模型近视问题，没有任何一个单模型能在所有场景下取得最优性能。
### 3.2.2. 启发式智能体检索范式
为解决单模型的缺陷，近期研究开始用智能体编排多个工具完成CIR，技术演进路径为：
1.  早期工作：将任务拆分为查询重写、检索、修正等固定步骤，按固定流程执行。
2.  近期工作：用ReAct等迭代式智能体框架，让VLM自主决策下一步调用什么工具，比如AutoCIR的多智能体协作、X^R的跨模态工具编排，灵活性大幅提升，但依赖启发式试错，没有全局最优性保证，工具调用冗余、计算成本高。
## 3.3. 差异化分析
本文与现有两类方法的核心区别为：
1.  与统一嵌入范式相比：OSCAR不依赖单一检索模型，而是编排多个不同能力的检索工具，通过集合运算组合结果，适配异构查询意图，解决单模型近视问题。
2.  与启发式智能体范式相比：OSCAR不用迭代试错，而是通过离线MIP求解得到全局最优的规划轨迹作为示范，引导VLM在单次推理中生成近最优规划，同时保证了规划效果和推理效率。
    本文是首个将MIP全局优化引入智能体CIR任务的工作，填补了启发式智能体无最优性保证的空白。

# 4. 方法论
OSCAR的整体框架采用离线-在线两阶段范式，如下图（原文Figure 2）所示：

![Figure 2: The overall framework of our proposed OSCAR](images/2.jpg)
*该图像是示意图，展示了我们提出的OSCAR框架的整体结构。框架包括原子检索构建、回忆导向的选择性混合整数规划以及精确导向的组合混合整数规划，旨在通过优化机制实现复合图像检索优化。*

离线阶段：对每个训练样本，先构建原子检索空间，再通过两阶段MIP求解最优规划轨迹，存入黄金库；在线阶段：对测试查询，检索黄金库中相似样本的最优轨迹作为上下文示范，引导VLM规划器生成工具调用序列，经集合运算得到候选集，最后用VLM验证器排序得到最终结果。
## 4.1. 问题定义
设组合查询为 $q = \{ q_{img}, q_{txt} \}$，其中 $q_{img}$ 是参考图像，$q_{txt}$ 是修改文本；CIR的目标是从图像库 $\mathcal{I}$ 中检索出符合要求的真值图像集合 $\mathcal{I}^+ \subset \mathcal{I}$。
本文将现有所有检索模型抽象为原子工具集合 $\mathcal{T} = \{f_1, f_2, ..., f_m\}$，每个工具输入查询返回一个候选图像集合，CIR任务被重构为：选择合适的原子工具、调用参数、集合运算方式，组合工具输出得到包含尽可能多真值、尽可能少无关图像的结果集。
## 4.2. 原子检索构建
为实现细粒度规划，OSCAR将最小执行单元定义为**原子检索**，是一个四元组：
$r = (f, \hat{q}, p, k)$
各符号含义：
- $f \in \mathcal{T}$：调用的检索工具（比如多模态嵌入检索器、文本 caption 检索器）
- $\hat{q} = \{ q_{img}, \hat{q}_{txt} \}$：重写后的查询，其中 $\hat{q}_{txt}$ 是VLM将原始修改文本拆分为的具体属性约束
- $p \in \{+, -\}$：检索极性，`+`表示要包含对应属性的图像，`-`表示要排除对应属性的图像
- $k$：返回结果的截断数，即返回top-k个候选
  每个原子检索返回一个候选集合 $S_r \subset \mathcal{I}$。
### 4.2.1. 查询分解与极性标注
OSCAR首先用VLM将原始修改文本分解为多个独立的语义约束，每个约束标注极性：
- 正约束（p=+）：目标图像必须包含的属性，比如“蓝色”、“长袖”
- 负约束（p=-）：目标图像必须排除的属性，比如“没有花纹”、“不要帽子”
  这种极性拆分让后续优化阶段可以用布尔集合运算实现精确的包含/排除，是单检索模型不具备的能力。
### 4.2.2. Top-k截断离散化
截断参数k控制单个检索的召回-精度 trade-off：k越大召回越高但噪声越多，k越小精度越高但可能漏检真值。OSCAR将k离散化为有限的取值集合 $\mathcal{K} = \{k_1, k_2, ..., k_{max}\}$（本文取5到50，步长5），为避免冗余计算，对每个工具和查询，仅用最大k做一次检索，更小k的结果通过切片得到。
最终通过工具、重写查询、极性、k的笛卡尔积，为每个训练样本构建约1182个原子检索，作为后续MIP优化的决策空间。
## 4.3. 第一阶段：召回导向的选择MIP
第一阶段MIP的目标是从所有正原子检索中选出最小的子集，最大化真值图像的覆盖，最小化无关图像的引入，同时保证工具多样性，避免单模型近视。
### 4.3.1. 决策变量定义
- $x_r \in \{0,1\}$：二进制决策变量，=1表示选择正原子检索 $r \in \mathcal{R}^+$，=0表示不选
- $c_i \in \{0,1\}$：辅助变量，=1表示图像 $i$ 被至少一个选中的正原子检索覆盖
- $t_f \in \{0,1\}$：辅助变量，=1表示工具 $f$ 被至少一个选中的原子检索使用
  此外将原子检索按「工具+查询+极性」分组为家族 $\mathcal{F}$，同一个家族的检索仅k不同，返回结果是嵌套的（k越大的集合包含k更小的集合），因此每个家族最多选一个k即可，避免冗余。
### 4.3.2. MIP公式
$$
\begin{array} { r l } { \underset { \{ x _ { r } \} _ { r \in \mathcal { R } ^ { + } } } { \operatorname* { m a x } } } & { \frac { w _ { R } } { | \mathcal { I } ^ { + } | } \displaystyle \sum _ { i \in \mathcal { I } ^ { + } } c _ { i } - \frac { w _ { P } } { | \mathcal { I } ^ { - } | } \displaystyle \sum _ { i \in \mathcal { I } ^ { - } } c _ { i } + \lambda _ { \mathrm { d i v } } \displaystyle \sum _ { f \in \mathcal { T } } t _ { f } } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { r \in F } x _ { r } \leq 1 , \quad \forall F \in \mathcal { F } , } \\ & { x _ { r } , c _ { i } , t _ { f } \in \{ 0 , 1 \} . } \end{array}
$$
#### 目标函数解释
目标函数由三部分组成：
1.  召回最大化项：$\frac{w_R}{|\mathcal{I}^+|} \sum_{i \in \mathcal{I}^+} c_i$，其中 $w_R$ 是召回权重，最大化选中的原子检索覆盖的真值比例。
2.  噪声惩罚项：$\frac{w_P}{|\mathcal{I}^-|} \sum_{i \in \mathcal{I}^-} c_i$，其中 $w_P$ 是精度惩罚权重，最小化选中的原子检索覆盖的无关图像比例。
3.  多样性正则项：$\lambda_{div} \sum_{f \in \mathcal{T}} t_f$，其中 $\lambda_{div}$ 是正则系数，鼓励使用不同的检索工具，避免单模型依赖，解决单模型近视问题。
#### 约束解释
- $\sum_{r \in F} x_r \leq 1$：每个原子检索家族最多选一个k，避免嵌套结果的冗余。
- 所有变量为二进制0-1变量。
### 4.3.3. 输出结果
求解该MIP得到最优的正原子检索子集 $\mathcal{R}_*^+$，其返回结果的并集为召回导向的候选集 $\mathcal{U} = \bigcup_{r \in \mathcal{R}_*^+} S_r$，该集合保证了高真值覆盖，但还包含噪声，需要第二阶段做精度优化。
## 4.4. 第二阶段：精度导向的组合MIP
第二阶段MIP的目标是引入负原子检索，通过布尔集合运算过滤候选集 $\mathcal{U}$ 中的噪声，同时避免误删真值。为保证稳定性和可解释性，OSCAR采用固定的两子句集合运算结构：
$$
S_{final} = \Bigl ( \bigcup_{r \in \mathcal{R}_{**}^+} S_r \Bigr ) \setminus \Bigl ( \bigcap_{r \in \mathcal{R}_*^-} S_r \Bigr )
$$
其中：
- $\mathcal{R}_{**}^+ \subseteq \mathcal{R}_*^+$：第一阶段选出的正原子检索的最优子集
- $\mathcal{R}_*^- \subseteq \mathcal{R}^-$：选出的负原子检索子集
- 负检索用交集（INTERSECT）组合：只有所有负检索都返回的图像才会被排除，实现保守排除，避免单个工具的幻觉导致误删真值。
### 4.4.1. 决策变量定义
- $x_r \in \{0,1\}$：二进制决策变量，=1表示选择原子检索 $r \in \mathcal{R}_*^+ \cup \mathcal{R}^-$
- $u_i \in \{0,1\}$：辅助变量，=1表示图像 $i$ 被至少一个选中的正原子检索覆盖（属于正并集）
- $v_i \in \{0,1\}$：辅助变量，=1表示图像 $i$ 被所有选中的负原子检索覆盖（属于负交集）
- $z_i \in \{0,1\}$：辅助变量，=1表示图像 $i$ 留在最终结果集，满足 $z_i = 1 \iff u_i=1 \land v_i=0$（正并集减去负交集）
### 4.4.2. MIP公式
$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \{ x _ { r } \} _ { r \in \mathcal { R } _ { * } ^ { + } \cup \mathcal { R } ^ { - } } } } & { \displaystyle \sum _ { i \in \mathcal { U } \cap \mathcal { I } ^ { + } } z _ { i } - \lambda _ { r e g } \sum _ { i \in \mathcal { I } ^ { - } } z _ { i } } \\ { \mathrm { s . t . } } & { \displaystyle \sum _ { r \in \mathcal { R } _ { * } ^ { + } } x _ { r } \geq 1 , } \\ & { \displaystyle x _ { r } , z _ { i } \in \{ 0 , 1 \} . } \end{array}
$$
#### 目标函数解释
目标函数由两部分组成：
1.  真值保留项：$\sum_{i \in \mathcal{U} \cap \mathcal{I}^+} z_i$，最大化最终结果集中保留的真值数量。
2.  假阳惩罚项：$\lambda_{reg} \sum_{i \in \mathcal{I}^-} z_i$，其中 $\lambda_{reg}$ 是正则系数，惩罚最终结果集中的无关图像。
#### 约束解释
- $\sum_{r \in \mathcal{R}_*^+} x_r \geq 1$：至少选择一个正原子检索，保证正并集非空。
- 所有变量为二进制0-1变量。
### 4.4.3. 输出结果
求解该MIP得到最优的正负原子检索组合 $(\mathcal{R}_{**}^+, \mathcal{R}_*^-)$，以及对应的工具调用序列、集合运算逻辑，即该训练样本的最优规划轨迹，存入黄金库。
## 4.5. 在线推理流程
离线MIP求解依赖真值信息，无法直接用于测试阶段，OSCAR通过黄金库的上下文示范将最优规划逻辑迁移到推理阶段：
1.  黄金库构建：对每个训练样本，用Qwen3-VL-32B生成参考图像的caption，将修改文本和caption拼接为问题上下文，用Qwen3-Embedding-8B编码为向量作为索引键，对应最优轨迹作为值存入黄金库。
2.  测试阶段：对测试查询，同样生成问题上下文并编码，从黄金库中检索top-N个最相似的最优轨迹作为上下文示范，输入给VLM规划器，让其模仿示范的逻辑生成当前查询的工具调用序列和集合运算逻辑。
3.  验证排序：集合运算得到的候选集输入给VLM验证器，判断每个候选是否符合查询要求，按相关性排序得到最终结果。

# 5. 实验设置
## 5.1. 数据集
本文在4个基准上做实验，包括3个公开CIR基准和1个私有工业相册检索基准，数据集统计如下（原文Table 1）：

<table>
<thead>
<tr>
<th rowspan="2">Split</th>
<th rowspan="2">Num</th>
<th rowspan="2">CIRCO</th>
<th rowspan="2">CIRR</th>
<th colspan="3">FashionIQ</th>
</tr>
<tr>
<th>Dress</th>
<th>Shirt</th>
<th>Toptee</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">Training</td>
<td>#Query</td>
<td>-</td>
<td>28,225</td>
<td>5,985</td>
<td>5,988</td>
<td>6,027</td>
</tr>
<tr>
<td>#Image</td>
<td>-</td>
<td>16,939</td>
<td>11,452</td>
<td>19,036</td>
<td>16,121</td>
</tr>
<tr>
<td rowspan="2">Validation</td>
<td>#Query</td>
<td>220</td>
<td>4,181</td>
<td>2,017</td>
<td>2,038</td>
<td>1,961</td>
</tr>
<tr>
<td>#Image</td>
<td>123,403</td>
<td>2,297</td>
<td>3,817</td>
<td>6,346</td>
<td>5,373</td>
</tr>
<tr>
<td rowspan="2">Testing</td>
<td>#Query</td>
<td>800</td>
<td>4,148</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>#Image</td>
<td>123,403</td>
<td>2,315</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

各数据集特点：
- **CIRCO**：零样本组合图像检索基准，面向开放域自然图像，每个查询对应多个真值，用mAP评估排序质量。
- **CIRR**：真实场景组合图像检索基准，包含用户真实的修改查询，覆盖日常物品、场景等。
- **FashionIQ**：时尚领域CIR基准，包含连衣裙、衬衫、T恤三个子集，查询为时尚属性修改。
- **私有工业基准**：包含3个真实用户相册，共3582张图片、1188个真实用户查询，模拟相册搜索场景。
  本文仅用10%的训练数据构建黄金库，最大程度验证泛化性。
## 5.2. 评估指标
本文用到三类评估指标，分别解释如下：
### 5.2.1. Recall@K（召回率@K）
#### 概念定义
衡量检索结果的前K个中，包含多少比例的真实目标图像，关注模型“找全”目标的能力，值越高越好。
#### 数学公式
$$Recall@K = \frac{| \{ Top-K\text{检索结果} \} \cap \{ \text{真实目标图像集} \} |}{| \{ \text{真实目标图像集} \} |}$$
#### 符号解释
- 分子：前K个检索结果与真实目标集合的交集大小，即前K个中正确召回的目标数量。
- 分母：所有真实目标图像的总数量。
### 5.2.2. mAP@K（平均精度均值@K）
#### 概念定义
衡量检索结果的排序质量，不仅考虑是否召回目标，还考虑目标在结果中的位置：目标排的越靠前，得分越高，适合一个查询对应多个真值的场景（比如CIRCO），值越高越好。
#### 数学公式
首先对每个查询计算平均精度（AP）：
$$
AP@K = \frac{1}{|G|} \sum_{k=1}^K P(k) \cdot \mathbb{1}(rank=k的样本是真值)
$$
其中 $G$ 是该查询的真实目标集合，$P(k) = \frac{\text{前}k\text{个结果中的真值数量}}{k}$ 是前k个结果的精度，$\mathbb{1}(\cdot)$ 是指示函数，满足条件时为1否则为0。
mAP@K是所有查询的AP@K的平均值：
$$
mAP@K = \frac{1}{N} \sum_{i=1}^N AP_i@K
$$
其中 $N$ 是查询的总数量。
### 5.2.3. NDCG@K（归一化折损累积增益@K）
#### 概念定义
考虑不同结果的相关性等级差异和排序位置权重，位置越靠前权重越高，适合工业场景下有不同相关性等级的检索任务，值越高越好。
#### 数学公式
$NDCG@K = \frac{DCG@K}{IDCG@K}$
其中折损累积增益 `DCG@K` 为：
$$
DCG@K = \sum_{k=1}^K \frac{2^{rel_k} - 1}{\log_2(k+1)}
$$
$rel_k$ 是第k个结果的相关性得分，`IDCG@K` 是理想情况下（所有相关结果按相关性从高到低排在最前）的DCG@K，用来做归一化。
## 5.3. 对比基线
本文将OSCAR与四类代表性基线对比，覆盖所有现有CIR范式：
1.  **多模态嵌入模型**：通用大规模多模态检索模型，包括Ops-MM-embedding-v1-7B、RzenEmbed-v2-7B、VLM2Vec、B3-Qwen2-7B、QQMM-embed-v2，是单嵌入范式的SOTA。
2.  **基于Caption的文本嵌入模型**：将图像转为caption，把CIR转为文本检索任务，包括bge-m3、Qwen3-Embedding（0.6B/4B/8B）。
3.  **CIR专用方法**：专门为CIR设计的方法，包括Pic2Word、SEARLE、SEARLE-XL-OTI、CIReVL、LinCIR、LDRE、FiRE，是单嵌入范式下CIR专用的SOTA。
4.  **智能体CIR方法**：基于智能体编排的CIR方法，包括MRA-CIR、AutoCIR、$X^R$，是启发式智能体范式的SOTA。

# 6. 实验结果与分析
## 6.1. 核心结果分析
### 6.1.1. 公开基准结果
公开基准的对比结果如下（原文Table 2）：

<table>
<thead>
<tr>
<th rowspan="2">Type</th>
<th rowspan="2">Method</th>
<th rowspan="2">Training Free</th>
<th colspan="4">CIRCO</th>
<th colspan="4">CIRR</th>
</tr>
<tr>
<th>m@5</th>
<th>m@10</th>
<th>m@25</th>
<th>m@50</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="5">Multimodal Embedding</td>
<td>Ops-MM-v1-7B</td>
<td>✓</td>
<td>13.56</td>
<td>15.96</td>
<td>18.61</td>
<td>19.68</td>
<td>1.90</td>
<td>50.29</td>
<td>66.68</td>
<td>91.64</td>
</tr>
<tr>
<td>RzenEmbed-v2-7B</td>
<td>✓</td>
<td>32.20</td>
<td>34.19</td>
<td>37.41</td>
<td>38.61</td>
<td>19.30</td>
<td>70.36</td>
<td>83.08</td>
<td>96.77</td>
</tr>
<tr>
<td>VLM2Vec</td>
<td></td>
<td>3.40</td>
<td>4.07</td>
<td>5.08</td>
<td>5.74</td>
<td>0.10</td>
<td>26.48</td>
<td>41.35</td>
<td>71.74</td>
</tr>
<tr>
<td>B3_Qwen2_7B</td>
<td>✓</td>
<td>3.67</td>
<td>4.55</td>
<td>5.53</td>
<td>6.13</td>
<td>0.80</td>
<td>37.95</td>
<td>54.39</td>
<td>81.01</td>
</tr>
<tr>
<td>QQMM-embed-v2</td>
<td>✓</td>
<td>45.92</td>
<td>47.13</td>
<td>50.39</td>
<td>51.45</td>
<td>28.98</td>
<td>73.66</td>
<td>82.98</td>
<td>96.72</td>
</tr>
<tr>
<td rowspan="4">Text Embedding</td>
<td>bge-m3</td>
<td>✓</td>
<td>8.15</td>
<td>8.61</td>
<td>9.62</td>
<td>10.21</td>
<td>12.53</td>
<td>34.00</td>
<td>48.63</td>
<td>75.06</td>
</tr>
<tr>
<td>Qwen3-Embed-0.6B</td>
<td>✓</td>
<td>9.55</td>
<td>10.35</td>
<td>11.61</td>
<td>12.33</td>
<td>14.87</td>
<td>39.11</td>
<td>54.53</td>
<td>82.51</td>
</tr>
<tr>
<td>Qwen3-Embed-4B</td>
<td>✓</td>
<td>13.55</td>
<td>14.60</td>
<td>16.33</td>
<td>17.20</td>
<td>22.17</td>
<td>51.57</td>
<td>64.84</td>
<td>88.19</td>
</tr>
<tr>
<td>Qwen3-Embed-8B</td>
<td>✓</td>
<td>16.45</td>
<td>17.23</td>
<td>18.98</td>
<td>19.88</td>
<td>23.21</td>
<td>52.48</td>
<td>66.51</td>
<td>89.71</td>
</tr>
<tr>
<td rowspan="7">CIR Dedicated</td>
<td>Pic2Word</td>
<td></td>
<td>8.72</td>
<td>9.51</td>
<td>10.46</td>
<td>11.29</td>
<td>23.90</td>
<td>51.70</td>
<td>65.30</td>
<td>87.80</td>
</tr>
<tr>
<td>SEARLE</td>
<td></td>
<td>11.68</td>
<td>12.73</td>
<td>14.33</td>
<td>15.12</td>
<td>24.24</td>
<td>52.48</td>
<td>66.29</td>
<td>88.84</td>
</tr>
<tr>
<td>SEARLE-XL-OTI</td>
<td></td>
<td>10.18</td>
<td>11.03</td>
<td>12.72</td>
<td>13.67</td>
<td>24.87</td>
<td>52.31</td>
<td>66.29</td>
<td>88.58</td>
</tr>
<tr>
<td>CIReVL</td>
<td></td>
<td>18.57</td>
<td>19.01</td>
<td>20.89</td>
<td>21.80</td>
<td>24.55</td>
<td>52.31</td>
<td>64.92</td>
<td>86.34</td>
</tr>
<tr>
<td>LinCIR</td>
<td></td>
<td>12.59</td>
<td>13.58</td>
<td>15.00</td>
<td>15.85</td>
<td>25.04</td>
<td>53.25</td>
<td>66.68</td>
<td>-</td>
</tr>
<tr>
<td>LDRE</td>
<td></td>
<td>23.35</td>
<td>24.03</td>
<td>26.44</td>
<td>27.50</td>
<td>26.53</td>
<td>55.57</td>
<td>67.54</td>
<td>88.50</td>
</tr>
<tr>
<td>FiRE</td>
<td></td>
<td>31.03</td>
<td>32.08</td>
<td>34.40</td>
<td>35.50</td>
<td>43.33</td>
<td>74.02</td>
<td>83.51</td>
<td>95.83</td>
</tr>
<tr>
<td rowspan="4">Agentic</td>
<td>MRA-CIR</td>
<td></td>
<td>27.14</td>
<td>28.85</td>
<td>31.54</td>
<td>32.63</td>
<td>37.98</td>
<td>67.45</td>
<td>78.07</td>
<td>93.98</td>
</tr>
<tr>
<td>AutoCIR</td>
<td></td>
<td>24.05</td>
<td>25.14</td>
<td>27.35</td>
<td>28.36</td>
<td>31.81</td>
<td>61.95</td>
<td>73.86</td>
<td>92.07</td>
</tr>
<tr>
<td>X^R</td>
<td></td>
<td>31.38</td>
<td>32.88</td>
<td>35.46</td>
<td>36.50</td>
<td>43.13</td>
<td>73.59</td>
<td>83.09</td>
<td>94.05</td>
</tr>
<tr>
<td>OSCAR (Ours)</td>
<td>✓</td>
<td>56.54</td>
<td>58.53</td>
<td>61.92</td>
<td>62.67</td>
<td>51.18</td>
<td>79.50</td>
<td>87.45</td>
<td>96.56</td>
</tr>
<tr>
<td colspan="2">Relative Improvement (%)</td>
<td></td>
<td>23.13%</td>
<td>24.19%</td>
<td>22.88%</td>
<td>21.81%</td>
<td>18.67%</td>
<td>7.40%</td>
<td>5.25%</td>
<td>-0.22%</td>
</tr>
</tbody>
</table>

FashionIQ数据集的对比结果如下（原文Table 3）：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">Dress</th>
<th colspan="2">Shirt</th>
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
<td>Ops-MM-v1-7B</td>
<td>19.39</td>
<td>38.13</td>
<td>31.45</td>
<td>48.87</td>
<td>27.69</td>
<td>46.51</td>
<td>26.18</td>
<td>44.50</td>
</tr>
<tr>
<td>RzenEmbed-v2-7B</td>
<td>37.38</td>
<td>61.97</td>
<td>45.63</td>
<td>64.97</td>
<td>46.56</td>
<td>67.57</td>
<td>43.19</td>
<td>64.84</td>
</tr>
<tr>
<td>VLM2Vec</td>
<td>4.76</td>
<td>14.77</td>
<td>15.60</td>
<td>30.62</td>
<td>11.37</td>
<td>22.95</td>
<td>10.58</td>
<td>22.78</td>
</tr>
<tr>
<td>B3_Qwen2_7B</td>
<td>8.53</td>
<td>22.31</td>
<td>19.53</td>
<td>34.49</td>
<td>14.99</td>
<td>29.88</td>
<td>14.35</td>
<td>28.89</td>
</tr>
<tr>
<td>QQMM-embed-v2</td>
<td>36.44</td>
<td>60.09</td>
<td>46.07</td>
<td>65.65</td>
<td>46.35</td>
<td>68.49</td>
<td>42.95</td>
<td>64.74</td>
</tr>
<tr>
<td>bge-m3</td>
<td>10.81</td>
<td>23.75</td>
<td>20.66</td>
<td>33.66</td>
<td>15.96</td>
<td>27.33</td>
<td>15.81</td>
<td>28.25</td>
</tr>
<tr>
<td>Qwen3-Embed-0.6B</td>
<td>9.32</td>
<td>19.83</td>
<td>21.00</td>
<td>34.69</td>
<td>15.96</td>
<td>27.84</td>
<td>15.43</td>
<td>27.45</td>
</tr>
<tr>
<td>Qwen3-Embed-4B</td>
<td>12.69</td>
<td>27.02</td>
<td>26.69</td>
<td>40.68</td>
<td>21.88</td>
<td>36.05</td>
<td>20.42</td>
<td>34.58</td>
</tr>
<tr>
<td>Qwen3-Embed-8B</td>
<td>12.64</td>
<td>29.20</td>
<td>28.70</td>
<td>43.13</td>
<td>23.41</td>
<td>36.97</td>
<td>21.58</td>
<td>36.43</td>
</tr>
<tr>
<td>Pic2Word</td>
<td>20.20</td>
<td>40.20</td>
<td>26.20</td>
<td>43.60</td>
<td>27.90</td>
<td>47.40</td>
<td>24.77</td>
<td>43.73</td>
</tr>
<tr>
<td>SEARLE</td>
<td>20.48</td>
<td>43.13</td>
<td>26.89</td>
<td>45.58</td>
<td>29.32</td>
<td>49.97</td>
<td>25.56</td>
<td>46.23</td>
</tr>
<tr>
<td>SEARLE-XL-OTI</td>
<td>21.57</td>
<td>44.47</td>
<td>30.37</td>
<td>47.49</td>
<td>30.90</td>
<td>51.76</td>
<td>27.61</td>
<td>47.91</td>
</tr>
<tr>
<td>CIReVL</td>
<td>24.79</td>
<td>44.76</td>
<td>29.49</td>
<td>47.40</td>
<td>31.36</td>
<td>53.65</td>
<td>28.55</td>
<td>48.60</td>
</tr>
<tr>
<td>LinCIR</td>
<td>20.92</td>
<td>42.44</td>
<td>29.10</td>
<td>46.81</td>
<td>28.81</td>
<td>50.18</td>
<td>26.28</td>
<td>46.48</td>
</tr>
<tr>
<td>LDRE</td>
<td>22.93</td>
<td>46.76</td>
<td>31.04</td>
<td>51.22</td>
<td>31.57</td>
<td>53.64</td>
<td>28.51</td>
<td>50.54</td>
</tr>
<tr>
<td>FiRE</td>
<td>29.60</td>
<td>50.87</td>
<td>39.84</td>
<td>60.06</td>
<td>35.64</td>
<td>57.83</td>
<td>35.03</td>
<td>56.25</td>
</tr>
<tr>
<td>MRA-CIR</td>
<td>31.87</td>
<td>54.23</td>
<td>40.43</td>
<td>60.20</td>
<td>41.25</td>
<td>62.51</td>
<td>37.85</td>
<td>58.98</td>
</tr>
<tr>
<td>AutoCIR</td>
<td>24.94</td>
<td>45.81</td>
<td>34.00</td>
<td>53.43</td>
<td>33.10</td>
<td>55.58</td>
<td>30.68</td>
<td>51.61</td>
</tr>
<tr>
<td>X^R</td>
<td>28.71</td>
<td>52.50</td>
<td>38.91</td>
<td>56.82</td>
<td>43.91</td>
<td>62.57</td>
<td>37.18</td>
<td>57.30</td>
</tr>
<tr>
<td>OSCAR (Ours)</td>
<td>38.47</td>
<td>65.15</td>
<td>44.50</td>
<td>67.52</td>
<td>48.24</td>
<td>71.24</td>
<td>43.73</td>
<td>67.97</td>
</tr>
<tr>
<td>Rel. Improv. (%)</td>
<td>2.92%</td>
<td>5.13%</td>
<td>-3.40%</td>
<td>2.84%</td>
<td>3.61%</td>
<td>4.02%</td>
<td>1.25%</td>
<td>4.82%</td>
</tr>
</tbody>
</table>

核心结论：
1.  **整体性能**：OSCAR在所有数据集上全面超过所有基线，仅用10%的训练数据构建黄金库，且是训练-free的框架，甚至超过了做了域特定微调、闭源LLM的基线。
2.  **与单嵌入方法对比**：OSCAR在CIRCO上mAP@5相对提升23.13%，CIRR上Recall@1相对提升76.60%，FashionIQ上Recall@10提升1.25%，证明通过优化引导的规划和集合运算组合多个工具，比依赖单一统一表示的效果好得多。
3.  **与CIR专用方法对比**：OSCAR无需为CIR设计专门的融合模块或微调组件，仅通过通用工具的优化规划就取得了大幅提升，证明该范式的通用性和可复用性。
4.  **与智能体方法对比**：OSCAR在CIRR上Recall@1相对提升18.67%，FashionIQ上Recall@10提升15.54%，且不需要迭代交互或多轮推理，证明优化引导的示范比启发式试错的规划效果更好、效率更高。
### 6.1.2. 工业基准结果
私有工业相册检索基准的结果如下（原文Table 6）：

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Gallery1</th>
<th colspan="2">Gallery2</th>
<th colspan="2">Gallery3</th>
<th colspan="2">Avg</th>
</tr>
<tr>
<th>N@10</th>
<th>R@10</th>
<th>N@10</th>
<th>R@10</th>
<th>N@10</th>
<th>R@10</th>
<th>N@10</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td>VLM2Vec</td>
<td>44.29</td>
<td>47.78</td>
<td>44.91</td>
<td>48.69</td>
<td>38.96</td>
<td>41.86</td>
<td>42.72</td>
<td>46.11</td>
</tr>
<tr>
<td>B3_Qwen2_7B</td>
<td>42.87</td>
<td>47.32</td>
<td>41.84</td>
<td>45.71</td>
<td>37.49</td>
<td>40.76</td>
<td>40.73</td>
<td>44.60</td>
</tr>
<tr>
<td>Ops-MM-v1</td>
<td>51.78</td>
<td>53.74</td>
<td>48.95</td>
<td>51.44</td>
<td>43.98</td>
<td>46.99</td>
<td>48.24</td>
<td>50.72</td>
</tr>
<tr>
<td>RzenEmbed-v2-7B</td>
<td>51.04</td>
<td>53.79</td>
<td>49.62</td>
<td>51.93</td>
<td>46.23</td>
<td>47.59</td>
<td>48.96</td>
<td>51.10</td>
</tr>
<tr>
<td>QQMM-embed-v2</td>
<td>51.64</td>
<td>54.43</td>
<td>48.42</td>
<td>51.29</td>
<td>44.60</td>
<td>46.63</td>
<td>48.22</td>
<td>50.78</td>
</tr>
<tr>
<td>bge-m3</td>
<td>39.61</td>
<td>41.98</td>
<td>38.36</td>
<td>41.47</td>
<td>39.87</td>
<td>41.41</td>
<td>39.28</td>
<td>41.62</td>
</tr>
<tr>
<td>Qwen3-Embed-0.6B</td>
<td>40.73</td>
<td>42.78</td>
<td>40.11</td>
<td>42.54</td>
<td>39.60</td>
<td>41.11</td>
<td>40.15</td>
<td>42.14</td>
</tr>
<tr>
<td>Qwen3-Embed-4B</td>
<td>43.06</td>
<td>44.93</td>
<td>44.55</td>
<td>46.80</td>
<td>41.48</td>
<td>43.07</td>
<td>43.03</td>
<td>44.93</td>
</tr>
<tr>
<td>Qwen3-Embed-8B</td>
<td>42.92</td>
<td>45.05</td>
<td>44.57</td>
<td>46.86</td>
<td>41.89</td>
<td>43.13</td>
<td>43.13</td>
<td>45.01</td>
</tr>
<tr>
<td>OSCAR (Ours)</td>
<td>56.01</td>
<td>65.28</td>
<td>53.12</td>
<td>57.43</td>
<td>58.70</td>
<td>63.92</td>
<td>55.94</td>
<td>62.21</td>
</tr>
<tr>
<td>Rel.Improve. (%)</td>
<td>8.17%</td>
<td>19.93%</td>
<td>7.05%</td>
<td>10.59%</td>
<td>26.97%</td>
<td>34.31%</td>
<td>14.26%</td>
<td>21.74%</td>
</tr>
</tbody>
</table>

OSCAR在工业场景下平均NDCG@10相对提升14.26%，Recall@10相对提升21.74%，证明该方法能很好地泛化到真实世界的用户相册场景，适配复杂的用户意图和异构的数据分布。
## 6.2. 消融实验与分析
本文做了两类消融实验验证各组件的有效性，结果如下（原文Table 4）：

<table>
<thead>
<tr>
<th rowspan="2">Variants</th>
<th colspan="2">CIRCO</th>
<th colspan="2">CIRR</th>
<th colspan="2">FIQ.Avg</th>
</tr>
<tr>
<th>m@25</th>
<th>m@50</th>
<th>R@10</th>
<th>R@50</th>
<th>R@10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td>OSCAR (Ours)</td>
<td>61.92</td>
<td>62.67</td>
<td>87.45</td>
<td>96.56</td>
<td>43.73</td>
<td>67.97</td>
</tr>
<tr>
<td>w/o Demo.</td>
<td>59.72</td>
<td>59.84</td>
<td>75.01</td>
<td>82.02</td>
<td>46.57</td>
<td>61.62</td>
</tr>
<tr>
<td>w/o Set.Diff.</td>
<td>59.54</td>
<td>59.62</td>
<td>87.32</td>
<td>93.80</td>
<td>46.46</td>
<td>62.08</td>
</tr>
<tr>
<td>w/o Set.Diff. & Set.Int.</td>
<td>58.63</td>
<td>58.66</td>
<td>39.11</td>
<td>54.53</td>
<td>49.44</td>
<td>53.96</td>
</tr>
</tbody>
</table>

### 6.2.1. 最优轨迹示范的作用
去掉黄金库的轨迹示范（w/o Demo.）后，性能普遍下降，尤其在CIRR上R@10掉了12.44%，证明MIP求解的最优轨迹提供了有效的工具选择和组合的信号，引导VLM做出更合理的规划。
### 6.2.2. 集合运算的作用
- 去掉集合差操作（w/o Set.Diff.）后，性能下降，因为无法排除包含负属性的无关图像。
- 同时去掉集合差和集合交操作（w/o Set.Diff. & Set.Int.）后，性能大幅下降，尤其CIRR上R@10掉了48.34%，此时只能做简单的多检索结果并集，无法过滤负样本，证明布尔集合运算对CIR任务的重要性。
## 6.3. 泛化性与鲁棒性分析
### 6.3.1. 不同VLM骨干的泛化性
本文测试了不同VLM作为规划器的性能，结果如下（原文Table 5）：

<table>
<thead>
<tr>
<th>Model</th>
<th>m@10</th>
<th>m@25</th>
<th>m@50</th>
<th>R@10</th>
<th>R@25</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td>QQMM-embed-v2</td>
<td>47.13</td>
<td>50.39</td>
<td>51.45</td>
<td>79.75</td>
<td>90.25</td>
<td>95.26</td>
</tr>
<tr>
<td>Qwen3-VL-4B w/o OSCAR</td>
<td>58.12</td>
<td>59.22</td>
<td>59.33</td>
<td>84.75</td>
<td>88.62</td>
<td>88.75</td>
</tr>
<tr>
<td>Qwen3-VL-4B w/ OSCAR</td>
<td>58.11</td>
<td>61.18</td>
<td>61.91</td>
<td>84.75</td>
<td>93.75</td>
<td>96.12</td>
</tr>
<tr>
<td>Rel.Improv. (%)</td>
<td>-0.01%</td>
<td>3.31%</td>
<td>4.35%</td>
<td>0.00%</td>
<td>5.79%</td>
<td>8.30%</td>
</tr>
<tr>
<td>Qwen3-VL-8B w/o OSCAR</td>
<td>57.85</td>
<td>59.22</td>
<td>59.31</td>
<td>83.75</td>
<td>88.12</td>
<td>88.25</td>
</tr>
<tr>
<td>Qwen3-VL-8B w/ OSCAR</td>
<td>58.28</td>
<td>61.65</td>
<td>62.40</td>
<td>85.38</td>
<td>94.88</td>
<td>97.12</td>
</tr>
<tr>
<td>Rel.Improv. (%)</td>
<td>0.074%</td>
<td>4.10%</td>
<td>5.21%</td>
<td>1.95%</td>
<td>7.68%</td>
<td>10.05%</td>
</tr>
<tr>
<td>Qwen3-VL-32B w/o OSCAR</td>
<td>58.32</td>
<td>59.72</td>
<td>59.84</td>
<td>82.75</td>
<td>86.88</td>
<td>87.00</td>
</tr>
<tr>
<td>Qwen3-VL-32B w/ OSCAR</td>
<td>58.53</td>
<td>61.92</td>
<td>62.67</td>
<td>85.50</td>
<td>94.62</td>
<td>97.50</td>
</tr>
<tr>
<td>Rel.Improv. (%)</td>
<td>0.36%</td>
<td>3.68%</td>
<td>4.73%</td>
<td>3.32%</td>
<td>8.91%</td>
<td>12.07%</td>
</tr>
<tr>
<td>InternVL3.5-38B w/o OSCAR</td>
<td>57.88</td>
<td>60.29</td>
<td>60.62</td>
<td>84.88</td>
<td>90.88</td>
<td>91.75</td>
</tr>
<tr>
<td>InternVL3.5-38B w/ OSCAR</td>
<td>58.74</td>
<td>61.20</td>
<td>61.58</td>
<td>85.50</td>
<td>92.38</td>
<td>93.75</td>
</tr>
<tr>
<td>Rel.Improv. (%)</td>
<td>1.49%</td>
<td>1.51%</td>
<td>1.58%</td>
<td>0.72%</td>
<td>1.65%</td>
<td>2.18%</td>
</tr>
</tbody>
</table>

OSCAR对所有VLM骨干都能带来稳定的性能提升，越强的VLM（工具调用能力越好）提升幅度越大，证明OSCAR是插件式的通用框架，不依赖特定VLM，兼容性强。
### 6.3.2. 示范数量的鲁棒性
本文测试了从黄金库检索的示范数量（1-4个）对性能的影响，结果如下图（原文Figure 3）所示：

![该图像是一个比较不同演示次数下，三组基准任务（CIRCO、CIRR和FashionIQ）性能的柱状图。图中展示了每组任务在不同演示数量下的平均精度（mAP）和召回率（Recall），用不同颜色表示各性能指标。](images/3.jpg)
*该图像是一个比较不同演示次数下，三组基准任务（CIRCO、CIRR和FashionIQ）性能的柱状图。图中展示了每组任务在不同演示数量下的平均精度（mAP）和召回率（Recall），用不同颜色表示各性能指标。*

OSCAR的性能对示范数量不敏感，1-4个示范的性能波动很小，证明黄金库提供的是通用的规划逻辑，而不是样本记忆，仅需1个示范就能传递足够的规划策略。
## 6.4. 案例研究
本文的案例研究如下（原文Figure 4）：

![该图像是示意图，展示了优化驱动的代理规划在组合图像检索中的应用。图中包含参考图像、修改查询和检索图像集，同时展示了规划轨迹的步骤和操作符，涉及的操作包括交集和差集的运算。](images/4.jpg)
*该图像是示意图，展示了优化驱动的代理规划在组合图像检索中的应用。图中包含参考图像、修改查询和检索图像集，同时展示了规划轨迹的步骤和操作符，涉及的操作包括交集和差集的运算。*

没有黄金库示范时，规划器错误地将“移除一个甲虫”理解为检索“一个甲虫”然后做差集，导致目标图像被误删；加入黄金库的相似示范后，规划器正确生成查询“单只甲虫趴着，没有第二只甲虫”，正确检索到目标图像并排在第一位，证明示范能有效引导VLM避免错误的工具调用和集合运算。

# 7. 总结与思考
## 7.1. 结论总结
本文提出OSCAR优化引导的智能体规划框架，首次将智能体组合图像检索从启发式搜索重构为有理论支撑的轨迹优化问题，通过两阶段MIP离线求解最优规划轨迹作为示范，在线引导VLM规划器在单次推理中完成近最优规划，同时解决了单模型近视和启发式编排次优的问题。
在3个公开基准和1个私有工业基准上的实验证明，OSCAR全面超过SOTA基线，仅用10%的训练数据就能取得优异性能，具有极强的泛化性和工业落地价值。该范式为复杂多模态智能体任务的规划提供了全新的思路：用离线优化得到的最优轨迹作为示范，替代启发式试错，同时保证效果和效率。
## 7.2. 局限性与未来工作
### 7.2.1. 作者指出的局限性与未来方向
1.  **失败模式**：当前常见的错误是负证据的极性错配，比如规划器可能错误地将需要排除的“多个甲虫”转为检索“一个甲虫”然后做差集，导致真值被误删，未来可以优化负查询的生成逻辑，提升极性标注的准确性。
2.  **扩展性**：当前的集合运算结构是固定的两子句，未来可以扩展到更灵活的逻辑结构，支持更复杂的组合查询。
3.  **跨任务迁移**：该优化引导的智能体规划范式可以扩展到其他复杂推理任务，比如多跳问答、开放域工具使用等。
### 7.2.2. 潜在的可改进方向
1.  **MIP求解效率**：当前离线阶段需要为每个训练样本求解MIP，虽然仅用10%的训练数据，但对于超大规模数据集仍然有较高的计算成本，未来可以用小模型拟合MIP的求解结果，无需显式求解MIP，进一步降低离线成本。
2.  **黄金库的检索优化**：当前黄金库用文本相似度检索相似轨迹，未来可以加入查询意图的语义分类，更精准地匹配示范轨迹，进一步提升规划准确性。
3.  **动态工具集**：当前工具集是固定的，未来可以支持动态扩展工具，自动适配新的检索模型，无需重新求解MIP。
## 7.3. 个人启发与批判
本文的核心贡献在于首次将运筹学中的整数规划与大模型智能体规划打通，用全局最优的离线示范替代启发式试错，这种范式的通用性很强，不仅可以用于CIR，还可以迁移到所有多工具编排的智能体任务（比如多模态问答、文档问答、机器人决策等），为智能体规划提供了全新的、有理论保证的优化方向。
同时该方法具有极高的工业落地价值：无需微调大模型，仅需少量训练数据生成示范，推理效率高，性能远超现有方法，已经可以直接应用于电商检索、相册搜索等场景。
潜在的风险点在于：当测试查询的分布与黄金库的样本分布差异过大时，示范的引导效果可能下降，未来需要进一步提升跨分布的泛化能力。