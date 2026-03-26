# 1. 论文基本信息
## 1.1. 标题
UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience（UI-Voyager：一种从失败经验中学习的自进化GUI智能体）
核心主题：提出一种无需人工标注的两阶段自进化框架，解决移动GUI智能体从失败轨迹学习效率低、长程任务稀疏奖励下信用分配模糊的痛点，实现高性能移动GUI自动化。
## 1.2. 作者
作者团队包括Zichuan Lin、Feiyu Liu、Yijun Yang等，通讯作者为Deheng Ye和Jie Jiang，全部隶属于腾讯混元（Tencent Hunyuan）团队，核心研究方向为多模态大模型、智能体、强化学习。
## 1.3. 发表状态
本论文为arXiv预印本，尚未正式发表于期刊或会议，发布时间为2026年3月25日。
## 1.4. 发表年份
2026年
## 1.5. 摘要
针对现有移动GUI智能体存在的两大核心问题：从失败轨迹中学习效率低、长程GUI任务稀疏奖励下信用分配模糊，本文提出了新型两阶段自进化移动GUI智能体UI-Voyager：第一阶段采用拒绝微调（Rejection Fine-Tuning, RFT），在完全自主的闭环中实现数据和模型的持续协同进化；第二阶段引入组相对自蒸馏（Group Relative Self-Distillation, GRSD），识别多组推演轨迹中的关键分叉点，从成功轨迹中构建密集的步级监督信号修正失败轨迹。在AndroidWorld基准上的大量实验表明，本文的4B参数模型实现了81.0%的Pass@1成功率，优于众多近期基线，且超过人类水平。消融实验和案例研究进一步验证了GRSD的有效性。该方法无需昂贵的人工数据标注，是向高效、自进化、高性能移动GUI自动化迈进的重大突破。
## 1.6. 原文链接
- 预印本链接：https://arxiv.org/abs/2603.24533v1
- PDF链接：https://arxiv.org/pdf/2603.24533v1
- 代码仓库：https://github.com/ui-voyager/ui-voyager

# 2. 整体概括
## 2.1. 研究背景与动机
### 2.1.1. 核心问题
移动设备已经成为人们日常生活的核心载体，自主移动GUI（图形用户界面）智能体可以自动完成用户指定的手机操作，具有极高的实用价值。但现有方法存在两大难以解决的痛点：
1. **失败轨迹利用效率低**：传统方法要么依赖昂贵的人工标注成功轨迹，要么无法有效从大量失败的交互轨迹中提取有效学习信号，数据利用率极低。
2. **稀疏奖励下信用分配模糊**：长程GUI任务通常只有最终的成功/失败二元奖励（稀疏奖励），传统强化学习方法无法定位序列决策中具体哪一步动作导致了任务失败，无法实现稳定的策略优化。
### 2.1.2. 研究价值
随着多模态大语言模型（MLLM）的发展，GUI智能体的理解能力已经有了显著提升，但训练效率和性能瓶颈一直没有得到解决。本文的研究可以在无需人工标注的前提下，大幅提升GUI智能体的训练效率和性能，推动GUI自动化的大规模落地。
### 2.1.3. 创新思路
本文提出完全自进化的两阶段训练框架：第一阶段用模型自主生成、筛选成功轨迹来迭代微调自身，实现数据和模型的协同进化；第二阶段利用同任务多轨迹的分叉点，直接从成功轨迹中提取步级监督信号修正失败轨迹，从根源上解决稀疏奖励下的信用分配问题。
## 2.2. 核心贡献/主要发现
### 2.2.1. 核心贡献
1. <strong>提出拒绝微调（RFT）机制</strong>：无需人工标注，通过自主生成任务、筛选成功轨迹、迭代微调的闭环，实现训练数据和模型能力的协同进化，样本效率远高于直接用强化学习从基础模型开始训练。
2. <strong>提出组相对自蒸馏（GRSD）方法</strong>：通过SSIM状态匹配识别成功和失败轨迹的分叉点，直接用成功轨迹的正确动作作为步级监督信号，不需要外部模型或标注，彻底解决了长程稀疏奖励任务的信用分配问题。
3. **实现了SOTA性能**：仅4B参数的UI-Voyager在AndroidWorld基准上达到81.0%的Pass@1成功率，超过所有基线模型（包括235B的超大模型），也超过了官方公布的人类水平（80.0%）。
### 2.2.2. 关键结论
1. 无人工标注的自进化框架完全可以实现比人工标注+传统RL更高的GUI智能体性能。
2. 利用同任务多轨迹的分叉点做步级自蒸馏，是解决长程稀疏奖励任务信用分配问题的高效方案。
3. 小参数模型通过针对性的训练框架优化，可以实现远超超大参数模型的GUI任务性能。

# 3. 预备知识与相关工作
## 3.1. 基础概念
为了帮助初学者理解本文，首先对核心专业术语进行解释：
1. <strong>图形用户界面智能体（GUI Agent）</strong>：可以自主理解GUI的视觉内容、进行推理规划、执行点击/滑动/输入等操作，完成用户给定任务的智能体，属于具身智能的分支。
2. <strong>多模态大语言模型（Multimodal Large Language Model, MLLM）</strong>：可以同时处理文本、图像等多种模态输入的大语言模型，是GUI智能体的主干网络（backbone），负责理解屏幕截图（视觉模态）和用户指令（文本模态）。
3. <strong>监督微调（Supervised Fine-Tuning, SFT）</strong>：在预训练大模型的基础上，用标注好的下游任务数据进一步训练，让模型适配下游任务的过程。
4. <strong>强化学习（Reinforcement Learning, RL）</strong>：智能体通过和环境交互，根据获得的奖励反馈调整自身策略的学习范式。GUI智能体和手机环境交互，成功完成任务获得正奖励，失败获得负奖励，属于典型的强化学习场景。
5. <strong>信用分配（Credit Assignment）</strong>：强化学习的经典问题，当智能体执行长序列动作后才获得最终奖励时，如何确定序列中每个动作对最终结果的贡献程度，也就是定位哪一步是对的、哪一步是错的。
6. <strong>近端策略优化（Proximal Policy Optimization, PPO）</strong>：目前最常用的强化学习算法之一，通过限制策略更新的步长，保证训练的稳定性和样本效率，是本文的基线方法之一。
7. <strong>组相对策略优化（Group Relative Proximal Policy Optimization, GRPO）</strong>：针对大模型微调优化的PPO变体，对同一个任务采样一组推演（rollout）轨迹，用组内的相对奖励计算优势函数，不需要单独的奖励模型，适合大语言模型的对齐，是本文的基线方法之一。
8. <strong>结构相似性指数（Structural Similarity Index, SSIM）</strong>：计算机视觉中衡量两张图像相似度的指标，取值范围为0~1，值越大说明两张图像越相似，本文用来判断两个屏幕截图是否对应同一个GUI状态。
9. **AndroidWorld基准**：目前最权威的移动GUI智能体评测基准，包含116个覆盖真实移动应用的多样化任务，任务难度从几步到几十步不等，支持随机化任务参数生成大量训练数据，官方提供基于adb命令的自动验证器，无需人工判断任务是否完成。
10. <strong>推演（rollout）</strong>：强化学习中智能体在环境中执行策略生成完整交互轨迹的过程，本文中也指生成的轨迹本身。
## 3.2. 前人工作
### 3.2.1. GUI智能体的技术演进
1. **早期规则-based智能体**：比如Siri、Cortana等，只能执行预定义的简单任务，泛化能力极差。
2. **MLLM零/少样本智能体**：基于通用MLLM零样本或少量提示执行GUI任务，不需要训练，但效果很差，无法处理复杂任务。
3. **人工标注SFT微调智能体**：用人工标注的GUI交互轨迹做SFT微调，效果有所提升，但人工标注成本极高，难以规模化，泛化能力受限。
4. **强化学习训练智能体**：用PPO、GRPO等强化学习算法训练，不需要人工标注步级动作，但长程稀疏奖励下信用分配问题突出，学习效率极低，性能瓶颈明显。
### 3.2.2. 近期相关工作
近期的EvoCUA等工作已经开始尝试识别轨迹的分叉点，但需要依赖外部的大型VLM生成纠正指令，依赖外部模型，训练成本高。而本文的GRSD不需要任何外部模型，直接用自身生成的成功轨迹作为监督信号，属于自蒸馏，成本更低、效率更高。
## 3.3. 技术演进脉络
GUI智能体的技术发展始终围绕「降低标注成本、提升训练效率、提高泛化性能」三个核心目标：
```
规则预定义 → 通用MLLM零样本调用 → 人工标注SFT微调 → 传统RL训练 → 本文自进化框架
（成本最低/效果最差）               （成本最高/效果一般）      （成本低/效率低） （成本低/效果好/效率高）
```
本文的工作处于当前GUI智能体技术的最前沿，首次实现了无人工标注前提下的高性能自进化。
## 3.4. 差异化分析
与现有方法的核心区别：

| 方法类型 | 人工标注需求 | 信用分配能力 | 外部模型依赖 | 性能上限 |
| :--- | :--- | :--- | :--- | :--- |
| 人工标注SFT | 高 | 有（人工标注步级动作） | 无 | 受标注数据质量/规模限制 |
| 传统RL（PPO/GRPO） | 无 | 差（稀疏奖励无步级信号） | 无 | 低，容易停滞 |
| 依赖外部模型的分叉点方法 | 无 | 较好 | 高（需要外部大模型） | 受外部模型能力限制 |
| 本文UI-Voyager | 无 | 好（自生成步级监督） | 无 | 高，可自主迭代进化 |

# 4. 方法论
本文的核心是两阶段迭代的自进化训练框架，整体流程如下图（原文Figure 2）所示：

![Figure : The whole pipeline of training UIVoyager for mobile GUI tasks. It consists of two iterative sages: (R a ou a vrier t collect high-quality samples for spervised fine-tuning; () Group Relative Sel-Distiation (GRSD), whic identies "ork points" betwee succesful and file trajectoy groups using SMmatching and cot erronous actions o rthereine te poliy $\\pi _ { m }$ through mixed-data training.](images/2.jpg)
*该图像是一个示意图，展示了UI-Voyager在移动GUI任务中训练的整个流程。图中分为两个阶段：第一阶段为多轮拒绝微调（RFT），通过基于规则的验证器收集高质量样本进行监督微调；第二阶段为多轮组相对自蒸馏（GRSD），通过检测关键分叉点和自我纠正，从成功和失败的轨迹组中构建更密集的监督。此方法旨在提高移动GUI自动化的效率与性能。*

## 4.1. 方法原理
核心直觉分为两部分：
1. 模型自己生成的成功轨迹天然是高质量的SFT训练数据，不需要人工标注，通过不断生成新任务、筛选成功轨迹、迭代微调的闭环，可以让数据质量和模型能力互相促进、协同进化。
2. 同一个任务的多条推演轨迹中，成功和失败轨迹的分叉点就是模型决策错误的位置，成功轨迹在该点的动作就是正确的监督信号，不需要额外标注，直接用这些信号做自蒸馏就可以解决稀疏奖励下的信用分配问题。
## 4.2. 核心方法详解
### 4.2.1. 第一阶段：拒绝微调（RFT）
RFT的目标是快速提升模型的基础能力，为后续的GRSD训练提供高质量的初始模型，分为三个核心步骤：
#### 步骤1：轨迹生成
设计种子任务生成器，通过扰动现有任务模板的关键参数（比如时间约束、操作数量、界面元素属性）生成大量全新的多样化任务，然后用当前版本的GUI智能体在AndroidWorld环境中自动执行这些任务，生成大量原始交互轨迹，全程无需人工参与。
#### 步骤2：拒绝采样
对生成的原始轨迹进行筛选，仅保留通过AndroidWorld官方验证器检查的**成功轨迹**，因为成功轨迹中的每一步动作都是正确的，可以直接作为SFT的训练数据，不需要人工标注每一步的正确性。
#### 步骤3：迭代训练
- 初始迭代：用Qwen3-VL-4B-Instruct作为基础模型，生成第一批轨迹，筛选成功轨迹做SFT微调，得到第一代模型。
- 后续迭代：用上一代迭代得到的模型作为新的智能体，生成全新任务的轨迹，筛选成功轨迹后微调得到下一代模型，每一轮都使用全新生成的任务，避免过拟合。
  原文实验显示，经过3轮RFT迭代，模型的Pass@1成功率从初始的37%提升到73.2%，效果远高于直接用RL从基础模型训练。
### 4.2.2. 第二阶段：组相对自蒸馏（GRSD）
RFT阶段只能利用成功轨迹，无法利用大量的失败轨迹，GRSD的目标就是从失败轨迹中提取学习信号，进一步提升模型性能。
#### 现有RL方法的缺陷
首先给出原文中GRPO和PPO的目标函数，分析其固有缺陷：
1. **GRPO目标函数**
   $$
\mathcal { I } _ { G R P O } = \mathbb { E } _ { q , o _ { i } } \left[ \frac { 1 } { \sum _ { i = 1 } ^ { G } \left| o _ { i } \right| } \sum _ { i = 1 } ^ { G } \sum _ { t = 1 } ^ { \left| o _ { i } \right| } \operatorname* { m i n } \Bigl ( r _ { i , t } ( \theta ) \hat { A } _ { i , t } , \mathrm { c l i p } \Bigl ( r _ { i , t } ( \theta ) , 1 - \epsilon _ { \mathrm { l o w } } , 1 + \epsilon _ { \mathrm { h i g h } } \Bigr ) \hat { A } _ { i , t } \Bigr ) \right]
$$
符号解释：
- $\mathbb{E}_{q,o_i}$：对任务$q$和对应的$G$条推演轨迹$o_i$求期望
- $|o_i|$：第$i$条轨迹的步数
- $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})}$：词元级重要性采样比例，$\pi_\theta$是当前待更新策略，$\pi_{\theta_{old}}$是更新前的旧策略
- $\hat{A}_{i,t} = \frac{R^{(i)} - \mathrm{mean}(\{R^{(i)}\}_{i=1}^G)}{\mathrm{std}(\{R^{(i)}\}_{i=1}^G)}$：归一化优势值，其中$R^{(i)}$是第$i$条轨迹的最终奖励（成功为1，失败为0），**同一个轨迹内所有步的优势值完全相同**
- $\epsilon_{low}, \epsilon_{high}$：裁剪阈值，限制策略更新步长，保证训练稳定
2. **PPO目标函数**
   $$
\mathcal { I } _ { P P O } = \mathbb { E } _ { q , o } \left[ \frac { 1 } { | o | } \sum _ { t = 1 } ^ { | o | } \operatorname* { m i n } \Bigl ( r _ { t } ( \theta ) \hat { A } _ { t } ^ { G A E } , \mathrm { c l i p } \Bigl ( r _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon \Bigr ) \hat { A } _ { t } ^ { G A E } \Bigr ) \right]
$$
符号解释：
- $\hat{A}_t^{GAE}$：广义优势估计得到的步级优势值，虽然比GRPO的优势值更平滑，但由于只有最终的稀疏奖励，仍然无法准确区分每一步的对错
- 其他符号含义与GRPO一致
  可以看到，传统RL方法在GUI任务中无法获取步级的奖励信号，同一个轨迹的所有步共享同一个最终奖励，无法定位错误动作，学习效率极低。
#### GRSD核心逻辑
GRSD的核心是从同任务的成功和失败轨迹对中识别分叉点，用成功轨迹的正确动作作为监督信号修正失败轨迹的错误动作，如下图（原文Figure 3）所示：

![该图像是一个示意图，展示了UI-Voyager中关键的叉点检测和成功与失败的路径。图中显示了正向和反向的观察及动作，成功路径以绿色标记，而失败路径以红色标记。通过识别叉点，系统能够在不同的路径间进行自我调整，以提高移动GUI代理的学习效率。](images/3.jpg)
*该图像是一个示意图，展示了UI-Voyager中关键的叉点检测和成功与失败的路径。图中显示了正向和反向的观察及动作，成功路径以绿色标记，而失败路径以红色标记。通过识别叉点，系统能够在不同的路径间进行自我调整，以提高移动GUI代理的学习效率。*

GRSD分为两个核心步骤：叉点检测、步级自蒸馏。
##### 步骤1：叉点检测
输入为同任务的一条成功轨迹$\tau^+ = \{(o_0^+,a_0^+), ..., (o_{T^+}^+, a_{T^+}^+)\}$和一条失败轨迹$\tau^- = \{(o_0^-,a_0^-), ..., (o_{T^-}^-, a_{T^-}^-)\}$，其中$o_t$是第$t$步的屏幕截图，$a_t$是对应动作，目标是找出所有分叉点：两条轨迹处于同一个GUI状态，但选择了不同动作导致后续路径分歧的位置。
叉点检测分为三个子步骤：
1. **跨轨迹状态匹配**
   首先定义观测等价函数，判断两张截图是否属于同一个GUI状态：先用感知哈希过滤相似度低于0.8的明显不相似对，再计算SSIM：
$$
\mathrm { S A M E } \big ( o _ { a } , o _ { b } \big ) = \mathbb { 1 } \big [ \mathrm { S S I M } \big ( \phi \big ( o _ { a } \big ) , \phi \big ( o _ { b } \big ) \big ) \geq \theta \big ]
$$
符号解释：
- $\phi(\cdot)$：截图预处理流程，包括裁剪掉固定的状态栏、导航栏等无关区域、resize、转灰度图，减少干扰
- $\theta$：SSIM相似度阈值，超过阈值则判定为同一个状态
- $\mathbb{1}[\cdot]$：指示函数，条件成立时为1，否则为0
2. **过渡对齐**
   为了保证时间顺序的一致性，在匹配失败轨迹第$j$步之前，先检查是否存在成功轨迹的第$i$步满足：$o_i^+$和$o_j^-$是同一个状态，且$o_{i+1}^+$和$o_{j+1}^-$也是同一个状态，说明两条轨迹的前序路径完全对齐，此时跳过失败轨迹的第$j$步，将最小匹配成功索引$i_{min}$更新为$i+1$，后续的失败步只能匹配$i\geq i_{min}$的成功步。
3. **教师步选择**
   对剩余的失败步$j$，搜索所有$i\geq i_{min}$的成功步，满足两个条件：
- 观测等价：$\mathrm{SAME}(o_i^+, o_j^-)$为真
- 过渡分歧：
  $$
\mathrm { D I V E R G E } ( i , j ) = \left\{ \begin{array} { l l } { \mathbf { t r u e } } & { \mathrm { i f } \ i = T ^ { + } \ \mathrm { o r } \ j = T ^ { - } } \\ { \mathbf { t r u e } } & { \mathrm { i f } \ \mathrm { S S I M } ( \phi ( o _ { i + 1 } ^ { + } ) , \phi ( o _ { j + 1 } ^ { - } ) ) < \theta } \\ { \mathbf { f a l s e } } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$
即成功步$i$的下一个状态和失败步$j$的下一个状态不同，说明动作选择导致了路径分歧，该点就是分叉点。
从所有符合条件的候选教师步中，选择SSIM最高的，SSIM相同则选择索引最小的，保证匹配准确性和时间顺序：
$$
i ^ { * } ( j ) = \arg \operatorname* { m a x } _ { i \in \mathcal { C } ( j ) } \left( \operatorname { S S I M } ( \phi ( o _ { i } ^ { + } ) , \phi ( o _ { j } ^ { - } ) ) , - i \right)
$$
其中$\mathcal{C}(j)$是失败步$j$的候选教师步集合，找到$i^*(j)$后，将$i_{min}$更新为$i^*(j)$，后续失败步只能匹配更靠后的成功步。
##### 步骤2：步级自蒸馏
对每个检测到的叉点$(j, i^*(j))$，构造训练样本：保留失败轨迹第$j$步之前的全部上下文（任务指令、之前的动作和观察）作为提示，将失败轨迹第$j$步的错误动作替换为成功轨迹第$i^*(j)$步的正确动作，作为训练目标：
$$\mathbf { x } _ { j } ^ { \mathrm { t r a i n } } = [ \underbrace { \mathrm { p r o m p t } _ { j } ^ { - } } _ { \mathrm { \text{失败轨迹上下文提示} } } | \underbrace { \mathrm { r e s p o n s e } _ { i ^ { * } ( j ) } ^ { + } } _ { \mathrm { \text{正确动作} } } |$$
训练目标为标准的自回归下一词元预测损失，仅对响应部分的词元计算损失：
$$
\mathcal { L } _ { \mathrm { G R S D } } = - \frac { 1 } { \left| \mathcal { D } \right| } \sum _ { \mathbf { x } \in \mathcal { D } } \frac { 1 } { T _ { \mathbf { x } } } \sum _ { t = 1 } ^ { T _ { \mathbf { x } } } \log \pi _ { \theta } ( y _ { t } \mid s _ { 1 } , \ldots , s _ { P _ { \mathbf { x } } } , y _ { < t } )
$$
符号解释：
- $\mathcal{D}$是所有构造的训练样本集合
- $s_{1:P_\mathbf{x}}$是提示部分的词元，$y_{1:T_\mathbf{x}}$是响应部分的词元
- $P_\mathbf{x}$是提示长度，$T_\mathbf{x}$是响应长度
- $\pi_\theta(y_t | s_{1:P_\mathbf{x}}, y_{<t})$是当前策略在给定提示和之前响应词元的情况下，预测第$t$个响应词元$y_t$的概率
  通过这种方式，模型直接学习到对应上下文下的正确动作，相当于给每个错误步提供了明确的监督信号，彻底解决了信用分配问题。

# 5. 实验设置
## 5.1. 数据集
实验采用AndroidWorld基准，该数据集的特点：
- 包含116个覆盖真实移动应用的多样化任务，涵盖系统设置、浏览器、社交、工具等多个场景，任务步数从几步到30多步不等，难度分布合理。
- 支持随机化任务参数，可生成大量不同的训练任务，避免过拟合，本文训练集包含超过7000个生成的任务。
- 官方提供基于adb命令的规则化验证器，自动判断任务是否完成，不需要人工评估，评测结果可复现。
  选择该数据集的原因：它是目前移动GUI智能体领域最权威、使用最广泛的基准，能够有效验证模型的通用GUI操作能力。
## 5.2. 评估指标
本文采用的核心评估指标为**Pass@K**：
1. **概念定义**：衡量智能体完成任务的成功率，指对同一个任务给$K$次尝试机会，至少成功完成一次的概率。$K=1$时对应一次尝试的成功率，是最常用的指标；$K$越大说明智能体的容错能力越强。
2. **数学公式**
   $$
\text{Pass@K} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}\left[ \exists k \in \{1,...,K\}, \text{第}i\text{个任务的第}k\text{次尝试成功} \right]
$$
3. **符号解释**
- $N$：测试任务的总数量
- $\mathbb{1}[\cdot]$：指示函数，括号内条件成立时取值为1，否则为0
## 5.3. 对比基线
本文的对比基线涵盖了三大类，具有充分的代表性：
1. **通用多模态大模型**：包括Qwen3-VL全系列（2B/4B/8B/32B/235B）、Gemini-2.5-Pro、Seed1.5、Seed1.8等，验证本文方法相对于通用MLLM的性能优势。
2. **专用GUI智能体**：包括UI-Tars系列、MAI-UI系列、Step-GUI系列、GUI-Owl系列、UI-Venus系列、ScaleCUA、Ferret-UI Lite等当前最先进的GUI智能体，验证本文方法相对于专用GUI模型的优势。
3. **人类水平**：AndroidWorld官方公布的人类测试者平均Pass@1为80.0%，验证本文方法的实用性。

# 6. 实验结果与分析
## 6.1. 核心结果分析
本文的核心实验结果如下表（原文Table 2）所示：

| MODEL | #PARAMS | Success Rate |
| :--- | :--- | :--- |
| Baselines | | |
| Qwen3-VL-2B (Bai et al., 2025a) | 2B | 36.4 |
| MAI-UI-2B (Zhou et al., 2025b) | 2B | 49.1 |
| ScaleCUA-3B (Liu et al., 2025) | 3B | 23.7 |
| Ferret-UI Lite-3B (Yang et al., 2025c) | 3B | 28.0 |
| Qwen3-VL-4B (Bai et al., 2025a) | 4B | 45.3 |
| Step-GUI-4B (Yan et al., 2025b) | 4B | 63.9 |
| UI-Tars-7B (Qin et al., 2025) | 7B | 33.0 |
| UI-Tars-1.5-7B (Seed, 2025b) | 7B | 30.0 |
| UI-Venus-7B (Gu et al., 2025) | 7B | 49.1 |
| GUI-Owl-7B (Ye et al., 2025b) | 7B | 66.4 |
| Step-GUI-8B (Yan et al., 2025a) | 8B | 67.7 |
| Qwen3-VL-8B (Bai et al., 2025a) | 8B | 47.6 |
| MAI-UI-8B (Zhou et al., 2025b) | 8B | 70.7 |
| Step-GUI-8B (Yan et al., 2025b) | 8B | 67.7 |
| GUI-Owl-1.5-8B-Thinking (Xu et al., 2026) | 8B | 71.6 |
| UI-Venus-1.5-30B-A3B (Gao et al., 2026) | 30B | 77.6 |
| Qwen3-VL-32B (Bai et al., 2025a) | 32B | 57.3 |
| MAI-UI-32B (Zhou et al., 2025b) | 32B | 73.3 |
| UI-Tars-SFT-72B (Qin et al., 2025) | 72B | 46.6 |
| UI-Venus-72B (Gu et al., 2025) | 72B | 65.9 |
| Seed1.5-VL (Guo et al., 2025b) | - | 62.1 |
| UI-Tars-2 (Wang et al., 2025) | 230B | 73.3 |
| Qwen3-VL-235B-A22B (Bai et al., 2025a) | 235B | 63.7 |
| UI-Tars-1.5 (Seed, 2025b) | - | 64.2 |
| Gemini-2.5-Pro (DeepMind, 2025) | - | 69.7 |
| Seed1.8 (Seed, 2025a) | - | 70.7 |
| MAI-UI-235B-A22B (Zhou et al., 2025b) | 235B | 76.7 |
| Human (Rawles et al., 2025) | - | 80.0 |
| Ours | | |
| UI-Voyager | 4B | **81.0** |

从结果可以看出：
1. UI-Voyager的4B模型达到了81.0%的Pass@1成功率，超过了所有基线模型，包括参数规模是其几十倍的235B超大模型（MAI-UI-235B-A22B仅76.7%），也超过了官方公布的人类水平（80.0%），性能优势非常显著。
2. 与同参数规模的模型相比，UI-Voyager的性能远超Step-GUI-4B（63.9%）和基础Qwen3-VL-4B（45.3%），充分验证了两阶段训练框架的有效性。
   性能对比的可视化结果如下图（原文Figure 1）所示：

   ![Figure 1: Performance comparison of various GUI agents on AndroidWorld. Our UI-Voyager (4B) achieves an $8 1 . 0 \\%$ Pass `@ 1` success rate, outperforming larger models and exceeding reported human-level performance.](images/1.jpg)
   *该图像是图表，展示了不同GUI智能体在AndroidWorld上的成功率比较。UI-Voyager（4B）实现了81.0%的成功率，超过了多个较大模型，亦超越了人类性能。*

## 6.2. 消融实验分析
### 6.2.1. RFT的有效性验证
RFT的性能提升效果如下图（原文Figure 4）所示：

![Figure 4: RFT significantly boosts agent performance. Left: Pass $@ \\mathrm { K }$ performance across four iterative rounds of RFT. The results show consistent improvement in both $\\mathrm { P a s s } @ 1$ and Pass $@ \\mathbf { k }$ as the self-evolution progresses. We select the checkpoint from the third RFT round $( \\mathrm { P a s s } @ 1 = 7 3 . 2 \\% )$ for subsequent training. Right: Training curves of GRPO and PPO initialized from Qwen3-VL-4B-Instruct. The results show that directly deploying RL algorithms from Qwen3-VL-4B-Instruct model yelds marginal gains and exhibits high sample ineffciency.](images/4.jpg)
*该图像是图表，展示了UI-Voyager在不同训练轮次和方法下的成功率。左侧图表显示了在四轮拒绝微调（RFT）中的Pass@$K$成功率表现，随着自我进化的进行，成功率逐步提高。右侧则比较了GRPO和PPO算法的训练过程，表明直接从Qwen3-VL-4B-Instruct模型实施强化学习的效果有限且样本效率低。整体上，RFT显著提升了智能体的性能。*

- 左图显示四轮RFT迭代的Pass@K表现：每一轮迭代都有稳定的提升，三轮之后Pass@1从初始的37%提升到73.2%，Pass@3、Pass@5等指标也同步提升。
- 右图显示直接用GRPO和PPO从基础模型训练的曲线：需要175步才能达到一轮RFT的64%的效果，样本效率远低于RFT，证明RFT是非常高效的初始化方法。
### 6.2.2. GRSD的有效性验证
GRSD与传统RL方法的对比结果如下图（原文Figure 8）所示：

![Figure 8: Training performance comparison of GRSD, GRPO, and PPO. All methods start from the same RFT model with $7 3 . 2 \\%$ success rate. GRSD successfully boosts the agent's Pass `@ 1` performance to $81 \\%$ (Left), while GRPO and PPO show slower progress and plateau around $76 \\%$ (Right). The results demonstrate that GRSD's fork point detection and sel-corection mechanisms enable more effectiv learning compared to standard RL baselines.](images/8.jpg)
*该图像是一个图表，展示了GRSD、GRPO和PPO在训练过程中的表现对比。所有方法均从相同的RFT模型开始，GRSD的Pass@K成功率随着K的增加而显著提升，最终达到约87.5%，而GRPO和PPO的成功率相对平稳，在76%左右。结果表明，GRSD的学习效果优于标准RL基线。*

所有方法都从三轮RFT的73.2%的初始模型开始训练：
- GRSD可以将Pass@1进一步提升到81.0%，提升幅度接近8个百分点。
- GRPO和PPO的提升幅度有限，最终仅达到76%左右就陷入停滞，证明GRSD解决信用分配问题的效果远好于传统RL方法。
### 6.2.3. 困难任务的性能验证
在10个基础模型成功率极低的代表性困难任务上，不同方法的性能对比如下图（原文Figure 9）所示：

![Figure 9: Performance comparison of GRSD, GRPO, and PPO on ten representative low-success-rate tasks. GRSD consistently achieves the highest success rate across all tasks, significantly outperorming both PPO and GRPO. In cntrast, PO nd GRPOstrumakesusanal ais ue thearcy eu sam t lac edi asset eanis. The eult strate hat R ables effnt from failed trajectories even in sparse-reward environments.](images/9.jpg)
*该图像是图表，展示了图9中GRSD、GRPO、PPO和RFT在十个代表性低成功率任务上的成功率比较。结果显示GRSD在所有任务中均取得最高成功率，显著优于其他方法，体现了其在稀疏奖励环境下有效利用失败经验的能力。*

GRSD在所有10个任务上的成功率都显著高于GRPO和PPO，证明GRSD可以有效从失败轨迹中学习，即使在稀疏奖励的困难任务上也有出色的表现。
## 6.3. 案例分析
### 6.3.1. 叉点检测案例1：BrowserMaze任务
如下图（原文Figure 5）所示：

![该图像是示意图，展示了UI-Voyager中两个不同步骤的行动过程。上部分显示在第12步前选择错误路径导致的失败，标记为红色的Fork Point，而下部分则展示了选择正确按钮以顺利完成任务，标记为黄色的成功路径。此图表明了关键决策点对任务结果的重要性。](images/5.jpg)
*该图像是示意图，展示了UI-Voyager中两个不同步骤的行动过程。上部分显示在第12步前选择错误路径导致的失败，标记为红色的Fork Point，而下部分则展示了选择正确按钮以顺利完成任务，标记为黄色的成功路径。此图表明了关键决策点对任务结果的重要性。*

成功和失败轨迹在第12步处于完全相同的迷宫状态，失败轨迹选择向右走被墙挡住，最终任务失败；成功轨迹选择向下走，最终到达终点完成任务。叉点检测可以准确识别该分叉点，为失败轨迹提供正确的动作监督。
### 6.3.2. 叉点检测案例2：SystemBluetoothTurnOff任务
如下图（原文Figure 6）所示：

![Figure 6: Example of fork point detection on SystemBluetoothTurnOff task. The fork point occurs at Step l sat h a aro om Te i j ttt the etti p wanc war swipe while heul trajey use downwar swipe the otti hadeanc quc tts. Forkpoin deteonens this itl diveren, pri corrective supervision at the very first step.](images/6.jpg)
*该图像是一个示意图，展示了在SystemBluetoothTurnOff任务中，失败与成功的轨迹对比。图中标识了分叉点，分别显示了向上滑动与向下滑动的操作，强调在首次步骤提供纠正监督的重要性。*

叉点出现在第0步，两条轨迹都处于主屏幕状态：失败轨迹选择向上滑动进入设置页，路径冗长最终失败；成功轨迹选择向下滑动拉出快捷设置栏，直接关闭蓝牙，快速完成任务。叉点检测可以在第一步就识别到错误，提供纠正监督。
### 6.3.3. 自纠正样本案例
如下图（原文Figure 7）所示：

![该图像是一个示意图，展示了在失败和成功的轨迹中，移动GUI代理的不同操作步骤。其中包含用户提示、任务进度和助手响应，强调了自我纠正的过程。](images/7.jpg)
*该图像是一个示意图，展示了在失败和成功的轨迹中，移动GUI代理的不同操作步骤。其中包含用户提示、任务进度和助手响应，强调了自我纠正的过程。*

失败轨迹的推理为「需要继续向右走然后向下」，执行右走动作后被墙挡住；成功轨迹的推理为「下一步逻辑是直接向下走向右下角」，执行向下动作正确。构造的训练样本会保留失败轨迹的上下文，将响应部分替换为成功轨迹的正确推理和动作，让模型学习到正确的决策逻辑。

# 7. 总结与思考
## 7.1. 结论总结
本文针对移动GUI智能体的两大核心痛点——失败轨迹学习效率低、长程稀疏奖励下信用分配模糊，提出了完全自进化的两阶段训练框架UI-Voyager：
1. RFT阶段实现了无人工标注的闭环迭代，数据质量和模型能力协同进化，样本效率远高于传统RL方法。
2. GRSD阶段通过叉点检测从失败轨迹中提取步级监督信号，彻底解决了稀疏奖励下的信用分配问题，进一步大幅提升性能。
   仅4B参数的UI-Voyager在AndroidWorld基准上达到81.0%的Pass@1成功率，超过所有基线模型和人类水平，为无人工标注的高性能GUI智能体训练提供了全新的可行路径，推动了GUI自动化技术的落地。
## 7.2. 局限性与未来工作
### 7.2.1. 作者指出的局限性
1. **SSIM状态匹配的局限**：一是时间不对齐问题，页面加载、键盘弹出等过渡状态可能导致SSIM偏低，出现漏匹配；二是瞬态视觉干扰，比如光标闪烁、通知弹出等会改变局部像素但不改变语义状态，导致误匹配或漏匹配。
2. **动作空间的局限**：本文采用AndroidWorld的高层抽象动作空间（点击、滑动、输入等），没有考虑底层的触摸动态（比如滑动速度、按压力度等），训练得到的策略迁移到需要细粒度控制的场景时鲁棒性不足。
### 7.2.2. 未来工作方向
1. 优化状态匹配方法：采用短时间窗口的多帧匹配代替单帧匹配，同时屏蔽状态栏、通知栏、键盘等高方差区域，结合OCR和布局信息提升状态匹配的鲁棒性。
2. 分层动作建模：保留高层动作保证样本效率，在后训练阶段加入底层手势和扰动，提升模型的跨设备、跨场景迁移鲁棒性。
3. 扩展到其他GUI场景：本文仅在Android移动GUI上验证了效果，未来可以扩展到桌面GUI、Web GUI、车载GUI等其他场景。
## 7.3. 个人启发与批判
### 7.3.1. 启发
1. 本文的无人工标注自进化思路具有极高的通用性，不仅可以用于GUI智能体，还可以迁移到其他具身智能场景（比如机器人操作、自动驾驶仿真训练），大幅降低数据标注成本，解决长程稀疏奖励任务的训练痛点。
2. GRSD的叉点检测思路不需要额外的奖励模型或人工标注，仅利用同任务多轨迹的分歧点就可以获得步级监督信号，为所有长序列决策任务提供了新的优化思路。
### 7.3.2. 潜在改进点
1. SSIM阈值需要针对不同设备、不同场景手动调优，通用性不足，未来可以用学习到的语义视觉嵌入代替人工设计的SSIM做状态匹配，进一步提升鲁棒性。
2. 目前RFT的任务生成基于现有模板的扰动，任务多样性有限，未来可以结合大模型自动生成完全新颖的任务，进一步提升模型的泛化能力。
3. 目前GRSD需要对同一个任务采样多条轨迹才能找到叉点，样本效率还有提升空间，未来可以研究单条失败轨迹的错误自动定位方法，不需要依赖同任务的成功轨迹。