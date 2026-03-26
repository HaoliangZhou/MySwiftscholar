# 迈向有效的体验学习：利用与内化的双重指导

白飞\*, 陈志鹏\*, 郭欢$\mathrm { H a o } ^ { 2 }$，杨明2， 陶然$\mathrm { T a o } ^ { 2 }$，戴布莱恩2，赵伟鑫1†，杨剑$\mathrm { Y a n g } ^ { 3 }$，许弘腾${ \mathrm { X u } } ^ { 1 }$ 1中国人民大学高龄人工智能学院，21Quest研究所，3北京航空航天大学 \*共同贡献 \*通讯作者 邮箱：feibai@ruc.edu.cn, batmanfly@gmail.com

# 摘要

近年来，强化学习（RL）已成为提升大语言模型（LLM）能力的重要方法。特别是，基于可验证奖励的强化学习（RLVR）作为一个有前景的推理任务范式应运而生。然而，现有的基于强化学习的训练仍然仅是对人类学习的粗略近似。人类学习者利用外部和内部经验来指导探索，并逐渐将有用的轨迹内化为稳定的知识。基于这一差距，我们提出了以下问题：LLM在RLVR训练中如何更好地利用和内化经验？为了解决这个问题，我们提出了双重引导优化（DGO），这是一个统一框架，利用外部和内部经验来提高训练效果。具体而言，DGO首先从先前探索的轨迹中构建经验库。然后，策略在经验库和模型内部知识的共同引导下进行探索。由此产生的轨迹进一步用于优化经验库和模型参数，从而形成一个经验利用和内化的闭环。实验表明，DGO始终优于基线方法，表明更好地利用和内化经验能够导致更有效的推理。

# 1. 引言

最近，强化学习（RL）在提升大语言模型（LLM）的性能方面表现出了巨大的潜力[10, 38]。通过将LLM视为智能体，RL鼓励它们调整行为以获得更高的奖励，这与日益流行的基于可验证奖励的强化学习范式（RLVR）相符。在这一过程中，模型学习正确的推理轨迹，同时避免错误的轨迹，从而逐步提升其推理能力[6, 23]。文献中提出了多种方法来改善RL训练性能，其中经验学习作为这一研究方向的重要技术逐渐显现[25, 26, 35]。经验学习的概念实际上是借鉴自人类学习者。在现实世界中，当人类学习者表现出错误行为时，他们往往会避免重复这些行为，并逐渐积累有效策略以取得更好的表现，这本质上反映了一种经验内化的形式。另一方面，这些经验也可以从外部来源获得，例如书籍或任务领域的其他专家[15]。人类自然发展出一种双重机制，以这种方式利用和内化经验。基于这一学习范式，他们不断进步，逐步扩大自己的知识和解决问题的能力[24]。类比于人类学习，一个理想的强化学习训练算法应在利用和内化经验两个方面都表现出色。然而，现有的工作往往主要集中于这两个机制中的一个。有些方法强调从内部知识中学习[19, 27, 32]，而另一些方法则侧重于利用外部经验进行推理[2, 29, 36]。前者对显性经验的利用重视较少，因此可能存在遗忘的风险，而后者则更依赖于外部知识，可能限制内在能力的发展。因此，我们认为统一这两个方面为持续能力提升提供了一种更具原则性的方向[9]。

![](images/1.jpg)  
Figure 1. Comparison of DGO and baseline methods on six in-domain and out-of-domain benchmarks using Qwen3-4B/8B/14B-Base.

为了解决这一问题，本文提出了双重指导优化（DGO），这是一个统一框架，利用外部和内部经验作为指导，以提高训练效果。DGO将模型演化表述为一个由利用和内化驱动的耦合过程：前者根据双重指导引导探索更好的轨迹，后者则利用这些探索的轨迹优化模型参数。两者共同定义了经验指导探索与参数学习之间的闭环动态。具体而言，这一动态通过三个迭代阶段展开，即经验构建、联合轨迹策略细化和经验内化。首先，我们利用经验生成器从之前收集的轨迹构建经验库。接下来，策略在外部和内部经验的指导下探索给定任务的可能解决方案。最后，探索到的轨迹被重写并提炼为策略参数，以巩固稳定能力。重复这一循环逐步改善模型规模下的经验利用和内化。我们的贡献可总结为以下几点： • 我们显示出有效的经验利用和内化提高了大语言模型的推理能力，并扩大了模型可靠达到的推理行为范围。在第4.2节中，我们观察到随着训练迭代和推理轮次的增加，推理性能得到了提升。 • 我们引入了双重指导优化（DGO），这是一个形式化参量训练与非参量经验之间相互作用的统一框架，为模型学习范式提供了新的视角。 • DGO在Qwen3-8B-Base的六个挑战性基准上，在内在推理下取得了$32.41\%$的最佳平均得分，并且测试时间扩展进一步提高至$39.38\%$，证明了我们方法的有效性。

# 2. 相关工作

经验学习可以分为非参数范式和参数范式。我们将从这两个视角讨论相关工作。非参数学习。非参数学习旨在通过使模型能够利用外部指导（在此称为经验），如来自历史会话的记忆 [5, 22]、从生成解决方案中提炼的可重用策略 [2] 和专家演示 [1]，来改善推理。现有工作主要分为两个类别：1）经验构建与检索，2）经验利用与内化。以往研究主要通过分层经验库 [25, 31]、更新机制和验证策略 [3, 14] 促进了前者。而相对而言，后者的探索仍然较少。尽管一些近期研究探讨了模型在推理过程中如何更好地利用外部经验 [28] 或如何将这种经验内化为模型参数 [37]，但它们往往只强调一个方面。在统一框架中结合利用和内化仍然重要且具有挑战性。

<table><tr><td rowspan="2">Method</td><td colspan="2">Optimization Paradigm</td><td colspan="2">Enhanced Capability</td></tr><tr><td>Non-parameter Parameter</td><td></td><td></td><td>Utilization Internalization</td></tr><tr><td>SFT</td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>×</td><td>:</td><td>×x</td><td>:</td></tr><tr><td>On-Policy Self-Distillation</td><td>✓</td><td></td><td>X</td><td>✓</td></tr><tr><td>Skill-Augmented RL</td><td>✓</td><td></td><td>✓</td><td>X</td></tr><tr><td>DGO (Ours)</td><td>✓</td><td>√</td><td>✓</td><td>✓</td></tr></table>

Table 1. Comparison of representative methods in terms of optimization form and their primary emphasis on experience utilization and internalization. DGO integrates parametric and non-parametric optimization within a unified framework that considers both aspects.

参数学习。参数学习通过直接更新模型参数来强化有效的推理行为并促进能力内化，从而提升大语言模型的推理能力。代表性的方法包括监督微调、强化学习、偏好优化和基于验证器的训练。这些方法在通过显式参数优化来提升推理表现方面显示了强大的有效性。然而，它们通常将高质量的轨迹主要视为参数更新的训练信号，而不是可保留并在当前优化步骤之外加以利用的可重用经验。因此，宝贵的推理轨迹往往未被显式重用，限制了其在未来问题解决中的持续指导潜力。

# 3. 方法

在本节中，我们介绍DGO的详细信息。为清晰起见，表1提供了DGO相对于代表性方法的概念定位。

![](images/2.jpg)  
Figure 2. Our framework consists of three components: 1) Experience Construction, 2) Joint TrajectoryPolicy Refinement, and 3) Experience Internalization, forming a closed-loop process that enhances experience utilization and internalization. I denotes the training iteration, and the final model is the RL policy from the last iteration.

1: 输入：训练集 $\mathcal { D }$ ，初始模型 $\pi _ { \theta ^ { ( 0 ) } }$ ，预构建经验库 $\varepsilon$ ，经验生成器 $G$ ，阶段数 $K$ ，退火计划 $\{ \alpha _ { k } \} _ { k = 1 } ^ { K }$ ，内化比例 $\{ \beta _ { k } \} _ { k = 1 } ^ { K }$ 2: 输出：Finl plcy () 3: $\theta _ { \mathrm { s t a r t } } ^ { ( 1 ) } \theta ^ { ( 0 )}$ 4: $S _ { \mathrm { i m p } } ^ { ( \leq 0 ) } \gets \emptyset , S _ { e } ^ { ( \leq 0 ) } \gets \emptyset$ 5: 对于 $\bar { k } = 1$ 到 $K$ do 6: // 联合轨迹-策略精炼 7: $\varepsilon$ ${ \mathcal { D } } _ { e } ^ { ( k ) }$ 8: 使用退火系数 $\alpha _ { k }$ 构造混合数据 10: $\begin{array} { r l } & { \mathcal { D } _ { \mathrm { R L } } ^ { ( k ) } \alpha _ { k } \mathcal { D } _ { e } ^ { ( k ) } + ( 1 - \alpha _ { k } ) \mathcal { D } _ { \mathrm { f r e e } } ^ { ( k ) } } \\ & { \theta _ { \mathrm { R L } } ^ { ( k ) } \mathrm { U p d a t e } ( \theta _ { \mathrm { s t a r t } } ^ { ( k ) } , \mathcal { D } _ { \mathrm { R L } } ^ { ( k ) } ) } \end{array}$ b 11: 收集推演轨迹 ${ \mathcal { S } } ^ { ( k ) }$ 12: 使用 ${ \mathcal { S } } ^ { ( k ) }$ 和 $G$ 更新经验库 $\varepsilon$ 13: // 经验内化 14: $\begin{array} { r l } & { \widetilde { S } _ { e } ^ { ( k ) } \gets \mathrm { R e w r i t e } ( S _ { e } ^ { ( k ) } ) } \\ & { S _ { \mathrm { i m p } } ^ { ( k ) } \gets \mathrm { F i l t e r } ( \widetilde { S } _ { e } ^ { ( k ) } ) } \\ & { S _ { \mathrm { i m p } } ^ { ( \le k ) } \gets S _ { \mathrm { i m p } } ^ { ( \le k - 1 ) } \cup S _ { \mathrm { i m p } } ^ { ( k ) } } \\ & { S _ { e } ^ { ( \le k ) } \gets S _ { e } ^ { ( \le k - 1 ) } \cup S _ { e } ^ { ( k ) } } \\ & { \mathcal { D } _ { \mathrm { d i s t } } ^ { ( k ) } \gets \beta _ { k } S _ { \mathrm { i m p } } ^ { ( \le k ) } + ( 1 - \beta _ { k } ) S _ { e } ^ { ( \le k ) } } \\ & { \theta _ { \mathrm { s t a r t } } ^ { ( k + 1 ) } \gets \mathrm { D i s t i l l } ( \theta ^ { ( 0 ) } , \mathcal { D } _ { \mathrm { d i s t } } ^ { ( k ) } ) } \\ & { \theta _ { \mathrm { s t a r t } } ^ { ( k ) } \gets \mathrm { D i s t i l l } ( \theta ^ { ( 0 ) } , \mathcal { D } _ { \mathrm { d i s t } } ^ { ( k ) } ) } \end{array}$ 15: 16: 17: 18: 19: 20: // 用于热启动下一个RL迭代 21: 结束循环 22: 返回 ()

# 3.1. 双重引导优化概述

双重引导优化（DGO）基于一个简单的观察：推理可以通过两种互补的引导来源得到改进。外部引导帮助模型探索在当前策略下难以发现的解决轨迹，而内部引导则编码在模型参数中，支持模型自身的推理能力。DGO通过将外部引导下发现的轨迹转化为参数更新，将两者联系起来，从而使有用的推理模式能够逐步内化。形式上，给定输入问题 $x$、经验库 $\varepsilon$ 和策略 $\pi _ { \theta }$，模型在外部和内部引导的共同作用下生成轨迹。

$$
s \sim \pi _ { \theta } ( \cdot \mid x , { \mathcal { E } } ) .
$$

所得的轨迹随后被转化为参数更新的监督信号：

$$
\theta ^ { \prime } \gets \mathrm { U p d a t e } ( \theta ; \mathcal { T } ( s ) ) ,
$$

其中 $\mathcal { T } ( \cdot )$ 表示从生成轨迹到训练信号的转换，例如过滤和重写。通过这一过程，外部和内部指导形成一个闭环反馈机制，外部经验改善轨迹发现，而由此产生的轨迹反过来又提升经验，增强模型的内部指导，同时支持外部经验的更有效利用。

# 3.2. 迭代双重引导训练

基于第3.1节，我们实现了一个迭代训练周期，如图2所示，该周期包括经验构建、联合轨迹-策略优化和经验内化。伪代码见算法1。

# 3.2.1. 经验构建

经验来自历史推理轨迹的提炼，为未来问题解决提供外部指导。我们并不直接重用完整的推理轨迹，而是提炼可迁移的策略和诊断线索，这些策略和线索可以在不同问题间进行概括。每个经验表示为三元组 $\boldsymbol { e } = \left( c , \kappa , \tau \right)$ ，其中 c 指定更广泛的领域或问题阶段，$\kappa$ 确定关键条件、场景或潜在错误，$\tau$ 规定要采取或避免的相应行动。给定一条轨迹 $s$ ，统一经验生成器 $G$ 提取经验为 $e = G ( s )$ 。从正确轨迹导出的经验捕获有效的解决策略，而从错误轨迹导出的经验 $\mathcal { E } = \{ e _ { i } \} _ { i = 1 } ^ { N }$ 为了确保经验库提供可迁移的推理指导，而非特定答案的监督，任何包含或强烈暗示最终答案的经验都将被严格过滤。提取提示显示在表 7 中，示例在表 11 中提供。

# 3.2.2. 联合轨迹-策略优化

在这个阶段，我们采用强化学习（例如，GRPO）来增强经验引导推理，同时不断生成更高质量的轨迹，以便后续的经验内化。通过经验退火进行探索。我们利用强化学习来增强模型利用外部经验的能力，同时逐渐减少对这种指导的依赖。为此，在第 $k$ 次迭代中，训练从经验引导分布 ${ D } _ { e } ^ { ( k ) }$ 和无经验分布 $D _ { i } ^ { ( k ) }$ 的混合中抽取实例，比例为 $\alpha _ { k }$。

$$
D ^ { ( k ) } = \alpha _ { k } D _ { e } ^ { ( k ) } + ( 1 - \alpha _ { k } ) D _ { i } ^ { ( k ) } .
$$

随着训练的进行，$\alpha _ { k }$ 逐渐减小，使得策略在早期阶段更依赖于外部经验以进行引导探索，而在后期阶段逐渐转向较弱的引导。令 $\boldsymbol { z } ~ = ~ ( \boldsymbol { x } , \tilde { e } )$ 表示一个训练实例，其中 $\tilde { e }$ 是用于经验引导实例的检索经验，若没有则为空。该策略在混合训练分布上进行优化，使用如下公式：$\begin{array} { r } { \rho _ { t } ( \theta ) = \frac { \pi _ { \theta } \left( s _ { t } | z , s _ { < t } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( s _ { t } | z , s _ { < t } \right) } } \end{array}$ 这种方法在初期提供较强的引导，同时逐渐减少对外部经验的依赖，而不失去有效使用它的能力。

$$
\begin{array} { l } { \displaystyle \mathcal { L } ( \theta ) = \mathbb { E } _ { z \sim D ^ { ( k ) } , s \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | z ) } \left[ \frac { 1 } { | s | } \sum _ { t = 1 } ^ { | s | } \operatorname* { m i n } \left( \rho _ { t } ( \theta ) A _ { t } , \right. \right. } \\ { \displaystyle \left. \left. \mathrm { c l i p } ( \rho _ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { t } \right) \right] , } \end{array}
$$

通过经历更新进行优化。在每次强化学习迭代 $k$ 后，收集一组推演轨迹 $S^{(k)} = \{ s_j^{(k)} \}$。经验 $G$ 更新经验：${ \mathcal E }_{\mathrm{new}}^{(k)} = G \big ( S^{(k)} \big )$ 选择方式详见附录 D.1：

$$
\mathscr { E } ^ { ( k + 1 ) } = \mathrm { S e l e c t } \Big ( \mathscr { E } ^ { ( k ) } \cup \mathscr { E } _ { \mathrm { n e w } } ^ { ( k ) } \Big ) ,
$$

其中，Select(·) 保留了对下一次迭代最具信息量的经验。更新后的经验库 $\varepsilon ^ { ( k + 1 ) }$ 然后用于构建混合训练分布 $D ^ { ( k + 1 ) }$ 以便后续的策略优化。随着迭代过程中轨迹质量的提高，提取的经验对于指导后续探索变得越来越有用。这形成了一个积极的反馈循环：改进的轨迹产生更高质量的经验，从而支持进一步的轨迹优化，并为后续的经验内化提供更好的数据。

# 3.2.3. 经验内化

经验内化将外部经验所诱导的推理增益通过轨迹重述和策略蒸馏转化为策略参数。通过这种方式，模型逐渐吸收外部经验所鼓励的有用推理模式，提升其独立推理能力，即便没有显式经验。轨迹重述。外部经验可以引导模型产生通过无指导推理难以获得的轨迹。然而，这些轨迹往往包含对所提供经验的显式引用，如果直接用于训练，可能引入不必要的依赖。为了解决这个问题，我们将经验引导的轨迹重写为自包含的推理痕迹，去除显式引用，同时保留其基本原理。给定问题 $x$ 及其附带经验 $\mathcal{E}_{x}$ 和经验引导的轨迹 $s_{e}^{(k)} \in S_{e}^{(k)}$，$\theta_{\mathrm{RL}}^{(k)}$ $p$ 轨迹：

$$
\tilde { s } _ { e } ^ { ( k ) } \sim \pi _ { \theta _ { \mathrm { R L } } ^ { ( k ) } } \Big ( \cdot \mid x , s _ { e } ^ { ( k ) } , \mathcal { E } _ { x } , p \Big ) .
$$

一个 ${ \cal S } _ { e } ^ { ( k ) }$ ${ \tilde { S } } _ { e } ^ { ( k ) }$ 重写轨迹考虑了排除和长度控制，详细设置见附录 D.2。将过滤算子表示为 $\mathcal { F } ( \cdot )$ ，在第 $k$ 次迭代时隐式轨迹集定义为每个问题最多保留 $K$ 个解。

$$
S _ { \mathrm { i m p } } ^ { ( k ) } = \mathrm { T o p - K } \Big ( \mathcal { F } \big ( \tilde { S } _ { e } ^ { ( k ) } \big ) \Big ) ,
$$

策略蒸馏。我们通过对选定轨迹的监督蒸馏，进一步将外部经验内化为模型参数。在第 $k$ 次迭代中，我们将隐式轨迹和经验指导轨迹累积为 $S _ { \mathrm { i m p } } ^ { ( \leq k ) } = \cup _ { i = 1 } ^ { \check { k } } S _ { \mathrm { i m p } } ^ { ( i ) }$ 和 $\textstyle S _ { e } ^ { ( \leq k ) } = \bigcup _ { i = 1 } ^ { k } S _ { e } ^ { ( i ) }$。为了利用外部经验，我们通过混合这两个轨迹集来构建蒸馏数据集：

$$
\mathcal { D } _ { \mathrm { d i s t } } ^ { ( k ) } = \beta _ { k } S _ { \mathrm { i m p } } ^ { ( \leq k ) } + \left( 1 - \beta _ { k } \right) S _ { e } ^ { ( \leq k ) } ,
$$

其中 $\beta _ { k } \in [ 0 , 1 ]$ 控制混合比例。然后，通过最大似然方法优化策略，参数从初始检查点 $\mathbf { \bar { \theta } } ^ { ( 0 ) }$ 初始化，以减轻过拟合问题：

$$
\boldsymbol { \theta } ^ { ( k + 1 ) } = \arg \operatorname* { m a x } _ { \boldsymbol { \theta } } \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { s } ) \sim \mathcal { D } _ { \mathrm { d i s t } } ^ { ( k ) } } \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { s } \mid \boldsymbol { x } ) .
$$

生成的模型被称为内化模型，然后用于初始化下一次迭代。

# 3.3. 推理范式

内在推理。该模式在没有外部指导的情况下评估模型的推理能力。给定一个问题 $x$，模型仅根据其参数知识生成一个推理轨迹：

$$
s \sim \pi _ { \theta } ( \cdot \mid x ) .
$$

Table 2. Performance comparison of our method with other approaches across three model scales on both in-domain and out-of-domain benchmarks. Bold indicates the best result within each model, and underlined indicates the second-best. DGO (zero) and DGO (TTS) denote intrinsic inference and test-time scaling, respectively.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="4">In-domain (%)</td><td colspan="2">Out-of-domain (%)</td><td rowspan="2">Avg</td></tr><tr><td>AIME24</td><td>AIME25</td><td>HMMT25</td><td>MATH500</td><td>GPQA-D</td><td>HLE</td></tr><tr><td rowspan="6">Qwen3-4B-Base</td><td>EGI</td><td>1.46</td><td>1.46</td><td>0.00</td><td>23.52</td><td>12.12</td><td>1.31</td><td>6.64</td></tr><tr><td>SFT</td><td>5.83</td><td>5.62</td><td>1.04</td><td>31.52</td><td>22.92</td><td>2.56</td><td>11.58</td></tr><tr><td>GRPO</td><td>26.88</td><td>22.50</td><td>8.75</td><td>72.75</td><td>42.55</td><td>2.73</td><td>29.36</td></tr><tr><td>DAPO</td><td>26.04</td><td>22.29</td><td>9.17</td><td>71.35</td><td>41.79</td><td>2.34</td><td>28.83</td></tr><tr><td>DGO (zero)</td><td>27.29</td><td>23.54</td><td>11.67</td><td>72.60</td><td>43.56</td><td>3.37</td><td>30.34</td></tr><tr><td>DGO (TTS)</td><td>40.21</td><td>27.71</td><td>15.83</td><td>78.35</td><td>52.53</td><td>4.45</td><td>36.51</td></tr><tr><td rowspan="6">Qwen3-8B-Base</td><td>EGI</td><td>6.04</td><td>4.79</td><td>2.08</td><td>41.50</td><td>22.22</td><td>3.12</td><td>13.29</td></tr><tr><td>SFT</td><td>15.00</td><td>11.46</td><td>3.54</td><td>70.38</td><td>36.30</td><td>4.34</td><td>23.50</td></tr><tr><td>GRPO</td><td>29.79</td><td>20.42</td><td>9.17</td><td>73.20</td><td>44.32</td><td>3.95</td><td>30.14</td></tr><tr><td>DAPO</td><td>25.00</td><td>19.17</td><td>9.17</td><td>80.77</td><td>45.83</td><td>4.66</td><td>30.77</td></tr><tr><td>DGO (zero)</td><td>30.21</td><td>23.33</td><td>11.88</td><td>75.25</td><td>49.75</td><td>4.02</td><td>32.41</td></tr><tr><td>DGO (TTS)</td><td>49.38</td><td>30.63</td><td>14.37</td><td>80.42</td><td>54.99</td><td>6.51</td><td>39.38</td></tr><tr><td rowspan="6">Qwen3-14B-Base</td><td>EGI</td><td>8.33</td><td>5.42</td><td>1.67</td><td>47.02</td><td>28.91</td><td>3.29</td><td>15.77</td></tr><tr><td>SFT</td><td>18.12</td><td>13.33</td><td>6.46</td><td>74.58</td><td>37.12</td><td>4.96</td><td>25.76</td></tr><tr><td>GRPO</td><td>38.12</td><td>27.08</td><td>14.37</td><td>89.00</td><td>45.01</td><td>5.15</td><td>36.45</td></tr><tr><td>DAPO</td><td>35.83</td><td>26.25</td><td>14.17</td><td>89.50</td><td>49.37</td><td>5.45</td><td>36.76</td></tr><tr><td>DGO (zero)</td><td>38.54</td><td>29.17</td><td>15.21</td><td>90.60</td><td>50.13</td><td>5.63</td><td>38.04</td></tr><tr><td>DGO (TTS)</td><td>54.48</td><td>39.38</td><td>22.29</td><td>93.17</td><td>51.64</td><td>7.71</td><td>44.78</td></tr></table>

迭代经验指导的测试时缩放。给定一个问题 $x$，模型进行 $K$ 轮轨迹生成。在每一轮 $k$ 中，它生成 $N$ 条轨迹，从中经验生成器 $G$ 提取在线经验 $\mathcal { E } _ { \mathrm { o n l i n e } } ^ { ( k ) }$ 和离线经验 $\mathcal { E } _ { \mathrm { o f f l i n e } }$ 为冷启动提供支持，而后续轮次使用从上一轮提取的在线经验。

$$
s ^ { ( k ) } \sim \left\{ \begin{array} { l l } { \pi _ { \theta } ( \cdot \mid x , \mathcal { E } _ { \mathrm { o f f i n e } } ) , \quad k = 1 , } \\ { \pi _ { \theta } ( \cdot \mid x , \mathcal { E } _ { \mathrm { o n l i n e } } ^ { ( k - 1 ) } ) , \quad k > 1 . } \end{array} \right.
$$

# 4. 实验

在本节中，我们首先详细介绍实验设置，然后报告结果和详细分析。

# 4.1. 实验设置

训练。我们使用三种规模的 Qwen3 模型（4B、8B 和 14B）作为策略主干，并从 SkyWork 中抽取 9,600 个问题用于强化学习训练。训练包括三次基于 GRPO 的迭代，伴随迭代经验更新和比例退火：经验比例 $\alpha$ 从 1.0 逐渐降低到 0.5 和 0.25，而内部化比例 $\beta _ { k }$ 固定为 0.5。在每次迭代后，选定的轨迹被重写为结构化经验，并提炼为策略。对于经验生成器，我们从 DeepSeek v3.2 中提炼 40,000 个样本，并对 Qwen3-8B-Base 进行微调，该模型在整个训练过程中使用。详细信息见附录 A。

![](images/3.jpg)  
Figure 3. Accuracy of Qwen3-8B-Base across successive TTS rounds on AIME24 and AIME25, compared with DGO and other baseline methods.

评估。我们在领域内基准测试上评估我们的方法，包括 AIME24、AIME25、HMMT25- Feb 和 MATH500 [12]，以及领域外基准测试，包括 GPQA-Diamond [21] 和 HLE-Verified [34]。除非另有说明，我们使用温度 1.0，top- $p \ 1 . 0$ ，最大序列长度为 16K，并报告 $\mathrm { a v g } @ 1 6$ 。我们考虑两种设置：内在推理，用于测量独立推理，以及测试时间缩放（TTS）[13]，用于评估基于经验的推理。对于 TTS，我们最多使用 $K = 5$ 次迭代， 每次迭代总结最多 $N = 4$ 个经验，并通过与 all-MiniLM-L6-v2 的密集向量相似性检索经验。基线。我们将我们的方法与训练无关和基于训练的基线进行比较。作为一个代表性的训练无关基线，我们考虑简单的经验引导推理（EGI），该方法在推理时根据外部经验对模型进行条件设置，而不更新其参数。对于基于训练的基线，我们包括 SFT、GRPO [23] 和 DAPO [32]，这些方法仅通过参数优化提高推理性能，而不在训练期间融合外部经验。

# 4.2. 主要结果

DGO 在各模型规模中持续增强内在推理。如表 2 所示，DGO 在包括 SFT 和基于强化学习的方法在内的强基线中始终表现优越，4B 模型的平均得分为 $30.34\%$，8B 模型为 $32.41\%$，14B 模型为 $38.04\%$，在挑战性的数学基准（如 AIME25 和 HMMT25）以及更广泛的基准（如 MATH500）上均有提升。这些结果表明，更有效的内在化有助于 DGO 更强的内在推理能力。DGO 使经验的利用更加有效。融入 TTS 显著提升了零-shot 推理的效果。例如，在 4B 模型上，平均得分从 $30.34\%$ 上升至 $36.51\%$，在 14B 模型上从 $38.04\%$ 上升至 $44.78\%$，这表明 DGO 训练的模型能够有效地从测试时的经验中获益。此外，如图 3 所示，随着连续 TTS 回合的进行，性能持续改善，并始终超过其他方法。关于经验利用的更详细分析见第 5.1 节。DGO 显示出跨领域的泛化能力。尽管仅在数学数据上进行训练，DGO 在知识密集型基准（如 GPQA-Diamond 和 HLE-Verified）上仍表现出竞争力。DGO（零）和 DGO（TTS）在这些任务上表现良好，表明两者存在互补效应。首先，内化在数学领域的推理经验可以支持更广泛的问题解决。其次，利用先前推理经验的能力也可能跨领域转移。

Table 3. Ablation study on Qwen-4B-Base. EA, ER, TR, and PD denote Experience Annealing, Experience Renewal, Trajectory Rephrasing, and Policy Distillation respectively. Experiments are conducted between iteration 1 and 2.   

<table><tr><td rowspan="2">Method</td><td colspan="4">In-domain (%)</td><td colspan="2">Out-of-domain (%)</td><td rowspan="2">Avg</td></tr><tr><td>AIME24</td><td>AIME25</td><td>HMMT25</td><td>MATH500</td><td>GPQA-D</td><td>HLE</td></tr><tr><td>DGO (zero)</td><td>27.08</td><td>21.04</td><td>9.17</td><td>70.08</td><td>40.47</td><td>2.99</td><td>28.47</td></tr><tr><td>w/o EA</td><td>19.38</td><td>17.50</td><td>7.71</td><td>68.08</td><td>38.64</td><td>3.24</td><td>25.76</td></tr><tr><td>w/o ER</td><td>19.58</td><td>18.75</td><td>7.50</td><td>68.53</td><td>38.57</td><td>2.90</td><td>25.97</td></tr><tr><td>w/o TR</td><td>22.08</td><td>17.50</td><td>6.04</td><td>66.65</td><td>39.90</td><td>2.47</td><td>25.77</td></tr><tr><td>w/o PD</td><td>23.75</td><td>20.83</td><td>7.71</td><td>67.73</td><td>38.26</td><td>2.79</td><td>26.84</td></tr><tr><td>DGO (TTS)</td><td>40.83</td><td>26.67</td><td>15.42</td><td>73.90</td><td>52.65</td><td>4.58</td><td>35.68</td></tr><tr><td>w/o EA</td><td>37.71</td><td>24.58</td><td>13.12</td><td>70.43</td><td>48.30</td><td>3.37</td><td>32.92</td></tr><tr><td>w/o ER</td><td>42.92</td><td>23.75</td><td>15.21</td><td>70.78</td><td>48.42</td><td>3.11</td><td>34.03</td></tr><tr><td>w/o TR</td><td>31.87</td><td>20.42</td><td>11.67</td><td>72.10</td><td>48.11</td><td>3.99</td><td>31.36</td></tr><tr><td>w/o PD</td><td>34.58</td><td>22.71</td><td>15.00</td><td>67.00</td><td>49.31</td><td>3.20</td><td>31.97</td></tr></table>

# 4.3. 消融研究

在本节中，我们进行消融研究以评估DGO中关键设计选择的贡献。

# 4.3.1. 组件消融

我们进行组件级别的消融实验，以评估 DGO 中每个组件的贡献。如表 3 所示，去除四个组件中的任何一个都会导致性能显著下降，表明每个组件对整体有效性都是重要的。经验更新使得经验池在整个训练过程中保持信息丰富和多样化。去除经验更新后，DGO（zero）的平均得分从 $28.47\%$ 降低至 $25.97\%$，DGO（TTS）的平均得分从 $35.68\%$ 降低至 $34.03\%$，这表明保持新鲜经验能支持更强的后续轨迹和更有效的内化。轨迹重构通过去除明确复制经验的噪声模式来改善训练轨迹，使其更易于内化。去掉轨迹重构后，DGO（zero）的平均得分从 $28.47\%$ 降低至 $25.77\%$，DGO（TTS）的平均得分从 $35.68\%$ 降低至 $31.36\%$。这也表明模型能够自行实现有效的轨迹清理，而无需更强大的外部重写器。经验退火和策略蒸馏对将外部经验转化为持久的参数收益至关重要。退火减少了对显式经验的过度依赖，鼓励模型吸收潜在的推理模式，而蒸馏则将这些收益巩固到策略中。去除退火后，DGO（zero）的平均得分从 $28.47\%$ 降低至 $25.76\%$，DGO（TTS）的平均得分从 $35.68\%$ 降低至 $32.92\%$，去除蒸馏也会导致明显的性能下降。

# 4.3.2. 迭代训练的作用

我们进一步研究了迭代训练本身的作用。如图4所示，从第1次迭代到第3次迭代，每轮迭代的初始准确率越来越高，并且在大多数训练步骤中保持了更强的性能曲线。这表明，早期迭代中积累的经验为后续训练提供了越来越有价值的指导，突显了迭代训练在DGO中的重要性。附录C.4提供了一个案例研究，展示了经验质量如何在各次迭代中得到改善。

![](images/4.jpg)  
Figure 4. Accuracy curves of two models on AIME25 across training iterations.

![](images/5.jpg)  
Figure 5. Experience utilization under three settings on AIME24. DGO benefits more from relevant experience and is more robust to irrelevant experience. Numbers denote relative improvement over zero experience.

# 5. 分析

在本节中，我们从利用和内化两个互补方面分析经验在我们方法中的作用。我们还在附录 C.2 和附录 C.3 中提供案例研究，以直观证据支持这两种能力。

# 5.1. 经验在利用中的作用

即使是高质量的经验，如果模型无法有效利用，仍然无法充分发挥其价值[8]。我们在三种设置下比较了DGO和GRPO训练的模型：没有经验、具有噪声的经验和具有相关经验。相关经验是从同一问题的历史轨迹中总结而来的，而噪声经验则来自与该问题无关的情况。在每种设置中为两个模型提供相同的经验。DGO更有效地利用相关经验。如图5所示，添加相关经验对DGO带来了显著更大的收益。在Qwen-4B-Base上，DGO的改进为$2 8 . 3 \%$，而GRPO为$1 4 . 7 \%$；在Qwen-8B-Base上，DGO的收益为$3 7 . 9 \%$，与GRPO的$1 8 . 9 \%$相比更为突出。这表明DGO在从相关经验中提取有用信号方面表现更佳。DGO对噪声经验更具鲁棒性。在两个模型规模上，注入噪声经验对DGO的降级明显小于GRPO，并且在Qwen-8B-Base上，DGO的性能几乎没有变化。这表明，DGO不仅从有益的经验中获益更多，而且还表现得更加稳健。

![](images/6.jpg)  
Figure 6. t-SNE visualization of generated trajectories on AIME24 for the DGO- and GRPO-trained models.

![](images/7.jpg)  
Figure 7. Comparison of the DGO- and GRPO-trained models on AIME24 in terms of Pass $@ \mathrm { K }$ Accuracy.

# 5.2. 经验在内化中的作用

经验内化指的是将有用的推理模式吸收为模型参数，而不是在推理时依赖于显式经验。这种效应反映在表2中，在没有外部经验的内在推理下，DGO始终优于GRPO。图4进一步显示，后期训练迭代从更高的初始准确度开始，并维持更好的性能，这表明模型逐步保留了早期回合的经验。综合这些结果表明，DGO不仅改善了经验的利用，还提高了可重复推理模式的内化。为了进一步检查内化的内容，我们从两个行为层面比较DGO和GRPO。首先，我们分析它们在语义空间中的轨迹。遵循之前的研究[4]，稀有轨迹指的是DGO轨迹中与GRPO分布最远的$10\%$，该距离是指与其$k=5$个最近邻的平均距离。其次，我们使用Pass $@ \mathrm{K}$来测试这些行为是否对应于在采样期间可达的有效解决模式[33]。DGO拓宽了有效推理模式的范围。如图6所示，DGO生成的轨迹在语义上与GRPO分布隔离，这表明其推理行为超出了单靠标准强化学习所诱发的行为。图7进一步表明，在各种采样预算下，DGO始终实现了高于GRPO的Pass $@ \mathbf{K}$，这表明这些行为对应于生成期间可达的有效解决模式。

# 6. 结论

在本论文中，我们提出了DGO，这是一个统一框架，通过外部和内部经验的双重指导来改善大语言模型的推理。DGO将经验构建、联合轨迹-策略优化和经验内化整合为一个迭代过程，使模型能够更好地利用外部经验，并逐步将有用的推理模式吸收到其参数中。多个基准上的实验表明，DGO在内在推理和测试时扩展性能方面持续有所提升。未来的工作包括使模型能够更自主地管理和更新自身经验，以及更有效地协调外部和内部经验。这可能进一步提高大语言模型中经验学习的效率、鲁棒性和可持续性。项目链接：https://github.com/RUCAIBox/DualGuidanceOptimization。

# References

1 Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askel, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:18771901, 2020.   
2 Zhicheng Cai, Xinyuan Guo, Yu Pei, Jiangtao Feng, Jinsong Su, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, and Hao Zhou. Flex: Continuous agent evolution via forward learning from experience. arXiv preprint arXiv:2511.06449, 2025.   
3 Zouying Cao, Jiaji Deng, Li Yu, Weikang Zhou, Zhaoyang Liu, Bolin Ding, and Hai Zhao. Remember me, refine me: A dynamic procedural memory framework for experience-driven agent evolution. arXiv preprint arXiv:2512.10696, 2025.   
4 Daixuan Cheng, Shaohan Huang, Xuekai Zhu, Bo Dai, Wayne Xin Zhao, Zhenliang Zhang, and Furu Wei. Reasoning with exploration: An entropy perspective. arXiv preprint arXiv:2506.14758, 2025.   
5Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Ydav. Mem: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.   
JiDeng, Ji hen, Zhipeg Chen, Dain Cheng,Fei Bai, Beichn Zhang Yan Min, Yan Gao, Wayne Xin Zhao, and Ji-Rong Wen. From trial-and-error to improvement: A systematic analysis of llm exploration mechanisms in rlvr. arXiv preprint arXiv:2508.07534, 2025.   
Aniket Didolkar, Nicolas Ballas, Sanjeev Arora, and Anirudh Goyal. Metacogitive reuse: Turning recurring llm reasoning into concise behaviors. arXiv preprint arXiv:2509.13237, 2025.   
8 Shihan Dou, Ming Zhang, Zhangyue Yin, Chenhao Huang, Yujiong Shen, Junzhe Wang, Jiayi Chen, Yuchen Ni, Junjie Ye, Cheng Zhang, Huaibing Xie, Jianglu Hu, Shaolei Wang, Weichao Wang, Yanling Xiao, Yiting Liu, Zenan Xu, Zhen Guo, Pluto Zhou, Tao Gui, Zuxuan Wu, Xipeng Qiu, Qi Zhang, Xuanjing Huang, Yu-Gang Jiang, Di Wang, and Shunyu Yao. Cl-bench: A benchmark for context learning, 2026. URL https://arxiv.org/abs/2602.03587.   
9 Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, et al. A survey of self-evolving agents: On path to artificial super intelligence. arXiv preprint arXiv:2507.21046, 1, 2025.   
10 Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
11 Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner series, 2025. Notion Blog. Da Hendcs, ol Burs Su avath Akul Aro Seve Basart, i , Dag, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.   
13 Jingcheng Hu, Yinmin Zhang, Shijie Shang, Xiaobo Yang, Yue Peng, Zhewei Huang, Hebin Zhou, Xin Wu, Jie Cheng, Fanqi Wan, et al. Pacore: Learning to scale test-time compute with parallel coordinated reasoning. arXiv preprint arXiv:2601.05593, 2026.   
14 Shashank Kirtania, Param Biyani, Priyanshu Gupta, Yasharth Bajpai, Roshni Iyer, Sumit Gulwani, and Gustavo Soares. Improving language agents through brew. arXiv preprint arXiv:2511.20297, 2025.   
15 David A Kolb. Experiential learning: Experience as the source of learning and development. FT press, 2014.   
16 Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In The twelfth international conference on learning representations, 2023.   
17 Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhan Chaofan Lin, Chen Dong, t al. Deepseek-v3. : Pushing the frontir o open large langage models. arXiv preprint arXiv:2512.02556, 2025.   
18 Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. IEEE Transactions on Audio, Speech and Language Processing, 2025.   
19 Long Ouyang, Jff Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:2773027744, 2022.   
20 Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2023.   
21 David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First conference on language modeling, 2024.   
22 Rana Salama, Jason Cai, Michelle Yuan, Anna Currey, Monica Sunkara, Yi Zhang, and Yassine Benajiba. Meminsight: Autonomous memory augmentation for llm agents. In Proceedings of the   
2025 Conference on Empirical Methods in Natural Language Processing, pages 3312433140, 2025.   
23 Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
24 Hajime Shirouzu, Naomi Miyake, and Hiroyuki Masukawa. Cognitively active externalization for situated reflection. Cognitive science, 26(4):469501, 2002.   
25 Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, et al. Agent kb: Leveraging cross-domain experience for agentic problem solving. arXiv preprint arXiv:2507.06229, 2025.

26 Sai Wang, Yu Wu, and Zhongwen Xu. Cogito, ergo ludo: An agent that learns to play by reasoning and planning. arXiv preprint arXiv:2509.25052, 2025.

27 Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, et al. Bapo: Stabilizing off-policy reinforcement learning for llms via balanced policy optimization with adaptive clipping. arXiv preprint arXiv:2510.18927, 2025.

28 Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.

29 Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.

30 An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

31 Ling Yang, Zhaochen Yu, Bin Cui, and Mengdi Wang. Reasonflux: Hierarchical llm reasoning via scaling thought templates. arXiv preprint arXiv:2502.06772, 2025.

32 Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

33 Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

34 Weiqi Zhai, Zhihai Wang, Jinghang Wang, Boyu Yang, Xiaogang Li, Xiang Xu, Bohan Wang, Peng Wang, Xingzhe Wu, Anfeng Li, et al. Hle-verified: A systematic verification and structured revision of humanity's last exam. arXiv preprint arXiv:2602.13964, 2026.

35 Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F Wong, and Yu Cheng. Exgrpo: Learning to reason from experience. arXiv preprint arXiv:2510.02245, 2025.

36 Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 1963219642, 2024.

37 Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

38 Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023.

3Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqing Liu, Rui Men, An Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

# A. Training Details

# A.1. RL Stage

To ensure stability and effectiveness in our RL experiments, we adopt the following configuration. Please refer to Table 4. All training experiments are implemented on $3 2 \times \mathrm { H } 2 0$ GPUs.

Table 4. RL Training hyper-parameters.   

<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Learning Rate</td><td>1e-6</td></tr><tr><td>Total Steps</td><td>200</td></tr><tr><td>Batch Size</td><td>128</td></tr><tr><td>Mini Batch Size</td><td>32</td></tr><tr><td>KL Loss Coefficient</td><td>0.0</td></tr><tr><td>Clip Higher</td><td>(0.28, 0.2)</td></tr><tr><td>Temperature</td><td>1.0</td></tr><tr><td>Number of Rollouts</td><td>16</td></tr><tr><td>Maximum Prompt Length</td><td>2K</td></tr><tr><td>Maximum Response Length</td><td>12K</td></tr></table>

Table 5. SFT Training hyper-parameters.   

<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Learning Rate</td><td>1e-6</td></tr><tr><td>Total Epochs</td><td>4</td></tr><tr><td>Batch Size</td><td>128 for 8B/14B; 64 for 4B</td></tr><tr><td>Maximum Length</td><td>6144</td></tr></table>

# A.2. Internalization Stage

We show the internalization training configurations in Table 5.

# A.3. Experience Generator Training

We show the training configurations of experience generator in Table 6. The training data is sourced from SkyWork as well.

# B. Prompt Template

# B.1. Prompt for Experience Extraction

Our prompt for experience extraction is shown in Table 7.

# B.2. Prompt for Reasoning

B.2.1. Reasoning without Experience Our prompt for reasoning without experience is shown in Table 8.

# B.2.2. Reasoning with Experience

Our prompt for reasoning with experience is shown in Table 9.

# B.3. Prompt for Rephrasing

Our prompt for solution rephrasing is shown in Table 10.

# C. Case Study

# C.1. Example of Experience

We present several examples of experiences generated by our experience generator in Table 11.

Table 6. Experience Generator Training hyper-parameters.   

<table><tr><td>Hyper-parameter</td><td>Value</td></tr><tr><td>Learning Rate</td><td>1e-5</td></tr><tr><td>Total Epochs</td><td>4</td></tr><tr><td>Batch Size</td><td>128</td></tr><tr><td>Maximum Length</td><td>16384</td></tr></table>

# C.2. Comparison of Utilization between DGO and GRPO

In Table 12, both the DGO-trained model and the GRPO-trained model are provided with the same retrieved experiences, all of which are clearly irrelevant to the current problem. The experiences concern set counting and inclusion-exclusion, whereas the target problem is a product over roots of unity. This setup therefore serves as a direct test of robustness to noisy retrieved experience.

The DGO-trained model remains largely problem-driven despite the presence of irrelevant experience. Its trajectory quickly identifies the algebraic structure of the expression, factorizes $x ^ { 2 } - 2 x + 2$ into $( x - ( 1 + i ) ) ( x - ( 1 - i ) )$ , and then applies the standard roots-of-unity product identity to transform the original product into evaluations at $1 \pm i$ The overall reasoning path is coherent and tightly aligned with the mathematical structure of the problem. Importantly, although noisy experiences are present in the prompt, they do not meaningfully alter the solution strategy. This suggests that the DGO-trained model is able toflter u relevant external uidance ndrelyontrinsically relevan stucture when the eved experience is unhelpful.

By contrast, the GRPO-trained model is more vulnerable to the same noisy experience. Although it and eventually makes an invalid substitution, incorrectly reducing the target product to $Q ( 2 )$ .This error relects not simply a calculation mistake, but a weaker ability to discriminate between structurally relevant and irrelevant guidance. Under noisy retrieval, the model appears more prone to drifting toward a loose template-based manipulation rather than preserving the exact algebraic requirements of the problem. As a result, it reaches the wrong answer.

This comparison highlights an important difference in experience utilization. Stronger utilization is not merely the ability to use retrieved experience when it is helpful, but also the ability toreject or suppress it when it is irrelevant. The DGO-trained model demonstrates better robustness under noisy experience, whereas the GRPO-trained model is more easily distracted by irrelevant context. This suggests that DGO improves not only the effectiveness of leveraging useful experience, but also the selectivity required to avoid being misled by unhelpful experience.

# C.3. Comparison of Internalization between DGO and GRPO

In Table 13, no external experience is provided at inference time, so the difference between the two trajectories mainly reflects the extent to which useful reasoning patterns have been internalized into the ml parameers. The problem isel is sructurally siplte expanding the lgarithms, the key ste is to introduce $\log _ { x } y$ and $\log _ { y } x$ and then use the reciprocal relation between them. Once this relation is recognized, the quantity $x y$ collapses immediately to a constant.

The DGO-trained model exhibits exactly this kind of internalized competence. Its trajectory quickly abstracts the problem into two compact symbolic relations, explicitly introduces auxiliary variables, and then invokes the identity $\log _ { y } x = 1 / \log _ { x } y$ to connect the two equations. The subsequent derivation is short, stable, and fully symbolic, leading directly to $x y = 2 5$ . Notably, the model does not rely on trial-and-error search orunnecessary algebraic expansion. This suggests that the relevant reasoning pattern has been internalized as a reusable parametric capability.

By contrast, the GRPO-trained model does not show the same level of internalization. Although it begins from the same logarithmic expansions, it fails to consolidate them into the reciprocal-log structure that makes the problem easy. Instead, it substitutes one equation into the other, produces a complicated self-referential expression, and then falls back to repeated numerical guessing. The trajectory becomes increasingly unstable and eventually loses focus altogether. This behavior indicates that the model has not fully absorbed the underlying simplification pattern into its parameters, even though the problem can in principle be solved by a short symbolic argument.

This comparison reveals the role of experience internalization. Strong internalization is reflected not only in getting the correct answer, but in whether the model can spontaneously recover the right abstraction without external guidance. The DGO-trained model demonstrates stronger internalization by directly activating a compact and transferable reasoning schema, whereas the GRPO-trained model is more likely to remain trapped in local manipulations without reaching the core structural insight.

# C.4. Quality of Experience across Iterations

As shown in Table 14, across the three rounds, the extracted experience becomes progressively more structured, more actionable, and better aligned with the true computational bottleneck of the problem.

The Round 0 experience already captures the key algebraic reduction: the problem can be transformed into the divisibility condition $( b - 1 ) \mid k ( k - 1 )$ together with the formula $\begin{array} { r } { d _ { 1 } ^ { \setminus } = \frac { k ( k - 1 ) } { b - 1 } } \end{array}$ k(This is a meaniul step beyond naiv bruteforce search, since it identifes the centralarithmetic constraint underlyinvalid $b$ -beautiful numbers. However, the experience is still relatively coarse-grained. It points to factoring $b - 1$ and using divisibility, but does not yet describe a complete reasoning chain from the original digit condition to a systematic counting procedure.

The Round 1 experience is more complete and operational. Instead of starting from an already reduced form, it begins from the original condition $a + d = { \sqrt { a b + d } }$ , explicitly introduces the substitution $k = a + d$ , derives $a = { \frac { k ( k - 1 ) } { b - 1 } }$ $d = k - a$ Compared with Round 0, this version better preserves the full reasoning pipeline from the original statement to the final verification conditions. As a result, it is more directly reusable as a solution template rather than merely a high-level hint.

The Round 2 experience goes one step further by making the counting structure itself more explicit. Rather than stopping at the divisibility condition, it turns that condition into a sharper congruence-based restriction on $k$ , and uses it to narrow the candidate space to a much smaller set. This makes the experience more closely matched to the real objective of the problem, which is not just to verify a single candidate, but to identify the smallest base $b$ for which the number of valid solutions exceeds a threshold. In this sense, the third-round experience is the most strategi it abstracts awayfrom local derivation details and focuses on how to reduce the search space efficiently.

Overall, the progression from Round 0 to Round 2 shows a clear refinement trend. The experience evolves from identifying a useful arithmetic condition, to providing a full derivation-and-checking template, and fnally exposinhehgherevel structural principlehat uportsfct countiTh illustrateshow iterative refinement can gradually transform raw problem-solving traces into more compact, transferable, and strategically valuable experience.

# D. Additional Details

# D.1. Rule of Experience Update

Experience update follows a redundancy-first, quality-second procedure:

•Redundancy check. For each new experience, we first compute its semantic similarity with the existing experiences in the bank. If the maximum similarity is below a threshold $\beta$ , the new experience is treated as a candidate addition. Otherwise, it is regarded as redundant with the most similar existing experience and is treated as a candidate replacement. We use a predefined similarity threshold $\beta = 0 . 8$ for redundancy detection.   
Quality check. We then evaluate the candidate action on the validation set using $\mathrm { a v g } @ 1 6$ If the new experience is a candidate addition, it is added only when it improves the validation performance of the current bank. If it is a candidate replacement, it replaces the old experience only when it yields a higher avg $@ 1 6$ than the old one.

# D.2. Rule of Trajectory Filter

When rewriting experience-guided trajectories, strict filtering is applied to the rewritten trajectories. The filtering criteria include:

trajectories with incorrect final answers;   
trajectories that still explicitly rely on experience, as indicated by keywords such as 'refer to experience';   
trajectories that are excessively long, exceeding 6K in length;   
noisy trajectories containing long segments of meaningless strings.

### You are an expert knowledge extractor specializing in converting problem-solving $\hookrightarrow$ trajectories into abstract, reusable reasoning rules that preserve crucial $\hookrightarrow$ mathematical specificity and improve future problem solving efficiency.

Your task is to analyze a given problem and a candidate solution trajectory. Based on $\hookrightarrow$ logical consistency, mathematical validity, and alignment with the problem $\hookrightarrow$ requirements, judge whether the solution is likely correct or incorrect, and then $\hookrightarrow$ extract $\star \star$ one concise general rule\*\* as reusable experience.

The solution may be correct or incorrect. You must infer this by critically examining $\hookrightarrow$ the reasoning and results.

#### Instructions:

1. $\star \star$ Input Analysis $\star \star$ : \*\*problem\*\*: The original problem statement. \*\*solution $\star \star$ : A full solution trajectory attempting to solve the problem.

2. $\star \star$ Experience Extraction Guidelines $\star \star$ : $\star \star$ Self-judgment $\star \star$ : Evaluate whether the solution is likely correct by checking: (a) whether the final answer satisfies the problem conditions, and (b) whether each reasoning step is logically and mathematically valid.

- \*\*If the solution appears correct $\star \star$ :

Do Not merely restate why it works.

Instead, focus on extracting $\star \star$ optimization experience\*\*, such as:

how the solution could be made shorter or more direct, • which intermediate steps, case splits, or calculations are redundant, • which key insight could be used earlier to avoid detours, • how to reformulate the approach into a cleaner standard method.

Summarize this as a rule that helps future solvers reach the answer $\star \star$ faster and $\hookrightarrow$ with less complexity $\star \star$ .

− $\star \star$ If the solution appears incorrect $\star \star$ • what warning sign to look for, and •what should be done instead to prevent this type of mistake.

- \*\*One Output $\star \star$ :

Generate \*\*exactly one rule $\star \star$ that reflects either: (a) an optimization rule distilled from a correct but improvable solution, or (b) an avoiding-errors rule distilled from an incorrect solution.

- \*\*Abstraction with Specificity $\star \star$ : Generalize the rule while retaining essential mathematical structures, formulas, $\hookrightarrow$ ′ constraints, or thresholds that make the experience precise and actionable. Use LaTeX for mathematical expressions where appropriate.

3. $\star \star$ Rule Format Requirement\*\*:

Output \*\*exactly one rule $\star \star$ in the following structure, with no extra text: WHEN <Broad domain or phase>. IF <Specific condition, scenario, or inefficiency / potential error>. THEN <Action to take or avoid, emphasizing simplification, direct strategy, or $\hookrightarrow$ error prevention, with key formulas if needed>.

# \*\*Input:\*\* # problem:\n{problem}\n # solution:\n{solution}\n

# \*\*Output:\*\*   
WHEN <Broad domain or phase>. IF <Specific condition, scenario, or inefficiency / → potential error>. THEN <Action to take or avoid>.

Table 7. Prompt template for experience extraction.

Table 8. Prompt template for reasoning without experience.

<table><tr><td>Please reason step by step, and put your final answer within \boxed{{}}. You will be</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>→ →</td><td>given some reusable problem-solving experiences, and you can refer to them during the reasoning.</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>[Experiences]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>{experiences}</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>[Problem]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>{problem}</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9. Prompt template for reasoning with experience.

You are an expert mathematician. You are given a math problem, some background $\hookrightarrow$ [Experience Content], and a [Reference Solution] which may have used those $\hookrightarrow$ experiences explicitly.

Your task is to re-author a "Gold Standard" solution. This solution must be entirely $\hookrightarrow$ self-contained, as if you solved the problem using only your innate mathematical $\hookrightarrow$ intuition.

Strict requirements:

1. Total De-Reference: You MUsT NOT mention, quote, or even subtly refer to the $\hookrightarrow$ [Experience Content] provided. Do not use phrases like "By logic of...", "Following $\hookrightarrow$ the principle of...", or any terminology that implies you are looking at an external $\hookrightarrow$ guide. The reader should have no idea that any "Experience Content" was provided to $\hookrightarrow$ you.

2. Invisible Integration: Instead of "using" the experience, you must "be" the expert $\hookrightarrow$ who already knows it. If the experience suggests a shortcut or a formula, justify $\hookrightarrow$ its use through first-principles reasoning within the flow of the solution. For $\hookrightarrow$ example, do not say "Using the formula for X," say "Since the problem requires $\hookrightarrow$ [reason], we can derive that [logical step]..."

3. Silent Audit & Correction: Treat the [Reference Solution] as a potentially flawed $\hookrightarrow$ scratchpad. Verify every calculation and logical leap. If it is wrong, fix it $\hookrightarrow$ silently. Your final output must be a clean, error-free, and authoritative proof. Do $\hookrightarrow$ NOT mention that the reference was wrong.

4. Strategic Fidelity: Follow the general strategy of the [Reference Solution] (e.g., if $\hookrightarrow$ it uses energy conservation, you should too), but execute it with perfect precision $\hookrightarrow$ and without any meta-talk or conversational fillers.

5. Final Answer Consistency: The final numerical/symbolic answer must exactly match the $\hookrightarrow$ [Reference Solution] and be enclosed in \boxed{{}}.

[Experience Content] {experiences}

[Problem] {problem}

[Reference Solution] {solution}

[Your Solution]

Table 10. Prompt template for solution rewriting.

![](images/8.jpg)  
Table 11. An example of retrieved experiences associated with a problem.

Table 12. Comparison of two trajectory excerpts under the same retrieved experiences. Both trajectories receive the same irrelevant experiences, but the left one stays problem-driven while the right one makes an invalid polynomial substitution.   

<table><tr><td>DGO-trained Trajectory (Correct)</td><td>GRPO-trained Trajectory (Incorrect)</td></tr><tr><td>When \(k=0\), the factor is \[</td><td>Let \[</td></tr><tr><td>2-2(1)+1^2=1,</td><td>f(y)=2-2y+y^2. \]</td></tr><tr><td>\] so it remains to evaluate</td><td>We need</td></tr><tr><td>\[ \prod_{k=1}^{12}(2-2\omega^k+\omega^{2k} →− ).</td><td>\[ P=\prod_{y\in\mu_{13}\setminus\{1\}}</td></tr><tr><td>\] Let</td><td>→ (2-2y+y^2). \]</td></tr><tr><td>\[ P(x)=x^2-2x+2=(x-(1+i))(x-(1-i)).</td><td>Consider \[</td></tr><tr><td>\]</td><td>Q(x)=\prod_{y\in\mu_{13}\setminus\{1\}}(j → x-y)</td></tr><tr><td>Then</td><td>=\frac{x^{13}-1}{x-1}.</td></tr><tr><td>\[ \prod_{k=1}^{12}P(\omega^k)</td><td>\]</td></tr><tr><td>= \left(\prod_{k=1}^{12}(\omega^k-(1+i))\r]</td><td>Now, we can express \(P\) in terms of → \(Q(x)\). Notice that</td></tr><tr><td>→ ight)</td><td>\[</td></tr><tr><td>\left(\prod_{k=1}^{12}(\omega^k-(1-i))\r] → ight).</td><td>P=Q(2). \]</td></tr><tr><td>\]</td><td></td></tr><tr><td></td><td>Thus</td></tr><tr><td>Using</td><td>\[ P=2^{12}+2^{11}+\cdots+2+1=8191,</td></tr><tr><td>\[ \prod_{y^{13}=1,\y\neq</td><td>\]</td></tr><tr><td>→1}(y-a)=\frac{a^{13}-1}{a-1},</td><td>so the remainder is</td></tr><tr><td>\]</td><td>\[</td></tr><tr><td>the product becomes</td><td>8191 \bmod 1000=191.</td></tr><tr><td>\[</td><td>\]</td></tr><tr><td>[(1+i)^{13}-1][(1-i)^{13}-1].</td><td></td></tr><tr><td>\]</td><td>The trajectory later rewrites</td></tr><tr><td></td><td>\[</td></tr><tr><td>(-65-64i)(-65+64i)=8321.</td><td>2-2y+y^2=(y-1)^2+1,</td></tr><tr><td>\]</td><td>\]</td></tr><tr><td></td><td>but does not derive a valid product</td></tr><tr><td>Hence,</td><td></td></tr><tr><td></td><td>→ identity from it, and</td></tr><tr><td>\[</td><td>still returns</td></tr><tr><td>8321 \bmod 1000 = 321.</td><td></td></tr><tr><td></td><td>\[</td></tr><tr><td>\]</td><td></td></tr><tr><td></td><td>\boxed{191}.</td></tr><tr><td></td><td></td></tr><tr><td></td><td>\]</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>\[</td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>\boxed{321}</td><td></td></tr><tr><td></td><td></td></tr><tr><td>From</td><td>From</td></tr><tr><td>\[</td><td>\[</td></tr><tr><td>\log_x(y^x)=10,</td><td>\log_x(y^x)=10,</td></tr><tr><td>\]</td><td>\]</td></tr><tr><td>we get</td><td>the solution rewrites</td></tr><tr><td>\[</td><td>\[</td></tr><tr><td>x\log_x y=10.</td><td>y=x^{10/x}.</td></tr><tr><td>\]</td><td>\]</td></tr><tr><td>Let</td><td>From</td></tr><tr><td>\[</td><td>\[</td></tr><tr><td>a=\log_x y.</td><td>\log_y(x^{4y})=10,</td></tr><tr><td>\]</td><td>\]</td></tr><tr><td>Then</td><td>it rewrites</td></tr><tr><td>\[</td><td>\[</td></tr><tr><td>x=\frac{10}{a}.</td><td>x=y^{5/(2y)}.</td></tr><tr><td>\]</td><td>\]</td></tr><tr><td>From</td><td>Then it substitutes one equation into the</td></tr><tr><td>\[</td><td>→ other:</td></tr><tr><td>\log_y(x^{4y})=10,</td><td>\[</td></tr><tr><td>\]</td><td>x=\left(x^{10/x}\right)^{5/(2x^{10/x})}</td></tr><tr><td>we get</td><td>=x^{25/(x^{10/x+1})},</td></tr><tr><td>\[</td><td>]</td></tr><tr><td>4y\log_y x=10.</td><td>and concludes</td></tr><tr><td>\]</td><td>\[</td></tr><tr><td></td><td>x^{10/x+1}=25.</td></tr><tr><td>Let \[</td><td>\]</td></tr><tr><td>b=\log_y x.</td><td>Instead of using the reciprocal relation</td></tr><tr><td>\]</td><td>→ between</td></tr><tr><td>Then</td><td>\(\log_x y\) and \(\log_y x\), the</td></tr><tr><td>\[</td><td>→ trajectory then turns to</td></tr><tr><td>y=\frac{5}{2b}.</td><td>repeated numerical guessing and fails to</td></tr><tr><td>\]</td><td>obtain a stable</td></tr><tr><td></td><td>symbolic simplification.</td></tr><tr><td>Using the reciprocal relation \[</td><td>The repeated trial steps are omitted</td></tr><tr><td>\log_y x=\frac{1}{\log_x y}, \]</td><td>→− here.</td></tr><tr><td>we have</td><td>Final answer:</td></tr><tr><td>\[</td><td>\[</td></tr><tr><td>b=\frac{1}{a}.</td><td>\boxed{\frac{3}{10}}</td></tr><tr><td>\]</td><td>\]</td></tr><tr><td></td><td></td></tr><tr><td>Thus</td><td></td></tr><tr><td>\[</td><td></td></tr><tr><td>y=\frac{5}{2\cdot(1/a)}=\frac{5a}{2}. \]</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Therefore,</td><td></td></tr><tr><td>\[</td><td></td></tr><tr><td>xy=\frac{10}{a}\cdot\frac{5a}{2}=25.</td><td></td></tr><tr><td>\]</td><td></td></tr><tr><td></td><td></td></tr><tr><td>\[</td><td></td></tr><tr><td>\boxed{25}</td><td></td></tr><tr><td>\]</td><td></td></tr></table>

Table 13. Comparison of two trajectory excerpts without retrieved experience. The DGO-trained model directly invokes a compact logarithmic relation and reaches the correct result, while the GRPO-trained model fails to internalize the same simplification pattern and falls into unstable symbolic manipulation.

# [Problem]

Let \(b\ge 2\) be an integer. Call a positive integer \(n\) \(b\text-\textit{eautiful}\) $\hookrightarrow$ if it has exactly two digits when expressed in base \(b\) and these two digits sum $\hookrightarrow$ to \(\sqrt n\). For example, \(81\) is \(13\text-\textit{eautiful}\) because \ $\begin{array} { r l } { { 8 } \operatorname { 1 } } & { { } = } \end{array}$ $\hookrightarrow$ \underline{6}\ \underline{3}_{13}\) and $\backslash ( 6 + 3 = \backslash { \tt s q r t } \{ 8 1 \} \setminus )$ . Find the least $\hookrightarrow$ integer \(b\ge 2\) for which there are more than ten \(b\text-\textit{eautiful}\) $\hookrightarrow$ integers.

# [Round 0]

WHEN solving for numbers with digit sum equal to the square root in a given base. IF the condition leads to \ $( { \mathrm { ~ n ~ } } = { \mathrm { ~ k } } ^ { \mathord { \wedge } } 2 \backslash$ and $\setminus ( \mathrm { d } _ { - } 1 ~ = ~ \setminus \mathrm { f r a c } \{ \mathrm { k } ( \mathrm { k - 1 } ) \} \{ \mathrm { b - 1 } \} \setminus )$ where \(d_1\) is $\hookrightarrow$ the leading digit in base \(b\).   
THEN use the divisibility condition \((b-1) \mid k(k-1)\) to find all \(k|) for each $\iff \quad \setminus ( \mathrm { b } \setminus )$ by factoring \(b-1\), which is more efficient than brute-forcing over the $\hookrightarrow$ range of \(k\). [Round 1]   
WHEN solving problems about digit conditions in a given base.   
IF the condition is \( a+d $=$ \sqrt{ab $+ \mathrm { d }$ } \) with \( a \) the leading digit, \( 1 \le a → \1e b-1 \), \( 0 \le d \le b-1 \).   
THEN set \( $\mathrm { ~ k ~ } ~ = ~ \mathsf { a } + \mathsf { d } ~ ^ { \mathrm { ~ } }$ so $\mathrm { ~  ~ n ~ } = \mathrm { ~  ~ k ~ } ^ { \wedge } 2 \mathrm { ~  ~ \rho ~ } \backslash )$ , derive $( \texttt { a } = \texttt { \backslash f r a c } \{ \texttt { k } ( \texttt k - 1 ) \} \{ \texttt b - 1 \} \setminus )$ , and find $\hookrightarrow \cdot ($ b \) such that \( b-1 \mid $\mathrm { ~ k ~ } ( \mathrm { k } - 1 )$ \) for multiple $\setminus ( \mathrm { ~ \bf ~ k ~ } \setminus )$ , ensuring \( a \) and $\hookrightarrow \mathrm { ~  ~ { ~ \varphi ~ } ~ } \mathrm { ~  ~ { ~ \psi ~ } ~ } ( \mathrm { ~  ~ { ~ d ~ } ~ } = \mathrm { ~ \ k - a ~ \pi ~ } \setminus )$ satisfy digit bounds, to avoid inefficient brute-force enumeration. [Round 2]   
WHEN solving problems about numbers with two digits in base \(b\) satisfying a digit-sum $\hookrightarrow$ condition involving \(\sqrt $\{ \neg \} \setminus \}$ :   
IF you have derived that \ $( \texttt { n } = \texttt { a b } + \texttt { d } )$ with \ $\mathbf { \Omega } ( \mathbf { a } + \mathbf { d \Omega } = \textbf { k } = \mathbf { \Omega } \backslash \mathbf { s q r t } \{ \mathbf { n } \} \setminus )$ and obtained the $\hookrightarrow$ relation\ $a =$ \frac{ $\mathrm { ~ \bf ~ k ~ } ^ { \star } 2 \mathrm { ~ \bf ~ - ~ } \mathrm { ~ \bf ~ k ~ } ]$ {b-1}\).   
THEN note that $\setminus ( \mathrm { k } \ ( \mathrm { k - 1 } ) \setminus )$ must be divisible by $\setminus ( \mathrm { b - 1 } \setminus )$ , which is equivalent to \(k $\hookrightarrow$ \equiv 0\) or \(1 \pmod $\{ \mathrm { b - 1 } \} \setminus \{ \mathrm { ) }$ ; use this congruence to restrict \(k\) to two $\hookrightarrow$ arithmetic progressions within the range \(\lceil \sqrt{b} \rceil \le k \le b-1\), $\hookrightarrow$ drastically reducing the number of candidates to check.

Table 14. The problem and one representative extracted experience from each round.