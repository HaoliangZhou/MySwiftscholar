# 在线推理校准：测试时训练使得通用的符合性大型语言模型推理成为可能

蔡周\* 123 王泽凯\* 13 孟华 ${ \bf W } { \bf u } ^ { 1 2 }$ 钱宇·朱莉 $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 3 4 }$ 芙洛拉·C·施13 陈昱·王12 阿希亚·威尔逊13 托米·雅卡拉12 斯蒂芬·贝茨13 1麻省理工学院电气工程与计算机科学系 2麻省理工学院计算机科学与人工智能实验室 3本体论与决策系统 4计算科学与工程（麻省理工学院CSE） 麻省理工学院 {caiz428,zekai}@mit.edu

# 摘要

尽管测试时缩放使大型语言模型能够解决高度困难的任务，但最先进的结果带来了巨额的计算成本。这些低效率归因于后训练语言模型的误校准以及流行采样技术中的校准缺失。在此，我们提出了在线推理校准（ORCÅ）框架，该框架基于符合预测和测试时训练对采样过程进行校准。具体来说，我们引入了一种元学习程序，为每个输入更新校准模块。这使我们能够在分布转移下提供有效的置信度估计，例如在不同推理阶段出现的思维模式，或者模型开发与部署之间的提示分布。ORCÅ不仅在符合风险方面提供理论保证，还在不同推理任务中实证显示出更高的效率和泛化能力。在风险水平 $\delta { = } 0 . 1$ 下，ORCÅ在有监督标签的情况下，提升了Qwen2.5-32B在同分布任务上的效率，节省高达 $4 7 . 5 \%$，在自一致性标签下节省 $4 0 . 7 \%$。在零-shot的域外设置中，它将MATH-500的节省从静态校准基线的 $2 4 . 8 \%$ 改善至 $6 \hat { 7 } . 0 \%$，同时保持低的经验误差率，并且这一趋势在模型族和下游基准之间也保持一致。我们的代码已公开发布在 https:/ / github.com/wzekai99/ORCA。

# 1 引言

大型语言模型通过大幅提升测试时的计算能力解决了日益复杂的问题，例如奥林匹克数学或软件工程任务。诸如并行采样（Qi et al., 2025）、序列蒙特卡洛（Feng et al., 2025）、蒙特卡洛树搜索（Zhang et al., 2024）、验证器引导采样（Yu et al., 2025）和自一致性（Xie et al., 2024；Huang et al., 2025）等策略是从大型语言模型中引发高级推理能力的关键，但往往面临效率和可靠性的瓶颈。已知后训练的语言模型在判断其中间状态和最终答案是否正确时存在误校准（Li et al., 2025a）。因此，测试时的采样策略往往涉及手工设定的参数，以平衡样本质量和资源分配，例如并行推演的数量或验证器的提示启发式，这些都容易受到奖励操控和分布偏移的影响（Snell et al., 2024）。本研究通过一种原则性的方法来应对这些挑战。我们的目标是（i）支持根据任务难度进行自适应计算分配，同时在测试时提供样本质量和效率的统计保证，以及（ii）在分布偏移下保持稳健，因为在部署时的提示分布可能与模型开发时的分布不同。

符合预测（Shafer & Vovk, 2008；Angelopoulos & Bates, 2021）和校准方法提供有限样本的覆盖保证，并且可以估计一组大语言模型（LLM）输出中是否包含正确答案的置信度。在测试时间缩放的背景下，这些方法用于限制必须抽样的词元或例子的数量，同时仍确保响应的高质量。例如，思维校准（Wu et al., 2025）等方法（Xie et al., 2025；Wang et al., 2026）将推理效率表述为一个风险控制的停止问题，并产生用于在线探测的停止概率的校准阈值。然而，这些方法假设固定的推理过程，其动态在采样过程中是统一的。也就是说，它们并未解决在两个层面上分布转变下置信度估计的有效性。在样本层面，不同位置的推理模式可能在长推理链（CoT）中有所不同。在数据集层面，模型通常在训练期间未见过的分布外（OOD）环境中部署。另一方面，测试时间训练（TTT）（Sun et al., 2020；2024）为LLM的在线适应提供了一种自然机制。从高层次上看，其目标是在推理时根据输入的特征调整模型权重。具体来说，内部循环基于每个传入的词元或推理步骤，通过最小化自监督目标来更新一小组“快速”权重。一个单独的外部循环在许多序列上进行训练，以学习共享的初始化和特征映射，从而使内部循环的更新稳定且可迁移（Sun et al., 2020）。TTT框架旨在通过元学习提高在新领域或长序列中的整体建模能力和泛化能力。其标准损失基于重建，并不直接与校准或风险控制对齐。在本文中，我们提出在线推理校准（ORCA），通过将校准本身构建为可在推理时优化的目标，实现高效且可信的测试时间缩放。具体而言，内部循环优化一个评分函数（LLM尝试的正确性或一致性），该函数通过TTT层实现，并在推理/搜索轨迹中在线更新。因此，校准可以在实例层面适应推理的不同阶段。同时，外部循环是一个元训练过程，学习共享的“慢”权重（校准层的初始化和特征映射），以便在线更新在数据集层面保持稳定、数据高效且可迁移。该设计解决了以往工作的两项关键局限。首先，通过校准在部署时实际执行的算法，保持了统计有效性。其次，通过允许逐实例的在线适应置信度估计，提高了对分布转变的鲁棒性，同时基本的LLM保持不变。

经验研究表明，我们的方法在控制风险的情况下提供可靠的停止与候选选择，减少了在简单实例上的不必要测试时间计算，并且仅在不确定性较高时扩展计算。在目标风险水平 $\delta { = } 0 . 1$ 的分布内测试集上，将 ORCA 应用于 Qwen2.5-32B 实现了在监督模式下最高节省 $4 7 . 5 \%$ 的计算资源，在一致性模式下节省 $4 0 . { \dot { 7 } } \%$ 的计算资源。在零样本 OOD 环境下，我们的方法将 MATH-500 的节省从静态基线的 $2 4 . 8 \%$ 提高至有监督标签的 $6 7 . 0 \%$，并且在各种模型家族（Qwen2.5-32B、QwQ-32B 和 Llama-3.3-70B）及基准（MATH、GPQA 和 AIME）测试中始终优于静态探测方法。

# 2 初步研究

测试时训练。语言模型在实例级变化和分布转移下可能表现出脆弱性（Lu 等，2022；Tang 等，2024）。由于列举新用例并微调模型是不可行的，测试时训练（TTT）通过元训练在推理过程中适应轻量级模块来解决这一问题（Sun 等，2020）。具体而言，TTT 引入了在每个测试序列中通过最小化自监督或代理目标来更新的快速权重 $W _ { t }$。遵循 (Sun 等，2024) 中的标准公式，

$$
\begin{array} { r } { \ell ( W ; x _ { t } ) = \| f ( \theta _ { K } x _ { t } ; W ) - C _ { t } \| _ { 2 } ^ { 2 } , \quad \quad \quad } \\ { W _ { t } = W _ { t - 1 } - \eta G _ { t } , \quad G _ { t } \approx \nabla _ { W } \ell ( W _ { t - 1 } ; x _ { t } ) , } \\ { z _ { t } = f ( \theta _ { Q } x _ { t } ; W _ { t } ) . \quad \quad } \end{array}
$$

Table 1: Notation summary.   

<table><tr><td>Symbol</td><td>Learned in</td><td>Description</td></tr><tr><td>φt</td><td></td><td>Step embedding: mean-pooled LLM hidden state at step t</td></tr><tr><td>Ct</td><td></td><td>Step label: correctness or consistency; Ct=0 at inference</td></tr><tr><td>Wo, b0</td><td>outer loop</td><td>Initial probe weights, learned via meta-training</td></tr><tr><td>Wt, bt</td><td>inner loop</td><td>Probe weights at step t, updated online during inference</td></tr><tr><td>θQ,K</td><td>outer loop</td><td>Unified outer view parameters (optional learned projections or identity)</td></tr><tr><td>η</td><td>outer loop</td><td>Inner learning rate; fixed or learned</td></tr><tr><td>S</td><td></td><td>Probe score: st = σ(f (θQφt; Wt))</td></tr><tr><td>l(W;φt)</td><td>−</td><td>Inner-loop objective: (st  Ct)2 (Brier score)</td></tr><tr><td>$λ</td><td>calibration</td><td>LTT-calibrated stopping threshold</td></tr><tr><td>δ</td><td></td><td>Target risk upper bound</td></tr><tr><td>E</td><td></td><td>LTT failure probability level</td></tr></table>

在这里，$C_{t}$ 是设计目标，它是一个投影 $\theta_{V} x_{t}$，旨在实现 Sun 等人（2024）的原始公式中的自重建，但实际上可以是任何适当定义的目标（例如，在我们的设置中为校准标签）；内部循环通过快速权重 $W_{t}$ 执行逐步在线适应，而外部循环则通过跨多个序列的共享慢权重参数（例如，可选投影 $\theta_{Q,K,V}$，初始化 $W_{0}$）进行元学习，以确保内部循环更新的稳定性和可转移性。为了简化起见，我们使用 $\bar{{\boldsymbol{\theta}}_{Q,K,V}}$ 作为外部循环视图参数的统一符号：如果可学习，它们将实例化 $\mathrm{Q/K/V}$ 风格的投影作为特征提取器，或者在设置为单位时恢复无 QK 特殊情况。TTT 可以视为双层优化：外部目标学习如何使快速在线学习有效，而内部目标实现推断时的实例特定适应。在我们的设置中，这一机制用于提高校准质量，同时通过所部署程序的符合校准来维护下游风险控制。符合预测与风险控制。在实际部署中，我们需要使停止/选择决策在统计上可靠的估计不确定性，而非启发式方法。符合预测通过将非符合性评分转换为校准预测集，提供有限样本的分布无关有效性（Shafer & Vovk，2008；Angelopoulos & Bates，2021）。在分裂符合预测中，评分模型在训练数据上进行拟合，决策阈值在保留的校准集上进行校准（Vovk 等，2005；Papadopoulos，2008）。具体而言，给定校准评分 $\{ u_{i} \}_{i=1}^{n}$ 和未覆盖水平 $\epsilon \in (0, 1)$，定义经验分位数

$$
\tau _ { 1 - \epsilon } = \mathrm { Q u a n t i l e } _ { \lceil ( n + 1 ) ( 1 - \epsilon ) \rceil / ( n + 1 ) } \big ( u _ { 1 } , \dots , u _ { n } \big ) .
$$

在测试时，我们接受分数低于（或置信度高于）此阈值的输出。等价地，我们形成一个符合性集合，包含所有满足标定标准的候选者。这产生了边际有限样本覆盖（同样是风险控制），通常形式为 $\mathbb { P } ( \overset { \cdot } { Y } \in \widehat { \mathcal { C } } ( X ) ) \overset { \cdot } { \geq } 1 - \epsilon$。学习后测试（LTT）通过标定决策规则而非预测集来补充符合性预测。给定一组候选规则（或阈值），LTT 在保留的标定集上检验形式为 $H _ { j } : r _ { j } \geq \delta$ 的均值风险原假设，并给出有效的 p 值，然后应用多重检验（例如，固定序列检验）来选择具有有限样本保证的规则 $\mathbb { P } ( r ( \hat { j } ) \le \delta ) \ge \bar { 1 } - \bar { \epsilon }$（Angelopoulos et al., 2021; Wu et al., 2025; Wang et al., 2026）。因此，符合性与 LTT 共享基于可交换性有效性的原则，但目标不同：集合覆盖与规则级风险控制。

# 3 在线推理校准

# 3.1 设置

我们研究输入 $x \in \mathcal { X }$ 的测试时推理过程。在推理步骤 $t$，模型生成了思维前缀 $\mathsf { \bar { y } } _ { t } = [ y ^ { ( 1 ) } , \ldots , y ^ { ( \hat { t } ) } ]$，此时当前答案候选 ans $\left( y _ { t } \right)$ 可以从推理状态中导出。我们用 $\phi _ { t } \in \mathbb { R } ^ { d _ { \phi } }$ 表示在步骤 $t$ 从基础 LLM 中提取的隐藏表示。该特征向量输入到校准模块，然后输出探测分数 $\dot { z } _ { t } = f ( \phi _ { t } ; W _ { t } )$，其中 $W _ { t }$ 是与步骤相关的快速权重。为了风险控制，令 $L ( \hat { y } , y ) \in \{ 0 , 1 \}$ 表示最终决策损失（例如，过早停止或错误接受答案）。

![](images/1.jpg)  

Figure 1: Framework of Online Reasoning Calibration (ORCA).

我们的公式遵循 Sun 等人（2024）的 TTT 更新结构以及 Angelopoulos 等人（2021）和 Wu 等人（2025）的 Learn-then-Test (LTT) 风险控制框架。TTT 更新使用内循环快速权重适应结合外循环学习的映射/初始化，通过 LTT 校准的方法强制执行风险控制。我们重新定义每步目标，侧重于校准质量而非重构。由于推理包括对 $\mathsf { \bar { \boldsymbol { W } } } _ { t }$ 的在线更新，我们校准整个自适应算法（包括采样、更新、停止规则），而不是依赖静态评分器。表 1 总结了符号对应关系。

# 3.2 内部循环：通过快速权重更新的在线自适应探测

在每个推理步骤 $t _ { \iota }$，基础 LLM 生成一个隐藏表示 $\phi _ { t } \in \mathbb { R } ^ { d _ { \phi } }$（例如，均值池化的最后一层隐藏状态）。我们的探测器维护快速权重 $W _ { t }$，这些权重在推理链中实时更新，在每个步骤产生一个置信度得分 $s _ { t } \in [ 0 , 1 ]$。该过程遵循先得分再更新的协议：探测器首先使用累计的权重对当前步骤进行打分，然后在进入下一个步骤之前更新其权重。我们首先介绍一种基础形式的内部循环更新。设 $f ( \cdot ; W ) : \mathbb { R } ^ { d _ { \phi } } \to [ 0 , 1 ]$ 为探测模型（例如，$f ( u ; W ) = \sigma ( W \cdot u + b )$）。我们首先描述基础版本，在每个步骤的三个操作如下：得分（使用来自前一步骤的权重）：

$$
s _ { t } = f ( \phi _ { t } ; W _ { t - 1 } )
$$

内层损失（与标签 $C _ { t }$ 的 Brier 评分）：

$$
\ell ( W _ { t - 1 } ; \phi _ { t } ) = \left( s _ { t } - C _ { t } \right) ^ { 2 }
$$

权重更新（在线梯度下降）：

$$
W _ { t } = W _ { t - 1 } - \eta \nabla _ { W } \ell \bigl ( W _ { t - 1 } ; \phi _ { t } \bigr )
$$

这里 $C_{t} \in \{ 0, 1 \}$ 是一个步骤级别的标签，指示当前答案尝试的质量，取决于校准指标。如 Wu 等人（2025）所讨论，$C_{t}$ 可以来自元训练中的多种来源：（i）监督：$C_{t} = \mathbb{I}\{ \mathsf{ans}(y_{t})$ 是正确的}，需要真值答案；（ii）一致性：$C_{t} \dot{=} \mathbb{I}\{ \mathsf{ans}(y_{t}) = \mathsf{ans}(y_{T}) \}$，与没有真值的全预算答案进行比较；或（iii）来自外部模型的教师/验证者标签。算法 1 训练阶段：外部循环 TTT 训练 需要：训练提示 $\mathcal{D}_{\mathrm{train}}$；外部参数 $\Theta_{\mathrm{outer}} = (\theta_{Q, K}, W_{0}, \eta)$。

1: 对于每个提示 $x \in { \overline { { \mathcal { D } } } } _ { \operatorname { t r a i n } }$ 执行 2: 沿推理轨迹展开内部更新以获得 $\{ W _ { t } \} _ { t = 1 } ^ { T }$，使用 $W _ { t } =$ $W _ { t - 1 } - \eta \nabla _ { W } \ell \bar { ( W _ { t - 1 } ; \phi _ { t } ) }$ 。 3: 计算 $\begin{array} { r } { \mathcal { L } _ { \mathrm { o u t e r } } \big ( \boldsymbol { x } , \{ W _ { t } \} _ { t = 1 } ^ { T } ; \Theta _ { \mathrm { o u t e r } } \big ) = \sum _ { t = 1 } ^ { T } \big ( s _ { t } - C _ { t } ^ { \mathrm { t r u e } } \big ) ^ { 2 } } \end{array}$ 。 4: 通过对展开的 $\mathcal { L } _ { \mathrm { o u t e r } }$ 进行微分来更新可训练组件 $\Theta _ { \mathrm { o u t e r } }$ 。 5: 结束循环 6: 返回训练好的 $\Theta _ { \mathrm { o u t e r } }$ 。

# 3.3 外部循环：用于通用自适应校准的元学习

单独的实例级更新可能会过拟合局部噪声，并在分布变化下降低风险控制。类似于元学习，外部循环学习适当的初始化和特征接口，以便内部循环适应能够在任务之间传递（Sun et al., 2020）。具体而言，内部循环学习进行校准（任务特定的在线适应），而外部循环学习如何进行校准。通过在异构数据集和提示上的元训练，模型学习初始化和更新动态，这些动态可以推广超出单一训练分布，从而在分布变化下实现稳健和高效的校准。方程（5）（7）定义了最简单的TTT更新规则，其中一个具有快速权重的探针 $\bar { W } \in \mathbb { R } ^ { 1 \times d _ { \phi } }$ $\phi _ { t }$ T ztion s y $d _ { \phi } + 1$ 可学习参数（初始化 $W _ { 0 }$ 和偏置 $b _ { 0 }$）可以视为在线自适应逻辑回归，称为基础或无QK变体。类似于Sun et al.（2024），自然扩展引入了学习的投影 $\theta _ { K } , \theta _ { Q } \in$ $\mathbb { R } ^ { d _ { h } \times d _ { \phi } }$，它们在更新和评分操作之前将 $\phi _ { t }$ 映射到一个低维空间：

$$
\begin{array} { r } { \begin{array} { r l } { s _ { t } = f ( \theta _ { Q } \phi _ { t } ; W _ { t - 1 } ) , \quad W \in \mathbb { R } ^ { 1 \times d _ { h } } } \\ { \ell ( W ; \phi _ { t } ) = \left( f ( \theta _ { K } \phi _ { t } ; W ) - C _ { t } \right) ^ { 2 } } \end{array} } \end{array}
$$

这个 QK 变体允许更新方向 $( \theta _ { K } )$ 和评分方向 $( \theta _ { Q } )$ 关注隐藏状态的不同方面。投影 $\theta _ { K }, \theta _ { Q }$ 是“慢权重”，并与 $W _ { 0 }$ 和 $\eta$ 一起在外循环中学习。这两种变体具有相同的单步表达能力，但在在线适应的动态上有所不同。无 QK 变体更新在全 $d _ { \phi }$ 维空间中，而 $\mathrm { Q } / \mathrm { K }$ 更新则限制在 $d _ { h }$ 维子空间中。为全面性起见，我们实验了使用和不使用 QK 更新的 TTT，并观察到两种变体在静态基线上的显著改进。我们现在正式介绍慢权重训练的一般框架。令 $\Theta _ { \mathrm { o u t e r } } = ( \theta _ { Q , K }, W _ { 0 }, \eta )$ 表示外部参数。请记住，在我们的框架中，基础 LLM 参数在训练期间默认不更新。给定训练提示 $x _ { \mathrm { . } }$，我们首先展开 $\{ W _ { t } \} _ { t = 1 } ^ { T }$，然后进行优化：

$$
\operatorname* { m i n } _ { \Theta _ { \mathrm { o u t e r } } } \ \mathbb { E } _ { x \sim \mathcal { D } _ { \mathrm { t r a i n } } } \mathcal { L } _ { \mathrm { o u t e r } } \left( x , \{ W _ { t } \} _ { t = 1 } ^ { T } ; \Theta _ { \mathrm { o u t e r } } \right) ,
$$

外部损失的定义为 $W _ { t } = W _ { t - 1 } - \eta \nabla _ { W } \ell ( W _ { t - 1 } ; x _ { t } )$。我们通过截断反向传播方法优化该双层目标，并在由相同部署过程产生的保留校准数据集上运行 LTT，以选择停止阈值。算法 1 概述了外层训练算法的一般形式，详见图 1。

$$
\mathcal { L } _ { \mathrm { o u t e r } } \left( \boldsymbol { x } , \{ W _ { t } \} _ { t = 1 } ^ { T } ; \Theta _ { \mathrm { o u t e r } } \right) : = \sum _ { t = 1 } ^ { T } \left( s _ { t } - C _ { t } ^ { \mathrm { t r u e } } \right) ^ { 2 } .
$$

算法2 校准和推断阶段

需求：标定提示 $\mathcal { D } _ { \mathrm { c a l } } $；阈值网格 $\Lambda = \{ \lambda _ { 1 } > \cdots > \lambda _ { m } \} $；提示 $x _ { \mathrm { . } }$ 训练的 $\Theta _ { \mathrm { o u t e r } }$。 1: (A) 通过 LTT 标定停止阈值。 2: 对每个 $\lambda _ { j } \in \Lambda$ 进行如下操作： 3: 在 $\mathcal { D } _ { \mathrm { c a l } }$上使用阈值 $\lambda _ { j }$ 运行已部署的程序；计算 $\widehat { R } _ { n } ( \lambda _ { j } )$ 和 p 值 $p _ { j }$。 4: 结束循环。 对 $\{ H _ { j } \} _ { j = 1 } ^ { m }$ 应用固定序列检验以控制 FWER，获得 $\Lambda _ { \mathrm { v a l i d } }$，并选择最具攻击性的 $\lambda ^ { \star } \in \Lambda _ { \mathrm { v a l i d } }$。 6: (B) 使用在线自我标定部署推理程序。 初始化快速权重 $\bar { \boldsymbol { W } } \boldsymbol { W } _ { 0 }$。 8: 对于 $t = 1 , \dots , T$： 9: 从推理状态获得当前隐藏表示 $\phi _ { t }$。 10: 计算探测分数 $s _ { t } = f { \big ( } \phi _ { t } ; W { \big ) }$。 11: 如果 $s _ { t } \subseteq \lambda ^ { \star }$，则： 12: 停止并输出 $\hat { z } \gets \mathrm { a n s } ( y _ { t } )$。 13: 返回 $z$。 14: 结束如果。 15: 设定伪目标 $C _ { t } \gets 0$ 并执行内循环更新 $W \gets W - \eta \nabla _ { W } \ell \bigl ( W ; \phi _ { t } \bigr )$。 16: 结束循环。 17: 如果预算耗尽，则返回 ans $\left( y _ { T } \right)$。

# 3.4 风险控制的保形推理与在线自我校准

根据吴等人（2025），校准是在停止决策规则（即完整的部署程序）上进行的，而不是在未经校准的原始分数上进行的。对于阈值 $\lambda \in \Lambda$，回忆由部署程序生成的评分过程 $s_{t}(x) := f(\phi_{t}; W_{t-1})$。定义停止时间及其相应的部署决策规则/程序输出，即在停止时间的最终答案尝试（或者，如果预算耗尽则为 ans $\left(y_{T}\right)$）。学习后测试（LTT）通过在一个保留的校准集上对整个程序 $\mathbf{\mathcal{A}}_{\lambda}$ 的风险进行校准来选择 $\lambda^{\star}$。特别地，LTT 从保守到激进地遍历网格 $\Lambda = {\bf \bar{\{}} \lambda_{1} > \cdots > \lambda_{m} \bf{\dot{>}} \}$，在每个 $\lambda$ 点测试经验风险是否超出容忍值，使用二项式 p 值进行测试，即我们测试 $\delta$ 为目标风险上限的地方，并构造 p 值，其中 $\epsilon$ 关于 $\{ H_{j} \}_{j=1}^{m}$ 拒绝阈值 $\lambda^{*}$。所选择的阈值满足

$$
\tau _ { \lambda } ( x ) : = \operatorname* { m i n } \{ t \leq T \colon s _ { t } ( x ) \geq \lambda \} ,
$$

$$
\ A _ { \lambda } ( x ) : = \mathrm { a n s } \big ( y _ { \tau _ { \lambda } ( x ) } \big ) ,
$$

$$
H _ { j } : \mathbb { E } [ R ( y _ { \tau _ { \lambda _ { j } } } ) ] \geq \delta ,
$$

$$
p _ { j } ^ { \mathrm { B T } } : = \mathbb { P } \big ( \mathrm { B i n o m } ( n , \delta ) \leq n \widehat { R } _ { n } \big ( \lambda _ { j } \big ) \big ) .
$$

$$
\mathbb { P } \left( \mathbb { E } [ R ( y _ { \tau _ { \lambda ^ { * } } } ) ] \le \delta \right) \ge 1 - \epsilon .
$$

因此，在可交换性下，FWER 控制可以在水平 $(\delta, \epsilon)$ 下实现有限样本风险控制。值得注意的是，这些保证适用于整个部署过程，包括推理链扩展、在线快速权重更新和基于阈值的停止（见图 1）。ORCA 的完整校准和推理时部署在算法 2 中呈现。我们在附录 A 中详细说明了理论保证，并在附录 B 中进行了更多讨论。

# 4 实验

我们评估在线推理校准框架在推理效率上的表现：给定风险容忍度 $\delta$，如何通过提前停止来节省计算量，同时保持答案质量？我们将其与 Wu 等人（2025）提出的静态线性探针进行了比较，涵盖了多个模型、标签模式和分布外基准。更多的实验结果和消融研究将在附录 C 中给出。

# 4.1 设置

数据集。我们通过结合三个来源构建了一个5000个样本的训练语料库：(i) 来自Muennighoff等人（2025）的s1K数据集，包含1000个数学问题；(ii) 来自OpenR1（Hugging Face，2025）的2000个问题；以及(iii) 来自DeepMath（He等人，2025）的2000个问题。问题按3:1:1的比例分为训练集（3000个）、校准集（1000个）和测试集（1000个）。对于分布外（OOD）评估，我们使用五个保留基准：MATH-500（Hendrycks等人，2021）、GPQA-Diamond（Rein等人，2024）（198个问题），以及AIME 2024/2025/2026（每个30个问题）。训练集和校准集中没有出现OOD问题。模型。我们的主要实验使用Qwen2.5-32B-Instruct（Yang等人，2024），在每个推理步骤提取平均池化的最后一层隐藏状态$ ( d _ { \phi } = 5 , 1 2 0 ) $。我们还在QwQ-32B（Team，2025）和Llama-3.3-70B-Instruct（Grattafiori等人，2024）上进行评估，$( d _ { \phi } = 8 , 1 9 2 ) $，以测试跨模型的泛化能力。推理轨迹由DeepSeek-R1-671B（Guo等人，2025a）生成，步骤标签由教师模型（Qwen-3-32B用于正确性，GPT-4.1用于评估）生成。标签模式。我们评估了两种标签策略，参考了Wu等人（2025）：监督型，其中$ C _ { t } ~ = ~ \mathbb { I } \{ z _ { t } $ 是正确的 $\}$ 需要真实标签；一致型，其中$ C _ { t } ~ = \mathbb { I } \{ z _ { t } = z _ { T } \} $ 将中间答案与全预算答案进行比较（不需要标签）。

度量指标。在每个风险容忍度 $\delta _ { \iota }$ 下，Learn-then-Test（LTT；Angelopoulos等，2021）校准确定阈值 $\lambda ^ { * }$ 。我们报告两个指标：节省 $= 1 - \bar { t } _ { \mathrm { s t o p } } / \bar { t } _ { \mathrm { t o t a l } }$，通过提前停止节省的推理步骤的比例。我们在C.3节验证步骤级和词元级节省高度一致，因此我们全程报告步骤级节省。$\pmb { \varrho }$ 错误率，即模型在其答案仍然错误时被停止的任务比例。由于步骤标签是累积的（在第一次正确尝试后翻转），因此仅过早停止会导致错误。这两个指标形成了一个自然的权衡，由阈值 $\bar { \lambda } ^ { * }$ 控制：较低的阈值提前停止，导致更高的节省但更高的错误风险；较高的阈值更为保守，错误更低但节省的计算量更少。LTT选择 $\lambda ^ { * }$ 以满足保证 $\begin{array} { r } { \mathbb { P } ( R \le \delta ) \ge 1 - \epsilon , } \end{array}$ 其中我们固定 $\epsilon = 0 . 0 \dot { 5 }$ 并调整风险容忍度 $\delta$。除非另有说明，我们在 $\delta = 0 . 1$ 时报告结果。训练和轮次选择。所有TTT-Probe变体使用Adam进行元训练（外层 $\ln = 1 0 ^ { - 3 }$），梯度裁剪为1.0，内层学习率 $\eta = 0 . 0 1$。我们为无-QK变体选择第20轮，为所有QK变体选择第10轮（详见C.4节）。得分轨迹使用10步的滚动窗口进行平滑处理。

Table 2: In-distribution early-stopping performance on the 5K test set (Qwen2.5-32B, ${ \epsilon } \mathrm { { = } } 0 . 0 5$ . TTT-Probe (no-QK) improves savings by $2 4 . 9 \%$ relative over the static baseline.   

<table><tr><td rowspan="2">Method</td><td colspan="2">δ = 0.05</td><td colspan="2">δ = 0.1</td><td colspan="2">δ = 0.15</td><td colspan="2">δ = 0.2</td></tr><tr><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td></tr><tr><td colspan="9">Supervised labels</td></tr><tr><td>Static Probe</td><td>.220</td><td>.055</td><td>.380</td><td>.105</td><td>.512</td><td>.159</td><td>.625</td><td>.208</td></tr><tr><td>TTT no-QK</td><td>.282</td><td>.053</td><td>.475</td><td>.110</td><td>.575</td><td>.152</td><td>.673</td><td>.192</td></tr><tr><td>TTT QK (d=128)</td><td>.233</td><td>.046</td><td>.414</td><td>.103</td><td>.560</td><td>.150</td><td>.674</td><td>.204</td></tr><tr><td colspan="9">Consistent labels (no ground truth)</td></tr><tr><td>Static Probe</td><td>.166</td><td>.049</td><td>.345</td><td>.098</td><td>.483</td><td>.156</td><td>.573</td><td>.197</td></tr><tr><td>TTT no-QK</td><td>.220</td><td>.045</td><td>.407</td><td>.096</td><td>.529</td><td>.141</td><td>.644</td><td>.193</td></tr><tr><td>TTT QK (dh=128)</td><td>.232</td><td>.064</td><td>.397</td><td>.113</td><td>.524</td><td>.150</td><td>.629</td><td>.187</td></tr></table>

# 4.2 在分布内的结果

表2（附录中的图2）比较了静态基线与TTT-Probe在四个风险水平上的表现。在$\delta = 0 . { \overset { \triangledown } { 1 } }$时，监督式TTT-Probe（无QK）相比静态基线节省了$4 7 . 5 \%$的推理步骤，而静态基线为$3 8 . 0 \%$，相对改善为$2 4 . 9 \%$。QK变体实现了$4 1 . 4 \%$和$8 . 9 \%$的相对改善。在一致性模式下，无QK探测器节省了$4 0 . 7 \%$相比$3 4 . 5 \%$，相对改善为$1 8 . 2 \%$，而无需任何真实标注数据。在所有四个$\delta$水平上，TTT-Probe始终优于基线。无论是无QK还是QK变体，其错误率始终维持在规定的$\delta$预算内或接近预算，证明在线适应提供了真实的校准改善。

# 4.3 分布外泛化

TTT-Probe 的一个关键动机是对分布转移的鲁棒性。表3评估了在5K语料库上训练的探针，并对五个OOD基准进行零-shot应用。在监督标签下，两个TTT变体在MATH-500上实现了强大的OOD泛化：no-QK节省了 $6 3 . 7 \%$，而QK节省了 $6 7 . 0 \%$，相比之下，基线为 $2 4 . 8 \%$（改善了 $2 . 6 { - } 2 . 7 \times$），同时保持错误率低于 $2 . 3 \%$。在GPQA-Diamond上，no-QK探针实现了 $7 1 . 5 \%$ 的节省。在一致标签下，QK变体在MATH-500上达到 $6 3 . 7 \%$ （是基线的 $2 . 7 \times$），显示出无标签的TTT在OOD部署中的可行性。

Table 3: O0D generalization at $\delta { = } 0 . 1$ The TTT-Probe achieves $2 . 6 { - } 2 . 7 \times$ the baseline savings on MATH-500 under supervised labels.   

<table><tr><td rowspan="2">Method</td><td colspan="2">MATH-500</td><td colspan="2">GPQA</td><td colspan="2">AIME&#x27;24</td><td colspan="2">AIME&#x27;25</td><td colspan="2">AIME&#x27;26</td></tr><tr><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td></tr><tr><td colspan="9">Supervised labels</td><td></td><td></td></tr><tr><td>Static Probe</td><td>.248</td><td>.008</td><td>.643</td><td>.270</td><td>.158</td><td>.050</td><td>.139</td><td>.000</td><td>.147</td><td>.050</td></tr><tr><td>TTT no-QK</td><td>.637</td><td>.023</td><td>.715</td><td>.300</td><td>.293</td><td>.150</td><td>.265</td><td>.056</td><td>.198</td><td>.050</td></tr><tr><td>TTT QK (d=128)</td><td>.670</td><td>.021</td><td>.665</td><td>.210</td><td>.295</td><td>.100</td><td>.258</td><td>.000</td><td>.134</td><td>.050</td></tr><tr><td colspan="9">Consistent labels (no ground truth)</td><td></td><td></td></tr><tr><td>Static Probe</td><td>.239</td><td>.004</td><td>.602</td><td>.328</td><td>.118</td><td>.033</td><td>.101</td><td>.000</td><td>.147</td><td>.100</td></tr><tr><td>TTT no-QK</td><td>.555</td><td>.012</td><td>.598</td><td>.318</td><td>.141</td><td>.033</td><td>.166</td><td>.067</td><td>.154</td><td>.067</td></tr><tr><td>TTT QK (dh=128)</td><td>.637</td><td>.016</td><td>.653</td><td>.328</td><td>.185</td><td>.033</td><td>.139</td><td>.000</td><td>.092</td><td>.000</td></tr></table>

# 4.4 跨模型性能

为了验证我们的发现是否超出了 Qwen2.5-32B 的范围，我们在 QwQ-32B $( d _ { \phi } = \bar { 5 } , 1 2 0 )$ 和 Llama-3.3-70B-Instruct $( d _ { \phi } = 8 , 1 9 2 )$ 上评估相同的配置。所有探测器均在各自模型的嵌入上独立训练和评估。两种 TTT-Probe 变体在所有三个模型上均一致地超越了静态基线。no-QK 探测器在 Qwen 上实现了相对提高 $2 4 . 9 \%$，在 ${ \mathrm { Q w Q } }$ 上提高了 $3 3 . 7 \%$，在 Llama 上提高了 $1 9 . 8 \%$。QK 变体在所有模型上也超过了基线 $( 6 . 8 \mathrm { - } 2 7 . 6 \%$ 相对)。所有错误率均在或接近 $\delta { = } 0 . 1$ 预算内，确认在线适应机制是模型无关的。

Table 4: Cross-model results $\delta \mathrm { = } 0 . 1 .$ , supervised). TTT-Probe consistently outperforms the static baseline across all three model families.   

<table><tr><td></td><td colspan="2">Qwen2.5-32B</td><td colspan="2">QwQ-32B</td><td colspan="2">Llama-3.3-70B</td></tr><tr><td>Method</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td></tr><tr><td>Static Probe</td><td>.380</td><td>.105</td><td>.295</td><td>.094</td><td>.354</td><td>.104</td></tr><tr><td>TTT no-QK</td><td>.475</td><td>.110</td><td>.394</td><td>.081</td><td>.424</td><td>.090</td></tr><tr><td>TTT QK (=128)</td><td>.414</td><td>.103</td><td>.376</td><td>.076</td><td>.378</td><td>.081</td></tr></table>

# 5 相关工作

高效的测试时间缩放。近期文献涵盖两个方向。首先，许多研究集中于减少后训练大型语言模型中的过度思考，无论是在训练阶段还是推理阶段。例子包括显式停止策略和计算感知生成控制（Guo et al., 2025b; Sui et al., 2025; Han et al., 2024; Hou et al., 2025; Yang et al., 2025; Zhang et al., 2025; Sun et al., 2025）。其次，平行轨迹之间的自一致性可用于对推理尝试进行排名或终止（Wang et al., 2022; Mitchell et al., 2022; Weng et al., 2023; Wang et al., 2024a），也可以与校准方法结合，以实现可控的效率提升（Xie et al., 2024; Huang et al., 2025; Liu et al., 2026）。我们的工作同样旨在有效地从语言模型中进行推理时间采样，但在范围上有所不同。尽管先前的工作校准了预测停止时机的探针，我们则对停止的端到端决策规则进行了保形化处理，涵盖了推理扩展和对采样动态的在线适应。

大语言模型推理的不确定性量化与校准。不确定性量化方法主要用于校准语言模型输出的自一致性（Rubin-Toles等，2025）、高质量（Quach等，2024；Qiu & Miikkulainen，2024；Jiang等，2024；Li等，2025b）和事实性（Mohri & Hashimoto，2024；Cherian等，2024a;b；Liu等，2024；Prinster等，2026）。一些近期论文将框架从大语言模型扩展到智能体推理（Feng等，2024；Sadhuka等，2025；Lee等，2026）。然而，这些方法是事后过滤文本集，而非在在线环境中引导解码。与此工作更为相似的是，一些方法对输出集的采样进行校准（Quach等，2024；Wu等，2025；Huang等，2026b；Xiong等，2025；Xie等，2025；Wang等，2026；Huang等，2026a），但这些方法通常假设推理步骤和提示的分布是静态的。我们的工作在于建模采样动态以及部署时提示分布的潜在变化。 测试时间训练与在线适应。测试时间训练（TTT）旨在通过学习在测试时间进行轻量级参数更新来改善在分布变化下的泛化能力，通常通过自监督损失实现（Sun等，2020）。该框架在设计高效架构（包括 RNNs 和线性变换器）时被广泛采用（Sun等，2024；Zhang等，2026），或者采用其他设计如样本特定向量（Hu等，2025b）、LoRA（Wang等，2024b）或输入困惑度最小化（Hu等，2025a）以在测试时间对语言模型进行适应。相比之下，我们首次将在线适应和测试时间训练引入到大语言模型推理的校准中。

# 6 结论

我们提出了在线推理校准（ORCA），这是一个统一的风险控制测试时间缩放框架，结合了在线测试时间训练与已部署停止规则的合适校准。其关键思想是将校准本身视为一个自适应预测问题：内循环执行特定实例的学习校准更新，而外循环则元学习初始化和更新动态，这些动态可以跨数据集迁移，并在分布转移下保持稳健。通过通过学习时间训练（LTT）校准完整的已部署过程，ORCA提供有限样本风险控制，同时启用自适应计算分配。在实证方面，ORCA在多个模型系列和基准测试中显著提高了效率，并在控制错误率方面表现出色，包括具有挑战性的零样本OOD设置。这些结果表明，动态更新的校准模块相比于静态置信度估计器，能够显著提高可靠性和计算效率。更广泛地说，这项工作强调了将元学习与合适决策整合以实现高效推理的实际方向，并证明联合设计适应性和校准能够产生有效且稳健的系统。

# 伦理声明

本论文涉及大型语言模型推理的技术方法，与任何伦理问题没有直接关系。此外，校准方法可以应用于产生更可靠的大型语言模型输出，这可能有利于大型语言模型使用中的伦理方面。

# References

Anastasios N Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511, 2021.

Anastasios N Angelopoulos, Stephen Bates, Emmanuel J Candès, Michael I Jordan, and Lihua Lei. Learn then Test: Calibrating predictive algorithms to achieve risk control. arXiv preprint arXiv:2110.01052, 2021.

John Cherian, Isaac Gibbs, and Emmanuel Candes. Large language model validity via enhanced conformal prediction methods. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024a. URL https: //openreview. net/forum?id=JD3NYpeQ3R.

John Cherian, Isaac Gibbs, and Emmanuel Candes. Large language model validity via enhanced conformal prediction methods. Advances in Neural Information Processing Systems, 37:114812114842, 2024b.

Shengyu Feng, Xiang Kong, Shuang Ma, Aonan Zhang, Dong Yin, Chong Wang, Ruoming Pang, and Yiming Yang. Step-by-step reasoning for math problems via twisted sequential monte carlo. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=Ze4aPP0tIn.

Yu Feng, Phu Mon Htut, Zheng Qi, Wei Xiao, Manuel Mager, Nikolaos Pappas, Kishaloy Halder, Yang Li, Yassine Benajiba, and Dan Roth. Diverseagententropy: Quantifying black-box ll unerainty throuh divers perspective andmultiagent interaction. Xi preprint arXiv:2412.09572, 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025b.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. Token-budget-aware LLM reasoning. arXiv preprint arXiv:2412.18547, 2024.

Zhiwei He, Tian Liang, Jiahao Xu, Qiuzhi Liu, Xingyu Chen, Yue Wang, Linfeng Song, Dian Yu, Zhenwen Liang, Wenxuan Wang, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu." Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning, 2025. URL https: / /arxiv. org/ abs/2504.11456.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Bairu Hou, Yang Zhan, Jiabao Ji, Yujian Liu, Kaizhi Qian, JacobAndreas, and Shiyu hang. ThinkPrune: Pruning long chain-of-thought of LLMs via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.

Jinwu Hu, Zhitian Zhang, Guohao Chen, Xutao Wen, Chao Shuai, Wei Luo, Bin Xiao, Yuanqing Li, and Mingkui Tan. Test-time learning for large language models, 2025a. URL https://arxiv.org/abs/2505.20633.

Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, and Guojun Qi. Slot: Sample-specific language model optimization at test-time, 2025b. URL https://arxiv.org/abs/2505.12392.

Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. Efficient test-time scaling via self-calibration. arXiv preprint arXiv:2503.00031, 2025.

Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, and Jiaxin Huang. CaTS: Calibrated test-time scaling for efficient LLM inference. In The Fourteenth International Conference on Learning Representations, 2026a. URL https: //openreview. net/forum?id= jrSc4RJXy1.

Jianguo Huang, Zhiyi Lyu, Fuxiang Zhang, Yanchen Deng, Deheng Ye, and Bo An. Uncertainty-aware tree search for efficient LLM reasoning, 2026b. URL https: // openreview.net/forum?id=RrLQbXCflj.

Hugging Face. Open r1: A fully open reproduction of deepseek-r1, January 2025. URL https://github.com/huggingface/open-r1.

Mingjian Jiang, Yangjun Ruan, Prasanna Sattigeri, Salim Roukos, and Tatsunori Hashimoto. Graph-based uncertainty metrics for long-form language model generations. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https: //openreview.net/forum?id=YgJPQw0lkO.

Nicholas Lee, Lutfi Eren Erdogan, Chris Joseph John, Surya Krishnapillai, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. Agentic test-time scaling for webagents. arXiv preprint arXiv:2602.12276, 2026.

Chengzu Li, Han Zhou, Goran Glava, Anna Korhonen, and Ivan Vuli. Large language models are miscalibrated in-context learners. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 1157511596, Vienna, Austria, July 2025a. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl. 603. URL https://aclanthology.org/2025.findings-acl.603/.

Yawei Li, David Rügamer, Bernd Bischl, and Mina Rezaei. Calibrating llms with informationtheoretic evidential deep learning, 2025b. URL https: //arxiv . org/abs/2502 . 06351.

Linyu Liu, Yu Pan, Xiaocheng Li, and Guanting Chen. Uncertainty estimation and quantification for llms: A simple supervised approach, 2024. URL https: //arxiv . org/abs/2404. 15993.

Zhangyi Liu, Huaizhi Qu, Xiaowei Yin, He Sun, Yanjun Han, Tianlong Chen, and Zhun Deng. Pets: A principled framework towards optimal trajectory allocation for efficient test-time self-consistency, 2026. URL https: //arxiv . org/abs/2602 . 16745.

Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 80868098, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.556. URL https://aclanthology.org/2022.ac1-1ong.556/.

Eric Mitchell, Joseph Noh, Siyan Li, Will Armstrong, Ananth Agarwal, Patrick Liu, Chelsea Finn, and Christopher D Manning. Enhancing self-consistency and performance of pretrained language models through natural language inference. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 17541768, 2022.

Christopher Mohri and Tatsunori Hashimoto. Language models with conformal factuality guarantees. In Proceedings of the 41st International Conference on Machine Learning, ICML'24. JMLR.org, 2024.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. s1: Simple test-time scaling. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 2027520321, 2025.

Harris Papadopoulos. Inductive conformal prediction: Theory and application to neural networks. In Tools in artificial intelligence. Citeseer, 2008.

Drew Prinster, Clara Fannjiang, Ji Won Park, Kyunghyun Cho, Anqi Liu, Suchi Saria, and Samuel Stanton. Conformal policy control. arXiv preprint arXiv:2603.02196, 2026.

Jianing Qi, Xi Ye, Hao Tang, Zhigang Zhu, and Eunsol Choi. Learning to reason across parallel samples for llm reasoning, 2025. URL https: //arxiv. org/abs/2506. 09014.

Xin Qiu and Risto Miikkulainen. Semantic density: Uncertainty quantification for large language models through confidence measurement in semantic space. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 134507134533, 2024.

Victor Quach, Adam Fisch, Tal Schuster, Adam Yala, Jae Ho Sohn, Tommi S Jaakkola, and Regina Barzilay. Conformal language modeling. In The Twelfth International Conference on Learning Representations, 2024.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. GPQA: A graduate-level Googleproof Q&A benchmark. In First Conference on Language Modeling, 2024.

Maxon Rubin-Toles, Maya Gambhir, Keshav Ramji, Aaron Roth, and Surbhi Goel. Conformal language model reasoning with coherent factuality. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview. net/forum?id= AJpUZd8C1b.

Shuvom Sadhuka, Drew Prinster, Clara Fannjiang, Gabriele Scalia, Aviv Regev, and Hanchen Wang. E-valuator: Reliable agent verifiers with sequential hypothesis testing. arXiv preprint arXiv:2512.03109, 2025.

Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal of Machine Learning Research, 9(3), 2008.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, et al. Stop overthinking: A survey on efficient reasoning for large language models. arXiv preprint arXiv:2503.16419, 2025.

Renliang Sun, Wei Cheng, Dawei Li, Haifeng Chen, and Wei Wang. Stop when enough: Adaptive early-stopping for chain-of-thought reasoning, 2025. URL https: //arxiv. org/ abs/2510.10103.

Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros, and Moritz Hardt. Testtime training with self-supervision for generalization under distribution shifts, 2020. URL https://arxiv.org/abs/1909.13231.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024.

Raphael Tang, Crystina Zhang, Xueguang Ma, Jimmy Lin, and Ferhan Ture. Found in the middle: Permutation self-consistency improves listwise ranking in large language models. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), June 2024.

Qwen Team. QwQ-32B: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github.io/blog/qwq-32b/.

Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world, volume 29. Springer, 2005.

Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 94269439, 2024a.

Xi Wang, Anushri Suresh, Alvin Zhang, Rishi More, William Jurayj, Benjamin Van Durme, Mehrdad Farajtabar, Daniel Khashabi, and Eric Nalisnick. Conformal thinking: Risk control for reasoning on a compute budget, 2026. URL https: //arxiv. org/abs/2602. 03814.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

Y. Wang, D. Ma, and D. Cai. With greater text comes greater necessity: Inference-time training helps long text generation, 2024b. URL https: //arxiv . org/abs/2401. 11504.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 25502575, 2023.

Menghua Wu, Cai Zhou, Stephen Bates, and Tommi Jaakkola. Thought calibration: Efficient and confident test-time scaling. arXiv preprint arXiv:2505.18404, 2025.

Yangxinyu Xie, Tao Wang, Soham Mallick, Yan Sun, Georgy Noarov, Mengxin Yu, Tanwi Mallick, Weijie J Su, and Edgar Dobriban. Statistical early stopping for reasoning models. In NeurIPS 2025 Workshop on Efficient Reasoning, 2025. URL https: //openreview. net/ forum?id=cmi4FBheRq.

Zhihui Xie, Jizhou Guo, Tong Yu, and Shuai Li. Calibrating reasoning in language models with internal consistency. arXiv preprint arXiv:2405.18711, 2024.

Jing Xiong, Qiujiang Chen, Fanghua Ye, Zhongwei Wan, Chuanyang Zheng, Chenyang Zhao, Hui Shen, Alexander Hanbo Li, Chaofan Tao, Haochen Tan, et al. Atts: Asynchronous test-time scaling via conformal prediction. arXiv preprint arXiv:2509.15148, 2025.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, and Weiping Wang. Dynamic early exit in reasoning models. arXiv preprint arXiv:2504.15895, 2025.

Fei Yu, Yingru Li, and Benyou Wang. Scaling flaws of verifier-guided search in mathematical reasoning, 2025. URL https://arxiv.org/abs/2502.00271.

Anqi Zhang, Yulin Chen, Jane Pan, Chen Zhao, Aurojit Panda, Jinyang Li, and He He. Reasoning models know when they're right: Probing hidden states for self-verification. arXiv preprint arXiv:2504.05419, 2025.

Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. Rest-mcts\*: Llm self-training via process reward guided tree search. Advances in Neural Information Processing Systems, 37:6473564772, 2024.

Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T. Freeman, and Hao Tan. Test-time training done right.  In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=Tb9qAxT3xv.

# Appendix

# A Theoretical Guarantees

We provide theoretical guarantees of risk control through ORCA in this section, with slightly more complicated and rigorous notations. The central question is whether intra-instance online updates invalidate Learn-then-Test (LTT) risk control. Our key observation is that LTT calibrates the entire deployed procedure as a black-box algorithm. As long as (i) the procedure resets its internal state across instances and (i) calibration and test runs are exchangeable under the same deployed procedure, finite-sample risk control remains valid.

Calibration and test data. Let $\mathcal { D } _ { \mathtt { c a l } } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ be a calibration set and let $\left( X _ { n + 1 } , Y _ { n + 1 } \right)$ be a fresh test point. For the p-value construction below, we assume $( X _ { i } , Y _ { i } )$ are i.i.d. from a distribution $\bar { P }$ (hence exchangeable), and independent of the random seeds used by the deployed algorithm. All trained outer-loop parameters (e.g., $\Theta _ { \mathrm { o u t e r } }$ and any fixed feature extractors) are treated as fixed constants independent of $\mathcal { D } _ { \mathrm { c a l } }$ (e.g., learned on disjoint data); equivalently, all guarantees hold conditional on these fixed parameters.

Deployed procedure as a randomized map. For each threshold $\lambda \in \Lambda ,$ define the entire deployed reasoning procedure as a (possibly randomized) mapping

$$
\mathcal { A } _ { \lambda } : \mathcal { X } \times \mathcal { U } \to \mathcal { V } , \qquad \hat { Y } = \mathcal { A } _ { \lambda } ( \boldsymbol { x } ; \boldsymbol { U } ) ,
$$

where $U \sim P _ { U }$ denotes all internal randomness (LLM sampling, search randomness, etc.). This mapping includes (i) initializing fast weights $W _ { 0 }  \hat { \Theta _ { \mathrm { o u t e r } } } ,$ (ii) computing step representations $\phi _ { t } ,$ (iii) computing scores $\begin{array} { r } { { s } _ { t } = f ( \phi _ { t } ; \mathbf { \breve { W } } _ { t - 1 } ) } \end{array}$ , (iv) updating fast weights within the instance, and (v) stopping at $\tau _ { \lambda } ( x ; U ) = \operatorname* { m i n } \{ t \leq T \colon s _ { t } ( x ; \hat { U } ) \geq \lambda \}$ .

Risk definition (includes algorithm randomness). Let $L : \mathcal { y } \times \mathcal { y }  \{ 0 , 1 \}$ be a 01 risk (e.g., incorrectness indicator). For a single instance $( X , Y )$ and randomness $U$ , define

$$
R ( \lambda ; X , Y , U ) : = L ( \mathcal { A } _ { \lambda } ( X ; U ) , Y ) \in \{ 0 , 1 \} .
$$

Let the marginal (deployment) risk of threshold $\lambda$ be

$$
r ( \lambda ) : = \mathbb { E } _ { ( X , Y ) \sim P } \mathbb { E } _ { U \sim P _ { U } } [ R ( \lambda ; X , Y , U ) ] .
$$

On the calibration set, for each $\lambda$ we run the deployed procedure independently per instance with fresh randomness $U _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } P _ { U }$ and obtain

$$
R _ { i } ( \lambda ) : = R ( \lambda ; X _ { i } , Y _ { i } , U _ { i } ) , \qquad i = 1 , \ldots , n .
$$

Lemma A.1 (Intra-instance adaptation preserves inter-instance exchangeability). Fix any threshold $\lambda \in \Lambda$ $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n + 1 }$ are exchangeable and $\{ U _ { i } \} _ { i = 1 } ^ { n + 1 }$ are id.andindependent of $\{ ( X _ { i } , Y _ { i } ) \} .$ I hen tthe sequence $\{ R _ { i } ( \lambda ) \} _ { i = 1 } ^ { n + 1 }$ with $R _ { i } ( \lambda ) : = R ( \lambda ; X _ { i } , Y _ { i } , U _ { i } )$ is exchangeable (indeed i.i.d. under the i.i.d. assumption).

Proof. Because the deployed procedure resets its internal state at the start of each instance, $( \boldsymbol { X _ { i } } , \boldsymbol { \dot { Y _ { i } } } , \boldsymbol { U _ { i } } ) \mapsto R ( \lambda ; \boldsymbol { X _ { i } } , \boldsymbol { Y _ { i } } , \boldsymbol { U _ { i } } )$ is the same measurable function applied separately to each triple. Exchangeability of $\{ ( X _ { i } , Y _ { i } , U _ { i } ) \}$ implies exchangeability of $\{ R _ { i } ( \lambda ) \}$ by closure of exchangeability under measurable mappings. □

Theorem A.2 (Finite-sample risk control of ORCA via LTT fixed-sequence testing). Let $\delta \in ( 0 , 1 )$ be a target risk level and $\epsilon \in ( 0 , 1 )$ a failure-probability level. Fix an ordered threshold grid $\dot { \Lambda } = \{ \lambda _ { 1 } > \breve { \lambda } _ { 2 } > \cdot \cdot \cdot > \lambda _ { m } \}$ For each $\lambda _ { j } ,$ consider the null hypothesis

$$
H _ { j } : \ r ( \lambda _ { j } ) \geq \delta \quad ( e q u i v a l e n t l y , t h e \ p r o c e d u r e \ i s \ \mathrm { n o t } \ \delta \ – s a f e ) .
$$

For each $j ,$ let $\begin{array} { r } { \widehat { r } _ { n } ( \lambda _ { j } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } R _ { i } ( \lambda _ { j } ) } \end{array}$ and define the one-sided binomial $p$ value

$$
p _ { j } : = \mathbb { P } \big ( \mathtt { B i n o m } ( n , \delta ) \leq n \widehat { r } _ { n } ( \lambda _ { j } ) \big ) .
$$

Assume $\{ R _ { i } ( \lambda _ { j } ) \} _ { i = 1 } ^ { n }$ are i.id. Beliwih mean $r ( \lambda _ { j } )$ (whichholdndethe assuptinse). Apply fixed-sequence testing (FST): test $H _ { 1 } , H _ { 2 } , \ldots$ in order at level $\epsilon _ { \iota }$ rejecting $H _ { j }$ if $p _ { j } \leq \epsilon ,$ and stop at the first $j$ such that $p _ { j } > \epsilon$ .Let $\hat { \jmath }$ be the last rejected index (or $\hat { \jmath } = 0$ if none are rejected) and output $\lambda ^ { \star } : = \lambda _ { \widehat { \jmath } }$ (the most aggressive rejected threshold). Then

$$
\begin{array} { r } { \mathbb { P } _ { \mathcal { D } _ { \mathrm { c a l } } } \mathopen { } \mathclose \bgroup \left( r \mathopen { } \mathclose \bgroup \left( \lambda ^ { \star } \aftergroup \egroup \right) \leq \delta \aftergroup \egroup \right) ~ \geq ~ 1 - \epsilon . } \end{array}
$$

Proof. Under $H _ { j }$ (i.e., $r ( \lambda _ { j } ) \geq \delta )$ , the p-value $p _ { j }$ is super-uniform: for any $\alpha \in [ 0 , 1 ]$ ,

$$
\mathbb { P } ( p _ { j } \le \alpha \mid H _ { j } ) \le \alpha
$$

(standard one-sided binomial test; see Angelopoulos et al. (2021)). In FST, if any true null is ever rejected, then the first true null in the ordered sequence must be rejected at level $\epsilon$ Therefore, by super-uniformity,

$$
\mathbb { P } ( \mathrm { F S T ~ r e j e c t s ~ a n y ~ t r u e ~ } H _ { j } ) \ \leq \ \epsilon ,
$$

i.e., FST controls the family-wise error rate (FWER) at level $\epsilon$ On the complement event (probability at least $1 - \epsilon )$ , every rejected $H _ { j }$ is false, hence $r ( \lambda _ { j } ) ~ < ~ \delta$ for all rejected indices, and in particular for the selected $\lambda ^ { \star }$ (the most aggressive rejected threshold). Thus $\begin{array} { r } { \mathbb { P } ( r ( \lambda ^ { \star } ) \le \delta ) \ge 1 - \epsilon } \end{array}$ . □

Remark A.3 (Marginal guarantee). Theorem A.2 provides marginal guarantee (expectation of all prompt distribution), not conditional guarantee (specific for a certain prompt).

Remark A.4 (General bounded risks). If one wishes to allow $L ( \cdot , \cdot ) \in [ 0 , 1 ]$ (not necessarily $\{ 0 , 1 \} )$ , the same proof structure holds by replacing the binomial p-values with any valid super-uniform $\mathrm { p }$ -values for the mean-risk null $r ( \lambda _ { j } ) \geq \delta$ (e.g., Hoeffding-type tests under independence), without changing the FST argument.

# B Discussions

Self-supervised inner loop at inference. A central design choice is that, during deployment (both calibration-set runs and test inference), the inner-loop update uses $\dot { C } _ { t } = 0$ at every non-stopping step. Formally, updates are applied only while $\bar { s } _ { t } < \lambda ^ { * }$ ;once $s _ { t } \geq \lambda ^ { * }$ , the procedure stops and no further gradient step is taken. This yields a fully self-supervised inference-time update rule, since no external label is required after deployment. It also improves traininginference consistency: in meta-training, all pre-transition steps (for which $C _ { t } ^ { \mathrm { t r u e } } = 0 ,$ ipeii only through the outer objective.

Detecting the reasoning breakthrough. For each problem at training time, we assume (after cumulative transformation) that the true step labels $C _ { t } ^ { \mathrm { t r u e } }$ follow a monotone binary sequence, $[ 0 , 0 , \ldots , 0 , 1 , 1 , \ldots , 1 ] ,$ with a single transition point at which the answer first becomes correct. The outer-loop objective $\bar { \sum _ { t } } ( s _ { t } - C _ { t } ) ^ { 2 }$ therefore encourages the probe to localize this transition, assigning low scores before the transition and high scores afterward.

At inference time, we do not assume access to $C _ { t } ^ { \mathrm { t r u e } }$ and we do not claim that "not stopping" implies $C _ { t } ^ { \mathrm { t r u e } } = 0$ Instead, we use the pseudo-target $C _ { t }$ to adapt the fast weights toward the instance-specific pre-transition baseline, which empirically acts as a novelty detector for the transition. Consequently, the probe accumulates an instance-specific representation of typical pre-transition reasoning patterns. When a substantive reasoning breakthrough occurs and the embedding $\phi _ { t }$ shifts accordingly, the probe score increases because the new state is no longer well explained by the current adaptation. Meta-training embeds this behavior into the slow parameters: $\dot { W } _ { 0 }$ sets the initialization and $\eta$ controls the adaptation rate to the problem-specific baseline. Under this view, the $C _ { t } { = } 0$ inner loop acts as an implicit novelty detector, where "novelty" corresponds to the transition pattern learned during meta-training. From another perspective, the entire deployment procedure can be treated as a binary classification task, hence there is no train-test gap at all.

Onieadaptation vs. on-policy valiity.Our metho isonline: during a single reasi trajectory, the fast weights are updated step-by-step, and the stopping decision is made from the current adapted state. This describes the algorithmic form of the method and does not, by itself, imply a distributional guarantee. By contrast, on-policy is the condition required by conformal/LTT calibration: calibration trajectories and deployment trajectories should be generated under the same effective policy (including search behavior, stopping logic, and update dynamics). Therefore, ORCÅ is best described as an online method with on-policy validity guarantees. If the deployment policy changes, one should re-calibrate under the new policy or apply explicit off-policy correction (e.g., importance weighting); otherwise, exchangeability assumptions can fail.

# C Additional Experiments

# C.1 Ablation Studies

![](images/2.jpg)  
Figure 2: Compute savings vs. risk tolerance $\delta$ for supervised (left) and consistent (right) labels (Qwen2.5-32B). TTT no-QK consistently outperforms the baseline across all risk levels, with the largest gap at low $\delta$ .

TTT is essential, not just the architecture. Table 5 ablates the TTT-Probe by comparing against two controls: (1) standard supervised training on the same architecture, and (2) random initialization without any training. The "standard" variants train the same probe architecture (no-QK or QK) via standard Adam optimization, using the same learning rate $( 1 0 ^ { - 3 } )$ and number of epochs (20 for no-QK, 10 for QK) as the corresponding TTT variants. At inference, standard-trained probes apply a single forward pass per step without online updates. The "no meta-training" variants use randomly initialized weights with online updates only; since no training occurs, these results are epoch-independent.

Standard training is insufficient; TTT meta-learning is the key contributor. The standardtrained no-QK probe achieves only $2 3 . 9 \%$ savings, substantially below the static baseline $( 3 8 . 0 \% )$ . Without $\mathrm { P C A } _ { \iota }$ training a linear model on the full 5,120-dimensional embedding space is prone to overfitting on 150K training samples. The standard-trained QK variant $( { \bar { 3 } } 9 . 4 \% )$ performs comparably to the static baseline, since its learned projection $( \theta _ { Q } \colon 5 , 1 2 0 \to$ 128) serves a similar role to PCA. Neither standard-trained variant benefits from online adaptation at inference. Replacing standard training with TTT meta-learning improves savings from $2 3 . 9 \%$ to $4 7 . 5 \%$ for no-QK (a $2 . 0 \times$ increase) and from $3 9 . 4 \%$ to $4 1 . { \overset { \cdot } { 4 } } \%$ for QK. The meta-learning outer loop trains the probe not just to classify correctness, but to produce scores that improve through online updates at inference. Without any training at all (random initialization), online updates alone achieve only $2 5 . 4 \%$ savings, confirming that meta-learned initialization is essential.

Figure 2 visualizes the full risksavings tradeoff in the in-distribution setting. The gap between TTT-Probe and the static baseline is most pronounced in the moderate-risk regime $( \delta \in [ 0 . 0 5 , 0 . 2 ] )$ , where the online adaptation provides the largest marginal benefit. At higher $\delta _ { . }$ , all methods converge as most steps can be skipped regardless of probe quality.

Table 5: Core mechanism ablation (supervised, $\delta \mathrm { = } 0 . 1$ . Standard supervised training on the same architecture underperforms the static baseline (no-QK) or only matches it (QK). TTT meta-learning with online updates is necessary for substantial improvement. Rows marked with $^ *$ use random initialization.   

<table><tr><td>Configuration</td><td>Architecture</td><td>Training</td><td>Online update</td><td>Savings</td><td>Error</td></tr><tr><td>Full TTT (no-QK)</td><td>Linear</td><td>Meta-learn</td><td>✓</td><td>.475</td><td>.110</td></tr><tr><td>Standard (no-QK)</td><td>Linear</td><td>Supervised</td><td></td><td>.239</td><td>.095</td></tr><tr><td>Full TTT (Q, h=128)</td><td>QK proj.</td><td>Meta-learn</td><td>√</td><td>.414</td><td>.103</td></tr><tr><td>Standard (QK, dh=128)</td><td>QK proj.</td><td>Supervised</td><td></td><td>.394</td><td>.108</td></tr><tr><td>No meta-training*</td><td>QK proj.</td><td>None</td><td>√</td><td>.254</td><td>.099</td></tr><tr><td>No meta + no update*</td><td>QK proj.</td><td>None</td><td></td><td>.173</td><td>.091</td></tr><tr><td>Static Probe</td><td>PCA+LogReg</td><td>Supervised</td><td></td><td>.380</td><td>.105</td></tr></table>

Together, meta-training and online updates are complementary: meta-training provides the foundation for calibration quality, while online updates enable instance-level adaptation.

Architecture variants. Table 6 evaluates alternative probe designs. QK architecture variants achieve competitive in-distribution savings (0.410.45). LayerNorm, residual, and shared QK variants reach 0.4490.451, approaching the no-QK probe (0.475).

Table 6: Architecture ablation (supervised, $d _ { h } = 1 2 8 , \delta = 0 . 1 )$ .   

<table><tr><td colspan="3"></td><td colspan="5">OOD Savings</td></tr><tr><td>Variant</td><td>Sav.</td><td>Err.</td><td>MATH</td><td>GPQA</td><td>A&#x27;24</td><td>A&#x27;25</td><td>A&#x27;26</td></tr><tr><td>QK (dh=128)</td><td>.414</td><td>.103</td><td>.670</td><td>.665</td><td>.295</td><td>.258</td><td>.134</td></tr><tr><td>+ LayerNorm</td><td>.451</td><td>.095</td><td>.697</td><td>.726</td><td>.265</td><td>.256</td><td>.144</td></tr><tr><td>+ LN + Residual</td><td>.450</td><td>.094</td><td>.697</td><td>.726</td><td>.265</td><td>.256</td><td>.144</td></tr><tr><td>+ Shared QK</td><td>.449</td><td>.094</td><td>.698</td><td>.724</td><td>.264</td><td>.256</td><td>.144</td></tr><tr><td>+ Learnable η</td><td>.421</td><td>.109</td><td>.679</td><td>.656</td><td>.364</td><td>.260</td><td>.104</td></tr><tr><td>+ MLP (2-layer)</td><td>.441</td><td>.093</td><td>.717</td><td>.633</td><td>.258</td><td>.231</td><td>.325</td></tr><tr><td>no-QK (ep20)</td><td>.475</td><td>.110</td><td>.637</td><td>.715</td><td>.293</td><td>.265</td><td>.198</td></tr></table>

The OOD results reveal complementary strengths across architectures. The MLP variant achieves the highest MATH-500 savings (0.717) and AIME'26 (0.325), while LayerNorm variants lead on GPQA (0.726). The no-QK probe is strongest on $\mathrm { A I M E } ^ { \prime } 2 5$ (0.265) and AIME'26 (0.198), while the learnable $\eta$ variant achieves the best $\mathrm { A I M E ^ { \prime } } 2 4$ savings (0.364).

The no-QK architecture remains the recommended default due to its simplicity (only $d _ { \phi } + 1$ parameters), stability across epochs, and consistently strong performance. QK variants with LayerNorm offer a viable alternative when MATH-style OOD generalization is prioritized.

Table 7: Effect of QK projection dimension (supervised, $\delta { = } 0 . 1$ ).   

<table><tr><td>dh</td><td>Parameters</td><td>Savings</td><td>Error</td></tr><tr><td>32</td><td>328K</td><td>.440</td><td>.105</td></tr><tr><td>64</td><td>656K</td><td>.439</td><td>.106</td></tr><tr><td>128</td><td>1.3M</td><td>.414</td><td>.103</td></tr><tr><td>256</td><td>2.6M</td><td>.408</td><td>.104</td></tr><tr><td>512</td><td>5.2M</td><td>.398</td><td>.099</td></tr><tr><td>no-QK</td><td>5.1K</td><td>.475</td><td>.110</td></tr></table>

Projection dimension. Table 7 evaluates the effect of QK projection dimension. The smallest QK dimension $( d _ { h } = 3 2 )$ achieves the highest QK savings (0.440) with the fewest parameters (328K), while larger dimensions show diminishing returns. The no-QK variant, with only 5.1K parameters, outperforms all QK dimensions in savings. This suggests that for the early-stopping task, a high-capacity projection is unnecessary; the raw hidden states already contain sufficient signal for adaptive confidence estimation.

In learnig r nsy. The -QK pro is akay robs o the in leg rate. Across a $\mathrm { \bar { 1 0 0 } } \times$ range $( \mathrm { l r ~ \bar { \epsilon } _ { \mathrm { ~ \tiny ~ ( ~ 0 . 0 0 1 , 0 . 0 0 5 , 0 . 0 1 , 0 . 0 5 , 0 . 1 } } } \} )$ ), supervised savings vary only between 0.461 and 0.463, a fluctuation of less than $0 . 5 \%$ In consistent mode, the invariance is even more extreme: al five learning rates produce identical savings of 0.418. This robustness eliminates a key hyperparameter from practical deployment.

# C.2 Supervised vs. Consistent Labels

Table 8: Supervised vs. consistent comparison( $\delta \mathrm { = } 0 . 1 $ , including average OOD savings and error across five benchmarks.   

<table><tr><td rowspan="3"></td><td colspan="4">Supervised</td><td colspan="4">Consistent</td></tr><tr><td colspan="2">In-dist</td><td colspan="2">OOD avg</td><td colspan="2">In-dist</td><td colspan="2">OOD avg</td></tr><tr><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td><td>Sav.</td><td>Err.</td></tr><tr><td>Static Probe</td><td>.380</td><td>.105</td><td>.267</td><td>.076</td><td>.345</td><td>.098</td><td>.241</td><td>.093</td></tr><tr><td>TTT no-QK</td><td>.475</td><td>.110</td><td>.422</td><td>.116</td><td>.407</td><td>.096</td><td>.323</td><td>.099</td></tr><tr><td>TTT QK (dh=128)</td><td>.414</td><td>.103</td><td>.404</td><td>.076</td><td>.397</td><td>.113</td><td>.341</td><td>.076</td></tr></table>

Supervised labels yield $1 0 – 1 7 \%$ higher savings than consistent labels on in-distribution data, reflecting the additional information provided by ground-truth correctness (Table 8). The no-QK probe achieves $4 7 . 5 \%$ savings in supervised mode and $4 0 . 7 \%$ in consistent mode, both substantial improvements over the static baseline $( 2 4 . 9 \%$ and $1 8 . 2 \%$ relative respectively). On OOD data, supervised no-QK averages $4 2 . 2 \%$ savings vs. $3 2 . 3 \%$ for consistent. The consistent TTT-Probe remains attractive for deployment where ground-truth labels are unavailable: it achieves a $3 4 . 0 \%$ relative improvement on OOD data over the consistent baseline while requiring only the model's own full-budget answers as supervision.

# C.3 Step-Level vs. Token-Level Savings

Table 9: Step-level vs. token-level savings $\delta \mathrm { = } 0 . 1 .$ , supervised). The two metrics are highly consistent across models.   

<table><tr><td>Configuration</td><td>Step Sav.</td><td>Token Sav.</td><td>∆</td></tr><tr><td>Qwen2.5-32B</td><td></td><td></td><td></td></tr><tr><td>Static Probe</td><td>.377</td><td>.377</td><td>.000</td></tr><tr><td>TTT no-QK</td><td>.475</td><td>.471</td><td>−.004 .000</td></tr><tr><td>TTT QK (dh=128)</td><td>.414</td><td>.414</td><td></td></tr><tr><td>QwQ-32B</td><td></td><td></td><td></td></tr><tr><td>Static Probe</td><td>.293</td><td>.296</td><td>+.003</td></tr><tr><td>TTT no-QK</td><td>.394</td><td>.397</td><td>+.003</td></tr><tr><td>TTT QK (dh=128)</td><td>.376</td><td>.380</td><td>+.003</td></tr><tr><td>Llama-3.3-70B</td><td></td><td></td><td></td></tr><tr><td>Static Probe</td><td>.352</td><td>.372</td><td>+.020</td></tr><tr><td>TTT no-QK</td><td>.424</td><td>.438</td><td>+.014</td></tr><tr><td>TTT QK (dh=128)</td><td>.378</td><td>.399</td><td>+.021</td></tr></table>

We verify that step-level savings translate to comparable token-level savings. Table 9 compares the two metrics for representative configurations at $\delta { = } 0 . 1$ . On Qwen and QwQ, step-level and token-level savings differ by less than 0.5 percentage points, confirming that reasoning steps are roughly uniform in length. On Llama, token-level savings are 12 percentage points higher than step-level savings, indicating that later reasoning steps tend to be longer; early stopping thus saves proportionally more tokens than steps. Given the close agreement, we report step-level savings throughout this paper.

![](images/3.jpg)  
Figure 3: Actual error rate vs. target risk $\delta$ (supervised, Qwen2.5-32B). All methods track the diagonal, confirming valid risk control. Points below the diagonal satisfy the LTT guarantee.

# C.4 Epoch Selection

Table 10 shows in-distribution savings at selected epochs. The no-QK probe is stable across epochs due to its small parameter count (5.1K), so we select epoch 20 where savings are near-peak. The QK variant $\scriptstyle { d _ { h } = 1 2 8 } ,$ 1.3M parameters) peaks at epoch 10 and degrades at later epochs. We therefore use epoch 10 for all QK variants.

Table 10: Savings at selected epochs (supervised, $\delta \mathrm { = } 0 . 1$ ). The no-QK probe is stable; QK peaks early and overfits.   

<table><tr><td>Epoch</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td></tr><tr><td>no-QK</td><td>.443</td><td>.475</td><td>.471</td><td>.476</td><td>.464</td></tr><tr><td>QK (dh=128)</td><td>.414</td><td>.387</td><td>.381</td><td>.357</td><td>.324</td></tr></table>

# C.5 Calibration Quality

LTT calibration guarantees that the selected decision rule has deployment risk at most $\delta$ with probability at least $1 - \epsilon$ Figure 3 validates this empirically by plotting the actual test-set error rate against the target $\delta$ .

All three methods closely track the $y = x$ diagonal, confirming that the LTT guarantee holds empirically. At low $\delta$ (0.05 to 0.15), all probes are slightly conservative, with actual error rates below the target risk level. This is expected since LT'T calibration optimizes for finitesample validity. Notably, TTT-Probe achieves higher savings (Figure 2) while maintaining the same calibration quality as the static baseline, indicating that online adaptation improves efficiency without degrading risk control.

Figure 4 shows the distribution of per-problem savings at $\delta { = } 0 . 1$ .All three methods exhibit high variance, with a significant mass near zero (problems where early stopping is not triggered) and near one (problems stopped very early). The TTT no-QK distribution has a higher mean (0.475) and median (0.444) than the static baseline (mean 0.377, median 0.313), confirming that the improvement is broadly distributed across problems rather than driven by a few outliers.

![](images/4.jpg)  
Figure 4: Distribution of per-problem savings at $\delta { = } 0 . 1$ (supervised, Qwen2.5-32B, 902 problems). Solid lines: mean; dashed lines: median. TTT no-QK shifts the distribution toward higher savings across the full range.

![](images/5.jpg)  
Figure 5: Probe score trajectories for a test problem (Qwen2.5-32B, $\delta \mathrm { = } 0 . 1$ . The green line marks the first correct step. The static probe (top) never crosses its threshold and saves $0 \%$ . The TTT no-QK probe (bottom) crosses the threshold at step 22 and saves $4 1 \%$ .

# C.6 Score Trajectory Analysis

Figure 5 compares the probe score trajectories on a representative test problem. The green vertical line marks the first correct reasoning step.

The static probe score rises gradually after the first correct step but remains below its threshold (0.77) throughout the entire trajectory. As a result, reasoning runs to completion with zero savings. The TTT no-QK probe starts at a higher score $( \sim 0 . 6 5 )$ due to meta-learned initialization and adapts online via the $C _ { t } { = } 0$ update rule. After the first correct step, the score increases and crosses the calibrated threshold (0.83) at step 22, stopping reasoning 16 steps early and saving $4 1 \%$ of compute. This example illustrates the two components at work: meta-training provides a good initialization, and online updates adapt the probe to the specific problem instance.