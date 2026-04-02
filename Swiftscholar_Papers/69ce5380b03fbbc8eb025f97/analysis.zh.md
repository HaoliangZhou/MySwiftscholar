# 1. 论文基本信息

## 1.1. 标题
**Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning**
（在线推理校准：测试时训练实现可泛化的共形大语言模型推理）

## 1.2. 作者
**Cai Zhou** (MIT), **Zekai Wang** (MIT), **Menghua Wu** (MIT), **Qianyu Julie Zhu** (MIT), **Flora C. Shi** (MIT), **Chenyu Wang** (MIT), **Ashia Wilson** (Boston University), **Tommi Jaakkola** (MIT), **Stephen Bates** (Boston University)。

这些作者主要来自麻省理工学院（MIT）的电气工程与计算机科学系（EECS）、计算机科学与人工智能实验室（CSAIL）以及计算科学与工程系（CSE），部分来自波士顿大学（Boston University）。他们在机器学习理论、共形预测以及大语言模型推理优化领域具有深厚的研究背景。

## 1.3. 发表期刊/会议
该论文目前发布于 **arXiv** (arXiv:2604.01170)，状态为预印本。考虑到其发表时间标注为 2026 年，这可能是一篇具有前瞻性的研究工作或特定会议的投稿稿件。

## 1.4. 发表年份
2026 年

## 1.5. 摘要
随着测试时计算规模的扩展，大语言模型（LLM）能够解决高难度任务，但最先进的结果伴随着高昂的计算成本。这种低效归因于后训练语言模型的校准失准，以及流行采样技术中缺乏校准。本文提出了 <strong>在线推理校准（ORCA）</strong> 框架，该框架借鉴了共形预测和测试时训练来校准采样过程。具体而言，作者引入了一种元学习过程，针对每个输入更新校准模块。这使得模型能够在分布偏移下（例如推理不同阶段的思维模式，或模型开发与部署之间的提示分布差异）提供有效的置信度估计。ORCA 不仅在共形风险上提供了理论保证，而且在各种推理任务中经验性地展示了更高的效率和泛化能力。在风险水平 $\delta=0.1$ 时，ORCA 在分布内任务上使用监督标签和自一致性标签分别将 Qwen2.5-32B 的效率提高了 47.5% 和 40.7%。在零样本域外设置下，它将 MATH-500 的节省率从静态校准基线的 24.8% 提高到 67.0%，同时保持了较低的实证错误率，这种趋势在不同模型族和下游基准测试中均成立。

## 1.6. 原文链接
https://arxiv.org/abs/2604.01170

# 2. 整体概括

## 2.1. 研究背景与动机
**核心问题：**
当前的大语言模型为了解决复杂的推理任务（如奥林匹克数学或软件工程），往往需要极大的测试时计算量。现有的测试时扩展策略（如并行采样、蒙特卡洛树搜索、自一致性等）虽然能提升性能，但存在严重的效率和可靠性瓶颈。这主要是因为：
1.  **模型失准：** 后训练的 LLM 对其中间状态和最终答案是否正确缺乏准确的置信度判断。
2.  **静态校准的局限性：** 现有的校准方法（如静态线性探测器）假设推理过程是静态的，忽略了推理过程中思维模式的动态变化以及部署环境与训练环境之间的分布偏移。

**研究动机与切入点：**
作者旨在解决上述挑战，提出一个原则性的方法，目标是：
1.  基于任务难度实现自适应计算分配，并在测试时提供样本质量和效率的统计保证。
2.  在分布偏移下保持鲁棒性，例如部署时的提示分布与开发时不同。

    本文的切入点是将 **校准本身视为一个可以在推理时优化的目标**。通过结合测试时训练（TTT）的在线适应能力和共形预测（LTT）的风险控制能力，构建一个能够随推理过程动态更新的校准框架。

## 2.2. 核心贡献/主要发现
**主要贡献：**
1.  **提出了 ORCA 框架：** 这是一个结合了测试时训练（TTT）和共形预测（LTT）的统一框架，用于风险可控的测试时扩展。
2.  **双层优化机制：**
    *   **内循环：** 在推理过程中，针对每个输入实例，通过快速权重在线更新校准模块（探测器），实现实例级别的自适应。
    *   **外循环：** 通过元学习学习共享的初始化参数和特征映射，确保内循环的更新在数据集层面保持稳定和可迁移。
3.  **全流程风险控制：** 首次将校准应用于包含推理链扩展、在线权重更新和基于阈值的停止规则的完整部署过程，并提供了有限样本的风险控制理论保证。

**关键发现：**
1.  **显著提升效率：** 在风险水平 $\delta=0.1$ 下，ORCA 在分布内任务上相比静态基线实现了显著的计算节省（监督模式下相对提升约 24.9%）。
2.  **强大的泛化能力：** 在零样本域外（OOD）设置下（如 MATH-500），ORCA 的节省率远超静态基线（从 24.8% 提升至 67.0%），证明了其对分布偏移的鲁棒性。
3.  **跨模型通用性：** 该方法在不同模型族（Qwen, QwQ, Llama）上均表现出一致的优越性。

# 3. 预备知识与相关工作

## 3.1. 基础概念
为了理解本文，我们需要掌握以下核心概念：

*   **测试时训练：**
    这是一种在推理时根据输入特征动态调整模型参数的技术。通常包含两层优化：
    *   **内循环：** 在处理单个序列时，通过最小化自监督或代理目标来更新一组“快速权重”。
    *   **外循环：** 在多个序列上通过元学习来学习共享的“慢速权重”（如初始化参数和特征映射），以确保内循环的更新是稳定且可迁移的。
    *   *直觉：* 就像学生在考试时，针对每一道题（输入）调整自己的解题思路（内循环），而平时的训练（外循环）教会了他们如何快速适应新题型。

*   **共形预测：**
    这是一种统计框架，用于在不需要强分布假设的情况下，为预测模型提供有限样本的覆盖率保证。它通过非一致性分数将预测转化为预测集。
    *   *直觉：* 如果我们要预测一个数值，与其给出一个单点估计，不如给出一个区间。共形预测保证这个区间包含真实值的概率至少是 $1-\epsilon$。

*   **学习后测试：**
    这是共形预测的一种变体，用于校准决策规则而非预测集。它在一组校准数据上测试关于风险的零假设，并选择满足风险控制要求的最激进的规则。
    *   *直觉：* 我们有很多备选的停止策略（有的激进省时间，有的保守保质量），LTT 帮我们选出那个在保证错误率不超过阈值的前提下最省时间的策略。

*   **思维链：**
    一种提示策略，引导模型通过生成中间推理步骤来解决复杂问题。这些步骤串联起来形成一条“链”。

## 3.2. 前人工作
作者提到了以下关键先前研究：
*   **测试时扩展：** 包括并行采样、顺序蒙特卡洛、验证器引导采样等。这些方法通过增加计算量来提升性能，但往往缺乏效率优化。
*   **不确定性量化与校准：** 现有工作主要关注校准输出是否自一致或高质量，或者过滤文本集合。类似的工作如 `Thought Calibration` (Wu et al., 2025) 将推理效率建模为风险控制的停止问题，但它们通常假设推理过程是静态的，无法处理分布偏移。
*   **测试时训练：** Sun et al. (2020) 提出了通过自监督损失进行测试时训练以改善分布偏移下的泛化。本文是首次将 TTT 引入 LLM 推理的校准中。

## 3.3. 技术演进
该领域的技术演进路径如下：
1.  **静态推理：** 模型直接生成答案，计算量固定。
2.  **测试时扩展：** 引入多路径采样、搜索算法（如 MCTS），通过暴力增加计算量换取准确率。
3.  **静态校准：** 引入静态探测器预测何时停止，以减少不必要的计算，但探测器参数固定。
4.  <strong>在线自适应校准（本文）：</strong> 引入 TTT 机制，使探测器参数随推理过程动态更新，并结合共形预测保证风险可控，实现了效率和鲁棒性的双重提升。

## 3.4. 差异化分析
本文方法与相关工作（特别是 `Thought Calibration`）的核心区别在于：
*   **动态性：** 传统方法校准的是固定的推理过程，而 ORCA 校准的是包含在线权重更新的完整动态算法。
*   **适应性：** ORCA 通过内循环实现了实例级别的在线适应，能够捕捉推理过程中思维模式的变化。
*   **泛化性：** 通过外循环的元学习，ORCA 在面对分布偏移（OOD）时表现出更强的鲁棒性。

# 4. 方法论

## 4.1. 方法原理
ORCA 的核心思想是将校准视为一个自适应预测问题。它利用测试时训练（TTT）的双层结构：
1.  **内循环：** 在推理轨迹中，针对每个步骤更新一个校准探测器。这个探测器根据当前的隐藏状态给出一个置信度分数，用于决定是否停止推理。
2.  **外循环：** 通过元学习，学习探测器的初始化参数和特征映射，使得内循环的更新在跨任务时是稳定且有效的。
3.  **风险控制：** 使用 LTT 对包含上述更新逻辑的完整部署过程进行校准，选择满足风险上限 $\delta$ 的停止阈值 $\lambda^*$。

    下图展示了 ORCA 的整体框架，包括外循环训练阶段和校准推理阶段：

    ![Figure 1: Framework of Online Reasoning Calibration (ORCA).](images/1.jpg)
    *该图像是示意图，展示了在线推理校准（ORCA）的框架。图中分为两个主要阶段：外循环TTT训练阶段和校准与推理阶段。在外循环阶段，显示了步骤标签、探测分数、探测权重和步骤嵌入的关系，而在校准与推理阶段，展示了输出如何依赖于停止阈值$\lambda^*$。整体结构帮助理解模型训练和推理的过程。*

## 4.2. 核心方法详解 (逐层深入)

### 4.2.1. 符号定义与设置
首先，我们需要明确文中使用的符号。以下是原文 Table 1 的符号总结：

以下是原文 Table 1 的结果：

<table>
<thead>
<tr>
<th>Symbol</th>
<th>Learned in</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>$\phi_t$</td>
<td>-</td>
<td>Step embedding: mean-pooled LLM hidden state at step t</td>
</tr>
<tr>
<td>$C_t$</td>
<td>-</td>
<td>Step label: correctness or consistency; $C_t=0$ at inference</td>
</tr>
<tr>
<td>$W_0, b_0$</td>
<td>outer loop</td>
<td>Initial probe weights, learned via meta-training</td>
</tr>
<tr>
<td>$W_t, b_t$</td>
<td>inner loop</td>
<td>Probe weights at step t, updated online during inference</td>
</tr>
<tr>
<td>$\theta_{Q,K}$</td>
<td>outer loop</td>
<td>Unified outer view parameters (optional learned projections or identity)</td>
</tr>
<tr>
<td>$\eta$</td>
<td>outer loop</td>
<td>Inner learning rate; fixed or learned</td>
</tr>
<tr>
<td>$S$</td>
<td>-</td>
<td>Probe score: `s_t = \sigma(f(\theta_Q \phi_t; W_t))`</td>
</tr>
<tr>
<td>$l(W;\phi_t)$</td>
<td>-</td>
<td>Inner-loop objective: $(s_t - C_t)^2$ (Brier score)</td>
</tr>
<tr>
<td>$\lambda$</td>
<td>calibration</td>
<td>LTT-calibrated stopping threshold</td>
</tr>
<tr>
<td>$\delta$</td>
<td>-</td>
<td>Target risk upper bound</td>
</tr>
<tr>
<td>$\epsilon$</td>
<td>-</td>
<td>LTT failure probability level</td>
</tr>
</tbody>
</table>

### 4.2.2. 内循环：在线自适应探测器
在推理的每一步 $t$，基础 LLM 产生一个隐藏表示 $\phi_t$。我们的探测器维护快速权重 $W_t$，这些权重沿着推理链在线更新，并在每一步产生一个置信度分数 $s_t \in [0, 1]$。过程遵循“先评分后更新”的协议。

首先，我们介绍内循环更新的基本形式。设 $f(\cdot; W): \mathbb{R}^{d_\phi} \to [0, 1]$ 为一个探测器模型（例如逻辑回归）。

**步骤 1：评分**
使用上一步的权重对当前步骤进行评分：
$$
s_t = f(\phi_t; W_{t-1})
$$
这里，$s_t$ 是模型认为当前答案正确的概率。

**步骤 2：计算内循环损失**
使用 Brier 分数作为损失函数，衡量预测分数与标签 $C_t$ 的差异：
$$
\ell(W_{t-1}; \phi_t) = (s_t - C_t)^2
$$
其中 $C_t \in \{0, 1\}$ 是步骤级别的标签。在训练时，它可以是监督标签（是否正确）或一致性标签（是否与最终答案一致）。在推理时，由于不知道真值，我们使用伪目标 $C_t=0$（详见讨论部分）。

**步骤 3：权重更新**
通过在线梯度下降更新权重：
$$
W_t = W_{t-1} - \eta \nabla_W \ell(W_{t-1}; \phi_t)
$$
这里 $\eta$ 是内循环学习率。这一步使得探测器能够适应当前问题实例的特定推理模式。

### 4.2.3. 外循环：元学习
仅靠实例级别的更新可能会过拟合局部噪声。外循环通过元学习学习共享的初始化和特征接口，使得内循环的适应具有迁移性。

外循环优化的目标是最小化训练数据上的期望损失：
$$
\operatorname*{min}_{\Theta_{outer}} \ \mathbb{E}_{x \sim \mathcal{D}_{train}} \mathcal{L}_{outer} \left( x, \{ W_t \}_{t=1}^T ; \Theta_{outer} \right)
$$
其中 $\Theta_{outer} = (\theta_{Q, K}, W_0, \eta)$ 是外层参数。外层损失定义为所有步骤上预测分数与真实标签差异的平方和：
$$
\mathcal{L}_{outer} \left( x, \{ W_t \}_{t=1}^T ; \Theta_{outer} \right) := \sum_{t=1}^T \left( s_t - C_t^{true} \right)^2
$$
约束条件是内循环的更新规则 `W_t = W_{t-1} - \eta \nabla_W \ell(W_{t-1}; x_t)`。

**QK 变体：**
为了增加表达能力，作者引入了 QK 变体，引入了可学习的投影 $\theta_K, \theta_Q$：
$$
\begin{array} { r l } { s_t = f( \theta_Q \phi_t ; W_{t-1} ), \quad W \in \mathbb{R}^{1 \times d_h} } \\ { \ell( W ; \phi_t ) = \left( f( \theta_K \phi_t ; W ) - C_t \right)^2 } \end{array}
$$
这里，更新方向（$\theta_K$）和评分方向（$\theta_Q$）可以关注隐藏状态的不同方面。

### 4.2.4. 风险控制的共形推理
ORCA 的关键在于它校准的是整个部署过程，而不仅仅是分数。给定一个阈值 $\lambda$，定义停止时间 $\tau_\lambda(x)$ 为分数首次超过阈值的时刻：
$$
\tau_\lambda(x) := \operatorname*{min}\{t \leq T : s_t(x) \geq \lambda\}
$$
对应的部署决策规则输出为停止时刻的答案：
$$
\mathcal{A}_\lambda(x) := \mathrm{ans}(y_{\tau_\lambda(x)})
$$

ORCA 使用 LTT 来选择 $\lambda^*$。LTT 在校准集上测试一系列阈值 $\Lambda = \{\lambda_1 > \dots > \lambda_m\}$，测试零假设 $H_j: \mathbb{E}[R(y_{\tau_{\lambda_j}})] \geq \delta$。其中 $R$ 是风险函数（如错误指示器）。

对于每个 $\lambda_j$，计算经验风险 $\widehat{R}_n(\lambda_j)$ 并构造二项式 p 值：
$$
p_j^{BT} := \mathbb{P} \left( \mathrm{Binom}(n, \delta) \leq n \widehat{R}_n(\lambda_j) \right)
$$
通过固定序列测试控制 FWER，选择最激进的被拒绝的阈值 $\lambda^*$。这保证了：
$$
\mathbb{P} \left( \mathbb{E} [ R ( y _ { \tau _ { \lambda ^ * } } ) ] \le \delta \right) \ge 1 - \epsilon
$$
这意味着在有限样本下，所选决策规则的风险以概率 $1-\epsilon$ 被控制在 $\delta$ 以下。

### 4.2.5. 算法流程
<strong>算法 1：训练阶段（外循环 TTT 训练）</strong>
1.  对于训练集中的每个提示 $x$：
2.  沿着推理轨迹展开内循环更新，得到 $\{W_t\}_{t=1}^T$。
3.  计算外层损失 $\mathcal{L}_{outer}$。
4.  通过反向传播更新可训练组件 $\Theta_{outer}$。

**算法 2：校准与推理阶段**
1.  **校准：** 在校准集上运行部署过程，使用 LTT 选择最优阈值 $\lambda^*$。
2.  **推理：**
    *   初始化快速权重 $W = W_0$。
    *   对于 $t = 1$ 到 $T$：
        *   获取隐藏状态 $\phi_t$。
        *   计算分数 `s_t = f(\phi_t; W)`。
        *   如果 $s_t \geq \lambda^*$，停止并输出答案。
        *   否则，设置伪目标 $C_t=0$ 并执行内循环更新 $W \leftarrow W - \eta \nabla_W \ell(W; \phi_t)$。

# 5. 实验设置

## 5.1. 数据集
实验使用了以下数据集：
*   **训练集：** 结合了三个来源构建的 5K 语料库：s1K (1k), OpenR1 (2k), DeepMath (2k)。按 3:1:1 分割为训练、校准和测试集。
*   <strong>域外（OOD）评估集：</strong>
    *   **MATH-500：** 高中数学竞赛问题。
    *   **GPQA-Diamond：** 研究生级别的科学问题。
    *   **AIME 2024/2025/2026：** 美国数学邀请赛问题。
        OOD 数据未出现在训练或校集中。

## 5.2. 评估指标
论文主要关注以下两个指标：

1.  **节省率：**
    *   **概念定义：** 指通过提前停止推理所节省的计算步骤（或词元）的百分比。它量化了方法的效率提升。
    *   **数学公式：** $\text{Savings} = 1 - \frac{\bar{t}_{stop}}{\bar{t}_{total}}$
    *   **符号解释：** $\bar{t}_{stop}$ 是平均停止步数，$\bar{t}_{total}$ 是总预算步数。

2.  **错误率：**
    *   **概念定义：** 模型在答案尚不正确的步骤停止的比例。它量化了提前停止带来的质量风险。
    *   **数学公式：** $\text{Error Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i \neq y_i^*)$
    *   **符号解释：** $N$ 是样本总数，$\hat{y}_i$ 是模型停止时的输出，$y_i^*$ 是真实标注数据。

## 5.3. 对比基线
论文主要将 ORCA 与以下基线进行比较：
*   <strong>Static Probe (静态探测器)：</strong> Wu et al. (2025) 提出的方法，使用 PCA 降维后接逻辑回归，参数固定，不进行在线更新。

# 6. 实验结果与分析

## 6.1. 核心结果分析
实验结果表明，ORCA 在分布内（ID）和分布外（OOD）场景下均显著优于静态基线。在保持错误率低于目标风险 $\delta$ 的同时，实现了更高的计算节省。

### 6.1.1. 分布内结果
以下是原文 Table 2 的结果，展示了在分布内测试集上的性能：

以下是原文 Table 2 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">δ = 0.05</th>
<th colspan="2">δ = 0.1</th>
<th colspan="2">δ = 0.15</th>
<th colspan="2">δ = 0.2</th>
</tr>
<tr>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9"><b>Supervised labels</b></td>
</tr>
<tr>
<td>Static Probe</td>
<td>.220</td>
<td>.055</td>
<td>.380</td>
<td>.105</td>
<td>.512</td>
<td>.159</td>
<td>.625</td>
<td>.208</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.282</td>
<td>.053</td>
<td>.475</td>
<td>.110</td>
<td>.575</td>
<td>.152</td>
<td>.673</td>
<td>.192</td>
</tr>
<tr>
<td>TTT QK (d=128)</td>
<td>.233</td>
<td>.046</td>
<td>.414</td>
<td>.103</td>
<td>.560</td>
<td>.150</td>
<td>.674</td>
<td>.204</td>
</tr>
<tr>
<td colspan="9"><b>Consistent labels (no ground truth)</b></td>
</tr>
<tr>
<td>Static Probe</td>
<td>.166</td>
<td>.049</td>
<td>.345</td>
<td>.098</td>
<td>.483</td>
<td>.156</td>
<td>.573</td>
<td>.197</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.220</td>
<td>.045</td>
<td>.407</td>
<td>.096</td>
<td>.529</td>
<td>.141</td>
<td>.644</td>
<td>.193</td>
</tr>
<tr>
<td>TTT QK (dh=128)</td>
<td>.232</td>
<td>.064</td>
<td>.397</td>
<td>.113</td>
<td>.524</td>
<td>.150</td>
<td>.629</td>
<td>.187</td>
</tr>
</tbody>
</table>

**分析：**
在 $\delta=0.1$ 时，监督模式下的 TTT no-QK 实现了 47.5% 的节省率，相比静态基线的 38.0% 提升了约 24.9%。一致性模式下的 TTT no-QK 也达到了 40.7% 的节省率。所有方法的错误率都控制在 $\delta$ 附近或以下，验证了风险控制的有效性。

下图展示了计算节省与风险容忍度 $\delta$ 的关系。可以看到 TTT no-QK 在所有风险水平下均优于基线。

![Figure 2: Compute savings vs. risk tolerance $\\delta$ for supervised (left) and consistent (right) labels (Qwen2.5-32B). TTT no-QK consistently outperforms the baseline across all risk levels, with the largest gap at low $\\delta$ .](images/2.jpg)
*该图像是一个图表，展示了在监督标签（左）和一致性标签（右）情况下的计算节省与风险容忍度 `oldsymbol{ heta}` 之间的关系。TTT no-QK 在所有风险水平下均优于基线，特别是在低风险容忍度时差距最大。*

### 6.1.2. 域外泛化结果
以下是原文 Table 3 的结果，展示了在 OOD 基准测试上的性能：

以下是原文 Table 3 的结果：

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">MATH-500</th>
<th colspan="2">GPQA</th>
<th colspan="2">AIME'24</th>
<th colspan="2">AIME'25</th>
<th colspan="2">AIME'26</th>
</tr>
<tr>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="11"><b>Supervised labels</b></td>
</tr>
<tr>
<td>Static Probe</td>
<td>.248</td>
<td>.008</td>
<td>.643</td>
<td>.270</td>
<td>.158</td>
<td>.050</td>
<td>.139</td>
<td>.000</td>
<td>.147</td>
<td>.050</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.637</td>
<td>.023</td>
<td>.715</td>
<td>.300</td>
<td>.293</td>
<td>.150</td>
<td>.265</td>
<td>.056</td>
<td>.198</td>
<td>.050</td>
</tr>
<tr>
<td>TTT QK (d=128)</td>
<td>.670</td>
<td>.021</td>
<td>.665</td>
<td>.210</td>
<td>.295</td>
<td>.100</td>
<td>.258</td>
<td>.000</td>
<td>.134</td>
<td>.050</td>
</tr>
<tr>
<td colspan="11"><b>Consistent labels (no ground truth)</b></td>
</tr>
<tr>
<td>Static Probe</td>
<td>.239</td>
<td>.004</td>
<td>.602</td>
<td>.328</td>
<td>.118</td>
<td>.033</td>
<td>.101</td>
<td>.000</td>
<td>.147</td>
<td>.100</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.555</td>
<td>.012</td>
<td>.598</td>
<td>.318</td>
<td>.141</td>
<td>.033</td>
<td>.166</td>
<td>.067</td>
<td>.154</td>
<td>.067</td>
</tr>
<tr>
<td>TTT QK (dh=128)</td>
<td>.637</td>
<td>.016</td>
<td>.653</td>
<td>.328</td>
<td>.185</td>
<td>.033</td>
<td>.139</td>
<td>.000</td>
<td>.092</td>
<td>.000</td>
</tr>
</tbody>
</table>

**分析：**
在 MATH-500 上，TTT no-QK 达到了 63.7% 的节省率，TTT QK 达到了 67.0%，远超静态基线的 24.8%。这证明了 ORCA 在 OOD 场景下的强大泛化能力。

### 6.1.3. 跨模型性能
以下是原文 Table 4 的结果，展示了在不同模型上的表现：

以下是原文 Table 4 的结果：

<table>
<thead>
<tr>
<th></th>
<th colspan="2">Qwen2.5-32B</th>
<th colspan="2">QwQ-32B</th>
<th colspan="2">Llama-3.3-70B</th>
</tr>
<tr>
<th>Method</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Static Probe</td>
<td>.380</td>
<td>.105</td>
<td>.295</td>
<td>.094</td>
<td>.354</td>
<td>.104</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.475</td>
<td>.110</td>
<td>.394</td>
<td>.081</td>
<td>.424</td>
<td>.090</td>
</tr>
<tr>
<td>TTT QK (=128)</td>
<td>.414</td>
<td>.103</td>
<td>.376</td>
<td>.076</td>
<td>.378</td>
<td>.081</td>
</tr>
</tbody>
</table>

**分析：**
ORCA 在 Qwen, QwQ 和 Llama 三个模型族上均一致优于静态基线，证明了其模型无关性。

## 6.2. 消融实验/参数分析

### 6.2.1. 核心机制消融
以下是原文 Table 5 的结果，展示了 TTT 机制、元训练和在线更新的必要性：

以下是原文 Table 5 的结果：

<table>
<thead>
<tr>
<th>Configuration</th>
<th>Architecture</th>
<th>Training</th>
<th>Online update</th>
<th>Savings</th>
<th>Error</th>
</tr>
</thead>
<tbody>
<tr>
<td>Full TTT (no-QK)</td>
<td>Linear</td>
<td>Meta-learn</td>
<td>✓</td>
<td>.475</td>
<td>.110</td>
</tr>
<tr>
<td>Standard (no-QK)</td>
<td>Linear</td>
<td>Supervised</td>
<td></td>
<td>.239</td>
<td>.095</td>
</tr>
<tr>
<td>Full TTT (Q, h=128)</td>
<td>QK proj.</td>
<td>Meta-learn</td>
<td>√</td>
<td>.414</td>
<td>.103</td>
</tr>
<tr>
<td>Standard (QK, dh=128)</td>
<td>QK proj.</td>
<td>Supervised</td>
<td></td>
<td>.394</td>
<td>.108</td>
</tr>
<tr>
<td>No meta-training*</td>
<td>QK proj.</td>
<td>None</td>
<td>√</td>
<td>.254</td>
<td>.099</td>
</tr>
<tr>
<td>No meta + no update*</td>
<td>QK proj.</td>
<td>None</td>
<td></td>
<td>.173</td>
<td>.091</td>
</tr>
<tr>
<td>Static Probe</td>
<td>PCA+LogReg</td>
<td>Supervised</td>
<td></td>
<td>.380</td>
<td>.105</td>
</tr>
</tbody>
</table>

**分析：**
*   标准监督训练（无在线更新）效果很差（23.9%），甚至不如静态基线。
*   仅在线更新而无元训练效果有限（25.4%）。
*   完整的 TTT（元训练 + 在线更新）效果最好（47.5%），证明了两者缺一不可。

### 6.2.2. 架构变体
以下是原文 Table 6 的结果，展示了不同架构变体对 OOD 性能的影响：

以下是原文 Table 6 的结果：

<table>
<thead>
<tr>
<th colspan="3"></th>
<th colspan="5">OOD Savings</th>
</tr>
<tr>
<th>Variant</th>
<th>Sav.</th>
<th>Err.</th>
<th>MATH</th>
<th>GPQA</th>
<th>AIME'24</th>
<th>AIME'25</th>
<th>AIME'26</th>
</tr>
</thead>
<tbody>
<tr>
<td>QK (dh=128)</td>
<td>.414</td>
<td>.103</td>
<td>.670</td>
<td>.665</td>
<td>.295</td>
<td>.258</td>
<td>.134</td>
</tr>
<tr>
<td>+ LayerNorm</td>
<td>.451</td>
<td>.095</td>
<td>.697</td>
<td>.726</td>
<td>.265</td>
<td>.256</td>
<td>.144</td>
</tr>
<tr>
<td>+ LN + Residual</td>
<td>.450</td>
<td>.094</td>
<td>.697</td>
<td>.726</td>
<td>.265</td>
<td>.256</td>
<td>.144</td>
</tr>
<tr>
<td>+ Shared QK</td>
<td>.449</td>
<td>.094</td>
<td>.698</td>
<td>.724</td>
<td>.264</td>
<td>.256</td>
<td>.144</td>
</tr>
<tr>
<td>+ Learnable η</td>
<td>.421</td>
<td>.109</td>
<td>.679</td>
<td>.656</td>
<td>.364</td>
<td>.260</td>
<td>.104</td>
</tr>
<tr>
<td>+ MLP (2-layer)</td>
<td>.441</td>
<td>.093</td>
<td>.717</td>
<td>.633</td>
<td>.258</td>
<td>.231</td>
<td>.325</td>
</tr>
<tr>
<td>no-QK (ep20)</td>
<td>.475</td>
<td>.110</td>
<td>.637</td>
<td>.715</td>
<td>.293</td>
<td>.265</td>
<td>.198</td>
</tr>
</tbody>
</table>

**分析：**
不同架构在不同任务上各有千秋。例如，MLP 变体在 MATH-500 上表现最好，而 LayerNorm 变体在 GPQA 上表现最好。no-QK 变体虽然参数少，但在 AIME 上表现稳健。

### 6.2.3. 校准质量与轨迹分析
下图展示了实际错误率与目标风险 $\delta$ 的关系。所有方法都紧密跟踪对角线，验证了 LTT 保证的有效性。

![Figure 3: Actual error rate vs. target risk $\\delta$ (supervised, Qwen2.5-32B). All methods track the diagonal, confirming valid risk control. Points below the diagonal satisfy the LTT guarantee.](images/3.jpg)
*该图像是图表，展示了实际错误率与目标风险 $δ$ 之间的关系（监督，Qwen2.5-32B）。不同方法的表现由曲线表示，图中标注了过于自信和良好校准的区域，所有方法都跟踪对角线，确认了风险控制的有效性。图中点位于对角线下方满足 LTT 保证。*

下图展示了每个问题的节省率分布。TTT no-QK 的分布整体向高节省率方向移动。

![Figure 4: Distribution of per-problem savings at $\\delta { = } 0 . 1$ (supervised, Qwen2.5-32B, 902 problems). Solid lines: mean; dashed lines: median. TTT no-QK shifts the distribution toward higher savings across the full range.](images/4.jpg)
*该图像是图表，显示了在 `oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{oldsymbol{0. 1 }}`oldsymbol{oldsymbol{Static probe}}\$ 统计每个问题的节省。 TT{ T no-QK 使得节省在全范围内更高。*

下图对比了一个测试问题的探测器分数轨迹。静态探测器从未超过阈值，节省率为 0%。而 TTT no no-QK 探测器在第 22 步超过阈值，节省了 41% 的计算。这直观地展示了在线适应如何捕捉推理过程中的“突破”时刻。

![Figure 5: Probe score trajectories for a test problem (Qwen2.5-32B, $\\delta \\mathrm { = } 0 . 1$ . The green line marks the first correct step. The static probe (top) never crosses its threshold and saves $0 \\%$ . The TTT no-QK probe (bottom) crosses the threshold at step 22 and saves $4 1 \\%$ .](images/5.jpg)
*该图像是图表，展示了测试问题（Qwen2.5-32B, $ heta = 0.1$）的探测评分轨迹。静态探测器在第38步停止，未能跨越阈值，保存率为0%；而TTT无QK探测器在第22步过阈，保存率为41%。绿色线表示第一次正确步骤。*

# 7. 总结与思考

## 7.1. 结论总结
本文提出了 ORCA 框架，通过将测试时训练（TTT）与共形预测（LTT）相结合，实现了 LLM 推理的高效且风险可控的测试时扩展。ORCA 的核心创新在于将校准本身视为一个可在线优化的目标：内循环进行实例级别的自适应校准，外循环通过元学习确保这种适应性是可迁移的。实验表明，ORCA 在多个模型族和基准测试上显著优于静态校准基线，特别是在分布外（OOD）场景下展现出强大的泛化能力。

## 7.2. 局限性与未来工作
*   **局限性：**
    *   虽然整体计算节省显著，但在线更新本身引入了一定的计算开销。不过，净收益仍然是巨大的。
    *   方法依赖于可交换性假设，如果部署策略发生剧烈变化，可能需要重新校准。
*   **未来工作：**
    *   探索不同的内循环更新规则或损失函数。
    *   将 ORCA 与更复杂的搜索算法（如 MCTS）结合。
    *   研究在非推理任务（如长文本生成）中的应用。

## 7.3. 个人启发与批判
*   **启发：** 这篇论文最大的启发在于“校准即优化”的视角。通常我们将校准看作是一个静态的映射或后处理步骤，但 ORCA 证明了将其动态化、在线化可以带来巨大的收益。特别是利用 $C_t=0$ 作为伪目标进行在线更新，实际上是将探测器训练成了一个“新颖性检测器”，当推理状态发生显著变化（即出现正确答案的征兆）时，分数会上升。这是一种非常巧妙的无需标签的自适应机制。
*   **批判：** 尽管结果令人印象深刻，但 TTT 的训练过程涉及双层优化，训练复杂度和资源消耗可能较高。此外，对于极其复杂的推理任务，简单的线性探测器（即使是 TTT 版本）可能无法捕捉足够细微的特征，此时可能需要更复杂的探测器架构（如文中的 MLP 变体）。未来的研究可以关注如何降低 TTT 的训练门槛，以及如何更自动地选择适合当前任务的探测器架构。