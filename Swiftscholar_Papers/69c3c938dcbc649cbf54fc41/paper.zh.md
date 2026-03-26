# SpecEyes：通过推测感知和规划加速智能多模态大语言模型

黄浩宇1，黄金发2，万中伟³，郑晓伍1，纪戎戎1，骆杰博2，1厦门大学，2罗彻斯特大学，3俄亥俄州立大学 \* 共同贡献者

# 摘要

代理多模态大语言模型（MLLMs）（例如，OpenAI o3 和 Gemini Agentic Vision）通过迭代的视觉工具调用实现了显著的推理能力。然而，级联的感知、推理和工具调用循环引入了显著的顺序开销。这种开销被称为代理深度，导致了高昂的延迟，并严重限制了系统级的并发性。为此，我们提出了SpecEyes，一个代理级的推测加速框架，旨在打破这一顺序瓶颈。我们的关键洞见是，一个轻量级、无工具的MLLM可以作为推测规划器来预测执行轨迹，从而在不牺牲准确性的情况下提前终止昂贵的工具链。为了调节这种推测规划，我们引入了一种基于答案可分离性的认知门控机制，该机制量化模型自我验证的信心，而无需依赖参考标签。此外，我们设计了一种异构并行漏斗，利用小模型的无状态并发性来掩盖大模型的有状态串行执行，最大化系统吞吐量。在 $\mathrm { V } ^ { \ast }$ Bench、HR-Bench 和 POPE 上开展的广泛实验表明，SpecEyes在保持或甚至提高准确率（高达 $+ 6 . 7 \%$ ）的同时，实现了相较于代理基线的 ${ \bf 1 . 1 - 3 . 3 5 \times }$ 加速，提升了并发负载下的服务吞吐量。邮箱：Jinfa Huang jhuang90@ur.rochester.edu，Haoyu Huang huanghaoyu@stu.xmu.edu.cn 代码：github.com/MAC-AutoML/SpecEyes

# 1 介绍

多模态大语言模型（MLLMs）经历了范式的转变，从静态的单次视觉感知到与视觉世界的动态、有自主性的交互。早期的MLLMs仅对图像进行一次编码，并在单次前向传播中生成响应，将视觉视为被动输入通道。近期的突破性进展根本上改变了这种设计：模型主动调用外部感知工具（例如，放大、裁剪、光学字符识别）来形成感知、推理和工具调用的迭代循环，从而逐步完善其理解。这种自主性范式在需要细致检视、多步组合推理和主动信息获取的挑战性视觉任务中表现出色。然而，赋予自主性MLLMs的机制同时引入了严重的效率危机。如图1所示，每个查询都会触发一系列工具调用步骤的级联，这一数量我们称之为自主深度 $D$ ，其中每一步依赖于前一步的观察。这种严格的数据依赖对系统性能造成了双重灾难：（i）延迟爆炸：单个查询的端到端响应时间与 $D$ 线性增长，因为每个推理与工具循环必须在下一个开始之前完成；（ii）并发崩溃：由于每个查询的工具使用链变更每个查询的状态，从而有效地抵消了GPU批处理，自主模型每次只能针对一个查询前进一步，导致大量硬件并行性闲置。因此，这些影响使得自主性MLLMs的速度比非自主性模型慢几个数量级，从而对现实世界的应用部署构成了根本性障碍。

![](images/1.jpg)  
Fig. 1. Motivation and overview of SpecEyes. Top: Agentic MLLMs evaluate each query via a Markovian sequence of stateful tool invocations of depth $D$ . This strict causal dependency prohibits parallelization, imposing a serving complexity of $\mathcal { O } ( B D C )$ for $B$ queries, where $C$ denotes the tool per-step inference cost. Bottom: SpecEyes enables agentic-evel speculative bypass with a stateless small model and an answer-separability gate. Here, $\beta$ is the fraction of tool-free candidates after screening (Sec. 3.4) and $\alpha$ is the acceptance rate of speculative answers among them (Secs. 3.2 and 3.3), averaging $8 0 \%$ and $7 1 \%$ across all benchmarks, respectively. All reported accuracy and speedup values are averaged across V $^ *$ [52], HR-Bench [50], and POPE [26].

现有的高效推理方法未能有效解决这一瓶颈。词元级别的推测解码通过让一个小型草稿模型为大型模型提出词元，从而加速单个生成步骤。然而，这些方法仍然在固定的推理轨迹内操作：智能体的管道本身，即感知与推理的多轮循环，依然完全是串行的，每个工具仍必须按顺序调用。另外，草稿/验证互动的增加往往会扩展生成的轨迹（更长的词元序列和额外的回合），引入显著的额外开销，可能抵消实际中的逐步加速。同样，多模态词元修剪和时间压缩在固定模型内减少了每步的计算，但并未消除主导智能体延迟的重复工具调用。总之，所有以往的方法都在智能体循环内操作，没有质疑是否每个查询都必须经过这一循环。在本文中，我们做出了一个概念性的飞跃：我们将推测范式从词元/语义水平提升到智能体水平。我们的关键观察是，针对智能体大语言模型的大部分查询实际上并不需要深度工具辅助推理。相反，一个轻量级的无工具视觉模型可以仅凭原始图像正确回答这些查询，前提是我们能够可靠地识别出哪些查询属于这一类别。这激励了一个异构的“快速思考，慢速思考”架构：一个小型非智能体模型快速生成推测答案（“直觉”即快速思考），而大型智能体模型则专门用于真正需要多步骤工具交互的查询（慢速思考）。

我们通过引入SpecEyes这一代理级别的推测性加速框架来具体化这个想法，旨在多模态推理。该框架包括三个紧密集成的组件：(1) 一个四阶段的推测性管道（第3.2节），通过启发式工具使用判断、小模型推测、基于置信度的切换和代理回退来处理每个查询。(2) 认知门控（第3.3节），通过一种新颖的答案可分离性度量$S _ { \mathrm { s e p } }$来实现，该度量评估前$K$个logits之间的竞争边际，为信任小模型的输出提供无校准、尺度不变的决策边界。(3) 一个异构并行服务架构（第3.4节），并行运行无状态小模型，仅将低置信度查询转发给代理模型，将推测接受率转化为乘法吞吐量增益。在V $^ *$ Bench、HR-Bench和POPE上的大量实验表明，SpecEyes在显著降低延迟和提高吞吐量的同时，保持了代理管道的完整准确性。总之，我们做出了以下贡献：我们识别并形式化了代理多模态大语言模型的有状态瓶颈，显示出工具使用链中固有的数据依赖性对每个查询的延迟和系统级并发性造成了根本性障碍。我们提出SpecEyes，这是首个将推测性加速从词元级提升到代理级的框架，绕过不需要工具使用循环的查询，同时保持完整的准确性。我们引入了基于前$K$个logits之间答案可分离性的认知门控，为小模型决定何时信任自身输出与何时升级到代理模型提供了一种无标签、尺度不变的标准。我们设计了一个异构并行漏斗，利用小模型的无状态特性实现并发查询处理，使得吞吐量增益与推测接受率成正比。

# 2 相关工作

智能体多模态大语言模型。语言模型中的智能体推理源于工具增强的框架，该框架将行动生成与外部反馈交错组合。基于此，多模态大语言模型（MLLMs）采用了类似的智能体范式，使得通过外部视觉工具主动交错感知与推理成为可能，而不是依赖于被动的单次编码。早期的大规模MLLMs建立了智能体扩展所依赖的主干架构。DeepEyes展示了强化学习可以训练模型在推理过程中调用感知工具；后续工作通过代码生成和视觉操作实现了可执行推理，并通过多轮交互和自我反思进一步扩展了智能体的深度。尽管这些方法有效，但它们依赖于深度顺序的感知推理工具循环，从而引入了显著的延迟和有限的并发性，这是一个以前研究大多忽视的系统级瓶颈。高效推理。词元级的推测解码通过让一个小型草稿模型提出词元供更大模型验证，从而加速生成。最近的扩展将这一思想应用于协作推理：SpecReason将较简单的步骤委托给一个通过语义一致性验证的轻量级模型；RelayLLM在关键步骤动态调用更强的专家；SpecTemp和MSD减少在多模态和交互设置中的冗余视觉处理。自适应计算和提前退出方法进一步绕过某些层以处理简单输入。尽管所有这些方法在固定轨迹内加速步骤，但智能体循环本身仍然完全是串行的。

高效的多模态感知。一系列平行研究减少了多模态感知的每步计算负担。基于频率的压缩截断高频视觉信号；令牌剪枝通过注意力得分或多模态相关性保留视觉显著的令牌；动态稀疏化优化跨层的保留。令牌合并通过合并冗余表示来减少序列长度，而在视频设置中利用帧间的时间冗余来合并或剪枝空间令牌。KV缓存压缩还通过驱逐缓存的视觉键和值来降低内存和解码成本。尽管取得了这些进展，但所有这些方法仍运行在单一模型中，并保持了顺序智能体管道不变，因为大型模型仍需执行完整的感知推理循环。相比之下，SpecEyes 关注的是智能体层面的效率，而不是加速管道内的单个操作，它通过一个轻量级的非智能体模型以一种认知门控机制来推测性地绕过整个不太有用的循环。这一设计打破了现有智能体 MLLM 的严格顺序依赖，实现异构的并行执行，从而最大限度地提高硬件利用率，并显著改善延迟和系统级吞吐量。

# 3 方法论

我们首先对智能体多模态推理中固有的有状态瓶颈进行形式化描述（第3.1节），然后介绍我们的四阶段推测加速框架SpecEyes（第3.2节）。接着，我们详细阐述了控制推测旁路的认知门控机制（第3.3节），最后描述了最大化系统吞吐量的异构并行架构（第3.4节）。

![](images/2.jpg)  
Fig. 2. Pipeline overview of SpecEyes. A batch of $B$ queries passes through a four-phase funnel. I: $\mathcal { M } _ { L }$ screens tool necessity, splitting queries into tool-free and tool-required. II: A stateless $\mathcal { M } _ { S }$ speculatively answers all tool-free queries with token-level logits. III: An answer separability score $S _ { \mathrm { s e p } }$ gates each answer; those above $\tau$ are accepted directly. IV: Remaining queries fall back to the full agentic loop. The funnel yields ${ \approx } 1 / ( 1 { - } \beta \alpha ) \times$ throughput speedup.

# 3.1 模型化智能大语言模型的状态瓶颈

初步研究。我们将具备代理特性的多模态大型语言模型（MLLM）形式化为一个有状态的推理系统 $\mathcal{A} = ( \mathcal{S}, \mathcal{T}, \pi )$，其中 $\boldsymbol{S}$ 表示状态空间，$\mathcal{T} = \{ t_{1}, \ldots, t_{N} \}$ 是一个有限的感知工具集合（例如，放大、裁剪、光学字符识别），而 $\pi$ 是联合选择工具调用并生成推理词元的策略。给定一个查询和输入图像 $I$，模型在 $D$ 个推理步骤 $q$ 之上保持一个状态轨迹 $\{ s_{0}, s_{1}, \ldots, s_{D} \}$。初始状态为 $s_{0} = ( q, I )$。在每一步 $d$，策略生成一个动作 $ \boldsymbol{a}_{d} = \pi( \boldsymbol{s}_{d} )$，该动作要么调用一个工具 $t \in \tau$，要么输出一个最终答案。当调用一个工具时，状态转移如下：

$$
\boldsymbol { s } _ { d + 1 } = \boldsymbol { f } ( \boldsymbol { s } _ { d } , t _ { d } ( \boldsymbol { s } _ { d } ) ) ,
$$

其中 $t _ { d } ( s _ { d } )$ 将选择的工具 $t _ { d }$ 应用到当前的视觉上下文（例如，从 $I$ 中裁剪出感兴趣区域），$f$ 则将得到的观察结果融合到下一个状态中。我们将 $D$ 称为查询的智能深度。状态依赖性与序列瓶颈。方程 (1) 的一个关键特性是后续工具选择在因果上依赖于先前的观察。具体来说，设 $t _ { d + 1 } \sim \pi ( \cdot \mid s _ { d + 1 } )$ 为第 $d + 1$ 步选择的工具。由于 $s d \substack { + 1 }$ 包含了 $t _ { d }$ 的输出，因此马尔可夫链 $( s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , . . . )$ 形成了严格的数据依赖性：

$$
a _ { 0 } , \dots , s _ { d } ) = p ( a _ { d + 1 } \mid s _ { d } , t _ { d } ( s _ { d } ) ) \neq p ( a _ { d + 1 } \mid s _ { 0 } )
$$

这种依赖关系使得智能体管道本质上是顺序的：步骤 $d { + 1 }$ 不能在步骤 $d$ 完成之前开始。因此，单个查询的端到端延迟与智能体深度呈线性关系：

$$
L _ { \mathrm { a g e n t } } ( q ) = \sum _ { d = 0 } ^ { D ( q ) } \big ( \underbrace { c _ { \mathrm { l l m } } } _ { \mathrm { r e a s o n i n g } } + \underbrace { c _ { \mathrm { t o o l } } ( t _ { d } ) } _ { \mathrm { p e r c e p t i o n } } \big ) ,
$$

其中 $c_{ \mathrm{l l m} }$ 和 $c_{ \mathrm{t o o l} }( t_{d} )$ 分别表示步骤 $d$ 时 LLM 推理和工具执行的延迟。吞吐量的影响。在系统层面，这种严格的串行化也限制了并发性。考虑一个包含 $B$ 个查询的服务场景 $\mathcal{Q} = \{ q_{1}, \dots, q_{B} \}$。由于每个查询的状态特性，大型智能体模型 $\mathcal{A}$ 每次只能处理一条工具使用循环，从而导致每个查询的占用时间为 $L_{ \mathrm{a g e n t} }( q_{i} )$。因此，最大吞吐量受到以下限制：

$$
\Theta _ { \mathrm { a g e n t } } \leq \frac { B } { \sum _ { i = 1 } ^ { B } L _ { \mathrm { a g e n t } } ( q _ { i } ) } .
$$

随着平均智能体深度 $D$ 的增加，这一界限变得越来越严格，这促使我们采取措施来投机性地消除不必要的工具调用。

# 3.2 SpecEyes：智能体级别的推测推理

我们的关键见解是，并非所有查询都需要深度代理推理。对于相当一部分输入，一个小型非代理的多语言模型 $\mathcal{M}_{S}$ 可以直接从原始图像 $I$ 中生成正确答案，而无需任何工具调用。SpecEyes 利用这一观察，通过一个四阶段流程（见图2）来推测性地绕过昂贵的工具链，每当 $\mathcal{M}_{S}$ 有足够的信心时，否则则回退到完整的代理模型 $\mathcal{M}_{L}$。我们将小型非代理模型称为 $\mathcal{M}_{S}$，而大型代理多语言模型称为 $\mathcal{M}_{L} = \mathcal{A}$。这四个连续阶段的逐步执行将在下文系统性详细说明。阶段I：启发式工具使用判断。给定查询 $q$ 和图像 $I$，大型代理模型 $\mathcal{M}_{L}$ 首先确定是否需要调用工具。我们用一个轻量级的二分类头来提示 $\mathcal{M}_{L}$：

$$
g ( q , I ) = \mathcal { M } _ { L } ( q , I ; \mathcal { P } _ { \mathrm { j u d g e } } ) \in \{ 0 , 1 \} ,
$$

其中 $\mathcal { P } _ { \mathrm { j u d g e } }$ 是一个提示，指示模型评估工具的必要性，$g = 0$ 表示 $\mathcal { M } _ { L }$ 判断查询可以仅通过全局图像回答，而 $g = 1$ 表示可能需要借助工具的感知能力。$g = 0$ 的查询直接进入第二阶段；$g = 1$ 的查询则立即转发到第四阶段（智能体回退）。虽然第一阶段由 $\mathcal { M } _ { L }$ 执行，但它仅生成一个单一的二元标记，不涉及工具调用，因此开销微乎其微。我们使用 $\mathcal { M } _ { L }$ 而非 $\mathcal { M } _ { S }$，因为其工具调用能力使其成为更可靠的工具必要性评判者，从而实现更准确的筛选。第二阶段：推测性预测。对于通过第一阶段的查询（即 $g = 0$），$\mathcal { M } _ { S }$ 直接生成答案 ${ \hat { y } } _ { S }$ 以及完整的输出对数分布：

$$
\hat { y } _ { S } , \{ \ell ^ { ( n ) } \} _ { n = 1 } ^ { \lvert \hat { y } _ { S } \rvert } = \mathcal { M } _ { S } ( q , I ) ,
$$

其中 $\ b { \ell } ^ { ( n ) } \in \mathbb { R } ^ { | \mathcal { V } | }$ 是第 $n ^ { \mathrm { t h } }$ 个生成词元的词汇 $\nu$ 上的 logits 向量。重要的是，这一推理是无状态的：它无需执行任何工具，并且可以并发处理批次中的所有查询。第三阶段：小型 MLLM 自信度切换。第二阶段的 logits 被传递到一个认知门控函数 $S _ { \mathrm { s e p } }$（详见第 3.3 节），该函数在不需要真实标签的情况下量化 $\mathcal { M } _ { S }$ 的答案信心。我们计算推测答案 ${ \hat { y } } _ { S }$ 的标量可分离性得分。

$$
\mathrm { d e c i s i o n } = \left\{ \begin{array} { l l } { \mathrm { a c c e p t } ~ \hat { y } _ { S } , } & { \mathrm { i f } ~ S _ { \mathrm { s e p } } ( \hat { y } _ { S } ) \geq \tau , } \\ { \mathrm { f a l l b a c k ~ t o } ~ \mathcal { M } _ { L } , } & { \mathrm { i f } ~ S _ { \mathrm { s e p } } ( \hat { y } _ { S } ) < \tau , } \end{array} \right.
$$

其中 $\tau$ 是在小规模保留验证集上校准的阈值。被接受的回答会被立即返回，完全绕过智能体流水线；被拒绝的查询则进入第四阶段。第四阶段：智能体回退。未能通过置信度切换的查询会被路由到完整的智能体模型 $\mathcal { M } _ { L }$，该模型执行完整的状态感知-推理循环：

$$
\begin{array} { r } { \hat { y } _ { L } = \mathcal { M } _ { L } ( q , I ) = \pi \big ( s _ { 0 } \xrightarrow [ ] { t _ { 0 } } s _ { 1 } \xrightarrow [ ] { t _ { 1 } } \cdot \cdot \cdot \xrightarrow [ ] { t _ { D - 1 } } s _ { D } \big ) . } \end{array}
$$

智能体模型保留对所有工具 $\tau$ 的完全访问权限，并以顺序延迟 $L _ { \mathrm { a g e n t } } ( q )$ 为代价执行多步推理。根据设计，第四阶段充当安全网：将低置信度查询路由回完整的智能体管道显著减轻了潜在的准确性损失，即使相较于基线的边际性能差距仍然存在，这是由于门控机制的不完善性质。端到端延迟。设 $\beta \in [ 0 , 1 ]$ 表示第一阶段的无工具筛选比例，$\alpha \in [ 0 , 1 ]$ 表示第三阶段的认知门接收率。所有查询均需承担判断成本 $c _ { J }$；仅通过第一阶段的 $\beta$ 部分还需额外承担小模型成本 $c _ { S }$；余下的 $( 1 - \beta \alpha )$ 部分转发至 $M _ { L }$，支付完整的智能体成本 $L _ { \mathrm { a g e n t } }$。因此，在 SpecEyes 下，期望每个查询的延迟为：

$$
\mathbb { E } [ L _ { \mathrm { S p e c E y e s } } ] = c _ { J } + \beta c _ { S } + \left( 1 - \beta \alpha \right) L _ { \mathrm { a g e n t } } ,
$$

当 $c _ { J } + \beta c _ { S } \ll L _ { \mathrm { a g e n t } }$ 时。当 $\beta \alpha$ 较大时（例如，$\beta \alpha > 0.6$），预期延迟主要受轻量级前端成本的影响，从而在纯智能体基线之上实现显著的加速。

# 3.3 通过答案可分离性实现小型多语言模型的认知门控

SpecEyes 的有效性在于第三阶段中置信度切换机制的质量。我们现在引入答案可分离性评分 $S _ { \mathrm { s e p } }$，它作为认知门。基于概率的置信度的局限性。一个常见的基于概率的序列生成置信度通过几何平均聚合每个词元的最大 softmax 概率 [66]。具体而言，对于第 $n$ 个生成的词元，其 logits 为 $\ell ^ { ( n ) }$，我们定义最大 softmax 概率 $p _ { \mathrm { m a x } } ^ { ( n ) }$ 为：

$$
p _ { \operatorname* { m a x } } ^ { ( n ) } = \operatorname* { m a x } _ { v \in \mathcal { V } } \sigma ( \ell ^ { ( n ) } ) _ { v } ,
$$

其中 $\sigma ( \cdot )$ 表示 softmax 操作，$\nu$ 是词汇表。总体信心计算为：

$$
S _ { \log } ( \hat { y } _ { S } ) = \exp \left( \frac { 1 } { \vert \hat { y } _ { S } \vert } \sum _ { n = 1 } ^ { \vert \hat { y } _ { S } \vert } \log p _ { \operatorname* { m a x } } ^ { ( n ) } \right) ,
$$

$\{ p _ { \mathrm { m a x } } ^ { ( n ) } \}$ 然而，$S _ { \mathrm { l o g } }$ 是众所周知的softmax错误标定问题，其中大幅度的logit可能导致过于自信的概率；(2) 在低熵或几乎确定性的位置（例如标点符号、格式化词元）上，逐词的 $p _ { \mathrm { m a x } } ^ { ( n ) }$ 可能会虚假地偏高，几何聚合并未明确测量顶级预测与强竞争者之间的分离程度。这些问题增加了我们投机绕过中误接受的风险。答案分离性评分。我们设计了一个指标，不依赖于原始softmax概率，用于评估深度生成模型。对于第 $n ^ { \mathrm { th } }$ 生成的词元 $\ell ^ { ( n ) }$，我们有 $\ell _ { [ 1 ] } ^ { ( n ) } \geq \ell _ { [ 2 ] } ^ { ( n ) } \geq \cdots \geq \ell _ { [ | \mathcal { V } | ] } ^ { ( n ) }$，通过将领先的logit标准化与其最近竞争者进行比较来进行词元级分离性定义：

$$
S _ { \mathrm { s e p } } ^ { ( n ) } = \frac { \ell _ { [ 1 ] } ^ { ( n ) } - \mu _ { K } ^ { ( n ) } } { \sigma _ { K } ^ { ( n ) } + \epsilon } ,
$$

其中 $\mu(K)$ 和 $\sigma(n)$ 表示 $K$ 个 logits $\{ \ell _ { [ 1 ] } ^ { ( n ) } , \ldots , \ell _ { [ K ] } ^ { ( n ) } \}$，$\epsilon > 0$ 是 $S _ { \mathrm { s e p } } ^ { ( n ) }$ 离其最近竞争对手的差值：较大的值表示明确的决策边界，而较小的值则表明 $S _ { \mathrm { s e p } } ^ { ( n ) }$ 是尺度不变的，因为分子和分母都随 logits 的幅度线性缩放，从而中和了 softmax 的标定伪影；(i) 它通过方差项 $\sigma _ { K } ^ { ( n ) }$ 明确建模顶尖候选者之间的竞争格局，提供了更具信息的置信度。基于 token 的 $S _ { \mathrm { s e p } } ^ { ( n ) }$ 必须在所有生成的 $| \hat { y } _ { S } |$ token 之间聚合，以获得答案级别的置信度。我们考虑三种自然的聚合策略：

$$
S _ { \mathrm { s e p } } ^ { \mathrm { m e a n } } = \frac { 1 } { | \hat { y } _ { S } | } \sum _ { n = 1 } ^ { | \hat { y } _ { S } | } S _ { \mathrm { s e p } } ^ { ( n ) } , \quad S _ { \mathrm { s e p } } ^ { \mathrm { m i n } } = \operatorname* { m i n } _ { n \in [ | \hat { y } _ { S } | ] } S _ { \mathrm { s e p } } ^ { ( n ) } , \quad S _ { \mathrm { s e p } } ^ { \mathrm { b o t t o m } } = \frac { 1 } { | \mathcal { B } | } \sum _ { n \in \mathcal { B } } S _ { \mathrm { s e p } } ^ { ( n ) } ,
$$

其中 $\boldsymbol { \beta }$ 是 $S _ { \mathrm { s e p } } ^ { ( n ) }$ 值的 $r$ 比率，即 $| B | = \lceil r \rceil \hat { y } _ { S } | \rceil$，其中 $r \in ( 0 , 1 )$ 是经验选择的比率。聚合得分随后通过 sigmoid 函数进行归一化。我们将最小聚合作为默认策略，基于以下风险理论论证：命题 1. 令 $\hat { y } _ { S } = ( y _ { 1 } , \dots , y _ { | \hat { y } _ { S } | } )$ 为投机答案。定义答案级别错误事件 $\mathcal { E } = \bigcup _ { n } { \mathcal { E } } _ { n }$，其中 $\xi _ { n }$ 表示令牌 $y _ { n }$ 错误的事件。然后：

$$
P ( \mathcal { E } ) = P { \bigg ( } \bigcup _ { n } \mathcal { E } _ { n } { \bigg ) } \leq \sum _ { n } P ( \mathcal { E } _ { n } ) .
$$

如果每个 $P ( \mathcal { E } _ { n } )$ 在 $S _ { s e p } ^ { ( n ) }$ 中是单调递减的，那么对 $\mathrm { m i n } _ { n } S _ { s e p } ^ { ( n ) }$ 进行阈值处理确保每个词元都超过置信度阈值，从而在三种策略中最紧密地界定了联合概率 $P ( \mathcal { E } )$。直观上，mi 策略充当了一个最坏情况的保护：在任何答案显示可分离性时都进行回退。这种保守设计优先考虑精确度（避免错误接受），以维护智能体管道的准确性保证。

# 3.4 异构并行性以加速吞吐量

除了每个查询延迟的减少，SpecEyes 通过将四个阶段组织成一个异构并行漏斗，解耦无状态并发与有状态执行，从而实现系统级的吞吐量提升。 批量并行前端。我们以大小为 $B$ 的批次处理请求。设 $\beta \in [ 0 , 1 ]$ 为第一阶段筛选的无工具查询的比例（$g$ =0），$\alpha \in [ 0 , 1 ]$ 为在这些候选中认知门的接受率。筛选（第一阶段，延迟 $c _ { J }$）和推测推理（第二阶段，延迟 $c _ { S }$）都是无状态单回合前向传递，因此完全可以进行批量并行化，前端的并行成本为 $c _ { J } + c _ { S }$。漏斗形服务。被接受的查询 $( \alpha \beta B )$ 会立即返回；其余的残余集 $_ { \mathcal { R } }$，包括被门控拒绝的查询和需要工具的查询，返回到顺序智能体执行。

$$
\begin{array} { r l } & { \underbrace { \vphantom { \int } { \int } \underbrace { \begin{array} { l } { B } \\ { \sum _ { g = 0 } ^ { B } \frac { \mathcal { M } _ { L } \mathrm { ~ s c r e n ~ ( p a r . ) } } { g = 0 } } \underbrace { \beta B } _ { g = 0 } + \underbrace { ( 1 - \beta ) B } _ { g = 1 } } \\ { \underbrace { \beta B } _ { g = 0 } \frac { \mathcal { M } _ { S } \mathrm { ~ s p e c u l a t e ~ ( p a r . ) } } { \mathrm { ~ \underbrace { ~ \alpha ~ \in ~ \beta ~ \mathbb { S } _ { \varepsilon } ~ } _ { a c c e p t } ~ } } + \underbrace { ( 1 - \alpha ) \beta B } _ { \mathrm { ~ r e j e c t } } } \end{array} } _ { \underbrace { \left( 1 - \beta \right) B + ( 1 - \alpha ) \beta B } _ { \mathcal { R } } } } \\ & { \underbrace { \left( 1 - \beta \right) B + ( 1 - \alpha ) \beta B } _ { \mathcal { R } } \xrightarrow { \mathcal { M } _ { L } \mathrm { ~ a g e n t i c ~ ( s e q . ) } } \underbrace { \left( 1 - \beta \alpha \right) B } _ { \mathrm { ~ f a l l b a c k } } . } \end{array}
$$

由于 $c _ { J } + c _ { S } \ll B \bar { L } _ { \mathrm { a g e n t } }$，在实际的批量大小下，批处理时间主要由智能体在剩余集 $| \mathcal { R } | = ( 1 - \beta \alpha ) B$ 上的回退操作主导，从而导致吞吐量加速，这一加速共同受筛选比率 $\beta$ 和门控接受率 $\alpha$ 的影响。

$$
\Theta _ { \mathrm { S p e c E y e s } } / \Theta _ { \mathrm { a g e n t } } \approx 1 / ( 1 - \beta \alpha ) ,
$$

# 4 实验

# 4.1 实验设置

基准测试与基线。我们在三个多模态基准上评估SpecEyes，涵盖细粒度感知、高分辨率理解和幻觉鲁棒性。V $^ *$ [52] 提供了两个多项选择子集：用于属性识别的直接属性（115个问题）和用于空间推理的相对位置（76个问题）。HR-Bench [50] 测试高分辨率感知，包含4K和8K子集（各800个问题）。POPE [26] 是一个是/否幻觉探测器，分为对抗性、流行和随机拆分（各3000个问题）。小型非智能体模型 $M _ { S }$ 是Qwen3-VL-2B [45]；大型智能体模型 $M _ { L }$ 则采用DeepEyes [67] 和Thyme [63]，每个查询限制为5步工具使用。实现细节。所有模型使用贪婪解码（温度0），并且所有报告的延迟均包括工具执行时间。对于认知门控（第3.3节），我们设置 $K { = } 6 4$ , $\scriptstyle \epsilon = 1 0 ^ { - 6 }$ ，并采用最小词元聚合；对于底部聚合变体，我们将底部比例设置为 $r { = } 0 . 2$ ，灵感来源于[13]。门控阈值通过在每个基准上运行一次 $M _ { S }$ 来选择，以收集经验置信度分布 ${ \sim } 5 \mathrm { - } 1 0 \mathrm { m i n }$ 的值，从中均匀采样多个操作点，以表征准确性-延迟的权衡。所有实验均在单个NVIDIA A100 40 GB GPU上运行。

# 4.2 主要结果

表1将SpecEyes与代理基线和SpecReason [37]在所有七个评估分割上进行比较，使用两个代理主干网络（DeepEyes [67]和Thyme [63]）搭配无工具的推测模型Qwen3-VL-2B [45]。对于每个SpecEyes变体，我们报告在保持基线水平准确率的最佳操作点阈值下的结果。在四种置信度聚合策略中，SpecEyes (min)始终提供最强的准确性与速度配置，验证了第3.3节中的最坏情况下的保护设计；我们将在下文中集中讨论这一变体。表1. 在${ \mathbf { V } } ^ { * }$、HR-Bench和POPE上的主要结果。Spd.表示每个基础模型的墙钟速度提升。粗体字表示组内最佳准确率，突出显示的行代表推荐变体。SpecEyes (min)在两个代理mllm主干网络中提供了速度和准确性的最佳折中。

<table><tr><td rowspan="3">Method</td><td colspan="4">V*</td><td colspan="4">HR-Bench</td><td colspan="6">POPE</td><td rowspan="2" colspan="2">Avg.</td></tr><tr><td colspan="2">Attr.</td><td colspan="2">Pos.</td><td colspan="2">4K</td><td colspan="2">8K</td><td colspan="2">Adv.</td><td colspan="2">Pop.</td><td colspan="2">Rand.</td></tr><tr><td>Acc.</td><td>Spd.</td><td>Acc.</td><td>Spd.</td><td>Acc.</td><td>Spd.</td><td>Acc.</td><td>Spd.</td><td>Acc.</td><td>Spd.</td><td>Acc.</td><td>Spd.</td><td>Acc.</td><td>Spd.</td><td>Acc.</td><td>Spd.</td></tr><tr><td>Qwen3-VL-2B (draft only)</td><td>77.39</td><td>5.44×</td><td>82.89</td><td>5.31×</td><td>71.38</td><td>3.20×</td><td>68.00</td><td>2.90×</td><td>82.56</td><td>4.20×</td><td>83.80</td><td>3.78×</td><td>86.47</td><td>4.07×</td><td>78.93</td><td>4.13×</td></tr><tr><td>Based on DeepEyes [67]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DeepEyes [67]</td><td>90.43</td><td>1.00×</td><td>82.89</td><td>1.00×</td><td>75.85</td><td>1.00×</td><td>71.43</td><td>1.00×</td><td>78.43</td><td>1.00×</td><td>81.90</td><td>1.00×</td><td>88.83</td><td>1.00×</td><td>81.39</td><td>1.00×</td></tr><tr><td>SpecReason [37]</td><td>80.19</td><td>0.61×</td><td>73.91</td><td>0.38×</td><td>80.43</td><td>0.44×</td><td>72.54</td><td>0.42×</td><td>49.10</td><td>0.38×</td><td>51.55</td><td>0.38×</td><td>60.20</td><td>0.37×</td><td>66.85</td><td>0.43×</td></tr><tr><td>SpecEyes (log)</td><td>83.48</td><td>2.06×</td><td>88.16</td><td>2.05×</td><td>73.71</td><td>1.35×</td><td>69.67</td><td>1.28×</td><td>83.97</td><td>1.89×</td><td>86.70</td><td>1.95×</td><td>90.50</td><td>2.05×</td><td>82.31</td><td>1.80×</td></tr><tr><td>SpecEyes (mean)</td><td>78.26</td><td>2.89×</td><td>84.21</td><td>3.35×</td><td>71.62</td><td>1.88×</td><td>67.38</td><td>1.77×</td><td>85.13</td><td>2.06×</td><td>87.00</td><td>2.10×</td><td>90.13</td><td>2.14×</td><td>80.53</td><td>2.31×</td></tr><tr><td>SpecEyes (bottom)</td><td>83.48</td><td>2.13×</td><td>84.21</td><td>2.12×</td><td>75.22</td><td>1.20×</td><td>71.18</td><td>1.04×</td><td>85.13</td><td>2.08×</td><td>87.00</td><td>2.08×</td><td>90.13</td><td>2.11×</td><td>82.34</td><td>1.82×</td></tr><tr><td> SpecEyes (min)</td><td>90.43</td><td>1.53×</td><td>89.47</td><td>1.90×</td><td></td><td>75.85 1.13×</td><td>71.80</td><td>1.08×</td><td>85.13</td><td>2.13×</td><td>87.00</td><td>2.15×</td><td>90.13</td><td>2.19×</td><td>84.26</td><td>1.73×</td></tr><tr><td>Based on Thyme [63]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Thyme [63]</td><td></td><td>86.96 1.00×</td><td>82.89 1.00×</td><td></td><td></td><td>77.72 1.00×</td><td></td><td>72.43 1.00×</td><td></td><td>81.32 1.00×</td><td></td><td>84.53 1.00×</td><td>90.17 1.00×</td><td></td><td>82.29</td><td>1.00×</td></tr><tr><td>SpecReason [37]</td><td>89.57</td><td>0.48×</td><td>75.00</td><td>0.53×</td><td>80.01</td><td>0.52×</td><td>81.02</td><td>0.51×</td><td>84.62</td><td>0.46×</td><td>85.97</td><td>0.43×</td><td>90.27</td><td>0.46×</td><td>83.78</td><td>0.48×</td></tr><tr><td>SpecEyes (log)</td><td>80.87</td><td>1.82×</td><td>82.89</td><td>1.45×</td><td>74.97</td><td>1.13×</td><td>70.84</td><td>1.06×</td><td>85.76</td><td>1.68×</td><td>87.80</td><td>1.67×</td><td>91.47</td><td>1.59×</td><td>82.09</td><td>1.49×</td></tr><tr><td> SpecEyes (mean)</td><td>77.39</td><td>2.34×</td><td>80.26</td><td>1.83×</td><td>72.62</td><td>1.27×</td><td>68.00</td><td>1.21×</td><td>85.89</td><td>1.78×</td><td>88.30</td><td>1.80×</td><td>91.27</td><td>1.65×</td><td>80.53</td><td>1.70×</td></tr><tr><td>SpecEyes (bottom)</td><td>78.26</td><td>2.18× 1.32× 82.89</td><td>80.26</td><td>1.84×</td><td>77.35</td><td>1.05×</td><td>72.31</td><td>0.99×</td><td>85.89</td><td>1.81×</td><td>88.30</td><td>1.81×</td><td>91.27</td><td>1.73×</td><td>81.95</td><td>1.63×</td></tr><tr><td> SpecEyes (min)</td><td>87.83</td><td></td><td></td><td>1.42×</td><td>78.47</td><td>1.01×</td><td>73.31</td><td>0.95×</td><td>85.87</td><td>1.77×</td><td>88.30</td><td>1.78×</td><td>91.27</td><td>1.70×</td><td>83.99</td><td>1.42×</td></tr></table>

使用 DeepEyes，SpecEyes（最小）实现了 $1 . 7 3 \times$ 的平均加速，并将平均准确率从 $8 1 . 3 9 \%$ 提升至 $8 4 . 2 6 \%$。在 V\* Bench [52]上，它在直接属性上与基线相匹配（90.43%, $1 . 5 3 \times$），并将相对位置从 $8 2 . 8 9 \%$ 提升至 89.47%，加速比为 $1 . 9 0 \times$。POPE 的收益最大（$2.132.19 \times$），且准确率始终高于基线（例如，针对对抗样本：$7 8 . 4 3 \% \to 8 5 . 1 3 \%$），表明跳过不必要的工具轨迹也可以减少幻觉错误。HR-Bench 产生适度的加速（$1.081.13 \times$），因为查询更频繁地需要细粒度的工具辅助检查。用 Thyme 替换主干网络验证了泛化能力：SpecEyes（最小）实现了 $1 . 4 2 \times$ 的平均加速，同时将准确率从 $8 2 . 2 9 \%$ 提升至 $8 3 . 9 9 \%$。每个基准的模式显示出相似性：POPE 的收益最大（$1.701.78 \times$），V $^ *$ 获得了稳固的提升（$1.321.42 \times$），而 HR-Bench 仍然是瓶颈（$0.951.01 \times$）。HR-Bench 8K 上亚 $1 \times$ 的边际加速源于高分辨率输入抑制了 $\beta$ 和 $\alpha$，保持 $\beta \alpha$ 低。在这种情况下，运行 $M _ { S }$ 的固定成本略高于任何节省，符合 Eq. (9)。相比之下，SpecReason [37] 始终减慢推理（使用 DeepEyes 时为 $0.370.61 \times$；使用 Thyme 时为 $0.430.53 \times$），因为小模型缺乏结构化的工具调用能力，并产生大量的词元和轮次开销（平均 414 词元和 3.48 轮）。它在 POPE 上也急剧下降（低至 $4 9 . 1 0 \%$）。相比之下，SpecEyes 允许接受的查询完全跳过工具使用链，避免了这种开销。Qwen3-VL-2B（草稿）行确立了一个加速上限 $( 4 . 1 3 \times )$，代价是显著的准确率下降（78.93%）；SpecEyes 捕获了大部分延迟节省，同时保留了完整的推理质量。

# 4.3 信心校准分析

一个可靠的门控信号必须具有区分性：正确答案的置信度得分应当在统计上高于错误答案的得分。图3通过每个置信度得分在$M_{S}$的正确样本和错误样本上的核密度估计（KDE）来可视化这一属性，且每个子图都标注了$\Delta$（两个分布之间的峰值距离）作为区分性的一项直接度量。 $S_{\mathrm{l o g}}$（图3a）和$S_{\mathrm{s e p}}^{\mathrm{m e a n}}$（图3b）均产生了较小的$\Delta$，而$S_{\mathrm{s e p}}^{\mathrm{b o t t o m}}$（图3d）则提升了$\Delta$。

置信分布（KDE）置信分布（KDE）置信分布（KDE）置信分布（KDE）600 Coroct 10 40 Corc 125 Corect C 100 80.004 Δ=0.001 Δ=0.030 Δ=0.010 400- 50 10 10 200 25- 0- 0+ 0f 0- 0.93 0.94 0.95 0.96 0.97 0.98 0.99 1.00 0.9955 0.9960 0.9965 0.9970 0.9975 0.9980 0.94 0.95 0.96 0.97 0.98 0.99 0.970 0.975 0.980 0.985 0.990 0.995 置信得分 置信得分 置信得分 置信得分 (aSlog Smepan (c) Sme Sbotom

![](images/3.jpg)  
Fig. 3. KDE of confidence scores for correct vs. incorrect samples on ${ \mathbf { V } } ^ { * }$ (Qwen3-VL-2B). $\Delta$ measures gating discriminabil a peakcea $( \mathbf { a } , \mathbf { b } , \mathbf { d } )$ or (c) $S _ { \mathrm { s e p } } ^ { \mathrm { m i n } }$ b $\Delta$   
Fig. 4. Ablation on the gating threshold of SpecEyes. Lowering the threshold increases speedup at cost of accuracy. Dashed horizontal lines indicate baseline accuracy.

$S _ { \mathrm { s e p } } ^ { \mathrm { m i n } }$（图 3c）实现了最大值：错误样本聚集到低分峰值，而正确样本形成一个尖锐的高分模式，这与命题 1 一致。表 1 显示出单一的阈值在保持准确性的同时最大化接受率，这解释了为什么 SpecEyes（min）提供了更优的准确性与速度提升的权衡。

# 4.4 消融研究

我们进行消融实验，以研究SpecEyes中三个关键超参数的影响：门控阈值、服务批大小和可分离性计算参数 $K$。

阈值的消融实验。图4可视化了随着门控阈值变化的准确性与加速比的权衡，使用 $S_{\mathrm{s e p}}^{\mathrm{m i n}}$ 增加接受比率，从而提升加速比，同时准确性优雅地下降。在 V$^*$ 和 POPE 上，准确性在较宽的阈值范围内（0.94 到 0.99）始终高于或接近代理基准，确认有很大一部分查询可以安全跳过。HR-Bench 对阈值更为敏感：加速收益适中，当阈值低于0.97时，准确性开始下降，反映出真正需要工具辅助检查的查询比例较高。在所有设置中，存在一个广泛的操作区域，在这个区域内，SpecEyes 同时在准确性和速度上优于基准，验证了阈值不是一个脆弱的超参数，而是一个平滑的控制旋钮，用于导航准确性与效率的帕累托前沿。 批量大小的消融实验。图5研究了在固定门控阈值的操作点下，服务批量大小的影响。我们观察到，增加批量大小始终能提升端到端的加速比，而准确性保持不变（批处理仅影响系统执行，而不影响模型决策）。这一趋势符合我们的异构漏斗设计（第3.4节）：推测阶段是无状态的，因此高度可批处理，其每个查询的开销随批量大小的增加而有效摊销。相比之下，代理回退阶段受每个查询工具使用依赖的主导，基本上是顺序的，这导致在更大的批量大小下加速收益逐渐递减。在各个基准测试中，跳过率较高的数据集（例如 V* 和 POPE）从批处理中获益更多，而 HR-Bench 则因工具所需查询的比例更大而较早饱和。

![](images/4.jpg)

![](images/5.jpg)  
Fi.Ablation ervbat zLarerbatheortiz he ate ecativ ageroi wih dminihimargialgainsas the stateulgenallbacbecmeshe bottleneckCurv repor etoen speedup over the serial agentic baseline $( 1 . 0 \times )$ .   
Fig. 6. Ablation on Top- $K$ in separability-based gating. Larger $K$ consistently increases speedup but may reduce accuracy, suggesting that $K$ acts as a knob that tunes speculative aggressiveness.

关于可分离性计算中的 Top-$K$ 消融实验。如图 6 所示，$K$ 作为一个控制参数：增加 $K$ 单调地提高了加速比，但降低了准确性，这反映了降低门控阈值的效果，因为更大的 $K$ 包括了对比信号较弱的词元，从而膨胀了置信度估计。我们将 $K$ 设置为 64 作为平衡的默认值，这与直接属性上的基线准确率相匹配（90.43%，$1.50 \times$），并在相对位置上实现了强大的加速（$1.94 \times$，89.47%），而过大的 $K$ 则过度优化原始执行速度，直接牺牲总体推理准确性。

# 5 结论与未来工作

在本文中，我们提出了SpecEye一个代理性选择计算框架，推动了从单个词元到整个代理性流程的生态范式。该模型轻量级且无工具，推测性地回答不需要多步骤工具使用的查询，受益于基于答案可分离性的认知门控机制，并通过异构并行漏斗提供服务，将每次查询的延迟节省转化为系统级的吞吐量增益。在三个不同的图像理解基准测试中，SpecEye将端到端延迟减少了多达$3.35\times$，同时在准确性上与代理基线相当，并在并发服务下提供了稳定的吞吐量提升。未来工作。然而，我们的推测模型目前仅在代理深度$D = 0$（完全无工具）下运行，限制了在大多数查询真正需要工具协助的基准（例如HR-Bench）上的加速。未来工作的一个自然扩展是多深度推测$(L = 1, 2, \ldots, n)$，允许推测模型在门控之前进行有限次数的轻量级工具调用。这一策略在足够早的深度拦截查询，进一步减少不必要的回退到重型主干网络。

# References

[1] Jean-Baptiste Alayrac, Jef Donahue, Pauline Luc, Antone Miech, Iain Bar, Yana Hasson, Karel Lenc, Arthu Mensch, Katherine Millican, Malcolm Reynolds,  al. Flamingo:a visual language model or few-shot learning. Advances in neural information processing systems, 35:2371623736, 2022.   
[2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.   
[3] Danel Bolya, Cheng-ang Fu, Xiaoliang Dai, Peizo Zhang, Christop Feichtenhoer, and Judy Hofan.Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.   
[4] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple lminference acceleration framework with multiple decoding heads.arXiv preprint arXiv:2401.10774, 2024.   
[5] ChareChen, Sebastian Borgeaud, Georey Irving, Jean-Baptiste Lespau, Laurent Sfre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.   
[YxiChen, Xuchen an, Yalang  Bol in, and Ji Zhoue-:Lrealrai nd early-exit large language models with 3d parallelism. arXiv preprint arXiv:2312.04916, 2023.   
[7] Yong Xien Chng, Tao Hu, Wenwen Tong, Xueheng Li, Jiandong Chen, Haojia Yu, Jiefan Lu, Hewei Guo, Hanming Deng, Chengjun Xie, Gao Huang, Dahua Lin, and Lewei Lu. Sensenova-mars: Empowering multimodal agentic reasoning and search via reinforcement learning. arXiv preprint arXiv:2512.24330, 2025.   
[8] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale  Fung, a Steven Ho Instructbi:Towar eneal-purpo isin-anguage mode wit instructio tunin.Av in neural information processing systems, 36:4925049267, 2023.   
[9] Rohan Doshi. Introducing Agentic Visionin Gemini 3 Flash.https://blog.google/inovation-and-ai/tech nology/developers-tools/agentic-vision-gemini-3-flash/, January 2026. Accessed: 2026-02-24.   
[10] Mark Endo, Xiaohan Wang, and Serena Yeung-Levy. Feather the throttle: Revisiting visual token prung for vision-language model acceleration. ICCV, 2025.   
[11] Siqi Fan, Xin Jiang, Xiang Li, Xuying Meng, Peng Han, Shuo Shang, Aixin Sun, Yequan Wang, and Zhonguan Wang. Not all layers of llms are necessary during inference. arXiv preprint arXiv:2403.02181, 2024.   
[12] Tanyu Fu, Tengxuan Lu, Qingho Han, Gohao Dai, Shengen Yan, HuazhogYang, Xuefei Ning, and Yu Wang. Framefusion:Combining similarity and importance forvideo token reduction on large vision language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22654-22663, 2025.   
[13] Yichao Fu, Xuewei Wang, Yuandong Tian, and Jiawei Zhao. Deep think with confidence. arXiv preprint arXiv:2508.15260, 2025.   
[14] Zirun Guo, Minjie Hong, Feng Zhang, Kai Jia, and Tao Jin. Thinking with programming vision: Towards a unified view for thinking with images. arXiv preprint arXiv:2512.03746, 2025.   
[5] Yefei He, Feng Chen, Jing Lu, Wenq Shao, Hong Zhou, Kaipg Zhang, and Bohan Zung. Zipv: Efict are vision-language models with dynamic token sparsification, 2024. URL https://arxiv.org/abs/2410.08584.   
[16] Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, and Xing Yu. Deepeyesv2: Toward agentic multimodal model. arXiv preprint arXiv:2511.05271, 2025.   
[17] Pengfei Hu, Meng Cao, Yingyao Wang, Yi Wang, Jiahua Dong, Jun Song, Yu Cheng, Bo Zheng, and Xiaodan Liang. Thinking with drafts: Speculative temporal reasoning for efficient long video understanding. ArXiv, abs/2512.00805, 2025. URL https://api.semanticscholar.org/CorpusID:283449335.   
[18] Chengong Huang, Tong Zheng, Langlin Huang, Jinyuan Li, Haolin Liu, and Jiaxin Huang. Relaylm: Eient reasoning via collaborative decoding. ArXiv, abs/2601.05167, 2026. URL https://api.semanticscholar.org/ CorpusID:284544142.   
[9] Jia Huang, Jinsheg an, Zhongi Wan, Hanja Lyu, and Jiebo Luo. Evolver:Chain--evolu proig to boost large multimodal models for hateul meme detection. In Proceedings of the 31st International Conference on Computational Linguistics, pages 73217330, 2025.   
[20] Minchul Kim, Shangqian Gao, Yen-Chang Hsu, Yilin Shen, and Hongxia Jin. Token fusion: Bridging the gap between token pruning and token merging. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 13831392, 2024.   
[21] Avinash Kumar, Shashank Nag, Jason Clemons, Lizy John, and Poulami Das. Helios: Adaptive model and early-exit selection for efficient llm inference serving. arXiv preprint arXiv:2504.10724, 2025.   
[2] Xin Lai, Junyi Li, Wi Li, Tao Lu, Tan  and Hengu ZhaoMin3:Scal u rea pattes and interaction turns for visual search. ArXiv, abs/2509.07969, 2025. URL https://api.semanticscholar.org/ CorpusID:281217747.   
[3] Yaniv Leviathan, Matan Kalman, and Yossi Matis. Fast inerence from transormers vi speculativ decng. In International Conference on Machine Learning, pages 1927419286. PMLR, 2023.   
[] Junan Li, Dongxu Li, Silvio Savare, and Steven HoiBlip-2:Botstrappin lnguage-image pre-trainig with frozen image encoders and large language models. In Interational conferenceon machine learning, pages 1973019742. PMLR, 2023.   
[25] Xu Li, Yuxuan Liang, Xiaolei Chen, Yi Zheng, Haotian Chen, Bin Li, and Xiangyang Xue. Hero: Rethinking visual token early dropping inhigh-resolution large vision-language models, 2025.URL https://arxiv.org/abs/ 2509.13067.   
[26] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen.Evaluating object hallucination in large vision-language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292305, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.20. URL https: //aclanthology.org/2023.emnlp-main.20/.   
[27] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077, 2024.   
[28] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-2: Fster iference of language models with dynamic drat tres. In Procdings of the 2024 conference on epirical methods in natural language proceng, pages 74217432, 2024.   
[9]Yuhui Li Fangyun Wei Chao Zhang, and Hongyag Zhang. Eagle-3: Scalng up inerece aeleration  arge language models via training-time test. arXiv preprint arXiv:2503.01840, 2025.   
[0Bin Lin Zhe Tag, Yane, Ji Hu, JZa Y ng, P n, Mung JiboLoa Li Yuan. Moe-ava: Mixture of experts or large vision-language modes. IEEE Transactions on Multimedia, 2026.   
[31] Luxi Lin, Zhihang Lin, Zhanpeng Zeng, and Rongrong Ji. Speculative decoding reimagined for multimodal large language models. arXiv preprint arXiv:2505.14260, 2025.   
[32] Ziyang Lin, Zixuan Sun, Sanhorn Chen, Xiaoyang Chen, and Roy Zhao. Accelerating multi-modal  gaming performance via input prediction and mishit correction. arXiv preprint arXiv:2512.17250, 2025.   
[33] Zuyan Liu, Benlin Liu, Jiahui Wang, Yuhao Dong, Guangyi Chen, Yongmig Rao, Ranjy Krishna, and Jiwen Lu. Efnterencvisnnstructionfollowimode wit elasti cache.In urope CnferencCu Vision, pages 5469. Springer, 2024.   
[34] Yongong Luo, Xiawu Zheng, Guilin Li, Shukang Yin, Haojia Lin,Chayou Fu, JinfaHuan, Jayi Ji, Fi Cha, Jiebo Luo, et al. Video-rag: Visually-aligned retrieval-augmented long video comprehension. arXiv preprint arXiv:2411.13093, 2024.   
[35] Yongdong Luo, Wang Chen, Weizhong Huang, Shukang Yin, Haojia Lin, Jinfa Huang, Chaoyou Fu, Jiayi Ji, Xiawu Zheng, and Jiebo Luo. Quota: Query-oriented token assignment via cot query decouple for long video comprehension. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 2416024168, 2026.   
[36] OpenAI. Introducing OpenAI o3 and 04-mini. https://openai.com/index/introducing-03-and-04-mini/, April 2025. Accessed: 2026-02-24.   
[3] Rui Pan, Yinwei Dai, Zhihao Zhang, Gabriele Oliaro, Zhihao Jia, and Ravi Netravali. Specreason: Fast and accurate inference-time compute via speculative reasoning. arXiv preprint arXiv:2504.07891, 2025.   
[38] Yi Peng, Peiyu Wang, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weije Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, et al. Skywork r1v: Pioneering multimodal reasoning with chain-o-thought. arXiv preprint arXiv:2504.05599, 2025.   
[39] TimoSchick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, MariaLomeli, Eric Hambro, Luke Zetteoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools.Advances in neural information processing systems, 36:6853968551, 2023.   
[40] Hui Shen, Xin Wang, Ping Zhang, Yunta Hsieh, Qi Han, Zhongwei Wan, Ziheng Zhang, Jingxuan Zhang, Jing Xiong, Ziyuan Liu, et al. Mmspec: Benchmarking speculative decoding for vision-language models.arXiv preprint arXiv:2603.14989, 2026.   
[41] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Huggingpt: Solving ai tasks with chatgpt and its friends in hugging face.Advances in Neural Information Processing Systems, 36: 3815438180, 2023.   
[42] Qi Song, Honglin Li, Yingchen Yu, Haoi Zhou, Lin Yang, Song Bai, Qi She, Zilong Huang, and Yunqing Zhao. Codedance: A dynamic tool-integrated mlm for executable visual reasoning. ArXiv, abs/2512.17312, 2025. URL https://api.semanticscholar.org/CorpusID:284058227.   
[3] GmTeam,Rohi SasBr, Jan-Baptislayr, Jiau, Rad Sri, JohanScha, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models.arXiv preprint arXiv:2312.11805, 2023.   
[44] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.   
[45] Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.   
[46] Surat Terapittaynon, Bradley McDanel, and Hsiang-Tsung Kung. Brancynet: Fst inferenceviaeary exiig from deep neural networks. In 2016 23rd international conference on patter recognition (ICPR), pages 24642469. IEEE, 2016.   
[7] Zoi Wan, Ziang Wu, Che iu, JinaHu, Zhig Zhu, Peng Jin, Lon Wang, and iYuan. Look-: Look-onctiizationn k cahe or ef ultmodal long-contextirenc. In Finif thesti for Computational Linguistics: EMNLP 2024, pages 40654078, 2024.   
[48] Zhongwei Wan, Hui Shen, Xin Wang, Che Liu, Zheda Mai, and Mi Zhang. Meda: Dynamic kv cache allocation for efficient multimodal long-context inference. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 24852497, 2025.   
[49] Huanyu Wang, Jushi Kai, Haoli Bai, Lu Hou, Bo Jiang, Ziwei He, and Zhouhan Lin. Fourier-vlm: Compressing vision tokens in the frequency domain for large vision-language models, 2025. URL https://arxiv.org/abs/25 08.06038.   
[50] Wenin Wang, Liang Ding, Minyan Zeng, Xiabin Zhou, Li Shen, Yong Luo, Wei Yu, and Dacheng Tao. Divide, co ndcinrainemework orhig-reolutimage erptionultimodal largee models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 79077915, 2025.   
[51] Yancheng Wang and Yingzhen Yng.Effent visalranormer by learable token merging. IEEE ranins on pattern analysis and machine intelligence, 2025.   
[52] Penghao Wu and Saining Xie. V $^ *$ : Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.   
[53] Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, pages 39093925, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.257. URL https://aclanthology.org/2023.findings-emnlp.257/.   
[54] Hemg Xia, Yongqi Li, Jun Zhanguno Du, and Wenj Li.wi:On-they sel-speulativedecodr llm inference acceleration. arXiv preprint arXiv:2410.06916, 2024.   
[55] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi W uyrg ur-ger reduction. arXiv preprint arXiv:2410.17247, 2024.   
[56 J Xu,Jiay Pan YonZho SmChen, Jii, Y Lian, Ju Wu, nd uDai.pee: Accelerating large language model inference with speculative early exiting. In Procdings of the 52nd Annual International Symposium on Computer Architecture, pages 467481, 2025.   
[57] Penghui Yang, Cunxiao Du, Fengzhuo Zhang, Haonan Wang, Tianyu Pang, Chao Du, and Bo An. Longspec: Long-context speculativedecoding with effcient drafting and vrication. arXiv e-prints, pages arXiv2502, 2025.   
[8] Sqang Yuk hen, Zhuo Tan, Che Wang, Jngy , Bei u, and Jay Ji.is:Lon is better but not necessary in vision language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1979219802, 2025.   
[59] Wehao Yang, Yu Xia, Jinlong Huang, Shiyin Lu, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang,Yuau Wan, and Lijun Zhang. Deep but reliable: Advancing multi-turn reasoning for thinking with images, 2026. URL https://arxiv.org/abs/2512.17306.   
[60] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.   
[1] ZhaangYu, Jayi Zhang, Huixu u, Yufan Zhao, Yfan Wu, Mingi Deng, Jinyu Xiang, Yizhang Lin, Lingxiao Tang, Yuyu Luo, et al. Recode: Unify plan and action for universal granularity control. arXiv preprint arXiv:2510.23564, 2025.   
[62] Ju Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehror. Dratveriy: Lossless large language model acceleration via sel-speculative decoding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1126311282, 2024.   
[63] Yi-Fan Zhang, Xingyu Lu, Shukang Yin, Chaoyou Fu, Wei Chen, Xiao Hu, Bin Wen, Kaiyu Jiang, Changyi Liu, Tianke Zhang, et al. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025.   
[64] Yifan Zhang, Liang Hu, Haofeng Sun, Peiyu Wang, Yichen Wei, Shukang Yin, Jiango Pei, Wei Shen, Peng Xia, Yi Peng, e al. Skywork-r1v4:Toward agenticmultimodalintelligence throug interleave thinking with ages and deepresearch. arXiv preprint arXiv:2512.02395, 2025.   
[] Shitian Zhao, Shaoheng Lin, Ming Li, Haoquan Zhang, Wenshuo Peng, Kaipeg Zhang, and ChenWei.Pyvison-rl: Forging open agentic vision models via rl. arXiv preprint arXiv:2602.20739, 2026.   
[66] Wangbo Zhao, Yizeng Han, Jiasheng Tang, Zhikai Li, Yibing Song, Kai Wang, Zhangyang Wang, and Yang You. A stitch in time saves nine Smal vlm is a precise guidance oraccelerating large vlms. In Procdings  the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19814-19824, 2025.   
[67] Ziwei Zheng, Michal Yang, Jack Hong, Chenxio Zhao, Guohai Xu, Le Yang, ChaoShen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.   
[68] Yunqi Zhu, Xuebing Yang, Yuanyuan Wu, and Wensheng Zhang. Hierarchical skip decoding for effcient autoregressive text generation. arXiv preprint arXiv:2403.14919, 2024.