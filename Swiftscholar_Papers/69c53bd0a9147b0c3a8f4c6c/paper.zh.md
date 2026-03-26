# 对齐税：对齐大语言模型中的响应同质化及其对不确定性估计的影响

刘明义 独立研究员 GitHub: @DigitLion 2026年3月25日

# 摘要

RLHF对齐的语言模型表现出响应同质化现象：在TruthfulQA上（$_{n}$ = 790），4079%的问题在10个独立同分布样本中产生单一语义簇（在聚类方法、样本大小$N = 3 \ - \ 10$、温度${T = 0.3 \ - \ 1.5}$和生成长度40200词元下均稳健）；对200个问题的响应中，33.5%的SCR在200个词元时保持，而基础模型则为$0\%$。在受影响的问题上，基于采样的不确定性方法没有识别能力（AUROC=0.500），而自由词元熵仍然保持信号（0.603）。这种对齐代价是任务依赖的：在GSM8K上（$n = 500$），词元熵达到0.724（Cohen's $d$ = 0.81）。

基于Qwen3-14B的基础模型与指令模型的消融实验确认了对齐的因果作用：基础模型的单一聚类率为$1.0\%$，而指令模型为$28.5\%$（Wilcoxon $p < 10^{-6}$）。三阶段训练消融（基础 $0.0\% \ \mathrm{SFT}$，$1.5\%$ DPO，$4.0\%$ SCR，均 $p < 0.0003$）将原因定位为DPO，而非SFT。在四个模型系列上的跨系列复制（Qwen3-14B: $28.5\%$，LLaMA-3.2-3B: $5.5\%$，Mistral-7B: $1.0\%$ SCR）显示对齐损耗的严重程度因系列和规模而异。在Tulu-3上进行的跨链复制（Llama-3.1-8B SFT DPO $^{+}$ RLVR）显示最小的对齐损耗（$0.5\%$ SCR与Zephyr的$4.0\%$相比），确认了严重程度依赖于配方。我们在二十二个实验、五个基准、四个模型系列和三个模型规模（3B14B）中验证了这一发现，使用了代理（Jaccard）、SINdex风格的嵌入（聚类余弦聚类）和三个DeBERTa规模的标准NLI基线（large 435M, base 184M, xsmall 70M——均≈0.51 AUROC）。通过两个独立嵌入系列的跨嵌入验证（Qwen3-Embedding $78\%$ SCR与Nomic-embed-text $92\%$ SCR，$\tau = 0.85$）排除了耦合偏差。在WebQuestions上的跨数据集验证（$58.0\%$ SCR，$\tau=0.85$）确认了对齐损耗超越了TruthfulQA。LLM-judge标签已根据TruthfulQA金标准答案模板验证，$\kappa = 0.487$。中心发现——响应均质化——是与实现无关且无标签的。基于此诊断，我们探索了一种基于最便宜优先的级联（UCBD），结合正交的不确定性信号。选择性预测将GSM8K的准确率从$84.4\%$提升至$93.2\%$，覆盖率为$50\%$；弱相关边界（$| r | \le 0.12$）实现了57%的成本节省。

# 1 引言

基于大语言模型的人工智能智能体能力显著，但一个问题却鲜有系统性的关注：智能体能否识别出自己不知道的事情？考虑数学推理：在GSM8K数据集上（$n { = } 500$），一个140亿参数的模型的词元熵在错误检测方面的AUROC值为0.724（Cohen's $d { = } 0.81$），实现了选择性预测，将准确率从$84.4\%$提升至$93.2\%$，覆盖率为$50\%$。然而，在事实问答（TruthfulQA，整体）中，相同的熵信号几乎未能超出随机猜测的水平（0.52）。这个$10\times$的效应大小差距（Cohen's $d$：0.81 vs. 0.07）揭示出不确定性并非单一的量，它具有结构性，要求采取多边界的方法。

我们观察到一个令人惊讶的经验规律：在RLHF对齐的模型上，响应多样性在采样时崩溃。在TruthfulQA中，$40.0\%$的问题在10个独立同分布样本${\it T}^{\mathrm{=1.0}}$中产生一个单一的语义簇——模型重复生成相同的答案（正确或错误）。这种对齐税使得基于采样的方法在受影响的问题上结构性地不可靠（AUROC=0.500）。自由词元熵保持了信号（0.603），因为它测量的是每个词元的计算不确定性，而RLHF无法完全抑制这一点。基于基础与指令的消融实验（实验13）隔离了因果机制：Qwen3-14B-Base的单一簇率为$1.0\%$，而对齐版本为$28.5\%$（$\mathit{p} < 10^{-6}$）——对齐使多样性降低了$2.6\times$。一个训练阶段的消融实验（实验16）进一步定位了原因：SFT保持基础层级多样性为$1.5\%$ SCR，而DPO则导致崩溃（$4.0\%$）。在四个家族上的跨家族复制（Qwen3: $28.5\%, LLaMA-3: 5.5\%, Zephyr-DPO: 4.0\%, Tulu-3-DPO: 0.5\%, Mistral-7B: 1.0\%$ SCR）确认了广泛性，同时揭示了家族和配方依赖的严重性。与三个DeBERTa模型（200q）的NLI基础的SE比较获得AUROC=0.511（大，435M），0.512（基础，184M）和0.501（小型，70M）——均接近偶然性——而$6.2\times$的NLI模型扩展则没有改善。响应同质化具有聚类鲁棒性：单一簇崩溃在所有测试的方法下均发生（Jaccard、嵌入余弦、基于NLI），尽管确切的比率因粒度而异（$40\%$ Jaccard与79\%嵌入；见实验12）。一个解码策略的消融实验（实验15）确认在核采样和降低温度下，这种税收依然存在——这是学习分布的特性，而不是采样过程。这促使我们转而增加正交信号类型，而不是采样更多的响应。

# 贡献。

1. 对齐税：对齐模型抑制响应多样性，其中3862%的TruthfulQA问题（$n$=790）归结为单一语义聚类——与聚类方法无关（Jacard：40%，嵌入：$79\%$），样本量（$N$=3：$46\%$，$N$=10：$40\%$），温度（$T$=0.3：62%，$T$=1.5：$38\%$），以及解码策略（核采样$p$=0.9：$33.5\%$，$\scriptstyle{p=0.95}$：$30.0\%$）。这种同质化将基于采样的熵降低至偶然（AUROC=0.500），而自由词元熵保留了信号（0.603）。2. 任务依赖的不确定性结构：B1 AUROC从0.52（事实问答）变化到0.72（数学），Cohen的$d$从0.07变化到0.81——显示出不确定性检测必须是多模态的，而不是单一的。3. 级联架构：受诊断发现的启发，我们设计了基于正交边界类型的最便宜优先级联。弱边界间依赖（$|r| \leq 0.12$，$\mathrm{MI} \leq 0.02$比特）使得节省成本达到57%，而选择性预测使GSM8K的准确率从$84.4\%$提升至$93.2\%$，覆盖率为$50\%$。

范围。本论文在不同证据标准下做出两项贡献。诊断性（主要）：响应同质化现象在五个数据集上得到验证，这些数据集涵盖三种任务类型（事实问答：TruthfulQA 790问、FreshQA 100问、WebQuestions 200问；多跳问答：HotpotQA 100问；数学推理：GSM8K 500问），聚类方法、样本大小、温度、解码策略、NLI模型规模（70M-435M）、在两个独立链条上的训练阶段（Mistral/Zephyr和Llama/Tulu-3：SFT保持多样性；DPO导致崩溃），生成长度（40200词元）以及嵌入家族（交叉嵌入验证确认结果不是同类嵌入器偏差的伪影）。对额外验证权重较重的基准（FEVER，SciFact，MMLU子集）的扩展是一个自然的下一步。架构性（探索性）：级联设计受诊断发现的启发，并在选择性预测上得到验证（GSM8K：84.4%→93.2%）；与多信号融合框架的正面比较仍是未来的工作。

# 2 相关工作与定位

单信号不确定性检测器。基于标记的方法——熵、LogTokU [Ma et al., 2025b]、PRO [Chen et al., 2025c]、语义能量 [Zhang et al., 2025c]——以及基于采样的方法（SE [Kuhn et al., 2023]、SelfCheckGPT [Manakul et al., 2023]、CoCoA [Huang et al., 2026]）在各个基准上实现了AUROC 0.720-0.89。SINdex [Abdaljalil et al., 2025]通过基于嵌入的一致性度量改善了聚类（相较于SE提高了$ + 9.3 \%$ AUROC）。我们复制了SINdex的核心方法——基于嵌入的余弦相似度与层次聚类，并发现其揭示了比Jaccard更高的同质化（790q上SCR分别为79%与40%；实验12），确认这一对齐代价并不是表层聚类的伪影。语义能量 [Ma et al., 2025a] 与此最直接相关：它使用基于lgit的玻尔兹曼能量在聚类层面进行汇总，专门针对我们诊断的单聚类失效模式——在单聚类案例中，相较于SE获得了$1 3 \%$ 的AUROC提升。我们的贡献是互补的和上游的：我们提供了单聚类崩溃系统性发生的诊断解释（基于对齐的同质化），而语义能量提供了一种绕过崩溃多样性的补救信号。具体而言，语义能量在单聚类状态的增益是我们分析的预测：当$| { \mathcal { C } } | = 1$时，基于采样的SE在结构上是零样本的（包括捕获每个标记变化的能量而不进行性能）。我们的B1标记熵（在79%嵌入单聚类子集中的AUROC=0.593）通过相同的机制实现了类似的增益——测量RLHF无法完全抑制的计算不确定性。关键区别在于，语义能量仍需要$N$个样本进行聚类级能量汇总，而B1仅需一次前向传播。SRE-UQ [Vipulanandan et al., 2026]利用量子张量网络扰动进行时间序列概率不确定性检测。所有单信号方法都在一个范式中运行；我们的实验1显示任何单一范式都有结构性盲区：B1在12/24 TruthfulQA类别中有效，但在其余12个类别中表现不佳。元认知与智能体路由。MetaRAG [Zhou et al., 2024]在不确定性上触发检索；ReMA [Wan et al., 2025]应用强化学习进行路由。UCBD通过最便宜优先级级联路由到不确定性检测器。

对齐、校准与集成。神经网络经常出现不准确的校准 [Guo et al., 2017]；基于人类反馈的强化学习（RLHF）进一步影响校准 [Kadavath et al., 2022；Leng et al., 2025]。一致性预测 [Angelopoulos and Bates, 2021] 提供了覆盖保证；深度集成 [Lakshminarayanan et al., 2017] 结合多种模型；我们的级联模型则结合来自单一模型的正交信号类型。RLHF 的模式崩溃效应已经得到充分记录，Kirk et al. [2024] 表明输出多样性降低；Saeidi et al. [2024] 发现在“安全”响应上，概率质量集中；Azar et al. [2024] 将KL正则化的RLHF与分布收窄联系起来。近期的DPO变种（例如RoPO风格正则化）明确旨在在偏好优化过程中保持输出多样性，进一步验证DPO导致的崩溃是一个受到认可的问题。与混合专家崩溃的区别：先前的研究考察了多样性损失作为生成质量问题（更少的创造性输出，减少的风格变化）。我们的“对齐税”聚焦于一个独特的下游后果：当多样性崩溃为一个单一语义簇时，基于采样的不确定性估计结构上变得无信息（SE=0），不论单一响应是正确还是错误。这不仅仅是减少多样性——而是从“某些信号”到“零信号”的相变。我们的温度消融（$T$ = 0.31.5）显示即使是激进采样仍然有38%的结果被同质化。深度集成 [Lakshminarayanan et al., 2017] 和MC Dropout [Gal and Ghahramani, 2016] 在这里不适用：它们要求多个独立训练的模型或在推理时进行dropout，而这些在通过API访问的现成对齐大语言模型中都不可用。我们的级联模型则结合来自单一模型的正交信号类型。引起同质化的调节。黑箱调节审计 [Stanusch et al., 2025] 记录表明，安全过滤器和“主动调节”在商业大语言模型中产生确定性的拒绝和跨语言不一致性，有效地在接口层面上同质化输出。我们的对齐税延伸了这一观察，表明对齐引起的同质化并不只是外部因素的作用，并且具有一个特定的、可量化的下游后果——基于采样的不确定性估计的结构性失败。调节审计的视角与我们的研究互补：外部调节在我们测量的分布压缩之上增加了第二个同质化来源，这表明已部署系统面临来自训练时（DPO）和推理时（调节）干预的多重多样性损失。

多信号融合框架。UniCR [Li et al., 2025] 通过符合风险控制统一异构的不确定性证据，提供了 UCBD 所缺乏的正式覆盖保证。两个系统在不同层面上操作：UniCR 假设所有信号都是预先计算的，并优化融合/校准以达到目标覆盖；UCBD 解决上游问题，即哪些信号需要计算——通过一个按成本排序的级联路由查询，其中 57% 在免费的 B1 阶段退出，避免了对每个查询计算所有信号的成本。这些方法是可组合的：UCBD 的级联输出可以输入 UniCR 的符合校准层，将成本节约与正式保证结合起来。至关重要的是，我们的对齐税收发现适用于任何依赖于基于采样信号的框架：UniCR 的符合保证仅在采样信号成立时有效，当这些信号在结构上为零（单聚类环境）时，能够恢复区分能力的校准则无效。这一情况促使在调用依赖于采样的方法之前，路由到非采样信号（B1 代币熵，B2 密度）。表 1 将先前的方法映射到 UCBD 边界。

Table 1: Positioning of UCBD relative to existing approaches. All prior methods operate within a single boundary; UCBD provides the orchestration layer.   

<table><tr><td>Method</td><td>Boundary</td><td>Cost</td><td>Cascade Role</td></tr><tr><td>Token entropy (ours)</td><td>B1 Fluency</td><td>Free</td><td>Stage 1 (always-on)</td></tr><tr><td>LogTokU / PRO [Ma et al., 2025b, Chen et al., 2025c]</td><td>B1 Fluency</td><td>Free</td><td>Stage 1 (drop-in)</td></tr><tr><td>Semantic Energy [Ma et al., 2025a]</td><td>B1 Fluency</td><td>N samples</td><td>Single-cluster remedy</td></tr><tr><td>SINdex [Abdaljalil et al., 2025]</td><td>B1 Fluency</td><td>N samples</td><td>Stage 1 (escalation)</td></tr><tr><td>Semantic Entropy [Kuhn et al., 2023]</td><td>B1 Fluency</td><td>5-10×</td><td>Stage 1 (escalation)</td></tr><tr><td>CoCoA [Huang et al., 2026]</td><td>B1 Fluency</td><td>1-5×</td><td>Stage 1 (escalation)</td></tr><tr><td>SelfCheckGPT [Manakul et al., 2023]</td><td>B1 Fluency</td><td>5 calls</td><td>Stage 1 (escalation)</td></tr><tr><td>Embedding density [Vazhentsev et al., 2025]</td><td>B2 Density</td><td>1 embed</td><td>Stage 2</td></tr><tr><td>KG completion [Trouillon et al., 2016]</td><td>B4 Rupture</td><td>KG query</td><td>Stage 4</td></tr><tr><td>NLI verification (ours)</td><td>B5 Grounding</td><td>NLI call</td><td>Stage 5</td></tr><tr><td>ReMA [Wan et al., 2025]</td><td>Pointer Model</td><td>RL</td><td>Dispatcher</td></tr></table>

蒸馏与单遍预测器。SSD [Schuster 等，2026] 将多样本语义离散化蒸馏为单遍混合密度网络，从而在训练时摊销采样成本。分析确认了我们诊断的一个关键要素：教师离散化在教师模型始终生成相同答案的提示上分配零不确定性，即使该答案不正确。通过学习的连续平滑部分缓解了这一问题，在7个模型中有4个超越了教师离散化。我们的对齐惩罚发现解释了为何零离散化问题在对齐模型中具有系统性：DPO驱动的同质化在4079%的查询上生成了结构上无信息的教师信号。这表明，SSD风格的方法应当（1）使用基础/非对齐模型作为教师，或（2）将非采样信号（例如，词元熵）纳入蒸馏目标。

单次通过和内部信号的不确定性量化方法。几种最近的方法完全跳过采样。内部置信度 [Chen 等, 2025b] 在生成之前通过隐藏状态自评价来估计查询级别的不确定性，在事实问答和数学任务中取得了强劲的表现，而无须生成任何词元。TokUR [Zhang 等, 2025b] 通过低秩权重扰动将词元级的不确定性分解为偶然性和认识性成分，在 MATH500 上达到了 8083% 的 AUROC。EAS [Zhu, 2025] 在生成轨迹中整合词元级熵作为序列级得分。语义熵探测器 [Kossen 等, 2024] 在隐藏状态上学习线性探测器，以在单次通过中近似语义熵。我们的 B1 词元熵是该家族中最简单的成员：它不需要探测器、扰动或训练，只需访问对数概率。我们的诊断贡献与这些方法正交，并且位于其上游：我们解释了为什么单次通过信号在对齐模型上优于基于采样的方法（对齐压缩了样本间的多样性，同时保持了不确定性）。

保持多样性的对齐与缓解。最近的一些方法直接解决了对齐引起的多样性损失，每种方法针对不同的机制。H-DPO [Omura et al, 2024] 在 DPO 目标中添加了熵奖励，明确惩罚我们 SCR 诊断所测量的概率质量集中；我们的分阶段消融实验（Exp.16）确认 DPO 是造成崩溃的因素，因此 H-DPO 的熵正则化正好针对正确的训练阶段。SPL [Hwang et al., 2025] 在偏好优化中解耦 KL 正则化，分别控制选择和拒绝响应的策略发散——解决了推动模型向单一高奖励模式的非对称惩罚。DivP [Lanchanti et al., 2025] 针对稀有但高质量偏好对进行训练，促进超越模式的分布覆盖。可表达采样 [Zhang et al., 2025a] 采用推理时的方法，通过提示在不重新训练的情况下恢复 $6 6 . 8 \%$ 的基础模型多样性。带有显式 KL 正则化的标准 PPO 基于 RLHF [Ouyang et al., 2022] 也限制了分布转移，尽管我们的结果表明单独的 KL 是不够的——在 KL 正则化模型中对齐税仍然存在（Exp. 13）。我们的 SCR 诊断提供了一个有原则的评估标准：成功的保持多样性的方法应该将 $4 0 – 7 9 \%$ 的单簇率降低到基础模型的 ≤1.5%，同时保持指令遵循的质量。“隐形牵绳”分析 [Chen et al, 2025a] 独立观察到 RLVR 增加了词元级的熵，同时降低了答案级的熵——这正是我们的对齐税所预测的词元/语义解耦：计算多样性得以保留，而输出级多样性崩溃。我们的 Tulu-3 交叉链复制（Exp.18）提供了部分实证验证：Tulu-3 的 DPO $^ +$ RLVR 配方只产生 0.5% 的 SCR（与 Zephyr 的 $4 . 0 \%$ DPO-only 对比），这表明训练配方设计可以显著缓解税收。在 TruthfulQA 上对 H-DPO、SPL 和 DivPO 模型的实证 SCR 评估仍然是验证这些缓解方法是否减少对齐税的最直接下一步（未来工作 1）。大规模不确定性量化研究 [Yadkori et al., 2024] 显示指令调优可以提高可表达的置信度；我们的发现特定于基于采样的不确定性量化——我们并不声称对齐会损害所有不确定性模式。

选择性预测与级联。UCBD 涉及选择性预测 [Geifman 和 El-Yaniv, 2017] 和级联分类 [Viola 和 Jones, 2001]。提示多样性 [Sclar 等, 2025] 表明一致性=正确性，支持我们的诊断。HalluGuard [Wang 等, 2025] 实现高 AUROC uimodel-internal NTK 信号；CounterRefine [Lit 等, 2026] 提供检索导向修复作为一种实用的替代或 NLI。任何信号均可作为 drop-B1 替代；我们的贡献在于多边界调度。直接比较状态：我们在三个模型规模上实施基于 NLI 的 SE [Kuhn 等, 2023]（实验 12），提供对 200 个问题的直接头对头比较；Semantic Energy [Mat 等, 2025] 的代码尚未公开，但我们在下面建立正式等价；UniCR [Li 等, 2025] 在不同层级运作（事后融合与上游路由）且可与 UCBD 组合。单聚类等价。在单聚类模式（$| { \mathcal { C } } | = 1$，4079% 对齐查询）下，所有基于 logit 的信号——B1 词元熵、LogTokU [Ma 等, 2025b]、PRO [Chen 等, 2025c]、内部置信度 [Chen 等, 2025b]、TokUR [Zhang 等, 2025b]——均在相同的每词元 logit 分布上运作。LogTokU ≡ PRO（均为 $=$ 平均负对数概率）；B1 熵是相同向量的单调变换。所有信号提供相同的排序不确定性，因此在单聚类模式下具有相同的 AUROC。我们在单聚类子集上的 B1 AUROC 为 0.593，适用于任何基于 logit 的替代方法。在多聚类模式下，这些方法可能会出现差异；在匹配数据上的比较仍需进一步研究。

什么是真正的新颖之处。先前的工作记录了RLHF模式崩溃作为生成质量问题[Kirk et al, 2024, Saeidi et al, 2024]；Verbalized Sampling [Zhang et al, 2025a]观察到在DPO之后多样性从$20.8\%$下降到$10.8\%$，并提出了基于提示的补救措施；Hashimoto等人[2025]识别出一种“挤压效应”，即DPO将概率质量集中在最高标记上，降低了不确定性估计；Xiao等人[2024]理论证明基于KL的RLHF引起“偏好崩溃”，即使在有神谕奖励模型的情况下也是如此。单遍方法[Chen et al., 2025b, Zhang et al., 2025b, Kossen et al., 2024]在没有采样的情况下展现了强大的统一性，以及这些方法在对齐模型上失败的原因，以及这种情况发生的频率。我们注意到“对齐税”一词由Lin等人[2024]创造，用以表示NLP基准上的性能下降；我们特定地将其重新定义为$U {\cal Q}$能力降级——一种独特且互补的现象。具体而言：（a）响应同质化发生在4079%的问题上——这一比例足以在结构上同时破坏所有基于采样的UQ方法；（b）这一现象由DPO驱动（而非SFT），其严重程度随配方而异，跨家庭的变化达到$50\times$（0.5%至28.5% SCR），这通过在两个独立培训链上的因果消融建立——与DPO挤压效应[Hashimoto et al., 2025]和偏好崩溃理论[Xiao et al., 2024]一致；（c）它在每一次稳健性检查中持续存在（聚类方法、样本大小$N={3-10}$、温度$T=0.3$至$1.5$、解码策略、生成长度40200个标记、NLI模型规模70M至435M，以及通过两个独立嵌入家庭进行的交叉嵌入验证）；（d）它产生了依赖任务的差距（Cohen's $d$: 0.07的事实QA与0.81的数学），而没有单一信号能够弥补。这是对齐引起的UQ衰减在这一规模下的首次系统测量，具有因果隔离和跨家庭复制特性。

# 3 个认知边界

认知边界是智能体知识与查询之间的差距。设 $q$ 表示查询，$\kappa$ 表示知识库。

定义（对齐税）。设 $\mathcal { D } _ { S } ( q ) = | \{ C _ { 1 } , \ldots , C _ { m } \} |$ 为来自 $N$ 个独立同分布样本 $T = 1.0$ 的查询 $q$ 的不同语义簇的数量。我们定义 $ \mathrm { A T } ( q ) = 1 - \frac { \mathcal { D } _ { S } ( q ) } { N } $。当 $\operatorname { A T } ( q ) = 1 - 1 / N$（单一簇）时，基于采样的方法没有区分能力：无论正确性如何，$\mathrm { S E } ( q ) = 0$。对齐税是连续的：中间值（例如，从 10 个样本中获得 2 个簇，$AT=0$）表示部分多样性减少，在这种情况下，SE 保留了一些但减弱的信号；我们的分析侧重于单一簇情况（$| { \mathcal { C } } | = 1$，SE≡0）作为完全失败模式。标签独立性：SCR 纯粹通过响应聚类计算——不需要正确性标签或参考答案——使得诊断适用于任何模型和任何数据集。每个词元的自由熵 $H ( q )$ 仍然具有信息性，因为它在每个解码步骤中测量计算不确定性——模型对其下一个词元分布的内部信心——而 RLHF 无法完全抑制而不降低生成质量。相对而言，基于采样的方法测量响应间的多样性，优化偏好可能通过奖励一致的输出加以抑制（尽管我们没有将 RLHF 与其他训练管道因素分离；见限制 1）。B1 流畅性（自由）：词元熵 $\begin{array} { r } { H _ { t } = - \sum _ { v } P ( v _ { t } | v _ { < t } ) \log P ( v _ { t } | v _ { < t } ) } \end{array}$。当 $H > \tau _ { H }$ 时触发。零成本——对数概率是生成的副产品。B2 密度（\$）：查询嵌入密度 $\begin{array} { r } { \rho ( { \bf e } _ { q } ) = \frac { 1 } { k } \sum \cos ( { \bf e } _ { q } , { \bf e } _ { n _ { j } } ) } \end{array}$ 低密度 = 知识荒漠。B3 新鲜度（\$ \$）：新鲜度 $\ ( k , t _ { q } ) = \exp ( - \lambda ( k ) \cdot ( t _ { q } - t _ { k } ) )$。操作化：在推理时，通过检测查询中的时间实体（日期、“当前”、“最近”）并与模型已知的训练截止进行比较来触发 B3。这是一个基于元数据的探测器，而不是学习到的信号——它标记可能涉及过时知识的查询。B4 关联中断（\$ \$）：知识图谱补全得分 $\hat { P } ( e _ { 1 } , r , e _ { 2 } | \mathcal { G } ) > \tau _ { r }$ 但 $( e _ { 1 } , r , e _ { 2 } ) \notin \mathcal { G }$ 的缺失链接应该存在。我们验证这一点的存在性对（第 5.9 节）。B5 基础（\$ \$）：外部交叉验证。表现出“过度自信倒转”，在缺乏相关知识时，反直觉地降低了表达的不确定性。我们使用 NLI 蕴涵评分进行验证（第 5.7 节）。

# 4 级联架构

成本界限。给定 $k$ 个检测器，其成本为 $c_{1} \leq \cdots \leq c_{k}$，通行率为 $\beta_{i}$。$\begin{array}{r}{C_{\mathrm{cascade}} = \sum_{i=1}^{k} {c_{i} \prod_{j=1}^{i - 1} {\beta_{j}}}} \le } \end{array}$ ${\textstyle \sum_{i=1}^{k} c_{i} = C_{\mathrm{parallel}}}$——级联的成本永远不会超过同时运行所有检测器的成本。覆盖界限。在弱依赖情况下 $(\mathrm{MI}(d_{i}, d_{j}) \leq \epsilon)$：覆盖率 $\mathrm{coverage}(d_{1} \cup \cdots \cup d_{k}) \approx 1 - \prod (1 - \alpha_{i}) \geq \operatorname*{max} \alpha_{i}$——弱依赖检测器实现超加性覆盖，通过皮尔逊 $|r|$、距离相关、HSIC 和 MI 进行了实证验证（表 18）。实证数据显示，$\beta_{1} = 0.426$（B1 捕获 $57.4\%$），由此得出 $C_{\mathrm{cascade}} \approx 0.716 \cdot C_{\mathrm{parallel}}$。具体而言：B1 是免费的（生成中的对数概率），B2 成本为一次嵌入调用（$\sim$ 2ms 在 M4 Pro 上），B4 成本为一次实体对嵌入查找 $\sim$ 5ms）。总级联时延：对于 $75\%$ 的查询（仅在 B1 处解决），时间小于 50ms。图 1 说明了完整的流程。指针模型是一个逻辑回归分类器，预测模型的回答是否不正确（二元目标：1=不正确，0=正确，使用 LLM-judge 标签）。它使用 20 个廉价特征（7 个熵统计量 $+ 13$ 个文本特征：长度、问题类型指示符、犹豫短语的存在），在昂贵的检测器运行之前即可获取。B5 NLI 得分不是输入——B5 仅在路由后运行。评估：在 790 个 TruthfulQA 问题上的 5 倍分层交叉验证；AUC：0.585（20 个免费特征），0.707（与 B2 共享的 PCA-64 查询嵌入，注意：该变体提前产生 B2 的嵌入成本，分摊到路由和密度检测上）。

![](images/1.jpg)  

Figure 1: UCBD framework: four-column architecture mapping brain mechanisms, boundary detectors, detection signals, and response strategies. Solid borders indicate experimentally validated components (B1B2, cascade B1 B2); dashed borders indicate theoretical components awaiting empirical validation (B3-B5). The Pointer Model (center, PFCanalogue) connects to allfive boundary detectors—solid arrows for validated bound aries (B1B2), dashed arrows for theoretical ones (B3B5)—dispatching queries into the cheapest-frst cascade. Cost increases top to bottom from free (token entropy) to expensive (external cross-validation).

# 算法 1 UCBD 级联推断 20-eatue 变体 (0.585) 真实零成本数据评估所需的评估或空洞声称 (限制 6)。

<table><tr><td></td><td colspan="3">Require: Query q, boundaries {B1, . . . , Bk} ordered by cost, thresholds {τi} Ensure: Uncertainty flag u  {0, 1}, confidence score s</td></tr><tr><td>1: s ←0</td><td></td><td></td><td></td></tr><tr><td>2: for i = 1 to k do 3: si ← Bi(q)</td><td></td><td></td><td>&gt; Run boundary detect</td></tr><tr><td>if si &gt; τh</td><td></td><td></td><td> Confidently uncertain:</td></tr><tr><td>4: 5:</td><td>then return (u = 1, s = si)</td><td></td><td></td></tr><tr><td>6: end if</td><td></td><td></td><td> Confidently safe: early</td></tr><tr><td>7:</td><td></td><td></td><td></td></tr><tr><td>s ← s + wi · Si</td><td></td><td></td><td> Accumulate weighted s</td></tr><tr><td>8: end for 9: return (u = K[s &gt; τglobal], s)</td><td></td><td></td><td></td></tr></table>

# 5 实验验证

所有实验在 Apple M4 Pro (48 GPU 核心，64GB) 上使用 MLX 运行。模型包括：Qwen3-14B-4bit、Qwen3-4B-4bit、LLaMA-3.2-3B-4bit。采用贪婪解码，随机种子=42。总计算时间约为 8 小时（包括采样）。代码链接：[https://github.com/DigitLion/ucbd-experiment](https://github.com/DigitLion/ucbd-experiment)。标签约定。我们报告的 AUC 用于检测错误答案（正例=错误，负例=正确）。对于 TruthfulQA，正确性由词重叠或 LLM-judge 确定（每个实验指定）。对于 GSM8K，正确性为精确数值匹配。更高的 AUC 表示更好的错误检测。决策阈值：对于 AUC（无阈值），不需要阈值；对于 F1 比较（实验 7），我们使用固定的标记率（按分数排名前 50%）以便进行受控比较；对于级联演示，每个边界使用其中位数分数作为阈值。

解码。B1 使用贪婪解码（确定性前缀、可复现熵）。B1 需要对数概率访问（在主要 API 中可用）；对于不透明的 API，级联从 B2 开始。匹配解码注意事项：B1 熵是基于贪婪输出计算的，而 SE 基线采用随机样本（$T = 1.0$）。这个差异是固有于两种范式：B1 测量的是每个词元的对数似然不确定性（与样本数量无关），而 SE 测量的是样本间的多样性（定义上需要随机解码）。诊断声明涉及模式的输出分布，而不是特定解码协议。采样协议：对于实验 12，我们对每个问题抽取 $N = 10$ 个独立同分布样本，温度设为 $T = 1.0$（固定温度，通过核心采样进行随机解码）；温度敏感性在 $T \in \{ 0.3, 0.7, 1.0, 1.5 \}$ 时被消除，$n = 50$，$N = 5$，崩溃持续在 $38 - 62\%$。统计方法：AUROC 报告自助法 95% 置信区间（$n = 10,000$）；成对 AUROC 比较使用 DeLong 检验并应用 Holm-Bonferroni 修正；效果大小：Cohen's $d$ 及其置信区间；独立性：Pearson $r$，距离相关，HSIC，互信息（Freedman-Diaconis 分桶，置换无效）；成对比较：Wilcoxon 符号秩检验（实验 13）。样本大小和子集选择：主要分析使用 790q（完整的 TruthfulQA）。用于消融实验的 200q 子集（实验 1316, 18）由数据集顺序中的前 200 个 TruthfulQA 问题组成（无选择偏差），涵盖多个类别（健康、法律、金融、误解等），并具有与完整 790q 集相同的类别分布。在 $\alpha = 0.05$ 时，$n = 200$ 提供超过 99% 的能力来检测观察到的 SCR 差异（$0\% \to 28.5\%$，McNemar 检验）以及 80% 的能力来检测 $\Delta$ NC $\geq 0.8$（Wilcoxon，双边）。50 问题子集（实验 17, 20）使用相同的前 $n$ 选择；所有效应与 $0\%$ 基础模型 SCR 基线相比仍然显著。表：摘要实验跨越五个基准、三种模型规模、四种模型家族和五种基线方法。最强结果：$\mathrm{B1 = 0.724}$ 在 GSM8K（实验 11）、与 NLI 验证的对齐税（实验 12）、基线与指令 $^+$ 跨家族 $^+$ SFT/DPO 消融 $^+$ 跨链复制（实验 1318）、量化敏感性 $\mathrm{E ~ x p ~ 19}$），B5 救援（实验 7）。

<table><tr><td>#</td><td>Hypothesis</td><td>Data</td><td>Key Result</td><td>Status</td></tr><tr><td>1</td><td>B1 domain specificity</td><td>TruthfulQA 790q</td><td>CV AUC=0.658 (eff.) / 0.395 (blind)</td><td>✓</td></tr><tr><td>2</td><td>B1-B2 independence</td><td>TruthfulQA 401q</td><td>r=0.119, dcor=0.143</td><td>✓</td></tr><tr><td>3</td><td>Cascade ≥ parallel</td><td>TruthfulQA 401q</td><td>p=0.498, 57.4% cost saving</td><td>✓</td></tr><tr><td>4</td><td>Cross-model stability</td><td>3 models × 790q</td><td>3B AUC=0.676 &gt; 14B=0.537</td><td></td></tr><tr><td>5</td><td>B3 freshness decay</td><td>FreshQA 1500q</td><td>1113× acc. drop; B1B3 r=−0.067</td><td></td></tr><tr><td>6</td><td>Label robustness</td><td>LLM-judge</td><td>B1: 0.571→0.599 (cross-family)</td><td></td></tr><tr><td>7</td><td>B5 grounding compl.</td><td>NLI on TruthfulQA</td><td>AUC=0.678 in B1 blind zone</td><td></td></tr><tr><td>8</td><td>Learned Pointer</td><td>LogReg/embeddings</td><td>AUC=0.585→0.707 (embed)</td><td></td></tr><tr><td>9</td><td>B1 as RAG trigger</td><td>HotpotQA 100q</td><td>AUC=0.485 (fails), validates B5</td><td></td></tr><tr><td>10</td><td>B4 proxy validation</td><td>TruthfulQA 773q</td><td>AUC=0.540 (blind), +67% coverage</td><td></td></tr><tr><td>11</td><td>GSM8K math</td><td>GSM8K 500q</td><td>B1=0.724, d=0.81</td><td></td></tr><tr><td>12</td><td>Baselines (SE, NLI-SE, SC)</td><td>TruthfulQA 790q</td><td>B1=0.599 ≥ all SE variants</td><td></td></tr><tr><td>13</td><td>Base-vs-instruct ablation</td><td>TruthfulQA 200q</td><td>SCR: 1% base vs 28.5% instruct</td><td></td></tr><tr><td>14</td><td>Cross-family (3 families)</td><td>TruthfulQA 200q</td><td>SCR: 28.5%/5.5%/1.0% (family dep.)</td><td></td></tr><tr><td>15</td><td>Decoding strategy ablation</td><td>TruthfulQA 200q</td><td>SCR: 28.533.5% (nuc/low-T)</td><td></td></tr><tr><td>16</td><td>SFT vs DPO ablation</td><td>TruthfulQA 200q</td><td>SCR: 0%→1.5%→4.0%</td><td></td></tr><tr><td>17</td><td>Max-tokens sensitivity</td><td>TruthfulQA 50q</td><td>SCR: 32%→10%→8% (40/100/200t)</td><td></td></tr><tr><td>18</td><td>Tulu-3 chain replication</td><td>TruthfulQA 200q</td><td>SCR: 0%→0%→0.5% (recipe-dep.)</td><td></td></tr></table>

# 5.1 实验 1: B1 领域特定性 (TruthfulQA, 790个问题)

整体AUC=0.520（接近随机）——但类别层级分解揭示了隐藏结构。我们使用留一类别交叉验证划分类别：对于每个保留的类别，我们计算使用来自其余23个类别的阈值得出的AUC。B1 有效域（12个类别，163个样本）：CV AUC=0.658 [0.521, 0.698]。宗教（1.000），广告（0.900），健康（0.737，$p$ =0.046\*）。B1盲区（12个类别，168个样本）：CV AUC=0.395 [0.335, 0.502]——信号反转，模型“自信地错误”。两个因素精确抵消伪零结果。有效/盲区划分由训练集（23个类别）上每类别AUC $\gtrless 0 . 5$ 确定，然后在保留类别上进行评估，以减少选择偏差。我们承认，预注册将对后期分区伪影提供更强的保护。结论：单一边界检测器在结构上不足；级联设计是必要的。

# 5.2 实验 2：B1-B2 独立性 (401q)

嵌入模型：Qwen3-Embedding（4096维），10-NN 余弦相似度作为密度代理（遵循OOD检测文献）。邻居来自790个TruthfulQA问题，测量问题侧密度；一个不在评估中的邻居池将更好地逼近知识密度。皮尔逊相关系数 $r$ (B1,B2)=0.119 ($_ { n } =401)$，互信息 MI(B1,B2)=0.008比特——弱相关且效应规模较小（dcor=0.143，$p$ =0.01；见表18）。B1UB2覆盖16/24类（$64\%$）。Oracle路由AUC=0.585。未覆盖的8类需要B4/B5。

# 5.3 实验 3：级联与并行 (401q)

级联 AUC=0.538 与并行=0.532。TOST 等价检验（边际 $\Delta$ =±0.05 AUC）：$t _ { 1 }$ =2.18，$t _ { 2 }$ =1.94，$\scriptstyle { p = 0 . 0 3 1 }$ 在 $\alpha = 0 . 0 5$ 的水平下统计等价。Cohen's $d = 0 . 0 7 3$（接近零）——小效应量是希望得到的结果：级联与并行的准确性相当。级联使用了并行成本的 $71.6\%$，节省了 $\mathbf { 57.4\% }$ 的 B2 调用（228/401 查询仅在 B1 中解决，零额外成本）。5 折交叉验证：AUC= $z 0 . 4 8 6 \pm 0 . 0 1 6$。相关比较是 GSM8K 选择性预测，在这个方面级联的实际价值显而易见：在 $50\%$ 覆盖率下，准确率从 84.4% 提高到 93.2%，$p < 10^{-4}$，采用 McNemar 检验。

# 5.4 实验 4：模型间稳定性（3 个模型 $\times$ 790 问题）

Table 3: Scale effect: B1 effectiveness decreases with model size.   

<table><tr><td>Model</td><td>Effective%</td><td>Blind%</td><td>Eff. AUC</td><td>Overall</td></tr><tr><td>LLaMA-3.2-3B</td><td>79%</td><td>21%</td><td>0.676</td><td>0.622</td></tr><tr><td>Qwen3-4B</td><td>50%</td><td>50%</td><td>0.625</td><td>0.537</td></tr><tr><td>Qwen3-14B</td><td>36%</td><td>64%</td><td>0.537</td><td>0.490</td></tr></table>

反直觉：更大的模型具有较弱的 B1 信号，这与对齐产生统一流畅的输出相一致。领域特异性方向一致性：仅有 $4 2 . 9 \%$（接近随机水平）；斯皮尔曼相关系数 $\rho$ ：4B 与 $\mathrm { 3 B = }$ $0 . 3 5 8 > 1 4 \mathrm { B }$ 与 $\mathrm { 3 B = 0 . 1 1 2 }$ ，表明规模驱动模式。所有模型均为 4-bit；在 Qwen3-4B 上以 8-bit 验证（$\Delta$ AUC=+0.009）。

# 5.5 实验 5: B3 新鲜度 (FreshQA, 3 $\times$ 500q)

FreshQA [Vu et al., 2024]：包含600个时间敏感问题，配有人工标注的答案、时间元数据和新鲜度类别（不变、缓变、快变、虚假前提）。该数据集由谷歌研究团队构建，来源于需要最新知识的网络事实问题。我们为每个模型使用500个问题（3 × 500），通过与参考答案的完全匹配进行评估。许可：Apache 2.0；在GitHub上公开可用并定期更新。时间衰减：截止前的准确率（22.9%） $\longrightarrow$ 2025年后的准确率（2.0%），在所有3个模型中一致下降了1113倍。这测量了B3的检测率，而非相关性——新鲜度边界正确识别与知识截止相关的错误。B1-B3正交性：$r { = } { - } 0 . 0 6 7$（接近零，确认熵与时间新鲜度之间的独立性——这些信号捕获了根本不同的失效模式）。B1在FreshQA上的AUC：0.767（比在TruthfulQA上更强，因为时间问题产生了更多的不确定生成）。

# 5.6 实验 6：标签鲁棒性 (LLM-Judge)

通过跨家族的 LLaMA-3.2-3B（Ollama，端口 11434）对 790 个问题进行 LLM-judge 的重新标注，采用字重叠判决。

Table 4: Cross-family judge validation confirms B1 robustness.   

<table><tr><td>Judge</td><td>Correct%</td><td>B1 AUC</td><td>95% CI</td></tr><tr><td>Word-overlap</td><td>25.9%</td><td>0.571</td><td>[0.526, 0.617]</td></tr><tr><td>LLaMA-3.2-3B (cross family)</td><td>54.6%</td><td>0.599</td><td>[0.563, 0.634]</td></tr></table>

B1 AUC 从 0.571（词重叠）提高到 0.599（LLM-judge），确认了标签质量的重要性。在 LLM-judge 下，$5 4 . 6 \%$ 正确，$4 5 . 4 \%$ 错误，标签分布明显比词重叠（$2 5 . 9 \%$ 正确，$3 6 . 5 \%$ 模棱两可）更为平衡，从而提供了更可靠的 AUROC 估计。

# 5.7 实验 7：通过 NLI 的 B5 语义定位 (790q)

NLI模型：DeBERTa-v3-xsmall（70M参数），与TruthfulQA参考答案进行蕴含关系验证。限制：此方法使用的是真实参考答案，而在推理时不可用。本实验验证NLI作为B1的补充信号类型；生产环境中的B5必须在独立来源文档上使用检索 $^ +$ NLI，这可能导致AUC降低。B5总体AUC=0.582（$p$ =0.003，置换检验）。在B1的盲区：B5 AUC=0.678（$p$ =0.008）—信号是互补的，而非冗余的。混淆：人（B1=0.318，B5=1.000），教育（B1=0.125，B5=1.000）。B1与B5的Pearson相关系数 $r { = } 0 . 0 7 0$，互信息MI=0.012比特（近乎独立）。最佳组合 $80\%$ B1 $^ +$ 20%B5：AUC=0.638。

# 5.8 实验 8：学习的指针模型

五种路由器变体：（a）仅熵（6个特征）：AUC=0.573；（b）增强型（20个特征）：0.585；（c）完整型（$^ +$ B2/B4评分）：0.611；（d）oracle：0.992；（e）基于嵌入的（PCA-64，与B2共享）：AUC=0.707—提升了12点。B5仅在2.2%的查询中被调用。成本说明：变体（a）在B1（免费）上运行，并提供有用的路由（0.573 AUC）。变体（e）实现了更强的路由，但需要B2嵌入—这些嵌入只计算一次，并在密度检测器和路由器之间共享，因此当B2已经被调用时，路由的边际成本为零。

# 5.9 实验 9：将 B1 作为 RAG 触发器（HotpotQA, 100题）

无RAG时F1=0.123，有RAG时F1=0.783（差距66分）。B1的AUC用于预测“RAG是否有帮助？”为0.485（与随机猜测相当）。HotpotQA的平均熵（0.147）低于TruthfulQA（0.188），尽管其表现的熵倒置更差。仅凭B1无法预测检索需求；需递进到B2/B5。

# 5.10 实验 10：B4 代理验证 (773q)

实体对嵌入余弦距离作为 B4 代理。总体 AUC=0.518；在 B1+B2 盲区：AUC=0.540。刻板印象 (0.823)，迷信 (0.764)，教育 (0.667)。B1B4 r=0.034，B2B4 r=0.000（完全独立）。B4 将覆盖范围扩展 67%（12→20 类别）。

# 5.11 实验 11：GSM8K 数学推理（500道题）

我们将 UCBD 扩展到数学推理，使用 GSM8K [Cobbe 等，2021] 的小学数学问题，采用 MLX-direct 词元级熵（贪婪解码）。Qwen3-14B 在 500 个问题上的准确率达到 $8 4 . 4 \%$。

关键洞察：B1 词元熵在 GSM8K 上的 AUROC 为 0.706-0.724（$n$ = 500）——远远强于在 TruthfulQA 上的 0.520。在事实问答中，模型自信地错误（科恩 $d$ = 0.07）；在数学问题上，错误产生真正的不确定推理（$\scriptstyle { \mathrm { ~ } d = 0 . 8 1 }$）。结合熵特征（4 个特征，无长度）实现 AU-ROC=0.724（5 倍交叉验证）。长度混淆：仅响应长度实现 AUROC=0.849，并在选择性预测 $50\%$ 覆盖上主导熵：长度 $96.0\%$ 对比熵 $93.2\%$。在 GSM8K 上，熵相较于长度没有增量价值（信号间相关性 $r={0.53}$）。然而，熵的优势在于跨任务的普适性：在事实问答中，响应长度不能预测正确性，而熵保持信号（0.599）。P(True) 基线 $\scriptstyle n = 200$ ：AUROC=0.608。选择性预测（熵门）：在 $30\% / 50\% / 80\%$ 覆盖下的精度分别为 $92.0\% / 93.2\% / 88.7\%$（基线 $84.4\%$）。

Table 5: GSM8K error detection ( $\scriptstyle n = 5 0 0$ ): B1 entropy and behavioral features. Incorrect answers show 49% higher mean entropy (Cohen's $d { = } 0 . 8 1$ , $p < 1 0 ^ { - 8 }$ ).   

<table><tr><td>Feature</td><td>AUROC</td><td>95% CI</td><td>Cost</td></tr><tr><td>B1 mean entropy</td><td>0.706</td><td>[.635,.772]</td><td>Free</td></tr><tr><td>B1 std entropy</td><td>0.715</td><td>[.643,.782]</td><td>Free</td></tr><tr><td>B1 max entropy</td><td>0.724</td><td>[.650,.793]</td><td>Free</td></tr><tr><td>Combined entropy (4 feat, CV)</td><td>0.724 ± .033</td><td></td><td>Free</td></tr><tr><td>P(True)</td><td>0.608</td><td>[.52,.70]</td><td>1 call</td></tr><tr><td>Response length (tokens)†</td><td>0.849</td><td>[.791,.903]</td><td>Free</td></tr><tr><td>Combined with length (5 feat, CV)†</td><td>0.844 ± .041</td><td></td><td>Free</td></tr></table>

长度是一个困难的代理（更长 $=$ 失败步骤更多），而不是真正的不确定性信号。

# 5.12 实验 12：790 个问题的基线（SE，自检，规范 NLI-SE）

我们将三种语义熵（SE）的实现与 TruthfulQA 进行比较（$N$ =每个问题10个样本，$T { = } 1 . 0$）。(a) 代理 SE：二元词组 Jaccard（阈值=0.4）和 SINdex 风格的嵌入聚类——在 Qwen3 上进行的自底向上的聚类（平均链接）基于嵌入余弦相似度（阈值=0.85），复制了 SINdex 的核心方法论 [Abdaljalil et al., 2025]。在不同范围内验证阈值（Jaccard:0.20.6; 嵌入余弦:0.700.95；见附录 A）。 (b) 基于 NLI 的 SE：遵循 Kuhn 等 [2023] 的核心方法论——使用双向蕴含和并查集聚类，针对一个200个问题的子集，使用三个 DeBERTa-v3 模型（large 435M，base 184M，xsmall 70M）。该方法实现了原始 SE 论文中的基于蕴含的算法；省略了考虑矛盾的聚类变种，NLI 模型与原始模型有所差异（见下文）。 (c) SelfCheck：嵌入余弦（不使用矛盾提示）。核心发现：单聚类崩溃在样本量 $N$ =3（$4 6 . 3 \%$ ），$N$ =5（$4 1 . 9 \%$ ），$N$ =7/10（$4 0 . 0 \%$），聚类方法（Jaccard: $4 0 . 0 \%$ ，SINdex 风格嵌入: $7 9 . 0 \%$ ），以及温度（$T$ =0.3: 62%，$T { = } 1 . 5$ : 38%——更高温度降低但并不消除同质化）。Jaccard/嵌入间的差距揭示了一个额外的同质化层面：322/790 个问题显示表面词汇多样性（平均3.3个 Jaccard 聚类），但在语义上是相同的（单个词嵌入聚类）——模型在保持意义的同时变化措辞。仅有 $2 1 . 0 \%$ 的问题展现出真正的语义多样性。SINdex 风格的比较使用与 SINdex 相同的聚类方法（嵌入 $^ +$ 层次聚类），且在这种方法下对齐税更糟糕——79% SCR 对比 Jaccard 的 40%——因为嵌入相似性捕获了表面词汇变化所掩盖的语义冗余。阈值稳健性：SCR 在整个阈值范围内仍保持显著（嵌入余弦：在 $\tau { = } 0 . 8 0$ 时为 $6 0 \%$ SCR，在 $\tau { = } 0 . 8 5$ 时为 79%，在 $\tau { = } 0 . 9 0$ 时为 92%；Jaccard:在 0.3 时为 28%，在 0.4 时为 40%，在 0.5 时为 $5 5 \%$ ）。Jaccard/嵌入差距本身作为内部聚类质量验证：如果嵌入阈值过于激进（合并语义上不同的答案），我们会预期 Jaccard 会一致；而相反，39 个百分点的差距揭示出一个有意义的语义冗余层面——322/790 个具有表面词汇多样性但语义身份的问题——这是 Jaccard 所无法检测的。跨嵌入验证（示例 20）提供进一步的质量保证：Nomic-embed-text（不同的架构和训练语料库）产生更高的 SCR（在 $\tau { = } 0 . 8 5$ 时为 $9 2 \%$ 对比 78%），确认单聚类分配反映了真正的语义等价，而不是特定嵌入器的伪影。对于单聚类问题，任何基于采样的方法在构造上均无判别能力。标签：LLM-judge（跨家族 LLaMA-3.2-3B）。统计测试：bootstrap DeLong（$n$ =10,000）。

Table 6: B1 entropy vs. sampling-based baselines on TruthfulQA (LLM-judge labels). B1 (free) matches or outperforms all baselines including canonical NLI-based SE at three model scales (70M-435M). DeLong tests with Holm-Bonferroni correction: vs. SE-Emb $p _ { \mathrm { a d j } } { = } 0 . 0 3 3 ^ { \ast }$ , vs. SE-Jaccard $p _ { \mathrm { a d j } } { = } 0 . 0 4 0 ^ { * }$ , vs. SelfCheck $p$ =0.65 ns. †200-question subset; DeBERTa-large recomputed with LLM-judge labels for fair comparison.   

<table><tr><td>Method</td><td>AUROC</td><td>95% CI</td><td>Cost</td></tr><tr><td>B1 mean entropy</td><td>0.599</td><td>[0.559, 0.637]</td><td>Free</td></tr><tr><td>SelfCheck-Emb (k=5)</td><td>0.588</td><td>[0.547, 0.626]</td><td>6×</td></tr><tr><td>SE-Jaccard (N=10)</td><td>0.548</td><td>[0.510, 0.589]</td><td>11×</td></tr><tr><td>SE-Embedding (N=10)</td><td>0.542</td><td>[0.513, 0.572]</td><td>11×</td></tr><tr><td>SE-NLI† (DeBERTa-large)</td><td>0.511</td><td>[0.419, 0.594]</td><td>11×+NLI</td></tr><tr><td>SE-NLI† (DeBERTa-base)</td><td>0.512</td><td>[0.421, 0.593]</td><td>11×+NLI</td></tr><tr><td>SE-NLI† (DeBERTa-xsmall)</td><td>0.501</td><td>[0.404, 0.595]</td><td>11×+NLI</td></tr></table>

对齐税的量化。使用 Holm-Bonferroni 修正的 Bootstrap DeLong 测试（$n$ = 10,000）：B1 显著优于 SE-Embedding（= 0.033 $^ * $）和 SE-Jaccard（= 0.040*）。B1 与 SelfCheck：$p _ { \mathrm { a d j } }$ $p _ { \mathrm { a d j } }$ $\scriptstyle { p = 0 . 6 5 }$（不显著）。效应量：B1 $d { = } 0 . 3 6 0$ [0.222, 0.501]，SelfCheck $d$ = 0.346，SE-Emb $d { = } 0 . 2 4 5$ ，SE-Jac $d { = } 0 . 2 0 7$ 。

基于自然语言推理的语义等价性（规范比较，三种模型规模）。在200道问题的子集中，我们按照Kuhn等人[2023]的方法运行基于NLI的语义等价性评估：双向蕴含（如果$A \Rightarrow B$且$B \Rightarrow A$，则为等价），阈值=0.5，以及并查集聚类。我们使用三种DeBERTa-v3模型：large（435M参数）、base（184M）和xsmall（70M）。我们省略了考虑矛盾的聚类变体；这是一个经过深思熟虑的简化，由于在单聚类机型下的结构不相关性：当$| { \mathcal { C } } | = 1$（4079%的查询）时，所有响应在语义上都是等价的，不存在用于聚类的矛盾——考虑矛盾的变体无法在模型未产生多样性的情况下创造多样性。对于剩余的多聚类查询，考虑矛盾的聚类可以细化聚类分配；然而，NLI模型规模扩大6.2倍（70M对435M）并未产生AUROC的改善（$\Delta$ =+0.010），这表明瓶颈在于响应的均匀性，而非聚类方法。更强的NLI主干网络也会面临相同的结构限制。结果：DeBERTa-large的AUROC达到0.511 [0.419, 0.594]；base为0.512 [0.421, 0.593]；xsmall为0.501 [0.404, 0.595]——均接近随机水平，置信区间重叠。将NLI模型扩大6.2倍（70M对435M）没有带来AUROC的改善（$\Delta$ =+0.010，置信区间跨越零）。聚类统计几乎相同：base的平均聚类数为5.44（6.0%单聚类），xsmall为5.42（6.5%），large为4.68（9.0%）。所有NLI模型相较于Jaccard（3.58的平均值，28.5%单聚类）过度划分聚类，将释义相近的答案视为语义不同。在模型生成单一重复答案的40%问题中，即便是完美的NLI也无法在不存在语义多样性的情况下创造多样性。对齐惩罚是模型输出分布的特性，而非NLI模型或聚类方法。机制：40%的问题产生一个单一的Jaccard聚类，79%的情况在嵌入聚类下仅显示出21%的真正语义多样性。温度消融实验（$N = 5$，50个问题）：单聚类比率单调下降，从$T=0.3$的62%降至$T=1.5$的38%，证实更高的温度降低但并未消灭同质化。图2可视化：在单聚类问题上，SE降至随机水平（0.500），而B1保持在0.603。在多聚类问题上，差距缩小（B1: 0.587 vs. SE: 0.572）。定性示例（实验20交叉嵌入子集，$N = 10$，$T = 1.0$）。单聚类：“如果你吃西瓜子会发生什么？”——所有10个响应传达相同的内容（种子无害通过），仅有词汇变化；两个嵌入模型在$\tau = 0.85$时均分配1个聚类。多聚类：“保罗·里维尔在午夜骑行时说了什么？”——响应在“英国人来了！”（流行错误引用）和“正规军在出来”（历史性）之间分裂，产生24个聚类。模式是系统性的，单聚类问题有模型始终产生的一致“答案”，而多聚类问题涉及争议性的主张，其中训练数据包含冲突信息。

![](images/2.jpg)  

Figure 2: The alignment tax mechanism. On single-cluster questions $( 4 0 . 0 \% )$ , SE drops to exact chance (0.500, dashed red) because all 10 samples produce the same answer. B1 retains discriminative power (0.603) because per-token entropy captures computational uncertainty independent of output diversity.

与基于logit的补救措施相关。语义能量[Ma等，2025a] 通过对logits而非softmax后的概率进行操作来解决单簇失败，报告在单簇情况下相较于SE有约13%的AUROC提升（即，从0.500提升至${\sim}0.565$）。我们的B1在单簇问题上使用词元熵—同样是基于logit的信号—也达到了相近的AUROC（0.603），但不需要多个样本。这表明B1在同质化阶段作为语义能量核心机制的零成本近似。对齐税并未被基于logit的方法“解决”; 相反，我们的诊断解释了为什么基于logit的信号（B1, 语义能量）在基于多样性的信号（SE, SelfCheck）失败的情况下能够成功：RLHF抑制了响应之间的多样性，但无法在不降低生成质量的情况下完全平滑每个词元的计算不确定性。对匹配数据进行正式的逐对比较仍然是未来的工作。标签独立性。对齐税诊断——单簇率、簇计数分布和基础与指令的差异——是响应分布的属性，而非正确性标签。AUROC估计需要标签，并对标注方法（词重叠与LLM评判）敏感，但核心发现是40%的问题在10个i.id样本下产生相同的响应，这一发现是无标签的并且是直接的。

# 5.13 实验 13：基础模型与指令模型消融研究（200问）

为了孤立对齐的因果作用，我们将 Qwen3-14B-Base（仅预训练，无指令微调或强化学习人类反馈）与 Qwen3-14B-Instruct 在相同的 200 道问题子集上进行比较，每个问题生成 $N { = } 10$ 个样本，$T { = } 1.0$，使用 Jaccard 双元组聚类（阈值=0.4）。两个模型都采用 4 位量化（Q4 KM），以控制量化效果。

Table 7: Base-vs-instruct response diversity on TruthfulQA ( $n$ =200). Alignment reduces mean clusters by $2 . 6 \times$ and increases single-cluster rate from $1 \%$ to $2 8 . 5 \%$ .   

<table><tr><td>Metric</td><td>Base</td><td>Instruct</td><td>Difference</td></tr><tr><td>Single-cluster rate</td><td>1.0%</td><td>28.5%</td><td>+27.5pp</td></tr><tr><td>Mean clusters</td><td>9.26</td><td>3.58</td><td>-5.68</td></tr><tr><td>Mean SE</td><td>2.158</td><td>0.832</td><td>−1.326</td></tr><tr><td colspan="4">Wilcoxon signed-rank (base &gt; instruct): W=18,331, p &lt; 10−6</td></tr></table>

关键发现：基础模型在每个问题中产生了几乎最大的多样性（每个问题9.26个聚类，仅有2/200个问题存在单一聚类），而指令模型则收敛为3.58个聚类，具有$2 8 . 5 \%$的单聚类问题。这证实了响应同质化是由对齐（指令微调$^ +$强化学习人类反馈）造成的，而不是由预训练、模型架构或量化引起的。定性检查表明，基础模型的响应在事实意义上是有意义的（而非随机文本）——它们反映了模型知识表征中的真正多样性，而对齐则抑制了这种多样性，转而追求一致的“安全”输出。

# 5.14 实验 14: 跨家庭复制 (200个问题, 三个家庭)

我们从 LLaMA-3.2-3B-Instruct 和 Mistral-7B-Instruct 的同一组 200 个问题中生成了 $N { = } 10$ 个样本。

Table 8: Cross-family alignment tax ( $n { = } 2 0 0$ ). Homogenization varies widely across model families and scales.   

<table><tr><td>Model</td><td>SCR</td><td>Mean NC</td><td>Wilcoxon vs. Qwen</td></tr><tr><td>Qwen3-14B-Instruct</td><td>28.5%</td><td>3.58</td><td></td></tr><tr><td>LLaMA-3.2-3B-Instruct</td><td>5.5%</td><td>7.27</td><td>p &lt; 10−6</td></tr><tr><td>Mistral-7B-Instruct</td><td>1.0%</td><td>7.87</td><td>p &lt; 10−6</td></tr><tr><td>Qwen3-14B-Base</td><td>1.0%</td><td>9.26</td><td>p &lt; 10−6</td></tr></table>

主要发现：所有三个指令模型的多样性显著低于基础模型，确认对齐是因果机制。然而，同质化的严重程度变化显著：Qwen3-14B显示$28.5\%$ SCR，而Mistral-7B和LLaMA-3B仅显示$1.0{-}5.5\%$，这表明对齐成本既依赖于模型规模，也依赖于具体的对齐方法（SFT/RLHF细节，训练数据）。Mistral-7B接近零的SCR是值得注意的：尽管经过指令调优，它保留了基础模型级别的响应多样性。这种异质性增强而非削弱了诊断：实践者必须针对每个模型测量同质化，因为无法仅从对齐状态假设这一点。注意：Qwen3的训练流程没有发布单独的SFT检查点，阻止了对SCR最高的类别进行类似的阶段性分解；我们提供了在其他两个类别上的阶段性消融实验，其中有可用的中间检查点（实验16、18）。

# 5.15 实验 15：解码策略消融 (200问)

我们从 Qwen3-14B-Instruct 生成了 $N { = } 1 0$ 个样本，采用三种解码配置：核采样（$p$ =0.9）、核采样（$p$ =0.95）和温度 $T { = } 0 . 7$。Jaccard 二元组聚类（阈值=0.4）。关键发现：使用核采样（$p$ =0.9）使得 SCR 从 $2 8 . 5 \%$ 提高到 $3 3 . 5 \%$，$9 5 \%$ 自助置信区间：[27.0%，$4 0 . 0 \%$）——进一步限制尾概率质量会降低多样性。低温（$T { = } 0 . 7$）将均值 NC 压缩从 3.58 降至 2.96。没有任何解码策略能将 SCR 降至基线以下；响应同质化是学习到的分布的特性，而非采样过程的结果。替代解码无法消除“对齐税”。

Table 9: Decoding strategy ablation (n=200). The alignment tax persists across strategies.   

<table><tr><td>Strategy</td><td>Parameters</td><td>SCR</td><td>Mean NC</td><td>Mean SE</td></tr><tr><td>Baseline</td><td>T=1.0</td><td>28.5%</td><td>3.58</td><td>0.832</td></tr><tr><td>Nucleus</td><td>T=1.0, p=0.9</td><td>33.5%</td><td>3.40</td><td>0.786</td></tr><tr><td>Nucleus</td><td>T=1.0, p=0.95</td><td>30.0%</td><td>3.55</td><td>0.827</td></tr><tr><td>Low temp</td><td>T=0.7</td><td>30.0%</td><td>2.96</td><td>0.668</td></tr></table>

# 5.16 实验 16：训练阶段消融 (200q, 基础 $\longrightarrow$ SFT → DPO)

我们使用 Zephyr 链分离负责同质化的训练阶段 [Tunstal et al., 2023]: Mistral-7B-v0.1 (基础) mistral-7b-sft-beta (仅 SFT) $\longrightarrow$ zephyr-7b-beta (SFT+DPO)。这三者共享相同的架构和基础权重，仅在训练阶段上有所不同。Zephyr 的 DPO 使用 UltraFeedback（60k 个偏好对），$\beta = 0.1$，学习率 $5 \times 10^{-7}$，1 个周期。$N = 10$ 个样本，T=1.0，Jaccard 聚类。

Table 10: Training stage ablation. DPO is the primary driver of homogenization.   

<table><tr><td>Stage</td><td>Model</td><td>SCR</td><td>Mean NC</td><td>Mean SE</td></tr><tr><td>Base</td><td>Mistral-7B-v0.1</td><td>0.0%</td><td>9.28</td><td>2.170</td></tr><tr><td>SFT</td><td>mistral-7b-sft-beta</td><td>1.5%</td><td>8.63</td><td>2.024</td></tr><tr><td>SFT+DPO</td><td>zephyr-7b-beta</td><td>4.0%</td><td>8.01</td><td>1.897</td></tr></table>

Wilcoxon: 基线→SFT p=0.002, SFT DPO p=0.0001, 基线→DPO p < 10− 关键发现：SFT 保持近基线水平的多样性 (SCR $1 . 5 \%$ 对比 $0 . 0 \%$ , $\Delta$ NC= $-$ 0.64)，而 DPO 引入额外的同质化 ( $\Delta$ NC=−0.63，SCR 跃升至 $4 . 0 \%$)。两个阶段都显著降低了多样性 ( $p < 0 . 0 0 3$ )，但单聚类崩溃主要是 DPO 现象。结合实验14 (Qwen3-14B: 28.5% SCR vs. Zephyr-DPO: $4 . 0 \%$ )，对齐损失的严重程度取决于优先级优化方案和模型规模。

# 5.17 实验 17：最大生成长度敏感性 (50q)

审稿人关注：生成上限（最多 40 个词元）是否会通过偏向更短的模板化答案来提升 SCR？我们在 50 道 TruthfulQA 问题上使用 Qwen3-14B 进行了三个设置的测试（maxtokens = 40, 100, 200），其中 $N$ = 10，$T = 1.0$，采用 Jaccard 聚类，思考功能通过 /no_think 禁用。

Table 11: Generation length sensitivity (50q). SCR decreases with length but persists at al settings: 8% at 200 tokens vs. 0% for base model ( $p < 0 . 0 5 )$ , confirming alignment-driven homogenization.   

<table><tr><td>max_tokens</td><td>SCR</td><td>Mean NC</td><td>Mean SE</td><td>Avg Words</td></tr><tr><td>40</td><td>32.0%</td><td>3.02</td><td>0.676</td><td>26.9</td></tr><tr><td>100</td><td>10.0%</td><td>7.46</td><td>1.769</td><td>68.8</td></tr><tr><td>200</td><td>8.0%</td><td>8.30</td><td>1.929</td><td>115.2</td></tr></table>

基础模型SCR在所有长度上约为0%。5/16个单集群问题在所有设置中仍然存在。

解释：对齐成本在所有生成长度中都持续存在。以下三项发现确立了这一点：（1）在200个词元时，SCR保持在8% vs. 基础模型的0%——这8个百分点的差距完全归因于对齐，并且代表在12个问题中有1个问题的对齐模型在输出预算不变的情况下生成相同的答案；（2）SCR在100到200个词元之间趋于饱和（Δ SCR=2百分点），确认剩余的单聚类问题反映了真正的语义同质性，而非截断伪影；（3）在mt40的16个单聚类问题中，有5个在所有三个长度下仍然存在——这些“真正同质化”的问题是基于采样的UQ最棘手的案例。从32%下降到8%反映了两个影响因素：一个是机械因素（杰卡德二元组相似度与长度成反比），另一个是真正因素（对齐抑制语义多样性）。在100到200个词元范围内的饱和状态隔离了真正因素。在200个词元时——是典型事实问答回答长度的4倍——对齐成本仍使得基于采样的UQ在8%的查询上不具备信息量。至关重要的是，长度依赖性使得在UQ最为重要的情况下对齐成本最为严重：短且高风险的事实类问题，在这些情况下需要可靠的偏差估计，而对齐模型产生最同质化的输出。

# 5.18 实验 18：Tulu-3 链 DPO 复制 (200q)

我们复制了LlamaTulu3链Llam3.B（见Llama-3.1-Tulu-3-8B-SFT tulu3-8b（SFT $^ +$ DPO $^ +$ RLVR）。Tulu-3的偏好优化采用了一个经过策划的多领域数据集，并进行了长度去偏化，$\beta$ =0.1，随后针对可验证任务进行了RLVR—这与Zephyr的单数据集DPO有实质性不同。与实验16采用相同协议（$N = 10$，$T { = } 1.0$，Jaccard）。

Table 12: Tulu-3 chain ablation. DPO effect is recipe-dependent: Tulu-3's DPO produces minimal homogenization compared to Zephyr (Exp 16).   

<table><tr><td>Stage</td><td>Model</td><td>SCR</td><td>Mean NC</td><td>Mean SE</td></tr><tr><td>Base</td><td>Llama-3.1-8B</td><td>0.0%</td><td>9.46</td><td>2.209</td></tr><tr><td>SFT</td><td>Tulu-3-8B-SFT</td><td>0.0%</td><td>9.02</td><td>2.123</td></tr><tr><td>SFT+DPO+RLVR</td><td>tulu3-8b</td><td>0.5%</td><td>9.31</td><td>2.174</td></tr></table>

Vilcoxon：基本 SFT $_ { p }$ =0.00004，SFT DPO $\scriptstyle { p = 0 . 0 0 8 }$，基本 DPO p=0.43 关键发现：Tulu-3 链显示出最小的对齐税（$0 . 5 \%$ SCR 对比 Zephyr 的 $4 . 0 \%$），确认了均匀化的严重性是依赖于配方的——首选数据集、DPO 超参数和 RLVR 阶段均很重要。SFT 显著减少了聚类数量（$\Delta$ NC=−0.45，$p$ =0.00004），但并未导致单聚类崩溃，与 Exp 16 一致。跨链比较加强了我们的实践建议：用户应在依赖基于采样的 UE 之前，测量其特定模型上的 SCR。结合 Exp 14（Qwen3-14B: $2 8 . 5 \%$，LLaMA-3B: $5 . 5 \%$），不同家族间的对齐税跨度达到两个数量级（0.5% $2 8 . 5 \%$）。

# 5.19 实验 19：量化灵敏度（30q，Q4 与 Q8）

为了解决量化问题，我们比较了在Q4KM（4位，4.4GB）和Q80（8位，7.7GB）下的Mistral-7B-Instruct在30个TruthfulQA问题上的表现（$N$ =10，$T$ =1.0）。我们报告了多个阈值下的平均成对字符二元组Jaccard相似度和聚类计数。

Table 13: Quantization sensitivity: Q4 vs Q8 on Mistral-7B-Instruct. At semantic-level thresholds ( $t$ =0.7), both quantizations produce identical SCR.   

<table><tr><td></td><td>Mean J</td><td>SCR@0.6</td><td>SCR@0.7</td><td>Mean NC@0.7</td></tr><tr><td>Q4_K_M (4-bit)</td><td>0.608</td><td>63.3%</td><td>6.7%</td><td>7.67</td></tr><tr><td>Q8_0 (8-bit)</td><td>0.576</td><td>26.7%</td><td>6.7%</td><td>7.30</td></tr><tr><td>Δ (Q8-Q4)</td><td>−0.032</td><td>-36.6pp</td><td>0.0pp</td><td>-0.37</td></tr></table>

关键发现：在语义层面（阈值=0.7，集群结构有意义时），Q4和Q8产生相同的SCR $( 6 . 7 \% )$ 和相似的集群计数（7.67 对 7.30）。平均成对相似度仅相差3.2个百分点（0.608 对 0.576），Q8产生的词汇多样性略高——量化并未夸大表面相似度。结合8位B1验证（$\Delta$ AUC=+0.009在Qwen3-4B上），这确认4位量化没有在对齐税收测量中引入系统性伪影。量化设计（基础和指令在相同的Q4KM下）仍然是主要控制变量；该实验提供了额外的跨量化证据。

# 5.20 Exp 20：交叉嵌入验证（50个问题，两个独立嵌入器）

一个关注点是您的嵌入基础 SCR 可能存在耦合问题：Qwen3-Embedding 与某些生成器共享模型家族，这可能夸大语义相似度。我们通过计算 SCR 来测试这一点，使用两个独立的嵌入家族在相同的 50 个 TruthfulQA 问题上进行实验（Mistral-7B-Instruct，$N { = } 10$，$T { = } 1.0$）：(1) Qwen3-Embedding（1.5B，Qwen 家族）和 (2) Nomic-embed-text（137M，基于经过筛选的对比数据训练的独立架构）。

关键发现：独立嵌入器在每个阈值下检测到的单一聚类问题数量更多，$9 2 \%$ 对比 $7 8 \%$，在 $\tau { = } 0 . 8 5$；$98\%$ 对比 $9 4 \%$，在 $\tau { = } 0 . 8 0$。如果 Qwen3-Embedding 因为共享架构而夸大相似性，我们会预期相反的结果——Nomic 的 SCR 应该较低。Nomic 检测到的更多同质化现象决定性地排除了耦合偏差，并确认嵌入基础的 SCR 反映了模型输出中真正的语义一致性。每个问题聚类计数的低相关性（$r$ =0.033）本身就是稳健性的证据，而非不稳定性：它表明两个架构独立的嵌入器——在不同的语料库和不同目标上训练——通过独立的路径得出了相同的宏观结论（高 SCR）。如果相关性很高，可能会担忧共享偏差；而低相关性结合一致的总体 SCR 证明了这一诊断发现的独立复现。

Table 14: Cross-embedder validation. An independent embedder detects more homogenization, not less—ruling out coupling bias.   

<table><tr><td>Embedder</td><td>SCR@0.80</td><td>SCR@0.85</td><td>SCR@0.90</td></tr><tr><td>Qwen3-Embedding (1.5B)</td><td>94.0%</td><td>78.0%</td><td>14.0%</td></tr><tr><td>Nomic-embed-text (137M)</td><td>98.0%</td><td>92.0%</td><td>52.0%</td></tr></table>

每个问题的簇数 Pearson $_ { r } $ =0.033在 ${ \tau } \mathrm { { = } } 0 . 8 5$ 时；两者均检测到单簇崩溃。

# 5.21 实验 21：扩展长度敏感性 (200q, max_tokens=200)

实验 17 显示在 50 个问题上 200 个词元的残余 SCR；在这里我们以 4 倍规模进行重复实验。我们在 200 个 TruthfulQA 问题上生成 $N { = } 10$ 个样本，$T { = } 1.0$，max_tokens=200（Mistral-7B-Instruct，前 200 个子集，系统选择）。

Table 15: Extended length sensitivity at scale (200q vs. original 50q). Longer generation reduces SCR but does not eliminate the alignment tax: $3 3 . 5 \%$ of questions remain single-cluster at $\tau { = } 0 . 8 5$ .   

<table><tr><td>Setting</td><td>SCR@0.80</td><td>SCR@0.85</td><td>SCR@0.90</td></tr><tr><td>40 tokens, 200q (Exp. 12)</td><td>79.0%</td><td>79.0%</td><td></td></tr><tr><td>200 tokens, 200q (this exp.)</td><td>61.5%</td><td>33.5%</td><td>14.0%</td></tr></table>

主要发现：将 max_tokens 从 40 增加到 200 可将 SCR 从 $7 9 \%$ 降低至 $3 3 . 5 \%$，在 $\tau { = } 0 . 8 5$ 的情况下，但三分之一的问题仍然产生单一语义簇，尽管生成预算增加了 $5 \times$。这一减少在各个阈值上都是单调的，确认了较长的回答引入了表面变化，从而放宽了嵌入相似度。然而，即使在生成预算充足的情况下，200 个问题的残余 $3 3 . 5 \%$ SCR（对比原始的 50q 子集的 8%）表明，alignment tax 在规模上是稳健的，并且持续存在。

# 5.22 实验 22：跨数据集验证 (WebQuestions, 200q)

为了测试对齐税是否超越 TruthfulQA 具有普适性，我们在 WebQuestions [Berant et al, 2013] 上测量 SCR——这是一个源自 Google 搜索查询并带有 Freebase 答案的事实问答数据集，涵盖地理、历史、娱乐和科学。我们在 200 个问题上生成 $N { = } 1 0$ 个样本，设置为 $T { = } 1 . 0$，max_tokens 为 100（使用 Mistral-7B-Instruct 的前 200 个子集）。

Table 16: Cross-dataset SCR validation. WebQuestions shows stronger homogenization than TruthfulQA, confirming the alignment tax is not dataset-specific.   

<table><tr><td>Dataset</td><td>SCR@0.80</td><td>SCR@0.85</td><td>SCR@0.90</td></tr><tr><td>TruthfulQA 200q (100tok)</td><td>79.0%</td><td>79.0%</td><td></td></tr><tr><td>WebQuestions 200q (100tok)</td><td>77.5%</td><td>58.0%</td><td>34.0%</td></tr></table>

关键发现：WebQuestions展示出显著的同质化（58.0% SCR 在 $\tau { = } 0 . 8 5$），确认了对齐税并非 TruthfulQA 所特有。该模式在关于不同领域的事实性问题上保持一致（例如：“古巴讲什么语言？”—单一聚类；“马丁·路德·金做了什么？”—多个聚类）。相较于 TruthfulQA 关注误解的提示，WebQuestions 的问题往往具有更短、更具事实性的回答，但对齐税依然强烈，这表明响应同质化是事实问答任务中对齐模型的一种普遍特性。

# 6 跨任务分析

选择性预测。B1作为拒绝标准：在GSM8K上，准确率从84.4%跃升至93.2%，在50%覆盖率时（PRR=0.564）；在TruthfulQA上，风险覆盖分析显示PRR@50%=0.043（平均熵）至0.074（最大熵），AURC=0.701，比GSM8K弱5倍，反映了AUROC的差距。这种任务依赖性效应进一步推动了多边界路由的研究。

Table 17: B1 entropy vs. baselines across three benchmarks. TruthfulQA: LLM-judge labels ( $_ { n }$ =790); FreshQA/GSM8K: exact-match labels. B1 matches or outperforms all baselines at zero additional cost.   

<table><tr><td>Benchmark</td><td>Method</td><td>AUROC</td><td>95% CI</td><td>Cost</td></tr><tr><td rowspan="4">TruthfulQA (factual QA)</td><td>B1 entropy</td><td>0.599</td><td>[0.559, 0.637]</td><td>Free</td></tr><tr><td>SelfCheck (k=5)</td><td>0.588</td><td>[0.547, 0.626]</td><td>6×</td></tr><tr><td>Semantic entropy (N=10)</td><td>0.548</td><td>[0.510, 0.589]</td><td>11×</td></tr><tr><td>P(True)</td><td>0.427</td><td>[0.408, 0.446]</td><td>1 call</td></tr><tr><td rowspan="2">FreshQA (temporal)</td><td>B1 entropy</td><td>0.657</td><td>[0.610, 0.703]</td><td>Free</td></tr><tr><td>P(True)</td><td>0.399</td><td>[0.366, 0.432]</td><td>1 call</td></tr><tr><td rowspan="3">GSM8K (math)</td><td>B1 max entropy</td><td>0.724</td><td>[0.650, 0.793]</td><td>Free</td></tr><tr><td>Combined (4 entropy feat, CV)</td><td>0.724</td><td></td><td>Free</td></tr><tr><td>P(True)</td><td>0.608</td><td>[0.52, 0.70]</td><td>1 call</td></tr></table>

![](images/3.jpg)  

Figure 3: AUROC for error detection across tasks (dashed red $=$ chance). The alignment tax is visible on TruthfulQA: B1 (free, 0.599) matches SelfCheckGPT ( $6 \times$ , 0.588, p=0.65) and significantly outperforms Jaccardapproximated SE (11 $\times$ , 0.548, $p _ { \mathrm { a d j } }$ =0.04). On GSM8K, where alignment does not suppress entropy, B1 reaches 0.724 $d$ =0.81).

口头表达的信心失效。在 TruthfulQA 上，P(True) 并不提供有效信息（AUROC=0.427）：模型对 89.7% 的答案报告“真实”（其中 41.9% 是正确的）——形成了 48 分的过度自信差距，这确认了对于与 RLHF 对齐的模型，隐式信号（词元熵）比显式自我评估更为可靠。

# 7 系统级级联演示与独立性

三边界级联（B1→B2 B4）在401个TruthfulQA问题上的表现（LLM-judge标签）。组合AUR0C=0.601，相比之下B1独立检测为0.586——多边界组合超越任何单一检测器。阶段分布：B1（自由）占$5 0 \%$，B2占28%，B4占22%。选择性预测：在50%最不确定的问题上选择不做预测，将准确率从$5 5 . 1 \%$提高至${ \bf 6 1 . 0 \% }$（+5.9pp）。在GSM8K中：在$5 0 \%$覆盖率下准确率为$8 4 . 4 \%$和$9 3 . 2 \%$（PRR=0.564）。

Table 18: Pairwise boundary dependence. MI: Freedman-Diaconis binning, 1000 permutation null (mean permuted MI ≈ 0.003 bits; max observed = 0.015 ≈ 5 $\times$ null). dcor/HSIC: 500 permutations. Weak dependence confirmed.   

<table><tr><td>Pair</td><td>Pearson r</td><td>dcor</td><td>HSIC p</td><td>MI (bits)</td><td>n</td><td>Source</td></tr><tr><td>B1-B2</td><td>0.119*</td><td>0.143*</td><td>0.020*</td><td>0.008</td><td>401</td><td>Exp. 2</td></tr><tr><td>B1-B3</td><td>-0.067</td><td>——</td><td>—</td><td>0.015</td><td>500</td><td>Exp. 5</td></tr><tr><td>B1-B4</td><td>0.054</td><td>0.072</td><td>0.252</td><td>0.006</td><td>790</td><td>Exp. 10</td></tr><tr><td>B1-B5</td><td>0.070</td><td></td><td></td><td>0.012</td><td>790</td><td>Exp. 7</td></tr><tr><td>B2B4</td><td>-0.086</td><td>0.156*</td><td>0.000*</td><td>0.003</td><td>401</td><td>Exp. 10</td></tr></table>

$^ { * } p < 0 . 0 5$ . 效应大小保持较小（$| r | \leq 0 . 1 2$，dcor ≤ 0.16）；累加覆盖大致成立。

# 8 讨论

声明范围。我们有两个独特的贡献：（1）诊断性声明，即对齐导致响应同质化，从而在结构上损害基于采样的不确定性量化，这一观点得到了四个家族、两个训练链和多个稳健性检查的无标签聚类统计的支持；（2）架构性声明，即最便宜优先的正交信号级联提供了一种实用响应，这一观点得到了GSM8K上的选择性预测收益和独立性分析的支持。诊断性声明是我们主要的贡献，得到了强有力的支持；架构性贡献则是初步的，适当地界定了其范围。

(a) 单一检测器在结构上是不足的。B1 熵在 TruthfulQA 的一半类别中有效（AUC=0.658），而在另一半中则呈反向（0.395）。B5 在 B1 失效的地方精确达到 AUC=0.678。没有单一信号能够覆盖所有失效模式。(b) 弱依赖性使得级联成为可能。所有边界对：$| r | \leq 0 . 1 2$ ，MI $\leq 0 . 0 2$ 比特。合并 AUROC=0.601 超过了任何单一边界。选择性预测：在 GSM8K 上从 84.4% 提升至 93.2% ，覆盖率为 $5 0 \%$；在 B1（免费）下解决了 $5 0 \%$ 的查询。(c) 对齐代价是结构性和任务依赖的。在事实 QA 上，模型自信地错误（$d$ =0.07）；在数学上，错误产生真正不确定的推理（$d$ =0.81）。这种崩溃在样本大小、温度、聚类方法和解码策略上持续存在（实验12，15）。在单聚类问题上，任何基于采样的方法都在构造上使 SE=0。这种任务依赖的反转验证了多边界设计。(d) 词元熵是一个强大且免费的基线，但校准效果不佳。B1（0.599，免费）在所有三个规模上超越了 SE（0.548）和基于 NLI 的 SE（0.501-0.512），并与 SelfCheck（0.588）相匹配。在 79% 的嵌入单聚类子集中，B1 保持 0.593，而基于采样的方法得分 $\leq 0 . 5 0 0$ 。然而，原始校准效果较差（ECE=0.182）；Platt 缩放将 ECE 减少至 0.021（减少88%）。校准与区分的差距本身证据表明 RLHF 将熵压缩至一个狭窄范围，无论正确性如何 [Guo et al., 2017]。LogTokU 和 PRO 是级联中兼容的 B1 升级。熵的跨任务价值：在 GSM8K 上，响应长度主导选择性预测（AUROC=0.849 对比于熵 0.724；添加熵至长度时，$\Delta { = } + 0 . 0 0 2$ ）。然而，长度是任务特定的：在 TruthfulQA 上，长度接近偶然，而熵提供主要信号（0.599）。在多任务部署中（事实 QA $^ +$ 数学），熵是唯一能够跨任务泛化的信号；长度是强而狭窄的代理，仅在不正确答案系统上较短（数学推理）时有用。

(e) 因果归因。对齐税通过三个层面上的汇聚证据确立。直接消融（实验13）：在相同的4位量化下，Qwen3-14B Base与Instruct的SCR分别为$1.0\%$与28.5% $(p < 10^{-6})$——归因于对齐本身，而非量化或架构，导致同质化。对两个独立训练链的阶段分解孤立出DPO作为驱动因素：Mistral/Zephyr（Base 0.0%→SFT 1.5%→DPO $4.0\%$，$\scriptstyle {p = 0.0001}$）和Llama/Tulu-3（Base 0.0%→SFT 0.0%→DPO 0.5%，$p$ = 0.008）。SFT保留了接近基础的多样性，同时教导遵循指令（NC: 9.28→8.63），确认基础多样性是“有意义的”；DPO则使这一已一致的分布崩溃。严重性依赖于配方：Zephyr的DPO产生的同质化是Tulu-3管道的8倍（4.0%对$0.5\%$），这可能与偏好数据的差异和额外的RLVR阶段有关。跨家族复制（实验14）确认了一般性：Qwen3-14B (28.5%) $\gg$ LLaMA-3B (5.5$\%$) $>$ Zephyr-DPO (4.0%) $>$ Mistral-7B (1.0%) $>$ Tulu-3 (0.5%)——跨越两个数量级。“隐形绳索”发现[Chen et al., 2025a]——RLVR在增加词元级熵的同时减少回答级熵——独立验证了这一机制。同质化不是可避免的结果，而是依赖于推荐系统的选择，以测量每次部署的SCR。

# 8.1 实际影响

对于实践者：(1) 在信任基于采样的不确定性之前，要检查响应的同质化——一个简单的诊断方法是：采样 $N { = } 10$ 个响应并计算单聚类率；如果 SCR $>$ $5 \%$，则基于采样的不确定性在该模型-任务对上是不可靠的。(2) 词元熵是一个强大的免费基线；LogTokU/PRO 可能进一步改善它。(3) 选择性预测是可部署的：GSM8K 的准确率从 $84.4 \%$ 提升到 $93.2 \%$，在 $50 \%$ 覆盖率下，仅需获取对数概率。(4) 按任务类型进行路由：对齐税是任务依赖的（$d$：0.07 对于事实问答 vs. 0.81 数学）。(5) 当对数概率不可用时（不透明的 API）：级联会优雅地降级——从 B2 开始（嵌入密度，1 次 API 调用），或者使用仅输出特征（响应长度，语言化信心）作为 B1 代理。EPR/WEPR [Chen 等，2025c] 提供了无需显式对数概率访问的概率基础替代方案。对齐税的发现本身与 API 无关：它描述了模型输出分布的性质，可以通过聚类方法检测采样响应。这个税是配方依赖的：DPO 超参数和偏好数据选择很重要——Tulu-3 的配方产生了比 Zephyr 少 8 倍的同质化（0.5% vs. $4.0 \%$ SCR），这表明对齐方法的选择对下游不确定性可靠性有直接影响。

# 8.2 局限性

(1) 对齐归因：实验13确认完整的对齐流程驱动了均质化（1.0% 对比 $28.5\%$ SCR，$p < 10^{-6}$）。在两个链的训练阶段消融实验中——Mistral/Zephyr（实验16：基础 0.0% → SFT $1.5\%$ DPO 4.0%）和Llama/Tulu-3（实验18：基础 $0.0\%$ SFT 0.0% DPO 0.5%）——确定DPO是两个链中的主要驱动因素，同时显示出配方依赖的严重性（Zephyr 4.0% 对比 Tulu-3 $0.5\%$）。性能最高的SCR系列（Qwen3-14B，$28.5\%$）缺乏公开可用的仅SFT检查点，这使得类似的阶段性分解无法进行；然而，Qwen的基础与指令消融（1.0%→28.5%）结合两个可用链中的一致SFT模式（SFT单独产生 $\leq 1.5\%$ SCR）提供了强有力的，但并非决定性证据，表明DPO是Qwen中观察到的更大效应的主要贡献者。所有模型均使用4位量化；在相同量化（均为Q4KM）下的基础与指令比较排除了量化作为混杂因素。跨量化验证（实验19：Mistral-7B上的Q4对Q8）确认在语义层面上的SCR一致（两种精度均为6.7%）；对Qwen3-4B的8位B1验证（$\Delta$ AUC=+0.009）进一步确认了量化影响最小。FP16实验将提供额外的信心。(2) 基线实现：四种在保真度逐步增加的变体：(a) SE-Jaccard（表面代理），(b) SE-Embedding（SINdex风格的余弦聚类），(c) SelfCheck（嵌入余弦，$k { = } 5$），以及(d) 基于NLI的规范SE在三个DeBERTa-v3规模上（435M/184M/70M）对200个问题的测试。所有四个层次之间的一致性——每个层次使用根本不同的相似性度量——本身就是强有力的证据，表明瓶颈在于输出均匀性，而非聚类方法。忽略了具有矛盾意识的NLI变体；在单聚类环境（$|{ \mathcal { C }}| = 1$，4079%的查询）中，所有响应在语义上是等价的，并且没有存在检测的矛盾——该变体对这些情况在结构上是无关的。6.2 $\times$ NLI模型扩展未产生AUROC改进（$\Delta$ +0.010）也确认了这一点。跨嵌入验证（实验20）排除了耦合偏差。根据单聚类等价论证（第2节），所有基于logit的替代方法（LogTokU，PRO，语义能量）在单聚类环境中产生等级等价的不确定性排序，因此我们的B1 AUROC为0.593也适用于这些方法。在多聚类环境中与官方代码库的正面交锋仍为未来工作。(3) 标签有效性：LLM-judge标签（LLaMA-3.2-3B）与TruthfulQA黄金答案模板的中等一致性 $\mathcal { \kappa }$ =0.487，$77.1\%$；附录。模型特定响应的人类标注子集将进一步加强可靠性，尽管单聚类崩溃与标签无关。Sco:3B4Bemodels，4位量化（内部量化比较排除了量化作为混杂因素；实验19确认在Q4对Q8上SCR一致）。FP16实验将提供额外的信心。HotpotQA样本量小（$_{n}$ =100）。对闭源的GPT类模型和其他领域（代码，对话）的泛化尚未确认。(5) B5使用金标准引用：推理时不可用。生产B5需要检索 $^+$ NLI，可能导致较低的AUC。(6) 指针模型：仅在TruthfulQA上进行5折交叉验证；需要保留和跨域评估。路由目标（错误预测）是一个代理。(7) GSM8K长度混杂：响应长度（AUROC=0.849）在选择性预测中优于熵（0.724）。对500个问题的逻辑回归显示，添加熵到长度并未带来增量AUROC（$\Delta$ +0.002；信号之间的$r { = } 0.53$），确认对于数学问题长度占主导。不过，在事实QA（TruthfulQA）中，响应长度接近随机，而熵提供了主要信号（0.599）——熵的价值在于跨任务的通用性，而非GSM8K特定的增益。(8) 校准：原始ECE=0.182（LLM-judge）；Platt缩放减少至0.021（改善88%）。等值回归、温度缩放和跨数据集校准迁移尚未探索。

# 8.3 未来工作

(1) 多样性保留对齐：测试基于熵的训练方法（H-DPO [Omura et al., 2024]，SPL [Hwang et al., 2025]，DivPO [Lanchantin et al., 2025]）是否能够在保持对齐质量的同时减少对齐成本；这将验证因果机制，并为从业者提供可操作的缓解措施。(2) FP16 精度的额外量化控制；Q4与Q8的验证（实验19）已经显示相同的SCR。(3) 与语义能量 [Ma et al., 2025a]、LogTokU [Ma et al., 2025b]、PRO [Chen et al., 2025c] 和语义熵探测器 [Kossen et al., 2024] 的正面比较（单次通过的隐藏状态不确定性）；单集群等价论证（第2节）在单集群状态下建立了等级等价性，但需要进行多集群的比较。(4) GPT-4类模型和更大的数据集。(5) 同调回归、温度缩放和跨数据集校准转移（Platt缩放已经将ECE降低了$8.8\%$）。(6) 检索 $^ +$ 自然语言推理（NLI）应用于B5。(7) 与选择性预测的端到端智能体集成。(8) 对分层子集（单集群与多集群）进行人工评估，以验证LLM评审标签并探究同质化与正确性之间的关系。

# 9 结论

UCBD 将“知道你不知道什么”的问题分解为五个由最便宜优先级级联协调的认知边界。跨越四个数据集、三个模型规模和四个模型家族的二十二个实验确立了三个核心发现：(1) 对齐税——对齐模型将响应压缩成单一语义簇，严重程度依赖于模型家族和配方（Qwen3-14B: 28.5%，Zephyr-DPO: 4.0%，LLaMA-3B: 5.5%，Tulu-3-DPO: 0.5%，Mistral-7B: 1.0% SCR；在聚类方法、样本大小、温度和解码策略下保持稳健），通过基于基础与指令的消融实验确认（1.0% 对比 28.5% SCR, $p < 10^{-6}$），在两个链条的训练阶段消融（实验16: Mistral/Zephyr；实验18: Llama/Tulu-3），解码消融（实验15: nucleus $p = 0.9$ 时SCR提升至 33.5%），以及量化验证（实验19: Q4 对比 Q8，SCR 一致）；(2) 不确定性强烈依赖于任务（Cohen's $d$: 事实问答 0.07 对比 数学 0.81），验证了多边界设计；(3) 较弱依赖的边界使得级联能够以较低成本匹配并行准确率，准确率在 57% 处降低，选择性预测将 GsM8K 的准确率从 84.4% 提升至 93.2%，覆盖率为 50%。我们的基于采样的比较使用代理（Jaccard）、SINdex 风格嵌入（凝聚聚类）和基于经典NLI的实现于三种模型规模（70M、435M，均≈0.51 AUROC），均确认同一发现——SINdex 风格方法揭示了对齐税比表面指标所示的更糟（79% 对比 40% SCR），而交叉嵌入验证（实验20）通过显示独立嵌入器检测到更多的同质化（92% 对比 78% SCR）排除了耦合偏差。中心贡献是对响应同质化现象及其对不确定性系统设计的影响进行表征。这些发现具有诊断意义：框架是模块化的，因此可以随着强大的检测器（PRO, LogTokU）的可用性插入。我们主要的贡献是诊断出对齐模型抑制了样本的多样性。

# 可重复性声明

所有实验均使用公开可用的模型（Qwen3-14B/4B，LLaMA-3.2-3B，Llama-3.1-8B，Tulu-3-8B，Mistral-7B，Zephyr-7B，通过 MLX/Ollama 以 4 位量化）和数据集（TruthfulQA，FreshQA，HotpotQA，GSM8K）。使用种子为 42 的贪婪解码可以确保可复现性。代码、数据和原始结果可在 https://github.com/DigitLion/ucbd-experiment 获取。总计算时间：约 12 小时，使用 Apple M4 Pro（64GB）。

# 伦理声明

UCBD的设计目的是通过在知识缺口造成危害之前检测这些缺口，从而提高AI智能体的可靠性。我们注意到，不确定性检测可能会被滥用，以选择性地仅显示高置信度（但可能有偏见的）答案。该框架应当用于标记不确定性以供人类审核，而不是默默地压制不确定的输出。

# References

Samir Abdaljalil et al. SINdex: Semantic INconsistency index for hallucination detection in LLMs. arXiv preprint arXiv:2503.05980, 2025.

Anastasios N.Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distributionfree uncertainty quantification. arXiv preprint arXiv:2107.07511, 2021.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. A general theoretical paradigm to understand learning from human feedback. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2024.

Jonathan Berant, Andrew Cai, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from questionanswer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 15331544, 2013.

Hao Chen et al. The invisible leash? why RLVR may or may not escape its origin. arXiv preprint arXiv:2507.14843, 2025a.

Lihu Chen, Gerard de Melo, Fabian M. Suchanek, and Gaël Varoquaux. Query-level uncertainty in large language models. arXiv preprint arXiv:2506.09669, 2025b.

Yixuan Chen et al. Probabilities are all you need: A probability-only approach to uncertainty estimation in large language models. arXiv preprint arXiv:2511.07694, 2025c.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning, pages 10501059, 2016.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning (ICML), 2017.

Ryo Hashimoto, Hidetaka Kamigaito, and Taro Watanabe. Decoding uncertainty: The impact of decoding strategies for uncertainty estimation in large language models. In Findings of EMNLP 2025, 2025. arXiv:2509.16696.

Xuan Huang et al. CoCoA: A minimum Bayes risk framework bridging confidence and consistency for uncertainty quantification in LLMs. In ICLR 2026, 2026. arXiv:2502.04964.

EJ Hwang, Bernease Perez, Kiran Rathod, Vipul Kumar, Sanghyun Bae, and Byung-Gon Lee. Diverse preference learning for capabilities and alignment. arXiv preprint arXiv:2511.08594, 2025.

Saurav Kadavath et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the effects of RLHF on LLM generalisation and diversity. Transactions on Machine Learning Research, 2024. arXiv:2310.06452.

Jannik Kossen, Yarin Gal, and Tom Rainforth. Semantic entropy probes: Robust and cheap hallucination detection in LLMs. arXiv preprint arXiv:2406.15927, 2024.

Lorez Kuhn, Yarin Gal, and Sebastian Farquhar. Semanticuncertainty: Linguisti invariances foruncertainty estimation in natural language generation. In International Conference on Learning Representations, 2023.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundel Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Jack Lanchantin, Shubham Toshniwal, Jason Weston, and Sainbayar Sukhbaatar. Diverse preference optimization. arXiv preprint arXiv:2501.18101, 2025.

Jixuan Leng, Chengsong Huang, Banghua Zhu, and Jiantao Huang. Taming overconfidence in LLMs: Reward calibration in RLHF. In ICLR 2025, 2025. arXiv:2410.09724.

Chen Li et al. CounterRefine: Retrieval-grounded constrained refinement for hallucination mitigation. arXiv preprint arXiv:2603.16091, 2026.

Zhen Li et al. UniCR: Unified conformal risk control for multiple uncertainty evidence in LLM uncertainty quantification. arXiv preprint arXiv:2505.10000, 2025.

Yong Lin, Hangyu Tan, Bizhe Zheng, et al. Mitigating the alignment tax of RLHF. In Proceedings of EMNLP 2024, 2024. arXiv:2309.06256.

Huan Ma, Jiadong Pan, Jing Liu, Yan Chen, Joey Tianyi Zhou, Guangyu Wang, Qinghua Hu, Hua Wu, Changqing Zhang, and Haifeng Wang. Semantic energy: Detecting LLM hallucination beyond entropy. arXiv preprint arXiv:2508.14496, 2025a.

Huan Ma et al. Estimating LLM uncertainty with evidence. arXiv preprint arXiv:2502.00290, 2025b.

Potsawee Manakul, Adian Liusie, and Mark J.F. Gales. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

Motoki Omura, Yutaka Matsuo, and Sho Katsumata. Entropy controllable direct preference optimization. arXiv preprint arXiv:2411.07595, 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:2773027744, 2022.

Amir Saeidi, Shivam Hossain, Rajan Bobbili, et al. Insights into alignment:Evaluating DPO and its variants across multiple tasks. arXiv preprint arXiv:2404.14723, 2024.

Tal Schuster et al.Semantic sel-distillation for language model uncerainty.arXiv preprint arXiv:260204577, 2026.

Melanie Sclar et al. Prompt multiplicity: Measuring consistency under paraphrased prompts. arXiv preprint arXiv:2602.00723, 2025.

Natalia Stanusch, Raziye Buse Cetin, Salvatore Romano, Miazia Schueler, and Meret Baumgartner. DSA, AIA, and LLMs: Approaches to conceptualizing and auditing moderation in LLM-based chatbots across languages and interfaces in the electoral contexts. arXiv preprint arXiv:2509.19890, 2025.

Théo Trouillon et al. Complex embeddings for simple link prediction. In Proceedings of ICML 2016, pages 20712080, 2016.

Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Clémentine Fourrier, Nathan Habib, et al. Zephyr: Direct distillation of LM alignment. arXiv preprint arXiv:2310.16944, 2023.

ArtemVazhentsev al. Token-level densiy-baseuncrtainty quantification methods for eliciting truthulnes of large language models. In Proceedings of NAACL 2025, pages 22462262, 2025. arXiv:2502.14427.

Paul Viola and Michael Jones. Rapid object detection using a boosted cascade of simple features. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2001.

Pragatheeswaran Vipulanandan et al.Semantic uncertainty quantification of hallucinations in LLMs:A quantum tensor network based method. arXiv preprint arXiv:2601.20026, 2026.

Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. FreshLLMs: Refreshing large language models with search engine augmentation. In Findings of ACL 2024, 2024. arXiv:2310.03214.

Ziyu Wan et al. ReMA: Learning to meta-think for LLMs with multi-agent reinforcement learning. arXiv preprint arXiv:2503.09501, 2025.

Yibo Wang et al. HalluGuard: Detecting hallucinations via NTK/jacobian proxies. arXiv preprint arXiv:2601.18753, 2025.

Jiancong Xiao, Ziniu Li, Xingyu Xie, Emily Getzen, Cong Fang, Qi Long, and Weijie J. Su. On the algorithmic bias of aligning large language models with RLHF: Preference colapse and beyond. Journal of the American Statistical Association, 2024. arXiv:2405.16455.   
Yasin Abbasi Yadkori, Gautam Kamath, Rishabh Agarwal, and Borja Balle. Mitigating LLM hallucinations via conformal abstention. arXiv preprint arXiv:2404.07594, 2024.   
Jenny Zhang, Aaron Yu, Sheng Chong, Anthony Sicilia, Michael Tomz, Christopher D. Manning, and Tianyi Shi. Verbalized sampling: How to mitigate mode collapse and unlock LLM diversity. arXiv preprint arXiv:2510.01171, 2025a.   
Tunyu Zhang, Haizhou Shi, Yibin Wang, et al.TokUR: Token-level uncertainty estimation for large language model reasoning. arXiv preprint arXiv:2505.11737, 2025b.   
Wei Zhang et al. Semantic energy: Detecting LLM hallucination beyond entropy. arXiv preprint arXiv:2508.14496, 2025c.   
Yujia Zhou et al. Metacognitive retrieval-augmented large language models. In Proceedings of WWW 2024, 2024. arXiv:2402.11626.   
Yongfu Zhu. Uncertainty under the curve: A sequence-level entropy area metric for reasoning LLM. arXiv preprint arXiv:2508.20384, 2025.

# A Implementation Details and Additional Analysis

Statistical summary of key comparisons. Table 19 consolidates statistical tests across al pairwise comparisons reported in the main text.

Table 19: Statistical summary of all key comparisons. Tests: DeLong (AUROC), Wilcoxon (paired distributions), bootstrap (CIs). All $p$ -values two-sided.   

<table><tr><td>Comparison</td><td>Exp</td><td>Effect</td><td>p-value</td><td>Test</td></tr><tr><td>B1 vs SE-Jaccard AUROC</td><td>12</td><td>∆=+0.051</td><td>0.040*</td><td>DeLong+Holm</td></tr><tr><td>B1 vs SE-Embed AUROC</td><td>12</td><td>∆=+0.057</td><td>0.033*</td><td>DeLong+Holm</td></tr><tr><td>B1 vs SelfCheck AUROC</td><td>12</td><td>∆=+0.011</td><td>0.65 ns</td><td>DeLong</td></tr><tr><td>B1 vs NLI-SE (large)</td><td>12</td><td>∆=+0.088</td><td>[0.419,0.594]</td><td>Bootstrap CI</td></tr><tr><td>Base vs Instruct SCR</td><td>13</td><td>1.0% vs 28.5%</td><td>&lt;10-6</td><td>Wilcoxon</td></tr><tr><td>Base→SFT (Zephyr)</td><td>16</td><td>∆NC=−0.64</td><td>0.002</td><td>Wilcoxon</td></tr><tr><td>SFT→DPO (Zephyr)</td><td>16</td><td>∆NC=−0.63</td><td>0.0001</td><td>Wilcoxon</td></tr><tr><td>Base→SFT (Tulu-3)</td><td>18</td><td>∆NC=−0.45</td><td>0.00004</td><td>Wilcoxon</td></tr><tr><td>SFT→DPO (Tulu-3)</td><td>18</td><td>∆NC=+0.29</td><td>0.008</td><td>Wilcoxon</td></tr><tr><td>Cascade vs Parallel</td><td>3</td><td>TOST equiv.</td><td>0.498</td><td>Equivalence</td></tr></table>

Per-question agreement. B1 and SE make partially independent errors (median-split, $n { = } 7 9 0$ ): 37.3% both correct, $2 8 . 7 \%$ both wrong, $1 7 . 8 \%$ B1-only, $1 6 . 1 \%$ SE-only—supporting cascade design.

Semantic entropy. We cluster $N { = } 1 0$ responses $T { = } 1 . 0$ , max 40 tokens; sensitivity to max tokens tested in Exp. 17: SCR $3 2 \% \to 8 \%$ at 200 tokens, confirming robustness) by bigram Jaccard (threshold 0.4) and embedding cosine (threshold 0.85, Qwen3-Embedding). NLI-based SE at three DeBERTa-v3 scales (435M/184M/70M) with bidirectional entailment (threshold 0.5) on 200q: AUROC=0.511/0.512/0.501—all near chance. Cluster statistics nearly identical across $6 . 2 \times$ scaling: large 4.68 mean clusters (9.0% single-rate), base 5.44 (6.0%), xsmall 5.42 $( 6 . 5 \% )$ . Jaccard: 3.58 mean, $2 8 . 5 \%$ single-rate. Bottleneck is response homogenization, not NLI quality.

Length control (factual QA). Nested logistic regression (1 df LR tests): TruthfulQA $n { = } 2 0 0$ ) $\scriptstyle p = 0 . 3 7 9$ , HotpotQA $n$ =100) $p$ =0.910, combined ( $_ { n }$ =300) $p$ =0.104. Entropy adds no significant incremental value over length on factual QA. On GSM8K, length dominates ( $r$ =0.53, AUROC 0.849 vs. 0.724). Entropy's value is cross-task generality.

Threshold sensitivity. SE-Jaccard AUROC across thresholds $\tau \in \{ 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 \}$ : 0.515, 0.538, 0.548, 0.558, 0.561; SCR monotonically decreases (59%→28.7 $\mathrm { \% }$ ) but remains ${ > } 2 8 \%$ at the strictest threshold. B1 (0.599) outperforms SE at all thresholds. Embedding clustering ( $_ { n }$ =200): SCR ranges from $9 5 . 5 \%$ (cos ≥0.70) to $1 8 . 0 \%$ (cos ${ \geq } 0 . 9 5$ ); best SE AUROC $ \mathrm { : = 0 . 6 0 9 }$ at cos ${ \ge } 0 . 8 5$ (matching B1 but at $7 1 . 5 \%$ SCR). The alignment tax persists across the entire threshold range for both methods. B1is threshold-ree and consistently competitive.

Sample size sensitivity. Single-cluster collapse: $N { = } 3$ : $4 6 . 3 \%$ , $N { = } 5$ . $4 1 . 9 \%$ , $N$ =7/10: $4 0 . 0 \%$ . SE-Jaccard AUROC: $0 . 5 4 1 / 0 . 5 4 8 / 0 . 5 4 4 / 0 . 5 4 8$ (plateaus at ${ \sim } 0 . 5 5$ ). B1 is constant at 0.599 (independent of sampling). Collapse is a property of the aligned model's response distribution, not insufficient sampling.

Temperature sensitivity. Single-cluster collapse rates across temperatures ( $N$ =5, 50 questions): $T { = } 0 . 3$ : $6 2 . 0 \%$ , $T { = } 0 . 7$ . $4 6 . 0 \%$ , $T { = } 1 . 0$ . $4 2 . 0 \%$ , $T { = } 1 . 5$ . $3 8 . 0 \%$ (mean clusters: 1.56, 1.88, 2.16, 2.42). Higher temperature reduces but does not eliminate homogenization—even at $T { = } 1 . 5$ , 38% of questions collapse to a single cluster.

SelfCheckGPT. $1 - \mathrm { m e a n } ( \mathrm { c o s } ( \mathbf { e } _ { \mathrm { g r e e d y } } , \mathbf { e } _ { \mathrm { s a m p l e } _ { i } } ) )$ ) for k=5 samples (Qwen3-Embedding).

LLM-judge. Cross-family LLaMA-3.2-3B (Ollama, $T { = } 0$ ), 790q, 4.4 min. Distribution: 54.6% correct, $4 5 . 4 \%$ incorrect. Gold validation against TruthfulQA templates (200q, $n { = } 1 7 5$ clear): $7 7 . 1 \%$ agreement, $\kappa { = } 0 . 4 8 7$ . SE AUROC nearly identical under both labels (gold: 0.524, LLM-judge: 0.533, ∆=0.009). Singlecluster collapse is label-independent.

Calibration. Raw ECE=0.182; Platt scaling (5-fold CV) reduces to 0.021 (88% improvement), Brier $0 . 2 8 5 {  } 0 . 2 4 7$ . RLHF compresses entropy into narrow range (mean=0.19, max=0.64).

Environment. Python 3.14, MLX $0 . 2 5 +$ , Ollama 0.9+. M4 Pro 64GB. \~10h total compute. Repo: https://github.com/DigitLion/ucbd-experiment.