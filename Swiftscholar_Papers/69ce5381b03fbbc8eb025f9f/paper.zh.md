# KUET在StanceNakba共享任务中的表现：StanceMoE：用于立场检测的混合专家架构

阿卜杜拉·阿尔·沙菲，Md. 米隆·伊斯兰，Sk. 伊姆兰·霍赛因，K. M. 阿兹哈鲁尔·哈桑 洪拉大学工科学校计算机科学与工程系 abdullah@iict.kuet.ac.bd，{milonislam，imran，az}@cse.kuet.ac.bd

# 摘要

Acor-eve aneeioneiuhepr posttwaseceolital mentpliatethonsor-basmodelshavaiveelatively o p csal e moatheeaptivhitctushliclmoeldivanresi pattI S BERT encoder ractor-level sance detection.Ourmodel integrates si expert modules designed to ptue c atri ilwublaivuashap areconducted on the StanceNakba 2026 Subtask A dataset, comprising 1,401 annotated English texts where the target actor is implicit in the text. StanceMoE achieves a macro-F1 score of $9 4 . 2 6 \%$ ,outperforming traditional baselines, and alternative BERT-based variants. eywords: stance detection, mixture-of-experts, context-aware gating, adaptive weightin

# 1. 引言

立场检测是指自动识别作者在文本中对特定主题、实体或命题的态度（Garg和Caragea，2024）。该任务旨在确定所表达的观点是支持、反对还中立于特定目标。立场检测的研究与常规情感分析不同，因为它需要衡量依赖于特定目标的情感表达，而同一文本在不同目标下可能表现出不同的立场（Niu等，2024）。这个任务的特点要求检测目标的意识，因为它带来了语义复杂性和计算困难，特别是在目标未被提及且立场以间接方式表达时（Küçük和Can，2020）。为了解决这些挑战，我们介绍了StanceMoE1，这是一种上下文增强的混合专家（MoE）架构，旨在进行参与者级立场检测。与依赖单一聚合表示的传统变换器方法不同，我们的方法明确地将立场建模分解为互补的专家模块，以捕获多样的语言和话语层现象。通过引入上下文感知的门控机制，该架构实现了异质立场信号的自适应和输入敏感融合。通过在StanceNakba 2026共享任务数据集（Aldous等，2026）上进行全面实验和详细的消融分析，我们证明了明确建模这些多样模式可以实现更稳健和细致的参与者级立场检测，并在比赛中获得第三名。

# 相关工作

姿态检测领域经历了从基于规则的系统（Küçük 和 Can，2020）到经典监督系统（Alturayeif 等，2023）的转型，这些系统依赖于手动创建的特征。深度学习方法，如卷积神经网络（CNN）和具有注意力机制的长短期记忆网络（LSTM），使得模型能够自动提取特征并更好地捕捉目标特定的姿态信息（Gera 和 Neal，2025）。目前的Transformer架构使用BERT（Garg 和 Caragea，2024）和大型语言模型（LLM）（Pangtey 等，2025）通过其理解上下文的能力和少量学习的能力，提供了出色的结果。

# 3. 提议的 StanceMoE 架构

我们提出了一种基于微调BERT编码器的上下文增强混合专家架构，用于态度检测。整体框架包括三个组成部分：（i）上下文编码器，（ii）六个并行专家模块，用于捕捉互补的语言信号，以及（iii）一个上下文感知的门控与融合机制。该架构如图1所示。

![](images/1.jpg)  

Figure 1: Proposed Mixture-of-Experts architecture for actor-level stance detection.

# 3.1. 上下文编码器

根据输入序列，我们使用微调的 BERT 编码器来获得上下文相关的词元表示和一个特殊的 [CLS] 嵌入。

# 3.2. 专家模块

虽然 BERT 提供了有前景的上下文理解，但立场检测常常依赖于微妙的词汇线索、对比标记和话语模式。标准的上下文嵌入池化技术可能无法持续捕捉这些异质模式。为了解决这个问题，我们引入了六个互补的专家，每个专家针对一种独特的语言现象。 均值池专家（全局取向）：该专家通过对词元表示进行平均，捕捉帖子整体的语义方向。当立场在整个句子中持续表达时效果显著。例如，“以色列有权自卫。”（亲以色列），或“巴勒斯坦人应获得建国权和平等权利。”（亲巴勒斯坦）。 最大池专家（显著的极性词元）：一些帖子包含孤立但极具指示性的词元来决定立场。例如，在亲巴勒斯坦的表述“这一持续的占领必须结束”中，词元“占领”作为主要的立场指标。类似地，在亲以色列的表述“The terrorist attacks cannot be justified”中，词元“恐怖分子”作为主要的词汇指标。最大池化选择最激活的特征维度，以强调整个序列中的重要词元。 自注意力池专家（上下文依赖的关注）：句子的特定小句是立场存在的地方。句子“尽管人道主义情况悲惨，以色列必须对安全威胁做出反应”展示了此类配置的例子。在这种表述中，一个小句呈现主要立场，而附加信息则通过句子的其余部分呈现。注意力机制在评估过程中对立场组成部分给予更多重视，因为它评估它们的重要性。 多核卷积神经网络专家（短语级模式）：地缘政治立场通常通过短语口号传达，例如“与以色列站在一起”（亲以色列）和“解放巴勒斯坦”（亲巴勒斯坦）。这些紧凑的 n-grams 表达作为有效的立场指标，尽管上下文信息有限，依然保持其功能。多核卷积捕获这种局部短语模式。 词汇线索意识专家（框架和报告信号）：中立帖子使用报告语言和疏远语言来表达内容，例如“据官员称，谈判正在进行中。”动词“声称”、“报道”、“说明”等，作为指标，表明说话者通过间接的方法呈现信息，而不是直接支持他们的陈述。该专家结合不同的方式展示预定的线索词元，以提高将中立内容与带有倾向性的帖子区分开的能力。 对比意识专家（话语转变建模）：像“但是”或“然而”这样的话语对比标记通常指示立场的细化或强调。例如，“我支持和平努力，但封锁必须结束。”或“平民伤亡是悲惨的；然而，安全反应是必要的。”在这种情况下，包含对比标记的小句通常承载更强的立场权重。因此，我们放大对比词元的表示，以更好地捕捉话语层面的立场转变。

# 3.3. 上下文感知门控与融合

我们引入了一种门控机制，该机制动态地对专家进行加权，而不是平等对待它们。最终表示通过专家输出的加权组合计算得出。这个设计实现了自适应路由。例如，强烈表达观点的帖子可能更依赖于最大池化，强调对比的语句可能会增加对对比感知专家的权重，而报告式中立文本可能会强调线索感知表示。融合后的表示随后传递给线性分类器。我们使用带标签平滑的交叉熵损失来训练模型。为了提高鲁棒性，我们在训练过程中采用分层 $K$ -折交叉验证，并在推理过程中对这些折进行加权对数平均。加权是通过对验证过程中具有更高 F1 分数的折给予更大权重来完成的。

# 4 实验设置

# 4.1. 数据集

本文中使用的数据集由StanceNakba 2026共享任务的组织者提供（Aldous等人，2026）。我们专注于子任务A，这是一个包含1,401个英文文本的演员级立场检测数据集，这些文本被标记为亲巴勒斯坦、亲以色列或中立。立场标签表示作者与文本中隐含的行为者之间的对齐关系，这需要由模型推断出来。官方数据集划分遵循70/15/15的比例，分别用于训练集、验证集和测试集。在我们的实验中，我们对结合的训练和验证集应用分层K折交叉验证（占数据的85%）。官方测试集（占15%）在训练过程中严格保持未见。

# 4.2. 基线方法

我们对现有的立场检测技术进行了广泛的实证评估，这些技术分为三大主要方法类别：基于TF-IDF特征的机器学习（ML）方法、基于监督训练的深度神经网络（DNN），以及BERT。机器学习方法包括逻辑回归（LR）（Rahman 等，2024）、多项式朴素贝叶斯（MNB）（Zannat 等，2025）、支持向量机（SVM）（Rahman 等，2024）和随机森林（RF）（Shafi 等，2025）。我们实施了几种流行的DNN架构作为基线，包括双向长短期记忆网络（BiLSTM）（Rahman 等，2024）、目标特定注意力网络（TAN）（Du 等，2017）、带有方面嵌入的门控卷积网络（GCAE）（Xue 和 Li，2018）、交叉网络（CrossNet）（Du 等，2017）。此外，我们评估了两种在BERT主干上开发的替代变体：（i）堆叠架构（Khan等，2025），其中专家模块按顺序应用，以及（ii）特征融合模型（Lee等，2021），其中专家输出在没有自适应加权的情况下进行融合。

# 5. 实验结果

# 5.1. 行为者级别立场检测

表1展示了所有模型在参与者级别立场检测中的比较性能。在传统基准中，随机森林（RF）取得了最佳结果，F1分数为$84.19\%$，这表明集成树模型的表现优于其他方法。在神经模型中，GCAE的表现最高，F1分数为$87.62\%$，这表明基于注意力的架构在建模立场方面表现良好。使用上下文嵌入导致了更好的性能结果。BERT的F1分数为$89.86\%$，而基于专家的架构，如堆叠（Stacked）模型（$91.83\%$）和融合（Fusion）模型（$91.17\%$）显示出良好的结果，但它们的改进仍然在固定的专家集成边界内。所提出的StanceMoE以F1分数$94.26\%$实现了最佳性能，超越了所有其他比较案例。此外，我们团队在竞赛中获得了第三名。该系统显示出性能提升，得益于其自适应专家加权系统，该系统使用了一个门控机制。所提出的系统在交叉验证中也表现出稳定的性能，方差较低（约$\pm 1.1\%$）。

# 5.2. 消融研究

表2展示了消融分析，以检验提出的StanceMoE架构中每个专家的贡献。去除任何专家通常会导致整体性能下降，确认了专家们捕捉到互补的态度信号。当去除自注意力或对比专家时，性能下降最为明显，表明它们在建模上下文依赖和话语层面态度变化方面的重要性。排除平均池化或最大池化专家也会降低性能，这表明突出的词汇线索和全局上下文信号在态度识别中的重要性。

Table 1: Performance comparison of baseline models and the proposed StanceMoE on the held-out test set. "K-fold $( m e a n \pm s t d ) ^ { \prime }$ reports the mean and standard deviation of test performance across fold-specific models, while "weighted logit ensemble" is the final ensemble prediction obtained via logit averaging. Here, Acc $=$ Accuracy, Pre $=$ Precision, Rec $=$ Recall, and $\mathsf { F } 1 = \mathsf { F } 1$ -Score.   

<table><tr><td rowspan="2">Methods</td><td colspan="4">K-fold (mean±std)</td><td colspan="4">Weighted Logit Ensemble</td></tr><tr><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>LR</td><td>80.91±1.79</td><td>80.90±1.78</td><td>80.93±1.78</td><td>80.86±1.78</td><td>81.28</td><td>81.25</td><td>81.33</td><td>81.22</td></tr><tr><td>MNB</td><td>77.25±1.64</td><td>75.88±2.02</td><td>78.49±1.65</td><td>77.20±1.65</td><td>77.38</td><td>76.02</td><td>78.63</td><td>77.33</td></tr><tr><td>SVM</td><td>83.25±1.71</td><td>84.73±1.68</td><td>83.26±1.71</td><td>83.27±1.72</td><td>83.37</td><td>84.94</td><td>83.35</td><td>83.43</td></tr><tr><td>RF</td><td>84.05±1.83</td><td>84.73±1.73</td><td>84.05±1.82</td><td>84.05±1.90</td><td>84.16</td><td>84.83</td><td>84.14</td><td>84.19</td></tr><tr><td>BiLSTM</td><td>85.63±2.99</td><td>85.87±3.00</td><td>85.63±2.99</td><td>85.51±3.07</td><td>85.74</td><td>85.98</td><td>85.74</td><td>85.72</td></tr><tr><td>TAN</td><td>85.99±3.10</td><td>86.01±2.81</td><td>86.36±2.08</td><td>85.91±2.15</td><td>86.13</td><td>86.10</td><td>86.43</td><td>86.03</td></tr><tr><td>GCAE</td><td>87.69±2.85</td><td>87.98±2.92</td><td>87.48±2.16</td><td>87.49±2.78</td><td>87.82</td><td>88.10</td><td>87.53</td><td>87.62</td></tr><tr><td>CrossNet</td><td>85.13±2.28</td><td>85.55±1.89</td><td>84.90±1.90</td><td>84.77±2.25</td><td>85.29</td><td>85.65</td><td>84.97</td><td>84.88</td></tr><tr><td>BERT</td><td>89.77±2.35</td><td>90.04±2.30</td><td>89.77±2.31</td><td>89.61±2.29</td><td>90.05</td><td>90.28</td><td>90.03</td><td>89.86</td></tr><tr><td>Stacked</td><td>91.61±2.46</td><td>91.73±2.43</td><td>91.62±2.44</td><td>91.60±2.47</td><td>91.94</td><td>92.14</td><td>91.93</td><td>91.83</td></tr><tr><td>Fusion</td><td>91.03±2.26</td><td>91.20±2.29</td><td>91.03±2.24</td><td>91.02±2.26</td><td>91.18</td><td>91.45</td><td>91.17</td><td>91.17</td></tr><tr><td>StanceMoE</td><td>94.09±1.11</td><td>94.18±1.12</td><td>94.08±1.12</td><td>94.03±1.12</td><td>94.31</td><td>94.45</td><td>94.31</td><td>94.26</td></tr></table>

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Acc</td><td colspan="3">Pro-Palestine</td><td colspan="3">Pro-Israel</td><td colspan="3">Neutral</td></tr><tr><td>Pre</td><td>Rec</td><td>F1</td><td>Pre</td><td>Rec</td><td>F1</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>w/o Mean</td><td>92.89</td><td>89.47</td><td>95.75</td><td>92.52</td><td>94.52</td><td>98.57</td><td>96.50</td><td>95.16</td><td>84.29</td><td>89.39</td></tr><tr><td>w/o Max</td><td>93.36</td><td>91.89</td><td>95.77</td><td>93.79</td><td>92.00</td><td>98.57</td><td>95.17</td><td>96.77</td><td>85.71</td><td>90.91</td></tr><tr><td>w/o Self-Attention</td><td>91.94</td><td>89.33</td><td>94.37</td><td>91.78</td><td>93.24</td><td>98.57</td><td>95.83</td><td>93.55</td><td>82.86</td><td>87.88</td></tr><tr><td>w/o CNN</td><td>93.36</td><td>92.96</td><td>92.96</td><td>92.96</td><td>93.33</td><td>99.26</td><td>96.55</td><td>93.85</td><td>87.14</td><td>90.37</td></tr><tr><td>w/o Lexical-cue</td><td>93.36</td><td>90.54</td><td>94.37</td><td>92.41</td><td>95.83</td><td>98.57</td><td>97.18</td><td>95.08</td><td>82.86</td><td>88.55</td></tr><tr><td>w/o Contrastive</td><td>91.94</td><td>90.41</td><td>92.96</td><td>91.67</td><td>90.91</td><td>99.32</td><td>95.24</td><td>93.85</td><td>87.14</td><td>90.37</td></tr><tr><td>StanceMoE</td><td>94.31</td><td>94.37</td><td>94.37</td><td>94.37</td><td>92.11</td><td>100</td><td>95.89</td><td>96.88</td><td>88.57</td><td>92.54</td></tr></table>

表 : 使用加权对数集成的 StanceMo 分类消融研究，展示了移除各个专家模块的效果。词汇提示专家特别有利于亲巴勒斯坦类别，而对比专家改善了对立场转换和论证结构的识别。尽管某些变体在各个类别中保持了竞争力的分数，但通常表现出各类别间的平衡性降低，尤其是在中立类别中。总体而言，完整的 StanceMoE 模型在所有类别中实现了最佳且最平衡的性能，证明了自适应专家整合的有效性。

# 6. 结论

在本文中，我们介绍了StanceMoE，一种增强上下文的混合专家（MoE）架构，用于立场检测。考虑到统一变换器表征在捕捉异构语言现象方面的局限性，我们的方法通过专家模块显式建模互补的立场指示信号，并通过上下文感知的门控机制将它们集成。这一设计能够自适应地建模在敏感语境中表征立场表达的词汇、语义和话语级别模式。大量实验表明，StanceMoE显著优于传统基线模型。消融分析进一步确认了各个专家的互补贡献以及自适应专家融合对于稳健检测的重要性。总体发现强调了显式解耦和动态集成多样语言信号在立场检测中的有效性。

# 参考文献

Kholoud Khalil Aldous、Md Rafiul Biswas、Mabrouka Bessghaier、Shimaa Ibrahim、Kais Attia 和 Wajdi Zaghouani。2026年。StanceNakba共享任务：公共话语中的行为者和主题感知立场检测。载于第15届国际语言资源与评估会议论文集（LREC'26），西班牙帕尔马。Izzat Alsmadi、Iyad Alazzam、Mohammad Al-Ramahi 和 Mohammad Zarour。2024年。在假新闻背景下的立场检测——一种新方法。《未来互联网》，16(10)：364。Nora Alturayeif、Hamzah Luqman 和 Moataz Ahmed。2023年。关于立场检测及其应用的机器学习技术的系统评估。《神经计算与应用》，35(7)：51135144。Michael Burnham。2025年。立场检测：分类文本中政治信念的实用指南。《政治科学研究与方法》，13(3)：611628。Jiachen Du、Ruifeng Xu、Yulan He 和 Lin Gui。2017年。基于目标特定神经注意网络的立场分类。载于2017年第26届国际人工智能联合会议（IJCAI），第39883994页。Krishna Garg 和 Cornelia Caragea。2024年。Stanceformer：面向立场检测的目标感知变换器。载于计算语言学协会研究成果：EMNLP 2024，第49694984页。Parush Gera 和 Tempestt Neal。2025年。立场检测中的深度学习：一项综述。《ACM计算调查》，58(1)：137。Zhijiang Guo、Michael Schlichtkrull 和 Andreas Vlachos。2022年。关于自动化事实检查的调查。《计算语言学协会交易》，10：178206。Lal Khan、Atika Qazi、Hsien-Tsung Chang、Mousa Alhajlah 和 Awais Mahmood。2025年。增强乌尔都语情感分析：基于注意力的堆叠CNN-Bi-LSTM深度神经网络与多语言BERT。《复杂与智能系统》，11(1)：10。Dilek Küçük 和 Fazli Can。2020年。立场检测：一项综述。《ACM计算调查》，53(1)：137。Sanghyun Lee、David K Han 和 Hanseok Ko。2021年。结合BERT的多模态情感识别融合分析。IEEE Access, 9: 94557-94572。Junxia Ma、Changjiang Wang、Lu Rong、Bo Wang 和 Yaoli Xu。2025年。探索多智能体辩论用于零样本立场检测：一种新方法。《应用科学》，15(9)：4612。Fuqiang Niu、Min Yang、Ang Li、Baoquan Zhang、Xiaojiang Peng 和 Bowen Zhang。2024年。一个针对对话立场检测的挑战数据集和有效模型。载于2024年联合国际计算语言学会议、语言资源与评估会议（LREC-COLING 2024）论文集，第122-132页。Lata Pangtey、Anukriti Bhatnagar、Shubhi Bansal、Shahid Shafi Dar 和 Nagendra Kumar。2025年。大型语言模型与立场检测的结合：任务、方法、应用、挑战与未来方向的综述。arXiv:2505.08464。Arifur Rahman、Shahriar Parvej、Kazi Saeed Alam 和 HM Abdul Fattah。2024年。优化短信垃圾邮件检测：混合投票集成与分层交叉验证的比较分析。载于2024年第五届国际数据智能与认知信息学会议（ICDICl），第1030-1035页。IEEE。Abdullah Al Shafi、Rowzatul Zannat、Abdul Muntakim 和 Mahmudul Hasan。2025年。一个结构化的疾病-症状关联数据集以提高诊断准确性。arXiv:2506.13610。Wei Xue 和 Tao Li。2018年。基于门控卷积网络的方面情感分析。载于第56届计算语言学协会年会论文集（第一卷：长篇论文），第25142523页。Ruichao Yang、Jing Ma、Wei Gao 和 Hongzhan Lin。2025年。结合社交背景信息的LIm增强型多实例学习，用于联立谣言和立场检测。《ACM智能系统与技术交易》，16(3)：1-27。Rowzatul Zannat、Abdullah Al Shafi 和 Abdul Muntakim。2025年。弥合孟加拉国医疗保健的差距：基于症状-疾病数据集的机器学习疾病预测。载于2025年电气、计算机与通信工程国际会议（ECCE），第16页。IEEE。Bowen Zhang、Genan Dai、Fuqiang Niu、Nan Yin、Xiaomao Fan、Senzhang Wang、Xiaochun Cao 和 Hu Huang。2024年。社交媒体上立场检测的综述：新方向和视角。arXiv:2409.15690。Zhengyuan Zhu、Zeyu Zhang、Haiqi Zhang 和 Chengkai Li。2025年。RATSD：从社交媒体帖子中获取增强真实性的立场检测，针对事实主张。载于计算语言学协会研究成果：NAACL 2025，第33663381页。

# A. 立场检测的必要性

态度检测的能力变得越来越重要，因为在线话语的增长。应用领域广泛，包括选举趋势分析（Burnham, 2025）、在线辩论中论证互动的研究（Ma et al., 2025）、社交媒体讨论的监测（Zhang et al., 2024）、谣言评估和验证（Yang et al., 2025）、自动化事实核查系统（Guo et al., 2022）、假新闻识别（Alsmadi et al., 2024）以及基于态度的信息检索（Zhu et al., 2025）。在这些背景中，理解对某一主张或参与者的方向性意见比简单测量情感更具信息价值（Garg and Caragea, 2024）。

# B. 专家模块的操作细节

给定输入序列 $X = \{ x _ { 1 } , \ldots , x _ { T } \}$，预训练的 BERT 编码器获取上下文相关的词元表示：$H = \{ h _ { 1 } , h _ { 2 } , . . . , h _ { T } \}$，其中 $h _ { i } \in \mathbb { R } ^ { d }$，$d$ 表示隐层维度，$T$ 是标记化输入序列的长度。均值池化专家：该专家捕捉全局语义信息，如 (1) 所示。

$$
e _ { 1 } = W _ { 1 } \left( \frac { 1 } { T } \sum _ { i = 1 } ^ { T } h _ { i } \right)
$$

最大池化专家：为了提取显著的词元级特征，我们利用最大池化专家，如公式 (2) 所示。使用核大小 $k \in \{ 2 , 3 , 4 , 5 \}$ 的解决方案，如公式 (5) 所示。

$$
e _ { 4 } = W _ { 4 } \left( { \mathsf { C o n c a t } } ( \mathsf { M e a n P o o l } ( \mathsf { R e L U } ( \mathsf { C o n v } _ { k } ( H ) ) ) ) \right)
$$

词汇线索意识专家：我们定义了一组态度指示性词汇线索，并针对其位置生成一个二进制掩码 $C$ 。线索意识表示的计算方法如公式 (6) 所示。

$$
e _ { 5 } = W _ { 5 } \left( { \frac { \sum _ { i \in C } h _ { i } } { | C | + \epsilon } } \right)
$$

其中，$\epsilon$ 避免除以零。对比意识专家：为了建模话语中的对比标记（例如，“但是”，“然而”），我们增强它们的上下文影响。令 $D$ 表示对比标记位置的集合。整体过程如 (8) 所示。

$$
\begin{array} { r } { \tilde { h } _ { i } = \left\{ \begin{array} { l l } { 3 h _ { i } } & { \mathsf { i f } ~ i \in D , } \\ { h _ { i } } & { \mathsf { o t h e r w i s e } , } \end{array} \right. } \end{array}
$$

自注意力池化专家：我们引入一个可训练的注意力向量 $v$ 来计算词元的重要性，如（3）所述。

$$
e _ { 6 } = W _ { 6 } \left( \frac { \sum _ { i = 1 } ^ { T } \widetilde { h } _ { i } } { | D | + \epsilon } \right)
$$

$$
e _ { 2 } = W _ { 2 } \left( \underset { i = 1 } { \overset { T } { \operatorname* { m a x } } } h _ { i } \right)
$$

$$
\alpha _ { i } = \frac { \mathsf { e x p } ( \mathsf { t a n h } ( h _ { i } ^ { \top } v ) ) } { \sum _ { j = 1 } ^ { T } \mathsf { e x p } ( \mathsf { t a n h } ( h _ { j } ^ { \top } v ) ) }
$$

# C. 上下文感知门控的操作细节

$$
e _ { 3 } = W _ { 3 } \left( \sum _ { i = 1 } ^ { T } \alpha _ { i } h _ { i } \right)
$$

多核卷积神经网络专家：为了建模局部n-gram模式，我们应用一维卷积。门控网络将来自BERT编码器的[CLS]表示 $h _ { \mathsf { C l s } } \in \mathbb { R } ^ { d }$ 作为输入。一个可学习的线性层 $W _ { g } ~ \in ~ \mathbb { R } ^ { N \times d }$ 及偏置 $b _ { g } \in \mathbb { R } ^ { N }$ 计算 $N = 6$ 个专家的logits，这些logits通过softmax进行归一化，以生成门控权重，如式（9）所示。

$$
g = { \mathsf { S o f t m a x } } ( W _ { g } h _ { \mathsf { c l s } } + b _ { g } )
$$

其中 $g \in \mathbb{R}^{K}$，且对于所有 $i$ 都有 $g_{i} \geq 0$，并且 $\textstyle \sum_{i=1}^{K} g_{i} = 1$。每个门控权重 $g_{i}$ 确定了相应专家输出 $e_{i}$ 对最终表示的贡献，该表示通过使用公式 (10) 的加权聚合获得。

$$
h _ { \mathsf { m o e } } = \sum _ { i = 1 } ^ { K } g _ { i } e _ { i }
$$

融合表示 $h _ { \mathsf { m o e } }$ 然后通过具有可学习权重 $W _ { o }$ 和偏置 $b _ { o }$ 的任务特定线性层生成模型预测，如 (11) 所示。

$$
\hat { y } = \mathsf { s o f t m a x } ( W _ { o } h _ { \mathsf { m o e } } + b _ { o } )
$$

其中 $\hat { y }$ 表示预测的类别概率。参数 $W _ { g }$、$b _ { g }$、$W _ { o }$ 和 $b _ { o }$ 全部是可学习的，并与网络其余部分通过反向传播共同优化，从而使模型能够学习如何组合专家以及如何做出准确预测。

![](images/2.jpg)  

Figure 2: Confusion matrix using the proposed StanceMoE architecture.   

Table 3: Training hyperparameters used in Stance-MoE.

（1）et al.（2024）用于创建双向上下文模型，能够在不需要目标信息的情况下预测立场。（2）GCAE（Xue 和 Li，2018）提供了一种卷积系统，采用门控机制来屏蔽非目标特征。（3）TAN（Du 等，2017）使用增强注意力的双向长短期记忆网络（BiLSTM）来识别重要的上下文细节，以帮助确定立场。（4）Cross-Net（Du 等，2017）通过基于方面的注意力层增强注意力处理，该层在分类过程之前操作，以增强基于目标的特征提取。

# G. 评估指标

我们使用宏观F1分数作为主要指标来评估模型性能。此外，还报告准确率、宏观精度和宏观召回率。

# D. 超参数

表3展示了实验中使用的超参数。

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Max sequence length</td><td>128</td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Number of epochs Learning rate</td><td>10 5 × 10-5</td></tr><tr><td>Number of splits (k)</td><td>10</td></tr><tr><td>Label smoothing factor</td><td>0.25</td></tr><tr><td>Random seed</td><td>42</td></tr></table>

# E. 基线机器学习方法

我们利用四种流行的机器学习模型作为基线系统：(1) 逻辑回归 (LR)：一种线性模型，通过加权特征预测类别概率，适用于高维文本数据。(2) 多项式朴素贝叶斯 (MNB)：一种基于词频分布的概率模型，假设特征独立，非常适合文本分类。(3) 支持向量机 (SVM)：一种基于边界的分类器，通过找到最佳超平面来分隔类别，在稀疏文本特征空间中具有鲁棒性。(4) 随机森林 (RF)：一种集成决策树的模型，通过袋装法和特征随机性降低过拟合，从而提高分类性能。

# F. 基准深度神经网络

我们使用四种流行的深度神经网络模型作为我们的基线系统：（1）双向长短期记忆网络（BiLSTM）

# H. 整体消融研究

表4展示了在测试集上的整体消融研究，而不是特定类别的研究。

# I. 混淆矩阵

图 2 显示了使用所提议的 StanceMoE 架构的混淆矩阵。

# J. 误差分析

表5中的定性错误分析揭示了几个系统性模式。(1) 模型偶尔表现出对强词汇极性线索的过度依赖。第一和第五个例子展示了宗教赞美以及人道主义语言，错误地被解读为明确的政治倾向。(2) 元话语评论有时与立场混淆。第三个例子中的文本描述了反犹太主义和社会规范，但并未显示出任何具体支持。系统利用目标相关关键词对内容进行立场预测，这很困难，因为这要求模型区分提及的实体与评估定位。(3) 复杂的多从句和对比推理依然具有挑战性。第二和第六个例子包含复杂的语篇结构，通过多层次的论证展示立场。系统设计包括专家来处理注意力和对比解码任务，但在某些情况下，门控系统未能对这些专业语篇元素提供适当的评估。(4) 像第四个例子这样非常简短和缺乏上下文的陈述本质上是模糊的。在这种情况下，模型似乎将极性与主导训练模式关联，而不是将预测建立在对目标的明确评估上。

Table 4: Overall K-fold and weighted ensemble ablation study on the test set.   

<table><tr><td rowspan="2">Methods</td><td colspan="4">K-fold (mean±std)</td><td colspan="4">Weighted Logit Ensemble</td></tr><tr><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>w/o Mean</td><td>91.75±1.76</td><td>92.02±1.54</td><td>91.74±1.77</td><td>91.65±1.86</td><td>92.89</td><td>93.05</td><td>92.88</td><td>92.80</td></tr><tr><td>w/o Max</td><td>91.52±1.23</td><td>91.83±1.06</td><td>91.51±1.23</td><td>91.43±1.30</td><td>93.36</td><td>93.56</td><td>93.35</td><td>93.29</td></tr><tr><td>w/o Self-att</td><td>91.47±1.04</td><td>91.65±0.99</td><td>91.46±1.04</td><td>91.38±1.08</td><td>91.94</td><td>92.04</td><td>91.93</td><td>91.83</td></tr><tr><td>w/o CNN</td><td>92.09±1.04</td><td>92.20±0.97</td><td>92.08±1.04</td><td>92.00±1.11</td><td>93.36</td><td>93.38</td><td>93.37</td><td>93.29</td></tr><tr><td>w/o Lexical-cue</td><td>91.70±1.84</td><td>91.83±1.84</td><td>91.70±1.84</td><td>91.67±1.83</td><td>93.36</td><td>93.41</td><td>93.36</td><td>93.32</td></tr><tr><td>w/o Contrastive</td><td>91.75±1.81</td><td>91.99±1.67</td><td>91.74±1.82</td><td>91.64±1.87</td><td>91.94</td><td>92.13</td><td>91.94</td><td>91.82</td></tr><tr><td>StanceMoE</td><td>94.09±1.11</td><td>94.18±1.12</td><td>94.08±1.12</td><td>94.03±1.12</td><td>94.31</td><td>94.45</td><td>94.31</td><td>94.26</td></tr></table>

表：StanceMoE 的具有代表性的真实误分类案例。使用不同的背景颜色以便于视觉区分。

<table><tr><td>Text</td><td>Actual</td><td>Predicted</td><td>Primary Error Source</td></tr><tr><td>This is shameful. And the Jewish people, God&#x27;s cho- sen people. May God bless and protect you.</td><td>Neutral</td><td>Pro-Israel</td><td>Religious praise misinter- preted as political stance due to strong positive lexi- cal cues.</td></tr><tr><td>Most comments are clothed with sycophancy. Nigeri- ans have a right of association, but you can stand with Gaza. The people of Gaza started this aggressive be- haviour on 7th October when they launched a coordi- nated attack against the state of Israel.</td><td>Pro- Palestine</td><td>Neutral</td><td>Complex multi-clause rea- soning; insufficient empha- sis on dominant stance clause.</td></tr><tr><td>If you are talking about the doctor not talking about Palestine. He is German living in Germany where any anti-Jewish sentiment is taken very seriously and to many people anti-Israel equals antisemitism, so it is in his better interest not to speak about it at all, not nec- essarily meaning he supports anyone.</td><td>Neutral</td><td>Pro-Israel</td><td>Confusion between topic discussion and stance ex- pression due to keyword ac- tivation.</td></tr><tr><td>Against the Islamist terrorists who hate Jews and the West.</td><td>Neutral</td><td>Pro- Palestine</td><td>Short ambiguous input; po- larity incorrectly associated with a stance.</td></tr><tr><td>As I stood up with BLM, I stand against oppression in all forms. Release Palestinian hostages before talking about anything else.</td><td>Neutral</td><td>Pro- Palestine</td><td>Humanitarian framing inter- preted as explicit alignment due to lexical cues.</td></tr><tr><td>I was raised Muslim, and I know intimately how oppres- sive the religion can be in various ways. That being said, because I understand that Palestinian gay peo- ple are being oppressed for more than their sexuality right now, and that matters too.</td><td>Pro- Palestine</td><td>Neutral</td><td>Failure to capture nuanced contrastive stance; insuffi- cient weighting of discourse signals.</td></tr></table>

总体而言，这些错误表明，尽管 Stance-MoE 对于显性和词汇极化的立场表达有效，但细致且隐含框架的 discourse 仍然是未来改进的一个具有挑战性的方向。

# K. 未来工作

未来的研究可以将这一框架扩展到明确的目标感知立场建模，或探索跨主题和跨领域的立场检测，以进一步评估所提出架构的鲁棒性和泛化能力。研究多语言适应性也可能增强其在更广泛地缘政治话语背景下的适用性。

# L. 致谢

我们衷心感谢 StanceNakba 2026 共享任务的组织者以及匿名审稿人的支持。作者承认使用 ChatGPT（OpenAI）作为语言润色、代码建议和概念结构的辅助工具。所有生成的内容均经过作者的严格审查、验证，并进行了适当的调整，作者对作品的准确性和完整性负有全部责任。