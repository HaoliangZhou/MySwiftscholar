# RETLLM：用于多模态信息检索的训练和无数据MLLM

深圳大学计算机科学与软件工程学院，苏大伟，王东生，深圳，中国

# 摘要

多模态信息检索（MMIR）因其在处理文本、图像或混合查询和候选项方面的灵活性而受到关注。近年来，多模态大型语言模型（MLLMs）的突破通过在对比微调框架下结合MLLM的知识，提升了MMIR的性能。然而，它们存在预训练不一致性的问题，并且需要大量的数据集。在本研究中，我们提出了一种新颖的框架Ret LLM，旨在以无训练和无数据的方式查询MLLM以进行MMIR。具体而言，我们将MMIR表述为一个相似度分数生成任务，并提示MLLM直接预测检索分数，采用粗到细的流程。在粗略阶段，采用top-k过滤策略为每个查询构建一个小而优质的候选池，使得MLLM能够关注语义相关的候选项。随后，在细化阶段，通过将查询和候选项同时输入MLLM来预测检索分数。值得注意的是，我们在推理过程中提出了一种视觉增强模块，以帮助MLLM重新选择被遗忘的视觉信息，从而改善检索效果。在MMIR基准上进行的大量实验表明，Ret LLM的表现超越了微调模型。消融研究进一步验证了每个组件的有效性。我们的工作表明，MLLM能够在无需任何训练的情况下实现强大的MMIR性能，突显其在简单、可扩展框架中的固有多模态推理能力。我们的代码已发布于：https://github.com/alivecat05/RETLLM。关键词— 多模态大型语言模型（MLLMs）、多模态信息检索（MMIR）、提示工程。

# 引言

多模态信息检索（MMIR）系统被期望在各种模态中进行搜索，并根据用户输入提供相关的信息片段，其中查询和候选项可以是纯图像、文本或两者的组合。这些系统在各种下游任务中扮演着至关重要的角色，如图像-文本检索、视觉问答（VQA）和检索增强生成（RAG）[1]。作为多模态表示学习中的一种开创性算法，CLIP [2] 通过将每种模态对齐到共享嵌入空间，并对图像-文本对进行对比训练，展现了在图像-文本检索中的强大性能。然而，由于其依赖于模态特定的编码器，CLIP未能覆盖更具挑战性的案例，如长文本和交错的图像-文本内容。

与此同时，最近的研究探索将多模态大语言模型（MLLMs）作为通用编码器，通过用MLLM派生的表示替换CLIP风格的嵌入[3][4][5]，将MLLMs视为通用编码器，并插入总结提示：“<query>。用一个词总结上述句子：”，其中<query>表示多模态内容。然后，应用于最后一个词元的对比损失用于微调整MLLM的参数。例如，E5-V [5] 在文本对上使用单模态对比学习训练MLLMs，在复杂的多模态检索任务中取得了强劲的表现。后续工作[6][7][8][9]通过大规模多模态数据、两阶段训练和困难负样本挖掘增强了E5-V。除了嵌入学习，最近的研究[10] 将MLLMs视为重新排序器，以细化检索结果，通常需要专门的训练策略，如噪声注入。尽管在多模态信息检索（MMIR）性能上取得了这些进展，基于训练的方法仍然存在几个局限性：1）目标不一致：自回归预训练与对比微调之间的不一致可能削弱MLLM的多模态推理能力；2）可扩展性瓶颈：对大规模多模态训练对的依赖需要高昂的收集成本和计算资源，限制了实际应用。

为了解决上述不足，我们提出了RetLLM，旨在以无训练和无数据的方式探索多模态大语言模型（MLLMs）的零-shot检索潜力。受到MLLMs通过基于字符串的数值预测在回归任务中取得的最新成功的启发，Ret LLM将检索重新表述为相似度评分预测任务，使得在不进行微调的情况下能够处理诸如长文本或组合输入等复杂查询。为了在效率和准确性之间取得平衡，Ret LLM采用了粗到细的检索管道。对于多模态查询，我们首先通过使用轻量级的基于嵌入的模型（例如，CLIP相似度）收集一个小规模且高质量的候选池。这一粗选过程过滤掉与查询语义相关性低的样本，不仅减少了检索时间，还使得MLLMs能够更专注于难处理的候选。在细选阶段，我们通过将查询和每个候选输入到多模态指令中，提示MLLMs预测查询与每个候选之间的相似度评分。最终的预测是通过选择具有最大语义评分的候选获得的。重要的是，最近的研究表明，MLLMs中的幻觉常常导致不切实际的响应。为了解决这个问题，我们开发了一种视觉增强策略，将视觉词元视为补充证据，并允许MLLMs在预测过程中重新拾起被遗忘的特征。最后，我们进一步设计了一种基于熵的决策策略，以处理在细选阶段多个候选获得相同最高相似度评分的平局情况。这使得我们的Ret LLM能够考虑混淆候选之间的不确定性评分，从而提高检索结果。我们将我们的贡献总结如下： • 我们将多模态检索任务重新表述为相似度评分生成任务，并展示了MLLMs在各种区分性任务中的强大潜力。

![](images/1.jpg)  
Fig. 1: Overview of the RetLLM framework, which integrates Top $k$ filtering, vision enhancement, and entropy-based selection for effective multimodal retrieval

• 我们介绍了 Ret LLM，这是一个无需训练和数据的框架，旨在利用多模态大语言模型（MLLMs）进行多模态信息检索（MMIR），采用粗到细的策略以实现快速而精确的检索。 • 在图像-文本检索和组合图像检索任务上的大量实验表明，Ret LLM 的有效性。它的表现超越了基于 CLIP 的基线，同时在性能上与基于训练的 MLLM 模型相当。

# 2. 方法学

我们定义一个用户查询 $q$ 和 $N$ 个候选项 $\Omega = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { N } \}$，其中每个 $q$ 和 $c _ { n }$ 可以是图像、文本或交错的图像-文本内容。在本研究中，我们专注于 top-1 检索准确率，旨在为每个 $q$ 搜索最佳目标 $c$。我们的想法非常简单：我们利用多模态大语言模型（MLLM）作为相似性评分生成器，并通过将其输入指令，提示预训练的 MLLM 生成 $q$ 和 $c _ { n }$ 的检索得分。如图 1 所示，我们的 Ret LLM 在粗到细的检索框架中执行多模态信息检索（MMIR），以平衡效率和准确性。同时，进一步开发了视觉增强和基于熵的决策机制，以提高最终检索性能。

# 2.1. 粗到细框架

通过语义相似性进行粗略选择。直观上，可以直接提示多语言模型生成查询 $q$ 与每个候选 $c \in \Omega$ 之间的相似性得分。不幸的是，这种简单的尝试需要 $N$ 次多语言模型查询，导致耗时严重。通常，只有少量的样本在 $\Omega$ 中充当 $q$ 的有价值候选，因此我们引入粗略选择模块，为每个 $q$ 形成一个小规模高质量的候选池 $\mathcal { C }$。在数学上，我们根据候选与 $q$ 之间的语义相似性选择有价值的候选：

$$
\mathcal { C } = \mathrm { T o p K } ( s ) , \quad s _ { i } = \frac { { \bf q } ^ { \top } { \bf c } _ { i } } { | | { \bf q } | | | \bf { c } | | } , \quad i = 1 , 2 , \ldots , N ,
$$

其中 $\mathbf{q}$ 和 c 分别表示查询和候选的特征。TopK 表示我们从 $\Omega$ 中选择具有最大相似度得分 $s$ 的 $k$ 个候选，以形成候选池 $\mathcal{C}$，作为后续细粒度重排序阶段的输入。与多语言大模型的细粒度选择。正如上面所讨论的，候选池 $\mathcal{C}$ 包含与 $q$ 具有高语义相关性的困难样本，而基于嵌入的模型（如 CLIP）未能正确区分它们。基于近期多语言大模型出色的推理和生成能力，我们将检索任务视为相似度得分预测问题。具体而言，与之前在嵌入空间计算检索得分的模型不同，我们在这里期望多语言大模型直接生成该得分：

$$
f _ { i } = \mathbf { M L L M } ( q , c _ { i } ) , \quad c _ { i } \in \mathcal { C } .
$$

在此，指令模板（如图 1 所示）旨在将查询 $q$ 和其候选 $c$ 作为输入，并提示大语言模型（MLLMs）预测它们之间的语义相似性得分。由于候选集 $\mathcal{C}$ 的规模较小，这一细粒度选择过程将大语言模型的查询时间从 $N$ 减少到 $K$，使得大语言模型能够更加专注于难度较大的候选项。需要注意的是，我们的粗到细选择框架可以被视为一种混合算法，它结合了基于嵌入模型的表示学习和大语言模型的多模态推理能力。Ret LLM 首先根据嵌入空间中的语义特征检索 $K$ 个高质量候选项，然后探索在大语言模型中编码的预训练知识，以理解查询与难候选项之间的细粒度差异。前者阶段提供快速推断速度但数值不准确，而后者则提供细粒度的相似性得分但推断速度较慢。所提出的框架在粗到细的策略下结合了二者的优势，有效平衡了效率和准确性。

# 2.2. 视觉增强与基于熵的决策制定

已有研究[12][13]表明，由于细粒度模态失衡，多模态大语言模型在生成过程中常常因丢失细粒度视觉细节而出现幻觉。受到前人研究[11, 14, 15]的启发，我们在Transformer模块的前馈网络（FFN）中进行视觉重注入。具体而言，我们首先将标准的FFN重构为一个键值检索过程。令 $\textbf{x} \in \mathbb{R}^{d}$ 为FFN的输入隐状态，其常规形式定义为：和Urban1K）以及组合基准（SugarCrepe）。最佳结果以粗体显示。

<table><tr><td rowspan="3">Method</td><td colspan="4">Short Caption Retrieval</td><td colspan="4">Long Caption Retrieval</td><td colspan="3">Compositional Retrieval</td></tr><tr><td colspan="2">Flickr30K</td><td colspan="2">COCO</td><td colspan="2">ShareGPT4V</td><td colspan="2">Urban1K</td><td rowspan="2">Replace</td><td rowspan="2">Swap</td><td rowspan="2">Add</td></tr><tr><td>$qi  c}$</td><td>$q frxc{ }$</td><td>qi → ct</td><td>$q frac{xc{ }$</td><td>$qi  c}$</td><td>qt → ci</td><td>qi → ct qt → ci</td><td></td></tr><tr><td>CLIP(ViT-L)</td><td>87.2</td><td>67.3</td><td>58.1</td><td>37.0</td><td>81.8</td><td>84.0</td><td>47.0</td><td>47.0</td><td>79.5</td><td>62.7</td><td>74.9</td></tr><tr><td>EEVA-CLIP</td><td>93.9</td><td>78.8</td><td>68.8</td><td>51.1</td><td>93.1</td><td>81.2</td><td>80.</td><td>77.</td><td>885.9</td><td>70.3</td><td>86.7</td></tr><tr><td>E5V</td><td>88.7</td><td>79.5</td><td>62.0</td><td>52.0</td><td>85.1</td><td>82.1</td><td>88.9</td><td>83.2</td><td>86.3</td><td>67.6</td><td>6.9</td></tr><tr><td>VLM2Vec</td><td>90.6</td><td>76.0</td><td>6.6</td><td>46.</td><td>889.8</td><td>86.9</td><td>91.</td><td>82.4</td><td>8.55</td><td>64.8</td><td>94.2</td></tr><tr><td>UniME</td><td>93.4</td><td>81.9</td><td>70.1</td><td>53.7</td><td>97.2</td><td>93.9</td><td>95.9</td><td>95.2</td><td>89.0</td><td>7.6</td><td>94.4</td></tr><tr><td>RetLLM</td><td>94.5</td><td>82.0</td><td>70.4</td><td>54.1</td><td>97.6</td><td>94.2</td><td>88.9</td><td>78.6</td><td>94.8</td><td>92.7</td><td>96.2</td></tr></table>

$$
\mathrm { F F N } ( \mathbf { x } ) = \phi ( \mathbf { x } \mathbf { W _ { 1 } } ) \mathbf { W _ { 2 } } ^ { \top } ,
$$

其中 $\phi$ 是激活函数（例如 ReLU 或 SiLU），且 $\mathbf { W _ { 1 } } , \mathbf { W _ { 2 } } \in \mathbb { R } ^ { d \times D }$ 且 $D = 4 d$。我们可以将 $\mathbf { W _ { 1 } }$ 和 $\mathbf { W _ { 2 } }$ 重写为：

$$
\mathbf { W _ { 1 } } = ( \mathbf { k _ { 1 } } , \mathbf { k _ { 2 } } , \ldots , \mathbf { k _ { D } } ) , \quad \mathbf { W _ { 2 } } = ( \mathbf { v _ { 1 } } , \mathbf { v _ { 2 } } , \ldots , \mathbf { v _ { D } } ) ,
$$

其中 $k _ { i }, v _ { i } \in \mathbb { R } ^ { d }$ 表示第 $_ { i }$ 个键和值向量。因此，前馈神经网络（FFN）可以重新表述为

$$
\mathrm { F F N } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { D } \phi ( \left. \mathbf { x } , \mathbf { k } _ { \mathbf { i } } \right. ) \cdot \mathbf { v _ { i } } .
$$

该公式揭示了前馈神经网络（FFN）作为记忆模块的作用，使用 $_ x$ 作为查询来检索相关值。直观上，我们引入视觉词元集 $Z _ { v } = \{ z _ { v , 1 , . . . , z _ { v , N _ { v } } } \}$ 作为补充视觉知识。当在层 $l$ 中激活视觉重注入时，我们将视觉词元视为新的键值条目并计算修正项：

$$
\Delta ( \mathbf { x } \propto \mathbf { Z } _ { \mathbf { v } } ) = \sum _ { j = 1 } ^ { N _ { v } } \phi ( \langle \mathbf { x } , \mathbf { z } _ { \mathbf { v } , \mathbf { j } } \rangle ) \cdot \mathbf { z } _ { \mathbf { v } , \mathbf { j } } .
$$

最后，原始前馈神经网络的输出与视觉校正结果进行了融合：

$$
\mathrm { F F N } ^ { ( l ) } ( \mathbf { x } \propto \mathbf { Z } _ { \mathbf { v } } ) = \alpha \Delta ( \mathbf { x } \propto \mathbf { Z } _ { \mathbf { v } } ) + ( 1 - \alpha ) \mathrm { F F N } ( \mathbf { x } ) ,
$$

其中 $\alpha \in [ 0 , 1 ]$ 是注入比率，$x \propto Z _ { v }$ 表示将视觉重新注入从 $x$ 到视觉特征 $Z _ { v }$。该操作将视觉证据重新注入到中间层，而不会引入额外的可训练参数，从而显著增强模型对输入视觉内容的忠诚度。另一个挑战来自 MLLMs 生成的相似度分数。我们在实证中发现，多个候选对象在公式 2 中可能会获得相同的语义分数，导致排名的模糊性。为了解决这种平局，我们引入了一种基于熵的置信度校准策略。具体而言，我们设计了一种基于置信度的指令来测量查询和候选 $( q , c )$ 对的模型不确定性：“<query>, $<$ candidate>. 该候选是否与查询匹配，是真的还是假的。”随后，获得的不确定性分数是最后一个 token 输出 logits 的归一化熵：

$$
H _ { i } = - \sum _ { v = 1 } ^ { V } p _ { v } \log p _ { v } ,
$$

其中 $V$ 是词汇表大小，$p _ { v }$ 是模型输出分布中 token $v$ 的 softmax 概率。较低的熵 $H _ { i }$ 表示模型更高的确定性。在共享相同语义分数的候选中，我们选择熵最小的那个：

$$
C ^ { * } = \arg \operatorname* { m i n } _ { C _ { i } \in \mathcal { P } } H _ { i } ,
$$

其中 $\mathcal { P }$ 表示具有相同最高评分的候选集。该基于置信度的选择策略有助于在语义差异微妙时优化排名，从而提高最终检索结果的可靠性。

# 3. 实验

# 3.1. 数据集和基线

为了全面评估我们提出的 Ret LLM 的有效性，我们在六个基准上以零样本的方式进行了评估：Flickr30K、COCO、ShareGPT4V、Urban1K、SugarCrepe 和 MMEB 基准。作为比较，我们包含了几个强有力的基线模型：CLIP、EVA-CLIP、E5-V、VLM2Vec、SigLIP 和 UniME，所有模型均在相同的零样本协议下进行评估。

# 3.2. 实现细节与评估指标

我们的主要实验采用 Qwen2.5-VL-7B 作为多模态语言模型（MLLM），并使用 CLIP-ViT-I $\mathcal { I } 1 4 @ 3 3 6 \mathrm { p x }$ 进行粗略阶段检索，将前五个候选项传递到细化阶段。主要改进包括：1) 视觉再注入 $( \alpha { = } 0 . 3 )$ 以减轻幻觉现象；2) 基于熵的决策方法用于处理模糊的最高得分。为了进行消融实验，我们改变 CLIP 主干网络（ViT-B, Long-CLIP-L）和多模态语言模型（Phi-3.5-V, Qwen2-VL）。所有实验均为零样本学习。主要指标：召回率 $@ 1$，测量查询中正确目标排名第一的情况。在 MMEB 上，我们报告所有元任务的平均精度 $@ 1$。

# .3. 结果分析

Ret LLM在所有基准测试中实现了强大的零-shot性能，且无需任何训练或微调。如表1所示，其始终优于零-shot基线（例如，CLIP，EVA-CLIP）和MLLM检索模型，如E5-V和VLM2Vec。例如，在Flickr30K $\begin{array} { l } { { ( q ^ { i } \ \to \ c ^ { t } , } } \end{array}$上，Ret LLM达到了 $94.5\%$ $\mathbb { R } \ @ 1$，超越了E5-V $( 88.7\% )$ 和VLM2Vec $( 90.6\% )$。在ShareGPT4V $( q ^ { i } \to c ^ { i } )$上，Ret LLM实现了 $94.2\%$ $\mathbf { R } \ @ 1$，优于VLM2Vec $( 86.9\% )$。在SugarCrepe "Add"上，Ret LLM达到了 $96.2\%$，比VLM2Vec $( 94.2\% )$ 提高了 $2\%$，展现了出色的零-shot推理能力。在MMEB（表2）上，Ret LLM的总体精度 $@ 1$ 达到 $54.2\%$，比最强零-shot基线UniME 提高了 $12.6\%$。它在检索 $( 62.4\% )$、分类 $( 60.2\% )$ 和VQA $( 27.8\% )$方面表现优异，证明了其在零-shot场景中的鲁棒性。这些结果证实，通过我们的粗到细管道、视觉增强和基于熵的选择，Ret LLM作为一个强大的零-shot检索器表现出色。报告的分数是对应数据集的零-shot方式下的平均精度 $@ 1$。最佳结果以粗体标出。

<table><tr><td>Models</td><td>#Parameters</td><td colspan="4">Per Meta-Task Score</td><td colspan="3">Average Score</td></tr><tr><td></td><td></td><td>Classification</td><td>VQA</td><td>Retrieval</td><td>Grounding</td><td>IND</td><td>OOD</td><td>Overall</td></tr><tr><td># of Datasets →</td><td></td><td>10</td><td>10</td><td>12</td><td>4</td><td>20</td><td>16</td><td>36</td></tr><tr><td>CLIP(ViT-L)</td><td>0.4B</td><td>42.8</td><td>9.1</td><td>53.0</td><td>51.8</td><td>37.1</td><td>38.7</td><td>39.2</td></tr><tr><td>OpenCLIP(ViT-L)</td><td>0.4B</td><td>41.5</td><td>6.9</td><td>44.6</td><td>53.5</td><td>32.8</td><td>36.0</td><td>36.6</td></tr><tr><td>SigLIP(So/14)</td><td>0.9B</td><td>40.3</td><td>8.4</td><td>31.6</td><td>559.5</td><td>32.3</td><td>38.0</td><td>35.0</td></tr><tr><td>CLIP(ViT-BigG/14)</td><td>2.5B</td><td>52.3</td><td>14.0</td><td>50.5</td><td>60.3</td><td>38.9</td><td>45.8</td><td>44.3</td></tr><tr><td>EVA-CLIP</td><td>8B</td><td>56.00</td><td>10.4</td><td>49.2</td><td>558.9</td><td>38.1</td><td>456</td><td>43.7</td></tr><tr><td>E5-V UniME</td><td>7B 7B</td><td>39.7 43.0</td><td>10.8</td><td>39.4</td><td>60.2</td><td>34.2</td><td>33.4</td><td>37.5</td></tr><tr><td></td><td></td><td></td><td>17.7</td><td>42.5</td><td>63.2</td><td>37.6</td><td>38.6</td><td>41.6</td></tr><tr><td>RetLLM</td><td>7B</td><td>60.3</td><td>27.8</td><td>62.4</td><td>60.2</td><td>52.0</td><td>50.2</td><td>54.2</td></tr></table>

Table 3: Ablation study of visual enhancement and entropy-based selection on Flickr30k and COCO.   

<table><tr><td>Components</td><td colspan="2">Flickr30k</td><td colspan="2">COCO</td></tr><tr><td></td><td>qi → ct qt → ci</td><td></td><td>qi → ct</td><td>qt → ci</td></tr><tr><td>ALL</td><td>94.5</td><td>81.8</td><td>69.2</td><td>52.1</td></tr><tr><td>entropy only</td><td>94.0</td><td>81.2</td><td>68.7</td><td>50.8</td></tr><tr><td>enhancement only</td><td>94.0</td><td>80.7</td><td>68.8</td><td>51.5</td></tr><tr><td>MLLM only</td><td>93.6</td><td>80.2</td><td>66.9</td><td>50.3</td></tr></table>

# 3.4. 消融研究

组件有效性 如表 3 所示，去除视觉增强会导致 COCO $( q ^ { i } \to c ^ { \bar { t } } )$ 的效果显著下降 $1 . 5 \%$ ，确认了其在零-shot 检索中保持视觉保真度的关键作用。禁用基于熵的选择导致 Flickr30K $( q ^ { t } \to c ^ { i \cdot }$ ) 下降 $1 . 1 \%$ ，展示了其在解决模糊排名方面的有效性。完整模型相较于“仅 MLLM”的一致优势凸显了两者组合所带来的协同增益。

Table 4: Performance comparison using different CLIP versions with fixed Qwen2.5-VL.   
Top $k$ Sensitivity As shown in Fig. 2, performance varies with different $k$ values (3, 5, 7, 9), revealing a clear trade-off between precision and efficiency: a larger $k$ improves recall at higher computational cost, while $k { = } 5$ (our default) offers the optimal balance for practical deployment.   

<table><tr><td rowspan="2">CLIP-version</td><td colspan="2">ShareGPT4V</td><td colspan="2">Urban1k</td></tr><tr><td>qi → ct qt → ci</td><td></td><td>qi → ct qt → ci</td><td></td></tr><tr><td>CLIP-ViT-B</td><td>94.8</td><td>88.1</td><td>84.0</td><td>71.8</td></tr><tr><td>CLIP-ViT-L</td><td>97.6</td><td>94.2</td><td>88.9</td><td>78.6</td></tr><tr><td>Long-CLIP-L</td><td>96.6</td><td>95.1</td><td>95.2</td><td>95.8</td></tr></table>

![](images/2.jpg)  
Fig. 2: Ablation studies on the impact of top- $\mathbf { \nabla } \cdot \mathbf { k }$ values on retrieval performance and inference efficiency.

Table 5: Performance comparison using different MLLMs with fixed CLIP-ViT-L.   

<table><tr><td>MLLMs</td><td colspan="2">ShareGPT4V</td><td colspan="2">Urban1k</td></tr><tr><td></td><td>qi → ct</td><td>qt → ci</td><td>qi → ct</td><td>qt → ci</td></tr><tr><td>phi3.5v</td><td>86.5</td><td>72.3</td><td>78.9</td><td>73.5</td></tr><tr><td>Qwen2-VL</td><td>93.8</td><td>94.1</td><td>87.2</td><td>78.2</td></tr><tr><td>Qwen2.5-VL</td><td>97.6</td><td>94.2</td><td>88.9</td><td>78.6</td></tr></table>

模型可扩展性 如表4和表5所示，RetLLM受益于更强大和更大的主干模型，随着基础组件容量的增加，性能持续改善。这突显了我们框架的可扩展性及其利用CLIP模型和多模态大语言模型进展的能力。

# 4. 结论

在本研究中，我们提出了Ret LLM，这是一种无训练的多模态检索框架，通过粗到精的搜索、视觉增强和基于熵的选择实现了强大的零-shot性能。关键的是，Ret LLM具有高度的可扩展性：它以即插即用的方式自然继承了更强基础模型的性能提升，使其成为未来检索系统向前兼容且可持续的解决方案。

5. 参考文献 [1] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau iel Ral 知识密集型 lp 任务," Advances in neural information processing systems, vol. 33, pp. 94599474, 2020. [2] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark等，"从自然语言监督中学习可转移的视觉模型," 在国际机器学习会议. PmLR, 2021, pp. 87488763. [3] Dongsheng Wang, Miaoge Li, Xinyang Liu, MingSheng Xu, Bo Chen, 和 Hanwang Zhang， "跨模态调优的多模态词元级提示对齐," Advances in Neural Information Processing Systems, vol. 36, pp. 5279252810, 2023. [4] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, ARiter n Wn ChnUTag n ben uversal ultodal oratio etrvers," 在欧洲计算机视觉会议. Springer, 2024, pp. 387404. [5] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, 和 Fuzhen Zhuang， "E5-v: 具有多模态大型语言模型的通用嵌入," arXiv 预印本 arXiv:2407.12580, 2024. [6] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, 和 Wei Ping， "Mm-embed: 具有多模态大型语言模型的通用多模态检索," arXiv 预印本 arXiv:2411.02571, 2024. [7] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, 和 Jinsong Su， "Llave: 具有困难加权对比学习的大型语言和视觉嵌入模型," arXiv 预印本 arXiv:2503.04812, 2025. [8] Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, 和 Weidi Xie， "Lamra: 将大型多模态模型作为您的高级检索助手," 在计算机视觉与模式识别会议论文集中，2025, pp. 40154025. [9] Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, 和 Jiankang Deg "打破模态壁垒: 使用多模态大型语言模型的通用嵌入学习," arXiv 预印本 arXiv:2504.17432, 2025. [10] Zhanpeng Chen, Chengjin Xu, Yiyan Qi, 和 Jian Guo， "Mllm 是一个强大的重排序器: 通过知识增强重排序和噪声注入训练推动多模态检索增强生成," arXiv 预印本 arXiv:2407.21439, 2024. [11] Xin Zou, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Kening Zheng, Sirui Huang, Junkai Chen, Peijie Jiang, Jia Liu, Chang Tang等， "在回答之前多看两次: 视觉记忆空间回溯以减轻多模态大型语言模型中的幻觉问题," arXiv 预印本 arXiv:2410.03577, 2024. [12] Dongsheng Wang, Jiequan Cui, Miaoge Li, Wang Lin, Bo Chen, 和 Hanwang Zhang， "无指令调优的多模态大型语言模型的视觉词元补全," 在欧洲计算机视觉会议. Springer, 2024, pp. 446462. [13] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, 和 Kate Saenko， "图像字幕中的对象幻觉," arXiv 预印本 arXiv:1809.02156, 2018. [14] Junyan Lin, Haoran Chen, Yue Fan, Yingqi Fan, Xin Jin, Hui Su, Jinlan Fu, 和 Xiaoyu Shen， "多层视觉特征融合在多模态大型语言模型中的方法、分析和最佳实践," 在计算机视觉与模式识别会议论文集中，2025, pp. 4156 4166. [15] Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, 和 Huajun Chen， "Mllm 可以看到吗? 动态校正解码以减轻幻觉问题," arXiv 预印本 arXiv:2410.11779, 2024. [16] Peter Young, Alice Lai, Micah Hodosh, 和 Julia Hockenmaier， "从图像描述到视觉指称: 事件描述语义推理的新相似性度量," 计算语言学协会交易, vol. 2, pp. 6778, 2014. [17] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, 和 C Lawrence Zitnick， "Microsoft coco: 背景下的常见对象," 在欧洲计算机视觉会议. Springer, 2014, pp. 740755. [18] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, 和 Dahua Lin， "Sharegpt4v: 用更好的字幕改善大型多模态模型," 在欧洲计算机视觉会议. Springer, 2024, pp. 370387. [19] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, 和 Jiaqi Wang， "Long-clip: 解锁 clip 的长文本能力," 在欧洲计算机视觉会议. Springer, 2024, pp. 310325. [20] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, 和 Ranjay Krishna， "Sugarcrepe: 修复可黑客攻击的视觉语言组合基准," Advances in neural information processing systems, vol. 36, pp. 3109631116, 2023. [21] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, 和 Wenhu Chen， "Vlm2vec: 为大规模多模态嵌入任务训练视觉-语言模型," arXiv 预印本 arXiv:2410.05160, 2024. [22] Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, 和 Xinlong Wang， "Eva-clip-18b: 将 clip 扩展到 180 亿参数," arXiv 预印本 arXiv:2402.04252, 2024. [23] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, 和 Lucas Beyer， "用于语言图像预训练的 Sigmoid 损失," 在 IEEE/CVF 国际计算机视觉会议论文集，2023, pp. 1197511986. [24] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, 和 Junyang Lin， "Qwen2.5-v 技术报告," arXiv 预印本 arXiv:2502.13923, 2025. [25] Marah Abdin, Jyoti Aneja, 和 Hany Awadalla 等，"Phi-3 技术报告: 一种高度智能的语言模型可在您的手机上本地运行," 2024. [26] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, 和 Junyang Lin， "Qwen2-vl: 在任何分辨率下增强视听语言模型对世界的感知," arXiv 预印本 arXiv:2409.12191, 2024.