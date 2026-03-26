# 多模态大语言模型知晓观察重点：无训练的小视觉细节感知

贾瑞·张 马哈亚尔·凯亚托赫 伊普拉特克·奇卡拉 费利普·伊列夫斯基 $\clubsuit$ 南加州大学，美国 $\nleftarrow$ 阿姆斯特丹自由大学，荷兰

# 摘要

多模态大型语言模型（MLLMs）近年来在视觉识别任务中取得了快速进展。鉴于它们在许多关键应用中的潜在整合，理解其视觉感知的局限性显得尤为重要。在本研究中，我们考察了MLLMs在回答关于图像的问题时，能否像处理大视觉细节一样有效地感知小视觉细节。我们观察到，它们的表现对问题视觉主题的大小非常敏感，并通过干预研究进一步证明了这一效果实际上是因果关系。接下来，我们研究MLLMs在回答视觉问题时的注意力模式，有趣的是，即使在提供错误答案时，它们也能始终知道该关注哪里。基于这些发现，我们提出了无训练的视觉干预方法，利用任何MLLM自身的内部知识（以注意力和梯度图的形式）来增强其对小视觉细节的感知。我们在两种广泛使用的MLLMs上及七个视觉问答基准上评估了所提方法，结果表明，这些方法能够显著提高MLLMs的准确性，而无需进行任何训练。我们的结果阐明了将MLLMs应用于涉及小细节的视觉识别任务时的风险，并表明使用模型内部状态进行视觉干预是减轻此风险的一个有前景的方向。

# 1 引言

多模态大语言模型（MLLMs）已经大大推动了多模态推理和规划的最新进展，并迅速被集成到各种下游应用中，这些应用范围从机器人（李等，2024b；陈等，2024）、生物医学（李等，2023a）、自动驾驶（许等，2024b；张等，2023a）到视觉数学推理（高等，2023；张等，2024c；b）甚至食品食谱生成（Chhikara等，2024）。鉴于MLLMs的应用迅速增长，尤其是在生物医学和安全等关键领域，研究它们的视觉感知限制对于阐明可能影响其下游应用的潜在风险至关重要。

为了激发本研究聚焦的局限性，我们首先在图 1 中展示三个揭示性的视觉问答示例，在这些示例中，我们请求一款流行的多模态语言模型 BLIP-2 $( \mathrm { F l a n T 5 _ { X L } }$ ) (Li et al., 2023b) 识别每张图像中物体的存在或类型，同时我们改变物体的大小。在没有任何先前证据的情况下，我们可能合理地预期该模型的回答与物体的大小无关，因为该模型具有较大的表征能力，并在包含各种大小物体的多种图像上进行了预训练。相反，在图 1（左侧）中，我们观察到模型最初并未识别出小街标的存在，并且为正确答案分配了较低的概率；然而，通过更聚焦的视觉裁剪放大图像后，针对街标的正确答案所分配的概率逐渐增加，这表明模型逐渐感知到更多与街标相关的细节。

![](images/1.jpg)  

图：视觉裁剪对BLIP-2 FlanTL零-shot VQA模型预测答案概率的影响。$\mathbf{X}$轴标签是每个图下方显示的相应裁剪图像的索引，模型在每个步骤中看到这些图像。模型逐渐找到了正确的答案。在图1（中），我们观察到进一步的证据，表明模型在感知小细节方面存在困难：模型最初将鸟的类型预测为白色，但当我们将图像放大到鸟的部分时，而不以任何方式改变问题，我们观察到模型逐渐对正确的鸟类（白鹭）赋予更高的概率。这表明模型并没有在理解“类型”时出现语义错误，而是无法感知足够的细节来区分白鹭和其他白色鸟类，而视觉裁剪减轻了这一问题。类似地，在图1（右），我们观察到模型的初始回答并非完全无关（“ama”与正确答案“moma”相比），这表明模型根据问题知道该看哪里，但无法准确感知实际的单词，这同样是通过视觉裁剪得以缓解的。

在本研究中，我们将深入探讨图1中观察到的限制，阐明其原因，并提出潜在解决方案以减轻其影响。在第3节中，我们定量展示了在各种广泛使用的大语言模型中，确实存在感知小视觉概念的困难。我们的发现与先前关于评估视觉-语言联合嵌入模型中文本-图像匹配的研究一致，这些研究观察到了图像中视觉物体大小与文本-图像匹配分数之间的负相关关系（Zhao et al., 2022），但我们进一步通过干预研究建立了视觉概念大小与大语言模型感知能力之间的因果关系。在第4节中，我们探讨了大语言模型在感知小视觉概念时的困难是否源于对视觉细节的感知困难，还是由于其小尺寸导致的概念定位困难。我们定量显示，即使在无法正确回答问题时，大语言模型依然能够准确知道该关注在哪里。在第5节中，我们提出了三种自动视觉裁剪方法——利用大语言模型自身的注意力图和梯度——作为可扩展且无须训练的解决方案来应对视觉感知限制。最后，在第6节中，我们将提出的方法应用于两种流行的大语言模型，并在七个视觉问答（VQA）基准测试中对其进行评估，显示出其在提高大语言模型准确性方面的有效性，特别是在对细节敏感的基准上。

# 2 相关工作

多模态大语言模型（MLLMs）。MLLMs 是能够处理多样化语言和视觉任务的基础模型。这些模型分为两类：端到端预训练模型和模块化预训练模型。端到端模型通过双编码器（Radford等，2021）、融合编码器（Li等，2021）、编码器-解码器（Cho等，2021）和统一变换器（Wang等，2022）等架构处理联合图像-语言数据，使用图像-文本匹配、对比学习和掩蔽语言建模等目标。模块化预训练模型在近期的最先进方法中占据主导地位，通过调整现有组件来避免高昂的完全预训练成本：BLIP2（Li等，2023b）和InstructBLIP（Dai等，2023）在冻结的预训练ViT（Dosovitskiy等，2021）图像编码器和冻结的大语言模型之间训练一个基于变换器的连接器，该连接器将ViT输出词元转换为大语言模型输入空间中的固定图像词元集合；Qwen-VL（Bai等，2023）同样使用固定长度的词元连接器（一个单一的交叉注意力层），但同时训练连接器和大语言模型；LLaVA（Liu等，2023b）和LLaVA-1.5（Liu等，2023a）则分别使用线性投影和两层的多层感知机作为它们的连接器，且均进行训练。我们的工作将有助于更好地理解MLLM的感知限制，并在不进行训练的情况下可扩展地改善它们的感知能力，为现有方法提供正交的益处。

视觉定位方法。专门的视觉定位方法，如 YOLO（Redmon 等，2016）、SAM（Kirillov 等，2023）和 GLIP（Li 等，2022b），在识别显著图像区域时高度依赖于密集的空间注释。本地方法，如 Grad-CAM（Selvaraju 等，2017），通过分析分类器决策的梯度来定位区域，而不需要空间监督。先前的工作将 Grad-CAM 适配到 BLIP（Li 等，2022a），利用其专用的图像-文本相似度计算神经网络，即图像-文本匹配网络（Tiong 等，2022；Guo 等，2023）。在本研究中，我们推导了一种更通用的方法，以定位 MLLM 对图像的关注，而不依赖于特定的 BLIP 架构。一些近期的研究探索了改善 MLLM 在视觉问答中视觉定位能力的方法，包括思维链（Shao 等，2024；Liu 等，2024b）、工具使用（Wu 和 Xie，2023）以及视觉编程方法（Surís 等，2023；Gupta 和 Kembhavi，2023）。相反，我们展示了 MLLM 通常能够有效地在其内部状态中定位问题的视觉主题，并提出无培训方法以利用其内部状态来改善其视觉感知。

MLLM 中的视觉感知限制。若干先前和同时进行的研究已观察到回答关于图像中小物体的问题的困难（Zhang 等，2023b；2024a；Liu 等，2024a；Wu 和 Xie，2023），这些研究探讨了基于高分辨率微调（Liu 等，2024a；Dehghani 等，2023；Wang 等，2024）、多智能体管道（Wu 和 Xie，2023）以及视觉裁剪的缓解方案（Zhang 等，2023b）。在本研究中，我们提供了更广泛的证据来说明这一困难，确立其对 MLLM 性能的因果影响，并表明其根源在于未能观察到小的视觉细节，而不是未能定位小物体。还有若干研究表明 MLLM 存在物体幻觉问题（Li 等，2023c；Yu 等，2024）。此外，Zhang 等（2024a）展示了 MLLM 中的视觉盲点——即 MLLM 感知降级的图像位置，以及它们对视觉质量、图像中视觉干扰物的存在，甚至局部物体位置扰动的敏感性。

# 3 MLLMs对视觉概念大小的敏感性

在这一部分，我们的目标是定量研究我们在图1中的定性观察，即大语言模型在描述图像中的小视觉细节方面存在困难。为此，我们考虑TextVQA数据集，在该数据集中，对于每个问题，我们可以找到包含正确文本答案的图像真实标注边界框。我们将其验证集根据真实标注边界框的相对大小$\begin{array} { r } { S = \frac { A _ { b b } } { A _ { t o t a l } } } \end{array}$进行三个组的划分，其中$A _ { b b }$和$A _ { t o t a l }$分别是图像的总面积：1）$S < 0 . 0 0 5$（小）包含773个问题-图像对，2）$0 . 0 0 5 \le S < 0 . 0 5$（中）包含2411个问题-图像对，3）$S \geq 0 . 0 5$（大）包含1186个问题-图像对。我们选择TextVQA进行这项研究，因为它包含大量关于小视觉概念的问题，而文本答案相较于图像中的其他侧面信息（与物体类型和属性相比），更难被大语言模型猜测。

Table 1: Sensitivity of the accuracy of MLLMs to the size of visual concepts in TextVQA. As the relative visual size of the answer decreases (right to left in each row), we observe a decline in the accuracy of the original models (no cropping) in answering questions, whereas visual cropping (human-CROP) significantly improves accuracy on smaller objects.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">Answer Bbox Size (S)</td></tr><tr><td>small</td><td>medium</td><td>large</td></tr><tr><td rowspan="2">BLIP-2 (FlanT5xL)</td><td>no cropping</td><td>12.13</td><td>19.57</td><td>36.32</td></tr><tr><td>human-CROP</td><td>55.76</td><td>52.02</td><td>45.73</td></tr><tr><td rowspan="2">InstructBLIP (Vicuna-7B)</td><td>no cropping</td><td>21.79</td><td>30.58</td><td>45.30</td></tr><tr><td>human-CROP</td><td>69.60</td><td>61.56</td><td>53.39</td></tr><tr><td rowspan="2">LLaVA-1.5 (Vicuna-7B)</td><td>no cropping</td><td>39.38</td><td>47.74</td><td>50.65</td></tr><tr><td>human-CROP</td><td>69.95</td><td>65.36</td><td>56.96</td></tr><tr><td rowspan="2">Qwen-VL (Qwen-7B)</td><td>no cropping</td><td>56.42</td><td>65.09</td><td>68.60</td></tr><tr><td>human-CROP</td><td>70.35</td><td>75.49</td><td>71.05</td></tr><tr><td rowspan="2">GPT-40</td><td>no cropping</td><td>65.76</td><td>72.81</td><td>69.17</td></tr><tr><td>human-CROP</td><td>75.63</td><td>81.32</td><td>71.72</td></tr></table>

敏感性研究。如果一个模型对视觉概念的大小不敏感，我们预计它在所有三个分区中的准确性将相似。在表1中，我们观察到，所有多模态语言模型（MLLM）的准确性在真实边界框相对变小时（在未裁剪的行中从右到左）会下降。BLIP-2和InstructBLIP未在TextVQA上进行训练（即它们是零-shot模型），它们在大分区和小分区之间的准确性分别下降了24和23个百分点。LLaVA-1.5和Qwen-VL在TextVQA的训练集上进行过训练，但它们在大分区和小分区之间的准确性也分别下降了11和12个百分点。最后，即便是最新的商业模型GPT-4o，尽管其训练集可能包含全部TextVQA，仍在小分区和中分区之间的准确性下降了7个百分点。这些发现表明，多模态语言模型在感知较小视觉概念方面存在偏见。

干预研究。我们上面观察到的感知限制可能仅与大小相关。为了研究这一限制是否与大小存在因果关系，我们开展了一项干预研究，向大型语言模型提供基于真实标注边界框的视觉裁剪图像，称为人类裁剪（human-CROP）。更具体地，对于每个图像-问题对和每个大型语言模型，我们从图像中裁剪出包含真实标注边界框的最小正方形区域，并将其调整到大型语言模型的输入图像分辨率（该正方形裁剪可防止图像缩放时的极端变形）。然后，将裁剪后的图像与原始图像-问题对一起提供给大型语言模型（详见图4）。我们在表1中观察到，人类裁剪显著提高了所有大型语言模型在小型和中型分区的准确性，而在大型分区的提升幅度较小。这些发现表明，感知限制确实是由视觉概念的大小引起的，而视觉裁剪可能是缓解这一限制的有希望的方向。

# 4 多模态大语言模型知道该看哪里吗？

感知小型视觉概念的局限性可能有两个主要原因：1）它们在较大图像中很难定位，2）它们的小细节难以正确感知。我们在图1中观察到，MLLM的不正确回答可能包含部分正确的信息，这暗示它可能知道在图像中该在哪里查找。在本节中，我们定量研究这一观察，以回答MLLM对大小的敏感性是根源于感知局限还是定位局限。为此，我们首先利用MLLM的Transformer层内计算的注意力图来量化其在图像上的空间注意力，然后将这一注意力在真实边界框内的总量与其他同一大小的边界框进行比较。MLLM的设置。考虑的MLLM在四个步骤中处理给定的图像-问题对$( x , q )$（如图4所示）：1）图像被划分为$N \times N$非重叠的块，并通过ViT图像编码器处理为$N \times N$输出词元；2）ViT输出词元通过MLP（LLaVA-1.5）或Transformer连接器（BLIP-2, InstructBLIP和Qwen-VL）转换为主干LLM的输入空间，我们称之为图像词元；3）图像词元随后被预先添加到问题词元和预定义的起始答案词元中，并馈送到LLM；4）LLM根据起始答案词元进行自回归采样（我们使用贪婪采样）。

量化大语言模型（MLLMs）对图像空间注意力的影响。我们首先通过提取主干大语言模型中起始答案标记对所有图像标记的 softmax 交叉注意力，来衡量每个图像标记对 MLLM 决策的重要性（答对标记注意力），结果为 $A _ { s t } ( x , q ) \in \mathbb { R } ^ { L \times H \times 1 \times T }$，其中 $L , H$ 是主干大语言模型的层数和每层头数，$T$ 是提供给大语言模型的图像标记数量。然后我们对注意力进行平均 $\begin{array} { r } { \hat { A } _ { s t } ( x , q ) = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } A _ { s t } ( x , q ) } \end{array}$。接下来，我们衡量每个图像区域对每个图像标记的重要性（标记对图像注意力）。对于使用 Transformer 连接器将 ViT 输出标记重新采样为固定数量图像标记的 MLLM（BLIP-2、InstructBLIP 和 Qwen-VL），我们提取每个图像标记到 $A _ { t i } \in \mathbb { R } ^ { L _ { c } \times H _ { c } \times T \times N ^ { 2 } }$ 的 softmax 交叉注意力，其中 $L _ { c } , H _ { c }$ 是连接器中的层数和每层头数，$T$ 是可学习的查询标记数量（作为大语言模型的输入图像标记），$N ^ { 2 }$ 是 ViT 图像编码器的图像块数量。然后我们对注意力头进行平均，以获得标记对图像注意力 $\begin{array} { r } { \hat { A } _ { t i } ( x ) = \frac { 1 } { H _ { c } } \sum _ { h = 1 } ^ { H _ { c } } A _ { t i } ( x ) } \end{array}$。对于图像标记，我们将 $\hat { A } _ { t i } ( x )$ 设置为单位张量。最后，我们通过计算答对标记和标记对图像注意力的张量积来计算答对图像注意力，结果为 $A _ { s i } ( x , q ) \in \mathbb { R } ^ { L \times L _ { c } \times 1 \times N ^ { 2 } }$，其中 $A _ { s i } ^ { m k } ( x , q ) = \hat { A } _ { s t } ^ { m } ( x , q ) \hat { A } _ { t i } ^ { k } ( x )$（上标 $m$ 和 $k$ 分别表示大语言模型和连接器的层索引）。

![](images/2.jpg)  

Figure 2: Examples of MLLMs knowing where to look despite answering incorrectly. The right panel in each example displays relative attention to the image (defined in Sec. 4) of one layer in the MLLM.

相对注意力。使用 softmax 交叉注意力的一个问题是，并非所有受到高度关注的词元在语义上都与输入问题相关。例如，最近的研究观察到，变换器可能使用多个词元作为寄存器来聚合全局信息（Darcet et al., 2023）。为了强调语义上相关的注意力，我们建议通过在通用指令 $q ^ { \prime }$ 上的值来归一化图像-问题对 $( x , q )$ 的答案到图像的注意力。具体而言，我们考虑固定的指令 $q ^ { \prime } = \mathbf { \dot { \bar { \Sigma } } }$ “写一段关于图像的一般描述。”并计算相对注意力为 $\begin{array} { r } { A _ { r e l } ( x , q ) = \frac { A _ { s i } ( x , q ) } { A _ { s i } ( x , q ^ { \prime } ) } } \end{array}$ LLaVA-1.5 和 InstructBLIP 的 $\cdot A _ { r e l } ^ { m k }$ 分别在层 $m = 1 4 , k = 0$ 和 $m = 1 5 , k = 2$。

MLLMs 知道该看哪里吗？配备相对注意力后，我们重新回到 MLLMs 是否存在定位限制或感知限制的问题。为此，我们再次考虑 TextVQA 的验证集。对于每对图像-问题，我们首先计算相对注意力。然后，我们定义注意力比率为答案真实边界框内总（求和）相对注意力与图像中所有与真实值大小相同的边界框的平均相对注意力的比率。这个比率提供了 MLLM 在多大程度上关注于真实边界框的度量（根据马尔可夫不等式的意义）。在图 3 中，我们绘制了 TextVQA 验证集上所有考虑的 MLLMs 各层的注意力比率的平均值（置信区间为 95%）。横轴显示了组合层索引 $l = m + k \times L$，其中 $m \in \{ 0 \ldots L - 1 \}$ 代表主干 LLM 中交叉注意力层的数量，而 $k \in \{ 0 \ldots L _ { c } - 1 \}$ 代表连接器中的交叉注意力层的数量（BLIP-2: $L = 24 , L _ { c } = 6$；InstructBLIP: $L = 32 , L _ { c } = 6$；Qwen-VL: $L = 32 , L _ { c } = 1$；LLaVA-1.5: $L = 32 , L _ { c } = 1$）。在所有 MLLMs 中，我们观察到大多数层的注意力比率显著大于 1，表明模型在图像上的真实边界框区域上显著关注。有趣的是，无论模型能否正确回答问题，它们都表现出对正确区域的同样强烈关注。这些观察结果表明，尽管模型可能回答错误，MLLMs 趋向于知道该看哪里。

![](images/3.jpg)  

Figure 3: MLLMs' attention ratio across all layers (average with $9 5 \%$ CI over TextVQA). The attention ratio measures how significantly the MLLM is attending to the ground-truth bounding box (defined in Sec. 4). We observe that it is greater than 1 in most layers, showing that the MLLMs know where to look in the image even when they fail to answer correctly.

# 5 自动化视觉裁剪 (VICRoP)

我们在第4节中观察到，MLLMs对视觉概念大小的敏感性主要是感知限制（而非定位限制）。因此，缓解这一限制的一种解决方案是通过保持每个图像块的分辨率，简单地用更多的图像块对MLLMs进行训练（从而提高MLLMs的图像分辨率）。然而，将输入图像的分辨率提高一个倍数 $\alpha$ ，会将ViT输入块（和输出词元）的数量从 $\bar { N } ^ { 2 }$ 增加到 $\alpha ^ { 2 } N ^ { 2 }$ ，这反过来会使得softmax注意力计算的复杂度按 $\alpha ^ { 4 } N ^ { 4 }$ 的量级增加。鉴于这一解决方案对于当前基于Transformer的MLLMs并不可扩展，我们选择探索不需要任何训练且可扩展到任意图像分辨率的替代方案。我们注意到，几项并行的工作已经探索了用更高分辨率图像块训练MLLMs的第一方向（Li et al., 2024c; Sun et al., 2024; Li et al., 2024d; McKinzie et al, 2024; Xu et al., 2024a; Luo et al., 2024），值得注意的是，LLaVA-Next（Liu et al., 2024a）在撰写时已在多个VQA基准上达到了VQA的最先进水平。我们相信我们的工作与这些努力是平行的：与其不断训练更高分辨率的MLLMs以使其能够看到所有分辨率（这是不可避免的上限），我们探索如何智能地调整输入图像，使其更接近MLLMs已经能够看到的内容，而无需额外的训练。我们提供证据表明，我们的无训练方法可以在D和E附录中为基于训练的方法提供正交益处。受到我们发现的启发，即MLLMs往往知道在哪里看（第4节）以及视觉裁剪可以缓解感知限制（第3节），在本节中我们构建了三种自动视觉裁剪方法，以减轻MLLMs的感知限制。这些方法旨在利用MLLM自身的内部信息——以注意力图和梯度的形式——来找到图像中感兴趣的近似区域（即包含问题主题的区域），然后通过视觉裁剪放大该区域。视觉裁剪的一个潜在缺点是某些问题可能需要对图像有全局视角。为了解决这个问题，我们利用MLLMs通常会将图像转换为一系列词元的事实。这使我们能够通过连接视觉裁剪后的图像词元，直接扩展原始图像词元，如图4所示。在对MLLMs应用我们所有方法时，我们使用这种连接方式。

![](images/4.jpg)  

Figure 4: Illustration of the proposed visual cropping approach applied to two MLLMs.

相对注意力ViCrop（rel-att）。在此方法中，我们直接计算在第4节中定义的每个图像-问题对$( x , q )$的相对注意力$A _ { r e l } ( x , q )$。然后，我们根据TextVQA中的一个小的保留样本集选择一个目标层（在LLM和连接器中），并使用其相对注意力作为视觉裁剪的重要性图（在下文中讨论）。我们在第6节中对层的选择进行了消融实验。

梯度加权注意力ViCrop（grad-att）。相对注意力通过MLLM运行额外的通用指令，以归一化答案与图像之间的注意力，并强调语义相关的注意力。作为一种不需要第二次前向传播的替代方案，我们考虑使用梯度来归一化注意力，因为模型决策相对于注意力得分的梯度显示了决策对该注意力变化的敏感性，从而体现了该注意力在回答问题时的语义相关性。为了获得模型决策的可微分表示，我们考虑起始答案词元处最大输出概率的对数，$v = \log \operatorname { s o f t m a x } ( z ( x , q ) ) _ { t ^ { * } } \ \bar { \in } \mathbb { R }$，其中$z \in \mathbb { R } ^ { D }$是LLM在起始答案位置的输出logit，$D$是词汇表大小，$t ^ { * } = \arg \operatorname* { m a x } _ { t } z _ { t }$。然后，按照第4节的符号，我们可以计算答案与词元之间注意力的梯度加权版本$\tilde { A } _ { s t } ( x , q ) = A _ { s t } ( x , q ) \odot \sigma ( \nabla _ { A _ { s t } } v ( x , q ) )$和词元与图像之间注意力的梯度加权版本$\tilde { A } _ { t i } ( x , q ) = A _ { t i } ( x ) \odot \sigma ( \nabla _ { A _ { t i } } v ( x , q ) )$，其中$\odot$是逐元素乘积，而$\sigma ( w ) = \mathrm { \bar { m a x } } ( 0 , w )$。我们去除负梯度，因为它们对应的词元如果被关注，将降低模型的确定性，因此通常是分散注意力的位置（Selvaraju等，2017）。最后，我们计算梯度加权的答案与图像之间的注意力，作为以下张量积$\mathring { A } _ { s i } ( x , q ) = \mathring { A } _ { s t } ( x , \bar { q } ) \otimes \tilde { A } _ { t i } ( x , \bar { q } ) \in \mathbb { R } ^ { L \times L _ { c } \times 1 \times N ^ { 2 } }$。我们将从$\tilde { A } _ { s i } ( x , q )$中选择在rel-att中识别的相同目标层作为视觉裁剪的重要性图。

输入梯度 ViCrop（纯梯度）。在该方法中，我们直接使用 MLLM 对输入图像的决策的梯度来寻找图像上的相关区域。与之前基于注意力的方法相比，纯梯度方法更加灵活，因为它不依赖于基于 Transformer 的架构。具体来说，对于每对图像-问题 $( x , q )$ ，我们将计算 $\boldsymbol { G } ( \boldsymbol { x } , \boldsymbol { q } ) = \| \nabla _ { \boldsymbol { x } } \boldsymbol { v } ( \boldsymbol { x } , \boldsymbol { q } ) \| _ { 2 }$ ，其中 $v ( x , q )$ 是 LLM 在起始答案词元处的最大输出概率的对数（如上述 grad-at t 所定义），而 L2 范数在图像通道维度上进行计算。然而，梯度在完全一致的颜色区域（例如蓝天）中有时会显示出高值。由于这些非边缘区域不包含任何视觉细节，我们将明确降低它们在 $G$ 中的影响。为此，我们首先对图像应用一个 $3 \times 3$ 尺寸的高通 Gaussian 滤波器，然后使用同样大小的中值滤波器来减少椒盐噪声。经过滤波的图像在其空间中值处进行阈值处理，以形成二进制掩膜，并与 $G$ 进行逐元素相乘。最后，强调边缘的 $G$ 被空间平均池化为 MLLM 的 $N \times N$ 贴片，生成用于视觉裁剪的重要性图。

用于视觉裁剪的边界框选择。为了将重要性图（来自上述每种方法）转换为边界框，我们借鉴了目标检测文献中提出的滑动窗口方法（Redmon 等，2016）。具体而言，对于每个 MLLM，我们定义了一组窗口，其大小等于 MLLM 输入图像分辨率的倍数。倍数在 $\{ 1 , 1 . 2 , . . . 2 \}$ 中。我们以步幅为 1 的方式将每个窗口在图像上滑动，并找到内部重要性图之和最大的最佳位置。在为每个窗口选择最佳位置之后，我们选择内和与其他窗口的内和差异最大的窗口。如果选定的窗口太小或太大（请注意，此时将窗口稍微向左/右或向上/下移动并不会显著改变其内部和），则被选中的窗口将从图像中裁剪出来，调整为 MLLM 的输入图像分辨率，并与图像-问题对一起提供给 MLLM。

高分辨率视觉裁剪。在我们考虑的一个基准测试中，$\mathbf { V } ^ { * } \mathbf { \Sigma } \mathbf { W } \mathbf { u }$ 和 Xie (2023)，图像的分辨率非常高（始终超过1K），因此提供给多语言大型模型的调整大小的输入图像可能完全丧失与问题相关的视觉概念。为了解决这个问题，在这个基准测试中，我们采用了两阶段策略。在第一阶段，我们将图像分割成小于 $1 0 2 4 \times 1 0 2 4$ 的不重叠块，并保持长宽比接近1，按照ViCrop方法分别计算每个块的显著性图，然后将这些块重新拼接在一起。在第二阶段，我们在重新拼接的显著性图上找到视觉裁剪的边界框，正如之前所述，并将原始图像-问题对与调整大小的裁剪图像一起提供给多语言大型模型。

![](images/5.jpg)  

Figure 5: Examples of re1-at t helping MLLMs correct their mistakes (cyan-colored bounding box shows cropped region by rel-att; zoom-in insets are displayed for better readability).

# 6 VICROP 方法分析

在这一部分中，我们将我们提出的视觉裁剪方法应用于两个开源的最先进的多模态大语言模型，InstructBLIP (Vicuna-7B) (Dai et al., 2023) 和 LLaVA-1.5 (Vicuna-7B) (Liu et al., 2023a)。我们评估它们在四个对细节敏感的数据集上提高较小视觉概念感知的有效性（TextVQA 2 (Singh et al., 2019)、$\mathbf { V } ^ { * }$ (Wu and Xie, 2023)、POPE (Li et al., 2023c)、DocVQA (Mathew et al., 2021)），以及它们在三个主要包含大型物体的一般目的数据集上保持对大型视觉概念性能的能力（GQA (Hudson and Manning, 2019)、AOKVQA (Schwenk et al., 2022)、VQAv2 (Goyal et al., 2017)）。InstructBLIP 使用超参数 $N = 16, m = 15, k = 2$ 和输入图像分辨率为 $224 \times 224$。LLaVA-1.5 使用 $N = 24, m = 14$ 和输入图像分辨率为 $336 \times 336$。在报告准确率时，我们对所有基准计算 VQA 分数³，除了 GQA。对于 GQA，我们使用官方代码计算准确率。有关实现、数据集和提示的更多详细信息，请参见附录 A 至 C。

ViCrop 改善了视觉问答（VQA）准确性。在图 5 中，我们展示了 ViCrop 帮助 MLLM 自我纠正的一些例子（更多例子见附录 G），在表 2 中，我们报告了所提议的 ViCrop 方法在 VQA 基准测试上的准确性。我们观察到，所有方法在细节敏感的基准测试上显著提高了原始 MLLM（未裁剪）的准确性，而不需要任何训练，同时维持了 MLLM 在拥有更大视觉概念的基准上的表现。因此，在细节（尤其是在 TextVQA 和 $\mathbf { V } ^ { * }$）上的准确性提升似乎并没有以更大视觉细节和关系的准确性为代价。我们还观察到，LLaVA-1.5 的准确性提升比 InstructBLIP 更为显著。这可以通过以下事实来解释：在调优过程中，InstructBLIP 仅训练其连接器，而不训练其主干 LLM——LLM 并没有适应使用图像词元，而是图像词元被调整以最佳方式提示 LLM——因此 LLM 无法有效利用通过视觉裁剪提供的额外（裁剪的）图像词元。尽管如此，结果显示 ViCrop 可以有效应用于不同的 MLLM，并且是缓解第 3 节中观察到的感知限制的有前景的推理时解决方案。

Table 2: Accuracy of the proposed ViCrop methods on visual question answering benchmarks.   

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="4">Smaller Visual Concepts</td><td colspan="3">Larger Visual Concepts</td></tr><tr><td>TextVQA†</td><td>V*</td><td>POPE</td><td>DocVQA</td><td>AOKVQA</td><td>GQA</td><td>VQAv2</td></tr><tr><td rowspan="4">LLaVA-1.5</td><td>no cropping</td><td>47.80</td><td>42.41</td><td>85.27</td><td>15.97</td><td>59.01</td><td>60.48</td><td>75.57</td></tr><tr><td>rel-att</td><td>55.17</td><td>62.30</td><td>87.25</td><td>19.63</td><td>60.66</td><td>60.97</td><td>76.51</td></tr><tr><td>grad-att</td><td>56.06</td><td>57.07</td><td>87.03</td><td>19.84</td><td>59.94</td><td>60.98</td><td>76.06</td></tr><tr><td>pure-grad</td><td>51.67</td><td>46.07</td><td>86.06</td><td>17.70</td><td>59.92</td><td>60.54</td><td>75.94</td></tr><tr><td rowspan="4">InstructBLIP</td><td>no cropping</td><td>33.48</td><td>35.60</td><td>84.89</td><td>9.20</td><td>60.06</td><td>49.41</td><td>76.25</td></tr><tr><td>rel-att</td><td>45.44</td><td>42.41</td><td>86.64</td><td>9.95</td><td>61.28</td><td>49.75</td><td>76.84</td></tr><tr><td>grad-att</td><td>45.71</td><td>37.70</td><td>86.99</td><td>10.81</td><td>61.77</td><td>50.33</td><td>76.08</td></tr><tr><td>pure-grad</td><td>42.23</td><td>37.17</td><td>86.84</td><td>8.99</td><td>61.60</td><td>50.08</td><td>76.71</td></tr></table>

层选择和高分辨率视觉裁剪使用的消融研究表。

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="3">Choice of Layer</td><td colspan="3">High-Resolution ViCrop</td></tr><tr><td>Selective</td><td>Average</td><td>∆</td><td>w/ High-Res</td><td>w/o High-Res</td><td>∆</td></tr><tr><td rowspan="4">LLaVA-1.5</td><td>no cropping</td><td>47.80</td><td>−</td><td>−</td><td>42.41</td><td>42.41</td><td></td></tr><tr><td>rel-att</td><td>55.17</td><td>55.45</td><td>+0.28</td><td>62.30</td><td>47.64</td><td>-14.66</td></tr><tr><td>grad-att</td><td>56.06</td><td>56.26</td><td>+0.20</td><td>57.07</td><td>49.74</td><td>-7.33</td></tr><tr><td>pure-grad</td><td>51.67</td><td></td><td></td><td>46.07</td><td>45.03</td><td>-1.04</td></tr><tr><td rowspan="4">InstructBLIP</td><td>no cropping</td><td>33.48</td><td>−</td><td></td><td>35.60</td><td>35.60</td><td></td></tr><tr><td>rel-att</td><td>45.44</td><td>44.40</td><td>-1.04</td><td>42.41</td><td>38.74</td><td>-3.67</td></tr><tr><td>grad-att</td><td>45.71</td><td>44.98</td><td>-0.73</td><td>37.70</td><td>42.41</td><td>+4.71</td></tr><tr><td>pure-grad</td><td>42.23</td><td></td><td></td><td>37.17</td><td>42.41</td><td>+5.24</td></tr></table>

关于层选择的消融研究。为了理解选择信息丰富的层对 rel-att 和 grad-att 的重要性（如第 5 节所讨论），在表 3 中，我们比较了这些方法在 TextVQA 上简单取 $A _ { r e l }$ 和 $\tilde { A } _ { s i }$ 中所有层的平均值时的准确率。我们观察到 rel-att 对这一选择具有鲁棒性，而 grad-att 的准确率下降了约 3.5 个百分点。重要的是，即使在使用层平均的情况下，这两种方法仍然提高了 MLLMs 的准确率，这表明在没有选择层的数据时，取平均是一个合适的选择。关于高分辨率 ViCrop 的消融研究。在第 5 节中，我们提出了一种两阶段策略，用于处理 $\mathbf { V } ^ { * }$ 基准中的超高分辨率图像。为了评估这一策略的有效性，在表 3 中，我们比较了在 $\mathbf { V } ^ { * }$ 上使用和不使用此高分辨率策略的 ViCrop 方法的准确率。我们观察到，虽然该策略对 LLaVA-1.5 非常有益，但对 InstructBLIP 的 grad-att 和 pure-grad 性能有所下降。然而，所有方法在有无此策略的情况下仍然提高了 MLLMs 的准确率。带有外部工具的 ViCrop。除了内部的 ViCrop 方法外，我们还考虑使用外部现成模型找到图像中的兴趣区域进行视觉裁剪。具体而言，我们使用了 SAM（Kirillov 等，2023）、YOLO（Redmon 等，2016）和 CLIP（Radford 等，2021）来找到与给定问题最相关的图像部分（这些外部 ViCrop 方法的详细信息见附录 F）。在表 4 中，我们比较了外部 ViCrop 方法与内部方法在 TextVQA 上的准确率。尽管外部模型在提高 MLLMs 的准确率方面也有效，但它们的效果不及所有提出的内部 ViCrop 方法，因此我们没有进一步探索它们。

Table 4: Accuracy of ViCrop using external tools compared to attention/gradient (on TextVQA); and the inference time overhead of ViCrop methods (in seconds). Original's time is per answer token.   

<table><tr><td></td><td>Model</td><td>Original</td><td>SAM</td><td>YOLO</td><td>CLIP</td><td>rel-att</td><td>grad-att</td><td>pure-grad</td></tr><tr><td rowspan="2">Accuracy (TextVQÅ)</td><td>LLaVA-1.5</td><td>47.80</td><td>49.42</td><td>48.84</td><td>48.55</td><td>55.17</td><td>56.06</td><td>51.67</td></tr><tr><td>InstructBLIP</td><td>33.48</td><td>39.23</td><td>36.49</td><td>39.61</td><td>45.44</td><td>45.71</td><td>42.23</td></tr><tr><td rowspan="2">CPU Time</td><td>LLaVA-1.5</td><td>2.26</td><td rowspan="2">91.53</td><td rowspan="2">0.97</td><td rowspan="2">5.46</td><td>14.43</td><td>11.33</td><td>29.86 7.04</td></tr><tr><td>InstructBLIP</td><td>0.66</td><td></td><td>4.35</td><td>3.78</td></tr><tr><td rowspan="2">GPU Time</td><td>LLaVA-1.5</td><td>0.17 0.06</td><td rowspan="2">3.33</td><td rowspan="2">0.35</td><td rowspan="2">1.07</td><td>1.16 0.28</td><td>0.89 0.29</td><td>2.36</td></tr><tr><td>InstructBLIP</td><td></td><td></td><td></td><td>0.60</td></tr></table>

推理时间开销。在表 4 中，我们报告了所提视觉裁剪方法在 GPU（NVIDIA RTX A6000）和 CPU（Intel(R) Gold 5317 CPU @ 3.00GHz）上的平均推理时间开销，并与 MLLMs 的每个回答词元处理时间进行比较。我们看到，所有提议的方法（除 SAM 外）都相对快速（在 GPU 上的开销为 1 到 2 秒）。例如，使用 rel-att 进行视觉裁剪的时间仅相当于生成 5 个词元所需的时间。请注意，我们方法的时间开销不会随着回答词元数量的增加而变化，且无论回答多长都是常数，因为我们的外部方法不需要任何回答词元，内部方法只需起始回答词元（见第 5 节）。相比之下，MLLMs 的推理时间大致随着回答词元数量的增加而线性增长。

# 7 结论

在本工作中，我们定性和定量地展示了广泛使用的多模态语言模型（MLLMs）对于小视觉细节存在感知偏差。然后，我们发现即使MLLM无法回答问题，它们通常也知道该在哪里寻找，这表明对小视觉细节的偏见源于感知限制，而非定位限制。为了解决这一限制，我们提出了多种基于模型内部动态的自动视觉定位方法，作为可扩展且无需训练的解决方案，旨在回答视觉问题。通过对多个多模态基准的评估，我们证明了我们的方法能够显著提高MLLMs的准确性，而无需任何训练，特别是在对细节敏感的场景中。我们的发现表明，在对细节敏感的应用中应谨慎使用MLLMs，并且基于模型自身知识的视觉裁剪/定位是提高其性能的一个有前景方向。

限制与未来工作。所提议的 ViCrop 方法并未对所有类型的问题提供均等的增强效果。我们观察到，关于关系和计数的问题对于 ViCrop 方法尤其难以帮助回答。这是可以预期的，因为所提议的 ViCrop 只能关注图像中的一个区域。我们将 ViCrop 扩展到同时关注多个区域的工作留待未来进行。另一个限制是所提方法的时间开销和视觉词元的增加。尽管开销合理（几秒钟），我们认为可以作为推理时间机制进行显著优化，例如通过利用较低的精度和权重量化。此外，Matryoshka Query Transformer（MQT）（Hu et al., 2024）使得 MLLM 在推理过程中能够具有不同的视觉上下文大小。在当前的结果中，我们已经表明我们的方法能够与两种具有不同视觉上下文大小的 MLLM 一起工作，因此我们的算法在 MQT 下仍然可以适应不同的视觉上下文大小，这可能通过重新缩放裁剪图像进一步降低我们的计算成本。我们将这些推理成本的优化留待未来的工作。最后，我们观察到所提方法倾向于具有一些互补的优势，因此探索将它们结合的方式（例如基于预测不确定性）也是未来研究的一个非常有趣的方向。

# 鸣谢

我们感谢胡锦义和乔·马赛提供的宝贵见解。我们也对匿名评审的宝贵反馈表示感谢。本研究部分由美国国家科学基金会在合同号IIS-2153546支持下进行。

# REFERENCES

Anthropic. The claude 3 model family: Opus, Sonnet, Haiku, March 2024. URL https : / / www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/ Model_Card_Claude_3.pdf.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

Boyuan Chen, Zhuo Xu, Sean Kirmani, Brain Ichter, Dorsa Sadigh, Leonidas Guibas, and Fei Xia. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14455 14465, June 2024.

Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision-language models with less than $\$ 3$ https://github.com/Deep-Agent/ R1-V, 2025. Accessed: 2025-02-02.

Prateek Chhikara, Dhiraj Chaurasia, Yifan Jiang, Omkar Masur, and Filip Iievski. Fire: Food image to recipe generation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 81848194, 2024.

Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text generation. In International Conference on Machine Learning, pages 19311942. PMLR, 2021.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP Towards general-purpose vision-language models with instruction tuning, 2023.

Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023.

Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n'pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:22522274, 2023.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 69046913, 2017.

Jiaxian Guo, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Boyang Li, Dacheng Tao, and Steven Hoi. From images to textual prompts: Zero-shot visual question answering with frozen large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1086710877, 2023.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1495314962, 2023.

Wenbo Hu, Zi-Yi Dou, Liunian Harold Li, Amita Kamath, Nanyun Peng, and Kai-Wei Chang. Matryoshka query transformer for large vision-language models. arXiv preprint arXiv:2405.19315, 2024.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 67006709, 2019.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Glenn Jocher, Ayush Chaurasia, and Jing Qiu. YOLO by Ultralytics. January 2023. URL ht tps : //github.com/ultralytics/ultralytics.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large language-and-vision assistant for biomedicine in one day. arXiv preprint arXiv:2306.00890, 2023a.

Junnan Li, Ramprasaath Selvaraju, Akhilesh Gotmare, Shafiq Joty, Caiming Xiong, and Steven Chu Hong Hoi. Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems, 34:96949705, 2021.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 1288812900. PMLR, 2022a.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023b.

Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, et al. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1096510975, 2022b.

Xiang Li, Cristina Mata, Jongwoo Park, Kumara Kahatapitiya, Yoo Sung Jang, Jinghuan Shang, Kanchana Ranasinghe, Ryan Burgert, Mu Cai, Yong Jae Lee, et al. Llara: Supercharging robot learning data for vision-language policy. arXiv preprint arXiv:2406.20095, 2024b.

Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814, 2024c.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023c.

Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2676326773, 2024d.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740755. Springer, 2014.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023b.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024a. URL https : //llava-vl.github.io/blog/2024-01-30-llava-next/.

Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. In European Conference on Computer Vision, pages 126142. Springer, 2024b.

Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-of-resolution adaptation for multimodal large language models. arXiv preprint arXiv:2403.03003, 2024.

Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 22002209, 2021.

Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. Mm1: Methods, analysis & insights from multimodal llm pre-training. arXiv preprint arXiv:2403.09611, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 87488763. PMLR, 2021.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 779788, 2016.

Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In Computer Vision ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 2327, 2022, Proceedings, Part VIII, pages 146162. Springer, 2022.

Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pages 618626, 2017.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Unleashing chain-of-thought reasoning in multi-modal language models, 2024.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 83178326, 2019.

Hai-Long Sun, Da-Wei Zhou, Yang Li, Shiyin Lu, Chao Yi, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, et al. Parrot: Multilingual visual instruction tuning. arXiv preprint arXiv:2406.02539, 2024.

Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128, 2023.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Anthony Meng Huat Tiong, Junnan Li, Boyang Li, Silvio Savarese, and Steven CH Hoi. Plug-andplay vqa: Zero-shot vqa by conjoining large pretrained models with zero training. arXiv preprint arXiv:2210.08773, 2022.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. arXiv preprint arXiv:2208.10442, 2022.

Penghao Wu and Saining Xie. V\*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Ruyi Xu, Yuan Yao, Zonghao Guo, Junbo Cui, Zanlin Ni, Chunjiang Ge, Tat-Seng Chua, Zhiyuan Liu, Maosong Sun, and Gao Huang. Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images. arXiv preprint arXiv:2403.11703, 2024a.

Zhenhua Xu, Yujia Zhang, Enze Xie, Zhen Zhao, Yong Guo, Kwan-Yee K Wong, Zhenguo Li, and Hengshuang Zhao. Drivegpt4: Interpretable end-to-end autonomous driving via large language model. IEEE Robotics and Automation Letters, 2024b.

Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1380713816, 2024.

Jiarui Zhang, Filip lievski, Kaixin Ma, Aravinda Kollaa, Jonathan Francis, and Alessandro Oltramari. A study of situational reasoning for traffic understanding. KDD, 2023a.

Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, and Filip Ilievski. Visual cropping improves zero-shot question answering of multimodal large language models. In R0-FoMo: Robustness of Few-shot and Zero-shot Learning in Large Foundation Models, 2023b.

Jiarui Zhang, Jinyi Hu, Mahyar Khayatkhoei, Filip Iievski, and Maosong Sun. Exploring perceptual limitation of multimodal large language models. arXiv preprint arXiv:2402.07384, 2024a.

Jiarui Zhang, Ollie Liu, Tianyu Yu, Jinyi Hu, and Willie Neiswanger. Euclid: Supercharging multimodal llms with synthetic high-fidelity visual descriptions. arXiv preprint arXiv:2412.08737, 2024b.

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Ziyu Guo, Shicheng Li, Yichi Zhang, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, et al. Mavis: Mathematical visual instruction tuning with an automatic data engine. arXiv preprint arXiv:2407.08739, 2024c.

Tiancheng Zhao, Tianqi Zhang, Mingwei Zhu, Haozhan Shen, Kyusong Lee, Xiaopeng Lu, and Jianwei Yin. Vl-checklist: Evaluating pre-trained vision-language models with objects, attributes and relations. arXiv preprint arXiv:2207.00221, 2022.

# A ImPLeMentation Details

We use python 3.10.6, transformers 4.29.1 and torch 2.1.2 for all the experiments. Our environment consists of an Intel(R) Gold 5317 CPU $ @ ~ 3 . 0 0 \mathrm { G H z }$ with 48 cores and 756 GB of RAM, and we utilize NVIDIA RTX A6000 GPUs for our experiments. We use the huggingface implementations of all studied MLLMs with the recommended hyper-parameters according to the respective papers. For GPT-4o, we use the official public API, which is available at the time of submission.

Regarding the evaluation setting of the TextVQA dataset in Tab. 2, our setting is slightly different from the one used by the LLaVA-1.5 original paper Liu et al. (2023a). They report accuracy on TextVQA by using externally extracted OCR tokens to enrich its text prompt. This is a textspecific trick that essentially out-sources the perception of text to an external OCR model. This textspecific trick is not mentioned in their paper or supplementary material, but see their clarification in response to a GitHub issue here: https://github.com/haotian-liu/LLaVA/issues/ 515#issuecomment-1763779341. In contrast, we treat TextVQA the same as any other vision dataset in our experiments, that is, we do not provide any OCR extracted tokens to MLLMs when applying them to TextVQA (only image and question, in the evaluation prompt format specified in their respective papers). This results in a slightly lower accuracy compared to the one reported in the original paper, but instead, this number shows the true perception ability of LLaVA-1.5 on TextVQA, not confounded by the ability of an external OCR model. For completeness, we also measured TextVQA accuracy in the presence of OCR tokens, which results in 59.8 for LLaVA-1.5 without any visual cropping, and 63.95 with re1at t, showing that our proposed visual cropping can still be beneficial even when OCR tokens are provided to the MLLM.

# B DATASET STATISTICS

In this section, we present the details of the datasets used for evaluation in this paper. We report the average height and weight of the images in the dataset. We also report the number of images and questions in each dataset. For VQAv2, we run our experiment on a random 50K subset of the official validation set. We use the entire validation set in all other datasets.

Table 5: Average width $( \hat W )$ and height $( { \bar { H } } )$ of images, number of images, and number of questions on all datasets.   

<table><tr><td></td><td>V*</td><td>DocVQA</td><td>TextVQA</td><td>POPE</td><td>AOKVQA</td><td>GQA</td><td>VQAv2</td></tr><tr><td>¯W</td><td>2246</td><td>1776</td><td>954</td><td>584</td><td>581</td><td>578</td><td>577</td></tr><tr><td>•</td><td>1582</td><td>2084</td><td>818</td><td>478</td><td>480</td><td>482</td><td>485</td></tr><tr><td># Images</td><td>191</td><td>1286</td><td>3166</td><td>500</td><td>1122</td><td>398</td><td>14206</td></tr><tr><td># Questions</td><td>191</td><td>5349</td><td>5000</td><td>8910</td><td>1145</td><td>10781</td><td>50000</td></tr></table>

For our analysis presented in Table 1 and Figure 3, we focused on TextVQA dataset, which includes bounding box annotations for OCR-detected text within images. However, this dataset does not specify which bounding boxes correspond to the regions where answers are located, necessitating a manual annotation process. The TextVQA dataset comprises 5000 questions and 3166 images. We manually annotated these question-image pairs, ensuring accurate bounding boxes over all the regions of interest where the answers could be found. This manual annotation process was essential for our analysis, allowing us to provide precise and reliable ground-truth data for the study. Given that some questions were associated with multiple bounding boxes in their corresponding images, we undertook a filtering process to isolate the question-image pairs. This effort resulted in a refined set of 4370 question-image pairs, where there is only one instance of the subject of the question in the image. For example, if the question is "what type of drink is sold here?" and there are two different cans of drinks in the image, we remove this image-question pair.

# C PRompt FoRmat for Zero-sHot InfEREnCe

In this section, we provide details about the prompt format used in models for zero-shot inference. We use a different prompt format for LLaVA and InstructBLIP which we adapt from the original papers, as shown below.

# LLaVA-1.5

<image> USER:{question} Answer the question using a single word or phrase. ASSISTANT:

# InstructBLIP

<image> Question:{question} Short Answer:

# D ORTHogoNAL BENEFITS To LLAVA-NEXT

We apply our proposed re1-at t visual cropping method to an additional newer MLLM  LLaVA-NeXT (Liu et al., 2024a) current SOTA in several VQA benchmarks  that has support for higherresolution compared to LLaVA-1.5. In Tab. 6, we observe that our method can still boost the MLLM's performance, without requiring any training. This provides further evidence for the generalizability of our proposed visual cropping and its orthogonal benefits to training MLLMs with higher image patch resolution.

Table 6: Orthogonal benefits of visual cropping when applied to LLaV-NeXT that is trained to adapt to processing high-resolution images.   

<table><tr><td>Model</td><td>TextVQA</td><td>V*</td></tr><tr><td>LLaVA-NeXT (Mistral-7B)</td><td>65.17</td><td>58.11</td></tr><tr><td>LLaVA-NeXT (Mistral-7B) + re1-att</td><td>68.65</td><td>61.78</td></tr></table>

# E COMPArISON WITH THE ${ \mathrm { V } } ^ { * }$ METHOD (SEAL)

The $\mathrm { V } ^ { \ast }$ method (SEAL) (Wu and Xie, 2023) proposes a multi-agent fine-tuning approach to enhance the ability of an underlying MLLM to answer questions about small visual concepts. However, SEAL requires substantial training and finetuning of several neural networks, whereas our methods are completely training-free, so a direct comparison would not be fair. Nonetheless, to provide an idea of how our method compares to SEAL in an "as-is" fashion (i.e. if a user just wants to pick one method as-is off-the-shelf), we report the accuracy of SEAL compared to LLaVA-1.5+re1-at t in Tab. 7. We observe that our method outperforms SEAL except on the $\mathbf { V } ^ { \ast }$ benchmark. We think this might be because SEAL is designed and tuned specifically toward high-resolution images in its $\mathrm { V } ^ { \ast }$ benchmark. We also note that the inference time of SEAL is slower than our method (4.44s compared to 1.88s on average per question, tested on the same random 100 TextVQA samples with one A6000 GPU). That being said, we note that our methods and SEAL can both help enhance MLLMs, and our methods can be integrated into SEAL or other multi-agent pipelines.

Table 7: Performance comparison between our re1-at t applied on LLaVA-1.5 and SEAL (Wu and Xie, 2023) across multiple vision-language benchmarks.   

<table><tr><td>Model</td><td>TextVQA</td><td>V*</td><td>POPE</td><td>DocVQA</td><td>AOKVQA</td><td>GQA</td><td>VQAV2</td></tr><tr><td>SEAL</td><td>36.30</td><td>75.30</td><td>82.40</td><td>5.31</td><td>55.34</td><td>50.18</td><td>65.35</td></tr><tr><td>LLaVA-1.5+rel-att</td><td>55.17</td><td>62.30</td><td>87.25</td><td>19.63</td><td>60.66</td><td>660.97</td><td>76.29</td></tr></table>

# F External Tools ViCRop

In this section, we present three automatic question-guided localization methods based on popular off-the-shelf vision-based models, namely CLIP Radford et al. (2021), YOLO Redmon et al. (2016), and SAM Kirillov et al. (2023). These three methods utilize external vision-based knowledge for the localization process through multimodal encoding, object detection, and semantic segmentation, respectively. See Tab. 4 for their results compared to internal ViCrop methods.

CLIP ViCrop. The intuition of this method is to progressively refine the image towards the region of highest relevance to a given question using CLIP Radford et al. (2021). CLIP consists of an image encoder and a text encoder, which are trained on a large dataset of image-caption pairs to map each image (caption) close to its caption (image) and far from all other captions (images). The result is an aligned shared space where various images can be directly compared with various texts. To find the region of interest, given an image-question pair, we first crop the image from the four sides (top, bottom, left, and right) at a cropping ratio of 0.9 to produce four overlapping cropped images. We then use CLIP to assess the semantic similarity between these cropped images and the question. The highest-scoring crop is chosen as the input for the next iteration. This process is repeated for 20 iterations, and the cropped image with the highest CLIP similarity to the question is selected for visual cropping.

YOLO ViCrop. Instead of a progressive approach to finding the region of interest, in this method we select candidate regions based on a state-of-the-art object detection method: YOLOv8 (Jocher et al., 2023) pretrained on COCO Lin et al. (2014). Using YOLO, we filter out regions that contain no salient objects  i.e., regions for which CLIP could mistakenly assign high similarity. More concretely, for each question-image pair, we first use YOLO to collect bounding boxes for al predicted objects with confidence higher than 0.25 (the recommended default).5 Then, for each predicted bounding box, we crop its corresponding image and compute its similarity to the question using CLIP. Finally, the bounding box with the highest similarity score is selected as the region of interest for visual cropping.

SAM ViCrop. A limitation of YOLO is that it only provides bounding boxes corresponding to a fixed number of object classes. To overcome this issue, we use the segment anything model (SAM) Kirillov et al. (2023), which has shown state-of-the-art zero-shot segmentation performance. SAM can provide an extensive set of segmentation masks for each image, thus providing a more granular set of salient candidate regions compared to YOLO. More concretely, for each image-question pair, we feed the image into SAM, which provides an extensive set of segmentation masks corresponding to all objects and object parts. Then, we translate these masks into bounding boxes by computing the smallest bounding box that covers each segmentation mask. Finaly, the bounding box with the highest CLIP similarity to the question is selected as the region of interest for visual cropping.

Finally, for each method, we crop the smallest covering square (so that the cropped image is not deformed when resized to the input resolution of the MLLM), and provide it to the MLLM in addition to the original image-question pair (as depicted in Fig. 4).

# G Additional Examples on ModeL's Predictions

![](images/6.jpg)  
Figure 6: Success (first 3) and failure (last) examples of LLaVA-1.5 (re1-at t) on the $\mathrm { V } ^ { \ast }$ benchmark (cyan-colored bounding box shows cropped region by rel-att; zoom-in insets are displayed for better readability).

![](images/7.jpg)  
Figure 7: Success (first 9) and failure (last 6) examples of LLaVA-1.5 (re1-at t) on the TextVQA benchmark (cyan-colored bounding box shows cropped region by rel-at t ).

![](images/8.jpg)  
Figure 8: Success (first 9) and failure (last 6) examples of InstructBLIP (re1att) on the TextVQA benchmark (cyan-colored bounding box shows cropped region by rel-at t ).