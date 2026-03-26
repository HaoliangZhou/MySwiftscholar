# ADAST: 自适应语义变换视觉表征用于无训练的零-shot复合图像检索

匿名作者论文正在进行双盲审稿。

# 摘要

组合图像检索（CIR）根据参考图像和文本修改检索目标图像。说明文明确了预期的变化，同时保留其他视觉属性以确保一致性。近期的研究探索了无训练方法，通过将参考图像与文本修改相结合合成代理图像。然而，这些方法计算开销大且耗时，而仅依赖文本查询往往会导致关键信息的视觉细节丢失。为了解决这些问题，我们提出了自适应语义转换（AdaST），这是一种新的无训练方法，通过文本指导将参考图像特征转换为代理特征。与生成图像不同，AdaST高效地通过特征级转换保留视觉信息。为了实现更细粒度的转换，我们引入了一种自适应加权机制，平衡代理特征和文本特征，使模型在可靠时仅利用代理信息。我们的方法轻量且能够以即插即用的方式无缝应用于现有的无训练基线。大量实验表明，该方法在三个CIR基准上实现了最先进的性能，同时避免了图像生成的高额成本，并且与基于文本的基线相比，仅增加了少量的推理开销。

# 1 引言

组合图像检索（CIR）是指在给定参考图像和文本修改指令的情况下检索目标图像的任务。（Vo 等，2019）核心挑战在于多模态理解和组合推理，因为系统必须准确整合指明所需修改的文本线索与保持参考图像不变细节的视觉线索。最近，CIR 引起了广泛关注，因为它提供了一种自然直观的界面，用于探索大规模图像集合，超越了传统的基于关键字的搜索。这一能力在时尚电子商务和在线搜索引擎等领域尤其重要（Wu 等，2021；Tian 等，2023），用户可能提供参考产品图像，并要求修改，例如“相同的衬衫但要红色”。更广泛地说，CIR 代表了推进视觉语言理解的一个基本步骤，因为它需要对异质模态进行对齐，并进行连接视觉与文本信息的细粒度推理。

尽管有强烈的实际需求，许多专有数据集通常被保留在内部，并未与外部开发者共享（Zhang et al., 2024; Kolouju et al., 2025）。收集带标签的CIR数据，包含带有修改指令的参考图像和目标图像，成本高且劳动密集（Liu et al., 2021; Wu et al., 2021）。这限制了可扩展性和对未见领域的泛化，促使基于无标注预训练模型的零-shot CIR（ZS-CIR）方法的出现。早期的ZS-CIR工作（Karthik et al., 2023; Yang et al., 2024b;a; Saito et al., 2023; Baldrati et al., 2023; Gu et al., 2024）将CIR视为文本到图像检索任务，通过将参考图像和修改指令编码为单一文本表示。虽然有效，但这一策略忽视了细粒度线索，常导致在时尚领域中产生语义正确但视觉不匹配的结果。最近的研究（Li et al., 2025; Zhou et al., 2024）通过条件生成合成修改后的图像，并将其作为检索输入。尽管这保留了视觉细节，但在高维像素空间中的生成成本高昂，通常超过每幅图像30秒，超出了互动检索的可用性阈值（Nielsen, 1994）。

![](images/1.jpg)  

Figure 1: Comparison of existing input-level approaches and AdaST. Given a reference image and a textual instruction (left), prior approaches encode modifications at the input level using text or image generation (midle), which often sacrifice either visual integrity or efficiency. In contrast, AdaST applies instruction-guided transformations directly in the feature space of pretrained VLMs (right), preserving visual details while remaining efficient.

受限于这些局限性，我们提出了自适应语义转换（AdaST），这是一种无需训练的零样本图像转化（ZS-CIR）方法，能够保留视觉细节，同时高效且独立于外部生成模型。AdaST不在像素空间中操作或强行将图像转换为文本空间，而是在预训练的视觉语言模型（VLM）的特征空间中直接进行指令引导的转换（Radford等，2021），如图1所示，受到特征级编辑策略的启发（Kwon & Ye，2022；Ye-Bin等，2023）。在大语言模型（LLM）生成的标题的指导下，我们的方法从文本中导出语义变化，并将其转移到图像嵌入中，产生更好逼近目标的代理特征。为了确保鲁棒性，我们进一步引入了一种自适应相似性机制，平衡代理特征和基于文本特征的贡献，使模型能够在不依赖计算成本高昂的图像生成的情况下保留细粒度线索。在三个CIR基准上的大量实验证明，AdaST不仅实现了最先进的性能，而且仍然高度高效。特别是，AdaST在CIRCO数据集上相对于ViT-G基线提升了$+ 3.47 \mathrm{mAP}@5$的性能。此外，其运行速度比最先进的方法IP-CIR（Li等，2025）快$186 \times$，同时仍然实现了更高的准确性。最后，我们的方法可以以即插即用的方式无缝应用于多种模型，始终提升它们的性能。我们的贡献总结如下：• 我们提出AdaST，这是一种无需训练的零样本图像转化方法，通过引导LLM衍生的文本变化将参考图像嵌入转换为代理特征，从而保留细粒度的视觉细节，而不依赖生成模型。我们引入了一种自适应相似性机制，动态平衡代理相似性和基于文本的相似性，使模型在可靠时能够利用代理特征，同时通过文本对齐确保鲁棒性。• 大量实验证明，AdaST在多个CIR基准上实现了最先进的性能，同时比基于生成的方法快得多且更轻量。

# 2 相关工作

# 2.1 复合图像检索

复合图像检索（CIR）是指在给定参考图像和文本指令的情况下检索目标图像的任务（Vo等, 2019）。该任务在现实应用中尤为重要，例如时尚电子商务和在线搜索引擎。之前的研究主要集中在设计模型，通过对比学习使得图像-文本对在共享嵌入空间中对齐（Vo等, 2019；Chen & Bazzani, 2020；Lee等, 2021），或者采用跨模态注意力机制以捕捉组合关系（Delmas等, 2022）。然而，这些工作依赖于使用特定任务数据集的监督学习（Wu等, 2021；Liu等, 2021；Baldrati等, 2023），其大规模构建成本高昂，限制了可扩展性和泛化能力。

# 2.2 零样本组合图像检索

ZS-CIR 方法旨在通过利用大规模预训练的视觉语言模型（VLMs）降低数据集构建的成本，无需在特定于CIR的数据集上进行训练，以便处理未见过的图像。早期的方法将参考图像和修改文本编码为单一的文本表示，通常通过文本反演或基于大语言模型（LLM）的方法实现。文本反演方法训练一个反演模型，将图像映射到词元空间，然后与修改指令结合以便提取。基于LLM的方法则是将图像描述为自然语言，并让大语言模型联合推理参考描述与修改文本。然而，两种方法都不可避免地将视觉信息压缩为文本形式，丢失了参考图像中的细粒度细节。最近的研究直接使用条件生成模型从参考图像和文本指令合成修改后的图像，保持了视觉细节，但在高维像素空间生成的成本较高，因为依赖于大规模生成模型。基于这一限制，我们提出了一种高效的替代方案，能够在不依赖昂贵图像生成的情况下保持视觉保真度。

# 2.3 文本引导的语义转化

文本引导的语义转换方法（Fu et al., 2022；Kwon & Ye, 2022；Gal et al., 2022；Ye-Bin et al., 2023；Park et al., 2025）利用预训练视觉语言模型（VLMs）如 CLIP 的潜在空间，将图像特征与文本指导对齐。这些方法假设图像和文本嵌入在一个联合特征空间中，其中它们的表示在语义上是对齐的。在该空间中，源文本特征与目标文本特征之间的差异向量可以应用于相应的图像特征，从而生成转换后的图像表示。一些CIR方法（Vo et al., 2019；Li et al., 2025）也采用了这个想法，通过直接将文本特征差异向量应用于图像特征，但这种直接转移的效果有限。为了解决这一局限性，我们提出了一种新颖的重标定策略，该策略保留文本特征差异向量的方向，同时调整其在图像特征空间中的幅度。这将产生更加真实的转换图像特征，并使组合图像检索既有效又高效。

# 3 方法

我们提出了 AdaST（自适应语义转换），这是一种无训练的零-shot 组合图像检索方法。给定参考图像 $I_{r}$、文本指令 $T_{\mathrm{inst}}$ 和图像数据库 $\mathbf{\check{\mathcal{D}}} = \{ I_{i}^{\mathrm{DB}} \}_{i=1}^{N}$，$\boldsymbol{I}_{t} \in \mathcal{D}$ 由该指令指定。我们的方法由三个主要组件组成。首先，在第 3.1 节中，我们生成文本指导，形式为参考标题和目标标题，使用了标题生成模型和大语言模型。其次，我们引入了一种文本引导的语义转换，能够将标题之间的语义转换映射到图像嵌入空间中，以构建代理嵌入，在第 3.2 节中进行详细介绍。最后，我们设计了一个自适应相似性融合模块，配备门控机制，能够选择性地结合基于代理的相似性，从而在第 3.3 节中实现语义对齐与视觉线索的平衡，确保可靠的检索。

![](images/2.jpg)  

Figure 2: Overall pipeline of AdaST. It consists of three stages. (1) Text guidance generation: a reference caption is obtained from the input image using a captioning model, and an LLM combines it with the textual instruction to generate a target caption. (2) Text-guided semantic transformation: both captions and the reference image are embedded with CLIP, where the feature difference between the reference and target captions is transferred to the reference image feature with a scaling factor, yielding a proxy feature. (3) Adaptive similarity fusion: an adaptive gating mechanism fuses proxy similarity with text-based similarity, allowing proxy similarity to contribute only when supported by consistent semantic cues.

# 3.1 文本引导生成

我们首先构建文本指导，这作为视觉参考与文本指令之间的语义桥梁，从而促进对修改目标图像的准确和稳健的检索。该指导包括两个组成部分：描述参考图像的参考标题 $T _ { r }$ 和基于 $T _ { r }$ 和 $T _ { \mathrm { i n s t } }$ 条件编码的意图修改的目标标题 $T _ { t }$。参考标题 $T _ { r }$ 是通过将参考图像 $I _ { r }$ 输入最近的标题生成模型（如 BLIP-2（Li et al., 2023））获得的。然后，我们使用基于最近大语言模型（LLM）的方法（Karthik et al., 2023）生成以 $T _ { r }$ 和 $T _ { \mathrm { i n s t } }$ 为条件的目标标题 $T _ { t }$。目标标题至关重要，因为它明确编码了指令所指定的语义变化，确保检索强调意图修改，而不仅仅是与参考图像的视觉相似性。

# 3.2 文字引导的语义转换

我们提出了一种文本引导的语义变换，将文本空间中的语义变化转移到图像空间，以构建代理嵌入 $f _ { P }$。该方法是无训练的，直接在特征空间中操作，能够高效检索而无需耗费成本的图像生成，同时保留视觉信息。具体而言，我们使用预训练的视觉语言模型（如 CLIP (Radford et al., 2021)）中的文本编码器 $E _ { T }$ 和图像编码器 $E _ { I }$ 将标题和图像嵌入到联合表示空间中。由此产生以下嵌入：

$$
f _ { T _ { r } } = E _ { T } ( T _ { r } ) \ : , \ : \ : f _ { T _ { t } } = E _ { T } ( T _ { t } ) \ : , \ : \ : f _ { I _ { r } } = E _ { I } ( I _ { r } ) \ : .
$$

正式地，目标字幕与参考字幕之间的语义偏移定义为

$$
\Delta T = f _ { T _ { t } } - f _ { T _ { r } } \ .
$$

相应的代理嵌入定义为，其中 $\alpha$ 是一个缩放因子，用于控制语义转换的强度。

$$
f _ { P } ^ { ( \alpha ) } = f _ { I _ { r } } + \alpha \Delta T ,
$$

最优缩放 选择像 $\underline{ {\alpha = 1} }$ 的简单缩放因子往往会导致次优行为：代理嵌入过于接近参考图像嵌入，从而无法充分捕捉预期的修改，如第 4.2 节所示。为了解决这个问题，我们提出了一种基于优化的方法来获取 $\alpha$ 的最优值。关键原则是代理嵌入 $f _ { P } ^ { ( \alpha ) }$ 应与目标字幕嵌入 $f _ { T _ { t } }$ 良好对齐，确保预期的修改得到准确表达。同时，它应与参考嵌入 $f _ { I _ { r } }$ 保持足够的区分，以防止检索回到原始视觉内容。为了编码这些要求，我们 formulates 了优化问题：

$$
\alpha ^ { * } = \underset { \alpha } { \arg \operatorname* { m i n } } \left( 1 - \sin ( f _ { P } ^ { ( \alpha ) } , f _ { T _ { t } } ) + \beta \cdot \sin ( f _ { P } ^ { ( \alpha ) } , f _ { I _ { r } } ) \right) ,
$$

其中，$\beta$ 是一个权重系数，用于控制罚项的影响，$\mathrm{sim}(\cdot, \cdot)$ 表示余弦相似度。第一个项鼓励与目标描述的对齐，而第二个项则惩罚与参考图像的过度相似。该目标具有封闭形式的解，可以直接通过内积计算得出：

$$
\alpha ^ { * } = \frac { \boldsymbol { x } ^ { \top } \boldsymbol { y } \cdot \boldsymbol { x } ^ { \top } d - d ^ { \top } \boldsymbol { y } \cdot \| \boldsymbol { x } \| ^ { 2 } } { d ^ { \top } \boldsymbol { y } \cdot \boldsymbol { x } ^ { \top } d - \boldsymbol { x } ^ { \top } \boldsymbol { y } \cdot \| d \| ^ { 2 } } ,
$$

其中 $x = f _ { I _ { r } }$，$y = \tilde { f } _ { T _ { t } } - \beta \tilde { f } _ { I _ { r } }$，$d = \Delta T$。这里，$\tilde { f } = f / \| f \| _ { 2 }$。得到的代理嵌入 $f _ { P } ^ { ( \alpha ^ { * } ) }$。

# 3.3 自适应相似性融合

虽然代理嵌入可以通过测量其与候选图像特征的相似性直接用于检索，但单靠基于代理的相似性引入了一个缺点。由于其主要捕捉视觉线索，代理可能会给视觉上相似但与指令语义无关的图像分配高分。为了解决这个问题，我们通过目标标题特征 $f _ { T _ { t } }$ 引入语义指导，该特征编码了丰富的语义信息。我们计算目标标题特征与候选特征之间的目标标题相似性，然后将该分数与基于代理的相似性融合。具体而言，我们提出了一种受到 (Yang et al., 024b;a) 启发的门控机制，能够自适应调整代理相似性的贡献。该门控策略确保仅在得到语义证据支持时，基于代理的相似性才会影响检索，从而减少由纯粹视觉相似性导致的误报。为实现这一点，我们首先提取数据库的特征。

$$
f _ { I _ { i } ^ { \mathrm { D B } } } = E _ { I } ( I _ { i } ^ { \mathrm { D B } } ) , \ \forall i = \{ 1 , \dots , N \} \ .
$$

为简便起见，令 $f _ { I ^ { \mathrm { D B } } }$ 表示所有数据库嵌入的集合。对于每个查询，我们接着计算三个相似度得分：

$$
S _ { T _ { t } } = \sin ( f _ { T _ { t } } , f _ { I ^ { \mathrm { D B } } } ) , S _ { T _ { r } } = \sin ( f _ { T _ { r } } , f _ { I ^ { \mathrm { D B } } } ) , S _ { P } = \sin ( f _ { P } , f _ { I ^ { \mathrm { D B } } } ) .
$$

所提议的门控函数定义为，其中 $\lambda$ 是一个权重系数，控制 $S_{P}$ 的影响，而 $m$ 是一个边缘值，用于确定语义对齐是否已实现。只有当目标标题显示出比参考标题更大的语义对齐时，门才会激活，从而确保基于代理的相似度仅在有支持性证据的情况下被纳入。

$$
G ( \Delta S _ { T } ) = \left\{ \begin{array} { l l } { { \lambda , } } & { { \Delta S _ { T } + m \ge 0 } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e } } } \end{array} , \right. \ \Delta S _ { T } = S _ { T _ { t } } - S _ { T _ { r } } \ ,
$$

根据该规定，最终相似度评分由以下公式给出，其中 $S _ { A }$ 表示对代理相似度 $S _ { P }$ 的自适应权重。最后，通过选择具有最大相似度评分的数据库图像来获得检索结果：

$$
S _ { \mathrm { t o t a l } } = S _ { A } \cdot S _ { P } + S _ { T _ { t } } , \quad S _ { A } = S _ { T _ { t } } \cdot G ( \Delta S _ { T } ) ,
$$

$$
\begin{array} { r } { I _ { t } = \arg \operatorname* { m a x } _ { { \cal { S } } _ { \mathrm { t o t a l } } } . } \\ { I _ { i } ^ { \mathrm { D B } } { \in } \mathcal { D } } \end{array}
$$

Table 1: Quantitative results on the CIRCO and CIRR benchmarks using three backbones (ViTB/32, ViT-L/14, and ViT-G/14), where our method is applied on top of two representative baselines (CIReVL (Karthik et al., 2023) and SEIZE (Yang et al., 2024a)). Across all three architectures and both benchmarks, combining our approach with the baselines yields consistent improvements and achieves state-of-the-art performance on most metrics. $\dagger$ represents our reproduced results.

<table><tr><td colspan="4">Benchmark</td><td colspan="3">CIRCO (mAP@K)</td><td colspan="4">CIRR (Recall@K)</td><td colspan="2">CIRR (Recallsubset@K)</td><td colspan="2"></td></tr><tr><td>Backbone</td><td>SEARLE</td><td>Method</td><td>k=5</td><td>k=10</td><td>k=25</td><td>k=50</td><td>k=1</td><td>k=5</td><td>k=10</td><td>k=50</td><td>k=1</td><td>k=2</td><td>k=3</td></tr><tr><td rowspan="7">ViT-B/32</td><td>CIReVL†</td><td>ICCV23 ICLR24</td><td>9.35 13.28</td><td>9.94 13.69</td><td>11.13 15.13</td><td>11.84 15.94</td><td>24 20.84</td><td>53.42</td><td>66.82 60.19</td><td>59.78 84.70</td><td>54.89</td><td>76.60</td><td>88.19</td></tr><tr><td>LDRE</td><td>SIGIR24</td><td>17.96</td><td>18.32</td><td>20.21</td><td>21.11</td><td>25.69</td><td>46.96 55.13</td><td></td><td>89.9</td><td>54.30 60.53</td><td>76.5</td><td>88.10</td></tr><tr><td>SSEIZE*</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>69.04</td><td></td><td></td><td>80.65</td><td>90.7</td></tr><tr><td></td><td>ACMMM24</td><td>18.75</td><td>19.37</td><td>21.09</td><td>22.07</td><td>26.96</td><td>55.59</td><td>68.24</td><td>88.34</td><td>66.82</td><td>85.23</td><td>93.35</td></tr><tr><td>OSrCIR</td><td>CVPR25</td><td>18.04</td><td>19.17</td><td>20.94</td><td>21.85</td><td>25.42</td><td>54.54</td><td>68.19</td><td>-</td><td>62.31</td><td>80.86</td><td>91.13</td></tr><tr><td>CIReVL† + Ours</td><td></td><td>15.20</td><td>15.73</td><td>17.25</td><td>18.12</td><td>25.23</td><td>52.41</td><td>64.48</td><td>85.35</td><td>60.12</td><td>78.96</td><td>89.11</td></tr><tr><td>SEIZE† + Ours</td><td></td><td>21.16</td><td>21.89</td><td>23.76</td><td>24.62</td><td>30.15</td><td>59.71</td><td>72.60</td><td>89.81</td><td>66.72</td><td>84.94</td><td>93.45</td></tr><tr><td rowspan="9">ViT-L/14</td><td>Pic2Word</td><td>CVPR23</td><td>8.72</td><td>9.51</td><td>10.64</td><td>11.29</td><td>23.9</td><td>51.7</td><td>65.3</td><td>87.8</td><td>-</td><td>-</td><td></td></tr><tr><td>SEARLE</td><td>ICCV23</td><td>11.68</td><td>12.73</td><td>14.33</td><td>15.12</td><td>24.24</td><td>52.48</td><td>66.29</td><td>88.84</td><td>53.76</td><td>75.01</td><td>88.19</td></tr><tr><td>LinCIR</td><td>CVPR24</td><td>12.59</td><td>13.58</td><td>15.00</td><td>15.85</td><td>25.04</td><td>53.25</td><td>66.68</td><td>-</td><td>57.11</td><td>77.37</td><td>88.89</td></tr><tr><td>CREeVLt</td><td>ICLR24</td><td>16.54</td><td>17.42</td><td>19.27</td><td>20.22</td><td>21.28</td><td>47.47</td><td>60.6</td><td>83.4</td><td>54.5</td><td>75.28</td><td>87.88</td></tr><tr><td>LDRE</td><td>SIGIR24</td><td>23.35</td><td>24.03</td><td>26.44</td><td>27.5</td><td>26.53</td><td>55.57</td><td>67.54</td><td>88.5</td><td>60.43</td><td>80.31</td><td>89.9</td></tr><tr><td>SEIZE</td><td>ACMMM24</td><td>24.71</td><td>25.52</td><td>27.99</td><td>29.03</td><td>28.43</td><td>56.53</td><td>69.88</td><td>88.17</td><td>66.43</td><td>84.68</td><td>92.96</td></tr><tr><td>OSrCIR</td><td>CVPR25</td><td>23.87</td><td>25.33</td><td>27.84</td><td>28.97</td><td>29.45</td><td>57.68</td><td>69.86</td><td></td><td>62.12</td><td>81.92</td><td>91.10</td></tr><tr><td>LDRE + IP-CIR</td><td>CVPR25</td><td>26.43</td><td>27.41</td><td>29.87</td><td>31.07</td><td>29.76</td><td>58.82</td><td>71.21</td><td>90.41</td><td>62.48</td><td>81.64</td><td>90.89</td></tr><tr><td>CIReVL† + Ours</td><td></td><td>20.32</td><td>20.92</td><td>22.81</td><td>23.71</td><td>25.35</td><td>52.92</td><td>66.41</td><td>86.89</td><td>60.75</td><td>80.77</td><td>90.92</td></tr><tr><td rowspan="8"></td><td>SEIZE† + Ours</td><td></td><td>28.94</td><td>29.65</td><td>32.04</td><td>33.03</td><td>30.72</td><td>59.78</td><td>71.13</td><td>88.68</td><td>67.21</td><td>84.96</td><td>93.04</td></tr><tr><td>LinCIR</td><td>CVPR24</td><td>19.71</td><td>21.01</td><td>23.13</td><td>24.18</td><td>35.25</td><td>64.72</td><td>76.05</td><td>-</td><td>63.35</td><td>82.22</td><td>91.98</td></tr><tr><td>CIReVL†</td><td>ICLR24</td><td>26.47</td><td>27.46</td><td>29.91</td><td>30.86</td><td>30.7</td><td>59.66</td><td>70.89</td><td>89.86</td><td>63.54</td><td>82.02</td><td>91.52</td></tr><tr><td>LDRE ViT-G/14 SEIZE†</td><td>SIGIR24 ACMMM24</td><td>31.12 35.61</td><td>32.24 36.92</td><td>34.95</td><td>36.03</td><td>36.15</td><td>66.39</td><td>77.25</td><td>93.95</td><td>68.82</td><td>85.66</td><td>93.76</td></tr><tr><td>OSrCIR</td><td>CVPR25</td><td>30.47</td><td>31.14</td><td>39.67 35.03</td><td>40.61 36.59</td><td>40.87 37.26</td><td>69.52 67.25</td><td>78.94 77.33</td><td>92.27</td><td>75.04 69.22</td><td>90.31 85.28</td><td>96.02</td></tr><tr><td>LDRE + IP-CIR</td><td>CVPR25</td><td>32.75</td><td>34.26</td><td>36.86</td><td>38.03</td><td>39.25</td><td>70.07</td><td>80.00</td><td>- 94.89</td><td>69.95</td><td>86.87</td><td>93.55</td></tr><tr><td>CIReVL† + Ours</td><td></td><td>32.32</td><td>33.49</td><td>35.98</td><td>36.81</td><td>35.04</td><td>65.06</td><td>75.98</td><td>91.57</td><td>65.57</td><td></td><td>94.22</td></tr><tr><td>SEIZE† + Ours</td><td></td><td>39.08</td><td>39.93</td><td>42.53</td><td>43.34</td><td>42.84</td><td>72.29</td><td>80.82</td><td>93.28</td><td>74.82</td><td>83.52 89.95</td><td>92.53 96.00</td></tr></table>

# 4 实验

CIRR（Liu等，2021）由21,552张图像组成，这些图像来自NLVR数据集（Suhr等，2018），并附有36,554个相关查询。该数据集旨在支持细粒度自然语言的修改，使得基于图像之间微妙语义差异的检索成为可能。CIRR的一个局限性是可能存在假阴性，因为图库中的多个图像可能符合同一指令，但仅有一个被标注为真实标注数据。CIRCO（Baldrati等，2023）是一个基准数据集，构建于C0C0 2017（Lin等，2014），明确解决这个假阴性问题，通过为每个查询提供多个标注目标图像。它包括一个包含220个查询的验证集和一个包含800个查询的测试集，涵盖涉及属性编辑、对象替换和风格修改的指令，这些内容对组合推理尤其具有挑战性。Fashion-IQ（Wu等，2021）是一个针对时尚检索的领域特定基准，包含30,135个查询和77,683个产品图像，分为三类：衬衫、连衣裙和上衣。其查询由标注者编写，描述对参考服装的修改。

在评估中，我们遵循每个数据集的官方协议。CIRR的评估使用Recall $@ \mathrm { K }$ $( \mathbf { K } \in \{ 1 , 5 , 1 0 , 5 0 \} )$和RecallSubset $@ \mathrm { K }$ $( \mathbf { K } \in \{ 1 , 2 , 3 \} )$，后者仅关注与参考和目标在同一语义集中的图像，从而捕捉在紧密相关的干扰组下的检索性能。CIRCO的评估使用前K个结果的平均准确率 $( \mathrm { m A P } @ \mathrm { K }$ $\mathsf { K } \in \{ 5 , 1 0 , 2 \bar { 5 } , 5 \bar { 0 } \} )$，因为它具有多个正目标。Fashion-IQ则对每个服装类别和它们的平均值使用Recall $@ 1 0$和Recall $\textcircled { a } 5 0$进行评估。

实施细节。对于检索模型，我们采用包括 ViT-B/32、ViT-L/14 和 ViT-G/14 的 CLIP 主干网络。我们默认使用 OpenAI (Radford 等, 2021) 的官方权重，而对于 ViT-G/14，我们使用 OpenCLIP (llharco 等, 2021) 的权重。我们使用 CIReVL (Karthik 等, 2023) 和 SEIZE (Yang 等, 2024a) 作为基准模型进行比较。对于 Fashion-IQ 数据集，我们额外采用 LinCIR (Gu 等, 2024) 作为我们的基准，并应用基于 LLM 的标题生成过程。在图像描述方面，我们采用 BLIP-2 (Li 等, 2023) 模型。关于 LLM 模型，基准代码库使用 GPT-3.5-turbo，但该版本已不可用。因此，根据最新研究 (Tang 等, 2025)，我们使用 GPT4o 重新实现了基准，确保我们的方法和基准在相同模型下进行评估，以实现公平比较。所有实验均在单个 A6000 GPU 上进行。对于所有基准和数据集，我们设定 $\beta = 0.25$、$\lambda = 4$ 以及 $m = 0.1$，其中 CIRCO 数据集使用 $m = 0$。

![](images/3.jpg)  

Figure 3: Qualitative comparison between CIReVL and our method on three benchmarks (CIRCO, CIRR, and Fashion-IQ). Given a reference image and an instruction, reference and target captions are generated, and the top-5 retrieved images from each method are shown, with ground-truth targets highlighted in green. Our method leverages visual features more effectively, enabling accurate retrieval even when the target caption is underspecified, by exploiting fine-grained details such as dog breeds or dress shapes.

# 4.1 与最先进方法的比较

CIRCO 和 CIRR 如表 1 所示，我们的方法在所有主干网络上均优于 CIReVL 和 SEIZE 基线，在大多数指标上实现了最先进的性能。这些改进在 ViT-G/14 等大主干网络上尤其显著，展示了 AdaST 的可扩展性。在 CIRCO 基准上，我们的方法取得了显著的改善。例如，在 ViT-G/14 主干网络下，我们的方法在 $+ 5 . 8 5 \ \mathrm { m A P } @ 5 $ 超过了 CIReVL 基线，而这一提升远远大于 LDRE $^ +$ IP-CIR 所取得的 $+ 1 . 6 3 \mathrm { m A P } @ 5$ 的提升。同样，应用于 SEIZE 时，我们的方法再次取得了 $+ 3 . 4 7 \ \mathrm { m A P } @ 5 $ 的进一步提升，证明了其在多目标场景中的有效性。这突显了所提出的特征级转换在利用参考图像的视觉线索方面的有效性。图 3 展示了我们的方法与 CIReVL 的定性比较。在 CIRCO 示例中，我们的模型将参考图像的线索与目标标题融合，并检索出在卡车顶部（或旁边）出现四轮摩托车的图像。相较之下，CIReVL 仅使用目标标题进行检索，常常错过两个关键物体中的一个，甚至返回参考图像本身。在 CIRR 示例中，目标标题省略了品种，所以 CIReVL 检索出了与“同一只狗”不符的狗，而我们的方法利用视觉证据返回了与参考图像相同品种的狗（边境梗）；请注意，Top-1 和 Top-2 是因数据集噪声而重复的。Fashion-IQ 如表 2 所示，在所有三种架构和两个基准上，将我们的方法与基线相结合取得了一致的改进，并在大多数指标上实现了最先进的性能。需要注意的是，LinCIR 是一种基于训练的基线，利用了文本反转，这解释了它相对强大的表现；与我们的方法结合时，表现更为优越。这证实了即使在自然语言修改高度细粒度和多样化的时尚领域，所提出的指令引导特征转换在利用参考图像线索方面依然有效。此外，图 3 中 FashionIQ 的定性结果强调了我们的方法捕捉了来自参考图像的纹理和轮廓信息，这些信息在文本表达中难以完全呈现，从而为诸如“绿色 A 型连衣裙”这样的指令生成了更忠实的匹配。

Table 2: Quantitative results on the Fashion-IQ benchmark using the ViT-G/14 backbone, where our method is applied on top of three representative baselines (CIReVL (Karthik et al., 2023), SEIZE (Yang et al., 2024a), and LinCIR (Gu et al., 2024)). Across all three categories (Shirt, Dress, and Toptee), our method combined with the baselines shows consistent improvements and reaches state-of-the-art performance on the majority of metrics. $\dagger$ indicates our reproduced results.   

<table><tr><td colspan="3">Type</td><td colspan="2">Shirt</td><td colspan="2">Dress</td><td colspan="2">Toptee</td><td colspan="2">Average</td></tr><tr><td>Backbone</td><td>Method</td><td></td><td>R@10 R@50</td><td></td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td rowspan="10">ViT-G/14</td><td>Pic2Word</td><td>CVPR23</td><td>33.17</td><td>50.39 55.35</td><td>25.43</td><td>47.65</td><td>35.24</td><td>57.62</td><td>31.28</td><td>51.89</td></tr><tr><td>SEARLE</td><td>ICCV23</td><td>36.46</td><td></td><td>28.16</td><td>50.32</td><td>39.83</td><td>61.45</td><td>34.81</td><td>55.71</td></tr><tr><td>LinCIR</td><td>CVPR24</td><td>46.76</td><td>65.11</td><td>38.08</td><td>60.88</td><td>50.48</td><td>71.09</td><td>45.11</td><td>65.69</td></tr><tr><td>CIReVL†</td><td>ICLR24</td><td>35.13</td><td>52.65</td><td>27.52</td><td>49.03</td><td>37.33</td><td>58.75</td><td>33.33</td><td>53.48</td></tr><tr><td>LDRE</td><td>SIGIR24</td><td>35.94</td><td>58.58</td><td>26.11</td><td>51.12</td><td>35.42</td><td>56.67</td><td>32.49</td><td>55.46</td></tr><tr><td>SEIZE</td><td>ACMMM24</td><td>39.50</td><td>57.65</td><td>33.37</td><td>55.88</td><td>41.66</td><td>64.20</td><td>38.12</td><td>59.24</td></tr><tr><td>OSrCIR</td><td>CVPR25</td><td>38.65</td><td>54.71</td><td>33.02</td><td>54.78</td><td>41.04</td><td>61.83</td><td>37.57</td><td>57.11</td></tr><tr><td>LinCIR + IP-CIR</td><td>CVPR25</td><td>48.04</td><td>66.68</td><td>39.02</td><td>61.03</td><td>50.18</td><td>71.14</td><td>45.74</td><td>66.28</td></tr><tr><td>CIReVL†+Ours</td><td>40.38</td><td>59.08</td><td>36.49</td><td></td><td>58.70</td><td>43.65</td><td>64.10</td><td>40.17</td><td>60.63</td></tr><tr><td>SEIZE†+Ours LinCIR+Ours</td><td></td><td>44.36 48.28</td><td>62.22 67.17</td><td>40.21 43.53</td><td>62.12 64.70</td><td>48.55 52.68</td><td>69.30 73.13</td><td>44.37 48.16</td><td>64.55 68.33</td></tr></table>

Table 3: Comparison of inference time.   

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Fashion-IQ Dress (DB size = 4K)</td><td colspan="2">CIRCO (DB size = 120K)</td></tr><tr><td>time</td><td>+∆t</td><td>time</td><td>+∆t</td></tr><tr><td>CIReVL +Ours</td><td>1.76s 1.87s</td><td>0.11s</td><td>2.16s 2.77s</td><td>0.61s</td></tr><tr><td>+IP-CIR</td><td>119.82s</td><td>118.06s</td><td>120.84s</td><td>118.68s</td></tr><tr><td>SEIZE</td><td>26.05s</td><td>−</td><td>26.25s</td><td>−</td></tr><tr><td>+Ours</td><td>26.17s</td><td>0.12s</td><td>26.89s</td><td>0.64s</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+IP-CIR</td><td>144.19s</td><td>118.14s</td><td>145.21s</td><td>118.96s</td></tr></table>

Table 4: Ablation study on CIRCO dataset.   

<table><tr><td>Method</td><td>Proxy</td><td>Scaling</td><td>Gating</td><td>mAP@5</td><td>∆</td></tr><tr><td>CIReVL</td><td></td><td></td><td></td><td>26.47</td><td>-</td></tr><tr><td rowspan="3"></td><td>✓</td><td></td><td></td><td>27.42</td><td>+0.95</td></tr><tr><td></td><td>v</td><td></td><td>29.41</td><td>+2.94</td></tr><tr><td>✓</td><td></td><td>✓</td><td>32.32</td><td>+5.85</td></tr></table>

推理时间 我们进一步评估了我们的方法与基于生成的方法（Li et al., 2025）相比的效率。如表 3 所示，我们的方法与基线检索模型（CIReVL 和 SEIZE）相比，仅引入了微不足道的开销。具体而言，在 Fashion-IQ Dress（4K 数据库）上，额外计算时间仅为 0.11-0.12 秒，在 CIRCO（120K 数据库）上，开销保持在 0.61-0.64 秒以内。相比之下，基于生成的 IP-CIR 在这两个数据集上需要超过 118 秒的额外处理时间，使其速度慢了两个数量级。这些结果清楚地表明，我们的方法在维护接近实时的效率的同时，显著提高了准确性。通过避免成本高昂的图像生成，我们的方法能够有效扩展到大规模数据库，并为基于生成的解决方案提供了实际替代方案。

# 4.2 消融研究

组件分析 为了深入了解每个组件的影响，我们进行了一项消融研究，以评估我们框架中每个模块的有效性：代理嵌入、最优缩放和门控功能，如表4所示。基线模型（CIReVL）的表现为 $26.47 \mathrm{mAP}@5$。单独使用代理嵌入（$\alpha = 1$）的结果为 $27.42 \mathrm{mAP}@5$ $(+0.95)$。引入缩放策略进一步提高了性能，达到 $29.41 \mathrm{mAP}@5$ $(+2.94)$。最后，将门控功能与代理和缩放结合使用，取得了最佳结果 $32.32 \mathrm{mAP}@5$ $(+5.85)$。这些结果证实适应性调节代理相似性的贡献对于抑制误导性视觉线索和提高检索准确性至关重要。

![](images/4.jpg)  

Figure 4: Retrieval performance with the reference image as ground truth.

![](images/5.jpg)  

Figure 5: Ablation study of optimal scaling.

语义变换的最优缩放分析 为了更好地理解代理嵌入的行为，我们在CIRCO数据集上进行了一项受控检索实验，目标是检索参考图像而非目标图像。该实验使我们能够直接检查代理嵌入与原始参考表示之间的偏差程度。如图4所示，代理大多数情况下未能有效偏离参考。特别是，当文本空间的语义转移被直观地转移时，生成的代理嵌入实现了98.0的$\mathbb { R } \ @ 1$，这表明它与参考几乎没有变化。这个发现促使我们探索一种最优的缩放策略。

根据结果，我们发现仅强制满足第一个条件，即要求代理嵌入与目标标题嵌入对齐（由 $\alpha$ 控制），是不够的。尽管该条件将代理嵌入推离参考，但它仍然留在其附近。通过引入额外的参数 $\beta$ ，我们观察到代理嵌入逐渐远离参考图像，这种偏离与检索性能密切相关，如图5所示。当 $\alpha$ 固定为1时，性能相对于基线有所改善，但没有显著提升。相反，采用我们提出的缩放策略会导致性能的持续提高，直到 $\beta$ 超过某个阈值，此时性能急剧下降。性能下降的原因在于代理嵌入与参考变得过于脱离，这与直观一致——目标图像仍然保留了参考的基本信息。此外，当我们直接使用参考图像（$\alpha = 0$）时，性能显著下降，仅达到 $\mathrm { m A P } @ 5$ 的 22.75。与代理嵌入保持接近参考但朝向目标方向偏移不同，这一结果表明语义转变对于有效引导代理嵌入至关重要。

# 5 结论

在本工作中，我们提出了自适应语义转换（AdaST），这是一种无训练的复合图像检索方法，在准确性和效率上达到了新的最先进水平。我们识别了现有方法中的一个核心挑战：在文本基础方法的效率与生成基础方法的保真度之间的强制选择，前者损失了视觉细节，而后者计算开销较大。AdaST通过在潜在空间中直接转换参考图像特征并以文本指令为指导解决了这一困境。这种方法有效地保留了细粒度的视觉信息，而不依赖于成本高昂的图像生成。我们在三个基准上的实验表明，AdaST显著超越了先前的方法。自适应相似性机制的引入进一步通过智能加权视觉和文本线索提高了鲁棒性。因此，AdaST不仅更准确，而且在速度上也明显快于基于生成的方法，使其非常适合实际应用。其模块化、即插即用的设计也便于与现有的零样本复合图像检索（ZS-CIR）管道集成。我们相信，我们的特征空间转换方法为多模态检索和理解的未来提供了一个有前景和高效的方向。

参考文献 Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, 和 Alberto Del Bimbo. 零样本组合图像检索与文本反演. 在 IEEE/CVF 国际计算机视觉会议论文集, 第 15338-15347 页, 2023. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell 等. 语言模型是少样本学习者. 神经信息处理系统进展, 33:1877-1901, 2020. Yanbei Chen 和 Loris Bazzani. 学习语言引导检索的联合视觉语义匹配嵌入. 在欧洲计算机视觉会议论文集, 第 1361-1365 页. Springer, 2020. Ginger Delmas, Rafael Sampaio de Rezende, Gabriela Csurka, 和 Diane Larlus. Artemis：基于注意力的检索，通过显式文本匹配和隐式相似度. arXiv 预印本 arXiv:2203.08101, 2022. TsuJui Fu, Xin Eric Wang, 和 Wiliam Yang Wang. 基于语言驱动的艺术风格转换. 在欧洲计算机视觉会议 (ECCV), 2022. Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, 和 Daniel Cohen-Or. Stylegan-nada：基于 Clip 的图像生成器领域适应. ACM 图形学事务 (TOG), 41(4):113, 2022. Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, 和 Daniel Cohen-or. 一幅图像值一字：使用文本反演个性化文本到图像生成. 在第十一届国际表示学习会议, 2023. URL https://openreview.net/forum?id=NAQvFO8TcyG. Geonmo Gu, Sanghyuk Chun, Wonjae Kim, Yoohoon Kang, 和 Sangdoo Yun. 仅基于语言的零样本组合图像检索训练. 在 IEEE/CVF 计算机视觉与模式识别会议论文集, 第 13225-13234 页, 2024. Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave 等. OpenCLIP. https://github.com/mlfoundations/open_clip, 2021. Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, 和 Zeynep Akata. 通过语言进行无训练组合图像检索. arXiv 预印本 arXiv:2310.09291, 2023. Klo, riXiRobr e a Jacos, nySy G 为组合图像检索提供详细合成标题. 在计算机视觉与模式识别会议论文集, 第 3148-3157 页, 2025. Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, 和 Jun-Yan Zhu. 文本到图像扩散的多概念定制. 在 IEEE/CVF 计算机视觉与模式识别会议论文集, 第 1931-1941 页, 2023. Gihyun Kwon 和 Jong Chul Ye. Clipstyler：基于单一文本条件的图像风格转换. 在 IEEE/CVF 计算机视觉与模式识别会议论文集, 第 18062-18071 页, 2022. Seungmin Lee, Dongwan Kim, 和 Bohyung Han. Cosmo：用于文本反馈的图像检索的内容-风格调制. 在 IEEE/CVF 计算机视觉与模式识别会议论文集, 第 802-812 页, 2021. Juna Li, Dongxu Li, Silvio Savarese, 和 Steven Hoi. Blip-2：使用冻结的图像编码器和大型语言模型引导语言-图像预训练. 在国际机器学习会议论文集, 第 19730-19742 页. PMLR, 2023. You Li, Fan Ma, 和 Yi Yang. Imagined and seek：通过想象的代理改善组合图像检索. 在计算机视觉与模式识别会议论文集, 第 3984-3993 页, 2025. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, 和 C Lawrence Zitnick. Microsoft coco：上下文中的常见物体. 在欧洲计算机视觉会议论文集, 第 740-755 页. Springer, 2014. Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, 和 Stephen Gould. 使用预训练视觉和语言模型对真实图像进行图像检索. 在 IEEE/CVF 国际计算机视觉会议论文集, 第 2125-2134 页, 2021. Jakob Nielsen. 可用性工程. Morgan Kaufmann, 1994. Jihun Park, Jongmin Gim, Kyoungmin Lee, Seunghun Lee, 和 Sunghoon Im. Style-editor：基于文本驱动的面向对象的风格编辑. 在计算机视觉与模式识别会议论文集, 第 18281-18291 页, 2025. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark 等. 从自然语言监督中学习可转移的视觉模型. 在国际机器学习会议论文集, 第 8748-8763 页. PmLR, 2021. Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, 和 Kfir Aberman. Dreambooth：针对主题驱动生成的文本到图像扩散模型的微调. 在 IEEE/CVF 计算机视觉与模式识别会议论文集, 第 22500-22510 页, 2023. Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, 和 Tomas Pfister. Pic2word：将图像映射到单词以实现零样本组合图像检索. 在 IEEE/CVF 计算机视觉与模式识别会议论文集, 第 19305-19314 页, 2023. Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, 和 Yoav Artzi. 关于基于照片的自然语言推理的语料库. arXiv 预印本 arXiv:1811.00491, 2018. Yuanmin Tang, Jue Zhang, Xiaoting Qin, Jing Yu, Gaopeng Gou, Gang Xiong, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, 和 Qi Wu. 先推理再检索：用于无训练零样本组合图像检索的一阶段反思链思维. 在计算机视觉与模式识别会议论文集, 第 14400-14410 页, 2025. Yuxin Tian, Shawn Newsam, 和 Kofi Boakye. 通过加性注意力组合学习的文本反馈时尚图像检索. 在 IEEE/CVF 计算机视觉应用冬季会议论文集, 第 1011-1021 页, 2023. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar 等. Llama：开放高效的基础语言模型. arXiv 预印本 arXiv:2302.13971, 2023. Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, 和 James Hays. 组合文本和图像以进行图像检索的经验之旅. 在 IEEE/CVF 计算机视觉与模式识别会议论文集, 第 6439-6448 页, 2019. Yuxiang Wei, Yabo Zhang, Zhilong Ji, Jinfeng Bai, Lei Zhang, 和 Wangmeng Zuo. Elite：将视觉概念编码为文本嵌入以定制文本到图像生成. 在 IEEE/CVF 国际计算机视觉会议论文集, 第 15943-15953 页, 2023. Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, 和 Rogerio Feris. Fashion IQ：通过自然语言反馈检索图像的新数据集. 在 IEEE/CVF 计算机视觉与模式识别会议论文集, 第 11307-11317 页, 2021. Zhenyu Yang, Shengsheng Qian, Dizhan Xue, Jiahong Wu, Fan Yang, Weiming Dong, 和 Changsheng Xu. 语义编辑增益零样本组合图像检索. 在第 32 届 ACM 国际多媒体大会论文集, 第 1245-1254 页, 2024a.

Zhenyu Yang, Dizhan Xue, Shengsheng Qian, Weiming Dong, 和 Changsheng Xu. Ldre: 基于LLM的发散推理与集成用于零-shot复合图像检索. 见于第47届国际ACM SIGIR信息检索研究与开发会议论文集，pp. 8090, 2024b. Moon Ye-Bin, Jisoo Kim, Hongyeob Kim, Kilho Son, 和 Tae-Hyun Oh. Textmania: 通过文本驱动的流形增强丰富视觉特征. 见于IEEE/CVF国际计算机视觉会议论文集，pp. 2526-2537, 2023. Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu Su, 和 Ming-Wei Chang. Magiclens: 在开放式指令下进行自监督图像检索. arXiv预印本 arXiv:2403.19651, 2024. Dewei Zhou, You Li, Fan Ma, Zongxin Yang, 和 Yi Yang. Migc $^{++}$ : 高级多实例生成控制器用于图像合成. IEEE模式分析与机器智能学报, 2024.

# 附录

# A.1 伦理声明

根据 ICLR 2026 指导原则，我们披露在准备本文稿时使用了大型语言模型（LLM）进行语法校正、文本优化、稿件审阅和通过深度研究进行相关研究检索。LLM 还用于支持实验并生成绘制图形所需的代码（图 4 和图 5），这些图形基于我们实验获得的数据。所有研究贡献、实验结果和科学主张均由作者全权负责。