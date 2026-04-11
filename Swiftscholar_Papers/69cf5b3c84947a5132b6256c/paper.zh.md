# 超越全局相似性：朝向细粒度多条件多模态检索

徐璐1,3* 李康乐1,2* 黄浩航3 孟锐† 曾文俊2,3 沈晓宇2,3 1上海交通大学 2宁波东部科技学院数字双胞胎研究所 3宁波空间智能与数字衍生重点实验室 lux1997@sjtu.edu.cn xyshen@eitech.edu.cn

# 摘要

最近的多模态大型语言模型（MLLMs）进展显著扩展了多模态检索的能力，使系统能够在视觉和文本模态之间进行信息对齐和检索。然而，现有基准主要集中在粗粒度或单条件对齐上，忽视了现实世界场景中用户查询指定多个相互依赖的条件。为了填补这一空白，我们引入了MCMR（多条件多模态检索）：一个大规模基准，旨在评估在自然语言查询下细粒度、多条件的跨模态检索。MCMR涵盖五个产品领域：上装和下装、珠宝、鞋子和家具。它还保留了丰富的长格式元数据，这对于组合匹配至关重要。每个查询整合了互补的视觉和文本属性，要求模型共同满足所有指定条件的相关性。我们基准测试了一套多样化的基于MLLM的多模态检索器和视觉语言重排序器，以评估其条件感知推理能力。实验结果揭示了：（i）模型之间存在明显的模态不对称；（ii）视觉线索主导了早期排名精度，而文本元数据稳定了长尾排序；（iii）基于MLLM的逐点重排序器通过明确验证查询与候选项的一致性，显著改善了细粒度匹配。总体而言，MCMR为推动多模态检索向组合、条件感知和可解释理解建立了一个具有挑战性和诊断性的基准。我们的代码和数据集可在 https://github.com/EIT-NLP/MCMR 获取。

![](images/1.jpg)  
Figure 1. Multi-condition multimodal retrieval driven by naturallanguage queries with fine-grained visual and textual constraints.

# 1. 引言

多模态检索系统旨在对齐和检索跨异质和复合模态的信息，如文本、图像或它们的交织组合[1, 29, 38, 40, 41]。早期模型如CLIP[37]、ALIGN[14]和BLIP[19]通过对比学习在整体图像-文本对上进行了训练，并被广泛应用于网页搜索[15, 21, 31]、电子商务[39, 44]和推荐系统[11, 25]。然而，标题通常仅提供视觉内容的整体和一般性描述，因此这些模型倾向于强调全局语义一致性，而非细粒度的跨模态理解。近年来，多模态大语言模型（MLLM）的快速发展开始重塑这一领域，通过支持开放式自然语言指令下的检索[17, 31, 33, 46, 48]。基于这一能力，代表性方法如VLM2Vec[17]、MM-Embed[21]和GME[46]进一步扩大了多模态检索的范围，生成能够捕捉指令条件语义的统一嵌入。这一演变标志着从静态全局对齐向更具表现力和灵活性的检索范式的转变。

尽管取得了这一进展，但现有基准测试很少能同时满足复杂多模态检索所需的三个关键属性：（i）细粒度属性推理，（ii）多条件查询，以及（iii）跨模态证据，其中不同条件必须在图像和文本中得到验证。经典的图像文本数据集，如 MS-COCO 和 Flickr30K，主要评估图像与标题之间的粗略全局对齐：每个标题被视为一个整体描述，相关性不依赖于满足多个独立约束。时尚和定位等领域的细粒度基准，如 FashionIQ 和 CIRR，引入了局部编辑或相对调整（例如，“更短的袖子”，“相同风格但更亮的颜色”），并依赖参考图像来锚定这些修改。虽然这些数据集提供了比全局标题基准更细的信息，但它们仍然围绕单一视觉编辑展开，并且仍然有效地是单一模态，因为大多数属性可以仅从图像中验证。同时，多条件检索已在仅文本的设置中进行探索，在这种情况下，查询列举了多个约束，但所有证据都存在于文本中，从而消除了整合异构视觉和文本线索的需求。现有基准要么捕捉细粒度细节，要么捕捉多条件结构，但在跨模态上不能同时满足这两者。最近的研究，如 MERIT，通过多语言、交错的图像文本查询扩展了检索设置，这些查询将参考图像与简短的文本片段结合。在 MERIT 中，许多查询条件与示例图像相关制定（例如，“与产品 1 相同风格，且与产品 2 相同颜色”），这将任务框定为与提供的示例进行视觉比较，而不是独立指定的属性约束。这种表达还假设用户能够在查询时提供代表性的参考图像，而在许多现实世界的搜索场景中，用户更常通过自然语言描述直接表达他们的需求。此外，MERIT 并未明确区分需要视觉锚定的属性与仅存在于文本元数据中的属性，这使得分析模型如何在模态之间平衡证据变得困难。为了解决这些局限性，我们引入了 MCMR（多条件多模态检索），这是一个根据上述三个维度（细粒度、多条件和跨模态）显性设计的大规模基准。如图 1 所示，每个查询是一个自然语言描述，结合了多个关于视觉和文本属性的组合约束，只有在所有条件都满足时，候选结果才被视为相关。关键的是，MCMR 强制实施双重证据设计，其中某些属性只能从图像中推断（例如，特定的图形布局或纹理），而其他属性则仅由长文本元数据支持（例如，材料、合身度或制造细节）。这使得从单一模态解决该任务变得不可能，并直接测试模型在跨模态整合互补证据的能力。

我们在统一协议下对一系列具有代表性的多模态检索器进行了基准测试，以评估它们在细粒度、条件意识推理方面的能力，并进一步引入基于大语言模型的逐点重排序器，检验大型视觉语言模型是否可以通过一对一、受限意识的相关性估计来弥合细粒度语义差距。我们的实验揭示了多条件多模态检索中持续存在的挑战。模态消融分析揭示了模态依赖性中的明显不对称：例如，GME [46] 和 LamRA [27] 等模型在移除文本元数据时仍相对稳健，这表明它们对视觉特征的依赖更强，而像 LLaVE [18] 这样的模型在仅图像设置下则遭受显著下降。进一步分析表明，视觉线索主导了顶级排名的准确性，文本元数据则稳定了长尾排序，基于大语言模型的重排序器通过显式验证每个查询-候选对显著改善了细粒度匹配。综合来看，这些发现揭示了当前系统在真正多条件、跨模态检索上的局限性，并强调了需要集成组合推理却不牺牲可扩展性的架构。我们的主要贡献总结如下：基准。我们提出了 MCMR，一个大规模基准，联合满足三个期望——细粒度属性、多条件查询和跨模态证据——用于评估自然语言查询下的多模态检索。•全面评估。我们对具有代表性的多模态检索器和基于大语言模型的逐点重排序器进行了全面研究，揭示了在受控组合设置下的系统性模态不对称和细粒度推理差距。•研究结果。我们的分析显示，目前的检索器难以同时满足多个异构跨模态约束，而基于大语言模型的重排序显著提高了精度，突显出对可扩展的、具约束意识的检索架构的需求。

# 2. 相关工作

多模态检索模型。多模态检索旨在将异构模态（最常见的为图像和文本）对齐到统一的语义空间中，从而实现跨模态搜索和推理。一种主流范式使用预训练的视觉-语言模型（如 CLIP、BLIP 和 ALIGN）对每种模态进行编码，并通过共享嵌入空间中的余弦距离来测量相似性。在这些双编码器主干的基础上，任务无关的检索器通常采用图像和文本的独立编码，然后通过后期融合来调和两种模态。例如 UniVL-DR 和 UniIR 通过规范化异构数据集并使用 CLIP 或类似 BLIPlike 的编码器将它们映射到共享空间，然后在推理时应用校准融合，以在检索任务中平衡模态贡献。另一类研究将轻量级视觉适配器附加到强大的文本嵌入模型上，允许图像-文本对以组合形式进行编码，而无需对整个模型进行重训练。具有代表性的例子包括 VISTA 和 MARVEL，它们将视觉信息注入文本编码器中，以实现统一的多模态表示。最近，多模态大型语言模型（MLLM）通过直接从最终变换器层汇聚隐藏状态作为嵌入，转变为通用检索器，例如 E5-V 和 VLM2Vec。这种 MLLM 作为嵌入的范式增强了可迁移性，统一了文本-文本与跨模态检索，更好地与检索增强生成管道对齐。在这两类模型中——双编码器和基于 MLLM 的检索器——模型通常针对全局语义对齐进行优化，并且通常表现出明显的模态依赖性：一些模型更依赖于视觉外观，而另一些则重度依赖于文本语义。这种不对称性是我们研究中实证调查的核心方面之一。

多模态检索基准。早期的多模态检索基准强调图像与标题之间的粗粒度或单一条件对齐。像 MS-COCO [22] 和 Flickr30K [36] 这样的数据集定义了标题与图像之间的一对一对应关系，关注整体相似性而不是组合推理。领域特定的数据集，包括 VisualNews [24] 用于新闻图像与标题的对齐，以及 Fashion200k [10] 用于产品检索，将这一范式扩展到专门的设置，但仍然在整体层面上运作。超越自然图像，像 DocVQA 家族通过 ViDoRe [9] 处理的文档中心套件将检索视为在给定自然语言问题的情况下选择正确的文本丰富文档图像。随后“组合”检索基准将重点从整体相似性转向有针对性的属性编辑 [48]。FashionIQ [43] 和 CIRR [28] 引入根据简短文本指令修改参考图像的概念——例如检索“红色的同款衬衫”——以测试模型是否能够结合局部属性变化。同时，融合模态基准如 WebQA [3] 和 EDIS [26] 在开放领域检索中整合文本和视觉证据，而 OVEN [13] 和 INFOSEEK [4] 将图像条件问题与维基百科段落或标题配对，以评估混合模态监督下的检索表现。进一步的扩展如 ReMuQ [34]、OKVQA [35] 和 LLaVA 风格的对话数据集 [23] 扩大了查询和候选模态，以支持问答和基于对话的检索。最近，MERIT [7] 通过引入多语言和交错的多模态输入，超越了单一参考查询，可以同时引用多个图像和文本片段，从而实现更复杂的条件规范。

<table><tr><td>Benchmark</td><td>Q</td><td>T</td><td>DE</td><td>MA</td><td>LM</td></tr><tr><td>MS-COCO [22]</td><td>T</td><td>I</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Flickr30K [36]</td><td>T</td><td>I</td><td>X</td><td>X</td><td>X</td></tr><tr><td>FashionIQ [43]</td><td>T+I</td><td>I</td><td>X</td><td>X</td><td>X</td></tr><tr><td>CIRR [28]</td><td>T+I</td><td>I</td><td>X</td><td>X</td><td>X</td></tr><tr><td>WebQA [3]</td><td>T</td><td>T+I</td><td>X</td><td>X</td><td>X</td></tr><tr><td>EDIS [26]</td><td>T</td><td>T+I</td><td>X</td><td>X</td><td>X</td></tr><tr><td>MERIT [7]</td><td>T+I</td><td>T+I</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>MultiConIR [32]</td><td>T</td><td>T</td><td>X</td><td>√</td><td>✓</td></tr><tr><td>MCMR (ours)</td><td>T</td><td>T+I</td><td>V</td><td></td><td></td></tr></table>

Table 1. Comparison of representative multimodal retrieval benchmarks. $\mathbf { Q } =$ query modality, $\textrm { T } =$ target modality, $\mathrm { D E = }$ Dual-Evidence, $\mathbf { M A } =$ Multi-Attribute , $\mathbf { L M } =$ Long-form Metadata. Entries in $\mathrm { Q } / \mathrm { T }$ use $\mathrm { T }$ for text, I for image, and $\mathrm { T } { + } \mathrm { I }$ when both modalities are available. MCMR uniquely supports dual-evidence grounding, multi-attribute conditions, and long-form metadata.

# 3. 多模态多条件检索

我们介绍了MCMR，一个多模态产品检索数据集，包含5种检索场景下的10,400个产品。基本单元是一个产品实例，包含图像和长文本描述，其中两种模态提供互补信息。表1总结了MCMR在查询-目标模态、证据结构和元数据丰富性方面与先前多模态检索基准的比较。

# 3.1. 数据收集与预处理

构建于 Amazon Reviews (2023) 语料库之上，MCMR 涵盖了五个产品领域——上衣和下装、珠宝、鞋类和家具——同时保留了对精细多模态检索至关重要的长文本产品元数据。为了确保数据质量和可重复性，我们通过一个多阶段流程构建数据集，该流程结合了自动过滤与人工验证，逐步从广泛的候选采集精炼到精细的质量控制。我们遵循三个原则：广泛覆盖、高质量和跨模态一致性。从 Amazon Reviews 元数据开始，我们保留了在四个目标领域中同时具有可解析图像和长文本描述的产品。数据策划分为三个阶段：

![](images/2.jpg)  
two judging stages, and composed into queries for fine-grained retrieval evaluation.

Table 2. Domain-wise counts and token-length statistics $\left( \mathrm { m a x } / \mathrm { m i n } / \mathrm { a v g } \right)$ for queries and candidates.   

<table><tr><td rowspan="2"></td><td colspan="6">Domain Distribution (Counts)</td><td colspan="3">Token Statistics (Tokens)</td></tr><tr><td>Upper</td><td>Bottom</td><td>Shoe</td><td>Jewelry</td><td>Furniture</td><td>Total</td><td>max</td><td>min</td><td>Avg.</td></tr><tr><td>Queries</td><td>991</td><td>803</td><td>847</td><td>602</td><td>754</td><td>3997</td><td>57.00</td><td>25.00</td><td>35.86</td></tr><tr><td>Candidates</td><td>29 986</td><td>29514</td><td>24 997</td><td>5491</td><td>14 993</td><td>104 981</td><td>269.00</td><td>54.00</td><td>190.94</td></tr></table>

(1) 属性规范化。我们通过统一单位和货币、标准化日期，以及对材料和尺寸的控制词汇进行对齐，来标准化结构化属性，从而在各个领域创建统一的表示。 (2) 质量过滤和去重。我们基于文本长度、图像分辨率和纵横比，以及感知或嵌入相似性应用过滤和近重复检测。低质量、不完整或冗余的记录被移除。直接标识符如ASIN和URL被屏蔽或排除，以防止信息泄露。 (3) 互补性约束。每个项目必须至少包含一个仅文本和一个仅图像的细粒度属性，确保两种模态提供独特的证据，而不是允许任务仅依赖于单一模态解决。

# 3.2. 数据集构建流程

为了平衡成本和保真度，我们采用一种合作流水线，将高效的中型模型与强大的模型进行配对，以便于大规模生成并进行验证和优化。目标是生成多条件、跨模态和可验证的自然语言查询，这些查询基于互补的视觉和文本证据。整个构建流水线在图2中进行了说明。 图像侧的结构扩展。针对每个产品图像，Qwen2.5-VL-32B-Instruct生成一个包含类别标签和图像独有属性的结构化、有证据支撑的摘要（例如，颜色、纹理、结构细节、形状、结构）。生成内容严格排除功能性或推测性内容，形成用于(i) 互补性验证和(ii) 查询构建的视觉证据层。 文本侧的结构扩展。产品标题、描述和特征列表被转换为使用JSON提取模板的结构化文本档案。必填字段包括attributes_text_only和category_text；可选字段（例如，价格、发行年份、材料、风格、品牌）在存在时添加。标识符被清理，品牌提及仅在与图像独有属性同时出现时允许。方案感知合并保留标准元数据，同时附加结构化字段以实现多模态对齐。 文本描述生成。我们采用Qwen3-32B-Instruct生成简洁的（80-120字）目录式文本摘要，仅基于文本元数据，明确排除任何视觉描述。使用DeepSeek-R1-Distill-Qwen-32B的验证编辑循环检测跨模态泄漏并强制一致性，同时进行人工抽查以确保鲁棒性。 查询生成。在图像属性和文本摘要的条件下，Qwen3-32B-Instruct生成以第一人称表述的多条件查询，结合来自两种模态的互补约束。数字和时间信息被标准化，标识符被屏蔽，并保持中立的“购物者”语气。特定领域的提示变体（针对服装、珠宝、鞋子和家具）指导属性选择——例如，服装的fabric/fit/care或珠宝的gemstone/cut/setting，以保留语义多样性。 MLLM作为重标定器：Qwen2.5-VL-32B、Qwen2.5-VL-7B、Qwen3-VL-4B、Qwen3-VL-8B、InternVL3-8B-Instruct、Qwen3-VL-Reranker-8B及lychee-reranker-mm。在检索中，我们使用模型的融合图像文本接口对候选项目进行嵌入，而查询则仅从文本编码。所有系统在以零样本模式运行，使用公开发布的检查点。对于重标定，我们采用最强一级检索器返回的前50个候选项，应用视觉语言多模态学习模型（MLLM）作为逐点重标定器。每个查询-候选对独立评估：重标定器接收文本查询以及候选的图像和元数据，并通过统一提示生成二元相关性判断（真/假）。模型的“真”标记的归一化对数分数用作相关性分数，候选项按此分数排序以获得最终顺序。这种成对评估补充了一级检索器，通过提供细粒度的跨模态相关性估计。 查询验证。DeepSeek-R1-Distill-Qwen-32B作为独立验证器，评估跨模态覆盖和数字/时间一致性。失败或边缘情况将被重新生成，并手动检查一小部分被接受的样本。为了进一步验证质量，我们进行了一项100个样本的人类研究：一名标注员根据每个产品的图像和元数据编写自然语言查询，第二名标注员在双盲条件下评估人类编写和生成查询的属性正确性、跨模态覆盖和自然性（5点李克特量表）。这两种类型的查询在平均得分（4.33 vs. 4.41）和偏好率（47% vs. 49%）上表现相当，确认生成的查询与人类撰写的查询高度匹配，且验证开销最小。表2展示了MCMR数据集的关键统计信息，包括查询和候选项的数量以及它们的文本词元长度。我们报告了三种广泛使用的信息检索指标：Recall@$K$、nDCG@$K$和MRR@$10$。Recall@$K$衡量在前$K$（$K \in \{1, 5, 10, 50, 100\}$）中检索到的相关项的比例，反映系统在多条件约束下展现正确结果的能力。nDCG@$K$根据文档位置对相关文档进行加权评估排名质量，强调前结果中的细粒度排序。MRR@$10$通过第一个相关项在前10个位置中的平均逆排名总结早期排名效果。

# 4.2. 主要结果

我们在三种候选可见性模式下评估多模态检索器在 MCMR 上的表现：融合模式（图像+文本）、仅图像模式和仅文本模式。完整结果见表 3。

# 4. 实验

本节首先介绍实验设置（见 $S 4.1$），随后是主要结果（见 $S 4.2$），包括检索模型的整体比较。接着，$S 4.3$ 提供了 MLLM 的重排序性能。所有实验均在两台 NVIDIA A100 GPU（80 GB）上进行。

# 4.1. 基线和指标

我们在相同的预处理和推理设置下评估五种代表性的多模态检索器和五种多语言大模型重排名器：• 多模态检索器：GME-Qwen2-VL-7B [46]，LLaVE-7B [18]，VLM2Vec [17]，LamRA-Ret-Qwen2.5-VL-7B [27]，以及CORAL [6]。

融合候选器 $\mathbf{( I m a g e + T e x t )}$。如表3所示，当同时提供视觉和文本元数据时，整体准确率仍然适中：主流检索模型仅达到 $18 - 27 \%$ 的 Recall $@ 1$，而 $V L M 2 V e c$ 的表现则显著更差，仅为 $1.83 \%$。尽管如此，大多数模型最终在更大的截止点上检索到正确的项目。最佳的 Recall $@ 10$ 达到 $53.34 \%$（CORAL），而最强的远程表现—$78.64 \%$ 的 Recall $@ 100$—则由 LLaVE 实现，这表明相关项目通常被找到，但排名并不靠前。这种早期排名和长截止检索之间的差距揭示了系统间的一种一致模式：模型在粗略检索方面相当有能力，但在多重条件约束下进行精细排序时则表现不佳。这一差异凸显了下游重新排序器的巨大潜力，能够在检索到正确候选项后优化排序。

文本移除的影响（仅图像候选）。移除文本元数据同时保留图像会显著影响早期排名的准确性。对于多个模型，$\mathbb { R } \ @ \mathbb { 1 } / 5 / 1 0$ 发生了急剧下降，在某些情况下，top-1 准确率低于 $1 \%$。相较之下，少数视觉表现强劲的编码器（例如，GME）保持了具有竞争力的得分，甚至在 $\mathbb { R } \ @ 1$ 中有所提升，这表明某些系统能够仅通过视觉线索满足多种条件。在较大的截断点，性能变得更加稳定：$\textR @ 5 0 / 1 0 0$ 保持相对较高，而 $\scriptstyle \mathrm { n D C G } @ 1 0 0$ 的降幅远小于 $\mathrm { n D C G } @ 1 0$ 的下降。这种模式表明，文本移除主要减少了细粒度的区分能力，但并没有严重妨碍在候选集内检索正确项的能力。

图像去除效果（仅文本候选）。去除图像并仅依赖文本元数据，导致所有指标的准确性明显下降，与融合候选相比，初步检索的表现特别薄弱：最佳的 Recall $@ 1$ 仅达到 $12.98\%$（MM-EMBED），大多数模型在 $7\mathrm{-}12\%$ 范围内。中间及长期截止的性能也有所下降—Recall $\textcircled{a} 50/100$ 远低于融合结果，最高仅为 $53\mathrm{-}62\%$。仅文本检索的表现也始终弱于仅图像检索。视觉上强大的模型，如 GME-Qwen2VL，在 Recall $@ 10$ 时从 $51.10\%$ 显著下降至 $29.60\%$，在 Recall $@ 100$ 时从 $78.86\%$ 降至 $57.50\%$。这些结果突显了 MCMR 判别线索的主要视觉特性。文本主要作为一种补充信号，当与图像结合时增强对齐，而最强的表现始终出现在两种模态均可用的情况下。

点对点重新排序的收益 我们对LLaVE-7B返回的固定前50个候选项应用了第二阶段的点对点重新排序器，在融合设置下实现了最高的Recall $\textcircled { a } 5 0$（72.01）。每个重新排序器都是一个视觉语言多模态大语言模型，独立评估与融合图像文本候选项配对的文本查询的相关性。模型输出第一个答案位置上词元“True”的标准化概率，我们将此用作相关性分数 $p ( \mathrm { T r u e } )$；然后按此分数对候选项进行排序，平局的情况下按其原始检索排名打破平局。图4显示的结果表明，在固定的前50个候选池中，点对点重新排序器在早期排名排序上提供了显著的改进，$\mathrm { n D C G } @ 1$ 分数始终维持在7080范围内。lychee-reranker-mm在每个截止点上都表现最强—对于 nDCG $@$ 1/5/10/50分别为92.35 $\textit { / } 9 3 . 4 1 \textit { / } 9 4 . 4 2 \textit { / } 9 4 . 8 6$—其后是 internVL-8B 和 Qwen3-VL-Reranker-8B，而基于 Qwen3-VL 的模型则略微落后。参数数量并不能预测排名能力；架构设计和视觉文本对接似乎对成对相关性评估有更大影响。在 $\mathrm { n D C G } @ 1$ 处表现差距最大，并向 $\mathrm { n D C G @ 5 0 }$ 收窄，表明多模态大语言模型重新排序器将其收益集中在最顶部，而长范围的排序保持相对稳定。多模态大语言模型重新排序器的强有力一对一相关性判断，与第一阶段检索器取得的适度 Recall $@ 1$ 形成鲜明对比，揭示了明显的差距：当前检索模型在细粒度、多条件语义推理方面表现乏力，而生成的视觉语言模型在独立评估每一对时能够弥补这一差距。这一对比也验证了MCMR作为基准的设计：它有效暴露了跨模态融合和细粒度检索的弱点，为评估模型的多模态对齐和条件感知推理能力提供了有意义的测试平台。

# 5. 分析

# 5.1. 候选方模态分析

模态贡献与不对称性。表3中的模态消融结果显示了模态依赖性明显的不对称性。在仅使用图像的候选中，GME和LamRA在$\mathrm { R @ 1 0 }$上的表现几乎接近融合性能，而LLaVE在移除文本后几乎崩溃，$\mathbf { R } \ @ 1$从24.99下降到0.90。切换到仅使用文本的候选同样影响了所有模型相较于融合设置的表现，MM-EMBED和CORAL在$\mathrm { R @ 1 0 }$的下降幅度相对较小。总体而言，在$\mathrm { R @ 1 0 }$下，仅使用文本的候选远不如融合候选，且在五个共享模型中有四个的表现也落后于仅使用图像的候选，这表明视觉线索在MCMR中比单独的文本元数据更具辨别力。然而，融合查询在$\mathrm { R @ 1 0 }$上的表现仍比仅使用图像的候选高出约48分，表明文本元数据提供了超越视觉证据的补充约束。这与数据集设计相符，其中每个项目至少包括一个仅使用图像和一个仅使用文本的属性，使得纯粹的单模态推理不完整。模型级方差与设计启示。模态移除加剧了跨模型方差。GME保持稳定，而LLaVE和CORAL在早期排名表现出现强烈下降，表明对文本先验的高度依赖。相对而言，MM-EMBED似乎更具鲁棒性。这些观察突显了当前检索模型的一个关键局限性：在融合输入下实现强大的全局语义相似性并不保证当单个模态减弱时的鲁棒性。未来的设计可能需要明确查询条件，并分别检查每一项，同时减少模态之间的冗余重叠，以维持性能。

<table><tr><td>model</td><td>size</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>R@100</td><td>MRR</td><td>N@1</td><td>N@5</td><td>N@10</td><td>N@50</td><td>N@100</td></tr><tr><td>LLaVE</td><td>7B</td><td>24.99</td><td>43.85</td><td>53.13</td><td>72.01</td><td>78.64</td><td>33.15</td><td>24.99</td><td>34.88</td><td>37.88</td><td>42.11</td><td>43.19</td></tr><tr><td>GME-Qwen2VL</td><td>7B</td><td>21.23</td><td>38.20</td><td>45.74</td><td>64.71</td><td>73.52</td><td>28.35</td><td>21.23</td><td>30.06</td><td>32.48</td><td>36.66</td><td>38.08</td></tr><tr><td>LamRA-Qwen2.5VL</td><td>7B</td><td>17.96</td><td>34.99</td><td>43.30</td><td>64.36</td><td>73.24</td><td>25.27</td><td>17.96</td><td>26.85</td><td>29.53</td><td>34.25</td><td>35.69</td></tr><tr><td>MM-EMBED</td><td>8B</td><td>21.74</td><td>39.58</td><td>47.91</td><td>66.22</td><td>74.16</td><td>29.35</td><td>21.74</td><td>31.05</td><td>33.75</td><td>37.82</td><td>39.11</td></tr><tr><td>CORAL</td><td>3B</td><td>26.57</td><td>46.69</td><td>53.34</td><td>70.90</td><td>77.73</td><td>34.94</td><td>26.57</td><td>37.20</td><td>39.35</td><td>43.27</td><td>44.37</td></tr><tr><td>VLM2Vec</td><td>4B</td><td>1.83</td><td>4.88</td><td>7.03</td><td>14.38</td><td>18.96</td><td>3.11</td><td>1.83</td><td>3.33</td><td>4.02</td><td>5.63</td><td>6.38</td></tr><tr><td colspan="10">(b) Image-only</td><td></td><td></td><td></td></tr><tr><td>model</td><td>size</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>R@100</td><td>MRR</td><td>N@1</td><td>N@5</td><td>N@10</td><td>N@50</td><td>N@100</td></tr><tr><td>LLaVE</td><td>7B</td><td>0.90</td><td>2.53</td><td>3.93</td><td>9.48</td><td>13.68</td><td>1.67</td><td>0.90</td><td>1.52</td><td>2.19</td><td>3.39</td><td>4.06</td></tr><tr><td>GME-Qwen2VL</td><td>7B</td><td>21.79</td><td>41.30</td><td>51.10</td><td>71.36</td><td>78.86</td><td>30.13</td><td>21.79</td><td>31.91</td><td>35.08</td><td>39.63</td><td>40.86</td></tr><tr><td>LamRA-Qwen2.5VL</td><td>7B</td><td>18.05</td><td>36.25</td><td>43.30</td><td>66.83</td><td>76.73</td><td>25.96</td><td>18.05</td><td>27.63</td><td>30.51</td><td>35.35</td><td>36.96</td></tr><tr><td>MM-EMBED CORAL</td><td>8B</td><td>13.23</td><td>28.15</td><td>35.68</td><td>57.29</td><td>67.00</td><td>19.72</td><td>13.23</td><td>21.06</td><td>23.50</td><td>28.30</td><td>29.89</td></tr><tr><td></td><td>3B</td><td>11.51</td><td>25.99</td><td>33.53</td><td>54.72</td><td>64.15</td><td>17.83</td><td>11.51</td><td>19.11</td><td>21.54</td><td>26.19</td><td>27.72</td></tr><tr><td colspan="10">(c) Text-only</td><td></td><td></td><td></td><td></td></tr><tr><td>model</td><td>size</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>R@100</td><td>MRR</td><td>N@1</td><td>N@5</td><td>N@10</td><td>N@50</td><td>N@100</td></tr><tr><td>LLaVE</td><td>7B</td><td>11.95</td><td>23.38</td><td>29.43</td><td>48.02</td><td>56.23</td><td>16.83</td><td>11.95</td><td>17.85</td><td>19.80</td><td>23.92</td><td>25.25</td></tr><tr><td>GME-Qwen2VL</td><td>7B</td><td>10.62</td><td>22.65</td><td>29.60</td><td>48.55</td><td>57.50</td><td>16.00</td><td>10.62</td><td>116.95</td><td>19.21</td><td>23.42</td><td>24.87</td></tr><tr><td>LamRA-Qwen2.5VL</td><td>7B</td><td>7.28</td><td>16.21</td><td>22.31</td><td>38.53</td><td>48.49</td><td>11.27</td><td>7.28</td><td>16.21</td><td>13.85</td><td>17.36</td><td>18.98</td></tr><tr><td>MM-EMBED</td><td>8B</td><td>12.98</td><td>26.88</td><td>34.50</td><td>53.66</td><td>62.37</td><td>18.94</td><td>12.98</td><td>20.15</td><td>22.61</td><td>26.86</td><td>28.67</td></tr><tr><td>CORAL</td><td>3B</td><td>8.58</td><td>17.10</td><td>22.88</td><td>39.30</td><td>47.73</td><td>12.37</td><td>8.58</td><td>12.98</td><td>14.83</td><td>18.43</td><td>19.80</td></tr></table>

$\operatorname { R @ K }$ （K=1,5,10,50,100），NDCG $@ \mathrm { K }$ 和 MRR $@$ 10。值以百分比表示。

![](images/3.jpg)  
Figure 3. Retrieval performance (Recall $@ 1 0$ diffent quey regis.The coplee results are provided  the supplemetar material.

# 5.2. 查询侧组合效应

查询侧模态消融。为了分析查询侧的模态贡献，我们在一个包含100个查询的子集上构建了受控的查询变体。我们从融合的自然语言查询开始，这些查询是根据每个产品的图像特征列表和文本描述元数据共同生成的。在仅文本条件下，我们使用ChatGPT-4o将每个融合查询与图像特征列表进行比较，并删除所有与视觉属性相关的短语，从而得到仅保留元数据派生约束的版本。对称地，在仅图像条件下，我们将融合查询与文本描述进行比较，并去除基于文本元数据的短语，仅保留视觉相关的属性。所有生成的查询三元组，包括融合、仅文本和仅图像，均经过人工检查以确认其准确性，同时候选池在所有设置中保持固定，使用融合的图像—文本元数据。

Table 4. NDCG $@ \mathrm { K }$ for pointwise rerankers on MCMR evaluated on the LLaVE-7B top-50 candidate pool.   

<table><tr><td>model</td><td>size</td><td>N@1</td><td>N@5</td><td>N@10</td><td>N@50</td></tr><tr><td>Qwen2.5-VL</td><td>32B</td><td>78.22</td><td>79.87</td><td>82.58</td><td>84.88</td></tr><tr><td>Qwen2.5-VL</td><td>7B</td><td>74.16</td><td>77.26</td><td>80.26</td><td>82.84</td></tr><tr><td>internVL</td><td>8b</td><td>80.28</td><td>81.95</td><td>84.66</td><td>86.61</td></tr><tr><td>Qwen3-VL</td><td>8B</td><td>72.45</td><td>75.48</td><td>78.32</td><td>81.44</td></tr><tr><td>Qwen3-VL</td><td>4B</td><td>69.92</td><td>73.39</td><td>76.81</td><td>79.95</td></tr><tr><td>Qwen3-VL-Reranker-8B</td><td>8B</td><td>78.69</td><td>80.79</td><td>83.51</td><td>85.57</td></tr><tr><td>lychee-reranker-mm</td><td>8B</td><td>92.35</td><td>93.41</td><td>94.42</td><td>94.86</td></tr></table>

图3总结了在$\textrm { R @ 1 0 }$下三种查询模式的检索性能：融合文本+图像查询、仅文本查询和仅图像查询。去除仅文本查询中的图像派生约束，会导致每个模型在早期排名上明显下降：与融合设置相比，$\mathrm { R @ 1 0 }$大约降低了2030个点。相比之下，在仅图像查询中去除文本派生约束的损失要小得多，在多个案例中，仅图像的得分仍然接近或在融合性能的上下大约510个点内。系统的排名在不同模式中也发生了变化：在融合查询下，MM-EMBED和LLaVE表现最强；在仅文本设置下，CORAL成为最佳表现者；而当仅有视觉证据可用时，LLaVE显然占据主导地位。总的来说，这些模式显示出所有系统在同时具备这两种模态时收益最大，但在被迫进入单一模态时，更依赖于图像线索，而非文本元数据。

查询约束数量的影响。我们进一步改变查询中的组合约束数量。令 $k _ { T }$ 和 $k _ { I }$ 分别表示文本派生约束和图像派生约束的数量。我们考虑配置，其中 $k _ { T } = k _ { I } \in \{ 1 , 2 , 3 , 4 , 5 \}$ ，使用 $1 \mathrm { T } { + } 1 \mathrm { I }$ 作为低约束基线，使用 $5 \mathrm { T } { + } 5 \mathrm { I }$ 作为最强约束设置，同时固定候选池以融合图像和文本元数据，并保持评估协议不变。对于每个 $k _ { T } = k _ { I }$ 的值，我们使用 ChatGPT-4o 从 image_feature 列表中选择 $k$ 个视觉基础属性，从 text_desc 中选择 $k$ 个元数据派生属性，并将它们组合成一个单一的自然语言查询。所有生成的查询都经过人工检查以确保忠实性和流畅性，所有查询均以文本形式编码。

图 4 显示了在不同数量的组合约束下，召回率 $@ 1 0$ 的变化情况，其中文本导出约束和图像导出约束的数量相等 $k _ { T } ~ = ~ k _ { I }$，范围从 $1 \mathrm { T } { + } 1 \mathrm { I }$ 到 $5 \mathrm { T } { + } 5 \mathrm { I }$，候选项使用融合的图像—文本元数据进行固定。4 所有模型的性能随着约束数量的增加而持续提升，尽管从 $4 \mathrm { T } { \cdot } 4 4 \mathrm { I }$ 到 $5 \mathrm { T } { + } 5 \mathrm { I }$ 的增幅小于从 $2 \mathrm { T } { + } 2 \mathrm { I }$ 到 $3 \mathrm { T } { + } 3 \mathrm { I }$ 的增幅，这表明在较高数量时收益递减。

![](images/4.jpg)  
Figure 4. Recall $@ 1 0$ under varying numbers of compositional constraints $( k _ { T } = k _ { I } \in { 1 , 2 , 3 , 4 , 5 } )$ . Candidates are fixed with fused image—text metadata.

# 5.3. 点对点重新排序性能

如表4所示，所有逐点重排序器在早期排名准确性方面均表现强劲，NDCG $@ 1$ 达到7080的范围；最佳模型InternVL-8B则达到80.3。当一个多模态大语言模型（MLLM）对查询和候选项进行一对一比较时，它可以通过明确的跨模态基础和条件验证可靠地识别正实例。这与第一阶段的检索器形成尖锐对比，其最佳的$\mathrm { N D C G } @ 1$仅为26.57（CORAL），揭示了显著的性能差距。重排序与第一阶段检索之间的巨大差距表明，当前的多模态检索器在细粒度的多条件匹配上仍然面临挑战。全局嵌入模型强调整体语义相似性，但往往无法验证所有查询条件是否同时满足。相比之下，重排序器独立评估每个查询-候选对，从而实现详细的跨模态基础和明确的约束检查。其权衡在于计算效率：逐点重排序代价较高，且无法扩展到大型检索语料库。这些发现凸显了未来系统的一个核心挑战——在检索模型中集成细粒度多条件多模态融合，同时保持现实世界应用所需的可扩展性。

# 6. 结论

在本研究中，我们引入了 MCMR（多条件多模态检索），这是一个用于评估自然语言查询下细粒度多条件检索的大规模基准测试。通过明确区分视觉和文本约束，MCMR 使得系统地评估模型在跨模态多个相互依赖条件下推理的能力成为可能。通过对多样化多模态检索器和基于大语言模型（MLLM）的重排序器进行全面实验，我们揭示了一致的模态不对称性——视觉特征主导了早期排名的精度，而文本元数据则稳定了长尾排序。此外，基于 MLLM 的逐点重排序器通过明确验证查询与候选项的一致性，显著提升了细粒度匹配，尽管这需要更高的计算成本。这些发现揭示了当前多模态检索系统的一个基本局限性：尽管具有强大的全局语义对齐能力，但在大规模集成具有组合性和条件敏感推理的能力方面仍然存在困难。我们希望 MCMR 能成为一个诊断性和挑战性的基准，激励未来的研究朝着在可扩展性、多模态基础和组合理解之间取得平衡的检索器方向发展。

# References

[1] Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, and Ehsaneddin Asgari. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 1677616809, Vienna, Austria, 2025. Association for Computational Linguistics. 1   
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5   
[3] Yingshan Chang, Guihong Cao, Mridu Narang, Jianfeng Gao, Hisami Suzuki, and Yonatan Bisk. Webqa: Multihop and multimodal QA. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1647416483, 2022. 3   
[4] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can pre-trained vision and language models answer visual information-seeking questions? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 1494814968, Singapore, 2023. Association for Computational Linguistics. 3   
[5] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2418524198, 2024. 5 [6] Wei Chow, Yuan Gao, Linfeng Li, Xian Wang, Qi Xu, Hang Song, Lingdong Kong, Ran Zhou, Yi Zeng, Yidong Cai, Botian Jiang, Shilin Xu, Jiajun Zhang, Minghui Qiu, Xi Li Tianshu Y, Silig Tg, and Jun Li. Merit: Multilingual semantic retrieval with interleaved multi-condition query. arXiv preprint arXiv:2506.03144, 2025. Introduces the CoRAL fine-tuning framework for multimodal retrieval. 5 [7] Wei Chow, Yuan Gao, Linfeng Li, Xian Wang, Qi Xu, Hang Song, Lingdong Kong, Ran Zhou, Yi Zeng, Yidong Cai, Botian Jiang, Shilin Xu, Jiajun Zhang, Minghui Qiu, Xiangtai Li, Tianshu Yang, Siliang Tang, and Juncheng Li.Merit: Multilingual semantic retrieval with interleaved multi-condition query, 2025. 2, 3 [8] Ziqi Dai, Xin Zhang, Mingxin Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Supervised fine-tuning or contrastive learning? towards better multimodal llm reranking, 2025. 5 [9] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. Colpali: Efficient document retrieval with vision language models. In The Thirteenth International Conference on Learning Representations, 2025. 3   
[10] Xintong Han, Zuxuan Wu, Phoenix X. Huang, Xiao Zhang, Menglong Zhu, Yuan Li, Yang Zhao, and Larry S. Davis. Automatic spatially-aware fashion concept discovery. In IEEE International Conference on Computer Vision, ICCV 2017, pages 14721480, Venice, Italy, 2017. IEEE Computer Society. 2, 3   
[11] Xintian Han, Honggang Chen, Quan Lin, Jingyue Gao, Xiangyuan Ren, Lifei Zhu, Zhisheng Ye, Shikang Wu, Xiong-Hang Xie, Xiaochu Gan, Bingzheng Wei, Peng Xu, Zhe Wang, Yuchao Zheng, Jingjian Lin, Di Wu, and Junfeng Ge. Lemur: Large scale end-to-end multimodal recommendation, 2025. 1   
[12] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952, 2024. 3   
[13] Hexiang Hu, Yi Luan, Yang Chen, Urvashi Khandelwal, Mandar Joshi, Kenton Lee, Kristina Toutanova, and Ming-Wei Chang. Open-domain visual entity recognition: Towards recognizing millions of wikipedia entities. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, pages 1203112041, Paris, France, 2023. IEEE. 3   
[14] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 49044916. PMLR, 2021. 1, 3   
[15] Yiming Jia, Jiachen Li, Xiang Yue, Bo Li, Ping Nie, Kai Zou, and Wenhu Chen. VisualWebInstruct: Scaling up multimodal instruction data through web search. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 13731393, Suzhou, China, 2025. Association for Computational Linguistics. 1   
[16] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. E5-V: universal embeddings with multimodal large language models. CoRR, abs/2407.12580, 2024. 3   
[17] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen. Vlm2vec: Training vision-language models for massive multimodal embedding tasks. In The Thirteenth International Conference on Learning Representations, 2024. 1, 2, 3, 5   
[18] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. LLaVE: Large language and vision embedding models with hardness-weighted contrastive learning. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 1372113735, Suzhou, China, 2025. Association for Computational Linguistics. 2, 5   
[19] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning, pages 1288812900. PMLR, 2022. 1, 3   
[20] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, S SongShuai Bi Zhib ang, e Xi A , Dayiheng Liu, Jingren Zhou, and Junyang Lin. Qwen3- vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv, 2026.5   
[21] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. Mm-embed: Universal multimodal retrieval with multimodal llms. In The Thirteenth International Conference on Learning Representations, 2024. 1, 2   
[22] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In 13th European Conference on Computer Vision, ECCV 2014, pages 740755, Zurich, Switzerland, 2014. Springer. 2, 3   
[23] Weizhe Lin, Jingbiao Mei, Jinghong Chen, and Bill Byrne. PreFLMR: Scaling up fine-grained late-interaction multimodal retrievers. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 52945316, Bangkok, Thailand, 2024. Association for Computational Linguistics. 3   
[24] Fuxiao Liu, Yinghan Wang, Tianlu Wang, and Vicente Ordonez. Visual news: Benchmark and challenges in news image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 67616771, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. 3   
[25] Qidong Liu, Jiaxi Hu, Yutian Xiao, Xiangyu Zhao, Jingtong Gao, Wanyu Wang, Qing Li, and Jiliang Tang. Multimodal recommender systems: A survey. ACM Comput. Surv., 57 (2), 2024. 1   
[26] Siq1 Liu, Weix1 Feng, T'su-Jui Fu, Wenhu Chen, and William Wang. EDIS: Entity-driven image search over multimodal web content. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 48774894, Singapore, 2023. Association for Computational Linguistics. 3   
[27] Yikun Liu, Pingan Chen, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. Lamra: Large multimodal model as your advanced retrieval assistant. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 40154025. IEEE/CVF, 2025. 2, 5   
[28] Zheyuan Liu, Cristian Rodriguez Opazo, Damien Teney, and Stephen Gould. Image retrieval on real-life images with pre-trained vision-and-language models. In 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, pages 21052114, Montreal, Canada, 2021. IEEE. 2, 3   
[29] Zheyuan Liu, Weixuan Sun, Yicong Hong, Damien Teney, and Stephen Gould. Bi-directional training for composed image retrieval via text prompt learning, 2023. 1   
[30] Zhenghao Liu, Chenyan Xiong, Yuanhuiyi Lv, Zhiyuan Liu, and Ge Yu. Universal vision-language dense retrieval: Learning a unified representation space for multi-modal retrieval. In The Eleventh International Conference on Learning Representations, 2023. 3   
[31] Xuan Lu, Haohang Huang, Rui Meng, Yaohui Jin, Wenjun Zeng, and Xiaoyu Shen. Rethinking reasoning in document ranking: Why chain-of-thought falls short. arXiv preprint arXiv:2510.08985, 2025. 1   
[32] Xuan Lu, Sifan Liu, Bochao Yin, Yongqi Li, Xinghao Chen, Hui Su, Yaohui Jin, Wenjun Zeng, and Xiaoyu Shen. MultiConIR: Towards multi-condition information retrieval. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 1347113494, Suzhou, China, 2025. Association for Computational Linguistics. 2, 3   
[33] Xuan Lu, Haohang Huang, Rui Meng, Yaohui Jin, Wenjun Zeng, and Xiaoyu Shen. Tools are under-documented: Simple document expansion boosts tool retrieval. In The Fourteenth International Conference on Learning Representations, 2026. 2   
[34] Man Luo, Zhiyuan Fang, Tejas Gokhale, Yezhou Yang, and Chitta Baral. End-to-end knowledge retrieval with multimodal queries. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 85738589, Toronto, Canada, 2023. Association for Computational Linguistics. 3   
[35] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. OK-VQA: A visual question answering benchmark requiring external knowledge. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, pages 31953204, Long Beach, CA, USA, 2019. Computer Vision Foundation / IEEE. 3   
[36] Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In 2015 IFFF. International Conference on Comnuter Vision ICCV 2015, pages 26412649, Santiago, Chile, 2015. IEEE Computer Society. 2, 3   
[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, SandhiniAgarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 87488763. PmLR, 2021. 1, 3   
[38] Xiaoyu Shen, Svitlana Vakulenko, Marco del Tredici, Gianni Barlacchi, Bill Byrne, and Adrià de Gispert. Low-resource dense retrieval for open-domain question answering: A comprehensive survey, 2022. 1   
[39] Xiaoyu Shen, Akari Asai, Bill Byrne, and Adria De Gispert. xPQA: Cross-lingual product question answering in 12 languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track), pages 103115, Toronto, Canada, 2023. Association for Computational Linguistics. 1   
[40] Xiaoyu Shen, Svitlana Vakulenko, Marco del Tredici, Gianni Barlacchi, Bill Byrne, and Adria de Gispert. Neural ranking with weak supervision for open-domain question answering : A survey. In Findings of the Association for Computational Linguistics: EACL 2023, pages 17361750, Dubrovnik, Croatia, 2023. Association for Computational Linguistics. 1   
[41] Tianshi Wang, Fengling Li, Lei Zhu, Jingjing Li, Zheng Zhang, and Heng Tao Shen. Cross-modal retrieval: A systematic review of methods and future directions, 2024. 1   
[42] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, Alan Ritter, and Wenhu Chen. Uniir: Training and benchmarking universal multimodal information retrievers. In 18th European Conference on Computer Vision, page 387404, Milan, Italy, 2024. Springer-Verlag. 3   
[43] Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogério Feris. Fashion IQ: A new dataset towards retrieving images by natural language feedback. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, pages 1130711317. Computer Vision Foundation / IEEE, 2021. 3   
[44] Bingbing Zhang, Yi Han, and Xiaofei Han. Research on multi-modal retrieval system of e-commerce platform based on pre-training model. Artificial Intelligence Technology Research, 2(9), 2025. 1   
[45] Xin Zhang, Mingxin Li, Yanzhao Zhang, Dingkun Long, Yongqi Li, Yinghui Li, Pengjun Xie, Meishan Zhang, Wenjie Li, Min Zhang, et al. Ssrb: Direct natural language querying to massive heterogeneous semi-structured data. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. 2   
[46] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. Bridging modalities: Improving universal multimodal retrieval by multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 92749285, 2025. 2, 5   
[47] Junjie Zhou, Zheng Liu, Shitao Xiao, Bo Zhao, and Yongping Xiong. VISTA: Visualized text embedding for universal multi-modal retrieval. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 31853200, Bangkok, Thailand, 2024. Association for Computational Linguistics. 3   
[48] Jianqun Zhou, Yuanlei Zheng, Wei Chen, Qianqian Zheng, Hui Su, Wei Zhang, Rui Meng, and Xiaoyu Shen. Beyond content relevance: Evaluating instruction following in retrieval models. arXiv preprint arXiv:2410.23841, 2024. 2, 3   
[49] Tianshuo Zhou, Sen Mei, Xinze Li, Zhenghao Liu, Chenyan Xiong, Zhiyuan Liu, Yu Gu, and Ge Yu. MARVEL: Unlocking the multi-modal capability of dense retrieval via visual module plugin. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1460814624, Bangkok, Thailand, 2024. Association for Computational Linguistics. 3

# Beyond Global Similarity: Towards Fine-Grained, Multi-Condition Multimodal Retrieval

Supplementary Material

# 7. Details of Baselines

For each model used in this paper, Tab. 5 summarizes the parameter size, architecture, whether we use an explicit retrieval instruction, the maximum input context length, and the underlying backbone checkpoint. We group systems into retrievers and rerankers according to their role in our two-stage pipeline.

Tab. 5: Details of retriever and reranker models used in experiments. Size denotes the number of parameters of each model. Architecture indicates the model type used in our setup (all models are operated in decoder-only mode). Instruction specifies whether we prepend a task-specific natural-language prefix when encoding queries (for example, retrieval prompts or templates that produce an <emb> token), rather than feeding bare queries. Max length denotes the maximum input length in tokens used for each model in our experiments, typically matching the backbone context window. Backbone identifies the pretrained checkpoint on which each retriever or reranker is built.

# 8. Examples of MCMR

Tab. 6 presents examples from the five MCMR sub-datasets, covering upper clothing, bottom clothing, shoes, jewelry, and furniture.

# 9. Complete Results

Tab. 7 reports the full numerical results of our query-side modality ablation study. For each of the five retrievers evaluated in our main experiments, we provide detailed scores under three query regimes: (i) fused queries containing both image-derived and text-derived constraints, (ii) textonly queries obtained by removing all image-grounded constraints, and (ii) image-only queries obtained by removing all text-grounded constraints. Each subtable reports Recall $@ \mathrm { K }$ , MRR, and ${ \mathrm { N D C G } } @ { \mathrm { K } }$ across a broad range of cutoffs, enabling fine-grained inspection of early-rank precision as well as long-range ordering behavior. These full results make clear how different models respond to the removal of cross-modal evidence.

Tab. 8 reports the full numerical results for the queryside compositional constraint study described in $\ S 4 . 3$ For each retriever (CORAL, GME, LamRA, LLaVE, and MM-Embed) and for each configuration of the text- and imagederived constraint counts, we list Recall $@ \mathrm { K }$ , MRR, and ${ \mathrm { N D C G } } @ { \mathrm { K } }$ in percentage form. We consider matched constraint settings with $k _ { T } = k _ { I } \in \{ 1 , 2 , 3 , 4 , 5 \}$ while keeping the fused candidate pool and evaluation protocol identical to the main experiments. The main text focuses on the trends for $k _ { T } = k _ { I } \in \{ 2 , 3 , 4 , 5 \}$ ; here we additionally include the $1 \mathrm { T } + 1 \mathrm { I }$ configuration for completeness.

# 10. Prompts

# 10.1. Prompts for Constructing MMR

Step 1: Image-side Attribute Extraction Fig. 5 shows the prompt template used to generate fine-grained visual attribute descriptions from each product image. The model is instructed to output structured, image-grounded features only, without inferring hidden or functional properties.

Step 2: Text-side Description Generation Fig. 6 illustrates the prompt design for generating textual product descriptions from metadata. It enforces strict separation from visual evidence and standardizes phrasing for key attributes such as price and release date.

Step 3: Universal Leakage Checker Fig. 7 shows the verification prompt used to detect visual-to-text leakage after description generation. It flags any exact or paraphrased overlap between image_feature and text_desc, as well as category mismatches. This step guarantees that textside descriptions remain strictly text-grounded before query synthesis.

Step 4: Query Generation Fig. 8 illustrates the prompt used to simulate realistic customer queries. The model combines visual and textual attributes under strict composition rules, ensuring each query naturally expresses cross-modal information while maintaining linguistic diversity and authenticity.

Step 5: Query Check. Fig. 9 presents the verification prompt that filters low-quality or unfaithful queries. The model checks for balanced modality coverage (at least two image-derived and one text-derived matches), valid price and date phrasing, and normalization issues such as unit inconsistencies or brand leakage. Only queries passing all logical gates are retained for dataset construction.

Pointwise. Fig. 10 shows the pointwise prompt used to score query-candidate pairs during reranking.

Table 5. Details of retriever and reranker models used in our experiments.   
Figure 5. Prompt for Image Attribute Extraction used in image-side annotation.   

<table><tr><td>Model</td><td>Size</td><td>Architecture</td><td>Instruction</td><td>Max length</td><td>Backbone</td></tr><tr><td colspan="6">Retriever</td></tr><tr><td>GME-Qwen2-VL-7B-Instruct</td><td>7B</td><td>Decoder</td><td>Yes</td><td>32K</td><td>Qwen2-VL-7B-Instruct</td></tr><tr><td>LLaVE-7B</td><td>7B</td><td>Decoder</td><td>Yes</td><td>32K</td><td>LLaVA-OneVision-7B</td></tr><tr><td>LamRA-Ret-Qwen2.5VL-7B</td><td>7B</td><td>Decoder</td><td>Yes</td><td>128K</td><td>Qwen2.5-VL-7B</td></tr><tr><td>MM-Embed</td><td>8B</td><td>Decoder</td><td>Yes</td><td>32K</td><td>NV-Embed</td></tr><tr><td>CORAL</td><td>3B</td><td>Decoder</td><td>No</td><td>128K</td><td>Qwen2.5-3B-Instruct</td></tr><tr><td>VLM2VEC</td><td>4B</td><td>Decoder</td><td>No</td><td>131K</td><td>Phi3.5V</td></tr><tr><td colspan="6">MLLM-as-Reranker</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>7B</td><td>Decoder</td><td>Yes</td><td>128K</td><td>Qwen2.5-VL-7B-Instruct</td></tr><tr><td>Qwen2.5-VL-32B-Instruct</td><td>32B</td><td>Decoder</td><td>Yes</td><td>128K</td><td>Qwen2.5-VL-32B-Instruct</td></tr><tr><td>InternVL3-8B-Instruct</td><td>32B</td><td>Decoder</td><td>Yes</td><td>32K</td><td>InternVL3-8B-Instruct</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>4B</td><td>Decoder</td><td>Yes</td><td>262K</td><td>Qwen3-VL-4B-Instruct</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>8B</td><td>Decoder</td><td>Yes</td><td>262K</td><td>Qwen3-VL-8B-Instruct</td></tr></table>

# Prompt for Image Attribute Extraction

You are a meticulous vision annotator.   
Task   
1. Identify the product category you see in one or two lowercase words.   
2. List short appearance descriptors capturing fine-grained, objective visual details of the product only.   
Rules   
Output exactly one JSON array: the category first, then the descriptors.   
Aim for 410 descriptors; if fewer are certain, output only those (do not guess).   
Each descriptor $\leq 8$ English words, all lowercase, American spelling.   
Describenly what is learly visible;do notmention ize, price release date, perormancercomor claims. is visibly printed.   
c/tetu ttahc/part/ve/ti sure/hardware (e.g., zipper, buckle, laces, clasp), shape/silhouette, logos/text when readable.   
expeceheex weext  ooovanal. Exclude background or other items not part of the product (e.g., other garments, props, body parts).   
Skip any feature you are not $100 \%$ certain is visible; do not infer hidden details.   
Before you output, ensure coverage of at least 4 of these slots if visible:{metal color/finish}, {stone color &shape/cut},{settig/arrangement},{band/chain/braceletdetails},{closure/clasp},{engraving/motif/overlays}, {logos/text/hallmarks}.   
Return only the JSON array (no prose).

# 11. Human Evaluation Protocol

We conduct a small-scale human study to verify the naturalness and attribute fidelity of the generated queries. We randomly sample 100 target products from all domains. For each product, Annotator A is given the product image, its structured metadata, and the internally extracted finegrained attributes, and is asked to write a natural-language search query that a user might realistically issue—without access to the model-generated query.

Annotator B then evaluates, under double-blind conditions, both the human-written and model-generated queries.

Table 6. Examples of retrieval samples from our MCMR benchmark.   

<table><tr><td colspan="4"></td></tr><tr><td>Category</td><td>Query Text</td><td>Target Text</td><td>Target Image</td></tr><tr><td>Upper</td><td>I&#x27;m looking for a men&#x27;s jacket in gray with a plaid pattern and green accents, size L. It should be waterproof with a hood and front pockets, made of durable nylon twill that needs hand washing. Prefer something released around 2013 and priced about $200.</td><td>&quot;title&quot;: &quot;Columbia Men&#x27;s Whirlibird III Interchange Jacket&quot;, &quot;description&quot;: &quot;Three jackets in one  a warm, repellent Omni-Heat liner; a waterproof-breathable and critically seam-sealed shell; and a combination of both  giving you ultimate versatility to stay warm, dry, protected and comfortable in fluctuating winter conditions.&quot;, &quot;price&quot;: 200.0, ..</td><td></td></tr><tr><td>Bottom</td><td>I&#x27;m looking for a pair of men&#x27;s brown cotton jeans with a slim, straight-leg fit. I&#x27;d like a flat front and zipper fly, made from 100% cotton denim that&#x27;s soft, durable, and machine washable. Perfect for casual or work wear, ideally under $30 and released around 2021.</td><td>&quot;title&quot;: &quot;World of Leggings Plus Size Spandex Knee High Boy Shorts - Shop 16 Colors&quot;, &quot;description&quot;: &quot;Our knee high and seamless plus size boy shorts are a one size seamless nylon spandex plus size leg piece that are a must for any woman&#x27;s leggings wardrobe. They are made with high quality stitching and have fantastic stretch ….</td><td></td></tr><tr><td>Shoes</td><td>I&#x27;m looking for men&#x27;s brown leather high-top work boots with a lace-up closure and a cushioning OrthoLite footbed. They should have slip-resistant soles and meet ASTM EH safety standards for electrical hazard protection. I&#x27;d like a pair under $260.</td><td>&quot;title&quot;: &quot;Danner Men&#x27;s Bull Run 8Work Boot&quot;, &quot;description&quot;: &quot;Built in the USA with a dedication to quality that goes back to 1932, the Bull Run is a utilitarian work boot with a timeless design that stays in style when you punch out. The full-grain leather upper is the perfect blend of strong and soft. Combine that with the sturdy Danner Wedge outsole and you get all-day comfort that lasts.&quot;, &quot;price&quot;: 259.95, &quot;features&quot;: &quot;100% Leather&quot;, ..</td><td></td></tr><tr><td>Jewelry</td><td>I&#x27;m looking for a silver-tone bracelet with a Cuban link chain and slide clasp. Please show options in 14k gold over 925 sterling silver, about $450, released around 2022.</td><td>&quot;title&quot;: &quot;SAVEARTH DIAMONDS 5.80 Ct to 10.90 Ct Lab Created Moissanite Diamond 6MM Width Cuban Link Chain Necklace For Men In 14k Gold Over 925 Sterling Silver, 16to 30Length, Color: G-H, Clarity: VVS1&quot;, &quot;description&quot;: &quot;Jewelry has the power to be this one little thing that can make you feel unique, ..&#x27; &quot;title&quot;: &quot;CYNOSA Halloween Decorations</td><td></td></tr><tr><td>Furniture</td><td>I&#x27;m looking for some cute rustic Halloween decorations for my home, i want a tiered-tray setup with small wooden signs and fall-themed accents, under $20.</td><td>Halloween Tiered Tray Decor Fall Decor Hocus Pocus I Smell Children Boo Wooden Signs and Orange Plaid Gnomes Plush Farmhouse Rustic Tiered Tray Decor for Home Table&quot;, &quot;description&quot;: &quot;Package including: 1 x Black and Orange Check Plaid Gnome; 1 x Round Shape Sign Smell Children; 1 x Square Shape Halloween Themed Sign Öctober 31; 1 x Rectangle Shape Black and Orange Plaid Sign (Boo!); ..</td><td></td></tr></table>

<table><tr><td>model</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>MRR</td><td>N@5</td><td>N@10</td><td>N@50</td></tr><tr><td>LLaVE</td><td>22.00</td><td>38.00</td><td>46.00</td><td>73.00</td><td>29.09</td><td>30.53</td><td>33.10</td><td>39.39</td></tr><tr><td>GME-Qwen2VL</td><td>9.00</td><td>34.00</td><td>42.00</td><td>64.00</td><td>19.50</td><td>22.32</td><td>24.91</td><td>29.81</td></tr><tr><td>LamRA-Qwen2.5VL</td><td>13.00</td><td>31.00</td><td>39.00</td><td>63.00</td><td>21.14</td><td>22.73</td><td>25.43</td><td>30.67</td></tr><tr><td>MM-EMBED</td><td>20.00</td><td>40.00</td><td>47.00</td><td>64.00</td><td>27.58</td><td>29.94</td><td>32.19</td><td>35.88</td></tr><tr><td>CORAL</td><td>24.00</td><td>39.00</td><td>45.00</td><td>71.00</td><td>30.67</td><td>32.22</td><td>34.11</td><td>39.86</td></tr><tr><td colspan="9">(b) Text-only</td></tr><tr><td>model</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>MRR</td><td>N@5</td><td>N@10</td><td>N@50</td></tr><tr><td>LLaVE</td><td>2.00</td><td>12.00</td><td>17.00</td><td>30.00</td><td>6.54</td><td>7.40</td><td>9.02</td><td>11.98</td></tr><tr><td>GME-Qwen2VL</td><td>6.00</td><td>16.00</td><td>24.00</td><td>45.00</td><td>11.09</td><td>11.65</td><td>14.08</td><td>18.43</td></tr><tr><td>LamRA-Qwen2.5VL</td><td>2.00</td><td>8.00</td><td>15.00</td><td>27.00</td><td>4.77</td><td>4.92</td><td>7.10</td><td>9.68</td></tr><tr><td>MM-EMBED</td><td>12.00</td><td>29.00</td><td>33.00</td><td>47.00</td><td>18.72</td><td>20.90</td><td>22.18</td><td>25.47</td></tr><tr><td>CORAL</td><td>9.00</td><td>16.00</td><td>18.00</td><td>30.00</td><td>12.00</td><td>12.82</td><td>13.45</td><td>16.05</td></tr><tr><td colspan="9">(c) Image-only</td></tr><tr><td>model</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@50</td><td>MRR</td><td>N@5</td><td>N@10</td><td>N@50</td></tr><tr><td>LLaVE</td><td>20.00</td><td>33.00</td><td>41.00</td><td>65.00</td><td>26.04</td><td>27.06</td><td>29.55</td><td>35.07</td></tr><tr><td>GME-Qwen2VL</td><td>13.00</td><td>25.00</td><td>33.00</td><td>56.00</td><td>18.07</td><td>18.91</td><td>21.57</td><td>26.41</td></tr><tr><td>LamRA-Qwen2.5VL</td><td>12.00</td><td>30.00</td><td>38.00</td><td>60.00</td><td>19.11</td><td>20.94</td><td>23.57</td><td>28.42</td></tr><tr><td>MM-EMBED</td><td>14.00</td><td>28.00</td><td>33.00</td><td>50.00</td><td>19.42</td><td>21.09</td><td>22.64</td><td>26.59</td></tr><tr><td>CORAL</td><td>10.00</td><td>31.00</td><td>37.00</td><td>54.00</td><td>18.82</td><td>21.27</td><td>23.21</td><td>26.81</td></tr></table>

n

Each query is assessed on three dimensions using a 15 Likert scale: (i) attribute correctness, (ii) cross-modal coverage, and (iii) naturalness and clarity. Annotator B additionally selects which query better matches the target product.

Overall, model-generated queries achieve scores comparable to human-written ones across all criteria, with only a small gap in naturalness. The near-equal preference distribution indicates no strong annotator bias toward either source. These results confirm that the proposed generation pipeline produces high-quality, multi-condition queries suitable for cross-modal retrieval research.

<table><tr><td rowspan="2">model</td><td rowspan="2"></td><td colspan="4">Recall@K (%)</td><td rowspan="2">MRR (%)</td><td colspan="3">NDCG@K (%)</td></tr><tr><td>1</td><td>5</td><td>10</td><td>50</td><td>5</td><td>10</td><td>50</td></tr><tr><td rowspan="5">CORAL</td><td>1t1i</td><td>9.00</td><td>15.00</td><td>22.00</td><td>41.00</td><td>12.38</td><td>12.30</td><td>14.61</td><td>18.84</td></tr><tr><td>2t2i</td><td>15.00</td><td>31.00</td><td>38.00</td><td>57.00</td><td>22.09</td><td>23.64</td><td>25.89</td><td>30.11</td></tr><tr><td>3t3i</td><td>21.00</td><td>36.00</td><td>40.00</td><td>62.00</td><td>27.25</td><td>29.06</td><td>30.34</td><td>35.10</td></tr><tr><td>4t4i</td><td>26.00</td><td>42.00</td><td>48.00</td><td>68.00</td><td>33.21</td><td>34.86</td><td>36.77</td><td>41.13</td></tr><tr><td>5t5i</td><td>33.00</td><td>50.00</td><td>57.00</td><td>79.00</td><td>40.19</td><td>41.94</td><td>44.19</td><td>48.93</td></tr><tr><td rowspan="5">GME</td><td>1t1i</td><td>2.00</td><td>18.00</td><td>20.00</td><td>36.00</td><td>7.92</td><td>10.22</td><td>10.87</td><td>14.32</td></tr><tr><td>2t2i</td><td>10.00</td><td>23.00</td><td>28.00</td><td>46.00</td><td>15.75</td><td>17.03</td><td>18.70</td><td>22.67</td></tr><tr><td>3t3i</td><td>12.00</td><td>32.00</td><td>34.00</td><td>52.00</td><td>19.43</td><td>22.32</td><td>23.01</td><td>27.13</td></tr><tr><td>4t4i</td><td>18.00</td><td>39.00</td><td>42.00</td><td>59.00</td><td>25.78</td><td>28.73</td><td>29.73</td><td>33.55</td></tr><tr><td>5t5i</td><td>21.00</td><td>42.00</td><td>48.00</td><td>68.00</td><td>29.40</td><td>31.93</td><td>33.87</td><td>38.36</td></tr><tr><td rowspan="5">LamRA</td><td>1t1i</td><td>5.00</td><td>12.00</td><td>21.00</td><td>42.00</td><td>8.51</td><td>8.49</td><td>11.38</td><td>15.94</td></tr><tr><td>2t2i</td><td>8.00</td><td>19.00</td><td>25.00</td><td>50.00</td><td>13.06</td><td>13.90</td><td>15.90</td><td>21.38</td></tr><tr><td>3t3i</td><td>8.00</td><td>24.00</td><td>31.00</td><td>53.00</td><td>14.78</td><td>16.31</td><td>18.67</td><td>23.80</td></tr><tr><td>4t4i</td><td>12.00</td><td>29.00</td><td>39.00</td><td>63.00</td><td>19.48</td><td>20.81</td><td>24.08</td><td>29.41</td></tr><tr><td>5t5i</td><td>19.00</td><td>35.00</td><td>44.00</td><td>70.00</td><td>26.66</td><td>27.84</td><td>30.79</td><td>36.52</td></tr><tr><td rowspan="5">LLaVE</td><td>1t1i</td><td>5.00</td><td>23.00</td><td>29.00</td><td>50.00</td><td>12.55</td><td>14.57</td><td>16.49</td><td>21.14</td></tr><tr><td>2t2i</td><td>16.00</td><td>32.00</td><td>39.00</td><td>57.00</td><td>21.83</td><td>23.49</td><td>24.54</td><td>29.67</td></tr><tr><td>3t3i</td><td>19.00</td><td>37.00</td><td>43.00</td><td>68.00</td><td>25.98</td><td>28.06</td><td>30.04</td><td>35.45</td></tr><tr><td>4t4i</td><td>21.00</td><td>42.00</td><td>49.00</td><td>69.00</td><td>28.93</td><td>31.40</td><td>33.72</td><td>38.10</td></tr><tr><td>5t5i</td><td>27.00</td><td>50.00</td><td>59.00</td><td>72.00</td><td>35.89</td><td>38.50</td><td>41.35</td><td>44.28</td></tr><tr><td rowspan="5">MM-EMBED</td><td>1t1i</td><td>8.00</td><td>18.00</td><td>20.00</td><td>40.00</td><td>12.06</td><td>13.36</td><td>13.98</td><td>18.57</td></tr><tr><td>2t2i</td><td></td><td>22.00</td><td></td><td>44.00</td><td>17.31</td><td>18.03</td><td>19.55</td><td>22.09</td></tr><tr><td>3t3i</td><td>14.00 19.00</td><td>29.00</td><td>27.00 34.00</td><td>49.00</td><td>22.96</td><td>23.96</td><td>25.55</td><td>28.80</td></tr><tr><td>4t4i</td><td>23.00</td><td>40.00</td><td>45.00</td><td>64.00</td><td>30.04</td><td>31.96</td><td>33.65</td><td>37.83</td></tr><tr><td>5t5i</td><td>30.00</td><td>43.00</td><td>48.00</td><td>68.00</td><td>36.30</td><td>37.43</td><td>39.15</td><td>43.67</td></tr></table>

Table 8. Query-side modality ablation under compositional constraints $( k _ { T } = k _ { I } \in \{ 1 , 2 , 3 , 4 , 5 \} )$ Values are percentages.

Table 9. Background and qualifications of annotators involved in the human evaluation study.   

<table><tr><td>Annotator</td><td>Background</td><td>English Proficiency</td></tr><tr><td>Annotator A</td><td>B.A. in Linguistics</td><td>IELTS 7.0</td></tr><tr><td>Annotator B</td><td>M.S. in Computer Science</td><td>TOEFL 99</td></tr></table>

Table 10. Summary of human evaluation comparing humanwritten vs. generated queries.   

<table><tr><td>Metric</td><td>Human</td><td>Generated</td></tr><tr><td>Attribute correctness</td><td>4.45</td><td>4.37</td></tr><tr><td>Cross-modal coverage</td><td>4.29</td><td>4.28</td></tr><tr><td>Naturalness &amp; clarity</td><td>4.48</td><td>4.33</td></tr><tr><td>Average score</td><td>4.41</td><td>4.33</td></tr><tr><td>Preference rate</td><td>49%</td><td>47%</td></tr></table>

Task: Generate a concise English description for a product from its JSON metadata.

l NOT mention or paraphrase): image

Hard rules

• Write 24 clear sentences, total 80120 words, single paragraph.

hee iivan.

•Always mention price and first-availability time with strict phrasing:

Price: strictly "priced at $\$ 123,456$ . If absent: "price information not provided"

Date: strictly "released Month Day, Year". If absent: "release date not provided".

t, gloss, matte, style cues).

Treat every item in forbidden_visuals as banned; do not include those words or their paraphrases.

ep

Exlude shipping service, and marketing slogans; compress long size tables into concise ranges whennecesary.

•Output only the final paragraph, no notes or reasoning.

Quality gate (internal)

Remove any overlap with forbidden_visuals or residual visual wording.

origin).

If fields conflict, prefer "features" over "description". Never invent facts.

Input JSON template

# Prompt for Cross-Modal Leakage Detection

Return ONE valid JSON object ONLY, no prose.

Use caseinsensitive matching. Treat hyphen and space as equivalent. Handle simple singular or plural.

Catory:I the rs imagefeature item is a generic category inore it or leakage counting, but  tegory_conflic=rue only f textdesc clearly names adifferent catory.

D l metal". Be conservative: mark paraphrase only when meaning is the same.

scheatrue alaesature "yect, prase ice ], "category_conflict": true, false

Input: image feature: [ .] text_desc:

Return JSON now.

You are a real shopper typing in the search box of a large e-commerce site.

Wriu r quyual s-per vis  nd ehe tal len e 35-60 words.

Sources

Description $:$ desc

Image tags $:$ image

Hard rules — all must be met

Blend both sources $- i$ Useexacty -3isal ag nd exacty-3 deitin facts, wit oveaps.

pa te sc ral pr ahmori patOherieyhe bas Ment exactly netareizenl  ret, nevers ultipleize  ng.

the exact price.

R the sources, never use exact dates.

Use first-person tone throughout.

No details beyond the two sources.

Output only the query text, no bullet points, quotes, or backslashes.

Materials phrasing

W he et t pre peag r hei ualtativ a

tc i  oye itoste cot o blend", "nylon-rich", "wool blend". Each such materials phrase counts as one description fact.

Style hints

Emphasize fabric, fit, closure, care before marketing fluff

Vary phrasing so each query feels unique.

# Prompt for Query Quality Judge

You are a strict e-commerce query judge. Output ONE valid JSON object only, no extra text.   
Inuts:query, mageeature listvisual phrases), text_des metadata, candidate_price, candidateyear.   
.   
.   
toexpensive. n cear price phraseor candidate_price is missig, treat pric s ine.   
date as in range.   
ligtly editd version of the query that onlyfixes clear issues without adding newiormaton.   
Return the final JSON object now.

System.

You are a strict relevance judge. Given a user query and a candidate (image $^ +$ textual attributes), answer with a ingle TulTrumeans he ndiatches he que hulalherwisoher wor, punctuation.

User. <image>

Query: {query_text} Candidate: {candidate_text} Does the candidate match the query? True or False