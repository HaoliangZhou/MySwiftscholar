# 轻量级提示引导的 CLIP 适配用于单目深度估计

Reyhaneh Ahani Manghotay1 和 Jie Liang2\* $\bot$ 工程科学学院，西蒙弗雷泽大学，加拿大，不列颠哥伦比亚省，伯纳比 raa112@sfu.ca $^ 2$ 信息科学与技术学院，东部科技学院，中国宁波 Jliang@eitech.edu.cn

摘要。利用视觉-语言模型（VLM）如 CLIP 的丰富语义特征进行单目深度估计任务是一种有前景的方向，但通常需要大量微调或缺乏几何精度。我们提出了一种名为 MoA-DepthCLIP 的参数高效框架，利用最小化监督来适应预训练的 CLIP 表示用于单目深度估计。我们的方法将一个轻量级的混合适配器（MoA）模块集成到预训练的视觉变换器（ViT-B/32）主干网络中，并选择性地微调最后几层。该设计实现了空间感知的适应，受全球语义上下文向量的指导，并结合了一种将深度区间分类与直接回归相结合的混合预测架构。为增强结构准确性，我们采用了一种复合损失函数，以强制执行几何约束。在 NYU Depth V2 基准测试中，MoA-DepthCLIP 实现了竞争力的结果，显著超越了 DepthCLIP 基线，将 $\delta _{1}$ 准确率从 0.390 提高到 0.745，并将均方根误差（RMSE）从 1.176 降低到 0.520。这些结果是在需要极少可训练参数的情况下获得的，表明轻量化、提示引导的 MoA 是一种将 VLM 知识转移到细粒度单目深度估计任务的高效策略。关键词：单目深度估计 · CLIP 适应 · 混合适配器 · 参数高效微调

# 1 引言

VLM的出现，尤其是CLIP，彻底改革了计算机视觉和自然语言处理领域。通过从大量图像-文本对中学习联合表示，这些模型在零样本和少样本学习任务中展现出了卓越的能力，如图像分类，并成功适配于密集预测问题，如目标检测和分割。它们的影响甚至扩展到了3D感知，应用于点云理解。然而，一个根本的挑战依然存在：将VLM中封装的高级语义知识转化为几何任务所需的细粒度度量预测，如单目深度估计。单目深度估计是计算机视觉中的一项基础任务，为从自主导航和机器人到增强现实的应用提供关键的信心水平。它也是其他复杂任务的基础步骤，包括单目3D目标检测和单幅图像的点云重建。历史上，该领域的进展主要依赖于全监督方法，这些方法虽然精确，但依赖于大规模、密集深度标注的数据集（例如，NYU Depth V2），而这些数据集的创建被认为耗时且成本高昂。为了减轻这种数据依赖性，出现了一种新范式：大规模的基础模型。这些模型通过在多样化、聚合的数据集上训练，实现了最先进的性能，但引入了大量的计算和参数负担，限制了它们的实际部署。这留下了一个关键的研究缺口，即寻求数据和计算效率兼具的方法。DepthCLIP首次探索了一种替代方向，利用VLM对齐技术进行零样本深度估计。通过将深度估计重新构建为一种语言驱动的分类问题，将图像区域与“近”或“远”等文本提示匹配，DepthCLIP在没有任何特定任务训练的情况下提供了概念验证。然而，虽然这一方法新颖，但其对手工提示和粗略深度离散的依赖限制了其实际效用，输出结果缺乏几何细节。随后的努力旨在克服这些限制，专注于学习更有效的提示、创建自适应、图像感知的深度区间以及开发高效的少样本适应机制。受到这些挑战的激励，我们提出了MoA-DepthCLIP，一个将CLIP的语义力量与深度估计所需几何精度相结合的框架，同时保持卓越的效率。我们的核心贡献是对两个强大但以往分离的范式的独特整合。首先，尽管MoA在其他领域（如自然语言处理）中证明了其在参数高效微调方面的有效性，但其在单目深度估计等密集几何任务中的应用尚未探讨。其次，混合分类-回归架构在传统深度估计中是已知的标准，但尚未与现代、轻量级的VLM适应策略（如MoA）结合。我们的方法整合了这两个概念：我们在ViT主干网络中引入了轻量级的MoA模块和选择性微调，并将得到的适应特征输入到这一经验证的混合头架构中。这一新颖的组合实现了对深度任务的参数高效、空间感知适应。我们的模型由一个全球场景上下文向量引导，该向量源自平均文本提示嵌入，并通过组合损失函数加以约束，该函数结合了分类和回归损失（L1和尺度不变项）。我们的贡献总结如下：

![](images/1.jpg)  
Fig. 1: Overall architecture of MoA-DepthCLIP. Scene prompts are encoded using a frozen CLIP text encoder to form a global scene context vector. The image is encoded with a frozen ViT-B/32 backbone augmented with MoAs. Fused features are then passed to a dual-head prediction module: one head performs depth bin classification and produces a binned depth map via weighted summation, while the other head performs direct regression. The final output depth map is a fusion of both predictions.

我们引入了MoA-DepthCLIP，这是基于轻量级MoA PEFT方法的单目深度估计的首个适应性策略，并结合了选择性主干微调。我们展示了如何将这一现代的、原生于VLM的适应性策略与经典的、以几何为中心的混合（分类-回归）预测头相结合，以恢复细粒度的度量细节。通过在NYU Depth V2上的实验，我们证明了我们合成的方法显著优于之前基于VLM的方法，如DepthCLIP，$\delta _ { 1 }$ 精度从0.390提高到0.745，RMSE减少超过$55\%$，同时只使用了通常所需基础模型可训练参数的一小部分。

# 2 相关工作

# 2.1 视觉-语言模型

大规模视觉-语言模型的发展，尤其是CLIP，标志着一个重要的里程碑，通过从图像-文本对中学习强大的、可迁移的表示。CLIP在零-shot 分类中的成功激发了对密集预测任务的适应。例如，DenseCLIP展示了如何通过精心设计提示有效地利用CLIP的特征进行语义分割。尽管这些模型对语义概念有较强的掌握，但将这一知识应用于深度估计等细粒度几何任务仍然是一个关键挑战。

# 2.2 单目深度估计

从单幅图像进行深度估计是计算机视觉中的一个长期任务。监督方法[6,11]始终在准确性上扩展边界，但它们对大规模标注数据集（如NYU Depth V2 [18]）的依赖是一个重大瓶颈。同时，自监督方法试图通过利用视频序列中的几何约束作为监督信号来缓解这一问题[14,23]。近年来，趋势已转向“基础模型”，它们通过在海量多样化的数据集上进行训练，实现了最先进的泛化能力，尽管这种性能伴随着高昂的计算成本[2,12,15,21]。

# 2.3 基于视觉-语言的深度估计

直接利用视觉-语言模型进行深度估计是一项相对较新的方向。这方面的第一个重要工作是 DepthCLIP [26]，它通过将深度预测重新表述为分类问题，引入了一种零样本方法。它将图像区域与手工设计的文本提示（如“近”或“远”）匹配，从而生成粗略的深度图，不需要任何微调。虽然这是一种新颖的概念验证，其实际应用受到输出粗糙性和对手动设计提示依赖的限制。后续的努力旨在克服这些局限，集中在学习更有效的提示 [1,10]、创建自适应的图像感知深度区间 [19] 和开发高效的少样本适应机制 [9] 等策略上。我们的工作也建立在这个想法上，但在简单的零样本提示之上显著推动了这一范式。我们引入了 MoA-DepthCLIP，一个用 MoA 替代静态提示工程的可学习、轻量级适应策略的框架。此外，我们通过整合全局场景上下文和细粒度的混合预测头，解决了先前工作中的几何局限。这种整体方法有效地弥合了 CLIP 的高层语义理解与单目深度估计的精确度量要求之间的差距。

# 3 方法

# 3.1 适配器混合

MoA-DepthCLIP方法引入了一种参数高效的策略，通过将混合适配器（Mixture-of-Adapters，MoA）模块集成到预训练的CLIP [16] 视觉主干网络中，该网络基于ViT-B/32架构 [4]。我们完整架构的概述，展示了这些模块的放置，见图1。每个MoA模块由三个主要组件组成：一组轻量级专家、一个确定与token相关的路由权重的门控网络以及一个将适配后的特征注入回主干网络的残差混合操作。专家架构。每个专家作为一个具有瓶颈结构的两层多层感知器（MLP）实现：

$$
\mathrm { E x p e r t } ( x ) = W _ { 2 } \sigma ( W _ { 1 } x ) ,
$$

![](images/2.jpg)

(a) 选择性MoA放置。我们在ViT-B/32编码器（棕色）的关键层（2、5、8、11）中插入MoA模块（绿色）。

![](images/3.jpg)  
(b) Internal architecture of a single MoA module, showing the Gating Network, $\mathrm { K } { = } 4$ Experts, and residual injection.

图 2：我们的轻量级适配策略概述。 (a) 显示了 MoA 模块在 ViT 主干网络中的选择性放置。 (b) 详细介绍了单个 MoA 模块的内部结构，其中 $W _ { 1 } \in \mathbb { R } ^ { d \times d _ { b } }$ , $W _ { 2 } \in \mathbb { R } ^ { d _ { b } \times d }$ , $d$ 是变压器的隐藏维度（ViT-B/32 为 768），$d _ { b } = 6 4$ 是适配器瓶颈大小，而 $\sigma$ 是 GELU 激活函数。该设计遵循标准适配器公式，但故意保持小巧以最小化参数开销。门控网络。为了实现针对特定令牌的专业化，我们采用了门控网络，其原理类似于 Mixture-of-Experts PEFT 方法中的 AdaMix [20]。对于每个令牌表示 $x _ { i }$ ，网络使用标准 softmax 函数预测 $K$ 个专家的路由概率：

$$
g _ { i } = \mathrm { s o f t m a x } \left( \frac { G ( x _ { i } ) } { \tau } \right) ,
$$

其中 $G ( \cdot )$ 是一个两层的多层感知机 ( $7 6 8 \to 1 2 8 \to K$ )，采用 ReLU 激活函数，$\tau = 2.0$ 是一个温度超参数。我们与 AdaMix [20] 的关键区别在于，后者在训练过程中使用随机路由并在推理时进行专家融合，而我们直接且确定性地使用这些概率。我们在训练和推理过程中直接使用计算得到的 $g _ { i }$ 值，以形成专家输出的加权组合（公式 3）。这构成了我们在该组件中的贡献：在 MoA 框架内应用更简单、确定性的门控机制，以实现深度估计中的稳定适应。混合与残差注入。在我们的确定性门控机制下，对于给定的令牌 $x _ { i }$，所有 $K$ 个专家的输出通过使用公式 2 中的门概率 $g _ { i , k }$ 进行加权求和。重要的是，这个混合结果然后通过残差连接添加回原始令牌表示中，这是一种在适配器模块 [8] 中的标准技术，用以确保训练稳定性并保留在预训练过程中学习到的有价值特征：

$$
\tilde { x } _ { i } = x _ { i } + \sum _ { k = 1 } ^ { K } g _ { i , k } \mathrm { E x p e r t } _ { k } ( x _ { i } ) .
$$

我们在 MoA-DepthCLIP 中利用这种残差设计，无缝集成专业化的专家适配，同时严格保持主干网络的原始表征能力，这一平衡对于诸如深度估计等细粒度任务至关重要。此外，在训练过程中，我们监控来自 $g _ { i , k }$ 的专家使用情况，以确保有效的负载平衡，使我们的方法与缺乏这种动态专业化的简单适配方法有所区别。选择性层集成。我们工作的一个关键设计选择是 MoA 模块的选择性放置。我们并未在每个变压器块后均匀插入模块，而是采用针对性的策略。我们实证确定仅在 ViT-B/32 编码器的四个关键变压器块中插入 MoA 模块，具体为层 $\{ 2 , 5 , 8 , 1 1 \}$，有效地平衡了早期、中层和晚期特征之间的适配。这种稀疏放置策略构成了我们方法的一个特定贡献，优化了性能的同时保持了参数数量的最小化。专家多样性。在 MoE 和 MoA 系统中，一个潜在的挑战是专家崩溃，即门控网络主要将标记路由到仅少数几个专家，从而抵消了专业化的好处 [20]。虽然一些方法在训练过程中引入辅助损失项以强制进行负载平衡，但我们采用了一种更简单的关注监控的方法。我们在训练期间跟踪门控分布的熵（公式 4）：

$$
H ( g _ { i } ) = - \sum _ { k = 1 } ^ { K } g _ { i , k } \log g _ { i , k } .
$$

较高的平均熵表明多个专家正在被积极利用。我们的贡献在于仅将熵作为诊断工具来验证有效的专家使用和训练稳定性，而不是引入复杂的负载均衡损失。这一监控确认我们的 MoA 实现达到了期望的专业化而没有崩溃。总体而言，我们的适应策略结合了轻量级 MoA 模块和最后四层主干网络的选择性微调，实现了空间感知的词元级专业化。虽然 MoA 模块本身增加的参数数量可以忽略不计，但我们模型中的可训练参数总数仍占主干网络完整参数数量的小部分，使我们的方案成为全量微调的高效替代方案。

# 3.2 全球场景上下文融合

原始的 DepthCLIP [26] 框架依赖于在像素级别应用的简单手工提示（例如，“近”，“远”）来表示粗略的深度类别。虽然这一方法在零-shot 预测中具有创新性，但它缺乏对更广泛场景上下文的敏感性。为了提供相比学习连续提示的方法 [28] 更强但更简单的语义先验，我们引入了一种机制，将固定的全球场景上下文向量与视觉特征融合。MoA-DepthCLIP 以参数无关的方式利用 CLIP 文本编码器 [16]。为了提供与我们的目标基准（NYU Depth V2 [18]）相关的语义先验，我们定义了一组与其常见室内场景类别对齐的固定文本提示（例如，“厨房的照片”，“教室的照片”等）。这些提示使用冻结的 CLIP 文本编码器进行编码，并且它们的嵌入经过 $\ell_{2}$ 归一化。我们通过逐元素平均这些提示嵌入计算出一个单一的通用上下文向量 c，代表“室内场景”的一般概念。这一策略在训练和推理过程中建立了一个恒定的语义锚点，而不引入任何可学习参数，从而确保为视觉特征提供一致的指导。然后，这个全球上下文向量 c 与 MoA 适配主干网络的视觉特征图在空间上进行融合。我们对 $\mathbf{c}$ 进行广播，使其匹配视觉特征的空间维度，并沿着通道维度进行拼接。这种全球融合与 DepthCLIP 的像素级提示匹配和提示学习中的可适应向量显著不同，而是提供了一个在整个图像中均匀的高层次先验，以补充局部视觉细节。最终得到的融合表示作为后续混合预测模块的输入。

# 3.3 深度分 bin 策略

我们采用了一种混合预测头，涉及深度区间分类，基于深度离散化技术的有效性[6]。具体而言，我们对每个像素预测$N$个离散深度区间的分布，遵循[26]中使用的通用公式。然而，我们在定义这些区间的方式上与DepthCLIP [26]有显著不同。DepthCLIP采用了少量（10个）基于语义距离的固定手工设计区间。认识到区间粒度至关重要，但我们目标是采用比像AdaBins [3]等预测每图像自适应区间的方法更简单的方法，因此我们系统性地探索了固定区间的不同数量。通过消融实验（在第5节中详细说明），我们发现使用$N = 1 2 8$的区间在我们的MoA改编CLIP框架中特别提供了准确性和鲁棒性之间的最佳权衡。我们实证证明，采用固定的区间数量$N \ = \ 1 2 8$在不需要自适应分 bin 的计算复杂度情况下，相较于DepthCLIP的粗离散化提供了显著的改善。这种配置提供了在VLM适应背景下精细准确性和模型鲁棒性之间的最佳权衡。因此，我们在所有实验中采用$N = 1 2 8$的区间。

# 3.4 组合损失函数

为了训练我们具有并行分类和回归任务头的混合架构，我们采用复合损失函数。这种将适用于分类（用于稳定性）和回归（用于细节）的损失结合起来的策略是在现代深度估计框架中一种成熟的做法 [6,3]。我们的总损失是三个标准组成部分的加权和：

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } + \lambda _ { \mathrm { s i l o g } } \mathcal { L } _ { \mathrm { s i l o g } } ,
$$

每个术语针对预测的特定方面。我们调整这种加权配置，以稳定我们轻量级 MoA 专家的训练，确保它们捕获高层次的结构布局和低层次的度量细节。分类损失 $ ( \mathcal { L } _ { c l s } ) $：为了监督分类头，我们使用标准的逐像素交叉熵（CE）损失。该损失鼓励模型为每个像素预测正确的离散深度区间，通过将输出的 logits 与真实的区间索引进行比较。它提供了一个强而稳定的监督信号，对于学习场景的粗略布局特别有效。回归损失。回归头预测连续的深度图，受两个互补损失项的监督。L1 损失 $ \mathcal { L } _ { \mathbf { r e g } } $：我们采用逐像素的 L1 损失，惩罚预测深度 $ \hat { d } _ { i } $ 与真实深度 $ d _ { i } $ 之间的绝对差异。这为提高局部几何准确性提供了直接而细致的信号。尺度不变对数损失 $ \underline { { \mathcal { L } } } _ { \mathbf { s i l o g } } $：为了确保对单目深度估计中固有的全局尺度和位移模糊性的鲁棒性，我们使用受 [5] 中公式启发的尺度不变对数（SILog）损失。对于有效遮罩中的每个像素 $_ i $，损失基于对数差值的方差和均值定义，$ g _ { i } = \log \hat { d } _ { i } - \log d _ { i } $：

$$
\mathcal { L } _ { \mathrm { s i l o g } } = \alpha \left( \mathrm { V a r } ( g ) + \lambda \mathrm { M e a n } ( g ) ^ { 2 } \right) ,
$$

其中 $\operatorname { V a r } ( g )$ 和 $\operatorname { M e a n } ( g )$ 在所有有效像素上计算，$\lambda = 0 . 8 5$ 是加权项，$\alpha = 1 0 . 0$ 是缩放因子。损失加权。各组件使用固定权重组合以平衡它们的贡献。基于我们的实验设置，我们将权重设置为 $\lambda _ { \mathrm { c l s } } =$ 1.0，$\lambda _ { \mathrm { r e g } } = 1 . 0$，以及 $\lambda _ { \mathrm { s i l o g } } ~ = ~ 0 . 5$。这一加权方案提供了一个平衡的目标，优先考虑分类头的粗略类别准确性和回归头的精细几何保真度。

# 4 实验

# 4.1 数据集与评估指标

我们在 NYU Depth V2 数据集 [18] 上评估我们的框架，该数据集是一个广泛使用的室内 RGB-D 基准，由 Kinect 传感器捕获。深度值遵循标准协议限制在 10 米内。

Table 1: Quantitative results on NYU Depth V2. We evaluate different configurations by progressively introducing composite loss, MoA, and increasing the number of bins. The final configuration, MoA-DepthCLIP (with ViT-B/32, MoA, composite loss, and 128 bins), achieves the best performance across all metrics.   

<table><tr><td>Method</td><td>Backbone</td><td>Comp. Loss</td><td>MoA</td><td>#Bins</td><td>δ1 ↑$</td><td>δ2 ↑$</td><td>δ3 ↑</td><td>AbsRel ↓</td><td>log10 ↓</td><td>RMSE ↓</td></tr><tr><td>DEPTHCLIP</td><td>ResNet-50</td><td></td><td></td><td>10</td><td>0.390</td><td>0.680</td><td>0.848</td><td>0.393</td><td>0.158</td><td>1.176</td></tr><tr><td>Our Baseline</td><td>ViT-B/32</td><td></td><td></td><td>10</td><td>0.417</td><td>0.701</td><td>0.890</td><td>0.377</td><td>0.147</td><td>1.096</td></tr><tr><td>+ COmposItE Loss</td><td>ViT-B/32</td><td>✓</td><td></td><td>10</td><td>0.503</td><td>0.791</td><td>0.925</td><td>0.310</td><td>0.121</td><td>0.843</td></tr><tr><td>+ MoA</td><td>ViT-B/32</td><td>✓</td><td>✓</td><td>10</td><td>0.508</td><td>0.797</td><td>0.931</td><td>0.308</td><td>0.120</td><td>0.821</td></tr><tr><td>MoA-DePTHCLIP</td><td>ViT-B/32</td><td>✓</td><td>✓</td><td>128</td><td>0.745</td><td>0.841</td><td>0.895</td><td>0.321</td><td>0.098</td><td>0.520</td></tr></table>

为了评估，我们报告了常用的指标，包括绝对相对误差（AbsRel）、均方根误差（RMSE）、log10误差以及阈值准确率 $\delta _ { 1 }$、$\delta _ { 2 }$、$\delta _ { 3 }$。在这里，$\delta _ { i }$ 衡量的是预测值在真实值的 $1.25^{i}$ 倍范围内的像素比例。综合这些指标可以捕捉全局准确性和尺度鲁棒性。

# 4.2 实现细节

MoA-DepthCLIP 在 PyTorch 中实现，使用来自 OpenCLIP 的预训练 ViT-B/32 作为视觉主干网络。为了平衡特征保留和任务适应，我们保持大部分主干网络冻结，但对最后 4 个 Transformer 块进行微调。我们模型的主要可训练组件是这些未冻结的块、MoA 模块以及两个预测头（分类和回归）。源自文本提示的全局上下文向量在训练过程中保持固定。我们为每个 MoA 模块使用 $K = 4$ 个专家，瓶颈维度设为 $d _ { b } = 6 4$。模型在单个 NVIDIA H100 GPU 上训练 30 轮，批量大小为 8。我们使用 AdamW 优化器，学习率保持为 $1 \times 1 0 ^ { - 5 }$，权重衰减为 $1 \times 1 0 ^ { - 4 }$。分类头的深度区间数量设置为 $N = 1 2 8$，我们发现该设置在消融研究中表现最佳（见第 5 节）。

# 4.3 量化结果

表1总结了我们消融研究的结果，展示了框架中每个组件的逐步影响，起始于重现的Depth-CLIP基线。我们的分析揭示了一条明显的改进轨迹。首先，仅仅将原始的ResNet-50主干网络替换为ViT-B/32便带来了显著的性能提升。引入我们的复合损失函数时，性能跃升最为显著，这大幅提高了准确性（例如，将$\delta _{1}$从0.417提升至0.503）。随后，集成MoA模块进一步实现了增量收益。最后，将深度区间的数量优化至128，带来了另一项显著提升，特别是在阈值准确性和均方根误差（RMSE）方面。MoA-DepthCLIP显著优于重现的DepthCLIP基线，将$\delta _{1}$的准确性从0.390提升至0.745，RMSE从1.176降低至0.520。

在分析最后一步时，一项显著的权衡变得明显。虽然将箱数增加到128在更严格的准确度阈值（$\delta _ { 1 }$ 和 $\delta _ { 2 }$）上带来了显著的提升，但我们观察到在最宽松的阈值（$\mathbf { \delta _ { \delta } } _ { \mathbf { \delta _ { 3 } } }$）上却有轻微下降。我们将这种现象归因于最终模型的专业化程度增加。通过在一个更加细粒度的预测空间中操作，模型能够进行高度精确的预测，从而成功纠正大量像素。然而，这种精确性可能导致一小部分模糊的预测从“粗略但安全”的估计转变为更专业化的预测，这些预测刚好落在$\delta _ { 3 }$的宽容范围之外。最终，我们将其解读为一个强有力的积极指标：这表明MoA-DepthCLIP成功地学习到进行细粒度、高精度预测的能力，而这一能力在其在更具挑战性的$\delta _ { 1 }$和$\delta _ { 2 }$指标上的优异表现中得到了强烈反映。此系统分析验证了每个组件的贡献，表明强大的主干网络、复合损失、专业化适配器和优化的分箱策略的组合对于实现高性能至关重要。

# 5 消融研究

我们进行消融研究，以检验MoA-DepthCLIP中两个关键超参数的影响：（1）在混合适配器（Mixture-of-Adapters, MoA）层中使用的专家数量，以及（2）用于离散化的深度区间数量。这个两阶段的程序提供了清晰的、分解的见解，并推动了我们在主要实验中最终选择的超参数。

# 5.1 专家数量的影响

在第一阶段，我们将深度区间的数量固定为粗略值10，并改变每个MoA层中的专家数量。如表2所示，性能在不同专家数量下保持相对稳定，表明我们的框架对这一超参数具有鲁棒性。尽管单专家基线（$K = 1$）在AbsRel误差方面表现竞争力，但缺乏实现MoA设计中固有的语义专业化所需的多专家能力。相反，尽管将$K$增加到8可以获得最佳准确性（$\delta_{1} = 0.665$），但它将计算开销增加了一倍，所带来的增益却微乎其微。因此，我们选择$K = 4$作为一个平衡配置，提供足够的专家专业化能力，而不出现在更高数量时观察到的收益递减。

# 5.2 深度区间数量的影响

在将专家数量固定为4之后，我们调整深度区间的数量，以探讨离散化粒度的影响。表3显示了一个明显的趋势。

Table 2: Ablation on the number of experts, with 10 bins. $K = 4$ is chosen as a balanced trade-off.   

<table><tr><td rowspan=1 colspan=1>#Experts</td><td rowspan=1 colspan=1>#Experts AbsRel ↓ RMSE↓δ1 ↑</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.394    0.560  0.660</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.396     0.559  0.661</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.396     0.561  0.661</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>0.395    0.557 0.665</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>0.395     0.558  0.661</td></tr></table>

随着箱数增加到 128，性能提升，达到了所有关键指标的最佳结果。进一步将箱数增加到 180 或 200，会导致性能略微下降。我们推测这是由于每个箱的数据稀疏，使得模型在有限数据下更难学习到稳定的分布。此分析确认 $\mathbf { N = 1 2 8 }$ 是 MoA-DepthCLIP 的最佳选择。

Table 3: Ablation on the number of depth bins, with 4 experts. $N = 1 2 8$ provides the best performance.   

<table><tr><td rowspan=1 colspan=1>#Bins</td><td rowspan=1 colspan=1>AbsRel ↓ RMSEδ1 ↑</td></tr><tr><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>0.327     0.522  0.737</td></tr><tr><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>0.329     0.534  0.730</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0.321     0.521  0.745</td></tr><tr><td rowspan=1 colspan=1>180</td><td rowspan=1 colspan=1>0.323     0.532   0.735</td></tr><tr><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>0.329     0.541   0.717</td></tr></table>

# 6 结论

在本论文中，我们提出了MoA-DepthCLIP，这是一种参数高效的框架，用于适应预训练的CLIP表示以进行单目深度估计。我们的方法通过结合两种有效策略来避免昂贵的全主干微调：插入轻量级的MoA模块和选择性地微调ViT主干网络的最后几层。我们方法的核心是一个双头预测架构，可以同时进行深度分类和回归，处理与全局场景上下文向量融合的视觉特征。我们综合的实验和消融研究验证了我们的设计选择，确定了一个包含4个专家和128个深度区间的配置为最佳配置。我们框架的一个关键要素是复合损失函数，该函数通过同时监督一个具有交叉熵损失的分类头和一个具有L1和SILog损失的回归头，来训练这个双头系统。结果确认了我们方法的有效性。MoA-DepthCLIP显著优于DepthCLIP基线，并与更大基础模型的竞争性能相当，尽管仅使用了它们可训练参数的一小部分。这项工作展示了针对性的轻量级适应策略在弥合VLM的语义丰富性与密集预测任务所需的几何精度之间的巨大潜力。未来的研究方向包括将我们的框架扩展到多样的户外数据集。此外，结合动态组件，如基于注意力的提示选择，可能进一步提升模型的性能。

# References

1. Auty, D., Mikolajczyk, K.: Learning to prompt clip for monocular depth estimation: Exploring the limits of human language. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops. pp. 20392047 (2023)   
2. Bhat, S.F., Birkl, R., Wofk, D., Wonka, P., Müller, M.: Zoedepth: Zero-shot transfer by combining relative and metric depth. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023)   
3. Bhat, S.F., Wofk, D., Birkl, R., Müller, M., Wonka, P.: AdaBins: Adaptive discretization for monocular depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14494 14503 (2021)   
4. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (ICLR) (2021)   
5. Eigen, D., Puhrsch, C., Fergus, R.: Depth map prediction from a single image using a multi-scale deep network. In: Advances in Neural Information Processing Systems (2014)   
6. Fu, H., Gong, M., Wang, C., Batmanghelich, K., Tao, D.: Deep ordinal regression network for monocular depth estimation. In: Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR). pp. 20022011 (2018)   
7. Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: Clip-adapter: Better vision-language models with feature adapters. International Journal of Computer Vision (IJCV) 132, 581595 (2024)   
8. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-efficient transfer learning for NLP. In: International Conference on Machine Learning (ICML). pp. 27902799. PMLR (2019)   
9. Hu, X., Zhang, C., Zhang, Y., Hai, B., Yu, K., He, Z.: Learning to adapt clip for few-shot monocular depth estimation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 55945603 (2024)   
0. Kim, S., Kang, J., Kim, D., Lee, S.: Clip can understand depth. arXiv preprint arXiv:2402.03251 (2024)   
1. Li, Z., Chen, Z., Liu, X., Jiang, J.: Depthformer: Exploiting long-range correlation and local information for accurate monocular depth estimation. arXiv preprint arXiv:2203.14211 (2022)   
12. Li, Z., Liu, X., Jiang, J.: Metric3d: 'Towards zero-shot metric depth estimation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2023)   
13. Liu, Y., Liu, Z., Lan, X., Yang, W., Li, Y., Liao, Q.: Dm-adapter: Domain-aware mixture-of-adapters for text-based person retrieval. In: Proceedings of the AAAI Conference on Artificial Intelligence (2025)   
14. Mahjourian, R., Wicke, M., Angelova, A.: Unsupervised learning of depth and ego-motion from monocular video using 3d geometric constraints. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2018)   
15. Piccinelli, L., Yang, Y.H., Sakaridis, C., Segu, M., Li, S., Van Gool, L., Yu, F.: Unidepth: Universal monocular metric depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)   
16. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Proceedings of the International Conference on Machine Learning (ICML). PMLR (2021)   
17. Rao, Y., Zhao, W., Chen, G., Tang, Y., Zhu, Z., Huang, G., Zhou, J., Lu, J.: Denseclip: Extract free dense labels from clip. arXiv preprint arXiv:2112.01071 (2021)   
18. Silberman, N., Hoiem, D., Kohli, P., Fergus, R.: Indoor segmentation and support inference from rgbd images. In: European Conference on Computer Vision (ECCV). Springer (2012)   
19. Son, E., Lee, S.J.: Cabins: Clip-based adaptive bins for monocular depth estimation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 45574567 (2024)   
20. Wang, Y., Agarwal, S., Mukherjee, S., Liu, X., Gao, J., Awadallah, A.H., Gao, J.: Adamix: Mixture-of-adaptations for parameter-efficient model tuning. In: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 52955309. Association for Computational Linguistics (2022)   
2Yang, L., Kang, B. Huang, Z., Xu, X., Feng, J. Zhao, H. Depth anything: Unleashing the power of large-scale unlabeled data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)   
. Zeng, W., Karaoglu, S., Gevers, T.: Inferring point clouds from single monocular images by depth intermediation. In: Proceedings of the European Conference on Computer Vision (ECCV) (2018)   
23. Zhang, M., Ye, X., Fan, X., Zhong, W.: Unsupervised depth estimation from monocular videos with hybrid geometric-refined loss and contextual attention. Neurocomputing 379 (2020)   
Zhag, R. Guo, Z. Zag W., Li, K. Miao, X. Cui B. Qiao, Y. Gao,P. Li, H: Pointclip: Point cloud understanding by clip. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022)   
ZaR., Qiu H. Wa T. Guo, Z. Tag, Y. Xu, X. Cui, Z., Qiao, Y. Li, H., Gao, P.: Monodetr: Depth-guided transformer for monocular 3d object detection. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2023)   
26. Zhang, R., Zeng, Z., Guo, Z., Li, Y.: Can language understand depth? In: Proceedings of the 30th ACM International Conference on Multimedia. Association for Computing Machinery (2022)   
27. Zhang, R., Zhang, W., Fang, R., Gao, P., Li, K., Dai, J., Qiao, Y., Li, H.: Tipadapter: Training-free adaption of clip for few-shot classification. In: Proceedings of the European Conference on Computer Vision (ECCV) (2022)   
28. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. arXiv preprint arXiv:2109.01134 (2021)   
29. Zhou, X., Girdhar, R., Joulin, A., Krahenbuhl, P., Misra, I.: Detecting twentythousand classes using image-level supervision. In: Proceedings of the European Conference on Computer Vision (ECCV). pp. 350368 (2022)