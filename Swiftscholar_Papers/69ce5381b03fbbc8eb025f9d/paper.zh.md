# 筛查足够

中西健 M. 理化学研究所（RIKEN）新兴物质科学中心（CEMS） 东京大学理学院

# 摘要

标准软最大注意力的一个核心限制是它未能定义绝对的查询-键相关性：注意力权重是通过根据相对分数在所有键之间重新分配固定的单位质量来获得的。因此，相关性仅相对于竞争键定义，无法明确拒绝无关键。我们引入了Multiscreen，这是一种建立在我们称之为筛选机制的语言模型架构上，能够实现绝对的查询-键相关性。筛选不再在所有键之间重新分配注意力，而是对每个键进行明确阈值评估，丢弃无关键并聚合剩余键，从而消除了键之间的全局竞争。在各项实验中，Multiscreen在参数量减少约$40\%$的情况下，验证损失与Transformer基线相当，能够在显著更大的学习率下实现稳定优化，在长上下文困惑度中保持强劲表现，并且即使在超出训练上下文长度的情况下检索性能几乎没有下降，同时在100K上下文长度下将推理延迟减少最多$3.2 \times$。

# 1 引言

处理远超训练期间所见上下文的情况仍然是大语言模型（LLMs）的一个核心挑战。较长的上下文对于有效利用上下文信息至关重要，但实际上，使用长序列进行训练在计算上非常昂贵，因为 Transformer 自注意力的成本随着序列长度呈平方增长。因此，模型通常是在相对较短的上下文上进行训练，然后在推理时期待能够推广到更长的上下文。然而，仅仅增加名义上下文长度并不能保证模型能够有效利用其中的相关信息。

先前的研究表明，长范围依赖建模仍然困难，模型往往无法可靠地利用上下文中远处的信息。这一挑战不仅仅是上下文长度的问题，还涉及如何从上下文中聚合相关信息。在标准的softmax注意力中，所有未屏蔽的键被联合评估，注意力权重是通过根据每个查询-键分数与其他分数的比较结果重新分配固定的单位质量而获得的。因此，无论是注意力分数还是注意力权重都未定义绝对相关性的概念：一个键的注意力权重较大并不是因为它的分数超过了固定的阈值，而是因为它的分数相对于竞争键的分数足够大。无关的键也无法在没有显式屏蔽的情况下被干净地拒绝：即使没有键真正相关，仍然必须在可用键之间分配一些注意力权重。此外，由于所有未屏蔽的键都接收非零权重，因此增加上下文长度必然会稀释对多个键的注意力，使得随着上下文的增长，保留相关词元的强贡献变得越来越困难。这种行为源于注意力生成未界定分数的归一化权重，而没有应用绝对阈值到查询-键分数的机制。在本文中，我们提出了Multiscreen，这是一种受Transformer启发的语言模型架构，但围绕我们称之为筛选的机制构建，能够实现绝对查询-键相关性。筛选不是在所有键之间重新分配固定的单位质量，而是计算界限有限的查询-键相似度，并根据显式阈值独立评估每个键以计算相关性，丢弃无关的键并聚合剩余的键。这种形式消除了键之间的全局竞争。因此，该模型可以表示相关上下文的缺失，更有效地利用长范围信息。Multiscreen进一步学习筛选窗口，以确定有效的上下文范围，使得每个筛选单元能够调整其上下文范围，避免不必要的长范围计算。同时，它还采用了最小的位置信息编码，仅在筛选窗口足够小的情况下启用，否则保持非活动状态。因此，长范围行为不依赖于超出训练期间看到的位置信息模式的推断，避免了通常由于位置外推而产生的不匹配。评估语言模型本质上具有挑战性，因为它们展现一系列能力，包括下一个词预测、检索和遵循指令，这些能力无法通过单一指标完全捕捉。标准的下一个词预测指标如验证损失未必反映模型提取和使用相关信息的能力。此外，某些基准设计并未孤立提取行为，因为性能可能由语义线索驱动，或受到语义屏蔽的影响，或受特定提示效果的影响，从而使得难以区分潜在的提取能力与依赖提示的行为。为了解决这些局限性，除了长上下文困惑度外，我们引入了ABCDigits，这是一个基于合成完成的键值检索基准，去除了自然语言语义，固定了不同上下文长度下的键数，并确保目标输出是唯一确定的，而无需依赖遵循指令或语义线索。我们总结了主要的实证发现如下。在扩展实验中，Multiscreen在相同的词元预算下以约40%的参数数量实现了与Transformer基线相当的验证损失。它还在允许比Transformer基线大得多的学习率下实现了稳定优化。在长上下文评价中，Multiscreen在困惑度上保持强性能，并且在检索性能上几乎没有退化，即使在远超训练期间看到的上下文长度时。在设计用于孤立提取行为的基准ABCDigits上，Multiscreen在检索性能上几乎没有退化，即使在远超训练期间看到的上下文长度时，并且持续优于Transformer基线，即使在训练上下文长度内，尽管其验证损失大幅较高。最后，针对100K词元上下文的下一个词预测，Multiscreen相较于Transformer基线减少了2.3-3.2倍的推理延迟。这些结果表明，以筛选为基础的架构能够同时提高参数效率、在大学习率下的训练稳定性、检索能力和推理延迟。它们进一步暗示，提高长上下文行为需要超越基于重新分配的机制，转向基于绝对相关性标准的显式选择相关信息。我们的主要贡献如下： • 我们引入了Multiscreen，一种通过我们称之为筛选的机制实现绝对查询-键相关性的语言模型架构。 • 我们展示了Multiscreen在参数效率、大学习率下的训练稳定性、检索能力和推理延迟等方面相较于Transformer基线的改善。 • 我们引入了ABCDigits，一个不依赖遵循指令或语义线索的语义自由完成型检索基准，以孤立提取行为。

# 2 相关工作

基于Softmax的注意力及其变体。大量研究探讨了Softmax注意力的修改，主要因其在长上下文环境中的局限性以及效率考量。近期的研究显示，随着上下文长度的增长，Softmax归一化所引发的问题变得明显。可扩展Softmax（SSMax）通过在上下文长度增加时锐化注意力分布来应对注意力衰退现象。选择性注意力在Softmax框架内引入了查询和位置依赖的温度缩放。其他方法如sparsemax、entmax及其变体则修改归一化注意力分布的形状，以鼓励稀疏性。更近期的研究探索了稀疏或基于检索的注意力机制，这些机制限制了注意的键集以提高效率，同时仍对所选子集应用归一化注意力。尽管这些方法存在差异，但它们仍然在重新分配竞争键的注意力框架内。我们的研究在更根本的层面上有所不同：Multiscreen围绕筛选构建，使得通过独立评估每个键的查询-键相似度实现绝对的查询-键相关性，并在键之间无竞争地聚合信息。 超越注意力的序列建模。另一个研究方向探讨了用于序列建模的替代交互结构，旨在实现次二次或线性时间的计算。架构如Mamba、Hyena和RetNet采用了替代交互机制，包括选择性状态空间、长卷积或递归保持机制，以建模长距离依赖关系。混合方法进一步将高效的非注意力主干与局部或选择性注意力组件相结合。尽管这些方法展示了无需明确的全词元间交互就能实现长序列建模，先前的实证研究表明这样的模型在召回导向的行为和上下文检索上可能落后。相比之下，Multiscreen保留了完整的词元到词元的连接性，同时通过学习的筛选窗口减少不必要的计算。特别地，具有有限筛选窗口的块可以在有效的线性时间内操作，同时允许模型学习在哪些情况下需要全面连接。这使得Multiscreen能够将强大的检索能力与改善的计算效率相结合。 长上下文建模和位置表示。通过架构变更和位置表示的修改，已经广泛研究了扩展语言模型可用上下文长度的方法，特别是为了应对超出训练上下文的长度泛化。代表性的方法包括基于ALiBi或RoPE的外推方法，以及LongRoPE等明确调整长上下文位置行为的方法。相关研究还探讨了用于长度泛化的学习型或基于函数的相对位置方案，例如FIRE。其他研究则探讨了完全去除显式位置编码的方法，包括NoPE及其长度泛化行为的后续分析。我们的位置信息设计与这一系列研究密切相关，但在一个关键方面存在差异。Multiscreen采用了最小的位置信息编码，这一编码仅在学习到的筛选窗口足够小时启用，其他时候处于非活动状态，有效上下文范围由学习的筛选窗口决定。因此，长距离行为并不依赖于超出训练范围的位置信息模式外推。据我们所知，之前的研究尚未探讨将学习到的筛选窗口与这种有条件激活的位置信息机制相结合的方式。 检索评估和合成基准。人们越来越认识到，仅依靠下一个词元预测不足以捕捉语言模型行为的所有方面。特别是在自然和合成环境中，检索的失败现象得到观察，包括迷失于中间的现象以及高效语言模型的检索导向分析。同时，评估检索本身并非易事：基准行为可能受到语义线索的影响、被语义掩蔽所遮蔽或受特定提示的影响，使得难以孤立出纯粹的检索能力。合成的联想回忆和键值检索任务长期以来一直用于研究序列模型中的记忆。最近的基准包括MQAR和相关的合成回忆任务，以及针对长上下文检索的针在大堆草中的评估和通行证式的评估。我们的ABCDigits基准在精神上最接近于这一检索导向的研究方向，但不同之处在于采用无语义的基于完成的形式，以固定的键集跨上下文使用。该设计消除了自然语言语义，固定了不同上下文长度下的键数，并确保目标输出独特确定，而无需依赖于指令遵循或语义线索，从而通过将其与语义和提示依赖效应隔离，提供了更直接的检索行为测量。

![](images/1.jpg)  

Figure 1: (a) Multiscreen architecture. The model comprises a stack of $N _ { \mathrm { L } }$ residual layers, each containing $N _ { \mathrm { H } }$ parallel gated screening tiles. The input embedding matrix is normalized and shared with the language-modeling head, with learned scalars $\mathrm { e } ^ { s _ { \mathrm { E } } }$ and $\mathrm { e } ^ { s _ { \mathrm { F } } }$ controlling input and output scaling. (b) A gated screening tile. The tile computes query, key, value, and gate projections, applies a screening unit to the projected queries, keys, and values, modulates the result with a nonlinear gate, and projects back to the model dimension. (c) A screening unit. The unit normalizes queries, keys, and values to unit length, applies minimal positional encoding (MiPE) to queries and keys, computes distance-aware relevance through Trim, Square, and Softmask, aggregates the surviving values, and applies TanhNorm. In the diagrams, $@$ " denotes matrix multiplication and "/RSS" denotes row-wise normalization to unit length.

# 3 模型架构

Transformer 层通常由自注意力模块和前馈网络组成。在标准的 softmax 注意力中，所有键被共同评估，其贡献通过在竞争键之间重新分配固定的单位质量来确定。因此，相关性仅相对于其他键定义，而无关键无法被清晰拒绝。为了解决这一限制，我们提出了一种称为筛选的机制，它能够实现绝对的查询-键相关性。筛选不是在所有键上形成一个归一化分布，而是精确地丢弃无关键，仅根据其相关性聚合剩余的键。这种表述消除了键之间的全局竞争，并取消了对总贡献的和为一的限制。多重筛选模型如图 1 所示。每一层用一组并行的门控筛选单元替换标准的注意力-前馈对。从高层次来看，门控筛选单元将词元表示投影到查询、键、值和门向量中，应用筛选单元以检索相关上下文，用受 GLU 风格乘法门控启发的非线性门调制检索到的表示，并将结果投影回模型空间。下面的方程描述了数学上等效的计算。在实际实现中，多个操作被融合，并且为了提高效率，学习到的筛选窗口外的项被跳过。

# 3.1 概述

给定一个词元输入序列 $( t _ { 1 } , \dots , t _ { T } )$ ，其中每个词元 $t _ { i } \in \{ 1 , \ldots , | \mathcal { V } | \}$ 索引到词汇表 $\nu$ ，我们定义一个嵌入矩阵

$$
W _ { \mathrm { E } } = [ e _ { 1 } ; \ldots ; e _ { | \mathcal { V } | } ] \in \mathbb { R } ^ { | \mathcal { V } | \times d _ { \mathrm { E } } } .
$$

我们将每个嵌入行归一化为单位长度，并使用学习到的缩放因子 $s _ { \mathrm { E } }$ 将每个词元映射到其嵌入。

$$
\bar { \boldsymbol { e } } _ { j } = \frac { \boldsymbol { e } _ { j } } { \left\| \boldsymbol { e } _ { j } \right\| } , \qquad \overline { { W } } _ { \mathrm { E } } = [ \bar { e } _ { 1 } ; \ldots ; \bar { e } _ { \left| \mathcal { V } \right| } ] ,
$$

$$
\pmb { x } _ { i } ^ { ( 0 ) } = \mathrm { e } ^ { s _ { \mathrm { E } } } \bar { \pmb { e } } _ { t _ { i } } .
$$

Table 1: Architectural hyperparameters of Multiscreen and their scaling with the supraparameter $\Psi$ .   

<table><tr><td>Hyperparameter</td><td>Symbol</td><td>Suggested</td><td>Used in our experiments</td></tr><tr><td>Number of layers</td><td>NL</td><td>I</td><td>I</td></tr><tr><td>Number of heads</td><td>NH</td><td>I</td><td>I</td></tr><tr><td>Embedding dim</td><td>dE</td><td>Ψ2</td><td>J2</td></tr><tr><td>Key dim</td><td>dK</td><td>16 or 32</td><td>16</td></tr><tr><td>Value dim</td><td>dv</td><td>64 or 128</td><td>64</td></tr><tr><td>MiPE threshold</td><td>Wth</td><td></td><td>256</td></tr><tr><td>Vocabulary size</td><td>|V|</td><td>−</td><td>50,257</td></tr></table>

Table 2: Comparison between Transformer attention and Multiscreen screening.   

<table><tr><td></td><td>Transformer</td><td>Multiscreen</td></tr><tr><td>Query-key dot product</td><td>attention score (unbounded)</td><td>similarity ( [−1, 1])</td></tr><tr><td>Weight computation</td><td>relative (softmax)</td><td>absolute (Trim / Square / Softmask)</td></tr><tr><td>Weights</td><td>attention weights (sum to 1)</td><td>relevance values ( [0, 1])</td></tr></table>

模型接着应用了 $N _ { \mathrm { L } }$ 个残差层，每个层包含 $N _ { \mathrm { H } }$ 个并行的门控筛选单元。我们将这些单元称为 tiles，它们在层和头的维度上形成一个规则的 $N _ { \mathrm { L } } \times N _ { \mathrm { H } }$ 网格。设 Δx(l) 表示层 $\ell$ 中由 tile $h$ 生成的对 token $i$ 的更新，则表示更新为

$$
\pmb { x } _ { i } ^ { ( \ell ) } = \pmb { x } _ { i } ^ { ( \ell - 1 ) } + \sum _ { h = 1 } ^ { N _ { \mathrm { H } } } \Delta \pmb { x } _ { i } ^ { ( \ell , h ) } .
$$

这种跨瓦片的聚合类似于Transformer中的多头聚合，但在每个瓦片中使用独立的筛选和GLU风格的门控。经过最后一层后，每个词元表示使用与输入嵌入相同的归一化嵌入矩阵 $\overline { { W _ { \mathrm { E } } } }$ 投影到词汇空间，并结合一个学习到的标量 $s _ { \mathrm { F } }$：

$$
z _ { i j } = \pmb { x } _ { i } ^ { ( N _ { \mathrm { L } } ) } \left( \mathrm { e } ^ { s _ { \mathrm { F } } } \bar { \pmb { e } } _ { j } ^ { \top } \right) , \qquad j \in \{ 1 , \dotsc , | \mathcal { V } | \} ,
$$

其中 $z _ { i j }$ 表示与词汇索引 $j$ 对应的标记 $i$ 的对数几率。标准的 softmax 产生下一个标记的概率。该模型采用了绑定和归一化的输入-输出嵌入结构，并使用单独学习的标量来控制输入和输出的尺度。最后，模型的尺度由单一的尺度参数 $\Psi$ 控制，该参数决定了层数、头数和嵌入维度，而所有其他架构超参数可以在模型尺度上保持固定，无需重新调整，如表 1 所示。在我们的默认缩放规则中，$N _ { \mathrm { L } } = N _ { \mathrm { H } } = \Psi$ 且 $d _ { \mathrm { E } } = \Psi ^ { 2 }$。术语：相似性与相关性。为了明确术语，我们区分注意力分数、注意力权重、相似性和相关性。在标准 Transformer 注意力中，查询和键的点积定义了注意力分数，该分数是无界的。这些分数通过 softmax 在键上进行归一化，以产生注意力权重，注意力权重是正数且和为一，从而在键之间引入竞争。相比之下，在 Multiscreen 中，查询和键的点积定义了范围在 $[ - 1 , 1 ]$ 的相似性，因为查询和键向量被归一化到单位长度，确保点积在范围 $[ - 1 , 1 ]$ 内。这种相似性随后独立地进行阈值处理和转换，以产生范围在 $[ 0 , 1 ]$ 的相关性值，而不在键上进行归一化。

# 3.2 筛选单元

我们现在描述图1c中显示的筛选单元。给定投影的查询、键和值向量，为每个词元计算输出向量 $\pmb { u } _ { i } \in \mathbb { R } ^ { 1 \times d \mathrm v }$。筛选通过阈值处理查询-键相似性，独立评估每个键，并进行转换以计算相关性，而不对键进行归一化。

$$
\pmb { q } _ { i } \in \mathbb { R } ^ { 1 \times d _ { \mathrm { K } } } , \qquad \pmb { k } _ { i } \in \mathbb { R } ^ { 1 \times d _ { \mathrm { K } } } , \qquad \pmb { v } _ { i } \in \mathbb { R } ^ { 1 \times d _ { \mathrm { V } } } ,
$$

筛选单元有两个学习到的标量参数 $s _ { \mathrm { w } }$ 和 $s _ { \mathrm { r } }$，分别定义了 $w$ 为筛选窗口，而 $1 / r$ 为相似性的接受宽度。

$$
w = \mathrm { e } ^ { s _ { \mathrm { w } } } + 1 , \quad \quad r = \mathrm { e } ^ { s _ { \mathrm { r } } } + 1 ,
$$

单位长度归一化。我们首先将查询、键和值归一化为单位长度：

$$
\bar { q } _ { i } = \frac { q _ { i } } { \left\| q _ { i } \right\| } , \qquad \bar { k } _ { i } = \frac { k _ { i } } { \left\| k _ { i } \right\| } , \qquad \bar { v } _ { i } = \frac { v _ { i } } { \left\| v _ { i } \right\| } .
$$

这确保了查询-键相似度在范围 $[ - 1 , 1 ]$ 内，从而为相关性提供了一致的尺度，并在后续筛选操作中设定了明确的阈值。它还消除了向量范数对这些相似度的影响，使得相关性仅依赖于查询和键之间的方向对齐。归一化值可以防止异常大的值范数主导聚合，从而消除在先前分析中突显的值范数效应 [31, 32]。最小位置编码。为了融入位置信息，我们引入了最小位置编码（MiPE），这是一种类似 RoPE 的旋转 [17]，仅应用于查询和键的前两个坐标，并且仅在学习到的筛选窗口足够小时激活，其中旋转角度由学习到的窗口参数 $w$ 自适应控制。对于位置 $i$ 的向量 $\boldsymbol { z } _ { i } \in \mathbb { R } ^ { 1 \times d _ { \mathrm { K } } }$，MiPE 被定义为：

$$
\tilde { z } _ { i } = z _ { i } M _ { i } ( w ) ,
$$

$$
M _ { i } ( w ) = \left( \begin{array} { c c } { { R ( \phi ( i , w ) ) } } & { { 0 } } \\ { { 0 } } & { { I _ { d _ { \mathrm { K } } - 2 } } } \end{array} \right) , \qquad R ( \phi ) = \left( \begin{array} { c c } { { \cos \phi } } & { { - \sin \phi } } \\ { { \sin \phi } } & { { \cos \phi } } \end{array} \right) ,
$$

$$
\phi ( i , w ) = \frac { \pi i \gamma ( w ) } { w } .
$$

这里 $\gamma(w)$ 是一个确定性函数，随着 $w$ 接近固定阈值 $w_{\mathrm{th}}$ 平滑地从 1 降至 0，并且当 $w \geq w_{\mathrm{th}}$ 时变为 0，从而在这一点之后禁用位置旋转：

$$
\gamma ( w ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \left( \cos \frac { \pi w } { w _ { \mathrm { t h } } } + 1 \right) , } & { w < w _ { \mathrm { t h } } , } \\ { 0 , } & { w \geq w _ { \mathrm { t h } } . } \end{array} \right.
$$

MiPE 仅在学习到的筛选窗口较短时发挥作用，且在需要长范围访问时变为恒等函数。因为 $M _ { i } ( w )$ 是正交的，MiPE 保持向量的规范性。将 MiPE 应用于归一化的查询和键，我们得到

$$
\tilde { \pmb q } _ { i } = \bar { \pmb q } _ { i } M _ { i } ( w ) , \qquad \tilde { \pmb k } _ { j } = \bar { \pmb k } _ { j } M _ { j } ( w ) .
$$

与 RoPE 一样，得到的相似度仅依赖于相对位置：

$$
\tilde { \pmb q } _ { i } \tilde { \pmb k } _ { j } ^ { \top } = \bar { \pmb q } _ { i } M _ { i - j } ( w ) \bar { \pmb k } _ { j } ^ { \top } .
$$

距离无关的相关性。使用位置编码的查询和键，我们计算查询键相似度。

$$
s _ { i j } = \tilde { q } _ { i } \tilde { k } _ { j } ^ { \top } , \qquad s _ { i j } \in [ - 1 , 1 ] .
$$

然后我们使用修剪平方变换定义一个距离无关的相关性 $\alpha _ { i j }$：

$$
\alpha _ { i j } = \left[ \operatorname* { m a x } \bigl ( 1 - r ( 1 - s _ { i j } ) , 0 \bigr ) \right] ^ { 2 } .
$$

当 $\begin{array} { r } { s _ { i j } \ \leq \ 1 - \ \frac { 1 } { r } } \end{array}$ 时，此变换将相关性精确设置为零，并平滑地强调接近最大值1的相似性，如图2所示。受限范围 $[ - 1 , 1 ]$ 使得该阈值定义良好。

![](images/2.jpg)  

Figure 2: Illustration of the Trim-and-Square transform (here shown with acceptance width $1 / r =$ $1 / 3 )$ . Only similarities greater than $1 - 1 / r$ produce nonzero relevance, illustrating the effective acceptance threshold.

软掩码。接下来，我们应用一种因果和距离感知的软掩码：

$$
m _ { i j } ( w ) = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \left( \cos \frac { \pi ( j - i ) } { w } + 1 \right) , } & { - w < j - i \leq 0 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

距离感知相关性是

$$
\alpha _ { i j } ^ { \mathrm { d } } = \alpha _ { i j } m _ { i j } ( w ) .
$$

因此，只有当一个关键在基于内容和基于距离的筛选中都存活时，它才会贡献。软掩码还可以使窗口边界的过渡平滑。由于窗口参数 $w$ 是可学习的，因此有效的上下文范围在训练过程中自适应地确定。此外，由于相关性是独立计算的，没有跨关键的归一化，每个值都可以独立调整，而不会影响其他值，从而允许对每个贡献进行简单和透明的控制。为了定性说明距离感知的相关性，我们在附录 E 中可视化它们的模式。在推理过程中，如果学习到的窗口宽度超过训练期间看到的最大序列长度，我们显式设置 $w = \infty$。这有效地去除了窗口约束，使得在前缀上的完全因果交互成为可能。在这个极限下，软掩码简化为标准的全因果掩码。加权聚合和 TanhNorm。筛选单元通过距离感知的相关性对值进行加权聚合：

$$
h _ { i } = \sum _ { j = 1 } ^ { T } \alpha _ { i j } ^ { \mathrm { d } } \bar { v } _ { j } .
$$

由于相关性 $\alpha _ { i j } ^ { \mathrm { d } }$ 未规范化为总和为一，因此筛选单元也可以表示缺乏相关上下文。为了在软性防止其范数过度增长的同时保留聚合表示，我们引入了一种规范化函数，称为 TanhNorm。

$$
\mathrm { T a n h N o r m } ( { \pmb x } ) = \frac { \operatorname { t a n h } \| { \pmb x } \| } { \| { \pmb x } \| } { \pmb x } .
$$

TanhNorm 保持向量的方向，对于小范数的输入大致表现为恒等映射，并平滑地将输出范数限制在 1。筛选单元的最终输出为

$$
\begin{array} { r } { { \pmb u } _ { i } = \mathrm { T a n h N o r m } ( { \pmb h } _ { i } ) . } \end{array}
$$

总体而言，筛选单元返回一个界定明确的上下文相关表示，该表示仅由经过基于相似性和基于距离的筛选存活下来的关键字组成。这种定义绝对相关性的能力使其在根本上与softmax注意力区分开来。

# 3.3 门控筛选瓦片

门控筛选模块是图1b所示的头部模块。它由三个组成部分构成：一个筛选单元，从投影的查询、键和值中提取上下文；一个受GLU风格乘法门控启发的非线性门，用于调节提取的信息；以及一个输出投影到模型维度。

Table 3: Parameter shapes and initialization of Multiscreen.   

<table><tr><td>Parameter</td><td>Shape</td><td>Initialization</td></tr><tr><td>WQ</td><td>(dE, )</td><td>N (0, 0.1/√dκ)</td></tr><tr><td>WK</td><td>(dE, dκ)</td><td>N(0, 0.1/√dκ)</td></tr><tr><td>Wv</td><td>(dE, v)</td><td>N (0, 0.1/√dv)</td></tr><tr><td>WG</td><td>(dE, v)</td><td>N(0, 0.1)</td></tr><tr><td>Wo</td><td>(, E)</td><td>N (0, 0.1/√dE)</td></tr><tr><td>WE</td><td>(V|, dE)</td><td>N(0, 0.1/√dE)</td></tr><tr><td>Sw</td><td>scalar</td><td>linearly spaced across heads from 0 to log wth in each layer</td></tr><tr><td>Sr</td><td>scalar</td><td>0</td></tr><tr><td>SO</td><td>scalar</td><td>log(1/√NHNL)</td></tr><tr><td>SE</td><td>scalar</td><td>0</td></tr><tr><td>SF</td><td>scalar</td><td>log √dE</td></tr></table>

给定输入词元表示 $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { T } \in \mathbb { R } ^ { 1 \times d _ { \mathrm { E } } }$，一个平铺首先为每个词元计算四个线性投影：

$$
\begin{array} { r } { q _ { i } = x _ { i } W _ { \mathrm { Q } } , \qquad k _ { i } = x _ { i } W _ { \mathrm { K } } , \qquad v _ { i } = x _ { i } W _ { \mathrm { V } } , \qquad g _ { i } = x _ { i } W _ { \mathrm { G } } , } \end{array}
$$

其中 $W _ { \mathrm { Q } } , W _ { \mathrm { K } } \in \mathbb { R } ^ { d _ { \mathrm { E } } \times d _ { \mathrm { K } } }$ 和 $W _ { \mathrm { V } } , W _ { \mathrm { G } } \in \mathbb { R } ^ { d _ { \mathrm { E } } \times d _ { \mathrm { V } } }$。使用 3.2 节中的符号，筛选单元产生

$$
\pmb { \mathscr { u } } _ { i } = \mathrm { S c r e e n i n g } \big ( \{ \pmb { q } _ { j } , \pmb { k } _ { j } , \pmb { v } _ { j } \} _ { j = 1 } ^ { T } \big ) _ { i } .
$$

同时，我们计算一个门矢量。

$$
\hat { \pmb { g } } _ { i } = \operatorname { t a n h } ( \operatorname { S i L U } ( \pmb { g } _ { i } ) ) .
$$

这个门在功能上类似于变换器前馈网络中的GLU风格门控。在这个结构中，可以将其视为GLU风格门控的推广，其中线性变换被基于筛选的聚合所替代。在我们的实验中，我们使用元素级非线性函数tanh(SiLU(·))进行门控。然后通过逐元素相乘将筛选的上下文和门结合起来：

$$
\begin{array} { r } { { \pmb h } _ { i } = { \pmb u } _ { i } \odot \hat { { \pmb g } } _ { i } . } \end{array}
$$

最后，瓷砖将其更新投影到模型维度：

$$
\Delta { \pmb x } _ { i } = \pmb { h } _ { i } \left( \mathrm { e } ^ { s _ { 0 } } W _ { 0 } \right) ,
$$

其中 $W _ { 0 } \in \mathbb { R } ^ { d _ { \mathrm { v } } \times d _ { \mathrm { E } } }$，$s _ { 0 }$ 是一个学习到的标量。因此，门控筛选模块首先对上下文进行筛选，然后应用依赖于输入的特征选择，最后将结果投影到同一空间。从高层次来看，这相当于在单一操作中进行基于筛选的上下文检索和GLU风格的门控初始化。我们将所有投影和嵌入矩阵初始化为零均值的高斯分布，其标准差与输出维度的平方根成反比，如表 3 所总结。门控投影 $W _ { \mathrm { G } }$ 的初始化采用独立于维度的固定尺度。窗口参数 $s _ { \mathrm { w } }$ 在每一层中被初始化为均匀分布在头部从 0 到 $\log w _ { \mathrm { t h } }$ 的值。残差输出尺度 $s _ { 0 }$ 被初始化为大约标准化所有模块的总贡献，确保随着模块数量的增加，聚合更新保持良好尺度。参数 $s _ { \mathrm { E } }$ 和 $s _ { \mathrm { F } }$ 将输入和输出尺度初始化为确保稳定训练的值。

# 4 实验

我们从多个维度对Multiscreen进行评估。首先是实验设置，其次分析在与标准Transformer相比时，在大学习率下的扩展行为和稳定性。接着，我们通过基于位置的困惑度和合成的键值检索基准研究长上下文能力，最后测量推理延迟。

# 4.1 实验设置

我们将Multiscreen与在匹配数据和词元预算下的Transformer基线进行比较。词元化。我们采用GPT-2词元器，词汇表规模为50,257个词元。训练数据。我们在SlimPajama数据集上对所有模型进行预训练，该数据集是RedPajama数据集的压缩版本。经过词元化后，数据集包含大约6280亿个词元。我们使用约44%的数据集进行预训练。模型规模。为了分析缩放行为，我们在多个参数规模下训练Transformer和Multiscreen模型。对于Transformer基线，我们采用了一种LLaMA风格的架构，架构超参数基于Pythia中使用的参数，并在输入嵌入和语言建模头之间应用权重绑定，以提高参数效率并增强基线的表现。Transformer基线的详细配置在附录A中提供。对于Multiscreen，架构超参数总结在表1中。训练设置。基础模型使用$2^{38}$个词元和序列长度为$2^{12}$进行预训练。对于长序列$2^{15}$，使用额外的$2^{27}$个词元。我们在两个阶段都使用全局批次大小为$2^{22}$个词元。所有模型都使用AdamW进行优化，参数为$(\beta_{1}, \beta_{2}) = (0.9, 0.95)$。在基础预训练期间，我们使用$2^{10}$个预热步骤，而在持续预训练中，我们从预训练的检查点继续，不进行额外的预热，继承优化器状态。对于所有模型，学习率在预热阶段后保持不变。对于Transformer，我们按照标准实践使用RoPE，设置$\theta = 10{,}000$，并采用一个基于Pythia的松散训练配置，包括权重衰减（0.1）、梯度裁剪（阈值1.0）和依赖于模型规模的学习率。学习率设置为在Pythia中为每个模型规模使用的峰值值：8M、18M和45M模型为$1 \times 10^{-3}$，124M模型为$6 \times 10^{-4}$，353M模型为$3 \times 10^{-4}$。对于Multiscreen，我们遵循与Transformer基线相同的优化器配置，同时省略权重衰减和梯度裁剪，这是我们发现对于Multiscreen的稳定训练而言并不必要的。我们使用的学习率为$\bar{2}^{-4}$，明显大于Transformer使用的学习率，利用了在第4.3节中展示的Multiscreen的改进稳定性。

# 4.2 扩展效率

我们评估了Multiscreen的扩展性，通过在多个参数规模上进行模型预训练。所有模型均以固定的词元预算进行训练，这可能在较大规模下导致训练不足。然而，这种比较反映了在相同训练预算下架构的相对效率。除了4B模型外，每个实验都进行了三次，使用不同的随机种子。由于计算限制，对于4B模型，我们仅报告一次运行结果。我们观察到在模型规模间的趋势是一致的。在图3中，我们报告了多次运行的平均值，误差条表示一个标准差。图3显示了模型大小与验证损失之间的关系。在评估的参数规模中，Multiscreen的扩展曲线在可比较的验证损失下，大约比Transformer少$40 \%$的参数。这表明，在相同训练预算下，Multiscreen的参数效率得到了改善。我们还验证了Multiscreen在其他模型大小度量下表现出一致的扩展性行为（见附录B）。

![](images/3.jpg)  

Figure 3: Scaling behavior of Transformer and Multiscreen. Validation loss is plotted against model size (number of parameters) on a log scale. Markers represent the mean over three runs, and error bars indicate one standard deviation (smaller than the marker size). For the 4B model, only a single run is available due to computational constraints. Multiscreen achieves similar validation loss at roughly $40 \%$ fewer parameters along the scaling trend compared to Transformer.

![](images/4.jpg)  

Figure 4: Learning rate sweep comparing Transformer and Multiscreen. The learning rate is shown on a log scale. Multiscreen remains stable even at large learning rates, while Transformer training becomes unstable as the learning rate increases. For Transformer, runs with learning rates $\geq 2 ^ { - \overline { { 4 } } }$ diverged and are omitted from the plot.

为了透明性，4B Multiscreen 模型的训练轨迹与较小的模型略有不同。它最初使用 Multiscreen 的一个稍早的架构变体训练了大约 $2 ^ { 3 8 }$ 个词元，之后转换为所有其他 Multiscreen 模型中使用的最终架构，并额外训练了 $2 ^ { 3 4 }$ 个词元（大约为最初训练的 $1 / 1 6$）。我们在缩放图中包含此模型，因为它的最终架构与其他 Multiscreen 系列相匹配。

# 4.3 学习率稳定性

基于标准 Transformer 架构训练大语言模型需要对学习率进行仔细调整。如果学习率过小，优化进展缓慢，训练变得低效。相反，过大的学习率往往导致训练不稳定或发散。为了研究 Multiscreen 对学习率的稳定性，我们对 45M Transformer 和 28M Multiscreen 模型进行了学习率扫频。我们在对数尺度上扫频学习率，从 $2^{ - 1 4 }$ 到 $2^{ 0 }$。对于每种架构，我们在与第 4.1 节相同的设置下训练模型，仅变化学习率。如图 4 所示，当学习率超过某个阈值时，Transformer 表现出不稳定的训练，且在相对适中的学习率下观察到发散。相反，即使在较大的学习率下，Multiscreen 仍然保持稳定，没有出现发散的迹象，包括 Transformer 训练失败的值。从相同实验的训练损失轨迹进一步说明了不稳定如何在优化过程中出现，Transformer 在较大的学习率下表现出越来越嘈杂或发散的行为，而 Multiscreen 则保持稳定（见附录 C）。这种改进的学习率稳定性与键之间缺乏竞争相一致。在实践中，这使得能够使用更大的学习率，从而促进更稳定和高效的优化。在我们的主要实验中，Multiscreen 的学习率为 $2^{ - 4 }$，这远大于 Transformer 的学习率（从 $3 \times 1 0^{ - 4 }$ 到 $1 \times 1 0^{ - 3 }$）。对于 Transformer，我们使用与 Pythia 一致的学习率，其中约 $1 \times 1 0^{ - 3 }$ 的值对于 45M 模型是最优的，正如我们的扫频结果所确认的。此外，我们进一步观察到两个架构之间的梯度动态 qualitatively 有所不同。Multiscreen 显示出快速衰减的梯度范数，这些范数保持接近零，而 Transformer 则维持着一个非零的梯度下限且变异性显著（见附录 D）。这一差异与 Multiscreen 在大学习率下的改进稳定性一致。

# 4.4 长上下文评估

我们使用两个互补的基准评估 Multiscreen 的长文本能力。首先，我们分析位置依赖的困惑度，以测量在长文本上下文中的语言建模性能。其次，我们通过一个合成的键值检索基准（ABCDigits）评估信息检索能力，该基准旨在隔离检索行为。

# 4.4.1 长文本困惑度

我们使用位置相关的困惑度评估长上下文语言建模性能，针对长序列进行评估。具体而言，我们比较了353M Transformer和286M Multiscreen模型。对于每种架构，我们使用与4.2节中相同的三个独立训练的基础模型，以及三个经过持续预训练的模型。每个位置的困惑度在一个集中于上下文长度的$\pm 10\%$窗口内取平均，以减少局部方差。我们使用PG-19数据集[2]，这是一个来自古腾堡计划的1919年前出版书籍的语料库进行评估。从数据集中，我们提取了5,747篇文档，其词元长度超过$2^{17}$。对于每个文档，我们提取一个包含$2^{17} + 1$个词元的连续片段，该片段位于文档的中间，以构建评估集。这确保了在长上下文的中间进行预测评估，而不是在文档边界附近。对于Transformer，我们在持续预训练阶段使用RoPE缩放因子为$\times 8$。在评估过程中，我们测试多个RoPE缩放因子，以评估超出训练长度的更长上下文的外推能力。对于基础模型，我们评估缩放因子$\times 1$、$\times 2$、$\times 4$、$\times 8$、$\times 16$、$\times 32$和$\times 64$，而对于持续预训练的模型，我们评估$\times 8$、$\times 16$、$\times 32$和$\times 64$。结果如图5所示，左侧面板评估基础模型，右侧面板显示持续预训练后的模型。Multiscreen在上下文长度增加时保持稳定的困惑度，并且在训练上下文之外没有明显退化。相比之下，当上下文长度超过训练范围时，Transformer表现出困惑度的突然增加。尽管提高RoPE缩放因子可以延迟这种崩溃，但也导致总体困惑度的增加。这种行为在基础模型和经过持续预训练的模型中是一致的。

# 4.4.2 合成键值检索：ABCDigits

为了更好地隔离检索能力，我们基于之前工作中研究的合成联想回忆和键值检索基准进行构建。我们将此任务表述为一个结构化补全问题，直接要求恢复相关值，从而更直接地探测检索行为，并减少来自语义、指令遵循和特定提示因素的干扰效应。

![](images/5.jpg)  

Figure 5: Long-context perplexity comparison between $3 5 3 \mathbf { M }$ Transformer and 286M Multiscreen models. The horizontal axis is context position, and the vertical axis is perplexity. The left panel shows the base models, while the right panel shows models after long-context continual pretraining. The black curve corresponds to Multiscreen, while colored curves correspond to Transformer with different RoPE scaling factors. Shaded regions indicate one standard deviation across three independently trained models. The dashed and dotted vertical lines indicate the sequence lengths used during base pretraining $( 2 ^ { 1 2 } )$ and long-context continual pretraining $( 2 ^ { 1 5 } )$ , respectively.

我们介绍了ABCDigits，这是一项在受控且无语义的环境中进行的基于补全的检索测试，旨在隔离检索行为。任务展示了一组被打乱的方程式，将每个大写字母映射到一个$n$位整数（例如，$\scriptstyle \mathtt { A } = 9 6 7 8 9 2$）。查询通过附加目标字母后跟等号（例如，${ \mathrm { L } } =$）形成，模型必须补全相应的整数。目标映射在上下文中恰好出现一次，这要求模型定位到这一唯一的出现。由于每个字母在上下文中始终映射到单个整数，这项任务在不需要明确指令的情况下，隐含地强制执行一一对应的键值关系。此外，在我们的构造中，独特键的数量保持固定（26个大写字母），与上下文长度无关。这消除了由增多键数引起的混淆效应，使我们能够更好地隔离检索行为。

为了确保极低频率的方程在上下文中不显得异常，我们分两个阶段构建上下文的非目标部分。首先，我们包括25个非目标字母数字方程的每个实例的恰好一个。然后，通过从一个高度偏斜的类别分布中抽样额外的非目标方程来填充剩余上下文，其权重与 $2^{0}, 2^{1}, \ldots, 2^{\dot{2}4}$ 成正比，其中字母到权重的分配对每个实例是随机的。这避免了不频繁的键值对显得异常，确保在上下文中自然地代表了各种频率。所有方程随后被洗牌。最后，唯一的目标方程（例如，$\mathtt{L} = 169428$）被插入到指定深度，并附加查询后缀 $\mathrm{L} = $。图6a中展示了一个具体例子。遵循通常在稀疏评估中使用的可视化协议，我们在上下文长度和目标深度的网格上测量检索准确性。对于每个上下文长度和深度，我们生成1000个独立的ABCDigits实例，并评估在贪婪解码（温度 $= 0$）下的精确匹配准确性。上下文长度范围从 $2^{12}$ 到 $2^{17}$，深度设置为0.1、0.3、0.5、0.7 和 0.9。深度是相对于上下文中字母数字方程的总数定义的，而不是令牌位置。我们评估与4.4.1节中相同的353M Transformer和286M Multiscreen模型。此外，我们还包括较小的28M Multiscreen模型，使用与4.2节中相同的基础模型以及经过持续预训练后的相应模型，以评估检索性能如何随模型大小的变化而变化。

![](images/6.jpg)  

Figure 6: (a) Example prompt for ABCDigits. (b) Retrieval accuracy heatmaps over context length (columns) and target depth (rows). Columns correspond to the two training settings: base models trained with context length $2 ^ { 1 2 }$ (left) and models after continual pretraining with context length $2 ^ { 1 5 }$ (right). Rows correspond to 353M Transformer (top), 286M Multiscreen (middle), and 28M Multiscreen (bottom). Colors indicate exact-match retrieval accuracy. For Transformer, each cell shows accuracy under the best-performing RoPE scaling factor selected from multiple candidates, and the number inside the cell indicates the selected factor. A dash ("-") indicates that no correct retrieval occurred. The dashed and dotted vertical lines mark the context lengths used during base pretraining $( 2 ^ { 1 2 } )$ and long-context continual pretraining $( 2 ^ { 1 5 } )$ , respectively.

对于Transformer，我们评估了多个RoPE缩放因子。在基础模型中，我们测试了缩放因子从$\times 1$到$\times 6 4$，而在持续预训练模型中，我们测试了从$\times 8$到$\times 6 4$。对于每个上下文长度和深度，我们报告了缩放因子中的最佳平均准确率。对于Multiscreen，我们报告了三种模型的平均准确率。结果显示在图6中。286M的Multiscreen在所有评估的上下文长度中均达到了近乎完美的检索准确率，即使没有持续预训练并且长于训练上下文。更小的28M Multiscreen表现略有下降，但仍然保持高准确率，在最长上下文长度$(2^{17})$下也能实现强大的检索表现，准确率保持在约$80\%$。相比之下，尽管为每个设置选择了最佳的RoPE缩放因子，Transformer的表现依然较差。当上下文长度超过训练长度时，检索准确率明显下降，错误在训练范围内已是常见现象。虽然持续预训练改善了Transformer的性能，但并未弥补与Multiscreen之间的巨大差距。这些结果表明，Multiscreen显著提升了检索能力，并展现出强大的长度泛化能力。值得注意的是，28M的Multiscreen在检索准确率上始终优于353M的Transformer，即使在训练上下文长度内，尽管其验证损失明显更高，这表明像验证损失这样的下一个标记预测指标并未完全捕捉到检索能力。最后，ABCDigits提供了一个受控的、无语义的环境，去除了自然语言的语义，在上下文长度之间固定键的数量，并确保目标输出是唯一确定的，而不依赖于遵循指令或语义提示。这一设计减轻了语义遮蔽和提示特定效应的混淆影响，使得检索行为的测量更加直接。ABCDigits的实现对外公开。

Table 4: Inference latency (seconds) for next-token prediction with context length 100,000.   

<table><tr><td>Model</td><td>Base</td><td>After continual pretraining</td></tr><tr><td>353M Transformer</td><td>4.04 ± 0.03 s</td><td>4.05 ± 0.04 s</td></tr><tr><td>286M Multiscreen</td><td>1.72 ± 0.05 s</td><td>1.26 ± 0.06 s</td></tr></table>

# 4.5 推理延迟

我们通过测量长输入序列的下一个标记预测的延迟来评估推理效率。具体来说，我们测量在输入上下文长度为 100,000 的情况下计算单个下一个标记预测所需的时间。我们比较与 4.4.1 节中相同的 353M Transformer 模型和 286M Multiscreen 模型。所有实验都在 NVIDIA RTX 4090 GPU 上进行。在推理之前，模型权重被转换为 bfloat16 精度。在因果掩码下以批量大小为 1 进行推理。我们不使用 KV 缓存，而是将完整输入序列在单次前向传递中处理。测量之前进行一次热身运行以稳定 GPU 执行。对于 Transformer，我们使用 torch.nn.functional.scaled_dot_product_attention。对于 Multiscreen，我们使用自定义的 Triton 实现的筛选模块 [41]。每个模型都使用与其架构相适应的实现。对于每个模型，我们进行 100 次独立测量，每种配置的总运行次数为 300 次。我们报告测量延迟的均值和标准差。如表 4 所示，Multiscreen 的延迟显著低于 Transformer。对于基础模型，Multiscreen 约快 $2.3 \times$，而经过持续预训练后的模型，加速提升至超过 $3 \times$。持续预训练后额外的加速可归因于在推理时筛选窗口的处理方式。如 3.2 节所述，在推理过程中，只要学习的筛选窗口超过训练期间看到的最大序列长度，我们就会显式设置 $w = \infty$，从而在前缀上实现完全因果交互。通过在更长序列上进行持续预训练，提高这个最大序列长度，可以让更多学习的筛选窗口在推理时保持有限。因此，更多的平铺操作于 $w \ne \infty$，减少所需的计算量。在我们的模型中，$w = \infty$ 的平铺约占基础模型的 $9.4 \%$，而在持续预训练后降至 $4.7 \%$。这些结果表明，Multiscreen 在保持相当或更好的语言建模性能的同时，显著提高了推理效率，特别是在长上下文设置中。

# 5 结论

在本研究中，我们识别出标准softmax注意力的一个基本局限性：它没有定义绝对的查询-键相关性。所有键都是共同评估的，注意力权重是根据相对的查询-键得分重新分配固定单位质量而获得的。因此，相关性仅相对于竞争键而言而定义，这阻止了对不相关键的明确拒绝，并使表示缺乏相关上下文变得困难。为了解决这一局限性，我们提出了Multiscreen，这是一种语言模型架构，围绕我们称之为筛选的机制构建，能够实现绝对查询-键相关性。筛选不是在所有键之间重新分配注意力，而是根据有限的查询-键相似度和明确的阈值独立评估每个键，丢弃不相关的键并汇聚剩余的键。这种结构消除了键之间的全局竞争，使每个键的相关性可以独立确定。在经验上，与Transformer基线相比，Multiscreen同時改善了参数效率、大学习率下的训练稳定性、检索能力和推理效率，同时在长上下文困惑度方面保持强劲表现，并在远远超过训练期间所见的上下文长度下几乎没有检索性能的下降。从更广泛的角度看，我们的发现表明，改善长上下文行为需要超越基于重新分配的机制，转向定义和利用绝对相关性的架构。这一观点为理解模型如何处理和利用上下文提供了新的视角，并可能使模型行为的分析更加透明和可解释。

# 致谢与资金披露

本研究使用了 Supermicro ARS-111GL-DNHR-LCC 和 FUJITSU 服务器 PRIMERGY CX2550 M7（Miyabi），在联合高性能计算先进中心（JCAHPC）进行，同时也利用了九州大学信息技术研究所提供的通用项目类别下的计算资源。KMN 获得了可持续量子人工智能创新中心的支持（JST 资助编号 JPMJPF2221）。此项工作的支持来自理研（RIKEN）的机构资金。

# References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.   
[2] Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In International Conference on Learning Representations, 2020.   
[3] Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Ré. Zoology: Measuring and improving recal in efficient language models. In International Conference on Learning Representations, 2024.   
[4] Ken Shi and Gerald Penn. Semantic masking in a needle-in-a-haystack test for evaluating large language model long-text capabilities. In Proceedings of the First Workshop on Writing Aids at the Crossroads of AI, Cognitive Science and NLP (WRAICOGS 2025), pages 1623, 2025.   
[5] Ken M. Nakanishi. Scalable-softmax is superior for attention. arXiv preprint arXiv:2501.19399, 2025.   
[6] Xuechen Zhang, Xiangyu Chang, Mingchen Li, Amit Roy-Chowdhury, Jiasi Chen, and Samet Oymak. Selective attention: Enhancing transformer through principled context control. In Advances in Neural Information Processing Systems, 2024.   
[7] André F. T. Martins and Ramón Fernandez Astudillo. From softmax to sparsemax: A sparse model of attention and multi-label classification. In International Conference on Machine Learning, 2016.   
[8] Ben Peters, Vlad Niculae, and André F.T. Martins. Sparse sequence-to-sequence models. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.   
[9] Gonçalo M. Correia, Vlad Niculae, and André F.T. Martins. Adaptively sparse transformers. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019.   
10] Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025.   
[11] Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, and Lili Qiu. Retrievalattention: Accelerating long-context LLM inference via vector retrieval. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.   
[12] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024.   
[13] Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, and Christopher Ré. Hyena hierarchy: Towards larger convolutional language models. In International Conference on Machine Learning, 2023.   
[14] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023.   
[15] Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. Samba: Simple hybrid state space models for efficient unlimited context language modeling. In International Conference on Learning Representations, 2025.   
[16] Ofir Press, Noah Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022.   
[17] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.   
[18] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.   
[19] Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. Longrope: Extending llm context window beyond 2 million tokens. In International Conference on Machine Learning, 2024.   
[20] Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, and Srinadh Bhojanapalli. Functional interpolation for relative positions improves long context transformers. In The Twelfth International Conference on Learning Representations, 2024.   
[21] Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. In Advances in Neural Information Processing Systems, 2023.   
[22] Jie Wang, Tao Ji, Yuanbin Wu, Hang Yan, Tao Gui, Qi Zhang, Xuan-Jing Huang, and Xiaoling Wang. Length generalization of causal transformers without position encoding. In Findings of the Association for Computational Linguistics: ACL 2024, 2024.   
[23] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157173, 2024.   
[24] Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014.   
[25] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.   
[26] Amirkeivan Mohtashami and Martin Jaggi. Random-access infinite context length for transformers. In Advances in Neural Information Processing Systems, 2023.

[27] Greg Kamradt. Needle in a haystack - pressure testing llms, 2023. Accessed on Jan 19, 2024.

[28] Aaron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, and Koray Kavukcuoglu. Conditional image generation with pixelcnn decoders. In Advances in Neural Information Processing Systems, 2016.

[29] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In International Conference on Machine Learning, 2017.

30] Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

[31] Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. Attention is not only a weight: Analyzing transformers with vector norms. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 2020.

[32] Zhiyu Guo, Hidetaka Kamigaito, and Taro Watanabe. Attention score is not all you need for token importance indicator in kv cache reduction: Value also matters. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024.

[33] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks, 107:311, 2018.

[34] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

[35] Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras.net/blog/ slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, 2023.

[36] Maurice Weber, Daniel Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, Ben Athiwaratkun, Rahul Chalamala, Kezhen Chen, Max Ryabinin, Tri Dao, Percy Liang, Christopher Ré, Irina Rish, and Ce Zhang. Redpajama: an open dataset for training large language models. In Advances in Neural Information Processing Systems, 2024.

[37] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

[38] Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, 2023.

[39] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

[40] Arize AI. Needle in a haystack - pressure testing llms, 2023. Accessed on Jan 19, 2024.

[41] Philippe Tllet, H. T. Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, 2019.

[42] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In North American Chapter of the Association for Computational Linguistics, 2019.

# A Transformer Baseline Configurations

We provide detailed architecture configurations for the Transformer baseline models used in our experiments (table 5).

Table 5: Architecture hyperparameters for the Transformer baseline models across different parameter scales.   

<table><tr><td>Hyperparameter</td><td>8M</td><td>18M</td><td>45M</td><td>124M</td><td>353M</td></tr><tr><td>Number of layers (NL)</td><td>6</td><td>6</td><td>6</td><td>12</td><td>24</td></tr><tr><td>Number of heads (NH)</td><td>4</td><td>8</td><td>8</td><td>12</td><td>16</td></tr><tr><td>Embedding dim (dE)</td><td>128</td><td>256</td><td>512</td><td>768</td><td>1024</td></tr><tr><td>Total params</td><td>7,613,312</td><td>17,584,384</td><td>44,609,024</td><td>123,550,464</td><td>353,453,056</td></tr><tr><td>Non-embedding params</td><td>1,180,416</td><td>4,718,592</td><td>18,877,440</td><td>84,953,088</td><td>301,989,888</td></tr></table>

The Transformer baseline adopts a LLaMA-style architecture [37], with architecture hyperparameters based on Pythia [38]. We apply weight tying between the input embedding and the language modeling head.

The head dimension is defined as $d _ { \mathrm { E } } / N _ { \mathrm { H } }$ The feeorward dimension is  to $\lfloor \frac { 8 } { 3 } d _ { \mathrm { E } } \rfloor$

Initialization. All weights are initialized from a normal distribution with standard deviation 0.02, following common practice [42, 34]. Following standard Transformer initialization practice, residual projection layers in attention and feed-forward modules are additionally scaled by $\mathrm { \dot { 1 } } / \mathrm { \sqrt { 2 N _ { L } } }$ .

# B Additional Scaling Analysis

We further analyze scaling behavior using alternative definitions of model size, as shown in fig. 7.

![](images/7.jpg)  
Figure 7: Scaling behavior under alternative definitions of model size. Left: scaling behavior of Transformer and Multiscreen with respect to non-embedding parameters. Right: scaling behavior of Multiscreen with respect to the supraparameter $\Psi$ The 4B Multiscreen model deviates from the scaling trend, which we attribute to undertraining.

Non-embedding parameters. In the left panel of fig. 7, we plot validation loss as a function of non-embedding parameters for both Transformer and Multiscreen. The same trend is observed, with an even clearer linear relationship, and the relative advantage of Multiscreen over Transformer is preserved under this parameterization, with a slightly steeper trend observed for Multiscreen.

Supraparameter $\Psi$ In the right panel of fig. 7, we analyze scaling behavior using a unified supraparameter $\Psi$ defined for Multiscreen. This parameterization also yields a clean scaling relationship, suggesting that $\Psi$ provides a unified characterization of model scaling for Multiscreen.

![](images/8.jpg)  
Ti l ajetorfrom e sme uns   how  rentai rates. Curves are smoothed using a moving average over 256 training steps. Transformer becomes unstable at moderately large learning rates, exhibiting noisy and eventually divergent behavior, whereas Multiscreen maintains stable convergence even at very large learning rates (e.g., $2 ^ { 0 }$ ).

In both parameterizations, the 4B Multiscreen model deviates from the fitted scaling trend, which we attribute to undertraining at this scale, as larger models require more tokens to reach comparable convergence, suggesting that it would align with the scaling law if sufficiently trained.

# C Training Loss Dynamics

To complement the learning-rate sweep in fig. 4, we visualize training-loss trajectories from the same training runs, highlighting representative learning rates.

As shown in fig. 8, Transformer exhibits increasingly unstable behavior as the learning rate increases. At moderate learning rates, training becomes increasingly noisy with frequent spikes, and at larger values the optimization fails to converge. In contrast, Multiscreen demonstrates stable and smooth convergence even at large learning rates. Notably, even at very large learning rates (e.g., $2 ^ { 0 }$ ), Multiscreen continues to train reliably without divergence.

These results provide a complementary view to the validation-based learning-rate sweep. While fig. 4 summarizes the final performance, the training trajectories shown here illustrate how instability emerges during optimization. To better understand these behaviors, we next analyze the gradient dynamics in Appendix D. Together, these observations suggest that Multiscreen mitigates destabilizing gradient dynamics, leading to more robust optimization and enabling the use of substantially larger learning rates.

# D Gradient Norm Dynamics

We report gradient norms from the same training runs used in section 4.2, focusing on 353M Transformer and 286M Multiscreen. For each architecture, models are trained with three random seeds; we observe consistent behavior across runs and visualize a representative run.

As shown in fig. 9, Multiscreen exhibits rapidly vanishing gradient norms with minimal variance, while Transformer maintains a non-zero gradient floor with occasional spikes.

This suggests that the observed gradient dynamics are consistent with the absence of competition across keys, and with the improved stability of Multiscreen under large learning rates.

# E Visualization of Distance-Aware Relevance

We visualize distance-aware relevance maps $\underset { . } { \alpha _ { i j } ^ { \mathrm { d } } }$ for all tiles in a Multiscreen base model with $\Psi = 8$ (approximately 4M parameters), corresponding to one of the models used in section 4.2. Each map corresponds to a single tile.

![](images/9.jpg)  
Figure 9: Gradient norm dynamics during training for Transformer and Multiscreen. Multiscreen exhibits rapidly decaying gradient norms with minimal variance, while Transformer maintains a non-zero gradient floor with occasional spikes. For visualization, values above 1 are clipped and shown with $\times$ markers.

The input sequence is a 203-token natural-language passage (the abstract of this paper). All layers and heads are shown, arranged in a grid where rows correspond to layers and columns correspond to heads.

Within each map, dark gray regions indicate positions outside the learned screening window, while color intensity represents the magnitude of $\dot { \alpha } _ { i j } ^ { \mathrm { d } }$ within the window. Each tile is annotated with its layer and head indices, the learned window width $w$ , the acceptance width $1 / r$ , and the fraction of nonzero relevance values within the window, $\mathrm { P r } ( \alpha _ { i j } ^ { \mathrm { d } } > 0 ) ,$ .

Different tiles cover different context ranges, with some focusing on local neighborhoods and others covering broader portions of the sequence. Many maps contain a substantial number of zero entries, although the degree of sparsity varies across tiles.

![](images/10.jpg)  
Figure 10: Distance-aware relevance maps across layers and heads. Each map shows the distanceaware relevance, with rows and columns corresponding to query and key positions. Darker gray regions indicate positions outside the learned screening window. Each tile is annotated with its layer and head indices, the learned window width $w$ , the acceptance width $1 / r$ , and the fraction of nonzero relevance values $\mathrm { P r } ( \alpha _ { i j } ^ { \mathrm { d } } > 0 )$ within the window.