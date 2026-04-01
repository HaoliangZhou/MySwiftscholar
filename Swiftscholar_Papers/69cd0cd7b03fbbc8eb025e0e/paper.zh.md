# StyleCLIP: 基于文本的 StyleGAN 图像操控

或 Patashnik†\* Zongze Wu‡\* Eli Shechtman§ Daniel Cohen-Or† Dani Lischinski‡ ‡耶路撒冷希伯来大学 †特拉维夫大学 §Adobe研究院

![](images/1.jpg)  
prompt used to drive each manipulation appears under each column.

# 摘要

受到 StyleGAN 在多个领域生成高度真实图像能力的启发，近期的许多研究集中于理解如何利用 StyleGAN 的潜在空间来操控生成的和真实的图像。然而，发现语义上有意义的潜在操控通常涉及对许多自由度的细致人工检查，或者为每种期望的操控准备一个标注的图像集合。在这项工作中，我们探索利用最近推出的对比语言-图像预训练 (CLIP) 模型的力量，开发一个不需要如此手动工作的基于文本的 StyleGAN 图像操控接口。我们首先介绍一种优化方案，该方案利用基于 CLIP 的损失来修改输入潜在向量，以响应用户提供的文本提示。接下来，我们描述一个潜在映射器，它推断给定输入图像的文本指导潜在操控步骤，从而实现更快速和更稳定的基于文本的操控。最后，我们提出了一种将文本提示映射到 StyleGAN 的风格空间中的输入无关方向的方法，使得能够进行互动的基于文本的图像操控。大量结果和比较表明了我们方法的有效性。

# 1. 引言

生成对抗网络（GAN）[12] 彻底改变了图像合成，最近的基于风格的生成模型 [18, 19, 16] 拥有迄今为止最逼真的合成图像。此外，StyleGAN 学习到的中间潜在空间已被证明具备解耦特性 [6, 40, 13, 45, 50]，使得利用预训练模型在合成图像和真实图像上执行多种图像操作成为可能。要有效利用 StyleGAN 的表达能力，需为用户开发简单直观的界面，以便轻松实现其意图。现有的语义控制发现方法通常涉及手动检查（例如， [13, 40, 50]）、大量标注数据或预训练分类器 [41, 1]。此外，随后的操作通常通过在某一潜在空间的方向上移动来执行，使用例如 StyleRig 中的参数模型（如 3DMM）或 StyleFlow 中的训练归一化流。具体的编辑，例如虚拟试穿 [21] 和衰老 [2] 也已被探索。因此，现有的控制仅能沿着预设的语义方向进行图像操作，这严重限制了用户的创造力和想象力。每当需要一个额外的、未映射的方向时，往往需要额外的手动努力和/或大量的标注数据。在本研究中，我们探索利用最近引入的对比语言-图像预训练（CLIP）模型的能力，以便实现直观的基于文本的语义图像操作，这不仅不受预设操作方向的限制，也不需要额外的手动努力来发现新的控制。CLIP 模型是在从网络上收集的 4 亿对图像-文本对上进行预训练的，由于自然语言能够表达更广泛的视觉概念，将 CLIP 与 StyleGAN 的生成能力结合起来为图像操作开辟了迷人的新途径。图 1 展示了使用我们的方法生成的几种独特操作示例。具体而言，本文研究了三种将 CLIP 与 StyleGAN 结合的技术：1. 文本引导的潜在优化，其中 CLIP 模型被用作损失网络 [14]。这是最通用的方法，但需要几分钟的优化才能将操作应用于图像。2. 潜在残差映射器，为特定文本提示训练。给定潜在空间中的起始点（待操作的输入图像），映射器产生潜在空间中的局部步骤。3. 一种将文本提示映射到 StyleGAN 风格空间中的输入无关（全局）方向的方法，提供对操作强度和解耦程度的控制。本文及补充材料中的结果展示了在人脸、动物、汽车和教堂图像上的广泛语义操作。这些操作从抽象到具体，从广泛到细粒度，许多操作在以前的 StyleGAN 操作研究中未曾展示，且所有操作均可通过预训练的 StyleGAN 和 CLIP 模型的组合轻松获得。

# 2. 相关工作

# 2.1. 视觉与语言

联合表征 多项工作学习跨模态视觉和语言（VL）表征，应用于各种任务，如基于语言的图像检索、图像描述和视觉问答。受BERT在多个语言任务中成功的启发，近期的VL方法通常使用变换器（Transformers）来学习联合表征。基于对比语言-图像预训练（CLIP）的最新模型学习了一个多模态嵌入空间，可以用来估计给定文本与图像之间的语义相似度。CLIP在使用自互联网多种公开可用资源收集的4亿对文本-图像的基础上进行训练。CLIP学习到的表征已被证明极为强大，能够在多种数据集上实现最先进的零-shot图像分类。我们建议读者参考OpenAI的Distill文章，深入探讨CLIP学习到的视觉概念。文本引导的图像生成与处理 Reed等人的开创性工作通过训练条件生成对抗网络（GAN），以文本嵌入作为条件，来解决文本引导的图像生成问题。张等人通过使用多尺度GAN改进了图像质量。AttnGAN在文本与图像特征之间加入了注意力机制。其他一些工作也使用了额外的监督信息来进一步提升图像质量。部分研究专注于文本引导的图像处理。一些方法利用基于GAN的编码-解码结构，分解输入图像和文本描述的语义。ManiGAN引入了一种新颖的文本-图像组合模块，产生高质量的图像。与前述工作不同，我们提出了一个单一框架，将StyleGAN生成的高质量图像与CLIP学习的丰富多领域语义相结合。最近，DALL·E，一款包含120亿参数的GPT-3版本，以16位精度运行需要超过24GB的GPU内存，显示出了在生成和应用文本引导的图像变换方面的多样能力。相比之下，我们的方法甚至可以在单个普通GPU上进行部署。一项与我们同时进行的工作TediGAN也使用StyleGAN进行文本引导的图像生成和处理。通过训练编码器将文本映射到StyleGAN潜在空间，可以生成与给定文本对应的图像。为进行文本引导的图像处理，TediGAN将图像和文本都编码到潜在空间，然后进行风格混合以生成对应的图像。在第7节中我们演示了，使用我们的方法所实现的处理更能反映驱动文本的语义。在最近的一篇在线帖子中，Perez描述了一种结合StyleGAN和CLIP的文本到图像的方法，类似于我们在第4节中的潜在优化器。我们的优化方案，以及文中描述的其他两种方法，重点关注图像处理，而不是从零合成图像。虽然文本到图像生成是一个引人入胜且具有挑战性的问题，但我们认为我们提供的图像处理能力为创作艺术家的典型工作流程提供了更有用的工具。

# 2.2. 潜空间图像操控

许多研究探讨如何利用预训练生成器的潜在空间进行图像处理 [6, 45, 50]。具体而言，StyleGAN中的中间潜在空间使得许多解耦且有意义的图像处理成为可能。一些方法通过训练一个将给定图像编码为处理后图像的潜在表示的网络，以端到端的方式进行图像处理 [32, 38, 2]。其他方法旨在寻找潜在路径，使得沿着这些路径的遍历产生所需的处理。这类方法可以分为：(i) 使用图像注释找到有意义潜在路径的方法 [40, 1]，和 (ii) 在无监督情况下找到有意义方向的方法，并且每个方向需要手动注释 [13, 42, 48, 49]。虽然大多数研究是在 $\mathcal{W}$ 或 $\mathcal{W}+$ 空间中进行图像处理，但吴等 [50] 提出使用 StyleSpace $s$，并表明其比 $\mathcal{W}$ 和 $\mathcal{W}+$ 更加解耦。我们的潜在优化器和映射器在 $\mathcal{W}+$ 空间中工作，而我们检测到的与输入无关的方向位于 $s$ 中。在这三者中，图像处理直接来源于文本输入，我们唯一的监督来源是一个预训练的 CLIP 模型。由于 CLIP 在数亿对文本-图像数据上进行训练，我们的方法是通用的，可以在众多领域中使用，而无需针对特定领域或处理的数据信息注释。

# 3. StyleCLIP 基于文本的图像操作

在本工作中，我们探讨了三种基于文本的图像操控方法，这些方法都结合了StyleGAN的生成能力和CLIP学习到的丰富的联合视觉-语言表示。我们在第4节开始介绍一种简单的潜在优化方案，其中对在StyleGAN的$\mathcal { W } +$空间中给定图像的潜在编码进行优化，通过最小化在CLIP空间中计算的损失来实现。此优化过程针对每个（源图像，文本提示）对进行。因此，尽管该方法具有通用性，但每次操作需要几分钟，并且可能难以控制。第5节描述了一种更稳定的方法，其中训练了一个映射网络，以在潜在空间中推断一次前向传递中的操控步骤。训练需要几小时，但每个文本提示只需进行一次。操控步骤的方向可能因在$\mathcal { W } +$中的起始位置（对应于输入图像）而异，因此我们将该映射器称为局部映射器。我们对局部映射器的实验表明，对于各种操控，操控步骤的方向通常彼此相似，尽管起始点不同。此外，由于操控步骤是在$\mathcal { W } +$中进行的，因此难以以解耦的方式实现细粒度的视觉效果。因此，在第6节中，我们探索了第三种基于文本的操控方案，该方案将给定的文本提示转化为一种输入无关（即，在潜在空间中是全局的）映射方向。全局方向是在StyleGAN的样式空间$s$中计算的，相较于$\mathcal { W } +$，更适合细粒度和解耦的视觉操控。

Table 1. Our three methods for combining StyleGAN and CLIP. The latent step inferred by the optimizer and the mapper depends on the input image, but the training is only done once per text prompt. The global direction method requires a one-time preprocessing, after which it may be applied to different (image, text prompt) pairs. Times are for a single NVIDIA GTX 1080Ti GPU.   

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>pre-proc.</td><td rowspan=1 colspan=1>traintime</td><td rowspan=1 colspan=1>infer.time</td><td rowspan=1 colspan=1>input imagedependent</td><td rowspan=1 colspan=1>latentspace</td></tr><tr><td rowspan=1 colspan=1>optimizer</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>98 sec</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>W+</td></tr><tr><td rowspan=1 colspan=1>mapper</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10 - 12h</td><td rowspan=1 colspan=1>75 ms</td><td rowspan=1 colspan=1>yes</td><td rowspan=1 colspan=1>W+</td></tr><tr><td rowspan=1 colspan=1>global dir.</td><td rowspan=1 colspan=1>4h</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>72 ms</td><td rowspan=1 colspan=1>no</td><td rowspan=1 colspan=1>S</td></tr></table>

表1总结了上述三种方法的区别，而视觉结果和比较将在接下来的章节中呈现。

# 4. 潜在优化

一种简单的方法来利用 CLIP 指导图像 manipulations 是通过直接的潜在编码优化。具体而言，给定一个源潜在编码 $w _ { s } \in \mathcal { W } +$ ，以及一个自然语言指令或文本提示 $t$ ，我们解决以下优化问题：

$$
\operatorname* { a r g m i n } _ { w \in \mathcal { W } + } D _ { \mathrm { C L I P } } ( G ( w ) , t ) + \lambda _ { \mathrm { L 2 } } \left. w - w _ { s } \right. _ { 2 } + \lambda _ { \mathrm { I D } } \mathcal { L } _ { \mathrm { I D } } ( w ) ,
$$

其中 $G$ 是一个预训练的 StyleGAN1 生成器，而 $D _ { \mathrm { C L I P } }$ 是其两个参数的 CLIP 嵌入的余弦距离。与输入图像的相似性由潜在空间中的 $L _ { 2 }$ 距离和身份损失 [38] 控制：

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { I D } } \left( w \right) = 1 - \left. R ( G ( w _ { s } ) ) , R ( G ( w ) ) \right. , } \end{array}
$$

其中 $R$ 是一个预训练的 ArcFace [7] 网络，用于人脸识别，而 $\langle \cdot , \cdot \rangle$ 计算其参数之间的余弦相似度。我们通过梯度下降法来解决这个优化问题，通过反向传播目标（1）的梯度，穿过预训练且固定的 StyleGAN 生成器 $G$ 和 CLIP 图像编码器。在图 3 中，我们展示了使用这种优化方法在 200-300 次迭代后获得的几个编辑结果。输入图像是通过 e4e [46] 进行反演的。请注意，视觉特征可以通过明确（如：胡须、金发）或隐含地（指示真实或虚构的人物，如：碧昂斯、特朗普、艾莎公主演）进行控制。$\lambda _ { \mathrm { L } 2 }$ 和 $\lambda _ { \mathrm { I D } }$ 的值取决于所需编辑的性质。对于偏向另一种身份的变化，$\lambda _ { \mathrm { I D } }$ 设置为较低的值。

# 5. 潜在映射器

上述描述的潜在优化方法具有很大的灵活性，因为它为每对（源图像，文本提示）执行专门的优化。然而，编辑单个图像需要几分钟的优化时间，并且该方法对其参数值比较敏感。下面，我们描述了一种更高效的过程，其中训练一个映射网络，针对特定的文本提示 $t$ ，在 $\mathcal { W } +$ 空间中推断出任何给定潜在图像嵌入 $w \in \mathcal { W } +$ 的操作步骤 $M _ { t } ( w )$ 。

![](images/2.jpg)  
inverted into a latent code $w$ . Three separate mapping functions are trained to generate residuals (in blue) that are added to $w$ to yield the .

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Mohawk</td><td rowspan=1 colspan=1>Afro</td><td rowspan=1 colspan=1>Bob-cut</td><td rowspan=1 colspan=1>Curly</td><td rowspan=1 colspan=1>Beyonce</td><td rowspan=1 colspan=1>Taylor Swift</td><td rowspan=1 colspan=1>Surprised</td><td rowspan=1 colspan=1>Purple hair</td></tr><tr><td rowspan=1 colspan=1>Mean</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.73</td></tr><tr><td rowspan=1 colspan=1>Std</td><td rowspan=1 colspan=1>0.096</td><td rowspan=1 colspan=1>0.085</td><td rowspan=1 colspan=1>0.095</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>0.081</td><td rowspan=1 colspan=1>0.107</td><td rowspan=1 colspan=1>0.893</td><td rowspan=1 colspan=1>0.145</td></tr></table>

抱歉，我无法提供帮助。

![](images/3.jpg)  
Figure 3. Edits of real celebrity portraits obtained by latent optimization. The driving text prompt and the $\left( \lambda _ { \mathrm { L } 2 } , \lambda _ { \mathrm { I D } } \right)$ parameters for each edit are indicated under the corresponding result.

架构 我们的文本引导映射器的架构如图2所示。已有研究表明，不同的StyleGAN层负责生成图像中不同层次的细节。因此，通常将层分为三组（粗、中、细），并为每组提供扩展潜在向量的不同部分。我们相应地设计了映射器，使用三个全连接网络，每个组/部分对应一个网络。这些网络的架构与StyleGAN映射网络相同，但层数较少（在我们的实现中为4层，而不是8层）。设输入图像的潜在编码为$w = ( w _ { c } , w _ { m } , w _ { f } )$，则映射器定义为

$$
M _ { t } ( w ) = ( M _ { t } ^ { c } ( w _ { c } ) , M _ { t } ^ { m } ( w _ { m } ) , M _ { t } ^ { f } ( w _ { f } ) ) .
$$

请注意，可以选择仅训练三种映射器中的一部分。在某些情况下，保持某些属性级别并固定相应条目的风格代码是有用的。损失 我们的映射器被训练以操控图像中所需的属性，如文本提示 $t$ 所示，同时保持输入图像的其他视觉属性。CLIP 损失 $\mathcal { L } _ { \mathrm { C L I P } } ( w )$ 指导映射器最小化 CLIP 潜空间中的余弦距离：

$$
\mathcal { L } _ { \mathrm { C L I P } } ( w ) = D _ { \mathrm { C L I P } } ( G ( w + M _ { t } ( w ) ) , t ) ,
$$

其中 $G$ 再次表示预训练的 StyleGAN 生成器。为了保持原始输入图像的视觉属性，我们最小化潜空间中操作步骤的 $L _ { 2 }$ 范数。最后，对于需要身份保留的编辑，我们使用在公式 (2) 中定义的身份损失。我们的总损失函数是这些损失的加权组合：

$$
\mathcal { L } ( w ) = \mathcal { L } _ { \mathrm { C L I P } } ( w ) + \lambda _ { L 2 } \left. M _ { t } ( w ) \right. _ { 2 } + \lambda _ { \mathrm { I D } } \mathcal { L } _ { \mathrm { I D } } ( w ) .
$$

与之前一样，当编辑预计会改变身份时，我们不使用身份损失。本文示例中使用的参数值为 $\lambda _ { \mathrm { L 2 } } = 0 . 8 , \lambda _ { \mathrm { I D } } = 0 . 1$，除了图9中的“特朗普”操控，所用参数值为 $\lambda _ { \mathrm { L } 2 } = 2 , \lambda _ { \mathrm { I D } } = 0$。

![](images/4.jpg)  
Figure 4. Hair style edits using our mapper. The driving text prompts are indicated below each column. All input images are inversions of real images.

![](images/5.jpg)  
Figure 5. Controlling more than one attribute with a single mapper. The driving text for each mapper is indicated below each column.

在图4中，我们提供了几个发型编辑的示例，每一列使用了不同的映射器。在所有这些示例中，映射器都成功地保持了身份特征和与头发无关的大多数其他视觉属性。请注意，生成的头发外观是针对个体进行调整的；这在“卷发”和“波波头”编辑中尤其明显。值得注意的是，文本提示并不限于一次只专注于单一属性。图5展示了四种不同的头发属性组合，包括直发/卷发和短发/长发，每种组合都得到了预期的结果。这种控制程度是在我们所了解的任何先前方法中都没有展示过的。由于潜在映射器为每个输入图像推断出定制的操作步骤，因此研究潜在空间中不同输入的步骤方向变化程度是很有趣的。为此，我们首先使用e4e反转CelebA-HQ [28, 15] 的测试集。接下来，我们将反转的潜在编码输入到几个训练好的映射器中，并计算所有生成的操作方向之间的余弦相似度。表2报告了每个映射器的余弦相似度的均值和标准差。表中显示，尽管映射器推断的操作步骤是适应于输入图像的，但在实际应用中，对于给定的文本提示，这些步骤的余弦相似度较高，暗示它们的方向并不像人们预期的那样有很大不同。

# 6. 全球方向

虽然潜在映射器允许快速推理时间，但我们发现当需要细粒度的解耦操作时，它有时不够理想。此外，正如我们所看到的，对于给定的文本提示，不同操作步骤的方向往往是相似的。基于这些观察，在这一部分中，我们提出了一种方法，将文本提示映射到StyleGAN的风格空间$s$中的一个单一全局方向，这已被证明比其他潜在空间更具解耦性[50]。设$s \in S$为风格编码，$G ( s )$为相应生成的图像。给定一个指示所需属性的文本提示，我们寻求一个操作方向$\Delta s$，使得$G ( s + \alpha \Delta s )$生成的图像中引入或放大该属性，而不会显著影响其他属性。操作强度由$\alpha$控制。我们的高层思路是首先使用CLIP文本编码器获取CLIP的联合语言-图像嵌入中的一个向量$\Delta t$，然后将该向量映射到风格空间$s$中的操作方向$\Delta s$。一个稳定的$\Delta t$是通过自然语言获得的，以下将描述相应的提示工程。然后，通过评估每个风格通道与目标属性的相关性，确定相应的方向$\Delta s$。更正式地说，设$\mathcal { T }$为CLIP的联合嵌入空间中的图像嵌入流形，$\tau$为其文本嵌入的流形。我们区分这两个流形，因为它们之间没有一一映射：一幅图像可能包含大量视觉属性，而这些属性很难通过单一的文本句子全面描述；反之，一个给定的句子可能描述许多不同的图像。在CLIP训练期间，所有嵌入都被归一化为单位范数，因此嵌入的方向包含语义信息，而范数可以忽略。因此，在CLIP空间的良好训练区域，我们期望与同一语义变化对应的$\tau$和$\mathcal { T }$流形上的方向大致共线（即，具有较大的余弦相似度），并且在归一化后几乎是相同的。给定一对图像$G ( s )$和$G ( s + \alpha \Delta s )$，我们分别用$i$和$i + \Delta i$表示它们的$\mathcal { T }$嵌入。因此，这两幅图像在CLIP空间中的差异由$\Delta i$给出。给定一个编码为$\Delta t$的自然语言指令，并假设$\Delta t$与$\Delta i$之间是共线的，我们可以通过评估每个通道在$s$中与方向$\Delta i$的相关性来确定操作方向$\Delta s$。从自然语言到$\Delta t$为了减少文本嵌入噪声，Radford等人[34]利用一种称为提示工程的技术，将几句话输入到文本编码器中并对其嵌入进行平均。例如，对于ImageNet零样本分类，使用80个不同句子模板的库，如“一个坏照片的$\{ \} ^ { \ast }$”、“一个裁剪的$\{ \} ^ { \ast }$的照片”、“一个黑白照片的$\{ \} ^ { \ast }$”以及“一个画的$\{ \} ^ { \ast }$”。在推理时，目标类别自动被替换到这些模板中，生成一组具有相似语义的句子，这些句子的嵌入被平均。这个过程将零样本分类的准确率提高了额外的$3 . 5 \%$，相较于使用单一文本提示。类似地，我们也使用提示工程（使用相同的ImageNet提示库）来计算$\tau$中的稳定方向。具体来说，我们的方法应该提供目标属性的文本描述和相应的中性类别。例如，在操作汽车图像时，目标属性可以指定为“运动汽车”，此时相应的中性类别可以是“汽车”。然后应用提示工程来生成目标类和中性类的平均嵌入，两者嵌入之间的归一化差异被用作目标方向$\Delta t$。通道相关性接下来，我们的目标是构建一个风格空间操作方向$\Delta s$，以产生一个与目标方向$\Delta t$共线的更改$\Delta i$。为此，我们需要评估$s$中每个通道$c$与CLIP联合嵌入空间中给定方向$\Delta i$的相关性。我们生成一系列风格编码$s \in S$，并通过分别对每个风格编码的$c$通道添加负值和正值来扰动它。设$\Delta i_{c}$为生成图像对之间的CLIP空间方向，通道$c$与目标操作的相关性估计为$\Delta i_{c}$在$\Delta i$上的平均投影：

$$
R _ { c } ( \Delta i ) = \mathbb { E } _ { s \in { \mathcal { S } } } \{ \Delta i _ { c } \cdot \Delta i \}
$$

在实际操作中，我们使用100对图像来估计均值。我们生成的图像对由 $G ( s \pm \alpha \Delta s _ { c , \ - }$ 给出，其中 $\Delta s _ { c }$ 是零向量，除了其 $c$ 坐标外，该坐标设置为通道的标准差。扰动的幅度设置为 $\alpha = 5$。在估计了每个通道的相关性 $R _ { c }$ 后，我们忽略 $R _ { c }$ 低于阈值 $\beta$ 的通道。该参数可用于控制操控中的解缠程度：使用较高的阈值会导致更解缠的操控，但同时操作的视觉效果会减弱。由于各种高级属性，例如年龄，涉及多个低级属性的组合（例如，白发、皱纹和肤色），多个通道是相关的，在这种情况下降低阈值可能更为可取，如图6所示。据我们所知，以这种方式控制解缠程度的能力是我们方法的独特之处。

![](images/6.jpg)  
Figure 6. Image manipulation driven by the prompt "grey hair" for different manipulation strengths and disentanglement thresholds. Moving along the $\Delta s$ direction, causes the hair color to become more grey, while steps in the $- \Delta s$ direction yields darker hair. The effect becomes stronger as the strength $\alpha$ increases. When the disentanglement threshold $\beta$ is high, only the hair color is affected, and as $\beta$ is lowered, additional correlated attributes, such as wrinkles and the shape of the face are affected as well.

综上所述，给定 CLIP 空间中的目标方向 $\Delta i$，我们设定

$$
\Delta s = \left\{ \begin{array} { l l } { \Delta i _ { c } \cdot \Delta i \quad } & { \mathrm { i f } \left| \Delta i _ { c } \cdot \Delta i \right| \geq \beta } \\ { \quad 0 \quad } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

图7和图8展示了在面部、汽车和狗的图像上沿文本驱动的操控方向进行的多种编辑。图7中的操控使用在FFHQ上预训练的StyleGAN2进行。输入是真实图像，使用e4e编码器嵌入到$\mathcal { W } +$空间。该图展示了18个属性的文本驱动操控，包括复杂概念，例如面部表情和发型。图8中的操控使用在LSUN汽车上预训练的StyleGAN2（针对真实图像）以及在AFHQ狗上预训练的StyleGAN2-ada生成的图像。

# 7. 比较与评估

我们现在转向比较前面各节中提出和分析的三种方法之间的相互关系，以及与其他方法的比较。我们处理的所有真实图像都使用 e4e 编码器进行了反转[46]。文本驱动的图像操控方法：我们首先在图 9 中比较几种文本驱动的面部图像操控方法。我们比较了我们的潜在映射器方法（第 5 节）、我们的全局方向方法（第 6 节）和 TediGAN [51]。对于 TediGAN，我们使用了作者的官方实现，该实现最近已更新以利用 CLIP 进行图像操控，因此与他们论文中提出的方法有些不同。我们不包括第 4 节中提出的优化方法的结果，因为其对超参数的敏感性使其耗时且不具可扩展性。

![](images/7.jpg)  
using StyleGAN2 pretrained on FFHQ [18]. The inputs are real images, embedded in $\mathcal { W } +$ space using the e4e encoder [46]. The target attribute used in the text prompt is indicated above each column.

![](images/8.jpg)  
F

我们使用三种属性进行比较，这些属性从复杂而具体（例如，“特朗普”），到不太复杂和不太具体（例如，“莫霍克发型”），再到更简单和更常见（例如，“无皱纹”）。复杂的“特朗普”操作涉及多个属性，如金色头发、眯眼、张嘴、略微肿胀的脸以及特朗普的身份。虽然全球潜在方向能够捕捉主要视觉属性，但这些属性并不特定于特朗普，因此未能捕获特定身份。相反，潜在映射器则表现得更为成功。“莫霍克发型”是一个不太复杂的属性，因为它仅涉及头发，而且并不是很具体。因此，我们的方法都能够生成令人满意的操作。由全球方向生成的操作略显不够显著，因为在CLIP空间中的方向是一个平均值。最后，对于“无皱纹”的提示，全球方向成功去除了皱纹，同时保持其他属性大多不受影响，而映射器则失败。我们将此归因于 $\mathcal { W } +$ 的解耦程度较低。我们在另一组属性（“奥巴马”，“愤怒”，“胡须”）上观察到了类似的行为。我们得出结论，对于复杂且具体的属性（尤其是涉及身份的属性），映射器能够生成更好的操作。对于更简单和/或更常见的属性，全球方向足以提供更多解耦的操作。我们注意到，TediGAN在图9中展示的所有三个操作都未能取得成功。

![](images/9.jpg)  
Figure 9.We compare three methods that utilize StyleGAN and CLIP using three different kinds of attrbutes.

其他StyleGAN操作方法：在图10中，我们展示了我们的全局方向方法与几种最先进的StyleGAN图像操作方法的比较：GANSpace [13]、InterFaceGAN [41]和StyleSpace [50]。该比较仅考察所有比较方法都能够操作的属性（性别、灰发和口红），因此未包含我们方法所启用的许多新颖操作。由于这些都是常见属性，我们在这次比较中不包含我们的映射器。根据Wu等人[50]的研究，操作步骤强度被选择为在相应分类器的对数值中引起相同量的变化（在CelebA上进行预训练）。可以看出，在GANSpace [13]中，操作与肤色和光照相互纠缠，而在InterFace-GAN [41]中，身份可能会显著变化（例如操作口红时）。我们的操作与StyleSpace [50]非常相似，仅更改目标属性，而其他所有属性保持不变。

![](images/10.jpg)  
Figure 10. Comparison with state-of-the-art methods using the same amount of manipulation according to a pretrained attribute classifier.

在补充材料中，我们还展示了与最先进的非线性方法 StyleFLow [1] 的比较。尽管 StyleFlow 同时使用了多个属性分类器和回归器（来自微软人脸API），因此能够操控有限的属性集，但我们的方法产生了类似质量的结果。相对而言，我们的方法无需额外的监督。局限性。我们的方法依赖于预训练的 Style-GAN 生成器和 CLIP 模型进行联合语言-视觉嵌入。因此，无法期望操控图像到超出预训练生成器领域的程度（或保持在领域内，但处于生成器覆盖较少的区域）。类似地，映射到 CLIP 空间中图像稀疏区域的文本提示，不能指望产生忠实反映提示语义的视觉操控。我们还观察到，在视觉多样性高的数据集中，进行剧烈操控是困难的。例如，虽然老虎容易转变为狮子（见图1），但当将老虎转变为狼时，我们的成功率较低，具体情况在补充材料中展示。

# 8. 结论

我们介绍了三种新颖的图像操控方法，结合了 StyleGAN 的强大生成能力与 CLIP 的非凡视觉概念编码能力。我们已经证明，这些技术使得各种独特的图像操控成为可能，其中一些是现有依赖于注释数据的方法无法实现的。我们还展示了 CLIP 提供了精细的编辑控制，例如指定所需的发型，而我们的方法能够控制操控强度和解耦程度。总之，我们相信，基于文本的操控是一种强大的图像编辑工具，其能力和重要性只会不断增长。

# References

[1] Rameen Abdal, Peihao Zhu, Niloy Mitra, and Peter Wonka. StyleFlow: attribute-conditioned exploration of StyleGANgenerated images using conditional continuous normalizing flows. arXiv preprint arXiv:2008.02401, 2020. 1, 3, 8, 12, 17   
[2] Yuval Alaluf, Or Patashnik, and Daniel Cohen-Or. Only a matter of style: Age transformation using a style-based regression model. arXiv preprint arXiv:2102.02754, 2021. 1, 3   
[3] T. Brown, B. Mann, Nick Ryder, Melanie Subbiah, J. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, G. Krüger, T. Henighan, R. Child, Aditya Ramesh, D. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, E. Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, J. Clark, Christopher Berner, Sam McCandlish, A. Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. arXiv, abs/2005.14165, 2020. 2   
[4] Yen-Chun Chen, Linjie Li, Licheng Yu, A. E. Kholy, Faisal Ahmed, Zhe Gan, Y. Cheng, and Jing jing Liu. Uniter: Universal image-text representation learning. In ECCV, 2020. 2   
[5] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. StarGAN v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 81888197, 2020. 6, 7, 12, 15, 18   
[6] Edo Collins, Raja Bala, Bob Price, and Sabine Süsstrunk. Editing in style: Uncovering the local semantics of GANs. arXiv preprint arXiv:2004.14367, 2020. 1, 2   
[7] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proc. CVPR, pages 46904699, 2019. 3   
[8] Karan Desai and J. Johnson. VirTex: Learning visual representations from textual annotations. ArXiv, abs/2006.06666, 2020.2   
[9] J. Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.BRT Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT, 2019. 2   
[10] H. Dog, Sm Yu, Chao Wu, and Y. Guo. Smae synthesis via adversarial learning. Proc. ICCV, pages 5707 5715, 2017. 2   
[11] Gabriel Goh, Nick Cammarata, Chelsea Voss, Shan Carter, Michael Petrov, Ludwig Schubert, Alec Radford, and Chris Olah. Multimodal neurons in artificial neural networks. Distill, https://distill.pub/2021/multimodal-neurons/, 2021. 2   
[12] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pages 26722680, 2014. 1   
[13] Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. GANSpace: Discovering interpretable GAN controls. arXiv preprint arXiv:2004.02546, 2020. 1, 3, 8, 12   
[14] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In Proc. ECCV, 2016. 2   
[15] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. arXiv:1710.10196, 2017. 5   
[16] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In Proc. NeurIPS, 2020. 1   
[17] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. arXiv preprint arXiv:2006.06676, 2020. 6, 7, 12, 15, 18   
[18] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proc. CVPR, pages 44014410, 2019. 1, 4, 6, 7   
[19] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In Proc. CVPR, pages 8110 8119, 2020. 1, 3   
[20] J. Y. Koh, Jason Baldridge, H. Lee, and Yinfei Yang. Textto-image generation grounded by fine-grained user attention. arXiv, abs/2011.03775, 2020. 2   
[21] Kathleen M Lewis, Srivatsan Varadharajan, and Ira Kemelmacher-Shlizerman. VOGUE: Try-on by StyleGAN interpolation optimization. arXiv:2101.02285, 2021. 1   
[22] Bowen Li, Xiaojuan Qi, Thomas Lukasiewicz, and Philip HS Torr. ManiGAN: Text-guided image manipulation. In Proc. CVPR, pages 78807889, 2020. 2   
[23] Gen Li, N. Duan, Yuejian Fang, Daxin Jiang, and M. Zhou. Unicoder-VL: A universal encoder for vision and language by cross-modal pre-training. In Proc. AAAI, 2020. 2   
[24] Liunian Harold Li, Mark Yatskar, Da Yin, C. Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. ArXiv, abs/1908.03557, 2019. 2   
[25] Wenbo Li, Pengchuan Zhang, Lei Zhang, Qiuyuan Huang, X. He, Siwei Lyu, and Jianfeng Gao. Object-driven text-toimage synthesis via adversarial training. Proc. CVPR, pages 12166 12174 2010 2   
[26] Xiujun Li, Xi Yin, C. Li, X. Hu, Pengchuan Zhang, Lei Zhang, Longguang Wang, H. Hu, Li Dong, Furu Wei, Yejin Choi, and Jianfeng Gao. Oscar: Object-semantics aligned pre-training for vision-language tasks. In ECCV, 2020. 2   
[27] Yahui Liu, Marco De Nadai, Deng Cai, Huayang Li, Xavier Alameda-Pineda, N. Sebe, and Bruno Lepri. Describe what to change: A text-guided unsupervised image-to-image translation approach. Proceedings of the 28th ACM International Conference on Multimedia, 2020. 2   
[28] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild, 2015. 5   
[29] Jiasen Lu, Dhruv Batra, D. Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In NeurIPS, 2019. 2   
[30] Mehdi Mirza and Simon Osindero. Conditional generative adversarial nets. arXiv:1411.1784, 2014. 2   
[31] Seonghyeon Nam, Yunji Kim, and S. Kim. Text-adaptive generative adversarial networks: Manipulating images with natural language. In NeurIPS, 2018. 2   
[32] Yotam Nitzan, Amit Bermano, Yangyan Li, and Daniel Cohen-Or. Face identity disentanglement via latent space mapping. ACM Trans. Graph., 39(6), Nov. 2020. 3   
[33] Victor Perez. Generating images from prompts using CLIP and StyleGAN. https://towardsdatascience.com/generatingimages-from-prompts-using-clip-and-stylegan-1f9ed495ddda, 2021. 2   
[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. Image, 2:T2, 2021. 2, 6   
[35] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, and Scott Gray. DALL.E: Creating Images from Text. https://openai.com/blog/dall-e/, 2021. 2   
[36] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. arXiv:2102.12092, 2021.2   
[37] S. Reed, Zeynep Akata, Xinchen Yan, L. Logeswaran, B. Schiele, and H. Lee. Generative adversarial text to image synthesis. In ICML, 2016. 2   
[38] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in style: a StyleGAN encoder for image-to-image translation. arXiv:2008.00951, 2020. 3   
[39] Mert Bulent Sariyildiz, Julien Perez, and Diane Larlus. Learning visual representations with caption annotations. arXiv preprint arXiv:2008.01392, 2020. 2   
[40] Yujun Shen, Jinjin Gu, Xiaoou Tang, and Bolei Zhou. Interpreting the latent space of GANs for semantic face editing. In Proc. CVPR, pages 92439252, 2020. 1, 3   
[41] Yujun Shen, Ceyuan Yang, Xiaoou Tang, and Bolei Zhou. InterFaceGAN: interpreting the disentangled face representation learned by GANs. arXiv preprint arXiv:2005.09635,   
[42] Yujun Shen and Bolei Zhou. Closed-form factorization of latent semantics in GANs. arXiv preprint arXiv:2007.06600, 2020.3   
[43] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. VL-BERT: Pre-training of generic visual-linguistic representations. In Proc. ICLR, 2020. 2   
[44] Hao Hao Tan and Mohit Bansal. LXMERT: Learning crossmodality encoder representations from transformers. In EMNLP/IJCNLP, 2019. 2   
[45] Ayush Tewari, Mohamed Elgharib, Gaurav Bharaj, Florian Bernard, Hans-Peter Seidel, Patrick Pérez, Michael Zollhöfer, and Christian Theobalt. StyleRig: Rigging Style-GAN for 3d control over portrait images. arXiv preprint arXiv:2004.00121, 2020. 1, 2   
[46] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for stylegan image manipulation. arXiv preprint arXiv:2102.02766, 2021. 3, 5, 6, 7, 12   
[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. 2   
[48] Andrey Voynov and Artem Babenko. Unsupervised discovery of interpretable directions in the GAN latent space. arXiv preprint arXiv:2002.03754, 2020. 3   
[49] Binxu Wang and Carlos R Ponce. A geometric analysis of deep generative image models and its applications. In Proc. ICLR, 2021. 3   
[50] Zongze Wu, Dani Lischinski, and Eli Shechtman. StyleSpace analysis: Disentangled controls for Style-GAN image generation. arXiv:2011.12799, 2020. 1, 2, 3, 5, 8, 12   
[51] Weihao Xia, Yujiu Yang, Jing-Hao Xue, and Baoyuan Wu. TediGAN: Text-guided diverse face image generation and manipulation. arXiv preprint arXiv: 2012.03308, 2020. 2, 6   
[52] T. Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, and X. He. AttnGAN: Fine-grained text to image generation with attentional generative adversarial networks. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13161324, 2018. 2   
[53] Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015. 6, 7, 15   
[54] Han Zhang, Tao Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N Metaxas. Stack-GAN: Text to photo-realistic image synthesis with stacked generative adversarial networks. In Proc. ICCV, pages 5907 5915, 2017. 2   
[55] Han Zhang, T. Xu, Hongsheng Li, Shaoting Zhang, Xiaogang Wang, Xiaolei Huang, and Dimitris N. Metaxas. Stack-${ \mathrm { G A N } } + +$ :Realistic image synthesis with stacked generative adversarial networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41:19471962, 2019. 2

# A. Latent Mapper  Ablation Study

In this section, we study the importance of various choices in the design of our latent mapper (Section 5).

# A.1. Architecture

The architecture of the mapper is rather simple and with relatively small number of parameters. Moreover, it has negligible effect on the inference time. Yet, it is natural to compare the presented architecture, which consists of three different mapping networks, to an architecture with a single mapping network. Intuitively, using a separate network for each group of style vector entries should better enable changes at several different levels of detail in the image. Indeed, we find that with driving text that requires such changes, e.g. "Donald Trump", a single mapping network does not yield results that are as effective as those produced with three. An example is shown in Figure 11.

Although the full, three network mapper, gives better results for some driving texts, as mentioned in Section 5, we note that not all the three are needed when the manipulation should not affect some attributes. For example, for the hairstyle edits shown in Figure 5, the manipulation should not affect the color scheme of the image. Therefore, we perform these edits when training $M ^ { c }$ and $M ^ { m }$ only, that is, $M _ { t } ( w ) = ( M _ { t } ^ { c } ( w _ { c } ) , M _ { t } ^ { m } ( w _ { m } ) , 0 )$ . We show a comparison in Figure 12. As can be seen, by removing $M _ { f }$ from the architecture, we get slightly better results. Therefore, for the sake of simplicity and generalization of the method, we chose to describe the method with all three networks. In the main paper, the results shown were obtained with all three networks, while here we also show results with only

![](images/11.jpg)  
Figure 11. Comparing our mapper architecture with a simpler architecture that uses a single mapping network. The simpler mapper fails to infer multiple changes correctly. The changes in the expression and in the hair-style are not strong enough to capture the identity of the target individual. On the other hand, there are unnecessary changes in the background color in the second row when using a single network.

![](images/12.jpg)  
Figure 12. Removing $M ^ { f }$ from our full architecture for edits which do not require color scheme manipulation yields slightly better results.

![](images/13.jpg)  
Figure 13. Replacing the CLIP loss with identity loss for the Beyonce edit. The identity loss is computed with respect to an image of Beyonce.

two (without $M _ { f }$ ).

# A.2. Losses

CLIP Loss To show the uniqueness of using a "celeb edit" with CLIP, we perform the following experiment. Instead of using the CLIP loss, we use the identity loss with respect to a single image of the desired celeb. Specifically, we perform this experiment by using an image of Beyonce. The results are shown in Figure 13. As can be seen, CLIP guides the mapper to perform a unique edit which cannot be achieved by simply using a facial recognition network.

ID Loss Here we show that the identity loss is significant for preserving the identity of the person in the input image. When using the default parameter setting of $\lambda _ { \mathrm { L } 2 } = 0 . 8$ with $\lambda _ { \mathrm { I D } } = 0$ (i.e., no identity loss), we observe that the mapper fails to preserve the identity, and introduces large changes. Therefore, we also experiment with $\lambda _ { \mathrm { L } 2 } ~ = ~ 1 . 6$ ,however, this still does not preserve the original identity well enough. The results are shown in Figure 14.

![](images/14.jpg)  
Figure 14. Identity loss ablation study. Under each column we specify $\left( \lambda _ { \mathrm { L } 2 } , \lambda _ { \mathrm { I D } } \right)$ . In the second and the third columns we did not use the identity loss. As can be seen, the identity of individual in the input image is not preserved.

# B. Additional Results

In this section we provide additional results to those presented in the paper. Specifically, we begin with a variety of image manipulations obtained using our latent mapper. All manipulated images are taken from the CelebA-HQ and were inverted by e4e [46]. In Figure 15 we show a large gallery of hair style manipulations. In Figures 16 and 17 we show "celeb" edits, where the input image is manipulated to resemble a certain target celebrity. In Figure 18 we show a variety of expression edits.

Next, Figure 19 shows a variety of edits on non-face datasets, performed along text-driven global latent manipulation directions (Section 6).

Figure 20 shows image manipulations driven by the prompt "a photo of a male face" for different manipulation strengths and disentanglement thresholds. Moving along the global direction, causes the facial features to become more masculine, while steps in the opposite direction yields more feminine features. The effect becomes stronger as the strength $\alpha$ increases. When the disentanglement threshold $\beta$ is high, only the facial features are affected, and as $\beta$ is lowered, additional correlated attributes, such as hair length and facial hair are affected as well.

In Figure 21, we show another comparison between our global direction method and several state-of-the-art Style-GAN image manipulation methods: GANSpace [13], Inter-

FaceGAN [41], and StyleSpace [50]. The comparison only examines the attributes which all of the compared methods are able to manipulate (Gender, Grey hair, and Lipstick), and thus it does not include the many novel manipulations enabled by our approach. Following Wu et al. [50], the manipulation step strength is chosen such that it induces the same amount of change in the logit value of the corresponding classifiers (pretrained on CelebA). It may be seen that in GANSpace [13] manipulation is entangled with skin color and lighting, while in InterFaceGAN [41] the identity may change significantly (when manipulating Lipstick). Our manipulation is very similar to StyleSpace [50], which only changes the target attribute, while all other attributes remain the same.

Figure 22 shows a comparison between StyleFlow [1] and our global directions method. It may be seen that our method is able to produce results of comparable visual quality, despite the fact that StyleFlow requires the simultaneous use of several attribute classifiers and regressors (from the Microsoft face API), and is thus able to manipulate a limited set of attributes. In contrast, our method required no extra supervision to produce these and all of the other manipulations demonstrated in this work.

Figure 23 shows an additional comparison between textdriven manipulation using our global directions method and our latent mapper. Our observations are similar to the ones we made regarding Figure 10 in the main paper.

Finally, Figure 24 demonstrates that drastic manipulations in visually diverse datasets are sometimes difficult to achieve using our global directions. Here we use StyleGAN-ada [17] pretrained on AFHQ wild [5], which contains wolves, lions, tigers and foxes. There is a smaller domain gap between tigers and lions, which mainly involves color and texture transformations. However, there is a larger domain gap between tigers and wolves, which, in addition to color and texture transformations, also involves more drastic shape deformations. This figure demonstrates that our global directions method is more successful in transforming tigers into lions, while failing in some cases to transform tigers to wolves.

# C. Video

We show examples of interactive text-driven image manipulation in our supplementary video. We use a simple heuristic method to determine the initial disentanglement threshold $( \beta )$ . The threshold is chosen such that $k$ channels will be active. For real face manipulation, we set the initial strength to $\alpha = 3$ and the disentanglement threshold so that $k = 2 0$ . For real car manipulation, we set the initial values to $\alpha = 3$ and $k = 1 0 0$ .For generated cat manipulation, we set the initial values to $\alpha = 7$ and $k = 1 0 0$ .

![](images/15.jpg)

![](images/16.jpg)  
Figure 16. Celeb edits performed by the latent mapper.

![](images/17.jpg)  
Input Trump Mark Zuckerberg Johnny Depp Figure 17. Celeb edits performed by the latent mapper.

![](images/18.jpg)  
Figure 18. Expression edits performed by the latent mapper.

![](images/19.jpg)  
indicated above each column.

![](images/20.jpg)  
opposite direction yields more feminine features. The effect becomes stronger as the strength $\alpha$ increases. When the disentanglement threshold $\beta$ is high, only the facial features are affected, and as $\beta$ is lowered, additional correlated attributes, such as hair length and facial hair are affected as well.

![](images/21.jpg)

![](images/22.jpg)  
a limited set of attributes. In contrast, our method requires no extra supervision.

![](images/23.jpg)  
Figure.We compare our gloal directions with our latent mapper using three different kinds o attrbutes.

![](images/24.jpg)  
$" + "$ and $\cdot _ { + + } ,$ indicate medium and strong manipulation strength, respectively.