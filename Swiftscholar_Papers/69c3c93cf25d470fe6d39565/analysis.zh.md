# 1. 论文基本信息
## 1.1. 标题
论文标题为《Sparser, Faster, Lighter Transformer Language Models》，核心主题是通过挖掘Transformer大语言模型前馈层的非结构化稀疏性，设计适配现代GPU的稀疏存储格式和专用CUDA内核，在几乎不损失模型性能的前提下，同时提升推理和训练阶段的吞吐量、降低内存占用与能耗。
## 1.2. 作者
作者团队包括：Edoardo Cetin*、Stefano Peluchetti、Emilio Castillo*、Akira Naruse、Mana Murakami、Llion Jones。隶属机构为：1. Sakana AI（日本AI创业公司，专注于大模型效率与多模态研究）；2. 工业界硬件研究团队（从作者研究背景推断为GPU厂商相关团队）。
## 1.3. 发表状态
2026年3月24日发布于arXiv预印本平台，尚未正式发表于学术期刊或会议。
## 1.4. 发表年份
2026年
## 1.5. 摘要
本文针对大语言模型（LLM）缩放带来的极高计算成本问题，聚焦占模型大部分参数和计算量的前馈层，利用其非结构化稀疏性降低开销。核心创新包括：
1. 提出新的稀疏打包格式TwELL，以及配套的CUDA内核，可无缝适配现代GPU的优化执行流水线，同时支持推理和训练阶段的高效稀疏计算；
2. 证明仅用简单的L1正则即可让前馈层达到99%以上的稀疏度，且下游性能损失可忽略；
3. 该方案带来的吞吐量、能效、内存收益随模型规模增大而提升。
   作者将开源所有代码和内核，推动稀疏性成为大模型效率优化的实用方向。
## 1.6. 原文链接
- 预印本主页：images/1.jpg)
*该图像是一个图表，比较了密集矩阵与三种稀疏格式（a. ELL，b. TwELL，c. Hybrid），展示了它们在值、索引和非零计数方面的不同布局和表示。这样的比较有助于理解在大语言模型推理和训练中的应用效果。*

TwELL的核心设计是不对整行进行打包，而是将前馈层的隐藏维度划分为多个大小为$T$的水平分块（Tile），每个分块内部独立使用类似ELL的格式打包非零值、对应索引，以及该分块内的非零值计数。这样的设计有以下优势：
1. 可以直接在稠密矩阵乘法（GEMM）内核的尾声生成TwELL格式的输出，不需要额外的格式转换内核，也不需要跨CTA同步，因为每个分块的计算刚好对应一个CTA的输出，打包完全在CTA内部完成。
2. 相比传统ELL格式，TwELL对各行非零数的不均匀性更鲁棒，不需要将行长度填充到全局最大非零数，仅需要填充到分块内的最大非零数，存储空间浪费更少。
   TwELL存储的三个核心部分：
- $h_\nu$：存储所有分块内的非零值，维度为$M \times N/C$，$C$是压缩系数，设置为大于等于任意分块内的最大非零数，避免溢出。
- $h_I$：存储所有非零值对应的列索引，维度和$h_\nu$相同。
- $h_{nz}$：存储每个分块内的非零值计数，维度为$M \times N_T$，$N_T = \lceil N/T \rceil$是总分块数。
## 4.4. 推理内核设计
推理阶段仅需要两个内核即可完成整个前馈层的计算，大幅减少全局内存访问，降低开销。
### 4.4.1. 门投影内核（生成TwELL格式）
第一个内核执行门投影的矩阵乘法$h_g = \text{ReLU}(xW_g)$，同时直接将输出打包为TwELL格式，算法伪代码如下（原文Algorithm 1）：
$$
Algorithm 1 | Algorithmic description of gate projection with our matmul kernel with TwELL storage
Parameters: Tile sizes `T _ { n } , T _ { m }` , compression ratio C   
Input: Dense $\boldsymbol { x } \in \mathbb { R } ^ { M \times K }$ , $W _ { g } \in \mathbb { R } ^ { K \times N }$ ,   
Output: Sparse $h _ { \nu } \in \mathbb { R } ^ { M \times N / C }$ , $h _ { I } \in \mathbb { N } ^ { M \times N / C }$ , $h _ { n z } \in \mathbb { N } ^ { M \times N _ { T } }$   
for all tiles starting at `( m _ { 0 } , n _ { 0 } )` in parallel across CTAs do   
    $S \gets x [ m _ { 0 } : m _ { 0 } + T _ { m } , : ] @ W _ { g } [ : , n _ { 0 } : n _ { 0 } + T _ { n } ]$   
    for $r \gets 0 \ldots T _ { m } { - } 1$ do   
        $m \gets m _ { 0 } + r$ {global row index}   
        $z \gets 0$ {running count of non-zeros in tile}   
        for $c \gets 0 \ldots T _ { n } { - } 1$ do   
            if $( S [ r , c ] > 0 )$ then   
                $n \gets n _ { 0 } / C + z$ {global TwELL column index}   
                $h _ { I } [ m , n ] \gets n _ { 0 } + c$ {store non-zero index}   
                $h _ { \nu } [ m , n ] \gets S [ r , c ]$ {store non-zero value}   
                $z \gets z + 1$ {increment non-zero count}   
            end if   
        end for   
        $h _ { n z } [ m , n _ { 0 } / T _ { n } ] \gets z$ {store final count of non-zeros}   
    end for   
end for
$$
算法解释：
1. 所有输出分块并行处理，每个CTA负责计算一个大小为$T_m \times T_n$的输出分块$S$。
2. 对分块内的每一行，遍历所有元素，筛选出大于0的非零值，将其值和索引存储到$h_\nu$和$h_I$的对应位置，同时统计该分块内的非零值数量，存储到$h_{nz}$。
3. 整个过程仅需要CTA内部的warp级同步，不需要跨CTA同步，也不需要额外的全局内存访问，直接在GEMM计算完成后完成打包，开销极低。
### 4.4.2. 融合上投影+下投影内核
第二个内核利用TwELL格式的稀疏信息，融合上投影和下投影的计算，不需要存储中间的上投影激活值，大幅减少全局内存访问，算法伪代码如下（原文Algorithm 2）：
$$
Algorithm 2 | Algorithmic description of fused up and down projections from gate activations in the TwELL format
Parameters: Tile size `T _ { n }` , compression ratio C   
Input: Sparse $h _ { \nu } \in \mathbb { R } ^ { M \times N / C }$ , $\bar { h _ { I } } \in \mathbb { N } ^ { M \times N / C }$ , $h _ { n z } \in \mathbb { N } ^ { M \times N _ { T } }$ ; dense $\boldsymbol { x } \in \mathbb { R } ^ { M \times K }$ , $W _ { u } \in \mathbb { R } ^ { K \times N }$ , $W _ { d } \in \mathbb { R } ^ { N \times K }$   
Output: Dense $y \in \mathbb { R } ^ { M \times K }$   
for all $m \in \pi ( 0 . . M { - } 1 )$ in parallel across CTAs do   
    $x _ { m } \gets x [ m , : ]$ ; $y _ { m } \gets 0$   
    for $t \gets 0 \ldots N _ { T } - 1$ do   
        $z \gets h _ { n z } [ m , t ]$   
        for $c \gets 0 \ldots z - 1$ do   
            $n \gets h _ { I } [ m , t \times T _ { n } / C + c ]$ {non-zero column index}   
            `w _ { u } = W _ { u } [ : , n ]` {$n$-th column of `W _ { u }`}   
            $u \gets ( x _ { m } \cdot w _ { u } )$ {sparse `h _ { u } [ m , n ]` element}   
            $w _ { d } \gets W _ { d } [ n , : ]$ {$n$-th row of `W _ { d }`}   
            $y _ { m } \gets y _ { m } + \left( h _ { \nu } [ m , t \times T _ { n } / C + c ] \times u \right) w _ { d }$   
        end for   
    end for   
    $y [ m , : ] \gets y _ { m }$   
end for
$$
算法解释：
1. 每个CTA负责处理一个输入token（一行输入$x_m$），最小化CTA大小以提升并行度和L2缓存命中率。
2. 首先静态遍历所有分块，然后动态遍历每个分块内的非零值，仅需要处理非零的门激活对应的上投影列和下投影行，跳过所有零激活对应的计算。
3. 上投影的结果$u$不需要存储到全局内存，直接在寄存器中与门激活值相乘，再乘以下投影的对应行，累加到输出$y_m$中，大幅减少全局内存访问。
   整个前馈层的计算仅需要两次内核启动，不需要存储中间激活，充分利用了稀疏性减少计算量，同时避免了传统稀疏方案的格式转换开销。
## 4.5. 训练内核设计
训练阶段需要存储中间激活用于反向传播，内存是主要瓶颈，本文设计了混合稀疏格式和配套的训练内核，解决非零值分布不均匀的问题，同时降低内存开销。
### 4.5.1. 混合稀疏格式
训练阶段的激活稀疏性存在高度不均匀性：不同token的非零值数量差异很大，最大非零值数量可能比平均值高几个数量级，如果直接使用TwELL或ELL格式，需要按最大非零值数量分配存储空间，会浪费大量内存。
为了解决这个问题，本文提出混合稀疏格式（见原文Figure 1c），根据每行的非零值数量，将行分为两类：
1. 非零值数量低于阈值的行：用紧凑的ELL格式存储，降低内存开销。
2. 非零值数量超过阈值的行：用稠密格式存储，避免ELL格式的填充浪费。
   混合格式存储的核心部分：
- $h_g^s$：稀疏存储的非溢出行的非零值，维度$M^s \times N_{\hat{n}z}$，$M^s$是稀疏行的数量，$N_{\hat{n}z}$是ELL格式的最大行长度。
- $h_g^d$：稠密存储的溢出行的激活值，维度$M^d \times N$，$M^d$是稠密行的数量。
- $h_I$：稀疏行的非零值索引，维度和$h_g^s$相同。
- $h_b$：二进制向量，标识每行是稀疏存储还是稠密存储，维度$M$。
  实践中，将$N_{\hat{n}z}$设置为比平均非零值高一个数量级，即可让溢出的稠密行占比极低，同时大幅降低整体内存开销。
### 4.5.2. 混合格式计算内核
本文设计了适配混合格式的矩阵乘法内核，同时支持稀疏行和稠密行的高效计算，算法伪代码如下（原文Algorithm 3）：
$$
Algorithm 3 | Algorithmic description of matmul from input activations in the hybrid format
Input: `T _ { m } , T _ { k } , T _ { n }` , $h : = ( h ^ { s } , h ^ { d } , h _ { I } , h _ { b } )$ , $W \in \mathbb { R } ^ { K \times N }$   
Output: $y \in \mathbb { R } ^ { M \times N }$   
$\pi _ { s } \gets \{ m : h _ { b } [ m ] = 0 \}$ ; $\pi _ { d } \gets \{ m : h _ { b } [ m ] = 1 \}$   
for all $m _ { s } \in 0 . . M ^ { s } - 1$ in parallel do   
    $m \gets \pi _ { s } [ m _ { s } ]$ {global row index}   
    $y _ { m } \gets 0$ {row accumulator}   
    for $j \gets 0 \ldots N _ { \hat { n } z } - 1$ do   
        $n \gets h _ { I } [ m _ { s } , j ]$ {non-zero column index}   
        $\nu \gets h ^ { s } [ m _ { s } , j ]$ {sparse value}   
        $y _ { m } \gets y _ { m } + \nu \cdot W [ n , : ]$ {sparse row update}   
    end for   
    $y [ m , : ] \gets y _ { m }$   
end for   
for all tiles starting at `( m _ { 0 } , n _ { 0 } )` in parallel do   
    $S \gets h ^ { d } [ m _ { 0 } : m _ { 0 } { + } T _ { m } , : ] @ W [:, n _ { 0 } { : } n _ { 0 } { + } T _ { n } ]$   
    $y \big [ \pi _ { d } \big [ m _ { 0 } { : } m _ { 0 } { + } T _ { m } \big ]$ , $n _ { 0 } : n _ { 0 } + T _ { n } ] \gets S$   
end for
$$
算法解释：
1. 首先将输入行分为稀疏行集合$\pi_s$和稠密行集合$\pi_d$。
2. 对稀疏行，使用类似ELL的稀疏内核处理，仅计算非零值对应的部分，跳过零值。
3. 对稠密行，使用高度优化的Tensor Core稠密矩阵乘法内核处理，保证效率。
   这种混合设计既利用了稀疏性减少计算和内存开销，又对非零值分布不均匀的情况鲁棒，避免了传统稀疏内核的性能波动。
### 4.5.3. 反向传播支持
训练阶段的反向传播也完全利用稀疏性，不需要执行任何稠密计算：
1. 首先利用存储的混合格式激活模式，通过混合格式内核计算输出梯度$\nabla y$对应的隐藏层梯度$\nabla h = \nabla y W_d^T$。
2. 再通过逐元素乘法得到上投影和门投影的梯度：$\nabla h_u = \nabla h \odot h_g$，$\nabla h_g = \nabla h \odot h_u$。
3. 最后通过混合格式内核计算输入和权重的梯度：
   $$
\nabla W _ { u } = x ^ { \top } \nabla h _ { u } , \quad \nabla W _ { g } = x ^ { \top } \nabla h _ { g } , \quad \nabla W _ { d } = h ^ { \top } \nabla y ,
$$
$$
\nabla x = \nabla h _ { u } W _ { u } ^ { \top } + \nabla h _ { g } W _ { g } ^ { \top } .
$$
整个反向传播过程仅处理非零值，同时稀疏的中间激活大幅降低了内存占用，允许使用更大的批次大小，进一步提升训练吞吐量。
---
# 5. 实验设置
## 5.1. 数据集
### 训练数据集
实验使用的训练数据集是FineWeb-Edu的去重版本，FineWeb是2024年发布的大规模高质量网页文本数据集，包含超过10万亿token的文本，广泛用于大模型预训练。训练token数量符合Chinchilla最优规则：0.5B参数模型训练10B token，1B参数模型训练20B token，1.5B参数模型训练30B token，2B参数模型训练40B token。
### 下游评估数据集
下游任务评估使用7个常用的推理和常识推理基准数据集：
1. ARC（AI2 Reasoning Challenge）：小学科学选择题，分为Easy和Challenge两个子集，后者设计用来击败简单的检索和共现基线。
2. HellaSwag：常识句子补全任务，专门设计用来对抗大模型的简单模式匹配。
3. OpenBookQA：科学知识问答任务，基于一套基础科学知识点出题。
4. PIQA：物理常识推理任务，考察日常物理场景的推理能力。
5. WinoGrande：大规模Winograd模式挑战，考察指代消解能力。
6. CommonsenseQA：通用常识推理任务。
## 5.2. 评估指标
本文使用的评估指标如下，每个指标按要求给出定义、公式和符号解释：
### 5.2.1. 交叉熵损失（Cross-Entropy Loss）
- **概念定义**：衡量大模型预训练时预测下一个token的准确率，值越低表示模型的语言建模能力越好。
- **公式**：
  $$
\mathcal{L}_{CE} = -\frac{1}{M} \sum_{i=1}^M \sum_{v=1}^V y_{i,v} \log p_{i,v}
$$
- **符号解释**：$M$是批次中的token数量，$V$是词表大小，$y_{i,v}$是真实标签（one-hot向量，正确token的位置为1，其余为0），$p_{i,v}$是模型预测的第$i$个token为词表中第$v$个词的概率。
### 5.2.2. 下游任务准确率（Accuracy）
- **概念定义**：衡量模型在下游多选任务上的正确率，值越高表示模型的推理能力越好。
- **公式**：
  $$
\text{Accuracy} = \frac{\text{正确预测的样本数}}{\text{总样本数}}
$$
### 5.2.3. 吞吐量（Throughput）
- **概念定义**：衡量单位时间内模型可以处理的输入token数量，值越高表示速度越快。本文中分为推理前向吞吐量（单位：input tokens/ms）和训练吞吐量（单位：input tokens/ms）。
### 5.2.4. 每token能耗（Energy per Token）
- **概念定义**：衡量处理每个token消耗的能量，单位为mJ，值越低表示能效越高。
### 5.2.5. 峰值内存（Peak Memory）
- **概念定义**：训练或推理过程中GPU占用的最大显存，单位为GB，值越低表示内存开销越小。
## 5.3. 对比基线
本文的对比基线是同参数规模的稠密Transformer大模型，包括：
1. 门控FFN+ReLU激活的稠密模型：和稀疏模型的架构仅相差L1正则，用于控制变量比较稀疏性的收益。
2. 门控FFN+SiLU激活的稠密模型：当前主流大模型的标准配置，用于比较稀疏方案和现有主流模型的性能和效率差异。
3. 非门控FFN+ReLU激活的稠密模型：原始Transformer的前馈层配置，用于验证方案对不同架构的兼容性。
   ---
# 6. 实验结果与分析
## 6.1. 核心结果分析
### 6.1.1. 不同L1正则下的性能与稀疏度
首先在1.5B参数模型上测试不同L1正则系数的效果，训练曲线如下图（原文Figure 2）所示：

![Figure 2 | Training curves of LLMs across L1 regularization levels.](images/2.jpg)
*该图像是一个图表，展示了不同 L1 正则化水平下语言模型的训练曲线。随着训练 token 数量的增加，交叉熵损失逐渐降低，各条曲线对应不同的 L1 正则化值，从 $L_1 = 0$ 到 $L_1 = 10^{-4}$。*

可以看到，不同L1正则系数的训练曲线几乎重合，仅在最高正则系数$L_1=1e-4$时交叉熵损失略有上升。
下游任务准确率和稀疏度的结果如下图（原文Figure 3）所示：

![Figure 3 | Downstream accuracy and sparsity statistics of LLMs across L1 regularization levels.](images/3.jpg)
*该图像是图表，展示了不同 L1 正则化系数下，LLM 的平均任务准确率和非零参数数量的关系。随着 L1 系数变化，平均准确率呈现出一定的波动，同时非零参数数量也在显著变化。图中蓝色条表示平均任务准确率，而橙色条表示非零参数数量的对数值。*

结果显示：
1. 当$L_1 \leq 3e-5$时，下游任务准确率几乎没有下降，稀疏度已经达到99%以上，平均每个token仅激活不到30个隐藏神经元（总隐藏维度为5632，稀疏度>99.4%）。
2. 当$L_1 > 3e-5$时，稀疏度继续提升，但下游任务准确率开始明显下降。
   因此本文推荐的最优正则系数为$L_1=2e-5$，可以在几乎不损失性能的前提下获得极高的稀疏度。
### 6.1.2. 推理效率收益
使用推荐的$L_1=2e-5$时，推理阶段的速度提升和能耗节省如下图（原文Figure 4）所示：

![Figure 4 | Forward pass speedups and energy savings from our sparse LLM inference kernels across L1 regularization levels.](images/4.jpg)
*该图像是一个柱状图，展示了在不同 L1 正则化系数下，稀疏 LLM 推理内核相比于稠密模型的推理加速和能源节省百分比。随着 L1 系数的降低，推理速度提升和能源节省都显著增加，最高可达 30% 和 25%。*

可以看到，稀疏模型的前向推理速度提升最高达30%，同时能耗降低最高达17%，且稀疏度越高，收益越大。
### 6.1.3. 训练效率收益
训练阶段的速度提升和峰值内存降低如下图（原文Figure 5）所示：

![Figure 5 | Training speedups and peak memory reduction from our sparse LLM training kernels across L1 regularization levels.](images/5.jpg)
*该图像是一个图表，展示了L1正则化水平对训练速度提升和内存减少的影响。横轴为L1系数，纵轴为百分比，分别表示与密集模型的训练速度提升和内存减少的比例。可以看出，随着L1系数的变化，训练速度和内存使用量有显著变化。*

结果显示，训练速度提升最高达24%，峰值内存降低最高达24%，大幅降低了大模型训练的硬件门槛。
### 6.1.4. 不同模型规模的结果
不同参数规模的稀疏模型和稠密基线的对比如下表（原文Table 1）所示：

<table>
<thead>
<tr>
<th rowspan="2">Model scale</th>
<th rowspan="2">Sparse</th>
<th rowspan="2">Mean task accuracy</th>
<th colspan="2">Forward execution (input tokens/ms)</th>
<th colspan="2">Energy per token (mJ)</th>
<th colspan="2">Training step (input token/ms)</th>
<th colspan="2">Peak memory (GB)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">0.5B params<br>10B tokens</td>
<td>✗</td>
<td>40.4%</td>
<td>410</td>
<td>(0.0%)</td>
<td>1.63</td>
<td>(0.0%)</td>
<td>97.3</td>
<td>(0.0%)</td>
<td>26.2</td>
<td>(0.0%)</td>
</tr>
<tr>
<td>✓</td>
<td>40.4%</td>
<td>480</td>
<td>(+17.0%)</td>
<td>1.43</td>
<td>(-11.8%)</td>
<td>95.9</td>
<td>(-1.5%)</td>
<td>21.2</td>
<td>(-19.2%)</td>
</tr>
<tr>
<td rowspan="2">1B params<br>20B tokens</td>
<td>✗</td>
<td>44.6%</td>
<td>185</td>
<td>(0.0%)</td>
<td>3.71</td>
<td>(0.0%)</td>
<td>48.6</td>
<td>(0.0%)</td>
<td>44.5</td>
<td>(0.0%)</td>
</tr>
<tr>
<td>✓</td>
<td>44.7%</td>
<td>219</td>
<td>(+18.1%)</td>
<td>3.17</td>
<td>(-14.6%)</td>
<td>52.1</td>
<td>(+7.1%)</td>
<td>33.1</td>
<td>(-25.5%)</td>
</tr>
<tr>
<td rowspan="2">1.5B params<br>30B tokens</td>
<td>✗</td>
<td>46.4%</td>
<td>119</td>
<td>(0.0%)</td>
<td>5.73</td>
<td>(0.0%)</td>
<td>31.8</td>
<td>(0.0%)</td>
<td>62.8</td>
<td>(0.0%)</td>
</tr>
<tr>
<td>✓</td>
<td>46.2%</td>
<td>141</td>
<td>(+18.8%)</td>
<td>4.87</td>
<td>(-15.0%)</td>
<td>35.5</td>
<td>(+11.6%)</td>
<td>45.1</td>
<td>(-28.1%)</td>
</tr>
<tr>
<td rowspan="2">2B params<br>40B tokens</td>
<td>✗</td>
<td>49.1%</td>
<td>87.8</td>
<td>(0.0%)</td>
<td>7.85</td>
<td>(0.0%)</td>
<td>22.4</td>
<td>(0.0%)</td>
<td>46.7</td>
<td>(0.0%)</td>
</tr>
<tr>
<td>✓</td>
<td>48.8%</td>
<td>106</td>
<td>(+20.5%)</td>
<td>6.51</td>
<td>(-17.0%)</td>
<td>27.3</td>
<td>(+21.9%)</td>
<td>57.1</td>
<td>(+22.3%)</td>
</tr>
</tbody>
</table>

结果显示：
1. 所有规模的稀疏模型的下游任务准确率和稠密基线几乎一致，差异在随机波动范围内。
2. 效率收益随模型规模增大而提升：2B参数模型的推理速度提升20.5%，训练速度提升21.9%，能耗降低17%，内存降低22.3%，符合大模型的缩放趋势。
### 6.1.5. 稀疏性分布分析
稀疏大模型的非零激活分布呈现出明显的规律：
1. **层间分布**：不同层的稀疏度差异很大，前两层稀疏度最高，中间层稀疏度最低，和模型的推理过程匹配：前几层负责提取通用特征，中间层负责知识检索和推理，需要激活更多神经元。每层的速度提升和该层的平均非零值数量呈极强的负相关，皮尔逊系数达-0.996，如下图（原文Figure 6）所示：

   ![Figure 6 | Sparsity statistics and speedup contributions across different layers of our sparse LLMs.](images/6.jpg)
   *该图像是图表，展示了稀疏 LLM 各层的非零参数数量以及速度提升贡献。图中以对数尺度显示了每层的最大非零值、平均非零值和速度贡献比例，揭示了不同层次的稀疏性对模型效率的影响。*

2. **token分布**：不同token的非零激活数量差异很大，常见的停用词、链接片段等token的激活非常少，而包含关键信息的名词、动词等token的激活数量更多；序列开头的token激活数量更多，随序列位置增加呈指数下降，如下图（原文Figure 7）所示：

   ![Figure 7 | Sparsity statistics across LLM input tokens and positions.](images/7.jpg)
   *该图像是图表，展示了不同 LLM 输入标记和位置的稀疏性统计信息。图表左侧呈现了各个标记的非零值数量，右侧则显示了不同位置的非零值数量，底部部分展示了随标记位置变化的非零值数量的变化趋势。*

这说明大模型会自动将计算资源集中在信息含量更高的token和位置，稀疏性不仅提升了效率，还为模型行为提供了可解释性的视角。
## 6.2. 消融实验与参数分析
### 6.2.1. 激活函数的影响
对比ReLU和SiLU激活的稠密模型，以及ReLU激活的稀疏模型，结果如下：

| Activation | Sparse | Mean task accuracy | Forward speed | Energy per token |
| --- | --- | --- | --- | --- |
| ReLU | ✗ | 46.4% | 100% | 100% |
| SiLU | ✗ | 47.1% | 99.5% | 100.1% |
| ReLU | ✓ | 46.2% | 117.9% | 87.9% |

可以看到，SiLU激活的准确率仅比ReLU高0.7%，但无法利用稀疏性，效率远低于稀疏ReLU模型，说明稀疏性带来的效率收益远超过SiLU的微小性能提升。
### 6.2.2. 前馈层架构的影响
对比门控FFN和非门控FFN的稀疏模型，结果如下：

| Gated | L1 coeff | Mean task accuracy | Forward speedup |
| --- | --- | --- | --- |
| ✓ | 2e-5 | 46.2% | +17.9% |
| ✓ | 3e-5 | 44.8% | +25.5% |
| ✗ | 2e-5 | 46.4% | +11.2% |
| ✗ | 3e-5 | 44.7% | +13.1% |

可以看到，门控FFN的稀疏收益更高，因为门控结构的稀疏模式更适合本文的融合内核设计，但非门控FFN也能获得明显的效率提升，说明本文的方案具有广泛的兼容性。
### 6.2.3. 死神经元缓解策略
稀疏训练中会出现死神经元的问题，即部分神经元永远不会被激活，浪费模型容量。本文测试了两种缓解策略：
1. **稀疏度预热**：前5000步不使用L1正则，之后线性提升正则系数。
2. **死神经元重初始化**：每次训练步后对死神经元的权重重新注入噪声。
   结果如下：

   | Strategy | Mean task accuracy | Forward speedup | Dead neuron ratio |
   | --- | --- | --- | --- |
   | 无（基线） | 46.2% | +17.9% | ~30% |
   | 稀疏度预热 | 45.9% | +1.9% | ~0% |
   | 死神经元重初始化 | 46.6% | +19.1% | ~0% |

可以看到，死神经元重初始化策略不仅缓解了死神经元问题，还小幅提升了准确率和速度，是更优的方案；而稀疏度预热策略会导致稀疏度大幅下降，收益几乎消失。
### 6.2.4. 不同硬件的收益
对比本文的方案在H100和RTX6000 GPU上的训练速度提升，结果如下图（原文Figure 12）所示：

![Figure 12 | Training speedups from our sparse LLM training kernels across L1 regularization levels for both H100 and RTX6000 devices.](images/11.jpg)
*该图像是图表，展示了在不同 L1 正则化系数下，H100 和 RTX6000 设备的训练速度提升百分比。数据表明，在 L1 系数增大时，H100 的训练加速效果显著，尤其是在较小的 L1 系数下。*

可以看到，在RTX6000上的收益更高，最高达30%以上，因为RTX6000的Tensor Core性能更弱，显存带宽更低，稀疏计算带来的收益更加明显，说明本文的方案可以降低大模型对高端GPU的依赖，让消费级GPU也能高效运行和训练大模型。
---
# 7. 总结与思考
## 7.1. 结论总结
本文提出了一套实用的非结构化稀疏大模型落地方案，核心结论包括：
1. 仅用温和的L1正则和ReLU激活，即可让大模型前馈层达到99%以上的稀疏度，且下游任务性能损失可忽略。
2. 新设计的TwELL稀疏格式和配套CUDA内核，解决了传统稀疏方案和GPU架构不匹配的问题，同时实现了推理和训练阶段的效率提升，收益随模型规模增大而提升：2B参数模型的推理速度提升20.5%，训练速度提升21.9%，同时能耗降低17%，内存占用降低22%。
3. 该方案不需要修改主流大模型的架构，仅需要极小的训练流程修改，兼容现有硬件和软件栈，具有很高的实用性。
   本文证明了稀疏性可以成为大模型效率优化的新维度，为缓解大模型缩放带来的算力和成本压力提供了可行的方向。
## 7.2. 局限性与未来工作
作者指出的局限性和未来工作方向：
1. 目前仅对前馈层进行了稀疏优化，未来可以扩展到注意力层，进一步提升效率。
2. 目前需要使用ReLU激活，未来可以研究如何让SiLU等更优的激活函数也能支持高稀疏度，或者通过微调将现有预训练的SiLU大模型转换为稀疏模型，让现有成熟模型也能享受稀疏性的收益。
3. 死神经元问题还可以进一步优化，研究更好的稀疏训练策略，在更高稀疏度下也能保持模型性能。
4. 目前的内核仅支持NVIDIA GPU，未来可以扩展到其他硬件平台（如AMD GPU、AI加速器等）。
## 7.3. 个人启发与批判
### 启发
1. **软硬件协同是大模型效率落地的关键**：单纯的算法创新（比如证明稀疏性的存在）很难带来实际的收益，必须结合硬件的特性设计配套的存储格式和内核，才能将理论上的计算量减少转化为实际的速度提升。
2. **稀疏性是大模型缩放的新方向**：过去大模型的效率优化大多聚焦于量化、蒸馏等方向，本文证明稀疏性在几乎不损失性能的前提下可以带来明显的收益，且随规模增大收益更高，和当前大模型的缩放趋势高度契合，未来可能成为大模型的标准配置。
3. **降低大模型的硬件门槛**：本文的方案在消费级GPU上的收益更高，让中小团队也能高效训练和部署大模型，推动大模型的普惠化。
### 潜在问题与改进方向
1. **正则系数的调参成本**：不同模型规模、不同领域的模型可能需要不同的L1正则系数，未来可以研究自动调整正则系数的方法，避免手动调参的开销。
2. **注意力层的稀疏扩展**：注意力层的计算和内存开销随着序列长度的提升快速增长，未来如果能将类似的稀疏格式和内核扩展到注意力层，可能带来更大的收益，尤其是长序列大模型。
3. **多硬件支持**：目前的内核仅支持NVIDIA GPU，未来可以适配开源的编程模型（如OpenCL、SYCL），让更多硬件平台都能享受稀疏性的收益。
4. **多技术融合**：本文的方案可以和量化、蒸馏、MoE等其他效率技术结合，可能带来叠加的收益，进一步提升大模型的效率。
   总体来说，本文是稀疏大模型落地的重要突破，兼具学术创新性和工业实用性，可能对未来大模型的设计和部署产生深远的影响。