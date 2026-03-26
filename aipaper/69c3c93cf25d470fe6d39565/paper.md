# Sparser, Faster, Lighter Transformer Language Models

Edoardo Cetin\*1, Stefano Peluchett1, Emilio Castll\*2, Akira Naruse², Mana Murakami2 and Llion Jones1 1SakanaI, , Core contruor

Scaling autoregressive large language models (LLMs) has driven unprecedented progress but comes with vast computational costs. In this work, we tackle these costs by leveraging unstructured sparsity within an LLM's feedforward layers, the components accounting for most of the model parameters and execution FLOPs. To achieve this, we introduce a new sparse packing format and a set of CUDA kernels designed to seamlessly integrate with the optimized execution pipelines of modern GPUs, enabling efficient sparse computation during LLM inference and training. To substantiate our gains, we provide a quantitative study of LLM sparsity, demonstrating that simple L1 regularization can induce over $9 9 \%$ sparsity with negligible impact on downstream performance. When paired with our kernels, we show that these sparsity levels translate into substantial throughput, energy efficiency, and memory usage benefits that increase with model scale. We will release all code and kernels under an open-source license to promote adoption and accelerate research toward establishing sparsity as a practical axis for improving the efficiency and scalability of modern foundation models. $\pmb { \bigcirc }$ Code: github.com/SakanaAI/sparser-faster-llms

# 1. Introduction

Large Language Models (LLMs) have revolutionized natural language processing, demonstrating unprecedented capabilities in text generation, reasoning, and knowledge retrieval (OpenAI, 2023; Team et al., 2023). The core component driving these advancements has been massive computational investments into scaling the seminal Transformer architecture, with current LLMs reaching hundreds of billions of parameters Brown et al. (2020); Radford et al. (2019); Vaswani et al. (2017). However, with increasingly larger models requiring vast computational resources for both inference and training, there is a growing need for fundamental efficiency improvements to ensure the present and future sustainability of the field (Luccioni et al., 2023; Schwartz et al., 2020). One seminal avenue for improving the efficiency of machine learning models is sparsity (Han et al., 2015; Hoefler et al., 2021; Le-Cun et al., 1989). For modern overparameterized LLMs, recent investigations have even documented that sparsity arises naturally in their feed-forward layers, with only a small fraction of hidden neurons activated for any given token (Li et al., 2023; Zhang et al., 2022b). Thus, with feed-forward computation accounting for over two-thirds of the parameters and over $8 0 \%$ of the total FLOPs in larger models (Pope et al., 2023), sparsity seemingly offers a natural opportunity for concrete computational savings.

![](images/1.jpg)  
Figure 1 | Comparison of ELL with our new TwELL and Hybrid sparse formats designed for LLM inference and training, respectively.

However, a frustrating paradox has blocked progress: despite performing far less theoretical computation, official kernels implementing sparse operations can often run slower than dense operations on modern GPUs. The culprit is a fundamental mismatch between unstructured sparsity and GPU architectures, whose hardware and software stacks have been heavily optimized for dense computation patterns (Lawson et al., 1979; NVIDIA, 2025a,b,c). In contrast, heterogeneous workloads together with the overheads from materializing and managing sparse indices have been critical challenges preventing generalized computational savings. Due to these challenges, previous attempts to realize efficiency gains have relied on considerable deviations from modern training recipes and have yet to see practical adoption (Liu et al., 2023; Wang et al., 2024). In this work, we introduce new kernels designed for modern NVIDIA GPUs to bridge this gap and leverage unstructured sparsity to deliver substantial speedups while reducing memory requirements and energy consumption during both LLM inference and training. Our kernels build on Tile-wise ELLPACK (TwELL), a new packing format for sparse data that can be naturally materialized in the epilogue of highly-optimized matrix multiplication kernels, removing a canonical bottleneck of prior packing schemes. Starting from TwELL, our inference kernels fuse multiple matrix-multiplications into a single optimized pipeline that minimizes computation, while our training kernels further reduce the sparse representation to a hybrid format that trivializes storage costs of intermediate activations. To substantiate our gains, we provide a quantitative study of LLM sparsity across model scales, demonstrating that mild levels of L1 regularization can achieve over $9 9 \%$ sparsity with negligible impact on downstream performance. Through our new kernels, we show these sparsity levels translate into increasing benefits with larger parameter counts in terms of processing throughput, energy savings, and memory requirements  delivering up to $2 0 . 5 \%$ and $2 1 . 9 \%$ speedups in forward execution and training for models with billions of parameters. We analyze how these benefits specifically come from the computational unevenness across network layers and natural language data, which can be inherently leveraged in sparse models. By providing a clear demonstration of its practical benefits, we hope this work will help establish sparsity as a new axis for improving the scalability and performance of modern foundation models. In summary, our main contributions are threefold: 1.We introduce and share new CUDA kernels for inference and training, with several key innovations to make sparse LLMs cheaper, faster, and lighter on modern GPUs. 2. We provide a quantitative analysis showing that high levels of unstructured sparsity can be achieved using mild L1 regularization with negligible compromises on performance. 3. We demonstrate and analyze how our kernels leverage such sparsity with substantial and increasing benefits at larger scales for LLMs with billions of parameters.

# 2. Large Language Models, Feedforward Blocks, and Sparsity

While the original transformer used a simple 2-layer feedforward block, this module has seen considerable evolution since its conception (Vaswani et al., 2017). The most recent architectures have largely converged to a 3-layer gated design that has consistently proven empirical superiority when evaluated at large scale (Shazeer, 2020). While in this work we release kernels for both the original and gated blocks, we focus our main text on the newer design and defer to Appendix C for further discussions, results, and comparisons with the older variant.

# 2.1. Feed-forward Modules as Sparse Knowledge Stores

A modern gated feed-forward block (Shazeer, 2020) is parameterized by three weight matrices $W _ { g } \in \mathbb { R } ^ { K \times N }$ $W _ { u } \in \mathbb { R } ^ { K \times N }$ , and $W _ { d } \in \mathbb { R } ^ { N \times K }$ representing the gate, up, and down projection matrices, respectively. In our notation, we use $M$ to denote the feed-forward block's effective batch size over all batched sequences and positions, $K$ to denote its input/output dimensions, and $N$ to denote its hidden expanded dimension. The gate and up projection matrices both process the block's input batch $x \in \mathbb { R } ^ { M \times K }$ and produce the up and gate activations $h _ { g }$ and $h _ { u } \in \mathbb { R } ^ { M \times N }$ , where the symmetry between the two is broken with a non-linear activation function $\sigma$ These projections are then combined with elementwise multiplication into a unified hidden representation $h \in \mathbb { R } ^ { M \times N }$ , before being projected back to their original dimensionality using the down projection weights $W _ { d }$ , to compute the block's outputs y  RM×K.

$$
h _ { u } = x W _ { u } , \quad h _ { g } = \sigma ( x W _ { g } ) , \quad h = h _ { u } \odot h _ { g } , \quad y = h W _ { d } .
$$

Since the hidden dimension $N$ is typically much larger than $K$ , feed-forward blocks can often account for most of the model's parameters and FLOPs. We note that a common conceptualization of these architectural components is that of a dynamic key-value memory (Dai et al., 2022; Geva et al., 2021). In this mental model, the inner products between $x$ and the columns of $W _ { g }$ and $W _ { u }$ induce keys $h$ , while the rows of $W _ { d }$ are seen as values acting as memory slots that can be dynamically retrieved based on the input.

# 2.2. Simple Ingredients for Training Sparse LLMs

We employ a simple recipe to induce varying levelsof sparsity in the feed-forward activations, making minimal deviations from established architectures and training objectives. First, we use the ReLU as the activation function of choice following the gate projections. Second, we add a simple L1 loss to the standard cross-entropy with a tunable coefficient $L _ { 1 }$ to promote sparsity across the model's $L$ layers:

$$
L _ { 1 } \times \frac { 1 } { L } \sum _ { l = 1 } ^ { L } \frac { 1 } { M N } \sum _ { m = 1 } ^ { M } \sum _ { n = 1 } ^ { N } \lvert h ^ { l } [ m , n ] \rvert .
$$

We note that many recent LLM architectures have deviated from using ReLUs in favor of smoother activation functions such as SiLU, with minor but consistent benefits (Shazeer, 2020; Touvron et al. 2023; Yang et al., 2024). In Appendix C, we provide direct empirical comparisons between these choices and also refer to orthogonal studies in the recent literature showing that domain-specific performance differences can be bridged with targeted training techniques (Lomeli et al., 2025; Mirzadeh et al., 2023).

# 3. Making Sparse LLMs Fast

We introduce new CUDA kernels for inference and training that leverage unstructured sparsity to en r e   e rr bl.   nd our kernels build on TwELL, a new sparse format specifically designed for seamless kernel fusion to realize the inherent throughput and memory benefits of sparsity with minimal overheads. In this section, we describe the core components and advantages of our new kernels with algorithmic descriptions that summarize their logic at the level of individual cooperative thread arrays (CTAs). We refer to Appendix A for code listings and more detailed design discussions of the thread-level CUDA implementations for H100 GPUs.

# Algorithm 1 | Algorithmic description of gate projection with our matmul kernel with TwELL storage

Parameters: Tile sizes $T _ { n } , T _ { m }$ , compression ratio C   
2: Input: Dense $\boldsymbol { x } \in \mathbb { R } ^ { M \times K }$ , $W _ { g } \in \mathbb { R } ^ { K \times N }$ ,   
3: Output: Sparse $h _ { \nu } \in \mathbb { R } ^ { M \times N / C }$ , $h _ { I } \in \mathbb { N } ^ { M \times N / C }$ , $h _ { n z } \in \mathbb { N } ^ { M \times N _ { T } }$   
4:for all tiles starting at $( m _ { 0 } , n _ { 0 } )$ in parallel across CTAs do   
5: $S \gets x [ m _ { 0 } ! m _ { 0 } + T _ { m } , :$ l $W _ { g } [ : , n _ { 0 } : n _ { 0 } + T _ { n } ]$   
6: for $r \gets 0 \ldots T _ { m } { - } 1$ do   
7: $m \gets m _ { 0 } + r$ {global row index}   
8: $z \gets 0$ {running count of non-zeros in tile}   
9: for $c \gets 0 \ldots T _ { n } { - } 1$ do   
10: if $( S [ r , c ] > 0 )$ then   
11: $n  n _ { 0 } / C + z$ {global TwELL column index}   
12: $h _ { I } [ m , n ] \gets n _ { 0 } + c$ {store non-zero index}   
13: $h _ { \nu } [ m , n ] \gets S [ r , c ]$ {store non-zero value}   
14: $z \gets z + 1$ {increment non-zero count}   
15: end if   
16: end for   
17: $h _ { n z } [ m , n _ { 0 } / T _ { n } ] \gets z$ {store final count of non-zeros}   
18: end for   
19: end for

# Algorithm 2 | Algorithmic description of fused up and down projections from gate activations in the TwELL format

1: Parameters: Tile size $T _ { n }$ , compression ratio C   
2: Input: Sparse $h _ { \nu } \in \mathbb { R } ^ { M \times N / C }$ , $\bar { h _ { I } } \in \mathbb { N } ^ { M \times N / C }$ , $h _ { n z } \in \mathbb { N } ^ { M \times N _ { T } }$ ; dens   
$\boldsymbol { x } \in \mathbb { R } ^ { M \times K }$ , $W _ { u } \in \mathbb { R } ^ { K \times N }$ , $W _ { d } \in \mathbb { R } ^ { N \times K }$   
3: Output: Dense $y \in \mathbb { R } ^ { M \times K }$   
4: for all $m \in \pi ( 0 . . M { - } 1 )$ in parallel across CTAs do   
5: $x _ { m } \gets x [ m , : ]$ ; $y _ { m } \gets 0$   
6: o $\mathfrak { r } t \gets 0 \ldots N _ { T } - 1$ do   
7: $z \gets h _ { n z } [ m , t ]$   
8: for $c \gets 0 \ldots z - 1$ do   
9: $n \gets h _ { I } [ m , t \times T _ { n } / C + c ]$ {non-zero column index}   
10: $w _ { u } = W _ { u } [ : , n ]$ $\{ n$ -th column of $W _ { u } )$   
11: $u \gets ( x _ { m } \cdot w _ { u } )$ {sparse $h _ { u } [ m , n ]$ element}   
12: $w _ { d } \gets W _ { d } [ n , : ]$ $\mathbf { \xi } _ { n }$ -th row of $W _ { d } \}$   
13: $y _ { m } \gets y _ { m } + \left( h _ { \nu } [ m , t \times T _ { n } / C + c ] \times u \right) w _ { c }$   
14: end for   
15: end for   
16: $y [ m , : ] \gets y _ { m }$   
17:end for

# 3.1. Sparse Formats and Kernels

The ELLPACK format (ELL) is considered the state-of-the-art for fast and efficient sparse matmuls (Kincaid et al., 1989). This format was leveraged in some of the earliest GPU implementations of sparse algebra (Bell and Garland, 2009), with more recent work focused on developing packing and sorting variants for better performance (Anzt et al., 2014; Kreutzer et al., 2014). As shown in part a. of Figure 1, an $M \times N$ matrix $h$ in the ELL format is stored as two padded matrices $h _ { \nu }$ and $h _ { I }$ of size $M \times N _ { n z }$ with the non-zero values of $x$ and their column indices packed at the beginning of each row. This format prioritizes downstream usability over storage, padding the rows up to the maximum number of nonzero elements $N _ { n z }$ for efficient retrieval.

The main logic in most matmul kernels to perform $y = h W$ with ELL, is to launch different parallel accumulations for each row $m = 0 , \ldots , M - 1$ of the sparse matrix $h$ using a set number of threads. In each accumulation, the kernel iterates for $n = 0 , . . . , N _ { n z } - 1$ times, loading each column index $i = h _ { I } [ i , j ]$ and value $\nu = h _ { \nu } [ i , j ]$ of $h$ , and multiplying it with the $K$ dimensional row of the dense weight $W [ i , : ]$ . The key advantage of this format is that only a fraction of the weight columns and input values need to be processed, skipping the remaining zeros. To further reduce data access and computation, some later extensions like ELLPACK-R (Vazquez et al., 2010) also store the number of non-zeros in each row in a separate vector $h _ { n z }$ .

# 3.2. TwELL, a Sparse Data Format for Kernel Fusion

An effective predominant design for modern kernel pipelines is to maximize operator fusion and avoid unnecessary global memory accesses in order to best leverage the high compute throughput of modern NVIDIA GPUs. To this end, in a gated feed-forward block where sparsity patterns are determined by the gate activations $h _ { g }$ , prior sparse formats such as ELL suffer a major drawback. In essence, representing $h _ { g }$ with ELL requires first accessing all elements in every row to count, compare, and align the non-zero values and indices. However, existing matmul kernels for dense inputs rely on parallelizing computation across small 2D tiles $T _ { m } \times T _ { n }$ of the outputs, computed independently in separate CTAs. Thus, obtaining the gate activations directly in the ELL format from the non-sparse inputs cannot be done in the same kernel of $h _ { g } = \mathrm { R e L U } ( x W )$ without introducing expensive synchronization among different CTAs. In contrast, launching a separate kernel to do the conversion inherently introduces non-trivial overheads that concretely limit attainable throughput gains of the whole computation.

To address these limitations, we introduce Tile-wise ELLPACK (TwELL). As illustrated in part b. of Figure 1, rather than focusing on whole rows, TwELL divides the columns of $h _ { g }$ in groups of horizontal 1D tiles of size $T$ . Within each group of columns, TwELL stores the non-zero values present and their indices in a local ELL-based packing format, with the data of each row aligned at the beginning of each horizontal tile. This results in two matrices containing locally aligned values $h _ { \nu } \in R ^ { M \times N / C }$ and indices $h _ { I } \in R ^ { M \times N / C }$ , where $C$ is a compression factor set such that $T / C$ is higher than the maximum number of non-zeros in any tile to avoid storage overflow. In our implementation of TwELL, we also store an additional matrix with the number of non-zero elements $h _ { n z } \in R ^ { M \times N _ { T } }$ to facilitate further computations, with as many columns as total tiles $N _ { T } = \lceil N / T \rceil$ . While inherently less expensive to derive, the main advantage of TwELL over ELL is actually ease of materialization following a modern tiled matmul: by setting the horizontal tiling dimensions to match, $T = T _ { n }$ , the TwELL format can be recovered in the same kernel performing $h _ { g } = \mathrm { R e L U } ( x W )$ before storing the outputs to DRAM. Fusing the two operations removes the requirement of performing additional kernel spawns, memory reads, or synchronization steps, leading to a natural integration into existing LLM pipelines.

# 3.3. Kernels for TwELL Construction and Fast, Fused Inference

In Algorithm 1, we provide pseudocode to summarize the logic of our CUDA matmul kernel storing the sparse outputs in the TwELL format (lines 6-18). Given the output distribution patterns of tensor core operations, we obtain the memory addresses to store the packed non-zero values $h _ { \nu }$ and their indices $h _ { I }$ by keeping a local non-zero count that only requires warp-level synchronization. While not an inherent requirement of TwELL, storing the number of non-zeros in each tile, $h _ { n z }$ , allows us to forego the overhead from initializing $h _ { I }$ with any "padding" value and from the additional control logic of checking validity in future usages. While omitted from Algorithm 1, we leverage fast asynchronous TMA reads and writes by first caching the dense inputs and sparse TwELL outputs to shared memory. We also pipeline computation and global memory accesses with a persistent cooperative design similar to the one in CUTLASS (NVIDIA, 2025c).

For inference, we introduce a single additional kernel to perform the rest of the computation in the feedforward block, leveraging the gate activations stored in the TwELL format to efficiently fuse the up and down projections together. This kernel, summarized in Algorithm 2, is launched on a grid made of single warp CTAs each processing a different row $m$ of the input activations $x$ . Minimizing the size of each CTA serves the purpose of maximizing concurrency and L2-hits across the grid, as non-zero activations tend to have high correlation within input sequences. The fused matmuls are executed by traversing over the sparsified activations with two nested for loops: the first one statically-unrolled over the number of column tiles (line 6) and the second one dynamically iterating over the corresponding number of non-zeros in each tile (line 8). For each non-zero activation at index $n$ , the CTA collectively loads the $n ^ { t h }$ row of $W _ { u }$ and column of $W _ { d }$ to perform a dot product, followed by a scalar-vector product and accumulating its results (lines 9-13)  corresponding to the following computation:

$$
[ m , : ] = \sum _ { t = 0 } ^ { N _ { T } - 1 } \sum _ { c = 0 } ^ { h _ { \mathbb { Z } } [ m , t ] - 1 } \underbrace { h _ { \boldsymbol \nu } [ m , t \times T _ { n } / C + c ] } _ { h _ { \boldsymbol \nu } \mathrm { ~ n o n - z e r o ~ v a l u e } } \underbrace { ( \boldsymbol { x } [ m , : ] \cdot \boldsymbol { W _ { u } } [ : , n ] ) } _ { h _ { \boldsymbol \nu } \mathrm { ~ e l e m e n t } } \underbrace { \boldsymbol { W _ { d } } [ n , : ] } _ { \begin{array} { c } { \boldsymbol { W _ { d } } \mathrm { ~ r o w } } \\ { \boldsymbol { W _ { d } } \mathrm { ~ r o w } } \end{array} } , \mathrm { ~ w h e r e ~ } n = \underbrace { h _ { I } [ m , t \times T _ { n } / C + c ] } _ { h _ { I } \mathrm { ~ n o n - z e r o ~ i n d e x } }
$$

Implicitly materializing the $h _ { u }$ values only inside the kernel serves to further reduce DRAM access to maximize throughput. Together, the kernels in our inference pipeline align the core principles of tiling and operator fusion into a single execution flow, harnessing the computational advantages of sparsity while minimizing its canonical overheads.

# 3.4. Hybrid Conversion for Efficient Storage

During training, memory becomes a key bottleneck for throughput as large intermediate activations and optimizer states are needed for backpropagation. Here, sparsity provides a natural opportunity to tackle these bottlenecks by trivializing intermediate storage costs and accelerating gradient computations. However, directly using TwELL with a high compression ratio or other ELL-based formats for this purpose inherently relies on the maximum number of non-zeros $N _ { n z }$ to be known ahead of time and strictly small. However, as we will illustrate in Section 4, we find that these conditions are practically never met during LLM training as sparsity patterns exhibit significant non-uniformity across different tokens, with the maximum number of non-zeros often orders of magnitude larger than the average.

We overcome these limitations by first converting the TwELL activations to a hybrid sparse format and introducing a new set of custom kernels designed specifically for memory-efficient training. As illustrated part c. of Figure 1, our format dynamically partitions and stores the rows of $h _ { g }$ either in ive  Ex $h _ { g } ^ { s } \in R ^ { M ^ { s } \times N _ { \hat { n } z } }$ or a dense backup $h _ { g } ^ { d } \in R ^ { M ^ { d } \times N }$ The partitioning logic simply routes the rows of $h _ { g }$ based on their non-zero counts, which are cheaply computed from the locally aligned TwELL tiles. Our hybrid format also maintains a lightweight array of column indices $h _ { I } \ \in \ R ^ { M ^ { s } \times N _ { \hat { n } z } }$ matching the size of the sparsified ELL matrix, and a simple binary vector indicating the storage location of each row $h _ { b } \in R ^ { M }$ . In practice, we find that we can set $N _ { \hat { n } \tilde { z } }$ over an order-of-magnitude lower than $N$ with minimal overflow into $h _ { g } ^ { d }$ , avoiding stringent ELL requirements while still trivializing memory and computation during the rest of the training step.

# 3.5. Kernels for Lightweight Efficient Training

After materializing $h _ { g }$ in our hybrid format from the pre-activations $x W _ { g }$ , we design custom kernels to perform efficient hybrid-to-dense and dense-to-hybrid matmuls. We directly use these kernels to execute the rest of the forward pass, computing $h _ { u } = x W _ { u }$ and $y = h W _ { d }$ Unlike inference, during training we execute the up and down projections in separate steps, allowing us to efficiently store the sparsified hidden states and minimize recomputation in the backward pass. In Algorithm 3, we outline the logic of the hybrid-to-dense matmul for a generic input $h$ and weight W, with the dense-to-hybrid variant also following the same general structure. Our approach combines a typical ELL kernel, with each CTA processing individual rows of the output $y$ (lines 4-13), and a traditional tiled kernel using Tensor cores for the dense backup rows (lines 14-17). During the sparse portion of the matmul computation, we opt to statically-unrol the accumulation up to the maximum number of non-zeros $N _ { \hat { n } \tilde { z } }$ for each row. Moreover, we also statically pre-allocate the dense backup portions of all the activations based on the sparsity statistics observed during training. We note that these design choices introduce minimal extra computation and storage costs, which are largely offset by avoiding dynamic overheads. During the backward pass, we retrieve the sparsified activations together with the L1 and output gradients $\nabla \boldsymbol { y }$ , allowing us to backpropagate without performing any expensive dense computation. This is achieved using two additional kernels that support efficient injection of L1 gradients into a given sparsity pattern and efficient transposition of our hybrid format for future coalesced accesses. We first use the stored sparsity pattern of $h$ to obtain its gradients via our efficient dense-to-hybrid matmul $\nabla y W _ { d } ^ { T }$ , followed by the L1 injection. With $\nabla h$ available, we recover the rest of the input and weight gradients with direct applications of our hybrid-to-dense and transposed kernels:

$$
\begin{array} { r l } & { \quad \quad \nabla h _ { u } = \nabla h \odot h _ { g } , \quad \nabla h _ { g } = \nabla h \odot h _ { u } , } \\ & { \quad \quad \nabla W _ { u } = x ^ { \top } \nabla h _ { u } , \quad \nabla W _ { g } = x ^ { \top } \nabla h _ { g } , \quad \nabla W _ { d } = h ^ { \top } \nabla y , } \\ & { \quad \quad \quad \nabla x = \nabla h _ { u } W _ { u } ^ { \top } + \nabla h _ { g } W _ { g } ^ { \top } . } \end{array}
$$

![](images/2.jpg)  
Figure 2 | Training curves of LLMs across L1 regularization levels.

# Algorithm 3 | Algorithmic description of matmul from input activations in the hybrid format

1: Input: $T _ { m } , T _ { k } , T _ { n }$ , $h : = ( h ^ { s } , h ^ { d } , h _ { I } , h _ { b } )$ , $W \in \mathbb { R } ^ { K \times N }$   
2: Output: $y \in \mathbb { R } ^ { M \times N }$   
3: $\pi _ { s } \gets \{ m : h _ { b } [ m ] = 0 \}$ ; $\pi _ { d } \gets \{ m : h _ { b } [ m ] = 1 \}$   
4:for all $m _ { s } \in 0 . . M ^ { s } - 1$ in parallel do   
5: $m \gets \pi _ { s } [ m _ { s } ]$ {global row index}   
6: $y _ { m } \gets 0$ {row accumulator}   
7: for $j \gets 0 \ldots N _ { \hat { n } z } - 1$ do   
8: $n \gets h _ { I } [ m _ { s } , j ]$ {non-zero column index}   
9: $\nu \gets h ^ { s } [ m _ { s } , j ]$ {sparse value}   
10: $y _ { m } \gets y _ { m } + \nu \cdot W [ n , : ]$ {sparse row update}   
11: end for   
12: $y [ m , : ] \gets y _ { m }$   
13: end for   
14: for all tiles starting at $( m _ { 0 } , n _ { 0 } )$ in parallel do   
15: $S \gets h ^ { d } [ m _ { 0 } ; m _ { 0 } { + } \bar { T } _ { m } , ;$ ] [:, $n _ { 0 } { : } n _ { 0 } { + } T _ { n } ]$   
16: $y \big [ \pi _ { d } \big [ m _ { 0 } { : } m _ { 0 } { + } T _ { m } \big ]$ , $n _ { 0 } ! n _ { 0 } + T _ { n } ] \gets S$   
17:end for Crucially, our execution logic reflects a deliberate design choice: rather than aggressively fusing individual operators, the trainig kernel pipeline s sructurearound the trainn step  t ey In this setting, the hybrid format miniizes backward computation and memory overheads, alowing us to avoid dense calculations while remaining robust to the highly non-uniform sparsity patterns that make ELL-based approaches traditionally brittle.

# 4. Experimental Results

# 4.1. Training and Evaluation Settings

We provide quantitative results evaluating the performance and efficiency of LLMs at different sparsity levels and scales. Our models are based on the "Transformer $+ + { } ^ { \dag }$ architecture, common to recent LLMs such as Qwen and Llama (Touvron et al., 2023; Yang et al., 2024) with the gated feedforward blocks described in Section 2. We train our models just above the chinchilla-optimal number of tokens for each model size (Hoffmann et al., 2022), using the fineweb dataset (Penedo et al., 2024). We default to a context length of 2048, a batch size of 1M tokens, and the AdamW optimizer with a weight decay of 0.1 and a cosine schedule (Loshchilov and Hutter, 2017). Other hyperparameters, such as the hidden size and total number of layers, are based on the model size and chosen based on modern practices (Hoffmann et al., 2022). We note that our sparse models use the same training hyperparameters as our non-sparse baselines, as the addition of L1 regularization in the feedforward blocks did not seem to affect other choices.

To measure model performance, we use cross-entropy scores and seven different common downstream tasks assessing logic and reasoning capabilities after pretraining (Bisk et al., 2020; Clark et al., 2018; Mihaylov et al., 2018; Sakaguchi et al., 2021; Talmor et al., 2019; Zellers et al., 2019). In this Section, we focus on aggregated performance metrics for conciseness, and we refer to Appendix D for the full per-task breakdowns. To measure efficiency, we analyze the throughput gains at different sparsity levels when integrating our training and inference kernels by recording execution times, memory requirements, and energy consumption at each stage. Across our experiments, we keep a fixed sequence length of 2048 and vary the micro batch size based on the available memory. Unless otherwise specified, we train and collect our results on a single node of eight H100 PCIe GPUs, a commonly available infrastructure in current listings of cloud providers and scientific clusters. We refer to Appendix B for further details on our training and evaluation settings, together with full hyperparameters specific to each of the considered model sizes.

![](images/3.jpg)  
Figure 3 | Downstream accuracy and sparsity statistics of LLMs across L1 regularization levels.

![](images/4.jpg)  
Figure 4 | Forward pass speedups and energy savings from our sparse LLM inference kernels across L1 regularization levels.

# 4.2. More Efficient LLMs with Unstructured Sparsity

Performance across sparsity levels. We start by evaluating the effect of introducing different levels of L1 regularization on the performance and sparsity of a 1.5B parameter model. In particular, we consider eight values for the $L _ { 1 }$ coefficient, ranging from no regularization $\left( L _ { 1 } = 0 \right)$ to the point where less than a single neuron on average remains activated after training $( L _ { 1 } = 1 \times 1 0 ^ { - 4 } )$ . In Figure 2, we show the training curves of the different models, while in Figure 3 we report downstream task performance together with the final number of nonzero activations averaged across the feed-forward blocks. While our 1.5B model has a hidden feedforward dimensionality of 5632, we find that the nonregularized model already attains more than $2 0 \%$ sparsity with only 911 neurons activated. Moreover, consistently with Mirzadeh et al. (2023), we find that introducing small levels of regularization already pushes the average number of non-zeros orders of magnitude lower but with high variations across different tokens and layers. In particular, even at the highest regularization point, we find that a small fraction of tokens stl excite several hundred neurons, indicating a reallocation of capacity. Despite this adaptivity, performance-wise, we do start seeing some performance degradation below $0 . 5 \%$ of activated neurons. Nonetheless, our results suggest that smaller levels of regularization do not visibly hinder capacity beyond the weight decay already induced by the AdamW optimizer: up until $L _ { 1 } = 3 \times 1 0 ^ { - 5 }$ , we record essentially no drop in task performance and a negligible increase of final cross-entropy within $2 \%$ of the unregularized baseline.

![](images/5.jpg)  
Figure 5 | Training speedups and peak memory reduction from our sparse LLM training kernels across L1 regularization levels.

Leveraging sparsity for faster and lighter LLMs. We contrast our performance results by analyzing the efficiency improvements from integrating our kernels at different sparsity levels. In Table 1 | Comparison of performance and efficiency statistics of sparse LLMs leveraging our kernels with traditional models.   

<table><tr><td>Model scale</td><td>Sparse</td><td>Mean task accuracy</td><td colspan="2">Forward execution (input tokens/ms)</td><td colspan="2">Energy per token (mJ)</td><td colspan="2">Training step Peak memory (input token/ms)</td></tr><tr><td>0.5B params</td><td>X</td><td>40.4%</td><td>410 (0.0%)</td><td>1.63</td><td>(0.0%)</td><td>97.3 (0.0%)</td><td>26.2</td><td>(0.0%)</td></tr><tr><td>10B tokens</td><td>✓</td><td>40.4%</td><td>480 (+17.0%)</td><td></td><td>1.43 (-11.8%)</td><td>95.9 (-1.5%)</td><td>21.2</td><td>(-19.2%)</td></tr><tr><td>1B params 20B tokens</td><td>X</td><td>44.6%</td><td>185</td><td>(0.0%)</td><td>3.71 (0.0%)</td><td>48.6 (0.0%)</td><td>44.5</td><td>(0.0%)</td></tr><tr><td></td><td>✓</td><td>44.7%</td><td>219 (+18.1%)</td><td></td><td>3.17 (-14.6%)</td><td>52.1 (+7.1%)</td><td>33.1</td><td>(-25.5%)</td></tr><tr><td>1.5B params</td><td>X</td><td>46.4%</td><td>119</td><td>(0.0%)</td><td>5.73 (0.0%)</td><td>31.8 (0.0%)</td><td>62.8</td><td>(0.0%)</td></tr><tr><td>30B tokens</td><td>✓</td><td>46.2%</td><td>141 (+18.8%)</td><td></td><td>4.87 (-15.0%)</td><td>35.5 (+11.6%)</td><td>45.1</td><td>(-28.1%)</td></tr><tr><td>2B params</td><td>X</td><td>49.1%</td><td>87.8</td><td>(0.0%)</td><td>7.85 (0.0%)</td><td>22.4 (0.0%)</td><td>46.7</td><td>(0.0%)</td></tr><tr><td>40B tokens</td><td>✓</td><td>48.8%</td><td>106 (+20.5%)</td><td></td><td>6.51 (-17.0%)</td><td>27.3 (+21.9%)</td><td>57.1</td><td>(+22.3%)</td></tr></table>

Figure 4, we provide the average relative speedups and total energy savings recorded during forward execution through our LLMs. Across all considered sparsity levels above $L _ { 1 } = 0$ , we find that our inference kernels lead to visible throughput gains ranging up to $3 0 \%$ . These throughput gains are compounded by nearly $3 \%$ less GPU power draw above $L _ { 1 } = 3 \times 1 0 ^ { - 5 }$ , reslting in even higher energy savings. In Figure 5, we also show the average relative speedups and peak memory reduction with our training kernel. In line with our inference kernels, the speedups recorded throughout training significantly increase up to $2 4 \%$ with sparser models. Furthermore, the peak GPU memory required for training decreases by more than $2 4 \%$ even for the lowest considered sparsity level, reducing hardware barriers for efficient training at billion-parameter scales (we refer to Appendix D for results on an RTx6000). Taken together, we believe our results provide compelling evidence that specialized targeted kernels can make sparsity a new viable axis for the design of modern LLMs, leading to significant efficiency improvements across their full lifecycle.

Sparsity across model scales. We analyze how model scale affects the performance and efficiency of sparse LLMs. For this analysis, we set $L _ { 1 } = 2 \times 1 0 ^ { - 5 }$ based on our earlier results on the 1.5B model, which we recommend as a conservative threshold to avoid any significant performance degradation. In Table 1, we compare the performance and efficiency of sparse and non-sparse LLMs at the chinchillaoptimality boundary  ranging from a 0.5B model trained on 10B tokens to a 2B model trained on 40B tokens. Consistent with our earlier results, we find no performance drops beyond random deviations for all scales when mild L1 regularization is introduced. Furthermore, we find that LLMs become increasingly effective at supporting sparsity at larger scales, resulting in a lowernumber of average non-zero elements (from 39 to 24, going from the 0.5B to the 2B model). In turn, this makes all the aforementioned throughput and memory benefits of our kernels grow: the 2B sparse model processes tokens $2 0 . 5 \%$ faster during inference and trains $2 1 . 9 \%$ more efficiently with a larger micro-batch size. Theefindings suggest that sparsity aligns well with recently prevailing scaling trends, highlighting its growing potential relevance for future model development.

# 4.3. The Properties of Sparse Large Language Models

We analyze how LLMs effectively allocate sparsity across their layers and batched samples. For our analysis, we collect the activations from $2 ^ { 2 0 }$ input tokens using our 1.5B model trained with the suggested performance-preserving $L _ { 1 } = 2 \times 1 0 ^ { - 5 }$ . We complement this subsection with additional results in Appendix D, looking at additional levels of L1 regularization together with how sparsity evolves throughout training and its effects on dead neurons.

Sparsity and model depth. Figure 6 examines activations across model depth, relating the non-zero statistics of each layer to its respective contribution to inference speed-ups. While the average non-zeros across all layers is less than 30, the figure highlights clear variations in sparsity both across and within individual layers. In particular, the first twolayers are the least active,followed by a pronounced hump in the average number of non-zeros across the first half of the network. This sparsity pattern, peaking during early-middle layers, appears consistent with prior work suggesting that a substantial portion of an LLM's reasoning and knowledge retrieval occurs precisely at these depths (Wendler et al., 2024). Furthermore, within each layer, the maximum number of non-zeros often exceeds the layer's mean by more than an order of magnitude and shows no consistent pattern across the architecture. We also observe an intuitive and pronounced inverse correlation between each layer's average non-zeros and its relative speed-up, with a Pearson coefficient of over -0.996. In contrast, the maximum activation counts have a more limited effect on inference speedups, only noticeably in layer 8. This robustness is due to our kernel design, which hides the latency of highly activated tokens through maximally parallelized execution.

![](images/6.jpg)  
Figure 6 | Sparsity statistics and speedup contributions across different layers of our sparse LLMs.

Sparsity and input properties. Given the high level of unevenness across activations, we analyze what inputs spur the peaks and troughs in non-zero activation counts. In part a. of Figure 7, we identify common tokens with the six lowest and highest average number of non-zeros, filtering out outliers occurring at a lower frequency than $1 / 2 ^ { 1 4 }$ . We find that the tokens with the lowest non-zero activity often represent parts of common web links (doi, nlm, gov, nih) or contractions (doesn, couldn) that precede predictable next tokens in crawled web corpora. In contrast, tokens providing important contextual information about a passage have the highest activity, including particular verbs (loud, enduring) or nouns representing specific locations or substances (Vermont, Greeks, formaldehyde, ACH"). In part b. of Figure 7, we then plot how the average non-zeros vary with token position in the input sequence on a log-log scale. Interestingly, we find that the LLM allocates a much greater number of non-zeros to the very first tokens in a sequence, with an exponential decrease thereafter. Intuitively, these results indicate that LLMs appear to effectively focus their computational efforts on tokens with high information content and sequence positions where contextual cues from prior tokens are missing. Here, introducing sparsity not only provides an interpretable lens on model behavior, but also enables our kernels to leverage this inherent information unevenness for significant training and inference speedups.

![](images/7.jpg)  
Figure 7 | Sparsity statistics across LLM input tokens and positions.

# 5. Related Work

The emergence of sparsity in LLMs with ReLU activations and its theoretical benefits have been repeatedly documented in earlier work (Li et al., 2023; Mirzadeh et al., 2023). Since then, more recent methods have been proposed to enhance sparsity by altering modern gated architectures, claiming speedups running sparse feed-forward layers in isolation on older generations of devices. TurboSparse (Song et al., 2024) studies boosting sparsity via repeated ReLU non-linearities, while ProSparse (Song et al., 2025) finetunes pretrained models by manually thresholding activations. Q-Sparse (Wang et al., 2024) further deviates from standard models by using a straight-through estimator and retaining only top-K activations. Other work instead focused on introducing structured sparsity post-training, such as by predicting (Liu et al., 2023) and pruning activations to accelerate computation (Lee et al., 2024; Liu et al., 2024). Unlike these efforts, our paper introduces generalpurpose kernels to leverage unstructured sparsity, demonstrating empirical efficiency benefits during both LLM inference and training. We refer to Appendix E for an extended overview of prior work that more fundamentally reshapes architecture design.

# 6. Discussion and Future Work

In this work, we leverage unstructured sparsity to lessen the computational burdens of modern LLMs. For inference, we design a new sparse format and fused operations to efficiently execute the whole gated feed-forward blocks in only two kernel launches, minimizing global memory accesses and computation. For training, we introduce a new hybrid algorithm that dynamically schedules computation on both CUDA and Tensor cores, while trivializing storage costs of intermediate activations for backpropagation. We demonstrate that mild L1 regularization induces considerable levels of sparsity with negligible impact on downstream performance  which our kernels translate into significant gains in throughput, energy efficiency, and memory footprint at billion-parameter scales. While our work serves to provide a concrete demonstration of the benefits of sparse LLMs, there are numerous exciting avenues for future extensions. For instance, in Appendix C, we provide preliminary results indicating that the performance of highly sparse LLMs can be further improved with strategies targeted at dead-neuron mitigation. Moreover, ine-tuning existing dense models via recent sparsification approaches (Mirzadeh et al., 2023; Song et al., 2025) would allow bringing the benefits of our kernels to the vast library of pretrained LLMs available in the wild. By sharing our kernels, we hope our work will help promote sparsity as a new design axis to leverage for efficiency, ultimately reducing the growing energy and hardware costs of large-scale foundation models.

# Author contribution

Edoardo Cetin conceived the TwELL format, led the implementation and design of the CUDA kernels using TwELL, led model training and benchmarking, and made contributions to writing. Stefano Peluchetti did early work on sparse model training, advised the project, and made contributions to writing. Emilio Castillo conceived the hybrid format, co-led the implementation of the CUDA kernels and designed the training extensions, made contributions to kernel benchmarking, and made contributions to writing. Akira Naruse advised the project, was involved in early discussions about method design, and worked on early implementations of the sparse kernels. Mana Murakami was involved in early discussions about method design. Llion Jones did initial explorations of sparse model training, advised the project, was involved in early discussions about method design, and made contributions to writing.

# References

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. Smollm - blazingly fast and remarkably powerful, 2024. Hartwig Anzt, Stanimire Tomov, and Jack Dongarra. Implementing a sparse matrix vector product for the sell-c/sell-c-σ formats on nvidia gpus. University of Tennessee, Tech. Rep. ut-eecs-14-727, 2014. Jordan Ash and Ryan P Adams. On warm-starting neural network training. Advances in neural information processing systems, 33:38843894, 2020. Peter Belcak and Roger Wattenhofer. Fast feedforward networks, 2023. URL https: / /arxiv.org/ abs/2308.14711. Nathan Bell and Michael Garland. Implementing sparse matrix-vector multiplication on throughputoriented processors. In Proceedings of the conference on high performance computing networking, storage and analysis, pages 111, 2009. Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 74327439, 2020. Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:18771901, 2020. Siddhartha Chatterjee, Alvin R Lebeck, Praveen K Patnala, and Mithuna Thottethodi. Recursive array layouts and fast parallel matrix multiplication. In Proceedings of the eleventh annual ACM symposium on Parallel algorithms and architectures, pages 222231, 1999. Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 84938502, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.581. William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trllion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):139, 2022. Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 54845495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.446. Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. Advances in neural information processing systems, 28, 2015. Xu Owen He. Mixture of a million experts, 2024. URL https: //arxiv. org/abs/2407. 04153. D Hendrycks. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016. Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. Journal of Machine Learning Research, 22(241):1124, 2021. Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022. Zihao Huang, Qiyang Min, Hongzhi Huang, Defa Zhu, Yutao Zeng, Ran Guo, and Xun Zhou. Ultrasparse memory network, 2025. URL https://arxiv.org/abs/2411.12364. David R Kincaid, Thomas C Oppe, and David M Young. Itpackv 2d user's guide. Technical report, Texas Univ., Austin, TX (USA). Center for Numerical Analysis, 1989. Moritz Kreutzer, Georg Hager, Gerhard Wellein, Holger Fehske, and Alan R Bishop. A unified sparse matrix data format for efficient general sparse matrix-vector multiplication on modern processors with wide simd units. SIAM Journal on Scientific Computing, 36(5):C401C423, 2014. Guillaume Lample, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer, and Hervé Jégou. Large memory layers with product keys, 2019. URL https : //arxiv. org/abs/1907. 05242. Chuck L Lawson, Richard J. Hanson, David R Kincaid, and Fred T. Krogh. Basic linear algebra subprograms for fortran usage. ACM Transactions on Mathematical Software (TOMS), 5(3):308323, 1979. Yann LeCun, John Denker, and Sara Solla. Optimal brain damage. Advances in neural information processing systems, 2, 1989. Donghyun Lee, Je-Yong Lee, Genghan Zhang, Mo Tiwari, and Azalia Mirhoseini. Cats: Contextuallyaware thresholding for sparsity in large language models. arXiv preprint arXiv:2404.08763, 2024. Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020. Zonglin Li, Chong You, Srinadh Bhojanapalli, Daliang Li, Ankit Singh Rawat, Sashank J. Reddi, Ke Ye, Felix Chern, Felix Yu, Ruiqi Guo, and Sanjiv Kumar. The lazy neuron phenomenon: On emergence of activation sparsity in transformers, 2023. URL https: //arxiv . org/abs/2210 . 06313. James Liu, Pragaash Ponnusamy, Tianle Cai, Han Guo, Yoon Kim, and Ben Athiwaratkun. Training-free activation sparsity in large language models. arXiv preprint arXiv:2408.14690, 2024. Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher Re, et al. Deja vu: Contextual sparsity for efficient llms at inference time. In International Conference on Machine Learning, pages 2213722176. PMLR, 2023. Maria Lomeli, Matthijs Douze, Gergely Szilvasy, Loic Cabannes, Jade Copet, Sainbayar Sukhbaatar, Jason Weston, Gabriel Synnaeve, Pierre-Emmanuel Mazaré, and Hervé Jégou. Stochastic activations, 2025. Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. Alexandra Sasha Luccioni, Sylvain Viguier, and Anne-Laure Ligozat. Estimating the carbon footprint of BLOOM, a 176b parameter language model. Journal of Machine Learning Research, 24(253): 115, 2023. Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018. Seyed Iman Mirzadeh, Keivan Alizadeh-Vahid, Sachin Mehta, Carlo C. del Mundo, Oncel Tuzel, Golnoosh Samei, Mohammad Rastegari, and Mehrdad Farajtabar. ReLU Strikes Back: Exploiting Activation Sparsity in Large Languag Models. In The Twelfth International Conferenceon Leari Representations, 2023. NVIDIA. cuBLAS Library, 2025a. URL https://docs.nvidia. com/cuda/cublas/index.html. Release 13.1. NVIDIA. NVIDIA cuBLASDx Documentation, 2025b. URL https://docs.nvidia. com/cuda/ cublasdx/. Version 0.5.0. NVIDIA. Cutlass: Cuda templates and python dsls for high-performance linear algebra, Dec 2025c. URL https://github.com/NVIDIA/cutlass. OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023. Guilherme Penedo, Hynek Kydlíek, Anton Lozhkov, Margaret Mitchell, Colin A Raffel, Leandro Von Werra, Thomas Wolf, et al. The fineweb datasets: Decanting the web for the finest text data at scale. Advances in Neural Information Processing Systems, 37:3081130849, 2024. Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. Procedings of machine learning and systems, 5:606624, 2023. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. Prajit Ramachandran, Barret Zoph, and Quoc V Le. Searching for activation functions. arXiv preprint arXiv:1710.05941, 2017. Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM, 64(9):99106, 2021. Roy Schwartz, Jesse Dodge, Noah A. Smith, and Oren Etzioni. Green AI. Communications of the ACM, 63(12):5463, 2020. Pranjal Shankhdhar. Outperforming cublas on h100: a worklog, November 2024. URL https: //cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog. Noam Shazeer. Glu variants improve transformer. arXiv preprint arXiv:2002.05202, 2020. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geoffrey E. Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. CoRR, abs/1701.06538, 2017. URL http://arxiv.org/abs/1701.06538. Chenyang Song, Xu Han, Zhengyan Zhang, Shengding Hu, Xiyu Shi, Kuai Li, Chen Chen, Zhiyuan Liu, Guangli Li, Tao Yang, et al. Prosparse: Introducing and enhancing intrinsic activation sparsity within large language models. In Proceedings of the 31st International Conference on Computational Linguistics, pages 26262644, 2025. Yixin Song, Haotong Xie, Zhengyan Zhang, Bo Wen, Li Ma, Zeyu Mi, and Haibo Chen. Turbo sparse: Achieving llm sota performance with minimal activated parameters, 2024. Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 41494158, 2019. Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. Francisco Vazquez, G Ortega, José-Jesús Fernández, and Ester M Garzón. Improving the performance of the sparse matrix vector product with gpus. In 2010 10th IEEE International Conference on Computer and Information Technology, pages 11461151. IEEE, 2010. Hongyu Wang, Shuming Ma, Ruiping Wang, and Furu Wei. Q-sparse: All large language models can be fully sparsely-activated. arXiv preprint arXiv:2407.10969, 2024. Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. Do llamas work in english? on the latent language of multilingual transformers. arXiv preprint arXiv:2402.10588, 2024. An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report, 2024. URL https://arxiv. org/abs/2407.10671, 7:8, 2024. Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019. Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022a. Zhengyan Zhang, Yankai Lin, Zhiyuan Liu, Peng Li, Maosong Sun, and Jie Zhou. Moefication: Transformer feed-forward layers are mixtures of experts. In Findings of the Association for Computational Linguistics: ACL 2022, pages 877890, 2022b.

# A. Kernels Implementation Details

# A.1. Inference Kernels Selection 1 template <const int T_m, const int T_n, const int T_k>   
2 struct Tiles   
3 {   
4 alignas(128) __nv_bfloat16 a[T_m][T_k];   
5 alignas(128) __nv_bfloat16 b[T_n][T_k];   
6 };   
7   
8 template <   
9 const int T_m,   
10 const int T Tn,   
11 const int T_k,   
12 const int QUEUE_SIZE,   
13 const int T_n_compressed,   
14 int PADDING $\ c = ~ 4$   
15 >   
16 struct SmemStorage   
17 {   
18 Tiles<T_m, T_n, T_k> queue[QUEUE_SIZE];   
19 alignas(128)uint32_t c_packed[T_m][T_n_compressed $^ +$ PADDING];   
20 };   
21   
22 template <   
23 const int T_m,   
24 constnt n,   
25 const int T_k,   
26 const int CLUSTER_DIM_m,   
27 const int CLUSTER DIM_n,   
28 const int QUEUE_SIZE,   
29 const int NUM_ACTIVE_SMs,   
30 const int T_n_compressed,   
31 const bool LOOP_OVERFLOW_STORAGE   
32 >   
33 __global_ launch_bounds__(NUM_THREADS_PER_BLOCK)   
34 __cluster_dims__(CLUSTER_DIM_m $^ *$ CLUSTER_DIM_n, 1, 1)   
35 void mm_wgmma_nt_kernel(   
36 const CUtensorMap __grid_constant__ A_tm,   
37 const CUtensorMap _grid_constant__ B_tm,   
38 const CUtensorMap___grid_constant__ C_packed_tm,   
39 const int\* schedule_gmem_ptr,   
40 const int schedule_size_per_sm,   
41 const int K   
42   
43 {   
44 static_assert(   
45 b $( \mathrm { T } \_ \mathrm { m } \ = = \ 6 4 \ * \ 2 )$ ,   
46 "Only $\mathrm { ~ T ~ } _ { - } \mathrm { m } \ = = \ 1 2 8$ supported"   
47 );   
48   
49 constexpr int CLUSTER_SIZE $\equiv$ CLUSTER_DIM_m $^ *$ CLUSTER_DIM_n;   
50 extern __shared_ _align__(1024) unsigned char dynamic_smem[];   
51   
52 int cluster_idx;   
53 asm ("mov.u32 %   
54   
55 int cluster_lane_m;   
56 asm volatile("mov.u32 %   
57   
58 int cluster_lane_n $=$ cluster_lane_m %   
59 cluster_lane_m $/ =$ CLUSTER_DIM_n;   
60   
61 auto& tiles_s $\equiv$   
62 \*reinterpret_cast<   
63 SmemStorage<T_m, T_n, T_k, QUEUE_SIZE, T_n_compressed>\*   
64 >(dynamic_smem);   
65 int\* schedule_s $\equiv$ reinterpret_cast<int\*>(   
66 dynamic_smem   
67 + sizeof(SmemStorage<T_m, T_n, T_k, QUEUE_SIZE, T_n_compressed>)   
68 );   
69   
70 schedule_gmem_ptr $+ =$ cluster_idx $^ *$ schedule_size_per_sm;   
71 if (threadIdx.x < schedule_size_per_sm) {   
72 schedule_s[threadIdx.x] $\equiv$ schedule_gmem_ptr[threadIdx.x];   
73 }   
74   
75 _syncthreads();   
76   
shared _align_ 8uint64_t queue_full[QUEUE_SIZE];   
78 shared _align 8uint64_t queue_empty[QUEUE_SIZE];   
79   
80 if (threadIdx. $\texttt { x } = = \texttt { 0 }$ {   
81 #pragma unroll   
82 for (int queue_idx $\qquad = \ 0$ ; queue_idx $<$ QUEUE_SIZE; ++queue_idx) {   
83 ptx_init_smem_barrier(&queue_full[queue_idx], 1);   
84 ptx_init_smem_barrier(&queue_empty[queue_idx], $^ \mathrm { ~ 2 ~ * ~ }$ CLUSTER_SIZE);   
85 }   
86 }   
87   
88 asm volatile("barrier.cluster.arrive;\n" : :);   
89 asm volatile("barrier.cluster.wait; $\sin " : : \ u : )$ ;   
90   
91 if (threadIdx.x < WARP_GROUP_SIZE) {   
92 asm volatile("setmaxnreg.dec.sync.aligned.u32 %   
93   
94 if (threadIdx. $\texttt { x } = = \texttt { 0 }$ {   
95 int queue_idx $\qquad = \ 0$ ;   
96 int queue_phase $\qquad = \ 0$ ;   
97 uinti6_t mask_multicast $\mathrm { ~ \underline { ~ } { ~ m ~ } ~ } = \mathrm { ~ 0 ~ }$ ;   
98   
99 if constexpr (CLUSTER_DIM $\mathrm { ~ m ~ } > ~ 1$ {   
100 for (int $\mathrm { ~ \ ~ { ~ i ~ } ~ } = \mathrm { ~ 0 ~ }$ ; $\mathrm { ~ i ~ } <$ CLUSTER_DIM_m; $+ + \dot { 1 }$ ) {   
101 mask_multicast_m $| =$ (1u << (i \* CLUSTER_DIM_n));   
102 }   
103 mask_multicast_m $< < =$ cluster_lane_n;   
104   
105   
106 uint16_t mask_multicast_n;   
107 if constexpr (CLUSTER_DIM_n $> ~ 1$ ) {   
108 mask_multicast_n $=$   
109 ( $\mathrm { . 1 u \ < < }$ CLUSTER_DIM_n) - 1)   
110 $< <$ (cluster_lane_m $^ *$ CLUSTER_DIM_n);   
111 }   
112   
113 for (int schedule_it $\mathit { \Theta } = \mathit { \Theta } 0$ ;   
114 schedule_it < schedule_size_per_sm;   
115 ++schedule_it{   
116 const int packed_tile $=$ schedule_s[schedule_it];   
117 if (packed_tile $\ c = - 1$ {   
118 break;   
119   
120   
121 int tile_coord_m $=$ packed_tile $> > ~ 1 6$ ;   
122 int tile_coord_n $=$ packed_tile & OxFFF;   
123   
124 if constexpr (CLUSTER_DIM_n > 1) {   
125 tile_coord_n $\ast =$ CLUSTER_DIM_n;   
126 tilecoord_n $+ =$ cluster_lane_n;   
127 }   
128 if constexpr (CLUSTER_DIM_m $> ~ 1$ ){   
129 tile_coord_m $\ast =$ CLUSTER_DIM_m;   
130 tile_coord_m $+ =$ cluster_lane_m;   
131   
132   
133 for (int tile_start_k $\qquad = \ 0$ ;   
134 tile_start_k $< \mathtt { K }$ ;   
135 tile_start_k $+ =$ T_k, $^ { + + }$ queue_idx) {   
136 if (queue_idx $= =$ QUEUE_SIZE){   
105 cιle_cοu⊥u_ 1∥l,   
164 tile_start_k,   
165 &queue_full[queue_idx]   
166 ;   
167   
168   
169 if constexpr (CLUSTER_DIM_m $> ~ 1$ {   
170 if (cluster_lane_m $\scriptstyle = = 0$ ) {   
171 ptx_load_tile_tma_multicast_2d(   
172 &tiles_s.queue[queue_idx].b[0][0],   
173 &B_tm,   
174 tile_coord_ $\texttt { 1 * T } _ { - } \texttt { n }$ ,   
175 tile_start_k,   
176 mask_multicast_m,   
177 &queue_full[queue_idx]   
178 );   
179 }   
180 } else {   
181 ptx_load_tile_tma_2d(   
182 &tiles_s.queue[queue_idx].b[0][0],   
183 &B_tm,   
184 tile_coord_n $^ *$ T_n,   
185 tile_start_k,   
186 &queue_full[queue_idx]   
187 );   
188   
189   
190 }   
191   
192 } else {   
193 asm volatile("setmaxnreg.inc.sync.aligned.u32 %   
194 int queue_idx $\qquad = \ 0$ ;   
195 int queue_phase $\qquad = \ 0$ ;   
196 const int consumer_warpgroup_id $\equiv$   
197 (threadIdx.x - WARP_GROUP_SIZE) / WARP_GROUP_SIZE;   
198 const int tile_start_m $\equiv$ consumer_warpgroup_id $^ *$ WGMMA_m;   
199 const int consumer_thread_id $\equiv$ threadIdx.x %   
200 const uint thread_lane_idx_n $=$ (consumer_thread_id %   
201   
202 const int thread_store_offset_m $\equiv$ (   
203 tile_start_m   
204 $^ +$ consumer_thread_id / 32 \* 16   
205 $^ +$ (consumer_thread_id %   
206 );

const int thread_store_offset_n $\equiv$ ((consumer_thread_id %   
if (consumer_thread_id $<$ CLUSTER_SIZE){ for (int queue_idx $\qquad = \ 0$ ; queue_idx $<$ QUEUE_SIZE; ++queue_idx) { ptx_arrive_barrier_across_cluster( &queue_empty[queue_idx], consumer_thread_id, 1 ); }   
}   
at C_accum[T_n/16] [8];   
(int schedule_it $\qquad = \ 0$ ; schedule_it < schedule_size_per_sm; ++schedule_it {   
const int packed_tile $\equiv$ schedule_s[schedule_it];   
if (packed_tile $\mathbf { \Phi } = \mathbf { \Phi } - \mathbf { 1 } \dot { } .$ { break;   
}   
int tile_coord_m $\equiv$ packed_tile >> 16;   
int tile_coord_n $\equiv$ packed_tile & OxFFFF;   
if constexpr (CLUSTER_DIM $\smash { \texttt { n } > 1 }$ { tile_coord_n $\ast =$ CLUSTER_DIM_n; tile_coord_n $+ =$ cluster_lane_n;   
}   
if constexpr (CLUSTER_DIM_m > 1) { tile_coord_m $\ast =$ CLUSTER_DIM_m; tile_coord_m $+ =$ cluster_lane_m;   
}   
if (queue_idx $= =$ QUEUE_SIZE){ queue_idx $\qquad = \ 0$ ; queue_phase $\mathit { \hat { \Pi } } = \mathit { \Pi } 1$ ;   
}   
ptx_wait_barrier(&queue_full[queue_idx], queue_phase);   
asm volatile("wgmma.fence.sync.aligned;" ::: "memory");   
wgmma<T_n, 0, 1, 1, 0, $\mathtt { O } >$ ( C_accum, &tiles_s.queue[queue_idx].a[tile_start_m][0], &tiles_s.queue[queue_idx].b[0] [0]   
);   
#pragma unroll   
for (int wgmma_start_k $=$ WGMMA_k; wgmma_start_k < T_k; wgmma_start_k $\mathbf { \curlyeq } = \mathsf { \normalfont ~ W G M M A \_ k }$ { wgmma<T_n, 1, 1, 1, 0, $\mathtt { O } >$ ( C_accum, &tiles_s.queue[queue_idx].a[tile_start_m][wgmma_start_k], &tiles_s.queue[queue_idx].b[0] [wgmma_start_k] );   
}   
asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");   
asm volatile("wgmma.wait_group.sync.aligned %   
if (consumer thread id $<$ CLUSTER SIZE) { ptx_arrive_barrier_across_cluster( &queue_empty[queue_idx], consumer_thread_id, 1 ; }

Sparser, Faster, Lighter Transformer Language Models   
277 queue_idx++;   
278   
279 for (int tile_idx_k $\mathit { \Theta } = \mathit { \Theta } 1$ ;   
280 tile_ix_k $<$ K/T_k;   
281 ++tile_idx_k, $^ { + + }$ queue_idx){   
282 if (queue_idx $= =$ QUEUE_SIZE){   
283 queue_idx $\qquad = \ 0$ ;   
284 queue_phase $\mathit { \hat { \Pi } } = \mathit { \Pi } 1$ ;   
285   
286   
287 ptx_wait_barrier(&queue_full[queue_idx], queue_phase);   
288 asm volatile("wgmma.fence.sync.aligned;" ::: "memory");   
289   
290 #pragma unroll   
291 for (int wgmma_start_k $\qquad = \ 0$ ;   
293 wgmma_start_k $<$ T_k;   
wgmma_start_k $+ =$ WGMMA_k){   
294 wgmma<T_n, 1, 1, 1, 0, O>(   
295 C_ccum,   
296 &tiles_s.queue[queue_idx].a[tile_start_m][wgmma_start_k],   
297 &tiles_s.queue[queue_idx].b[0][wgmma_start_k]   
298 );   
299   
300   
301 asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");   
302 asm volatile("wgmma.wait_group.sync.aligned %   
303   
304 if (consumer_thread_id $<$ CLUSTER_SIZE){   
305 ptx_arrive_barrier_across_cluster(   
306 &queue_empty[queue_idx],   
307 consumer_thread_id,   
308 1   
309 );   
310 }   
311 }   
312   
313 {   
314 asm volatile("cp.async.bulk.wait_group.read $0 ; \ln ^ { \mathfrak { n } } )$ ;   
315 if (thread_lane_idx_n $\scriptstyle < = ~ 1$ {   
316 tiles_s.c_packed[   
317 thread_store_offset $\underline { { \boldsymbol { \ m } } } \ + \ \boldsymbol { 8 } \ *$ thread_lane_idx_n   
318 $] [ 0 ] ~ = ~ 0$   
319 }   
320 _syncwarp();   
321   
32 #pragma unroll   
for (int quadrant_slice_ $\mathtt { m } \ = \ 0$ ;   
324 quadrant_slice_m $< ~ 4$ ;   
325 quadrant_slice_m $+ = ~ 2$ $\begin{array} { r } { \mathopen { } \mathclose \bgroup \left\{ \begin{array} { r l r l } \end{array} \aftergroup \egroup \right. } \end{array}$   
326 int quadrant_store_offset_m =   
thread_store_offset_m $^ +$ quadrant_slice_m $\ast \textit { 4 }$ ;   
328   
329 #pragma unroll   
330 for (int wgmma_slice_n $\qquad = \ 0$ ;   
331 wgmma_slice_n < T_n / 16;   
302 ++wgmma_slice_n){   
int quadrant_store_offset_n $\equiv$   
334 thread_store_offset_n $^ +$ wgmma_slice_n \* 16;   
335   
336 #pragma unroll   
337 for (int quadrant_slice_n $\qquad = \ 0$ ;   
quadrant_slice_n $< ~ 8$ ;   
339 quadrant_slice_n $+ = 4$ {   
340 #pragma unroll   
341 for (int element $\mathrm { ~ \bf ~ \underline { ~ } { ~ n ~ } ~ } = \mathrm { ~ 0 ~ }$ ;   
343 element_n $< ~ 2$ ;   
++element_n) {   
344 if   
345 C_accum[wgmma_slice_n][   
346 quadrant_slice_m

In Figure 1, we provide code listings with the device code for our kernel implementing a custom matmul with our new TwELL storage, which we use to run the gate projection in our model. We omit device functions wrapping longer PTX injections for readability. As explained in Section 3 in the main text, this kernel executes an efficient tiled matrix multiplication, loading the dense input and the dense weight matrix and storing the output values using the Tensor Memory Accelerator (TMA) introduced with Hopper GPUs, while storing the output in the TwELL format during the kernel's epilogue. The base kernel follows a persistent design with pipelined computation based on persistent cooperative kernels in CUTLASS (NVIDIA, 2025c) and open-source CUDA reproductions (Shankhdhar, 2024). Unlike CUTLASS, the tile scheduler follows a pre-constructed ordering based on a Hilbert curve to maximize the reuse of the L2 cache (Chatterjee et al., 1999; Shankhdhar, 2024). In practice, we opt to pack the TwELL values $h$ ,indices $h _ { I }$ , and number of non-zeros $h _ { n z }$ in a single 32-bit matrix in $\mathbb { R } ^ { M \times N / C }$ .Thi is done by placing the number of non-zeros for each tile row in the first column and fitting the 16-bit TwELL value and index in the remaining entries. This design ensures strong locality and allows storing and loading the number of non-zeros together with the first 31 TwELL indices and values in a single coalesced access. While this loses a storage position, we set TwELL compression factors very conservatively for each sparsity level, making the occurrence of overflow practically impossible. For instance, we set the compression factor to 8 for our recommended sparsity regularization studied in our main results, with models ranging from 39-24 average non-zeros and an hanc eo $1 0 ^ { - 3 4 }$ Te TL cnvein rs whe mappi te partial outputs of the asynchronous warpgroup level matmul instructions (WGMMA) from registers to shared memory via a fast CTA-scoped atomic operation with relaxed semantics. To avoid bank conflicts when resetting the number of non-zeros, we minimally pad the TwELL output with four extra elements in the last dimension. In this instance, we found our padding approach to work significantly better than swizzling, due to the lower register pressure introduced in the kernel's epilogue. In an alternative implementation, we also explored a different packing layout, placing the number of non-zeros across the diagonal of the first 32-dimensional subtile to cover all memory banks, an approach we found brought minimal throughput improvements at the cost of extra complexity. We note that for the non-gated variant of our models, we use this same kernel to perform the up projection, as this layer is the one determining the overall sparsity pattern of the tile.

template < const int T_n, const int T_n_compressed, const int NUM_T_n, const int OUT_DIM > _global __launch_bounds__(WARP_SIZE) 8 void mm_t2d_kernel( const __nv_bfloat16\* IN_d, const uint32_t\* GATE_OUT_twell_packed_d, const __nv_bfloat16\* UP_transposed_d, const __nv_bfloat16\* DOWN_d, __nv_bfloat16\* OUT_d { static_assert( (OUT_DIM % "OUT_DIM must be divisible by WARP_SIZE." ); static_assert(T_n_compressed $= =$ WARP_SIZE, "Warp-sync TwELL-to-dense assumes a 32-wide compressed tile."); constexpr int NUM_LOAD_ITERS $\equiv$ OUT_DIM / STRIDE_8xWARP; float OUT_accum[NUM_LOAD_ITERS] $[ 8 ] ~ = ~ \{ 0 . 0 \mathbf { f } \}$ ; __nv_bfloat162 IN_cached[NUM_LOAD_ITERS][4]; $\mathbb { T N } _ { - } \mathbb { d } \ + =$ blockIdx.x \* OUT_DIM $^ +$ threadIdx. $\texttt { x * 8 }$ ; GATE_OUT_twell_packed_d $+ =$ ( blockIdx. $\texttt { x * }$ T_n_compressed $^ *$ NUM_T_n $^ +$ threadIdx.X ); Uw rar er $+ =$ I x. $\texttt { x * 8 }$ ; $+ =$ $\texttt { x * 8 }$ OUT_d $+ =$ blockIdx. $\texttt { x * }$ OUT_DIM $^ +$ threadIdx. $\texttt { x * 8 }$ ;

#pragma unroll   
for (int iter_idx $\qquad = \ 0$ ; iter_idx $<$ NUM_LOAD_ITERS; ++iter_idx) { \*reinterpret_cast<uint4 $\ast >$ (&IN_cached[iter_idx][0]) $=$ \*reinterpret_cast<const uint $4 * >$ ( IN_d $^ +$ iter_idx $^ *$ STRIDE_8xWARP );   
}   
#pragma unroll 1   
for (int tile_idx $\qquad = \ 0$ ; tile_idx $<$ NUM_T_n; $^ { + + }$ tile_idx) { const int lane_tile_register $\equiv$ GATE_OUT_twell_packed_d[tile_idx $^ *$ T_n_compressed]; const int num_nonzeros $\equiv$ __shfl_sync(OxFFFFFFFFu, lane_tile_register, 0);   
#pragma unroll   
for (int iter_idx $\qquad = \ 0$ ; iter_idx $<$ NUM_LOAD_ITERS; ++iter_idx) { const uint4 packed_bfloats_x8 $\equiv$ \*reinterpret_cast<const uint4\*>( UP_transposed_d $^ +$ nonzero_idx $^ *$ OUT_DIM $^ +$ iter_idx $^ *$ STRIDE_8xWARP ); const __nv_bfloat162 packed_bfloats_1 $=$ \*reinterpret_cast<const __nv_bfloat162\*>( &packed_bfloats_x8.x ); __nv_bfloat162 scaled_bfloats_1 = _hmul2(IN_cached[iter_idx][0], packed_bfloats_1); float2 scaled_floats_1 $\equiv$ __bfloat1622float2(scaled_bfloats_1); UP_OUT_accum $+ =$ scaled_floats_ $. 1 . \mathrm { ~ x ~ } + \mathrm { ~ \ }$ scaled_floats_1.y; const __nv_bfloat162 packed_bfloats_ $. 2 \ =$ \*reinterpret_cast<const __nv_bfloat $1 6 2 * >$ ( &packed_bfloats_x8.y ); __nv_bfloat162 scaled_bfloats_ $. 2 \ =$ _hmul2(IN_cached[iter_idx][1], packed_bfloats_2); float2 scaled_floats $. 2 \ =$ __bfloat1622float2(scaled_bfloats_2); UP_OUT_accum $+ =$ scaled_floats_ $_ { 2 . \mathrm { ~ x ~ } + }$ scaled_floats_2.y; const __nv_bfloat162 packed_bfloats_ $_ 3 \ =$ \*reinterpret_cast<const __nv_bfloat162\*>( &packed_bfloats_x8.z ); __nv_bfloat162 scaled_bfloats_ $_ 3 \ =$ _hmul2(IN_cached[iter_idx][2], packed_bfloats_3); float2 scaled_floats_ $. 3 \ =$ __bfloat1622float2(scaled_bfloats_3); UP_OUT_accum $+ =$ scaled_floats_ $. 3 . \mathrm { \textbf { x } } +$ scaled_floats_3.y; const __nv_bfloat162 packed_bfloats_ $. 4 \ =$ \*reinterpret_cast<const __nv_bfloat162\*>( &packed_bfloats_x8.w ); __nv_bfloat162 scaled_bfloats_ $. 4 \ =$ _hmul2(IN_cached[iter_idx][3], packed_bfloats_4); float2 scaled_floats_ $. 4 \ =$ __bfloat1622float2(scaled_bfloats_4); UP_OUT_accum $+ =$ scaled_floats_4 $\cdot \texttt { x } +$ scaled_floats_4.y; } #pragma unroll   
for (int butterfly_stride $\equiv$ WARP_SIZE / 2; butterfly_stride > 0; butterfly_stride $\mathbf { \Psi } / = \mathbf { \Psi } 2$ { UP_OUT_accum $+ =$ __shfl_xor_sync( OxFFFFFFFFu, UP_OUT_accum, butterfly_stride );   
} const __nv_bfloat162 nonzero_feature $=$ bfloat162bfloat162( _hmul( reinterpret_cast<const __nv_bfloat16\*>( &compressed_idx_bf16 )[1], _float2bfloat16_rn(UP_OUT_accum) ) );

#pragma unroll   
for (int iter_idx $\qquad = \ 0$ ; iter_idx $<$ NUM_LOAD_ITERS; ++iter_idx) { const uint4 packed_bfloats_x8 $\equiv$ \*reinterpret_cast<const uint4\*>( DOWN_d $^ +$ nonzero_idx $^ *$ OUT_DIM $^ +$ iter_idx $^ *$ STRIDE_8xWARP ); const __nv_bfloat162 packed_bfloats_1 $=$ \*reinterpret_cast<const __nv_bfloat162\*>( &packed_bfloats_x8.x ); __nv_bfloat162 scaled_bfloats_1 $=$ _hmul2(nonzero_feature, packed_bfloats_1); float2 scaled_floats_1 $\equiv$ __bfloat1622float2(scaled_bfloats_1); OUT_accum[iter_idx][0] $+ =$ scaled_floats_1.x; OUT_accum[iter_idx][1] $+ =$ scaled_floats_1.y;

141   
142 const _nv_bfloat162 packed_bfloats_ $. 2 \ =$   
143 \*reinterpret_cast<const __nv_bfloat $1 6 2 * >$ (   
144 &packed_bfloats_x8.y   
145 );   
146 __nv_bfloat162 scaled_bfloats_ $. 2 \ =$   
147 _hmul2(nonzero_feature, packed_bfloats_2);   
148 float2 scaled_floats_2 $\equiv$ __bfloat1622float2(scaled_bfloats_2);   
149 OUT_accum[iter_idx][2] $+ =$ scaled_floats_2.x;   
150 OUT_accum[iter_idx] [3] $+ =$ scaled_floats_2.y;   
151   
152 const __nv_bfloat162 packed_bfloats_ $. 3 \ =$   
153 \*reinterpret_cast<const___nv_bfloat162\*>(   
154 &packed_bfloats_ $_ { \textrm x 8 . z }$   
155 );   
156 __nv_bfloat162 scaled_bfloats_ $_ 3 \ =$   
157 _hmul2(nonzero_feature, packed_bfloats_3);   
158 float2 scaled_floats $. 3 \ =$ __bfloat1622float2(scaled_bfloats_3);   
159 OUT_accum[iter_idx][4] $+ =$ scaled_floats_3.x;   
160 OUT_accum[iter_idx] [5] $+ =$ scaled_floats_3.y;   
161   
162 const __nv_bfloat162 packed_bfloats_4 $=$   
163 \*reinterpret_cast<const __nv_bfloat162\*>(   
164 &packed_bfloats_x8.w   
165 );   
166 __nv_bfloat162 scaled_bfloats_ $. 4 \ =$   
167 _hmul2(nonzero_feature, packed_bfloats_4);   
168 float2 scaled_floats_4 $\equiv$ __bfloat1622float2(scaled_bfloats_4);   
169 OUT_accum[iter_idx][6] $+ =$ scaled_floats_4.x;   
170 OUT_accum[iter_idx] [7] $+ =$ scaled_floats_4.y;   
171 }   
172 }   
173 }   
174   
175 #pragma unroll   
176 for (int iter_idx $\qquad = \ 0$ ; iter_idx < NUM_LOAD_ITERS; ++iter_idx) {   
177 ._nv_bfloat162 _align__(8) packed_bfloats_x8[4];   
178 packed_bfloats_ $. { \bf x } 8 [ 0 ] =$ __floats2bfloat162_rn(   
179 OUT_accum[iter_idx][0], OUT_accum[iter_idx][1]   
180 );   
181 packed_bfloats_x8[1] $=$ _floats2bfloat162_rn(   
182 OUT_accum[iter_idx][2], OUT_accum[iter_idx][3]   
183 );   
184 packed_bfloats_x8[2] $=$ . _floats2bfloat162_rn(   
185 OUT_accum[iter_idx][4], OUT_accum[iter_idx][5]   
186 );   
187 packed_bfloats_x8[3] $=$ __floats2bfloat162_rn(   
188 OUT_accum[iter_idx] [6], OUT_accum[iter_idx][7]   
189 );   
190   
191 \*reinterpret_cast<uint4\*>(OUT_d $^ +$ iter_idx $^ *$ STRIDE_8xWARP) =   
192 \*reinterpret_cast<uint4 $\ast >$ (packed_bfloats_x8);   
193 }   
194 }

In Figure 2, we provide code listings with the device code for our kernel implementing the custom fused up and down projection kernel that leverages the gate projections stored in the TwELL format. As explained in Section 3 in the main text, this kernel is launched on a grid of warp-sized CTAs and fuses the two operations by keeping in memory the input dense feature row and an accumulator. Then, iterating first statically through the TwELL tiles and then dynamically through the number of non-zeros in each tile, it loads the corresponding gate index, which directly maps to a unique column of the up projection and row of the down projection weight matrices. The kernel computes the up-projected feature from a dot product between the input dense feature row and the up projection weight column, multiplies it by the gate value, and finally uses it to scale the down projection weight row before accumulating the output. To ensure coalesced access, we note that the up projection weight matrix is stored in transposed format. This version of the kernel is specialized to handle the case where $T _ { n } = 2 5 6$ and the compression ratio is 8, leading to a total of 32 elements for each packed TwELL tile. In this specific instance, we load both the number of non-zeros and all the indices and values for the tile in a single fully coalesced access over the CTA's warp, which later allows loading the full TwELL tile information via minimal warp register shuffle operations without incurring any shared memory overheads. In preliminary experiments, we also found that re-ordering the kernel calls in descending order of non-zeros can further accelerate performance with low batch sizes. However, we note that we did not find this optimization necessary with large batches and omitted it for simplicity.

template < const int T_n, const int T_n_compressed, const int NUM_T_n, const int OUT_DIM, const int SPLIT_OUT_DIM > global _launch_bounds__(32) 9 void mm_t2d_kernel( const uint32_t\* IN_twell_packed_d, const __nv_bfloat16\* DOwn_d, __nv_bfloat16\* OUT_d { static_assert( (SPLIT_OUT_DIM % "OUT_DIM must be divisible by WARP_SIZE." ); static_assert( (OUT_DIM % "OUT_DIM must be divisible by SPLIT_OUT_DIM." ); static_assert(T_n_compressed $= =$ WARP_SIZE, "Warp-sync TwELL-to-dense assumes a 32-wide compressed tile."); float OUT_accum[OUT_DIM / STRIDE_8xWARP] [8] $= \ \{ 0 . 0 \mathbf { f } \}$ ; constexpr int NUM_LOAD_ITERS $\equiv$ SPLIT_OUT_DIM / STRIDE_8xWARP; IN_twell_packed_d $+ =$ blockIdx.x \* T_n_compressed $^ *$ NUM_T_n $^ +$ threadIdx.x; DOWN_d $+ =$ threadIdx. $\textbf { x * 8 + }$ blockIdx.y $^ *$ SPLIT_OUT_DIM; OUT_d $+ =$ blockIdx.x \* OUT_DIM $^ +$ threadIdx. $\texttt { x * 8 + }$ blockIdx.y $^ *$ SPLIT_OUT_DIM;

#pragma unroll 1   
for (int tile_idx $\qquad = \ 0$ ; tile_idx $<$ NUM_T_n; $^ { + + }$ tile_idx){ const int lane_tile_register $\equiv$ IN_twell_packed_d[tile_idx $^ *$ T_n_compressed]; const int num_nonzeros $\equiv$ _shfl_sync(OxFFFFFFFFu, lane_tile_register, 0);   
#pragma unroll 1   
for (int idx $\ c = ~ 1$ ; idx $<$ num_nonzeros $^ { + 1 }$ ; $+ + \mathrm { i } \mathrm { d } \mathrm { x }$ { const uint32_t compressed_idx_bf16 $\equiv$ _shfl_sync(OxFFFFFFFFu, lane_tile_register, idx); const uint32_t nonzero_idx $\equiv$ compressed_idx_bf16 & OxFFFFu; const _nv_bfloat162 nonzero_feature $=$ bfloat162bfloat162( reinterpret_cast<const __nv_bfloat16\*>( &compressed_idx_bf16 )[1] );   
#pragma unroll   
for (int iter_idx $\qquad = \ 0$ ; iter_idx $<$ NUM_LOAD_ITERS; ++iter_idx) { const uint4 packed_bfloats_x8 $\equiv$ \*reinterpret_cast<const uint4\*>( DOWN_d $^ +$ nonzero_idx $^ *$ OUT_DIM $^ +$ iter_idx $^ *$ STRIDE_8xWARP ); const __nv_bfloat162 packed_bfloats_1 $=$ \*reinterpret_cast<const__nv_bfloat162\*>( &packed_bfloats_x8.x ); __nv_bfloat162 scaled_bfloats_ $. 1 \ =$ _hmul2(nonzero_feature, packed_bfloats_1); float2 scaled_floats_1 $\equiv$ . _bfloat1622float2(scaled_bfloats_1); OUT_accum[iter_idx][0] $+ =$ scaled_floats_1.x; OUT_accum[iter_idx][1] $+ =$ scaled_floats_1.y; const __nv_bfloat162 packed_bfloats_ $. 2 \ =$ 71 \*reinterpret_cast<const __nv_bfloat162\*>(   
72 &packed_bfloats_x8.y   
73 );   
74 __nv_bfloat162 scaled_bfloats_ $. 2 \ =$   
75 _hmul2(nonzero_feature, packed_bfloats_2);   
76 float2 scaled_floats_2 $\equiv$ __bfloat1622float2(scaled_bfloats_2);   
77 OUT_accum[iter_idx][2] $+ =$ scaled_floats_2.x;   
78 OUT_accum[iter_idx] [3] $+ =$ scaled_floats_2.y;   
79   
80 const __nv_bfloat162 packed_bfloats_ $. 3 \ =$   
81 \*reinterpret_cast<const __nv_bfloat162\*>(   
82 &packed_bfloats_ $. \tt { x 8 . z }$   
83 );   
84 __nv_bfloat162 scaled_bfloats_ $_ 3 \ =$   
85 _hmul2(nonzero_feature, packed_bfloats_3);   
86 float2 scaled_floats_ $. 3 \ =$ . _bfloat1622float2(scaled_bfloats_3);   
87 OUT_accum[iter_idx][4] $+ =$ scaled_floats_3.x;   
88 OUT_accum[iter_idx] [5] $+ =$ scaled_floats_3.y;   
89   
90 const __nv_bfloat162 packed_bfloats_ $. 4 \ =$   
91 \*reinterpret_cast<const__nv_bfloat162\*>(   
92 &packed_bfloats_x8.w   
93 );   
94 __nv_bfloat162 scaled_bfloats_ $. 4 \ =$   
95 _hmul2(nonzero_feature, packed_bfloats_4);   
96 float2 scaled_floats $\underline { { 4 } } \ =$ __bfloat1622float2(scaled_bfloats_4);   
97 OUT_accum[iter_idx][6] $+ =$ scaled_floats_4.x;   
98 OUT_accum[iter_idx][7] $+ =$ scaled_floats_4.y;   
99 }   
100 }   
101 }   
102   
103 #pragma unroll   
104 for (int iter_idx $\qquad = \ 0$ ; iter_idx $<$ NUM_LOAD_ITERS; ++iter_idx) {   
105 __nv_bfloat162 _align__(8) packed_bfloats_x8[4];   
106 packed_bfloats_x8[0] $=$ _floats2bfloat162_rn(   
107 OUT_accum[iter_idx][0], OUT_accum[iter_idx][1]   
108 );   
109 packed_bfloats_x8[1] $=$ - _floats2bfloat162_rn(   
110 OUT_accum[iter_idx][2], OUT_accum[iter_idx][3]   
111 );   
112 packed_bfloats_x8[2] $=$ . _floats2bfloat162_rn(   
113 OUT_accum[iter_idx][4], OUT_accum[iter_idx][5]   
114 );   
115 packed_bfloats_x8[3] $=$ __floats2bfloat162_rn(   
116 OUT_accum[iter_idx][6], OUT_accum[iter_idx][7]   
117 );   
118   
119 \*reinterpret_cast<uint4\*>(OUT_d $^ +$ iter_idx $^ *$ STRIDE_8xWARP) $\equiv$   
120 \*reinterpret_cast<uint4\*>(packed_bfloats_x8);   
121 }   
122 }

As mentioned in Section 2 of the main text, together with modern gated LLMs, we also provide specific kernels that support older non-gated variants, which we empirically evaluate in Appendix C. In Figure 3, we provide code listings with the device code for our kernel implementing the custom down projection kernel that leverages up projection activations stored in the TwELL format for these experiments. Similarly to the fused kernel explained in Section 3 and examined above, this kernel is launched on a grid of warp-sized CTAs and reads the sparsity pattern, this time from the up projection activations stored in the TwELL format. This time, the kernel maintains in memory the out projection and a float32 accumulator for a small output segment. Then, it first statically iterates through the TwELL tile and then dynamically iterates over the number of non-zeros in each tile. At each iteration, it loads the non-zero index and the corresponding activation down projection column segment, before multiplying the two and accumulating the result. In contrast to our fused kernel, where we have to consider full rows on the input and output to perform dot products between the input features and the up projection weights, introducing the split formulation in this kernel is a deliberate and purposeful choice: by introducing trivial duplication of the non-zero reads we can further increase parallelism, reduce register pressure, increase occupancy, and hide longer latencies from uneven sparsity. In practice, we note that using a split dimension of half the base output dimension, leading to two CTAs per output row, appears optimal on our Hopper GPUs.

# A.2. Training Kernels Selection global void twell_to_ell_kernel(   
2 const __nv_bfloat16\* _restrict__ C_vals,   
3 const uint8_t\* __restrict__ C_idx,   
4 const uint32_t\* __restrict__ C_nnz,   
5 __nv_bfloat16\* __restrict__ ell_val,   
int16_t\* restrict ell_col,   
int32_t\* __restrict__ row_nnz,   
8 float\* __restrict__ l0_out,   
9 float\* _restrict_ 11out,   
10 int M,   
11 int N_TILES,   
12 int BW,   
13 int  EL_,   
14 int T_n   
15   
16 {   
17 const int row $\equiv$ blockIdx.x \* blockDim.y $^ +$ threadIdx.y;   
18 if (row $> = \mathrm { ~ \mathbb { M } ~ }$ {   
19 return;   
20 }   
21   
22 const int tid $\equiv$ threadIdx.x;   
23 int cnt $\equiv$ (tid $< ~ 1$ V_TILES) ? C_nnz[(size_t)tid \* M $^ +$ row] : 0;   
24 cnt $\equiv$ min(cnt, BW);   
25   
26 int offset $\equiv$ cnt;   
27 for (int delta $\ c = ~ 1$ ; delta $<$ WARP_SIZE; delta $< < = ~ 1$ {   
28 const int recv $\equiv$ __shfl_up_sync(OxFFFFFFFu, offset, delta);   
29 if (tid $> =$ delta) $\bar { \boldsymbol { \updownarrow } }$   
30 offset $+ =$ recv;   
31 }   
32 }   
33   
34 const int start $\equiv$ offset - cnt;   
35 const int total $\equiv$   
36 __shfl_sync(OxFFFFFFFFu, offset, min(N_TILES - 1, WARP_SIZE - 1));   
37   
38 const __nv_bfloat16\* sv $\equiv$   
39 C_vals $^ +$ (size_t)row $^ *$ N_TILES $^ *$ BW $^ +$ (size_t)tid $^ *$ BW;   
40 const uint8_t\* si $\equiv$   
41 C_idx $^ +$ (size_t)row $^ *$ N_TILES $^ *$ BW $^ +$ (size_t)tid $^ *$ BW;   
42   
43 float 10_acc $= \ 0 . 0 \mathbf { f }$ ;   
44 float 11_acc $= \ 0 . 0 \mathbf { f }$ ;   
45 if (10_out) {   
46 const float inv_ $\mathrm { ~  ~ \cal ~ M ~ } = \mathrm { ~  ~ 1 ~ } . 0 \dot { \bf f }$ / (float)M;   
47 10_acc $=$ (float)cnt $^ *$ inv_M;   
48 for (int $\mathrm { ~ \tt ~ { ~ i ~ } ~ } = \mathrm { ~ 0 ~ }$ ; i < cnt; $^ { + + \dot { \bf 1 } }$ {   
49 11_acc $+ =$ __bfloat162float(sv[i]) \* inv_M;   
50 }   
51 }   
52   
53 if (cnt > O && start $<$ ELL_W){   
54 const int copy_n $\equiv$ min(cnt, ELL_W - start);   
55 _nv_bfloat $^ { 1 6 * }$ dv $\equiv$ ell_val $^ +$ (size_t)row $^ *$ ELL_W + start;   
56 int1tt\*dc $\equiv$ ell_col $^ +$ (sizet)row $^ *$ ELL_W $^ +$ start;   
57 for (int $\mathrm { ~ \tt ~ { ~ i ~ } ~ } = \mathrm { ~ 0 ~ }$ ; i < copy_n; $+ + \dot { 1 }$ {   
58 dv [i] $=$ sv[i];   
59 dc[i] $=$ (int16_t)(si[i]) + (int16_t)(tid \*T_n);   
60 }   
61 }   
62   
63 if $\mathbf { \Psi } _ { \mathbf { { t i d } } } ~ = = ~ 0 .$ {   
64 row_nnz[row] $\equiv$ total;   
65 }   
66   
67 if (10_out){   
68 for (int $\mathrm { ~ s ~ } = \ 1 6$ ; s > 0; s $> > = ~ 1 \dot { }$ {   
69 10_acc $+ =$ __shfl_down_sync(OxFFFFFFFFu, 10_acc, s);

<table><tr><td>70</td><td>11_acc += __shfl_down_sync(OxFFFFFFFFu, 11_acc, s);</td><td></td></tr><tr><td>71</td><td>}</td><td></td></tr><tr><td>72</td><td>if (tid == 0){</td><td></td></tr><tr><td>73</td><td>atomicAdd(10_out, 10_acc);</td><td></td></tr><tr><td>74</td><td>if (11_out) {</td><td></td></tr><tr><td>75</td><td>atomicAdd(l1_out, 11_acc);</td><td></td></tr><tr><td>76</td><td>}</td><td></td></tr><tr><td>77</td><td>}</td><td></td></tr><tr><td>78</td><td>}</td><td></td></tr><tr><td>79</td><td></td><td></td></tr></table>

Listing 4 | Conversion from TwELL to the hybrid format logic.

In Figure 4, we provide code listings with the device code for our training kernel used to convert gate activations stored in the TwELL format into the compact ELL component of our hybrid training representation while accumulating $L _ { 0 }$ and $L _ { 1 }$ statistics. As discussed in Section 3, the conversion dynamically partitions the rows based on the non-zero counts. We allocate a warp to each row, and let each thread read the number of active entries in a single tile. We then use warp register shuffles to obtain an inclusive prefix scan and determine the starting offset of that tile within the destination ELL row. This design allows for directly compacting the tiled representation into contiguous row-wise ELL storage without requiring any synchronization beyond warp-level or shared memory accesses. The kernel writes the true row occupancy to row_nnz even when the row exceeds the configured ELL width ELL_W, allowing overflow rows to be detected and promoted to the dense tail of the hybrid format. During training, each warp also reduces simple $L _ { 0 }$ and $L _ { 1 }$ sparsity statistics to compute the sparsity levels and L1 loss before issuing a single atomic update.

_global_ void matmul_save_sparse_like_ell(   
2 bfloat16\* A,   
3 bfloat16\* B_T,   
4 ELLL\* out,   
5 int M,   
6 int K,   
int N   
8 )   
9 {   
10 const int row $\equiv$ blockIdx.x;   
11 const int ell_n $\equiv$ out->row_counts[row];   
12 if $\mathbf { \hat { \Pi } } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde \Pi } } \mathbf { \Pi } _ { \mathbf { \tilde \Pi { \Pi } } } \mathbf { \Pi \tilde { \Pi } } _ { \mathbf { \tilde \Pi } } \mathbf { \Pi } _ { \mathbf { \tilde \Pi } } \mathbf { \Pi \tilde { \Pi } } _ \mathbf { \tilde \Pi } _ { \mathbf \tilde { \Pi \Pi } } \mathbf { \Pi } _ \mathbf { \tilde \tilde { \Pi \Pi } } _ \mathbf { \tilde \tilde \Pi } _ { \mathbf \tilde { \Pi \Pi } \Psi } _ \mathbf { \tilde \Psi \Psi } _ \mathbf { \Psi \Psi \Psi } _ \mathbf { \Psi \Psi \Psi } \mathbf \Psi \Psi _ { \Psi \Psi \Psi } \mathbf \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi $ ll ell_n $>$ ELL_WIDTH){   
13 return;   
(14 }   
15   
16 bfloat16\* A_row_ptr $\equiv$ A + row \* K;   
17 const int lane_id $\equiv$ threadIdx.x & 31;   
18 const int warp_id $\equiv$ threadIdx. $\texttt { x } > > 5$ ;   
19 const int num_warps $\equiv$ blockDim. $\texttt { x } > > 5$ ;   
20 const int num_chunks $= \texttt { K } / \texttt { 8 }$ ;   
21   
22 for (int out_idx $\equiv$ warp_id; out_idx $<$ ell_n; out_idx $+ =$ num_warps) {   
23 const int col $\equiv$ out->cols [row $^ *$ ELL_WIDTH $^ +$ out_idx];   
24 bfloat $^ { 1 6 * }$ B_row_ptr $= \texttt { B } _ { - } \texttt { T }$ $^ +$ col $* \texttt { K }$ ;   
25 float acc $= \ 0 . 0 \mathbf { f }$ ;   
26   
27 for (int chunk_base $\qquad = \ 0$ ; chunk_base $<$ num_chunks; chunk_base $+ = ~ 3 2$ {   
28 const int chunk $=$ chunk_base $^ +$ lane_id;   
29 if (chunk $> =$ num_chunks){   
30 break;   
31 }   
32   
33 int4 a_raw $\qquad = ~ *$ (int4\*)(A_row_ptr $^ +$ chunk $* \ 8 \mathrm { , }$ ;   
34 int4 b_raw $\qquad = ~ *$ (int4\*)(B_row_ptr $^ +$ chunk $* \ 8 )$ ;   
35 bfloat16_2\* a_vec $=$ bfloat16) $^ { 2 * }$ )&a_raw;   
36 bfloat16_2\* b_vec $=$ (bfloat16_2\*)&b_raw;   
37   
38 for (int $\texttt { t } = \texttt { 0 }$ ; t < 4; $+ + \mathtt { t }$ {   
39 float2 af $\equiv$ bfloat1622float2(a_vec[t]);   
40 float2 bf $\equiv$ bfloat1622float2(b_vec[t]);   
41 acc $\equiv$ fmaf(af.x, bf.x, acc);   
42 acc $\equiv$ fmaf(af.y, bf.y, acc);   
43 }   
44 }   
45   
46 for (int offset $\ l = \ 1 6$ ; offset $> ~ 0$ ; offset $> > = ~ 1$ {   
47 acc $+ =$ __shfl_xor_sync(OxFFFFFFFFu, acc, offset);   
48 }   
49 if (lane_id $\scriptstyle = = 0$ {   
50 out->vals[row $^ *$ ELL_WIDTH $^ +$ out_idx] $=$ float2bfloat16(acc);   
51 }   
52 }   
53

In Figure 5, we provide code listings of our custom kernel used to perform the efficient denseto-hybrid matmuls used during training, focusing on the sparse component. This kernel shows the logic of the dynamic hybrid partitioning, skipping the sparse operation in the non-zeros is recognized to exceed the size of the aggressively compact ELL format. The kernel takes two dense matrices, A and $B$ (provided as $B _ { T } \mathrm { ~ . ~ }$ ), and a pre-allocated ELL output "out" of shape $M \times N$ , whose column indices encode the sparsity pattern to be evaluated. Rather than computing all MN outputs, each thread block processes a single output row and iterates only over the column indices stored for that row in out. For each selected column, the kernel computes the dot product between $A \left[ r o w , : \right]$ and $B _ { T } [ c o l , : ]$

To maximize coalescing and enable vectorized memory accesses, $B$ is stored transposed so that rows of $B _ { T }$ are contiguous in memory. To maximize throughput with bfloat16, threads load $A$ and $B _ { T }$ in 128-bit transactions (8 bfloat16 values at a time) and accumulate in float32 using fused multiply-adds. Each warp reduces partial sums using shuffle-based reduction, and the final value is written to the corresponding slot in the ELL value array. This design aligns with ELL's row-oriented storage: the sparsity pattern is known up front, so the kernel avoids both dense materialization and irregular gathers beyond the indexed rows of $B _ { T }$ . In the forward pass, we use this kernel to compute only the entries of the up projection operation $x W _ { u }$ that willsurvive the subsequent gating, by copying the sparsity pattern from $h _ { g }$ into out and filling its values with the corresponding dot products. In the backward pass, the same kernel is reused for masked gradient matmuls that share a known sparsity pattern. For instance, we use it to compute $\nabla h = \nabla y , W _ { d } ^ { T }$ . Rows that exceed the ELL capacity are handled by routing the overflow to the dense backup matrix and computing that portion with Tensor Coreoptimized kernels, as described in Algorithm 3, and they are multiplied by a binary mask containing the sparsity pattern to be applied.

global void hybrid_to_dense_mamtul(   
2 ELL\*A,   
3 bfloat16\* B,   
4 bfloat16\* C,   
5 int M,   
6 int N,   
int K   
8 )   
9 {   
10 const int row $\equiv$ blockIdx.x;   
11 const int ell_n $\equiv$ A->row_counts[row];   
12 if $\mathbf { \hat { \Pi } } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde { \Pi } } } \mathbf { \Pi } _ { \mathbf { \tilde \Pi } } \mathbf { \Pi } _ { \mathbf { \tilde \Pi { \Pi } } } \mathbf { \Pi \tilde { \Pi } } _ { \mathbf { \tilde \Pi } } \mathbf { \Pi } _ { \mathbf { \tilde \Pi } } \mathbf { \Pi \tilde { \Pi } } _ \mathbf { \tilde \Pi } _ { \mathbf \tilde { \Pi \Pi } } \mathbf { \Pi } _ \mathbf { \tilde \tilde { \Pi \Pi } } _ \mathbf { \tilde \tilde \Pi } _ { \mathbf \tilde { \Pi \Pi } \Psi } _ \mathbf { \tilde \Psi \Psi } _ \mathbf { \Psi \Psi \Psi } _ \mathbf { \Psi \Psi \Psi } \mathbf \Psi \Psi _ { \Psi \Psi \Psi } \mathbf \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi \Psi $ ll ell_n $>$ ELL_WIDTH){   
13 return;   
14 }   
15   
16 bfloat16\* A_row_vals $\equiv$ A->vals $^ +$ row $^ *$ ELL_WIDTH;   
17 uint16_t\* A_row_idxs $\equiv$ A->cols $^ +$ row $^ *$ ELL_WIDTH;   
18   
19 for (int n_out $=$ threadIdx. $\texttt { x * 8 }$ ; n_out $< \mathbb { N }$ ; n_out $\scriptstyle + = \ 8 \ *$ blockDim.x) {   
20 float2 acc[4];   
21 for (int $\mathrm { ~ \tt ~ { ~ i ~ } ~ } = \mathrm { ~ 0 ~ }$ ; $\dot { \mathrm { ~ \scriptsize ~ 1 ~ } } < \mathrm { ~ \scriptsize ~ 4 ~ }$ ; $^ { + + \dot { \bf 1 } }$ ) {   
22 acc [i] $\equiv$ make_float2(0.f, 0.f);   
23 }   
24   
25 for (int $\mathtt { k } \ = \ 0$ ; k < ELL_WIDTH; $+ + \mathbf { k } { \mathrm { . } }$ {   
26 if $( \tt k > = e l l _ { - } n )$ {   
27 break;   
28 }   
29   
30 const bfloat16 a_val $=$ A_row_vals[k];   
31 const uint16_t col_idx $=$ A_row_idxs[k];   
32 bfloat $^ { 1 6 * }$ B_row_ptr $=$ B $^ +$ col_idx \* N + n_out;   
33 int4 b_vec_raw $\qquad = ~ *$ (int4\*)(B_row_ptr);   
34 bfloat $^ { 1 6 2 * }$ b_vec $=$ (bfloat162\*)(&b_vec_raw);   
35 const float a $=$ bfloat162float(a_val);   
36   
37 for (int $\texttt { t } = \texttt { 0 }$ ; $\ t \ < \ 4$ ; $+ + \mathtt { t }$ {   
38 float2 b_f32 $\equiv$ bfloat1622float2(b_vec[t]);   
39 acc[t]. $\begin{array} { r l } { \mathbf { x } } & { { } = } \end{array}$ fmaf(a, b_f32.x, acc[t].x);   
40 acc [t] $\texttt { y } =$ fmaf(a, b_f32.y, acc[t].y);   
41 }   
42 }   
43   
44 bfloat162\* C_ptr $\equiv$ (bfloat162\*)(C + row \* N + n_out);   
45 for (int $\mathrm { ~ \tt ~ { ~ i ~ } ~ } = \mathrm { ~ 0 ~ }$ ; i < 4; $^ { + + \dot { \bf 1 } }$ {   
46 C_ptr[i] $\equiv$ float22bfloat162(acc[i]);   
47 }   
48 }   
4 Listing 6 | Hybrid-to-dense sparse matmul using the ELL component during training using CUDA cores.

In Figure 6, we provide code listings of our custom kernel used to perform the efficient hybrid-todense used during training, focusing on the sparse component. Again, this kernel implements the same dynamic hybrid partitioning logic, skipping the sparse operation in the non-zeros is recognized to exceed the size of the aggressively compact ELL format. In particular, the kernel computes a sparse—dense matrix multiplication $C = A B$ , where $A$ is stored in ELL format and $B$ and $C$ are dense row-major matrices. The kernel maps one CTA per output row of $C$ , which aligns naturally with ELL's row-wise storage and lets the CTA reuse the same sparse row metadata while sweeping across the output columns. Within a CTA, threads partition the output row into contiguous column tiles. For each tile, they iterate over the non-zeros in the corresponding ELL row of $A$ and accumulate contributions of the form $a \cdot B [ c o l _ { i } d x , : ]$ into $C \left[ r o w , : \right]$ . To maximize memory throughput for bfloat16, the kernel accesses $B$ using 128-bit SIMD loads, so that each thread processes 8 output elements at a time. Accumulation is performed in float32, and the results are written back in vectorized form, providing a simple and efficient SpMM for the fixed-width ELL layout. Rows that exceed the ELL capacity are handled by routing the overflow to the dense backup matrix and computing that portion with Tensor Coreoptimized kernels, as described in Algorithm 3. This kernel is used in the forward pass to compute the feedforward layer output. In the backward pass, it is also used to compute gradients with respect to the layer parameters as well as the input activations.

global void hybrid_transpose(   
2 ELL\*A,   
3 ELL\* A_T,   
4 bfloat16\* tail_A,   
5 bfloat16_t\* tail_A_T,   
6 int M,   
7 int N   
8 )   
9 {   
10 for (int row $=$ blockIdx.x; row $< ~ \mathbb { M }$ ; row $+ =$ gridDim.x){   
11 const int nnz_row $=$ A->row_counts[row];   
12 if (nnz_row $\scriptstyle = = 0$ ll nnz_row $>$ ELL_WIDTH){   
13 continue;   
14 }   
15   
16 for (int $\mathrm { ~ \bf ~ k ~ } =$ threadIdx.x; k < nnz_row; k $+ =$ blockDim.x) {   
17 const uint16_t col $\equiv$ A->cols [row $^ *$ ELL_WIDTH $^ +$ k];   
18 const bfloat16 val $\equiv$ A->vals [row $^ *$ ELL_WIDTH $^ +$ k];   
19 const int out_row $=$ col;   
20 const int out_col $=$ row;   
21 const int pos $\equiv$ atomicAdd(A_T->row_counts[out_row], 1);   
22   
23 if (pos $<$ ELL_WIDTH){   
24 const size_t addr $=$ out_row $^ *$ ELL_WIDTH $^ +$ pos;   
25 A_T->cols [addr] $=$ out_col;   
26 A_T->vals [addr] $=$ val;   
27 } else {   
28 const int d_row $\equiv$   
29 get_or_allocate_dense_row(out_row, A_T->tail_map);   
30 tail_A_T[d_row $^ *$ M $^ +$ out_col] $=$ val;   
31 }   
32 }   
33 }   
35 for (int d_row $=$ blockIdx.x; d_row $<$ A->tail_rows; d_row $+ =$ gridDim.x){   
36 const int src_row $\mathbf { \lambda } = \mathbf { \lambda } \mathsf { A } - \mathbf { \boldsymbol { > } } \mathbf { \lambda }$ tail_map_reverse[d_row];   
bfloat16_t\* src $\equiv$ tail_A $^ +$ d_row $\ast \intercal$ ;   
38   
39 for (int colo $\equiv$ threadIdx. $\texttt { x * 8 }$ ; $\mathsf { c o l o \mathrm { ~ < ~ } N }$ ; colo $+ =$ blockDim. $\texttt { x * 8 }$ {   
40 int4 raw $\mathbf { \mu } = \mathbf { \sigma } * ( \mathrm { i n t } 4 * )$ (src $^ +$ colo);   
41 if ((raw.x | raw.y Ⅰ raw.z | raw.w) == O) {   
42 continue;   
43 }   
44   
45 for (int $\mathrm { ~ \tt ~ { ~ i ~ } ~ } = \mathrm { ~ 0 ~ }$ ; $\textup { i } < \textup { 8 }$ ; $+ + \dot { 1 }$ {   
46 const bfloat16_t val $=$ unpack_element(&raw,i);   
47 if (val $\scriptstyle = = \ 0 . 0 \mathrm { f }$ {   
48 continue;   
49 }   
50   
51 const int out_row $\equiv$ $\mathbf { \lambda } = \mathbf { \lambda } _ { \mathsf { C O I O } } \mathbf { \lambda } + \mathbf { \lambda } \mathbf { i }$ ;   
52 const int out_col $\equiv$ src_row;   
53 const int pos $=$ atomicAdd(A_T->row_counts[out_row], 1);   
54   
55 if (pos < ELL_WIDTH) {   
56 const size_t addr $=$ out_row $^ *$ ELL_WIDTH $^ +$ pos;   
57 A_T->cols [addr] $\equiv$ out_col;   
58 A_T->vals[addr] $\equiv$ val;   
59 } else {   
60 const int dense_row $\equiv$   
61 get_or_allocate_dense_row(out_row, A_T->tail_map);   
62 tail_A_T[dense_row $^ *$ M $^ +$ out_col] $\equiv$ val;   
63 }   
64 }   
65 }   
66 }   
67 } Listing 7 | Transposition of the hybrid sparse used during training.

In Figure 7, we provide code listings of our custom kernel used to perform efficient transposition of a matrix stored in our hybrid training format. The kernel takes as input a matrix A and produces $A _ { T }$ in the same representation: an ELL matrix, plus a dense backup for rows that overflow the maximum number of non-zeros. It operates in two parts. First, it transposes the non-overflow rows stored in the ELL structure by iterating over each row's non-zeros and inserting them into the corresponding row of $A _ { T }$ (since a non-zero at (row, col) becomes an entry in row col of the transpose). Because many source rows may map to the same destination row, the kernel uses atomic increments to reserve an insertion slot. If the destination row still has capacity, the entry is written into the ELL arrays of $A _ { T }$ ; otherwise, it is routed to the dense backup, using a per-row mapping that allocates a dense-tail row on demand. Second, it handles the overflow rows that are materialized in the dense tail of A. These rows are scanned in vectorized chunks (128-bit loads, i.e., 8 bfloat16 values at a time) with a fast zero check to skip all-zero vectors. Only non-zero elements are emitted into $A _ { T }$ using the same hybrid partitioning logic. This approach keeps transposition efficient while preserving the hybrid format and avoiding expensive conversions to more general sparse layouts. After this kernel completes, we launch a small helper kernel to copy the entries stored in the ELL arrays for rows that overflowed into the corresponding dense-backup rows. We note that the necessity of this final small step comes from the fact that dense rows are only allocated and populated after the ELL slots for a given output row have been exhausted.

# B. Hyperparameters and Datasets

# B.1. Training Details

Table 2 | Default Hyperparameters for Pretraining Sparse and Non-Sparse LLMs.

<table><tr><td>Hyperparameter</td><td>Gated LLM</td><td>Non-Gated LLM</td></tr><tr><td colspan="3">Model architecture</td></tr><tr><td>Hidden size</td><td>2048</td><td>2048</td></tr><tr><td>Hidden MLP size (intermediate)</td><td>5632</td><td>8192</td></tr><tr><td>Gated MLP</td><td>true</td><td>false</td></tr><tr><td>Hidden activation</td><td>ReLU</td><td>ReLU</td></tr><tr><td>Number of hidden layers</td><td>8/18/28/38</td><td>8/18/28/38</td></tr><tr><td>Number of attention heads</td><td>32</td><td>32</td></tr><tr><td>Number of keyvalue heads</td><td>32</td><td>32</td></tr><tr><td>Head dimension</td><td>64</td><td>64</td></tr><tr><td>Attention bias</td><td>false</td><td>false</td></tr><tr><td>Attention dropout</td><td>0.0</td><td>0.0</td></tr><tr><td>Initializer range</td><td>0.02</td><td>0.02</td></tr><tr><td>RoPE θ</td><td>10,000</td><td>10,000</td></tr><tr><td>Tied word embeddings Vocabulary size</td><td>true</td><td>true</td></tr><tr><td>Tokenizer</td><td>49,152</td><td>49,152</td></tr><tr><td></td><td>GPT2</td><td>GPT2</td></tr><tr><td>Computation dtype</td><td>bfloat16</td><td>bfloat16</td></tr><tr><td>MLP bias</td><td>false</td><td>false</td></tr><tr><td colspan="3">Training setup</td></tr><tr><td>Dataset</td><td>fineweb</td><td>finewebB</td></tr><tr><td>Maximum sequence length</td><td>2048</td><td>2048</td></tr><tr><td>Tokens per training step</td><td>1,048,576</td><td>1,048,576</td></tr><tr><td>Training steps</td><td>10K/20K/30K/40K</td><td>10K/20K/30K/40K</td></tr><tr><td>Total training tokens</td><td>10.49B/20.97B/31.46B/41.94B</td><td>10.49B/20.97B/31.46B/41.94B</td></tr><tr><td colspan="3">Optimization</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>1.0 × 10-3</td><td>1.0 × 10-3</td></tr><tr><td>Weight decay</td><td>0.1</td><td>0.1</td></tr><tr><td>Adam parameters (β1, β2, )</td><td>(0.9, 0.95, 1×10-8)</td><td>(0.9, 0.95, 1×10-8)</td></tr><tr><td>Learning rate scheduler</td><td>Cosine decay</td><td>Cosine decay</td></tr><tr><td>Warmup steps</td><td>600</td><td>600</td></tr><tr><td>Max grad norm</td><td>1.0</td><td>1.0</td></tr></table>

As explained in Section 4 of the main text, our sparse models and dense baselines in the main text implement a "Transformer $+ + { } ^ { \mathfrak { n } }$ architecture with gated feedforward blocks, as common in recent LLMs such as Qwen and Llama (Touvron et al., 2023; Yang et al., 2024). Moreover, in Appendix C, we also collect results on a non-gated variant of the same architecture, more similar to the traditional architecture, more similar to the original transformer design (Vaswani et al., 2017). We train all models using the fineweb (Penedo et al., 2024). In particular, we consider a deduplicated version of the fineweb-edu split, obtained by from an open corpus used to pretrain the SmolLM family of models (Allal et al., 2024). We note that all our models are trained with the chinchilla-optimal number of tokens (Hoffmann et al., 2022): around 10B tokens for our 0.5B models, 20B tokens for our 1B models, 30B tokens for our 1.5B models, and 40B tokens for our 2B models.

We provide a full list of hyperparameters and training specifications for our training settings and models in Table 2. For all models, we use context lengths of 2048 tokens, with batches of 512 sequences, resulting in a global batch size of 1M tokens. For our gated variant, we use a dimensionality of 2048 and a hidden dimension of 5632 in the feedforward blocks, roughly eight-thirds of the hidden size. The main difference with the non-gated variant is that we use a much larger intermediate size of 8192, four times the hidden size, leading to the same total number of parameters. We note that both these choices are considered optimal in the current literature with larger model design practices. When varying model sizes, we modify the number of layers to achieve the target parameter counts. In practice, modern small models have also considered shifting even more of the parameters and flops to the feedforward blocks: for instance, the 1B model of the llama 3 family has a feedforward size of $_ { 4 \mathrm { X } }$ the hidden size even while implementing the gated design (Grattafiori et al., 2024). While the gains from our sparse kernels could be even greater in these settings, we opted for a more conservative design to avoid biasing our results toward smaller models. To optimize our models, we use the AdamW optimizer (Loshchilov and Hutter, 2017) with a weight decay of 0.1 and a cosine learning rate schedule starting from a peak learning rate of $1 . 0 \times 1 0 ^ { - 3 }$ , after a small warmup of 600 steps. We use the default Adam parameters of $( \beta _ { 1 } , \beta _ { 2 } , \epsilon ) = ( 0 . 9 , 0 . 9 5 , 1 \times 1 0 ^ { - 8 } )$ and clip gradients at a maximum norm of 1.0. Our vocabulary of tokens comes from a GPT2 tokenizer (Radford et al., 2019). We train using standard mixed precision with the bfloat16 format, with our optimizer states stored in full precision.

# B.2. Task Evaluation Details

We evaluate our models using cloze-formulation scores on seven standard downstream multiplechoice benchmarks that probe logical and commonsense reasoning after pretraining. In particular, we consider ARC (Easy and Hard versions) (Clark et al., 2018), a grade-school science question answering benchmark comprising both Easy and Challenge subsets, with the latter designed to defeat simple retrieval- and co-occurrence-based baselines; HellaSwag (Zellers et al., 2019), a commonsense sentence completion task that was designed for counterintuitive LLM challenge; OpenBookQA (Mihaylov et al., 2018), focused on probing curated sets of science-based and commonsense knowledge; PIQA (Bisk et al., 2020), a benchmark benchmark focused on physical commonsense reasoning; WinoGrande (Sakaguchi et al., 2021), a Winograd-style large-scale conference benchmark; and CommonsenseQA (Talmor et al., 2019), evaluating broader commonsense reasoning. We follow standard evaluation protocols and hyperparameters for formatting the input questions.

# B.2.1. Sparse data structures sizing

We note that the hybrid training format proposed in this paper introduces two core hyperparameters necessary to fulfillits targeted static allocation design: the ELL maximum number of elements per row, and the number of rows in the dense matrix that stores overflowing elements. Both hyperparameters effectively control a trade-off between performance and memory savings, making their value partially dependent on the sparsity level. Moreover, because sparsity can change abruptly during training, we evaluate a set of sizes that can tolerate sudden decreases in sparsity while remaining performant. In practice, we find that setting the maximum number of elements to 128, and the maximum number of backup rows to one-eighth of the token batch size, to be a robust choice for all sparsity levels above $1 . 5 \times 1 0 ^ { - 5 }$ .Moreover, below this point, simply doubling the ELL non-zeros prevents other instabilities. However, we note that with prior knowledge of the sparsity evolution, these structures can often be made smaller within training itself. For example, for $L _ { 1 } = 1 \times 1 0 ^ { - 4 }$ , we observe that after training stabilizes, we can reduce the number of rows in the dense overflow matrix to 512, enabling higher speedups and additional memory savings. Moreover, the requirements on these two limits differ between the forward and backward passes due to the sparse-matrix transposition used in backpropagation. We note that relevant future extensions could characterize these requirements and develop online tuning of these hyperparameters to improve performance and memory savings further. Finally, when the number of elements exceeds the capacity of our data structures, we currently discard the excess values to avoid a hard failure and set a flag that is reported to the CPU at the next GPU synchronization point. This allows the training system to adaptively increase the structure sizes and repeat the latest training optimization step to avoid any deterioration in the learning dynamics. Table 3 | Comparison of performance and efficiency statistics of sparse LLMs leveraging our kernels with traditional gated models using both ReLU and SiLU activations (Hendrycks, 2016; Ramachandran et al., 2017; Shazeer, 2020).   

<table><tr><td>Model scale Activation Sparse L1 coeff. accuracy</td><td></td><td></td><td></td><td>Mean task</td><td></td><td></td><td colspan="4">Forward execution Energy per token</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Cross-entropy # non-zeros</td><td></td><td colspan="2">(input tokens/ms)</td><td colspan="2">(mJ)</td></tr><tr><td rowspan="3">1.5B params 30B tokens</td><td>ReLU</td><td>X</td><td>0</td><td>46.4%</td><td>2.255</td><td>911</td><td>117.1</td><td>(0.0%)</td><td>5.77</td><td>(0.0%)</td></tr><tr><td>SiLU</td><td>X</td><td>0</td><td>47.1%</td><td>2.240</td><td>5632</td><td>116.5</td><td>(-0.5%)</td><td>5.82</td><td>(+0.1%)</td></tr><tr><td>ReLU</td><td>✓</td><td>2×10-5</td><td>46.2%</td><td>2.297</td><td>29</td><td>138.0 (+17.9%)</td><td></td><td>5.07</td><td>7 (-12.1%)</td></tr></table>

Table 4 | Comparison of performance and efficiency statistics of sparse LLMs leveraging our kernels with traditional baselines, considering both gated models (Shazeer, 2020), and their original nongated counterparts used in the original transformer (Vaswani et al., 2017).

<table><tr><td colspan="5"></td><td colspan="2">Forward execution (input tokens/ms)</td><td colspan="2">Energy per token (mJ)</td></tr><tr><td>Model scale</td><td>Gated</td><td>Sparse X</td><td>L1 coefficient 0</td><td>Mean task accuracy 46.36%</td><td>117.1</td><td>(0.0%)</td><td>5.79</td></tr><tr><td rowspan="3">1.5B params 30B tokens</td><td>✓</td><td>✓</td><td>2× 10-5</td><td>46.20%</td><td>138.0 (+17.9%)</td><td>5.07</td><td>(0.0%) (-12.5%)</td></tr><tr><td></td><td>✓</td><td>3 ×10-5</td><td>44.83%</td><td>147.0 (+25.5%)</td><td>4.75</td><td>(-18.0%)</td></tr><tr><td></td><td></td><td>0</td><td>46.57%</td><td></td><td></td><td></td></tr><tr><td rowspan="3">1.5B params 30B tokens</td><td>X</td><td>X V</td><td>2×10-5</td><td>46.46%</td><td>125.8 (0.0%) 139.9 (+11.2%)</td><td>5.52 5.03</td><td>(0.0%) (-8.8%)</td></tr><tr><td></td><td>√</td><td>3 × 10-5</td><td>44.71%</td><td>142.3 (+13.1%)</td><td>4.86</td><td>(-12.0%)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

# C. Parameter Studies and Ablations

# C.1. Performance and Efficiency Across Activation Functions

As noted in Section 2, many recent LLM architectures have deviated from using ReLUs in favor of smoother activation functions such as SiLU (Hendrycks, 2016; Ramachandran et al., 2017). To provide a direct comparison between the two activations, we collect additional training runs on 30B tokens with our 1.5B model and collect efficiency and performance results. In Table 3, we find that, while final cross-entropy appears equivalent, SiLU activations indeed yield slightly higher task accuracy in our evaluation set. However, we note that SiLU LLMs are already marginally slower than non-sparse ReLU LLMs by $0 . 5 \%$ , and due to their inherent non-sparse nature, they cannot support integration with sparsity and, therefore, the benefits of our kernels. Overall, we find these results are consistent with the ones from Mirzadeh et al. (2023) using larger OPT models (Zhang et al., 2022a)  appearing to indicate that the advantages of smooth activation functions are minor and could potentially be offset by efficiency considerations. Table 5 | Comparison of performance and efficiency statistics of sparse LLMs leveraging our kernels with traditional baselines trained using our standard recipe, or with dead neuron mitigation strategies such as warming up the coefficient of the L1 loss and applying targeted reinitialization to the gate projection's weights.   

<table><tr><td rowspan="2">Model scale</td><td rowspan="2">Training modification</td><td colspan="5">Mean task Sr1 ce  Co no(/)</td><td colspan="3">Forward execution Energy per token</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>(mJ)</td></tr><tr><td rowspan="4">1.5B params 30B tokens</td><td></td><td>X</td><td>0</td><td>46.4%</td><td>2.255</td><td>911</td><td>117.1</td><td>(0.0%)</td><td>5.77 (0.0%)</td></tr><tr><td></td><td>✓</td><td>2 × 10-5</td><td>46.2%</td><td>2.297</td><td>29</td><td>138.0 (+17.9%)</td><td></td><td>5.07 (-12.1%)</td></tr><tr><td>Dead neuron reinit.</td><td>✓</td><td>2×10-5</td><td>46.6%</td><td>2.298</td><td>29</td><td>139.4 (+19.1%)</td><td></td><td>4.96 (-14.0%)</td></tr><tr><td>Sparsity warmup</td><td>✓</td><td>3 × 10-4</td><td>45.9%</td><td>2.293</td><td>108</td><td>119.3 (+1.9%)</td><td></td><td>5.76 (-0.1%)</td></tr></table>

# C.2. Non-gated Sparse LLMs

As explained in Section 2, from the simple 2-layer feed-forward block used in the original transformer, there has been a notable shift, with modern models adopting a gated variant due to small but consistent superior empirical results. Nonetheless, in our work, we introduce training and inference kernels for both variants. In contrast to the gated variant, computing the output activations following 1, for the non-gated variant, the computation simplifies to:

$$
h = \phi ( x W _ { u } ) , y = h W _ { d } ,
$$

where $\phi$ is, once again, the non-linear activation function. Thus, when $\phi$ is a ReLU activation, the sparsity pattern is determined by the up-projection rather than the gate projection. For inference kernels, we note this implies that a difference between the two variants is that the non-gated version performs the up projection rather than the gate projection with our matrix multiplication kernel with TwELL storage introduced in Section 3. Moreover, as detailed in Appendix A, we designed an additional kernel optimized to perform the down projection alone starting from the TwELL format.

Thus, to provide a direct comparison between the two variants and the relative effects of sparsity and our custom kernels, we collect additional training runs on 30B tokens with our 1.5B model implementing the non-gated parameterization. In particular, we consider two sparsity levels in addition to a non-sparse baseline $( L _ { 1 } = 0 )$ :our recommended conservative regularization of $L _ { 1 } =$ $2 \times 1 0 ^ { - 5 }$ and a more aggressive regularization of $L _ { 1 } = 3 \times 1 0 ^ { - 5 }$ . In Table 4, we report the collected relative performance and efficiency results for both variants and all three sparsity levels. As shown, we find only minor performance differences between the two variants, which are likely not significant and attributable to random variations. However, we do note that such differences might become visible only when training with token budgets beyond chinchilla optimality (Hoffmann et al., 2022). Efficiency-wise, while both our variants benefit significantly from our sparse kernels, we find such benefits to be larger for the gated variant. The inference speedup of the non-gated variant is $1 1 . 2 \%$ at $L _ { 1 } = 2 \times 1 0 ^ { - 5 }$ compared to $1 7 . 9 \%$ for the gated variant at the same sparsity level. At larger sparsity levels, this divide is more pronounced, with the gated variant achieving a $2 5 . 5 \%$ speedup at $L _ { 1 } = 3 \times 1 0 ^ { - 5 }$ compared to only $1 3 . 1 \%$ for the non-gated variant. These results are intuitively based on the nature of both models, as the gated variant allows our new inference kernels to leverage the opportunity of efficient fast fusion of both up and down projections. Nonetheless, they also demonstrate that the benefits of our kernels extend beyond a single architectural choice.

# C.3. Strategies for Dead Neuron Mitigation

While we find that using an L1 coefficient of $L _ { 1 } = 2 \times 1 0 ^ { - 5 }$ provides a relevant boost in efficiency vithout any noticeable downstream performance degradation, we explore preliminary directions to mitigate the potential downsides of sparse training. In particular, as detailed in Appendix D, when examining the number of active neurons throughout training, we see that over $3 0 \%$ of the neurons become permanently inactive on average across layers when using our recommended L1 coefficient, with this metric considerably rising for higher regularizations. While for our recommended coeffcient, this symptom does not seem to evidently reflect on downstream performance, reducing such an effect could potentially allow supporting even higher sparsity before incurring performance degradation.

![](images/8.jpg)  
Figure 8 | Number of non-zeros and fraction of dead neurons of LLMs with different strategies for dead neuron mitigation throughout training.

Based on these considerations, we explore two preliminary extensions to our simple L1-regularized training recipe explained in Section 2. First, we consider simply scheduling the L1 regularization, motivated by our findings that dead neurons appear to arise very early during training. Concretely, we first train our models for 5000 steps without any L1 regularization, followed by a further 5000 steps of linear increase of the L1 coefficient. We make the training setting artificially similar to prior work that focuses on finetuning and continued-pretraining (Song et al., 2025; Wang et al., 2024). Second, we consider implementing a target reinitialization strategy to lower the magnitude and reinject random noise only in the columns of the gate projection that lead to always negative outputs (which then, after ReLU, lead to dead neurons). Given the model's initialization standard deviation $\sigma = 0 . 0 2$ , we noised and rescaled to regress the weights to their initial state, essentially interpolating with a coefficient $\lambda$ :

$$
W _ { g } [ : , j ] \gets ( 1 - \lambda ) W _ { g } [ : , j ] + \lambda N ( 0 , \sigma ^ { 2 } ) ,
$$

We apply this targeted reinitialization after every training step, which we find does not significantly affect training time. In preliminary experiments, We found $\lambda = 0 . 1$ to be a good choice that avoids affecting training dynamics while injecting sufficient noise to revive dead neurons. We note this strategy is similar to older techniques for reinjecting plasticity into architectures in continual learning and other non-stationary settings (Ash and Adams, 2020).

In Table 5, we report the performance and efficiency results of our two strategies compared to our standard recipe and the non-sparse baseline, while in Figure 8 we analyze the number of non-zero activations and dead neurons throughout training. When looking at the dead neuron statistics, we find that both strategies almost entirely mitigate the emergence of dead neurons. However, we immediately see a concerning pattern with the sparsity-warmup strategy, as the number of non-zeros considerably increases throughout training. In particular, the considered coefficient of $L _ { 1 } = 3 \times 1 0 ^ { - 4 }$ which is ten times larger than our recommended value, leads to over 100 non-zeros on average across layers at the end of training, compared to only 29 non-zeros when using our standard recipe with $L _ { 1 } = 2 \times 1 0 ^ { - 5 }$ . We note that, in early experiments, we found that increasing the L1 coefficient further led to training instabilities and loss spikes. In contrast, using the targeted dead neuron reinitialization, we find similar non-zero statistics to our standard recipe while stillefectively mitigating dead neurons. Furthermore, as reported in Table 5, we find that this latter strategy provides a small boost in both downstream performance and efficiency, processing tokens $1 9 . 1 \%$ faster than the non-sparse baseline with our default L1 coefficient of $L _ { 1 } = 2 \times 1 0 ^ { - 5 }$ .We believe these preliminary results suggest that further research in examining optimal sparse training would potentially further increase the relevance and efficiency upsides of sparse LLMs.

![](images/9.jpg)  
Figure 9 | Number of non-zeros and fraction of dead neurons of LLMs across L1 regularization levels throughout training.

# D. Extended Results

# D.1. Sparsity and Dead Neurons During Training

In Figure 9, we provide detailed results about how activation sparsity and dead neuron occurrence evolve during training for all our different L1 regularization levels. In particular, we record dead neurons at the end of each training step by keeping track, for each hidden feedforward activation of each layer, the last time it was non-zero. If a neuron was never active for a whole training step (just above 1M tokens), we consider it dead for that step.

We make two immediate observations from these results. First, we find that the sparsity levels settle early on to low values after only around 1000 training steps (around 1B tokens). Due to this property, we note that the throughput and memory advantages of our training kernels become relevant almost at the inception of our training runs. Second, we observe that the same trend applies to the number of dead neurons: our recommended $L _ { 1 } = 2 \times 1 0 ^ { - 5 }$ already exceeds $3 0 \%$ inactivity, which further monotonically increases with higher regularization levels. While for our recommended coefficient, this symptom does not seem to evidently reflect on downstream performance, reducing such an effect could potentially allow supporting even higher sparsity before incurring performance degradation. To this end, we note that in Appendix C we provide preliminary results indicating that the performance of sparse LLMs could be further improved with strategies targeted at dead-neuron mitigation.

# D.2. Task Performance Details

To complement the results in Section 4 in the main text, we provide the detailed granular results of downstream task performance across the seven downstream tasks considered, targeting logic and reasoning capabilities after pretraining (Bisk et al., 2020; Clark et al., 2018; Mihaylov et al., 2018; Sakaguchi et al., 2021; Talmor et al., 2019; Zellers et al., 2019). In particular, we report the per-task accuracies for both sparse models, using our recommended conservative L1 regularization of $2 \times 1 0 ^ { - 5 }$ , and their non-sparse counterparts across all the examined model scales. As shown in Table 6 and consistently with our main text analysis, we do not find significant performance differences between sparse and non-sparse models for our regularization level and all considered tasks. We do, indeed, observe an expected performance rise with larger models across the great majority of tasks. Table 6 | Granular comparison of per-task downstream performance across model scales to complement Table 1.   

<table><tr><td colspan="4">Model scale Sparse Mean Accuracy HellaSwag</td><td rowspan="2">CQA PIQA</td><td colspan="5">Winogrande ARC-easy ARC-challenge OpenBookQA</td></tr><tr><td rowspan="2">0.5B params 10B tokens</td><td>X</td><td>40.4%</td><td>33.7%</td><td>20.9%</td><td>64.5%</td><td>50.9% 64.1%</td><td>28.1%</td><td>20.8%</td></tr><tr><td>✓</td><td>40.4%</td><td>34.0%</td><td>22.3%</td><td>66.4%</td><td>53.8%</td><td>60.5%</td><td>27.5%</td><td>18.0%</td></tr><tr><td rowspan="2">1B params 20B tokens</td><td>X</td><td>44.6%</td><td>39.2%</td><td>20.0%</td><td>68.7%</td><td>54.4%</td><td>72.6%</td><td>34.0%</td><td>23.6%</td></tr><tr><td>✓</td><td>44.7%</td><td>39.8%</td><td>18.6%</td><td>68.1%</td><td>54.8%</td><td>71.6%</td><td>35.3%</td><td>24.4%</td></tr><tr><td rowspan="2">1.5B params 30B tokens</td><td>X</td><td>46.4%</td><td>41.0%</td><td>20.8%</td><td>70.2%</td><td>55.9%</td><td>72.5%</td><td>36.7%</td><td>27.4%</td></tr><tr><td>✓</td><td>46.2%</td><td>41.1%</td><td>21.0%</td><td>69.1%</td><td>54.4%</td><td>74.3%</td><td>37.5%</td><td>26.0%</td></tr><tr><td rowspan="2">2B params 40B tokens</td><td></td><td>49.1%</td><td></td><td>21.0%</td><td>72.0%</td><td>57.8%</td><td>77.2%</td><td></td><td></td></tr><tr><td>X ✓</td><td>48.8%</td><td>45.7% 45.0%</td><td>21.3%</td><td>70.9%</td><td>57.5%</td><td>75.6%</td><td>41.7% 42.2%</td><td>28.6% 28.8%</td></tr></table>

![](images/10.jpg)  
Figure 10 | Sparsity statistics and speedup contributions across different layers of non-sparse LLMs.

# D.3. Activation Sparsity at High and Low Levels

To complement the analysis results provided in Section 4 of the main text, we examine how sparsity regularization affects the distribution of non-zero activations across model depth and relate these metrics to the corresponding speed-up contributions from our kernels during inference. While in our main analysis we reported and analyzed the LLM trained with our recommended conservative L1 regularization of $2 \times 1 0 ^ { - 5 }$ , in Figures 10 we provide analogous results for a non-sparse LLM while in Figure 11 we analyze an LLM trained with the highest regularization regularization level considered $( 1 \times 1 0 ^ { - 4 } )$ . We note that for non-sparse models, due to the high number of non-zeros, the contributions of applying our kernel are actually detrimental  and as such, we report the speed-up contributions as negative percentages. A first observation from the sparsity statistics is that the average number of non-zeros also follows a noticeable trend in the non-sparse model, with the first few layers being the least active, followed by a hump with a peak in activations. However, a key difference comes with the location of the hump: while in our recommended sparse model the peak occurs around layer 6, in the non-sparse LLM the peak still occurs within the first half of the network but is shifted visibly deeper into the architecture around layer 13. Interestingly, in the high-regularization LLM, we actually observe that while the very first layer is again the least active, there are two different peaks  one very early around the second layer and another one in the last layer of the model. Once again, we find that maximum activation counts can easily be well over an order of magnitude higher than the average, with no clear pattern across layers. For the non-sparse model, we again observe a strong inverse correlation between each layer's average non-zeros and its relative speed-up. In contrast to the high-regularization LLM, this correlation is much less visible, as given the high sparsity encountered, the speedups of our kernels are already at their achievable maximum for almost all layers, essentially making executing the up and down projection negligible in the overall computation time.

![](images/11.jpg)  
Figure 12 | Training speedups from our sparse LLM training kernels across L1 regularization levels for both H100 and RTX6000 devices.

![](images/12.jpg)  
Figure 11 | Sparsity statistics and speedup contributions across different layers of an LLM with high regularization $L _ { 1 } = 1 0 ^ { 4 }$ .

# D.4. Improving Efficiency of other Devices

As mentioned in Section 4, given that our kernels consistently reduce memory requirements during training, and as a side benefit, reduce reliance on newer tensor core units, they immediately have higher potential relevance for less capable hardware. Thus, to empirically validate these considerations, we provide additional results comparing the performance speedups of our kernels during training on NVIDIA's RTX PRO 6000 GPUs against the H100 PCIe GPUs used throughout our main paper and other experiments. Some of the other crucial differences of this GPU come from the memory side, with a considerably reduced memory bandwidth (1.59 TB/s vs. 2.0 TB/s). In contrast, the RTX PRO 6000 can benefit from a larger number of Streaming Multiprocessors than the H100 (188 vs. 114), potentially allowing for greater occupancy for sparse workloads. As shown in Figure 12, and in line with our considerations, we find significantly higher speedups on the RTX 6000 GPU across al L1 regularization levels considered. These speedup differences are even more pronounced at higher regularization levels, extending the practical range of L1 coefficients make sparsity provide meaningful efficiency improvements. When dissecting what causes these greater speedups, we first find that thanks to the specific H100 features, such as the higher tensor cores throughput, the runtime of the dense GEMM operations increases from around 400 to 800 microseconds on the RTX 6000. Similarly, kernels that are memory bandwidth bound, including the dense to hybrid matrix multiplication, are also slightly slower by $1 9 \%$ on the RTX 6000 than on the H100. However, once in our hybrid sparse format, due to the larger Streaming Multiprocessors count of the RTX 6000 GPU, the sparse operations run faster than on the H100, with speedups of $1 . 3 4 \times$ and $2 . 1 \times$ for sparse-to-dense and transposition operations, respectively. We find these results indicate that leveraging sparsity with targeted kernels could significantly improve the performance of cheaper devices, which do not implement the latest hardware innovations of higher-end units such as the H100, lowering the field's canonical hardware barriers.

# E. Further Related Work

# E.1. Activation Sparsity in Transformers

Expanding on the findings of Zhang et al. (2022b), Li et al. (2023) documents that Transformer MLP layers with ReLU activations exhibit inherent activation sparsity across architectures, depths, and data distributions. Building on this observation, Mirzadeh et al. (2023) shows that replacing GELU with ReLU in non-gated feed-forward layers yields negligible performance degradation while enabling up to three times theoretical inference speedup with less computation. However, they focus on older architectures (OPT models) with non-gated feed-forward blocks and leave efficient kernel implementation to future work.

More recent methods have also been proposed to enhance sparsity after altering modern gated architectures and have claimed speedups when running sparse feedforward layers in isolation on older generations of devices. TurboSparse (Song et al., 2024) proposes a modification to the feedforward block itself, introducing dReLU, which applies ReLU to both gate and up projections: $h =$ $\mathrm { R e L U } ( x W _ { g } ) \odot \mathrm { R e L U } ( x W _ { u } )$ . ProSparse (Song et al., 2025) proposes finetuning pretrained models and artificial thresholding of the activations to increase sparsity. Q-Sparse (Wang et al., 2024), further deviates from standard architectures via maintaining only the top-K activations and applying a straight-through estimator. We also note that additional works proposed introducing structured sparsity post-training, such as by predicting (Liu et al., 2023) and pruning activation to set sparsity levels (Lee et al., 2024; Liu et al., 2024). Unlike these works, our paper introduces general-purpose kernels to leverage unstructured sparsity, demonstrating empirical efficiency benefits during LLM training and inference.

# E.2. Architectural Approaches to Sparsity

Mixture-of-Experts (MoE) architectures (Fedus et al., 2022; Lepikhin et al., 2020; Shazeer et al., 2017) partition feed-forward layers into separately routed experts, decoupling model capacity from per-token computation. However, MoE requires predetermining the number of experts and sparsity level before training, limiting adaptability to input complexity. Product key memory (Lample et al., 2019) maintains fixed sparsity patterns through $O ( \log n )$ key retrieval. PEER (He, 2024) extends this approach to over one million single-neuron experts with $9 9 . 9 9 \%$ architectural sparsity. UltraMem (Huang et al., 2025) improves PKM and scales to 20 million memory slots, showing that it can outperform MoE with the same parameter and computation budgets. Fast Feedforward Networks (Belcak and Wattenhofer, 2023) use differentiable binary trees to achieve $9 9 \%$ sparsity. While these architectural approaches achieve extreme sparsity, they require substantial modifications to standard Transformer training pipelines. Our approach instead works with conventional architectures, requiring only a change of activation function and optional regularization, making it readily applicable to existing models and training infrastructure.