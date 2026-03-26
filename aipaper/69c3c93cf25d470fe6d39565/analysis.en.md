# 1. Bibliographic Information
## 1.1. Title
The paper's title is *Sparser, Faster, Lighter Transformer Language Models*, indicating its core focus is improving the efficiency of Transformer-based large language models (LLMs) via unstructured sparsity, with custom kernel implementations to deliver practical speed, memory, and energy benefits.
## 1.2. Authors
The research team is from Sakana AI and an anonymous industrial partner:
- Edoardo Cetin (Sakana AI, lead of kernel design and experimentation)
- Stefano Peluchetti (Sakana AI, sparse training lead)
- Emilio Castillo (partner institution, hybrid format and training kernel co-lead)
- Akira Naruse (partner institution, kernel design advisor)
- Mana Murakami (partner institution, method design advisor)
- Llion Jones (Sakana AI, co-author of the seminal 2017 Transformer architecture paper, project lead)
## 1.3. Publication Venue & Status
This work is published as a preprint on arXiv, the primary open-access repository for computer science and machine learning research. As of the current date (2026-03-25), it has not yet been peer reviewed for conference or journal publication, but its open-source release and practical hardware-optimized implementation make it highly relevant for industrial LLM development teams.
## 1.4. Publication Date
Published publicly on UTC 2026-03-24.
## 1.5. Abstract
The paper addresses the extreme computational cost of scaling autoregressive LLMs by targeting the feedforward layers, which account for most LLM parameters and floating-point operations (FLOPs). It introduces a new tile-wise sparse packing format (TwELL) and custom CUDA kernels that integrate seamlessly with modern GPU execution pipelines, eliminating the historical overhead of unstructured sparse computation on dense-optimized GPU hardware. The authors demonstrate that simple L1 regularization can induce >99% activation sparsity in feedforward layers with negligible downstream performance loss. When paired with the custom kernels, this sparsity delivers throughput, energy efficiency, and memory usage benefits that grow with model scale, including up to 20.5% faster inference and 21.9% faster training for billion-parameter models. All code will be released open-source to promote adoption of sparsity as a standard efficiency axis for LLM design.
## 1.6. Source Links
- Preprint abstract: https://arxiv.org/abs/2603.23198v1
- Full PDF: https://arxiv.org/pdf/2603.23198v1
- Open-source code repository: github.com/SakanaAI/sparser-faster-llms

  ---
# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Scaling LLMs to larger parameter counts has driven state-of-the-art performance across natural language tasks, but it also leads to exponentially rising computational, memory, and energy costs for both training and inference. Feedforward layers, a core component of every Transformer block, account for >2/3 of LLM parameters and >80% of execution FLOPs in large models, making them a high-priority target for efficiency optimizations.
### Prior Research Gap
Unstructured sparsity (the practice of skipping computation on zero-valued activations or weights) is a well-documented efficiency method for overparameterized models, and modern LLMs naturally exhibit high activation sparsity in feedforward layers. However, sparse computation has historically been impractical on modern GPUs, which are heavily optimized for dense matrix multiplication workloads. Existing sparse kernel implementations require expensive format conversion steps, have high index management overhead, and do not integrate with standard LLM training pipelines, so they have not seen widespread industry adoption. Prior attempts to leverage sparsity in LLMs also typically require major changes to model architecture or training recipes, increasing implementation friction.
### Innovative Entry Point
This work avoids architectural modifications entirely, instead targeting the system-level bottleneck of sparse computation on GPUs. It introduces purpose-built sparse data formats and CUDA kernels tailored explicitly for LLM feedforward layer computation patterns, which integrate natively with modern GPU features (e.g., Hopper Tensor Memory Accelerator, Tensor Cores) to eliminate the overhead of sparse operations.
## 2.2. Main Contributions & Findings
The paper's three primary contributions are:
1.  **Novel Sparse Formats & CUDA Kernels:** The TwELL (Tile-wise ELLPACK) sparse format for inference, and a hybrid sparse-dense format for training, paired with end-to-end optimized CUDA kernels for both inference and training that require no changes to standard LLM pipelines beyond using ReLU activations and adding lightweight L1 regularization.
2.  **Sparsity Induction Study:** A quantitative analysis showing that mild L1 regularization on feedforward activations can induce >99% sparsity with no meaningful drop in downstream task performance, for model sizes ranging from 0.5B to 2B parameters.
3.  **Scalable Efficiency Benefits:** Empirical validation that the kernel-sparsity combination delivers increasing benefits with model scale: for 2B parameter models, it delivers 20.5% faster inference, 17% lower energy per token, 21.9% faster training, and 22.3% higher training batch throughput, with no meaningful performance loss compared to dense baselines.

    ---
# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All core technical terms are defined below for beginner readers:
1.  **Autoregressive LLM:** A type of language model that generates text one token at a time, using previous tokens as context to predict the next token. Examples include Llama, GPT, and Qwen.
2.  **Transformer Feedforward Layer:** A core component of every Transformer block, run after the self-attention layer. Modern LLMs use a *gated feedforward layer* with three weight matrices (gate, up, down projection) that performs non-linear transformation of token embeddings.
3.  **Sparsity:** The fraction of values in a tensor that are zero. For example, 99% sparsity means only 1% of values are non-zero, so 99% of computation can theoretically be skipped.
    - *Unstructured sparsity:* Zero values can be located anywhere in the tensor, with no fixed pattern, offering maximum flexibility but high implementation overhead.
    - *Structured sparsity:* Zero values follow a fixed pattern (e.g., entire rows/columns are zero), lower overhead but less flexibility to preserve model performance.
4.  **CUDA Kernel:** A low-level function written to run in parallel on NVIDIA GPUs, optimized for specific computation patterns. Standard dense matrix multiplication kernels are highly optimized, but sparse kernels have historically lagged in performance.
5.  **Cooperative Thread Array (CTA):** A group of GPU threads that run in parallel on a single GPU Streaming Multiprocessor (SM), with access to fast shared memory. Modern GPU kernels parallelize work across hundreds of CTAs.
6.  **FLOPs:** Floating-point operations, a standard measure of computation cost for machine learning models.
7.  **Tensor Core:** Specialized hardware units on modern NVIDIA GPUs that accelerate dense matrix multiplication operations, delivering order-of-magnitude higher throughput than general-purpose CUDA cores for dense workloads.
8.  **ELLPACK (ELL) Sparse Format:** A standard sparse matrix storage format that stores non-zero values and their column indices in row-major order, padded to the maximum number of non-zeros per row for fast access.
9.  **L1 Regularization:** A training technique that adds a penalty proportional to the absolute value of activation/weight values to the model loss, encouraging values to be zero to reduce the penalty, thus inducing sparsity.
## 3.2. Previous Works
### Core Background Technologies
1.  **Transformer Architecture (Vaswani et al., 2017):** The standard backbone for all modern LLMs. Each Transformer block consists of a self-attention layer followed by a feedforward layer. The feedforward layer computation for the gated variant (used in all modern LLMs) is:
    \$
    h_u = x W_u, \quad h_g = \sigma(x W_g), \quad h = h_u \odot h_g, \quad y = h W_d
    \$
    Where $x$ is the input token embedding batch, $W_u, W_g, W_d$ are the up, gate, and down projection weight matrices, $\sigma$ is a non-linear activation function (e.g., SiLU, ReLU), and $\odot$ is element-wise multiplication.
2.  **Sparsity in Deep Learning:** Han et al. (2015) first demonstrated that deep neural networks could be pruned to >90% sparsity with no performance loss, but practical implementation on GPUs remained a bottleneck. Hoefler et al. (2021) provides a comprehensive survey of sparsity research, noting that unstructured sparsity has failed to deliver real-world speedups due to hardware-software mismatches.
### Prior Sparse LLM Work
- **Activation Sparsity Discovery:** Li et al. (2023) and Mirzadeh et al. (2023) found that LLMs with ReLU activations naturally exhibit high activation sparsity in feedforward layers, but did not provide practical kernel implementations to leverage this for speedups.
- **Modified Architecture Methods:** TurboSparse (Song et al., 2024), ProSparse (Song et al., 2025), and Q-Sparse (Wang et al., 2024) modify LLM activation functions or add post-training thresholding to increase sparsity, but require architectural changes and only demonstrate speedups on isolated layers, not end-to-end models.
- **Mixture of Experts (MoE):** MoE architectures (Fedus et al., 2022) use structured sparsity by routing each token to a small subset of feedforward "expert" layers, but require major changes to training pipelines, introduce load balancing overhead, and have fixed sparsity levels set at initialization.
## 3.3. Technological Evolution
Sparse computation research has evolved alongside GPU hardware:
1.  Early 2010s: Sparse kernels for CPU and early GPUs focused on ELL and CSR (Compressed Sparse Row) formats, but were far slower than dense kernels for most workloads.
2.  2020s: The introduction of Tensor Cores accelerated dense matrix multiplication by 10-100x, widening the performance gap between dense and sparse workloads.
3.  2023+: Hopper (H100) GPUs introduced hardware features like Tensor Memory Accelerator (TMA) and warp-level matrix operations (WGMMA) that enable more flexible kernel design, creating an opportunity to build high-performance sparse kernels that integrate with dense pipeline features. This work is the first to leverage these features for end-to-end sparse LLM training and inference.
## 3.4. Differentiation Analysis
Compared to prior work, this approach has three key unique advantages:
1.  **Minimal Integration Friction:** It requires no changes to LLM architecture, only switching to ReLU activations and adding lightweight L1 regularization, making it compatible with all existing LLM training pipelines.
2.  **End-to-End Support for Training + Inference:** Prior sparse kernel work only targets inference, while this work provides optimized kernels for both training and inference, delivering efficiency gains across the full LLM lifecycle.
3.  **Scalable Benefits:** Unlike MoE or structured sparsity methods, the efficiency gains grow with model size, aligning with industry trends toward larger model scales.

    ---
# 4. Methodology
## 4.1. Core Principles
The method is built on three core design principles:
1.  **Fuse Sparse Format Conversion:** Eliminate the overhead of converting dense matmul outputs to sparse formats by packing the sparse representation directly in the epilogue of the dense matrix multiplication kernel, avoiding extra kernel launches and global memory access.
2.  **Tailor to LLM Workloads:** Design kernels explicitly for the computation pattern of Transformer feedforward layers, rather than building general-purpose sparse kernels, to minimize overhead.
3.  **Robustness to Non-Uniform Sparsity:** Handle the high variation in non-zero counts across different tokens and layers, which has historically made ELL-based sparse formats brittle for LLM workloads.
## 4.2. Sparse Training Recipe
### Gated Feedforward Layer Specification
The method uses the standard modern gated feedforward layer architecture, with a single modification: the non-linear activation $\sigma$ for the gate projection is set to ReLU, which outputs zero for all negative inputs, naturally creating sparse activations. The exact computation is:
\$
h_u = x W_u, \quad h_g = \mathrm{ReLU}(x W_g), \quad h = h_u \odot h_g, \quad y = h W_d
\$
Where:
- $x \in \mathbb{R}^{M \times K}$: Input batch of token embeddings, $M$ = total number of tokens in the batch, $K$ = hidden dimension of the model
- $W_g \in \mathbb{R}^{K \times N}$: Gate projection weight matrix, $N$ = feedforward hidden expanded dimension (typically 3-4x $K$)
- $W_u \in \mathbb{R}^{K \times N}$: Up projection weight matrix
- $W_d \in \mathbb{R}^{N \times K}$: Down projection weight matrix
- $h_g, h_u, h \in \mathbb{R}^{M \times N}$: Gate, up, and combined hidden activations
- $y \in \mathbb{R}^{M \times K}$: Output of the feedforward layer
### L1 Regularization for Sparsity Induction
To increase activation sparsity beyond the natural level from ReLU, a lightweight L1 regularization term is added to the standard cross-entropy training loss. The exact formula for the regularization term is:
\$
L_{\text{reg}} = \lambda_{L1} \times \frac{1}{L} \sum_{l=1}^{L} \frac{1}{MN} \sum_{m=1}^{M} \sum_{n=1}^{N} \lvert h^l[m,n] \rvert
\$
Where:
- $\lambda_{L1}$: Tunable L1 regularization coefficient (typically $10^{-5}$ to $10^{-4}$)
- $L$: Total number of Transformer layers in the model
- $h^l[m,n]$: Activation value at position `m,n` of the combined hidden activation $h$ for layer $l$
  This penalty encourages the model to set as many activation values to zero as possible, as non-zero values increase the total loss.
## 4.3. TwELL (Tile-wise ELLPACK) Sparse Format
### Limitations of Standard ELL Format
The standard ELL sparse format stores non-zero values and indices in padded row-major order, but requires cross-CTA synchronization to count non-zeros across entire rows of the matmul output, which cannot be done in the same kernel as the dense matmul without incurring high overhead. This requires a separate format conversion kernel, which eliminates most of the speed gains from sparsity.
### TwELL Design
TwELL solves this by splitting the columns of the activation matrix into fixed-size horizontal tiles of size $T_n$ (matched to the tile size of the dense matmul kernel). Each tile is packed locally in ELL format, with no synchronization required across tiles, so the TwELL representation can be generated directly in the epilogue of the dense matmul kernel that computes $h_g = \mathrm{ReLU}(x W_g)$.
The TwELL format stores three components:
1.  $h_v \in \mathbb{R}^{M \times N/C}$: Non-zero activation values, packed per tile, $C$ = compression factor (set so $T_n/C$ is larger than the maximum expected non-zeros per tile to avoid overflow)
2.  $h_I \in \mathbb{N}^{M \times N/C}$: Column indices of each non-zero value
3.  $h_{nz} \in \mathbb{N}^{M \times N_T}$: Number of non-zeros per tile per row, $N_T = \lceil N / T_n \rceil$ = total number of column tiles
    The following figure compares ELL, TwELL, and the hybrid training format:

    ![Figure 1 | Comparison of ELL with our new TwELL and Hybrid sparse formats designed for LLM inference and training, respectively.](images/1.jpg)
    *该图像是一个图表，比较了密集矩阵与三种稀疏格式（a. ELL，b. TwELL，c. Hybrid），展示了它们在值、索引和非零计数方面的不同布局和表示。这样的比较有助于理解在大语言模型推理和训练中的应用效果。*

## 4.4. Inference Kernel Pipeline
The inference pipeline uses only two kernel launches for the entire feedforward layer, eliminating intermediate global memory writes to maximize speed:
### Kernel 1: Gate Projection with TwELL Packing
This kernel computes the gate projection $h_g = \mathrm{ReLU}(x W_g)$ and packs the output directly into TwELL format in the kernel epilogue, with no separate conversion step. The algorithm logic is summarized below:
1.  Parallelize computation across CTAs, each processing a 2D tile of the gate projection output
2.  Compute the dense tile output using Tensor Cores
3.  For each row in the tile, iterate through output values, pack only non-zero values and their indices into the TwELL $h_v$ and $h_I$ arrays, and store the count of non-zeros in $h_{nz}$
4.  Write only the TwELL output to global memory, skipping all zero values
### Kernel 2: Fused Up + Down Projection
This kernel leverages the TwELL-formatted gate activations to fuse the up projection, element-wise multiplication, and down projection into a single kernel, with no intermediate writes of $h_u$ or $h$ to global memory. The exact computation per row $m$ of the input is:
\$
y[m,:] = \sum_{t=0}^{N_T - 1} \sum_{c=0}^{h_{nz}[m,t] - 1} h_v[m, t \times T_n/C + c] \times (x[m,:] \cdot W_u[:,n]) \times W_d[n,:]
\$
Where $n = h_I[m, t \times T_n/C + c]$ is the column index of the non-zero gate activation.
The kernel uses warp-sized CTAs (32 threads per CTA) to maximize parallelism and L2 cache hits, iterating only over non-zero gate activations to skip 99%+ of the computation.
## 4.5. Training Kernel Pipeline & Hybrid Sparse Format
### Limitation of TwELL for Training
During training, intermediate activations must be stored for backpropagation, but sparsity patterns are highly non-uniform across tokens: the maximum number of non-zeros per row can be 10-100x higher than the average. Using TwELL with padding to the maximum non-zero count wastes memory, eliminating the memory benefits of sparsity.
### Hybrid Sparse-Dense Format
To solve this, the training pipeline converts TwELL activations to a hybrid format that dynamically routes rows to one of two storage locations:
1.  **Sparse ELL partition:** Rows with non-zero counts below a threshold $N_{\hat{n}z}$ are stored in a compact ELL format, minimizing memory usage
2.  **Dense backup partition:** Rows with non-zero counts above $N_{\hat{n}z}$ are stored as dense rows to avoid padding overhead
    The format also stores a binary vector $h_b$ indicating which partition each row is in, and the ELL indices for sparse rows. In practice, $N_{\hat{n}z}$ is set to 128, with <1% of rows routed to the dense backup for the recommended L1 regularization level.
### Training Kernel Design
The training pipeline uses three custom kernels:
1.  **TwELL to Hybrid Conversion Kernel:** Converts the TwELL output of the gate projection to the hybrid format, accumulates L1 regularization loss, and routes rows to the sparse/dense partitions.
2.  **Dense-to-Hybrid Matmul Kernel:** Computes the up projection $h_u = x W_u$ only for positions where the gate activation is non-zero, using the sparsity pattern from the hybrid format.
3.  **Hybrid-to-Dense Matmul Kernel:** Computes the down projection $y = h W_d$, processing sparse rows with CUDA cores and dense backup rows with Tensor Cores for maximum efficiency.
### Backward Pass
The backward pass leverages the stored hybrid sparsity pattern to skip computation on zero values, using the same custom matmul kernels. The exact gradient computation formulas are:
\$
\begin{align*}
\nabla h_u &= \nabla h \odot h_g, \quad \nabla h_g = \nabla h \odot h_u, \\
\nabla W_u &= x^\top \nabla h_u, \quad \nabla W_g = x^\top \nabla h_g, \quad \nabla W_d = h^\top \nabla y, \\
\nabla x &= \nabla h_u W_u^\top + \nabla h_g W_g^\top.
\end{align*}
\$
Where $\nabla y$ is the gradient of the loss with respect to the feedforward layer output, and all matrix multiplications use the custom hybrid kernels to skip zero operations.

---
# 5. Experimental Setup
## 5.1. Datasets
### Pretraining Dataset
All models are trained on the deduplicated FineWeb-Edu dataset, a high-quality corpus of educational web text commonly used for LLM pretraining. Models are trained with chinchilla-optimal token counts for their parameter size:
- 0.5B parameter model: 10B training tokens
- 1B parameter model: 20B training tokens
- 1.5B parameter model: 30B training tokens
- 2B parameter model: 40B training tokens
### Downstream Evaluation Datasets
Model performance is evaluated on 7 standard multiple-choice commonsense and reasoning benchmarks:
1.  **ARC (Easy/Challenge):** Grade-school science questions, with the Challenge subset designed to defeat simple retrieval baselines
2.  **HellaSwag:** Commonsense sentence completion task designed to be difficult for LLMs
3.  **OpenBookQA:** Science knowledge question answering with a curated set of background facts
4.  **PIQA:** Physical commonsense reasoning benchmark
5.  **WinoGrande:** Large-scale Winograd schema benchmark testing coreference reasoning
6.  **CommonsenseQA:** General commonsense reasoning benchmark
## 5.2. Evaluation Metrics
All metrics are defined below with their standard formulas:
### 1. Cross-Entropy Loss (Pretraining)
Measures how well the model predicts the next token in the pretraining corpus. Lower values indicate better performance.
Formula:
\$
L_{\text{CE}} = -\frac{1}{M} \sum_{i=1}^{M} \sum_{v=1}^{V} y_{i,v} \log(p_{i,v})
\$
Where $M$ = number of tokens, $V$ = vocabulary size, $y_{i,v}$ = 1 if token $v$ is the ground-truth next token for position $i$, 0 otherwise, $p_{i,v}$ = model predicted probability of token $v$ for position $i$.
### 2. Downstream Task Accuracy
Measures the fraction of correct answers on the multiple-choice downstream benchmarks. Higher values indicate better performance.
Formula:
\$
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of test samples}}
\$
### 3. Inference Throughput
Measures the number of input tokens processed per millisecond during forward pass. Higher values indicate faster inference.
### 4. Energy Per Token
Measures the energy consumed in millijoules (mJ) to process one input token. Lower values indicate better energy efficiency.
### 5. Training Throughput
Measures the number of input tokens processed per millisecond during a full training step (forward + backward pass + optimizer update). Higher values indicate faster training.
### 6. Peak Memory Usage
Measures the maximum GPU memory used during training, in gigabytes (GB). Lower values reduce hardware requirements.
## 5.3. Baselines & Hardware
### Baselines
All sparse models are compared against dense baselines with identical architecture, training data, and hyperparameters:
1.  **Dense ReLU Baseline:** Identical gated feedforward architecture with ReLU activation, no L1 regularization
2.  **Dense SiLU Baseline:** Standard LLM architecture with SiLU activation (used in Llama, Qwen), no sparsity
3.  **Non-Gated Dense Baseline:** Original Transformer feedforward architecture, for ablation testing
### Hardware
All main experiments are run on a node of 8x H100 PCIe GPUs (standard cloud GPU for LLM training). Cross-device experiments are run on NVIDIA RTX 6000 GPUs (consumer/professional workstation GPU) to test performance on lower-end hardware.

---
# 6. Results & Analysis
## 6.1. Core Results Analysis
The following table (Table 1 from the original paper) summarizes the performance and efficiency results across model scales, using the recommended L1 regularization coefficient of $2 \times 10^{-5}$ that preserves full downstream performance:

<table>
<thead>
<tr>
<th rowspan="2">Model scale</th>
<th rowspan="2">Sparse</th>
<th rowspan="2">Mean task accuracy</th>
<th colspan="2">Forward execution (input tokens/ms)</th>
<th colspan="2">Energy per token (mJ)</th>
<th colspan="2">Training step throughput (input tokens/ms)</th>
<th colspan="2">Peak memory (GB)</th>
</tr>
<tr>
<th>Value</th>
<th>Relative gain</th>
<th>Value</th>
<th>Relative gain</th>
<th>Value</th>
<th>Relative gain</th>
<th>Value</th>
<th>Relative gain</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">0.5B params<br>10B tokens</td>
<td>❌</td>
<td>40.4%</td>
<td>410</td>
<td>0.0%</td>
<td>1.63</td>
<td>0.0%</td>
<td>97.3</td>
<td>0.0%</td>
<td>26.2</td>
<td>0.0%</td>
</tr>
<tr>
<td>✅</td>
<td>40.4%</td>
<td>480</td>
<td>+17.0%</td>
<td>1.43</td>
<td>-11.8%</td>
<td>95.9</td>
<td>-1.5%</td>
<td>21.2</td>
<td>-19.2%</td>
</tr>
<tr>
<td rowspan="2">1B params<br>20B tokens</td>
<td>❌</td>
<td>44.6%</td>
<td>185</td>
<td>0.0%</td>
<td>3.71</td>
<td>0.0%</td>
<td>48.6</td>
<td>0.0%</td>
<td>44.5</td>
<td>0.0%</td>
</tr>
<tr>
<td>✅</td>
<td>44.7%</td>
<td>219</td>
<td>+18.1%</td>
<td>3.17</td>
<td>-14.6%</td>
<td>52.1</td>
<td>+7.1%</td>
<td>33.1</td>
<td>-25.5%</td>
</tr>
<tr>
<td rowspan="2">1.5B params<br>30B tokens</td>
<td>❌</td>
<td>46.4%</td>
<td>119</td>
<td>0.0%</td>
<td>5.73</td>
<td>0.0%</td>
<td>31.8</td>
<td>0.0%</td>
<td>62.8</td>
<td>0.0%</td>
</tr>
<tr>
<td>✅</td>
<td>46.2%</td>
<td>141</td>
<td>+18.8%</td>
<td>4.87</td>
<td>-15.0%</td>
<td>35.5</td>
<td>+11.6%</td>
<td>45.1</td>
<td>-28.1%</td>
</tr>
<tr>
<td rowspan="2">2B params<br>40B tokens</td>
<td>❌</td>
<td>49.1%</td>
<td>87.8</td>
<td>0.0%</td>
<td>7.85</td>
<td>0.0%</td>
<td>22.4</td>
<td>0.0%</td>
<td>46.7</td>
<td>0.0%</td>
</tr>
<tr>
<td>✅</td>
<td>48.8%</td>
<td>106</td>
<td>+20.5%</td>
<td>6.51</td>
<td>-17.0%</td>
<td>27.3</td>
<td>+21.9%</td>
<td>57.1</td>
<td>+22.3% (higher batch size)</td>
</tr>
</tbody>
</table>

Key takeaways from these results:
1.  **No Meaningful Performance Loss:** Mean task accuracy is within 0.3% of dense baselines for all model sizes, confirming that L1 regularization induces high sparsity without hurting model capability.
2.  **Benefits Scale With Model Size:** Inference speedup increases from 17% for 0.5B models to 20.5% for 2B models, training speedup increases from -1.5% (negligible slowdown) to 21.9% for 2B models.
3.  **Memory and Energy Savings:** Energy per token decreases by 11.8% to 17% across model sizes, peak memory usage decreases by 19-28% for all models except the 2B model, where the memory savings allow a 22.3% larger batch size, further increasing training throughput.
    The following figures visualize the relationship between L1 regularization level and efficiency gains:
- Training curves across L1 levels (no major loss degradation up to very high regularization):

  ![Figure 2 | Training curves of LLMs across L1 regularization levels.](images/2.jpg)
  *该图像是一个图表，展示了不同 L1 正则化水平下语言模型的训练曲线。随着训练 token 数量的增加，交叉熵损失逐渐降低，各条曲线对应不同的 L1 正则化值，从 $L_1 = 0$ 到 $L_1 = 10^{-4}$。*

- Downstream accuracy vs average non-zero activations (accuracy is flat until <0.5% of neurons are active):

  ![Figure 3 | Downstream accuracy and sparsity statistics of LLMs across L1 regularization levels.](images/3.jpg)
  *该图像是图表，展示了不同 L1 正则化系数下，LLM 的平均任务准确率和非零参数数量的关系。随着 L1 系数变化，平均准确率呈现出一定的波动，同时非零参数数量也在显著变化。图中蓝色条表示平均任务准确率，而橙色条表示非零参数数量的对数值。*

- Inference speedup and energy savings across L1 levels (up to 30% speedup at highest regularization):

  ![Figure 4 | Forward pass speedups and energy savings from our sparse LLM inference kernels across L1 regularization levels.](images/4.jpg)
  *该图像是一个柱状图，展示了在不同 L1 正则化系数下，稀疏 LLM 推理内核相比于稠密模型的推理加速和能源节省百分比。随着 L1 系数的降低，推理速度提升和能源节省都显著增加，最高可达 30% 和 25%。*

- Training speedup and memory reduction across L1 levels (up to 24% speedup and 24% memory reduction):

  ![Figure 5 | Training speedups and peak memory reduction from our sparse LLM training kernels across L1 regularization levels.](images/5.jpg)
  *该图像是一个图表，展示了L1正则化水平对训练速度提升和内存减少的影响。横轴为L1系数，纵轴为百分比，分别表示与密集模型的训练速度提升和内存减少的比例。可以看出，随着L1系数的变化，训练速度和内存使用量有显著变化。*

## 6.2. Ablation Studies
### Activation Function Comparison
The following table (Table 3 from the original paper) compares dense SiLU, dense ReLU, and sparse ReLU models:

<table>
<thead>
<tr>
<th>Model scale</th>
<th>Activation</th>
<th>Sparse</th>
<th>L1 coeff</th>
<th>Mean task accuracy</th>
<th>Cross-entropy</th>
<th># non-zeros</th>
<th>Forward throughput</th>
<th>Energy per token</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">1.5B params<br>30B tokens</td>
<td>ReLU</td>
<td>❌</td>
<td>0</td>
<td>46.4%</td>
<td>2.255</td>
<td>911</td>
<td>117.1 (0.0%)</td>
<td>5.77 (0.0%)</td>
</tr>
<tr>
<td>SiLU</td>
<td>❌</td>
<td>0</td>
<td>47.1%</td>
<td>2.240</td>
<td>5632</td>
<td>116.5 (-0.5%)</td>
<td>5.82 (+0.1%)</td>
</tr>
<tr>
<td>ReLU</td>
<td>✅</td>
<td>2e-5</td>
<td>46.2%</td>
<td>2.297</td>
<td>29</td>
<td>138.0 (+17.9%)</td>
<td>5.07 (-12.1%)</td>
</tr>
</tbody>
</table>

SiLU delivers a small 0.7% accuracy gain, but cannot leverage sparsity, so the sparse ReLU model is 18% faster and 12% more energy efficient, making it a better tradeoff for most production use cases.
### Dead Neuron Mitigation
The following table (Table 5 from the original paper) tests two strategies to reduce dead neurons (neurons that are always zero for all inputs):

<table>
<thead>
<tr>
<th>Model scale</th>
<th>Training modification</th>
<th>Sparse</th>
<th>L1 coeff</th>
<th>Mean task accuracy</th>
<th># non-zeros</th>
<th>Forward throughput</th>
<th>Energy per token</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">1.5B params<br>30B tokens</td>
<td>None</td>
<td>❌</td>
<td>0</td>
<td>46.4%</td>
<td>911</td>
<td>117.1 (0.0%)</td>
<td>5.77 (0.0%)</td>
</tr>
<tr>
<td>None</td>
<td>✅</td>
<td>2e-5</td>
<td>46.2%</td>
<td>29</td>
<td>138.0 (+17.9%)</td>
<td>5.07 (-12.1%)</td>
</tr>
<tr>
<td>Dead neuron reinit.</td>
<td>✅</td>
<td>2e-5</td>
<td>46.6%</td>
<td>29</td>
<td>139.4 (+19.1%)</td>
<td>4.96 (-14.0%)</td>
</tr>
<tr>
<td>Sparsity warmup</td>
<td>✅</td>
<td>3e-4</td>
<td>45.9%</td>
<td>108</td>
<td>119.3 (+1.9%)</td>
<td>5.76 (-0.1%)</td>
</tr>
</tbody>
</table>

Targeted reinitialization of dead neuron weights improves accuracy by 0.4% (exceeding the dense baseline) and increases speedup to 19.1%, while warming up the L1 coefficient reduces sparsity and eliminates efficiency benefits.
## 6.3. Extended Results
### Per-Layer Sparsity Analysis
The following figure shows sparsity statistics and speedup contributions across layers for the 1.5B sparse model:

![Figure 6 | Sparsity statistics and speedup contributions across different layers of our sparse LLMs.](images/6.jpg)
*该图像是图表，展示了稀疏 LLM 各层的非零参数数量以及速度提升贡献。图中以对数尺度显示了每层的最大非零值、平均非零值和速度贡献比例，揭示了不同层次的稀疏性对模型效率的影响。*

There is a near-perfect inverse correlation (-0.996 Pearson coefficient) between the average number of non-zeros per layer and the speedup contribution from the kernel, with earlier layers (lower non-zeros) delivering larger speedups. The kernel is robust to large spikes in maximum non-zeros per layer, with only minimal speed reduction.
### Cross-Device Performance
The following figure compares training speedups on H100 and RTX 6000 GPUs:

![Figure 12 | Training speedups from our sparse LLM training kernels across L1 regularization levels for both H100 and RTX6000 devices.](images/11.jpg)
*该图像是图表，展示了在不同 L1 正则化系数下，H100 和 RTX6000 设备的训练速度提升百分比。数据表明，在 L1 系数增大时，H100 的训练加速效果显著，尤其是在较小的 L1 系数下。*

Speedups are significantly higher on the RTX 6000 (up to 30% vs 24% on H100), as the lower memory bandwidth and higher SM count of the consumer GPU benefits more from the reduced memory usage and CUDA-core focused sparse computation. This makes the method particularly valuable for running LLMs on lower-end consumer or workstation hardware.

---
# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This work solves the long-standing problem of practical unstructured sparsity for LLMs by introducing hardware-optimized sparse data formats and CUDA kernels that integrate seamlessly with standard LLM training and inference pipelines. It demonstrates that mild L1 regularization can induce >99% activation sparsity in feedforward layers with negligible downstream performance loss, and that the custom kernels translate this sparsity into scalable efficiency benefits: up to 20.5% faster inference, 21.9% faster training, 17% lower energy use, and 28% lower peak memory for billion-parameter models. The open-source release of the kernels will lower the barrier to adoption of sparsity as a standard efficiency axis for LLM design, reducing the computational and environmental cost of large-scale foundation models.
## 7.2. Limitations & Future Work
The authors note the following limitations and future research directions:
1.  **Limited to Feedforward Layers:** The current work only targets feedforward layers, but future work could extend the TwELL format and kernels to leverage sparsity in self-attention layers as well.
2.  **ReLU Activation Requirement:** The current method relies on ReLU activations for natural sparsity, but future work could extend it to SiLU or other activations via learnable thresholding to recover the small accuracy benefit of smooth activations while retaining sparsity benefits.
3.  **Pretraining Only:** The current work demonstrates benefits for pretraining, but future work could adapt the method to finetune existing dense LLMs to induce sparsity, delivering efficiency gains for the large library of existing pretrained models.
4.  **Dead Neuron Mitigation:** Preliminary results show that targeted reinitialization of dead neurons improves performance and efficiency, and more advanced mitigation strategies could enable even higher sparsity levels with no performance loss.
## 7.3. Personal Insights & Critique
This work is a major breakthrough for practical LLM efficiency, as it avoids the major adoption friction of prior sparsity methods that require architectural changes or custom training pipelines. It is particularly valuable for edge and consumer hardware deployments, as demonstrated by the higher speedups on the RTX 6000 GPU, which could enable running larger LLMs on local workstations without expensive data center GPUs.
One potential limitation is that the current method does not combine with other efficiency methods like weight quantization, but there is no fundamental barrier to combining 4/8-bit quantization with the sparse kernels, which could deliver multiplicative efficiency gains. Another open question is how the method scales to models larger than 2B parameters, as the efficiency benefits are shown to grow with model size, so it could deliver even larger gains for 10B+ parameter models that are standard in industrial settings.
Overall, this work has the potential to become a standard component of LLM training pipelines in the near future, as it delivers meaningful efficiency gains with almost no implementation cost.