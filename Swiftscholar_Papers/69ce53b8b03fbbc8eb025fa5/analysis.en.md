# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is "Spectral Compact Training: Pre-Training Large Language Models via Permanent Truncated SVD and Stiefel QR Retraction." This title indicates a novel training methodology designed to overcome memory limitations in Large Language Models (LLMs) by utilizing spectral decomposition (SVD) and manifold optimization techniques.

## 1.2. Authors
The author is Björn R. Kohlberger, affiliated with EctoSpace, Dublin, Ireland. The contact information provided is `bkohlberger@icloud.com`.

## 1.3. Journal/Conference
The paper is currently available as a preprint on arXiv (arXiv:2604.00733). It has not yet been published in a specific journal or conference proceedings as of the provided text, though it references a patent application filed in March 2026. arXiv is a reputable open-access repository for preprints in physics, mathematics, and computer science, allowing for early dissemination of research.

## 1.4. Publication Year
The paper is dated March 2026.

## 1.5. Abstract
The paper addresses the "memory wall"—the primary bottleneck for training LLMs on consumer hardware. It introduces Spectral Compact Training (SCT), a method that replaces standard dense weight matrices with permanent truncated Singular Value Decomposition (SVD) factors ($W = U \mathrm{diag}(s) V^{\top}$). Crucially, the full dense matrix is never materialized during training or inference. Gradients flow through these compact factors via standard backpropagation, and the matrices $U$ and $V$ are retracted to the Stiefel manifold using QR decomposition after each optimizer step to maintain orthonormality. The method achieves significant memory reduction (up to $199\times$ per MLP layer at rank 32), enabling training steps for 70B-parameter architectures on a Steam Deck (7.2 GB peak memory). Experiments on SmolLM2-1.7B show that various ranks converge to a similar loss floor, suggesting that learning rate scheduling, rather than rank capacity, is the main bottleneck for matching dense training performance.

## 1.6. Original Source Link
The paper is available as a preprint on arXiv.
*   Link: https://arxiv.org/abs/2604.00733
*   PDF: https://arxiv.org/pdf/2604.00733.pdf
*   Status: Preprint.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem this paper aims to solve is the prohibitive memory cost of training Large Language Models (LLMs). A standard 70B parameter model requires over 1,200 GB of memory just to store weights, gradients, and optimizer states (like Adam) in full precision. This restricts foundational AI research to large institutions with access to expensive multi-GPU clusters. Even smaller 7B models can strain single-GPU setups. The "memory wall" effectively prevents individual researchers and hobbyists from training or fully fine-tuning state-of-the-art models on consumer hardware.

The paper's entry point is the observation that while weights are dense, they often possess an underlying low-rank structure, or can be effectively approximated by one. Instead of compressing a model *after* training or adding small adapters *to* a frozen dense model, the authors propose replacing the weight matrices entirely with a low-rank spectral factorization from the very beginning of training.

## 2.2. Main Contributions / Findings
The paper's primary contributions include:
1.  **SCT Method:** A new training method where weights are permanently stored and updated as truncated SVD factors (`U, s, V`) with Stiefel manifold retraction. The dense matrix is never constructed.
2.  **Architectural Validation:** Demonstrating that a full training step (forward, backward, optimizer, retraction) for a 70B-class architecture can be performed on a Steam Deck with only 7.2 GB of RAM and an Apple M4 Pro with 7.9 GB.
3.  **Rank-Sweep Analysis:** Experiments on SmolLM2-1.7B showing that different ranks (32, 64, 128, 256) converge to the same loss floor, identifying rank 128 as a Pareto-optimal configuration balancing compression and quality.
4.  **Bottleneck Identification:** Providing evidence that the performance gap between SCT and dense training is driven by learning rate configuration, not the spectral rank capacity.

    The key finding is that by leveraging permanent low-rank factorization and geometric constraints (Stiefel manifold), it is possible to drastically reduce the memory footprint of LLM training to fit consumer devices, opening the door to democratized large-scale model training.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, several foundational concepts in linear algebra and deep learning are required:

*   **Singular Value Decomposition (SVD):** This is a factorization of a real or complex matrix. For a matrix $W$ of dimensions $m \times n$, SVD decomposes it into three matrices: $W = U \Sigma V^{\top}$.
    *   $U$ is an $m \times m$ orthogonal matrix (columns are orthonormal).
    *   $\Sigma$ is an $m \times n$ rectangular diagonal matrix containing non-negative singular values on the diagonal.
    *   $V^{\top}$ is the transpose of an $n \times n$ orthogonal matrix $V$.
    *   **Truncated SVD:** In practice, we often keep only the top $k$ largest singular values and their corresponding vectors. This approximates $W$ with a rank-$k$ matrix: $W \approx U_k \Sigma_k V_k^{\top}$. This is the core of the compression in this paper.

*   **Stiefel Manifold:** This is a mathematical space representing all matrices with orthonormal columns. If a matrix $U$ has dimensions $m \times k$ and belongs to the Stiefel manifold, then $U^{\top} U = I_k$ (the identity matrix). Optimization on this manifold ensures that the columns of $U$ remain perpendicular to each other and of unit length, which stabilizes the training of the low-rank factors.

*   **QR Decomposition:** A factorization of a matrix $A$ into a product $A = QR$, where $Q$ is an orthogonal matrix and $R$ is an upper triangular matrix. In this paper, QR decomposition is used as a "retraction" operation: after a gradient step moves the matrix off the manifold, QR decomposition projects it back onto the nearest point on the Stiefel manifold.

*   **MLP (Multi-Layer Perceptron):** A feedforward neural network layer, typically consisting of a linear projection followed by a non-linear activation function. In LLMs (like LLaMA), the MLP block often contains very large weight matrices (e.g., projecting from hidden dimension to intermediate dimension and back), which are the primary targets for compression in this paper.

## 3.2. Previous Works
The paper situates itself among several lines of research:
*   **Low-Rank Adaptation (LoRA):** Proposed by Hu et al. (2021), LoRA freezes the pre-trained dense weights $W$ and trains small adapter matrices $A$ and $B$ such that the update is $\Delta W = BA$. While efficient, the full dense matrix $W$ must still reside in memory.
*   **GaLore (Gradient Low-Rank Projection):** Zhao et al. (2024) proposed projecting gradients into low-rank subspaces to reduce optimizer state memory. However, GaLore still keeps full-rank weights and gradients, whereas SCT eliminates dense gradients entirely.
*   **StelLA (Subspace Learning in Low-rank Adaptation):** Li et al. (2025) applied Stiefel manifold constraints to LoRA adapters. Similar to SCT, it uses Riemannian optimization, but it applies it only to the adapter $\Delta W$ while keeping the base model $W$ frozen and dense. SCT replaces $W$ entirely.

## 3.3. Technological Evolution
The field has evolved from standard dense training $\rightarrow$ post-training compression (quantization/pruning) $\rightarrow$ Parameter-Efficient Fine-Tuning (PEFT) like LoRA (keeping dense weights) $\rightarrow$ memory-efficient training methods like GaLore (reducing optimizer states) $\rightarrow$ SCT. SCT represents a paradigm shift by moving the low-rank constraint into the weight representation itself from the start, rather than treating it as an add-on or a post-processing step.

## 3.4. Differentiation Analysis
The core difference between SCT and related works is the **permanence** and **completeness** of the factorization.
*   **vs. LoRA:** LoRA is $W_{final} = W_{frozen} + BA$. SCT is $W_{final} = USV^{\top}$. SCT does not require a frozen $W$, saving massive memory.
*   **vs. Post-training SVD:** Post-training methods optimize $W \approx USV^{\top}$ after $W$ is already trained. SCT trains `U, S, V` directly. The loss landscape is different because the model is constrained to be low-rank during the entire learning process, forcing the network to find the best low-rank solution rather than approximating a high-rank one.
*   **vs. StelLA:** StelLA optimizes the adapter on the manifold. SCT optimizes the *entire* weight on the manifold.

# 4. Methodology

## 4.1. Principles
The core principle of Spectral Compact Training (SCT) is to redefine the fundamental unit of storage and computation in a neural network layer. Instead of storing a dense weight matrix $W \in \mathbb{R}^{m \times n}$, SCT stores three smaller components: a left singular matrix $U$, a vector of singular values $s$, and a right singular matrix $V$.

The theoretical basis relies on the Eckart–Young–Mirsky theorem, which states that the truncated SVD provides the best low-rank approximation (in terms of Frobenius norm) to a matrix. By constraining the model to this form, we limit its capacity but drastically reduce memory. To ensure the optimization remains stable (since $U$ and $V$ must remain orthonormal for the SVD interpretation to hold), SCT employs Riemannian optimization techniques, specifically "retraction" via QR decomposition, to keep parameters on the Stiefel manifold.

## 4.2. Core Methodology In-depth

### 4.2.1. Spectral Representation
The method begins by replacing every target weight matrix $W$ (typically in the MLP layers) with its rank-$k$ truncated SVD factors. The relationship is defined by the following equation:

$$
W = U \cdot \mathrm{diag}(s) \cdot V^{\top}
$$

In this formula:
*   $W$ is the conceptual dense weight matrix of size $m \times n$. It is **never stored**.
*   $U \in \mathbb{R}^{m \times k}$ is the left singular matrix. It has $k$ columns, each of dimension $m$.
*   $V \in \mathbb{R}^{n \times k}$ is the right singular matrix. It has $k$ columns, each of dimension $n$.
*   $s \in \mathbb{R}^{k}$ is a vector containing the $k$ singular values.
*   $\mathrm{diag}(s)$ converts the vector $s$ into a $k \times k$ diagonal matrix.
*   $V^{\top}$ denotes the transpose of matrix $V$.

    Storage requirements are reduced from $m \times n$ numbers to $k(m + n + 1)$ numbers. For example, in a LLaMA-70B MLP layer where $m=8192$ and $n=28672$, using a rank $k=32$ reduces the parameters from 234.9M to 1.18M.

### 4.2.2. Forward Pass
During the forward pass, the input $x$ (batch size $b$) is processed without ever constructing the full matrix $W$. The operation $y = xW$ is decomposed into three sequential matrix-vector (or matrix-matrix) multiplications.

The process is defined as:
1.  **Projection to latent space:**
    $h = x \cdot U$
    Here, the input $x$ (shape $[b \times m]$) is multiplied by $U$ (shape $[m \times k]$). The result $h$ has shape $[b \times k]$. The computational cost is $O(bmk)$.

2.  **Scaling by singular values:**
    $h_{s} = h \odot s$
    The intermediate representation $h$ is scaled element-wise by the singular values vector $s$. The symbol $\odot$ denotes element-wise multiplication (Hadamard product). The cost is $O(bk)$.

3.  **Projection to output space:**
    $y = h_{s} \cdot V^{\top}$
    The scaled latent vector $h_{s}$ (shape $[b \times k]$) is multiplied by the transpose of $V$ (shape $[k \times n]$). The final output $y$ has shape $[b \times n]$. The computational cost is $O(bkn)$.

The total cost of the forward pass is $O(bk(m + n))$, which is significantly lower than the $O(bmn)$ cost of a dense layer when $k \ll \min(m, n)$.

### 4.2.3. Backward Pass
The backward pass relies on standard automatic differentiation (autograd). PyTorch computes the gradients of the loss $\mathcal{L}$ with respect to the three factors: $\nabla_U \mathcal{L}$, $\nabla_s \mathcal{L}$, and $\nabla_V \mathcal{L}$.

*   The shapes of these gradients are $(m \times k)$, `(k)`, and $(n \times k)$, respectively.
*   Crucially, **no gradient of size $m \times n$ is ever computed or stored**.
*   The gradients are exact with respect to the factored parameterization. However, they are not identical to the gradients of a full-rank dense model because the model is constrained to a low-rank manifold.

### 4.2.4. Stiefel Manifold Retraction
After the optimizer (e.g., AdamW) updates the parameters $U$ and $V$ using the computed gradients, the orthonormality constraint ($U^{\top}U = I$ and $V^{\top}V = I$) might be violated. To restore this constraint, the parameters are "retracted" back to the Stiefel manifold using QR decomposition.

The retraction step is performed as follows:
$$
Q, R = \mathrm{QR}(U_{\mathrm{updated}})
$$
$$
U \leftarrow Q \cdot \mathrm{sign}(\mathrm{diag}(R))
$$

In this procedure:
*   $\mathrm{QR}$ is the QR decomposition function, which factors a matrix into an orthogonal matrix $Q$ and an upper triangular matrix $R$.
*   $U_{\mathrm{updated}}$ is the matrix $U$ after the optimizer step.
*   $Q$ becomes the new orthonormal basis for $U$.
*   $\mathrm{sign}(\mathrm{diag}(R))$ creates a diagonal matrix of signs based on the diagonal elements of $R$. Multiplying by this ensures continuity of the transformation (preventing sign flipping that could confuse the optimizer).
*   This process is repeated identically for $V$. The computational cost is $O(mk^2)$ per layer.

### 4.2.5. Algorithm 1: SCT Training Step
The paper formalizes the entire process in a single training step algorithm:

**Algorithm 1 SCT Training Step**

**Require:** Model with SpectralLinear layers, learning rate $\eta$, batch `(x, y)`
1:  **Forward:** $\hat{y} = \mathrm{model}(x)$ {uses $h = (x \cdot U) \odot s \cdot V^{\top}$}
2:  **Loss:** $\mathcal{L} = \mathrm{CrossEntropy}(\hat{y}, y)$
3:  **Backward:** Compute $\nabla_U \mathcal{L}$, $\nabla_s \mathcal{L}$, $\nabla_V \mathcal{L}$ via autograd
4:  **Optimizer:** AdamW step on $U$, $s$, $V$
5:  **Retract:** For each SpectralLinear layer:
6:  $\quad Q, R \gets \mathrm{QR}(U); \quad U \gets Q \cdot \mathrm{sign}(\mathrm{diag}(R))$
7:  $\quad Q, R \gets \mathrm{QR}(V); \quad V \gets Q \cdot \mathrm{sign}(\mathrm{diag}(R))$

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize the **Alpaca dataset**.
*   **Source/Domain:** Alpaca is a popular instruction-tuning dataset generated by Self-Instruct from GPT-4. It consists of 52,002 instruction-following demonstrations, such as "Write a poem about AI" or "Explain quantum physics."
*   **Sample Form:** A typical data sample includes an `instruction` (the task), an `input` (optional context), and an `output` (the desired response).
*   **Usage:** The dataset is used to fine-tune the SmolLM2 models (1.7B and 135M) to evaluate the convergence and quality of the SCT method compared to dense training. It is chosen because it represents a standard, computationally manageable fine-tuning task that tests a model's ability to follow instructions.

## 5.2. Evaluation Metrics
The paper evaluates the models using two primary metrics:

1.  **Loss (Cross-Entropy Loss)**
    *   **Conceptual Definition:** This measures the difference between two probability distributions: the true distribution of the target tokens and the predicted distribution output by the model. In language modeling, it quantifies how "surprised" the model is by the actual next word; lower loss means better predictions.
    *   **Mathematical Formula:**
        $$
        H(p, q) = -\sum_{x} p(x) \log q(x)
        $$
    *   **Symbol Explanation:**
        *   `H(p, q)` is the cross-entropy.
        *   `p(x)` is the true probability of the target token (usually 1 for the correct token, 0 otherwise).
        *   `q(x)` is the predicted probability assigned by the model to the target token.
        *   $\log$ is the natural logarithm.

2.  **Perplexity (PPL)**
    *   **Conceptual Definition:** Perplexity is the exponential of the average cross-entropy loss. It represents the "branching factor" or the weighted average number of possibilities the model considers for the next token. A lower perplexity indicates the model is more confident (less perplexed) about its predictions.
    *   **Mathematical Formula:**
        $$
        \mathrm{PPL} = \exp\left(\frac{1}{N} \sum_{i=1}^{N} \log p(x_i | x_1, \dots, x_{i-1})\right)
        $$
    *   **Symbol Explanation:**
        *   $\exp$ is the exponential function.
        *   $N$ is the total number of tokens in the sequence.
        *   $x_i$ is the $i$-th token in the sequence.
        *   $p(x_i | x_1, \dots, x_{i-1})$ is the probability the model assigns to token $x_i$ given the previous tokens.

## 5.3. Baselines
The primary baseline is **Dense Training**.
*   **Description:** Standard training where the full weight matrices are stored in memory, and standard backpropagation computes full-rank gradients.
*   **Representativeness:** This is the gold standard for model quality but serves as the contrast for memory efficiency. The paper compares SCT's loss and perplexity against this dense baseline to quantify the "cost" of compression.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The paper presents compelling evidence that SCT drastically reduces memory requirements while maintaining a viable training path.

**1. 70B Architecture Validation:**
The most striking result is the ability to perform a full training step (forward, backward, optimizer, retraction) for a 70B-parameter model architecture on a Steam Deck with 16GB RAM (using only 7.2 GB peak memory) and an Apple M4 Pro (7.9 GB). In contrast, dense FP32 training with Adam would require 1,245 GB. This represents a roughly $172\times$ reduction in total memory usage for the step. This validates the paper's primary claim: the memory wall can be breached to allow consumer-hardware training of massive models.

The following are the results from Table 1 of the original paper, detailing the per-MLP-layer training memory savings:

| Model | Layer (m × n) | Dense+Adam | SCT (k=32) | Compression |
| :--- | :--- | :--- | :--- | :--- |
| SmolLM2-135M | 576 × 1536 | 14.2 MB | 1.1 MB | 13× |
| SmolLM2-360M | 1024 × 4096 | 67.1 MB | 2.6 MB | 26× |
| SmolLM2-1.7B | 2048 × 8192 | 268.4 MB | 5.2 MB | 51× |
| LLaMA-7B | 4096 × 11008 | 721.4 MB | 7.7MB | 93× |
| Qwen-27B | 4096 × 17408 | 1,141 MB | 11.0 MB | 104× |
| LLaMA-70B | 8192 × 28672 | 3,758 MB | 18.9 MB | 199× |

The following are the results from Table 2 of the original paper, showing the 70B architecture validation on consumer hardware:

<table>
<thead>
<tr>
<th>Metric</th>
<th>Apple M4 Pro (48 GB)</th>
<th>Steam Deck (16 GB)</th>
</tr>
</thead>
<tbody>
<tr>
<td>Peak Memory</td>
<td>7,907 MB</td>
<td>7,236 MB</td>
</tr>
<tr>
<td>Forward Pass</td>
<td>0.08 s</td>
<td>0.43 s</td>
</tr>
<tr>
<td>Backward Pass</td>
<td>0.09 s</td>
<td>0.92 s</td>
</tr>
<tr>
<td>Optimizer Step</td>
<td>0.22 s</td>
<td>2.35 s</td>
</tr>
<tr>
<td>QR Retraction</td>
<td>3.02 s</td>
<td>2.58 s</td>
</tr>
<tr>
<td>Total Step</td>
<td>3.41 s</td>
<td>6.28 s</td>
</tr>
<tr>
<td>Ortho. Error</td>
<td>&lt; 2 × 10<sup>−6</sup></td>
<td>&lt; 2 × 10<sup>−6</sup></td>
</tr>
</tbody>
</table>

![Figure 1: Training memory at 70B scale. SCT requires $1 7 2 \\times$ less memory than dense training.](images/1.jpg)

**2. Rank Sweep on SmolLM2-1.7B:**
The experiments on the 1.7B model reveal nuanced behavior regarding rank and convergence.
*   **Convergence Floor:** All tested ranks (32, 64, 128, 256) converged to a similar loss floor ($\sim 4.2 - 4.5$), significantly higher than the dense baseline (1.29). This suggests that simply increasing the rank (capacity) does not immediately solve the convergence gap.
*   **Efficiency Sweet Spot:** Rank 128 emerged as the Pareto-optimal configuration. It achieved the best perplexity (65.6) among the SCT variants while offering 11.7x compression.
*   **Learning Rate Sensitivity:** The authors observe that the convergence gap is likely due to the learning rate configuration. The SCT models used a learning rate of $5 \times 10^{-4}$ (25x higher than dense), which caused instability at higher ranks (where more pre-trained structure is preserved) but aided recovery at lower ranks.

    The following are the results from Table 3 of the original paper, showing the rank sweep results:

    | Method | Params | MLP Comp. | Loss | PPL | GPU Mem. | Step Time |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | Dense | 1,711M | 1.0× | 1.29 | 3.6 | 35.5 GB | 1.17s |
    | SCT r=256 | 692M | 5.9× | 4.33 | 75.6 | 21.3 GB | 1.05 s |
    | SCT r=128 | 598M | 11.7× | 4.18 | 65.6 | 20.0 GB | 0.74 s |
    | SCT r=64 | 551M | 23.5× | 4.34 | 76.7 | 19.3 GB | 0.62 s |
    | SCT r=32 | 527M | 46.9× | 4.47 | 86.9 | 19.0 GB | 0.56 s |

    ![Figure 2: Loss convergence for all ranks. All SCT configurations converge to the same loss floo $( { \\sim } 4 . 2 – 4 . 5 )$ regardless of rank. Dense converges to 1.29.](images/2.jpg)
    *该图像是图表，展示了所有 SCT 配置的损失收敛情况。所有 SCT 配置在损失上均收敛至相同的下限 $( ilde{4.2} - ilde{4.5})$，而 Dense 收敛至 1.29。*

    ![Figure 3: Left: Compression vs. quality Pareto frontier. Rank 128 achieves the best PPL at $1 1 . 7 \\times$ compression. Right: GPU memory by method. Even rank 256 saves $4 0 \\%$ of VRAM.](images/3.jpg)

    **3. Gradient Integrity:**
On the smaller SmolLM2-135M model, SCT (at 95% energy retention) was able to fine-tune and recover from an initial loss spike to reach a perplexity only 1.38x higher than the dense baseline. This confirms that the gradient flow through the factored layers and the QR retraction is mathematically sound and functional.

The following are the results from Table 4 of the original paper, showing the SmolLM2-135M fine-tuning results:

| Method | Final Loss | Final PPL | Trainable Params | PPL Ratio |
| :--- | :--- | :--- | :--- | :--- |
| Dense + AdamW | 0.235 | 1.3 | 134,515,008 | 1.0× |
| SCT (95% energy) | 0.594 | 1.8 | 84,333,271 | 1.38× |

## 6.2. Ablation Studies / Parameter Analysis
The rank sweep experiment serves as the primary ablation study, analyzing the effect of the hyperparameter $k$ (rank).
*   **Rank vs. Quality:** Increasing rank from 32 to 256 did not monotonically improve loss in this specific setup. Rank 128 performed best, suggesting that for a fixed learning rate and training duration, there is an optimal capacity that balances preservation of pre-trained features and learnability.
*   **Rank vs. Speed/Memory:** Lower ranks consistently reduced GPU memory and increased training speed (up to 2.1x faster at rank 32). This confirms the expected trade-off: less computation and memory for lower rank, at the potential cost of model expressiveness (though the "loss floor" phenomenon complicates this).

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper demonstrates that **Spectral Compact Training (SCT)** is a viable method for training extremely large language models on consumer hardware. By permanently storing weights in truncated SVD form ($W = U \mathrm{diag}(s) V^{\top}$) and maintaining orthonormality via Stiefel manifold retraction, SCT achieves massive memory savings (up to孁199x per layer). The authors successfully validated a 70B architecture training step on a Steam Deck (7.2 GB). Experiments on SmolLM2-1.7B revealed that while a convergence gap exists compared to dense training, it appears to be driven by learning rate configuration rather than rank capacity limitations, with Rank 128 identified as a particularly efficient configuration.

## 7.2. Limitations & Future Work
The authors explicitly identify several limitations:
1.  **Convergence Gap:** The $\sim 3$ loss gap between SCT and dense training remains unresolved. While the authors hypothesize it is a learning rate issue, this requires further validation via per-component learning rate scheduling.
2.  **QR Retraction Cost:** The QR decomposition step, while accounting for a small portion of parameters, can take 40-50% of the total step time in large models (like the 70B test). Future work could explore Cayley retractions for lower cost.
3.  **Attention Layers:** The current implementation only applies SCT to MLP layers. Extending it to attention projections (`q, k, v, o`) is necessary for full model compression but requires careful handling of attention patterns.
4.  **Full Pre-training:** The experiments focused on fine-tuning (Alpaca) or single-step validation. Scaling SCT to a full pre-training run from scratch on massive datasets remains to be seen.

## 7.3. Personal Insights & Critique
This paper presents a highly innovative and practically significant approach to the memory bottleneck in AI. The idea of completely discarding the dense matrix representation is bold and effectively distinguishes SCT from adapter-based methods like LoRA.

*   **Strengths:** The memory reduction metrics are undeniable and game-changing for democratization. The mathematical grounding in Stiefel manifolds provides a robust framework for optimization that avoids the instability often seen in naive low-rank training.
*   **Critical Analysis:** The convergence gap is the most critical point. The fact that Rank 32 and Rank 256 converge to the *same* loss floor is a fascinating observation. It suggests that the model might be getting stuck in a local minimum on the manifold, or that the learning rate is mismatched for the geometry of the manifold. If this gap cannot be closed via better tuning, the utility of the method for high-quality model generation might be limited despite the memory savings.
*   **Unverified Assumptions:** The paper assumes that the 70B single-step validation translates to viable full training. However, the QR retraction time (3.02s on M4 Pro vs 0.09s backward pass) is a red flag. For training to be practical, the retraction must be optimized or the wall-clock time penalty might be too high, even if memory fits.
*   **Future Potential:** If the convergence gap can be closed, SCT could revolutionize how we train models, potentially making "infinite width" concepts feasible on finite hardware. It also opens interesting research avenues into "manifold learning rates" and the geometry of the loss landscape on low-rank matrix manifolds.