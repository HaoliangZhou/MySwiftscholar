# 1. Bibliographic Information
## 1.1. Title
The paper's title is *Screening Is Enough*, which centers on introducing a new attention-replacement mechanism called "screening" for language models, with the core claim that this screening mechanism alone can outperform standard softmax attention on multiple critical dimensions.
## 1.2. Authors
The sole author is **Ken M. Nakanishi**, affiliated with two institutions:
1.  Center for Emergent Matter Science (CEMS), RIKEN (a leading Japanese research institute for natural sciences)
2.  Graduate School of Science, The University of Tokyo
    His research focuses on attention mechanism optimization and long-context language modeling, with prior work on Scalable-Softmax (SSMax) for improved attention performance.
## 1.3. Publication Venue
As of the current date (2026-04-02), the paper is published as a preprint on arXiv, and has not yet been peer-reviewed or accepted for publication at a formal conference or journal.
## 1.4. Publication Year
The preprint was published on April 1, 2026 (UTC).
## 1.5. Abstract
The paper identifies a core limitation of standard softmax attention: it only calculates *relative* query-key relevance, redistributing a fixed unit of weight across all keys, with no ability to explicitly reject irrelevant keys. To solve this, the authors introduce **Multiscreen**, a language model architecture built around a *screening* mechanism that evaluates each key against an explicit absolute threshold, discards irrelevant keys, and aggregates only remaining relevant values, eliminating global competition between keys.
Empirically, Multiscreen achieves:
1.  Comparable validation loss to Transformer baselines with ~40% fewer parameters
2.  Stable training at much larger learning rates than Transformers
3.  Near-zero degradation in retrieval performance even at context lengths far exceeding training context
4.  Up to 3.2x faster inference latency at 100K context length
    The paper also introduces **ABCDigits**, a semantics-free synthetic retrieval benchmark that isolates pure retrieval ability from confounding factors like instruction following and semantic cues.
## 1.6. Original Source Links
- Preprint landing page: https://arxiv.org/abs/2604.01178
- PDF link: https://arxiv.org/pdf/2604.01178v1
- Status: Public preprint, not yet peer-reviewed.

  ---

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem Addressed
The core problem is the fundamental limitation of standard softmax self-attention (the core of Transformer architectures) for long-context language modeling:
> Softmax attention computes *relative* relevance between queries and keys: it normalizes all query-key dot product scores across the entire set of keys to produce weights that sum to 1. This means a key only gets a high weight if it is better than *other* keys, not because it meets an absolute threshold of relevance.

This causes three critical issues:
1.  **No explicit rejection of irrelevant keys**: Even if no keys are relevant, softmax must still distribute the full unit of weight across all available keys, leading to noise.
2.  **Attention dilution as context grows**: As context length increases, the fixed unit of weight is spread across more keys, reducing the weight assigned to genuinely relevant keys.
3.  **Positional encoding extrapolation failure**: Most long-context Transformer extensions rely on extrapolating positional encoding patterns beyond training context lengths, which leads to sharp performance degradation at inference time when context is much longer than training context.

### Importance of the Problem
Long-context capabilities are critical for real-world LLM use cases like document summarization, long-form code generation, and legal document analysis, where models need to access information from tens or hundreds of thousands of tokens earlier in the context. Current Transformer-based models are either computationally expensive for long contexts, or fail to reliably retrieve information from distant positions in long contexts.

### Innovative Entry Point
The paper's core innovative idea is to abandon the redistribution-based weight calculation of softmax attention entirely, and instead use *absolute relevance*: each key is evaluated independently against a learned threshold, and only keys that exceed the threshold contribute to the output. This eliminates global competition between keys entirely.
## 2.2. Main Contributions / Findings
The paper's three primary contributions are:
1.  **Multiscreen architecture**: A new language model architecture built around the screening mechanism, which supports absolute query-key relevance, conditionally activated positional encoding (only for short local windows, no extrapolation needed for long range), and learned per-tile context windows to reduce unnecessary computation.
2.  **ABCDigits benchmark**: A new synthetic retrieval benchmark that eliminates confounding factors (semantic cues, instruction following, variable number of keys across context lengths) to provide a pure measurement of a model's retrieval ability.
3.  **Comprehensive empirical validation**: The paper demonstrates that Multiscreen outperforms standard Transformer baselines on four critical dimensions simultaneously:
    - Parameter efficiency (40% fewer parameters for same validation loss)
    - Training stability (supports 10x larger learning rates without divergence)
    - Long-context performance (stable perplexity and near-perfect retrieval at 4x longer context than training length)
    - Inference efficiency (2.3x to 3.2x faster at 100K context length)

      The key finding is that moving beyond redistribution-based attention to absolute relevance-based selection solves multiple long-standing limitations of Transformer architectures for long-context use cases.

---

# 3. Prerequisite Knowledge & Related Work
This section explains all foundational concepts required to understand the paper, for readers with no prior deep learning/NLP expertise.
## 3.1. Foundational Concepts
### 3.1.1. Standard Softmax Self-Attention
Self-attention is the core mechanism of Transformer architectures, first introduced in 2017. It allows each token in a sequence to retrieve information from all other tokens in the context.
The standard calculation formula for self-attention is:
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
Explanation of symbols:
- $Q$ (Query matrix): A projection of the input token representations, where each row corresponds to a token's query vector, used to search for relevant keys.
- $K$ (Key matrix): A projection of the input token representations, where each row corresponds to a token's key vector, used to be matched against queries.
- $V$ (Value matrix): A projection of the input token representations, where each row corresponds to a token's value vector, holding the content that will be retrieved if the key matches the query.
- $d_k$: Dimension of each key vector, the square root is used as a scaling factor to prevent the dot product from becoming too large for large $d_k$, which would make the softmax output extremely sharp (all weight on one key).
- $\mathrm{softmax}(\cdot)$: A normalization function that takes a vector of unbounded scores, and outputs a vector of values between 0 and 1 that sum to exactly 1. This is the step that introduces relative weighting and global competition between keys.

  The core limitation of this formulation, as the paper highlights, is that attention weights are always relative to other keys, with no notion of absolute relevance.
### 3.1.2. Key Definitions for Beginners
- **Parameter efficiency**: A measure of how much performance a model achieves per number of trainable parameters. Higher parameter efficiency means better performance with fewer parameters, reducing training and inference costs.
- **Inference latency**: The time taken for a model to generate a single output token given an input context, measured in seconds. Lower latency means faster generation, which is critical for real-world deployments.
- **Perplexity (PPL)**: A standard metric for evaluating language model performance on next-token prediction. Lower perplexity means the model is better at predicting the next token in a sequence.
- **Retrieval task**: A task where the model must find and use a specific piece of information hidden somewhere in the input context to generate the correct output.
- **RoPE (Rotary Positional Encoding)**: A widely used positional encoding method for Transformers that encodes position information by rotating query and key vectors, so that the similarity between two vectors depends only on their relative position. It requires extrapolation to handle contexts longer than those seen during training, which often leads to performance degradation.
- **GLU (Gated Linear Unit)**: A common activation function used in Transformer feed-forward networks, which uses a gating mechanism to dynamically select which features to pass through, improving model performance.
## 3.2. Previous Works
The paper groups related work into three core categories:
### 3.2.1. Softmax Attention Variants
A large body of work modifies softmax attention to address long-context limitations, but all remain within the redistribution framework:
- **Scalable-Softmax (SSMax)**: Sharpens the attention distribution as context length grows to reduce attention fading, but still uses softmax normalization across keys.
- **Selective Attention**: Adds query- and position-dependent temperature scaling to softmax, but still computes relative weights.
- **Sparsemax/Entmax**: Modify the softmax function to produce sparse attention weights (many zero values), but still normalize across the set of non-zero keys, so relevance remains relative.
- **Sparse/Retrieval-based Attention**: Restrict the set of keys that attention is calculated over (e.g., only local keys, or retrieved relevant keys) to reduce computation, but still apply softmax normalization over the selected subset of keys.

  All of these methods retain the core limitation of relative relevance, with no ability to explicitly reject all keys if none are relevant.
### 3.2.2. Non-Attention Sequence Models
Recent work proposes linear-time sequence modeling architectures to replace attention, including:
- **Mamba**: Uses selective state spaces to model long-range dependencies in linear time.
- **Hyena**: Uses long convolutions for sequence modeling.
- **RetNet**: Uses recurrent retention mechanisms for linear-time inference.

  While these methods are computationally efficient for long contexts, prior empirical work shows they consistently underperform Transformers on retrieval-focused tasks, as they do not support full token-to-token connectivity.
### 3.2.3. Long-Context Positional Encoding Methods
Many methods aim to extend Transformer context length by improving positional encoding extrapolation:
- **ALiBi**: Uses linear attention biases based on relative position, no learned positional embeddings.
- **LongRoPE**: Retunes RoPE parameters to support extrapolation to much longer contexts.
- **FIRE**: Uses function-based relative position encoding for better length generalization.
- **NoPE**: Removes explicit positional encodings entirely to avoid extrapolation failure.

  While these methods improve length generalization to some extent, they still rely on softmax attention's relative weighting, so they still suffer from attention dilution as context grows.
## 3.3. Technological Evolution
The evolution of sequence modeling architectures has followed three key phases:
1.  **2017-2022**: Standard Transformer architectures became dominant, with focus on scaling model size and training data.
2.  **2022-2025**: Focus shifted to efficiency and long-context support, with development of linear-time sequence models (Mamba, etc.) and softmax attention optimizations (sparse attention, RoPE scaling).
3.  **2026 onwards**: The paper's work represents a new phase, moving beyond redistribution-based attention entirely to absolute relevance-based selection, which solves multiple limitations of prior approaches simultaneously.
## 3.4. Differentiation Analysis
Compared to all prior work, Multiscreen has three core unique innovations:
1.  **Absolute relevance calculation**: No normalization across keys, each key is evaluated independently against a threshold, eliminating global competition entirely.
2.  **Conditional positional encoding (MiPE)**: Positional encoding is only activated for short learned windows, and disabled for long-range interaction, so no positional extrapolation is required for long contexts.
3.  **Learned per-tile context windows**: Each head (tile) learns its own context window size, so the model only pays computation cost for long-range interaction where it is actually needed, leading to faster inference.

    ---

# 4. Methodology
This section deconstructs the Multiscreen architecture step-by-step, with all formulas exactly as presented in the original paper, and integrated explanations for each step.
## 4.1. Principles
The core theoretical intuition behind Multiscreen is:
> By evaluating each key independently against an absolute threshold of relevance, rather than normalizing weights across all keys, we eliminate the attention dilution, positional extrapolation, and training instability issues of softmax attention, while retaining full token-to-token connectivity for strong retrieval performance.

The design prioritizes three goals:
1.  Absolute, not relative, query-key relevance
2.  No positional extrapolation required for long contexts
3.  Adaptive computation (only compute long-range interaction where needed)
## 4.2. Core Methodology In-depth (Layer by Layer)
The overall Multiscreen architecture is illustrated in Figure 1 from the original paper:

![Figure 1: (a) Multiscreen architecture. The model comprises a stack of $N _ { \\mathrm { L } }$ residual layers, each containing $N _ { \\mathrm { H } }$ parallel gated screening tiles. The input embedding matrix is normalized and shared with the language-modeling head, with learned scalars $\\mathrm { e } ^ { s _ { \\mathrm { E } } }$ and $\\mathrm { e } ^ { s _ { \\mathrm { F } } }$ controlling input and output scaling. (b) A gated screening tile. The tile computes query, key, value, and gate projections, applies a screening unit to the projected queries, keys, and values, modulates the result with a nonlinear gate, and projects back to the model dimension. (c) A screening unit. The unit normalizes queries, keys, and values to unit length, applies minimal positional encoding (MiPE) to queries and keys, computes distance-aware relevance through Trim, Square, and Softmask, aggregates the surviving values, and applies TanhNorm. In the diagrams, `@` " denotes matrix multiplication and "/RSS" denotes row-wise normalization to unit length.](images/1.jpg)
*该图像是示意图，介绍了Multiscreen架构及其组成部分。图（a）展示了Multiscreen模型的整体结构，包含多个残差层和并行的门控筛选单元；图（b）说明了门控筛选单元的计算流程，包括查询、键、值的投影和筛选操作；图（c）则介绍了筛选单元的具体实现，包含对查询、键、值的归一化和距离感知相关性计算。*

The model follows a standard residual layer stack, but replaces the standard attention + feed-forward pair in each layer with a set of parallel gated screening tiles.
### 4.2.1. Overall Architecture Overview
First, define the input and embedding layer:
Given a tokenized input sequence $(t_1, \dots, t_T)$, where each $t_i$ is an index into the vocabulary $\mathcal{V}$ of size $|\mathcal{V}|$, the embedding matrix is defined as:
$$
W_{\mathrm{E}} = [e_1; \dots; e_{|\mathcal{V}|}] \in \mathbb{R}^{|\mathcal{V}| \times d_{\mathrm{E}}}
$$
where $d_{\mathrm{E}}$ is the embedding dimension.
Each embedding row is normalized to unit length, with a learned scalar $s_{\mathrm{E}}$ controlling the input scale:
$$
\bar{\boldsymbol{e}}_j = \frac{\boldsymbol{e}_j}{\left\| \boldsymbol{e}_j \right\|}, \qquad \overline{W}_{\mathrm{E}} = [\bar{e}_1; \dots; \bar{e}_{|\mathcal{V}|}],
$$
$$
\boldsymbol{x}_i^{(0)} = \mathrm{e}^{s_{\mathrm{E}}} \bar{\boldsymbol{e}}_{t_i}.
$$
The model then applies a stack of $N_{\mathrm{L}}$ residual layers, each containing $N_{\mathrm{H}}$ parallel gated screening tiles. The output of layer $\ell$ for token $i$ is:
$$
\boldsymbol{x}_i^{(\ell)} = \boldsymbol{x}_i^{(\ell-1)} + \sum_{h=1}^{N_{\mathrm{H}}} \Delta \boldsymbol{x}_i^{(\ell, h)}.
$$
where $\Delta \boldsymbol{x}_i^{(\ell, h)}$ is the update from tile $h$ in layer $\ell$.
After the final layer, the token representation is projected to the vocabulary space using the *same normalized embedding matrix* $\overline{W}_{\mathrm{E}}$ (weight tying, to improve parameter efficiency), with a learned scalar $s_{\mathrm{F}}$ controlling output scale:
$$
z_{ij} = \boldsymbol{x}_i^{(N_{\mathrm{L}})} \left( \mathrm{e}^{s_{\mathrm{F}}} \bar{\boldsymbol{e}}_j^\top \right), \qquad j \in \{1, \dots, |\mathcal{V}|\},
$$
where $z_{ij}$ is the logit for vocabulary index $j$ for token $i$. A standard softmax is applied to $z_{ij}$ to get next-token probabilities.
All architectural hyperparameters scale with a single supraparameter $\Psi$, as shown in Table 1 from the paper:

| Hyperparameter | Symbol | Suggested | Used in experiments |
| --- | --- | --- | --- |
| Number of layers | $N_L$ | $\Psi$ | $\Psi$ |
| Number of heads | $N_H$ | $\Psi$ | $\Psi$ |
| Embedding dim | $d_E$ | $\Psi^2$ | $\Psi^2$ |
| Key dim | $d_K$ | 16 or 32 | 16 |
| Value dim | $d_V$ | 64 or 128 | 64 |
| MiPE threshold | $W_{th}$ | - | 256 |
| Vocabulary size | $|\mathcal{V}|$ | - | 50,257 |

This single-parameter scaling makes it easy to adjust model size for different use cases without retuning all hyperparameters.
The core difference between Transformer attention and Multiscreen screening is summarized in Table 2:

| | Transformer | Multiscreen |
| --- | --- | --- |
| Query-key dot product | attention score (unbounded) | similarity ([-1, 1]) |
| Weight computation | relative (softmax) | absolute (Trim / Square / Softmask) |
| Weights | attention weights (sum to 1) | relevance values ([0,1]) |

---
### 4.2.2. Gated Screening Tile
Each gated screening tile (the head-level module) has three components: a screening unit, a nonlinear GLU-style gate, and an output projection.
First, the tile computes four linear projections of the input token representations $\boldsymbol{x}_i$:
$$
q_i = x_i W_{\mathrm{Q}}, \qquad k_i = x_i W_{\mathrm{K}}, \qquad v_i = x_i W_{\mathrm{V}}, \qquad g_i = x_i W_{\mathrm{G}},
$$
where:
- $W_{\mathrm{Q}}, W_{\mathrm{K}} \in \mathbb{R}^{d_{\mathrm{E}} \times d_{\mathrm{K}}}$ are query and key projection matrices
- $W_{\mathrm{V}}, W_{\mathrm{G}} \in \mathbb{R}^{d_{\mathrm{E}} \times d_{\mathrm{V}}}$ are value and gate projection matrices
  The projected queries, keys, and values are passed to the screening unit, which outputs a context representation $\boldsymbol{u}_i$ for each token (detailed in the next section).
In parallel, the gate vector is computed as:
$$
\hat{\boldsymbol{g}}_i = \operatorname{tanh}(\operatorname{SiLU}(\boldsymbol{g}_i)).
$$
where $\operatorname{SiLU}(\cdot)$ is the Sigmoid-weighted Linear Unit activation function, $\operatorname{SiLU}(x) = x \cdot \sigma(x)$, with $\sigma(x)$ the sigmoid function.
The screening output and gate are combined via elementwise multiplication:
$$
\boldsymbol{h}_i = \boldsymbol{u}_i \odot \hat{\boldsymbol{g}}_i.
$$
Finally, the tile's output update is projected back to the model dimension:
$$
\Delta \boldsymbol{x}_i = \boldsymbol{h}_i \left( \mathrm{e}^{s_0} W_0 \right),
$$
where $W_0 \in \mathbb{R}^{d_{\mathrm{V}} \times d_{\mathrm{E}}}$ is the output projection matrix, and $s_0$ is a learned scalar controlling the residual update scale, to ensure stable training as the number of tiles increases.
All parameter shapes and initialization values are defined in Table 3 from the paper:

| Parameter | Shape | Initialization |
| --- | --- | --- |
| $W_Q$ | $(d_E, d_K)$ | $\mathcal{N}(0, 0.1/\sqrt{d_K})$ |
| $W_K$ | $(d_E, d_K)$ | $\mathcal{N}(0, 0.1/\sqrt{d_K})$ |
| $W_V$ | $(d_E, d_V)$ | $\mathcal{N}(0, 0.1/\sqrt{d_V})$ |
| $W_G$ | $(d_E, d_V)$ | $\mathcal{N}(0, 0.1)$ |
| $W_O$ | $(d_V, d_E)$ | $\mathcal{N}(0, 0.1/\sqrt{d_E})$ |
| $W_E$ | $(|\mathcal{V}|, d_E)$ | $\mathcal{N}(0, 0.1/\sqrt{d_E})$ |
| $s_w$ | scalar | linearly spaced across heads from 0 to $\log w_{th}$ in each layer |
| $s_r$ | scalar | 0 |
| $s_O$ | scalar | $\log(1/\sqrt{N_H N_L})$ |
| $s_E$ | scalar | 0 |
| $s_F$ | scalar | $\log \sqrt{d_E}$ |

The window parameter $s_w$ is initialized to be linearly spaced across heads, so that tiles start with a diverse range of window sizes at the beginning of training.
---
### 4.2.3. Screening Unit (Core Mechanism)
The screening unit is the component that implements absolute relevance calculation. It has two learned scalar parameters $s_w$ and $s_r$, which define the screening window size $w$ and acceptance width $1/r$:
$$
w = \mathrm{e}^{s_w} + 1, \quad \quad r = \mathrm{e}^{s_r} + 1,
$$
The screening unit follows 9 sequential steps:
#### Step 1: Unit-length normalization
Queries, keys, and values are all normalized to unit length:
$$
\bar{q}_i = \frac{q_i}{\left\| q_i \right\|}, \qquad \bar{k}_i = \frac{k_i}{\left\| k_i \right\|}, \qquad \bar{v}_i = \frac{v_i}{\left\| v_i \right\|}.
$$
This ensures that the dot product between any query and key (similarity) is bounded to the range $[-1, 1]$, providing a consistent scale for thresholding, and removing the influence of vector norm on similarity (only directional alignment matters).
#### Step 2: Minimal Positional Encoding (MiPE)
MiPE is a conditional positional encoding that is only activated for short screening windows, avoiding the need for positional extrapolation for long-range interaction. It applies a RoPE-like rotation only to the first two coordinates of queries and keys, controlled by the learned window size $w$.
For a vector $\boldsymbol{z}_i$ at position $i$, MiPE is defined as:
$$
\tilde{z}_i = z_i M_i(w),
$$
where the rotation matrix $M_i(w)$ is:
$$
M_i(w) = \left( \begin{array}{cc} R(\phi(i,w)) & 0 \\ 0 & I_{d_{\mathrm{K}} - 2} \end{array} \right), \qquad R(\phi) = \left( \begin{array}{cc} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{array} \right),
$$
and the rotation angle $\phi(i,w)$ is:
$$
\phi(i,w) = \frac{\pi i \gamma(w)}{w}.
$$
The gating function $\gamma(w)$ smoothly disables MiPE when the window size $w$ exceeds the threshold $w_{th}$ (set to 256 in experiments):
$$
\gamma(w) = \left\{ \begin{array}{ll} \frac{1}{2} \left( \cos \frac{\pi w}{w_{\mathrm{th}}} + 1 \right), & w < w_{\mathrm{th}}, \\ 0, & w \geq w_{\mathrm{th}}. \end{array} \right.
$$
When $\gamma(w) = 0$, $M_i(w)$ becomes the identity matrix, so MiPE has no effect. This means for long-range windows, no positional encoding is applied, so no extrapolation of positional patterns is required.
MiPE is applied to normalized queries and keys:
$$
\tilde{\boldsymbol{q}}_i = \bar{\boldsymbol{q}}_i M_i(w), \qquad \tilde{\boldsymbol{k}}_j = \bar{\boldsymbol{k}}_j M_j(w).
$$
As with RoPE, the resulting similarity between $\tilde{\boldsymbol{q}}_i$ and $\tilde{\boldsymbol{k}}_j$ depends only on their relative position, not absolute position.
#### Step 3: Query-key similarity calculation
The similarity between query $i$ and key $j$ is the dot product of the MiPE-transformed vectors, bounded to `[-1,1]`:
$$
s_{ij} = \tilde{q}_i \tilde{k}_j^\top, \qquad s_{ij} \in [-1, 1].
$$
#### Step 4: Trim-and-Square transform (distance-unaware relevance)
This step converts similarity to distance-unaware relevance, applying an absolute threshold to discard irrelevant keys:
$$
\alpha_{ij} = \left[ \operatorname*{max} \bigl( 1 - r(1 - s_{ij}), 0 \bigr) \right]^2.
$$
Explanation:
- The term $1 - r(1 - s_{ij})$ equals zero when $s_{ij} \leq 1 - 1/r$, so all similarities below this threshold are set to exactly zero (irrelevant keys are discarded).
- The square operation emphasizes high similarities close to 1, so that very relevant keys have much higher relevance than marginally relevant keys.
  This transform is visualized in Figure 2 from the paper, with acceptance width $1/r = 1/3$:

  ![Figure 2: Illustration of the Trim-and-Square transform (here shown with acceptance width $1 / r =$ $1 / 3 )$ . Only similarities greater than $1 - 1 / r$ produce nonzero relevance, illustrating the effective acceptance threshold.](images/2.jpg)
  *该图像是示意图，展示了 Trim 和 Trim & Square 变换的评分机制。横轴表示查询-键相似性 `q ullet k`，纵轴表示筛选得分。图中展示的接受宽度为 $1 / r$，只有相似度大于 $1 - 1 / r$ 的值才产生非零相关性，反映了有效的接受阈值。*

#### Step 5: Causal softmask (distance-aware relevance)
A causal, distance-aware softmask is applied to restrict attention to a learned window of previous tokens, with a smooth transition at the window edge:
$$
m_{ij}(w) = \left\{ \begin{array}{ll} \frac{1}{2} \left( \cos \frac{\pi (j - i)}{w} + 1 \right), & -w < j - i \leq 0, \\ 0, & \mathrm{otherwise}. \end{array} \right.
$$
This mask only allows keys from the previous $w$ tokens (causal masking, no access to future tokens), and smoothly reduces the weight of keys as they approach the window edge, avoiding hard cutoffs.
The final distance-aware relevance is the product of the distance-unaware relevance and the softmask:
$$
\alpha_{ij}^{\mathrm{d}} = \alpha_{ij} m_{ij}(w).
$$
A key only contributes to the output if it survives both the content-based threshold (step 4) and the distance-based window mask (step 5).
#### Step 6: Value aggregation
The relevant values are aggregated by weighting each value with its distance-aware relevance:
$$
h_i = \sum_{j=1}^T \alpha_{ij}^{\mathrm{d}} \bar{v}_j.
$$
Unlike softmax attention, there is no normalization across keys, so the sum of weights can be zero (if no keys are relevant), allowing the model to explicitly represent the absence of relevant context.
#### Step 7: TanhNorm
To prevent the aggregated vector norm from growing excessively while preserving its direction, TanhNorm is applied:
$$
\mathrm{TanhNorm}(\boldsymbol{x}) = \frac{\operatorname{tanh}\|\boldsymbol{x}\|}{\|\boldsymbol{x}\|} \boldsymbol{x}.
$$
TanhNorm behaves like the identity function for small vector norms, and smoothly bounds the output norm to 1 for large norms.
The final output of the screening unit is:
$$
\boldsymbol{u}_i = \mathrm{TanhNorm}(\boldsymbol{h}_i).
$$
---

# 5. Experimental Setup
## 5.1. Datasets
Three datasets are used for experiments:
### 5.1.1. Pretraining Dataset: SlimPajama
- Source: A cleaned, deduplicated version of the RedPajama dataset, developed by Cerebras.
- Scale: 628 billion total tokens, 44% (~276 billion tokens) used for pretraining in the paper.
- Domain: General text, including books, websites, code, and scientific papers, suitable for training general-purpose language models.
- Rationale for choice: It is a standard widely used dataset for LLM pretraining, ensuring fair comparison to Transformer baselines trained on the same data.
### 5.1.2. Long-Context Perplexity Evaluation Dataset: PG-19
- Source: A corpus of books published before 1919 from Project Gutenberg.
- Scale: 5,747 documents with token length exceeding $2^{17}$ (131,072 tokens) are used for evaluation.
- Rationale for choice: It is a standard benchmark for long-context language modeling, as the long book texts require models to track context over tens of thousands of tokens.
### 5.1.3. Retrieval Evaluation Benchmark: ABCDigits
ABCDigits is a new synthetic benchmark introduced in the paper, designed to isolate pure retrieval ability without confounding factors.
- Structure: Each instance consists of a shuffled list of equations mapping 26 uppercase letters to random 6-digit integers (e.g., $A=967892$, $B=123456$). A query is formed by appending a target letter followed by `=` (e.g., $L=$), and the model must generate the correct 6-digit integer for that letter.
- Key design features:
  1.  No natural language semantics: Letters and digits are randomly mapped, so no prior semantic knowledge can help the model.
  2.  Fixed number of keys: Exactly 26 unique letter-digit pairs exist in every instance, regardless of context length, eliminating confounding effects from increasing number of keys as context grows.
  3.  No instruction following required: The model only needs to complete the pattern seen in the context, no explicit instructions are given.
- Example prompt from the paper (Figure 6a):

  ![Figure 6: (a) Example prompt for ABCDigits. (b) Retrieval accuracy heatmaps over context length (columns) and target depth (rows). Columns correspond to the two training settings: base models trained with context length $2 ^ { 1 2 }$ (left) and models after continual pretraining with context length $2 ^ { 1 5 }$ (right). Rows correspond to 353M Transformer (top), 286M Multiscreen (middle), and 28M Multiscreen (bottom). Colors indicate exact-match retrieval accuracy. For Transformer, each cell shows accuracy under the best-performing RoPE scaling factor selected from multiple candidates, and the number inside the cell indicates the selected factor. A dash ("-") indicates that no correct retrieval occurred. The dashed and dotted vertical lines mark the context lengths used during base pretraining $( 2 ^ { 1 2 } )$ and long-context continual pretraining $( 2 ^ { 1 5 } )$ , respectively.](images/6.jpg)
  *该图像是图表，展示了图6中的两个部分：(a) ABCDigits的示例提示与(b) 随着上下文长度（列）和目标深度（行）变化的检索准确率热图。上半部分显示353M Transformer模型的准确率，底部是286M和28M Multiscreen模型。颜色表示准确匹配的检索精度，特定单元格中的数字是所选的RoPE缩放因子，"-"表示未成功检索。虚线标记了用于基础和长期预训练的上下文长度。*

- Rationale for choice: Existing retrieval benchmarks (like needle-in-haystack) often have semantic cues or require instruction following, so model performance can be affected by factors unrelated to pure retrieval ability. ABCDigits eliminates these confounders.
## 5.2. Evaluation Metrics
Four metrics are used to evaluate performance:
### 5.2.1. Validation Cross-Entropy Loss
- **Conceptual Definition**: Measures how well the model predicts the next token in a held-out validation set. Lower loss means better next-token prediction performance.
- **Mathematical Formula**:
  $$
  \mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log p(t_i | t_1, \dots, t_{i-1})
  $$
- **Symbol Explanation**:
  - $N$: Total number of tokens in the validation set
  - $p(t_i | t_1, \dots, t_{i-1})$: The model's predicted probability of the ground-truth next token $t_i$ given the previous context
### 5.2.2. Perplexity (PPL)
- **Conceptual Definition**: The inverse geometric mean of the model's predicted next-token probabilities. Lower perplexity means the model is more confident and accurate at next-token prediction. It is directly derived from cross-entropy loss.
- **Mathematical Formula**:
  $$
  \text{PPL} = \exp\left(\mathcal{L}\right) = 2^{-\frac{1}{N}\sum_{i=1}^N \log_2 p(t_i | t_1, \dots, t_{i-1})}
  $$
- **Symbol Explanation**: Same as cross-entropy loss, $\exp(\cdot)$ is the exponential function.
### 5.2.3. Exact-Match Retrieval Accuracy
- **Conceptual Definition**: The percentage of ABCDigits instances where the model generates the exact correct 6-digit integer for the target letter. Higher accuracy means better retrieval ability.
- **Mathematical Formula**:
  $$
  \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of instances}} \times 100\%
  $$
### 5.2.4. Inference Latency
- **Conceptual Definition**: The time taken to generate a single next token given an input context of 100,000 tokens, measured in seconds. Lower latency means faster inference.
## 5.3. Baselines
The only baseline used is a standard LLaMA-style Transformer with RoPE positional encoding, weight tying between input embedding and language modeling head, and hyperparameters matched to the Pythia suite of open-source LLMs.
The baseline model configurations across different parameter scales are shown in Table 5 from the paper:

| Hyperparameter | 8M | 18M | 45M | 124M | 353M |
| --- | --- | --- | --- | --- | --- |
| Number of layers ($N_L$) | 6 | 6 | 6 | 12 | 24 |
| Number of heads ($N_H$) | 4 | 8 | 8 | 12 | 16 |
| Embedding dim ($d_E$) | 128 | 256 | 512 | 768 | 1024 |
| Total params | 7,613,312 | 17,584,384 | 44,609,024 | 123,550,464 | 353,453,056 |
| Non-embedding params | 1,180,416 | 4,718,592 | 18,877,440 | 84,953,088 | 301,989,888 |

All baseline models are trained with exactly the same token budget, data, and batch size as the Multiscreen models, to ensure fair comparison.

---

# 6. Results & Analysis
## 6.1. Core Results Analysis
### 6.1.1. Scaling Efficiency
Figure 3 from the paper shows the scaling relationship between model size (number of parameters) and validation loss for Transformer and Multiscreen:

![Figure 3: Scaling behavior of Transformer and Multiscreen. Validation loss is plotted against model size (number of parameters) on a log scale. Markers represent the mean over three runs, and error bars indicate one standard deviation (smaller than the marker size). For the 4B model, only a single run is available due to computational constraints. Multiscreen achieves similar validation loss at roughly $40 \\%$ fewer parameters along the scaling trend compared to Transformer.](images/3.jpg)
*该图像是图表，展示了Transformer和Multiscreen的验证损失与模型规模（参数数量）之间的关系。验证损失在对数刻度上绘制，标记代表三次运行的均值，误差条表示一个标准偏差。可见，Multiscreen在参数数量大约减少40%的情况下，取得了与Transformer相似的验证损失。*

Key findings:
- Multiscreen achieves nearly identical validation loss to Transformer baselines with approximately 40% fewer parameters across all evaluated scales (from 8M to 4B parameters).
- The gap is even larger when considering only non-embedding parameters (Figure 7 left), as Multiscreen's tied normalized embedding structure is more parameter-efficient.
- The 4B Multiscreen model deviates slightly from the scaling trend, which the authors attribute to undertraining (larger models require more tokens to reach full convergence).

  ![Figure 7: Scaling behavior under alternative definitions of model size. Left: scaling behavior of Transformer and Multiscreen with respect to non-embedding parameters. Right: scaling behavior of Multiscreen with respect to the supraparameter $\\Psi$ The 4B Multiscreen model deviates from the scaling trend, which we attribute to undertraining.](images/7.jpg)
  *该图像是一个图表，展示了Multiscreen与Transformer在非嵌入参数和超参数$ ext{Ψ}$下的验证损失缩放行为。左侧显示了不同非嵌入参数数量下的验证损失，而右侧则展示了与超参数$ ext{Ψ}$的关系。Multiscreen在较少参数的情况下仍能保持较低的验证损失。*

### 6.1.2. Learning Rate Stability
Figure 4 from the paper shows the results of a learning rate sweep for 45M Transformer and 28M Multiscreen models:

![Figure 4: Learning rate sweep comparing Transformer and Multiscreen. The learning rate is shown on a log scale. Multiscreen remains stable even at large learning rates, while Transformer training becomes unstable as the learning rate increases. For Transformer, runs with learning rates $\\geq 2 ^ { - \\overline { { 4 } } }$ diverged and are omitted from the plot.](images/4.jpg)
*该图像是图表，展示了Transformers和Multiscreen在不同学习率下的验证损失比较。横坐标为学习率（对数刻度），纵坐标为验证损失。图中显示，45M的Transformer（灰色虚线）在较高学习率下训练不稳定，验证损失增加；而28M的Multiscreen（蓝色实线）在较大学习率下依然保持稳定的性能表现。*

Key findings:
- Transformer training becomes unstable and diverges when learning rate exceeds $2^{-4}$ (~0.0625).
- Multiscreen remains stable even at much larger learning rates, including $2^0$ (1.0), with no signs of divergence.
  Training loss trajectories (Figure 8) confirm this: Transformer loss becomes noisy and spiky at moderate learning rates, while Multiscreen loss remains smooth even at very high learning rates.
Gradient norm dynamics (Figure 9) explain the stability difference: Multiscreen gradients decay rapidly to near-zero with very low variance, while Transformer gradients maintain a non-zero floor with frequent large spikes, leading to instability at high learning rates.

![Figure 9: Gradient norm dynamics during training for Transformer and Multiscreen. Multiscreen exhibits rapidly decaying gradient norms with minimal variance, while Transformer maintains a non-zero gradient floor with occasional spikes. For visualization, values above 1 are clipped and shown with $\\times$ markers.](images/9.jpg)
*该图像是图表，展示了Transformer和Multiscreen模型在训练过程中的梯度范数动态。图中显示，Multiscreen（蓝色）梯度范数迅速衰减且方差很小，而Transformer（灰色）维持了非零的梯度底线，有偶尔的峰值。对于可视化，超过1的值被裁剪，并用`imes`标记表示。*

This stability allows Multiscreen to be trained with a learning rate of $2^{-4}$ (~0.0625), which is 6x to 30x larger than the learning rates used for Transformer baselines (3e-4 to 1e-3), leading to faster training convergence.
### 6.1.3. Long-Context Perplexity
Figure 5 from the paper compares long-context perplexity of 353M Transformer and 286M Multiscreen models:

![Figure 5: Long-context perplexity comparison between $3 5 3 \\mathbf { M }$ Transformer and 286M Multiscreen models. The horizontal axis is context position, and the vertical axis is perplexity. The left panel shows the base models, while the right panel shows models after long-context continual pretraining. The black curve corresponds to Multiscreen, while colored curves correspond to Transformer with different RoPE scaling factors. Shaded regions indicate one standard deviation across three independently trained models. The dashed and dotted vertical lines indicate the sequence lengths used during base pretraining $( 2 ^ { 1 2 } )$ and long-context continual pretraining $( 2 ^ { 1 5 } )$ , respectively.](images/5.jpg)
*该图像是一个比较 $2^{12}$ 基础模型和 $2^{15}$ 长期持续预训练模型的长上下文困惑度图。左侧展示了基础模型的困惑度，右侧展示了经过长期预训练后的模型。黑色曲线对应于286M Multiscreen模型，彩色曲线则对应于不同RoPE缩放因子的353M Transformer模型，虚线指示了序列长度。*

Key findings:
- Multiscreen (black curve) maintains stable perplexity even at context lengths far exceeding the base training length of $2^{12}$ (4096 tokens) and long-context pretraining length of $2^{15}$ (32768 tokens).
- Transformer models exhibit sharp spikes in perplexity once context length exceeds the training range, even when using RoPE scaling factors to improve extrapolation. Higher RoPE scaling factors delay the breakdown but increase overall perplexity.
  This confirms that Multiscreen's conditional positional encoding design eliminates the positional extrapolation failure common to Transformer models.
### 6.1.4. Retrieval Performance (ABCDigits)
Figure 6b from the paper shows retrieval accuracy heatmaps across context lengths and target depths:

![Figure 6: (a) Example prompt for ABCDigits. (b) Retrieval accuracy heatmaps over context length (columns) and target depth (rows). Columns correspond to the two training settings: base models trained with context length $2 ^ { 1 2 }$ (left) and models after continual pretraining with context length $2 ^ { 1 5 }$ (right). Rows correspond to 353M Transformer (top), 286M Multiscreen (middle), and 28M Multiscreen (bottom). Colors indicate exact-match retrieval accuracy. For Transformer, each cell shows accuracy under the best-performing RoPE scaling factor selected from multiple candidates, and the number inside the cell indicates the selected factor. A dash ("-") indicates that no correct retrieval occurred. The dashed and dotted vertical lines mark the context lengths used during base pretraining $( 2 ^ { 1 2 } )$ and long-context continual pretraining $( 2 ^ { 1 5 } )$ , respectively.](images/6.jpg)
*该图像是图表，展示了图6中的两个部分：(a) ABCDigits的示例提示与(b) 随着上下文长度（列）和目标深度（行）变化的检索准确率热图。上半部分显示353M Transformer模型的准确率，底部是286M和28M Multiscreen模型。颜色表示准确匹配的检索精度，特定单元格中的数字是所选的RoPE缩放因子，"-"表示未成功检索。虚线标记了用于基础和长期预训练的上下文长度。*

Key findings:
- The 286M Multiscreen model achieves near-perfect (~100%) retrieval accuracy across all evaluated context lengths (up to $2^{17}$ = 131072 tokens, 8x longer than base training length) even without long-context pretraining.
- The smaller 28M Multiscreen model maintains ~80% accuracy even at the longest context length of 131072 tokens.
- The 353M Transformer model (even with the best RoPE scaling factor selected for each context length) performs very poorly: retrieval accuracy degrades sharply once context length exceeds training length, and even within the training length, accuracy is much lower than Multiscreen.
  Notably, the 28M Multiscreen model consistently outperforms the 353M Transformer model on retrieval accuracy, despite having 12x fewer parameters and higher validation loss. This confirms that validation loss (next-token prediction performance) does not reliably reflect a model's retrieval ability, which is a critical finding for LLM evaluation.
To understand how screening works in practice, Figure 10 from the paper visualizes distance-aware relevance maps for all tiles in a small Multiscreen model:

![Figure 10: Distance-aware relevance maps across layers and heads. Each map shows the distanceaware relevance, with rows and columns corresponding to query and key positions. Darker gray regions indicate positions outside the learned screening window. Each tile is annotated with its layer and head indices, the learned window width $w$ , the acceptance width $1 / r$ , and the fraction of nonzero relevance values $\\mathrm { P r } ( \\alpha _ { i j } ^ { \\mathrm { d } } > 0 )$ within the window.](images/10.jpg)
*该图像是示意图，展示了不同层和头的距离感知相关性图。每个图中行和列对应查询和键的位置，深色区域表示超出学习的筛选窗口的位置。每个小块中标注了其层和头的索引、学习的窗口宽度 $w$ 、接受宽度 $1 / r$ 以及窗口内非零相关性值的比例 $\mathrm{Pr}(\alpha_{ij}^{d} > 0)$。*

Different tiles learn different window sizes: some focus on very local context (small $w$), while others use very large windows (near full context). Many tiles have high sparsity (large fraction of zero relevance values), as they discard most irrelevant keys.
### 6.1.5. Inference Latency
Table 4 from the paper shows inference latency for next-token prediction with 100,000 token context, measured on an NVIDIA RTX 4090 GPU:

| Model | Base | After continual pretraining |
| --- | --- | --- |
| 353M Transformer | $4.04 \pm 0.03$ s | $4.05 \pm 0.04$ s |
| 286M Multiscreen | $1.72 \pm 0.05$ s | $1.26 \pm 0.06$ s |

Key findings:
- Base Multiscreen is 2.3x faster than Transformer for 100K context inference.
- After long-context continual pretraining, Multiscreen is 3.2x faster than Transformer.
  The additional speedup after pretraining comes from the fact that more tiles learn finite window sizes during long-context pretraining, so fewer tiles require full context computation at inference time. After pretraining, only 4.7% of tiles have infinite windows (full context access), compared to 9.4% for base models.
## 6.2. Ablation Studies / Parameter Analysis
The paper implicitly validates the effectiveness of each component of Multiscreen through the experimental results:
1.  **Screening threshold (r)**: The Trim-and-Square transform with learned acceptance width $1/r$ is critical for discarding irrelevant keys: removing the threshold would lead to dense relevance values, eliminating the computational efficiency and stability benefits.
2.  **MiPE conditional positional encoding**: Disabling MiPE entirely would reduce performance on local tasks that require positional information, while always enabling MiPE would lead to extrapolation failure for long contexts, as seen in Transformer baselines.
3.  **Learned window sizes (w)**: Fixing all window sizes to a single value would either reduce long-range retrieval performance (if fixed to small windows) or eliminate the inference speedup (if fixed to large windows). The learned per-tile windows balance performance and efficiency.

    ---

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces Multiscreen, a new language model architecture built around the screening mechanism that implements absolute query-key relevance, eliminating the core limitation of softmax attention's relative weighting.
The key conclusions are:
1.  Absolute relevance-based screening outperforms redistribution-based softmax attention on parameter efficiency, training stability, long-context performance, and inference efficiency simultaneously.
2.  Conditional positional encoding (MiPE) eliminates the positional extrapolation failure that plagues long-context Transformer models.
3.  Next-token prediction metrics like validation loss do not reliably reflect a model's retrieval ability, so specialized benchmarks like ABCDigits are needed to evaluate long-context capabilities.
4.  Multiscreen achieves up to 3.2x faster inference at 100K context length, making it highly suitable for real-world long-context LLM deployments.
## 7.2. Limitations & Future Work
The authors note several limitations and future research directions:
1.  **Undertraining of large models**: The 4B Multiscreen model was undertrained due to computational constraints, so future work will train larger Multiscreen models with more tokens to fully validate scaling behavior.
2.  **Inference implementation optimization**: The current Triton implementation of the screening module can be further optimized for even longer contexts (1M+ tokens) and to support KV caching, which is standard for Transformer inference deployments.
3.  **Extension to other tasks**: The current Multiscreen architecture is designed for causal language modeling (autoregressive generation). Future work will adapt it to bidirectional tasks like text classification and question answering, and to cross-attention for multimodal models.
## 7.3. Personal Insights & Critique
This paper represents a significant paradigm shift in sequence modeling, moving beyond the 9-year-old softmax attention framework to absolute relevance-based selection. The key insights are highly promising for real-world LLM deployments:
1.  **Practical impact**: The 3.2x inference speedup at 100K context length alone makes Multiscreen extremely valuable for use cases like legal document analysis, long-form code generation, and customer support chatbots that require processing long contexts quickly.
2.  **Evaluation contribution**: The ABCDigits benchmark fills an important gap in LLM evaluation, as it provides a pure measurement of retrieval ability without confounding factors. This will help the community better evaluate and compare long-context models in the future.
3.  **Potential improvements**:
    - The current architecture uses fixed key and value dimensions (16 and 64) across all model scales. Tuning these dimensions for larger models could further improve performance.
    - The acceptance threshold $r$ is learned per tile, not per query. Adding query-dependent thresholding could improve relevance selection for different types of queries.
    - The paper only evaluates on English text. Future work should validate Multiscreen's performance on other languages and non-text modalities (audio, images).
4.  **Unverified assumption**: The paper assumes that unit-length normalization of queries, keys, and values does not hurt performance. It would be useful to conduct an ablation study to confirm that this normalization is indeed beneficial, and that removing it would reduce performance.
    Overall, this paper's core idea of absolute relevance is highly generalizable, and is likely to inspire a new generation of sequence modeling architectures that abandon softmax attention's redistribution framework entirely.