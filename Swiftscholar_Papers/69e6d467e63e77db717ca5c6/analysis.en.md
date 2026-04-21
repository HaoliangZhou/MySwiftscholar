# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is the proposal of a novel framework named **PLUME** (Latent Reasoning Based Universal Multimodal Embedding). The research focuses on improving the efficiency and effectiveness of Universal Multimodal Embedding (UME) by replacing traditional explicit chain-of-thought (CoT) reasoning with a compact, continuous latent reasoning process within the model's hidden states.

## 1.2. Authors
The authors of the paper are Chenwei He, Xiangzhao Hao, Tianyu Yang, Yuxiang Ma, Yuheng Jia, Lingxiang Wu, Chaoyang Zhao, Haiyun Guo, and Jinqiao Wang.
*   **Affiliations:**
    *   **Southeast University:** Chenwei He, Yuxiang Ma, Yuheng Jia.
    *   **Institute of Automation, Chinese Academy of Sciences:** Xiangzhao Hao, Tianyu Yang, Lingxiang Wu, Chaoyang Zhao, Haiyun Guo, Jinqiao Wang.
    *   **University of Chinese Academy of Sciences:** Xiangzhao Hao, Tianyu Yang, Lingxiang Wu, Chaoyang Zhao, Haiyun Guo, Jinqiao Wang.
*   **Research Backgrounds:** Based on their affiliations and the content of the paper, the authors are likely researchers specializing in computer vision, natural language processing, and multimodal learning. Their work involves leveraging Multimodal Large Language Models (MLLMs) for advanced retrieval tasks, indicating a strong background in deep learning and representation learning.

## 1.3. Journal/Conference
The paper is currently available as a preprint on **arXiv** (arXiv:2604.02073). The provided metadata indicates a "Published at (UTC)" date of 2026-04-02, which suggests it is a very recent work or a placeholder for a future publication. Given the comprehensive benchmarking and the depth of the methodological contribution, it is highly likely intended for a top-tier conference in the fields of Computer Vision (e.g., CVPR, ICCV) or Machine Learning (e.g., NeurIPS, ICLR).

## 1.4. Publication Year
The publication year listed in the metadata is **2026**.

## 1.5. Abstract
The paper addresses the challenge of **Universal Multimodal Embedding (UME)**, which aims to map heterogeneous inputs (text, images, videos, visual documents) into a shared retrieval space using a single model. Recent approaches have improved UME by generating explicit **chain-of-thought (CoT)** rationales before extracting embeddings, allowing Multimodal Large Language Models (MLLMs) to better infer complex query intent. However, explicit CoT incurs substantial inference overhead (generating hundreds of tokens) and creates a "textual bottleneck" that compresses rich multimodal evidence into narrow text.
The authors propose **PLUME**, a latent reasoning framework that advances UME by replacing verbalized CoT with a short autoregressive rollout of continuous latent states. To support diverse multimodal queries, PLUME introduces a **semantic-anchor-guided transition adapter** that steers the latent rollout along different reasoning trajectories under a fixed computation budget. To stabilize training, PLUME adopts a **progressive explicit-to-latent curriculum** that uses verbalized reasoning as a temporary scaffold and gradually transfers this behavior into hidden-state computation.
On the 78-task **MMEB-v2** benchmark, PLUME outperforms strong explicit-CoT UME baselines while reducing reasoning from hundreds of tokens to fewer than 10 latent steps, delivering over **30x faster inference**. PLUME is particularly effective for retrieval settings with dense, structurally complex evidence, such as video and visual document retrieval.

## 1.6. Original Source Link
The official source is the arXiv preprint server.
*   **Link:** https://arxiv.org/abs/2604.02073
*   **PDF Link:** https://arxiv.org/pdf/2604.02073
*   **Publication Status:** Preprint.

# 2. Executive Summary

## 2.1. Background & Motivation
*   **Core Problem:** The paper aims to solve the efficiency and representational bottleneck in Universal Multimodal Embedding (UME). UME requires mapping diverse inputs (text, image, video, documents) into a shared vector space for retrieval. While recent methods use Multimodal Large Language Models (MLLMs) to generate explicit "chain-of-thought" (CoT) rationales to improve embedding quality for complex queries, this process is computationally expensive (generating hundreds of tokens) and introduces a "textual bottleneck." This bottleneck forces rich, continuous multimodal information (like video frames or document layouts) to be compressed into discrete text tokens, potentially losing fine-grained details.
*   **Importance & Challenges:** Efficient and accurate retrieval is crucial for real-world applications like search engines and databases. Explicit CoT, while improving accuracy, is too slow for practical deployment. Furthermore, converting complex visual or temporal evidence into text loses information. The challenge is to retain the benefits of intermediate reasoning (better understanding of complex queries) without the cost and information loss of generating text.
*   **Entry Point:** The authors propose shifting reasoning from the explicit textual space to the continuous latent space of the model. Instead of generating text tokens, the model performs a short sequence of internal latent state transitions. This preserves the multi-step computation structure of reasoning while avoiding the overhead of text generation.

## 2.2. Main Contributions / Findings
*   **Primary Contributions:**
    1.  **Latent Reasoning Framework for UME:** PLUME internalizes intermediate reasoning into a short continuous latent process (a rollout of hidden states), replacing costly explicit CoT generation. This preserves the benefits of reasoning without the textual bottleneck.
    2.  **Input-Adaptive Architecture:** A "semantic-anchor-guided transition adapter" is introduced. This adapter steers the latent reasoning process differently based on the input's semantic structure, allowing the same fixed computational budget to handle diverse modalities (images, videos, documents) effectively.
    3.  **Progressive Curriculum:** A training strategy that starts with explicit CoT and progressively shifts the reasoning into the latent space. This stabilizes the training process by using explicit text as a scaffold that is gradually removed.
*   **Key Findings:**
    *   PLUME achieves state-of-the-art performance on the MMEB-v2 benchmark (78 tasks), outperforming strong explicit-CoT baselines like UME-R1.
    *   It drastically improves efficiency, reducing reasoning from hundreds of tokens to fewer than 10 latent steps, resulting in over 30x faster inference.
    *   The method is particularly effective for complex modalities like video and visual documents where evidence is dense and structurally complex, areas where explicit CoT struggles with efficiency and information loss.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must be familiar with the following concepts:

*   **Universal Multimodal Embedding (UME):** A task where a single model learns to map inputs from different modalities (e.g., text, images, videos, PDFs) into a shared vector space (embedding space). In this space, semantically similar items from different modalities (e.g., an image of a dog and the text "a puppy") are close to each other. This is fundamental for cross-modal retrieval systems.
*   **Multimodal Large Language Models (MLLMs):** These are models like GPT-4V or LLaVA that extend Large Language Models (LLMs) to understand and generate content involving images and other modalities. They typically use a vision encoder to process images into feature vectors, which are then fed into the LLM alongside text tokens.
*   **Chain-of-Thought (CoT) Reasoning:** A prompting technique where the model is encouraged to generate intermediate reasoning steps (e.g., "First, I see a cat. The cat is sitting...") before producing the final answer. This helps break down complex problems.
*   **Latent Space / Hidden States:** In deep neural networks, the "latent space" refers to the internal representations of the data learned by the model. "Hidden states" are the vectors computed at each layer of the network. These states capture abstract features of the input that are not directly observed.
*   **KV Cache (Key-Value Cache):** An optimization technique used in autoregressive generation (like in Transformers). To avoid recomputing the attention keys and values for previous tokens at every new step, the model stores (caches) them. This is crucial for efficient generation and is reused in PLUME's latent rollout.
*   **Mixture of Experts (MoE):** An architecture where instead of one large network, there are multiple smaller sub-networks ("experts"). A "router" decides which experts are most relevant for the current input and activates only them. This allows the model to specialize for different types of inputs without increasing the total computation for every single input.

## 3.2. Previous Works
The paper discusses three main lines of related work:

1.  **Universal Multimodal Embedding (UME):** Early methods like CLIP (Contrastive Language-Image Pre-training) use dual encoders and contrastive learning. More recent works like VLM2Vec, GME, and UniME leverage MLLMs as backbones to get better semantic alignment. However, most of these extract embeddings in a single forward pass, which limits their ability to handle complex, multi-step reasoning queries.
2.  **Reasoning-Enhanced Embedding:** Methods like TTE (Think-then-Embed), UME-R1, and TRACE explicitly generate a CoT rationale *before* extracting the embedding. This improves accuracy but is slow because generating text is an autoregressive process (generating one token at a time). PLUME is positioned as a more efficient alternative to these methods.
3.  **Latent Reasoning in LLMs:** This is a parallel line of research. Methods like Quiet-STaR and Coconut train models to generate "thoughts" in the hidden state rather than as text. LaSER applies this to text-only retrieval. PLUME extends this idea to the *multimodal* domain, which is more complex due to the heterogeneity of inputs (images, videos, etc.).

**Critical Formula from Related Work (Attention Mechanism):**
To understand how the latent rollout works, one must understand the Transformer's self-attention mechanism, which is the foundation of MLLMs.
The formula for scaled dot-product attention is:
\$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\$
*   `Q, K, V`: Matrices representing Queries, Keys, and Values.
*   $d_k$: The dimension of the keys.
*   The attention score is calculated by taking the dot product of the Query and Key (measuring similarity), scaling it by $\sqrt{d_k}$ to prevent gradients from becoming too small, applying the softmax function to get a probability distribution, and finally multiplying by the Values to get the output.
    In PLUME, the latent rollout leverages this attention mechanism, where the latent state $z^{(k)}$ attends to the previous latent states and the multimodal prefix via the KV cache.

## 3.3. Technological Evolution
The field has evolved from simple contrastive learning (CLIP) to using powerful MLLMs as encoders (VLM2Vec). The current frontier is "reasoning-aware" retrieval, where the model performs intermediate computation. The first wave of this (UME-R1, TTE) used explicit text generation. The paper represents the next wave: moving this reasoning into the latent space for efficiency. PLUME fits into the timeline as an evolution from explicit reasoning (UME-R1) to latent reasoning, specifically adapted for the complexity of multimodal inputs.

## 3.4. Differentiation Analysis
The core difference between PLUME and previous reasoning-enhanced methods (like UME-R1) is the *location* and *form* of the reasoning.
*   **UME-R1 (Explicit CoT):** Generates a sequence of discrete text tokens (e.g., "The image shows a cat..."). This is slow and creates a bottleneck where visual information is forced into a textual format.
*   **PLUME (Latent Reasoning):** Performs reasoning:**
    *   **Innovation 1:** It is much faster (no text decoding).
    *   **Innovation 2:** It avoids the textual bottleneck, preserving richer continuous information.
    *   **Innovation 3:** Unlike text-only latent reasoning methods (LaSER), PLUME uses a "semantic-anchor-guided adapter" to handle the diversity of multimodal inputs (e.g., video reasoning vs. image reasoning) within the same model.

# 4. Methodology

## 4.1. Principles
The core principle of PLUME is that "what retrieval needs is intermediate computation, not necessarily verbalized intermediate text." Instead of generating a chain of text tokens, PLUME performs a "latent rollout"—a short sequence of updates to the model's internal hidden state. This preserves the sequential, multi-step nature of reasoning but operates entirely in the continuous vector space. To make this work for diverse inputs, the model uses a router to adapt its computation path based on the input's semantic content. Finally, to train this complex latent behavior, the model starts by learning from explicit text rationales and progressively "distills" this knowledge into the latent rollout.

## 4.2. Core Methodology In-depth (Layer by Layer)
The PLUME framework can be broken down into four main stages: Multimodal Prefix Encoding, Latent State Initialization, Iterative Latent Rollout, and Suffix Decoding/Embedding Extraction.

### 4.2.1. Multimodal Prefix Encoding
The process begins with the input $x$, which can be a combination of text, image, video, or document features. The backbone MLLM processes this multimodal "prefix". The sequence includes a special token $<slt>$ (start-latent-thinking). This token signals the beginning of the latent reasoning block.

The prefix pass produces two crucial outputs that are used throughout the subsequent latent rollout:
1.  **KV Cache, $\mathcal{C}(x)$:** This is the standard Key-Value cache from the transformer's attention layers. It contains the keys and values for all tokens in the multimodal prefix. This cache allows the future latent steps to attend back to the original input (images, text, etc.) without reprocessing them.
2.  **Semantic Anchor, $\mathbf{c}(x)$:** This is a vector extracted from the hidden state at a dedicated $<anchor>$ token in the prefix. This vector serves as a high-level, fixed summary of the input's semantic intent. It is used later by the router to decide how to process the input.

### 4.2.2. Latent State Initialization
The latent reasoning process starts with an initial latent state, denoted as $\mathbf{z}^{(0)}$. This state is initialized from the hidden state of the backbone at the $<slt>$ position.
\$
\mathbf{z}^{(0)} = \mathbf{h}_L
\$
where $\mathbf{h}_L$ is the hidden state at the last position of the prefix (the $<slt>$ token). This initialization makes sense because $\mathbf{h}_L$ already contains a summary of the processed multimodal context.

### 4.2.3. Iterative Latent Rollout
This is the core of the PLUME method. The model performs $K$ steps of latent reasoning. At each step $k$ (from 1 to $K$), two main operations occur.

**Step 1: Latent State Adaptation via the Routed Adapter**
Before feeding the state back into the backbone, it is first processed by a lightweight "Semantic-Anchor-Guided Transition Adapter". This adapter makes the reasoning process input-adaptive.

The router determines which "experts" (specialized sub-networks) should be activated for this step. The routing decision is based on a combination of the current latent state, the semantic anchor, and a step embedding.
The routing weights $\boldsymbol{\pi}^{(k)}$ over $M_e$ experts are calculated as:
\$
\boldsymbol{\pi}^{(k)} = \mathrm{Softmax}\biggl( W_r \left[ \mathbf{z}^{(k-1)} + \mathbf{c}(x) ; \mathbf{e}^{(k)} \right] + \mathbf{b}_r \biggr)
\$
*   $\mathbf{z}^{(k-1)}$: The latent state from the previous step.
*   $\mathbf{c}(x)$: The semantic anchor vector from the prefix. Adding it to the latent state injects global semantic information into the routing decision.
*   $\mathbf{e}^{(k)}$: A learnable "step embedding" that tells the router which step of the rollout it is (e.g., step 1 vs. step 5). This allows the router to behave differently at different stages of reasoning.
*   $[\cdot; \cdot]$: Concatenation operation.
*   $W_r, \mathbf{b}_r$: Learnable weights and biases of the router.
*   $\mathrm{Softmax}$: Converts the raw scores into a probability distribution (summing to 1) over the experts.

    Once the weights $\boldsymbol{\pi}^{(k)}$ are determined, the adapter computes the adapted state $\tilde{\mathbf{z}}^{(k-1)}$. It uses a shared expert $E_0$ and a weighted sum of the top-$K_r$ specialized experts selected by the router.
\$
\tilde{\mathbf{z}}^{(k-1)} = \mathbf{z}^{(k-1)} + E_0(\hat{\mathbf{z}}^{(k-1)}) + \sum_{m \in \mathrm{Top}K_r(\boldsymbol{\pi}^{(k)})} \pi_m^{(k)} E_m(\hat{\mathbf{z}}^{(k-1)})
\$
*   $\hat{\mathbf{z}}^{(k-1)} = \mathrm{LN}(\mathbf{z}^{(k-1)})$: The layer-normalized input state.
*   $E_0(\cdot)$: The shared expert, which is always active.
*   $E_m(\cdot)$: The $m$-th specialized expert.
*   $\pi_m^{(k)}$: The routing weight for expert $m$ at step $k$.
*   The formula shows a residual connection: the output is the original state plus the transformation from the shared expert and the weighted mixture from the top specialized experts. This structure is common in deep networks to help with gradient flow and training stability.

**Step 2: Backbone Forward Pass**
The adapted state $\tilde{\mathbf{z}}^{(k-1)}$ is then fed into the main backbone MLLM as if it were the input embedding for the next token in the sequence. The model performs a single forward pass for the position corresponding to the $k$-th latent step.
\$
\mathbf{z}^{(k)} = \mathcal{B}_{\theta}\Big( \tilde{\mathbf{z}}^{(k-1)}, \ \mathcal{C}^{(k-1)}, \ p_{<slt>} + k \Big)
\$
*   $\mathcal{B}_{\theta}$: The backbone transformer model with parameters $\theta$.
*   $\tilde{\mathbf{z}}^{(k-1)}$: The adapted latent state from the adapter, acting as the input embedding.
*   $\mathcal{C}^{(k-1)}$: The accumulated KV cache. It starts with $\mathcal{C}(x)$ from the prefix and grows with each step. This allows the current step to attend to all previous latent states and the original multimodal prefix.
*   $p_{<slt>} + k$: The positional index for the current step.
*   $\mathbf{z}^{(k)}$: The new latent state, which is the hidden state at the output of this forward pass.

    This process repeats for $K$ steps, producing a sequence of latent states $\mathbf{z}^{(1)}, \dots, \mathbf{z}^{(K)}$. This sequence is the "latent reasoning trace."

### 4.2.4. Suffix Decoding and Embedding Extraction
After the $K$ latent steps, the latent block is closed with an $<elt>$ (end-latent-thinking) token. Immediately after this, a special $<gen>$ token is placed. The final retrieval embedding is extracted from the hidden state of the backbone at this $<gen>$ position.
\$
\mathbf{e}_{\mathrm{gen}}(x) = \mathrm{Norm}(\mathbf{h}_{<\mathrm{gen}>})
\$
*   $\mathbf{h}_{<\mathrm{gen}>}$: The hidden state at the $<gen>$ token.
*   $\mathrm{Norm}(\cdot)$: L2 normalization, which ensures the embedding vector has a unit length. This is standard practice in retrieval to make cosine similarity calculations more stable.

    The paper also mentions an auxiliary "anchor embedding" $\mathbf{e}_{\mathrm{anc}}(x)$ derived from the semantic anchor $\mathbf{c}(x)$. This embedding is used during training to provide an additional supervision signal and help stabilize the router's learning, but it is discarded at inference time.

### 4.2.5. Progressive Explicit-to-Latent Curriculum
Training a model to perform latent reasoning directly is difficult. To solve this, PLUME uses a curriculum learning strategy.
1.  **Stage 0 (Warm-up):** The model is trained with the full explicit CoT rationales. The model learns to generate the reasoning text normally.
2.  **Intermediate Stages (1-3):** The explicit CoT is progressively replaced. The rationale is split into segments. In early stages, the first few segments are replaced by the latent block (the $<slt>...<elt>$ tokens), and the model is trained to continue the explicit reasoning from this latent prefix. In later stages, more of the rationale is replaced by the latent block.
3.  **Final Stage (4):** The entire explicit rationale is removed. The latent block connects directly to the $<gen>$ token. The model now performs fully latent reasoning without any explicit text scaffold.

    This gradual transfer allows the model to learn the complex behavior of latent reasoning in a stable manner, using the explicit text as a guide that is slowly removed.

The following figure (Figure 3 from the original paper) illustrates the PLUME framework, including the semantic-anchor-guided adapter and the progressive curriculum:

![该图像是一个示意图，展示了PLUME框架的结构，包括语义锚定指导的转接适配器和渐进式显式到潜在的课程。该框架通过替代显式推理，利用潜在状态的自回归滚动来支持多模态查询，以提高效率并减少推理时间。图中还包含对比学习和多模态输入的流程。](images/3.jpg)
*该图像是一个示意图，展示了PLUME框架的结构，包括语义锚定指导的转接适配器和渐进式显式到潜在的课程。该框架通过替代显式推理，利用潜在状态的自回归滚动来支持多模态查询，以提高效率并减少推理时间。图中还包含对比学习和多模态输入的流程。*

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted on the **MMEB-v2** benchmark.
*   **Source & Scale:** MMEB-v2 is a comprehensive benchmark for Universal Multimodal Embedding. It extends the original MMEB benchmark by including video and visual document retrieval scenarios. It consists of 78 test tasks across 9 meta-tasks.
*   **Characteristics & Domain:** The benchmark covers a wide range of vision-language retrieval settings:
    *   **Image:** Image classification, QA, retrieval, and grounding.
    *   **Video:** Video classification, QA, retrieval, and multi-modal retrieval.
    *   **Visual Document (VisDoc):** Visual document retrieval (Vidore), Visual RAG (VisRAG), and out-of-distribution (OOD) tasks.
*   **Why Chosen:** This benchmark is ideal because it is large-scale, diverse, and includes the complex tasks (video, documents) where the authors hypothesize PLUME will have the biggest advantage over single-pass or explicit-CoT methods.
*   **Data Sample Example:** The paper provides qualitative examples in the appendix. For instance, a query might be an image of an egg carton with the question "What material is this egg carton made of?", and the model must retrieve the correct answer ("styrofoam" or "cardboard") from a candidate pool. Another example is a video query asking to "find a clip where a person puts a pen similar to others that are already on the table."

## 5.2. Evaluation Metrics
The paper uses two primary metrics, depending on the task type.

1.  **Hit@1 (for Image and Video tasks):**
    *   **Conceptual Definition:** Hit@1 measures the accuracy of the top-1 result. It checks whether the single item that the model ranks as the most similar (the "hit") is the correct ground-truth item. It is a binary metric (1 if correct, 0 if incorrect) averaged over all queries.
    *   **Mathematical Formula:**
        \$
        \mathrm{Hit@1} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\mathrm{rank}_i = 1)
        \$
    *   **Symbol Explanation:**
        *   $N$: The total number of queries in the test set.
        *   $\mathrm{rank}_i$: The rank of the ground-truth item for the $i$-th query (where rank 1 is the best).
        *   $\mathbb{I}(\cdot)$: The indicator function, which is 1 if the condition is true and 0 otherwise.

2.  **NDCG@5 (for Visual Document tasks):**
    *   **Conceptual Definition:** Normalized Discounted Cumulative Gain (NDCG@5) measures the quality of the ranked list of results, considering the top 5 items. Unlike Hit@1, it gives partial credit if the correct item appears in the top 5 but is not ranked first. It also uses a "discount" factor to penalize correct items that appear lower in the list. The "normalized" part scales the score to be between 0 and 1, allowing comparison across queries with different ideal scores.
    *   **Mathematical Formula:**
        \$
        \mathrm{NDCG@5} = \frac{1}{\mathrm{IDCG}} \sum_{j=1}^{5} \frac{2^{\mathrm{rel}_j} - 1}{\log_2(j + 1)}
        \$
    *   **Symbol Explanation:**
        *   $j$: The position in the ranked list (from 1 to 5).
        *   $\mathrm{rel}_j$:ariance score of the item at position $j$. In binary relevance, this is 1 for relevant and 0 for irrelevant.
        *   $\log_2(j + 1)$: The discount logarithm for position $j$. This means a relevant item at position 1 contributes more than one at position 5.
        *   $\mathrm{IDCG}$ (Ideal Discounted Cumulative Gain): The maximum possible DCG score for the query, achieved if the most relevant items are ranked at the top positions. Dividing by IDCG normalizes the score.

## 5.3. Baselines
PLUME is compared against two groups of strong baselines:
1.  **Early UME Methods (Single-Pass):**
    *   **LamRA, VLM2Vec, GME, VLM2Vec-V2, DUME:** These are state-of-the-art UME models that do not use explicit CoT. They extract embeddings in a single forward pass. Comparing against these shows whether the added reasoning in PLUME provides a benefit over efficient, non-reasoning baselines.
2.  **Reasoning-Enhanced UME Methods:**
    *   **UME-R1:** This is the primary baseline. It uses the same backbone (Qwen2-VL-2B) and training data but generates explicit CoT rationales. Comparing against UME-R1 is the most critical experiment, as it directly tests PLUME's claim of being more efficient while maintaining or improving accuracy.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main results are presented in Table 1 of the paper. PLUME achieves an overall score of 61.6 on the MMEB-v2 benchmark.
*   **Comparison with Reasoning Baseline (UME-R1):** PLUME outperforms UME-R1 (60.1) by 1.5 points overall. This is a significant result because it shows that latent reasoning can be *more effective* than explicit CoT, not just faster. The gains are particularly strong in Video (+1.9) and VisDoc (+3.6) modalities, validating the hypothesis that continuous latent reasoning is better for preserving temporal and structural information.
*   **Comparison with Single-Pass Baselines:** PLUME surpasses the best single-pass method, VLM2Vec-V2 (58.0), by a large margin of 3.6 points. This confirms that the intermediate computation is beneficial for retrieval accuracy.
*   **Efficiency:** Table 2 shows the efficiency gains. PLUME uses only 8 latent steps compared to UME-R1's 403 reasoning tokens. This results in a 30.3x speedup in throughput (3.3 samples/s vs. 0.11 samples/s) and a massive reduction in latency (298ms vs. 9023ms). Even compared to the fast single-pass VLM2Vec-V2, PLUME is only 1.9x slower but achieves much higher accuracy, representing a great trade-off.

    The following figure (Figure 1 from the original paper) visualizes the accuracy-efficiency tradeoff. PLUME sits on the Pareto frontier, offering better accuracy than single-pass methods and much better efficiency than explicit-CoT methods.

    ![Figure 1. PLUME achieves a favorable accuracy-efficiency tradeoff on MMEB-v2. The $\\mathbf { X }$ -axis shows inference throughput on a single H20 GPU and the y-axis shows average MMEB-v2 performance.](images/1.jpg)
    *该图像是一个图表，展示了PLUME在MMEB-v2基准测试中的准确性与效率的权衡。X轴表示单个H20 GPU上的推理吞吐量（samples/s），Y轴表示平均MMEB-v2性能。PLUME在不同步骤下的表现相比于UME-R1有显著提升，并且PLUME的推理速度快了30倍。*

## 6.2. Data Presentation (Tables)
The paper provides several tables with detailed results.

**Table 1: Main Comparison on MMEB-v2**
This table shows the performance across Image, Video, and VisDoc groups.
*   **Image:** PLUME (66.3) is slightly behind UME-R1 (66.6) but ahead of all single-pass baselines.
*   **Video:** PLUME (44.1) outperforms UME-R1 (42.2) and all other baselines.
*   **VisDoc:** PLUME (67.5) significantly outperforms UME-R1 (63.9).

    The following are the results from Table 1 of the original paper:

    <table>
    <thead>
    <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Venue</th>
    <th colspan="5">Image</th>
    <th colspan="5">Video</th>
    <th colspan="5">VisDoc</th>
    <th rowspan="2">All</th>
    </tr>
    <tr>
    <th>CLS</th>
    <th>QA</th>
    <th>RET</th>
    <th>GD</th>
    <th>Overall</th>
    <th>CLS</th>
    <th>QA</th>
    <th>RET</th>
    <th>MRET</th>
    <th>Overall</th>
    <th>VDRv1</th>
    <th>VDRv2</th>
    <th>VR</th>
    <th>OOD</th>
    <th>Overall</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>LamRA</td>
    <td>CVPR'25</td>
    <td>59.2</td>
    <td>26.5</td>
    <td>70.0</td>
    <td>62.7</td>
    <td>54.1</td>
    <td>39.3</td>
    <td>42.6</td>
    <td>24.3</td>
    <td>34.6</td>
    <td>35.2</td>
    <td>22.0</td>
    <td>11.5</td>
    <td>37.4</td>
    <td>21.0</td>
    <td>23.9</td>
    <td>40.4</td>
    </tr>
    <tr>
    <td>VLM2Vec</td>
    <td>ICLR'25</td>
    <td>58.7</td>
    <td>49.3</td>
    <td>65.0</td>
    <td>72.9</td>
    <td>59.7</td>
    <td>33.4</td>
    <td>30.5</td>
    <td>20.6</td>
    <td>33.0</td>
    <td>29.0</td>
    <td>49.8</td>
    <td>13.5</td>
    <td>51.8</td>
    <td>33.5</td>
    <td>41.6</td>
    <td>47.0</td>
    </tr>
    <tr>
    <td>GME</td>
    <td>CVPR'25</td>
    <td>54.4</td>
    <td>29.9</td>
    <td>66.9</td>
    <td>55.5</td>
    <td>51.9</td>
    <td>34.9</td>
    <td>42.0</td>
    <td>25.6</td>
    <td>32.4</td>
    <td>33.9</td>
    <td>86.1</td>
    <td>54.0</td>
    <td>82.5</td>
    <td>43.1</td>
    <td>72.7</td>
    <td>54.1</td>
    </tr>
    <tr>
    <td>VLM2Vec-V2</td>
    <td>TMLR'26</td>
    <td>62.0</td>
    <td>56.3</td>
    <td>69.5</td>
    <td>77.3</td>
    <td>64.9</td>
    <td>39.3</td>
    <td>34.3</td>
    <td>28.8</td>
    <td>38.5</td>
    <td>34.9</td>
    <td>75.5</td>
    <td>44.9</td>
    <td>79.4</td>
    <td>39.4</td>
    <td>65.4</td>
    <td>58.0</td>
    </tr>
    <tr>
    <td>DUME</td>
    <td>ICLR'26</td>
    <td>59.3</td>
    <td>55.0</td>
    <td>66.3</td>
    <td>78.0</td>
    <td>62.5</td>
    <td>37.7</td>
    <td>46.6</td>
    <td>17.1</td>
    <td>30.0</td>
    <td>33.2</td>
    <td>67.6</td>
    <td>43.3</td>
    <td>47.1</td>
    <td>33.8</td>
    <td>52.8</td>
    <td>52.7</td>
    </tr>
    <tr>
    <td>Reasoning UME</td>
    <td colspan="10"></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>UME-R1</td>
    <td>ICLR'26</td>
    <td>64.8</td>
    <td>62.8</td>
    <td>67.6</td>
    <td>77.2</td>
    <td>66.6</td>
    <td>44.3</td>
    <td>51.2</td>
    <td>32.9</td>
    <td>39.7</td>
    <td>42.2</td>
    <td>72.4</td>
    <td>46.2</td>
    <td>79.2</td>
    <td>37.2</td>
    <td>63.9</td>
    <td>60.1</td>
    </tr>
    <tr>
    <td>PLUME</td>
    <td>Ours</td>
    <td>66.5</td>
    <td>59.2</td>
    <td>67.6</td>
    <td>79.7</td>
    <td>66.3</td>
    <td>45.0</td>
    <td>52.3</td>
    <td>33.5</td>
    <td>46.7</td>
    <td>44.1</td>
    <td>72.1</td>
    <td>49.8</td>
    <td>78.1</td>
    <td>57.4</td>
    <td>67.5</td>
    <td>61.6</td>
    </tr>
    </tbody>
    </table>

**Table 2: Inference Efficiency**
This table quantifies the speedup. PLUME's latency is 298ms/sample, compared to 9023ms for UME-R1. The throughput is 3.3 samples/s for PLUME vs. 0.11 for UME-R1.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th>Metric</th>
<th>PLUME</th>
<th>UME-R1</th>
<th>VLM2Vec-V2</th>
</tr>
</thead>
<tbody>
<tr>
<td>Reasoning tokens/steps</td>
<td>8</td>
<td>403</td>
<td>0</td>
</tr>
<tr>
<td>Latency (ms/sample)</td>
<td>298±12</td>
<td>9023±187</td>
<td>156±8</td>
</tr>
<tr>
<td>Throughput (samples/s)</td>
<td>3.3±0.1</td>
<td>0.11±0.01</td>
<td>6.4±0.3</td>
</tr>
<tr>
<td>Speedup vs. UME-R1</td>
<td>30.3×</td>
<td>1.0×</td>
<td>−</td>
</tr>
<tr>
<td>Overhead vs. VLM2Vec-V2</td>
<td>1.9×</td>
<td>−</td>
<td>1.0×</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The ablation studies (Table 3, 4, 5) provide strong evidence for the paper's claims.

**Table 3: Component Ablation**
This table validates the contribution of each core component.
*   **w/o Latent Transition:** Removing the iterative rollout causes a drop of 2.8 points, proving that multi-step computation is key.
*   **w/o MoE:** Replacing the mixture-of-experts adapter with a single MLP causes a 2.4 point drop, showing the importance of input-adaptive computation.
*   **w/o Semantic Anchor:** Removing the anchor from the router causes a 1.5 point drop, confirming its role in stabilizing routing.
*   **w/o Curriculum:** This causes the largest drop (-6.8 points), highlighting the necessity of the progressive training strategy.

    The following are the results from Table 3 of the original paper:

    <table>
    <thead>
    <tr>
    <th>Configuration</th>
    <th>Image</th>
    <th>Video</th>
    <th>VisDoc</th>
    <th>All</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Full PLUME</td>
    <td>66.3</td>
    <td>44.1</td>
    <td>67.5</td>
    <td>61.6</td>
    </tr>
    <tr>
    <td>w/o Latent Transition</td>
    <td>63.6</td>
    <td>41.0</td>
    <td>64.8</td>
    <td>58.8</td>
    </tr>
    <tr>
    <td>w/o MoE (single MLP)</td>
    <td>64.2</td>
    <td>41.8</td>
    <td>64.4</td>
    <td>59.2</td>
    </tr>
    <tr>
    <td>w/o Semantic Anchor</td>
    <td>65.4</td>
    <td>42.3</td>
    <td>66.1</td>
    <td>60.1</td>
    </tr>
    <tr>
    <td>w/o Curriculum</td>
    <td>60.2</td>
    <td>36.5</td>
    <td>60.2</td>
    <td>54.8</td>
    </tr>
    </tbody>
    </table>

**Table 4: Effect of Latent Steps K**
This table shows how performance scales with the number of latent steps $K$.
*   Increasing $K$ from 4 to 8 improves accuracy from 59.9 to 61.6, with a corresponding increase in latency. The gain from 6 to 8 is smaller, suggesting diminishing returns.

    The following are the results from Table 4 of the original paper:

    <table>
    <thead>
    <tr>
    <th></th>
    <th>K</th>
    <th>Image</th>
    <th>Video</th>
    <th>VisDoc</th>
    <th>All</th>
    <th>Latency (ms)</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td></td>
    <td>4</td>
    <td>64.3</td>
    <td>43.3</td>
    <td>65.7</td>
    <td>59.9</td>
    <td>232</td>
    </tr>
    <tr>
    <td></td>
    <td>6</td>
    <td>65.9</td>
    <td>43.6</td>
    <td>66.7</td>
    <td>61.1</td>
    <td>268</td>
    </tr>
    <tr>
    <td></td>
    <td>8</td>
    <td>66.3</td>
    <td>44.1</td>
    <td>67.5</td>
    <td>61.6</td>
    <td>300</td>
    </tr>
    </tbody>
    </table>

**Table 5: Adapter Design Ablation**
This table tests specific design choices of the adapter.
*   Removing the shared expert or using Top-1 routing both hurt performance.
*   Removing the semantic anchor $\mathbf{c}(x)$ or the step embedding $\mathbf{e}^{(k)}$ from the router also leads to performance drops.

    The following are the results from Table 5 of the original paper:

    <table>
    <thead>
    <tr>
    <th>Configuration</th>
    <th>Image</th>
    <th>Video</th>
    <th>VisDoc</th>
    <th>All</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Default (Me = 4, Kr = 2, shared)</td>
    <td>66.3</td>
    <td>44.1</td>
    <td>67.5</td>
    <td>61.6</td>
    </tr>
    <tr>
    <td>w/o Shared Expert</td>
    <td>65.4</td>
    <td>42.5</td>
    <td>66.1</td>
    <td>60.3</td>
    </tr>
    <tr>
    <td>Top-1 expert (instead of Top-2)</td>
    <td>65.8</td>
    <td>43.0</td>
    <td>66.8</td>
    <td>60.8</td>
    </tr>
    <tr>
    <td>Router: w/o c(x)</td>
    <td>65.4</td>
    <td>42.3</td>
    <td>66.1</td>
    <td>60.1</td>
    </tr>
    <tr>
    <td>Router: w/o e(k)</td>
    <td>65.7</td>
    <td>41.9</td>
    <td>66.2</td>
    <td>60.4</td>
    </tr>
    </tbody>
    </table>

The following figure (Figure 5 from the original paper) visualizes the activation preferences of the experts, showing that they learn to specialize for different tasks (e.g., Expert 2 for video classification, Expert 1 for QA).

![Figure 5. Activation preferences of specialized experts across image and video retrieval sub-tasks.](images/5.jpg)
*该图像是一个热图，展示了在图像和视频检索子任务中，各个专家的激活倾向。不同专家在不同输入模态下的激活率各异，数据以百分比形式呈现，为理解多模态嵌入提供了重要见解。*

The following figure (Figure 6 from the original paper) compares the latent trajectories of PLUME and UME-R1, showing that PLUME's trajectory is smoother and has less variance.

![Figure 6. Average cosine similarity between intermediate states and the positive target over 200 samples, reported separately on image and video retrieval. PLUME shows a smoother trajectory with consistently smaller variance than UME-R1 across reasoning steps.](images/6.jpg)
*该图像是一个图表，展示了在200个样本上，PLUME与UME-R1在不同推理步骤下的平均余弦相似度。左侧为图像检索的结果，右侧为视频检索的结果，PLUME在推理步骤中显示出更平滑的轨迹和较小的方差。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully introduces PLUME, a novel framework for Universal Multimodal Embedding. By replacing explicit chain-of-thought generation with a compact latent reasoning process, PLUME achieves a superior accuracy-efficiency trade-off. Key contributions include the latent rollout mechanism, a semantic-anchor-guided adapter for adaptive computation, and a progressive curriculum for stable training. The results on the MMEB-v2 benchmark demonstrate that PLUME outperforms state-of-the-art reasoning-enhanced baselines like UME-R1 while being over 30 times faster, particularly excelling in complex modalities like video and visual documents.

## 7.2. Limitations & Future Work
The authors identify a key limitation: PLUME shows a gap compared to UME-R1 on the Image QA subset, particularly on text-rich or knowledge-intensive benchmarks like ChartQA and InfographicsVQA. They hypothesize that this is because these tasks rely on preserving fine-grained textual details, which might be lost when compressing reasoning into a short latent rollout.
*   **Future Work:** The authors suggest that formal interpretability guarantees for continuous latent trajectories remain an open problem. Future research could focus on improving the preservation of fine-grained information in the latent space or designing hybrid models that use latent reasoning for most tasks but can fall back to explicit reasoning for highly granular tasks.

## 7.3. Personal Insights & Critique
*   **Inspirations:** The paper's core idea of "latent reasoning" is highly inspiring. It challenges the prevailing trend of simply generating more text to improve performance and shows that internal computation can be a more powerful and efficient substrate for intelligence. The "semantic-anchor-guided adapter" is a clever solution to the heterogeneity problem in multimodal learning.
*   **Transferability:** The principles of PLUME are likely transferable to other domains beyond retrieval. Any task that currently relies on CoT generation (e.g., complex planning, multi-step tool use) could potentially benefit from a latent reasoning approach to improve speed and reduce token costs.
*   **Potential Issues:** One potential issue is the complexity of the training pipeline. The progressive curriculum, while effective, adds significant complexity to the training process compared to standard fine-tuning. Additionally, the reliance on a "semantic anchor" and a fixed number of steps $K$ might be a limitation for tasks where the required reasoning depth is highly variable.
*   **Unverified Assumptions:** The paper assumes that a fixed number of latent steps ($K=8$) is sufficient for all tasks. While the results support this, it's possible that some extremely complex queries might benefit from a dynamic, variable-length rollout, which is not explored in this work.