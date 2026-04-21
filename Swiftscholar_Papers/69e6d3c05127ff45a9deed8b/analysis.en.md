# 1. Bibliographic Information

## 1.1. Title
The title of the paper is **"CREM: Compression-Driven Representation Enhancement for Multimodal Retrieval and Comprehension"**. This title highlights the core contribution of the work: a model named CREM that utilizes a "compression-driven" approach to improve representations for two distinct but related tasks—multimodal retrieval (finding relevant items) and comprehension (understanding and generating content).

## 1.2. Authors
The authors of this paper are **Lihao Liu**, **Yan Wang**, **Biao Yang**, **Da Li**, **Jiangxia Cao**, **Yuxiao Luo**, **Xiang Chen**, **Xiangyu Wu**, **Wei Yuan**, **Fan Yang**, **Guiguang Ding**, **Tingting Gao**, and **Guorui Zhou**.

The affiliations listed are **Tsinghua University** and **Kuaishou Technology**. Tsinghua University is one of the most prestigious research institutions in China, particularly renowned for its work in computer science and artificial intelligence. Kuaishou Technology is a major technology company known for its short-video platforms, which implies a strong practical interest in multimodal content understanding and recommendation systems.

## 1.3. Journal/Conference
The paper is currently available as a preprint on **arXiv** (arXiv:2602.19091) with a publication timestamp of **2026-02-22**. As a preprint, it has not yet been published in a specific peer-reviewed journal or conference proceedings at the time of this analysis. However, the topic is highly relevant to top-tier AI conferences such as NeurIPS, ICML, CVPR, or ACL, which frequently feature research on Multimodal Large Language Models (MLLMs) and representation learning.

## 1.4. Publication Year
The paper was published in **2026**.

## 1.5. Abstract
The paper addresses the challenge of applying Multimodal Large Language Models (MLLMs) to embedding-based tasks like retrieval. While MLLMs excel at generative tasks (e.g., visual description, VQA), their direct use for retrieval is difficult due to differences in output formats and optimization objectives. Previous methods that fine-tune MLLMs for retrieval often sacrifice their generative capabilities.

The authors propose **CREM (Compression-driven Representation Enhanced Model)**, a unified framework designed to enhance multimodal representations for retrieval while preserving generative ability. The core innovations include:
1.  A **compression-based prompt design** using learnable "chorus tokens" to aggregate multimodal semantics.
2.  A **compression-driven training strategy** that integrates contrastive and generative objectives through "compression-aware attention."

    Experiments show that CREM achieves state-of-the-art retrieval performance on the MMEB benchmark while maintaining strong generative performance on comprehension benchmarks. The findings suggest that generative supervision can improve the representational quality of MLLMs under this new paradigm.

## 1.6. Original Source Link
The paper is available on arXiv at the following link:
**https://arxiv.org/abs/2602.19091**

The publication status is a **preprint**.

# 2. Executive Summary

## 2.1. Background & Motivation
**Core Problem:**
Multimodal Large Language Models (MLLMs) have revolutionized tasks requiring understanding and generation, such as Visual Question Answering (VQA) and image captioning. However, they struggle with embedding-based tasks like **retrieval** (e.g., searching for an image based on a text query). The fundamental issue is a discrepancy in how these models operate: generation relies on predicting the next token in a sequence, while retrieval relies on extracting a fixed, high-quality embedding vector that represents the entire input.

**Why is this important?**
In real-world applications, a single AI system often needs to be versatile—it must be able to generate detailed descriptions *and* efficiently retrieve relevant information from a database. Training separate models for each task is computationally expensive and inefficient. Ideally, we want a single "universal" model that excels at both.

**Challenges in Prior Research:**
Previous attempts to bridge this gap, typically using **contrastive fine-tuning** (a technique to align embeddings of positive pairs), often resulted in a "trade-off." When an MLLM is fine-tuned to be good at retrieval, it tends to lose its ability to generate coherent text or answer questions. This happens because the optimization objectives for generation (predicting tokens) and retrieval (aligning vectors) conflict, or because the training process treats them as entirely separate tasks.

**Entry Point:**
The authors argue that generation and embedding actually rely on shared cognitive mechanisms: **cross-modal alignment** (matching text to image) and **contextual comprehension** (understanding the scene). They hypothesize that if these shared capabilities can be optimized correctly, a model can do both without sacrificing one for the other. Their entry point is the concept of "compression"—condensing rich visual and textual information into a compact set of tokens that can serve as the foundation for both tasks.

## 2.2. Main Contributions / Findings
The paper makes three primary contributions:

1.  **Compression-Based Prompt Design:** The authors introduce a novel prompt structure that inserts learnable "chorus tokens" into the input sequence. These tokens act as a bottleneck that compresses the semantic information from the raw image and text inputs. This compressed representation serves as a shared bridge for both retrieval and generation tasks.
2.  **Compression-Driven Training Strategy:** They propose a training method that uses a "compression-aware attention" mechanism. This mechanism forces the model to generate answers based *only* on the compressed chorus tokens, rather than the raw input, ensuring the compressed representation contains all necessary information. The strategy jointly optimizes for both contrastive learning (for retrieval) and language modeling (for generation).
3.  **Unified Performance Validation:** Through extensive experiments, they demonstrate that CREM achieves state-of-the-art results on the MMEB retrieval benchmark while maintaining (and in some cases improving) generative performance on benchmarks like MMB and MMMU. A key finding is that generative supervision (training the model to generate text) actually helps improve the quality of retrieval embeddings under this unified framework.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must be familiar with several key concepts in Deep Learning and Natural Language Processing (NLP).

*   **Multimodal Large Language Models (MLLMs):** These are models like GPT-4V or LLaVA that extend standard text-based Large Language Models (LLMs) to process visual inputs (images). They typically consist of a **Vision Encoder** (to convert images into feature vectors) and an **LLM** (to process text and image features together).
*   **Embeddings:** In the context of machine learning, an embedding is a vector (a list of numbers) that represents a piece of data (like a word, sentence, or image). Good embeddings capture semantic meaning; similar items should have similar embeddings.
*   **Contrastive Learning:** This is a technique used to train models to learn good embeddings without explicit labels. The core idea is to pull the embeddings of "positive pairs" (e.g., an image and its matching caption) closer together and push apart "negative pairs" (e.g., an image and a random unrelated caption). A common loss function used here is **InfoNCE**.
*   **Next-Token Prediction (Autoregressive Modeling):** This is the standard training objective for LLMs. Given a sequence of tokens (words or parts of words), the model tries to predict the next token. The loss is typically calculated using **Cross-Entropy**, measuring the difference between the predicted probability distribution and the actual next token.
*   **Attention Mechanism:** The core component of Transformer models (like GPT). It allows the model to weigh the importance of different words in a sentence when processing a specific word. The formula for scaled dot-product attention is:
    $$ \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
    Where $Q$ (Query), $K$ (Key), and $V$ (Value) are matrices derived from the input, and $d_k$ is the dimension of the keys.
*   **KV Cache:** During the generation of long sequences, Transformers store the Key and Value matrices of previous tokens to avoid recomputing them. This stored memory is called the KV Cache. Reducing the size of this cache is crucial for efficiency.

## 3.2. Previous Works
The paper discusses several lines of prior research:

*   **MLLMs for Comprehension:** Models like **LLaVA** and **Qwen-VL** are mentioned as state-of-the-art models that integrate vision encoders with LLMs for understanding tasks. They are typically trained via instruction tuning.
*   **Multimodal Representation Learning:** Traditional models like **CLIP** (Contrastive Language-Image Pre-training) learn aligned image and text embeddings using contrastive learning on massive datasets. However, they are not generative.
*   **MLLM-based Embedding Models:** Recent works like **VLM2Vec**, **E5-V**, and **UniME** attempt to adapt powerful MLLMs to serve as embedding models for retrieval. They often use contrastive fine-tuning. The limitation highlighted is that these models lose their generative capabilities after fine-tuning.
*   **Unified Generative Embedding Models:** Works like **CAFe**, **GRITLM**, and **MM-GEM** try to unify generation and embedding. CAFe, for instance, jointly optimizes contrastive and autoregressive losses. However, the authors argue that these methods often simply add two loss functions together without deeply integrating the tasks, leading to suboptimal results where the model still struggles to balance the two objectives.
*   **Multimodal Token Compression:** Techniques like **Q-Former** (from BLIP-2) or **Perceiver Resampler** are used to compress the large number of visual tokens (features) produced by vision encoders into a smaller set to reduce computational cost. The authors draw inspiration from this, viewing compression not just as an efficiency trick, but as a way to create a unified representation for both tasks.

## 3.3. Technological Evolution
The field has evolved from **dual-encoder models** (like CLIP), which are great for retrieval but cannot generate text, to **encoder-decoder or decoder-only MLLMs** (like LLaVA), which are great for generation but poor at retrieval. Recent efforts have focused on "unified" models that attempt to do both. This paper represents the next step in this evolution by moving beyond simply adding losses to deeply integrating the tasks through a shared "compressed" representation space.

## 3.4. Differentiation Analysis
The core difference between CREM and previous unified models (like CAFe) lies in the **mechanism of unification**.
*   **Previous approaches (e.g., CAFe):** Often use different prompts for different tasks (e.g., "Compress this..." for retrieval, "Describe this..." for generation) and simply sum the contrastive loss and generation loss. The tasks are treated somewhat independently.
*   **CREM:** Uses a **single, unified prompt structure** with "chorus tokens." Crucially, it employs a **compression-aware attention mask** that physically forces the generation task to rely on the compressed tokens. This architectural constraint ensures that the "retrieval representation" (the compressed tokens) is actually useful for "generation," creating a stronger, intrinsic link between the two objectives.

# 4. Methodology

## 4.1. Principles
The core principle of CREM is **semantic compression**. The authors argue that the raw visual and textual tokens fed into an MLLM contain a lot of redundancy. By compressing this information into a small set of learnable tokens called **"chorus tokens"** ($\mathcal{U}$), the model creates a dense, information-rich representation.

This compressed representation serves two purposes simultaneously:
1.  **For Retrieval:** It acts as the embedding vector used to calculate similarity scores.
2.  **For Generation:** It acts as the sole context from which the model must generate an answer.

    By forcing the model to perform generation using only this compressed context, the training process ensures that the compression is high-quality (it retains all necessary information). This "compression-driven" strategy aligns the optimization of both tasks: a good embedding for retrieval must also be a good context for generation.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Compression-Based Prompt Design
The method begins by restructuring the input prompt. In a standard MLLM, the prompt might look like: $[Image] [Instruction] -> [Answer]$. CREM inserts a set of learnable **chorus tokens** $\mathcal{U}$ between the embedding instruction and the generation instruction.

The prompt structure is defined as:
`System: You are a helpful assistant...`
$User: <image> [eInst] <chorus> [gInst]$
$Assistant: <answer>$

*   $<image>$ ($\mathcal{V}$): The visual tokens.
*   `[eInst]` ($\tau$): The embedding instruction (e.g., "Represent this image").
*   $<chorus>$ ($\mathcal{U}$): The learnable tokens inserted to aggregate semantics.
*   `[gInst]` ($\mathcal{Q}$): The generation instruction (e.g., "Describe the image").
*   $<answer>$ ($\mathcal{A}$): The generated response.

    The goal is to maximize the mutual information between the chorus tokens and the multimodal inputs. The paper presents the mutual information formula as:

$$ \mathbb { I } _ { \mathcal { V } , \mathcal { T } ; \mathcal { U } } = D _ { \mathrm { K L } } \left( p ( \mathcal { V } , \mathcal { T } , \mathcal { U } ) \parallel p ( \mathcal { V } , \mathcal { T } ) \otimes p ( \mathcal { U } ) \right) $$

Here, $\mathbb{I}$ represents the mutual information. $D_{\mathrm{KL}}$ is the Kullback-Leibler divergence, which measures the difference between two probability distributions. The formula implies that the joint distribution of inputs and chorus tokens should be as close as possible to the product of their marginal distributions, indicating that the chorus tokens capture the essential information of the inputs.

### 4.2.2. Compression-Aware Attention
To ensure the chorus tokens actually function as a bottleneck, the authors modify the standard causal attention mask used in Transformer models. They want the Question ($\mathcal{Q}$) and Answer ($\mathcal{A}$) tokens to attend *only* to the Chorus tokens ($\mathcal{U}$), and *not* to the raw Vision ($\mathcal{V}$) and Text ($\mathcal{T}$) tokens. Conversely, the Chorus tokens must attend to everything to compress the information.

This is achieved by defining a specific attention mask $M_{ij}$, where $i$ is the index of the query token and $j$ is the index of the key token. The paper defines the mask as:

$$ M _ { i j } = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { i ~ } } i \in ( \mathscr { Q } , \mathscr { A } ) { \mathrm { ~ a n d ~ } } j \in ( \mathscr { V } , \mathscr { T } ) , } \\ { 1 ( j \leq i ) , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. } $$

In this formula:
*   $M_{ij}$ determines if token $i$ can attend to token $j$.
*   The condition $i \in (\mathscr{Q}, \mathscr{A}) \text{ and } j \in (\mathscr{V}, \mathscr{T})$ checks if the current token is a Question or Answer token and the target token is a raw Vision or Text token.
*   If this condition is true, $M_{ij} = 0$, meaning attention is **masked** (prevented).
*   Otherwise, $M_{ij} = 1 (j \leq i)$, which follows the standard causal mask (attending only to previous tokens).

    This asymmetric attention design is crucial. It forces the model to distill all necessary information from $\mathcal{V}$ and $\mathcal{T}$ into $\mathcal{U}$, because $\mathcal{Q}$ and $\mathcal{A}$ literally cannot "see" the raw inputs.

The following figure (Figure 2 from the original paper) illustrates the framework and the attention mask design:

![该图像是示意图，展示了论文中提出的压缩驱动训练方法的框架（图a）。图中包括不同类型的标记（如视觉标记、合唱标记、文本标记和问答标记），以及通过压缩感知注意力机制增强多模态表示的过程。图b展示了压缩感知注意力掩码的设计，图c则介绍了生成训练数据的混合策略。整体结构阐明了该模型在检索与生成任务中的应用。](images/2.jpg)
*该图像是示意图，展示了论文中提出的压缩驱动训练方法的框架（图a）。图中包括不同类型的标记（如视觉标记、合唱标记、文本标记和问答标记），以及通过压缩感知注意力机制增强多模态表示的过程。图b展示了压缩感知注意力掩码的设计，图c则介绍了生成训练数据的混合策略。整体结构阐明了该模型在检索与生成任务中的应用。*

### 4.2.3. Compression-Driven Training Strategy
The training strategy integrates two types of data: retrieval pairs and generation samples.

**1. Contrastive Learning for Retrieval:**
For retrieval, the model uses the standard InfoNCE loss. The embedding is derived by pooling the representations of the chorus tokens. The loss function is:

$$ \mathcal { L } _ { \mathrm { r } } = - \log \frac { \phi( \mathbf { h } _ { q } , \mathbf { h } _ { t ^ { + } } ) } { \phi( \mathbf { h } _ { q } , \mathbf { h } _ { t ^ { + } } ) + \displaystyle \sum _ { t ^ { - } \in \mathbb { N } } \phi( \mathbf { h } ) } $$

*   $\mathcal{L}_{\mathrm{r}}$: The retrieval loss.
*   $\phi$: A similarity function (cosine similarity).
*   $\mathbf{h}_q, \mathbf{h}_{t^+}$: The embeddings for the query and the positive target.
*   $\mathbb{N}$: The set of negative samples.
*   $\mathbf{h}_{t^-}$: The embedding of a negative sample.
*   The formula maximizes the similarity of the positive pair relative to all negative pairs.

**2. Stochastic Compression-Driven Language Modeling:**
For generation, the model uses a standard language modeling loss (Cross-Entropy), but with a twist. It stochastically decides whether to condition the generation on the full context or just the compressed chorus tokens. This is controlled by a Bernoulli random variable $z$.

The generative objective is formalized as:

$$ \mathcal { L } _ { \mathrm { g } } = - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \log p _ { C } ( y _ { t } \mid y _ { < t } , \mathcal { U } , 1 _ { z = 0 } ( \mathcal { V } , \mathcal { T } ) ) $$

*   $\mathcal{L}_{\mathrm{g}}$: The generation loss.
*   $T$: The total sequence length.
*   $y_t$: The target token at position $t$.
*   $p_C$: The probability distribution predicted by the model.
*   $1_{z=0}(\mathcal{V}, \mathcal{T})$: This is an indicator function. If $z=0$, it includes the raw tokens $(\mathcal{V}, \mathcal{T})$ in the context. If $z=1$, they are excluded, leaving only $\mathcal{U}$.

    This stochastic approach (sometimes training with full context, sometimes with compressed) helps the model learn to compress information effectively without immediately losing the ability to generate fluently.

**3. Joint Optimization:**
The final loss is a weighted sum of the two objectives:

$$ { \mathcal { L } } = \alpha _ { \mathrm { { r } } } { \mathcal { L } } _ { \mathrm { r } } + \alpha _ { \mathrm { { g } } } { \mathcal { L } } _ { \mathrm { g } } $$

Where $\alpha_{\mathrm{r}}$ and $\alpha_{\mathrm{g}}$ are loss weights to balance the two tasks.

### 4.2.4. Multi-Task Inference Modes
The paper describes three modes of inference enabled by this training:

1.  **Retrieval:** The model computes the embedding by mean pooling the final layer representations of the chorus tokens $\mathcal{U}$. The raw inputs are processed only to feed into the chorus tokens.
2.  **Native Generation:** The model generates answers using the full context (all vision and text tokens), just like a standard MLLM. This mode checks if the model retained its generative capabilities.
3.  **Compressed Generation:** The model generates answers using *only* the chorus tokens $\mathcal{U}$. The raw tokens are discarded. This mode is highly efficient as it drastically reduces the KV cache size.

    The following figure (Figure 3 from the original paper) illustrates these three inference modes:

    ![Figure 3. CREM Inference Modes. (a) Retrieval embeddings are derived from pooled chorus tokens. (b) Native next-token prediction is performed with full access to all vision tokens (Nat.). (c) Efficient inference is achieved by pruning vision tokens and reducing KV caches (Comp.).](images/3.jpg)
    *该图像是示意图，展示了 CREM 模型在多模态检索和生成任务中的推理模式。图(a)展示了多模态检索中从聚合的合唱令牌得到的嵌入，图(b)显示了在多模态语言生成中对所有视觉令牌的完整访问，图(c)描述了基于压缩的多模态语言生成过程，其中包括预填充和解码阶段。*

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize two main categories of datasets to evaluate both retrieval and generation capabilities.

**1. Retrieval Datasets:**
*   **MMEB (Massive Multimodal Embedding Benchmark):** This is the primary dataset for evaluating retrieval. It contains 36 datasets categorized into 4 meta-tasks: Classification (10 tasks), Visual Question Answering (10 tasks), Retrieval (12 tasks), and Visual Grounding (4 tasks).
    *   **Why chosen:** MMEB is a comprehensive benchmark designed to test the generalization of embedding models across a wide range of multimodal tasks, not just simple image-text search. It includes both in-distribution (IND) and out-of-distribution (OOD) data.

**2. Generation Datasets:**
*   **ShareGPT-4V:** A dataset containing high-quality image-text conversations.
*   **Synthetic QA Data:** The authors used an off-the-shelf MLLM (Qwen2.5-VL-7B) to generate Question-Answer pairs based on the images and instructions from the MMEB dataset. This creates "homogeneous" data where the same content is used for both retrieval and generation supervision.

## 5.2. Evaluation Metrics
The paper uses specific metrics for retrieval and generation tasks.

**1. Precision@1 (Retrieval)**
*   **Conceptual Definition:** Precision@1 measures the accuracy of the top result returned by the retrieval system. It answers the question: "Is the correct (ground truth) item the very first item the model retrieves?" This is a strict metric focusing on the model's ability to identify the single best match.
*   **Mathematical Formula:**
    $$ P@1 = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q^+) \leq 1) $$
*   **Symbol Explanation:**
    *   $|Q|$: The total number of queries in the test set.
    *   $q$: A specific query.
    *   $q^+$: The positive (correct) item for query $q$.
    *   $\text{rank}(q^+)$: The rank position of the positive item in the retrieved list.
    *   $\mathbb{I}(\cdot)$: The indicator function, which is 1 if the condition is true and 0 otherwise.

**2. Standard Generation Metrics**
*   For generation benchmarks like **MMBench**, **MMVet**, and **MMMU**, the paper follows the standard evaluation protocols of those benchmarks, which typically involve calculating accuracy based on multiple-choice answers or using GPT-based evaluation for open-ended responses.

## 5.3. Baselines
The paper compares CREM against several strong baselines:
*   **CLIP & OpenCLIP:** Traditional dual-encoder models.
*   **VLM2Vec & VLM2Vec-V2:** State-of-the-art MLLM-based embedding models.
*   **CAFe:** A unified model that jointly optimizes contrastive and autoregressive losses.
*   **UniME & UNITE:** Advanced embedding models utilizing hard-negative sampling and modality-aware training.
*   **mmE5:** A model leveraging synthetic data for multilingual embeddings.

    These baselines are representative because they cover the spectrum from traditional contrastive learning to the latest attempts at unifying generation and retrieval.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The results demonstrate that CREM successfully bridges the gap between retrieval and generation.

**1. Retrieval Performance (MMEB):**
CREM achieves state-of-the-art performance on the MMEB benchmark. For example, in the 2B parameter category, CREM achieves an overall score of **66.7**, outperforming VLM2Vec-V2 (64.9) and CAFe (59.6). In the 7B category, CREM scores **72.1**, beating UNITE (70.3) and mmE5 (69.8).
*   **Analysis:** This validates that the compression-driven training does not degrade retrieval quality; in fact, it improves it. The authors attribute this to the generative supervision (training to generate text from compressed tokens) forcing the model to learn richer, more semantically complete embeddings.

**2. Generative Performance:**
On benchmarks like MMB and MMVet, CREM maintains performance comparable to the original Qwen2-VL base model.
*   **Analysis:** This is a critical finding. Typically, fine-tuning for retrieval destroys generative ability (as seen in the $\text{CREM}_R$ baseline in the paper, which drops to 17.3 on MMVet). CREM's ability to keep scores like MMVet at **45.1** (vs 45.7 for the base) proves that the unified framework successfully preserves the model's generative knowledge.

## 6.2. Data Presentation (Tables)

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Backbone</th>
<th rowspan="2">#Params</th>
<th colspan="4">Per Meta-Task Score</th>
<th colspan="3">Average Score</th>
</tr>
<tr>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>Grounding</th>
<th>IND</th>
<th>OOD</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td># of Datasets →</td>
<td></td>
<td></td>
<td>10</td>
<td>10</td>
<td>12</td>
<td>4</td>
<td>20</td>
<td>16</td>
<td>36</td>
</tr>
<tr>
<td>CLIP [41]</td>
<td>-</td>
<td>0.4B</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>45.4</td>
</tr>
<tr>
<td>OpenCLIP [10]</td>
<td>-</td>
<td>0.4B</td>
<td>56.0</td>
<td>21.9</td>
<td>55.4</td>
<td>64.1</td>
<td>50.5</td>
<td>43.1</td>
<td>47.2</td>
</tr>
<tr>
<td colspan="10">&lt; 3B Models</td>
</tr>
<tr>
<td>VLM2Vec [21]</td>
<td>Qwen2-VL</td>
<td>2B</td>
<td>59.0</td>
<td>49.4</td>
<td>65.4</td>
<td>73.4</td>
<td>66.0</td>
<td>52.6</td>
<td>59.3</td>
</tr>
<tr>
<td>VLM2Vec-V2 [37]</td>
<td>Qwen2-VL</td>
<td>2B</td>
<td>62.9</td>
<td>56.3</td>
<td>69.5</td>
<td>77.3</td>
<td>68.7</td>
<td>60.1</td>
<td>64.9</td>
</tr>
<tr>
<td>GME [56]</td>
<td>Qwen2-VL</td>
<td>2B</td>
<td>54.4</td>
<td>29.9</td>
<td>66.9</td>
<td>55.5</td>
<td>-</td>
<td>-</td>
<td>51.9</tdtd>
</tr>
<tr>
<td>UNITTE [23]</td>
<td>Qwen2-VL</td>
<td>2B</td>
<td>63.2</td>
<td>55.9</td>
<td>65.4</td>
<td>75.6</td>
<td>65.8</td>
<td>60.1</td>
<td>63.3</td>
</tr>
<tr>
<td>LLaVE [24]</td>
<td>Aquila-VL</td>
<td>2B</td>
<td>62.1</td>
<td>60.2</td>
<td>65.2</td>
<td>84.9</td>
<td>69.4</td>
<td>59.8</td>
<td>65.2</td>
</tr>
<tr>
<td>CAFe [50]</td>
<td>LLaVA-OV</td>
<td>1B</td>
<td>59.1</td>
<td>49.1</td>
<td>61.0</td>
<td>83.0</td>
<td>64.3</td>
<td>53.7</td>
<td>59.6</td>
</tr>
<tr>
<td>CREM (Ours)</td>
<td>Qwen2-VL</td>
<td>2B</td>
<td>65.8</td>
<td>60.7</td>
<td>68.3</td>
<td>78.9</td>
<td>70.8</td>
<td>61.5</td>
<td>66.7</td>
</tr>
<tr>
<td colspan="10">&gt; 7B Models</td>
</tr>
<tr>
<td>E5-V [20]</td>
<td>LLaVA-1.6</td>
<td>7B</td>
<td>39.7</td>
<td>10.8</td>
<td>39.4</td>

>
      <td>60.2</td>
      <td>34.2</td>
      <td>33.9</td>
      <td>33.9</td>
    </tr>
    <tr>
      <td>MMRet [58]</td>
      <td>LLaVA-1.6</td>
      <td>7B</td>
      <td>56.0</td>
      <td>57.4</td>
      <td>69.9</td>
      <td>83.6</td>
      <td>68.0</td>
      <td>59.1</td>
      <td>65.8</td>
    </tr>
    <tr>
      <td>VLM2Vec [21]</td>
      <td>Qwen2-VL</td>
      <td>7B</td>
      <td>62.6</td>
      <td>57.8</td>
      <td>69.9</td>
      <td>81.7</td>
      <td>72.2</td>
      <td>57.8</td>
      <td>65.8</td>
    </tr>
    <tr>
      <td>UniME [16]</td>
      <td>LLaVA-OV</td>
      <td>7B</td>
      <td>66.8</td>
      <td>66.6</td>
      <td>70.5</td>
      <td>90.9</td>
      <td>-</td>
      <td>-</td>
      <td>70.7</td>
    </tr>
    <tr>
      <td>UNITE [23]</td>
      <td>Qwen2-VL</td>
      <td>7B</td>
      <td>68.3</td>
      <td>65.1</td>
      <td>71.6</td>
      <td>84.8</td>
      <td>73.6</td>
      <td>66.3</td>
      <td>70.3</td>
    </tr>
    <tr>
      <td>mmE5 [5]</td>
      <td>Llama-3.2-Vision</td>
      <td>11B</td>
      <td>67.6</td>
      <td>62.8</td>
      <td>70.9</td>
      <td>89.7</td>
      <td>72.3</td>
      <td>66.7</td>
      <td>69.8</td>
    </tr>
    <tr>
      <td>CAFe [50]</td>
      <td>LLaVA-OV</td>
      <td>7B</td>
      <td>65.2</td>
      <td>65.6</td>
      <td>70.0</td>
      <td>91.2</td>
      <td>75.8</td>
      <td>62.4</td>
      <td>69.8</td>
    </tr>
    <tr>
      <td>CREM (Ours)</td>
      <td>Qwen2-VL</td>
      <td>7B</td>
      <td>68.3</td>
      <td>69.4</td>
      <td>72.9</td>
      <td>86.1</td>
      <td>75.6</td>
      <td>67.8</td>
      <td>72.1</td>
    </tr>
  </tbody>
</table>

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th>MMB</th>
<th>MMVet</th>
<th>AI2D</th>
<th>Hallusion</th>
<th>MMMU</th>
<th>MMStar</th>
<th>AVG</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8">2B Models</td>
</tr>
<tr>
<td>Qwen2-VL</td>
<td>72.3</td>
<td>45.7</td>
<td>73.8</td>
<td>41.9</td>
<td>41.1</td>
<td>46.1</td>
<td>53.5</td>
</tr>
<tr>
<td>CREMG</td>
<td>73.1</td>
<td>44.5</td>
<td>72.9</td>
<td>41.0</td>
<td>41.2</td>
<td>46.2</td>
<td>53.2</td>
</tr>
<tr>
<td>CREMR</td>
<td>64.3</td>
<td>17.3</td>
<td>66.7</td>
<td>33.6</td>
<td>34.9</td>
<td>43.9</td>
<td>43.4</td>
</tr>
<tr>
<td>CREM</td>
<td>72.5</td>
<td>45.1</td>
<td>72.8</td>
<td>41.2</td>
<td>41.4</td>
<td>45.5</td>
<td>53.1</td>
</tr>
<tr>
<td colspan="8">7B Models</td>
</tr>
<tr>
<td>Qwen2-VL</td>
<td>80.9</td>
<td>58.0</td>
<td>82.2</td>
<td>50.9</td>
<td>53.7</td>
<td>59.5</td>
<td>64.2</td>
</tr>
<tr>
<td>CREMG</td>
<td>80.7</td>
<td>56.8</td>
<td>81.9</td>
<td>49.2</td>
<td>51.7</td>
<td>59.8</td>
<td>63.4</td>
</tr>
<tr>
<td>CREMR</td>
<td>77.3</td>
<td>41.4</td>
<td>80.9</td>
<td>44.5</td>
<td>47.0</td>
<td>56.9</td>
<td>58.0</td>
</tr>
<tr>
<td>CREM</td>
<td>80.5</td>
<td>56.7</td>
<td>81.9</td>
<td>48.8</td>
<td>52.1</td>
<td>59.3</td>
<td>63.2</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The authors conducted several ablation studies to verify the necessity of different components.

**1. Analysis of Retrieval and Generation (Table 3):**
*   **Retrieval Only ($\text{CREM}_R$):** Good retrieval (62.3 MMEB) but terrible generation (43.4 AVG).
*   **Generation Only ($\text{CREM}_G$):** Good generation (53.2 AVG) but poor retrieval (2.9 MMEB).
*   **CREM (Full):** Excellent retrieval (66.7) and good generation (53.1).
*   **Analysis:** This confirms that simply having the data or the architecture isn't enough; the specific **Compression-Driven Training Strategy (CTS)** is required to bridge the gap.

**2. Analysis on Chorus Tokens (Table 4):**
*   **EOS Token:** Using the standard End-of-Sentence token for retrieval leads to poor generation (43.4).
*   **Learnable Chorus Tokens:** Using 1, 4, 8, 16, or 32 tokens significantly improves generation.
*   **Optimal Count:** 16 tokens offer the best balance, achieving 66.7 MMEB and 53.1 Generation. Increasing to 64 gives slightly better generation but hurts retrieval slightly.

    The following are the results from Table 3 of the original paper:

    <table>
    <thead>
    <tr>
    <th rowspan="2">Ret.</th>
    <th rowspan="2">Gen.</th>
    <th rowspan="2">CPD</th>
    <th rowspan="2">CTS</th>
    <th colspan="2">Generation</th>
    <th rowspan="2">Retrieval MMEB</th>
    </tr>
    <tr>
    <th>Nat.</th>
    <th>Comp.</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>✓</td>
    <td></td>
    <td></td>
    <td></td>
    <td>43.4</td>
    <td>-</td>
    <td>62.3</td>
    </tr>
    <tr>
    <td></td>
    <td>✓</td>
    <td></td>
    <td></td>
    <td>53.2</td>
    <td></td>
    <td>2.9</td>
    </tr>
    <tr>
    <td></td>
    <td>✓</td>
    <td></td>
    <td></td>
    <td>47.2</td>
    <td></td>
    <td>66.1</td>
    </tr>
    <tr>
    <td></td>
    <td>✓</td>
    <td></td>
    <td></td>
    <td>53.0</td>
    <td>43.9</td>
    <td>21.1</td>
    </tr>
    <tr>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>52.8</td>
    <td></td>
    <td>65.5</td>
    </tr>
    <tr>
    <td></td>
    <td></td>
    <td></td>
    <td>√</td>
    <td>53.1</td>
    <td>44.2</td>
    <td>66.7</td>
    </tr>
    </tbody>
    </table>

The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>CTok.</th>
<th>MMEB</th>
<th>Nat.</th>
<th>Comp.</th>
<th>Cache(%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>EOS</td>
<td>62.3</td>
<td>43.4</td>
<td></td>
<td>100%</td>
</tr>
<tr>
<td>1</td>
<td>65.6</td>
<td>51.9</td>
<td>38.2</td>
<td>0.07%</td>
</tr>
<tr>
<td>4</td>
<td>66.1</td>
<td>52.3</td>
<td>41.7</td>
<td>0.28%</td>
</tr>
<tr>
<td>8</td>
<td>66.5</td>
<td>52.5</td>
<td>43.1</td>
<td>0.56%</td>
</tr>
<tr>
<td>16</td>
<td>66.7</td>
<td>53.1</td>
<td>44.2</td>
<td>1.12%</td>
</tr>
<tr>
<td>32</td>
<td>66.7</td>
<td>52.9</td>
<td>44.6</td>
<td>2.24%</td>
</tr>
<tr>
<td>64</td>
<td>66.6</td>
<td>53.1</td>
<td>46.2</td>
<td>4.48%</td>
</tr>
</tbody>
</table>

**3. Visualization Analysis (Figure 4):**
The paper visualizes the attention maps of the chorus tokens.
*   **Retrieval-Only:** Sparse attention, focused on few regions (redundant).
*   **Generation-Only:** Broader attention but limited global coverage.
*   **CREM:** Evenly distributed attention across the image, capturing distinct and complementary regions.

    The following figure (Figure 4 from the original paper) shows the visualization of attention maps:

    ![该图像是对比图，展示了三种不同模型在同一场景下的表现：左侧为仅检索模型（Retrieval-Only），中间为仅生成模型（Generation-Only），右侧为CREM模型。各模型在视觉场景中的语义聚合效果有所不同，CREM模型在保持生成能力的同时，实现了更好的检索效果。](images/4.jpg)
    *该图像是对比图，展示了三种不同模型在同一场景下的表现：左侧为仅检索模型（Retrieval-Only），中间为仅生成模型（Generation-Only），右侧为CREM模型。各模型在视觉场景中的语义聚合效果有所不同，CREM模型在保持生成能力的同时，实现了更好的检索效果。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper presents CREM, a novel framework that successfully unifies multimodal retrieval and comprehension within a single MLLM. By introducing "chorus tokens" and a "compression-aware attention" mechanism, CREM forces the model to compress visual and textual information into a dense representation that serves as the foundation for both tasks. Extensive experiments on the MMEB benchmark and various comprehension tasks demonstrate that CREM achieves state-of-the-art retrieval performance while preserving the generative capabilities of the base model. The key takeaway is that generative supervision, when applied through a compression-driven paradigm, can significantly enhance the quality of retrieval embeddings.

## 7.2. Limitations & Future Work
The authors identify several limitations:
1.  **Fixed Token Count:** The number of chorus tokens is fixed after training. Different tasks might benefit from different compression ratios. Future work could explore dynamic or task-aware token allocation.
2.  **OCR Performance:** Compression-based inference leads to poor performance in OCR (Optical Character Recognition) tasks. This is likely because fine-grained text details are lost during compression. Adding OCR-specific data could mitigate this.
3.  **Data Diversity:** The training relies heavily on MMEB and ShareGPT-4V. Incorporating more diverse data could further improve generalization.

## 7.3. Personal Insights & Critique
**Strengths:**
*   **Elegant Solution:** The idea of using compression as the bridge between two conflicting objectives is theoretically sound and elegant. It aligns with the intuition that good embeddings should be "summaries" of the content.
*   **Practical Benefits:** The ability to reduce KV cache size (up to 80x token reduction mentioned in the abstract) is a significant practical advantage for deploying large models in resource-constrained environments.
*   **Rigorous Validation:** The paper does not just claim unification; it provides strong empirical evidence across multiple benchmarks.

**Potential Issues & Critique:**
*   **Complexity:** The training strategy is more complex than standard fine-tuning, requiring specific data mixing and attention mask modifications. This might make it harder to reproduce or adapt to new base models without significant engineering effort.
*   **OCR Degradation:** The drop in OCR performance during compressed inference is a notable drawback. For applications requiring document understanding, the "Comp." mode might be unusable, limiting the utility of the efficiency gains.
*   **Hyperparameter Sensitivity:** The performance seems sensitive to the number of chorus tokens and the loss weights ($\alpha_r, \alpha_g$). Finding the optimal configuration for a new dataset might require extensive tuning.

**Future Inspiration:**
This approach could be applied beyond vision-language models. For instance, in long-text processing, one could use a similar "compression-driven" strategy to create embeddings for document retrieval while maintaining the ability to summarize the document. The concept of "generative supervision for embedding learning" is a powerful direction for future representation learning research.