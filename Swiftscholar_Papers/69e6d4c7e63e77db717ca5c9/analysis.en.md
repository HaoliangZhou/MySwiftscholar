# 1. Bibliographic Information

## 1.1. Title
The title of the paper is **"Magic-MM-Embedding: Towards Visual-Token-Efficient Universal Multimodal Embedding with MLLMs"**. This title indicates that the research focuses on a specific model architecture called "Magic-MM-Embedding" designed to create efficient embeddings for multimodal data (specifically text and images) using Multimodal Large Language Models (MLLMs). The key emphasis is on "Visual-Token-Efficiency," suggesting a solution to the computational bottleneck caused by processing too many visual tokens.

## 1.2. Authors
The authors of this paper are **Qi Li, Yanzhe Zhao, Yongxin Zhou, Yameng Wang, Yandong Yang, Yuanjia Zhou, Jue Wang, Zuojian Wang, and Jinxiang Liu**. The corresponding author is marked with an asterisk (*) next to Jinxi Liu. All authors are affiliated with **Honor Device Co., Ltd**, a technology company known for consumer electronics and intelligent devices. This affiliation suggests the research may have practical applications in on-device AI, mobile search, or recommendation systems where computational efficiency is paramount.

## 1.3. Journal/Conference
The paper is published as a preprint on **arXiv** (arXiv:2602.05275). The "Published at (UTC)" timestamp is **2026-02-05**, which is in the future relative to the current training data cutoff, indicating this is a very recent or hypothetical/future-dated paper in the context of the prompt. As an arXiv preprint, it has not yet undergone peer review by a specific conference or journal journal board, but it is presented as a complete research contribution.

## 1.4. Publication Year
The publication year listed is **2026**.

## 1.5. Abstract
The paper addresses the challenge of **universal multimodal retrieval**—finding retrieval items of various modalities (text, images, documents) for a given query. While **Multimodal Large Language Models (MLLMs)** have shown great promise in this domain, their practical application is hindered by the high computational cost of processing a large number of visual tokens (the internal units of data the model processes). The authors propose **Magic-MM-Embedding**, a series of models that achieve high efficiency and state-of-the-art performance. The approach relies on two pillars: (1) an efficient MLLM architecture with **visual token compression** to reduce latency and memory, and (2) a **multi-stage progressive training strategy** to recover and boost performance. This training strategy involves continue pretraining, contrastive pretraining with hard negative mining, and task-aware fine-tuning using an **MLLM-as-a-Judge**. Experiments show the model outperforms existing methods while being more efficient.

## 1.6. Original Source Link
The paper is available on arXiv at the following link:
**https://arxiv.org/abs/2602.05275**
The status is a **preprint**.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the **computational inefficiency** of current state-of-the-art Multimodal Large Language Models (MLLMs) when used for retrieval tasks. MLLMs typically work by converting images into a long sequence of "visual tokens" which are then fed into the language model. For example, a standard image might be broken down into 576 tokens. Processing these tokens, especially within the self-attention mechanism of the Transformer architecture, incurs a computational cost that grows quadratically with the sequence length. This makes deployment in latency-sensitive or resource-constrained environments (like mobile devices or large-scale retrieval systems) very difficult.

The paper identifies that while the field has rapidly advanced in improving the *accuracy* of these models through better data and training strategies, the *efficiency* bottleneck—specifically the redundancy of visual tokens—has been largely overlooked. The authors argue that many of these dense visual tokens are redundant for the final embedding representation, contributing little to semantic quality while costing a lot of computation.

The entry point for the paper is the hypothesis that one can drastically reduce the number of visual tokens (e.g., by 75%) without losing performance, provided the model is trained with a specialized, progressive curriculum designed to distill the essential information into the compressed representation.

## 2.2. Main Contributions / Findings
The primary contributions of the paper are:

1.  **A Novel Framework for Efficiency and Performance:** The authors propose Magic-MM-Embedding, a framework that successfully reconciles the trade-off between efficiency and performance. It demonstrates that a model using aggressive visual token compression can outperform non-compressed counterparts.
2.  **Visual Token Compression Architecture:** They introduce a parameter-free visual token compression module using spatial interpolation (bilinear downsampling). This reduces the visual token sequence length significantly without adding trainable parameters, avoiding the optimization difficulties of learned compressors.
3.  **Coarse-to-Fine Progressive Training Pipeline:** To recover the information lost during compression, the authors design a three-stage training strategy:
    *   **Stage 1 (Restoration):** Generative continue pretraining to align compressed features with the LLM.
    *   **Stage 2 (Contrastive Pretraining):** Large-scale contrastive learning with hard negative mining to build discriminative power.
    *   **Stage 3 (Task-Aware Refinement):** Fine-tuning on curated data where hard negatives are identified and filtered by an external "MLLM-as-a-Judge."
4.  **Synergistic Reranker:** They construct a comprehensive retrieval system by training a reranker on the judge-curated data, which further boosts performance in a two-stage retrieval pipeline.
5.  **State-of-the-Art Results:** The method establishes new state-of-the-art results on benchmarks like MMEB and ViDoRe, often outperforming larger models while using significantly fewer visual tokens and lower latency.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several fundamental concepts in machine learning and computer vision:

*   **Multimodal Large Language Models (MLLMs):** These are AI models capable of understanding and generating content across different modalities, such as text and images. They typically consist of a vision encoder (like a Vision Transformer, ViT) that converts images into feature vectors, and a Large Language Model (LLM) that processes these features alongside text to perform reasoning or generation.
*   **Embeddings:** In the context of AI, an embedding is a vector representation of data (like a word, sentence, or image) in a continuous vector space. Similar items are placed close together in this space. Retrieval systems rely on comparing the embeddings of a query and a database item.
*   **Visual Tokens:** When an image is processed by a Transformer-based vision encoder, it is typically split into a grid of patches. Each patch is converted into a vector called a "token." A standard image might result in hundreds or thousands of these tokens, which are treated similarly to words in a sentence by the model.
*   **Self-Attention Mechanism:** The core component of Transformers. It allows the model to weigh the importance of different tokens in the sequence relative to each other. The computational complexity of self-attention is quadratic with respect to the sequence length ($O(N^2)$), meaning doubling the number of tokens quadruples the computation time.
*   **Contrastive Learning:** A training technique where the model learns to make similar items (positives) close in the embedding space and dissimilar items (negatives) far apart. A common loss function used is **InfoNCE**.
*   **Hard Negative Mining:** A strategy in contrastive learning where "negative" examples (samples that should be far from the query) are chosen to be very similar or confusing to the query (e.g., an image of a different dog breed when the query is a specific dog). This forces the model to learn finer distinctions.
*   **Bilinear Interpolation:** A resampling technique used in image processing to estimate values between known data points. In this paper, it is used to downsample feature maps, effectively merging information from multiple visual tokens into fewer ones.

## 3.2. Previous Works
The paper summarizes the evolution of multimodal representation learning:

*   **CLIP-style Models (Dual-Tower):** Early models like CLIP (Contrastive Language-Image Pre-training) used separate encoders for images and text. They were efficient but had limitations in deep cross-modal interaction and complex reasoning because they processed features independently before fusing them.
*   **Unified Representation Models (UniR, MagicLens):** These extended CLIP to handle interleaved image-text content but still relied on the dual-tower paradigm, limiting fine-grained interaction.
*   **MLLM-based Embeddings:** Recent works like **E5-V**, **VLM2Vec**, and **UniME** shifted to using MLLMs. These models inject visual tokens directly into the LLM, allowing for deep token-level cross-modal fusion and leveraging the LLM's world knowledge.
    *   **VLM2Vec:** Introduced MMEB, a comprehensive benchmark for multimodal embedding.
    *   **UniME & UniME-V2:** Focused on improving discriminative power through data curation, hard negative mining, and multi-stage training.
    *   **LLaVE:** Introduced hardness-weighted contrastive learning.

        The paper notes that while these MLLM-based methods improved accuracy, they inherited the high inference cost of processing long visual token sequences from their general-purpose MLLM backbones.

## 3.3. Technological Evolution
The field has evolved from simple, independent encoders (CLIP) to complex, deeply integrated MLLMs. The focus has shifted from simply aligning modalities to enabling complex reasoning and handling diverse tasks (universal retrieval). However, this increase in capability came with a significant computational cost. The current paper represents the next step in this evolution: **Efficient MLLMs**. It addresses the "scalability crisis" of MLLM-based retrieval by introducing architectural efficiency (compression) without sacrificing the capabilities gained by the MLLM paradigm.

## 3.4. Differentiation Analysis
The core differentiation of this paper lies in its explicit focus on **visual token efficiency**. Previous works (UniME, VLM2Vec) generally used the standard, dense visual token output of the vision encoder (e.g., 576 or 1024 tokens). Magic-MM-Embedding introduces a **parameter-free spatial interpolation module** to compress these tokens aggressively (down to 25% of the original count). Furthermore, unlike simple compression which might hurt performance, the authors introduce a specific **3-stage progressive training pipeline** designed explicitly to mitigate the information loss caused by this compression. This combination of architectural efficiency and a specialized recovery curriculum is distinct from prior work that focused primarily on data scaling or loss function modifications.

# 4. Methodology

## 4.1. Principles
The methodology is built on the principle that **semantic information is dense and can be compressed**. Standard MLLMs use high-resolution visual features primarily for generative tasks (like OCR or detailed captioning), which require pixel-perfect details. For retrieval tasks, however, the global semantic meaning is often sufficient. The authors propose to compress the visual feature map spatially before feeding it into the LLM. To ensure the model doesn't lose the ability to understand the content due to this compression, they employ a "coarse-to-fine" training strategy. This strategy first aligns the compressed features with the LLM's knowledge space (generative training), then sharpens the model's ability to distinguish between similar items (contrastive training), and finally refines it for specific tasks using high-quality data curated by an AI judge.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Problem Formulation and Embedding Extraction
The paper formulates universal multimodal retrieval as learning a unified mapping function. Let $\mathcal{X}$ be the space of inputs, where an input $x \in \mathcal{X}$ can be a query $q$ or a candidate $c$. The input consists of task instructions, visual context, and textual context.

The model uses an MLLM with visual token compression as the encoder $f: \mathcal{X} \to \mathbb{R}^{L \times D}$. It maps an input $x$ to a sequence of hidden states $\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_L$, where each $\mathbf{h}_i \in \mathbb{R}^D$.

To obtain the final embedding vector for the input, the model takes the hidden representation of the last token in the sequence, $\mathbf{h}_L$, and applies $\ell_2$ normalization. This ensures the vector has a unit length, which is standard practice for cosine similarity-based retrieval.

The formula for the final embedding $\mathbf{z}_x$ is:

$$
\mathbf{z}_x = \frac{\mathbf{h}_L}{\|\mathbf{h}_L\|_2}.
$$

Here, $\mathbf{z}_x$ is the final normalized embedding, $\mathbf{h}_L$ is the hidden state of the last token, and $\|\cdot\|_2$ denotes the Euclidean norm (L2 norm).

To train the model to learn a discriminative space where relevant items are close, the authors use the **InfoNCE loss**. For a query $q$, there is a set of candidates $\mathcal{C}_q = \{c_q^+\} \cup \mathcal{C}_q^-$, containing one ground-truth positive $c_q^+$ and a set of negatives $\mathcal{C}_q^-$. The loss function is defined as:

$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\log \frac{\exp(\mathbf{z}_q^{\top} \mathbf{z}_{c_q^+} / \tau)}{\exp(\mathbf{z}_q^{\top} \mathbf{z}_{c_q^+} / \tau) + \sum_{c_q^- \in \mathcal{C}_q^-} \exp(\mathbf{z}_q^{\top} \mathbf{z}_{c_q^-} / \tau)}.
$$

In this formula:
*   $\mathcal{L}_{\mathrm{InfoNCE}}$ is the contrastive loss to be minimized.
*   $\mathbf{z}_q$ is the embedding of the query.
*   $\mathbf{z}_{c_q^+}$ is the embedding of the positive candidate.
*   $\mathbf{z}_{c_q^-}$ represents the embeddings of the negative candidates.
*   $\tau$ is the temperature parameter, which controls the concentration of the distribution (scaling the logits).
*   $\exp(\cdot)$ is the exponential function.
*   $\top$ denotes the transpose operation for the dot product.

    The model aims to maximize the dot product similarity between the query and the positive candidate (numerator) while minimizing it with negatives (denominator).

### 4.2.2. Parameter-free Visual Token Compression
The standard MLLM pipeline involves a visual encoder $e_v$ that takes an image $\mathbf{I}$ and produces a feature map $\mathbf{F} \in \mathbb{R}^{H \times W \times C}$, where $H \times W$ is the spatial resolution and $C$ is the channel dimension. This map is flattened into $N = H \times W$ tokens and fed into the LLM.

The proposed method inserts a compression module between the encoder and the connector. Instead of using a learnable network (like a Q-Former or MLP), they use a direct **bilinear interpolation** strategy. This operation downsamples the spatial dimensions by a factor of $s$.

The compressed feature map $\mathbf{F}'$ is computed as:

$$
\mathbf{F}' = \Phi(\mathbf{F}; H', W') \in \mathbb{R}^{H' \times W' \times C},
$$

where the target dimensions are $H' = H/s$ and $W' = W/s$. $\Phi$ represents the bilinear downsampling operation.

By flattening $\mathbf{F}'$, the number of tokens is reduced from $N$ to $N/s^2$. For example, if $s=2$, the token count drops by 75%. This drastically reduces the quadratic complexity of the LLM's attention mechanism, lowering latency and memory usage without adding any trainable parameters to the compression step itself.

### 4.2.3. Progressive Coarse-to-Fine Training Pipeline
Directly training a compressed model with contrastive loss can be suboptimal because the LLM expects a specific density of visual features. To address this, the authors propose a three-stage pipeline.

**Stage 1: Multimodal Foundational Capability Restoration**
The goal here is alignment, not retrieval. The model undergoes generative continue training on general multimodal instruction-following datasets (32M samples). This restores the model's ability to understand and generate text from the compressed visual features.

The loss function used here is the standard auto-regressive **Next Token Prediction (NTP)** loss. Given a text response sequence $y_1, y_2, \dotsc, y_T$, the loss is:

$$
\mathcal{L}_{\mathrm{NTP}} = -\sum_{t=1}^{T} \log P(y_t | y_{<t}, x).
$$

Here:
*   $\mathcal{L}_{\mathrm{NTP}}$ is the negative log-likelihood loss.
*   $y_t$ is the ground-truth token at position $t$.
*   $y_{<t}$ represents the sequence of previous tokens.
*   $x$ is the multimodal input (image + text).
*   $P(y_t | y_{<t}, x)$ is the probability of predicting token $y_t$ given the history and input.

    This stage bridges the distribution gap between the original dense features and the new compressed features.

**Stage 2: Multimodal Contrastive Pretraining**
With foundational capabilities restored, the model moves to representation learning using a 16M sample retrieval corpus. This stage has two steps:
1.  **Warm-up:** Standard contrastive training using InfoNCE with in-batch negatives.
2.  **Global Hard Negative Mining:** To make the model more discriminative, the authors introduce harder negatives. For each query, they retrieve a ranked list of candidates from the dataset, exclude the ground truth, and use the top-ranked items (which are likely similar but incorrect) as hard negatives in the contrastive loss.

**Stage 3: Task-Aware Finetuning with an MLLM as a Judge**
Standard datasets often contain "false negatives" (items labeled as negative but actually relevant). To fix this and provide challenging training signals, the authors use an expert MLLM (Qwen3-VL-7B) as a judge.

For each query $q$, they retrieve top-$K$ candidates using the Stage 2 model. They then feed each $(q, c_i)$ pair into the Judge MLLM with a specific prompt asking for a binary "yes/no" relevance judgment. The decision is made by comparing the output logits of the 'yes' and 'no' tokens.

If $\mathrm{logit}(\mathbf{yes}) > \mathrm{logit}(\mathbf{no})$, the candidate is deemed relevant (a potential true positive). If $\mathrm{logit}(\mathbf{no}) > \mathrm{logit}(\mathbf{yes})$, it is deemed a hard negative. The model is then fine-tuned using these curated hard negatives while keeping the original ground truth as the sole positive.

### 4.2.4. Synergistic Reranker
To build a complete system, a reranker is trained on top of the Stage 3 model. It uses the judge-curated data: augmented positives $\mathcal{C}_{\text{aug}}^+$ (original + judge-identified) and judge-identified negatives $\mathcal{C}_{\text{juge}}^-$.

The reranker is trained with two objectives:
1.  **Pointwise Reranking:** The model evaluates a single query-candidate pair and outputs "Yes" or "No". The loss is Cross Entropy (CE):
    $$
    \mathcal{L}_{\text{point}} = \mathcal{L}_{\mathrm{CE}}(\mathrm{Yes}, r(q, c^+)) + \mathcal{L}_{\mathrm{CE}}(\mathrm{No}, r(q, c^-)).
    $$
    Here, $r(\cdot)$ is the autoregressive output of the reranker.

2.  **Listwise Reranking:** The model is given a query and a list of candidates (one positive, $M$ negatives). It must output the index $k$ of the positive candidate. The loss is:
    $$
    \mathcal{L}_{\text{list}} = \mathcal{L}_{\mathrm{CE}}(k, r(q, c_1^-, \ldots, c^+, \ldots, c_M^-)).
    $$

The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{point}} + \mathcal{L}_{\text{list}}$.

The following figure illustrates the architecture of the proposed visual token compression model (InternVL3-VTC) and how it is utilized in the embedding and reranking pipeline.

![Figure:Overview the proposed visual-token-cient architecture orniversalmultimodal retrieval.) The proposed MLLM architecture with Visual Token Compression, InternVL3-VTC. (b, c) The proposed inferenceefficient, universal multimodal embedder and reranker, both of which are built upon InternVL3-VTC.](images/2.jpg)
*该图像是示意图，展示了提议的视觉令牌压缩架构 InternVL3-VTC 及其在通用多模态检索中的应用。图中包括了各个组件，如MLP投影器、文本标记器和视觉编码器，以及魔法多模态嵌入和重排方法的示意。*

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize a diverse set of datasets to evaluate the model's universal retrieval capabilities.

**Training Datasets:**
*   **Stage 1 (Restoration):** A 32M sample multimodal instruction-following dataset. Sources include Infinity-MM, Bunny-v1.1, ShareGPT4V, and others covering captioning, grounding, and classification.
*   **Stage 2 (Contrastive Pretraining):** A 16M sample retrieval dataset categorized into:
    *   *Single-Modal:* Text-to-Text (BAAI-MTP) and Image-to-Image (ImageNet-1K).
    *   *Cross-Modal:* Text-to-Image (MSCOCO, VisualNews), Text-to-Visual-Document (Docmatix, Colpali), Image-to-Text (ImageNet-1K, HatefulMemes).
    *   *Fused-Modal:* Image-Text-to-Image (MegaPairs), Image-Text-to-Text (Docmatix, OK-VQA).
*   **Stage 3 (Task-Aware Finetuning):** 1.5M high-quality samples from MMEB-train, Colpali, and VisRAG.

**Evaluation Datasets:**
*   **MMEB Benchmark:** A comprehensive benchmark with 36 sub-datasets covering Classification, VQA, Retrieval, and Grounding.
*   **Visual Document Retrieval (VisDoc):** Includes ViDoRe v1/v2, VisRAG, and ViDoSeek. This tests fine-grained retrieval on documents.
*   **Cross-Modal Retrieval:** Flickr30K, MSCOCO, ShareGPT4V, Urban1K, and SugarCrepe (a compositional reasoning benchmark).

    These datasets were chosen to validate the model's ability to handle diverse modalities (text, image, document) and tasks (classification, retrieval, VQA) effectively.

## 5.2. Evaluation Metrics
The paper uses several standard retrieval metrics:

1.  **Precision@1 (P@1):**
    *   **Conceptual Definition:** This metric measures the accuracy of the top-1 result. It checks if the very first item retrieved by the model is the correct (ground-truth) item. It is a strict metric for ranking quality.
    *   **Mathematical Formula:**
        $$
        P@1 = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q, \text{correct}) = 1)
        $$
    *   **Symbol Explanation:** $|Q|$ is the total number of queries. $\mathbb{I}(\cdot)$ is the indicator function (returns 1 if true, 0 otherwise). $\text{rank}(q, \text{correct})$ is the rank position of the correct item for query $q$.

2.  **Normalized Discounted Cumulative Gain (NDCG@5):**
    *   **Conceptual Definition:** NDCG measures the quality of the ranking by considering the position of the relevant items. It gives higher weight to relevant items appearing higher in the list. The "Discounted" part means that relevance scores are logarithmically reduced as the rank goes down. It is normalized by the ideal ranking (IDCG) so the score is between 0 and 1.
    *   **Mathematical Formula:**
        $NDCG@K = \frac{DCG@K}{IDCG@K}$
        Where `DCG@K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}`.
    *   **Symbol Explanation:** $K$ is the cutoff rank (e.g., 5). $rel_i$ is the relevance score of the item at position $i$ (usually 1 for relevant, 0 for irrelevant). `IDCG@K` is the DCG score of the ideal ranking where all relevant items are at the top.

3.  **Inference Latency and Memory Footprint:**
    *   These measure the computational efficiency. Latency is the time taken to process a query or candidate (in milliseconds). Memory footprint refers to the GPU memory usage.

## 5.3. Baselines
The paper compares against strong baselines representing different paradigms:
*   **Dual-Tower Models:** CLIP, SigLIP, EVA-CLIP, MagicLens.
*   **MLLM-based Embeddings:** E5-V, VLM2Vec, VLM2Vec-V2, UniME, UniME-V2, LLaVE, QQMM, GME, ColPali, Ops-MM-embedding.
*   **Rerankers:** LamRA-Qwen2, LamRA-Qwen2.5.
    These baselines are representative as they include previous state-of-the-art models on the MMEB and VisDoc benchmarks.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results demonstrate that Magic-MM-Embedding achieves state-of-the-art performance across multiple benchmarks while significantly reducing computational costs.

**MMEB Benchmark:** The proposed model (1.9B and 8.1B versions) outperforms all previous baselines, including the strong UniME-V2 and LLaVE models. Notably, the 1.9B Magic-MM-Embedding (E+R) achieves an overall score of 70.2, surpassing the 8.0B LLaVE-OV (70.3) and the 8.3B UniME-V2 (69.0). This highlights the efficiency of the method, achieving comparable or better results with fewer parameters and significantly fewer visual tokens.

**Visual Document Retrieval (VisDoc):** This is a challenging domain requiring fine-grained understanding. Magic-MM-Embedding achieves state-of-the-art results here as well. The 8.1B model with reranker scores 75.8 overall, beating GME (75.2) and other baselines. This is significant because document retrieval typically requires high-resolution inputs; the fact that a compressed model excels here validates the robustness of the training pipeline.

**Cross-Modal Retrieval:** On datasets like Flickr30K, MSCOCO, and the challenging SugarCrepe (which tests compositional reasoning), Magic-MM-Embedding shows substantial improvements. For instance, on SugarCrepe, the 2B model improves over UniME-V2 by large margins (e.g., 91.6% vs 70.9% on one sub-task).

**Efficiency:** The efficiency analysis shows that Magic-MM-Embedding drastically reduces the number of visual tokens (e.g., from 3699 to 99.6 for LLaVE vs Magic-MM 2B). This leads to a massive reduction in inference latency (e.g., 162.8ms to 29.9ms for MMEB queries).

The following are the results from Table 3 of the original paper:

| Model | Backbone (Model Size) | Classification | VQA | Retrieval | Grounding | IND | OOD | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| # of datasets → | | 10 | 10 | 12 | 4 | 20 | 16 | 36 |
| **Zero-shot Results** | | | | | | | | |
| CLIP [74] | -(0.4B) | 42.8 | 9.1 | 53.0 | 51.8 | 37.1 | 38.7 | 37.8 |
| SigLIP [96] | -(0.9B) | 40.3 | 8.4 | 31.6 | 59.5 | 32.3 | 38.0 | 34.8 |
| EVA-CLIP [79] | -(8.1B) | 56.0 | 10.4 | 49.2 | 58.9 | 38.1 | 45.6 | 43.7 |
| MagicLens [99] | -(0.4B) | 38.8 | 8.3 | 35.4 | 26.0 | 31.0 | 23.7 | 27.8 |
| E5-V [34] | Phi3.5-V (4.2B) | 39.1 | 9.6 | 38.0 | 57.6 | 33.1 | 31.9 | 36.1 |
| E5-V [34] | LLaVA-1.6 (8.4B) | 39.7 | 10.8 | 39.4 | 60.2 | 34.2 | 33.4 | 37.5 |
| **Trained with MMEB** | | | | | | | | |
| VLM2Vec-V1 [35] | Qwen2VL (2.2B) | 59.0 | 49.4 | 65.4 | 73.4 | 66.0 | 52.6 | 59.3 |
| UniME [22] | Phi3.5-V (4.2B) | 54.8 | 55.9 | 64.5 | 81.8 | 68.2 | 52.7 | 64.2 |
| LLaVE [41] | Aquila-VL (2.0B) | 62.1 | 60.2 | 65.2 | 84.9 | 69.4 | 59.8 | 65.2 |
| UniME-V2 (E) [23] | Qwen2VL (2.2B) | 62.1 | 56.3 | 68.0 | 72.7 | 67.4 | 58.9 | 63.6 |
| UniME-V2 (E+) [23] | Qwen2VL (2.2B) | 64.1 | 64.3 | 71.6 | 70.6 | 69.8 | 64.3 | 67.4 |
| **Magic-MM-Embedding (E)** | InternVL3-VTC (1.9B) | 60.9 | 63.3 | 72.2 | 84.6 | 74.7 | 59.5 | 68.0 |
| **Magic-MM-Embedding (E+R)** | InternVL3-VTC (1.9B) | 61.3 | 67.2 | 73.5 | 89.8 | 75.2 | 63.9 | 70.2 |
| VLM2Vec-V1 [35] | Qwen2VL (8.3B) | 62.6 | 57.8 | 69.9 | 81.7 | 65.2 | 56.3 | 65.8 |
| UniME [22] | LLaVA-OV (8.0B) | 66.8 | 66.6 | 70.5 | 90.9 | 74.6 | 65.8 | 70.7 |
| LLaVE [41] | LLaVA-OV (8.0B) | 65.7 | 65.4 | 70.9 | 91.9 | 75.0 | 64.4 | 70.3 |
| QQMM [89] | LLaVA-OV (8.0B) | 66.8 | 66.8 | 70.5 | 90.4 | 74.7 | 65.6 | 70.7 |
| UniME-V2 [23] | LLaVA-OV (8.0B) | 65.3 | 67.6 | 72.9 | 90.2 | 74.8 | 66.7 | 71.2 |
| UniME-V2 (E) [23] | Qwen2VL (8.3B) | 64.0 | 60.1 | 73.1 | 82.8 | 72.0 | 63.0 | 68.0 |
| UniME-V2 (E+R) [23] | Qwen2VL (8.3B) | 63.8 | 66.3 | 73.5 | 75.0 | 71.7 | 65.6 | 69.0 |
| **Magic-MM-Embedding (E)** | InternVL3-VTC (8.1B) | 64.8 | 68.1 | 75.0 | 88.7 | 78.3 | 63.6 | 71.8 |
| **Magic-MM-Embedding (E+R)** | InternVL3-VTC (8.1B) | 64.3 | 70.9 | 75.7 | 90.4 | 78.4 | 65.9 | 72.8 |

The following are the results from Table 4 of the original paper:

| Model | Backbone (Model Size) | VDRv1 | VDRv2 | VR | OOD | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| # of Datasets → | | 10 | 4 | 6 | 4 | 24 |
| GME [102] | Qwen2VL (2.2B) | 86.1 | 54.0 | 82.5 | 43.1 | 72.7 |
| ColPali [18] | Paligemma (2.9B) | 83.6 | 52.0 | 81.1 | 43.1 | 71.0 |
| Ops-MM-embedding-v1 [72] | Qwen2VL (8.3B) | 80.1 | 59.6 | 79.3 | 43.3 | 70.3 |
| VLM2Vec-V2 [71] | Qwen2VL (2.2B) | 75.5 | 44.9 | 79.4 | 39.4 | 65.4 |
| **Magic-MM-Embedding (E)** | InternVL3-VTC (1.9B) | 83.4 | 53.3 | 85.6 | 42.2 | 72.1 |
| **Magic-MM-Embedding (E+R)** | InternVL3-VTC (1.9B) | 84.4 | 56.1 | 87.4 | 41.8 | 73.3 |
| Ops-MM-embedding-v1 [72] | Qwen2VL (8.3B) | 80.1 | 59.6 | 79.3 | 43.3 | 70.3 |
| GME [102] | Qwen2VL (8.3B) | 89.4 | 55.6 | 85.0 | 44.4 | 75.2 |
| LamRA-Qwen2 [59] | Qwen2VL (8.3B) | 22.0 | 11.5 | 37.4 | 21.0 | 23.9 |
| LamRA-Qwen2.5 [59] | Qwen2.5VL (8.3B) | 56.3 | 33.3 | 58.2 | 40.1 | 50.2 |
| VLM2Vec-V2 [71] | Qwen2VL (8.3B) | 78.8 | 52.6 | 82.7 | 42.1 | 69.3 |
| **Magic-MM-Embedding (E)** | InternVL3-VTC (8.1B) | 86.1 | 59.9 | 87.6 | 43.4 | 75.0 |
| **Magic-MM-Embedding (E+R)** | InternVL3-VTC (8.1B) | 86.9 | 60.4 | 89.2 | 43.1 | 75.8 |

The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th rowspan="2">Backbone (Model Size)</th>
<th colspan="3">Short Caption</th>
<th colspan="4">Long Caption</th>
<th colspan="3">Compositional</th>
</tr>
<tr>
<th>Flickr30K</th>
<th>MSCOCO</th>
<th></th>
<th>ShareGPT4V</th>
<th>Urban1K</th>
<th></th>
<th></th>
<th>SugarCrepe</th>
<th></th>
<th></th>
</tr>
<tr>
<th></th>
<th></th>
<th>T → I</th>
<th>I → T</th>
<th>T → I</th>
<th>T → I</th>
<th>I → T</th>
<th>T → I</th>
<th>I → T</th>
<th>Replace</th>
<th>Swap</th>
<th>Add</th>
</tr>
</thead>
<tbody>
<tr>
<td>OpenCLIP [74]</td>
<td>- (0.4B)</td>
<td>67.3</td>
<td>87.2</td>
<td>37.0</td>
<td>58.1</td>
<td>81.8</td>
<td>84.0</td>
<td>47.0</td>
<td>47.0</td>
<td>79.5</td>
<td>62.7</td>
<td>74.9</td>
</tr>
<tr>
<td>CLIP [10]</td>
<td>- (2.5B)</td>
<td>79.5</td>
<td>92.9</td>
<td>51.3</td>
<td>67.3</td>
<td>90.1</td>
<td>93.6</td>
<td>77.8</td>
<td>80.7</td>
<td>86.5</td>
<td>68.9</td>
<td>88.4</td>
</tr>
<tr>
<td>EVA-CLIP [79]</td>
<td>-(8.1B)</td>
<td>80.3</td>
<td>94.5</td>
<td>52.0</td>
<td>70.1</td>
<td>93.1</td>
<td>91.2</td>
<td>80.4</td>
<td>77.8</td>
<td>85.9</td>
<td>70.3</td>
<td>86.7</td>
</tr>
<tr>
<td>E5-V [34]</td>
<td>Phi3.5-V (4.2B)</td>
<td>72.2</td>
<td>79.6</td>
<td>44.7</td>
<td>53.4</td>
<td>86.0</td>
<td>88.5</td>
<td>83.8</td>
<td>83.6</td>
<td>88.2</td>
<td>66.6</td>
<td>75.3</td>
</tr>
<tr>
<td>VLM2Vec [35]</td>
<td>Qwen2-VL (2.2B)</td>
<td>69.3</td>
<td>89.6</td>
<td>40.0</td>
<td>62.5</td>
<td>78.1</td>
<td>88.2</td>
<td>78.7</td>
<td>83.9</td>
<td>67.2</td>
<td>46.5</td>
<td>66.4</td>
</tr>
<tr>
<td>UniME [22]</td>
<td>Qwen2-VL (2.2B)</td>
<td>74.9</td>
<td>90.6</td>
<td>44.0</td>
<td>63.5</td>
<td>83.6</td>
<td>88.6</td>
<td>83.3</td>
<td>83.2</td>
<td>65.6</td>
<td>45.2</td>
<td>65.7</td>
</tr>
<tr>
<td>UniME-V2 [23]</td>
<td>Qwen2-VL (2.2B)</td>
<td>79.8</td>
<td>89.9</td>
<td>53.7</td>
<td>65.1</td>
<td>91.6</td>
<td>94.2</td>
<td>95.6</td>
<td>92.2</td>
<td>70.9</td>
<td>51.2</td>
<td>70.2</td>
</tr>
<tr>
<td>Magic-MM-Embedding_InternVL3-VTC (1.9B)</td>
<td></td>
<td>84.4</td>
<td>93.0</td>
<td>61.4</td>
<td>75.8</td>
<td>97.2</td>
<td>97.3</td>
<td>98.4</td>
<td>97.8</td>
<td>91.6</td>
<td>82.6</td>
<td>94.2</td>
</tr>
<tr>
<td>E5-V [34]</td>
<td>LLaVA-1.6 (8.4B)</td>
<td>77.3</td>
<td>85.7</td>
<td>49.1</td>
<td>57.6</td>
<td>85.1</td>
<td>82.1</td>
<td>88.9</td>
<td>83.2</td>
<td>86.3</td>
<td>68.7</td>
<td>66.9</td>
</tr>
<tr>
<td>VLM2Vec [35]</td>
<td>Qwen2-VL (8.3B)</td>
<td>80.0</td>
<td>94.2</td>
<td>49.2</td>
<td>68.5</td>
<td>78.5</td>
<td>90.4</td>
<td>94.0</td>
<td>94.2</td>
<td>70.0</td>
<td>51.7</td>
<td>72.2</td>
</tr>
<tr>
<td>UniME [22]</td>
<td>Qwen2-VL (8.3B)</td>
<td>80.8</td>
<td>92.7</td>
<td>50.9</td>
<td>69.8</td>
<td>86.5</td>
<td>93.8</td>
<td>95.3</td>
<td>94.0</td>
<td>68.8</td>
<td>53.0</td>
<td>69.8</td>
</tr>
<tr>
<td>UniME [22]</td>
<td>LLaVA-OV (8.0B)</td>
<td>83.3</td>
<td>94.4</td>
<td>54.8</td>
<td>74.0</td>
<td>93.9</td>
<td>89.3</td>
<td>94.3</td>
<td>95.5</td>
<td>80.5</td>
<td>65.5</td>
<td>82.2</td>
</tr>
<tr>
<td>UniME-V2 [23]</td>
<td>Qwen2-VL (8.3B)</td>
<td>84.6</td>
<td>93.5</td>
<td>57.3</td>
<td>70.3</td>
<td>94.3</td>
<td>95.2</td>
<td>97.2</td>
<td>96.3</td>
<td>77.8</td>
<td>62.2</td>
<td>79.0</td>
</tr>
<tr>
<td>UniME-V2 [23]</td>
<td>LLaVA-OV (8.0B)</td>
<td>85.5</td>
<td>93.7</td>
<td>60.9</td>
<td>74.1</td>
<td>95.1</td>
<td>94.1</td>
<td>96.3</td>
<td>96.7</td>
<td>88.6</td>
<td>73.7</td>
<td>90.5</td>
</tr>
<tr>
<td>Magic-MM-Embedding InternVL3-VTC (8.1B)</td>
<td></td>
<td>82.9</td>
<td>93.1</td>
<td>63.2</td>
<td>79.3</td>
<td>98.5</td>
<td>98.3</td>
<td>98.5</td>
<td>98.7</td>
<td>92.6</td>
<td>86.9</td>
<td>95.1</td>
</tr>
</tbody>
</table>

The following are the results from Table 6 of the original paper:

| Model | Backbone (Model Size) | #VTq (MMEB) | lq (ms) (MMEB) | #VTc (MMEB) | lc (ms) (MMEB) | #VTq (VisDoc) | lq (ms) (VisDoc) | #VTc (VisDoc) | lc (ms) (VisDoc) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VLM2Vec [35] | Phi3.5-V (4.2B) | 757.0 | 99.4 | 757.0 | 85.9 | 0 | 34.0 | 757.0 | 128.6 |
| GME [102] | Qwen2VL (2.2B) | 362.8 | 46.8 | 256.0 | 34.5 | 0 | 19.3 | 1024.0 | 153.8 |
| LLaVE [41] | Aquila-VL (2.0B) | 3699.0 | 162.8 | 3699.0 | 143.0 | 0 | 18.5 | 3699.0 | 233.6 |
| InternVL3 [107] | InternVL3 (1.9B) | 398.4 | 37.1 | 256.0 | 29.2 | 0 | 19.8 | 1280.0 | 103.6 |
| **Magic-MM-Embedding** | **InternVL3-VTC (1.9B)** | **99.6** | **29.9** | **64.0** | **26.1** | **0** | **19.7** | **320.0** | **57.3** |
| VLM2Vec [35] | LLaVA-1.6 (8.4B) | 2928.0 | 332.3 | 2928.0 | 278.9 | 0 | 32.4 | 2928.0 | 458.1 |
| GME [102] | Qwen2VL (8.3B) | 362.8 | 82.2 | 256.0 | 56.7 | 0 | 26.6 | 1024.0 | 268.2 |
| LamRA [59] | Qwen2.5VL (8.3B) | 362.8 | 83.4 | 256.0 | 61.6 | 0 | 28.9 | 1024.0 | 251.7 |
| UniME-V2 [23] | LLaVA-OV (8.0B) | 7371.0 | 906.9 | 7371.0 | 788.1 | 0 | 32.1 | 7371.0 | 1341.1 |
| InternVL3 [107] | InternVL3 (8.1B) | 398.4 | 76.7 | 256.0 | 55.9 | 0 | 33.8 | 1280.0 | 260.4 |
| **Magic-MM-Embedding** | **InternVL3-VTC (8.1B)** | **99.6** | **50.9** | **64.0** | **40.6** | **0** | **33.8** | **320.0** | **94.8** |

## 6.2. Ablation Studies / Parameter Analysis
The authors conduct thorough ablation studies to validate the design choices.

**Progressive Training Pipeline (Table 7):**
The study confirms the necessity of each stage.
*   **Warm-up (Stage 2):** Baseline MMEB score 62.9, VisDoc 68.4.
*   **+ Global Hard Negative Mining:** Improves to 65.4 (MMEB) and 70.7 (VisDoc). This validates that hard negatives are crucial for discriminative power.
*   **+ MLLM-Judge Finetuning (Stage 3):** Further improves to 68.0 (MMEB) and 72.1 (VisDoc). This shows that data curation by the judge helps adapt to complex tasks.
*   **+ Reranker:** Final scores 70.2 (MMEB) and 73.3 (VisDoc). The reranker provides the final boost.

**Hard Negatives (Table 8):**
The authors compare MLLM-based hard negatives vs. simple rule-based negatives (top-K retrieved excluding ground truth). MLLM-based negatives consistently outperform rule-based ones across different counts ($n=4, 8, 12, 16, 20$). The optimal number of hard negatives is found to be around 12-16.

**Visual Token Compression for Training Efficiency (Table 9):**
This ablation compares the training time of the vanilla InternVL3 vs. the compressed InternVL3-VTC.
*   **Vanilla:** 52h 43m 35s training time.
*   **VTC (Ours):** 22h 57m 6s training time.
    The compressed model trains **~2.3x faster** while achieving slightly better performance (63.7 vs 62.9 MMEB). This proves that compression is beneficial not just for inference, but also for training efficiency.

**LoRA Rank (Table 10):**
The authors test LoRA ranks of 8, 16, and 32. Rank 16 yields the best average performance. Higher ranks (32) lead to a slight decline, possibly due to overfitting.

The following are the results from Table 7 of the original paper:

| Stage 2 (Warm-Up) | Stage 2 (Global-HNM) | Stage 3 (MLLM-Judge-FT) | Inference (Reranker) | MMEB | VisDoc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| X | | | X | 62.9 | 68.4 |
| X | X | | X | 65.4 | 70.7 |
| X | X | X | X | 68.0 | 72.1 |
| X | X | X | ; | 70.2 | 73.3 |

The following are the results from Table 8 of the original paper:

| #HN (n) | MMEB (MLLM-based HN) | VisDoc (MLLM-based HN) | Avg. (MLLM-based HN) | MMEB (Rule-based HN) | VisDoc (Rule-based HN) | Avg. (Rule-based HN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 65.5 | 70.6 | 68.1 | 65.5 | 70.6 | 68.1 |
| 4 | 67.4 | 71.7 | 69.5 | 65.7 | 69.5 | 67.6 |
| 8 | 67.8 | 71.9 | 69.9 | 66.5 | 70.6 | 68.6割 |
| 12 | 68.0 | 72.1 | 70.0 | 67.0 | 70.1 | 68.5 |
| 16 | 67.9 | 72.1 | 70.0 | 65.9 | 70.7 | 68.3 |
| 20 | 67.6 | 71.9 | 69.8 | 67.1 | 70.5 | 68.8 |

The following are the results from Table 9 of the original paper:

| Backbone | Training Duration | MMEB | VisDoc | Global Batch Size |
| :--- | :--- | :--- | :--- | :--- |
| InternVL3 (vanilla) | 52h 43m 35s | 62.9 | 68.4 | 6144 |
| InternVL3-VTC (ours) | 22h 57m 6s | 63.7 | 68.5 | 3456 |

The following are the results from Table 10 of the original paper:

| LoRA Rank | MMEB | VisDoc | Avg. |
| :--- | :--- | :--- | :--- |
| 8 | 62.9 | 68.0 | 65.5 |
| 16 | 62.9 | 68.4 | 65.7 |
| 32 | 62.6 | 67.6 | 65.1 |

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully addresses the computational bottleneck of MLLM-based retrieval by proposing **Magic-MM-Embedding**. By introducing a parameter-free visual token compression module and a carefully designed three-stage progressive training pipeline, the authors demonstrate that it is possible to drastically reduce the number of visual tokens (by 75%) without sacrificing, and in fact improving, retrieval accuracy. The method achieves state-of-the-art results on MMEB and VisDoc benchmarks while offering significant improvements in inference latency and memory footprint compared to existing MLLM embedders. The work highlights that with the right training strategy, efficient architectures can outperform dense, computationally expensive ones.

## 7.2. Limitations & Future Work
The authors do not explicitly list limitations in the conclusion, but the text implies that the method relies on the effectiveness of the progressive training to recover information lost by compression. If the training data or pipeline were insufficient, the compression could be detrimental. Future work could involve exploring different compression ratios or adaptive compression mechanisms based on the input complexity. Additionally, while the model is efficient, the use of an external MLLM-as-a-Judge adds overhead to the data curation process; automating this more efficiently could be a future direction.

## 7.3. Personal Insights & Critique
The paper presents a compelling solution to a very real problem in the deployment of large multimodal models. The insight that "semantic information is dense" challenges the prevailing trend of simply scaling up resolution and token count.

*   **Strengths:** The rigorous ablation studies validating each stage of the training pipeline are a strong point. The separation of "Restoration" (generative) and "Discrimination" (contrastive) training is a logical and effective approach for handling compressed features. The results on VisDoc are particularly impressive, as document retrieval is usually sensitive to resolution.
*   **Potential Issues:** The use of bilinear interpolation is simple and parameter-free, which is great for efficiency, but it might discard high-frequency details that are critical for some very fine-grained tasks not covered in the benchmarks. The paper claims the training pipeline recovers this, but there might be an upper bound to how much compression is tolerable.
*   **Transferability:** The idea of "compress-then-train-progressive" is highly transferable. It could be applied to other MLLM tasks beyond retrieval, such as Video Question Answering (VQA) or long-context reasoning, where token reduction is also crucial. The "MLLM-as-a-Judge" concept for data curation is also a versatile tool for improving dataset quality in weakly supervised settings.

    Overall, the paper makes a significant contribution by shifting the focus from just "better accuracy" to "better accuracy with efficiency," which is essential for the real-world application of these technologies.