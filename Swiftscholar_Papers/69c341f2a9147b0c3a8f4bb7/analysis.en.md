# 1. Bibliographic Information
## 1.1. Title
The paper's central topic is the development of `ViCLIP-OT`, the first foundation vision-language model (VLM) optimized for Vietnamese image-text retrieval, which integrates CLIP-style contrastive learning with a novel Similarity-Graph Regularized Optimal Transport (SIGROT) loss to improve cross-modal alignment and reduce modality gaps for low-resource Vietnamese.
## 1.2. Authors
The authors are from Can Tho University, Can Tho, Vietnam:
- Quoc-Khang Tran
- Minh-Thien Nguyen
- Nguyen-Khang Pham (corresponding author)
  All authors are affiliated with the university's computing/AI research groups focused on Vietnamese natural language processing and multimodal learning.
## 1.3. Journal/Conference
The paper is currently published as a preprint on arXiv, a widely used open-access preprint repository for computer science and related fields. It has not yet been formally accepted to a peer-reviewed conference or journal at the time of writing.
## 1.4. Publication Year
The paper was published on arXiv on 26 February 2026 (UTC).
## 1.5. Abstract
The paper addresses the gap that most existing vision-language models are optimized for high-resource languages (e.g., English) and perform poorly on low-resource languages like Vietnamese. The authors propose ViCLIP-OT, a dual-encoder VLM that combines CLIP/SigLIP contrastive learning with a SIGROT loss to enhance global cross-modal consistency and mitigate modality gaps. Extensive experiments on three Vietnamese benchmarks (UIT-OpenViIC, KTVIC, Crossmodal-3600) show that ViCLIP-OT outperforms CLIP and SigLIP baselines in both in-domain and zero-shot settings: it achieves 67.34% average Recall@K on UIT-OpenViIC (5.75 percentage points improvement over CLIP) and 56.85% average Recall@K on zero-shot Crossmodal-3600 evaluation (11.72 percentage points improvement over CLIP). Embedding analysis confirms improved alignment and reduced modality gap, demonstrating that SIGROT is an effective strategy for low-resource cross-modal retrieval.
## 1.6. Original Source Link
- Official preprint page: https://arxiv.org/abs/2602.22678
- PDF link: https://arxiv.org/pdf/2602.22678v1
- Publication status: Preprint, not yet peer-reviewed.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Image-text retrieval is a core component of intelligent multimedia systems, but state-of-the-art vision-language models are almost exclusively trained on English data, making them suboptimal for low-resource languages like Vietnamese.
### Importance & Existing Gaps
For Vietnamese, there are two critical gaps:
1.  Lack of large-scale native Vietnamese image-text pre-training datasets, limiting direct application of CLIP-style training paradigms.
2.  Common workarounds (translating Vietnamese captions to English and using English VLMs) introduce translation noise and fail to preserve language-specific semantic meaning.
    Additionally, standard CLIP-style contrastive learning only enforces instance-level pairwise alignment, and does not explicitly model relational structure between samples in a batch, leading to large modality gaps between image and text embeddings that hurt retrieval performance.
### Innovative Entry Point
The authors propose to combine CLIP-style instance-level contrastive learning with a optimal transport (OT)-based regularization loss that leverages precomputed cross-modality similarity graphs to enforce global distribution-level alignment between image and text embeddings, while using native Vietnamese pre-trained encoders to avoid translation noise.
## 2.2. Main Contributions / Findings
The paper's three primary contributions are:
1.  **ViCLIP-OT, the first foundation VLM for Vietnamese image-text retrieval**: It uses a DINOv3-based image encoder and a Vietnamese Sentence-BERT text encoder, trained on native Vietnamese image-text data to avoid translation artifacts.
2.  **Novel SIGROT loss**: A Similarity-Graph Regularized Optimal Transport loss that explicitly models relational structure between samples in a training batch, improving global cross-modal alignment and reducing the modality gap.
3.  **State-of-the-art performance on three Vietnamese benchmarks**: ViCLIP-OT outperforms both vanilla CLIP/SigLIP baselines and zero-shot multilingual VLMs across in-domain and zero-shot settings, with strong transferability to unseen datasets.
    Key findings include:
- Integrating SIGROT with contrastive loss consistently improves retrieval performance across all tested model architectures.
- SIGROT reduces the modality gap between image and text embeddings by 30-50% compared to vanilla contrastive baselines.
- Cross-modality similarity graphs (combining intra-modal and inter-modal similarities) outperform single-modality or dual-modality only graphs for SIGROT regularization.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
Below are detailed beginner-friendly explanations of core concepts required to understand the paper:
### 3.1.1. Image-Text Retrieval
Image-text retrieval is a cross-modal task that involves two subtasks:
1.  **Text-to-image retrieval**: Given a text query, return the most relevant images from a database.
2.  **Image-to-text retrieval**: Given an image query, return the most relevant captions from a database.
    Models for this task typically use a dual-encoder architecture that maps images and text to a shared embedding space, where similarity between embeddings is measured using cosine similarity.
### 3.1.2. Contrastive Learning
Contrastive learning is a representation learning paradigm that trains models to pull semantically similar samples closer together and push dissimilar samples apart in a shared embedding space. For cross-modal tasks, matched image-text pairs are treated as positives, and all mismatched pairs in a training batch are treated as negatives.
### 3.1.3. CLIP
Contrastive Language-Image Pre-training (CLIP, Radford et al. 2021) is the standard dual-encoder baseline for cross-modal tasks. It is trained with a symmetric cross-entropy loss that uses softmax normalization over all in-batch samples to align image and text embeddings.
### 3.1.4. SigLIP
SigLIP (Zhai et al. 2023) is a variant of CLIP that replaces the softmax cross-entropy loss with a sigmoid loss that operates on individual image-text pairs independently, removing the need for global batch normalization and enabling more efficient training with larger batch sizes.
### 3.1.5. Optimal Transport (OT)
Optimal Transport is a mathematical framework that finds the minimal cost to move mass from one probability distribution to another. It is widely used for distribution alignment tasks in machine learning, including cross-modal representation learning. For large-scale applications, entropic regularization is added to the OT problem to enable fast computation via the Sinkhorn-Knopp algorithm.
### 3.1.6. Unbalanced Optimal Transport (UOT)
Standard OT enforces strict mass conservation (the total mass of the source and target distributions must be equal), which is sensitive to outliers (e.g., noisy captions, irrelevant background in images). Unbalanced OT relaxes this constraint using KL divergence penalties on the marginals, allowing partial matching and improving robustness to noise.
### 3.1.7. Modality Gap
The modality gap is a well-documented phenomenon in multimodal contrastive learning where image embeddings and text embeddings form separate, distinct clusters in the shared embedding space, even for semantically matched pairs. This gap reduces cross-modal retrieval performance by increasing the distance between matched image-text pairs.
## 3.2. Previous Works
### 3.2.1. Cross-Modal Contrastive Learning
The standard CLIP framework (Radford et al. 2021) uses the following symmetric cross-entropy loss for training:
$$
\mathcal{L}_{\text{CLIP}} = \frac{1}{2N} \sum_{i=1}^N \left( -\log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^N \exp(s_{ij}/\tau)} -\log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^N \exp(s_{ji}/\tau)} \right)
$$
Where $s_{ij} = \cos(z_i^{\text{image}}, z_j^{\text{text}})$ is the cosine similarity between the $i$-th image embedding and $j$-th text embedding, $\tau$ is a learnable temperature parameter, and $N$ is the batch size. This loss enforces that matched pairs have higher similarity than all mismatched pairs in the batch, but only models pairwise instance-level relationships.
SigLIP (Zhai et al. 2023) replaces this with a sigmoid loss:
$$
\mathcal{L}_{\text{SigLIP}} = \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N \left( y_{ij} \log \sigma(s_{ij}/\tau + b) + (1-y_{ij}) \log (1-\sigma(s_{ij}/\tau + b)) \right)
$$
Where $y_{ij}=1$ if the $i$-th image and $j$-th text are a matched pair, `0` otherwise, $\sigma$ is the sigmoid function, and $b$ is a learnable bias parameter.
### 3.2.2. Optimal Transport in Cross-Modal Learning
Prior work (Khamis et al. 2024; Montesuma et al. 2024) has shown that OT can be used to align distributions of embeddings across modalities, improving global consistency of the shared embedding space. The standard entropic regularized OT problem is solved via the Sinkhorn-Knopp algorithm, which has linear time complexity relative to batch size for practical use cases.
### 3.2.3. Vietnamese Vision-Language Research
Prior work on Vietnamese VL has focused primarily on image captioning, with datasets including UIT-OpenViIC (Bui et al. 2026) and KTVIC (Pham et al. 2024), but no foundation VLMs for Vietnamese image-text retrieval existed prior to this work.
## 3.3. Technological Evolution
The field of cross-modal retrieval has evolved in three key phases:
1.  Early cross-modal models (pre-2021): Used separate handcrafted features for images and text, with shallow alignment mechanisms, leading to poor generalization.
2.  CLIP era (2021-2023): Large-scale dual-encoder models trained on billions of English image-text pairs achieved strong zero-shot performance, but were limited to high-resource languages.
3.  Low-resource and enhanced alignment era (2023-present): Research has focused on adapting CLIP-style models to low-resource languages, and adding regularization mechanisms (including OT) to improve cross-modal alignment and reduce modality gaps.
    This paper's work falls into the third phase, as the first to combine contrastive learning with similarity-graph regularized OT for a low-resource Vietnamese VLM.
## 3.4. Differentiation Analysis
Compared to prior related work, this paper has three core innovations:
1.  **Native Vietnamese training**: Unlike approaches that use translation to adapt English VLMs to Vietnamese, ViCLIP-OT uses a native Vietnamese pre-trained text encoder and is trained on native Vietnamese image-text pairs, avoiding translation noise.
2.  **SIGROT loss**: No prior work has used a cross-modality similarity graph to regularize OT-based alignment for cross-modal contrastive learning. Existing OT-based cross-modal methods only use instance-level similarity to define transport costs, not relational structure between samples.
3.  **First Vietnamese retrieval foundation model**: Prior Vietnamese VL work is limited to captioning tasks, and no open foundation VLMs for Vietnamese image-text retrieval existed before this work.

# 4. Methodology
## 4.1. Principles
The core principle of ViCLIP-OT is to combine two complementary alignment mechanisms:
1.  **Instance-level alignment**: Standard CLIP/SigLIP contrastive loss that ensures matched image-text pairs are close in the shared embedding space, and mismatched pairs are far apart.
2.  **Distribution-level alignment**: The proposed SIGROT loss that uses optimal transport to align the global structure of image and text embedding distributions, while respecting relational similarity between samples in a batch (e.g., multiple captions describing similar visual concepts).
    This hybrid approach addresses the limitations of vanilla contrastive learning, which only models pairwise relationships and often produces large modality gaps between the two modalities.
## 4.2. Core Methodology In-depth
### 4.2.1. Architecture Overview
ViCLIP-OT follows a dual-encoder design with separate image and text towers that project inputs to a shared $d$-dimensional embedding space:
$$
\mathbf{z}_i^{\text{image}} = f_{\text{image}}(x_i) \in \mathbb{R}^d, \qquad \mathbf{z}_i^{\text{text}} = f_{\text{text}}(t_i) \in \mathbb{R}^d
$$
Where:
- $x_i$ is the $i$-th input image, $t_i$ is the $i$-th input caption
- $f_{\text{image}}$ is the image encoder, $f_{\text{text}}$ is the text encoder

  All embeddings are $\ell_2$-normalized before similarity calculation:
$$
\tilde{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}
$$
#### Image Tower
The image encoder uses a DINOv3 (Siméoni et al. 2025) ViT-B/16 backbone, pre-trained via self-supervised learning on large-scale image datasets. Patch-level features from the DINOv3 backbone are aggregated via mean pooling to get a global image representation, which is then projected to the shared $d$-dimensional space via a linear projection layer.
#### Text Tower
The text encoder uses a Vietnamese Sentence-BERT (SBERT, Reimers & Gurevych 2019) model pre-trained on large-scale Vietnamese text corpora. Token-level outputs from SBERT are aggregated via mean pooling over non-padding tokens to get a sentence embedding, which is projected to the shared $d$-dimensional space via an optional linear layer (if the SBERT hidden size does not match $d$).

The overall architecture is visualized in Figure 1 from the original paper:

![Figure 1: ViCLIP-OT architecture overview. The model consists of a DINOv3-based image encoder and a Vietnamese Sentence-BERT text encoder that project images and texts into a shared embedding space. The hybrid training objective combines a CLIPstyle contrastive loss with the proposed SIGROT loss, which uses a similarity graph and optimal transport to enforce global cross-modal alignment.](images/1.jpg)
*该图像是ViCLIP-OT模型架构的示意图。图中展示了如何通过DINOv3图像编码器和越南句子BERT文本编码器，将图像和文本投影到共享嵌入空间。该模型的混合训练目标结合了CLIP风格对比损失与SIGROT损失，通过相似性图和最优传输机制实现全球跨模态对齐。*

### 4.2.2. Similarity-Graph Regularized Optimal Transport (SIGROT) Loss
The SIGROT loss is designed to inject global relational structure into batch-wise cross-modal matching, complementing the instance-level contrastive loss.
#### Step 1: Batch Similarity Graph Construction
For each training batch, a cross-modality similarity graph is precomputed using a robust pre-trained multimodal embedding model (Qwen3-VL-Embedding-2B in this work). Let $E_{\text{text}}$ and $E_{\text{image}}$ be matrices of precomputed $\ell_2$-normalized caption and image embeddings for the batch. The following similarity matrices are computed:
$$
G_{\text{text}} = E_{\text{text}} E_{\text{text}}^\intercal, \qquad G_{\text{image}} = E_{\text{image}} E_{\text{image}}^\intercal
$$
$$
G_{\text{text-image}} = E_{\text{text}} E_{\text{image}}^\intercal, \qquad G_{\text{image-text}} = E_{\text{image}} E_{\text{text}}^\intercal
$$
Where:
- $G_{\text{text}}$ is the intra-modal text-text similarity matrix
- $G_{\text{image}}$ is the intra-modal image-image similarity matrix
- $G_{\text{text-image}}$ and $G_{\text{image-text}}$ are the inter-modal cross-similarity matrices

  These four matrices are averaged to get the final cross-modality similarity graph:
$$
G_{\text{cross}} = \frac{1}{4} \left( G_{\text{text}} + G_{\text{image}} + G_{\text{text-image}} + G_{\text{image-text}} \right)
$$
This graph captures both intra-modal and inter-modal relational structure between all samples in the batch.
#### Step 2: Optimal Transport Formulation with Graph Regularization
Given the model's output normalized embeddings for the batch $\tilde{Z}_{\text{image}}$ (image embeddings stacked) and $\tilde{Z}_{\text{text}}$ (text embeddings stacked), the cross-modal similarity matrix is computed as:
$$
S_{\text{image-text}} = \tilde{Z}_{\text{image}} \tilde{Z}_{\text{text}}^\intercal
$$
The transport cost matrix is defined as $C_{\text{image-text}} = \mathbf{1} - S_{\text{image-text}}$, where higher similarity between an image and text corresponds to lower transport cost.
For the image-to-text direction, the optimal transport plan is solved via unbalanced OT:
$$
\begin{array}{rl}
\gamma_{\mathrm{i2t}}^* = \underset{\gamma \in \mathbb{R}_+^{N \times N}}{\arg \operatorname{min}} & \langle \gamma, C_{\mathrm{image-text}} \rangle_F - \varepsilon H(\gamma) \\
& + \tau_{m1} \mathrm{KL} (\gamma \mathbf{1}_N \| \mu) + \tau_{m2} \mathrm{KL} (\gamma^\top \mathbf{1}_N \| \nu)
\end{array}
$$
Where:
- $N$ is the batch size
- $\langle \cdot, \cdot \rangle_F$ is the Frobenius inner product
- $H(\gamma) = -\sum_{i,j} \gamma_{ij} (\log \gamma_{ij} -1)$ is the entropy of the transport plan
- $\varepsilon > 0$ is the entropic regularization coefficient
- $\mu = \nu = \frac{1}{N}\mathbf{1}_N$ are uniform distributions over the batch
- $\tau_{m1}, \tau_{m2} > 0$ are relaxation coefficients for the marginal constraints
- `\mathrm{KL}(\mathbf{u}\|\mathbf{v}) = \sum_{i=1}^n u_i \log\left(\frac{u_i}{v_i}\right) - u_i + v_i` is the Kullback-Leibler divergence

  The image-to-text SIGROT loss measures the divergence between the normalized optimal transport plan and the similarity graph distribution:
$$
\mathcal{L}_{\mathrm{SIGROT}}^{\mathrm{i2t}} = \mathrm{KL} (N \gamma_{\mathrm{i2t}}^* \Vert \mathrm{softmax}(G_{\mathrm{cross}}))
$$
Where $N \gamma_{\mathrm{i2t}}^*$ is scaled to sum to 1 to form a valid probability distribution, and $\mathrm{softmax}(G_{\mathrm{cross}})$ normalizes the similarity graph to a probability distribution.
For the text-to-image direction, the optimal transport plan is solved as:
$$
\begin{array}{rl}
\gamma_{\mathrm{t2i}}^* = \underset{\gamma \in \mathbb{R}_+^{N \times N}}{\arg \operatorname{min}} & \langle \gamma, C_{\mathrm{image-text}}^\intercal \rangle_F - \varepsilon H(\gamma) \\
& + \tau_{m1} \mathrm{KL} (\gamma \mathbf{1}_N \| \nu) + \tau_{m2} \mathrm{KL} (\gamma^\intercal \mathbf{1}_N \| \mu)
\end{array}
$$
And the text-to-image SIGROT loss is:
$$
\mathcal{L}_{\mathrm{SIGROT}}^{\mathrm{t2i}} = \mathrm{KL} (N \gamma_{\mathrm{t2i}}^* || \mathrm{softmax}(G_{\mathrm{cross}}))
$$
The final SIGROT loss is the average of the two directions:
$$
\mathcal{L}_{\mathrm{SIGROT}} = \frac{1}{2} \left( \mathcal{L}_{\mathrm{SIGROT}}^{\mathrm{i2t}} + \mathcal{L}_{\mathrm{SIGROT}}^{\mathrm{t2i}} \right)
$$
### 4.2.3. Hybrid Training Objective
ViCLIP-OT is trained with a hybrid objective that combines the contrastive loss (either CLIP or SigLIP) with the SIGROT loss:
$$
\mathcal{L}_{\mathrm{CLIP-SIGROT}} = \lambda \mathcal{L}_{\mathrm{CLIP}} + \mathcal{L}_{\mathrm{SIGROT}}
$$
$$
\mathcal{L}_{\mathrm{SigLIP-SIGROT}} = \lambda \mathcal{L}_{\mathrm{SigLIP}} + \mathcal{L}_{\mathrm{SIGROT}}
$$
Where $\lambda \geq 0$ is a hyperparameter that balances the contribution of the two loss components.
To stabilize training, two learnable parameters are added:
1.  A temperature parameter `\tau = \exp(\tau')`, where $\tau'$ is a learnable scalar, ensuring the temperature remains positive throughout training.
2.  A bias parameter $b$, used only for the SigLIP loss.

# 5. Experimental Setup
## 5.1. Datasets
Three Vietnamese image-text datasets are used for training and evaluation:
### 5.1.1. UIT-OpenViIC
UIT-OpenViIC (Bui et al. 2026) is the primary training and in-domain evaluation dataset. It is a large-scale open-domain Vietnamese image captioning dataset collected via web search using Vietnamese keywords, with:
- 13,100 total images
- 61,241 human-annotated Vietnamese captions
- Splits: 9,088 train images, 2,011 validation images, 2,001 test images
  It contains diverse real-world scenes and Vietnamese-specific content (e.g., Ao dai traditional clothing, street food, Vietnamese landmarks).
### 5.1.2. KTVIC
KTVIC (Pham et al. 2024) is a Vietnamese image captioning benchmark focused on daily-life scenarios in Vietnam, with:
- 4,327 total images
- 21,635 human-annotated captions (5 per image)
  Near-duplicate detection against the UIT-OpenViIC training set found 71.9% of test images and 65.4% of training images were near-duplicates, which were removed to avoid train-test contamination. The deduplicated splits used for zero-shot evaluation have 1,305 training images and 157 test images.
### 5.1.3. Crossmodal-3600 (XM3600)
Crossmodal-3600 (Thapliyal et al. 2022) is a geographically diverse multilingual dataset with 3,600 images and captions in 36 languages, including Vietnamese. No near-duplicates with the UIT-OpenViIC training set were found, so it is used for zero-shot evaluation with all 3,600 images and 7,350 Vietnamese captions.
## 5.2. Evaluation Metrics
### 5.2.1. Recall@K (R@K)
- **Conceptual Definition**: Recall@K measures the proportion of queries for which the correct matching sample is present in the top-K highest-ranked retrieval results. It is the standard metric for cross-modal retrieval tasks, with higher values indicating better performance.
- **Mathematical Formula**:
  $$
  \text{Recall@K} = \frac{1}{Q} \sum_{q=1}^Q \mathbb{1}\left( \text{rank}(q, g_q) \leq K \right)
  $$
- **Symbol Explanation**:
  - $Q$ is the total number of queries
  - $\mathbb{1}(\cdot)$ is the indicator function, equal to 1 if the condition inside is true, 0 otherwise
  - $\text{rank}(q, g_q)$ is the rank of the ground-truth match $g_q$ for query $q$ in the sorted list of retrieval results
  - $K$ is the number of top results considered (typically 1, 5, 10 for image-text retrieval)
### 5.2.2. Alignment Score
- **Conceptual Definition**: The alignment score measures the average cosine similarity between matched image-text pairs, quantifying how well semantically matched pairs are aligned in the shared embedding space. Higher values indicate better pairwise alignment.
- **Mathematical Formula**:
  $$
  \mathrm{Alignment} = \frac{1}{N} \sum_{i=1}^N \mathrm{sim}(\mathbf{z}_i^{\mathrm{image}}, \mathbf{z}_i^{\mathrm{text}})
  $$
- **Symbol Explanation**:
  - $N$ is the number of image-text pairs
  - $\mathrm{sim}(\cdot, \cdot)$ is the cosine similarity function
  - $\mathbf{z}_i^{\mathrm{image}}$ and $\mathbf{z}_i^{\mathrm{text}}$ are the normalized embeddings of the $i$-th matched image-text pair
### 5.2.3. Modality Gap
- **Conceptual Definition**: The modality gap measures the Euclidean distance between the centroid of all image embeddings and the centroid of all text embeddings in the shared space. Lower values indicate smaller separation between the two modalities, which improves cross-modal retrieval performance.
- **Mathematical Formula**:
  $$
  \Delta_{\mathrm{gap}} = \left\| \frac{1}{N} \sum_{i=1}^N \mathbf{z}_i^{\mathrm{image}} - \frac{1}{N} \sum_{i=1}^N \mathbf{z}_i^{\mathrm{text}} \right\|_2
  $$
- **Symbol Explanation**:
  - $\|\cdot\|_2$ is the $\ell_2$-norm (Euclidean distance)
  - $\frac{1}{N}\sum_{i=1}^N \mathbf{z}_i^{\mathrm{image}}$ is the centroid of all image embeddings
  - $\frac{1}{N}\sum_{i=1}^N \mathbf{z}_i^{\mathrm{text}}$ is the centroid of all text embeddings
## 5.3. Baselines
The proposed models are compared against two groups of baselines:
1.  **Same-architecture baselines**:
    - Vanilla CLIP: Same dual-encoder architecture as ViCLIP-OT, trained only with the standard CLIP loss.
    - Vanilla SigLIP: Same dual-encoder architecture as ViSigLIP-OT, trained only with the standard SigLIP loss.
2.  **Zero-shot multilingual VLM baselines**:
    - mSigLIP-base: Multilingual SigLIP model trained on 50+ languages.
    - Jina CLIP v2: Multilingual CLIP model with 865M parameters.
    - Jina Embedding v4: Large multilingual multimodal embedding model with 4B parameters.
    - Qwen3-VL-Embedding-2B: State-of-the-art multilingual multimodal embedding model with 2B parameters.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### 6.1.1. In-Domain Evaluation on UIT-OpenViIC
The following are the results from Table 1 of the original paper:

| Method/Model | # Params | Text → Image | | | Image → Text | | | Avg. |
|---|---|---|---|---|---|---|---|---|
| | | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 | |
| mSigLIP-base* [46] | 370M | 14.34 | 28.94 | 36.21 | 20.49 | 32.23 | 37.43 | 28.27 |
| Jina CLIP v2* ] | 865M | 30.01 | 52.09 | 61.70 | 40.23 | 65.02 | 74.41 | 53.91 |
| Jina Embedding v4* [14] | 4B | 23.97 | 42.22 | 50.29 | 41.48 | 66.77 | 75.61 | 50.06 |
| Qwen3-VL-Embedding-2B* [21] | 2B | 32.13 | 54.00 | 62.93 | 39.83 | 66.52 | 77.01 | 55.40 |
| CLIP | 221M | 31.19 | 59.80 | 71.23 | 46.60 | 75.53 | 85.19 | 61.59 |
| SigLIP | 221M | 34.75 | 63.01 | 72.96 | 50.10 | 79.78 | 88.04 | 64.77 |
| CLIP + UOT | 221M | 29.27 | 57.62 | 69.07 | 43.59 | 75.03 | 84.03 | 59.77 |
| SigLIP + UOT | 221M | 37.84 | 65.30 | 74.98 | 53.95 | 80.95 | 88.81 | 66.97 |
| SIGROT | 221M | 40.75 | 70.72 | 80.90 | 37.99 | 61.11 | 71.68 | 60.53 |
| ViCLIP-OT (Eq. 19) | 221M | 37.57 | 65.65 | 75.43 | 54.35 | 81.83 | 89.19 | 67.34 |
| ViSigLIP-OT (Eq. 20) | 221M | 39.19 | 66.71 | 76.04 | 57.21 | 83.83 | 90.79 | 68.96 |

Key observations:
1.  ViCLIP-OT outperforms the vanilla CLIP baseline by 5.75 percentage points in average R@K, and ViSigLIP-OT outperforms the vanilla SigLIP baseline by 4.19 percentage points, demonstrating the effectiveness of adding the SIGROT loss.
2.  Both ViCLIP-OT and ViSigLIP-OT outperform all zero-shot multilingual VLM baselines by large margins (11.94 pp and 13.56 pp improvement over the best multilingual baseline Qwen3-VL-Embedding-2B), showing the advantage of training on native Vietnamese data.
3.  The hybrid CLIP+SIGROT and SigLIP+SIGROT objectives outperform using either loss alone, confirming that the two loss components are complementary.

    The R@K performance comparison across models is visualized in Figure 2 from the original paper:

    ![Figure 2: R@K comparison on UIT-OpenViIC for text-to-image (left) and image-to-text (right) retrieval tasks. Incorporating the SIGROT loss consistently improves performance over both CLIP and SigLIP baselines across all R@K metrics.](images/2.jpg)
    *该图像是图表，展示了在UIT-OpenViIC上进行文本至图像（左侧）和图像至文本（右侧）检索任务的召回率对比。图表中显示，结合SIGROT损失后，ViCLIP-OT在各个召回率指标上均优于CLIP和SigLIP基线模型。*

### 6.1.2. Zero-Shot Evaluation
The following are the zero-shot results from Table 2 of the original paper, formatted as HTML to handle the grouped dataset headers:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">Text → Image</th>
<th colspan="3">Image → Text</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8"><strong>KTVIC-train</strong></td>
</tr>
<tr>
<td>CLIP</td>
<td>21.12</td>
<td>46.99</td>
<td>59.22</td>
<td>31.65</td>
<td>59.46</td>
<td>72.49</td>
<td>48.49</td>
</tr>
<tr>
<td>SigLIP</td>
<td>23.16</td>
<td>48.78</td>
<td>60.57</td>
<td>35.48</td>
<td>62.22</td>
<td>73.64</td>
<td>50.64</td>
</tr>
<tr>
<td>ViCLIP-OT</td>
<td>26.24</td>
<td>52.46</td>
<td>64.14</td>
<td>38.47</td>
<td>64.37</td>
<td>75.48</td>
<td>53.52</td>
</tr>
<tr>
<td>ViSigLIP-OT</td>
<td>26.28</td>
<td>52.58</td>
<td>63.49</td>
<td>39.62</td>
<td>66.44</td>
<td>77.78</td>
<td>54.37</td>
</tr>
<tr>
<td colspan="8"><strong>KTVIC-test</strong></td>
</tr>
<tr>
<td>CLIP</td>
<td>50.32</td>
<td>82.80</td>
<td>89.94</td>
<td>63.06</td>
<td>92.36</td>
<td>97.45</td>
<td>79.32</td>
</tr>
<tr>
<td>SigLIP</td>
<td>52.61</td>
<td>83.31</td>
<td>89.94</td>
<td>71.97</td>
<td>94.27</td>
<td>96.18</td>
<td>81.38</td>
</tr>
<tr>
<td>ViCLIP-OT</td>
<td>56.69</td>
<td>85.61</td>
<td>91.97</td>
<td>70.06</td>
<td>93.63</td>
<td>98.09</td>
<td>82.68</td>
</tr>
<tr>
<td>ViSigLIP-OT</td>
<td>56.56</td>
<td>85.99</td>
<td>91.72</td>
<td>71.34</td>
<td>93.63</td>
<td>97.45</td>
<td>82.78</td>
</tr>
<tr>
<td colspan="8"><strong>Crossmodal-3600</strong></td>
</tr>
<tr>
<td>CLIP</td>
<td>22.52</td>
<td>45.55</td>
<td>58.01</td>
<td>26.22</td>
<td>53.42</td>
<td>65.06</td>
<td>45.13</td>
</tr>
<tr>
<td>SigLIP</td>
<td>26.67</td>
<td>50.31</td>
<td>61.78</td>
<td>31.17</td>
<td>57.78</td>
<td>69.83</td>
<td>49.59</td>
</tr>
<tr>
<td>ViCLIP-OT</td>
<td>28.90</td>
<td>55.29</td>
<td>66.37</td>
<td>42.56</td>
<td>68.81</td>
<td>79.17</td>
<td>56.85</td>
</tr>
<tr>
<td>ViSigLIP-OT</td>
<td>32.04</td>
<td>57.90</td>
<td>68.95</td>
<td>37.97</td>
<td>64.64</td>
<td>75.53</td>
<td>56.17</td>
</tr>
</tbody>
</table>

Key observations:
1.  ViCLIP-OT outperforms the CLIP baseline by 3.36 pp on KTVIC-test and 11.72 pp on Crossmodal-3600 in zero-shot settings, demonstrating that SIGROT improves the generalization ability of the model to unseen domains and datasets.
2.  Both ViCLIP-OT and ViSigLIP-OT consistently outperform their respective baselines across all zero-shot datasets, confirming that the SIGROT loss does not cause overfitting to the training domain.
### 6.1.3. Embedding Space Analysis
The UMAP visualization of embeddings from different models is shown in Figure 3 from the original paper:

![Figure 3: UMAP visualization of image and text embeddings on the UIT-OpenViIC test set. Each subplot corresponds to a different training objective. Circles represent image embeddings and triangles represent text embeddings, with colors indicating pseudo labels obtained via K-Means clustering ( $k = 2 0$ ). SIGROT-based methods exhibit tighter crossmodal clustering compared to baselines.](images/3.jpg)
*该图像是UMAP可视化图，展示了UIT-OpenViIC测试集上的图像和文本嵌入。每个子图对应不同的训练目标，圆圈代表图像嵌入，三角形代表文本嵌入，颜色表示通过K-Means聚类获得的伪标签（$k=20$）。与基线方法相比，SIGROT方法展现了更紧密的跨模态聚类。*

The quantitative alignment and modality gap results are presented in Table 3 from the original paper:

| Method | UIT-OpenViIC | | KTVIC-test | | Crossmodal-3600 | |
|---|---|---|---|---|---|---|
| | A↑ | ∥Δgap∥ ↓ | A ↑ | ∥Δgap∥ ↓ | A↑ | ∥Δgap∥ ↓ |
| SIGROT | 0.8061 | 0.1323 | 0.7670 | 0.2135 | 0.6976 | 0.1625 |
| CLIP | 0.5201 | 0.1952 | 0.4696 | 0.2032 | 0.5329 | 0.2558 |
| ViCLIP-OT | 0.6624 | 0.1026 | 0.6212 | 0.1636 | 0.6225 | 0.1273 |
| SigLIP | 0.3637 | 0.5843 | 0.3182 | 0.5757 | 0.3790 | 0.5789 |
| ViSigLIP-OT | 0.3928 | 0.3177 | 0.3373 | 0.3385 | 0.4142 | 0.3442 |

Key observations:
1.  Adding SIGROT improves the alignment score and reduces the modality gap for both CLIP and SigLIP models across all datasets. For example, SigLIP's modality gap on UIT-OpenViIC drops from 0.5843 to 0.3177 with the addition of SIGROT, a 45.6% reduction.
2.  ViCLIP-OT achieves the lowest modality gap across all datasets, confirming that the hybrid objective produces a more coherent shared embedding space.
### 6.1.4. Visual Interpretability
GradCAM visualizations comparing SigLIP and ViSigLIP-OT are shown in Figure 4 from the original paper:

![Figure 4: GradCAM visualization comparing the baseline SigLIP and the proposed ViSigLIP-OT on the UIT-OpenViIC test set. Each row shows the original image alongside the GradCAM heatmaps from both models for a given Vietnamese text query. In the first two rows, ViSigLIP-OT focuses more precisely on the query-relevant objects (the girl wearing an Ao dai and the man holding apples in his hands), while SigLIP spreads activations over background regions. In the third row, SigLIP correctly attends to the man standing next to a car, whereas ViSigLIP-OT highlights irrelevant background areas.](images/4.jpg)
*该图像是GradCAM可视化结果，比较了基线模型SigLIP和提出的ViSigLIP-OT在UIT-OpenViIC测试集上的表现。每行展示了原始图像和针对特定越南文本查询的热力图，其中ViSigLIP-OT在与查询相关的对象上表现更突出，而SigLIP则更多地关注背景区域。*

Key observations:
1.  In most cases, ViSigLIP-OT attends more precisely to query-relevant objects in images (e.g., the girl wearing Ao dai, the man holding apples), while SigLIP spreads attention over irrelevant background regions.
2.  This indicates that the SIGROT loss encourages the model to focus on semantically relevant visual content that matches the text query, rather than contextual background.
## 6.2. Ablation Studies / Parameter Analysis
### 6.2.1. Partial Fine-Tuning of the Image Encoder
The effect of unfreezing different numbers of the last transformer groups in the DINOv3 image encoder is shown in Figure 5a from the original paper:

![Figure 5: Effect of (a) the number of last unfrozen groups in the image encoder using ViCLIP-OT, and (b) the hybrid loss weight $\\lambda$ where $\\lambda = 0$ corresponds to SIGROT only. Peak performance occurs at 13 unfrozen groups $( 6 9 . 6 2 \\% )$ and $\\lambda = 0 . 2$ for ViCLIP-OT, $\\lambda = 0 . 1$ for ViSigLIP-OT.](images/5.jpg)
*该图像是图表，展示了使用 ViCLIP-OT 模型的 (a) 最后未冻结组数对平均 Recall@K 的影响，以及 (b) 混合损失权重 `heta` 对模型性能的影响。其中，最高性能出现在 13 个未冻结组时为 69.62%。*

Key findings:
- Unfreezing the last 13 groups of the image encoder yields the best average R@K of 69.62%.
- Unfreezing too many layers (all 14 groups) causes a small performance drop, likely due to overfitting or destabilization of pre-trained visual features.
- Even unfreezing just the last 2 groups improves performance by nearly 7 percentage points over a fully frozen image encoder, showing that adapting higher-level visual features to Vietnamese data is highly beneficial.
### 6.2.2. Hybrid Loss Weight $\lambda$
The effect of varying the loss balancing weight $\lambda$ is shown in Figure 5b from the original paper. Key findings:
- For ViCLIP-OT, the optimal $\lambda$ is 0.2, achieving 69.20% average R@K.
- For ViSigLIP-OT, the optimal $\lambda$ is 0.1, achieving 70.76% average R@K.
- Using $\lambda=0$ (only SIGROT loss) leads to significantly lower performance, as does using too high a value of $\lambda$ that overweights the contrastive loss and overshadows the SIGROT regularization.
### 6.2.3. Similarity Graph Combination Strategy
The performance of different similarity graph combination strategies is shown in Figure 6 from the original paper:

![Figure 6: Average Recall@K for different similarity graph combination strategies. The cross-modality approach achieves the highest performance for both loss configurations.](images/6.jpg)
*该图像是一个图表，展示了不同相似性图组合策略下的平均 Recall@K 结果。ViCLIP-OT 在所有组合策略中，特别是在跨模态策略上表现最佳。*

Key findings:
- The cross-modality strategy (averaging all four intra-modal and inter-modal similarity matrices) achieves the highest performance for both ViCLIP-OT (69.04% avg R@K) and ViSigLIP-OT (70.76% avg R@K).
- Single-modality strategies (using only text-text or only image-image similarities) perform the worst, as they ignore inter-modal relational information.
- Dual-modality strategies (averaging only intra-modal similarities) perform better than single-modality but worse than cross-modality, confirming the value of including inter-modal similarity information in the graph.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces ViCLIP-OT, the first foundation vision-language model for Vietnamese image-text retrieval. By combining CLIP/SigLIP contrastive learning with the novel SIGROT loss that uses cross-modality similarity graphs and optimal transport to enforce global distribution-level alignment, the model achieves state-of-the-art performance on three Vietnamese benchmarks, outperforming both vanilla contrastive baselines and large multilingual VLMs in both in-domain and zero-shot settings. The results demonstrate that optimal transport-based structural regularization is an effective strategy for improving cross-modal retrieval performance in low-resource language settings, and the open-sourced model and code provide a strong foundation for future Vietnamese multimodal research.
## 7.2. Limitations & Future Work
The authors identify the following limitations and future research directions:
1.  **Limited pre-training data**: ViCLIP-OT is trained only on the relatively small UIT-OpenViIC dataset (9k training images). Future work will explore large-scale pre-training on larger Vietnamese image-text datasets to further improve performance.
2.  **Precomputed similarity graph**: The similarity graph used in SIGROT is precomputed using a fixed pre-trained model, which may not adapt to the model's evolving representations during training. Future work will explore end-to-end learning of the similarity graph alongside the main model.
3.  **Task extension**: The current model is designed for image-text retrieval. Future work will extend the framework to other Vietnamese multimodal tasks, including visual question answering and multimodal reasoning.
## 7.3. Personal Insights & Critique
### Key Insights
1.  **Generalizability to other low-resource languages**: The SIGROT-based hybrid training approach is not specific to Vietnamese, and can be applied to other low-resource languages with limited image-text data to improve cross-modal alignment and reduce modality gaps, without requiring large-scale pre-training datasets.
2.  **Importance of fair evaluation**: The rigorous near-duplicate removal step for zero-shot evaluation datasets is critical for obtaining reliable performance estimates, as many low-resource datasets have significant overlap with existing web-crawled data.
3.  **Complementarity of instance and distribution alignment**: The results clearly show that combining pairwise contrastive learning with distribution-level OT regularization yields better performance than either approach alone, providing a general recipe for improving cross-modal model performance.
### Potential Improvements
1.  **Dynamic similarity graphs**: The current precomputed similarity graph is fixed throughout training. Using a dynamic graph updated periodically with the model's own embeddings could further improve alignment, as the graph would adapt to the representations the model is learning.
2.  **Larger batch sizes**: The current experiments use a batch size of 128 (with gradient accumulation). Larger batch sizes would improve the quality of the in-batch negatives for contrastive learning and provide more samples for the OT calculation, potentially further improving performance.
3.  **Vietnamese-specific embedding model for graph construction**: The current similarity graph is computed using a multilingual embedding model. Using a Vietnamese-specific multimodal embedding model (once available) could improve the quality of the similarity graph and thus the effectiveness of the SIGROT loss.
4.  **Robustness to noisy captions**: The unbalanced OT formulation already provides some robustness to noise, but future work could explore adding explicit noise handling mechanisms to further improve performance on low-quality web-crawled datasets, which are common for low-resource languages.