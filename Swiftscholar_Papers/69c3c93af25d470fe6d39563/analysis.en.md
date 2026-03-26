# 1. Bibliographic Information
## 1.1. Title
The full title is *MLLM-HWSI: A Multimodal Large Language Model for Hierarchical Whole Slide Image Understanding*. Its central topic is the development of a novel multimodal large language model (MLLM) purpose-built for analyzing gigapixel histopathology whole slide images (WSIs) by explicitly leveraging their inherent biological hierarchical structure across four scales: cell, patch, region, and full WSI.
## 1.2. Authors
The authors and their affiliations are:
- Basit Alawode, Muaz Khalifa Al-Radi, Shahad Albastaki, Asim Khan, Moshira Ali Abdalla, Sajid Javed: Department of Computer Science, Khalifa University of Science and Technology, UAE
- Arif Mahmood: Information Technology University, Pakistan
- Muhammad Bilal: King Abdulaziz University (KAU), Saudi Arabia
- Mohammed Bennamoun: University of Western Australia, Australia
  The research team has established expertise in computational pathology (CPath), vision-language modeling, and self-supervised learning for medical imaging, with multiple prior publications in top venues (CVPR, Nature Medicine) in the CPath domain.
## 1.3. Journal/Conference
As of the current date, the paper is released as a preprint on arXiv, the leading open-access preprint server for computer science and medical AI research. It has not yet undergone formal peer review or been accepted for publication at a conference or journal.
## 1.4. Publication Year
2026 (preprint released 24 March 2026 UTC)
## 1.5. Abstract
### Research Objective
Address the critical limitation of existing CPath MLLMs, which compress entire WSIs into a single global embedding, losing fine-grained hierarchical diagnostic information and failing to replicate pathologists' multi-scale reasoning workflow.
### Core Methodology
The proposed MLLM-HWSI aligns visual features with pathology language at four semantically analogous scales (cell = word, patch = phrase, region = sentence, WSI = paragraph). It uses scale-specific vision-language (VL) projectors, a joint training objective of hierarchical contrastive loss and cross-scale consistency loss, a lightweight *Cell-Cell Attention Fusion (CCAF)* transformer to aggregate cell embeddings per patch, and a *Semantic Patch Filtering (SPF)* module to select diagnostically relevant patches. Multi-scale visual tokens are fused with text tokens and fed to an instruction-tuned LLM for open-ended reasoning tasks.
### Main Results
MLLM-HWSI achieves new state-of-the-art (SOTA) results on 13 WSI-level benchmarks across 6 core CPath tasks: zero-shot classification, retrieval, visual question answering (VQA), report generation, caption generation, and cross-modal retrieval, outperforming 24 prior SOTA models.
### Key Conclusions
Aligning multi-scale WSI visual features with pathology language produces accurate, interpretable outputs that mirror clinical diagnostic workflows, significantly advancing holistic WSI understanding.
## 1.6. Original Source Link
- Official preprint source: https://arxiv.org/abs/2603.23067v1
- PDF link: https://arxiv.org/pdf/2603.23067v1
- Publication status: Preprint (not yet peer-reviewed or formally published)

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Existing WSI-level CPath MLLMs (e.g., SlideChat, WSI-LLaVA) aggregate all patch-level embeddings from a gigapixel WSI into a single global vector aligned to full pathology reports. This compression loses fine-grained spatial and hierarchical semantic information, prevents grounding of textual diagnostic claims (e.g., "pleomorphic nuclei") to their corresponding visual evidence, and does not replicate how human pathologists reason across scales.
### Problem Importance
WSIs are the clinical gold standard for cancer diagnosis and prognosis. AI-powered CPath tools can reduce pathologist workload, improve diagnostic reproducibility, and enable earlier cancer detection. However, the lack of interpretability and evidence grounding in existing MLLMs prevents their clinical adoption, as clinicians cannot verify the visual basis for a model's diagnostic claim.
### Research Gap
Prior work ignores the inherent hierarchical structure of tissue (cellular morphology → regional tissue organization → global WSI context), which forms the foundation of pathological reasoning. No existing WSI MLLM explicitly aligns each level of this hierarchy to corresponding semantic units in pathology reports.
### Innovative Entry Point
The paper models the WSI as a structured natural language narrative, where each level of biological structure maps directly to a level of linguistic structure (cell = word, patch = phrase, region = sentence, WSI = paragraph). This design aligns the model's architecture with both biological reality and the clinical diagnostic workflow of pathologists.
## 2.2. Main Contributions / Findings
### Primary Contributions
1. A novel multi-scale hierarchical MLLM that performs explicit alignment of cell, patch, region, and WSI-level visual features with pathology report semantics, enabling unified multi-scale reasoning over WSIs.
2. A joint training objective combining hierarchical contrastive alignment loss and cross-scale consistency loss to preserve semantic coherence across scales, eliminating semantic drift between hierarchical levels and enabling evidence-based reasoning.
3. Two novel efficiency modules: *Semantic Patch Filtering (SPF)* to reduce computational cost by selecting only diagnostically relevant patches, and *Cell-Cell Attention Fusion (CCAF)* to aggregate cell-level embeddings into compact per-patch tokens.
4. Extensive empirical validation showing the model outperforms 24 SOTA CPath models across 6 tasks and 13 benchmarks, with consistent double-digit performance gains in many cases.
### Key Findings
- Explicitly modeling the hierarchical structure of WSIs and aligning each level to text semantics significantly improves performance across all CPath tasks, compared to global-only MLLMs.
- Cross-scale consistency loss is critical to ensure semantic coherence between hierarchical levels, preventing drift between cell/patch/region/WSI representations.
- The model's outputs are inherently interpretable, as diagnostic claims are grounded in multi-scale visual evidence matching pathologists' reasoning processes.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All technical terms are defined for novice readers:
- **Whole Slide Image (WSI):** A gigapixel (typically >100,000 × 100,000 pixels) digital scan of a histopathology tissue slide, usually stained with Hematoxylin and Eosin (H&E) to visualize cellular and tissue structure. It is the clinical gold standard for cancer diagnosis.
- **Computational Pathology (CPath):** A subfield of medical AI that uses computer vision and machine learning to analyze histopathology images for diagnostic, prognostic, and research applications.
- **Multimodal Large Language Model (MLLM):** A large language model (LLM, a deep learning model trained on large text corpora to understand and generate natural language) extended to process non-text modalities (e.g., images) by aligning visual and textual representations into a shared semantic space.
- **Contrastive Learning:** A learning technique that trains models to map semantically similar inputs (e.g., an image of abnormal nuclei and the text "pleomorphic nuclei") to nearby points in embedding space, and dissimilar inputs to distant points.
- **Vision-Language (VL) Alignment:** The process of training a model to understand the correspondence between visual content and natural language descriptions, enabling tasks like VQA, image captioning, and report generation.
- **Instruction Tuning:** A fine-tuning process for LLMs/MLLMs where the model is trained on a large set of task-specific instructions and corresponding outputs, to improve its ability to follow user commands and perform diverse downstream tasks.
- **Gigapixel Image Processing Challenge:** WSIs are too large to be processed end-to-end by standard computer vision models (which typically accept <1000×1000 pixel inputs), so they require hierarchical decomposition into smaller units (patches, regions) for efficient processing.
## 3.2. Previous Works
The paper categorizes prior work into three core domains:
### 1. MLLMs in CPath
Prior WSI-level MLLMs including SlideChat, WSI-LLaVA, and TITAN aggregate patch embeddings into a single global WSI embedding aligned to full pathology reports. These models enable open-ended WSI reasoning, but lose fine-grained hierarchical information and cannot ground text to specific visual regions.
### 2. VLMs in CPath
Patch-level vision-language models (VLMs) including CONCH, PLIP, QuiltNet, and CPLIP align histopathology patches with text descriptions, producing strong patch-level representations. However, they operate only at the patch level, and require aggregation for WSI-level tasks, which loses spatial and hierarchical context.
The standard contrastive loss used in all prior VLMs is:
\$
\mathcal{L}_{\text{contrastive}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j)/\tau)}
\$
Where:
- $v_i$ = visual embedding of sample $i$
- $t_i$ = text embedding of the corresponding description for sample $i$
- $\text{sim}(\cdot, \cdot)$ = cosine similarity
- $\tau$ = temperature parameter controlling the sharpness of the similarity distribution
- $N$ = batch size
  This loss pushes matching visual-text pairs close to each other, and non-matching pairs far apart. It forms the base of the paper's hierarchical contrastive loss.
### 3. Visual Foundation Models (VFMs) in CPath
Self-supervised visual models including UNI, CTransPath, Virchow, and GigaPath are trained on large unlabeled pathology image datasets, and act as strong feature extractors for patch/WSI-level tasks. However, they are vision-only, and cannot be directly used for text-interactive tasks like VQA without additional VL alignment.
## 3.3. Technological Evolution
The evolution of CPath models follows a clear trajectory:
1. 2010s: Supervised patch-level classification models for narrow, task-specific applications (e.g., tumor detection).
2. Early 2020s: Self-supervised visual foundation models enabling transfer learning across diverse CPath tasks.
3. 2023-2024: Patch-level VLMs enabling zero-shot patch classification and text alignment.
4. 2024-2025: Global-only WSI-level MLLMs enabling WSI-level VQA and report generation.
5. 2026 (this work): Hierarchical WSI-level MLLMs aligning multiple scales to text, addressing the limitations of global-only models. This work is the first to explicitly model the four-level hierarchy of WSIs and align each level to corresponding text semantics.
## 3.4. Differentiation Analysis
The core innovations of MLLM-HWSI compared to prior work are:
1. **Multi-scale alignment:** Unlike all prior WSI MLLMs that use only a single global WSI embedding, MLLM-HWSI extracts and aligns embeddings at four distinct hierarchical scales (cell, patch, region, WSI).
2. **Novel training objective:** Unlike prior works that use only a single WSI-level contrastive loss, MLLM-HWSI uses a joint objective of hierarchical contrastive loss (aligning each scale to corresponding text) and cross-scale consistency loss (ensuring semantic coherence across scales).
3. **Efficiency modules:** The novel SPF and CCAF modules reduce computational cost while preserving diagnostically relevant information.

# 4. Methodology
## 4.1. Principles
### Core Idea
Model the WSI as a hierarchical natural language narrative, where each level of biological structure (cell → patch → region → WSI) corresponds to a level of linguistic structure (word → phrase → sentence → paragraph). Align each of these four visual levels to corresponding text semantics, enforce cross-scale consistency to avoid semantic drift, and fuse the aligned multi-scale visual tokens with text tokens to enable LLM reasoning that mirrors pathologists' multi-scale diagnostic workflow.
### Theoretical Intuition
Pathologists' diagnostic reasoning inherently operates across scales: they start with low-magnification global context to identify abnormal regions, zoom into regional tissue structure to assess architectural patterns, then inspect cellular morphology to confirm diagnostic features, and integrate information across all scales to reach a diagnosis. The model's design directly replicates this clinical workflow.
## 4.2. Core Methodology In-depth
The methodology is broken into 5 sequential steps, with integrated formula explanations:
---
### Step 1: Hierarchical Decomposition of WSIs
WSIs are too large to process directly, so they are decomposed into a four-level hierarchy:
1. A WSI $I$ at 20× magnification is split into non-overlapping regions: $I = \{R_i\}_{i=1}^{n_r}$, where each region $R_i \in \mathbb{R}^{4096 \times 4096 \times 3}$ (4096×4096 pixels, 3 color channels for H&E staining). This size is chosen to capture mesoscopic (medium-scale) tissue organization patterns like gland formation.
2. Each region $R_i$ is further split into non-overlapping patches: $R_i = \{P_{ij}\}_{j=1}^{n_p}$, where each patch $P_{ij} \in \mathbb{R}^{256 \times 256 \times 3}$, the standard input size for patch-level pathology feature extractors.
   The authors extracted 0.356M regions and 91.33M patches from 9,642 WSIs for training.
---
### Step 2: Hierarchical Multi-Scale Encoder
The encoder extracts features at four hierarchical levels:
#### 2.1 Patch-Level Encoder & Semantic Patch Filtering (SPF)
a. Patch feature extraction: Patch embeddings are extracted using the CONCH encoder (SOTA pathology VLM):
\$
f_{ij} = \mathcal{F}_{\text{CONCH}}(P_{ij})
\$
Where $f_{ij} \in \mathbb{R}^{d_p}$ is the embedding of patch $P_{ij}$, $d_p$ = embedding dimension of CONCH.
b. Redundant patch removal: To reduce computational cost, homogeneous redundant patches are removed using pairwise cosine similarity:
First, normalize patch embeddings: $\hat{f}_{ij} = \frac{f_{ij}}{\|f_{ij}\|_2}$, where $\| \cdot \|_2$ is the L2 norm.
Compute pairwise cosine similarity between patch $j$ and $k$ in region $i$: $s_i^{j,k} = \hat{f}_{ij} \cdot \hat{f}_{ik}$ (dot product of normalized embeddings equals cosine similarity).
Compute mean similarity $\mu_i = \frac{1}{n_p^2} \sum_j \sum_k s_i^{j,k}$ and variance $\sigma_i^2 = \frac{1}{n_p^2} \sum_j \sum_k (s_i^{j,k} - \mu_i)^2$ of all pairwise similarities in region $i$.
Set redundancy threshold: $\tau_i = \mu_i + \sigma_i$. A patch $P_{ij}$ is redundant if its mean similarity to all other patches in the region $\mu_i^j = \frac{1}{n_p} \sum_{k=1}^{n_p} s_i^{j,k} > \tau_i$, so it is removed. The remaining patches form subset $R_i' = \{P_{ij}\}_{j=1}^{h_i}$, $h_i < n_p$.
c. Diagnostically relevant patch selection: Patches relevant to the corresponding pathology report are selected:
Tokenize the pathology report $D$ into $M$ semantic entities (diagnostic keywords): $D = \{w_1, w_2, ..., w_M\}$.
Encode each text token with CONCH's text encoder: $\mathbf{t}_m = \mathcal{T}_{\text{CONCH}}(w_m)$, then normalize: $\hat{\mathbf{t}}_m = \frac{\mathbf{t}_m}{\|\mathbf{t}_m\|_2}$.
Compute relevance score of patch $P_{ij}$ to the report: $r_{ij} = \sum_{m=1}^M \hat{f}_{ij}^\top \hat{\mathbf{t}}_m$ (sum of cosine similarities between the patch embedding and all report token embeddings).
Select the top-k patches with the highest $r_{ij}$ scores to form the final filtered patch subset $\hat{R}_i$ for the region.
---
#### 2.2 Cell-Level Encoder & Cell-Cell Attention Fusion (CCAF)
a. Cell feature extraction: For each filtered patch $P_{ij} \in \hat{R}_i$, use CellViT (SOTA cell segmentation and feature extraction model for pathology) to segment individual cells and extract their embeddings: $\{c_{ijk}\}_{k=1}^{n_{ij}}$, where $c_{ijk} \in \mathbb{R}^{d_c}$ is the embedding of cell $k$ in patch $P_{ij}$, $n_{ij}$ = number of cells in the patch.
b. Cell embedding aggregation: CCAF aggregates cell embeddings into a single compact token per patch to avoid excessive computational cost:
Append a <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> (class) token (standard in Vision Transformers) to the sequence of cell embeddings: $[\text{<[BOS\_never\_used\_51bce0c785ca2f68081bfa7d91973934]>}]_{ij}, \{c_{ijk}\}_{k=1}^{n_{ij}}$.
Pass this sequence through a lightweight ViT (2 transformer blocks, 2 self-attention heads) to compute the aggregated cell token:
\$
c_{ij} = \text{ViT}_{\text{cell-cell}}([\text{<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>}]_{ij}, \{c_{ijk}\}_{k=1}^{n_{ij}})
\$
Where $c_{ij} \in \mathbb{R}^{784}$. This token captures cell-cell interactions, nuclear diversity, and intra-patch morphological context.
---
#### 2.3 Region-Level Encoder
Use the HIPT (Hierarchical Vision Transformer for Pathology) encoder to aggregate patch embeddings into region embeddings:
1. Each patch $P_{ij}$ is processed by a patch-level ViT $\text{ViT}_p$: `p_{ij} = \text{ViT}_p(\{P_{ij}\}_{j=1}^{256})`.
2. The set of patch embeddings in a region are processed by a region-level ViT $\text{ViT}_r$: `r_i = \text{ViT}_r(\{p_{ij}\}_{j=1}^{256})`.
   This embedding captures micro-architectural patterns like tissue polarity, glandular organization, and stromal invasion.
---
#### 2.4 WSI-Level Encoder
Aggregate all region embeddings $\{r_i\}_{i=1}^{n_r}$ using a WSI-level ViT $\text{ViT}_{\text{WSI}}$ (modified HIPT architecture pretrained for global tissue representation learning):
\$
f_{\text{WSI}} = \text{ViT}_{\text{WSI}}(\{r_i\}_{i=1}^{n_r})
\$
This embedding captures global patterns like tumor distribution, necrosis, and invasion margins.
---
The final hierarchical representation of a WSI is: $\{c_{ij}\}, \{f_{ij}\}, \{r_i\}, f_{\text{WSI}}$, covering all four scales.
---
### Step 3: Hierarchical Alignment V→L Projectors
Four separate trainable 2-layer MLP projectors map each scale's visual embeddings into the latent space of the LLM, to align visual and textual feature dimensions:
- Cell-level projector: `z_c = A_c(c_{ij})`
- Patch-level projector: `z_p = A_p(f_{ij})`
- Region-level projector: `z_r = A_r(r_i)`
- WSI-level projector: $z_{\text{WSI}} = A_{\text{WSI}}(f_{\text{WSI}})$
  Where $A_c, A_p, A_r, A_{\text{WSI}}$ are trainable projection layers with output dimension matching the LLM's embedding dimension.
---
### Step 4: Multimodal LLM Integration
Concatenate the projected multi-scale visual embeddings with the tokenized textual instruction embeddings $\mathbf{z}_{\text{text}} \in \mathbb{R}^{l \times d_t}$ (where $l$ = length of the instruction text, $d_t$ = LLM embedding dimension) to form the input sequence to the LLM:
\$
Z = [z_c, z_p, z_r, z_{\text{WSI}}, z_{\text{text}}]
\$
The LLM backbone used is Qwen2.5-7B-Instruct, a SOTA open-source 7B parameter LLM with strong instruction-following and reasoning capabilities.
---
### Step 5: Three-Stage Training Strategy
#### Stage 1: Hierarchical Cross-Modal Alignment
In this stage, the hierarchical encoders (ViT_cell-cell, CONCH encoder, ViT_r, ViT_WSI) and text encoder are trained, while VL projectors and LLM weights are frozen. Training uses 9,642 WSI-report pairs. The loss function has three components:
a. Scale-specific contrastive loss for cell, patch, region levels:
\$
\mathcal{L}_s = -\frac{1}{n_s} \sum_{i} \log \frac{\exp(\text{sin}(z_{s,i}, t_i)/\tau)}{\sum_{j} \exp(\text{sin}(z_{s,i}, t_j)/\tau)}
\$
Where:
- $s \in \{c, p, r\}$ (cell, patch, region scales)
- $n_s$ = number of visual tokens at scale $s$
- $z_{s,i}$ = projected visual embedding of token $i$ at scale $s$
- $t_i$ = corresponding text token embedding from the pathology report
- $\tau$ = temperature parameter (set to 0.02)
- $\text{sin}(\cdot, \cdot)$ = cosine similarity
  This loss aligns each visual token at the cell/patch/region level to its corresponding semantic text token.
b. WSI-level contrastive loss:
\$
\mathcal{L}_{\text{WSI}} = -\frac{1}{n_b} \sum_{b} \log \frac{\exp(\text{sin}(z_{\text{WSI},b}, t_{r,b})/\tau)}{\sum_{l=1}^{n_b} \exp(\text{sin}(z_{\text{WSI},b}, t_{r,l})/\tau)}
\$
Where:
- $n_b$ = batch size
- $z_{\text{WSI},b}$ = projected WSI embedding of sample $b$
- $t_{r,b}$ = text embedding of the full pathology report for sample $b$
  This aligns the global WSI embedding to the full report embedding.
c. Cross-scale consistency loss: This enforces semantic coherence across scales, preventing drift between adjacent hierarchical levels:
\$
\mathcal{L}_c = \frac{1}{2n_r} \sum_{s \in \{c,p\}} \sum_{k=1}^{n_r} \left\| z_{r,k} - \frac{1}{n_s} \sum_{i=1}^{n_s} z_{s,k,i} \right\|_2^2 + \frac{1}{n_p} \sum_{j=1}^{n_p} \left\| z_{c_j} - z_{p_j} \right\|_2^2
\$
Where:
- First term: For each region $k$, the region embedding $z_{r,k}$ should be close to the average of the cell/patch embeddings within that region
- Second term: The aggregated cell embedding for patch $j$ $z_{c_j}$ should be close to the patch embedding $z_{p_j}$ for the same patch
- $\| \cdot \|_2^2$ = squared L2 norm measuring distance between two embeddings
  d. Total Stage 1 loss:
\$
\mathcal{L}_{\text{HCA}} = \frac{1}{n_b} \sum_{k=1}^{n_b} (\mathcal{L}_{s \in \{c,p,r\}}^k + \mathcal{L}_c^k) + \mathcal{L}_{\text{WSI}}
\$
Where $k$ indexes samples in the batch.
---
#### Stage 2: Feature Space Alignment
Only the four VL projectors are trained (encoders and LLM frozen) on the 9,642 WSI-report pairs, to map the multi-scale visual embeddings into the LLM's latent space.
---
#### Stage 3: Task-Specific Instruction Tuning
The VL projectors and LLM are jointly fine-tuned (encoders frozen) on 175,450 WSI-level VQA pairs from WSI-Bench, to enable task-specific reasoning. Low-Rank Adaptation (LoRA, rank=128, α=256) is used to efficiently fine-tune the LLM without full parameter updates.

# 5. Experimental Setup
## 5.1. Datasets
The model is evaluated on 6 core CPath tasks across 13 standard benchmark datasets:

| Task Category | Datasets | Description |
|---------------|----------|-------------|
| Zero-shot/Linear Probe Classification | BRACS, UBC-Ocean, TCGA-OT, EBRAINS, PANDA, IMP-CRC | Cover 7-class breast cancer subtyping, 5-class ovarian cancer subtyping, 46-class pan-cancer classification, 30-class brain tumor classification, 6-class prostate cancer Gleason grading, and 3-class colorectal cancer diagnosis. |
| Zero-shot VQA | WSI-Bench, WSI-VQA, SlideBench-VQA (BCNB), SlideBench-VQA (TCGA) | 27,862 total VQA pairs covering microscopy, diagnosis, morphological analysis, and treatment planning questions. |
| Report Generation | WSI-Bench, HistGen | 908 total WSI-report pairs. |
| WSI Retrieval | TCGA-OT, EBRAINS, IMP-CRC | Large-scale retrieval benchmarks. |
| Cross-modal Retrieval | TCGA Reports | Slide-to-report and report-to-slide retrieval tasks. |
| Caption Generation | SlideBench | WSI captioning benchmark. |

These datasets are widely used in the CPath community, cover diverse tissue types and clinical tasks, and are well-suited to validate the model's generalizability.
## 5.2. Evaluation Metrics
All metrics are explained with definition, formula, and symbol breakdown:
### 1. Balanced Accuracy (BA)
- **Conceptual Definition:** A classification metric for imbalanced datasets (common in medical imaging) that averages the recall of each class, avoiding bias towards majority classes.
- **Formula:**
  \$
  \text{BA} = \frac{1}{C} \sum_{i=1}^C \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i}
  \$
- **Symbol Explanation:** $C$ = number of classes, $\text{TP}_i$ = true positives for class $i$, $\text{FN}_i$ = false negatives for class $i$.
  ---
### 2. Weighted F1 Score
- **Conceptual Definition:** Harmonic mean of precision and recall, weighted by class size to account for imbalance, balancing false positive and false negative performance.
- **Formula:**
  \$
  \text{Weighted F1} = \sum_{i=1}^C \left( \frac{N_i}{N} \times \frac{2 \times \text{Precision}_i \times \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i} \right)
  \$
- **Symbol Explanation:** $N_i$ = number of samples in class $i$, $N$ = total samples, $\text{Precision}_i = \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i}$ (FP = false positives).
  ---
### 3. BLEU-n
- **Conceptual Definition:** Text generation metric measuring n-gram overlap between generated text and human-written reference text, with BLEU-1 measuring single-word overlap, BLEU-2 measuring 2-consecutive-word overlap, etc.
- **Formula:**
  \$
  \text{BLEU-n} = \text{BP} \times \exp\left( \sum_{i=1}^n \frac{1}{n} \log p_i \right)
  \$
- **Symbol Explanation:** $\text{BP}$ = brevity penalty for short generated text, $p_i$ = precision of i-grams (fraction of generated i-grams present in reference).
  ---
### 4. ROUGE-L
- **Conceptual Definition:** Text generation metric measuring the longest common subsequence (LCS) between generated and reference text, capturing sentence structure and semantic similarity better than BLEU for long clinical text.
- **Formula:**
  \$
  \text{ROUGE-L} = \frac{2 \times \text{R}_{\text{LCS}} \times \text{P}_{\text{LCS}}}{\text{R}_{\text{LCS}} + \text{P}_{\text{LCS}}}
  \$
- **Symbol Explanation:** $\text{R}_{\text{LCS}}$ = LCS recall (fraction of reference LCS in generated text), $\text{P}_{\text{LCS}}$ = LCS precision (fraction of generated LCS in reference text).
  ---
### 5. METEOR
- **Conceptual Definition:** Text generation metric accounting for synonymy, stemming, and word order, with higher correlation to human judgment of clinical text quality than BLEU.
- **Formula:**
  \$
  \text{METEOR} = (1 - \text{Penalty}) \times \frac{F \times R \times P}{0.5 R + 0.5 P}
  \$
- **Symbol Explanation:** $F$ = unigram F-score, $R$ = unigram recall, $P$ = unigram precision, $\text{Penalty}$ = penalty for incorrect word order.
  ---
### 6. Recall@K (R@K)
- **Conceptual Definition:** Retrieval metric measuring the fraction of queries where the correct target is present in the top K retrieved results.
- **Formula:**
  \$
  \text{R@K} = \frac{1}{Q} \sum_{q=1}^Q \mathbb{I}(\text{target}_q \in \text{top-K results for query } q)
  \$
- **Symbol Explanation:** $Q$ = number of queries, $\mathbb{I}(\cdot)$ = indicator function (1 if condition is true, 0 otherwise).
  ---
### 7. Top-1% Accuracy
- **Conceptual Definition:** Retrieval metric for large datasets, measuring if the correct target is present in the top 1% of retrieved results, suitable for real-world WSI databases with tens of thousands of slides.
- **Formula:**
  \$
  \text{Top-1\% Accuracy} = \frac{1}{Q} \sum_{q=1}^Q \mathbb{I}(\text{rank of target}_q \leq 0.01 \times N)
  \$
- **Symbol Explanation:** $N$ = total number of WSIs in the database, $\text{rank of target}_q$ = rank of the correct WSI in retrieved results.
## 5.3. Baselines
The model is compared against 24 representative SOTA models, grouped into categories:
1. **Zero-shot classification/retrieval baselines:** 10 leading CPath VLMs: PLIP, PathCLIP, MI-Zero, CONCH, QuiltNet, CPLIP, MR-PLIP, PathGenCLIP, TITAN, KEP, PRISM.
2. **Linear probe classification baselines:** 11 leading CPath visual foundation models: HIPT, TITAN, UNI, CTransPath, REMEDIS, CHIEF, DINOPath, Virchow, GigaPath, RudolfV.
3. **VQA/report/caption generation baselines:**
   - General MLLMs: GPT-4V, LLaVA-1.5, Qwen-VL-Max, Gemini-Pro-Vision.
   - CPath-specific MLLMs: Quilt-LLaVA, SlideChat, WSI-LLaVA, PRISM, MedDr, LLaVA-Med, HistGen, PathGen-LLaVA.
     These baselines cover all major model categories in the CPath domain, ensuring fair and meaningful comparisons.

# 6. Results & Analysis
## 6.1. Core Results Analysis
MLLM-HWSI achieves new SOTA performance across all 6 task categories:
1. **Zero-shot Classification:** Average balanced accuracy (BA) of 71.86%, surpassing prior SOTA TITAN (64.56%) by 7.30% and WSI-LLaVA (61.01%) by 10.85%.
2. **Linear Probe Classification:** Average BA of 82.48%, outperforming TITAN (75.68%) by 6.80% and UNI (72.86%) by 9.62%, confirming the discriminative power of its multi-scale representations.
3. **WSI Retrieval:** Average top-1% accuracy of 85.62%, outperforming TITAN (80.06%) by 5.56% and CONCH (73.74%) by 11.88%.
4. **VQA:** Average accuracy of 89.60% on SlideBench-VQA (TCGA), 68.70% on SlideBench-VQA (BCNB), 97.90% on WSI-Bench, and 69.20% on WSI-VQA, surpassing all prior models by large margins.
5. **Report Generation:** Outperforms all SOTA models across BLEU, ROUGE-L, and METEOR metrics on both WSI-Bench and HistGen datasets.
6. **Caption Generation:** BLEU-1/2/3/4 of 46.20%/32.40%/26.70%/23.10%, ROUGE-L = 36.70%, METEOR = 62.70% on SlideBench-Caption, outperforming WSI-LLaVA by 12-15% across all metrics.
7. **Cross-modal Retrieval:** Average R@K of 87.5% (report-to-slide) and 92.0% (slide-to-report), outperforming WSI-LLaVA by 4.70% and 6.10% respectively.
   The consistent performance gains across all tasks and datasets strongly validate the effectiveness of the hierarchical multi-scale alignment approach.
## 6.2. Data Presentation (Tables)
### Table 1: Ablation Study on Hierarchical Representations
The following are the results from Table 1 of the original paper:

| Models | Cell Feat. | Patch Feat. | Region Feat. | WSI Feat. | PANDA [12] (BA) | EBRAINS [58] (BA) | WSI-VQA [14] (A) | SlideBench-VQA (BCNB) [17] (A) |
|--------|------------|-------------|--------------|-----------|-----------------|-------------------|------------------|--------------------------------|
| WSI-LLaVA [44] | × | × | × | ✓ | 0.644 | 0.501 | 0.546 | 0.553 |
| SlideChat [17] | × | × | × | ✓ | 0.633 | 0.466 | 0.601 | 0.541 |
| MLLM-HWSI1 | × | × | × | ✓ | 0.661 | 0.519 | 0.616 | 0.576 |
| MLLM-HWSI2 | × | × | ✓ | ✓ | 0.686 | 0.534 | 0.611 | 0.592 |
| MLLM-HWSI3 | × | ✓ | ✓ | ✓ | 0.711 | 0.566 | 0.661 | 0.621 |
| MLLM-HWSI4 | ✓ | × | × | ✓ | 0.674 | 0.531 | 0.613 | 0.588 |
| MLLM-HWSI5 | × | ✓ | × | ✓ | 0.698 | 0.548 | 0.623 | 0.606 |
| MLLM-HWSI6 | ✓ | ✓ | × | ✓ | 0.715 | 0.575 | 0.669 | 0.640 |
| MLLM-HWSI7 | ✓ | × | ✓ | ✓ | 0.714 | 0.587 | 0.668 | 0.653 |
| MLLM-HWSI | ✓ | ✓ | ✓ | ✓ | 0.748 | 0.612 | 0.692 | 0.687 |

---
### Table 3: VQA Performance Comparison
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">General MLLMs</th>
<th colspan="4">SlideBench-VQA (TCGA)</th>
<th colspan="4">WSI-Bench</th>
<th rowspan="2">SlideBench-VQA(BCNB)</th>
<th rowspan="2">WSI-VQA</th>
</tr>
<tr>
<th>Micro.</th>
<th>Diag.</th>
<th>Clinical</th>
<th>Average</th>
<th>MA</th>
<th>Diag.</th>
<th>TP</th>
<th>Average</th>
</tr>
</thead>
<tbody>
<tr>
<td>InstructBLIP-FLAN</td>
<td>0.366</td>
<td>0.186</td>
<td>0.221</td>
<td>0.257</td>
<td>0.198</td>
<td>0.221</td>
<td>0.389</td>
<td>0.269</td>
<td>0.189</td>
<td>0.102</td>
</tr>
<tr>
<td>LLaVA-1.5</td>
<td>0.451</td>
<td>0.219</td>
<td>0.389</td>
<td>0.353</td>
<td>0.232</td>
<td>0.271</td>
<td>0.677</td>
<td>0.393</td>
<td>0.201</td>
<td>0.121</td>
</tr>
<tr>
<td>Qwen-VL-MAX</td>
<td>0.496</td>
<td>0.288</td>
<td>0.405</td>
<td>0.396</td>
<td>0.288</td>
<td>0.322</td>
<td>0.706</td>
<td>0.438</td>
<td>0.223</td>
<td>0.133</td>
</tr>
<tr>
<td>GeminiProV</td>
<td>0.506</td>
<td>0.304</td>
<td>0.587</td>
<td>0.465</td>
<td>0.403</td>
<td>0.433</td>
<td>0.821</td>
<td>0.552</td>
<td>0.282</td>
<td>0.167</td>
</tr>
<tr>
<td>GPT-4V</td>
<td>0.628</td>
<td>0.466</td>
<td>0.667</td>
<td>0.587</td>
<td>0.471</td>
<td>0.530</td>
<td>0.875</td>
<td>0.625</td>
<td>0.414</td>
<td>0.304</td>
</tr>
<tr>
<th rowspan="2">CPath MLLMs</th>
<th colspan="4">SlideBench-VQA (TCGA)</th>
<th colspan="4">WSI-Bench</th>
<th rowspan="2">SlideBench-VQA(BCNB)</th>
<th rowspan="2">WSI-VQA</th>
</tr>
<tr>
<th>Micro.</th>
<th>Diag.</th>
<th>Clinical</th>
<th>Average</th>
<th>MA</th>
<th>Diag.</th>
<th>TP</th>
<th>Average</th>
</tr>
<tr>
<td>LLaVA-Med</td>
<td>0.458</td>
<td>0.275</td>
<td>0.408</td>
<td>0.380</td>
<td>0.803</td>
<td>0.866</td>
<td>0.732</td>
<td>0.912</td>
<td>0.836</td>
<td>0.187</td>
</tr>
<tr>
<td>Quilt-LLaVA</td>
<td>0.491</td>
<td>0.269</td>
<td>0.447</td>
<td>0.402</td>
<td>0.947</td>
<td>0.849</td>
<td>1.000</td>
<td>0.932</td>
<td>0.415</td>
<td>0.354</td>
</tr>
<tr>
<td>PathGen-LLaVA</td>
<td>0.566</td>
<td>0.321</td>
<td>0.509</td>
<td>0.465</td>
<td>0.882</td>
<td>0.781</td>
<td>0.922</td>
<td>0.861</td>
<td>0.401</td>
<td>0.331</td>
</tr>
<tr>
<td>MedDr</td>
<td>0.733</td>
<td>0.577</td>
<td>0.742</td>
<td>0.684</td>
<td>0.902</td>
<td>0.831</td>
<td>0.922</td>
<td>0.885</td>
<td>0.336</td>
<td>0.543</td>
</tr>
<tr>
<td>WSI-VQA</td>
<td>0.334</td>
<td>0.189</td>
<td>0.306</td>
<td>0.276</td>
<td>0.758</td>
<td>0.577</td>
<td>0.771</td>
<td>0.702</td>
<td>0.113</td>
<td>0.469</td>
</tr>
<tr>
<td>TITAN</td>
<td>0.851</td>
<td>0.745</td>
<td>0.824</td>
<td>0.806</td>
<td>0.940</td>
<td>0.883</td>
<td>1.000</td>
<td>0.941</td>
<td>0.551</td>
<td>0.586</td>
</tr>
<tr>
<td>SlideChat</td>
<td>0.876</td>
<td>0.732</td>
<td>0.842</td>
<td>0.816</td>
<td>0.932</td>
<td>0.858</td>
<td>0.971</td>
<td>0.920</td>
<td>0.541</td>
<td>0.601</td>
</tr>
<tr>
<td>WSI-LLaVA</td>
<td>0.882</td>
<td>0.752</td>
<td>0.841</td>
<td>0.825</td>
<td>0.951</td>
<td>0.863</td>
<td>1.000</td>
<td>0.938</td>
<td>0.553</td>
<td>0.546</td>
</tr>
<tr>
<td>MLLM-HWSI</td>
<td>0.956</td>
<td>0.824</td>
<td>0.908</td>
<td>0.896</td>
<td>0.989</td>
<td>0.962</td>
<td>0.986</td>
<td>0.979</td>
<td>0.687</td>
<td>0.692</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies
### 1. Hierarchical Representation Ablation (Table 1)
- Progressive addition of hierarchical features shows consistent performance gains: using only WSI features (MLLM-HWSI1) already outperforms baseline global-only models, adding region/patch/cell features sequentially improves performance across all datasets.
- Subtractive ablations (removing one scale at a time, e.g., MLLM-HWSI4 removes patch/region features) cause consistent performance drops, confirming that every hierarchical level contributes to the model's performance.
### 2. Loss Function Ablation
- Removing any component of the joint loss function degrades performance. When only the WSI-level contrastive loss is retained (removing cross-scale consistency loss and scale-specific contrastive losses), PANDA classification BA drops by 8.70%, EBRAINS BA drops by 9.30%, WSI-VQA accuracy drops by 7.60%, and SlideBench-VQA accuracy drops by 11.10%.
- This confirms that both hierarchical contrastive alignment across scales and cross-scale consistency are necessary for optimal performance.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces MLLM-HWSI, the first hierarchical WSI-level MLLM that explicitly aligns visual features across four scales (cell, patch, region, WSI) with pathology language, mirroring the multi-scale diagnostic workflow of human pathologists. The model uses a novel joint training objective of hierarchical contrastive loss and cross-scale consistency loss, plus efficiency modules for patch filtering and cell embedding aggregation, and achieves new SOTA results across 13 benchmarks and 6 core CPath tasks. The work establishes a new paradigm for interpretable foundation models in CPath, with significant potential to assist pathologists in clinical decision-making by providing evidence-grounded, diagnostically accurate outputs.
## 7.2. Limitations & Future Work
### Limitations noted by authors:
1. The model currently only supports histopathology WSIs, not other medical modalities.
2. The cell segmentation step (using CellViT) may introduce errors in cases with poor staining or overlapping cells, which could degrade performance.
3. The model requires large amounts of annotated WSI-report pairs for training, which are expensive and time-consuming to collect.
### Future work suggested by authors:
1. Extend MLLM-HWSI to integrate other medical modalities (radiology images, genomics data, clinical records) to enable holistic patient-level reasoning.
2. Improve cell segmentation robustness for low-quality WSIs.
3. Explore semi-supervised/self-supervised training methods to reduce reliance on annotated WSI-report pairs.
## 7.3. Personal Insights & Critique
### Key Inspirations
The analogy between WSI hierarchical structure and natural language structure (cell=word, patch=phrase, etc.) is a highly intuitive and effective design choice that aligns the model's architecture with both biological reality and clinical workflow. This approach is broadly transferable to other domains with hierarchical data, including remote sensing (satellite images have pixel → object → region → scene hierarchy) and document analysis (documents have character → word → sentence → paragraph hierarchy).
### Potential Improvements
1. **Computational efficiency:** The current model uses four separate encoders for each scale, increasing inference cost compared to global-only MLLMs. Exploring shared encoder architectures across scales would reduce deployment costs for clinical settings.
2. **Explicit interpretability:** While the model uses multi-scale features, it does not explicitly output bounding boxes or heatmaps for the visual evidence supporting its diagnostic claims, which is a key requirement for clinical adoption. Adding explicit grounding outputs would significantly improve clinical utility.
3. **Adaptive hierarchy:** The paper assumes a fixed four-level hierarchy is optimal for all tasks, but adaptive hierarchical levels for different tissue types or tasks (e.g., more granular levels for cytology tasks, coarser levels for screening tasks) could further improve performance and efficiency.
### Unverified Assumptions
The paper assumes that the semantic mapping (cell=word, patch=phrase, etc.) is optimal for all pathology tasks, but this mapping has not been explicitly validated against human pathologists' semantic organization of diagnostic information. Future work could validate this mapping via human subject studies with pathologists.