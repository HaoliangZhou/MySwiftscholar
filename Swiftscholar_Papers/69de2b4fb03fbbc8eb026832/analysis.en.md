# 1. Bibliographic Information
## 1.1. Title
The central topic of the paper is the development of **UniME-V2**, a universal multimodal embedding model that uses Multimodal Large Language Models (MLLMs) as an automated judge to improve hard negative mining and representation learning, alongside a complementary reranking module for enhanced retrieval performance.
## 1.2. Authors
The authors and their affiliations are:
- Tiancheng Gu, Yueyi Zhang: MiroMind AI
- Kaicheng Yang, Kaichen Zhang, Xiang An, Ziyong Feng, Weidong Cai: The University of Sydney
- Jiankang Deng: Imperial College London
- Lidong Bing: M.R.L Team, LMMs-Lab Team
  The research group has extensive prior work in multimodal learning, including the earlier UniME model, and expertise in MLLM development and embedding learning.
## 1.3. Journal/Conference
The paper is published as a preprint on arXiv, a widely used open-access repository for academic research in computer science. As of 2026-04, it has not yet been peer-reviewed or accepted for publication in a formal conference or journal. arXiv preprints are standard for quickly disseminating cutting-edge AI research results.
## 1.4. Publication Year
The paper was published on 2025-10-15 (UTC).
## 1.5. Abstract
### Research Objective
To address critical limitations of existing multimodal embedding models: poor ability to capture subtle semantic differences between candidates, low diversity of negative training samples, and limited discriminative power to distinguish false negatives from hard negatives.
### Core Methodology
1.  An `MLLM-as-a-Judge` pipeline: First construct a global potential hard negative set via retrieval, then use an MLLM to generate continuous soft semantic matching scores for each query-candidate pair to filter false negatives and select high-quality diverse hard negatives.
2.  A distribution alignment training framework: Use the MLLM-derived soft scores as soft labels, align the embedding similarity matrix with the semantic score matrix via Jensen-Shannon (JS) divergence loss to relax rigid one-to-one mapping constraints.
3.  UniME-V2-Reranker: A reranking model trained on the mined hard negatives via joint pairwise and listwise optimization to further improve retrieval performance.
### Main Results
UniME-V2 achieves state-of-the-art (SOTA) average performance across all 36 tasks in the Massive Multimodal Embedding Benchmark (MMEB), as well as on short/long caption cross-modal retrieval and compositional retrieval tasks. The reranker further boosts performance, especially on hard compositional retrieval tasks.
### Key Conclusions
Leveraging MLLMs' strong cross-modal semantic understanding for both hard negative mining and soft supervision significantly improves the quality of universal multimodal embeddings, leading to better discriminative ability and transferability.
## 1.6. Original Source Link
- Official preprint page: https://arxiv.org/abs/2510.13515
- PDF link: https://arxiv.org/pdf/2510.13515v3
- Publication status: Preprint (not yet peer-reviewed for formal conference/journal publication)

  ---

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Existing universal multimodal embedding models (e.g., CLIP, VLM2Vec, UniME) rely on in-batch negative mining for contrastive learning, which suffers from three critical flaws:
1.  Cannot capture fine-grained semantic differences between candidates, as they use rigid binary labels (1 for positive, 0 for all negatives).
2.  Negative samples have limited diversity, as they are restricted to samples within the same training batch.
3.  Embeddings have low discriminative power, as the models cannot effectively distinguish false negatives (samples incorrectly labeled as negative that are actually semantically related to the query) from hard negatives (samples that are semantically close to the query but not a match).
### Importance of the Problem
Universal multimodal embeddings are foundational for a wide range of downstream applications, including visual question answering (VQA), multimodal retrieval, visual grounding, and classification. Poor embedding quality directly limits the performance of all these applications.
### Research Gap
Prior work has not fully exploited the strong cross-modal semantic understanding capabilities of modern MLLMs to improve negative mining and training supervision for embedding learning.
### Innovative Entry Point
Use MLLMs as an automated judge to assess query-candidate semantic alignment, generate continuous soft matching scores, and use these scores both to mine high-quality hard negatives and as soft supervision for model training.
## 2.2. Main Contributions / Findings
### Primary Contributions
1.  **MLLM-as-a-Judge Hard Negative Mining Pipeline**: Combines global retrieval to build a diverse potential negative set, with MLLM scoring to filter false negatives and select high-quality, diverse hard negatives.
2.  **UniME-V2 Embedding Model**: Uses a distribution alignment training framework that aligns the embedding similarity distribution with the MLLM-derived semantic score distribution via JS divergence loss, relaxing rigid one-to-one mapping constraints and improving discriminative ability.
3.  **UniME-V2-Reranker**: A reranking module trained on the mined hard negatives with joint pairwise and listwise optimization, further boosting retrieval performance, especially on hard compositional tasks.
4.  **Extensive Empirical Validation**: Comprehensive experiments on the MMEB benchmark and multiple retrieval tasks demonstrate SOTA performance across all evaluated settings.
### Key Findings
- MLLMs' semantic assessment can be used as a high-quality supervision signal for embedding learning, outperforming traditional binary label supervision.
- Mining diverse, high-quality hard negatives with MLLM filtering significantly improves the model's ability to distinguish fine-grained semantic differences, especially for compositional tasks.
- The two-stage pipeline (embedding retrieval + reranker) achieves the best balance of efficiency and performance, with the reranker delivering large gains on hard tasks while only requiring processing of a small top-k candidate set.

  ---

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All key terms are explained below for beginners:
- **Universal Multimodal Embedding**: A model that encodes different types of multimodal data (text, images, interleaved image-text, etc.) into the same dense vector space, so that semantically similar content (regardless of modality) has vectors that are close to each other. These embeddings are used for retrieval, classification, VQA, and other downstream tasks.
- **Contrastive Learning**: A training paradigm where the model learns to pull embeddings of matching (positive) pairs close to each other, and push embeddings of non-matching (negative) pairs far apart.
- **InfoNCE Loss**: The most widely used loss function for contrastive learning, which calculates the probability that the positive pair is more similar to the query than all negative pairs. The standard formula is:
  $$
  \mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, c^+) / \tau)}{\exp(\text{sim}(q, c^+) / \tau) + \sum_{i=1}^k \exp(\text{sim}(q, c_i^-) / \tau)}
  $$
  Explanation: $q$ = query embedding, $c^+$ = positive candidate embedding, $c_i^-$ = negative candidate embeddings, $\text{sim}()$ = cosine similarity function, $\tau$ = temperature hyperparameter that controls the sharpness of the probability distribution.
- **Hard Negatives**: Negative samples that are semantically very close to the query, making them hard for the model to distinguish from the positive sample. Training on hard negatives significantly improves the model's discriminative ability.
- **False Negatives**: Samples that are incorrectly labeled as negatives, but are actually semantically related to the query. Training on false negatives hurts model performance, as it teaches the model to push semantically similar content apart.
- **MLLM (Multimodal Large Language Model)**: A large language model extended to process visual inputs (images, videos) alongside text, with strong cross-modal semantic understanding capabilities.
- **Reranking**: A two-stage retrieval process: first, use fast embedding-based similarity search to retrieve a small set of top relevant candidates; second, use a more powerful (but slower) model to reorder these candidates to get more accurate final results.
- **KL Divergence (Kullback-Leibler Divergence)**: A measure of how different two probability distributions are, which is asymmetric (i.e., $\text{KL}(P||Q) \neq \text{KL}(Q||P)$).
- **JS Divergence (Jensen-Shannon Divergence)**: A symmetric, bounded version of KL divergence (values range from 0 to 1), used to measure the similarity between two distributions.
## 3.2. Previous Works
### CLIP (Radford et al. 2021)
The pioneering dual-encoder cross-modal contrastive learning model, trained on millions of web-crawled image-text pairs. It uses InfoNCE loss to align image and text embeddings. Limitations include: 77-token text limit, independent encoding of images and text (no cross-modal fusion), poor compositional understanding, and bag-of-words behavior for text.
### Recent MLLM-based Embedding Works
- **E5-V (Jiang et al. 2024)**: Uses unimodal contrastive learning on the language component of MLLMs to reduce cross-modal representation gaps.
- **VLM2Vec (Jiang et al. 2025)**: Introduces the MMEB benchmark (36 datasets across 4 meta-tasks) and adapts pre-trained vision-language models into embedding models via contrastive learning on MMEB.
- **QQMM (Xue et al. 2025)**: Analyzes gradients of InfoNCE loss and amplifies gradients associated with hard negatives to improve embedding discriminability.
- **UniME (Gu et al. 2025a)**: Two-stage framework with an LLM-based teacher model to refine language embeddings, and uses in-batch hard negative sampling.
### Limitations of Prior Work
All existing methods rely on in-batch negative mining, which has limited diversity and high false negative rates. They also use rigid binary labels that cannot capture fine-grained semantic differences between candidates, and do not leverage MLLMs' strong semantic understanding for supervision.
## 3.3. Technological Evolution
The evolution of multimodal embedding learning follows this timeline:
1.  2021: CLIP establishes the dual-encoder cross-modal contrastive learning paradigm.
2.  2023: MLLM boom (LLaVA, CogVLM, Qwen-VL) demonstrates strong cross-modal semantic understanding capabilities.
3.  2024: Works like E5-V and VLM2Vec start adapting MLLMs for universal multimodal embedding learning.
4.  2025: UniME, QQMM improve hard negative sampling for MLLM-based embedding models, but still use in-batch mining and binary labels.
5.  2025 (this work): UniME-V2 is the first work to use MLLMs as judges for both global hard negative mining and soft label supervision, achieving SOTA performance.
## 3.4. Differentiation Analysis
Compared to prior methods, UniME-V2 has three core innovations:
1.  **Negative Mining**: Prior work uses in-batch negative mining, while UniME-V2 uses global retrieval to get diverse potential negatives, then MLLM judgment to filter false negatives and select high-quality hard negatives.
2.  **Training Supervision**: Prior work uses rigid binary labels, while UniME-V2 uses continuous soft semantic matching scores from MLLMs as soft labels, aligning the embedding similarity distribution with the semantic score distribution via JS divergence loss to capture fine-grained semantic differences.
3.  **Reranking**: Prior rerankers use generic negative samples, while UniME-V2-Reranker is trained on the high-quality mined hard negatives with joint pairwise and listwise optimization, delivering especially large gains on compositional tasks.

    ---

# 4. Methodology
## 4.1. Principles
The core idea of UniME-V2 is to leverage MLLMs' advanced cross-modal semantic understanding capabilities to solve the two key pain points of existing multimodal embedding learning: low-quality hard negative samples, and rigid binary supervision that cannot capture subtle semantic differences between candidates.
The theoretical intuition is: modern MLLMs have achieved near-human-level performance on cross-modal semantic alignment judgment tasks, so their assessment of query-candidate matching can be used as a high-quality, fine-grained supervision signal for embedding learning.
The following figure (Figure 1 from the original paper) illustrates the difference between traditional methods and UniME-V2:

![Figure 1: Comparison between previous works and UniME-V2. UniME-V2 exploits the understanding capabilities of MLLMs for hard negatives mining and generates a soft semantic matching score to supervise the model in learning the semantic difference among candidates.](images/1.jpg)
*该图像是示意图，展示了UniME-V2模型与传统方法的对比。上半部分为传统方法，展示了查询及候选框架和训练目标；下半部分为UniME-V2，展示了MLLM-as-a-Judge机制进行困难负样本挖掘的流程和语义匹配分数的生成。通过这些机制，UniME-V2提升了模型的语义区分能力。*

## 4.2. Core Methodology In-depth (Layer by Layer)
### Task Definition
The universal multimodal retrieval task is defined as follows:
Given a query $q$ (can be text, image, or interleaved image-text) and a candidate set $\Omega_c = \{c_1, c_2, ..., c_n\}$ (supports the same modalities as the query):
1.  The embedding model $\Phi_{emb}$ encodes the query and all candidates into the same vector space, then retrieves the top-$k$ most relevant candidates $\Omega_k = \Phi_{emb}(q, \Omega_c)$ using cosine similarity.
2.  The reranker model $\Phi_{rank}$ reranks the top-$k$ candidates to produce the final ranked output $\hat{\Omega}_k = \Phi_{rank}(q, \Omega_k)$.
    ---
### Component 1: MLLM-as-a-Judge for Hard Negatives Mining
The following figure (Figure 2 from the original paper) shows this pipeline:

![该图像是一个示意图，展示了UniME-V2模型在多模态嵌入学习中的过程。图中包含查询、潜在的困难负样本以及候选样本的结构，体现了MLLM-as-a-Judge机制如何用于评估查询与候选对的语义对齐，并生成软语义匹配分数，助力挖掘高质量的困难负样本。](images/2.jpg)
*该图像是一个示意图，展示了UniME-V2模型在多模态嵌入学习中的过程。图中包含查询、潜在的困难负样本以及候选样本的结构，体现了MLLM-as-a-Judge机制如何用于评估查询与候选对的语义对齐，并生成软语义匹配分数，助力挖掘高质量的困难负样本。*

#### Step 1: Construct Potential Hard Negative Set
To address the limited diversity of in-batch negatives, the authors first build a global potential hard negative set:
1.  Use a pre-trained embedding model (VLM2Vec) to generate embeddings for all queries and candidates in the training dataset.
2.  For each query, retrieve the top 50 candidates with the highest cosine similarity to the query.
3.  Apply a similarity threshold $\delta$ to filter out candidates that are too similar to the query (to avoid including positive samples), resulting in the potential hard negative set $\Omega_p$:
    $$
    \Omega_p = \text{Rank}_{50}\left(\{x_1, ..., x_n\}\right), \text{ where } x_i < \delta
    $$
    Explanation: $x_i$ is the cosine similarity between query $q$ and candidate $i$ calculated by VLM2Vec, $\text{Rank}_{50}$ means selecting the top 50 candidates with highest similarity that are below threshold $\delta$.
This set contains candidates that are semantically close to the query, so they are potential hard negatives, but may still include false negatives.
#### Step 2: Generate Semantic Matching Scores
After obtaining $\Omega_p$, use an MLLM as a judge to assess the semantic alignment of each query-candidate pair in $\Omega_p$:
1.  The MLLM is given the following instruction:
    > "I will provide you with a query and a candidate. Please evaluate whether the candidate meets the requirements of the query. If it does, respond with 'Yes'; if it doesn't, respond with 'No'. Query: <Query>, Candidates: <Candidate>."
2.  Instead of using the binary Yes/No output, calculate a continuous semantic matching score $s_i$ for the $i$-th query-candidate pair using the logits of the "Yes" token $e_y^i$ and "No" token $e_n^i$ from the MLLM's output layer:
    $$
    s_i = \frac{e_y^i}{e_y^i + e_n^i}
    $$
Explanation: $s_i$ ranges from 0 to 1, where 1 means the candidate is perfectly aligned with the query, and 0 means fully unrelated. This score captures the degree of semantic alignment, not just a binary match/mismatch. The full matrix of scores for all query-candidate pairs is $S \in \mathbb{R}^{n_q \times 50}$, where $n_q$ is the number of queries.
#### Step 3: Hard Negative Sampling
Use the semantic matching scores to filter false negatives and select diverse high-quality hard negatives:
1.  Set a threshold $\alpha = \sigma_{q, c_t} - \beta$, where $c_t$ is the positive sample for query $q$, and $\beta$ is a hyperparameter set to 0.01. Any candidate with a score higher than $\alpha$ is considered a false negative (since it is almost as aligned as the positive sample) and is excluded.
2.  Use a cyclical sampling strategy with 5-step intervals to select candidates from the remaining set, to ensure diversity of hard negatives.
3.  If the refined set has fewer than 10 candidates, duplicate selections to get at least 10. For rare cases with no valid candidates (<1% of all queries), randomly select 10 candidates from the initial 50 and assign them a semantic matching score of 1.0.
4.  Final output per query: hard negative set $\Omega_h = \{c_1, ..., c_k\}$ and their corresponding semantic matching scores $S_h = \{s_{q,c_1}, ..., s_{q,c_k}\}$, where $k$ is the number of hard negatives per query (set to 8 in experiments).
    ---
### Component 2: MLLM Judgment Based Training Framework
The following figure (Figure 3 from the original paper) shows this framework:

![Figure 3: The architecture of the MLLM Judgment Based Training Framework. UniME-V2 uses soft semantic matching scores as supervised signals to enhance semantic distinction learning between candidates. UniME-V2-Reranker employs joint pairwise and listwise optimization to enhance reranking performance.](images/3.jpg)
*该图像是一个示意图，展示了UniME-V2模型及其重排序模块UniME-V2-Reranker的结构。图中使用软语义匹配分数作为监督信号，以增强候选之间的语义区分学习，同时采用联合对偶和列表重排序的方法提升重排序性能。*

#### Subcomponent 1: UniME-V2 Embedding Model Training
Prior contrastive learning uses rigid binary labels: only the positive sample has label 1, all negatives have label 0, which ignores the different degrees of semantic similarity between the query and different negatives. UniME-V2 uses the soft semantic matching scores as supervision to align the embedding similarity distribution with the semantic score distribution.
##### Step 1: Extract Embeddings
For each query $q$, its full candidate set is $\Omega_c = \{c_t, c_1, ..., c_k\}$, where $c_t$ is the positive sample, and $c_1 ... c_k$ are the mined hard negatives. Input all samples to the MLLM backbone, and extract the embedding of the last token as the representation of the input:
- Query embedding: $e_q$
- Candidate embeddings: $E_c = \{e_c^+, e_{c_1}^-, ..., e_{c_k}^-\}$, where $e_c^+$ is the positive candidate embedding, $e_{c_i}^-$ are hard negative embeddings.
##### Step 2: Compute Embedding Similarity Probability Matrix $\mathbb{P}$
Calculate the probability that the positive sample is the most similar to the query, using cosine similarity and temperature $\tau$:
$$
\mathbb{P}(e_q, E_c) = \frac{\exp(\cos(e_q, e_c^+) / \tau)}{\exp(\cos(e_q, e_c^+) / \tau) + \sum_{i=1}^k \exp(\cos(e_q, e_{c_i}^-) / \tau)}
$$
Explanation: $\cos(a,b)$ is the cosine similarity between vectors $a$ and $b$, $\tau$ is a temperature hyperparameter (set to 0.02 in experiments) that controls the sharpness of the probability distribution. $\mathbb{P}$ represents the similarity distribution output by the embedding model.
##### Step 3: Compute Semantic Matching Probability Matrix $\mathbb{Q}$
The semantic matching scores for the candidate set are $S_c = \{s_{q,c_t}, s_{q,c_1}, ..., s_{q,c_k}\}$. Use the same temperature $\tau$ to compute the probability distribution of the semantic scores, which serves as the "ground truth" distribution from the MLLM judge:
$$
\mathbb{Q}(S_c) = \frac{\exp(s_{q,c_t} / \tau)}{\exp(s_{q,c_t} / \tau) + \sum_{i=1}^k \exp(s_{q,c_i} / \tau)}
$$
##### Step 4: Compute Loss Function
To align the two distributions $\mathbb{P}$ and $\mathbb{Q}$, the authors use JS Divergence, which is symmetric (unlike KL Divergence) to improve training robustness. The JS Divergence is the average of $\text{KL}(\mathbb{P}||\mathbb{Q})$ and $\text{KL}(\mathbb{Q}||\mathbb{P})$:
$$
\mathcal{L} = \frac{1}{2}\left( \frac{1}{N}\sum_{i=1}^N \text{KL}(\mathbb{P}(e_i, E_c) || \mathbb{Q}(S_c)) + \frac{1}{N}\sum_{i=1}^N \text{KL}(\mathbb{Q}(S_c) || \mathbb{P}(e_i, E_c)) \right)
$$
Explanation: $N$ is the number of training samples, $\text{KL}(A||B)$ is the KL divergence from distribution B to A. This loss encourages the embedding model's similarity distribution to match the MLLM's semantic alignment distribution, so the model learns to capture subtle semantic differences between candidates.
---
#### Subcomponent 2: UniME-V2-Reranker Training
To further improve retrieval performance, the authors train a reranker model on the mined hard negatives, using joint pairwise and listwise optimization.
##### Step 1: Pairwise Training
For each query $q$, construct two pairs: (query $q$, positive candidate $c_t$) and (query $q$, hardest negative $c_h$). The reranker is instructed to output "Yes" for the positive pair and "No" for the negative pair. The pairwise loss is cross-entropy loss:
$$
\mathcal{L}_{pair} = \mathcal{L}_{ce}(\text{YES}, \eta(q, c_t)) + \mathcal{L}_{ce}(\text{NO}, \eta(q, c_h))
$$
Explanation: $\eta$ is the autoregressive output process of the reranker MLLM, $\mathcal{L}_{ce}$ is cross-entropy loss. This loss teaches the reranker to distinguish between positive and hardest negative pairs.
##### Step 2: Listwise Training
For each query, select the top-$x$ candidates from the hard negative set, insert the positive candidate $c_t$ at a random position, and get the index of the positive candidate $I_{c_t}$. The reranker is instructed to output the position of the best matching candidate (the positive one). The listwise loss is cross-entropy loss on the output index:
$$
\mathcal{L}_{list} = \mathcal{L}_{ce}(I_{c_t}, \eta(q, c_t, \{c_1, ..., c_x\}))
$$
Explanation: This loss teaches the reranker to rank the full list of candidates correctly, not just compare pairs.
##### Step 3: Final Reranker Loss
Combine the two losses for joint optimization:
$$
\mathcal{L} = \mathcal{L}_{pair} + \mathcal{L}_{list}
$$
---
### Inference Pipeline
1.  **First stage**: Use the trained UniME-V2 embedding model to encode all queries and candidates, compute cosine similarity between the query and all candidates, and retrieve the top-10 most relevant candidates.
2.  **Second stage**: Use UniME-V2-Reranker to rerank the top-10 candidates, using the following instruction:
    > "I will provide you with a query followed by multiple candidates in the format: (1) candidate1 (2) candidate2, etc. Each candidate is independent of the others. Evaluate each candidate against the query, and respond with the number corresponding to the candidate that best meets the requirements of the query. Query: <Query>, Candidates: <Candidate list>."

---

# 5. Experimental Setup
## 5.1. Datasets
### Training Datasets
The authors use 20 in-distribution datasets from the MMEB benchmark, covering four core multimodal tasks: classification, VQA, multimodal retrieval, and visual grounding. The training corpus includes both unimodal and multimodal input data, with a total of 662k curated training pairs.
### Evaluation Datasets

| Dataset Category | Datasets | Characteristics |
|------------------|----------|-----------------|
| MMEB Benchmark | 36 total datasets: 20 in-distribution (IND) test sets, 16 out-of-distribution (OOD) test sets | Covers 4 meta-tasks: Classification (10 datasets), VQA (10 datasets), Retrieval (12 datasets), Visual Grounding (4 datasets) |
| Short Caption Cross-Modal Retrieval | Flickr30K, COCO2014 | Flickr30K: 1k test queries, 5k candidates; COCO: 5k test queries, 25k candidates, tests retrieval performance on short, simple captions |
| Long Caption Cross-Modal Retrieval | ShareGPT4V, Urban1K | 1k test queries, 1k candidates each, tests retrieval performance on long, detailed captions with rich semantic information |
| Compositional Cross-Modal Retrieval | SugarCrepe | 7.5k test queries, 2 candidates per query, tests ability to distinguish hard compositional negatives (e.g., swapped attributes, added/removed objects) |

### Rationale for Dataset Selection
- MMEB is the standard, most comprehensive benchmark for universal multimodal embedding models, covering diverse tasks and domains to test generalizability.
- The retrieval datasets test specific capabilities (short/long text, compositional understanding) that are critical for real-world retrieval applications.
## 5.2. Evaluation Metrics
The paper uses two main evaluation metrics:
### Metric 1: Precision (for MMEB Benchmark)
1.  **Conceptual Definition**: Precision@k measures the proportion of correct relevant candidates among the top-k retrieved results for each query. It focuses on the accuracy of the top retrieved results. For all results in this paper, k=1 is used.
2.  **Mathematical Formula**:
    $$
    \text{Precision@1} = \frac{\text{Number of queries where the top-1 retrieved result is relevant}}{\text{Total number of queries}}
    $$
3.  **Symbol Explanation**: No additional variables, it is a count-based ratio.
### Metric 2: Recall@1 (for retrieval tasks)
1.  **Conceptual Definition**: Recall@1 measures the proportion of queries for which the correct positive candidate is ranked first in the retrieval results. It focuses on whether the most relevant result is found at the top position.
2.  **Mathematical Formula**:
    $$
    \text{Recall@1} = \frac{\text{Number of queries where the positive candidate is ranked 1st}}{\text{Total number of queries}}
    $$
3.  **Symbol Explanation**: No additional variables, it is a count-based ratio.
## 5.3. Baselines
The paper compares against representative state-of-the-art baselines:
### Zero-shot Baselines
CLIP (ViT-L, ViT-BigG/14), OpenCLIP (ViT-L), MagicLens (ViT-L), SigLIP (So/14), BLIP2 (ViT-L), EVA-CLIP, E5-V (Phi3.5-V, LLaVA-1.6). These represent traditional CLIP-family models and early MLLM-based embedding models evaluated without fine-tuning on MMEB.
### Fine-tuned Baselines
CLIP (ViT-L) fine-tuned on MMEB, VLM2Vec (Qwen2-VL 2B/7B), LLaVE (LLaVA-OV 7B), QQMM (LLaVA-OV 7B), UniME (Qwen2-VL 2B/7B, LLaVA-OV 7B). These are the latest MLLM-based embedding models fine-tuned on MMEB, representing the prior state-of-the-art.
### Reranker Baseline
LamRA (7B), a state-of-the-art multimodal reranker, to compare the performance of UniME-V2-Reranker.
These baselines are representative of the full range of existing approaches, so comparing against them fairly validates the effectiveness of UniME-V2.

---

# 6. Results & Analysis
## 6.1. Core Results Analysis
### MMEB Benchmark Results
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th rowspan="2">#Parameters</th>
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
<tr>
<td># of Datasets →</td>
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
<th colspan="9">Zero-shot on MMEB</th>
</tr>
<tr>
<td>CLIP (ViT-L)(Jiang et al. 2025)</td>
<td>0.4B</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.1</td>
<td>38.7</td>
<td>39.2</td>
</tr>
<tr>
<td>OpenCLIP (ViT-L)(Radford et al. 2021)</td>
<td>0.4B</td>
<td>41.5</td>
<td>6.9</td>
<td>44.6</td>
<td>53.5</td>
<td>32.8</td>
<td>36.0</td>
<td>36.6</td>
</tr>
<tr>
<td>Magiclens (ViT-L)(Zhang et al. 2024b)</td>
<td>0.4B</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td>31.0</td>
<td>23.7</td>
<td>27.1</td>
</tr>
<tr>
<td>SigLIP (So/14)(Zhai et al. 2023)</td>
<td>0.9B</td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>32.3</td>
<td>38.0</td>
<td>35.0</td>
</tr>
<tr>
<td>BLIP2 (ViT-L)(Li et al. 2023)</td>
<td>1.2B</td>
<td>27.0</td>
<td>4.2</td>
<td>33.9</td>
<td>47.0</td>
<td>25.3</td>
<td>25.1</td>
<td>28.0</td>
</tr>
<tr>
<td>CLIP (ViT-BigG/14)(Cherti et al. 2022)</td>
<td>2.5B</td>
<td>52.3</td>
<td>14.0</td>
<td>50.5</td>
<td>60.3</td>
<td>38.9</td>
<td>45.8</td>
<td>44.3</td>
</tr>
<tr>
<td>EVA-CLIP(Sun et al. 2023)</td>
<td>8B</td>
<td>56.0</td>
<td>10.4</td>
<td>49.2</td>
<td>58.9</td>
<td>38.1</td>
<td>45.6</td>
<td>43.7</td>
</tr>
<tr>
<td>E5-V (Phi3.5-V)(Jiang et al. 2024)</td>
<td>4.2B</td>
<td>39.1</td>
<td>9.6</td>
<td>38.0</td>
<td>57.6</td>
<td>33.1</td>
<td>31.9</td>
<td>36.1</td>
</tr>
<tr>
<td>E5-V (LLaVA-1.6)(Jiang et al. 2024)</td>
<td>7B</td>
<td>39.7</td>
<td>10.8</td>
<td>39.4</td>
<td>60.2</td>
<td>34.2</td>
<td>33.4</td>
<td>37.5</td>
</tr>
<tr>
<th colspan="9">Fine-tuning on MMEB</th>
</tr>
<tr>
<td>CLIP (ViT-L)(Jiang et al. 2025)</td>
<td>0.4B</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>47.6</td>
</tr>
<tr>
<td>VLM2Vec (Qwen2-VL)(Jiang et al. 2025)</td>
<td>2B</td>
<td>59.0</td>
<td>49.4</td>
<td>65.4</td>
<td>73.4</td>
<td>66.0</td>
<td>52.6</td>
<td>60.1</td>
</tr>
<tr>
<td>VLM2Vec (Qwen2-VL)(Jiang et al. 2025)</td>
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
<td>LLaVE (LLaVA-OV)(Lan et al. 2025)</td>
<td>7B</td>
<td>65.7</td>
<td>65.4</td>
<td>70.9</td>
<td>91.9</td>
<td>75.0</td>
<td>64.4</td>
<td>70.3</td>
</tr>
<tr>
<td>QQMM (LLaVA-OV)(Xue, Li, and Liu 2025b)</td>
<td>7B</td>
<td>66.8</td>
<td>66.8</td>
<td>70.5</td>
<td>90.4</td>
<td>74.7</td>
<td>65.6</td>
<td>70.7</td>
</tr>
<tr>
<td>UniME (Qwen2-VL)(Gu et al. 2025a)</td>
<td>2B</td>
<td>59.0</td>
<td>53.4</td>
<td>64.9</td>
<td>69.6</td>
<td>65.5</td>
<td>54.6</td>
<td>60.6</td>
</tr>
<tr>
<td>UniME (Qwen2-VL)(Gu et al. 2025a)</td>
<td>7B</td>
<td>64.7</td>
<td>59.0</td>
<td>71.6</td>
<td>82.7</td>
<td>72.2</td>
<td>61.4</td>
<td>67.4</td>
</tr>
<tr>
<td>UniME (LLaVA-OV)(Gu et al. 2025a)</td>
<td>7B</td>
<td>66.8</td>
<td>66.6</td>
<td>70.5</td>
<td>90.9</td>
<td>74.6</td>
<td>65.8</td>
<td>70.7</td>
</tr>
<tr>
<td>UniME-V2(Qwen2-VL)</td>
<td>2B</td>
<td>62.1(+3.1)</td>
<td>56.3(+2.9)</td>
<td>68.0(+3.1)</td>
<td>72.7(+3.1)</td>
<td>67.4(+1.9)</td>
<td>58.9(+4.3)</td>
<td>63.6(+3.0)</td>
</tr>
<tr>
<td>UniME-V2(Qwen2-VL)</td>
<td>7B</td>
<td>64.0(-0.7)</td>
<td>60.1(+1.1)</td>
<td>73.1(+1.5)</td>
<td>82.8(+0.1)</td>
<td>72.0(-0.2)</td>
<td>63.0(+1.6)</td>
<td>68.0(+0.6)</td>
</tr>
<tr>
<td>UniME-V2(LLaVA-OV)</td>
<td>7B</td>
<td>65.3(-1.5)</td>
<td>67.6(+1.0)</td>
<td>72.9(+2.4)</td>
<td>90.2(-0.7)</td>
<td>74.8(+0.2)</td>
<td>66.7(+0.9)</td>
<td>71.2(+0.5)</td>
</tr>
</thead>
</table>

#### Analysis
- UniME-V2 consistently outperforms prior models across all base backbones: 3.0% improvement over VLM2Vec Qwen2-VL 2B, 2.2% over VLM2Vec Qwen2-VL 7B, and 0.5% improvement over prior SOTA (QQMM, UniME) on LLaVA-OV 7B.
- UniME-V2 achieves particularly strong performance on out-of-distribution (OOD) datasets (66.7 score), outperforming all prior models, demonstrating better transferability to unseen tasks and domains.
- Performance scales with model size, as expected, with the 7B LLaVA-OV backbone achieving the highest overall score.
  ---
### Cross-Modal Retrieval Results
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="3">Models</th>
<th rowspan="3">#Parameters</th>
<th colspan="4">Short Caption</th>
<th colspan="4">Long Caption</th>
<th colspan="3">Compositional</th>
</tr>
<tr>
<th colspan="2">Flickr30K</th>
<th colspan="2">COCO</th>
<th colspan="2">ShareGPT4V</th>
<th colspan="2">Urban1K</th>
<th colspan="3">SugarCrepe</th>
</tr>
<tr>
<th>qt→ci</th>
<th>qi→ct</th>
<th>qt→ci</th>
<th>qi→ct</th>
<th>qt→ci</th>
<th>qi→ct</th>
<th>qt→ci</th>
<th>qi→ct</th>
<th>Replace</th>
<th>Swap</th>
<th>Add</th>
</tr>
<tr>
<td>OpenCLIP (ViT-L) (Radford et al. 2021)</td>
<td>0.4B</td>
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
<td>CLIP (ViT-BigG/14) (Cherti et al. 2022)</td>
<td>2.5B</td>
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
<td>EVA-CLIP (Sun et al. 2023)</td>
<td>8B</td>
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
<td>E5-V (Phi3.5-V) (Jiang et al. 2024)</td>
<td>4.2B</td>
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
<td>E5-V (LLaVA-1.6) (Jiang et al. 2024)</td>
<td>7B</td>
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
<td>VLM2Vec (Qwen2-VL) (Jiang et al. 2025)</td>
<td>2B</td>
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
<td>VLM2Vec (Qwen2-VL) (Jiang et al. 2025)</td>
<td>7B</td>
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
<td>UniME (Qwen2-VL) (Gu et al. 2025a)</td>
<td>2B</td>
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
<td>UniME (Qwen2-VL) (Gu et al. 2025a)</td>
<td>7B</td>
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
<td>UniME (LLaVA-OV) (Gu et al. 2025a)</td>
<td>7B</td>
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
<td>UniME-V2 (Qwen2-VL)</td>
<td>2B</td>
<td>79.8(+4.9)</td>
<td>90.7(+0.1)</td>
<td>55.1(+11.1)</td>
<td>65.4(+1.9)</td>
<td>94.0(+10.4)</td>
<td>95.6(+7.0)</td>
<td>95.9(+12.6)</td>
<td>94.0(+10.8)</td>
<td>70.9(+5.3)</td>
<td>51.2(+6.0)</td>
<td>70.2(+4.5)</td>
</tr>
<tr>
<td>UniME-V2 (Qwen2-VL)</td>
<td>7B</td>
<td>83.6(+2.8)</td>
<td>93.5(+0.8)</td>
<td>57.3(+6.4)</td>
<td>70.3(+0.5)</td>
<td>94.3(+7.8)</td>
<td>95.1(+1.3)</td>
<td>97.2(+1.9)</td>
<td>96.3(+2.3)</td>
<td>77.8(+9.0)</td>
<td>62.2(+9.2)</td>
<td>79.0(+9.2)</td>
</tr>
<tr>
<td>UniME-V2 (LLaVA-OV)</td>
<td>7B</td>
<td>86.2(+2.9)</td>
<td>95.3(-0.7)</td>
<td>60.9(+6.1)</td>
<td>74.1(+0.1)</td>
<td>95.1(+1.2)</td>
<td>94.1(+4.8)</td>
<td>96.3(+2.0)</td>
<td>96.7(+1.2)</td>
<td>88.6(+8.1)</td>
<td>73.7(+8.2)</td>
<td>90.5(+8.3)</td>
</tr>
</thead>
</table>

#### Analysis
- **Short Caption Retrieval**: UniME-V2 achieves 2.2%-9.7% improvement over UniME on image-to-text retrieval. Text-to-image retrieval performance is comparable to UniME, which the authors attribute to limited text-to-image training data in MMEB and limited semantic information in short captions.
- **Long Caption Retrieval**: UniME-V2 achieves significant improvements on both ShareGPT4V and Urban1K, as long captions contain richer semantic information, and the model's improved discriminative ability is better suited for these tasks.
- **Compositional Retrieval (SugarCrepe)**: UniME-V2 delivers consistent improvements across all three compositional metrics (Replace, Swap, Add), with up to 9.2% improvement over UniME. This validates that the hard negative mining strategy effectively improves the model's ability to distinguish fine-grained compositional differences.
  The following figure (Figure 4 from the original paper) shows that UniME-V2 has a much smaller modality gap between text and image embeddings compared to EVA-CLIP, which contributes to its better retrieval performance:

  ![Figure 4: Comparison of representation distributions between EVA-CLIP-8B and UniME-V2 (LLaVA-OneVision-7B).](images/4.jpg)
  *该图像是图表，比较了EVA-CLIP与UniME-V2模型的表示分布。左侧为EVA-CLIP，右侧为UniME-V2，分别用不同颜色的点表示文本（蓝色）和图像（橙色）的分布情况，展示了两者在语义空间中的差异。*

---
### Reranking Results
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Embedding Model</th>
<th>Reranker</th>
<th>#Data</th>
<th>MMEB</th>
<th>RShort</th>
<th>RLong</th>
<th>RCompos</th>
</tr>
</thead>
<tbody>
<tr>
<td>UniME(2B)</td>
<td>-</td>
<td>-</td>
<td>60.6</td>
<td>68.3</td>
<td>84.7</td>
<td>58.8</td>
</tr>
<tr>
<td>UniME-V2(2B)</td>
<td>-</td>
<td>-</td>
<td>63.6</td>
<td>72.1</td>
<td>93.4</td>
<td>64.1</td>
</tr>
<tr>
<td>UniME-V2(2B)</td>
<td>LamRA(7B)</td>
<td>1.1M</td>
<td>67.3</td>
<td>76.4</td>
<td>96.4</td>
<td>87.4</td>
</tr>
<tr>
<td>UniME-V2(2B)</td>
<td>UniME-V2(7B)</td>
<td>0.6M</td>
<td>67.6</td>
<td>76.4</td>
<td>96.9</td>
<td>94.8</td>
</tr>
<tr>
<td>UniME(7B)</td>
<td>-</td>
<td>-</td>
<td>67.4</td>
<td>73.6</td>
<td>92.4</td>
<td>63.9</td>
</tr>
<tr>
<td>UniME-V2(7B)</td>
<td>-</td>
<td>-</td>
<td>68.0</td>
<td>76.4</td>
<td>95.8</td>
<td>73.0</td>
</tr>
<tr>
<td>UniME-V2(7B)</td>
<td>LamRA(7B)</td>
<td>1.1M</td>
<td>69.1</td>
<td>78.3</td>
<td>97.2</td>
<td>87.4</td>
</tr>
<tr>
<td>UniME-V2(7B)</td>
<td>UniME-V2(7B)</td>
<td>0.6M</td>
<td>69.6</td>
<td>78.7</td>
<td>97.5</td>
<td>94.8</td>
</tr>
</tbody>
</table>

#### Analysis
- UniME-V2-Reranker outperforms the SOTA LamRA reranker across all tasks, while using only half the training data (0.6M vs 1.1M samples).
- The largest improvement is on compositional retrieval (7.4% higher than LamRA), as the reranker is trained on high-quality hard negatives mined by the MLLM-as-a-Judge pipeline, making it much better at distinguishing hard compositional samples.
## 6.2. Ablation Studies / Parameter Analysis
### Ablation on Core Components
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>Hard Negatives</th>
<th>Soft Score</th>
<th>MMEB</th>
<th>RShort</th>
<th>RLong</th>
<th>RCompos</th>
</tr>
</thead>
<tbody>
<tr>
<td>✗</td>
<td>✗</td>
<td>60.1</td>
<td>63.4</td>
<td>82.2</td>
<td>60.0</td>
</tr>
<tr>
<td>✓</td>
<td>✗</td>
<td>61.6</td>
<td>68.9</td>
<td>89.8</td>
<td>63.7</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>63.6</td>
<td>72.1</td>
<td>93.4</td>
<td>64.1</td>
</tr>
</tbody>
</table>

#### Analysis
- Adding the MLLM-as-a-Judge hard negative mining alone improves performance by 1.5% on MMEB, 5.5% on short retrieval, 7.6% on long retrieval, and 3.7% on compositional retrieval, compared to the baseline (no hard negatives, no soft scores, equivalent to VLM2Vec).
- Adding the soft score distribution alignment further improves performance by 2.0% on MMEB, 3.2% on short retrieval, 3.6% on long retrieval, and 0.4% on compositional retrieval, demonstrating that both core components contribute significantly to the performance gains.
### Ablation on Different MLLM Judges
The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th>Judge Model</th>
<th>MMEB</th>
<th>RShort</th>
<th>RLong</th>
<th>RCompos</th>
</tr>
</thead>
<tbody>
<tr>
<td>Qwen2.5VL-7B</td>
<td>63.6</td>
<td>72.1</td>
<td>93.4</td>
<td>64.1</td>
</tr>
<tr>
<td>InternVL3-8B</td>
<td>58.5</td>
<td>70.2</td>
<td>91.3</td>
<td>64.1</td>
</tr>
<tr>
<td>InternVL3-14B</td>
<td>63.2</td>
<td>72.9</td>
<td>93.1</td>
<td>63.2</td>
</tr>
</tbody>
</table>

#### Analysis
The final model performance is highly dependent on the quality of the MLLM judge's semantic matching scores. Qwen2.5VL-7B delivers the best performance, while InternVL3-8B performs significantly worse. InternVL3-14B is close but slightly worse than Qwen2.5VL-7B, which the authors attribute to differences in the instruction tuning data distribution used for the MLLMs.
### Ablation on Number of Hard Negatives
The following are the results from Table 6 of the original paper:

<table>
<thead>
<tr>
<th>#Negatives</th>
<th>MMEB</th>
<th>RShort</th>
<th>RLong</th>
<th>RCompos</th>
</tr>
</thead>
<tbody>
<tr>
<td>4</td>
<td>61.3</td>
<td>69.2</td>
<td>91.0</td>
<td>62.4</td>
</tr>
<tr>
<td>6</td>
<td>61.8</td>
<td>70.8</td>
<td>91.7</td>
<td>61.2</td>
</tr>
<tr>
<td>8</td>
<td>63.6</td>
<td>72.1</td>
<td>93.4</td>
<td>64.1</td>
</tr>
<tr>
<td>10</td>
<td>63.0</td>
<td>72.0</td>
<td>93.4</td>
<td>63.4</td>
</tr>
</tbody>
</table>

#### Analysis
Performance increases as the number of hard negatives increases from 4 to 8, as more diverse hard negatives help the model learn better discriminative ability. Performance drops slightly when using 10 negatives, as easier negatives are included which reduce the effectiveness of discriminative learning. 8 is the optimal number of hard negatives per query.
### Ablation on Temperature
The following are the results from Table 10 of the original paper:

<table>
<thead>
<tr>
<th>Temperature</th>
<th>MMEB</th>
<th>RShort</th>
<th>RLong</th>
<th>RCompos</th>
</tr>
</thead>
<tbody>
<tr>
<td>0.03</td>
<td>61.9</td>
<td>70.6</td>
<td>92.0</td>
<td>65.8</td>
</tr>
<tr>
<td>0.02</td>
<td>63.6</td>
<td>72.1</td>
<td>93.4</td>
<td>64.1</td>
</tr>
<tr>
<td>0.01</td>
<td>62.1</td>
<td>70.1</td>
<td>91.1</td>
<td>66.5</td>
</tr>
</tbody>
</table>

#### Analysis
A temperature value of 0.02 delivers the best overall performance across all tasks, so it is selected as the optimal hyperparameter.
### Qualitative Results
The following figure (Figure 5 from the original paper) shows qualitative examples of retrieval and reranking results:

![Figure 5: Qualitative examples. We present the retrieval and reranking results of our method across different tasks.](images/5.jpg)
*该图像是示意图，展示了不同任务中我们方法的检索和重排序结果。图中包含分类、视觉问答和视觉定位等任务的示例，以及对应的指示和语义标签。*

The following figure (Figure 7 from the original paper) shows additional qualitative examples:

![Figure 7: Qualitative examples. We present the additional retrieval and reranking results of our method across different tasks.](images/7.jpg)
*该图像是示意图，展示了多模态嵌入学习中的不同任务示例，包括分类、视觉问答和视觉定位。每个任务都有相应的指导说明和示例答案，体现了如何通过示例图像得到分类结果、回答问题以及识别目标对象。*

The following figure (Figure 6 from the original paper) shows examples of mined hard negatives with their corresponding semantic matching scores:

![Figure 6: Qualitative examples. We present examples showing queries and their corresponding hard negative candidates processed after our hard negative mining pipeline.](images/6.jpg)
*该图像是插图，展示了查询和对应的经过我们硬负样本挖掘流程处理后的候选图像。在每个查询下，列出了与之匹配的候选图像及其相似度分数，展示了不同场景的多模态嵌入学习实例。*

---

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper proposes UniME-V2, a state-of-the-art universal multimodal embedding model that leverages MLLMs as automated judges to improve representation learning. The core contributions and findings are:
1.  The **MLLM-as-a-Judge** pipeline for hard negative mining effectively addresses the limitations of in-batch mining, delivering diverse, high-quality hard negatives with low false negative rates.
2.  The distribution alignment training framework, which uses MLLM-derived soft semantic matching scores as soft labels, relaxes rigid one-to-one mapping constraints and significantly improves the model's discriminative ability to capture fine-grained semantic differences.
3.  The **UniME-V2-Reranker**, trained on mined hard negatives with joint pairwise and listwise optimization, further boosts retrieval performance, especially on hard compositional tasks, while using only half the training data of competing rerankers.
4.  Extensive experiments demonstrate that UniME-V2 achieves SOTA average performance across all 36 tasks in the MMEB benchmark, as well as on short/long caption retrieval and compositional retrieval tasks, with particularly strong performance on out-of-distribution datasets, showing excellent transferability.
## 7.2. Limitations & Future Work
### Limitations (Inferred from the paper)
1.  **Computational Cost**: The MLLM-as-a-Judge step adds significant computational overhead during training, as it requires running a large MLLM to score millions of query-candidate pairs in the potential hard negative set.
2.  **Judge Dependence**: The final model performance is highly dependent on the quality of the MLLM judge. If the judge has biases or poor understanding of certain domains, the mined negatives and soft scores will be low quality, hurting the final embedding model performance.
3.  **Two-Stage Inference Latency**: The inference pipeline has two stages (embedding retrieval + reranking), which adds latency compared to single-stage embedding-only retrieval, which may be a limitation for low-latency real-world applications.
### Future Research Directions
1.  Optimize the MLLM-as-a-Judge pipeline to reduce computational cost, e.g., use a smaller distilled MLLM as the judge, or develop online hard negative mining methods that do not require pre-scoring all candidate pairs.
2.  Explore dynamic thresholds for false negative filtering that adapt to different queries, datasets, and tasks, instead of using a fixed threshold.
3.  Extend the framework to support additional modalities, such as video, audio, and 3D data, to build even more general universal embedding models.
4.  Improve inference efficiency by developing end-to-end models that combine embedding and reranking capabilities, or by distilling the reranker's knowledge into the embedding model to reduce reliance on a separate reranking stage.
## 7.3. Personal Insights & Critique
### Key Inspirations
1.  The **strong-model-as-supervisor** paradigm demonstrated in this paper is highly generalizable: using stronger, more capable models (like MLLMs) as supervisors to train smaller, more efficient models for specific tasks can be applied to many other domains beyond multimodal embedding, including text retrieval, recommendation systems, and computer vision tasks.
2.  Using continuous soft labels from strong models instead of rigid binary labels is a very effective way to capture subtle semantic differences, which is especially valuable for tasks where semantic similarity is a continuous rather than binary property.
3.  The hard negative mining pipeline (global retrieval + MLLM filtering) solves two common pain points in contrastive learning: negative sample diversity and false negative rate, and can be easily transferred to other contrastive learning tasks.
### Potential Areas for Improvement
1.  The paper does not provide a detailed analysis of the computational cost of the MLLM-as-a-Judge step, which is a critical factor for real-world adoption. For large training datasets with millions of samples, scoring all query-candidate pairs with a 7B MLLM can be prohibitively expensive.
2.  The fixed threshold for false negative filtering may not be optimal for all tasks and datasets. A dynamic threshold that adapts to the query's semantic complexity or the dataset's characteristics would likely deliver better performance.
3.  The paper uses a separate reranker module, which adds inference latency. It would be interesting to explore whether the embedding model can be optimized to achieve comparable performance without a separate reranker, or whether a much smaller reranker can deliver similar gains.
### Transferability to Other Domains
- The MLLM-as-a-Judge pipeline can be easily transferred to other contrastive learning tasks, such as text retrieval, video retrieval, and recommendation systems, where hard negative mining is a critical challenge.
- The soft label distribution alignment method can be applied to any supervised learning task where a strong teacher model is available to generate continuous quality scores for samples, not just embedding learning.