# 1. Bibliographic Information
## 1.1. Title
The title of the paper is *MegaPairs: Massive Data Synthesis For Universal Multimodal Retrieval*. Its central topic is the development of a novel low-cost, scalable data synthesis pipeline (named MegaPairs) for generating high-quality large-scale instruction tuning datasets for universal multimodal retrieval, along with a series of state-of-the-art multimodal retrievers (named MMRet) trained on the synthetic dataset.
## 1.2. Authors
The authors and their affiliations are as follows:
- Junjie Zhou: Beijing University of Posts and Telecommunications
- Zheng Liu (corresponding author): Beijing Academy of Artificial Intelligence, The Hong Kong Polytechnic University
- Ze Liu: University of Science and Technology of China
- Shitao Xiao: Beijing Academy of Artificial Intelligence
- Yueze Wang: Beijing Academy of Artificial Intelligence
- Bo Zhao: Beijing Academy of Artificial Intelligence, Shanghai Jiaotong University
- Chen Jason Zhang: The Hong Kong Polytechnic University
- Defu Lian: University of Science and Technology of China
- Yongping Xiong: Beijing University of Posts and Telecommunications
  All authors are affiliated with leading Chinese universities and research institutions focused on artificial intelligence, information retrieval, and multimodal learning.
## 1.3. Journal/Conference
As of the published date (2024-12-19), the paper is released as a preprint on arXiv, the leading open-access preprint server for computer science and artificial intelligence research. It has not yet undergone formal peer review or been published in a peer-reviewed conference or journal.
## 1.4. Publication Year
2024
## 1.5. Abstract
### Research Objective
To address the critical scarcity of large-scale, high-quality, diverse training data that limits progress in universal multimodal retrieval.
### Core Methodology
The MegaPairs pipeline mines correlated image pairs from open-domain image corpora using three distinct similarity models to capture diverse visual and semantic relationships. It then uses open-source vision-language models (VLMs) and large language models (LLMs) to automatically generate natural retrieval instructions for these pairs, producing a large synthetic training dataset.
### Main Results
- 26 million high-quality training instances are generated in the first stage.
- Models trained on just 0.5 million MegaPairs instances outperform baseline models trained on 70 times more data from existing datasets.
- The proposed MMRet models achieve state-of-the-art (SOTA) zero-shot performance across 4 popular composed image retrieval (CIR) benchmarks, and the highest overall score on the 36-dataset Massive Multimodal Embedding Benchmark (MMEB).
- The models also show significant performance gains after downstream fine-tuning, with strong generalization to out-of-distribution tasks.
### Key Conclusions
MegaPairs is a low-cost, highly scalable pipeline that produces far more data-efficient training instances than existing data collection methods. The publicly released dataset, models, and synthesis pipeline will accelerate future research in multimodal retrieval.
## 1.6. Original Source Link
- Official preprint link: https://arxiv.org/abs/2412.14475
- PDF link: https://arxiv.org/pdf/2412.14475v1
- Publication status: Preprint (not formally peer-reviewed or published in a conference/journal as of the provided date)

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
The progress of universal multimodal retrieval systems is severely constrained by the lack of large, diverse, high-quality, and publicly available instruction tuning datasets.
### Importance of the Problem
Multimodal retrieval is a foundational technology for real-world applications including image search, visual question answering (VQA), and multimodal retrieval-augmented generation (RAG). Improving universal multimodal retrievers will directly advance performance across all these use cases.
### Gaps in Prior Research
Existing data synthesis methods for multimodal retrieval (most notably MagicLens) have critical limitations:
1. **Low scalability**: They rely on co-occurring images from the same webpage, but only a small fraction of webpages contain multiple relevant images.
2. **Low quality**: Many co-occurring web images are unrelated or near-duplicates.
3. **Low diversity**: Correlated web images often have monotonous relationships (e.g., different angles of the same object).
4. **Poor accessibility**: Most large multimodal retrieval datasets are held privately by research labs and not released to the public.
### Innovative Entry Point
Instead of relying on scarce natural co-occurring image pairs, the authors mine diverse correlated image pairs from generic open-domain image corpora using three different similarity models to capture heterogeneous relationships. They then use open-source VLMs and LLMs to automatically generate high-quality retrieval instructions for these pairs, creating a fully open, low-cost, scalable pipeline.
## 2.2. Main Contributions / Findings
### Primary Contributions
1. **Novel data synthesis pipeline**: The MegaPairs pipeline for generating large-scale multimodal retrieval instruction data.
2. **Large public dataset**: A 26 million instance synthetic training dataset generated using the MegaPairs pipeline, released publicly to the research community.
3. **SOTA multimodal retrievers**: The MMRet series of models (CLIP-based base/large, MLLM-based) trained on MegaPairs, which outperform all existing baselines on standard benchmarks.
4. **Extensive empirical validation**: Rigorous experiments demonstrating the data quality, scalability, and effectiveness of MegaPairs and MMRet.
### Key Findings
1. MegaPairs data is 70 times more efficient than existing datasets: 0.5 million MegaPairs instances are sufficient to outperform models trained on 36.7 million instances from the state-of-the-art MagicLens dataset.
2. MMRet achieves SOTA zero-shot performance on 4 CIR benchmarks and the 36-dataset MMEB benchmark, outperforming even much larger models and proprietary baselines.
3. Pre-training on MegaPairs significantly improves downstream fine-tuning performance and generalization to out-of-distribution tasks, with MMRet outperforming direct fine-tuning of the same backbone by 9.1% on MMEB.
4. Combining all three similarity models (visual-semantic, visual-pattern, caption) and including mined hard negatives yields the best overall model performance.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
Below are beginner-friendly explanations of core concepts required to understand the paper:
- **Multimodal Retrieval**: A task where a system retrieves relevant information (text, images, or multimodal documents) in response to a query that can consist of text, images, or a combination of both. For example, a query like *"find an image like this cat photo but with the cat sitting on a couch"* combines an image and text.
- **Composed Image Retrieval (CIR)**: A subtype of multimodal retrieval where the query is a reference image plus a text instruction describing how to modify the reference image to get the desired target image.
- **Vision-Language Model (VLM)**: An AI model trained on both image and text data that can understand and process content across both modalities. Examples include CLIP and LLaVA.
- **Instruction Tuning**: A training technique where a pre-trained model is fine-tuned on a diverse set of task instructions to improve its ability to generalize to unseen tasks, originally developed for large language models (LLMs) and later adapted for embedding and multimodal models.
- **Contrastive Learning**: A learning paradigm that trains models to map semantically similar (positive) samples to nearby points in a shared embedding space, and semantically dissimilar (negative) samples to distant points.
- **InfoNCE Loss**: The standard loss function used in contrastive learning, which penalizes models for ranking negative samples higher than positive samples for a given query.
- **Hard Negatives**: In contrastive learning, negative samples that are semantically close to the positive sample, which makes the learning task more challenging and improves the model's discriminative ability.
- **mAP@k (mean Average Precision at top k)**: A retrieval metric that calculates the average precision of relevant results in the top k positions, averaged across all queries. It accounts for both the number of relevant results and their rank in the result list.
- **R@k (Recall at top k)**: A retrieval metric that measures the percentage of queries for which at least one relevant result appears in the top k retrieved items.
## 3.2. Previous Works
Key prior studies relevant to this paper:
1. **CLIP (Radford et al., 2021)**: A dual-encoder VLM with separate image and text encoders, trained on 400 million image-text pairs with contrastive loss to align image and text embeddings in the same shared space. It forms the foundation for most modern multimodal retrieval models, but its training on single image-text matching tasks leaves it poorly suited for CIR and multimodal document retrieval.
2. **MagicLens (Zhang et al., 2024)**: The closest prior work to this paper. It generates CIR instruction data by mining pairs of co-occurring images from the same webpage, then generating instructions describing their relationship. It achieves strong performance but suffers from the scalability, quality, diversity, and accessibility gaps noted earlier, and its dataset is private.
3. **UniIR (Wei et al., 2024)**: An instruction-tuned universal multimodal retriever, but it is trained on the M-BEIR dataset which includes 10 of the 12 retrieval tasks in MMEB, so it does not qualify as a zero-shot model for MMEB evaluation.
4. **VLM2Vec (Jiang et al., 2024b)**: A method for fine-tuning VLMs for multimodal embedding tasks, used as a baseline in the paper's MMEB fine-tuning experiments.
## 3.3. Technological Evolution
The evolution of multimodal retrieval technology follows this timeline:
1. **Early 2010s**: Early multimodal retrieval systems use separate unimodal encoders for text and images with late fusion, leading to limited performance.
2. **2021**: The release of CLIP enables aligned cross-modal embeddings, making cross-modal text-to-image retrieval practical, but the model fails at composed multimodal queries.
3. **2022-2023**: Instruction tuning is adapted from LLMs to text embedding models, then to multimodal models, but progress is limited by the lack of large, diverse instruction tuning datasets for multimodal retrieval.
4. **2024**: MagicLens proposes synthetic generation of CIR instruction data from web co-occurring images, but has critical limitations. This paper's MegaPairs pipeline addresses these limitations, providing the first open, scalable, high-quality synthetic data source for universal multimodal retrieval.
## 3.4. Differentiation Analysis
Compared to the closest prior work MagicLens, MegaPairs has four core advantages:
1. **Higher scalability**: It can mine pairs from any open-domain image corpus, rather than relying on scarce webpages with multiple images.
2. **Higher diversity**: It uses three distinct similarity metrics to capture heterogeneous relationships between image pairs, rather than relying on web co-occurrence which produces limited, monotonous relationships.
3. **Better accessibility**: The entire pipeline uses only open-source models, and the full dataset and code are released publicly, while MagicLens' dataset is private.
4. **Higher data efficiency**: MegaPairs data is 70 times more effective than MagicLens data, with 0.5 million instances outperforming 36.7 million MagicLens instances.

# 4. Methodology
## 4.1. Principles
### Core Idea
Instead of relying on scarce, low-quality naturally occurring multimodal training data, mine diverse correlated image pairs from large open-domain image corpora using multiple pre-trained similarity models, then automatically generate high-quality retrieval instructions for these pairs using open-source VLMs and LLMs. This creates a low-cost, scalable pipeline for producing large volumes of data-efficient training instances for universal multimodal retrievers.
### Intuition
Large open-domain image corpora contain abundant correlated images with diverse visual and semantic relationships, which can be mined at low cost using existing pre-trained models. Open-source VLMs and LLMs can accurately describe these relationships and convert them into natural, diverse retrieval instructions suitable for training multimodal retrievers.
## 4.2. Core Methodology In-depth
The methodology consists of three core components: 1) MegaPairs dataset construction, 2) MMRet model architecture, 3) multimodal contrastive training objective.
### 4.2.1. MegaPairs Dataset Construction
Each training instance in MegaPairs is a triplet: $( \mathcal{T}_q, \mathcal{T}_{qt}, \mathcal{T}_t )$, where $\mathcal{T}_q$ is the query image, $\mathcal{T}_{qt}$ is the text instruction describing how to transform $\mathcal{T}_q$ into the target, and $\mathcal{T}_t$ is the target image. The construction pipeline has two steps:
#### Step 1: Mining Correlated Image Pairs
The input is a large open-domain corpus of captioned images. For each query image $(\mathcal{T}_q, \mathcal{C}_q)$ (where $\mathcal{C}_q$ is the caption of $\mathcal{T}_q$), three different similarity models are used to retrieve diverse correlated target images:
1. **Visual-semantic correlation**: Uses the EVA-CLIP image encoder to measure semantic similarity between two images, regardless of visual appearance (e.g., two different views of the same car have high visual-semantic similarity).
2. **Visual-pattern correlation**: Uses the DINOv2 image encoder to measure visual similarity between two images, regardless of semantic meaning (e.g., a cat and a dog with similar fur color and background have high visual-pattern similarity).
3. **Caption correlation**: Uses the EVA-CLIP text encoder to measure textual similarity between the captions of two images.

   Retrieved pairs are filtered to keep only those with similarity scores between 0.8 and 0.96, eliminating both weakly associated pairs (too low similarity) and near-duplicates (too high similarity). For each positive pair $(\mathcal{T}_q, \mathcal{T}_{t_i})$, other retrieved images (not the positive target) are kept as hard negative samples to improve the model's discriminative ability.
#### Step 2: Generating Open-Ended Instructions
For each mined image pair $(\mathcal{T}_q, \mathcal{T}_{t_i})$:
1. First, both images are fed to an open-source Multimodal Large Language Model (MLLM, InternVL2-26B in this work) to generate a detailed description $\mathcal{D}_i$ of the common concepts and differences between $\mathcal{T}_q$ and $\mathcal{T}_{t_i}$. The prompt for the MLLM asks for a 60-100 word description of similarities and differences, as shown below:

   ![Figure 3: The specific prompts for MLLM. The value of WORD_NUM ranges from 60 to 100 in our practical data generation to enhance the diversity of the generated description.](images/3.jpg)
   *该图像是插图，展示了观察源图像和目标图像之间的连接与差异的具体提示。两个图像都展示了关键共性。然而，源图像包含特定的细节，而目标图像则展示其他不同的特征，旨在引导比较分析。*

2. Next, the description $\mathcal{D}_i$ is fed to an open-source LLM (Llama-3-8B in this work) to generate at least 3 natural, diverse retrieval instructions $\mathcal{T}_{qt_i}$. The LLM is prompted to avoid revealing explicit details of the query image (using non-specific pronouns instead) and to clearly describe the unique features of the target image, with multiple examples of high-quality instructions provided in the prompt.

   The final output is a training triplet $(\mathcal{T}_q, \mathcal{T}_{qt_i}, \mathcal{T}_{t_i})$ plus associated hard negatives. In this work, 26.2 million such instances are generated from a 20 million image subset of the Recap-DataComp-1B dataset, with 3 instructions per pair and 5 hard negatives per pair. Examples of MegaPairs instances are shown below:

   ![Figure 5: The visualized examples of MegaPairs. Each row represents a single example, with the query item highlighted in a blue rectangle and the target items enclosed within a dashed box.](images/4.jpg)
   *该图像是一个示意图，展示了 MegaPairs 的视觉化示例。每一行代表一个单独的示例，查询项用蓝色矩形突出显示，目标项则被虚线框围住。*

### 4.2.2. MMRet Model Architecture
Two types of MMRet models are developed, based on different VLM backbones:
#### CLIP-based MMRet
Built on the standard CLIP dual-encoder architecture, with separate image encoder $\Phi_I$ and text encoder $\Phi_T$:
- For a single image $I$, its embedding is computed as:
  $$\mathbf{e}_i = \Phi_I(I)$$
- For a single text $T$, its embedding is computed as:
  $$\mathbf{e}_t = \Phi_T(T)$$
- For a composed multimodal query (image + text, e.g., a CIR query), its embedding is computed via element-wise addition of the image and text embeddings (following the score-fusion strategy from UniIR):
  $$\mathbf{e}_{it} = \Phi_I(I) + \Phi_T(T)$$
Both base (CLIP-B) and large (CLIP-L) CLIP variants are used to create MMRet-Base and MMRet-Large respectively.
#### MLLM-based MMRet
Built on the LLaVA-1.6 (Mistral 7B) multimodal large language model, which integrates a visual encoder into a LLM to process interleaved image and text tokens:
- The input to the model is structured as: $instruct {task_inst} {query {q_t} {q_i}} [EOS]$, where `{task_inst}` is the task-specific instruction, ${q_t}$ is the query text, ${q_i}$ is the query image, and `[EOS]` is the end-of-sequence token.
- The embedding of the input is the normalized last hidden state of the `[EOS]` token, following standard practice for LLM-based embedding models. This variant is named MMRet-MLLM.
### 4.2.3. Multimodal Contrastive Learning
The standard InfoNCE loss is used as the training objective for all MMRet variants:
$$
\mathcal{L} = - \frac{1}{|\mathcal{Q}|} \sum_{q_i \in \mathcal{Q}} \log \frac{\exp(\mathbf{e}_{q_i} \cdot \mathbf{e}_{c_i^+} / \tau)}{\sum_{c_j \in \mathcal{C}} \exp(\mathbf{e}_{q_i} \cdot \mathbf{e}_{c_j} / \tau)}
$$
Symbol explanation:
- $\mathcal{Q}$: The set of all query samples in a training batch
- $q_i$: A single query sample, which can be an image, text, or composed image-text query
- $\mathbf{e}_{q_i}$: The embedding of query $q_i$ output by the MMRet model
- $c_i^+$: The positive candidate sample corresponding to query $q_i$ (the target the query should retrieve)
- $\mathbf{e}_{c_i^+}$: The embedding of the positive candidate $c_i^+$
- $\mathcal{C}$: The set of all candidate samples in the batch, including both positive and negative candidates
- $\tau$: Temperature parameter that modulates the penalty on negative samples, set to 0.02 in this work
- The dot product $\mathbf{e}_a \cdot \mathbf{e}_b$ measures cosine similarity between the normalized embeddings $\mathbf{e}_a$ and $\mathbf{e}_b$

  Intuition: The loss trains the model to make the similarity between a query and its positive candidate much higher than the similarity between the query and all negative candidates in the batch, effectively learning to rank relevant results higher than irrelevant ones.

# 5. Experimental Setup
## 5.1. Datasets
### Evaluation Datasets
Two groups of experiments are conducted, using the following standard benchmarks:
1. **Composed Image Retrieval (CIR) Benchmarks**:
   - **CIRCO**: A challenging zero-shot CIR benchmark with 123,403 candidate natural images, 800 test queries each annotated with multiple ground-truth images. It is the primary benchmark due to its large candidate pool and high-quality annotations.
   - **CIRR**: The first natural image CIR dataset, with 4,148 test queries and 2,315 candidate images, each query having exactly one positive target. It supports both standard and subset retrieval settings.
   - **FashionIQ**: A fashion-domain CIR dataset with 6,016 validation queries and 15,536 fashion product images, covering three subtasks: dress, shirt, and top-tee.
   - **GeneCIS**: A conditional image similarity benchmark with four subtasks (focus attribute, change attribute, focus object, change object), each query having a small candidate subset of average size 13.8 images.
2. **MMEB (Massive Multimodal Embedding Benchmark)**: A comprehensive benchmark with 36 datasets across four meta-task categories: 10 classification, 10 VQA, 12 retrieval, and 4 visual grounding. It includes 20 in-distribution (IND) datasets for training and 16 out-of-distribution (OOD) datasets for evaluation, to test generalization ability.
### Training Dataset
All MMRet models are pre-trained on the 26 million instance MegaPairs synthetic dataset generated from the 20 million captioned image subset of Recap-DataComp-1B.
### Example Data Sample
An example of a MegaPairs instance:
- Query image: A round tufted ottoman, caption: *"Round ottoman, tufted surface"*
- Target image: A car seat with tufted upholstery
- Instruction: *"Find an image of the same tufted texture but on a car seat"*
## 5.2. Evaluation Metrics
### mAP@k (mean Average Precision at top k)
1. **Conceptual Definition**: Measures the overall quality of ranked retrieval results by calculating the average precision for each query (precision averaged over all positions where a relevant result appears) and taking the mean across all queries. It accounts for both the number of relevant results and their rank in the result list.
2. **Mathematical Formula**:
   $$
   mAP@k = \frac{1}{Q} \sum_{q=1}^{Q} AP_q@k
   $$
   Where $AP_q@k$ (Average Precision for query $q$ at top $k$) is:
   $$
   AP_q@k = \frac{1}{R_q} \sum_{i=1}^{k} P(i) \times rel(i)
   $$
3. **Symbol Explanation**:
   - $Q$: Total number of queries
   - $R_q$: Total number of relevant results for query $q$
   - `P(i)`: Precision at position $i$ (number of relevant results in top $i$ positions divided by $i$)
   - `rel(i)`: Indicator function equal to 1 if the result at position $i$ is relevant, 0 otherwise
### R@k (Recall at top k)
1. **Conceptual Definition**: Measures the percentage of queries for which at least one relevant result appears in the top $k$ retrieved items. It focuses on whether the relevant result is found in the top $k$ positions, regardless of its exact rank.
2. **Mathematical Formula**:
   $$
   R@k = \frac{\text{Number of queries with at least one relevant result in top }k}{\text{Total number of queries}}
   $$
### Precision@1
1. **Conceptual Definition**: The percentage of queries where the top-1 retrieved result is the correct relevant candidate. It is a strict metric measuring whether the most relevant result is ranked first, used for MMEB evaluation.
2. **Mathematical Formula**:
   $$
   Precision@1 = \frac{\text{Number of queries where top-1 result is relevant}}{\text{Total number of queries}}
   $$
## 5.3. Baselines
### CIR Experiment Baselines
- CLIP-based baselines: SEARLE, CIReVL, LDRE, MagicLens-B/L, Pic2Word, PLI, CompoDiff, IP-CIR (state-of-the-art CIR models with CLIP backbones of varying sizes)
- MLLM-based baselines: E5-V, MM-Embed (existing multimodal embedding models based on LLaVA-1.6)
### MMEB Experiment Baselines
- Zero-shot baselines: BLIP2, SigLIP, CLIP, OpenCLIP, UniIR, MagicLens, E5-V
- Fine-tuning baselines: CLIP, OpenCLIP, VLM2Vec (LLaVA-1.6 and Phi-3.5-V variants)

# 6. Results & Analysis
## 6.1. Core Results Analysis
### Zero-shot CIR Results
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th rowspan="2">Backbone</th>
<th rowspan="2"># Params</th>
<th>CIRCO</th>
<th colspan="2">CIRR</th>
<th>FashionIQ</th>
<th>GeneCIS</th>
</tr>
<tr>
<th>mAP@5</th>
<th>R@1</th>
<th>R_s@1</th>
<th>R@10</th>
<th>R@1</th>
</tr>
</thead>
<tbody>
<tr>
<td>SEARLE (Baldrati et al., 2023)</td>
<td>CLIP-B</td>
<td>165M</td>
<td>9.4</td>
<td>24.0</td>
<td>54.9</td>
<td>22.9</td>
<td>-</td>
</tr>
<tr>
<td>CIReVL (Karthik et al., 2023)</td>
<td>CLIP-B</td>
<td>12.3B†</td>
<td>14.9</td>
<td>23.9</td>
<td>60.2</td>
<td>28.3</td>
<td>15.9</td>
</tr>
<tr>
<td>LDRE (Yang et al., 2024)</td>
<td>CLIP-B</td>
<td>7.9B†</td>
<td>18.0</td>
<td>25.7</td>
<td>60.5</td>
<td>24.8</td>
<td>-</td>
</tr>
<tr>
<td>MagicLens-B (Zhang et al., 2024)</td>
<td>CLIP-B</td>
<td>166M</td>
<td>23.1</td>
<td>27.0</td>
<td>66.7</td>
<td>26.3</td>
<td>15.0</td>
</tr>
<tr>
<td>MagicLens-B‡ (Zhang et al., 2024)</td>
<td>CoCa-B</td>
<td>267M</td>
<td>30.8</td>
<td>31.6</td>
<td>69.3</td>
<td>35.2</td>
<td>17.4*</td>
</tr>
<tr>
<td>MMRet-Base</td>
<td>CLIP-B</td>
<td>149M</td>
<td>34.3</td>
<td>36.1</td>
<td>71.6</td>
<td>31.9</td>
<td>18.0</td>
</tr>
<tr>
<td>Pic2Word (Saito et al., 2023)</td>
<td>CLIP-L</td>
<td>429M</td>
<td>8.7</td>
<td>23.9</td>
<td>-</td>
<td>24.7</td>
<td>11.2</td>
</tr>
<tr>
<td>PLI (Chen and Lai, 2023)</td>
<td>CLIP-L</td>
<td>428M</td>
<td>10.4</td>
<td>25.5</td>
<td>55.6</td>
<td>35.4</td>
<td>-</td>
</tr>
<tr>
<td>SEARLE (Baldrati et al., 2023)</td>
<td>CLIP-L</td>
<td>442M</td>
<td>11.7</td>
<td>24.2</td>
<td>53.8</td>
<td>25.6</td>
<td>12.3</td>
</tr>
<tr>
<td>CompoDiff (Gu et al., 2024a)</td>
<td>CLIP-L</td>
<td>568M</td>
<td>12.6</td>
<td>18.2</td>
<td>57.4</td>
<td>36.0</td>
<td>14.9</td>
</tr>
<tr>
<td>CIReVL (Karthik et al., 2023)</td>
<td>CLIP-L</td>
<td>12.5B†</td>
<td>18.6</td>
<td>24.6</td>
<td>59.5</td>
<td>28.6</td>
<td>15.9</td>
</tr>
<tr>
<td>LDRE (Yang et al., 2024)</td>
<td>CLIP-L</td>
<td>8.2B†</td>
<td>23.4</td>
<td>26.5</td>
<td>60.4</td>
<td>28.5</td>
<td>-</td>
</tr>
<tr>
<td>MagicLens-L (Zhang et al., 2024)</td>
<td>CLIP-L</td>
<td>465M</td>
<td>29.6</td>
<td>30.1</td>
<td>68.1</td>
<td>30.7</td>
<td>16.3</td>
</tr>
<tr>
<td>MagicLens-L‡ (Zhang et al., 2024)</td>
<td>CoCa-L</td>
<td>613M</td>
<td>34.1*</td>
<td>33.3*</td>
<td>70.9*</td>
<td>38.0</td>
<td>16.7</td>
</tr>
<tr>
<td>MMRet-Large</td>
<td>CLIP-L</td>
<td>428M</td>
<td>39.2</td>
<td>38.0</td>
<td>73.2</td>
<td>34.6</td>
<td>18.1</td>
</tr>
<tr>
<td>LDRE (Yang et al., 2024)</td>
<td>CLIP-G</td>
<td>10.3B†</td>
<td>31.1</td>
<td>36.2</td>
<td>68.8</td>
<td>32.5</td>
<td>-</td>
</tr>
<tr>
<td>CIReVL (Karthik et al., 2023)</td>
<td>CLIP-G</td>
<td>14.6B†</td>
<td>26.8</td>
<td>34.7</td>
<td>68.0</td>
<td>32.2</td>
<td>17.4*</td>
</tr>
<tr>
<td>IP-CIR (Li et al., 2024c)</td>
<td>CLIP-G</td>
<td>43.8B†</td>
<td>32.8</td>
<td>39.3</td>
<td>70.0</td>
<td>45.7*</td>
<td>-</td>
</tr>
<tr>
<td>E5-V (Jiang et al., 2024a)</td>
<td>LLaVA-1.6</td>
<td>8.35B</td>
<td>19.1</td>
<td>33.9</td>
<td>-</td>
<td>31.8</td>
<td>-</td>
</tr>
<tr>
<td>MM-Emded (Lin et al., 2024)</td>
<td>LLaVA-1.6</td>
<td>7.57B</td>
<td>32.3</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>MMRet-MLLM</td>
<td>LLaVA-1.6</td>
<td>7.57B</td>
<td>42.2</td>
<td>46.7</td>
<td>75.4</td>
<td>35.6</td>
<td>21.1</td>
</tr>
</tbody>
</table>

*Note: † indicates methods with additional external components (e.g., GPT-3.5), * indicates prior SOTA, ‡ indicates proprietary models.*

Key observations:
1. MMRet-MLLM achieves SOTA performance on 3 of 4 benchmarks: 8.1% higher mAP@5 on CIRCO than the previous proprietary SOTA, 7.4% higher R@1 and 4.5% higher R_s@1 on CIRR, and 3.7% higher R@1 on GeneCIS.
2. MMRet outperforms comparable models across all scales: MMRet-Base (149M params) outperforms MagicLens-B (166M) by 11.2% mAP@5 on CIRCO, while MMRet-Large (428M) outperforms MagicLens-L (465M) by 9.6% mAP@5 on CIRCO.
3. MMRet-Base even outperforms much larger models (e.g., 7.57B parameter MM-Embed) on CIRCO, demonstrating the extremely high quality of MegaPairs training data.
### MMEB Zero-shot Results
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th colspan="4">Per Meta-Task Score</th>
<th rowspan="2">Overall</th>
</tr>
<tr>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>Grounding</th>
</tr>
</thead>
<tbody>
<tr>
<td>number of datasets</td>
<td>10</td>
<td>10</td>
<td>12</td>
<td>4</td>
<td>36</td>
</tr>
<tr>
<td>BLIP2 (Li et al., 2023)</td>
<td>27.0</td>
<td>4.2</td>
<td>33.9</td>
<td>47.0</td>
<td>25.2</td>
</tr>
<tr>
<td>SigLIP (Zhai et al., 2023)</td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>34.8</td>
</tr>
<tr>
<td>CLIP (Radford et al., 2021)</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.8</td>
</tr>
<tr>
<td>OpenCLIP (Cherti et al., 2023)</td>
<td>47.8</td>
<td>10.9</td>
<td>52.3</td>
<td>53.3</td>
<td>39.7</td>
</tr>
<tr>
<td>UniIR (Wei et al., 2024)</td>
<td>42.1</td>
<td>15.0</td>
<td>60.1†</td>
<td>62.2</td>
<td>42.8</td>
</tr>
<tr>
<td>MagicLens (Zhang et al., 2024)</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td>27.8</td>
</tr>
<tr>
<td>E5-V (LLaVA-1.6) (Jiang et al., 2024a)</td>
<td>21.8</td>
<td>4.9</td>
<td>11.5</td>
<td>19.0</td>
<td>13.3</td>
</tr>
<tr>
<td>MMRet-MLLM (LLaVA-1.6)</td>
<td>47.2</td>
<td>18.4</td>
<td>56.5</td>
<td>62.2</td>
<td>44.0</td>
</tr>
</tbody>
</table>

*Note: † indicates UniIR is not zero-shot for retrieval tasks as it was trained on 10 of 12 retrieval datasets in MMEB.*

Key observation: MMRet-MLLM achieves the highest overall zero-shot score (44.0) on MMEB, outperforming all baselines. It is the true zero-shot SOTA for retrieval tasks, as the second-highest retrieval score comes from the non-zero-shot UniIR.
### MMEB Fine-tuning Results
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
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
<td>number of datasets</td>
<td>10</td>
<td>10</td>
<td>12</td>
<td>4</td>
<td>20</td>
<td>16</td>
<td>36</td>
</tr>
<tr>
<td>CLIP</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>45.4</td>
</tr>
<tr>
<td>OpenCLIP</td>
<td>56.0</td>
<td>21.9</td>
<td>55.4</td>
<td>64.1</td>
<td>50.5</td>
<td>43.1</td>
<td>47.2</td>
</tr>
<tr>
<td>VLM2Vec (LLaVA-1.6)</td>
<td>54.7</td>
<td>50.3</td>
<td>56.2</td>
<td>64.0</td>
<td>61.0</td>
<td>47.5</td>
<td>55.0</td>
</tr>
<tr>
<td>VLM2Vec (Phi-3.5-V)</td>
<td>54.8</td>
<td>54.9</td>
<td>62.3</td>
<td>79.5</td>
<td>66.5</td>
<td>52.0</td>
<td>60.1</td>
</tr>
<tr>
<td>MMRet-MLLM</td>
<td>56.0</td>
<td>57.4</td>
<td>69.9</td>
<td>83.6</td>
<td>68.0</td>
<td>59.1</td>
<td>64.1</td>
</tr>
</tbody>
</table>

Key observation: After fine-tuning on MMEB in-distribution datasets, MMRet-MLLM achieves an overall SOTA score of 64.1%, 9.1% higher than VLM2Vec (LLaVA-1.6) which directly fine-tunes the same backbone. It also outperforms the two VLM2Vec variants by 11.6% and 7.1% respectively on out-of-distribution datasets, showing significantly better generalization ability from pre-training on MegaPairs.
## 6.2. Ablation Studies / Parameter Analysis
### Data Scalability and Quality
The following figure (Figure 2 from the original paper) shows the performance scaling of MMRet-Base as the size of MegaPairs training data increases:

![Figure 2: Performance scaling of MMRet-base on the MegaPairs as data size increases. The dashed lines indicate the performance of MagicLens-B (CLIP) trained on their dataset of $3 6 . 7 \\mathbf { M }$ data pairs.](images/2.jpg)
*该图像是一个图表，展示了在 MegaPairs 数据集上，MMRet-base 随着训练数据对数的增加而性能的变化情况。不同的曲线代表了 CIRCO、CIRR (val)、FashionIQ、GeneCIS 和平均性能。图中显示，随着训练数据数量的增加，性能逐渐提升，以达成最佳效果。*

Key observation: Performance consistently improves as training data size increases, demonstrating the scalability of MegaPairs. Only 0.5 million MegaPairs instances are sufficient to outperform MagicLens-B trained on 36.7 million instances (70x more data), proving the extreme data efficiency of MegaPairs.
### Impact of Hard Negatives
The following are the results from Table 4 of the original paper, evaluating the impact of different negative sampling strategies:

<table>
<thead>
<tr>
<th colspan="2">Negatives</th>
<th>CIRCO</th>
<th>CIRR†</th>
<th>FIQ</th>
<th>CIS</th>
</tr>
<tr>
<th>Qry</th>
<th>HN (hard negatives)</th>
<th>mAP@5</th>
<th>R@1</th>
<th>R@10</th>
<th>R@1</th>
</tr>
</thead>
<tbody>
<tr>
<td>×</td>
<td>×</td>
<td>10.1</td>
<td>0.2</td>
<td>25.3</td>
<td>14.4</td>
</tr>
<tr>
<td>√</td>
<td>×</td>
<td>29.7</td>
<td>32.1</td>
<td>27.6</td>
<td>16.6</td>
</tr>
<tr>
<td>√</td>
<td>√</td>
<td>32.3</td>
<td>33.7</td>
<td>30.1</td>
<td>17.0</td>
</tr>
</tbody>
</table>

Key observation: Using the mined hard negatives significantly improves performance across all benchmarks, e.g., increasing CIRCO mAP@5 by 2.6% compared to using only query image negatives.
### Data Pair Search Strategy
The following are the results from Table 5 of the original paper, evaluating different image pair mining strategies:

<table>
<thead>
<tr>
<th colspan="3">Strategy</th>
<th>CIRCO</th>
<th>CIRR†</th>
<th>FIQ</th>
<th>CIS</th>
</tr>
<tr>
<th>D (DINOv2)</th>
<th>I (CLIP Image)</th>
<th>T (CLIP Text)</th>
<th>mAP@5</th>
<th>R@1</th>
<th>R@10</th>
<th>R@1</th>
</tr>
</thead>
<tbody>
<tr>
<td>√</td>
<td>×</td>
<td>×</td>
<td>29.0</td>
<td>31.5</td>
<td>24.7</td>
<td>17.2</td>
</tr>
<tr>
<td>×</td>
<td>√</td>
<td>×</td>
<td>30.0</td>
<td>30.0</td>
<td>29.6</td>
<td>15.3</td>
</tr>
<tr>
<td>×</td>
<td>×</td>
<td>√</td>
<td>31.6</td>
<td>32.2</td>
<td>28.7</td>
<td>17.3</td>
</tr>
<tr>
<td>√</td>
<td>√</td>
<td>×</td>
<td>31.0</td>
<td>32.1</td>
<td>28.5</td>
<td>17.1</td>
</tr>
<tr>
<td>√</td>
<td>×</td>
<td>√</td>
<td>32.4</td>
<td>33.3</td>
<td>28.9</td>
<td>17.5</td>
</tr>
<tr>
<td>×</td>
<td>√</td>
<td>√</td>
<td>32.2</td>
<td>33.3</td>
<td>29.7</td>
<td>16.4</td>
</tr>
<tr>
<td>√</td>
<td>√</td>
<td>√</td>
<td>32.3</td>
<td>33.7</td>
<td>30.1</td>
<td>17.0</td>
</tr>
</tbody>
</table>

Key observation: Combining multiple similarity strategies consistently outperforms using any single strategy. Using all three strategies (DINOv2, CLIP Image, CLIP Text) yields the best overall performance, as it provides the most diverse set of relationships between image pairs.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces MegaPairs, a novel, low-cost, scalable data synthesis pipeline for generating high-quality instruction tuning data for universal multimodal retrieval. By mining diverse correlated image pairs from open-domain corpora using three similarity models and generating instructions with open-source VLMs and LLMs, the authors produce a 26 million instance public dataset that is 70 times more data-efficient than existing state-of-the-art datasets. The MMRet models trained on MegaPairs achieve SOTA zero-shot performance on 4 CIR benchmarks and the 36-dataset MMEB benchmark, and also show superior fine-tuning performance and generalization to out-of-distribution tasks. The public release of the dataset, models, and synthesis pipeline will significantly accelerate future research in multimodal retrieval.
## 7.2. Limitations & Future Work
The authors identify the following limitations:
1. Currently only three similarity models are used for mining image pairs. Future work can explore additional pairing methods, such as using more advanced text retrievers (e.g., BGE) or image-text retrieval strategies to capture even more diverse relationships.
2. While the dataset is screened for harmful content by the source dataset curators, there may still be omissions. Future work can improve content filtering to further reduce harmful instances.
## 7.3. Personal Insights & Critique
This paper is a landmark contribution to the multimodal retrieval field, as it addresses the long-standing critical bottleneck of lack of large, high-quality public training data for universal multimodal retrievers. The 70x data efficiency improvement over the prior state-of-the-art is particularly impressive, demonstrating that carefully curated synthetic data can far outperform naturally collected data for this task. The fully open pipeline and dataset will democratize research in multimodal retrieval, enabling small research teams without access to large private web crawls to train state-of-the-art models.
Potential improvements and extensions:
1. The current instruction generation focuses only on image-to-image transformation instructions for CIR. Future work can expand the instruction types to support other multimodal retrieval tasks, such as multimodal document retrieval and cross-modal retrieval, to further improve the model's universality.
2. The CLIP-based MMRet uses simple element-wise addition for fusing image and text embeddings for composed queries. Future work can explore more advanced fusion methods (e.g., cross-attention) to improve performance on composed query tasks.
3. The pipeline can be extended to support other modalities, such as video-text pairs, to generate training data for video retrieval tasks.
   The methodology is also highly transferable to other multimodal tasks requiring instruction tuning data, such as multimodal generation and visual question answering.