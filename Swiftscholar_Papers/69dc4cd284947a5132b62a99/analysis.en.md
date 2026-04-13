# 1. Bibliographic Information
## 1.1. Title
The paper's central topic is the development of **Paracosm**, a training-free zero-shot Composed Image Retrieval (CIR) method that constructs a unified synthetic image space to eliminate synthetic-to-real domain gaps, enabling accurate retrieval using generated "mental images" of multimodal queries.
## 1.2. Authors
The authors are:
- Tong Wang: University of Macau, research focus on computer vision and multimodal learning
- Yunhan Zhao: UC Irvine, research focus on vision-language systems
- Shu Kong: University of Macau & Institute of Collaborative Innovation, University of Macau, senior researcher in multimodal retrieval and foundation model applications
## 1.3. Journal/Conference
The paper is currently hosted as a preprint on arXiv, the leading open-access preprint server for computer science research. It has not yet been peer-reviewed or published in a formal conference/journal as of April 2026.
## 1.4. Publication Year
The preprint was published on January 31, 2026 (UTC).
## 1.5. Abstract
The paper addresses the core challenge of Composed Image Retrieval (CIR): the "mental image" implied by a multimodal query (reference image + modification text) does not physically exist for matching. Existing zero-shot CIR methods rely on Large Multimodal Model (LMM) generated text descriptions of queries, which lose rich visual detail. The proposed Paracosm method directly generates the mental image from the query using an LMM, then generates synthetic counterparts for all database images to bridge the synthetic-to-real domain gap. Matching is performed in this unified synthetic "paracosm" space. Paracosm is fully training-free, achieves state-of-the-art performance on all standard zero-shot CIR benchmarks, and even outperforms some supervised CIR methods.
## 1.6. Original Source Link
- Abstract link: https://arxiv.org/abs/2602.00813
- PDF link: https://arxiv.org/pdf/2602.00813
- Publication status: Preprint, not yet peer-reviewed.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Composed Image Retrieval (CIR) requires retrieving a target image from a database using a query consisting of a reference image and a text description of how to modify the reference to match the target. The fundamental challenge is that the target "mental image" implied by the query does not exist physically, so prior methods rely on indirect representations.
### Research Gap
- Supervised CIR methods require expensive annotated triplet data (reference image, modification text, target image), which cannot scale.
- Existing zero-shot CIR methods either train auxiliary models (not fully training-free) or generate only text descriptions of queries, which fail to capture fine-grained visual details critical for accurate retrieval.
- Generative zero-shot CIR methods that produce synthetic query images suffer from large synthetic-to-real domain gaps when matching against real database images.
### Innovative Idea
The paper addresses CIR from first principles: directly generate the missing mental image for the query, then eliminate the synthetic-to-real domain gap by generating synthetic counterparts for all database images, so matching occurs in a unified synthetic space (the "paracosm").
## 2.2. Main Contributions / Findings
The paper makes three core contributions:
1. **Novel training-free pipeline**: The first fully training-free zero-shot CIR method that uses generated visual mental images for queries, rather than relying solely on text descriptions.
2. **Domain gap mitigation**: Proposes generating synthetic counterparts for all database images to align query and database representations in the same synthetic domain, eliminating the synthetic-to-real matching gap.
3. **State-of-the-art performance**: Paracosm outperforms all existing zero-shot CIR methods (both training-based and training-free) on standard benchmarks, and even exceeds the performance of some supervised CIR methods, despite using no task-specific training data.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All key terms are explained for beginner understanding:
- **Composed Image Retrieval (CIR)**: A retrieval task where the input is a multimodal query: 1 reference image + 1 modification text that specifies how to alter the reference image to match the desired target. For example: reference = red short dress, modification text = "make it blue, floor-length, add lace sleeves", target = long blue lace-sleeve dress.
- **Zero-Shot CIR (ZS-CIR)**: A CIR approach that does not use annotated triplet (reference, modification text, target) training data for task-specific training.
- **Training-Free ZS-CIR**: A subset of ZS-CIR that does not train any new models (no fine-tuning, no auxiliary network training) at all, only uses off-the-shelf pre-trained foundation models as-is.
- **Large Multimodal Model (LMM)**: A pre-trained model capable of processing and generating multiple modalities (text, images, video, etc.). Examples: GPT-4V, Qwen2.5-VL, Gemini. LMMs can perform image captioning, visual question answering, image editing, and text-to-image generation.
- **Vision-Language Model (VLM)**: A pre-trained model trained on large-scale image-text pairs, which maps images and text to the same high-dimensional embedding space, enabling similarity calculation between any combination of images and text. Examples: CLIP, OpenCLIP, BLIP.
- **Synthetic-to-Real Domain Gap**: The distribution difference between synthetically generated images and real-world images. Models or similarity metrics that work well on synthetic data often perform poorly on real data due to this gap.
- **Paracosm (as defined in the paper)**: A virtual, fully synthetic image space constructed by the method, where both the query's mental image and synthetic versions of all database images exist, so matching is performed in a single domain to eliminate gaps.
## 3.2. Previous Works
CIR research can be categorized into three major groups:
1. **Supervised CIR (2019-2023)**:
   These methods train on large annotated triplet datasets. Representative works include:
   - Combiner (CVPR 2022): Trains a fusion network on top of CLIP to combine reference image and modification text features.
   - BLIP4CIR (WACV 2024): Fine-tunes the BLIP VLM for CIR tasks.
     Limitation: Annotating triplet data is extremely labor-intensive and expensive, limiting scalability.
2. **Training-Based Zero-Shot CIR (2023-2025)**:
   These methods avoid triplet annotations but still require training auxiliary models:
   - Pic2Word (CVPR 2023): Trains a textual inversion network to map reference images to text tokens, which are combined with modification text for matching.
   - SEARLE (ICCV 2023): Improves textual inversion for better ZS-CIR performance.
   - IP-CIR (CVPR 2025): Generates pseudo-target images from modified text descriptions, but still relies on a trained textual inversion network.
     Limitation: Require training of auxiliary components, adding deployment complexity.
3. **Training-Free Zero-Shot CIR (2024-2025)**:
   These methods use only pre-trained LMMs/VLMs without any training:
   - CIReVL (ICLR 2024): Uses an LMM to caption the reference image, then an LLM to modify the caption with the modification text, and uses the modified text for text-to-image retrieval.
   - OSrCIR (CVPR 2025): Uses chain-of-thought prompting for LMMs to generate higher-quality query descriptions.
   - CoTMR (ICCV 2025): Generates both global and object-level descriptions for queries, using multi-grained scoring.
     Limitation: All rely exclusively on text descriptions of queries, which lose critical fine-grained visual details, and none address synthetic-to-real domain gaps for generative query representations.
## 3.3. Technological Evolution
The evolution of CIR methods follows a clear trajectory:
1. 2019-2022: Fully supervised CIR dominates, requiring triplet annotations.
2. 2023: First zero-shot CIR methods emerge, using textual inversion but still requiring auxiliary model training.
3. 2024: First training-free ZS-CIR methods appear, using LMMs to generate query text descriptions, no training required but limited to text representations.
4. 2025: Generative ZS-CIR methods start producing synthetic query images, but suffer from domain gaps and still require training.
5. 2026 (this work): First training-free generative ZS-CIR that addresses synthetic-to-real domain gaps by creating a unified synthetic space for both queries and database images.
## 3.4. Differentiation Analysis
Paracosm has four core innovations over prior work:
1. Vs. supervised CIR: Requires no annotated triplet data at all, is fully training-free, reducing deployment cost by orders of magnitude.
2. Vs. training-based ZS-CIR: Requires no training of any models (no textual inversion, no fine-tuning), only uses off-the-shelf pre-trained models.
3. Vs. existing training-free ZS-CIR: Uses generated visual mental images instead of only text descriptions, preserving far more fine-grained visual detail for better retrieval.
4. Vs. generative ZS-CIR methods: Eliminates the synthetic-to-real domain gap by generating synthetic counterparts for all database images, so matching occurs in the same synthetic domain, rather than matching synthetic query images to real database images.

# 4. Methodology
## 4.1. Principles
The core intuition of Paracosm is that visual representations (images) carry far more fine-grained information than text descriptions, so directly generating the mental image of the query will lead to more accurate retrieval than using text alone, as long as the synthetic-to-real domain gap is resolved. The method constructs a unified synthetic "paracosm" space where both the query's mental image and synthetic versions of database images reside, enabling fair and accurate matching.
## 4.2. Core Methodology In-depth
The full pipeline of Paracosm is shown in Figure 2 from the original paper:

![Fig. 2: Flowchart of our training-free zero-shot CIR method Paracosm. Given a multimodal query that consists of a reference image and a modification text, we feed it to an LMM to generate a "mental image". We further generate a brief description for it. Both the "mental image" and description, as well as the modification text, are used as feature representations for the query. As the "mental image" is synthetic, we mitigate synthetic-to-real domain gaps by generating synthetic counterparts of database images. To do so, we use the LMM to generate detailed descriptions, which are used as prompts for image generation. For a database image, we use both itself (i.e., the real photo) and its synthetic counterpart as representation for retrieval. In sum, our method uses LMMs to create a virtual paracosm, where it matches the query and database images.](images/2.jpg)
*该图像是示意图，展示了我们的训练-free zero-shot CIR 方法 Paracosm 的流程。左侧处理多模态查询，包括引用图像和文本描述；右侧为数据库图像预处理，生成合成图像和详细描述，以缩小合成与真实图像之间的域差距。*

The pipeline is broken into three sequential steps:
---
### Step 1: Process the multimodal query
The input query is defined as $(\mathbf{I}_{ref}, \mathbf{t}_{mod})$, where:
- $\mathbf{I}_{ref}$ = the input reference image
- $\mathbf{t}_{mod}$ = the modification text describing how to alter $\mathbf{I}_{ref}$ to get the target image
  Three sub-steps are performed:
1. **Generate mental image $\mathbf{I}_{mental}$**: Use a pre-trained LMM with image editing capability (e.g., Qwen-Image-Edit) to directly edit $\mathbf{I}_{ref}$ according to $\mathbf{t}_{mod}$. The prompt is designed to handle both simple modification texts and complex texts with relative captions or shared concepts (as in datasets like CIRCO).
2. **Generate brief query description $\mathbf{t}_{query}$**: Use an LMM (e.g., Qwen2.5-VL-7B-Instruct) to generate a single-sentence description of $\mathbf{I}_{mental}$, focused exclusively on visual content with no aesthetic details. This acts as a text representation of the target image.
3. **Query feature components**: The query uses three components for feature construction: $\mathbf{I}_{mental}$, $\mathbf{t}_{query}$, and the original modification text $\mathbf{t}_{mod}$.
   Examples of query processing are shown in Figure 4 from the original paper:

   ![Fig. 4: Illustration of processing a multimodal query. Two random examples from CIRR and CIRCO datasets are displayed in the two rows, respectively. For a multimodal query consisting of a reference image and modification texts, we design a prompt incorporating the latter to edit the former, generating a mental image representing this query. As a modification text can contain a relative caption and a shared concept (ref. CIRCO in the second row), we design the prompt to incorporate both. Further, for the mental image, we prompt an LMM to generate a a concise, single-sentence description, exclusively focusing on its visual content while minimizing aesthetic details. We use both mental image and short description to retrieve the target image from the database.](images/4.jpg)
   *该图像是一个示意图，展示了如何处理一个多模态查询。在左侧是参考图像，中间是修改提示，右侧是生成的心理图像和描述。此例说明了如何根据修改提示编辑参考图像，并生成简短描述，以便检索目标图像。*

---
### Step 2: Preprocess database images to mitigate synthetic-to-real domain gap
The database contains $n$ real images $\{\mathbf{I}^1, \mathbf{I}^2, ..., \mathbf{I}^n\}$. Two sub-steps are performed for each database image $\mathbf{I}^i$:
1. **Generate detailed description**: Use an LMM to generate a comprehensive description of $\mathbf{I}^i$, capturing all visible objects, attributes, spatial relationships, and fine-grained visual elements to maximize fidelity.
2. **Generate synthetic counterpart $\mathbf{I}_{syn}^i$**: Use the detailed description as a prompt for a text-to-image generation LMM (e.g., Qwen-Image) to generate a synthetic version of the real database image $\mathbf{I}^i$.
3. **Database feature components**: Each database image uses both the original real image $\mathbf{I}^i$ and its synthetic counterpart $\mathbf{I}_{syn}^i$ for feature construction.
   Examples of database image processing are shown in Figure 5 from the original paper:

   ![Fig. 5: Ilustration of processing database images. For a database image, we first prompt an LMM to generate a comprehensive description about its visual content, capturing all visible objects, attributes, and visual elements. Using this description, we prompt a text-to-image generation model to produce a synthetic counterpart for this database image. For a database image, we use both itself (i.e., the real photo) and its synthetic counterpart as representation for retrieval.](images/5.jpg)
   *该图像是示意图，描述了数据库图像的处理过程。首先，提示大型多模态模型生成图像内容的全面描述，然后利用该描述生成数据库图像的合成对应物，以便于检索。*

---
### Step 3: Feature construction and matching
A pre-trained VLM with a visual encoder $V(\cdot)$ (maps images to embedding vectors) and text encoder $T(\cdot)$ (maps text to embedding vectors in the same space as image embeddings) is used for all feature extraction.
#### Query Feature Calculation
The combined query feature $\mathbf{q}$ is computed as:
$$
\mathbf{q} = \lambda ( V(\mathbf{I}_{mental}) + T(\mathbf{t}_{query}) ) + (1-\lambda) T(\mathbf{t}_{mod})
$$
Symbol explanation:
- $\lambda$: Hyperparameter between 0 and 1 that controls the relative weight of the mental image + its description vs. the original modification text. Set to 0.3 based on experimental results.
- $V(\mathbf{I}_{mental})$: Embedding vector of the generated mental image, output by the VLM's visual encoder.
- $T(\mathbf{t}_{query})$: Embedding vector of the brief description of the mental image, output by the VLM's text encoder.
- $T(\mathbf{t}_{mod})$: Embedding vector of the original modification text, output by the VLM's text encoder.
  Intuition: This combines the rich visual representation of the target (from the mental image) with the original modification text, which often contains high-impact keywords that improve matching accuracy.
#### Database Feature Calculation
The feature vector for the $i$-th database image $\phi^i$ is computed as:
$$
\phi^i = V(\mathbf{I}^i) + V(\mathbf{I}_{syn}^i)
$$
Symbol explanation:
- $V(\mathbf{I}^i)$: Embedding vector of the original real database image, output by the VLM's visual encoder.
- $V(\mathbf{I}_{syn}^i)$: Embedding vector of the synthetic counterpart of the database image, output by the VLM's visual encoder.
  Intuition: Combining both real and synthetic embeddings aligns the database representation to both domains, so it matches well with the synthetic mental image's embedding while retaining alignment to real image features.
#### Similarity Calculation and Ranking
The cosine similarity between the query feature $\mathbf{q}$ and each database feature $\phi^i$ is computed, and database images are ranked by similarity. The top retrieved result is:
$$
i^* = \underset{i=1}{\operatorname{argmax}} \frac{\mathbf{q}^T \phi^i}{\|\mathbf{q}\|_2 \|\phi^i\|_2}
$$
Symbol explanation:
- $\mathbf{q}^T \phi^i$: Dot product between the query and database embedding vectors, measuring raw similarity.
- $\|\mathbf{q}\|_2$: L2 norm of the query embedding, $\|\phi^i\|_2$: L2 norm of the database embedding. Dividing by their product normalizes the similarity to the range [-1, 1], the standard cosine similarity metric for embedding spaces.
- $i^*$ = the index of the database image with the highest similarity to the query.
  ---
An extended variant of the pipeline that adds brief text descriptions of database images was tested, but did not yield consistent performance gains, so it is not included in the final method. The extended pipeline is shown below:

![Fig. 8: Extended flowchart of Paracosm incorporating brief database image descriptions $( \\mathbf { w } / \\mathbf { \\nabla } \\mathbf { t } _ { b r i e f } ^ { i } )$ . This variant incorporates brief textual descriptions of database images into the retrieval process. This version supports the study in Table 8, which evaluates the impact of brief database image descriptions on performance.](images/8.jpg)
*该图像是示意图，展示了 Paracosm 方法在处理多模态查询和图像数据库预处理中的流程。左侧展示了如何处理多模态查询，包括生成 'mental image' 和简要描述；右侧展示了数据库图像的预处理，包含生成详细和简要描述的过程。*

# 5. Experimental Setup
## 5.1. Datasets
Three standard, widely used zero-shot CIR benchmarks are used. No training data is used (Paracosm is training-free, so only validation/test sets are evaluated):
1. **CIRR (Composed Image Retrieval on Real-life images)**:
   - Test set: 4,148 queries, 2,315 database images
   - Characteristics: Organizes visually similar images into subsets to test fine-grained discriminative capability.
   - License: CC-BY 4.0
2. **CIRCO (Composed Image Retrieval on Common Objects in context)**:
   - Test set: 800 queries, 123,403 database images
   - Characteristics: First ZS-CIR dataset with multiple ground-truth targets per query, fine-grained open-domain annotations.
   - License: CC-BY-NC 4.0
3. **Fashion IQ**:
   - Fashion domain dataset with three subsets: Shirt (2,038 queries, 6,346 database images), Dress (2,017 queries, 3,817 database images), Toptee (1,961 queries, 5,373 database images)
   - Characteristics: Only the validation set is publicly available, so evaluation is performed on this split.
   - License: Community Data License Agreement (CDLA)
     Example query (Fashion IQ): Reference image = white short-sleeve cotton shirt, modification text = "change color to black, make it long sleeve, add a small chest pocket on the left side", target = black long-sleeve cotton shirt with left chest pocket.
These datasets are chosen because they cover both general open-domain (CIRR, CIRCO) and vertical domain (Fashion IQ) use cases, making results generalizable to real-world applications.
## 5.2. Evaluation Metrics
Two standard metrics are used, explained fully for beginners:
### 1. Recall@k (R@k)
- **Conceptual Definition**: Measures the percentage of queries where the ground-truth target image appears in the top-k retrieved results. Higher values indicate better performance. It answers the question: "How often do we find the correct target in the first k results?"
- **Mathematical Formula**:
  $$
  \text{Recall@k} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(\text{rank of target for query } i \leq k)
  $$
- **Symbol Explanation**:
  - $N$: Total number of queries in the test set.
  - $\mathbb{1}(\cdot)$: Indicator function, equals 1 if the condition inside is true, 0 otherwise.
  - $\text{rank of target for query } i$: The position of the ground-truth target in the ranked list of retrieved images for query $i$ (rank 1 = top result).
    A variant, **RecallSubset@k**, is used for CIRR: it calculates Recall@k only within a subset of visually similar images to the target, making the task harder and testing fine-grained discriminative ability.
### 2. Mean Average Precision at k (mAP@k)
- **Conceptual Definition**: Used for datasets with multiple ground-truth targets per query (e.g., CIRCO). It first calculates the average precision (AP) for each query (the average of precision values at all positions where a ground-truth target appears in the top-k results), then takes the mean over all queries. Higher values indicate better performance, as it measures both how many correct targets are retrieved and how high they are ranked.
- **Mathematical Formula**:
  First, average precision (AP) for a single query:
  $$
  \text{AP@k} = \frac{1}{M} \sum_{j=1}^k P(j) \cdot \mathbb{1}(j\text{-th result is ground truth})
  $$
  Where `P(j)` = precision at position j: $\frac{\text{number of ground-truth targets in top } j \text{ results}}{j}$, and $M$ = total number of ground-truth targets for the query.
  Then mAP@k is the mean of AP@k over all queries:
  $$
  \text{mAP@k} = \frac{1}{N} \sum_{i=1}^N \text{AP@k}_i
  $$
- **Symbol Explanation**:
  - $N$: Total number of queries.
  - $\text{AP@k}_i$: Average precision at k for the i-th query.
  - $M$: Number of ground-truth targets for the current query.
  - `P(j)`: Precision at the j-th position in the retrieved list.
  - $\mathbb{1}(j\text{-th result is ground truth})$: Indicator function, 1 if the j-th retrieved image is a ground-truth target, 0 otherwise.
## 5.3. Baselines
Paracosm is compared against three groups of representative baselines:
1. **Simple baselines**:
   - Image-only: Uses only the reference image's embedding to match database images.
   - Text-only: Uses only the modification text's embedding to match database images.
   - Image+Text: Sums the reference image embedding and modification text embedding for matching.
2. **Training-based ZS-CIR baselines**: Pic2Word, SEARLE, LinCIR, LDRE, CIReVL, IP-CIR, CIG (all state-of-the-art training-based ZS-CIR methods).
3. **Training-free ZS-CIR baselines**: CIReVL, LDRE, AutoCIR, CoTMR, OSrCIR (all state-of-the-art training-free ZS-CIR methods).
   These baselines cover the full range of existing ZS-CIR approaches, so comparisons fairly measure Paracosm's effectiveness.

# 6. Results & Analysis
## 6.1. Core Results Analysis
An overview of Paracosm's performance relative to other methods is shown in the radar chart in Figure 1 from the original paper:

![Fig. 1: Overview of our method and benchmarking results. Unlike existing training-free methods \[21, 49, 62\] generating descriptions for multimodal queries, which use an LMM to generate descriptions for multimodal queries, we use it to generate "mental images" for multimodal queries and synthetic counterparts of database images. Matching them effectively mitigates synthetic-to-real domain gaps and boosts CIR performance. Our final training-free zero-shot method Paracosm (Fig. 2) significantly outperforms existing zero-shot CIR methods, as summarized in the radar chart on standard benchmarks. Detailed results are provided in Section E.](images/1.jpg)
*该图像是一个示意图，展示了Paracosm方法在训练-free零-shot组合图像检索中的应用。左侧雷达图比较了不同方法在CIR性能上的表现，右侧展示了如何生成''心理图像''及其对应的合成图像，并进行匹配以提高性能。*

### Quantitative Results on CIRR and CIRCO Test Sets
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="3" colspan="3">Backbone & Method & venue&amp;year</th>
<th colspan="6">CIRR</th>
<th colspan="4">CIRCO</th>
</tr>
<tr>
<th colspan="3">Recall@k</th>
<th colspan="3">RecallSubset@k</th>
<th colspan="4">mAP@k</th>
</tr>
<tr>
<th>k=1</th>
<th>k=5</th>
<th>k=10</th>
<th>k=1</th>
<th>k=2</th>
<th>k=3</th>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=50</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" colspan="2">Supervised methods</td>
<td>Combiner [5] CVPR'22</td>
<td>33.59</td>
<td>65.35</td>
<td>77.35</td>
<td>62.39</td>
<td>81.81</td>
<td>92.02</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>BLIP4CIR [33] WACV'24</td>
<td>40.17</td>
<td>71.81</td>
<td>83.18</td>
<td>72.34</td>
<td>88.70</td>
<td>95.23</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>CLIP-ProbCR [27] ICMR'24</td>
<td>23.32</td>
<td>54.36</td>
<td>68.64</td>
<td>54.32</td>
<td>76.30</td>
<td>88.88</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td rowspan="10">ViT-L/14</td>
<td colspan="2">Image-only baseline</td>
<td>7.47</td>
<td>23.86</td>
<td>34.10</td>
<td>20.82</td>
<td>41.88</td>
<td>61.23</td>
<td>1.80</td>
<td>2.43</td>
<td>3.04</td>
<td>3.45</td>
</tr>
<tr>
<td colspan="2">Text-only baseline</td>
<td>22.05</td>
<td>45.64</td>
<td>57.54</td>
<td>61.69</td>
<td>80.31</td>
<td>90.41</td>
<td>2.99</td>
<td>3.18</td>
<td>3.68</td>
<td>3.92</td>
</tr>
<tr>
<td colspan="2">Image+Text baseline</td>
<td>10.58</td>
<td>32.65</td>
<td>45.69</td>
<td>31.08</td>
<td>55.71</td>
<td>73.90</td>
<td>3.89</td>
<td>4.79</td>
<td>5.93</td>
<td>6.47</td>
</tr>
<tr>
<td>Pic2Word [39]</td>
<td>CVPR'23</td>
<td>23.90</td>
<td>51.70</td>
<td>65.30</td>
<td>53.76</td>
<td>74.46</td>
<td>87.07</td>
<td>8.72</td>
<td>9.51</td>
<td>10.64</td>
<td>11.29</td>
</tr>
<tr>
<td>SEARLE [4]</td>
<td>ICCV'23</td>
<td>24.24</td>
<td>52.48</td>
<td>66.29</td>
<td>53.76</td>
<td>75.01</td>
<td>88.19</td>
<td>11.68</td>
<td>12.73</td>
<td>14.33</td>
<td>15.12</td>
</tr>
<tr>
<td>LinCIR [16]</td>
<td>CVPR'24</td>
<td>25.04</td>
<td>53.25</td>
<td>66.68</td>
<td>57.11</td>
<td>77.37</td>
<td>88.89</td>
<td>12.59</td>
<td>13.58</td>
<td>15.00</td>
<td>15.85</td>
</tr>
<tr>
<td>LDRE [62]</td>
<td>SIGIR'24</td>
<td>26.53</td>
<td>55.57</td>
<td>67.54</td>
<td>60.43</td>
<td>80.31</td>
<td>89.90</td>
<td>23.35</td>
<td>24.03</td>
<td>26.44</td>
<td>27.50</td>
</tr>
<tr>
<td>CIReVL [21]</td>
<td>ICLR'24</td>
<td>24.55</td>
<td>52.31</td>
<td>64.92</td>
<td>59.54</td>
<td>79.88</td>
<td>89.69</td>
<td>18.57</td>
<td>19.01</td>
<td>20.89</td>
<td>21.80</td>
</tr>
<tr>
<td>IP-CIR + LDRE [29]</td>
<td>CVPR'25</td>
<td>29.76</td>
<td>58.82</td>
<td>71.21</td>
<td>62.48</td>
<td>81.64</td>
<td>90.89</td>
<td>26.43</td>
<td>27.41</td>
<td>29.87</td>
<td>31.07</td>
</tr>
<tr>
<td>CIG + SEARLE [57]</td>
<td>CVPR'25</td>
<td>26.72</td>
<td>55.52</td>
<td>68.10</td>
<td>57.95</td>
<td>77.81</td>
<td>89.45</td>
<td>12.84</td>
<td>13.64</td>
<td>15.32</td>
<td>16.17</td>
</tr>
<tr>
<td></td>
<td>Paracosm</td>
<td>ours</td>
<td>31.95</td>
<td>61.56</td>
<td>72.96</td>
<td>64.68</td>
<td>82.89</td>
<td>91.47</td>
<td>30.24</td>
<td>31.51</td>
<td>34.29</td>
<td>35.42</td>
</tr>
<tr>
<td rowspan="9">ViT-G/14</td>
<td>Pic2Word [39]</td>
<td>CVPR'23</td>
<td>30.41</td>
<td>58.12</td>
<td>69.23</td>
<td>68.92</td>
<td>85.45</td>
<td>93.04</td>
<td>5.54</td>
<td>5.59</td>
<td>6.68</td>
<td>7.12</td>
</tr>
<tr>
<td>SEARLE [4]</td>
<td>ICCV'23</td>
<td>34.80</td>
<td>64.07</td>
<td>75.11</td>
<td>68.72</td>
<td>84.70</td>
<td>93.23</td>
<td>13.20</td>
<td>13.85</td>
<td>15.32</td>
<td>16.04</td>
</tr>
<tr>
<td>LinCIR [16]</td>
<td>CVPR'24</td>
<td>35.25</td>
<td>64.72</td>
<td>76.05</td>
<td>63.35</td>
<td>82.22</td>
<td>91.98</td>
<td>19.71</td>
<td>21.01</td>
<td>23.13</td>
<td>24.18</td>
</tr>
<tr>
<td>LDRE [62]</td>
<td>SIGIR'24</td>
<td>36.15</td>
<td>66.39</td>
<td>77.25</td>
<td>68.82</td>
<td>85.66</td>
<td>93.76</td>
<td>31.12</td>
<td>32.24</td>
<td>34.95</td>
<td>36.03</td>
</tr>
<tr>
<td>CIReVL [21]</td>
<td>ICLR'24</td>
<td>34.65</td>
<td>64.29</td>
<td>75.06</td>
<td>67.95</td>
<td>84.87</td>
<td>93.21</td>
<td>26.77</td>
<td>27.59</td>
<td>29.96</td>
<td>31.03</td>
</tr>
<tr>
<td>CIG + LinIR [57]</td>
<td>CVPR'25</td>
<td>36.05</td>
<td>66.31</td>
<td>76.96</td>
<td>64.94</td>
<td>83.18</td>
<td>91.93</td>
<td>20.64</td>
<td>21.90</td>
<td>24.04</td>
<td>25.20</td>
</tr>
<tr>
<td>CoTMR [44]</td>
<td>ICCV'25</td>
<td>36.36</td>
<td>67.52</td>
<td>77.82</td>
<td>71.19</td>
<td>86.34</td>
<td>93.87</td>
<td>32.23</td>
<td>32.72</td>
<td>35.60</td>
<td>36.83</td>
</tr>
<tr>
<td>OSrCIR [49]</td>
<td>CVPR'25</td>
<td>37.26</td>
<td>67.25</td>
<td>77.33</td>
<td>69.22</td>
<td>85.28</td>
<td>93.55</td>
<td>30.47</td>
<td>31.14</td>
<td>35.03</td>
<td>36.59</td>
</tr>
<tr>
<td>Paracosm</td>
<td>ours</td>
<td>39.30</td>
<td>70.41</td>
<td>80.39</td>
<td>70.82</td>
<td>86.92</td>
<td>94.46</td>
<td>39.82</td>
<td>40.86</td>
<td>43.96</td>
<td>45.05</td>
</tr>
</tbody>
</table>

### Quantitative Results on Fashion IQ Validation Set
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Method</th>
<th rowspan="2">venue&amp;year</th>
<th colspan="2">Shirt</th>
<th colspan="2">Dress</th>
<th colspan="2">Toptee</th>
<th colspan="2">Average</th>
</tr>
<tr>
<th>R@10</th>
<th>R@50</th>
<th>R@10</th>
<th>R@50</th>
<th>R@10</th>
<th>R@50</th>
<th>R@10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Supervised methods</td>
<td>Combiner [5]</td>
<td>CVPR'22</td>
<td>36.36</td>
<td>58.00</td>
<td>31.63</td>
<td>56.67</td>
<td>38.19</td>
<td>62.42</td>
<td>35.39</td>
<td>59.03</td>
</tr>
<tr>
<td>PL4CIR [66]</td>
<td>SIGIR'22</td>
<td>33.22</td>
<td>59.99</td>
<td>46.17</td>
<td>68.79</td>
<td>46.46</td>
<td>73.84</td>
<td>41.98</td>
<td>67.54</td>
</tr>
<tr>
<td>Uncertainty retrieval [11]</td>
<td>ICLR'24</td>
<td>32.61</td>
<td>61.34</td>
<td>33.23</td>
<td>62.55</td>
<td>41.40</td>
<td>72.51</td>
<td>35.75</td>
<td>65.47</td>
</tr>
<tr>
<td rowspan="9">ViT-L/14</td>
<td>Image-only</td>
<td>baseline</td>
<td>10.45</td>
<td>20.76</td>
<td>5.21</td>
<td>13.49</td>
<td>8.01</td>
<td>18.05</td>
<td>7.89</td>
<td>17.43</td>
</tr>
<tr>
<td>Text-only</td>
<td>baseline</td>
<td>20.26</td>
<td>34.10</td>
<td>15.12</td>
<td>33.71</td>
<td>21.98</td>
<td>39.98</td>
<td>19.12</td>
<td>35.93</td>
</tr>
<tr>
<td>Image+Text</td>
<td>baseline</td>
<td>19.14</td>
<td>32.63</td>
<td>14.38</td>
<td>31.09</td>
<td>20.50</td>
<td>36.26</td>
<td>18.01</td>
<td>33.33</td>
</tr>
<tr>
<td>Pic2Word [39]</td>
<td>CVPR'23</td>
<td>26.20</td>
<td>43.60</td>
<td>20.00</td>
<td>40.20</td>
<td>27.90</td>
<td>47.40</td>
<td>24.70</td>
<td>43.70</td>
</tr>
<tr>
<td>SEARLE [4]</td>
<td>ICCV'23</td>
<td>26.89</td>
<td>45.58</td>
<td>20.48</td>
<td>43.13</td>
<td>29.32</td>
<td>49.97</td>
<td>25.56</td>
<td>46.23</td>
</tr>
<tr>
<td>LinCIR [16]</td>
<td>CVPR'24</td>
<td>29.10</td>
<td>46.81</td>
<td>20.92</td>
<td>42.44</td>
<td>28.81</td>
<td>50.18</td>
<td>26.28</td>
<td>46.49</td>
</tr>
<tr>
<td>CIReVL [21]</td>
<td>ICLR'24</td>
<td>29.49</td>
<td>47.40</td>
<td>24.79</td>
<td>44.76</td>
<td>31.36</td>
<td>53.65</td>
<td>28.55</td>
<td>48.57</td>
</tr>
<tr>
<td>CIG + LinCIR [57]</td>
<td>CVPR'25</td>
<td>28.90</td>
<td>47.25</td>
<td>21.12</td>
<td>43.88</td>
<td>29.78</td>
<td>50.54</td>
<td>26.60</td>
<td>47.22</td>
</tr>
<tr>
<td>Paracosm</td>
<td>ours</td>
<td>31.80</td>
<td>49.51</td>
<td>24.99</td>
<td>47.45</td>
<td>31.82</td>
<td>52.83</td>
<td>29.45</td>
<td>49.93</td>
</tr>
<tr>
<td rowspan="7">ViT-G/14</td>
<td>Pic2Word [39]</td>
<td>CVPR'23</td>
<td>33.17</td>
<td>50.39</td>
<td>25.43</td>
<td>47.65</td>
<td>35.24</td>
<td>57.62</td>
<td>31.28</td>
<td>51.89</td>
</tr>
<tr>
<td>SEARLE [4]</td>
<td>ICCV'23</td>
<td>36.46</td>
<td>55.35</td>
<td>28.16</td>
<td>50.32</td>
<td>39.83</td>
<td>61.45</td>
<td>34.82</td>
<td>55.71</td>
</tr>
<tr>
<td>CIReVL [21]</td>
<td>ICLR'24</td>
<td>33.71</td>
<td>51.42</td>
<td>27.07</td>
<td>49.53</td>
<td>35.80</td>
<td>56.14</td>
<td>32.19</td>
<td>52.36</td>
</tr>
<tr>
<td>LDRE [62]</td>
<td>SIGIR'24</td>
<td>35.94</td>
<td>58.58</td>
<td>26.11</td>
<td>51.12</td>
<td>35.42</td>
<td>56.67</td>
<td>32.49</td>
<td>55.46</td>
</tr>
<tr>
<td>AutoCIR [12]</td>
<td>KDD'25</td>
<td>36.36</td>
<td>55.84</td>
<td>26.18</td>
<td>47.69</td>
<td>37.28</td>
<td>60.38</td>
<td>33.27</td>
<td>54.63</td>
</tr>
<tr>
<td>OSrCIR [49]</td>
<td>CVPR'25</td>
<td>38.65</td>
<td>54.71</td>
<td>33.02</td>
<td>54.78</td>
<td>41.04</td>
<td>61.83</td>
<td>37.57</td>
<td>57.11</td>
</tr>
<tr>
<td>Paracosm</td>
<td>ours</td>
<td>40.48</td>
<td>57.80</td>
<td>33.17</td>
<td>55.18</td>
<td>42.58</td>
<td>64.20</td>
<td>38.74</td>
<td>59.06</td>
</tr>
</tbody>
</table>

### Key Result Observations
1. Paracosm outperforms all existing zero-shot CIR methods (both training-based and training-free) on all benchmarks, for both VLM backbones (ViT-L/14 and the stronger ViT-G/14). For example, on CIRCO with ViT-G/14, Paracosm achieves mAP@5 of 39.82, which is ~7.6 percentage points higher than the next best method CoTMR (32.23), a very large improvement.
2. Paracosm even outperforms some supervised CIR methods. For example, on Fashion IQ average R@10 with ViT-G/14, Paracosm achieves 38.74, which is higher than the supervised Combiner method's 35.39, an impressive result for a training-free method with no triplet annotations.
3. Qualitative results confirm the advantage of using mental images over text descriptions. Figure 3 from the original paper compares Paracosm to OSrCIR (which uses only text descriptions):

   ![Fig. 3: Comparison of qualitative results between OSrCIR \[49\] and our Paracosm. We show four examples from the CIRCO dataset \[1\] in the first column, followed by generated descriptions and top-4 retrievals by OSrCIR, and the mental images and top-4 retrievals by Paracosm. For each multimodal query, OSrCIR uses an LMM to generate a description, uses it to match database images, and returns top-ranked ones. Instead, Paracosm uses an LMM to generate a "mental image" for each query, which contains much richer information than a description, allowing image-to-image matching for better retrieval. Consequently, Paracosm yields better retrievals than OSrCIR.](images/3.jpg)
   *该图像是一个示意图，展示了多模态查询生成的描述与检索结果的对比。左侧展示了生成的描述和OSrCIR的前四个检索图像，而右侧展示了Paracosm生成的‘心智图像’与相应的前四个检索图像。通过图像到图像的匹配，Paracosm提供了更准确的检索结果。*

OSrCIR's text descriptions often miss fine-grained visual details, leading to incorrect retrievals, while Paracosm's mental image captures all relevant details, leading to correct retrievals.
4. Efficiency comparison results (Table 1 from original paper):
   The following are the results from Table 1 of the original paper:

   <table>
   <thead>
   <tr>
   <th>Method</th>
   <th>Offline time (hrs)</th>
   <th>Image storage (GB)</th>
   <th>Feature storage (GB)</th>
   <th>Inference GPU memory (GB)</th>
   <th>Inference latency (sec)</th>
   <th>CIRCO mAP@5</th>
   <th>CIRR R@1</th>
   <th>FashionIQ R@10</th>
   </tr>
   </thead>
   <tbody>
   <tr>
   <td>AutoCIR [12]</td>
   <td>0.1</td>
   <td>n/a</td>
   <td>0.38</td>
   <td>2.3</td>
   <td>24</td>
   <td>24.05</td>
   <td>31.81</td>
   <td>30.68</td>
   </tr>
   <tr>
   <td>CoTMR [44]</td>
   <td>0.1</td>
   <td>n/a</td>
   <td>0.38</td>
   <td>70.2</td>
   <td>1</td>
   <td>27.61</td>
   <td>35.02</td>
   <td>35.05</td>
   </tr>
   <tr>
   <td>OSSCR [49]</td>
   <td>0.1</td>
   <td>n/a</td>
   <td>0.38</td>
   <td>2.3</td>
   <td>3</td>
   <td>23.87</td>
   <td>29.45</td>
   <td>33.26</td>
   </tr>
   <tr>
   <td>Paracosm</td>
   <td>12.9</td>
   <td>41.4</td>
   <td>0.38</td>
   <td>2.7</td>
   <td>14</td>
   <td>37.40</td>
   <td>38.24</td>
   <td>36.45</td>
   </tr>
   </tbody>
   </table>

Observation: Paracosm has higher one-time offline preprocessing time (to generate synthetic database counterparts), but online inference latency is acceptable (14 seconds per query), and feature storage is identical to other methods (only 0.38 GB, since synthetic images are deleted after feature extraction). The large performance gain far outweighs the one-time offline cost.
## 6.2. Ablation Studies / Parameter Analysis
### Parameter Analysis for $\lambda$
$\lambda$ controls the weight of the mental image + query description vs. the original modification text in the query feature. The effect of $\lambda$ on performance is shown in Figure 6 from the original paper:

![Fig. 6: Analysis of $\\lambda$ which controls the importance of incorporating modification text in Eq. 1. We set $\\lambda = 0 . 3$ based on the results on the CIRR validation set. Interestingly, on all datasets, setting $\\lambda =$ 0.3 consistently yields the highest numeric metrics reported for all the datasets.](images/6.jpg)
*该图像是一个图表，展示了参数 `eta` 对不同检索性能指标的影响。横轴为 `eta` 值，纵轴为检索性能，数据从多个数据集收集。结果表明，在所有数据集中，设置 $eta = 0.3$ 时，检索性能达到最高。*

Observation: $\lambda=0.3$ yields the best performance across all datasets, so this value is used in all experiments.
### Parameter Analysis for $\beta$
$\beta$ controls the weight of the real database image vs. its synthetic counterpart in the database feature (in the extended formula `\phi^i = \beta V(I^i) + (1-\beta)V(I_{syn}^i)`). The effect of $\beta$ is shown in Figure 9 from the original paper:

![Fig. 9: Analysis of $\\beta$ . The parameter $\\beta$ balances the contributions of real database images $V ( I ^ { i } )$ and their synthetic counterparts $V ( I _ { \\mathrm { s y n } } ^ { i } )$ in Eq. 1. Based on studies on the CIRR and CIRCO validation sets using CLIP ViT-B/32, we set $\\beta \\ : = \\ : 0 . 5$ throughout our experiments, which not only yields competitive performance but also aligns with the principle of model simplicity by avoiding asymmetric weighting and reducing hyperparameter tuning.](images/9.jpg)
*该图像是一个图表，展示了参数 `eta` 对检索性能的影响。图中显示了 CIRR 和 CIRCO 测试集及验证集在不同 `eta` 值下的 R@1 和 mAP@5 结果，表明当 `eta` 设置为 0.5 时，性能最佳，符合模型简单性的原则。*

Observation: $\beta=0.5$ (equal weight for real and synthetic features) gives the best performance and simplifies the model, so it is used in all experiments.
### Comparison of Mental Image Generation Methods
The following are the results from Table 2 of the original paper, comparing different ways to generate the mental image:

<table>
<thead>
<tr>
<th>Method</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td>T2I Generation (generate description first, then generate image from text)</td>
<td>31.71</td>
<td>61.37</td>
<td>73.59</td>
<td>91.54</td>
</tr>
<tr>
<td>Image Edit w/ Qwen (directly edit reference image using mod text)</td>
<td>32.27</td>
<td>62.60</td>
<td>75.16</td>
<td>92.60</td>
</tr>
<tr>
<td>Image Edit w/ LongCat (another image editing model)</td>
<td>32.12</td>
<td>62.20</td>
<td>74.43</td>
<td>92.24</td>
</tr>
</tbody>
</table>

Observation: Directly editing the reference image (Paracosm's design) is better than text-to-image generation, as it preserves more details from the reference image. Paracosm also performs consistently well with different image editing models, showing robustness.
### Core Component Ablation Study
The following are the results from Table 5 of the original paper, ablating each core component of Paracosm:

<table>
<thead>
<tr>
<th colspan="3">multimodal query</th>
<th colspan="2">database images</th>
<th colspan="7">CIRR</th>
<th colspan="4">CIRCO</th>
</tr>
<tr>
<th rowspan="2">$t_{query}$</th>
<th rowspan="2">$I_{mental}$</th>
<th rowspan="2">$t_{mod}$</th>
<th rowspan="2">$I^i$</th>
<th rowspan="2">$I_{syn}^i$</th>
<th colspan="4">Recall@k</th>
<th colspan="3">Recallsubset@k</th>
<th colspan="4">mAP@k</th>
</tr>
<tr>
<th>k=1</th>
<th>k=5</th>
<th>k=10</th>
<th>k=50</th>
<th>k=1</th>
<th>k=2</th>
<th>k=3</th>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=50</th>
</tr>
</thead>
<tbody>
<tr>
<td>√</td>
<td></td>
<td></td>
<td>√</td>
<td></td>
<td>17.21</td>
<td>43.49</td>
<td>55.90</td>
<td>81.16</td>
<td>52.00</td>
<td>72.31</td>
<td>84.92</td>
<td>14.91</td>
<td>15.34</td>
<td>16.79</td>
<td>17.60</td>
</tr>
<tr>
<td>√</td>
<td>√</td>
<td></td>
<td>√</td>
<td></td>
<td>18.80</td>
<td>44.96</td>
<td>58.84</td>
<td>82.65</td>
<td>50.39</td>
<td>72.80</td>
<td>83.93</td>
<td>13.71</td>
<td>13.89</td>
<td>15.19</td>
<td>15.92</td>
</tr>
<tr>
<td>√</td>
<td>√</td>
<td>√</td>
<td>√</td>
<td></td>
<td>27.93</td>
<td>57.11</td>
<td>70.29</td>
<td>90.31</td>
<td>61.88</td>
<td>80.70</td>
<td>90.70</td>
<td>18.29</td>
<td>18.69</td>
<td>20.45</td>
<td>21.27</td>
</tr>
<tr>
<td>√</td>
<td></td>
<td></td>
<td>√</td>
<td>√</td>
<td>17.21</td>
<td>43.93</td>
<td>56.29</td>
<td>80.10</td>
<td>50.94</td>
<td>71.81</td>
<td>84.53</td>
<td>13.58</td>
<td>13.92</td>
<td>15.44</td>
<td>16.30</td>
</tr>
<tr>
<td>√</td>
<td>√</td>
<td></td>
<td>√</td>
<td>√</td>
<td>19.95</td>
<td>46.29</td>
<td>59.98</td>
<td>82.34</td>
<td>51.42</td>
<td>72.17</td>
<td>84.80</td>
<td>14.07</td>
<td>14.43</td>
<td>15.76</td>
<td>16.42</td>
</tr>
<tr>
<td>√</td>
<td>√</td>
<td>√</td>
<td>√</td>
<td>√</td>
<td>25.95</td>
<td>54.24</td>
<td>67.16</td>
<td>88.84</td>
<td>60.80</td>
<td>79.71</td>
<td>89.74</td>
<td>17.43</td>
<td>17.98</td>
<td>19.62</td>
<td>20.46</td>
</tr>
<tr>
<td></td>
<td>√</td>
<td></td>
<td>√</td>
<td>√</td>
<td>22.46</td>
<td>50.99</td>
<td>63.59</td>
<td>84.58</td>
<td>53.88</td>
<td>74.80</td>
<td>86.27</td>
<td>16.72</td>
<td>17.41</td>
<td>19.19</td>
<td>20.11</td>
</tr>
<tr>
<td></td>
<td>√</td>
<td>√</td>
<td>√</td>
<td>√</td>
<td>24.60</td>
<td>52.68</td>
<td>65.86</td>
<td>86.53</td>
<td>54.84</td>
<td>74.92</td>
<td>86.82</td>
<td>16.57</td>
<td>17.10</td>
<td>18.59</td>
<td>19.30</td>
</tr>
<tr>
<td>√</td>
<td>√</td>
<td>√</td>
<td>√</td>
<td>√</td>
<td>32.27</td>
<td>62.60</td>
<td>75.16</td>
<td>92.60</td>
<td>65.16</td>
<td>83.25</td>
<td>92.34</td>
<td>26.10</td>
<td>27.02</td>
<td>29.29</td>
<td>30.45</td>
</tr>
</tbody>
</table>

Key observations:
1. Adding the mental image $I_{mental}$ improves performance (e.g., CIRR R@1 increases from 17.21 to 18.80 when added to the text-only query baseline).
2. Adding the original modification text $t_{mod}$ gives a very large performance boost (e.g., CIRR R@1 increases from 18.80 to 27.93 when $t_{mod}$ is added to the query with $t_{query}$ and $I_{mental}$).
3. Adding synthetic database counterparts $I_{syn}^i$ gives another large improvement (e.g., CIRR R@1 increases from 27.93 to 32.27 when $I_{syn}^i$ is added).
4. All components combined yield the best performance, confirming each part contributes significantly to the final result.
### Failure Cases
Paracosm fails when the LMM generates incorrect mental images, as shown in Figure 7 from the original paper:

![Fig. 7: Failure cases on the CIRCO dataset. Paracosm can fail due to limitations of generative models that can generate implausible and counterfactual mental images. Four examples, respectively, demonstrate different failures of Paracosm. (1) It incorrectly generates a cartoon-style duck, making it fail to return the correct database image, which captures a plush toy duck. (2) It generates a counterfactual oven that has gas burners on its door, making it incorrectly retrieve an image that captures a burning oven. (3) It fails to edit the door color of the specified refrigerator and hence fails to return the correct target image. (4) It fails to comprehend the multimodal query, resulting in an incorrect mental image and hence failing to return the target image.](images/7.jpg)
*该图像是示意图，展示了在CIRCO数据集上Paracosm的失败案例。四个示例分别展示了生成的心理图像与真实目标图像的差异，导致错误的检索结果。. 例如，一个错误生成的卡通风格鸭子与真实的毛绒玩具鸭子不符。*

Common failure modes include: generating implausible/cartoonish objects, generating counterfactual objects (e.g., an oven with burners on the door), failing to perform the requested modification (e.g., not changing the color of a refrigerator), or failing to understand the modification text. These failures are caused by limitations in current LMM image editing capabilities, not the Paracosm pipeline itself.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper proposes Paracosm, the first fully training-free zero-shot CIR method that uses generated visual mental images for queries instead of relying exclusively on text descriptions. To resolve the synthetic-to-real domain gap between generated mental images and real database images, Paracosm generates synthetic counterparts for all database images, creating a unified synthetic "paracosm" space where matching is performed. Paracosm achieves state-of-the-art performance on all standard zero-shot CIR benchmarks, even outperforming some supervised CIR methods, with acceptable inference latency and only a one-time offline preprocessing cost.
## 7.2. Limitations & Future Work
### Stated Limitations
1. Performance is dependent on the quality of LMMs for image editing, captioning, and generation. Incorrectly generated mental images or synthetic database images lead to retrieval failures.
2. Current LMM-generated images often lack factual fidelity and fine-grained details, leading to avoidable failures.
3. Requires manual prompt tuning for different datasets, as modification text formats vary across benchmarks.
### Stated Future Work
1. Improve image generation/editing methods to produce more accurate, factually consistent mental images and synthetic database images.
2. Develop adaptive prompt generation methods that automatically create optimal prompts for different datasets and query types, eliminating the need for manual tuning.
3. Develop methods to detect and handle erroneously generated visuals to reduce failure cases.
## 7.3. Personal Insights & Critique
### Strengths
1. The core idea of constructing a unified synthetic domain to eliminate synthetic-to-real matching gaps is highly intuitive and effective, solving a long-standing problem in generative retrieval with a simple, elegant pipeline.
2. The fully training-free design makes Paracosm extremely easy to deploy in real-world applications, with no need for expensive data annotation or model training.
3. The performance gains are very large, especially on the challenging CIRCO dataset, clearly demonstrating the value of using visual query representations instead of only text.
### Potential Improvements
1. The offline preprocessing cost can be prohibitive for very large databases (e.g., millions of images), as each database image requires a synthetic counterpart to be generated. Future work can explore reducing this cost by using faster image generation models, or generating synthetic embeddings directly without rendering full images.
2. The current method uses fixed weights ($\lambda=0.3$, $\beta=0.5$) for all queries. Dynamic weight adjustment per query (e.g., higher weight for the mental image for visually complex modifications, higher weight for text for simple keyword modifications) could further improve performance.
3. The current pipeline uses simple cosine similarity for matching. More advanced multi-grained scoring methods that combine real and synthetic similarity scores separately could yield better results.
### Transferability
The core idea of creating a unified synthetic domain for matching can be applied to many other retrieval tasks, including video retrieval with text modifications, 3D model retrieval with multimodal queries, and cross-domain product retrieval. Any retrieval task with a domain gap between query representations and database representations can benefit from this approach.