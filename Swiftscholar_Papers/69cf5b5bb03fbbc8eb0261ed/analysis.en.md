# 1. Bibliographic Information
## 1.1. Title
Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark
## 1.2. Authors & Affiliations
- Yifei Deng: State Key Laboratory of Opto-Electronic Information Acquisition and Protection Technology, School of Computer Science and Technology, Anhui University
- Chenglong Li: State Key Laboratory of Opto-Electronic Information Acquisition and Protection Technology, Anhui University (corresponding author)
- Yuyang Zhang: The University of Hong Kong
- Guyue Hu: School of Artificial Intelligence, Anhui University
- Jin Tang: School of Computer Science and Technology, Anhui University

  All authors are primarily affiliated with Anhui University, with one collaborator from The University of Hong Kong, focusing on computer vision and cross-modal retrieval research.
## 1.3. Publication Venue
The paper is currently a preprint hosted on arXiv, and has not yet been formally published in a journal or conference.
## 1.4. Publication Date
Published on arXiv: 2026-03-21 UTC
## 1.5. Abstract
This paper studies the task of text-aerial person retrieval, which aims to identify target persons in UAV-captured aerial images from natural language descriptions, with applications in public security and intelligent transportation. Compared to ground-view text-image person retrieval, aerial images suffer from degraded visual information due to drastic viewpoint and altitude variations, making semantic alignment with text very challenging. To address this, the authors propose a novel Cross-modal Fuzzy Alignment Network that uses fuzzy logic to quantify token-level reliability for accurate fine-grained alignment, and incorporates ground-view images as a bridge agent to reduce the cross-modal gap between aerial images and text. Two core modules are designed: (1) Fuzzy Token Alignment, which uses fuzzy membership functions to dynamically model token-level association strength and suppress noisy/unobservable tokens, alleviating semantic inconsistency from missing visual cues; (2) Context-Aware Dynamic Alignment, which adaptively combines direct alignment and agent-assisted alignment using ground-view as a bridge to improve robustness. The authors also construct a new large-scale benchmark AERI-PEDES, which uses a chain-of-thought framework to generate high-quality, semantically consistent captions. Experiments on AERI-PEDES and existing TBAPR benchmark demonstrate the superiority of the proposed method.
## 1.6. Original Source Link
- Preprint link: https://arxiv.org/abs/2603.20721
- PDF link: https://arxiv.org/pdf/2603.20721
- Publication status: Preprint, working paper.

# 2. Executive Summary
## 2.1. Background & Motivation
- **Core Problem:** The paper aims to solve Text-Aerial Person Retrieval (TAPR), the task of retrieving a target person from a gallery of UAV-captured aerial images given a natural language description.
- **Importance:** Existing text-image person retrieval research focuses only on images from fixed ground-based cameras, which have limited coverage. UAVs can provide wide-area dynamic coverage, so extending retrieval to aerial images has high practical value for public security, intelligent surveillance, and search and rescue.
- **Key Challenges & Research Gaps:** Aerial images have severe degraded visual information, missing visual cues, and large viewpoint/altitude variations, leading to large semantic gaps between text descriptions and aerial images. Existing methods do not explicitly model the uncertainty of token-level alignment when some text attributes are not observable in the aerial image, and do not dynamically leverage available ground-view images of the same identity as a semantic bridge to reduce the alignment gap.
- **Innovative Entry Point:** The paper introduces fuzzy logic to quantify the reliability of each semantic token, suppressing alignment from unobservable/noisy tokens, and uses dynamic adaptive weighting to balance direct text-aerial alignment and ground-assisted bridge alignment per sample.
## 2.2. Main Contributions / Findings
1. **Methodological Innovation:** Propose the Cross-modal Fuzzy Alignment Network (CFAN), the first method to apply fuzzy logic to token-level alignment for TAPR, which explicitly handles uncertainty from missing visual cues.
2. **Core Modules:** Design two novel modules:
   - Context-Aware Dynamic Alignment (CDA): Adaptively balances direct alignment and ground-view bridge alignment based on per-sample alignment difficulty, improving alignment robustness.
   - Fuzzy Token Alignment (FTA): Uses fuzzy membership functions to model token reliability, suppressing noisy tokens and improving fine-grained alignment accuracy.
3. **New Benchmark:** Construct the largest TAPR benchmark to date, AERI-PEDES, with high-quality CoT-generated training captions and manually annotated test captions, which provides a standardized testbed for future TAPR research.
4. **Empirical Finding:** Extensive experiments show CFAN achieves new state-of-the-art performance on both AERI-PEDES and TBAPR, outperforming all existing baselines by a clear margin. Both proposed modules contribute significantly to performance improvements.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
- **Text-Image Person Retrieval (TIPR):** A cross-modal retrieval task where the input is a natural language description of a person, and the goal is to retrieve the matching person image from a gallery. It is widely used in surveillance scenarios where witnesses can provide descriptions instead of a target image.
- **Text-Aerial Person Retrieval (TAPR):** A variant of TIPR where all gallery images are captured by UAVs from aerial viewpoints, rather than fixed ground cameras.
- **Cross-modal Alignment:** The core step in cross-modal retrieval, where text and image features are mapped into a shared semantic embedding space, such that matching text-image pairs have high similarity and non-matching pairs have low similarity. Token-level alignment aligns fine-grained individual semantic tokens (e.g., "red shirt", "blue pants") instead of just aligning global whole-text/whole-image features.
- **Fuzzy Logic:** A logic paradigm that handles partial truth, where values range from 0 (completely unreliable/false) to 1 (completely reliable/true), unlike traditional binary logic. It is well-suited for modeling uncertainty and ambiguity in real-world data.
- **CLIP (Contrastive Language-Image Pre-training):** A widely used pre-trained vision-language model that learns aligned image and text embeddings from large-scale web data, used as a backbone for most modern cross-modal retrieval tasks.
- **Chain-of-Thought (CoT):** A prompting technique for large language models that decomposes a complex task into a sequence of intermediate reasoning steps, improving output accuracy and consistency.
## 3.2. Previous Works
### 3.2.1. Text-Image Person Retrieval
- Early work (2017-2019) focused on global feature alignment, which only aligned whole-text and whole-image features, missing fine-grained semantic details.
- Later work (2020-2022) shifted to local/part-based alignment, including part-aware matching, attribute alignment, and multi-scale feature matching, improving fine-grained alignment accuracy.
- Recent work (2022-present) uses pre-trained vision-language models like CLIP as backbones, with techniques like implicit relation reasoning, hierarchical distillation, and LLM-generated data for pre-training, improving generalization.
- The first work on TAPR was proposed by Wang et al. (2025) with the TBAPR benchmark and AEA-FIRM method, but it does not handle token-level uncertainty from missing visual cues, and does not dynamically adapt alignment based on sample difficulty.
### 3.2.2. Fuzzy Deep Learning
Fuzzy deep learning integrates fuzzy logic with deep learning to model uncertainty, and has been applied to medical image analysis and general cross-modal retrieval. However, prior work has not applied fuzzy logic to token-level reliability modeling for fine-grained alignment in TAPR.
## 3.3. Technological Evolution
The evolution of the field can be summarized as:
1. 2017: TIPR task introduced, global feature alignment paradigm.
2. 2020: Fine-grained part/local alignment becomes mainstream.
3. 2022: Pre-trained vision-language models become standard backbones for TIPR.
4. 2023: TAPR task (extension of TIPR to aerial UAV images) first proposed.
5. 2026: This work addresses the unique challenges of TAPR with fuzzy logic and dynamic bridge alignment, and releases a new large-scale benchmark.
## 3.4. Differentiation Analysis
Compared to prior work:
1. Unlike existing TAPR methods that assume all text tokens are observable in the aerial image, this paper explicitly models token-level reliability with fuzzy logic, suppressing wrong alignment from unobservable tokens.
2. Unlike prior work that uses fixed alignment strategies (either only direct alignment or always bridge alignment), this paper dynamically weights direct and bridge alignment per sample based on alignment difficulty, which is more flexible and adaptive.
3. This paper releases the largest TAPR benchmark to date, with higher quality captions than the existing TBAPR benchmark, which fills the gap of limited data resources for TAPR research.

# 4. Methodology
## 4.1. Core Principles
The method is designed around two key observations of TAPR:
1. Some semantic tokens in text descriptions are not observable in aerial images due to missing visual cues, so these tokens should not contribute to cross-modal alignment. Fuzzy logic is used to quantify token reliability, so only reliable tokens contribute to alignment.
2. For samples where direct text-aerial alignment is poor (e.g., high-altitude aerial images with very blurry visual cues), ground-view images of the same identity have clearer visual cues that align better with text, so they can be used as a semantic bridge. We dynamically adjust the weight of direct vs bridge alignment per sample, instead of using a fixed strategy.
## 4.2. Overall Architecture
The overall architecture of the proposed network is shown below:

![Figure 2. Overview of the proposed Cross-Modal Fuzzy Alignment Network.](images/2.jpg)
*该图像是示意图，展示了拟议的跨模态模糊对齐网络的整体架构，包括文本编码器、地面图像和航空图像的动态对齐模块，以及模糊标记对齐部分。该网络通过自注意力机制和逻辑与操作实现信息的有效整合。*

The architecture uses a shared CLIP image encoder to extract features for aerial images ($A$) and ground-view images ($G$), and a CLIP text encoder to extract text description features ($T$). It has two core modules:
1. Context-Aware Dynamic Alignment (CDA): Works on global features to compute adaptive weights for direct and bridge alignment.
2. Fuzzy Token Alignment (FTA): Works on token-level features to do robust fine-grained alignment, using fuzzy membership to suppress unreliable tokens.

   The total training loss is the sum of the CDA loss and FTA loss.
## 4.3. Context-Aware Dynamic Alignment (CDA) Module
### 4.3.1. Input
The module takes three global features as input:
- Global text features $\mathbf{T}^C \in \mathbb{R}^{B \times D}$
- Global aerial image features $A^C \in \mathbb{R}^{B \times D}$
- Global ground image features $G^C \in \mathbb{R}^{B \times D}$
  where $B$ is the batch size, and $D$ is the feature dimension.
### 4.3.2. Step 1: Compute Similarity Difference
For each sample $i$ ($i=1,...,B$), compute the difference between direct text-aerial similarity and text-ground similarity:
$$
\Delta _ { i } = \cos ( \mathbf { T _ { i } ^ { C } } , \mathbf { A _ { i } ^ { C } } ) - \cos ( \mathbf { T _ { i } ^ { C } } , \mathbf { G _ { i } ^ { C } } )
$$
Where $\cos(\cdot, \cdot)$ is cosine similarity, ranging from -1 to 1.
- If $\Delta_i > 0$: Direct text-aerial similarity is higher than text-ground similarity, so direct alignment is sufficient.
- If $\Delta_i < 0$: Direct text-aerial similarity is lower than text-ground similarity, so we need to rely more on the ground bridge.
### 4.3.3. Step 2: Compute Dynamic Weighting Coefficient
Map $\Delta_i$ to a soft weight $\alpha_i \in [0,1]$ using the sigmoid function:
$$
\alpha _ { i } = \frac { 1 } { 1 + \exp \Big [ - k \cdot \Delta _ { i } \Big ] }
$$
Where $k$ is a sensitivity hyperparameter that controls the steepness of the mapping.
- If direct alignment is good: $\alpha_i \approx 1$, so we weight direct alignment more.
- If direct alignment is bad: $\alpha_i \approx 0$, so we weight bridge alignment more.
### 4.3.4. Step 3: Compute CDA Loss
The CDA loss is a weighted average of direct alignment loss and bridge alignment loss:
$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { CDA } } = \displaystyle \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \Big [ \alpha _ { i } \cdot \mathcal { L } _ { \mathrm { d i r e c t } } \big ( \mathbf { T } _ { \mathrm { i } } ^ { \mathrm { C } } , \mathbf { A } _ { \mathrm { i } } ^ { \mathrm { C } } \big ) + } \\ & { \big ( 1 - \alpha _ { i } \big ) \cdot \mathcal { L } _ { \mathrm { b r i d g e } } \big ( \mathbf { T } _ { \mathrm { i } } ^ { \mathrm { C } } , \mathbf { G } _ { \mathrm { i } } ^ { \mathrm { C } } , \mathbf { A } _ { \mathrm { i } } ^ { \mathrm { C } } \big ) \Big ] } \end{array}
$$
The direct alignment loss uses the standard Similarity Distribution Matching (SDM) loss:
$$
\mathcal { L } _ { \mathrm { d i r e c t } } ( \mathbf { T _ { i } ^ { C } } , \mathbf { A _ { i } ^ { C } } ) = \mathrm { SDM } ( \mathbf { T _ { i } ^ { C } } , \mathbf { A _ { i } ^ { C } } )
$$
The bridge alignment loss aligns text to ground, then aligns ground to aerial, with stop-gradient on ground features to avoid updating ground representations:
$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { b r i d g e } } ( \mathbf { T } _ { \mathrm { i } } ^ { \mathrm { C } } , \mathbf { G } _ { \mathrm { i } } ^ { \mathrm { C } } , \mathbf { A } _ { \mathrm { i } } ^ { \mathrm { C } } ) = \mathrm { SDM } ( \mathbf { T _ { i } ^ { C } } , \mathbf { G _ { i } ^ { C } } ) + } \\ { \mathrm { SDM } ( sg ( \mathbf { G } _ { \mathrm { i } } ^ { \mathrm { C } } ) , \mathbf { A } _ { \mathrm { i } } ^ { \mathrm { C } } ) } \end{array}
$$
Where $sg(\cdot)$ is the stop-gradient operator, which prevents gradients from updating ground features. This ensures ground features act as a fixed semantic bridge, and only aerial features are updated to align with the bridge. When ground images are not available, CDA loss automatically degenerates to standard direct SDM loss.
## 4.4. Fuzzy Token Alignment (FTA) Module
### 4.4.1. Core Idea
Each semantic token has a reliability (membership degree) between 0 and 1, indicating how reliably it is observed in both text and image. Only tokens with high reliability in both modalities contribute to alignment, while unreliable/noisy tokens are suppressed.
### 4.4.2. Step 1: Cross-Modal Query Projection
Input:
- Text features $\mathbf{T} \in \mathbb{R}^{B \times N_t \times D}$ ($N_t$ = number of text tokens)
- Aerial image features $\mathbf{A} \in \mathbb{R}^{B \times N_a \times D}$ ($N_a$ = number of image patch tokens)

  We use $K$ shared learnable semantic query tokens $\mathbf{Q} \in \mathbb{R}^{K \times D}$, and project them into modality-aware representations via CrossFormer (a cross-attention based interaction module):
$$
\mathbf{Q}_a = CrossFormer(\mathbf{Q}, \mathbf{A}, \mathbf{A})
$$
$$
\mathbf{Q}_t = CrossFormer(\mathbf{Q}, \mathbf{T}, \mathbf{T})
$$
CrossFormer consists of a cross-attention layer (queries = learnable $Q$, keys/values = input modality features) followed by self-attention and feed-forward layers. The output $\mathbf{Q}_a$ (image queries) and $\mathbf{Q}_t$ (text queries) are both of shape $B \times K \times D$, where each of the $K$ tokens corresponds to a semantic concept aligned across both modalities.
### 4.4.3. Step 2: Compute Fuzzy Membership for Image Tokens
For each aerial image sample, we predict an adaptive Gaussian scale $\sigma$ from the global image class token $A^C$ via an MLP:
$$
\log \pmb { \sigma } = \mathrm { MLP } ( \mathbf { A } ^ { \mathbf { C } } ), \quad \pmb { \sigma } = \exp ( \log \pmb { \sigma } )
$$
Log is used to ensure $\sigma$ is always positive, as required for the Gaussian function. For each query token $j$, compute the cosine similarity between the query token and the global class token, then map it to membership degree via Gaussian function:
$$
r _ { j } = \frac { Q_a^{(j)} \cdot A^C } { |Q_a^{(j)}|_2 |A^C|_2 }, \quad \mu _ { j } ^ { a } = \exp \Big ( - \frac { (1 - r_j)^2 } { 2 \sigma^2 } \Big )
$$
Where $\mu_j^a$ is the membership degree (reliability) of token $j$ in the image modality. If the token is well-aligned with the global semantic ($r_j \approx 1$), $\mu_j^a \approx 1$ (high reliability). If the token is misaligned, $\mu_j^a$ is close to 0 (low reliability).
### 4.4.4. Step 3: Compute Membership for Text Tokens
Membership for text tokens is computed exactly the same way as for image tokens, resulting in $\mu_j^t$, the membership degree of token $j$ in the text modality.
### 4.4.5. Step 4: Fuse Joint Membership
We fuse the membership from both modalities using a fuzzy logic AND operation: a token is only reliable if it is reliable in both modalities, so we multiply the two memberships:
$$
\mu _ { j } ^ { \mathrm { joint } } = \mu _ { j } ^ { a } \cdot \mu _ { j } ^ { t }
$$
Multiplication is used as a differentiable soft AND operation. If either membership is low, the joint membership is low, so the token's contribution is suppressed.
### 4.4.6. Step 5: Compute Weighted Similarity and FTA Loss
First, compute the cosine similarity between the $j$-th image query and text query, then weight by joint membership to get the overall similarity:
$$
s _ { j } = \frac { { Q_a^{(j)} } ^ { \top } Q_t^{(j)} } { \vert Q_a^{(j)} \vert _ { 2 } \vert Q_t^{(j)} \vert _ { 2 } }, \quad \cos ( Q_a, Q_t ) = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \mu _ { j } ^ { \mathrm { j o i n t } } s _ { j }
$$
Then compute the FTA loss using bidirectional similarity distribution matching:
$$
p _ { i , j } = \frac { \exp ( \cos ( Q_a^i , Q_t^j ) / \tau ) } { \sum _ { k = 1 } ^ { N } \exp ( \cos ( Q_a^i , Q_t^k ) / \tau ) }, \quad q _ { i , j } = \frac { y _ { i , j } } { \sum _ { k = 1 } ^ { N } y _ { i , k } }
$$
$$
\mathcal { L } _ { \mathrm { FTA } } = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \left[ p _ { i , j } \log \frac { p _ { i , j } } { q _ { i , j } + \epsilon } + p _ { j , i } \log \frac { p _ { j , i } } { q _ { j , i } + \epsilon } \right]
$$
Where:
- $N$ = number of samples in the batch, $\tau$ = temperature hyperparameter, $\epsilon$ = small constant to avoid division by zero.
- $p_{i,j}$ = model predicted probability that the $i$-th aerial sample matches the $j$-th text query.
- $q_{i,j}$ = ground-truth probability, where $y_{i,j}=1$ if $i$ and $j$ are a matching pair, 0 otherwise.
- The loss is bidirectional KL divergence between the predicted and ground-truth similarity distributions, for both text-to-image and image-to-text retrieval.

# 5. Experimental Setup
## 5.1. Datasets
The authors evaluate on two TAPR benchmarks:
### 5.1.1. AERI-PEDES (Proposed in this paper)
- Source: Collected from three existing aerial-ground person re-identification datasets, with each identity having images from both aerial and ground viewpoints.
- Scale: 4,659 total identities. Training set: 3,659 identities, 112,672 aerial images, 26,351 ground images, 26,213 CoT-generated captions. Test set: 1,000 identities, 5,525 aerial images, 6,141 manually annotated captions. Total images: 144,548.
- Characteristics: It is the largest TAPR benchmark to date. Captions have an average length of 38.6 words, maximum 105 words. Training captions are CoT-generated to reduce annotation cost, while test captions are manually annotated to reflect real-world usage.
- Example samples from AERI-PEDES are shown below:

  ![该图像是图表和插图的组合，左侧部分展示了不同高度下的人物图像及其描述，包括高和低高度的实例；右侧部分展示了三个不同模型（AERI-PEDES、CUHK-PEDES 和 RSTPReid）的性能比较图，表现为三条曲线，这些曲线显示了模型在特定任务下的表现。图像整体用于说明研究中不同条件和模型的影响。](images/3.jpg)
  *该图像是图表和插图的组合，左侧部分展示了不同高度下的人物图像及其描述，包括高和低高度的实例；右侧部分展示了三个不同模型（AERI-PEDES、CUHK-PEDES 和 RSTPReid）的性能比较图，表现为三条曲线，这些曲线显示了模型在特定任务下的表现。图像整体用于说明研究中不同条件和模型的影响。*

### 5.1.2. TBAPR (Existing benchmark)
- Source: First TAPR benchmark, proposed by Wang et al. 2025.
- Scale: 65,880 total images, 1,180 training identities, 529 test identities.
- Characteristics: Contains many low-altitude aerial images, so the visual difference between aerial and ground images is smaller than in AERI-PEDES.
## 5.2. Evaluation Metrics
All metrics are standard for cross-modal person retrieval:
### 5.2.1. Rank-k Accuracy
**Conceptual Definition:** Rank-k accuracy measures the proportion of queries where the correct matching image is ranked in the top-$k$ positions of the result list. Rank-1 (proportion of correct matches at first position) is the most commonly used metric.
**Formula:**
$$
\text{Rank-}k = \left( \frac{\text{Number of queries with correct match in top-}k}{\text{Total number of queries}} \right) \times 100\%
$$
**Interpretation:** Higher Rank-k = better performance. The authors report Rank-1, Rank-5, and Rank-10.
### 5.2.2. Mean Average Precision (mAP)
**Conceptual Definition:** mAP measures the overall quality of the entire ranked result list, accounting for both precision and recall. It penalizes methods where correct matches are ranked low even if they appear in the top-$k$.
**Formula:**
$$
AP(q) = \frac{1}{R_q} \sum_{k=1}^{G} P(k) \cdot rel(k), \quad mAP = \frac{1}{Q} \sum_{q=1}^{Q} AP(q)
$$
Where:
- $R_q$ = number of correct matches for query $q$, $G$ = total number of images in the gallery, $Q$ = total number of queries.
- `P(k)` = precision at position $k$, $rel(k) = 1$ if the image at position $k$ is a correct match, 0 otherwise.
  **Interpretation:** Higher mAP = better overall retrieval performance.
### 5.2.3. RSum (Sum of Ranks)
**Conceptual Definition:** RSum is the sum of Rank-1, Rank-5, and Rank-10 accuracy, used to give an overall summary of performance across all rank thresholds.
**Formula:**
$$
\text{RSum} = \text{Rank-1} + \text{Rank-5} + \text{Rank-10}
$$
**Interpretation:** Higher RSum = better overall performance.
## 5.3. Baselines
The authors compare with 12 recent state-of-the-art methods, including general TIPR methods and the existing TAPR-specific method: IRRA (CVPR2023), APTM (MM2023), RDE (CVPR2024), CFAM (CVPR2024), NAM (CVPR2024), VFE (KBS2025), DM-Adpeter (AAAI2025), LPNC (TIFS2025), LPNC+Pretrain (TIFS2025), AEA-FIRM (TCSVT2025), AEA-FIRM+Pretrain (TCSVT2025), HAM (CVPR2025). This is a comprehensive set of baselines covering all recent state-of-the-art approaches.
## 5.4. Implementation Details
- Backbone: CLIP-based image and text encoders, initialized with public pre-trained weights.
- Image preprocessing: Standard augmentation, resized to $384 \times 128$ (standard for person retrieval).
- FTA query tokens: 4 learnable 512-dimensional tokens.
- Optimization: Adam optimizer, 60 epochs, initial learning rate $5 \times 10^{-6}$, cosine decay schedule, batch size 64, trained on a single RTX 4090 GPU.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The full comparison results are shown in the table below:
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Ref</th>
<th colspan="5">AERI-PEDES</th>
<th colspan="5">TBAPR</th>
</tr>
<tr>
<th>Rank-1</th>
<th>Rank-5</th>
<th>Rank-10</th>
<th>mAP</th>
<th>RSum</th>
<th>Rank-1</th>
<th>Rank-5</th>
<th>Rank-10</th>
<th>mAP</th>
<th>RSum</th>
</tr>
</thead>
<tbody>
<tr>
<td>IRRA [13]</td>
<td>CVPR23</td>
<td>35.14</td>
<td>53.23</td>
<td>63.19</td>
<td>33.42</td>
<td>151.57</td>
<td>39.63</td>
<td>58.72</td>
<td>67.69</td>
<td>35.31</td>
<td>166.04</td>
</tr>
<tr>
<td>APTM [44]</td>
<td>MM23</td>
<td>34.62</td>
<td>53.95</td>
<td>64.5</td>
<td>31.09</td>
<td>153.07</td>
<td>43.59</td>
<td>62.03</td>
<td>69.75</td>
<td>38.71</td>
<td>175.37</td>
</tr>
<tr>
<td>RDE [32]</td>
<td>CVPR24</td>
<td>38.56</td>
<td>58.26</td>
<td>67.89</td>
<td>37.16</td>
<td>164.71</td>
<td>37.31</td>
<td>54.06</td>
<td>60.75</td>
<td>32.17</td>
<td>152.12</td>
</tr>
<tr>
<td>CFAM [49]</td>
<td>CVPR24</td>
<td>30.77</td>
<td>51.37</td>
<td>61.61</td>
<td>30.4</td>
<td>143.75</td>
<td>48.34</td>
<td>66.31</td>
<td>73.21</td>
<td>42.67</td>
<td>184.78</td>
</tr>
<tr>
<td>NAM [36]</td>
<td>CVPR24</td>
<td>42.47</td>
<td>61.72</td>
<td>69.99</td>
<td>40.22</td>
<td>174.17</td>
<td>46.56</td>
<td>63.13</td>
<td>70.13</td>
<td>40.92</td>
<td>179.82</td>
</tr>
<tr>
<td>VFE [34]</td>
<td>KBS25</td>
<td>35.76</td>
<td>55.35</td>
<td>65.56</td>
<td>35.35</td>
<td>156.67</td>
<td>47.94</td>
<td>62.5</td>
<td>68.17</td>
<td>42.18</td>
<td>178.63</td>
</tr>
<tr>
<td>DM-Adpeter [25]</td>
<td>AAAI25</td>
<td>33.42</td>
<td>53.17</td>
<td>62.79</td>
<td>32.41</td>
<td>149.37</td>
<td>37.81</td>
<td>58.34</td>
<td>66.56</td>
<td>33.28</td>
<td>162.71</td>
</tr>
<tr>
<td>LPNC [37]</td>
<td>TIFS25</td>
<td>35.65</td>
<td>53.61</td>
<td>63.69</td>
<td>35.19</td>
<td>152.95</td>
<td>41.78</td>
<td>58.03</td>
<td>65.50</td>
<td>37.87</td>
<td>165.31</td>
</tr>
<tr>
<td>LPNC+Pretrain [37]</td>
<td>TIFS25</td>
<td>43.79</td>
<td>61.49</td>
<td>70.40</td>
<td>42.22</td>
<td>175.68</td>
<td>45.41</td>
<td>62.31</td>
<td>69.94</td>
<td>42.17</td>
<td>177.66</td>
</tr>
<tr>
<td>AEA-FIRM [38]</td>
<td>TCSVT25</td>
<td>37.94</td>
<td>56.66</td>
<td>65.71</td>
<td>34.89</td>
<td>160.31</td>
<td>44.75</td>
<td>62.38</td>
<td>69.13</td>
<td>36.28</td>
<td>176.26</td>
</tr>
<tr>
<td>AEA-FIRM+Pretrain [14]</td>
<td>TCSVT25</td>
<td>44.42</td>
<td>61.96</td>
<td>71.03</td>
<td>41.12</td>
<td>177.41</td>
<td>48.15</td>
<td>63.87</td>
<td>71.21</td>
<td>42.01</td>
<td>183.23</td>
</tr>
<tr>
<td>HAM [14]</td>
<td>CVPR25</td>
<td>44.58</td>
<td>63.52</td>
<td>72.67</td>
<td>42.45</td>
<td>180.77</td>
<td>47.81</td>
<td>64.96</td>
<td>72.53</td>
<td>41.86</td>
<td>185.30</td>
</tr>
<tr>
<td>Ours (W/O Ground)</td>
<td>-</td>
<td>45.06</td>
<td>64.53</td>
<td>73.21</td>
<td>43.27</td>
<td>182.80</td>
<td>49.15</td>
<td>65.88</td>
<td>73.47</td>
<td>42.89</td>
<td>188.50</td>
</tr>
<tr>
<td>Ours (With Ground)</td>
<td>-</td>
<td>47.16</td>
<td>65.66</td>
<td>73.83</td>
<td>44.79</td>
<td>186.65</td>
<td>49.47</td>
<td>66.50</td>
<td>73.06</td>
<td>43.96</td>
<td>189.03</td>
</tr>
</tbody>
</table>

### Analysis of Results:
1. **AERI-PEDES:** The proposed method outperforms all baselines by a clear margin. Even without ground images, it already achieves 45.06% Rank-1, higher than the previous best HAM's 44.58%. With ground bridge, it achieves 47.16% Rank-1 and 186.65 RSum, a ~6% RSum improvement over the previous state-of-the-art. This confirms CFAN effectively handles large viewpoint variations and missing visual cues in challenging AERI-PEDES.
2. **TBAPR:** CFAN also outperforms all baselines, achieving 49.47% Rank-1 and 189.03 RSum, higher than the previous best CFAM's 184.78 RSum. The improvement from adding ground bridge is smaller here because TBAPR has many low-altitude aerial images with clear visual cues, so direct alignment is already good, but CFAN still gains from adaptive alignment.
## 6.2. Ablation Studies
### 6.2.1. Core Component Ablation
Ablation on AERI-PEDES to test the contribution of each module:
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Methods</th>
<th>Rank-1</th>
<th>Rank-5</th>
<th>Rank-10</th>
<th>mAP</th>
<th>RSum</th>
</tr>
</thead>
<tbody>
<tr>
<td>Baseline (fixed bridge, no CDA/FTA)</td>
<td>43.88</td>
<td>61.11</td>
<td>69.85</td>
<td>41.58</td>
<td>174.84</td>
</tr>
<tr>
<td>+ CDA</td>
<td>46.18</td>
<td>64.40</td>
<td>72.46</td>
<td>43.98</td>
<td>183.04</td>
</tr>
<tr>
<td>+ FTA</td>
<td>44.55</td>
<td>61.77</td>
<td>70.32</td>
<td>41.89</td>
<td>176.64</td>
</tr>
<tr>
<td>+ CDA + FTA (full model)</td>
<td>47.16</td>
<td>65.66</td>
<td>73.83</td>
<td>44.79</td>
<td>186.65</td>
</tr>
</tbody>
</table>

Analysis:
- Adding CDA to baseline improves RSum by 8.2 points, confirming that dynamic adaptive weighting is far more effective than fixed bridge alignment.
- Adding FTA to baseline improves RSum by 1.8 points, and adding FTA on top of CDA gives an additional 3.61 point RSum improvement, confirming fuzzy token alignment effectively suppresses noisy tokens and improves fine-grained alignment.
- Both modules contribute positively, and combining them gives the best performance.
### 6.2.2. Bridge Modality Ablation
Ablation to test different bridge modalities:
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>Bridge</th>
<th>Rank-1</th>
<th>Rank-5</th>
<th>Rank-10</th>
<th>mAP</th>
<th>RSum</th>
</tr>
</thead>
<tbody>
<tr>
<td>None (no bridge)</td>
<td>45.06</td>
<td>64.53</td>
<td>73.21</td>
<td>43.27</td>
<td>182.80</td>
</tr>
<tr>
<td>Aerial (low-altitude aerial as bridge)</td>
<td>46.08</td>
<td>64.60</td>
<td>73.33</td>
<td>44.20</td>
<td>184.01</td>
</tr>
<tr>
<td>Ground (ground-view as bridge)</td>
<td>47.16</td>
<td>65.66</td>
<td>73.83</td>
<td>44.79</td>
<td>186.65</td>
</tr>
</tbody>
</table>

Analysis:
- Any bridge improves performance over no bridge, confirming intermediate bridges help reduce the cross-modal gap.
- Ground-view bridge gives the best performance, because ground images have the clearest visual cues that align best with text.
- CFAN can flexibly use different bridge modalities when ground is not available, and still achieve performance gains.
## 6.3. Parameter Sensitivity Analysis
### 6.3.1. Number of Learnable Query Tokens
The effect of query token count is shown below:

![Figure 5. Impact of the Number of Learnable Query Tokens.](images/5.jpg)
*该图像是图表，展示了不同学习查询令牌数量对 RSum 和 mAP 的影响，其中 RSum 用蓝线表示，mAP 用橙线表示。可以观察到 RSUM 在 2 和 4 之间达到最大值 186.65，而 mAP 的变化较小，最高值为 47。*

Analysis: Performance is relatively stable across different token counts. mAP changes very little, and RSum peaks at 4 tokens. When token count exceeds 4, performance degrades due to token redundancy and over-parameterization. So 4 tokens is optimal, which is what the authors use in the final model.
### 6.3.2. Sensitivity of hyperparameter $k$ (CDA)
The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th>k</th>
<th>1</th>
<th>2</th>
<th>4</th>
<th>8</th>
<th>12</th>
<th>16</th>
</tr>
</thead>
<tbody>
<tr>
<td>Rank-1</td>
<td>47.16</td>
<td>46.91</td>
<td>47.04</td>
<td>46.77</td>
<td>46.23</td>
<td>46.04</td>
</tr>
<tr>
<td>mAP</td>
<td>44.79</td>
<td>44.58</td>
<td>44.50</td>
<td>43.83</td>
<td>43.70</td>
<td>43.55</td>
</tr>
<tr>
<td>RSum</td>
<td>186.65</td>
<td>185.88</td>
<td>185.91</td>
<td>184.11</td>
<td>183.99</td>
<td>183.31</td>
</tr>
</tbody>
</table>

Analysis: Best performance is achieved at $k=1$. As $k$ increases, performance gradually degrades. This is because larger $k$ makes the sigmoid function too steep, leading to hard binary weighting (either 0 or 1) instead of soft adaptive weighting, which loses the benefit of dynamic adjustment. So $k=1$ is optimal.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper addresses the unique challenges of text-aerial person retrieval, where aerial images have degraded visual information and missing visual cues that make semantic alignment with text very challenging. The authors propose the Cross-modal Fuzzy Alignment Network (CFAN), which introduces fuzzy logic to quantify token-level reliability, suppressing alignment from unobservable/noisy tokens, and uses a dynamic context-aware alignment module that adaptively balances direct text-aerial alignment and ground-view bridge alignment. The authors also construct AERI-PEDES, the largest high-quality TAPR benchmark to date, with CoT-generated training captions and manual test captions. Extensive experiments on two benchmarks show CFAN achieves new state-of-the-art performance, with both core modules contributing significantly to performance improvements.
## 7.2. Limitations & Future Work
Implicit limitations from the paper:
1. The bridge module requires ground-view images of the same identity to achieve maximum performance, which may not always be available in all real-world application scenarios (though CFAN can still work without ground, with only slightly lower performance).
2. Fuzzy membership is calculated based on similarity to the global class token, which may not always accurately capture reliability for very fine-grained attributes.
   Potential future directions:
- Extend the fuzzy alignment framework to other cross-modal tasks with high uncertainty, such as low-resolution person retrieval, cross-view person re-identification, and cross-modal medical retrieval.
- Explore unsupervised/zero-shot TAPR, where no labeled aerial data is required for training.
- Develop lightweight versions of CFAN for deployment on edge UAV devices.
- Improve the caption generation pipeline to further reduce manual annotation cost while maintaining caption quality.
## 7.3. Personal Insights & Critique
- **Key Strengths:** This paper makes two very novel contributions: first, applying fuzzy logic to model token-level uncertainty for fine-grained cross-modal alignment is a very fitting and effective solution to the problem of missing visual cues in aerial images. Second, the dynamic adaptive alignment strategy is more flexible than fixed alignment approaches, and the new AERI-PEDES benchmark fills an important data gap in the emerging TAPR field, which will greatly facilitate future research.
- **Transferability:** The core ideas of fuzzy token alignment and dynamic bridge alignment can be easily transferred to other cross-modal tasks that suffer from missing or noisy information, not just TAPR.
- **Potential Improvements:** One possible improvement is to learn the fuzzy membership function end-to-end instead of using a fixed Gaussian function, which may further improve adaptability to different data distributions. Another direction is to explore domain adaptation from ground-view TIPR data to aerial TAPR, which would reduce the need for large-scale labeled aerial training data.