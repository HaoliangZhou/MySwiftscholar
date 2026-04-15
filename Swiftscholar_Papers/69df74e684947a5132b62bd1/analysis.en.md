# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is "Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval." This title highlights the shift from traditional semantic-level image retrieval to a more precise, instance-level retrieval task that utilizes referential anchoring (bounding boxes) to ensure consistency.

## 1.2. Authors
The authors are Yuxin Yang, Yinan Zhou, Yuxin Chen, Ziqi Zhang, Zongyang Ma, Chunfeng Yuan, Bing Li, Jun Gao, and Weiming Hu. They are affiliated with a mix of academic and industrial institutions, including the Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences; Xi'an Jiaotong University; Tencent Inc.; PeopleAI Inc.; HelloGroup Inc.; and ShanghaiTech University. This suggests a collaborative effort combining academic rigor with industrial-scale data and application needs.

## 1.3. Journal/Conference
The paper is published at **CVPR 2026** (IEEE/CVF Conference on Computer Vision and Pattern Recognition). CVPR is one of the most prestigious and influential conferences in the field of computer vision, known for setting state-of-the-art benchmarks in image recognition, retrieval, and generation.

## 1.4. Publication Year
The paper was published in **2026**.

## 1.5. Abstract
The paper addresses the limitation of Composed Image Retrieval (CIR), which often prioritizes broad semantic matching over specific instance fidelity. The authors propose a new task called **Object-Anchored Composed Image Retrieval (OACIR)**, which mandates strict instance-level consistency by using a bounding box to anchor the object in the reference image. To support this, they introduce **OACIRR**, a large-scale, multi-domain benchmark with over 160K quadruples and challenging distractor galleries. They also propose **AdaFocal**, a framework featuring a Context-Aware Attention Modulator that dynamically balances focus between the anchored instance and the broader context. Experiments show AdaFocal significantly outperforms existing models in maintaining instance-level fidelity.

## 1.6. Original Source Link
The paper is available on arXiv as a preprint (or published version) with the ID `2604.05393`.
Link: https://arxiv.org/abs/2604.05393
PDF Link: https://arxiv.org/pdf/2604.05393

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the inherent limitation of standard Composed Image Retrieval (CIR) systems. While CIR allows flexible queries combining a reference image and modification text (e.g., "find this dress but in a different color"), it primarily operates at a **semantic level**. This means it retrieves images that are semantically similar (same category, same style) but often fails to retrieve the exact same **instance** (the specific physical object) across different contexts or backgrounds. In real-world applications like digital memory retrieval or identity tracing, preserving the exact instance identity is often more critical than just finding a visually similar item. The gap in prior research is the lack of mechanisms and datasets that enforce strict **instance-level consistency** within a compositional retrieval framework.

## 2.2. Main Contributions / Findings
The paper makes three primary contributions:
1.  **Task Definition:** It proposes the novel **Object-Anchored Composed Image Retrieval (OACIR)** task, which extends the standard CIR query by adding a bounding box to visually anchor the specific instance in the reference image.
2.  **Benchmark Construction:** It constructs **OACIRR**, the first large-scale, multi-domain benchmark (over 160K real-world quadruples) specifically designed for OACIR. It includes rigorous evaluation protocols with "hard-negative" distractors to test fine-grained discrimination.
3.  **Methodology:** It proposes **AdaFocal**, a new framework that uses a **Context-Aware Attention Modulator (CAAM)**. This module dynamically predicts a modulation scalar to intensify visual attention on the anchored instance region during feature fusion, balancing instance preservation with the understanding of the modification text.

    The key findings demonstrate that existing semantic-level retrieval models struggle significantly with this task. The AdaFocal framework, combined with the OACIRR dataset, establishes a robust baseline, substantially outperforming existing methods in instance recall metrics.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several fundamental concepts in computer vision and multimodal learning:

*   **Composed Image Retrieval (CIR):** A retrieval paradigm where the query consists of two parts: a reference image ($I_r$) and a modification text ($T_m$). The goal is to retrieve a target image ($I_t$) that visually represents the reference image modified according to the text. For example, showing a picture of a "red car" and asking for "blue car" should retrieve an image of that specific car model but in blue.
*   **Instance-Level vs. Semantic-Level Matching:** Semantic matching focuses on category or attribute similarity (e.g., any "spotted dog"). Instance-level matching requires identifying the exact same physical entity (e.g., "this specific dog named Spot").
*   **Vision-Language Pre-training (VLP):** Models like CLIP or BLIP that are trained on massive pairs of images and text to learn a shared embedding space where visual and textual concepts can be compared. These often serve as the backbone for retrieval systems.
*   **Cross-Attention:** A mechanism from the Transformer architecture where queries from one modality (e.g., text) attend to keys and values from another modality (e.g., image patches) to fuse information. The standard formula for scaled dot-product attention is:
    $$
    \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    where $Q$ is the query matrix, $K$ is the key matrix, $V$ is the value matrix, and $d_k$ is the dimension of the keys.
*   **Contrastive Learning:** A training technique where the model learns to pull similar items (positive pairs) closer in the embedding space and push dissimilar items (negative pairs) apart. A common loss function is the InfoNCE loss.

## 3.2. Previous Works
The paper summarizes prior work into two main categories:
1.  **Composed Image Retrieval:** This includes supervised methods (like CIRR, FashionIQ), Zero-Shot CIR (ZS-CIR) which converts images to pseudo-text (like Pic2Word), and methods using synthetic data. The authors note that these operate at the semantic level and lack instance fidelity.
2.  **Instance Consistency:** This includes Person Re-identification (ReID) tasks and methods using textual inversion (associating a visual concept with a learnable text token). The authors argue that ReID is too domain-specific (people only), and textual inversion requires per-instance optimization, which is not scalable.

## 3.3. Technological Evolution
The field has evolved from simple text-to-image or image-to-image retrieval to more complex Composed Image Retrieval (CIR) to handle relative queries. Initially, CIR focused on semantic alignment. Recently, there has been a push towards more fine-grained and personalized retrieval. This paper represents the next step: moving from "semantic composition" to "instance-aware composition," requiring explicit visual grounding (bounding boxes) rather than relying solely on textual descriptions.

## 3.4. Differentiation Analysis
Unlike standard CIR methods that treat the reference image as a global visual anchor, OACIR uses a **bounding box** to explicitly define the region of interest. Unlike Person ReID methods which are specialized for humans, OACIR is general-purpose (fashion, cars, products, landmarks). Unlike textual inversion methods that require fine-tuning for new instances, OACIR uses the bounding box as an inference-time visual prompt, making it flexible and scalable.

# 4. Methodology

## 4.1. Principles
The core principle of the **AdaFocal** framework is **adaptive visual attention**. The method posits that the degree of focus on the anchored instance should not be static but should depend dynamically on the semantic context of the query (the reference image and the modification text). For instance, if the modification text describes a background change ("in a snowy mountain"), the model needs to pay attention to the background context. If the text describes a subtle attribute change, the model needs to focus intensely on the instance. AdaFocal achieves this by learning to predict a modulation scalar that biases the attention mechanism towards the anchored region.

## 4.2. Core Methodology In-depth (Layer by Layer)

The AdaFocal framework consists of a **Multimodal Encoder** backbone and a specialized **Context-Aware Attention Modulator (CAAM)**. The process involves two parallel branches: a Query Branch and a Target Branch.

### Step 1: Contextual Perception (The CAAM)
The first step involves processing the query inputs—the reference image ($I_r$), the modification text ($T_m$), and the bounding box ($B_r$)—to determine how much focus should be placed on the instance.

The **Context-Aware Attention Modulator (CAAM)** takes the reference image and modification text as input. It passes them through a frozen Image Encoder ($\mathcal{E}_{\mathcal{T}}$) and a Text Tokenizer. These features are fed into the shared Multimodal Encoder ($\mathcal{E}_{\mathcal{M}}$).

Crucially, the CAAM introduces a set of $K$ learnable **Contextual Probe Tokens**, denoted as $\{ \mathsf{p}_k \}_{k=1}^K$. These tokens interact with the multimodal inputs to extract contextual cues. The output of the encoder, along with a learnable Contextual [CLS] Token, is processed by a **Contextual Reasoning Module (CRM)** (a Transformer-based aggregator). The CRM aggregates information to produce a final contextual representation.

This representation is then projected by a linear mapping layer, $\mathrm{Linear}_{\mathcal{C}}(\cdot)$, to output a single scalar value, $\beta$, which serves as the modulation signal for the specific query.

The formula for calculating the modulation scalar $\beta$ is:

$$
\beta = \operatorname{Linear}_{\boldsymbol{\mathcal{C}}}(\operatorname{CRM}(\mathcal{E}_{\boldsymbol{\mathcal{M}}}(\mathcal{E}_{\boldsymbol{\mathcal{T}}}(I_r), T_m, \{\mathfrak{p}_k\}))).
$$

**Symbol Explanation:**
*   $\beta$: The predicted modulation scalar. A higher value indicates a stronger need to focus on the anchored instance.
*   $\mathrm{Linear}_{\mathcal{C}}$: A linear projection layer that maps the contextual features to a scalar value.
*   $\mathrm{CRM}$: The Contextual Reasoning Module, responsible for aggregating features from the probe tokens and inputs.
*   $\mathcal{E}_{\mathcal{M}}$: The Multimodal Encoder (e.g., BLIP-2 Q-Former).
*   $\mathcal{E}_{\mathcal{T}}$: The frozen Image Encoder (e.g., ViT).
*   $I_r$: The reference image.
*   $T_m$: The modification text.
*   $\{\mathfrak{p}_k\}$: The set of learnable Contextual Probe Tokens.

### Step 2: Adaptive Focus (Attention Activation)
Once the modulation scalar $\beta$ is determined, it is used to modulate the attention mechanism within the Query Branch of the Multimodal Encoder.

The Multimodal Encoder performs feature fusion via cross-attention. It has $M$ learnable **Multimodal Fusion Queries**, denoted as $\{ \mathsf{q}_m \}_{m=1}^M$. These queries attend to the $N$ visual patch embeddings $\{ \mathbf{e}_n \}_{n=1}^N$ extracted from the reference image.

AdaFocal injects the learned scalar $\beta$ as a dynamic bias into the cross-attention computation. To ensure this bias is applied only to the relevant region, a binary mask $M_{B_r}$ is created. This mask is spatially aligned with the patch embeddings corresponding to the bounding box $B_r$ (i.e., it has high values inside the box and low values outside).

The modulated cross-attention formula, which produces the updated queries $\{ \hat{\mathbf{q}}_m \}$, is:

$$
\{ \hat{\mathbf{q}}_m \} = A' V = \mathrm{Softmax}\left( \frac{Q K^T + \beta \cdot M_{B_r}}{\sqrt{d_k}} \right) V,
$$

**Symbol Explanation:**
*   $\{ \hat{\mathbf{q}}_m \}$: The output updated queries after attention modulation.
*   $A'$: The modulated attention weights matrix.
*   $V$: The value matrix, derived from visual patch embeddings $\{ \mathbf{e}_n \}$ via a projection $f_v$.
*   $Q$: The query matrix, derived from $\{ \mathsf{q}_m \}$ via a projection $f_q$.
*   $K$: The key matrix, derived from $\{ \mathbf{e}_n \}$ via a projection $f_k$.
*   $\beta$: The context-aware modulation scalar predicted by the CAAM.
*   $M_{B_r}$: The binary mask corresponding to the bounding box region.
*   $d_k$: The dimension of the key vectors, used for scaling.

    This mechanism effectively "highlights" the attention scores for the patches inside the bounding box by adding $\beta$ to them before the softmax operation. This forces the model to pay more attention to the anchored instance during the fusion of visual and textual information.

### Step 3: Feature Projection and Training
After the modulated attention, the output [CLS] token from the query branch is projected into a shared embedding space using a linear layer $\mathrm{Linear}_{\mathcal{M}}(\cdot)$ to form the final query representation $f_q$.

$$
f_q = \mathrm{Linear}_{\mathcal{M}}(\mathcal{E}_{\mathcal{M}}'(\mathcal{E}_{\mathcal{T}}(I_r), B_r, T_m, \{q_m\})),
$$

where $\mathcal{E}_{\mathcal{M}}'$ denotes the multimodal encoder operating with the Attention Activation Mechanism.

Similarly, the target image $I_t$ is processed through the Target Branch (which uses the standard frozen encoder without the CAAM modulation) to get the target representation $f_t$.

$$
f_t = \mathrm{Linear}_{\mathcal{T}}(\mathcal{E}_{\mathcal{M}}(\mathcal{E}_{\mathcal{T}}(I_t), \{q_m\})).
$$

The entire framework is trained end-to-end using a batch-based contrastive learning objective. The Contrastive Alignment Loss is formulated as:

$$
\mathcal{L}_{\mathrm{Align}} = - \frac{1}{|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|} \log \frac{\mathbb{S}(f_q^{(i)}, f_t^{(i)})}{\sum_{j=1}^{|\mathcal{B}|} \mathbb{S}(f_q^{(i)}, f_t^{(j)})},
$$

where $\mathbb{S}(a, b) := \exp(\mathrm{Sim}(a, b) / \tau)$, $\mathrm{Sim}(\cdot)$ is the cosine similarity, $\tau$ is the temperature hyper-parameter, and $\mathcal{B}$ denotes the training batch.

The following figure (Figure 4 from the original paper) shows the system architecture:

![Figure 4. Overall architecture of our proposed AdaFocal framework.](images/4.jpg)
*该图像是图示，展示了我们提出的AdaFocal框架的整体架构。框架包含上下文感知注意调制模块（CAAM）和多模态编码器（$E_M$），通过上下文推理模块和注意力激活机制实现图像和文本的融合，旨在提升实例级一致性和检索精度。*

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted on the newly proposed **OACIRR** benchmark.
*   **Source:** Constructed from four real-world datasets: DeepFashion2, Stanford Cars, Products-10K, and Google Landmarks v2.
*   **Scale:** It contains over 160K quadruples $(I_r, B_r, T_m, I_t)$.
*   **Structure:** It includes a training set (127K quadruples) and an evaluation benchmark (33.4K queries) across four domains: Fashion, Car, Product, and Landmark.
*   **Characteristics:** The benchmark is designed to be challenging. It includes "hard-negative" distractors in the candidate galleries—images that share the same category as the target but are different instances. This forces the model to discriminate fine-grained details.
*   **Example Data:** A data sample consists of a reference image (e.g., a specific red shoe), a bounding box around the shoe, a modification text (e.g., "shown on a wooden floor"), and a target image (the same specific red shoe shown on a wooden floor).

## 5.2. Evaluation Metrics
The paper employs two primary metrics to evaluate performance:

1.  **Recall@K (R@K):**
    *   **Conceptual Definition:** This is a standard metric in information retrieval. It measures the percentage of queries where the correct target image appears within the top-K retrieved results.
    *   **Mathematical Formula:**
        $$
        \text{Recall}@K = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{rank}(i) \leq K)
        $$
    *   **Symbol Explanation:**
        *   $N$: Total number of queries.
        *   $\text{rank}(i)$: The rank position of the ground-truth target image for the $i$-th query.
        *   $\mathbb{I}(\cdot)$: An indicator function that is 1 if the condition is true and 0 otherwise.

2.  **Instance Recall@K (R_ID@K):**
    *   **Conceptual Definition:** This is a stricter metric proposed specifically for the OACIR task. A retrieval is only counted as correct if the retrieved image contains the **exact same instance** specified in the reference image's bounding box. This ignores semantic correctness; if the image matches the text description but shows a different instance (e.g., a different car of the same model), it is considered a failure.
    *   **Mathematical Formula:**
        $$
        \text{Recall}_{ID}@K = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{rank}_{ID}(i) \leq K)
        $$
    *   **Symbol Explanation:**
        *   $N$: Total number of queries.
        *   $\text{rank}_{ID}(i)$: The rank position of the first retrieved image that shares the exact instance ID with the query's reference image.
        *   $\mathbb{I}(\cdot)$: Indicator function.

## 5.3. Baselines
The paper compares AdaFocal against three groups of state-of-the-art baselines:
1.  **Universal Multimodal Retrieval (UMR):** Models like UniIR, GME, and U-MARVEL. These are general-purpose models capable of handling various multimodal tasks.
2.  **Zero-Shot CIR (ZS-CIR):** Methods like Pic2Word and LinCIR that attempt to solve CIR without task-specific training data.
3.  **Composed Image Retrieval (CIR):** Supervised methods like SPRC (Sentence-level Prompts for Composed Image Retrieval).

    To ensure a fair comparison, the evaluation protocols were adapted. For UMR models (which support visual grounding), the bounding box was rendered on the image with an instruction. For CIR models (which do not support boxes), the instance constraint was converted into a textual prompt (e.g., "Same [Object]").

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results demonstrate that the OACIR task is highly challenging for existing semantic-level models. Even powerful UMR models showed limited instance-level fidelity (low R_ID@1) because they are trained for broad semantic correspondence. Zero-shot CIR methods performed even worse due to the lack of fine-grained visual input.

A critical finding is the importance of **Instance-Aware Training**. When a strong baseline (SPRC) was fine-tuned on the standard CIRR dataset (semantic), it performed poorly on OACIRR. However, when fine-tuned on the OACIRR dataset (instance-consistent), its performance improved dramatically (from ~37% to ~74% average recall). This validates that the dataset construction is crucial.

Finally, **AdaFocal** outperformed all baselines. With a ViT-G backbone, it achieved significant improvements in both Recall@1 and Instance Recall@1 compared to the OACIRR-trained SPRC baseline. This confirms that the adaptive visual grounding mechanism is more effective than simply using textual prompts or static attention.

## 6.2. Data Presentation (Tables)
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Domain</th>
<th rowspan="2">Method</th>
<th rowspan="2">Pretraining Data</th>
<th colspan="3">Fashion</th>
<th colspan="3">Car</th>
<th colspan="3">Product</th>
<th colspan="3">Landmark</th>
<th rowspan="2">Avg.</th>
</tr>
<tr>
<th>RID@1</th>
<th>R@1</th>
<th>R@5</th>
<th>RID@1</th>
<th>R@1</th>
<th>R@5</th>
<th>RID@1</th>
<th>R@1</th>
<th>R@5</th>
<th>RID@1</th>
<th>R@1</th>
<th>R@5</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">UMR</td>
<td>UniIR-CLIPF [43]</td>
<td>M-BEIR [43]</td>
<td>17.33</td>
<td>12.26</td>
<td>24.76</td>
<td>32.67</td>
<td>16.95</td>
<td>41.89</td>
<td>33.71</td>
<td>18.22</td>
<td>40.10</td>
<td>29.47</td>
<td>15.51</td>
<td>43.24</td>
<td>27.18</td>
</tr>
<tr>
<td>UniIR-BLIPFF [43]</td>
<td></td>
<td>28.53</td>
<td>22.41</td>
<td>39.63</td>
<td>37.21</td>
<td>19.97</td>
<td>46.51</td>
<td>37.76</td>
<td>20.98</td>
<td>43.19</td>
<td>31.71</td>
<td>17.14</td>
<td>52.12</td>
<td>33.10</td>
</tr>
<tr>
<td>LamRA-Ret [30]</td>
<td>M-BEIR + NLI [36]</td>
<td>27.45</td>
<td>21.63</td>
<td>37.10</td>
<td>61.03</td>
<td>35.44</td>
<td>74.51</td>
<td>69.45</td>
<td>39.53</td>
<td>70.25</td>
<td>58.64</td>
<td>32.58</td>
<td>68.74</td>
<td>49.70</td>
</tr>
<tr>
<td>MM-Embed [28]</td>
<td>M-BEIR + MTEB [35]</td>
<td>41.38</td>
<td>34.55</td>
<td>52.50</td>
<td>53.21</td>
<td>30.06</td>
<td>62.80</td>
<td>71.03</td>
<td>41.47</td>
<td>71.15</td>
<td>78.85</td>
<td>38.88</td>
<td>79.32</td>
<td>54.60</td>
</tr>
<tr>
<td>GME (2B) [52]</td>
<td>UMRB [52]</td>
<td>38.13</td>
<td>32.14</td>
<td>51.50</td>
<td>58.84</td>
<td>31.60</td>
<td>66.03</td>
<td>76.89</td>
<td>44.11</td>
<td>74.20</td>
<td>73.86</td>
<td>38.99</td>
<td>75.61</td>
<td>55.16</td>
</tr>
<tr>
<td>GME (7B) [52]</td>
<td></td>
<td>44.98</td>
<td>39.24</td>
<td>60.18</td>
<td>63.11</td>
<td>38.34</td>
<td>75.38</td>
<td>83.44</td>
<td>54.60</td>
<td>84.15</td>
<td>77.11</td>
<td>47.09</td>
<td>82.69</td>
<td>62.53</td>
</tr>
<tr>
<td>U-MARVEL [24]</td>
<td>M-BEIR + NLI</td>
<td>46.05</td>
<td>40.38</td>
<td>60.59</td>
<td>62.92</td>
<td>39.96</td>
<td>74.90</td>
<td>83.26</td>
<td>54.69</td>
<td>84.13</td>
<td>69.81</td>
<td>37.67</td>
<td>73.08</td>
<td>60.62</td>
</tr>
<tr>
<td rowspan="2">ZS-CIR</td>
<td>Pic2Word [38]</td>
<td>CC3M [9]</td>
<td>14.98</td>
<td>11.15</td>
<td>21.82</td>
<td>12.07</td>
<td>4.07</td>
<td>11.32</td>
<td>45.95</td>
<td>13.66</td>
<td>34.19</td>
<td>55.98</td>
<td>20.99</td>
<td>52.12</td>
<td>24.84</td>
</tr>
<tr>
<td>LinCIR [16]</td>
<td></td>
<td>15.78</td>
<td>12.04</td>
<td>21.55</td>
<td>5.55</td>
<td>2.23</td>
<td>7.28</td>
<td>47.55</td>
<td>14.63</td>
<td>34.91</td>
<td>42.76</td>
<td>19.57</td>
<td>47.15</td>
<td>22.61</td>
</tr>
<tr>
<td rowspan="3">CIR</td>
<td>SPRC (ViT-L) [4]</td>
<td>CIRR [31]</td>
<td>28.54</td>
<td>25.49</td>
<td>36.78</td>
<td>52.55</td>
<td>44.26</td>
<td>54.80</td>
<td>75.85</td>
<td>61.09</td>
<td>80.29</td>
<td>22.47</td>
<td>68.99</td>
<td>15.23</td>
<td>46.48</td>
</tr>
<tr>
<td>SPRC (ViT-L) [4]</td>
<td>OACIRR (Ours)</td>
<td>61.09</td>
<td>44.26</td>
<td>54.80</td>
<td>75.85</td>
<td>61.09</td>
<td>86.95</td>
<td>80.29</td>
<td>33.35</td>
<td>67.14</td>
<td>90.41</td>
<td>37.31</td>
<td>72.62</td>
<td>24.20</td>
<td>54.27</td>
</tr>
<tr>
<td>SPRC (ViT-G) [4]</td>
<td>OACIRR (Ours)</td>
<td>65.25</td>
<td>58.51</td>
<td>80.89</td>
<td>72.87</td>
<td>49.82</td>
<td>89.57</td>
<td>86.05</td>
<td>34.85</td>
<td>70.61</td>
<td>93.68</td>
<td>40.41</td>
<td>76.32</td>
<td>26.29</td>
<td>56.04</td>
</tr>
<tr>
<td rowspan="2">OACIR</td>
<td>AdaFocal (ViT-L)</td>
<td>OACIRR (Ours)</td>
<td>72.60</td>
<td>61.95</td>
<td>85.30</td>
<td>75.68</td>
<td>51.87</td>
<td>90.04</td>
<td>87.76</td>
<td>69.94</td>
<td>93.32</td>
<td>80.50</td>
<td>57.55</td>
<td>90.25</td>
<td>58.47</td>
<td>76.40</td>
</tr>
<tr>
<td>AdaFocal (ViT-G)</td>
<td>OACIRR (Ours)</td>
<td>77.15</td>
<td>65.31</td>
<td>86.88</td>
<td>78.42</td>
<td>53.63</td>
<td>92.22</td>
<td>91.86</td>
<td>74.11</td>
<td>95.39</td>
<td>82.92</td>
<td>58.47</td>
<td>91.63</td>
<td>79.00</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The authors conducted several ablation studies to validate the design of AdaFocal:

1.  **CAAM Architecture (Table 3):** They tested different aggregation methods for the Contextual Reasoning Module (CRM) and different types of probe tokens.
    *   **Findings:** Using a **Transformer** for the CRM significantly outperformed simple Average Pooling or MLP aggregations. This suggests that complex reasoning is needed to interpret the context. Additionally, using **learnable** probe tokens yielded better results than frozen tokens, indicating that task-adapted tokens help capture nuanced cues.

        The following are the results from Table 3 of the original paper:

        <table>
        <thead>
        <tr>
        <th colspan="2">CAAM</th>
        <th colspan="4">OACIRR Benchmark</th>
        </tr>
        <tr>
        <th>CRM</th>
        <th>Probe Tokens</th>
        <th>RID@1</th>
        <th>R@1</th>
        <th>R@5</th>
        <th>Avg.</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td colspan="2">Baseline (w/o CAAM)</td>
        <td>77.74</td>
        <td>58.39</td>
        <td>88.61</td>
        <td>74.91</td>
        </tr>
        <tr>
        <td rowspan="2">Average Pooling</td>
        <td>Frozen</td>
        <td>79.70</td>
        <td>59.84</td>
        <td>89.62</td>
        <td>76.39</td>
        </tr>
        <tr>
        <td>Learnable</td>
        <td>79.83</td>
        <td>59.57</td>
        <td>89.54</td>
        <td>76.31</td>
        </tr>
        <tr>
        <td rowspan="2">MLP</td>
        <td>Frozen</td>
        <td>80.51</td>
        <td>60.55</td>
        <td>90.15</td>
        <td>77.07</td>
        </tr>
        <tr>
        <td>Learnable</td>
        <td>81.10</td>
        <td>61.10</td>
        <td>90.40</td>
        <td>77.53</td>
        </tr>
        <tr>
        <td rowspan="2">Transformer</td>
        <td>Frozen</td>
        <td>81.59</td>
        <td>61.85</td>
        <td>91.13</td>
        <td>78.19</td>
        </tr>
        <tr>
        <td>Learnable</td>
        <td>82.59</td>
        <td>62.88</td>
        <td>91.53</td>
        <td>79.00</td>
        </tr>
        </tbody>
        </table>

2.  **Modulation Scalar $\beta$ (Figure 5):** They compared the adaptive $\beta$ against fixed values.
    *   **Findings:**
        *   Any positive $\beta$ (focusing on the instance) is better than $\beta=0$ (baseline).
        *   There is a trade-off: As $\beta$ increases, **Instance Recall (RID@1)** improves (better identity preservation), but **Standard Recall (R@1)** eventually drops (model ignores context/modification text too much).
        *   The optimal fixed $\beta$ varies by domain (e.g., Fashion needs high focus, Product needs less).
        *   **AdaFocal (adaptive $\beta$)** consistently outperforms all fixed strategies, proving the necessity of context-aware modulation.

            The following figure (Figure 5 from the original paper) shows the ablation study on the Modulation Scalar $\beta$:

            ![Figure 5. Ablation study on the Modulation Scalar $\\beta$ .](images/5.jpg)
            *该图像是图表，展示了在不同 `eta` 值下，AdaFocal 方法在多个子集上的召回率（Recall）。图中的四个子图分别对应 Fashion、Car、Product 和 Landmark 子集的结果，显示了随着 `eta` 的变化，召回率的变化情况。AdaFocal 方法在各个子集上均优于基线，特别是在 Fashion 和 Product 子集上表现显著。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully introduces the **Object-Anchored Composed Image Retrieval (OACIR)** task, addressing the critical gap between semantic similarity and instance fidelity in retrieval systems. By constructing the **OACIRR** benchmark, the authors provide a rigorous testing ground with real-world data and hard negatives. The proposed **AdaFocal** framework, with its Context-Aware Attention Modulator, establishes a strong baseline by dynamically balancing focus on the anchored instance and the compositional context. The results show that this approach significantly outperforms existing semantic-level models, particularly in preserving the identity of the specific object of interest.

## 7.2. Limitations & Future Work
The authors note that while AdaFocal is robust to moderate noise in the bounding box, it still relies on the presence of the box as an input. Future work could explore extending this to scenarios where the bounding box might be implicitly derived or replaced by other forms of referential expression (e.g., pointing gestures or text-based spatial references). Additionally, while the current method focuses on single-instance anchoring, complex scenes involving multiple anchored instances could be a future direction.

## 7.3. Personal Insights & Critique
The shift from "semantic search" to "referential anchoring" represents a significant maturation in the field of image retrieval. It moves from finding "something like this" to finding "this specific thing, but different." The use of a **modulation scalar ($\beta$)** is a particularly elegant solution; instead of hard-coding attention weights, the model learns *how much* to trust the visual anchor versus the textual instruction based on the specific context. This mimics human cognitive flexibility—we focus intensely on an object when verifying its identity but relax our focus when assessing the scene context.

One potential area for further exploration is the computational overhead. While the CAAM is described as lightweight, introducing a Transformer-based reasoning module (CRM) for every query adds inference cost. It would be valuable to see if this logic can be distilled into a simpler operation for deployment on resource-constrained devices. Furthermore, the benchmark's reliance on specific domains (Fashion, Car, etc.) suggests that performance might vary in "wild" uncurated datasets; testing generalization to fully open-domain web-scale search would be a compelling next step.