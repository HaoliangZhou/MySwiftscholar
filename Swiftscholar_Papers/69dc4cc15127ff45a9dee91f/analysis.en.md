# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is "ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval." This title highlights the paper's focus on addressing a specific phenomenon called "Capability Degradation" that occurs when adapting Multimodal Large Language Models (MLLMs) for the task of Composed Image Retrieval (CIR).

## 1.2. Authors
The authors are Tianyu Yang, Chenwei He, Xiangzhao Hao, Tianyue Wang, Jiarui Guo, Haiyun Guo, Leigang Qu, Tat-Seng Chua, and Jinqiao Wang. They are affiliated with several institutions, including the Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences; Southeast University; Beijing University of Posts and Telecommunications; and the National University of Singapore. The research backgrounds of the authors likely center on computer vision, multimodal learning, and information retrieval.

## 1.3. Journal/Conference
The paper is currently available as a preprint on arXiv (arXiv:2602.01639) with a publication date of February 2, 2026. As of this analysis, it has not yet been published in a specific journal or conference, but it targets top-tier venues in computer vision and multimedia given the rigorous methodology and state-of-the-art comparisons.

## 1.4. Publication Year
2026.

## 1.5. Abstract
The paper aims to solve the problem of Composed Image Retrieval (CIR), which involves retrieving a target image using a hybrid query consisting of a reference image and a modification text. The authors identify that adapting generative Multimodal Large Language Models (MLLMs) for discriminative retrieval tasks leads to "Capability Degradation," where the model's fine-grained reasoning abilities deteriorate. To address this, they propose ReCALL, a model-agnostic framework following a diagnose-generate-refine pipeline. This pipeline diagnoses the retriever's blind spots, generates corrective instructions using the foundation MLLM, and refines the retriever via grouped contrastive learning. Experiments on CIRR and FashionIQ datasets show that ReCALL achieves state-of-the-art performance by effectively recalibrating these degraded capabilities.

## 1.6. Original Source Link
The paper is available as a preprint on arXiv: https://arxiv.org/abs/2602.01639. The PDF link is https://arxiv.org/pdf/2602.01639.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper addresses is **Composed Image Retrieval (CIR)**CIR. In this task, a user provides a reference image (e.g., a red dress) and a text modification (e.g., "change the color to blue"), and the system must retrieve the target image (a blue dress) from a gallery.

Early approaches used dual-tower Vision-Language Models (VLMs) like CLIP. However, these models often struggle with fine-grained, cross-modal reasoning because they rely on shallow alignments between image and text features. Recently, researchers have started adapting **Multimodal Large Language Models (MLLMs)**, such as GPT-4V or LLaVA, for this task. MLLMs are excellent at deep reasoning and following instructions.

However, the authors discovered a critical issue: when you fine-tune a generative MLLM (which is designed to generate text step-by-step) to act as a discriminative retriever (which outputs a single vector for similarity comparison), a "paradigm conflict" occurs. This conflict causes **Capability Degradation**, meaning the model loses its native ability to perform fine-grained reasoning (like distinguishing subtle details) during the retrieval process. The paper's entry point is identifying this degradation and proposing a way to fix it without losing the benefits of MLLMs.

## 2.2. Main Contributions / Findings
The paper makes three primary contributions:

1.  **Identification of Capability Degradation:** The authors provide qualitative and quantitative evidence that simply fine-tuning MLLMs for retrieval suppresses their intrinsic fine-grained reasoning capabilities.
2.  **Proposal of ReCALL Framework:** They introduce a model-agnostic framework called ReCALL. It uses a "diagnose-generate-refine" pipeline to self-improve the retrieval model. It finds where the model fails (diagnose), creates corrective training data using the original MLLM's reasoning (generate), and trains the retriever to distinguish these hard cases (refine).
3.  **State-of-the-Art Performance:** Extensive experiments on standard benchmarks (CIRR and FashionIQ) demonstrate that ReCALL not only fixes the degradation but also achieves state-of-the-art performance, outperforming existing methods.

    The following figure illustrates the concept of Capability Degradation and how ReCALL addresses it.

    ![Figure 1. Empirical illustration of Capability Degradation and the effectiveness of ReCALL $\\left( \\mathcal { R } _ { \\mathrm { r e f i n e } } \\right)$ . (a) We compare the Foundation MLLM $( \\mathcal { F } )$ under its native VQA-based generative paradigm with its fine-tuned retrieval counterpart $( \\mathcal { R } _ { \\mathrm { b a s e } } )$ under a similarity-based discriminative paradigm using a challenging query that requires fine-grained reasoning. The base retriever $\\mathcal { R } _ { \\mathrm { b a s e } }$ fails due to fine-grained grounding errors, while $\\mathcal { F }$ succeeds through step-wise reasoning. b) Quantitative evidence of Capability Degradation and Recalibration. We test $\\mathcal { R } _ { \\mathrm { b a s e } }$ on a subset of 1k instances where $\\mathcal { F }$ successfully retrieves the target (i.e., $\\mathcal { F }$ achieves $1 0 0 \\% \\mathrm { R } @ 1 .$ ). The low $\\mathbb { R } \\ @ 1$ performance of $\\mathcal { R } _ { \\mathrm { b a s e } }$ (only $6 2 . 3 3 \\%$ on CIRR and $5 5 . 8 0 \\%$ on FashionIQ) on this ${ \\mathcal { F } }$ -solvable subset provides quantifiable proof of capability degradation. Our proposed ReCALL framework effectively recovers the lost abilities, elevating $\\mathcal { R } _ { \\mathrm { b a s e } }$ to ${ \\mathcal { R } } _ { \\mathrm { r e f i n e } }$ with significant gains.](images/1.jpg)
    *该图像是图表，展示了能力退化的定性说明和 ReCALL 框架带来的定量校准结果。左侧 (a) 通过实例展示了基础模型 $R_{base}$ 在复杂查询下的细粒度错误，而右侧 (b) 则展示了在 CIRR 和 FashionIQ 数据集上的 `R@1` 性能，表明 ReCALL 的有效性。*

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, several foundational concepts are necessary:

*   **Composed Image Retrieval (CIR):** A retrieval task where the query is composed of two parts: a reference image $I_r$ and a relative caption $T_m$ describing the modification. The goal is to find the target image $I_t$ in a database that matches the composed query.
*   **Vision-Language Models (VLMs):** Models like CLIP that learn joint representations of images and text. They typically use two separate encoders (towers) to map images and text into a shared embedding space where similarity is measured via cosine similarity.
*   **Multimodal Large Language Models (MLLMs):** Advanced models (e.g., GPT-4V, LLaVA) that extend Large Language Models (LLMs) to process visual inputs. They excel at generative tasks, conversational understanding, and complex reasoning (Chain-of-Thought).
*   **Contrastive Learning:** A technique used to learn representations by bringing positive pairs (e.g., a query and its correct target) closer together and pushing negative pairs apart in the embedding space. A common loss function is **InfoNCE**.
*   **Chain-of-Thought (CoT):** A prompting technique where the model is encouraged to generate intermediate reasoning steps before arriving at a final answer. This helps break down complex tasks.
*   **Low-Rank Adaptation (LoRA):** A parameter-efficient fine-tuning method for large models. Instead of updating all weights, it adds low-rank matrices to the existing weights, significantly reducing the computational cost and memory usage.

## 3.2. Previous Works
The paper discusses two main categories of prior work:

1.  **Dual-tower VLMs for CIR:** Early methods like TIRG and PolyFormer adapted CLIP for CIR. They often relied on "combiners" to fuse the reference image and text. While effective, they are limited by the shallow interaction depth of VLMs.
2.  **MLLM-based Retrieval:** Recent works like CIR-LVLM have started using MLLMs as feature encoders for retrieval. These methods leverage the strong representation power of MLLMs but typically rely on standard fine-tuning, which the authors argue leads to the identified "Capability Degradation."

    The paper also touches upon **Self-Improvement for MLLMs**, referencing works like STaR and Reflexion. These methods use the model's own outputs to improve itself. ReCALL adapts this philosophy to the retrieval domain, which is novel compared to the static fine-tuning of previous CIR methods.

## 3.3. Technological Evolution
The field has evolved from simple metric learning on hand-crafted features to deep learning with dual-tower architectures (CLIP era). The current frontier involves leveraging the reasoning capabilities of MLLMs. However, the paper identifies a gap in this evolution: the transition from generative reasoning to discriminative embedding is not seamless and causes a loss of capability. ReCALL represents the next step in this evolution, introducing a mechanism to preserve and recalibrate the reasoning power during the adaptation phase.

## 3.4. Differentiation Analysis
Compared to previous MLLM-based retrieval methods (e.g., CIR-LVLM), ReCALL is distinct because it does not assume that standard fine-tuning is sufficient. Instead of treating the MLLM as a "black box" encoder to be fine-tuned once, ReCALL actively interacts with the foundation model's generative capabilities. It uses the foundation model to "teach" the retriever where it went wrong by synthesizing hard training examples (corrective triplets). This explicit self-diagnosis and repair loop is the core innovation.

# 4. Methodology

## 4.1. Principles
The core principle of ReCALL is to mitigate the **Capability Degradation** caused by the paradigm conflict between generative reasoning and discriminative retrieval. The framework operates on a **diagnose-generate-refine** pipeline:

1.  **Diagnose:** Identify the "cognitive blind spots" of the retrieval model—specifically, the instances where it fails to retrieve the correct target despite the foundation model having the knowledge to solve it.
2.  **Generate:** Use the foundation model's generative reasoning (Chain-of-Thought) to create "corrective supervision." This involves synthesizing new text instructions that describe the subtle differences between the target and the hard negative images found in the diagnose step.
3.  **Refine:** Train the retrieval model on these new, hard triplets using a specialized contrastive learning scheme to force it to learn the fine-grained distinctions it previously missed.

    The following figure provides an overview of the ReCALL framework architecture.

    ![Figure 2. Overview of the ReCALL framework. (1) Stage 1: A baseline retriever $\\mathcal { R } _ { b a s e }$ is adapted from the foundation model $\\mathcal { F }$ via standard fine-tuning. (2) Stage 2 (Diagnose): $\\mathcal { R } _ { b a s e }$ surfaces its own failure cases via self-guided informative instance mining. (3) Stage 3 (Generate): Leveraging native reasoning (CoT), $\\mathcal { F }$ synthesizes minimally edited corrective instructions for the mined informative produce the final $\\mathcal { R } _ { r e f i n e }$ , effectively recalibrating the degraded capabilities.](images/2.jpg)
    *该图像是图表，展示了 ReCALL 框架的概述，包括四个阶段的流程：阶段 1 为基线检索模型的适应，阶段 2 为自导向信息实例挖掘，阶段 3 为生成校准，以及阶段 4 的针对性精炼，涉及多个查询和目标实例的调整与反向矢量相似性度量 $L_{InfoNCE}$ 的使用。*

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Stage 1: Baseline Retrieval Model Adaptation
The process begins with a Foundation Model $\mathcal{F}$, which is a pre-trained MLLM with strong generative capabilities. The goal is to adapt this into a Baseline Retrieval Model $\mathcal{R}_{base}$.

The model is initialized from $\mathcal{F}$ and fine-tuned on standard CIR triplets $(I_r, T_m, I_t)$. The training objective is the standard **InfoNCE loss**, which encourages the similarity between the query embedding and the positive target embedding to be higher than the similarity with any other target in the batch.

However, as discussed, this step inevitably introduces Capability Degradation. The model learns coarse-grained alignment but loses sensitivity to fine details.

### 4.2.2. Stage 2: Self-Guided Informative Instance Mining (Diagnose)
To find where the model is failing, ReCALL performs inference on the training set using $\mathcal{R}_{base}$. It specifically looks for **failure cases**—queries where the ground-truth target $I_t$ is not ranked first.

For each failure case, the model identifies the top-$K$ images that were incorrectly ranked *higher* than the ground truth. These images, denoted as $\{I_h\}$, are "informative instances." They are visually similar to the target but differ in subtle semantic ways that confuse the degraded retriever. These instances serve as the anchors for the next stage.

### 4.2.3. Stage 3: Generative Calibration (Generate)
In this stage, the framework leverages the full reasoning power of the frozen Foundation Model $\mathcal{F}$ to understand *why* the retriever confused $I_h$ with $I_t$.

**CoT-Assisted Generation:**
The system prompts $\mathcal{F}$ with a Chain-of-Thought (CoT) process. It asks the model to compare the reference image $I_r$, the modification text $T_m$, and the hard negative image $I_h$.
1.  **Intent Decomposition & Verification:** $\mathcal{F}$ breaks down the modification text into atomic intents (e.g., "change color to red", "add stripes") and checks which ones are violated by $I_h$.
2.  **Minimal Edit Synthesis:** $\mathcal{F}$ generates a minimally edited version of the text, $\tilde{T}_m$, which accurately describes $I_h$.

    This creates a new triplet $(I_r, \tilde{T}_m, I_h)$. This triplet is highly informative because the text $\tilde{T}_m$ explicitly describes the visual difference that the retriever failed to capture.

**VQA-Assisted Quality Control:**
To ensure the generated text is accurate and not hallucinated, a VQA (Visual Question Answering) check is performed. The foundation model is asked binary questions (e.g., "Does the image have stripes?") based on $\tilde{T}_m$ and $I_h$. Only triplets that pass this consistency check are kept.

### 4.2.4. Stage 4: Targeted Refinement (Refine)
The final stage trains the model to internalize these fine-grained distinctions. The model to be refined is $\mathcal{R}_{refine}$, initialized from $\mathcal{R}_{base}$.

**Grouped Contrastive Refinement:**
Instead of random batching, the training data is organized into micro-groups. Each group contains the original positive triplet $(I_r, T_m, I_t)$ and the newly synthesized corrective triplet $(I_r, \tilde{T}_m, I_h)$. This forces the model to see the "correct" answer and the "confusing but incorrect" answer together, learning to discriminate based on the subtle text difference.

**Dual Optimization Objective:**
The training is guided by a combined loss function.

First, the standard **InfoNCE Loss** is applied to maintain global retrieval structure. The formula used in the paper is:

$$
\mathcal { L } _ { i n f o N C E } = - \log \frac { \exp ( s ( z _ { q } , z _ { t ^ { + } } ) / \tau ) } { \sum _ { z _ { t } \in \mathcal { B } } \exp ( s ( z _ { q } , z _ { t } ) / \tau ) } ,
$$

Where:
*   $\mathcal{B}$ is the batch of target representations.
*   $\tau$ is the temperature parameter (a scalar hyperparameter controlling the sharpness of the softmax distribution).
*   $s(u, v) = \frac{u^\top v}{\|u\| \|v\|}$ is the cosine similarity function between two vectors.
*   $z_q$ is the query representation derived from the input $(I_r, T_m)$.
*   $z_{t^+}$ is the representation of the positive ground-truth image $I_t$.
*   $z_t$ is a generic target representation from the batch $\mathcal{B}$.

    Second, to specifically target the confusion between the target and the hard negative, an **In-Group Triplet Margin Loss** is added:

$$
\mathcal { L } _ { t r i p l e t } = \operatorname* { m a x } ( 0 , s ( z _ { q } , z _ { t ^ { - } } ) - s ( z _ { q } , z _ { t ^ { + } } ) + m ) ,
$$

Where:
*   $m$ is a margin hyperparameter (a scalar value enforcing a minimum distance between positive and negative pairs).
*   $z_{t^-}$ corresponds to the hard negative representation $I_h$ identified in the diagnose stage.
*   This loss maximizes the distance between the query and the hard negative while minimizing the distance to the positive target, ensuring the separation is at least $m$.

    The final total loss combines these two objectives:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { i n f o N C E } + \lambda \mathcal { L } _ { t r i p l e t } ,
$$

Where $\lambda$ is a weighting hyperparameter that balances the global alignment (InfoNCE) and the targeted fine-grained refinement (Triplet).

# 5. Experimental Setup

## 5.1. Datasets
The authors evaluate ReCALL on two standard benchmarks for Composed Image Retrieval:

1.  **FashionIQ:** This dataset focuses on the fashion domain. It consists of triplets of reference and target images (e.g., clothing items) with natural language instructions describing the change (e.g., "change the color to black"). It is divided into categories like Dress, Shirt, and Top&Tee. It is known for requiring fine-grained attribute understanding (texture, pattern, sleeve length).
2.  **CIRR:** This dataset is derived from the NLVR2 dataset and is more general-domain. It involves complex object interactions and relational changes (e.g., "the animal that is looking at the camera"). It tests the model's ability to handle open-domain reasoning and spatial relationships.

    These datasets were chosen because they represent the two main challenges in CIR: fine-grained attribute manipulation (FashionIQ) and complex relational reasoning (CIRR).

## 5.2. Evaluation Metrics
The primary metric used is **Recall@K (R@K)**.

1.  **Conceptual Definition:** Recall@K measures the percentage of queries for which the correct ground-truth image appears within the top-$K$ retrieved results. It assesses the ranking quality of the retrieval system. For example, R@1 checks if the top result is correct, while R@10 checks if the correct answer is in the top 10 list.
2.  **Mathematical Formula:**
    $$
    Recall@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q, I_{gt}) \leq K)
    $$
3.  **Symbol Explanation:**
    *   $|Q|$: The total number of queries in the test set.
    *   $q$: A specific query in the set $Q$.
    *   $\mathbb{I}(\cdot)$: The indicator function, which returns 1 if the condition inside is true and 0 otherwise.
    *   $\text{rank}(q, I_{gt})$: The rank position of the ground-truth image $I_{gt}$ for query $q$ in the retrieved list.
    *   $K$: The cutoff rank (e.g., 1, 5, 10, 50).

        For CIRR, the authors also report a subset metric, Recall_subset@K, which measures performance on a curated subset of 6 candidates, focusing specifically on fine-grained discriminative power.

## 5.3. Baselines
The paper compares ReCALL against a strong set of baselines, including:
*   **Traditional Dual-tower methods:** TIRG, ARTEMIS, CoVR-2.
*   **Recent MLLM-based methods:** CIR-LVLM, QuRe, TME.
    These baselines are representative of the evolution of the field, from early combiner modules to the latest MLLM adaptations.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The results demonstrate that ReCALL effectively addresses Capability Degradation.

*   **On CIRR:** ReCALL achieves 55.52% on R@1, outperforming the concurrent MLLM method CIR-LVLM (53.64%). Crucially, compared to the paper's own baseline $\mathcal{R}_{base}$ (51.23%), ReCALL shows a significant boost of +4.29%. This proves that the diagnose-generate-refine pipeline successfully recovers the lost reasoning capabilities.
*   **On FashionIQ:** ReCALL achieves 57.04% on average R@10, outperforming CIR-LVLM (56.21%). Again, the improvement over the internal baseline (53.23%) is substantial (+3.81%), particularly in the "Dress" category (+10.71%), highlighting the framework's strength in fine-grained attribute discrimination.

    The following are the results from Table 1 of the original paper:

    <table>
    <thead>
    <tr>
    <td rowspan="2">Method</td>
    <td rowspan="2">Venue</td>
    <td colspan="4">Recall@K</td>
    <td colspan="3">Recall_subset@K</td>
    <td rowspan="2">Avg.</td>
    </tr>
    <tr>
    <td>K = 1</td>
    <td>K = 5</td>
    <td>K = 10</td>
    <td>K = 50</td>
    <td>K = 1</td>
    <td>K = 2</td>
    <td>K = 3</td>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>TIRG [51]</td>
    <td>CVPR'19</td>
    <td>14.61</td>
    <td>48.37</td>
    <td>64.08</td>
    <td>90.03</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>ARTEMIS [11]</td>
    <td>ICLR'22</td>
    <td>16.96</td>
    <td>46.10</td>
    <td>61.31</td>
    <td>87.73</td>
    <td>39.99</td>
    <td>62.20</td>
    <td>75.67</td>
    <td>43.05</td>
    </tr>
    <tr>
    <td>TG-CIR [57]</td>
    <td>MM'23</td>
    <td>45.25</td>
    <td>78.29</td>
    <td>87.16</td>
    <td>97.30</td>
    <td>72.84</td>
    <td>89.25</td>
    <td>95.13</td>
    <td>75.57</td>
    </tr>
    <tr>
    <td>SPRC [3]</td>
    <td>ICLR'24</td>
    <td>51.96</td>
    <td>82.12</td>
    <td>89.74</td>
    <td>97.69</td>
    <td>80.65</td>
    <td>92.31</td>
    <td>96.60</td>
    <td>81.39</td>
    </tr>
    <tr>
    <td>LIMN [56]</td>
    <td>TPAMI'24</td>
    <td>43.64</td>
    <td>75.37</td>
    <td>85.42</td>
    <td>97.04</td>
    <td>69.01</td>
    <td>86.22</td>
    <td>94.19</td>
    <td>72.19</td>
    </tr>
    <tr>
    <td>CoVR-2 [50]</td>
    <td>TPAMI'24</td>
    <td>50.43</td>
    <td>81.08</td>
    <td>88.89</td>
    <td>98.05</td>
    <td>76.75</td>
    <td>90.34</td>
    <td>95.78</td>
    <td>79.28</td>
    </tr>
    <tr>
    <td>CaLa [20]</td>
    <td>SIGIR'24</td>
    <td>49.11</td>
    <td>81.21</td>
    <td>89.59</td>
    <td>98.00</td>
    <td>76.27</td>
    <td>91.04</td>
    <td>96.46</td>
    <td>78.74</td>
    </tr>
    <tr>
    <td>ENCODER [26]</td>
    <td>AAAI'25</td>
    <td>46.10</td>
    <td>77.98</td>
    <td>87.16</td>
    <td>97.64</td>
    <td>76.92</td>
    <td>90.41</td>
    <td>95.95</td>
    <td>77.45</td>
    </tr>
    <tr>
    <td>CIR-LVLM [45]</td>
    <td>AAAI'25</td>
    <td>53.64</td>
    <td>83.76</td>
    <td>90.60</td>
    <td>97.93</td>
    <td>79.12</td>
    <td>92.33</td>
    <td>96.67</td>
    <td>81.44</td>
    </tr>
    <tr>
    <td>QuRe [22]</td>
    <td>ICML'25</td>
    <td>52.22</td>
    <td>82.53</td>
    <td>90.31</td>
    <td>98.17</td>
    <td>78.51</td>
    <td>91.28</td>
    <td>96.48</td>
    <td>80.52</td>
    </tr>
    <tr>
    <td>CCIN [48]</td>
    <td>CVPR'25</td>
    <td>53.41</td>
    <td>84.05</td>
    <td>91.17</td>
    <td>98.00</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>TME [25]</td>
    <td>CVPR'25</td>
    <td>53.42</td>
    <td>82.99</td>
    <td>90.24</td>
    <td>98.15</td>
    <td>81.04</td>
    <td>92.58</td>
    <td>96.94</td>
    <td>82.01</td>
    </tr>
    <tr>
    <td>Baseline (Rbase)</td>
    <td>-</td>
    <td>51.23</td>
    <td>82.15</td>
    <td>90.20</td>
    <td>98.20</td>
    <td>77.57</td>
    <td>91.83</td>
    <td>96.34</td>
    <td>79.86</td>
    </tr>
    <tr>
    <td>ReCALL (Rrefine)</td>
    <td>-</td>
    <td>55.52</td>
    <td>84.07</td>
    <td>91.83</td>
    <td>98.55</td>
    <td>81.49</td>
    <td>93.35</td>
    <td>97.64</td>
    <td>82.81</td>
    </tr>
    <tr>
    <td>Improvement (∆)</td>
    <td></td>
    <td>+8.38%</td>
    <td>+2.34%</td>
    <td>+1.81%</td>
    <td>+0.36%</td>
    <td>+5.06%</td>
    <td>+1.65%</td>
    <td>+1.35%</td>
    <td>+3.70%</td>

</</tr>
</tbody>
</table>

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<td rowspan="2">Method</td>
<td rowspan="2">Venue</td>
<td colspan="2">Dress</td>
<td colspan="2">Shirt</td>
<td colspan="2">Top&amp;Tee</td>
<td colspan="2">Avg.</td>
</tr>
<tr>
<td>R@10</td>
<td>R@50</td>
<td>R@10</td>
<td>R@50</td>
<td>R@10</td>
<td>R@50</td>
<td>R@10</td>
<td>R@50</td>
</tr>
</thead>
<tbody>
<tr>
<td>TIRG [51]</td>
<td>CVPR'19</td>
<td>14.13</td>
<td>34.61</td>
<td>13.10</td>
<td>30.91</td>
<td>14.79</td>
<td>34.37</td>
<td>14.01</td>
<td>33.30</td>
</tr>
<tr>
<td>ARTEMIS [11]</td>
<td>ICLR'22</td>
<td>25.68</td>
<td>51.05</td>
<td>21.57</td>
<td>44.13</td>
<td>28.59</td>
<td>55.06</td>
<td>25.28</td>
<td>50.08</td>
</tr>
<tr>
<td>FashionSAP [15]</td>
<td>CVPR'23</td>
<td>33.71</td>
<td>60.43</td>
<td>41.91</td>
<td>70.93</td>
<td>33.17</td>
<td>61.33</td>
<td>36.26</td>
<td>64.23</td>
</tr>
<tr>
<td>FAME-ViL [14]</td>
<td>CVPR'23</td>
<td>42.19</td>
<td>67.38</td>
<td>47.64</td>
<td>68.79</td>
<td>50.69</td>
<td>73.07</td>
<td>46.84</td>
<td>69.75</td>
</tr>
<tr>
<td>SyncMask [42]</td>
<td>CVPR'24</td>
<td>33.76</td>
<td>61.23</td>
<td>35.82</td>
<td>62.12</td>
<td>44.82</td>
<td>72.06</td>
<td>38.13</td>
<td>65.14</td>
</tr>
<tr>
<td>SADN [55]</td>
<td>MMI'24</td>
<td>40.01</td>
<td>65.10</td>
<td>43.67</td>
<td>66.05</td>
<td>48.04</td>
<td>70.93</td>
<td>43.91</td>
<td>67.36</td>
</tr>
<tr>
<td>CaLa [20]</td>
<td>SIGIR'24</td>
<td>42.38</td>
<td>66.08</td>
<td>46.76</td>
<td>68.16</td>
<td>50.93</td>
<td>73.42</td>
<td>46.69</td>
<td>69.22</td>
</tr>
<tr>
<td>CoVR-2 [50]</td>
<td>TPAMI'24</td>
<td>46.53</td>
<td>69.60</td>
<td>51.23</td>
<td>70.64</td>
<td>52.14</td>
<td>73.27</td>
<td>49.96</td>
<td>71.17</td>
</tr>
<tr>
<td>SPRC [3]</td>
<td>ICLR'24</td>
<td>49.18</td>
<td>72.43</td>
<td>55.64</td>
<td>73.89</td>
<td>59.35</td>
<td>78.58</td>
<td>54.72</td>
<td>74.97</td>
</tr>
<tr>
<td>CIR-LVLM [45]</td>
<td>AAAI'25</td>
<td>50.42</td>
<td>73.60</td>
<td>58.59</td>
<td>75.86</td>
<td>59.61</td>
<td>78.99</td>
<td>56.21</td>
<td>76.14</td>
</tr>
<tr>
<td>CCIN [48]</td>
<td>CVPR'25</td>
<td>49.38</td>
<td>72.58</td>
<td>55.93</td>
<td>74.14</td>
<td>57.93</td>
<td>77.56</td>
<td>54.41</td>
<td>74.76</td>
</tr>
<tr>
<td>TME [25]</td>
<td>CVPR'25</td>
<td>49.73</td>
<td>71.69</td>
<td>56.43</td>
<td>74.44</td>
<td>59.31</td>
<td>78.94</td>
<td>55.15</td>
<td>75.02</td>
</tr>
<tr>
<td>QuRe [22]</td>
<td>ICML'25</td>
<td>46.80</td>
<td>69.81</td>
<td>53.53</td>
<td>72.87</td>
<td>57.47</td>
<td>77.77</td>
<td>52.60</td>
<td>73.48</td>
</tr>
<tr>
<td>Baseline (Rbase)</td>
<td></td>
<td>46.80</td>
<td>70.60</td>
<td>55.00</td>
<td>74.39</td>
<td>57.88</td>
<td>78.12</td>
<td>53.23</td>
<td>74.37</td>
</tr>
<tr>
<td>ReCALL (Rrefine)</td>
<td></td>
<td>51.81</td>
<td>73.48</td>
<td>58.49</td>
<td>76.59</td>
<td>60.83</td>
<td>79.19</td>
<td>57.04</td>
<td>76.42</td>
</tr>
<tr>
<td>Improvement (∆)</td>
<td></td>
<td>+10.71%</td>
<td>+4.08%</td>
<td>+6.35%</td>
<td>+2.96%</td>
<td>+5.10%</td>
<td>+1.37%</td>
<td>+7.16%</td>
<td>+2.76%</td>
</tr>
</tbody>
</table>

## 6.2. Ablation Studies / Parameter Analysis
The authors conducted extensive ablation studies to validate each component of ReCALL.

*   **Mining Strategy:** They compared "Self-Guided Mining" (mining failure cases) with "Random Mining." Self-guided mining significantly outperformed random mining (57.04% vs 53.80% Avg R@10 on FashionIQ). This proves that targeting the model's specific blind spots is more efficient than random data augmentation.
*   **Generative Calibration:** Removing the CoT-assisted generation (CG) or the VQA-based quality control (VC) led to a drop in performance, confirming that high-quality, reasoning-based synthetic data is crucial.
*   **Refinement Strategy:** Removing the "Grouped Contrastive Refinement" (GR) and just using standard batching hurt performance. This validates that the specific grouping of positive and hard negatives is necessary to force the model to learn the fine-grained differences.
*   **Generalizability:** The framework was tested on different backbones (Qwen2.5-VL-7B and Qwen3-VL-8B). As shown in the figure below, ReCALL consistently improved performance regardless of the base model strength, confirming its model-agnostic nature.

    ![Figure 3. Generalizability across backbones. We validate ReCALL on different foundation models (Qwen2.5-VL-7B and Qwen3-VL-8B). Despite higher baselines, ReCALL consistently delivers performance gains on both (a) CIRR and (b) FashionIQ, confirming the strong generalizability of our framework.](images/3.jpg)
    *该图像是柱状图，展示了在不同基础模型（Qwen2.5-VL-7B 和 Qwen3-VL-8B）上，ReCALL 在 CIRR 和 FashionIQ 数据集上的表现。图中展示了 R@1 和 R@10 的具体数值，体现了 ReCALL 的强大通用性。*

*   **Data Scale:** The authors analyzed the effect of the mining hyperparameter $K$ (the number of informative instances per query). As shown in Figure 5, increasing the data scale (by increasing $K$) consistently improved performance, demonstrating the framework's scalability.

    ![Figure 5. Effect of data scale on the FashionIQ validation set. The visualization employs dual $\\mathbf { X }$ -axes to map the mining hyperparameter $K$ (bottom) to the corresponding number of mined samples (top). The dual y-axes (left for $\\mathrm { R @ 1 0 }$ , right for ${ \\textrm R @ 5 0 }$ with zoomed-in scales highlight the monotonic performance gains as the data scale increases.](images/5.jpg)
    *该图像是图表，展示了数据规模对FashionIQ验证集的影响。图中使用双横坐标轴映射了挖掘超参数$K$与对应的挖掘样本数量。在双纵坐标中，左侧表示$\mathrm{R@10}$的平均值（以蓝线表示），右侧表示$\mathrm{R@50}$的平均值（以橙线表示）。随着数据规模的增大，性能逐渐提升，表现出单调增长的趋势。*

*   **Hyperparameters:** They analyzed the triplet loss weight $\lambda$ and margin $m$. Figure 6 shows that the optimal settings were $\lambda=0.3$ and $m=0.05$. A smaller margin was preferred, likely because the mined hard negatives are visually very similar to the targets, requiring fine-grained discrimination rather than coarse separation.

    ![Figure 6. Hyperparameter sensitivity analysis on the FashionIQ validation set. We report Avg. $R @ 1 0 ~ ( \\% )$ under varying triplet loss weights $( \\lambda )$ and margins `( m )` . The red box highlights the optimal configuration adopted in our final model.](images/6.jpg)
    *该图像是图表，展示了FashionIQ验证集上的超参数敏感性分析。图中以二维热图形式显示了在不同的三元组损失权重 $(\lambda)$ 和边际 `(m)` 下，平均 $R@10~(\%)$ 的变化情况。红框突出显示了在最终模型中采用的最佳配置。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully identifies and addresses the "Capability Degradation" problem in MLLM-based Composed Image Retrieval. By proposing the ReCALL framework, the authors demonstrate a novel way to bridge the gap between generative reasoning and discriminative retrieval. The diagnose-generate-refine pipeline effectively uses the MLLM's own intelligence to self-diagnose and repair its deficiencies when adapted for retrieval. The method achieves state-of-the-art results on CIRR and FashionIQ benchmarks.

## 7.2. Limitations & Future Work
The authors acknowledge that some failure cases stem from the inherent ambiguity of natural language instructions and the "false negative" problem (where multiple valid images exist but only one is labeled). Future work could focus on one-to-many evaluation protocols to better handle this. Additionally, while ReCALL improves spatial reasoning, complex geometric transformations (like specific rotations) remain a challenge, suggesting a need for even stronger backbones or specialized geometric modules.

## 7.3. Personal Insights & Critique
ReCALL offers a compelling perspective on model adaptation. Instead of viewing fine-tuning as a one-shot process, it treats it as an iterative self-improvement loop. This is a significant shift in how we might approach adapting Large Foundation Models for downstream tasks.

The separation of the "Teacher" (Foundation Model $\mathcal{F}$) and the "Student" (Retriever $\mathcal{R}$) is elegant. It allows the student to specialize in efficient retrieval (vector embeddings) while still having access to the teacher's slow but powerful reasoning (CoT) for calibration.

One potential area for improvement is the computational cost of the "Generate" stage, which involves running CoT inference on failure cases. While the paper notes this is a one-time offline cost, for extremely large datasets, this could still be prohibitive. Future work might explore more efficient ways to approximate these corrective signals or distill the generation process.

Overall, the paper provides a robust, generalizable solution to a fundamental problem in multimodal AI, with strong empirical backing and clear methodological contributions.