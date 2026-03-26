# 1. Bibliographic Information

## 1.1. Title
WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval

## 1.2. Authors
The paper was authored by Tianyue Wang, Leigang Qu, Tianyu Yang, Xiangzhao Hao, Yifan Xu, Haiyun Guo, and Jinqiao Wang. The authors are affiliated with:
*   **SAIS, UCAS:** School of Artificial Intelligence, University of Chinese Academy of Sciences.
*   **Institute of Automation, CAS:** Institute of Automation, Chinese Academy of Sciences.
*   **National University of Singapore.**
*   **Minzu University of China.**

## 1.3. Journal/Conference
The paper is hosted on **arXiv** (a major preprint repository for computer science and AI). It is marked as published at 2026-02-26. Given the high-quality benchmarks and comparisons, it is likely intended for a top-tier computer vision conference like CVPR, ICCV, or ECCV.

## 1.4. Publication Year
2026

## 1.5. Abstract
Zero-Shot Composed Image Retrieval (`ZS-CIR`) involves finding a target image based on a reference image and a modification text without training on triplet data. Existing methods usually convert the query into either just text (`T2I`) or just an image (`I2I`), both of which have flaws: `T2I` loses visual detail, while `I2I` fails at complex logic. The authors propose `WISER`, a training-free framework that unifies both through a **"retrieve-verify-refine"** pipeline. It performs **Wider Search** (dual-path retrieval), **Adaptive Fusion** (confidence-based merging), and **Deeper Thinking** (iterative self-reflection for uncertain cases). `WISER` achieves massive improvements—45% on `CIRCO` and 57% on `CIRR`—outperforming many training-dependent models.

## 1.6. Original Source Link
*   **PDF Link:** [https://arxiv.org/pdf/2602.23029v1](https://arxiv.org/pdf/2602.23029v1)
*   **Code Repository:** [https://github.com/Physicsmile/WISER](https://github.com/Physicsmile/WISER)
*   **Status:** Preprint (arXiv).

    ---

# 2. Executive Summary

## 2.1. Background & Motivation
**Composed Image Retrieval (CIR)** is a task where a user wants to find an image that looks like a "Reference Image" but with "Modification Text" applied (e.g., "Show me this shirt, but in red with long sleeves"). 

Standard methods require massive datasets of annotated triplets (Image A + Text -> Image B), which are expensive to create. This led to the development of **Zero-Shot Composed Image Retrieval (ZS-CIR)**, which aims to solve the problem using pre-trained models like `CLIP` without task-specific training.

**The Problem:**
1.  **Text-to-Image (T2I) Paradigm:** Converts the image and text into a single caption. It’s good at understanding "change the dog to a cat" but bad at remembering the exact color or texture of the original dog's leash.
2.  **Image-to-Image (I2I) Paradigm:** Edits the reference image to create a "fake" target image. It’s good at keeping visual details but often fails when the text instruction is complex or ambiguous.

**Innovation:**
The authors introduce `WISER` to bridge this gap. Instead of choosing one path, `WISER` uses both. It treats retrieval as an iterative process of searching, verifying the results with a visual judge, and "thinking deeper" to fix errors if the results aren't good enough.

## 2.2. Main Contributions / Findings
1.  **Unified Training-Free Framework:** Proposes the first framework that adaptively leverages both `T2I` and `I2I` without needing any fine-tuning.
2.  **"Retrieve-Verify-Refine" Pipeline:**
    *   **Wider Search:** Broadens the candidate pool by searching through both text and image descriptions.
    *   **Adaptive Fusion:** A verifier (Multi-modal Large Language Model) scores candidates, allowing the system to merge results based on intent and uncertainty.
    *   **Deeper Thinking:** If the system is "unsure" (low confidence), it uses an LLM to self-reflect on why the search failed and tries again with better queries.
3.  **State-of-the-Art Performance:** It sets new records on `Fashion-IQ`, `CIRR`, and `CIRCO` benchmarks, often beating models that were specifically trained for these tasks.

    ---

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Zero-Shot Learning (ZSL)
`Zero-Shot Learning` is a machine learning setup where a model must recognize or retrieve items from categories it never saw during training. It relies on high-level semantic descriptions to connect seen and unseen concepts.

### 3.1.2. CLIP (Contrastive Language-Image Pre-training)
`CLIP` is the backbone of most modern retrieval tasks. It consists of an Image Encoder ($E_{img}$) and a Text Encoder ($E_{txt}$). It is trained on millions of image-text pairs so that the vector (embedding) of an image and its caption are close together in a mathematical "embedding space."
The similarity between an image $I$ and text $T$ is typically calculated using **Cosine Similarity**:
\$
\text{sim}(I, T) = \frac{E_{img}(I) \cdot E_{txt}(T)}{\|E_{img}(I)\| \|E_{txt}(T)\|}
\$
This formula measures the cosine of the angle between two vectors. A value of 1 means they are identical in direction (high similarity), and 0 means they are unrelated.

### 3.1.3. Multi-modal Large Language Models (MLLMs)
Models like `GPT-4o` or `Qwen2-VL` that can "see" images and "read" text. In this paper, these models are used as "Verifiers" (to check if a retrieved image is correct) and "Refiners" (to suggest better search terms).

## 3.2. Previous Works
*   **Text-to-Image (T2I) methods:** `Pic2Word` and `SEARLE` use "textual inversion" to turn an image into a special word token. Recent models like `CIReVL` use `LLMs` to generate target captions.
*   **Image-to-Image (I2I) methods:** `CompoDiff` uses diffusion models to edit the image itself.
*   **Limitation of Prior Work:** Most methods use "Static Fusion," meaning they always combine the text and image results with a fixed weight (e.g., 50/50). This is bad because some queries are 90% about text and 10% about the image, or vice versa.

## 3.3. Differentiation Analysis
`WISER` is different because it is **Intent-Aware** and **Uncertainty-Aware**. It doesn't just guess; it checks its own work using a verifier and iterates. It is modular, meaning you can swap out the models it uses (like changing the `LLM` from `GPT-4o` to `Qwen`) without retraining.

---

# 4. Methodology

## 4.1. Principles
The core intuition behind `WISER` is to simulate human behavior: search for something, look at the results, realize it's not quite right, and then refine the search terms to get a better result.

The following figure (Figure 2 from the original paper) shows the system architecture:

![该图像是示意图，展示了WISER框架在零样本组合图像检索中的工作流程。图中包括三个主要步骤：首先通过文本和图像编码器生成候选图像，然后通过验证器分析结果，最后提供改进建议以优化后续检索流程。每个步骤都对应具体操作，强调了更深层次思考与自适应融合的重要性。](images/2.jpg)
*该图像是示意图，展示了WISER框架在零样本组合图像检索中的工作流程。图中包括三个主要步骤：首先通过文本和图像编码器生成候选图像，然后通过验证器分析结果，最后提供改进建议以优化后续检索流程。每个步骤都对应具体操作，强调了更深层次思考与自适应融合的重要性。*

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Step 1: Wider Search (Dual-path Retrieval)
The goal is to generate two different representations of the "Target Image."

**1. Text-to-Image (T2I) Path:**
First, a captioner describes the reference image ($I_{ref}$) as $C_{ref}$. Then, a textual editor $\mathcal{F}_{txt}$ (like an `LLM`) combines this with the user's instructions $T_{mod}$ to create an edited caption:
\$
C_{edit} = \mathcal{F}_{txt}(C_{ref}, T_{mod})
\$
This $C_{edit}$ explicitly describes what the target should look like.

**2. Image-to-Image (I2I) Path:**
An image editor $\mathcal{F}_{img}$ (like a diffusion-based editor) modifies the pixels of $I_{ref}$ based on $T_{mod}$ to produce an edited image:
\$
I_{edit} = \mathcal{F}_{img}(I_{ref}, T_{mod})
\$

**3. Candidate Pooling:**
The system retrieves the top-$K$ candidates for each path using `CLIP`. Let $\mathcal{R}_{p}$ be the set of images retrieved by path $p \in \{T2I, I2I\}$. The "expanded candidate pool" is the union:
\$
\mathcal{R}_{union} = \mathcal{R}_{T2I} \cup \mathcal{R}_{I2I}
\$
This ensures that if the target was found by *either* method, it is included.

### 4.2.2. Step 2: Adaptive Fusion (Verification and Scoring)
Not all candidates are equally good. `WISER` uses a **Verifier** $\Phi$ (an `MLLM`) to judge them.

**1. Verification-based Scoring:**
For a candidate $I_{p}^{k}$, the verifier is asked: "Does this image match the instruction applied to the reference?" It provides two "logits" (raw scores) for "yes" and "no":
\$
\{\ell_{p,k}^{(yes)}, \ell_{p,k}^{(no)}\} = \Phi(I_{ref}, T_{mod}, I_{p}^{k})
\$
The **Confidence Score** $c_{p}^{k}$ is then calculated using a **Softmax** function:
\$
c_{p}^{k} = \frac{\exp(\ell_{p,k}^{(yes)})}{\exp(\ell_{p,k}^{(yes)}) + \exp(\ell_{p,k}^{(no)})}
\$
$c_{p}^{k}$ is a value between 0 and 1. A score of 0.9 means the model is very confident the image is correct.

**2. Multi-Level Fusion:**
*   **Branch-Level Reliability:** The system finds the "best" candidate in each path and its score:
    \$
    r_{p} = \max_{k} c_{p}^{k}
    \$
    If $r_{p}$ is below a threshold $\tau$ (default 0.7), the system marks that branch as "uncertain" and triggers **Deeper Thinking**.
*   **Candidate-Level Fusion:** If reliable, it calculates a fused score:
    \$
    c_{fused}^{k} = c_{T2I}^{k} + c_{I2I}^{k}
    \$
    The final list is sorted by a priority function $\Psi(I^{k})$:
    \$
    \Psi(I^{k}) = \left( -c_{fused}^{k}, -\max(c_{T2I}^{k}, c_{I2I}^{k}), -c_{T2I}^{k} \right)
    \$
    *   **Primary Key:** Total fused score (both paths agree).
    *   **Secondary Key:** Max single-path confidence (one path is very sure).
    *   **Tertiary Key:** `T2I` confidence (used as a final tie-breaker).

### 4.2.3. Step 3: Deeper Thinking (Self-Reflection)
If the verifier says "I'm not sure about these results," the system enters a refinement loop.
1.  **Identify Modifications:** An `LLM` analyzes what changes were *supposed* to happen (e.g., "Change dog breed to Husky").
2.  **Analyze Failures:** The `LLM` looks at the top retrieved image and explains why it’s wrong (e.g., "This is a Golden Retriever, not a Husky").
3.  **Provide Suggestions:** The `LLM` generates a specific tip (e.g., "Focus more on the pointed ears and blue eyes").
4.  **Regenerate:** These tips are fed back into the editors to create a new $C_{edit}$ or $I_{edit}$, and the search repeats.

    ---

# 5. Experimental Setup

## 5.1. Datasets
The authors tested `WISER` on three major benchmarks:
1.  **Fashion-IQ:** A dataset of 70,000 fashion images. Queries involve changing clothing attributes (e.g., "shorter sleeves," "different pattern").
2.  **CIRR (Composed Image Retrieval on Real-life images):** Uses real-world images from the `NLVR2` dataset. It is known for being noisy and having subtle differences between images.
3.  **CIRCO:** A more recent, high-quality dataset based on `COCO`. It is unique because it provides **multiple ground truths** (multiple images can be "correct" for one query).

## 5.2. Evaluation Metrics

### 5.2.1. Recall@K (R@K)
*   **Conceptual Definition:** Measures the percentage of queries for which the correct "ground truth" image is found within the top $K$ results.
*   **Mathematical Formula:**
    \$
    \text{Recall@K} = \frac{1}{N} \sum_{i=1}^{N} \text{rel}_i(K)
    \$
*   **Symbol Explanation:**
    *   $N$: Total number of queries.
    *   $\text{rel}_i(K)$: An indicator function that is `1` if the ground truth for query $i$ is in the top $K$ results, and `0` otherwise.

### 5.2.2. mAP@K (Mean Average Precision at K)
*   **Conceptual Definition:** Used for datasets like `CIRCO` where there are multiple correct answers. It accounts for both the precision and the ranking order of correct images.
*   **Mathematical Formula:**
    \$
    \text{mAP@K} = \frac{1}{Q} \sum_{q=1}^{Q} \left( \frac{1}{m_q} \sum_{k=1}^{K} P_q(k) \times \text{rel}_q(k) \right)
    \$
*   **Symbol Explanation:**
    *   $Q$: Total number of queries.
    *   $m_q$: The number of ground truth images for query $q$.
    *   $P_q(k)$: Precision at rank $k$.
    *   $\text{rel}_q(k)$: Binary indicator (1 if image at rank $k$ is correct).

## 5.3. Baselines
The paper compares `WISER` against:
*   **Textual Inversion:** `PALAVRA`, `SEARLE`, `LinCIR`.
*   **LLM/MLLM-based Training-Free:** `CIReVL`, `LDRE`, `AutoCIR`, `CoTMR`.
*   **Diffusion-based:** `IP-CIR`.

    ---

# 6. Results & Analysis

## 6.1. Core Results Analysis
The results show that `WISER` significantly outperforms all existing training-free methods. On `CIRR`, it improved `Recall@1` by **57%** relative to previous methods. On `CIRCO`, it improved `mAP@5` by **45%**. 

A key finding is that `WISER` scales better with larger backbones (like `ViT-G/14`) than previous methods, which often "saturated" or stopped improving.

## 6.2. Data Presentation (Tables)

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Backbone / Method</th>
<th rowspan="2">Training-Free</th>
<th colspan="4">CIRCO (mAP@k)</th>
<th colspan="4">CIRR (Recall@k)</th>
</tr>
<tr>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=50</th>
<th>k=1</th>
<th>k=5</th>
<th>k=10</th>
<th>k=50</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10"><b>ViT-B/32</b></td>
</tr>
<tr>
<td>SEARLE (ICCV'23)</td>
<td>X</td>
<td>9.35</td>
<td>9.94</td>
<td>11.13</td>
<td>11.84</td>
<td>24.00</td>
<td>53.42</td>
<td>66.82</td>
<td>89.78</td>
</tr>
<tr>
<td>CoTMR (ICCV'25)</td>
<td>✓</td>
<td>22.23</td>
<td>22.78</td>
<td>24.68</td>
<td>25.74</td>
<td>31.50</td>
<td>60.80</td>
<td>73.04</td>
<td>91.06</td>
</tr>
<tr>
<td style="background-color: #f2f2f2;"><b>WISER (Ours)</b></td>
<td style="background-color: #f2f2f2;">✓</td>
<td style="background-color: #f2f2f2;"><b>32.23</b></td>
<td style="background-color: #f2f2f2;"><b>33.18</b></td>
<td style="background-color: #f2f2f2;"><b>34.82</b></td>
<td style="background-color: #f2f2f2;"><b>35.35</b></td>
<td style="background-color: #f2f2f2;"><b>49.45</b></td>
<td style="background-color: #f2f2f2;"><b>76.55</b></td>
<td style="background-color: #f2f2f2;"><b>85.21</b></td>
<td style="background-color: #f2f2f2;"><b>93.81</b></td>
</tr>
<tr>
<td colspan="10"><b>ViT-L/14</b></td>
</tr>
<tr>
<td>LinCIR (CVPR'24)</td>
<td>X</td>
<td>12.59</td>
<td>13.58</td>
<td>15.00</td>
<td>15.85</td>
<td>25.04</td>
<td>53.25</td>
<td>66.68</td>
<td>-</td>
</tr>
<tr>
<td>IP-CIR (CVPR'25)</td>
<td>✓</td>
<td>26.43</td>
<td>27.41</td>
<td>29.87</td>
<td>31.07</td>
<td>29.76</td>
<td>58.82</td>
<td>71.21</td>
<td>90.41</td>
</tr>
<tr>
<td style="background-color: #f2f2f2;"><b>WISER (Ours)</b></td>
<td style="background-color: #f2f2f2;">✓</td>
<td style="background-color: #f2f2f2;"><b>35.10</b></td>
<td style="background-color: #f2f2f2;"><b>36.30</b></td>
<td style="background-color: #f2f2f2;"><b>38.46</b></td>
<td style="background-color: #f2f2f2;"><b>39.15</b></td>
<td style="background-color: #f2f2f2;"><b>49.23</b></td>
<td style="background-color: #f2f2f2;"><b>76.72</b></td>
<td style="background-color: #f2f2f2;"><b>85.11</b></td>
<td style="background-color: #f2f2f2;"><b>94.17</b></td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The authors tested the importance of each part of `WISER`. 

The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th colspan="2">Wider Search</th>
<th rowspan="2">Deeper Thinking</th>
<th rowspan="2">Fusion Type</th>
<th colspan="2">Fashion-IQ (Avg R)</th>
<th colspan="2">CIRCO (mAP)</th>
</tr>
<tr>
<th>T2I</th>
<th>I2I</th>
<th>R@10</th>
<th>R@50</th>
<th>mAP@5</th>
<th>mAP@50</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td>✓</td>
<td>-</td>
<td>-</td>
<td>22.65</td>
<td>38.84</td>
<td>7.00</td>
<td>8.95</td>
</tr>
<tr>
<td>✓</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>28.59</td>
<td>49.18</td>
<td>17.28</td>
<td>20.51</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>-</td>
<td>AVG (Fixed)</td>
<td>33.40</td>
<td>52.92</td>
<td>13.53</td>
<td>16.77</td>
</tr>
<tr>
<td>✓</td>
<td>✓</td>
<td>-</td>
<td>ADA (Ours)</td>
<td>40.83</td>
<td>57.86</td>
<td>31.32</td>
<td>34.24</td>
</tr>
<tr style="background-color: #f2f2f2;">
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>ADA (Ours)</td>
<td><b>41.99</b></td>
<td><b>58.74</b></td>
<td><b>32.23</b></td>
<td><b>35.35</b></td>
</tr>
</tbody>
</table>

**Key Findings:**
1.  **Dual-path is crucial:** Using both `T2I` and `I2I` is significantly better than either alone.
2.  **Adaptive Fusion is the "Secret Sauce":** Replacing standard average fusion (`AVG`) with the proposed Adaptive Fusion (`ADA`) nearly doubled the `mAP@5` on `CIRCO` (from 13.53 to 31.32).
3.  **Threshold Sensitivity:** As seen in Figure 4, setting the confidence threshold $\tau$ between 0.5 and 0.7 yields the best results. If $\tau$ is too high, it wastes time refining good results; if too low, it keeps bad ones.

    ![Figure 4. Sensitivity analysis on confidence threshold $\\tau$ and refinement iteration N on CIRCO.](images/4.jpg)
    *该图像是图表，展示了在CIRCO数据集上，置信阈值 $\tau$（左侧）和细化迭代次数 $N$（右侧）对mAP和细化率的敏感性分析。左侧图表显示不同置信阈值下的mAP（精确度平均值）变化，而右侧则展示了随着细化迭代次数增加，mAP和细化率之间的关系。这些结果揭示了在不同设置下模型性能的变化趋势。*

---

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
`WISER` represents a major step forward for training-free image retrieval. By combining two complementary retrieval paths and using a visual verifier to guide fusion and refinement, it achieves performance that was previously only possible with large, expensive training datasets. It proves that "thinking" about a query and "verifying" results is more effective than simply "guessing" with a single model.

## 7.2. Limitations & Future Work
*   **Computational Cost:** `WISER` requires multiple calls to Large Language Models and Image Editors. While the authors optimized this by only refining 30% of cases, it is still slower than a single-pass `CLIP` search.
*   **Real-time applications:** Future work needs to explore how to make this pipeline fast enough for instant search results on a mobile device.
*   **Editor Dependencies:** The quality of the retrieval is capped by the quality of the image editor and captioner.

## 7.3. Personal Insights & Critique
*   **Innovation:** The "lexicographical sorting" in Adaptive Fusion is a clever way to handle ties and prioritize intent without needing complex math.
*   **Generalization:** Because it is training-free, this model can be applied to medical imaging, satellite data, or any new field where triplet data doesn't exist.
*   **Critique:** The paper uses `GPT-4o` for refinement. In a real-world setting, using such a high-end API for every search might be cost-prohibitive. It would be interesting to see if smaller, local models (like `Llama-3-8B`) could perform the "Deeper Thinking" step equally well.