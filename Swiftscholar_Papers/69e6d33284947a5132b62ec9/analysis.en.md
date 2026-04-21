# 1. Bibliographic Information

## 1.1. Title
The title of the paper is **"A Sanity Check on Composed Image Retrieval"**. This indicates that the research focuses on evaluating the validity and performance of existing methods in the field of Composed Image Retrieval (CIR), likely identifying issues with current benchmarks and proposing more rigorous evaluation standards.

## 1.2. Authors
The authors are **Yikun Liu**, **Jiangchao Yao**, **Weidi Xie**, and **Yanfeng Wang**.
*   **Affiliations:** Yikun Liu, Weidi Xie, and Yanfeng Wang are affiliated with the School of Artificial Intelligence, Shanghai Jiao Tong University, China. Jiangchao Yao is affiliated with CMIC, Shanghai Jiao Tong University, China.
*   **Research Background:** The authors appear to be researchers in computer vision and multimodal learning, specifically focusing on image retrieval, generative models, and the application of large language models in vision tasks.

## 1.3. Journal/Conference
The paper is available on **arXiv** (ID: 2604.12904) with a publication date of April 14, 2026. As of the provided text, it is a **preprint** and has not yet been assigned to a specific journal or conference volume, though the "Published at" timestamp suggests a recent submission or update. arXiv is a reputable preprint server widely used in the AI and computer science communities to disseminate research before formal peer review.

## 1.4. Publication Year
The paper was published in **2026**.

## 1.5. Abstract
The paper addresses the task of **Composed Image Retrieval (CIR)**, which aims to retrieve a target image using a query consisting of a reference image and a relative caption describing the desired modification. The authors argue that existing benchmarks are flawed because they contain "indeterminate queries" where multiple candidate images satisfy the criteria, making accurate evaluation difficult. Furthermore, current evaluations overlook multi-round interactive scenarios.
To solve this, the authors propose two main contributions:
1.  **FISD (Fully-Informed Semantically-Diverse benchmark):** A new benchmark using generative models to create precise reference-target pairs across six dimensions, eliminating query ambiguity.
2.  **Multi-round Agentic Evaluation Framework:** An automated framework to evaluate CIR models in interactive scenarios, simulating user feedback to refine queries over multiple rounds.
    Experiments show that current models struggle with negation and cardinality logic but significantly improve in multi-round interactions.

## 1.6. Original Source Link
*   **arXiv Link:** https://arxiv.org/abs/2604.12904
*   **PDF Link:** https://arxiv.org/pdf/2604.12904v1
*   **Status:** Preprint.

# 2. Executive Summary

## 2.1. Background & Motivation
**Core Problem:**
The core problem addressed in this paper is the **inadequacy of current evaluation protocols for Composed Image Retrieval (CIR)**. CIR is a task where a system retrieves a target image based on a reference image and a text instruction (e.g., "retrieve the same dress but in blue"). The authors identify two critical gaps:
1.  **Indeterminate Queries:** Popular benchmarks like FashionIQ and CIRR contain "spurious samples" or "indeterminate queries." In these cases, multiple images in the database satisfy the query criteria, not just the single ground-truth target. This ambiguity makes it impossible to rigorously judge a model's performance—if a model retrieves a valid image that isn't the specific ground truth, it is unfairly penalized.
2.  **Lack of Interactive Evaluation:** Real-world retrieval is often interactive. A user might not find the perfect result in the first try and would provide feedback to refine the search. Existing evaluations only measure "single-round" performance, ignoring how models adapt over time.

**Importance & Challenges:**
Accurate evaluation is the "sanity check" for any field. Without a reliable benchmark, it is unclear whether model improvements are genuine or just overfitting to noisy data. The challenge lies in creating a dataset that is perfectly controlled (to remove ambiguity) while remaining realistic, and simulating human interaction without the high cost of actual user studies.

**Entry Point:**
The authors leverage recent advancements in **Generative AI** (Large Language Models and Diffusion Models) to synthesize data. This allows them to control every variable in the image generation process, ensuring that for every query, there is exactly one correct answer. They also use these foundation models to simulate a "user" in a multi-round loop.

## 2.2. Main Contributions / Findings
**Primary Contributions:**
1.  **FISD Benchmark:** A novel benchmark comprising 1,200 triplets (reference image, target image, relative caption) across six semantic dimensions: **Cardinality, Addition, Negation, Change, Background, and Complex Instruction**. It uses Stable Diffusion XL for image generation and Mixtral for caption generation to ensure precise control and no ambiguity.
2.  **Multi-round Evaluation Framework:** An automated framework consisting of a CIR model, a Ranker, and a User Simulator. The User Simulator uses Multimodal Large Language Models (MLLMs) to compare the current top candidate with the target and generate a new relative caption for the next round.
3.  **Comprehensive Analysis:** The paper provides a rigorous evaluation of 11 state-of-the-art CIR methods using these new tools.

**Key Conclusions:**
*   **Semantic Weaknesses:** Current CIR models perform poorly on **Negation** and **Cardinality** semantics (e.g., "remove the cat" or "change 2 dogs to 3 dogs"). They perform better on "direct" semantics like "addition" or "change".
*   **Multi-round Potential:** CIR models have significant untapped potential. When allowed to interact over multiple rounds (up to 5), performance improves drastically (e.g., Hits@1 on CIRR improves from ~55% to ~91% for the best model), often surpassing single-round performance by a large margin.
*   **Simulator Validity:** The proposed automated user simulator correlates well with real human user behavior, validating it as a cost-effective alternative to human-in-the-loop evaluation.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several key concepts in computer vision and multimodal learning.

*   **Composed Image Retrieval (CIR):** Unlike traditional image retrieval (text-to-image or image-to-image), CIR combines both. The input is a pair $(I_{ref}, t)$, where $I_{ref}$ is a reference image and $t$ is a relative caption describing the modification. The goal is to find the target image $I_{tgt}$ in a database $\Omega$.
    *   *Example:* Reference image = "Red car", Caption = "make it blue", Target = "Blue car".

*   **Zero-shot Learning:** In machine learning, "zero-shot" refers to a model's ability to perform a task for which it has seen no explicit training examples. In CIR, zero-shot methods attempt to solve the retrieval problem without training on specific (reference, caption, target) triplets, often by mapping image features into the text feature space of pre-trained models like CLIP.

*   **Contrastive Learning:** A technique used to learn representations by bringing similar items (positive pairs) closer together in the feature space and pushing dissimilar items (negative pairs) further apart. This is the foundation of models like CLIP.

*   **Diffusion Models:** A class of generative models (e.g., Stable Diffusion) that learn to generate data by reversing a process of gradually adding noise. They are used in this paper to generate the high-quality images for the FISD benchmark.

*   **Multimodal Large Language Models (MLLMs):** Models like LLaVA or GPT-4V that can process and understand both visual and textual inputs. They are used here to act as the "eyes" of the user simulator, describing images to generate feedback.

## 3.2. Previous Works
The paper categorizes previous work into three main areas:

1.  **Training-based CIR Methods:**
    *   These methods require training on datasets containing triplets $(I_{ref}, t, I_{tgt})$.
    *   **CLIP4CIR / BLIP4CIR:** Utilize pre-trained vision-language models (like CLIP or BLIP) and fine-tune them using contrastive learning on CIR data to learn how to compose the image and text features.
    *   **SPRC (Sentence-level Prompts):** A method that uses sentence-level prompts to improve the composed representation.

2.  **Zero-shot CIR Methods:**
    *   These do not require specific CIR training data.
    *   **Pic2Word / LinCIR:** These methods learn a mapping network to project image features into the text embedding space. Once an image is "translated" to a text-like vector, it can be concatenated with the actual text caption and used for retrieval in standard text-to-image models.

3.  **Interactive Retrieval:**
    *   Previous work in interactive retrieval often focused on text-to-image or video retrieval.
    *   **Whittle-Search:** Uses tag-based feedback.
    *   **FashionIQ:** Proposed a relative captioner to generate text feedback between images.
    *   *Differentiation:* This paper proposes a fully automated, agentic framework that doesn't require training a separate feedback generator but uses off-the-shelf MLLMs to simulate the loop.

**Technological Evolution:**
The field has evolved from simple feature concatenation to sophisticated compositional learning. Initially, benchmarks were small and manually annotated. Then, synthetic data generation became popular. The current trend is moving towards **interactive agents** and using **foundation models (LLMs/MLLMs)** not just for feature extraction but for reasoning and evaluation. This paper sits at the forefront of using generative AI to *evaluate* retrieval systems, rather than just *powering* them.

## 3.3. Differentiation Analysis
The core difference between this work and previous studies is the focus on **Evaluation** rather than **Model Architecture**.
*   **Standard Approach:** Most papers propose a new network architecture (e.g., a new fusion module) and report accuracy on FashionIQ or CIRR.
*   **This Paper's Approach:** It questions the benchmarks themselves. It argues that FashionIQ/CIRR are noisy. It proposes a new "sanity check" benchmark (FISD) where the ground truth is synthetically guaranteed to be unique. It also introduces a "Multi-round" metric, which is a new dimension of evaluation (efficiency over time) that static benchmarks cannot measure.

# 4. Methodology

The paper proposes two distinct methodological components: the **FISD Benchmark** construction and the **Multi-round Evaluation Framework**.

## 4.1. Principles
The core principle is **Controlled Synthesis and Simulation**.
*   For the benchmark, the principle is: *If we generate the data, we know the ground truth is perfect.* By using generative models, the authors can ensure that for a given reference and caption, there is exactly one correct target image in the database, eliminating the "indeterminate query" problem.
*   For the evaluation framework, the principle is: *Agents can simulate users.* By using MLLMs to compare the retrieved image with the target, the system can generate the next text instruction automatically, mimicking a human refining their search.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. FISD Benchmark Construction
The construction of the Fully-Informed Semantically-Diverse (FISD) benchmark involves two main stages: Caption Generation and Image Generation.

**Step 1: Description of Categories**
The authors define six semantic dimensions to cover a wide range of retrieval logic:
1.  **Cardinality:** Changing the number of objects (e.g., "one dog" to "three dogs").
2.  **Addition:** Adding new elements.
3.  **Negation:** Removing elements.
4.  **Change:** Altering attributes (color, shape).
5.  **Background:** Changing the scene.
6.  **Complex Instruction:** Combinations of the above.

**Step 2: Caption Generation**
The authors utilize the **Mixtral-8x7B-Instruct-v0.1** Large Language Model.
*   They select a base caption for the reference image from existing datasets (like COCO).
*   They prompt the LLM to generate a "relative caption" (the modification instruction) and the "target image caption" based on the reference caption and a specific semantic category (e.g., "generate a negation instruction").
*   *Example Prompt:* "I need you to perform a reasonable editing step on an image... The semantic aspect of the instruction you generate is {Negation}."

**Step 3: Image Generation**
Images are generated using **Stable-Diffusion-xl-base-1.0**.
*   To ensure consistency between the reference and target images (so they don't look like completely different scenes), the authors use the **same random seed** for the diffusion process. This keeps the layout, lighting, and style similar while changing the specific content dictated by the captions.
*   **Hard Negatives:** To make the benchmark challenging, they also generate "hard negative" images. These are images that look very similar to the target but are incorrect (e.g., if the target is "long pants", a hard negative might be "short pants"). This forces the model to understand the subtle difference in the relative caption.

### 4.2.2. Multi-round Evaluation Framework
This framework evaluates how a CIR model performs when allowed to interact over multiple rounds. It consists of three components: the CIR Model, the Ranker, and the User Simulator.

**Step 1: Initialization**
The process starts with a reference image $I_1^{cand}$ (which is the initial reference image provided by the user) and an initial relative caption $t_1$.

**Step 2: CIR Model Feature Composition**
In the $r$-th round of interaction, the system takes the current candidate image $I_r^{cand}$ and the relative caption $t_r$.
The CIR model encodes the image and text and fuses them to create a composed query feature vector $\mathbf{q}_r$.
The formula for this composition is:
$$
\mathbf{q}_r = \text{fusion}\big( f_{\text{visual}}(\mathbf{I}_r^{\text{cand}}), f_{\text{text}}(\mathbf{t}_r) \big) \in \mathbb{R}^{1 \times d}
$$
*   $\mathbf{I}_r^{\text{cand}}$: The candidate image from the previous round (or initial reference).
*   $\mathbf{t}_r$: The relative caption for the current round.
*   $f_{\text{visual}}$: The visual encoder (e.g., CLIP ViT).
*   $f_{\text{text}}$: The text encoder.
*   $\text{fusion}(\cdot, \cdot)$: The specific fusion method of the CIR model (e.g., concatenation, element-wise sum, or a learned attention layer).
*   $d$: The dimension of the feature vector.

**Step 3: Ranker with History**
The Ranker is responsible for selecting the best image from the database $\Omega$. Crucially, it maintains a **history list** $\mathbf{H} = [\mathbf{q}_1, \dots, \mathbf{q}_r]$ of all query features from previous rounds.
The Ranker computes a "history representation" by averaging the features in the history list:
$$
f_h(\mathbf{H}) = \frac{1}{r} \sum_{i=1}^{r} \mathbf{q}_i
$$
*   $f_h(\mathbf{H})$: The aggregated query feature representing the user's intent over all rounds.
*   $r$: The current round number.

    The Ranker then calculates the similarity (e.g., cosine similarity) between this history feature and all image features $\mathbb{V} = \{v_i\}_{i=1}^N$ in the database. It selects the image with the highest similarity (or smallest distance) as the next candidate:
$$
\mathbf{I}_{r+1}^{\text{cand}} = \mathop{\mathrm{argmin}}_{v \in \mathbb{V}} \mathrm{sim}(f_h(\mathbf{H}), v)
$$
*   $\mathbf{I}_{r+1}^{\text{cand}}$: The candidate image selected for the next round.
*   $\mathrm{sim}(\cdot, \cdot)$: The similarity metric (cosine distance).

    ![该图像是示意图，展示了多轮检索的过程和特定轮次的执行机制。左侧显示用户请求和系统回应的多轮交互，右侧描述了CIR模型如何根据用户历史和图像特征进行排名。](images/3.jpg)
    *The figure above illustrates the multi-round interaction workflow. The left side shows the iterative process where the user (simulator) provides feedback, and the system updates the candidate. The right side details the internal mechanism where the Ranker uses the history of composed queries to find the next best match.*

    **Step 4: User Simulator**
If the target image $\mathbf{I}^{\text{tgt}}$ is not in the top-$k$ results, the User Simulator generates the next relative caption $\mathbf{t}_{r+1}$.
This simulator uses a **Multimodal Large Language Model (MLLM)** to describe the current candidate and the target, and a **Large Language Model (LLM)** to generate the difference text.
The formula for generating the feedback is:
$$
\mathbf{t}_{r+1} = \mathtt{LLM} \left( \mathtt{MLLM}(\mathbf{I}_{r+1}^{\text{cand}}), \mathtt{MLLM}(\mathbf{I}^{\text{tgt}}) \right)
$$
*   $\mathtt{MLLM}(\cdot)$: Generates a detailed text description of the input image.
*   $\mathtt{LLM}(\cdot)$: Takes two descriptions (candidate and target) and outputs a relative caption describing how to change the candidate to look like the target.

**Step 5: Loop**
The process repeats from Step 2 using the new candidate image $\mathbf{I}_{r+1}^{\text{cand}}$ and the new caption $\mathbf{t}_{r+1}$ until the target is found or the maximum number of rounds $R_{\text{max}}$ is reached.

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted on four benchmarks to ensure comprehensive evaluation:
1.  **CIRR:** A standard real-world benchmark for Composed Image Retrieval.
2.  **FashionIQ:** A benchmark focused on fashion items (Shirt, Dress, Top), requiring relative captions like "change the color to red".
3.  **CIRCO:** Another zero-shot CIR benchmark.
4.  **FISD:** The proposed **Fully-Informed Semantically-Diverse benchmark**.
    *   **Scale:** 1,200 triplets in total (200 per semantic category).
    *   **Characteristics:** Synthetic images generated by SDXL. Each category (Cardinality, Negation, etc.) forms an independent database of 600 images (including hard negatives).
    *   **Why Chosen:** FISD is chosen to provide a "sanity check" free from the ambiguity of real-world datasets. It allows the authors to isolate specific semantic capabilities (like negation) that are hard to test in noisy data.

## 5.2. Evaluation Metrics
The paper uses standard retrieval metrics adapted for the multi-round setting.

1.  **Recall@K (Recall at K):**
    *   **Conceptual Definition:** This metric measures the ability of the system to find the correct target image within the top $K$ retrieved results. For example, Recall@1 checks if the very first result is correct; Recall@5 checks if the correct result is among the top 5.
    *   **Mathematical Formula:**
        $$
        \text{Recall@K} = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \mathbb{I}(\text{rank}(I_i^{\text{tgt}}) \leq K)
        $$
    *   **Symbol Explanation:**
        *   $|\mathcal{D}|$: The total number of queries in the dataset.
        *   $I_i^{\text{tgt}}$: The ground truth target image for the $i$-th query.
        *   $\text{rank}(\cdot)$: The rank position of the target image in the retrieved list (1 is the best).
        *   $\mathbb{I}(\cdot)$: An indicator function that returns 1 if the condition is true and 0 otherwise.

2.  **Hits@K (Multi-round):**
    *   **Conceptual Definition:** Used specifically for the multi-round evaluation. It measures success if the target image appears in the top $K$ results in **any** round up to the current round $r$. This captures the cumulative success of the interactive process.
    *   **Mathematical Formula:**
        $$
        \text{Hits@K}_r = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \mathbb{I}\left( \min_{1 \leq j \leq r} \text{rank}_j(I_i^{\text{tgt}}) \leq K \right)
        $$
    *   **Symbol Explanation:**
        *   $r$: The current round number.
        *   $\text{rank}_j(\cdot)$: The rank of the target image in the $j$-th round.
        *   The condition checks if the target was ever in the top $K$ during rounds 1 through $r$.

## 5.3. Baselines
The paper evaluates 11 representative CIR methods to ensure the findings are generalizable.
*   **Zero-shot Methods:** Pic2Word, Context-I2W, LinCIR. These are chosen to represent the class of methods that don't require specific CIR training.
*   **Training-based Methods:** TransAgg, CLIP4CIR, BLIP4CIR+Bi, TG-CIR, CoVR*, CIReVL, SPRC, SPN4CIR. These are chosen as they represent the state-of-the-art in supervised CIR, including different architectures (compositional, mapping, etc.).

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experiments reveal two major insights: the specific semantic weaknesses of current models and the massive potential of multi-round retrieval.

**FISD Benchmark Analysis (Table 1):**
The results on the synthetic FISD benchmark highlight that while models are good at direct changes, they fail at logic.
*   **Negation & Cardinality:** Almost all models score near 0% on Negation and very low on Cardinality. For instance, even the best model (SPN4CIR) only gets 13.0% Recall@1 on Negation. This suggests that current vision-language encoders struggle to understand "remove" or "count" operations effectively.
*   **Direct Semantics:** Models perform much better on "Addition" (up to 69.5%) and "Change" (up to 77.0%). These are easier because the relative caption often explicitly names the new feature.
*   **Overall:** The average Recall@1 hovers around 30-55%, indicating there is still significant room for improvement even in "easy" synthetic tasks.

**Multi-round Analysis (Table 2):**
The multi-round results demonstrate that interaction is a powerful tool.
*   **Significant Improvement:** Every single model improves across all benchmarks when moving from Round 1 to Round 3 or 5. For example, on the CIRR dataset, the SPRC model improves Hits@1 from 55.39% to 81.37% in Round 3, and to 89.76% in Round 5.
*   **Diminishing Returns:** The gain is largest between Round 1 and Round 3. Between Round 3 and Round 5, the improvement is smaller, suggesting the model converges on the correct answer.
*   **Consistency:** The trend holds true across FashionIQ (Dress, Shirt, Top), CIRR, and the synthetic FISD dataset.

## 6.2. Data Presentation (Tables)

The following are the results from Table 1 of the original paper:

| Model | Cardinality (Recall@1) | Addition (Recall@1) | Negation (Recall@1) | Change (Recall@1) | Background (Recall@1) | Complex Inst. (Recall@1) | Average (Recall@1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Pic2Word [39] | 6.00 | 39.00 | 0.00 | 24.00 | 18.00 | 41.00 | 21.33 |
| Context-I2W [44] | 14.00 | 48.00 | 0.00 | 29.50 | 27.50 | 48.00 | 27.83 |
| LinCIR [10] | 5.50 | 52.50 | 1.50 | 43.50 | 36.50 | 55.00 | 32.42 |
| TransAgg [28] | 15.50 | 56.00 | 2.00 | 52.00 | 26.00 | 57.00 | 34.75 |
| CLIP4CIR [4] | 31.00 | 67.50 | 4.00 | 65.00 | 43.50 | 72.00 | 47.17 |
| BLIP4CIR+Bi [32] | 35.50 | 65.50 | 3.50 | 58.50 | 42.50 | 65.50 | 45.17 |
| TG-CIR [49] | 43.50 | 69.00 | 1.50 | 64.50 | 47.50 | 65.50 | 48.58 |
| CoVR* [47] | 47.50 | 70.00 | 8.50 | 77.00 | 61.50 | 70.00 | 55.75 |
| CIReVL [15] | 24.00 | 53.50 | 17.50 | 61.00 | 40.00 | 69.00 | 44.17 |
| SPRC [2] | 41.00 | 66.50 | 9.50 | 73.00 | 48.50 | 62.00 | 50.08 |
| SPN4CIR [8] | 58.50 | 69.50 | 13.00 | 77.00 | 56.00 | 61.00 | 55.83 |

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">FashionIQ-Dress</th>
<th colspan="2">FashionIQ-Shirt</th>
<th colspan="2">FashionIQ-Toptee</th>
<th colspan="2">CIRR</th>
<th>CIRCO</th>
<th>FISD</th>
</tr>
<tr>
<th>Hits@10</th>
<th>Hits@50</th>
<th>Hits@10</th>
<th>Hits@50</th>
<th>Hits@10</th>
<th>Hits@50</th>
<th>Hits@1</th>
<th>Hits@5</th>
<th>MAP@5</th>
<th>Hits@1</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="11"><strong>Round 1</strong></td>
</tr>
<tr>
<td>Pic2Word [39]</td>
<td>20.00</td>
<td>40.20</td>
<td>26.20</td>
<td>43.60</td>
<td>27.90</td>
<td>47.40</td>
<td>23.25</td>
<td>51.42</td>
<td>7.39</td>
<td>21.33</td>
</tr>
<tr>
<td>Context-I2W [44]</td>
<td>23.10</td>
<td>45.30</td>
<td>29.70</td>
<td>48.60</td>
<td>30.60</td>
<td>52.90</td>
<td>26.96</td>
<td>56.59</td>
<td>11.96</td>
<td>27.83</td>
</tr>
<tr>
<td>LinCIR [10]</td>
<td>20.92</td>
<td>42.44</td>
<td>29.10</td>
<td>46.81</td>
<td>28.81</td>
<td>50.18</td>
<td>25.09</td>
<td>54.41</td>
<td>10.61</td>
<td>32.42</td>
</tr>
<tr>
<td>TransAgg [28]</td>
<td>30.24</td>
<td>51.91</td>
<td>34.45</td>
<td>53.97</td>
<td>38.40</td>
<td>59.51</td>
<td>38.79</td>
<td>69.58</td>
<td>11.06</td>
<td>34.75</td>
</tr>
<tr>
<td>CLIP4CIR [4]</td>
<td>39.46</td>
<td>64.55</td>
<td>44.41</td>
<td>65.26</td>
<td>47.48</td>
<td>70.98</td>
<td>45.37</td>
<td>78.47</td>
<td>10.35</td>
<td>47.17</td>
</tr>
<tr>
<td>BLIP4CIR+Bi [32]</td>
<td>42.09</td>
<td>67.33</td>
<td>41.76</td>
<td>64.28</td>
<td>46.61</td>
<td>70.32</td>
<td>42.36</td>
<td>75.48</td>
<td>10.09</td>
<td>45.17</td>
</tr>
<tr>
<td>SPRC [2]</td>
<td>49.18</td>
<td>72.43</td>
<td>55.64</td>
<td>73.89</td>
<td>59.35</td>
<td>78.58</td>
<td>55.39</td>
<td>84.26</td>
<td>21.17</td>
<td>50.08</td>
</tr>
<tr>
<td>SPN4CIR [8]</td>
<td>50.57</td>
<td>74.12</td>
<td>57.70</td>
<td>75.27</td>
<td>60.84</td>
<td>79.96</td>
<td>56.47</td>
<td>85.29</td>
<td>21.78</td>
<td>55.83</td>
</tr>
<tr>
<td colspan="11"><strong>Round 3</strong></td>
</tr>
<tr>
<td>Pic2Word [39]</td>
<td>29.65</td>
<td>55.08</td>
<td>39.99</td>
<td>59.76</td>
<td>39.16</td>
<td>61.75</td>
<td>40.28</td>
<td>71.42</td>
<td>12.93</td>
<td>64.67</td>
</tr>
<tr>
<td>Context-I2W [44]</td>
<td>39.86</td>
<td>64.20</td>
<td>49.71</td>
<td>69.28</td>
<td>51.35</td>
<td>73.79</td>
<td>43.05</td>
<td>75.27</td>
<td>21.14</td>
<td>50.25</td>
</tr>
<tr>
<td>LinCIR [10]</td>
<td>33.66</td>
<td>59.94</td>
<td>51.03</td>
<td>69.92</td>
<td>48.24</td>
<td>68.59</td>
<td>43.03</td>
<td>75.68</td>
<td>17.64</td>
<td>51.08</td>
</tr>
<tr>
<td>TransAgg [28]</td>
<td>48.79</td>
<td>72.88</td>
<td>60.06</td>
<td>79.34</td>
<td>65.73</td>
<td>83.53</td>
<td>61.83</td>
<td>89.38</td>
<td>27.22</td>
<td>77.67</td>
</tr>
<tr>
<td>CLIP4CIR [4]</td>
<td>53.74</td>
<td>77.99</td>
<td>65.95</td>
<td>85.18</td>
<td>68.08</td>
<td>87.10</td>
<td>67.54</td>
<td>95.72</td>
<td>23.57</td>
<td>74.25</td>
</tr>
<tr>
<td>BLIP4CIR+Bi [32]</td>
<td>59.54</td>
<td>81.95</td>
<td>57.56</td>
<td>79.49</td>
<td>64.41</td>
<td>83.02</td>
<td>54.99</td>
<td>89.48</td>
<td>17.70</td>
<td>70.58</td>
</tr>
<tr>
<td>SPRC [2]</td>
<td>67.38</td>
<td>85.52</td>
<td>77.13</td>
<td>80.37</td>
<td>78.74</td>
<td>91.74</td>
<td>81.37</td>
<td>97.32</td>
<td>36.86</td>
<td>77.08</td>
</tr>
<tr>
<td>SPN4CIR [8]</td>
<td>70.70</td>
<td>88.00</td>
<td>84.45</td>
<td>94.80</td>
<td>87.25</td>
<td>95.72</td>
<td>84.31</td>
<td>97.66</td>
<td>38.02</td>
<td>81.00</td>
</tr>
<tr>
<td colspan="11"><strong>Round 5</strong></td>
</tr>
<tr>
<td>Pic2Word [39]</td>
<td>33.12</td>
<td>59.05</td>
<td>43.42</td>
<td>64.13</td>
<td>42.22</td>
<td>65.37</td>
<td>57.47</td>
<td>75.60</td>
<td>17.19</td>
<td>83.50</td>
</tr>
<tr>
<td>Context-I2W [44]</td>
<td>44.42</td>
<td>68.12</td>
<td>55.69</td>
<td>74.19</td>
<td>56.81</td>
<td>78.68</td>
<td>56.83</td>
<td>78.71</td>
<td>26.89</td>
<td>69.08</td>
</tr>
<tr>
<td>LinCIR [10]</td>
<td>37.04</td>
<td>62.96</td>
<td>56.43</td>
<td>74.93</td>
<td>51.91</td>
<td>72.41</td>
<td>58.48</td>
<td>79.14</td>
<td>21.51</td>
<td>68.92</td>
</tr>
<tr>
<td>TransAgg [28]</td>
<td>53.59</td>
<td>76.85</td>
<td>67.12</td>
<td>84.25</td>
<td>73.33</td>
<td>89.14</td>
<td>78.04</td>
<td>92.54</td>
<td>32.26</td>
<td>88.33</td>
</tr>
<tr>
<td>CLIP4CIR [4]</td>
<td>57.86</td>
<td>81.51</td>
<td>70.71</td>
<td>88.42</td>
<td>72.67</td>
<td>89.95</td>
<td>82.92</td>
<td>97.06</td>
<td>29.32</td>
<td>82.08</td>
</tr>
<tr>
<td>BLIP4CIR+Bi [32]</td>
<td>63.41</td>
<td>84.28</td>
<td>61.83</td>
<td>82.09</td>
<td>67.87</td>
<td>86.18</td>
<td>67.11</td>
<td>91.32</td>
<td>20.71</td>
<td>80.42</td>
</tr>
<tr>
<td>SPRC [2]</td>
<td>70.40</td>
<td>87.70</td>
<td>80.72</td>
<td>92.54</td>
<td>83.32</td>
<td>93.98</td>
<td>89.76</td>
<td>98.49</td>
<td>41.93</td>
<td>84.92</td>
</tr>
<tr>
<td>SPN4CIR [8]</td>
<td>75.21</td>
<td>90.83</td>
<td>84.45</td>
<td>94.80</td>
<td>87.25</td>
<td>95.72</td>
<td>91.32</td>
<td>98.76</td>
<td>40.79</td>
<td>86.92</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The authors perform several ablation studies to validate the design of their multi-round framework (Table 6).

**1. Importance of History:**
*   **No History:** Using only the current round's feature (without averaging with previous rounds) results in worse performance (Rank 2.93 vs 1.79 in Round 3). This proves that maintaining context over rounds is crucial.

**2. Importance of Varied Feedback:**
*   **Cap. Unchanged:** Reusing the *same* relative caption from Round 1 in subsequent rounds causes performance to degrade significantly (Rank jumps to 9.64). This confirms that the *new* feedback generated by the simulator is essential for guiding the search.

**3. Selection Strategy:**
*   **Top-10 random:** Randomly selecting a candidate from the top-10 results to be the next reference image yields similar results to always picking the top-1. This suggests the system is robust as long as the candidate is reasonably good.

**4. User Simulator vs. Real User (Table 5):**
*   The authors compared their automated simulator with real human users. Both showed significant improvement in ranking (reducing rank by ~2000 positions). Interestingly, the simulator sometimes outperformed humans because it provides more detailed/descriptive feedback, whereas humans tend to be brief. This validates the simulator as a rigorous (and potentially stricter) evaluation tool.

    ![Figure 4. Performance across various rounds on FashionIQ.](images/4.jpg)
    *Figure 4 illustrates the performance trends across different rounds on the FashionIQ dataset. It confirms that performance gains are most significant in the early rounds (1 to 3) and start to plateau as the number of rounds increases beyond 5.*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully argues that the evaluation of Composed Image Retrieval (CIR) models requires a paradigm shift. By introducing the **FISD benchmark**, the authors demonstrated that current state-of-the-art models have significant blind spots, particularly in handling **negation** and **cardinality** logic, despite performing well on simpler tasks. Furthermore, the proposed **multi-round agentic evaluation framework** revealed that CIR models are far more capable than single-round metrics suggest; when allowed to interact and refine their queries, performance improves dramatically. This work provides a more rigorous, "sanity-checked" way to assess and drive future research in CIR.

## 7.2. Limitations & Future Work
**Limitations:**
*   **FISD Scale:** The FISD benchmark currently contains only 1,200 triplets. While high-quality, this is relatively small compared to massive real-world datasets. Scaling this synthetic generation process is a challenge.
*   **Synthetic Gap:** Although the authors validated FISD against real-world data, there is always a concern about the "domain gap" between AI-generated images (even high-quality ones) and natural photographs.
*   **Framework Scope:** The current framework evaluates *existing* CIR models in a multi-round setting. It does not propose a new model architecture specifically optimized for this multi-round interaction.

**Future Work:**
*   **Scaling FISD:** Developing methods to scale the benchmark to millions of images to better stress-test models.
*   **Multi-round CIR Models:** Designing new CIR architectures that are explicitly trained to handle multi-round history and feedback, rather than just being evaluated in that setting.
*   **Cost Reduction:** While the simulator is cheaper than humans, running MLLMs (like LLaVA) for every round is computationally expensive. Optimizing this pipeline is necessary for practical deployment.

## 7.3. Personal Insights & Critique
**Inspiration:**
The most inspiring aspect of this paper is its use of **Generative AI to evaluate Discriminative AI**. Instead of just using Stable Diffusion to generate art, they use it to generate "test cases" (benchmarks). This is a powerful meta-application of generative models. The multi-round framework effectively turns a static retrieval task into a dynamic dialogue, which is much closer to how humans actually search for things (e.g., shopping).

**Critique:**
*   **Complexity of Negation:** The failure on "negation" is a profound finding. It suggests that current CLIP-like embeddings might be biased towards "presence" (what is in the image) rather than "absence" (what is not). This points to a fundamental need for new loss functions or training objectives that penalize false positives more heavily.
*   **Simulator Detail:** The finding that the simulator outperforms humans because it is "more verbose" is interesting. It raises a question: should we optimize models for natural, brief human feedback, or for the idealized, detailed feedback an AI agent can provide? The paper suggests the latter might be a better target for maximizing retrieval accuracy.
*   **Validity of Synthetic Data:** The authors did a good job checking if FISD results correlate with CIRR (they do). However, one must be careful not to optimize models solely for FISD, as "perfect" synthetic data might lack the long-tail noise and messiness of the real world, which models often have to be robust against.

**Transferability:**
The evaluation framework is highly transferable. It could be applied to **Video Retrieval**, **Audio Retrieval**, or even **Text-to-Image Generation** (e.g., a multi-round image editing loop where the user refines the generated image). The concept of an "agentic evaluator" is a general-purpose tool for the next generation of AI systems.