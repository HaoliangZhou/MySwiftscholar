# 1. Bibliographic Information
## 1.1. Title
The paper focuses on **Zero-Shot Composed Image Retrieval (ZS-CIR)** via automatic multi-agent collaboration. Its full title is *Generative Thinking, Corrective Action: User-Friendly Composed Image Retrieval via Automatic Multi-Agent Collaboration*. The core topic is designing a training-free, closed-loop multi-agent framework to address the rigidity and poor user experience of existing ZS-CIR methods.
## 1.2. Authors
The primary authors and their affiliations are:
- Zhangtao Cheng, Yuhao Ma, Jian Lang, Ting Zhong, Fan Zhou: University of Electronic Science and Technology of China (UESTC), Chengdu, China. Fan Zhou is the corresponding author, affiliated with the Key Laboratory of Intelligent Digital Media Technology of Sichuan Province.
- Kunpeng Zhang: University of Maryland, College Park, United States.
- Yong Wang: Aiwen Tech (Zhengzhou, China) and Hong Kong University of Science and Technology (Hong Kong, China).
  The research team has expertise in vision-language tasks, multi-agent systems, and information retrieval.
## 1.3. Journal/Conference
The paper is accepted for publication at the **31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2025)**, held in Toronto, Canada. KDD is the top-tier, highest-impact academic conference in the fields of data mining, knowledge discovery, and applied machine learning, with an acceptance rate of ~15% for full papers.
## 1.4. Publication Year
2025 (upcoming conference publication).
## 1.5. Abstract
The paper targets ZS-CIR: a task that retrieves images matching a composed query of a reference image + modification text, without training on labeled triplet (reference image, modification text, target image) datasets. Existing ZS-CIR methods rely on fixed hand-crafted templates to combine image and text signals, suffering from two critical limitations: non-adaptive retrieval queries that fail on complex modification scenarios, and user-unfriendly rigid pipelines that require manual trial-and-error.
To address these gaps, the authors propose **AutoCIR (Automatic multi-agent collaboration for zero-shot Composed Image Retrieval)**, a training-free framework with three specialized agents: a planner, a retriever, and a corrector. The planner generates and iteratively refines target image captions for retrieval. The retriever performs cross-modal zero-shot retrieval using pre-trained Vision-Language Models (VLMs). The corrector uses Chain-of-Thought (CoT) reasoning to evaluate retrieval results and generate correction feedback for the planner. Extensive experiments on three standard ZS-CIR benchmarks show AutoCIR consistently outperforms all prior state-of-the-art (SOTA) methods.
## 1.6. Original Source Link
- Original source: `uploaded://87da78a0-9b2a-4f07-b55a-a25833185128`
- PDF link: `/files/papers/69e2f556e63e77db717ca479/paper.pdf`
- Publication status: Officially accepted for publication at KDD 2025. The code is open-sourced at https://github.com/coloreyes/AutoCIR.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
ZS-CIR is a high-value vision-language task with applications in e-commerce product search, content creation, and image database management. Unlike supervised Composed Image Retrieval (CIR) that requires expensive manually annotated triplet datasets, ZS-CIR eliminates the need for task-specific training, making it far more deployable in real-world scenarios.
### Limitations of Prior Work
As shown in Figure 1 below, existing ZS-CIR methods follow a rigid single-pass pipeline:
1.  Map the reference image to pseudo-text tokens via a pre-trained or fine-tuned textual inversion module.
2.  Combine the pseudo-text tokens with modification text via fixed hand-crafted templates (e.g., "a photo of [reference content] that [modification]") to generate a target retrieval caption.
3.  Perform retrieval once using a VLM.
    This pipeline has two critical flaws:
1.  **Non-adaptive retrieval queries**: Fixed templates fail to handle complex modifications (e.g., object removal, multi-attribute adjustment, context-aware changes), leading to semantic mismatch between the generated caption and user intent.
2.  **User-unfriendly process**: If retrieval results are incorrect, users must manually edit the modification text and re-run the pipeline, leading to tedious trial-and-error cycles.

    ![Figure 1: Previous textual inversion method vs. our work.](images/1.jpg)
    *该图像是示意图，展示了以前的文本反演方法与我们提出的方法之间的对比。在图中，两个过程分别展示了如何生成、检索和评估图像，同时说明了各步骤的改进。我们的方法通过合并多个智能体的协作，能够更灵活地进行图像检索。*

### Innovative Idea
The paper draws inspiration from human problem-solving workflows: humans first create a plan for a task, execute it, check if results meet requirements, refine the plan if errors are found, and repeat until satisfactory. The authors translate this workflow into a closed-loop multi-agent system for ZS-CIR, which requires no task-specific training and enables automatic iterative self-correction.
## 2.2. Main Contributions / Findings
The paper makes three key contributions:
1.  **Novel Framework**: It proposes AutoCIR, the first closed-loop multi-agent collaboration framework for ZS-CIR. The framework automatically identifies and rectifies retrieval mismatches via iterative agent interactions, requiring no additional training data or fine-tuning.
2.  **Plug-and-Play Agent Design**: The three core agents (planner, retriever, corrector) are model-agnostic: they can be integrated with any pre-trained LLM, image captioning model, and VLM retrieval backbone. The entire pipeline operates in natural language, making it fully interpretable and user-friendly.
3.  **SOTA Performance**: Extensive experiments on three standard ZS-CIR benchmarks (CIRCO, CIRR, FashionIQ) show AutoCIR outperforms all prior competitive methods. For example, it achieves an average 7.26% improvement in Recall@10 on the FashionIQ e-commerce dataset compared to the previous SOTA.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
Below are core concepts required to understand the paper, explained for beginners:
- **Composed Image Retrieval (CIR)**: A vision-language task where the input is a *composed query*: a reference image + a natural language modification instruction. The goal is to retrieve images that retain most characteristics of the reference image while incorporating the requested modification. For example, if the reference is an image of a red short-sleeved shirt and the modification is "change color to blue, make it long-sleeved", the target is an image of a blue long-sleeved shirt with the same style as the reference.
- **Zero-Shot Composed Image Retrieval (ZS-CIR)**: A variant of CIR where the model is not trained on labeled triplet datasets (reference image, modification text, target image). Labeled triplets are extremely labor-intensive to curate, so ZS-CIR eliminates this requirement to enable faster real-world deployment.
- **Vision-Language Model (VLM)**: Pre-trained models trained on millions of image-text pairs that map images and text to a shared high-dimensional embedding space. The cosine similarity between an image embedding and a text embedding in this space measures how well the text describes the image. Common examples include CLIP, BLIP, BLIP-2, and CoCa.
- **Large Language Model (LLM)**: Pre-trained autoregressive language models trained on trillions of text tokens, with strong natural language understanding, reasoning, and generation capabilities. Common examples include GPT-3.5, GPT-4, Qwen, and LLaMA.
- **Chain-of-Thought (CoT) Reasoning**: A prompting technique that asks LLMs to generate step-by-step intermediate reasoning before outputting a final answer. This significantly improves LLM performance on complex reasoning tasks (e.g., evaluation, error detection) by reducing hallucination and improving output consistency.
- **Textual Inversion**: A technique used in prior ZS-CIR methods that maps reference image embeddings to pseudo-text tokens in the VLM's text embedding space. These pseudo-tokens are then combined with modification text to generate retrieval queries, eliminating the need for triplet training.
## 3.2. Previous Works
The paper groups related work into two core categories:
### Compositional Image Retrieval
1.  **Supervised CIR (2017-2022)**: Early CIR methods rely entirely on labeled triplet datasets for training. For example, Vo et al. (2019) proposed supervised fusion architectures for combining image and text features for retrieval. These methods achieve strong performance but are limited by the high cost of curating triplet annotations.
2.  **ZS-CIR (2022-present)**: To eliminate triplet training requirements, ZS-CIR methods use textual inversion to map reference images to pseudo-text tokens. Core prior works include:
    - *Pic2word (2023)*: Trains a mapping network to convert image features to pseudo-word tokens, which are fused with modification text for retrieval.
    - *SEARLE (2023)*: Uses optimization-based textual inversion to generate pseudo-tokens for unlabeled images, then distills this into a lightweight mapping network.
    - *CIReVL (2024)*: The first LLM-based ZS-CIR method, which uses a pre-trained captioning model to generate a natural language caption for the reference image, then uses an LLM to edit this caption according to the modification text for retrieval. It requires no training but follows a rigid single-pass pipeline with no self-correction.
    - *LDRE (2024)*: An LLM-based ZS-CIR method that uses divergent reasoning to generate multiple candidate target captions and ensembles retrieval results. It also follows a single-pass pipeline with no iterative refinement.
### Standard ZS-CIR Pipeline
All prior ZS-CIR methods follow the same core pipeline, formalized as:
1.  Given a composed query $\{\mathcal{I}_q, \mathcal{T}_q\}$ where $\mathcal{I}_q$ is the reference image and $\mathcal{T}_q$ is the modification text, use a mapping function $\phi$ to project $\mathcal{I}_q$ to pseudo-text tokens $\mathcal{Z}_q$: `\mathcal{Z}_q = \phi(\mathcal{I}_q)`.
2.  Combine $\mathcal{Z}_q$ and $\mathcal{T}_q$ via a fixed hand-crafted prompt to generate a target caption $\mathcal{T}_t$, e.g., $\mathcal{T}_t = \text{"a photo of } \mathcal{Z}_q \text{ that } \mathcal{T}_q \text{"}$.
3.  Use a VLM to encode all candidate images in the corpus $\mathcal{D}$ with its image encoder $\Psi_i$, encode $\mathcal{T}_t$ with its text encoder $\Psi_t$. Retrieve the image with the highest cosine similarity to $\mathcal{T}_t$:
    $$
    \mathcal{I}_t = \underset{I \in \mathcal{D}}{\operatorname{argmax}} \cos(\Psi_i(I), \Psi_t(\mathcal{T}_t)) = \underset{I \in \mathcal{D}}{\operatorname{argmax}} \frac{\Psi_i(I)^\top \Psi_t(\mathcal{T}_t)}{\|\Psi_i(I)\| \cdot \|\Psi_t(\mathcal{T}_t)\|}
    $$
    Where $\Psi_i(I)^\top$ denotes the transpose of the image embedding vector, and $\|\cdot\|$ denotes the L2 norm of the vector.
## 3.3. Technological Evolution
The evolution of CIR technology follows a clear trajectory:
1.  **2017-2022: Supervised CIR**: Requires large labeled triplet datasets, limited scalability.
2.  **2022-2023: Textual Inversion ZS-CIR**: Eliminates triplet training requirements, but relies on fixed template query construction, fails on complex modification scenarios.
3.  **2024: Single-Pass LLM-based ZS-CIR**: Uses LLMs to generate flexible target captions, but lacks self-correction mechanisms, still requires manual user intervention for failed retrievals.
4.  **2025: AutoCIR**: Introduces closed-loop multi-agent iterative self-correction, eliminates manual trial-and-error, achieves SOTA performance while remaining training-free.
## 3.4. Differentiation Analysis
Compared to prior ZS-CIR methods, AutoCIR has three core unique advantages:
1.  **No fixed templates**: Instead of rigid prompt-based query construction, the planner agent dynamically generates and refines target captions to adapt to complex modification scenarios.
2.  **Closed-loop self-correction**: The CoT corrector agent automatically identifies retrieval mismatches and generates correction feedback, eliminating the need for manual user trial-and-error.
3.  **Full interpretability**: The entire pipeline operates in natural language, so users can easily understand the retrieval process and intervene directly on the target caption if needed, significantly improving user experience.

# 4. Methodology
## 4.1. Principles
The core principle of AutoCIR is mimicking human iterative problem-solving workflows. The framework uses three specialized, training-free agents that interact in a closed loop:
1.  The planner creates an initial retrieval plan (target caption) based on the composed query, then refines the plan based on correction feedback.
2.  The retriever executes the retrieval plan using a pre-trained VLM.
3.  The corrector evaluates retrieval results, identifies mismatches with user intent, and generates specific correction feedback for the planner.
    The loop repeats until retrieval results meet requirements or a maximum number of iterations is reached. The overall framework architecture is shown in Figure 2 below:

    ![该图像是一个示意图，展示了自动多智能体协作的零-shot 组合图像检索框架（AutoCIR）。图中包括参考图像和修改文本的流程，涵盖了图像说明、跨模态检索及评估的三个主要步骤。各步骤通过反思规划者、检索器和思维链校正器协同工作，解决检索过程中的不匹配问题。](images/2.jpg)
    *该图像是一个示意图，展示了自动多智能体协作的零-shot 组合图像检索框架（AutoCIR）。图中包括参考图像和修改文本的流程，涵盖了图像说明、跨模态检索及评估的三个主要步骤。各步骤通过反思规划者、检索器和思维链校正器协同工作，解决检索过程中的不匹配问题。*

## 4.2. Core Methodology In-depth
### Formal Preliminaries
The goal of ZS-CIR is defined as: Given a composed query $\{\mathcal{I}_q, \mathcal{T}_q\}$ where $\mathcal{I}_q$ is the reference image and $\mathcal{T}_q$ is the modification text, retrieve a target image $\mathcal{I}_t$ from an image corpus $\mathcal{D}$ that incorporates all modifications specified in $\mathcal{T}_q$ while retaining all other visual characteristics of $\mathcal{I}_q$.
AutoCIR consists of three core agents, detailed below:
---
### 4.2.1. Reflective Planner
The planner is responsible for generating and iteratively refining high-quality target image captions aligned with user intent. It operates in three stages:
#### Stage 1: Visual Perception
Instead of using complex trained textual inversion modules to map the reference image to pseudo-text tokens, the planner uses a pre-trained image captioning model $\varphi$ (e.g., BLIP-2) to generate a natural language description $C_q$ of the reference image $\mathcal{I}_q$:
$$
C_q = \varphi(\mathcal{I}_q)
$$
This converts visual signals to interpretable natural language, eliminating the need for task-specific training and enabling transparent user oversight of the retrieval process.
#### Stage 2: Recaption
Instead of combining the reference caption $C_q$ and modification text $\mathcal{T}_q$ via fixed templates, the planner uses an LLM-based modifier $\Phi_p$ with a custom prompt $\mathcal{P}$ to generate a unified initial target caption $C_q^t$:
$$
C_q^t = \Phi_p(\mathcal{P} \circ C_q \circ \mathcal{T}_q)
$$
Where $\circ$ denotes string concatenation. The prompt $\mathcal{P}$ (provided in the paper's appendix) instructs the LLM to edit the reference caption according to the modification text, keep the output concise, and avoid adding imaginary content. The resulting caption fully integrates the reference image content and requested modifications without rigid template constraints.
#### Stage 3: Refining
After receiving correction feedback $S_q$ from the corrector agent, the planner uses a refining prompt $\mathcal{P}_r$ to adjust the initial caption to generate a refined target caption $C_q^{re}$:
$$
C_q^{re} = \Phi_p(\mathcal{P}_r \circ C_q^t \circ S_q)
$$
The refinement stage repeats until the corrector confirms retrieval results are satisfactory, or the maximum number of iterations $N$ is reached.
---
### 4.2.2. Cross-Modal Retriever
The retriever uses a pre-trained VLM (e.g., CLIP) to perform zero-shot text-to-image retrieval using the target caption generated by the planner. It follows the standard VLM retrieval pipeline:
1.  Encode all candidate images in the corpus $\mathcal{D}$ using the VLM's image encoder $\Psi_i$ to get normalized image embeddings.
2.  Encode the refined target caption $\mathcal{T}_t^{re}$ using the VLM's text encoder $\Psi_t$ to get a normalized text embedding.
3.  Retrieve the image with the highest cosine similarity to the target caption embedding:
    $$
    \mathcal{I}_t = \underset{I \in \mathcal{D}}{\operatorname{argmax}} \frac{\Psi_i(I)^\top \Psi_t\left(\mathcal{T}_t^{re}\right)}{\|\Psi_i(I)\| \cdot \|\Psi_t\left(\mathcal{T}_t^{re}\right)\|}
    $$
    Where:
    - $\Psi_i(I)$: Normalized 512/768-dimensional embedding of candidate image $I$ from the VLM image encoder (dimension depends on VLM variant, e.g., 512 for CLIP ViT-B/32, 768 for CLIP ViT-L/14).
    - $\Psi_t(\mathcal{T}_t^{re})$: Normalized embedding of the refined target caption from the VLM text encoder, matching the dimension of the image embedding.
    - The term in the formula is the cosine similarity between the image and text embeddings, ranging from -1 (completely unrelated) to 1 (perfectly aligned).
      The retriever returns the top-$n$ highest similarity images to the corrector agent for evaluation.
---
### 4.2.3. Chain-of-Thought Corrector
The corrector is an LLM-based agent equipped with CoT reasoning that evaluates retrieval results and generates correction feedback for the planner. It operates in three sequential stages:
#### Stage 1: See
To enable LLM evaluation of retrieved images, the corrector uses the same captioning model $\varphi$ as the planner to generate natural language captions $C_t^i$ for each of the top-$n$ retrieved images $\mathcal{I}_t^i$:
$$
C_t^i = \varphi(\mathcal{I}_t^i), \quad i \in \{1, 2, ..., n\}
$$
This converts visual retrieval results to text format compatible with LLM reasoning.
#### Stage 2: Think
The corrector first parses the user's modification intent to generate fine-grained, explicit editing operations $\varepsilon_q$. It takes the reference caption $C_q$ and modification text $\mathcal{T}_q$ as input, and outputs a list of atomic editing operations (add object, delete object, modify object attribute) that specify exactly what changes are required. For example, if $C_q =$ "a red short-sleeved cotton shirt" and $\mathcal{T}_q =$ "make it blue, add long sleeves, change fabric to linen", $\varepsilon_q$ would be:
1.  Change the color of the shirt from red to blue
2.  Change the sleeves of the shirt from short to long
3.  Change the fabric of the shirt from cotton to linen
#### Stage 3: Judge
The corrector then compares the editing operations $\varepsilon_q$ to the captions of the retrieved images $C_t^i$ to check if all modifications are satisfied. It uses a custom CoT prompt to guide step-by-step evaluation:
1.  If all editing operations are satisfied by any of the top-$n$ retrieved images, the corrector outputs "Good retrieval, no more loops needed" and terminates the pipeline.
2.  If any editing operations are not satisfied, the corrector generates specific, actionable correction suggestions $S_q$ (e.g., "The retrieved shirts are still red, need to explicitly specify the shirt color is blue in the target caption") and sends $S_q$ back to the planner's refinement stage.
    ---
### Closed-Loop Workflow Summary
The full AutoCIR pipeline operates as an iterative closed loop:
1.  Planner generates initial target caption from reference image and modification text.
2.  Retriever retrieves top-$n$ images using the target caption.
3.  Corrector evaluates retrieved images against modification intent.
4.  If results are satisfactory: output results, terminate.
5.  If results are unsatisfactory: corrector sends correction suggestions to planner, planner refines target caption, return to step 2.

# 5. Experimental Setup
## 5.1. Datasets
The authors evaluate AutoCIR on three standard widely used ZS-CIR benchmarks, covering both general and vertical domains:
### CIRCO (Composed Image Retrieval on COCO)
- **Source/Scale**: An extension of the MS COCO dataset, designed specifically for ZS-CIR evaluation. It has no training split, with a validation set of 220 queries and a test set of 800 queries.
- **Characteristics**: Each query has an average of 4.53 ground truth target images, to account for inherent ambiguity in modification instructions (e.g., "change the car to red" may have multiple valid matching images).
- **Use Case**: Evaluates model performance on ambiguous, open-domain retrieval scenarios.
### CIRR (Composed Image Retrieval on Real-life images)
- **Source/Scale**: Extended from the NLVR2 dataset, with 21,552 real-world images across diverse domains. Split into training, validation, and test sets, with test set results evaluated via a remote official server.
- **Characteristics**: Contains complex, natural modification descriptions. Reference images often provide limited cues for the target image, with the modification text being the primary retrieval signal.
- **Use Case**: Evaluates model robustness to noisy, real-world modification inputs.
### FashionIQ
- **Source/Scale**: A fashion-domain dataset with 77,684 fashion product images and 30,134 labeled triplets. It covers three product categories: Shirt, Dress, Toptee, split into 6:2:2 training/validation/test sets.
- **Characteristics**: Modification texts are collected via a visual chat interface, simulating real e-commerce user search queries. It is the most widely used benchmark for fashion-domain CIR.
- **Use Case**: Evaluates model performance on vertical-domain e-commerce retrieval scenarios.
  All three datasets are standard benchmarks for ZS-CIR, so experimental results are directly comparable to all prior published methods.
## 5.2. Evaluation Metrics
The authors use two standard retrieval metrics, detailed below:
### Recall@k (R@k)
Used for single-target datasets (CIRR, FashionIQ), where each query has exactly one ground truth target image.
1.  **Conceptual Definition**: Measures the percentage of queries where the ground truth target image appears in the top $k$ retrieved results. Higher Recall@k indicates better performance, as it means the correct image is more likely to be returned in the top $k$ positions.
2.  **Mathematical Formula**:
    $$
    \text{Recall@k} = \frac{1}{Q} \sum_{i=1}^Q \mathbb{1}\left( \text{rank}(\mathcal{I}_t^i) \leq k \right)
    $$
3.  **Symbol Explanation**:
    - $Q$: Total number of evaluation queries.
    - $\text{rank}(\mathcal{I}_t^i)$: The rank of the ground truth target image for query $i$ in the list of retrieved images sorted by similarity descending (rank 1 = highest similarity).
    - $\mathbb{1}(\cdot)$: Indicator function, equals 1 if the condition inside is true, 0 otherwise.
### mAP@k (mean Average Precision at k)
Used for multi-target datasets (CIRCO), where each query has multiple ground truth target images. It is a more fine-grained metric than Recall@k, as it accounts for both the number of correct retrieved images and their ranking.
1.  **Conceptual Definition**: The average of the Average Precision (AP) scores across all queries. AP for a single query measures the average precision of all correct images retrieved in the top $k$ positions, rewarding higher rankings for correct results.
2.  **Mathematical Formula**:
    First, compute AP for a single query $q$:
    $$
    \text{AP@k}_q = \frac{1}{M_q} \sum_{j=1}^k P(j) \cdot \mathbb{1}\left( I_j \in G_q \right)
    $$
    Then compute mAP@k as the average of AP across all queries:
    $$
    \text{mAP@k} = \frac{1}{Q} \sum_{q=1}^Q \text{AP@k}_q
    $$
3.  **Symbol Explanation**:
    - $M_q$: Number of ground truth target images for query $q$.
    - $G_q$: Set of ground truth target images for query $q$.
    - `P(j)`: Precision at rank $j$, equal to the number of correct images in the top $j$ retrieved results divided by $j$.
    - $I_j$: The image at rank $j$ in the retrieved list.
    - $\mathbb{1}(\cdot)$: Indicator function, equals 1 if the image at rank $j$ is a ground truth target, 0 otherwise.
## 5.3. Baselines
The authors compare AutoCIR to 11 competitive baseline methods, grouped into three categories:
1.  **Simple Modality Baselines (training-free)**:
    - `Image-only`: Only use the reference image embedding for retrieval, ignores modification text.
    - `Text-only`: Only use the modification text embedding for retrieval, ignores reference image.
    - $Image+Text$: Average the reference image embedding and modification text embedding for retrieval.
      These baselines measure the lower bound of ZS-CIR performance.
2.  **Training-Dependent Textual Inversion Baselines**:
    - `Pic2word`, `Slerp`, `Lin-CIR`, `PALAVRA`, `SEARLE`, `SEARLE-OTI`: All require additional training on image-text pairs or triplet datasets, use pseudo-text tokens to combine reference image and modification text signals.
3.  **Training-Free LLM-Based Baselines (most relevant to AutoCIR)**:
    - `CIReVL`: Single-pass LLM-based method that edits reference captions to generate target retrieval captions.
    - `LDRE`: Single-pass LLM-based method that generates multiple candidate captions and ensembles retrieval results.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The following are the results from Table 1 of the original paper, covering CIRCO and CIRR test set performance:

<table>
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Model</th>
<th rowspan="2">Training-free</th>
<th colspan="4">CIRCO (mAP@k)</th>
<th colspan="4">CIRR (Recall@k)</th>
<th colspan="3">CIRR (R@k)</th>
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
<th>k=1</th>
<th>k=2</th>
<th>k=3</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="10">ViT-B/32</td>
<td>Image-only</td>
<td>✔️</td>
<td>1.34</td>
<td>1.60</td>
<td>2.12</td>
<td>2.41</td>
<td>6.89</td>
<td>22.99</td>
<td>33.68</td>
<td>59.23</td>
<td>21.04</td>
<td>41.04</td>
<td>60.31</td>
</tr>
<tr>
<td>Text-only</td>
<td>✔️</td>
<td>2.56</td>
<td>2.67</td>
<td>2.98</td>
<td>3.18</td>
<td>21.81</td>
<td>45.22</td>
<td>57.42</td>
<td>81.01</td>
<td>62.24</td>
<td>81.13</td>
<td>90.70</td>
</tr>
<tr>
<td>Image+Text</td>
<td>✔️</td>
<td>2.65</td>
<td>3.25</td>
<td>4.14</td>
<td>4.54</td>
<td>11.71</td>
<td>35.06</td>
<td>48.94</td>
<td>77.49</td>
<td>32.77</td>
<td>56.89</td>
<td>74.96</td>
</tr>
<tr>
<td>PALAVRA</td>
<td>❌</td>
<td>4.61</td>
<td>5.32</td>
<td>6.33</td>
<td>6.80</td>
<td>16.62</td>
<td>43.49</td>
<td>58.51</td>
<td>83.95</td>
<td>41.61</td>
<td>65.30</td>
<td>80.94</td>
</tr>
<tr>
<td>SEARLE</td>
<td>❌</td>
<td>9.35</td>
<td>9.94</td>
<td>11.13</td>
<td>11.84</td>
<td>24.00</td>
<td>53.42</td>
<td>66.82</td>
<td>89.78</td>
<td>54.89</td>
<td>76.60</td>
<td>88.19</td>
</tr>
<tr>
<td>SEARLE-OTI</td>
<td>❌</td>
<td>7.14</td>
<td>7.83</td>
<td>8.99</td>
<td>9.60</td>
<td>24.27</td>
<td>53.25</td>
<td>66.10</td>
<td>88.84</td>
<td>54.10</td>
<td>75.81</td>
<td>87.33</td>
</tr>
<tr>
<td>Slerp</td>
<td>❌</td>
<td>9.34</td>
<td>10.26</td>
<td>11.65</td>
<td>12.33</td>
<td>28.19</td>
<td>55.88</td>
<td>68.77</td>
<td>88.51</td>
<td>61.13</td>
<td>80.63</td>
<td>90.68</td>
</tr>
<tr>
<td>CIReVL</td>
<td>✔️</td>
<td>14.94</td>
<td>15.42</td>
<td>17.00</td>
<td>17.82</td>
<td>23.94</td>
<td>52.51</td>
<td>66.00</td>
<td>86.95</td>
<td>60.17</td>
<td>80.05</td>
<td>90.19</td>
</tr>
<tr>
<td>LDRE</td>
<td>✔️</td>
<td>17.96</td>
<td>18.32</td>
<td>20.21</td>
<td>21.11</td>
<td>25.69</td>
<td>55.13</td>
<td>69.04</td>
<td>89.90</td>
<td>60.53</td>
<td>80.65</td>
<td>90.70</td>
</tr>
<tr>
<td>AutoCIR</td>
<td>✔️</td>
<td><strong>18.82</strong></td>
<td><strong>19.41</strong></td>
<td><strong>21.38</strong></td>
<td><strong>22.32</strong></td>
<td><strong>30.53</strong></td>
<td><strong>59.42</strong></td>
<td><strong>72.19</strong></td>
<td><strong>91.47</strong></td>
<td><strong>65.11</strong></td>
<td><strong>84.02</strong></td>
<td><strong>92.70</strong></td>
</tr>
<tr>
<td rowspan="8">ViT-L/14</td>
<td>Pic2Word</td>
<td>❌</td>
<td>8.72</td>
<td>9.51</td>
<td>10.64</td>
<td>11.29</td>
<td>23.90</td>
<td>51.70</td>
<td>65.30</td>
<td>87.80</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>SEARLE</td>
<td>❌</td>
<td>11.68</td>
<td>12.73</td>
<td>14.33</td>
<td>15.12</td>
<td>24.24</td>
<td>52.48</td>
<td>66.29</td>
<td>88.84</td>
<td>53.76</td>
<td>75.01</td>
<td>88.19</td>
</tr>
<tr>
<td>SEARLE-OTI</td>
<td>❌</td>
<td>10.18</td>
<td>11.03</td>
<td>12.72</td>
<td>13.67</td>
<td>24.87</td>
<td>52.31</td>
<td>66.29</td>
<td>88.58</td>
<td>53.80</td>
<td>74.31</td>
<td>86.94</td>
</tr>
<tr>
<td>Slerp</td>
<td>❌</td>
<td>18.46</td>
<td>19.41</td>
<td>21.43</td>
<td>22.41</td>
<td>30.94</td>
<td>59.40</td>
<td>70.94</td>
<td>89.18</td>
<td>64.70</td>
<td>82.92</td>
<td>92.31</td>
</tr>
<tr>
<td>LinCIR</td>
<td>❌</td>
<td>12.59</td>
<td>13.58</td>
<td>15.00</td>
<td>15.85</td>
<td>25.04</td>
<td>53.25</td>
<td>66.68</td>
<td>-</td>
<td>57.11</td>
<td>77.37</td>
<td>88.89</td>
</tr>
<tr>
<td>CIReVL</td>
<td>✔️</td>
<td>18.57</td>
<td>19.01</td>
<td>20.89</td>
<td>21.80</td>
<td>24.55</td>
<td>52.31</td>
<td>64.92</td>
<td>86.34</td>
<td>59.54</td>
<td>79.88</td>
<td>89.69</td>
</tr>
<tr>
<td>LDRE</td>
<td>✔️</td>
<td>23.35</td>
<td>24.03</td>
<td>26.44</td>
<td>27.50</td>
<td>26.53</td>
<td>55.57</td>
<td>67.54</td>
<td>88.50</td>
<td>60.43</td>
<td>80.31</td>
<td>89.90</td>
</tr>
<tr>
<td>AutoCIR</td>
<td>✔️</td>
<td><strong>24.05</strong></td>
<td><strong>25.14</strong></td>
<td><strong>27.35</strong></td>
<td><strong>28.36</strong></td>
<td><strong>31.81</strong></td>
<td><strong>61.95</strong></td>
<td><strong>73.86</strong></td>
<td><strong>92.07</strong></td>
<td><strong>67.21</strong></td>
<td><strong>84.89</strong></td>
<td><strong>93.13</strong></td>
</tr>
</tbody>
</table>

The following are the results from Table 2 of the original paper, covering FashionIQ validation set performance:

<table>
<thead>
<tr>
<th rowspan="2">Backbone</th>
<th rowspan="2">Model</th>
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
<td rowspan="10">ViT-B</td>
<td>Image-only</td>
<td>6.92</td>
<td>14.23</td>
<td>4.46</td>
<td>12.19</td>
<td>6.32</td>
<td>13.77</td>
<td>5.90</td>
<td>13.37</td>
</tr>
<tr>
<td>Text-only</td>
<td>19.87</td>
<td>34.99</td>
<td>15.42</td>
<td>35.05</td>
<td>20.81</td>
<td>40.49</td>
<td>18.70</td>
<td>36.84</td>
</tr>
<tr>
<td>Image+Text</td>
<td>13.44</td>
<td>26.25</td>
<td>13.83</td>
<td>30.88</td>
<td>17.08</td>
<td>31.67</td>
<td>14.78</td>
<td>29.60</td>
</tr>
<tr>
<td>PALAVRA</td>
<td>21.49</td>
<td>37.05</td>
<td>17.25</td>
<td>35.94</td>
<td>20.55</td>
<td>38.76</td>
<td>19.76</td>
<td>37.25</td>
</tr>
<tr>
<td>SEARLE</td>
<td>24.44</td>
<td>41.61</td>
<td>18.54</td>
<td>39.51</td>
<td>25.70</td>
<td>46.46</td>
<td>22.89</td>
<td>42.53</td>
</tr>
<tr>
<td>SEARLE-OTI</td>
<td>25.37</td>
<td>41.32</td>
<td>24.12</td>
<td>45.79</td>
<td>19.24</td>
<td>42.14</td>
<td>22.44</td>
<td>42.34</td>
</tr>
<tr>
<td>Slerp</td>
<td>23.06</td>
<td>41.95</td>
<td>26.57</td>
<td>47.78</td>
<td>19.24</td>
<td>42.14</td>
<td>22.96</td>
<td>43.96</td>
</tr>
<tr>
<td>CIReVL</td>
<td>27.38</td>
<td>46.27</td>
<td>19.97</td>
<td>41.84</td>
<td>27.07</td>
<td>48.78</td>
<td>24.81</td>
<td>45.63</td>
</tr>
<tr>
<td>LDRE</td>
<td>28.36</td>
<td>47.84</td>
<td>25.29</td>
<td>46.36</td>
<td>31.21</td>
<td>53.85</td>
<td>28.29</td>
<td>49.35</td>
</tr>
<tr>
<td>AutoCIR</td>
<td><strong>32.43</strong></td>
<td><strong>51.67</strong></td>
<td>26.52</td>
<td>46.36</td>
<td><strong>33.96</strong></td>
<td><strong>56.09</strong></td>
<td><strong>30.97</strong></td>
<td><strong>51.37</strong></td>
</tr>
<tr>
<td rowspan="8">ViT-L</td>
<td>Pic2Word</td>
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
<td>SEARLE</td>
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
<td>SEARLE-OTI</td>
<td>30.37</td>
<td>47.49</td>
<td>21.57</td>
<td>44.47</td>
<td>30.90</td>
<td>51.76</td>
<td>27.61</td>
<td>47.90</td>
</tr>
<tr>
<td>Slerp</td>
<td>29.64</td>
<td>46.47</td>
<td>23.35</td>
<td>45.12</td>
<td>31.97</td>
<td>51.20</td>
<td>28.32</td>
<td>47.60</td>
</tr>
<tr>
<td>LinCIR</td>
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
<td>CIReVL</td>
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
<td>LDRE</td>
<td>31.04</td>
<td>51.22</td>
<td>22.93</td>
<td>46.76</td>
<td>31.57</td>
<td>53.64</td>
<td>28.51</td>
<td>50.54</td>
</tr>
<tr>
<td>AutoCIR</td>
<td><strong>34.00</strong></td>
<td><strong>53.43</strong></td>
<td>24.94</td>
<td>45.81</td>
<td><strong>33.10</strong></td>
<td><strong>55.58</strong></td>
<td><strong>30.68</strong></td>
<td><strong>51.60</strong></td>
</tr>
<tr>
<td rowspan="4">ViT-G</td>
<td>Pic2Word</td>
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
<td>CIReVL</td>
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
<td>LDRE</td>
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
<td>AutoCIR</td>
<td><strong>36.36</strong></td>
<td>55.84</td>
<td>26.18</td>
<td>47.69</td>
<td><strong>37.28</strong></td>
<td><strong>60.38</strong></td>
<td><strong>33.27</strong></td>
<td>54.63</td>
</tr>
</tbody>
</table>

### Key Result Observations
1.  **Cross-Dataset Performance**: AutoCIR consistently outperforms all baseline methods across all three datasets, all metrics, and all VLM backbones, while remaining fully training-free.
2.  **CIRCO Results**: On the ambiguous multi-target CIRCO dataset, AutoCIR (ViT-B/32) outperforms the second-best training-free baseline LDRE by 6.29% on mAP@5 and 5.94% on mAP@10, showing the iterative correction mechanism is highly effective for ambiguous retrieval scenarios.
3.  **CIRR Results**: On the noisy real-world CIRR dataset, AutoCIR (ViT-B/32) achieves 8.30% improvement on Recall@1 and 6.33% improvement on Recall@5 over the second-best baseline, demonstrating strong robustness to complex modification inputs.
4.  **FashionIQ Results**: On the e-commerce FashionIQ dataset, AutoCIR achieves an average 8.46% improvement on Recall@10 and 5.16% improvement on Recall@50 over prior methods, showing strong practical value for real-world e-commerce search applications.
5.  **Method Category Comparison**: All LLM-based methods (CIReVL, LDRE, AutoCIR) outperform textual inversion methods, indicating LLM reasoning about user intent aligns better with VLM text encoding than pseudo-text token strategies.
## 6.2. Ablation Studies
The authors conduct ablation studies on the Toptee category of the FashionIQ dataset to validate the effectiveness of each core component of AutoCIR. The results are shown in Table 3 below:

<table>
<thead>
<tr>
<th>Arch</th>
<th>Captioner</th>
<th>LLM</th>
<th>R@10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td>ViT-B/32</td>
<td>BLIP-2</td>
<td>Default*</td>
<td>33.96</td>
<td>56.09</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>BLIP-2</td>
<td>w/o Corrector</td>
<td>32.14</td>
<td>54.27</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>BLIP-2</td>
<td>w/o COT</td>
<td>30.81</td>
<td>52.95</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>BLIP-2</td>
<td>GPT-4o-Mini</td>
<td>33.96</td>
<td>56.09</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>QwenVL</td>
<td>GPT-4o-Mini</td>
<td>33.71</td>
<td>55.07</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>LLAVA-OV</td>
<td>GPT-4o-Mini</td>
<td>33.10</td>
<td>55.07</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>CoCa</td>
<td>GPT-4o-Mini</td>
<td>36.66</td>
<td>57.83</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>BLIP-2</td>
<td>No LLM (-)</td>
<td>21.58</td>
<td>35.67</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>BLIP-2</td>
<td>Qwen-Turbo</td>
<td>32.78</td>
<td>55.47</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>BLIP-2</td>
<td>GPT-3.5-Turbo</td>
<td>34.16</td>
<td>56.39</td>
</tr>
<tr>
<td>ViT-B/32</td>
<td>BLIP-2</td>
<td>GPT-4</td>
<td>34.57</td>
<td>56.82</td>
</tr>
</tbody>
</table>

*: Default model implementation; -: No LLM, uses fixed templates.

### Key Ablation Findings
1.  **Impact of CoT Corrector**: Removing the corrector module reduces R@10 by 1.82%, and removing the CoT reasoning mechanism from the corrector reduces R@10 by 3.15%. This confirms both the corrector module and CoT reasoning are critical for identifying retrieval mismatches and generating high-quality correction feedback.
2.  **Impact of Captioning Quality**: Different public captioning models yield similar performance, with CoCa providing a slight 2.7% R@10 improvement over BLIP-2. This indicates the framework is robust to captioning model choice, while higher-quality captions do provide measurable performance gains.
3.  **Impact of LLM Reasoning**: Using fixed templates (no LLM) yields only 21.58% R@10, while even a relatively small LLM (Qwen-Turbo) improves this to 32.78% R@10. Larger, more capable LLMs (GPT-3.5, GPT-4) provide further incremental gains, confirming dynamic LLM reasoning is critical for generating high-quality target captions.
## 6.3. Parameter Analysis
The authors conduct sensitivity analysis on the maximum number of feedback iterations $N$ (feedback rounds) of the AutoCIR pipeline, using the Toptee category of the FashionIQ dataset. The results are shown in Figure 3 below:

![Figure 3: Sensitivity Analysis of feedback round $N$ on the Toptee validation set, in terms of Recall $@ \\mathbf { k }$ .](images/3.jpg)
*该图像是图表，展示了在Toptee验证集上反馈轮次$N$对Recall@50和Recall@10的敏感性分析。图中分为两个部分，左侧为ViT-B/32，右侧为ViT-L/14，分别显示了不同反馈轮次下的召回率变化。*

### Key Parameter Findings
- Performance peaks at $N=2$ iterations, after which performance decreases for $N>2$. This is because excessive iterations introduce hallucinated correction suggestions from the LLM corrector, adding noise to the target caption.
- At $N=5$, Recall@50 increases but Recall@10 decreases: more iterations increase the diversity of retrieved results but reduce the accuracy of top-ranked results, aligning with the inherent fuzzy nature of ZS-CIR.
- The optimal value of $N$ for all scenarios is 2, balancing performance and computational cost.
## 6.4. Further Analysis
### Qualitative Results
Figure 4 below shows qualitative comparison between AutoCIR and the baseline LDRE:

![该图像是示意图，展示了不同方法生成的服装图像及其对应的描述。其中，'Ours'和'LDRE'分别表示两种方法的结果，图像旁边附有修改说明，展示了对描述的灵活调整与比较。](images/4.jpg)
*该图像是示意图，展示了不同方法生成的服装图像及其对应的描述。其中，'Ours'和'LDRE'分别表示两种方法的结果，图像旁边附有修改说明，展示了对描述的灵活调整与比较。*

AutoCIR correctly handles complex modifications (color change, object insertion, attribute adjustment) that LDRE fails to address, as the iterative correction mechanism ensures all modification requirements are captured in the target caption.
### Alignment Evaluation
The authors use the TIFA metric (a fine-grained text-image alignment metric based on visual question answering) to evaluate the alignment between AutoCIR's generated captions and target images. The results are shown in Figure 5 below:

![Figure 5: Alignment evaluation on the Dress and Shirt datasets using TIFA metric, including two aspects: (1) Comparing the planner-edited description with the base caption on the target image; (2) Comparing the target image with the retrieved image on the planner-edited description.](images/5.jpg)
*该图像是一个图表，展示了在 Dress 和 Shirt 数据集上使用 TIFA 指标的对齐评估情况，包括对比 AutoCIR 方法与参考描述、目标图像和检索图像的得分。*

AutoCIR's modified captions have significantly higher alignment with target images than base reference captions. However, retrieved images have lower alignment with the modified captions than ground truth target images, indicating the VLM retriever (CLIP) is a key performance bottleneck: even with a perfect target caption, the VLM may fail to retrieve the correct matching image.
### Failure Cases
Figure 6 below shows representative failure cases of AutoCIR:

![该图像是示意图，展示了自动多代理协作在零样本组合图像检索中的应用。左侧为参考图像及对应的初始和修正的字幕，右侧显示了检索图像和样本目标。该示意图帮助理解系统如何解析模糊信息并生成更准确的描述。](images/6.jpg)
*该图像是示意图，展示了自动多代理协作在零样本组合图像检索中的应用。左侧为参考图像及对应的初始和修正的字幕，右侧显示了检索图像和样本目标。该示意图帮助理解系统如何解析模糊信息并生成更准确的描述。*

Failures are primarily caused by two factors:
1.  Lack of domain-specific knowledge in the LLM (e.g., the LLM does not understand niche attributes like "Volkswagen saying").
2.  VLM retrieval limitations: even with an accurate target caption, the CLIP retriever fails to find a matching image.
### Self-Correction Interpretability
Figure 7 below shows an example of the AutoCIR self-correction process:

![该图像是示意图，展示了自动多智能体协作进行零-shot 组合图像检索（AutoCIR）的过程。左边是参考图像与初始描述，右边为经过校正后的描述，展示了如何逐步修改和优化描述以提升检索准确性。](images/7.jpg)
*该图像是示意图，展示了自动多智能体协作进行零-shot 组合图像检索（AutoCIR）的过程。左边是参考图像与初始描述，右边为经过校正后的描述，展示了如何逐步修改和优化描述以提升检索准确性。*

The entire pipeline operates in natural language, so the correction process is fully interpretable. Users can directly intervene to edit the target caption if needed, making the framework highly user-friendly.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper proposes AutoCIR, the first training-free closed-loop multi-agent collaboration framework for ZS-CIR. The framework consists of three plug-and-play agents: a planner that generates and refines target captions, a retriever that performs cross-modal zero-shot retrieval, and a CoT-based corrector that evaluates results and generates correction feedback. The iterative closed-loop design eliminates the rigidity and manual trial-and-error requirements of prior ZS-CIR methods, while remaining fully interpretable and user-friendly. Extensive experiments on three standard ZS-CIR benchmarks show AutoCIR consistently outperforms all prior state-of-the-art methods, with particularly strong gains on e-commerce retrieval scenarios.
## 7.2. Limitations & Future Work
### Stated Limitations
1.  **Computational Overhead**: The iterative multi-agent pipeline and use of LLMs lead to higher inference cost than single-pass ZS-CIR methods, which may limit deployment on low-resource edge devices.
2.  **Backbone Dependence**: Performance is limited by the capabilities of the underlying LLM and VLM backbones: LLM hallucination and VLM retrieval mismatches are common failure modes.
### Future Work Directions
1.  Design a lightweight multi-agent framework to reduce computational and cost overhead for real-world large-scale retrieval systems.
2.  Extend the AutoCIR framework to other vision-language tasks such as visual question answering, image editing, and autonomous driving scene understanding.
## 7.3. Personal Insights & Critique
### Strengths
1.  **Intuitive, High-Impact Design**: The multi-agent closed-loop design is highly intuitive and mimics human retrieval workflows, addressing critical pain points of existing ZS-CIR methods. The training-free nature of the framework makes it extremely easy to deploy in real-world systems.
2.  **Full Interpretability**: The natural language pipeline enables full user oversight and intervention, which is a critical requirement for user-facing retrieval systems (e.g., e-commerce search) where transparency and user control are valued.
3.  **Rigorous Evaluation**: The extensive experiments, ablation studies, and sensitivity analysis provide strong evidence for the effectiveness of each component of the framework, and the results are directly comparable to all prior published work.
### Potential Improvements
1.  **Address VLM Bottleneck**: The current framework only refines text captions to improve retrieval. To address the VLM retrieval bottleneck identified in the alignment evaluation, future work could integrate feedback directly into the VLM embedding space (e.g., adjusting the text embedding based on correction feedback) instead of only refining the text caption.
2.  **Dynamic Stopping Condition**: The current framework uses a fixed maximum iteration $N$. Adding a dynamic stopping condition based on the corrector's confidence in retrieval results could reduce unnecessary iterations and lower inference cost.
3.  **User Feedback Integration**: The natural language pipeline makes it easy to integrate explicit user feedback into the loop: users could directly edit the target caption or mark incorrect retrieval results, further improving retrieval accuracy and user experience.
### Broader Impact
The AutoCIR framework represents a promising direction for applying multi-agent systems to vision-language tasks: its modular, training-free, interpretable design can be easily adapted to a wide range of tasks beyond ZS-CIR, such as image editing, visual navigation, and multi-modal content creation.