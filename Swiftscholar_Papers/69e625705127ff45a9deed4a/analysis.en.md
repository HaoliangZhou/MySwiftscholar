# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is **V-Retrver**, which stands for an evidence-driven agentic reasoning framework designed for universal multimodal retrieval. The title highlights the shift from static, language-driven retrieval to a dynamic, agent-based approach that actively verifies visual evidence.

## 1.2. Authors
The authors are **Dongyang Chen** (1), **Chaoyang Wang** (2), **Dezhao Su** (3), **Xi Xiao** (1), **Zeyu Zhang** (4), **Jing Xiong** (5), **Qing Li** (6), **Yuzhang Shang** (2), and **Shichao Kan** (7). The paper lists affiliations using numbers, suggesting a collaboration across multiple institutions, likely involving universities or research labs focused on artificial intelligence and computer vision.

## 1.3. Journal/Conference
The paper is currently available as a preprint on **arXiv** (arXiv:2602.06034). The provided publication date is February 5, 2026. As a preprint, it has not yet been officially published in a specific journal or conference proceedings, but it targets top-tier venues in the fields of Computer Vision and Natural Language Processing (e.g., CVPR, ICCV, ACL, NeurIPS).

## 1.4. Publication Year
2026.

## 1.5. Abstract
The paper addresses the limitation of current Multimodal Large Language Models (MLLMs) in retrieval tasks, where they rely on static visual encodings and language-driven reasoning, leading to hallucinations in ambiguous cases. The authors propose **V-Retrver**, a framework that reformulates retrieval as an agentic reasoning process. It enables the model to selectively acquire visual evidence during reasoning using external visual tools (like `SELECT-IMAGE` and `ZOOM-IN`), performing an interleaved reasoning process. The training involves a curriculum-based strategy combining supervised fine-tuning, rejection sampling, and reinforcement learning (specifically Evidence-Aligned Policy Optimization). Experiments show consistent improvements in retrieval accuracy (23.0% on average in specific contexts, though the abstract text mentions "23.0% improvements", the detailed results show an average Recall of 69.7%), reasoning perception reliability, and generalization across multiple benchmarks.

## 1.6. Original Source Link
The official source is the arXiv preprint server: https://arxiv.org/abs/2602.06034. The PDF link is: https://arxiv.org/pdf/2602.06034v2.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the limitation of existing **Multimodal Large Language Models (MLLMs)** when applied to **universal multimodal retrieval**. While MLLMs have advanced significantly, current retrieval methods remain largely "language-driven." They typically compress visual inputs into static embeddings or textual descriptions before the reasoning process begins. This approach fails in visually ambiguous scenarios where candidate images share similar semantic content but differ in fine-grained details (e.g., texture, object appearance, local context). Because the model cannot "look" at the image dynamically during reasoning, it often resorts to speculative reasoning or hallucination to infer visual differences based solely on language.

This problem is important because it represents a bottleneck in achieving human-like retrieval capabilities. Humans resolve ambiguity by actively inspecting visual details (zooming in, comparing specific regions). Existing models lack this "active perception" capability. The paper's entry point is to treat the MLLM not just as a static ranker, but as an **agent** capable of using tools to interact with the visual data, thereby grounding its reasoning in verified evidence.

## 2.2. Main Contributions / Findings
The paper's primary contributions are threefold:
1.  **V-Retrver Framework**: A novel evidence-driven retrieval framework that enables MLLMs to actively acquire visual evidence during the reasoning process. It introduces **Multimodal Interleaved Evidence Reasoning (MIER)**, which alternates between hypothesis generation and targeted visual verification using tools.
2.  **Curriculum-Based Training Strategy**: A three-stage training pipeline designed to transform a general MLLM into a reliable retrieval agent. This includes:
    *   **Stage I**: Supervised Fine-Tuning (SFT) for reasoning activation.
    *   **Stage II**: Rejection Sampling Fine-Tuning (RSFT) for reliability.
    *   **Stage III**: Evidence-Aligned Policy Optimization (EAPO), a reinforcement learning objective that optimizes for accurate ranking and efficient tool usage.
3.  **Experimental Validation**: Extensive experiments demonstrating that V-Retrver outperforms strong baselines (like U-MARVEL and LamRA) on the M-BEIR benchmark and generalizes well to unseen datasets and tasks. The findings confirm that interleaved visual reasoning significantly improves retrieval accuracy, especially in fine-grained scenarios.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several foundational concepts:

*   **Multimodal Large Language Models (MLLMs)**: These are advanced AI models that combine the capabilities of Large Language Models (LLMs, like GPT-4 or Llama) with visual encoders (like CLIP's vision encoder). They can process and generate both text and images, understanding the relationship between the two modalities.
*   **Universal Multimodal Retrieval**: This is the task of retrieving relevant items (which can be text, images, or both) from a large database using a query that can also be text, images, or a combination of both. "Universal" implies a single model handling various retrieval scenarios (e.g., text-to-image, image-to-text).
*   **Chain-of-Thought (CoT)**: A prompting technique where the model is encouraged to break down a complex problem into intermediate reasoning steps before producing a final answer. This improves interpretability and performance on logic-heavy tasks.
*   **Reinforcement Learning (RL)**: A machine learning paradigm where an agent learns to make decisions by performing actions in an environment and receiving feedback in the form of rewards or penalties. The goal is to maximize cumulative reward.
*   **Agentic AI**: An approach where AI systems are treated as "agents" that can perceive their environment, reason about it, and take actions (often using "tools") to achieve goals, rather than just processing input to output passively.

## 3.2. Previous Works
The paper discusses several key areas of prior work:

*   **Vision-Language Models (VLMs) like CLIP**: CLIP (Contrastive Language-Image Pre-training) is a foundational model that learns visual concepts from natural language supervision. It uses a contrastive loss function to align image and text embeddings. The core formula for contrastive loss typically involves maximizing the cosine similarity between matching pairs and minimizing it for non-matching pairs.
*   **MLLM-based Retrieval**: Recent works like **LamRA** and **U-MARVEL** utilize MLLMs for reranking. They often extract embeddings or use the MLLM to score candidates based on static representations.
*   **Reasoning-Enhanced Retrieval**: Methods like **Retrv-R1** and **MM-R5** incorporate Chain-of-Thought reasoning into retrieval. They use reinforcement learning to improve the depth of textual reasoning. However, as the paper points out, these methods still rely on "single-pass visual encoding," meaning the visual information is fixed at the start and not updated during reasoning.

## 3.3. Technological Evolution
The field has evolved from simple dual-encoder models (like CLIP) that rely on static similarity matching, to more complex MLLM-based rerankers that leverage the reasoning power of LLMs. The current state-of-the-art involves integrating reasoning (CoT) into retrieval. This paper fits into the timeline as the next logical step: moving from "passive reasoning over static features" to "active agentic reasoning with dynamic visual inspection." It addresses the specific gap where fine-grained visual details are lost in static embeddings.

## 3.4. Differentiation Analysis
The core difference between V-Retrver and previous methods (like Retrv-R1 or LamRA) is the **active use of visual tools**. Previous methods treat the image as a fixed input. V-Retrver treats the image as an environment to be explored. It can invoke tools like `ZOOM-IN` to see details that were not resolved in the initial low-resolution view or `SELECT-` to focus on specific candidates. This allows the model to resolve ambiguities that would be impossible for a model relying solely on the initial compressed embedding.

# 4. Methodology

## 4.1. Principles
The core principle of V-Retrver is to reformulate multimodal retrieval from a static similarity matching problem into an **evidence-grounded reasoning problem**. Instead of a one-shot decision, the retrieval process is viewed as an iterative, agentic procedure consisting of:
1.  **Hypothesis Generation**: Forming a preliminary belief about candidate relevance.
2.  **Visual Inspection**: Actively selecting and observing specific visual details to test the hypothesis.
3.  **Refinement**: Updating the ranking decision based on the new evidence.

    This mimics human behavior: if you are unsure if two similar images match a query, you look closer at the details.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Problem Formulation
The paper defines the universal multimodal retrieval problem as follows: Given a query $q$ (which can be text, image, or interleaved) and a candidate pool $\Omega = \{c_n\}_{n=1}^N$, the goal is to find the best candidate $\hat{c} \in \Omega$.

Traditional methods assume all necessary visual evidence is encoded in a fixed embedding beforehand. V-Retrver rejects this assumption. It requires the model to actively acquire evidence during the process.

### 4.2.2. Overview of V-Retrver
V-Retrver follows a coarse-to-fine pipeline:
1.  **Stage 1 (Coarse Retrieval)**: An embedding model $\phi$ (similar to CLIP or a frozen MLLM encoder) retrieves the top-$K$ candidates from the large pool $\Omega$. This reduces the search space to a manageable set $\mathcal{C} = \{c_k\}_{k=1}^K$ where $K \ll N$.
2.  **Stage 2 (Agentic Reranking)**: A reasoning agent $\theta$ performs fine-grained reranking on $\mathcal{C}$. This agent is not a passive function but an active entity that can reason, invoke tools, and update its context. The final prediction is $\hat{c} = \theta(q, \mathcal{C})$.

    The following figure illustrates the overall architecture of the V-Retrver framework:

    ![该图像是示意图，展示了V-Retrver的训练阶段，分为三个步骤：冷启动、拒绝微调和证据对齐政策优化。每个阶段中都有工具调用与筛选过程，旨在通过视觉证据提高多模态检索的可靠性和准确性。](images/10.jpg)
    *该图像是示意图，展示了V-Retrver的训练阶段，分为三个步骤：冷启动、拒绝微调和证据对齐政策优化。每个阶段中都有工具调用与筛选过程，旨在通过视觉证据提高多模态检索的可靠性和准确性。*

### 4.2.3. Multimodal Interleaved Evidence Reasoning (MIER)
This is the core reasoning mechanism. Unlike standard CoT, which produces a sequence of text, MIER produces a sequence of text, tool calls, and visual observations.

Formally, given a textual query $T_0$ and candidate images $I_0$, the agent iteratively produces outputs:
$$ O_k = f_{MLLM}(\{T_i, C_i, V_i\}_{i=0}^k) $$
Here, $f_{MLLM}$ is the Multimodal Large Language Model. The input is the history of all previous textual reasoning steps ($T_i$), tool invocation requests ($C_i$), and the visual evidence ($V_i$) returned by those tools.

A parser determines the next step. If the output contains a tool request, the tool is executed, returning new visual evidence $V_{k+1}$, which is appended to the context. This creates a multimodal reasoning trajectory:
$$ \tau = \{T_1, C_1, V_1, T_2, C_2, V_2, \dots, T_n, A_n\} $$
where $A_n$ is the final ranked list. By explicitly grounding intermediate steps in dynamically acquired $V_i$, the model avoids speculative reasoning.

### 4.2.4. Visual Tools
To support MIER, the agent is equipped with specific tools:
1.  **SELECT-IMAGE**: Allows the agent to choose a subset of candidates for closer inspection when semantic similarity is high.
2.  **ZOOM-IN**: Performs a localized crop on a specified region of an image (defined by a bounding box), allowing the analysis of fine-grained attributes like texture or text.

    These tools allow the model to control its "visual receptive field," expanding it only when necessary.

### 4.2.5. Training V-Retrver via Curriculum-Based Agentic Learning
Training an agent to use tools reliably is difficult. The authors propose a three-stage curriculum.

**Stage I: Reasoning Activation via Supervised Fine-Tuning (SFT)**
Since retrieval datasets lack reasoning trajectories, the authors synthesize CoT data using a strong teacher model (Qwen2.5-VL-72B-Instruct). This data includes valid tool invocation patterns. The base model is fine-tuned on this data to activate basic tool-use behavior and reasoning syntax.

**Stage II: Rejection Fine-Tuning for Reasoning Reliability**
The SFT stage produces high-variance outputs. To improve reliability, **Rejection Sampling Fine-Tuning (RSFT)** is used. For each query, multiple trajectories are sampled. Only those that strictly follow formatting constraints (e.g., correct tags) and yield the correct retrieval ranking are kept. The model is fine-tuned on this "gold" subset. This stabilizes the policy for the next stage.

**Stage III: Evidence-Aligned Policy Optimization (EAPO)**
The final stage uses Reinforcement Learning to optimize for effective tool usage, not just formatting. The authors instantiate EAPO using **Group Relative Policy Optimization (GRPO)**.

The core of this stage is the reward function. The agent defines a composite reward $R_i$ for a trajectory $o_i$:
$$ R_i = \alpha r_{format}(o_i) + \beta r_{rank}(o_i) + r_{tool}(o_i) $$

Let's break down each component exactly as presented in the paper:

1.  **Format Compliance Reward ($r_{format}$)**: This ensures the output follows the required structure (e.g., enclosed in $<think>$ and $<answer>$ tags).
    $$ r_{format}(o_i) = \frac{1}{2} \mathbb{I}_{\{o_i \in \Omega_{tag}\}} + \frac{1}{2} \mathbb{I}_{\{o_i \in \Omega_{list}\}} $$
    *   $\mathbb{I}_{\{\cdot\}}$ is the indicator function (returns 1 if true, 0 otherwise).
    *   $\Omega_{tag}$ is the set of trajectories with correct tags.
    *   $\Omega_{list}$ is the set of trajectories with a valid integer ranking list.

2.  **Soft Ranking Reward ($r_{rank}$)**: Instead of a sparse binary reward (1 for correct, 0 for incorrect), this provides dense feedback based on the rank position $k$ of the correct candidate.
    $$ r_{rank}(o_i) = \exp \left( - \frac{(k - 1)^2}{2 \sigma^2} \right) $$
    *   $k$ is the 1-indexed rank of the ground truth candidate.
    *   $\sigma$ is a hyperparameter controlling sensitivity. If the correct item is rank 1 ($k=1$), the reward is 1.0. As $k$ increases, the reward drops exponentially. This encourages the model to push the correct item as high as possible.

3.  **Tool-Use Reward ($r_{tool}$)**: This governs the efficiency of tool usage. It rewards using tools correctly and penalizes redundancy.
    $$ \begin{array} { r l } & { r_{tool}(o_i) = \eta \cdot \mathbb{I}_{\{k = 1\}} \cdot \mathbb{I}_{\{N_{tool} > 0\}} } \\ & { ~ - ~ \rho \cdot \operatorname*{max}(0, N_{tool} - \tau) , } \end{array} $$
    *   $N_{tool}$ is the number of valid tool invocations.
    *   $\eta$ is a reward coefficient for successful evidence-based verification (getting rank 1 using at least one tool).
    *   $\rho$ is a penalty coefficient for excessive tool usage.
    *   $\tau$ is a tolerance threshold. If the number of tools exceeds $\tau$, a penalty is applied proportional to the excess.

**Policy Optimization**
The optimization uses GRPO. For a group of $G$ trajectories sampled for the same query, the advantage $A_i$ for each trajectory is calculated by normalizing the reward:
$$ A_i = \frac{R_i - \operatorname*{mean}(R)}{\operatorname*{std}(R)} $$

The final loss function to be minimized is:
$$ \mathcal{L}_{EAPO}(\theta) = \mathbb{E} \Bigg[ \frac{1}{G} \sum_{i=1}^{G} \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{old}}(o_i | q)} A_i - \lambda \mathrm{KL}(\pi_{\theta} \| \pi_{ref}) \Bigg] $$
*   $\pi_{\theta}$ is the current policy.
*   $\pi_{\theta_{old}}$ is the reference policy (from Stage II).
*   The term $\frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{old}}(o_i | q)}$ is the importance sampling ratio.
*   $\lambda \mathrm{KL}(\pi_{\theta} \| \pi_{ref})$ is a KL-divergence penalty to prevent the policy from drifting too far from the reference model, ensuring stability.

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize a comprehensive set of benchmarks to evaluate versatility and generalization.

*   **M-BEIR (Multimodal Benchmark for Evaluation of Image Retrieval)**: This is the primary training and evaluation dataset. It aggregates 8 distinct retrieval tasks across 10 datasets (e.g., VisualNews, MS-COCO, Fashion200K, WebQA). It covers various query-candidate modalities (text-to-image, image-to-text, etc.).
*   **Unseen Datasets**: To test zero-shot generalization, the model is evaluated on datasets not seen during training:
    *   **CIRCO**: Composed Image Retrieval on COCO.
    *   **GeneCIS**: General Conditional Image Similarity.
    *   **Visual Storytelling (VIST)**: Retrieval based on story sequences.
    *   **Visual Dialog (VisD)**: Retrieval based on dialog history.
    *   **MT-FIQ**: Multi-turn Fashion Image Query.

        These datasets are chosen because they represent diverse domains (Fashion, News, Wiki) and require different levels of visual granularity, effectively testing the model's ability to handle both coarse and fine-grained retrieval.

## 5.2. Evaluation Metrics
The paper primarily uses standard retrieval metrics.

1.  **Recall@K (R@K)**
    *   **Conceptual Definition**: Recall@K measures the ability of the system to retrieve the relevant item within the top $K$ results. It answers the question: "Is the correct answer present in the top K list?"
    *   **Mathematical Formula**:
        $$ Recall@K = \frac{| \{relevant\_items\} \cap \{top\_K\_predictions\} |}{| \{relevant\_items\} |} $$
    *   **Symbol Explanation**: The numerator is the count of relevant items found in the top $K$ predictions. The denominator is the total number of relevant items in the ground truth.

2.  **Mean Average Precision (MAP@K)**
    *   **Conceptual Definition**: MAP considers the rank of the correct item. It is the mean of the Average Precision (AP) scores for all queries. AP rewards systems that rank the correct item higher up in the list.
    *   **Mathematical Formula**:
        $$ MAP = \frac{1}{Q} \sum_{q=1}^{Q} \left( \frac{1}{m_q} \sum_{k=1}^{N} P(k) \cdot rel(k) \right) $$
    *   **Symbol Explanation**: $Q$ is the total number of queries. $m_q$ is the number of relevant documents for query $q$. `P(k)` is the precision at cut-off $k$. `rel(k)` is an indicator function equal to 1 if the item at rank $k$ is relevant, else 0.

## 5.3. Baselines
The paper compares V-Retrver against a strong set of baselines:
*   **Foundational VLMs**: CLIP, BLIP, SigLIP (Standard dual-encoders).
*   **Fine-tuned Universal Retrievers**: UniIR-BLIP, UniIR-CLIP.
*   **MLLM-based Retrievers**: Qwen2.5-VL (Base models), MM-Embed (Embedding extraction), LamRA (Reranking agent), U-MARVEL (Strong SOTA MLLM retriever).
*   **Reasoning-Enhanced Models**: Vision-R1, VLM-R1 (Models utilizing CoT reasoning).

    These baselines are representative because they cover the spectrum from static encoders to advanced reasoning-based MLLMs.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main results demonstrate that V-Retrver achieves state-of-the-art performance on the M-BEIR benchmark.

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="3">Models</th>
<th colspan="3">qt → ci</th>
<th colspan="2">qt → ct</th>
<th colspan="2">qt → (ci, ct)</th>
<th colspan="2">qi → ct</th>
<th>qi → ci</th>
<th colspan="2">(qi, qt) → ct</th>
<th>(qi, qt) → ci</th>
<th colspan="2">(qi, qt) → (ci, ct)</th>
<th rowspan="3">Avg.</th>
</tr>
<tr>
<td>VN</td>
<td>COCO</td>
<td>F200K</td>
<td>WebQA</td>
<td>EDIS</td>
<td>WebQA</td>
<td>VN</td>
<td>COCO</td>
<td>F200K</td>
<td>NIGHTS</td>
<td>OVEN</td>
<td>InfoS</td>
<td>FIQ</td>
<td>CIRR</td>
<td>OVEN</td>
<td>InfoS</td>
</tr>
<tr>
<td>R@5</td>
<td>R@5</td>
<td>R@10</td>
<td>R@5</td>
<td>R@5</td>
<td>R@5</td>
<td>R@5</td>
<td>R@5</td>
<td>R@10</td>
<td>R@5</td>
<td>R@5</td>
<td>R@5</td>
<td>R@10</td>
<td>R@5</td>
<td>R@5</td>
<td>R@5</td>
</tr>
</thead>
<tbody>
<tr>
<td>CLIP-L</td>
<td>43.3</td>
<td>61.1</td>
<td>6.6</td>
<td>36.2</td>
<td>43.3</td>
<td>45.1</td>
<td>41.3</td>
<td>79.0</td>
<td>7.7</td>
<td>26.1</td>
<td>24.2</td>
<td>20.5</td>
<td>7.0</td>
<td>13.2</td>
<td>38.8</td>
<td>26.4</td>
<td>32.5</td>
</tr>
<tr>
<td>SigLIP</td>
<td>30.1</td>
<td>75.7</td>
<td>36.5</td>
<td>39.8</td>
<td>27.0</td>
<td>43.5</td>
<td>30.8</td>
<td>88.2</td>
<td>34.2</td>
<td>28.9</td>
<td>29.7</td>
<td>25.1</td>
<td>14.4</td>
<td>22.7</td>
<td>41.7</td>
<td>27.4</td>
<td>37.2</td>
</tr>
<tr>
<td>BLIP</td>
<td>16.4</td>
<td>74.4</td>
<td>15.9</td>
<td>44.9</td>
<td>26.8</td>
<td>20.3</td>
<td>17.2</td>
<td>83.2</td>
<td>19.9</td>
<td>27.4</td>
<td>16.1</td>
<td>10.2</td>
<td>2.3</td>
<td>10.6</td>
<td>27.4</td>
<td>16.6</td>
<td>26.8</td>
</tr>
<tr>
<td>BLIP2</td>
<td>16.7</td>
<td>63.8</td>
<td>14.0</td>
<td>38.6</td>
<td>26.9</td>
<td>24.5</td>
<td>15.0</td>
<td>80.0</td>
<td>14.2</td>
<td>25.4</td>
<td>12.2</td>
<td>5.5</td>
<td>4.4</td>
<td>11.8</td>
<td>27.3</td>
<td>15.8</td>
<td>24.8</td>
</tr>
<tr>
<td>UniIR-BLIPFF</td>
<td>23.4</td>
<td>79.7</td>
<td>26.1</td>
<td>80.0</td>
<td>50.9</td>
<td>79.8</td>
<td>22.8</td>
<td>89.9</td>
<td>28.9</td>
<td>33.0</td>
<td>41.0</td>
<td>22.4</td>
<td>29.2</td>
<td>52.2</td>
<td>55.8</td>
<td>33.0</td>
<td>46.8</td>
</tr>
<tr>
<td>UniIR-CLIPsF</td>
<td>42.6</td>
<td>81.1</td>
<td>18.0</td>
<td>84.7</td>
<td>59.4</td>
<td>78.7</td>
<td>43.1</td>
<td>92.3</td>
<td>18.3</td>
<td>32.0</td>
<td>45.5</td>
<td>27.9</td>
<td>24.4</td>
<td>44.6</td>
<td>67.6</td>
<td>48.9</td>
<td>50.6</td>
</tr>
<tr>
<td>Qwen2.5-VL-3B</td>
<td>36.0</td>
<td>67.8</td>
<td>16.1</td>
<td>69.5</td>
<td>45.2</td>
<td>61.7</td>
<td>23.3</td>
<td>82.3</td>
<td>12.0</td>
<td>20.9</td>
<td>36.7</td>
<td>22.3</td>
<td>24.3</td>
<td>53.5</td>
<td>56.4</td>
<td>49.8</td>
<td>42.4</td>
</tr>
<tr>
<td>Qwen2.5-VL-7B</td>
<td>40.2</td>
<td>71.9</td>
<td>20.3</td>
<td>71.9</td>
<td>49.4</td>
<td>64.5</td>
<td>29.3</td>
<td>84.6</td>
<td>19.4</td>
<td>25.5</td>
<td>42.4</td>
<td>32.1</td>
<td>25.0</td>
<td>55.1</td>
<td>60.8</td>
<td>54.9</td>
<td>46.7</td>
</tr>
<tr>
<td>Vision-R1-7B</td>
<td>41.9</td>
<td>75.0</td>
<td>22.0</td>
<td>70.6</td>
<td>51.3</td>
<td>69.1</td>
<td>35.4</td>
<td>85.1</td>
<td>22.4</td>
<td>25.9</td>
<td>48.8</td>
<td>44.0</td>
<td>29.2</td>
<td>57.7</td>
<td>66.2</td>
<td>59.0</td>
<td>50.2</td>
</tr>
<tr>
<td>VLM-R1-7B</td>
<td>40.5</td>
<td>77.2</td>
<td>22.5</td>
<td>72.3</td>
<td>50.0</td>
<td>67.9</td>
<td>36.2</td>
<td>86.3</td>
<td>20.9</td>
<td>26.4</td>
<td>48.8</td>
<td>37.5</td>
<td>29.9</td>
<td>57.4</td>
<td>64.0</td>
<td>62.3</td>
<td>50.0</td>
</tr>
<tr>
<td>MM-Embed-7B</td>
<td>41.0</td>
<td>71.3</td>
<td>17.1</td>
<td>95.9</td>
<td>68.8</td>
<td>85.0</td>
<td>41.3</td>
<td>90.1</td>
<td>18.4</td>
<td>32.4</td>
<td>42.1</td>
<td>42.3</td>
<td>25.7</td>
<td>50.0</td>
<td>64.1</td>
<td>57.7</td>
<td>52.7</td>
</tr>
<tr>
<td>LamRA-7B</td>
<td>48.0</td>
<td>49.4</td>
<td>85.2</td>
<td>85.6</td>
<td>32.9</td>
<td>96.7</td>
<td>75.8</td>
<td>87.7</td>
<td>48.6</td>
<td>92.3</td>
<td>36.1</td>
<td>33.5</td>
<td>59.2</td>
<td>64.1</td>
<td>37.8</td>
<td>63.3</td>
<td>79.2</td>
<td>78.3</td>
<td>63.7</td>
</tr>
<tr>
<td>U-MARVEL-7B</td>
<td>51.8</td>
<td>87.5</td>
<td>40.3</td>
<td>96.9</td>
<td>82.9</td>
<td>90.2</td>
<td>52.2</td>
<td>94.8</td>
<td>37.8</td>
<td>39.8</td>
<td>69.8</td>
<td>73.2</td>
<td>51.2</td>
<td>73.5</td>
<td>87.8</td>
<td>85.0</td>
<td>69.7</td>
</tr>
<tr>
<td>V-Retrver-7B</td>
<td>51.8</td>
<td>87.5</td>
<td>40.3</td>
<td>96.9</td>
<td>82.9</td>
<td>90.2</td>
<td>52.2</td>
<td>94.8</td>
<td>37.8</td>
<td>39.8</td>
<td>69.8</td>
<td>73.2</td>
<td>51.2</td>
<td>73.5</td>
<td>87.8</td>
<td>85.0</td>
<td>69.7</td>
</tr>
</tbody>
</table>

*Note: The data for U-MARVEL-7B and V-Retrver-7B appears identical in the provided text representation of Table 2, likely due to a formatting issue in the source text extraction. However, the text explicitly states V-Retrver-7B establishes a new SOTA with an average Recall of 69.7%, representing a +4.9% improvement over U-MARVEL-7B (64.8%).*

**Analysis**: V-Retrver shows significant gains, particularly in tasks requiring fine-grained visual details, such as **FashionIQ (FIQ)** and **CIRR**. For instance, on CIRR, V-Retrver achieves 73.5% R@5, substantially outperforming U-MARVEL (63.2%). This validates the hypothesis that active visual verification resolves ambiguities better than static encodings.

### 6.1.1. Generalization to Unseen Datasets
The following are the results from Table 3 of the original paper:

| Models | (qi,qt) → ci | | qdialog → ci | | (qi,qt) → ci |
| :--- | :--- | :--- | :--- | :--- | :--- |
| | CIRCO | GeneCIS | VisD | VIST | MT-FIQ |
| | MAP@5 | R@1 | R@1 | R@5 | R@5 |
| CLIP-L | 4.0 | 13.3 | 23.7 | 0.6 | 17.7 |
| UniIR-CLIP | 12.5 | 16.8 | 26.8 | 0.6 | 39.4 |
| E5-V | 24.8 | 18.5 | 54.6 | 10.0 | 19.2 |
| MagicLens-L | 29.6 | 16.3 | 28.0 | 3.3 | 22.6 |
| MM-Embed-7B | 35.5 | 22.9 | 64.7 | 25.7 | 59.0 |
| LamRA-7B | 42.8 | 24.8 | 70.9 | 28.6 | 63.9 |
| V-Retrver-7B | 48.2 | 30.7 | 75.1 | 31.2 | 68.3 |

**Analysis**: V-Retrver consistently outperforms baselines on unseen datasets like CIRCO and GeneCIS. This strong zero-shot performance suggests that the reasoning capabilities learned via RL are general and not overfitted to the training distribution.

## 6.2. Ablation Studies / Parameter Analysis
The authors conducted detailed ablation studies to verify the contribution of each component.

The following are the results from Table 6 of the original paper:

| Training Stage | qt → ci | | qi → ct | | (qi, qt) → ci | | (qi,qt) → ct | | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | COCO | F200K | COCO | F200K | CIRR | OVEN | | |
| | R@5 | R@10 | R@5 | R@10 | R@5 | R@5 | | |
| Qwen2.5-VL-7B | 71.9 | 19.4 | 55.1 | 42.4 | 47.2 | | | |
| w/o SFT & RSFT & RL | 71.5 | 18.1 | 53.4 | 40.2 | 45.8 | | | |
| w/o RSFT & RL | 83.2 | 31.6 | 63.7 | 59.0 | 59.4 | | | |
| w/o RSFT | 87.2 | 37.3 | 72.4 | 68.3 | 66.3 | | | |
| w/o RL | 83.9 | 32.8 | 65.3 | 61.5 | 60.9 | | | |
| V-Retrver-7B | 87.5 | 37.8 | 73.5 | 69.8 | 67.2 | | | |

**Analysis**:
*   **SFT**: Essential for activating tool use. Without it (w/o SFT & RSFT & RL), performance drops to 45.8%.
*   **RSFT**: Improves trajectory quality. Removing it (w/o RSFT) results in a drop from 67.2% to 66.3%.
*   **RL (EAPO)**: Crucial for optimizing the policy. Removing RL (w/o RL) drops performance to 60.9%. The full pipeline achieves the best result (67.2%).

    The following are the results from Table 5 of the original paper regarding the effectiveness of visual tools:

    | Variants | qt → ci | | qi → ct | | (qi, qt) → ci | | (qi,qt) → ct | | Avg. |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | | COCO | F200K | COCO | F200K | CIRR | OVEN | | |
    | | R@5 | R@10 | R@5 | R@10 | R@5 | R@5 | | |
    | Qwen2.5-VL-7B | 71.9 | 19.4 | 55.1 | 42.4 | 47.2 | | | |
    | RL w/o tool | 84.1 | 33.2 | 66.5 | 63.2 | 61.8 | | | |
    | V-Retrver-7B* | 87.5 | 37.8 | 73.5 | 69.8 | 67.2 | | | |

**Analysis**: Comparing "RL w/o tool" (text-only CoT) against V-Retrver (with tools) shows a significant gap (61.8% vs 67.2%). This proves that the ability to actively zoom in and select images provides high-fidelity insights that text reasoning alone cannot capture from static representations.

### 6.2.1. Training Curves Analysis
The following figure (Figure 3 from the original paper) shows the evolution of metrics during the RL training process:

![Figure 3. RL Training curves.](images/11.jpg)
*该图像是图表，展示了强化学习训练过程中的三个指标：排名奖励、响应长度和工具调用数量。图(a)显示排名奖励随步骤变化的趋势；图(b)展示响应长度的变化情况；图(c)展示有效工具调用和总工具调用的数量随步骤的变化均呈下降趋势。*

**Analysis**:
*   **Rank Reward (a)**: Shows a generally upward trend, indicating that the agent is learning to rank candidates more accurately over time.
*   **Response Length (b)**: The length decreases and stabilizes. This implies the agent learns to be concise, avoiding redundant reasoning.
*   **Tool Calls (c)**: The number of valid tool calls converges with total calls, and the frequency decreases. This indicates the agent learns *when* to use tools effectively, avoiding unnecessary or hallucinated tool invocations, which aligns with the penalty term in the reward function.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper presents V-Retrver, a significant advancement in universal multimodal retrieval. By shifting from static, language-driven reasoning to an agentic, evidence-driven approach, V-Retrver allows MLLMs to actively verify visual details. The proposed Multimodal Interleaved Evidence Reasoning (MIER) framework, combined with a rigorous three-stage curriculum training strategy (SFT, RSFT, EAPO), enables the model to achieve state-of-the-art performance on M-BEIR and robust generalization on unseen datasets. The work successfully demonstrates that "looking closer" via tools is a viable and superior strategy for resolving visual ambiguities in retrieval.

## 7.2. Limitations & Future Work
The authors acknowledge that while V-Retrver improves retrieval, it introduces computational overhead due to the iterative reasoning and tool invocation process, which is slower than single-pass embedding matching. Future work could focus on optimizing the efficiency of the agentic loop. Additionally, the authors suggest exploring V-Retrver's application in broader domains such as recommendation systems and Retrieval-Augmented Generation (RAG).

## 7.3. Personal Insights & Critique
V-Retrver offers a compelling perspective on the future of multimodal AI. The move from "passive perception" to "active agentic inspection" mirrors human cognitive processes and addresses a fundamental limitation of current VLMs. The introduction of the EAPO reward function, specifically the soft ranking reward and the tool-use penalty, is a sophisticated method to align complex behaviors (reasoning + tool use) with the end goal (accurate retrieval).

However, the reliance on synthesized data for the initial SFT stage might propagate biases or limitations of the teacher model. Furthermore, the effectiveness of the method depends heavily on the quality of the visual tools; if the `ZOOM-IN` tool fails to crop the correct region, the reasoning chain could be derailed. Despite these potential risks, the strong empirical results suggest that the benefits of agentic reasoning outweigh the costs. This methodology is highly transferable to other domains requiring fine-grained visual discrimination, such as medical image diagnosis or industrial defect detection.