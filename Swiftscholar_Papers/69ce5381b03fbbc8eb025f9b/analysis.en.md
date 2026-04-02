# 1. Bibliographic Information

## 1.1. Title
The title of the paper is **"Beyond Symbolic Solving: Multi Chain-of-Thought Voting for Geometric Reasoning in Large Language Models"**. The central topic revolves around improving Geometric Problem Solving (GPS) by moving beyond traditional symbolic or single-path neural approaches. It proposes a method that leverages the reasoning capabilities of Large Language Models (LLMs) through multiple parallel reasoning paths (Chain-of-Thoughts) and a sophisticated voting mechanism.

## 1.2. Authors
The authors of the paper are:
*   **Md. Abu Bakor Siddique** (Equal contribution)
*   **Shahrin Hossain** (Equal contribution)
*   **Sadman Ahmed Siam** (Equal contribution)
*   **Syed Rifat Raiyan** (Corresponding author)
*   **Hasan Mahmud Md Kamrul Hasan**

    They are affiliated with the **Systems and Software Lab (SSL), Department of Computer Science and Engineering, Islamic University of Technology, Dhaka, Bangladesh**. The research background appears to be focused on Artificial Intelligence (AI), specifically in the intersection of computer vision (diagram understanding) and natural language processing (mathematical reasoning).

## 1.3. Journal/Conference
The paper is listed as published on **arXiv** (Cornell University's open-access repository for preprints) with the ID `arXiv:2604.00890`. The publication date provided is April **1, 2026**. As a preprint, it has not yet undergone the full peer-review process associated with formal journals or conferences, though it serves as a preliminary report of their findings. The "UTC" timestamp suggests it is a very recent submission.

## 1.4. Publication Year
The paper is dated **2026**.

## 1.5. Abstract
The paper addresses **Geometric Problem Solving (GPS)**, which is a key challenge in enhancing mathematical reasoning in AI. GPS requires three distinct skills: diagrammatic understanding, symbolic manipulation, and logical inference. The authors argue that existing literature focuses heavily on the first two (synchronizing diagrams with text) using neural, symbolic, or neuro-symbolic approaches, but leaves logical inference underdeveloped, often limited to a single chain-of-thought (CoT).

To address this, the paper proposes **MARS-GPS** (Multi-path Aggregated Reasoning System for Geometry Problem Solving). This method generates multiple parallel reasoning rollouts (attempts to solve the problem) augmented with Python code execution for numerical verification. It ranks these solutions using token-level entropy as a confidence signal and aggregates the answers through a multi-stage voting and self-verification pipeline.

Empirically, MARS-GPS with 8 parallel rollouts achieves **88.8%** accuracy on the **Geometry3K** dataset, representing a nearly **+11%** improvement over the previous state-of-the-art. The accuracy scales consistently as the number of rollouts increases.

## 1.6. Original Source Link
The official source link is: **https://arxiv.org/abs/2604.00890**
The PDF link is: **https://arxiv.org/pdf/2604.00890v1**
The publication status is a **preprint** (arXiv).

# 2. Executive Summary

## 2.1. Background & Motivation
**The Core Problem:**
The core problem is **Geometric Problem Solving (GPS)**. This is a complex task that requires an AI system to interpret a diagram and a textual description, identify the given information, apply relevant geometric theorems, and derive the correct answer, often from multiple-choice options.

**Importance and Challenges:**
GPS is considered a "pinnacle of human reasoning" and is crucial for advancing mathematical reasoning in Large Language Models (LLMs). The specific challenge lies in the multifaceted nature of the task:
1.  **Diagrammatic Understanding:** The model must "see" and understand the geometric figure.
2.  **Symbolic Manipulation:** It must handle mathematical expressions and equations.
3.  **Logical Inference:** It must select the correct theorems and apply them in a logical sequence.

    The authors identify a gap in existing research: most methods (like Pi-GPS, MINT-CoT, PGPSNet-v2) excel at the first two steps but treat logical inference as a single, deterministic path (one Chain-of-Thought). If that single path makes a mistake, the answer is wrong. The paper's entry point is the hypothesis that logical inference can be significantly improved not by training a better model, but by generating multiple diverse reasoning paths and intelligently aggregating them.

## 2.2. Main Contributions / Findings
**Primary Contributions:**
1.  **MARS-GPS Framework:** A novel two-stage system that separates parsing (diagram/text to formal logic) from inference-time reasoning.
2.  **Parallel Rollout Sampling:** Demonstrating that sampling multiple independent reasoning paths from a frozen LLM outperforms complex symbolic solvers.
3.  **Zero-Cost Confidence Signal:** Introducing the use of **token-level entropy** (calculated from the model's log probabilities) as a proxy for confidence in a specific answer, without needing extra training.
4.  **Aggregation Algorithm:** A six-step pipeline that combines majority voting, entropy ranking, and LLM self-verification to select the final answer.

**Key Findings:**
*   **Performance:** MARS-GPS achieves **88.8%** on Geometry3K and **77.48%** on PGPS9K, significantly outperforming previous state-of-the-art models like Pi-GPS.
*   **Scalability:** Accuracy scales log-linearly with the number of parallel rollouts (from 1 to 16), suggesting that "inference-time scaling" (spending more compute during testing) is highly effective for geometry.
*   **Component Importance:** The self-verification step is the most critical component, followed by code augmentation (Python execution).

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several foundational concepts:

*   **Geometric Problem Solving (GPS):** The task of solving geometry problems typically involving a diagram and text. This requires understanding spatial relationships (e.g., "Point A lies on Line BC") and applying theorems (e.g., Pythagorean theorem).
*   **Chain-of-Thought (CoT):** A prompting technique where an LLM is encouraged to generate intermediate reasoning steps before arriving at a final answer. This mimics human step-by-step thinking.
*   **Self-Consistency:** An advanced prompting strategy where the model samples multiple diverse CoT paths for the same problem and takes the "majority vote" as the final answer. This is based on the idea that if the model reasons correctly multiple times, it will likely arrive at the same answer.
*   **Token Entropy:** In information theory, entropy measures uncertainty or randomness. In the context of LLMs, each token prediction comes with a probability distribution. High entropy means the model is unsure (many possible next tokens), while low entropy means the model is confident. The formula for Shannon Entropy $H$ is $H(X) = -\sum p(x) \log p(x)$.
*   **Neuro-Symbolic AI:** An approach that combines neural networks (which are good at perception and pattern recognition from data) with symbolic AI (which is good at logic, rules, and exact manipulation).
*   **First-Order Logic Predicates:** A formal way to represent facts using logic. For example, `Perpendicular(Line(A, B), Line(C, D))` is a predicate stating that two lines are perpendicular.
*   **Inference-Time Scaling:** The concept of improving model performance by increasing the amount of computation (e.g., generating more text, sampling more paths) during the inference phase (when the model is answering questions), rather than during the training phase.

## 3.2. Previous Works
The paper categorizes previous research into three main streams:

1.  **Symbolic Solvers (e.g., Inter-GPS):** These systems attempt to solve geometry problems by converting them into formal logic and using symbolic theorem search.
    *   *Gap:* They are often brittle and have limited scalability; they fail if the problem doesn't fit their predefined rules perfectly.
2.  **Neuro-Symbolic Solvers (e.g., PGPSNet, Pi-GPS):** These mix neural networks (to parse diagrams) with symbolic solvers (to find the answer).
    *   *Gap:* While more scalable, they still rely on fixed sets of theorems and struggle with complex, multi-step reasoning chains. Pi-GPS, for instance, uses diagrams to fix text but still relies on a provided theorem set.
3.  **Multimodal Large Language Models (MLLMs) (e.g., G-LLaVA, GPT-4o):** These treat geometry as a general visual-language task.
    *   *Gap:* They suffer from unreliable reasoning. They might "hallucinate" geometric details or fail at precise arithmetic calculations.

        The paper also references **Balachandran et al. (2025)** regarding "inference-time scaling," noting that while this works for text-heavy math, it has been less successful in geometry due to the multimodal requirement (understanding the image).

## 3.3. Technological Evolution
The field has evolved from purely **Symbolic** approaches (rigid, rule-based) to **Neural** approaches (flexible but "black box"), and then to **Neuro-Symbolic** hybrids (trying to get the best of both worlds). Most recently, **MLLMs** have been applied as general-purpose solvers.

This paper represents the next step: **Inference-Time Scaling with MLLMs**. Instead of trying to train a "smarter" model, it takes a powerful, frozen model and makes it "think harder" (more rollouts, self-checking) at test time. It fits into the timeline as a shift from training-time optimization to inference-time optimization.

## 3.4. Differentiation Analysis
The core difference between MARS-GPS and prior work is the **treatment of logical inference**.
*   **Prior Work:** Typically commits to a **single** reasoning path. If the model chooses the wrong theorem at step 1, the final answer is wrong.
*   **MARS-GPS:** Generates **multiple** parallel paths ($k$ rollouts). It does not trust a single path. Instead, it uses a **confidence-aware aggregation strategy**. It calculates how "sure" the model was about each path (using entropy) and asks the model to **verify** its own work before committing. This makes the system more robust to individual reasoning errors.

# 4. Methodology

## 4.1. Principles
The core principle of MARS-GPS is **"Decomposition and Ensemble Reasoning."**
1.  **Decomposition:** The problem is split into two distinct stages. First, a "Parsing Stage" converts the raw image and text into a structured, formal language (First-Order Logic predicates). This handles the "diagrammatic understanding" requirement. Second, an "Inference Stage" takes this formal text and performs reasoning.
2.  **Ensemble Reasoning:** Instead of relying on a single deterministic solver, the system uses a **frozen** Large Language Model (GPT-OSS 120B) to generate $k$ independent solution attempts (rollouts) in parallel. The principle is that while a single attempt might fail, the "correct" answer will likely emerge repeatedly or with higher confidence across multiple attempts.
3.  **Confidence as a Signal:** The method leverages the model's internal uncertainty (entropy) as a zero-cost signal to weigh the votes of different rollouts.

## 4.2. Core Methodology In-depth (Layer by Layer)

The methodology consists of two main stages: Problem Parsing and Inference-Time Ensemble Reasoning.

### Stage 1: Problem Parsing and Unified Representation
The first step is to convert the raw inputs—text $T$ and image $I$—into a unified formal representation $\mathcal{F}^*$. This ensures the LLM doesn't need to "see" the image directly during the reasoning phase; it only processes the structured text.

**Step 1.1: Text Parsing**
The text $T$ is processed using a rule-based parser (specifically, regular expressions). This extracts formal literals $\mathcal{F}_T$.
*   *Example:* "Area of Triangle ABC is 10" becomes `Equals(AreaOf(Triangle(A, B, C)), 10)`.
*   *Rationale:* Rule-based parsing is preferred here for precision, as geometry datasets are small and specific.

**Step 1.2: Diagram Parsing**
The diagram image $I$ is processed by a neural network called **PGDPNet**. This network extracts geometric primitives and relationships as formal literals $\mathcal{F}_D$.
*   *Example:* It detects points and lines and outputs `PointLiesOnLine(D, Line(B, C))`.

**Step 1.3: Unification**
The final representation $\mathcal{F}^*$ is simply the combination of text and diagram literals: $\mathcal{F}^* = \mathcal{F}_T \cup \mathcal{F}_D$. This structured text is the only input passed to the reasoning model.

The following figure illustrates the full pipeline, including the parsing stage on the left and the inference stage on the right.

![Figure 1: Overview of the Multi-path Aggregated Reasoning System for Geometry Problem Solving (MARs-GPS) pipeline. Left: the problem parsing stage takes the diagram and problem text as input and produces a unified formal context ${ \\mathcal { F } } ^ { * }$ via PGDPNet and a rulebased semantic parser. Right: the inference-time ensemble reasoning stage samples $k$ parallel rollouts from $f _ { \\theta }$ , each augmented with a Python sandbox $\\mathcal { E }$ for numerical computation. The rollout outputs feed into the answer aggregation pipeline, which applies majority voting, entropy-ranked self-verification, and a weighted fallback to produce the final answer $a ^ { * }$ .](images/1.jpg)
*该图像是一个示意图，展示了多路径聚合推理系统（MARs-GPS）在几何问题解决中的工作流程。左侧为问题解析阶段，其中通过 PGDPNet 和基于规则的语义解析器，将图形和问题文本转换为统一的形式上下文 `{ ilde{ ext{F}}}`。右侧为推理阶段，描述了并行推理的样本和答案聚合流程。*

### Stage 2: Inference-Time Ensemble Reasoning
This is the core contribution. The system takes $\mathcal{F}^*$ and generates a final answer $a^*$.

**Step 2.1: Parallel Rollout Sampling**
The system constructs a prompt $\mathcal{P}$ containing the problem text, choices, and the formal context $\mathcal{F}^*$. It instructs the model $f_\theta$ (the LLM) to output its final answer in a specific format, e.g., $\boxed{N}$.

Instead of running the model once, it samples $k$ independent reasoning traces (rollouts) in parallel. Mathematically, this is represented as:

$$
\{ r _ { 1 } , r _ { 2 } , \dots , r _ { k } \} \sim f _ { \theta } ( \mathcal { P } \mid \mathcal { F } ^ { * } )
$$

*   **Symbol Explanation:**
    *   $r_i$: A single reasoning trace (chain-of-thought) generated by the model.
    *   $k$: The number of parallel rollouts (e.g., 8 or 16).
    *   $f_\theta$: The frozen LLM model.
    *   $\mathcal{P}$: The prompt constructed from the problem.
    *   $\mathcal{F}^*$: The unified formal context derived from parsing.
    *   $\sim$: Indicates sampling (generating text probabilistically).

        Each rollout $r_i$ produces a candidate answer $a_i$ (extracted by finding the $\boxed{N}$ pattern) and a sequence of tokens.

**Step 2.2: Confidence Estimation via Token Entropy**
To determine which of the $k$ answers is most reliable, the system calculates a confidence score for each rollout. It uses the **per-token log probabilities** returned by the LLM.

For each token position $t$ in a rollout $r_i$, the system computes the **Shannon Entropy** over the top vocabulary entries. The formula used is:

$$
H _ { t } = - \sum _ { j } e ^ { \ell _ { t , j } } \cdot \log _ { 2 } \left( e ^ { \ell _ { t , j } } \right)
$$

*   **Symbol Explanation:**
    *   $H_t$: The entropy at token position $t$.
    *   $\ell_{t, j}$: The log probability of the $j$-th token candidate at position $t$.
    *   $e^{\ell_{t, j}}$: Converts the log probability back to a standard probability $p$.
    *   $\log_2(\dots)$: The logarithm base 2.
    *   $\sum$: Summation over the relevant vocabulary entries.

        High entropy means the model was uncertain about which token to pick next. Low entropy means the model was confident. The system aggregates this into a mean entropy for the entire rollout:

$$
\bar { H } _ { i } = \frac { 1 } { T _ { i } } \sum _ { t = 1 } ^ { T _ { i } } H _ { t }
$$

*   **Symbol Explanation:**
    *   $\bar{H}_i$: The mean entropy for rollout $i$.
    *   $T_i$: The total number of tokens in rollout $i$.

        $\bar{H}_i$ serves as an **inverse confidence score**. Lower $\bar{H}_i$ means higher confidence.

**Step 2.3: Code-Augmented Reasoning**
Geometry often requires precise calculation (e.g., solving $3x + 6 = 4x - 2$). LLMs can be bad at arithmetic. To fix this, each rollout is paired with a **Python Sandbox** $\mathcal{E}$.

$$
r _ { i } = f _ { \theta } ( \mathcal { P } , \mathcal { E } ), \quad \mathcal { E } : \mathrm { c o d e } \mapsto \mathrm { o u t p u t }
$$

*   **Symbol Explanation:**
    *   $\mathcal{E}$: The Python execution environment.
    *   $\mathrm{code} \mapsto \mathrm{output}$: The sandbox takes code as input and returns the execution output.

        If the LLM writes a Python code block during its reasoning, the system executes it in $\mathcal{E}$ and injects the result back into the prompt. This allows the model to use exact calculations rather than guessing.

**Step 2.4: Verification and Self-Consistency (Aggregation)**
Finally, the system must choose the best answer from the $k$ candidates $\{a_i\}$ using their confidence scores $\{\bar{H}_i\}$. It uses a six-step algorithm:

1.  **Early Consensus:** Check if any answer has a strict majority ($\ge k/2 + 1$ votes). If yes, return it immediately.
    $$
    \mathrm { i f ~ } \sum _ { i = 1 } ^ { k } { \bf 1 } [ a _ { i } = a ] \geq k / 2 + 1 \quad \Rightarrow \quad a ^ { \ast } = a
    $$
2.  **Hard Accept:** If no strict majority, check for a "weak" majority ($\ge \lceil k/2 \rceil$ votes). If found, return it.
3.  **Candidate Selection:** If no majority, collect all answers that appear at least $k/4$ times. These are the candidates $\mathcal{A}_{\mathrm{cand}}$.
    $$
    \mathcal { A } _ { \mathrm { c a n d } } = \left\{ a \bigg | \sum _ { i = 1 } ^ { k } { \bf 1 } [ a _ { i } = a ] \geq \lceil k / 4 \rceil \right\}
    $$
4.  **Entropy-Ranked Verification:** Sort the candidates by their mean entropy $\bar{H}(a)$ (lowest first, i.e., most confident first).
5.  **LLM Self-Verification:** For each candidate (in the confident order), ask the LLM: "Is this answer CORRECT or WRONG?". If the LLM says "CORRECT", accept that answer and stop.
6.  **Weighted Fallback:** If all candidates are rejected by self-verification, calculate a weighted score combining votes and confidence, and pick the max.
    $$
    a ^ { * } = \arg \operatorname* { m a x } _ { a \in { \mathcal { A } } _ { \mathrm { c a n d } } } \lambda \cdot \mathrm { v o t e s } ( a ) - ( 1 - \lambda ) \cdot { \bar { H } } ( a )
    $$
    *   **Symbol Explanation:**
        *   $\lambda$: A weighting parameter (e.g., 0.5).
        *   $\mathrm{votes}(a)$: The number of rollouts that voted for answer $a$.
        *   $\bar{H}(a)$: The mean entropy of answer $a$ (subtracted because lower is better).

# 5. Experimental Setup

## 5.1. Datasets
The experiments were conducted on two primary benchmarks:

1.  **Geometry3K:**
    *   **Source:** A standard benchmark for plane geometry problems.
    *   **Scale:** Contains 3,002 problems.
    *   **Splits:** 2,101 for training, 300 for validation, and 601 for testing.
    *   **Characteristics:** Each problem includes a text statement, a diagram, and formal language annotations. It covers high-school level geometry.

2.  **PGPS9K:**
    *   **Source:** An expanded version of Geometry3K.
    *   **Scale:** Contains 9,022 problems with 4,000 unique diagrams.
    *   **Characteristics:** It includes the 2,891 problems from Geometry3K plus additional problems from high school textbooks.

**Data Sample Example:**
To help understand the data format, consider the following example from the paper's Appendix D:
*   **Problem:** Find N. Choices: A) 25, B) 30, C) 50, D) 60.
*   **Diagram Logic Forms:** `PointLiesOnLine(N, Line(M, C))`, `Perpendicular(Line(M, L), Line(P, L))`, etc.
*   **Text Logic Forms:** `Find(LengthOf(Line(P,N)))`.

    These datasets were chosen because they are widely recognized in the field and require both visual understanding (diagrams) and logical reasoning, making them ideal for testing the MARS-GPS pipeline.

## 5.2. Evaluation Metrics
The primary metric used in the paper is **Top-1 Accuracy**.

1.  **Conceptual Definition:** Top-1 Accuracy measures the percentage of problems for which the model's highest-confidence prediction exactly matches the ground-truth label. Since the tasks are multiple-choice, this means checking if the selected option (A, B, C, or D) is the correct one.

2.  **Mathematical Formula:**
    $$
    \text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(\hat{y}_i = y_i)
    $$

3.  **Symbol Explanation:**
    *   $N$: The total number of test samples.
    *   $\hat{y}_i$: The predicted answer for the $i$-th sample.
    *   $y_i$: The ground-truth correct answer for the $i$-th sample.
    *   $\mathbb{1}(\dots)$: The indicator function, which equals 1 if the condition inside is true, and 0 otherwise.

## 5.3. Baselines
The paper compares MARS-GPS against a comprehensive set of baselines categorized as follows:

*   **Neural Methods:** Models that primarily use neural networks to encode diagrams and text (e.g., **NGS**, **Geoformer**, **SCA-GPS**, **GOLD**, **PGPSNet-v2-S**, **LANS**). These represent the "deep learning" approach.
*   **Neural-Symbolic Methods:** Hybrids that use neural nets for parsing and symbolic logic for solving (e.g., **Inter-GPS**, **GeoDRL**, **E-GPS**, **Pi-GPS**). These are the most direct competitors as they also use formal logic.
*   **Multimodal Large Language Models (MLLMs):** General-purpose models like **GPT-4o**, **Gemini 2**, and **Claude 3.5 Sonnet**. These test if a generalist model can outperform a specialized pipeline.
*   **Proprietary LLMs:** Even larger proprietary models (e.g., **GPT-5**, **GPT-5.2**) evaluated on the parsed formal text to isolate reasoning ability from parsing ability.

    These baselines are representative because they cover the entire spectrum of approaches: from pure symbolic to pure neural, and from specialized systems to general-purpose foundation models.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main results demonstrate that MARS-GPS significantly outperforms all baselines.

*   **Geometry3K:** MARS-GPS achieves **88.8%** accuracy.
    *   Compared to the previous SOTA neural-symbolic model (**Pi-GPS** at 77.8%), this is a massive **+11%** improvement.
    *   Compared to general MLLMs like **GPT-4o** (58.6%), it shows that specialized parsing + inference scaling is far superior to end-to-end generation for geometry.
*   **PGPS9K:** MARS-GPS achieves **77.48%** accuracy.
    *   This is nearly **8%** higher than Pi-GPS (69.8%).

        The results strongly validate the paper's hypothesis: that spending compute on generating multiple reasoning paths and aggregating them intelligently is more effective than relying on a single, complex symbolic solver or a general-purpose MLLM.

The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Category</th>
<th rowspan="2">Method</th>
<th rowspan="2">Geometry3K</th>
<th rowspan="2">PGPS9K</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">MLLMs</td>
<td>Qwen-VL (Bai et al., 2023)</td>
<td>26.7</td>
<td>23.2</td>
</tr>
<tr>
<td>GPT-4o (OpenAI et al., 2024)</td>
<td>58.6</td>
<td>51.0</td>
</tr>
<tr>
<td>Claude 3.5 Sonnet (Anthropic, 2024)</td>
<td>56.4</td>
<td>45.9</td>
</tr>
<tr>
<td>Gemini 2 (Google Gemini Team, 2023)</td>
<td>60.7</td>
<td>56.8</td>
</tr>
<tr>
<td rowspan="3">Proprietary LLMs</td>
<td>GPT-5 (OpenAI Team, 2025)</td>
<td>61.5</td>
<td></td>
</tr>
<tr>
<td>GPT-5.2 (Zhang et al., 2026)</td>
<td>73.1</td>
<td></td>
</tr>
<tr>
<td>Claude 4.5 Sonnet (Anthropic Team, 2025)</td>
<td>75.8</td>
<td></td>
</tr>
<tr>
<td rowspan="6">Neural Methods</td>
<td>NGS (Chen et al., 2021)</td>
<td>58.8</td>
<td>46.1</td>
</tr>
<tr>
<td>Geoformer (Chen et al., 2022)</td>
<td>59.3</td>
<td>47.3</td>
</tr>
<tr>
<td>SCA-GPS (Ning et al., 2023)</td>
<td>76.7</td>
<td></td>
</tr>
<tr>
<td>GOLD* (Zhang &amp; Moshfeghi, 2024)</td>
<td>62.7</td>
<td>60.6</td>
</tr>
<tr>
<td>PGPSNet-v2-S* (Zhang et al., 2024a)</td>
<td>76.4</td>
<td>69.2</td>
</tr>
<tr>
<td>LANS (Diagram GT)* (Li et al., 2024)</td>
<td>82.3</td>
<td>74.0</td>
</tr>
<tr>
<td rowspan="4">Neural-symbolic Methods</td>
<td>Inter-GPS (Lu et al., 2021)</td>
<td>57.5</td>
<td></td>
</tr>
<tr>
<td>GeoDRL (Peng et al., 2023)</td>
<td>68.4</td>
<td>66.7</td>
</tr>
<tr>
<td>E-GPS (Wu et al., 2024)</td>
<td>67.9</td>
<td></td>
</tr>
<tr>
<td>Pi-GPS (Zhao et al., 2025)</td>
<td>77.8</td>
<td>69.8</td>
</tr>
<tr>
<td></td>
<td>MARS-GPS (ours)</td>
<td>88.8</td>
<td>77.48</td>
</tr>
</tbody>
</table>

## 6.2. Ablation Studies / Parameter Analysis
The authors performed detailed ablation studies to understand the contribution of each component.

**Voting Strategies:**
The authors compared three strategies for aggregating the $k$ rollouts:
1.  **Majority Voting:** Pick the answer with the most votes. (Accuracy: 85.5%)
2.  **Entropy Sorting:** Pick the answer associated with the rollout that had the lowest average entropy. (Accuracy: 85.5%)
3.  **Entropy-Weighted Voting:** The proposed method, which combines votes and entropy. (Accuracy: 87.5%)

    The following figure (Figure 2a) illustrates the comparison of these voting strategies:

    ![该图像是图表，展示了不同投票策略的准确性。图表标示了三种策略的准确率：多数投票为85.5%，熵排序为85.5%，而熵加权投票则达到87.5%。](images/2.jpg)

    **Component Ablation:**
The authors removed key components to see their impact:
*   **Removing Self-Verification:** Accuracy dropped from 87.5% to 83.0% (-4.5 percentage points). This was the largest drop, highlighting the importance of the model checking its own work.
*   **Removing Code Augmentation:** Accuracy dropped from 87.5% to 85.0% (-2.5 percentage points). This confirms that allowing the model to run Python code helps avoid arithmetic errors.

    The following figure (Figure 2b) shows the impact of removing these components:

    ![该图像是一个图表，展示了MARS-GPS模型与不同组分消融实验的准确率。MARS-GPS模型的准确率为87.5%，而去掉代码增强和自我验证的准确率分别为85.0%和83.0%。](images/3.jpg)

    **Number of CoT Samples (Scaling):**
The authors tested the system with $k \in \{1, 2, 4, 8, 16\}$ rollouts.
*   **Result:** Accuracy increased log-linearly with the number of samples.
    *   $k=1$: 82.0%
    *   $k=8$: 87.5%
    *   $k=16$: 88.0%
*   **Analysis:** This confirms that "inference-time scaling" is effective. However, there are diminishing returns; doubling from 8 to 16 only yielded a 0.5% gain.

    The following figure (Figure 3) illustrates the accuracy scaling with the number of CoT samples:

    ![Figure 3: Accuracy vs. number of CoT samples.](images/4.jpg)
    *该图像是图表，展示了正确率与 CoT 样本数量的关系。随着 CoT 样本数量增加，从 1 到 16，正确率从约 82% 上升至接近 89%，显示出一种正相关趋势。*

The following are the results from Table 2 of the original paper, detailing the accuracy vs. number of CoT samples:

<table>
<thead>
<tr>
<th>Run</th>
<th>Accuracy (%)</th>
</tr>
</thead>
<tbody>
<tr>
<td>CoT 1</td>
<td>82.0</td>
</tr>
<tr>
<td>CoT 2</td>
<td>85.0</td>
</tr>
<tr>
<td>CoT 4</td>
<td>86.5</td>
</tr>
<tr>
<td>CoT 8</td>
<td>87.5</td>
</tr>
<tr>
<td>CoT 16</td>
<td>88.0</td>
</tr>
</tbody>
</table>

**Additional Analysis (Appendix E):**
*   **Python Sandbox Usage:** Problems where the model attempted to use Python but was blocked (sandbox disabled) saw accuracy drop to 75.0%, compared to 92.2% for problems where Python wasn't needed. This shows the sandbox is crucial for hard calculation problems.
*   **Execution Time:** Wrong predictions took 4.2x longer to generate than correct ones (146.1s vs 34.6s). This is because the system had to exhaust all verification steps (Steps 1-6) for hard/ambiguous problems, whereas easy problems hit "Early Consensus" (Step 1) instantly.

    The following are the results from Tables 3, 4, 5, 6, and 7 of the original paper, providing further breakdown of the ablation and analysis:

    <table>
    <thead>
    <tr>
    <th>Problem subset</th>
    <th>% of problems</th>
    <th>Accuracy</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>All problems</td>
    <td>100%</td>
    <td>85.0%</td>
    </tr>
    <tr>
    <td>Did not attempt Python</td>
    <td>58.0%</td>
    <td>92.2%</td>
    </tr>
    <tr>
    <td>Attempted Python (sandbox blocked)</td>
    <td>42.0%</td>
    <td>75.0%</td>
    </tr>
    </tbody>
    </table>

    <table>
    <thead>
    <tr>
    <th>Python calls attempted</th>
    <th>% of problems</th>
    <th>Accuracy</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>0 calls</td>
    <td>58.0%</td>
    <td>92.2%</td>
    </tr>
    <tr>
    <td>1-2 calls</td>
    <td>17.0%</td>
    <td>79.4%</td>
    </tr>
    <tr>
    <td>3-5 calls</td>
    <td>10.0%</td>
    <td>90.0%</td>
    </tr>
    <tr>
    <td>6+ calls</td>
    <td>15.0%</td>
    <td>60.0%</td>
    </tr>
    </tbody>
    </table>

    <table>
    <thead>
    <tr>
    <th>Outcome</th>
    <th>Accuracy</th>
    <th>Avg time (s)</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>All problems</td>
    <td>88.8%</td>
    <td>47.0</td>
    </tr>
    <tr>
    <td>Correct predictions</td>
    <td>—</td>
    <td>34.6</td>
    </tr>
    <tr>
    <td>Wrong predictions</td>
    <td></td>
    <td>146.1</td>
    </tr>
    </tbody>
    </table>

    <table>
    <thead>
    <tr>
    <th>Time range</th>
    <th>% of problems</th>
    <th>Accuracy</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>0-10s</td>
    <td>35.3%</td>
    <td>98.6%</td>
    </tr>
    <tr>
    <td>10-30s</td>
    <td>31.2%</td>
    <td>94.1%</td>
    </tr>
    <tr>
    <td>30-60s</td>
    <td>14.8%</td>
    <td>89.9%</td>
    </tr>
    <tr>
    <td>60-120s</td>
    <td>9.0%</td>
    <td>72.2%</td>
    </tr>
    <tr>
    <td>120-180s</td>
    <td>3.0%</td>
    <td>61.1%</td>
    </tr>
    <tr>
    <td>180-300s</td>
    <td>3.5%</td>
    <td>47.6%</td>
    </tr>
    <tr>
    <td>&gt;300s</td>
    <td>3.2%</td>
    <td>42.1%</td>
    </tr>
    </tbody>
    </table>

    <table>
    <thead>
    <tr>
    <th>Category</th>
    <th>Accuracy</th>
    <th>Avg time (s)</th>
    <th>Correct (s)</th>
    <th>Wrong (s)</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Similar figures</td>
    <td>96.2%</td>
    <td>12.9</td>
    <td>11.0</td>
    <td>60.0</td>
    </tr>
    <tr>
    <td>Trigonometry</td>
    <td>85.7%</td>
    <td>12.8</td>
    <td>5.4</td>
    <td>57.1</td>
    </tr>
    <tr>
    <td>Triangle</td>
    <td>100%</td>
    <td>31.2</td>
    <td>31.2</td>
    <td>—</td>
    </tr>
    <tr>
    <td>Quadrilateral</td>
    <td>96.9%</td>
    <td>34.9</td>
    <td>28.7</td>
    <td>227.7</td>
    </tr>
    <tr>
    <td>Length/Other</td>
    <td>85.6%</td>
    <td>48.3</td>
    <td>27.1</td>
    <td>173.6</td>
    </tr>
    <tr>
    <td>Angle</td>
    <td>91.8%</td>
    <td>51.7</td>
    <td>45.5</td>
    <td>121.1</td>
    </tr>
    <tr>
    <td>Area</td>
    <td>77.4%</td>
    <td>53.3</td>
    <td>37.8</td>
    <td>106.5</td>
    </tr>
    <tr>
    <td>Circle</td>
    <td>90.0%</td>
    <td>56.6</td>
    <td>42.7</td>
    <td>181.8</td>
    </tr>
    </tbody>
    </table>

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that **MARS-GPS**, a framework based on multi-path Chain-of-Thought voting and entropy-based aggregation, significantly advances the state-of-the-art in Geometry Problem Solving. By decoupling diagram parsing from logical inference and leveraging inference-time scaling (parallel rollouts), the model achieves **88.8%** accuracy on Geometry3K. The key takeaway is that logical inference in geometry is not just about finding *one* correct path, but about exploring *multiple* paths and intelligently selecting the best one using confidence signals and self-verification.

## 7.2. Limitations & Future Work
**Limitations:**
1.  **Parsing Dependency:** The system relies heavily on the initial parsing stage (PGDPNet). If the parser produces incorrect formal logic $\mathcal{F}^*$, the downstream reasoning cannot recover, regardless of how many rollouts are used.
2.  **Computational Cost:** The cost scales linearly with the number of rollouts $k$. While batching helps, it is still more expensive than a single-pass model.
3.  **Scope:** Currently limited to multiple-choice problems. It has not been tested on open-ended answer generation or formal theorem proving.

**Future Work:**
The authors suggest several promising directions:
1.  **Improved Parsing:** Integrating MLLMs directly into the parsing stage to fix errors in $\math{F}^*$.
2.  **Training + Inference:** Combining inference-time scaling with training-time improvements (fine-tuning the base LLM on geometry data).
3.  **Broader Applicability:** Extending the framework to open-ended problems and formal theorem proving (e.g., using Lean).
4.  **Adaptive Budgeting:** Dynamically adjusting the number of rollouts $k$ based on problem difficulty (easy problems get fewer rollouts, hard problems get more).

## 7.3. Personal Insights & Critique
**Inspirations:**
The most inspiring aspect of this paper is the validation of **"Inference-Time Scaling"**. It suggests that we don't always need to train bigger, more expensive models. Instead, we can make existing models "smarter" simply by letting them think longer and check their work. The use of **token entropy** as a free confidence signal is a clever, low-hanging fruit that other domains could likely adopt.

**Potential Issues:**
*   **Model Dependency:** The results rely on "GPT-OSS 120B," a very large, proprietary model. It is unclear if these gains would translate to smaller, open-source models (like Llama-3-8B) that might be less robust in their base reasoning.
*   **Complexity:** The pipeline is quite complex (Parsing -> Parallel Rollouts -> Entropy Calc -> 6-Step Aggregation). This complexity might make it harder to debug or deploy in real-time systems compared to a single end-to-end model.
*   **Data Contamination:** Given the use of massive proprietary models trained on vast internet data, there is always a small risk of data contamination (the model having seen the Geometry3K problems during training), though the authors mitigate this by focusing on the *methodology* (aggregation gains) rather than just the raw score.

**Transferability:**
The methodology is highly transferable to any domain requiring multi-step reasoning where multiple paths exist, such as:
*   **Code Generation:** Generating multiple code snippets and voting on the one that compiles/runs successfully.
*   **Legal/Medical Reasoning:** Generating multiple arguments and selecting the most consistent one.
*   **Math Word Problems:** Beyond geometry, this approach should work well for general algebra or calculus.