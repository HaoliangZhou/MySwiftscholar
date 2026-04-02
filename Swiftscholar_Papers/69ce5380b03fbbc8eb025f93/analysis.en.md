# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is "Query-Conditioned Evidential Keyframe Sampling for MLLM-Based Long-Form Video Understanding". This title highlights the core focus: developing a method to intelligently select keyframes from long videos, specifically tailored to a user's query, to enhance the performance of Multimodal Large Language Models (MLLMs).

## 1.2. Authors
The authors are Yiheng Wang, Lichen Zhu, Yueqian Lin, Yudong Liu, Jingyang Zhang, Hai "Helen" Li, and Yiran Chen.
*   **Affiliations**: Most authors (1-6) are affiliated with Duke University, Durham, North Carolina, USA. Jingyang Zhang (2) is listed as an Independent Researcher.
*   **Research Background**: The affiliation with Duke University suggests a strong academic focus in computer vision, machine learning, and efficient AI systems. The presence of an "Independent Researcher" indicates collaboration spanning both institutional and independent research contexts.

## 1.3. Journal/Conference
The paper is currently available as a preprint on arXiv (arXiv:2604.01002). The "Published at (UTC)" timestamp is **2026-04-01**, indicating it is a very recent work (from the perspective of the simulation) or a future-dated publication. It has not yet been assigned to a specific journal or conference proceedings in the provided text, though it references recent conferences like CVPR and ICCV in its citations.

## 1.4. Publication Year
The publication year is **2026**.

## 1.5. Abstract
The paper addresses the challenge of applying Multimodal Large Language Models (MLLMs) to long-form videos, which is constrained by limited context windows and high computational costs. The authors propose an evidence-driven keyframe sampling framework grounded in **Information Bottleneck theory**. Instead of relying on simple semantic relevance or inefficient reinforcement learning, they formulate keyframe selection as maximizing the **conditional mutual information** between selected frames and the query. This objective is decomposed into tractable, independent frame-level scoring using a modular upper bound relaxation. They introduce a query-conditioned evidence scoring network trained with a contrastive objective. Experiments on benchmarks like LVBench and VideoMME show that the method outperforms prior sampling strategies under strict token budgets and significantly improves training efficiency.

## 1.6. Original Source Link
*   **arXiv Link**: https://arxiv.org/abs/2604.01002
*   **PDF Link**: https://arxiv.org/pdf/2604.01002
*   **Status**: Preprint (arXiv).

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem is the **context length bottleneck** in Multimodal Large Language Models (MLLMs) when processing long-form videos. MLLMs have shown strong capabilities in video question answering, but their practical deployment is limited because feeding every frame of a long video (e.g., hours of footage) exceeds the model's token limit and incurs prohibitive computational costs. Therefore, **keyframe sampling**—selecting a small subset of representative frames—is essential.

Existing approaches face significant challenges:
1.  **Semantic-matching heuristics**: These methods select frames that look similar to the query text (e.g., using CLIP similarity). However, the paper argues that "visual relevance" does not equal "evidential usefulness." A frame might visually match a query word but lack the specific evidence needed to answer the question (e.g., a frame showing a "car" is relevant, but the answer depends on the "license" plate visible in a specific split second).
2.  **Reinforcement Learning (RL)**: While RL methods can optimize for answer accuracy, they suffer from inefficient combinatorial optimization (searching through all possible frame subsets) and sparse rewards, making training unstable and slow.

    The paper's innovative entry point is to treat keyframe selection as an **Information Bottleneck** problem. Instead of just matching semantics, the goal is to select frames that maximize the **information** about the answer given the query.

## 2.2. Main Contributions / Findings
The paper makes three primary contributions:
1.  **Evidence Scoring Framework**: It proposes a theoretical formulation for keyframe selection as maximizing conditional mutual information $I(S; O | Q)$. It proves this objective is submodular and uses a modular upper bound relaxation to decompose the complex subset selection problem into efficient, independent frame-level scoring.
2.  **Query-Conditioned Evidence Scoring Network**: It designs a neural network architecture that estimates the evidential value of each frame relative to a query. This network includes a temporal evidence aggregator and a query-guided gating mechanism.
3.  **Efficient Training via Contrastive Learning**: It trains the scoring network using a contrastive objective (InfoNCE), which is significantly more efficient than RL-based methods (0.6 hours vs. 78 hours) because it avoids "MLLM-in-the-loop" supervision.

    Key findings include consistent outperformance of state-of-the-art methods on LVBench and VideoMME benchmarks (e.g., a 10.1% accuracy gain for Qwen2.5-VL-7B on LVBench) and the demonstration that hallucinations in long-video understanding are largely driven by missing evidence, which the proposed method effectively mitigates.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several fundamental concepts:

*   **Multimodal Large Language Models (MLLMs)**: These are advanced AI models (like GPT-4V, LLaVA, Qwen-VL) that can process and understand multiple modalities—typically text and images/videos simultaneously. They generate text responses based on visual inputs.
*   **Keyframe Sampling**: The process of selecting a small set of frames from a video that best represent its content, used to reduce data redundancy and computational load.
*   **Information Bottleneck (IB) Principle**: A technique from information theory that finds the best trade-off between compressing a variable (selecting few frames) and preserving relevant information about another variable (the answer). It aims to maximize $I(\text{Compressed}; \text{Target})$ while minimizing $I(\text{Compressed}; \text{Input})$.
*   **Mutual Information (MI)**: A measure of the mutual dependence between two variables. In this context, `I(X; Y)` quantifies how much knowing $X$ reduces uncertainty about $Y$. The paper uses **Conditional Mutual Information**, $I(S; O | Q)$, which measures the information shared between the selected frames $S$ and the output $O$, given the query $Q$.
*   **Submodular Functions**: A set function $F$ is submodular if it exhibits "diminishing returns." Formally, for sets $A \subseteq B$ and an element $x \notin B$, $F(A \cup \{x\}) - F(A) \geq F(B \cup \{x\}) - F(B)$. This property is crucial because it allows greedy algorithms to provide provable guarantees (e.g., $(1 - 1/e)$-approximation) for maximization problems.
*   **Contrastive Learning**: A training paradigm where a model learns to distinguish between "similar" (positive) and "dissimilar" (negative) pairs of data points. The **InfoNCE loss** is a common objective function used in this context.

## 3.2. Previous Works
The paper categorizes prior work into two main streams:

1.  **Training-free Semantic-matching Methods**:
    *   **AKS and FOCUS**: These methods balance semantic relevance and temporal coverage. They typically use pre-trained Vision-Language Models (like CLIP) to calculate the similarity between a frame and the query text.
    *   **Q-Frame**: Performs query-aware sampling with dynamic multi-resolution scaling.
    *   *Limitation*: As noted, these rely on heuristics that prioritize visual alignment over evidential utility. They might select a frame that looks like the query but doesn't contain the answer.

2.  **Trainable Policies (RL-based)**:
    *   **TSPO (Temporal Sampling Policy Optimization)**: Trains an agent using Reinforcement Learning (specifically GRPO) to select frames. It explores the combinatorial space of frame subsets to maximize a reward (usually answer accuracy).
    *   **ReaSon**: Uses a causal information bottleneck with counterfactual interventions.
    *   *Limitation*: These methods face the "curse of dimensionality" in the search space. They require evaluating the MLLM repeatedly during training (MLLM-in-the-loop), which is computationally expensive. Furthermore, they rely on sparse rewards (e.g., 1 if correct, 0 if wrong), making it hard to assign credit to individual frames.

## 3.3. Technological Evolution
The field has evolved from simple **uniform sampling** (taking every $n$-th frame) to **semantic-aware sampling** (using CLIP scores), and recently to **RL-based optimization**. This paper represents the next step: **Information-Theoretic Optimization**. It bridges the gap by providing a theoretically grounded objective (like RL) but with the efficiency of heuristic methods (like semantic matching) by decomposing the problem into independent scoring.

## 3.4. Differentiation Analysis
The core differentiation lies in the **objective function** and the **optimization strategy**.
*   **vs. Semantic Matching**: Instead of maximizing $P(\text{frame} | \text{query})$ (relevance), this method maximizes $I(\text{frame}; \text{answer} | \text{query})$ (evidence).
*   **vs. RL**: Instead of searching the subset space directly (combinatorial), it leverages the **submodularity** of the mutual information objective to relax it into a sum of independent scores. This removes the need for expensive MLLM rollouts during training, replacing the sparse RL reward with a dense, learnable scoring signal.

# 4. Methodology

## 4.1. Principles
The core principle is the **Evidence Bottleneck**. The authors define the optimal set of keyframes $S$ as the one that provides the most information about the MLLM's output (answer) $O$, given the query $Q$, subject to a budget constraint $|S| \le m$.

Mathematically, the problem is formulated as:
$$
\operatorname*{max}_{S} I(S; O \mid Q) \quad \text{s.t.} \quad |S| \leq m
$$
where:
*   $S$: The set of selected frames.
*   $O$: The output (answer) of the MLLM.
*   $Q$: The input query.
*   $I(S; O \mid Q)$: The conditional mutual information between the selected frames and the answer, given the query.

    The intuition is that a frame is valuable only if it reduces the model's uncertainty about the answer.

## 4.2. Core Methodology In-depth

### Step 1: Objective Expansion and Equivalence
The authors first expand the mutual information term to align the objective with maximizing answer accuracy. The definition of conditional mutual information is:
$$
I(S; O \mid Q) = \mathbb{E}_{Q} \left[ \mathbb{E}_{p(S, O \mid Q)} \left[ \log \frac{p(O \mid S, Q)}{p(O \mid Q)} \right] \right]
$$
Here, $\mathbb{E}$ denotes the expectation (average) over the distributions. Since $p(O \mid Q)$ (the probability of the answer given only the query, without video) is constant with respect to the selection $S$, maximizing $I(S; O \mid Q)$ is equivalent to maximizing the log-likelihood of the answer given the selected frames:
$$
\operatorname*{max}_{S} \mathbb{E} [ \log p(O \mid S, Q) ]
$$
This confirms that the theoretical goal of maximizing information is practically the same as maximizing the probability of the correct answer.

### Step 2: Submodularity and Modular Relaxation
Direct optimization is intractable because the number of subsets $\binom{n}{m}$ is exponential. The authors leverage the property that Shannon entropy is submodular. The objective function can be rewritten as:
$$
F(S) = I(S; O \mid Q) = H(O \mid Q) - H(O \mid S, Q)
$$
where $H$ is the entropy (uncertainty). Since entropy is submodular, `F(S)` is monotone non-decreasing and submodular.

To avoid the expensive greedy selection required for submodular functions, the authors use a **modular upper bound relaxation**. For any monotone submodular function $F$ with $F(\varnothing) = 0$, the following inequality holds due to diminishing returns:
$$
F(S) \leq \sum_{f_i \in S} F(\{f_i\}) = \sum_{f_i \in S} I(f_i; O \mid Q)
$$
This relaxation transforms the subset selection problem into selecting the top $m$ frames with the highest individual mutual information scores $I(f_i; O \mid Q)$. This is computationally efficient as it requires only sorting, not combinatorial search.

### Step 3: Temporal Selection Strategy
To ensure temporal diversity (avoiding clustering all selected frames in one second), the video is partitioned into $B$ equal-length temporal segments (bins). The top $k$ frames are selected from each bin based on their scores.
$$
S^{*} = \bigcup_{b=1}^{B} \mathrm{top}-k_{f_i \in BIN_b} I(f_i; O \mid Q)
$$
where $BIN_b$ is the set of frames in the $b$-th segment. This strategy balances local evidence density with global temporal coverage.

### Step 4: Query-Conditioned Evidence Scoring Network
Since the true mutual information $I(f_i; O \mid Q)$ depends on the unknown answer $O$, the authors train a proxy network $g_\theta(f_i, Q)$ to estimate it.

The architecture consists of four main components:

1.  **Vision-Language Encoder**: A shared encoder (e.g., CLIP) processes the frame and the query into dense vector representations.
    $$
    \mathbf{v}_i = \mathcal{E}_\nu(f_i) \in \mathbb{R}^d, \quad \mathbf{q} = \mathcal{E}_t(Q) \in \mathbb{R}^d
    $$
    $\mathbf{v}_i$ is the visual feature vector, and $\mathbf{q}$ is the query feature vector.

2.  **Temporal Evidence Aggregator**: To capture short-term temporal context (e.g., an action unfolding over a few frames), a causal window attention mechanism is used. For a frame at position $\tau$, it aggregates features from a local window $\mathcal{W}_\tau$.
    $$
    \mathbf{h}_\tau = \mathcal{T} \big ( \mathbf{v}_\tau \mid \{ \mathbf{v}_j : f_j \in \mathcal{W}_\tau \} \big )
    $$
    $\mathcal{T}$ represents the temporal aggregation function (like attention), and $\mathbf{h}_\tau$ is the context-aware visual feature.

3.  **Query-Guided Evidence Gating**: This module acts as a filter. It uses the query vector $\mathbf{q}$ to compute a gate vector $\mathbf{g}_i$ that suppresses irrelevant features in $\mathbf{h}_i$.
    $$
    \mathbf{g}_i = \sigma ( \mathbf{W}_h \mathbf{h}_i + \mathbf{W}_q \mathbf{q} + \mathbf{b} ), \quad \mathbf{u}_i = \mathbf{h}_i \odot \mathbf{g}_i
    $$
    $\sigma$ is the sigmoid function (outputting values between 0 and 1), $\mathbf{W}$ are weight matrices, $\mathbf{b}$ is a bias, and $\odot$ is element-wise multiplication. The resulting vector $\mathbf{u}_i$ contains only the query-relevant evidence.

4.  **Evidence Score Head**: The final score is computed by measuring the alignment between the gated visual feature $\mathbf{u}_i$ and the query $\mathbf{q}$ across multiple semantic subspaces.
    In the $k$-th subspace:
    $$
    s_{i,k} = \frac{1}{\gamma_k} \frac{ ( \mathbf{W}_\nu^{(k)} \mathbf{u}_i )^\top ( \mathbf{W}_q^{(k)} \mathbf{q} ) }{ \| \mathbf{W}_\nu^{(k)} \mathbf{u}_i \|_2 \| \mathbf{W}_q^{(k)} \mathbf{q} \|_2 }
    $$
    This calculates the cosine similarity between projected versions of the visual and query features. The overall score combines this subspace similarity with a direct global similarity term:
    $$
    g_\theta(f_i, Q) = \lambda \frac{ \mathbf{v}_i^\top \mathbf{q} }{ \| \mathbf{v}_i \|_2 \| \mathbf{q} \|_2 } + (1 - \lambda) \frac{1}{K} \sum_{k=1}^K s_{i,k}
    $$
    $\lambda$ is a learnable parameter balancing the two terms.

The following figure illustrates the overall framework:

![该图像是一个示意图，展示了基于查询条件的证据关键帧采样框架。框架利用图像编码器和时间证据聚合模块，通过查询引导的证据门控来评估关键帧的证据重要性，并通过 $E_{score}$ 输出评分，显示哪些关键帧对于回答问题最为重要。最终，经过加权聚合的关键帧被选为多模态大语言模型的输入。](images/2.jpg)
*该图像是一个示意图，展示了基于查询条件的证据关键帧采样框架。框架利用图像编码器和时间证据聚合模块，通过查询引导的证据门控来评估关键帧的证据重要性，并通过 $E_{score}$ 输出评分，显示哪些关键帧对于回答问题最为重要。最终，经过加权聚合的关键帧被选为多模态大语言模型的输入。*

### Step 5: Training with Contrastive Objective
The network is trained to distinguish between "positive" frames (those inside the annotated evidence segment sufficient to answer the query) and "negative" frames (background frames). The authors use the **InfoNCE loss**, a standard contrastive loss function.
$$
\mathcal{L} = -\log \frac{ \sum_{x \in \mathcal{F}^+} \exp g_\theta(x, Q) }{ \sum_{x \in \mathcal{F}^+} \exp g_\theta(x, Q) + \sum_{x \in \mathcal{F}^-} \exp g_\theta(x, Q) }
$$
*   $\mathcal{F}^+$: Set of positive frames.
*   $\mathcal{F}^-$: Set of negative frames.
*   The loss encourages the model to assign high scores to positive frames and low scores to negative frames.

    This loss implies that the optimal score function is proportional to the log-ratio of the probabilities:
$$
g_\theta^*(x, Q) \propto \log \frac{ p(x \mid \mathcal{F}^+, Q) }{ p(x \mid \mathcal{F}^-, Q) } + C
$$
This effectively trains the network to recognize "evidence" vs. "background" for a specific query.

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize two distinct types of datasets for training and evaluation:

*   **Training Dataset**: The model is trained on the **LLaVA-Video subset of the Seek-173K dataset**. This dataset provides annotated "evidence segments"—specific time ranges in the video verified to contain the necessary information to answer the query. Frames within these segments are treated as positive samples ($\mathcal{F}^+$), and frames outside are treated as negatives ($\mathcal{F}^-$).
*   **Evaluation Datasets**:
    *   **LVBench**: A benchmark for extreme long-video understanding with an average duration of over 4,000 seconds (~68 minutes). It requires models to handle very long contexts and retrieve specific details.
    *   **VideoMME**: A comprehensive benchmark designed to evaluate MLLMs on video analysis tasks, including both short and long videos.

## 5.2. Evaluation Metrics
The primary metric used is **Question Answering (QA) Accuracy**.

1.  **Conceptual Definition**: This metric measures the percentage of questions for which the model's generated response matches the ground-truth answer. It directly reflects the model's effectiveness in understanding the video content and resolving the user's query.
2.  **Mathematical Formula**:
    $$
    \text{Accuracy} = \frac{ \text{Number of Correct Answers} }{ \text{Total Number of Questions} }
    $$
3.  **Symbol Explanation**: The formula is a simple ratio where the numerator counts correct predictions and the denominator is the total test set size.

## 5.3. Baselines
The paper compares the proposed method against a comprehensive set of baselines:
*   **Agentic MLLMs**: VideoTree, VideoAgent, VCA.
*   **SFT/RL MLLMs**: MovieChat, TimeMarker, VideoLLaMA3, Video-R1, Video-Thinker, Video-o3.
*   **Keyframe Sampling Methods**:
    *   **Heuristic/Training-free**: AKS, FOCUS, Q-Frame.
    *   **Trainable**: MLLM-Selector.
    *   **RL-based**: TSPO, ReTaKe, FrameThinker.

        These baselines represent the state-of-the-art in spectrum from simple heuristics to complex reinforcement learning approaches.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results demonstrate that the proposed evidence-driven sampling method consistently outperforms existing strategies, particularly under strict frame budgets.

*   **LVBench Performance**: On the LVBench benchmark (Table 1), the method achieves significant improvements over the base MLLMs (Qwen2-VL-7B, Qwen2.5-VL-7B, LLaVA-Video-7B). For instance, with Qwen2.5-VL-7B and a budget of only 32 frames, the method achieves an overall accuracy of **47.7%**, compared to **37.6%** for the base model. This represents a substantial **10.1% absolute gain**. It also outperforms other sampling methods like TSPO and ReTaKe.
*   **VideoMME Performance**: On VideoMME (Table 2), the method boosts the average accuracy of Qwen2.5-VL-7B from **60.7%** (uniform) to **63.6%**. It shows particularly strong results on the "Long" video subset, improving from **50.6%** to **55.0%**, validating its capability on long-duration content.

    The following are the results from Table 1 of the original paper:

    <table>
    <thead>
    <tr>
    <th rowspan="2">Method</th>
    <th rowspan="2">Frames</th>
    <th colspan="7">LVBench</th>
    </tr>
    <tr>
    <th>Overall</th>
    <th>ER</th>
    <th>EU</th>
    <th>KIR</th>
    <th>TG</th>
    <th>Rea</th>
    <th>Sum</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td colspan="9"><strong>Agentic MLLMs</strong></td>
    </tr>
    <tr>
    <td>VideoTree [31]</td>
    <td></td>
    <td>28.8</td>
    <td>30.3</td>
    <td></td>
    <td>25.1</td>
    <td>26.5</td>
    <td></td>
    <td>27.7</td>
    </tr>
    <tr>
    <td>VideoAgent [32]</td>
    <td></td>
    <td>29.3</td>
    <td>28.0</td>
    <td>30.3</td>
    <td>28.0</td>
    <td></td>
    <td></td>
    <td>29.3</td>
    </tr>
    <tr>
    <td>VCA [33]</td>
    <td></td>
    <td>41.3</td>
    <td>43.7</td>
    <td>40.7</td>
    <td>37.8</td>
    <td>38.0</td>
    <td>46.2</td>
    <td>27.3</td>
    </tr>
    <tr>
    <td colspan="9"><strong>SFT/RL MLLMs</strong></td>
    </tr>
    <tr>
    <td>MovieChat-7B [34]</td>
    <td>&gt;10000</td>
    <td>22.5</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>21.3</td>
    </tr>
    <tr>
    <td>TimeMarker-8B [35]</td>
    <td>&le;128</td>
    <td>41.3</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>42.8</td>
    </tr>
    <tr>
    <td>VideoLLaMA3-7B [19]</td>
    <td>-</td>
    <td>45.3</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>Video-R1-7B [36]</td>
    <td>32</td>
    <td>37.4</td>
    <td>-</td>
    <td>-</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>Video-Thinker-7B [37]</td>
    <td>32</td>
    <td>38.4</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>Video-03 [28]</td>
    <td>&le; 768</td>
    <td>47.6</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td colspan="9"><strong>Keyframe Sampling for MLLMs</strong></td>
    </tr>
    <tr>
    <td>Qwen2-VL-7B [2]</td>
    <td>32</td>
    <td>39.1</td>
    <td></td>
    <td>38.7</td>
    <td>39.0</td>
    <td>36.8</td>
    <td>37.3</td>
    <td>39.8</td>
    </tr>
    <tr>
    <td>+ ReTaKe [38]</td>
    <td>&le; 2048</td>
    <td>47.8</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>+ TSPO [14]</td>
    <td>64</td>
    <td>46.4</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>+ Ours</td>
    <td>32</td>
    <td>46.6</td>
    <td></td>
    <td>47.9</td>
    <td>44.5</td>
    <td>55.0</td>
    <td>42.7</td>
    <td>43.8</td>
    </tr>
    <tr>
    <td>Qwen2.5-VL-7B [15]</td>
    <td>32</td>
    <td>37.6</td>
    <td></td>
    <td>36.8</td>
    <td>38.6</td>
    <td>40.6</td>
    <td>32.7</td>
    <td>37.3</td>
    </tr>
    <tr>
    <td>+ FrameThinker [39]</td>
    <td>23.9</td>
    <td>36.6</td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>+ Ours</td>
    <td>32</td>
    <td>47.7</td>
    <td></td>
    <td>49.8</td>
    <td>44.5</td>
    <td>57.0</td>
    <td>37.3</td>
    <td>41.8</td>
    </tr>
    <tr>
    <td>LLaVA-Video-7B [40]</td>
    <td>64</td>
    <td>41.7</td>
    <td></td>
    <td>41.5</td>
    <td>40.2</td>
    <td>42.3</td>
    <td>33.2</td>
    <td>49.8</td>
    </tr>
    <tr>
    <td>+ ReTaKe [38]</a></td>
    <td>&le; 1024</td>
    <td>48.5</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>+ TSPO [14]</td>
    <td>64</td>
    <td>45.3</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    </tr>
    <tr>
    <td>+ Ours</td>
    <td>64</td>
    <td>49.4</td>
    <td></td>
    <td>52.4</td>
    <td>45.9</td>
    <td>54.6</td>
    <td>39.0</td>
    <td>46.8</td>
    </tr>
    </tbody>
    </table>

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">Architecture</th>
<th rowspan="2">Frames</th>
<th colspan="2">VideoMME</th>
</tr>
<tr>
<th>Long</th>
<th>Average</th>
</tr>
</thead>
<tbody>
<tr>
<td>Qwen2-VL-7B</td>
<td></td>
<td>32</td>
<td>48.7</td>
<td>58.0</td>
</tr>
<tr>
<td>+ AKS [9]</td>
<td>BLIP-0.5B</td>
<td>32</td>
<td>-</td>
<td>59.9</td>
</tr>
<tr>
<td>+ FOCUS [11]</td>
<td>BLIP-0.5B</td>
<td>32</td>
<td>-</td>

<      <td>59.7</td>
    </tr>
    <tr>
      <td>+ Q-Frame [10]</td>
      <td>CLIP-0.4B</td>
      <td>44</td>
      <td>48.3</td>
      <td>58.3</td>
    </tr>
    <tr>
      <td>+ MLLM-Selector [20]</td>
      <td>MLLM-1.5B</td>
      <td>32</td>
      <td>-</td>
      <td>58.7</td>
    </tr>
    <tr>
      <td>+ Ours</td>
      <td>CLIP-0.4B</td>
      <td>32</td>
      <td>51.4</td>
      <td>60.1</td>
    </tr>
    <tr>
      <td>+ ReTaKe [38]</td>
      <td></td>
      <td>&le; 2048</td>
      <td>56.2</td>
      <td>63.9</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-7B</td>
      <td>-</td>
      <td>32</td>
      <td>50.6</td>
      <td>60.7</td>
    </tr>
    <tr>
      <td>+ K-Frames [41]</td>
      <td>MLLM-3B</td>
      <td>32</td>
      <td>-</td>
      <td>62.1</td>
    </tr>
    <tr>
      <td>+ AKS [9]</td>
      <td>CLIP-0.4B</td>
      <td>32</td>
      <td></td>
      <td>62.4</td>
    </tr>
    <tr>
      <td>+ BOLT [42]</td>
      <td>CLIP-0.4B</td>
      <td>32</td>
      <td></td>
      <td>62.0</td>
    </tr>
    <tr>
      <td>+ ASCS [43]</td>
      <td>CLIP-0.4B</td>
      <td>32</td>
      <td></td>
      <td>63.1</td>
    </tr>
    <tr>
      <td>+ Ours</td>
      <td>CLIP-0.4B</td>
      <td>32</td>
      <td>55.0</td>
      <td>63.6</td>
    </tr>
  </tbody>
</table>

## 6.2. Ablation Studies / Parameter Analysis

### Impact of Pre-trained Encoders
Table 3 presents an ablation study on the choice of the vision-language encoder (CLIP, SigLIP, SigLIP2). The results show that while the specific choice of encoder impacts the final score (e.g., SigLIP slightly outperforms CLIP for Qwen2.5-VL-7B), the proposed sampling method consistently outperforms uniform sampling across all encoder choices. This demonstrates the robustness and generality of the framework.

The following are the results from Table 3 of the original paper:

| Method | Sampling | Encoder | VideoMME | LVBench |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2-VL-7B | Uniform | - | 58.0 | 39.1 |
| Qwen2-VL-7B | Ours | CLIP | 60.1 | 46.6 |
| Qwen2-VL-7B | Ours | SigLIP | 59.9 | 45.8 |
| Qwen2-VL-7B | Ours | SigLIP2 | 59.0 | 44.2 |
| Qwen2.5-VL-7B | Uniform | - | 60.7 | 37.6 |
| Qwen2.5-VL-7B | Ours | CLIP | 63.6 | 47.7 |
| Qwen2.5-VL-7B | Ours | SigLIP | 64.4 | 48.5 |
| Qwen2.5-VL-7B | Ours | SigLIP2 | 61.8 | 43.6 |

### Impact of Keyframe Sampling Budgets
Table 4 analyzes performance under varying frame budgets (8, 16, 32, 64). The proposed method consistently outperforms uniform sampling at every budget level. Notably, with only **8 frames**, the proposed method achieves **45.1%** accuracy, which is higher than uniform sampling with 64 frames (**41.7%**). This highlights the method's exceptional efficiency in identifying high-value evidence.

The following are the results from Table 4 of the original paper:

| Frames | LVBench (Uniform) | LVBench (Ours) |
| :--- | :--- | :--- |
| 8 | 35.8 | 45.1 (↑ 9.3) |
| 16 | 37.1 | 46.0 (↑ 8.9) |
| 32 | 40.9 | 48.3 (↑ 7.4) |
| 64 | 41.7 | 49.4 (↑ 7.7) |

### Efficiency Analysis
The paper reports significant efficiency gains compared to RL-based methods. The training time for the proposed method is **0.6 hours** on 264K samples, whereas the RL baseline (TSPO) requires **78 hours** on only 10K samples. This massive speedup (over 100x faster in terms of samples processed per hour) is attributed to decoupling the MLLM from the training loop and using dense contrastive signals instead of sparse RL rewards.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully addresses the bottleneck of long-form video understanding for MLLMs by proposing a query-conditioned evidential keyframe sampling framework. By grounding the selection process in Information Bottleneck theory and maximizing conditional mutual information, the method prioritizes frames that provide actual evidence for the answer. The use of a modular upper bound relaxation allows for efficient, independent frame scoring, avoiding the combinatorial explosion of RL methods. The results demonstrate state-of-the-art performance on LVBench and VideoMME, alongside significantly improved training efficiency.

## 7.2. Limitations & Future Work
The authors identify several key limitations:
1.  **Audio-Centric Questions**: The current framework operates exclusively on visual frames. It fails for questions requiring auditory evidence (e.g., dialogue, lyrics, sound effects). Future work could integrate audio encoders.
2.  **Timestamp-Grounded Questions**: The scoring network is not explicitly trained to reason about absolute timestamps. It may underperform on questions asking "What happens between 02:45 and 04:00?". Future work could incorporate timestamp-aware supervision.
3.  **Global Understanding**: As noted in the qualitative analysis, the method focuses on evidence density, which can sometimes lead to temporal clustering. For questions requiring broad, global video summarization, uniform sampling might still be superior due to better coverage.

## 7.3. Personal Insights & Critique
The paper presents a compelling theoretical framework that effectively bridges the gap between heuristic efficiency and RL performance. The shift from "relevance" to "evidence" is a crucial conceptual advancement for video QA.

*   **Strengths**: The rigorous theoretical grounding (Information Bottleneck) is a major strength. The efficiency gains over RL are practically significant, making long-video training accessible to more researchers. The qualitative analysis in the supplementary material provides honest insights into failure cases.
*   **Potential Issues**: The reliance on annotated "evidence segments" in the training data (Seek-173K) is a potential dependency. If such annotations are not available for new domains, the method might require expensive labeling or a synthetic data generation pipeline. Additionally, while the modular relaxation is efficient, it is an approximation; the true optimal subset might require inter-frame dependencies that the independent scoring misses.
*   **Future Directions**: Extending this framework to multi-modal evidence (audio + text + video) seems like a natural and highly valuable next step. Furthermore, integrating the scoring network directly into the MLLM's attention mechanism (e.g., as a bias for token importance) could yield end-to-end improvements.