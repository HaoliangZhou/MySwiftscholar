# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is "Query-Conditioned Evidential Keyframe Sampling for MLLM-Based Long-Form Video Understanding". This title highlights the paper's focus on optimizing the processing of long videos by Multimodal Large Language Models (MLLMs). The key concepts are:
- **Keyframe Sampling:** The process of selecting a subset of frames from a video to represent its content, thereby reducing computational load.
- **Query-Conditioned:** The selection process is driven by a specific user question or query, rather than being generic.
- **Evidential:** The selected frames must provide evidence necessary to answer the query.
- **MLLM-Based:** The context is using Multimodal Large Language Models for the video understanding task.
- **Long-Form Video:** The specific challenge is handling videos that are long in duration, which poses significant computational and context-window challenges.

## 1.2. Authors
The authors are Yiheng Wang, Lichen Zhu, Yueqian Lin, Yudong Liu, Jingyang Zhang, Hai "Helen" Li, and Yiran Chen.
- **Affiliations:** Most authors (1-5, 7) are affiliated with **Duke University**, Durham, North Carolina, USA. Author 6 (Jingyang Zhang) is listed as an **Independent Researcher**.
- **Research Background:** Based on the affiliations and the topic, the authors are likely researchers in computer vision, machine learning, and natural language processing, specifically focusing on multimodal learning and video understanding. Duke University is a prominent institution with strong research groups in these areas.

## 1.3. Journal/Conference
The paper was published on **arXiv** (a preprint server) with the ID `2604.01002`. The date provided is April 1, 2026.
- **Venue Reputation:** arXiv is a reputable preprint server widely used in computer science, physics, and mathematics to share research before formal peer review. It allows for rapid dissemination of new ideas.
- **Publication Status:** The paper is currently a **preprint**. It has not yet been published in a peer-reviewed journal or conference proceedings (like CVPR, ICCV, NeurIPS, etc.).

## 1.4. Publication Year
The publication year listed is **2026**.

## 1.5. Abstract
The paper's research objective is to address the limitations of applying Multimodal Large Language Models (MLLMs) to long-form videos. The core issue is that MLLMs have limited context windows and high computational costs, making it infeasible to process every frame. Therefore, an efficient keyframe sampling method is essential.
- **Core Methodology:** The authors propose an "evidence-driven keyframe sampling framework" grounded in **Information Bottleneck Theory**. They formulate keyframe selection as maximizing the **conditional mutual information** between the selected frames and the query. To make this computationally tractable, they derive a decomposed optimization that reduces the complex subset selection problem to independent frame-level scoring. They introduce a "query-conditioned evidence scoring network" trained with a contrastive objective to estimate this importance efficiently.
- **Main Results:** Experiments on long-form video understanding benchmarks (LVBench, Video-MME) show that the proposed method consistently outperforms prior sampling strategies (like semantic matching or reinforcement learning) under strict token budgets.
- **Key Conclusions:** The method achieves significant accuracy gains (e.g., +10.1% on LVBench with Qwen2.5-VL-7B) while being substantially more training-efficient than RL-based methods.

## 1.6. Original Source Link
The official source link is: https://arxiv.org/abs/2604.01002
The PDF link is: https://arxiv.org/pdf/2604.01002
The publication status is **Preprint** on arXiv.

# 2. Executive Summary

## 2.1. Background & Motivation
- **Core Problem:** The paper aims to solve the problem of **efficiently processing long-form videos** using Multimodal Large Language Models (MLLMs). MLLMs are powerful but have a fixed context window (token limit) and high computational cost. Feeding every frame from a long video (e.g., hours long) into an MLLM is impossible. Therefore, we must select a small subset of "keyframes" that are most useful for answering a specific user query.
- **Importance & Challenges:** This is crucial because real-world video understanding often involves long content (movies, surveillance, educational videos). Existing approaches have flaws:
    1.  **Semantic Relevance Heuristics:** Methods that select frames similar to the query text often fail because "visual similarity" doesn't always mean "evidential usefulness." A frame might look like the query but not contain the specific evidence needed to answer the question.
    2.  **Reinforcement Learning (RL):** Methods that train an agent to select frames to maximize answer accuracy are powerful but suffer from **inefficient combinatorial optimization** (searching through all possible frame subsets is exponentially hard) and **sparse rewards** (the model only gets a reward at the very end if the answer is correct, making it hard to learn which specific frame was good or bad).
- **Entry Point/Innovative Idea:** The paper's innovative idea is to use **Information Bottleneck Theory** to provide a principled mathematical objective. Instead of heuristics or end-to-end RL, they frame the problem as: "Select the frames that maximize the mutual information with the answer, given the query." This shifts the focus from "what looks like the query" to "what reduces uncertainty about the answer."

## 2.2. Main Contributions / Findings
- **Primary Contributions:**
    1.  **Evidence Scoring Framework:** A new theoretical formulation for keyframe sampling based on maximizing conditional mutual information. This formulation is proven to be submodular, allowing for efficient approximation.
    2.  **Decomposed Optimization:** The authors show how to relax the complex subset selection problem into a simple, independent frame-level scoring problem using a modular upper bound. This makes the method highly efficient and parallelizable.
    3.  **Query-Conditioned Evidence Scoring Network:** A practical neural network architecture designed to estimate the evidential value of each frame relative to a query. It uses a contrastive learning objective to learn to rank frames by their usefulness.
- **Key Conclusions/Findings:**
    - The proposed method outperforms existing state-of-the-art keyframe sampling techniques (like AKS, FOCUS, TSPO) on benchmarks like LVBench and Video-MME.
    - It achieves these gains with significantly better training efficiency compared to RL-based methods (e.g., 0.6 hours vs. 78 hours).
    - The method effectively mitigates hallucinations in MLLMs by ensuring the input frames actually contain the necessary evidence.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must be familiar with several key concepts:

1.  **Multimodal Large Language Models (MLLMs):** These are models like GPT-4V, LLaVA, or Qwen-VL that can process and generate both text and images (or video frames). They typically consist of a vision encoder (to turn images into features) and a large language model (to process text and image features).
2.  **Keyframe Sampling:** The process of selecting a representative subset of frames from a video sequence. This is a compression technique. In the context of MLLMs, it's critical because the "context window" (the amount of input text/tokens the model can handle) is limited.
3.  **Information Bottleneck (IB) Principle:** This is a principle from information theory. The basic idea is that when processing a signal $X$ to predict a target $Y$, we want to find a compressed representation $T$ of $X$ that is as informative as possible about $Y$ but as simple as possible (maximizing `I(T; Y)` while minimizing `I(T; X)`). In this paper, the "bottleneck" is the limited set of keyframes $S$ we can select from the full video $V$.
4.  **Mutual Information (MI):** A measure of the dependence between two random variables. `I(X; Y)` measures how much knowing one variable tells us about the other. **Conditional Mutual Information**, $I(X(X; Y | Z)$, measures the dependence between $X$ and $Y$ given that we already know $Z$. The paper uses $I(S; O | Q)$, meaning "how much does the selected frame set $S$ tell us about the output $O$ (answer), given we already know the query $Q$?"
5.  **Submodularity:** A property of set functions. A function $F$ is submodular if it has "diminishing returns." Adding an element to a small set gives a bigger gain than adding it to a large set. Formally, for $A \subseteq B$, $F(A \cup \{x\}) - F(A) \geq F(B \cup \{x\}) - F(B)$. This is important because maximizing a monotone submodular function can be done efficiently with a simple greedy algorithm, guaranteeing a $(1 - 1/e)$ approximation.
6.  **Contrastive Learning:** A training paradigm where the model learns to distinguish between "similar" (positive) and "dissimilar" (negative) pairs of data. For example, in this paper, the model learns to give high scores to frames that are useful for answering a question (positives) and low scores to irrelevant frames (negatives).
7.  **Reinforcement Learning (RL):** A machine learning paradigm where an agent learns to make decisions by performing actions in an environment and receiving rewards. In video sampling, an "action" is selecting a frame, and the "reward" is whether the final answer is correct. The paper notes RL is inefficient here due to the huge search space.

## 3.2. Previous Works
The paper discusses two main categories of prior work:

1.  **Training-Free Semantic-Matching Methods:**
    - **Concept:** These methods use pre-trained vision-language models (like CLIP) to calculate the similarity between a frame's visual content and the text query. They then select frames with the highest similarity scores.
    - **Examples:** **AKS** (Adaptive Keyframe Sampling), **FOCUS**, **Q-Frame**.
    - **Critique:** The authors argue these methods fail because "semantic relevance" (visual similarity) $\neq$ "evidential usefulness." A frame might look like the query concept but not contain the specific evidence needed to answer the question (e.g., a query about a "goal" might match frames showing the goalpost, but the critical evidence is the moment the ball crosses the line).

2.  **Trainable Policies (including RL):**
    - **Concept:** These methods introduce learnable components to select frames. Some use RL agents trained to maximize the final answer accuracy.
    - **Examples:** **MLLM-Selector**, **FFS**, **TSPO** (Temporal Sampling Policy Optimization), **ReaSon** (Reinforced Causal Search).
    - **Critique:** The authors argue these methods suffer from **inefficient combinatorial optimization**. The search space of all possible frame subsets is exponential. RL agents struggle with this "credit assignment problem"—it's hard to know which specific frame led to the correct answer, making training slow and unstable.

## 3.3. Technological Evolution
The field has evolved from simple, uniform sampling (taking every $n$-th frame) to more sophisticated methods.
- **Early Stage:** Uniform sampling or random sampling. Simple but often misses critical moments.
- **Intermediate Stage:** Semantic matching using CLIP. Better, as it's query-aware, but still heuristic-based and not optimized for the final task.
- **Current Stage (RL-based):** End-to-end optimization. Powerful but computationally prohibitive and hard to train.
- **This Paper's Position:** This paper proposes a middle ground that is theoretically principled (like RL) but computationally efficient (like semantic matching). It uses information theory to define the objective and a clever mathematical relaxation to make it solvable as a simple scoring problem, avoiding the exponential search space of RL.

## 3.4. Differentiation Analysis
The core differences and innovations compared to prior work are:
- **vs. Semantic Matching:** Instead of just matching visual features to text features (CLIP similarity), this method explicitly optimizes for **evidential value**—how much a frame reduces uncertainty about the *answer*. It uses a contrastive loss to learn this, rather than relying on pre-trained similarity.
- **vs. RL Methods:** Instead of treating selection as a sequential decision process over a massive state space, this paper mathematically decomposes the problem into **independent frame scoring**. This removes the need for complex RL policies and makes training much faster (dense, frame-level supervision vs. sparse, answer-level supervision).

# 4. Methodology

## 4.1. Principles
The core principle of the method is to treat keyframe selection as an **Information Bottleneck** problem. The goal is to find the smallest possible subset of frames $S$ from the full video $V$ that maximizes the information about the answer $O$, given the query $Q$.
- **Theoretical Basis:** The objective is to maximize the **conditional mutual information** $I(S; O | Q)$. This means we want the selected frames $S$ to share as much information as possible with the correct answer $O$, assuming we already know the question $Q$.
- **Intuition:** If a frame is critical for answering the question, observing it should significantly reduce our uncertainty (entropy) about what the answer is. If a frame is irrelevant, observing it changes nothing. The method selects frames that provide the greatest "surprise reduction" or "information gain" regarding the answer.

## 4.2. Core Methodology In-depth (Layer by Layer)

The methodology can be broken down into three main layers: the theoretical formulation, the optimization strategy, and the practical implementation (the scoring network).

### Layer 1: Theoretical Formulation

1.  **Problem Definition:**
    - Input: A video $V = \{f_1, \dots, f_n\}$ (a sequence of $n$ frames) and a natural language query $Q$.
    - Goal: Find a subset of keyframes $S \subseteq V$ such that the size of $S$ is less than or equal to a budget $m$ ($|S| \le m$). This subset should allow an MLLM to answer $Q$ as accurately as if it had seen the whole video $V$.

2.  **Information Bottleneck Objective:**
    The authors formulate the selection as maximizing the conditional mutual information between the selected frames $S$ and the MLLM output (answer) $O$, given the query $Q$. The mathematical objective is:

    $$
    \operatorname*{max}_{S} I(S; O \mid Q) \quad \mathrm{s.t.} \quad |S| \leq m
    $$

    - $I(S; O \mid Q)$: This is the conditional mutual information. It quantifies how much knowing the selected frames $S$ tells us about the answer $O$, assuming we already know the query $Q$.
    - $m$: The maximum number of frames we are allowed to select (the token budget).

      To understand why this makes sense, the authors expand the mutual information term:

    $$
    I(S; O \mid Q) = \mathbb{E}_Q \left[ \mathbb{E}_{p(S, O \mid Q)} \left[ \log \frac{p(O \mid S, Q)}{p(O \mid Q)} \right] \right]
    $$

    - $\mathbb{E}$: The expectation (average) over the distribution of queries and frame-answer pairs.
    - $p(O \mid S, Q)$: The probability of the answer given the selected frames and the query. This is what the MLLM computes.
    - $p(O \mid Q)$: The probability of the answer given only the query (without seeing any frames). This is a prior probability.

      Since $p(O \mid Q)$ is a constant with respect to the selection $S$ (it doesn't change based on which frames we pick), maximizing the mutual information is equivalent to maximizing the log-probability of the correct answer:

    $$
    \operatorname*{max}_{S} \mathbb{E} [ \log p(O \mid S, Q) ]
    $$

    This confirms the intuition: the objective is to select the frames that make the correct answer most probable according to the MLLM.

### Layer 2: Optimization Strategy (From Theory to Practice)

Directly optimizing the objective $\operatorname*{max}_{S} I(S; O \mid Q)$ is intractable because there are $\binom{n}{m}$ possible ways to choose $m$ frames from $n$ total frames. This number grows exponentially with video length.

1.  **Submodularity and Greedy Selection:**
    The authors prove that the objective function $F(S) = I(S; O \mid Q)$ is **monotone submodular**.
    - **Monotone:** Adding more frames never decreases the mutual information (you can't lose information by seeing more).
    - **Submodular:** The gain from adding a frame diminishes as the set grows. Adding a frame to a small set gives more information than adding it to a large set that already has similar information.

      A classic result in optimization (Nemhauser et al.) states that for a monotone submodular function, a simple **greedy algorithm** achieves a $(1 - 1/e) \approx 63\%$ approximation of the optimal solution. The greedy algorithm would iteratively pick the frame that provides the largest marginal gain in mutual information.

    However, even the greedy algorithm requires $K$ passes over the $n$ frames, which is still expensive for very long videos.

2.  **Modular Upper Bound Relaxation:**
    To make the selection fully parallelizable and efficient, the authors use a **modular upper bound relaxation**. For any monotone submodular function $F$ with $F(\emptyset) = 0$, the sum of individual gains is an upper bound on the function value for any set:

    $$
    F(S) \leq \sum_{f_i \in S} F(\{f_i\}) = \sum_{f_i \in S} I(f_i; O \mid Q)
    $$

    - $F(\{f_i\}) = I(f_i; O \mid Q)$: This is the mutual information contributed by a single frame $f_i$ alone.
    - The inequality holds because of diminishing returns (submodularity). The marginal gain of adding a frame to a set is always less than or equal to the gain of adding it to an empty set.

      By relaxing the objective to this upper bound, the combinatorial subset selection problem is reduced to a simple **independent frame-level scoring problem**. We can now simply score every frame independently using $I(f_i; O \mid Q)$ and pick the top $m$ frames.

3.  **Temporally-Adaptive Selection:**
    The authors implement a "temporally-adaptive" strategy to balance global evidence with temporal coverage. They divide the video into $B$ bins (segments) and select the top $k$ frames from each bin. The total frames selected is $m = B \times k$.

    $$
    S^* = \bigcup_{b=1}^{B} \mathrm{top}{-k}_{f_i \in BIN_b} I(f_i; O \mid Q)
    $$

    - If `1`, they pick exactly one frame per bin (strict temporal diversity).
    - If $k=m$ (and $B=1$), they pick the global top $m$ frames (pure evidence-based, no temporal constraint).
    - This allows the method to adapt to different video types. For LVBench (long videos with localized evidence), they use global top-$m$. For VideoMME (requiring broader coverage), they use the diverse regime ($k=1$).

### Layer 3: Practical Implementation (The Scoring Network)

The quantity $I(f_i; O \mid Q)$ (mutual information of a single frame) is still not directly computable because it depends on the unknown answer $O$. The solution is to train a neural network to **estimate** this score.

1.  **Query-Conditioned Evidence Scoring Network ($g_\theta$):**
    The authors design a network $g_\theta(f_i, Q)$ that takes a frame $f_i$ and the query $Q$ and outputs a scalar score representing the frame's evidential value. The architecture consists of four main components:
    - **Vision-Language Encoder:** A shared encoder (like CLIP) processes the frame and the query to get feature vectors.
      $$
        \mathbf{v}_i = \mathcal{E}_\nu(f_i) \in \mathbb{R}^d, \quad \mathbf{q} = \mathcal{E}_t(Q) \in \mathbb{R}^d
        $$
        - $\mathcal{E}_\nu$: The vision encoder.
        - $\mathcal{E}_t$: The text encoder.
        - $\mathbf{v}_i, \mathbf{q}$: The resulting $d$-dimensional feature vectors for the frame and query.

    - **Temporal Evidence Aggregator:** To capture short-term temporal context (since a single frame might be ambiguous), the model uses a "causal window attention" mechanism. For a frame at position $\tau$, it looks at a small window of previous frames $\mathcal{W}_\tau$.
      $$
        \mathbf{h}_\tau = \mathcal{T} \big ( \mathbf{v}_\tau \mid \{ \mathbf{v}_j : f_j \in \mathcal{W}_\tau \} \big )
        $$
        - $\mathcal{T}$: The temporal aggregation function (e.g., a lightweight attention layer).
        - $\mathbf{h}_\tau$: The context-aware frame representation.

    - **Query-Guided Evidence Gating:** A gating mechanism filters the frame features based on the query. It acts like a valve, keeping only the channels (features) that are relevant to the query.
      $$
        \mathbf{g}_i = \sigma ( \mathbf{W}_h \mathbf{h}_i + \mathbf{W}_q \mathbf{q} + \mathbf{b} ), \quad \mathbf{u}_i = \mathbf{h}_i \odot \mathbf{g}_i
        $$
        - $\mathbf{W}_h, \mathbf{W}_q$: Learnable weight matrices.
        - $\sigma$: The sigmoid activation function.
        - $\mathbf{g}_i$: The gate vector (values between 0 and 1).
        - $\odot$: Element-wise multiplication.
        - $\mathbf{u}_i$: The gated, query-relevant frame representation.

    - **Evidence Score Head:** This head computes the final score. It decomposes the score into $K$ semantic subspaces to capture different aspects of alignment (e.g., object, action, scene).
      For the $k$-th subspace:
        $$
        s_{i,k} = \frac{1}{\gamma_k} \frac{ ( \mathbf{W}_\nu^{(k)} \mathbf{u}_i )^\top ( \mathbf{W}_q^{(k)} \mathbf{q} ) }{ \| \mathbf{W}_\nu^{(k)} \mathbf{u}_i \|_2 \| \mathbf{W}_q^{(k)} \mathbf{q} \|_2 }
        $$
        - $\mathbf{W}_\nu^{(k)}, \mathbf{W}_q^{(k)}$: Projection matrices for the $k$-th subspace.
        - The fraction is the cosine similarity between the projected frame and query vectors.
        - $\gamma_k$: A learnable scaling factor.

          The final score is a weighted sum of the subspace scores and a direct CLIP similarity term:
        $$
        g_\theta(f_i, Q) = \lambda \frac{ \mathbf{v}_i^\top \mathbf{q} }{ \| \mathbf{v}_i \|_2 \| \mathbf{q} \|_2 } + (1 - \lambda) \frac{1}{K} \sum_{k=1}^{K} s_{i,k}
        $$
        - $\lambda$: A learnable parameter balancing the global CLIP similarity and the fine-grained subspace scores.

          The following figure (Figure 2 from the original paper) illustrates the architecture of this scoring network:

          ![该图像是一个示意图，展示了基于查询条件的证据关键帧采样框架。框架利用图像编码器和时间证据聚合模块，通过查询引导的证据门控来评估关键帧的证据重要性，并通过 $E_{score}$ 输出评分，显示哪些关键帧对于回答问题最为重要。最终，经过加权聚合的关键帧被选为多模态大语言模型的输入。](images/2.jpg)
          *该图像是一个示意图，展示了基于查询条件的证据关键帧采样框架。框架利用图像编码器和时间证据聚合模块，通过查询引导的证据门控来评估关键帧的证据重要性，并通过 $E_{score}$ 输出评分，显示哪些关键帧对于回答问题最为重要。最终，经过加权聚合的关键帧被选为多模态大语言模型的输入。*

2.  **Training Objective:**
    The network is trained to predict the ranking of frames by their evidential value. The authors use a contrastive learning approach.
    - **Positive Frames ($\mathcal{F}^+$):** Frames that fall within annotated "evidence segments" in the training data. These are frames known to be sufficient for answering the query.
    - **Negative Frames ($\mathcal{F}^-$):** Frames outside these segments.

      The training objective is the **InfoNCE loss**, which maximizes the score of positive frames relative to negative frames.

    $$
    \mathcal{L} = - \log \frac{ \sum_{x \in \mathcal{F}^+} \exp g_\theta(x, Q) }{ \sum_{x \in \mathcal{F}^+} \exp g_\theta(x, Q) + \sum_{x \in \mathcal{F}^-} \exp g_\theta(x, Q) }
    $$

    - The numerator sums the exponentiated scores of positive frames.
    - The denominator sums the exponentiated scores of both positive and negative frames.
    - Minimizing this loss pushes the model to assign high scores to positives and low scores to negatives.

      The authors show that the optimal solution to this loss, $g_\theta^*(x, Q)$, is proportional to the log-ratio of the probabilities:
    $$
    g_\theta^*(x, Q) \propto \log \frac{p(x \mid \mathcal{F}^+, Q)}{p(x \mid \mathcal{F}^-, Q)} + C
    $$
    This confirms that the learned score captures how much more "evidential" a frame is compared to the background.

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize two main datasets for training and evaluation.

1.  **Training Dataset: LLaVA-Video subset of Seek-173K**
    - **Source:** The Seek-173K dataset.
    - **Characteristics:** This dataset provides annotated "evidence segments" for video-query pairs. These segments are the parts of the video that contain the necessary information to answer the query.
    - **Usage:** The authors use frames from within these segments as **positive samples** and frames outside as **negative samples** to train the evidence scoring network. This provides dense, frame-level supervision, which is a key advantage over RL methods that only get sparse answer-level rewards.

2.  **Evaluation Datasets:**
    - **LVBench (Long Video Benchmark):**
        - **Source:** Published in ICCV 2025.
        - **Characteristics:** A benchmark for extreme long video understanding. The average video duration is **4101 seconds** (over an hour). It contains questions that require retrieving specific information, understanding events, and reasoning over long temporal spans.
        - **Why chosen:** It is the most challenging benchmark for testing the method's ability to handle very long videos and find sparse evidence.
    - **Video-MME:**
        - **Source:** Published in CVPR 2025.
        - **Characteristics:** A comprehensive evaluation benchmark for MLLMs in video analysis. It includes both short and long videos.
        - **Why chosen:** It provides a standardized evaluation to compare against a wide range of existing MLLMs and sampling methods.

## 5.2. Evaluation Metrics
The primary evaluation metric used in the paper is **Accuracy**.

1.  **Question-Answering Accuracy:**
    - **Conceptual Definition:** This metric measures the percentage of questions for which the model's generated answer matches the ground-truth answer. It is the most direct measure of success for a video QA system. The paper focuses on whether the selected keyframes enable the MLLM to arrive at the correct answer.
    - **Mathematical Formula:**
      $$
        \text{Accuracy} = \frac{\text{Number of Correct Answers}}{\text{Total Number of Questions}}
        $$
    - **Symbol Explanation:**
        - $\text{Number of Correct Answers}$: The count of questions where the model's output exactly matches the ground truth.
        - $\text{Total Number of Questions}$: The total count of questions in the evaluation set.

          In the supplementary material, the authors also introduce a **Coverage Metric** to analyze hallucinations.
    - **Coverage:** The percentage of cases where the sampled frames overlap with the ground-truth evidence segment. This measures whether the sampling method successfully found the relevant part of the video.

## 5.3. Baselines
The paper compares the proposed method against several strong baselines, categorized as follows:

1.  **Agentic MLLMs:** Methods that treat the MLLM as an agent to iteratively query or process the video (e.g., VideoTree, VideoAgent, VCA).
2.  **SFT/RL MLLMs:** MLLMs that have been fine-tuned (SFT) or trained with RL specifically for video tasks (e.g., MovieChat, TimeMarker, VideoLLaMA3, Video-R1, Video-Thinker).
3.  **Keyframe Sampling for MLLMs:** This is the most relevant category for direct comparison.
    - **Uniform Sampling:** Selecting frames at fixed intervals.
    - **ReTaKe:** A method that reduces temporal and knowledge redundancy.
    - **TSPO:** A Reinforcement Learning-based method (Temporal Sampling Policy Optimization). This is a strong baseline representing the RL approach the paper critiques.
    - **FrameThinker:** A method that learns to think with long videos via multi-turn frame spotlighting.
    - **AKS, FOCUS, Q-Frame:** Semantic matching-based methods.

      These baselines are representative because they cover the spectrum of existing solutions: from simple heuristics (uniform) to semantic matching (AKS) to complex RL (TSPO). Comparing against them validates the claim that the proposed method offers a better balance of performance and efficiency.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main experimental results strongly validate the effectiveness of the proposed method.

1.  **Performance on LVBench:**
    - The proposed method ("Ours") consistently outperforms other keyframe sampling methods under the same MLLM backbone and frame budget.
    - For example, with **Qwen2.5-VL-7B** and a budget of 32 frames, the proposed method achieves **47.7%** accuracy. This is significantly higher than the baseline Qwen2.5-VL-7B with uniform sampling (37.6%) and the FrameThinker method (36.6%).
    - It even achieves performance comparable to methods that use many more frames (e.g., ReTaKe uses $\le 2048$ frames and gets 47.8%, while "Ours" uses only 32 frames to get 47.7%). This highlights the **efficiency** of the method—it extracts the same information with far fewer frames.
    - The method shows particular strength in "Key Information Retrieval" (KIR), improving from 40.6% (baseline) to 57.0%. This confirms its ability to find specific, sparse evidence.

2.  **Performance on Video-MME:**
    - On Video-MME, the method also shows consistent gains.
    - With **Qwen2.5-VL-7B**, the average accuracy improves from 60.7% (uniform) to **63.6%**.
    - The improvement is even more pronounced on the **Long Video subset** of Video-MME, jumping from 50.6% to **55.0%**. This reinforces the conclusion that the method is particularly well-suited for long-form content.

3.  **Efficiency Analysis:**
    - The results demonstrate a massive improvement in training efficiency compared to RL-based methods.
    - **TSPO (RL-based):** Requires 78 hours to train on 10K samples using 8 NVIDIA L40 GPUs.
    - **Proposed Method:** Completes training in **0.6 hours** on 264K samples using the same hardware.
    - This >100x speedup is attributed to the decomposed optimization (independent scoring) which avoids the expensive combinatorial search and MLLM-in-the-loop training required by RL.

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
      <th></th>
      <th>Rea Sum</th>
      </tr>
      </thead>
      <tbody>
      <tr>
      <td colspan="10"><strong>Agentic MLLMs</strong></td>
      </tr>
      <tr>
      <td>VideoTree [31]</td>
      <td></td>
      <td>28.8</td>
      <td>30.3</td>
      <td></td>
      <td>25.1 26.5</td>
      <td></td>
      <td></td>
      <td>27.7 31.9 25.5</td>
      </tr>
      <tr>
      <td>VideoAgent [32]</td>
      <td></td>
      <td>29.3</td>
      <td>28.0</td>
      <td>30.3 28.0</td>
      <td></td>
      <td></td>
      <td></td>
      <td>29.3 28.0 36.4</td>
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
      <td colspan="10"><strong>SFT/RL MLLMs</strong></td>
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
      <td>21.3 23.1 25.9 22.3 24.0 17.22</td>
      </tr>
      <tr>
      <td>TimeMarker-8B [35]</td>
      <td>≤128</td>
      <td>41.3</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>42.8 39.1 34.9 38.7 38.2 48.8</td>
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
      <td>≤ 768</td>
      <td>47.6</td>
      <td></td>
      <td></td>
      <td></