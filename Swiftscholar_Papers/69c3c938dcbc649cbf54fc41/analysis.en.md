# 1. Bibliographic Information
## 1.1. Title
The paper is titled *SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning*. Its central topic is developing a novel agentic-level speculative acceleration framework to reduce the latency and improve the serving throughput of agentic multimodal large language models (MLLMs), while preserving or even improving their reasoning accuracy.
## 1.2. Authors
The authors and their affiliations are:
- Haoyu Huang, Xiawu Zheng, Rongrong Ji: Xiamen University (China)
- Jinfa Huang, Jiebo Luo: University of Rochester (USA)
- Zhongwei Wan: The Ohio State University (USA)
  Haoyu Huang and Jinfa Huang contributed equally to this work. The research team spans leading institutions in computer vision and multimodal learning, with expertise in efficient LLM serving, agentic reasoning, and multimodal model design.
## 1.3. Publication Venue & Status
As of the current date (2026-03-25 UTC), the paper is published as a preprint on arXiv, and has not yet been officially accepted or presented at a peer-reviewed conference or journal.
## 1.4. Publication Year
The preprint was published on 2026-03-24 UTC.
## 1.5. Abstract
This paper addresses the critical efficiency bottleneck of agentic MLLMs, which suffer from high latency and low concurrency due to sequential cascaded loops of perception, reasoning, and visual tool invocation (a property termed *agentic depth*). The proposed SpecEyes framework uses a lightweight, tool-free small MLLM as a speculative planner to predict execution trajectories, allowing early termination of expensive tool chains for queries that do not require multi-step tool assistance. A novel cognitive gating mechanism based on *answer separability* quantifies the small model's confidence without requiring oracle labels, to regulate speculative planning. A heterogeneous parallel funnel architecture exploits the stateless concurrency of the small model to mask the serial execution of the large agentic model, maximizing system throughput. Experiments on V* Bench, HR-Bench, and POPE show that SpecEyes achieves 1.1–3.35× speedup over agentic baselines, while improving accuracy by up to 6.7% and boosting serving throughput under concurrent workloads.
## 1.6. Original Source Links
- Abstract page: https://arxiv.org/abs/2603.23483v1
- Full PDF: https://arxiv.org/pdf/2603.23483v1
- Official code repository: github.com/MAC-AutoML/SpecEyes

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Agentic MLLMs (e.g., OpenAI o3, Gemini Agentic Vision) achieve state-of-the-art performance on complex visual reasoning tasks by iteratively invoking external visual tools (e.g., zoom-in, crop, optical character recognition (OCR)). However, each tool call step depends on the output of the previous step, creating strict sequential dependencies. This *agentic depth* bottleneck causes two critical system issues:
1.  **Latency explosion**: End-to-end response time scales linearly with the number of tool call steps per query.
2.  **Concurrency collapse**: Each query's tool use chain maintains a per-query mutable state, making GPU batching ineffective and leaving massive hardware parallelism idle.
### Gaps in Prior Research
All existing efficiency methods operate *inside* the agentic loop, and do not address the core sequential bottleneck:
- Token-level speculative decoding accelerates individual generation steps, but still follows the full serial agentic trajectory, requiring every tool to be invoked in sequence, and often introduces extra interaction overhead.
- Multimodal token pruning, compression, and KV-cache optimization reduce per-step compute cost, but do not eliminate repeated tool invocations that dominate agentic latency.
### Innovative Entry Point
The core insight of this work is that a large fraction of queries sent to agentic MLLMs do not actually require tool-assisted reasoning. A lightweight, tool-free MLLM can answer these queries correctly directly from the original input image, if we can reliably identify such queries. This shifts the speculative paradigm from the token/step level to the full agentic pipeline level.
## 2.2. Main Contributions / Findings
### Primary Contributions
1.  **Formalization of agentic bottleneck**: The paper first formalizes the stateful dependency bottleneck of agentic MLLMs, proving that the sequential nature of tool use chains imposes fundamental limits on both per-query latency and system-level concurrency.
2.  **SpecEyes framework**: The first agentic-level speculative acceleration framework, which bypasses entire tool-use loops for queries that do not require them, preserving full reasoning accuracy.
3.  **Answer separability cognitive gate**: A label-free, scale-invariant confidence metric that allows the small model to decide when to trust its own output, or escalate the query to the large agentic model.
4.  **Heterogeneous parallel funnel architecture**: A serving design that exploits the stateless nature of the small model to achieve concurrent query processing, yielding throughput gains proportional to the speculative acceptance rate.
### Key Findings
- SpecEyes achieves 1.1–3.35× end-to-end speedup over state-of-the-art agentic MLLM baselines across three diverse benchmarks, while improving average accuracy by up to 6.7%, as it eliminates hallucination errors introduced by unnecessary tool invocations.
- Average screening rate (fraction of queries identified as tool-free) across benchmarks is 80%, and average acceptance rate of speculative answers is 71%, leading to average throughput gains of ~1.7× over baselines.
- The framework generalizes across two different agentic MLLM backbones (DeepEyes and Thyme), showing robust performance regardless of the underlying agentic model design.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
We define all core terms required to understand this work, for a beginner audience:
1.  **Multimodal Large Language Model (MLLM)**: A type of large language model that can process both text and visual inputs (images, video) to generate text responses, combining capabilities of computer vision and natural language processing.
2.  **Agentic MLLM**: An advanced MLLM that can actively invoke external tools (e.g., zoom, crop, OCR for visual inputs, search engines for text inputs) in iterative loops to solve complex tasks, instead of generating responses in a single forward pass. It maintains a mutable state that updates after each tool call to refine its understanding of the task.
3.  **Agentic depth**: The number of sequential tool invocation and reasoning steps an agentic MLLM executes to answer a single query.
4.  **Token-level speculative decoding**: An existing efficiency method for LLMs, where a small "draft" model generates a sequence of candidate tokens quickly, and a large target model verifies all candidate tokens in a single forward pass. Accepted tokens are returned directly, while rejected tokens are regenerated by the large model. This reduces latency without accuracy loss, but only operates at the individual token level, and does not change the overall reasoning trajectory of the model.
5.  **Logit**: The raw unnormalized output score of a neural network for each possible token in the vocabulary, before applying the softmax function to convert scores to probabilities.
6.  **POPE (Polling-based Object Probing Evaluation)**: A standard benchmark for evaluating object hallucination in MLLMs, where models are asked yes/no questions about the presence of objects in images, to test if they generate non-existent objects (hallucinations).
## 3.2. Previous Works
We summarize the three lines of related work most relevant to this paper:
### 1. Agentic Multimodal LLMs
Agentic reasoning in MLLMs evolved from tool-augmented text LLM frameworks like ReAct [60] and Toolformer [39], which interleaved action generation and external feedback. Recent agentic MLLMs like DeepEyes [67] and Thyme [63] extend this paradigm to visual tasks, using reinforcement learning or code generation to invoke visual tools iteratively for fine-grained perception and multi-step reasoning. While these models achieve state-of-the-art accuracy, their fully sequential tool use loops impose extreme latency overhead, a gap unaddressed by prior work.
### 2. Speculative Decoding for Efficient Reasoning
Token-level speculative decoding [23] was first proposed to accelerate text LLM generation, and has been extended to collaborative reasoning settings:
- SpecReason [37] delegates simpler reasoning steps to a lightweight model, verified via semantic consistency.
- RelayLLM [18] dynamically invokes a stronger expert model only at critical reasoning steps.
  All these methods accelerate individual steps *within* a fixed reasoning trajectory, but the agentic loop itself remains fully serial, and they introduce extra turn and token verification overhead that often offsets speed gains in agentic settings.
### 3. Efficient Multimodal Perception
A large body of work reduces per-step compute cost of MLLMs via techniques like visual token pruning [10], token merging [3], frequency-domain compression [49], and KV-cache compression [47]. These methods optimize individual forward passes of the model, but leave the sequential agentic tool use pipeline fully intact, so they cannot eliminate the dominant latency source of repeated tool invocations.
## 3.3. Technological Evolution
The development of MLLMs has followed two parallel trajectories that converge in this work:
1.  **Capability evolution**: MLLMs evolved from static single-pass visual encoding (e.g., BLIP-2 [24], Flamingo [1]) to dynamic agentic tool use (e.g., DeepEyes [67], Thyme [63]), dramatically improving accuracy on complex tasks but introducing severe efficiency bottlenecks.
2.  **Efficiency evolution**: Speculative acceleration evolved from token-level (accelerating individual token generation) to step-level (delegating individual reasoning steps to small models), but no prior work lifted this paradigm to the full agentic pipeline level, to skip entire tool use loops entirely.
    This paper's work sits at the intersection of these two trajectories, addressing the efficiency gap of state-of-the-art agentic MLLMs via a novel agentic-level speculative design.
## 3.4. Differentiation Analysis
The core difference between SpecEyes and all prior methods is that SpecEyes operates *outside* the agentic loop, rather than optimizing steps inside the loop:
- All prior efficiency methods preserve the full sequential agentic tool use pipeline, only reducing the cost of individual steps within the pipeline.
- SpecEyes identifies queries that do not require any tool use, and bypasses the entire agentic loop for these queries, eliminating all sequential tool invocation overhead for a large fraction of inputs.
  This paradigm shift enables order-of-magnitude higher speed gains than prior methods, without accuracy loss.

# 4. Methodology
## 4.1. Core Principles & Intuition
The core intuition behind SpecEyes is a "think fast, think slow" heterogeneous design:
- **Fast thinking**: A lightweight, stateless, tool-free small MLLM ($\mathcal{M}_S$) rapidly generates speculative answers for queries that do not require tool assistance.
- **Slow thinking**: A large, stateful agentic MLLM ($\mathcal{M}_L$) is reserved only for queries that genuinely require multi-step tool-assisted reasoning.
  A reliable confidence gating mechanism ensures that only high-confidence answers from the small model are accepted, while low-confidence queries are routed to the large agentic model as a safety net, preserving full accuracy.
The following figure (Figure 1 from the original paper) illustrates the motivation and high-level overview of SpecEyes:

![Fig. 1. Motivation and overview of SpecEyes. Top: Agentic MLLMs evaluate each query via a Markovian sequence of stateful tool invocations of depth $D$ . This strict causal dependency prohibits parallelization, imposing a serving complexity of $\\mathcal { O } ( B D C )$ for $B$ queries, where $C$ denotes the tool per-step inference cost. Bottom: SpecEyes enables agentic-evel speculative bypass with a stateless small model and an answer-separability gate. Here, $\\beta$ is the fraction of tool-free candidates after screening (Sec. 3.4) and $\\alpha$ is the acceptance rate of speculative answers among them (Secs. 3.2 and 3.3), averaging $8 0 \\%$ and $7 1 \\%$ across all benchmarks, respectively. All reported accuracy and speedup values are averaged across V $^ *$ \[52\], HR-Bench \[50\], and POPE \[26\].](images/1.jpg)
*该图像是一个示意图，展示了Agentic MLLMs的状态瓶颈及SpecEyes框架的工作原理。图中上半部分展示了Agentic MLLMs在深度$D$的状态序列中具有的依赖关系，导致服务复杂度为$\mathcal{O}(BDC)$，同时显示了其准确率和加速比。下半部分则描述了SpecEyes如何通过无状态小模型和并行处理来提高效率，改善准确率和加速比，复杂度降低为$\mathcal{O}((1 - \beta\alpha)BDC)$。*

## 4.2. Step-by-Step Deconstruction of SpecEyes Framework
We first formalize the agentic MLLM bottleneck, then deconstruct each component of SpecEyes in order, integrating all mathematical formulas exactly as presented in the original paper, with full symbol explanations.
### 4.2.1. Formalization of the Agentic MLLM Stateful Bottleneck
We first formalize an agentic MLLM as a stateful reasoning system:
$$\mathcal{A} = (\mathcal{S}, \mathcal{T}, \pi)$$
Where:
- $\mathcal{S}$ = the state space of the agent, storing all information about the query, input image, and previous tool use observations.
- $\mathcal{T} = \{t_1, t_2, ..., t_N\}$ = a finite set of available perception tools (e.g., Zoom-in, Crop, OCR).
- $\pi$ = the agent's policy, which jointly selects tool invocations and generates reasoning tokens given the current state.

  Given a query $q$ and input image $I$, the agent maintains a state trajectory $\{s_0, s_1, ..., s_D\}$ over $D$ reasoning steps (where $D$ is the agentic depth of the query). The initial state is $s_0 = (q, I)$. At each step $d$, the policy produces an action `a_d = \pi(s_d)` that either invokes a tool $t \in \mathcal{T}$ or emits a final answer. When a tool is invoked, the state transitions as follows:
$s_{d+1} = f(s_d, t_d(s_d))$
Where:
- $t_d(s_d)$ = the output of applying the selected tool $t_d$ to the current visual context in state $s_d$ (e.g., cropping a region of interest from the input image).
- $f$ = a fusion function that combines the tool output observation into the next state $s_{d+1}$.

  A critical property of this state transition is strict causal dependency: the tool selected at step $d+1$ depends entirely on the observation from step $d$, so step $d+1$ cannot begin until step $d$ completes:
$$
p(a_{d+1} | s_0, a_0, ..., s_d) = p(a_{d+1} | s_d, t_d(s_d)) \neq p(a_{d+1} | s_0)
$$
This sequential dependency leads to linear scaling of per-query latency with agentic depth:
$$
L_{\mathrm{agent}}(q) = \sum_{d=0}^{D(q)} \left( \underbrace{c_{\mathrm{llm}}}_{\mathrm{reasoning}} + \underbrace{c_{\mathrm{tool}}(t_d)}_{\mathrm{perception}} \right)
$$
Where:
- $c_{\mathrm{llm}}$ = latency of LLM reasoning inference per step.
- $c_{\mathrm{tool}}(t_d)$ = latency of executing tool $t_d$ at step $d$.

  At the system level, this serialization limits concurrency. For a batch of $B$ queries, the large agentic model can only process one tool-use step per query at a time, so maximum throughput is bounded by:
$$
\Theta_{\mathrm{agent}} \leq \frac{B}{\sum_{i=1}^B L_{\mathrm{agent}}(q_i)}
$$
This bound becomes increasingly restrictive as average agentic depth grows.
### 4.2.2. Four-Phase Speculative Pipeline
SpecEyes uses a four-phase funnel pipeline to route queries to either the small speculative model or the large agentic model, as illustrated in Figure 2 from the original paper:

![Fig. 2. Pipeline overview of SpecEyes. A batch of $B$ queries passes through a four-phase funnel. I: $\\mathcal { M } _ { L }$ screens tool necessity, splitting queries into tool-free and tool-required. II: A stateless $\\mathcal { M } _ { S }$ speculatively answers all tool-free queries with token-level logits. III: An answer separability score $S _ { \\mathrm { s e p } }$ gates each answer; those above $\\tau$ are accepted directly. IV: Remaining queries fall back to the full agentic loop. The funnel yields ${ \\approx } 1 / ( 1 { - } \\beta \\alpha ) \\times$ throughput speedup.](images/2.jpg)
*该图像是示意图，展示了SpecEyes的处理流程。图中描述了一个批量查询（B queries）的四个阶段：第一阶段为工具使用判断，第二阶段进行投机预测，第三阶段引入认知门控机制，最后则是回退至完整的智能代理环节。每个阶段的操作和关系通过图形和箭头清晰呈现，以提高系统的吞吐量。*

We explain each phase in detail:
#### Phase I: Heuristic Tool-Use Judgment
Given a query $q$ and image $I$, the large agentic model $\mathcal{M}_L$ first runs a lightweight binary classification to determine if tool invocation is necessary:
$$
g(q, I) = \mathcal{M}_L(q, I; \mathcal{P}_{\mathrm{judge}}) \in \{0, 1\}
$$
Where:
- $\mathcal{P}_{\mathrm{judge}}$ = a prompt instructing the model to assess if the query can be answered from the global image alone without tools.
- $g=0$ = query is judged to be tool-free, proceeds to Phase II.
- $g=1$ = query requires tool assistance, immediately forwarded to Phase IV (agentic fallback).
  This phase incurs negligible overhead, as it only generates a single binary token with no tool invocation. The large model is used for this judgment because its tool-calling capability makes it more reliable at identifying when tools are needed than the small model.
#### Phase II: Speculative Prediction
For queries with $g=0$ (tool-free), the small non-agentic model $\mathcal{M}_S$ generates a speculative answer along with full output logit distributions for each generated token:
$$
\hat{y}_S, \{\ell^{(n)}\}_{n=1}^{|\hat{y}_S|} = \mathcal{M}_S(q, I)
$$
Where:
- $\hat{y}_S$ = the speculative answer generated by $\mathcal{M}_S$.
- $\ell^{(n)} \in \mathbb{R}^{|\mathcal{V}|}$ = the logit vector over the full vocabulary $\mathcal{V}$ for the $n$-th generated token in $\hat{y}_S$.
  This inference is fully stateless, requires no tool execution, and can be parallelized across all queries in a batch.
#### Phase III: Small MLLM Confidence Switching
A cognitive gating function computes a confidence score for the speculative answer $\hat{y}_S$, and makes a decision:
$$
\mathrm{decision} = \left\{
\begin{array}{ll}
\mathrm{accept}~\hat{y}_S, & \mathrm{if}~S_{\mathrm{sep}}(\hat{y}_S) \geq \tau \\
\mathrm{fallback~to}~\mathcal{M}_L, & \mathrm{if}~S_{\mathrm{sep}}(\hat{y}_S) < \tau
\end{array}
\right.
$$
Where:
- $S_{\mathrm{sep}}$ = the answer separability confidence score (detailed in Section 4.2.3).
- $\tau$ = a threshold calibrated on a small held-out validation set.
  Accepted answers are returned immediately, completely bypassing the agentic pipeline. Rejected queries proceed to Phase IV.
#### Phase IV: Agentic Fallback
Queries that fail confidence switching or are judged to require tools are routed to the full agentic model, which executes the complete stateful perception-reasoning loop:
$$
\hat{y}_L = \mathcal{M}_L(q, I) = \pi\left(s_0 \xrightarrow{t_0} s_1 \xrightarrow{t_1} \cdot\cdot\cdot \xrightarrow{t_{D-1}} s_D\right)
$$
This phase retains full access to all tools, and serves as a safety net to preserve the accuracy of the agentic baseline.
The expected per-query latency of the full SpecEyes pipeline is:
$$
\mathbb{E}[L_{\mathrm{SpecEyes}}] = c_J + \beta c_S + (1 - \beta\alpha) L_{\mathrm{agent}}
$$
Where:
- $c_J$ = latency of Phase I tool necessity judgment.
- $\beta$ = fraction of queries screened as tool-free in Phase I.
- $c_S$ = latency of Phase II small model inference.
- $\alpha$ = fraction of tool-free queries accepted by the cognitive gate in Phase III.
- $c_J + \beta c_S \ll L_{\mathrm{agent}}$, so when $\beta\alpha$ is large (e.g., >0.6), latency is dominated by the lightweight front-end, delivering substantial speedups.
### 4.2.3. Cognitive Gating via Answer Separability
The effectiveness of SpecEyes depends entirely on the quality of the confidence gating mechanism. The paper proposes a novel answer separability score that addresses critical limitations of standard probability-based confidence metrics.
#### Limitations of Probability-Based Confidence
A standard probability-based confidence metric aggregates per-token maximum softmax probabilities via geometric mean:
$$
p_{\mathrm{max}}^{(n)} = \max_{v \in \mathcal{V}} \sigma(\ell^{(n)})_v
$$
$$
S_{\mathrm{log}}(\hat{y}_S) = \exp\left( \frac{1}{|\hat{y}_S|} \sum_{n=1}^{|\hat{y}_S|} \log p_{\mathrm{max}}^{(n)} \right)
$$
Where $\sigma(\cdot)$ is the softmax function. This metric has two critical flaws:
1.  Softmax probabilities are often miscalibrated: large logit magnitudes can lead to overconfident probabilities even when the model is uncertain.
2.  The geometric mean does not explicitly measure how well the top prediction is separated from competing candidates, and can be inflated by high-confidence low-entropy tokens (e.g., punctuation, formatting tokens).
#### Answer Separability Score
The proposed answer separability score measures how clearly the top logit is separated from its nearest competitors, avoiding the calibration issues of softmax probabilities. For the $n$-th generated token, we first sort logits in descending order: $\ell_{[1]}^{(n)} \geq \ell_{[2]}^{(n)} \geq ... \geq \ell_{[|\mathcal{V}|]}^{(n)}$. The token-level separability score is:
$$
S_{\mathrm{sep}}^{(n)} = \frac{\ell_{[1]}^{(n)} - \mu_K^{(n)}}{\sigma_K^{(n)} + \epsilon}
$$
Where:
- $\ell_{[1]}^{(n)}$ = the largest logit for the $n$-th token.
- $\mu_K^{(n)}$ = mean of the top $K$ logits for the $n$-th token.
- $\sigma_K^{(n)}$ = standard deviation of the top $K$ logits for the $n$-th token.
- $\epsilon = 10^{-6}$ = small constant to avoid division by zero.
  This metric is scale-invariant (both numerator and denominator scale linearly with logit magnitude), neutralizing softmax calibration artifacts, and explicitly models the competitive landscape among top candidates.
#### Aggregation to Answer-Level Confidence
Token-level separability scores are aggregated across all generated tokens to get an answer-level confidence score. The paper tests three aggregation strategies:
$$
S_{\mathrm{sep}}^{\mathrm{mean}} = \frac{1}{|\hat{y}_S|} \sum_{n=1}^{|\hat{y}_S|} S_{\mathrm{sep}}^{(n)}, \quad S_{\mathrm{sep}}^{\mathrm{min}} = \min_{n \in [|\hat{y}_S|]} S_{\mathrm{sep}}^{(n)}, \quad S_{\mathrm{sep}}^{\mathrm{bottom}} = \frac{1}{|\mathcal{B}|} \sum_{n \in \mathcal{B}} S_{\mathrm{sep}}^{(n)}
$$
Where $\mathcal{B}$ is the set of tokens with the bottom $r$ fraction of $S_{\mathrm{sep}}^{(n)}$ values, with $r=0.2$ as default.
The paper adopts the min aggregation as default, justified by the following proposition:
> **Proposition 1**: Let $\hat{y}_S = (y_1, ..., y_{|\hat{y}_S|})$ be the speculative answer, and $\mathcal{E} = \bigcup_n \mathcal{E}_n$ be the answer-level error event, where $\mathcal{E}_n$ is the event that token $y_n$ is incorrect. Then:
> $$P(\mathcal{E}) = P\left(\bigcup_n \mathcal{E}_n\right) \leq \sum_n P(\mathcal{E}_n)$$
> If each $P(\mathcal{E}_n)$ is monotonically decreasing in $S_{\mathrm{sep}}^{(n)}$, then thresholding on $\min_n S_{\mathrm{sep}}^{(n)}$ ensures every token exceeds the confidence threshold, bounding the union error probability $P(\mathcal{E})$ most tightly.
The min aggregation acts as a worst-case guard, triggering fallback whenever any token in the answer exhibits weak separability, prioritizing avoidance of false acceptances to preserve accuracy.
### 4.2.4. Heterogeneous Parallelism for Throughput Acceleration
SpecEyes organizes the four phases into a heterogeneous parallel funnel to maximize system throughput:
1.  **Batch-parallel front-end**: Both Phase I (tool judgment) and Phase II (speculative inference) are stateless single-turn forward passes, so they can be fully parallelized across a batch of $B$ queries, with a fixed front-end cost of $c_J + c_S$ regardless of batch size.
2.  **Funnel-shaped serving**: Accepted queries ($\alpha\beta B$) are returned immediately, while only the residual set of queries $\mathcal{R} = (1-\beta\alpha)B$ (tool-required and gate-rejected) are sent to the sequential agentic execution stage.
    For practical batch sizes, the total batch processing time is dominated by the agentic fallback stage, so throughput speedup over the baseline is:
$$
\Theta_{\mathrm{SpecEyes}} / \Theta_{\mathrm{agent}} \approx 1/(1 - \beta\alpha)
$$
Throughput gains are directly proportional to the product of the tool-free screening rate $\beta$ and gate acceptance rate $\alpha$.

# 5. Experimental Setup
## 5.1. Datasets
The paper evaluates SpecEyes on three diverse multimodal benchmarks spanning different task types, to test generalizability:
1.  **V* Bench [52]**: A benchmark for guided visual search, with two multiple-choice subsets:
    - Direct Attributes: 115 questions testing attribute recognition (e.g., "What color is the car?").
    - Relative Position: 76 questions testing spatial reasoning (e.g., "Is the cat to the left of the dog?").
2.  **HR-Bench [50]**: A high-resolution image understanding benchmark, with two subsets:
    - 4K subset: 800 questions on 4K resolution images.
    - 8K subset: 800 questions on 8K resolution images, requiring fine-grained visual inspection.
3.  **POPE [26]**: A benchmark for evaluating object hallucination in MLLMs, with three yes/no question subsets:
    - Adversarial: 3000 questions designed to trigger hallucinations.
    - Popular: 3000 questions about common objects.
    - Random: 3000 questions about randomly selected objects.
      These datasets are chosen because they cover the full range of tasks agentic MLLMs are used for: fine-grained perception, spatial reasoning, high-resolution understanding, and hallucination robustness, making them ideal for validating the performance of SpecEyes across diverse scenarios.
## 5.2. Evaluation Metrics
The paper uses two core evaluation metrics, defined below:
### 1. Accuracy (Acc.)
**Conceptual Definition**: The fraction of queries answered correctly by the model, measuring reasoning performance.
**Mathematical Formula**:
$$
\mathrm{Acc} = \frac{N_{\mathrm{correct}}}{N_{\mathrm{total}}}
$$
**Symbol Explanation**:
- $N_{\mathrm{correct}}$ = number of queries answered correctly.
- $N_{\mathrm{total}}$ = total number of queries in the evaluation split.
### 2. Speedup (Spd.)
**Conceptual Definition**: The ratio of the baseline agentic model's end-to-end latency to SpecEyes' end-to-end latency, measuring efficiency gains. A speedup >1 means SpecEyes is faster than the baseline.
**Mathematical Formula**:
$$
\mathrm{Spd} = \frac{\bar{L}_{\mathrm{agent}}}{\bar{L}_{\mathrm{SpecEyes}}}
$$
**Symbol Explanation**:
- $\bar{L}_{\mathrm{agent}}$ = average per-query end-to-end latency of the agentic baseline model.
- $\bar{L}_{\mathrm{SpecEyes}}$ = average per-query end-to-end latency of the SpecEyes framework.
## 5.3. Baselines
The paper compares SpecEyes against four representative baselines:
1.  **Agentic baselines**: DeepEyes [67] and Thyme [63], two state-of-the-art agentic MLLMs, capped at 5 tool use steps per query. These are the primary baselines to compare speed and accuracy against.
2.  **SpecReason [37]**: A state-of-the-art step-level speculative reasoning method for LLMs, to compare against existing speculative acceleration approaches.
3.  **Qwen3-VL-2B (draft only)**: The small tool-free MLLM used as the speculative model in SpecEyes, run without any fallback to the large agentic model. This establishes an upper bound on possible speedup, and illustrates the accuracy cost of using only the small model.

    All experiments are run on a single NVIDIA A100 40GB GPU, with greedy decoding (temperature = 0), and latency measurements include tool execution time.
# 6. Results & Analysis
## 6.1. Core Results Analysis
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="3">Method</th>
<th colspan="4">V*</th>
<th colspan="4">HR-Bench</th>
<th colspan="6">POPE</th>
<th colspan="2" rowspan="2">Avg.</th>
</tr>
<tr>
<th colspan="2">Attr.</th>
<th colspan="2">Pos.</th>
<th colspan="2">4K</th>
<th colspan="2">8K</th>
<th colspan="2">Adv.</th>
<th colspan="2">Pop.</th>
<th colspan="2">Rand.</th>
</tr>
<tr>
<th>Acc.</th>
<th>Spd.</th>
<th>Acc.</th>
<th>Spd.</th>
<th>Acc.</th>
<th>Spd.</th>
<th>Acc.</th>
<th>Spd.</th>
<th>Acc.</th>
<th>Spd.</th>
<th>Acc.</th>
<th>Spd.</th>
<th>Acc.</th>
<th>Spd.</th>
<th>Acc.</th>
<th>Spd.</th>
</tr>
</thead>
<tbody>
<tr>
<td>Qwen3-VL-2B (draft only)</td>
<td>77.39</td>
<td>5.44×</td>
<td>82.89</td>
<td>5.31×</td>
<td>71.38</td>
<td>3.20×</td>
<td>68.00</td>
<td>2.90×</td>
<td>82.56</td>
<td>4.20×</td>
<td>83.80</td>
<td>3.78×</td>
<td>86.47</td>
<td>4.07×</td>
<td>78.93</td>
<td>4.13×</td>
</tr>
<tr>
<td colspan="18"><em>Based on DeepEyes [67]</em></td>
</tr>
<tr>
<td>DeepEyes [67]</td>
<td>90.43</td>
<td>1.00×</td>
<td>82.89</td>
<td>1.00×</td>
<td>75.85</td>
<td>1.00×</td>
<td>71.43</td>
<td>1.00×</td>
<td>78.43</td>
<td>1.00×</td>
<td>81.90</td>
<td>1.00×</td>
<td>88.83</td>
<td>1.00×</td>
<td>81.39</td>
<td>1.00×</td>
</tr>
<tr>
<td>SpecReason [37]</td>
<td>80.19</td>
<td>0.61×</td>
<td>73.91</td>
<td>0.38×</td>
<td>80.43</td>
<td>0.44×</td>
<td>72.54</td>
<td>0.42×</td>
<td>49.10</td>
<td>0.38×</td>
<td>51.55</td>
<td>0.38×</td>
<td>60.20</td>
<td>0.37×</td>
<td>66.85</td>
<td>0.43×</td>
</tr>
<tr>
<td>SpecEyes (log)</td>
<td>83.48</td>
<td>2.06×</td>
<td>88.16</td>
<td>2.05×</td>
<td>73.71</td>
<td>1.35×</td>
<td>69.67</td>
<td>1.28×</td>
<td>83.97</td>
<td>1.89×</td>
<td>86.70</td>
<td>1.95×</td>
<td>90.50</td>
<td>2.05×</td>
<td>82.31</td>
<td>1.80×</td>
</tr>
<tr>
<td>SpecEyes (mean)</td>
<td>78.26</td>
<td>2.89×</td>
<td>84.21</td>
<td>3.35×</td>
<td>71.62</td>
<td>1.88×</td>
<td>67.38</td>
<td>1.77×</td>
<td>85.13</td>
<td>2.06×</td>
<td>87.00</td>
<td>2.10×</td>
<td>90.13</td>
<td>2.14×</td>
<td>80.53</td>
<td>2.31×</td>
</tr>
<tr>
<td>SpecEyes (bottom)</td>
<td>83.48</td>
<td>2.13×</td>
<td>84.21</td>
<td>2.12×</td>
<td>75.22</td>
<td>1.20×</td>
<td>71.18</td>
<td>1.04×</td>
<td>85.13</td>
<td>2.08×</td>
<td>87.00</td>
<td>2.08×</td>
<td>90.13</td>
<td>2.11×</td>
<td>82.34</td>
<td>1.82×</td>
</tr>
<tr>
<td><strong>SpecEyes (min)</strong></td>
<td>90.43</td>
<td>1.53×</td>
<td>89.47</td>
<td>1.90×</td>
<td>75.85</td>
<td>1.13×</td>
<td>71.80</td>
<td>1.08×</td>
<td>85.13</td>
<td>2.13×</td>
<td>87.00</td>
<td>2.15×</td>
<td>90.13</td>
<td>2.19×</td>
<td>84.26</td>
<td>1.73×</td>
</tr>
<tr>
<td colspan="18"><em>Based on Thyme [63]</em></td>
</tr>
<tr>
<td>Thyme [63]</td>
<td>86.96</td>
<td>1.00×</td>
<td>82.89</td>
<td>1.00×</td>
<td>77.72</td>
<td>1.00×</td>
<td>72.43</td>
<td>1.00×</td>
<td>81.32</td>
<td>1.00×</td>
<td>84.53</td>
<td>1.00×</td>
<td>90.17</td>
<td>1.00×</td>
<td>82.29</td>
<td>1.00×</td>
</tr>
<tr>
<td>SpecReason [37]</td>
<td>89.57</td>
<td>0.48×</td>
<td>75.00</td>
<td>0.53×</td>
<td>80.01</td>
<td>0.52×</td>
<td>81.02</td>
<td>0.51×</td>
<td>84.62</td>
<td>0.46×</td>
<td>85.97</td>
<td>0.43×</td>
<td>90.27</td>
<td>0.46×</td>
<td>83.78</td>
<td>0.48×</td>
</tr>
<tr>
<td>SpecEyes (log)</td>
<td>80.87</td>
<td>1.82×</td>
<td>82.89</td>
<td>1.45×</td>
<td>74.97</td>
<td>1.13×</td>
<td>70.84</td>
<td>1.06×</td>
<td>85.76</td>
<td>1.68×</td>
<td>87.80</td>
<td>1.67×</td>
<td>91.47</td>
<td>1.59×</td>
<td>82.09</td>
<td>1.49×</td>
</tr>
<tr>
<td>SpecEyes (mean)</td>
<td>77.39</td>
<td>2.34×</td>
<td>80.26</td>
<td>1.83×</td>
<td>72.62</td>
<td>1.27×</td>
<td>68.00</td>
<td>1.21×</td>
<td>85.89</td>
<td>1.78×</td>
<td>88.30</td>
<td>1.80×</td>
<td>91.27</td>
<td>1.65×</td>
<td>80.53</td>
<td>1.70×</td>
</tr>
<tr>
<td>SpecEyes (bottom)</td>
<td>78.26</td>
<td>2.18×</td>
<td>80.26</td>
<td>1.84×</td>
<td>77.35</td>
<td>1.05×</td>
<td>72.31</td>
<td>0.99×</td>
<td>85.89</td>
<td>1.81×</td>
<td>88.30</td>
<td>1.81×</td>
<td>91.27</td>
<td>1.73×</td>
<td>81.95</td>
<td>1.63×</td>
</tr>
<tr>
<td><strong>SpecEyes (min)</strong></td>
<td>87.83</td>
<td>1.32×</td>
<td>82.89</td>
<td>1.42×</td>
<td>78.47</td>
<td>1.01×</td>
<td>73.31</td>
<td>0.95×</td>
<td>85.87</td>
<td>1.77×</td>
<td>88.30</td>
<td>1.78×</td>
<td>91.27</td>
<td>1.70×</td>
<td>83.99</td>
<td>1.42×</td>
</tr>
</tbody>
</table>

Key observations from the results:
1.  **SpecEyes (min) delivers the best accuracy-speed tradeoff**: With DeepEyes backbone, it achieves 1.73× average speedup while improving average accuracy from 81.39% to 84.26% (a +2.87% gain). With Thyme backbone, it achieves 1.42× average speedup while improving average accuracy from 82.29% to 83.99% (a +1.7% gain).
2.  **POPE benchmark benefits the most**: Speedup ranges from 1.70× to 2.19× on POPE, with accuracy improvements up to +6.7% (Adversarial split with DeepEyes backbone). Bypassing unnecessary tool trajectories reduces hallucination errors caused by intermediate tool observations.
3.  **HR-Bench has moderate speedups**: Speedup ranges from 0.95× to 1.13× on HR-Bench, as high-resolution queries more frequently require fine-grained tool-assisted inspection, leading to lower $\beta\alpha$. The marginal sub-1× speedup on HR-Bench 8K with Thyme backbone occurs when the fixed cost of running the small model slightly exceeds savings from bypassed queries, consistent with the latency formula.
4.  **SpecReason performs worse than baselines**: SpecReason consistently decelerates inference (0.37–0.61× speedup) and degrades accuracy, as it incurs substantial token and turn overhead from draft/verification interactions inside the agentic loop.
5.  **Draft-only model shows speedup upper bound**: The small model alone achieves 4.13× average speedup, but with significant accuracy loss (78.93% average accuracy, 2.46% lower than DeepEyes baseline). SpecEyes captures most of this speedup while preserving or improving accuracy.
## 6.2. Ablation Studies & Parameter Analysis
### 6.2.1. Confidence Calibration
The following figure (Figure 3 from the original paper) shows kernel density estimates (KDE) of confidence scores for correct vs. incorrect speculative answers on V* Bench:

![Fig. 3. KDE of confidence scores for correct vs. incorrect samples on ${ \\mathbf { V } } ^ { * }$ (Qwen3-VL-2B). $\\Delta$ measures gating discriminabil a peakcea $( \\mathbf { a } , \\mathbf { b } , \\mathbf { d } )$ or (c) $S _ { \\mathrm { s e p } } ^ { \\mathrm { m i n } }$ b $\\Delta$ Fig. 4. Ablation on the gating threshold of SpecEyes. Lowering the threshold increases speedup at cost of accuracy. Dashed horizontal lines indicate baseline accuracy.](images/3.jpg)
*该图像是图表，展示了在不同阈值下，V*、HR-Bench 和 POPE 数据集上，DeepEyes 和 Thyme 方法与我们的改进方法的准确性（Acc）和加速比（Spd）的关系。数据表格和曲线显示了随着阈值变化，准确性和加速比的波动趋势。*

$\Delta$ measures the peak distance between the correct and incorrect score distributions, quantifying gating discriminability. The min aggregation $S_{\mathrm{sep}}^{\mathrm{min}}$ achieves the highest discriminability ($\Delta=0.030$), with incorrect samples concentrated at low scores and correct samples concentrated at high scores, validating that it is the most reliable gating signal.
### 6.2.2. Gating Threshold Ablation
The right subplot in Figure 3 shows the accuracy-speedup tradeoff as the gating threshold $\tau$ varies. Lowering the threshold increases the acceptance rate $\alpha$, leading to higher speedup, while accuracy degrades gracefully. There is a wide operating range (0.94–0.99 threshold on V* and POPE) where accuracy remains above or near the agentic baseline, showing the threshold is a robust control knob for navigating the accuracy-efficiency Pareto front.
### 6.2.3. Batch Size Ablation
The following figure (Figure 4 from the original paper) shows the effect of serving batch size on speedup:

![该图像是示意图，展示了在不同批量大小下，基于 V* Bench、HR-Bench 和 POPE 的速度提升对比。图(a) 和图(b) 中的插图分别显示了速度的边际增益 `riangle`加速，反映了 SpecEyes 框架的性能提升。](images/4.jpg)
*该图像是示意图，展示了在不同批量大小下，基于 V* Bench、HR-Bench 和 POPE 的速度提升对比。图(a) 和图(b) 中的插图分别显示了速度的边际增益 `riangle`加速，反映了 SpecEyes 框架的性能提升。*

Increasing batch size consistently improves speedup, as the stateless front-end overhead is amortized across more queries. Datasets with higher bypass rates (V* and POPE) benefit more from batching, while HR-Bench saturates earlier due to a larger fraction of tool-required queries. Accuracy remains unchanged across all batch sizes, as batching only affects system execution, not model decisions.
### 6.2.4. Top-K Parameter Ablation
The following figure (Figure 5 from the original paper) shows the effect of the Top-K parameter in the separability score calculation:

![该图像是图表，展示了不同Top-K值下，V*（直接属性和相对位置）及HR-Bench（4k和8k）模型的推理速度与准确率的关系。橙色柱状图显示速度，蓝色曲线表示准确率。](images/5.jpg)
*该图像是图表，展示了不同Top-K值下，V*（直接属性和相对位置）及HR-Bench（4k和8k）模型的推理速度与准确率的关系。橙色柱状图显示速度，蓝色曲线表示准确率。*

Increasing $K$ monotonically improves speedup but degrades accuracy, as larger $K$ includes tokens with weaker contrastive signal, inflating confidence estimates. The default $K=64$ provides a balanced tradeoff, matching baseline accuracy while achieving strong speedup.
# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
SpecEyes is the first agentic-level speculative acceleration framework for agentic MLLMs, addressing the fundamental sequential bottleneck of iterative tool use loops. It uses a lightweight tool-free small MLLM to answer easy queries directly, governed by a novel answer separability cognitive gate that reliably identifies high-confidence outputs without oracle labels. A heterogeneous parallel funnel architecture converts per-query latency savings into system-level throughput gains. Extensive experiments across three diverse benchmarks show that SpecEyes achieves 1.1–3.35× speedup over state-of-the-art agentic MLLM baselines, while improving accuracy by up to 6.7% and boosting throughput under concurrent workloads. This work represents a paradigm shift in speculative acceleration, lifting the paradigm from the token/step level to the full agentic pipeline level.
## 7.2. Limitations & Future Work
### Limitations
The current SpecEyes framework only supports speculation at agentic depth $D=0$ (fully tool-free), so speedups are limited on datasets like HR-Bench where most queries genuinely require tool assistance. For very high-resolution datasets with extremely low $\beta\alpha$, the fixed overhead of running the small speculative model can even lead to marginal slowdowns relative to the baseline.
### Future Work
The authors propose extending SpecEyes to multi-depth speculation, allowing the speculative model to make a bounded number of lightweight tool calls (agentic depth $D=1, 2, ..., n$) before applying the cognitive gate. This will intercept queries at the earliest sufficient depth, further reducing unnecessary fallbacks to the heavy agentic backbone and improving speedup on tool-heavy datasets.
## 7.3. Personal Insights & Critique
### Key Inspirations
The agentic-level speculation paradigm introduced in this work is highly generalizable beyond multimodal MLLMs: it can be applied to any agentic system, including tool-augmented text LLMs, robotics agents, and scientific computing agents, where sequential tool invocation is a major latency bottleneck. The answer separability confidence metric is also a general contribution that can be used to improve calibration of any sequence generation model, beyond the speculative acceleration setting.
### Potential Improvements
1.  **Domain-agnostic gating**: The current gating threshold requires per-benchmark calibration. A domain-agnostic gating mechanism that adapts automatically to different query distributions would improve deployability.
2.  **Lightweight tool use for small model**: Allowing the small speculative model to use a small set of low-cost tools (e.g., fast low-resolution OCR) would increase the acceptance rate $\alpha$ on tool-heavy datasets like HR-Bench, delivering higher speedups without accuracy loss.
3.  **Dynamic batch prioritization**: Prioritizing high-confidence queries in the serving pipeline could further improve throughput under high concurrency, by returning easy queries immediately while processing hard queries in the background.
### Unverified Assumptions
The paper assumes that the large agentic model's tool necessity judgment (Phase I) is perfectly reliable. In practice, the large model may misjudge some tool-required queries as tool-free, leading to increased fallback rates or accuracy loss. A joint judgment mechanism using both the small and large models could improve the reliability of Phase I screening.