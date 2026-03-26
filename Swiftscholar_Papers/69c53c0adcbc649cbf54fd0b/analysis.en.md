# 1. Bibliographic Information
## 1.1. Title
The paper focuses on **Dual Guidance Optimization (DGO)**, a novel experiential learning framework for large language models (LLMs) that unifies external non-parametric experience guidance and internal parametric model knowledge to improve both utilization of experience and internalization of useful reasoning patterns during reinforcement learning from verifiable rewards (RLVR) training for reasoning tasks.
## 1.2. Authors
The authors and their affiliations are:
- *Equal contribution*: Fei Bai, Zhipeng Chen (Gaoling School of Artificial Intelligence, Renmin University of China)
- Co-authors: Huan Hao, Ming Yang, Ran Tao, Bryan Dai (21Quest Research); Wayne Xin Zhao (corresponding author, Renmin University of China); Jian Yang (Beihang University); Hongteng Xu (Renmin University of China)
  The research team has core expertise in LLM reasoning, reinforcement learning for LLMs, and natural language processing.
## 1.3. Journal/Conference
The paper is currently published as a preprint on arXiv, the leading open-access preprint server for computer science research. It has not yet undergone peer review for formal publication in a conference or journal.
## 1.4. Publication Year
2026, published at UTC 2026-03-25T08:52:56.000Z.
## 1.5. Abstract
Existing RL-based LLM training, especially the RLVR paradigm for reasoning tasks, is only a rough approximation of human experiential learning. Humans leverage both external experience (e.g., past solutions, expert guidance) and internal knowledge to guide exploration of solutions, and gradually internalize useful reasoning trajectories into stable, reusable knowledge. To address this gap, the paper proposes DGO, a unified framework that combines external (experience bank of historical trajectories) and internal (model parameters) guidance to improve training effectiveness. DGO follows a closed-loop process: it first constructs an experience bank from past trajectories, then guides exploration under dual guidance, then refines the experience bank and optimizes model parameters with the newly generated trajectories. Experiments show DGO consistently outperforms all baselines across model scales, confirming that better experience utilization and internalization lead to stronger reasoning performance.
## 1.6. Original Source Link
- Preprint link: https://arxiv.org/abs/2603.24093v1
- PDF link: https://arxiv.org/pdf/2603.24093v1
- Publication status: Preprint, not yet formally peer-reviewed.

  ---
# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Existing approaches to improving LLM reasoning via RLVR fall into two siloed categories, both failing to replicate the full human experiential learning process:
1. **Parametric learning methods** (e.g., SFT, GRPO, DAPO): Update model parameters directly to encode knowledge, but do not explicitly reuse high-quality trajectories as reusable experience beyond the current training step, leading to potential knowledge forgetting and wasted useful exploration signals.
2. **Non-parametric experience methods** (e.g., experience-guided inference, skill-augmented RL): Use external memory banks of past trajectories to guide exploration, but do not internalize the reasoning patterns into model parameters, so the model remains dependent on external guidance to perform well.
### Importance of the Problem
Reasoning is a core capability for LLMs to perform complex real-world tasks (e.g., mathematical problem solving, scientific research, tool use). Current RLVR methods have hit performance bottlenecks due to their limited ability to reuse and internalize experience, so unifying experience utilization and internalization is a critical step toward building LLMs with sustained, improving reasoning capabilities.
### Innovative Idea
The paper draws direct inspiration from human experiential learning: humans use external experience to guide exploration of new solutions, then internalize the useful patterns into their own knowledge so they can solve similar problems without external guidance later. DGO formalizes this dual-guidance closed loop for LLMs, combining parametric and non-parametric learning to get the benefits of both paradigms.
## 2.2. Main Contributions / Findings
1. **Empirical finding**: The paper demonstrates that effective experience utilization and internalization directly improve LLM reasoning performance, which increases consistently with more training iterations and more test-time reasoning rounds.
2. **Methodological contribution**: The paper proposes DGO, the first unified framework that formalizes the interaction between parametric model training and non-parametric experience banks, offering a new paradigm for LLM experiential learning.
3. **State-of-the-art performance**: DGO achieves an average score of 32.41% across 6 challenging reasoning benchmarks on Qwen3-8B-Base under intrinsic inference (no external guidance), and 39.38% with test-time scaling, outperforming all strong baselines by clear margins across all model scales tested.

   ---
# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
To understand the paper, beginners first need to master the following core concepts:
1. **Large Language Models (LLMs)**: Transformer-based neural networks trained on massive amounts of text data, capable of understanding and generating human-like text, and solving a wide range of tasks including reasoning, translation, and code generation. Knowledge is stored in the model's trainable parameters (weights).
2. **Reinforcement Learning (RL)**: A machine learning paradigm where an agent (here, the LLM) learns to make decisions by interacting with an environment. The agent receives a reward signal for good actions (e.g., generating a correct answer to a math problem) and updates its policy (behavior) to maximize cumulative reward over time.
3. **Reinforcement Learning from Verifiable Rewards (RLVR)**: A specialized RL paradigm for LLM reasoning tasks, where rewards are fully objective and verifiable (e.g., checking if the final numerical answer to a math problem matches the ground truth), rather than subjective human preference scores used in RLHF (Reinforcement Learning from Human Feedback) for alignment.
4. **Experiential Learning**: A learning approach inspired by human cognition, where knowledge and skills are acquired by reflecting on, reusing, and internalizing past problem-solving experiences (both correct trajectories and mistakes).
5. **Parametric vs Non-parametric Learning**:
   - *Parametric learning*: Knowledge is encoded directly in the fixed set of model parameters (e.g., fine-tuning an LLM on labeled data). Once trained, no external storage is needed for inference.
   - *Non-parametric learning*: Knowledge is stored in external, dynamic storage (e.g., a memory bank of past trajectories) that is separate from model parameters. The model retrieves relevant knowledge from the external storage during inference or training.
6. **GRPO (Group Relative Policy Optimization)**: A state-of-the-art RL algorithm for LLMs, optimized for RLVR settings. It compares multiple trajectories generated for the same prompt to compute relative advantage values, making training more stable and effective for reasoning tasks than standard PPO.
7. **Pass@K**: A standard evaluation metric for reasoning tasks, measuring the probability that at least one of K generated trajectories for a problem is correct. It evaluates the model's ability to explore valid solution paths when given multiple attempts.
8. **avg@16**: A common evaluation metric for LLM reasoning, where 16 trajectories are generated per problem, the average correctness of these 16 trajectories is computed per problem, then averaged across all problems in the benchmark. It measures the average quality of the model's generated solutions.
## 3.2. Previous Works
The paper categorizes related work into two main lines of research:
### Non-parametric Experiential Learning
These methods focus on building external experience banks to guide reasoning, but do not internalize patterns into model parameters:
- Experience construction and retrieval work: Methods like Agent KB, Reasonflux build hierarchical experience repositories with update and verification mechanisms to store reusable reasoning strategies, templates, and past solutions. These methods improve exploration but do not encode the strategies into the model itself.
- Experience utilization work: Methods like EGI, SkillRL retrieve relevant experience during inference to guide the model to generate better solutions, but the model is not trained to internalize these strategies, so performance drops sharply when external experience is removed.
### Parametric Experiential Learning
These methods update model parameters directly to encode reasoning patterns, but do not explicitly reuse experience as external guidance:
- Supervised Fine-Tuning (SFT): Trains the model on labeled correct reasoning trajectories using cross-entropy loss, but does not leverage exploration signals from incorrect trajectories or reuse past trajectories for iterative improvement.
- RLVR methods: GRPO, DAPO, BAPO use verifiable rewards to optimize the model's policy, but treat high-quality trajectories only as one-time training signals, not as reusable experience to guide future exploration.
- Self-distillation methods: On-Policy Self-Distillation distills past generated trajectories into the model to improve performance, but does not use external experience to guide exploration of better trajectories in the first place.
## 3.3. Technological Evolution
The evolution of LLM reasoning improvement has followed a clear trajectory:
1. First stage: SFT on labeled Chain-of-Thought reasoning data, which teaches the model to follow step-by-step reasoning formats but is limited by the scale and quality of labeled data.
2. Second stage: RLVR for reasoning, which uses objective rewards to let the model explore and learn from its own trajectories, overcoming the limitation of labeled data scarcity, but lacking explicit experience reuse mechanisms.
3. Third stage: Experiential learning for LLMs, first with non-parametric experience banks to guide inference, then with parametric self-distillation to reuse past trajectories. The paper's DGO framework is the next step, unifying both parametric and non-parametric experiential learning into a single closed loop.
## 3.4. Differentiation Analysis
Compared to all existing methods, DGO is the only approach that combines both parametric and non-parametric optimization, and prioritizes both experience utilization and internalization. The following table (Table 1 from the original paper) summarizes this differentiation:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">Optimization Paradigm</th>
<th colspan="2">Enhanced Capability</th>
</tr>
<tr>
<th>Non-parameter</th>
<th>Parameter</th>
<th>Utilization</th>
<th>Internalization</th>
</tr>
</thead>
<tbody>
<tr>
<td>SFT</td>
<td>×</td>
<td>✓</td>
<td>×</td>
<td>✓</td>
</tr>
<tr>
<td>GRPO</td>
<td>×</td>
<td>✓</td>
<td>×</td>
<td>✓</td>
</tr>
<tr>
<td>On-Policy Self-Distillation</td>
<td>✓</td>
<td>✓</td>
<td>×</td>
<td>✓</td>
</tr>
<tr>
<td>Skill-Augmented RL</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>×</td>
</tr>
<tr>
<td>DGO (Ours)</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
</tr>
</tbody>
</table>

---
# 4. Methodology
## 4.1. Principles
The core principle of DGO is mimicking the dual-guidance mechanism of human experiential learning:
- External guidance (from an experience bank of past trajectories) helps the model explore high-quality solution trajectories that it would not be able to discover using only its current internal knowledge.
- Internal guidance (encoded in the model's parameters) represents the model's accumulated reasoning capability, which improves as it internalizes patterns from the trajectories discovered with external guidance.
  DGO forms a closed feedback loop: external experience improves trajectory discovery, the resulting trajectories are used to refine the experience bank and optimize the model's internal parameters, and the improved model and experience bank support even more effective exploration in subsequent iterations.
## 4.2. Core Methodology In-depth
DGO runs K iterative training stages, each consisting of three sequential steps: Experience Construction, Joint Trajectory-Policy Refinement, and Experience Internalization. The following figure (Figure 2 from the original paper) illustrates the full closed-loop framework:

![Figure 2. Our framework consists of three components: 1) Experience Construction, 2) Joint TrajectoryPolicy Refinement, and 3) Experience Internalization, forming a closed-loop process that enhances experience utilization and internalization. I denotes the training iteration, and the final model is the RL policy from the last iteration.](images/2.jpg)
*该图像是示意图，展示了我们的框架的三个组成部分：1) 经验构建，2) 联合轨迹-策略优化，3) 经验内化，形成一个闭环过程，增强经验利用和内化。I 表示训练迭代，最终模型为最后迭代的强化学习策略。*

The full training procedure is formalized in Algorithm 1 of the paper, which we break down step-by-step with all corresponding mathematical formulas integrated below:
### Inputs and Initialization
The algorithm takes as input:
- Training set $\mathcal{D}$ of reasoning problems
- Initial base model $\pi_{\theta^{(0)}}$ (untrained policy)
- Pre-built initial experience bank $\mathcal{E}$
- Experience generator $G$ (model that extracts reusable experience from trajectories)
- Number of training stages $K$
- Annealing schedule $\{\alpha_k\}_{k=1}^K$ (controls fraction of experience-guided training data per iteration)
- Internalization ratios $\{\beta_k\}_{k=1}^K$ (controls fraction of implicit trajectories in distillation data per iteration)

  Initialization steps: Set the starting policy for the first iteration to the initial base model $\theta_{\text{start}}^{(1)} = \theta^{(0)}$, initialize empty sets for accumulated implicit trajectories $S_{\text{imp}}^{(\leq 0)} = \emptyset$ and accumulated experience-guided trajectories $S_e^{(\leq 0)} = \emptyset$.

---
### Step 1: Experience Construction
In this step, reusable, generalizable experience is distilled from historical reasoning trajectories, rather than storing full trajectories directly. Each experience is represented as a triplet $\boldsymbol{e} = (c, \kappa, \tau)$:
- $c$: Broad domain or problem phase (e.g., "solving algebraic product problems over roots of unity")
- $\kappa$: Specific condition, scenario, or potential error (e.g., "when computing product over roots of unity, avoid incorrect polynomial substitution")
- $\tau$: Corresponding action to take or avoid (e.g., "use the identity $\prod_{y^n=1, y \neq 1} (y - a) = \frac{a^n - 1}{a-1}$ to compute the product efficiently")

  For any generated trajectory $s$, the experience generator $G$ extracts an experience `e = G(s)`. Experiences from correct trajectories capture effective solving strategies, while experiences from incorrect trajectories capture common errors and how to avoid them. All experiences are strictly filtered to remove any content that contains or directly suggests the final answer to a problem, to ensure the experience provides transferable reasoning guidance rather than answer-specific supervision. The experience bank is defined as the set of all valid extracted experiences: $\mathcal{E} = \{e_i\}_{i=1}^N$.

---
### Step 2: Joint Trajectory-Policy Refinement
This step uses RL (GRPO) to train the model to utilize external experience while generating high-quality trajectories for subsequent internalization. It has two sub-components:
#### Sub-step 2.1: Exploration with Experience Annealing
The goal is to train the model to use external experience when available, while gradually reducing its dependence on external guidance to encourage internalization of reasoning patterns. At iteration $k$, the training dataset is a mixture of two distributions:
- $D_e^{(k)}$: Experience-guided distribution, where each instance includes the problem $x$ and relevant retrieved experience from the experience bank
- $D_i^{(k)}$: Experience-free distribution, where each instance includes only the problem $x$, identical to standard RL training

  The mixture is defined as:
$$
D^{(k)} = \alpha_k D_e^{(k)} + (1-\alpha_k) D_i^{(k)}
$$
Where $\alpha_k$ is the annealing coefficient for iteration $k$, which decreases as training progresses (in experiments, $\alpha$ is set to 1.0, 0.5, 0.25 for 3 iterations respectively). In early iterations, high $\alpha$ lets the model use external experience to discover good trajectories it would not find on its own; in later iterations, lower $\alpha$ forces the model to rely more on its internalized knowledge.

Each training instance is represented as $z = (x, \tilde{e})$, where $\tilde{e}$ is the retrieved experience for experience-guided instances, and empty for experience-free instances. The policy is optimized using the standard GRPO loss over the mixed training distribution:
$$
\mathcal{L}(\theta) = \mathbb{E}_{z \sim D^{(k)}, s \sim \pi_{\theta_{\text{old}}}(\cdot | z)} \left[ \frac{1}{|s|} \sum_{t=1}^{|s|} \min \left( \rho_t(\theta) A_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$
Explanation of symbols:
- $\rho_t(\theta) = \frac{\pi_\theta(s_t | z, s_{<t})}{\pi_{\theta_{\text{old}}}(s_t | z, s_{<t})}$: Probability ratio of the current policy $\pi_\theta$ and the old policy $\pi_{\theta_{\text{old}}}$ for generating token $s_t$ at step $t$, given the instance $z$ and previous tokens $s_{<t}$
- $A_t$: Advantage value at step $t$, estimating how much better taking action $s_t$ is than the average action at that step
- $\epsilon$: Clipping threshold (set to 0.2 in experiments), which prevents excessively large policy updates that would destabilize training
- $|s|$: Total length of the generated trajectory $s$

  This loss ensures the model learns to perform well both with and without external experience guidance.

#### Sub-step 2.2: Refinement via Experience Renewal
After optimizing the policy with RL for iteration $k$, collect a set of rollout trajectories $S^{(k)} = \{s_j^{(k)}\}$ generated by the updated policy. Use the experience generator $G$ to extract new experiences from these trajectories: $\mathcal{E}_{\text{new}}^{(k)} = G(S^{(k)})$. Then update the experience bank by selecting the most informative experiences from the union of the old bank and new experiences:
$$
\mathcal{E}^{(k+1)} = \text{Select}\left( \mathcal{E}^{(k)} \cup \mathcal{E}_{\text{new}}^{(k)} \right)
$$
The $\text{Select}(\cdot)$ function follows two rules:
1. **Redundancy check**: Compute semantic similarity between each new experience and existing experiences in the bank. If the maximum similarity is below a threshold of 0.8, the new experience is a candidate for addition; otherwise, it is a candidate for replacement of the most similar existing experience.
2. **Quality check**: Add or replace the experience only if it improves the validation set avg@16 performance of the experience bank.

   The updated experience bank $\mathcal{E}^{(k+1)}$ is used to construct the mixed training distribution for the next iteration, creating a positive feedback loop: better trajectories → higher-quality experience → better guidance for future exploration → even better trajectories.

---
### Step 3: Experience Internalization
This step transfers the reasoning gains from external experience guidance into the model's parameters, so the model can perform well even without external experience at inference time. It has two sub-components:
#### Sub-step 3.1: Trajectory Rephrasing
Experience-guided trajectories generated in Step 2 contain explicit references to the external experience included in the prompt. If used directly for training, the model would learn to rely on these explicit references rather than internalizing the underlying reasoning patterns. To address this, we rewrite experience-guided trajectories to remove all explicit references to experience, while preserving the full underlying reasoning logic.

For a problem $x$, retrieved experience $\mathcal{E}_x$, experience-guided trajectory $s_e^{(k)} \in S_e^{(k)}$ (set of all experience-guided trajectories from iteration $k$), and rephrasing prompt $p$, the rewritten trajectory is generated as:
$$
\tilde{s}_e^{(k)} \sim \pi_{\theta_{\text{RL}}^{(k)}} \left( \cdot | x, s_e^{(k)}, \mathcal{E}_x, p \right)
$$
The rephrasing prompt instructs the model to produce a fully self-contained solution that does not mention external experience at all, justifies all steps from first principles, fixes any errors in the original trajectory, and preserves the correct final answer.

Rewritten trajectories are then filtered to remove:
- Trajectories with incorrect final answers
- Trajectories that still explicitly mention experience
- Trajectories longer than 6K tokens
- Noisy trajectories with meaningless text

  The top highest-quality rewritten trajectories per problem are retained to form the implicit trajectory set $S_{\text{imp}}^{(k)}$:
$$
S_{\text{imp}}^{(k)} = \text{Top-K}\left( \mathcal{F}\left( \tilde{S}_e^{(k)} \right) \right)
$$
Where $\mathcal{F}(\cdot)$ is the filtering operator, and $\text{Top-K}(\cdot)$ selects the best K rewritten trajectories per problem.

#### Sub-step 3.2: Policy Distillation
Accumulate all implicit trajectories from all iterations up to $k$: $S_{\text{imp}}^{(\leq k)} = \bigcup_{i=1}^k S_{\text{imp}}^{(i)}$, and all experience-guided trajectories from all iterations up to $k$: $S_e^{(\leq k)} = \bigcup_{i=1}^k S_e^{(i)}$. The distillation dataset is a mixture of these two sets, to train both intrinsic reasoning ability (from implicit trajectories) and the ability to utilize external experience (from experience-guided trajectories):
$$
\mathcal{D}_{\text{dist}}^{(k)} = \beta_k S_{\text{imp}}^{(\leq k)} + (1-\beta_k) S_e^{(\leq k)}
$$
Where $\beta_k$ is the internalization ratio for iteration $k$, fixed at 0.5 in experiments.

The policy is then optimized via maximum likelihood estimation (supervised fine-tuning) on this distillation dataset, initialized from the original base model $\theta^{(0)}$ to mitigate overfitting:
$$
\theta^{(k+1)} = \arg\max_{\theta} \mathbb{E}_{(x, s) \sim \mathcal{D}_{\text{dist}}^{(k)}} \log \pi_\theta(s | x)
$$
The resulting model $\theta^{(k+1)}$ is used as the starting policy for the next RL iteration, completing the closed loop.

---
### Inference Paradigms
DGO supports two inference modes:
1. **Intrinsic Inference**: Evaluates the model's standalone reasoning ability without any external experience. The model generates a trajectory solely from its internal parametric knowledge:
   $s \sim \pi_\theta(\cdot | x)$
2. **Iterative Experience-Guided Test-Time Scaling (TTS)**: Uses external experience to improve inference performance. For a problem $x$, run K rounds of trajectory generation. The first round uses the pre-built offline experience bank for guidance, and subsequent rounds use online experience extracted from trajectories generated in the previous round:
   $$
s^{(k)} \sim
\begin{cases}
\pi_\theta(\cdot | x, \mathcal{E}_{\text{offline}}), & k=1 \\
\pi_\theta(\cdot | x, \mathcal{E}_{\text{online}}^{(k-1)}), & k>1
\end{cases}
$$
---
# 5. Experimental Setup
## 5.1. Datasets
### Training Dataset
The model is trained on 9,600 mathematical reasoning questions sampled from the SkyWork open reasoner dataset, a high-quality dataset of math problems with reasoning trajectories.
### Evaluation Benchmarks
The model is evaluated on 6 benchmarks covering both in-domain (math) and out-of-domain (general reasoning/knowledge) tasks:
1. **In-domain (math reasoning)**:
   - AIME24: 2024 American Invitational Mathematics Examination, hard high school math problems
   - AIME25: 2025 AIME, same difficulty as AIME24
   - HMMT25: 2025 Harvard-MIT Mathematics Tournament, very advanced high school math problems
   - MATH500: 500 high school math problems of varying difficulty from the standard MATH dataset
2. **Out-of-domain (general reasoning/knowledge)**:
   - GPQA-Diamond: Graduate-level, Google-proof multi-choice questions across STEM, humanities, and social sciences
   - HLE-Verified: "Humanity's Last Exam" dataset of high-difficulty general knowledge and reasoning questions

     These benchmarks are chosen because they cover a wide range of difficulty levels and task domains, allowing rigorous testing of both in-domain performance and cross-domain generalization of reasoning capabilities.
## 5.2. Evaluation Metrics
The paper uses two core evaluation metrics, explained in detail below:
### Metric 1: avg@16
1. **Conceptual Definition**: For each problem, generate 16 trajectories, compute the average correctness of these 16 trajectories, then average this value across all problems in the benchmark. It measures the average quality of the model's generated solutions, and how consistently the model produces correct answers when sampling 16 times per problem.
2. **Mathematical Formula**:
   $$
\text{avg@16} = \frac{1}{M} \sum_{i=1}^M \left( \frac{1}{16} \sum_{j=1}^{16} \mathbb{I}\left( s_{i,j} \text{ is correct} \right) \right)
$$
3. **Symbol Explanation**:
   - $M$: Total number of problems in the benchmark
   - $s_{i,j}$: The j-th generated trajectory for problem i
   - $\mathbb{I}(\cdot)$: Indicator function that equals 1 if the trajectory is correct, 0 otherwise

### Metric 2: Pass@K
1. **Conceptual Definition**: Measures the probability that at least one of K generated trajectories for a problem is correct. It evaluates the model's ability to explore at least one valid solution path when given K attempts.
2. **Mathematical Formula**:
   $$
\text{Pass@K} = \frac{1}{M} \sum_{i=1}^M \left( 1 - \frac{\binom{N_i - C_i}{K}}{\binom{N_i}{K}} \right)
$$
Where $N_i \geq K$, and $N_i$ is the total number of trajectories generated for problem i (16 in all experiments in the paper).
3. **Symbol Explanation**:
   - $M$: Total number of problems in the benchmark
   - $C_i$: Number of correct trajectories for problem i out of $N_i$ total generated trajectories
   - $\binom{a}{b}$: Binomial coefficient, the number of ways to choose b elements from a elements

## 5.3. Baselines
The paper compares DGO to four representative baselines covering all existing paradigms:
1. **EGI (Experience-Guided Inference)**: Training-free baseline that adds external experience to prompts at inference time, without updating the model's parameters. It tests the performance gain from just adding external experience without training the model to use or internalize it.
2. **SFT (Supervised Fine-Tuning)**: Standard parametric baseline that fine-tunes the model on labeled correct reasoning trajectories using cross-entropy loss, no RL or experience reuse.
3. **GRPO (Group Relative Policy Optimization)**: State-of-the-art parametric RLVR baseline, no external experience guidance during training.
4. **DAPO (Distributed Adaptive Policy Optimization)**: Scalable open-source parametric RL system for LLMs, another strong RLVR baseline with no external experience guidance.

   These baselines are representative of all existing approaches to LLM reasoning improvement, so they allow clear measurement of the incremental benefits of DGO's unified dual-guidance paradigm.

---
# 6. Results & Analysis
## 6.1. Core Results Analysis
The main experimental results are presented in Table 2 from the original paper, which compares DGO to all baselines across three model scales (4B, 8B, 14B parameters) on all 6 benchmarks:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Method</th>
<th colspan="4">In-domain (%)</th>
<th colspan="2">Out-of-domain (%)</th>
<th rowspan="2">Avg</th>
</tr>
<tr>
<th>AIME24</th>
<th>AIME25</th>
<th>HMMT25</th>
<th>MATH500</th>
<th>GPQA-D</th>
<th>HLE</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="6">Qwen3-4B-Base</td>
<td>EGI</td>
<td>1.46</td>
<td>1.46</td>
<td>0.00</td>
<td>23.52</td>
<td>12.12</td>
<td>1.31</td>
<td>6.64</td>
</tr>
<tr>
<td>SFT</td>
<td>5.83</td>
<td>5.62</td>
<td>1.04</td>
<td>31.52</td>
<td>22.92</td>
<td>2.56</td>
<td>11.58</td>
</tr>
<tr>
<td>GRPO</td>
<td>26.88</td>
<td>22.50</td>
<td>8.75</td>
<td>72.75</td>
<td>42.55</td>
<td>2.73</td>
<td>29.36</td>
</tr>
<tr>
<td>DAPO</td>
<td>26.04</td>
<td>22.29</td>
<td>9.17</td>
<td>71.35</td>
<td>41.79</td>
<td>2.34</td>
<td>28.83</td>
</tr>
<tr>
<td>DGO (zero)</td>
<td>27.29</td>
<td>23.54</td>
<td>11.67</td>
<td>72.60</td>
<td>43.56</td>
<td>3.37</td>
<td>30.34</td>
</tr>
<tr>
<td>DGO (TTS)</td>
<td>40.21</td>
<td>27.71</td>
<td>15.83</td>
<td>78.35</td>
<td>52.53</td>
<td>4.45</td>
<td>36.51</td>
</tr>
<tr>
<td rowspan="6">Qwen3-8B-Base</td>
<td>EGI</td>
<td>6.04</td>
<td>4.79</td>
<td>2.08</td>
<td>41.50</td>
<td>22.22</td>
<td>3.12</td>
<td>13.29</td>
</tr>
<tr>
<td>SFT</td>
<td>15.00</td>
<td>11.46</td>
<td>3.54</td>
<td>70.38</td>
<td>36.30</td>
<td>4.34</td>
<td>23.50</td>
</tr>
<tr>
<td>GRPO</td>
<td>29.79</td>
<td>20.42</td>
<td>9.17</td>
<td>73.20</td>
<td>44.32</td>
<td>3.95</td>
<td>30.14</td>
</tr>
<tr>
<td>DAPO</td>
<td>25.00</td>
<td>19.17</td>
<td>9.17</td>
<td>80.77</td>
<td>45.83</td>
<td>4.66</td>
<td>30.77</td>
</tr>
<tr>
<td>DGO (zero)</td>
<td>30.21</td>
<td>23.33</td>
<td>11.88</td>
<td>75.25</td>
<td>49.75</td>
<td>4.02</td>
<td>32.41</td>
</tr>
<tr>
<td>DGO (TTS)</td>
<td>49.38</td>
<td>30.63</td>
<td>14.37</td>
<td>80.42</td>
<td>54.99</td>
<td>6.51</td>
<td>39.38</td>
</tr>
<tr>
<td rowspan="6">Qwen3-14B-Base</td>
<td>EGI</td>
<td>8.33</td>
<td>5.42</td>
<td>1.67</td>
<td>47.02</td>
<td>28.91</td>
<td>3.29</td>
<td>15.77</td>
</tr>
<tr>
<td>SFT</td>
<td>18.12</td>
<td>13.33</td>
<td>6.46</td>
<td>74.58</td>
<td>37.12</td>
<td>4.96</td>
<td>25.76</td>
</tr>
<tr>
<td>GRPO</td>
<td>38.12</td>
<td>27.08</td>
<td>14.37</td>
<td>89.00</td>
<td>45.01</td>
<td>5.15</td>
<td>36.45</td>
</tr>
<tr>
<td>DAPO</td>
<td>35.83</td>
<td>26.25</td>
<td>14.17</td>
<td>89.50</td>
<td>49.37</td>
<td>5.45</td>
<td>36.76</td>
</tr>
<tr>
<td>DGO (zero)</td>
<td>38.54</td>
<td>29.17</td>
<td>15.21</td>
<td>90.60</td>
<td>50.13</td>
<td>5.63</td>
<td>38.04</td>
</tr>
<tr>
<td>DGO (TTS)</td>
<td>54.48</td>
<td>39.38</td>
<td>22.29</td>
<td>93.17</td>
<td>51.64</td>
<td>7.71</td>
<td>44.78</td>
</tr>
</tbody>
</table>

The following figure (Figure 1 from the original paper) summarizes the average performance across model scales, clearly showing DGO's consistent advantages:

![Figure 1. Comparison of DGO and baseline methods on six in-domain and out-of-domain benchmarks using Qwen3-4B/8B/14B-Base.](images/1.jpg)
*该图像是一个柱状图，展示了DGO和基线方法在四种不同大小（4B、8B、14B）模型上的平均准确率（%）比较。不同颜色分别标识了SFT、GRPO、DGO(zero)和DGO(TTS)的表现，显示出DGO方法的优势。*

Key takeaways from the core results:
1. **Superior intrinsic reasoning performance**: DGO (zero, no external experience) outperforms all baselines across all three model scales. For the 8B model, DGO achieves 32.41% average score vs 30.77% for the best baseline (DAPO), and for the 14B model, it achieves 38.04% vs 36.76% for DAPO. This confirms that DGO's experience internalization mechanism effectively encodes reusable reasoning patterns into the model's parameters, improving standalone reasoning ability.
2. **Strong experience utilization ability**: Adding Test-Time Scaling (TTS) leads to large, consistent gains over intrinsic inference: 4B model improves from 30.34% to 36.51%, 8B from 32.41% to 39.38%, 14B from 38.04% to 44.78%. As shown in Figure 3 below (Figure 3 from the original paper), performance increases steadily with more TTS rounds, and DGO outperforms all baselines at every round, confirming that DGO trains the model to effectively leverage external experience to improve performance.

   ![Figure 3. Accuracy of Qwen3-8B-Base across successive TTS rounds on AIME24 and AIME25, compared with DGO and other baseline methods.](images/3.jpg)
   *该图像是图表，展示了Qwen3-8B-Base在AIME24和AIME25上在不同轮次（Round K）中的累计准确率变化。图中包含DGO、GRPO、DAPO、SFT和EGI等不同方法的表现，说明DGO在各轮次中具有较好的准确率。*

3. **Cross-domain generalization**: DGO is trained exclusively on math data, but still outperforms baselines on out-of-domain non-math benchmarks. For example, the 8B DGO (zero) achieves 49.75% on GPQA-D vs 45.83% for DAPO, showing that the reasoning patterns internalized by DGO are generalizable beyond the training domain.
## 6.2. Ablation Studies
### Component Ablation
The paper conducts component-level ablation studies to verify the contribution of each core component of DGO, with results shown in Table 3 from the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">In-domain (%)</th>
<th colspan="2">Out-of-domain (%)</th>
<th rowspan="2">Avg</th>
</tr>
<tr>
<th>AIME24</th>
<th>AIME25</th>
<th>HMMT25</th>
<th>MATH500</th>
<th>GPQA-D</th>
<th>HLE</th>
</tr>
</thead>
<tbody>
<tr>
<td>DGO (zero)</td>
<td>27.08</td>
<td>21.04</td>
<td>9.17</td>
<td>70.08</td>
<td>40.47</td>
<td>2.99</td>
<td>28.47</td>
</tr>
<tr>
<td>w/o EA</td>
<td>19.38</td>
<td>17.50</td>
<td>7.71</td>
<td>68.08</td>
<td>38.64</td>
<td>3.24</td>
<td>25.76</td>
</tr>
<tr>
<td>w/o ER</td>
<td>19.58</td>
<td>18.75</td>
<td>7.50</td>
<td>68.53</td>
<td>38.57</td>
<td>2.90</td>
<td>25.97</td>
</tr>
<tr>
<td>w/o TR</td>
<td>22.08</td>
<td>17.50</td>
<td>6.04</td>
<td>66.65</td>
<td>39.90</td>
<td>2.47</td>
<td>25.77</td>
</tr>
<tr>
<td>w/o PD</td>
<td>23.75</td>
<td>20.83</td>
<td>7.71</td>
<td>67.73</td>
<td>38.26</td>
<td>2.79</td>
<td>26.84</td>
</tr>
<tr>
<td>DGO (TTS)</td>
<td>40.83</td>
<td>26.67</td>
<td>15.42</td>
<td>73.90</td>
<td>52.65</td>
<td>4.58</td>
<td>35.68</td>
</tr>
<tr>
<td>w/o EA</td>
<td>37.71</td>
<td>24.58</td>
<td>13.12</td>
<td>70.43</td>
<td>48.30</td>
<td>3.37</td>
<td>32.92</td>
</tr>
<tr>
<td>w/o ER</td>
<td>42.92</td>
<td>23.75</td>
<td>15.21</td>
<td>70.78</td>
<td>48.42</td>
<td>3.11</td>
<td>34.03</td>
</tr>
<tr>
<td>w/o TR</td>
<td>31.87</td>
<td>20.42</td>
<td>11.67</td>
<td>72.10</td>
<td>48.11</td>
<td>3.99</td>
<td>31.36</td>
</tr>
<tr>
<td>w/o PD</td>
<td>34.58</td>
<td>22.71</td>
<td>15.00</td>
<td>67.00</td>
<td>49.31</td>
<td>3.20</td>
<td>31.97</td>
</tr>
</tbody>
</table>

Where EA = Experience Annealing, ER = Experience Renewal, TR = Trajectory Rephrasing, PD = Policy Distillation. Removing any component leads to consistent performance drops for both DGO (zero) and DGO (TTS), confirming all components are critical to DGO's effectiveness:
- Removing ER (experience renewal) reduces average score by 2.5% (zero) and 1.65% (TTS), showing that keeping the experience bank updated with fresh, high-quality experience is necessary for effective guidance.
- Removing TR (trajectory rephrasing) reduces average score by 2.7% (zero) and 4.32% (TTS), showing that removing explicit references to experience from trajectories is necessary to avoid the model learning to rely on external cues.
- Removing EA (experience annealing) reduces average score by 2.71% (zero) and 2.76% (TTS), showing that gradually reducing experience guidance during training encourages internalization of patterns.
- Removing PD (policy distillation) reduces average score by 1.63% (zero) and 3.71% (TTS), showing that distilling accumulated trajectories into the model is necessary to consolidate gains into stable knowledge.

### Iterative Training Ablation
The following figure (Figure 4 from the original paper) shows accuracy curves across training iterations for 4B and 8B models on AIME25:

![Figure 4. Accuracy curves of two models on AIME25 across training iterations.](images/4.jpg)
*该图像是图表，展示了 Qwen3-8B-Base 和 Qwen3-4B-Base 两个模型在 AIME25 上的平均准确率随训练步数的变化。图中包含三个迭代的准确率曲线，分别标记为 iteration 1、iteration 2 和 iteration 3。*

Each subsequent iteration starts from a higher initial accuracy and maintains higher performance across all training steps, confirming that experience accumulated in earlier iterations provides better guidance for later training, and the closed-loop iterative process works as intended.

## 6.3. Additional Analysis
### Experience Utilization Analysis
The following figure (Figure 5 from the original paper) compares DGO and GRPO performance under three settings: no experience, noisy/irrelevant experience, and relevant experience:

![Figure 5. Experience utilization under three settings on AIME24. DGO benefits more from relevant experience and is more robust to irrelevant experience. Numbers denote relative improvement over zero experience.](images/5.jpg)
*该图像是图表，展示了在AIME24上不同设置下的平均准确率。左侧（Qwen-8B-Base）和右侧（Qwen-4B-Base）的结果分别比较了GRPO和DGO在零经验、噪声经验和相关经验下的表现。DGO在相关经验下的改进幅度达到了37.9%，而GRPO的最大提升为18.9%。*

DGO benefits significantly more from relevant experience than GRPO: for 8B model, DGO gets 37.9% relative improvement with relevant experience, vs 18.9% for GRPO. DGO is also much more robust to noisy experience: for 8B model, noisy experience causes almost no performance drop for DGO, while GRPO performance drops significantly. This confirms DGO trains the model to both use useful experience effectively and ignore irrelevant experience.

### Experience Internalization Analysis
The following two figures confirm DGO's better internalization of reasoning patterns:
1. Figure 6 (t-SNE visualization of trajectories): DGO generates trajectories that are semantically distinct from GRPO's trajectories, meaning it learns new reasoning behaviors not captured by standard RL.

   ![Figure 6. t-SNE visualization of generated trajectories on AIME24 for the DGO- and GRPO-trained models.](images/6.jpg)
   *该图像是t-SNE可视化图，展示了在AIME24上DGO和GRPO训练模型生成的轨迹分布。左侧为Qwen3-8B-Base模型的结果，右侧为Qwen3-4B-Base模型的结果，图中不同颜色和形状的点表示不同的模型设置，具有一定的聚类特征。*

2. Figure 7 (Pass@K comparison): DGO achieves higher Pass@K than GRPO across all K values, showing that the new reasoning behaviors it learns are valid solution paths, so DGO broadens the range of valid reasoning modes the model can produce.

   ![Figure 7. Comparison of the DGO- and GRPO-trained models on AIME24 in terms of Pass $@ \\mathrm { K }$ Accuracy.](images/7.jpg)
   *该图像是图表，展示了DGO和GRPO模型在AIME24上基于不同K值（256, 512, 1024）的Pass `@ K` 准确率对比。图中可以看出，DGO模型的性能显著优于GRPO模型，特别是在较高的K值时。*

---
# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
The paper proposes Dual Guidance Optimization (DGO), a unified experiential learning framework for LLMs that combines external non-parametric experience bank guidance and internal parametric model knowledge in a closed loop of experience utilization and internalization. DGO iteratively constructs a reusable experience bank, guides exploration of high-quality trajectories under dual guidance, then rewrites these trajectories to remove explicit experience references and distills them into the model's parameters, updating the experience bank with new trajectories each iteration. Experiments across three model scales on six challenging reasoning benchmarks show DGO consistently outperforms all state-of-the-art baselines, improving both intrinsic reasoning ability (no external experience) and test-time experience utilization ability. The results confirm that unifying experience utilization and internalization is a highly effective approach to improving LLM reasoning capabilities.
## 7.2. Limitations & Future Work
### Limitations (as identified by the authors)
1. The experience generator is a separate trained model, which adds significant computational cost to the training pipeline.
2. The experience update and selection rules are hand-crafted, not learned adaptively by the model.
3. The framework is currently tested only on reasoning tasks, and its effectiveness on other LLM tasks (e.g., alignment, creative generation, tool use) is not verified.
### Future Work (as proposed by the authors)
1. Enable models to manage and update their own experience more autonomously, without requiring a separate experience generator and hand-crafted selection rules.
2. Improve coordination between external and internal experience, to better balance reliance on each depending on the task and model state.
3. Extend DGO to other tasks beyond reasoning, such as tool use, code generation, and LLM alignment.
4. Improve the efficiency of the iterative training loop to reduce computational cost, making it feasible for very large models.
## 7.3. Personal Insights & Critique
### Inspirations
DGO's design is highly aligned with human learning principles, which is a promising direction for LLM training beyond just scaling parameters and data. The closed-loop paradigm of utilization and internalization can be applied to many other domains:
- For tool use agents: Store past successful tool use trajectories as experience, guide future tool use exploration, then internalize tool use patterns into the model so it can use tools effectively even without external memory.
- For code generation: Store past correct, efficient code trajectories as experience, guide exploration of better code, then internalize coding patterns into the model to improve standalone code generation quality.
- For medical diagnosis agents: Store past successful diagnosis trajectories as experience, guide exploration of correct diagnosis paths, then internalize diagnostic reasoning patterns into the model for better clinical performance.

### Potential Issues & Improvements
1. **Computational cost**: The current iterative training process requires multiple rounds of RL training and distillation, which is computationally expensive, especially for very large models (70B+ parameters). Future work could optimize the loop to reduce the number of iterations needed, or use more efficient distillation methods.
2. **Experience bank scalability**: The experience bank can grow very large over time, leading to higher retrieval cost and potential redundancy, even with the current selection rules. Future work could implement adaptive pruning of the experience bank to keep it small and high-quality.
3. **Resetting model initialization for distillation**: The paper initializes the distillation step from the original base model each iteration to avoid overfitting, but this may limit the amount of knowledge that can be accumulated across iterations, as the model does not build on previous distillation results. A better trade-off between overfitting and knowledge accumulation could be achieved by using a smaller learning rate and initializing from the previous iteration's distilled model, rather than resetting to the base model each time.
4. **Experience format**: The current experience is structured as a fixed triplet, which may not capture more complex multi-step reasoning strategies or contextual dependencies. Future work could explore more flexible experience formats to support a wider range of reasoning patterns.