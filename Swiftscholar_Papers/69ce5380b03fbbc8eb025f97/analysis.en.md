# 1. Bibliographic Information

## 1.1. Title
The title of the paper is **"Online Reasoning Calibration: Test-Time Training Enables Generalizable Conformal LLM Reasoning"**. The central topic revolves around improving the efficiency and reliability of Large Language Models (LLMs) during reasoning tasks. Specifically, it addresses the problem of "test-time scaling"—where models spend excessive computational resources generating long chains of thought—by introducing a calibration framework that allows the model to stop reasoning early when confident, while maintaining rigorous statistical guarantees on correctness.

## 1.2. Authors
The authors are **Cai Zhou**, **Zekai Wang**, **Menghua Wu**, **Qianyu Julie Zhu**, **Flora C. Shi**, **Chenyu Wang**, **Ashia Wilson**, **Tommi Jaakkola**, and **Stephen Bates**.

*   **Research Backgrounds & Affiliations:** The authors are affiliated with prestigious institutions, primarily the **Massachusetts Institute of Technology (MIT)**. Specifically, their affiliations include:
    *   **MIT EECS:** Department of Electrical Engineering and Computer Science.
    *   **MIT CSAIL:** Computer Science and Artificial Intelligence Laboratory.
    *   **boto:** A research organization (boto Dec Systs).
    *   **MIT CSE:** Computational Science and Engineering.
*   The group includes experts in machine learning theory (e.g., Tommi Jaakkola, Stephen Bates) and deep learning systems, indicating a strong foundation in both theoretical rigor and practical application.

## 1.3. Journal/Conference
The paper is currently available as a preprint on **arXiv** (arXiv:2604.01170). While the provided metadata lists a publication date of April 1, 2026, the source link confirms it is an arXiv submission. arXiv is a highly reputable open-access archive for scientific papers in fields like physics, mathematics, and computer science (specifically the AI/ML sub-sections). Publication here signifies that the work has undergone initial screening but may not yet have completed peer review for a specific conference or journal, though it represents cutting-edge research.

## 1.4. Publication Year
According to the provided metadata, the publication date is **2026**.

## 1.5. Abstract
The paper introduces **Online Reasoning Calibration (ORCA)**, a framework designed to address the inefficiency of modern LLMs that rely on massive "test-time scaling" (generating many samples or long reasoning chains) to solve difficult tasks. The authors identify that current models are "miscalibrated"—they do not accurately know when their intermediate reasoning steps are correct—leading to wasted computation.

*   **Research Objective:** To enable LLMs to allocate compute adaptively (stop early when confident) while providing statistical guarantees on the quality of the answer, even when deployed in new environments (distribution shift).
*   **Core Methodology:** ORCA combines **Test-Time Training (TTT)** and **Conformal Prediction**. It uses a meta-learning procedure (outer loop) to learn how to initialize a calibration module, and an online update procedure (inner loop) that adapts this module for every specific input instance during reasoning.
*   **Main Results:** ORCA significantly improves efficiency. For example, on the Qwen2.5-32B model, it saves up to **47.5%** of compute on in-distribution tasks and **67.0%** on out-of-distribution tasks (like MATH-500) while keeping error rates low.
*   **Key Conclusions:** The approach is generalizable across different model families (Qwen, Llama) and benchmarks, proving that dynamically updated calibration modules are superior to static ones.

## 1.6. Original Source Link
The official source is the arXiv preprint server.
*   **Link:** https://arxiv.org/abs/2604.01170
*   **PDF Link:** https://arxiv.org/pdf/2604.01170
*   **Status:** Preprint.

# 2. Executive Summary

## 2.1. Background & Motivation
*   **Core Problem:** Large Language Models (LLMs) have achieved remarkable success in complex reasoning tasks (e.g., mathematics, coding) by scaling "test-time compute." This involves strategies like generating multiple parallel solutions or extending the "Chain of Thought" (CoT). However, this comes at a prohibitive cost. The root cause identified by the authors is **miscalibration**: post-trained LLMs are poor at judging whether their current intermediate reasoning state is correct. Consequently, systems often use fixed, handcrafted parameters (e.g., "always generate 100 steps") which are inefficient—wasting compute on easy problems and potentially failing on hard ones.
*   **Importance & Challenges:** Efficient reasoning is critical for deploying powerful AI in real-world applications where latency and cost matter. Existing calibration methods (like "Thought Calibration") often rely on "static" probes—fixed models trained to predict correctness. These static probes fail when the distribution of data changes at deployment (e.g., different prompt styles or reasoning patterns) because they cannot adapt to the specific instance being solved.
*   **Entry Point:** The authors propose treating the calibration process itself as a learning problem that happens *at test time*. Instead of a fixed model, they use a mechanism that updates its internal state ("fast weights") *while* the LLM is reasoning, adapting specifically to the current problem instance.

## 2.2. Main Contributions / Findings
*   **Primary Contributions:**
    1.  **ORCA Framework:** A novel framework integrating **Test-Time Training (TTT)** with **Conformal Prediction**. It introduces a calibration probe that updates its weights online during the generation of a reasoning chain.
    2.  **Meta-Learning for Calibration:** A bilevel optimization approach where an "outer loop" learns a good initialization for the probe, and an "inner loop" adapts it to specific inputs. This ensures the updates are stable and transferable.
    3.  **Risk-Controlled Stopping:** The application of **Learn-then-Test (LTT)** to calibrate the *entire* adaptive procedure (including the online updates). This provides finite-sample statistical guarantees that the error rate will stay below a user-defined threshold $\delta$.
*   **Key Findings:**
    *   ORCA achieves substantial compute savings (up to 67% on MATH-500) compared to static baselines while strictly adhering to risk constraints.
    *   The method generalizes well to Out-Of-Distribution (OOD) data, a notorious failure mode for static calibration methods.
    *   The "no-QK" variant (a simple linear probe with online updates) is surprisingly effective and robust, requiring very few parameters.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several key concepts in machine learning and LLM inference:

*   **Chain of Thought (CoT):** A prompting technique where an LLM is instructed to generate intermediate reasoning steps before producing a final answer. This helps the model break down complex problems.
*   **Test-Time Scaling:** The strategy of spending more computational resources (tokens, time, number of samples) during the inference phase to improve the quality of the output, rather than relying solely on a larger pre-trained model.
*   **Miscalibration:** A model is miscalibrated if its predicted confidence does not match the actual probability of being correct. For example, a model might assign 90% confidence to an answer that is actually wrong 50% of the time.
*   **Test-Time Training (TTT):** A paradigm where model parameters are updated during the *inference* phase on a per-input basis. Unlike standard fine-tuning which happens offline on a dataset, TTT adapts the model to the specific data instance currently being processed.
    *   **Fast Weights:** In TTT, "fast weights" refer to the parameters that are updated quickly and frequently (e.g., at every token step) during inference, distinct from the "slow weights" of the base model which remain frozen.
*   **Conformal Prediction:** A statistical framework for constructing prediction sets that guarantee a certain level of coverage (e.g., "the true answer is in this set 95% of the time"). It is distribution-free, meaning it doesn't rely on strong assumptions about the data distribution, only on exchangeability (the data points are interchangeable).
*   **Learn-then-Test (LTT):** A specific calibration method related to conformal prediction. Instead of just creating a set of possible answers, LTT calibrates a *decision rule* (like "stop if confidence > threshold"). It tests a sequence of thresholds on a calibration set to select the most aggressive one that still satisfies a risk constraint.
*   **Exchangeability:** A statistical property of a sequence of data points where the joint probability distribution is invariant under permutation. It is a weaker assumption than "Independent and Identically Distributed" (i.i.d.) but sufficient for many conformal prediction guarantees.

## 3.2. Previous Works
The paper builds upon several strands of research:

*   **Efficient Test-Time Scaling:** Works like *Self-Consistency* (Wang et al., 2022) and *Thought Calibration* (Wu et al., 2025) aim to reduce compute. Thought Calibration is a direct baseline; it uses a static linear probe to predict if a reasoning step is correct and stops based on a calibrated threshold. However, it assumes a fixed inference procedure.
*   **Test-Time Training:** The seminal work by *Sun et al. (2020)* introduced TTT as a way to adapt models to distribution shifts using a self-supervised objective (often reconstruction). *Sun et al. (2024)* extended this to RNNs and Transformers. The current paper diverges by using TTT not for reconstruction or improving prediction accuracy directly, but for **calibration** (predicting correctness).
*   **Conformal Prediction & Risk Control:** *Angelopoulos & Bates (2021)* and *Angelopoulos et al. (2021)* laid the groundwork for LTT. The paper applies these theoretical tools to the dynamic, non-stationary process of LLM reasoning.

## 3.3. Technological Evolution
The field has evolved from simple "greedy decoding" to complex "search" and "sampling" strategies (like Monte Carlo Tree Search). As these strategies became more expensive, the focus shifted to **efficiency** (stopping early). Early attempts at stopping used heuristics. The field then moved to **static calibration** (training a probe on a dataset). The current paper represents the next step: **adaptive calibration**, where the calibration mechanism itself learns and changes during the generation process, leveraging meta-learning to ensure this adaptation is safe and generalizable.

## 3.4. Differentiation Analysis
The core difference between ORCA and prior work (specifically *Thought Calibration*) is the **dynamic nature of the calibration module**.
*   **Prior Work (Static):** A probe $f(\phi; W)$ is trained once. At inference, for every step $t$, it calculates a score `s_t = f(\phi_t; W)`. The weights $W$ never change.
*   **ORCA (Dynamic):** The probe has weights $W_t$ that change at every step: $W_t = W_{t-1} - \eta \nabla \ell$. The probe "learns" what the current problem's reasoning pattern looks like as it unfolds. Furthermore, ORCA calibrates this *entire adaptive algorithm* using LTT, whereas prior work might only calibrate the static scores.

# 4. Methodology

## 4.1. Principles
The core principle of ORCA is to treat calibration as an **adaptive prediction problem**. Instead of relying on a fixed model to judge correctness, ORCA uses a lightweight module that "learns on the fly."
*   **Intuition:** As an LLM generates a solution, the reasoning pattern evolves. A static probe might miss subtle shifts indicating a correct answer. An adaptive probe can track the "baseline" behavior of the current instance and detect deviations (novelty) that signal a correct solution.
*   **Theoretical Basis:** The method relies on **Bilevel Optimization** (Meta-Learning) for training and **Conformal Prediction** (specifically LTT) for risk control.
    1.  **Inner Loop:** Adapts the probe to the specific input instance during inference to minimize a calibration loss.
    2.  **Outer Loop:** Meta-trains the initialization and feature mappings so that the inner loop updates are effective and stable across different tasks.
    3.  **Risk Control:** Treats the combination of (LLM + Adaptive Probe + Stopping Rule) as a black-box algorithm and calibrates its stopping threshold to guarantee a maximum error rate $\delta$.

## 4.2. Core Methodology In-depth
The methodology consists of three distinct phases: Setup, Training (Outer Loop), and Calibration/Inference (Inner Loop + LTT).

### 4.2.1. Setup and Notation
We consider an input prompt $x$. The LLM generates a reasoning trajectory step-by-step. At step $t$, the model has produced a thought prefix $\bar{y}_t = [y^{(1)}, \dots, y^{(t)}]$.
*   **Hidden State:** We extract a hidden representation $\phi_t \in \mathbb{R}^{d_\phi}$ from the base LLM (e.g., the mean-pooled last layer state).
*   **Probe:** A function $f(\cdot; W)$ maps the hidden state to a confidence score $s_t \in [0, 1]$.
*   **Fast Weights:** The parameters $W$ of the probe are not fixed; they are "fast weights" $W_t$ that are updated online.

### 4.2.2. Inner Loop: Online-Adaptive Probe
This is the inference-time adaptation mechanism. For each step $t$ in the reasoning chain, the probe follows a "score-then-update" protocol.

**Step 1: Scoring**
First, the probe calculates a confidence score for the current step using the weights accumulated from previous steps ($W_{t-1}$).
$$
s_t = f(\phi_t; W_{t-1})
$$
*   $s_t$: The predicted probability that the current answer attempt is correct.
*   $\phi_t$: The hidden state of the LLM at step $t$.
*   $W_{t-1}$: The fast weights from the previous step.

**Step 2: Inner-Loop Loss**
The probe evaluates its performance against a label $C_t$. The label $C_t$ can be:
*   **Supervised:** `1` if the answer is correct (requires ground truth), `0` otherwise.
*   **Consistent:** `1` if the answer matches the final answer at the end of the budget, `0` otherwise.
    The loss used is the **Brier Score** (mean squared error), which is a proper scoring rule for probabilities.
$$
\ell(W_{t-1}; \phi_t) = (s_t - C_t)^2
$$
*   $\ell$: The loss function.
*   $C_t$: The target label (0 or 1).

**Step 3: Weight Update**
The probe updates its weights using online gradient descent to minimize this loss.
$$
W_t = W_{t-1} - \eta \nabla_W \ell(W_{t-1}; \phi_t)
$$
*   $\eta$: The inner-loop learning rate.
*   $\nabla_W \ell$: The gradient of the loss with respect to the weights.

    This update rule allows the probe to adapt. For example, if the probe consistently sees "wrong" patterns ($C_t=0$) with high confidence ($s_t \approx 1$), the weights will adjust to lower confidence for those patterns in the future.

### 4.2.3. Outer Loop: Meta-Learning
Training the probe is not standard supervised learning. We must learn the *initialization* $W_0$ and the feature mappings (projections) such that the inner-loop updates are effective. This is a bilevel optimization problem.

The paper introduces two variants of the architecture:
1.  **No-QK:** The probe operates directly on $\phi_t$. $f(u; W) = \sigma(W \cdot u + b)$.
2.  **QK (Query-Key):** Uses learned projections $\theta_Q$ and $\theta_K$ to map the high-dimensional $\phi_t$ to a lower-dimensional space before scoring and updating.
    *   Score: `s_t = f(\theta_Q \phi_t; W_{t-1})`
    *   Loss: $\ell(W; \phi_t) = (f(\theta_K \phi_t; W) - C_t)^2$
    *   Here, $\theta_Q$ determines what we "look at" to score, and $\theta_K$ determines what we "look at" to learn/update.

**The Outer Objective**
Let $\Theta_{outer} = (\theta_{Q,K}, W_0, \eta)$ be the "slow weights" we want to learn. We minimize the expected loss over the training data, but crucially, the loss depends on the trajectory of fast weights $\{W_t\}$ generated by the inner loop.
$$
\min_{\Theta_{outer}} \mathbb{E}_{x \sim \mathcal{D}_{train}} \mathcal{L}_{outer}\left(x, \{W_t\}_{t=1}^T; \Theta_{outer}\right)
$$
Subject to the constraint that $W_t$ are generated by the inner loop update rule. The outer loss is defined as the sum of Brier scores over the trajectory:
$$
\mathcal{L}_{outer}\left(x, \{W_t\}_{t=1}^T; \Theta_{outer}\right) := \sum_{t=1}^T (s_t - C_t^{true})^2
$$
*   **Training Process:** For each training input, we "unroll" the inner loop for $T$ steps. We calculate the outer loss. Then, we differentiate through the entire unrolled computation (backpropagation through time) to update $\Theta_{outer}$. This teaches the model *how to learn* effectively at test time.

    The following figure (Figure 1 from the original paper) illustrates the framework, showing the separation between the outer loop training and the calibration/inference phase.

    ![Figure 1: Framework of Online Reasoning Calibration (ORCA).](images/1.jpg)
    *该图像是示意图，展示了在线推理校准（ORCA）的框架。图中分为两个主要阶段：外循环TTT训练阶段和校准与推理阶段。在外循环阶段，显示了步骤标签、探测分数、探测权重和步骤嵌入的关系，而在校准与推理阶段，展示了输出如何依赖于停止阈值$\lambda^*$。整体结构帮助理解模型训练和推理的过程。*

### 4.2.4. Risk-Controlled Conformal Reasoning
Once the probe is meta-trained, we need a threshold $\lambda$ to decide when to stop. We cannot just pick a threshold that looks good; we need a statistical guarantee. ORCA uses **Learn-then-Test (LTT)**.

**The Deployed Procedure**
For a given threshold $\lambda$, the full deployed procedure $\mathcal{A}_\lambda$ is:
1.  Initialize $W_0$.
2.  For $t=1 \dots T$:
    *   Get $\phi_t$ from LLM.
    *   Calculate score `s_t = f(\phi_t; W_{t-1})`.
    *   **Stop** if $s_t \ge \lambda$. Output answer.
    *   Else, update $W_t$ using the inner loop rule (with pseudo-target $C_t=0$ usually at inference).

        We define the **Stopping Time** $\tau_\lambda(x)$ as the first step where the score exceeds the threshold:
$$
\tau_\lambda(x) := \min \{t \le T : s_t(x) \ge \lambda\}
$$
And the **Decision Rule** output $A_\lambda(x)$ is the answer at that stopping time:
$$
A_\lambda(x) := \text{ans}(y_{\tau_\lambda(x)})
$$

**LTT Calibration**
We have a grid of thresholds $\Lambda = \{\lambda_1 > \dots > \lambda_m\}$ (from conservative to aggressive). We want to pick the smallest $\lambda$ (most aggressive) such that the risk (error rate) is $\le \delta$.

For each $\lambda_j$, we run the deployed procedure on a calibration set and calculate the empirical risk $\hat{R}_n(\lambda_j)$. We test the null hypothesis $H_j: \mathbb{E}[R(y_{\tau_{\lambda_j}})] \ge \delta$ (the risk is too high).

We calculate a p-value using the Binomial distribution (since errors are 0/1):
$$
p_j^{BT} := \mathbb{P}(\text{Binom}(n, \delta) \le n \hat{R}_n(\lambda_j))
$$
*   $n$: Size of calibration set.
*   $\hat{R}_n$: Empirical error rate.

    We apply **Fixed-Sequence Testing** (FST): check $H_1, H_2, \dots$ in order. If $p_j \le \epsilon$ (significance level), we reject $H_j$ (meaning the risk is acceptable) and move to the next (more aggressive) threshold. We stop at the first failure. The selected threshold $\lambda^*$ guarantees:
$$
\mathbb{P}(\mathbb{E}[R(y_{\tau_{\lambda^*}})] \le \delta) \ge 1 - \epsilon
$$
This guarantee holds for the *entire* adaptive procedure, including the online weight updates, because LTT treats the algorithm as a black box and relies on exchangeability of the calibration data.

# 5. Experimental Setup

## 5.1. Datasets
The authors construct a robust experimental setup involving both in-distribution and out-of-distribution data.

*   **Training Corpus (In-Distribution):** A combined dataset of 5,000 math problems.
    *   **Sources:** s1K (1,000 problems), OpenR1 (2,000 problems), DeepMath (2,000 problems).
    *   **Split:** 3,000 for training, 1,000 for calibration, 1,000 for testing.
    *   **Reasoning Trajectories:** Generated by **DeepSeek-R1-671B**, a powerful reasoning model.
    *   **Labels:** Step-level labels indicating correctness are generated by a teacher model (Qwen-3-32B or GPT-4.1).
*   **Out-Of-Distribution (OOD) Benchmarks:** To test generalization, the model is evaluated zero-shot on:
    *   **MATH-500:** A standard high-school math benchmark (500 problems).
    *   **GPQA-Diamond:** Graduate-level QA (198 problems).
    *   **AIME:** American Invitational Mathematics Examination (2024, 2025, 2026 editions).

## 5.2. Evaluation Metrics
The paper evaluates performance based on two primary metrics that form a trade-off controlled by the risk level $\delta$.

1.  **Compute Savings**
    *   **Conceptual Definition:** This metric quantifies the efficiency gain achieved by early stopping. It measures the fraction of the total reasoning steps (or tokens) that the model did *not* have to generate because it stopped early. Higher savings mean the model is more efficient.
    *   **Mathematical Formula:**
        $$ \text{Savings} = 1 - \frac{\bar{t}_{stop}}{\bar{t}_{total}} $$
    *   **Symbol Explanation:**
        *   $\bar{t}_{stop}$: The average number of steps taken before stopping (or the total budget if not stopped).
        *   $\bar{t}_{total}$: The average total number of steps in the full reasoning trajectory (the budget).

2.  **Error Rate**
    *   **Conceptual Definition:** This metric measures the reliability of the stopping decision. It is the fraction of problems where the model stopped early and outputted an *incorrect* answer. A high error rate means the model is stopping prematurely (overconfident). The goal is to keep this below the risk threshold $\delta$.
    *   **Mathematical Formula:**
        $$ \text{Error Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{stopped}_i \land \text{incorrect}_i) $$
    *   **Symbol Explanation:**
        *   $N$: Total number of test problems.
        *   $\mathbb{I}(\cdot)$: Indicator function (1 if condition is true, 0 otherwise).
        *   $\text{stopped}_i$: Boolean indicating if the model stopped before the budget for problem $i$.
        *   $\text{incorrect}_i$: Boolean indicating if the output answer for problem $i$ is wrong.

## 5.3. Baselines
The primary baseline is the **Static Probe** (from *Wu et al., 2025*).
*   **Description:** A standard linear probe (PCA + Logistic Regression) trained on the training data to predict correctness from the hidden state $\phi_t$.
*   **Why Representative:** It represents the state-of-the-art in static calibration for reasoning. It does not update weights online and does not use meta-learning. Comparing against this isolates the benefits of the "Online" and "Meta-learning" components of ORCA.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The results demonstrate that ORCA significantly outperforms the static baseline across all settings (In-Distribution, OOD, and Cross-Model).

*   **In-Distribution Performance:** At a risk level of $\delta=0.1$ (allowing 10% error), the **TTT no-QK** probe achieves **47.5%** savings compared to **38.0%** for the static baseline. This is a **24.9% relative improvement**. The error rates for both methods remain within the $\delta$ budget, validating the risk control.
*   **OOD Generalization:** This is the strongest result. On MATH-500, the static baseline only saves **24.8%**. ORCA (TTT QK) saves **67.0%**, a massive **2.7x improvement**. This confirms that the online adaptation allows the probe to handle reasoning patterns it hasn't seen during training.
*   **Cross-Model:** ORCA works consistently across different base LLMs (Qwen2.5, QwQ, Llama-3.3), suggesting the method is model-agnostic.

## 6.2. Data Presentation (Tables)

The following are the results from Table 2 of the original paper, showing In-Distribution early-stopping performance:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">δ = 0.05</th>
<th colspan="2">δ = 0.1</th>
<th colspan="2">δ = 0.15</th>
<th colspan="2">δ = 0.2</th>
</tr>
<tr>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9"><strong>Supervised labels</strong></td>
</tr>
<tr>
<td>Static Probe</td>
<td>.220</td>
<td>.055</td>
<td>.380</td>
<td>.105</td>
<td>.512</td>
<td>.159</td>
<td>.625</td>
<td>.208</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.282</td>
<td>.053</td>
<td>.475</td>
<td>.110</td>
<td>.575</td>
<td>.152</td>
<td>.673</td>
<td>.192</td>
</tr>
<tr>
<td>TTT QK (d=128)</td>
<td>.233</td>
<td>.046</td>
<td>.414</td>
<td>.103</td>
<td>.560</td>
<td>.150</td>
<td>.674</td>
<td>.204</td>
</tr>
<tr>
<td colspan="9"><strong>Consistent labels (no ground truth)</strong></td>
</tr>
<tr>
<td>Static Probe</td>
<td>.166</td>
<td>.049</td>
<td>.345</td>
<td>.098</td>
<td>.483</td>
<td>.156</td>
<td>.573</td>
<td>.197</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.220</td>
<td>.045</td>
<td>.407</td>
<td>.096</td>
<td>.529</td>
<td>.141</td>
<td>.644</td>
<td>.193</td>
</tr>
<tr>
<td>TTT QK (dh=128)</td>
<td>.232</td>
<td>.064</td>
<td>.397</td>
<td>.113</td>
<td>.524</td>
<td>.150</td>
<td>.629</td>
<td>.187</td>
</tr>
</tbody>
</table>

The following are the results from Table 3 of the original paper, showing Out-of-Distribution generalization:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">MATH-500</th>
<th colspan="2">GPQA</th>
<th colspan="2">AIME'24</th>
<th colspan="2">AIME'25</th>
<th colspan="2">AIME'26</th>
</tr>
<tr>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
<th>Sav.</th>
<th>Err.</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="11"><strong>Supervised labels</strong></td>
</tr>
<tr>
<td>Static Probe</td>
<td>.248</td>
<td>.008</td>
<td>.643</td>
<td>.270</td>
<td>.158</td>
<td>.050</td>
<td>.139</td>
<td>.000</td>
<td>.147</td>
<td>.050</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.637</td>
<td>.023</td>
<td>.715</td>
<td>.300</td>
<td>.293</td>
<td>.150</td>
<td>.265</td>
<td>.056</td>
<td>.198</td>
<td>.050</td>
</tr>
<tr>
<td>TTT QK (d=128)</td>
<td>.670</td>
<td>.021</td>
<td>.665</td>
<td>.210</td>
<td>.295</td>
<td>.100</td>
<td>.258</td>
<td>.000</td>
<td>.134</td>
<td>.050</td>
</tr>
<tr>
<td colspan="11"><strong>Consistent labels (no ground truth)</strong></td>
</tr>
<tr>
<td>Static Probe</td>
<td>.239</td>
<td>.004</td>
<td>.602</td>
<td>.328</td>
<td>.118</td>
<td>.033</td>
<td>.101</td>
<td>.000</td>
<td>.147</td>
<td>.100</td>
</tr>
<tr>
<td>TTT no-QK</td>
<td>.555</td>
<td>.012</td>
<td>.598</td>
<td>.318</td>
<td>.141</td>
<td>.033</td>
<td>.166</td>
<td>.067</td>
<td>.154</td>
<td>.067</td>
</tr>
<tr>
<td>TTT QK (dh=128)</td>
<td>.637</td>
<td>.016</td>
<td>.653</td>
<td>.328</td>
<td>.185</td>
<td>.033</td>
<td>.139</td>
<td>.000</td>
<td>.092</td>
<td>.000</td>
</tr>
</tbody>
</table>

## 6.3. Ablation Studies / Parameter Analysis
The authors perform several ablation studies to verify the necessity of the TTT mechanism.

*   **TTT vs. Standard Training:** Table 5 compares "Full TTT" against "Standard" supervised training (where the same architecture is trained without unrolling/meta-learning) and "No meta-training" (random weights).
    *   **Result:** Standard training performs poorly (23.9% savings vs 47.5% for TTT). This proves that simply having the architecture isn't enough; the *meta-learning objective* (learning to initialize for fast updates) is the key driver of performance.
    *   **Result:** Random initialization with online updates (No meta-training) also performs poorly (25.4%), proving that the learned initialization $W_0$ is crucial.

*   **Architecture Variants:** Table 6 explores adding LayerNorm, Residual connections, or MLPs to the probe.
    *   **Result**:
        *   **no-QK** remains the strongest baseline for In-Distribution and AIME.
        *   **QK + LayerNorm** performs best on MATH and GPQA.
        *   **MLP** achieves the highest savings on MATH-500 (71.7%).
    *   **Conclusion:** While more complex architectures can squeeze out more performance on specific benchmarks, the simple **no-QK** variant offers the best balance of performance, stability, and parameter efficiency (only 5.1K parameters).

*   **Risk vs. Savings Trade-off:** Figure 2 (below) visualizes the trade-off across different risk levels $\delta$. It shows that ORCA consistently achieves higher savings than the static baseline for any given error rate.

    ![Figure 2: Compute savings vs. risk tolerance $\\delta$ for supervised (left) and consistent (right) labels (Qwen2.5-32B). TTT no-QK consistently outperforms the baseline across all risk levels, with the largest gap at low $\\delta$ .](images/2.jpg)
    *该图像是一个图表，展示了在监督标签（左）和一致性标签（右）情况下的计算节省与风险容忍度 `oldsymbol{ heta}` 之间的关系。TTT no-QK 在所有风险水平下均优于基线，特别是在低风险容忍度时差距最大。*

*   **Calibration Quality:** Figure 3 plots the actual error rate against the target risk $\delta$. The points lie close to the diagonal $y=x$, confirming that the LTT calibration works correctly—the actual error respects the user-defined risk budget.

    ![Figure 3: Actual error rate vs. target risk $\\delta$ (supervised, Qwen2.5-32B). All methods track the diagonal, confirming valid risk control. Points below the diagonal satisfy the LTT guarantee.](images/3.jpg)
    *该图像是图表，展示了实际错误率与目标风险 $δ$ 之间的关系（监督，Qwen2.5-32B）。不同方法的表现由曲线表示，图中标注了过于自信和良好校准的区域，所有方法都跟踪对角线，确认了风险控制的有效性。图中点位于对角线下方满足 LTT 保证。*

*   **Score Trajectories:** Figure 5 compares the behavior of the Static Probe vs. TTT no-QK on a single test problem.
    *   **Static Probe:** The score rises slowly and never crosses the threshold, resulting in 0% savings (runs to completion).
    *   **TTT no-QK:** The score starts higher (due to meta-learned init) and adapts. It detects the "breakthrough" (correct answer) quickly, crosses the threshold, and stops at step 22 (saving 41%).
    *   This illustrates the mechanism: TTT effectively detects the *change* in reasoning state associated with finding the correct answer.

        ![Figure 5: Probe score trajectories for a test problem (Qwen2.5-32B, $\\delta \\mathrm { = } 0 . 1$ . The green line marks the first correct step. The static probe (top) never crosses its threshold and saves $0 \\%$ . The TTT no-QK probe (bottom) crosses the threshold at step 22 and saves $4 1 \\%$ .](images/5.jpg)
        *该图像是图表，展示了测试问题（Qwen2.5-32B, $ heta = 0.1$）的探测评分轨迹。静态探测器在第38步停止，未能跨越阈值，保存率为0%；而TTT无QK探测器在第22步过阈，保存率为41%。绿色线表示第一次正确步骤。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that **Online Reasoning Calibration (ORCA)** is a powerful framework for efficient LLM inference. By combining **Test-Time Training (TTT)** with **Conformal Prediction**, the authors created a system that:
1.  **Adapts Online:** It adjusts its calibration parameters for every specific input, allowing it to handle distribution shifts and varying reasoning patterns.
2.  **Is Theoretically Sound:** It provides finite-sample risk guarantees (via LTT) on the entire adaptive procedure.
3.  **Is Efficient:** It achieves significant compute savings (up to 67% on OOD data) over static baselines while maintaining strict error rate controls.

## 7.2. Limitations & Future Work
*   **Marginal Guarantees:** The theoretical guarantees provided are *marginal* (they hold on average over the data distribution), not *conditional* (they don't guarantee correctness for a specific prompt). This is a standard limitation of conformal prediction methods.
*   **Exchangeability Assumption:** The validity relies on the exchangeability of the calibration data and test data. If the deployment environment introduces severe non-exchangeable dependencies (e.g., time-series data where order matters), the guarantees might not hold without adjustment.
*   **Compute Overhead:** While the method saves tokens by stopping early, it introduces a small computational overhead for the inner-loop gradient updates at every step. However, the authors note this overhead is negligible compared to the cost of generating LLM tokens.

## 7.3. Personal Insights & Critique
*   **Innovation:** The most insightful aspect is the reframing of "calibration" as a "learning problem." Most works treat calibration as a static regression task. ORCA recognizes that the *definition* of a "good" reasoning state changes from problem to problem, and thus the calibration model must be capable of learning *during* the process.
*   **Practicality:** The "no-QK" variant being the recommended default is a strong point for practical adoption. It requires very few parameters (5.1K) and no complex projections, making it easy to integrate into existing systems without massive memory overheads.
*   **Robustness:** The OOD results are particularly impressive. The fact that a model trained on a generic 5K math corpus can generalize so well to AIME and GPQA suggests that the meta-learning component is capturing fundamental features of "correctness" rather than overfitting to the training data distribution.
*   **Future Potential:** This approach could be extended beyond early stopping. For example, the adaptive probe could guide the *search* process (e.g., in Monte Carlo Tree Search) by identifying which branches of reasoning are most promising, effectively acting as a dynamic value function.