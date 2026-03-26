# 1. Bibliographic Information
## 1.1. Title
The central topic of the paper is **response homogenization (called the "alignment tax") in RLHF-aligned large language models (LLMs)**, its harmful impact on sampling-based uncertainty estimation (UQ), and a practical cost-efficient cascade solution for robust UQ on aligned models. The full title is *The Alignment Tax: Response Homogenization in Aligned LLMs and Its Implications for Uncertainty Estimation*.
## 1.2. Authors
The sole author is **Mingyi Liu**, an independent researcher with GitHub handle @DigitLion. No additional institutional affiliations are listed.
## 1.3. Journal/Conference
As of the publication timestamp (2026-03-25), the paper is a preprint hosted on arXiv, and has not yet been published in a peer-reviewed journal or conference. arXiv is the most widely used preprint server for computer science research, particularly for LLM and NLP work, and allows rapid dissemination of cutting-edge findings before formal peer review.
## 1.4. Publication Year
The paper was published (as a preprint) in **2026**.
## 1.5. Abstract
The paper's core research objective is to quantify the impact of LLM alignment on the reliability of uncertainty estimation methods, identify the causal driver of observed performance drops, and propose a practical mitigation. Key details:
1.  Core observation: RLHF-aligned LLMs exhibit *response homogenization*: 40-79% of TruthfulQA questions produce only a single semantic cluster across 10 independent sampled responses.
2.  Impact: On these affected questions, all sampling-based UQ methods have zero discriminative power (AUROC=0.500, equivalent to random guessing), while free per-token entropy retains meaningful signal (AUROC=0.603).
3.  Task dependence: The alignment tax is far weaker on mathematical reasoning (GSM8K: token entropy achieves AUROC=0.724, Cohen's d=0.81, large effect size).
4.  Causal ablations: Alignment is confirmed as the cause: base (unaligned) models have 1.0% single-cluster rate (SCR) vs 28.5% for the aligned instruct variant (p < 1e-6). The effect is localized to Direct Preference Optimization (DPO) training, not Supervised Fine-Tuning (SFT).
5.  Replication: The effect generalizes across 4 model families, 5 benchmarks, and 3 model scales (3B-14B parameters).
6.  Solution: A cheapest-first cascade (UCBD) over orthogonal uncertainty signals achieves 57% cost savings vs running all detectors in parallel, and raises GSM8K accuracy from 84.4% to 93.2% at 50% coverage via selective prediction.
## 1.6. Original Source Link
- Official preprint landing page: https://arxiv.org/abs/2603.24124v1 (preprint status)
- Full PDF link: https://arxiv.org/pdf/2603.24124v1

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
LLM-powered AI agents need to reliably detect when they do not know an answer (uncertainty estimation) to avoid hallucinations in high-stakes use cases (e.g., healthcare, legal advice). However, state-of-the-art sampling-based UQ methods (which rely on measuring diversity across multiple sampled responses to the same query) perform poorly on aligned LLMs, especially for factual question answering (QA) tasks. Prior work had observed that RLHF reduces output diversity (mode collapse) but did not systematically measure how this affects UQ reliability, isolate which alignment stage causes the problem, or provide a practical, cost-effective mitigation for deployments.
### Gap in Prior Research
1.  Prior work framed RLHF-induced diversity loss exclusively as a generation quality issue (e.g., less creative outputs, less stylistic variation), not as a structural failure mode for UQ.
2.  No studies had isolated whether SFT or preference optimization (RLHF/DPO) drives the diversity loss that breaks UQ.
3.  Existing UQ methods are almost all single-signal, and fail to account for the task-dependent performance gap (strong performance on math, near-chance on factual QA) observed in aligned models.
### Innovative Entry Point
The authors observe that aligned LLMs often produce semantically identical responses even when sampling at high temperature, which eliminates the inter-response diversity that sampling-based UQ methods rely on. They frame this as an "alignment tax": a measurable degradation in UQ capability caused by alignment training, and propose a multi-signal cascade that leverages orthogonal, low-cost uncertainty signals to work around this failure mode.
## 2.2. Main Contributions / Findings
The paper makes three core, empirically validated contributions:
1.  **Alignment Tax Diagnosis**: Aligned LLMs exhibit response homogenization on 40-79% of TruthfulQA factual queries, rendering all sampling-based UQ methods completely uninformative on these queries (AUROC=0.500). This effect is independent of clustering method, sample size, temperature, and decoding strategy, and is driven primarily by DPO training (not SFT), with severity varying by up to 50x across model families and alignment recipes (0.5% to 28.5% SCR).
2.  **Task-Dependent Uncertainty Structure**: Uncertainty signal performance varies drastically across tasks: token entropy has a near-zero effect size on factual QA (Cohen's d=0.07) but a large effect size on mathematical reasoning (d=0.81). No single UQ signal can cover all failure modes, motivating a multi-modal approach.
3.  **UCBD Cascade Architecture**: A cheapest-first cascade over weakly dependent, orthogonal uncertainty signals matches the performance of running all detectors in parallel while reducing cost by 57%. It enables selective prediction that raises GSM8K accuracy from 84.4% to 93.2% at 50% coverage, using only free logprob outputs for 50% of queries.

# 3. Prerequisite Knowledge & Related Work
This section explains all core concepts and prior work required to understand the paper, for a beginner audience.
## 3.1. Foundational Concepts
We define all key terms in order of appearance:
1.  **Reinforcement Learning from Human Feedback (RLHF)**: The dominant alignment pipeline for modern LLMs, consisting of three stages:
    1.  *Supervised Fine-Tuning (SFT)*: The base LLM is fine-tuned on high-quality instruction-response pairs to teach it to follow user instructions.
    2.  *Reward Model Training*: A separate model is trained to score LLM responses based on human preference rankings.
    3.  *Reinforcement Learning*: The SFT model is fine-tuned via reinforcement learning using the reward model's scores as the reward signal, with a KL divergence penalty to avoid drifting too far from the SFT model.
2.  **Direct Preference Optimization (DPO)**: A simplified alternative to RLHF that directly optimizes the LLM on pairwise preference data (which of two responses a human prefers) without a separate reward model or reinforcement learning loop. It is now the most widely used alignment method for open-source LLMs.
3.  **Uncertainty Estimation (UQ) for LLMs**: The task of quantifying how confident an LLM is in its generated response, to detect hallucinations or knowledge gaps. UQ methods fall into two broad categories:
    1.  *Sampling-based methods*: Generate multiple independent responses to the same query, and use the diversity across responses as a measure of uncertainty (higher diversity = higher uncertainty).
    2.  *Single-pass methods*: Compute uncertainty from a single forward pass of the model (e.g., using per-token log probabilities, hidden states) with no additional sampling cost.
4.  **Single-Cluster Rate (SCR)**: The key diagnostic metric introduced in the paper: the percentage of queries where all N sampled responses fall into a single semantic cluster (i.e., all responses mean the same thing, even if wording varies). SCR = 0% means full response diversity, SCR = 100% means complete homogenization.
5.  **Semantic Entropy (SE)**: The most widely used sampling-based UQ method, which clusters responses by semantic equivalence and computes entropy over cluster frequencies to measure uncertainty. SE = 0 when all responses are in a single cluster (no uncertainty signal).
6.  **Token Entropy**: A single-pass UQ metric that computes the average entropy of the model's next-token probability distribution across all generated tokens. It measures the model's internal per-step uncertainty, not inter-response diversity. Formula:
    $$H_t = -\sum_{v} P(v_t | v_{<t}) \log P(v_t | v_{<t})$$
    Where $P(v_t | v_{<t})$ is the model's predicted probability of token $v_t$ given all preceding tokens $v_{<t}$.
7.  **AUROC (Area Under the Receiver Operating Characteristic Curve)**: A standard evaluation metric for binary classification tasks (here, classifying responses as correct vs incorrect). AUROC ranges from 0.5 (random guessing, no discriminative power) to 1.0 (perfect classification).
8.  **Cohen's $d$**: A standard measure of effect size for the difference between the means of two groups. Values of 0.2 = small effect, 0.5 = medium effect, 0.8 = large effect. Formula:
    $$d = \frac{\mu_1 - \mu_2}{\sigma_{\text{pooled}}}$$
    Where $\mu_1, \mu_2$ are the means of the two groups, and $\sigma_{\text{pooled}}$ is the pooled standard deviation of the two groups.
## 3.2. Previous Works
The paper builds on three broad lines of prior research:
### 1. Single-Signal UQ Methods
- Token-level single-pass methods: LogTokU, PRO, Semantic Energy, all use per-token log probabilities or hidden states to compute uncertainty in a single forward pass, achieving AUROC 0.7-0.9 on individual benchmarks.
- Sampling-based methods: Semantic Entropy (SE), SelfCheckGPT, CoCoA, SINdex, all measure diversity across multiple sampled responses. SINdex, in particular, uses embedding-based clustering to measure semantic inconsistency, which the authors replicate in their experiments.
- Key limitation of all prior single-signal methods: They have structural blind zones: token entropy fails on subsets of factual QA where the model is "confidently wrong", while sampling-based methods fail entirely on homogenized queries.
### 2. Alignment and Mode Collapse
Prior work has documented that RLHF/DPO reduces output diversity:
- Kirk et al. (2024) showed RLHF reduces generalisation and output diversity.
- Saeidi et al. (2024) showed DPO concentrates probability mass on "safe" consistent responses.
- Azar et al. (2024) theoretically connected KL-regularized RLHF to distribution narrowing.
- Key limitation: All prior work framed this diversity loss as a generation quality issue, not as a structural failure for UQ.
### 3. Multi-Signal UQ and Cascades
- UniCR (Li et al., 2025) uses conformal risk control to fuse multiple uncertainty signals for formal coverage guarantees, but assumes all signals are pre-computed for every query.
- HalluGuard (Wang et al., 2025) uses model-internal NTK signals for hallucination detection.
- Key limitation: No prior work uses a cost-ordered cascade to minimize inference cost while maintaining performance, or accounts for the alignment tax failure mode.
## 3.3. Technological Evolution
The field of LLM UQ has evolved in three stages:
1.  **Early Stage (2020-2023)**: Focus on developing individual UQ signals, either sampling-based (SE, SelfCheckGPT) or single-pass (token entropy).
2.  **Intermediate Stage (2023-2025)**: Observation that sampling-based methods perform poorly on aligned LLMs, leading to development of better single-pass signals (LogTokU, PRO, hidden-state probes).
3.  **Current Stage (2025-present)**: Recognition that no single signal works across all tasks and failure modes, leading to multi-signal fusion and cascade approaches (like the UCBD cascade proposed in this paper).
    This paper's work sits at the transition between the intermediate and current stages: it provides a systematic diagnosis of why sampling-based methods fail on aligned models, and proposes a practical cost-efficient multi-signal solution.
## 3.4. Differentiation Analysis
Compared to prior work, this paper has three core differentiators:
1.  **Novel Framing**: It is the first work to frame RLHF/DPO-induced mode collapse as a structural UQ failure, rather than just a generation quality issue. It quantifies exactly how often sampling-based UQ becomes completely uninformative (40-79% of factual queries).
2.  **Causal Isolation**: It is the first work to isolate DPO (not SFT) as the primary driver of response homogenization, via stage-wise ablation across two independent alignment pipelines (Mistral/Zephyr and Llama/Tulu-3).
3.  **Practical Solution**: Unlike prior work that only proposes better single signals, the UCBD cascade is a deployable, cost-efficient solution that works with existing signals and achieves 57% cost savings vs parallel signal computation, with validated selective prediction gains.

# 4. Methodology
## 4.1. Principles
The methodology is built on three core intuitive principles:
1.  **Alignment only suppresses inter-response diversity, not internal per-token uncertainty**: Preference optimization (DPO/RLHF) trains the model to produce consistent, high-reward responses, which reduces semantic diversity across sampled outputs, but does not eliminate per-token computational uncertainty (the model's uncertainty over which exact token to generate next). This means single-pass token-level signals will retain discriminative power even when sampling-based methods fail completely.
2.  **Uncertainty signals are orthogonal and have different costs**: Different UQ signals capture fundamentally different failure modes (e.g., token entropy captures generation fluency uncertainty, embedding density captures out-of-distribution queries, freshness signals capture temporal knowledge gaps) and have drastically different inference costs (from free for token entropy to expensive for external retrieval+verification).
3.  **Weak dependence enables cost-efficient cascading**: If signals are only weakly correlated, we can order them by cost, trigger more expensive signals only when cheaper signals are uninformative, and achieve the same performance as running all signals in parallel at a fraction of the cost.
## 4.2. Core Methodology In-depth
We break down the methodology into three parts: the alignment tax definition, the five cognitive boundary detectors, and the UCBD cascade architecture, with all formulas exactly as presented in the original paper.
### 4.2.1. Alignment Tax Definition
The paper first formalizes the alignment tax as a label-free metric that can be computed for any model on any dataset without ground truth labels:
$$\mathrm{AT}(q) = 1 - \frac{\mathcal{D}_S(q)}{N}$$
Where:
- $\mathcal{D}_S(q)$ = the number of distinct semantic clusters from $N$ independent identically distributed (i.i.d.) sampled responses to query $q$ (generated at temperature $T=1.0$ by default)
- $N$ = number of sampled responses per query
- $\mathrm{AT}(q) = 1 - 1/N$ (max value) when all responses fall into a single semantic cluster (complete homogenization), at which point all sampling-based UQ methods have zero discriminative power by definition (semantic entropy = 0, regardless of whether the response is correct or incorrect)
- Intermediate values of $\mathrm{AT}(q)$ indicate partial diversity reduction, where sampling-based methods retain weakened but non-zero signal.
### 4.2.2. Five Cognitive Boundary Detectors
The authors define five orthogonal "cognitive boundaries" that capture different types of LLM knowledge gaps, ordered by increasing inference cost:
#### B1: Fluency Boundary (Free, zero additional cost)
This uses per-token entropy (computed from log probabilities that are a standard byproduct of LLM generation) to detect uncertain generation:
$$H_t = -\sum_{v} P(v_t | v_{<t}) \log P(v_t | v_{<t})$$
Where:
- $P(v_t | v_{<t})$ = the model's predicted probability of token $v$ at position $t$, given all preceding tokens $v_{<t}$
- The trigger threshold is $\tau_H$: if the average (or max) token entropy across the generated response exceeds $\tau_H$, the model is flagged as uncertain.
- Cost: Free, since log probabilities are already produced during generation.
#### B2: Density Boundary (Low cost, 1 embedding call)
This measures how close the query embedding is to the model's training data distribution, to detect out-of-distribution "knowledge desert" queries:
$$\rho(\mathbf{e}_q) = \frac{1}{k} \sum \cos(\mathbf{e}_q, \mathbf{e}_{n_j})$$
Where:
- $\mathbf{e}_q$ = the embedding vector of the input query
- $\mathbf{e}_{n_j}$ = the embedding vectors of the $k$ nearest neighbor queries from the model's training domain
- $\cos(\cdot)$ = cosine similarity between two embedding vectors
- Low density = the query is far from the model's training distribution, so the model is likely to hallucinate.
- Cost: ~2ms on consumer hardware, requires one embedding API call for opaque models.
#### B3: Freshness Boundary (Low cost, metadata check only)
This detects queries that require time-sensitive knowledge that may be outside the model's training cutoff:
$$F(k, t_q) = \exp(-\lambda(k) \cdot (t_q - t_k))$$
Where:
- $t_q$ = timestamp of the query
- $t_k$ = timestamp of the most recent training data for knowledge domain $k$
- $\lambda(k)$ = decay rate for domain $k$ (higher for fast-changing domains like news, lower for stable domains like math)
- Operationalization: At inference, this is triggered by detecting temporal entities (dates, terms like "current", "latest") in the query and comparing against the model's known training cutoff. No model inference is required.
- Cost: Near-zero, simple rule-based check.
#### B4: Association Rupture Boundary (Medium cost, KG query/entity embedding lookup)
This detects missing or incorrect factual associations in the model's knowledge, by checking if a predicted triple $(e_1, r, e_2)$ (e.g., (Paris, capital of, France)) should exist in a knowledge graph (KG) but does not, or vice versa.
- Operationalization: The paper uses a proxy of entity-pair embedding cosine distance to detect anomalous associations, without requiring a full KG.
- Cost: ~5ms per query, requires entity linking and embedding lookup.
#### B5: Grounding Boundary (High cost, external verification)
This uses external sources (retrieved documents, knowledge bases) to verify the correctness of the generated response, via natural language inference (NLI) entailment scoring against external evidence.
- Cost: High, requires retrieval API calls and NLI inference.
### 4.2.3. UCBD Cascade Architecture
The cascade runs the boundary detectors in order of increasing cost, exiting early when a detector confidently flags uncertainty, to minimize average inference cost.
#### Cost Bound Guarantee
The cascade is guaranteed to never cost more than running all detectors in parallel:
$$C_{\mathrm{cascade}} = \sum_{i=1}^k c_i \prod_{j=1}^{i-1} \beta_j \leq \sum_{i=1}^k c_i = C_{\mathrm{parallel}}$$
Where:
- $c_i$ = cost of boundary detector $i$, ordered such that $c_1 \leq c_2 \leq ... \leq c_k$
- $\beta_i$ = pass-through rate of detector $i$ (fraction of queries that are not flagged as uncertain by detector $i$, and proceed to the next more expensive detector)
- For the paper's implementation, 57.4% of queries are flagged at the free B1 stage, so only 42.6% proceed to more expensive detectors, leading to a 57% cost saving vs parallel computation.
#### Coverage Bound Guarantee
Under weak dependence between detectors (mutual information ≤ 0.02 bits, Pearson $|r| ≤ 0.12$), the combined coverage of the cascade is superadditive:
$$\mathrm{Coverage}(d_1 \cup ... \cup d_k) \approx 1 - \prod_{i=1}^k (1 - \alpha_i) \geq \max_i \alpha_i$$
Where $\alpha_i$ = coverage of detector $i$ (fraction of incorrect answers it correctly flags). This means combining weakly dependent detectors gives better coverage than any single detector alone.
#### UCBD Inference Algorithm
The full inference logic is defined in Algorithm 1, reproduced below with step-by-step explanation:
```
Algorithm 1 UCBD Cascade Inference
Require: Query q, boundaries {B1, ..., Bk} ordered by cost, thresholds {τi}
Ensure: Uncertainty flag u ∈ {0, 1}, confidence score s
1: s ← 0
2: for i = 1 to k do
3:   s_i ← B_i(q)                                    # Run boundary detector i on query q
4:   if s_i > τ_i then                               # If detector i is confidently uncertain
5:       return (u = 1, s = s_i)                      # Early exit, return uncertain flag
6:   end if
7:   s ← s + w_i · s_i                                # Accumulate weighted detector score
8: end for
9: return (u = 𝟙[s > τ_global], s)                   # Final check of accumulated score
```
Step-by-step explanation for beginners:
1.  Line 1: Initialize the accumulated confidence score $s$ to 0.
2.  Line 2: Iterate over the boundary detectors in order of increasing cost (cheapest first).
3.  Line 3: Compute the uncertainty score $s_i$ for query $q$ using the $i$-th detector.
4.  Lines 4-6: If the score $s_i$ exceeds the pre-defined threshold $\tau_i$ for detector $i$, the model is confidently uncertain, so we immediately return $u=1$ (uncertain) and exit early, avoiding the cost of running all subsequent more expensive detectors.
5.  Line 7: If detector $i$ does not flag uncertainty, add the weighted score $w_i \cdot s_i$ to the accumulated total score $s$ (weights $w_i$ are learned on a held-out validation set).
6.  Line 9: If none of the individual detectors flag uncertainty, check if the accumulated score $s$ exceeds a global threshold $\tau_{global}$. Return $u=1$ if it does, else $u=0$ (confident response).
#### Pointer Model (Router)
A lightweight logistic regression classifier that predicts whether the model's answer will be incorrect, using 20 free features available before running any expensive detectors: 7 entropy statistics + 13 text features (response length, question type indicators, presence of hedging phrases like "I think" or "maybe"). It is used to optimize routing of queries to the most appropriate detectors, achieving 0.585 AUC with only free features, and 0.707 AUC when adding query embeddings (shared with B2 to avoid additional cost).
---
The UCBD framework architecture is visualized in Figure 1 from the paper:

![Figure 1: UCBD framework: four-column architecture mapping brain mechanisms, boundary detectors, detection signals, and response strategies. Solid borders indicate experimentally validated components (B1B2, cascade B1 B2); dashed borders indicate theoretical components awaiting empirical validation (B3-B5). The Pointer Model (center, PFCanalogue) connects to allfive boundary detectors—solid arrows for validated bound aries (B1B2), dashed arrows for theoretical ones (B3B5)—dispatching queries into the cheapest-frst cascade. Cost increases top to bottom from free (token entropy) to expensive (external cross-validation).](images/1.jpg)
*Figure 1: UCBD framework: four-column architecture mapping brain mechanisms, boundary detectors, detection signals, and response strategies. Solid borders indicate experimentally validated components (B1-B2, cascade B1+B2); dashed borders indicate theoretical components awaiting empirical validation (B3-B5).*

# 5. Experimental Setup
## 5.1. Datasets
The paper uses 5 publicly available datasets spanning 3 task types, to validate the generalizability of findings:

| Dataset | Task Type | Size | Description & Example | Rationale for Selection |
|---------|-----------|------|-----------------------|--------------------------|
| TruthfulQA | Factual QA (misconception-focused) | 790 questions | Questions designed to test common LLM hallucinations and misconceptions. Example: *"What happens if you eat watermelon seeds?"* | Primary testbed for factual QA uncertainty, covers domains including health, law, finance, and common misconceptions. |
| GSM8K | Mathematical reasoning | 500 grade-school math problems | Multi-step arithmetic word problems with exact numerical answers. Example: *"If 3 apples cost \$6, how much do 5 apples cost?"* | Tests UQ performance on mathematical reasoning, where alignment is not expected to suppress diversity of reasoning steps. |
| FreshQA | Temporal factual QA | 1500 time-sensitive questions | Questions requiring up-to-date knowledge, with annotations for knowledge cutoff dates. Example: *"Who is the current CEO of Twitter?"* | Validates the B3 freshness boundary, tests UQ on temporal knowledge gaps. |
| HotpotQA | Multi-hop factual QA | 100 questions | Questions requiring combining information from multiple sources to answer. Example: *"Which author wrote the novel that the 1994 film *The Shawshank Redemption* is based on?"* | Tests UQ performance on complex multi-hop reasoning, validates the need for retrieval-based B5 boundary. |
| WebQuestions | General factual QA | 200 questions | Factual questions drawn from Google search queries, with Freebase answers. Example: *"What language does Cuba speak?"* | Cross-dataset validation, confirms alignment tax generalizes beyond TruthfulQA's misconception-focused prompts. |

All datasets are publicly available and widely used in LLM evaluation, ensuring results are comparable to prior work.
## 5.2. Evaluation Metrics
We provide full definitions, formulas, and symbol explanations for every evaluation metric used:
### 1. Single-Cluster Rate (SCR)
- **Conceptual Definition**: The percentage of queries where all N sampled responses fall into a single semantic cluster, measuring the severity of response homogenization. 0% = full diversity, 100% = complete homogenization.
- **Formula**:
  $$\mathrm{SCR} = \frac{\text{Number of queries with } |\mathcal{C}|=1}{\text{Total number of queries}} \times 100\%$$
- **Symbol Explanation**: $|\mathcal{C}|$ = number of distinct semantic clusters across N sampled responses for a query.
### 2. AUROC (Area Under the Receiver Operating Characteristic Curve)
- **Conceptual Definition**: Measures the ability of an uncertainty signal to distinguish between correct and incorrect responses. 0.5 = random guessing (no discriminative power), 1.0 = perfect classification.
- **Formula**:
  $$\mathrm{AUROC} = \int_{0}^{1} \mathrm{TPR}(\mathrm{FPR}) d\mathrm{FPR}$$
- **Symbol Explanation**:
  - $\mathrm{TPR}$ = True Positive Rate: fraction of incorrect responses correctly flagged as uncertain.
  - $\mathrm{FPR}$ = False Positive Rate: fraction of correct responses incorrectly flagged as uncertain.
### 3. Cohen's $d$
- **Conceptual Definition**: Measures the effect size of the difference between the uncertainty scores of correct vs incorrect responses. 0.2 = small effect, 0.5 = medium, 0.8 = large.
- **Formula**:
  $$d = \frac{\mu_{\text{incorrect}} - \mu_{\text{correct}}}{\sigma_{\text{pooled}}}$$
- **Symbol Explanation**:
  - $\mu_{\text{incorrect}}, \mu_{\text{correct}}$ = mean uncertainty score for incorrect and correct responses, respectively.
  - $\sigma_{\text{pooled}} = \sqrt{\frac{(n_{\text{incorrect}}-1)\sigma_{\text{incorrect}}^2 + (n_{\text{correct}}-1)\sigma_{\text{correct}}^2}{n_{\text{incorrect}} + n_{\text{correct}} - 2}}$, pooled standard deviation of the two groups.
  - $n_{\text{incorrect}}, n_{\text{correct}}$ = number of incorrect and correct responses, respectively.
### 4. Mutual Information (MI)
- **Conceptual Definition**: Measures the amount of information gained about one random variable by observing another, used to quantify dependence between uncertainty signals. 0 bits = completely independent, higher values = stronger dependence.
- **Formula**:
  $$I(X; Y) = \sum_{x \in X} \sum_{y \in Y} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right)$$
- **Symbol Explanation**: `p(x,y)` = joint probability distribution of variables $X$ and $Y$, `p(x), p(y)` = marginal probability distributions of $X$ and $Y$.
### 5. Expected Calibration Error (ECE)
- **Conceptual Definition**: Measures how well the model's predicted confidence matches actual accuracy. Lower values = better calibration.
- **Formula**:
  $$\mathrm{ECE} = \sum_{m=1}^M \frac{n_m}{N} |\mathrm{acc}_m - \mathrm{conf}_m|$$
- **Symbol Explanation**:
  - $M$ = number of confidence bins, $n_m$ = number of samples in bin $m$, $N$ = total samples.
  - $\mathrm{acc}_m$ = actual accuracy of samples in bin $m$, $\mathrm{conf}_m$ = average predicted confidence of samples in bin $m$.
## 5.3. Baselines
The paper compares the B1 token entropy signal against state-of-the-art sampling-based UQ baselines, all representative of standard practice in the field:
1.  **SE-Jaccard**: Semantic Entropy using bigram Jaccard similarity to cluster responses (surface-level lexical similarity).
2.  **SE-Embedding**: Semantic Entropy using embedding cosine similarity and agglomerative clustering to group semantically equivalent responses (replicates the SINdex method from prior work).
3.  **SE-NLI**: Canonical Semantic Entropy using bidirectional natural language inference (NLI) to check semantic equivalence between responses, tested at three DeBERTa-v3 scales (70M xsmall, 184M base, 435M large parameters).
4.  **SelfCheckGPT**: A widely used sampling-based UQ method that measures the average cosine similarity between the greedy response and 5 sampled responses.
5.  **P(True)**: Verbalized confidence, where the model is prompted to explicitly rate its confidence in its answer on a 0-1 scale.

# 6. Results & Analysis
## 6.1. Core Results Analysis
We break down the results into three categories: alignment tax diagnosis, causal ablation results, and cascade performance results.
### 6.1.1 Alignment Tax Diagnosis
The core finding is that response homogenization is widespread in aligned LLMs, and completely breaks sampling-based UQ on affected queries:
- SCR on TruthfulQA is 40.0% with Jaccard clustering, 79.0% with embedding clustering (meaning 79% of queries have no semantic diversity across 10 sampled responses, even if wording varies).
- On single-cluster queries, all sampling-based baselines have AUROC ~0.5 (random guessing), while B1 token entropy retains AUROC 0.603, as visualized in Figure 2:

  ![Figure 2: The alignment tax mechanism. On single-cluster questions $( 4 0 . 0 \\% )$ , SE drops to exact chance (0.500, dashed red) because all 10 samples produce the same answer. B1 retains discriminative power (0.603) because per-token entropy captures computational uncertainty independent of output diversity.](images/2.jpg)
  *Figure 2: On single-cluster questions (40.0%), sampling-based SE drops to exact chance (0.500) because all 10 samples produce the same answer. B1 retains discriminative power (0.603) because it measures per-token computational uncertainty.*

- B1 token entropy (free) matches or outperforms all sampling-based baselines, even the most expensive NLI-based SE variants: B1 AUROC = 0.599 vs SE-NLI AUROC = 0.501-0.512 (all near chance).
### 6.1.2 Causal Ablation Results
A series of ablations confirm alignment (specifically DPO) causes response homogenization:
1.  **Base vs Instruct Ablation**: Qwen3-14B base (unaligned) model has 1.0% SCR vs 28.5% for the aligned instruct variant (p < 1e-6, Wilcoxon signed-rank test), confirming alignment is the causal driver.
2.  **Training Stage Ablation**: On the Mistral/Zephyr pipeline: Base model (0.0% SCR) → SFT (1.5% SCR) → DPO (4.0% SCR). SFT preserves near-base diversity, while DPO drives the majority of homogenization. This is replicated on the Llama/Tulu-3 pipeline: Base (0.0% SCR) → SFT (0.0% SCR) → DPO+RLVR (0.5% SCR).
3.  **Cross-Family Replication**: SCR varies drastically across model families and alignment recipes: Qwen3-14B (28.5%), LLaMA-3.2-3B (5.5%), Mistral-7B (1.0%), Tulu-3 (0.5% SCR), spanning two orders of magnitude.
4.  **Robustness Checks**:
    - Decoding strategy ablation: SCR remains 28.5-33.5% across nucleus sampling and low temperature (T=0.7), so homogenization is a property of the model's learned distribution, not the sampling procedure.
    - Generation length ablation: SCR is 32% at 40 tokens, 8% at 200 tokens, but still significantly higher than the base model's 0% SCR.
    - Cross-embedder validation: An independent embedding model (Nomic-embed-text) detects even higher SCR (92% vs 78% for Qwen embedding at τ=0.85), ruling out coupling bias from same-family embedders.
    - Cross-dataset validation: WebQuestions has 58.0% SCR at τ=0.85, confirming the alignment tax generalizes beyond TruthfulQA.
### 6.1.3 Cascade Performance Results
The UCBD cascade delivers strong performance at low cost:
- The cascade achieves AUROC = 0.538, statistically equivalent to running all detectors in parallel (AUROC = 0.532, p=0.498, TOST equivalence test), while using only 71.6% of the parallel cost, saving 57% of B2 calls (50% of queries are resolved at the free B1 stage).
- Selective prediction on GSM8K: Abstaining on the 50% most uncertain queries raises accuracy from 84.4% to 93.2% at 50% coverage (p < 1e-4, McNemar's test).
- Cross-task AUROC performance is visualized in Figure 3:

  ![Figure 3: AUROC for error detection across tasks (dashed red `=` chance). The alignment tax is visible on TruthfulQA: B1 (free, 0.599) matches SelfCheckGPT ( $6 \\times$ , 0.588, p=0.65) and significantly outperforms Jaccardapproximated SE (11 $\\times$ , 0.548, $p _ { \\mathrm { a d j } }$ =0.04). On GSM8K, where alignment does not suppress entropy, B1 reaches 0.724 $d$ =0.81).](images/3.jpg)
  *Figure 3: AUROC across tasks. On TruthfulQA, B1 (free, 0.599) matches SelfCheckGPT (6x cost, 0.588) and outperforms Jaccard SE (11x cost, 0.548). On GSM8K, B1 reaches 0.724 (large effect size, d=0.81).*

## 6.2. Data Presentation (Tables)
We reproduce all key tables from the paper exactly as written:
### Table 1: Positioning of UCBD relative to existing approaches

| Method | Boundary | Cost | Cascade Role |
|--------|----------|------|--------------|
| Token entropy (ours) | B1 Fluency | Free | Stage 1 (always-on) |
| LogTokU / PRO [Ma et al., 2025b, Chen et al., 2025c] | B1 Fluency | Free | Stage 1 (drop-in) |
| Semantic Energy [Ma et al., 2025a] | B1 Fluency | N samples | Single-cluster remedy |
| SINdex [Abdaljalil et al., 2025] | B1 Fluency | N samples | Stage 1 (escalation) |
| Semantic Entropy [Kuhn et al., 2023] | B1 Fluency | 5-10× | Stage 1 (escalation) |
| CoCoA [Huang et al., 2026] | B1 Fluency | 1-5× | Stage 1 (escalation) |
| SelfCheckGPT [Manakul et al., 2023] | B1 Fluency | 5 calls | Stage 1 (escalation) |
| Embedding density [Vazhentsev et al., 2025] | B2 Density | 1 embed | Stage 2 |
| KG completion [Trouillon et al., 2016] | B4 Rupture | KG query | Stage 4 |
| NLI verification (ours) | B5 Grounding | NLI call | Stage 5 |
| ReMA [Wan et al., 2025] | Pointer Model | RL | Dispatcher |

---
### Table 6: B1 entropy vs sampling-based baselines on TruthfulQA

| Method | AUROC | 95% CI | Cost |
|--------|-------|--------|------|
| B1 mean entropy | 0.599 | [0.559, 0.637] | Free |
| SelfCheck-Emb (k=5) | 0.588 | [0.547, 0.626] | 6× |
| SE-Jaccard (N=10) | 0.548 | [0.510, 0.589] | 11× |
| SE-Embedding (N=10) | 0.542 | [0.513, 0.572] | 11× |
| SE-NLI† (DeBERTa-large) | 0.511 | [0.419, 0.594] | 11×+NLI |
| SE-NLI† (DeBerta-base) | 0.512 | [0.421, 0.593] | 11×+NLI |
| SE-NLI† (DeBERTa-xsmall) | 0.501 | [0.404, 0.595] | 11×+NLI |

*†200-question subset; DeBERTa-large recomputed with LLM-judge labels for fair comparison.*
---
### Table 7: Base-vs-instruct response diversity on TruthfulQA (n=200)

| Metric | Base | Instruct | Difference |
|--------|------|----------|------------|
| Single-cluster rate | 1.0% | 28.5% | +27.5pp |
| Mean clusters | 9.26 | 3.58 | -5.68 |
| Mean SE | 2.158 | 0.832 | -1.326 |

*Wilcoxon signed-rank (base > instruct): W=18,331, p < 1e-6*
---
### Table 10: Training stage ablation (Mistral/Zephyr pipeline)

| Stage | Model | SCR | Mean NC | Mean SE |
|-------|-------|-----|---------|---------|
| Base | Mistral-7B-v0.1 | 0.0% | 9.28 | 2.170 |
| SFT | mistral-7b-sft-beta | 1.5% | 8.63 | 2.024 |
| SFT+DPO | zephyr-7b-beta | 4.0% | 8.01 | 1.897 |

*Wilcoxon: Base→SFT p=0.002, SFT→DPO p=0.0001, Base→DPO p <1e-6*
---
### Table 5: GSM8K error detection (n=500)

| Feature | AUROC | 95% CI | Cost |
|---------|-------|--------|------|
| B1 mean entropy | 0.706 | [.635,.772] | Free |
| B1 std entropy | 0.715 | [.643,.782] | Free |
| B1 max entropy | 0.724 | [.650,.793] | Free |
| Combined entropy (4 feat, CV) | 0.724 ± .033 | - | Free |
| P(True) | 0.608 | [.52,.70] | 1 call |
| Response length (tokens)† | 0.849 | [.791,.903] | Free |
| Combined with length (5 feat, CV)† | 0.844 ± .041 | - | Free |

*†Length is a difficulty proxy (longer = more failed steps), not a true uncertainty signal.*
## 6.3. Ablation Studies / Parameter Analysis
The authors conduct extensive ablation studies to validate the robustness of their findings:
### 1. Temperature Ablation
SCR decreases monotonically with temperature, but remains substantial even at very high temperature: T=0.3 (62% SCR), T=0.7 (46% SCR), T=1.0 (42% SCR), T=1.5 (38% SCR). Even aggressive sampling at T=1.5 fails to eliminate homogenization on 38% of queries.
### 2. Sample Size Ablation
SCR is largely independent of sample size: N=3 (46.3% SCR), N=5 (41.9% SCR), N=7/10 (40.0% SCR). Collapse is a property of the model's output distribution, not insufficient sampling.
### 3. Quantization Ablation
4-bit and 8-bit quantization produce identical SCR (6.7%) at semantic clustering thresholds, ruling out quantization artifacts as a confound. Mean pairwise similarity differs by only 3.2pp between 4-bit and 8-bit models.
### 4. Clustering Threshold Ablation
SCR remains substantial across all reasonable clustering thresholds: embedding cosine threshold τ=0.80 (60% SCR), τ=0.85 (79% SCR), τ=0.90 (92% SCR). The alignment tax is not sensitive to arbitrary clustering threshold choices.
### 5. Model Scale Ablation
B1 token entropy effectiveness decreases with model size: LLaMA-3.2-3B (overall AUC=0.622), Qwen3-4B (0.537), Qwen3-14B (0.490). Larger aligned models produce more uniformly fluent outputs, compressing token entropy into a narrower range and reducing its discriminative power.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
The paper makes three well-validated core contributions:
1.  **Alignment Tax Diagnosis**: Alignment (specifically DPO training) causes response homogenization on 40-79% of factual QA queries, rendering all sampling-based UQ methods completely uninformative on these queries (AUROC=0.500). This effect is robust across clustering methods, sampling parameters, model families, and datasets, with severity varying by up to 50x across alignment recipes (0.5% to 28.5% SCR).
2.  **Task-Dependent Uncertainty**: Uncertainty signal performance is highly task-dependent: token entropy has near-zero effect size on factual QA (Cohen's d=0.07) but a large effect size on mathematical reasoning (d=0.81), because alignment suppresses inter-response diversity but not per-step reasoning uncertainty for math problems. No single UQ signal works across all task types.
3.  **UCBD Cascade Solution**: A cheapest-first cascade over weakly dependent orthogonal uncertainty signals matches the performance of running all detectors in parallel while reducing average cost by 57%. It enables selective prediction that raises GSM8K accuracy from 84.4% to 93.2% at 50% coverage, with 50% of queries resolved using only free logprob outputs.
    The core practical takeaway for practitioners is that sampling-based UQ methods are unreliable for aligned LLMs on factual QA tasks, and should be replaced or augmented with single-pass token-level signals and multi-signal cascades.
## 7.2. Limitations & Future Work
### Author-Identified Limitations
1.  **Alignment Attribution**: The highest-SCR model family (Qwen3-14B, 28.5% SCR) does not have a publicly available SFT-only checkpoint, so the authors cannot definitively confirm DPO is the driver of homogenization for this specific family, though the consistent pattern across two other pipelines provides strong evidence.
2.  **Baseline Implementations**: The NLI-based SE baseline omits the contradiction-aware clustering variant from the original SE paper, though this is structurally irrelevant for single-cluster queries (which make up 40-79% of all queries). Head-to-head comparison with official implementations of other single-pass UQ methods (LogTokU, Semantic Energy) on multi-cluster queries remains future work.
3.  **Label Validity**: LLM-judge labels have moderate agreement with TruthfulQA gold answer templates (Cohen's κ=0.487), though the core SCR finding is label-free and unaffected by labeling methodology.
4.  **Scope**: Experiments are limited to 3B-14B parameter models with 4-bit quantization; generalization to closed-source GPT-4 class models and other domains (code, dialogue) is unconfirmed.
5.  **B5 Boundary**: The B5 grounding boundary uses gold reference answers unavailable at inference time; a production implementation would require retrieval + NLI against external documents, which may yield lower performance.
6.  **Pointer Model**: The routing model is only evaluated on TruthfulQA; cross-domain evaluation on other datasets is needed.
7.  **GSM8K Length Confound**: Response length (a difficulty proxy) outperforms token entropy on GSM8K selective prediction, though length is not a generalizable signal across tasks (it performs near-chance on factual QA).
### Author-Suggested Future Work
1.  Test diversity-preserving alignment methods (H-DPO, SPL, DivPO) to see if they reduce the alignment tax while maintaining alignment quality.
2.  Run experiments at FP16 precision for additional quantization control.
3.  Conduct head-to-head comparison with other state-of-the-art single-pass UQ methods (LogTokU, PRO, Semantic Energy, hidden-state probes).
4.  Test the alignment tax and UCBD cascade on large closed-source models (GPT-4, Claude) and additional domains (code, dialogue).
5.  Explore better calibration methods (isotonic regression, temperature scaling) for the cascade's uncertainty scores.
6.  Implement a production-grade B5 boundary using retrieval + NLI against external sources.
7.  Integrate the UCBD cascade into end-to-end agent systems with selective prediction and retrieval fallback.
8.  Conduct human evaluation to validate LLM-judge labels and the homogenization-correctness relationship.
## 7.3. Personal Insights & Critique
### Key Inspirations
This paper addresses an extremely practical, under-studied problem that every team deploying aligned LLMs faces: sampling-based UQ methods that work well on base models often fail completely on production aligned models. The SCR diagnostic is a simple, label-free tool that any practitioner can implement in minutes to check if sampling-based UQ is reliable for their specific model and task, which alone makes this paper highly valuable.
The cascade design is also very practical for real-world deployments, where inference cost is a major constraint, and the weak dependence between signals ensures the approach is robust to different task types.
### Potential Improvements
1.  The UCBD cascade currently only fully validates B1 and B2 boundaries; B3-B5 are theoretical and require more empirical validation on real-world retrieval systems.
2.  The paper does not explore how the cascade performs on other task types like code generation or open-ended dialogue, where response diversity is often desirable and homogenization may be less severe.
3.  The routing Pointer Model is a simple logistic regression; more powerful routing models (e.g., small fine-tuned LLMs) could potentially improve routing accuracy and further reduce cost.
### Transferability
The core findings are highly transferable to other domains:
- The SCR diagnostic can be used for any LLM deployment to evaluate UQ reliability, regardless of task.
- The cheapest-first cascade approach can be adapted to other use cases like content moderation, retrieval-augmented generation (RAG) routing, and tool use for agents, where different signals have different costs and capture different failure modes.
### Critical Assessment
One potential unstated limitation is that the alignment tax is most severe for short factual answers, which are common in customer support and information retrieval use cases, but less severe for long-form generation (e.g., essay writing, creative writing) where the paper shows SCR drops to ~8% at 200 tokens. Practitioners working on long-form generation may not face as severe a UQ failure from sampling-based methods, though the cascade approach is still useful for cost reduction.
Overall, this is a landmark paper for LLM UQ research, as it provides a clear, empirically validated explanation for a widely observed but poorly understood performance gap, and a practical, deployable solution that works with existing tools and models.