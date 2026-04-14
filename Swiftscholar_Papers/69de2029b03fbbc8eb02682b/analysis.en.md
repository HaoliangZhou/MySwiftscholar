# 1. Bibliographic Information
## 1.1. Title
The paper is titled *OSCAR: Optimization-Steered Agentic Planning for Composed Image Retrieval*. Its central topic is the design of a new framework for composed image retrieval (CIR) that replaces heuristic agent planning with mathematically optimized trajectories to address limitations of existing CIR methods.
## 1.2. Authors
The work is a joint industry-academia collaboration between OPPO and Shanghai Jiao Tong University (SJTU), with authors:
- Teng Wang (OPPO, The University of Hong Kong)
- Rong Shan (SJTU)
- Jianghao Lin (SJTU)
- Junjie Wu (OPPO)
- Tianyi Xu (SJTU)
- Jianping Zhang (SJTU)
- Wenteng Chen (SJTU)
- Changwang Zhang (OPPO)
- Zhaoxiang Wang (OPPO)
- Weinan Zhang (SJTU, leading academic researcher in reinforcement learning and operations research)
- Jun Wang (OPPO, leading industry researcher in multimedia systems)
## 1.3. Journal/Conference
As of the provided date, the paper is published as a preprint on arXiv, and has not yet been officially accepted to a peer-reviewed conference or journal. arXiv is the primary preprint repository for computer science research, and this work is in the fields of computer vision, information retrieval, and agentic AI.
## 1.4. Publication Year
2026 (submitted to arXiv on 2026-02-09)
## 1.5. Abstract
The paper addresses two key limitations of existing composed image retrieval (CIR) systems: (1) unified embedding methods suffer from single-model myopia (one model cannot handle all diverse query intents and domains), and (2) heuristic agentic CIR methods rely on trial-and-error iterative orchestration with no global optimality guarantees. The core proposed solution is OSCAR, the first framework to reframe agentic CIR as a principled trajectory optimization problem using a novel offline-online paradigm:
- Offline phase: Model CIR as a two-stage mixed-integer programming (MIP) problem to derive optimal retrieval trajectories (tool calls and set operations) for training samples using boolean set operations, maximizing ground-truth coverage.
- Online phase: Store optimized trajectories in a golden library as in-context demonstrations to steer a vision-language model (VLM) planner for inference, avoiding expensive iterative search.
  Extensive experiments show OSCAR consistently outperforms state-of-the-art (SOTA) baselines on 3 public CIR benchmarks and 1 private industrial dataset, and achieves superior performance using only 10% of training data, demonstrating strong generalization of planning logic rather than dataset-specific memorization.
## 1.6. Original Source Link
- Official preprint: https://arxiv.org/abs/2602.08603
- PDF link: https://arxiv.org/pdf/2602.08603v1
- Publication status: Public preprint on arXiv, not yet peer-reviewed.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Composed Image Retrieval (CIR) is the task of retrieving images from a gallery that match a composed query: a reference image plus a natural language modification instruction (e.g., reference = red dress, modification = "make it blue, longer length, add lace details"). CIR is critical for real-world retrieval systems, as user queries are increasingly multimodal, compositional, and require fine-grained reasoning over visual and textual constraints.
### Gaps in Prior Research
Existing CIR approaches fall into two flawed paradigms:
1. **Unified Embedding Retrieval**: Fuse the reference image and modification text into a single vector, then perform nearest-neighbor search. These methods suffer from *single-model myopia*: no single embedding space can handle the full diversity of query intents (high-level style changes vs. fine-grained attribute constraints), domains (fashion vs. natural scenes), and granularities. A model optimized for fashion retrieval will fail at natural scene queries, and vice versa.
2. **Heuristic Agentic Retrieval**: Use large language models (LLMs) or VLMs to orchestrate external tools (captioners, retrievers, rewriters) via iterative ReAct-style loops. These methods suffer from *suboptimal orchestration*: greedy, local decision-making leads to redundant tool calls, poor logical ordering, unreliable handling of inclusion/exclusion constraints, and no global optimality guarantees, with high computational overhead from repeated model interactions.
### Innovative Entry Point
The paper's core insight is that agentic CIR can be reframed from a heuristic search problem to a principled trajectory optimization problem. For training samples with known ground truth, optimal retrieval tool selection and composition can be mathematically derived via MIP, and these optimized trajectories can be transferred to unseen test queries via in-context learning, eliminating the need for iterative trial-and-error at inference.
## 2.2. Main Contributions / Findings
The paper makes four primary contributions:
1. **Optimization Perspective**: First work to formulate agentic CIR as a MIP problem. Global optimization is used to derive optimal planning trajectories for training samples that maximize ground-truth coverage while minimizing redundancy, providing high-quality demonstration signals with no human annotation.
2. **Set-Theoretic Composition Logic**: Introduce rigorous boolean set operations (union, intersection, difference) for CIR result composition, enabling explicit inclusion and conservative exclusion reasoning that is mathematically intractable for single-embedding models and heuristic agent methods.
3. **OSCAR Framework**: Novel offline-online paradigm bridging MIP optimization and agentic planning. A golden library of MIP-derived optimal trajectories steers VLMs to perform complex compositional CIR planning in a single inference pass, no iterative search required.
4. **Empirical Superiority**: OSCAR consistently outperforms all SOTA baselines (unified embedding, CIR-dedicated, heuristic agent) on 3 public benchmarks (CIRCO, CIRR, FashionIQ) and 1 private industrial user photo gallery dataset. It achieves these gains with only 10% of training data for golden library construction, demonstrating strong generalization of abstract planning logic rather than dataset memorization.
   Key findings include:
- Optimized planning over existing retrieval tools delivers larger performance gains than designing new specialized single CIR models.
- Set-theoretic operations are critical for filtering irrelevant results and handling exclusion constraints.
- Optimization-derived trajectories transfer general planning logic, not just query-specific answers, enabling strong generalization with limited training data.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
To understand this paper, the following core concepts are defined for beginners:
1. **Composed Image Retrieval (CIR)**: A multimodal retrieval task where the input is a tuple of a reference image $q_{img}$ and a modification text $q_{txt}$, and the output is all images in a gallery that match the reference image with the specified modification applied. It requires fine-grained reasoning over both visual and textual constraints.
2. **Mixed-Integer Programming (MIP)**: A class of mathematical optimization problems where some decision variables are constrained to be integers (most often binary 0/1 variables here) and others are continuous. The goal is to maximize or minimize a linear objective function subject to a set of linear constraints. It is used here to select optimal retrieval tools and composition rules.
3. **In-Context Learning (ICL)**: The ability of large pre-trained LLMs/VLMs to perform tasks without fine-tuning, by being provided with task demonstrations (examples) directly in the input prompt. The OSCAR framework uses optimized trajectories as in-context demonstrations for the VLM planner.
4. **Boolean Set Operations for Retrieval**:
   - **Union ($A \cup B$)**: Returns all items present in either set A or set B, used to improve recall by aggregating results from multiple retrieval tools.
   - **Intersection ($A \cap B$)**: Returns only items present in both set A and set B, used to improve precision by retaining only results that satisfy multiple constraints.
   - **Difference ($A \setminus B$)**: Returns items present in set A but not in set B, used to filter out unwanted results that match excluded attributes.
5. **Vision-Language Model (VLM)**: A type of pre-trained large model that can process and understand both image and text inputs, enabling cross-modal reasoning tasks like CIR, image captioning, and visual question answering.
## 3.2. Previous Works
CIR research is split into two main lines of prior work, as outlined in the paper:
### Unified Embedding Retrieval for CIR
This paradigm fuses the reference image and modification text into a single unified embedding vector, then performs nearest-neighbor search over precomputed gallery embeddings. Representative works include:
- **CLIP4CIR (2023)**: Fine-tunes CLIP vision-language encoders and learns a dedicated fusion layer to combine reference image and modification text features.
- **Pic2Word (2023)**: Maps image features to pseudo-word tokens in the textual embedding space, enabling image-text composition via pre-trained text encoders for zero-shot CIR.
- **FiRE (2025)**: Fine-tunes VLMs to enhance fine-grained context understanding for CIR, improving retrieval of subtle modifications.
  All unified embedding methods share the core limitation of single-model myopia: no single embedding space can accommodate the full diversity of query intents, domains, and constraint granularities.
### Heuristic Agentic Retrieval for CIR
This paradigm uses LLM/VLM agents to decompose CIR into multi-step workflows, invoking external tools iteratively to handle complex queries. Representative works include:
- **AutoCIR (2025)**: A multi-agent pipeline that iteratively revises queries and refines retrieval results via heuristic feedback loops.
- **MRA-CIR (2025)**: A multimodal reasoning agent that decouples intent reasoning from retrieval execution, using LLM logic to coordinate retrieval tools.
- **$X^R$ (2026)**: A cross-modal agent framework that coordinates specialized visual and textual tools via synergistic planning.
  All heuristic agent methods share the core limitation of suboptimal orchestration: greedy, iterative decision-making lacks global optimality guarantees, leads to redundant tool calls, and fails reliably at set-based inclusion/exclusion constraints, with high computational cost.
### Background on Set Cover Problem
The MIP formulation used in OSCAR is a variation of the classic *set cover problem*, a well-known NP-hard problem in operations research:
Given a universe of elements $U$ and a collection of sets $S$ whose union equals $U$, the set cover problem seeks the smallest number of sets in $S$ whose union covers all elements in $U$. For OSCAR, ground-truth images are the universe elements, and atomic retrieval result sets are the collection $S$, with additional constraints for precision and tool diversity. MIP solvers (like COPT used in this work) can efficiently solve moderate-sized set cover problems to global optimality.
## 3.3. Technological Evolution
The evolution of CIR methods follows this timeline:
1. **2019-2021: Early Unified Embedding**: Focus on designing custom fusion operators for CNN-based and early multimodal model features, e.g., TIRG, DC-Net.
2. **2022-2024: CLIP-Based Zero-Shot CIR**: Leverage large pre-trained CLIP-style VLMs to enable zero-shot CIR without task-specific fine-tuning, e.g., Pic2Word, SEARLE, LinCIR.
3. **2024-2025: Heuristic Agentic CIR**: Use LLM/VLM agents to orchestrate multiple tools, overcoming single-model limitations via multi-step reasoning, e.g., AutoCIR, MRA-CIR.
4. **2026 onwards: Optimization-Steered Agentic CIR**: This paper's work, replacing heuristic agent planning with mathematically optimized trajectories to deliver global optimality guarantees and reduce inference cost.
## 3.4. Differentiation Analysis
OSCAR's core innovations differentiate it from all prior CIR approaches:
1. **vs. Unified Embedding Methods**: OSCAR does not rely on a single monolithic embedding space, instead composing outputs from multiple complementary retrieval tools via set operations to avoid single-model myopia. It uses dynamic planning to select the optimal tools for each individual query, rather than using a fixed model for all queries.
2. **vs. Heuristic Agentic CIR Methods**: OSCAR eliminates iterative trial-and-error ReAct loops. Instead, it uses offline MIP to generate globally optimal retrieval trajectories, which are used as in-context demonstrations to steer the VLM planner to produce near-optimal plans in a single inference pass. It explicitly models set-theoretic operations to handle inclusion/exclusion constraints reliably, with guarantees of optimality for offline trajectories.

# 4. Methodology
## 4.1. Principles
The core principle of OSCAR is to separate planning into offline optimization and online inference phases to combine the optimality guarantees of mathematical programming with the generalization capability of VLMs:
- **Offline Phase**: For training samples with known ground truth, solve two sequential MIP problems to find the optimal sequence of atomic retrieval tool calls and set operations that maximize retrieval performance (recall first, then precision). These optimal trajectories are stored in a searchable golden library.
- **Online Phase**: For unseen test queries, retrieve semantically similar optimal trajectories from the golden library as in-context demonstrations for a VLM planner. The VLM generates a near-optimal retrieval plan in a single pass, with no iterative search required.
  The theoretical basis for this approach is that optimal retrieval planning logic for similar query types is transferable. In-context learning with MIP-derived optimal trajectories enables the VLM to replicate this optimal reasoning for unseen queries without fine-tuning.
## 4.2. Core Methodology In-depth (Layer by Layer)
### 4.2.1. Problem Formulation
Let the composed query be defined as $q = \{ q_{img}, q_{txt} \}$, where $q_{img}$ is the reference image and $q_{txt}$ is the natural language modification instruction. The goal of CIR is to retrieve the set of ground-truth target images $\mathcal{I}^+ \subset \mathcal{I}$ from a full image gallery $\mathcal{I}$.
We define a set of available atomic retrieval tools $\mathcal{T} = \{f_1, f_2, ..., f_m\}$, where each tool $f$ accepts a query input and returns a set of candidate images. The OSCAR framework treats CIR as a compositional planning problem over this tool space, rather than a single embedding inference step.
### 4.2.2. Atomic Retrieval Construction
First, we define the fundamental planning unit: the *atomic retrieval* $r$, formalized as a 4-tuple:
$r = ( f, \hat{q}, p, k )$
Where:
- $f \in \mathcal{T}$: The selected retrieval tool (e.g., embedding-based searcher, caption-based text searcher).
- $\hat{q} = \{ q_{img}, \hat{q}_{txt} \}$: The rewritten query, where $\hat{q}_{txt}$ is generated by a VLM that decomposes the original modification text into explicit, discrete visual attributes and semantic constraints.
- $p \in \{+, -\}$: Polarity of the retrieval: `+` = positive (the attribute must be present in target results), `-` = negative (the attribute must be explicitly excluded from target results).
- $k$: Top-$k$ truncation threshold, the number of top candidate images returned by the retrieval tool.
  To avoid redundant inference, we discretize the truncation parameter $k$ into a finite set of levels $\mathcal{K} = \{k_1, k_2, ..., k_{max}\}$ (in this work, values from 5 to 50 with step size 5). For a fixed tool $f$ and query $\hat{q}$, retrieval results are monotonic with respect to $k$: $S_{k_1} \subset S_{k_2}$ if $k_1 < k_2$, where $S_k$ is the set of top-$k$ results. We only run retrieval once with the maximum $k_{max}$, and generate smaller top-$k$ result sets via slicing, eliminating redundant model inferences.
The full set of atomic retrievals $\mathcal{R}$ is the Cartesian product of available tools $f$, rewritten queries $\hat{q}$, polarities $p$, and truncation levels $k$, resulting in 1,182 atomic retrievals per sample in this work. This set forms the decision space for the offline MIP optimization stages.
### 4.2.3. Offline Phase: Two-Stage MIP Optimization
The offline phase generates optimal retrieval trajectories for each training sample with known ground truth $\mathcal{I}^+$. A two-stage MIP design is used to avoid the combinatorial explosion of a single monolithic MIP formulation, following a prune-and-refine strategy.
#### Stage 1: Recall-Oriented Selection MIP
The goal of Stage 1 is to select a compact subset of positive atomic retrievals $\mathcal{R}^+$ that maximizes coverage of ground-truth images $\mathcal{I}^+$ while minimizing inclusion of non-ground-truth images $\mathcal{I}^- = \mathcal{I} \setminus \mathcal{I}^+$. This prunes the large atomic retrieval space to a high-recall candidate universe $\mathcal{U}$.
**Variables for Stage 1 MIP**:
- $x_r \in \{0,1\}$: Binary decision variable, 1 if positive atomic retrieval $r \in \mathcal{R}^+$ is selected, 0 otherwise.
- $c_i \in \{0,1\}$: Auxiliary variable, 1 if image $i \in \mathcal{I}$ is covered by at least one selected positive atomic retrieval, 0 otherwise.
- $t_f \in \{0,1\}$: Auxiliary variable, 1 if tool $f$ is used by at least one selected atomic retrieval, 0 otherwise.
- Families $\mathcal{F}$: Groups of atomic retrievals that share the same tool $f$, query $\hat{q}$, and polarity $p$, differing only in truncation threshold $k$.
  **Stage 1 MIP Formulation**:
$$
\begin{array}{rl}
\underset{\{x_r\}_{r \in \mathcal{R}^+}}{\operatorname{max}} & \frac{w_R}{|\mathcal{I}^+|} \sum_{i \in \mathcal{I}^+} c_i - \frac{w_P}{|\mathcal{I}^-|} \sum_{i \in \mathcal{I}^-} c_i + \lambda_{\mathrm{div}} \sum_{f \in \mathcal{T}} t_f \\
\mathrm{s.t.} & \sum_{r \in F} x_r \leq 1, \quad \forall F \in \mathcal{F}, \\
& x_r, c_i, t_f \in \{0, 1\}.
\end{array}
$$
**Explanation of Formulation**:
- The first term in the objective maximizes recall (coverage of ground-truth images $\mathcal{I}^+$), weighted by $w_R$ (set to a high value to prioritize recall for this stage).
- The second term penalizes inclusion of non-ground-truth images $\mathcal{I}^-$, weighted by $w_P$, to avoid expanding the candidate space with excessive noise.
- The third term is a diversity regularizer $\lambda_{div}$ that encourages use of multiple distinct tools, preventing over-reliance on a single model and mitigating single-model myopia.
- The constraint enforces that at most one atomic retrieval is selected per family, avoiding redundant inclusion of nested result sets (since larger $k$ values strictly subsume smaller ones in the same family).
  **Output of Stage 1**: The optimal set of selected positive atomic retrievals $\mathcal{R}_*^+$, which produces the high-recall candidate universe $\mathcal{U} = \bigcup_{r \in \mathcal{R}_*^+} S_r$, where $S_r$ is the result set of atomic retrieval $r$.
#### Stage 2: Precision-Oriented Composition MIP
The goal of Stage 2 is to refine the candidate universe $\mathcal{U}$ by incorporating negative atomic retrievals and set-theoretic operations to filter out irrelevant noise, maximizing precision while preserving recall. A fixed two-clause composition structure is used for stability, interpretability, and computational tractability:
$$
S_{final} = \left( \bigcup_{r \in \mathcal{R}_{**}^+} S_r \right) \setminus \left( \bigcap_{r \in \mathcal{R}_*^-} S_r \right)
$$
**Explanation of Composition Structure**:
- **Positive Union**: All images retrieved by any of the selected positive atomic retrievals $\mathcal{R}_{**}^+ \subseteq \mathcal{R}_*^+$ (selected from the Stage 1 output).
- **Negative Intersection**: Only images retrieved by *all* selected negative atomic retrievals $\mathcal{R}_*^- \subseteq \mathcal{R}^-$. Intersection is used for the negative branch to enforce conservative exclusion: an image is only removed if all negative tools agree it is irrelevant, preventing accidental deletion of ground-truth images due to single-tool hallucination.
- **Set Difference**: The final result set contains images that are in the positive union *and not* in the negative intersection.
  **Variables for Stage 2 MIP**:
- $x_r \in \{0,1\}$: Binary decision variable, 1 if atomic retrieval $r \in \mathcal{R}_*^+ \cup \mathcal{R}^-$ is selected, 0 otherwise.
- $u_i \in \{0,1\}$: Auxiliary variable, 1 if image $i \in \mathcal{U}$ is covered by the positive union, 0 otherwise.
- $v_i \in \{0,1\}$: Auxiliary variable, 1 if image $i \in \mathcal{U}$ is in the negative intersection, 0 otherwise.
- $z_i \in \{0,1\}$: Auxiliary variable, 1 if image $i$ is included in the final result set $S_{final}$. Logically, $z_i = 1$ if and only if $u_i = 1$ AND $v_i = 0$.
  **Stage 2 MIP Formulation**:
$$
\begin{array}{rl}
\underset{\{x_r\}_{r \in \mathcal{R}_*^+ \cup \mathcal{R}^-}}{\operatorname{max}} & \sum_{i \in \mathcal{U} \cap \mathcal{I}^+} z_i - \lambda_{reg} \sum_{i \in \mathcal{I}^-} z_i \\
\mathrm{s.t.} & \sum_{r \in \mathcal{R}_*^+} x_r \geq 1, \\
& x_r, z_i \in \{0, 1\}.
\end{array}
$$
**Explanation of Formulation**:
- The first term in the objective maximizes retention of ground-truth images in the final result set.
- The second term is a regularizer $\lambda_{reg}$ that penalizes inclusion of non-ground-truth images, preventing trivial degenerate solutions (e.g., selecting no negative tools to maximize recall at the cost of extreme precision loss).
- The constraint enforces that at least one positive atomic retrieval is selected, ensuring the positive union is non-empty.
  **Output of Stage 2**: The optimal trajectory for the training sample, defined by the selected positive retrievals $\mathcal{R}_{**}^+$ and negative retrievals $\mathcal{R}_*^-$, plus the fixed two-clause set composition. All optimal trajectories are stored in the**Golden Library**, indexed by an embedding of the query context (concatenation of the modification text and reference image caption).
### 4.2.4. Online Phase: Optimization-Steered Inference
The offline MIP pipeline requires ground truth and is computationally expensive, so it cannot be used at inference. The online phase transfers the optimized planning logic from the Golden Library to test queries via in-context learning, following these steps:
1. **Query Embedding**: For a test query, generate a caption for the reference image using Qwen3-VL-32B, then concatenate this caption with the modification text. Encode this combined context using Qwen3-Embedding-8B to get the query context embedding.
2. **Demo Retrieval**: Retrieve the top-$N$ most similar optimal trajectories from the Golden Library using cosine similarity between the test query embedding and the library entry embeddings.
3. **VLM Planning**: Insert the retrieved trajectories as in-context demonstrations in the VLM planner prompt. The VLM planner generates the full retrieval plan (sequence of atomic retrieval tool calls, $k$ values, and set operations) in a single inference pass, no iterative search required.
4. **Plan Execution**: Run the selected atomic retrievals, apply the specified set operations to produce the filtered candidate set.
5. **VLM Verification**: A VLM verifier reranks the candidate set using binary relevance scoring: for each candidate, compute the logit difference between the VLM generating "yes" and "no" in response to the question "Does this candidate image match the composed query?", then convert to a relevance score via the sigmoid function: `s_i = \sigma(z_{yes} - z_{no})`, where $z_{yes}$ and $z_{no}$ are the next-token logits for "yes" and "no".
6. **Result Return**: Return the reranked final results.

   The overall OSCAR framework is illustrated in Figure 2 from the original paper:

   ![Figure 2: The overall framework of our proposed OSCAR](images/2.jpg)
   *该图像是示意图，展示了我们提出的OSCAR框架的整体结构。框架包括原子检索构建、回忆导向的选择性混合整数规划以及精确导向的组合混合整数规划，旨在通过优化机制实现复合图像检索优化。*

# 5. Experimental Setup
## 5.1. Datasets
Experiments are conducted on four diverse datasets covering zero-shot, domain-specific, and real-world industrial scenarios:
1. **CIRCO**: A zero-shot composed image retrieval dataset with no training set, 220 validation queries, 800 test queries, and a gallery of 123,403 real-world images. Queries have multiple ground-truth targets, requiring fine-grained compositional reasoning.
2. **CIRR**: A real-life image CIR dataset with 28,225 training queries, 4,181 validation queries, 4,148 test queries, and a gallery of ~19,000 images from complex real-world scenes.
3. **FashionIQ**: A fashion-domain CIR dataset with three subsets (Dress, Shirt, Toptee), each with dedicated training and validation splits, and gallery sizes ranging from 11,452 to 19,036 images per subset. Evaluations follow standard practice using the validation set.
4. **Private Industrial Photo Galleries**: Three real-world user photo galleries with 1,069/1,466/1,047 personal images and 483/336/369 real-user text-to-image retrieval queries, simulating practical personal album search scenarios with highly diverse image distributions and complex user intents.
   The dataset statistics are provided in Table 1 from the original paper:

   <table>
   <thead>
   <tr>
   <th rowspan="2">Split</th>
   <th rowspan="2">Num</th>
   <th rowspan="2">CIRCO</th>
   <th rowspan="2">CIRR</th>
   <th colspan="3">FashionIQ</th>
   </tr>
   <tr>
   <th>Dress</th>
   <th>Shirt</th>
   <th>Toptee</th>
   </tr>
   </thead>
   <tbody>
   <tr>
   <td rowspan="2">Training</td>
   <td>#Query</td>
   <td>-</td>
   <td>28,225</td>
   <td>5,985</td>
   <td>5,988</td>
   <td>6,027</td>
   </tr>
   <tr>
   <td>#Image</td>
   <td>-</td>
   <td>16,939</td>
   <td>11,452</td>
   <td>19,036</td>
   <td>16,121</td>
   </tr>
   <tr>
   <td rowspan="2">Validation</td>
   <td>#Query</td>
   <td>220</td>
   <td>4,181</td>
   <td>2,017</td>
   <td>2,038</td>
   <td>1,961</td>
   </tr>
   <tr>
   <td>#Image</td>
   <td>123,403</td>
   <td>2,297</td>
   <td>3,817</td>
   <td>6,346</td>
   <td>5,373</td>
   </tr>
   <tr>
   <td rowspan="2">Testing</td>
   <td>#Query</td>
   <td>800</td>
   <td>4,148</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
   </tr>
   <tr>
   <td>#Image</td>
   <td>123,403</td>
   <td>2,315</td>
   <td>-</td>
   <td>-</td>
   <td>-</td>
   </tr>
   </tbody>
   </table>

The dataset selection is highly comprehensive, covering zero-shot, general-domain, domain-specific, and industrial use cases to fully validate OSCAR's generalization and real-world applicability.
## 5.2. Evaluation Metrics
Three standard retrieval metrics are used for evaluation:
### 1. Recall@K (R@K)
- **Conceptual Definition**: Measures the fraction of all ground-truth images for a query that appear in the top-K retrieved results. It quantifies how many relevant items are retrieved, with higher values indicating better recall performance.
- **Mathematical Formula**:
  $$
\text{Recall@K} = \frac{|\text{Ground truth images in top-K results}|}{|\text{Total ground truth images for the query}|}
$$
- **Symbol Explanation**: $K$ is the number of top-ranked retrieved results considered for evaluation.
### 2. mean Average Precision @ K (mAP@K)
- **Conceptual Definition**: Average Precision (AP) measures the precision of retrieved results at every position where a new ground-truth image is found, averaged over all ground truths for a query. mAP@K is the average of AP@K across all queries, quantifying both recall and ranking quality (higher values are better). It is used for the CIRCO dataset, which has multiple ground truths per query.
- **Mathematical Formula**:
  For a single query $q$, AP@K is defined as:
$$
\text{AP@K}(q) = \frac{\sum_{k=1}^K P(k) \cdot rel(k)}{|\text{Total ground truth images for } q|}
$$
where `P(k)` is the precision at position $k$ (fraction of retrieved results up to position $k$ that are ground truth), and $rel(k) = 1$ if the result at position $k$ is ground truth, 0 otherwise. mAP@K is the average over all queries:
$$
\text{mAP@K} = \frac{1}{Q} \sum_{q=1}^Q \text{AP@K}(q)
$$
- **Symbol Explanation**: $Q$ is the total number of evaluation queries.
### 3. NDCG@K (Normalized Discounted Cumulative Gain @ K)
- **Conceptual Definition**: Measures the ranking quality of retrieved results, assigning higher weight to relevant items that appear in higher positions. It is normalized by the maximum possible DCG (ideal DCG) for the query, ranging from 0 to 1, with higher values indicating better ranking performance. It is used for the industrial gallery experiments.
- **Mathematical Formula**:
  Discounted Cumulative Gain (DCG) @ K is:
$$
\text{DCG@K} = \sum_{k=1}^K \frac{rel(k)}{\log_2(k+1)}
$$
Normalized DCG @ K is:
$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$
- **Symbol Explanation**: `rel(k)` is the relevance score of the item at position $k$, and $\text{IDCG@K}$ is the ideal DCG@K (the maximum possible DCG value when all ground-truth items are ranked in the top positions in order of relevance).
## 5.3. Baselines
OSCAR is compared against four representative categories of existing CIR methods to ensure comprehensive benchmarking:
1. **Multimodal Embedding Models**: General-purpose pre-trained vision-language embedding models: Ops-MM-embedding-v1-7B, RzenEmbed-v2-7B, VLM2Vec, B3-Qwen2-7B, QQMM-embed-v2.
2. **Caption-based Text Embedding Models**: Convert images and queries to text captions, then use text embedding for retrieval: bge-m3, Qwen3-Embedding (0.6B/4B/8B), with Qwen3-VL-32B for image captioning.
3. **CIR-dedicated Methods**: Specialized models designed specifically for the CIR task: Pic2Word, SEARLE, SEARLE-XL-OTI, CIReVL, LinCIR, LDRE, FiRE.
4. **Heuristic Agentic CIR Methods**: Existing agent-based CIR approaches using iterative heuristic planning: MRA-CIR, AutoCIR, $X^R$.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### Public Benchmark Results
The performance of OSCAR and all baselines on the CIRCO, CIRR, and FashionIQ benchmarks are shown in Table 2 and Table 3 from the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Type</th>
<th rowspan="2">Method</th>
<th rowspan="2">Training Free</th>
<th colspan="4">CIRCO</th>
<th colspan="4">CIRR</th>
</tr>
<tr>
<th>m@5</th>
<th>m@10</th>
<th>m@25</th>
<th>m@50</th>
<th>R@1</th>
<th>R@5</th>
<th>R@10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="5">Multimodal Embedding</td>
<td>Ops-MM-v1-7B</td>
<td>✓</td>
<td>13.56</td>
<td>15.96</td>
<td>18.61</td>
<td>19.68</td>
<td>1.90</td>
<td>50.29</td>
<td>66.68</td>
<td>91.64</td>
</tr>
<tr>
<td>RzenEmbed-v2-7B</td>
<td>✓</td>
<td>32.20</td>
<td>34.19</td>
<td>37.41</td>
<td>38.61</td>
<td>19.30</td>
<td>70.36</td>
<td>83.08</td>
<td>96.77</td>
</tr>
<tr>
<td>VLM2Vec</td>
<td></td>
<td>3.40</td>
<td>4.07</td>
<td>5.08</td>
<td>5.74</td>
<td>0.10</td>
<td>26.48</td>
<td>41.35</td>
<td>71.74</td>
</tr>
<tr>
<td>B3_Qwen2_7B</td>
<td>✓</td>
<td>3.67</td>
<td>4.55</td>
<td>5.53</td>
<td>6.13</td>
<td>0.80</td>
<td>37.95</td>
<td>54.39</td>
<td>81.01</td>
</tr>
<tr>
<td>QQMM-embed-v2</td>
<td>✓</td>
<td>45.92</td>
<td>47.13</td>
<td>50.39</td>
<td>51.45</td>
<td>28.98</td>
<td>73.66</td>
<td>82.98</td>
<td>96.72</td>
</tr>
<tr>
<td rowspan="4">Text Embedding</td>
<td>bge-m3</td>
<td>✓</td>
<td>8.15</td>
<td>8.61</td>
<td>9.62</td>
<td>10.21</td>
<td>12.53</td>
<td>34.00</td>
<td>48.63</td>
<td>75.06</td>
</tr>
<tr>
<td>Qwen3-Embed-0.6B</td>
<td>✓</td>
<td>9.55</td>
<td>10.35</td>
<td>11.61</td>
<td>12.33</td>
<td>14.87</td>
<td>39.11</td>
<td>54.53</td>
<td>82.51</td>
</tr>
<tr>
<td>Qwen3-Embed-4B</td>
<td>✓</td>
<td>13.55</td>
<td>14.60</td>
<td>16.33</td>
<td>17.20</td>
<td>22.17</td>
<td>51.57</td>
<td>64.84</td>
<td>88.19</td>
</tr>
<tr>
<td>Qwen3-Embed-8B</td>
<td>✓</td>
<td>16.45</td>
<td>17.23</td>
<td>18.98</td>
<td>19.88</td>
<td>23.21</td>
<td>52.48</td>
<td>66.51</td>
<td>89.71</td>
</tr>
<tr>
<td rowspan="7">CIR Dedicated</td>
<td>Pic2Word</td>
<td></td>
<td>8.72</td>
<td>9.51</td>
<td>10.46</td>
<td>11.29</td>
<td>23.90</td>
<td>51.70</td>
<td>65.30</td>
<td>87.80</td>
</tr>
<tr>
<td>SEARLE</td>
<td></td>
<td>11.68</td>
<td>12.73</td>
<td>14.33</td>
<td>15.12</td>
<td>24.24</td>
<td>52.48</td>
<td>66.29</td>
<td>88.84</td>
</tr>
<tr>
<td>SEARLE-XL-OTI</td>
<td></td>
<td>10.18</td>
<td>11.03</td>
<td>12.72</td>
<td>13.67</td>
<td>24.87</td>
<td>52.31</td>
<td>66.29</td>
<td>88.58</td>
</tr>
<tr>
<td>CIReVL</td>
<td></td>
<td>18.57</td>
<td>19.01</td>
<td>20.89</td>
<td>21.80</td>
<td>24.55</td>
<td>52.31</td>
<td>64.92</td>
<td>86.34</td>
</tr>
<tr>
<td>LinCIR</td>
<td></td>
<td>12.59</td>
<td>13.58</td>
<td>15.00</td>
<td>15.85</td>
<td>25.04</td>
<td>53.25</td>
<td>66.68</td>
<td>-</td>
</tr>
<tr>
<td>LDRE</td>
<td></td>
<td>23.35</td>
<td>24.03</td>
<td>26.44</td>
<td>27.50</td>
<td>26.53</td>
<td>55.57</td>
<td>67.54</td>
<td>88.50</td>
</tr>
<tr>
<td>FiRE</td>
<td></td>
<td>31.03</td>
<td>32.08</td>
<td>34.40</td>
<td>35.50</td>
<td>43.33</td>
<td>74.02</td>
<td>83.51</td>
<td>95.83</td>
</tr>
<tr>
<td rowspan="4">Agentic</td>
<td>MRA-CIR</td>
<td></td>
<td>27.14</td>
<td>28.85</td>
<td>31.54</td>
<td>32.63</td>
<td>37.98</td>
<td>67.45</td>
<td>78.07</td>
<td>93.98</td>
</tr>
<tr>
<td>AutoCIR</td>
<td></td>
<td>24.05</td>
<td>25.14</td>
<td>27.35</td>
<td>28.36</td>
<td>31.81</td>
<td>61.95</td>
<td>73.86</td>
<td>92.07</td>
</tr>
<tr>
<td>XR</td>
<td></td>
<td>31.38</td>
<td>32.88</td>
<td>35.46</td>
<td>36.50</td>
<td>43.13</td>
<td>73.59</td>
<td>83.09</td>
<td>94.05</td>
</tr>
<tr>
<td>OSCAR (Ours)</td>
<td></td>
<td>56.54</td>
<td>58.53</td>
<td>61.92</td>
<td>62.67</td>
<td>51.18</td>
<td>79.50</td>
<td>87.45</td>
<td>96.56</td>
</tr>
<tr>
<td colspan="2">Relative Improvement (%)</td>
<td></td>
<td>23.13%</td>
<td>24.19%</td>
<td>22.88%</td>
<td>21.81%</td>
<td>18.67%</td>
<td>7.40%</td>
<td>5.25%</td>
<td>-0.22%</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="2">Dress</th>
<th colspan="2">Shirt</th>
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
<td>Ops-MM-v1-7B</td>
<td>19.39</td>
<td>38.13</td>
<td>31.45</td>
<td>48.87</td>
<td>27.69</td>
<td>46.51</td>
<td>26.18</td>
<td>44.50</td>
</tr>
<tr>
<td>RzenEmbed-v2-7B</td>
<td>37.38</td>
<td>61.97</td>
<td>45.63</td>
<td>64.97</td>
<td>46.56</td>
<td>67.57</td>
<td>43.19</td>
<td>64.84</td>
</tr>
<tr>
<td>VLM2Vec</td>
<td>4.76</td>
<td>14.77</td>
<td>15.60</td>
<td>30.62</td>
<td>11.37</td>
<td>22.95</td>
<td>10.58</td>
<td>22.78</td>
</tr>
<tr>
<td>B3_Qwen2_7B</td>
<td>8.53</td>
<td>22.31</td>
<td>19.53</td>
<td>34.49</td>
<td>14.99</td>
<td>29.88</td>
<td>14.35</td>
<td>28.89</td>
</tr>
<tr>
<td>QQMM-embed-v2</td>
<td>36.44</td>
<td>60.09</td>
<td>46.07</td>
<td>65.65</td>
<td>46.35</td>
<td>68.49</td>
<td>42.95</td>
<td>64.74</td>
</tr>
<tr>
<td>bge-m3</td>
<td>10.81</td>
<td>23.75</td>
<td>20.66</td>
<td>33.66</td>
<td>15.96</td>
<td>27.33</td>
<td>15.81</td>
<td>28.25</td>
</tr>
<tr>
<td>Qwen3-Embed-0.6B</td>
<td>9.32</td>
<td>19.83</td>
<td>21.00</td>
<td>34.69</td>
<td>15.96</td>
<td>27.84</td>
<td>15.43</td>
<td>27.45</td>
</tr>
<tr>
<td>Qwen3-Embed-4B</td>
<td>12.69</td>
<td>27.02</td>
<td>26.69</td>
<td>40.68</td>
<td>21.88</td>
<td>36.05</td>
<td>20.42</td>
<td>34.58</td>
</tr>
<tr>
<td>Qwen3-Embed-8B</td>
<td>12.64</td>
<td>29.20</td>
<td>28.70</td>
<td>43.13</td>
<td>23.41</td>
<td>36.97</td>
<td>21.58</td>
<td>36.43</td>
</tr>
<tr>
<td>Pic2Word</td>
<td>20.20</td>
<td>40.20</td>
<td>26.20</td>
<td>43.60</td>
<td>27.90</td>
<td>47.40</td>
<td>24.77</td>
<td>43.73</td>
</tr>
<tr>
<td>SEARLE</td>
<td>20.48</td>
<td>43.13</td>
<td>26.89</td>
<td>45.58</td>
<td>29.32</td>
<td>49.97</td>
<td>25.56</td>
<td>46.23</td>
</tr>
<tr>
<td>SEARLE-XL-OTI</td>
<td>21.57</td>
<td>44.47</td>
<td>30.37</td>
<td>47.49</td>
<td>30.90</td>
<td>51.76</td>
<td>27.61</td>
<td>47.91</td>
</tr>
<tr>
<td>CIReVL</td>
<td>24.79</td>
<td>44.76</td>
<td>29.49</td>
<td>47.40</td>
<td>31.36</td>
<td>53.65</td>
<td>28.55</td>
<td>48.60</td>
</tr>
<tr>
<td>LinCIR</td>
<td>20.92</td>
<td>42.44</td>
<td>29.10</td>
<td>46.81</td>
<td>28.81</td>
<td>50.18</td>
<td>26.28</td>
<td>46.48</td>
</tr>
<tr>
<td>LDRE</td>
<td>22.93</td>
<td>46.76</td>
<td>31.04</td>
<td>51.22</td>
<td>31.57</td>
<td>53.64</td>
<td>28.51</td>
<td>50.54</td>
</tr>
<tr>
<td>FiRE</td>
<td>29.60</td>
<td>50.87</td>
<td>39.84</td>
<td>60.06</td>
<td>35.64</td>
<td>57.83</td>
<td>35.03</td>
<td>56.25</td>
</tr>
<tr>
<td>MRA-CIR</td>
<td>31.87</td>
<td>54.23</td>
<td>40.43</td>
<td>60.20</td>
<td>41.25</td>
<td>62.51</td>
<td>37.85</td>
<td>58.98</td>
</tr>
<tr>
<td>AutoCIR</td>
<td>24.94</td>
<td>45.81</td>
<td>34.00</td>
<td>53.43</td>
<td>33.10</td>
<td>55.58</td>
<td>30.68</td>
<td>51.61</td>
</tr>
<tr>
<td>XR</td>
<td>28.71</td>
<td>52.50</td>
<td>38.91</td>
<td>56.82</td>
<td>43.91</td>
<td>62.57</td>
<td>37.18</td>
<td>57.30</td>
</tr>
<tr>
<td>OSCAR (Ours)</td>
<td>38.47</td>
<td>65.15</td>
<td>44.50</td>
<td>67.52</td>
<td>48.24</td>
<td>71.24</td>
<td>43.73</td>
<td>67.97</td>
</tr>
<tr>
<td>Rel. Improv. (%)</td>
<td>2.92%</td>
<td>5.13%</td>
<td>-3.40%</td>
<td>2.84%</td>
<td>3.61%</td>
<td>4.02%</td>
<td>1.25%</td>
<td>4.82%</td>
</tr>
</tbody>
</table>

Key observations from public benchmark results:
1. **Overall Performance**: OSCAR achieves the best performance across all datasets, even when the golden library is built using only 10% of available training data. It is a training-free framework based on open-source models, outperforming even baselines that use domain-specific fine-tuning or closed-source models.
2. **vs. Single Embedding Methods**: OSCAR delivers large relative improvements: 23.13% mAP@5 on CIRCO, 76.60% R@1 on CIRR, and 1.25% average R@10 on FashionIQ. This demonstrates that optimized planning over multiple complementary tools delivers significantly better performance than relying on a single unified embedding.
3. **vs. CIR-dedicated Methods**: OSCAR outperforms all specialized CIR models, even without task-specific component design or fine-tuning. This shows that explicit optimized planning over general-purpose tools is a more effective and reusable approach than designing custom CIR-specific models.
4. **vs. Heuristic Agentic Methods**: OSCAR outperforms all prior agentic CIR approaches: 18.67% R@1 improvement on CIRR, 15.54% average R@10 improvement on FashionIQ, with no iterative agent interaction required. This validates that optimization-steered planning delivers better tool orchestration than heuristic trial-and-error loops.
### Industrial Gallery Results
Performance on the private industrial user photo gallery dataset is shown in Table 6 from the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Gallery1</th>
<th colspan="2">Gallery2</th>
<th colspan="2">Gallery3</th>
<th colspan="2">Avg</th>
</tr>
<tr>
<th>N@10</th>
<th>R@10</th>
<th>N@10</th>
<th>R@10</th>
<th>N@10</th>
<th>R@10</th>
<th>N@10</th>
<th>R@10</th>
</tr>
</thead>
<tbody>
<tr>
<td>VLM2Vec</td>
<td>44.29</td>
<td>47.78</td>
<td>44.91</td>
<td>48.69</td>
<td>38.96</td>
<td>41.86</td>
<td>42.72</td>
<td>46.11</td>
</tr>
<tr>
<td>B3_Qwen2_7B</td>
<td>42.87</td>
<td>47.32</td>
<td>41.84</td>
<td>45.71</td>
<td>37.49</td>
<td>40.76</td>
<td>40.73</td>
<td>44.60</td>
</tr>
<tr>
<td>Ops-MM-v1</td>
<td>51.78</td>
<td>53.74</td>
<td>48.95</td>
<td>51.44</td>
<td>43.98</td>
<td>46.99</td>
<td>48.24</td>
<td>50.72</td>
</tr>
<tr>
<td>RzenEmbed-v2-7B</td>
<td>51.04</td>
<td>53.79</td>
<td>49.62</td>
<td>51.93</td>
<td>46.23</td>
<td>47.59</td>
<td>48.96</td>
<td>51.10</td>
</tr>
<tr>
<td>QQMM-embed-v2</td>
<td>51.64</td>
<td>54.43</td>
<td>48.42</td>
<td>51.29</td>
<td>44.60</td>
<td>46.63</td>
<td>48.22</td>
<td>50.78</td>
</tr>
<tr>
<td>bge-m3</td>
<td>39.61</td>
<td>41.98</td>
<td>38.36</td>
<td>41.47</td>
<td>39.87</td>
<td>41.41</td>
<td>39.28</td>
<td>41.62</td>
</tr>
<tr>
<td>Qwen3-Embed-0.6B</td>
<td>40.73</td>
<td>42.78</td>
<td>40.11</td>
<td>42.54</td>
<td>39.60</td>
<td>41.11</td>
<td>40.15</td>
<td>42.14</td>
</tr>
<tr>
<td>Qwen3-Embed-4B</td>
<td>43.06</td>
<td>44.93</td>
<td>44.55</td>
<td>46.80</td>
<td>41.48</td>
<td>43.07</td>
<td>43.03</td>
<td>44.93</td>
</tr>
<tr>
<td>Qwen3-Embed-8B</td>
<td>42.92</td>
<td>45.05</td>
<td>44.57</td>
<td>46.86</td>
<td>41.89</td>
<td>43.13</td>
<td>43.13</td>
<td>45.01</td>
</tr>
<tr>
<td>OSCAR (Ours)</td>
<td>56.01</td>
<td>65.28</td>
<td>53.12</td>
<td>57.43</td>
<td>58.70</td>
<td>63.92</td>
<td>55.94</td>
<td>62.21</td>
</tr>
<tr>
<td>Rel.Improve. (%)</td>
<td>8.17%</td>
<td>19.93%</td>
<td>7.05%</td>
<td>10.59%</td>
<td>26.97%</td>
<td>34.31%</td>
<td>14.26%</td>
<td>21.74%</td>
</tr>
</tbody>
</table>

OSCAR consistently outperforms all baselines across all three industrial galleries, with average relative improvements of 14.26% NDCG@10 and 21.74% Recall@10 over the best baseline. This demonstrates that OSCAR's optimization-steered planning generalizes effectively to real-world scenarios with diverse image distributions and complex user intents.
## 6.2. Ablation Studies
The ablation study results verifying the contribution of OSCAR's core components are shown in Table 4 from the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Variants</th>
<th colspan="2">CIRCO</th>
<th colspan="2">CIRR</th>
<th colspan="2">FIQ.Avg</th>
</tr>
<tr>
<th>m@25</th>
<th>m@50</th>
<th>R@10</th>
<th>R@50</th>
<th>R@10</th>
<th>R@50</th>
</tr>
</thead>
<tbody>
<tr>
<td>OSCAR (Ours)</td>
<td>61.92</td>
<td>62.67</td>
<td>87.45</td>
<td>96.56</td>
<td>43.73</td>
<td>67.97</td>
</tr>
<tr>
<td>w/o Demo.</td>
<td>59.72</td>
<td>59.84</td>
<td>75.01</td>
<td>82.02</td>
<td>46.57</td>
<td>61.62</td>
</tr>
<tr>
<td>w/o Set.Diff.</td>
<td>59.54</td>
<td>59.62</td>
<td>87.32</td>
<td>93.80</td>
<td>46.46</td>
<td>62.08</td>
</tr>
<tr>
<td>w/o Set.Diff. & Set.Int.</td>
<td>58.63</td>
<td>58.66</td>
<td>39.11</td>
<td>54.53</td>
<td>49.44</td>
<td>53.96</td>
</tr>
</tbody>
</table>

Key observations from ablation studies:
1. **Impact of Golden Demonstrations**: Removing the optimal trajectory demonstrations (w/o Demo.) causes significant performance drops across all datasets, especially on CIRR (R@10 down 12.44, R@50 down 14.54). This confirms that the MIP-derived trajectories provide critical guidance for the VLM planner to select and compose retrieval tools effectively.
2. **Impact of Set-Theoretic Operations**: Removing the set difference operation (w/o Set.Diff.) degrades performance, particularly for Recall@50 across datasets, as it eliminates the ability to filter out unwanted attributes. Removing both set difference and set intersection (reducing composition to simple union of results) causes drastic performance drops, confirming that structured set operations are critical for OSCAR's ability to handle inclusion/exclusion constraints and filter noise.
   Additional analysis results:
- **VLM Backbone Generalization**: OSCAR consistently improves performance across all tested VLM backbones (Qwen3-VL 4B/8B/32B, InternVL3.5 38B), with stronger VLMs achieving larger gains. This demonstrates OSCAR is a plug-and-play framework compatible with any VLM with tool-calling capabilities.
- **Robustness to Number of Demonstrations**: OSCAR's performance remains stable when varying the number of in-context demonstrations from 1 to 4, as shown in Figure 3 from the original paper:

  ![该图像是一个比较不同演示次数下，三组基准任务（CIRCO、CIRR和FashionIQ）性能的柱状图。图中展示了每组任务在不同演示数量下的平均精度（mAP）和召回率（Recall），用不同颜色表示各性能指标。](images/3.jpg)
  *该图像是一个比较不同演示次数下，三组基准任务（CIRCO、CIRR和FashionIQ）性能的柱状图。图中展示了每组任务在不同演示数量下的平均精度（mAP）和召回率（Recall），用不同颜色表示各性能指标。*

  This confirms that the golden library transfers generalizable planning logic, not just query-specific memorized answers, so only a small number of demonstrations are required.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
OSCAR is a novel optimization-steered agentic planning framework for composed image retrieval that addresses the core limitations of both unified embedding and heuristic agentic CIR methods. By reframing agentic CIR as a principled trajectory optimization problem, OSCAR uses two-stage MIP to derive optimal retrieval trajectories for training samples offline, which are stored in a golden library and used as in-context demonstrations to steer a VLM planner to produce near-optimal plans in a single inference pass online.
Extensive experiments demonstrate OSCAR consistently outperforms all SOTA baselines on 3 public CIR benchmarks and 1 private industrial dataset, even when using only 10% of training data for golden library construction. It generalizes well across VLM backbones and real-world scenarios, demonstrating that combining mathematical optimization with agentic AI delivers significant performance and efficiency gains for complex multimodal reasoning tasks.
## 7.2. Limitations & Future Work
The authors identify the following limitations and future research directions:
1. **Fixed Set Composition Structure**: The current two-stage MIP uses a fixed two-clause set composition structure for stability and interpretability, which may not be optimal for all query types. Future work will explore more flexible composition structures to further improve performance.
2. **Negative Polarity Mismatch**: A common failure mode is polarity mismatch in the negative branch, where the planner mistakenly retrieves the negated concept (e.g., "one beetle" instead of "multiple beetles" for exclusion) leading to incorrect set difference operations. Future work will improve guidance for negative query generation to reduce this error.
3. **Generalization to Other Tasks**: The optimization-steered agent planning paradigm is not limited to CIR. Future work will extend this approach to other complex reasoning tasks, such as multi-hop question answering, robot task planning, and open-ended tool use.
## 7.3. Personal Insights & Critique
This work represents a highly promising new direction for agentic AI research, with both academic and industrial value:
- **Key Inspiration**: The core insight of using operations research methods (MIP) to generate high-quality in-context demonstrations for agent planning avoids the many limitations of heuristic agent loops and expensive fine-tuning/RLHF. This paradigm is applicable to any agentic task where ground truth is available for training data, including document retrieval, code generation, and robot planning.
- **Transferability**: The OSCAR framework can be easily adapted to other retrieval tasks with minimal modification, as it is agnostic to the specific retrieval tools used and relies on generalizable planning logic.
- **Potential Improvements**:
  1. The offline MIP solving cost may be high for very large training datasets, though it is a one-time offline cost. Future work can explore approximate MIP solvers or trajectory clustering to reduce this cost.
  2. The current trajectory retrieval relies on semantic similarity of query text, using structural similarity of query intent (e.g., type of modification, number of constraints) could improve the relevance of retrieved demonstrations.
  3. The atomic retrieval space is fixed in the current work, dynamically generating custom atomic retrievals for each query type could further improve performance.
     Overall, this work successfully bridges the gap between classic operations research and modern agentic AI, delivering a practical, high-performance solution for CIR while opening a new research direction for optimization-steered agent planning.