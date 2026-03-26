# 1. Bibliographic Information
## 1.1. Title
The paper's title is *HGNet: Scalable Foundation Model for Automated Knowledge Graph Generation from Scientific Literature*. Its central topic is the design of an end-to-end, lightweight, zero-shot capable framework for building high-quality, hierarchical, logically consistent knowledge graphs from unstructured scientific text, addressing key limitations of existing NER/RE methods and large LLMs for this specialized task.
## 1.2. Authors
The authors are:
- Devvrat Joshi: PhD researcher at BASIRA Lab, Imperial-X, Department of Computing, Imperial College London, UK, with research focus on natural language processing, graph learning, and scientific information extraction.
- Islem Rekik: Principal Investigator of BASIRA Lab, Imperial College London, with research expertise in graph neural networks, medical imaging, and knowledge representation for scientific domains.
  Both authors are affiliated with Imperial College London, a top-tier global institution for computer science and AI research.
## 1.3. Journal/Conference
As of the current date, the paper is published as a preprint on arXiv, and has not yet been peer reviewed or formally accepted for publication in a conference or journal.
## 1.4. Publication Year
The paper was published on arXiv on 2026-03-24 (UTC).
## 1.5. Abstract
This work aims to resolve four persistent gaps in automated scientific knowledge graph (KG) construction: poor recognition of long multi-word entities, limited cross-domain generalization, ignorance of the hierarchical structure of scientific knowledge, and the high computational cost/inconsistent performance of large general-purpose LLMs for specialized scientific tasks. The authors propose a two-stage end-to-end framework:
1.  Z-NERD: A zero-shot NER model with two core innovations: Orthogonal Semantic Decomposition (OSD) for domain-agnostic entity boundary detection, and Multi-Scale TCQK attention for accurate multi-word entity recognition.
2.  HGNet: A hierarchy-aware graph neural network for relation extraction (RE), with three-channel message passing for parent/child/peer relations, plus two structural regularizers: Differentiable Hierarchy Loss to enforce acyclic DAG structure (no cycles/shortcut edges), and Continuum Abstraction Field (CAF) Loss to embed abstraction levels along a learnable Euclidean axis (a simpler alternative to hyperbolic hierarchical embeddings).
    The authors also release SPHERE, the first large multi-domain benchmark for hierarchical RE, with 10,000 documents and 111,000 annotated relations across 4 scientific domains. The framework sets new state-of-the-art (SOTA) performance across all tested benchmarks, with 8.08% improvement in NER, 5.99% improvement in RE on out-of-distribution test sets, and 10.76% NER / 26.2% RE improvement in zero-shot settings.
## 1.6. Original Source Link
- Preprint source: https://arxiv.org/abs/2603.23136v1
- PDF link: https://arxiv.org/pdf/2603.23136v1
- Publication status: Public preprint, not yet peer reviewed for formal conference/journal publication as of 2026-03-25.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
The exponential growth of scientific literature (now far exceeding human manual review capacity) creates an urgent need for automated systems that distill unstructured text into structured, machine-readable knowledge graphs for research exploration and synthesis. Existing KG construction methods suffer from four critical, unresolved limitations:
1.  Poor performance on long multi-word scientific entities (e.g., "in situ transmission electron microscopy"), as standard models treat token boundaries as incidental rather than an explicit optimization objective.
2.  Limited cross-domain generalization: Supervised models degrade sharply on out-of-domain text, while large LLMs (7B+ parameters) are computationally prohibitively expensive for routine large-scale KG construction.
3.  Ignorance of the hierarchical nature of scientific knowledge (e.g., "Deep Learning" is a subfield of "Machine Learning"): Conventional models rely on shallow co-occurrence statistics and fail to model layered conceptual structures.
4.  Lack of global consistency guarantees: Existing methods produce KGs with logical contradictions (e.g., cycles, conflicting hierarchical relations) that undermine their reliability for downstream use.
    These limitations result in shallow, inconsistent scientific KGs with limited practical utility.
### Innovative Entry Point
The paper proposes the first end-to-end framework that explicitly addresses all four of the above challenges, via carefully designed structural inductive biases for entity recognition, hierarchical relation modeling, and global consistency regularization, plus a new large-scale benchmark to support research in this space.
## 2.2. Main Contributions / Findings
The paper's four core contributions are:
1.  **Z-NERD**: A lightweight (~300M total parameters for the full pipeline) zero-shot NER model that outperforms all existing SOTA NER baselines on scientific benchmarks, via OSD (domain-agnostic semantic turn detection) and Multi-Scale TCQK attention (explicit n-gram pattern specialization for multi-word entities).
2.  **HGNet**: A hierarchy-aware GNN for RE that sets new SOTA on hierarchical RE benchmarks, via three separate message passing channels for parent/child/peer relations, explicitly modeling the directional flow of hierarchical information.
3.  **Geometric Theory of Abstraction**: The first method to formalize hierarchical abstraction as a continuous geometric property of standard Euclidean embedding space, via the CAF Loss that learns a universal "abstraction axis" for concept embedding. This approach is simpler, more interpretable, and more effective than complex hyperbolic embedding methods for scientific KG construction.
4.  **SPHERE Benchmark**: The first large-scale multi-domain benchmark for hierarchical RE, with 10,000 documents and 111,000 annotated relations across Computer Science, Physics, Biology, and Material Science, built via a constrained LLM generate-and-annotate pipeline to ensure global hierarchical consistency.
    Key findings: The proposed framework significantly outperforms both specialized SOTA models and large general-purpose LLMs across all tested benchmarks, with especially strong zero-shot cross-domain performance, proving that explicit structural inductive biases for hierarchy and consistency deliver far better performance per parameter than unstructured large models for this specialized task.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
This section explains all core technical terms required to understand the paper, for a beginner audience:
### 3.1.1. Knowledge Graph (KG)
A structured representation of knowledge, where:
- Nodes = entities (e.g., "Machine Learning", "LSTM", "AdamW Optimizer")
- Edges = semantic relations between pairs of entities (e.g., "is a subfield of", "uses", "is part of")
  KGs are machine-readable, enabling automated search, reasoning, and synthesis of structured knowledge.
### 3.1.2. Named Entity Recognition (NER)
A core NLP task that identifies spans of text corresponding to named entities, and classifies them into predefined categories (e.g., `Method`, `Dataset`, `Concept` for scientific text). For example, in the sentence "We use BERT for text classification", NER would identify "BERT" as a `Method` entity.
### 3.1.3. Relation Extraction (RE)
A core NLP task that identifies semantic relations between pairs of extracted entities, outputting triplets of the form `(head entity, relation, tail entity)`. For the example sentence above, RE would output the triplet `("text classification", "uses", "BERT")`.
### 3.1.4. Zero-Shot Learning
A machine learning paradigm where a model can perform a task on data from domains or classes it has never explicitly seen during training, with no additional fine-tuning required. This is critical for generalizing across diverse scientific domains without expensive retraining.
### 3.1.5. Graph Neural Network (GNN)
A class of neural networks designed to process graph-structured data. GNNs learn representations for nodes and edges by propagating "messages" between connected nodes, aggregating information from local neighbors to build global context-aware representations.
### 3.1.6. Transformer & Self-Attention
The Transformer is the dominant deep learning architecture for NLP tasks, built on the self-attention mechanism, which computes the relevance (weight) of every token in a sequence relative to every other token, enabling capture of long-range dependencies. The standard self-attention formula is:
\$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\$
Where:
- $Q$ (Query), $K$ (Key), $V$ (Value) = matrices derived from input token embeddings via separate linear projections
- $d_k$ = dimension of the Key vectors, the $\sqrt{d_k}$ term normalizes scores to prevent gradient instability
- The softmax function converts raw similarity scores into normalized weights that sum to 1 for each query token.
### 3.1.7. Euclidean vs Hyperbolic Embeddings
- Euclidean embeddings: Representations in standard flat Cartesian space, simple to train and interpret, but distort deep hierarchical tree-like structures (they cannot represent nested hierarchies without large distortion of distances between nodes).
- Hyperbolic embeddings: Representations in curved non-Euclidean space, naturally optimized for hierarchical tree data, but require complex Riemannian optimization, are harder to interpret, and struggle to model non-hierarchical "peer" relations between nodes at the same abstraction level.
## 3.2. Previous Works
The paper categorizes prior work along the three core components of scientific KG construction:
### 3.2.1. Scientific NER
Prior SOTA approaches rely on domain-specific pre-trained Transformers like SciBERT (Beltagy et al., 2019) and BioBERT (Lee et al., 2019), fine-tuned on task-specific labeled data. These models achieve strong in-domain performance, but:
1.  Fail to reliably recognize long multi-word entities, as this ability is only an emergent property of contextual embeddings, not an explicit optimization objective.
2.  Degrade sharply on out-of-domain text, as they overfit to domain-specific vocabulary.
    Zero-shot NER methods like GLiNER (Zaratiana et al., 2023) and UniversalNER (Zhou et al., 2024) reformulate NER as a span matching task, but still rely on surface semantic similarity, rather than domain-agnostic structural signals in text.
### 3.2.2. Scientific Relation Extraction
Early RE models used pipeline architectures (NER first, then RE), which suffered from error propagation from the NER step to the RE step. Joint models that extract entities and relations simultaneously (e.g., PL-Marker (Ye et al., 2022), HGERE (Yan et al., 2023)) improved performance, but are limited to sentence-level reasoning, and fail to capture long-range hierarchical dependencies across sentences/documents.
GNN-based RE methods use co-occurrence and syntactic proximity features to build graphs, but conflate textual adjacency with genuine conceptual relatedness, and are "hierarchy-blind" (they use uniform message passing across all edges, with no distinction between hierarchical relation directions).
### 3.2.3. Hierarchical Knowledge Representation
The standard approach for modeling hierarchical data is hyperbolic embeddings (Nickel & Kiela, 2017), which embed tree structures in curved Poincaré ball space with low distortion. However, these methods require complex non-Euclidean optimization, are difficult to interpret, and perform poorly on non-hierarchical peer relations that are common in scientific KGs.
## 3.3. Technological Evolution
The field of scientific KG construction has evolved along the following timeline:
1.  2010s: Rule-based and statistical supervised models for NER/RE, limited to narrow specific domains, requiring extensive manual feature engineering.
2.  2018-2020: Domain-pre-trained Transformer models (SciBERT, BioBERT) drastically improve in-domain NER/RE performance, but suffer from poor cross-domain generalization and limited ability to model hierarchical structure.
3.  2021-2024: Zero-shot NER/RE methods, GNN-based RE, and hyperbolic hierarchical embeddings address individual limitations, but no end-to-end system addresses all four core challenges (multi-word entities, generalization, hierarchy modeling, consistency) simultaneously.
4.  2026 (this work): The first end-to-end framework that explicitly addresses all four core challenges, with a lightweight model that outperforms both specialized SOTA models and large general-purpose LLMs for scientific KG construction.
## 3.4. Differentiation Analysis
The paper's approach has four core differentiations from prior work:
1.  **NER Design**: Unlike prior NER models that rely on emergent multi-word recognition and surface semantic features, Z-NERD uses explicit OSD to detect domain-agnostic semantic turns (entity boundary signals) and Multi-Scale TCQK attention that forces heads to specialize in n-gram patterns for multi-word entity recognition.
2.  **RE Design**: Unlike hierarchy-blind GNNs that use uniform message passing, HGNet uses three separate dedicated message passing channels for parent-to-child, child-to-parent, and peer-to-peer relations, explicitly modeling the directional flow of hierarchical information.
3.  **Hierarchy Consistency**: Instead of complex hyperbolic embeddings, the paper uses a learnable Euclidean abstraction axis (CAF Loss) that is simpler, more interpretable, and supports both hierarchical and peer relations, plus a Differentiable Hierarchy Loss to enforce a valid DAG structure (no cycles, no shortcut edges).
4.  **Benchmark**: SPHERE is the first large multi-domain benchmark with globally consistent hierarchical structure (unlike prior benchmarks that have local per-document hierarchies), addressing critical data scarcity for hierarchical RE research.

# 4. Methodology
## 4.1. Principles
The core guiding principle of the work is that scientific KG construction requires explicit structural inductive biases, rather than relying on emergent capabilities of overparameterized large models. Specifically:
1.  Entity recognition can be made domain-agnostic and robust to long multi-word entities by explicitly modeling structural signals in text (semantic turns, n-gram entity boundaries).
2.  Relation extraction can be made accurate and consistent by explicitly modeling hierarchical structure, and enforcing both logical (acyclic DAG) and geometric (abstraction ordering) constraints on the KG.
    The framework uses a two-stage decoupled design (NER first, then RE) to simplify optimization, while co-training both stages with a shared SciBERT encoder to ensure end-to-end consistency.
## 4.2. Core Methodology In-depth
### 4.2.1. Stage 1: Z-NERD (Zero-Shot Entity Recognition)
Z-NERD processes raw scientific text to extract entity mentions, with two core components: Orthogonal Semantic Decomposition (OSD) and Multi-Scale TCQK attention.
#### 4.2.1.1. Orthogonal Semantic Decomposition (OSD)
OSD is designed to extract domain-agnostic signals of "semantic turns" (points where new concepts/entities are introduced) to enable zero-shot generalization, based on the hypothesis that shifts in semantic direction are consistent across domains, independent of specific vocabulary.
Step 1: Compute the change vector between the contextual embedding of the current word ($E_{\text{text}_t}$) and the previous word ($E_{\text{text}_{t-1}}$), both derived from the shared SciBERT encoder:
\$
\Delta E_t = E_{\text{text}_t} - E_{\text{text}_{t-1}}
\$
Step 2: Decompose $\Delta E_t$ into two orthogonal components:
1.  **Sustaining component**: The projection of $\Delta E_t$ onto the previous word's embedding, representing elaboration of the existing concept (no semantic turn):
    \$
    v_{\text{sustaining}_t} = \frac{\Delta E_t \cdot E_{\text{text}_{t-1}}}{\|E_{\text{text}_{t-1}}\|^2} E_{\text{text}_{t-1}}
    \$
    Where $\cdot$ is the dot product operation, and $\|E_{\text{text}_{t-1}}\|$ is the L2 norm of the previous word's embedding, used to normalize the projection.
2.  **Divergent component**: The component of $\Delta E_t$ orthogonal to the previous word's embedding, representing a semantic turn (introduction of a new concept/entity):
    \$
    v_{\text{divergent}_t} = \Delta E_t - v_{\text{sustaining}_t}
    \$
Step 3: The enriched input embedding for the next layer is the concatenation of the original current word embedding and the divergent component: $[E_{\text{text}_t} || v_{\text{divergent}_t}]$, where `||` denotes vector concatenation. This gives the model an explicit, domain-agnostic signal of when a new entity starts.
#### 4.2.1.2. Multi-Scale Temporal Convolutional Queries & Keys (TCQK) Attention
TCQK attention is designed to improve recognition of multi-word entities of varying lengths, based on the hypothesis that explicit specialization of attention heads to n-gram patterns of different lengths will outperform standard attention's emergent ability to capture multi-word spans.
Standard self-attention uses linear projections for Query (Q), Key (K), and Value (V) matrices, with no explicit bias for local n-gram patterns. TCQK modifies Q and K before attention calculation using 1D convolutions of varying kernel sizes:
Step 1: Split the total H attention heads into G groups, each group g assigned a 1D convolutional kernel $C_g$ with a fixed kernel size $k_g$ (e.g., kernel sizes 1, 3, 5, 7 for 1-gram, 3-gram, 5-gram, 7-gram patterns respectively).
Step 2: For each attention head h in group g, compute convolved Q and K matrices using the group's convolution kernel:
\$
Q_{\text{conv}, h} = C_g(Q_h); \quad K_{\text{conv}, h} = C_g(K_h)
\$
Where $Q_h$ and $K_h$ are the original Query and Key matrices for head h, derived from the enriched OSD embeddings via linear projection. The 1D convolution is applied along the sequence dimension, forcing heads in each group to focus on local n-gram patterns of the kernel size.
Step 3: Compute attention scores using the convolved Q and K matrices, with standard softmax normalization and value multiplication as in standard self-attention. This modification ensures different heads specialize in capturing entities of different lengths, from short acronyms (1-2 tokens) to long technical terms (5+ tokens).
The following figure (Figure 6 from the original paper) illustrates the full Z-NERD pipeline:

![Figure 6: Main figure explaining the proposed Z-NERD algorithm. For TCQK, multi-head for each convolution has been shown as single head for simplicity. B refers to begin entity, I refers to inside entity and O refers to outside entity.](images/6.jpg)
*该图像是示意图，展示了提议的 Z-NERD 算法的工作机制。图中首先通过 SciBERT 编码器进行嵌入，并利用正交语义分解生成融合嵌入。接着，TCQK 机制使用卷积操作处理查询、键和值的投影矩阵，并最终生成嵌入和注意力得分。最后，通过 MLP 分类器对识别的实体进行标记，分别标识为 B-Tag、I-Tag 和 O-Tag。*

---
### 4.2.2. Stage 2: HGNet (Hierarchy Graph Network)
HGNet takes the contextual entity embeddings output from Z-NERD as input, and learns hierarchical and peer relations between entities to construct a globally consistent KG, with three core components: Probabilistic Hierarchical Message Passing, Differentiable Hierarchy Loss, and Continuum Abstraction Field Loss.
#### 4.2.2.1. Probabilistic Hierarchical Message Passing
This component explicitly models hierarchical relation directions via separate message passing channels, based on the hypothesis that GNNs can preserve hierarchical structure only if they distinguish between information flowing up from children, down from parents, and sideways from peers.
Step 1: A Latent Relation Predictor (MLP) estimates the probability distribution over three internal relation types for every pair of entity nodes (u, v):
\$
P_{uv} = \mathrm{softmax}\left(\mathrm{MLP}([h_u || h_v])\right)
\$
Where:
- $h_u, h_v$ = contextual embeddings of entities u and v from the shared SciBERT encoder
- $P_{uv}$ = 3-dimensional vector with probabilities for the three internal relations: `parent-of` (u is parent of v), `peer-of` (u and v are at the same abstraction level), `no-edge` (no relation between u and v)
- These probabilities are used as soft edge weights for message passing.
  Step 2: Aggregate messages from three separate relation channels, each with its own learnable weight matrix to capture unique relational semantics:
1.  **Parental (Upstream) Aggregation**: Aggregate messages from all parent nodes of v (nodes where u is parent of v):
    \$
    \pmb{m}_v^{\mathrm{parents}} = \sum_{u \in V} P_{uv}^{\mathrm{parent}} \cdot (W_{\mathrm{up}} \pmb{h}_u^{(k)})
    \$
    Where $P_{uv}^{\mathrm{parent}}$ is the probability that u is parent of v, $W_{\mathrm{up}}$ is the learnable weight matrix for parent-to-child messages, $\pmb{h}_u^{(k)}$ is the embedding of node u at GNN layer k, and $V$ is the set of all entity nodes.
2.  **Child (Downstream) Aggregation**: Aggregate messages from all child nodes of v (nodes where v is parent of u):
    \$
    \pmb{m}_v^{\mathrm{children}} = \sum_{u \in V} P_{vu}^{\mathrm{parent}} \cdot (W_{\mathrm{down}} \pmb{h}_u^{(k)})
    \$
    Where $P_{vu}^{\mathrm{parent}}$ is the probability that v is parent of u (so u is child of v), and $W_{\mathrm{down}}$ is the learnable weight matrix for child-to-parent messages.
3.  **Peer Aggregation**: Aggregate messages from all peer nodes of v (nodes where u is peer of v):
    \$
    \pmb{m}_v^{\mathrm{peers}} = \sum_{u \in V} P_{uv}^{\mathrm{peer}} \cdot (W_{\mathrm{peer}} \pmb{h}_u^{(k)})
    \$
    Where $P_{uv}^{\mathrm{peer}}$ is the probability that u is peer of v, and $W_{\mathrm{peer}}$ is the learnable weight matrix for peer-to-peer messages.
Step 3: Update the node embedding by concatenating the original node embedding with the three aggregated messages, then passing through an update MLP to produce the embedding for GNN layer k+1:
\$
\pmb{h}_v^{(k+1)} = \mathrm{UpdateMLP}\left([\pmb{h}_v^{(k)} || \pmb{m}_v^{\mathrm{parents}} || \pmb{m}_v^{\mathrm{children}} || \pmb{m}_v^{\mathrm{peers}}]\right)
\$
This design ensures node embeddings capture context from all hierarchical directions, preserving the layered structure of scientific knowledge.
#### 4.2.2.2. Differentiable Hierarchy Loss (DHL, $\mathcal{L}_{\mathrm{hierarchy}}$)
DHL enforces the parent-of subgraph forms a valid Directed Acyclic Graph (DAG) with no cycles or shortcut edges that skip intermediate hierarchical levels, based on the hypothesis that explicit penalization of structural inconsistencies will improve global KG consistency.
DHL is a regularizer applied to the parent-of adjacency matrix $A_{\mathrm{parent}}$, where $A_{\mathrm{parent}}[u,w] = P_{uw}^{\mathrm{parent}}$ (the probability u is parent of w). It is a weighted sum of two components:
\$
\mathcal{L}_{\mathrm{hierarchy}} = \lambda_{\mathrm{acyclic}} \mathcal{L}_{\mathrm{acyclic}} + \lambda_{\mathrm{separation}} \mathcal{L}_{\mathrm{separation}}
\$
Where $\lambda_{\mathrm{acyclic}}$ and $\lambda_{\mathrm{separation}}$ are hyperparameters weighting the two loss components.
1.  **Acyclicity Loss ($\mathcal{L}_{\mathrm{acyclic}}$)**: Penalizes cycles in the parent-of graph. A graph is a valid DAG if and only if it contains no cycles of any length, which is enforced via the trace of the matrix exponential:
    \$
    \mathcal{L}_{\mathrm{acyclic}} = \mathrm{tr}\left(e^{A_{\mathrm{parent}} \circ A_{\mathrm{parent}}}\right) - d
    \$
    Where:
    - $\mathrm{tr}(\cdot)$ = the trace of a matrix (sum of its diagonal entries)
    - $e^{\cdot}$ = matrix exponential, computed via Taylor series expansion: $e^X = \sum_{k=0}^\infty \frac{X^k}{k!}$
    - $\circ$ = element-wise (Hadamard) product operation
    - $d$ = number of entity nodes in the graph
      This loss equals 0 if and only if the graph is a perfect DAG (no cycles), and is positive otherwise, as cycles add to the trace of the matrix exponential.
2.  **Hierarchical Separation Loss ($\mathcal{L}_{\mathrm{separation}}$)**: Penalizes shortcut edges that skip intermediate hierarchical levels (e.g., directly connecting a grandparent to a grandchild instead of routing through the parent node):
    \$
    \mathcal{L}_{\mathrm{separation}} = \sum_{u,w} (A_{\mathrm{parent}}^2)_{uw} \cdot (A_{\mathrm{parent}})_{uw}
    \$
    Where:
    - $A_{\mathrm{parent}}^2$ = matrix square of $A_{\mathrm{parent}}$, where entry $(A_{\mathrm{parent}}^2)_{uw}$ equals the sum of weights of all 2-step paths from u to w (e.g., u → v → w, meaning u is grandparent of w)
    - The element-wise product with $(A_{\mathrm{parent}})_{uw}$ selects only pairs where there is both a 2-step path from u to w *and* a direct edge u→w (a shortcut edge). The loss penalizes these shortcuts to enforce a strict parent-child hierarchy.
#### 4.2.2.3. Continuum Abstraction Field (CAF) Loss ($\mathcal{L}_{\mathrm{caf}}$)
CAF Loss embeds hierarchical abstraction levels as a continuous geometric property of the standard Euclidean embedding space, based on the hypothesis that all scientific concepts can be organized along a single universal "axis of abstraction", where higher values correspond to more abstract/general concepts, and lower values correspond to more specific/concrete concepts.
First, define a learnable unit vector $\pmb{w}_{\mathrm{abs}}$ (Abstraction Field Vector) that defines the universal abstraction axis in the embedding space. The abstraction score of any entity v is its dot product projection onto this axis:
\$
\hat{y}_{\mathrm{abs}}(v) = h_v \cdot w_{\mathrm{abs}}
\$
Where $h_v$ is the embedding of entity v. This abstraction score is a continuous real value, not discrete levels, modeling abstraction as a fluid continuum.
CAF Loss is a weighted sum of three components:
\$
\mathcal{L}_{\mathrm{caf}} = \mathcal{L}_{\mathrm{ranking}} + \gamma_1 \mathcal{L}_{\mathrm{anchor}} + \gamma_2 \mathcal{L}_{\mathrm{regression}}
\$
Where $\gamma_1$ and $\gamma_2$ are hyperparameters weighting the components.
1.  **Ranking Component ($\mathcal{L}_{\mathrm{ranking}}$)**: Enforces parent entities (more abstract) have higher abstraction scores than their child entities (more specific), with a fixed margin $\delta$:
    \$
    \mathcal{L}_{\mathrm{ranking}} = \frac{1}{|\mathcal{E}_{\mathrm{part-of}}|} \sum_{(c,p) \in \mathcal{E}_{\mathrm{part-of}}} \max\left(0, (h_c - h_p) \cdot w_{\mathrm{abs}} + \delta\right)
    \$
    Where:
    - $\mathcal{E}_{\mathrm{part-of}}$ = set of all ground-truth parent-child (part-of) relation pairs, with `(c,p)` representing child c and parent p
    - $|\mathcal{E}_{\mathrm{part-of}}|$ = size of the set, for normalization
    - $\delta$ = margin hyperparameter, ensuring the difference between parent and child abstraction scores is at least $\delta$
      If the parent p has an abstraction score at least $\delta$ higher than child c, the term inside the max is ≤ 0, so no loss is added. Otherwise, loss is added to correct the ordering.
2.  **Anchoring Component ($\mathcal{L}_{\mathrm{anchor}}$)**: Pins known root nodes (most abstract, e.g., "Computer Science") to an abstraction score of 1, and known leaf nodes (most specific, e.g., "AdamW Optimizer") to an abstraction score of 0:
    \$
    \mathcal{L}_{\mathrm{anchor}} = \frac{1}{|\mathcal{V}_s|} \sum_{v_s \in \mathcal{V}_s} (\pmb{h}_{v_s} \cdot \pmb{w}_{\mathrm{abs}} - 1)^2 + \frac{1}{|\mathcal{V}_t|} \sum_{v_t \in \mathcal{V}_t} (\pmb{h}_{v_t} \cdot \pmb{w}_{\mathrm{abs}} - 0)^2
    \$
    Where:
    - $\mathcal{V}_s$ = set of ground-truth root (source) nodes (highest abstraction level)
    - $\mathcal{V}_t$ = set of ground-truth leaf (target) nodes (lowest abstraction level)
      This is a mean squared error loss that pulls root nodes to score 1 and leaf nodes to score 0, anchoring the abstraction axis range.
3.  **Regression Component ($\mathcal{L}_{\mathrm{regression}}$)**: Pulls predicted abstraction scores towards ground-truth topological depth scores $y_{\mathrm{topo}}(v)$, which are derived by topological sort of the ground-truth KG hierarchy (normalized to 0 = leaf, 1 = root):
    \$
    \mathcal{L}_{\mathrm{regression}} = \frac{1}{|\mathcal{V}_{\mathrm{train}}|} \sum_{v \in \mathcal{V}_{\mathrm{train}}} \left((h_v \cdot w_{\mathrm{abs}}) - y_{\mathrm{topo}}(v)\right)^2
    \$
    Where $\mathcal{V}_{\mathrm{train}}$ is the set of entity nodes in the training set. This component ensures abstraction scores align with the actual hierarchical depth of entities, not just relative ordering.
The following figure (Figure 1 from the original paper) illustrates the continuous abstraction axis for physics topics:

![Figure 1: Continuous axis of abstraction for topics in physics.](images/1.jpg)
*该图像是一个展示物理学主题抽象连续体的示意图。图中垂直轴表示抽象程度，顶部的红色节点为高级概念，底部的黄色节点为具体领域，展示了从基本法则到应用领域的层次关系。*

#### 4.2.2.4. Final Prediction & Joint Optimization
The three internal relation types (parent-of, peer-of, no-edge) are only used for structural regulation. The final task-specific relation triplets (e.g., "uses", "is a subfield of", as required by evaluation benchmarks) are predicted by a downstream classification head that takes the final HGNet node embeddings $\pmb{h}_v^{(k+1)}$ as input. The loss for this task is $\mathcal{L}_{\mathrm{RE}}$, the primary training objective.
All components are trained end-to-end with the following total composite loss:
\$
\mathcal{L}_{\mathrm{Total}} = \mathcal{L}_{\mathrm{RE}} + \lambda_1 \mathcal{L}_{\mathrm{hierarchy}} + \lambda_2 \mathcal{L}_{\mathrm{caf}}
\$
Where $\lambda_1$ and $\lambda_2$ are hyperparameters weighting the structural regularizers relative to the primary RE task loss.
The following figure (Figure 7 from the original paper) illustrates the full HGNet pipeline:

![Figure 7: Main figure of proposed HGNet illustrating all proposed components. For clarity, we omit $\\mathcal { L } _ { \\mathrm { r e g r e s s i o n } }$ , since it is simply a regression loss applied over the graph topology, similar in nature to standard losses such as mean squared error or binary cross entropy.](images/7.jpg)
*该图像是示意图，展示了提出的HGNet框架的各个组成部分。图中包含了模型的层次关系损失、概率消息传递机制以及不同损失函数（包括$L_{\text{acyclic}}$和$L_{\text{separation}}$）的概念。整体结构清晰地展示了在图的拓扑结构上如何进行节点的更新与关系预测，从而实现科学知识图谱的自动构建。*

# 5. Experimental Setup
## 5.1. Datasets
The paper evaluates the framework on 5 datasets spanning multiple scientific domains:
1.  **SciERC**: Standard scientific information extraction benchmark, focused on Computer Science (AI) domain, with 500 abstracts and ~4,600 relations, using local document-level hierarchies.
2.  **SciER**: Computer Science domain benchmark, with 106 full papers and ~12,000 relations, focusing on relations between datasets, methods, and research tasks.
3.  **BioRED**: Biomedical domain benchmark, with 600 abstracts and ~38,000 relations, containing only peer relations (no hierarchical relations).
4.  **SemEval-2017 Task 10 (ScienceIE)**: Multi-domain scientific benchmark, focused on keyphrase extraction and relation extraction across multiple scientific fields.
5.  **SPHERE**: New benchmark proposed in the paper, spanning 4 domains (Computer Science, Physics, Biology, Material Science), with 10,000 documents and 111,000 annotated relations, built from a global KG scaffold to ensure hierarchical consistency across the entire corpus (not per-document local hierarchies).
    A sample data point from the SPHERE Biology domain:
> **Text**: "The SRY gene is responsible for encoding the Testis-determining factor protein, which contains the High-mobility group (HMG) box DNA-binding domain."
> **Entities**: ["SRY gene", "Testis-determining factor protein", "High-mobility group (HMG) box"]
> **Relations**: ("Testis-determining factor protein", "part-of", "SRY gene"), ("High-mobility group (HMG) box", "part-of", "Testis-determining factor protein")
These datasets are chosen because they cover diverse scientific domains, include both flat and hierarchical relations, and are standard widely used benchmarks for scientific information extraction, enabling direct comparison to prior SOTA work. The SPHERE benchmark addresses critical data scarcity for large-scale hierarchical RE research.
## 5.2. Evaluation Metrics
Two standard strict metrics are used for evaluation:
### 5.2.1. Micro F1 Score for NER
1.  **Conceptual Definition**: Measures the overall accuracy of NER, balancing precision (fraction of predicted entities that are correct) and recall (fraction of ground-truth entities that are correctly predicted), weighted by the total number of instances across all entity classes (not averaged per class).
2.  **Mathematical Formula**:
    First compute aggregate true positives (TP, correctly predicted entities), false positives (FP, predicted entities that are not ground truth), and false negatives (FN, ground truth entities not predicted) across all entity classes:
    \$
    \mathrm{Micro\ Precision} = \frac{TP}{TP + FP}, \quad \mathrm{Micro\ Recall} = \frac{TP}{TP + FN}
    \$
    Micro F1 is then computed as the harmonic mean of micro precision and micro recall:
    \$
    \mathrm{Micro\ F1} = 2 \times \frac{\mathrm{Micro\ Precision} \times \mathrm{Micro\ Recall}}{\mathrm{Micro\ Precision} + \mathrm{Micro\ Recall}}
    \$
3.  **Symbol Explanation**:
    - `TP`: Count of entities where the predicted span and type exactly match the ground truth.
    - `FP`: Count of predicted entities that do not match any ground truth entity (either wrong span, wrong type, or both).
    - `FN`: Count of ground truth entities that are not predicted by the model.
### 5.2.2. Strict Rel+ F1 Score for RE
1.  **Conceptual Definition**: A very strict metric for end-to-end RE that requires the model to correctly predict three components for a triplet to count as correct: 1) the span and type of the head entity, 2) the span and type of the tail entity, 3) the relation type between the two entities. Any error in any of these three components counts as a wrong prediction, making this a rigorous measure of end-to-end KG construction accuracy.
2.  **Mathematical Formula**:
    Uses the same F1 calculation as above, but with strict definition of TP/FP/FN:
    \$
    \mathrm{Rel+\ F1} = 2 \times \frac{P_{rel+} \times R_{rel+}}{P_{rel+} + R_{rel+}}
    \$
    Where:
    \$
    P_{rel+} = \frac{TP_{rel+}}{TP_{rel+} + FP_{rel+}}, \quad R_{rel+} = \frac{TP_{rel+}}{TP_{rel+} + FN_{rel+}}
    \$
3.  **Symbol Explanation**:
    - $TP_{rel+}$: Count of predicted triplets where both entities and the relation exactly match the ground truth.
    - $FP_{rel+}$: Count of predicted triplets where either entity is incorrect, or the relation is incorrect.
    - $FN_{rel+}$: Count of ground truth triplets that are not fully correctly predicted by the model.
## 5.3. Baselines
The paper compares against three categories of representative SOTA baselines:
### 5.3.1. NER Baselines
- **Supervised SOTA**: SciBERT, PL-Marker, HGERE (standard supervised scientific NER models).
- **Specialized Zero-Shot NER**: UniversalNER-7b (current SOTA zero-shot NER model).
- **Zero-Shot General-Purpose LLMs**: Llama-3.3-70b, Qwen3-32b, Llama-3.1-8b-instant (large general-purpose LLMs tested in zero-shot setting).
### 5.3.2. RE Baselines
- **Supervised SOTA**: PL-Marker, HGERE, PURE (standard end-to-end RE models).
- **Zero-Shot LLMs**: GPT-3.5 Turbo, GPT-oss-120b, Llama-3.3-70b, Qwen3-32b, Llama-3.1-8b-instant (large LLMs tested in zero-shot setting).
- **Standard GNNs**: GCN, GAT (standard GNN architectures for graph-based RE).
- **Geometric Hierarchical Baselines**: HGCN (Hyperbolic GCN), Order-Embeddings (non-Euclidean hierarchical embedding methods).
- **Few-Shot LLM**: Llama-3-8B with 3-shot Chain-of-Thought (CoT) prompting, to test LLM reasoning capability with limited demonstration examples.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### 6.1.1. NER Results (Z-NERD)
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th rowspan="2">SciERC</th>
<th rowspan="2">SciER</th>
<th rowspan="2">BioRED</th>
<th rowspan="2">SemEval</th>
<th colspan="2">CS</th>
<th colspan="2">Physics</th>
<th colspan="2">Bio</th>
<th colspan="2">MS</th>
</tr>
<tr>
<th>Sup</th>
<th>ZS</th>
<th>Sup</th>
<th>ZS</th>
<th>Sup</th>
<th>ZS</th>
<th>Sup</th>
<th>ZS</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="13">Supervised Baselines</td>
</tr>
<tr>
<td>SciBERT Ye et al. (2022)</td>
<td>67.52</td>
<td>70.71</td>
<td>89.15</td>
<td>49.14</td>
<td>68.19</td>
<td>57.02</td>
<td>72.90</td>
<td>61.22</td>
<td>75.83</td>
<td>68.45</td>
<td>67.29</td>
<td>57.14</td>
</tr>
<tr>
<td>PL-Marker Yan et al. (2023)</td>
<td>70.32</td>
<td>74.04</td>
<td>86.41</td>
<td>47.69</td>
<td>68.64</td>
<td>56.39</td>
<td>72.83</td>
<td>60.51</td>
<td>-</td>
<td>66.17</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>HGERE Yan et al. (2023)</td>
<td>75.92</td>
<td>81.19</td>
<td>89.43</td>
<td>48.25</td>
<td>69.82</td>
<td>-</td>
<td>72.46</td>
<td>-</td>
<td>75.78</td>
<td>-</td>
<td>66.72</td>
<td>57.92</td>
</tr>
<tr>
<td>UniversalNER-7b Zhou et al. (2024)</td>
<td>66.09</td>
<td>73.13</td>
<td>88.46</td>
<td>47.60</td>
<td>-</td>
<td>58.95</td>
<td>-</td>
<td>60.67</td>
<td>76.42</td>
<td>68.51</td>
<td>67.24</td>
<td>58.03</td>
</tr>
<tr>
<td colspan="13">Zero-Shot LLM Baselines</td>
</tr>
<tr>
<td>Llama-3.3-70b Touvron et al. (2023)</td>
<td>46.20</td>
<td>49.57</td>
<td>54.82</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>OOM</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Qwen3-32b Qwen et al. (2025)</td>
<td>41.63</td>
<td>46.52</td>
<td>31.71</td>
<td>30.16</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>OOM</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>Llama-3.1-8b-instant Touvron et al. (2023)</td>
<td>31.21</td>
<td>33.96</td>
<td>33.58</td>
<td>26.48</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>OOM</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td colspan="13">Proposed Approach (Z-NERD)</td>
</tr>
<tr>
<td>Z-NERD w/ TCQK w/o OSD</td>
<td>73.43</td>
<td>75.12</td>
<td>84.43</td>
<td>47.85</td>
<td>68.47</td>
<td>59.35</td>
<td>74.92</td>
<td>61.74</td>
<td>73.92</td>
<td>68.30</td>
<td>69.48</td>
<td>57.73</td>
</tr>
<tr>
<td>Z-NERD w/ OSD w/o TCQK</td>
<td>74.39</td>
<td>80.27</td>
<td>90.12</td>
<td>50.98</td>
<td>76.93</td>
<td>62.04</td>
<td>76.68</td>
<td>65.17</td>
<td>82.40</td>
<td>73.29</td>
<td>78.24</td>
<td>63.45</td>
</tr>
<tr>
<td>Z-NERD (full)</td>
<td>78.84</td>
<td>82.71</td>
<td>91.05</td>
<td>52.26</td>
<td>80.47</td>
<td>69.52</td>
<td>82.39</td>
<td>73.19</td>
<td>84.35</td>
<td>74.21</td>
<td>83.96</td>
<td>72.28</td>
</tr>
</tbody>
</table>

Key observations:
- Full Z-NERD outperforms all supervised baselines by an average of 8.08% micro F1 across all benchmarks, and outperforms zero-shot baselines by an average of 10.76% on zero-shot SPHERE domains.
- Large zero-shot LLMs perform very poorly, failing to accurately identify multi-word entity boundaries, and many run out of memory (OOM) even on moderate-sized datasets.
- Z-NERD is only ~300M parameters, far smaller than 7B/32B/70B LLMs, making it drastically more efficient for large-scale deployment.
### 6.1.2. RE Results (HGNet)
#### Standard Benchmark Results
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th colspan="3">SciERC</th>
<th colspan="3">SciER</th>
<th colspan="3">BioRED</th>
</tr>
<tr>
<th>Hier.</th>
<th>Peer</th>
<th>Overall</th>
<th>Hier.</th>
<th>Peer</th>
<th>Overall</th>
<th>Hier.</th>
<th>Peer</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="10">Supervised Models</td>
</tr>
<tr>
<td>PL-Marker Ye et al. (2022)</td>
<td>35.60</td>
<td>44.97</td>
<td>41.63</td>
<td>40.25</td>
<td>61.84</td>
<td>56.78</td>
<td>29.87</td>
<td>43.40</td>
<td>37.19</td>
</tr>
<tr>
<td>HGERE Yan et al. (2023)</td>
<td>37.72</td>
<td>47.35</td>
<td>43.86</td>
<td>43.79</td>
<td>64.35</td>
<td>58.47</td>
<td>32.39</td>
<td>45.73</td>
<td>38.63</td>
</tr>
<tr>
<td>PURE Zhong & Chen (2021)</td>
<td>34.39</td>
<td>38.46</td>
<td>36.78</td>
<td>38.53</td>
<td>56.21</td>
<td>49.35</td>
<td>29.41</td>
<td>41.35</td>
<td>34.92</td>
</tr>
<tr>
<td colspan="10">Zero-Shot LLM Models</td>
</tr>
<tr>
<td>GPT-3.5 Turbo Ye et al. (2023)</td>
<td>14.97</td>
<td>15.02</td>
<td>14.98</td>
<td>8.35</td>
<td>8.91</td>
<td>8.58</td>
<td>6.36</td>
<td>17.13</td>
<td>16.74</td>
</tr>
<tr>
<td>openai/gpt-oss-120b Ye et al. (2023)</td>
<td>19.68</td>
<td>21.27</td>
<td>20.45</td>
<td>27.93</td>
<td>27.52</td>
<td>27.64</td>
<td>7.15</td>
<td>24.16</td>
<td>23.88</td>
</tr>
<tr>
<td>llama-3.3-70b-versatile Touvron et al. (2023)</td>
<td>22.15</td>
<td>22.53</td>
<td>22.39</td>
<td>23.97</td>
<td>25.06</td>
<td>24.59</td>
<td>7.29</td>
<td>25.38</td>
<td>24.12</td>
</tr>
<tr>
<td>qwen/qwen3-32b Qwen et al. (2025)</td>
<td>16.57</td>
<td>19.33</td>
<td>18.20</td>
<td>24.02</td>
<td>24.45</td>
<td>24.28</td>
<td>6.71</td>
<td>21.38</td>
<td>21.09</td>
</tr>
<tr>
<td>llama-3.1-8b-instant Touvron et al. (2023)</td>
<td>13.30</td>
<td>14.27</td>
<td>13.92</td>
<td>17.15</td>
<td>17.69</td>
<td>17.43</td>
<td>5.48</td>
<td>14.46</td>
<td>14.24</td>
</tr>
<tr>
<td colspan="10">Supervised GNN-based Models</td>
</tr>
<tr>
<td>GCN</td>
<td>40.13</td>
<td>48.78</td>
<td>45.62</td>
<td>47.37</td>
<td>63.89</td>
<td>57.35</td>
<td>31.93</td>
<td>45.92</td>
<td>38.96</td>
</tr>
<tr>
<td>GAT</td>
<td>40.37</td>
<td>49.11</td>
<td>46.21</td>
<td>47.35</td>
<td>64.29</td>
<td>57.64</td>
<td>32.40</td>
<td>46.19</td>
<td>39.25</td>
</tr>
<tr>
<td colspan="10">Proposed Approaches</td>
</tr>
<tr>
<td>HGNet w/o DHL</td>
<td>42.70</td>
<td>52.14</td>
<td>47.33</td>
<td>54.75</td>
<td>61.21</td>
<td>58.67</td>
<td>33.09</td>
<td>43.28</td>
<td>41.19</td>
</tr>
<tr>
<td>HGNet w/o CAF Loss</td>
<td>50.96</td>
<td>55.41</td>
<td>53.19</td>
<td>62.36</td>
<td>67.02</td>
<td>65.38</td>
<td>33.85</td>
<td>50.64</td>
<td>47.03</td>
</tr>
<tr>
<td>HGNet (full)</td>
<td>58.52</td>
<td>55.37</td>
<td>51.68</td>
<td>59.10</td>
<td>65.95</td>
<td>62.79</td>
<td>34.31</td>
<td>49.42</td>
<td>45.05</td>
</tr>
</tbody>
</table>

Key observations:
- Full HGNet outperforms all supervised and GNN baselines by an average of 5.99% Rel+ F1 across standard benchmarks.
- Zero-shot LLMs perform very poorly, with even large 70B models achieving less than 25% Rel+ F1, far below HGNet's performance.
- HGNet has especially large gains on hierarchical relations, which are the most challenging for prior models.
#### SPHERE Supervised Results
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th colspan="3">Comp. Sci.</th>
<th colspan="3">Physics</th>
<th colspan="3">Biology</th>
<th colspan="3">Mat. Sci.</th>
</tr>
<tr>
<th>Hier.</th>
<th>Peer</th>
<th>All</th>
<th>Hier.</th>
<th>Peer</th>
<th>All</th>
<th>Hier.</th>
<th>Peer</th>
<th>All</th>
<th>Hier.</th>
<th>Peer</th>
<th>All</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="13">Supervised Models</td>
</tr>
<tr>
<td>PL-Marker Ye et al. (2022)</td>
<td>51.98</td>
<td>57.04</td>
<td>55.29</td>
<td>50.22</td>
<td>56.48</td>
<td>53.51</td>
<td>52.35</td>
<td>53.76</td>
<td>53.03</td>
<td>52.96</td>
<td>53.27</td>
<td>53.12</td>
</tr>
<tr>
<td>HGERE Yan et al. (2023)</td>
<td>54.20</td>
<td>59.86</td>
<td>57.93</td>
<td>53.17</td>
<td>58.90</td>
<td>56.28</td>
<td>54.52</td>
<td>56.47</td>
<td>55.21</td>
<td>55.84</td>
<td>55.86</td>
<td>55.43</td>
</tr>
<tr>
<td colspan="13">Proposed Approaches</td>
</tr>
<tr>
<td>HGNet (ours)</td>
<td>77.40</td>
<td>81.36</td>
<td>79.51</td>
<td>76.93</td>
<td>83.47</td>
<td>80.60</td>
<td>82.53</td>
<td>84.29</td>
<td>83.74</td>
<td>81.91</td>
<td>85.64</td>
<td>83.65</td>
</tr>
<tr>
<td>w/o DHL</td>
<td>73.62</td>
<td>74.83</td>
<td>74.17</td>
<td>74.01</td>
<td>75.30</td>
<td>74.66</td>
<td>79.15</td>
<td>78.64</td>
<td>78.90</td>
<td>77.43</td>
<td>76.92</td>
<td>77.28</td>
</tr>
<tr>
<td>w/o CAF</td>
<td>67.14</td>
<td>65.89</td>
<td>66.50</td>
<td>64.51</td>
<td>66.24</td>
<td>65.96</td>
<td>75.17</td>
<td>73.29</td>
<td>74.13</td>
<td>75.95</td>
<td>77.38</td>
<td>76.32</td>
</tr>
</tbody>
</table>

Key observations:
- HGNet outperforms SOTA baseline HGERE by ~22-28% Rel+ F1 across all SPHERE domains, a massive improvement on the large hierarchical benchmark.
- Removing either DHL or CAF Loss leads to large performance drops, especially removing CAF Loss, showing both structural regularizers are critical for performance on hierarchical data.
#### Zero-Shot RE Results
The following are the zero-shot results from Table 4 of the original paper (models trained on Physics + Biology domains, tested on unseen Computer Science + Material Science domains):

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th colspan="3">Comp. Sci.</th>
<th colspan="3">Mat. Sci.</th>
</tr>
<tr>
<th>Hier.</th>
<th>Peer</th>
<th>All</th>
<th>Hier.</th>
<th>Peer</th>
<th>All</th>
</tr>
</thead>
<tbody>
<tr>
<td>PL-Marker Ye et al. (2022)</td>
<td>28.72</td>
<td>28.41</td>
<td>28.56</td>
<td>33.10</td>
<td>34.22</td>
<td>33.85</td>
</tr>
<tr>
<td>HGERE Yan et al. (2023)</td>
<td>29.93</td>
<td>29.63</td>
<td>29.81</td>
<td>36.27</td>
<td>39.41</td>
<td>37.97</td>
</tr>
<tr>
<td>HGNet (ours)</td>
<td>59.36</td>
<td>64.07</td>
<td>62.60</td>
<td>69.92</td>
<td>71.33</td>
<td>70.62</td>
</tr>
</tbody>
</table>

Key observation: HGNet achieves ~32% average improvement over SOTA baseline HGERE in zero-shot settings, with a total 26.2% average gain across zero-shot test sets, showing exceptional cross-domain generalization capability.
## 6.2. Ablation Studies & Analysis
### 6.2.1. Z-NERD Ablations
From Table 1:
- Removing Multi-Scale TCQK attention leads to an average ~4% drop in F1 across all datasets, with even larger drops for domains with long entities (e.g., Material Science, where entities are often long technical material names). This confirms that TCQK is critical for accurate multi-word entity recognition.
- Removing OSD leads to an average ~2% drop in supervised F1, but a ~11% drop in zero-shot F1. This confirms that OSD provides the domain-agnostic semantic turn signals that enable strong zero-shot generalization.
  The following figure (Figure 2 from the original paper) provides visual evidence for OSD, showing that the average orthogonal semantic velocity norm is significantly higher at entity boundaries (start/end of entities) than for non-entity tokens:

  ![Figure 2: Average Orthogonal Semantic Velocity for tokens at entity boundaries ('Start''End') vs. 'Non-entity' tokens for SPHERE-CS (left) and SciER (right). The clear separation provides visual evidence for Hypothesis 3.1.](images/2.jpg)
  *该图像是图表，展示了SPHERE-CS（左）和SciER（右）中实体边界处的平均正交语义速度范数（'Start'、'End'）与非实体的对比。数据清晰分离，为假设3.1提供了视觉证据。*

### 6.2.2. HGNet Ablations
From Table 2 and Table 3:
- Removing Differentiable Hierarchy Loss (DHL) leads to ~4-6% drop in overall Rel+ F1, with an especially large drop in hierarchical relation F1. This confirms that enforcing DAG structure (no cycles, no shortcuts) is critical for accurate hierarchical RE.
- Removing CAF Loss leads to ~7-13% drop in overall Rel+ F1, with an even larger drop in zero-shot settings. This confirms that the geometric abstraction axis is critical for hierarchical reasoning and cross-domain generalization.
  The following figure (Figure 3 from the original paper) shows the learned abstraction score distributions across domains, aligning with the expected exponential decay pattern (more concrete entities at low scores, fewer abstract entities at high scores):

  ![Figure 3: Distribution of abstraction scores across different scientific domains, showing distinct patterns for each field.](images/3.jpg)
  *该图像是图表，展示了不同科学领域的抽象分数分布。包括材料科学、物理学、生物学和计算机科学的数据集，每个图表展示了对应领域中不同抽象分数下的实体数量。*

### 6.2.3. Additional Ablations
- **Geometric Baseline Comparison**: HGNet outperforms Hyperbolic GCN (HGCN) by ~7-13% Rel+ F1, and Order-Embeddings by ~9-11% Rel+ F1, showing that the Euclidean CAF approach is more effective and stable than non-Euclidean methods, especially for peer relations that do not fit strict tree structures.
- **Few-Shot LLM Comparison**: Even 3-shot Chain-of-Thought prompting for Llama-3-8B only achieves ~19% Rel+ F1 on SciERC, far below HGNet's 53.19% Rel+ F1, showing that even with few-shot demonstration examples, LLMs cannot match the performance of the specialized lightweight model.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper presents a novel two-stage end-to-end framework for automated scientific KG construction that addresses all four core long-standing challenges in the field:
1.  Z-NERD with Orthogonal Semantic Decomposition and Multi-Scale TCQK attention enables accurate recognition of long multi-word entities and strong zero-shot cross-domain generalization.
2.  HGNet with three-channel hierarchical message passing, Differentiable Hierarchy Loss, and Continuum Abstraction Field Loss enables accurate relation extraction that explicitly models hierarchical structure and enforces global logical and geometric consistency.
3.  The SPHERE benchmark provides the first large-scale multi-domain dataset for hierarchical RE, addressing critical data scarcity in the field.
    The framework establishes new state-of-the-art performance across all tested benchmarks, with exceptional zero-shot performance, proving that explicit structural inductive biases for hierarchy and consistency significantly outperform both standard specialized models and large general-purpose LLMs for scientific KG construction, while being far more computationally efficient (~300M parameters vs 7B+ parameters for LLMs).
## 7.2. Limitations & Future Work
### Limitations Identified by Authors
1.  The current framework only processes text input, and does not incorporate multimodal information from figures, tables, and equations in scientific papers, which contain a large volume of critical scientific knowledge.
2.  The SPHERE benchmark currently covers only 4 scientific domains, and can be expanded to more fields to support broader generalization.
3.  The model struggles with rare cases of circular definitions in scientific text (e.g., A defines B, B defines A), where the acyclicity loss forces the model to arbitrarily break the loop, potentially dropping a valid semantic link.
### Suggested Future Work
1.  Extend the framework to incorporate multimodal information from figures, tables, and equations in scientific papers.
2.  Apply the framework to build dynamic, continuously updated KGs that track the real-time evolution of scientific fields as new papers are published.
3.  Add syntactic filtering via dependency parsing as a preprocessing step to prune unlikely entity pairs, improving relation extraction precision.
4.  Leverage the constructed KGs for downstream scientific reasoning tasks, such as automated hypothesis generation and research gap detection.
## 7.3. Personal Insights & Critique
### Key Inspirations
1.  The paper's approach of explicitly encoding structural inductive biases (semantic turns, n-gram patterns, hierarchical message passing, geometric abstraction axis) instead of relying on emergent capabilities of overparameterized large LLMs is a highly promising direction for specialized NLP tasks. It achieves better performance at a fraction of the computational cost, which is critical for scalable deployment of scientific KG systems.
2.  The CAF Loss is a particularly innovative contribution: it offers a simple, interpretable alternative to hyperbolic embeddings for hierarchical data, while supporting both hierarchical and peer relations, which hyperbolic methods struggle with. This paradigm can be applied to many other hierarchical learning tasks beyond KG construction, e.g., taxonomy learning, document hierarchy modeling, and organizational structure mapping.
3.  The SPHERE benchmark's constrained LLM generate-and-annotate methodology is a clever solution to data scarcity for specialized tasks, avoiding the prohibitive cost of large-scale human annotation. This methodology can be adapted to create high-quality labeled datasets for many other low-resource NLP tasks.
### Potential Improvements
1.  The current two-stage pipeline uses separate SciBERT encoders for Z-NERD and HGNet (per Table 11 parameter breakdown), leading to ~293M total parameters. A fully shared encoder could reduce parameter count further, improving efficiency without significant performance loss.
2.  The CAF Loss uses a single universal abstraction axis, which works well for scientific knowledge, but may not be sufficient for domains with multiple independent types of hierarchy. Future work could explore multiple learnable abstraction axes for more complex hierarchical structures.
3.  The current model processes documents independently, so cross-document entity linking and coreference resolution are not handled. Adding cross-document entity resolution would enable construction of larger, global KGs across entire scientific corpora, not just per-document KGs.
4.  The model's performance on circular definitions can be improved by adding special case handling for mutually dependent concepts, instead of forcing strict acyclicity in all cases, as some scientific fields do have valid mutually dependent concepts.
    Overall, this paper represents a significant step forward for automated scientific KG construction, with practical, scalable solutions that can be deployed immediately to help researchers navigate the exponentially growing body of scientific literature.