# 1. Bibliographic Information
## 1.1. Title
The title of the paper is **Embed-RL: Reinforcement Learning for Reasoning-Driven Multimodal Embeddings**. Its central topic is the development of a reinforcement learning (RL) powered universal multimodal embedding (UME) framework that optimizes retrieval-aligned multimodal chain-of-thought (CoT) reasoning to improve cross-modal retrieval performance.
## 1.2. Authors & Affiliations
The authors and their affiliations are:
1.  Haonan Jiang, Yuji Wang, Yansong Tang: Tsinghua Shenzhen International Graduate School, Tsinghua University
2.  Yongjie Zhu, Xin Lu, Wenyu Qin, Meng Wang, Pengfei Wan: Kling Team, Kuaishou Technology
    The research is a collaborative effort between academic (Tsinghua) and industrial (Kuaishou, a leading short-video platform) teams, combining fundamental research in multimodal learning with practical deployment considerations for real-world retrieval systems.
## 1.3. Publication Venue & Status
As of the current date, the paper is published as a preprint on arXiv, with no formal peer-reviewed conference or journal publication listed. arXiv is a widely used preprint server for computer science research, allowing rapid dissemination of cutting-edge work prior to formal peer review.
## 1.4. Publication Year
The paper was published on arXiv on 2026-02-14 (UTC).
## 1.5. Abstract
The paper addresses two key limitations of existing generative multimodal embedding methods: (1) generated CoT reasoning chains are text-only, not aligned to retrieval target requirements, and (2) joint optimization of generative reasoning and embedding objectives causes gradient conflicts that degrade performance. To solve these issues, the authors propose Embed-RL, a decoupled reasoning-driven UME framework that uses Embedder-Guided Reinforcement Learning (EG-RL) to optimize a Reasoner module to generate evidential Traceability CoT (T-CoT) chains that include multimodal cues (text keywords, image bounding boxes, video keyframes) aligned to retrieval tasks. The paper's three core contributions are: (1) the EG-RL framework that uses a pre-trained frozen Embedder as a reward model to supervise Reasoner training, eliminating gradient conflicts; (2) the T-CoT structured reasoning format that integrates retrieval-relevant multimodal cues; (3) state-of-the-art (SOTA) performance on the MMEB-V2 and UVRB benchmarks with limited computational resources, outperforming prior leading embedding models across image, video, and visual document retrieval tasks.
## 1.6. Source Links
- Original arXiv preprint link: https://arxiv.org/abs/2602.13823
- PDF link: https://arxiv.org/pdf/2602.13823v3
- Publication status: Preprint, not yet formally peer-reviewed or published in a conference/journal.

  ---

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Universal Multimodal Embeddings (UME) are the core supporting technology for cross-modal tasks including image-text retrieval, video moment localization, and visual document understanding, which map data from different modalities (text, image, video) into a shared semantic space where similarity between embeddings corresponds to semantic relevance. Recent work has shown that integrating generative CoT reasoning into UME models improves representation quality, but existing methods suffer from two critical limitations:
1.  **CoT Retrieval Misalignment**: Generated CoT chains are limited to textual analysis of queries, do not include multimodal cues, and are not optimized to improve target retrieval performance, often introducing noise or hallucinations.
2.  **Gradient Conflict**: Methods that jointly optimize generative CoT objectives and contrastive embedding objectives face conflicting gradient signals that lead to suboptimal embedding performance.
### Importance of the Problem
As multimodal content (images, videos, visual documents) becomes the dominant form of data on the internet and consumer platforms, high-performance cross-modal retrieval systems are critical for search engines, recommendation systems, and retrieval-augmented generation (RAG) applications. The limitations of existing embedding methods directly restrict the accuracy and generalization of these downstream systems, especially for fine-grained and complex cross-modal matching scenarios.
### Innovative Entry Point
The paper adopts a decoupled Reasoner-Embedder paradigm to eliminate gradient conflicts, and uses RL to optimize the Reasoner to generate CoT chains that directly improve the Embedder's retrieval performance. The key insight is that a pre-trained Embedder can act as a stable reward model to provide explicit supervision for Reasoner training, ensuring generated CoT chains are strictly aligned to embedding and retrieval objectives rather than generic reasoning quality.
## 2.2. Main Contributions & Findings
### Primary Contributions
The paper's three core contributions are:
1.  **Embedder-Guided Reinforcement Learning (EG-RL) Framework**: A decoupled RL framework where a frozen pre-trained Embedder provides reward signals to optimize the Reasoner's CoT generation, eliminating gradient conflicts between generative and embedding objectives, and ensuring CoT outputs are aligned to retrieval tasks.
2.  **Evidential Traceability CoT (T-CoT)**: A structured multimodal reasoning format that explicitly integrates text keywords, 2D image bounding boxes, and video keyframe indices to focus the Embedder on retrieval-relevant multimodal cues, filtering redundant information and improving cross-modal alignment.
3.  **Efficient SOTA Performance**: With limited computational resources (batch sizes of 256-512, 1-2 training epochs), the Embed-RL framework outperforms all prior leading embedding models on both the MMEB-V2 (78 multimodal tasks) and UVRB (16 video retrieval tasks) benchmarks, with particularly large improvements on fine-grained grounding, video retrieval, and out-of-domain visual document tasks.
### Key Findings
The paper's core experimental findings are:
- Decoupling Reasoner and Embedder training and using RL for Reasoner optimization eliminates the gradient conflict problem that plagued prior generative embedding methods, leading to consistent performance improvements across all modality types.
- Adding explicit multimodal cues (bounding boxes, keyframes) to CoT chains significantly improves fine-grained cross-modal matching performance, especially for video and visual grounding tasks.
- The dual reward mechanism (process reward for T-CoT alignment, outcome reward for retrieval performance) ensures the Reasoner generates both logically consistent and retrieval-effective T-CoT chains.
- The framework has strong out-of-domain generalization ability, with over 20 point improvements on out-of-domain visual document retrieval tasks compared to prior baselines.

  ---

# 3. Prerequisite Knowledge & Related Work
This section explains all foundational concepts and prior work required to understand the paper, with beginner-friendly definitions and context.
## 3.1. Foundational Concepts
### 3.1.1. Universal Multimodal Embedding (UME)
UME is a class of models that map data from multiple modalities (text, image, video, visual documents) into a single shared high-dimensional vector space, where the cosine similarity between two embeddings corresponds to their semantic relevance. UME models are the core of cross-modal retrieval systems, allowing users to search for images using text queries, search for videos using image queries, etc.
### 3.1.2. Chain-of-Thought (CoT) Reasoning
CoT is a technique that makes models generate intermediate reasoning steps before producing a final output, rather than directly generating the output. For complex reasoning tasks (including retrieval, question answering), CoT improves performance by allowing the model to decompose complex tasks into smaller steps, explicitly capture relevant cues, and reduce hallucinations. Traditional CoT methods for multimodal tasks generate only textual reasoning steps, without explicitly referencing multimodal cues.
### 3.1.3. Contrastive Learning & InfoNCE Loss
Contrastive learning is the dominant training paradigm for embedding models, which trains the model to maximize the similarity between embeddings of semantically related "positive" sample pairs, and minimize the similarity between embeddings of unrelated "negative" sample pairs.
The most common loss function for contrastive learning is the InfoNCE loss, which the paper uses to train the Embedder module. Its standard formulation (exactly as presented in the paper) is:
$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{ \exp \Bigl( \cos( h_{q_i}, h_{t_i^+} ) / \tau \Bigr) }{ \exp \Bigl( \cos( h_{q_i}, h_{t_i^+} ) / \tau \Bigr) + \sum_{t^- \in \mathcal{T}^-} \exp \bigl( \cos( h_{q_i}, h_{t^-} ) / \tau \bigr) }
$$
Where:
- $N$ = number of samples in the batch
- $q_i$ = $i$-th query sample
- $t_i^+$ = positive target sample semantically related to $q_i$
- $\mathcal{T}^-$ = set of negative target samples unrelated to $q_i$
- $h_{q_i}, h_{t}$ = embeddings of the query and target, extracted from the final hidden state of a special <emb> token in the Embedder
- $\cos(\cdot, \cdot)$ = cosine similarity function between two normalized vectors, ranging from -1 (dissimilar) to 1 (identical)
- $\tau$ = temperature hyperparameter that controls the sharpness of the similarity distribution (lower values make the model more sensitive to small similarity differences)
### 3.1.4. Reinforcement Learning (RL) for Language Models
RL is a machine learning paradigm where an agent learns to take actions to maximize a cumulative reward signal, rather than learning from labeled supervised data. For large language models (LLMs) and multimodal large language models (MLLMs), RL is used to fine-tune model outputs to align with specific objectives that are difficult to encode as supervised loss functions (e.g., retrieval performance, human preference).
### 3.1.5. Group Relative Policy Optimization (GRPO)
GRPO is a recently proposed RL algorithm for LLMs that is a variant of Proximal Policy Optimization (PPO) optimized for stable and efficient training of generative models. It reduces training variance by calculating relative advantages across multiple rollouts (output sequences) generated for the same input sample, rather than absolute advantages across the entire batch. The paper uses GRPO to optimize the Reasoner module's CoT generation policy.
## 3.2. Previous Works
### 3.2.1. Universal Multimodal Embedding Methods
The field of UME has evolved through three main generations:
1.  **Dual-Encoder Contrastive Methods (Pre-2024)**: Pioneering models like CLIP, BLIP, and SigLIP use separate encoders for each modality, trained on paired cross-modal data with contrastive loss. Limitation: They have weak ability to process complex interleaved multimodal inputs (e.g., visual documents, long videos) and limited fine-grained reasoning ability.
2.  **MLLM-Powered Discriminative Embedding Methods (2024)**: Methods like VLM2Vec, LLaVE, and GME leverage the strong multimodal understanding ability of MLLMs to extract embeddings directly from the final hidden states of input tokens. Limitation: They do not leverage the generative reasoning ability of MLLMs, leaving significant performance on the table.
3.  **Generative Reasoning + Embedding Methods (Late 2024 - Early 2025)**: Two leading approaches prior to this paper are:
    - **UME-R1**: Unifies discriminative and generative embeddings by having the MLLM generate textual CoT chains before producing embeddings. Limitation: Joint optimization of contrastive loss and next-token prediction loss leads to gradient conflicts that degrade performance.
    - **TTE**: Uses a decoupled Reasoner-Embedder paradigm, where a pre-trained MLLM generates offline CoT chains, and only the Embedder is trained on the CoT-augmented inputs. Limitation: The Reasoner is not trained together with the Embedder, so generated CoT chains are not aligned to embedding tasks, often introducing noise and hallucinations, and CoTs are text-only without multimodal cues.
### 3.2.2. Multimodal Reasoning with RL
Recent work has applied RL to improve multimodal reasoning in MLLMs, with key prior work including:
- **GRIT**: Trains MLLMs to generate reasoning chains interleaved with bounding box coordinates of relevant image regions, using RL with dual rewards to improve grounded reasoning.
- **Ground-R1**: Uses RL to achieve grounded visual reasoning without extra annotations, guiding response generation to avoid hallucinations.
- **DeepEyes**: Uses end-to-end RL to teach MLLMs to "think with images", improving performance on visual reasoning tasks.
  These methods focus on improving reasoning performance for question answering tasks, not on optimizing reasoning to improve embedding quality for retrieval tasks, which is the core novelty of this paper.
## 3.3. Technological Evolution
The technological timeline for multimodal embeddings relevant to this work is:
1.  2021: CLIP is proposed, establishing the dual-encoder contrastive learning paradigm for cross-modal embeddings.
2.  2023-2024: MLLMs (e.g., LLaVA, Qwen-VL) achieve strong multimodal understanding performance, leading to the development of MLLM-powered discriminative embedding models.
3.  Late 2024: First generative reasoning embedding methods (UME-R1, TTE) are proposed, demonstrating the benefits of integrating CoT reasoning into embedding models, but suffering from gradient conflict and CoT alignment issues.
4.  2026: This paper's Embed-RL framework is proposed, solving the gradient conflict and CoT alignment issues via decoupled RL optimization and multimodal T-CoT, achieving new SOTA performance.
## 3.4. Differentiation Analysis
Compared to prior generative embedding methods, the core innovations of Embed-RL are:

| Method | Optimization Paradigm | CoT Type | Gradient Conflict? | CoT Retrieval Alignment? |
|--------|------------------------|----------|---------------------|---------------------------|
| UME-R1 | Joint optimization of Reasoner and Embedder | Text-only CoT | Yes | Partial |
| TTE | Decoupled, Reasoner frozen, Embedder trained | Text-only CoT | No | No |
| Embed-RL | Decoupled, Embedder frozen, Reasoner optimized via RL | Multimodal T-CoT with bounding boxes/keyframes | No | Yes (explicitly supervised via Embedder reward) |

---

# 4. Methodology
This section provides a step-by-step deconstruction of the paper's technical solution, with all formulas presented exactly as in the original paper, and full explanations of all components.
## 4.1. Core Principles
The paper's method is built on three core design principles:
1.  **Decoupling to Eliminate Gradient Conflict**: The Reasoner (generates T-CoT) and Embedder (produces final embeddings) are trained separately: the Embedder is first trained fully with contrastive loss, then frozen during RL training of the Reasoner. This eliminates the gradient conflict that arises when optimizing generative and embedding objectives jointly.
2.  **Embedder as Reward Model**: The frozen pre-trained Embedder acts as a stable, task-aligned reward model for Reasoner training: if a generated T-CoT chain improves the Embedder's retrieval performance for that sample, it receives a high reward; if it degrades performance, it receives a low reward. This ensures T-CoT generation is strictly optimized for embedding performance, not generic reasoning quality.
3.  **Multimodal Evidence Integration**: The T-CoT reasoning format explicitly integrates structured multimodal cues (text keywords, image bounding boxes, video keyframe indices) to direct the Embedder's attention to retrieval-relevant content, filtering redundant information and improving fine-grained cross-modal alignment.

    The overall framework is illustrated in Figure 1 from the original paper:

    ![Fig. 1: Multimodal embedding optimization via Embedder-Guided Reinforcement Learning (EG-RL). (a) Frameworks evolution. (b) Reasoning enhancement with RL-optimized evidential Traceability CoT (T-CoT). (c) Comparison of multi-task performance.](images/1.jpg)
    *该图像是示意图，展示了通过Embedder-Guided Reinforcement Learning (EG-RL)进行多模态嵌入优化的框架演变和性能对比。在图中，左侧比较了不同的多模态嵌入框架，右侧展示了经强化学习优化的证据追溯链（T-CoT）的推理增强效果，以及在多任务性能方面的对比。*

## 4.2. Core Methodology In-Depth
The method follows a two-stage pipeline: first, high-quality training data is constructed and the Embedder is trained via contrastive learning; second, the Reasoner is optimized via EG-RL using the frozen Embedder as a reward model.
### 4.2.1. Data Construction
The paper first constructs a high-quality multimodal dataset for training via a four-step "sampling-annotation-filtering-splitting" pipeline, illustrated in Figure 2a of the original paper:

![Fig. 2: Overview of the proposed data synthesis and EG-RL framework. (a) Data Construction generates T-CoT annotations for query-positive pairs, filters and splits the dataset to enable contrastive and reinforcement learning, laying the groundwork for reasoning-aware embedding. (b) Embedder-Guided Reinforcement Learning finetunes the MLLM with a process-outcome reward function, encouraging T-CoT trajectories that yield more discriminative and beneficial generative embeddings.](images/2.jpg)
*该图像是示意图，展示了数据构建和嵌入者引导的强化学习框架。 (a) 数据构建生成查询-正对的证据可追溯性链（T-CoT）注释，并将数据集过滤和分割以支持对比和强化学习。 (b) 嵌入者引导的强化学习微调多模态大语言模型（MLLM），使用过程-结果奖励函数提高生成嵌入的区分度和有效性。*

#### Step 1: Data Sampling
The initial data pool is built via stratified sampling across three core sources, following the VLM2Vec-V2 data paradigm:
1.  Image-centric tasks from MMEB-train: Covering image classification, visual question answering (VQA), retrieval, and grounding tasks.
2.  Video-language instruction data from LLaVA-Hound: Covering video captioning, VQA, and retrieval tasks.
3.  Visual document retrieval data from ViDoRe and VisRAG: Covering visual document understanding and retrieval tasks.
    A maximum of 50k samples are taken per image dataset, 100k per visual document dataset, and 300k per video dataset, resulting in an initial dataset of 2.22 million samples.
#### Step 2: T-CoT Annotation
For all query-positive sample pairs, the paper annotates structured evidential Traceability CoT (T-CoT) chains, which follow a fixed three-part format:
1.  $<thinking>$: Extracts modality-specific retrieval-relevant cues, output in structured JSON format:
    - Text cues: `text_keywords` list of core semantic keywords
    - Image cues: $bbox_2d$ list of 2D bounding box coordinates `[x1, y1, x2, y2]` for relevant image regions
    - Video cues: `key_frames` list of 1-indexed frame numbers for relevant video keyframes
2.  $<rethink>$: Refines reasoning logic to focus on key retrieval-relevant aspects, verifying that the extracted cues are relevant to the retrieval task.
3.  $<answer>$: Summarizes core retrieval-relevant information about the sample.
    Task-specific prompts are used to guide annotation, ensuring T-CoTs are aligned to different retrieval scenarios (text-to-image, text-to-video, visual document retrieval, etc.). An example of T-CoT for a video retrieval task is shown in Figure 3 of the original paper:

    ![Fig. 3: Example visualization of our reasoning-driven embedding framework on multimodal retrieval tasks. The figure shows the evidential Traceability CoT reasoning process for video and visual document retrieval.](images/3.jpg)
    *该图像是示意图，展示了我们基于推理的嵌入框架在多模态检索任务中的应用。图中展示了针对视频和视觉文档检索的证据追踪思维链（T-CoT）推理过程。*

#### Step 3: CoT-Guided Relevance Filtering
A custom judgment prompt is used to filter out noisy samples where the T-CoT of the query or positive sample is obviously irrelevant or contradictory to the task description. Only samples labeled "No" (relevant, not contradictory) are retained for contrastive learning.
After filtering, 1.83 million samples are retained (80% retention rate). 20% of the filtered-out "hard" samples are reserved for the RL training stage, as they provide valuable exploration signals for the Reasoner.
#### Step 4: Weighted Sampling
Training weights are assigned to different datasets based on task importance and data quality, to ensure balanced training across diverse task types.
### 4.2.2. Embedder Pre-Training
First, the Embedder module (based on Qwen3-VL-2B/4B MLLMs) is fully trained via contrastive learning using the InfoNCE loss (presented in Section 3.1.3) on the filtered T-CoT-augmented dataset. The input to the Embedder is the concatenation of the original multimodal input, the T-CoT chain, and a special $<emb>$ token. The embedding of the $<emb>$ token's final hidden state is extracted as the final output embedding of the sample.
After training, the Embedder is frozen completely and used as a reward model for the subsequent RL training stage.
### 4.2.3. Embedder-Guided Reinforcement Learning (EG-RL) Framework
The EG-RL framework optimizes the Reasoner module (based on Qwen3-VL-8B) to generate high-quality T-CoT chains, using the frozen Embedder to provide reward signals. The workflow is illustrated in Figure 2b of the original paper.
#### Input Construction
The Reasoner takes the original multimodal query as input, outputs a structured T-CoT chain. The T-CoT is concatenated with the original input to form the Embedder's input:
$$
\mathcal{T} = [x_{\text{text}}, x_{\text{img}}, x_{\text{vid}}, \text{T-CoT}(x), \text{<emb>}]
$$
Where $x_{\text{text}}, x_{\text{img}}, x_{\text{vid}}$ are the original text, image, and video components of the input, $\text{T-CoT}(x)$ is the T-CoT chain generated by the Reasoner, and $<emb>$ is the special embedding token. The embedding of the $<emb>$ token is used to calculate the reward signal.
#### Reward Function Design
The paper uses a three-component weighted reward function to align T-CoT generation with embedding quality, combining format compliance, process-level T-CoT alignment, and outcome-level retrieval effectiveness. All reward components are computed symmetrically for both queries and their corresponding positive targets to ensure consistent embedding alignment in both directions.
1.  **Format Reward ($\mathcal{R}_{format}$)**: Ensures generated T-CoT strictly follows the predefined three-part template ($<thinking>$ → $<rethink>$ → $<answer>$) and includes all required multimodal cues (text keywords, bboxes, keyframes) for the input modality.
    $$
    \mathcal{R}_{format} =
    \begin{cases}
    1, & \text{if T-CoT fully complies with format requirements} \\
    0, & \text{otherwise}
    \end{cases}
    $$
    This reward guarantees T-CoT output interpretability and compatibility with the Embedder module's input format.
2.  **Embedder-Guided Outcome Reward ($\mathcal{R}_{outcome}$)**: Measures how much the generated T-CoT improves embedding alignment and retrieval performance, by jointly assessing the top-$k$ retrieval accuracy of the positive sample and the similarity margin between the positive sample and hard negative samples.
    The exact formula from the paper is:
    $$
    \mathcal{R}_{\mathrm{outcome}}(o_i^q) = \mathrm{Acc}_k(e_{q_i}, t_i^+) \cdot \Big( \mathrm{sim}(e_q, e_{t_i^+}) - \mathbb{E}_\tau \big[ \mathrm{sim}(e_{q_i}, e_{t_j^-}) \big] \Big)
    $$
    Where:
    - $o_i^q$ = T-CoT output for query $q_i$
    - $\mathrm{Acc}_k(e_{q_i}, t_i^+)$ = top-$k$ retrieval accuracy: 1 if the positive target $t_i^+$ is among the top $k$ targets ranked by cosine similarity to the query embedding $e_{q_i}$, 0 otherwise. The paper sets $k=8$ for all experiments.
    - $\mathrm{sim}(\cdot, \cdot)$ = cosine similarity between normalized embeddings
    - $\mathbb{E}_\tau \big[ \mathrm{sim}(e_{q_i}, e_{t_j^-}) \big]$ = softmax-weighted average of cosine similarities between the query embedding $e_{q_i}$ and embeddings of in-batch negative targets $t_j^-$, computed as:
      $$
      \mathbb{E}_\tau \big[ \mathrm{sim}(e_{q_i}, e_{t_j^-}) \big] = \frac{ \sum_{j \neq i} \exp \left( \frac{ \mathrm{sim}(e_{q_i}, e_{t_j^-}) }{ \tau } \right) \cdot \mathrm{sim}(e_{q_i}, e_{t_j^-}) }{ \sum_{j \neq i} \exp \left( \frac{ \mathrm{sim}(e_{q_i}, e_{t_j^-}) }{ \tau } \right) }
      $$
      Where $\tau$ = temperature hyperparameter (set to 0.5 in all experiments) that weights hard negatives (samples with high similarity to the query) more heavily.
    The outcome reward is high when the positive sample is ranked high in retrieval results, and there is a large margin between the positive sample's similarity and the average similarity of hard negatives.
3.  **T-CoT Process Reward ($\mathcal{R}_{process}$)**: Measures the alignment between the T-CoT chain of the query and the T-CoT chain of its positive target, using an independent pre-trained VLM discriminator $\mathcal{D}$ to perform listwise comparison.
    The exact formula from the paper is:
    $$
    \mathcal{R}_{\mathrm{process}}(o_i) =
    \begin{cases}
    1, & \text{if } \mathcal{D} \big( q_{\mathrm{cot}}, \{ c_{\mathrm{cot}}^j \}_{j=1}^m \big) \in \mathcal{P} \\
    0, & \text{otherwise}
    \end{cases}
    $$
    Where:
    - $o_i$ = T-CoT generation outcome for the $i$-th sample
    - $q_{\mathrm{cot}}$ = T-CoT chain of the query
    - $\{ c_{\mathrm{cot}}^j \}_{j=1}^m$ = shuffled set of T-CoT chains from $m$ samples, including the positive target's T-CoT and multiple negative T-CoTs
    - $\mathcal{P}$ = index set of ground-truth positive T-CoT chains
    - $\mathcal{D}(\cdot, \cdot)$ = independent pre-trained VLM discriminator that selects the candidate T-CoT most aligned with the query T-CoT
      The process reward is 1 if the discriminator correctly selects the positive T-CoT from the shuffled set, indicating the query and positive T-CoTs are well-aligned; otherwise it is 0.
4.  **Total Reward**: The final reward is a weighted combination of the three components:
    $$
    \mathcal{R}_{\mathrm{total}} = \alpha \mathcal{R}_{\mathrm{format}} + \beta \mathcal{R}_{\mathrm{process}} + \gamma \mathcal{R}_{\mathrm{outcome}}
    $$
    Where $\alpha, \beta, \gamma$ are non-negative weighting coefficients. The paper uses $\alpha=0.05$, $\beta=0.8$, $\gamma=0.2$ in all experiments, prioritizing process alignment of T-CoT chains.
#### Policy Optimization with GRPO
The paper uses the Group Relative Policy Optimization (GRPO) algorithm to optimize the Reasoner's T-CoT generation policy, which stabilizes training by calculating relative advantages across multiple rollouts for the same input.
For each query-target pair, $G=8$ candidate T-CoT sequences are sampled from the current Reasoner policy. The GRPO optimization objective, exactly as presented in the paper, is:
$$
\mathcal{L}_{\mathrm{grpo}} = \mathbb{E}_{\{o_i\} \sim \pi_{\theta_{\mathrm{old}}}} \Bigg[ \frac{1}{G} \sum_{i=1}^G \bigg( \min\big( r_\theta(o_i) A_i, \mathrm{clip}(r_\theta(o_i), 1-\epsilon, 1+\epsilon) A_i \big) - \beta \mathbb{D}_{\mathrm{KL}}(\pi_\theta \parallel \pi_{\mathrm{ref}}) \bigg) \Bigg]
$$
Where:
- $\pi_{\theta_{\mathrm{old}}}$ = Reasoner policy before the update
- $r_\theta(o_i) = \frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\mathrm{old}}}(o_i | q)}$ = importance ratio, measuring how much the current policy $\pi_\theta$ favors the T-CoT sequence $o_i$ compared to the old policy
- $A_i$ = advantage function, calculated as the group-normalized reward for the $i$-th T-CoT sequence: $A_i = \frac{r_i - \mu_r}{\sigma_r}$, where $\mu_r$ and $\sigma_r$ are the mean and standard deviation of rewards across the $G$ rollouts for the same input
- $\epsilon$ = clipping threshold (set to 0.2) that limits the size of policy updates to avoid destabilizing training
- $\mathbb{D}_{\mathrm{KL}}(\pi_\theta \parallel \pi_{\mathrm{ref}})$ = Kullback-Leibler (KL) divergence between the current policy and the reference policy (the Reasoner policy before RL training), which prevents the model from deviating too far from its original general reasoning ability
- $\beta$ = KL divergence weight (set to 0.01) that balances reward maximization and policy stability

  ---

# 5. Experimental Setup
## 5.1. Datasets
The paper evaluates performance on two public benchmarks, and uses its own constructed dataset for training.
### 5.1.1. Training Dataset
The paper's custom constructed training dataset has 2.22 million initial samples, filtered to 1.83 million high-quality samples, covering:
- 1.12 million image-based samples from MMEB-train
- 900k video-based samples from LLaVA-Hound
- 200k visual document samples from ViDoRe and VisRAG
  A 19k sample subset of hard cases is used for RL training of the Reasoner, uniformly sampled across all task types.
### 5.1.2. Evaluation Benchmarks
1.  **MMEB-V2 (Massive Multimodal Embedding Benchmark V2)**: The most comprehensive public multimodal embedding benchmark, consisting of 78 diverse tasks across three core modalities:
    - Image: 36 tasks covering classification, VQA, retrieval, and visual grounding
    - Video: 18 tasks covering classification, VQA, retrieval, and moment retrieval
    - Visual Document: 24 tasks covering in-domain and out-of-domain visual document retrieval
      It is designed to evaluate the universal generalization ability of multimodal embedding models across diverse task types.
2.  **UVRB (Universal Video Retrieval Benchmark)**: A specialized video retrieval benchmark consisting of 16 datasets, designed to evaluate generalization across diverse video retrieval scenarios:
    - Coarse-grained (CG) retrieval: High-level semantic matching of video content
    - Fine-grained (FG) retrieval: Fine-grained matching of spatial details, temporal dynamics, and partially relevant content
    - Long-context (LC) retrieval: Matching long videos and long text queries
## 5.2. Evaluation Metrics
The paper uses three standard retrieval metrics, explained in detail below:
### 5.2.1. Hit@k
Hit@k measures the percentage of test queries where the positive target sample appears in the top $k$ results ranked by embedding similarity to the query. It is the primary metric for image and video retrieval tasks in the paper, with $k=1$ used for all MMEB-V2 image and video tasks.
#### Conceptual Definition
Hit@k quantifies the basic retrieval accuracy of the model, measuring how often the correct target is found in the first $k$ results. A higher Hit@k indicates better retrieval performance.
#### Mathematical Formula
$$
\text{Hit@k} = \frac{1}{N} \sum_{i=1}^N \mathbb{1} \left( \text{rank}(t_i^+, q_i) \leq k \right)
$$
Where:
- $N$ = number of test queries
- $\mathbb{1}(\cdot)$ = indicator function, equals 1 if the condition is true, 0 otherwise
- $\text{rank}(t_i^+, q_i)$ = rank of the positive target $t_i^+$ when all candidates are sorted by cosine similarity to the query $q_i$ (lower rank = more similar)
### 5.2.2. NDCG@k (Normalized Discounted Cumulative Gain @ k)
NDCG@k is a ranking metric that accounts for the position of relevant items in the result list, giving higher weight to relevant items that appear earlier in the list. It is the primary metric for visual document retrieval tasks in the paper, with $k=5$.
#### Conceptual Definition
NDCG@k quantifies the quality of the entire ranked result list, not just whether the positive item is present in the top k. It is normalized to a range of 0 (worst) to 1 (best), making it suitable for comparing performance across datasets with different numbers of relevant items.
#### Mathematical Formula
NDCG@k is calculated as the ratio of the Discounted Cumulative Gain (DCG) of the predicted ranking to the DCG of the ideal optimal ranking:
$$
\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
$$
Where:
$$
\text{DCG@k} = \sum_{j=1}^k \frac{ 2^{rel_j} - 1 }{ \log_2(j+1) }
$$
- $rel_j$ = relevance score of the item at rank $j$ in the predicted list (for retrieval tasks with one positive item, $rel_j=1$ if the item is positive, 0 otherwise)
- $\text{IDCG@k}$ = DCG of the ideal ranking where all relevant items appear at the top of the list
### 5.2.3. mAP (mean Average Precision)
mAP is the mean of the Average Precision (AP) scores across all test queries. AP measures the precision of retrieval results at all positions where a relevant item appears, summarizing the precision-recall curve for a query into a single value. It is the primary metric for UVRB video retrieval tasks.
#### Conceptual Definition
mAP quantifies the overall retrieval performance across all queries, accounting for both precision and recall. It is particularly suitable for retrieval tasks where multiple relevant items may exist for a single query.
#### Mathematical Formula
$$
\text{mAP} = \frac{1}{N} \sum_{i=1}^N \text{AP}(q_i)
$$
Where:
$$
\text{AP}(q_i) = \sum_{j=1}^M P(j) \cdot \mathbb{1}(j \text{ is a relevant item})
$$
- $N$ = number of test queries
- $M$ = number of candidate items for query $q_i$
- `P(j)` = precision of the result list up to rank $j$ (fraction of items in the first $j$ results that are relevant)
## 5.3. Baselines
The paper compares Embed-RL against 12 representative state-of-the-art multimodal embedding models, covering all major paradigm types to ensure fair comparison:
1.  **Visual Document Specialized Models**: ColPali (visual document retrieval model)
2.  **Dual-Encoder Contrastive Models**: InternVideo2 (video embedding model)
3.  **MLLM-Powered Discriminative Embedding Models**: GME, VLM2Vec, VLM2Vec-V2, LamRA, CAFe, LLaVE, Unite
4.  **Generative Reasoning Embedding Models**: TTE, UME-R1
5.  **Video Retrieval Specialized Models**: GVE

    These baselines are representative of the current state-of-the-art in multimodal embedding, covering all relevant architectures, modality specializations, and parameter scales (2B to 7B parameters).

---

# 6. Results & Analysis
## 6.1. Core Results Analysis
### 6.1.1. MMEB-V2 Benchmark Results
The paper's main results on the MMEB-V2 benchmark are presented in Table 1 below (exactly as in the original paper, formatted with HTML to handle merged column headers):

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4">Image</th>
<th colspan="5">Video</th>
<th colspan="4">VisDoc</th>
<th rowspan="2">All</th>
</tr>
<tr>
<th>CLS</th>
<th>QA</th>
<th>RET</th>
<th>GD</th>
<th>Overall</th>
<th>CLS</th>
<th>QA</th>
<th>RET</th>
<th>MRET</th>
<th>Overall</th>
<th>VDRv1</th>
<th>VDRv2</th>
<th>VR</th>
<th>OOD</th>
<th>Overall</th>
</tr>
<tr>
<th># of Datasets</th>
<th>10</th>
<th>10</th>
<th>12</th>
<th>4</th>
<th>36</th>
<th>5</th>
<th>5</th>
<th>5</th>
<th>3</th>
<th>18</th>
<th>10</th>
<th>4</th>
<th>6</th>
<th>4</th>
<th>24</th>
<th>78</th>
</tr>
</thead>
<tbody>
<tr><td colspan="17">Baseline Models</td></tr>
<tr><td>ColPali-V1.3-3B [16]</td><td>40.3</td><td>11.5</td><td>48.1</td><td>40.3</td><td>34.9</td><td>26.7</td><td>37.8</td><td>21.6</td><td>25.5</td><td>28.2</td><td>83.6</td><td>52.0</td><td>81.1</td><td>43.1</td><td>71.0</td><td>44.4</td></tr>
<tr><td>GME-2B [72]</td><td>54.4</td><td>29.9</td><td>66.9</td><td>55.5</td><td>51.9</td><td>34.9</td><td>42.0</td><td>25.6</td><td>32.4</td><td>33.9</td><td>86.1</td><td>54.0</td><td>82.5</td><td>43.1</td><td>72.7</td><td>54.1</td></tr>
<tr><td>GME-7B [72]</td><td>57.7</td><td>34.7</td><td>71.2</td><td>59.3</td><td>56.0</td><td>37.4</td><td>50.4</td><td>28.4</td><td>38.2</td><td>38.6</td><td>89.4</td><td>55.6</td><td>85.0</td><td>44.4</td><td>75.2</td><td>57.8</td></tr>
<tr><td>LamRA-2-7B [39]</td><td>59.2</td><td>26.5</td><td>70.0</td><td>62.7</td><td>54.1</td><td>39.3</td><td>42.6</td><td>24.3</td><td>34.6</td><td>35.2</td><td>22.0</td><td>11.5</td><td>37.4</td><td>21.0</td><td>23.9</td><td>40.4</td></tr>
<tr><td>LamRA-2.5-7B [39]</td><td>51.7</td><td>34.1</td><td>66.9</td><td>56.7</td><td>52.4</td><td>32.9</td><td>42.6</td><td>23.2</td><td>37.6</td><td>33.7</td><td>56.3</td><td>33.3</td><td>51.8</td><td>33.5</td><td>50.2</td><td>47.4</td></tr>
<tr><td>VLM2Vec-2B [26]</td><td>58.7</td><td>49.3</td><td>65.0</td><td>72.9</td><td>59.7</td><td>33.4</td><td>30.5</td><td>39.1</td><td>30.0</td><td>29.0</td><td>49.8</td><td>13.5</td><td>59.1</td><td>38.1</td><td>41.6</td><td>47.0</td></tr>
<tr><td>VLM2Vec-7B [26]</td><td>62.7</td><td>56.9</td><td>69.4</td><td>82.2</td><td>65.5</td><td>39.3</td><td>30.0</td><td>40.6</td><td>33.0</td><td>34.0</td><td>56.9</td><td>9.4</td><td></td><td>38.1</td><td>46.4</td><td>52.3</td></tr>
<tr><td>VLM2Vec-V2-2B [41]</td><td>62.9</td><td>56.3</td><td>69.5</td><td>77.3</td><td>64.9</td><td>34.3</td><td>28.8</td><td>45.9</td><td>33.9</td><td>36.4</td><td>75.5</td><td>44.9</td><td>79.4</td><td>39.4</td><td>65.4</td><td>58.0</td></tr>
<tr><td>VLM2Vec-V2-7B [41]</td><td>65.7</td><td>61.5</td><td>70.0</td><td>85.2</td><td>68.1</td><td>35.8</td><td>58.7</td><td>34.4</td><td>39.5</td><td>42.4</td><td>70.7</td><td>49.6</td><td>79.5</td><td>38.1</td><td>63.9</td><td>60.6</td></tr>
<tr><td>CAFe-7B [63]</td><td>63.6</td><td>61.7</td><td>69.1</td><td>87.6</td><td>67.6</td><td>47.3</td><td>49.1</td><td>34.4</td><td>33.2</td><td>32.1</td><td>77.5</td><td>53.2</td><td>83.2</td><td>41.1</td><td>68.8</td><td>63.1</td></tr>
<tr><td>TTEs-2B [13]</td><td>67.9</td><td>66.6</td><td>70.2</td><td>84.1</td><td>70.1</td><td>44.3</td><td>51.2</td><td>32.9</td><td>39.7</td><td>42.2</td><td>72.4</td><td>46.2</td><td>79.2</td><td>37.2</td><td>63.9</td><td>60.1</td></tr>
<tr><td>UME-R1-2B [30]</td><td>64.8</td><td>62.8</td><td>67.6</td><td>77.2</td><td>66.6</td><td>44.3</td><td>51.2</td><td>32.9</td><td>39.7</td><td>42.2</td><td>72.4</td><td>46.2</td><td>79.2</td><td>37.2</td><td>63.9</td><td>60.1</td></tr>
<tr><td>UME-R1-7B [30]</td><td>67.1</td><td>69.2</td><td>71.9</td><td>84.9</td><td>71.3</td><td>48.6</td><td>60.7</td><td>38.2</td><td>39.3</td><td>47.5</td><td>75.7</td><td>50.5</td><td>83.7</td><td>37.6</td><td>67.1</td><td>64.5</td></tr>
<tr><td colspan="17">Ours</td></tr>
<tr><td>Embed-RL-2B</td><td>62.8</td><td>67.9</td><td>68.6</td><td>90.4</td><td>69.2</td><td>57.0</td><td>55.9</td><td>45.1</td><td>49.4</td><td>52.1</td><td>79.9</td><td>52.0</td><td>84.6</td><td>65.7</td><td>74.1</td><td>66.8</td></tr>
<tr><td>Embed-RL-4B</td><td>63.7</td><td>70.5</td><td>71.3</td><td>91.4</td><td>70.1</td><td>57.6</td><td>58.4</td><td>45.1</td><td>49.5</td><td>53.0</td><td>80.2</td><td>53.4</td><td>84.9</td><td>67.1</td><td>74.7</td><td>68.1</td></tr>
</tbody>
</table>

The results show:
- **Overall Performance**: Embed-RL-4B achieves an overall score of 68.1, outperforming the closest baseline UME-R1-7B by 3.6 points, while using less computational resources for training. Embed-RL-2B (66.8) also outperforms all prior 2B-7B baseline models.
- **Modality-Specific Performance**:
  - Image: Embed-RL-4B achieves SOTA grounding (GD) performance of 91.4, a 3.8 point improvement over the closest baseline CAFe-7B, demonstrating strong fine-grained visual alignment ability.
  - Video: Embed-RL-2B and 4B outperform all baselines by a large margin, with overall video scores of 52.1 and 53.0, 5.5 points higher than the closest baseline UME-R1-7B (47.5). Video retrieval (RET) performance is 45.1, 6.9 points higher than prior baselines.
  - Visual Document: Embed-RL achieves particularly large improvements on out-of-domain (OOD) visual document retrieval, with Embed-RL-4B scoring 67.1, 26 points higher than the closest baseline, demonstrating very strong out-of-domain generalization ability.
### 6.1.2. UVRB Video Retrieval Benchmark Results
The results on the UVRB benchmark are presented in Table 2 below:

| Model | CG | FG | LC |
|---|---|---|---|
| InternVideo2-6B [55] | 50.4 | 41.7 | 42.3 |
| VLM2Vec-V2 [41] | 49.8 | 50.2 | 76.2 |
| GME-7B [72] | 51.8 | 50.7 | 78.8 |
| Unite-7B [18] | 54.1 | 53.9 | 74.6 |
| GVE-3B [20] | 55.2 | 54.1 | 76.4 |
| Embed-RL-2B | 59.1 | 54.6 | 86.9 |
| Embed-RL-4B | 60.7 | 55.6 | 86.1 |

The results show that Embed-RL outperforms all baselines across all three video retrieval scenarios:
- Embed-RL-4B achieves SOTA performance on coarse-grained (CG: 60.7) and fine-grained (FG:55.6) retrieval, outperforming the closest baseline by 5.5 and 1.5 points respectively.
- Embed-RL-2B achieves SOTA performance on long-context (LC:86.9) retrieval, outperforming the closest baseline by 8.1 points.
  These results validate the effectiveness of T-CoT's multimodal cue integration for video retrieval, especially for long and complex videos where keyframe extraction helps the model focus on relevant content.
### 6.1.3. Qualitative T-CoT Improvement Examples
The paper provides examples of T-CoT before and after EG-RL optimization, demonstrating that RL training significantly improves the accuracy and relevance of T-CoT chains, as shown in the example below (Figure 6 from the original paper):

![该图像是一个市场场景，展示了一堆香蕉和其他水果，周围环境显得十分繁忙与生活化。视觉焦点是黄色香蕉的堆放，给人以丰富而新鲜的感觉。](images/6.jpg)
*该图像是一个市场场景，展示了一堆香蕉和其他水果，周围环境显得十分繁忙与生活化。视觉焦点是黄色香蕉的堆放，给人以丰富而新鲜的感觉。*

Before RL, the Reasoner incorrectly identifies apples to the right of the empty crate, leading to a low embedding similarity of 0.3613 to the target. After RL, the Reasoner correctly identifies bananas as the fruit to the right of the crate, with accurate bounding box localization, leading to a higher similarity of 0.4883.
## 6.2. Ablation Study Results
The paper conducts extensive ablation studies to verify the contribution of each component of the framework, using Embed-RL-2B as the base model.
### 6.2.1. Reward Component Ablation
The results of the ablation study on EG-RL reward components are presented in Table 3:

| Model | Image | Video | VisDoc | All |
|---|---|---|---|---|
| Embed-RL-2B | 69.2 | 52.1 | 74.1 | 66.8 |
| w/o EG-RL | 68.0 | 50.1 | 72.7 | 65.3 |
| w/o weighted negative | 68.9 | 51.7 | 73.9 | 66.5 |
| w/o process reward | 68.3 | 51.3 | 73.5 | 66.0 |
| w/o outcome reward | 68.1 | 51.2 | 73.1 | 65.8 |

The results show:
- Removing the entire EG-RL stage reduces overall performance by 1.5 points, confirming that RL optimization of the Reasoner is critical for embedding alignment.
- The outcome reward contributes 1.0 point to overall performance, and the process reward contributes 0.8 points. The process reward has a particularly large impact on video tasks, confirming that step-by-step reasoning alignment is critical for video understanding.
### 6.2.2. T-CoT Component Ablation
The results of the ablation study on T-CoT components are presented in Table 4:

| Model | Image | Video | VisDoc | All |
|---|---|---|---|---|
| Embed-RL-2B | 69.2 | 52.1 | 74.1 | 66.8 |
| w/o reasoning | 67.9 | 50.5 | 73.1 | 65.5 |
| w/o multimodal cues | 68.1 | 51.4 | 73.3 | 65.8 |
| w/ raw input (no T-CoT) | 60.4 | 43.7 | 72.4 | 60.2 |

The results show:
- Removing T-CoT entirely (using only raw input) causes a catastrophic 6.6 point drop in overall performance, with an 8.4 point drop on video tasks. This confirms that high-quality T-CoT is critical for retrieval performance, especially for complex video tasks.
- Removing multimodal cues (using only textual CoT) reduces performance by 1.0 point, confirming that integrating bounding boxes and keyframes improves cross-modal alignment.
### 6.2.3. Discriminative Ability Analysis
The paper measures the model's ability to distinguish between similar candidates by calculating the difference between the cosine similarity of the query to the top-ranked positive sample and the second-ranked similar negative sample ($\Delta s$), before and after EG-RL optimization. The results are shown in Figure 4:

![Fig. 4: Similarity difference `\\varDelta s = \\sin ( \\mathrm { q u e r y } , \\mathrm { t o p 1 } ) - \\sin ( \\mathrm { q u e r y } , \\mathrm { t o p 2 } )` before and after EG-RL. Here, $\\mathrm { s i m } ( \\cdot , \\cdot )$ denotes cosine similarity of normalized embeddings, top1 is the most similar positive candidate and top2 the second-most similar. This metric quantifies the model's discriminative ability over similar candidates on multimodal datasets.](images/4.jpg)
*该图像是一个示意图，展示了在增强学习（RL）之前和之后，模型在图像、文档和视频任务中的相似性差异 `riangle s`。其中，模型在不同数据集上的 discriminative 能力通过该指标进行量化，表明在对比候选时的表现差异。*

The radar chart shows that after RL optimization, the similarity difference $\Delta s$ is consistently larger across all datasets, meaning the model assigns a significantly higher similarity to positive samples than to similar negative samples. This confirms that EG-RL optimization improves the model's discriminative ability for fine-grained matching.
### 6.2.4. Traceable Evidence Count Analysis
The paper analyzes the relationship between the number of traceable evidence pieces (bounding boxes for images/documents, keyframes for videos) and retrieval performance, before and after RL, as shown in Figure 5:

![Fig. 5: Relationship between traceable evidence counts and retrieval metrics across datasets. $\\mathrm { H i t @ 1 }$ is employed for Image and Video; NDCG $\\ @ 5$ is used for VisDoc. Bounding box counts are shown for Image and VisDoc, while keyframe counts for Video.](images/5.jpg)
*该图像是图表，展示了不同数据集中的可追溯证据计数与检索指标之间的关系。图中分别使用 `ext{Hit@1}` 对图像和视频进行评估，NDCG `ext{@} 5` 用于VisDoc。横轴显示不同的数据集，纵轴为计数和指标值。*

The results show:
- For image and visual document tasks: After RL, the Reasoner generates more bounding boxes, capturing more visual evidence to improve reasoning accuracy and recall, leading to higher Hit@1/NDCG@5 scores.
- For video tasks: After RL, the Reasoner focuses on fewer keyframes, filtering redundant frames and concentrating on critical content to improve retrieval performance.

  ---

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper proposes Embed-RL, a reasoning-driven universal multimodal embedding framework that solves two key limitations of prior generative embedding methods: gradient conflict between generative and embedding objectives, and misalignment between generated CoT chains and retrieval tasks. The core innovations are:
1.  The decoupled EG-RL framework, which uses a frozen pre-trained Embedder as a reward model to optimize the Reasoner's T-CoT generation via RL, eliminating gradient conflicts and ensuring CoT chains are aligned to retrieval objectives.
2.  The structured multimodal T-CoT format, which integrates text keywords, image bounding boxes, and video keyframes to direct the Embedder's attention to retrieval-relevant content, improving fine-grained cross-modal alignment.
    Extensive experiments on the MMEB-V2 and UVRB benchmarks show that Embed-RL outperforms all prior state-of-the-art embedding models across image, video, and visual document retrieval tasks, with particularly large improvements on fine-grained grounding, video retrieval, and out-of-domain generalization tasks, while using limited computational resources. The framework provides a practical and efficient solution for developing high-performance reasoning-driven UME models for real-world deployment.
## 7.2. Limitations & Future Work
The authors explicitly note the following limitations of the current work:
1.  **Empirical Reward Weights**: The weight coefficients of the three reward components are set empirically, lacking an adaptive optimization mechanism for diverse multimodal tasks, which may lead to suboptimal performance on specific task types.
2.  **Weak Classification Performance**: The training dataset excludes several classification datasets to avoid false negative issues in contrastive learning, leading to relatively weak performance on image classification subtasks.
3.  **No Hard Negative Mining or Curriculum Learning**: The current framework does not incorporate hard negative sample mining or curriculum learning strategies, which could further improve the model's discriminative ability and training stability.
    The authors suggest the following future research directions:
- Develop adaptive reward weight mechanisms that automatically adjust to different task types to improve generalization across diverse scenarios.
- Design specialized loss functions for classification tasks to avoid false negative issues, allowing the inclusion of classification datasets in training to improve classification performance.
- Incorporate hard negative mining and curriculum learning strategies to further enhance the model's discriminative ability and training efficiency.
## 7.3. Personal Insights & Critique
### Key Inspirations
1.  **Deployment Practicality**: The Embed-RL framework has significant advantages for industrial deployment: T-CoT chains are generated offline once for all items in the retrieval database, and the resulting embeddings can be cached permanently, eliminating the need for online reasoning during inference. This results in negligible latency overhead compared to online generative embedding methods, making it suitable for large-scale industrial retrieval systems (e.g., short-video search, e-commerce product search).
2.  **Cross-Domain Transferability**: The core idea of using a downstream task model as a reward signal to optimize an upstream reasoning module can be transferred to other domains beyond embedding. For example, in RAG systems, a retriever model could be used as a reward signal to optimize a query rewriting module to generate more retrieval-effective queries; in recommendation systems, a ranking model could be used as a reward signal to optimize user interest reasoning modules.
3.  **Multimodal Reasoning as a First-Class Component**: This paper demonstrates that treating multimodal reasoning as a separate optimized component, rather than a side feature, can yield significant performance improvements for downstream tasks. This suggests a new paradigm for multimodal system design: separate specialized reasoning modules optimized for downstream task performance, rather than using generic off-the-shelf reasoning outputs.
### Potential Issues & Improvements
1.  **RL Training Data Scale**: The RL training stage uses only 19k samples, which is relatively small. While the paper shows strong performance on the tested benchmarks, the Reasoner may struggle to generalize to extremely out-of-domain tasks (e.g., medical images, specialized industrial video) not covered in the RL training set. Expanding the RL training dataset to cover more diverse domains would likely improve generalization further.
2.  **T-CoT Storage Overhead**: While T-CoT is generated offline, storing T-CoT chains for all items in a large-scale retrieval database (e.g., billions of videos/images) would add significant storage overhead. Future work could explore compressing T-CoT chains or distilling the reasoning ability directly into the Embedder to eliminate the need to store T-CoT chains, while retaining the performance benefits.
3.  **One-Size-Fits-All T-CoT**: The current T-CoT format uses the same structure for all task types. Future work could explore task-adaptive T-CoT formats that generate different types of cues for different tasks (e.g., more keyframes for long video retrieval, more text keywords for document retrieval) to further improve performance.
    Overall, Embed-RL represents a significant advance in multimodal embedding research, providing a practical, high-performance framework that addresses key limitations of prior methods, and opens up promising new directions for integrating reasoning and representation learning.