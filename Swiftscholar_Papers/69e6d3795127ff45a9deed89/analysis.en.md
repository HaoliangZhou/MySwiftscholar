# 1. Bibliographic Information

## 1.1. Title
The title of the paper is `MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model`. This indicates that the central topic is a novel framework named `MuCo` designed to improve the training of multimodal embedding models by employing a multi-turn, dialogue-inspired approach to contrastive learning.

## 1.2. Authors
The authors are Geonmo Gu, Byeongho Heo, Jaemyung Yu, Jaehui Hwang, Taekyung Kim, Sangmin Lee, HeeJae Jun, Yoohoon Kang, Sangdoo Yun, and Dongyoon Han.
Their research backgrounds and affiliations are as follows:
*   **NAVER AI Lab:** Geonmo Gu, Byeongho Heo, Jaemyung Yu, Jaehui Hwang, Taekyung Kim, Sangdoo Yun, Dongyoon Han.
*   **NAVER AI Search Platform:** HeeJae Jun, Yoohoon Kang.
*   **Korea University:** Geonmo Gu, Sangmin Lee.

    The team's affiliation with NAVER, a major technology corporation, and Korea University suggests a strong industry-academia collaboration focused on advancing artificial intelligence, specifically in the domains of Multimodal Large Language Models (MLLMs) and representation learning.

## 1.3. Journal/Conference
The paper was published on **arXiv** (arXiv:2602.06393) on February 6, 2026. As an arXiv preprint, it has not yet undergone peer review by a journal or conference. However, arXiv is a highly reputable and widely used repository for preliminary scientific research in fields like computer science, signaling that the work is ready for community scrutiny.

## 1.4. Publication Year
The publication year is **2026**.

## 1.5. Abstract
The paper's research objective is to enhance the efficiency and effectiveness of universal multimodal embedding models built on Multimodal Large Language Models (MLLMs). The authors identify a key limitation in existing methods: they rely on a "single-turn" contrastive learning formulation where each query-target pair is processed independently. This leads to computational inefficiency (requiring separate forward passes for each pair) and fails to capture the rich contextual relationships between multiple queries related to the same context (e.g., an image).

The core methodology is **Multi-turn Contrastive Learning (MuCo)**. This framework reframes embedding extraction as a multi-turn dialogue. Instead of processing isolated pairs, MuCo processes multiple related query-target pairs for a single image within a single forward pass. It leverages the conversational nature of MLLMs to extract multiple query and target embeddings simultaneously, conditioned on a shared context representation.

The main results demonstrate that MuCo, when trained on a newly curated 5M multimodal multi-turn dataset (M3T), achieves state-of-the-art retrieval performance on the MMEB and M-BEIR benchmarks. The method also significantly enhances training efficiency and representation coherence across modalities.

## 1.6. Original Source Link
*   **arXiv Link:** https://arxiv.org/abs/2602.06393
*   **PDF Link:** https://arxiv.org/pdf/2602.06393v2
*   **Publication Status:** Preprint (not yet peer-reviewed).

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the inefficiency and suboptimal representation learning in existing universal multimodal embedding models. These models, which map diverse data types like images and text into a unified vector space, are crucial for tasks such as image-text retrieval, visual question answering, and classification. They are typically trained using contrastive learning, a technique that aligns representations of related items and pushes apart unrelated ones.

This problem is important because as these models scale, the computational cost of training becomes a major bottleneck. Existing "single-turn" contrastive learning methods have two critical limitations:
1.  **Computational Inefficiency:** They require a separate forward pass for each query-target pair. Since processing images is computationally expensive (much more so than text), simply increasing the batch size to get more negative samples leads to a prohibitive increase in FLOPs (floating-point operations).
2.  **Overlooked Contextual Relationships:** By treating each query-target pair as an independent data point, these methods fail to model the rich interdependence between multiple queries that can be derived from the same image. For instance, different questions about the same image are not independent; they share a common visual context. Processing them independently misses an opportunity to build a more coherent and context-aware representation.

    The paper's innovative entry point is to rethink the entire training paradigm. Instead of the traditional "single-turn" approach, the authors propose a "multi-turn" dialogue-inspired framework. This idea is rooted in the observation that MLLMs are inherently designed for multi-turn conversations. By structuring the training data as a dialogue where multiple queries and their targets are presented sequentially for a single image, the model can leverage its built-in causal attention mechanisms to process them efficiently and learn richer, context-dependent embeddings.

## 2.2. Main Contributions / Findings
The paper's primary contributions are:
1.  **The MuCo Framework:** A novel multi-turn contrastive learning framework that processes multiple query-target pairs for a single image in a single forward pass. This is achieved by using a dialogue template where an image is presented once, followed by a sequence of text-only query and target turns. This approach dramatically increases the *effective* batch size (providing more negative samples for the contrastive loss) with minimal additional computational cost.
2.  **The M3T Dataset:** A newly curated, large-scale (5M images, 35M pairs) multimodal multi-turn dataset. This dataset is synthesized using a two-step pipeline: first, a powerful MLLM generates a dense caption for each image; second, a Large Language Model (LLM) generates seven diverse query-target pairs (classification, retrieval, and various types of VQA) based on the caption. This dataset is specifically designed to train the MuCo framework.
3.  **A Single-turn Adaptation Strategy:** A method to apply the MuCo framework to standard, single-turn datasets during fine-tuning. This is done by simulating a multi-turn interaction using a prompt-based in-context reconstruction task, where the model is asked to reconstruct a masked counterpart of the query or target. This allows the benefits of multi-turn learning to be transferred to conventional benchmarks.
4.  **Comprehensive Experimental Validation:** Extensive experiments showing that MuCo achieves state-of-the-art performance on the MMEB and M-BEIR benchmarks, outperforming previous methods. The results also demonstrate remarkable efficiency gains, achieving a 7x increase in effective batch size with only a ~3% increase in FLOPs.

    The key findings are:
*   **Performance:** MuCo-7B achieves the best overall score on the MMEB benchmark (73.6%) and the M-BEIR benchmark (56.6%), surpassing previous state-of-the-art models.
*   **Efficiency:** The multi-turn approach is far more efficient than simply scaling the batch size. For example, to achieve a similar effective batch size, MuCo requires ~18 PFLOPs, whereas a standard contrastive learning method would require ~122.7 PFLOPs.
*   **Effectiveness of Components:** Ablation studies confirm the importance of the proposed "compounded supervision" (where gradients from later turns flow back to refine earlier embeddings) and the "logit masking" strategy (to prevent semantic overlap issues).
*   **Generalization:** The method generalizes well, as evidenced by strong out-of-distribution (OOD) performance on the MMEB benchmark.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, a reader must be familiar with several foundational concepts:

*   **Multimodal Large Language Models (MLLMs):** These are advanced AI models that can process and generate information across multiple modalities, such as text and images. They are built by combining a vision encoder (like a Vision Transformer, ViT) with a Large Language Model (LLM). The vision encoder converts an image into a sequence of feature vectors, which are then fed into the LLM alongside text tokens. This allows the model to "see" and "read" simultaneously, enabling it to answer questions about images, describe scenes, and perform other complex vision-language tasks. Examples include GPT-4V, LLaVA, and Qwen-VL.
*   **Embedding Models:** An embedding model is a neural network that maps input data (like text, images, or audio) into a vector space, where each data point is represented by a fixed-size vector (an "embedding"). The key property of a good embedding space is that semantically similar items are close together, and dissimilar items are far apart. This is fundamental for tasks like retrieval (finding the most similar image to a text query).
*   **Contrastive Learning:** This is a self-supervised learning technique used to train embedding models. The core idea is to pull together the embeddings of "positive" pairs (e.g., an image and its matching caption) and push apart the embeddings of "negative" pairs (e.g., an image and an unrelated caption). The most common loss function for this is **InfoNCE loss**.
    *   **InfoNCE Loss:** This loss function is used in contrastive learning. For a given query (e.g., an image), it maximizes the similarity to its positive target (e.g., the correct caption) while minimizing the similarity to all other targets in the batch (which act as negatives). The formula is typically:
        $$ \mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{K} \exp(\text{sim}(z_i, z_k)/\tau)} $$
        where $z_i$ is the query embedding, $z_j$ is the positive target embedding, $z_k$ are all target embeddings in the batch, $\text{sim}$ is a similarity function (like cosine similarity), and $\tau$ is a temperature parameter that controls the sharpness of the distribution.
*   **Causal Attention (or Autoregressive Modeling):** This is a key mechanism in transformer-based LLMs. It means that when processing a sequence of tokens, the model can only attend to previous tokens and not future ones. This property is essential for generating text sequentially. In the context of this paper, it allows the model to process a multi-turn dialogue where the representation of a later turn can depend on all previous turns in the conversation.

## 3.2. Previous Works
The authors situate their work within the broader context of universal multimodal embedding models, which can be grouped into two main paradigms:

1.  **CLIP-based Methods:** These methods, like CLIP itself, are foundational. They use a dual-encoder architecture with separate image and text encoders and are trained with a simple contrastive loss on large-scale image-text pairs. Later works introduced architectural or objective variations to improve alignment.
2.  **MLLM-based Approaches:** This is the more recent and powerful paradigm. Instead of separate encoders, these methods leverage the strong multimodal understanding and instruction-following capabilities of MLLMs (like LLaVA or Qwen-VL) to produce embeddings. They often use a single MLLM to process both modalities.
    *   **Scaling and Data Curation:** Works like **VLM2Vec**, **MMRet**, and **mmE5** focused on creating better, larger, or more diverse training datasets to improve embedding quality. mmE5, for instance, automatically generated aligned supervision data.
    *   **Training Procedure Improvements:** Other works focused on the training objective itself. **E5-V** unified representations through MLLM distillation. **UniME** and **LLaVE** improved supervision through refined negative sampling. **B3** enhanced scalability via optimized batch construction. **M3Task-UEM** introduced task-adaptive representation learning.

        The paper explicitly notes that while these MLLM-based models advance the field, they predominantly rely on **single-turn supervision**, which is the limitation MuCo aims to address.

## 3.3. Technological Evolution
The field has has evolved from simple, dual-encoder CLIP-like models to more sophisticated MLLM-based models. The shift to MLLMs brought about a significant increase in multimodal understanding capabilities. The focus then moved from architectural changes to data curation (creating better datasets) and then to refining the training objectives (e.g., better negative sampling).

The current state-of-the-art involves using powerful MLLMs as backbones for embedding tasks. The key challenge now is scaling these models efficiently. The traditional way to improve contrastive learning is to increase the batch size, which provides more negative samples for the InfoNCE loss. However, in the multimodal setting, this is extremely expensive because each item in the batch contains an image.

This paper's work fits into this timeline as the next logical step: instead of just scaling data or batch size, it proposes a fundamental change in the *training paradigm*. It moves from a "single-turn" to a "multi-turn" formulation, which is a novel way to exploit the architectural properties of MLLMs (causal attention) to achieve the benefits of a large batch size without the associated computational cost.

## 3.4. Differentiation Analysis
Compared to the main methods in related work, MuCo's core differences and innovations are:

*   **Training Paradigm:** The most significant difference is the shift from single-turn to multi-turn contrastive learning. Previous methods treat each query-target pair independently. MuCo processes multiple related pairs in a single, dialogue-style forward pass.
*   **Computational Efficiency:** Unlike methods like B3 that try to optimize batch construction, MuCo's efficiency gain comes from a structural change in how data is presented to the model. By processing an image once and then having multiple text-only turns, it avoids the repeated, expensive image encoding. This allows for a large number of effective training pairs (queries and targets) without a linear increase in computational cost.
*   **Contextual Learning:** MuCo is designed to learn contextually coherent representations. Because later turns in the dialogue can attend to the history of the conversation, the model is forced to learn embeddings that are not just aligned with their specific target but are also consistent with the broader context of the conversation. This "compounded supervision" is a unique feature not present in single-turn methods.
*   **Data Construction:** The paper introduces a novel, dialogue-oriented dataset (M3T) specifically for its training paradigm. This is different from previous works that might have used aggregated single-turn datasets or synthetically generated single pairs. M3T explicitly contains multiple, diverse query-target pairs for each image, structured as a conversation.

# 4. Methodology

## 4.1. Principles
The core principle of the MuCo method is to reframe the embedding extraction process as a **multi-turn dialogue**. This is inspired by the inherent design of MLLMs, which are built to handle conversational inputs using causal attention. The intuition is that if we present an image and then a sequence of related queries and targets, the model can process them all in one go. This is more efficient than processing each pair separately. Furthermore, because each turn can attend to the history of the conversation, the model can learn richer, more context-aware embeddings. The gradients from later turns can flow back to refine the representations of earlier turns, creating a "compounded supervisory signal."

The theoretical basis rests on two key ideas:
1.  **Efficiency through Shared Context:** The heavy visual encoding is done only once per image. The subsequent turns are text-only and computationally cheap. This allows for a large number of effective training pairs (queries and targets) without a linear increase in computational cost.
2.  **Representation Coherence through Causal Attention:** By processing queries sequentially, the model is encouraged to learn embeddings that are not only aligned with their specific target but are also consistent with the broader context established by previous turns in the dialogue.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. The MuCo Pretraining Framework
The method is implemented through a two-stage training process: pretraining on the M3T dataset and fine-tuning on standard benchmarks. We will first detail the pretraining stage.

**Step 1: Dialogue-Templated Data Construction (M3T Dataset)**
The foundation of MuCo is a specially constructed dataset called **M3T** (Multi-modal Multi-turn). The authors synthesize this dataset using a two-step pipeline:
1.  **Dense Captioning:** An image is first processed by a powerful MLLM (e.g., Qwen2.5-VL-75B) to generate a dense, objective caption that captures key objects, attributes, and spatial relationships.
2.  **Multi-turn Pair Synthesis:** This dense caption is then fed into a Large Language Model (LLM) with a carefully designed prompt. The LLM is instructed to generate seven diverse query-target pairs for the image, structured to align with the core categories of the MMEB benchmark:
    *   1 **Classification** pair
    *   1 **Retrieval** pair
    *   5 **Visual Question Answering (VQA)** pairs (2 global, 2 local, 1 creative)

        This process yields 35 million query-target text pairs from 5 million images. The key is that all seven pairs for a single image are synthesized together, forming a coherent, multi-turn dialogue.

**Step 2: Multi-turn Input Formatting**
For training, these multiple pairs are formatted into a single input sequence for the MLLM. The format is a dialogue template where the image is presented once at the beginning. This is followed by a sequence of turns. Each turn consists of a user query and an assistant response. Crucially, a special embedding token, $<|emb|>$, is placed at the end of the assistant's response in each turn. This token's final hidden state is extracted as the embedding for that turn.

Let $\mathcal{T}_i$ be the $i$-th image. Let $q_i^j$ and $p_i^j$ be the $j$-th query text and its corresponding positive target text for that image. The formatted input for the $j$-th turn, $\bar{q}_i^j$ and $\bar{p}_i^j$, is constructed by concatenating the image with all previous queries and targets:

$$ \bar{q}_i^j = (\mathcal{T}_i, (q_i^l)_{l \leq j}), \quad \bar{p}_i^j = (p_i^l)_{l \leq j} $$

This cumulative structure is the essence of MuCo. The model processes the entire sequence in a single forward pass. The embedding for the $j$-th turn is extracted from the $<|emb|>$ token at the end of the $j$-th turn. Because of the MLLM's causal attention, this embedding is conditioned on the image and all previous turns in the dialogue.

**Step 3: The Multi-turn Contrastive Loss**
The standard contrastive loss, InfoNCE, is applied to these multi-turn embeddings. For a mini-batch $\mathcal{B}$ of images, the loss is:

$$ \mathcal{L}_{\text{MuCo}} = \frac{1}{|\mathcal{B}|} \sum_{(\bar{q}_i^j, \bar{p}_i^j) \in \mathcal{B}} -\log \frac{\phi(\bar{q}_i^j, \bar{p}_i^j)}{\sum_{\bar{p} \in \mathcal{N}_i \cup \{\bar{p}_i^j\}} \phi(\bar{q}_i^j, \bar{p})} $$

Here, $\phi(\bar{q}_i^j, \bar{p}) = \exp(f(\bar{q}_i^j)^\top f(\bar{p}) / \tau)$ is the similarity score, where $f(\cdot)$ is the embedding function, and $\tau$ is the temperature. The denominator sums over the positive target and all negative targets.

**Step 4: Logit Masking Strategy**
A critical innovation is the definition of the negative set $\mathcal{N}_i$. In the standard contrastive setting, all other targets in the batch are negatives. However, in MuCo, multiple targets originate from the *same* image. Treating these as negatives would be incorrect, as they are semantically related (e.g., different answers to questions about the same image). To address this, the authors define the negative set $\mathcal{N}_i$ to explicitly exclude all other positive targets from the same image $\mathcal{T}_i$:

$$ \mathcal{N}_i = \{ \bar{p}_k^l \ | \ (\cdot, \bar{p}_k^l) \in \mathcal{B}, \ k \neq i \} $$

In practice, this is implemented by masking the corresponding logits in the similarity matrix to negative infinity ($-\infty$), effectively removing them from consideration. This "logit masking" is crucial for preventing the model from being penalized for correctly aligning related queries from the same image.

![Figure 3. Logit masking strategy in our MuCo framework. (a) The conventional method with a batch size of $N = 4$ yields a $N \\times N$ (i.e. $4 \\times 4 )$ matrix. In contrast, MuCo (b) uses a batch size of $N = 2$ and $k = 4$ turns to construct a larger $N k \\times N k$ (i.e. $8 \\times 8 )$ matrix. Crucially, our method masks out pairs originating from the same image (gray, $- \\infty )$ to prevent a semantic overlap issue. True positives (blue, `+` ) and true negatives (orange, `-` are used for the loss. Crucially, other pairs originating from the same image (gray, $- \\infty )$ are masked to prevent a semantic overlap issue.](images/3.jpg)
*该图像是示意图，展示了MuCo框架中的Logit掩蔽策略。从左侧的(a)部分可以看到传统方法的logit矩阵，批量大小为$N = 4$，生成一个$4 \times 4$的矩阵。而右侧的(b)部分展示了MuCo方法，使用批量大小$N = 2$和$k = 4$轮次，构建了一个$8 \times 8$的矩阵。此外，图中用灰色表示的$- \infty$对来自同一图像的配对进行了掩蔽，以防止语义重叠。*

This figure illustrates the logit masking strategy. The conventional method (a) uses a batch size of $N=4$, resulting in a $4 \times 4$ logit matrix. MuCo (b) uses a batch size of $N=2$ images but $k=4$ turns per image, constructing a larger $8 \times 8$ matrix. The gray cells, representing pairs from the same image, are masked to $-\infty$ to prevent them from being treated as negatives.

### 4.2.2. The MuCo Fine-tuning Framework (Single-turn Adaptation)
The pretraining stage uses the multi-turn M3T dataset. To apply MuCo to standard, single-turn benchmarks (like MMEB or M-BEIR), the authors introduce an adaptive strategy.

**Step 1: Simulating Multi-turn Interactions**
For a given single query-target pair `(q, p)`, the authors create an augmented version that simulates a multi-turn dialogue. The core idea is an "in-context reconstruction task."

The augmented query $q'$ and target $p'$ are constructed by adding a prompt that asks the model to reconstruct a masked version of the counterpart.
*   $q' = (q, \pi_1, \tilde{p}, \pi_2)$
*   $p' = (p, \pi_1, \tilde{q}, \pi_2)$

    Here, $\pi_1$ is a prompt like "User: Please rewrite your last response in human-readable language," and $\pi_2$ is a prompt like "User: Reconstruct the previous response, acknowledge my query, and seamlessly integrate the answer." $\tilde{p}$ and $\tilde{q}$ are the masked versions of the target and query, respectively. The masking is done by randomly replacing 50% of the words with a special mask token (e.g., $<|mask|>$).

    ![Figure 4. Multi-turn template for fine-tuning MuCo on singlepair datasets. We illustrate Query (left) and Positive (right) templates. The initial query (cyan) is reused as a masked target on the Positive side, and the positive target (pink) becomes a masked target on the Query side. This process simulates multi-turn interactions from a single pair, guiding the model to reconstruct its counterpart and enrich the learned embeddings.](images/4.jpg)
    *该图像是示意图，展示了针对单对数据集的MuCo微调中的多回合模板。左侧为查询，右侧为正例模板，初始查询在正例中作为掩码目标重用，而正例的目标则在查询中成为掩码目标。这一过程模拟了来自单一对的多回合交互，指导模型重构对应内容，丰富学习的嵌入。*

This figure shows the multi-turn template for fine-tuning. The initial query (cyan) is reused as a masked target on the positive side, and the positive target (pink) becomes a masked target on the query side. This simulates a multi-turn interaction, compelling the model to reason about the relationship between the query and target to reconstruct the masked parts.

**Step 2: Augmented Batch Construction**
For an original mini-batch $\mathcal{B}$ of single pairs, the authors create an augmented batch $\bar{\mathcal{B}}$ that includes all combinations of the original and augmented pairs:

$$ \bar{\mathcal{B}} = \bigcup_{(q, p) \in \mathcal{B}} \{ (q, p), (q, p'), (q', p), (q', p') \} $$

**Step 3: Contrastive Loss with Logit Masking**
The contrastive loss is then computed on this augmented batch. Crucially, to prevent the augmented versions of a pair from being treated as negatives of each other, a logit masking strategy is again employed. The negative set $\mathcal{N}_{p_i}$ for a target $p_i$ is defined as:

$$ \mathcal{N}_{p_i} = \{ p \vert (\cdot, p) \in \bar{\mathcal{B}} \} \backslash \{ p_i, p_i^- \} $$

where $p_i^-$ denotes the "opposite form" of $p_i$ (i.e., if $p_i$ is the original target, $p_i^-$ is its augmented version, and vice versa). This ensures that the model learns to align the original pair and its augmented versions without penalizing their semantic similarity.

The key intuition is that this reconstruction task forces the model to deeply understand the relationship between the query and the target. The gradients from the reconstruction task flow back to refine the initial embedding, making it more discriminative and context-aware.

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize several datasets across the pretraining and fine-tuning stages.

### 5.1.1. Pretraining Datasets
*   **M3T (Ours):** This is the primary dataset for pretraining MuCo. It is a newly constructed dataset containing 5 million images and 35 million query-target pairs. It is synthesized from the DataComp dataset using a two-step pipeline involving an MLLM (for dense captioning) and an LLM (for multi-turn pair generation). It is designed to be multi-turn and dialogue-oriented, with pairs covering classification, retrieval, and various types of VQA.
*   **mmE5:** A dataset of 560,000 samples used for comparison. It is a high-quality synthetic dataset for multimodal embeddings. The authors also experiment with a multi-turn version of mmE5 by converting its single-turn data into their dialogue format.
*   **MegaPairs:** A large-scale dataset of 26 million samples, also used for comparison. It consists solely of retrieval samples.

### 5.1.2. Fine-tuning Datasets (Benchmarks)
The models are fine-tuned and evaluated on two major benchmarks:

*   **MMEB (Multimodal Embedding Benchmark):** This benchmark is designed to evaluate universal multimodal embeddings. It contains 36 individual datasets across four meta-tasks:
    *   **Classification (10 datasets):** Including ImageNet-1K, N24News, HatefulMemes, VOC2007, SUN397, ImageNet-A, ImageNet-R, Place365, ObjectNet, and Country-211.
    *   **Visual Question Answering (10 datasets):** Including OK-VQA, A-OKVQA, DocVQA, InfographicsVQA, ChartQA, Visual7W-telling, ScienceQA, VizWiz, TextVQA, and GQA.
    *   **Retrieval (12 datasets):** Including VisDial, CIRR, VisualNews, MSCOCO, NIGHTS, WebQA, FashionIQ, Wiki-SS-NQ, OVEN, and EDIS.
    *   **Visual Grounding (4 datasets):** Including MSCOCO, RefCOCO, RefCOCO+, and RefCOCOg.
    
        The benchmark provides training samples from 20 of these datasets (In-Distribution) and reserves the remaining 16 for evaluation only (Out-of-Distribution).

*   **M-BEIR (Multimodal Benchmark for Evaluating Information Retrieval):** This benchmark focuses specifically on multimodal retrieval performance. It consists of 16 retrieval datasets across 8 retrieval tasks, covering various modality combinations (text-to-text, text-to-image, image-to-text, etc.). The datasets include VisualNews, MSCOCO, Fashion-200K, WebQA, EDIS, OVEN, InfoSeek, FashionIQ, and CIRR.

    The choice of these benchmarks is effective because they are comprehensive and widely recognized in the field. MMEB evaluates generalization across a wide range of tasks, while M-BEIR provides a deep dive into retrieval capabilities. Using both allows for a holistic assessment of the model's performance.

### 5.1.3. Data Sample Example
To illustrate the form of the data, here is a concrete example from the M3T dataset provided in the paper's supplementary material. This example shows the multi-turn structure for a single image of a "smoothie and cookies."

**Image:** A glass of pink smoothie on a white plate, with cookies and a banana in the background.

**Turn 1 (Classification):**
*   **Query:** "What type of dessert is served on the white plate?"
*   **Target:** "Smoothie."

**Turn 2 (Global VQA):**
*   **Query:** "Which topping appears on top of the pink smoothie?"
*   **Target:** "Crumbled nuts."

**Turn 3 (Global VQA):**
*   **Query:** "Which color, other than white, dominates the surface where the items are placed?"
*   **Target:** "Red."

**Turn 4 (Local VQA):**
*   **Query:** "To add crunch to the smoothie, which item in the setting would serve that purpose?"
*   **Target:** "Golden-brown cookies."

**Turn 5 (Local VQA):**
*   **Query:** "What is the texture of the golden-brown items on the right?"
*   **Target:** "Golden-brown."

**Turn 6 (Creative VQA):**
*   **Query:** "Which fruit in the background is ripe?"
*   **Target:** "Banana."

**Turn 7 (Retrieval):**
*   **Query:** "Describe the arrangement of food items presented."
*   **Target:** "A glass filled with a pink smoothie or yogurt-based dessert is placed on a white plate atop a red checkered tablecloth. The smoothie is topped with crumbled nuts and garnished with a dark berry. To the right of the glass, there are two round, golden-brown cookies resting on the tablecloth. In the background, a ripe banana and a small pink bowl containing dark berries are visible, adding to the overall fresh and healthy presentation. The setting appears bright and clean, emphasizing the vibrant colors of the food items."

    This example clearly shows how multiple, diverse queries are derived from a single image, forming a coherent dialogue.

## 5.2. Evaluation Metrics
The paper uses standard evaluation metrics for the tasks involved.

### 5.2.1. Precision@1 (P@1)
*   **Conceptual Definition:** Precision@1 is a metric used for ranking and retrieval tasks. It measures the proportion of queries for which the top-ranked (i.e., the first) result is the correct (relevant) one. In the context of the MMEB benchmark, which reframes tasks like classification and VQA as ranking problems with up to 1,000 candidates, P@1 is the primary metric. A higher P@1 indicates that the model's embedding is highly discriminative, placing the correct answer at the very top of the ranked list.
*   **Mathematical Formula:**
    $$ \text{Precision@1} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbb{I}(\text{rank}(q, \text{correct\_target}) = 1) $$
*   **Symbol Explanation:**
    *   $|\mathcal{Q}|$ is the total number of queries in the evaluation set.
    *   $\text{rank}(q, \text{correct\_target})$ is the rank of the correct target for query $q$ in the list of candidates sorted by similarity.
    *   $\mathbb{I}(\cdot)$ is the indicator function, which is 1 if the condition is true and 0 otherwise.

### 5.2.2. Recall@k (R@k)
*   **Conceptual Definition:** Recall@k is another common metric for retrieval tasks. It measures the proportion of queries for which the correct target appears within the top-$k$ ranked results. It is less strict than Precision@1, as it only requires the correct answer to be in the top-$k$ list, not necessarily the very first. The M-BEIR benchmark uses Recall@5 or Recall@10, depending on the specific dataset. This metric is useful for evaluating how well a model retrieves relevant items from a larger pool.
*   **Mathematical Formula:**
    $$ \text{Recall@k} = \frac{1}{|\mathcal{Q}|} \sum_{q \in \mathcal{Q}} \mathbb{I}(\text{rank}(q, \text{correct\_target}) \leq k) $$
*   **Symbol Explanation:**
    *   $|\mathcal{Q}|$ is the total number of queries.
    *   $\text{rank}(q, \text{correct\_target})$ is the rank of the correct target.
    *   $\mathbb{I}(\cdot)$ is the indicator function.
    *   $k$ is the cutoff rank (e.g., 5 or 10).

## 5.3. Baselines
The paper compares MuCo against a strong set of baseline models, which are representative of the state-of-the-art in MLLM-based embedding models.

*   **CLIP:** The foundational contrastive language-image pre-training model. It serves as a classic, non-MLLM baseline.
*   **VLM2Vec:** A method that trains vision-language models for massive multimodal embedding tasks.
*   **LLaVE:** A model that uses hardness-weighted contrastive learning to improve training.
*   **UniME:** A model that breaks the modality barrier and improves supervision through refined negative sampling.
*   **B3 (Breaking the Batch Barrier):** A model that enhances scalability via optimized batch construction. It is a strong baseline for efficient contrastive learning.
*   **MoCa:** A model that uses modality-aware continual pre-training for better bidirectional multimodal embeddings.
*   **mmE5:** A model that improves multimodal multilingual embeddings via high-quality synthetic data. It is a previous state-of-the-art model, particularly noted for its zero-shot performance.
*   **MMRet:** A model that focuses on scaling and curating data for embedding learning.
*   **M3Task-UEM:** A model that introduces task-adaptive representation learning.

    These baselines are representative because they cover the main approaches in the field: data-centric (mmE5, MMRet), objective-centric (LLaVE, UniME, MoCa), and efficiency-centric (B3). Comparing against them provides a comprehensive validation of MuCo's performance.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main experimental results strongly validate the effectiveness of the proposed MuCo framework.

### 6.1.1. MMEB Benchmark Results
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Models</th>
<th rowspan="2"># Params</th>
<th colspan="4">Per Meta-Task Score</th>
<th colspan="3">Average Score</th>
</tr>
<tr>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>Grounding</th>
<th>ID</th>
<th>OOD</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="9">Zeroshot setting (pretrained) on MMEB benchmark</td>
</tr>
<tr>
<td>CLIP [53]</td>
<td>0.4B</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td></td>
<td></td>
<td>37.8</td>
</tr>
<tr>
<td>MagicLens [67]</td>
<td>0.6B</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td></td>
<td></td>
<td>27.8</td>
</tr>
<tr>
<td>E5-V [29]</td>
<td>8B</td>
<td>21.8</td>
<td>4.9</td>
<td>11.5</td>
<td>19.0</td>
<td>−</td>
<td>−</td>
<td>13.3</td>
</tr>
<tr>
<td>MMRet [69]</td>
<td>7B</td>
<td>47.2</td>
<td>18.4</td>
<td>56.5</td>
<td>62.2</td>
<td>−</td>
<td>−</td>
<td>44.0</td>
</tr>
<tr>
<td>mmE5 [8]</td>
<td>11B</td>
<td>60.6</td>
<td>55.7</td>
<td>54.7</td>
<td>72.4</td>
<td>−</td>
<td></td>
<td>58.6</td>
</tr>
<tr>
<td><strong>MuCo-2B</strong></td>
<td>2B</td>
<td>53.6</td>
<td>59.9</td>
<td>55.2</td>
<td>74.6</td>
<td></td>
<td></td>
<td><strong>58.2</strong></td>
</tr>
<tr>
<td><strong>MuCo-7B</strong></td>
<td>7B</td>
<td><strong>56.0</strong></td>
<td><strong>64.7</strong></td>
<td><strong>58.9</strong></td>
<td><strong>75.7</strong></td>
<td></td>
<td></td>
<td><strong>61.6</strong></td>
</tr>
<tr>
<td colspan="9">Fine-tuning on MMEB benchmark (&lt; 7B Models)</td>
</tr>
<tr>
<td>CLIP [53]</td>
<td>0.4B</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>45.4</td>
</tr>
<tr>
<td>VLM2Vec [30]</td>
<td>4B</td>
<td>54.8</td>
<td>54.9</td>
<td>62.3</td>
<td>79.5</td>
<td>66.5</td>
<td>52.0</td>
<td>60.1</td>
</tr>
<tr>
<td>LLaVE [33]</td>
<td>2B</td>
<td>62.1</td>
<td>60.2</td>
<td>65.2</td>
<td>84.9</td>
<td>69.4</td>
<td>59.8</td>
<td>65.2</td>
</tr>
<tr>
<td>UniME [21]</td>
<td>4B</td>
<td>54.8</td>
<td>55.9</td>
<td>64.5</td>
<td>81.8</td>
<td>68.2</td>
<td>52.7</td>
<td>64.2</td>
</tr>
<tr>
<td>B3-2B [58]</td>
<td>2B</td>
<td>67.0</td>
<td>61.2</td>
<td>70.9</td>
<td>79.9</td>
<td>72.1</td>
<td>63.1</td>
<td>68.1</td>
</tr>
<tr>
<td>MoCa-3B [7]</td>
<td>3B</td>
<td>59.8</td>
<td>62.9</td>
<td>70.6</td>
<td>88.6</td>
<td>72.3</td>
<td>61.5</td>
<td>67.5</td>
</tr>
<tr>
<td><strong>MuCo-2B</strong></td>
<td>2B</td>
<td>66.2</td>
<td>65.6</td>
<td>70.1</td>
<td>85.8</td>
<td>72.9</td>
<td>65.0</td>
<td><strong>69.5</strong></td>
</tr>
<tr>
<td colspan="9">Fine-tuning on MMEB benchmark (≥ 7B Models)</td>
</tr>
<tr>
<td>VLM2Vec [30]</td>
<td>7B</td>
<td>61.2</td>
<td>49.9</td>
<td>67.4</td>
<td>86.1</td>
<td>67.5</td>
<td>57.1</td>
<td>62.9</td>
</tr>
<tr>
<td>MMRet [69]</td>
<td>7B</td>
<td>56.0</td>
<td>57.4</td>
<td>69.9</td>
<td>83.6</td>
<td>68.0</td>
<td>59.1</td>
<td>64.1</td>
</tr>
<tr>
<td>mmE5 [8]</td>
<td>11B</td>
<td>67.6</td>
<td>62.7</td>
<td>71.0</td>
<td>89.7</td>
<td>72.4</td>
<td>66.6</td>
<td>69.8</td>
</tr>
<tr>
<td>LLaVE [33]</td>
<td>7B</td>
<td>65.7</td>
<td>65.4</td>
<td>70.9</td>
<td>91.9</td>
<td>75.0</td>
<td>64.4</td>
<td>70.3</td>
</tr>
<tr>
<td>UniME [21]</td>
<td>7B</td>
<td>66.8</td>
<td>66.6</td>
<td>70.6</td>
<td>90.9</td>
<td>74.6</td>
<td>65.8</td>
<td>70.7</td>
</tr>
<tr>
<td>B3-7B [58]</td>
<td>7B</td>
<td>70.0</td>
<td>66.5</td>
<td>74.1</td>
<td>84.6</td>
<td>75.9</td>
<td>67.1</td>
<td>72.0</td>
</tr>
<tr>
<td>MoCa-7B [7]</td>
<td>7B</td>
<td>65.8</td>
<td>64.7</td>
<td>75.0</td>
<td>92.4</td>
<td>74.7</td>
<td>67.6</td>
<td>71.5</td>
</tr>
<tr>
<td><strong>MuCo-7B</strong></td>
<td>7B</td>
<td>68.3</td>
<td><strong>71.9</strong></td>
<td><strong>73.7</strong></td>
<td>90.9</td>
<td><strong>77.3</strong></td>
<td><strong>69.1</strong></td>
<td><strong>73.6</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Zero-shot Performance:** In the zero-shot setting, MuCo-7B achieves an overall score of **61.6%**, surpassing the previous SOTA, mmE5-11B (58.6%), despite having fewer parameters. This is a significant achievement, demonstrating the power of the multi-turn pretraining on the M3T dataset fanc. MuCo-2B also performs very competitively (58.2%).
*   **Fine-tuning Performance:** After fine-tuning, MuCo establishes new SOTA scores in both the <7B and ≥7B categories. MuCo-2B achieves **69.5%**, outperforming B3-2B (68.1%). MuCo-7B achieves the top score of **73.6%**, surpassing B3-7B (72.0%) and mmE5-11B (69.8%).
*   **Out-of-Distribution (OOD) Generalization:** A key strength of MuCo is its OOD performance. MuCo-7B achieves an OOD score of **69.1%**, which is the highest among all models. This indicates that the multi-turn, dialogue-inspired training leads to more robust and generalizable representations that transfer well to unseen tasks.
*   **Per-Task Performance:** MuCo shows particularly strong performance in VQA and Retrieval tasks, suggesting that the multi-turn structure is highly effective for tasks requiring reasoning and context understanding.

### 6.1.2. M-BEIR Benchmark Results
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th>Models</th>
<th colspan="5"># Params</th>
<th>Overall</th>
</tr>
<tr>
<th></th>
<th>S → S</th>
<th>S → M</th>
<th>M → S</th>
<th>M → M</th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td>UniIR [62]</td>
<td>0.4B</td>
<td>51.0</td>
<td>69.1</td>
<td>32.9</td>
<td>52.4</td>
<td>48.9</td>
</tr>
<tr>
<td>LamRA-Ret [43]</td>
<td>2B</td>
<td>47.6</td>
<td>66.2</td>
<td>41.0</td>
<td>61.0</td>
<td>50.0</td>
</tr>
<tr>
<td><strong>MuCo-2B</strong></td>
<td>2B</td>
<td>49.1</td>
<td>68.2</td>
<td>41.6</td>
<td><strong>64.8</strong></td>
<td><strong>51.6</strong></td>
</tr>
<tr>
<td>MM-Embed [38]</td>
<td>7B</td>
<td>50.9</td>
<td>76.9</td>
<td>40.0</td>
<td>60.9</td>
<td>52.7</td>
</tr>
<tr>
<td>LamRA-Ret [43]</td>
<td>7B</td>
<td>52.9</td>
<td>71.8</td>
<td>45.2</td>
<td>65.0</td>
<td>54.9</td>
</tr>
<tr>
<td>M3Task-UEM [56]</td>
<td>7B</td>
<td>54.0</td>
<td>74.9</td>
<td>41.9</td>
<td>55.7</td>
<td>53.9</td>
</tr>
<tr>
<td><strong>MuCo-7B</strong></td>
<td>7B</td>
<td><strong>54.0</strong></td>
<td><strong>71.6</strong></td>
<td><strong>47.4</strong></td>
<td><strong>70.4</strong></td>
<td><strong>56.6</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Overall Performance:** MuCo achieves state-of-the-art performance on the M-BEIR benchmark. MuCo-2B (51.6%) and MuCo-7B (56.6%) outperform all competing models in their respective parameter categories.
*   **Multi-modal to Multi-modal (M→M) Retrieval:** MuCo demonstrates particularly strong performance in the complex M→M setting, where both the query and the candidate are multimodal. MuCo-7B achieves a score of **70.4%**, significantly outperforming the next best model, LamRA-Ret-7B (65.0%). This suggests that MuCo's ability to learn rich, context-aware embeddings is highly beneficial for complex retrieval tasks.

### 6.1.3. Efficiency Analysis
The paper also presents a compelling analysis of the computational efficiency of MuCo compared to simply scaling the batch size in a standard contrastive learning setup.

The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th>#turns</th>
<th>#batch</th>
<th>#effective batch</th>
<th>PFLOPs</th>
<th>MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>1024</td>
<td>1024</td>
<td>17.5</td>
<td>57.1</td>
</tr>
<tr>
<td>1</td>
<td>2048</td>
<td>2048</td>
<td>35.1</td>
<td>57.3</td>
</tr>
<tr>
<td>1</td>
<td>4096</td>
<td>4096</td>
<td>70.2</td>
<td>57.4</td>
</tr>
<tr>
<td>1</td>
<td>7168</td>
<td>7168</td>
<td>122.7</td>
<td>57.5</td>
</tr>
<tr>
<td>1</td>
<td>8192</td>
<td>8192</td>
<td>140.4</td>
<td>57.8</td>
</tr>
<tr>
<td>2</td>
<td>1024</td>
<td>2048</td>
<td>17.6</td>
<td>57.4</td>
</tr>
<tr>
<td>4</td>
<td>1024</td>
<td>4096</td>
<td>17.7</td>
<td>57.7</td>
</tr>
<tr>
<td><strong>7</strong></td>
<td><strong>1024</strong></td>
<td><strong>7168</strong></td>
<td><strong>18.0</strong></td>
<td><strong>58.2</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Effective Batch Size vs. Computational Cost:** The table compares scaling the batch size (rows 1-5) versus scaling the number of turns (rows 6-8). The key finding is that MuCo can achieve a 7x increase in the *effective* batch size (from 1024 to 7168) with only a minimal increase in PFLOPs (from 17.5 to 18.0, a ~3% increase).
*   **Comparison with Standard Scaling:** To achieve the same effective batch size of 7168 using the standard method (row 4), the computational cost skyrockets to 122.7 PFLOPs. MuCo achieves this with only 18.0 PFLOPs, which is a massive efficiency gain.
*   **Performance vs. Efficiency:** Not only is MuCo more efficient, but it also performs better. The 7-turn MuCo model (effective batch 7168) achieves a score of **58.2%**, which is higher than the standard 1-turn model with a batch size of 7168 (57.5%) and even surpasses the larger 8192-batch baseline (57.8%). This demonstrates that the multi-turn structure provides a superior learning signal, not just a more efficient one.

## 6.2. Ablation Studies / Parameter Analysis
The authors conduct several ablation studies to validate the design choices of MuCo.

### 6.2.1. Impact of Pretraining Dataset
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Pre-training</th>
<th>Dataset</th>
<th>Samples</th>
<th>ZS MMEB</th>
<th>FT MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">Single-turn</td>
<td>None</td>
<td></td>
<td>−</td>
<td>68.5</td>
</tr>
<tr>
<td>mmE5 [8]</td>
<td>0.6M</td>
<td>55.6</td>
<td>68.6</td>
</tr>
<tr>
<td>MegaPairs [69]</td>
<td>26M</td>
<td>41.5</td>
<td>68.7</td>
</tr>
<tr>
<td rowspan="4">Multi-turn MuCo-§4.1</td>
<td>mmE5 [8]</td>
<td>0.6M</td>
<td>57.0</td>
<td>69.0</td>
</tr>
<tr>
<td>M3T (20%)</td>
<td>1M</td>
<td>57.1</td>
<td>69.0</td>
</tr>
<tr>
<td>M3T (60%)</td>
<td>3M</td>
<td>57.7</td>
<td>69.2</td>
</tr>
<tr>
<td><strong>M3T</strong></td>
<td><strong>5M</strong></td>
<td><strong>58.2</strong></td>
<td><strong>69.5</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   **MuCo Pretraining Effectiveness:** Converting the single-turn mmE5 dataset to the multi-turn format (row 4) improves both zero-shot (57.0 vs. 55.6) and fine-tuning (69.0 vs. 68.6) performance compared to using it in its original form. This confirms the advantage of the multi-turn signal.
*   **M3T Dataset Quality and Scale:** The performance improves monotonically as the scale of the M3T dataset increases (from 1M to 5M). The full 5M M3T dataset yields the best results (58.2 ZS, 69.5 FT). Notably, the model without any pretraining (None) achieves a competitive 68.5 FT score, showing that the MuCo fine-tuning strategy alone is very powerful.

### 6.2.2. Impact of Compounded Supervision
The following are the results from Table 6 of the original paper:

<table>
<thead>
<tr>
<th>Setup</th>
<th>ZS MMEB</th>
<th>FT MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o compounded supervision</td>
<td>57.3</td>
<td>68.4</td>
</tr>
<tr>
<td><strong>w/ compounded supervision</strong></td>
<td><strong>58.2</strong></td>
<td><strong>69.5</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   This ablation study tests the importance of the "compounded supervision" by disabling causal attention so that each turn cannot attend to previous turns.
*   The results show a clear performance drop when compounded supervision is disabled (68.4 vs. 69.5 FT MMEB). This confirms that the ability of later turns to refine earlier embeddings is a key factor in MuCo's success.

### 6.2.3. Impact of Logit Masking
The following are the results from Table 7 of the original paper:

<table>
<thead>
<tr>
<th colspan="4">Logit masking</th>
<th></th>
<th></th>
</tr>
<tr>
<th>Pretraining</th>
<th>Fine-tuning</th>
<th>ZS MMEB</th>
<th>FT MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td>-</td>
<td>-</td>
<td>57.7</td>
<td>31.1</td>
</tr>
<tr>
<td></td>
<td>✓</td>
<td>57.7</td>
<td>69.2</td>
</tr>
<tr>
<td>✓</td>
<td></td>
<td>58.2</td>
<td>30.9</td>
</tr>
<tr>
<td><strong>✓</strong></td>
<td><strong>✓</strong></td>
<td><strong>58.2</strong></td>
<td><strong>69.5</strong></td>
</tr>
</tbody>
</table>

**Analysis:**
*   This study validates the critical importance of logit masking, especially during fine-tuning.
*   Disabling logit masking in fine-tuning (rows 1 and 3) causes a catastrophic performance collapse (FT MMEB drops to ~31%). This is because without masking, the model treats the semantically similar augmented pairs as negatives, confusing the learning process.
*   The impact during pretraining is less severe, as the diverse multi-turn data is less prone to this semantic overlap issue.

### 6.2.4. Impact of Subsequent Turn Design for Fine-tuning
The following are the results from Table 8 of the original paper:

<table>
<thead>
<tr>
<th>Setup</th>
<th>MMEB</th>
</tr>
</thead>
<tbody>
<tr>
<td>Counterpart masking ratio 25%</td>
<td>69.3</td>
</tr>
<tr>
<td><strong>Counterpart masking ratio 50%</strong></td>
<td><strong>69.5</strong></td>
</tr>
<tr>
<td>Counterpart masking ratio 75%</td>
<td>68.9</td>
</tr>
<tr>
<td>Rephrasing template</td>
<td>68.5</td>
</tr>
<tr>
<td>Without reconstruction guidance</td>
<td>69.0</td>
</tr>
<tr>
<td>Image captioning (BLIP Large [36])</td>
<td>69.4</td>
</tr>
<tr>
<td>Image captioning (Qwen2-VL-7B [60])</td>
<td>69.5</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Masking Ratio:** A 50% masking ratio yields the best performance. Lower ratios (25%) provide too weak a signal, while higher ratios (75%) make the reconstruction task too difficult.
*   **Reconstruction Guidance:** Using the explicit reconstruction prompt is crucial. The "Without reconstruction guidance" variant (69.0) performs worse than the full method (69.5), confirming that the prompt-based reasoning task is beneficial.
*   **Image Captioning Model:** The choice of the offline image captioning model (BLIP vs. Qwen2-VL) has a minimal impact on performance, suggesting the method is robust to this choice.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper concludes by presenting MuCo as a novel and effective framework for training universal multimodal embedding models. Its main findings are that by reframing representation learning as a multi-turn dialogue, MuCo can learn richer, more coherent embeddings than traditional single-turn methods. Furthermore, it addresses the scalability bottleneck of contrastive learning by processing multiple query-target pairs in a single forward pass, leading to significant computational efficiency gains. The combination of the MuCo framework and the newly curated M3T dataset achieves state-of-the-art performance on major benchmarks (MMEB and M-BEIR). The work successfully demonstrates that it is possible to simultaneously enhance model performance and training scalability, effectively redefining the efficiency-capacity trade-off in multimodal alignment.

## 7.2. Limitations & Future Work
The authors do not explicitly list a dedicated "Limitations" section in the main text. However, the nature of the work suggests some implicit limitations and potential future directions:
*   **Dataset Dependence:** The method's success is heavily tied to the quality and structure of the M3T dataset. While the authors provide a synthesis pipeline, the reliance on synthetic data generated by other powerful models (MLLMs and LLMs) could be a limitation if the synthesis process introduces biases or errors.
*   **Complexity of Fine-tuning Adaptation:** The single-turn adaptation strategy, while effective, adds complexity to the fine-tuning process (requiring prompt-based reconstruction and data augmentation). Future work could explore simpler or more automatic ways to adapt the multi-turn paradigm to existing single-turn data.
*   **Generalization to Other Modalities:** The current work focuses on image-text multimodality. A natural future direction would be to explore whether the multi-turn dialogue paradigm can be effectively extended to other modalities like audio, video, or 3D data.
*   **Exploration of Dialogue Structures:** The paper uses a specific dialogue template. Future research could investigate other, more complex dialogue structures or interaction patterns to further enhance representation learning.

## 7.3. Personal Insights & Critique
The MuCo framework is a highly innovative and well-motivated piece of research. Its core strength lies in its clever exploitation of the architectural properties of MLLMs (causal attention) to solve a fundamental problem (efficiency of contrastive learning).

*   **Inspirations:** The idea of treating training as a dialogue is inspiring. It moves beyond the simple "query-target" pairing and embraces a more natural, human-like way of processing information. This "dialogue-as-training" paradigm could likely be transferred to other domains where sequential or contextual understanding is key, such as video understanding or time-series analysis. The "compounded supervision" concept, where later tasks refine earlier representations, is a powerful idea that could be applied in continual learning or multi-task learning scenarios.
*   **Potential Issues:** One potential issue is the reliance on a specific, synthesized dataset-d (M3T). While the results are strong, it raises the question of how well MuCo would perform on purely human-annotated, single-turn datasets without the elaborate fine-tuning adaptation strategy. The paper shows that fine-tuning alone is strong, but the full power comes from the multi-turn pretraining.
*   **Unverified Assumptions:** The paper assumes that processing multiple queries in a single forward pass is always beneficial. While the results support this, there might be scenarios where the queries are too diverse or unrelated, and forcing them into a single context could