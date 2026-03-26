# 1. Bibliographic Information

## 1.1. Title
SDR-CIR: Semantic Debias Retrieval Framework for Training-Free Zero-Shot Composed Image Retrieval

## 1.2. Authors
Yi Sun\*, Jinyu Xu\*, Qing Xie, Jiachen Li, Yongjian Liu (Wuhan University of Technology, Wuhan, China); Yanchun Ma (Wuhan Vocational College of Software and Engineering, Wuhan, China).
\* Indicates co-first authors.

## 1.3. Journal/Conference
Proceedings of the ACM Web Conference 2026 (WWW '26), April 13-17, 2026, Dubai, United Arab Emirates. This is a highly reputable conference in the field of the World Wide Web, covering a broad range of topics including information retrieval, AI, and social computing, indicating a strong peer-review process and high impact within the research community.

## 1.4. Publication Year
2026

## 1.5. Abstract
Composed Image Retrieval (CIR) aims to retrieve a target image based on a query consisting of a reference image and modification text. Recent training-free zero-shot CIR (ZS-CIR) methods frequently use Multimodal Large Language Models (MLLMs) with `Chain-of-Thought` (CoT) reasoning to generate a target image description for retrieval. However, due to the inherent fuzzy matching nature of ZS-CIR, the generated description often suffers from `semantic bias` relative to the actual target image.

This paper introduces `SDR-CIR`, a novel training-free `Semantic Debias Ranking` method built upon CoT reasoning. `SDR-CIR` addresses two main challenges:
1.  **Visual Noise Reduction:** It first employs `Selective CoT` to guide the MLLM to extract only visual content relevant to the modification text during image understanding, thereby reducing `visual noise` at its source.
2.  **Semantic Bias Mitigation:** It then introduces a `Semantic Debias Ranking` module, which operates in two steps:
    *   **Anchor Step:** This step fuses `reference image features` with `target description features` to reinforce useful semantics and supplement any omitted cues.
    *   **Debias Step:** This step explicitly models the `visual semantic contribution` of the `reference image` to the `description` and incorporates it into the similarity score as a penalty term.

        By supplementing omitted cues while suppressing redundancy, `SDR-CIR` effectively mitigates `semantic bias` and significantly enhances retrieval performance. Experimental evaluations on three standard CIR benchmarks (CIRR, CIRCO, and FashionIQ) demonstrate that `SDR-CIR` achieves state-of-the-art results among one-stage methods while maintaining high efficiency. The accompanying code is publicly available.

## 1.6. Original Source Link
https://arxiv.org/abs/2602.04451 (Preprint, last updated 2026-02-04T11:24:35.000Z)

## 1.7. PDF Link
https://arxiv.org/pdf/2602.04451v2 (Preprint, last updated 2026-02-04T11:24:35.000Z)

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem addressed by the paper is `Composed Image Retrieval (CIR)`, which involves retrieving a target image using a query composed of a `reference image` and `modification text`. This task is crucial for applications like e-commerce and web search, where users might want to find images similar to a given one but with specific changes (e.g., "this shirt but in red").

`CIR` is challenging because it requires a joint understanding of the visual content of the `reference image` and the semantic changes requested by the `modification text`. Traditional `CIR` methods rely on extensive training data in the form of `(reference image, modification text, target image)` triplets. Constructing such triplets is labor-intensive and limits the generalization capability of these models.

To overcome this data scarcity, `Zero-Shot CIR (ZS-CIR)` methods have emerged, leveraging large-scale pre-trained `visual-language models (VLMs)` to perform retrieval without task-specific triplet supervision. Among `ZS-CIR` methods, `training-free` approaches are particularly promising due to the rapid advancements in `Large Language Models (LLMs)` and `Multimodal Large Language Models (MLLMs)`. These methods typically involve two stages: generating a `target image description` from the multimodal query and then performing text-to-image retrieval based on this description.

However, existing `training-free ZS-CIR` methods face two significant challenges:
1.  **Visual Noise in Description Generation:** Many one-stage `MLLM-based` methods use `Chain-of-Thought (CoT)` prompts to first extract almost all visual information from the `reference image` and then filter irrelevant details. This `extract-then-filter` approach, lacking `semantic selectivity` during initial extraction, can introduce irrelevant visual information (visual noise) into the generated description, degrading retrieval performance.
2.  **Semantic Bias in Ranking:** `ZS-CIR` is inherently a `fuzzy matching` task, meaning the `target image` might differ from the `reference image` in implicit semantic ways beyond the explicit modification. Consequently, the generated `target image description` often only partially corresponds to the actual `target image`. This leads to `semantic bias`, which can be either `redundancy bias` (description contains irrelevant details from the reference image) or `omission bias` (description omits useful cues from the reference image). This bias misguides the ranking process, leading to incorrect retrieval results.

    The paper's entry point is to address these specific challenges within `training-free one-stage ZS-CIR` by introducing mechanisms to reduce visual noise during description generation and mitigate semantic bias during ranking.

## 2.2. Main Contributions / Findings
The paper proposes `SDR-CIR` and highlights the following primary contributions:
*   **Introduction of `Selective CoT`:** A novel `Chain-of-Thought` prompting strategy that guides `MLLMs` to selectively extract visual content relevant to the `modification text` during `reference image understanding`. This significantly reduces `visual noise` in the generated `target image description`, improving its semantic alignment with the intended target.
*   **Development of `Semantic Debias Ranking` (SDR):** A two-step module (`Anchor` and `Debias`) designed to explicitly model and correct `semantic bias` stemming from the `reference image` in the `ranking process`.
    *   The `Anchor` step fuses `reference image features` with `target description features` to reinforce useful semantics and supplement any omitted cues, creating a more robust composed query.
    *   The `Debias` step explicitly models the `visual semantic contribution` of the `reference image` to the `description` and incorporates it as a penalty term in the similarity score, thereby suppressing redundant semantics and preventing non-target candidates from being highly ranked.
*   **Demonstration of State-of-the-Art Performance:** Experiments on three standard `CIR` benchmarks (CIRR, CIRCO, FashionIQ) show that `SDR-CIR` achieves state-of-the-art results among one-stage methods, consistently outperforming existing approaches. This validates its effectiveness in mitigating semantic bias and maintaining high retrieval accuracy across diverse modification types (object-level and attribute-level) while maintaining high efficiency.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### 3.1.1. Composed Image Retrieval (CIR)
`Composed Image Retrieval (CIR)` is a specialized form of image retrieval where the query is a combination of two modalities: a `reference image` and a `modification text`. The goal is to find a `target image` that matches the `reference image` after applying the changes described in the `modification text`. For example, given an image of a red car and the text "change to blue", the system should retrieve an image of a blue car.

### 3.1.2. Zero-Shot Composed Image Retrieval (ZS-CIR)
`Zero-Shot Composed Image Retrieval (ZS-CIR)` refers to performing `CIR` without requiring task-specific training data (i.e., `(reference image, modification text, target image)` triplets). Instead, `ZS-CIR` methods leverage the pre-trained knowledge of large `Visual-Language Models (VLMs)` or `Multimodal Large Language Models (MLLMs)` to generalize to the `CIR` task directly. This is crucial for real-world applications where collecting vast amounts of `CIR` triplet data is prohibitively expensive.

### 3.1.3. Visual-Language Models (VLMs)
`Visual-Language Models (VLMs)` are neural networks pre-trained on massive datasets of image-text pairs to understand and connect information across visual and textual modalities. These models learn a shared embedding space where images and texts with similar semantic content are mapped close to each other. A prominent example is `CLIP`.
*   **CLIP (Contrastive Language-Image Pre-training):** Developed by OpenAI, `CLIP` learns to associate images with text by predicting which text caption goes with which image. It consists of an `image encoder` and a `text encoder`. When an image and a text are encoded, their resulting feature vectors (embeddings) can be compared using `cosine similarity` to determine their semantic relatedness. This makes `CLIP` highly effective for `zero-shot` tasks like text-to-image retrieval.

### 3.1.4. Large Language Models (LLMs)
`Large Language Models (LLMs)` are advanced deep learning models, typically based on the Transformer architecture, that are trained on vast amounts of text data. They can understand, generate, and process human language with remarkable fluency and coherence, performing tasks like summarization, translation, question answering, and complex reasoning. Examples include `GPT` models.

### 3.1.5. Multimodal Large Language Models (MLLMs)
`Multimodal Large Language Models (MLLMs)` extend `LLMs` by integrating multiple modalities, most commonly vision, in addition to text. This allows them to process and generate responses based on both image inputs and text instructions. `MLLMs` can "see" an image and "reason" about its content in conjunction with textual prompts. `LLaVA`, `BLIP-2`, and `GPT-4.1` are examples of `MLLMs`.

### 3.1.6. Chain-of-Thought (CoT)
`Chain-of-Thought (CoT)` is a prompting technique used with `LLMs` and `MLLMs` that encourages the model to generate a series of intermediate reasoning steps before arriving at a final answer. Instead of directly asking for the answer, the prompt guides the model to "think step-by-step." This approach significantly improves the model's ability to tackle complex reasoning tasks, leading to more accurate and interpretable results. For example, instead of asking "What is 123 + 456?", a CoT prompt might ask "Let's break this down. First, add the ones digits, then the tens, then the hundreds...".

### 3.1.7. Cosine Similarity
`Cosine similarity` is a measure of similarity between two non-zero vectors in an inner product space. It is often used to measure how similar two documents or images are regardless of their size. The `cosine similarity` of two vectors $A$ and $B$ is calculated as the dot product of the two vectors divided by the product of their magnitudes:
\$
\mathrm{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}
\$
Where:
*   $A$ and $B$ are the feature vectors (embeddings).
*   $A_i$ and $B_i$ are components of vectors $A$ and $B$.
*   $\|A\|$ and $\|B\|$ are the magnitudes (L2 norms) of vectors $A$ and $B$.
    A cosine similarity of 1 means the vectors are perfectly aligned (most similar), 0 means they are orthogonal (no similarity), and -1 means they are diametrically opposed (most dissimilar). In `CIR`, it's used to rank candidate images based on their semantic similarity to the query features.

## 3.2. Previous Works
The paper categorizes `ZS-CIR` methods into `training-dependent` and `training-free`.

### 3.2.1. Training-Dependent ZS-CIR Methods
These methods still require additional training or fine-tuning on large-scale image-text pairs, even if they avoid `CIR`-specific triplet data.
*   **Textual Inversion-based methods (e.g., Pic2Word [35], SEARLE [2], MLLM-I2W [5]):** These methods learn to represent a `reference image` as `pseudo-tokens` or `pseudo-words` in the text embedding space. These `pseudo-tokens` are then combined with the `modification text` to form a new textual query. This allows for compositionality.
    *   **Pic2Word [35]:** Maps `reference image features` into `pseudo-tokens`.
    *   **SEARLE [2]:** Combines the `pseudo-word token` with a caption generated by `GPT`.
    *   **MLLM-I2W [5]:** Uses `MLLM contextual prompts` to map `reference image features` to `pseudo-word tokens`.
    *   **Limitation:** Despite leveraging `VLMs`, they still necessitate extra training steps, which can be computationally expensive and limit their true `zero-shot` applicability.

### 3.2.2. Training-Free ZS-CIR Methods
These methods directly use `LLMs` or `MLLMs` to generate the composed query, without any `CIR`-specific training. They generally consist of a `description generation process` and a `ranking process`.

#### 3.2.2.1. Two-Stage Methods
These methods first preprocess the `reference image` using a `captioning model` and then feed the generated caption along with the `modification text` into an `LLM` for `target image description` generation.
*   **CIReVL [18]:** Generates a `target image description` using `BLIP-2` [19] for captioning and an `LLM`.
*   **LDRE [47]:** Generates and aggregates multiple `target image descriptions` from an `LLM`.
*   **SEIZE [46]:** Also generates and aggregates multiple `target image descriptions` and introduces `semantic editing search` increments.
*   **Limitation:** The `reference image captioning` is done independently of the `modification text`. This means the caption might miss critical visual details relevant to the intended modification, leading to `LLMs` inferring inaccurate semantics and degrading retrieval performance.

#### 3.2.2.2. One-Stage Methods
These methods directly input both the `reference image` and the `modification text` into an `MLLM` through specially designed prompts. This ensures that visual details and textual intent are considered simultaneously.
*   **OSrCIR [41]:** Employs a `reflective CoT` strategy to help `MLLMs` understand and implement modifications, then performs direct search with the generated description.
*   **CoTMR [37]:** Leverages `CoT reasoning` to apply modifications step-by-step and performs `multi-scale inference` for retrieval.
*   **Common Limitations (addressed by SDR-CIR):**
    1.  **Visual Noise:** Existing one-stage `CoT`-based methods often extract nearly all visual information from the `reference image` before attempting to filter irrelevant details. This `extract-then-filter` approach, lacking `semantic selectivity`, can embed `visual noise` into the generated description.
    2.  **Semantic Bias:** Even with `MLLMs`, the `ZS-CIR` task involves `fuzzy matching`. The generated `target image description` may not perfectly align with the `target image`, leading to `redundancy bias` (irrelevant details from the reference) or `omission bias` (missing useful cues).

## 3.3. Technological Evolution
The evolution of `CIR` has seen a shift from `conventional training-dependent` methods that require specific triplet data to `zero-shot` approaches. Early `ZS-CIR` models often involved learning `pseudo-tokens` or `embeddings` for images, essentially trying to make images look like text (training-dependent `ZS-CIR`). The breakthrough of `large-scale pre-trained VLMs` like `CLIP` enabled true `zero-shot` capabilities by mapping images and text into a shared semantic space.

More recently, the advent of powerful `LLMs` and `MLLMs` has driven the `training-free ZS-CIR` paradigm. Initially, `two-stage` methods used separate `captioning models` and `LLMs`. However, the limitations of decoupled captioning led to `one-stage` methods, which directly feed multimodal inputs into `MLLMs`, often enhanced by `Chain-of-Thought (CoT)` prompting for improved reasoning. `SDR-CIR` fits into this `one-stage training-free ZS-CIR` category, building on `CoT` but specifically addressing its shortcomings related to `visual noise` and `semantic bias` in the generated descriptions and subsequent ranking.

## 3.4. Differentiation Analysis
`SDR-CIR` differentiates itself from previous `training-free ZS-CIR` methods, particularly one-stage `CoT`-based approaches, in two key areas:
*   **Selective Visual Extraction (Selective CoT vs. Standard CoT):**
    *   **Standard CoT (e.g., OSrCIR [41], CoTMR [37]):** These methods typically guide `MLLMs` to first extract *all* visual information from the `reference image` and then attempt to filter out irrelevant details based on the `modification text`. This `extract-then-filter` approach is prone to introducing `visual noise` because the initial extraction is broad and not focused on the modification intent.
    *   **SDR-CIR's Selective CoT:** Instead of a comprehensive extraction, `Selective CoT` actively guides the `MLLM` to *selectively* extract only visual content that is explicitly or implicitly relevant to the `modification text` during the `reference image understanding` phase. This `reasoning-control mechanism` reduces `visual noise` at the source, leading to a more semantically aligned `target image description`.
*   **Explicit Semantic Debias Ranking (SDR vs. Other Ranking):**
    *   **Most One-Stage Methods (e.g., OSrCIR [41]):** Primarily rely on directly using the generated description for retrieval via similarity ranking, which is vulnerable to `semantic bias`.
    *   **Some Advanced Two-Stage Methods (e.g., SEIZE [46]):** Address `semantic bias` by focusing on `semantic editing increments`, calculating the difference between initial and modified descriptions. However, `SEIZE` is a two-stage method and is also more computationally intensive (generating multiple descriptions).
    *   **SDR-CIR's Semantic Debias Ranking:** `SDR-CIR` explicitly models and corrects the `visual semantic contribution` of the `reference image` itself to the description, recognizing that this is the root cause of the `semantic bias` (both redundancy and omission). It employs a two-step `Anchor-then-Debias` strategy:
        1.  **Anchor:** Fuses `reference image features` with `target description features` to bolster useful semantics and recover omitted cues.
        2.  **Debias:** Penalizes candidates that show high similarity to the potentially biased `visual semantic contribution` of the `reference image` in the description. This approach directly tackles the `redundancy` and `omission` biases by selectively enhancing relevant information and suppressing irrelevant contributions from the reference image, leading to more accurate and balanced ranking. This is a direct and efficient way to mitigate `semantic bias` compared to more complex multi-description generation or editing increment strategies, especially within a one-stage framework.

# 4. Methodology

## 4.1. Principles
The core idea behind `SDR-CIR` is to systematically mitigate `semantic bias` in `training-free zero-shot Composed Image Retrieval (ZS-CIR)` by addressing it at two critical stages: `description generation` and `candidate ranking`. The framework operates on the principle that the `reference image`, while providing essential context, is also the primary source of `visual noise` and `semantic bias` in the generated `target image description`. Therefore, the method aims to enhance the relevance of generated descriptions by `selectively extracting` visual content and subsequently `debiasing` the ranking process by reinforcing useful semantics and penalizing biased contributions from the reference image.

The overall framework is illustrated in Figure 2, highlighting its two main components: `Selective CoT reasoning` and `Semantic Debias Ranking`.

The following figure (Figure 2 from the original paper) illustrates the overall framework.

![该图像是示意图，展示了SDR-CIR框架的两个主要步骤：选择性CoT推理和语义去偏排名。第一步通过多模态大语言模型从参考图像和修改文本中提取特征，以减少视觉噪声。第二步通过融合特征强化有用语义，同时使用$S=(1+β)Sq - βSi$的公式对相似性得分进行调整，以减少语义偏差。](images/2.jpg)
*该图像是示意图，展示了SDR-CIR框架的两个主要步骤：选择性CoT推理和语义去偏排名。第一步通过多模态大语言模型从参考图像和修改文本中提取特征，以减少视觉噪声。第二步通过融合特征强化有用语义，同时使用$S=(1+β)Sq - βSi$的公式对相似性得分进行调整，以减少语义偏差。*

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Selective CoT Reasoning
In `CIR`, the `reference image` often contains redundant visual information. Existing one-stage `Chain-of-Thought (CoT)` methods typically instruct `Multimodal Large Language Models (MLLMs)` to extract nearly all visual information and then filter it. This process can introduce `visual noise` into the generated description. `SDR-CIR` addresses this by introducing `Selective CoT`, which guides the `MLLM` to focus on visual content relevant to the `modification text` right from the initial image understanding stage.

The `Selective CoT` prompt maintains a four-stage structure similar to existing `CoT` prompts, but with a crucial modification in the first stage:
1.  **Reference Image Understanding (Key Difference):** Instead of extracting all visual details, the `MLLM` is first guided to parse the `modification text` to identify `explicit modified targets` (objects directly specified, e.g., "red car") and `implicit modified targets` (objects implied by the text, e.g., if "change color" is given, the main object is the implicit target). With these targets in mind, the `MLLM` then selectively extracts only the visual content from the `reference image` that is relevant to the `modification text`. This proactive filtering minimizes `visual noise` at the source. For instance, if the modification is about a "peacock on grass", the `MLLM` will focus on the peacock and the grass, largely ignoring irrelevant background details, as depicted in Figure 3.
2.  **Modification Text Understanding:** The `MLLM` then infers the precise modification intent from the `modification text`, breaking it down into specific steps (e.g., "change color to blue", "add a hat").
3.  **Applying Modification:** The `MLLM` applies these modification steps iteratively to the selectively extracted visual content, updating the visual description.
4.  **Target Image Description Generation:** Finally, the `MLLM` generates a coherent and concise `target image description` that accurately reflects all applied modifications and is as simple as possible, avoiding content that will not be present in the target image.

    The following figure (Figure 3 from the original paper) compares the `CoT` prompt between `OSrCIR` and `SDR-CIR`, illustrating how `SDR-CIR`'s `Selective CoT` focuses on relevant details.

    ![Figure 3: Comparison on CoT prompt between OSrCIR and ours.](images/3.jpg)
    *该图像是一个示意图，比较了OSrCIR和我们提出的选择性CoT提示。左侧为查询图像，描述一只孔雀旁边的两个人和环境信息；右侧为目标图像，显示孔雀在一辆车旁的草地上，且没有人出现。这种比较展示了我们方法在去除不相关信息方面的优势。*

The detailed `Selective CoT` prompt from the appendix is provided below. This prompt explicitly instructs the `MLLM` to identify targets, selectively extract relevant content, ignore irrelevant details, analyze modification steps, apply them, and then generate a concise description without extraneous reasoning output.

The following figure (Figure 6 from the original paper) shows our `Selective CoT` prompt.

![Figure 6: Our Selective CoT prompt.](images/6.jpg)
*该图像是示意图 (a) 和 (b)，展示了通过查询和描述进行图像检索的过程。在 (a) 中，查询描述了一只狗，目标图像显示了相关的狗照片；而 (b) 中，查询描述了冰箱的颜色，目标图像则是与描述匹配的冰箱。通过这些示例，展示了检索结果的多样性和相关性。*

### 4.2.2. Semantic Debias Ranking (SDR)
After generating the `target image description`, `SDR-CIR` introduces a `Semantic Debias Ranking` module to address the `semantic bias` that might still be present in the description due to the fuzzy matching nature of `ZS-CIR`. This bias can manifest as `redundancy` (irrelevant details from the reference image) or `omission` (missing useful cues). The `SDR` module consists of two steps: `Anchor` and `Debias`.

#### 4.2.2.1. Anchor: Reinforcing and Supplementing Information
The `Anchor` step aims to construct a more robust `composed query feature` by leveraging both the `reference image feature` and the `target image description feature`. This helps to reinforce useful visual semantics that might be implicitly present in the reference image but not fully captured in the description, and to supplement any key visual cues that were omitted from the description.

Specifically, given a generated `target image description` $T_d$, it is encoded into a feature vector $F_d$ using a `CLIP text encoder`. For any candidate image $I_t^i$ from the candidate image set $\mathcal{D}$, and the `reference image` $I_r$, they are encoded into feature vectors $F(I_t^i)$ and $F_r$ respectively, using a `CLIP image encoder`. All these features reside in the same `CLIP` embedding space.

The `composed query feature` $F_q$ is then obtained by fusing the `reference image feature` $F_r$ with the `target image description feature` $F_d$:
\$
F_q = (1 - \alpha) F_d + \alpha F_r
\$
Where:
*   $F_q$: The resulting `composed query feature` that serves as a semantic anchor.
*   $F_d$: The feature vector encoding the `target image description` generated by the `MLLM`.
*   $F_r$: The feature vector encoding the `reference image`.
*   $\alpha$: A tunable weighting factor (hyperparameter) that controls the contribution of the `reference image feature` to the `composed query feature`. A higher $\alpha$ means more emphasis on the reference image's visual content.

    The intuition behind this step is that:
*   For descriptions with `redundancy bias`, adding $F_r$ (which contains both useful and redundant information) helps to increase the relative weight of the useful information within the anchor. This prepares for the `Debias` step, preventing useful semantics from being overly weakened when the `visual semantic contribution` is penalized.
*   For descriptions with `omission bias`, $F_r$ can directly supplement those omitted key visual cues from the `reference image` that were not captured in $F_d$. This creates a more complete and robust query.

#### 4.2.2.2. Debias: Penalizing the Visual Semantic Contribution
The `Debias` step aims to explicitly reduce the impact of the `semantic bias` introduced by the `reference image` during the final similarity computation. Since the `reference image` is the primary source of this bias (either redundant details or omitted cues that don't match the target), its `visual semantic contribution` to the description can be modeled and used as a penalty.

First, `cosine similarities` are computed between the `composed query feature` $F_q$, the `target image description feature` $F_d$, and the `modification text feature` $F_m$ (where $F_m$ is obtained by encoding the `modification text` using the `CLIP text encoder`), and each candidate image feature $F(I_t^i)$ from the set $\mathcal{D}$:
\$
[S_q, S_d, S_m] = \mathrm{sim}([F_q, F_d, F_m], F(I_t^i)), \forall I_t^i \in \mathcal{D}
\$
Where:
*   $S_q$: The `cosine similarity` between the `composed query feature` $F_q$ and the candidate image feature $F(I_t^i)$.
*   $S_d$: The `cosine similarity` between the `target image description feature` $F_d$ and the candidate image feature $F(I_t^i)$.
*   $S_m$: The `cosine similarity` between the `modification text feature` $F_m$ and the candidate image feature $F(I_t^i)$.
*   $\mathrm{sim}(\cdot, \cdot)$: Denotes the `cosine similarity` function.
*   $F(I_t^i)$: The feature vector of a candidate image $I_t^i$.
*   $\mathcal{D}$: The set of all candidate images.

    The `visual semantic contribution` of the `reference image` to the `description` is approximated by calculating the difference between $S_d$ and $S_m$. The rationale is that the description ($T_d$) is a composition of the reference image and the modification text ($T_m$). If $S_m$ represents the part of the description's similarity that comes from the modification text (which is assumed to be correct), then $S_d - S_m$ can be seen as the part of the description's similarity that primarily comes from the reference image, which might be biased. This gives us $S_i$:
\$
S_i = S_d - S_m
\$
Where:
*   $S_i$: Represents the `cosine similarity` between the candidate image and the `visual semantic contribution` of the `reference image` to the `target description`.

    If $S_i$ is high for a candidate image, it means that candidate image is highly similar to the `visual semantic contribution` from the reference image that is present in the description. This `contribution` might contain redundant information (in cases of `redundancy bias`) or represent elements that were omitted but still influence the description's similarity (in cases of `omission bias`). Such a high $S_i$ value often indicates that the candidate image is a non-target image that accidentally aligns with the biased aspects of the reference image.

Therefore, $S_i$ is used as a penalty term in the final similarity score $S_f$:
\$
S_f = (1 + \beta) S_q - \beta S_i
\$
Where:
*   $S_f$: The final similarity score used for ranking.
*   $\beta$: A tunable weighting factor (hyperparameter) that controls the strength of the penalty for the `visual semantic contribution`. A higher $\beta$ means a stronger penalty.
*   The term $(1 + \beta)$ on $S_q$ is included to balance the score, ensuring that the primary composed query similarity ($S_q$) remains dominant while incorporating the debiasing effect.

    Finally, candidate images are ranked based on their calculated $S_f$ values, with higher scores indicating greater relevance to the desired target image. This `Debias` step explicitly penalizes non-target images that might otherwise score high due to spurious similarities with the reference-induced `semantic bias` in the description, thus improving retrieval accuracy.

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted on three widely recognized `Composed Image Retrieval (CIR)` datasets:

### 5.1.1. CIRR (Composed Image Retrieval on Real-world Images)
*   **Source:** Introduced by Liu et al. [25].
*   **Characteristics:** This is the first open-domain dataset for `CIR`. It contains 36,554 queries, where each query is associated with a single `target image`.
*   **Domain:** Emphasizes `object-level changes`, such such as adding, removing, or replacing objects within an image (e.g., "add a red ball", "remove the cat").
*   **Data Sample:** Not explicitly provided in the text, but a query would typically consist of a reference image (e.g., a room) and modification text (e.g., "add a bookshelf on the left"), with the target image being the room with the bookshelf.

### 5.1.2. CIRCO (Composed Image Retrieval on COCO)
*   **Source:** Built from real images in the COCO 2017 unlabeled set [23] by Baldrati et al. [2].
*   **Characteristics:** Includes 800 test queries and 220 validation queries. A key feature is that each query is associated with *multiple ground truths*, averaging 4.53 target images per query. This reflects real-world scenarios where multiple images might satisfy a query.
*   **Domain:** Similar to CIRR, it focuses on `object-level changes`.
*   **Data Sample:** Not explicitly provided, but examples in Figure 5 (a) show an image of a person next to a peacock and the modification "remove the two people and replace the peacock with a bigger one", implying the target image will have a larger peacock but no people.

### 5.1.3. FashionIQ (Fashion Image Query)
*   **Source:** Introduced by Wu et al. [45].
*   **Characteristics:** This dataset specifically targets the fashion domain. It covers three distinct clothing categories: `shirt`, `dress`, and `toptee`. It comprises 30,135 query triplets and a large pool of 77,683 candidate images. Each query is linked to a single `target image`.
*   **Domain:** Unlike CIRR and CIRCO, FashionIQ primarily emphasizes `attribute-level modifications` (e.g., "change color to blue", "make it a v-neck", "add stripes"). This requires finer-grained understanding of visual attributes.
*   **Data Sample:** Not explicitly provided, but examples in Figure 5 (b) show a reference image of a dress and modification text like "add a necklace, make it a full-length skirt", with the target image showing these attribute changes.

    These datasets were chosen because they are standard benchmarks in the `CIR` field, representing diverse `modification needs` (object-level vs. attribute-level) and different characteristics (single vs. multiple ground truths), making them effective for thoroughly validating the method's performance and generalization ability.

## 5.2. Evaluation Metrics
The paper employs standard evaluation metrics commonly used in image retrieval tasks.

### 5.2.1. Recall@k ($\mathrm{R@k}$)
*   **Conceptual Definition:** $\mathrm{Recall@k}$ measures the proportion of queries for which at least one of the ground-truth target images is present among the top $k$ retrieved results. It indicates how well the system can find *any* correct answer within a small set of top predictions.
*   **Mathematical Formula:**
    \$
    \mathrm{R@k} = \frac{\text{Number of queries with at least one ground truth in top k}}{\text{Total number of queries}}
    \$
    For a single query $q$:
    \$
    \mathrm{R@k}_q = \mathbb{I}(\exists g \in G_q \text{ s.t. } g \in R_{q,k})
    \$
    Then, the overall $\mathrm{R@k}$ is the average over all queries:
    \$
    \mathrm{R@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathrm{R@k}_q
    \$
*   **Symbol Explanation:**
    *   $\mathbb{I}(\cdot)$: Indicator function, which is 1 if the condition is true, and 0 otherwise.
    *   $G_q$: The set of all ground-truth target images for query $q$.
    *   $R_{q,k}$: The set of top $k$ retrieved images for query $q$.
    *   $g$: A ground-truth target image.
    *   $Q$: The set of all queries.
    *   $|Q|$: The total number of queries.

### 5.2.2. Mean Average Precision@k ($\mathrm{mAP@k}$)
*   **Conceptual Definition:** $\mathrm{mAP@k}$ is a widely used metric for ranking tasks, especially when there can be multiple relevant items for a query. It calculates the `Average Precision (AP)` for each query and then averages these `AP` values across all queries. `AP` itself calculates the weighted mean of precisions achieved at each threshold, where the weight is the increase in recall from the previous threshold. It gives a more holistic view of the ranking quality beyond just the presence of a target.
*   **Mathematical Formula:**
    First, `Precision@n` is defined as:
    \$
    \mathrm{Precision@n} = \frac{\text{Number of relevant items in top n}}{\text{n}}
    \$
    Then, `Average Precision (AP)` for a single query $q$:
    \$
    \mathrm{AP}_q = \sum_{n=1}^k (\mathrm{Precision@n} \times \Delta \mathrm{Recall@n})
    \$
    where $\Delta \mathrm{Recall@n} = \mathrm{Recall@n} - \mathrm{Recall@n-1}$. This sum is only over ranks where a relevant document is retrieved.
    The overall $\mathrm{mAP@k}$ is the average of $\mathrm{AP}_q$ over all queries:
    \$
    \mathrm{mAP@k} = \frac{1}{|Q|} \sum_{q \in Q} \mathrm{AP}_q
    \$
*   **Symbol Explanation:**
    *   $\mathrm{Precision@n}$: The precision at rank $n$.
    *   $\Delta \mathrm{Recall@n}$: The change in recall from rank `n-1` to rank $n$.
    *   $k$: The maximum rank considered (e.g., 5, 10, 25, 50).
    *   $Q$: The set of all queries.
    *   $|Q|$: The total number of queries.

### 5.2.3. Recallsub@k ($\mathrm{Recall_{sub@k}}$)
*   **Conceptual Definition:** This is a specialized `Recall@k` metric used specifically on a subset of the CIRR dataset. This subset contains samples that are `highly similar` to the target image. Evaluating on this subset provides a more rigorous assessment of the model's ability to discriminate subtle differences and find the *most* accurate target when multiple similar options exist, indicating performance on `finer discrimination` tasks.
*   **Mathematical Formula:** The formula is identical to $\mathrm{Recall@k}$, but the calculation is performed only on the designated subset of queries from CIRR.
*   **Symbol Explanation:** Same as $\mathrm{Recall@k}$, but applied to a specific subset of the dataset.

## 5.3. Baselines
The paper compares `SDR-CIR` against a range of `ZS-CIR` methods, broadly categorized as `training-dependent` and `training-free`.

### 5.3.1. Training-Dependent ZS-CIR Methods
*   **Pic2Word [35]:** Maps the `reference image feature` into `pseudo-tokens` for retrieval.
*   **SEARLE [2]:** Combines a `pseudo-word token` (learned from the image) with a caption produced by `GPT` for composed queries.
*   **MLLM-I2W [5]:** Uses `MLLM contextual prompts` to map `reference image features` to `pseudo-word tokens`.

### 5.3.2. Training-Free ZS-CIR Methods
#### 5.3.2.1. Two-Stage Methods
These methods first caption the reference image and then use an `LLM` to compose the caption with the modification text.
*   **CIReVL [18]:** Generates a `target image description` using a two-stage process involving `BLIP-2` [19] for captioning and an `LLM` for composition.
*   **LDRE [47]:** Generates and aggregates multiple `target image descriptions` from an `LLM` to improve robustness.
*   **SEIZE [46]:** Generates and aggregates multiple `target image descriptions` and introduces `semantic editing search` to capture modification increments.

#### 5.3.2.2. One-Stage Methods
These methods directly use an `MLLM` to handle both the `reference image` and `modification text` simultaneously.
*   **OSrCIR [41]:** Generates a `target image description` using an `MLLM` based on `Chain-of-Thought (CoT)` and performs direct search.
*   **CoTMR [37]:** Employs `CoT reasoning` within an `MLLM` to apply modifications step-by-step and performs `multi-scale inference` for retrieval.

## 5.4. Implementation Details
*   **Main MLLM:** `GPT-4.1` [31] is used for generating `target image descriptions`. For ablation analysis, `Qwen2.5-VL-72B` [1] and `GPT-4o-mini` [30] are also evaluated.
*   **Retrieval Encoder:** Pre-trained `CLIP` models from `OpenCLIP` [15] are used as the backbone for encoding images and text into feature vectors. Three variants are employed: `ViT-B/32`, `ViT-L/14`, and `ViT-G/14`.
*   **Scaling Factors (Hyperparameters):**
    *   $\alpha$ (weight for `reference image feature` in `Anchor` step):
        *   CIRR: 0.05
        *   CIRCO: 0.15
        *   FashionIQ: 0.2
    *   $\beta$ (weight for `visual semantic contribution` penalty in `Debias` step):
        *   CIRR: 0.5
        *   CIRCO: 0.35
        *   FashionIQ: 0.4
*   **Computational Environment:** All experiments are implemented in `PyTorch` [32] and run on a single `NVIDIA RTX 3090 GPU`.
*   **CoT Context Examples:** For a fair comparison with baselines, `SDR-CIR` typically does not use in-context examples in the `CoT` prompt unless specified otherwise (implied by the asterisk `*` for baselines in tables).

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results demonstrate `SDR-CIR`'s effectiveness across three standard `CIR` benchmarks, often achieving state-of-the-art performance among one-stage methods.

### 6.1.1. CIRCO and CIRR Benchmarks
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<td colspan="3" rowspan="2">CIRCO+CIRR→</td>
<td colspan="4" colspan="1">CIRCO</td>
<td colspan="8" colspan="1">CIRR</td>
</tr>
<tr>
<td colspan="1" rowspan="1"></td>
<td colspan="3" colspan="1">mAP@k</td>
<td colspan="1" rowspan="1"></td>
<td colspan="3" colspan="1">Recall@k</td>
<td colspan="3" colspan="1">Recallsub@k</td>
</tr>
<tr>
<td rowspan="2">Backbone</td>
<td rowspan="2">Method</td>
<td rowspan="2">Training-free</td>
<td rowspan="2">Type</td>
<td>k=5</td>
<td>k=10</td>
<td>k=25</td>
<td>k=50</td>
<td>k=1</td>
<td>k=5</td>
<td>k=10</td>
<td>k=50</td>
<td>k=1</td>
<td>k=2</td>
<td>k=3</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">ViT-B/32</td>
<td>SEARLE [2]</td>
<td>×</td>
<td></td>
<td>9.35</td>
<td>9.94</td>
<td>11.13</td>
<td>11.84</td>
<td>24.00</td>
<td>53.42</td>
<td>66.82</td>
<td>89.78</td>
<td>54.89</td>
<td>76.60</td>
<td>88.19</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>14.95</td>
<td>15.42</td>
<td>17.00</td>
<td>17.82</td>
<td>23.94</td>
<td>52.51</td>
<td>66.00</td>
<td>86.95</td>
<td>60.17</td>
<td>80.05</td>
<td>90.19</td>
</tr>
<tr>
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
<td>17.96</td>
<td>18.32</td>
<td>20.21</td>
<td>21.11</td>
<td>25.69</td>
<td>55.13</td>
<td>69.04</td>
<td>89.90</td>
<td>60.53</td>
<td>80.65</td>
<td>90.70</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
<td>19.04</td>
<td>19.64</td>
<td>21.55</td>
<td>22.49</td>
<td>27.47</td>
<td>57.42</td>
<td>70.17</td>
<td>-</td>
<td>65.59</td>
<td>84.48</td>
<td>92.77</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td></td>
<td>1S</td>
<td>16.85</td>
<td>17.39</td>
<td>19.15</td>
<td>20.01</td>
<td>28.07</td>
<td>57.95</td>
<td>69.71</td>
<td>88.94</td>
<td>62.31</td>
<td>81.18</td>
<td>91.04</td>
</tr>
<tr>
<td>CoTR*[37]</td>
<td>√</td>
<td>1S</td>
<td>21.16</td>
<td>21.77</td>
<td>23.71</td>
<td>24.70</td>
<td>30.12</td>
<td>60.19</td>
<td>71.71</td>
<td>90.34</td>
<td>67.11</td>
<td>85.13</td>
<td>93.64</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>23.78</td>
<td>24.43</td>
<td>26.58</td>
<td>27.50</td>
<td>34.48</td>
<td>65.74</td>
<td>76.87</td>
<td>93.06</td>
<td>69.90</td>
<td>87.04</td>
<td>94.48</td>
</tr>
<tr>
<td rowspan="7">ViT-L/14</td>
<td>Pic2Word [35]</td>
<td>×</td>
<td>-</td>
<td>8.72</td>
<td>9.51</td>
<td>10.64</td>
<td>11.29</td>
<td>23.90</td>
<td>51.70</td>
<td>65.30</td>
<td>87.80</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>SEARLE [2]</td>
<td>×</td>
<td>-</td>
<td>11.68</td>
<td>12.73</td>
<td>14.33</td>
<td>15.12</td>
<td>24.24</td>
<td>52.48</td>
<td>66.29</td>
<td>88.84</td>
<td>53.76</td>
<td>75.01</td>
<td>88.19</td>
</tr>
<tr>
<td>MLLM-I2W [5]</td>
<td>×</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>28.30</td>
<td>57.90</td>
<td>70.20</td>
<td>93.90</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>18.57</td>
<td>19.01</td>
<td>20.89</td>
<td>21.80</td>
<td>24.55</td>
<td>52.31</td>
<td>64.92</td>
<td>86.34</td>
<td>59.54</td>
<td>79.88</td>
<td>89.69</td>
</tr>
<tr>
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
<td>23.25</td>
<td>24.03</td>
<td>26.44</td>
<td>27.50</td>
<td>26.53</td>
<td>55.57</td>
<td>67.54</td>
<td>88.50</td>
<td>60.43</td>
<td>80.31</td>
<td>89.90</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
<td>24.98</td>
<td>25.82</td>
<td>28.24</td>
<td>29.35</td>
<td>28.65</td>
<td>57.16</td>
<td>69.23</td>
<td>-</td>
<td>66.22</td>
<td>84.05</td>
<td>92.34</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>21.83</td>
<td>22.46</td>
<td>24.49</td>
<td>25.44</td>
<td>30.63</td>
<td>60.34</td>
<td>72.00</td>
<td>89.86</td>
<td>64.31</td>
<td>82.29</td>
<td>91.35</td>
</tr>
<tr>
<td rowspan="7">ViT-G/14</td>
<td>CoTR*[37]</td>
<td>√</td>
<td>1S</td>
<td>26.52</td>
<td>27.13</td>
<td>29.51</td>
<td>30.56</td>
<td>33.54</td>
<td>63.25</td>
<td>74.63</td>
<td>91.08</td>
<td>69.88</td>
<td>86.53</td>
<td>94.19</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>35.08</td>
<td>37.61</td>
<td>67.71</td>
<td>79.13</td>
<td>93.81</td>
<td>71.90</td>
<td>88.39</td>
<td>94.63</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>26.77</td>
<td>27.59</td>
<td>29.96</td>
<td>31.03</td>
<td>34.65</td>
<td>64.29</td>
<td>75.06</td>
<td>91.66</td>
<td>67.95</td>
<td>84.87</td>
<td>93.21</td>
</tr>
<tr>
<td>LDRE [47]</td>
<td>v</td>
<td>2S</td>
<td>31.12</td>
<td>32.24</td>
<td>34.95</td>
<td>36.03</td>
<td>36.15</td>
<td>66.39</td>
<td>77.25</td>
<td>93.95</td>
<td>68.82</td>
<td>85.66</td>
<td>93.76</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
<td>32.46</td>
<td>33.77</td>
<td>36.46</td>
<td>37.55</td>
<td>38.87</td>
<td>69.42</td>
<td>79.42</td>
<td>-</td>
<td>74.15</td>
<td>89.23</td>
<td>95.71</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>24.73</td>
<td>25.99</td>
<td>28.09</td>
<td>29.19</td>
<td>33.52</td>
<td>61.66</td>
<td>73.30</td>
<td>90.17</td>
<td>65.42</td>
<td>82.75</td>
<td>91.54</td>
</tr>
<tr>
<td>CoTR*[37]</td>
<td>√</td>
<td>1S</td>
<td>29.59</td>
<td>30.74</td>
<td>33.37</td>
<td>34.44</td>
<td>35.93</td>
<td>65.11</td>
<td>75.33</td>
<td>91.45</td>
<td>70.82</td>
<td>87.16</td>
<td>94.48</td>
</tr>
<tr>
<td colspan="4">SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>33.05</td>
<td>34.50</td>
<td>37.21</td>
<td>38.42</td>
<td>40.17</td>
<td>69.76</td>
<td>79.88</td>
<td>94.00</td>
<td>73.30</td>
<td>88.89</td>
<td>94.99</td>
</tr>
</tbody>
</table>

*   **CIRCO (Multiple Ground Truths):** On the CIRCO test dataset, which features multiple ground-truth images per query, `SDR-CIR` consistently achieves the best `mAP@k` scores across all $k$ values and `CLIP` backbones (e.g., `ViT-L/14`: `mAP@5` of 30.91%). This is particularly important because an overly precise description might only match a single target, reducing coverage. `SDR-CIR`'s `semantic debiasing` helps suppress `reference-induced irrelevant details`, broadening the retrieval scope to cover more valid targets. For example, with the `ViT-L/14` backbone, `SDR-CIR` improves `mAP@5` by 9.08% over `OSrCIR` and 4.39% over `CoTMR`.
*   **CIRR (Single Ground Truths, Fine-grained Subsets):** On the CIRR test dataset, `SDR-CIR` also demonstrates superior performance. For instance, with `ViT-L/14`, `SDR-CIR` achieves a `Recall@1` of 37.61%, improving by 4.07% compared to the best training-free method, `CoTMR`. CIRR's `false negatives` and `modification text dominance` make `redundant reference image content` particularly detrimental, which `SDR-CIR` effectively suppresses.
    *   On the `CIRR subset` (Recallsub@k), which demands finer discrimination for highly similar images, `SDR-CIR` again outperforms all one-stage baselines. Using `ViT-G/14`, `SDR-CIR` achieves a `Recallsub@1` of 73.30%, a 2.48% improvement over `CoTMR`. However, it still lags slightly behind the two-stage method `SEIZE` (e.g., `ViT-G/14` `Recallsub@1` of 74.15%). The authors attribute this gap to `SEIZE`'s reliance on `semantic editing increments` which might provide stronger cues for highly similar images, whereas `SDR-CIR` primarily focuses on suppressing `reference image redundancy`.

### 6.1.2. FashionIQ Benchmark
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<td colspan="4" rowspan="2">FashionIQ→</td>
<td colspan="2">Shirt</td>
<td colspan="2">Dress</td>
<td colspan="2">Toptee</td>
<td colspan="2">Average</td>
</tr>
<tr>
<td>R@10</td>
<td>R@50</td>
<td>R@10</td>
<td>R@50</td>
<td>R@10</td>
<td>R@50</td>
<td>R@10</td>
<td>R@50</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">ViT-B/32</td>
<td>SEARLE [2]</td>
<td>×</td>
<td>-</td>
<td>24.44</td>
<td>41.61</td>
<td>18.54</td>
<td>39.51</td>
<td>25.70</td>
<td>46.46</td>
<td>22.89</td>
<td>42.53</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>28.36</td>
<td>47.84</td>
<td>25.29</td>
<td>46.36</td>
<td>31.21</td>
<td>53.85</td>
<td>28.29</td>
<td>49.35</td>
</tr>
<tr>
<td>LDRE [47]</td>
<td>v</td>
<td>2S</td>
<td>27.38</td>
<td>46.27</td>
<td>19.97</td>
<td>41.84</td>
<td>27.07</td>
<td>48.78</td>
<td>24.81</td>
<td>45.63</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>v</td>
<td>2S</td>
<td>29.38</td>
<td>47.97</td>
<td>25.37</td>
<td>46.84</td>
<td>32.07</td>
<td>54.78</td>
<td>28.94</td>
<td>49.86</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td></td>
<td>1S</td>
<td>31.31</td>
<td>50.74</td>
<td>27.07</td>
<td>48.83</td>
<td>33.71</td>
<td>55.23</td>
<td>30.70</td>
<td>51.60</td>
</tr>
<tr>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>30.77</td>
<td>50.54</td>
<td>29.45</td>
<td>50.92</td>
<td>34.93</td>
<td>57.57</td>
<td>31.72</td>
<td>53.01</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>36.41</td>
<td>57.02</td>
<td>36.84</td>
<td>58.85</td>
<td>43.14</td>
<td>64.71</td>
<td>38.80</td>
<td>60.19</td>
</tr>
<tr>
<td rowspan="9">ViT-L/14</td>
<td>Pic2Word [35]</td>
<td>×</td>
<td>-</td>
<td>26.20</td>
<td>43.60</td>
<td>20.00</td>
<td>40.20</td>
<td>27.90</td>
<td>47.40</td>
<td>24.70</td>
<td>43.70</td>
</tr>
<tr>
<td>SEARLE [2]</td>
<td>×</td>
<td>-</td>
<td>26.89</td>
<td>45.58</td>
<td>20.48</td>
<td>43.13</td>
<td>29.32</td>
<td>49.97</td>
<td>25.56</td>
<td>46.23</td>
</tr>
<tr>
<td>MLLM-I2W [5]</td>
<td>×</td>
<td>-</td>
<td>27.30</td>
<td>46.50</td>
<td>29.90</td>
<td>48.60</td>
<td>33.80</td>
<td>55.20</td>
<td>30.30</td>
<td>50.10</td>
</tr>
<tr>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>29.49</td>
<td>47.40</td>
<td>24.79</td>
<td>44.76</td>
<td>31.36</td>
<td>53.65</td>
<td>28.55</td>
<td>48.57</td>
</tr>
<tr>
<td>LDRE [47]</td>
<td>√</td>
<td>2S</td>
<td>31.04</td>
<td>51.22</td>
<td>22.93</td>
<td>46.76</td>
<td>31.57</td>
<td>53.64</td>
<td>28.51</td>
<td>50.54</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>v</td>
<td>2S</td>
<td>33.04</td>
<td>53.22</td>
<td>30.93</td>
<td>50.76</td>
<td>35.57</td>
<td>58.64</td>
<td>33.18</td>
<td>54.21</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>34.00</td>
<td>51.86</td>
<td>27.57</td>
<td>48.34</td>
<td>32.94</td>
<td>54.46</td>
<td>31.50</td>
<td>51.55</td>
</tr>
<tr>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>33.42</td>
<td>52.31</td>
<td>29.70</td>
<td>50.42</td>
<td>34.73</td>
<td>57.47</td>
<td>32.62</td>
<td>53.40</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>41.02</td>
<td>59.27</td>
<td>37.04</td>
<td>59.15</td>
<td>44.47</td>
<td>65.32</td>
<td>40.84</td>
<td>61.25</td>
</tr>
<tr>
<td rowspan="6">ViT-G/14</td>
<td>CIReVL [18]</td>
<td>√</td>
<td>2S</td>
<td>33.71</td>
<td>51.42</td>
<td>27.07</td>
<td>49.53</td>
<td>35.80</td>
<td>56.14</td>
<td>32.19</td>
<td>52.36</td>
</tr>
<tr>
<td>LDRE [47]</td>
<td>v</td>
<td>2S</td>
<td>35.94</td>
<td>58.58</td>
<td>26.11</td>
<td>51.12</td>
<td>35.42</td>
<td>56.67</td>
<td>32.49</td>
<td>55.46</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>√</td>
<td>2S</td>
<td>43.60</td>
<td>65.42</td>
<td>39.61</td>
<td>61.02</td>
<td>45.94</td>
<td>71.12</td>
<td>43.05</td>
<td>65.85</td>
</tr>
<tr>
<td>OSrCIR* [41]</td>
<td>√</td>
<td>1S</td>
<td>35.03</td>
<td>54.02</td>
<td>30.94</td>
<td>53.50</td>
<td>36.66</td>
<td>58.29</td>
<td>34.21</td>
<td>55.27</td>
</tr>
<tr>
<td>CoTMR* [37]</td>
<td>√</td>
<td>1S</td>
<td>35.03</td>
<td>53.39</td>
<td>32.82</td>
<td>55.48</td>
<td>37.89</td>
<td>59.46</td>
<td>35.25</td>
<td>56.11</td>
</tr>
<tr>
<td>SDR-CIR</td>
<td>√</td>
<td>1S</td>
<td>44.55</td>
<td>62.37</td>
<td>42.74</td>
<td>63.41</td>
<td>48.29</td>
<td>69.71</td>
<td>45.19</td>
<td>65.16</td>
</tr>
</tbody>
</table>

*   **FashionIQ (Attribute Modifications):** `SDR-CIR` achieves the best average `R@10` and `R@50` performance across the `ViT-B/32` and `ViT-L/14` backbones, outperforming both `textual inversion` and `training-free` methods. For instance, with `ViT-L/14`, `SDR-CIR`'s average `R@10` is 40.84%, an 8.22% improvement over the previous state-of-the-art method, `CoTMR` (32.62%). This demonstrates `SDR-CIR`'s strong capability not only in `object-level modifications` but also in `attribute-level modifications`.
*   **Larger Backbones:** On `ViT-G/14`, `SDR-CIR`'s average `R@50` (65.16%) is slightly lower than `SEIZE` (65.85%). This suggests that larger backbones might further amplify the benefits of `semantic editing increments` used in `SEIZE` for very fine-grained distinctions. However, `SDR-CIR` is noted to be substantially more efficient (see efficiency analysis below).

    In summary, these results highlight `SDR-CIR`'s strong generalization ability across different backbones and datasets, validating its effectiveness in mitigating `semantic bias` while maintaining high retrieval accuracy across various `CIR` challenges.

## 6.2. Ablation Studies / Parameter Analysis

### 6.2.1. Ablation Study on Key Components
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<td rowspan="2">Method</td>
<td colspan="3">CIRCO</td>
<td colspan="2">FashionIQ</td>
</tr>
<tr>
<td>k=5</td>
<td>k=10</td>
<td>k=25</td>
<td>k=10</td>
<td>k=50</td>
</tr>
</thead>
<tbody>
<tr>
<td>SDR-CIR</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>40.84</td>
<td>61.25</td>
</tr>
<tr>
<td colspan="6">1. Chain-of-Thought</td>
</tr>
<tr>
<td>w/o CoT</td>
<td>24.82</td>
<td>25.61</td>
<td>27.79</td>
<td>38.57</td>
<td>59.26</td>
</tr>
<tr>
<td>w/o Selective extraction</td>
<td>30.65</td>
<td>31.42</td>
<td>33.82</td>
<td>40.51</td>
<td>60.15</td>
</tr>
<tr>
<td colspan="6">2. Key modules of SDR-CIR</td>
</tr>
<tr>
<td>only description</td>
<td>26.61</td>
<td>27.54</td>
<td>30.01</td>
<td>32.26</td>
<td>52.61</td>
</tr>
<tr>
<td>+Anchor</td>
<td>28.56</td>
<td>29.48</td>
<td>31.39</td>
<td>38.41</td>
<td>59.01</td>
</tr>
<tr>
<td>+Debias</td>
<td>26.88</td>
<td>27.86</td>
<td>30.38</td>
<td>33.63</td>
<td>54.43</td>
</tr>
<tr>
<td>+SDR</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>40.84</td>
<td>61.25</td>
</tr>
<tr>
<td colspan="6">3. Different MLLMs</td>
</tr>
<tr>
<td>Qwen2.5-VL-72B</td>
<td>27.82</td>
<td>28.46</td>
<td>30.80</td>
<td>38.71</td>
<td>58.60</td>
</tr>
<tr>
<td>GPT-4omini</td>
<td>30.67</td>
<td>31.33</td>
<td>33.66</td>
<td>39.26</td>
<td>59.85</td>
</tr>
<tr>
<td>GPT-4.1</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>40.84</td>
<td>61.25</td>
</tr>
</tbody>
</table>

The ablation studies, conducted on CIRCO and FashionIQ with a `ViT-L/14` backbone, confirm the individual contributions of `Selective CoT` and the `Semantic Debias Ranking` (SDR) module.

#### 6.2.1.1. Effect of Selective CoT Prompt
*   **`w/o CoT` (without `CoT` reasoning):** Removing `CoT` entirely leads to a significant performance drop. On CIRCO, `mAP@5` decreases by 6.09% (from 30.91% to 24.82%). On FashionIQ, average `R@10` drops by 2.27% (from 40.84% to 38.57%). This highlights the crucial role of `CoT` reasoning in effectively understanding both the `reference image` and `modification text`. The larger drop on CIRCO suggests `CoT` is more impactful for complex images, where intricate reasoning is needed.
*   **`w/o Selective extraction` (using comprehensive image understanding in CoT instead of selective):** When `Selective CoT` is replaced with a comprehensive image understanding (as in typical `CoT` methods that extract all info), `mAP@5` on CIRCO drops by 0.26% (from 30.91% to 30.65%) and average `R@10` on FashionIQ drops by 0.33% (from 40.84% to 40.51%). These consistent, albeit small, declines demonstrate that `Selective CoT` effectively reduces `visual noise` at the source, confirming its benefit over simply extracting all content.

#### 6.2.1.2. Effect of Key Steps in SDR
*   **`only description` (Baseline: direct retrieval using only generated descriptions):** This represents a baseline where no `SDR` components are applied. Performance is notably lower (e.g., CIRCO `mAP@5` at 26.61%, FashionIQ `R@10` at 32.26%).
*   **$+Anchor$ (Adding `Anchor` step to `only description`):** Introducing the `Anchor` step significantly improves performance. CIRCO `mAP@5` increases by 1.95% (from 26.61% to 28.56%), and FashionIQ average `R@10` improves by 6.15% (from 32.26% to 38.41%). This validates the `Anchor` step's ability to stabilize useful `visual semantics` and recover `omitted cues` by fusing `reference image features`. The larger gain on FashionIQ implies that `omission bias` might be more prevalent in this dataset, making the integration of image features particularly beneficial.
*   **$+Debias$ (Adding `Debias` step to `only description` without `Anchor`):** When `Debias` is applied without the preceding `Anchor` step, improvements are slight (CIRCO `mAP@5` increases by 0.27%, FashionIQ `R@10` by 1.37%). This indicates that directly penalizing `visual semantic contribution` without first reinforcing useful semantics (via `Anchor`) can potentially weaken valuable information, limiting the overall benefit.
*   **$+SDR$ (Full `SDR-CIR`, including both `Anchor` and `Debias`):** The full `SDR-CIR` (which is the same as `SDR-CIR` in the table) achieves the highest performance. Compared to `only description`, CIRCO `mAP@5` improves by 4.30% (from 26.61% to 30.91%), and FashionIQ `R@10` by 8.58% (from 32.26% to 40.84%). More importantly, compared to $+Anchor$ alone, both datasets show further increases (CIRCO `mAP@5` by 2.35%, FashionIQ `R@10` by 2.43%). This confirms that the `Debias` step is most effective when useful semantics have already been anchored, demonstrating the complementary nature of the two steps within the `SDR` module.

#### 6.2.1.3. Comparison of Different MLLMs
*   **GPT-4.1:** Achieves the best results (CIRCO `mAP@5` 30.91%, FashionIQ `R@10` 40.84%).
*   **Qwen2.5-VL-72B:** Performance is lower than `GPT-4.1` by 3.09% on CIRCO and 2.13% on FashionIQ.
*   **GPT-4o-mini:** Highly comparable to `GPT-4.1`, with only 0.24% lower on CIRCO and 1.58% lower on FashionIQ, while being more efficient.
    These results demonstrate that `SDR-CIR` is robust across different `MLLMs`, with performance scaling with the capability of the underlying `MLLM`.

### 6.2.2. Impact of SDR on Different One-Stage Methods
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<td rowspan="2">Method</td>
<td rowspan="2">SDR</td>
<td colspan="4">CIRCO (mAP@k)</td>
</tr>
<tr>
<td>k=5</td>
<td>k=10</td>
<td>k=25</td>
<td>k=50</td>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">OSrCIR [41]</td>
<td>×</td>
<td>21.83</td>
<td>22.46</td>
<td>24.49</td>
<td>25.44</td>
</tr>
<tr>
<td>√</td>
<td>25.86</td>
<td>26.56</td>
<td>28.73</td>
<td>29.77</td>
</tr>
<tr>
<td>∆</td>
<td>+4.03</td>
<td>+4.10</td>
<td>+4.24</td>
<td>+4.33</td>
</tr>
<tr>
<td rowspan="3">CoTMR [37]</td>
<td>×</td>
<td>26.52</td>
<td>27.13</td>
<td>29.51</td>
<td>30.56</td>
</tr>
<tr>
<td>√</td>
<td>30.12</td>
<td>30.74</td>
<td>33.20</td>
<td>34.28</td>
</tr>
<tr>
<td>∆</td>
<td>+3.60</td>
<td>+3.61</td>
<td>+3.69</td>
<td>+3.72</td>
</tr>
<tr>
<td rowspan="3">Ours</td>
<td>×</td>
<td>26.61</td>
<td>27.54</td>
<td>30.01</td>
<td>31.14</td>
</tr>
<tr>
<td>√</td>
<td>30.91</td>
<td>31.50</td>
<td>34.03</td>
<td>35.08</td>
</tr>
<tr>
<td>∆</td>
<td>+4.30</td>
<td>+3.96</td>
<td>+4.02</td>
<td>+3.94</td>
</tr>
</tbody>
</table>

This analysis shows that the `Semantic Debias Ranking` (SDR) module is a plug-and-play component that can effectively enhance other one-stage `ZS-CIR` methods. When `SDR` is applied to `OSrCIR` and `CoTMR` (using `ViT-L/14` and no in-context examples), both baselines achieve substantial improvements: `OSrCIR` gains 4.03% in `mAP@5`, and `CoTMR` gains 3.60%. This confirms that `SDR` successfully mitigates the `semantic bias` that `MLLMs` alone struggle to resolve, making it a valuable addition to existing one-stage frameworks.

### 6.2.3. Effect of Hyperparameters $\alpha$ and $\beta$
The following figure (Figure 4 from the original paper) presents the hyperparameter analysis.

![Figure 4: Hyperparameter analysis of $\\alpha$ and $\\beta$ on CIRCO test set, CIRR test set and FashionIQ val set. All experiments are performed with the ViT-L/14.](images/4.jpg)
*该图像是图表，展示了超参数$\alpha$和$\beta$在CIRCO测试集、CIRR测试集和FashionIQ验证集上的分析结果。左侧（a）显示了不同$\alpha$值对各指标的影响，右侧（b）则展示了不同$\beta$值的效果。所有实验均使用ViT-L/14模型进行。图中不同颜色的曲线代表了不同的评估指标。*

*   **Hyperparameter $\alpha$ (Weight for Reference Image Feature):** As shown in Figure 4(a), increasing $\alpha$ generally improves metrics up to a moderate range, after which performance can degrade. For FashionIQ, performance continues to improve and stabilizes around $\alpha = 0.2$. However, CIRR's $\mathrm{R_{sub@1}}$ (a metric for highly similar images) consistently shows a downward trend with increasing $\alpha$. This suggests that an excessively large $\alpha$ might amplify `reference image noise` (especially in subtle discrimination tasks), weakening retrieval accuracy for the `subset`.
*   **Hyperparameter $\beta$ (Weight for Debias Penalty):** As shown in Figure 4(b), increasing $\beta$ generally improves overall performance across all datasets. CIRCO and FashionIQ reach their peaks at $\beta = 0.35$ and $\beta = 0.4$, respectively. CIRR's $\mathrm{R@1}$ and $\mathrm{R_{sub@1}}$ are optimal at $\beta = 0.5$. These trends indicate that a moderate to strong penalty on the `visual semantic contribution` of the `reference image` is beneficial for debiasing and improving retrieval.

### 6.2.4. Efficiency Analysis
The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<td rowspan="2">Method</td>
<td colspan="2">Accuracy metic</td>
<td colspan="2">Efficiency metic (per query)</td>
</tr>
<tr>
<td>CIRCO (k=1)</td>
<td>FashionIQ (k=10)</td>
<td>Infer time (s)</td>
<td>Calls (times)</td>
</tr>
</thead>
<tbody>
<tr>
<td>LDRE [47]</td>
<td>31.12</td>
<td>32.49</td>
<td>19.24</td>
<td>15</td>
</tr>
<tr>
<td>SEIZE [46]</td>
<td>32.46</td>
<td>43.05</td>
<td>19.24</td>
<td>15</td>
</tr>
<tr>
<td>OSrCIR [41]</td>
<td>24.73</td>
<td>34.21</td>
<td>1.01</td>
<td>1</td>
</tr>
<tr>
<td>CoTMR [37]</td>
<td>29.59</td>
<td>35.25</td>
<td>1.48</td>
<td>2</td>
</tr>
<tr>
<td>Ours</td>
<td>33.05</td>
<td>45.19</td>
<td>0.37</td>
<td>1</td>
</tr>
</tbody>
</table>

Table 5 compares the effectiveness (accuracy) and efficiency of various training-free methods using the `ViT-G/14` backbone.
*   **Accuracy:** `SDR-CIR` achieves the highest accuracy among all listed training-free methods (CIRCO `mAP@1` of 33.05%, FashionIQ `R@10` of 45.19%).
*   **Efficiency:** `SDR-CIR` is remarkably efficient, requiring only 0.37 seconds per query and making just 1 `MLLM` call. In contrast, two-stage methods like `LDRE` and `SEIZE`, despite having comparable or slightly lower accuracy, take significantly longer (19.24 seconds per query) and require 15 `MLLM` calls. Even other one-stage methods like `OSrCIR` (1.01s, 1 call) and `CoTMR` (1.48s, 2 calls) are less efficient.
    This analysis underscores that `SDR-CIR` not only achieves state-of-the-art accuracy but also does so with superior efficiency, making it highly practical.

## 6.3. Qualitative Results
The following figure (Figure 5 from the original paper) illustrates the robustness of `SDR-CIR` to redundant and missing information.

![Figure 5: Robustness to redundant and missing information on CIRCO and FashionIQ. Top-2 retrieval results of SDR-CIR and a description-only baseline (Base Result) are compared. Red text marks redundant or missing information; green boxes indicate targets.](images/5.jpg)
*该图像是检索示例，展示了SDR-CIR与描述基线在CIRCO和FashionIQ数据集上的检索效果对比。图中红色文字标记了冗余或缺失的信息，而绿色框表示目标对象。*

Figure 5 visually illustrates how `SDR-CIR` addresses `redundancy bias` and `omission bias` compared to a `description-only` baseline.

*   **Robustness to Redundant Information (Figure 5a - CIRCO):**
    *   **Example 1:** The `description-only` baseline includes "a red cowboy hat" from the `reference image`, which is irrelevant to the `target image`. This redundancy misleads the retrieval to an incorrect image. `SDR-CIR` successfully retrieves the target by reducing the influence of this redundant phrase, preserving the main information.
    *   **Example 2:** Similarly, the `description-only` output includes "near a person standing in shallow water", causing it to retrieve an image with similar irrelevant context. `SDR-CIR` debiases this, leading to the correct target.
        These examples confirm that the `Debias` step effectively suppresses the impact of irrelevant details propagated from the `reference image`.

*   **Robustness to Missing Information (Figure 5b - FashionIQ):**
    *   **Example 1:** The `description-only` approach fails because the generated description omits a key cue like "necklace" from the `reference image` that is relevant to the `target image`.
    *   **Example 2:** Another omission is "skirt length".
        In both cases, the `description-only` baseline fails to retrieve the correct `target image`. `SDR-CIR`, through its `Anchor` step, can supplement these `omitted elements` by fusing `reference image features`, enabling it to successfully retrieve the `target image`.

These qualitative examples provide strong intuitive evidence for the effectiveness of `SDR-CIR`'s `Anchor` and `Debias` steps in mitigating `semantic bias`.

## 6.4. Failure Analysis
The following figure (Figure 7 from the original paper) shows some failure cases of `SDR-CIR`.

![](images/7.jpg)

The paper also presents `failure cases` (Figure 7) to highlight `SDR-CIR`'s limitations, primarily when `modification intentions` are `ambiguous` or `underspecified`.
*   **Ambiguous Editing Intentions:**
    *   **Example (Figure 7a):** If the `modification text` is "the photo is shot from a different angle", the `MLLM` struggles because it doesn't specify *which* different angle (e.g., top-down, side, low-angle). Since the `target image` has a fixed viewpoint, the `MLLM` cannot determine the specific change. Even if it guesses, it might not match the specific target.
    *   **Example (Figure 7b):** If the `modification text` is "has a different color and there is a door next to it", and the color is not specified (e.g., "different from red"), the `MLLM` cannot infer the exact `target color`.
        These failures are mainly caused by a lack of sufficient constraints in the `modification text` to uniquely determine the `target image`. This leads to `plausible but non-discriminative descriptions` that don't precisely match the intended target. This suggests that while `SDR-CIR` improves upon existing biases, it is still reliant on the clarity of the user's `modification intent`.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper introduces `SDR-CIR`, a novel `training-free one-stage framework` for `Zero-Shot Composed Image Retrieval (ZS-CIR)`. `SDR-CIR` effectively addresses two key challenges: `visual noise` during `description generation` and `semantic bias` during `candidate ranking`. It achieves this through a sequential application of two core components:
1.  **Selective CoT:** This component guides the `Multimodal Large Language Model (MLLM)` to `selectively extract` only visual content relevant to the `modification text` from the `reference image`, thereby significantly reducing `visual noise` and generating more semantically aligned `target image descriptions`.
2.  **Semantic Debias Ranking (SDR):** This module employs a two-step `Anchor-then-Debias` strategy. The `Anchor` step fuses `reference image features` with `target description features` to reinforce useful semantics and supplement omitted cues. The `Debias` step then explicitly models the `visual semantic contribution` of the `reference image` to the `description` and applies it as a penalty term, effectively suppressing redundant semantics from the reference image that could mislead retrieval.

    Experimental results across three standard `CIR` benchmarks (CIRR, CIRCO, and FashionIQ) consistently demonstrate that `SDR-CIR` achieves state-of-the-art performance among one-stage methods, showcasing its robust capability in mitigating `semantic bias` while maintaining high retrieval efficiency.

## 7.2. Limitations & Future Work
The authors highlight the following limitations:
*   **Ambiguous Editing Intentions:** `SDR-CIR` struggles when the `modification text` is `underspecified` or `ambiguous` (e.g., "change to a different angle" or "a different color"). In such cases, the `MLLM` cannot infer the precise intended change, leading to `non-discriminative descriptions` that fail to uniquely identify the target image. This indicates a fundamental reliance on the specificity of the user query.
*   **Complex Visual Scenes:** While not explicitly stated as a separate point, the failure cases and the slight underperformance compared to `SEIZE` on the `CIRR subset` (which requires finer discrimination in complex scenes) implicitly suggest that `SDR-CIR` might still face challenges in highly complex visual scenarios where subtle distinctions are critical and `semantic editing increments` (as used by `SEIZE`) might offer stronger cues.

    Potential future research directions could involve:
*   Developing methods to handle `ambiguous` or `underspecified` `modification texts`, perhaps by incorporating user feedback or generating multiple diverse interpretations of the ambiguous instruction.
*   Exploring more sophisticated ways to model `semantic editing increments` within a one-stage framework, potentially combining `SDR` with techniques that capture finer-grained differences.
*   Investigating the synergy between `Selective CoT` and the `MLLM`'s internal reasoning capabilities to further refine the description generation process for even more complex modifications.

## 7.3. Personal Insights & Critique
`SDR-CIR` presents a highly intuitive and effective approach to tackling core challenges in `training-free ZS-CIR`. The innovation lies in its two-pronged strategy:
1.  **Proactive Noise Reduction:** The `Selective CoT` is a clever and necessary improvement over generic `CoT` prompting. By guiding the `MLLM` to be `semantically selective` during initial image understanding, it prevents `visual noise` from entering the description pipeline at the earliest stage. This is a more efficient strategy than generating a verbose description and then trying to filter it later.
2.  **Explicit Bias Correction:** The `Semantic Debias Ranking` module, particularly the `Debias` step, directly confronts the `fuzzy matching` nature of `ZS-CIR`. The idea of modeling the `reference image's visual semantic contribution` as a penalty term ($S_i = S_d - S_m$) is quite insightful. It explicitly acknowledges that not all information derived from the reference image is beneficial for retrieval, and some parts can actively mislead the ranking. The `Anchor` step, in turn, ensures that this debiasing doesn't inadvertently remove truly useful context. The complementarity of `Anchor` and `Debias` is a strong design choice.

    The method's `training-free` nature and impressive `efficiency` are significant advantages, making it highly practical for real-world deployments. Its strong performance across diverse datasets (object-level vs. attribute-level modifications) also speaks to its generalizability.

**Critique & Potential Issues:**
*   **Reliance on MLLM Quality:** While `SDR-CIR` is robust across different `MLLMs`, its ultimate performance is inherently tied to the capabilities of the underlying `MLLM` (as seen in the `GPT-4.1` vs. `Qwen2.5-VL-72B` comparison). As `MLLMs` continue to evolve, `SDR-CIR` will likely benefit, but its success is coupled with advancements in these foundation models.
*   **Hyperparameter Sensitivity:** The parameters $\alpha$ and $\beta$ are dataset-specific. This suggests that careful tuning is required for new datasets or domains. While common in many methods, it indicates that the optimal balance between `anchoring` and `debiasing` is not universal.
*   **Interpretability of $S_i$:** While $S_i = S_d - S_m$ is a reasonable heuristic for the `reference image's visual semantic contribution`, its precise interpretability could be further explored. Does it perfectly isolate "biased" content, or does it also implicitly penalize some useful, but not explicitly modified, aspects of the reference image that are relevant to the target?
*   **Limited Scope for Ambiguity:** The acknowledged limitation regarding `ambiguous modification texts` is crucial. Real-world users often provide vague instructions. Future work could integrate mechanisms for disambiguation, such as interactive prompts or leveraging world knowledge beyond the current image context.

**Transferability:**
The principles of `Selective CoT` and `Semantic Debias Ranking` could be highly transferable.
*   `Selective CoT` could be applied to other `MLLM`-based generation tasks where input contains redundancy (e.g., conditional image generation, video editing instruction following).
*   The `Semantic Debias Ranking` mechanism is quite general. Any retrieval task where a generated query description might suffer from `redundancy` or `omission` relative to the actual target (e.g., text-to-video retrieval, text-to-3D model retrieval, cross-modal summarization) could benefit from explicitly modeling and penalizing `source-induced bias` in the ranking phase. The idea of fusing features from the source and generated query for `anchoring` is also broadly applicable.

    Overall, `SDR-CIR` offers a well-reasoned and empirically validated solution to fundamental problems in `ZS-CIR`, representing a significant step forward in making these systems more accurate and practical.