# 1. Bibliographic Information

## 1.1. Title
AdaST: Adaptive Semantic Transformation of Visual Representation for Training-Free Zero-Shot Composed Image Retrieval

## 1.2. Authors
Anonymous authors (Paper under double-blind review, implying that author names and affiliations are withheld for the review process).

## 1.3. Journal/Conference
The paper is currently under double-blind review for a conference, likely a major computer vision or machine learning venue (e.g., ICLR, NeurIPS, ICCV, CVPR, AAAI, ACL, EMNLP, etc.), given the common practice of double-blind reviews in such fields. Its status as "Paper under double-blind review" means it has not yet been officially published in a proceedings, but is undergoing evaluation by peers. The references section lists publications from various top-tier conferences like CVPR, ICCV, ICLR, SIGIR, and ACMMM, indicating the caliber of the research typically submitted to this venue.

## 1.4. Publication Year
The publication year is not explicitly stated, but based on the references, many recent works cited are from 2023, 2024, and 2025. This suggests the paper was likely submitted for a conference in late 2024 or early 2025.

## 1.5. Abstract
This paper introduces `Adaptive Semantic Transformation (AdaST)`, a novel training-free method for `Composed Image Retrieval (CIR)`. CIR aims to retrieve a target image based on a reference image and a textual modification instruction, preserving unchanged visual attributes. Existing training-free CIR methods either synthesize proxy images (computationally expensive and time-consuming) or rely solely on text queries (lose crucial visual details). AdaST addresses these issues by transforming reference image features into proxy features, guided by text, at the feature level, rather than generating images. This approach efficiently preserves visual information. For finer-grained transformation, AdaST employs an adaptive weighting mechanism that balances proxy and text features, allowing the model to utilize proxy information only when it is deemed reliable. The method is lightweight, plug-and-play, and demonstrates state-of-the-art performance across three CIR benchmarks. It avoids the high cost of image generation and incurs only marginal inference overhead compared to text-based baselines.

## 1.6. Original Source Link
The paper is available at: `uploaded://abe2e778-d730-4df6-b564-4e1d5aa65edf` and `/files/papers/69a29ffa75362d74e45257f5/paper.pdf`.
This indicates the paper was uploaded as a PDF. Its publication status is "Paper under double-blind review."

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper addresses is the inherent trade-off in existing `Zero-Shot Composed Image Retrieval (ZS-CIR)` methods between visual fidelity and computational efficiency. `Composed Image Retrieval (CIR)` is the task of retrieving a target image given a reference image and a textual instruction specifying a modification (e.g., "the same shirt but in red").

This problem is highly important due to strong practical demand in domains like fashion e-commerce and online search engines, offering a more natural and intuitive interface than traditional keyword search. It also represents a fundamental step for advancing `vision-language understanding (VLU)` and fine-grained compositional reasoning.

Challenges and gaps in prior research include:
1.  **Costly Data Collection:** Collecting labeled CIR data is expensive and labor-intensive, limiting scalability and generalization to new domains. This motivates `zero-shot` methods that leverage pre-trained models without task-specific annotations.
2.  **Limitations of Text-based ZS-CIR:** Early ZS-CIR works often encode both the reference image and modification into a single textual representation (e.g., via textual inversion or `Large Language Models (LLMs)`). While effective, this approach compresses visual information into text, leading to a loss of fine-grained visual details and potentially visually mismatched results.
3.  **Limitations of Generation-based ZS-CIR:** More recent works attempt to synthesize a modified image using conditional generative models. While this preserves visual detail, it's computationally expensive and time-consuming (e.g., over 30 seconds per image), making it impractical for interactive retrieval systems.

    The paper's entry point and innovative idea is to overcome this dilemma by performing instruction-guided transformations directly in the *feature space* of a pre-trained `vision-language model (VLM)` like `CLIP`. This approach aims to preserve visual information efficiently by avoiding costly pixel-space generation, while still allowing for semantic manipulation guided by text.

## 2.2. Main Contributions / Findings
The paper's primary contributions are:

*   **Adaptive Semantic Transformation (AdaST):** Proposing a novel training-free ZS-CIR method that transforms reference image embeddings into proxy features. This transformation is guided by semantic shifts derived from `LLM`-generated captions, effectively preserving fine-grained visual details without relying on computationally expensive generative models.
*   **Adaptive Similarity Mechanism:** Introducing an adaptive weighting mechanism that dynamically balances the contributions of proxy-based and text-based similarities. This mechanism intelligently exploits proxy features when they are reliable (i.e., when semantic cues support the transformation) and ensures robustness through textual alignment, reducing false positives.
*   **State-of-the-Art Performance and Efficiency:** Demonstrating through extensive experiments that AdaST achieves state-of-the-art performance on three major CIR benchmarks (CIRCO, CIRR, Fashion-IQ). Crucially, it does so while being substantially faster and more lightweight than generation-based methods, incurring only marginal inference overhead compared to text-based baselines. For instance, it runs $186 \times$ faster than IP-CIR (a generation-based method) while achieving superior accuracy.
*   **Plug-and-Play Design:** Highlighting that AdaST is lightweight and can be seamlessly applied to existing training-free baselines in a plug-and-play manner, consistently boosting their performance.

    These findings solve the problem of balancing visual fidelity and computational efficiency in ZS-CIR, offering a practical and high-performing solution for real-world applications.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts

### Composed Image Retrieval (CIR)
`Composed Image Retrieval (CIR)` is a specialized information retrieval task where a user queries a database of images using two inputs: a `reference image` and a `textual modification instruction`. The goal is to find target images that incorporate the specified modification while largely retaining the visual attributes of the reference image that were *not* mentioned in the instruction. For example, given an image of a "blue car" and the instruction "change to red," the system should retrieve images of "red cars" that are otherwise similar to the reference blue car. This task requires complex multimodal understanding and compositional reasoning.

### Zero-Shot Composed Image Retrieval (ZS-CIR)
`Zero-Shot Composed Image Retrieval (ZS-CIR)` refers to performing the CIR task without training on task-specific `(reference image, modification text, target image)` triplets. Instead, ZS-CIR methods rely on pre-trained models, typically `Vision-Language Models (VLMs)` and `Large Language Models (LLMs)`, to generalize to unseen compositions. This is particularly valuable because collecting large-scale labeled CIR datasets is labor-intensive and costly.

### Vision-Language Models (VLMs)
`Vision-Language Models (VLMs)` are neural networks designed to process and understand information from both visual (images/videos) and textual modalities. They learn to align representations from these different modalities into a common, joint embedding space. A prominent example is `CLIP (Contrastive Language-Image Pre-training)`, which learns to associate images with their corresponding text descriptions through a contrastive learning objective. In this joint embedding space, images and texts that are semantically similar are embedded close to each other.
*   **CLIP (Contrastive Language-Image Pre-training):** CLIP consists of an `image encoder` and a `text encoder`. The image encoder maps an image to an embedding, and the text encoder maps a text phrase to an embedding. During training, CLIP is fed pairs of images and texts (e.g., an image of a dog and the text "a photo of a dog"). It learns to maximize the similarity (e.g., cosine similarity) between matching image-text pairs and minimize similarity between non-matching pairs. This allows CLIP to learn powerful, zero-shot transferable representations for various vision-language tasks.
    *   **Image Encoder $E_I$:** A neural network (e.g., a `Vision Transformer (ViT)`) that converts an input image $I$ into a fixed-size numerical vector (embedding) `f_I = E_I(I)`.
    *   **Text Encoder $E_T$:** A neural network (e.g., a Transformer-based text model) that converts an input text $T$ into a fixed-size numerical vector (embedding) `f_T = E_T(T)`.

### Large Language Models (LLMs)
`Large Language Models (LLMs)` are advanced deep learning models, typically based on the Transformer architecture, trained on massive amounts of text data. They can understand, generate, and process human language with remarkable fluency and coherence. LLMs are capable of various tasks, including text generation, summarization, translation, question answering, and complex reasoning. In the context of CIR, LLMs can be used to generate descriptive captions for images or to interpret and reformulate textual modification instructions based on existing captions. Examples include GPT-3.5, GPT-4, Llama, etc.

### Feature-level Transformation
`Feature-level transformation` refers to manipulating the numerical representations (features or embeddings) of data directly in a latent space, rather than operating on the raw input data (e.g., pixels of an image or tokens of a text). In this paper, it means transforming the numerical embedding of a reference image to reflect a textual modification, all within the embedding space learned by a VLM. This is in contrast to `pixel-level transformation`, which would involve generating a new image in pixel space, or `text-level transformation`, which would involve manipulating text strings.

### Cosine Similarity
`Cosine similarity` is a measure of similarity between two non-zero vectors in an inner product space. It measures the cosine of the angle between them. A cosine similarity of 1 indicates that the vectors are in the same direction (perfectly similar), 0 indicates they are orthogonal (no similarity), and -1 indicates they are in opposite directions (perfectly dissimilar). It is often used in high-dimensional spaces to compare document or image embeddings, as it is less sensitive to magnitude differences and focuses on the orientation of vectors.
Given two vectors $A$ and $B$, their cosine similarity is calculated as:
\$
\mathrm{sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}
\$
Where $A_i$ and $B_i$ are components of vector $A$ and $B$ respectively, and $\|A\|$ and $\|B\|$ are their magnitudes (Euclidean norms).

## 3.2. Previous Works
The paper discusses several categories of previous work in CIR and ZS-CIR:

*   **Supervised CIR (Vo et al., 2019; Chen & Bazzani, 2020; Lee et al., 2021; Delmas et al., 2022):** These methods align image-text pairs within a shared embedding space, often using contrastive learning, or employ cross-modal attention. They require large, task-specific datasets (e.g., Fashion-IQ, CIRR) for training.
    *   **Limitation:** High cost and labor-intensity of dataset construction, limiting scalability and generalization to unseen domains.

*   **Zero-Shot CIR (ZS-CIR):** These approaches aim to reduce annotation costs by leveraging large-scale pre-trained VLMs.
    *   **Early ZS-CIR Approaches (Textual Inversion and LLM-based):**
        *   **Textual Inversion (Saito et al., 2023; Baldrati et al., 2023; Gu et al., 2024):** These methods train an inversion model (e.g., Kumari et al., 2023; Gal et al., 2023; Ruiz et al., 2023) to map images into the text token space. The inverted text representation is then combined with the modification instruction to form a query.
            *   **Example (Conceptual):** If you have an image of a "red car," textual inversion might learn a pseudo-word like $<red-car-token>$ that represents this image. Then, for the instruction "change to blue," the query becomes $<red-car-token> change to blue$.
        *   **LLM-based Approaches (Karthik et al., 2023; Yang et al., 2024b;a; Tang et al., 2025):** These methods caption images into natural language (e.g., "a red car") and then use LLMs to reason jointly over this reference description and the modification text (e.g., LLM processes "a red car" + "change to blue" to produce "a blue car"). The LLM's output (e.g., a modified text caption) is then used for retrieval.
            *   **Example (Conceptual):** Given an image, a captioning model generates "a black dog with a collar." The modification is "wearing a red bandana." An LLM might combine these to generate "a black dog with a collar wearing a red bandana." This modified caption is then used to search.
        *   **Limitation (for both):** Both textual inversion and LLM-based approaches inevitably compress visual information into textual form. This leads to a loss of fine-grained visual details from the reference image, potentially yielding semantically correct but visually mismatched results.

    *   **Generation-based ZS-CIR (Li et al., 2025; Zhou et al., 2024; Wei et al., 2023):** These methods directly synthesize a modified image from the reference image and textual instruction using conditional generative models (e.g., Diffusion Models). The generated image then serves as the retrieval query.
        *   **Example (Conceptual):** Given an image of a dog and the instruction "add a hat," a generative model would produce an image of that dog wearing a hat. This *generated image* is then used to retrieve similar images from the database.
        *   **Limitation:** While preserving visual detail, generation in high-dimensional pixel space is extremely costly and time-consuming (often exceeding 30 seconds per image), making it impractical for interactive applications.

*   **Text-guided Semantic Transformation (Fu et al., 2022; Kwon & Ye, 2022; Gal et al., 2022; Ye-Bin et al., 2023; Park et al., 2025):** These methods exploit the latent space of pre-trained VLMs (like CLIP) to align image features with textual guidance. The core idea is that the semantic difference between a source text and a target text can be represented as a difference vector in the joint embedding space. This difference vector can then be applied to an image's feature to transform it semantically.
    *   **Example (Conceptual):** If $f_{text\_blue}$ is the embedding for "blue" and $f_{text\_red}$ for "red," then $\Delta T = f_{text\_red} - f_{text\_blue}$ represents the "change to red" vector. If an image $I_{blue\_car}$ has an embedding $f_{I_{blue\_car}}$, a naive transformation would be $f_{I_{blue\_car}} + \Delta T$.
    *   **Limitation:** Direct application of the text-feature difference vector to image features has shown limited effectiveness in prior CIR methods (Vo et al., 2019; Li et al., 2025). The magnitude of the text-space shift might not directly translate effectively to the image-feature space.

## 3.3. Technological Evolution
The field of CIR has evolved from supervised, task-specific models to `zero-shot` approaches necessitated by the high cost of data collection. Early ZS-CIR methods primarily focused on converting everything to text, leveraging the strong language understanding capabilities of `LLMs` or `textual inversion` models. While efficient, this led to a loss of visual granularity. The next wave introduced `generation-based` methods, which aimed to recover visual fidelity by synthesizing modified images. However, this came at a significant computational cost, making real-time applications infeasible.

This paper's work (`AdaST`) fits into the timeline as a crucial step that bridges the gap between these two extremes. It builds upon the idea of `text-guided semantic transformation` in `VLM` latent spaces, a concept explored in image style transfer and editing. AdaST refines this by proposing an *efficient* and *effective* feature-level transformation, ensuring visual fidelity without the heavy computational burden of image generation, while also addressing the limitations of simple direct feature transfer. It represents an advancement towards practical and scalable ZS-CIR systems.

## 3.4. Differentiation Analysis
Compared to the main methods in related work, AdaST offers several core innovations and differentiations:

*   **vs. Text-based ZS-CIR (e.g., CIReVL, SEIZE, Pic2Word, LDRE):**
    *   **Differentiation:** AdaST does not discard fine-grained visual details by compressing the reference image into a textual representation. Instead, it operates directly on the visual features of the reference image in the `VLM`'s latent space. This preserves more information about the original object's identity, shape, and style, leading to more visually faithful retrievals.
    *   **Innovation:** By transforming *features* rather than *text*, AdaST avoids the semantic drift or loss of specificity that can occur when visual cues are entirely rephrased into language.

*   **vs. Generation-based ZS-CIR (e.g., IP-CIR):**
    *   **Differentiation:** AdaST avoids the computationally expensive and time-consuming process of generating new images in high-dimensional pixel space. It performs transformations in the much lower-dimensional feature space.
    *   **Innovation:** This efficiency makes AdaST practical for interactive retrieval systems, where generation-based methods are prohibitively slow. It achieves comparable or superior accuracy without the heavy computational cost.

*   **vs. Direct Text-guided Semantic Transformation (naive approaches):**
    *   **Differentiation:** AdaST introduces an `optimal scaling` strategy for the semantic transformation vector ($\Delta T$) and an `adaptive similarity fusion` mechanism. Prior work found direct transfer of the text-feature difference vector to image features to be of `limited effectiveness`.
    *   **Innovation:** The optimal scaling ensures that the proxy embedding adequately captures the modification without becoming too detached from the original image. The adaptive gating mechanism intelligently balances the proxy's visual cues with the target caption's semantic guidance, making the retrieval more robust and preventing misleading visual matches. This `finer-grained transformation` and `robust fusion` are key to AdaST's improved performance.

        In essence, AdaST uniquely combines the efficiency of text-based methods with the visual detail preservation of generation-based methods, while also refining the core idea of feature-space transformation to be more effective and robust.

# 4. Methodology

The `Adaptive Semantic Transformation (AdaST)` method is a training-free approach for `zero-shot composed image retrieval (ZS-CIR)`. It operates by transforming reference image features into "proxy features" guided by textual instructions, and then adaptively fusing these with text-based similarities for robust retrieval. The method is structured into three main components: `Text Guidance Generation`, `Text-Guided Semantic Transformation`, and `Adaptive Similarity Fusion`.

## 4.1. Principles
The core idea behind AdaST is to perform semantic modifications directly within the feature space of a pre-trained `Vision-Language Model (VLM)`, rather than generating new images or relying solely on text. This is based on the principle that VLMs (like `CLIP`) embed images and text into a joint semantic space where similar concepts are represented by nearby vectors. By deriving a "semantic shift" from text (i.e., the difference between the text embeddings of a reference description and a modified description), this shift can be applied to the reference image's embedding. The theoretical basis is that this feature-level manipulation can effectively alter the image's semantics in line with the instruction while preserving unmentioned visual attributes, thus offering a balance between visual fidelity and computational efficiency. The method also introduces mechanisms to optimize this transformation and dynamically weight the contribution of the transformed features to ensure robust retrieval.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Text Guidance Generation
The first step is to create explicit textual guidance that bridges the visual content of the reference image and the desired textual modification. This guidance consists of two components: a `reference caption` and a `target caption`.

1.  **Reference Caption ($T_r$) Generation:**
    *   The reference caption $T_r$ is a natural language description of the input `reference image` $I_r$.
    *   This caption is obtained by passing $I_r$ through an existing, powerful `captioning model`, such as `BLIP-2` (Li et al., 2023).
    *   **Purpose:** $T_r$ provides a textual summary of the visual content of $I_r$, which will be used by an `LLM` to understand the reference image context for generating the target caption.

2.  **Target Caption ($T_t$) Generation:**
    *   The `target caption` $T_t$ explicitly encodes the intended modification specified by the `textual instruction` $T_{\mathrm{inst}}$, conditioned on the `reference caption` $T_r$.
    *   This is achieved using a `Large Language Model (LLM)` (e.g., `GPT-4o`), similar to approaches used in previous `LLM-based ZS-CIR` methods (Karthik et al., 2023). The LLM takes $T_r$ and $T_{\mathrm{inst}}$ as input and generates $T_t$.
    *   **Purpose:** $T_t$ represents the desired semantic state of the target image in natural language, emphasizing the instructed change. It serves as a semantic anchor for aligning the transformed image features.

### 4.2.2. Text-Guided Semantic Transformation
This stage focuses on transforming the `reference image feature` into a `proxy feature` that reflects the semantic shift specified by the text guidance. This transformation occurs directly in the feature space of a pre-trained `VLM`.

1.  **Embedding into Joint Representation Space:**
    *   Both the generated captions ($T_r$, $T_t$) and the `reference image` ($I_r$) are embedded into a shared, joint representation space. This is done using a `text encoder` $E_T$ and an `image encoder` $E_I$ from a pre-trained `VLM` (e.g., `CLIP` (Radford et al., 2021)).
    *   The resulting embeddings are:
        \$
        f_{T_r} = E_T(T_r) \\
        f_{T_t} = E_T(T_t) \\
        f_{I_r} = E_I(I_r)
        \$
    *   **Explanation:**
        *   $f_{T_r}$: The numerical vector embedding of the reference caption, representing its semantic content.
        *   $f_{T_t}$: The numerical vector embedding of the target caption, representing the desired modified semantic content.
        *   $f_{I_r}$: The numerical vector embedding of the reference image, representing its visual content.
        *   $E_T$: The text encoder component of the VLM.
        *   $E_I$: The image encoder component of the VLM.
        *   These embeddings reside in the same high-dimensional space where semantic similarities are reflected by vector proximity.

2.  **Semantic Shift Calculation ($\Delta T$):**
    *   The `semantic shift` or difference vector $\Delta T$ is computed by subtracting the reference caption embedding from the target caption embedding. This vector represents the direction and magnitude of the semantic change specified by the instruction, as captured in the text embedding space.
    *   Formally:
        \$
        \Delta T = f_{T_t} - f_{T_r}
        \$
    *   **Explanation:** This vector captures "what needs to be changed" from the perspective of text. For instance, if $T_r$ is "blue car" and $T_t$ is "red car", $\Delta T$ represents the "change from blue to red" in the semantic embedding space.

3.  **Proxy Embedding Formation ($f_P^{(\alpha)}$):**
    *   The `proxy embedding` $f_P^{(\alpha)}$ is created by adding the `semantic shift` $\Delta T$ to the `reference image embedding` $f_{I_r}$. A `scaling factor` $\alpha$ controls the strength or magnitude of this applied transformation.
    *   Formally:
        \$
        f_P^{(\alpha)} = f_{I_r} + \alpha \Delta T
        \$
    *   **Explanation:** The idea is to move the reference image's feature in the direction of the desired textual modification. $\alpha$ determines how far it moves. A small $\alpha$ means a subtle change, while a large $\alpha$ means a more pronounced change.

4.  **Optimal Scaling ($\alpha^*$):**
    *   A naive choice of $\alpha=1$ often leads to suboptimal results, as the proxy embedding might not sufficiently capture the intended modification or might deviate too much. To find an optimal $\alpha$, the paper proposes an optimization-based approach.
    *   The objective is to ensure that the `proxy embedding` $f_P^{(\alpha)}$ is well-aligned with the `target caption embedding` $f_{T_t}$ (ensuring the modification is represented) but also sufficiently distinct from the `reference image embedding` $f_{I_r}$ (to prevent collapse to original content).
    *   This is formulated as an optimization problem:
        \$
        \alpha^* = \underset{\alpha}{\arg\min} \left( 1 - \mathrm{sim}( f_P^{(\alpha)}, f_{T_t} ) + \beta \cdot \mathrm{sim}( f_P^{(\alpha)}, f_{I_r} ) \right)
        \$
    *   **Explanation:**
        *   $\alpha^*$: The optimal scaling factor.
        *   $\mathrm{sim}(\cdot, \cdot)$: Denotes `cosine similarity`.
        *   $1 - \mathrm{sim}( f_P^{(\alpha)}, f_{T_t} )$: This term encourages $f_P^{(\alpha)}$ to be highly similar to $f_{T_t}$ (cosine similarity close to 1, so $1-\mathrm{sim}$ close to 0). It minimizes the angular distance to the target caption.
        *   $\beta \cdot \mathrm{sim}( f_P^{(\alpha)}, f_{I_r} )$: This term penalizes excessive similarity between $f_P^{(\alpha)}$ and $f_{I_r}$. It prevents the proxy from staying too close to the original reference image, ensuring the modification is sufficiently applied.
        *   $\beta$: A `weighting coefficient` that controls the influence of this penalty term. A higher $\beta$ means stronger discouragement of similarity to the reference.
    *   This optimization problem admits a `closed-form solution`, which can be computed directly using inner products:
        \$
        \alpha^* = \frac{ \boldsymbol{x}^{\top} \boldsymbol{y} \cdot \boldsymbol{x}^{\top} d - d^{\top} \boldsymbol{y} \cdot \| \boldsymbol{x} \|^2 }{ d^{\top} \boldsymbol{y} \cdot \boldsymbol{x}^{\top} d - \boldsymbol{x}^{\top} \boldsymbol{y} \cdot \| d \|^2 }
        \$
    *   **Explanation of terms in the closed-form solution:**
        *   $\boldsymbol{x} = f_{I_r}$: The normalized reference image embedding.
        *   $\boldsymbol{y} = \tilde{f}_{T_t} - \beta \tilde{f}_{I_r}$: A composite vector involving the normalized target caption embedding and the normalized reference image embedding, weighted by $\beta$.
        *   $d = \Delta T$: The semantic shift vector.
        *   $\tilde{f} = f / \| f \|_2$: Denotes the $L_2$-normalized version of a feature vector $f$. Normalization is crucial for cosine similarity and vector operations in this context.
        *   $\boldsymbol{x}^{\top} \boldsymbol{y}$: Dot product between $\boldsymbol{x}$ and $\boldsymbol{y}$.
        *   $\|\boldsymbol{x}\|^2$: Squared $L_2$-norm (magnitude) of $\boldsymbol{x}$.
    *   The resulting `proxy embedding` is $f_P = f_P^{(\alpha^*)}$.

### 4.2.3. Adaptive Similarity Fusion
While the `proxy embedding` $f_P$ aims to capture the modified visual content, relying solely on it for retrieval can be problematic. It might assign high scores to visually similar images that are semantically irrelevant to the instruction. To counteract this, `AdaST` incorporates semantic guidance from the `target caption feature` $f_{T_t}$ through an `adaptive similarity fusion` module.

1.  **Database Embeddings:**
    *   First, the features for all images in the `database` $\mathcal{D}$ are extracted using the `image encoder` $E_I$.
    *   Formally:
        \$
        f_{I_i^{\mathrm{DB}}} = E_I ( I_i^{\mathrm{DB}} ) , \ \forall i = \{ 1 , \dots , N \}
        \$
    *   **Explanation:** $f_{I_i^{\mathrm{DB}}}$ is the embedding for the $i$-th image in the database. The notation $f_{I^{\mathrm{DB}}}$ is used to collectively refer to these database embeddings.

2.  **Similarity Scores Calculation:**
    *   For each query, three types of similarity scores are computed between the query components and the database images:
        \$
        S_{T_t} = \mathrm{sim}( f_{T_t} , f_{I^{\mathrm{DB}}} ) \\
        S_{T_r} = \mathrm{sim}( f_{T_r} , f_{I^{\mathrm{DB}}} ) \\
        S_P = \mathrm{sim}( f_P , f_{I^{\mathrm{DB}}} )
        \$
    *   **Explanation:**
        *   $S_{T_t}$: `Target-caption similarity`. Measures how semantically similar the database images are to the desired target description.
        *   $S_{T_r}$: `Reference-caption similarity`. Measures how semantically similar the database images are to the original reference description.
        *   $S_P$: `Proxy similarity`. Measures how visually/semantically similar the database images are to the transformed proxy feature.

3.  **Gating Function:**
    *   A `gating mechanism` adaptively regulates the contribution of the `proxy similarity` $S_P$. This gate ensures that $S_P$ influences retrieval only when there's sufficient semantic evidence from the target caption, preventing false positives from purely visual resemblance.
    *   The gating function is defined as:
        \$
        G ( \Delta S_T ) = \left\{ \begin{array}{ll} { \lambda , } & { { \Delta S_T + m \ge 0 } } \\ { { 0 , } } & { { \mathrm{otherwise} } } \end{array} \right. , \ \Delta S_T = S_{T_t} - S_{T_r}
        \$
    *   **Explanation:**
        *   $\Delta S_T = S_{T_t} - S_{T_r}$: This term measures the difference in semantic alignment. If $\Delta S_T$ is positive, it means the database image is more semantically aligned with the target caption than with the reference caption.
        *   $m$: A `margin` parameter. If $\Delta S_T + m \ge 0$, it means the target caption shows sufficiently greater (or equal, considering the margin) semantic alignment than the reference.
        *   $\lambda$: A `weighting coefficient` that determines the maximum influence of $S_P$ when the gate is active.
        *   If the condition $\Delta S_T + m \ge 0$ is met, the gate activates, returning $\lambda$. Otherwise, it returns `0`, effectively disabling the proxy's contribution.

4.  **Final Similarity Score ($S_{\mathrm{total}}$):**
    *   The total similarity score for retrieval is a combination of the `adaptive proxy similarity` and the `target-caption similarity`.
    *   Formally:
        \$
        S_{\mathrm{total}} = S_A \cdot S_P + S_{T_t} \\
        S_A = S_{T_t} \cdot G ( \Delta S_T )
        \$
    *   **Explanation:**
        *   $S_A$: Represents an `adaptive weight` applied to the `proxy similarity` $S_P$. This weight is derived from the `target-caption similarity` $S_{T_t}$ scaled by the `gating function` $G(\Delta S_T)$. If the gate is open ($G(\Delta S_T) = \lambda$), $S_A$ becomes $\lambda \cdot S_{T_t}$, meaning the proxy's contribution is scaled by how well the image aligns with the *target caption semantically*. If the gate is closed ($G(\Delta S_T) = 0$), then $S_A$ becomes `0`, effectively removing the proxy's influence.
        *   The `target-caption similarity` $S_{T_t}$ is always included as a baseline, ensuring that semantic relevance to the instruction is consistently considered.

5.  **Retrieval Result:**
    *   The final retrieval result is the database image that yields the maximum `total similarity score`.
    *   Formally:
        \$
        I_t = \arg\max_{I_i^{\mathrm{DB}} \in \mathcal{D}} S_{\mathrm{total}}
        \$
    *   **Explanation:** The image $I_t$ that best matches the combined visual and semantic cues (as expressed by $S_{\mathrm{total}}$) is returned as the target image.

        Figure 2 from the original paper illustrates this overall pipeline:

        ![Figure 2: Overall pipeline of AdaST. It consists of three stages. (1) Text guidance generation: a reference caption is obtained from the input image using a captioning model, and an LLM combines it with the textual instruction to generate a target caption. (2) Text-guided semantic transformation: both captions and the reference image are embedded with CLIP, where the feature difference between the reference and target captions is transferred to the reference image feature with a scaling factor, yielding a proxy feature. (3) Adaptive similarity fusion: an adaptive gating mechanism fuses proxy similarity with text-based similarity, allowing proxy similarity to contribute only when supported by consistent semantic cues.](images/2.jpg)
        *该图像是AdaST的总体流程示意图，展示了三个阶段：文本引导生成、文本引导的语义变换，以及自适应相似度融合。在图中，输入图像的参考标题通过大语言模型（LLM）与文本指令结合生成目标标题，然后利用CLIP进行特征嵌入，以实现图像特征的转换和相似度计算。*

Figure 2: Overall pipeline of AdaST. It consists of three stages. (1) Text guidance generation: a reference caption is obtained from the input image using a captioning model, and an LLM combines it with the textual instruction to generate a target caption. (2) Text-guided semantic transformation: both captions and the reference image are embedded with CLIP, where the feature difference between the reference and target captions is transferred to the reference image feature with a scaling factor, yielding a proxy feature. (3) Adaptive similarity fusion: an adaptive gating mechanism fuses proxy similarity with text-based similarity, allowing proxy similarity to contribute only when supported by consistent semantic cues.

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted on three widely-used `Composed Image Retrieval (CIR)` benchmarks: `CIRR`, `CIRCO`, and `Fashion-IQ`.

### CIRR (Liu et al., 2021)
*   **Source:** Consists of 21,552 images collected from the `NLVR dataset` (Suhr et al., 2018).
*   **Scale:** Contains 36,554 associated queries.
*   **Characteristics:** Designed to support fine-grained natural language modifications, enabling retrieval based on subtle semantic differences between images.
*   **Limitation:** It has a potential `false-negative` issue, meaning multiple images in the gallery might satisfy the same instruction, but only one is annotated as the ground truth.
*   **Why chosen:** It's a standard benchmark for CIR, particularly for fine-grained linguistic modifications.

### CIRCO (Baldrati et al., 2023)
*   **Source:** Constructed from `COCO 2017` (Lin et al., 2014).
*   **Scale:** Includes a validation set with 220 queries and a test set with 800 queries.
*   **Characteristics:** Explicitly addresses the `false-negative` issue of CIRR by providing multiple annotated target images per query. The instructions cover diverse modifications like `attribute edits`, `object substitutions`, and `style modifications`, posing significant challenges for `compositional reasoning`.
*   **Why chosen:** Its multi-target nature provides a more robust evaluation of retrieval performance, especially for complex modifications.

### Fashion-IQ (Wu et al., 2021)
*   **Source:** A domain-specific benchmark focused on `fashion retrieval`.
*   **Scale:** Contains 30,135 queries and 77,683 product images across three categories: `Shirt`, `Dress`, and `Toptee`.
*   **Characteristics:** Queries are written by annotators and describe modifications to reference garments (e.g., "the same shirt but in red"). It features highly fine-grained and diverse natural language modifications related to fashion attributes (texture, silhouette, color, etc.).
*   **Why chosen:** Represents a real-world application of CIR where fine-grained visual details and specific attribute modifications are crucial.

## 5.2. Evaluation Metrics
The choice of evaluation metrics follows the official protocols of each dataset.

### Recall@K (R@K)
*   **Conceptual Definition:** `Recall@K` measures the proportion of queries for which at least one of the ground-truth target images is present among the top $K$ retrieved results. It assesses the model's ability to find relevant items within a small set of highly-ranked results. A higher R@K indicates better retrieval performance.
*   **Mathematical Formula:**
    \$
    \mathrm{R@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\exists t \in GT_q \text{ s.t. } t \in R_{q,K})
    \$
*   **Symbol Explanation:**
    *   $|Q|$: Total number of queries.
    *   $q$: An individual query.
    *   $GT_q$: The set of ground-truth target images for query $q$.
    *   $R_{q,K}$: The set of top $K$ retrieved images for query $q$.
    *   $\mathbb{I}(\cdot)$: An indicator function, which is 1 if the condition inside is true, and 0 otherwise.
    *   $\exists$: "There exists."
    *   `s.t.`: "Such that."
*   **Usage:** Used for CIRR with $K \in \{1, 5, 10, 50\}$. For Fashion-IQ with $K \in \{10, 50\}$.

### RecallSubset@K (RS@K)
*   **Conceptual Definition:** `RecallSubset@K` is a variant of Recall@K specifically designed for datasets like CIRR. It focuses only on images within the same semantic set as the reference and target, capturing retrieval performance under closely related distractor groups. This means it evaluates if the correct item is retrieved among other items that are semantically similar but not the target.
*   **Mathematical Formula:** The paper does not provide a specific formula for RecallSubset@K, but conceptually it follows the R@K definition applied to a constrained gallery (subset) of images that are semantically related to the query.
    \$
    \mathrm{RS@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\exists t \in GT_q \text{ s.t. } t \in R'_{q,K})
    \$
*   **Symbol Explanation:**
    *   $|Q|$, $q$, $GT_q$, $\mathbb{I}(\cdot)$, $\exists$, `s.t.` are as defined for R@K.
    *   `R'_{q,K}`: The set of top $K$ retrieved images for query $q$, where the retrieval is performed *only over a subset of the database* that contains images semantically related to the query's reference and target.
*   **Usage:** Used for CIRR with $K \in \{1, 2, 3\}$.

### Mean Average Precision at Top-K (mAP@K)
*   **Conceptual Definition:** `Mean Average Precision at Top-K (mAP@K)` is a common metric in retrieval tasks, especially when there can be multiple ground-truth targets for a query (as in CIRCO). For each query, `Average Precision (AP)` is calculated as the average of the precision values at each relevant item in the ranked list. `mAP@K` then averages these AP scores over all queries, considering only results up to rank K. It assesses both the precision and the ranking quality of the retrieved results. A higher mAP@K indicates better performance in finding all relevant items and ranking them highly.
*   **Mathematical Formula:**
    \$
    \mathrm{mAP@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathrm{AP}_q (\mathrm{K})
    \$
    where $\mathrm{AP}_q (\mathrm{K})$ for a given query $q$ is:
    \$
    \mathrm{AP}_q (\mathrm{K}) = \sum_{k=1}^{K} P(k) \cdot \Delta r(k)
    \$
    or more commonly for discrete lists:
    \$
    \mathrm{AP}_q (\mathrm{K}) = \frac{1}{|GT_q|} \sum_{k=1}^{K} P(k) \cdot \mathbb{I}(R_{q,k} \in GT_q)
    \$
*   **Symbol Explanation:**
    *   $|Q|$: Total number of queries.
    *   $\mathrm{AP}_q (\mathrm{K})$: Average Precision for query $q$ considering only ranks up to $K$.
    *   $K$: The maximum rank considered.
    *   `P(k)`: Precision at rank $k$, defined as $\frac{\text{number of relevant items at or above rank } k}{\text{total number of items at or above rank } k}$.
    *   $\Delta r(k)$: Change in recall from rank `k-1` to $k$. (In the discrete sum form, this is simplified by counting only at relevant items).
    *   $|GT_q|$: The total number of ground-truth target images for query $q$.
    *   $R_{q,k}$: The image retrieved at rank $k$ for query $q$.
    *   $\mathbb{I}(R_{q,k} \in GT_q)$: Indicator function, 1 if the image at rank $k$ is a ground-truth target, 0 otherwise.
*   **Usage:** Used for CIRCO with $K \in \{5, 10, 25, 50\}$.

## 5.3. Baselines
The paper compares AdaST against several representative baselines:

*   **CIReVL (Karthik et al., 2023):** An `LLM-based` ZS-CIR method. This is a primary baseline, and AdaST is applied on top of it.
*   **SEIZE (Yang et al., 2024a):** Another `LLM-based` ZS-CIR method. AdaST is also applied on top of this baseline.
*   **LinCIR (Gu et al., 2024):** A `training-based` baseline that leverages `textual inversion`. Used as an additional baseline for the Fashion-IQ dataset.
*   **IP-CIR (Li et al., 2025):** A `generation-based` ZS-CIR method that synthesizes modified images. Used for comparison of both accuracy and, crucially, `inference time`. It often appears in combination with other methods (e.g., `LDRE + IP-CIR`, `LinCIR + IP-CIR`).
*   **Pic2Word (Saito et al., 2023):** An early `textual inversion`-based ZS-CIR method.
*   **SEARLE (Baldrati et al., 2023):** Another `textual inversion`-based ZS-CIR method.
*   **LDRE (Yang et al., 2024b):** `LLM-based Divergent Reasoning and Ensemble` for ZS-CIR.
*   **OSrCIR (Kolouju et al., 2025):** A recent ZS-CIR method mentioned in the tables.

    These baselines are representative as they cover different popular strategies for ZS-CIR: `LLM-based` (converting everything to text), `textual inversion` (mapping images to text tokens), and `generation-based` (synthesizing new images).

## 5.4. Implementation Details
*   **Retrieval Model Backbones:**
    *   `CLIP` backbones are used for the `Vision-Language Model (VLM)`:
        *   `ViT-B/32` (Vision Transformer, Base size, patch size 32x32)
        *   `ViT-L/14` (Vision Transformer, Large size, patch size 14x14)
        *   `ViT-G/14` (Vision Transformer, Giant size, patch size 14x14)
    *   **Weights:** Official `OpenAI` (Radford et al., 2021) weights are used by default. For `ViT-G/14`, `OpenCLIP` (Ilharco et al., 2021) weights are used.

*   **Captioning Model:**
    *   `BLIP-2` (Li et al., 2023) is employed for generating the `reference caption` $T_r$ from the input image.

*   **LLM Model:**
    *   The baseline codebases originally used `GPT-3.5-turbo`, which is no longer available. Following a recent work (Tang et al., 2025), the authors re-implemented the baseline with `GPT-4o`. This ensures a fair comparison, as both AdaST and the baselines use the same `LLM` for text guidance generation.

*   **Hardware:** All experiments are conducted on a single `A6000 GPU`.

*   **Hyperparameters:**
    *   For all baselines and datasets, the following hyperparameters are set:
        *   $\beta = 0.25$: Weighting coefficient for the penalty term in the optimal scaling objective, controlling the influence of similarity to the reference image.
        *   $\lambda = 4$: Weighting coefficient in the gating function, controlling the maximum influence of proxy similarity.
        *   $m = 0.1$: Margin in the gating function, determining the threshold for semantic alignment.
    *   **Exception:** For the `CIRCO` dataset, $m = 0$ is used.

# 6. Results & Analysis

## 6.1. Core Results Analysis

The experimental results demonstrate that AdaST consistently achieves state-of-the-art performance across all evaluated benchmarks and backbones, while maintaining high efficiency.

### CIRCO and CIRR Benchmarks
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th colspan="4">Benchmark</th>
<th colspan="3">CIRCO (mAP@K)</th>
<th colspan="4">CIRR (Recall@K)</th>
<th colspan="2">CIRR (Recallsubset@K)</th>
<th></th>
</tr>
<tr>
<th>Backbone</th>
<th>SEARLE</th>
<th>Method</th>
<th></th>
<th>k=5</th>
<th>k=10</th>
<th>k=25</th>
<th>k=50</th>
<th>k=1</th>
<th>k=5</th>
<th>k=10</th>
<th>k=50</th>
<th>k=1</th>
<th>k=2</th>
<th>k=3</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">ViT-B/32</td>
<td>CIReVL†</td>
<td>ICCV23 ICLR24</td>
<td></td>
<td>9.35</td>
<td>9.94</td>
<td>11.13</td>
<td>11.84</td>
<td>24</td>
<td>53.42</td>
<td>66.82</td>
<td>84.70</td>
<td>54.89</td>
<td>76.60</td>
<td>88.19</td>
</tr>
<tr>
<td>LDRE</td>
<td>SIGIR24</td>
<td></td>
<td>17.96</td>
<td>18.32</td>
<td>20.21</td>
<td>21.11</td>
<td>25.69</td>
<td>46.96</td>
<td>55.13</td>
<td>89.9</td>
<td>54.30</td>
<td>76.5</td>
<td>88.10</td>
</tr>
<tr>
<td>SEIZE*</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>69.04</td>
<td></td>
<td></td>
<td>80.65</td>
<td>90.7</td>
</tr>
<tr>
<td></td>
<td>ACMMM24</td>
<td>18.75</td>
<td>19.37</td>
<td>21.09</td>
<td>22.07</td>
<td>26.96</td>
<td>55.59</td>
<td>68.24</td>
<td>88.34</td>
<td>66.82</td>
<td>85.23</td>
<td>93.35</td>
</tr>
<tr>
<td>OSrCIR</td>
<td>CVPR25</td>
<td>18.04</td>
<td>19.17</td>
<td>20.94</td>
<td>21.85</td>
<td>25.42</td>
<td>54.54</td>
<td>68.19</td>
<td>-</td>
<td>62.31</td>
<td>80.86</td>
<td>91.13</td>
</tr>
<tr>
<td>CIReVL† + Ours</td>
<td></td>
<td></td>
<td>15.20</td>
<td>15.73</td>
<td>17.25</td>
<td>18.12</td>
<td>25.23</td>
<td>52.41</td>
<td>64.48</td>
<td>85.35</td>
<td>60.12</td>
<td>78.96</td>
<td>89.11</td>
</tr>
<tr>
<td>SEIZE† + Ours</td>
<td></td>
<td></td>
<td>21.16</td>
<td>21.89</td>
<td>23.76</td>
<td>24.62</td>
<td>30.15</td>
<td>59.71</td>
<td>72.60</td>
<td>89.81</td>
<td>66.72</td>
<td>84.94</td>
<td>93.45</td>
</tr>
<tr>
<td rowspan="9">ViT-L/14</td>
<td>Pic2Word</td>
<td>CVPR23</td>
<td></td>
<td>8.72</td>
<td>9.51</td>
<td>10.64</td>
<td>11.29</td>
<td>23.9</td>
<td>51.7</td>
<td>65.3</td>
<td>87.8</td>
<td>-</td>
<td>-</td>
<td></td>
</tr>
<tr>
<td>SEARLE</td>
<td>ICCV23</td>
<td></td>
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
<td>LinCIR</td>
<td>CVPR24</td>
<td></td>
<td>12.59</td>
<td>13.58</td>
<td>15.00</td>
<td>15.85</td>
<td>25.04</td>
<td>53.25</td>
<td>66.68</td>
<td>-</td>
<td>57.11</td>
<td>77.37</td>
<td>88.89</td>
</tr>
<tr>
<td>CREeVLt</td>
<td>ICLR24</td>
<td></td>
<td>16.54</td>
<td>17.42</td>
<td>19.27</td>
<td>20.22</td>
<td>21.28</td>
<td>47.47</td>
<td>60.6</td>
<td>83.4</td>
<td>54.5</td>
<td>75.28</td>
<td>87.88</td>
</tr>
<tr>
<td>LDRE</td>
<td>SIGIR24</td>
<td></td>
<td>23.35</td>
<td>24.03</td>
<td>26.44</td>
<td>27.5</td>
<td>26.53</td>
<td>55.57</td>
<td>67.54</td>
<td>88.5</td>
<td>60.43</td>
<td>80.31</td>
<td>89.9</td>
</tr>
<tr>
<td>SEIZE</td>
<td>ACMMM24</td>
<td></td>
<td>24.71</td>
<td>25.52</td>
<td>27.99</td>
<td>29.03</td>
<td>28.43</td>
<td>56.53</td>
<td>69.88</td>
<td>88.17</td>
<td>66.43</td>
<td>84.68</td>
<td>92.96</td>
</tr>
<tr>
<td>OSrCIR</td>
<td>CVPR25</td>
<td></td>
<td>23.87</td>
<td>25.33</td>
<td>27.84</td>
<td>28.97</td>
<td>29.45</td>
<td>57.68</td>
<td>69.86</td>
<td></td>
<td>62.12</td>
<td>81.92</td>
<td>91.10</td>
</tr>
<tr>
<td>LDRE + IP-CIR</td>
<td>CVPR25</td>
<td></td>
<td>26.43</td>
<td>27.41</td>
<td>29.87</td>
<td>31.07</td>
<td>29.76</td>
<td>58.82</td>
<td>71.21</td>
<td>90.41</td>
<td>62.48</td>
<td>81.64</td>
<td>90.89</td>
</tr>
<tr>
<td>CIReVL† + Ours</td>
<td></td>
<td></td>
<td>20.32</td>
<td>20.92</td>
<td>22.81</td>
<td>23.71</td>
<td>25.35</td>
<td>52.92</td>
<td>66.41</td>
<td>86.89</td>
<td>60.75</td>
<td>80.77</td>
<td>90.92</td>
</tr>
<tr>
<td rowspan="8">ViT-G/14</td>
<td>SEIZE† + Ours</td>
<td></td>
<td></td>
<td>28.94</td>
<td>29.65</td>
<td>32.04</td>
<td>33.03</td>
<td>30.72</td>
<td>59.78</td>
<td>71.13</td>
<td>88.68</td>
<td>67.21</td>
<td>84.96</td>
<td>93.04</td>
</tr>
<tr>
<td>LinCIR</td>
<td>CVPR24</td>
<td></td>
<td>19.71</td>
<td>21.01</td>
<td>23.13</td>
<td>24.18</td>
<td>35.25</td>
<td>64.72</td>
<td>76.05</td>
<td>-</td>
<td>63.35</td>
<td>82.22</td>
<td>91.98</td>
</tr>
<tr>
<td>CIReVL†</td>
<td>ICLR24</td>
<td></td>
<td>26.47</td>
<td>27.46</td>
<td>29.91</td>
<td>30.86</td>
<td>30.7</td>
<td>59.66</td>
<td>70.89</td>
<td>89.86</td>
<td>63.54</td>
<td>82.02</td>
<td>91.52</td>
</tr>
<tr>
<td>LDRE ViT-G/14 SEIZE†</td>
<td>SIGIR24 ACMMM24</td>
<td></td>
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
<td>OSrCIR</td>
<td>CVPR25</td>
<td></td>
<td>30.47</td>
<td>31.14</td>
<td>39.67</td>
<td>40.61</td>
<td>40.87</td>
<td>69.52</td>
<td>78.94</td>
<td>92.27</td>
<td>75.04</td>
<td>90.31</td>
<td>96.02</td>
</tr>
<tr>
<td>LDRE + IP-CIR</td>
<td>CVPR25</td>
<td></td>
<td>32.75</td>
<td>34.26</td>
<td>36.86</td>
<td>38.03</td>
<td>39.25</td>
<td>70.07</td>
<td>80.00</td>
<td>94.89</td>
<td>69.95</td>
<td>86.87</td>
<td>93.55</td>
</tr>
<tr>
<td>CIReVL† + Ours</td>
<td></td>
<td></td>
<td>32.32</td>
<td>33.49</td>
<td>35.98</td>
<td>36.81</td>
<td>35.04</td>
<td>65.06</td>
<td>75.98</td>
<td>91.57</td>
<td>65.57</td>
<td></td>
<td>94.22</td>
</tr>
<tr>
<td>SEIZE† + Ours</td>
<td></td>
<td></td>
<td>39.08</td>
<td>39.93</td>
<td>42.53</td>
<td>43.34</td>
<td>42.84</td>
<td>72.29</td>
<td>80.82</td>
<td>93.28</td>
<td>74.82</td>
<td>89.95</td>
<td>96.00</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Consistent Improvements:** AdaST consistently improves the performance of its base models (CIReVL and SEIZE) across all `CLIP` backbones (ViT-B/32, ViT-L/14, ViT-G/14) and all metrics (mAP@K for CIRCO, Recall@K and RecallSubset@K for CIRR). This highlights its `plug-and-play` nature and general effectiveness.
*   **Scalability with Larger Backbones:** The improvements are particularly significant with larger backbones. For instance, on CIRCO with `ViT-G/14`, `SEIZE† + Ours` achieves `39.08 mAP@5`, a substantial gain over `SEIZE†`'s `35.61`. `CIReVL† + Ours` (32.32 mAP@5) also significantly surpasses `CIReVL†` (26.47 mAP@5).
*   **State-of-the-Art (SOTA) Performance:** For ViT-G/14, `SEIZE† + Ours` achieves `39.08 mAP@5` on CIRCO, surpassing all other methods including `LDRE + IP-CIR` (32.75). Similarly, `SEIZE† + Ours` leads in `Recall@1` for CIRR (`42.84`) and `RecallSubset@1` (`74.82`). This indicates AdaST's ability to extract and utilize visual cues effectively from the reference image, especially in multi-target scenarios like CIRCO.
*   **Comparison with Generation-based Methods:** The paper explicitly notes that on CIRCO, AdaST with `ViT-G/14` significantly improves over `CIReVL` by $+5.85 mAP@5$, which is much larger than the $+1.63 mAP@5$ gain achieved by `LDRE + IP-CIR` (a generation-based method) over `LDRE`. This demonstrates AdaST's superior accuracy even against methods that synthesize images.

### Qualitative Comparison (Figure 3)
The qualitative results further illustrate AdaST's strengths:

![Figure 3: Qualitative comparison between CIReVL and our method on three benchmarks (CIRCO, CIRR, and Fashion-IQ). Given a reference image and an instruction, reference and target captions are generated, and the top-5 retrieved images from each method are shown, with ground-truth targets highlighted in green. Our method leverages visual features more effectively, enabling accurate retrieval even when the target caption is underspecified, by exploiting fine-grained details such as dog breeds or dress shapes.](images/3.jpg)
*该图像是一个多模态查询结果的比较图，展示了CIReVL与我们的方法在CIRCO、CIRR和Fashion-IQ三个基准上的表现。每个查询下方显示了参考图像、目标说明及各自的五个检索结果，绿色框突出显示了真实目标。我们的算法通过更有效地利用视觉特征，实现了更精确的检索。*

Figure 3: Qualitative comparison between CIReVL and our method on three benchmarks (CIRCO, CIRR, and Fashion-IQ). Given a reference image and an instruction, reference and target captions are generated, and the top-5 retrieved images from each method are shown, with ground-truth targets highlighted in green. Our method leverages visual features more effectively, enabling accurate retrieval even when the target caption is underspecified, by exploiting fine-grained details such as dog breeds or dress shapes.

**Analysis:**
*   **CIRCO Example:** Given a reference image of a truck and the instruction "add a quad bike", `CIReVL` (which relies primarily on the target caption) struggles, often missing one of the key objects or returning the reference image itself. In contrast, AdaST successfully retrieves images where a quad bike is present on or next to the truck, indicating its better integration of visual and textual cues.
*   **CIRR Example:** For an instruction like "change the dog's collar to blue" where the target caption might omit the dog's breed, `CIReVL` retrieves various dog breeds. AdaST, by leveraging visual evidence from the reference image (a Border Terrier), accurately returns images of the same breed, demonstrating its capability to preserve fine-grained visual details.
*   **Fashion-IQ Example:** For an instruction like "green A-line dress", AdaST captures subtle visual information like texture and silhouette from the reference image that is hard to express purely in text. This leads to more faithful matches compared to `CIReVL`.

### Fashion-IQ Benchmark
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th colspan="3">Type</th>
<th colspan="2">Shirt</th>
<th colspan="2">Dress</th>
<th colspan="2">Toptee</th>
<th colspan="2">Average</th>
</tr>
<tr>
<th>Backbone</th>
<th>Method</th>
<th></th>
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
<td rowspan="10">ViT-G/14</td>
<td>Pic2Word</td>
<td>CVPR23</td>
<td>33.17</td>
<td>50.39</td>
<td>25.43</td>
<td>47.65</td>
<td>35.24</td>
<td>57.62</td>
<td>31.28</td>
<td>51.89</td>
</tr>
<tr>
<td>SEARLE</td>
<td>ICCV23</td>
<td>36.46</td>
<td>55.35</td>
<td>28.16</td>
<td>50.32</td>
<td>39.83</td>
<td>61.45</td>
<td>34.81</td>
<td>55.71</td>
</tr>
<tr>
<td>LinCIR</td>
<td>CVPR24</td>
<td>46.76</td>
<td>65.11</td>
<td>38.08</td>
<td>60.88</td>
<td>50.48</td>
<td>71.09</td>
<td>45.11</td>
<td>65.69</td>
</tr>
<tr>
<td>CIReVL†</td>
<td>ICLR24</td>
<td>35.13</td>
<td>52.65</td>
<td>27.52</td>
<td>49.03</td>
<td>37.33</td>
<td>58.75</td>
<td>33.33</td>
<td>53.48</td>
</tr>
<tr>
<td>LDRE</td>
<td>SIGIR24</td>
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
<td>SEIZE</td>
<td>ACMMM24</td>
<td>39.50</td>
<td>57.65</td>
<td>33.37</td>
<td>55.88</td>
<td>41.66</td>
<td>64.20</td>
<td>38.12</td>
<td>59.24</td>
</tr>
<tr>
<td>OSrCIR</td>
<td>CVPR25</td>
<td>38.65</td>
<td>54.71</td>
<td>33.02</td>
<td>54.78</td>
<td>41.04</td>
<td>61.83</td>
<td>37.57</td>
<td>57.11</td>
</tr>
<tr>
<td>LinCIR + IP-CIR</td>
<td>CVPR25</td>
<td>48.04</td>
<td>66.68</td>
<td>39.02</td>
<td>61.03</td>
<td>50.18</td>
<td>71.14</td>
<td>45.74</td>
<td>66.28</td>
</tr>
<tr>
<td>CIReVL†+Ours</td>
<td></td>
<td>40.38</td>
<td>59.08</td>
<td>36.49</td>
<td>58.70</td>
<td>43.65</td>
<td>64.10</td>
<td>40.17</td>
<td>60.63</td>
</tr>
<tr>
<td>SEIZE†+Ours LinCIR+Ours</td>
<td></td>
<td>44.36</td>
<td>62.22</td>
<td>40.21</td>
<td>62.12</td>
<td>48.55</td>
<td>69.30</td>
<td>44.37</td>
<td>64.55</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Consistent Gains:** AdaST again shows consistent improvements over `CIReVL` and `SEIZE` across all three garment categories (Shirt, Dress, Toptee) and average `Recall@10` and `Recall@50` metrics on the Fashion-IQ dataset. For example, $CIReVL†+Ours$ achieves an average `R@10` of `40.17` compared to `CIReVL†`'s `33.33`.
*   **Enhanced Performance for Strong Baselines:** Even when combined with `LinCIR`, a strong training-based baseline, AdaST ($LinCIR+Ours$) further boosts performance (e.g., average `R@10` of `48.16` vs. `LinCIR`'s `45.11`). This indicates that AdaST's feature transformation and adaptive fusion mechanisms provide unique benefits that complement even already robust models.
*   **Fine-grained Details:** The results on Fashion-IQ, a dataset requiring fine-grained understanding of fashion attributes, confirm that AdaST effectively leverages reference image cues to capture details like texture and silhouette, which are hard to convey through text alone.

### Inference Time
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th colspan="2">Fashion-IQ Dress (DB size = 4K)</th>
<th colspan="2">CIRCO (DB size = 120K)</th>
</tr>
<tr>
<th>time</th>
<th>+∆t</th>
<th>time</th>
<th>+∆t</th>
</tr>
</thead>
<tbody>
<tr>
<td>CIReVL</td>
<td>1.76s</td>
<td>−</td>
<td>2.16s</td>
<td>−</td>
</tr>
<tr>
<td>+Ours</td>
<td>1.87s</td>
<td>0.11s</td>
<td>2.77s</td>
<td>0.61s</td>
</tr>
<tr>
<td>+IP-CIR</td>
<td>119.82s</td>
<td>118.06s</td>
<td>120.84s</td>
<td>118.68s</td>
</tr>
<tr>
<td>SEIZE</td>
<td>26.05s</td>
<td>−</td>
<td>26.25s</td>
<td>−</td>
</tr>
<tr>
<td>+Ours</td>
<td>26.17s</td>
<td>0.12s</td>
<td>26.89s</td>
<td>0.64s</td>
</tr>
<tr>
<td>+IP-CIR</td>
<td>144.19s</td>
<td>118.14s</td>
<td>145.21s</td>
<td>118.96s</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Negligible Overhead:** AdaST introduces only a marginal increase in inference time. On `Fashion-IQ Dress` (database size 4K), adding AdaST to `CIReVL` or `SEIZE` incurs only `0.11-0.12 seconds` of additional computation. On `CIRCO` (database size 120K), the overhead is `0.61-0.64 seconds`. This demonstrates AdaST's efficiency and lightweight design.
*   **Massive Speedup vs. Generation-based Methods:** In stark contrast, `IP-CIR` (a generation-based method) requires over `118 seconds` of additional processing time on both datasets. This means AdaST is `~186x` faster than IP-CIR (118s / 0.61s or 118s / 0.64s approx).
*   **Practicality:** This efficiency is a critical advantage, making AdaST suitable for real-time interactive retrieval applications where generation-based methods are infeasible due to their high latency.

## 6.2. Ablation Studies / Parameter Analysis

### Component Analysis
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>Method</th>
<th>Proxy</th>
<th>Scaling</th>
<th>Gating</th>
<th>mAP@5</th>
<th>∆</th>
</tr>
</thead>
<tbody>
<tr>
<td>CIReVL</td>
<td></td>
<td></td>
<td></td>
<td>26.47</td>
<td>-</td>
</tr>
<tr>
<td>CIReVL + Proxy</td>
<td>✓</td>
<td></td>
<td></td>
<td>27.42</td>
<td>+0.95</td>
</tr>
<tr>
<td>CIReVL + Proxy + Scaling</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td>29.41</td>
<td>+2.94</td>
</tr>
<tr>
<td>CIReVL + Proxy + Scaling + Gating</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>32.32</td>
<td>+5.85</td>
</tr>
</tbody>
</table>

**Analysis:**
*   **Proxy Embedding Contribution:** Using only the `proxy embedding` (with a naive $\alpha=1$) on top of `CIReVL` (26.47 mAP@5) improves performance to `27.42 mAP@5` ($+0.95$). This shows that feature-level transformation itself provides a benefit by incorporating visual information, even without optimization.
*   **Optimal Scaling Importance:** Incorporating the `optimal scaling` strategy further boosts performance significantly to `29.41 mAP@5` ($+2.94$ over baseline). This validates the necessity of optimizing the scaling factor $\alpha$ to ensure the proxy embedding adequately reflects the intended modification.
*   **Gating Function's Crucial Role:** The `adaptive gating function` provides the largest individual performance gain. When added to the proxy with optimal scaling, it pushes the `mAP@5` to `32.32` ($+5.85$ over baseline). This confirms that adaptively regulating the contribution of proxy similarity based on semantic cues is crucial for suppressing misleading visual information and enhancing retrieval accuracy. It implies that sometimes the proxy might be "unreliable," and the gate helps filter its influence.

### Analysis for the Optimal Scaling of Semantic Transformation
To understand the behavior of the `proxy embedding` and the `optimal scaling` mechanism, two additional experiments are conducted.

1.  **Retrieval of Reference Image (Figure 4):**
    The following is the image showing retrieval performance with the reference image as ground truth:

    ![Figure 4: Retrieval performance with the reference image as ground truth.](images/4.jpg)
    *该图像是图表，展示了不同$\beta$值下，参考图像的检索性能（R@1）。绿线表示当$\alpha$为优化值$\alpha^*$时的表现，图中标注了各数据点的具体值。红色和蓝色虚线分别表示最佳效果98.00和文本基线6.50。整体趋势显示，随着$\beta$的增加，检索性能逐渐下降。*

    Figure 4: Retrieval performance with the reference image as ground truth.

    **Analysis:**
    *   This experiment assesses how much the proxy embedding diverges from the original reference. The goal is to retrieve the *reference image itself* using the generated proxy as a query.
    *   When the semantic shift is `naively transferred` (presumably with $\alpha=1$ or a fixed $\alpha$), the `R@1` is `98.0`. This high score indicates that the proxy embedding remains very close to the reference image embedding, meaning it fails to sufficiently deviate even after transformation.
    *   This finding motivates the need for an `optimal scaling` strategy that can force the proxy to move more decisively towards the target.

2.  **Ablation Study of Optimal Scaling (Figure 5):**
    The following is the image showing the ablation study of optimal scaling:

    ![Figure 5: Ablation study of optimal scaling.](images/5.jpg)
    *该图像是一个图表，展示了 `eta` 对 CIRCO - mAP@5 指标的影响。随着 `eta` 值的变化，图中显示了不同 `eta` 值下的性能结果，最优值达到了 29.41，红色和绿色水平线分别代表基准值 27.42 和 26.47。*

    Figure 5: Ablation study of optimal scaling.

    **Analysis:**
    *   This figure shows `mAP@5` on CIRCO as a function of the parameter $\beta$ (weighting coefficient in the optimal scaling objective).
    *   When $\alpha$ is fixed to 1 ($fixed alpha=1$ line), the performance improves over the baseline but is limited. This aligns with the "naive transfer" issue observed in Figure 4.
    *   Applying the `proposed scaling strategy` (with varying $\beta$) leads to a consistent increase in performance as $\beta$ increases, up to a certain threshold. This shows that explicitly penalizing similarity to the reference image during $\alpha$ optimization (i.e., making the proxy sufficiently distinct from the reference) improves results.
    *   Beyond a certain $\beta$ value, performance drops sharply. This suggests that too high a $\beta$ causes the proxy embedding to become `overly detached` from the reference image. The target image *still preserves essential information from the reference*, so completely abandoning the reference's characteristics leads to degraded performance.
    *   The $alpha=0$ point (direct use of the reference image for retrieval) yields significantly lower performance (`22.75 mAP@5`), demonstrating that the `semantic transformation` (even without optimal scaling) is crucial for guiding the proxy embedding towards the target.

        In summary, the ablation studies confirm that each component of AdaST (proxy feature, optimal scaling, and adaptive gating) contributes positively to the overall performance, with optimal scaling and adaptive gating being particularly critical for achieving robust and accurate `composed image retrieval`.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
This paper introduces `Adaptive Semantic Transformation (AdaST)`, a novel training-free method for `Composed Image Retrieval (CIR)` that achieves state-of-the-art performance in both accuracy and efficiency. AdaST addresses the fundamental dilemma in existing ZS-CIR methods: the trade-off between the efficiency of text-based approaches (which lose visual detail) and the visual fidelity of generation-based methods (which are computationally expensive). AdaST resolves this by transforming reference image features directly in the latent space of pre-trained `Vision-Language Models (VLMs)`, guided by textual instructions derived from `LLMs`. This feature-level manipulation efficiently preserves fine-grained visual information without costly image synthesis. The method further enhances robustness through an `adaptive similarity mechanism` that intelligently weights visual and textual cues, preventing misleading matches. Extensive experiments on CIRCO, CIRR, and Fashion-IQ benchmarks demonstrate AdaST's superior accuracy and its remarkable efficiency, being substantially faster than generation-based alternatives. Its modular, plug-and-play design allows for easy integration, making it a highly practical and promising solution for multi-modal retrieval.

## 7.2. Limitations & Future Work
The paper doesn't explicitly list a "Limitations" section, but some can be inferred from the approach and discussion:

*   **Reliance on LLM/Captioning Model Quality:** AdaST's performance is inherently dependent on the quality of the `reference captions` generated by models like `BLIP-2` and the `target captions` generated by `LLMs` like `GPT-4o`. If these models produce inaccurate or underspecified captions, the semantic shift $\Delta T$ and subsequent feature transformation may be suboptimal.
*   **Generalizability of Feature Space:** While `CLIP`'s feature space is powerful, there might be inherent limitations in how accurately a simple vector addition ($f_Ir + alpha * DeltaT$) can represent complex, non-linear visual transformations. The optimal scaling and gating mechanisms attempt to mitigate this but don't fundamentally change the linearity of the transformation within the VLM feature space.
*   **Interpretability of $\Delta T$:** The semantic shift $\Delta T$ is a high-dimensional vector, and its precise meaning in terms of visual attributes can be abstract. While effective, the direct mapping from text-space vector difference to image-space transformation might not always perfectly align with human intuition for all types of modifications.
*   **Computational Cost of LLM Inference:** Although AdaST is significantly faster than image generation, the use of powerful `LLMs` (like `GPT-4o`) for caption generation still incurs some computational cost, especially for large-scale query processing, although it's typically much less than pixel-level image generation.

    The paper suggests that its `feature-space transformation approach offers a promising and efficient direction for the future of multi-modal retrieval and understanding`. This implies future work could involve:
*   Exploring more sophisticated feature-space manipulation techniques beyond simple vector addition and scaling.
*   Investigating how to make the text guidance generation process even more robust and less reliant on external, potentially costly, LLM APIs.
*   Applying this methodology to other multi-modal tasks that involve fine-grained editing or compositional reasoning.
*   Further optimizing the adaptive weighting mechanism for different types of modifications or datasets.

## 7.3. Personal Insights & Critique
This paper presents an elegant and pragmatic solution to the `efficiency-fidelity` trade-off in `Zero-Shot Composed Image Retrieval`. My personal insights include:

*   **Elegance of Feature-Space Transformation:** The core idea of operating directly in the `VLM`'s feature space is highly appealing. It leverages the rich, semantically aligned representations learned by models like `CLIP` without the heavy computational burden of generative models. This feels like a "sweet spot" in the design space for CIR.
*   **Cleverness of Adaptive Gating:** The `adaptive similarity fusion` with its `gating mechanism` is a critical innovation. It acknowledges that not all text-guided transformations are equally reliable. By dynamically regulating the proxy's influence based on semantic alignment, the model becomes more robust and avoids being misled by purely visual similarities when the semantic instruction is ambiguous or the transformation less certain. This is a subtle but powerful enhancement.
*   **Practical Value:** The high accuracy combined with remarkable efficiency makes AdaST immediately practical for real-world applications such as e-commerce product search or interactive content editing tools, where response time is crucial. Its `plug-and-play` nature further lowers the barrier to adoption.
*   **Dependence on Pre-trained Models:** While training-free and powerful, the method's effectiveness is heavily tied to the quality of the underlying `VLM` (e.g., `CLIP`) and `LLM` (`GPT-4o`). As these foundational models improve, AdaST's performance would likely also improve, but it inherits any biases or limitations present in them.
*   **Potential for Generalization:** The `feature-level transformation` paradigm could potentially be extended to other `multi-modal editing` tasks, where specific attributes need to be modified in a visually consistent manner without full image regeneration. This could include tasks like scene editing or even video editing.
*   **Unverified Assumptions:** The paper assumes that the `semantic shift vector` $\Delta T$ derived from text embeddings in the `VLM`'s space directly and meaningfully corresponds to a visual transformation in the image embedding space. While the results empirically validate this to a large extent, the precise nature of this correspondence and its limitations for highly complex or abstract modifications could be an area for deeper theoretical investigation. For example, how well does adding $\Delta T$ for "change blue to red" truly modify only the color attribute in the image feature, without affecting other attributes, for all images? The optimal scaling and gating attempt to make this more robust, but the underlying assumption remains.

    Overall, AdaST is a significant step forward for `zero-shot composed image retrieval`, offering a well-thought-out, efficient, and highly effective approach that balances key trade-offs in the field.