# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is "Improving Context Understanding in Multimodal Large Language Models via Multimodal Composition Learning". The research focuses on enhancing the ability of Multimodal Large Language Models (MLLMs) to understand and process complex scenarios involving both visual and textual inputs, specifically through a novel training approach called Multimodal Composition Learning (MCL).

## 1.2. Authors
The authors of the paper are **Wei Li**, **Hehe Fan**, **Yongkang Wong**, **Yi Yang**, and **Mohan Kankanhalli**.
*   **Wei Li** and **Hehe Fan** are affiliated with institution 1 (likely the National University of Singapore or similar, based on the Acknowledgments mentioning "National University of Singapore" and "NUS" in the grant text, though specific affiliations are listed as numbers in the header).
*   **Yongkang Wong** and **Mohan Kankanhalli** are affiliated with institution 2.
*   **Yi Yang** is affiliated with institution 1.
*   **Mohan Kankanhalli** is a well-known researcher in the field of multimedia computing.

## 1.3. Journal/Conference
The provided text does not explicitly state the journal or conference name. However, the formatting, citation style (e.g., references to CVPR, ICCV, NeurIPS), and the nature of the contribution suggest it is a top-tier computer vision or multimodal learning conference paper (e.g., CVPR, ICCV, or ECCV). The references include papers from 2023 and 2024, indicating this is a very recent work, likely published in 2024.

## 1.4. Publication Year
Based on the references cited within the paper (e.g., Li et al., 2024; Yang et al., 2024), the publication year is **2024**.

## 1.5. Abstract
The paper addresses the limitations of previous efforts using frozen Large Language Models (LLMs) for visual understanding, specifically noting that simple image captioning or retrieval tasks are insufficient for complex multimodal scenarios. To solve this, the authors introduce **Multimodal Composition Learning (MCL)**, a method designed to map and align vision and language inputs more effectively. They propose two specific tasks: **Multimodal-Context Captioning (MC-Cap)** and **Multimodal-Context: Retrieval (MC-Ret)**. These tasks guide a frozen LLM to better comprehend the context of vision and language inputs, thereby improving its proficiency in generating accurate text or visual representations. Extensive experiments on retrieval tasks (zero-shot composed image retrieval, visual storytelling, visual dialog) and text generation tasks (visual question answering) demonstrate the method's effectiveness.

## 1.6. Original Source Link
The original source is provided as an uploaded file link: `uploaded://1634e285-c66f-4bf0-985d-7d3b54fde46f`.
The PDF link is: `/files/papers/69dc49ae5127ff45a9dee91d/paper.pdf`.
The publication status appears to be a preprint or recently published conference paper given the "uploaded://" prefix and recent citations.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the **deficiency of cross-modal interaction** in existing Multimodal Large Language Models (MLLMs). Many recent approaches utilize "frozen" Large Language Models (LLMs)—models whose weights are kept fixed during training—for visual understanding. These models typically learn a simple mapping between vision and language through basic tasks like image captioning (describing an image) or image-text retrieval (finding an image for a text).
While these models show versatility, the authors argue that these tasks act primarily as "modality translation." They lack the ability to deeply compose information from both modalities simultaneously. This limitation leads to subpar performance in complex benchmarks, such as **zero-shot composed image retrieval**, where the model must retrieve a target image based on a reference image combined with a text modification (e.g., "find the same chair but in red"). The motivation is to move beyond simple translation to true **multimodal composition**, enabling the model to synthesize information from visual and textual contexts to solve more intricate reasoning tasks.

## 2.2. Main Contributions / Findings
The paper makes four primary contributions:
1.  **Multimodal Composition Learning (MCL):** A novel method for vision-language mapping that enables a frozen LLM to perform accurate image retrieval and text generation within complex multimodal contexts. Unlike previous methods, MCL focuses on synthesizing information from both modalities rather than just translating between them.
2.  **MultiModal Composition (MMC) Dataset:** A large-scale dataset constructed by automatically augmenting existing web-collected image-text pairs using an LLM (Llama2). It contains 2.7 million tuples of `(reference image, reference caption, text condition, target caption)`, addressing the data scarcity issue for composition learning.
3.  **Stacking Retrieval Mechanism:** A new architectural mechanism to extract diverse multimodal information from the LLM's context. It uses multiple special retrieval tokens ([RET]) in a "stacking" order (independent of each other) rather than a "sequential" order, preventing the tokens from focusing on redundant information.
4.  **Experimental Validation:** The authors demonstrate that MCL achieves state-of-the-art or competitive results on four zero-shot multimodal context understanding tasks, including composed image retrieval, visual storytelling, visual dialog, and visual question answering.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must be familiar with several key concepts in machine learning and computer vision:

*   **Large Language Models (LLMs):** These are deep learning models (like GPT-4, Llama, OPT) trained on vast amounts of text data. They are generative models, meaning they predict the next token in a sequence. They possess strong reasoning and contextual understanding capabilities for text.
*   **Frozen LLMs:** In this context, "frozen" means the weights of the pre-trained LLM are not updated during the training of the multimodal system. Only a small adapter network is trained. This preserves the LLM's original knowledge and reduces computational cost.
*   **CLIP (Contrastive Language-Image Pre-training):** A model developed by OpenAI that learns visual concepts from natural language supervision. It consists of an image encoder and a text encoder trained jointly to maximize the cosine similarity of matching image-text pairs. It is widely used for zero-shot image classification and retrieval.
*   **Zero-Shot Learning:** The ability of a model to solve a task it was not explicitly trained on, typically by leveraging its pre-trained knowledge and generalization capabilities.
*   **Composed Image Retrieval (CIR):** A retrieval task where the query consists of a reference image and a relative text caption (e.g., "change the color to blue"). The goal is to retrieve a target image that matches the composition of the reference image and the text modification.

## 3.2. Previous Works
The authors summarize three main areas of related work:

1.  **Vision-Language Mapping:** Previous efforts (e.g., Flamingo, BLIP-2, FROMAGe) have connected visual modalities to frozen LLMs. Some map visual features to the LLM space for captioning/generation (Vision-to-Language). Others map LLM representations back to visual feature spaces (e.g., CLIP space) for retrieval (Language-to-Vision). These methods typically rely on simple tasks like image captioning or retrieval, which the authors argue are insufficient for deep composition.
2.  **Composed Image Retrieval (CIR):** Traditional CIR methods rely on human-labeled triplets (reference image, text, target image), which are expensive to collect. Recent works like Pic2Word and SEARLE attempt to solve this in a zero-shot setting by mapping images to pseudo-text tokens within the CLIP space. Other works generate synthetic triplets for training.
3.  **Multimodal Data Augmentation with LLMs:** Recent trends use LLMs to refine or generate training data for vision tasks. The paper cites works that generate triplets for image editing tasks.

**Crucial Background Formula: InfoNCE Loss**
To understand the retrieval objectives in this paper, one must understand the **InfoNCE loss**, used in contrastive learning (like CLIP). This loss function maximizes the agreement between positive pairs (matching image and text) while minimizing it for negative pairs (non-matching). The formula is:

$$
\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i, z_i^+) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i, z_j) / \tau)}
$$

Where:
*   $z_i$ is the representation of the anchor (e.g., image).
*   $z_i^+$ is the representation of the positive sample (e.g., matching text).
*   $z_j$ represents the representations of all samples (positive and negative) in the batch.
*   $\text{sim}(\cdot, \cdot)$ is a similarity metric (usually cosine similarity).
*   $\tau$ is a temperature parameter that scales the logits.
*   $N$ is the batch size.

    This loss is fundamental to the retrieval objectives ($\mathcal{L}_{\text{Ret}}$ and $\mathcal{L}_{\text{MC-Ret}}$) described in the paper.

## 3.3. Technological Evolution
The field has evolved from training separate encoders for vision and language to integrating them into a unified LLM space.
*   **Early Stage:** Dual-encoder models (like CLIP) where image and text are mapped to a joint embedding space.
*   **Current Stage:** Fusion-encoder models (like Flamingo, FROMAGe) where visual features are injected into a frozen LLM. This allows for more complex generative tasks but has struggled with compositional retrieval.
*   **This Paper's Position:** This work represents a refinement of the "Fusion-encoder" stage. It acknowledges that simply feeding images into an LLM isn't enough; the *training objective* must explicitly teach the model how to *compose* multimodal contexts (MCL) rather than just translate between them.

## 3.4. Differentiation Analysis
The core difference between this paper and previous works (like FROMAGe) lies in the **training objective** and the **data**.
*   **vs. FROMAGe:** FROMAGe trains using standard image captioning and image-text retrieval. The paper shows FROMAGe performs poorly on composed retrieval because it treats the task as simple translation. MCL introduces specific composition tasks (MC-Cap, MC-Ret) and a synthetic dataset (MMC) to force the model to learn to combine image and text conditions.
*   **vs. CLIP-based CIR (Pic2Word, SEARLE):** These methods operate primarily within the CLIP text encoder space. MCL operates within the LLM space. The authors argue that LLMs have superior reasoning capabilities (understanding word order, logic) compared to CLIP text encoders, making them better suited for complex composition.

# 4. Methodology

## 4.1. Principles
The core principle of **Multimodal Composition Learning (MCL)** is to enhance the alignment between the vision and language modalities by training the model to perform **multimodal composition** tasks. Instead of simply translating an image to text (captioning) or checking if text matches an image (retrieval), the model is trained to take a reference image and a text condition (e.g., "with a toy mouse") and generate a representation of a *target* that satisfies both.
The theoretical intuition is that by forcing the model to generate a target description or feature based on a composite query, the model learns to deeply interact with and fuse the visual and textual features, rather than processing them independently.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Data Generation: The MMC Dataset
The first step of the methodology is data generation. Since collecting real-world data for composed image retrieval (triplets of reference image, modification, target image) is expensive and hard to scale, the authors propose generating a synthetic dataset called **MMC (MultiModal Composition)**.

The process leverages an off-the-shelf LLM (Llama2). Given a web-collected pair consisting of a reference image $I_{\text{ref}}$ and its caption $T_{\text{refc}}$:
1.  The reference caption $T_{\text{refc}}$ is fed into the LLM.
2.  The LLM is prompted to generate a **text condition** $T_{\text{con}}$. This condition acts as an editing instruction (e.g., "wearing a hat").
3.  The LLM is then tasked to compose the reference caption and the text condition to generate a **target caption** $T_{\text{tgtc}}$ (e.g., "a person wearing a hat").
4.  The result is a tuple $\langle I_{\text{ref}}, T_{\text{refc}}, T_{\text{con}}, T_{\text{tgtc}} \rangle$.

    Crucially, the authors do not attempt to generate the actual target image. Instead, they use the **CLIP feature of the target caption** as a form of visual supervision for training. This allows for large-scale (2.7M tuples) training without requiring image generation models.

The following figure illustrates the data generation and overall framework:

![该图像是示意图，展示了多模态组合学习（MCL）的数据生成流程和模型架构。左侧部分描述了图像与文本的配对，包括生成参考标题和目标标题的过程。右侧阐述了多模态上下文的构建及其在多模态上下文检索（MC-Ret）和多模态上下文标题生成（MC-Cap）中的应用，强调了LLM在处理复杂多模态输入时的能力提升。](images/2.jpg)
*该图像是示意图，展示了多模态组合学习（MCL）的数据生成流程和模型架构。左侧部分描述了图像与文本的配对，包括生成参考标题和目标标题的过程。右侧阐述了多模态上下文的构建及其在多模态上下文检索（MC-Ret）和多模态上下文标题生成（MC-Cap）中的应用，强调了LLM在处理复杂多模态输入时的能力提升。*

### 4.2.2. Mapping Visual Input to the LLM
To process visual input, the model uses a **frozen CLIP ViT-L/14** image encoder to extract features from an input image $I$. These features are then mapped into the LLM's embedding space using a learnable linear projection layer (an adapter), denoted as $f_{\text{map}}$.

Mathematically, given an image $I$, the visual feature vectors $\mathbf{V}$ are:
$$
\mathbf{V} = [v_0, v_1, ..., v_n] = f_{\text{map}}(E_{\text{image}}(I))
$$
Where:
*   $E_{\text{image}}$ is the frozen CLIP image encoder.
*   $f_{\text{map}}$ is the learnable linear mapping layer.
*   $\mathbf{V}$ is a sequence of visual vectors compatible with the LLM's token embeddings.

### 4.2.3. Multimodal-Context Captioning (MC-Cap)
This objective trains the model to generate text. It is an extension of the standard image captioning objective but incorporates the "text condition" to force composition.

**Standard Captioning Objective ($\mathcal{L}_{\text{Cap}}$):**
For a standard image-caption pair, the model minimizes the negative log-likelihood of predicting the next token $t_i$ given the visual vectors $\mathbf{V}$ and previous tokens $t_{<i}$.
$$
\mathcal{L}_{\{ \text{Cap} \}}(\theta_m) = - \frac{1}{|t|} \sum_{i=1}^{|t|} \log P \Big( t_i \mid \mathbf{V}, t_{<i} \Big)
$$
Where:
*   $\theta_m$ are the weights of the mapping layer.
*   $P$ is the frozen LLM.

**MC-Cap Objective ($\mathcal{L}_{\text{MC-Cap}}$):**
Using the generated MMC dataset tuples $\langle I_{\text{ref}}, T_{\text{con}}, T_{\text{tgtc}} \rangle$, the model is trained to predict the target caption tokens conditioned on the visual vectors $\mathbf{V}$, the text condition tokens $c$, and previous target caption tokens.
$$
\mathcal{L}_{\{ \text{MC-Cap} \}}(\theta_m) = - \frac{1}{|t|} \sum_{i=1}^{|t|} \log P \Big( t_i \mid \mathbf{V}, c_1, ..., c_{|c|}, t_{<i} \Big)
$$
Where:
*   $c_i$ denotes the $i$-th token of the text condition.
*   $t_i$ denotes the $i$-th token of the target caption.

    This forces the LLM to "query" the visual vectors using the text condition to derive the correct description, enhancing the visual-to-language mapping.

### 4.2.4. Multimodal-Context Retrieval (MC-Ret)
This objective trains the model to retrieve images (or rather, generate visual features for retrieval). It uses a special learnable token, `[RET]`, appended to the input sequence. The hidden state of this token is used to output a visual representation.

**Standard Retrieval Objective ($\mathcal{L}_{\text{Ret}}$):**
For a paired image and caption, the `[RET]` token is appended after the caption tokens. Its hidden state $h([\text{RET}] \mid \mathbf{T})$ is projected to the CLIP latent space via a linear layer $f_{\text{proj}}$. An InfoNCE loss aligns this projection with the CLIP visual feature of the target image.
$$
\mathcal{L}_{\{ \text{Ret} \}}([\text{RET}], \theta_p) = - \frac{1}{N} \sum_{i=1}^{N} \left( \log \frac{\exp(\sin(p_v, \boldsymbol{e}_i) / \tau)}{\sum_{j=1}^{N} \exp(\sin(p_v, \boldsymbol{e}_j) / \tau)} \right)
$$
Where:
*   `p_v = f_{\text{proj}}(h([\text{RET}] \mid \mathbf{T}))`.
*   $\mathbf{T}$ denotes caption tokens.
*   $\boldsymbol{e}_i = E_{\text{image}}(I_i)$ is the CLIP feature of the $i$-th image in the batch.
*   $\sin(\cdot, \cdot)$ denotes cosine similarity.
*   $\tau$ is the temperature parameter.

**MC-Ret Objective ($\mathcal{L}_{\text{MC-Ret}}$):**
This objective enhances the `[RET]` token to handle multimodal contexts. Given a tuple $\langle I_{\text{ref}}, T_{\text{con}}, T_{\text{tgtc}} \rangle$, the input consists of the reference image features $\mathbf{V}$ and the text condition tokens $\mathbf{T}$. The `[RET]` token is appended at the end. The goal is to make the `[RET]` token's output match the CLIP feature of the *target caption* (which serves as a proxy for the target image).
$$
\mathcal{L}_{\{ \text{MC-Ret} \}}([\text{RET}], \theta_p, \theta_m) = - \frac{1}{N} \sum_{i=1}^{N} \left( \log \frac{\exp(\sin(p_v, \mathbf{e}_i) / \tau)}{\sum_{j=1}^{N} \exp(\sin(p_v, \mathbf{e}_j) / \tau)} \right)
$$
Where:
*   `p_v = f_{\text{proj}}(h(\mathsf{\Omega}[\mathsf{RET}] \mid \mathbf{V}, \mathbf{T}))`.
*   $\mathbf{V}$ denotes the mapped visual features of the reference image $I_{\text{ref}}$.
*   $\mathbf{T}$ denotes the caption tokens of the text condition $T_{\text{con}}$.
*   $\mathbf{e}$ represents the CLIP feature of the target caption $T_{\text{tgtc}}$.

    This trains the `[RET]` token to selectively extract and compose information from the multimodal context (Image + Text) to match the target.

### 4.2.5. Stacking Retrieval Mechanism
To extract more diverse information from the context, the authors propose using multiple `[RET]` tokens.

**Sequential Order (Naive):** If tokens are appended sequentially ($[\text{RET}]_1, [\text{RET}]_2...$), the hidden state of $[\text{RET}]_i$ is influenced by $[\text{RET}]_{<i}$. This causes tokens to focus on similar, redundant content.
$$
h_i = h([\text{RET}]_i \mid \mathbf{V}, \mathbf{T}, [\text{RET}]_{<i})
$$

**Stacking Order (Proposed):** The authors modify the attention mask so that each `[RET]` token is **independent** of the others. They are all conditioned only on the context $\mathbf{V}, \mathbf{T}$.
$$
h_i = h([\text{RET}]_i \mid \mathbf{V}, \mathbf{T})
$$
The final output feature $p_v$ is a fusion of all independent retrieval token states:
$$
p_v = f_{\text{fusion}}(h_1, ..., h_r)
$$
Where $f_{\text{fusion}}$ is a fusion function (e.g., a Transformer layer with mean pooling). This "stacking" mechanism allows the model to extract different aspects of the image and text (e.g., color, shape, background) into different tokens.

### 4.2.6. Total Training Loss
The model is trained with a combination of all four objectives:
$$
\mathcal{L} = \lambda_{\{ \text{Cap} \}} (\mathcal{L}_{\{ \text{Cap} \}} + \mathcal{L}_{\{ \text{MC-Cap} \}}) + \lambda_{\{ \text{Ret} \}} (\mathcal{L}_{\{ \text{Ret} \}} + \mathcal{L}_{\{ \text{MC-Ret} \}})
$$
Where $\lambda$ are weights balancing the generation and retrieval losses.

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize several datasets to evaluate different aspects of multimodal understanding:

1.  **CIRCO, CIRR, GeneCIS:** These are **Composed Image Retrieval (CIR)** benchmarks.
    *   **CIRCO:** An open-domain benchmark. Queries consist of a reference image and a relative caption.
    *   **CIRR:** Fashion-oriented dataset for composed retrieval.
    *   **GeneCIS:** A benchmark for general conditional image similarity, testing tasks like "Focus Attribute" or "Change Object".
    *   **Why chosen:** Chosen to evaluate the core contribution of MCL: the ability to compose image and text queries for retrieval.

2.  **Visual Storytelling (VIST):** A dataset where a model must retrieve the last image in a sequence of 5 images given the story context.
    *   **Why chosen:** To evaluate "Dense Multimodal Context Understanding" (handling multiple images and texts).

3.  **Visual Dialog (VisDial):** A dataset containing an image and a dialog history about it. The task is to retrieve the image based on the dialog.
    *   **Why chosen:** To evaluate understanding of complex, conversational text contexts paired with images.

4.  **VQAv2:** A standard Visual Question Answering dataset.
    *   **Why chosen:** To evaluate text generation capabilities (answering questions) in a zero-shot setting.

**Concrete Data Example (MMC Dataset):**
The paper provides an example of the generated MMC data:
*   **Reference Image:** An orange kitten.
*   **Reference Caption:** "cute orange kitten looking up".
*   **Text Condition:** "with a toy mouse".
*   **Target Caption:** "a cute orange kitten playing with a toy mouse".

## 5.2. Evaluation Metrics
The paper uses standard retrieval and generation metrics:

1.  **Recall@K (R@K)**
    *   **Conceptual Definition:** Recall@K measures the ability of the model to find the correct item within the top $K$ retrieved results. It answers the question: "Is the correct result present in the top K list?"
    *   **Mathematical Formula:**
        $$ \text{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q) \leq K) $$
    *   **Symbol Explanation:**
        *   $|Q|$: Total number of queries.
        *   $\text{rank}(q)$: The rank of the ground truth item for query $q$.
        *   $\mathbb{I}(\cdot)$: Indicator function (1 if condition is true, 0 otherwise).
        *   $K$: The cutoff rank (e.g., 1, 5, 10).

2.  **Mean Average Precision (mAP@K)**
    *   **Conceptual Definition:** Mean Average Precision considers the rank of the correct item across all queries. Unlike Recall, which is binary, AP gives higher credit if the correct item is ranked higher (closer to the top). mAP@K is the mean of these scores, often truncated at rank K.
    *   **Mathematical Formula:**
        $$ \text{AP@K} = \frac{1}{\min(m, K)} \sum_{k=1}^{K} P(k) \times \text{rel}(k) $$
        $$ \text{mAP@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP@K}(q) $$
    *   **Symbol Explanation:**
        *   $m$: Total number of relevant documents for the query (usually 1 in CIR).
        *   `P(k)`: Precision at rank $k$.
        *   $\text{rel}(k)$: Binary indicator of relevance of the item at rank $k$.

3.  **Accuracy (Acc)**
    *   **Conceptual Definition:** For VQA, this is the percentage of generated answers that exactly match the ground truth answers (usually allowing for soft matching of synonyms).

## 5.3. Baselines
The paper compares against several representative baselines:
*   **Image-only / Text-only:** Simple CLIP features.
*   **Image + Text:** Summing CLIP image and text features.
*   **Pic2Word / SEARLE:** Zero-shot CIR methods operating in the CLIP text encoder space.
*   **CompoDiff:** A method using latent diffusion models for CIR.
*   **FROMAGe:** A prior MLLM method trained on captioning and retrieval (without composition fancier objectives). This is a critical baseline to show the value of MCL.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The experimental results strongly validate the effectiveness of MCL.
*   **Composed Image Retrieval (CIR):** MCL significantly outperforms previous zero-shot CIR methods (like Pic2Word, SEARLE) and prior MLLM methods (FROMAGe). For instance, on CIRCO, MCL (Llama2-7B) achieves 17.67 mAP@5, compared to FROMAGe's 4.00. This massive gap proves that standard captioning/retrieval objectives are insufficient for composition tasks.
*   **LLM Superiority:** The results show that using an LLM backbone (Llama2, OPT) yields better results than CLIP-based textual inversion methods. This supports the authors' claim that LLMs handle complex composition logic better than CLIP text encoders.
*   **Dense Context:** In Visual Storytelling and Visual Dialog, MCL outperforms FROMAGe and CLIP. Notably, MCL's performance *improves* as more context (more images/text) is added, whereas CLIP's performance often degrades. This demonstrates MCL's ability to effectively utilize dense multimodal information.

## 6.2. Data Presentation (Tables)

### 6.2.1. Composed Image Retrieval Results
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th rowspan="2">LLM</th>
<th colspan="4">CIRCO (mAP@K)</th>
<th colspan="4">CIRR</th>
<th rowspan="2">GeneCIS R@1 (avg)</th>
</tr>
<tr>
<th>K=5</th>
<th>K=10</th>
<th>K=25</th>
<th>K=50</th>
<th>R@1</th>
<th>R@5</th>
<th>R@50</th>
<th>Rg@1</th>
</tr>
</thead>
<tbody>
<tr>
<td>Image-only</td>
<td rowspan="6">Non-LLM</td>
<td>2.79</td>
<td>3.18</td>
<td>3.75</td>
<td>4.12</td>
<td>7.13</td>
<td>23.04</td>
<td>56.63</td>
<td>20.55</td>
<td>11.0</td>
</tr>
<tr>
<td>Text-only</td>
<td>2.50</td>
<td>2.64</td>
<td>3.11</td>
<td>3.38</td>
<td>20.55</td>
<td>44.17</td>
<td>78.94</td>
<td>60.74</td>
<td>9.1</td>
</tr>
<tr>
<td>Image + Text</td>
<td>6.37</td>
<td>7.04</td>
<td>8.11</td>
<td>8.72</td>
<td>12.27</td>
<td>35.81</td>
<td>77.04</td>
<td>33.33</td>
<td>12.6</td>
</tr>
<tr>
<td>Pic2Word</td>
<td>8.72</td>
<td>9.51</td>
<td>10.64</td>
<td>11.29</td>
<td>23.90</td>
<td>51.70</td>
<td>87.80</td>
<td>54.12</td>
<td>11.2</td>
</tr>
<tr>
<td>SEARLE</td>
<td>11.68</td>
<td>12.73</td>
<td>1414.33</td>
<td>15.12</td>
<td>24.22</td>
<td>52.41</td>
<td>88.63</td>
<td>53.71</td>
<td>12.3</td>
</tr>
<tr>
<td>CompoDiff</td>
<td>12.55</td>
<td>13.36</td>
<td>15.83</td>
<td>16.43</td>
<td>18.24</td>
<td>53.14</td>
<td>90.25</td>
<td>57.42</td>
<td>14.9</td>
</tr>
<tr>
<td>Combiner-MMC</td>
<td></td>
<td>13.22</td>
<td>14.07</td>
<td>15.53</td>
<td>16.32</td>
<td>21.74</td>
<td>51.54</td>
<td>88.48</td>
<td>49.27</td>
<td>14.0</td>
</tr>
<tr>
<td>FROMAGe</td>
<td>OPT-6.7B</td>
<td>4.00</td>
<td>4.44</td>
<td>5.26</td>
<td>5.73</td>
<td>10.96</td>
<td>31.40</td>
<td>72.97</td>
<td>34.07</td>
<td>14.3</td>
</tr>
<tr>
<td>MCL (ours)</td>
<td>OPT-2.7B</td>
<td>14.55</td>
<td>15.79</td>
<td>17.38</td>
<td>18.27</td>
<td>23.28</td>
<td>54.17</td>
<td>90.05</td>
<td>58.24</td>
<td>15.8</td>
</tr>
<tr>
<td>MCL (ours)</td>
<td>OPT-6.7B</td>
<td>15.14</td>
<td>16.13</td>
<td>17.88</td>
<td>18.82</td>
<td>24.15</td>
<td>55.98</td>
<td>90.92</td>
<td>59.52</td>
<td>16.1</td>
</tr>
<tr>
<td>MCL (ours)</td>
<td>Llama2-7B</td>
<td>17.67</td>
<td>18.86</td>
<td>20.80</td>
<td>21.68</td>
<td>26.22</td>
<td>56.84</td>
<td>91.35</td>
<td>61.45</td>
<td>16.3</td>
</tr>
</tbody>
</table>

### 6.2.2. Visual Storytelling Results
The following are the results from Table 2 of the original paper:

| Method | Inputs | R@1 | R@5 | R@10 |
| :--- | :--- | :--- | :--- | :--- |
| CLIP ViT-L/14 | 1 caption | 11.9 | 25.5 | 32.2 |
| FROMAGe (OPT-6.7B) | 1 caption | 11.3 | 24.6 | 32.1 |
| MCL (OPT-2.7B) | 1 caption | 8.6 | 20.9 | 28.5 |
| MCL (OPT-6.7B) | 1 caption | 9.4 | 22.1 | 29.3 |
| MCL (Llama2-7B) | 1 caption | 11.4 | 25.8 | 33.9 |
| CLIP ViT-L/14 | 5 captions | 5.9 | 19.5 | 28.0 |
| FROMAGe (OPT-6.7B) | 5 captions | 10.8 | 23.8 | 31.7 |
| MCL (OPT-2.7B) | 5 captions | 9.8 | 25.2 | 35.7 |
| MCL (OPT-6.7B) | 5 captions | 11.9 | 28.8 | 38.4 |
| MCL (Llama2-7B) | 5 captions | 13.7 | 32.9 | 42.7 |
| CLIP ViT-L/14 | 5 captions, 4 images† | 2.4 | 21.3 | 34.0 |
| FROMAGe (OPT-6.7B) | 5 captions, 4 images† | 18.2 | 42.7 | 51.8 |
| GILL (OPT-6.7B) | 5 captions, 4 images† | 20.3 | 45.0 | 53.7 |
| MCL (OPT-2.7B) | 5 captions, 4 images† | 21.8 | 44.6 | 53.9 |
| MCL (OPT-6.7B) | 5 captions, 4 images† | 22.5 | 46.5 | 55.8 |
| MCL (Llama2-7B) | 5 captions, 4 images† | 23.1 | 46.7 | 56.1 |

### 6.2.3. Visual Dialog Results
The following are the results from Table 3 of the original paper:

| Method | R@1 | R@5 | R@10 |
| :--- | :--- | :--- | :--- |
| CLIP ViT-L/14 | 17.7 | 38.9 | 50.2 |
| FROMAGe (OPT-6.7B) | 20.8 | 44.9 | 56.0 |
| MCL (OPT-2.7B) | 25.6 | 51.9 | 65.2 |
| MCL (OPT-6.7B) | 27.2 | 51.0 | 64.0 |
| MCL (Llama2-7B) | 29.8 | 57.1 | 69.4 |

### 6.2.4. Visual Question Answering Results
The following are the results from Table 4 of the original paper:

| Model | LLM | Acc@zero-shot |
| :--- | :--- | :--- |
| Frozen (Tsimpoukelli et al., 2021) | GPT-2 | 29.5 |
| MAGMA (Eichenberg et al., 2021) | GPT-J-6B | 33.3 |
| LinearMapping (Merullo et al., 2022) | GPT-J-6B | 32.7 |
| Fromage (Koh et al., 2023b)† | OPT-6.7B | 36.8 |
| GILL (Koh et al., 2023a)† | OPT-6.7B | 38.8 |
| MCL (Ours) | OPT-2.7B | 38.4 |
| MCL (Ours) | OPT-6.7B | 40.2 |
| MCL (Ours) | Llama2-7B | 42.6 |

## 6.3. Ablation Studies / Parameter Analysis
The authors conducted ablation studies to verify the design choices, as shown in Table 5.

The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Method</th>
<th colspan="4">Objectives</th>
<th colspan="4">mAP@K (CIRCO)</th>
</tr>
<tr>
<th>LCap</th>
<th>LRet</th>
<th>LMC-Cap</th>
<th>CMC-Ret</th>
<th>K=5</th>
<th>K=10</th>
<th>K=25</th>
<th>K=50</th>
</tr>
</thead>
<tbody>
<tr>
<td>Single [RET] token</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>17.07</td>
<td>17.87</td>
<td>19.65</td>
<td>20.62</td>
</tr>
<tr>
<td>5 [RET] tokens w/o S.R.</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>16.48</td>
<td>17.67</td>
<td>19.48</td>
<td>20.35</td>
</tr>
<tr>
<td>5 [RET] tokens w/S.R.</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>17.67</td>
<td>18.86</td>
<td>20.80</td>
<td>21.68</td>
</tr>
<tr>
<td>Naive Mapping</td>
<td>✓</td>
<td></td>
<td></td>
<td></td>
<td>4.38</td>
<td>4.70</td>
<td>5.44</td>
<td>5.85</td>
</tr>
<tr>
<td>Naive + MC-Cap</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td>6.58</td>
<td>6.97</td>
<td>7.82</td>
<td>8.36</td>
</tr>
<tr>
<td>Naive + MC-Ret</td>
<td>✓</td>
<td>✓</td>
<td></td>
<td>✓</td>
<td>16.73</td>
<td>17.80</td>
<td>19.58</td>
<td>20.51</td>
</tr>
<tr>
<td>MC-Cap + MC-Ret</td>
<td></td>
<td></td>
<td>✓</td>
<td>✓</td>
<td>17.55</td>
<td>18.67</td>
<td>20.63</td>
<td>21.62</td>
</tr>
<tr>
<td>MCL (Full)</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>✓</td>
<td>17.67</td>
<td>18.86</td>
<td>20.80</td>
<td>21.68</td>
</tr>
</tbody>
</table>

**Analysis of Ablation:**
*   **Stacking Retrieval (S.R.):** Using 5 sequential `[RET]` tokens without the stacking mechanism actually performs worse than a single token (16.48 vs 17.07). This confirms the hypothesis that sequential tokens focus on redundant information. Using the proposed Stacking Retrieval mechanism with 5 tokens improves performance to 17.67.
*   **Objectives:** The "Naive Mapping" (only standard captioning/retrieval) performs poorly (4.38). Adding "MC-Cap" helps slightly, but adding "MC-Ret" provides a massive boost (16.73). Interestingly, the model works very well with *only* the proposed MC-Cap and MC-Ret objectives (17.55), suggesting the composition tasks are the primary driver of performance. Adding the naive objectives back in provides a final small boost.

    The following figure (Figure 4 from the original paper) visualizes the relevance of context tokens to retrieval tokens, supporting the ablation findings regarding the Stacking mechanism:

    ![该图像是一个示意图，展示了在使用与不使用组合学习（Compose Learning）的情况下，如何通过多模态上下文对视觉和文本查询进行映射及其相关性。图中展示的内容包括图像查询与文本查询的Token相关性，以及检索到的图像示例，强调了基于多模态组合学习的效果。](images/4.jpg)
    *该图像是一个示意图，展示了在使用与不使用组合学习（Compose Learning）的情况下，如何通过多模态上下文对视觉和文本查询进行映射及其相关性。图中展示的内容包括图像查询与文本查询的Token相关性，以及检索到的图像示例，强调了基于多模态组合学习的效果。*

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that **Multimodal Composition Learning (MCL)** significantly enhances the capability of Multimodal Large Language Models to understand complex contexts involving both vision and language. By introducing specific tasks (MC-Cap, MC-Ret) and a synthetic dataset (MMC), the authors overcome the limitations of standard "modality translation" training. The proposed **Stacking Retrieval Mechanism** effectively extracts diverse information from the LLM context. The experiments show state-of-the-art performance on multiple zero-shot benchmarks, proving that LLMs can be powerful engines for multimodal composition when trained correctly.

## 7.2. Limitations & Future Work
The authors acknowledge several limitations:
1.  **Failure Cases:** The model can fail when text conditions are complex or relate to image states/attributes that are hard to distinguish.
2.  **CLIP Bias:** Since MCL relies on a frozen CLIP model for the base visual features, it inherits biases and limitations present in CLIP (e.g., difficulty with certain visual concepts).
3.  **Data Quality:** The MMC dataset relies on an LLM to generate synthetic data. While scalable, there might be noise or hallucinations in the generated text conditions/target captions.

    Future work could involve:
*   Exploring MCL with more powerful vision encoders to mitigate CLIP bias.
*   Refining the data generation pipeline to ensure higher quality synthetic triplets.
*   Applying MCL to other modalities like audio or video.

## 7.3. Personal Insights & Critique
*   **Inspiration:** The most inspiring aspect is the use of LLMs to generate training data for itself (or a similar model). This "bootstrapping" approach (using LLMs to create MMC data) is a promising direction for overcoming data scarcity in specialized multimodal tasks. The "Stacking Retrieval" mechanism is also a clever, low-cost architectural tweak that yields tangible gains without changing the underlying LLM.
*   **Critique:** One potential issue is the reliance on CLIP features as the "ground truth" for the target in the retrieval loss. Since CLIP is not perfect, this might propagate errors. However, given the success, it seems a reasonable proxy.
*   **Transferability:** The methodology of generating synthetic composition data using text-only LLMs is highly transferable. It could be applied to video-text composition or audio-visual tasks where labeled data is scarce. The emphasis on "composition" over "translation" is a crucial lesson for designing future MLLM training objectives.