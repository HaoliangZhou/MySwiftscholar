# 1. Bibliographic Information
## 1.1. Title
The title *MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs* clearly indicates the core focus of the work: exploring the ability of Multimodal Large Language Models (MLLMs) to localize relevant visual regions, and proposing training-free methods to improve their performance on tasks involving small visual details.
## 1.2. Authors
The authors and their affiliations are:
- Jiarui Zhang, Mahyar Khayatkhoei, Prateek Chhikara, Filip Ilievski: University of Southern California (USC, USA)
- Filip Ilievski: Vrije Universiteit Amsterdam (The Netherlands)
  The research team has expertise in multimodal AI, vision-language reasoning, and natural language processing, with prior work on VQA, MLLM limitations, and applied multimodal systems.
## 1.3. Journal/Conference
This paper is published as a preprint on arXiv, the leading open-access preprint server for computer science and related fields. As of the provided publication date, it has not yet been peer-reviewed or accepted for publication at a formal conference or journal.
## 1.4. Publication Year
2025 (published UTC 2025-02-24)
## 1.5. Abstract
The paper first identifies a critical limitation of MLLMs: their visual question answering (VQA) performance is highly sensitive to the size of the relevant visual subject, with significant accuracy drops for small details. It proves this effect is causal via intervention experiments. Next, it makes a key observation: MLLMs consistently attend to the correct region of the image (know where to look) even when they generate an incorrect answer, meaning the limitation is in perception of fine details, not localization. Based on this finding, the paper proposes three training-free visual cropping (ViCrop) methods that leverage the MLLM's internal attention maps and gradients to automatically crop relevant regions of the image, then concatenate the cropped region tokens to the original input to boost performance. Evaluations on 2 popular open-source MLLMs and 7 VQA benchmarks show the methods significantly improve accuracy on detail-sensitive tasks without requiring any training, and do not degrade performance on general VQA tasks with large objects.
## 1.6. Original Source Link
- Preprint abstract page: https://arxiv.org/abs/2502.17422
- Full PDF: https://arxiv.org/pdf/2502.17422v1
- Status: Public preprint, not yet peer-reviewed.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
MLLMs are increasingly deployed in high-stakes domains such as autonomous driving, biomedicine, and document processing, where accurate perception of small visual details (e.g., small traffic signs, fine text in medical reports, tiny print on receipts) is critical. However, existing MLLMs exhibit severe performance degradation when answering questions about small visual subjects in images.
### Research Gap
Prior work has observed a correlation between visual object size and MLLM performance, but did not prove this relationship is causal. Existing solutions to this limitation (high-resolution fine-tuning, multi-agent pipelines, external tool integration) require expensive additional training, rely on external annotated models, or add significant overhead. No prior work has explored training-free interventions using the MLLM's own internal states to address this limitation.
### Innovative Entry Point
The authors observe that even when MLLMs answer incorrectly about small details, they still allocate significantly higher attention to the ground-truth relevant region of the image. This means the core limitation is not a failure to localize the relevant subject, but a failure to perceive fine details of small subjects. This insight enables a low-cost, training-free solution: use the MLLM's internal attention/gradient signals to automatically crop the relevant region, resize it to the MLLM's full input resolution, and add its tokens to the original input to boost detail perception.
## 2.2. Main Contributions / Findings
1.  **Causal Limitation Proof**: The paper proves via controlled intervention that small visual subject size *causes* MLLM performance drops, rather than just being correlated with it. Human-annotated cropping of relevant regions recovers most of the lost accuracy for small subjects.
2.  **Localization Ability Verification**: MLLMs consistently attend to the ground-truth relevant region of the image (attention ratio > 1) even when they generate incorrect answers, confirming the limitation is in fine detail perception, not localization.
3.  **Training-Free ViCrop Methods**: Three novel training-free automatic cropping methods are proposed, leveraging the MLLM's internal states (relative attention, gradient-weighted attention, and input gradients) to find relevant regions with no external tools or training required.
4.  **Extensive Empirical Validation**: Evaluations across 7 VQA benchmarks and 2 MLLMs show the methods deliver significant accuracy gains (up to ~20% on high-resolution V* dataset for LLaVA-1.5) on detail-sensitive tasks, with no performance degradation on general VQA tasks with large objects. The method also delivers orthogonal gains on newer high-resolution MLLMs like LLaVA-NeXT, and outperforms cropping based on external tools (YOLO, SAM, CLIP).

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
To understand this paper, beginners must first master the following core concepts:
### Multimodal Large Language Model (MLLM)
A foundation model that can process both image and text inputs, and generate text outputs. Most modern MLLMs use a modular architecture: a frozen Vision Transformer (ViT) image encoder to convert images to token sequences, a connector layer (MLP or small Transformer) to map image tokens to the input space of a frozen Large Language Model (LLM), and the LLM itself to generate responses to multimodal prompts.
### Visual Question Answering (VQA)
A task where a model is given an image and a natural language question about the image, and must output a correct short or long-form answer. VQA requires both visual perception and language understanding/reasoning.
### Vision Transformer (ViT)
A transformer model designed for image processing. It splits an input image into fixed-size non-overlapping patches, embeds each patch into a vector, adds positional embeddings to preserve spatial information, and processes the sequence of patch embeddings using standard transformer encoder layers, identical to how text tokens are processed in LLMs.
### Transformer Attention Mechanism
A core component of transformers that computes the relevance of each input token to every other token, enabling the model to focus on relevant parts of the input. The standard scaled dot-product attention formula is:
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
Where:
- $Q$ (Query): A vector representing the current token's search intent
- $K$ (Key): A vector representing each input token's content
- $V$ (Value): A vector representing the content of each input token to be aggregated
- $d_k$: Dimension of the key vectors, used to scale the dot product to avoid numerical instability
- $\mathrm{softmax}$: A function that converts raw scores to normalized weights summing to 1
### Gradient-weighted Class Activation Mapping (Grad-CAM)
A technique to generate a visual importance map for an image, indicating which regions of the image contributed most to a model's output decision. It uses gradients of the model's output with respect to intermediate feature maps to weight the importance of different spatial regions.
## 3.2. Previous Works
The paper situates its work against three key lines of prior research:
### MLLM Architectures
MLLMs are broadly categorized into two types:
1.  End-to-end pretrained models: Trained on joint image-text data from scratch using architectures like dual encoders, fusion encoders, or unified transformers.
2.  Modular pretrained models (the dominant SOTA paradigm): Combine frozen pre-trained components to avoid expensive full retraining. Examples include:
    - BLIP-2 / InstructBLIP: Use a small Transformer connector between a frozen ViT and frozen LLM, only training the connector.
    - LLaVA / LLaVA-1.5: Use a linear projection or 2-layer MLP connector, training both the connector and LLM on visual instruction tuning data.
    - Qwen-VL: Uses a single cross-attention layer connector, training both the connector and LLM.
### Visual Localization Methods
1.  Supervised localization methods: YOLO (object detection), SAM (semantic segmentation), GLIP (grounded vision-language pre-training) require large amounts of dense spatial annotation to identify salient regions.
2.  Unsupervised/semi-supervised methods: Grad-CAM and its variants localize relevant regions using model gradients without spatial supervision. Prior work adapted Grad-CAM to BLIP models, but relied on BLIP's dedicated Image-Text Matching network, making it architecture-specific.
### MLLM Perception Limitations
Prior work observed that MLLMs perform poorly on questions about small objects, and proposed solutions including high-resolution fine-tuning, multi-agent pipelines, and manual visual cropping. However, all these solutions require additional training or external tools, and none proved the causal link between object size and performance, or leveraged internal MLLM states for training-free intervention.
## 3.3. Technological Evolution
The field of vision-language AI has evolved in three key stages leading to this work:
1.  **Early vision-language representation learning (2021 and earlier)**: Models like CLIP and BLIP learned aligned image-text representations, but could not perform open-ended generative tasks like VQA.
2.  **Modular MLLM rise (2022-2023)**: Models like BLIP-2 and LLaVA demonstrated strong generative multimodal performance by combining frozen pre-trained LLMs and vision encoders, enabling widespread adoption of MLLMs.
3.  **MLLM limitation analysis and mitigation (2023-present)**: Researchers began identifying critical limitations of MLLMs including object hallucination, poor small object performance, and visual blind spots, with work focused on mitigating these issues via training or tool use.
    This paper's work falls into the third stage, offering a novel training-free, architecture-agnostic mitigation for the small detail perception limitation.
## 3.4. Differentiation Analysis
Compared to prior related work, this paper's approach has three core innovations:
1.  **Causal validation**: Unlike prior work that only observed a correlation between object size and MLLM performance, this paper uses controlled intervention experiments to prove small size *causes* performance drops.
2.  **Training-free design**: Unlike existing mitigation methods that require expensive fine-tuning, multi-agent training, or external tools, this paper's methods use only the target MLLM's internal states, with no additional training, data, or external dependencies required.
3.  **Superior performance**: The proposed internal state-based cropping methods outperform external tool-based cropping (using YOLO, SAM, or CLIP) by a significant margin, as they leverage the MLLM's own task-specific understanding of the question and image.

# 4. Methodology
## 4.1. Principles
The methodology is built on two empirically validated core intuitions:
1.  MLLM performance drops on small visual subjects because the subjects are too small in the input image to have sufficient detail for the ViT encoder to capture, not because the model cannot find the subject. Cropping the subject to the full input resolution of the MLLM resolves this limitation.
2.  MLLMs already encode the location of the relevant subject in their internal attention and gradient signals, even when they answer incorrectly. These signals can be extracted and used to automatically identify the region to crop, with no external supervision required.
    The proposed ViCrop (Visual Cropping) methods extract importance maps from the MLLM's internal states, use the maps to find the relevant region, crop and resize that region to the MLLM's input resolution, and concatenate the cropped region's image tokens to the original image tokens, preserving global context while boosting detail perception of the relevant region.
## 4.2. Core Methodology In-depth (Layer by Layer)
### Step 1: Standard MLLM Inference Pipeline
All evaluated MLLMs follow a shared 4-step inference pipeline for image-question pairs `(x, q)`:
1.  The input image $x$ is split into $N \times N$ non-overlapping patches, and processed by the ViT encoder into $N^2$ output tokens.
2.  The ViT output tokens are transformed into image tokens compatible with the LLM's input space, via either an MLP connector (LLaVA-1.5) or Transformer connector (BLIP-2, InstructBLIP, Qwen-VL).
3.  The image tokens are prepended to the question tokens and a predefined starting answer token, and fed as input to the LLM.
4.  The LLM generates the answer auto-regressively starting from the starting answer token (greedy sampling is used for all experiments).
    ---
### Step 2: Quantifying MLLM Spatial Attention (to validate localization ability)
To measure where the MLLM is looking when generating an answer, the authors compute three levels of attention maps:
#### 2.1 Answer-to-Token Attention
This measures how important each image token is to the model's first answer token, extracted as the cross-attention of the starting answer token to all image tokens across all layers and heads of the backbone LLM:
$$
A_{st}(x, q) \in \mathbb{R}^{L \times H \times 1 \times T}
$$
Where:
- $L$ = number of cross-attention layers in the backbone LLM
- $H$ = number of attention heads per layer in the LLM
- $T$ = number of image tokens provided to the LLM
  The attention values are averaged across all heads to get a head-agnostic attention map:
$$
\hat{A}_{st}(x, q) = \frac{1}{H} \sum_{h=1}^{H} A_{st}(x, q)
$$
#### 2.2 Token-to-Image Attention
This measures how important each spatial image patch is to each image token, extracted from the Transformer connector (for models that use one) as the cross-attention of each image token to all ViT output patch tokens:
$$
A_{ti}(x) \in \mathbb{R}^{L_c \times H_c \times T \times N^2}
$$
Where:
- $L_c$ = number of cross-attention layers in the connector
- $H_c$ = number of attention heads per layer in the connector
- $N^2$ = number of ViT output patch tokens
  The values are averaged across all connector heads:
$$
\hat{A}_{ti}(x) = \frac{1}{H_c} \sum_{h=1}^{H_c} A_{ti}(x)
$$
For models with an MLP connector (like LLaVA-1.5), where each image token maps directly to a single ViT patch, $\hat{A}_{ti}(x)$ is set to the identity tensor.
#### 2.3 Answer-to-Image Attention
The final spatial attention map for the entire model is computed as the tensor product of the answer-to-token and token-to-image attention maps:
$$
A_{si}(x, q) \in \mathbb{R}^{L \times L_c \times 1 \times N^2}, \quad A_{si}^{mk}(x, q) = \hat{A}_{st}^m(x, q) \hat{A}_{ti}^k(x)
$$
Where:
- $m$ = index of the LLM layer
- $k$ = index of the connector layer
#### 2.4 Relative Attention
To remove irrelevant attention (e.g., attention to register tokens used by ViTs to aggregate global information, not semantic content), the answer-to-image attention for the target question is normalized by the attention for a generic, task-agnostic question $q' =$ "Write a general description of the image.":
$$
A_{rel}(x, q) = \frac{A_{si}(x, q)}{A_{si}(x, q')}
$$
#### 2.5 Attention Ratio
To quantify how much the MLLM is attending to the ground-truth relevant region, the attention ratio is defined as the sum of relative attention inside the ground-truth bounding box, divided by the average sum of relative attention across all bounding boxes of the same size in the image. A ratio > 1 indicates the MLLM is attending to the ground-truth region more than random regions of the same size.

The following figure (Figure 2 from the original paper) shows examples of MLLMs attending to the correct region even when answering incorrectly:

![Figure 2: Examples of MLLMs knowing where to look despite answering incorrectly. The right panel in each example displays relative attention to the image (defined in Sec. 4) of one layer in the MLLM.](images/2.jpg)
*该图像是多个示例，展示了多模态大型语言模型（MLLMs）在回答视觉问题时如何知道注意力集中于何处，尽管回答错误。每个示例的右侧面板显示了MLLMs在该层中的相对注意力图。*

The following figure (Figure 3 from the original paper) shows the attention ratio across all layers for multiple MLLMs, for both correct and incorrect answers, confirming the ratio is > 1 for most layers even when answers are wrong:

![Figure 3: MLLMs' attention ratio across all layers (average with $9 5 \\%$ CI over TextVQA). The attention ratio measures how significantly the MLLM is attending to the ground-truth bounding box (defined in Sec. 4). We observe that it is greater than 1 in most layers, showing that the MLLMs know where to look in the image even when they fail to answer correctly.](images/3.jpg)
*该图像是一个关于MLLMs注意力比率的图表，展示了不同层级（$l_{th}$）的注意力比率（Attention Ratio）。图中显示了BLIP-2、InstructBLIP、LLLaVA-1.5和Qwen-VL在正确回答和错误回答时的注意力分布。可以观察到，在多个层级中，注意力比率均大于1，说明模型在图像中能够有效定位重要信息。*

---
### Step 3: ViCrop Automatic Cropping Methods
Three training-free methods are proposed to generate importance maps for cropping, using different internal model signals:
#### 3.1 Relative Attention ViCrop (`rel-att`)
This method uses the relative attention map $A_{rel}(x, q)$ defined above as the importance map. A target layer (of the LLM and connector) is selected using a small held-out set of TextVQA samples, or averaged across all layers if no held-out data is available.
#### 3.2 Gradient-Weighted Attention ViCrop (`grad-att`)
This method avoids the second forward pass required for the generic query in `rel-att`, by using gradients of the model's decision to weight attention values, emphasizing semantically relevant attention:
1.  First, define the model's decision confidence as the log probability of the most likely first answer token:
    $$
    v = \log \mathrm{softmax}(z(x, q))_{t^*}, \quad t^* = \arg\max_{t} z_t
    $$
    Where $z \in \mathbb{R}^D$ is the LLM output logit at the starting answer position, and $D$ is the LLM vocabulary size.
2.  The gradient of $v$ with respect to an attention score indicates how sensitive the model's decision is to that attention score. Negative gradients correspond to attention to distracting regions that reduce decision confidence, so they are removed using a ReLU function $\sigma(w) = \max(0, w)$.
3.  Gradient-weighted answer-to-token and token-to-image attention are computed as element-wise products of the original attention and the ReLU of their respective gradients:
    $$
    \tilde{A}_{st}(x, q) = A_{st}(x, q) \odot \sigma(\nabla_{A_{st}} v(x, q))
    $$
    $$
    \tilde{A}_{ti}(x, q) = A_{ti}(x) \odot \sigma(\nabla_{A_{ti}} v(x, q))
    $$
    Where $\odot$ denotes element-wise multiplication.
4.  The final gradient-weighted answer-to-image importance map is the tensor product of the two weighted attention maps:
    $$
    \tilde{A}_{si}(x, q) = \tilde{A}_{st}(x, q) \otimes \tilde{A}_{ti}(x, q) \in \mathbb{R}^{L \times L_c \times 1 \times N^2}
    $$
    The same target layer used for `rel-att` is selected from this map as the importance map for cropping.
#### 3.3 Input Gradient ViCrop (`pure-grad`)
This method does not rely on transformer attention architecture, and uses only gradients of the model's decision with respect to the input image, making it compatible with non-transformer MLLMs:
1.  Compute the gradient of the decision confidence $v$ with respect to the input image, and take the L2 norm across the image channel dimension to get a raw importance map:
    $$
    G(x, q) = \|\nabla_x v(x, q)\|_2
    $$
2.  Remove high-gradient constant regions (e.g., blue skies, plain backgrounds) that contain no useful visual details: apply a $3\times3$ Gaussian high-pass filter to the input image, followed by a $3\times3$ median filter to remove salt-and-pepper noise, threshold the result at its spatial median to get a binary edge mask, and multiply the mask element-wise with $G$.
3.  Spatially average-pool the edge-emphasized $G$ into $N\times N$ patches matching the ViT patch size, to get the final importance map.
    ---
### Step 4: Bounding Box Selection and Inference
To convert the importance map to a cropping region:
1.  Use sliding windows of sizes ranging from 1x to 2x the MLLM's input resolution (in increments of 0.2x), with stride 1, to find the window position that maximizes the sum of importance values inside the window, for each window size.
2.  Select the optimal window: choose the window whose internal importance sum has the largest difference from the sum of its immediate horizontal and vertical neighboring windows. This avoids selecting windows that are too small (sum changes little with movement) or too large (sum is saturated).
3.  Crop the selected window from the original image, resize it to the MLLM's input resolution, process it through the ViT and connector to get cropped image tokens, and concatenate the cropped tokens to the original image tokens before feeding to the LLM.
    ---
### Step 5: High-Resolution ViCrop for Large Images
For datasets with very high-resolution images (e.g., V* dataset, images > 1024px), a two-stage strategy is used:
1.  Split the original image into non-overlapping blocks of <1024x1024px with near-1 aspect ratio, compute the importance map separately for each block, then reattach the blocks to get a full-resolution importance map for the entire image.
2.  Select the bounding box from the full-resolution importance map using the same sliding window method as above.

    The following figure (Figure 4 from the original paper) illustrates the full ViCrop pipeline applied to LLaVA-1.5 and InstructBLIP:

    ![Figure 4: Illustration of the proposed visual cropping approach applied to two MLLMs.](images/4.jpg)
    *该图像是示意图，展示了提议的视觉裁剪方法在两个多模态大型语言模型（MLLMs）中的应用，分别为 LLaVA-1.5 和 InstructBLIP。图中展示了视觉编码器、MLP、Transformer 和 LLM 的结构，以及视觉裁剪后如何影响答案生成。*

The following figure (Figure 5 from the original paper) shows examples of `rel-att` helping MLLMs correct their mistakes by cropping the relevant region:

![Figure 5: Examples of re1-at t helping MLLMs correct their mistakes (cyan-colored bounding box shows cropped region by rel-att; zoom-in insets are displayed for better readability).](images/5.jpg)
*该图像是示意图，展示了多模态大语言模型（MLLMs）在回答视觉问题时的注意力模式和性能示例。图中展示了不同的视觉问题和对应的注意力映射，展示了如何利用内在知识改进对小细节的感知。*

# 5. Experimental Setup
## 5.1. Datasets
The experiments use 7 VQA benchmarks, split into 4 detail-sensitive datasets (to evaluate small detail performance) and 3 general-purpose datasets (to verify no performance degradation on large objects):
### Detail-Sensitive Datasets
1.  **TextVQA**: VQA on images containing text, requiring reading small text in scenes. The authors manually annotated 4370 question-image pairs with ground-truth bounding boxes for the answer region, filtered to have only one instance of the question subject per image.
2.  **V***: High-resolution image VQA dataset, with very small relevant subjects requiring fine detail perception.
3.  **POPE**: Dataset for evaluating MLLM object hallucination, with questions about small and easily hallucinated objects.
4.  **DocVQA**: VQA on scanned document images, requiring reading small text in structured documents.
### General-Purpose Datasets
1.  **AOKVQA**: VQA dataset requiring world knowledge and reasoning about large, salient objects.
2.  **GQA**: Compositional visual reasoning dataset, with questions about object relations and attributes of large objects.
3.  **VQAv2**: The most widely used general VQA dataset, with questions about common everyday scenes and large salient objects. A random 50k subset of the validation set is used for evaluation.

    The following are the dataset statistics from Table 5 of the original paper:

    <table>
    <thead>
    <tr>
    <th></th>
    <th>V*</th>
    <th>DocVQA</th>
    <th>TextVQA</th>
    <th>POPE</th>
    <th>AOKVQA</th>
    <th>GQA</th>
    <th>VQAv2</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>Average Width ($\bar{W}$)</td>
    <td>2246</td>
    <td>1776</td>
    <td>954</td>
    <td>584</td>
    <td>581</td>
    <td>578</td>
    <td>577</td>
    </tr>
    <tr>
    <td>Average Height ($\bar{H}$)</td>
    <td>1582</td>
    <td>2084</td>
    <td>818</td>
    <td>478</td>
    <td>480</td>
    <td>482</td>
    <td>485</td>
    </tr>
    <tr>
    <td>Number of Images</td>
    <td>191</td>
    <td>1286</td>
    <td>3166</td>
    <td>500</td>
    <td>1122</td>
    <td>398</td>
    <td>14206</td>
    </tr>
    <tr>
    <td>Number of Questions</td>
    <td>191</td>
    <td>5349</td>
    <td>5000</td>
    <td>8910</td>
    <td>1145</td>
    <td>10781</td>
    <td>50000</td>
    </tr>
    </tbody>
    </table>

## 5.2. Evaluation Metrics
Two standard metrics are used for evaluation:
### VQA-score (used for all datasets except GQA)
1.  **Conceptual Definition**: The standard metric for VQA tasks, which accounts for multiple valid human answers by giving partial credit for answers that are correct but not the majority response. It is designed to match human evaluation of VQA answers.
2.  **Mathematical Formula**:
    $$
    \text{VQA-score}(\hat{a}) = \min\left(\frac{\text{count}(\hat{a})}{3}, 1\right)
    $$
3.  **Symbol Explanation**:
    - $\hat{a}$: The answer generated by the model
    - $\text{count}(\hat{a})$: The number of human annotators that provided $\hat{a}$ as the answer to the question
    - The score ranges from 0 to 1: 1 if at least 3 annotators gave the answer, 0.666 if 2 annotators gave it, 0.333 if 1 annotator gave it, and 0 otherwise.
### GQA Exact Match Accuracy (used only for GQA)
1.  **Conceptual Definition**: Exact string match accuracy, used for GQA because every question in GQA has a single unambiguous ground-truth answer.
2.  **Mathematical Formula**:
    $$
    \text{Accuracy} = \frac{\text{number of questions where } \hat{a} = a^*}{\text{total number of questions}}
    $$
3.  **Symbol Explanation**:
    - $\hat{a}$: Model-generated answer
    - $a^*$: Ground-truth answer for the question
## 5.3. Baselines
The proposed methods are compared against three categories of baselines:
1.  **Original MLLM baselines**: The unmodified versions of LLaVA-1.5 (Vicuna-7B) and InstructBLIP (Vicuna-7B) with no cropping, to measure the raw performance gain of ViCrop.
2.  **External tool cropping baselines**: Cropping using off-the-shelf external models (SAM, YOLOv8, CLIP) to find relevant regions, to compare internal state-based localization against external tool-based localization.
3.  **Training-based baseline (SEAL)**: The multi-agent SEAL method from the V* dataset paper, which requires extensive fine-tuning of multiple neural networks, to compare the training-free ViCrop against state-of-the-art training-based solutions for small detail perception.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### Sensitivity and Causality Validation
The following are the results from Table 1 of the original paper, showing MLLM accuracy across different subject size bins, with and without human cropping:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Method</th>
<th colspan="3">Answer Bbox Size (S)</th>
</tr>
<tr>
<th>small</th>
<th>medium</th>
<th>large</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">BLIP-2 (FlanT5xL)</td>
<td>no cropping</td>
<td>12.13</td>
<td>19.57</td>
<td>36.32</td>
</tr>
<tr>
<td>human-CROP</td>
<td>55.76</td>
<td>52.02</td>
<td>45.73</td>
</tr>
<tr>
<td rowspan="2">InstructBLIP (Vicuna-7B)</td>
<td>no cropping</td>
<td>21.79</td>
<td>30.58</td>
<td>45.30</td>
</tr>
<tr>
<td>human-CROP</td>
<td>69.60</td>
<td>61.56</td>
<td>53.39</td>
</tr>
<tr>
<td rowspan="2">LLaVA-1.5 (Vicuna-7B)</td>
<td>no cropping</td>
<td>39.38</td>
<td>47.74</td>
<td>50.65</td>
</tr>
<tr>
<td>human-CROP</td>
<td>69.95</td>
<td>65.36</td>
<td>56.96</td>
</tr>
<tr>
<td rowspan="2">Qwen-VL (Qwen-7B)</td>
<td>no cropping</td>
<td>56.42</td>
<td>65.09</td>
<td>68.60</td>
</tr>
<tr>
<td>human-CROP</td>
<td>70.35</td>
<td>75.49</td>
<td>71.05</td>
</tr>
<tr>
<td rowspan="2">GPT-4o</td>
<td>no cropping</td>
<td>65.76</td>
<td>72.81</td>
<td>69.17</td>
</tr>
<tr>
<td>human-CROP</td>
<td>75.63</td>
<td>81.32</td>
<td>71.72</td>
</tr>
</tbody>
</table>

Key observations from this table:
1.  All MLLMs (including commercial GPT-4o) show consistent accuracy drops as the subject size decreases, confirming the small detail limitation is widespread.
2.  Human cropping of the relevant region drastically improves accuracy on small and medium size subjects, even exceeding the large subject accuracy for most models, proving the small size is the causal factor for performance drops.
    ---
### Main ViCrop Performance Results
The following are the results from Table 2 of the original paper, showing ViCrop performance across all 7 benchmarks:

<table>
<thead>
<tr>
<th rowspan="2" colspan="2">Model</th>
<th colspan="4">Smaller Visual Concepts</th>
<th colspan="3">Larger Visual Concepts</th>
</tr>
<tr>
<th>TextVQA</th>
<th>V*</th>
<th>POPE</th>
<th>DocVQA</th>
<th>AOKVQA</th>
<th>GQA</th>
<th>VQAv2</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">LLaVA-1.5</td>
<td>no cropping</td>
<td>47.80</td>
<td>42.41</td>
<td>85.27</td>
<td>15.97</td>
<td>59.01</td>
<td>60.48</td>
<td>75.57</td>
</tr>
<tr>
<td>rel-att</td>
<td>55.17</td>
<td>62.30</td>
<td>87.25</td>
<td>19.63</td>
<td>60.66</td>
<td>60.97</td>
<td>76.51</td>
</tr>
<tr>
<td>grad-att</td>
<td>56.06</td>
<td>57.07</td>
<td>87.03</td>
<td>19.84</td>
<td>59.94</td>
<td>60.98</td>
<td>76.06</td>
</tr>
<tr>
<td>pure-grad</td>
<td>51.67</td>
<td>46.07</td>
<td>86.06</td>
<td>17.70</td>
<td>59.92</td>
<td>60.54</td>
<td>75.94</td>
</tr>
<tr>
<td rowspan="4">InstructBLIP</td>
<td>no cropping</td>
<td>33.48</td>
<td>35.60</td>
<td>84.89</td>
<td>9.20</td>
<td>60.06</td>
<td>49.41</td>
<td>76.25</td>
</tr>
<tr>
<td>rel-att</td>
<td>45.44</td>
<td>42.41</td>
<td>86.64</td>
<td>9.95</td>
<td>61.28</td>
<td>49.75</td>
<td>76.84</td>
</tr>
<tr>
<td>grad-att</td>
<td>45.71</td>
<td>37.70</td>
<td>86.99</td>
<td>10.81</td>
<td>61.77</td>
<td>50.33</td>
<td>76.08</td>
</tr>
<tr>
<td>pure-grad</td>
<td>42.23</td>
<td>37.17</td>
<td>86.84</td>
<td>8.99</td>
<td>61.60</td>
<td>50.08</td>
<td>76.71</td>
</tr>
</tbody>
</table>

Key observations:
1.  All three ViCrop methods deliver significant accuracy gains on detail-sensitive datasets, with `rel-att` and `grad-att` delivering the largest gains (up to +19.89% on V* for LLaVA-1.5, +8.26% on TextVQA for LLaVA-1.5, +11.96% on TextVQA for InstructBLIP).
2.  No performance degradation is observed on general-purpose datasets with large objects, as concatenating the cropped tokens preserves the original global image context.
3.  Gains are larger for LLaVA-1.5 than InstructBLIP, because InstructBLIP only trains its connector (not the LLM backbone), so the LLM is less able to utilize the additional cropped image tokens.
    ---
### External Tool Comparison and Inference Overhead
The following are the results from Table 4 of the original paper, comparing ViCrop against external tool cropping and reporting inference time overhead:

<table>
<thead>
<tr>
<th></th>
<th>Model</th>
<th>Original</th>
<th>SAM</th>
<th>YOLO</th>
<th>CLIP</th>
<th>rel-att</th>
<th>grad-att</th>
<th>pure-grad</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="2">Accuracy (TextVQA)</td>
<td>LLaVA-1.5</td>
<td>47.80</td>
<td>49.42</td>
<td>48.84</td>
<td>48.55</td>
<td>55.17</td>
<td>56.06</td>
<td>51.67</td>
</tr>
<tr>
<td>InstructBLIP</td>
<td>33.48</td>
<td>39.23</td>
<td>36.49</td>
<td>39.61</td>
<td>45.44</td>
<td>45.71</td>
<td>42.23</td>
</tr>
<tr>
<td rowspan="2">CPU Time (s)</td>
<td>LLaVA-1.5</td>
<td>2.26</td>
<td rowspan="2">91.53</td>
<td rowspan="2">0.97</td>
<td rowspan="2">5.46</td>
<td>14.43</td>
<td>11.33</td>
<td>29.86</td>
</tr>
<tr>
<td>InstructBLIP</td>
<td>0.66</td>
<td>7.04</td>
<td>4.35</td>
<td>3.78</td>
</tr>
<tr>
<td rowspan="2">GPU Time (s)</td>
<td>LLaVA-1.5</td>
<td>0.17</td>
<td rowspan="2">3.33</td>
<td rowspan="2">0.35</td>
<td rowspan="2">1.07</td>
<td>1.16</td>
<td>0.89</td>
<td>2.36</td>
</tr>
<tr>
<td>InstructBLIP</td>
<td>0.06</td>
<td>0.28</td>
<td>0.29</td>
<td>0.60</td>
</tr>
</tbody>
</table>

Key observations:
1.  All internal ViCrop methods outperform external tool-based cropping by a large margin, as they leverage the MLLM's own task-specific understanding of the question and image.
2.  GPU inference overhead is very low (0.28s to 2.36s per query), constant regardless of answer length (as it only requires processing the starting answer token), making it suitable for real-world deployment.
## 6.2. Ablation Studies
### Layer Choice Ablation
The following are the layer choice ablation results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2" colspan="2">Model</th>
<th colspan="3">Choice of Layer</th>
</tr>
<tr>
<th>Selective</th>
<th>Average</th>
<th>Δ</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">LLaVA-1.5</td>
<td>no cropping</td>
<td>47.80</td>
<td>−</td>
<td>−</td>
</tr>
<tr>
<td>rel-att</td>
<td>55.17</td>
<td>55.45</td>
<td>+0.28</td>
</tr>
<tr>
<td>grad-att</td>
<td>56.06</td>
<td>56.26</td>
<td>+0.20</td>
</tr>
<tr>
<td>pure-grad</td>
<td>51.67</td>
<td>−</td>
<td>−</td>
</tr>
<tr>
<td rowspan="4">InstructBLIP</td>
<td>no cropping</td>
<td>33.48</td>
<td>−</td>
<td>−</td>
</tr>
<tr>
<td>rel-att</td>
<td>45.44</td>
<td>44.40</td>
<td>-1.04</td>
</tr>
<tr>
<td>grad-att</td>
<td>45.71</td>
<td>44.98</td>
<td>-0.73</td>
</tr>
<tr>
<td>pure-grad</td>
<td>42.23</td>
<td>−</td>
<td>−</td>
</tr>
</tbody>
</table>

The results show that averaging attention across all layers delivers nearly identical performance to selecting a single optimal layer, meaning no held-out tuning data is required to use the method.
### High-Resolution ViCrop Ablation
The following are the high-resolution processing ablation results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2" colspan="2">Model</th>
<th colspan="3">High-Resolution ViCrop</th>
</tr>
<tr>
<th>w/ High-Res</th>
<th>w/o High-Res</th>
<th>Δ</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">LLaVA-1.5</td>
<td>no cropping</td>
<td>42.41</td>
<td>42.41</td>
<td>0</td>
</tr>
<tr>
<td>rel-att</td>
<td>62.30</td>
<td>47.64</td>
<td>-14.66</td>
</tr>
<tr>
<td>grad-att</td>
<td>57.07</td>
<td>49.74</td>
<td>-7.33</td>
</tr>
<tr>
<td>pure-grad</td>
<td>46.07</td>
<td>45.03</td>
<td>-1.04</td>
</tr>
<tr>
<td rowspan="4">InstructBLIP</td>
<td>no cropping</td>
<td>35.60</td>
<td>35.60</td>
<td>0</td>
</tr>
<tr>
<td>rel-att</td>
<td>42.41</td>
<td>38.74</td>
<td>-3.67</td>
</tr>
<tr>
<td>grad-att</td>
<td>37.70</td>
<td>42.41</td>
<td>+4.71</td>
</tr>
<tr>
<td>pure-grad</td>
<td>37.17</td>
<td>42.41</td>
<td>+5.24</td>
</tr>
</tbody>
</table>

The two-stage high-resolution processing delivers very large gains for LLaVA-1.5 on the high-res V* dataset, while for InstructBLIP even without it performance is still better than the original no-cropping baseline.
## 6.3. Additional Results
The paper also shows that ViCrop delivers orthogonal gains on newer high-resolution MLLMs like LLaVA-NeXT, improving its TextVQA accuracy from 65.17 to 68.65, and V* accuracy from 58.11 to 61.78, without any additional training. It also outperforms the training-based SEAL method on 6 out of 7 benchmarks, with faster inference and no training requirement.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper makes three critical contributions to the field of multimodal AI:
1.  It rigorously proves that MLLMs' poor performance on small visual details is a causal effect of subject size, not just a correlation, and that this limitation is widespread across all tested MLLMs including commercial GPT-4o.
2.  It discovers that MLLMs consistently localize the relevant visual region correctly even when they generate incorrect answers, meaning the limitation is in fine detail perception, not localization.
3.  It proposes three training-free, architecture-agnostic ViCrop methods that leverage the MLLM's internal attention and gradient signals to automatically crop relevant regions, delivering significant accuracy gains on detail-sensitive VQA tasks without degrading performance on general tasks, and outperforming external tool-based cropping and even some training-based solutions.
    The work provides a low-cost, widely applicable solution to a critical limitation of MLLMs, making them more reliable for deployment in high-stakes domains requiring fine detail perception.
## 7.2. Limitations & Future Work
The authors explicitly note the following limitations and future directions:
1.  **Single region limitation**: The current ViCrop methods only support cropping a single region of interest, so they cannot help with questions requiring reasoning about multiple regions (e.g., counting questions, relation questions comparing multiple objects). Future work will extend the method to support multi-region cropping.
2.  **Inference overhead**: While the GPU overhead is low, it can be further optimized via lower-precision inference, weight quantization, and integration with architectures like Matryoshka Query Transformer (MQT) that support variable visual context sizes, reducing the cost of processing cropped tokens.
3.  **Method combination**: The three ViCrop methods have complementary strengths, so future work will explore combining them dynamically based on prediction uncertainty to further improve performance.
## 7.3. Personal Insights & Critique
This paper's core insight that MLLMs already encode localization information internally is highly impactful, as it avoids the need for expensive external tools or retraining, which are major barriers to deploying MLLM improvements in production. The method is generalizable across nearly all modern MLLMs, which is a significant advantage over architecture-specific solutions.
Potential areas for improvement and further exploration:
1.  The current method uses the first answer token to compute gradients for `grad-att` and `pure-grad`. If the first token is incorrect, this could lead to a flawed importance map. Future work could explore using ensemble of multiple likely first tokens, or full answer sequence gradients, to improve robustness.
2.  The method could be adapted for black-box closed-source MLLMs (like GPT-4o) where internal attention and gradients are not accessible, by using query-based perturbations to approximate importance maps, extending its utility to commercial MLLMs.
3.  Beyond VQA, the ViCrop paradigm could be applied to other multimodal tasks including image captioning, visual grounding, and autonomous driving perception, where accurate small detail detection is critical.
4.  The method could also be used to reduce MLLM inference cost for high-resolution inputs, by only processing high-resolution crops of relevant regions instead of the entire high-resolution image.
    Overall, this work is a highly valuable contribution to MLLM research, offering a practical, low-cost solution to a critical real-world limitation, and opening up a new direction of training-free intervention using MLLMs' internal states.