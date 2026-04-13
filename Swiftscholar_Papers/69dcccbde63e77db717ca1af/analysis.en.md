# 1. Bibliographic Information
## 1.1. Title
The core topic of the paper is a comprehensive systematic review of the rapidly growing field of **Composed Image Retrieval (CIR)**, including taxonomies of existing methods, benchmark datasets, experimental comparisons, and future research directions.
## 1.2. Authors and Affiliations
The authors and their affiliations are as follows:
- Xuemeng Song, Haoqiang Lin, Bohan Hou: School of Computer Science and Technology, Shandong University, Qingdao, China
- Haokun Wen: Harbin Institute of Technology (Shenzhen), Shenzhen, China and City University of Hong Kong, Hong Kong, China
- Mingzhu Xu: School of Software, Shandong University, Jinan, China
- Liqiang Nie: School of Computer Science and Technology, Harbin Institute of Technology (Shenzhen), Shenzhen, China
  All authors are active researchers in computer vision, multimedia retrieval, and vision-language learning fields, with extensive prior publications in top venues like SIGIR, CVPR, ACM MM, and IEEE TIP/TMM.
## 1.3. Journal/Conference
The paper is published in **ACM Transactions on Information Systems (TOIS)**, which is a top-tier, highly cited peer-reviewed journal in the fields of information retrieval, data science, and multimedia systems, with a 2024 impact factor of 11.7 and an acceptance rate of ~15% for regular papers.
## 1.4. Publication Year
The paper is officially published in 2025 (Volume 44, Issue 1, Article 19, November 2025), covering research work up to June 2025.
## 1.5. Abstract
The paper addresses the gap of no existing comprehensive review of the emerging CIR task, which allows users to search for target images using a multimodal query of a reference image + modification text specifying desired changes. The authors synthesize insights from over 150 top conference and journal publications (SIGIR, CVPR, ACM TOIS, etc.), construct a fine-grained taxonomy for both supervised and zero-shot CIR models, discuss closely related tasks (attribute-based, dialog-based CIR, etc.), summarize benchmark datasets, compare experimental results across multiple datasets, and propose promising future research directions to guide subsequent work in the field.
## 1.6. Original Source Link
The official publication link is https://doi.org/10.1145/3767328, and the full PDF is available at `/files/papers/69dcccbde63e77db717ca1af/paper.pdf`. It is an officially published peer-reviewed journal paper.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
CIR is a multimodal retrieval task first proposed in 2019, where users submit a query consisting of a reference image and a natural language modification text (e.g., "I want this dress, but make it black and more professional") to retrieve target images that match the modified intent. Prior to this survey, there was no comprehensive, systematic overview of the rapidly growing CIR field, creating a high barrier to entry for new researchers and a lack of clear guidance for future work.
### Importance and Challenges
CIR addresses critical limitations of traditional unimodal image retrieval (text-only or image-only queries), where users struggle to precisely express complex search intent. It has high practical value for e-commerce product search, internet image engines, and fashion retrieval. However, the field faces three core unsolved challenges:
1.  **Multimodal Query Fusion**: Effectively combining complementary reference image and modification text information, while preserving unchanged attributes of the reference image and applying only the requested modifications.
2.  **Target Image Matching**: Bridging the semantic gap between heterogeneous multimodal queries and target images, and handling the inherent ambiguity of short modification texts that can lead to one-to-many query-target relationships.
3.  **Training Data Scarcity**: Annotated CIR triplets ($<reference image, modification text, target image>$) are labor-intensive to construct, limiting dataset scale and model generalization.
### Entry Point
The paper is the first work to systematically organize all existing CIR research, building a fine-grained taxonomy covering both supervised and zero-shot CIR paradigms, as well as related extended tasks, and providing a unified empirical analysis of method performance across benchmarks.
## 2.2. Main Contributions / Findings
The paper's primary contributions are:
1.  **First Comprehensive Survey**: Covers over 150 primary CIR studies, providing a timely overview of the field from its inception to June 2025.
2.  **Fine-Grained Taxonomy**:
    - Categorizes supervised CIR methods by their four core framework components: feature extraction, image-text fusion, target matching, and data augmentation.
    - Categorizes zero-shot CIR (ZS-CIR, which requires no annotated triplets for training) methods into three groups: textual-inversion-based, pseudo-triplet-based, and training-free.
    - Summarizes 5 closely related CIR variants: attribute-based, sketch-based, remote sensing-based, dialog-based, and video-based CIR.
3.  **Benchmark and Experimental Analysis**: Summarizes 11 public CIR datasets and 8 related task datasets, and provides a standardized performance comparison of all state-of-the-art (SOTA) supervised and ZS-CIR methods across multiple benchmarks, identifying key trends in model performance drivers.
4.  **Future Directions**: Proposes prioritized, practical future research directions for supervised CIR, ZS-CIR, and related tasks to guide the field's development.
    Key findings from the analysis include:
- Vision-Language Pre-trained (VLP) encoder-based methods significantly outperform traditional CNN/RNN encoder-based methods for CIR, often by 20%+ in recall metrics, indicating encoder choice is more impactful than fusion strategy design for current performance.
- Some zero-shot CIR methods achieve performance comparable to supervised methods that use thousands of annotated triplets, demonstrating the potential of leveraging pre-trained VLP capabilities without task-specific annotation.
- Additional modules like data augmentation, re-ranking, and uncertainty modeling consistently improve performance across all CIR paradigms.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
Below are core concepts required to understand this survey, explained for beginners:
1.  **Composed Image Retrieval (CIR)**: A multimodal retrieval task where the query is a combination of a reference image and a natural language modification text. The goal is to retrieve images from a gallery that preserve the unchanged attributes of the reference image while matching the modifications specified in the text. Unlike traditional text-to-image or image-to-image retrieval, CIR requires compositional reasoning over both modalities.
2.  **Multimodal Fusion**: The process of combining information from different modalities (e.g., images and text) into a single unified representation that captures the complementary information from each modality. For CIR, this specifically requires fusing the reference image and modification text to represent the user's desired target.
3.  **Metric Learning**: A machine learning paradigm that trains models to measure similarity between samples, such that semantically similar samples are close in an embedding space, and dissimilar samples are far apart. For CIR, metric learning is used to optimize the model to place the fused query embedding close to the embedding of the correct target image, and far from incorrect gallery images.
4.  **Vision-Language Pre-trained (VLP) Models**: Large-scale models pre-trained on massive datasets of aligned image-text pairs (e.g., CLIP, BLIP) using contrastive learning, which learn a shared embedding space where images and semantically matching text are close. These models have strong cross-modal alignment capabilities that transfer well to CIR tasks.
5.  **Zero-Shot Learning (ZSL)**: A learning paradigm where a model is trained to perform a task without access to any task-specific annotated training data. For ZS-CIR, this means no annotated $<reference image, modification text, target image>$ triplets are used for training, instead leveraging pre-trained VLP models or automatically generated synthetic data.
6.  **Textual Inversion**: A technique that maps visual concepts (e.g., an image, an object in an image) into pseudo-text tokens that are compatible with the text encoder of a VLP model, allowing visual information to be processed and fused with real text tokens using the existing text encoder.
## 3.2. Previous Works
Prior to this survey, there was no comprehensive review of the CIR field. The most relevant prior works are individual CIR papers and surveys of broader multimodal retrieval fields:
### Core Background Formula: Contrastive Loss (Foundational for VLP and CIR Metric Learning)
While not repeated in the survey, the contrastive loss that underpins all VLP models and most CIR metric learning is critical for understanding the field. The standard normalized temperature-scaled cross-entropy (NT-Xent) loss used in CLIP is:
\$
\mathcal{L}_{\text{NT-Xent}} = -\frac{1}{2N} \sum_{i=1}^{N} \left[ \log \frac{\exp(\text{sim}(z_i, z_{i+N})/\tau)}{\sum_{j=1}^{2N} \mathbb{1}_{j \neq i} \exp(\text{sim}(z_i, z_j)/\tau)} \right]
\$
Where:
- $N$ is the batch size, with `2N` total samples (each paired sample has an image and text embedding, $z_i$ and $z_{i+N}$)
- $\text{sim}(\cdot, \cdot)$ is the cosine similarity function
- $\tau$ is a temperature hyperparameter that controls the sharpness of the similarity distribution
- $\mathbb{1}_{j \neq i}$ is an indicator function equal to 1 if $j \neq i$, 0 otherwise
  This loss trains the model to maximize similarity between matching image-text pairs and minimize similarity between non-matching pairs, learning the shared cross-modal embedding space that all modern CIR methods build on.
### Key Prior CIR Paradigms
- Early supervised CIR (2019-2021) used traditional encoders (ResNet for images, LSTM/Bi-GRU for text) and explicit fusion strategies like residual combination (e.g., TIRG, the first CIR model, 2019).
- Post-2022, VLP models (CLIP, BLIP) became the dominant backbone for CIR, significantly improving performance due to their pre-trained cross-modal alignment.
- ZS-CIR was first proposed in 2023 with Pic2Word, addressing the annotation scarcity problem by leveraging textual inversion to map images to pseudo-text tokens.
## 3.3. Technological Evolution
The evolution of CIR follows the broader development of computer vision and multimodal learning:
1.  **Pre-2019**: Unimodal image retrieval (text-to-image, image-to-image) was the dominant paradigm, with limited support for flexible compositional queries.
2.  **2019**: CIR was formally introduced with the TIRG model and CSS dataset, establishing the standard task formulation of reference image + modification text queries.
3.  **2019-2022**: Supervised CIR methods matured, exploring diverse fusion strategies (MLP-based, attention-based) and target matching optimizations (negative mining, uncertainty modeling).
4.  **2022-Present**: VLP backbones became the standard for supervised CIR, driving large performance gains. The ZS-CIR task was introduced, and methods rapidly evolved to match supervised performance without annotated triplets, leveraging textual inversion, synthetic pseudo-triplets, and large language model (LLM) reasoning.
    This survey captures the full evolution of the field from its inception to the current state-of-the-art in mid-2025.
## 3.4. Differentiation Analysis
This survey is the first work to provide a unified, fine-grained taxonomy of all CIR methods, covering both supervised and zero-shot paradigms, as well as extended related tasks. Prior partial reviews were either conference workshop papers covering only specific sub-tasks (e.g., fashion CIR) or outdated (pre-2023, before the rise of VLP and ZS-CIR). The survey also provides the first standardized cross-dataset performance comparison of all SOTA methods, eliminating inconsistencies in prior model comparisons that used different evaluation protocols.

# 4. Methodology
This survey's core methodology is the construction of a hierarchical taxonomy of all CIR methods, organized by paradigm and core framework components. Below is a detailed breakdown of each category, including all core formulas as presented in the original paper.
## 4.1. Supervised CIR Taxonomy
Supervised CIR methods rely on annotated training triplets $\mathcal{D} = \{(I_r, T_m, I_t)_i\}_{i=1}^N$, where $I_r$ is the reference image, $T_m$ is the modification text, $I_t$ is the target image, and $N$ is the number of triplets. The standard supervised CIR framework is shown in the figure below:

![Fig. 3. The illustration of the standard framework of supervised CIR.](images/3.jpg)
*该图像是图示，展示了监督式组成图像检索（CIR）的标准框架。图中包含参考图像、修改文本、特征提取、图像-文本融合、目标匹配及数据增强等关键步骤，系统描绘出完整的检索流程。*

The goal of supervised CIR is to learn a multimodal fusion function $f(\cdot)$ that combines the query $(I_r, T_m)$, and a visual embedding function $h(\cdot)$ that encodes gallery images, such that the fused query embedding is close to the target image embedding in the shared space, formally:
\$
f(I_r, T_m) \to h(I_t)
\$
Supervised methods are categorized by their four core components: feature extraction, image-text fusion, target matching, and data augmentation.
### 4.1.1. Feature Extraction
Feature extraction encoders are split into two categories:
1.  **Traditional Encoders**:
    - Text encoders: RNN-based (Bi-GRU, LSTM) or transformer-based (BERT, RoBERTa) encoders pre-trained on text corpora.
    - Image encoders: CNN-based (ResNet, MobileNet) or transformer-based (ViT, Swin Transformer) encoders pre-trained on image classification datasets like ImageNet.
2.  **VLP-Based Encoders**: Large-scale vision-language models (CLIP, BLIP, BLIP-2) pre-trained on massive aligned image-text datasets, which already have strong cross-modal alignment capabilities, making them ideal for CIR tasks.
### 4.1.2. Image-Text Fusion
Fusion methods are split into three categories:
#### 4.1.2.1. Explicit Combination-Based Fusion
These methods use handcrafted, interpretable combination rules to fuse image and text features:
##### Transformed Image-and-Residual Combination
This method keeps the reference image feature as the dominant component, combining a transformed reference image feature with a residual feature that captures the modification, formulated as:
\$
g([img; txt]) \odot img + res
\$
Where:
- `img` = reference image feature, `txt` = modification text feature
- $g(\cdot)$ = a neural network that computes scaling parameters for the reference image to apply modifications
- $\odot$ = element-wise multiplication
- `res` = residual offset information capturing the requested changes
  Two variants of this strategy exist:
1.  Residual directly fused from image-text features: `res = h([img; txt])`, used by TIRG, VAL, CRN, etc. The classic TIRG method uses a sigmoid gating function for $g(\cdot)$ to adaptively preserve unchanged image attributes, and an MLP for $h(\cdot)$ to compute the residual.
2.  Residual weighted by text features: `res = h([img; txt]) \odot txt`, used by TG-CIR, DWC, AlRet, etc., to prioritize modification text information for the residual.
##### Content-and-Style Combination
These methods decompose the reference image into separate content and style features, then apply text-guided modifications to each space separately, before fusing the modified features. For example, CoSMo first modifies content via a disentangled multimodal non-local block, then adjusts style for progressive adaptation.
#### 4.1.2.2. Neural Network-Based Fusion
These methods use data-driven neural network architectures to learn implicit fusion strategies, without fixed explicit rules:
1.  **MLP-Based**: Use multi-layer perceptrons to learn adaptive modality weights for fusion. The classic `combiner` network (Baldrati et al., 2022) uses MLP-based weighted summing and feature concatenation to fuse CLIP-based image and text features, and has been adopted by dozens of subsequent CIR methods.
2.  **Cross-Attention-Based**: Use cross-attention mechanisms to model interactions between each word in the modification text and every local region in the reference image, enabling fine-grained local modifications. For example, LGLI introduces a localization mask from Faster R-CNN to identify regions to modify, then uses spatial and channel cross-attention to align text tokens with relevant image regions.
3.  **Self-Attention-Based**: Feed the concatenated sequence of image patch tokens and text word tokens into a transformer encoder with self-attention, to fully learn cross-modal interactions. For example, AACL uses an additive self-attention layer to selectively preserve or modify reference image information based on the modification text.
4.  **Graph-Attention-Based**: Construct an attribute graph of image objects/regions as nodes, then use graph attention to infuse modification text semantic information into relevant nodes. For example, JAMMA uses a jumping graph attention network to weight text-relevant image attributes, then filters redundant attributes with a gate mechanism.
#### 4.1.2.3. Prototype Image Generation-Based Fusion
These methods directly synthesize a prototype image that matches the query requirements, converting CIR into an image-to-image retrieval task. For example, SynthTripletGAN uses Generative Adversarial Networks (GANs) to generate the prototype target image from the query, then retrieves gallery images similar to the prototype.
### 4.1.3. Target Matching
Target matching uses metric learning to optimize the embedding space, with three core loss functions and several enhancement strategies:
#### Core Metric Learning Loss Functions
1.  **Batch-Based Classification (BBC) Loss**: The most widely used CIR loss, which treats all other target images in the batch as negative samples, pushing them away from the query embedding while pulling the positive target embedding closer:
    \$
    L_{BBC} = \frac{1}{B} \sum_{i=1}^{B} \left[ - \log \left( \frac{exp\{\kappa(\phi^{(i)}, \mathbf{x}_t^{(i)}) / \tau\}}{\sum_{j=1}^{B} exp\{\kappa(\phi^{(i)}, \mathbf{x}_t^{(j)}) / \tau\}} \right) \right]
    \$
    Where:
    - $\phi^{(i)}$ = fused query embedding for the $i$-th sample in the batch
    - $\mathbf{x}_t^{(i)}$ = target image embedding for the $i$-th sample
    - $B$ = batch size
    - $\kappa(\cdot, \cdot)$ = cosine similarity function
    - $\tau$ = temperature hyperparameter
2.  **Soft Triplet-Based Loss**: A variant of BBC loss that selects one negative candidate per batch iteration:
    \$
    L_{ST} = \frac{1}{M * B} \sum_{i=1}^{B} \sum_{m=1}^{M} \log\{1 + exp\{\kappa(\phi^{(i)}, \mathbf{x}_t^{(i)}) - \kappa(\phi^{(i)}, \tilde{\mathbf{x}}_{(t,m)}^{(i)})\}\}
    \$
    Where $\tilde{\mathbf{x}}_{(t,m)}^{(i)}$ is the $m$-th negative sample for the $i$-th query, and $M$ is the number of negatives per query.
3.  **Hinge-Based Triplet Ranking Loss**: Focuses on optimizing hard negative samples, using a margin threshold:
    \$
    L_{rank} = max[0, \gamma - F(\phi^{(i)}, \mathbf{x}_t^{(i)}) + F(\phi^{(i)}, \tilde{\mathbf{x}}_t^{(i)})] + max[0, \gamma - F(\phi^{(i)}, \mathbf{x}_t^{(i)}) + F(\tilde{\phi}^{(i)}, \mathbf{x}_t^{(i)})]
    \$
    Where:
    - $F(\cdot)$ = semantic similarity function
    - $\gamma$ = margin hyperparameter
    - $\tilde{\phi}$ = hard negative query, $\tilde{\mathbf{x}}_t$ = hard negative target image
#### Target Matching Enhancement Strategies
1.  **Negative Mining**: Addresses false negatives (unannotated valid target images treated as negatives) and improves hard negative selection. For example, TG-CIR uses visual similarity between the ground truth target and other batch images to regularize metric learning, while ProVLA uses a moment queue to store historical embeddings for cross-batch hard negative mining.
2.  **Uncertainty Modeling**: Addresses modification text ambiguity and one-to-many query-target relationships. For example, Ranking-aware and SDQUR model queries and targets as Gaussian distributions instead of deterministic embeddings, aligning distributions rather than fixed points.
3.  **Image Difference Alignment**: Aligns the visual difference between the reference and target image to the modification text, as an auxiliary regularization loss:
    - BBC variant for difference alignment:
      \$
      L_{BBC}^{'} = \frac{1}{B} \sum_{i=1}^{B} \left[ - \log \left( \frac{exp\{\kappa(\mathbf{v}_d^{(i)}, \mathbf{t}_m^{(i)}) / \tau\}}{\sum_{j=1}^{B} exp\{\kappa(\mathbf{v}_d^{(i)}, \mathbf{t}_m^{(j)}) / \tau\}} \right) \right]
      \$
      Where $\mathbf{v}_d^{(i)}$ is the visual difference feature between the $i$-th reference-target pair, and $\mathbf{t}_m^{(i)}$ is the modification text feature.
    - MSE loss variant for difference alignment (JPM method):
      \$
      L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} \parallel \mathbf{v}_d^{(i)} - \mathbf{t}_m^{(i)} \parallel ^2
      \$
4.  **Re-Ranking**: Post-processes initial retrieval results to improve accuracy, e.g., using a dual encoder to re-score top candidates by jointly encoding the query with each candidate, or using MLLMs to verify if candidates match the modification requirements.
### 4.1.4. Data Augmentation and Noise Handling
Six data augmentation strategies are summarized:
1.  **Image Replacement-Based**: Replace reference/target images with visually similar alternatives to generate pseudo-triplets (e.g., CLIP-CD).
2.  **Image Difference Captioning (IDC)-Based**: Use image captioning models to auto-annotate modification text for unlabeled reference-target pairs (e.g., LIMN+).
3.  **Reverse Objective-Based**: Add a reverse retrieval task (retrieve reference image from target + modification text) to expand training data (e.g., CASE, BLIP4CIR).
4.  **LLM-Based**: Use LLMs like GPT-4V/ChatGPT to auto-generate modification texts or synthetic triplets (e.g., IUDC, SDA).
5.  **Query Unification-Based**: Convert the multimodal query into a unified text-only (combine image caption + modification text) or image-only (embed modification words onto the reference image) query, to leverage standard VLP retrieval capabilities (e.g., DQU-CIR).
6.  **Gradient-Based**: Augment gradients during training to improve generalization, rather than augmenting raw data (e.g., GA).
    Three noise handling strategies are also summarized: noise removal using Gaussian mixture models to filter bad triplets, noise-aware optimization that learns from both clean and noisy triplets, and data refinement that generates synthetic pseudo-targets to replace noisy targets.
## 4.2. Zero-Shot CIR (ZS-CIR) Taxonomy
ZS-CIR requires no annotated CIR triplets for training, and methods are categorized into three groups:
### 4.2.1. Textual-Inversion-Based
These methods map reference image embeddings into pseudo-text tokens compatible with a VLP text encoder, then combine them with modification text tokens to form a unified text query for retrieval. The framework is shown below:

![该图像是示意图，展示了两种文本反转方法在图像检索中的应用。左侧(a)为粗粒度文本反转，右侧(b)为细粒度文本反转，均通过映射网络将参考图像和修改文本转化为目标图像。](images/6.jpg)
*该图像是示意图，展示了两种文本反转方法在图像检索中的应用。左侧(a)为粗粒度文本反转，右侧(b)为细粒度文本反转，均通过映射网络将参考图像和修改文本转化为目标图像。*

1.  **Coarse-Grained Textual Inversion**: Map the global reference image feature to a single pseudo-word token, e.g., Pic2Word trains a lightweight mapping network to transform CLIP image embeddings into tokens compatible with the CLIP text encoder, representing the image as "a photo of $S^*$" where $S^*$ is the learnable pseudo-word.
2.  **Fine-Grained Textual Inversion**: Map local image features to multiple pseudo-tokens (e.g., subject token + attribute tokens) to preserve fine-grained visual details, e.g., FTI4CIR maps images to a subject pseudo-token plus multiple attribute pseudo-tokens, then aligns them to real word embeddings using BLIP-generated captions.
### 4.2.2. Pseudo-Triplet-Based
These methods automatically generate synthetic pseudo-triplets to train a CIR model without manual annotation, as shown below:

![该图像是示意图，展示了两种基于不同生成方法（LLM 和 Mask-based）的组成图像检索（CIR）模型。左侧展示了 LLM 基于三元组生成，右侧为 Mask-based 三元组生成示例，分别介绍了如何利用引用图像和修改文本生成目标图像并匹配相应的图像描述。](images/7.jpg)
*该图像是示意图，展示了两种基于不同生成方法（LLM 和 Mask-based）的组成图像检索（CIR）模型。左侧展示了 LLM 基于三元组生成，右侧为 Mask-based 三元组生成示例，分别介绍了如何利用引用图像和修改文本生成目标图像并匹配相应的图像描述。*

1.  **LLM-Based Triplet Generation**: Use LLMs to generate modification texts and target captions from image-text pairs or unlabeled images, then retrieve or generate target images to form triplets. For example, TransAgg defines 8 semantic modification operations (addition, negation, etc.) to guide LLM modification text generation from image captions.
2.  **Mask-Based Triplet Generation**: Generate pseudo-triplets by masking parts of an image, then using the original image as the target, and the masked image as the reference, with the image caption as the modification text. For example, PM uses Class Activation Map (CAM)-guided masking to remove regions corresponding to nouns in the caption, creating realistic reference images.
### 4.2.3. Training-Free
These methods require no training at all, leveraging pre-trained VLP and LLM capabilities directly, as shown below:

![该图像是示意图，展示了文本到图像检索任务中的两个方法。左侧部分展示了任务转换（CIReVL），使用参考图像和修改文本生成目标图像；右侧部分展示了基于VLP的目标图像检索，包含图像编码器和文本编码器的线性插值过程。](images/8.jpg)
*该图像是示意图，展示了文本到图像检索任务中的两个方法。左侧部分展示了任务转换（CIReVL），使用参考图像和修改文本生成目标图像；右侧部分展示了基于VLP的目标图像检索，包含图像编码器和文本编码器的线性插值过程。*

1.  **Task Transformation**: Use LLMs to generate a target image caption from the reference image caption and modification text, then perform standard text-to-image retrieval using the VLP model. For example, CIReVL uses a VLP model to caption the reference image, then an LLM to combine the caption with the modification text to get the target caption for retrieval.
2.  **Pre-Trained Space Mining**: Directly fuse VLP-extracted image and text features using simple operations in the pre-trained VLP embedding space. For example, Slerp uses spherical linear interpolation to combine the reference image and modification text embeddings:
    \$
    Slerp(v, t; \alpha) = \frac{\sin((1 - \alpha)\theta)}{\sin(\theta)} \cdot v + \frac{\sin((\alpha)\theta)}{\sin(\theta)} \cdot t
    \$
    Where $\theta = \cos^{-1}(v \cdot w)$ is the angle between the image embedding $v$ and text embedding $t$, and $\alpha$ is a balancing scalar.
## 4.3. Related CIR Tasks Taxonomy
Five extended CIR variants are summarized:
1.  **Attribute-Based CIR**: Queries use pre-defined attributes instead of natural language modification text, for domains with structured attribute sets (e.g., fashion, face retrieval).
2.  **Sketch-Based CIR**: Queries use a sketch + modification text instead of a reference photo.
3.  **Remote Sensing-Based CIR**: Queries are reference remote sensing images + modification text, for earth observation applications, with additional requirements to handle spatial relationships and scene graphs.
4.  **Dialog-Based CIR**: Multi-turn queries where users iteratively refine their intent via successive modification texts, requiring the model to integrate historical interaction context.
5.  **Video-Based CIR (CoVR)**: Queries use a reference video (or image) + modification text to retrieve target videos, requiring temporal understanding of video content.

# 5. Experimental Setup
## 5.1. Datasets
The survey summarizes 11 public CIR datasets and 8 related task datasets, with key statistics shown in the table below:
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>Dataset</th>
<th>Data Type</th>
<th>Vision Scale</th>
<th>Triplet Scale</th>
<th>Triplet Construction</th>
</tr>
</thead>
<tbody>
<tr>
<th colspan="5">Datasets for CIR</th>
</tr>
<tr>
<td>FashionIQ [196]</td>
<td>image+text</td>
<td>77.6K</td>
<td>30.1K</td>
<td>Human Annotation</td>
</tr>
<tr>
<td>Shoes [12]</td>
<td>image+text</td>
<td>14.7K</td>
<td>10.7K</td>
<td>Human Annotation</td>
</tr>
<tr>
<td>CIRR [127]</td>
<td>image+text</td>
<td>21.6K</td>
<td>36.5K</td>
<td>Human Annotation</td>
</tr>
<tr>
<td>B2W [49]</td>
<td>image+text</td>
<td>3.5K</td>
<td>16.1K</td>
<td>Human Annotation</td>
</tr>
<tr>
<td>CIRCO [6]</td>
<td>image+text</td>
<td>120K</td>
<td>1.0K</td>
<td>Human Annotation (multiple targets per query)</td>
</tr>
<tr>
<td>GeneCIS [176]</td>
<td>image+text</td>
<td>33.3K</td>
<td>8.0K</td>
<td>Human Annotation</td>
</tr>
<tr>
<td>Fashion200K [66]</td>
<td>image+text</td>
<td>200K</td>
<td>205K</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>MIT-States [84]</td>
<td>image+text</td>
<td>53K</td>
<td>-</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>CSS [180]</td>
<td>image+text</td>
<td>-</td>
<td>32K</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>SynthTriplets18M [62]</td>
<td>image+text</td>
<td>-</td>
<td>18.8M</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>LaSCo [100]</td>
<td>image+text</td>
<td>121.5K</td>
<td>389.3K</td>
<td>LLM-Based Generation</td>
</tr>
<tr>
<th colspan="5">Datasets for Related Tasks of CIR</th>
</tr>
<tr>
<td>Shopping100k [3]</td>
<td>image+attributes</td>
<td>101K</td>
<td>1.1M</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>WebVid-CoVR</td>
<td>video+text</td>
<td>130.8K</td>
<td>1.6M</td>
<td>LLM-Based Generation</td>
</tr>
<tr>
<td>FS-COCO [33]</td>
<td>sketch+image+text</td>
<td>10K</td>
<td>10K</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>SketchyCOCO [53]</td>
<td>sketch+image+text</td>
<td>14K</td>
<td>14K</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>CSTBIR [56]</td>
<td>sketch+image+text</td>
<td>108K</td>
<td>2M</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>PATTERNCOM</td>
<td>image+text</td>
<td>30K</td>
<td>21K</td>
<td>Template-Based Generation</td>
</tr>
<tr>
<td>Airplane, Tennis, and WHIRT [183]</td>
<td>image+scene graph+text</td>
<td>7.7K</td>
<td>8.7K</td>
<td>Human Annotation</td>
</tr>
<tr>
<td>Multi-Turn FashionIQ [220]</td>
<td>image+text</td>
<td>13.6K</td>
<td>11.5K sessions</td>
<td>Human Annotation</td>
</tr>
</tbody>
</table>

Key dataset characteristics:
- FashionIQ is the most widely used benchmark for fashion CIR, with 77.6K clothing images and 30.1K human-annotated triplets.
- CIRR is the standard open-domain CIR benchmark, with 36.5K triplets constructed to reduce false negatives by first clustering similar images.
- CIRCO is the first dataset with multiple ground-truth target images per query, addressing the one-to-many ambiguity problem in CIR evaluation.
- SynthTriplets18M is the largest synthetic CIR dataset, with 18.8M automatically generated triplets using diffusion models.
## 5.2. Evaluation Metrics
The survey uses two standard retrieval metrics:
### 1. Recall@k (R@k)
- **Conceptual Definition**: Measures the proportion of queries for which at least one relevant target image is retrieved in the top $k$ results. It is the most widely used CIR metric, evaluating the retrieval coverage of relevant results.
- **Mathematical Formula**:
  \$
  \mathrm{Recall}@k = \frac{1}{Q} \sum_{q=1}^{Q} \frac{|\mathcal{R}_q \cap \mathcal{D}_q^k|}{|\mathcal{R}_q|}
  \$
- **Symbol Explanation**:
  - $Q$ = total number of test queries
  - $\mathcal{R}_q$ = set of all relevant target images for query $q$ (size 1 for all datasets except CIRCO)
  - $\mathcal{D}_q^k$ = set of the top $k$ retrieved images for query $q$
  - $|\mathcal{R}_q \cap \mathcal{D}_q^k|$ = number of relevant targets in the top $k$ results
    For CIRR, a variant $\mathrm{Recall}_{subset}@k$ is also used, which measures recall within a subset of visually similar images to the target, testing fine-grained discrimination.
### 2. Mean Average Precision at k (mAP@k)
- **Conceptual Definition**: Measures the average precision across all queries, considering both the rank of relevant results and the number of relevant results. It is used for the CIRCO dataset which has multiple relevant targets per query, evaluating both precision and recall.
- **Mathematical Formula**:
  \$
  \mathrm{mAP}@k = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{\mathrm{min}(k, |\mathcal{R}_q|)} \sum_{i=1}^{k} P@i \cdot \mathrm{rel}@i
  \$
- **Symbol Explanation**:
  - `P@i` = precision at rank $i$ (proportion of relevant results in the top $i$ retrieved items)
  - $\mathrm{rel}@i$ = indicator function equal to 1 if the item at rank $i$ is relevant, 0 otherwise
  - $\mathrm{min}(k, |\mathcal{R}_q|)$ = normalization factor to account for queries with fewer than $k$ relevant targets
## 5.3. Baselines
The survey compares all state-of-the-art CIR methods as baselines, grouped by paradigm:
- Supervised CIR baselines include traditional encoder-based methods (TIRG, VAL, CoSMo, etc.) and VLP encoder-based methods (CLIP4CIR, BLIP4CIR, SPRC, etc.)
- ZS-CIR baselines include textual-inversion-based (Pic2Word, SEARLE, LinCIR, etc.), pseudo-triplet-based (TransAgg, MagicLens, CompoDiff, etc.), and training-free (CIReVL, LDRE, Slerp, etc.) methods

# 6. Results & Analysis
## 6.1. Supervised CIR Performance
The survey provides cross-dataset performance comparisons for supervised CIR. Key results on the widely used FashionIQ dataset (VAL split) are shown below:
The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="2">Dresses</th>
<th colspan="2">Shirts</th>
<th colspan="2">Tops and Tees</th>
<th colspan="2">Average</th>
<th rowspan="2">Avg.</th>
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
<tr>
<th colspan="10">Traditional Encoder-Based Methods</th>
</tr>
<tr>
<td>LSC4TCIR [17] (CVPRW'21)</td>
<td>19.33</td>
<td>43.52</td>
<td>14.47</td>
<td>35.47</td>
<td>19.73</td>
<td>44.56</td>
<td>17.84</td>
<td>41.18</td>
<td>29.51</td>
</tr>
<tr>
<td>CIRPLANT [127] (ICCV'21)</td>
<td>17.45</td>
<td>40.41</td>
<td>17.53</td>
<td>38.81</td>
<td>21.64</td>
<td>45.38</td>
<td>18.87</td>
<td>41.53</td>
<td>30.20</td>
</tr>
<tr>
<td>VAL [24] (CVPR'20)</td>
<td>21.12</td>
<td>42.19</td>
<td>21.03</td>
<td>43.44</td>
<td>25.64</td>
<td>49.49</td>
<td>22.60</td>
<td>45.04</td>
<td>33.82</td>
</tr>
<tr>
<td>... (truncated for brevity, full table available in original paper)</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>CRN (Swin-Large) [210] (TIP'23)</td>
<td>32.67</td>
<td>59.30</td>
<td>30.27</td>
<td>56.97</td>
<td>37.74</td>
<td>65.94</td>
<td>33.56</td>
<td>60.74</td>
<td>47.15</td>
</tr>
<tr>
<th colspan="10">VLP Encoder-Based Methods</th>
</tr>
<tr>
<td>CLIP-ProbCR [106] (ICMR'24)</td>
<td>30.71</td>
<td>56.55</td>
<td>28.41</td>
<td>52.04</td>
<td>35.03</td>
<td>61.11</td>
<td>31.38</td>
<td>56.57</td>
<td>43.98</td>
</tr>
<tr>
<td>... (truncated for brevity)</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>LIMN+ [194] (TPAMI'24)</td>
<td>52.11</td>
<td>75.21</td>
<td>57.51</td>
<td>77.92</td>
<td>62.67</td>
<td>82.66</td>
<td>57.43</td>
<td>78.60</td>
<td>68.01</td>
</tr>
<tr>
<td>FineCIR [114] (Arxiv'25)</td>
<td>55.29</td>
<td>79.61</td>
<td>64.84</td>
<td>84.82</td>
<td>63.42</td>
<td>83.64</td>
<td>61.18</td>
<td>82.69</td>
<td>71.94</td>
</tr>
</tbody>
</table>

### Key Observations for Supervised CIR
1.  **VLP Encoders Outperform Traditional Encoders Significantly**: The average performance of VLP-based methods is 15-25% higher than traditional encoder-based methods. For example, the traditional encoder CRN (Swin-Large) achieves an average recall of 47.15 on FashionIQ, while the VLP-based LIMN+ achieves 68.01, a 44% relative improvement. Even the weakest VLP-based method outperforms most traditional encoder-based methods.
2.  **Larger Backbones Improve Performance**: For both traditional and VLP encoders, larger parameter versions consistently outperform smaller ones. For example, CRN (Swin-Large) outperforms CRN (Swin-Base) by ~2 points, and FashionERN (CLIP-ViT-B/16) outperforms FashionERN (CLIP-RN50x4) by ~7 points. This indicates that encoder scale and pre-training quality are the largest drivers of current CIR performance, outweighing fusion strategy design.
3.  **Additional Modules Boost Performance**: Models with data augmentation, re-ranking, or uncertainty modeling consistently outperform their base versions. For example, SPRC+VQA (which uses VQA to re-rank results) outperforms base SPRC by ~2 points, and LIMN+ (which uses iterative pseudo-triplet augmentation) outperforms base LIMN by ~1.2 points.
## 6.2. Zero-Shot CIR Performance
Key results for ZS-CIR on FashionIQ (original split) are shown below:
The following are the results from Table 10 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th rowspan="2">Encoder</th>
<th colspan="2">Dresses</th>
<th colspan="2">Shirts</th>
<th colspan="2">Tops and Tees</th>
<th colspan="2">Average</th>
<th rowspan="2">Avg.</th>
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
<tr>
<th colspan="11">Textual-Inversion-Based Methods</th>
</tr>
<tr>
<td>SEARLE [6] (ICCV'23)</td>
<td>CLIP-B</td>
<td>18.54</td>
<td>39.51</td>
<td>24.44</td>
<td>41.61</td>
<td>25.70</td>
<td>46.46</td>
<td>22.89</td>
<td>42.53</td>
<td>32.71</td>
</tr>
<tr>
<td>... (truncated for brevity)</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>LinCIR [63] (CVPR'24)</td>
<td>CLIP-G</td>
<td>38.08</td>
<td>60.88</td>
<td>46.76</td>
<td>65.11</td>
<td>50.48</td>
<td>71.09</td>
<td>45.11</td>
<td>65.69</td>
<td>55.40</td>
</tr>
<tr>
<th colspan="11">Pseudo-Triplet-Based Methods</th>
</tr>
<tr>
<td>MagicLens [233] (ICML'24)</td>
<td>CoCa-L</td>
<td>32.30</td>
<td>52.70</td>
<td>40.50</td>
<td>59.20</td>
<td>41.40</td>
<td>63.00</td>
<td>38.00</td>
<td>58.20</td>
<td>48.10</td>
</tr>
<tr>
<td>... (truncated for brevity)</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<th colspan="11">Training-Free Methods</th>
</tr>
<tr>
<td>LinCIR + IP-CIR [112] (CVPR'25)</td>
<td>CLIP-G</td>
<td>39.02</td>
<td>61.03</td>
<td>48.04</td>
<td>66.68</td>
<td>50.16</td>
<td>71.13</td>
<td>45.74</td>
<td>66.28</td>
<td>56.01</td>
</tr>
</tbody>
</table>

### Key Observations for ZS-CIR
1.  **Some Zero-Shot Methods Match Supervised Performance**: The top ZS-CIR method (LinCIR + IP-CIR, CLIP-G) achieves an average score of 56.01 on FashionIQ, which outperforms all traditional encoder-based supervised methods and is comparable to ~50% of VLP-based supervised methods. This demonstrates that zero-shot CIR can achieve strong performance without any annotated triplets, by leveraging the pre-trained capabilities of large VLP and LLM models.
2.  **Pseudo-Triplet and Training-Free Methods Outperform Textual Inversion**: On average, pseudo-triplet and training-free methods outperform textual-inversion-based methods. For example, 9 pseudo-triplet methods and 6 training-free methods achieve average scores >45, compared to only 2 textual-inversion methods. This is because pseudo-triplet methods use training data that closely matches the supervised CIR paradigm, while training-free methods preserve the full pre-trained cross-modal capabilities of VLP models without fine-tuning degradation.
3.  **Larger Backbones Also Improve ZS-CIR Performance**: All ZS-CIR method variants achieve higher performance with larger VLP backbones. For example, LinCIR with CLIP-G achieves 55.40, compared to 36.38 with CLIP-L and 32.71 with CLIP-B, a 69% relative improvement from the smallest to largest backbone.
## 6.3. Ablation and Parameter Analysis
The survey summarizes ablation study findings from existing work:
- For fusion strategies: No single fusion strategy is universally optimal across all datasets and backbones. MLP-based, cross-attention, and self-attention fusion all achieve top performance on different benchmarks, indicating that fusion strategy performance is highly dependent on the specific backbone and dataset.
- For data augmentation: LLM-based augmentation and reverse objective augmentation consistently improve performance by 2-5% across most methods, while mask-based augmentation provides smaller gains (1-2%) but has lower computational cost.
- For temperature hyperparameter $\tau$ in metric learning: Optimal values range from 0.05 to 0.1 for most CIR models, with lower values leading to more discriminative embeddings but higher risk of overfitting.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper is the first comprehensive, systematic survey of the Composed Image Retrieval field, covering over 150 studies from the task's inception in 2019 to mid-2025. Its key contributions are:
1.  A fine-grained taxonomy of supervised CIR methods organized by the four core framework components (feature extraction, image-text fusion, target matching, data augmentation), and a three-category taxonomy of zero-shot CIR methods.
2.  A summary of 19 public CIR and related task datasets, and a standardized cross-dataset performance comparison of all state-of-the-art methods, identifying key trends (VLP encoders as the dominant performance driver, zero-shot methods closing the gap with supervised methods, etc.).
3.  A set of prioritized future research directions to guide the field's development, addressing current limitations in dataset quality, LLM integration, task synergy, and more.
    The survey provides an invaluable entry point for new researchers into the CIR field, and a unified reference for existing researchers to compare methods and identify open problems.
## 7.2. Limitations & Future Work
The authors identify the following key limitations and future directions:
### For Supervised CIR
1.  **LLM-Based Image-Text Fusion**: Current methods underutilize MLLMs for query reasoning and fusion. The key challenge is balancing fine-tuning for CIR performance while preserving the MLLM's native reasoning capabilities for complex, multi-attribute modification queries.
2.  **Benchmark Dataset Construction**: Existing datasets are narrow (mostly fashion), suffer from annotation bias and false negatives, and are too small to benefit from scaling laws. Future datasets need to be large-scale, open-domain, with many-to-many query-target mappings to reflect real-world ambiguity.
3.  **Task Synergy Between CIR and Image Editing**: CIR and text-based image editing share the same input structure and compositional reasoning requirements, but are currently treated as separate tasks. A unified framework that combines retrieval (for existing matches) and generation (for novel compositions) would enable more flexible and powerful systems.
4.  **Robust Target Matching with Noisy/Biased Data**: Current noise handling methods use rigid binary classification of triplets as clean/noisy, failing to leverage partially noisy samples as useful hard negatives. Causal intervention methods are needed to reduce annotation bias and improve generalization.
### For Zero-Shot CIR
1.  **High-Quality Pseudo-Triplet Generation**: Current pseudo-triplet generation pipelines lose fine-grained visual details when captioning images, leading to coarse modifications and synthetic bias. Future work needs dense captioning and iterative refinement to generate high-quality, diverse triplets.
2.  **MLLM-Based Training-Free Methods**: Current training-free methods use LLMs to generate captions, losing visual details. MLLMs that can directly process both image and text queries for retrieval can preserve fine-grained details and improve zero-shot performance.
3.  **Advanced Textual Inversion**: Current textual inversion methods suffer from training-inference objective misalignment and information loss during mapping. Future work needs to optimize inversion specifically for compositional retrieval tasks, not just generic image-text alignment.
4.  **Few-Shot CIR**: A middle ground between fully supervised and zero-shot CIR, where a small number of annotated triplets (100-1000) are used to fine-tune models, is a promising under-explored direction for practical deployment.
### For Related CIR Tasks
Current related task methods are underdeveloped, failing to address domain-specific requirements (e.g., temporal coherence for video CIR, spatial relationship reasoning for remote sensing CIR). Future work needs unified multimodal architectures that can dynamically adapt to different input modalities and task requirements.
## 7.3. Personal Insights & Critique
### Strengths
- The survey's fine-grained taxonomy is highly practical, providing a clear framework for understanding the design space of CIR models, and the standardized experimental comparison eliminates inconsistencies in prior model evaluations that used different protocols.
- The future directions are well-prioritized and actionable, addressing the most pressing bottlenecks in the field, particularly the dataset limitation which is the largest barrier to further progress.
- The coverage of zero-shot CIR is timely, as this paradigm is rapidly evolving and has high practical value for real-world deployment where annotation is costly.
### Potential Improvements
- The survey could provide more detailed coverage of practical deployment considerations, such as inference latency and memory footprint, which are critical for industrial applications but are largely ignored in current academic research.
- The related tasks section could include more analysis of cross-domain transferability of CIR models, e.g., how well models trained on fashion data transfer to remote sensing or video CIR.
- The survey notes that encoder choice is currently the largest driver of performance, but could more strongly emphasize the risk of over-reliance on large VLP backbones, which are costly to deploy and may have inherent biases from pre-training data.
  Overall, this survey is a landmark work for the CIR field, providing a comprehensive foundation that will accelerate future research and practical deployment of composed retrieval systems.