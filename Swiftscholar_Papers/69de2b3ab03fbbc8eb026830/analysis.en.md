# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is `VLM2Vec`, a framework designed to train Vision-Language Models (VLMs) to function as universal embedding models for a wide variety of multimodal tasks. The paper also introduces `MMEB` (Massive Multimodal Embedding Benchmark), a comprehensive benchmark for evaluating these models.

## 1.2. Authors
The authors are Ziyan Jiang (University of Waterloo, Salesforce Research), Rui Meng (Salesforce Research), Xinyi Yang (Salesforce Research), Semih Yavuz (Salesforce Research), Yingbo Zhou (Salesforce Research), and Wenhu Chen (University of Waterloo, Salesforce Research). The research is a collaboration between academia (University of Waterloo) and industry (Salesforce Research), indicating a focus on both theoretical advancement and practical application in large-scale model training.

## 1.3. Journal/Conference
The paper is currently available as a preprint on `arXiv` (arXiv:2410.05160). As of the provided date (October 2024), it has not yet been published in a specific peer-reviewed journal or conference, though it is associated with the "Tiger AI Lab" (https://tiger-ai-lab.github.io/VLM2Vec/). The preprint status suggests it is undergoing the peer-review process or awaiting acceptance at a top-tier AI venue (like NeurIPS, ICML, or CVPR).

## 1.4. Publication Year
2024.

## 1.5. Abstract
The paper addresses the lag in progress regarding universal multimodal embedding models compared to text embedding models. Its research objective is to explore the potential of building universal embeddings capable of handling a wide range of downstream tasks (classification, VQA, retrieval, grounding). The core methodology involves two contributions: (1) `MMEB`, a benchmark covering 4 meta-tasks and 36 datasets, and (2) `VLM2Vec`, a contrastive training framework that converts state-of-the-art VLMs (like Phi-3.5-V and LLaVA-1.6) into embedding models. Unlike previous models like CLIP, VLM2Vec processes images and text jointly based on task instructions. The main results show that VLM2Vec achieves an absolute average improvement of 10% to 20% over existing models on both in-distribution and out-of-distribution datasets. The key conclusion is that VLMs are "secretly strong embedding models."

## 1.6. Original Source Link
The official source is the arXiv preprint: https://arxiv.org/abs/2410.05160. The PDF link is: https://arxiv.org/pdf/2410.05160v3. The publication status is a preprint.

# 2. Executive Summary

## 2.1. Background & Motivation
The core problem the paper aims to solve is the lack of universal, high-performance multimodal embedding models. While text embeddings have seen significant success with benchmarks like `MTEB` (Massive Text Embedding Benchmark) and the development of universal models, progress in the multimodal domain has been slower. Existing models like `CLIP` and `BLIP` typically encode text and images independently or use shallow fusion techniques (e.g., simple concatenation or addition of features). This limits their ability to capture deep, complex relationships between modalities and their capacity to follow specific task instructions. The paper's entry point is the hypothesis that modern Vision-Language Models (VLMs), which are pre-trained on massive datasets and possess deep fusion architectures (like LLaVA or Phi-3.5-V), can be effectively adapted to serve as powerful, instruction-following embedding models.

## 2.2. Main Contributions / Findings
The paper's primary contributions are twofold:
1.  **MMEB (Massive Multimodal Embedding Benchmark):** A novel benchmark comprising 36 datasets across 4 meta-tasks: classification, visual question answering (VQA), retrieval, and visual grounding. It is split into 20 in-distribution datasets for training and 16 out-of-distribution datasets for evaluation. All tasks are reformulated as ranking problems where the model selects the correct target from candidates based on an instruction and query.
2.  **VLM2Vec (Vision-Language Model -> Vector):** A contrastive training framework that adapts pre-trained VLMs into embedding models. It leverages the deep integration of vision and language within VLMs, allowing the model to process any combination of images and text to generate a fixed-dimensional vector guided by task instructions.

    The key findings are that VLM2Vec significantly outperforms existing baselines (including CLIP, BLIP, UniR, and MagicLens). Specifically, it achieves an 18.2-point improvement over the best non-fine-tuned baseline and a 15.7-point improvement over the best fine-tuned baseline across all 36 datasets. The results demonstrate that VLMs, when trained with contrastive learning on diverse multimodal tasks, possess strong generalization capabilities, even in zero-shot settings on out-of-distribution data.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several fundamental concepts:

*   **Embeddings:** In machine learning, embeddings are continuous vector representations of discrete data (like words, sentences, or images). These vectors map similar items close together in a high-dimensional space, allowing models to perform tasks like semantic similarity search or clustering.
*   **Vision-Language Models (VLMs):** These are deep learning models designed to understand and generate content involving both visual and textual modalities. Examples include `LLaVA` and `Phi-3.5-V`. Unlike earlier models that processed images and text separately (dual encoders), VLMs often use a single transformer architecture to deeply fuse visual features (extracted by a vision encoder) with text tokens, enabling complex reasoning about image content.
*   **Contrastive Learning:** A training technique where a model learns to distinguish between similar (positive) and dissimilar (negative) data pairs. The goal is to pull positive pairs closer in the embedding space while pushing negative pairs apart.
*   **InfoNCE Loss:** A specific loss function often used in contrastive learning. It stands for "Noise Contrastive Estimation." It treats the correct positive pair as the "signal" and all other incorrect pairs in the batch as "noise."
*   **LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning method. Instead of updating all parameters in a large neural network, LoRA freezes the pre-trained weights and injects trainable rank decomposition matrices into each layer of the transformer architecture. This drastically reduces the number of trainable parameters and memory usage.
*   **GradCache:** A technique to decouple the backpropagation of the contrastive loss from the encoder. It allows for training with very large batch sizes (which provides many negative samples) without exceeding GPU memory limits by computing gradients in sub-batches and accumulating them.

## 3.2. Previous Works
The paper summarizes several key prior studies:
*   **CLIP (Contrastive Language-Image Pre-training):** A foundational model that learns visual concepts from natural language supervision. It uses dual encoders (one for text, one for images) and contrastive learning to align them in a shared space. However, it processes modalities independently and lacks instruction-following capabilities.
*   **BLIP / BLIP-2:** Models that bootstrap vision-language pre-training. BLIP-2, for instance, uses a frozen image encoder and a frozen Large Language Model (LLM) connected by a Q-Former (a lightweight transformer). While powerful, they are often used for generation or retrieval with shallow fusion.
*   **UniR:** A unified multimodal retriever that uses instruction guidance. However, it relies on CLIP/BLIP backbones and uses "shallow fusion" (e.g., adding text and visual features together) rather than the deep integration found in VLMs.
*   **MagicLens:** A self-supervised image retrieval model using a dual-encoder architecture initialized with CoCa or CLIP. It uses multi-head attention to pool features but still relies on a dual-encoder structure.
*   **E5-V:** A contemporary work that fine-tunes a VLM but trains exclusively on text pairs, limiting its ability to learn deep visual-textual relationships compared to VLM2Vec's multimodal training.

## 3.3. Technological Evolution
The field has evolved from simple word embeddings (Word2Vec) to contextual text embeddings (BERT), and then to multimodal embeddings (CLIP). Initially, multimodal models focused on aligning image and text globally (CLIP). The trend then shifted towards deeper fusion for generation and complex reasoning (Flamingo, BLIP-2, LLaVA). This paper represents the next step: leveraging these deep-fusion VLMs not just for generation or VQA, but specifically to create **universal, instruction-following embedding vectors** for a massive variety of downstream tasks, effectively bridging the gap between "generative VLMs" and "retrieval/embedding models."

## 3.4. Differentiation Analysis
The core difference between VLM2Vec and previous works lies in the **architecture and training objective**.
*   **vs. CLIP/BLIP:** CLIP uses dual encoders with no interaction between modalities until the final contrastive loss. VLM2Vec uses a VLM backbone (like LLaVA) where image tokens and text tokens are attended to each other throughout the transformer layers. This "deep fusion" allows VLM2Vec to capture nuanced relationships that CLIP misses.
*   **vs. UniR/MagicLens:** These models often use "shallow fusion" (e.g., adding feature vectors) or rely on dual encoders. VLM2Vec uses the full attention mechanism of the VLM to integrate information.
*   **Instruction Following:** Unlike CLIP, which encodes a text/image "as is," VLM2Vec prepends a specific task instruction (e.g., "Retrieve an image that matches this caption") to the input. This conditions the embedding on the specific task, a capability largely absent in earlier dual-encoder models.

# 4. Methodology

## 4.1. Principles
The core principle of `VLM2Vec` is to transform a generative or question-answering Vision-Language Model (VLM) into a discriminative embedding model. The intuition is that VLMs, having been trained on vast amounts of image-text data and possessing architectures that deeply fuse modalities, already contain rich representations. By applying contrastive learning with a ranking objective, VLM2Vec forces these representations to be aligned such that the dot product between a query embedding and a correct target embedding is maximized. Crucially, the model is conditioned on "instructions," allowing a single set of weights to handle diverse tasks (classification, retrieval, etc.) by simply changing the text prompt.

## 4.2. Core Methodology In-depth (Layer by Layer)

The methodology can be broken down into four main steps: Input Construction, VLM Encoding, Contrastive Loss Calculation, and Optimization via GradCache.

### Step 1: Input Construction and Instruction Formatting
The first step involves formatting the input data to include task-specific instructions. The model handles pairs of queries and targets. Let a relevant query-target pair be denoted as $(q, t^+)$. Both $q$ and $t^+$ can be a single image, single text, or a combination of both. We define the query as $q: (q_t, q_i)$ and the target as $t^+: (t_t^+, t_i^+)$.

To make the model task-aware, an instruction is prepended to the original query. The paper defines the formatted query $q_{\text{inst}}$ as follows:

$$
q_{\text{inst}} = [\text{IMAGE\_TOKEN}] \text{Instr}; \{task\_definition\} \backslash n \text{Query}; \{q\}
$$

In this formula:
*   $[\text{IMAGE\_TOKEN}]$ is a special token indicating the presence of an image (if applicable).
*   $\text{Instr}$ is the specific instruction for the task (e.g., "Retrieve an image...").
*   $\{task\_definition\}$ is a placeholder for a one-sentence description of the embedding task.
*   $\{q\}$ represents the actual query content (text or image).

    This structured input ensures the VLM understands the context and goal of the embedding generation.

### Step 2: VLM Encoding
Once the inputs are formatted, they are fed into the pre-trained VLM backbone (e.g., Phi-3.5-V or LLaVA-1.6). The VLM processes the sequence of tokens (visual and textual) through its transformer layers.

To obtain the final embedding vector, the model extracts the hidden state of the **last token** in the sequence. The paper denotes the embeddings for the query and the positive target as $(\mathbf{h}_{q_{\text{inst}}}, \mathbf{h}_{t^+})$.

### Step 3: Contrastive Loss Calculation (InfoNCE)
With the embeddings extracted, the model is trained using a contrastive learning objective. The goal is to maximize the similarity between the query embedding $\mathbf{h}_{q_{\text{inst}}}$ and the correct target embedding $\mathbf{h}_{t^+}$, while minimizing similarity with negative targets.

The paper adopts the standard `InfoNCE` loss. The loss function $\mathcal{L}$ is defined as:

$$
\operatorname*{min} \mathcal{L} = -\log \frac{\phi(\mathbf{h}_{q_{\text{inst}}}, \mathbf{h}_{t^+})}{\phi(\mathbf{h}_{q_{\text{inst}}}, \mathbf{h}_{t^+}) + \displaystyle \sum_{t^- \in \mathbb{N}} (\phi(\mathbf{h}_{q_{\text{inst}}}, \mathbf{h}_{t^-}))}
$$

Where:
*   $\mathcal{L}$ is the loss value to be minimized.
*   $\phi(\mathbf{h}_q, \mathbf{h}_t)$ is a function that computes the matching score (similarity) between a query vector $\mathbf{h}_q$ and a target vector $\mathbf{h}_t$.
*   $\mathbf{h}_{q_{\text{inst}}}$ is the embedding of the instruction-formatted query.
*   $\mathbf{h}_{t^+}$ is the embedding of the positive (correct) target.
*   $\mathbb{N}$ denotes the set of all negative targets (which includes in-batch negatives and hard negatives).
*   $t^-$ represents a specific negative target from the set $\mathbb{N}$.

    The similarity function $\phi$ is defined using temperature-scaled cosine similarity:

$$
\phi(\mathbf{h}_q, \mathbf{h}_t) = \exp(\frac{1}{\tau} \cos(\mathbf{h}_q, \mathbf{h}_t))
$$

Where:
*   $\cos(\mathbf{h}_q, \mathbf{h}_t)$ is the cosine similarity between the two vectors.
*   $\tau$ (tau) is a temperature hyper-parameter that controls the sharpness of the softmax distribution. A lower $\tau$ makes the model more confident in its predictions.

    The following figure illustrates the overall VLM2Vec workflow, including the instruction-based query, the VLM backbone, and the contrastive loss calculation:

    ![Figure 3: VLM2VE c uses a VLM as the backbone to deeply integrate image and text features. It is trained with a contrastive loss between the query and target, following task-specific instructions. The training data consists of diverse combinations of modalities on both the query and target sides, which may include images, text, or image-text pairs.](images/3.jpg)
    *该图像是示意图，展示了 VLM2Vec 模型的工作流程。左侧为查询部分，包含图像编码器与投影层，右侧为目标部分，同样包含图像编码器与投影层。两者通过对比损失训练，以实现图像和文本特征的深度融合。*

### Step 4: Increasing Batch Size Through GradCache
Training with contrastive learning benefits significantly from large batch sizes because larger batches provide more "in-batch negatives" (other examples in the same batch serve as negative samples). However, processing images and text together consumes substantial GPU memory, limiting batch size.

To overcome this, the paper employs `GradCache` (Gradient Caching). This technique decouples the backpropagation of the contrastive loss from the encoder.

Mathematically, suppose we have a large batch of queries $\mathcal{Q}$. We divide it into sub-batches, $\hat{Q}_1, \hat{Q}_2, \dots$, each small enough to fit in memory. The process involves two main steps: computing representation gradients and caching them, and then computing sub-batch gradients.

The gradients for the encoder parameters $\Theta$ are accumulated across all sub-batches using the following formula:

$$
\frac{\partial \mathcal{L}}{\partial \Theta} = \sum_{\hat{Q}_j \in Q} \sum_{q_i \in \hat{Q}_j} \frac{\partial \mathcal{L}}{\partial f(q_i)} \frac{\partial f(q_i)}{\partial \Theta} = \sum_{\hat{Q}_j \in \mathbb{Q}} \sum_{q_i \in \hat{Q}_j} \mathbf{u}_i \frac{\partial f(q_i)}{\partial \Theta}
$$

Where:
*   $\Theta$ represents the parameters of the encoder model.
*   $\mathcal{Q}$ is the set of all sub-batches of queries.
*   $f(q_i)$ is the forward pass function (encoding) for query $q_i$.
*   $\mathbf{u}_i$ represents the gradient of the loss with respect to the output representation of query $q_i$ (i.e., $\frac{\partial \mathcal{L}}{\partial f(q_i)}$).

    By computing $\mathbf{u}_i$ for each sub-batch and then re-using it to compute the weight gradients $\frac{\partial f(q_i)}{\partial \Theta}$, the model effectively trains on a massive virtual batch size without storing all intermediate activations for the entire batch at once.

# 5. Experimental Setup

## 5.1. Datasets
The experiments utilize the `MMEB` (Massive Multimodal Embedding Benchmark), which comprises 36 datasets organized into four meta-task categories.

*   **Classification (10 datasets): 20 in-distribution datasets (e.g., ImageNet-1K, HatefulMemes, VOC2007) and 16 out-of-distribution datasets (e.g., ImageNet-A, ImageNet-R, ObjectNet).
*   **Visual Question Answering (10 datasets):** Datasets like OK-VQA, A-OKVQA, DocVQA, ChartQA, and TextVQA. The query is an image + question, and the target is the answer text.
*   **Retrieval (12 datasets):** Datasets like MSCOCO, VisualNews, CIRR, and WebQA. This involves Text-to-Image (T2I), Image-to-Text (I2T), and composed image retrieval.
*   **Visual Grounding (4 datasets):** Datasets like RefCOCO and Visual7W-Pointing. The query is an image + referring expression, and the target is a cropped image of the object.

    The following are the statistics for the datasets used in MMEB from Table 1 of the original paper:

    <table>
    <thead>
    <tr>
    <th rowspan="2">Meta-Task</th>
    <th rowspan="2">Dataset</th>
    <th colspan="2">Query</th>
    <th colspan="2">Target</th>
    <th rowspan="2">OOD?</th>
    <th rowspan="2">#Training</th>
    <th rowspan="2">#Eval</th>
    <th rowspan="2">#Candidates</th>
    </tr>
    <tr>
    <th>I</th>
    <th>T</th>
    <th>I</th>
    <th>T</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td rowspan="10">Classification (10 Tasks)</td>
    <td>ImageNet-1K</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>100K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>N24News</td>
    <td>I</td>
    <td>T</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>49K</td>
    <td>1000</td>
    <td>24</td>
    </tr>
    <tr>
    <td>HatefulMemes</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>8K</td>
    <td>1000</td>
    <td>2</td>
    </tr>
    <tr>
    <td>VOC2007</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>8K</td>
    <td>1000</td>
    <td>20</td>
    </tr>
    <tr>
    <td>SUN397</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>20K</td>
    <td>1000</td>
    <td>397</td>
    </tr>
    <tr>
    <td>Place365</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>b</td>
    <td>1000</td>
    <td>365</td>
    </tr>
    <tr>
    <td>ImageNet-A</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>-</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>ImageNet-R</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>-</td>
    <td>1000</td>
    <td>200</td>
    </tr>
    <tr>
    <td>ObjectNet</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>-</td>
    <td>1000</td>
    <td>313</td>
    </tr>
    <tr>
    <td>Country-211</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>b</td>
    <td>1000</td>
    <td>211</td>
    </tr>
    <tr>
    <td rowspan="10">VQA (10 Tasks)</td>
    <td>OK-VQA</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>9K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>A-OKVQA</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>17K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>DocVQA</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>40K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>InfographicVQA</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>24K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>ChartQA</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>28K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>Visual7W</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>70K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>ScienceQA</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>b</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>VizWiz</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>-</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>GQA</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>-</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>TextVQA</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>T</td>
    <td>✓</td>
    <td>-</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td rowspan="12">Retrieval (12 Tasks)</td>
    <td>VisDial</td>
    <td></td>
    <td>T</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>123K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>CIRR</td>
    <td>I</td>
    <td>T</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>26K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>VisualNews_t2i</td>
    <td></td>
    <td>T</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>100K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>VisualNews_i2t</td>
    <td>I</td>
    <td></></td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>100K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>MSCOCO_t2i</td>
    <td></td>
    <td>T</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>100K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>MSCOCO_i2t</td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>T</td>
    <td></td>
    <td>113K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>NIGHTS</td>
    <td>I</td>
    <td></td>
    <td>I</td>
    <td></td>
    <td></td>
    <td>16K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>WebQA</td>
    <td></td>
    <td>T</td>
    <td>I</td>
    <td>T</td>
    <td></td>
    <td>17K</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>OVEN</td>
    <td>I</td>
    <td>T</td>
    <td>I</td>
    <td>T</td>
    <td>✓</td>
    <td>b</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>FashionIQ</td>
    <td>I</td>
    <td>T</td>
    <td>I</td>
    <td></td>
    <td>✓</td>
    <td>-</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>EDIS</td>
    <td></td>
    <td>T</td>
    <td>I</td>
    <td>T</td>
    <td>✓</td>
    <td>b</td>
    <td>1000</td>
    <td>1000</td>
    </tr>
    <tr>
    <td>Wiki-SS-NQ</td>
    <td></td>
    <td>T</td>
    <td>I</td>

      < <td></td>
      <td>✓</td>
      <td>-</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td rowspan="4">Visual Grounding (4 Tasks)</td>
      <td>MSCOCO</td>
      <td>I</td>
      <td>T</td>
      <td>I</td>
      <td></td>
      <td></td>
      <td>100K</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>Visual7W-Pointing</td>
      <td>I</td>
      <td>T</td>
      <td>I</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>RefCOCO</td>
      <td>I</td>
      <td>T</td>
      <td>I</td>
      <td></td>
      <td>✓</td>
      <td></td>
      <td>1000</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>RefCOCO-Matching</td>
      <td>I</td>
      <td>T</td>
      <td>I</td>
      <td>T</td>
      <td>✓</td>
      <td>-</td>
      <td>1000</td>
      <td>1000</td>
    </tr>
  </tbody>
</table>

These datasets were chosen because they cover a wide range of domains (common, news, fashion, etc.) and modalities, ensuring that a model trained on them generalizes well. The inclusion of "Out-of-Distribution" (OOD) datasets is crucial for testing the zero-shot generalization capabilities of the models.

To provide an intuitive understanding of the data format, the following are examples from the MMEB benchmark (Tables 7, 8, 9, 10):

<table>
<thead>
<tr>
<th>Category</th>
<th>Dataset</th>
<th>Query Text</th>
<th>Query Image</th>
<th>Target Text</th>
<th>Target Image</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="6">Classification</td>
<td>ImageNet-1K</td>
<td>Represent the given image for classification</td>
<td>[Image]</td>
<td>Italian greyhound</td>
<td></td>
</tr>
<tr>
<td>N24News</td>
<td>Represent the given news image with the following caption for domain classification. Ms. Goodman styled Amber Valletta with wings for a 1993 shoot by Peter Lindbergh for Harper's Bazaar.</td>
<td>[Image]</td>
<td>Style</td>
<td></td>
</tr>
<tr>
<td>HatefulMemes</td>
<td>Represent the given image for binary classification to determine whether it constitutes hateful speech or not.</td>
<td>[Image]</td>
<td>No</td>
<td></td>
</tr>
<tr>
<td>VOC2007</td>
<td>Identify the object shown in the image.</td>
<td>[Image]</td>
<td>bus</td>
<td></td>
</tr>
<tr>
<td>SUN397</td>
<td>Identify the scene shown in the image.</td>
<td>[Image]</td>
<td>firing range indoor</td>
<td></td>
</tr>
<tr>
<td>ObjectNet</td>
<td>Identify the object shown in the image.</td>
<td>[Image]</td>
<td>mug</td>
<td></td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>Category</th>
<th>Dataset</th>
<th>Query Text</th>
<th>Query Image</th>
<th>Target Text</th>
<th>Target Image</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">VQA</td>
<td>OK-VQA</td>
<td>Represent the given image with the following question. What breed of dog is this?</td>
<td>[Image]</td>
<td>chihuahua</td>
<td></td>
</tr>
<tr>
<td>A-OKVQA</td>
<td>Represent the given image with the following question. What is the metal basket near the net used to hold?</td>
<td>[Image]</td>
<td>tennis balls</td>
<td></td>
</tr>
<tr>
<td>DocVQA</td>
<td>Represent the given image with the following question. What is name of university?</td>
<td>[Image]</td>
<td>university of california</td>
<td></td>
</tr>
<tr>
<td>InfographicVQA</td>
<td>Represent the given image with the following question. Which social platform has heavy female audience?</td>
<td>[Image]</td>
<td>pinterest</td>
<td></td>
</tr>
<tr>
<td>ChartQA</td>
<td>Represent the given image with the following question. How many food item is shown in the bar graph?</td>
<td>[Image]</td>
<td>14</td>
<td></td>
</tr>
<tr>
<td>ScienceQA</td>
<td>Represent the given image with the following question. Which of these states is farthest north?</td>
<td>[Image]</td>
<td>South Carolina</td>
<td></td>
</tr>
<tr>
<td>Visual7W-telling</td>
<td>Represent the given image with the following question. Where is the man sitting?</td>
<td>[Image]</td>
<td>At the computer</td>
<td></td>
</tr>
<tr>
<td>VizWiz</td>
<td>Represent the given image with the following question. Can you tell me what this medicine is please?</td>
<td>[Image]</td>
<td>night time</td>
<td></td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>Category</th>
<th>Dataset</th>
<th>Query Text</th>
<th>Query Image</th>
<th>Target Text</th>
<th>Target Image</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="7">Retrieval</td>
<td>VisDial</td>
<td>Represent the given dialogue about an image, which is used for image retrieval. Q:do you see a lot of people A:just 3 Q:what is the tennis player wearing A:white tennis dress...</td>
<td></td>
<td>Represent the given image.</td>
<td>[Image]</td>
</tr>
<tr>
<td>VisualNews-t2i</td>
<td>Retrieve an image of this news caption. US goalkeeper Hope Solo makes a save.</td>
<td></td>
<td>Represent the given image.</td>
<td>[Image]</td>
</tr>
<tr>
<td>MSCOCO_t2i</td>
<td>Find me an everyday image that matches the given caption. Man riding a motor bike on a dirt road on the countryside.</td>
<td></td>
<td>Represent the given image.</td>
<td>[Image]</td>
</tr>
<tr>
<td>WebQA</td>
<td>Find a Wikipedia image-passage pair that answers this question. Do both the Hays County Courthouse in San Marcos, Texas and the Ike Wood House at 227 Mitchell Street in San Marcos, Texas have six columns on their front entrance?</td>
<td></td>
<td>Represent the given image with related text information. Hays County Courthouse (2018), San Marcos, TX...</td>
<td>[Image]</td>
</tr>
<tr>
<td>EDIS</td>
<td>Find a news image that matches the provided caption. Tom Holland makes his debut in the Spidey suit in Captain America Civil War.</td>
<td></td>
<td>Represent the given image with related text information. Comic RiffsJon Favreau is set to reprise his Iron Man role for Spider Man: Homecoming.</td>
<td>[Image]</td>
</tr>
<tr>
<td>Wiki-SS-NQ</td>
<td>Find the document screenshot that can answer the given query.</td>
<td></td>
<td>Represent the given document screenshot.</td>
<td>[Image]</td>
</tr>
<tr>
<td>VisualNews_i2t</td>
<td>Find a caption for the news in the given photo.</td>
<td>[Image]</td>
<td>Canadian Prime Minister Stephen Harper shakes hands with President Obama during the North American Leaders Summit in Toluca Mexico in February 2014.</td>
<td></td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>Category</th>
<th>Dataset</th>
<th>Query Text</th>
<th>Query Image</th>
<th>Target Text</th>
<th>Target Image</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="4">Grounding</td>
<td>MSCOCO</td>
<td>Select the portion of the image that isolates the object of the given label: red apple</td>
<td>[Image]</td>
<td>Represent the given cropped image of the object.</td>
<td>[Image]</td>
</tr>
<tr>
<td>RefCOCO</td>
<td>Select the portion of the image that answers the given question.</td>
<td>[Image]</td>
<td>Represent the given cropped image of the object.</td>
<td>[Image]</td>
</tr>
<tr>
<td>RefCOCO-Matching</td>
<td>Select the portion of the image that matches the description.</td>
<td>[Image]</td>
<td>Represent the given cropped image of the object.</td>
<td>[Image]</td>
</tr>
<tr>
<td>Visual7W-Pointing</td>
<td>Select the portion of the image that answers the given question. Which door is behind a person sitting on a bench?</td>
<td>[Image]</td>
<td>Represent the given cropped image of the object.</td>
<td>[Image]</td>
</tr>
</tbody>
</table>

## 5.2. Evaluation Metrics
The primary evaluation metric used in the paper is `Precision@1`.

1.  **Conceptual Definition:** `Precision@1` measures the accuracy of the model's top prediction. In a retrieval or ranking task, the model generates a list of candidates ranked by their similarity to the query. `Precision@1` checks if the correct (ground truth) item is ranked at the very top (a rank of 1). It is a strict metric that rewards the model only when it is most confident in the correct answer.
2.  **Mathematical Formula:**
    $$ P@1 = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}(\text{rank}(q, t^+) = 1) $$
3.  **Symbol Explanation:**
    *   $|Q|$ is the total number of queries in the evaluation set.
    *   $q$ represents a single query.
    *   $t^+$ represents the ground truth target for query $q$.
    *   $\text{rank}(q, t^+)$ is the rank position assigned to the ground truth target by the model.
    *   $\mathbb{I}(\cdot)$ is the indicator function, which returns 1 if the condition is true and 0 otherwise.

## 5.3. Baselines
The paper compares VLM2Vec against several representative baseline models:
*   **CLIP-family (CLIP, OpenCLIP, SigLIP, BLIP2):** These are foundational dual-encoder models. They represent the standard approach to image-text retrieval but lack instruction following and deep fusion.
*   **UniR (CLIP_SF, BLIP_FF):** A unified instruction-guided retriever. It uses shallow fusion (score-level or feature-level) on top of CLIP/BLIP. It is a strong baseline for instruction-aware retrieval.
*   **MagicLens:** A self-supervised image retrieval model designed for open-ended instructions. It uses a dual-encoder with a multi-head attention pooler.
*   **E5-V:** A model that leverages VLMs for multimodal embeddings but trains exclusively on text pairs, making it a relevant comparison to show the importance of multimodal training data.

    These baselines are representative because they cover the spectrum from non-instructional dual encoders (CLIP) to instruction-aware shallow fusion models (UniR) and VLM-based text-only training (E5-V).

# 6. Results & Analysis

## 6.1. Core Results Analysis
The main experimental results demonstrate the effectiveness of the VLM2Vec framework. The best performing model is VLM2Vec based on the `LLaVA-1.6` backbone, trained with `LoRA` and a high image resolution of $1344 \times 1344$.

*   **Overall Performance:** This model achieves an average `Precision@1` of **62.9%** across all 36 MMEB datasets.
*   **Comparison with Non-Fine-tuned Baselines:** Compared to the best baseline without fine-tuning on MMEB (UniR CLIP_SF, 44.7%), VLM2Vec achieves an **18.2 point absolute improvement**.
*   **Comparison with Fine-tuned Baselines:** Compared to the best baseline with fine-tuning (OpenCLIP-FFT, 47.2%), VLM2Vec achieves a **15.7 point absolute improvement**.
*   **Zero-shot Generalization:** On the 16 out-of-distribution (OOD) datasets, VLM2Vec achieves **57.1%**, significantly outperforming the best fine-tuned baseline (43.1%) by **14.0 points**. This indicates strong generalization capabilities.

    The results show that VLM2Vec is not only better on average but also more robust, achieving relatively high performance (at least 50%) across all four meta-task categories, whereas baselines often struggle in specific areas (e.g., VQA for CLIP).

The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="4">Per Meta-Task Score</th>
<th colspan="3">Average Score</th>
</tr>
<tr>
<th>Classification</th>
<th>VQA</th>
<th>Retrieval</th>
<th>Grounding</th>
<th>IND</th>
<th>OOD</th>
<th>Overall</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="8"><strong>Baseline Models (No Fine-tuning on MMEB Training)</strong></td>
</tr>
<tr>
<td>CLIP</td>
<td>42.8</td>
<td>9.1</td>
<td>53.0</td>
<td>51.8</td>
<td>37.1</td>
<td>38.7</td>
<td>37.8</td>
</tr>
<tr>
<td>BLIP2</td>
<td>27.0</td>
<td>4.2</td>
<td>33.9</td>
<td>47.0</td>
<td>25.3</td>
<td>25.1</td>
<td>25.2</td>
</tr>
<tr>
<td>SigLIP</td>
<td>40.3</td>
<td>8.4</td>
<td>31.6</td>
<td>59.5</td>
<td>32.3</td>
<td>38.0</td>
<td>34.8</td>
</tr>
<tr>
<td>OpenCLIP</td>
<td>47.8</td>
<td>10.9</td>
<td>52.3</td>
<td>53.3</td>
<td>39.3</td>
<td>40.2</td>
<td>39.7</td>
</tr>
<tr>
<td>UniR (BLIP_FF)</td>
<td>42.1</td>
<td>15.0</td>
<td>60.1</td>
<td>62.2</td>
<td>44.7</td>
<td>40.4</td>
<td>42.8</td>
</tr>
<tr>
<td>UniR (CLIP_SF)</td>
<td>44.3</td>
<td>16.2</td>
<td>61.8</td>
<td>65.3</td>
<td>47.1</td>
<td>41.7</td>
<td>44.7</td>
</tr>
<tr>
<td>E5-V</td>
<td>21.8</td>
<td>4.9</td>
<td>11.5</td>
<td>19.0</td>
<td>14.9</td>
<td>11.5</td>
<td>13.3</td>
</tr>
<tr>
<td>MagicLens</td>
<td>38.8</td>
<td>8.3</td>
<td>35.4</td>
<td>26.0</td>
<td>31.0</td>
<td>23.7</td>
<td>27.8</td>
</tr>
<tr>
<td colspan="8"><strong>Baseline Models (Fine-tuning on MMEB Training)</strong></td>
</tr>
<tr>
<td>CLIP-FFT</td>
<td>55.2</td>
<td>19.7</td>
<td>53.2</td>
<td>62.2</td>
<td>47.6</td>
<td>42.8</td>
<td>45.4</td>
</tr>
<tr>
<td>OpenCLIP-FFT</td>
<td>56.0</td>
<td>21.9</td>
<td>55.4</td>
<td>64.1</td>
<td>50.5</td>
<td>43.1</td>
<td>47.2</td>
</tr>
<tr>
<td colspan="8"><strong>Ours (VLM2VEC)</strong></td>
</tr>
<tr>
<td>Phi-3.5-V, FFT (bs=1024)</td>
<td>52.8</td>
<td>50.3</td>
<td>57.8</td>
<td>72.3</td>
<td>62.8</td>
<td>47.4</td>
<td>55.9</td>
</tr>
<tr>
<td>Phi-3.5-V, LoRA (bs=1024)</td>
<td>54.8</td>
<td>54.9</td>
<td>62.3</td>
<td>79.5</td>
<td>66.5</td>
<td>52.0</td>
<td>60.1</td>
</tr>
<tr>
<td>LLaVA-1.6, LoRA (bs=1024,res=336x336)</td>
<td>54.7</td>
<td>50.3</td>
<td>56.2</td>
<td>64.0</td>
<td>61.0</td>
<td>47.5</td>
<td>55.0</td>
</tr>
<tr>
<td>LLaVA-1.6, LoRA (bs=1024,res=1344x1344)</td>
<td>61.2</td>
<td>49.9</td>
<td>67.4</td>
<td>86.1</td>
<td>67.5</td>
<td>57.1</td>
<td>62.9</td>
</tr>
<tr>
<td>Δ - Best baseline (No Fine-tuning)</td>
<td>+16.9</td>
<td>+33.7</td>
<td>+5.6</td>
<td>+20.8</td>
<td>+20.4</td>
<td>+15.4</td>
<td>+18.2</td>
</tr>
<tr>
<td>Δ - Best baseline (Fine-tuning)</td>
<td>+5.2</td>
<td>+28.0</td>
<td>+12.0</td>
<td>+22.0</td>
<td>+17.0</td>
<td>+14.0</td>
<td>+15.7</td>
</tr>
</tbody>
</table>

The following are the detailed results per dataset from Table 6 of the original paper:

<table>
<thead>
<tr>
<th></th>
<th