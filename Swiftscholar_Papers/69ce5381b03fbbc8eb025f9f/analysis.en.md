# 1. Bibliographic Information
## 1.1. Title
The paper's central topic is the development of `StanceMoE`, a Mixture-of-Experts (MoE) architecture tailored for actor-level stance detection, submitted as part of the KUET team's participation in the StanceNakba 2026 Shared Task. It focuses on detecting implicit stances toward geopolitical actors in Nakba-related English discourse.
## 1.2. Authors
All authors are affiliated with the Department of Computer Science and Engineering, Khulna University of Engineering & Technology (KUET), Bangladesh:
- Abdullah Al Shafi (corresponding author, affiliated with IICT, KUET)
- Md. Milon Islam
- Sk. Imran Hossain
- K. M. Azharul Hasan
  Their research focuses on natural language processing, text classification, and deep learning applications for social discourse analysis.
## 1.3. Journal/Conference
The paper is a preprint submitted to the Proceedings of the 15th International Conference on Language Resources and Evaluation (LREC 2026) as part of the StanceNakba Shared Task. LREC is a top-tier, peer-reviewed conference in the field of computational linguistics and NLP, with high reputation for curating high-quality shared task datasets and methodological contributions.
## 1.4. Publication Year
2026, published publicly on arXiv at UTC 2026-04-01T13:24:03.000Z.
## 1.5. Abstract
The paper addresses the limitation of standard transformer-based stance detection models, which rely on unified representations that fail to capture heterogeneous linguistic signals (contrastive discourse, framing cues, salient lexical indicators) critical for accurate stance classification. It proposes StanceMoE, a context-enhanced MoE architecture built on fine-tuned BERT, with 6 specialized expert modules capturing complementary stance signals and a context-aware gating mechanism for dynamic expert weighting. Evaluated on the 1401-sample StanceNakba 2026 Subtask A dataset (implicit target actors), StanceMoE achieves a macro-F1 score of 94.26%, outperforming all traditional and BERT-based baselines, and secured 3rd place in the shared task.
## 1.6. Original Source Link
- Preprint source: https://arxiv.org/abs/2604.00878
- PDF link: https://arxiv.org/pdf/2604.00878v1
- Publication status: Preprint, under review for inclusion in LREC 2026 shared task proceedings.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Actor-level stance detection aims to identify an author's position (support, oppose, neutral) toward implicit geopolitical actors in text, a task far more complex than general sentiment analysis as it requires target-aware inference even when actors are not explicitly named.
### Field Importance & Research Gaps
Stance detection is critical for applications like social media monitoring, election trend analysis, and misinformation detection. Prior transformer-based models use single aggregated contextual embeddings, which fail to capture the diverse, heterogeneous linguistic signals that convey stance: global semantic orientation, salient lexical cues, clause-level focus, phrase-level slogans, framing indicators, and contrastive discourse shifts. No prior work has designed an adaptive architecture to explicitly model these complementary signals for implicit actor stance detection.
### Innovative Entry Point
The paper leverages the Mixture-of-Experts (MoE) paradigm to decompose stance modeling into specialized, complementary expert modules, each targeting one type of stance-related linguistic signal, paired with a context-aware gating mechanism that dynamically weights expert contributions based on input text characteristics.
## 2.2. Main Contributions / Findings
### Primary Contributions
1. Proposes `StanceMoE`, the first MoE architecture specifically designed for implicit actor-level stance detection.
2. Designs 6 complementary expert modules tailored to capture distinct stance-related linguistic phenomena, from global semantic orientation to contrastive discourse shifts.
3. Introduces a <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>-based context-aware gating mechanism for adaptive, input-sensitive fusion of expert outputs.
4. Achieves state-of-the-art performance on the StanceNakba 2026 Subtask A dataset, securing 3rd place in the shared task.
### Key Findings
1. Explicitly modeling complementary stance signals via specialized experts delivers significantly better performance than monolithic BERT representations or static expert integration methods.
2. Self-attention pooling (for clause-level focus) and contrast-aware (for discourse shifts) experts are the most critical components for accurate stance detection in geopolitical discourse.
3. The dynamic gating mechanism reduces prediction variance and improves model stability compared to static fusion approaches.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All core terms are explained for novice readers below:
- **Stance Detection**: A natural language processing task that identifies an author's explicit or implicit position (support, oppose, neutral) toward a specific target (topic, entity, actor) from text. Unlike general sentiment analysis, it is target-dependent: the same text can have different stances for different targets.
- **Actor-level Stance Detection**: A subtype of stance detection where the target is a social or geopolitical actor (e.g., state, activist group), which is often implicit (not explicitly named) in the input text.
- **BERT (Bidirectional Encoder Representations from Transformers)**: A pre-trained transformer model that generates context-aware token embeddings by learning bidirectional relationships between words in a text. It is fine-tuned for downstream tasks like text classification.
- **Mixture-of-Experts (MoE)**: A neural network architecture where multiple specialized sub-networks (called experts) process input, and a separate gating network dynamically weights their outputs based on input characteristics, instead of using a single monolithic network. This enables specialized modeling of heterogeneous input patterns.
- **Macro-F1 Score**: The primary evaluation metric for this task, calculated as the harmonic mean of precision and recall for each class, averaged equally across all classes. It is robust to class imbalance, as it does not weight larger classes more heavily.
- **Cross Entropy Loss with Label Smoothing**: A classification loss function that adjusts hard one-hot labels (e.g., [1,0,0] for class 1) to soft labels (e.g., [0.9, 0.05, 0.05] for class 1) to reduce overfitting and improve model generalization.
- **Stratified K-fold Cross Validation**: A resampling technique where a dataset is split into k subsets (folds), each preserving the same class distribution as the full dataset, to ensure robust evaluation and reduce variance in performance estimates.
## 3.2. Previous Works
The field of stance detection has evolved through three distinct eras, as summarized by the authors:
1. **Rule-based Systems (pre-2020)**: Relied on manually crafted rules (keyword matching, pattern matching) to detect stance, as surveyed in Küçük and Can (2020). Limitations: Not scalable, requires manual engineering, fails on implicit stance.
2. **Classical Supervised Systems (2020-2023)**: Used traditional machine learning models (Logistic Regression, SVM, Random Forest) with handcrafted features (TF-IDF, n-grams), as reviewed in Alturayeif et al. (2023). Limitation: Relies on manual feature engineering, poor performance on context-dependent implicit stance.
3. **Deep Learning Systems (2023-present)**:
   - CNN/LSTM with attention: Automatically extract features and capture target-specific information, as surveyed in Gera and Neal (2025). A representative example is the Target-specific Attentional Network (TAN, Du et al. 2017), which uses a trainable attention vector to weight stance-relevant tokens:
     $$\alpha_i = \frac{\mathsf{exp}(\mathsf{tanh}(h_i^\top v))}{\sum_{j=1}^{T} \mathsf{exp}(\mathsf{tanh}(h_j^\top v))}$$
     Where $\alpha_i$ is the attention weight for token $i$, $h_i$ is the token embedding, and $v$ is a trainable attention vector.
   - Transformer-based models (BERT, LLMs): Use pre-trained contextual embeddings to achieve state-of-the-art performance, as demonstrated by StanceFormer (Garg and Caragea 2024). Limitation: Use single aggregated representations, which fail to capture heterogeneous linguistic signals for stance detection.
## 3.3. Technological Evolution
The evolution of stance detection methods follows a clear trajectory:
1. Manual rule engineering → 2. Manual feature engineering for classical ML → 3. Automatic feature extraction via CNN/LSTM with attention → 4. Unified contextual embeddings via transformers → 5. This paper's adaptive MoE architecture for explicit modeling of heterogeneous stance signals.
   This work fills the critical gap in transformer-based methods by disentangling diverse stance signals instead of relying on a single unified embedding, enabling more fine-grained and robust stance detection.
## 3.4. Differentiation Analysis
Compared to prior state-of-the-art methods, StanceMoE has three core innovations:
1. Unlike standard BERT models that use a single <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token embedding for classification, StanceMoE decomposes stance modeling into 6 specialized expert modules, each targeting a distinct stance-related linguistic phenomenon.
2. Unlike static fusion or stacked expert architectures that weight all experts equally, StanceMoE uses a context-aware gating mechanism that dynamically adjusts expert weights based on input text characteristics.
3. Unlike general-purpose MoE architectures for NLP, this is the first MoE model specifically tailored for implicit actor-level stance detection, with experts designed to capture the unique linguistic patterns of geopolitical discourse.

# 4. Methodology
## 4.1. Principles
The core intuition of StanceMoE is that stance is expressed through multiple distinct, complementary linguistic signals, so a single unified representation cannot capture all of them effectively. The theoretical basis is that MoE architectures improve performance by enabling specialization of sub-networks, and adaptive fusion of heterogeneous signals outperforms static aggregation for fine-grained classification tasks like stance detection. The model is designed to explicitly disentangle different types of stance signals, then combine them dynamically based on the unique characteristics of each input text.
## 4.2. Core Methodology In-depth
The StanceMoE architecture has three sequential components: (1) Contextual BERT Encoder, (2) 6 Parallel Expert Modules, (3) Context-Aware Gating and Fusion Mechanism. The end-to-end architecture is illustrated below:
The following figure (Figure 1 from the original paper) shows the StanceMoE system architecture:

![Figure 1: Proposed Mixture-of-Experts architecture for actor-level stance detection.](images/1.jpg)
*该图像是提出的 Mixture-of-Experts 架构的示意图，用于执行演员级立场检测。该模型集成了多种专家模块，通过 CLS 基于的门控网络动态加权，以捕捉不同的语言信号。最终，模型采用加权专家融合的方法，得到最终的立场预测。*

### Step 1: Contextual Encoder
First, the input text is tokenized into a sequence $X = \{x_1, x_2, ..., x_T\}$, where $T$ is the length of the tokenized sequence. This sequence is passed through a fine-tuned BERT encoder to generate contextualized token embeddings:
$$H = \{h_1, h_2, ..., h_T\}, \quad h_i \in \mathbb{R}^d$$
Where $d$ is the hidden dimension of the BERT encoder (768 for base BERT), and $h_i$ is the context-aware embedding for token $i$. The BERT encoder also outputs a special <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token embedding $h_{\mathsf{cls}} \in \mathbb{R}^d$, a pooled representation of the entire input sequence used later for the gating network.

### Step 2: Specialized Expert Modules
Six parallel expert modules process the BERT token embeddings, each targeting a distinct linguistic signal for stance detection:
#### Expert 1: Mean Pooling Expert (Global Semantic Orientation)
This expert captures the overall consistent stance direction of the entire text, effective when stance is expressed uniformly across all tokens (e.g., "Palestinians deserve statehood and equal rights" is consistently Pro-Palestine).
$$e_1 = W_1 \left( \frac{1}{T} \sum_{i=1}^{T} h_i \right)$$
Symbol explanation:
- $e_1$: Output embedding of the mean pooling expert
- $W_1$: Trainable linear projection layer for this expert, mapping the pooled embedding to a fixed expert output dimension
- $\frac{1}{T} \sum_{i=1}^{T} h_i$: Average of all contextual token embeddings, capturing global semantic meaning

#### Expert 2: Max Pooling Expert (Salient Lexical Cues)
This expert captures isolated, highly indicative stance tokens (e.g., "occupation" in "This ongoing occupation must end", "terrorist" in "Terrorist attacks cannot be justified") by selecting the highest activation value across each dimension of the token embeddings.
$$e_2 = W_2 \left( \underset{i=1}{\overset{T}{\operatorname{max}}} h_i \right)$$
Symbol explanation:
- $e_2$: Output embedding of the max pooling expert
- $W_2$: Trainable linear projection layer for this expert
- $\underset{i=1}{\overset{T}{\operatorname{max}}} h_i$: Element-wise maximum of all contextual token embeddings, highlighting the most salient stance-related features

#### Expert 3: Self-Attention Pooling Expert (Clause-Level Focus)
This expert identifies and weights stance-relevant clauses in multi-clause sentences, where stance is concentrated in a single part of the text (e.g., "While the humanitarian situation is tragic, Israel must respond to security threats" has its main stance in the second clause). First, it calculates normalized attention weights for each token:
$$\alpha_i = \frac{\mathsf{exp}(\mathsf{tanh}(h_i^\top v))}{\sum_{j=1}^{T} \mathsf{exp}(\mathsf{tanh}(h_j^\top v))}$$
Then it computes the weighted sum of token embeddings:
$$e_3 = W_3 \left( \sum_{i=1}^{T} \alpha_i h_i \right)$$
Symbol explanation:
- $\alpha_i$: Normalized attention weight for token $i$, ranges between 0 and 1, sum of all $\alpha_i = 1$
- $v$: Trainable attention vector that learns to identify stance-relevant tokens
- $e_3$: Output embedding of the self-attention pooling expert
- $W_3$: Trainable linear projection layer for this expert
- $\sum_{i=1}^{T} \alpha_i h_i$: Weighted sum of token embeddings, prioritizing stance-relevant clauses

#### Expert 4: Multi-Kernel CNN Expert (Phrase-Level Patterns)
This expert captures short, slogan-like n-gram patterns that indicate stance (e.g., "Stand with Israel", "Free Palestine") using 1D convolutions of multiple kernel sizes.
$$e_4 = W_4 \left( \mathsf{Concat}(\mathsf{MeanPool}(\mathsf{ReLU}(\mathsf{Conv}_k(H)))) \right)$$
Symbol explanation:
- $e_4$: Output embedding of the multi-kernel CNN expert
- $W_4$: Trainable linear projection layer for this expert
- $\mathsf{Conv}_k(H)$: 1D convolution operation with kernel size $k$, where $k \in \{2,3,4,5\}$ to capture 2-gram to 5-gram patterns
- $\mathsf{ReLU}$: Rectified Linear Unit activation function, introducing non-linearity
- $\mathsf{MeanPool}$: Mean pooling operation to reduce the sequence length of convolution outputs
- $\mathsf{Concat}$: Concatenation of pooled outputs from all kernel sizes, combining different n-gram pattern features

#### Expert 5: Lexical Cue-Aware Expert (Framing Indicators)
This expert identifies neutral reporting language by targeting pre-defined lexical cues (e.g., "claims", "reports", "states") that indicate the author is presenting information rather than expressing a personal stance (e.g., "According to officials, negotiations are ongoing").
$$e_5 = W_5 \left( \frac{\sum_{i \in C} h_i}{|C| + \epsilon} \right)$$
Symbol explanation:
- $e_5$: Output embedding of the lexical cue-aware expert
- $W_5$: Trainable linear projection layer for this expert
- $C$: Set of positions of pre-defined framing/reporting lexical cues in the input
- $|C|$: Number of cue tokens in the input
- $\epsilon$: Small constant ($10^{-8}$) to avoid division by zero when no cue tokens are present
- $\frac{\sum_{i \in C} h_i}{|C| + \epsilon}$: Average embedding of all cue tokens, capturing neutral framing signals

#### Expert 6: Contrast-Aware Expert (Discourse Shift Modeling)
This expert captures stance shifts indicated by contrast markers (e.g., "but", "however"), where the clause following the contrast marker often carries the main stance (e.g., "I support peace efforts, but the blockade must end"). First, it amplifies the embeddings of contrast markers by a factor of 3:
$$\tilde{h}_i = \left\{ \begin{array}{ll} 3h_i & \mathsf{if}~i \in D, \\ h_i & \mathsf{otherwise}, \end{array} \right.$$
Then it computes the average of adjusted embeddings:
$$e_6 = W_6 \left( \frac{\sum_{i=1}^{T} \tilde{h}_i}{|D| + \epsilon} \right)$$
Symbol explanation:
- $e_6$: Output embedding of the contrast-aware expert
- $W_6$: Trainable linear projection layer for this expert
- $D$: Set of positions of discourse contrast markers in the input
- $|D|$: Number of contrast markers in the input
- $\epsilon$: Small constant to avoid division by zero
- $\tilde{h}_i$: Adjusted token embedding, with contrast marker embeddings amplified to increase their influence

### Step 3: Context-Aware Gating and Fusion
Instead of weighting all experts equally, a gating network uses the <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> embedding to dynamically compute weights for each expert based on input characteristics:
$$g = \mathsf{Softmax}(W_g h_{\mathsf{cls}} + b_g)$$
Symbol explanation:
- $g \in \mathbb{R}^6$: Normalized gating weights for the 6 experts, where $g_i \geq 0$ for all $i$ and $\sum_{i=1}^6 g_i = 1$
- $W_g \in \mathbb{R}^{6 \times d}$: Trainable linear layer for the gating network
- $b_g \in \mathbb{R}^6$: Trainable bias term for the gating network
- $\mathsf{Softmax}$: Activation function that normalizes logits to valid weights summing to 1

  Next, the final fused MoE embedding is computed as the weighted sum of expert outputs:
$$h_{\mathsf{moe}} = \sum_{i=1}^{6} g_i e_i$$
Symbol explanation:
- $h_{\mathsf{moe}}$: Final fused embedding used for classification
- $g_i$: Gating weight for expert $i$
- $e_i$: Output embedding of expert $i$

  Finally, the fused embedding is passed to a linear classifier to generate predicted class probabilities for the 3 stance classes (Pro-Palestine, Pro-Israel, Neutral):
$$\hat{y} = \mathsf{softmax}(W_o h_{\mathsf{moe}} + b_o)$$
Symbol explanation:
- $\hat{y} \in \mathbb{R}^3$: Predicted class probabilities
- $W_o \in \mathbb{R}^{3 \times d_o}$: Trainable linear classification layer, where $d_o$ is the dimension of expert outputs
- $b_o \in \mathbb{R}^3$: Trainable bias term for the classification layer

### Training and Inference Details
The model is trained end-to-end using cross-entropy loss with label smoothing. Stratified 10-fold cross validation is used on the combined training and development set. During inference, weighted logit averaging is performed across the 10 folds, with higher weight assigned to folds that achieved higher validation F1 scores.

# 5. Experimental Setup
## 5.1. Datasets
The experiments use the StanceNakba 2026 Shared Task Subtask A dataset, curated by Aldous et al. (2026) for LREC 2026:
- **Source**: Collected from public social media discourse related to the Nakba (the 1948 Palestinian expulsion and flight).
- **Scale**: 1401 annotated English texts.
- **Characteristics**: The target actors (Israel and Palestine) are implicit in all texts, not explicitly named, requiring the model to infer stance from context. Labels are 3 balanced classes: Pro-Palestine, Pro-Israel, Neutral.
- **Split**: Official 70% training / 15% development / 15% held-out test set split. The authors used the combined 85% training+development set for stratified 10-fold cross validation, with the 15% test set kept strictly unseen during all training steps.
- **Sample example**: Input text: "This ongoing occupation must end" → Label: Pro-Palestine.
  This dataset is ideal for validation because it is specifically designed for the implicit actor-level stance detection task StanceMoE targets, representing real-world sensitive geopolitical discourse where stances are often expressed indirectly.
## 5.2. Evaluation Metrics
The paper uses four standard multi-class classification metrics, explained in detail below:
### 1. Accuracy (Acc)
- **Conceptual Definition**: Measures the overall proportion of correctly classified samples out of all samples. It provides a general performance overview but can be misleading for imbalanced datasets.
- **Formula**:
  $$\text{Acc} = \frac{\text{Number of correctly predicted samples}}{\text{Total number of samples}}$$
- **Symbol Explanation**: For multi-class classification, this counts all samples where the predicted label matches the true label, divided by the total sample count.

### 2. Macro Precision (Pre)
- **Conceptual Definition**: Calculates precision (proportion of predicted positive samples that are actually positive) for each class independently, then averages them equally across all classes, ensuring each class contributes the same weight regardless of its size.
- **Formula**:
  $$\text{Macro Pre} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c}$$
- **Symbol Explanation**:
  - $C$: Number of classes (3 for this task)
  - $TP_c$: True Positives for class $c$ (samples correctly predicted as class $c$)
  - $FP_c$: False Positives for class $c$ (samples incorrectly predicted as class $c$)

### 3. Macro Recall (Rec)
- **Conceptual Definition**: Calculates recall (proportion of actual positive samples that are correctly predicted) for each class independently, then averages them equally across all classes.
- **Formula**:
  $$\text{Macro Rec} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}$$
- **Symbol Explanation**:
  - $FN_c$: False Negatives for class $c$ (samples of class $c$ incorrectly predicted as other classes)

### 4. Macro F1-Score (F1)
- **Conceptual Definition**: The primary evaluation metric for the task, it is the harmonic mean of macro precision and macro recall, balancing both metrics. It is robust to class imbalance, as it treats all classes equally.
- **Formula**:
  $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^{C} 2 \times \frac{\text{Pre}_c \times \text{Rec}_c}{\text{Pre}_c + \text{Rec}_c}$$
- **Symbol Explanation**:
  - $\text{Pre}_c$: Precision for class $c$
  - $\text{Rec}_c$: Recall for class $c$
## 5.3. Baselines
The paper compares StanceMoE against 11 representative baseline models covering all major prior stance detection approaches:
1. **Traditional ML Baselines (TF-IDF features)**:
   - Logistic Regression (LR): Linear classifier effective for high-dimensional text data
   - Multinomial Naive Bayes (MNB): Probabilistic model based on word frequency distributions
   - Support Vector Machine (SVM): Margin-based classifier robust to sparse text feature spaces
   - Random Forest (RF): Ensemble tree-based classifier that reduces overfitting via bagging
2. **Deep Neural Network Baselines**:
   - BiLSTM: Bidirectional Long Short-Term Memory network that captures sequential context
   - Target-specific Attentional Network (TAN): BiLSTM with target-specific attention, a standard state-of-the-art stance detection model
   - Gated Convolutional Network with Aspect Embedding (GCAE): Convolutional model with gating to filter non-target features
   - CrossNet: Aspect-based attention model for enhanced target-aware feature extraction
3. **BERT-based Baselines**:
   - Vanilla BERT: Standard fine-tuned BERT model with <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token classification
   - Stacked Expert BERT: Expert modules applied sequentially instead of in parallel
   - Fusion Expert BERT: Expert outputs fused with equal static weights, no adaptive gating
     These baselines are representative of all major existing approaches to stance detection, so comparison against them reliably validates the performance gains of StanceMoE's design.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The following are the results from Table 1 of the original paper, comparing all baseline models and StanceMoE on the held-out test set:

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th colspan="4">K-fold (mean±std)</th>
<th colspan="4">Weighted Logit Ensemble</th>
</tr>
<tr>
<th>Acc</th>
<th>Pre</th>
<th>Rec</th>
<th>F1</th>
<th>Acc</th>
<th>Pre</th>
<th>Rec</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>LR</td>
<td>80.91±1.79</td>
<td>80.90±1.78</td>
<td>80.93±1.78</td>
<td>80.86±1.78</td>
<td>81.28</td>
<td>81.25</td>
<td>81.33</td>
<td>81.22</td>
</tr>
<tr>
<td>MNB</td>
<td>77.25±1.64</td>
<td>75.88±2.02</td>
<td>78.49±1.65</td>
<td>77.20±1.65</td>
<td>77.38</td>
<td>76.02</td>
<td>78.63</td>
<td>77.33</td>
</tr>
<tr>
<td>SVM</td>
<td>83.25±1.71</td>
<td>84.73±1.68</td>
<td>83.26±1.71</td>
<td>83.27±1.72</td>
<td>83.37</td>
<td>84.94</td>
<td>83.35</td>
<td>83.43</td>
</tr>
<tr>
<td>RF</td>
<td>84.05±1.83</td>
<td>84.73±1.73</td>
<td>84.05±1.82</td>
<td>84.05±1.90</td>
<td>84.16</td>
<td>84.83</td>
<td>84.14</td>
<td>84.19</td>
</tr>
<tr>
<td>BiLSTM</td>
<td>85.63±2.99</td>
<td>85.87±3.00</td>
<td>85.63±2.99</td>
<td>85.51±3.07</td>
<td>85.74</td>
<td>85.98</td>
<td>85.74</td>
<td>85.72</td>
</tr>
<tr>
<td>TAN</td>
<td>85.99±3.10</td>
<td>86.01±2.81</td>
<td>86.36±2.08</td>
<td>85.91±2.15</td>
<td>86.13</td>
<td>86.10</td>
<td>86.43</td>
<td>86.03</td>
</tr>
<tr>
<td>GCAE</td>
<td>87.69±2.85</td>
<td>87.98±2.92</td>
<td>87.48±2.16</td>
<td>87.49±2.78</td>
<td>87.82</td>
<td>88.10</td>
<td>87.53</td>
<td>87.62</td>
</tr>
<tr>
<td>CrossNet</td>
<td>85.13±2.28</td>
<td>85.55±1.89</td>
<td>84.90±1.90</td>
<td>84.77±2.25</td>
<td>85.29</td>
<td>85.65</td>
<td>84.97</td>
<td>84.88</td>
</tr>
<tr>
<td>BERT</td>
<td>89.77±2.35</td>
<td>90.04±2.30</td>
<td>89.77±2.31</td>
<td>89.61±2.29</td>
<td>90.05</td>
<td>90.28</td>
<td>90.03</td>
<td>89.86</td>
</tr>
<tr>
<td>Stacked</td>
<td>91.61±2.46</td>
<td>91.73±2.43</td>
<td>91.62±2.44</td>
<td>91.60±2.47</td>
<td>91.94</td>
<td>92.14</td>
<td>91.93</td>
<td>91.83</td>
</tr>
<tr>
<td>Fusion</td>
<td>91.03±2.26</td>
<td>91.20±2.29</td>
<td>91.03±2.24</td>
<td>91.02±2.26</td>
<td>91.18</td>
<td>91.45</td>
<td>91.17</td>
<td>91.17</td>
</tr>
<tr>
<td>StanceMoE</td>
<td>94.09±1.11</td>
<td>94.18±1.12</td>
<td>94.08±1.12</td>
<td>94.03±1.12</td>
<td>94.31</td>
<td>94.45</td>
<td>94.31</td>
<td>94.26</td>
</tr>
</tbody>
</table>

### Key Result Insights:
1. Traditional ML baselines perform the worst, with Random Forest achieving the highest F1 of 84.19%, showing the limitation of handcrafted features for implicit stance detection.
2. DNN baselines outperform traditional ML, with GCAE achieving 87.62% F1, confirming the benefit of automatic feature extraction for stance tasks.
3. Vanilla BERT outperforms all DNN baselines with 89.86% F1, showing the strength of pre-trained contextual embeddings.
4. Stacked and Fusion expert BERT variants achieve 91.83% and 91.17% F1 respectively, showing that adding expert modules improves performance, but static integration limits gains.
5. StanceMoE outperforms all baselines by a large margin, achieving 94.26% F1: +4.4% over vanilla BERT, +2.43% over stacked expert BERT. It also has the lowest cross-validation standard deviation (±1.1%), confirming it is more stable and robust than all other models.

   The following figure (Figure 2 from the original paper) shows the confusion matrix for StanceMoE on the test set:

   ![Figure 2: Confusion matrix using the proposed StanceMoE architecture.](images/2.jpg)
   *该图像是一个混淆矩阵，展示了使用StanceMoE架构进行的分类结果。矩阵中展示了真实标签与预测标签的对应关系，反映出模型在中立、支持以色列和支持巴勒斯坦三类上的分类精度。整体上，模型在各类中表现良好，特别是在支持以色列的类别上达到了70的预测准确度。*

The confusion matrix confirms strong performance across all classes, with perfect recall for the Pro-Israel class, and most misclassifications occurring between the Neutral class and the two partisan classes, as expected for implicit stance detection.

## 6.2. Ablation Studies / Parameter Analysis
Ablation studies were conducted to measure the contribution of each expert module by removing them one at a time and measuring performance degradation. The following are the results from the overall ablation study (Table 4 of the original paper):

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th colspan="4">K-fold (mean±std)</th>
<th colspan="4">Weighted Logit Ensemble</th>
</tr>
<tr>
<th>Acc</th>
<th>Pre</th>
<th>Rec</th>
<th>F1</th>
<th>Acc</th>
<th>Pre</th>
<th>Rec</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o Mean</td>
<td>91.75±1.76</td>
<td>92.02±1.54</td>
<td>91.74±1.77</td>
<td>91.65±1.86</td>
<td>92.89</td>
<td>93.05</td>
<td>92.88</td>
<td>92.80</td>
</tr>
<tr>
<td>w/o Max</td>
<td>91.52±1.23</td>
<td>91.83±1.06</td>
<td>91.51±1.23</td>
<td>91.43±1.30</td>
<td>93.36</td>
<td>93.56</td>
<td>93.35</td>
<td>93.29</td>
</tr>
<tr>
<td>w/o Self-att</td>
<td>91.47±1.04</td>
<td>91.65±0.99</td>
<td>91.46±1.04</td>
<td>91.38±1.08</td>
<td>91.94</td>
<td>92.04</td>
<td>91.93</td>
<td>91.83</td>
</tr>
<tr>
<td>w/o CNN</td>
<td>92.09±1.04</td>
<td>92.20±0.97</td>
<td>92.08±1.04</td>
<td>92.00±1.11</td>
<td>93.36</td>
<td>93.38</td>
<td>93.37</td>
<td>93.29</td>
</tr>
<tr>
<td>w/o Lexical-cue</td>
<td>91.70±1.84</td>
<td>91.83±1.84</td>
<td>91.70±1.84</td>
<td>91.67±1.83</td>
<td>93.36</td>
<td>93.41</td>
<td>93.36</td>
<td>93.32</td>
</tr>
<tr>
<td>w/o Contrastive</td>
<td>91.75±1.81</td>
<td>91.99±1.67</td>
<td>91.74±1.82</td>
<td>91.64±1.87</td>
<td>91.94</td>
<td>92.13</td>
<td>91.94</td>
<td>91.82</td>
</tr>
<tr>
<td>StanceMoE</td>
<td>94.09±1.11</td>
<td>94.18±1.12</td>
<td>94.08±1.12</td>
<td>94.03±1.12</td>
<td>94.31</td>
<td>94.45</td>
<td>94.31</td>
<td>94.26</td>
</tr>
</tbody>
</table>

The following are the results from the class-wise ablation study of StanceMoE:

<table>
<thead>
<tr>
<th rowspan="2">Methods</th>
<th rowspan="2">Acc</th>
<th colspan="3">Pro-Palestine</th>
<th colspan="3">Pro-Israel</th>
<th colspan="3">Neutral</th>
</tr>
<tr>
<th>Pre</th>
<th>Rec</th>
<th>F1</th>
<th>Pre</th>
<th>Rec</th>
<th>F1</th>
<th>Pre</th>
<th>Rec</th>
<th>F1</th>
</tr>
</thead>
<tbody>
<tr>
<td>w/o Mean</td>
<td>92.89</td>
<td>89.47</td>
<td>95.75</td>
<td>92.52</td>
<td>94.52</td>
<td>98.57</td>
<td>96.50</td>
<td>95.16</td>
<td>84.29</td>
<td>89.39</td>
</tr>
<tr>
<td>w/o Max</td>
<td>93.36</td>
<td>91.89</td>
<td>95.77</td>
<td>93.79</td>
<td>92.00</td>
<td>98.57</td>
<td>95.17</td>
<td>96.77</td>
<td>85.71</td>
<td>90.91</td>
</tr>
<tr>
<td>w/o Self-Attention</td>
<td>91.94</td>
<td>89.33</td>
<td>94.37</td>
<td>91.78</td>
<td>93.24</td>
<td>98.57</td>
<td>95.83</td>
<td>93.55</td>
<td>82.86</td>
<td>87.88</td>
</tr>
<tr>
<td>w/o CNN</td>
<td>93.36</td>
<td>92.96</td>
<td>92.96</td>
<td>92.96</td>
<td>93.33</td>
<td>99.26</td>
<td>96.55</td>
<td>93.85</td>
<td>87.14</td>
<td>90.37</td>
</tr>
<tr>
<td>w/o Lexical-cue</td>
<td>93.36</td>
<td>90.54</td>
<td>94.37</td>
<td>92.41</td>
<td>95.83</td>
<td>98.57</td>
<td>97.18</td>
<td>95.08</td>
<td>82.86</td>
<td>88.55</td>
</tr>
<tr>
<td>w/o Contrastive</td>
<td>91.94</td>
<td>90.41</td>
<td>92.96</td>
<td>91.67</td>
<td>90.91</td>
<td>99.32</td>
<td>95.24</td>
<td>93.85</td>
<td>87.14</td>
<td>90.37</td>
</tr>
<tr>
<td>StanceMoE</td>
<td>94.31</td>
<td>94.37</td>
<td>94.37</td>
<td>94.37</td>
<td>92.11</td>
<td>100</td>
<td>95.89</td>
<td>96.88</td>
<td>88.57</td>
<td>92.54</td>
</tr>
</tbody>
</table>

### Ablation Insights:
1. Removing any expert module reduces overall F1 score, confirming that all 6 experts contribute complementary, non-redundant signals to the model.
2. The largest performance drops occur when removing the Self-Attention expert (F1 drops 2.43% to 91.83%) and the Contrastive expert (F1 drops 2.44% to 91.82%), showing these two components are the most critical for capturing clause-level focus and discourse shifts, which are ubiquitous in geopolitical argumentation.
3. Removing the Mean and Max pooling experts also leads to significant drops (1.46% and 0.97% F1 respectively), confirming the importance of global semantic orientation and salient lexical cues for stance detection.
4. Class-wise analysis shows the lexical-cue expert is particularly important for the Pro-Palestine class, while the contrastive expert improves performance across all classes. The full StanceMoE model achieves the most balanced performance across all 3 classes, especially for the hard-to-detect Neutral class.

### Error Analysis
The authors identify four main systematic error sources:
1. Over-reliance on strong lexical cues: Religious praise and humanitarian language are sometimes misinterpreted as explicit political stance.
2. Confusion between topic discussion and stance expression: Mentioning actor-related topics (e.g., antisemitism norms) is sometimes misclassified as taking a stance.
3. Complex multi-clause reasoning: Intricate discourse structures with multiple levels of argumentation are occasionally misclassified, even with the contrastive expert.
4. Short ambiguous inputs: Very short, context-poor statements are inherently ambiguous and often misclassified based on training pattern associations.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces StanceMoE, a context-enhanced Mixture-of-Experts architecture for implicit actor-level stance detection that addresses the limitation of unified transformer representations for capturing heterogeneous stance-related linguistic signals. By decomposing stance modeling into 6 specialized complementary expert modules, and fusing their outputs via a context-aware gating mechanism that adapts to input characteristics, StanceMoE achieves a state-of-the-art macro-F1 score of 94.26% on the StanceNakba 2026 Subtask A dataset, outperforming all traditional and BERT-based baselines by a large margin, and securing 3rd place in the shared task. Ablation studies confirm that all expert modules contribute complementary signals, with self-attention and contrast-aware experts being the most critical components. The work demonstrates that explicitly disentangling and dynamically integrating diverse linguistic signals significantly improves stance detection performance for implicit targets in sensitive geopolitical discourse.
## 7.2. Limitations & Future Work
### Limitations Identified by Authors
1. The model struggles with nuanced implicitly framed discourse, complex multi-clause reasoning, and very short ambiguous inputs, which account for all systematic misclassifications.
2. The model is only evaluated on a single English dataset focused on Nakba-related discourse, so its generalization to other domains, topics, and languages is untested.
3. The lexical-cue and contrast-aware experts rely on pre-defined static cue lists, which require manual adjustment for cross-domain or cross-language transfer.

### Suggested Future Work
1. Extend the framework to explicitly target-aware stance modeling for multi-target stance detection tasks.
2. Evaluate cross-topic and cross-domain transfer performance to validate the model's generalization capacity.
3. Investigate multilingual adaptation of the model to support stance detection in non-English geopolitical discourse.
## 7.3. Personal Insights & Critique
### Key Inspirations
The core design of StanceMoE has broad applicability beyond geopolitical stance detection:
1. The paradigm of decomposing a complex fine-grained classification task into specialized expert modules targeting distinct signal types can be adapted to other NLP tasks like hate speech detection, misinformation detection, and aspect-based sentiment analysis, where heterogeneous linguistic signals drive classification.
2. The context-aware gating mechanism can be integrated with other transformer-based models to improve performance on tasks where input characteristics vary widely and static feature weighting is suboptimal.
3. The model's strong performance on implicit stance detection addresses a critical gap for real-world applications like social media monitoring, where targets are often not explicitly named.

### Potential Improvements
1. The current model uses pre-defined static cue lists for lexical and contrast markers. These could be replaced with learnable cue detection modules, eliminating manual rule engineering and improving cross-domain generalization.
2. Integrating the MoE architecture with large language models (LLMs) could leverage their world knowledge to better handle ambiguous and implicitly framed discourse, which is the main source of errors in the current model.
3. The gating mechanism currently uses only the <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token to compute expert weights. Using more sophisticated input representations (e.g., combining <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> with expert-specific signals) could improve weighting accuracy.
4. The model is only evaluated on a small 1401-sample dataset. Testing on larger, more diverse stance detection datasets would better validate its generalizability.

### Broader Applications
The StanceMoE architecture can be adapted for multiple high-impact use cases:
- Brand stance detection: Monitoring social media for user stance toward brands and products.
- Political stance detection: Identifying voter stance toward candidates and policies from social media posts.
- Public health stance detection: Identifying public stance toward public health measures like vaccination and climate policies.