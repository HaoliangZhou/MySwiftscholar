# 1. Bibliographic Information
## 1.1. Title
The paper is titled *MedAidDialog: A Multilingual Multi-Turn Medical Dialogue Dataset for Accessible Healthcare*. Its central focus is the development of a multilingual, multi-turn conversational medical dataset and a corresponding lightweight deployable model to enable accessible preliminary diagnostic support for low-resource healthcare settings.
## 1.2. Authors
The authors and their affiliations are:
1.  Shubham Kumar Nigam, University of Birmingham, Dubai, United Arab Emirates
2.  Suparnojit Sarkar, Heritage Institute of Technology, Kolkata, India
3.  Piyush Patel, Madan Mohan Malaviya University of Technology, India
    All three authors focus on natural language processing (NLP) applications for healthcare, particularly low-resource and multilingual systems.
## 1.3. Publication Venue & Status
As of the current date, the paper is published as a preprint on arXiv, a widely used open-access preprint server for computer science and healthcare research. It has not yet been peer-reviewed or accepted for publication at a formal conference or journal.
## 1.4. Publication Year
The paper was published on March 25, 2026 (UTC).
## 1.5. Abstract
The research addresses gaps in existing medical conversational AI systems, which are largely single-turn, template-based, monolingual, and require high-end computational infrastructure to deploy. The core contributions are:
1.  **MedAidDialog**: A 7-language parallel multi-turn medical dialogue dataset built by augmenting the existing MDDial English corpus with LLM-generated synthetic dialogues, covering English, Hindi, Telugu, Tamil, Bengali, Marathi, and Arabic.
2.  **MedAidLM**: A lightweight conversational medical model fine-tuned using parameter-efficient techniques on quantized small language models (SLMs), enabling deployment on consumer-grade hardware without high-end GPUs.
3.  Personalization support via optional patient pre-context (age, gender, allergies, medical history) to tailor consultations.
    Experimental results show MedAidLM achieves 90.21% diagnostic accuracy, with a 95.3% medical safety pass rate validated by independent medical experts.
## 1.6. Original Source Links
- Preprint abstract: https://arxiv.org/abs/2603.24132v1
- Full PDF: https://arxiv.org/pdf/2603.24132v1

  ---

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem Addressed
Conversational AI has strong potential to provide preliminary medical guidance in settings with shortages of healthcare workers, but existing systems suffer from three critical limitations:
1.  **Single-turn paradigm**: Most systems expect patients to provide all symptom information in one prompt, which does not reflect real clinical practice, where doctors iteratively ask follow-up questions to gather diagnostic evidence.
2.  **Lack of multilingual support**: Most medical dialogue datasets and models are limited to English or a small number of high-resource languages, making them unusable for populations who speak regional low-resource languages.
3.  **High deployment cost**: State-of-the-art medical LLMs are large and require specialized high-end GPU infrastructure to run, which is unavailable in most low-resource rural healthcare settings.
4.  **No personalization**: Most systems do not incorporate basic patient context (age, gender, allergies) that doctors use to tailor diagnostic reasoning.

### Novel Entry Point
The paper proposes a two-part solution: (1) a synthetically augmented, parallel multilingual multi-turn medical dialogue dataset that avoids the inflexibility of template-based datasets, and (2) a lightweight fine-tuned SLM that can run on consumer-grade hardware, with a translation layer to support multilingual interaction without training separate models per language.

## 2.2. Main Contributions / Findings
### Primary Contributions
The paper lists four explicit contributions:
1.  Introduction of the multilingual multi-turn medical dialogue generation task, and release of the MedAidDialog parallel corpus covering 7 languages for low-resource environments.
2.  Integration of optional patient pre-context support to enable personalized conversational medical assistance.
3.  Development of MedAidLM, a parameter-efficient fine-tuned model based on quantized SLMs, deployable without high-end computational infrastructure.
4.  Rigorous medical expert evaluation of the model’s clinical safety, plausibility, and conversational quality.

### Key Findings
1.  Combining the original template-based MDDial dataset with LLM-generated synthetic dialogues improves diagnostic accuracy by 10-60% compared to training on either data source alone.
2.  The 3B-parameter LLaMA-3.2-3B-Instruct model fine-tuned on MedAidDialog achieves 90.21% diagnostic accuracy, outperforming larger 7B-parameter models and smaller 1.5B models.
3.  The model has a 95.3% medical safety pass rate per expert evaluation, with strong inter-annotator agreement (average Krippendorff's α = 0.81) between evaluating clinicians.
4.  Multilingual support via a bidirectional translation layer eliminates the need to train separate dialogue models for each target language, reducing development and deployment overhead.

    ---

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
Below are definitions of core technical and clinical terms required to understand the paper, designed for beginners:
1.  **Multi-turn dialogue**: A conversational exchange with multiple back-and-forth turns between two parties (in this case, patient and doctor), as opposed to single-turn question-answering where one query receives one response.
2.  **Parameter-Efficient Fine-Tuning (PEFT)**: A set of techniques to fine-tune large language models while only updating a small subset of model parameters (typically <2% of total parameters), reducing training time, memory requirements, and compute cost.
3.  **Low-Rank Adaptation (LoRA)**: The most widely used PEFT technique, which freezes the pre-trained base model weights and adds small trainable low-rank matrices to the attention layers of the transformer. The number of trainable parameters is reduced by up to 100x compared to full fine-tuning.
4.  **Quantization**: A technique to reduce the precision of model weights from 32-bit floating point (standard for ML models) to lower precision (4-bit in this work), cutting memory usage by up to 8x and enabling models to run on consumer-grade GPUs with <16GB VRAM.
5.  **Parallel multilingual corpus**: A dataset where every entry has aligned translations across multiple languages, so the same dialogue is available in all target languages for training or evaluation.
6.  **Differential diagnosis**: The process of distinguishing between multiple diseases with similar symptoms to identify the most likely cause of a patient’s symptoms.
7.  **Krippendorff’s Alpha (α)**: A statistical measure of inter-annotator agreement (how consistent multiple independent evaluators are in their judgments), ranging from -1 (perfect disagreement) to 1 (perfect agreement). Scores >0.8 are considered strong agreement.
8.  **Supervised Fine-Tuning (SFT)**: The standard process of fine-tuning a pre-trained LLM on labeled task-specific data, using next-token prediction as the training objective.
9.  **Group Relative Policy Optimization (GRPO)**: A reinforcement learning (RL) technique for fine-tuning LLMs, which uses a reward signal to optimize model behavior for specific task goals (e.g., conversational quality, diagnostic accuracy).

## 3.2. Previous Works
The paper builds on three key lines of prior research:
1.  **Medical Dialogue Datasets**:
    - *MDDial (Macherla et al., 2023)*: The foundational dataset for this work, an English multi-turn differential diagnosis dialogue dataset built using template-based generation from structured medical records. It lacks linguistic diversity and multilingual support.
    - *MedDG (Liu et al., 2022)* and *Zhongjing (Yang et al., 2024)*: Chinese multi-turn medical dialogue datasets built from real clinical conversations, but limited to a single language.
    - *BiMediX (Pieri et al., 2024)*: A bilingual medical LLM supporting English and Arabic, but limited to only two languages and not designed for multi-turn dialogue.
2.  **Medical LLMs**:
    - *ChatDoctor (Li et al., 2023)*: A medical LLM fine-tuned on LLaMA, but optimized for single-turn question answering and requiring large compute resources to deploy.
    - *AMIE (Tu et al., 2024)*: A conversational diagnostic AI model that frames diagnosis as multi-turn history taking, but is a large proprietary model not deployable on low-resource hardware.
3.  **Efficient LLM Training**:
    - *LoRA (Hu et al., 2022)*: The core PEFT technique used in this work. The original LoRA formula for updating transformer attention weights is:
      $W = W_0 + BA$
      Where $W_0$ is the frozen pre-trained weight matrix, $A$ is a low-rank trainable matrix of size $r \times d$, $B$ is a low-rank trainable matrix of size $d \times r$, and $r$ (rank) is a small hyperparameter (typically 8-64, set to 16 in this work). The total number of trainable parameters is $2 \times d \times r$, which is orders of magnitude smaller than the size of $W_0$.

## 3.3. Technological Evolution
The field of medical conversational AI has evolved in three distinct phases:
1.  **2018-2021**: Template-based, task-oriented systems focused on slot filling for symptom collection, with very limited natural language flexibility.
2.  **2022-2024**: LLM-based single-turn medical QA systems, fine-tuned on medical textbooks and single-turn question-answer pairs, but not designed for multi-turn interaction.
3.  **2025-Present**: Emerging multi-turn conversational medical systems, with a focus on diagnostic reasoning and accessibility for low-resource settings. This paper’s work falls directly into this latest phase, addressing gaps in multilingual support and low-resource deployment.

## 3.4. Differentiation Analysis
Compared to prior work, this paper’s approach has three core unique advantages:
1.  **Multilingual coverage**: It is the first multi-turn medical dialogue dataset covering 7 widely spoken low-resource languages (including 5 Indian languages and Arabic), serving populations totaling over 2 billion people.
2.  **Lightweight deployment**: Unlike large proprietary medical LLMs that require high-end GPUs, MedAidLM runs on consumer-grade hardware (e.g., free Google Colab/Kaggle tiers, edge devices) thanks to quantization and PEFT.
3.  **Personalization support**: The framework integrates optional patient pre-context to tailor diagnostic reasoning, a feature missing from most existing open-source medical dialogue systems.

    ---

# 4. Methodology
## 4.1. Core Principles
The core design principle of the work is accessibility: the system is built to be usable in low-resource settings with limited healthcare staff, limited internet connectivity, and no specialized computational infrastructure. This principle guides every design choice:
- Synthetic data augmentation is used to avoid the cost and privacy risks of collecting real clinical dialogue data.
- Quantized SLMs and PEFT are used to eliminate the need for high-end training and deployment hardware.
- A translation layer is used instead of separate per-language models to reduce development overhead and support more languages with minimal additional work.
- Patient pre-context is integrated to align the system with real clinical workflow patterns, improving diagnostic accuracy and personalization.

## 4.2. Step-by-Step Methodology Breakdown
The framework has three sequential stages, as illustrated in Figure 3 below:

![Figure 3: Overview of the proposed framework. Stage 1: Data Augmentation. The MDDial dataset is expanded with synthetic medical dialogues, followed by coherence and diversity filtering. Stage 2: Model Adaptation. Compact open-source language models are fine-tuned using parameter-efficient training and LoRA-based SFT. The dotted connection indicates an optional GRPO optimisation stage applied to selected models. Stage 3: Deployment. The best-perorming checkpoint is deployed as MedAid, whicoperates withinmultilingualnference loo that incorporates optional patient pre-context and bidirectional translation.](images/3.jpg)
*Figure 3: Overview of the proposed framework, including data augmentation, model adaptation, and deployment stages.*

### 4.2.1. MedAidDialog Dataset Construction
#### Step 1: Synthetic Dialogue Generation
The dataset is built by augmenting the original MDDial English corpus with synthetic multi-turn dialogues generated using the Llama-3.3-70B-Versatile model via the Groq API. The generation process is conditioned on:
- 12 disease categories and 118 symptoms from the original MDDial dataset
- Randomized patient demographic profiles
- Stylistic constraints to ensure natural, non-template conversational flow
  Each dialogue follows a realistic clinical flow: patient presents an initial complaint → doctor asks follow-up questions to collect symptoms → doctor provides a final diagnosis. Dialogues are designed to be 4-8 turns long, matching the length of real primary care consultations.

Two quality control filters are applied to generated dialogues:
1.  **Coherence check**: Verifies that the final diagnosis is logically consistent with the symptoms described in the dialogue.
2.  **Diversity check**: MinHash near-duplicate removal to eliminate repetitive or near-identical dialogues.
    The 1101 filtered synthetic dialogues are merged with the original 1879 MDDial dialogues to form the 2980-dialogue English training corpus.

#### Step 2: Multilingual Expansion
The full English corpus is translated into 6 additional languages (Hindi, Telugu, Tamil, Bengali, Marathi, Arabic) using a combination of TranslateGemma and TinyAya, two lightweight multilingual translation models. A structured prompting strategy is used to ensure medical semantic accuracy during translation, resulting in a parallel corpus where every dialogue is available in all 7 languages.

### 4.2.2. Task Definition & Model Objective
A medical consultation dialogue is defined as a sequence of turns $D = \{u_1, u_2, ..., u_T\}$, where odd turns are patient utterances and even turns are doctor responses. Each dialogue is associated with a gold diagnostic label $y \in \mathcal{V}$, where $\mathcal{V}$ is the set of 12 diseases covered in the dataset.

Given a dialogue context of all previous turns up to step `t-1`:
$$
C_t = \{u_1, u_2, ..., u_{t-1}\}
$$
The model’s objective is to generate the next appropriate doctor response $u_t$:
$$
u_t = \arg \max_u P(u \mid C_t)
$$
Where:
- $C_t$ = the full context of prior conversation turns
- $P(u \mid C_t)$ = the probability of generating utterance $u$ given the context $C_t$
- $\arg \max_u$ = the model selects the utterance with the highest probability of being medically appropriate and coherent.

  Optional patient pre-context (age, gender, allergies, pre-existing conditions) is prepended to $C_t$ to enable personalized reasoning.

### 4.2.3. Model Training Pipeline
#### Step 1: Dialogue Formatting
All dialogues are converted to ShareGPT-style instruction tuning format:
1.  A system message defines the diagnostic consultation setting
2.  Patient utterances are mapped to "human" turns
3.  Doctor utterances are mapped to "gpt" turns
4.  The final assistant turn contains the final diagnosis and recommendation

#### Step 2: Parameter-Efficient Fine-Tuning
Four compact open-source SLMs are evaluated:
1.  LLaMA-3.2-3B-Instruct
2.  Mistral-7B-Instruct
3.  Qwen3-4B
4.  DeepSeek-R1-Distill-Qwen-1.5B
    All models are loaded in 4-bit NF4 quantized format to reduce memory usage, enabling training on consumer GPUs with <16GB VRAM.

LoRA adapters are inserted into all attention projection layers of the transformer, with the following hyperparameters:
- Rank $r=16$
- LoRA alpha = 16
- LoRA dropout = 0
- Trainable parameters = ~2% of total model parameters

  Models are trained via supervised fine-tuning (SFT) for 3 epochs using 8-bit AdamW optimizer with a linear learning rate schedule, learning rate = $2 \times 10^{-4}$, and 5 warmup steps.

An optional GRPO reinforcement learning fine-tuning stage is also tested, with a reward signal combining diagnostic accuracy, symptom coverage, output format compliance, and KL-divergence regularization to avoid deviation from the supervised model.

### 4.2.4. Multilingual Inference Pipeline
The best-performing fine-tuned model (MedAidLM, based on LLaMA-3.2-3B-Instruct) operates in English, so a bidirectional translation layer is added to support multilingual interaction:
1.  Patient input in their native language is translated to English
2.  The English input, patient pre-context, and dialogue history are passed to MedAidLM
3.  MedAidLM generates an English response
4.  The response is translated back to the patient’s native language and displayed
    The process repeats until the model generates a dedicated `[PREDICT]` marker, indicating it has collected sufficient information to provide a final diagnosis.

A comparison of the behavior of general-purpose LLMs vs MedAidLM is shown in Figures 1 and 2 below:

![Figure 1: Example response from a general-purpose LLM (ChatGPT 5.3). The model produces a single explanatory response without collecting additional symptoms or conducting follow-up questioning.](images/1.jpg)
*Figure 1: General-purpose LLMs provide a single static response without follow-up questioning, unlike real clinical practice.*

![Figure 2: Example interaction with MedAidLM. The system first incorporates patient pre-context information (e.g., age, gender, and allergies) and then performs multiturn dialogue to collect symptoms before producing a diagnostic recommendation.](images/2.jpg)
*Figure 2: MedAidLM uses patient pre-context and multi-turn follow-up questioning to collect symptoms before providing a diagnosis, matching real clinical workflow.*

---

# 5. Experimental Setup
## 5.1. Datasets
The experiments use the MedAidDialog corpus, which combines the original MDDial dataset and the synthetically generated dialogues. The statistics of the dataset are shown in Table 1 below:
The following are the results from Table 1 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Dataset</th>
<th colspan="4">Dialogue Turns</th>
<th colspan="3">Average Words</th>
</tr>
<tr>
<th>Avg Turns</th>
<th>Total Dialogues</th>
<th>Min Turns</th>
<th>Max Turns</th>
<th>Per Dialogue</th>
<th>Patient Utterance</th>
<th>Doctor Utterance</th>
</tr>
</thead>
<tbody>
<tr>
<td>MDDial (MD)</td>
<td>4.9</td>
<td>1879</td>
<td>1</td>
<td>16</td>
<td>53.5</td>
<td>5.6</td>
<td>6.7</td>
</tr>
<tr>
<td>Synthetic (SYN)</td>
<td>6.6</td>
<td>1101</td>
<td>5</td>
<td>11</td>
<td>134.5</td>
<td>8.8</td>
<td>9.6</td>
</tr>
<tr>
<td>MD + SYN</td>
<td>5.7</td>
<td>2980</td>
<td>1</td>
<td>16</td>
<td>86.9</td>
<td>7.00</td>
<td>8.05</td>
</tr>
<tr>
<td>MDDial Test</td>
<td>5.9</td>
<td>237</td>
<td>1</td>
<td>13</td>
<td>55.4</td>
<td>5.6</td>
<td>6.6</td>
</tr>
</tbody>
</table>

The combined MD+SYN corpus is used for training, and the held-out MDDial test set (237 dialogues) is used for evaluation. This setup ensures the model is evaluated on unseen template-based dialogues that reflect real diagnostic patterns, validating its generalizability.

## 5.2. Evaluation Metrics
The paper uses a two-stage evaluation strategy combining automatic and expert evaluation metrics:
### 5.2.1. Automatic Metric: Diagnostic Accuracy
1.  **Conceptual Definition**: The percentage of final diagnoses generated by the model that match the gold standard disease label associated with the test dialogue. It measures the model’s core diagnostic reasoning capability.
2.  **Mathematical Formula**:
    $$
    \text{Accuracy} = \frac{N_{\text{correct}}}{N_{\text{total}}}
    $$
3.  **Symbol Explanation**:
    - $N_{\text{correct}}$ = number of test dialogues where the model’s final diagnosis matches the gold label
    - $N_{\text{total}}$ = total number of test dialogues

### 5.2.2. Expert Evaluation Metrics
All expert metrics are scored by 3 independent qualified medical practitioners (MBBS degree holders in postgraduate training):
1.  **Medical Safety**: Binary pass/fail metric measuring whether the model generates any unsafe, misleading, or dangerous medical advice.
2.  **Symptom Extraction**: 1-5 Likert scale measuring how accurately the model identifies and tracks patient symptoms across turns.
3.  **Context Memory**: 1-5 Likert scale measuring whether the model remembers previously mentioned patient information across dialogue turns.
4.  **Diagnostic Correctness**: 1-5 Likert scale measuring whether the final diagnosis is medically plausible given the collected symptoms.
5.  **Conversational Flow**: 1-5 Likert scale measuring whether the dialogue is natural, empathetic, and clinically appropriate, matching real clinical interaction patterns.
6.  **Efficiency**: 1-5 Likert scale measuring whether the model asks an appropriate number of questions, avoiding redundant queries while collecting sufficient diagnostic information.

### 5.2.3. Inter-Annotator Agreement: Krippendorff’s Alpha
1.  **Conceptual Definition**: Measures the consistency of judgments across the 3 medical experts, to validate that evaluation scores are reliable and not due to random chance.
2.  **Mathematical Formula**:
    $\alpha = 1 - \frac{D_o}{D_e}$
3.  **Symbol Explanation**:
    - $D_o$ = observed disagreement between annotators
    - $D_e$ = expected disagreement between annotators due to random chance
      Scores range from -1 (perfect disagreement) to 1 (perfect agreement), with scores >0.8 indicating strong, reliable agreement.

## 5.3. Baselines
The paper compares the performance of LLaMA-3.2-3B-Instruct (MedAidLM) against three other compact SLMs as baselines:
1.  **Mistral-7B-Instruct**: A widely used 7B-parameter open-source SLM with strong general conversational performance.
2.  **Qwen3-4B**: A 4B-parameter open-source SLM optimized for multilingual tasks.
3.  **DeepSeek-R1-Distill-Qwen-1.5B**: A small 1.5B-parameter distilled reasoning model optimized for low-resource deployment.

    Ablation studies also compare performance across:
1.  Training on only original MDDial (MD) data, only synthetic (SYN) data, and combined MD+SYN data
2.  Training with SFT only vs SFT + GRPO reinforcement learning

    ---

# 6. Results & Analysis
## 6.1. Core Results Analysis
The main automatic evaluation results for all model families trained on the full MedAidDialog (MD+SYN) corpus are shown in Table 2 below:
The following are the results from Table 2 of the original paper:

<table>
<thead>
<tr>
<th>Model Family</th>
<th>Dataset</th>
<th>Method</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>Mistral-7B-Instruct</td>
<td>MedAidDialog (MD+SYN)</td>
<td>SFT</td>
<td>88.09%</td>
</tr>
<tr>
<td>LLaMA 3.2 3B (MedAidLM)</td>
<td>MedAidDialog (MD+SYN)</td>
<td>SFT</td>
<td>90.21%</td>
</tr>
<tr>
<td>Qwen3-4B</td>
<td>MedAidDialog (MD+SYN)</td>
<td>SFT</td>
<td>80.00%</td>
</tr>
<tr>
<td>DeepSeek-R1-Distill-Qwen-1.5B</td>
<td>MedAidDialog (MD+SYN)</td>
<td>SFT</td>
<td>40.00%</td>
</tr>
</tbody>
</table>

Key takeaways:
1.  MedAidLM (LLaMA 3.2 3B) achieves the highest accuracy of 90.21%, outperforming even the larger 7B-parameter Mistral model, demonstrating that smaller models can achieve strong performance when fine-tuned on high-quality domain-specific data.
2.  The 1.5B-parameter DeepSeek model performs poorly, indicating that models below 3B parameters may lack sufficient reasoning capacity for multi-turn medical diagnostic tasks.

## 6.2. Ablation Study Results
The ablation study results analyzing the impact of dataset composition and training strategy are shown in Table 3 below:
The following are the results from Table 3 of the original paper:

<table>
<thead>
<tr>
<th>Model</th>
<th>Dataset</th>
<th>Avg. Turns</th>
<th>Dialogs</th>
<th>Method</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>Mistral-7B-Instruct</td>
<td>MD</td>
<td>4.90</td>
<td>1879</td>
<td>SFT</td>
<td>18.72%</td>
</tr>
<tr>
<td>Mistral-7B-Instruct</td>
<td>SYN</td>
<td>7.28</td>
<td>1101</td>
<td>SFT</td>
<td>61.28%</td>
</tr>
<tr>
<td>Mistral-7B-Instruct</td>
<td>MD+SYN</td>
<td>5.78</td>
<td>2980</td>
<td>SFT</td>
<td>80.85%</td>
</tr>
<tr>
<td>Mistral-7B-Instruct</td>
<td>MD+SYN</td>
<td>5.78</td>
<td>2980</td>
<td>SFT+GRPO</td>
<td>77.87%</td>
</tr>
<tr>
<td>LLaMA 3.2 3B</td>
<td>MD</td>
<td>4.90</td>
<td>1879</td>
<td>SFT</td>
<td>75.74%</td>
</tr>
<tr>
<td>LLaMA 3.2 3B</td>
<td>SYN</td>
<td>7.28</td>
<td>1101</td>
<td>SFT</td>
<td>71.97%</td>
</tr>
<tr>
<td>LLaMA 3.2 3B</td>
<td>MD+SYN</td>
<td>5.78</td>
<td>2980</td>
<td>SFT</td>
<td>77.87%</td>
</tr>
<tr>
<td>LLaMA 3.2 3B</td>
<td>MD+SYN</td>
<td>5.78</td>
<td>2980</td>
<td>SFT+GRPO</td>
<td>43.83%</td>
</tr>
<tr>
<td>Qwen3-4B</td>
<td>MD+SYN</td>
<td>5.78</td>
<td>2980</td>
<td>SFT</td>
<td>80.00%</td>
</tr>
<tr>
<td>DeepSeek-R1</td>
<td>MD+SYN</td>
<td>5.78</td>
<td>2980</td>
<td>SFT</td>
<td>40.00%</td>
</tr>
</tbody>
</table>

Key takeaways:
1.  Training on the combined MD+SYN dataset consistently outperforms training on either dataset alone, demonstrating that the original template data and synthetic data provide complementary supervision: the original data captures reliable clinical diagnostic patterns, while the synthetic data improves linguistic diversity and conversational flow.
2.  Adding GRPO reinforcement learning reduces performance for both models, indicating that the supervised signal from the multi-turn dialogue corpus is already sufficiently strong, and RL introduces training instability without consistent benefits.

## 6.3. Per-Disease Performance
The per-disease diagnostic accuracy of MedAidLM is shown in Table 4 below:
The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th>Disease</th>
<th>Correct</th>
<th>Total</th>
<th>Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>Asthma</td>
<td>16</td>
<td>19</td>
<td>84.2%</td>
</tr>
<tr>
<td>Conjunctivitis</td>
<td>19</td>
<td>21</td>
<td>90.5%</td>
</tr>
<tr>
<td>Coronary heart disease</td>
<td>16</td>
<td>19</td>
<td>84.2%</td>
</tr>
<tr>
<td>Dermatitis</td>
<td>19</td>
<td>20</td>
<td>95.0%</td>
</tr>
<tr>
<td>Enteritis</td>
<td>22</td>
<td>24</td>
<td>91.7%</td>
</tr>
<tr>
<td>Esophagitis</td>
<td>22</td>
<td>27</td>
<td>81.5%</td>
</tr>
<tr>
<td>External otitis</td>
<td>15</td>
<td>17</td>
<td>88.2%</td>
</tr>
<tr>
<td>Mastitis</td>
<td>12</td>
<td>15</td>
<td>80.0%</td>
</tr>
<tr>
<td>Pneumonia</td>
<td>12</td>
<td>20</td>
<td>60.0%</td>
</tr>
<tr>
<td>Rhinitis</td>
<td>15</td>
<td>15</td>
<td>100.0%</td>
</tr>
<tr>
<td>Thyroiditis</td>
<td>19</td>
<td>19</td>
<td>100.0%</td>
</tr>
<tr>
<td>Traumatic brain injury</td>
<td>19</td>
<td>19</td>
<td>100.0%</td>
</tr>
</tbody>
</table>

Key takeaways:
1.  The model achieves perfect accuracy on diseases with highly distinctive symptom profiles: Rhinitis, Thyroiditis, and Traumatic brain injury.
2.  Performance is lowest for Pneumonia (60.0%), which shares overlapping respiratory symptoms with Asthma, leading to frequent misclassifications.

## 6.4. Expert Evaluation Results
The expert evaluation results for MedAidLM across 50 sampled dialogues are shown in Table 7 below:
The following are the results from Table 7 of the original paper:

<table>
<thead>
<tr>
<th>Metric</th>
<th>Expert 1</th>
<th>Expert 2</th>
<th>Expert 3</th>
<th>Average</th>
</tr>
</thead>
<tbody>
<tr>
<td>Medical Safety (Pass Rate)</td>
<td>96%</td>
<td>94%</td>
<td>96%</td>
<td>95.3%</td>
</tr>
<tr>
<td>Symptom Extraction</td>
<td>4.2</td>
<td>4.1</td>
<td>4.3</td>
<td>4.20</td>
</tr>
<tr>
<td>Context Memory</td>
<td>4.4</td>
<td>4.3</td>
<td>4.5</td>
<td>4.40</td>
</tr>
<tr>
<td>Diagnostic Correctness</td>
<td>4.1</td>
<td>4.0</td>
<td>4.2</td>
<td>4.10</td>
</tr>
<tr>
<td>Conversational Flow</td>
<td>4.3</td>
<td>4.2</td>
<td>4.4</td>
<td>4.30</td>
</tr>
<tr>
<td>Efficiency</td>
<td>4.0</td>
<td>3.9</td>
<td>4.1</td>
<td>4.00</td>
</tr>
</tbody>
</table>

Inter-annotator agreement scores are shown in Table 9 below:
The following are the results from Table 9 of the original paper:

<table>
<thead>
<tr>
<th>Metric</th>
<th>Krippendorff's α</th>
</tr>
</thead>
<tbody>
<tr>
<td>Symptom Extraction</td>
<td>0.82</td>
</tr>
<tr>
<td>Context Memory</td>
<td>0.84</td>
</tr>
<tr>
<td>Diagnostic Correctness</td>
<td>0.80</td>
</tr>
<tr>
<td>Conversational Flow</td>
<td>0.83</td>
</tr>
<tr>
<td>Efficiency</td>
<td>0.78</td>
</tr>
<tr>
<td>Average</td>
<td>0.81</td>
</tr>
</tbody>
</table>

Key takeaways:
1.  The model has a 95.3% medical safety pass rate, meaning unsafe advice is very rare.
2.  All Likert scale scores are above 4/5, indicating strong performance on symptom tracking, conversational quality, and diagnostic plausibility.
3.  The average inter-annotator agreement of 0.81 confirms that the expert evaluation results are reliable and consistent.

## 6.5. Error Analysis
The most frequent disease misclassifications made by MedAidLM are shown in Table 8 below:
The following are the results from Table 8 of the original paper:

<table>
<thead>
<tr>
<th>Original Disease</th>
<th>Misclassified As</th>
<th>Frequency</th>
</tr>
</thead>
<tbody>
<tr>
<td>Pneumonia</td>
<td>Asthma</td>
<td>3</td>
</tr>
<tr>
<td>Esophagitis</td>
<td>Enteritis</td>
<td>2</td>
</tr>
<tr>
<td>Esophagitis</td>
<td>Asthma</td>
<td>2</td>
</tr>
<tr>
<td>Asthma</td>
<td>Pneumonia</td>
<td>2</td>
</tr>
<tr>
<td>Coronary heart disease</td>
<td>Asthma</td>
<td>2</td>
</tr>
<tr>
<td>Pneumonia</td>
<td>Enteritis</td>
<td>2</td>
</tr>
<tr>
<td>External otitis</td>
<td>Conjunctivitis</td>
<td>2</td>
</tr>
<tr>
<td>Conjunctivitis</td>
<td>Mastitis</td>
<td>2</td>
</tr>
<tr>
<td>Mastitis</td>
<td>Traumatic brain injury</td>
<td>2</td>
</tr>
<tr>
<td>Esophagitis</td>
<td>Coronary heart disease</td>
<td>1</td>
</tr>
</tbody>
</table>

Most misclassifications occur between diseases with overlapping symptom profiles (e.g., respiratory diseases like Pneumonia and Asthma, gastrointestinal diseases like Esophagitis and Enteritis). These errors are clinically plausible, as these conditions can present with similar symptoms in short text-based consultations.

---

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This work introduces MedAidDialog, the first 7-language parallel multi-turn medical dialogue dataset, and MedAidLM, a lightweight deployable conversational medical model designed for low-resource healthcare settings. The key findings are:
1.  Augmenting template-based medical dialogue datasets with LLM-generated synthetic dialogues significantly improves model performance, combining the clinical reliability of template data with the linguistic diversity of synthetic data.
2.  Compact 3B-parameter SLMs fine-tuned with PEFT on quantized weights can achieve over 90% diagnostic accuracy, matching or outperforming larger 7B-parameter models, and can be deployed on consumer-grade hardware.
3.  The model has a 95.3% medical safety pass rate per expert evaluation, making it suitable for preliminary diagnostic guidance in settings with limited access to healthcare workers.
4.  A bidirectional translation layer enables multilingual support without the need to train separate dialogue models per language, reducing deployment overhead for linguistically diverse regions.

## 7.2. Limitations & Future Work
The authors explicitly note the following limitations:
1.  Synthetic data may introduce biases or simplified patterns that do not fully capture the complexity of real clinical interactions.
2.  The dataset is limited to only 12 diseases, restricting the system’s ability to generalize to a broader range of medical conditions.
3.  The underlying dialogue model is trained only in English, so the translation layer may introduce subtle errors or loss of clinical nuance in low-resource languages.
4.  The system currently only supports text-based interaction, with no support for multimodal inputs like medical images, lab reports, or voice.

    Proposed future work directions:
1.  Expand the dataset to cover more diseases and additional low-resource languages.
2.  Add multimodal capabilities (speech interfaces, vision-language models) to support voice interaction and analysis of medical reports/images.
3.  Incorporate disease-specific patient context profiles to improve diagnostic reasoning for complex conditions.
4.  Conduct larger-scale clinical evaluation with more diverse medical experts and real patient populations.

## 7.3. Personal Insights & Critique
This work has significant potential for real-world impact, particularly in low-income and middle-income countries with severe shortages of healthcare workers: the World Health Organization estimates a global shortage of 18 million healthcare workers by 2030, and a system like MedAidLM could provide reliable preliminary guidance to patients in underserved areas while reducing the burden on overstretched clinical staff.

That said, several important improvements could be made to the system:
1.  **Safety guardrails**: While the 95.3% safety rate is strong, even 4.7% unsafe advice is unacceptable for clinical use. Adding a dedicated emergency symptom detection module that immediately redirects patients to urgent care for red-flag symptoms (chest pain, stroke signs, severe abdominal pain, etc.) would reduce safety risks significantly.
2.  **Translation layer improvement**: The current translation layer uses general-purpose multilingual models. Fine-tuning the translation models on domain-specific medical translation data for the 7 target languages would reduce the risk of clinical nuance loss during translation.
3.  **Differential diagnosis output**: Instead of outputting a single final diagnosis, the system could output a ranked list of differential diagnoses with confidence scores, helping patients and clinical staff understand the uncertainty of the prediction and prioritize next steps.
4.  **Real data augmentation**: Where privacy regulations allow, augmenting the synthetic dataset with anonymized real clinical dialogues would reduce synthetic data biases and improve generalizability to real-world interactions.

    Overall, this work is a strong step toward accessible, equitable conversational healthcare AI for underserved populations, and the open release of the dataset and model code will enable further research in this critical area.