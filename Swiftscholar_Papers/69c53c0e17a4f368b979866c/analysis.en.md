# 1. Bibliographic Information
## 1.1. Title
The paper's title is *From Oracle to Noisy Context: Mitigating Contextual Exposure Bias in Speech-LLMs*. Its central topic is addressing the train-test mismatch (contextual exposure bias) in context-aware automatic speech recognition (ASR) systems built on Speech Large Language Models (Speech-LLMs), where models are trained on perfect ground-truth conversation history but deployed on error-prone predicted history.
## 1.2. Authors
The authors and their affiliations are:
1.  Xiaoyong Guo, Nanjie Li, Kai Wang, Hao Huang (corresponding author): School of Computer Science and Technology, Xinjiang University, Urumqi, China; Hao Huang is also affiliated with the Joint International Research Laboratory of Silk Road Multilingual Cognitive Computing, Urumqi, China
2.  Zijie Zeng, Haihua Xu, Wei Shi: Timekettle (a speech translation technology company)
    The corresponding author contact is hwanghao@gmail.com.
## 1.3. Publication Venue
As of the current date, the paper is published as a preprint on arXiv, a widely used open-access repository for pre-publication research in computer science, physics, and related fields. arXiv preprints are widely cited in the speech processing and NLP communities, though they have not undergone formal peer review at the time of publication.
## 1.4. Publication Year
The paper was published on 2026-03-25 (UTC).
## 1.5. Abstract
The paper first defines *contextual exposure bias*: a train-test mismatch in context-aware Speech-LLM ASR, where models are trained on perfect oracle conversation history but must use error-prone predicted history at inference. To solve this problem, the authors propose a unified 3-component training framework: (1) Teacher Error Knowledge, which uses Whisper large-v3 decoded hypotheses as training-time context to simulate real-world inference noise; (2) Context Dropout, which randomly masks historical context to reduce over-reliance on text history; (3) Direct Preference Optimization (DPO) on curated failure cases to explicitly train the model to reject erroneous context. Experiments on in-domain TED-LIUM 3 and zero-shot out-of-domain LibriSpeech show consistent gains: with a 2-utterance history window, supervised fine-tuning (SFT) with Whisper hypotheses reduces word error rate (WER) from 5.59% (oracle-trained baseline) to 5.47%, and DPO further improves it to 5.17%. Under irrelevant context attacks, DPO yields the smallest performance degradation (5.17% → 5.63%), indicating strong robustness to misleading context.
## 1.6. Original Source Links
- Preprint abstract page: https://arxiv.org/abs/2603.24034v1
- Full PDF: https://arxiv.org/pdf/2603.24034v1
- Open-source code and models: https://github.com/XYGuo1996/Contextual_Speech_LLMs

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Context-aware ASR systems use prior conversation history to resolve ambiguous acoustic inputs (e.g., homophones, rare terms). However, standard training pipelines use perfect *oracle* (ground-truth) history, while real-world deployment can only access error-prone predicted history from earlier ASR outputs. This distribution mismatch in the context channel is termed *contextual exposure bias*, which leads to error propagation: small errors in earlier history are amplified by the model's over-reliance on context, leading to sharply degraded performance.
### Importance & Research Gaps
Contextual exposure bias is a critical barrier to deploying robust conversational ASR systems. Prior work on handling imperfect context relied on implicit regularization (e.g., context dropout in speech translation) or auxiliary modules, but lacked a unified framework that explicitly aligns training context to inference conditions, and no prior work used preference optimization to train models to actively reject erroneous contextual cues.
### Innovative Entry Point
The authors explicitly design their training pipeline to replicate the noisy context conditions seen at inference, and add a preference alignment stage to teach the model to prioritize acoustic evidence over misleading context, rather than just regularizing context usage.
## 2.2. Main Contributions / Findings
The paper's key contributions are:
1.  **Problem formalization**: It explicitly defines contextual exposure bias as a core failure mode for context-conditioned Speech-LLM ASR, and uses this mismatch as a guiding principle for training pipeline design.
2.  **Unified training framework**: It proposes a 3-component noise-aware training pipeline that aligns training and inference context distributions:
    - Teacher Error Knowledge: uses a strong pre-trained ASR model (Whisper large-v3) to generate realistic error-prone context for training
    - Context Dropout: randomly masks context to reduce over-reliance on text history
    - DPO on hard negatives: uses pairwise preference data from model failure cases to explicitly suppress error propagation
3.  **Strong empirical performance**: The proposed framework consistently outperforms oracle-trained baselines on both in-domain (TED-LIUM 3) and zero-shot out-of-domain (LibriSpeech) test sets, and shows far greater robustness to irrelevant context attacks than baseline methods.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All core technical terms are defined below for beginner readers:
1.  **Automatic Speech Recognition (ASR)**: A task that converts raw speech audio signals into written text transcripts.
2.  **Speech Large Language Model (Speech-LLM)**: A multimodal large language model that takes speech audio as input (instead of only text) and outputs text, typically composed of three parts: (a) a frozen pre-trained speech encoder that converts audio to hidden embeddings, (b) a trainable multi-layer perceptron (MLP) projector that aligns speech embeddings to the LLM's input embedding space, (c) a frozen pre-trained text LLM that generates text transcripts.
3.  **Contextual ASR**: A subfield of ASR that uses supplementary contextual information (e.g., prior conversation history, document metadata, named entity lists) to improve recognition accuracy for ambiguous acoustic inputs.
4.  **Contextual Exposure Bias**: A train-test distribution mismatch specific to contextual ASR: models are trained on perfect ground-truth (oracle) historical context, but must use error-prone model-predicted historical context at inference.
5.  **Supervised Fine-Tuning (SFT)**: A fine-tuning stage where a pre-trained model is trained on labeled input-output pairs to adapt to a specific task (in this case, ASR).
6.  **Low-Rank Adaptation (LoRA)**: A lightweight fine-tuning technique for large models that freezes the full pre-trained model backbone, and only trains small low-rank matrices inserted into the model's attention layers. This reduces the number of trainable parameters by up to 1000x compared to full fine-tuning, while matching full fine-tuning performance for most tasks. The LoRA weight update is calculated as $\Delta W = BA$, where $A$ and $B$ are low-rank matrices with rank $r \ll d$ (the hidden dimension of the model layer).
7.  **Direct Preference Optimization (DPO)**: A lightweight model alignment technique that trains models to prefer high-quality outputs over low-quality outputs using pairwise preference data, without requiring a separate reward model or policy gradient optimization.
8.  **Word Error Rate (WER)**: The standard evaluation metric for ASR, which measures the percentage of words in the predicted transcript that are incorrect relative to the ground truth. Lower WER indicates better performance. The standardized formula for WER is:
    \$
    WER = \frac{S + D + I}{N}
    \$
    where:
    - $S$ = number of substitutions (model outputs the wrong word, e.g., "cat" instead of ground truth "bat")
    - $D$ = number of deletions (model omits a word present in ground truth)
    - $I$ = number of insertions (model adds a word not present in ground truth)
    - $N$ = total number of words in the ground truth transcript
      Note that WER can exceed 100% if there are large numbers of insertions.
## 3.2. Previous Works
### Contextual ASR Paradigms
Prior to Speech-LLMs, contextual ASR followed two core paradigms:
1.  **Shallow Fusion**: An inference-only technique that scores ASR hypotheses generated by an acoustic model using an external pre-trained language model (LM) that encodes contextual information, then re-ranks hypotheses to select the most contextually consistent output.
2.  **Deep Fusion**: A training-time technique that encodes contextual information into vector embeddings, which are directly injected into the internal layers of an end-to-end ASR model, so the model learns joint representations of acoustic and contextual signals during training.
### Context in Speech-LLMs
Recent work on Speech-LLMs has explored injecting contextual cues into model prompts, including metadata, entity lists, and conversation history. Notably, Lakomkin et al. (2024) tested Speech-LLM robustness to contextual perturbations, but did not address the core train-test context mismatch problem, or use preference alignment to improve robustness. Retrieval-Augmented Generation (RAG) has also been used to inject contextual information into Speech-LLMs, but requires an external retrieval module at inference.
### Handling Noisy Context
Prior work on imperfect context used implicit regularization: for example, context dropout in speech translation (Hussein et al., 2024) randomly masks context during training to reduce over-reliance, and noise representation learning (Lee et al., 2024) trains models to recognize noisy context. However, these methods do not explicitly teach the model to prioritize acoustic evidence over erroneous context, leading to occasional catastrophic failure cases.
## 3.3. Technological Evolution
The field of ASR has evolved in three key stages:
1.  Traditional ASR: Uses hybrid architectures with separate acoustic models, pronunciation models, and language models (e.g., CTC, AED, RNN-T)
2.  Pre-trained speech models: Large self-supervised pre-trained speech encoders (e.g., HuBERT, Whisper) that achieve strong general ASR performance with limited fine-tuning
3.  Speech-LLMs: Multimodal models that combine pre-trained speech encoders with large text LLMs, enabling strong contextual reasoning and cross-task generalization
    This paper's work falls into the third stage, addressing a critical deployment limitation of context-aware Speech-LLMs.
## 3.4. Differentiation Analysis
Compared to prior work, this paper's approach has three core innovations:
1.  It explicitly aligns training context distribution to inference context distribution using realistic teacher-generated errors, rather than using synthetic noise or implicit regularization
2.  It combines noisy context training with context dropout to provide a stable regularization strategy that works across all context window sizes, unlike oracle-trained models which require context-length-specific hyperparameter tuning
3.  It uses DPO on hard negatives to explicitly align the model to prefer correct transcripts even when context is erroneous, which is far more effective than additional SFT passes at suppressing error propagation

# 4. Methodology
## 4.1. Principles
The core intuition behind the proposed framework is that contextual exposure bias arises because the training context distribution does not match the inference context distribution. The framework is designed to:
1.  Replicate inference-time noisy context conditions during training to close the distribution gap
2.  Regularize the model to avoid over-relying on context, so it falls back to acoustic evidence when context is unreliable
3.  Explicitly train the model to prioritize correct outputs over context-induced errors via preference alignment
## 4.2. Core Methodology In-depth
### 4.2.1. Utterance-level Contextual ASR Setting
The paper focuses on sequential utterance-by-utterance conversational ASR, where each utterance is decoded using transcripts of previous utterances as context.
First, an input audio stream is segmented into a sequence of utterances $\{X_t\}_{t=1}^T$, where $X_t$ is the raw audio signal of the $t$-th utterance, and $Y_t$ is its ground-truth text transcript. For each utterance $t$, the ASR model generates a predicted transcript $\hat{Y}_t$ conditioned on the current utterance audio $X_t$ and available historical context $C_t$:
\$
\hat{Y}_t \sim p_\theta(Y | X_t, C_t)
\$
where $\theta$ denotes the trainable parameters of the model.
The historical context $C_t$ is defined as the concatenation of the transcripts of the most recent $N$ utterances (where $N$ is the context window size, with $N=0$ corresponding to a no-context baseline):
\$
C_t = concat(S_{t-N}, \dots, S_{t-1})
\$
where $S_i$ is the transcript used as historical context for the $i$-th utterance.
Contextual exposure bias arises from the difference between training and inference values of $S_i$:
- At training: $S_i = Y_i$ (oracle ground-truth transcript, no errors)
- At inference: $S_i = \hat{Y}_i$ (model-predicted transcript, may contain recognition errors)
  This difference creates a distribution mismatch between training and inference context, leading to degraded performance at deployment.
### 4.2.2. Model Architecture
The model architecture follows a standard Speech-LLM design with separate LoRA adapters for SFT and DPO stages, as shown in Figure 1:

![Figure 1: Model architecture](images/1.jpg)
*该图像是示意图，展示了LLM模型架构及其模块之间的关系。图中包含两个LoRA模块：SFT LoRA和DPO LoRA，分别对应不同的处理路径，用于提高模型在上下文处理中的性能和鲁棒性。*

The architecture components are:
1.  **Frozen speech encoder**: The encoder from Whisper large-v3 (decoder discarded), which converts raw speech audio $X_t$ to speech embeddings
2.  **Trainable MLP projector**: Aligns the output speech embeddings from the Whisper encoder to the input embedding space of the LLM
3.  **Frozen LLM backbone**: Vicuna-7B v1.5, a 7-billion parameter open-source instruction-tuned LLM, which generates the final text transcript
4.  **Two separate LoRA adapters**:
    - SFT LoRA: Trained during the supervised fine-tuning stage to adapt the LLM to the ASR task
    - DPO LoRA: Trained separately during the preference alignment stage, with an inference-time scaling factor to control its strength
### 4.2.3. Training Pipeline
The model is trained in three sequential stages:
#### Stage 1: Context-free base model initialization
First, a context-free ASR base model is trained by freezing the Whisper encoder and Vicuna LLM, and only optimizing the MLP projector on standard ASR data without any historical context. This establishes a stable alignment between speech and text embeddings for later contextual training.
#### Stage 2: Supervised Fine-Tuning (SFT) with noisy context
This stage uses two complementary techniques to align training context to inference conditions:
1.  **Teacher Error Knowledge**: Instead of using oracle ground-truth history, the authors pre-decode the full training set using Whisper large-v3, and use the resulting Whisper hypotheses as the historical context $S_i$ for training. This exposes the model to realistic ASR errors during training, matching the noise distribution seen at inference.
    The SFT loss is the standard sequence negative log-likelihood loss conditioned on the noisy context:
    \$
    \mathcal{L}_{SFT} = -\sum_t \log p_\theta(Y_t | X_t, C_t, P)
    \$
    where $P$ is the fixed task prompt for ASR, and $\theta$ includes the MLP projector parameters and SFT LoRA parameters.
2.  **Context Dropout**: To further regularize over-reliance on context, the historical context $C_t$ is randomly masked to an empty string with probability $p_{drop}$ (set to 0.5 in all experiments), while the current utterance audio $X_t$ remains unchanged:
    \$
    \tilde{C}_t = \begin{cases}
    \emptyset, & \text{with } p_{drop}, \\
    C_t, & \text{otherwise}
    \end{cases}
    \$
    The SFT loss is then computed using the masked context $\tilde{C}_t$, which forces the model to retain strong acoustic modeling capabilities and avoid using context as a shortcut.
#### Stage 3: Preference Optimization with DPO
While SFT with noisy context improves general robustness, a small set of hard failure cases still remain, where the model propagates errors from context. To explicitly suppress these failures, the authors use DPO on curated hard negatives:
1.  **Hard negative curation**: The SFT model is used to decode the full training set, and instances where the SFT model achieves WER > 20% are selected as hard negatives.
2.  **Preference pair construction**: For each hard negative instance, a pairwise preference pair $(Y^+, Y^-)$ is constructed, where $Y^+$ is the ground-truth transcript (preferred output), and $Y^-$ is the erroneous output generated by the SFT model (dispreferred output).
3.  **DPO loss calculation**: The DPO loss optimizes the model to assign higher likelihood to $Y^+$ than $Y^-$ given the same input $\mathbf{X} = (X_t, C_t, P)$, using the frozen SFT model as a reference policy. The loss is computed as follows:
    First, calculate the log-likelihood gap between the preferred and dispreferred outputs under the current trainable policy $\pi_\theta$:
    \$
    \Delta_\theta = \log \pi_\theta(Y^+ | \mathbf{X}) - \log \pi_\theta(Y^- | \mathbf{X})
    \$
    Next, calculate the log-likelihood gap under the frozen reference SFT policy $\pi_r$:
    \$
    \Delta_r = \log \pi_r(Y^+ | \mathbf{X}) - \log \pi_r(Y^- | \mathbf{X})
    \$
    Then compute the scaled difference between the two gaps:
    \$
    m = \beta(\Delta_\theta - \Delta_r)
    \$
    where $\beta = 0.1$ is a fixed temperature parameter that controls the strength of preference sharpening. The final DPO loss is:
    \$
    \mathcal{L}_{DPO} = -\log \sigma(m)
    \$
    where $\sigma(\cdot)$ is the sigmoid function that maps $m$ to the range [0, 1].
### 4.2.4. Inference Setup
A separate LoRA adapter is used for DPO to decouple preference optimization from SFT adaptation. At inference, a scaling factor $\gamma$ is introduced to control the strength of the DPO LoRA adapter, to avoid reward over-optimization (where the model overfits to preference signals at the cost of general generation quality).
The effective weight matrix of the LLM during forward pass is:
\$
W = W_{LLM} + \frac{\alpha}{r}\Delta W_{SFT} + \gamma \frac{\alpha'}{r'}\Delta W_{DPO}
\$
where:
- $W_{LLM}$ = frozen base LLM weights
- $\Delta W_{SFT}$ = low-rank update from the SFT LoRA adapter, with rank $r$ and scaling factor $\alpha$
- $\Delta W_{DPO}$ = low-rank update from the DPO LoRA adapter, with rank $r'$ and scaling factor $\alpha'$
  In all experiments, the two LoRA adapters use the same rank and scaling: $r = r' = 8$, $\alpha = \alpha' = 32$. During DPO training, $\gamma = 1$ (full DPO LoRA strength). At inference, $\gamma = 0.25$ (DPO LoRA strength reduced by 4x) to balance robustness gains and general generation quality.

# 5. Experimental Setup
## 5.1. Datasets
Two datasets are used for evaluation:
1.  **In-domain: TED-LIUM 3**: A publicly available English ASR dataset consisting of 452 hours of transcribed TED Talks, segmented into utterance-level turns with sequential conversational context. The official train, development, and test splits are used for training, validation, and in-domain testing. This dataset is chosen because it contains natural sequential utterances from long-form speech, making it ideal for evaluating contextual ASR performance.
2.  **Zero-shot cross-domain: LibriSpeech**: A publicly available English ASR dataset consisting of 1000 hours of transcribed public domain audiobooks. No training data from LibriSpeech is used; the model is evaluated directly on the test-clean (5.4 hours of clear, read speech) and test-other (5.3 hours of noisy, accented speech) splits to test cross-domain generalization. This dataset is chosen because it is from a completely different domain (read audiobooks vs. conversational TED Talks) and provides a rigorous test of the model's robustness to distribution shifts.
## 5.2. Evaluation Metrics
The only primary evaluation metric used is **Word Error Rate (WER)**, which is fully defined in Section 3.1. Lower WER indicates better ASR performance.
## 5.3. Baselines
The proposed method is compared against four representative baselines:
1.  **No context baseline**: A model trained without any historical context, which provides a lower bound on the performance gains from contextual information.
2.  **Oracle-trained baseline**: A model trained on perfect ground-truth historical context, which is the standard approach for contextual ASR prior to this work.
3.  **SFT-only baseline**: The SFT model trained with Whisper-generated context and 0.5 context dropout, without the DPO refinement stage.
4.  **SFT2 baseline**: An additional SFT pass on the same hard negative cases used for DPO, which is used to compare the effectiveness of preference alignment against standard supervised fine-tuning for failure case suppression.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The following are the results from Table 1 of the original paper, which reports WER across different context window sizes, training context sources, dropout rates, and test sets:

<table>
<thead>
<tr>
<th rowspan="2">N</th>
<th rowspan="2">Con<sub>inf</sub>/Con<sub>train</sub></th>
<th colspan="4">0 Dropout WER (%) ↓</th>
<th colspan="4">0.5 Dropout WER (%) ↓</th>
</tr>
<tr>
<th>TED</th>
<th>Test-clean</th>
<th>Test-other</th>
<th>LS-Ave.</th>
<th>TED</th>
<th>Test-clean</th>
<th>Test-other</th>
<th>LS-Ave.</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>-/-</td>
<td>7.89</td>
<td>4.79</td>
<td>9.83</td>
<td>7.310</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td rowspan="5">1</td>
<td>GT / GT</td>
<td>5.6</td>
<td>4.49</td>
<td>10.36</td>
<td>7.425</td>
<td>7.89</td>
<td>4.31</td>
<td>9.68</td>
<td>6.995</td>
</tr>
<tr>
<td>hyp / GT</td>
<td>5.85</td>
<td>4.54</td>
<td>10.63</td>
<td>7.585</td>
<td>7.47</td>
<td>4.74</td>
<td>9.94</td>
<td>7.340</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>5.62</td>
<td>4.67</td>
<td>9.46</td>
<td>7.065</td>
<td>7.21</td>
<td>5.37</td>
<td>9.96</td>
<td>7.665</td>
</tr>
<tr>
<td>+ DPO</td>
<td>5.69</td>
<td>4.71</td>
<td>9.57</td>
<td>7.140</td>
<td>5.32</td>
<td>4.56</td>
<td>9.38</td>
<td>6.970</td>
</tr>
<tr>
<td>+ SFT2</td>
<td>5.76</td>
<td>4.67</td>
<td>9.49</td>
<td>7.080</td>
<td>7.26</td>
<td>5.14</td>
<td>9.30</td>
<td>7.220</td>
</tr>
<tr>
<td rowspan="5">2</td>
<td>GT / GT</td>
<td>6.73</td>
<td>4.10</td>
<td>8.36</td>
<td>6.230</td>
<td>5.66</td>
<td>4.10</td>
<td>8.37</td>
<td>6.235</td>
</tr>
<tr>
<td>hyp / GT</td>
<td>6.89</td>
<td>4.85</td>
<td>9.88</td>
<td>7.365</td>
<td>5.59</td>
<td>5.15</td>
<td>9.10</td>
<td>7.130</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>8.15</td>
<td>5.57</td>
<td>12.00</td>
<td>8.785</td>
<td>5.47</td>
<td>5.14</td>
<td>9.50</td>
<td>7.320</td>
</tr>
<tr>
<td>+ DPO</td>
<td>5.07</td>
<td>4.87</td>
<td>9.51</td>
<td>7.190</td>
<td>5.17</td>
<td>4.84</td>
<td>9.19</td>
<td>7.015</td>
</tr>
<tr>
<td>+ SFT2</td>
<td>6.90</td>
<td>4.55</td>
<td>11.17</td>
<td>7.860</td>
<td>6.10</td>
<td>5.43</td>
<td>9.66</td>
<td>7.545</td>
</tr>
<tr>
<td rowspan="5">3</td>
<td>GT / GT</td>
<td>7.35</td>
<td>4.24</td>
<td>8.29</td>
<td>6.265</td>
<td>10.42</td>
<td>4.89</td>
<td>10.36</td>
<td>7.625</td>
</tr>
<tr>
<td>hyp / GT</td>
<td>7.05</td>
<td>5.03</td>
<td>10.68</td>
<td>7.855</td>
<td>12.62</td>
<td>5.28</td>
<td>10.93</td>
<td>8.105</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>10.06</td>
<td>5.36</td>
<td>10.69</td>
<td>8.025</td>
<td>7.87</td>
<td>5.93</td>
<td>10.39</td>
<td>8.160</td>
</tr>
<tr>
<td>+ DPO</td>
<td>5.98</td>
<td>5.60</td>
<td>9.96</td>
<td>7.780</td>
<td>5.18</td>
<td>4.73</td>
<td>9.36</td>
<td>7.045</td>
</tr>
<tr>
<td>+ SFT2</td>
<td>9.30</td>
<td>5.22</td>
<td>10.49</td>
<td>7.855</td>
<td>8.01</td>
<td>6.11</td>
<td>10.20</td>
<td>8.155</td>
</tr>
<tr>
<td rowspan="5">4</td>
<td>GT / GT</td>
<td>8.54</td>
<td>4.26</td>
<td>9.01</td>
<td>6.635</td>
<td>9.22</td>
<td>4.75</td>
<td>10.23</td>
<td>7.490</td>
</tr>
<tr>
<td>hyp / GT</td>
<td>7.74</td>
<td>4.87</td>
<td>11.07</td>
<td>7.970</td>
<td>10.87</td>
<td>4.75</td>
<td>10.23</td>
<td>7.490</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>87.37</td>
<td>4.66</td>
<td>10.81</td>
<td>7.735</td>
<td>7.81</td>
<td>4.75</td>
<td>9.82</td>
<td>7.285</td>
</tr>
<tr>
<td>+ DPO</td>
<td>4.93</td>
<td>4.79</td>
<td>9.97</td>
<td>7.380</td>
<td>5.69</td>
<td>4.79</td>
<td>9.25</td>
<td>7.020</td>
</tr>
<tr>
<td>+ SFT2</td>
<td>113.95</td>
<td>4.90</td>
<td>11.34</td>
<td>8.120</td>
<td>9.16</td>
<td>4.75</td>
<td>9.83</td>
<td>7.290</td>
</tr>
<tr>
<td rowspan="5">5</td>
<td>GT / GT</td>
<td>8.72</td>
<td>5.46</td>
<td>9.49</td>
<td>7.475</td>
<td>9.57</td>
<td>4.90</td>
<td>10.04</td>
<td>7.470</td>
</tr>
<tr>
<td>hyp / GT</td>
<td>10.34</td>
<td>5.08</td>
<td>10.76</td>
<td>7.920</td>
<td>8.19</td>
<td>5.36</td>
<td>11.29</td>
<td>8.325</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>135.57</td>
<td>4.59</td>
<td>10.87</td>
<td>7.730</td>
<td>8.5</td>
<td>4.95</td>
<td>9.33</td>
<td>7.140</td>
</tr>
<tr>
<td>+ DPO</td>
<td>5.34</td>
<td>4.67</td>
<td>9.85</td>
<td>7.260</td>
<td>4.96</td>
<td>4.55</td>
<td>9.24</td>
<td>6.895</td>
</tr>
<tr>
<td>+ SFT2</td>
<td>72.55</td>
<td>4.90</td>
<td>10.57</td>
<td>7.735</td>
<td>8.51</td>
<td>5.23</td>
<td>9.33</td>
<td>7.280</td>
</tr>
</tbody>
</table>

Key takeaways from Table 1:
1.  **Context Dropout is critical for stability**: Without context dropout, training with Whisper-generated context leads to catastrophic performance degradation for window sizes $N \geq 2$ (e.g., N=2 0 dropout WER 8.15% vs 7.89% no context baseline), as the model overfits to noisy context. With 0.5 dropout, training with Whisper context outperforms the oracle-trained baseline (N=2 0.5 dropout WER 5.47% vs 5.59% for oracle-trained).
2.  **DPO consistently outperforms SFT2**: Additional SFT on hard negatives (SFT2) often degrades performance (e.g., N=2 0.5 dropout WER increases from 5.47% to 6.10%), while DPO improves performance across 9 out of 10 configurations, with particularly large gains for longer context windows where error propagation is more severe (e.g., N=3 0.5 dropout WER drops from 7.87% to 5.18%).
3.  **Cross-domain generalization**: DPO consistently improves zero-shot performance on LibriSpeech, with the best cross-domain result of 6.895% LS-Ave achieved at N=5 0.5 dropout + DPO, outperforming both the no-context baseline (7.31%) and SFT-only baseline (7.14%).

    The following are the results from Table 2 of the original paper, which evaluates the impact of the DPO LoRA scaling factor $\gamma$ on performance and robustness to irrelevant context attacks:

    <table>
    <thead>
    <tr>
    <th rowspan="2">γ</th>
    <th colspan="3">TED-LIUM 3 (WER %) ↓</th>
    <th colspan="3">LibriSpeech (WER %) ↓</th>
    </tr>
    <tr>
    <th>Attacks/o (normal context)</th>
    <th>Attacks/w (irrelevant context)</th>
    <th>Gap (degradation)</th>
    <th>Test-clean</th>
    <th>Test-other</th>
    <th>Ave.</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>0</td>
    <td>5.47</td>
    <td>7.93</td>
    <td>2.46</td>
    <td>5.14</td>
    <td>9.50</td>
    <td>7.320</td>
    </tr>
    <tr>
    <td>0.0625</td>
    <td>5.37</td>
    <td>7.13</td>
    <td>1.76</td>
    <td>5.12</td>
    <td>9.31</td>
    <td>7.215</td>
    </tr>
    <tr>
    <td>0.125</td>
    <td>5.11</td>
    <td>5.76</td>
    <td>0.65</td>
    <td>5.02</td>
    <td>9.53</td>
    <td>7.275</td>
    </tr>
    <tr>
    <td>0.1875</td>
    <td>5.06</td>
    <td>5.69</td>
    <td>0.63</td>
    <td>4.70</td>
    <td>9.08</td>
    <td>6.890</td>
    </tr>
    <tr>
    <td>0.25</td>
    <td>5.17</td>
    <td>5.63</td>
    <td>0.46</td>
    <td>4.84</td>
    <td>9.19</td>
    <td>7.015</td>
    </tr>
    <tr>
    <td>0.375</td>
    <td>5.55</td>
    <td>5.73</td>
    <td>0.18</td>
    <td>4.85</td>
    <td>9.63</td>
    <td>7.240</td>
    </tr>
    <tr>
    <td>0.5</td>
    <td>8.39</td>
    <td>8.67</td>
    <td>0.28</td>
    <td>6.44</td>
    <td>12.14</td>
    <td>9.290</td>
    </tr>
    <tr>
    <td>0.625</td>
    <td>53.26</td>
    <td>57.15</td>
    <td>3.89</td>
    <td>27.11</td>
    <td>28.96</td>
    <td>28.035</td>
    </tr>
    </tbody>
    </table>

Key takeaways from Table 2:
- There is a clear tradeoff between clean accuracy and robustness: $\gamma=0.1875$ gives the best clean WER (5.06%), while $\gamma=0.25$ gives the smallest performance degradation under irrelevant context attacks (gap = 0.46%).
- Excessively high $\gamma$ (> 0.375) leads to reward over-optimization, where the model overfits to preference signals and general ASR performance degrades sharply. The authors select $\gamma=0.25$ as the optimal value to balance robustness and clean accuracy.
## 6.2. Ablation Studies
### 6.2.1. Robustness to Irrelevant Context
The following are the results from Table 3 of the original paper, which evaluates robustness to irrelevant context attacks (where historical context is replaced with random unrelated text):

<table>
<thead>
<tr>
<th rowspan="2">N</th>
<th rowspan="2">Con<sub>inf</sub>/Con<sub>train</sub></th>
<th colspan="3">0 Dropout WER (%)↓</th>
<th colspan="3">0.5 Dropout WER (%)↓</th>
</tr>
<tr>
<th>Attacks/o</th>
<th>Attacks/w</th>
<th>Gap</th>
<th>Attacks/o</th>
<th>Attacks/w</th>
<th>Gap</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">1</td>
<td>hyp / GT</td>
<td>5.85</td>
<td>8.32</td>
<td>2.47</td>
<td>7.47</td>
<td>8.82</td>
<td>1.35</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>5.62</td>
<td>6.27</td>
<td>0.65</td>
<td>7.21</td>
<td>7.09</td>
<td>-0.12</td>
</tr>
<tr>
<td>+ DPO</td>
<td>5.69</td>
<td>5.5</td>
<td>-0.19</td>
<td>5.32</td>
<td>5.23</td>
<td>-0.09</td>
</tr>
<tr>
<td rowspan="3">2</td>
<td>hyp / GT</td>
<td>6.89</td>
<td>8.43</td>
<td>1.54</td>
<td>5.59</td>
<td>9.23</td>
<td>3.46</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>8.15</td>
<td>10.37</td>
<td>2.22</td>
<td>5.47</td>
<td>7.93</td>
<td>2.46</td>
</tr>
<tr>
<td>+ DPO</td>
<td>5.07</td>
<td>6.59</td>
<td>1.52</td>
<td>5.17</td>
<td>5.63</td>
<td>0.46</td>
</tr>
<tr>
<td rowspan="3">3</td>
<td>hyp / GT</td>
<td>7.05</td>
<td>7.64</td>
<td>0.59</td>
<td>12.62</td>
<td>10.19</td>
<td>-2.43</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>10.06</td>
<td>11.18</td>
<td>1.12</td>
<td>7.87</td>
<td>9.53</td>
<td>1.66</td>
</tr>
<tr>
<td>+ DPO</td>
<td>5.98</td>
<td>6.24</td>
<td>0.26</td>
<td>5.18</td>
<td>5.31</td>
<td>0.13</td>
</tr>
<tr>
<td rowspan="3">4</td>
<td>hyp / GT</td>
<td>7.74</td>
<td>9.75</td>
<td>2.01</td>
<td>10.87</td>
<td>13.15</td>
<td>2.28</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>87.37</td>
<td>8.8</td>
<td>-78.57</td>
<td>7.81</td>
<td>8.82</td>
<td>1.01</td>
</tr>
<tr>
<td>+ DPO</td>
<td>4.93</td>
<td>5.44</td>
<td>0.51</td>
<td>5.69</td>
<td>7.58</td>
<td>1.89</td>
</tr>
<tr>
<td rowspan="3">5</td>
<td>hyp / GT</td>
<td>10.34</td>
<td>11.83</td>
<td>1.49</td>
<td>8.19</td>
<td>10.41</td>
<td>2.22</td>
</tr>
<tr>
<td>hyp / Whisper</td>
<td>135.57</td>
<td>10.74</td>
<td>-124.83</td>
<td>8.5</td>
<td>11.34</td>
<td>2.84</td>
</tr>
<tr>
<td>+ DPO</td>
<td>5.34</td>
<td>6.20</td>
<td>0.86</td>
<td>4.96</td>
<td>5.51</td>
<td>0.55</td>
</tr>
</tbody>
</table>

Key takeaway: The DPO-refined model consistently achieves the lowest WER under irrelevant context attacks across all window sizes and dropout settings, with negligible performance degradation (gap < 1% for most settings). This confirms that the proposed framework teaches the model to ignore misleading context and rely on acoustic evidence.
### 6.2.2. Hard Negative Threshold Ablation
The authors conduct an ablation study on the WER threshold used to select hard negatives for DPO training (Appendix Table A). Results show consistent performance gains across all thresholds from 5% to 30%, with optimal performance at the 20% threshold used in the main experiments. The optimal $\gamma$ value of 0.25 remains stable across all thresholds, indicating that the inference strategy is decoupled from data curation, simplifying deployment.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper formalizes contextual exposure bias as a core deployment limitation of context-aware Speech-LLM ASR systems, and proposes a unified 3-component training framework to mitigate this bias: Teacher Error Knowledge to align training and inference context distributions, Context Dropout to reduce over-reliance on context, and DPO on hard negatives to explicitly suppress error propagation. Extensive experiments show that the proposed framework consistently outperforms standard oracle-trained baselines on both in-domain and zero-shot cross-domain test sets, and achieves far greater robustness to irrelevant context attacks. The framework establishes a practical, deployment-ready paradigm for long-form conversational ASR under realistic imperfect conditions.
## 7.2. Limitations & Future Work
The authors explicitly note two key limitations of their work:
1.  **No support for overlapping speech**: The experimental setup assumes sequential, non-overlapping turn-based speech (as in TED-LIUM 3 and LibriSpeech). The model has not been tested on multi-speaker overlapping speech ("cocktail party" scenarios), which is common in real-world conversational settings.
2.  **Limited diversity of teacher errors**: The current implementation uses only Whisper large-v3 as the source of training context errors, so the simulated error distribution is biased to Whisper's specific failure modes, and may not cover all possible error types encountered in real-world deployment.
    The authors suggest future work directions: modifying the architecture to handle overlapping speech, using multiple diverse teacher ASR models to generate a wider range of training context errors, and evaluating the framework on more languages and domains.
## 7.3. Personal Insights & Critique
This work addresses a highly practical, under-studied problem in Speech-LLM deployment, and the proposed framework is simple, effective, and requires no auxiliary modules at inference, making it easy to integrate into production systems. The use of separate LoRA adapters for SFT and DPO with an inference-time scaling factor is a clever solution to the common problem of reward over-optimization in preference alignment, and this technique could be transferred to other LLM alignment tasks beyond ASR.
Potential areas for improvement:
- Dynamic context dropout probability, which adjusts based on context quality or window size, could further improve stability across different settings
- Using multi-turn error propagation to curate hard negatives (where errors from earlier turns lead to failures in later turns) could further improve robustness to long-context error accumulation
- Testing the framework on low-resource languages, where context is even more critical for resolving ambiguous acoustic inputs
  One unverified assumption in the work is that Whisper's error distribution matches the error distribution of the target Speech-LLM at inference; using the model's own predicted history from earlier training iterations as training context could further align the training and inference error distributions, potentially leading to even better performance.