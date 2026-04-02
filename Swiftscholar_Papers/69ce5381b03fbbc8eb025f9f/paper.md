# KUET at StanceNakba Shared Task: StanceMoE: Mixture-of-Experts Architecture for Stance Detection

Abdullah Al Shafi, Md. Milon Islam, Sk. Imran Hossain, K. M. Azharul Hasan Department of Computer Science and Engineering, Khulna University of Engineering & Technology abdullah@iict.kuet.ac.bd, {milonislam, imran, az}@cse.kuet.ac.bd

# Abstract

Acor-eve aneeioneiuhepr posttwaseceolital mentpliatethonsor-basmodelshavaiveelatively o p csal e moatheeaptivhitctushliclmoeldivanresi pattI S BERT encoder ractor-level sance detection.Ourmodel integrates si expert modules designed to ptue c atri ilwublaivuashap areconducted on the StanceNakba 2026 Subtask A dataset, comprising 1,401 annotated English texts where the target actor is implicit in the text. StanceMoE achieves a macro-F1 score of $9 4 . 2 6 \%$ ,outperforming traditional baselines, and alternative BERT-based variants. eywords: stance detection, mixture-of-experts, context-aware gating, adaptive weightin

# 1. Introduction

Stance detection refers to the automatic identification of an author's position towards a specific topic, entity, or proposition expressed within a text (Garg and Caragea, 2024). The task aims to identify whether the presented viewpoint is supportive, opposing, or neutral toward the specific target. The study of stance detection differs from common sentiment analysis because it needs to measure emotional expressions that depend on particular targets, whereas the same text can show different stances based on which target the reader chooses to consider (Niu et al., 2024). The nature of this task requires target awareness because it generates both semantic complexity and computational difficulties, particularly when targets remain unmentioned and stances are expressed in an indirect way (Küçük and Can, 2020). To address these challenges, we introduce StanceMoE1, a context-enhanced Mixtureof-Experts (MoE) architecture designed for actor-level stance detection. Unlike conventional transformer-based approaches that rely on single aggregated representations, our method explicitly decomposes stance modeling into complementary expert modules that capture diverse linguistic and discourse-level phenomena. By incorporating a context-aware gating mechanism, the architecture enables adaptive and input-sensitive fusion of heterogeneous stance signals. Through comprehensive experimentation and detailed ablation analysis on StanceNakba 2026 Shared Task dataset (Aldous et al., 2026), we demonstrate that explicitly modeling such diverse patterns enables more robust and fine-grained actor-level stance detection with $3 ^ { r d }$ position in the competition.

# Related Works

The field of stance detection has evolved through its transition from rule-based systems (Küçük and Can, 2020) to classical supervised systems (Alturayeif et al., 2023), which rely on manually created features. Deep learning methods, such as Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTMs) with attention mechanisms, enable models to automatically extract features and better capture target-specific stance information (Gera and Neal, 2025). Current transformer architectures use BERT (Garg and Caragea, 2024) and Large Language Models (LLMs) (Pangtey et al., 2025) to deliver excellent results through their ability to understand context and their capacity for few-shot learning.

# 3. Proposed StanceMoE Architecture

We propose a context-enhanced Mixture-of-Experts architecture developed upon a fine-tuned BERT encoder for stance detection. The overall framework consists of three components: (i) a contextual encoder, (i) six parallel expert modules capturing complementary linguistic signals, and (ii) a context-aware gating and fusion mechanism. The architecture is illustrated in Fig. 1.

![](images/1.jpg)  

Figure 1: Proposed Mixture-of-Experts architecture for actor-level stance detection.

# 3.1. Contextual Encoder

Given an input sequence, we employ a fine-tuned BERT encoder to obtain contextualized token representations and a special [CLS] embedding.

# 3.2. Expert Modules

While BERT provides promising contextual understanding, stance detection often depends on subtle lexical cues, contrast markers, and discourse patterns. Standard pooling techniques over contextual embeddings may not consistently capture these heterogeneous patterns. To address this, we introduce six complementary experts, each targeting a distinct linguistic phenomenon. Mean Pooling Expert (Global Orientation): This expert captures the overall semantic direction of a post by averaging token representations. It is effective when stance is consistently expressed throughout the sentence. For example, "Israel has the right to defend itself." (Pro-Israel), or "Palestinians deserve statehood and equal rights." (Pro-Palestine). Max Pooling Expert (Salient Polarity Tokens): Some posts contain isolated but highly indicative tokens that determine stance. For instance, the token "occupation" serves as the main indicator of stance in the Pro-Palestine statement "This ongoing occupation must end". Similarly, the token "terrorist" serves as the main lexical indicator in the Pro-Israel statement "The terrorist attacks cannot be justified". Max pooling selects the most activated feature dimensions to emphasize important tokens throughout the entire sequence. Self-Attention Pooling Expert (Context-Dependent Focus): The specific clauses of a sentence work as the place where stance exists. The sentence "While the humanitarian situation is tragic, Israel must respond to security threats" demonstrate an example of such configurations. In such formulations, one clause presents the main stance while the additional information is presented through the rest of the sentence. The attention mechanism gives more weight to stance components because it assesses their importance throughout the evaluation process. Multi-Kernel CNN Expert (Phrase-Level Patterns): Geopolitical stance is mostly expressed through short slogan-like phrases which include "Stand with Israel" (Pro-Israel) and "Free Palestine" (Pro-Palestine). The compact n-grams of the expression serve as effective stance indicators which maintain their function despite minimal contextual information. Multi-kernel convolutions capture such localized phrase patterns. Lexical Cue-Aware Expert (Framing and Reporting Signals): Neutral posts use reporting language and distancing language to express content like the statement "According to officials, negotiations are ongoing." Verbs like "claims," "reports," "states," function as indicators that speakers use indirect methods to present information instead of showing direct support for their statements. This expert combines different ways of showing predetermined cue tokens to improve the ability to separate neutral content from opinionated postings. Contrast-Aware Expert (Discourse Shift Modeling): Discourse contrast markers such as "but" or "however" often indicate stance refinement or emphasis. For example, "I support peace efforts, but the blockade must end." or "Civilian harm is tragic; however, security responses are necessary." In such cases, the clause containing the contrast marker often carries stronger stance weight. We therefore amplify representations of contrast tokens to better capture discourse-level stance shifts.

# 3.3. Context-Aware Gating and Fusion

We introduce a gating mechanism that dynamically weights the experts instead of treating them equally. The final representation is computed as a weighted combination of expert outputs. This design enables adaptive routing. For instance, strongly opinionated posts may rely more on max pooling, contrast-heavy statements may increase the weight of the contrast-aware expert, reportstyle neutral texts may emphasize cue-aware representations. The fused representation is then passed to a linear classifier." We train the model using crossentropy loss with label smoothing. To improve robustness, we employ stratified $K$ -fold crossvalidation during training and perform weighted logit averaging of those folds during inference. The weighting is done by giving more weight to folds having a higher F1-Score during validation.

# 4 Experimental Setup

# 4.1. Dataset

The dataset used in this paper is provided by the organizers of the StanceNakba 2026 Shared Task (Aldous et al., 2026). We focus on Subtask A, an actor-level stance detection dataset comprising 1,401 English texts labeled as Pro-Palestine, Pro-Israel, or Neutral. The stance labels denote the author's alignment with actors, which are implicit in the text and must be inferred by the model. The official split follows a 70/15/15 split for training, development, and test sets. In our experiments, we apply stratified K-fold cross-validation on the combined training and development set $8 5 \%$ of the data). The official test set $( 1 5 \% )$ is kept strictly unseen during training.

# 4.2. Baseline Methods

We perform an extensive empirical assessment of existing stance detection techniques, which are classified into three major methodological categories: Machine Learning (ML) methods with TF-IDF features, supervised training with Deep Neural Networks (DNNs), and BERT. The ML methods include Logistic Regression (LR) (Rahman et al., 2024), Multinomial Naive Bayes (MNB) (Zannat et al., 2025), Support Vector Machine (SVM) (Rahman et al., 2024), and Random Forest (RF) (Shafi et al., 2025). We implement several popular DNN architectures as baselines, including Bi-directional Long Short-Term Memory (BiLSTM) (Rahman et al., 2024), Target specific Attentional Network (TAN) (Du et al., 2017), Gated Convolutional network with Aspect Embedding (GCAE) (Xue and Li, 2018), Cross network (CrossNet) (Du et al., 2017). Moreover, we evaluate two alternative variants developed on the BERT backbone: (i) a stacked architecture (Khan et al., 2025) where expert modules are applied sequentially, and (ii) a feature fusion model (Lee et al., 2021) where expert outputs are fused without adaptive weighting.

# 5. Experimental Results

# 5.1. Actor-level Stance Detection

Table 1 presents the comparative performance of all models for actor-level stance detection. Among traditional baselines, RF achieves the best results with an F1-Score of $8 4 . 1 9 \%$ , which shows that ensemble tree-based methods perform better than other methods. Among neural models, GCAE performs highest $( 8 7 . 6 2 \%$ F1), which shows that attention-based architectures perform well to model stance. The use of contextual embeddings results in better performance outcomes. BERT achieves an F1-Score of $8 9 . 8 6 \%$ ,while expert-based architectures like Stacked $( 9 1 . 8 3 \% )$ and Fusion $( 9 1 . 1 7 \% )$ show promising results, but their improvements remain within fixed expert integration boundaries. The proposed StanceMoE achieves the best performance with an F1-Score of $9 4 . 2 6 \%$ ,which surpasses all other cases of comparison. Additionally, our team has stood $3 ^ { r d }$ in the competition. The system demonstrates performance enhancement because of its adaptive expert weighting system, which uses a gating mechanism. The proposed system also shows stable performance with low cross-validation variance $( \approx \pm 1 . 1 \% )$ .

# 5.2. Ablation Study

Table 2 presents an ablation analysis to examine the contribution of each expert in the proposed StanceMoE architecture. Removing any expert generally degrades overall performance, confirming that the experts capture complementary stance signals. The largest drops occur when removing the self-attention or contrastive experts, indicating their importance in modeling contextual dependencies and discourse-level stance shifts. Excluding the mean or max pooling experts also reduces performance, suggesting the importance of salient lexical cues and global contextual signals for stance identification.

Table 1: Performance comparison of baseline models and the proposed StanceMoE on the held-out test set. "K-fold $( m e a n \pm s t d ) ^ { \prime }$ reports the mean and standard deviation of test performance across fold-specific models, while "weighted logit ensemble" is the final ensemble prediction obtained via logit averaging. Here, Acc $=$ Accuracy, Pre $=$ Precision, Rec $=$ Recall, and $\mathsf { F } 1 = \mathsf { F } 1$ -Score.   

<table><tr><td rowspan="2">Methods</td><td colspan="4">K-fold (mean±std)</td><td colspan="4">Weighted Logit Ensemble</td></tr><tr><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>LR</td><td>80.91±1.79</td><td>80.90±1.78</td><td>80.93±1.78</td><td>80.86±1.78</td><td>81.28</td><td>81.25</td><td>81.33</td><td>81.22</td></tr><tr><td>MNB</td><td>77.25±1.64</td><td>75.88±2.02</td><td>78.49±1.65</td><td>77.20±1.65</td><td>77.38</td><td>76.02</td><td>78.63</td><td>77.33</td></tr><tr><td>SVM</td><td>83.25±1.71</td><td>84.73±1.68</td><td>83.26±1.71</td><td>83.27±1.72</td><td>83.37</td><td>84.94</td><td>83.35</td><td>83.43</td></tr><tr><td>RF</td><td>84.05±1.83</td><td>84.73±1.73</td><td>84.05±1.82</td><td>84.05±1.90</td><td>84.16</td><td>84.83</td><td>84.14</td><td>84.19</td></tr><tr><td>BiLSTM</td><td>85.63±2.99</td><td>85.87±3.00</td><td>85.63±2.99</td><td>85.51±3.07</td><td>85.74</td><td>85.98</td><td>85.74</td><td>85.72</td></tr><tr><td>TAN</td><td>85.99±3.10</td><td>86.01±2.81</td><td>86.36±2.08</td><td>85.91±2.15</td><td>86.13</td><td>86.10</td><td>86.43</td><td>86.03</td></tr><tr><td>GCAE</td><td>87.69±2.85</td><td>87.98±2.92</td><td>87.48±2.16</td><td>87.49±2.78</td><td>87.82</td><td>88.10</td><td>87.53</td><td>87.62</td></tr><tr><td>CrossNet</td><td>85.13±2.28</td><td>85.55±1.89</td><td>84.90±1.90</td><td>84.77±2.25</td><td>85.29</td><td>85.65</td><td>84.97</td><td>84.88</td></tr><tr><td>BERT</td><td>89.77±2.35</td><td>90.04±2.30</td><td>89.77±2.31</td><td>89.61±2.29</td><td>90.05</td><td>90.28</td><td>90.03</td><td>89.86</td></tr><tr><td>Stacked</td><td>91.61±2.46</td><td>91.73±2.43</td><td>91.62±2.44</td><td>91.60±2.47</td><td>91.94</td><td>92.14</td><td>91.93</td><td>91.83</td></tr><tr><td>Fusion</td><td>91.03±2.26</td><td>91.20±2.29</td><td>91.03±2.24</td><td>91.02±2.26</td><td>91.18</td><td>91.45</td><td>91.17</td><td>91.17</td></tr><tr><td>StanceMoE</td><td>94.09±1.11</td><td>94.18±1.12</td><td>94.08±1.12</td><td>94.03±1.12</td><td>94.31</td><td>94.45</td><td>94.31</td><td>94.26</td></tr></table>

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Acc</td><td colspan="3">Pro-Palestine</td><td colspan="3">Pro-Israel</td><td colspan="3">Neutral</td></tr><tr><td>Pre</td><td>Rec</td><td>F1</td><td>Pre</td><td>Rec</td><td>F1</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>w/o Mean</td><td>92.89</td><td>89.47</td><td>95.75</td><td>92.52</td><td>94.52</td><td>98.57</td><td>96.50</td><td>95.16</td><td>84.29</td><td>89.39</td></tr><tr><td>w/o Max</td><td>93.36</td><td>91.89</td><td>95.77</td><td>93.79</td><td>92.00</td><td>98.57</td><td>95.17</td><td>96.77</td><td>85.71</td><td>90.91</td></tr><tr><td>w/o Self-Attention</td><td>91.94</td><td>89.33</td><td>94.37</td><td>91.78</td><td>93.24</td><td>98.57</td><td>95.83</td><td>93.55</td><td>82.86</td><td>87.88</td></tr><tr><td>w/o CNN</td><td>93.36</td><td>92.96</td><td>92.96</td><td>92.96</td><td>93.33</td><td>99.26</td><td>96.55</td><td>93.85</td><td>87.14</td><td>90.37</td></tr><tr><td>w/o Lexical-cue</td><td>93.36</td><td>90.54</td><td>94.37</td><td>92.41</td><td>95.83</td><td>98.57</td><td>97.18</td><td>95.08</td><td>82.86</td><td>88.55</td></tr><tr><td>w/o Contrastive</td><td>91.94</td><td>90.41</td><td>92.96</td><td>91.67</td><td>90.91</td><td>99.32</td><td>95.24</td><td>93.85</td><td>87.14</td><td>90.37</td></tr><tr><td>StanceMoE</td><td>94.31</td><td>94.37</td><td>94.37</td><td>94.37</td><td>92.11</td><td>100</td><td>95.89</td><td>96.88</td><td>88.57</td><td>92.54</td></tr></table>

Table : Classwise ablation study of StanceMo using weighted logit ensemble, showing the effect of removing individual expert modules. The lexical-cue expert particularly benefits the Pro-Palestine class, while the contrastive expert improves recognition of stance transitions and argumentative structures. Although some variants maintain competitive scores for individual classes, they generally exhibit reduced balance across classes, especially for the neutral category. Overall, the full StanceMoE model achieves the best and most balanced performance across all classes, demonstrating the effectiveness of adaptive expert integration.

# 6. Conclusion

In this paper, we introduced StanceMoE, a contextenhanced MoE architecture for stance detection. Motivated by the limitations of unified transformer representations capturing heterogeneous linguistic phenomena, our approach explicitly models complementary stance-indicative signals through expert modules and integrates them via a context-aware gating mechanism. This design enables adaptive modeling of lexical, semantic, and discourse-level patterns that characterize stance expression in sensitive contexts. Extensive experiments demonstrate that StanceMoE substantially outperforms traditional baseline models. Ablation analysis further confirms the complementary contributions of individual experts and the importance of adaptive expert fusion for robust detection. Overall findings highlight the effectiveness of explicitly disentangling and dynamically integrating diverse linguistic signals for stance detection.

# Bibliographical References

Kholoud Khalil Aldous, Md Rafiul Biswas, Mabrouka Bessghaier, Shimaa Ibrahim, Kais Attia, and Wajdi Zaghouani. 2026. StanceNakba shared task: Actor and topic-aware stance detection in public discourse. In Proceedings of the 15th International Conference on Language Resources and Evaluation (LREC'26), Palma, Spain. Izzat Alsmadi, lyad Alazzam, Mohammad Al-Ramahi, and Mohammad Zarour. 2024. Stance detection in the context of fake news—a new approach. Future Internet, 16(10):364. Nora Alturayeif, Hamzah Luqman, and Moataz Ahmed. 2023. A systematic review of machine learning techniques for stance detection and its applications. Neural Computing and Applications, 35(7):51135144. Michael Burnham. 2025. Stance detection: a practical guide to classifying political beliefs in text. Political Science Research and Methods, 13(3):611628. Jiachen Du, Ruifeng Xu, Yulan He, and Lin Gui. 2017. Stance classification with target-specific neural attention networks. In 2017 26th International Joint Conference on Artificial Intelligence (IJCAI), pages 39883994. Krishna Garg and Cornelia Caragea. 2024. Stanceformer: Target-aware transformer for stance detection. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 49694984. Parush Gera and Tempestt Neal. 2025. Deep learning in stance detection: A survey. ACM Computing Surveys, 58(1):137. Zhijiang Guo, Michael Schlichtkrull, and Andreas Vlachos. 2022. A survey on automated factchecking. Transactions of the Association for Computational Linguistics, 10:178206. Lal Khan, Atika Qazi, Hsien-Tsung Chang, Mousa Alhajlah, and Awais Mahmood. 2025. Empowering urdu sentiment analysis: an attention-based stacked cnn-bi-lstm dnn with multilingual bert. Complex & Intelligent Systems, 11(1):10. Dilek Küçük and Fazli Can. 2020. Stance detection: A survey. ACM Computing Surveys, 53(1):137. Sanghyun Lee, David K Han, and Hanseok Ko. 2021. Multimodal emotion recognition fusion analysis adapting bert with heterogeneous feature unification. IEEE Access, 9:94557-94572. Junxia Ma, Changjiang Wang, Lu Rong, Bo Wang, and Yaoli Xu. 2025. Exploring multi-agent debate for zero-shot stance detection: A novel approach. Applied Sciences, 15(9):4612. Fuqiang Niu, Min Yang, Ang Li, Baoquan Zhang, Xiaojiang Peng, and Bowen Zhang. 2024. A challenge dataset and effective models for conversational stance detection. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 122- 132. Lata Pangtey, Anukriti Bhatnagar, Shubhi Bansal, Shahid Shafi Dar, and Nagendra Kumar. 2025. Large language models meet stance detection: A survey of tasks, methods, applications, challenges and future directions. arXiv:2505.08464. Arifur Rahman, Shahriar Parvej, Kazi Saeed Alam, and HM Abdul Fattah. 2024. Optimizing sms spam detection: comparative analysis of hybrid voting ensembles and bi-Istm networks with stratified cross-validation. In 2024 5th International Conference on Data Intelligence and Cognitive Informatics (ICDICl), pages 1030-1035. IEEE. Abdullah Al Shafi, Rowzatul Zannat, Abdul Muntakim, and Mahmudul Hasan. 2025. A structured dataset of disease-symptom associations to improve diagnostic accuracy. arXiv:2506.13610. Wei Xue and Tao Li. 2018. Aspect based sentiment analysis with gated convolutional networks. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 25142523. Ruichao Yang, Jing Ma, Wei Gao, and Hongzhan Lin. 2025. LIm-enhanced multiple instance learning for joint rumor and stance detection with social context information. ACM Transactions on Intelligent Systems and Technology, 16(3):1 27. Rowzatul Zannat, Abdullah Al Shafi, and Abdul Muntakim. 2025. Bridging the gap in bangla healthcare: Machine learning based disease prediction using a symptoms-disease dataset. In 2025 International Conference on Electrical, Computer and Communication Engineering (ECCE), pages 16. IEEE. Bowen Zhang, Genan Dai, Fuqiang Niu, Nan Yin, Xiaomao Fan, Senzhang Wang, Xiaochun Cao, and Hu Huang. 2024. A survey of stance detection on social media: New directions and perspectives. arXiv:2409.15690. Zhengyuan Zhu, Zeyu Zhang, Haiqi Zhang, and Chengkai Li. 2025. Ratsd: Retrieval augmented truthfulness stance detection from social media posts toward factual claims. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 33663381.

# A. Need of Stance Detection

The ability to detect stance has become increasingly important due to the growth of online discourse. Applications span multiple domains, including electoral trend analysis (Burnham, 2025), examination of argumentative interactions in online debates (Ma et al., 2025), monitoring of social media discussions (Zhang et al., 2024), rumor assessment and verification (Yang et al., 2025), automated fact-checking systems (Guo et al., 2022), fake news identification (Alsmadi et al., 2024), and stance-aware information retrieval (Zhu et al., 2025). In these contexts, understanding directional opinion toward a claim or actor is more informative than simply measuring sentiment (Garg and Caragea, 2024).

# B. Operational Details of Expert Modules

Given an input sequence $X = \{ x _ { 1 } , \ldots , x _ { T } \}$ , pretrained BERT encoder obtains contextualized token representations: $H = \{ h _ { 1 } , h _ { 2 } , . . . , h _ { T } \}$ $h _ { i } \in$ $\mathbb { R } ^ { d }$ where $d$ denotes the hidden dimension and $T$ is the length of the tokenized input sequence. Mean Pooling Expert: This expert captures global semantic information as shown in (1).

$$
e _ { 1 } = W _ { 1 } \left( \frac { 1 } { T } \sum _ { i = 1 } ^ { T } h _ { i } \right)
$$

Max Pooling Expert: To extract salient tokenlevel features, we utilize max pooling expert as in (2). lutions with kernel sizes $k \in \{ 2 , 3 , 4 , 5 \}$ as demonstrated in (5).

$$
e _ { 4 } = W _ { 4 } \left( { \mathsf { C o n c a t } } ( \mathsf { M e a n P o o l } ( \mathsf { R e L U } ( \mathsf { C o n v } _ { k } ( H ) ) ) ) \right)
$$

Lexical Cue-Aware Expert: We define a set of stance-indicative lexical cues and generate a binary mask over their positions $C$ . The cue-aware representation is computed as (6).

$$
e _ { 5 } = W _ { 5 } \left( { \frac { \sum _ { i \in C } h _ { i } } { | C | + \epsilon } } \right)
$$

where, $\epsilon$ avoids division by zero. Contrast-Aware Expert: To model discourse contrast markers (e.g., "but", "however"), we enhance their contextual influence. Let, $D$ denote the set of contrast token positions. The overal process is shown in (8).

$$
\begin{array} { r } { \tilde { h } _ { i } = \left\{ \begin{array} { l l } { 3 h _ { i } } & { \mathsf { i f } ~ i \in D , } \\ { h _ { i } } & { \mathsf { o t h e r w i s e } , } \end{array} \right. } \end{array}
$$

Self-Attention Pooling Expert: We introduce a trainable attention vector $v$ to compute token importance as mentioned in (3).

$$
e _ { 6 } = W _ { 6 } \left( \frac { \sum _ { i = 1 } ^ { T } \widetilde { h } _ { i } } { | D | + \epsilon } \right)
$$

$$
e _ { 2 } = W _ { 2 } \left( \underset { i = 1 } { \overset { T } { \operatorname* { m a x } } } h _ { i } \right)
$$

$$
\alpha _ { i } = \frac { \mathsf { e x p } ( \mathsf { t a n h } ( h _ { i } ^ { \top } v ) ) } { \sum _ { j = 1 } ^ { T } \mathsf { e x p } ( \mathsf { t a n h } ( h _ { j } ^ { \top } v ) ) }
$$

# C. Operational Details of Context-Aware Gating

$$
e _ { 3 } = W _ { 3 } \left( \sum _ { i = 1 } ^ { T } \alpha _ { i } h _ { i } \right)
$$

Multi-Kernel CNN Expert: To model local ngram patterns, we apply one-dimensional convo- The gating network takes the [CLS]' representation $h _ { \mathsf { C l s } } \in \mathbb { R } ^ { d }$ from the BERT encoder as input. A learnable linear layer $W _ { g } ~ \in ~ \mathbb { R } ^ { N \times d }$ with bias $b _ { g } \in \mathbb { R } ^ { N }$ computes logits for $N = 6$ experts, which are normalized via a softmax to generate the gating weights as in (9).

$$
g = { \mathsf { S o f t m a x } } ( W _ { g } h _ { \mathsf { c l s } } + b _ { g } )
$$

where $g \in \mathbb { R } ^ { K }$ $g _ { i } \geq 0$ for alll $i$ and $\textstyle \sum _ { i = 1 } ^ { K } g _ { i } = 1$ Each gating weight $g _ { i }$ determines the contribution of the corresponding expert output $e _ { i }$ to the final representation, which is obtained via weighted aggregation using (10).

$$
h _ { \mathsf { m o e } } = \sum _ { i = 1 } ^ { K } g _ { i } e _ { i }
$$

The fused representation $h _ { \mathsf { m o e } }$ is then passed through a task-specific linear layer with learnable weights $W _ { o }$ and bias $b _ { o }$ to generate the model predictions through (11).

$$
\hat { y } = \mathsf { s o f t m a x } ( W _ { o } h _ { \mathsf { m o e } } + b _ { o } )
$$

where $\hat { y }$ represents the predicted class probabilities. The parameters $W _ { g }$ . $b _ { g }$ , $W _ { o }$ , and $b _ { o }$ are all learnable and optimized jointly with the rest of the network via backpropagation, allowing the model to learn both how to combine experts and how to make accurate predictions.

![](images/2.jpg)  

Figure 2: Confusion matrix using the proposed StanceMoE architecture.   

Table 3: Training hyperparameters used in Stance-MoE.

et al., 2024) is used to create bi-directional context models which predict stances without requiring to use target information. (2) GCAE (Xue and Li, 2018) provides a convolutional system which employs a gating mechanism to block non-target features. (3) TAN (Du et al., 2017) uses an attentionboosted BiLSTM to identify important contextual details which help to determine stances. (4) Cross-Net (Du et al., 2017) enhances attention processing through an aspect-based attention layer which operates before the classification process to enhance target-based feature extraction.

# G. Evaluation Metrics

We evaluate model performance using macro-F1 score as the primary metric. Additionally, accuracy, macro precision, and macro recall are reported.

# D. Hyperparameters

Table 3 shows the hyperparameters used in the experiment.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Max sequence length</td><td>128</td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Number of epochs Learning rate</td><td>10 5 × 10-5</td></tr><tr><td>Number of splits (k)</td><td>10</td></tr><tr><td>Label smoothing factor</td><td>0.25</td></tr><tr><td>Random seed</td><td>42</td></tr></table>

# E. Baseline ML methods

We utilize four popular ML models as our baseline systems: (1) Logistic Regression (LR): A linear model that predicts class probabilities using weighted features, effective for high-dimensional text data. (2) Multinomial Naive Bayes (MNB): A probabilistic model based on word frequency distributions, assuming feature independence, wellsuited for text classification. (3) Support Vector Machine (SVM): A margin-based classifier that finds the optimal hyperplane to separate classes, robust in sparse text feature spaces. (4) Random Forest (RF): An ensemble of decision trees that improves classification performance by reducing overfitting through bagging and feature randomness.

# F. Baseline DNN

We use four popular deep neural network models as our baseline systems: (1) BiLSTM (Rahman

# H. Overall Ablation Study

Table 4 shows the overall ablation study on the test set, rather than class-specific one.

# I. Confusion matrix

Fig. 2 shows the confusion matrix using proposed StanceMoE architecture.

# J. Error analysis

The qualitative error analysis in Table 5 reveals several systematic patterns. (1) The model occasionally exhibits an excessive dependence on strong lexical polarity cues. The first and fifth examples show religious praise as well as humanitarian language which are mistakenly interpreted as explicit political alignment. (2) Meta-discursive commentary is sometimes confused with stance. The text in the third example describes antisemitism together with social norms without showing any specific support. The system uses target-related keywords to make stance predictions about the content. This is difficult because it requires the model to differentiate between entity mentioned and evaluative positioning. (3) Complex multi-clause and contrastive reasoning remains challenging. The second and sixth examples contain intricate discourse structures that display stance through their multiple levels of argumentation. The system design includes experts to handle both attention and contrast decoding tasks, but in some cases the gating system fails to deliver proper evaluation for these specialized discourse elements. (4) Very short and context-poor statements like the fourth example are inherently ambiguous. In such cases, the model appears to associate the polarity with dominant training patterns rather than grounding the prediction in the explicit evaluation of the target.

Table 4: Overall K-fold and weighted ensemble ablation study on the test set.   

<table><tr><td rowspan="2">Methods</td><td colspan="4">K-fold (mean±std)</td><td colspan="4">Weighted Logit Ensemble</td></tr><tr><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td><td>Acc</td><td>Pre</td><td>Rec</td><td>F1</td></tr><tr><td>w/o Mean</td><td>91.75±1.76</td><td>92.02±1.54</td><td>91.74±1.77</td><td>91.65±1.86</td><td>92.89</td><td>93.05</td><td>92.88</td><td>92.80</td></tr><tr><td>w/o Max</td><td>91.52±1.23</td><td>91.83±1.06</td><td>91.51±1.23</td><td>91.43±1.30</td><td>93.36</td><td>93.56</td><td>93.35</td><td>93.29</td></tr><tr><td>w/o Self-att</td><td>91.47±1.04</td><td>91.65±0.99</td><td>91.46±1.04</td><td>91.38±1.08</td><td>91.94</td><td>92.04</td><td>91.93</td><td>91.83</td></tr><tr><td>w/o CNN</td><td>92.09±1.04</td><td>92.20±0.97</td><td>92.08±1.04</td><td>92.00±1.11</td><td>93.36</td><td>93.38</td><td>93.37</td><td>93.29</td></tr><tr><td>w/o Lexical-cue</td><td>91.70±1.84</td><td>91.83±1.84</td><td>91.70±1.84</td><td>91.67±1.83</td><td>93.36</td><td>93.41</td><td>93.36</td><td>93.32</td></tr><tr><td>w/o Contrastive</td><td>91.75±1.81</td><td>91.99±1.67</td><td>91.74±1.82</td><td>91.64±1.87</td><td>91.94</td><td>92.13</td><td>91.94</td><td>91.82</td></tr><tr><td>StanceMoE</td><td>94.09±1.11</td><td>94.18±1.12</td><td>94.08±1.12</td><td>94.03±1.12</td><td>94.31</td><td>94.45</td><td>94.31</td><td>94.26</td></tr></table>

Tabl :Representative genuine misclassification cases of StanceMoE. Different background colors are used for visual distinction.

<table><tr><td>Text</td><td>Actual</td><td>Predicted</td><td>Primary Error Source</td></tr><tr><td>This is shameful. And the Jewish people, God&#x27;s cho- sen people. May God bless and protect you.</td><td>Neutral</td><td>Pro-Israel</td><td>Religious praise misinter- preted as political stance due to strong positive lexi- cal cues.</td></tr><tr><td>Most comments are clothed with sycophancy. Nigeri- ans have a right of association, but you can stand with Gaza. The people of Gaza started this aggressive be- haviour on 7th October when they launched a coordi- nated attack against the state of Israel.</td><td>Pro- Palestine</td><td>Neutral</td><td>Complex multi-clause rea- soning; insufficient empha- sis on dominant stance clause.</td></tr><tr><td>If you are talking about the doctor not talking about Palestine. He is German living in Germany where any anti-Jewish sentiment is taken very seriously and to many people anti-Israel equals antisemitism, so it is in his better interest not to speak about it at all, not nec- essarily meaning he supports anyone.</td><td>Neutral</td><td>Pro-Israel</td><td>Confusion between topic discussion and stance ex- pression due to keyword ac- tivation.</td></tr><tr><td>Against the Islamist terrorists who hate Jews and the West.</td><td>Neutral</td><td>Pro- Palestine</td><td>Short ambiguous input; po- larity incorrectly associated with a stance.</td></tr><tr><td>As I stood up with BLM, I stand against oppression in all forms. Release Palestinian hostages before talking about anything else.</td><td>Neutral</td><td>Pro- Palestine</td><td>Humanitarian framing inter- preted as explicit alignment due to lexical cues.</td></tr><tr><td>I was raised Muslim, and I know intimately how oppres- sive the religion can be in various ways. That being said, because I understand that Palestinian gay peo- ple are being oppressed for more than their sexuality right now, and that matters too.</td><td>Pro- Palestine</td><td>Neutral</td><td>Failure to capture nuanced contrastive stance; insuffi- cient weighting of discourse signals.</td></tr></table>

Overall, these errors suggest that while Stance-MoE is effective for explicit and lexically polarized stance expressions, nuanced and implicitly framed discourse remains a challenging direction for future improvement.

# K. Future Work

Future research may extend this framework toward explicitly target-aware stance modeling or explore cross-topic and cross-domain stance detection to further evaluate the robustness and generalization capacity of the proposed architecture. Investigating multilingual adaptation may also enhance its applicability to broader geopolitical discourse contexts.

# L. Acknowledgement

We would like to express our sincere gratitude to the organizer of StanceNakba 2026 shared task, and anonymous reviewers for their support. The authors acknowledge the use of ChatGPT (OpenAl) as an assistive tool for language refinement, code suggestions, and conceptual structuring. All generated content was critically reviewed, validated, and appropriately adapted by the authors, who take full responsibility for the accuracy and integrity of the work.