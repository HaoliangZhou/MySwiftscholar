# RETLLM: TRAINING AND DATA-FREE MLLMS FOR MULTIMODAL INFORMATION RETRIEVAL

Dawei Su Dongsheng Wang College of Computer Science and Software Engineering Shenzhen University, Shenzhen, China

# ABSTRACT

Multimodal information retrieval (MMIR) has gained attention for its flexibility in handling text, images, or mixed queries and candidates. Recent breakthroughs in multimodal large language models (MLLMs) boost MMIR performance by incorporating MLLM knowledge under the contrastive finetuning framework. However, they suffer from pre-training inconsistency and require large datasets. In this work, we introduce a novel framework, Ret LLM, designed to query MLLMs for MMIR in a training- and data-free manner. Specifically, we formulate MMIR as a similarity score generation task and prompt MLLMs to directly predict retrieval scores in a coarse-then-fine pipeline. At the coarse stage, a top-k filtering strategy builds a small yet high-quality candidate pool for each query, enabling MLLMs to focus on semantically relevant candidates. Subsequently, the retrieval score is predicted by feeding both the query and candidate into MLLMs at the fine stage. Importantly, we propose a visual enhancement module during reasoning to help MLLMs re-pick forgotten visuals, improving retrieval. Extensive experiments on MMIR benchmarks show that Ret LLM outperforms fine-tuned models. Ablation studies further verify each component. Our work demonstrates that MLLMs can achieve strong MMIR performance without any training, highlighting their inherent multimodal reasoning ability in a simple, scalable framework. We release our code at: https://github.com/alivecat05/RETLLM Index Terms— Multimodal Large Language Models (MLLMs), Multimodal information retrieval (MMIR), Prompt Engineering.

# INTRODUCTION

Multimodal information retrieval (MMIR) systems are expected to search across various modalities and provide relevant pieces of information according to user inputs, where queries and candidates can consist of pure images, text, or a composition of both. These systems play a crucial role in various downstream tasks, such as image-text retrieval, visual question answering (VQA), and retrieval-augmented generation (RAG) [1]. As a pioneering algorithm in multimodal representation learning, CLIP [2] has demonstrated strong performance in image-text retrieval via aligning each modality into a shared embedding space with contrastive training on image-text pairs. However, due to its reliance on modality-specific encoders, CLIP fails to cover more challenging cases, such as long-form text and interleaved image-text content.

In parallel, recent studies explore using multimodal large language models (MLLMs) as universal encoders by replacing CLIPstyle embeddings with MLLM-derived representations [3][4][5], treating MLLMs as universal encoders by inserting a summarization prompt: "<query>. Summarize above sentences in one word:", where <query> denotes the multimodal content. Then, a contrastive loss applied to the last token is used to fine-tune the parameters of MLLMs. For example, E5-V [5] trains MLLMs on text pairs using unimodal contrastive learning, achieving strong performance on complex multimodal retrieval tasks. Follow-up works [6][7][8][9]enhance E5-V via large-scale multimodal data, two-stage training, and hard negative mining. In addition to embedding learning, recent studies [10] treat MLLMs as rerankers to refine retrieval results, often requiring specialized training strategies such as noise injection. Despite these advances in MMIR performance, training-based approaches still exhibit several limitations:1) Objective misalignment: the inconsistency between autoregressive pretraining and contrastive fine-tuning may undermine the multimodal reasoning capabilities of MLLMs; and 2) Scalability bottleneck: the dependence on massive multimodal training pairs requires expensive collection costs and computational resources, limiting practical applications.

To address the above shortcomings, we introduce RetLLM, aiming at exploring the zero-shot retrieval potential of MLLMs in a training and data-free manner. Inspired by the recent success of MLLMs in regression tasks via string-based numeric prediction [6], Ret LLM reformulates retrieval as a similarity score prediction task, enabling complex queries such as long-text or compositional inputs without fine-tuning. To balance efficiency and accuracy, Ret LLM completes the retrieval in a coarse-then-fine pipeline. Given a multimodal query, we first collect a small-sized and high-quality candidate pool by employing a lightweight embedding-based model (e.g.,CLIP similarity). This coarse selection filters out samples with low semantic relevance to the query, not only reducing the retrieval time but also allowing MLLMs to focus more on the hard candidates. During the fine-selection stage, we prompt the MLLMs to predict the similarity score of the query and each candidate by feeding both of them into a multimodal instruction. The final prediction is obtained by choosing the candidate with the largest semantic score. Importantly, recent studies have shown that hallucinations in MLLMs often lead to impractical responses [11]. To address this, a visual enhancement strategy is developed to view visual tokens as supplementary evidence and allow MLLMs to re-pick up the forgotten features during the prediction process. Lastly, we further design an entropy-based decision-making strategy to deal with tied cases where multiple candidates receive the same highest similarity score in the fine stage. This enables our Ret LLM to consider the uncertainty score among confusing candidates, resulting in higher retrieval results. We summarize our contributions as follows: •We reformulate the multimodal retrieval task as a similarity score generation task, and show that MLLMs possess strong potential for various discriminative tasks.

![](images/1.jpg)  
Fig. 1: Overview of the RetLLM framework, which integrates Top $k$ filtering, vision enhancement, and entropy-based selection for effective multimodal retrieval

•We introduce Ret LLM, a training and data-free framework designed to employ MLLMs for MMIR, where the coarsethen-fine strategy is adopted for quick yet precise retrieval. • Extensive experiments on image-text retrieval and composed image retrieval tasks demonstrate the effectiveness of RetLLM. It outperforms CLIP-based baselines while achieving performance comparable to training-based MLLM models.

# 2. METHODOLOGY

We define a user query $q$ and $N$ candidates $\Omega = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { N } \}$ , where each $q$ and $c _ { n }$ can be an image, text, or interleaved image-text content. In this work, we focus on top-1 retrieval accuracy and aim to search for the best target $c$ for each $q$ .Our idea is very simple: we employ an MLLM as a similarity score generator and prompt the pre-trained MLLMs to generate the retrieval score of $q$ and $c _ { n }$ by feeding them into the input instruction. As shown in Fig. 1, our Ret LLM performs MMIR in a coarse-then-fine retrieval framework to balance the efficiency and accuracy. A visual enhancement and entropy-based decision-making are further developed to improve the final retrieval performance.

# 2.1. Coarse-Then-Fine Framework

Coarse Selection via Semantic Similarity. Intuitively, one can directly prompt MLLMs to generate the similarity score between the query $q$ and each candidate $c \in \Omega$ .Unfortunately, this naive attempt requires $N$ MLLM queries, leading to heavy time consumption. Generally, only a few samples in $\Omega$ act as valuable candidates of $q$ , we thus introduce the coarse-selection module, which forms a small-sized high-quality candidate pool $\mathcal { C }$ for each $q$ Mathematically, we select valuable candidates according to their semantic similarity to $q$ :

$$
\mathcal { C } = \mathrm { T o p K } ( s ) , \quad s _ { i } = \frac { { \bf q } ^ { \top } { \bf c } _ { i } } { | | { \bf q } | | | \bf { c } | | } , \quad i = 1 , 2 , \ldots , N ,
$$

where $\mathbf { q }$ and c denote the features of the query and candidate, respectively. TopK denotes that we select $k$ candidates with the largest similarity scores $s$ from $\Omega$ to form candidate pool $\mathcal { C }$ , serving as input for the subsequent fine-grained reranking stage. Fine-grained Selection with MLLMs. As discussed above, the candidate pool $\mathcal { C }$ contains hard samples that show high semantic relevance to $q$ , and embedding-based models (such as CLIP) fail to distinguish them correctly. Motivated by the impressive reasoning and generation ability of recent MLLMs, we view the retrieval task as a similarity score prediction problem. Specifically, unlike previous models that calculate the retrieval score in the embedding space, we here expect the MLLMs to generate it directly:

$$
f _ { i } = \mathbf { M L L M } ( q , c _ { i } ) , \quad c _ { i } \in \mathcal { C } .
$$

where an instruction template (shown in Fig. 1) is designed to take the query $q$ and its candidate $c$ as inputs and prompt MLLMs to predict the semantic similarity scores between them. Due to the small size of $\mathcal { C }$ , this fine-grained selection process reduces the MLLM query time from $N$ to $K$ , allowing the MLLMs to focus more on the hard candidates Note that our coarse-then-fine selection framework can be viewed as a hybrid algorithm that combines the representation learning of embedding-based models and the multimodal reasoning capabilities of MLLMs. Ret LLM first retrieves $K$ high-quality candidates according to the semantic features in the embedding space, and then explores the pre-trained knowledge encoded in MLLMs to understand the fine-grained differences between the query and hard candidates. The former stage offers fast inference speed but with inaccurate values, while the latter contributes fine-grained similarity scores but with low inference speed. The proposed framework combines the strengths of both under the coarse-then-fine strategy, effectively balancing efficiency and accuracy.

# 2.2. Visual Enhancement & Entropy-based Decision Making

Previous work [12][13] shows that due to fine-grained modality imbalance, MLLMs often suffer from hallucinations by losing fine-grained visual details during generation. Inspired by previous works for addressing hallucinations [11, 14, 15], we perform visual re-injection within the Feed-Forward Network (FFN) of the Transformer blocks. Specifically, we first reformulate the standard FFN as a key-value retrieval process. Let $\textbf { x } \in \ \mathbb { R } ^ { d }$ be the input hidden state of the FFN, with its vanilla form defined as: and Urban1K) and compositional benchmark (SugarCrepe). The best results are shown in bold.   

<table><tr><td rowspan="3">Method</td><td colspan="4">Short Caption Retrieval</td><td colspan="4">Long Caption Retrieval</td><td colspan="3">Compositional Retrieval</td></tr><tr><td colspan="2">Flickr30K</td><td colspan="2">COCO</td><td colspan="2">ShareGPT4V</td><td colspan="2">Urban1K</td><td rowspan="2">Replace</td><td rowspan="2">Swap</td><td rowspan="2">Add</td></tr><tr><td>$qi  c}$</td><td>$q frxc{ }$</td><td>qi → ct</td><td>$q frac{xc{ }$</td><td>$qi  c}$</td><td>qt → ci</td><td>qi → ct qt → ci</td><td></td></tr><tr><td>CLIP(ViT-L)</td><td>87.2</td><td>67.3</td><td>58.1</td><td>37.0</td><td>81.8</td><td>84.0</td><td>47.0</td><td>47.0</td><td>79.5</td><td>62.7</td><td>74.9</td></tr><tr><td>EEVA-CLIP</td><td>93.9</td><td>78.8</td><td>68.8</td><td>51.1</td><td>93.1</td><td>81.2</td><td>80.</td><td>77.</td><td>885.9</td><td>70.3</td><td>86.7</td></tr><tr><td>E5V</td><td>88.7</td><td>79.5</td><td>62.0</td><td>52.0</td><td>85.1</td><td>82.1</td><td>88.9</td><td>83.2</td><td>86.3</td><td>67.6</td><td>6.9</td></tr><tr><td>VLM2Vec</td><td>90.6</td><td>76.0</td><td>6.6</td><td>46.</td><td>889.8</td><td>86.9</td><td>91.</td><td>82.4</td><td>8.55</td><td>64.8</td><td>94.2</td></tr><tr><td>UniME</td><td>93.4</td><td>81.9</td><td>70.1</td><td>53.7</td><td>97.2</td><td>93.9</td><td>95.9</td><td>95.2</td><td>89.0</td><td>7.6</td><td>94.4</td></tr><tr><td>RetLLM</td><td>94.5</td><td>82.0</td><td>70.4</td><td>54.1</td><td>97.6</td><td>94.2</td><td>88.9</td><td>78.6</td><td>94.8</td><td>92.7</td><td>96.2</td></tr></table>

$$
\mathrm { F F N } ( \mathbf { x } ) = \phi ( \mathbf { x } \mathbf { W _ { 1 } } ) \mathbf { W _ { 2 } } ^ { \top } ,
$$

where $\phi$ is the activation function (e.g., ReLU or SiLU), and $\mathbf { W _ { 1 } } , \mathbf { W _ { 2 } } \in \mathbb { R } ^ { d \times D }$ y $D = 4 d$ We can rewrite $\mathbf { W _ { 1 } }$ and $\mathbf { W _ { 2 } }$ as:

$$
\mathbf { W _ { 1 } } = ( \mathbf { k _ { 1 } } , \mathbf { k _ { 2 } } , \ldots , \mathbf { k _ { D } } ) , \quad \mathbf { W _ { 2 } } = ( \mathbf { v _ { 1 } } , \mathbf { v _ { 2 } } , \ldots , \mathbf { v _ { D } } ) ,
$$

where $k _ { i } , v _ { i } \in \mathbb { R } ^ { d }$ denote the $_ { i }$ -th key and value vectors, respectively. As a result, the FFN can be reformulated as

$$
\mathrm { F F N } ( \mathbf { x } ) = \sum _ { i = 1 } ^ { D } \phi ( \left. \mathbf { x } , \mathbf { k } _ { \mathbf { i } } \right. ) \cdot \mathbf { v _ { i } } .
$$

This formulation reveals that the FFN acts as a memory module, using $_ x$ as a query to retrieve the relevant values. Intuitively, we introduce the visual token set $Z _ { v } = \{ z _ { v , 1 , . . . , z _ { v , N _ { v } } } \}$ as supplementary visual knowledge". When activating visual re-inject in layer $l$ , we treat visual tokens as new key-value entries and compute the correction term:

$$
\Delta ( \mathbf { x } \propto \mathbf { Z } _ { \mathbf { v } } ) = \sum _ { j = 1 } ^ { N _ { v } } \phi ( \langle \mathbf { x } , \mathbf { z } _ { \mathbf { v } , \mathbf { j } } \rangle ) \cdot \mathbf { z } _ { \mathbf { v } , \mathbf { j } } .
$$

Finally, the vanilla FFN output is fused with the visual correction:

$$
\mathrm { F F N } ^ { ( l ) } ( \mathbf { x } \propto \mathbf { Z } _ { \mathbf { v } } ) = \alpha \Delta ( \mathbf { x } \propto \mathbf { Z } _ { \mathbf { v } } ) + ( 1 - \alpha ) \mathrm { F F N } ( \mathbf { x } ) ,
$$

where $\alpha \in [ 0 , 1 ]$ is the injection ratio, and $x \propto Z _ { v }$ denotes executing visual re-injection from $x$ to visual features $Z _ { v }$ . This operation re-injects visual evidence into the intermediate layer without introducing additional trainable parameters, significantly enhancing the model's faithfulness to the input visual content. Another challenge comes from the similarity scores generated by MLLMs. We empirically find that multiple candidates may receive the same semantic score in Eq. 2, leading to ambiguity in ranking. To resolve such ties, we introduce an entropy-based confidence calibration strategy. Specifically, we design a confidenceaware instruction to measure the model uncertainty of the query and candidate $( q , c )$ pair: "<query>, $<$ candidate>. Does the candidate match the query, True or False.". Subsequently, the uncertainty score is obtained as the normalized entropy of the output logits at the last token:

$$
H _ { i } = - \sum _ { v = 1 } ^ { V } p _ { v } \log p _ { v } ,
$$

where $V$ is the vocabulary size, and $p _ { v }$ is the softmax probability of token $v$ in the model's output distribution. Lower entropy $H _ { i }$ indicates higher model certainty. Among candidates that share the same semantic score, we select the one with the minimum entropy:

$$
C ^ { * } = \arg \operatorname* { m i n } _ { C _ { i } \in \mathcal { P } } H _ { i } ,
$$

where $\mathcal { P }$ denotes the set of candidates with the same top-1 score. This confidence-aware selection strategy helps to refine the ranking when semantic distinctions are subtle, improving the reliability of the final retrieval results.

# 3. EXPERIMENTS

# 3.1. Datasets and Baselines

To comprehensively assess the effectiveness of our proposed Ret LLM, we evaluate it in a zero-shot setting on six benchmarks: Flickr30K [16], COCO [17], ShareGPT4V [18], Urban1K [19], SugarCrepe [20], and the MMEB [21] benchmark. For comparison, we include several strong baselines: CLIP, EVA-CLIP [22], E5-V [5], VLM2Vec [21], SigLIP [23], and UniME [9], all evaluated under the same zero-shot protocol.

# 3.2. Implementation Details and Evaluation Metrics

Our main experiment employs Qwen2.5-VL-7B [24] as the MLLM and CLIP-ViT-I $\mathcal { I } 1 4 @ 3 3 6 \mathrm { p x }$ for coarse-stage retrieval, with top-5 candidates passed to the fine stage. Key enhancements include: 1) Visual re-injection $( \alpha { = } 0 . 3 )$ to mitigate hallucination; 2) Entropybased decision for ambiguous top scores. For ablation, we vary CLIP backbones (ViT-B, Long-CLIP-L[19]) and MLLMs (Phi-3.5-V [25], Qwen2-VL [26]). All experiments are zero-shot. Primary metrics: Recall $@ 1$ , measuring queries with correct target ranked first. On MMEB, we report average Precision $@ 1$ over all meta-tasks.

# .3. Results Analysis

Ret LLM achieves strong zero-shot performance across all benchmarks without any training or fine-tuning. As shown in Table 1, it consistently outperforms both zero-shot baselines (e.g., CLIP, EVA-CLIP) and MLLM retrievers such as E5-V and VLM2Vec. For example, on Flickr30K $\begin{array} { l } { { ( q ^ { i } \ \to \ c ^ { t } , } } \end{array}$ , RetLLM reaches $9 4 . 5 \%$ $\mathbb { R } \ @ 1$ , surpassing E5-V $( 8 8 . 7 \% )$ and VLM2Vec $( 9 0 . 6 \% )$ . On ShareGPT4V $( q ^ { i } \to c ^ { i } )$ , RetLLM achieves $9 4 . 2 \%$ $\mathbf { R } \ @ 1$ , outperforming VLM2Vec $( 8 6 . 9 \% )$ . On SugarCrepe "Add", RetLLM achieves $9 6 . 2 \%$ ,a $2 \%$ gain over VLM2Vec $( 9 4 . 2 \% )$ , demonstrating superior zero-shot reasoning. On MMEB (Table 2), Ret LLM obtains $5 4 . 2 \%$ overall Precision $@ 1$ , a $1 2 . 6 \%$ improvement over the strongest zero-shot baseline, UniME. It excels in Retrieval $( 6 2 . 4 \% )$ , Classification $( 6 0 . 2 \% )$ , and VQA $( 2 7 . 8 \% )$ , proving its robustness in zero-shot scenarios. These results confirm that, with our coarsethen-fine pipeline, visual enhancement, and entropy-based selection, Ret LLM serves as a powerful zero-shot retriever. The reported scores are the average Precision $@ 1$ over the corresponding datasets on zero-shot manner. The best results are marked in bold.   

<table><tr><td>Models</td><td>#Parameters</td><td colspan="4">Per Meta-Task Score</td><td colspan="3">Average Score</td></tr><tr><td></td><td></td><td>Classification</td><td>VQA</td><td>Retrieval</td><td>Grounding</td><td>IND</td><td>OOD</td><td>Overall</td></tr><tr><td># of Datasets →</td><td></td><td>10</td><td>10</td><td>12</td><td>4</td><td>20</td><td>16</td><td>36</td></tr><tr><td>CLIP(ViT-L)</td><td>0.4B</td><td>42.8</td><td>9.1</td><td>53.0</td><td>51.8</td><td>37.1</td><td>38.7</td><td>39.2</td></tr><tr><td>OpenCLIP(ViT-L)</td><td>0.4B</td><td>41.5</td><td>6.9</td><td>44.6</td><td>53.5</td><td>32.8</td><td>36.0</td><td>36.6</td></tr><tr><td>SigLIP(So/14)</td><td>0.9B</td><td>40.3</td><td>8.4</td><td>31.6</td><td>559.5</td><td>32.3</td><td>38.0</td><td>35.0</td></tr><tr><td>CLIP(ViT-BigG/14)</td><td>2.5B</td><td>52.3</td><td>14.0</td><td>50.5</td><td>60.3</td><td>38.9</td><td>45.8</td><td>44.3</td></tr><tr><td>EVA-CLIP</td><td>8B</td><td>56.00</td><td>10.4</td><td>49.2</td><td>558.9</td><td>38.1</td><td>456</td><td>43.7</td></tr><tr><td>E5-V UniME</td><td>7B 7B</td><td>39.7 43.0</td><td>10.8</td><td>39.4</td><td>60.2</td><td>34.2</td><td>33.4</td><td>37.5</td></tr><tr><td></td><td></td><td></td><td>17.7</td><td>42.5</td><td>63.2</td><td>37.6</td><td>38.6</td><td>41.6</td></tr><tr><td>RetLLM</td><td>7B</td><td>60.3</td><td>27.8</td><td>62.4</td><td>60.2</td><td>52.0</td><td>50.2</td><td>54.2</td></tr></table>

Table 3: Ablation study of visual enhancement and entropy-based selection on Flickr30k and COCO.   

<table><tr><td>Components</td><td colspan="2">Flickr30k</td><td colspan="2">COCO</td></tr><tr><td></td><td>qi → ct qt → ci</td><td></td><td>qi → ct</td><td>qt → ci</td></tr><tr><td>ALL</td><td>94.5</td><td>81.8</td><td>69.2</td><td>52.1</td></tr><tr><td>entropy only</td><td>94.0</td><td>81.2</td><td>68.7</td><td>50.8</td></tr><tr><td>enhancement only</td><td>94.0</td><td>80.7</td><td>68.8</td><td>51.5</td></tr><tr><td>MLLM only</td><td>93.6</td><td>80.2</td><td>66.9</td><td>50.3</td></tr></table>

# 3.4. Ablation Study

Components Effectiveness As shown in Table 3, removing visual enhancement causes a notable $1 . 5 \%$ drop on COCO $( q ^ { i } \to c ^ { \bar { t } } )$ , confirming its critical role in preserving visual fidelity during zero-shot retrieval. Disabling entropy-based selection leads to a $1 . 1 \%$ decrease on Flickr30K $( q ^ { t } \to c ^ { i \cdot }$ ), demonstrating its effectiveness in resolving ambiguous rankings. The consistent superiority of the full model over "MLLM only" underscores the synergistic gain from combining both components.

Table 4: Performance comparison using different CLIP versions with fixed Qwen2.5-VL.   
Top $k$ Sensitivity As shown in Fig. 2, performance varies with different $k$ values (3, 5, 7, 9), revealing a clear trade-off between precision and efficiency: a larger $k$ improves recall at higher computational cost, while $k { = } 5$ (our default) offers the optimal balance for practical deployment.   

<table><tr><td rowspan="2">CLIP-version</td><td colspan="2">ShareGPT4V</td><td colspan="2">Urban1k</td></tr><tr><td>qi → ct qt → ci</td><td></td><td>qi → ct qt → ci</td><td></td></tr><tr><td>CLIP-ViT-B</td><td>94.8</td><td>88.1</td><td>84.0</td><td>71.8</td></tr><tr><td>CLIP-ViT-L</td><td>97.6</td><td>94.2</td><td>88.9</td><td>78.6</td></tr><tr><td>Long-CLIP-L</td><td>96.6</td><td>95.1</td><td>95.2</td><td>95.8</td></tr></table>

![](images/2.jpg)  
Fig. 2: Ablation studies on the impact of top- $\mathbf { \nabla } \cdot \mathbf { k }$ values on retrieval performance and inference efficiency.

Table 5: Performance comparison using different MLLMs with fixed CLIP-ViT-L.   

<table><tr><td>MLLMs</td><td colspan="2">ShareGPT4V</td><td colspan="2">Urban1k</td></tr><tr><td></td><td>qi → ct</td><td>qt → ci</td><td>qi → ct</td><td>qt → ci</td></tr><tr><td>phi3.5v</td><td>86.5</td><td>72.3</td><td>78.9</td><td>73.5</td></tr><tr><td>Qwen2-VL</td><td>93.8</td><td>94.1</td><td>87.2</td><td>78.2</td></tr><tr><td>Qwen2.5-VL</td><td>97.6</td><td>94.2</td><td>88.9</td><td>78.6</td></tr></table>

Model Scalability As shown in Table 4 and Table 5, RetLLM benefits from stronger and larger backbone models, with performance improving consistently as the capacity of the underlying components increases. This highlights the scalability of our framework and its ability to leverage advances in both CLIP models and multimodal large language models.

# 4. CONCLUSION

In this work, we propose Ret LLM, a training-free multimodal retrieval framework that achieves strong zero-shot performance through coarse-then-fine search, visual enhancement, and entropybased selection. Crucially, Ret LLM is highly scalable: it naturally inherits performance gains from stronger foundation models in a plug-and-play manner, making it a forward-compatible and sustainable solution for future retrieval systems.

5. REFERENCES   
[1] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau iel   Ral knowledge-intensiv lp tasks,"Advances inneural inoration pocessing systems, vol. 33, pp. 94599474, 2020.   
[2] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., "Learning transferable visual models from natural language supervision," in International conference on machine learning. PmLR, 2021, pp. 87488763.   
[3] Dongsheng Wang, Miaoge Li, Xinyang Liu, MingSheng Xu, Bo Chen, and Hanwang Zhang, "Tuning multi-mode token-level prompt alignment across modalities," Advances in Neural Information Processing Systems, vol. 36, pp. 5279252810, 2023.   
[4] Cong Wei, Yang Chen, Haonan Chen, Hexiang Hu, Ge Zhang, Jie Fu, ARiter n Wn ChnUTag n ben uversal ultodal oratio etrvers," in uropen Confen on Computer Vision. Springer, 2024, pp. 387404.   
[5] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang, "E5-v: Universal embeddings with multimodal large language models," arXiv preprint arXiv:2407.12580, 2024.   
[6] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping, "Mm-embed: Universal multimodal retrieval with multimodal llms," arXiv preprint arXiv:2411.02571, 2024.   
[7] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su, "Llave: Large language and vision embedding models with hardnessweighted contrastive learning," arXiv preprint arXiv:2503.04812, 2025.   
[8] Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie, "Lamra: Large multimodal model as your advanced retrieval assistant," in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 40154025.   
[9] Tiancheng Gu, Kaicheng Yang, Ziyong Feng, Xingjun Wang, Yanzhao Zhang, Dingkun Long, Yingda Chen, Weidong Cai, and Jiankang Deg "Breaking the modality barrier: Universal embedding learning with multimodal llms," arXiv preprint arXiv:2504.17432, 2025.   
[10] Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo, "Mllm is a strong reranker: Advancing multimodal retrieval-augmented generation via knowledge-enhanced reranking and noise-injected training," arXiv preprint arXiv:2407.21439, 2024.   
[11] Xin Zou, Yizhou Wang, Yibo Yan, Yuanhuiyi Lyu, Kening Zheng, Sirui Huang, Junkai Chen, Peijie Jiang, Jia Liu, Chang Tang, et al., "Look twice before you answer: Memory-space visual retracing for hallucination mitigation in multimodal large language models," arXiv preprint arXiv:2410.03577, 2024.   
[12] Dongsheng Wang, Jiequan Cui, Miaoge Li, Wang Lin, Bo Chen, and Hanwang Zhang, "Instruction tuning-free visual token complement for multimodal llms," in European Conference on Computer Vision. Springer, 2024, pp. 446462.   
[13] Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko, "Object hallucination in image captioning," arXiv preprint arXiv:1809.02156, 2018.   
[14] Junyan Lin, Haoran Chen, Yue Fan, Yingqi Fan, Xin Jin, Hui Su, Jinlan Fu, and Xiaoyu Shen, "Multi-layer visual feature fusion in multimodal llms: Methods, analysis, and best practices," in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 4156 4166.   
[15] Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, and Huajun Chen, "Mllm can see? dynamic correction decoding for hallucination mitigation," arXiv preprint arXiv:2410.11779, 2024.   
[16] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier, "From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions," Transactions of the association for computational linguistics, vol. 2, pp. 6778, 2014.   
[17] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick, "Microsoft coco: Common objects in context," in European conference on computer vision. Springer, 2014, pp. 740755.   
[18] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin, "Sharegpt4v: Improving large multi-modal models with better captions," in European Conference on Computer Vision. Springer, 2024, pp. 370387.   
[19] Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang, "Long-clip: Unlocking the long-text capability of clip," in European conference on computer vision. Springer, 2024, pp. 310325.   
[20] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna, "Sugarcrepe: Fixing hackable benchmarks for visionlanguage compositionality," Advances in neural information processing systems, vol. 36, pp. 3109631116, 2023.   
[21] Ziyan Jiang, Rui Meng, Xinyi Yang, Semih Yavuz, Yingbo Zhou, and Wenhu Chen, "Vlm2vec: Training vision-language models for massive multimodal embedding tasks," arXiv preprint arXiv:2410.05160, 2024.   
[22] Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, and Xinlong Wang, "Eva-clip-18b: Scaling clip to 18 billion parameters," arXiv preprint arXiv:2402.04252, 2024.   
[23] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer, "Sigmoid loss for language image pre-training," in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 1197511986.   
[24] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin, "Qwen2.5-vl technical report," arXiv preprint arXiv:2502.13923, 2025.   
[25] Marah Abdin, Jyoti Aneja, and Hany Awadalla et.al., "Phi-3 technical report: A highly capable language model locally on your phone," 2024.   
[26] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin, "Qwen2-vl: Enhancing visionlanguage model's perception of the world at any resolution," arXiv preprint arXiv:2409.12191, 2024.