# A Sanity Check on Composed Image Retrieval

Yikun Liu1,2, Jiangchao $\mathrm { Y a o ^ { 2 \dagger } }$ , Weidi Xie1, Yanfeng Wang1† 1School of Artificial Intelligence, Shanghai Jiao Tong University, China 2CMIC, Shanghai Jiao Tong University, China

![](images/1.jpg)  
F

# Abstract

Composed Image Retrieval (CIR) aims to retrieve a target image based on a query composed of a reference image, and a relative caption that specifies the desired modification. Despite the rapid development of CIR models, their performance is not well characterized by existing benchmarks, which inherently contain indeterminate queries degrading the evaluation (i.e., multiple candidate images, rather than solely the target image, meet the query criteria), and have not considered their effectiveness in the context of the multiround system. Motivated by this, we consider improving the evaluation procedure from two aspects: 1) we introduce

FISD, a Fully-Informed Semantically-Diverse benchmark, which employs generative models to precisely control the variables of reference-target image pairs, enabling a more accurate evaluation of CIR methods across six dimensions, without query ambiguity; 2) we propose an automatic multiround agentic evaluation framework to probe the potential of the existing models in the interactive scenarios. By observing how models adapt and refine their choices over successive rounds of queries, this framework provides a more realistic appraisal of their efficacy in practical applications. Extensive experiments and comparisons prove the value of our novel evaluation on typical CIR methods. The project page is available here.

# 1. Introduction

Recently, Composed Image Retrieval (CIR) has gained significant momentum, driven by the impressive success of Vision-Language Pre-training (VLP) methods [20, 21, 38]. This task entails retrieving a target image from a query composed of a reference image, and a relative caption specifying the reference-to-target modification. In comparison to traditional image-to-image or text-to-image retrieval, CIR captures a broader and more nuanced semantic understanding of user's intentions, which is particularly advantageous in fields such as e-commerce [50] and internet search [56].

Existing explorations in CIR can be broadly divided into two categories. The first set [2, 4, 9, 12, 28, 42] involves training on the standardized triplets, each comprising a reference image, a relative caption, and a target image. These triplets may be manually annotated with extensive human effort or automatically generated via generative models with additional human refinement. The second line [3, 10, 39, 43] utilizes more readily accessible data, such as the image-text pairs, to learn a mapping network that projects the image feature to the text space, ultimately achieving feature fusion within the text domain.

Despite the progress made on mainstream benchmarks like FashionIQ [50] and CIRR [31], we argue that existing benchmarks fail to adequately evaluate CIR methods due to the presence of spurious samples. As shown in Figure 1, many mainstream benchmarks include indeterminate composed queries1, where multiple candidates satisfy the query requirements. Such ambiguity undermines the accurate assessment of CIR models. Besides, current evaluation often focuses on the performance of models in handling onetime queries, overlooking their effectiveness in the interactive scenarios, which is practical in the multi-round system. To take steps forward, we aim to build a more accurate and comprehensive evaluation suite to better characterize the performance of the CIR models.

In specific, to address the issue of in-determinate queries, we introduce FISD, a Fully-Informed Semantically Diverse benchmark, which employs the diffusion models to generate controllable reference-target image pairs, thereby reducing the occurrence of indeterminate queries. Our FISD evaluates the CIR models across six dimensions (cardinality, addition, negation, change, background, and complex instructions), revealing that existing CIR models still exhibit poor ability in handling negation and cardinality logic, and require more improvement. Furthermore, to probe the potential of CIR models in interactive scenarios, we propose an automated multi-round evaluation framework consisting of three key components: an off-the-shelf CIR model, a ranker, and a user simulator. Initially, the CIR model combines the reference/candidate image with the relative caption to generate a composed feature for the ranker. The ranker then iteratively updates the candidate image set based on historical interactions and the image database. The user simulator modeled by open-source vision-language foundation models evaluates whether the target image is within the top- $k$ candidates to promote search; The evaluation demonstrates that existing models, when subjected to multiple rounds of iteration, can achieve significantly improved performance, greatly surpassing their single-round performance.

In summary, our contributions are as follows: 1) we introduce FISD, a novel Fully-Informed Semantically Diverse benchmark to address the ambiguity issue across six dimensions in the evaluation of existing CIR models; 2) we propose an automated, multi-round evaluation framework, to measure the performance of CIR models under multi-round interaction. This framework employs the vision-language foundation models to simulate the user interactions, thereby generating relative caption feedback that more effectively mirrors real-world situations; 3) our comprehensive evaluation uncovers that current CIR models have significant potential for performance improvement, particularly in terms of semantics related to cardinality and negation. Furthermore, CIR models exhibit significant improvements when engaged in multi-round interactions, substantially outperforming their single-round counterparts.

# 2. Related Work

Composed Image Retrieval. Composed Image Retrieval (CIR) considers retrieving a target image based on the reference image and a relative caption that describes the modification. Current methods can be primarily divided into two categories. The first set involves methods like CLIP4CIR [4], BLIP4CIR [32] and SPRC [2] that are trained on human-annotated data or approaches [9, 19, 28] that use generative models to synthesize the data. These methods require the standardized triplets comprising a reference image, a relative caption, and a target image, using contrastive learning to minimize the distance between the composed features of the reference image and the relative caption, and the features of the target image. The second line, known as zero-shot CIR methods, does not rely on human-annotated triplets. Some of them [3, 10, 39, 44] train a mapping network using data in various modalities to project images into the textual feature space. By concatenating these features with those extracted from the relative captions, they can leverage the text-to-image retrieval capability of multimodal models like CLIP [38] to retrieve target images effectively. Additionally, several efforts [22, 45, 53] have explored training-free methods to address this task.

Interactive Retrieval. The interactive image/video retrieval system is designed to iteratively capture feedback to facilitate retrieval. For example, some studies [18, 23,

![](images/2.jpg)  
marked with an orange rame, the target mage with a red frame, and hard negative image with a rayrame.

33, 34] focus on enhancing text-to-image/video retrieval by iteratively acquiring more information in the form of visual question answering. It entails utilizing a question generator that formulates questions based on candidates from each round and historical data, followed by a user simulator to respond to these questions. Other approaches advocate providing differential feedback to describe the semantic gap between target and candidate images to improve image retrieval performance. Specifically, Whittle-Search [16] adopts a tag-based method, primarily offering feedback through attributes, whereas FashionIQ [50] utilizes a sentence-level method, training a relative captioner capable of generating the relative caption between candidate images and target images within the fashion domain. Inspired by these interactive methods, we propose an automated multi-round evaluation framework designed to assess how CIR models adapt and refine their selections over successive rounds of queries.

Generative Models. The recent significant advancements in Multimodal Large Language Models (MLLMs) and Diffusion Models have catalyzed the emergence of numerous innovative applications [30, 35, 51, 52, 55]. For instance, LLaVA [26] applies MLLMs to visual question answering tasks, while LISA [17] utilizes MLLMs for segmentation tasks. Likewise, LamRA [29] leverages MLLMs for universal retrieval tasks, and LLM-as-a-judge [57] uses LLMs as judges to evaluate the performance of various models on open-ended questions. In parallel, StoryGen [25] employs Diffusion Models to enhance visual storytelling tasks, and InstaGen [7] utilizes these models to synthesize data for object detection tasks. In this paper, we aim to build on this success by developing a comprehensive evaluation suite of CIR models using MLLMs and Diffusion Models. This suite aims to provide a robust framework for assessing and advancing CIR models, fostering further innovation and application in the field.

# 3. A New Sanity Check for CIR

In the Composed Image Retrieval (CIR) task, each sample can be denoted as a triplet, i.e., ${ \mathcal D } = \{ ( { \bf I } ^ { \mathrm { r e f } } , { \bf I } ^ { \mathrm { t g t } } , t ) \ |$ ${ \sf I ^ { \mathrm { r e f } } } , { \sf I ^ { \mathrm { t g t } } } \in \mathbb { R } ^ { H \times W \times 3 } , t \in \mathbb { R } ^ { L \times 1 } \}$ Here, $H$ and $W$ denote the height and width of an image, and $L$ refers to the sequence length of the relative caption. Specifically, the model takes the reference image $( \boldsymbol { \mathsf { I } } ^ { \bar { \mathrm { r e f } } } )$ and relative caption $\mathbf { \eta } ( t )$ as input, and construct a composed query $\pmb q$ ,which is further used to retrieve the target image $( { \bf l } ^ { \mathrm { t g t } } )$ from the set $( \Omega = \{ \mathsf { I } _ { i } , i = 1 , \cdot \cdot \cdot , N \} )$ , where $N$ represents the image number in the database.

In the literature, significant advancements [2, 4, 29] have been made in common benchmarks, for example, CIRR [31] and FashionIQ [50]. However, as illustrated in Figure 1 and Section F of the supplementary material, we argue that the indeterminate composed queries, which correspond to numerous candidate images in popular CIR datasets, actually pose challenges for precise evaluation. To comprehensively characterize the performance of current CIR models, we explore building a more comprehensive evaluation suite to better measure CIR models, encompassing a novel CIR benchmark and an automated multiround evaluation framework.

Table 1. Performance of various CIR models on FISD benchmark.   

<table><tr><td></td><td>Cardinality</td><td>Addition</td><td>Negation</td><td>Change</td><td>Background</td><td>Complex Inst.</td><td>Average</td></tr><tr><td>Model</td><td>Recall@1</td><td>Recall@1</td><td>Recall@1</td><td>Recall@1</td><td>Recall@1</td><td>Recall@1</td><td>Recall@1</td></tr><tr><td>Pic2Word [39]</td><td>6.00</td><td>39.00</td><td>0.00</td><td>24.00</td><td>18.00</td><td>41.00</td><td>21.33</td></tr><tr><td>Context-I2W [44]</td><td>14.00</td><td>48.00</td><td>0.00</td><td>29.50</td><td>27.50</td><td>48.00</td><td>27.83</td></tr><tr><td>LinCIR [10]</td><td>5.50</td><td>52.50</td><td>1.50</td><td>43.50</td><td>36.50</td><td>55.00</td><td>32.42</td></tr><tr><td>TransAgg [28]</td><td>15.50</td><td>56.00</td><td>2.00</td><td>52.00</td><td>26.00</td><td>57.00</td><td>34.75</td></tr><tr><td>CLIP4CIR [4]</td><td>31.00</td><td>67.50</td><td>4.00</td><td>65.00</td><td>43.50</td><td>72.00</td><td>47.17</td></tr><tr><td>BLIP4CIR+Bi [32]</td><td>35.50</td><td>65.50</td><td>3.50</td><td>58.50</td><td>42.50</td><td>65.50</td><td>45.17</td></tr><tr><td>TG-CIR [49]</td><td>43.50</td><td>69.00</td><td>1.50</td><td>64.50</td><td>47.50</td><td>65.50</td><td>48.58</td></tr><tr><td>CoVR* [47]</td><td>47.50</td><td>70.00</td><td>8.50</td><td>77.00</td><td>61.50</td><td>70.00</td><td>55.75</td></tr><tr><td>CIReVL [15]</td><td>24.00</td><td>53.50</td><td>17.50</td><td>61.00</td><td>40.00</td><td>69.00</td><td>44.17</td></tr><tr><td>SPRC [2]</td><td>41.00</td><td>66.50</td><td>9.50</td><td>73.00</td><td>48.50</td><td>62.00</td><td>50.08</td></tr><tr><td>SPN4CIR [8]</td><td>58.50</td><td>69.50</td><td>13.00</td><td>77.00</td><td>56.00</td><td>61.00</td><td>55.83</td></tr></table>

# 3.1. Fully-Informed Semantically-Diverse Benchmark

For our benchmark, we consider leveraging Large Language Models (LLMs) and Diffusion Models (DMs) to synthesize the samples, since it enables us to flexibly control the variable between paired reference-target images, allowing for an exact evaluation without the spurious samples. In addition, we focus on six categories (cardinality, addition, negation, change, background, and complex instruction) to cover different real-world retrieval types, and perform careful human verification to guarantee quality. In the following, we will present the FISD construction process, which consists of caption generation and image generation as illustrated in Figure 2.

Caption Generation. To obtain the triplet sample for CIR, on the textual side, we need the reference image caption, relative caption, and target image caption. For the reference image captions, we select from existing image caption datasets, such as COC0 [24], CC3M [40], Flickr30k [54], etc. Then, with this caption for the reference image, we prompt Mixtral-8x7B-Instruct-v0.1 [14] to generate the relative caption and the target image caption, with a designed prompt. For more details about the prompts, we refer the reader to Section A of the supplementary material.

Image Generation. After obtaining captions, we leverage Stable-Diffusion-xl-base-1.0 [37] to generate the images. To ensure consistency between the reference and target images during this process. To minimize the risk of false negatives, we employ the same random seed. Furthermore, to narrow the gap between the generated and natural images, we manually curate the dataset by excluding cartoon-style images, logically flawed images, or those that are overtly unrealistic. Further analysis is provided in Section 4.3. In total, we obtain 200 triplets for each category, resulting in a comprehensive dataset of 1200 triplets (3600 images).

To challenge the existing CIR models, and avoid trivial solutions, such as retrieving the target image based solely on image similarity, we generate hard negative images for each triplet. These images are crafted by manually adjusting the corresponding caption to ensure it closely resembles either the target image or the relative caption. For instance, in the "change" category, the relative caption can be "replacing the reference image with a girl wearing long pants", while the hard negative images might be "a girl wearing shorts or a boy wearing long pants". This characterizes the challenge of our FISD benchmark, where the model must fully comprehend both the reference image and the relative caption to effectively avoid retrieving the hard negative images.

# 3.2. Multi-round Evaluation Framework

Conventional evaluation metric often emphasizes the performance of models in managing single queries, neglecting the dynamic and evolving context that characterizes realworld applications. To overcome this limitation, we propose that the evaluation of CIR models should incorporate multiturn interactions. Ideally, multi-turn evaluations would involve human participation. If a query in the current round does not retrieve the target image, the user should be able to provide further feedback based on the retrieved images, and this process would continue iteratively.

However, employing humans for multi-round evaluation can be challenging and inconvenient. Therefore, we propose an automated multi-round evaluation framework to understand how the CIR models adapt and refine their choices over successive rounds of queries. Our evaluation framework is composed of three main components: a generic off-the-shelf CIR model, a ranker, and a user simulator intended to mimic user feedback. In the subsequent sections, we present a detailed description of the workflow and the critical components of this framework.

Interaction Workflow. As shown in Figure 3, a user first inputs a reference image and a relative caption into the retrieval system for searching a target image. Then, the system outputs candidate images, which contain $M$ items. This $M$ can be varied. If the target image is included in the top-$k$ candidate images, the retrieval is successfully finished. If not, the user simulator receives the candidate images and the target image to output a relative caption, which describes the difference between the candidate image and the target image. Here, this candidate image can vary from top-1 to top- $M$ , but we specifically use the top-1 candidate. Such a loop is iteratively performed until the user reaches the target image in the top- $k$ candidate images or the round of interaction reaches a predetermined upper limit $R _ { \mathrm { m a x } }$ .

![](images/3.jpg)

CIR Model. Considering the $r$ -th round interaction $\begin{array} { r c l } { ( r } & { = } & { 1 , \cdots , R _ { \mathrm { m a x } } ) } \end{array}$ , the CIR model takes the candidate image $\mathbf { I } _ { r } ^ { \mathrm { c a n d } }$ and the relative caption $\scriptstyle t _ { r }$ as inputs, and construct a composed query feature $\begin{array} { r c l } { \bar { { q _ { r } } } } & { = } & { \displaystyle \mathrm { ~ f u s ~ i ~ o n } \big ( f _ { \mathrm { v i s u a l } } ( \mathsf { I } _ { r } ^ { \mathrm { c a n d } } ) , \bar { f _ { \mathrm { t e x t } } } ( \pmb { t } _ { r } ) \big ) ^ { - } \in \mathbb ~ \mathbb { R } ^ { 1 \times d } } \end{array}$ , where $\mathtt { f u s i o n } ( \cdot , \cdot )$ refers to fusion method employed within CIR model, $f _ { \mathrm { v i s u a l } }$ denotes the visual encoder for extracting image features, $f _ { \mathrm { t e x t } }$ represents the text encoder for extracting text embeddings, and $d$ refers to the feature dimension. Following this, we utilize the visual encoder to encode all images $\{ \mathsf { I } _ { i } \} _ { i = 1 } ^ { N }$ i hea atab   e imag features $\mathbb { V } = \{ v _ { i } \} _ { i = 1 } ^ { N }$ where $v _ { i } \in \mathbb { R } ^ { 1 \times d }$ Fially, both the composed query feature $\pmb q _ { r }$ and image features $\mathbb { V }$ are fed into the ranker to perform the ranking. It is worth noting that any off-the-shelf CIR model can be adapted to our framework.

Ranker. The ranker receives $r$ th round's composed query feature $\pmb q _ { r }$ along with all image features $\mathbb { V }$ Furthermore, the ranker maintains a history list $\pmb { H } = [ \pmb { q } _ { 1 } , \cdots , \pmb { q } _ { r } ]$ , which aggregates the composed query feature from each round. This facilitates the tracking of query features across different rounds. Subsequently, the ranker computes the distance between the history representation $f _ { h } ( H )$ and each image feature vector $v \in \mathbb { V }$ using the equation:

$$
\mathbf { l } _ { r + 1 } ^ { \mathrm { c a n d } } = \mathop { \mathrm { a r g m i n } } _ { v \in \mathbb { V } } \mathrm { s i m } ( f _ { h } ( H ) , v ) .
$$

Here, $f _ { h }$ represents a fusion function applied to the history list, specifically a simple averaging operation defined by $\begin{array} { r } { f _ { h } ( { H } ) ^ { - } = \frac { 1 } { r } \sum _ { i = 1 } ^ { \bar { r } } q _ { i } } \end{array}$ e over all elements from the first round up to the current $r$ . th round. $\operatorname { s i m } ( \cdot , \cdot )$ denotes the cosine distance. Finally, the ranker utilizes a greedy selection strategy, opting for the image with the smallest cosine distance as the candidate.

User Simulator. The user simulator operates as an oracle, adept at identifying differences between the candidate image $\mathbf { I } _ { r + 1 } ^ { \mathrm { c a n d } }$ and the target image ${ \sf I } ^ { \mathrm { t g t } }$ [50]. Here, to mimic human interactions, we consider integrating the pre-trained foundation models, including the Multimodal Large Language Model (MLLM) and Large Language Model (LLM). Concretely, with MLLM and LLM, the user feedback $\mathbf { \delta } _ { t _ { r + 1 } }$ is provided in the form of a relative caption, which can be formally represented as follows:

$$
\begin{array} { r } { \pmb { t } _ { r + 1 } = \mathtt { L L M } \left( \mathtt { M L L M } ( \pmb { \Vert } _ { r + 1 } ^ { \mathrm { c a n d } } ) , \mathtt { M L L M } ( \pmb { \Vert } ^ { \mathrm { t g t } } ) \right) . } \end{array}
$$

The process involves generating detailed captions for both images using the MLLM. These captions are then processed by the LLM to generate user feedback that highlights the discrepancy. Upon generating user feedback, it is combined with the candidate image as input into the CIR model for subsequent iterations. This process is iterated until either of two conditions is met: the target image is ranked within the top $k$ results, namely, a successful retrieval, or the maximum number $R _ { m a x }$ is reached. For detailed prompts, please refer to Section B of the supplementary material.

Table 2. Multi-round evaluation on various state-of-the-art CIR models across a range of benchmarks.   

<table><tr><td rowspan="2">Method</td><td colspan="2">FashionIQ-Dress</td><td colspan="2">FashionIQ-Shirt</td><td colspan="2">FashionIQ-Toptee</td><td colspan="2">CIRR</td><td>CIRCO</td><td>FISD</td></tr><tr><td>Hits@10</td><td>Hits@50</td><td>Hits@10</td><td>Hits@50</td><td>Hits@10</td><td>Hits@50</td><td>Hits@1</td><td>Hits@5</td><td>MAP@5</td><td>Hits@1</td></tr><tr><td colspan="9">Round 1</td><td></td></tr><tr><td>Pic2Word [39]</td><td>20.00</td><td>40.20</td><td>26.20</td><td>43.60</td><td>27.90</td><td>47.40</td><td>23.25</td><td>51.42</td><td>7.39</td><td>21.33</td></tr><tr><td>Context-I2W [44]</td><td>23.10</td><td>45.30</td><td>29.70</td><td>48.60</td><td>30.60</td><td>52.90</td><td>26.96</td><td>56.59</td><td>11.96</td><td>27.83</td></tr><tr><td>LinCIR [10]</td><td>20.92</td><td>42.44</td><td>29.10</td><td>46.81</td><td>28.81</td><td>50.18</td><td>25.09</td><td>54.41</td><td>10.61</td><td>32.42</td></tr><tr><td>TransAgg [28]</td><td>30.24</td><td>51.91</td><td>34.45</td><td>53.97</td><td>38.40</td><td>59.51</td><td>38.79</td><td>69.58</td><td>11.06</td><td>34.75</td></tr><tr><td>CLIP4CIR [4]</td><td>39.46</td><td>64.55</td><td>44.41</td><td>65.26</td><td>47.48</td><td>70.98</td><td>45.37</td><td>78.47</td><td>10.35</td><td>47.17</td></tr><tr><td>BLIP4CIR+Bi [32]</td><td>42.09</td><td>67.33</td><td>41.76</td><td>64.28</td><td>46.61</td><td>70.32</td><td>42.36</td><td>75.48</td><td>10.09</td><td>45.17</td></tr><tr><td>SPRC [2] SPN4CIR [8]</td><td>49.18</td><td>72.43</td><td>55.64 57.70</td><td>73.89</td><td>59.35</td><td>78.58</td><td>55.39</td><td>84.26</td><td>21.17</td><td>50.08</td></tr><tr><td></td><td>50.57</td><td>74.12</td><td></td><td>75.27</td><td>60.84</td><td>79.96</td><td>56.47</td><td>85.29</td><td>21.78</td><td>55.83</td></tr><tr><td colspan="9">Round 3</td><td></td></tr><tr><td>Pic2Word [39]</td><td>29.65</td><td>55.08</td><td>39.99</td><td>59.76</td><td>39.16</td><td>61.75</td><td>40.28</td><td>71.42</td><td>12.93</td><td>64.67</td></tr><tr><td>Context-I2W [44]</td><td>39.86</td><td>64.20</td><td>49.71</td><td>69.28</td><td>51.35</td><td>73.79</td><td>43.05</td><td>75.27</td><td>21.14</td><td>50.25</td></tr><tr><td>LinCIR [10]</td><td>33.66</td><td>59.94</td><td>51.03</td><td>69.92</td><td>48.24</td><td>68.59</td><td>43.03</td><td>75.68</td><td>17.64</td><td>51.08</td></tr><tr><td>TransAgg [28]</td><td>48.79</td><td>72.88</td><td>60.06</td><td>79.34</td><td>65.73</td><td>83.53</td><td>61.83</td><td>89.38</td><td>27.22</td><td>77.67</td></tr><tr><td>CLIP4CIR [4]</td><td>53.74</td><td>77.99</td><td>65.95</td><td>85.18</td><td>68.08</td><td>87.10</td><td>67.54</td><td>95.72</td><td>23.57</td><td>74.25</td></tr><tr><td>BLIP4CIR+Bi [32]</td><td>59.54</td><td>81.95</td><td>57.56</td><td>79.49</td><td>64.41</td><td>83.02</td><td>54.99</td><td>89.48</td><td>17.70</td><td>70.58</td></tr><tr><td>SPRC [2]</td><td>67.38</td><td>85.52</td><td>77.13 80.37</td><td>90.43</td><td>78.74</td><td>91.74</td><td>81.37</td><td>97.32</td><td>36.86</td><td>77.08</td></tr><tr><td>SPN4CIR [8]</td><td>70.70</td><td>88.00</td><td></td><td>92.35</td><td>82.51</td><td>93.01</td><td>84.31</td><td>97.66</td><td>38.02</td><td>81.00</td></tr><tr><td colspan="10">Round 5</td></tr><tr><td>Pic2Word [39]</td><td>33.12</td><td>59.05</td><td>43.42</td><td>64.13</td><td>42.22</td><td>65.37</td><td>57.47</td><td>75.60</td><td>17.19</td><td>83.50</td></tr><tr><td>Context-I2W [44]</td><td>44.42</td><td>68.12</td><td>55.69</td><td>74.19</td><td>56.81</td><td>78.68</td><td>56.83</td><td>78.71</td><td>26.89</td><td>69.08</td></tr><tr><td>LinCIR [10]</td><td>37.04</td><td>62.96</td><td>56.43</td><td>74.93</td><td>51.91</td><td>72.41</td><td>58.48</td><td>79.14</td><td>21.51</td><td>68.92</td></tr><tr><td>TransAgg [28]</td><td>53.59</td><td>76.85</td><td>67.12</td><td>84.25</td><td>73.33</td><td>89.14</td><td>78.04</td><td>92.54</td><td>32.26</td><td>88.33</td></tr><tr><td>CLIP4CIR [4]</td><td>57.86</td><td>81.51</td><td>70.71</td><td>88.42</td><td>72.67</td><td>89.95</td><td>82.92</td><td>97.06</td><td>29.32</td><td>82.08</td></tr><tr><td>BLIP4CIR+Bi [32]</td><td>63.41</td><td>84.28</td><td>61.83</td><td>82.09</td><td>67.87</td><td>86.18</td><td>67.11</td><td>91.32</td><td>20.71</td><td>80.42</td></tr><tr><td>SPRC [2]</td><td>70.40</td><td>87.70</td><td>80.72</td><td>92.54</td><td>83.32</td><td>93.98</td><td>89.76</td><td>98.49</td><td>41.93</td><td>84.92</td></tr><tr><td>SPN4CIR [8]</td><td>75.21</td><td>90.83</td><td>84.45</td><td>94.80</td><td>87.25</td><td>95.72</td><td>91.32</td><td>98.76</td><td>40.79</td><td>86.92</td></tr></table>

# 4. Experiments

# 4.1. Experimental Setups

Datasets and Evaluation Metric. We evaluate various CIR models on four benchmarks, including three public benchmarks: CIRR [31], FashionIQ [50], CIRCO [3], and our own proposed FISD benchmark. For CIRR and CIRCO, we conduct evaluations on the validation set because the multiround setting requires the involvement of the target image, and the ground truth for the test set is not accessible. In the context of multi-round evaluations, we primarily adopt the standard metrics in multi-round retrieval, i.e., Hits $@ \mathrm { K }$ . where success is defined as the target image appearing in the top $K$ results in any round up to the current one. Moreover, we include the rank of the target image as an evaluation metric. Additional experiments involving Recall $@ \mathrm { K }$ metrics under the multi-round setting are included in Section D of the supplementary material.

Implementation Details. The evaluation process on the FISD is meticulously designed to distinguish among various semantic subsets, each of which constitutes an independent image database comprising 600 images per subset. Our multi-round evaluation framework is implemented with PyTorch and is compatible with off-the-shelf CIR models such as Pic2Word [39], LinCIR [10], and SPRC [2].

These models are evaluated using their official checkpoints. By default, we utilize llama3-llava-next-8b [27] and Meta-Llama3-8B-Instruct [6] as MLLM and LLM respectively to construct our user simulator. The maximum number of rounds, denoted as $R _ { m a x }$ , is set to 5. Experiments detailed on Section 4.3 are primarily conducted using the SPRC model. The entire process is implemented on an Nvidia A100 GPU with 80GB of memory. For more implementation details, please refer to Section C of the supplementary material.

# 4.2. Main Results

Evaluation on FISD Benchmark. We evaluate various CIR models across diverse backbones, training data, and methodologies using the FISD benchmark. As shown in Table 1, we can draw the following conclusions: (i) Current CIR models exhibit notable deficiencies in several semantic aspects, particularly in processing negation and cardinality semantics. For instance, regarding the semantics of negation, even the best-performing model in our evaluation achieves only a Recal $@ 1$ of 17.5. This may be attributed to the inherent limitations of the feature extraction backbone, which performs inadequately in these semantic aspects [36, 41].(ii) Current CIR models perform relatively well at the more "direct" semantic aspect. When dealing with semantics such as "addition" and "change", the relative caption often explicitly identifies the elements present in the target image. In these scenarios, existing models perform slightly better. For example, SPN4CIR achieves a recall $@ 1$ of 77 on the change subset. (iii) Current CIR models have room for improvement across various semantic dimensions. Despite the provision of precise details, the challenge of understanding certain semantics remains unresolved in current CIR methods. For example, the best-performing model we evaluated achieved an average Recall $@ 1$ of only 55.83, highlighting the need for further enhancement.

Multi-round Evaluation Results. To assess the performance of existing CIR models in a multi-round setting, we utilize our proposed automated multi-round evaluation framework for eight different publicly available CIR models. These models encompass a variety of training data and strategies. The experimental results are presented in Table 2. We can draw the following observations: (i) Multiround interactions can significantly enhance performance. The experimental results demonstrate that regardless of the initial performance of the CIR models, they achieve significant improvements in performance across all four benchmarks in a multi-round setting. For instance, with the Pic2Word model on the CIRR dataset, the Hits $@ 1$ in the first round is merely 23.25, but after three rounds of interaction, the result improves to 40.28, an increase of approximately $7 3 . 2 5 \%$ .Similarly, for SPN4CIR on the CIRR dataset, the single-round $\mathrm { H i t s } @ 5$ is 85.29, but after fiveround processing, the Hits $\textcircled { a } 5$ soars to 98.76. (ii) The effectiveness of a CIR model in a multi-round setting is closely linked to its initial performance. Generally, models with strong initial performance tend to achieve better outcomes following multiple rounds of interaction. (iii) The performance gains achieved through multi-round interactions tend to diminish as the number of interactions increases. Specifically, the improvement observed after three rounds is greater than the improvement seen when progressing from three to five rounds. For example, with the SPN4CIR on the FashionIQ-Dress, the Hits $@ 1 0$ improved by 20.23 points from the first to the third round, whereas it increased by only 4.51 points from the third to the fifth round.

# 4.3. Further Analysis

Analysis of Multiple Rounds. As shown in Figure 4, we investigate the changes in retrieval performance across different maximum rounds on FashionIQ. The results indicate that performance consistently improves as the number of rounds increases, demonstrating the effectiveness of current CIR models in enhancing performance over multiple iterations. Notably, the performance gains are more pronounced at lower rounds. As the number of rounds exceeds 5, the performance improvements gradually plateau. Therefore, we set the default maximum rounds $R _ { \mathrm { m a x } }$ to 5.

![](images/4.jpg)  
Figure 4. Performance across various rounds on FashionIQ.

Table 3. Performance of various MLLMs and LLMs on CIRR.   

<table><tr><td>MLLM</td><td>LLM</td><td>Hits@1</td><td>Hits@5</td></tr><tr><td>BLIP-2</td><td>Llama3-8B</td><td>85.82</td><td>97.54</td></tr><tr><td>LLaVA-1.5</td><td>Llama3-8B</td><td>89.62</td><td>98.37</td></tr><tr><td>LLaVA-Next</td><td>Llama3-8B</td><td>89.76</td><td>98.49</td></tr><tr><td>LLaVA-Next</td><td>Llama2-7B</td><td>78.43</td><td>94.07</td></tr><tr><td>LLaVA-Next</td><td>Mistral-7B</td><td>89.45</td><td>98.64</td></tr><tr><td>LLaVA-Next</td><td>Llama3-8B</td><td>89.76</td><td>98.49</td></tr></table>

Impact of MLLM&LLM on Evaluation Performance. In this part, we examine the impact of utilizing various MLLMs and LLMs as user simulators on the performance of multi-round evaluations based on a particular CIR model, SPRC. We utilize BLIP-2 [21], LLaVA-1.5 [26], and LLaVA-Next [27] as MLLMs, alongside Llama2-7B [46], Mistral-7B [13], and Llama3-8B [6] as LLMs. The experimental results are shown in Table 3. We can make the following observations: (i) Stronger MLLMs yield better outcomes. Specifically, BLIP-2 tends to generate coarse captions, whereas captions produced by LLaVA-1.5 and LLaVA-Next are more detailed, leading to superior performance. (ii) Models with stronger instruction-following capabilities are more suitable for use as user simulators. There are noticeable improvements with Mistral-7B and Llama3-8B over Llama2-7B, likely due to Llama2-7B's weaker instruction-following capabilities that often result in irrelevant content and thus mislead the model. Conversely, Mistral-7B and Llama3-8B show robust instruction-following abilities, and there is no significant difference in performance between them. (iii) Our evaluation framework leads to a consistent improvement using different combinations of MLLMs and LLMs. Despite some variations in the performance caused by different combinations of MLLMs and LLMs, there is a significant improvement compared to single-round results.

Different Simulation Methd. As MLLMs advance, they can now directly identify differences between images. Leveraging this functionality, we employ Qwen2- VL-7B [48] to directly generate the difference between two images as our user simulation. As shown in Table 4, this approach also achieves significant performance improvements. Compared to using LLMs to infer differences from captions, this simulation method may be more suitable when captions are missing or of low quality, though it incurs slightly higher inference costs.

Table 4. Comparison of various simulation methods on CIRR. R1 represents Round 1.   

<table><tr><td>Method</td><td>R1/(Hits@1↑/Rank↓) R3/(Hits@1↑/Rank↓) R5/(Hits@1↑/Rank↓)</td><td></td><td></td></tr><tr><td>Qwen2VL</td><td>55.39 / 6.21</td><td>80.80 / 2.01</td><td>90.00 / 1.53</td></tr><tr><td>Llama3</td><td>55.39 / 6.21</td><td>81.37 / 1.79</td><td>89.76 / 1.45</td></tr></table>

Table 5. Performance of real users and our user simulator. Our user simulator aligns with the performance of the real users.   
Table 6. Ablation studies on CIRR Benchmark based on SPRC model. Bold indicates the best result.   

<table><tr><td>Method</td><td>Dataset</td><td>Initial Rank</td><td>Final Rank</td></tr><tr><td>Human</td><td>CIRR</td><td>189.98</td><td>12.34 ↓177.64</td></tr><tr><td>Human</td><td>FashionIQ</td><td>2427.12</td><td>447.96↓1979.16</td></tr><tr><td>Human</td><td>FISD</td><td>9.62</td><td>2.92 ↓6.7</td></tr><tr><td>Simulator</td><td>CIRR</td><td>189.98</td><td>4.12↓185.86</td></tr><tr><td>Simulator</td><td>FashionIQ</td><td>2427.12</td><td>370.78↓2056.34</td></tr><tr><td>Simulator</td><td>FISD</td><td>9.62</td><td>4.16 ↓5.46</td></tr></table>

<table><tr><td>Method</td><td>Round 1 / Rank↓</td><td>Round 3 / Rank↓</td><td>Round 5 / Rank↓</td></tr><tr><td>No History</td><td>6.21</td><td>2.93</td><td>2.39</td></tr><tr><td>Cap. Unchanged</td><td>6.21</td><td>8.56</td><td>9.64</td></tr><tr><td>Top-10 random</td><td>6.21</td><td>1.86</td><td>1.70</td></tr><tr><td>Ours</td><td>6.21</td><td>1.79</td><td>1.45</td></tr></table>

User Study. To assess the multi-round performance of the CIR model in real-world application scenarios, we conduct user studies across three benchmarks. Specifically, we select 50 samples that exhibit poor performance in singleround CIR models from each benchmark. These samples undergo 5 interaction rounds with both real users and our user simulator. As illustrated in Table 5, it is evident that the ranking of the target images significantly decreases, regardless of whether feedback is provided by a user or the simulator. For instance, the ranking of the target image falls by approximately 2000 positions after 5 interaction rounds using either method. This underscores the versatility and practicality of our framework. It is noteworthy that, at times, user performance is inferior to that of the user simulator. This arises because users tend to provide short feedback, whereas simulators are prone to offer more detailed feedback, as discussed in Section E of the supplementary materials. The nature of feedback difference inherently makes some impact on the effectiveness of interaction. Overall, our user simulator closely aligns with the performance of real users, reflecting actual evaluation conditions.

Ablation Studies of our Evaluation Framework. In this section, we conduct ablation studies on CIRR to evaluate three key designs: (i) The effect of history information. (ii) The role of varied feedback. (iii) The rationale for selecting the top-1 candidate images as the next round's reference. The experimental results in Table 6 lead to the following conclusions: (i) Using candidate image feature at round $r$ directly (without averaging with previous rounds) degrades performance, confirming the effectiveness of historical information. (ii) Reusing the same relative caption across varying candidate images significantly harms performance, demonstrating the importance of diverse feedback. (iii) Randomly selecting a candidate image from the top-10 yields negligible performance differences, justifying the use of the top-1 candidate for the next round.

Can FISD Accurately Reflect the CIR Model's Capability? Since the images in the FISD benchmark are all synthetic, there remains a gap between these and natural images. This raises the question: can FISD fully capture the performance of CIR models? As shown in Table 1, CIR models perform poorly in handling negation semantics. To determine whether this issue is caused by the gap between synthetic and natural images, we use Llama3-8B to select and manually verify queries with negation semantics from the CIRR validation set. The experimental results, presented in Table 1 of the supplementary material, reveal that even with natural images, CIR models struggle with negation compared to other semantic aspects. This is consistent with our findings on FISD. Furthermore, the performance trends across different CIR models on the FISD align with those on CIRR and FashionIQ. Therefore, we believe that FISD can accurately reflect the capabilities of CIR models.

# 5. Conclusion and Future Work

In this paper, we have illuminated the presence of numerous indeterminate queries within the existing CIR benchmark, which significantly impairs the rigorous evaluation of CIR model performance. To thoroughly assess the capabilities of current CIR models, we introduced FISD, a novel Fully-Informed Semantic-Diverse benchmark tailored for CIR. The experimental results indicate that the current CIR models are inadequate for handling complex composed semantics, particularly in terms of negation and cardinality. Furthermore, we proposed an automated multi-round evaluation framework to assess how CIR models adapt and refine their choices over successive rounds of queries. Extensive experiments and user studies have confirmed the significant improvements achieved by current CIR models in a multiround setting. We aspire that our proposed benchmark and evaluation framework will provide valuable insights and inspire further research into the CIR problem.

In terms of future work, the current amount of data in our FISD benchmark remains relatively limited. One of the challenges that need to be addressed is how to scale this benchmark. Presently, our multi-round evaluation framework conducts assessments solely on existing CIR models. Therefore, how to develop strong multi-round CIR models is another critical issue that needs to be addressed.

# References

[1] Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In ICCV, 2019. 11   
[2] Yang Bai, Xinxing Xu, Yong Liu, Salman Khan, Fahad Khan, Wangmeng Zuo, Rick Siow Mong Goh, and Chun-Mei Feng. Sentence-level prompts benefit composed image retrieval. In ICLR, 2024. 2, 3, 4, 6, 14   
[3] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and Alberto Del Bimbo. Zero-shot composed image retrieval with textual inversion. In ICCV, 2023. 2, 6   
[4] Alberto Baldrati, Marco Bertini, Tiberio Uricchio, and Alberto Del Bimbo. Composed image retrieval using contrastive learning and task-oriented clip-based features. ACM Transactions on Multimedia Computing, Communications and Applications, 2023. 2, 3, 4, 6, 14   
[5] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 11   
[6] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. 6, 7   
[7] Chengjian Feng, Yujie Zhong, Zequn Jie, Weidi Xie, and Lin Ma. Instagen: Enhancing object detection by training on synthetic dataset. In CVPR, 2024. 3   
[8] Zhangchi Feng, Richong Zhang, and Zhijie Nie. Improving composed image retrieval via contrastive learning with scaling positives and negatives. arXiv preprint arXiv:2404.11317, 2024. 4, 6, 14   
[9] Geonmo Gu, Sanghyuk Chun, Wonjae Kim, HeeJae Jun, Yoohoon Kang, and Sangdoo Yun. Compodiff: Versatile composed image retrieval with latent diffusion. arXiv preprint arXiv:2303.11916, 2023. 2   
[10] Geonmo Gu, Sanghyuk Chun, Wonjae Kim, Yoohoon Kang, and Sangdoo Yun. Language-only efficient training of zeroshot composed image retrieval. In CVPR, 2024. 2, 4, 6, 14   
[11] Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing. arXiv preprint arXiv:2404.09990, 2024. 11   
[12] Young Kyun Jang, Donghyun Kim, Zihang Meng, Dat Huynh, and Ser-Nam Lim. Visual delta generator with large multi-modal models for semi-supervised composed image retrieval. In CVPR, 2024. 2   
[13] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023. 7   
[14] Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024. 4, 11 [15] Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, and Zeynep Akata. Vision-by-language for trainingfree compositional image retrieval. arXiv preprint arXiv:2310.09291, 2023. 4 [16] Adriana Kovashka and Kristen Grauman. Attributes for image retrieval. Visual Attributes, pages 89117, 2017. 3 [17] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. In CVPR, 2024. 3 [18] Matan Levy, Rami Ben-Ari, Nir Darshan, and Dani Lischinski. Chatting makes perfect: Chat-based image retrieval. In NeurIPS, 2023. 2 [19] Matan Levy, Rami Ben-Ari, Nir Darshan, and Dani Lischinski. Data roaming and quality assessment for composed image retrieval. In AAAI, 2024. 2 [20] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,   
2022. 2 [21] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,   
2023. 2, 7 [22] You Li, Fan Ma, and Yi Yang. Imagine and seek: Improving composed image retrieval with an imagined proxy. In CVPR,   
2025.2 [23] Kaiqu Liang and Samuel Albanie. Simple baselines for interactive video retrieval with questions and answers. In ICCV,   
2023.2 [24] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740755. Springer, 2014. 4 [25] Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm-open-ended visual storytelling via latent diffusion models. In CVPR, 2024.   
3 [26] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 3, 7 [27] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 6, 7 [28] Yikun Liu, Jiangchao Yao, Ya Zhang, Yanfeng Wang, and Weidi Xie. Zero-shot composed text-image retrieval. In BMVC, 2023. 2, 4, 6, 14 [29] Yikun Liu, Pingan Chen, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. Lamra: Large multimodal model as your advanced retrieval assistant. In CVPR, 2025. 3 [30] Yikun Liu, Yuan Liu, Shangzhe Di, Haicheng Wang, Zhongyin Zhao, Le Tian, Xiao Zhou, Jie Zhou, Jiangchao Yao, Yanfeng Wang, et al. Versavit: Enhancing mllm vision backbones via task-guided optimization. arXiv preprint arXiv:2602 09934 2026 3   
[31] Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, and Stephen Gould. Image retrieval on real-life images with pretrained vision-and-language models. In ICCV, 2021. 2, 3, 6, 12   
[32] Zheyuan Liu, Weixuan Sun, Yicong Hong, Damien Teney, and Stephen Gould. Bi-directional training for composed image retrieval via text prompt learning. In WACV, 2024. 2, 4, 6, 14   
[33] Avinash Madasu, Junier Oliva, and Gedas Bertasius. Learning to retrieve videos by asking questions. In ACM MM, 2022. 3   
[34] Sho Maeoki, Kohei Uehara, and Tatsuya Harada. Interactive video retrieval with dialog. In CVPR Workshops, 2020. 3   
[35] Yanxu Meng, Haoning Wu, Ya Zhang, and Weidi Xie. Scenegen: Single-image 3d scene generation in one feedforward pass. In 3DV, 2025. 3   
[36] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. In ICCV, 2023. 6   
[37] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. 4   
[38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In ICML, 2021. 2   
[39] Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister. Pic2word: Mapping pictures to words or zero-shot composediage retrieval. In CVPR, 2023. 2, 4, 6, 14   
[40] Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In ACL, 2018. 4, 11   
[41] Jaisidh Singh, Ishaan Shrivastava, Mayank Vatsa, Richa Singh, and Aparna Bharati. Learn" no" to say" yes" better: Improving vision-language models via negations. arXiv preprint arXiv:2403.20312, 2024. 6   
[42] Zelong Sun, Dong Jing, Guoxing Yang, Nanyi Fei, and Zhiwu Lu. Leveraging large vision-language model as user intent-aware encoder for composed image retrieval. In AAAI, 2025. 2   
[43] Yucheng Suo, Fan Ma, Linchao Zhu, and Yi Yang. Knowledge-enhanced dual-stream zero-shot composed image retrieval. In CVPR, 2024. 2   
[44] Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Yue Hu, and Qi Wu. Context-i2w: Mapping images to context-dependent words for accurate zero-shot composed image retrieval. In AAAI, 2024. 2, 4, 6, 14   
[45] Yuanmin Tang, Xiaoting Qin, Jue Zhang, Jing Yu, Gaopeng Gou, Gang Xiong, Qingwei Ling, Saravan Rajmohan, Dongmei Zhang, and Qi Wu. Reason-before-retrieve: One-stage reflective chain-of-thoughts for training-free zero-shot composed image retrieval. In CVPR, 2025. 2   
[40] Hugo 1ouvron, Louis Martin, Kevin Stone, Peter AiDert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 7   
[47] Lucas Ventura, Antoine Yang, Cordelia Schmid, and Güil Varol. Covr: Learning composed video retrieval from web video captions. In AAAI, 2024. 4   
[48] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 7   
[49] Haokun Wen, Xian Zhang, Xuemeng Song, Yinwei Wei, and Liqiang Nie. Target-guided composed image retrieval. In ACM MM, 2023. 4   
[50] Hui Wu, Yupeng Gao, Xiaoxiao Guo, Ziad Al-Halah, Steven Rennie, Kristen Grauman, and Rogerio Feris. Fashion iq: A new dataset towards retrieving images by natural language feedback. In CVPR, 2021. 2, 3, 5, 6   
[51] Haoning Wu, Shaocheng Shen, Qiang Hu, Xiaoyun Zhang, Ya Zhang, and Yanfeng Wang. Megafusion: Extend diffusion models towards higher-resolution image generation without further tuning. In WACV, 2025. 3   
[52] Haoning Wu, Ziheng Zhao, Ya Zhang, Yanfeng Wang, and Weidi Xie. Mrgen: Segmentation data engine for underrepresented mri modalities. In ICCV, 2025. 3   
[53] Zhenyu Yang, Dizhan Xue, Shengsheng Qian, Weiming Dong, and Changsheng Xu. Ldre: Llm-based divergent reasoning and ensemble for zero-shot composed image retrieval. In SIGIR, 2024. 2   
[54] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. In TACL, 2014. 4, 11   
[55] Fei Zhang, Zijian Zhou, Bohao Tang, Sen He, Hang Li, Zhe Wang, Soubhik Sanyal, Pengfei Liu, Viktar Atliha, Tao Xiang, et al. Transtext: Transparency aware image-to-video typography animation. arXiv preprint arXiv:2603.17944, 2026. 3   
[56] Kai Zhang, Yi Luan, Hexiang Hu, Kenton Lee, Siyuan Qiao, Wenhu Chen, Yu Su, and Ming-Wei Chang. Magiclens: Selfsupervised image retrieval with open-ended instructions. In ICML, 2024. 2   
[57] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS, 2023. 3

# A Sanity Check on Composed Image Retrieval

Supplementary Material

# A. Details of Generating Different Semantic Subsets

Aditon&Negation&Change&BackgroundThe erati detail heourmantiubsetslredycov tn F  [, while for Background, we draw from the validation set of the CC3M [40] dataset.

Given the image caption for the reference image, we prompt Mixtral- ${ \bf \nabla } \cdot 8 { \bf x } 7 { \bf B }$ -Instruct-v0.1 [14] to simultaneously generate a relative caption and the caption of the target image with the following prompt:

I need you  perorm ne reaonableediting step n an age. I will providehe caption original image, and I need you to speciy the following instruction and the caption of edited g.The emanticaspect o theinstruction you generat is {. Referenceimage captio{}

while the second placeholder refers to the reference image caption.

Cardinality. We adopt "a real-life image of $\{ n u m \} \ \{ n o u n  \}$ ." as a template caption, where num is chosen from ten numbers rfrom  t 0,an noun s elc from he jec cateory  CCOand ImageNet [.Then, wep e o u    e respective quantity.

![](images/5.jpg)  
Figure 5. The template for cardinality relative captions.

Cex.Welize H-dit [1 he  ur r hee ubse Q-Ed eve T-  roi-more than 25 words as our caption source.

# B. Prompts of Our Multi-round Evaluation Pipeline

Propts for MLLM.Wus MLLM ndLLM ulaeustec.pealy wes prot MLLM t pt r ocandat n targaeorRR IRCOand IS wepl  prop:ive

prompt: Give a short and precise English description of the clothes.

Prot r M.ethetns  rot LL elaiv between the candidate and target images. For CIRR and CIRCO, we use the prompt as follows:

I will give you two sentences, one of which is reference image caption and one is target image caption. I need you to give me an instruction to transition from reference image to target image. The type of instruction you give can be cardinality, addition, negation, direct addressing, compare&change, comparative, conjunction, spatial relation&background, viewpoint. The instruction you give does not need to have additional information, be detailed and highlight the key points. The instruction you generate is about 30 words. Reference Image Caption: {1}. Target Image Caption: $\{ 2 \}$ . Instruction:

For F   t FashionIQ dataset as mentioned in [31].

I will give you two sentences, one of which is reference image caption and one is target image caption. I need you to give me an instruction to transition from reference image to target image. The type of instruction you give can be addition, negation, direct addressing, compare&change, comparative, conjunction. The instruction you give does not need to have additional information, be brief and highlight the key points. The instruction you generate is about 10 words. Reference Image Caption: $\{ 1 \}$ . Target Image Caption: $\{ 2 \}$ . Instruction:

FFI the third one represents the target image caption.

I will give you two sentences, one of which is reference image caption and one is target image caption. I need you to give me an instruction to transition from reference image to target image. The type of instruction you give can only be {1} and can not contain other semantics. The instruction you give does not need to have additional information, be detailed and highlight the key points. The instruction you generate is about 10 words. Reference Image Caption: {2}. Target Image Caption: {3}. Instruction:

Regarding th complex instructon smantic aspec  theFSD bencmark, whic may encopass diver snc instructions, we do not limit the semantic type. The prompt is as follows:

I will give you two sentences, one of which is reference image caption and one is target image caption. I need you to give me an instruction to transition from reference image to target image. The instruction you give does not need to have additional information, be detailed and highlight the key points. The instruction you generate is about 30 words. Reference Image Caption: $\{ 1 \}$ . Target Image Caption: $\{ 2 \}$ . Instruction:

# C. Evaluation Details

Wu  h t e RRRCOn FIS bcar h ecohic av bee IRRdataseThisi becuse CIRR,CIRCO, andFIS ncpass real-imagescarsonversely  he evaluation on the FashioIQdatase, weemploy the official checkpoint specifically ne-tuned on the FashionQ.

# D. Detailed Experimental Results D.1. Additional Validation Experiments for FISD

Table 7. Comparison of performance on negation and overall semantics in the CIRR validation set.   

<table><tr><td rowspan="2">Method</td><td>Negation</td><td>All</td></tr><tr><td>R@1</td><td>R@1</td></tr><tr><td>Pic2Word</td><td>17.81</td><td>23.25</td></tr><tr><td>Context-I2W</td><td>20.55</td><td>26.96</td></tr><tr><td>SPRC</td><td>31.51</td><td>55.39</td></tr></table>

As shown in Section 4. of the main paper, CIR models perform poorly in handling negation semantics on FISD. T mlly    a  l petalu  r Ta . This is consistent with our findings on FISD.

# D.2. Additional Multi-round Evaluation Results

In this section, we present the multi-round evaluation results of various CIR models using the Recall $@ \mathrm { K }$ metric. The experimental results are shown in Table 8.

# E. Detailed Analysis of User Study

Anly  User Sud Experetal Resultss how Sct   themai pape i the valutihR anashnrku  rho details, whereas simulators provide more comprehensive and detailed feedback.

I e U y interaction session. We offer users a wage of $\$ 15$ per hour. Note that, our user study is conducted internally within the laboratory, with no potential risks. Additionally, we do not store any feedback provided by the users.

![](images/6.jpg)  
Figure 6. Some examples of feedback from both users and user simulators.

Table 8. Multi-round evaluation on various state-of-the-art CIR models across a range of benchmarks.   

<table><tr><td rowspan="2">Method</td><td colspan="2">FashionIQ-Dress</td><td colspan="2">FashionIQ-Shirt</td><td colspan="2">FashionIQ-Toptee</td><td colspan="2">CIRR</td><td>FISD</td></tr><tr><td>Recall@10</td><td>Recall@50</td><td>Recall@10</td><td>Recall@50</td><td>Recall@10</td><td>Recall@50</td><td>Recall@1</td><td>Recall@5</td><td>Recall@1</td></tr><tr><td colspan="10">Round 1</td></tr><tr><td>Pic2Word [39]</td><td>20.00</td><td>40.20</td><td>26.20</td><td>43.60</td><td>27.90</td><td>47.40</td><td>23.25</td><td>51.42</td><td>21.33</td></tr><tr><td>Context-I2W [44]</td><td>23.10</td><td>45.30</td><td>29.70</td><td>48.60</td><td>30.60</td><td>52.90</td><td>26.96</td><td>56.59</td><td>27.83</td></tr><tr><td>LinCIR [10]</td><td>20.92</td><td>42.44</td><td>29.10</td><td>46.81</td><td>28.81</td><td>50.18</td><td>25.09</td><td>54.41</td><td>32.42</td></tr><tr><td>TransAgg [28]</td><td>30.24</td><td>51.91</td><td>34.45</td><td>53.97</td><td>38.40</td><td>59.51</td><td>38.79</td><td>69.58</td><td>34.75</td></tr><tr><td>CLIP4CIR [4]</td><td>39.46</td><td>64.55</td><td>44.41</td><td>65.26</td><td>47.48</td><td>70.98</td><td>45.37</td><td>78.47</td><td>47.17</td></tr><tr><td>BLIP4CIR+Bi [32]</td><td>42.09</td><td>67.33</td><td>41.76</td><td>64.28</td><td>46.61</td><td>70.32</td><td>42.36</td><td>75.48</td><td>45.17</td></tr><tr><td>SPRC [2] SPN4CIR [8]</td><td>49.18</td><td>72.43</td><td>55.64</td><td>73.89</td><td>59.35</td><td>78.58</td><td>55.39</td><td>84.26</td><td>50.08</td></tr><tr><td></td><td>50.57</td><td>74.12</td><td>57.70</td><td>75.27</td><td>60.84</td><td>79.96</td><td>56.47</td><td>85.29</td><td>55.83</td></tr><tr><td colspan="10">Round 3</td></tr><tr><td>Pic2Word [39]</td><td>21.81</td><td>44.42</td><td>33.12</td><td>50.93</td><td>31.72</td><td>51.50</td><td>40.28</td><td>66.01</td><td>64.58</td></tr><tr><td>Context-I2W [44]</td><td>30.84</td><td>54.69</td><td>43.62</td><td>62.66</td><td>44.97</td><td>65.83</td><td>43.00</td><td>69.31</td><td>50.08</td></tr><tr><td>LinCIR [10]</td><td>24.94</td><td>48.34</td><td>45.04</td><td>62.41</td><td>39.32</td><td>60.38</td><td>42.86</td><td>70.65</td><td>50.67</td></tr><tr><td>TransAgg [28]</td><td>40.65</td><td>64.95</td><td>55.84</td><td>74.48</td><td>60.12</td><td>78.79</td><td>61.83</td><td>85.77</td><td>77.67</td></tr><tr><td>CLIP4CIR [4]</td><td>46.16</td><td>69.71</td><td>59.52</td><td>79.39</td><td>61.35</td><td>80.62</td><td>67.26</td><td>93.95</td><td>74.08</td></tr><tr><td>BLIP4CIR+Bi [32]</td><td>49.78</td><td>73.23</td><td>49.36</td><td>71.54</td><td>55.69</td><td>76.03</td><td>52.31</td><td>84.62</td><td>69.67</td></tr><tr><td>SPRC [2]</td><td>57.91</td><td>78.78</td><td>71.05</td><td>85.92</td><td>71.49</td><td>87.35</td><td>80.63</td><td>96.22</td><td>77.08</td></tr><tr><td>SPN4CIR [8]</td><td>63.16</td><td>82.85</td><td>74.39</td><td>88.52</td><td>75.57</td><td>89.39</td><td>83.11</td><td>96.48</td><td>79.33</td></tr><tr><td colspan="10">Round 5</td></tr><tr><td>Pic2Word [39]</td><td>22.71</td><td>43.18</td><td>34.79</td><td>51.52</td><td>32.02</td><td>51.30</td><td>57.43</td><td>68.93</td><td>83.42</td></tr><tr><td>Context-I2W [44]</td><td>31.73</td><td>54.39</td><td>46.61</td><td>65.60</td><td>47.42</td><td>68.18</td><td>56.66</td><td>71.30</td><td>68.92</td></tr><tr><td>LinCIR [10]</td><td>25.43</td><td>46.75</td><td>46.61</td><td>65.01</td><td>40.03</td><td>60.28</td><td>58.19</td><td>72.95</td><td>68.42</td></tr><tr><td></td><td>42.69</td><td>64.90</td><td>60.99</td><td>78.56</td><td>66.29</td><td>84.14</td><td>78.02</td><td>88.23</td><td>88.33</td></tr><tr><td>TransAgg [28] CLIP4CIR [4]</td><td>46.11</td><td>70.15</td><td>62.22</td><td>81.11</td><td>62.06</td><td>82.20</td><td>82.35</td><td>94.91</td><td>81.83</td></tr><tr><td>BLIP4CIR+Bi [32]</td><td>49.98</td><td>71.99</td><td>51.37</td><td>71.59</td><td>56.45</td><td>76.29</td><td>61.97</td><td>85.22</td><td>78.83</td></tr><tr><td>SPRC [2]</td><td>57.11</td><td>76.80</td><td>73.21</td><td>86.90</td><td>73.28</td><td>87.61</td><td>88.38</td><td>97.46</td><td>83.33</td></tr><tr><td>SPN4CIR [8]</td><td>64.20</td><td>83.74</td><td>77.53</td><td>91.41</td><td>79.30</td><td>91.59</td><td>89.26</td><td>97.54</td><td>84.92</td></tr></table>

# F. Additional Failure cases in Current Public Benchmark

I o   e between the composed query and the target image.

# G. Additional Examples of Our Proposed Benchmark

I dispy dataypeardinalidi egatincaebacronancpsrtiepevy.

# H. Qualitative Analysis

i tou uns.Theexame dmnstrateho urul-un stecenta prache h tare  y .   n bothproving the IRmodel'overall performanceand (i) developingcleaner, ore standardize bencmarks.

# I. Discussion on Inference Cost

A u poo based on the first-round search, which can also exponentially reduce the time cost in the search space.

![](images/7.jpg)  
image with red borders.

![](images/8.jpg)  
image with red borders.

![](images/9.jpg)

![](images/10.jpg)

Change the color of the jay from red to blue.

![](images/11.jpg)

Change the basketball to a soccer ball.

4

Change the pattern of the woman's shirt from solid to striped.

![](images/12.jpg)  
Replace the watermelon juice with pineapple juice

() Change

Change the background to a bench scene.

![](images/13.jpg)

Make the background more blurred to emphasize the red leaves

ET

Remove the busy background.

![](images/14.jpg)

Change the background to a desolate and barren landscape.

(f) Background

B

Transition the scene to nighttime, add a starry sky with a full moon, and illuminate the scene with warm artificial lighting from the buildings and street lamps.

![](images/15.jpg)

Change the setting to a winter scene with a red wooden house, replace the greenery with snow-covered pine trees, and dress the children in winter clothing.

X

Replace the sunflower with a cherry blossom tree with full pink blooms, adjust the background to include more cherry blossoms, and modify the sky to have a soft pink hue.

Transform the elderly gentleman into a young boy, change the clothing to a casual t-shirt with a graphic print, shorts, and sneakers. Replace the walking cane with boy's right hand resting on the bench.

(g) Complex

![](images/16.jpg)  
Figure 11. Qualitative results for two rounds.

![](images/17.jpg)  
Figure 12. Qualitative results for three rounds.

![](images/18.jpg)  
Figure 13. Qualitative results for four rounds.

![](images/19.jpg)  
Figure 14. Failure cases in multi-round CIR.