# 1. Bibliographic Information
## 1.1. Title
The paper's title is *Lightweight Prompt-Guided CLIP Adaptation for Monocular Depth Estimation*. Its central topic is developing a parameter-efficient framework to adapt the pre-trained CLIP vision-language model for the monocular depth estimation task, using lightweight Mixture-of-Adapters (MoA) modules and hybrid prediction designs to balance semantic knowledge transfer and geometric prediction precision.
## 1.2. Authors
1.  **Reyhaneh Ahani Manghotay**: Affiliated with the School of Engineering Science, Simon Fraser University (Burnaby, BC, Canada), with research focus on computer vision and vision-language model adaptation.
2.  **Jie Liang**: Corresponding author, affiliated with the School of Information Science and Technology, Eastern Institute of Technology (Ningbo, China), with research focus on 3D computer vision and efficient deep learning.
## 1.3. Journal/Conference
As of the current UTC date of 2026-04-02, the paper is hosted as a preprint on arXiv, and has not been officially published in a peer-reviewed journal or conference.
## 1.4. Publication Year
The preprint was released in April 2026, per the arXiv identifier `2604.01118`.
## 1.5. Abstract
This work addresses the challenge of transferring CLIP's high-level semantic knowledge to fine-grained geometric monocular depth estimation tasks, which previously required heavy full fine-tuning or suffered from low geometric precision. The proposed parameter-efficient framework MoA-DepthCLIP inserts lightweight MoA modules into the frozen CLIP ViT-B/32 backbone, combined with selective fine-tuning of the final backbone layers, a global semantic context vector derived from text prompts, and a hybrid prediction architecture that combines depth bin classification and direct regression. A composite loss function with geometric constraints is used to optimize the model. On the NYU Depth V2 benchmark, MoA-DepthCLIP outperforms the DepthCLIP baseline by a large margin: it improves the $\delta_1$ accuracy from 0.390 to 0.745, and reduces RMSE by 55% from 1.176 to 0.520, while requiring only a small fraction of trainable parameters compared to full fine-tuning approaches.
## 1.6. Original Source Link
- Preprint abstract page: https://arxiv.org/abs/2604.01118
- Full PDF link: https://arxiv.org/pdf/2604.01118.pdf
- Publication status: Unpublished preprint

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem and Importance
Monocular depth estimation (predicting per-pixel distance from the camera using a single RGB image) is a fundamental computer vision task that powers critical applications including autonomous navigation, robotics, augmented reality, and monocular 3D object detection. The task is inherently ill-posed, as a single 2D image can correspond to multiple valid 3D scene configurations.
### Gaps in Prior Research
1.  Traditional fully supervised depth estimation methods require large datasets with expensive dense depth annotations, which are costly to collect.
2.  Recent large foundation depth models achieve high accuracy but have billions of parameters and require massive training data, making them unsuitable for edge deployment.
3.  Prior vision-language model (VLM) based depth methods (e.g., DepthCLIP) enable zero-shot depth prediction by matching image regions to handcrafted text prompts, but produce coarse outputs with very low geometric precision, and rely on manual prompt design.
4.  Existing VLM adaptation strategies for depth either use heavy full fine-tuning or limited prompt learning, failing to balance parameter efficiency and prediction accuracy.
### Innovative Entry Point
The paper proposes to combine two previously separate paradigms: (1) Mixture-of-Adapters (MoA), a parameter-efficient fine-tuning (PEFT) method proven effective in NLP, and (2) hybrid classification-regression heads, a standard design for high-precision traditional depth estimation, to adapt CLIP to the depth task with minimal trainable parameters, while preserving CLIP's semantic knowledge and achieving high geometric precision.
## 2.2. Main Contributions / Findings
### Primary Contributions
1.  **Novel Framework**: MoA-DepthCLIP, the first MoA-based PEFT framework for monocular depth estimation, combining lightweight MoA module insertion, selective backbone fine-tuning, global text prompt-derived scene context fusion, and a hybrid classification-regression prediction head.
2.  **Technical Optimizations for Depth Task**:
    - Deterministic gating mechanism for MoA modules, designed for stable dense prediction instead of the stochastic routing used in NLP MoA methods.
    - Selective MoA placement at only 4 layers of the ViT backbone to minimize parameter overhead while balancing adaptation across feature levels.
    - Fixed 128 depth bin configuration, balancing accuracy and robustness without the computational cost of adaptive binning methods.
    - Composite loss function combining cross-entropy classification loss, L1 regression loss, and scale-invariant logarithmic (SILog) loss to enforce both coarse scene layout and fine-grained geometric accuracy.
### Key Findings
1.  The proposed framework outperforms the DepthCLIP baseline by a large margin: $\delta_1$ accuracy improves by 91% from 0.390 to 0.745, and RMSE is reduced by 55% from 1.176 to 0.520 on NYU Depth V2.
2.  Each component of the framework contributes incrementally to performance gains, with the composite loss and 128-bin discretization delivering the largest improvements.
3.  The framework achieves competitive performance while using only a small fraction of the trainable parameters required for full backbone fine-tuning, proving lightweight VLM adaptation is a viable strategy for dense geometric prediction tasks.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All core concepts are explained below for beginner readers:
1.  **Contrastive Language-Image Pre-training (CLIP)**: A large-scale VLM pre-trained on millions of image-text pairs, learning joint embeddings where semantically similar images and text are mapped to nearby vector space positions. It exhibits strong zero-shot performance on classification tasks, but requires adaptation for dense prediction tasks like depth estimation.
2.  **Monocular Depth Estimation**: A computer vision task that predicts a dense depth map (a 2D matrix where each value represents the distance between the corresponding image pixel and the camera) from a single RGB input image.
3.  **Parameter-Efficient Fine-Tuning (PEFT)**: A set of techniques that only train a small subset of parameters of a large pre-trained model (instead of full fine-tuning) to adapt it to downstream tasks. PEFT preserves pre-trained knowledge, reduces computational cost, and avoids catastrophic forgetting.
4.  **Vision Transformer (ViT)**: A transformer-based architecture for computer vision tasks that splits input images into fixed-size patches, encodes each patch as a token, and processes tokens using stacked transformer encoder layers. The paper uses the ViT-B/32 variant (base size, 32x32 patch size) from CLIP.
5.  **Mixture of Adapters (MoA)**: A PEFT method that inserts multiple small lightweight MLP modules (called experts) into pre-trained model layers, alongside a gating network that routes each input token to a weighted combination of experts, enabling token-level specialized adaptation with minimal parameter overhead.
6.  **Depth Binning**: A technique that discretizes continuous depth values into a fixed set of non-overlapping intervals (bins), converting the continuous regression task of depth prediction into a classification task, which provides more stable training signals for coarse scene layout prediction.
7.  **Scale-Invariant Logarithmic (SILog) Loss**: A loss function designed for monocular depth estimation that is invariant to global scaling of depth predictions, addressing the inherent scale ambiguity of single-image depth estimation.
## 3.2. Previous Works
1.  **CLIP (Radford et al., 2021)**: The foundational VLM used as the backbone of this work, pre-trained on 400 million image-text pairs, achieving state-of-the-art zero-shot classification performance.
2.  **DepthCLIP (Zhang et al., 2022)**: The first VLM-based zero-shot monocular depth estimation method, which reframes depth prediction as a classification task matching image regions to handcrafted text prompts like "close" or "far". It requires no task-specific training, but produces very coarse, low-precision outputs with only 10 depth bins.
3.  **AdaBins (Bhat et al., 2021)**: A high-performance traditional monocular depth estimation method that uses adaptive per-image depth bins to improve prediction precision. It serves as the inspiration for the hybrid classification-regression head in this work, but uses full fine-tuning of a dedicated depth model instead of VLM adaptation.
4.  **AdaMix (Wang et al., 2022)**: A MoA-based PEFT method for NLP tasks that uses stochastic expert routing during training and expert merging during inference. The MoA design in this paper is adapted from AdaMix, but modified to use deterministic routing for stable dense depth prediction.
5.  **Prompt Learning for CLIP Depth (Auty et al., 2023; Kim et al., 2024)**: Prior works that learn continuous text prompts to improve CLIP's depth prediction performance, but have limited adaptation capacity compared to PEFT methods like MoA.
## 3.3. Technological Evolution
The monocular depth estimation field has evolved along three parallel trajectories:
1.  **Supervised Depth (2010s)**: Fully supervised methods using annotated RGB-D datasets achieved high accuracy, but were limited by high annotation costs.
2.  **Self-Supervised Depth (2017-2022)**: Methods using video sequences and geometric constraints as supervision eliminated the need for depth annotations, but had lower accuracy than supervised methods.
3.  **Foundation Depth Models & VLM Adaptation (2022-present)**: Large foundation models trained on massive multi-domain datasets achieved state-of-the-art generalization but have very high computational cost. Concurrent work started adapting VLMs like CLIP to depth tasks, starting with zero-shot DepthCLIP, followed by prompt learning methods, and now PEFT-based adaptation methods like the one proposed in this paper.
    This work sits at the intersection of VLM adaptation and efficient dense prediction, filling the gap of low-parameter, high-precision VLM-based depth estimation.
## 3.4. Differentiation Analysis
Compared to prior related work, the core innovations of MoA-DepthCLIP are:
1.  **vs. Zero-shot DepthCLIP**: MoA-DepthCLIP uses learnable MoA adaptation instead of static handcrafted prompts, a hybrid prediction head instead of only coarse classification, and 128 fine-grained bins instead of 10 coarse bins, leading to 91% higher $\delta_1$ accuracy.
2.  **vs. Full Fine-Tuning Depth Models**: MoA-DepthCLIP freezes most of the CLIP backbone and only trains MoA modules, final 4 backbone layers, and prediction heads, using less than 10% of the parameters of full fine-tuning approaches.
3.  **vs. Prompt Learning VLM Depth Methods**: MoA enables spatially-aware token-level adaptation instead of only global prompt tuning, leading to better adaptation to local geometric patterns in the image.
4.  **vs. NLP MoA Methods**: The MoA design uses deterministic gating instead of stochastic routing, which is optimized for stable dense geometric prediction instead of sequence-level NLP tasks.

# 4. Methodology
## 4.1. Principles
The core design principle of MoA-DepthCLIP is to maximize the transfer of CLIP's pre-trained semantic knowledge to the depth estimation task, while minimizing trainable parameters and ensuring high geometric prediction precision. The key intuitions behind the design are:
1.  Lightweight MoA modules enable token-level spatially-aware adaptation without forgetting CLIP's pre-trained features, as they use residual connections and only modify a small subset of features.
2.  A global scene context vector derived from text prompts aligns CLIP's visual features to the depth task's semantic domain (e.g., indoor scenes for NYU Depth V2) without adding trainable parameters.
3.  A hybrid classification-regression head combines the stable coarse prediction of classification-based depth binning and the fine-grained precision of direct regression.
4.  A composite loss function enforces both categorical depth accuracy and geometric consistency, addressing the scale ambiguity inherent to monocular depth estimation.

    The overall architecture of MoA-DepthCLIP is illustrated in Figure 1 from the original paper below:

    ![Fig. 1: Overall architecture of MoA-DepthCLIP. Scene prompts are encoded using a frozen CLIP text encoder to form a global scene context vector. The image is encoded with a frozen ViT-B/32 backbone augmented with MoAs. Fused features are then passed to a dual-head prediction module: one head performs depth bin classification and produces a binned depth map via weighted summation, while the other head performs direct regression. The final output depth map is a fusion of both predictions.](images/1.jpg)
    *Fig. 1: Overall architecture of MoA-DepthCLIP. Scene prompts are encoded using a frozen CLIP text encoder to form a global scene context vector. The image is encoded with a frozen ViT-B/32 backbone augmented with MoAs. Fused features are then passed to a dual-head prediction module: one head performs depth bin classification and produces a binned depth map via weighted summation, while the other head performs direct regression. The final output depth map is a fusion of both predictions.*

## 4.2. Core Methodology In-depth
### 4.2.1 Mixture-of-Adapters (MoA) Adaptation
#### Selective MoA Placement
Instead of inserting MoA modules after every transformer layer of the ViT backbone (which would increase parameter overhead), the authors empirically found that inserting MoA modules only at layers 2, 5, 8, and 11 of the 12-layer ViT-B/32 backbone balances adaptation across early, mid-level, and late semantic features, while minimizing parameter count. This selective placement is illustrated in Figure 2a from the original paper below:

![该图像是示意图，展示了多个块和一个可选择的 MoA（Module of Action）。图中依次列出 Block 1、Block 2、MoA、Block 3、Block 4、Block 5 以及 Block 12，显示了流程的结构和相互关系。](images/2.jpg)
*Fig. 2a: Selective MoA Placement. MoA modules (green) are inserted at key layers (2, 5, 8, 11) of the ViT-B/32 encoder (tan).*

The internal architecture of a single MoA module is shown in Figure 2b from the original paper below:

![该图像是示意图，展示了一个具有多个专家的门控网络架构。输入标记通过门控网络处理后，分别传递给四个专家，之后合并求和生成输出标记，并包含残差注入机制。](images/3.jpg)
*Fig. 2b: Internal architecture of a single MoA module, showing the Gating Network, $K=4$ Experts, and residual injection.*

Each MoA module has three components: experts, a gating network, and residual mixing:
#### Expert Architecture
Each expert is a lightweight two-layer MLP with a bottleneck structure, designed to add minimal parameter overhead. The forward pass of an expert is defined by the exact formula from the original paper:
$$
\mathrm { E x p e r t } ( x ) = W _ { 2 } \sigma ( W _ { 1 } x )
$$
Symbol explanation:
- $x$: Input token feature vector with hidden dimension $d=768$ (for ViT-B/32)
- $W_1$: Weight matrix of the first linear layer, shape $\mathbb{R}^{d \times d_b}$, where $d_b=64$ is the small bottleneck dimension to reduce parameter count
- $\sigma$: GELU activation function, a non-linear activation commonly used in transformers
- $W_2$: Weight matrix of the second linear layer, shape $\mathbb{R}^{d_b \times d}$, mapping the bottleneck feature back to the original hidden dimension

  This design adds only ~100k parameters per expert, making the MoA module extremely lightweight.
#### Gating Network
To enable token-specific specialized adaptation, a gating network predicts routing weights for each input token across the $K$ experts. The gating formula from the original paper is:
$$
g _ { i } = \mathrm { s o f t m a x } \left( \frac { G ( x _ { i } ) } { \tau } \right)
$$
Symbol explanation:
- $x_i$: $i$-th input token feature vector
- $G(\cdot)$: Two-layer MLP with layer dimensions $768 \to 128 \to K$ and ReLU activation, outputting K logits for each token
- $\tau=2.0$: Temperature hyperparameter controlling the sharpness of the softmax distribution (higher temperature produces a more uniform distribution across experts)
- $g_i$: K-dimensional probability vector, where $g_{i,k}$ is the weight assigned to expert $k$ for token $i$

  Unlike prior NLP MoA methods like AdaMix that use stochastic routing during training and expert merging during inference, this work uses deterministic gating (the same $g_i$ weights are used for both training and inference), which provides more stable outputs for dense depth prediction.
#### Mixture and Residual Injection
The outputs of all experts are combined using the gating weights, then added back to the original input token via a residual connection to preserve pre-trained CLIP features and avoid catastrophic forgetting. The exact formula from the original paper is:
$$
\tilde { x } _ { i } = x _ { i } + \sum _ { k = 1 } ^ { K } g _ { i , k } \mathrm { E x p e r t } _ { k } ( x _ { i } )
$$
Symbol explanation:
- $\tilde{x}_i$: Adapted output token feature vector
- $x_i$: Original input token feature vector
- $g_{i,k}$: Routing weight for expert $k$ for token $i$, from the gating network
- $\mathrm{Expert}_k(x_i)$: Output of expert $k$ for input token $x_i$

  To avoid expert collapse (where the gating network routes most tokens to only a small number of experts, negating the benefits of specialization), the authors track the entropy of the gating distribution during training as a diagnostic tool, using the exact formula from the original paper:
$$
H ( g _ { i } ) = - \sum _ { k = 1 } ^ { K } g _ { i , k } \log g _ { i , k }
$$
A high average entropy value indicates that multiple experts are being actively used. The authors do not use auxiliary load-balancing losses, relying instead on the deterministic gating design to prevent collapse.

In addition to the MoA modules, the authors selectively fine-tune only the final 4 layers of the ViT-B/32 backbone, freezing all earlier layers to preserve pre-trained CLIP features.
### 4.2.2 Global Scene Context Fusion
Prior DepthCLIP uses handcrafted pixel-level prompts (e.g., "close", "far") that lack global scene context. To provide a fixed semantic prior aligned to the indoor NYU Depth V2 dataset without adding trainable parameters, the authors use the frozen CLIP text encoder to generate a global scene context vector:
1.  First, define a fixed set of text prompts corresponding to common indoor scene categories in NYU Depth V2, e.g., "a photo of a kitchen", "a photo of a bedroom", "a photo of a classroom".
2.  Encode each prompt using the frozen CLIP text encoder, then L2-normalize each embedding.
3.  Compute the element-wise average of all prompt embeddings to get a single global context vector $c$, representing a generic "indoor scene" semantic prior.

    This global context vector $c$ is then broadcast to match the spatial dimensions of the MoA-adapted visual feature map, and concatenated with the visual features along the channel dimension. This provides a uniform high-level semantic prior across the entire image, complementing local visual details.
### 4.2.3 Depth Binning Strategy
The hybrid prediction head uses classification over discrete depth bins for stable coarse prediction. The authors tested different fixed depth bin counts, and found that $N=128$ bins provides the optimal trade-off between prediction granularity and training stability for the MoA-adapted CLIP framework. This avoids the coarse 10-bin design of DepthCLIP, and the computational cost of adaptive per-image binning methods like AdaBins.
### 4.2.4 Hybrid Prediction Head
The fused features (visual features + global context vector) are fed into two parallel prediction heads:
1.  **Classification Head**: Predicts a per-pixel probability distribution over the 128 depth bins. The binned depth map is computed as the weighted sum of the depth bin centers, weighted by the predicted probabilities for each bin.
2.  **Regression Head**: Predicts a continuous per-pixel depth value directly, providing fine-grained geometric precision.

    The final output depth map is the fusion of the outputs of both heads, combining the stability of classification and the precision of regression.
### 4.2.5 Composite Loss Function
The model is trained using a weighted sum of three complementary loss terms, designed to balance coarse scene layout prediction and fine-grained geometric accuracy. The exact total loss formula from the original paper is:
$$
\mathcal { L } _ { \mathrm { t o t a l } } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } + \lambda _ { \mathrm { s i l o g } } \mathcal { L } _ { \mathrm { s i l o g } }
$$
The fixed loss weights are set to $\lambda_{cls}=1.0$, $\lambda_{reg}=1.0$, $\lambda_{silog}=0.5$. Each loss term is explained below:
#### Classification Loss $\mathcal{L}_{cls}$
Per-pixel cross-entropy loss for the classification head, which compares the predicted bin logits for each pixel to the ground truth depth bin index. This loss provides a stable training signal for learning the coarse overall layout of the scene.
#### Regression Loss $\mathcal{L}_{reg}$
Per-pixel L1 loss for the regression head, which penalizes the absolute difference between the predicted continuous depth $\hat{d}_i$ and ground truth depth $d_i$ for each valid pixel. This provides a fine-grained signal for local geometric accuracy.
#### Scale-Invariant Logarithmic Loss $\mathcal{L}_{silog}$
This loss addresses the inherent scale and shift ambiguity of monocular depth estimation, ensuring the model is invariant to global scaling of depth predictions. The exact formula from the original paper is:
$$
\mathcal { L } _ { \mathrm { s i l o g } } = \alpha \left( \mathrm { V a r } ( g ) + \lambda \mathrm { M e a n } ( g ) ^ { 2 } \right)
$$
Symbol explanation:
- $g_i = \log \hat{d}_i - \log d_i$: Log difference between predicted and ground truth depth for pixel $i$
- $\mathrm{Var}(g)$: Variance of $g_i$ across all valid pixels
- $\mathrm{Mean}(g)$: Mean of $g_i$ across all valid pixels
- $\lambda=0.85$: Weighting term balancing mean and variance components
- $\alpha=10.0$: Scaling factor to normalize the loss magnitude

# 5. Experimental Setup
## 5.1. Datasets
The experiments are conducted on the **NYU Depth V2 dataset**, the standard benchmark for indoor monocular depth estimation:
- **Source**: Collected using a Microsoft Kinect RGB-D sensor, covering 464 different indoor scenes (offices, homes, classrooms, etc.)
- **Scale**: 47,844 training images, 654 test images, all with resolution 640×480, with dense depth annotations.
- **Preprocessing**: Depth values are capped at 10 meters following standard protocol for this benchmark.
- **Justification**: NYU Depth V2 is the most widely used indoor depth dataset, and was used to evaluate the baseline DepthCLIP method, enabling direct fair comparison to prior work.
## 5.2. Evaluation Metrics
All metrics used in the paper are explained below with conceptual definitions, standardized formulas, and symbol explanations:
### Threshold Accuracy $\delta_1, \delta_2, \delta_3$ (higher = better)
1.  **Conceptual Definition**: Measures the share of pixels where the predicted depth is within a factor of $1.25^i$ of the ground truth depth. $\delta_1$ is the strictest and most important metric, as it requires predictions to be very close to ground truth, while $\delta_3$ is the most lenient.
2.  **Formula**:
    $$
    \delta_i = \frac{1}{M} \sum_{p \in \text{valid}} \mathbb{1}\left( \max\left( \frac{\hat{d}_p}{d_p}, \frac{d_p}{\hat{d}_p} \right) < 1.25^i \right)
    $$
3.  **Symbol Explanation**:
    - $M$: Number of valid pixels with ground truth depth annotations
    - $\mathbb{1}(\cdot)$: Indicator function, equals 1 if the condition inside is true, 0 otherwise
    - $\hat{d}_p$: Predicted depth for pixel $p$
    - $d_p$: Ground truth depth for pixel $p$
    - $i \in \{1,2,3\}$: Index for the three threshold metrics
### Absolute Relative Error (AbsRel) (lower = better)
1.  **Conceptual Definition**: Average of the absolute difference between predicted and ground truth depth, normalized by ground truth depth. This metric measures relative error, so it is not biased by differences in absolute depth values across scenes.
2.  **Formula**:
    $$
    \text{AbsRel} = \frac{1}{M} \sum_{p \in \text{valid}} \frac{|\hat{d}_p - d_p|}{d_p}
    $$
### log10 Error (lower = better)
1.  **Conceptual Definition**: Average absolute difference between the base-10 logarithm of predicted and ground truth depth. It penalizes errors in both small and large depth ranges equally.
2.  **Formula**:
    $$
    \text{log10} = \frac{1}{M} \sum_{p \in \text{valid}} |\log_{10}\hat{d}_p - \log_{10}d_p|
    $$
### Root Mean Squared Error (RMSE) (lower = better)
1.  **Conceptual Definition**: Square root of the average squared difference between predicted and ground truth depth. It penalizes large prediction errors more heavily than L1 loss, making it sensitive to outlier predictions.
2.  **Formula**:
    $$
    \text{RMSE} = \sqrt{ \frac{1}{M} \sum_{p \in \text{valid}} (\hat{d}_p - d_p)^2 }
    $$
## 5.3. Baselines
The paper uses progressive ablation baselines to measure the contribution of each component of the framework, plus the original DepthCLIP baseline:
1.  **DepthCLIP (Original)**: The baseline zero-shot VLM depth method with ResNet-50 backbone, 10 handcrafted bins, no task-specific training. This is the core state-of-the-art baseline for VLM-based depth estimation.
2.  **Our Baseline**: ViT-B/32 backbone, no composite loss, no MoA modules, 10 bins, used to measure the gain from switching to the ViT CLIP backbone.
3.  **+ Composite Loss**: Our baseline with the addition of the 3-term composite loss function, used to measure the gain from the loss design.
4.  **+ MoA**: Previous configuration with the addition of MoA modules, used to measure the gain from the MoA adaptation.
5.  **MoA-DepthCLIP (Full)**: Previous configuration with 128 depth bins, the full proposed framework.

    These baselines are representative, as they allow incremental measurement of the contribution of each individual component of the proposed framework, and direct comparison to the prior state-of-the-art VLM depth method.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The following are the results from Table 1 of the original paper:

| Method                  | Backbone   | Comp. Loss | MoA | #Bins | $\delta_1$ ↑ | $\delta_2$ ↑ | $\delta_3$ ↑ | AbsRel ↓ | log10 ↓ | RMSE ↓ |
|-------------------------|------------|------------|-----|-------|--------------|--------------|--------------|----------|---------|--------|
| DEPTHCLIP               | ResNet-50  | ❌          | ❌  | 10    | 0.390        | 0.680        | 0.848        | 0.393    | 0.158   | 1.176  |
| Our Baseline            | ViT-B/32   | ❌          | ❌  | 10    | 0.417        | 0.701        | 0.890        | 0.377    | 0.147   | 1.096  |
| + COMPOSITE LOSS        | ViT-B/32   | ✅          | ❌  | 10    | 0.503        | 0.791        | 0.925        | 0.310    | 0.121   | 0.843  |
| + MoA                   | ViT-B/32   | ✅          | ✅  | 10    | 0.508        | 0.797        | 0.931        | 0.308    | 0.120   | 0.821  |
| MoA-DepthCLIP (Full)    | ViT-B/32   | ✅          | ✅  | 128   | 0.745        | 0.841        | 0.895        | 0.321    | 0.098   | 0.520  |

### Result Interpretation
1.  **Incremental Gain of Each Component**:
    - Switching from ResNet-50 to ViT-B/32 backbone improves $\delta_1$ by 6.9% and reduces RMSE by 6.8%, showing ViT CLIP has stronger feature representation for depth tasks.
    - Adding the composite loss function delivers the largest initial gain, improving $\delta_1$ by 20.6% and reducing RMSE by 23.1% compared to the ViT baseline, proving the loss design is critical for training stability.
    - Adding MoA modules delivers an incremental 1.0% gain in $\delta_1$ and 2.6% RMSE reduction, showing the adaptation benefit of MoA modules.
    - Increasing the number of depth bins to 128 delivers the largest overall gain, improving $\delta_1$ by 46.7% and reducing RMSE by 36.7% compared to the 10-bin MoA configuration, showing fine-grained discretization is critical for high-precision depth prediction.
2.  **Performance vs Baseline**: The full MoA-DepthCLIP outperforms the original DepthCLIP baseline by 91% in $\delta_1$ (0.390 → 0.745) and reduces RMSE by 55% (1.176 → 0.520), a dramatic improvement.
3.  **Trade-off Analysis**: The only negative change is a small 3.9% drop in $\delta_3$ (0.931 → 0.895) when using 128 bins. The authors attribute this to the model's increased specialization: it makes more fine-grained predictions that are highly accurate for most pixels, but a small number of ambiguous pixels fall just outside the wide $1.25^3$ threshold. This is a positive trade-off, as $\delta_1$ and $\delta_2$ (the stricter, more practically relevant metrics) improve dramatically.
## 6.2. Ablation Studies / Parameter Analysis
The authors conduct two ablation studies to select optimal hyperparameters for the framework:
### Ablation 1: Effect of Number of Experts in MoA Modules
The following are the results from Table 2 of the original paper, with depth bins fixed to 10:

| #Experts | AbsRel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|----------|----------|--------|--------------|
| 1        | 0.394    | 0.560  | 0.660        |
| 2        | 0.396    | 0.559  | 0.661        |
| 4        | 0.396    | 0.561  | 0.661        |
| 8        | 0.395    | 0.557  | 0.665        |
| 16       | 0.395    | 0.558  | 0.661        |

**Interpretation**: Performance is relatively stable across different expert counts. $K=8$ delivers the highest $\delta_1$, but doubles the computational overhead compared to $K=4$ for only a 0.6% gain in $\delta_1$. The authors select $K=4$ as the optimal trade-off between specialization capacity and computational cost.
### Ablation 2: Effect of Number of Depth Bins
The following are the results from Table 3 of the original paper, with number of experts fixed to 4:

| #Bins | AbsRel ↓ | RMSE ↓ | $\delta_1$ ↑ |
|-------|----------|--------|--------------|
| 40    | 0.327    | 0.522  | 0.737        |
| 64    | 0.329    | 0.534  | 0.730        |
| 128   | 0.321    | 0.521  | 0.745        |
| 180   | 0.323    | 0.532  | 0.735        |
| 200   | 0.329    | 0.541  | 0.717        |

**Interpretation**: Performance improves as bin count increases up to 128, then degrades for larger bin counts. The authors attribute the degradation to data sparsity: with too many bins, there are too few training samples per bin for the model to learn a stable classification distribution. $N=128$ is selected as the optimal bin count.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper proposes MoA-DepthCLIP, a highly parameter-efficient framework for adapting pre-trained CLIP representations to monocular depth estimation. The framework combines lightweight MoA module insertion, selective backbone fine-tuning, global text-derived scene context fusion, a hybrid classification-regression prediction head, and a composite geometric loss function. Extensive experiments on the NYU Depth V2 benchmark show that MoA-DepthCLIP outperforms the DepthCLIP baseline by a large margin, improving $\delta_1$ accuracy by 91% and reducing RMSE by 55%, while using only a small fraction of the trainable parameters required for full fine-tuning. The work demonstrates that targeted lightweight VLM adaptation strategies can effectively bridge the gap between CLIP's high-level semantic knowledge and the fine-grained geometric precision requirements of dense prediction tasks.
## 7.2. Limitations & Future Work
### Stated Limitations
1.  The framework is only evaluated on the indoor NYU Depth V2 dataset, and has not been tested on outdoor depth datasets like KITTI.
2.  The global scene context vector is fixed for all indoor scenes, and does not adapt dynamically to individual input images.
### Suggested Future Work
1.  Extend the framework to outdoor datasets and cross-domain depth estimation settings to improve generalization.
2.  Incorporate dynamic components like attention-based prompt selection to generate per-image adaptive context vectors, improving performance on diverse scene types.
3.  Test the framework on larger ViT variants (e.g., ViT-L/14) to further improve performance.
## 7.3. Personal Insights & Critique
### Inspirations and Transferability
This work demonstrates a generalizable paradigm for adapting VLMs to dense geometric prediction tasks, which can be transferred to other tasks like semantic segmentation, surface normal estimation, and optical flow prediction, by only modifying the prediction head and loss function. The lightweight MoA design makes the framework suitable for edge deployment on devices with limited computational resources, which is critical for real-world applications like robotics and AR.
### Potential Improvements
1.  **Adaptive Context**: The fixed global context vector could be replaced with a per-image adaptive context retrieved from a text prompt bank based on the input image's content, which would improve performance on diverse scene types.
2.  **Missing Fusion Details**: The paper does not explain how the outputs of the classification and regression heads are fused (e.g., fixed weighted sum, learned fusion weight). This detail could be clarified to improve reproducibility.
3.  **Comparison to State-of-the-Art Depth Models**: The paper only compares to the DepthCLIP baseline, and does not compare to state-of-the-art fully supervised or foundation depth models, so it is unclear how competitive MoA-DepthCLIP is with non-VLM depth methods of similar parameter size.
4.  **$\delta_3$ Trade-off Mitigation**: The small drop in $\delta_3$ when using 128 bins could be mitigated by adding a small regularization term that penalizes predictions that deviate too far from the coarse classification output, reducing outlier predictions for ambiguous pixels.