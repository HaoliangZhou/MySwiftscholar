# 1. Bibliographic Information
## 1.1. Title
The title is *HUydra: Full-Range Lung CT Synthesis via Multiple HU Interval Generative Modelling*. Its central topic is a novel framework for generating high-fidelity full-range lung computed tomography (CT) scans by decomposing the Hounsfield Unit (HU) range into separate tissue-specific intervals, training generative models on each interval, and merging outputs via a learned reconstruction network to produce the final full-range scan.
## 1.2. Authors
All authors are affiliated with Portuguese research institutions focused on medical imaging and artificial intelligence:
- António Cardoso: INESC TEC, Faculty of Engineering, University of Porto (FEUP)
- Pedro Sousa: INESC TEC, Faculty of Sciences, University of Porto (FCUP)
- Tania Pereira: INESC TEC, Faculty of Engineering, University of Porto (FEUP)
- Hélder P. Oliveira: INESC TEC, Faculty of Sciences, University of Porto (FCUP)
  The research group has extensive prior work in lung CT analysis, generative modeling for medical imaging, and computer-aided diagnosis (CAD) systems.
## 1.3. Journal/Conference
As of the 2026-03-24 publication date, this work is a preprint hosted on arXiv, the leading open-access preprint server for computer science, machine learning, and medical imaging research. It has not yet undergone peer review for formal conference or journal publication.
## 1.4. Publication Year
2026 (preprint release date: 2026-03-24 UTC)
## 1.5. Abstract
### Research Objective
Address the critical data scarcity bottleneck for lung cancer CAD model deployment and validation, while solving the high computational cost and low fidelity of existing generative AI models for full HU range CT synthesis.
### Core Methodology
Decompose the complex full HU range CT generation task into separate sub-tasks for narrow, tissue-specific HU intervals. Train generative architectures on individual HU window views, then merge outputs via a learned reconstruction network that reverses the standard clinical HU windowing process. A multi-head Vector-Quantized Variational Autoencoder (VQVAE) is proposed as the highest-performing generative variant.
### Main Results
The framework achieves a 6.2% improvement in Fréchet Inception Distance (FID) over conventional 2D full-range generative baselines, plus superior Maximum Mean Discrepancy (MMD), precision, and recall across all HU intervals. It also reduces model complexity and computational cost compared to end-to-end full-range models.
### Key Conclusion
The proposed HU interval decomposition paradigm aligns generative modeling with clinical interpretation workflows, enabling structure-aware, explainable, high-fidelity medical image synthesis.
## 1.6. Original Source Link
- Preprint landing page: https://arxiv.org/abs/2603.23041v1
- Full PDF: https://arxiv.org/pdf/2603.23041v1
- Publication status: Unpeer-reviewed preprint.

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Data scarcity is the primary bottleneck for deploying and validating medical imaging CAD models. For lung cancer (the leading cause of cancer-related mortality globally), limited labeled CT datasets delay model development, reduce diagnostic accuracy, and worsen patient outcomes. While generative AI can synthesize synthetic data to address this gap, existing models struggle to model the complex full HU range distribution of CT scans (covering air, lung tissue, fat, water, blood, muscle, and bone from -1000 to 3000 HU), leading to high computational cost and low output fidelity.
### Prior Research Gap
All existing generative models for CT synthesis attempt to model the full HU range end-to-end, ignoring the standard clinical practice of HU windowing: radiologists do not view the full HU range at once, but use separate narrow HU windows to inspect different tissue types. No prior work leverages this clinical intuition to decompose the generation task.
### Innovative Entry Point
The paper translates clinical HU windowing practice to generative modeling: instead of modeling the complex full HU distribution, generate each narrow, tissue-specific HU interval separately (each with a simple, homogeneous distribution), then merge the interval outputs via a small learned reconstruction network to produce the full-range scan.
## 2.2. Main Contributions / Findings
### Primary Contributions
1. **Novel HU Decomposition Strategy**: First framework for full lung CT generation that splits the HU range into disjunct, tissue-specific intervals, aligning generative modeling with clinical workflow and enabling targeted tissue-specific synthesis.
2. **Architectural Variants**: Proposal and systematic comparison of three HU-aware generative variants (multi-channel, multi-decoder, multi-head) for VQVAEs, with the multi-head variant achieving the best performance.
3. **Enhanced Explainability**: Stepwise generation process lets clinicians validate synthetic outputs at both the individual HU interval (tissue) level and full scan level, unlike black-box end-to-end generative models.
4. **Learned HU Window Reversal**: First learned reconstruction method to reverse HU windowing with minimal information loss, expanding the utility of standard windowing workflows.
5. **Performance Improvements**: 6.2% FID improvement over full-range baselines, plus superior MMD, precision, recall, and sample diversity, with lower model complexity and computational cost.
6. **Clinical Validation**: Visual Turing Test with practicing clinicians confirms the synthetic scans have high anatomical realism and diagnostic relevance.
7. **Extendable Framework**: Open design supports extension to other medical imaging modalities where intensity-based decomposition (e.g., windowing for MRI, X-ray) is feasible.
### Key Findings
- Full-range CT scans can be reconstructed with near-perfect accuracy (99.9% precision, 100% recall) from only 4 HU interval views using a small 3-layer convolutional neural network (CNN) reconstruction model.
- The HU decomposition approach improves generative performance for VQVAE models, while WGAN-GP and diffusion model architectures do not benefit from the multi-channel interval formulation.
- The multi-head VQVAE achieves the best balance of performance and complexity: only 1 million additional parameters compared to a baseline full-range VQVAE, with consistent performance gains across all HU intervals.
- Synthetic scans from the pipeline are realistic enough to be processed by downstream CAD models (e.g., lung segmentation) trained on real CT data, making them suitable for data augmentation use cases.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
All core technical terms are explained for beginner readers:
1. **Hounsfield Unit (HU)**: Standardized unit of radiodensity used in CT scans, ranging from -1000 (air) to 3000 (dense bone). Each tissue type has a characteristic HU range (e.g., lung tissue = [-700, -600] HU, bone = [700, 3000] HU).
2. **HU Windowing**: Standard clinical preprocessing step where only HU values within a narrow interval (window) are retained, values outside the window are clipped to the window min/max, and the clipped values are scaled to [0,1] for visualization. Radiologists use different windows to inspect different tissues (e.g., lung window, bone window, soft tissue window).
3. **Computer-Aided Diagnosis (CAD)**: AI systems that assist clinicians in interpreting medical images, detecting abnormalities, and making diagnostic decisions.
4. **Generative Adversarial Network (GAN)**: Generative model with two competing components:
   - Generator: Maps random noise vectors to synthetic images.
   - Discriminator: Trained to distinguish real images from synthetic ones.
     The two components are trained in an adversarial game until the generator produces realistic enough images to fool the discriminator.
5. **Wasserstein GAN with Gradient Penalty (WGAN-GP)**: Improved GAN variant that uses Wasserstein distance instead of cross-entropy loss, plus a gradient penalty term to enforce a Lipschitz constraint on the discriminator. This stabilizes training, reduces mode collapse (where the generator only produces a limited set of outputs), and improves output diversity.
6. **Score-Based Diffusion Model (DM)**: Generative model that operates in two phases:
   - Forward process: Gradually adds Gaussian noise to real images over hundreds of timesteps until the image becomes pure random noise.
   - Reverse process: Trains a neural network to iteratively remove noise from random noise to generate new realistic images. DMs are known for high-fidelity, diverse outputs but have high computational cost.
7. **Vector-Quantized Variational Autoencoder (VQVAE)**: Generative model with three core components:
   - Encoder: Maps input images to continuous latent vectors.
   - Codebook: Fixed set of learned discrete latent vectors. The encoder output is quantized to the nearest codebook entry to get a discrete latent representation.
   - Decoder: Reconstructs the input image from the quantized latent vector.
     After training the encoder-decoder-codebook pipeline, a transformer model is trained to generate new sequences of codebook indices, which are decoded to produce new synthetic images. VQVAEs excel at generating high-resolution images with consistent anatomical structure.
8. **Evaluation Metrics (detailed formulas in Section 5.2)**:
   - **FID (Fréchet Inception Distance)**: Measures similarity between real and synthetic image distributions, lower = better fidelity.
   - **MMD (Maximum Mean Discrepancy)**: Non-parametric measure of distribution similarity, lower = better alignment between real and synthetic data.
   - **Precision**: Fraction of synthetic images that are realistic (lie within the real data distribution), higher = better.
   - **Recall**: Fraction of the real data distribution covered by synthetic images, higher = better diversity.
   - **MS-SSIM (Multi-Scale Structural Similarity Index)**: Measures structural similarity between images. For evaluating sample diversity, lower average pairwise MS-SSIM = higher diversity.

## 3.2. Previous Works
Prior CT synthesis research falls into four broad categories:
1. **Early Non-Deep Learning Methods (2000s to mid-2010s)**:
   - Used mathematical phantoms (e.g., Shepp-Logan phantom, XCAT phantom), finite element simulations, statistical shape models, or multi-atlas registration to generate synthetic CT scans.
   - Limitation: Relied on hand-designed priors, produced low-realism, low-diversity outputs, and could not capture the full variability of real patient anatomy.
2. **Autoencoder-Based Methods (mid-2010s to present)**:
   - Used Variational Autoencoders (VAEs) and VQVAEs for medical image synthesis.
   - Limitation: Often produced blurry outputs lacking fine-grained tissue textures.
3. **GAN-Based Methods (2017 to present)**:
   - Widely applied to lung CT synthesis, including synthetic nodule generation, data augmentation for COVID-19 detection, and low-dose CT denoising.
   - Limitation: Suffered from unstable training, mode collapse, and sensitivity to small dataset sizes.
4. **Diffusion Model-Based Methods (2022 to present)**:
   - Applied to 2D and 3D medical image synthesis, achieving state-of-the-art fidelity.
   - Limitation: Extremely computationally expensive, especially for full-range CT synthesis, and act as black boxes with no clinical interpretability.
### Critical Prior Gap
No prior work leverages standard clinical HU windowing to decompose the CT generation task. All existing models attempt to model the full HU range end-to-end, leading to high compute cost and suboptimal fidelity for individual tissue types.

## 3.3. Technological Evolution
- 2000s: Non-deep learning CT synthesis relies on hand-engineered priors.
- 2014: GANs are invented, revolutionizing generative modeling, and are quickly applied to medical imaging.
- 2017: VQVAEs are introduced, enabling high-resolution discrete latent generative modeling.
- 2020: Diffusion models emerge as the new state-of-the-art for high-fidelity image synthesis.
- 2020–2025: Explosion of generative medical imaging research, but all work focuses on end-to-end full-range generation.
- 2026: This work introduces the first HU interval decomposition framework for CT synthesis, bridging clinical workflow and generative AI.

## 3.4. Differentiation Analysis
Compared to all prior end-to-end full-range generative methods:
1. **Task Decomposition**: Splits the complex full HU range generation task into simpler sub-tasks for narrow HU intervals, each with a homogeneous distribution that is easier for generative models to learn.
2. **Clinical Alignment**: The pipeline directly mirrors standard clinical HU windowing practice, making outputs more interpretable and trustworthy for clinicians.
3. **Modular Design**: Separates generation of individual tissue intervals and reconstruction of the full scan, enabling independent optimization of each component and dynamic generation of custom HU windows on demand.
4. **Efficiency**: Reduces overall model complexity and computational cost compared to equivalent full-range diffusion models, while achieving better or equal performance.
5. **Explainability**: Clinicians can inspect each individual HU interval output to validate the synthetic scan, unlike black-box end-to-end models.

# 4. Methodology
## 4.1. Principles
The core intuition of the HUydra framework is inspired by clinical practice: radiologists do not interpret full-range CT scans directly, but use separate HU windows to inspect each tissue type. This work translates this intuition to generative modeling: instead of learning a single complex model for the full HU range distribution, learn separate simpler models for each narrow, tissue-specific HU interval, then combine the outputs to produce the full-range scan. The theoretical basis is that each narrow HU interval has a far more homogeneous distribution than the full range, so generative models can learn it with higher fidelity, lower compute cost, and better tissue-specific texture capture.
## 4.2. Core Methodology In-depth
The framework has two core components: (1) full-range CT reconstruction from HU interval views, and (2) multi-interval generative modeling.
### 4.2.1 Full-Range CT Reconstruction from HU Intervals
Step 1: Define HU intervals
Let $X: \mathcal{V} \to H \subset \mathbb{R}$ denote a CT scan, where $\mathcal{V}$ is the voxel domain (set of all voxel positions in the scan) and $H$ is the full HU range. We define a set of non-overlapping HU intervals:
$$
\mathcal{T} = \{ I_1, I_2, ..., I_K \}, \quad I_k = [\mathrm{hu}_{min}, \mathrm{hu}_{max}], \quad I_i \cap I_j = \emptyset
$$
Where $\mathcal{T}$ is the set of $K$ disjunct HU intervals, $I_k$ is the k-th interval with minimum HU value $\mathrm{hu}_{min}$ and maximum $\mathrm{hu}_{max}$. For this work, $K=4$ with intervals $[-950, -700]$, $[-500, -200]$, `[30, 70]`, `[100, 1000]`, chosen to cover all relevant lung tissue types.

Step 2: Generate HU-clipped views
For each interval $I_k$, we create a clipped and scaled view $X_k$ of the original CT scan $X$:
$$
X_k(v) = \frac{\mathrm{clip}(X(v), \mathrm{hu}_{min}, \mathrm{hu}_{max}) - \mathrm{hu}_{min}}{\mathrm{hu}_{max} - \mathrm{hu}_{min}}, \quad \mathrm{clip}(h, a, b) = \left\{
\begin{array}{ll}
a, & \text{if } h < a \\
h, & \text{if } h \in [a, b] \\
b, & \text{if } h > b.
\end{array}
\right.
$$
For each voxel $v$, the $\mathrm{clip}$ function limits the original HU value `X(v)` to the interval $I_k$, then the clipped value is min-max scaled to the range `[0,1]` for model processing. Each $X_k$ is a view of the CT scan focused exclusively on the tissue type in interval $I_k$. An example of these views is shown below:

![Figure 10: Example of a full-range CT sample and its respective HU-clipped views for the HU intervals $\[ - 9 5 0 , - 7 0 0 \]$ $\[ - \\mathrm { \\bar { 5 } 0 0 } , - 2 0 0 \]$ , \[30, 70\] and \[100, 1000\].](images/10.jpg)
*该图像是插图，展示了一例全范围CT扫描及其对应的HU区间剪裁视图，包含HU区间为 $[-950, -700]$、 $[-500, -200]$、 `[30, 70]` 和 `[100, 1000]` 的切片。图中的全范围CT图像位于左侧，各个HU区间的视图依次排列于右侧，展示了不同HU值下的图像特征变化。*

Step 3: Learned reconstruction network
The paper first proves that no fixed linear or affine combination of the $K$ HU views can reconstruct the full-range scan, as each $X_k$ has a locally linear scaling that is inconsistent across intervals. Instead, a non-linear learned reconstruction model $\mathcal{R}$ is trained to map the set of $K$ HU views $\{X_1, X_2, ..., X_K\}$ to the full-range scan $X$. Two reconstruction model architectures are explored:
1. **MLP-based Reconstruction**: Voxel-wise processing. For each voxel $v$, input a vector `\mathbf{x}_v = [X_1(v), X_2(v), ..., X_K(v)]^T \in [0,1]^K` of the $K$ clipped values at that voxel, output the reconstructed full-range HU value $\hat{X}(v) \in [0,1]$. The architecture consists of an input layer of size $K$, followed by a variable number of fully connected layers with ReLU activation, and a final output layer with Sigmoid activation to constrain outputs to `[0,1]`.
2. **CNN-based Reconstruction**: Image-level processing. Input a $K$-channel tensor where each channel is one $X_k$ view, output a single-channel full-range CT image $\hat{X}$. The architecture uses a stack of convolutional layers with ReLU activation, plus a final convolutional layer with Sigmoid activation. CNNs capture local spatial correlations between voxels that MLP models miss.
   The two architectures are illustrated below:

   ![该图像是一个示意图，展示了两种不同的重建网络结构：多层感知机（MLP）和卷积神经网络（CNN），用于综合重建肺部CT图像。图中的（a）部分说明了MLP的架构，而（b）部分则展示了CNN的架构，两个模型都旨在实现对CT图像的高质量重建。](images/4.jpg)
   *该图像是一个示意图，展示了两种不同的重建网络结构：多层感知机（MLP）和卷积神经网络（CNN），用于综合重建肺部CT图像。图中的（a）部分说明了MLP的架构，而（b）部分则展示了CNN的架构，两个模型都旨在实现对CT图像的高质量重建。*

### 4.2.2 Multiple HU Interval Generation
The goal is to train a generative model that produces synthetic HU-clipped views $\{\tilde{X}_1, \tilde{X}_2, ..., \tilde{X}_K\}$, such that when fed to the pre-trained fixed reconstruction model $\mathcal{R}$, the output $\tilde{X} = \mathcal{R}(\{\tilde{X}_k\})$ has a distribution matching the real full-range CT distribution $p_X$. Three generative paradigms are adapted, with three architectural variants for VQVAEs:
1. **Multi-Channel Approach (model-agnostic, works for all generative types)**:
   Adapt the generative model to output $K$-channel images, where each channel corresponds to one HU interval:
   - For WGAN-GP: The generator outputs $K$-channel HU interval images, which are passed through $\mathcal{R}$ to get the full-range scan, then fed to the critic that compares it to real full-range scans. The framework is shown below:

     ![Figure 5: Multi-channel WGAN-GP framework.](images/5.jpg)
     *该图像是一个多通道WGAN-GP框架示意图，显示了生成过程的步骤，包括数据集、生成器G、重建模型以及批判器D的协同工作。生成器通过输入随机噪声z产生CT图像，并经过重建模型恢复出最终的输出，批判器则评估生成结果的质量。*

   - For Score-Based Diffusion Model: A U-Net is adapted to take $K$-channel noisy images and predict $K$-channel score values. After sampling, the $K$-channel outputs are passed through $\mathcal{R}$ to get the full-range scan. The framework is shown below:

     ![Figure 6: Multi-channel score-based diffusion model framework.](images/6.jpg)
     *该图像是示意图，展示了多通道评分扩散模型框架的各个步骤。图中包括数据集输入、噪声调度器处理、时间信息 U-Net 以及预测得分的生成过程，体现了全流程的结构与逻辑关系。*

   - For VQVAE: The encoder takes $K$-channel input, compresses it to a shared latent codebook representation, and the decoder outputs $K$-channel HU interval images. After training, a transformer generates new codebook index sequences, which are decoded to $K$-channel views and passed through $\mathcal{R}$ to get the full scan. The framework is shown below:

     ![Figure 7: Multi-channel VQVAE framework.](images/7.jpg)
     *该图像是图示，展示了多通道VQVAE框架的结构。图中包含了数据集输入、编码器E、代码本Z、量化处理、解码器D以及重建模型，整个流程说明了肺部CT影像的生成和重建过程。*

2. **Multi-Decoder VQVAE Variant**:
   A single shared encoder processes the $K$-channel input to produce a unified latent representation that captures common anatomical structure across all HU intervals. Then $K$ separate decoders, each specialized for one HU interval, decode the latent representation to the corresponding $X_k$ view. The $K$ outputs are concatenated and passed through $\mathcal{R}$. This design lets each decoder specialize in the texture of one tissue type while sharing the global structural latent representation. The framework is shown below:

   ![Figure 8: Multi-decoder VQVAE framework.](images/8.jpg)
   *该图像是示意图，展示了多解码器VQVAE框架的工作流程。数据集被输入到编码器E，生成潜在表示z，然后通过变换器生成解码器D的输入，最终 reconstruct 模型输出合成的图像。该框架旨在提高医学影像合成的观感与一致性。*

3. **Multi-Head VQVAE Variant (best performing)**:
   $K$ separate encoder heads, each dedicated to processing one HU interval input, extract low-level tissue-specific features. These are fed to a shared encoder backbone that produces a unified latent representation capturing global anatomical structure. Then $K$ separate decoder heads, each specialized for one HU interval, decode the shared latent to the corresponding $X_k$ view. Outputs are concatenated and passed through $\mathcal{R}$. This design balances tissue-specific feature extraction and global structural consistency, leading to the best overall performance. The framework is shown below:

   ![Figure 9: Multi-head VQVAE framework.](images/9.jpg)
   *该图像是示意图，展示了多头 VQVAE 框架的结构。图中展示了数据集如何经过编码器 E 和解码器 D 的多个路径，最终通过重建模型生成全范围的肺部 CT 图像。*

### 4.2.3 Loss Function Formulations
All formulas are reproduced exactly from the original paper, with no modifications.
1. **WGAN-GP Loss**:
   The pre-trained reconstruction model $\mathcal{R}$ is fixed during generative training. Batch size is $B$, latent vector $z \sim \mathcal{N}(0,I)$ (standard Gaussian distribution).
   Generator loss (trains the generator to produce realistic scans that fool the critic):
   $$
   \mathcal{L}_G = \frac{1}{B} \sum_{b=1}^B \left[ - D\left( \mathcal{R}(G(z^{(b)})) \right) \right]
   $$
   Where $z^{(b)}$ is the latent vector for the b-th sample, $G(z^{(b)})$ is the $K$-channel generated HU views, $\mathcal{R}(G(z^{(b)}))$ is the reconstructed full-range scan, and $D(\cdot)$ is the critic's output (higher values indicate the critic judges the sample as real).
   
   Critic loss (trains the critic to distinguish real vs synthetic scans, with gradient penalty to enforce Lipschitz constraint):
   $$
   \mathcal{L}_D = \frac{1}{B} \sum_{b=1}^B \left[ D(\mathcal{R}(G(z^{(b)}))) - D(X^{(b)}) + \lambda_{GP} \cdot \left( \|\nabla_{\bar{X}^{(b)}} D(\bar{X}^{(b)})\|_2 - 1 \right)^2 \right]
   $$
   Where $X^{(b)}$ is the real full-range scan, $\bar{X}^{(b)}$ is an interpolation between real and generated samples, and $\lambda_{GP}=10$ (standard value for WGAN-GP).

2. **Score-Based Diffusion Model Loss**:
   Follows the standard denoising score matching objective, computed over all $K$ channels. For a real $K$-channel sample $X(0)$, add Gaussian noise to get $X(t) = X(0) + \sigma(t) \cdot \varepsilon$, where $\varepsilon \sim \mathcal{N}(0,I)$ and $\sigma(t)$ is the noise level at timestep $t$. The true score function (gradient of the log data distribution) is $\nabla_{X(t)} \log p_{0t}(X(t)|X(0)) = -\frac{\varepsilon}{\sigma(t)}$.
   Loss function (trains the U-Net to predict the score function):
   $$
   \mathcal{L}_{DM} = \frac{1}{B} \sum_{b=1}^B \left[ \| s(X(t^{(b)}), t^{(b)}) + \varepsilon^{(b)} / \sigma(t^{(b)}) \|_2^2 \right]
   $$
   Where $s(\cdot)$ is the score-prediction U-Net, which takes the noisy image $X(t^{(b)})$ and timestep $t^{(b)}$ as input. The loss is the mean squared error between the predicted score and the true score.

3. **VQVAE Loss**:
   The total loss has three components: pre-reconstruction loss, post-reconstruction loss, and quantization loss.
   First, compute per-interval weights to balance loss across HU intervals (intervals with more tissue content get higher weight, with a minimum weight to avoid zero weight for sparse intervals):
   $$
   \{w_k\} = \mathrm{softmax}\left( \left\{ \max\left( \frac{\#[X_k > 0]}{|X_k|}, w_{min} \right), \quad k \in \{1,2,...,K\} \right\} \right)
   $$
   Where $\#[X_k >0]$ is the number of non-zero pixels in $X_k$, $|X_k|$ is the total number of pixels, and $w_{min}=0.15$.
   
   Pre-reconstruction loss $\mathcal{L}_{Pre-Rec}$: Computed on the $K$ generated HU interval views vs real views, weighted by $w_k$. Combines three complementary loss terms: pixel-wise MSE, structural 1-SSIM loss, and perceptual loss from a pre-trained ResNet-50 on the RadImageNet medical imaging dataset:
   $$
   \mathcal{L}_{Pre-Rec} = \frac{1}{B} \sum_{b=1}^B \sum_{k=1}^K w_k^{(b)} \cdot \left[ \lambda_{Pre-MSE} \cdot \mathcal{L}_{MSE}(X_k^{(b)}, \hat{X}_k^{(b)}) + \lambda_{SSIM} \cdot \mathcal{L}_{SSIM}(X_k^{(b)}, \hat{X}_k^{(b)}) + \lambda_{Pre-RIN} \cdot \mathcal{L}_{RIN}(X_k^{(b)}, \hat{X}_k^{(b)}) \right]
   $$
   Where $\hat{X}_k^{(b)}$ is the generated k-th HU interval view, and hyperparameters are set to $\lambda_{Pre-MSE}=1.0$, $\lambda_{SSIM}=0.1$, $\lambda_{Pre-RIN}=1.0$.
   
   Post-reconstruction loss $\mathcal{L}_{Post-Rec}$: Computed on the full-range reconstructed scan $\hat{X} = \mathcal{R}(\{\hat{X}_k\})$ vs the real full-range scan $X$, ensuring the generated HU views combine to a realistic full scan:
   $$
   \mathcal{L}_{Post-Rec} = \frac{1}{B} \sum_{b=1}^B \left[ \lambda_{Post-MSE} \cdot \mathcal{L}_{MSE}(X^{(b)}, \hat{X}^{(b)}) + \lambda_{Post-RIN} \cdot \mathcal{L}_{RIN}(X^{(b)}, \hat{X}^{(b)}) \right]
   $$
   Hyperparameters are set to $\lambda_{Post-MSE}=0.1$, $\lambda_{Post-RIN}=0.25$.
   
   Quantization loss $\mathcal{L}_{VQ}$: Standard VQVAE loss that aligns the encoder output with the codebook entries, and updates the codebook to match the encoder output. The $\mathrm{sg}[\cdot]$ operator is the stop-gradient operator, which blocks gradient flow through that term during backpropagation:
   $$
   \mathcal{L}_{VQ} = \lambda_{VQ} \cdot \frac{1}{B} \sum_{b=1}^B \left[ \| \mathrm{sg}[E(X^{(b)})] - z_q^{(b)} \|_2^2 + \| \mathrm{sg}[z_q^{(b)}] - E(X^{(b)}) \|_2^2 \right]
   $$
   Where $E(X^{(b)})$ is the encoder output, $z_q^{(b)}$ is the quantized latent vector from the codebook, and $\lambda_{VQ}=1.0$. The first term updates the codebook to match the encoder output, while the second term updates the encoder to produce vectors close to codebook entries.
   
   Total VQVAE loss:
   $$
   \mathcal{L}_{VQVAE} = \mathcal{L}_{Pre-Rec} + \mathcal{L}_{Post-Rec} + \mathcal{L}_{VQ}
   $$

# 5. Experimental Setup
## 5.1. Datasets
- **Dataset Source**: LIDC-IDRI (Lung Image Database Consortium and Image Database Resource Initiative), a publicly available dataset hosted on The Cancer Imaging Archive (TCIA).
- **Scale & Characteristics**: 1018 thoracic CT scans from patients with lung nodules, each stored as a 3D volume of size $S \times 512 \times 512$, where $S$ is the number of axial slices per scan (varies per patient). Only mid-chest slices between Z=30 and Z=90 (containing lung tissue) are used, resulting in ~250,000 2D slices for training and testing.
- **Split**: 80% of patient volumes are allocated to the training set, 20% to the test set, with no patient overlap between splits to avoid data leakage.
- **Preprocessing**: Full HU range clipped to [-1000, 1000] (covers all relevant lung tissue types), min-max scaled to [0,1]. The 4 HU interval views are generated as described in Section 4.2.1.
- **Suitability**: LIDC-IDRI is the de facto benchmark dataset for lung CT research, widely used for CAD model development and generative modeling evaluation, so results are directly comparable to prior work. It includes expert annotations for lung nodules, enabling future validation of synthetic data for downstream nodule detection tasks.

## 5.2. Evaluation Metrics
All metrics are explained per the required three-part structure:
1. **Mean Squared Error (MSE)**
   - **Conceptual Definition**: Measures the average squared difference between predicted and ground truth pixel values, quantifying pixel-level intensity error. Lower values indicate better reconstruction accuracy.
   - **Formula**:
     $$
     \mathrm{MSE} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2
     $$
   - **Symbol Explanation**: $N$ = total number of pixels, $x_i$ = ground truth pixel value, $\hat{x}_i$ = predicted pixel value.

2. **Fréchet Inception Distance (FID)**
   - **Conceptual Definition**: Measures the distance between the distribution of real images and synthetic images, using 2048-dimensional feature vectors extracted from a pre-trained Inception-v3 model trained on ImageNet. Lower values indicate the synthetic distribution is more similar to the real distribution, with higher fidelity.
   - **Formula**:
     $$
     \mathrm{FID} = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})
     $$
   - **Symbol Explanation**: $\mu_r$ = mean of Inception-v3 feature vectors for real images, $\mu_g$ = mean of feature vectors for synthetic images, $\Sigma_r$ = covariance matrix of real features, $\Sigma_g$ = covariance matrix of synthetic features, $\mathrm{Tr}(\cdot)$ = trace of a matrix (sum of diagonal elements).

3. **Maximum Mean Discrepancy (MMD)**
   - **Conceptual Definition**: Non-parametric measure of the difference between two probability distributions, computed using kernel functions on feature vectors. Lower values indicate better alignment between the real and synthetic distributions. This work uses Inception-v3 features and a Gaussian kernel.
   - **Formula**:
     $$
     \mathrm{MMD}(P, Q) = \left\| \mathbb{E}_{x \sim P} [\phi(x)] - \mathbb{E}_{y \sim Q} [\phi(y)] \right\|_{\mathcal{H}}^2
     $$
   - **Symbol Explanation**: $P$ = real data distribution, $Q$ = synthetic data distribution, $\phi(\cdot)$ = feature mapping function (Inception-v3 feature extractor), $\mathcal{H}$ = reproducing kernel Hilbert space (RKHS) corresponding to the Gaussian kernel, $\mathbb{E}$ = expectation operator.

4. **Precision (Generative Model)**
   - **Conceptual Definition**: Fraction of synthetic images that lie within the real data distribution (i.e., are realistic and not outliers). Higher values indicate a larger share of generated images are clinically plausible.
   - **Formula (from Kynkäänniemi et al. 2019)**:
     $$
     \mathrm{Precision} = \frac{|\{ \tilde{x} \sim Q : \exists x \sim P, \|\phi(\tilde{x}) - \phi(x)\|_2 \leq r_x \}|}{|\{\tilde{x} \sim Q\}|}
     $$
   - **Symbol Explanation**: $\tilde{x}$ = synthetic image from distribution $Q$, $x$ = real image from distribution $P$, $r_x$ = distance to the k-th nearest neighbor of $\phi(x)$ in the real feature manifold.

5. **Recall (Generative Model)**
   - **Conceptual Definition**: Fraction of the real data distribution that is covered by the synthetic data distribution, quantifying the diversity of synthetic outputs. Higher values indicate the generative model captures more of the real data's anatomical variability.
   - **Formula (from Kynkäänniemi et al. 2019)**:
     $$
     \mathrm{Recall} = \frac{|\{ x \sim P : \exists \tilde{x} \sim Q, \|\phi(x) - \phi(\tilde{x})\|_2 \leq r_{\tilde{x}} \}|}{|\{x \sim P\}|}
     $$
   - **Symbol Explanation**: $r_{\tilde{x}}$ = distance to the k-th nearest neighbor of $\phi(\tilde{x})$ in the synthetic feature manifold.

6. **Multi-Scale Structural Similarity Index Measure (MS-SSIM)**
   - **Conceptual Definition**: Measures the structural similarity between two images at multiple spatial scales, with values ranging from 0 (no similarity) to 1 (identical). For evaluating sample diversity, average MS-SSIM is computed across all pairs of generated images: lower values indicate higher diversity (less redundancy between generated samples).
   - **Formula**:
     $$
     \mathrm{MS-SSIM}(x, \hat{x}) = \prod_{j=1}^M l_j(x, \hat{x})^{\alpha_j} \cdot c_j(x, \hat{x})^{\beta_j} \cdot s_j(x, \hat{x})^{\gamma_j}
     $$
   - **Symbol Explanation**: $M$ = number of spatial scales, $l_j$ = luminance comparison at scale j, $c_j$ = contrast comparison, $s_j$ = structure comparison, $\alpha_j, \beta_j, \gamma_j$ = weights for each component (all set to 1 in standard implementations).

## 5.3. Baselines
The paper uses full-range generative models as baselines (trained to generate full-range CT scans directly without HU decomposition):
1. **Full-range WGAN-GP**: Standard WGAN-GP architecture trained to output single-channel full-range CT scans.
2. **Full-range Score-based Diffusion Model**: Standard U-Net diffusion model trained to generate single-channel full-range CT scans.
3. **Full-range VQVAE**: Standard VQVAE architecture trained to output single-channel full-range CT scans.
   These baselines are representative of the current state-of-the-art for medical image synthesis, so comparing to them fairly isolates the performance benefit of the HU decomposition approach.

# 6. Results & Analysis
## 6.1. Core Results Analysis
### Reconstruction Model Performance
The following are the results from Table 4 of the original paper, evaluating reconstruction models on their ability to recover full-range CT scans from real HU interval views:

<table>
<thead>
<tr>
<th>Reconstruction Model</th>
<th>MSE (×10⁻⁴) ↓</th>
<th>FID ↓</th>
<th>MMD (×10⁻²) ↓</th>
<th>Precision ↑</th>
<th>Recall ↑</th>
<th>MS-SSIM ↑</th>
</tr>
</thead>
<tbody>
<tr>
<td>MLP0</td>
<td>4.68</td>
<td>44.9</td>
<td>3.29</td>
<td>0.969</td>
<td>0.989</td>
<td>0.962</td>
</tr>
<tr>
<td>MLP4</td>
<td>4.74</td>
<td>44.6</td>
<td>3.21</td>
<td>0.974</td>
<td>0.988</td>
<td>0.962</td>
</tr>
<tr>
<td>MLP4×4</td>
<td>4.74</td>
<td>44.6</td>
<td>3.21</td>
<td>0.974</td>
<td>0.988</td>
<td>0.962</td>
</tr>
<tr>
<td>MLP4×4×4</td>
<td>4.46</td>
<td>41.9</td>
<td>2.93</td>
<td>0.977</td>
<td>0.995</td>
<td>0.964</td>
</tr>
<tr>
<td>CNN3</td>
<td>4.62</td>
<td>40.4</td>
<td>2.66</td>
<td>0.989</td>
<td>0.993</td>
<td>0.962</td>
</tr>
<tr>
<td>CNN7</td>
<td>4.54</td>
<td>40.1</td>
<td>2.68</td>
<td>0.988</td>
<td>0.993</td>
<td>0.962</td>
</tr>
<tr>
<td>CNN11</td>
<td>4.53</td>
<td>40.7</td>
<td>2.79</td>
<td>0.987</td>
<td>0.991</td>
<td>0.962</td>
</tr>
<tr>
<td>CNN3×3</td>
<td>3.45</td>
<td>34.8</td>
<td>2.15</td>
<td>0.995</td>
<td>0.996</td>
<td>0.979</td>
</tr>
<tr>
<td>CNN3×3×3</td>
<td>2.49</td>
<td>26.0</td>
<td>1.56</td>
<td>0.999</td>
<td>1.000</td>
<td>0.985</td>
</tr>
</tbody>
</table>

Key observations:
- CNN models outperform MLP models significantly, as they capture local spatial correlations between voxels that voxel-wise MLPs miss.
- The optimal reconstruction model is CNN3×3×3 (3 layers of 3×3 convolutions), which achieves near-perfect precision (0.999) and recall (1.000), very low MSE (2.49e-4) and FID (26.0), proving full-range CT can be reconstructed from 4 HU views with negligible information loss.
  Qualitative reconstruction examples are shown below:

  ![该图像是多个肺部CT图像的比较示意图，展示了原始图像与使用不同模型（MLP、CNN）生成的图像。每个示例展示了不同的生成效果，从而评估模型性能与质量。](images/11.jpg)
  *该图像是多个肺部CT图像的比较示意图，展示了原始图像与使用不同模型（MLP、CNN）生成的图像。每个示例展示了不同的生成效果，从而评估模型性能与质量。*

### Generative Model Performance
FID results per HU interval from Table 5 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2">Model Type</th>
<th rowspan="2">Approach</th>
<th rowspan="2">Reconstruction Model</th>
<th colspan="5">FID per HU Interval</th>
</tr>
<tr>
<th>Full-range</th>
<th>[-950,-700]</th>
<th>[-500,-200]</th>
<th>[30, 70]</th>
<th>[100, 1000]</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3">WGAN-GP</td>
<td>Baseline</td>
<td>-</td>
<td>141.8</td>
<td>117.5</td>
<td>131.5</td>
<td>157.2</td>
<td>177.5</td>
</tr>
<tr>
<td rowspan="2">Multi-channel</td>
<td>MLP0</td>
<td>211.3</td>
<td>196.1</td>
<td>137.7</td>
<td>134.7</td>
<td>219.4</td>
</tr>
<tr>
<td>CNN3×3×3</td>
<td>184.9</td>
<td>145.0</td>
<td>121.9</td>
<td>138.2</td>
<td>205.9</td>
</tr>
<tr>
<td rowspan="3">Score-based DM</td>
<td>Baseline</td>
<td>-</td>
<td>66.8</td>
<td>81.3</td>
<td>56.3</td>
<td>85.6</td>
<td>81.2</td>
</tr>
<tr>
<td rowspan="2">Multi-channel</td>
<td>MLP0</td>
<td>84.9</td>
<td>106.5</td>
<td>134.9</td>
<td>134.5</td>
<td>192.3</td>
</tr>
<tr>
<td>CNN3×3×3</td>
<td>85.9</td>
<td>114.0</td>
<td>140.4</td>
<td>139.7</td>
<td>194.7</td>
</tr>
<tr>
<td rowspan="6">VQVAE</td>
<td>Baseline</td>
<td>-</td>
<td>71.6</td>
<td>95.5</td>
<td>65.2</td>
<td>116.6</td>
<td>105.5</td>
</tr>
<tr>
<td rowspan="2">Multi-channel</td>
<td>MLP0</td>
<td>76.3</td>
<td>75.8</td>
<td>52.9</td>
<td>83.7</td>
<td>74.6</td>
</tr>
<tr>
<td>CNN3×3×3</td>
<td>75.7</td>
<td>77.6</td>
<td>66.1</td>
<td>79.2</td>
<td>75.3</td>
</tr>
<tr>
<td rowspan="2">Multi-decoder</td>
<td>MLP0</td>
<td>70.5</td>
<td>100.0</td>
<td>70.5</td>
<td>86.7</td>
<td>76.3</td>
</tr>
<tr>
<td>CNN3×3×3</td>
<td>67.9</td>
<td>81.4</td>
<td>69.2</td>
<td>77.4</td>
<td>77.2</td>
</tr>
<tr>
<td rowspan="2">Multi-head</td>
<td>MLP0</td>
<td>68.1</td>
<td>96.3</td>
<td>63.6</td>
<td>73.7</td>
<td>71.3</td>
</tr>
<tr>
<td></td>
<td>CNN3×3×3</td>
<td>67.1</td>
<td>77.0</td>
<td>59.4</td>
<td>73.1</td>
<td>71.5</td>
</tr>
</tbody>
</table>

Key observations:
- WGAN-GP and diffusion models perform worse with the multi-channel HU decomposition approach than their full-range baselines, likely because their architectures are not optimized to learn shared structure across multiple channels.
- All VQVAE variants outperform the full-range VQVAE baseline for most HU intervals. The multi-head VQVAE + CNN3×3×3 achieves the best full-range FID of 67.1, a 6.2% improvement over the baseline FID of 71.6, matching the abstract's claim.
- For all VQVAE variants, using the CNN3×3×3 reconstruction model gives better performance than MLP0, consistent with the reconstruction model results.
  Qualitative samples from VQVAE methods are shown below:

  ![Figure 12: Full-range CT and HU-windowed samples obtained from the proposed VQVAE methods and $\\mathrm { C N N } _ { 3 \\times 3 \\times 3 }$ reconstruction model.](images/12.jpg)
  *该图像是图表，展示了使用 VQVAE 方法及 `ext{CNN}_{3 imes 3 imes 3}` 重建模型生成的全范围 CT 图像和不同 HU 窗口 [-950, -700]、[-500, -200]、[30, 70] 和 [100, 1000] 的样本对比。*

The samples are anatomically plausible, with consistent structure across all HU intervals, confirming the method preserves global anatomical consistency while generating tissue-specific textures.

### Downstream Task Validation
The paper runs a pre-trained lung segmentation model (trained exclusively on real CT scans) on synthetic scans from the multi-head VQVAE + CNN3×3×3 pipeline. Results are shown below:

![Figure 14: Synthetic full-range CT images sampled from the multi-head VQVAE and $\\mathrm { C N N } _ { 3 \\times 3 \\times 3 }$ reconstruction mode and respective segmentation masks outputted from the segmentation model from \[79\].](images/14.jpg)
*该图像是图14，展示了从多头 VQVAE 生成的合成全范围 CT 图像及其相应的分割掩膜。图像由多组样本和对应的分割结果组成，显示不同 HU 间隔的重建效果。*

The segmentation model produces well-defined, anatomically coherent lung masks from the synthetic scans, proving the synthetic outputs are realistic enough to be processed by downstream CAD models trained on real data, a critical requirement for data augmentation use cases.

## 6.2. Ablation Studies / Parameter Analysis
1. **Reconstruction Model Architecture Ablation**:
   As shown in Table 4, increasing CNN depth improves performance up to 3 layers of 3×3 convolutions (CNN3×3×3). Deeper networks or larger kernels do not yield further gains, making CNN3×3×3 the optimal trade-off between model size and performance.
2. **VQVAE Architecture Variant Ablation**:
   - Multi-channel VQVAE: Same parameter count as the full-range baseline (48.3M), but limited performance gains.
   - Multi-decoder VQVAE: Highest parameter count (68.9M, +42% over baseline), good performance but inconsistent across HU intervals.
   - Multi-head VQVAE: Only 1M additional parameters compared to baseline (49.3M, +2% over baseline), best overall performance across all metrics, achieving the optimal complexity-performance balance.
3. **Reconstruction Model Choice Ablation**:
   For all VQVAE generative variants, using the CNN3×3×3 reconstruction model consistently outperforms using MLP0, with average FID improvements of 5-8% across all intervals.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces HUydra, a novel clinically aligned framework for full-range lung CT synthesis that decomposes the generation task into separate HU interval sub-tasks, mirroring standard radiological HU windowing practice. Key conclusions are:
1. Full-range CT scans can be reconstructed with near-perfect accuracy (99.9% precision, 100% recall) from only 4 HU interval views using a small 3-layer CNN reconstruction model, with negligible information loss.
2. The HU decomposition approach delivers consistent performance gains for VQVAE models, with the multi-head VQVAE variant achieving a 6.2% FID improvement over full-range baselines, plus superior MMD, precision, recall, and sample diversity, while only adding 1M parameters to the baseline VQVAE.
3. Synthetic scans from the pipeline are realistic enough to be processed by downstream CAD models (e.g., lung segmentation) trained on real data, making them suitable for data augmentation to address the critical data scarcity bottleneck for lung CAD systems.
4. The framework is highly interpretable for clinicians, as it enables validation of synthetic outputs at both the individual tissue (HU interval) level and full scan level, unlike black-box end-to-end generative models.
5. The modular design supports dynamic generation of custom HU windows on demand, and can be extended to other medical imaging modalities where intensity-based decomposition is feasible.
## 7.2. Limitations & Future Work
### Stated Limitations
1. The current work uses 2D axial slices, not 3D CT volumes, so it does not capture inter-slice anatomical consistency.
2. The 4 HU intervals are fixed; adaptive interval selection based on task (e.g., nodule detection vs lung disease classification) could further improve performance.
3. Minor localized imperfections remain in anatomically complex regions (e.g., lung boundaries) in synthetic scans.
### Suggested Future Work
1. Extend the framework to 3D CT volume synthesis, to capture full 3D anatomical consistency.
2. Explore adaptive HU interval decomposition tailored to specific downstream tasks.
3. Improve boundary modeling for complex anatomical regions to eliminate remaining imperfections.
4. Extend the framework to other modalities including MRI and X-ray, where intensity windowing is commonly used.
5. Conduct formal validation of synthetic data for downstream CAD model training, quantifying the performance improvement from data augmentation with HUydra synthetic scans.
6. Explore multi-head diffusion model architectures to leverage the HU decomposition approach for diffusion models, which currently underperform in the multi-channel setting.
## 7.3. Personal Insights & Critique
### Key Inspirations
The paper's core insight of aligning generative modeling with existing clinical workflows is highly impactful, as it addresses two of the largest barriers to clinical adoption of generative medical AI: lack of interpretability and clinician trust. By mirroring the HU windowing process radiologists already use, the framework produces outputs that clinicians can easily validate, rather than forcing clinicians to adapt to black-box model outputs. Additionally, the task decomposition strategy of breaking a complex generative task into simpler homogeneous sub-tasks is generalizable to many other domains beyond medical imaging, including high-dynamic-range image synthesis, multi-spectral image generation, and video generation.
### Potential Improvements
1. The paper finds that WGAN-GP and diffusion models do not benefit from the HU decomposition approach, but does not investigate the root cause. Future work could explore multi-head diffusion model architectures optimized for multi-channel interval generation, as diffusion models are known for higher fidelity than VQVAEs and could deliver even better performance with appropriate architectural adaptations.
2. The 4 HU intervals used are not exactly aligned with standard clinical windows (e.g., the standard lung window is [-700, -600] HU, not [-950, -700] HU). Aligning intervals exactly with clinical standards would further improve clinical adoptability and may improve performance by ensuring each interval exactly matches tissue types radiologists are trained to interpret.
3. The paper mentions a Visual Turing Test with clinicians as a contribution, but does not present detailed quantitative results (e.g., percentage of synthetic scans judged as real, clinician ratings of diagnostic utility). Including these results would further strengthen the clinical validation of the method.
4. The paper mentions reduced computational cost, but does not provide a detailed quantitative comparison of training/inference time and memory usage between the proposed framework and full-range baselines. Including these metrics would better quantify the efficiency gains of the decomposition approach.
   Overall, HUydra represents a highly promising, clinically grounded advance in generative medical imaging, with significant potential to address the critical data scarcity bottleneck for lung cancer CAD systems.