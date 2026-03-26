# 1. Bibliographic Information
## 1.1. Title
The paper's title is *VTAM: Video-Tactile-Action Models for Complex Physical Interaction Beyond VLAs*. Its central topic is the development of a multimodal robotic control framework that integrates visual, tactile, and action modeling to enable high-performance execution of contact-rich manipulation tasks that are unsolvable by existing vision-only Vision-Language-Action (VLA) or Video-Action Model (VAM) systems.
## 1.2. Authors
The authors and their affiliations are:
- Haoran Yuan, Weigang Yi, Yuchen Mo, Jiashi Yin, Xinzhuo Li, Xiangyu Zeng: Carnegie Mellon University / University of Illinois Urbana-Champaign / Stanford University
- Zhenyu Zhang, Wendi Chen: Shanghai Jiao Tong University
  *Note: Full author affiliation details are partially redacted in the provided preprint, but the core research team spans leading robotics and AI research institutions in the US and China.*
## 1.3. Journal/Conference
The paper is published as a preprint on arXiv, a widely used open-access repository for pre-publication research in computer science, robotics, and artificial intelligence. As a 2026 preprint, it has not yet been peer-reviewed for conference or journal publication, but its contributions align with top robotics venues such as ICRA (International Conference on Robotics and Automation), IROS (International Conference on Intelligent Robots and Systems), and CoRL (Conference on Robot Learning).
## 1.4. Publication Year
The paper was published publicly on arXiv on March 24, 2026 (UTC time).
## 1.5. Abstract
The paper addresses the critical limitation of existing vision-only Video-Action Models (VAMs): their poor performance on contact-rich manipulation tasks, as vision alone cannot reliably encode fine-grained force modulation and occluded contact transitions. The authors propose the **Video-Tactile Action Model (VTAM)**, a multimodal world modeling framework that augments a pretrained video transformer with tactile input streams via lightweight modality transfer finetuning, eliminating the need for tactile-language paired data or independent tactile pretraining. A novel tactile regularization loss (virtual force prediction) is introduced to prevent visual latent dominance and stabilize multimodal fusion. Experiments show VTAM achieves an average 90% success rate across contact-rich tasks, outperforming the $\pi_{0.5}$ VLA baseline by 80% on the fragile potato chip pick-and-place task. The work demonstrates that tactile feedback is essential to correct visual estimation errors in embodied action models.
## 1.6. Original Source Link
The paper is available as an arXiv preprint at:
- Abstract page: https://arxiv.org/abs/2603.23481v1
- Full PDF: https://arxiv.org/pdf/2603.23481v1
  *Publication status: Unpeer-reviewed preprint.*

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Existing vision-only VLA and VAM systems excel at long-horizon tasks that rely on high-level visual semantic reasoning, but fail catastrophically on contact-rich manipulation tasks (e.g., grasping fragile objects, peeling deformable vegetables, wiping surfaces) for two key reasons:
1. Critical contact states (e.g., slip, force magnitude, contact onset) are often occluded from camera views or too fine-grained to be captured by visual tokens.
2. Current tactile integration methods for VLAs treat tactile as a supplementary input fused reactively at late stages of the pipeline, leading to **modality collapse**: dominant visual gradients suppress tactile signals during training, so the model ignores tactile feedback even when it is critical for task success.

### Importance of the Problem
Contact-rich manipulation is essential for real-world robotic deployment in homes, factories, healthcare, and food processing. Without reliable force-aware control, robots cannot safely interact with fragile, deformable, or slippery objects, limiting their real-world utility.

### Research Gap in Prior Work
Prior tactile-augmented VLA approaches suffer from three key limitations:
1. They project tactile embeddings into vision-language semantic spaces optimized for static scene understanding, not dynamic physical contact prediction.
2. They require large-scale annotated tactile data or expensive external force-torque sensors for supervision.
3. They fail to address modality collapse without imposing hardware-specific constraints.

### Innovative Entry Point
The paper's key innovative idea is to integrate tactile sensing as a primary modality into a predictive video world model, rather than fusing it reactively at downstream layers. By jointly predicting future visual and tactile streams, the model learns temporally consistent contact dynamics without explicit semantic annotations, and a lightweight virtual force regularization objective prevents modality collapse without requiring external force sensors.
## 2.2. Main Contributions / Findings
The paper's four core contributions are:
1. **VTAM Framework**: A generalist visuotactile world action model that integrates high-resolution tactile sensing with visual observations in a predictive video backbone for contact-rich manipulation.
2. **Joint Visuotactile Prediction**: A shared latent space framework that forecasts future visual and tactile streams, enabling the model to learn contact dynamics without explicit semantic annotations of contact events.
3. **Virtual Force Regularization**: A deformation-aware auxiliary objective that mitigates modality collapse during training by deriving a virtual force signal directly from tactile image deformation, eliminating the need for external force-torque sensors.
4. **Empirical Validation**: State-of-the-art performance on three challenging contact-rich tasks (potato chip pick-and-place, cucumber peeling, whiteboard wiping) with large improvements over vision-only and naive tactile integration baselines.

### Key Findings
- Predictive joint modeling of visual and tactile dynamics in a shared world model backbone substantially outperforms late-stage tactile fusion.
- Virtual force regularization is critical to prevent visual modality dominance, with performance dropping 80% when this objective is removed.
- Tactile feedback corrects visual estimation errors in occluded contact scenarios, enabling stable force control for fragile and deformable object manipulation.

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
To understand this paper, beginners need to master the following core concepts, explained in plain language:
### 3.1.1. Vision-Language-Action (VLA) Model
A VLA is an end-to-end AI model that maps two inputs (visual observations of the environment, natural language task instructions) to robot motor actions. VLAs are trained on large-scale vision-language datasets to generalize across diverse tasks and environments, but they are traditionally optimized for semantic understanding rather than fine-grained physical contact control.
### 3.1.2. Video-Action Model (VAM)
A VAM is a predictive model that learns implicit world dynamics from raw video streams to generate temporally consistent action sequences for robotic control, without requiring explicit language instructions. VLAs/VAMs rely almost exclusively on visual input, making them prone to failure in occluded contact-rich scenarios.
### 3.1.3. Generative World Model
A generative world model forecasts future environment states (e.g., video frames, tactile readings) conditioned on past observations and robot actions. World models enable robots to plan ahead and anticipate failure modes before they occur, which is critical for delicate manipulation tasks.
### 3.1.4. Flow Matching
Flow matching is a generative modeling technique that learns a continuous velocity vector field to map simple noise distributions to target data distributions (e.g., valid action sequences, video frames). It is an alternative to diffusion models with faster training and inference, making it well-suited for real-time robotic control. The core flow matching objective minimizes the mean squared error between the predicted velocity field and the ground truth velocity field that transforms noise to target data.
### 3.1.5. Self-Attention and Cross-Attention
Attention mechanisms allow neural networks to focus on relevant parts of input data. The standard self-attention formula is:
$$
Attention(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
Where:
- $Q$ (Query): Linear projection of the input embedding, representing what the current token is looking for
- $K$ (Key): Linear projection of the input embedding, representing what each input token offers
- $V$ (Value): Linear projection of the input embedding, representing the content of each input token
- $d_k$: Dimension of the Key vectors, used as a scaling factor to prevent excessively large dot products from making the softmax output too sharp (all weight assigned to a single token)
  *Self-attention* computes attention across tokens from the same input modality, while *cross-attention* computes attention across tokens from different modalities (e.g., visual and tactile tokens).
### 3.1.6. GelSight Tactile Sensor
A GelSight is a high-resolution optical tactile sensor that uses a transparent, deformable elastomer surface mounted over a camera. When the elastomer presses against an object, it deforms, and the camera captures these deformation patterns, which can be used to infer contact forces, slip, surface texture, and object shape with very high precision.
### 3.1.7. Modality Collapse
Modality collapse is a common failure mode in multimodal model training, where the model learns to rely almost entirely on one dominant modality (here, vision) and ignores input from other modalities (here, tactile) because the dominant modality has stronger, more consistent gradients during training.
## 3.2. Previous Works
The paper categorizes related work into three core research areas:
### 3.2.1. Vision-Language-Action Models
Modern VLAs such as RT-2, OpenVLA, and $\pi_{0.5}$ leverage large-scale vision-language pretraining to enable generalist robot control across diverse tasks. However, these vision-only models fail on contact-rich tasks where critical contact states are occluded or not visible to cameras. Recent extensions such as GeoVLA add 3D geometric priors, but they still do not address the core limitation of missing force feedback.
### 3.2.2. Generative World Models for Robotics
Generative world models such as DreamZero, UWM, and Genie Envisioner forecast future video frames to support planning and policy learning, achieving strong zero-shot generalization across robotic tasks. However, these models rely exclusively on visual prediction, so they cannot anticipate contact failure modes (e.g., slip, deformation) that are not visible to cameras.
### 3.2.3. Tactile Integration in Robotic Learning
Prior tactile integration approaches for VLAs (e.g., Tactile-VLA, ForceVLA) either inject tactile embeddings as additional semantic tokens into the vision-language latent space, or fuse tactile features with visual features at the downstream policy layer. These approaches suffer from two key limitations:
1. They require large-scale annotated tactile data or expensive external force-torque sensors for supervision.
2. They are prone to modality collapse, as visual gradients dominate during training and suppress tactile signals. Existing mitigations for modality collapse rely on hardware-specific force controllers, limiting their generality.
## 3.3. Technological Evolution
The evolution of embodied AI models for robotic manipulation follows this timeline:
1. **2020-2023**: Vision-only VLAs emerge as the dominant paradigm for generalist robot control, leveraging large-scale internet vision-language data for cross-task generalization.
2. **2024-2025**: Generative world models are integrated into VLA/VAM pipelines to add predictive reasoning capabilities, improving long-horizon task performance.
3. **2025-early 2026**: Tactile sensing is added to VLAs via late fusion, but performance remains poor on contact-rich tasks due to modality collapse and lack of predictive contact modeling.
4. **2026 (this work)**: Tactile sensing is integrated as a primary modality into the predictive world model backbone, with regularization to prevent modality collapse, enabling state-of-the-art contact-rich manipulation performance.
## 3.4. Differentiation Analysis
Compared to prior tactile-integrated VLA approaches, VTAM has three core differentiating innovations:
1. **Predictive vs. Reactive Fusion**: Unlike prior work that fuses tactile reactively at downstream layers, VTAM embeds tactile into the generative world model backbone to jointly predict future visual and tactile streams, enabling the model to learn temporally consistent contact dynamics.
2. **No External Force Sensors**: The virtual force regularization objective is derived directly from tactile image deformation, eliminating the need for expensive external force-torque sensors for supervision.
3. **No Tactile Pretraining/Paired Data**: VTAM uses lightweight modality transfer finetuning on a pretrained video backbone, requiring no independent tactile pretraining or tactile-language paired data.

# 4. Methodology
## 4.1. Principles
The core idea of VTAM is to treat tactile sensing as a primary input modality rather than a supplementary signal, integrating it into a predictive video world model to learn joint visuotactile dynamics. The method is designed around two core intuitions:
1. Joint predictive modeling of visual and tactile streams in a shared latent space enables the model to learn contact dynamics without explicit semantic annotations of contact events.
2. A lightweight deformation-aware regularization objective can prevent modality collapse by ensuring tactile gradients remain influential during training, without requiring external force sensors.
   The VTAM architecture overview is shown in Figure 2 from the original paper:

   ![Figure 2: VTAM Overview. A pretrained video backbone jointly models multi-view visual and tactile latents via alternating intra-view and cross-view attention. The resulting multimodal representation is injected into a conditional action diffusion head to predict action, virtual force, and proprioceptive state.](images/2.jpg)
   *该图像是示意图，展示了VTAM模型的结构。图中左侧为视频基础模型，整合来自不同视角的自注意力模块；右侧为动作扩散模型，采用多视角交叉注意力机制。该模型输入包括过往的动作和状态信息，预测未来的动作、状态及力的变化。*

## 4.2. Core Methodology In-depth
The VTAM framework is composed of three core components, explained step-by-step below with all formulas exactly as presented in the original paper.
### 4.2.1. Vision-Tactile Latent World Modeling via Multi-View Diffusion
This component builds a shared latent space for multi-view visual and tactile inputs, using a pretrained video Variational Autoencoder (VAE) and alternating intra-view/cross-view attention to model joint visuotactile dynamics.
#### Step 1: Latent Encoding
Input frames from three views are encoded into a shared latent space using a pretrained video VAE encoder $E$ (optimized for reconstruction, so it preserves fine-grained spatial and motion patterns):
$$
\mathbf{z}_t^v = E\left(\mathbf{I}_t^v\right), \quad v \in \{1,2,3\}
$$
Where:
- $\mathbf{I}_t^v$: Input frame at timestep $t$ from view $v$
- $v=1$: Third-person RGB camera view
- $v=2$: First-person RGB camera view mounted on the robot gripper
- $v=3$: GelSight tactile stream frame
- $\mathbf{z}_t^v$: Continuous latent representation of the input frame
- $E$: Pretrained video VAE encoder

#### Step 2: Alternating Attention Modeling
The latent representations are processed through $B$ transformer blocks with alternating intra-view self-attention and cross-view attention to model both intra-modality spatial structure and inter-modality interactions:
1. **Intra-View Self-Attention**: For each block $b$, self-attention is first applied independently to each modality to capture spatial structures within each view:
   $$
\tilde{\mathbf{z}}_{t,b}^v = \mathrm{SelfAttention}\left(\mathbf{z}_{t,b-1}^v\right) \quad \forall v \in \{1,2,3\}
$$
Where $\mathbf{z}_{t,b-1}^v$ is the latent representation for view $v$ at timestep $t$ from the previous block `b-1`, and $\tilde{\mathbf{z}}_{t,b}^v$ is the updated latent after intra-view self-attention.
2. **Cross-View Attention**: The updated latent tokens from all three views are concatenated, and cross-view attention is applied to model inter-modal interactions between visual and tactile signals:
   $$
\mathbf{Z}_b = \mathrm{CrossViewAttention}\left(\mathrm{Concat}\left(\tilde{\mathbf{z}}_{t,b}^1, \tilde{\mathbf{z}}_{t,b}^2, \tilde{\mathbf{z}}_{t,b}^3\right)\right)
$$
Where $\mathbf{Z}_b = \{\mathbf{z}_{t,b}^1, \mathbf{z}_{t,b}^2, \mathbf{z}_{t,b}^3\}$ is the set of updated latents for all views at block $b$.
This alternating attention structure is repeated across all $B$ blocks, building a dense joint visuotactile representation of the environment state and contact dynamics.

### 4.2.2. Deformation-Aware Regularization via Virtual Force Prediction
This component addresses modality collapse by adding an auxiliary supervision signal derived directly from tactile image deformation, ensuring tactile gradients remain influential during training without requiring external force sensors.
#### Step 1: Virtual Force Calculation
A virtual force proxy is computed from the tactile image deformation field, no external force sensor is required:
1. First, compute the dense optical flow $\mathbf{u}_t = (u_x, u_y)$ between a no-contact reference tactile frame $\mathbf{I}_0$ (when the GelSight is not touching any object) and the current tactile frame $\mathbf{I}_t$. The optical flow measures the displacement of each pixel in the tactile image caused by deformation of the GelSight elastomer when it contacts an object.
2. Derive the 3D virtual force vector $\boldsymbol{F}_t^v = [f_x, f_y, f_z]^T$ directly from the optical flow field:
   $$
f_x = \mathbb{E}\big[u_x\big], \quad f_y = \mathbb{E}\big[u_y\big], \quad f_z = \mathbb{E}\big[\boldsymbol{\nabla} \cdot \boldsymbol{u}_t\big]
$$
Where:
- $\mathbb{E}$: Spatial expectation (average) over all pixels in the tactile frame
- $f_x, f_y$: Average optical flow in x and y directions, encoding tangential shear forces between the gripper and the object
- $\boldsymbol{\nabla} \cdot \boldsymbol{u}_t$: Divergence of the optical flow field, which measures the outward expansion of the GelSight surface pattern when pressed against an object, encoding normal compression force $f_z$.

#### Step 2: Force Regularization Loss
The virtual force is added as an additional prediction target in the action head's flow matching objective, binding control gradients to tactile representations. The force regularization loss is:
$$
\mathcal{L}_{\mathrm{force}} = \mathbb{E}\left[ \left\| v_\theta^f(z_t, t \mid c) - v^{*f} \right\|^2 \right]
$$
Where:
- $v_\theta^f$: Predicted velocity field for the virtual force component from the model, parameterized by $\theta$
- $v^{*f}$: Ground truth velocity field for the virtual force component
- $c$: Conditioning signal (current robot proprioceptive state)
- The loss is the mean squared error between predicted and ground truth force velocity fields, ensuring the model learns to predict contact forces accurately from tactile inputs.

### 4.2.3. Optimization Objective
VTAM uses a two-stage training strategy to decouple modality alignment from action learning, preventing unstable convergence:
#### Stage 1: Multi-View Visuo-Tactile Latent Flow Matching
This stage adapts the pretrained visual-only video backbone to model joint visuotactile dynamics, no action supervision is used in this stage:
$$
\mathcal{L}_{\mathrm{stage1}} = \mathbb{E}\left[ \left. \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{z}_t, t) - \mathbf{v}^* \right.^2 \right]
$$
Where:
- $\mathbf{z}_0$: VAE-encoded latent sequence of future multi-view observations (both camera views + tactile stream)
- $\mathbf{v}_{\boldsymbol{\theta}}$: Predicted velocity field for the visuotactile latents
- $\mathbf{v}^*$: Ground truth velocity field for the visuotactile latents
  The loss is only applied to future prediction frames, not initial conditioning frames. This stage establishes a coherent multimodal latent space before any control signals are introduced.

#### Stage 2: Conditional Joint Action-State-Force Denoising
This stage trains the action head to predict robot actions, using the aligned visuotactile representation from Stage 1 as conditioning. The joint denoising target for the flow matching objective is constructed by concatenating three components:
$$
\mathbf{z}_0 = \big[ \mathbf{a}; \mathbf{f}; \mathbf{s} \big]
$$
Where:
- $\mathbf{a} \in \mathbb{R}^7$: Robot action vector, consisting of 6-DoF end-effector pose and 1D gripper width
- $\mathbf{f} \in \mathbb{R}^3$: 3D virtual force vector derived from tactile deformation
- $\mathbf{s} \in \mathbb{R}^{16}$: Robot proprioceptive state vector (joint positions, end-effector velocity, etc.)

  The conditioning signal for the flow matching objective is:
$$
\mathbf{c} = \left[ \mathbf{0}_{10}; \mathbf{s}_t \right]
$$
Where $\mathbf{0}_{10}$ is a 10-dimensional zero vector (zero-padding for the action and force dimensions during conditioning), and $\mathbf{s}_t$ is the current robot proprioceptive state at timestep $t$.

Three separate flow matching losses are computed for the action, state, and force components:
1. Action loss:
   $$
\mathcal{L}_{\mathrm{action}} = \mathbb{E}\left[ \left. \mathbf{v}_{\boldsymbol{\theta}}^{\mathbf{a}}(\mathbf{z}_t, t \mid \mathbf{c}) - \mathbf{v}^{*\mathbf{a}} \right.^2 \right]
$$
Where $\mathbf{v}_{\boldsymbol{\theta}}^{\mathbf{a}}$ is the predicted action velocity field, and $\mathbf{v}^{*\mathbf{a}}$ is the ground truth action velocity field.
2. State loss:
   $$
\mathcal{L}_{\mathrm{state}} = \mathbb{E}\left[ \left. \mathbf{v}_\theta^{\mathbf{s}}(\mathbf{z}_t, t \mid \mathbf{c}) - \mathbf{v}^{*s} \right.^2 \right]
$$
Where $\mathbf{v}_{\boldsymbol{\theta}}^{\mathbf{s}}$ is the predicted proprioceptive state velocity field, and $\mathbf{v}^{*\mathbf{s}}$ is the ground truth state velocity field.
3. Force loss: The $\mathcal{L}_{\mathrm{force}}$ loss defined earlier in Eq. 5.

   The total Stage 2 loss is the weighted sum of the three components:
$$
\mathcal{L}_{\mathrm{stage2}} = \mathcal{L}_{\mathrm{action}} + \lambda_1 \mathcal{L}_{\mathrm{state}} + \lambda_2 \mathcal{L}_{\mathrm{force}}
$$
Where $\lambda_1$ and $\lambda_2$ are loss weighting hyperparameters, set to $\lambda_1 = \lambda_2 = 1$ for all experiments in the paper (all losses are on comparable scales because they use the same flow matching formulation, so no aggressive hyperparameter tuning is required).

# 5. Experimental Setup
## 5.1. Datasets
The authors collected a custom real-world visuotactile dataset for three contact-rich manipulation tasks, using a 6-DoF xArm6 robotic manipulator equipped with a GelSight Mini tactile sensor and two Intel RealSense D455 RGB-D cameras. The dataset includes:
- 100 potato chip pick-and-place trajectories
- 61 cucumber peeling trajectories
- 105 whiteboard wiping trajectories
  All trajectories are collected via manual teleoperation, and include synchronized multi-view RGB streams, tactile deformation images, and robot proprioceptive state data, sampled at 30Hz.
### Sample Data Example
A single chip pick-and-place data sample includes:
- 9 consecutive frames of video from two camera views (third-person and first-person)
- 9 consecutive frames of tactile data from the GelSight sensor
- 54 steps of robot action commands (end-effector pose + gripper width)
- Corresponding robot proprioceptive state measurements
  This custom dataset is chosen because public robotic manipulation datasets rarely include high-resolution synchronized tactile data for force-sensitive contact-rich tasks, making it ideal for validating VTAM's performance.
## 5.2. Evaluation Metrics
The primary evaluation metric used is **task success rate**, defined as follows:
1. **Conceptual Definition**: Success rate measures the proportion of independent task execution trials that fully meet predefined task-specific completion criteria, quantifying the overall functional performance of the robotic policy on the target task.
2. **Mathematical Formula**:
   $$
\text{Success Rate} = \left( \frac{\text{Number of Successful Trials}}{\text{Total Number of Evaluation Trials}} \right) \times 100\%
$$
3. **Symbol Explanation**:
   - *Number of Successful Trials*: Count of test runs that satisfy all task-specific success conditions:
     - Chip pick-and-place: Grasp the chip without breakage, transport it to the target plate, and release it without dropping it during transport
     - Cucumber peeling: Produce a peel strip longer than 10cm, maintaining stable contact with the cucumber surface during peeling
     - Whiteboard wiping: Remove at least 90% of a randomly placed black stain within 3 wiping motions, maintaining consistent contact with the board surface
   - *Total Number of Evaluation Trials*: Total count of independent test runs performed for the task (20 trials per task per model in this paper).
## 5.3. Baselines
VTAM is compared against three representative state-of-the-art baselines:
1. **Genie Envisioner (GE)**: A state-of-the-art vision-only Video-Action Model with an instruction-conditioned video diffusion backbone and flow-matching action decoder. This baseline represents the performance limit of modern vision-only world action models.
2. **$\pi_{0.5}$ (Vision-Only)**: A state-of-the-art generalist Vision-Language-Action policy optimized for open-world generalization, with no tactile input. This baseline represents the performance limit of modern vision-only VLAs.
3. **$\pi_{0.5}$ + Naive Tactile Injection**: A modified version of $\pi_{0.5}$ where the GelSight tactile stream is injected as an additional visual view, with no regularization to prevent modality collapse. This baseline demonstrates the performance of naive tactile integration approaches that suffer from visual gradient dominance.
   These baselines are chosen because they cover the full range of existing state-of-the-art vision-only and tactile-integrated VLA/VAM systems, allowing fair comparison of VTAM's unique design choices.

# 6. Results & Analysis
## 6.1. Core Results Analysis
The overall performance comparison across all three contact-rich tasks is shown in Table 1 from the original paper:
The following are the results from Table 1 of the original paper:

| Model | Chip | Peel | Wipe |
|-------|------|------|------|
| Genie Envisioner | 0% | 0% | 2.5% |
| $\pi_{0.5}$ (Vision) | 10% | 0% | 0% |
| $\pi_{0.5}$ + Tactile | 5% | 0% | 0% |
| VTAM (Ours) | 90% | 85% | 95% |

### Key Observations:
1. **Vision-only baselines fail catastrophically on contact-rich tasks**: All vision-only baselines achieve near-0% success rates on all tasks except for the $\pi_{0.5}$ vision baseline which achieves 10% success on chip pick-and-place. This is because vision alone cannot reliably detect contact onset, grasp success, or force magnitude, especially under occlusion.
2. **Naive tactile injection performs worse than vision-only**: The $\pi_{0.5}$ + Tactile baseline achieves only 5% success on chip pick-and-place, worse than the vision-only baseline. This demonstrates modality collapse: the model tries to integrate tactile signals but dominant visual gradients suppress tactile information, leading to worse performance than using vision alone.
3. **VTAM achieves state-of-the-art performance across all tasks**: VTAM achieves 90% success on chip pick-and-place, 85% on cucumber peeling, and 95% on whiteboard wiping, outperforming all baselines by large margins. This validates the effectiveness of its joint visuotactile predictive modeling and virtual force regularization.
   Qualitative comparisons between VTAM and baselines are shown in Figure 4 from the original paper:

   ![Figure 4: Qualitative comparison between VTAM and baseline methods on real-world manipulation tassTop:Chippick-an-plaVisin-y baselne l detere whethehe hias beeuy raspe and proc the placeent tage even when he rasp fails.Mid:ucmber peelig int s. Baselines tend to follow a vision-driven trajectory that approaches the center of the cucumber but fail to maintain consistent contact with the surfae, indicating poor forc regulation and lackof contac awareness. Bot:Whiteboard wipng under varying heights and tilt angles. Baselines exhibit unstable wiping behaviors, oe applyieithesuient xcesivel largorce, particularyntilteurfacs. In cntrast, VTAM maintains stable contact and appropriate force regulation across all tasks, enabling robust manipulation behaviors. Red boxes highlight representative failure cases of baselines.](images/4.jpg)
   *该图像是VTAM与基线方法在真实世界操作任务中的定性比较，包括薯片拾取、黄瓜削皮以及白板擦拭。图中展示了VTAM在各种操作场景中维持稳定接触和适当力调节的表现，而基线方法则显示不稳定的操控行为，红框突出基线的失败案例。*

The figure highlights representative failure cases of baselines:
- Vision-only baselines cannot detect failed grasps, proceeding to the placement stage even when no chip is grasped.
- Baselines for peeling follow vision-driven trajectories but lose contact with the cucumber surface, failing to produce valid peel strips.
- Baselines for wiping apply either insufficient or excessive force, especially on tilted surfaces, failing to remove stains or pushing the board out of place.
  VTAM avoids all these failure modes by using tactile feedback to maintain stable contact and appropriate force regulation across all tasks.
### Prediction Visualization
The VTAM world model's ability to accurately predict future visual and tactile frames is shown in Figure 5 from the original paper:

![Figure 5: Prediction visualization of the backbone video model. From top to bottom: Camera-1 viev Camera-2 view, Tactile stream prediction. Ground-truth (top rows) and VTAM predictions (bottom rows).](images/5.jpg)
*该图像是VTAM模型的预测可视化，展示了从不同相机视角和触觉流的预测。包括真实数据（顶部行）和VTAM预测（底部行），展示了与物理交互相关的复杂动态。*

The model's predictions closely match ground truth frames for both camera views and the tactile stream, with only minor blurring in details irrelevant to manipulation, confirming that the world model learns reliable joint visuotactile dynamics for action generation. Additional prediction examples for cucumber peeling and whiteboard wiping are provided in Figures 7-12 in the Appendix, all showing high prediction accuracy across modalities.
## 6.2. Ablation Studies
Ablation studies are conducted on the most force-sensitive chip pick-and-place task to verify the effectiveness of VTAM's core architectural components. The results are shown in Table 2 from the original paper:
The following are the results from Table 2 of the original paper:

| Model Variant | Tactile Integration | Success Rate |
|---------------|---------------------|--------------|
| Vision-only (No Tactile) | None | 0% |
| Late-Fusion Tactile | Downstream Only | 0% |
| No Virtual-Force Reg. | Joint Latent | 10% |
| VTAM (Ours) | Hierarchical World Model | 90% |

### Key Ablation Findings:
1. **Q2: Latent Video Fusion vs. Late-Stage Injection**: The late-fusion tactile variant (tactile signals only injected at the action head, no joint visuotactile world modeling) achieves 0% success rate, identical to the vision-only baseline. This confirms that raw late-stage tactile injection is insufficient, and joint predictive modeling of visuotactile dynamics in the shared world model backbone is required for effective tactile integration.
2. **Q3: Impact of Virtual Force Regularization**: Removing the virtual force regularization loss reduces success rate from 90% to 10%, confirming that this auxiliary objective is critical to prevent modality collapse (visual gradient dominance) and ensure tactile signals influence the action prediction process.

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces VTAM, a novel visuotactile world action model that integrates high-resolution tactile sensing into a predictive video backbone for contact-rich robotic manipulation. The key contributions are joint visuotactile predictive modeling in a shared latent space (eliminating the need for explicit tactile annotations or pretraining) and a virtual force regularization objective (preventing modality collapse without requiring external force sensors). Experiments on three challenging contact-rich tasks show VTAM outperforms state-of-the-art vision-only and naive tactile integration baselines by large margins, demonstrating that tactile feedback is essential to correct visual estimation errors in embodied action models. The work provides a scalable, hardware-agnostic path to physically grounded embodied foundation models capable of reliable real-world physical interaction.
## 7.2. Limitations & Future Work
The authors identify the following limitations and future research directions:
1. **Limited Task Diversity**: The current model is evaluated on only three manipulation tasks. Future work will scale evaluation to a wider range of contact-rich tasks (e.g., assembly, surgical manipulation, food processing) to test generalization.
2. **Dataset Scale**: The current custom dataset is relatively small (less than 300 total trajectories). Future work will explore scaling to large-scale unlabeled visuotactile datasets to further improve cross-task generalization.
3. **Language Integration**: The current model does not integrate natural language instructions for task specification. Future work will extend VTAM to a full visuotactile-language-action model for generalist open-world contact-rich manipulation.
4. **Cross-Embodiment Transfer**: The current model is trained and evaluated on a single robot platform. Future work will explore cross-embodiment transfer to enable VTAM to control different robotic platforms with different tactile sensors.
## 7.3. Personal Insights & Critique
### Key Inspirations
The virtual force regularization objective is a particularly impactful innovation: it eliminates the need for expensive external force-torque sensors, which are a major barrier to widespread deployment of force-aware robotic systems. By deriving force supervision directly from tactile image deformation, VTAM makes force-aware control accessible to low-cost robotic setups equipped only with low-cost optical tactile sensors like GelSight Mini.
The two-stage training strategy is also a valuable design choice: decoupling modality alignment from action learning avoids the unstable convergence that often occurs when introducing new modalities to pretrained models, making it easier to adapt existing video foundation models to support tactile input.
### Potential Limitations and Improvements
1. **Computational Cost**: The two-stage training process requires finetuning a large video transformer backbone, which is computationally intensive (trained on 4x A100 GPUs for 70,000 total steps). Future work could explore more efficient parameter-efficient finetuning (PEFT) methods to reduce training cost.
2. **Tactile Sensor Dependency**: The current model relies on high-resolution GelSight tactile sensors, which are still more expensive than simpler tactile sensors (e.g., pressure-sensitive arrays). Future work could explore adapting the virtual force regularization objective to work with lower-resolution, lower-cost tactile sensors to improve accessibility.
3. **Dynamic Scene Generalization**: The current experiments are conducted in controlled static environments. Future work should test VTAM's performance in dynamic environments with moving objects or unexpected disturbances to validate real-world robustness.
### Broader Application Potential
VTAM's framework can be transferred to a wide range of domains that require contact-rich manipulation, including:
- Domestic service robots (handling fragile kitchenware, folding clothes, food preparation)
- Industrial manufacturing (precision assembly, polishing, quality inspection)
- Healthcare (surgical robotics, patient care, assistive robotics for people with disabilities)
- Agriculture (harvesting fragile produce, food processing)
  The work represents an important step toward building physically grounded embodied AI systems that can safely and reliably interact with the physical world in the same way humans do, using both vision and touch to sense and respond to their environment.