# 1. Bibliographic Information
## 1.1. Title
The paper is titled *UI-Voyager: A Self-Evolving GUI Agent Learning via Failed Experience*. Its central topic is the development of an autonomous mobile Graphical User Interface (GUI) agent that can self-improve without manual data annotation by efficiently learning from failed interaction trajectories and solving sparse reward credit assignment issues for long-horizon GUI tasks.
## 1.2. Authors
The authors are Zichuan Lin, Feiyu Liu, Yijun Yang, Jiafei Lyu, Yiming Gao, Yicheng Liu, Zhicong Lu, Yangbin Yu, Mingyu Yang, Junyou Li, Deheng Ye, and Jie Jiang. All authors are affiliated with **Tencent Hunyuan** (Tencent's foundation model research team), with research backgrounds in multimodal large language models, reinforcement learning, and GUI agent development. The project's open-source code and models are hosted at `github.com/ui-voyager/ui-voyager`.
## 1.3. Journal/Conference
This paper is published as a preprint on arXiv, a widely used open-access platform for sharing research outputs before formal peer review and conference/journal publication. Preprints on arXiv are widely cited in the AI and machine learning community, as they enable rapid dissemination of cutting-edge research.
## 1.4. Publication Year
The paper was published (preprint release) on March 25, 2026 (UTC).
## 1.5. Abstract
The paper targets two key limitations of existing Multimodal Large Language Model (MLLM)-powered mobile GUI agents: inefficient learning from failed interaction trajectories, and ambiguous credit assignment for long-horizon tasks with sparse only trajectory-level rewards. To address these issues, the authors propose a two-stage self-evolving framework called UI-Voyager:
1.  First stage: **Rejection Fine-Tuning (RFT)**, a fully autonomous closed-loop pipeline for co-evolving training data and model capabilities.
2.  Second stage: **Group Relative Self-Distillation (GRSD)**, a method that identifies critical "fork points" between successful and failed trajectories, then constructs dense step-level supervision from successful trajectories to correct failed ones.
    Extensive experiments on the AndroidWorld benchmark show the 4B-parameter UI-Voyager achieves an 81.0% Pass@1 success rate, outperforming all state-of-the-art baselines (including much larger models) and exceeding reported human-level performance on the benchmark. The method eliminates the need for expensive manual data annotation for GUI agent training.
## 1.6. Original Source Link
-   Preprint abstract page: https://arxiv.org/abs/2603.24533v1
-   Full PDF link: https://arxiv.org/pdf/2603.24533v1
-   Publication status: Public preprint, not yet formally accepted at a peer-reviewed conference/journal as of the current date (March 26, 2026 UTC).

    ---

# 2. Executive Summary
## 2.1. Background & Motivation
### Core Problem
Autonomous mobile GUI agents (systems that can interact with smartphone interfaces to complete user tasks) have become a high-priority research area with the advancement of MLLMs. However, two critical bottlenecks limit their performance:
1.  **Inefficient learning from failed experiences**: Most existing methods discard or underutilize failed interaction trajectories, wasting valuable learning signals.
2.  **Ambiguous credit assignment under sparse rewards**: For long-horizon GUI tasks (which may require up to 30 sequential actions), rewards are only provided at the end of a trajectory (1 for success, 0 for failure). This makes it impossible for standard reinforcement learning (RL) methods to identify which specific step caused a task failure, leading to unstable policy optimization and low sample efficiency.
### Importance of the Problem
Mobile GUI agents have massive real-world value: they can support accessibility for users with motor impairments, automate repetitive routine tasks (e.g., ordering food, setting reminders), and power cross-app automation workflows. Prior GUI agents (e.g., Siri, Cortana) are limited to predefined simple tasks, while newer MLLM-powered agents still fail to reliably complete complex long-horizon tasks due to the above bottlenecks.
### Innovative Entry Point
The authors' key insight is that when rolling out multiple trajectories for the same task, successful and failed trajectories often share identical screen states (fork points) before diverging due to different action choices. These fork points can be used to extract precise step-level corrective supervision without any manual annotation, by using successful trajectories as the "teacher" for failed trajectories at the same state.
## 2.2. Main Contributions / Findings
The paper's primary contributions are:
1.  **Two-stage self-evolving framework UI-Voyager**: A fully annotation-free training pipeline for GUI agents consisting of RFT for data-model co-evolution, and GRSD for solving sparse reward credit assignment.
2.  **Rejection Fine-Tuning (RFT)**: An iterative pipeline that automatically generates, filters, and fine-tunes on successful GUI interaction trajectories, enabling steady model performance improvements without manual data curation.
3.  **Group Relative Self-Distillation (GRSD)**: A novel method that detects fork points between successful and failed trajectory groups via Structural Similarity Index (SSIM) matching, then constructs dense step-level supervision to correct failed actions, eliminating the credit assignment problem for long-horizon tasks.
### Key Findings
-   The 4B-parameter UI-Voyager achieves 81.0% Pass@1 success rate on the AndroidWorld benchmark, exceeding the reported human-level performance of 80.0% and outperforming all baseline models (including 235B-parameter proprietary models like MAI-UI-235B-A22B and Gemini 2.5 Pro).
-   GRSD significantly outperforms standard RL methods (PPO, GRPO) for GUI agent training, as it provides precise step-level supervision instead of relying on ambiguous trajectory-level reward signals.
-   The self-evolving pipeline eliminates the need for expensive manual annotation of GUI interaction data, making large-scale GUI agent training far more accessible.

    ---

# 3. Prerequisite Knowledge & Related Work
## 3.1. Foundational Concepts
To understand this paper, beginners first need to grasp the following core concepts (all acronyms are defined on first use):
1.  **Graphical User Interface (GUI)**: The visual interactive interface of digital devices (e.g., smartphone screens, desktop apps) that users operate via actions like clicking, swiping, and typing.
2.  **Multimodal Large Language Model (MLLM)**: A type of large language model that can process and understand multiple input modalities (text, images, video) in addition to generating text output. MLLMs are the backbone of modern GUI agents, as they can understand screen screenshots and task instructions to generate appropriate interaction actions.
3.  **Supervised Fine-Tuning (SFT)**: A standard model training process where a pre-trained model is fine-tuned on a labeled dataset of input-output pairs to learn a specific downstream task. For GUI agents, SFT datasets typically consist of (screen screenshot + task instruction, correct action) pairs.
4.  **Reinforcement Learning (RL)**: A machine learning paradigm where an agent learns to make decisions by interacting with an environment, receiving rewards for good actions and penalties for bad actions. For GUI agents, the environment is the smartphone interface, actions are GUI operations, and rewards are given for completing tasks.
5.  **Proximal Policy Optimization (PPO)**: A widely used state-of-the-art RL algorithm that optimizes an agent's policy (decision-making function) while ensuring updates do not change the policy too drastically (to avoid unstable training).
6.  **Group Relative Proximal Policy Optimization (GRPO)**: A variant of PPO optimized for large language model training, where multiple responses (rollouts) are sampled for each task, and advantages are normalized across the group of responses to reduce training variance.
7.  **Credit Assignment Problem**: A core challenge in RL for long-horizon tasks, where it is difficult to determine which specific actions in a long sequence of decisions contributed to the final success or failure of the task.
8.  **Structural Similarity Index (SSIM)**: A metric used to quantify the similarity between two images, designed to match human perception of visual similarity. It is used in this paper to detect if two GUI screenshots represent the same screen state.
9.  **Self-Distillation**: A model training technique where a model learns from its own outputs (instead of external labeled data or a separate larger teacher model). In this paper, the agent learns from its own successful trajectories to correct its failed trajectories.
10. **Pass@K**: A standard evaluation metric for interactive agents, measuring the probability that at least one of K independent attempts at a task is successful. Pass@1 (K=1) is the most commonly reported metric, representing the success rate for a single attempt at each task.
## 3.2. Previous Works
### Interactive GUI Environments
Prior work has developed multiple benchmark environments for training and evaluating GUI agents:
-   Web-focused environments: WebShop, WebArena, VisualWebArena, designed for testing web browsing agents.
-   Desktop-focused environments: OSWorld, WindowsAgentArena, designed for testing desktop operating system agents.
-   Mobile-focused environments: Mobile-Env, MobileWorld, and AndroidWorld (the benchmark used in this paper), which includes 116 diverse real-world Android app tasks with rule-based success verifiers.
### MLLM-Powered GUI Agents
Recent work has built GUI agents by fine-tuning MLLMs on GUI interaction data:
-   General-purpose VLMs: Qwen3-VL, Gemini 2.5 Pro, Seed1.8, which have general multimodal understanding capabilities but are not optimized for GUI interaction tasks.
-   Specialized GUI agents: UI-Tars, GUI-Owl, Step-GUI, MAI-UI, UI-Venus, which are fine-tuned specifically for GUI interaction tasks, but rely on manually annotated training data and suffer from the credit assignment problem for long-horizon tasks.
### Credit Assignment Solutions for GUI Agents
Prior work like EvoCUA attempts to identify critical fork points in trajectories, but relies on external VLMs to generate corrective supervision, increasing compute cost and dependency on external models. Other methods rely on standard RL algorithms (PPO, GRPO) which have poor sample efficiency for sparse reward GUI tasks.
## 3.3. Technological Evolution
The field of GUI agents has evolved in three main stages:
1.  **Rule-based agents (2010s)**: Agents like Siri and Cortana that only support predefined, simple tasks, with no ability to generalize to unseen tasks.
2.  **MLLM-powered zero-shot agents (2023-2024)**: General-purpose MLLMs adapted for GUI tasks via prompt engineering, which can generalize to unseen tasks but have low success rates for complex tasks.
3.  **Fine-tuned specialized GUI agents (2024-2025)**: MLLMs fine-tuned on annotated GUI interaction datasets, which have higher success rates but require expensive manual annotation and still struggle with long-horizon tasks due to credit assignment issues.
4.  **Self-evolving GUI agents (2026+)**: The class of agents introduced by this paper, which self-improve via autonomous interaction without manual annotation, solving the credit assignment problem by learning from failed experiences.
## 3.4. Differentiation Analysis
Compared to existing GUI agent training methods, UI-Voyager has three core innovations:
1.  **No manual annotation required**: Unlike prior specialized GUI agents that rely on manually labeled interaction datasets, UI-Voyager generates all training data autonomously via interaction with the environment.
2.  **Efficient learning from failed experiences**: Unlike standard RL methods that waste failed trajectory data, UI-Voyager uses failed trajectories as training data by pairing them with successful trajectories at fork points.
3.  **No external teacher model required**: Unlike prior fork-point-based methods that use external VLMs to generate corrective supervision, GRSD uses the agent's own successful trajectories as the teacher, eliminating dependency on external models and reducing training cost.

    ---

# 4. Methodology
## 4.1. Principles
The core principle of UI-Voyager is a fully autonomous two-stage iterative self-evolution pipeline that eliminates manual annotation requirements and solves the sparse reward credit assignment problem:
1.  First, RFT builds a strong base model by iteratively generating, filtering, and fine-tuning on successful interaction trajectories, enabling co-evolution of data quality and model capability.
2.  Second, GRSD leverages the insight that identical screen states shared by successful and failed trajectories provide natural opportunities for step-level corrective supervision, by using successful trajectories as the teacher for failed trajectories at fork points.
    The full pipeline of UI-Voyager is shown in the figure below (Figure 2 from the original paper):

    ![Figure : The whole pipeline of training UIVoyager for mobile GUI tasks. It consists of two iterative sages: (R a ou a vrier t collect high-quality samples for spervised fine-tuning; () Group Relative Sel-Distiation (GRSD), whic identies "ork points" betwee succesful and file trajectoy groups using SMmatching and cot erronous actions o rthereine te poliy $\\pi _ { m }$ through mixed-data training.](images/2.jpg)
    *该图像是一个示意图，展示了UI-Voyager在移动GUI任务中训练的整个流程。图中分为两个阶段：第一阶段为多轮拒绝微调（RFT），通过基于规则的验证器收集高质量样本进行监督微调；第二阶段为多轮组相对自蒸馏（GRSD），通过检测关键分叉点和自我纠正，从成功和失败的轨迹组中构建更密集的监督。此方法旨在提高移动GUI自动化的效率与性能。*

## 4.2. Core Methodology In-depth
### Task Formulation
GUI agent interaction is modeled as a **Partially Observable Markov Decision Process (POMDP)** defined by the tuple $(\mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{T})$:
-   $\mathcal{S}$: The set of underlying true states of the GUI environment.
-   $\mathcal{O}$: The observation space, which combines the current screen screenshot and the natural language task instruction $\mathcal{T}$ the agent needs to complete.
-   $\mathcal{A}$: The action space, consisting of all valid GUI operations supported by the AndroidWorld benchmark, listed in Table 1 below (Table 1 from the original paper):

    <table>
    <thead>
    <tr>
    <th>Code Actions</th>
    <th>Descriptions</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>click(x,y)</td>
    <td>Clicking at coordinates (x, y)</td>
    </tr>
    <tr>
    <td>long_press(x,y)</td>
    <td>Long-pressing at coordinates (x, y )</td>
    </tr>
    <tr>
    <td>swipe(x,Y,x',y')</td>
    <td>Swiping from (x, y) to (x', y′ )</td>
    </tr>
    <tr>
    <td>open_app(app_name)</td>
    <td>Opening an app by name</td>
    </tr>
    <tr>
    <td>input_text(text)</td>
    <td>Typing input text</td>
    </tr>
    <tr>
    <td>keyboard_enter()</td>
    <td>Pressing the Enter key on the keyboard</td>
    </tr>
    <tr>
    <td>navigateback()</td>
    <td>Pressing the system Back button</td>
    </tr>
    <tr>
    <td>navigate_home()</td>
    <td>Pressing the system Home button</td>
    </tr>
    <tr>
    <td>wait()</td>
    <td>Waiting / no-op action (also used for unsupported actions)</td>
    </tr>
    <tr>
    <td>status(goal_status)</td>
    <td>Terminating the episode with status, e.g., success</td>
    </tr>
    <tr>
    <td>answer(text)</td>
    <td>Returning the final answer text</td>
    </tr>
    </tbody>
    </table>

-   $\mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$: The transition function, which defines how the environment state changes after an action is taken.
    At each step $t$, the agent generates the next action `a_t = \pi(\mathcal{T}, o_t, \mathcal{H}_t)`, where $\mathcal{H}_t = (a_{t-h}, o_{t-h}, ..., a_{t-1}, o_{t-1})$ is the history of previous actions and observations with a window size $h$. Task success is verified via a rule-based verifier that uses Android Debug Bridge (ADB) commands to check the application state, returning a trajectory-level reward of 1 for success and 0 for failure.
---
### Stage 1: Rejection Fine-Tuning (RFT)
RFT is a closed-loop iterative pipeline that generates high-quality SFT data without manual annotation, consisting of three modules:
1.  **Trajectory Generation**: A seed task generator synthesizes novel tasks by perturbing key parameters (temporal constraints, quantities, element identifiers) from existing task templates. The current agent model is used to autonomously execute these tasks in the GUI environment to generate raw interaction trajectories.
2.  **Rejection Sampling**: Raw trajectories are filtered via a rule-based verifier, and only successful trajectories (those that complete the task) are kept as high-quality SFT data. This eliminates the need for manual annotation of correct actions, as successful trajectories inherently contain correct step-level action sequences.
3.  **Iterative Training**:
    -   Initial iteration: The base model is Qwen3-VL-4B-Instruct, a general-purpose 4B-parameter MLLM, used to generate the first set of trajectories.
    -   Subsequent iterations: The fine-tuned model from the previous iteration is used as the agent to generate new trajectories, which are filtered via rejection sampling to fine-tune the next version of the model. New tasks are generated in each iteration to avoid overfitting.
        Empirically, 3 rounds of RFT improve the model's Pass@1 performance from 37% to 73.2%, providing a strong base for the second GRSD stage.
---
### Stage 2: Group Relative Self-Distillation (GRSD)
#### Limitations of Standard RL Methods (GRPO/PPO)
Standard RL algorithms like GRPO and PPO are poorly suited for long-horizon GUI tasks due to the sparse trajectory-level reward signal, as shown below:
The GRPO objective is defined as:
$$
\mathcal{I}_{GRPO} = \mathbb{E}_{q, o_i} \left[ \frac{1}{\sum_{i=1}^{G} \left| o_i \right|} \sum_{i=1}^{G} \sum_{t=1}^{\left| o_i \right|} \operatorname*{min} \Bigl( r_{i,t} ( \theta ) \hat{A}_{i,t} , \mathrm{clip} \Bigl( r_{i,t} ( \theta ) , 1 - \epsilon_{\mathrm{low}} , 1 + \epsilon_{\mathrm{high}} \Bigr) \hat{A}_{i,t} \Bigr) \right]
$$
Where:
-   $q$: The input task instruction
-   $o_i$: The i-th trajectory (rollout) sampled for task $q$
-   $G$: The number of rollouts sampled per task (group size)
-   $|o_i|$: The length (number of steps) of the i-th trajectory
-   $r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}$: The token-level importance sampling ratio, comparing the probability of the current policy $\pi_\theta$ generating the action at step $t$ to the probability of the old policy $\pi_{\theta_{old}}$ generating the same action
-   $\hat{A}_{i,t} = \frac{R^{(i)} - \mathrm{mean}(\{R^{(i)}\}_{i=1}^{G})}{\mathrm{std}(\{R^{(i)}\}_{i=1}^{G})}$: The normalized advantage, where $R^{(i)}$ is the trajectory-level reward (1 for success, 0 for failure) for the i-th rollout
-   $\epsilon_{low}, \epsilon_{high}$: Clipping thresholds to prevent large policy updates
    The PPO objective is a variant that uses Generalized Advantage Estimation (GAE) to estimate per-step advantages:
$$
\mathcal{I}_{PPO} = \mathbb{E}_{q, o} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} \operatorname*{min} \Bigl( r_{t} ( \theta ) \hat{A}_{t}^{GAE} , \mathrm{clip} \Bigl( r_{t} ( \theta ) , 1 - \epsilon , 1 + \epsilon \Bigr) \hat{A}_{t}^{GAE} \Bigr) \right]
$$
Where $\hat{A}_{t}^{GAE}$ is the GAE advantage estimate for step $t$, and $\epsilon$ is the clipping threshold.
For sparse reward GUI tasks, the trajectory-level reward $R^{(i)}$ is identical for all steps in a trajectory. This means that if a 30-step trajectory fails, all 30 steps receive an advantage of ~0, even if 29 of the steps were correct, making it impossible for the model to identify which step caused the failure.
---
#### Fork Point Detection
The core insight of GRSD is that for the same task, successful and failed trajectories often share identical screen states before diverging due to different action choices. These divergence points (fork points) are used to extract step-level supervision, as shown in the figure below (Figure 3 from the original paper):

![该图像是一个示意图，展示了UI-Voyager中关键的叉点检测和成功与失败的路径。图中显示了正向和反向的观察及动作，成功路径以绿色标记，而失败路径以红色标记。通过识别叉点，系统能够在不同的路径间进行自我调整，以提高移动GUI代理的学习效率。](images/3.jpg)
*该图像是一个示意图，展示了UI-Voyager中关键的叉点检测和成功与失败的路径。图中显示了正向和反向的观察及动作，成功路径以绿色标记，而失败路径以红色标记。通过识别叉点，系统能够在不同的路径间进行自我调整，以提高移动GUI代理的学习效率。*

Fork point detection takes a successful trajectory $\tau^+ = \{(o_0^+, a_0^+), ..., (o_{T^+}^+, a_{T^+}^+)\}$ and a failed trajectory $\tau^- = \{(o_0^-, a_0^-), ..., (o_{T^-}^-, a_{T^-}^-)\}$ as input, and outputs a set of fork points (pairs of matching steps in the successful and failed trajectories where actions diverge). The process is defined as follows:
1.  **Cross-Trajectory State Matching**: The observation equivalence function `SAME` is used to check if two screenshots represent the same screen state:
    $$
    \mathrm{SAME}(o_a, o_b) = \mathbb{1}\left[\mathrm{SSIM}\left(\phi(o_a), \phi(o_b)\right) \geq \theta\right]
    $$
    Where:
    -   $\phi(\cdot)$: A preprocessing pipeline that crops, resizes, and converts screenshots to grayscale to remove irrelevant variance (e.g., status bar changes)
    -   $\mathrm{SSIM}(\cdot, \cdot)$: Structural Similarity Index, which quantifies visual similarity between two images
    -   $\theta$: Similarity threshold (set to 0.8 in the paper)
    -   $\mathbb{1}[\cdot]$: Indicator function, equal to 1 if the condition inside is true, 0 otherwise
        To speed up computation, screenshots with a perceptual hash similarity below 0.8 are first filtered out before computing SSIM.
2.  **Transition Alignment**: For each failed step $j$, if there exists a successful step $i \geq i_{min}$ where `SAME`$(o_i^+, o_j^-)$ and `SAME`$(o_{i+1}^+, o_{j+1}^-)$ are both true, the trajectory prefixes are aligned. We then set $i_{min} = i+1$ for all subsequent failed steps, as the trajectories are still aligned at this point and no correction is needed.
3.  **Teacher Step Selection**: For each remaining failed step $j$, we find all successful steps $i \geq i_{min}$ that meet two conditions:
    1.  Observation equivalence: $\mathrm{SAME}(o_i^+, o_j^-) = \text{true}$
    2.  Transition divergence:
        $$
        \mathrm{DIVERGE}(i,j) = 
        \begin{cases}
        \mathbf{true} & \mathrm{if}\ i = T^+ \ \mathrm{or}\ j = T^- \\
        \mathbf{true} & \mathrm{if}\ \mathrm{SSIM}(\phi(o_{i+1}^+), \phi(o_{j+1}^-)) < \theta \\
        \mathbf{false} & \mathrm{otherwise}
        \end{cases}
        $$
        Divergence occurs if either trajectory ends at this step, or the next screenshots after the action are different (meaning the actions taken at this step led the trajectories to different states).
    From all qualifying teacher step candidates $\mathcal{C}(j)$, we select the best matching step:
    $$
    i^*(j) = \arg\operatorname*{max}_{i \in \mathcal{C}(j)} \left( \mathrm{SSIM}(\phi(o_i^+), \phi(o_j^-)), -i \right)
    $$
    This selects the step with the highest SSIM similarity, breaking ties by choosing the earliest occurring successful step to preserve temporal ordering. We then update $i_{min} = i^*(j)$ for subsequent failed steps.
The full fork point detection algorithm is summarized in Algorithm 1 below (from the original paper):
```
Algorithm 1 Fork Point Detection
Require: Successful trajectory $\tau^+$, failed trajectory $\tau^-$, threshold $\theta$   
Ensure: Fork point set $\mathcal{M}$   
1: $\mathcal{M} \leftarrow \emptyset$, $i_{\mathrm{min}} \gets 0$   
2: for $j = 0$ to $T^-$ do   
3:  if $\exists i \geq i_{\mathrm{min}}$ s.t. $\mathbf{SAME}(o_i^+, o_j^-)$ and $\mathbf{SAME}(o_{i+1}^+, o_{j+1}^-)$ then   
4:      $i_{\mathrm{min}} \leftarrow i + 1$   
5:      continue   
6:  $\mathcal{C}(j) \gets \{ i \geq i_{\mathrm{min}} ~ | ~ \mathrm{SAME}(o_i^+, o_j^-) \wedge \mathrm{DIVERGE}(i,j) \}$   
7:  if $\mathcal{C}(j) = \emptyset$ then   
8:      continue   
9:  $i^*(j) \leftarrow \arg\operatorname*{max}_{i \in \mathcal{C}(j)} ( \mathrm{SSIM}(\phi(o_i^+), \phi(o_j^-)), -i )$
10: $\mathcal{M} \leftarrow \mathcal{M} \cup \{(j, i^*(j))\}$
11: $i_{\mathrm{min}} \leftarrow i^*(j)$
12: return $\mathcal{M}$
```
---
#### Step-Level Self-Distillation
For each detected fork point $(j, i^*(j))$, we construct a training sample by keeping the full context of the failed trajectory up to step $j$, and replacing the failed action with the correct action from the successful trajectory at step $i^*(j)$:
$$
\mathbf{x}_j^{\mathrm{train}} = [\underbrace{\mathrm{prompt}_j^-}_{\mathrm{failed-context ~ prompt}} | \underbrace{\mathrm{response}_{i^*(j)}^+}_{\mathrm{correct ~ action}}]
$$
Where $prompt_j^-$ includes the task instruction, interaction history, and current screen screenshot from the failed trajectory at step $j$, and $response_{i^*(j)}^+$ is the correct action from the successful trajectory at step $i^*(j)$.
The training objective is the standard autoregressive next-token prediction loss, computed only over the response (action) tokens:
$$
\mathcal{L}_{\mathrm{GRSD}} = - \frac{1}{|\mathcal{D}|} \sum_{\mathbf{x} \in \mathcal{D}} \frac{1}{T_{\mathbf{x}}} \sum_{t=1}^{T_{\mathbf{x}}} \log \pi_\theta(y_t \mid s_{1}, ..., s_{P_{\mathbf{x}}}, y_{<t})
$$
Where:
-   $\mathcal{D}$: The dataset of constructed fork point training samples
-   $s_{1:P_{\mathbf{x}}}$: The prompt tokens (task context, history, current screen)
-   $y_{1:T_{\mathbf{x}}}$: The response tokens (correct action)
-   $P_{\mathbf{x}}$: Length of the prompt
-   $T_{\mathbf{x}}$: Length of the response

    ---

# 5. Experimental Setup
## 5.1. Datasets
All experiments are conducted on the **AndroidWorld** benchmark, a widely used mobile GUI agent evaluation platform with the following characteristics:
-   Source: Developed by Rawles et al. (2025), published at ICLR 2025.
-   Scale: 116 diverse tasks across real-world Android applications, with over 7000 training task variants generated by randomizing task parameters and initial device states.
-   Characteristics: Tasks span multiple complexity levels, from simple 1-step tasks (e.g., turning off Bluetooth) to long-horizon 30-step tasks (e.g., navigating a maze in a browser game). A rule-based verifier automatically checks task success via ADB commands, eliminating manual evaluation bias.
-   Rationale for selection: AndroidWorld is the most comprehensive and widely adopted mobile GUI agent benchmark, with standardized evaluation protocols and published human performance baselines, making it ideal for comparing UI-Voyager against state-of-the-art methods.
## 5.2. Evaluation Metrics
The primary evaluation metric used is **Pass@K**, defined as follows:
1.  **Conceptual Definition**: Pass@K measures the probability that an agent successfully completes a given task in at least one of K independent attempts. Pass@1 (K=1) is the most commonly reported metric, representing the agent's success rate for a single attempt per task.
2.  **Mathematical Formula**:
    $$
    \mathrm{Pass@K} = 1 - \frac{N_{\mathrm{fail}}}{N_{\mathrm{total}}}
    $$
    Where $N_{\mathrm{fail}}$ is the number of tasks where all K attempts failed, and $N_{\mathrm{total}}$ is the total number of tasks evaluated.
3.  **Symbol Explanation**:
    -   $K$: The number of independent attempts the agent is allowed per task.
    -   $N_{\mathrm{fail}}$: Count of tasks where none of the K attempts succeeded.
    -   $N_{\mathrm{total}}$: Total number of unique tasks in the evaluation set.
        For this paper, all reported results are Pass@1 (K=1) unless stated otherwise, averaged over 64 independent runs with randomized task parameters to ensure reproducibility.
## 5.3. Baselines
UI-Voyager is compared against three categories of representative baselines, covering both open-source and proprietary models, general-purpose VLMs and specialized GUI agents:
1.  **General-purpose VLMs**: Qwen3-VL (2B, 4B, 8B, 32B, 235B parameter variants), Seed1.5-VL, Seed1.8, Gemini 2.5 Pro. These represent state-of-the-art general multimodal models not optimized specifically for GUI tasks.
2.  **Specialized GUI agents**: MAI-UI (2B, 8B, 32B, 235B variants), Step-GUI (4B, 8B variants), UI-Tars (7B, 72B, 230B variants), UI-Venus (7B, 30B, 72B variants), GUI-Owl (7B, 8B variants), ScaleCUA-3B, Ferret-UI Lite-3B. These represent the state-of-the-art in specialized GUI agent models, fine-tuned on GUI interaction data.
3.  **Human baseline**: The reported human-level performance on AndroidWorld (80.0% Pass@1), collected from human users completing the benchmark tasks.
    These baselines are representative because they cover the full range of model sizes (2B to 235B parameters) and state-of-the-art approaches to GUI agent development, ensuring a fair and comprehensive comparison of UI-Voyager's performance.

---

# 6. Results & Analysis
## 6.1. Core Results Analysis
The main performance comparison on the AndroidWorld benchmark is shown in Table 2 below (Table 2 from the original paper):

<table>
<thead>
<tr>
<th>MODEL</th>
<th>#PARAMS</th>
<th>Success Rate</th>
</tr>
</thead>
<tbody>
<tr>
<td colspan="3"><strong>Baselines</strong></td>
</tr>
<tr>
<td>Qwen3-VL-2B (Bai et al., 2025a)</td>
<td>2B</td>
<td>36.4</td>
</tr>
<tr>
<td>MAI-UI-2B (Zhou et al., 2025b)</td>
<td>2B</td>
<td>49.1</td>
</tr>
<tr>
<td>ScaleCUA-3B (Liu et al., 2025)</td>
<td>3B</td>
<td>23.7</td>
</tr>
<tr>
<td>Ferret-UI Lite-3B (Yang et al., 2025c)</td>
<td>3B</td>
<td>28.0</td>
</tr>
<tr>
<td>Qwen3-VL-4B (Bai et al., 2025a)</td>
<td>4B</td>
<td>45.3</td>
</tr>
<tr>
<td>Step-GUI-4B (Yan et al., 2025b)</td>
<td>4B</td>
<td>63.9</td>
</tr>
<tr>
<td>UI-Tars-7B (Qin et al., 2025)</td>
<td>7B</td>
<td>33.0</td>
</tr>
<tr>
<td>UI-Tars-1.5-7B (Seed, 2025b)</td>
<td>7B</td>
<td>30.0</td>
</tr>
<tr>
<td>UI-Venus-7B (Gu et al., 2025)</td>
<td>7B</td>
<td>49.1</td>
</tr>
<tr>
<td>GUI-Owl-7B (Ye et al., 2025b)</td>
<td>7B</td>
<td>66.4</td>
</tr>
<tr>
<td>Step-GUI-8B (Yan et al., 2025a)</td>
<td>8B</td>
<td>67.7</td>
</tr>
<tr>
<td>Qwen3-VL-8B (Bai et al., 2025a)</td>
<td>8B</td>
<td>47.6</td>
</tr>
<tr>
<td>MAI-UI-8B (Zhou et al., 2025b)</td>
<td>8B</td>
<td>70.7</td>
</tr>
<tr>
<td>Step-GUI-8B (Yan et al., 2025b)</td>
<td>8B</td>
<td>67.7</td>
</tr>
<tr>
<td>GUI-Owl-1.5-8B-Thinking (Xu et al., 2026)</td>
<td>8B</td>
<td>71.6</td>
</tr>
<tr>
<td>UI-Venus-1.5-30B-A3B (Gao et al., 2026)</td>
<td>30B</td>
<td>77.6</td>
</tr>
<tr>
<td>Qwen3-VL-32B (Bai et al., 2025a)</td>
<td>32B</td>
<td>57.3</td>
</tr>
<tr>
<td>MAI-UI-32B (Zhou et al., 2025b)</td>
<td>32B</td>
<td>73.3</td>
</tr>
<tr>
<td>UI-Tars-SFT-72B (Qin et al., 2025)</td>
<td>72B</td>
<td>46.6</td>
</tr>
<tr>
<td>UI-Venus-72B (Gu et al., 2025)</td>
<td>72B</td>
<td>65.9</td>
</tr>
<tr>
<td>Seed1.5-VL (Guo et al., 2025b)</td>
<td>-</td>
<td>62.1</td>
</tr>
<tr>
<td>UI-Tars-2 (Wang et al., 2025)</td>
<td>230B</td>
<td>73.3</td>
</tr>
<tr>
<td>Qwen3-VL-235B-A22B (Bai et al., 2025a)</td>
<td>235B</td>
<td>63.7</td>
</tr>
<tr>
<td>UI-Tars-1.5 (Seed, 2025b)</td>
<td>-</td>
<td>64.2</td>
</tr>
<tr>
<td>Gemini-2.5-Pro (DeepMind, 2025)</td>
<td>-</td>
<td>69.7</td>
</tr>
<tr>
<td>Seed1.8 (Seed, 2025a)</td>
<td>-</td>
<td>70.7</td>
</tr>
<tr>
<td>MAI-UI-235B-A22B (Zhou et al., 2025b)</td>
<td>235B</td>
<td>76.7</td>
</tr>
<tr>
<td>Human (Rawles et al., 2025)</td>
<td>-</td>
<td>80.0</td>
</tr>
<tr>
<td colspan="3"><strong>Ours</strong></td>
</tr>
<tr>
<td>UI-Voyager</td>
<td>4B</td>
<td><strong>81.0</strong></td>
</tr>
</tbody>
</table>

The results are also visualized in Figure 1 below (Figure 1 from the original paper):

![Figure 1: Performance comparison of various GUI agents on AndroidWorld. Our UI-Voyager (4B) achieves an $8 1 . 0 \\%$ Pass `@ 1` success rate, outperforming larger models and exceeding reported human-level performance.](images/1.jpg)
*该图像是图表，展示了不同GUI智能体在AndroidWorld上的成功率比较。UI-Voyager（4B）实现了81.0%的成功率，超过了多个较大模型，亦超越了人类性能。*

Key takeaways from the core results:
1.  UI-Voyager (4B) achieves 81.0% Pass@1 success rate, outperforming all baseline models and exceeding the reported human-level performance of 80.0%.
2.  UI-Voyager achieves this result with only 4B parameters, demonstrating extreme parameter efficiency: it outperforms 235B-parameter models like MAI-UI-235B-A22B (76.7%) and Qwen3-VL-235B-A22B (63.7%) by a large margin, and even outperforms specialized GUI agents of similar size (e.g., Step-GUI-4B at 63.9%) by 17.1 percentage points.
## 6.2. Ablation Studies
### RFT Effectiveness Ablation
The effectiveness of the RFT stage is validated in Figure 4 below (Figure 4 from the original paper):

![Figure 4: RFT significantly boosts agent performance. Left: Pass $@ \\mathrm { K }$ performance across four iterative rounds of RFT. The results show consistent improvement in both $\\mathrm { P a s s } @ 1$ and Pass $@ \\mathbf { k }$ as the self-evolution progresses. We select the checkpoint from the third RFT round $( \\mathrm { P a s s } @ 1 = 7 3 . 2 \\% )$ for subsequent training. Right: Training curves of GRPO and PPO initialized from Qwen3-VL-4B-Instruct. The results show that directly deploying RL algorithms from Qwen3-VL-4B-Instruct model yelds marginal gains and exhibits high sample ineffciency.](images/4.jpg)
*该图像是图表，展示了UI-Voyager在不同训练轮次和方法下的成功率。左侧图表显示了在四轮拒绝微调（RFT）中的Pass@$K$成功率表现，随着自我进化的进行，成功率逐步提高。右侧则比较了GRPO和PPO算法的训练过程，表明直接从Qwen3-VL-4B-Instruct模型实施强化学习的效果有限且样本效率低。整体上，RFT显著提升了智能体的性能。*

Key findings:
1.  Left panel: Pass@K performance improves consistently across 4 rounds of RFT, with Pass@1 increasing from 37% to 73.2% after 3 rounds, demonstrating the robust self-improvement capability of the RFT pipeline. The 3rd round checkpoint is used as the base for the GRSD stage.
2.  Right panel: Standard RL methods (GRPO and PPO) initialized directly from the base Qwen3-VL-4B-Instruct model have extremely low sample efficiency, requiring ~175 training steps to reach the 64% Pass@1 performance of a single RFT iteration, validating that RFT provides a far more efficient initialization strategy for GUI agent training.
### GRSD Effectiveness Ablation
The effectiveness of GRSD compared to standard RL methods is validated in Figure 8 below (Figure 8 from the original paper):

![Figure 8: Training performance comparison of GRSD, GRPO, and PPO. All methods start from the same RFT model with $7 3 . 2 \\%$ success rate. GRSD successfully boosts the agent's Pass `@ 1` performance to $81 \\%$ (Left), while GRPO and PPO show slower progress and plateau around $76 \\%$ (Right). The results demonstrate that GRSD's fork point detection and sel-corection mechanisms enable more effectiv learning compared to standard RL baselines.](images/8.jpg)
*该图像是一个图表，展示了GRSD、GRPO和PPO在训练过程中的表现对比。所有方法均从相同的RFT模型开始，GRSD的Pass@K成功率随着K的增加而显著提升，最终达到约87.5%，而GRPO和PPO的成功率相对平稳，在76%左右。结果表明，GRSD的学习效果优于标准RL基线。*

All three methods (GRSD, GRPO, PPO) start from the same 3rd round RFT checkpoint with 73.2% Pass@1. GRSD improves performance to 81.0% Pass@1, while GRPO and PPO plateau at ~76% Pass@1, demonstrating that GRSD's step-level supervision is far more effective than trajectory-level RL for long-horizon GUI tasks.
The performance gap is even larger on 10 representative low-success-rate tasks (tasks where the base RFT model has <50% success rate), as shown in Figure 9 below (Figure 9 from the original paper):

![Figure 9: Performance comparison of GRSD, GRPO, and PPO on ten representative low-success-rate tasks. GRSD consistently achieves the highest success rate across all tasks, significantly outperorming both PPO and GRPO. In cntrast, PO nd GRPOstrumakesusanal ais ue thearcy eu sam t lac edi asset eanis. The eult strate hat R ables effnt from failed trajectories even in sparse-reward environments.](images/9.jpg)
*该图像是图表，展示了图9中GRSD、GRPO、PPO和RFT在十个代表性低成功率任务上的成功率比较。结果显示GRSD在所有任务中均取得最高成功率，显著优于其他方法，体现了其在稀疏奖励环境下有效利用失败经验的能力。*

GRSD consistently achieves the highest success rate across all low-success-rate tasks, while GRPO and PPO struggle to make improvements, validating that GRSD efficiently learns from failed trajectories even in sparse reward environments.
## 6.3. Case Studies
### Fork Point Detection Examples
Two example fork point detections are shown below:
1.  **BrowserMaze task (Figure 5 from original paper)**: The successful and failed trajectories share the same screen state at Step 12 (fork point). The failed trajectory takes a "Right" action that is blocked by a wall, while the successful trajectory takes a "Down" action that leads to task completion. Fork point detection identifies this divergence point to provide corrective supervision.

    ![该图像是示意图，展示了UI-Voyager中两个不同步骤的行动过程。上部分显示在第12步前选择错误路径导致的失败，标记为红色的Fork Point，而下部分则展示了选择正确按钮以顺利完成任务，标记为黄色的成功路径。此图表明了关键决策点对任务结果的重要性。](images/5.jpg)
    *该图像是示意图，展示了UI-Voyager中两个不同步骤的行动过程。上部分显示在第12步前选择错误路径导致的失败，标记为红色的Fork Point，而下部分则展示了选择正确按钮以顺利完成任务，标记为黄色的成功路径。此图表明了关键决策点对任务结果的重要性。*

2.  **SystemBluetoothTurnOff task (Figure 6 from original paper)**: The fork point occurs at Step 0. The failed trajectory uses an upward swipe to open settings, while the successful trajectory uses a downward swipe to open the notification shade and quick settings, completing the task far faster. Fork point detection identifies this divergence at the very first step.

    ![Figure 6: Example of fork point detection on SystemBluetoothTurnOff task. The fork point occurs at Step l sat h a aro om Te i j ttt the etti p wanc war swipe while heul trajey use downwar swipe the otti hadeanc quc tts. Forkpoin deteonens this itl diveren, pri corrective supervision at the very first step.](images/6.jpg)
    *该图像是一个示意图，展示了在SystemBluetoothTurnOff任务中，失败与成功的轨迹对比。图中标识了分叉点，分别显示了向上滑动与向下滑动的操作，强调在首次步骤提供纠正监督的重要性。*

### Self-Correction Example
An example of the self-correction training sample constructed from the BrowserMaze task fork point is shown in Figure 7 below (Figure 7 from the original paper):

![该图像是一个示意图，展示了在失败和成功的轨迹中，移动GUI代理的不同操作步骤。其中包含用户提示、任务进度和助手响应，强调了自我纠正的过程。](images/7.jpg)
*该图像是一个示意图，展示了在失败和成功的轨迹中，移动GUI代理的不同操作步骤。其中包含用户提示、任务进度和助手响应，强调了自我纠正的过程。*

The failed trajectory's context is kept unchanged, while the incorrect action ("Right" with wrong reasoning) is replaced with the correct action ("Down" with correct reasoning) from the successful trajectory, creating a high-quality training sample without manual annotation.

---

# 7. Conclusion & Reflections
## 7.1. Conclusion Summary
This paper introduces UI-Voyager, a fully self-evolving mobile GUI agent trained via a two-stage annotation-free pipeline that addresses two critical limitations of existing GUI agents: inefficient learning from failed trajectories, and ambiguous credit assignment for long-horizon sparse reward tasks. The RFT stage enables data-model co-evolution to build a strong base model, while the GRSD stage detects fork points between successful and failed trajectories to provide dense step-level supervision for self-correction.
Experimental results on the AndroidWorld benchmark show that the 4B-parameter UI-Voyager achieves 81.0% Pass@1 success rate, outperforming all state-of-the-art baselines (including much larger 235B models) and exceeding human-level performance. The method eliminates the need for expensive manual data annotation, making large-scale GUI agent training far more accessible.
## 7.2. Limitations & Future Work
The authors identify the following limitations of the current work:
1.  **SSIM-based matching limitations**: SSIM-based fork point detection can be affected by temporal mismatches (e.g., different loading speeds across trajectories) and transient visual perturbations (e.g., cursor blinking, status bar updates), leading to missed or incorrect fork point matches.
2.  **Action space limitations**: The current work uses high-level abstract actions (e.g., `click(x,y)`, `swipe()`) that abstract away low-level touch dynamics, which may reduce transfer robustness to environments with finer-grained control requirements.
    The authors suggest the following future research directions:
1.  Improve fork point matching by using temporal window matching instead of single-frame matching, masking high-variance UI regions (status bar, pop-ups), and combining SSIM with OCR/layout token matching to reduce noise.
2.  Explore hierarchical action modeling: retain high-level actions for sample-efficient learning, then add low-level gesture fine-tuning to improve transfer robustness.
3.  Extend the UI-Voyager framework to other GUI domains (web agents, desktop agents) beyond mobile Android.
4.  Develop more adaptive reasoning and self-correction mechanisms to improve performance on even more complex real-world GUI tasks.
## 7.3. Personal Insights & Critique
### Key Insights
The UI-Voyager framework has broad applicability beyond mobile GUI agents: its core insight of using fork points between successful and failed self-generated trajectories to provide dense step-level supervision can be applied to almost any interactive agent domain, including web agents, robot manipulation agents, and game-playing agents. The elimination of manual annotation requirements is a particularly impactful contribution, as it removes the largest bottleneck to scaling GUI agent training.
### Potential Improvements
1.  **Generalization testing**: The current work only tests on the AndroidWorld benchmark. Future work should validate cross-platform generalization to other mobile benchmarks (e.g., MobileWorld) and non-mobile GUI domains (e.g., WebArena, OSWorld) to confirm the framework's generalizability.
2.  **SSIM threshold sensitivity**: The current work uses a fixed SSIM threshold of 0.8 for fork point detection. The threshold may need to be tuned for different environments, and future work could explore adaptive thresholding or learned similarity metrics to improve matching robustness.
3.  **Computational cost**: The GRSD stage requires rolling out multiple trajectories per task to find successful/failed pairs, which increases inference cost during training. Future work could explore more efficient trajectory sampling strategies to reduce compute requirements.
### Broader Impact
UI-Voyager represents a significant step toward practical, high-performance GUI automation that can be deployed at low cost. It has particular value for accessibility use cases, enabling users with motor impairments to control mobile devices via natural language instructions, and for enterprise automation use cases, reducing the need for manual repetitive GUI tasks. The open-source release of the code and models will enable further research into self-evolving GUI agents by the wider community.