# 1. Bibliographic Information

## 1.1. Title
The central topic of the paper is "Collaborative Task and Path Planning for Heterogeneous Robotic Teams using Multi-Agent PPO." This title highlights the focus on coordinating teams of robots with different capabilities (heterogeneous) to solve planning problems involving both movement (path planning) and job assignment (task allocation) using a specific reinforcement learning algorithm called Multi-Agent Proximal Policy Optimization (MAPPO).

## 1.2. Authors
The authors are Matthias Rubio, Julia Richter, Hendrik Kolvenbach, and Marco Hutter. They are affiliated with the Robotics Systems Lab at ETH Zurich (indicated by the superscript `1`). Their research background lies in robotics, specifically legged locomotion, autonomous navigation, and the application of machine learning to complex robotic control and planning tasks.

## 1.3. Journal/Conference
The paper is published as a preprint on **arXiv** (arXiv:2604.01213). As a preprint published in April 2026 (according to the provided metadata), it has not yet appeared in a specific peer-reviewed journal or conference proceedings in the provided text, though it references "IAC 2023" and other robotics venues in its citations. arXiv is a reputable repository for preliminary scientific reports in fields like physics, mathematics, and computer science, allowing for rapid dissemination of research.

## 1.4. Publication Year
2026.

## 1.5. Abstract
The paper addresses the challenge of efficient extraterrestrial exploration using teams of robots with diverse capabilities (e.g., flying, driving, scientific tools). The core problem is coordinating these heterogeneous teams to maximize scientific value and utilization. Classical planning algorithms struggle with the combinatorial complexity of allocating robots to targets and planning paths, leading to high computational costs. The authors propose a collaborative planning strategy based on **Multi-Agent Proximal Policy Optimization (MAPPO)**. This learning-based method shifts the computational burden from runtime to training time, enabling real-time planning. The work benchmarks the approach against optimal solutions from exhaustive search and validates its ability to perform online replanning in dynamic planetary exploration scenarios.

## 1.6. Original Source Link
The paper is available as a preprint at: https://arxiv.org/abs/2604.01213
PDF Link: https://arxiv.org/pdf/2604.01213

# 2. Executive Summary

## 2.1. Background & Motivation
Efficient exploration of extraterrestrial environments (like Mars) requires robots to perform diverse tasks, such as navigating difficult terrain and conducting scientific measurements. A single robot cannot easily possess all necessary capabilities due to physical constraints. Therefore, a team of heterogeneous robots—where each robot has specific skills (e.g., a drone for aerial view, a rover for drilling)—is essential.

The central challenge is **coordination**: determining which robot should go to which location, in what order, and how they should move to avoid conflicts and minimize mission time. This problem is a variation of the **Multi-Traveling Salesperson Problem (MTSP)**, which is computationally very hard (NP-hard). Traditional, exact algorithms (like exhaustive search or A*) scale poorly; as the number of robots and targets increases, the time required to compute a plan explodes, making real-time replanning impossible in dynamic environments.

The authors' entry point is to use **Reinforcement Learning (RL)**. Unlike classical methods that "think" hard during the mission, RL methods "learn" a strategy beforehand during a training phase. This shifts the computational complexity to the training stage, allowing the robots to make decisions extremely quickly (inference) during the actual mission.

## 2.2. Main Contributions / Findings
The paper's primary contributions are:
1.  **A MAPPO-based Framework:** They present a complete reinforcement learning framework using Multi-Agent PPO that simultaneously handles path planning, task allocation, and scheduling for heterogeneous robots.
2.  **Benchmarking:** They rigorously compare their learned policy against "optimal" solutions found via Exhaustive Search (ES). They show that the RL method achieves near-optimal performance (up to 92% optimality in terms of team effort) while being significantly faster at runtime.
3.  **Replanning Capability:** They demonstrate that the method can handle dynamic environments where new tasks appear mid-mission, a crucial requirement for real exploration.
4.  **Open Source:** The authors have released their learning framework.

    The key finding is that learning-based methods can effectively solve complex multi-robot coordination problems that are intractable for classical optimizers at runtime, providing a viable path for real-time autonomous exploration.

# 3. Prerequisite Knowledge & Related Work

## 3.1. Foundational Concepts
To understand this paper, one must grasp several foundational concepts:

*   **Heterogeneous Multi-Robot Systems:** A team of robots where agents are not identical. Each agent possesses a unique set of **skills** or capabilities (e.g., "flying," "drilling," "spectral analysis"). A task might require specific skills to be completed.
*   **Task Allocation and Path Planning:** This involves two sub-problems:
    1.  **Allocation:** Assigning specific tasks to specific robots based on who has the right skills.
    2.  **Path Planning:** Finding the shortest or most efficient collision-free path for each robot to reach its assigned targets.
*   **Reinforcement Learning (RL):** A machine learning paradigm where an **agent** learns to make decisions by performing actions in an **environment** and receiving **rewards** or penalties. The goal is to maximize the cumulative reward over time.
*   **Multi-Agent Reinforcement Learning (MARL):** An extension of RL where multiple agents interact in the same environment. Agents must cooperate or compete.
*   **Proximal Policy Optimization (PPO):** A popular, state-of-the-art RL algorithm. It is "policy gradient" method, meaning it directly optimizes the policy (the strategy mapping states to actions). PPO is known for being stable and relatively easy to tune. It uses a "clipping" mechanism to prevent the policy from changing too drastically in a single update step, which ensures stable learning.
*   **Multi-Agent PPO (MAPPO):** An adaptation of PPO for multi-agent settings. It typically uses a **Centralized Training, Decentralized Execution (CTDE)** architecture.
    *   **Centralized Training:** During training, a "critic" network has access to the global state (information about all agents) to evaluate how good the joint action is. This helps coordination.
    *   **Decentralized Execution:** During the actual task (inference), each agent uses its own "actor" network based only on its local observations. This is efficient and realistic for robots.

## 3.2. Previous Works
The paper categorizes prior work into two main areas:

1.  **Classical and MTSP Approaches:**
    *   **Exact Methods:** These include Constraint Programming [7] and Integer Programming [8]. While they guarantee the optimal solution, the authors cite [6] to note that computation times can reach hours, making them unsuitable for real-time replanning on resource-constrained hardware like space rovers.
    *   **Metaheuristics:** Algorithms like NSGA-II (a genetic algorithm) [11] or Ant Colony Optimization [12]. These are faster than exact methods and easier to implement but do not guarantee optimality. The trade-off between speed and quality remains a challenge.
    *   **Learning-based MTSP:** Recent works like [16] and [17] use RL to solve MTSP. However, these often decouple the problem: first, a Graph Neural Network assigns targets, then a separate solver plans the path. The authors note that no prior work fully integrates collaborative tasks (requiring multiple robots to meet at one location) into a single learning-based solver.

2.  **Learning Collaborative Strategies:**
    *   **PRIMAL [18]:** An unsupervised approach for multi-agent pathfinding. It learns to navigate to targets while avoiding collisions.
    *   **Socially Aware Navigation [20]:** Uses MAPPO and Graph Neural Networks to help robots navigate around humans. This work demonstrates the effectiveness of MAPPO in cooperative settings, which inspired the authors' choice of algorithm.

## 3.3. Technological Evolution
The field has evolved from manual planning (humans telling robots what to do) to classical automated algorithms (which are slow but optimal) to metaheuristics (faster but less reliable), and finally to modern learning-based approaches (RL). RL represents a paradigm shift because it invests computation time *offline* (during training) to achieve *constant-time* inference *online*. This paper fits into the cutting edge of applying MARL specifically to the complex, coupled problem of heterogeneous task allocation and scheduling, moving beyond simple pathfinding.

## 3.4. Differentiation Analysis
The core differentiation of this paper is the **unified approach** to a complex problem.
*   **vs. Classical Methods:** It shifts complexity to training, enabling real-time performance.
*   **vs. Decoupled RL Methods:** Instead of separating "who goes where" (allocation) from "how do they get there" (planning), this method trains a single policy that handles both simultaneously. This allows the agents to learn emergent collaborative behaviors, such as waiting for a partner or adjusting paths dynamically, which decoupled methods might miss.
*   **vs. Homogeneous RL:** It explicitly handles **heterogeneous** agents with different skill sets and tasks that require specific combinations of these skills (AND/OR logic), which is more complex than standard homogeneous agent problems.

# 4. Methodology

## 4.1. Principles
The core principle is to formulate the robotic exploration mission as a **cooperative multi-agent game**. The "game" takes place on a 2D grid. Agents (robots) must visit targets (points of interest) to solve them. A target is solved only if agents with the required skills are present at the target's location.

The method uses **MAPPO** (Multi-Agent Proximal Policy Optimization). The intuition is that by training in a simulated environment millions of times, the neural networks (the "brains" of the robots) learn a robust strategy—a policy—that maps observations (where am I? where are the targets? what skills do we have?) to actions (move up, down, left, right, stay).

The authors introduce a specific **training strategy** called "Bootstrap and Refinement." Early in training, agents are rewarded simply for finding targets (learning to navigate). Later, "refinement" rewards are introduced to optimize for efficiency (minimizing time and effort), preventing the agents from getting stuck in bad habits early on.

## 4.2. Core Methodology In-depth (Layer by Layer)

### 4.2.1. Training Environment
The environment is modeled as a discrete **2D grid**.
*   **Agents ($q \in \mathcal{Q}$):** There are $N$ agents. They can move using actions from the set $\mathcal{A} = \{up, down, left, right, stay\}$.
*   **Targets ($t \in \mathcal{T}$):** There are $M$ targets.
*   **Skills:** Each agent has a set of skills. Each target requires a set of skills to be solved.
*   **Target Types:**
    *   **OR Type:** The target is solved if *any one* of the required skills is present.
    *   **AND Type:** The target is solved if *all* required skills are present simultaneously (this requires collaboration).
*   **Completion:** An episode (mission) is complete when the number of unsolved targets is zero, i.e., $|\mathcal{T}_U(j)| = 0$.

### 4.2.2. Observations
In RL, the "observation" is what the agent sees. The authors design a specific observation vector for each agent. This vector is fixed in size, meaning the network is trained for a specific number of agents and targets.

The observation consists of three main parts: Positions, Skills, and Goal Types.

**1. Position Observations ($\mathbf{o}_{qi}^{pos}$):**
This part of the observation tells the agent where everyone is. It includes the absolute positions of all agents and the *relative* positions of all targets from the current agent's perspective.

The authors define a helper function $g(\mathbf{r}_q^t)$ to handle target visibility. If a target is already solved, its relative position is set to zero (so the agent ignores it).

$$
g(\mathbf{r}_q^t) = \left\{ \mathbf{r}_q^t \qquad \mathrm{if~} t \in \mathcal{T}_U(j) \right. \\
\qquad \left. \mathbf{\Gamma}^0 \mathbf{, } 0 \right] \qquad \mathrm{if~} t \in \mathcal{T}_S(j) \qquad
$$

*   $\mathbf{r}_q^t \in \mathbb{Z}^2$: The relative position vector between agent $q$ and target $t$.
*   $\mathcal{T}_U(j)$: The set of unsolved targets at time step $j$.
*   $\mathcal{T}_S(j)$: The set of solved targets at time step $j$.
*   $\mathbf{\Gamma}^0 \mathbf{, } 0$: A vector representing zero position (masking the target).

    The full position observation vector for agent $q_i$ is constructed as:

$$
\mathbf{o}_{qi}^{pos} = [ \mathbf{p}_{qi}, . . ., \mathbf{p}_{qN}, g(\mathbf{r}_{qi}^{t1}), . . ., g(\mathbf{r}_{qi}^{tM}) ]
$$

*   $\mathbf{p}_{qi}$: The absolute position of agent $i$.
*   Note: The current agent's position is always the first entry, helping the network identify "self."

**2. Skill Observations ($\mathbf{o}_{qi}^{skill}$):**
Agents need to know who has which skills to coordinate. Skills are encoded as integer values using a function `f(S)`.

$$
\mathbf{o}_{qi}^{skill} = [ f(S_{qi}), . . ., f(S_{qN}), f(S_{t1}), . . ., f(S_{tM}) ]
$$

*   $S_{qi}$: The skill set of agent $i$.
*   $S_{tk}$: The skill set required by target $k$.
*   `f(S)`: A function mapping a skill set to a unique integer ID.

**3. Goal Type Observations ($\mathbf{o}^{goalType}$):**
This tells the agent whether a target requires collaboration (AND) or can be done solo (OR).

$$
\mathbf{o}^{goalType} = [ h_{t1}, . . ., h_{tM} ]
$$

*   $h_{tk}$: Encoded as 1 for AND type, 0 for OR type.

**Final Observation Vector:**
The complete observation is the concatenation of these three parts:

$$
\mathbf{o}_{qi} = [ \mathbf{o}_{qi}^{pos}, \mathbf{o}_{qi}^{skill}, \mathbf{o}^{goalType} ]
$$

### 4.2.3. Rewards
The "reward" is the feedback signal that guides learning. The authors use a composite reward function with several weighted components.

**1. Attraction Reward (AR):**
This acts like a magnetic field, guiding agents toward targets. It is an exponential reward that increases as the agent gets closer to a target it can solve.

$$
h_{qi}(t) = \left\{ \begin{array} { l l } { \exp ( - C_{AR} \cdot \| \mathbf{r}_t \|_2 ) } & { \mathrm{~if~} P } \\ { 0 } & { \mathrm{~otherwise} } \end{array} \right.
$$

$$
r_{qi}^{AR} = \frac { 1 } { M \cdot T_{max} } \displaystyle \sum_{t \in \mathcal{T}} h_{qi}(t)
$$

*   $\mathbf{r}_t$: Relative distance to target $t$.
*   $C_{AR}$: A constant determining the spread of the attraction field.
*   $P$: A condition that is true if the target is unsolved AND the agent has at least one matching skill for it ($|S_{qi} \cap S_t| > 0$).
*   $M$: Total number of targets.
*   $T_{max}$: Maximum steps in an episode (used for normalization).

**2. Target Reward (TR):**
A large fixed payout given when a target is solved. Crucially, this is given to *all* agents to promote cooperation rather than competition.

$$
r_{qi}^{TR} = \left\{ \begin{array} { l l } { \frac { 1 } { M } \quad } & { \mathrm{if } \ t \in \mathcal{T}_U(j - 1) \wedge t \in \mathcal{T}_S(j) } \\ { 0 \quad } & { \mathrm{otherwise} } \end{array} \right.
$$

*   The condition checks if the target was unsolved in the previous step (`j-1`) but is solved in the current step ($j$).

**3. Wrong Cost (WC):**
A penalty to prevent agents from stepping on targets they cannot help with (blocking others).

$$
r_{qi}^{WC} = \sum_{t \in \mathcal{T}_U} \left\{ \begin{array} { l l } { - 1 \quad } & { \mathrm{if~} ( \| \mathbf{r}_t \|_2 = 0 ) \land ( | S_{qi} \cap S_t | = 0 ) } \\ { 0 \quad } & { \mathrm{otherwise} } \end{array} \right.
$$

*   Condition: True if the agent is at the target location (distance is 0) AND has no matching skills (intersection is empty).

**4. Step Cost (SC):**
A penalty for every movement to encourage efficiency (shortest path).

$$
r_{qi}^{SC} = \left\{ \begin{array} { l l } { 0 \quad } & { \mathrm{if~} u(j - 1) = stay } \\ { - 1 \quad } & { \mathrm{otherwise} } \end{array} \right.
$$

*   `u(j-1)`: The action taken at the previous step.

**5. Time Cost (TC):**
A penalty that decreases as more targets are solved, encouraging speed.

$$
r_{qi}^{TC} = \frac { | \mathcal{T}_U(j) | } { M \cdot T_{max} }
$$

*   $|\mathcal{T}_U(j)|$: The number of unsolved targets remaining.

**6. Terminal Bonus (TB):**
A final reward for successfully completing the entire mission.

$$
r_{qi}^{TB} = \left\{ \begin{array} { r l } { 1 } & { \quad \mathrm{if~} ( \mathcal{T}_S(j - 1) \subset \mathcal{T} ) \land ( \mathcal{T} \subseteq \mathcal{T}_S(j) ) } \\ { 0 } & { \quad \mathrm{otherwise} } \end{array} \right.
$$

*   Condition: True if the set of solved targets transitions from not containing all targets to containing all targets.

**Total Reward:**
The final reward is a weighted sum of all components.

$$
\mathbf{r}_{qi} = [ r_{qi}^{AR}, r_{qi}^{TR}, r_{qi}^{WC}, r_{qi}^{SC}, r_{qi}^{TC}, r_{qi}^{TB} ]^\top
$$

$$
\mathbf{w} = [ w^{AR}, w^{TR}, w^{WC}, w^{SC}, w^{TC}, w^{TB} ]
$$

$$
r_{qi}^{full} = \mathbf{w} \cdot \mathbf{r}_{qi}
$$

### 4.2.4. Learning Architecture
The authors use the **MAPPO** algorithm (based on [21, 22]). The architecture follows the Centralized Training, Decentralized Execution paradigm.

*   **Actor Network:** Each agent has its own actor network (or shares one if they are homogeneous). It takes the agent's local observation $\mathbf{o}_{qi}$ as input and outputs a probability distribution over actions. It contains a **Gated Recurrent Unit (GRU)** to handle memory of past states.
*   **Critic Network:** There is a single centralized critic network. It takes a concatenation of *all* agents' observations as input. It learns a "value function" `V(s)`, estimating how good the current global state is. This global view helps the agents learn to cooperate.

    The following figure (Figure 2 from the original paper) illustrates the system architecture, including the interaction between the agents, the environment, and the MAPPO actor-critic networks:

    ![该图像是一个示意图，展示了基于多智能体近端策略优化（MAPPO）的协作任务与路径规划方法。图中包括对观察、模型更新、代理动作和奖励的描述，展示了不同代理在二维空间中的分布及其任务分配。整体系统旨在实现高效的机器人团队协同。](images/2.jpg)
    *该图像是一个示意图，展示了基于多智能体近端策略优化（MAPPO）的协作任务与路径规划方法。图中包括对观察、模型更新、代理动作和奖励的描述，展示了不同代理在二维空间中的分布及其任务分配。整体系统旨在实现高效的机器人团队协同。*

### 4.2.5. Training Strategy
To overcome the difficulty of learning complex behaviors from scratch, the authors use a two-stage curriculum:

1.  **Bootstrap:** In this phase, only the Attraction Reward (AR) and Target Reward (TR) are active. The Step Cost (SC) and Time Cost (TC) are disabled (set to 0). This allows agents to learn basic navigation and task completion without being penalized for inefficiency.
2.  **Refinement:** In this phase, all rewards, including the penalties (SC, TC), are activated. The agents now refine their strategy to minimize time and effort while maintaining the ability to solve tasks.

# 5. Experimental Setup

## 5.1. Datasets
The experiments are conducted in a **custom simulation environment** built using the JaxMARL framework [23].
*   **Scale:** The environment is a $32 \times 32$ grid.
*   **Agents:** 3 agents.
*   **Skills:** 2 distinct skills (e.g., Skill A and Skill B). This results in 3 possible agent skill sets (Agent with A, Agent with B, Agent with A+B).
*   **Targets:** 5 to 7 targets per episode (depending on the policy being trained).
*   **Target Types:** Randomly assigned as AND (collaborative) or OR (single-agent).
*   **Randomization:** Initial positions, skill sets, and target locations are randomized for every episode to ensure generalization.
*   **Why this dataset?** This setup is sufficiently complex to require coordination (due to AND targets and heterogeneous skills) but simple enough to allow for comparison against optimal Exhaustive Search (ES) baselines, which would be impossible on very large maps.

## 5.2. Evaluation Metrics
The authors use three specific metrics to evaluate performance:

**1. Success Rate ($M_{success}$):**
*   **Conceptual Definition:** The percentage of simulated missions in which the team successfully solved all targets. A high success rate indicates reliability.
*   **Mathematical Formula:**
    $$ M_{success} = \frac{K_{solved}}{K_{sims}} $$
*   **Symbol Explanation:**
    *   $K_{solved}$: The number of environments (episodes) where all targets were solved.
    *   $K_{sims}$: The total number of simulated environments.

**2. Solve Time ($M_{st}$):**
*   **Conceptual Definition:** A measure of how quickly the team completes the mission. The authors define it as the "time remaining" relative to a maximum limit. A higher value means the mission finished faster.
*   **Mathematical Formula:**
    $M_{st} = T_{max} - T_{solved}$
*   **Symbol Explanation:**
    *   $T_{max}$: The maximum allowed time steps (horizon) for an episode.
    *   $T_{solved}$: The actual time step at which all targets were solved.

**3. Total Team Effort ($M_{tte}$):**
*   **Conceptual Definition:** The sum of all movements made by all agents. It measures the energy or resource consumption of the plan. Similar to solve time, it is defined as "effort remaining."
*   **Mathematical Formula:**
    $$ M_{tte} = \sum_{q \in \mathcal{A}} \left( T_{max} - \sum_{j=1}^{T_{solved}} | r_{qi}^{step}(j) | \right) $$
*   **Symbol Explanation:**
    *   $\mathcal{A}$: The set of all agents.
    *   $r_{qi}^{step}(j)$: A function that evaluates to 1 if the agent moved at step $j$, and 0 otherwise.

## 5.3. Baselines
The primary baseline is **Exhaustive Search (ES)**.
*   **ES1:** An exhaustive search optimized for **Solve Time** (finding the fastest plan).
*   **ES2:** An exhaustive search optimized for **Total Team Effort** (finding the plan with the least movement).
*   **Why these baselines?** Exhaustive search checks every possible combination of assignments and paths. For small problems (like 5-7 targets), this is computationally feasible and provides the **theoretical optimum**. Comparing the RL policy against ES proves how close the learning method gets to perfection.

# 6. Results & Analysis

## 6.1. Core Results Analysis
The authors trained three policies: $\Pi_{5T}$ (5 targets), $\Pi_{6T}$ (6 targets), and $\Pi_{7T}$ (7 targets). The results show that the MAPPO-based approach achieves highly competitive performance compared to the optimal Exhaustive Search (ES) baselines.

The following are the results from Table 4 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2"></th>
<th colspan="3">Policy</th>
</tr>
<tr>
<th>I5T</th>
<th>I6T</th>
<th>I7T</th>
</tr>
</thead>
<tbody>
<tr>
<td>$M_{success}$</td>
<td>0.99</td>
<td>0.95</td>
<td>0.91</td>
</tr>
<tr>
<td>RL $M_{st}$</td>
<td>0.86</td>
<td>0.81</td>
<td>0.73</td>
</tr>
<tr>
<td>ES1 $M_{st}$</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>RL $M_{tte}$</td>
<td>0.92</td>
<td>0.91</td>
<td>0.84</td>
</tr>
<tr>
<td>ES2 $M_{tte}$</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
</tbody>
</table>

*Note: The table headers in the original text imply comparison against ES1 and ES2. The values 0.86, 0.81, 0.73 for RL $M_{st}$ represent the ratio of the RL score to the optimal ES1 score (normalized optimality). Similarly for $M_{tte}$ vs ES2.*

**Analysis:**
*   **Success Rate:** The policies are very reliable, with success rates above 90% even for 7 targets. This drops slightly as complexity increases, which is expected.
*   **Optimality:** The RL policy achieves **86%** optimality for solve time and **92%** for team effort in the 5-target scenario.
*   **Trade-off:** The policy performs better in minimizing team effort ($M_{tte}$) than minimizing solve time ($M_{st}$). This is a direct result of the reward weights chosen during training (the authors weighted step costs heavily).
*   **Scalability:** As targets increase (5 -> 7), optimality decreases (86% -> 73%). This indicates the problem is becoming harder for the fixed-size network to solve perfectly.

## 6.2. Inference Time Analysis
A major advantage of the proposed method is computation speed.

The following figure (Figure 3 from the original paper) compares the inference time of the RL method against the Exhaustive Search (ES) baseline:

![Fig. 3: RL inference and training time measurements compared to the inference time of the ES approach with respect to the trained policies by number of solved targets.](images/3.jpg)

**Analysis:**
*   **Exhaustive Search (ES):** The blue line shows an exponential increase in time (approx. 1.5 orders of magnitude) as the number of targets increases. This confirms that classical methods are not scalable.
*   **RL Inference:** The orange line remains relatively flat. Inference time is constant ($O(1)$) because it consists of a single forward pass through the neural network, regardless of map complexity. This enables real-time operation.
*   **RL Training:** The green line shows training time. While training is expensive and grows with complexity, it is a one-time cost performed offline. The key benefit is that this heavy cost is *not* incurred during the mission.

## 6.3. Replanning Capability
To test dynamic replanning, the authors trained a policy $\Pi_{5T5R}$ where 5 new targets appear mid-episode. They compared it to the baseline $\Pi_{5T}$ (which was not explicitly trained for new targets but used a "buffer" strategy to overwrite solved targets with new ones).

The following are the results from Table 5 of the original paper:

<table>
<thead>
<tr>
<th rowspan="2"></th>
<th>$M_{success}$</th>
<th>$M_{st}$</th>
<th>$M_{tte}$</th>
</tr>
</thead>
<tbody>
<tr>
<td>$\Pi_{5T}$</td>
<td>84.2 %</td>
<td>85.70 ± 20.1</td>
<td>232.5 ± 72.8</td>
</tr>
<tr>
<td>$\Pi_{5T5R}$</td>
<td>84.1 %</td>
<td>85.73 ± 20.9</td>
<td>220.0 ± 81.3</td>
</tr>
</tbody>
</table>

**Analysis:**
*   The results are nearly identical.
*   This suggests that the "buffer" strategy (treating new targets as just replacing old ones in the fixed observation vector) works well. Explicitly training on dynamic scenarios ($\Pi_{5T5R}$) did not yield a significant improvement over the baseline's generalization capability.
*   This is a positive finding, implying the learned policy is robust enough to handle new tasks without specific retraining for that exact scenario.

# 7. Conclusion & Reflections

## 7.1. Conclusion Summary
The paper successfully demonstrates that **Multi-Agent PPO (MAPPO)** can be used to coordinate heterogeneous robotic teams for complex exploration tasks. By shifting the computational burden from runtime to training, the method achieves near-optimal performance (up to 92% of the optimal team effort) with constant-time inference, making it suitable for real-time systems like space rovers. The method also proves capable of handling dynamic environments through online replanning.

## 7.2. Limitations & Future Work
The authors identify several limitations:
1.  **Fixed Observation Size:** The current architecture requires a fixed number of agents and targets. This limits scalability; if a mission requires more targets than the network was trained for, it cannot handle them directly.
2.  **Reward Tuning:** As with many RL approaches, finding the right balance of reward weights was difficult and non-intuitive. Weights that worked for 5 targets failed for 7 targets.
3.  **Training Complexity:** While inference is fast, training time and memory requirements grow with the number of targets.

**Future Directions:**
*   **Dynamic Observation Architectures:** The authors suggest using **Graph Neural Networks (GNNs)** [20] or world models like **DreamerV3** [24] to create observation encoders that can handle variable numbers of agents and targets. This would allow a single trained policy to generalize to any team size or mission scale.
*   **Real-world Validation:** Future work involves deploying this on physical robots (e.g., legged robots) to bridge the sim-to-real gap.

## 7.3. Personal Insights & Critique
This paper provides a compelling case for the use of MARL in robotics, particularly for the "last mile" of execution where real-time constraints are strict.
*   **Innovation:** The unified handling of heterogeneous skills (AND/OR logic) within the MAPPO framework is a strong contribution. Most simplified TSP solvers ignore the physical constraints of specific tool requirements.
*   **Practicality:** The comparison against Exhaustive Search is rigorous. However, the "fixed observation size" is a significant bottleneck for real-world deployment where the number of points of interest is unknown. The transition to Graph Neural Networks (as suggested by the authors) seems like the necessary next step to make this truly field-deployable.
*   **Safety:** In safety-critical domains like space exploration, 86-92% optimality is good, but the ~10% failure rate (or sub-optimality) needs consideration. However, compared to the *inability* of classical methods to run at all due to time constraints, the RL approach offers a "good enough" solution that is actually actionable.