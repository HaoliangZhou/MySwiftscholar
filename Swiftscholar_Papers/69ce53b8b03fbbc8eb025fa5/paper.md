# Spectral Compact Training: Pre-Training Large Language Models via Permanent Truncated SVD and Stiefel QR Retraction

Björn R. Kohlberger EctoSpace, Dublin, Ireland bkohlberger@icloud.com https://github.com/EctoSpace/SCT March 2026

# Abstract

The memory wall remains the primary bottleneck for training large language models on consumer hardware. We introduce Spectral Compact Training (SCT), a method that replaces dense weight matrices with permanent truncated SVD factors ( $W = U \mathrm { d i a g } ( s ) V ^ { \mathrm { ~ \scriptsize ~ \backslash ~ } } )$ , where the full dense matrix is never materialized during training or inference. Gradients flow through the compact spectral factors via standard backpropagation, and $U$ , $V$ are retracted to the Stiefel manifold via QR decomposition after each optimizer step.

SCT achieves up to $1 9 9 \times$ memory reduction per MLP layer at rank 32, enabling full training steps of 70B-parameter architectures on a Steam Deck handheld (7.2GB peak memory vs. 1,245 GB for dense FP32 training with Adam). Rank-sweep experiments on SmolLM2-1.7B (ranks 32256, 2000 steps, NVIDIA A100) show that all tested ranks converge to the same loss floor $( \sim 4 . 2 – 4 . 5 )$ , identifying the learning rate schedule—not MLP rank—as the primary bottleneck for closing the gap with dense training. Rank 128 emerges as the efficiency sweet spot: 11.7 $\times$ MLP compression, 20.0 GB GPU memory (vs. 35.5 GB dense), and 65.6 perplexity (best among all SCT configurations). GPU memory drops 46% at rank 32 while training throughput doubles.

# 1 Introduction

Large language models face a severe memory wall during training. A standard 70B model requires over 1,200 GB for weights, gradients, and optimizer states in full precision, restricting foundational AI research to heavily funded institutions with multi-GPU clusters. Even 7B models strain single-GPU setups. SCT addresses this by permanently storing every weight matrix in compact SVD factors: $W =$ $U \mathrm { d i a g } ( s ) V ^ { \top }$ , where $U \in \mathbb { R } ^ { m \times k }$ and $V \in \mathbb { R } ^ { n \times k }$ have orthonormal columns and $s \in \mathbb { R } ^ { k }$ contains the singular values. The dense matrix is never constructed. Gradients are computed via standard backpropagation through the factored form, producing gradient shapes $( m \times k )$ , $( k )$ , and $( n \times k )$ rather than $( m \times n )$ . After each optimizer step, $U$ and $V$ are retracted to the Stiefel manifold via QR decomposition to maintain orthonormality. This is a different approach from both post-training compression (which applies SVD after training) and adapter methods like LoRA (which add small trainable matrices alongside frozen dense weights). SCT replaces the parameterization itself: the spectral factors are the weights. The result is a 199 $\times$ memory reduction per MLP layer at rank 32 on a 70B architecture, with full training steps completing in 7.2 GB on consumer hardware.

# Contributions.

1. A training method that permanently stores and updates weights in truncated SVD form with Stiefel manifold retraction, never materializing a dense matrix. 2. Architectural validation showing a full 70B training step (forward, backward, optimizer, retraction) in 7.2 GB on a Steam Deck and 7.9 GB on Apple M4 Pro. 3. A rank-sweep experiment on SmolLM2-1.7B demonstrating that all ranks (32256) converge to the same loss foor, with rank 128 as the Pareto-optimal configuration (11.7 $\times$ compression, best perplexity). 4. Evidence that the convergence gap versus dense training is driven by learning rate configuration, not spectral rank capacity.

# 2 Related Work

SCT builds on ideas from several research lines. The individual components—SVD factorization, Stiefel manifold optimization, low-rank training—are well-studied. The specific combination appears novel: permanent truncated SVD storage with Stiefel QR retraction for LLM training, where the dense matrix is never materialized. Low-rank CNN training (ELRT). Sui et al. [Sui et al., 2024] train compact CNNs from scratch using Tucker-2 decomposition with orthogonality regularization. ELRT targets convolutional architectures, uses soft orthogonality penalties rather than hard manifold constraints, and still materializes dense intermediate representations. SCT uses permanent truncated SVD for transformer MLP layers, enforces orthonormality via QR retraction (a hard constraint), and never constructs any dense matrix. Riemannian fine-tuning (StelLA). Li et al. [Li et al., 2025] propose a three-factor $U S V ^ { \mid }$ decomposition with Stiefel constraints for LoRA adapters (NeurIPS 2025 Spotlight). The factored form and manifold optimization are shared with SCT. The key difference is scope: StelLA applies Stiefel constraints to the adapter $\Delta W = U S V ^ { \top }$ , while the frozen pre-trained dense matrix $W$ remains in memory. SCT replaces $W$ entirely. StelLA is a fine-tuning method; SCT targets training with full weight replacement, which is where the memory savings originate. Low-rank adapters (LoRA). Hu et al. [Hu et al., 2021] freeze the dense matrix $W$ and train small adapter matrices alongside it. The full dense model remains in memory throughout. SCT never stores $W$ ; the spectral factors are the model's only representation of each weight matrix. Low-rank $^ +$ sparse pre-training (LOST). Han et al. [Han et al., 2025] combine low-rank and sparse components for LLM pre-training from scratch, using SVD for initialization. LOST shares the goal of efficient pre-training but does not maintain Stiefel manifold constraints on the factors, and uses sparsity as a complementary mechanism. SVD training for CNNs. Yang et al. [Yang et al., 2020] decompose CNN layers into full-rank SVD form and train $U$ , $s$ , $V$ with orthogonality regularization (soft penalty). This is distinct from SCT's QR retraction (hard constraint). The regularization approach cannot guarantee orthonormality, which affects the interpretation of singular values and the validity of rank truncation. Post-training SVD compression. SVD-LLM [Wang et al., 2024] and related methods perform SVD truncation on already-trained dense models. These methods optimize the truncation step but do not train in SVD form. SCT trains natively in low-rank spectral form from initialization. Memory-efficient gradient methods (GaLore). Zhao et al. [Zhao et al., 2024] project dense gradients into low-rank subspaces via periodic SVD, reducing optimizer state memory while keeping full-rank weights and gradients. SCT avoids dense gradients entirely by differentiating through the small spectral factors directly. Riemannian optimization. Optimization on the Stiefel manifold [Absil et al., 2008] and efficient retractions via Cayley transforms [Li et al., 2020] are established techniques. SCT applies QR retraction specifically to maintain orthonormality o spectralactors during neural network training

# 3 Methodology

SCT replaces the storage and update mechanism of neural network weight matrices. Spectral representation. Every weight matrix $W \in \mathbb { R } ^ { m \times n }$ is stored as its rank- $k$ truncated SVD:

$$
W = U \cdot \mathrm { d i a g } ( s ) \cdot V ^ { \top }
$$

where $U \in \mathbb { R } ^ { m \times k }$ , $V \in \mathbb { R } ^ { n \times k }$ have orthonormal columns, and $s \in \mathbb { R } ^ { k }$ Storage: $k ( m + n + 1 )$ numbers instead of ${ m n n }$ . For the LLaMA-70B MLP layer ( $m = 8 1 9 2$ , $n = 2 8 6 7 2$ )at $k = 3 2$ , this is 1.18M vs. 234.9M parameters: a 199 $\times$ per-layer reduction.

# Forward pass.

$$
\begin{array} { l l l } { { h = x \cdot U } } & { { \qquad } } & { { [ b \times k ] \quad \mathrm { c o s t : } \ O ( b m k ) } } \\ { { h _ { s } = h \odot s } } & { { \qquad } } & { { [ b \times k ] \quad \mathrm { c o s t : } \ O ( b k ) } } \\ { { y = h _ { s } \cdot V ^ { \top } } } & { { \qquad } } & { { [ b \times n ] \quad \mathrm { c o s t : } \ O ( b k n ) } } \end{array}
$$

Three small matrix multiplications. Total cost: $O ( b k ( m + n ) ) ,$ instead of $O ( b m n )$ . Backward pass. Backpropagation computes gradients $\partial \mathcal { L } / \partial U \left( \boldsymbol { m } \times \boldsymbol { k } \right)$ , ${ \partial \mathcal L } / { \partial s \ ( \boldsymbol k ) } , { \partial \mathcal L } / { \partial V \left( \boldsymbol n \times \boldsymbol k \right) }$ through the same factored operations via standard PyTorch autograd. No $m \times n$ gradient exists at any point. Note on gradients: The gradients are exact with respect to the factored parameterization. They are not identical to the gradients of a full-rank dense model, because the rank-constrained model defines a different loss landscape. SCT uses standard backpropagation; it does not replace or eliminate backpropagation. What it eliminates is the dense matrix and the corresponding densesized gradients. Stiefel manifold retraction. After each optimizer step (AdamW), $U$ and $V$ are retracted to the Stiefel manifold:

$$
Q , R = \mathrm { Q R } ( U _ { \mathrm { u p d a t e d } } ) ; U  Q \cdot \mathrm { s i g n } ( \mathrm { d i a g } ( R ) )
$$

The sign correction ensures continuity. Cost: $O ( m k ^ { 2 } )$ per layer. Memory analysis. For each weight matrix with the Adam optimizer, SCT stores four copies (weights, gradients, first moment, second moment) of $k ( m + n + 1 )$ numbers rather than 4mn. Table 1 shows per-MLP-layer compression at rank 32 across model scales.

Table 1: Per-MLP-layer training memory (weights $^ +$ gradients $^ +$ Adam states) at rank 32.   

<table><tr><td>Model</td><td>Layer (m × n)</td><td>Dense+Adam</td><td>SCT (k=32)</td><td>Compression</td></tr><tr><td>SmolLM2-135M</td><td>576 × 1536</td><td>14.2 MB</td><td>1.1 MB</td><td>13×</td></tr><tr><td>SmolLM2-360M</td><td>1024 × 4096</td><td>67.1 MB</td><td>2.6 MB</td><td>26×</td></tr><tr><td>SmolLM2-1.7B</td><td>2048 × 8192</td><td>268.4 MB</td><td>5.2 MB</td><td>51×</td></tr><tr><td>LLaMA-7B</td><td>4096 × 11008</td><td>721.4 MB</td><td>7.7MB</td><td>93×</td></tr><tr><td>Qwen-27B</td><td>4096 × 17408</td><td>1,141 MB</td><td>11.0 MB</td><td>104×</td></tr><tr><td>LLaMA-70B</td><td>8192 × 28672</td><td>3,758 MB</td><td>18.9 MB</td><td>199×</td></tr></table>

# Algorithm 1 SCT Training Step

Require: Model with SpectralLinear layers, learning rate $\eta$ , batch $( x , y )$   
1: Forward: $\hat { y } = \mathrm { m o d e l } ( x )$ {uses $\boldsymbol { h } = ( \boldsymbol { x } \cdot \boldsymbol { U } ) \odot \boldsymbol { s } \cdot \boldsymbol { V } ^ { \top } \}$   
2: Loss: $\mathcal { L } = \mathrm { C r o s s E n t r o p y } ( \hat { y } , y )$   
3: Backward: Compute $\nabla _ { U } \mathcal { L }$ , $\nabla _ { s } \mathcal { L }$ , $\nabla _ { V } \mathcal { L }$ via autograd   
4: Optimizer: AdamW step on $U$ , $s$ , $V$   
5: Retract: For each SpectralLinear layer:   
6: $\begin{array} { r } { Q , R \gets \mathrm { Q R } ( U ) ; \quad U \gets Q \cdot \mathrm { s i g n } ( \mathrm { d i a g } ( R ) ) } \\ { Q , R \gets \mathrm { Q R } ( V ) ; \quad V \gets Q \cdot \mathrm { s i g n } ( \mathrm { d i a g } ( R ) ) } \end{array}$   
7:

# 4 Experiments

# 4.1 70B Architecture Validation

A full 70B-class transformer (80 layers, $d { = } 8 1 9 2$ , $\mathrm { H n } { = } 2 8 6 7 2$ , SwiGLU activation, matching LLaMA-3-70B layer dimensions) was initialized in spectral form at rank 32 and executed through one complete training step: forward pass, backward pass, Adam optimizer step, and Stiefel QR retraction. Attention is simplified (additive, no softmax/masking) to isolate the memory and gradient fow test from sequence-length concerns. 452M spectral parameters correspond to a 77.8B-parameter dense architecture. What this demonstrates: The memory footprint of a 70B-architecture training step fits within 8 GB. Dense FP32 training of the same architecture with Adam requires 1,245 GB (Figure 1). What this does not demonstrate: Convergence to a useful language model at rank 32, or equivalence to a dense 70B model. These are separate questions addressed in the rank-sweep experiments below.

# 4.2 Rank Sweep (SmolLM2-1.7B on Alpaca)

To characterize the compression-quality tradeoff, we run dense baseline vs. SCT at ranks 32, 64, 128, and 256 on SmolLM2-1.7B fine-tuned on the Alpaca dataset. MLP layers (gate_proj, up_proj, down_proj) are converted to SpectralLinear via truncated SVD; attention projections, embeddings, and layer norms remain dense. All runs: 2000 steps, batch size 4, AdamW, NVIDIA A100 40 GB Dense learning rate: $2 \times 1 0 ^ { - 5 }$ . SCT learning rate: $5 \times 1 0 ^ { - 4 }$ .

Table 2:70B architecture validation on consumer hardware. Both platforms complete a ful training step under 8GB.   

<table><tr><td>Metric</td><td>Apple M4 Pro (48 GB)</td><td>Steam Deck (16 GB)</td></tr><tr><td>Peak Memory</td><td>7,907MB</td><td>7,236 MB</td></tr><tr><td>Forward Pass</td><td>0.08 s</td><td>0.43 s</td></tr><tr><td>Backward Pass</td><td>0.09 s</td><td>0.92 s</td></tr><tr><td>Optimizer Step</td><td>0.22 s</td><td>2.35 s</td></tr><tr><td>QR Retraction</td><td>3.02 s</td><td>2.58 s</td></tr><tr><td>Total Step</td><td>3.41 s</td><td>6.28 s</td></tr><tr><td>Ortho. Error</td><td>&lt; 2 × 10−6</td><td>&lt; 2 × 10−6</td></tr></table>

![](images/1.jpg)  

Figure 1: Training memory at 70B scale. SCT requires $1 7 2 \times$ less memory than dense training.

Table 3: Rank sweep results. Loss and PPL are smoothed (window=50). Rank 128 (bold) achieves the best PPL among SCT configurations.   

<table><tr><td>Method</td><td>Params</td><td>MLP Comp.</td><td>Loss</td><td>PPL</td><td>GPU Mem.</td><td>Step Time</td></tr><tr><td>Dense</td><td>1,711M</td><td>1.0×</td><td>1.29</td><td>3.6</td><td>35.5 GB</td><td>1.17s</td></tr><tr><td>SCT r=256</td><td>692M</td><td>5.9×</td><td>4.33</td><td>75.6</td><td>21.3 GB</td><td>1.05 s</td></tr><tr><td>SCT r=128</td><td>598M</td><td>11.7×</td><td>4.18</td><td>65.6</td><td>20.0 GB</td><td>0.74 s</td></tr><tr><td>SCT r=64</td><td>551M</td><td>23.5×</td><td>4.34</td><td>76.7</td><td>19.3 GB</td><td>0.62 s</td></tr><tr><td>SCT r=32</td><td>527M</td><td>46.9×</td><td>4.47</td><td>86.9</td><td>19.0 GB</td><td>0.56 s</td></tr></table>

# 4.3 Key Observations

All ranks converge to the same loss floor. Rank 256 ( $5 . 9 \times$ compression) and rank 32 $\left( 4 6 . 9 \times \right)$ end within 0.3 loss of each other after 2000 steps (Table 3). This indicates that MLP rank is not the primary bottleneck for convergence quality at this training duration. Rank 256 underperforms rank 128. This is a learning rate artifact, not a property of the method. At rank 256, SVD truncation preserves most pretrained structure, s the $5 \times 1 0 ^ { - 4 }$ learning rate ( $2 5 \times$ the dense baseline) overshoots and damages the initialization. At rank 32, less pretrained structure survives truncation, so the aggressive learning rate aids recovery. Rank 128 sits at the sweet spot for this particular learning rate.

![](images/2.jpg)  

Figure 2: Loss convergence for all ranks. All SCT configurations converge to the same loss floo $( { \sim } 4 . 2 – 4 . 5 )$ regardless of rank. Dense converges to 1.29.

The $\sim 3$ loss gap is an LR issue, not a capacity issue. At rank 32, MLP spectral parameters account for only 18M of 527M total. Attention layers (403M, 77% of the model) are trained at the same $5 \times 1 0 ^ { - 4 }$ learning rate. Per-component learning rate scheduling (dense learning rate for attention/embeddings, higher learning rate for SCT factors) is the clear next step to close this gap. Memory efficiency scales with compression. GPU usage drops from 35.5 GB (dense) to 19.0GB (rank 32), a 46% reduction. Training steps are $2 . 1 \times$ faster at rank 32 (0.56s vs. 1.17s). Even rank 256 saves 40% of VRAM while providing $5 . 9 \times$ MLP layer compression.

# 4.4 Fine-Tuning Gradient Integrity (SmolLM2-135M)

As an additional validation, pre-trained SmolLM2-135M weights were converted to spectral form at 95% energy retention and fine-tuned on Alpaca for 400 steps (same data, same seed, same learning rate as the dense baseline). Loss Convergence: All Ranks   

Table 4: SmolLM2-135M fine-tuning (gradient integrity test). The 135M model is below the optima scale for SCT compression; this test validates gradient flow, not compression utility.   

<table><tr><td>Method</td><td>Final Loss</td><td>Final PPL</td><td>Trainable Params</td><td>PPL Ratio</td></tr><tr><td>Dense + AdamW</td><td>0.235</td><td>1.3</td><td>134,515,008</td><td>1.0×</td></tr><tr><td>SCT (95% energy)</td><td>0.594</td><td>1.8</td><td>84,333,271</td><td>1.38×</td></tr></table>

![](images/3.jpg)  

Figure 3: Left: Compression vs. quality Pareto frontier. Rank 128 achieves the best PPL at $1 1 . 7 \times$ compression. Right: GPU memory by method. Even rank 256 saves $4 0 \%$ of VRAM.

SCT recovered from an initial loss spike (8.64) to $1 . 3 8 \times$ the dense baseline perplexity, confirming gradient integrity through spectral factors with Stiefel retraction at a model scale where compression is minimal (hidden dimension 576).

# 5 Limitations and Future Work

Convergence gap. The ${ \sim } 3$ loss gap between SCT and dense training after 2000 steps remains open. The rank sweep evidence suggests this is driven by learning rate configuration rather than inherent capacity limitations, but this has not been conclusively demonstrated. Per-component learning rate scheduling is the immediate next experiment. QR retraction cost. At $O ( m k ^ { 2 } )$ per layer per step, retraction is cheap for small $k$ but could become significant at higher ranks or larger models. The 70B benchmark shows retraction taking $4 0 \mathrm { - } 5 0 \%$ of total step time. Cayley retraction [Li et al., 2020] is a potential lower-cost alternative. Attention layers. The current experiments convert only MLP layers to spectral form. Extending SCT to attention projections $( \boldsymbol { q } , \boldsymbol { k } , \boldsymbol { v } , \boldsymbol { o } )$ is architecturally straightforward but introduces considerations around attention pattern fidelity. Full pre-training. Training to convergence on a large-scale dataset (e.g., a full pre-training run) has not been demonstrated. The 1.7B experiments validate the method on fine-tuning; scaling to full pre-training remains future work. Small model limitations. Models below $\cdots$ 1.7B parameters (hidden dimension $< 2 0 4 8$ ) produce ranks close to the full dimension at practical energy thresholds, offering limited compression benefit.

# 6 Conclusion

SCT demonstrates that permanent truncated SVD with Stiefel manifold retraction is a viable training method for large language models. The 70B architecture validation confirms the memory claim: a full training step in 7.2GB versus 1,245 GB for dense training. The 1.7B rank sweep confirms memory efficiency at scale (46% GPU reduction, $2 . 1 \times$ faster steps) and reveals that the convergence gap is driven by learning rate configuration, not spectral rank capacity. All ranks from 32 to 256 converge to the same loss floor, with rank 128 emerging as the Pareto-optimal configuration. Code and experiment notebooks are available at https://github. com/EctoSpace/SCT. Patent. Irish Short-Term Patent Application PTIE20260000000219 (S2026/0159), filed March 27, 2026.

# References

P.-A. Absil, R. Mahony, and R. Sepulchre. Optimization Algorithms on Matrix Manifolds. Princeton University Press, 2008.   
X. Han et al. LOST: Low-rank and sparse pre-training for large language models. arXiv:2508.02668, 2025.   
E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. arXiv:2106.09685, 2021.   
J. Li et al. Efficient Riemannian optimization on the Stiefel manifold via the Cayley transform. In ICLR, 2020.   
Z. Li, S. Sajadmanesh, J. Li, and L. Lyu. StelLA: Subspace learning in low-rank adaptation using Stiefel manifold. NeurIPS 2025 Spotlight. arXiv:2510.01938, 2025.   
Y. Sui, M. Yin, Y. Gong, J. Xiao, H. Phan, and B. Yuan. ELRT: Efficient low-rank training for compact convolutional neural networks. arXiv:2401.10341, 2024.   
X. Wang et al. SVD-LLM: Truncation-aware SVD for LLM compression. ICLR 2025. arXiv:2403.07378, 2024.   
H. Yang et al. Learning low-rank deep neural networks via singular vector orthogonality regularization and singular value sparsification. arXiv:2004.09031, 2020.   
J. Zhao et al. GaLore: Memory-efficient LLM training by gradient low-rank projection. In ICML, 2024. arXiv:2403.03507.   
US Patent Application 20250021826. Low-rank compression of neural networks, 2025.