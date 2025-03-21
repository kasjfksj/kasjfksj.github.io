---
layout: post
title: Understanding Attention and Multi-Head Attention - From Basics to RoPE Optimization
date: 2025-03-20 16:15:09
description: 
tags: formatting images
categories: model architecture
tabs: true
related_posts: false
toc:  
  sidebar: left
---

# Intro
In recent years, Transformer models have achieved tremendous success in the field of Natural Language Processing (NLP), with the Attention mechanism being a core component. This article will delve into the principles of the Attention mechanism and its extension, Multi-Head Attention, while introducing an optimization method—Rotary Position Embedding (RoPE), which significantly improves model performance.

# Core Principles of the Attention Mechanism
The Attention mechanism was originally used to address long-range dependency issues in sequence-to-sequence (Seq2Seq) models. Its core idea is that, when generating each output, the model can "dynamically focus" on different parts of the input sequence rather than relying on fixed context.

## Scaled Dot-Product Attention
In Transformers, Attention is formalized as Scaled Dot-Product Attention. Given query matrices (Query), key matrices (Key), and value matrices (Value), the computation process is as follows:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:<br>
    $$Q \in \mathbb{R}^{n \times d_k}$$ :Query matrix,<br>
    $$K \in \mathbb{R}^{m \times d_k}$$ :Key matrix, <br>
    $$V \in \mathbb{R}^{m \times d_v}$$ :Value matrix, <br>
    $$d_k$$ :Dimension of the key vector,<br>
    $$\sqrt{d_k}$$ :Scaling factor to prevent excessively large dot product results.<br>


## Dynamic Weight Allocation
Attention calculates the similarity between queries and keys to generate a weight matrix, then performs weighted summation over the value matrix. This process can be decomposed into:

Similarity Calculation: $$ QK^T $$ measures the match between queries and keys,
Scaling and Normalization: Divide by $$\sqrt{d_k}$$ and normalize using softmax.
Weighted Summation : Apply the normalized weights to the value matrix $$V$$ to generate the final output.

# Multi-Head Attention: The Power of Parallelization
To enhance the model's expressive power, Transformers introduced Multi-Head Attention. Its core idea is to capture information from different subspaces through multiple independent Attention heads.

## Mathematical Form of Multi-Head Attention
Linear Transformation : For each head, map $$Q$$, $$K$$, $$V$$ to different subspaces:
$$Q_i = QW_i^Q, \quad K_i = KW_i^K, \quad V_i = VW_i^V$$        

Where<br>
    $$W_i^Q \in \mathbb{R}^{d_k \times d_k}$$,<br>
    $$W_i^K \in \mathbb{R}^{d_k \times d_k}$$,<br>
    $$W_i^V \in \mathbb{R}^{d_v \times d_v}$$ are learnable weight matrices.<br>


Parallel Computation of Attention : Each head independently computes Attention:<br>
$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

Concatenation and Output Transformation : Concatenate the outputs of all heads and apply a linear transformation:<br>
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

Where:<br>
    $$W^O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$$: Output weight matrix,<br>
    $$h$$:  Number of heads.<br>

## Why Use Multi-Head?
1. Multi-Perspective Modeling : Each head learns different feature patterns (e.g., syntax, semantics).
2. Enhanced Robustness : Avoids overfitting noise with a single Attention head.
3. Experimental Validation : In machine translation tasks, Multi-Head Attention reduces perplexity by approximately 20% compared to single-head Attention.

# Rotary Position Embedding (RoPE): Revolutionizing Position Encoding

Traditional Transformers use absolute position encoding (e.g., sine functions) to introduce sequence order information, but modeling positional relationships in long sequences remains limited. RoPE incorporates positional information into Attention calculations via rotation matrices, significantly improving performance.

## Mathematical Definition of RoPE
For positions $$m$$ and $$n$$，RoPE transforms query and key vectors using rotation matrices $$R_m$$ and $$R_n$$:

$$
\boldsymbol{q}' = R_m\boldsymbol{q}, \quad \boldsymbol{k}' = R_n\boldsymbol{k}
$$

Where the rotation matrix$$R(m)$$ is defined as:

$$
\mathbf{R}_t = \begin{pmatrix}
\cos t\theta_1 & -\sin t\theta_1 & \cdots & 0 \\
\sin t\theta_1 & \cos t\theta_1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \cos t\theta_{d/2} & -\sin t\theta_{d/2} \\
0 & 0 & \cdots & \sin t\theta_{d/2} & \cos t\theta_{d/2}
\end{pmatrix}
$$

## Physical Meaning of RoPE
Relative Position Encoding: Rotation operations make Attention scores depend only on relative positions $$ m - n $$

Theoretical Guarantee : RoPE satisfies translational invariance for position encoding $$\mathbf{q}_m' \cdot \mathbf{k}_n' = f(m - n)$$.

$$
\mathbf{q}'_m \cdot \mathbf{k}'_n = \sum_{i=1}^{d/2} \left[ q_m^{(2i-1)}k_n^{(2i-1)} + q_m^{(2i)}k_n^{(2i)} \right]\cos((m-n)\theta_i) \\ 
+ \left[ q_m^{(2i)}k_n^{(2i-1)} - q_m^{(2i-1)}k_n^{(2i)} \right]\sin((m-n)\theta_i)
$$

## Experimental Results
Using RoPE, the language model's perplexity dropped significantly from 144 to 97 (a reduction of 32.6%), especially effective in long-text generation tasks.

# Workflow explanation

## Parameter Initialization Deep Dive

Firstly, we initialize our Q, K, V matrices and the matrix $$W_O$$:

```python
# d_model is the dimension of word embedding 
self.qkv = torch.nn.Parameter(0.01*torch.randn((3*d_model, d_model)))
self.wo = torch.nn.Parameter(0.01*torch.randn((d_model, d_model)))
```

Why Combined QKV Matrix instead of using separate 3 matrices?

•	Architectural Efficiency: This allows for single contiguous memory block and better hardware utilization
•	Implementation Insight: Common pattern in modern transformers (e.g., GPT-2) vs older separate projections (original Transformer paper)

Dimension of Input:<br>
•	Input: (B, S, D) (Batch × Sequence × ModelDim)<br>
•	Projection: (3D, D) matrix transforms D → 3D features<br>
•	Output after x @ qkv.T: (B, S, 3D)<br>
Initialization Scale:<br>
•	0.01*randn keeps initial weights small to prevent large softmax gradients early in training<br>
 <br>
<br>
2. QKV Projection & Splitting

```python
QKV = x @ self.qkv.T  # (B, S, D) -> (B, S, 3D)
Q, K, V = torch.chunk(QKV, 3, -1)  # Each with (B, S, D)
```

Visualization:
For D=512:<br>
Input x: [Batch, SeqLen, 512]<br>
QKV:     [Batch, SeqLen, 1536]  # 3×512<br>
Split → [Batch, SeqLen, 512] each for Q/K/V<br>
Design Choice: Single projection vs multiple:<br>
•	Pro: Reduces memory fragmentation<br>
•	Con: Limits flexibility in head-specific projections<br>
 <br>
3. Multi-Head Splitting Mechanics

```python

# H: the number of heads; d_h: dimension for each head
dh = D // self.n_heads  # Head dimension
q_heads = Q.view(B, S, self.n_heads, dh).transpose(1,2)  # (B, S, D) -> (B, S, H, d_h) -> (B, H, S, d_h)

```

Example with Numbers:<br>
•	Let D=8, H=2 (heads) → dh=4<br>
•	Original Q: (2, 5, 8) (Batch=2, Seq=5)<br>
•	After view: (2, 5, 2, 4)<br>
•	Transpose: (2, 2, 5, 4) (Head dimension at position 1)<br>
Why Transpose?<br>
•	Aligns dimensions for batch matrix multiply in attention:<br>
o	(B, H, S, dh) @ (B, H, dh, S) → (B, H, S, S)<br>
Head Specialization:<br>
•	Each head processes dh-dimensional subspace (e.g., 512-dim → 64×8 heads)<br>
•	Enables capturing diverse linguistic features:<br>
o	Head 1: Subject-verb agreement<br>
o	Head 2: Pronoun references<br>
o	Head 3: Temporal relationships<br>
 
4. Attention Mask Construction
```python
mask = torch.tril(torch.ones(S, S_full), diagonal=past_length)
sq_mask = mask == 1  # Convert to boolean
```

Causal Mask Visualization (S=3):<br>
[[1 0 0]<br>
 [1 1 0]<br>
 [1 1 1]]<br>
•	Prevents attending to future positions in autoregressive generation<br>
•	Why Not Full Masking? Preserves parallel computation during training<br>
 <br>
5. Scaled Dot-Product Attention Core
```python
x = torch.nn.functional.scaled_dot_product_attention(
    q_heads, k_heads, v_heads,  # Shapes: (B, H, S, dh)
    attn_mask=sq_mask
)
```

Under the Hood:<br>
1.	Score Calculation:<br>
Scores=dhQKT<br>
•	Scaling prevents gradient saturation in softmax<br>
2.	Masked Softmax:<br>
AttentionWeights=softmax(Scores+MaskBias)<br>
•	Where MaskBias = -∞ for masked positions<br>
3.	Value Weighting:<br>
Output=AttentionWeights⋅V<br>
Optimization: Uses fused CUDA kernel for:<br>
•	Memory-efficient attention computation<br>
•	Automatic mixed-precision support<br>
 
6. Output Projection & Head Merging

```python
x = x.transpose(1, 2).reshape(B, S, D)  # (B, S, H, dh) → (B, S, D)
x = x @ self.wo.T  # Final projection
```

Dimension Restoration:<br>
After attention: (B, H, S, dh)<br>
Transpose → (B, S, H, dh)<br>
Reshape → (B, S, H*dh) = (B, S, D)<br>
Why Final Projection (wo)?<br>
1.	Feature Integration: Combines information from all heads<br>
2.	Dimension Matching: Ensures output matches original d_model<br>
3.	Learnable Mixing: Allows model to emphasize important heads<br>
 <br>
 ```markdown
Complete Dimension Flow Table
| Operation         | Input Shape       | Output Shape      | Key Notes               |
| :---------------- | :---------------: | ----------------: | :---------------------- |
| Input             | (B, S, D)         | -                 | Base embeddings         |
| QKV Projection    | (B, S, D)         | (B, S, 3D)        | Single matrix multiply  |
| Q/K/V Split       | (B, S, 3D)        | 3×(B, S, D)       | Chunk on last dim       |
| Head Splitting    | (B, S, D)         | (B, H, S, dh)     | View + transpose        |
| Attention         | 3×(B, H, S, dh)   | (B, H, S, dh)     | Masked softmax          |
| Head Merge        | (B, H, S, dh)     | (B, S, D)         | Transpose + reshape     |
| Output Projection | (B, S, D)         | (B, S, D)         | Final feature mixing    |
 ```
<br>
# Summary and Outlook
Attention : Captures sequence dependencies through dynamic weight allocation.
Multi-Head : Parallelizes modeling of multi-dimensional features, increasing model capacity.
RoPE : Innovates position encoding methods, significantly reducing perplexity.
Future directions include:

More efficient position encoding methods (e.g., hybrid absolute/relative position encoding).
Sparse acceleration of Attention computation (e.g., Longformer, BigBird).





To Do: Add architecture diagrams showing:
b) Multi-head attention's parallel processing
c) RoPE's rotation matrix operations

proof of translational invariance for position encoding

add code to showcase procedure

Add training optimization tips:
Include real-world impact metrics