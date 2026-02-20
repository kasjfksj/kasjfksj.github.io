---
layout: post
title: Diffusion Models for Image Compression and Transmission
date: 2026-02-06 19:48:23
description: 
categories: AI
giscus_comments: true
tabs: true
related_posts: false
toc: 
  sidebar: left
---

# Diffusion Models for Low-Bitrate Image Transmission

Traditional image compression fails at low bitrates, producing artifacts and blur. Diffusion models offer a solution by using generative priors to reconstruct plausible images from tiny bitstreams. Here I'm introducing one of the simplest way of doing diffusion for image transmitssion.

## The Core Mechanism

Supposedly you have a pretrained diffusion model that samples from distribution $$p(x_{t-1} \mid x_t)$$. In order to retrieve an image from such distribution, we need to recursively apply diffusion process on noisy images to produce a clean image. But to reconstruct a specific image accurately, you need a sample from $$q(x_{t-1} \mid x_t, x_0)$$, which is a slightly different distribution that depends on the original image.

The question is how to communicate a sample from $$q(x_{t-1} \mid x_t, x_0)$$ where you only have access to $$p(x_{t-1} \mid x_t)$$? 

The simplest way is to use rejection sampling, where:

1. Sample $$x_{t-1}$$ from $$p(x_{t-1} \mid x_t)$$
2. Accept with probability $$\frac{q(x_{t-1} \mid x_t, x_0)}{M \cdot p(x_{t-1} \mid x_t)}$$
3. If rejected, repeat until acceptance

The encoder performs this rejection sampling to find the accepted sample at index $$k$$. 

Now here comes a clever trick. Instead of directly sending such sample, the encoder sends $$k$$ to the receiver. The receiver generates samples from $$p(x_{t-1} \mid x_t)$$ using the same shared random seed and stops at the $$k$$-th sample, which is the accepted one. Because senders and receivers are using the same seed, it's guaranteed that the sequence of samples are identical. So the receiver can always get the same sample as the sender.

The number of bits required to transmit the index $$k$$ depends fundamentally on the divergence between $$q$$ and $$p$$. The expected number of rejection sampling trials before acceptance is $M$, where $M$ is the rejection sampling constant satisfying $q(x_{t-1} \mid x_t, x_0) \leq M \cdot p(x_{t-1} \mid x_t)$ for all $x_{t}$.

The expected value of $$k$$ follows a geometric distribution with success probability $1/M$, so $$\mathbb{E}[k] = M$$. To encode an index drawn from a geometric distribution, we need approximately:

$$\mathbb{E}[\text{bits}] \approx \log_2(M) + 2$$

bits on average.

More fundamentally, the communication cost is related to the KL divergence between $q$ and $p$:

$$D_{KL}(q \ \mid  p) = \mathbb{E}_{x_{t-1} \sim q}\left[\log \frac{q(x_{t-1} \mid x_t, x_0)}{p(x_{t-1} \mid x_t)}\right]$$

While $$M$$ provides an upper bound on the ratio $$q/p$$, the KL divergence captures the average information difference. The relationship can be bounded by:

$$\log M \geq D_{KL}(q \ \mid  p)$$

with equality when $q = p$ (in which case $M = 1$ and no communication is needed).

For the entire diffusion process from $t = T$ to $t = 0$, the total communication cost is:

$$\text{Total bits} \approx \sum_{t=1}^{T} D_{KL}(q(x_{t-1} \mid x_t, x_0) \ \mid  p(x_{t-1} \mid x_t))$$

This sum represents the total information gap that must be communicated to reconstruct the specific image $$x_0$$ from the diffusion model. When the pretrained model $$p$$ closely matches the posterior $$q$$, the KL divergence is small and very few bits are needed. Conversely, if $$p$$ is a poor approximation of $$q$$, many more rejection sampling trials are required, increasing the communication cost substantially.

# UQDM

The paper "Progressive Compression with Universally Quantized Diffusion Models" addresses the computational intractability of the above approach when using Gaussian distributions. The key insight is to replace Gaussian noise channels with uniform noise channels, enabling the use of **Universal Quantization (UQ)** instead of exponentially-complex Relative Entropy Coding.

### Forward Process

The forward process starts with a Gaussian initialization and then uses uniform noise channels:

$$q(z_T | x) = \mathcal{N}(\alpha_T x, \sigma_T^2 I)$$

$$q(z_{t-1} | z_t, x) = \mathcal{U}\left(b(t)z_t + c(t)x - \frac{\Delta(t)}{2}, b(t)z_t + c(t)x + \frac{\Delta(t)}{2}\right)$$

where the parameters are chosen to match the moments of Gaussian diffusion:

$$b(t) = \frac{\alpha_t}{\alpha_{t-1}} \frac{\sigma_{t-1}^2}{\sigma_t^2}, \quad c(t) = \frac{\sigma_{t|t-1}^2 \alpha_{t-1}}{\sigma_t^2}, \quad \Delta(t) = \sqrt{12} \sigma_{t|t-1} \frac{\sigma_{t-1}}{\sigma_t}$$

### Backward Process

The reverse process uses a learned density model convolved with uniform noise:

$$p(z_{t-1}|z_t) = g_\theta(z_{t-1}; z_t, t) \star \mathcal{U}(-\Delta(t)/2, \Delta(t)/2)$$

where $g_\theta$ is typically chosen as a Gaussian or logistic distribution:

$$g_\theta(z_{t-1}; z_t, t) = \text{Logistic}(b(t)z_t + c(t)\hat{x}_\theta(z_t; t), \sigma_Q^2(t) I)$$

The model is trained by minimizing the negative ELBO:

$$L(x) = \underbrace{KL(q(z_T|x) \| p(z_T))}_{L_T} + \underbrace{\mathbb{E}[-\log p(x|z_0)]}_{L_{x|z_0}} + \sum_{t=1}^{T} \underbrace{\mathbb{E}[KL(q(z_{t-1}|z_t, x) \| p(z_{t-1}|z_t))]}_{L_{t-1}}$$

### Progressive Compression Algorithm

**Encoding Process:**
1. **Initialize:** Draw $z_T \sim p(z_T) = \mathcal{N}(0, I)$ using a shared random seed
2. **For each step** $t = T, T-1, \ldots, 1$:
   - Compute the mean: $\mu_Q = b(t)z_t + c(t)x$
   - Draw uniform noise: $u_t \sim \mathcal{U}(-1/2, 1/2)$
   - Quantize: $k_t = \left\lfloor \frac{\mu_Q}{\Delta(t)} + u_t \right\rceil$
   - Entropy encode and transmit $k_t$ using the discretized $p(z_{t-1}|z_t)$
   - Update: $z_{t-1} = \Delta(t)(k_t - u_t)$
3. **Final step:** Entropy encode $x$ using $p(x|z_0)$

**Decoding Process:**
1. **Initialize:** Draw $z_T \sim p(z_T) = \mathcal{N}(0, I)$ using the same shared seed
2. **For each step** $t = T, T-1, \ldots, 1$:
   - Compute parameters of $p(z_{t-1}|z_t)$ using the denoising network $\hat{x}_\theta(z_t; t)$
   - Draw the same uniform noise: $u_t \sim \mathcal{U}(-1/2, 1/2)$ using shared seed
   - Entropy decode $k_t$ using $p(z_{t-1}|z_t)$
   - Reconstruct: $z_{t-1} = \Delta(t)(k_t - u_t)$
   - **Lossy reconstruction (progressive):** $\hat{x}_t = \hat{x}_\theta(z_{t-1}; t-1)$
3. **Lossless reconstruction:** Entropy decode $x$ using $p(x|z_0)$


## Conclusion

Diffusion compression works by sharing randomness between encoder and decoder. Instead of transmitting full samples, you only send indices telling the decoder which sample to pick from a shared random sequence. Combined with a compact content latent, this achieves remarkable compression at ultra-low bitrates.