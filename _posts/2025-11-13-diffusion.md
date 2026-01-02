---
layout: post
title: Diffusion series - DDPM model
date: 2025-08-19 15:53:28
description: 
tags: formatting images
categories: AI
tabs: true
related_posts: false
toc: 
  sidebar: left
---

# Introduction

The purpose writing this blog is to document what I've learnt so far about diffusion. To me diffusion is a powerful paradigm and a very interested research area where math (especially probability) collide with AI most. So I want to document everything I've learnt so far so that readers can gain better understanding of diffusion model. 

Since there are so many stuff about diffusion, I will write several blogs on it.

The beginning of diffusion model is for image generation. Thus, before discussing diffusion model. First discuss the task of image generation and the probability behind it

# Probability view of Image generation


The task of image generation is simple: generating realistic images that align with the style of training data, but what does it even mean to “generate realistic images”?

Having a strict standard or formula to dictate the realness of an image is not feasible as there are numerous pixels to be considered and the big number of pictures to consider. Therefore, it's much more appropriate to consider form a probability point of view where we basically want to sample from an unknown probability distribution of real images, denoted as $$p_{data}(x)$$. 

$$x$$ with high $$p_{data}(x)$$ look like real photographs,
x with tiny (near-zero) $$p_{data}(x)$$ look like random color patches or just noise.

However, such distribution is unknown. Therefore it's best to have a model that can model data distribution with its own distribution of $$p_{\theta}(x)$$ where we want such distribution is as close to $$p_{data}(x)$$ as possible. 

And the objective should be     $$\underset{\theta}{\argmin} \,\mathcal{F}(p_{data}(x),\,p_{\theta}(x))$$ where $$\mathcal{F}$$ is the measurement of the distance of two distributions.

Of course, the traditional way of measuring distance of 2 distributions is KL divergence, which is given as follow:

$$  
D_{\text{KL}}\!\bigl(p_{\text{data}} \,\|\, p_\theta\bigr)
= \mathbb{E}_{x \sim p_{\text{data}}(x)} \!\Bigl[\log p_{\text{data}}(x) - \log p_\theta(x)\Bigr]
  $$

**Note:** $$ p_{\text{data}}(x) $$ is fixed (it’s the real world), so minimizing this KL is **exactly** equivalent to maximizing:

$$  

\mathbb{E}_{x \sim p_{\text{data}}(x)} \!\bigl[\log p_\theta(x)\bigr]

$$

## VAE

However, learning such $$p_{\theta}(x)$$ is not so easy since $$x$$ typically lies in a very high-dimensional space — for example, a 256×256 color image has dimension 196,608. Modeling the distribution on such high dimension typically causes greater computation and curse of dimension (volume grows exponentially with dimension, requiring exponential number of data to accurately estimate the true distribution of the data).

Thus, we want to introduce a model where it generates samples on low-dimensional latent space $$z$$, 128 dimension or 512 dimension. We denote such probabilistic model as $$p_\theta(x \mid z) \;=\; \mathcal{N}\big(x \mid \mu_\theta(z),\; \sigma^2 I\big)$$ where $$\mu_θ(z)$$ is the image output by the decoder network.

Also we want the latent space have certain prior, usually $\mathcal{N}(0,I)$, so that we can easily sampled $$z$$. The observed data is then modeled as:
$$p_\theta(x) = \int p_\theta(x \mid z) \, p(z) \, dz$$

where $$p_\theta(x \mid z)$$ is a flexible decoder (typically a deep neural network)

You might wonder at this point, why don't we directly train a model via monte carlo estimation where $$p_\theta(x) \approx \frac{1}{S} \sum_{s=1}^S p_\theta(x \mid z_s),\quad z_s \sim p(z)$$

This would work mathematically. But in reality, it performs very bad, because the prior is still in high-dimension, so most sampled $$z_s \sim \mathcal{N}(0,I)$$ have tiny (almost zero) likelihood $$p_\theta(x \mid z_s)$$ 

Occasionally, by pure chance, you draw a $$z_s$$ that is close to the region where the decoder puts mass on $$x$$; then $$p_\theta(x \mid z_s)$$ is huge. This is why training model via Monte Carlo estimate from purely Gaussian distribution has insane variance and fails completely.

But during training we are not blind — we have the actual data point x as prior. So instead of randomly sample from the prior, we can directly sample $$z$$ based on original image $$x$$. This can be modeled as conditional distribution $$q_{\phi}(z\mid x)$$ which is essentially mapping image distribution to the latent distribution. Such conditional distribution is also parameterized by an encoder network which produces the mean and the variance:
$$q_\phi(z \mid x) \;=\; \mathcal{N}\!\left(z \;\middle|\; \mu_\phi(x),\; \operatorname{diag}\!\big(\sigma_\phi^2(x)\big)\right)$$

With this in mind, we can now rewrite $$\log p_{\theta}(x)$$ as the following



$$\log p_\theta(x) = \int q_\phi(z | x) \log p_\theta(x) \, dz$$


$$= \int q_\phi(z | x) \log \frac{p_\theta(x, z)}{p_\theta(z | x)} \, dz$$

$$= \int q_\phi(z | x) \log \frac{p_\theta(x, z)}{q_\phi(z | x)} \, dz + \int q_\phi(z | x) \log \frac{q_\phi(z | x)}{p_\theta(z | x)} \, dz$$


$$= \int q_\phi(z | x) \log \frac{p_\theta(x, z)}{q_\phi(z | x)} \, dz + D_{\text{KL}}(q_\phi(z | x) \| p_\theta(z | x))$$


$$\geq \int q_\phi(z | x) \log \frac{p_\theta(x, z)}{q_\phi(z | x)} \, dz \quad \text{(since KL} \geq 0\text{)}$$

Now rewite this integral form:

$$
\int q_\phi(z | x) \log \frac{p_{\theta}(x | z) p(z)}{q_\phi(z | x)} dz = \int q_\phi(z | x) \log p_{\theta}(x | z) dz + \int q_\phi(z | x) \log \frac{p_{\theta}(z)}{q_\phi(z | x)} dz
$$

$$
= \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x | z) \right]
 - D_{\text{KL}}(q_\phi(z | x) \| p_{\theta}(z))
$$


summing all up:

$$\log p_{\theta}(x) \ge \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x | z) \right]
 - D_{\text{KL}}(q_\phi(z | x) \| p_{\theta}(z))$$

Remember that our objective is to maximizing 

$$  
\mathbb{E}_{x \sim p_{\text{data}}(x)} \!\bigl[\log p_\theta(x)\bigr]
$$

So the training objective wil become minimizing:

$$
-\mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x | z) \right] + D_{\text{KL}}(q_\phi(z|x) \| p(z)) \quad x \sim p_{data}(x)
$$

The first term pushes the decoder to reconstruct $$x$$ well from encoder samples.

The second term forces the encoder to stay close to the standard Gaussian prior. 

In practice we take one sample $$z \sim q_\phi(z \mid x) $$ (reparameterised) for the expectation and compute the KL analytically. (For more details check online description of VAE architecture and the loss)

## DDPM

