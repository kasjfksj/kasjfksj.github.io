---
layout: post
title: Diffusion series - DDPM model
date: 2025-08-19 15:53:28
description: 
tags: formatting images
categories: AI
tabs: true
related_posts: false
giscus_comments: true
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

And the objective should be     $$\underset{\theta}{argmin} \,\mathcal{F}(p_{data}(x),\,p_{\theta}(x))$$ where $$\mathcal{F}$$ is the measurement of the distance of two distributions.

Of course, the traditional way of measuring distance of 2 distributions is KL divergence, which is given as follow:

$$  
D_{\text{KL}}\!\bigl(p_{\text{data}} \,\ \mid \, p_\theta\bigr)
= \mathbb{E}_{x \sim p_{\text{data}}(x)} \!\Bigl[\log p_{\text{data}}(x) - \log p_\theta(x)\Bigr]
  $$

**Note:** $$ p_{\text{data}}(x) $$ is fixed (it’s the real world), so minimizing this KL is **exactly** equivalent to maximizing:

$$  

\mathbb{E}_{x \sim p_{\text{data}}(x)} \!\bigl[\log p_\theta(x)\bigr]
$$

## VAE

However, learning such $$p_{\theta}(x)$$ is not so easy since $$x$$ typically lies in a very high-dimensional space — for example, a 256×256 color image has dimension 196,608. Modeling the distribution on such high dimension typically causes greater computation and curse of dimension (volume grows exponentially with dimension, requiring exponential number of data to accurately estimate the true distribution of the data).

Thus, we want to introduce a model where it generates samples on low-dimensional latent space $$z$$, 128 dimension or 512 dimension. We denote such probabilistic model as $$p_\theta(x \mid z) \;=\; \mathcal{N}\big(x \mid \mu_\theta(z),\; \sigma^2 I\big)$$ where $$\mu_θ(z)$$ is the image output by the decoder network.

Also we want the latent space have certain prior, usually $$\mathcal{N}(0,I)$$, so that we can easily sampled $$z$$. The observed data is then modeled as:
$$p_\theta(x) = \int p_\theta(x \mid z) \, p(z) \, dz$$

where $$p_\theta(x \mid z)$$ is a flexible decoder (typically a deep neural network)

You might wonder at this point, why don't we directly train a model via monte carlo estimation where $$p_\theta(x) \approx \frac{1}{S} \sum_{s=1}^S p_\theta(x \mid z_s),\quad z_s \sim p(z)$$

This would work mathematically. But in reality, it performs very bad, because the prior is still in high-dimension, so most sampled $$z_s \sim \mathcal{N}(0,I)$$ have tiny (almost zero) likelihood $$p_\theta(x \mid z_s)$$ 

Occasionally, by pure chance, you draw a $$z_s$$ that is close to the region where the decoder puts mass on $$x$$; then $$p_\theta(x \mid z_s)$$ is huge. This is why training model via Monte Carlo estimate from purely Gaussian distribution has insane variance and fails completely.

But during training we are not blind — we have the actual data point x as prior. So instead of randomly sample from the prior, we can directly sample $$z$$ based on original image $$x$$. This can be modeled as conditional distribution $$q_{\phi}(z\mid x)$$ which is essentially mapping image distribution to the latent distribution. Such conditional distribution is also parameterized by an encoder network which produces the mean and the variance:
$$
q_\phi(z \mid x) = \mathcal{N}\bigl(z \;\big|\; \mu_\phi(x),\,\operatorname{diag}(\sigma_\phi^2(x))\bigr)
$$

With this in mind, we can now rewrite $$\log p_{\theta}(x)$$ as the following



$$
\begin{aligned}
\log p_\theta(x) &= \int q_\phi(z  \mid  x) \log p_\theta(x) \, dz \\ &= \int q_\phi(z  \mid  x) \log \frac{p_\theta(x, z)}{p_\theta(z  \mid  x)} \, dz \\ &= \int q_\phi(z  \mid  x) \log \frac{p_\theta(x, z)}{q_\phi(z  \mid  x)} \, dz + \int q_\phi(z  \mid  x) \log \frac{q_\phi(z  \mid  x)}{p_\theta(z  \mid  x)} \, dz \\ &= \int q_\phi(z  \mid  x) \log \frac{p_\theta(x, z)}{q_\phi(z  \mid  x)} \, dz + D_{\text{KL}}(q_\phi(z  \mid  x) \ \mid  p_\theta(z  \mid  x))
\\ &
\geq \int q_\phi(z  \mid  x) \log \frac{p_\theta(x, z)}{q_\phi(z  \mid  x)} \, dz \quad \text{(since KL} \geq 0\text{)}
\end{aligned}
$$

Now rewite this integral form:

$$
\int q_\phi(z  \mid  x) \log \frac{p_{\theta}(x  \mid  z) p(z)}{q_\phi(z  \mid  x)} dz = \int q_\phi(z  \mid  x) \log p_{\theta}(x  \mid  z) dz + \int q_\phi(z  \mid  x) \log \frac{p_{\theta}(z)}{q_\phi(z  \mid  x)} dz
$$

$$
= \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x  \mid  z) \right]
 - D_{\text{KL}}(q_\phi(z  \mid  x) \ \mid  p_{\theta}(z))
$$


summing all up:

$$\log p_{\theta}(x) \ge \mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x  \mid  z) \right]
 - D_{\text{KL}}(q_\phi(z  \mid  x) \ \mid  p_{\theta}(z))$$

Remember that our objective is to maximizing 

$$  
\mathbb{E}_{x \sim p_{\text{data}}(x)} \!\bigl[\log p_\theta(x)\bigr]
$$

So the training objective wil become minimizing:

$$
-\mathbb{E}_{q_\phi(z \mid x)} \left[ \log p_\theta(x  \mid  z) \right] + D_{\text{KL}}(q_\phi(z \mid x) \ \mid  p(z)) \quad x \sim p_{data}(x)
$$

The first term pushes the decoder to reconstruct $$x$$ well from encoder samples.

The second term forces the encoder to stay close to the standard Gaussian prior. 

In practice we take one sample $$z \sim q_\phi(z \mid x) $$ (reparameterised) for the expectation and compute the KL analytically. (For more details check online description of VAE architecture and the loss)

## DDPM

So far, Variational Autoencoders seems to work well on learning compressed representations of data. They encode an image into a small latent vector and decode it back, while trying to keep the latent codes close to a simple Gaussian distribution $$\mathcal{N}(\mathbf{0}, \mathbf{I})$$.

The problem is that real-world data (like photos of faces or animals) has a very complicated distribution. Forcing everything through a small bottleneck and a simple Gaussian prior makes it hard for VAEs to capture all the sharp details. As a result, images generated by VAEs often look blurry.

So, can we model complex data distributions more accurately?

The answer is yes. Instead of trying to jump from simple Gaussian noise to complex data in one huge step (like VAEs do with decoding), we build a smooth probabilistic trajectory connecting the data distribution to a simple Gaussian noise. Each point on this trajectory is a slightly noisier version of the data. By breaking the entire path into many small steps, the change between consecutive points becomes tiny and easy to model. The neural network only needs to learn these small perturbations—predicting and removing a little noise at each step—which is a much simpler task than reconstructing the full image all at once. 

In practice, we can define a forward process that iteratively adds predetermined Gaussian noise to an image and then try to learn a backward process to convert Gaussian noise to image.

The concept is intuitive, but the math that makes it work efficiently is clever and complicated.

### Forward Process: Gradually Adding Noise
We start with a clean image $$\mathbf{x}_0$$. At each timestep $$t = 1$$ to $$T$$ (usually $$T \approx 1000$$), we add a small independent Gaussian noise:

$$
\mathbf{x}_t = \sqrt{1 - \beta_t} \, \mathbf{x}_{t-1} + \sqrt{\beta_t} \, \boldsymbol{\epsilon_t}, \quad \boldsymbol{\epsilon_t} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Here $$\beta_t$$ is a small variance schedule ($$\beta_t \to 1$$ as $$t$$ increases).

To make future mathematical deduction easier, let $$\alpha_t = 1 - \beta_t$$, the formula becomes: 

$$
\mathbf{x}_t = \sqrt{\alpha_t} \, \mathbf{x}_{t-1} + \sqrt{1-\alpha_t} \, \boldsymbol{\epsilon_t}
$$

Basically, we define the forward process to be:
$$p(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \sim \mathcal{N}(\sqrt{\alpha_t} \, \mathbf{x}_{t-1}, 1-\alpha_t)$$

Before retrieving backward process $$p(\mathbf{x}_{t-1}  \mid  \mathbf{x}_t)$$, we need to make a mathematical transformation:

$$
\begin{aligned}
\mathbf{x}_t &= \sqrt{\alpha_t} \, \mathbf{x}_{t-1} + \sqrt{1-\alpha_t} \, \boldsymbol{\epsilon_t} \\ &= \sqrt{\alpha_t} \, (\sqrt{\alpha_{t-1}} \, \mathbf{x}_{t-2} + \sqrt{1-\alpha_{t-1}} \, \boldsymbol{\epsilon_{t-1}}) + \sqrt{1-\alpha_t} \, \boldsymbol{\epsilon_t} \\ &= \sqrt{\alpha_t} \, \sqrt{\alpha_{t-1}} \, \mathbf{x}_{t-2} + \sqrt{\alpha_t} \,\sqrt{1-\alpha_{t-1}} \, \boldsymbol{\epsilon_{t-1}}+\sqrt{1-\alpha_t} \, \boldsymbol{\epsilon_t} 
\end{aligned}
$$

We can treat $$\sqrt{\alpha_t} \,\sqrt{1-\alpha_{t-1}} \, \boldsymbol{\epsilon_{t-1}} $$ and $$\sqrt{1-\alpha_t} \, \boldsymbol{\epsilon_t}$$ as representation of Gaussian distributions:

$$
\begin{aligned}
\mathbf{X}_1 
&\sim \sqrt{\alpha_t}\,\sqrt{1-\alpha_{t-1}}\,\boldsymbol{\epsilon}_{t-1}
= \mathcal{N}\!\left(0,\, \alpha_t(1-\alpha_t)\right) \\
\mathbf{X}_2 
&\sim \sqrt{1-\alpha_t}\,\boldsymbol{\epsilon}_t
= \mathcal{N}\!\left(0,\, (1-\alpha_t)\right)
\end{aligned}
$$

Since the noise are independently added, $$\mathbf X_1 + \mathbf X_2 \sim \mathcal{N}(0, \, 1-\alpha_t \, \alpha_{t-1})$$, so the formula becomes:
$$\mathbf{x}_t = \sqrt{\alpha_t \, \alpha_{t-1}} \, \mathbf{x}_{t-2} + \sqrt{1-\alpha_t \, \alpha_{t-1}} \, \epsilon $$

Applying this procedure recursively, we'll get:

$$\mathbf{x}_t = \sqrt{\alpha_t \, \alpha_{t-1}\,...\, \alpha_1} \, \mathbf{x}_0 + \sqrt{1-\alpha_t \, \alpha_{t-1} \, ... \, \alpha_1} \, \epsilon $$

Let $$\bar{\alpha_t} = \alpha_t \, \alpha_{t-1}\,...\, \alpha_1$$, we can simplify the above fomula to be $$\mathbf{x}_t = \sqrt{\bar{\alpha_t}} \, \mathbf{x}_0 + \sqrt{1-\bar{\alpha_t}}\, \epsilon$$

The forward process becomes $$p(\mathbf{x}_t  \mid  \mathbf{x}_0) \sim \mathcal{N}(\sqrt{\bar{\alpha_t}} \, \mathbf{x}_0, 1-\bar{\alpha_t})$$

### Backward Process

In order to get reverse process $$p(\mathbf{x}_{t-1}  \mid  \mathbf{x}_t)$$, we can try to apply bayesian rule: 

$$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \frac{p(\mathbf{x}_t  \mid  \mathbf{x}_{t-1}) \, p(\mathbf{x}_{t-1})}{p(\mathbf{x}_t)}$$

However, the problem is that $$p(\mathbf{x}_t)$$ and $$p(\mathbf{x}_{t-1})$$ have no explicit formula, so the alternative way is to include the source image $$\mathbf{x}_0$$, which rewrites the formula as the following:

$$
\begin{aligned}
p(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) &= \frac{p(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) \, p(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{p(\mathbf{x}_t \mid \mathbf{x}_0)} \\
&= \exp\left(-\frac{1}{2}\left(\frac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{1- \alpha_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\alpha_{t-1}}\mathbf{x}_0)^2}{1 - \bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1 - \bar{\alpha}_t}\right)\right) \\
&= \exp\left(-\frac{1}{2}\left(\left(\frac{\alpha_t}{1- \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}\right)\mathbf{x}_{t-1}^2 - \left(\frac{2\sqrt{\alpha_t}}{1- \alpha_t}\mathbf{x}_t + \frac{2\sqrt{\alpha_{t-1}}}{1 - \bar{\alpha}_{t-1}}\mathbf{x}_0\right)\mathbf{x}_{t-1} + C(\mathbf{x}_t, \mathbf{x}_0)\right)\right)
\end{aligned}
$$

This seems similar to Gaussian distribution formula. Therefore, we ignore the last term $$C(\mathbf{x}_t, \mathbf{x}_0)$$ since $$\mathbf{x}_t, \mathbf{x}_0$$ are not relevent to $$\mathbf{x}_{t-1}$$. 

The variance and mean of such probability function is this

$$

\tilde{\sigma}_t^2 = 1/\left(\frac{\alpha_t}{1- \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}\right) = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \alpha_t
\\
\tilde{\mu}_t = \left(\frac{\sqrt{\alpha_t}}{1-\alpha_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_t}}{1 - \bar{\alpha}_t}\mathbf{x}_0\right)/\left(\frac{\alpha_t}{1-\alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}\right) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1 - \bar{\alpha}_t}\mathbf{x}_0
$$

Remember that $$ \mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\left(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\epsilon_t\right)$$, so we can rewrite the mean to be $$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_t\right)$$

Therefore, $$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)=\mathcal{N}(\, \tilde{\mu}_t \, , \tilde{\sigma}_t^2)$$

However, the distribution $$p(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$$ is not what we want since this implies that we already gain access to generated image $$\mathbf{x}_0$$. The ideal distribution that we wish the model to learn is $$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$

This is the backward process and we want the model to learn such distribution.

### ELBO

If you remember, for an image generation task, we want to maximize the likelihood:

$$  

\mathbb{E}_{x \sim p_{\text{data}}(x)} \!\bigl[\log p_\theta(x)\bigr]
$$

$$
\begin{align*}
\log p_{\theta}(\boldsymbol{x}) &= \log \int p_{\theta}(\boldsymbol{x}_{0:T}) d\boldsymbol{x}_{1:T} \\
&= \log \int \frac{p_{\theta}(\boldsymbol{x}_{0:T})p_{\theta}(\boldsymbol{x}_{1:T} \mid \boldsymbol{x}_0)}{p_{\theta}(\boldsymbol{x}_{1:T} \mid \boldsymbol{x}_0)} d\boldsymbol{x}_{1:T} \\
&= \log \mathbb{E}_{p_{\theta}(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \frac{p_{\theta}(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T} \mid \boldsymbol{x}_0)} \right] \\
&\geq \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p(\boldsymbol{x}_{0:T})}{q(\boldsymbol{x}_{1:T} \mid \boldsymbol{x}_0)} \right] \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_T) \prod_{t=1}^{T} p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)}{\prod_{t=1}^{T} q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})} \right] \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_T)p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \prod_{t=2}^{T} p_\theta(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t)}{q(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \prod_{t=1}^{T-1} q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})} \right] \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_T)p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \prod_{t=1}^{T-1} p_\theta(\boldsymbol{x}_t \mid \boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \prod_{t=1}^{T-1} q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})} \right] \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_T)p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1)}{q(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1})} \right] + \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \prod_{t=1}^{T-1} \frac{p_\theta(\boldsymbol{x}_t \mid \boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})} \right] \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \right] + \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1})} \right] + \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \sum_{t=1}^{T-1} \log \frac{p_\theta(\boldsymbol{x}_t \mid \boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})} \right] \\
&= \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \right] + \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1})} \right] + \sum_{t=1}^{T-1} \mathbb{E}_{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_t \mid \boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})} \right] \\
&= \mathbb{E}_{q(\boldsymbol{x}_1|\boldsymbol{x}_0)} \left[ \log p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \right] + \mathbb{E}_{q(\boldsymbol{x}_{T-1},\boldsymbol{x}_T|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_T)}{q(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1})} \right] + \sum_{t=1}^{T-1} \mathbb{E}_{q(\boldsymbol{x}_{t-1},\boldsymbol{x}_t,\boldsymbol{x}_{t+1}|\boldsymbol{x}_0)} \left[ \log \frac{p_\theta(\boldsymbol{x}_t \mid \boldsymbol{x}_{t+1})}{q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1})} \right] \\
&= \underbrace{\mathbb{E}_{q(\boldsymbol{x}_1|\boldsymbol{x}_0)} \left[ \log p_\theta(\boldsymbol{x}_0 \mid \boldsymbol{x}_1) \right]}_{\text{reconstruction term}} - \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{T-1}|\boldsymbol{x}_0)} \left[ \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_T \mid \boldsymbol{x}_{T-1}) \,||\, p_\theta(\boldsymbol{x}_T)) \right]}_{\text{prior matching term}} \\
&\quad - \sum_{t=1}^{T-1} \underbrace{\mathbb{E}_{q(\boldsymbol{x}_{t-1},\boldsymbol{x}_{t+1}|\boldsymbol{x}_0)} \left[ \mathcal{D}_{\text{KL}}(q(\boldsymbol{x}_t \mid \boldsymbol{x}_{t-1}) \,||\, p_\theta(\boldsymbol{x}_t \mid \boldsymbol{x}_{t+1})) \right]}_{\text{consistency term}}
\end{align*}
$$