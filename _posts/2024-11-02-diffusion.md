---
layout: post
title: Diffusion model
date: 2024-11-08 16:15:09
description: 
tags: formatting images
categories: model architecture
tabs: true
related_posts: false
toc: 
  sidebar: left
---

## Introduction

In recent years, diffusion models have emerged as one of the most powerful techniques in the field of generative AI, yielding remarkable results in image, video, and audio synthesis. These models have gained significant attention for their ability to generate high-quality content, often outperforming traditional models like GANs (Generative Adversarial Networks) and VAEs (Variational Autoencoders) in various creative tasks. But what exactly are diffusion models, and why are they causing such a stir in the AI community?

In this blog, we'll dive deep into the fundamentals of diffusion models, explain how they work, and explore their applications in generative tasks. By the end, you'll have a clear understanding of this cutting-edge model and how it is revolutionizing the world of generative AI.

## Limitations of other models

Before diffusion model was proposed, there were already several generative models, but all of them had some limitations

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/diffusion_model/40841731207612_.pic.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Comparison of architectures of different models.
</div>

1. Variational Autoencoders (VAEs)
Variational Autoencoders (VAEs) are a class of generative models that rely on the variational inference framework. The model learns an encoder-decoder structure, where encoder maps the input data to a probabilistic latent space (usually Gaussian). Decoder maps the latent space back to the data space to generate new samples.

VAEs optimize a lower bound on the log-likelihood of the data using the ELBO (Evidence Lower Bound), which includes a reconstruction term and a regularization term that enforces the learned latent distribution to be close to a prior distribution (usually Gaussian).

There are 2 limitations of VAE. Firstly, its generated samples may lack sharpness and realism. Secondly, the regularization in VAEs can sometimes result in blurry outputs due to the continuous latent space.

2. Generative Adversarial Networks (GANs)
Generative Adversarial Networks (GANs) consist of two neural networks — a generator and a discriminator — that compete against each other in a minimax game. The generator creates synthetic data, and the discriminator tries to distinguish between real and fake data. The goal is for the generator to create data that is indistinguishable from real data, according to the discriminator.

There are 2 limitations of GAN. Firstly, GANs has instability during training session due to optimization of minmax game. Secondly ,they are prone to mode collapse, where the model fails to capture the full diversity of the data distribution.

3. Normalizing Flows (NF)
Normalizing Flows (NF) are another class of generative models that provide a way to transform simple distributions (e.g., Gaussian) into complex ones via a series of invertible transformations. The key idea is that the model learns a sequence of invertible functions that map a simple distribution to the target data distribution. 

Normalizing Flows resemble to diffusion models where both of them try to trace from simple distributions into complex data distribution via transformations. However, there are 2 limitations in NF. Firstly, it's computationally expensive due to the need for invertible transformations, which can make them less scalable for high-dimensional data. Secondly, it's not flexible when the transformations need to be invertible. 

## What is diffusion model

Diffusion models are a class of generative models that gradually add noise to data, transform it into pure noise, and then learn to recover the data through the reverse diffusion process. This is akin to simulating the way particles diffuse through a medium, but instead of particles, the model learns to diffuse data (like images, audio, etc.).

## Key insighsts behind diffusion model

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/diffusion_model/40831731207587_.pic.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

•	Forward Process (Noise Addition): This is where the model gradually adds noise to the data, step by step, until the original data is completely turned into noise. This process is typically modeled using a Markov chain.

•	Reverse Process (Denoising): In the reverse process, the model learns how to recover the original data from the noisy version, step by step. This is where the generative magic happens—by reversing the noise addition process, the model gradually recovers the original signal, ultimately producing clean data from random noise.

Diffusion models belong to the broader family of probabilistic models, where the generation of data is formulated as a sequence of probabilistic transitions from a simple distribution (like pure noise) to a complex data distribution (such as real images).

## How Do Diffusion Models Work?
To better understand diffusion models, let's break down the steps involved in both the forward and reverse processes.

1. Forward Process (Adding Noise)
In the forward process, we start with a data point (e.g., an image) and progressively add small Gaussian noise over several time steps. As the process continues, the image becomes increasingly noisy until, at the end of the process, it resembles pure Gaussian noise. For an image $$x_0$$, at each time step t, noise is added according to the formula $$x_{t+1}=\sqrt{1-\beta_t}x_t+\beta_t\epsilon$$. $$\beta_t$$ is called noise schedule, which controls how much noise is added at each step, typically starting with a small amount and increasing as time progresses. $$\epsilon$$ is random Gaussian noise. 

2. Reverse Process (Denoising)
Once the forward process is defined, the goal of the diffusion model is to learn the reverse process. The reverse process involves learning how to remove the noise step by step, recovering the original data distribution from the noisy version.

3. Training the Diffusion Model
Training a diffusion model involves learning the distribution of the reverse diffusion process. During training, the model tries to predict the noise added in time $$T$$, given by the noisy image and time information $$T$$. The noise prediction is often written in L2 loss.

4. Sampling from the Diffusion Model
Once trained, the model can generate new data by starting with a random noise sample and iteratively denoising it through the reverse process until a realistic sample is produced. This process the model predicting the noise that was added to the image at time T, so as t produce the less noisy image at time T-1.

## Applications of Diffusion Models

Diffusion models have been successfully applied across a wide range of domains. Here are a few notable applications:

1. Image Generation
The most prominent use case for diffusion models has been in image generation. Models like DALL·E 2 and Stable Diffusion leverage diffusion techniques to create high-quality, photorealistic images from text descriptions. These models are trained on large datasets of images and learn to generate realistic images by reversing the diffusion process.

2. Super-Resolution and Image Inpainting
Diffusion models have also been applied to tasks like image super-resolution (increasing the resolution of low-quality images) and inpainting (filling in missing parts of an image). By learning the reverse diffusion process on images with added noise, these models can recover fine details and generate high-resolution content from low-quality inputs.

3. Audio and Speech Synthesis
Diffusion models are also being used to generate audio and speech. By treating sound waves as data and adding noise over time, diffusion models can generate natural-sounding speech or music by reversing the noise process, step by step.

4. Video Generation
Generating video frames is a more complex task, but diffusion models are also making strides in this area. Video generation with diffusion models typically involves generating high-quality sequences of frames that are consistent and coherent over time.

## Why Are Diffusion Models So Effective?

There are several reasons why diffusion models have become so popular in generative AI:
1.	Stability: Unlike GANs, which can suffer from training instability (e.g., mode collapse), diffusion models tend to be more stable during training, as they optimize a simpler objective—predicting noise at each step rather than directly generating data.
2.	High-Quality Output: Diffusion models have been shown to produce exceptionally high-quality and diverse outputs. The gradual denoising process allows the model to generate highly detailed and coherent data.
3.	Flexibility: Diffusion models can be applied to a wide range of generative tasks, from images to audio to text, making them a versatile tool in the AI toolbox.
4.	Scalability: Diffusion models are scalable and can be trained on large datasets, enabling them to learn complex data distributions and generate high-dimensional outputs.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/diffusion_model/40851731207864_.pic.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Experiment results in the paper Denoising Diffusion Probabilistic Models. DDPM outperforms most of the models
</div>

## Challenges and Future Directions
While diffusion models have demonstrated significant promise, they are not without challenges:

•	Computational Expense: Diffusion models typically require a large number of time steps to generate data, which makes the sampling process computationally expensive. However, researchers are actively working on techniques to reduce this computational cost while maintaining output quality.
•	Sampling Speed: Due to the iterative nature of the reverse diffusion process, generating data with diffusion models can be slower than other generative models like GANs. There is ongoing research into speeding up the sampling process.
•	Data Efficiency: Diffusion models tend to require a large amount of training data to perform well, which can be a limitation in some cases.

Despite these challenges, diffusion models are rapidly advancing and are likely to remain at the forefront of generative AI research for the foreseeable future.

## Conclusion
Diffusion models represent a fascinating and powerful approach to generative AI, offering high-quality and stable generation of diverse data types. Their ability to model complex distributions through a gradual process of noise addition and removal has revolutionized fields like image synthesis, super-resolution, and audio generation. While there are still challenges to overcome, diffusion models are undoubtedly a key part of the future of generative AI, and we can expect to see even more exciting applications in the coming years.

As the field continues to evolve, diffusion models will likely play a crucial role in shaping the next generation of AI-powered creativity and innovation.


