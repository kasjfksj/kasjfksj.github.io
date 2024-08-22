---
layout: post
title: Some ideas on LoRA
date: 2024-08-21 16:15:09
description: 
tags: formatting images
categories: model architecture
tabs: true
---

When I first got to know about LoRA, I wasn't paying attention to it. I treated as another small but effective trick to train models on additional task with few parameter changed. Then it occured to me that LoRA can be treated as the cumulative gradient of the model. Thus, I started to research on LoRA as cumulative gradient of the model over task. 

Before we begin, let's first understand what is LoRA and how LoRA works. 

[LoRA](https://arxiv.org/abs/2106.09685) stands for Low Rank Adaptation, which is a techinque to finetune model on specific datasets with minimal parameter update. Previously, finetuning models on other datasets is painful because it's the same as regular training just on different datasets. Also, it suffers from catastrophic forgetting, meaning that after finetuning, the model will forget knowledge about previous datasets. This is because the gradient of later datasets will deviate from previous datasets, resulting the loss on previous dataset to digress from optimal point. 

<img url = "assets/img/blog/WechatIMG3501.png">Gradient of task 2 is moving away optimal point from Task 1, meaing the loss for Task 1 becomes greater, an example of catastrophic forgetting</img>

Of course, there are methods to reduce the computations, such as choosing only specific parameters to update or introducing external modules to handle updated parameters. However, LoRA takes a simple yet effective approach. They assume that the 'change in weights during model adaptation also has a low “intrinsic rank”', meaning that the updated parameter can be expressed as the multiplication of 2 low rank matrix.

$$
W = W_0 + AB \hspace{1em} where\hspace{0.25em}A\in R^{m\times r}\hspace{0.25em}and\hspace{0.25em}B\in R^{r\times n}.\hspace{1em}Here\hspace{0.25em}r\ll m\hspace{0.25em} and\hspace{0.25em} n
$$

Since we don't need to compute the gradient of a fully ranked matrix, i.e $$W$$. We only calculate gradients of low rank matrix that has fewer parameters, calculating gradient is much more efficient. 

Right now, variations of LoRA focuses on making it more efficienty, such as speeding up using quantization like QLoRA and increasing its accuracy like MLoRA. I am thinking about using LoRA as the only way to update model, without calculating gradient for whole weight matrix.

When we look at the equation closely, we can move $$W_0$$ to the left and obtain:

$$
\Delta W = AB where \Delta W = W - W_0
$$

In a sense, we are updating model just like gradient, with the assumption that the gradient is low intrinsic rank. For a regular training session, we are updating parameter by different batches of data. For LoRA, we are updating parameters by different datasets. There are some connections between fully weighted parameter update and LoRA, and this idea is explored by this [paper](https://arxiv.org/pdf/2307.05695). 

In this paper, they use LoRA as gradient to update model parameter. First, they train the model just like LoRA without updating model's parameters. After 2000 steps, the matrix A and B will multiply and add to the model's weight. Then both matrix will be reinitialize and get trained again. 

There are several advantage of this method. Firstly, this is much more parameter-efficient than fully-trained and LoRA. Both of them require gradient descent on fully ranked matrix. Secondly, according to the paper, ReLoRA outperforms LoRA though still can't compare with fully trained model. Thus, the idea of using low-rank update for high rank matrix does work.
<img url = "assets/img/blog/WechatIMG3500.png"></img>

Personally, I find LoRA and its variants, especially with the latter paper, interesting and potential. I want to create a model that can continuously run and update itself, so the strategy that requires less energy and computation resources can really help me realize my goal.




