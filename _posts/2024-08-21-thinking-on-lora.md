---
layout: post
title: LoRA - a potential parameter efficient fine-tuning method for large models
date: 2024-08-21 16:15:09
description: 
tags: formatting images
categories: fine-tune
tabs: true
---
### Background

Imagine you are a researcher testing out language models such as Small T5 or LSTM and you have decent but not great amouont of GPUs. You want to finetune the model to test their performances on a few datasets. You train your model on the datasets and get the results. You are satisfied.

But what if there are hundreds of datasets you want to test and not only that, you want to test out current LLM like LLama and Mistral that have billions of parameters. It's impossible to train on your limited GPU since there are so many parameters that need to be updated through back propogation. You need a way to minimize the number of parameter updates without costing too much computation resources. This is where LoRA comes to play.

### Gradient Descent

Every AI model, LLM or CNN, is basically a mathematical function that's based on some parameters $$ \theta $$. Parameters are usually weight matrix that multiply with input matrix. For most datasets, we have inputs $$X$$ and targets $$Y$$. For instance, the input can be a sentence and the output is 0 and 1 where 0 represents negative sentiment and 1 represents positive sentiment. We give the model input $$x \  \ and \  \ x \in X$$, and it outputs $$ \textit{f}(x, \theta)$$. However, we wish that the output will be the target $$y \  \ and \  \ y \in Y$$. We use gradient descent to minimize the distance between outcomes and desired outcomes. 

$$ W_t = W_{t-1} + \Delta W \  \ where\ \ W\ \ is \ \ parameter \ \ and \ \ t \ \ is \ \ each \ \ iteration $$

For the training session, most computation lies in computing gradient, notably matrix multiplication, which will take incredibly long time when the model size scales up. We can handle this amount of computation on a few datasets, but it'll be too much when there are many datasets to fine tune on.


### [LoRA](https://arxiv.org/abs/2106.09685)

 LoRA solves this problems by assuming that the 'change in weights during model adaptation also has a low “intrinsic rank”', meaning that the updated parameter can be expressed as the multiplication of 2 low rank matrix. "Rank" is a term in Linear Algebra, which is equivalent to new information. Higher the rank the matrix has, more information the matrix contains. 

 The update formula in LoRA is written as follow:

$$
W = W_0 + AB \  \ where \ \ A\in R^{m\times r} \ \ and \ \ B\in R^{r\times n}. \  \ Here \ \ r\ll m \ \ and \ \ n
$$

This way, the computation for gradient will be greatly reduced. For instance, suppose the weight matrix $$W$$ is 768 by 1024 matrix. When we calculate the gradient for this matrix, we need to compute 768 $$\times$$ 1024 parameters. For $$A$$ and $$B$$ matrix, we can let A be a 768 by 32 matrix and B be a 32 by 1024 matrix. When the model is doing gradient descent, we froze the model's parameter, so the gradient for the $$W$$ is 0. We only need to calculate gradient for $$A$$ and $$B$$, which has much less parameters than $$W$$. Thus, the computation cost reduces significantly.

### ReLoRA

Currently, LoRA and its variants, QLoRA, MLoRA, etc, focus on fine-tuning models that are fully trained on datasets. Although during the fine-tuning stage, they require less computations, when we take previously fully trained model into account, it still takes quite a lot of computations. Can we apply LoRA as a training method to a model instead of fully training?

When we look at the equation in LoRA closely, we can move $$W_0$$ to the left and obtain:

$$
\Delta W = AB \  \ where \  \ \Delta W = W - W_0
$$

In a way, we can treat the $$AB$$ as the gradient of the model, and yes, we can use LoRA to train a model. This idea is explored by ReLoRA. 

In their paper, they use LoRA as gradient to update model parameter. First, they train the model just like LoRA without updating model's parameters. After 2000 steps, the matrix A and B will multiply and add to the model's weight. Then both matrix will be reinitialize and get trained again. 

There are several advantage of this method. Firstly, this is much more parameter-efficient than fully-trained model. Secondly, according to the paper, ReLoRA outperforms LoRA though still can't compare with fully trained model. Thus, the idea of using low-rank update for high rank matrix does work.
<img src = "./img/35001724304205_.pic.jpg"></img>

### Conclusion

LoRA is one of the most popular PEFT method for its simplicity and effectiveness. From my intuition, the way LoRA update parameters is similar to how human learns new things. We often uses previous knowledge and updates the knowledge with some adjustment. For instance, we draw inspiration from real numbers and apply the same arithmetic operations on imaginary number with additional rule that $$i \times i = -1$$. I believe LoRA is possibly the key component towards continual learning, making the model more knowledgable about the world and therefore, become the world model.




