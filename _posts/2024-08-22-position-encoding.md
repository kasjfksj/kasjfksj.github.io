---
layout: post
title: History of Position Encoding
date: 2024-08-21 16:15:09
description: 
tags: formatting images
categories: encoding method
tabs: true
---

### Background

Compared with other sequential model like LSTM and RNN, Transformers are much more successful and serve as the foundation of current language models. These transformer-based LLM achieve higher accuracy than other architecture and can scale up easily due to the parallel computing of Transformer. However, Transformers have its flaws. It requires more computation resources than Sequential model with time complexity of $$O (N^2)$$. Also, without external help, Transformers can't acquire positional information that is important when dealing with sequential data.

What's positional information? Positional information is about the order of characters appear. For instance, "I eat fish" is different from "Fish eat I" because order of appearance is different. "I" appears first in the first setence and "fish" appears first in the second sentence, resulting each sentence having different meaning. 

When we train on sequential data, like sentiment analysis on sentences, on traditional model, each word is passed one at a time, giving the word sequential information. However, in Transformers, data are passed parallelly. sentences are dumped into the model together. It can't gain positional information from purely processing the data. Of course, we can give them positional information by assigning 1, 2, 3 and so on to each words, but researchers have found a clever idea to encode position.

### Transformers

We first talk about mechanism of transformers which is attention.



### Absolute Positional Encoding - Sinusoidal Positional encoding

Absolute positioinal encoding is first introduced in the paper Attention Is All You Need. It uses trigonometry function, sine and cosine to encode positions. The positional encoding will then be added to input vector and feed it to attention.

$$PE_{pos,2i} = sin(pos/10000^{2i/d_model})$$
$$PE_{pos,2i+1} = cos(pos/10000^{2i/d_model})$$

$$pos$$ is the position of the word and $$i$$ is the dimension of positional encoding. For a word $$p$$, the positional encoding will be expressed as:

$$
p_t = \begin{bmatrix} sin(w_1 \cdot t) \\ cos(w_1 \cdot t) \\ sin(w_2 \cdot t) \\ cos(w_2 \cdot t) \\ \vdots \\ sin(w_{d/2} \cdot t) \\ cos(w_{d/2} \cdot t) \end{bmatrix} \
$$

Using trigonometry function has a very good property. 

However, there are several issues using this encoding method. Firstly, while it captures the distance between each word, it can't distinguish which words come out. 

### Relative Positional Encoding - Rotary Positional encoding

The reason why previous one is called absolute encoding is that the formula only takes the position of current word into account, neglecting the position of other words. 

Why do we add positional embedding to input embedding? What about concatenation?
1. we may blur semantic meaning of input embedding by interwining positional embedding with semantic content
2. It'll increase dimension of input vector, . Also, there's no report that indicates the advantage of using concatenation

