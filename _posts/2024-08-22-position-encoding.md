---
layout: post
title: History of Position Encoding
date: 2024-08-21 16:15:09
description: 
tags: formatting images
categories: encoding method
tabs: true
---

### Background-Positional information

Compared with other sequential model like LSTM and RNN, Transformers are much more successful and serve as the foundation of current language models. These transformer-based LLM achieve higher accuracy than other architecture and can scale up easily due to the parallel computing of Transformer. However, Transformers have its flaws. It requires more computation resources than Sequential model with time complexity of $$O (N^2)$$. Also, without external help, Transformers can't acquire positional information that is important when dealing with sequential data.

What's positional information? Positional information is about the order of characters appear. For instance, "I eat fish" is different from "Fish eat I" because order of appearance is different. "I" appears first in the first setence and "fish" appears first in the second sentence, resulting each sentence having different meaning. 

Transformers can't acquire the sequential information of words because the way data is passed to the model. For a traditional model like RNN, each data is passed one at a time. For instance, in the sentence "I eat fish," "I" will be processed first, then "eat", then "fish." This way, RNN acquires sequential information of each word.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/recurrent_network.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

However, for transformers, all words are passed in all at once. The sequential meaning is lost when the model processes data parallelly instead of sequentially. All words in "I eat fish" will be passed into the model at the same time. The model can't distinquish which words come first, unless we pass in additional positional information about this word.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/Transformer.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Of course, we can just simply assign 1, 2, 3, etc to each word position, but researchers found out better ways to encode position of words.

### Absolute Positional Encoding - Sinusoidal Positional encoding

Absolute positioinal encoding is first introduced in the paper Attention Is All You Need. It uses trigonometry function, sine and cosine, to encode positions. 

For a word $$p$$, the positional encoding will be expressed as:

$$
p_t = \begin{bmatrix} sin(w_0 \cdot t) \\ cos(w_0 \cdot t) \\ sin(w_1 \cdot t) \\ cos(w_1 \cdot t) \\ \vdots \\ sin(w_{d/2} \cdot t) \\ cos(w_{d/2} \cdot t) \end{bmatrix} \  \ w_k = \frac{1}{10000^{\frac{2k}{d}}} \   \ where d is the dimension of the positional embedding
$$

Using trigonometry function has a very good property. Given word $$p_i$$ at position i and word $$p_{i+k}$$ at position i+k, we can deduce the position between them by taking dot product.

$$\begin{split} p_i \cdot p_{i+k} & = \[\sum{i=0}^{\frac{d}{2}-1}sin(w_it) \dot sin(w_i(t+k))+cos(w_it) \dot cos(w_i(t+k))] \\
& = \[\sum{i=0}^{\frac{d}{2}-1} cos(w_i(t-(t+k)))] \\
& = \[\sum{i=0}^{\frac{d}{2}-1} cos(w_ik)]
$$

We can see that the final result is only dependent on k, the relative distance between each word. However, it can only work when two positional embedding directly dot product with each other. In Transformer, there are always weight matrices K and V between the vectors, resulting the loss of relative position information.

### Relative Positional Encoding - Rotary Positional encoding

If we want positional embeddings maintaining relative position information in Transformers, we are essentially try to find functions $$f_q$$, $$f_k$$, $$g$$ such that:

$$\langle f_q(x_m,m),f_k(x_n,n)\rangle=g(x_m,x_n,m-n)$$

Suppose the positional encoding dimension is 2. In Rotary Positional encoding, researchers found out a set of functions that satisfied above equation.

$$f_q(x_m,m) = (W_qx_m)e^{im\theta}$$
$$f_k(x_n,n) = (W_kx_n)e^{in\theta}$$
$$g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)e^{i(m-n) \theta}]$$

In linear algebra, a complect number can be expressed as the following:


For $$e^{im\theta}$$, w



Why do we add positional embedding to input embedding? What about concatenation?
1. we may blur semantic meaning of input embedding by interwining positional embedding with semantic content
2. It'll increase dimension of input vector, . Also, there's no report that indicates the advantage of using concatenation

