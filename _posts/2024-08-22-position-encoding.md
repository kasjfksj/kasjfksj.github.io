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
p_t = \begin{bmatrix} sin(w_0 \cdot t) \\ cos(w_0 \cdot t) \\ sin(w_1 \cdot t) \\ cos(w_1 \cdot t) \\ \vdots \\ sin(w_{d/2} \cdot t) \\ cos(w_{d/2} \cdot t) \end{bmatrix} \  \ w_k = \frac{1}{10000^{\frac{2k}{d}}} \   \ where \ \ d \ \ is \ \ the \ \ dimension \ \ of \ \ the \ \ positional \ \ embedding
$$

Using trigonometry function has a very good property. Given word $$p_i$$ at position i and word $$p_{i+k}$$ at position i+k, we can deduce the position between them by taking dot product.

$$\begin{split} p_i \cdot p_{i+k} & = \[\sum{i=0}^{\frac{d}{2}-1}sin(w_it) \dot sin(w_i(t+k))+cos(w_it) \dot cos(w_i(t+k))] \\
& = \[\sum{i=0}^{\frac{d}{2}-1} cos(w_i(t-(t+k)))] \\
& = \[\sum{i=0}^{\frac{d}{2}-1} cos(w_ik)]
\end{split}
$$

We can see that the final result is only dependent on k, the relative distance between each word. 

Let's recall how Transformer works, with KQV stuffs.

$$softmax(\textbf{QK^T})\textbf{V}$$
$$\textbf{Q} = \textbf{W_Q}x$$
$$\textbf{K} = \textbf{W_K}x$$
$$\textbf{V} = \textbf{W_V}x$$

We may only focus on $$\textbf{QK^T}$$ because that's where two words interact each other. 
We now inject sinusoidal positional encoding to input vector, so instead of $$x$$, it's $$x+e$$. We then calculate attention score for words in position i and j, the result will be:

$$\textbf{W_Q}(x_i+e_i)(\textbf{W_K}(x_j+e_j))^T = $$
$$\textbf{W_Q}x_i{x_j}^T\textbf{W_K}+\textbf{W_Q}e_i{x_j}^T\textbf{W_K}+\textbf{W_Q}x_i{e_j}^T\textbf{W_K}+\textbf{W_Q}e_i{e_j}^T\textbf{W_K}$$

We can see that the first term doesn't contain any positional information of two words. The second and third term only contain positional information of 1 word, which alone can't deduce the relative positional information. Only the fourth one that has the dot product of two positional embedding contain relative position of two words, which is good for the model.

However, although it can learn about relative positions between words, it will fail to the order of appearance. If we can look closely at the dot product of two positional embedding:

$$p_i \cdot p_{i+k} =\[\sum{i=0}^{\frac{d}{2}-1} cos(w_ik)]$$

The final result is a summation cosine. We recall that cosine is an even function, meaning that:

$$\[\sum{i=0}^{\frac{d}{2}-1} cos(w_ik)] = \[\sum{i=0}^{\frac{d}{2}-1} cos(-w_ik)]$$

In the sentences "I eat fish" and "fish eat I", "fish" and "I" have the same distance of two. Thus, when calculating attention score between these words, the score will be the same in these sentences. This equivalence is not we want. We want the encoding method can not only distinquish distance between words but also which word comes out first.

### Relative Positional Encoding - Rotary Positional encoding

In order for the encoding method to contain relative position information as well as the order of each word, we can model it using the following equations:

$$\langle f_q(x_m,m),f_k(x_n,n)\rangle=g(x_m,x_n,m-n) \  \ where \ \ f \ \ is \ \ encoding \ \ method$$

Of course, g should not be an even function like sinusoidal encoding. In 

$$f_q(x_m,m) = (W_qx_m)e^{im\theta}$$

$$f_k(x_n,n) = (W_kx_n)e^{in\theta}$$

$$g(x_m,x_n,m-n)=Re[(W_qx_m)(W_kx_n)e^{i(m-n) \theta}]$$

In linear algebra, a complect number can be expressed as the following:

For $$e^{im\theta}$$, w



Why do we add positional embedding to input embedding? What about concatenation?
1. we may blur semantic meaning of input embedding by interwining positional embedding with semantic content
2. It'll increase dimension of input vector, . Also, there's no report that indicates the advantage of using concatenation

