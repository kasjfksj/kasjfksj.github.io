---
layout: post
title: Grokking - a possible way to achieve AGI
date: 2024-08-30 16:15:09
description: 
tags: formatting images
categories: encoding method
tabs: true
related_posts: false
toc: 
  sidebar: left
---

As mentioned in previous blog, grokking is a phenomenon that extremely long training time leads to a sharp incease in val accuracy. Some researchers try to find a way to speed up the grokking phenomenon. GrokFast proposed a new optimizer to boost slow-varying component of the gradient as they hypothesized that it was a contributing factor to Grokking phenomenon.

Initially, I believe that GrokFast could be significant and was puzzled when there was only 1 citation currently. However, further testing on GrokFast shows that it was not capable enough to speed up generalization on test dataset. Default setup, using 3 layer and 200 hidden layer size can achieve good result on fast grokking, but other setup will result in much slower learning procedure, resulting much later increase in train accuracy and val accuracy. The test play results are presented below

# GrokFast - testing 

I first test the default setup mentioned above. The result is quiet amazing, as we can see the much earlier increase in val acc.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/grok_fast/grok_fast_none.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    No GrokFast
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/grok_fast/grokfast_em.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    With GrokFast
</div>

Later, I changed some parameters in GrokFast to test its general capability over all kinds of network. However, the result was not satsifying.

# GrokFast - effect of network depth

First, I changed the number of layers to 4 and 5 and tested the performance. While train acc and val acc increased simultaneously, the number of steps required to significantly increase train acc and val acc rised from $$10^3$$ to around $$10^4$$. This phenomenon was quite puzzling, and I speculated that it was due to the magnifying low-varying gradient component.

However, there's still some interesting results of the experiment.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/grok_fast/grokfast_4layer.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    4layer Grokfast experienced sharper increase than 3layer Grokfast
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/grok_fast/grokfast_5layer.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    5layer Grokfast experienced sharper increase than 4layer Grokfast at $$10^4$$ steps
</div>

We can see that the network experienced sharper increase at $$10^4$$ steps. 

# GrokFast - effect of network width

Next, I want to test the effect of network width on GrokFast performance, with the setup of 3 layers of network.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="aassets/img/grok_fast/grokfast_128p.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    at around $$10^4$$ steps, the test acc experienced a decrease and then slowly increase at around 5*10^4 steps
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/grok_fast/grokfast_512p.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    at around $$10^4$$ steps, the test acc experienced a slight decrease and then rapidly increase.
</div>

Based on the observation above, I conclude that network width can help network regain its val accuracy after the mysterious drop in val acc. 

# Combination with LoRA

Due to the phenomenon, I planed to use Grokfast on LoRA. Specifically, using the gradient update code on matrix A and matrix B can work maybe. However, the first test result was confusing.

The first setup is traditional 3 layer + 200 hidden size. GrokFast implemented MA optimizer, and here's the result

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/grok_fast/grokfast+LoRA.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
While val acc and train acc climbed up together at around 10^4, val acc suddenly dropped and continued to decrease until the end of training time. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="aassets/img/grok_fast/grokfast_ema.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

For ema optimizer, the val acc sharply increase at $$10^4$$ steps and then increases at a very slow pace. 


