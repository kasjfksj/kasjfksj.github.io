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

Compared with other sequential model like LSTM and RNN, Transformers are much more successful and serve as the foundation of current language models. These transformer-based LLM achieve higher accuracy than other architecture and can scale up easily due to the parallel computing of Transformer. However, Transformers have its flaws. It requires more computation resources than Sequential model with time complexity of $$O (N^2)$$. 

Positional information is about the order of characters appear. For instance, "I eat fish" is different from "fish eat I" because order of appearance is different. "I" appear first in the first setence and "fish" appear first in the second sentence, resulting each sentence having different meaning. 

In a sequential model, 

### Positional Encoding


