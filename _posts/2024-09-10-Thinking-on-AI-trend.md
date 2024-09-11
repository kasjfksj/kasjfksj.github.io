---
layout: post
title: Thinking on current LLM trend
date: 2024-08-09 16:15:09
description: 
tags: experience
categories: classes
tabs: true
---

Recently, the Reflection 70B model created a lot of popularity but turned out to be fake. When it was first released, its developer Matt claimed it was the best open-source model, comparing it to Claude 3.5, Claude 3, and GPT-4o. Many were excited about its potential to set new records, but soon doubts arose about its legitimacy. Hugh Zhang pointed out on Twitter that the model scored 99.2% on the GSM8K benchmark, even though over 1% of the GSM8K data is mislabeled. This raised concerns that the developers might have trained the model using the test data to get a misleadingly high score.

Further doubts emerged when the model didn’t perform outstanding compared to others on tricky questions, like the number of 'r's in "strawberries" or comparison of 9.11 and 9.9. Some speculated that Reflection 70B was just LLama 3 with LoRA fine-tuning. When the API for Reflection 70B was released, it seemed to perform similarly to Sonnet-3.5, except with the word "Sonnet" filtered out. While the cheating methods were quite clear, the model's high evaluation score still raised questions about how we measure model performance. This case shows how easy it is to manipulate benchmarks to look better than they really are.

If we take a step back and look at AI development as a whole, we can see that it often relies on a simple two-part system: datasets and evaluation pipelines. Models are tested on datasets, and their performance is measured through pipelines. While this makes comparing models easier, it also has major flaws. The static nature of these evaluations allows for manipulation, as seen with Reflection 70B. Plus, datasets can quickly become outdated and unsuitable for training advanced models. For example, the Visual Question Answering (VQA) dataset from 2017 has been surpassed by more complex datasets that include multiple conversations along with images. Therefore, we need a new model evaluation standard that is dynamic and can automatically update itself.

I tried to search some benchmarks that are dynamic instead of static. Thus, it leads me to livebench, which is a benchmark with new questions monthly. The questions posted by live bench are various, covering reasoning, coding, math, etc. Moreover, it published test data monthly, preventing other models knowing the question in prior. 