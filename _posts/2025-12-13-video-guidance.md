---
layout: post
title: Diffusion Series - Video Diffusion for DMLab Maze Navigation
date: 2026-01-16 17:45:00
description: Comparing reconstruction-guided sampling and direct noised-frame injection for improving temporal consistency in long video generation
tags: diffusion
categories: AI
tabs: true
related_posts: false
giscus_comments: true
toc:
  sidebar: left
---

## Intro

This post examines the application of video diffusion models to generate 300-frame first-person maze navigation sequences from the DeepMind Lab (DMLab) environment at 64×64×3 resolution. The primary challenge addressed is maintaining temporal consistency across extended sequences.

## Motivation and Initial Approach

The initial objective was to implement a functional video diffusion model capable of generating coherent long-horizon navigation videos. A 3D U-Net architecture with temporal attention layers was trained on DMLab episodes.

However, as I tested reconstruction-guided sampling mechanisms, the generated videos frequently exhibited temporal inconsistencies such as abrupt layout shifts and discontinuous motion.

The reconstruction-guided sampling, as described in the Video Diffusion Models, applies a guidance term during the sampling process, utilizing gradients of the Mean Squared Error computed over overlapping frame regions to enforce reconstruction consistency.

Although MSE is a convex objective and the guidance formulation is theoretically sound, the resulting videos still showed progressive degradation in temporal fidelity over longer sequences. Multiple hyperparameter adjustments and diagnostic visualizations of intermediate noise estimates confirmed that information from prior frames was not being adequately propagated through the denoising trajectory.

## Proposed Alternative: Direct Noised-Frame Injection

The observed limitations suggested that early denoising steps — where both the noise prediction and partially denoised estimate contain minimal semantic content — may not provide sufficient context for long-horizon dependency modeling.

To address this, an alternative conditioning mechanism was implemented: direct injection of noised versions of overlapping previous frames into the input of the current denoising step. This approach explicitly supplies contextual information from prior timesteps, enabling the model to better leverage short-term dependencies when predicting subsequent frames.

This method was compared against standard reconstruction-guided sampling using the same trained checkpoint and sampling procedure.

## Experimental Setup

- **Dataset**: DeepMind Lab maze navigation episodes
- **Resolution & Length**: 64×64×3 per frame, 300 frames
- **Model**: 3D U-Net with temporal attention, trained from scratch in PyTorch
- **Sampling**: DDPM with 1000 steps
- **Evaluation Metrics**:
  - FID (lower is better): Distribution similarity
  - Temporal LPIPS (lower is better): Frame-to-frame perceptual smoothness
  - Flow Magnitude: Average optical flow strength
  - Flow Consistency: Standard deviation of flow magnitudes (lower is better)

## Quantitative Results

| Metric                              | Noised-Frame Injection        | Reconstruction Guidance       | Observation                          |
|-------------------------------------|-------------------------------|-------------------------------|--------------------------------------|
| FID                                 | 259.22 ± 8.08                 | **250.48 ± 19.62**            | Guidance slightly better, higher variance |
| Temporal LPIPS                      | **0.1450 ± 0.0115**           | 0.1790 ± 0.0229               | Injection substantially smoother     |
| Flow Magnitude                      | 4.35 ± 0.12                   | 4.29 ± 0.49                   | Comparable motion intensity          |
| Flow Consistency (std, ↓ better)    | 2.49                          | **2.31**                      | Guidance more uniform motion         |

## Analysis

Qualitative inspection revealed that noised-frame injection produced noticeably more fluid navigation sequences with smoother turns and fewer structural artifacts. Reconstruction-guided samples, while marginally sharper overall (reflected in FID), exhibited more frequent small discontinuities, consistent with the elevated Temporal LPIPS scores.

These findings support the hypothesis that direct injection of contextual information during early denoising steps can substantially improve temporal coherence, offering a practical alternative to purely gradient-based guidance for long-sequence generation.

Neither approach fully resolves the challenges of 300-frame generation at this resolution (high FID values indicate remaining distribution gaps), suggesting that future improvements may benefit from latent-space diffusion, enhanced architectures, or hybrid conditioning strategies.

Further experimentation combining elements of both methods is planned.

# References

- Ho, J., et al. (2022). Video Diffusion Models. arXiv:2204.03458
- DeepMind Lab: https://github.com/google-deepmind/lab