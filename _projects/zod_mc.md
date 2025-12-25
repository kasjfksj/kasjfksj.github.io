---
layout: page
title: Partial zod mc
description: 
img: assets/img/12.jpg
importance: 1
category: work
related_publications: false
---

Current Status
Sampling from complex probability distributions, particularly those defined by potential functions, remains a significant challenge in computational statistics and machine learning. Traditional methods like ZOD-MC often scale exponentially with dimensionality, limiting their efficiency for high-dimensional problems without exploitable structure.
Motivation
As an academic project, I developed the Partial ZOD-MC method to address inefficiencies in sampling from decomposable potential functions. The goal was to exploit block-independent structures to reduce computational complexity, drawing from background techniques like iterative refinement and importance sampling, while providing theoretical justification and empirical validation.
Achievements
I proposed the Partial ZOD-MC, a coordinate-wise variant that samples subsets of dimensions iteratively, reducing oracle complexity from O(exp(b)) to O(exp(d) · b/d) for decomposable potentials where d << b.
Empirical evaluations on independent distributions showed speedups, such as 64.34% in 5 dimensions, with comparable sample quality measured by MMD and W2 distances.
For higher dimensions (up to 30), P-ZOD-MC maintained superior time performance while keeping MMD low, though W2 increased.
On joint distributions like GMMs, time reductions were observed, with improving MMD but stable W2 in higher dimensions.
Challenges
Background methods like iterative refinement require storing extensive proposal information, leading to memory issues, while importance sampling demands exponentially many points and can be biased for finite samples. For P-ZOD-MC, the primary challenge is its failure on joint distributions where dependencies violate the block-independence assumption, resulting in poor capture of the target distribution despite time savings.
Reflections and Gains
This project highlighted the value of structural exploitation in sampling methods, demonstrating that decomposition can yield major efficiency gains for independent cases but requires careful validation of assumptions. I gained deeper understanding of trade-offs between complexity and accuracy, as well as experience in theoretical analysis and empirical benchmarking, reinforcing the need for adaptive techniques in handling dependent variables for future extensions.