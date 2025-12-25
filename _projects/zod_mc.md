---
layout: page
title: Partial zod mc
description: 
img: assets/img/12.jpg
importance: 1
category: work
related_publications: false
---

Now please write a project showcase about diffusion video generation where I want to generate consistent long video. Because currently diffusion model only generates constant number of frames. so I implemented 3 methods to generate. One is classifier free guidance to take previous history frames as condition. One is reconstructed-guided sampling. One is direct feature injection.
 
Right now, the direct feature injection performs better compared to other visually.
 
Actually, the challenge in first place is that training a CFG is not great quality due to the loss of compression of the historical frames (10 frames) with about and it will significantly reduce the quality of generated the following frames. Also the introduction of extra parameter that increase memory usage in training and sampling process. So I decided to discard the CFG and turn to reconstruction-guided sampling, which is an intuitive way of generating longer video (because assuming that diffusion model can generate fixed length of frames perfectly which contain temporal consistency, then forcing the model to generate so that the overlapping frames from previous frames and next frames are the same, which can ensure the consistency) However, even that this is difficult as I see the score function doesn't reduce during the sampling process and unstable, which maybe due to the complexity of distribution of frames so that it can stuck to local minimum (it needs to be explained more) So I decided to use direct feature injection from previous frames so that direct perturb the generation of next frames which do work.
 
Now rewrite this so that it contains the following structure:
    现状
    动机
    成果
    挑战
    反思收获
 
Also it should have images to showcase, either in demonstrating results or explaining the intuition.现状
Current diffusion models for video generation are limited to producing a fixed number of frames, often resulting in short clips that lack the ability to maintain long-term consistency across extended sequences. This constraint stems from the inherent design of these models, which are trained to generate a constant frame length without mechanisms to seamlessly extend or connect sequences.
theaisummer.comHow diffusion models work: the math from scratch | AI Summer
动机
The primary motivation for this project was to overcome the limitations of existing diffusion models by developing techniques to generate consistent, long-form videos. By enabling the model to produce extended sequences while preserving temporal coherence and visual quality, we aimed to expand the applicability of diffusion-based video generation in areas like storytelling, simulation, and content creation.
成果
We implemented three methods to achieve long video generation: classifier-free guidance conditioned on previous history frames, reconstruction-guided sampling, and direct feature injection. Among these, direct feature injection demonstrated superior visual performance, yielding more coherent and high-quality extended videos.
For classifier-free guidance, we conditioned the model on prior frames to guide the generation process.
theaisummer.comAn overview of classifier-free diffusion guidance: impaired model ...
Reconstruction-guided sampling enforced consistency by aligning overlapping frames between sequences.
lilianweng.github.ioDiffusion Models for Video Generation | Lil'Log
Direct feature injection perturbed the generation of subsequent frames using features from previous ones, proving most effective.
link.springer.comCompound-Attention Network with Original Feature injection for ...
Here are examples illustrating consistent long video frames achieved through these approaches.
medium.comVideo Generation with Diffusion Models (Text-to-Video & Motion ...
挑战
The initial challenge arose with classifier-free guidance, where compressing historical frames (around 10) led to quality degradation in subsequent frames, compounded by increased memory usage from additional parameters during training and sampling. We shifted to reconstruction-guided sampling, an intuitive method assuming perfect fixed-length generation with temporal consistency, enforced by matching overlapping frames. However, this proved difficult as the score function failed to decrease steadily during sampling, leading to instability—likely due to the complex frame distribution causing the process to get stuck in local minima, requiring further optimization to stabilize the trajectory.
反思收获
Through this project, we gained insights into the trade-offs of conditioning mechanisms in diffusion models, realizing that while intuitive approaches like reconstruction guidance offer theoretical promise, practical instabilities highlight the need for robust optimization strategies. Direct feature injection's success underscored the value of simpler, more direct interventions in feature space for maintaining consistency without heavy computational overhead. Overall, this reinforced the importance of iterative experimentation and balancing model complexity with performance in video generation tasks.5 sourcesactually, I'm trained on 1080ti GPU, so it's like a personal project for UCI project. However, this project showcase is also used for the professor to look into my background. So the motivation is not that high现状
当前扩散模型在视频生成领域通常受限于固定帧数的短序列输出，无法直接生成更长的视频，同时保持时序一致性。这限制了其在实际应用中的潜力，尤其是在资源有限的环境下进行实验时。
repairit.wondershare.comlilianweng.github.io

动机
作为UCI的个人项目，我使用单张GTX 1080 Ti GPU（11GB显存）训练模型，旨在探索扩散模型生成长序列一致视频的可行方法。通过实验不同扩展技术，了解如何在低资源条件下实现视频序列的时序延伸，并比较各种条件的优缺点。这有助于深化对扩散模型在视频任务中行为的理解，同时积累实际实现经验。
timdettmers.comA Full Hardware Guide to Deep Learning — Tim Dettmers
成果
我实现了三种方法来生成长视频序列：

基于历史帧条件的classifier-free guidance（CFG）。

researchgate.netTrade‐off of Diversity/Fidelity over Guidance Scale. Classifier ...

Reconstruction-guided sampling，通过强制重叠帧一致性来确保时序连贯。

bytez.comVIRES: Video Instance Repainting via Sketch and Text Guided ...

Direct feature injection，直接将前序帧特征注入后续生成过程。

其中，direct feature injection在视觉质量和一致性上表现最佳，能够生成更稳定的长序列视频。
mdpi.comUnleashing the Potential of Pre-Trained Diffusion Models for ...
以下为方法生成的示例帧序列对比，展示长视频的一致性效果：
lilianweng.github.iomarktechpost.comblog.metaphysic.ai


挑战
CFG方法在压缩历史帧（约10帧）作为条件时导致信息丢失，显著降低后续帧质量，同时引入额外参数增加了训练和采样时的内存开销。转而尝试reconstruction-guided sampling，虽然直观（假设模型完美生成固定长度序列后，通过重叠帧匹配强制一致性），但采样过程中score function下降不稳定，常陷入局部最优，可能源于视频帧分布的复杂性，导致优化轨迹难以收敛。最终，direct feature injection通过直接扰动特征空间，避免了上述问题，取得了更好效果。
反思收获
通过这个低资源个人项目，我深刻体会到扩散模型在视频扩展任务中的权衡：条件机制虽强大，但需仔细处理信息压缩和内存问题；直观的重构引导虽优雅，却易受分布复杂性影响而失效。Direct feature injection的成功证明了简单直接干预特征空间往往更鲁棒且高效。这次实验强化了迭代调试能力，也让我认识到在有限硬件下，通过针对性方法选择仍能探索前沿问题，积累了宝贵的独立研究经验。9 sources用英文说Current Status
Current diffusion models for video generation are typically limited to producing short sequences with a fixed number of frames, lacking built-in mechanisms for maintaining temporal consistency in longer videos. This restricts their practical use, especially in resource-constrained environments.
learnopencv.commask-cond-video-diffusion.github.io

Motivation
As a personal project at UCI, I trained the model on a single GTX 1080 Ti GPU (11GB VRAM) to explore feasible methods for extending diffusion models to generate long, consistent video sequences. The goal was to experiment with different extension techniques under limited resources, understand their trade-offs, and gain hands-on experience in implementing and comparing conditioning approaches for video tasks.
amazon.comAmazon.com: MSI GEFORCE GTX1080 TI GAMING X 11G : Electronics
Achievements
I implemented three methods for long video generation:

Classifier-free guidance (CFG) conditioned on previous history frames.

theaisummer.comAn overview of classifier-free diffusion guidance: impaired model ...

Reconstruction-guided sampling, enforcing consistency by matching overlapping frames.

researchgate.netThe comparison of sampling process of different video diffusion ...

Direct feature injection, directly incorporating features from prior frames into subsequent generation.

Among these, direct feature injection achieved the best visual quality and consistency for extended sequences.
researchgate.netArtificial neural network architecture (ANN i-h 1-h 2-h n-o ...
The following examples show frame sequences from the methods, highlighting consistency in long videos:
medium.comstreamingt2v.github.ioreddit.com


Challenges
The CFG approach suffered from information loss when compressing historical frames (around 10) as conditions, leading to degraded quality in subsequent frames, along with higher memory usage from additional parameters during training and sampling. Shifting to reconstruction-guided sampling was intuitive—assuming perfect fixed-length generation, consistency is enforced via overlapping frame matching—but proved unstable, with the score function failing to decrease smoothly during sampling, often trapping in local minima due to the complex distribution of video frames. Ultimately, direct feature injection avoided these issues by directly perturbing the feature space, yielding superior results.
Reflections and Gains
Through this low-resource personal project, I gained deep insights into the trade-offs of conditioning mechanisms in diffusion models: powerful but sensitive to compression and memory constraints in CFG, and theoretically elegant yet practically unstable in reconstruction guidance due to distribution complexity. The success of direct feature injection highlighted the robustness and efficiency of simpler, direct feature-space interventions. This experience strengthened my iterative debugging skills and demonstrated that targeted method selection can enable exploration of cutting-edge problems even with limited hardware, building valuable independent research experience.9 sourceshow to include an image in .md file in vscodeTo include an image in a Markdown (.md) file in Visual Studio Code, use the standard Markdown image syntax:
Markdown![Alt text](path/to/your/image.png)

Alt text: Descriptive text for accessibility (e.g., "Description of the image"). It shows if the image fails to load.
path/to/your/image.png: The file path to the image.

Common Ways to Add Images

Manual Syntax (Local Images):
Use a relative path (recommended for portability):
Same folder: ![Alt text](image.png)
Subfolder: ![Alt text](images/image.png)
Parent folder: ![Alt text](../images/image.png)

Or an absolute/workspace-root path (starts with /): ![Alt text](/assets/image.png) (where /assets is at the project root).
For online images: ![Alt text](https://example.com/image.png)

Drag and Drop (Easiest for Local Files):
Open your .md file in VS Code.
Drag an image file from the VS Code Explorer (or your file system).
Hold Shift while dropping it into the editor.
VS Code automatically inserts the correct Markdown syntax with a relative path.

Paste from Clipboard (Built-in in Recent VS Code Versions, e.g., 1.79+):
Copy an image (e.g., screenshot or from a website).
Place your cursor in the .md file and press Ctrl + V (Windows/Linux) or Cmd + V (macOS).
VS Code saves the image to your workspace (often in a default folder like images/ or next to the file) and inserts the link automatically.
Customize save location via settings: Search for "markdown.editor.filePaste" in VS Code Settings.

From Web/Browser:
Drag an image directly from your browser into the .md file (may copy to workspace if configured).


Preview the Image

Press Ctrl + Shift + V (Windows/Linux) or Cmd + Shift + V (macOS) to open Markdown preview.
Or click the preview icon (magnifying glass) in the top-right of the editor.

Troubleshooting (If Image Doesn't Show in Preview)

Ensure the path is correct and relative to the .md file.
Avoid spaces in filenames (or percent-encode them, e.g., my%20image.png).
Add a blank line after the image syntax if it's at the end of the file.
For external URLs: Adjust preview security settings (in preview: click "..." > Change Preview Security > Allow insecure content if needed).
Older VS Code versions may need extensions like "Markdown Paste" or "Paste Image" for advanced pasting.

This works natively in modern VS Code—no extensions required for basic use!25 sourcesfor instance, the image is here: assets/img/video_diffusion/DVM_arch.png
Also maybe use html to display images:
Markdown<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>To include your local image (e.g., at assets/img/video_diffusion/DVM_arch.png) in a Markdown (.md) file in VS Code, use the standard Markdown image syntax with a relative path. This works reliably in VS Code's built-in Markdown preview (Ctrl+Shift+V or Cmd+Shift+V on macOS).
Basic Syntax (Recommended)
Markdown![Architecture of the Diffusion Video Model](assets/img/video_diffusion/DVM_arch.png)

Replace the alt text ("Architecture of the Diffusion Video Model") with something descriptive.
The path is relative to your .md file's location. Adjust if needed (e.g., ../assets/img/... if the image is one folder up).
Open the preview to see it rendered. Local images display natively—no extra setup required.

Easier Insertion Methods

Drag and Drop (Fastest):
Open your .md file.
Drag the image file from VS Code's Explorer pane.
Hold Shift while dropping it into the editor.
VS Code auto-inserts the correct ![alt text](relative/path) syntax.

Paste from Clipboard (For screenshots or copied images):
Copy the image (e.g., screenshot).
Paste directly (Ctrl+V / Cmd+V) into the .md file.
VS Code saves it to your workspace (often in a folder like assets/ or next to the file) and inserts the link.


Using HTML for More Control (e.g., Size, Layout, Captions)
Markdown supports embedding raw HTML, and VS Code's preview renders it well. This is great for centering, resizing, or multi-image layouts.
Single Image with Caption and Styling:
Markdown<figure>
    <img src="assets/img/video_diffusion/DVM_arch.png" alt="Architecture of the Diffusion Video Model" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
    <figcaption style="text-align: center; font-style: italic;">Architecture of the proposed Diffusion Video Model (DVM)</figcaption>
</figure>
Side-by-Side or Grid Layout (Using Bootstrap-like Classes):
If your final output isn't plain Markdown (e.g., rendered via GitHub Pages, Jekyll, or a site), you can use simple <div> for layout. VS Code preview supports basic CSS:
Markdown<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
    <div style="flex: 1; min-width: 300px;">
        <img src="assets/img/video_diffusion/DVM_arch.png" alt="DVM Architecture" style="width: 100%; height: auto;">
        <p style="text-align: center; font-style: italic;">DVM Architecture</p>
    </div>
    <div style="flex: 1; min-width: 300px;">
        <img src="assets/img/video_diffusion/another_image.png" alt="Another Example" style="width: 100%; height: auto;">
        <p style="text-align: center; font-style: italic;">Comparison or Results</p>
    </div>
</div>
About the Liquid Code You Shared
The example you provided:
Markdown<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    ...
</div>
This is Jekyll-specific Liquid templating (common in themes like Chirpy or Minimal Mistakes). It uses a custom figure.liquid include for fancy figures/captions with Bootstrap classes.

It won't work in plain Markdown preview in VS Code (Liquid tags like {% include %} are ignored or shown as text).
It only renders properly when building with Jekyll (e.g., for a blog on GitHub Pages).
If your project showcase is just a plain .md file (e.g., for a professor or README), stick to standard Markdown or HTML as above.

Troubleshooting Tips

Ensure the path is correct (case-sensitive on some systems).
Add a blank line before/after the image syntax.
If images don't show: Right-click preview > "Change Preview Security" > Allow insecure local content.
For better Markdown features, install free extensions like "Markdown All in One" or "Markdown Preview Enhanced".

This should get your DVM_arch.png displaying nicely in your project showcase! If it's for a specific theme/site, let me know for tailored advice.40 sourceshow about write project showcase description for this project?PDFMath_199.pdfCurrent Status
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