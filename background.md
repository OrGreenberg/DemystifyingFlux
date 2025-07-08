---
layout: default
title: "Background"
permalink: /background/
---
---
To better understand the foundations of FLUX.1, we briefly present several key models and methods that preceded it and influenced its design. This overview is intentionally high-level, aiming only to provide the essential background necessary to grasp the core principles behind FLUX. For readers interested in a deeper exploration of these foundational works, references to the original papers and technical reports are provided throughout this section.

---
<br>
## ML-Based Image Generation
<a name="image-synthesis"></a>

The task of ML-based image generation refers to a family of models designed to generate novel samples from a simple distribution that emulates a source dataset. Specifically, some models learn to map a dataset of images to a simple distribution (e.g., Gaussian), from which new images can be sampled. Several types of generative models have been developed in recent years, including—but not limited to—Variational Autoencoders (VAEs) [^kingma2013auto], Generative Adversarial Networks (GANs) [^goodfellow2020generative], Normalizing Flows (NFs) [^papamakarios2021normalizing] and Diffusion Models (DMs) [^ho2020denoising].

Among these, **text-to-image models** (e.g., DALL·E [^ramesh2021zero], Stable Diffusion [^rombach2022high], and FLUX [^flux2024]) have achieved state-of-the-art performance and are widely used in applications that generate high-quality images from textual instructions, commonly referred to as *prompts*.

---
<br>
## Diffusion Models
<a name="diffusion-models"></a>

Within this scope, **Diffusion Models** (DMs) [^ho2020denoising] have recently emerged as the state-of-the-art (SoTA) approach for text-conditioned image generation. They work by learning to reverse a gradual noising process applied to training data. While early diffusion models relied on U-Net architectures composed of convolutional layers augmented with attention mechanisms for image–text alignment, recent models such as SD3 [^esser2024scaling] and FLUX.1 [^flux2024] have transitioned to fully Transformer-based architectures for the denoising process, offering improved scalability and context modeling. During the training process, clean images are incrementally corrupted by Gaussian noise whose magnitude is determined by a timestep-specific schedule, and the model learns to iteratively denoise them. 

In the most common version of DM optimization, the model is trained to predict the normalized added noise $\epsilon$, and optimized using a reconstruction loss. It is formulated as:

$$
\mathcal{L}_\epsilon = \mathbb{E}_{x_t,\epsilon,t}\left[  \|\epsilon_\theta(x_t,t) - \epsilon\|^2  \right]
$$

where:

$$
x_t = \sqrt{\alpha_t}x_0 + \sqrt{1 - \alpha_t}\epsilon
$$

represent the noisy version of the clean image $$x_0$$, corrupted by the gaussian noise $$\epsilon \sim \mathcal{N}(0,1)$$ to timestep $$t$$.

At inference time, the model begins from pure Gaussian noise and iteratively denoises it to produce realistic images. Unlike earlier generative models such as GANs, which rely on adversarial training, diffusion models optimize a reconstruction-based objective. This leads to more stable training and superior output quality, particularly for high-resolution image synthesis. Moreover, they offer fine-grained control over the generation process, enabling powerful applications such as super-resolution, inpainting, and text-conditioned image generation.

---
<br>
## Stable Diffusion
<a name="stable-diffusion"></a>

**Stable Diffusion** is a family of latent diffusion models (LDMs)  [^rombach2022high] designed to generate high-quality images conditioned on textual prompts. Unlike traditional pixel-space diffusion models, LDMs operate in a compressed latent space, significantly improving computational efficiency while maintaining visual fidelity. This is achieved by encoding input images into a lower-dimensional latent representation using a pre-trained autoencoder. The diffusion process is then applied in this latent space, and the final output is decoded back into pixel space.

The 1.x versions of Stable Diffusion (1.0 to 1.5) introduced this efficient framework for open-domain image synthesis, using a frozen CLIP [^radford2021learning]} text encoder for prompt conditioning and a U-Net architecture for latent denoising. These models, trained on subsets of LAION-2B [^schuhmann2022laionb], quickly gained popularity due to their open-source release, high versatility, and ease of fine-tuning.

**Stable Diffusion v2.1** introduced several architectural upgrades, including a switch to OpenCLIP [^ilharco_gabriel_2021_5143773] and training on higher resolutions ($$768×768$$), improving image structure and fidelity. **SDXL** [^podell2023sdxl], a major advancement in the series, uses a two-stage architecture with separate base and refiner models, allowing better handling of complex prompts and achieving significantly improved realism and prompt alignment. **SD-Turbo** [^sauer2024adversarial] further innovates by enabling near real-time image generation through a Adversarial-diffusion-distillation (ADD) process, where a student model trains to mimic the performance of a teacher model (SD2.1 or SDXL) using fewer inference steps.

Most recently, **Stable Diffusion 3** (SD3) [^esser2024scaling] integrates a diffusion transformer backbone and adopts multimodal training (see [Section: transormers](#transformers)), combining both text and image understanding for improved prompt adherence, compositional reasoning, and consistency. SD3 is designed for scalability and robustness, closing the gap between open models and proprietary systems like DALL·E 3 and Midjourney in terms of controllability and quality.

---
<br>
## Rectified Flows
<a name="rectified-flows"></a>

**Rectified Flows** (RF) [^liu2022flow] is a recent generative modeling framework that simplifies and generalizes diffusion models by replacing stochastic denoising with a deterministic vector field, based on the *Flow Matching* principle introduced in [^lipman2022flow]. 

While RF is a training paradigm rather than an architectural change, its relevance to FLUX.1 stems from the fact that FLUX.1 was trained using this method. In this blog, we do not provide a detailed explanation of Rectified Flow. Instead, we briefly highlight the key distinction between standard diffusion model training and the approach used in FLUX. This divergence in training schemes likely contributes to differences in performance and convergence behavior. For a comprehensive understanding of Rectified Flow, we refer readers to the original works.

While diffusion models (such as **DDPMs** [^ho2020denoising]) learn to predict the noise ($$\epsilon$$) added during a forward diffusion process (see [Section: Diffusion Models](#diffusion-models)), RFs train a model to predict a velocity vector $$v$$ that directly points from a noisy point $$x_0$$ (not to be confused with DM notation, where $$x_0$$ denotes the ``clean'' image) back to the data point $$x_1$$, along a straight interpolation path.

In diffusion models, the network is trained using the $\epsilon$-objective, which minimizes the difference between the predicted noise $\epsilon_\theta(x_t,t)$ and true noise $\epsilon$ added to a sample (see formulation in [Section: Diffusion Models](#diffusion-models)).

In contrast, RF defines a deterministic interpolation between the data $$x_1$$ and noise $$x_0$$, and trains the model $$v_\theta$$ to predict the velocity vector between the two:

$$
\mathcal{L}_v = \mathbb{E}_{x_0,x_1,t}
\left[ \|v_\theta(x_t,t) - (x_1 - x_0)\|^2 \right]
$$

where $$x_t=(1-t)x_0 + tx_1$$ is a linear interpolation between $$x_0 \sim \mathcal{N}(0, I)$$ and $$x_1 \sim p_{data}$$, and $$v(x_t, t)$$ represents the target vector pointing from $$x_0$$ to $$x_1$$. 


Unlike diffusion models that rely on stochastic sampling, noise schedules, and denoising objectives, Rectified Flow leverages a velocity-based formulation that defines a deterministic transport path from noise to data. This approach eliminates the need for iterative noise perturbation and reverse denoising, enabling faster and more stable image synthesis. By solving a simple ODE with a learned velocity field, Rectified Flow achieves high-quality generation with significantly reduces complexity in both training and inference—making it a compelling and efficient alternative to traditional diffusion-based methods.

---
<br>
## Transformers
<a name="transformers"></a>

**Transformers** are a class of neural network architectures originally developed for natural language processing (NLP) tasks [^vaswani2017attention]. 

The transformer’s core innovation is the *self-attention* mechanism, which enables the model to weigh relationships between all elements in an input sequence. This architecture has since become the foundation of many state-of-the-art models across modalities—including text, image, video, and multimodal systems.

In each attention block, the model computes three matrices from the input embeddings: queries (Q), keys (K), and values (V), using learned linear projections. The attention scores are computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

Here, $$d_k$$ is the dimensionality of the key vectors, and the softmax operation ensures the attention weights sum to 1. This mechanism allows the model to assign different levels of importance to different tokens when computing a new representation for each token in the sequence. In practice, multi-head attention is used, where multiple sets of $$Q$$/$$K$$/$$V$$ projections are computed in parallel to capture diverse types of relationships.

For vision tasks, Vision Transformers (ViTs) [^dosovitskiy2020image] tokenize an image into fixed-size non-overlapping patches (e.g., $16\times16$ pixels),flatten each patch, and project them into an embedding space. These patch embeddings are then processed by a standard transformer encoder, often with added positional encodings to preserve spatial structure. This enables ViTs to model long-range dependencies across an image without convolutional inductive biases.

While originally $$Q$$/$$K$$/$$V$$ represenations are all projected from the same input embedding in a process denotes *Self-Attention*, In text-to-image generation, transformers often use *Cross-Attention* to condition image synthesis on textual prompts. In this setup, the queries ($$Q$$) are projected from the image tokens, while the keys ($$K$$) and values ($$V$$) are projected from the text embeddings. This allows the model to modulate image generation based on semantic information from the prompt.

<br>
---
<br>
## References
[^kingma2013auto]: Zeng, Yu, Huchuan Lu, and Ali Borji. "Statistics of deep generated images." (2017).‏
[^goodfellow2020generative]: Goodfellow, Ian, et al. "Generative adversarial networks." (2020).
[^papamakarios2021normalizing]: Papamakarios, George, et al. "Normalizing flows for probabilistic modeling and inference." (2021).
[^ho2020denoising]: Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." (2020).‏‏
[^ramesh2021zero]: Ramesh, Aditya, et al. "Zero-shot text-to-image generation." (2021).‏ 
[^rombach2022high]: Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." (2022).
[^flux2024]: Black Forest Labs. \url{https://github.com/black-forest-labs/flux}. (2024)
[^esser2024scaling]: Esser, Patrick, et al. "Scaling rectified flow transformers for high-resolution image synthesis." (2024).‏
[^radford2021learning]: Radford, Alec, et al. "Learning transferable visual models from natural language supervision." (2021).‏
[^schuhmann2022laionb]: Schuhmann, Christoph, et al. "Laion-5b: An open large-scale dataset for training next generation image-text models." (2022).
[^ilharco_gabriel_2021_5143773]: Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., ... & Jitsev, J. Reproducible scaling laws for contrastive language-image learning. (2023).‏‏
[^podell2023sdxl]: Podell, Dustin, et al. "Sdxl: Improving latent diffusion models for high-resolution image synthesis." (2023).‏
[^sauer2024adversarial]: Sauer, Axel, et al. "Adversarial diffusion distillation." (2024).‏
[^liu2022flow]: Liu, Xingchao, Chengyue Gong, and Qiang Liu. "Flow straight and fast: Learning to generate and transfer data with rectified flow." (2022).‏
[^lipman2022flow]: Lipman, Yaron, et al. "Flow matching for generative modeling." (2022).‏
[^vaswani2017attention]: Vaswani, Ashish, et al. "Attention is all you need." (2017).‏
[^dosovitskiy2020image]: Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." (2020).‏


