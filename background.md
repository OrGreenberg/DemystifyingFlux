---
layout: page
title: "Background"
permalink: /background/
---
---
To better understand the foundations of **FLUX**, we briefly present several key models and methods that preceded it and influenced its design. This overview is intentionally high-level, aiming only to provide the essential background necessary to grasp the core principles behind FLUX. For readers interested in a deeper exploration of these foundational works, references to the original papers and technical reports are provided throughout this section.

---
## Image Synthesis

The task of *image synthesis* refers to a family of models designed to generate novel samples from a simple distribution that emulates a source dataset. Specifically, image synthesis models learn to map a dataset of images to a simple distribution (e.g., Gaussian), from which new images can be sampled. Several types of generative models have been developed in recent years, including—but not limited to- Variational Autoencoders (VAEs) [1], Generative Adversarial Networks (GANs) [2], Normalizing Flows (NFs) [3] and Diffusion Models (DMs) [4].

Among these, **text-to-image models** such as DALL·E [5], Stable Diffusion [6], and FLUX [7] have achieved state-of-the-art performance and are widely used in applications that generate high-quality images from textual instructions, commonly referred to as *prompts*.

---
## Diffusion Models

**Diffusion Models (DMs)** [4] are a class of text-to-image generative models that have recently emerged as state-of-the-art (SoTA) in image synthesis. They work by learning to reverse a gradual noising process applied to training data.

While early diffusion models relied on U-Net architectures composed of convolutional layers augmented with attention mechanisms for image–text alignment, recent models such as SD3 [8] and FLUX.1 [7] have transitioned to **fully Transformer-based architectures**, offering improved scalability and context modeling.

During training, clean images are progressively corrupted with Gaussian noise over a series of time steps, and the model is trained to denoise the corrupted images step-by-step. At inference time, the model starts from pure noise and iteratively denoises it to generate realistic samples.

Unlike earlier generative approaches such as GANs, which rely on adversarial training, diffusion models optimize a **likelihood-based objective**. They are therefore known for their training stability and high-quality outputs, especially in high-resolution image synthesis.

Their iterative nature allows fine-grained control over the generation process, enabling powerful applications such as Super-resolution, Inpainting and Text-conditioned image generation.

---
## Stable Diffusion

**Stable Diffusion** is a family of *latent diffusion models* (LDMs) [6] designed to generate high-quality images conditioned on textual prompts. Unlike traditional pixel-space diffusion models, LDMs operate in a compressed latent space, significantly improving computational efficiency while maintaining visual fidelity. This is achieved by encoding input images into a lower-dimensional latent representation using a pre-trained autoencoder. The diffusion process is then applied in this latent space, and the final output is decoded back into pixel space.

The original **Stable Diffusion v1.4** and **v1.5** models introduced this efficient framework for open-domain image synthesis, using a frozen CLIP [9] text encoder for prompt conditioning and a U-Net architecture for latent denoising. These models, trained on subsets of LAION-2B [10], quickly gained popularity due to their open-source release, high versatility, and ease of fine-tuning.

**Stable Diffusion v2.1** introduced several architectural upgrades, including a switch to OpenCLIP [11] and training on higher resolutions (768×768), improving image structure and fidelity. **SDXL** [12] further improved realism and prompt alignment using a two-stage architecture (base + refiner). **SD-Turbo** [13] enabled near real-time image generation via an **Adversarial-Diffusion-Distillation (ADD)** process, where a *student model* mimics a *teacher* (e.g., SD2.1 or SDXL) with fewer steps. 

Most recently, Stable Diffusion 3 (SD3) [8] integrates a diffusion transformer backbone and adopts multimodal training, combining both text and image understanding for improved prompt adherence, compositional reasoning, and consistency. SD3 is designed for scalability and robustness, closing the gap between open models and proprietary systems like DALL·E 3 and Midjourney in terms of controllability and quality.

---
## Rectified Flows

**Rectified Flow** [14] is a recent generative modeling framework that simplifies and generalizes diffusion models by replacing stochastic denoising with a **deterministic vector field**, based on *Flow Matching* [15].

While diffusion models (such as DDPMs [4]) learn to predict the noise ($$\epsilon$$) added during a forward diffusion process, Rectified Flow trains a model to predict a velocity vector $v$ that directly points from a noisy point $$x_0$$ (not to be confused with DM notation, where $$x_0$$ denotes the "clear" image) back to the data point $$x_1$$, along a straight interpolation path.

In diffusion models, the network is trained using the epsilon objective, which minimizes the difference between the predicted noise $$\epsilon_\theta(x_t,t)$$ and true noise $$\epsilon$$ added to a sample:

$$
\mathcal{L}_\epsilon = \mathbb{E}_{x_t,\epsilon,t}\left[  \|\epsilon_\theta(x_t,t) - \epsilon\|^2  \right]
$$

where:

$$
x_t = \sqrt{\alpha_t}x + \sqrt{1 - \alpha_t}\epsilon
$$

represent the noisy image $$x$$ at timestep $$t$$.

In contrast, Rectified-Flow defines a deterministic interpolation between the data $$x_1$$ and noise $$x_0$$, and trains the model $$v_\theta$$ to predict the velocity vector between the two:

$$
\mathcal{L}_v = \mathbb{E}_{x_0,x_1,t}
\left[ \|v_\theta(x_t,t) - (x_1 - x_0)\|^2 \right]
$$

where $$x_t=(1-t)x_0 + tx_1$$ is a linear interpolation between $$x_0 \sim \mathcal{N}(0, I)$$ and $$x_1 \sim p_{data}$$, and $$v(x_t, t)$$ represents the target vector pointing from $$x_0$$ to $$x_1$$. 


Unlike diffusion models that rely on stochastic sampling, noise schedules, and denoising objectives, Rectified Flow leverages a velocity-based formulation that defines a deterministic transport path from noise to data. This approach eliminates the need for iterative noise perturbation and reverse denoising, enabling faster and more stable image synthesis. By solving a simple ODE with a learned velocity field, Rectified Flow achieves high-quality generation with significantly reduces complexity in both training and inference—making it a compelling and efficient alternative to traditional diffusion-based methods.

While Rectified Flow is a training technique rather than an architectural innovation, its relevance to FLUX lies in the fact that FLUX was trained using this paradigm. Although this report primarily focuses on architectural aspects, we include this subsection to highlight the role Rectified Flow plays in shaping FLUX's performance and convergence behavior.

---
## Transformers

**Transformers** are a class of neural network architectures originally developed for NLP tasks [16]. Their core innovation is the **self-attention mechanism**, which lets the model learn relationships between all elements in a sequence.

In each attention block, queries (Q), keys (K), and values (V) are computed via learned projections:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

Where $$d_k$$ is the dimensionality of keys and the Softmax operation ensures the attention scores sum to 1. This mechanism allows the model to assign different levels of importance to different tokens when computing a new representation for each token in the sequence. In practice, multi-head attention is used, where multiple sets of $$Q$$/$$K$$/$$V$$ projections are computed in parallel to capture diverse types of relationships.

For vision tasks, Vision Transformers (ViTs) [17] tokenize an image into fixed-size non-overlapping patches (e.g., $16\times16$ pixels), flatten each patch, and project them into an embedding space. These patch embeddings are then processed by a standard transformer encoder, often with added positional encodings to preserve spatial structure. This enables ViTs to model long-range dependencies across an image without convolutional inductive biases.

In text-to-image generation, transformers often use **cross-attention** to condition image synthesis on textual prompts. In this setup, the image tokens (or latent features in the case of latent diffusion) act as queries, while the text embeddings serve as keys and values. This allows the model to modulate image generation based on semantic information from the prompt.
