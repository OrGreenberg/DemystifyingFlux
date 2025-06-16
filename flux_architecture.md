---
layout: default
title: "FLUX.1 Architecture"
permalink: /flux-architecture/
---

# FLUX.1 Architecture

## Intro

**FLUX.1** [^flux2024] is a rectified flow transformer trained in the latent space of an image encoder, introduced by **Black Forest Labs** in August 2024.

The FLUX.1 models (see [Section: Hub](#sec-hub)) demonstrate State-of-the-art (SoTA) performance for text-to-image tasks, in both terms of output quality and image-text alignment, as demonstrated in Figures [1](#figure-1) and [2](#figure-2) using the ELO-score metric, which ranks image generation models based on human preferences in head-to-head comparisons. A qualitative illustration and comparison to SD is provided in the Appendix.

![FLUX.1 defines a new state-of-the-art in image detail, prompt adherence, style diversity and scene complexity for text-to-image synthesis. Evaluation from FLUXAnnounce](assets/flux_score1.jpg)  
**Figure 1**  
<a name="figure-1"></a>

![ELO scores for different aspects: Prompt Following, Size/Aspect Variability, Typography, Output Diversity, Visual Quality. Evaluation from FLUXAnnounce](assets/flux_score2.jpg)  
**Figure 2**  
<a name="figure-2"></a>

While the model likely adheres to the Rectified Flow training paradigm, the exact details regarding the training setup — including the dataset, scheduling strategy, and hyperparameters — have not been publicly disclosed. However, the model’s architecture and inference scheme can be reverse-engineered from the publicly available inference code. In this section, we outline the model’s architecture to demystify its behavior at inference time. We begin by introducing key pre-trained components and foundational concepts in [Preliminaries](#preliminaries), followed by a detailed breakdown of the model’s individual blocks and the end-to-end inference pipeline in [Architecture](#architecture), which together constitute its text-to-image generation mechanism.

---

## Preliminaries  
<a name="preliminaries"></a>

In this section, we will briefly describe some key pre-trained components and foundational concepts used in FLUX.1 along the sampling pipeline.

### CLIP Text Encoder

CLIP (Contrastive Language–Image Pretraining) [^radford2021learning] is a foundation model developed by OpenAI, designed to bridge vision and language understanding. CLIP's text-encoder transforms natural language prompts into class-level or dense, high-dimensional embeddings that can be directly compared with image embeddings in a shared latent space. Trained on hundreds of millions of image–text pairs, the CLIP text encoder captures rich semantic information, enabling models to align textual descriptions with corresponding visual content effectively, making it widely used in text-to-image generation tasks.

### T5 Text Encoder

T5 (Text-To-Text Transfer Transformer) [^raffel2020exploring] is a versatile language model developed by Google that frames all NLP tasks—(e.g., translation, summarization, and question answering)—as text-to-text problems. Its text encoder converts input text into contextualized embeddings using a Transformer-based architecture trained with a denoising objective over massive language corpora. Unlike CLIP’s text encoder, which is trained jointly with an image encoder to produce embeddings aligned with visual features for contrastive learning, T5 is trained purely on textual data and optimized for language understanding and generation. This makes T5 well-suited for providing rich, token-level semantic representations even for long and complex textual prompts.

### Rotary Positional Embeddings

Rotary Positional Embeddings (RoPE) [^su2024roformer] are a method for injecting positional information into Transformer models by rotating the query and key vectors in multi-head self-attention according to their token positions. Unlike traditional sinusoidal or learned positional embeddings that are **added** to input tokens, RoPE applies a **rotation** in the complex plane that preserves relative positional relationships across sequences. This approach allows the model to generalize better to unseen sequence lengths and supports extrapolation beyond training data. RoPE has become popular in recent vision-language models, where understanding spatial relationships between tokens—especially when repurposing textual positions for image patches—is crucial.

### Adaptive Layer Normalization

Adaptive Layer Normalization (AdaLN) [^keddous2024vision] is a conditioning mechanism used in Transformer-based models [^nichol2021glide, ^sauer2023stylegan] to modulate intermediate activations based on external input, such as text or image embeddings. Unlike standard Layer Normalization, which applies fixed scaling and shifting parameters, AdaLN dynamically generates these parameters as functions of a conditioning vector (see Figure 3). This allows the model to adapt its behavior at each layer according to the input prompt or guidance signal.

![AdaLN layer, where MSA (Multi-head Self Attention) and MLP (Multi-Layer Processor) modulation parameters are computed based on the input tensor. In Single-Stream block, MLP modulation is not computed.](assets/ADALN.jpg)  
**Figure 3**  
<a name="figure-3"></a>

---

## References

[^flux2024]: FLUX.1 paper, Black Forest Labs (2024).  
[^radford2021learning]: Radford et al., “Learning Transferable Visual Models From Natural Language Supervision”, OpenAI (2021).  
[^raffel2020exploring]: Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer”, JMLR (2020).  
[^su2024roformer]: Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embedding”, (2024).  
[^keddous2024vision]: Keddous et al., “Vision Pro Transformers via Adaptive Normalization”, (2024).  
[^nichol2021glide]: Nichol et al., “GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models”, OpenAI (2021).  
[^sauer2023stylegan]: Sauer et al., “StyleGAN-T: Unlocking the Power of GANs for Fast, High-Resolution Text-to-Image Synthesis”, (2023).
