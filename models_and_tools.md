---
layout: default
title: "Model and Tools"
permalink: /models-and-tools/
---

<style>
  body {
    background-color: #f0f0f0; /* light gray */
  }
</style>

<a name="sec:hub"></a>

<div style="text-align: center;" markdown="1">

# **Model and Tools**
---
</div>
<br>

Below is a list of the currently available models in the FLUX family. We divide them into [Text-to-Image models](#text-to-image), and [tools](#tools) which provide additional capabilities such as image editing, spatial guidance, and image variation. While new models are being released rapidly, this list is current as of June 5, 2025. You can contribute to this blog by suggesting newly released models via the blog's [GitHub Issues](https://github.com/OrGreenberg/DemystifyingFlux/issues) page.

---
<br>
## text-to-image
<a name="text-to-image"></a>

The FLUX.1 suite of text-to-image models comes in three variants, all share the same architecture of 12B parameters, striking a balance between accessibility and model capabilities [^FLUXAnnounce].

- **FLUX.1[pro]**  
  This is the highest-performing variant of the FLUX.1 family, offering superior prompt alignment, visual detail, and output diversity. It is designed for commercial use and is accessible only through licensed API endpoints such as Replicate or Fal.ai. The model weights are not publicly released, making it a hosted-only solution. This version is ideal for production environments where top-tier image generation quality is critical and commercial licensing is feasible.

- **FLUX.1[dev]**  
  Provides an open-weight alternative to Pro, guidance-distilled [^meng2023distillation] from it. It is licensed strictly for non-commercial use, making it accessible to researchers, developers, and hobbyists for experimentation and academic work. Unlike Pro, Dev can be downloaded and run locally, offering full control and flexibility for non-commercial applications. The FLUX.1 [dev] model is available via Hugging Face and can be directly tried out on Replicate or Fal.ai. It should be noted that the Dev version might require more sampling steps than the Pro version to achieve high-level performance.

- **FLUX.1[schnell]**  
  A speed-optimized, distilled variant of FLUX.1 Pro, designed to significantly reduce inference time while retaining reasonable image quality. It shares the same 12B architecture but is trained via timesteps-distillation [^sauer2024fast] from the Pro model, meaning it learns to mimic Pro’s outputs using fewer sampling steps, up to a single step, by optimizing for speed and efficiency. This results in faster generations at the cost of some prompt fidelity and fine detail. Released under the permissive Apache 2.0 license, Schnell allows unrestricted commercial use, making it ideal for real-time, low-latency applications and lightweight deployments where speed and open licensing outweigh the need for peak visual fidelity.

- **FLUX1.1[pro]**  
  An enhanced version of the FLUX.1[pro] model, capable of faster image generation while also improving image quality, prompt adherence, and diversity. The model introduces new modes—**Ultra**, enabling 4× higher resolution without compromising speed, and **Raw**, producing hyper-realistic, candid-style images. Additionally, a prompt upsampling feature leverages large language models (LLMs) to expand and enrich user prompts, enhancing creative outputs. While not covered in this report, FLUX 1.1 Pro is available via API on platforms like Replicate and Fal.ai, with commercial licensing required.

---
<br>
## tools
<a name="tools"></a>

Originally developed for text-to-image generation, FLUX.1 has since been extended by Black Forest Labs with a suite of tools supporting a range of use cases beyond its original purpose. This report does not explore each tool in depth as it does for the FLUX.1 model itself. Below is a list of publicly available tools, current as of June 5th, 2025:

- **FLUX-Fill**  
  An inpainting and outpainting model, enabling editing and expansion of real and generated images given a text description and a binary mask.

- **FLUX-Canny**  
  Enables structural guidance based on Canny edges extracted from an input image and a text prompt.

- **FLUX-Depth**  
  Enables structural guidance based on a depth map extracted from an input image and a text prompt.

- **FLUX-Redux**  
  An adaptor that aligns dense (per-token) image embeddings extracted by the SigLIP image encoder [^zhai2023sigmoid] with the T5 text-embedding space for image-conditioned generation with a pre-trained FLUX.1 base model. While limited for image variation in the Dev version, it integrates into more complex workflows unlocking image restyling via prompt when using the FLUX1.1 [pro] Ultra model, allowing for combining input images and text prompts.

- **FLUX-Kontext**  
  Extends the text-to-image capabilities of the base FLUX model into powerful in-context generation and editing workflows by enabling multimodal conditioning—allowing users to guide generation using both text and reference images. While *FLUX-Redux* had already introduced basic multimodal conditioning for image variation and prompt-driven restyling, *FLUX-Kontext* unlocks advanced functionality such as localized edits, style transfer, and character or scene consistency across images. With models like Kontext [pro] and [max], the focus shifts toward fast, iterative generation and high-precision editing. As of June 5th, 2025, the *FLUX-Kontext* models are not yet publicly released, and therefore could not be directly explored or evaluated in this report.

  <br>
---
<br>
## References
[^meng2023distillation]: Meng, Chenlin, et al. "On distillation of guided diffusion models." (2023)‏.
[^sauer2024fast]: Sauer, Axel, et al. "Fast high-resolution image synthesis with latent adversarial diffusion distillation." (2024).‏
[^FLUXAnnounce]: [Black-Forest-Labs official FLUX.1 announcement](https://bfl.ai/announcements/24-08-01-bfl), (2024).
[^zhai2023sigmoid]: Zhai, Xiaohua, et al. "Sigmoid loss for language image pre-training." (2023).‏
