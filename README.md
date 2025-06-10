<div align="center">

# Demystifying Flux.1: An unofficial documentation of FLUX.1 text-to-image architecture
> ⚠️ This is an **unofficial** and **reverse-engineered** documentation project. It is not affiliated with the original authors or organizations behind FLUX.1.

[Or Greenberg](https://OrGreenberg.github.io)<sup>1</sup>,<sup>2</sup>&nbsp;&nbsp;&nbsp;

<div>
<sup>1</sup> GM R&D &nbsp;|&nbsp; <sup>2</sup> Hebrew University of Jerusalem, Israel
</div>

![image](https://github.com/user-attachments/assets/7136918d-0356-4cca-a516-b88de55bf327)

---
**[Or Greenberg]**
[Hebrew University of Jerusalem, Israel]
[General Motors R&D]
---

## Abstract
*FLUX.1* is a diffusion-based text-to-image generation model developed by Black Forest Labs, designed to achieve faithful text-image alignment while maintaining high image quality and diversity. FLUX is considered state-of-the-art in image synthesis, outperforming popular models such as Midjourney, DALL·E 3, Stable Diffusion 3 (SD3), and SDXL. Although publicly available as open source, the authors have not released official technical documentation detailing the model’s architecture or training setup. This report summarizes an extensive reverse-engineering effort aimed at demystifying FLUX’s architecture directly from its source code, to support its adoption as a backbone for future research and development.


## Content
- [Overview](#abstract)
