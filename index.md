---
layout: page
title:
permalink: /
---

<style>
  body {
    background-color: #f0f0f0; /* light gray */
  }
</style>

<div style="text-align: center;" markdown="1">

## **Demystifying Flux.1**
# An unofficial documentation of FLUX.1 text-to-image architecture

[Or Greenberg](https://scholar.google.com/citations?user=p6tIB6UAAAAJ&hl=iw)<sup>1,2</sup>&nbsp;&nbsp;&nbsp;

<br>
<sup>1</sup> General Motors R&D &nbsp;|&nbsp; <sup>2</sup> Hebrew University of Jerusalem, Israel
<br/>

> ⚠️ This is an **unofficial** and **reverse-engineered** documentation project. It is not affiliated with the original authors or organizations behind FLUX.1.

<br>

If you find this work useful, please cite our [technical report](https://arxiv.org/abs/2507.09595) and [give this repo a star ⭐](https://github.com/OrGreenberg/DemystifyingFlux/stargazers)
<br>


![image](assets/Picture1.png)
<br>

---
<br>
## 📄 Abstract

*FLUX.1* is a diffusion-based text-to-image generation model developed by Black Forest Labs, designed to achieve faithful text-image alignment while maintaining high image quality and diversity. FLUX is considered state-of-the-art in text-to-image generation, outperforming popular models such as Midjourney, DALL·E 3, Stable Diffusion 3 (SD3), and SDXL. Although publicly available as open source, the authors have not released official technical documentation detailing the model’s architecture or training setup. This report summarizes an extensive reverse-engineering effort aimed at demystifying FLUX’s architecture directly from its source code, to support its adoption as a backbone for future research and development. This document is an *unofficial* technical report and is **not published or endorsed by the original developers or their affiliated institutions**

---
</div>

## 📋 This blog is organized into three main sections:

- [1. Background](/DemystifyingFlux/background/): provides essential high-level context
- [2. FLUX.1 Architecture](/DemystifyingFlux/flux-architecture/): explores the architecture of FLUX.1 in detail
- [3. Models and Tools](/DemystifyingFlux/models-and-tools/): gives a brief overview of the different models in the FLUX family

---
<br>
## 💬 Contribute to the blog!
- Please report issues or point at new FLUX releases in the blog's [GitHub issues](https://github.com/OrGreenberg/DemystifyingFlux/issues) page.
- Do you want to discuss FLUX? Ask question? You can do that in the blog's [discussion](https://github.com/OrGreenberg/DemystifyingFlux/issues) page.

---
<br>
## 📝 Citation

```bibtex
@article{greenebrg2025demystifying,
    title     = {Demystifying Flux Architecture}, 
    author    = {Or Greenebrg},
    eprint    = {2507.09595},
    archivePrefix      = {2025},
    booktitle = {arXiv}
}
```
