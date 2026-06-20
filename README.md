# ✦ Lumos-Custom ✦

<p align="center">◇ ─── ◈ ─── ◇</p>

<h3 align="center">Lumos-Custom Project: research for customized video generation in the Lumos Project.</h3>

<p align="center">✧ · ꕤ · ✧</p>

This repository collects open-source research from **DAMO Academy, Alibaba Group** in customized video generation, currently including works to **customize identities/attributes and lighting for videos**, and **reasoning-driven unified video generation**. Code is organized into self-contained subprojects for separate setup and reproduction.

## 🔗 📜 News

### UniLumos

**[2025/9/19]** Accepted by [NeurIPS 2025](https://arxiv.org/abs/2511.01678) !

**[2025/10/29]** Code is available now!

### LumosX

**[2026/1/26]** Accepted by [ICLR 2026](https://arxiv.org/abs/2603.20192) !

**[2026/3/21]** Code is available now!

### Lumos-Nexus

**[2026/6/1]** Accepted by [ECCV 2026](https://arxiv.org/abs/2605.31603).

**[2026/5/31]** Code is available now!

**If you are interested in our foundational video generation research, please refer to the [Lumos](https://github.com/alibaba-damo-academy/Lumos) project.**

---

## Overview

| Project | Venue | In one sentence | Code & docs |
|--------|-------|-----------------|-------------|
| **Lumos-Nexus** | **arXiv preprint** | **Lumos-Nexus** advances reasoning-driven unified video generation through lightweight connector training and Unified Progressive Frequency Bridging at inference. | [`Lumos-Nexus/`](Lumos-Nexus/) · [README](Lumos-Nexus/README.md) |
| **LumosX** | **ICLR 2026** | **LumosX** advances personalized multi-subject video generation through relational data design and relational attention modeling. | [`LumosX/`](LumosX/) · [README](LumosX/README.md) |
| **UniLumos** | **NeurIPS 2025** | **UniLumos** advances unified image and video relighting through RGB-space geometry feedback on a flow-matching backbone. | [`UniLumos/`](UniLumos/) · [README](UniLumos/README.md) |
---
## ◆ Lumos-Nexus ◆

<p align="center"><span style="font-size: 1.2em;">✦ arXiv preprint ✦</span></p>

**Lumos-Nexus: Efficient Frequency Bridging with Homogeneous Latent Space for Video Unified Models**

[![arXiv](https://img.shields.io/badge/arXiv-2605.31603-b31b1b.svg)](https://arxiv.org/abs/2605.31603)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/alibaba-damo-academy/Lumos-Custom)
[![Project Page](https://img.shields.io/badge/Project-Page-purple)](https://jiazheng-xing.github.io/nexus-lumos-home/)

### Showcase ✧ animated

<p align="center"><i>Reasoning-driven video generation · VR-Bench tasks</i></p>

<p align="center"><b style="font-size: 1.15em;">✧ High-level physical world reasoning (ETV)</b></p>

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="70%">
      <b>Task prompt</b><br/>
      <img src="Lumos-Nexus/asserts/readme_png/etv_task.png" alt="ETV task prompt" width="100%" style="max-width: 560px;"/>
    </td>
    <td align="center" valign="middle" width="30%">
      <b>Lumos-Nexus</b><br/>
      <img src="Lumos-Nexus/asserts/readme_gif/etv_lumos.gif" alt="ETV Lumos-Nexus result" width="100%" style="max-width: 210px;"/>
    </td>
  </tr>
</table>

---

<p align="center"><b style="font-size: 1.15em;">✧ High-level commonsense reasoning (CCR)</b></p>

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="70%">
      <b>Task prompt</b><br/>
      <img src="Lumos-Nexus/asserts/readme_png/ccr_task.png" alt="CCR task prompt" width="100%" style="max-width: 560px;"/>
    </td>
    <td align="center" valign="middle" width="30%">
      <b>Lumos-Nexus</b><br/>
      <img src="Lumos-Nexus/asserts/readme_gif/ccr_lumos.gif" alt="CCR Lumos-Nexus result" width="100%" style="max-width: 210px;"/>
    </td>
  </tr>
</table>

---

<p align="center"><b style="font-size: 1.15em;">✧ Embodied physical reasoning (BBR)</b></p>

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="70%">
      <b>Task prompt</b><br/>
      <img src="Lumos-Nexus/asserts/readme_png/bbr_task.png" alt="BBR task prompt" width="100%" style="max-width: 560px;"/>
    </td>
    <td align="center" valign="middle" width="30%">
      <b>Lumos-Nexus</b><br/>
      <img src="Lumos-Nexus/asserts/readme_gif/bbr_lumos.gif" alt="BBR Lumos-Nexus result" width="100%" style="max-width: 210px;"/>
    </td>
  </tr>
</table>

<sub>➜ Task prompts: <code>Lumos-Nexus/asserts/readme_png/</code> · Result GIFs: <code>Lumos-Nexus/asserts/readme_gif/</code> · more in <a href="Lumos-Nexus/README.md">Lumos-Nexus/README.md</a></sub>

- **Venue:** **arXiv preprint** ([2603.20192](https://arxiv.org/abs/2603.20192); not yet peer-reviewed)
- **Summary:** We propose **Lumos-Nexus**, a training-efficient unified video generation framework. A lightweight generator is aligned with the understanding block during training; at inference, **Unified Progressive Frequency Bridging (UPFB)** hands off to a high-capacity pretrained generator in a shared latent space for high-fidelity output. Companion evaluation is provided by **VR-Bench** (see **`Lumos-Nexus/vr_bench_eval/`**).

**Quick links**

- **Paper:** [arXiv](https://arxiv.org/abs/2603.20192) · [Project page](https://jiazheng-xing.github.io/nexus-lumos-home/)
- **Documentation:** [Lumos-Nexus/README.md](Lumos-Nexus/README.md) — installation, OmniVideo checkpoints, batch inference, and VR-Bench evaluation

---

## ◆ LumosX ◆

<p align="center"><span style="font-size: 1.2em;">✦ ICLR 2026 ✦</span></p>

**LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation**

[![arXiv](https://img.shields.io/badge/arXiv-2603.20192-b31b1b.svg)](https://arxiv.org/abs/2603.20192)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/alibaba-damo-academy/Lumos-Custom)
[![Project Page](https://img.shields.io/badge/Project-Page-purple)](https://jiazheng-xing.github.io/lumosx-home/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue)](https://huggingface.co/Alibaba-DAMO-Academy/LumosX)

### Showcase ✧ animated

<p align="center"><i>Identity-consistent · Subject-consistent personalized generation</i></p>

<p align="center"><b style="font-size: 1.15em;">✧ Identity consistency</b></p>

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="70%">
      <b>Reference</b><br/>
      <img src="LumosX/asserts/images/Identity-Consistent%20Video%20Generation/video_person2_101_1.png" alt="LumosX identity reference" width="100%" style="max-width: 560px;"/>
    </td>
    <td align="center" valign="middle" width="30%">
      <b>Result</b><br/>
      <img src="LumosX/asserts/videos/Identity-Consistent%20Video%20Generation/LumosX/video_person2_101_1_sp1_480x832.gif" alt="LumosX identity-consistent demo" width="100%" style="max-width: 210px;"/>
    </td>
  </tr>
</table>

---

<p align="center"><b style="font-size: 1.15em;">✧ Subject consistency</b></p>

<table width="100%">
  <tr>
    <td align="center" valign="middle" width="70%">
      <b>Reference</b><br/>
      <img src="LumosX/asserts/images/Subject-Consistent%20Video%20Generation/video_person2_001.png" alt="LumosX subject reference" width="100%" style="max-width: 560px;"/>
    </td>
    <td align="center" valign="middle" width="30%">
      <b>Result</b><br/>
      <img src="LumosX/asserts/videos/Subject-Consistent%20Video%20Generation/LumosX/video_person2_001_sp1_480x832.gif" alt="LumosX subject-consistent demo" width="100%" style="max-width: 210px;"/>
    </td>
  </tr>
</table>

<sub>➜ Reference: <code>LumosX/asserts/images/</code> · Result GIFs: <code>LumosX/asserts/videos/</code> · more in <a href="LumosX/README.md">LumosX/README.md</a></sub>

- **Venue:** **ICLR 2026**
- **Summary:** We propose **LumosX**, a framework that advances both data and model design for personalized video generation. The data pipeline builds relational structure from captions and MLLM-derived priors; the model uses Relational Self-Attention and Relational Cross-Attention to encode subject–attribute dependencies. Companion evaluation resources live under **`LumosX/benchmark/`**.

**Quick links**

- **Model weights:** [Hugging Face · LumosX](https://huggingface.co/Alibaba-DAMO-Academy/LumosX)
- **Documentation:** [LumosX/README.md](LumosX/README.md) — installation, checkpoints, inference, and benchmark evaluation

---

## ◆ UniLumos ◆

<p align="center"><span style="font-size: 1.2em;">✦ NeurIPS 2025 ✦</span></p>

**UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback**

[![arXiv](https://img.shields.io/badge/arXiv-2511.01678-b31b1b.svg)](https://arxiv.org/abs/2511.01678)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Alibaba-DAMO-Academy/UniLumos)
[![Github](https://img.shields.io/badge/GitHub-Code-blue)](https://github.com/alibaba-damo-academy/Lumos-Custom)

### Showcase ✧ animated

<p align="center"><i>Unified image &amp; video relighting · physics-plausible feedback</i></p>

<table>
  <tr>
    <td align="center" width="50%"><img src="UniLumos/assets/UniLumos_Girl_1.gif" alt="UniLumos relighting demo 1" width="95%"/></td>
    <td align="center" width="50%"><img src="UniLumos/assets/UniLumos_Girl_2.gif" alt="UniLumos relighting demo 2" width="95%"/></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="UniLumos/assets/UniLumos_Robot_3.gif" alt="UniLumos relighting demo 3" width="95%"/></td>
    <td align="center" width="50%"><img src="UniLumos/assets/UniLumos_Robot_4.gif" alt="UniLumos relighting demo 4" width="95%"/></td>
  </tr>
</table>

<sub>➜ Assets live under <code>UniLumos/assets/</code> (same as <a href="UniLumos/README.md">UniLumos/README.md</a>) · add the GIFs locally if the folder is empty</sub>

- **Venue:** **NeurIPS 2025**
- **Summary:** We propose **UniLumos**, a unified relighting framework for images and videos. Supervision uses depth and normal maps from model outputs to align lighting with scene geometry; path consistency learning keeps this effective under few-step training. Companion evaluation is provided by **LumosBench** (see **`UniLumos/LumosBench/`**).

**Quick links**

- **Model weights:** [Hugging Face · UniLumos](https://huggingface.co/Alibaba-DAMO-Academy/UniLumos)
- **Documentation:** [UniLumos/README.md](UniLumos/README.md) — installation, checkpoints, inference, and LumosBench evaluation

---

## ✧ Repository layout ✧

```
Lumos-Custom/
├── README.md                 # This file: umbrella overview
├── LumosX/                   # ICLR 2026 · personalized multi-subject video generation
│   └── README.md
├── UniLumos/                 # NeurIPS 2025 · unified relighting + LumosBench/
│   ├── README.md
│   └── LumosBench/
└── Lumos-Nexus/              # arXiv preprint · reasoning-driven unified video generation + VR-Bench
    ├── README.md
    └── vr_bench_eval/
```

---

## ➤ Clone and enter a subproject

```bash
git clone https://github.com/alibaba-damo-academy/Lumos-Custom.git
cd Lumos-Custom

# LumosX
cd LumosX
# Follow LumosX/README.md

# or UniLumos
cd ../UniLumos
# Follow UniLumos/README.md

# or Lumos-Nexus
cd ../Lumos-Nexus
# Follow Lumos-Nexus/README.md
```

---

## ✶ Citation ✶

If you use any project, please cite the corresponding paper. BibTeX entries are in the **Citation** section of each subproject `README.md`.
```bibtex
@inproceedings{UniLumos,
  title={UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback},
  author={Liu, Pengwei and Yuan, Hangjie and Dong, Bo and Xing, Jiazheng and Wang, Jinwang and Zhao, Rui and Chen, Weihua and Wang, Fan},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}

@inproceedings{LumosX,
  title={LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation},
  author={Xing, Jiazheng and Du, Fei and Yuan, Hangjie and Liu, Pengwei and Xu, Hongbin and Ci, Hai and Niu, Ruigang and Chen, Weihua and Wang, Fan and Liu, Yong},
  booktitle={The Fourteenth International Conference on Learning Representations}
}

@misc{xing2026lumosnexusefficientfrequencybridging,
      title={Lumos-Nexus: Efficient Frequency Bridging with Homogeneous Latent Space for Video Unified Models}, 
      author={Jiazheng Xing and Hangjie Yuan and Lingling Cai and Xinyu Liu and Yujie Wei and Fei Du and Hai Ci and Tao Feng and Jiasheng Tang and Weihua Chen and Fan Wang and Yong Liu},
      year={2026},
      eprint={2605.31603},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

---

## ◈ Related work ◈

- Foundational video generation: **[Lumos](https://github.com/alibaba-damo-academy/Lumos)**.
