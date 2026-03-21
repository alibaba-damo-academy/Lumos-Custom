# ✦ Lumos-Custom ✦

<p align="center">
  <sub>◇ ─── ◈ ─── ◇</sub><br/>
  <b>Personalized video · Physically plausible relighting · Open research code</b><br/>
  <sub>✧ · ꕤ · ✧</sub>
</p>

This repository collects open-source research from **DAMO Academy, Alibaba Group** and academic partners in **personalized video generation** and **physically plausible relighting**. Code is organized into self-contained subprojects for separate setup and reproduction.

## 🔗 📜 News

**If you are interested in our foundational video generation research, please refer to the [Lumos](https://github.com/alibaba-damo-academy/Lumos) project.**

### UniLumos

**[2025/9/19]** Accepted by [NeurIPS 2025](https://openreview.net/forum?id=e9B2NPQanB) !

**[2025/10/29]** Code is available now!

### LumosX

**[2026/1/26]** Accepted by [ICLR 2026](https://iclr.cc/Conferences/2026) !

**[2026/3/21]** Code is available now!

---

## Overview

| Project | Venue | In one sentence | Code & docs |
|--------|-------|-----------------|-------------|
| **LumosX** | **ICLR 2026** | **LumosX** advances personalized multi-subject video generation through relational data design and relational attention modeling. | [`LumosX/`](LumosX/) · [README](LumosX/README.md) |
| **UniLumos** | **NeurIPS 2025** | **UniLumos** advances unified image and video relighting through RGB-space geometry feedback on a flow-matching backbone. | [`UniLumos/`](UniLumos/) · [README](UniLumos/README.md) |

---

## ◆ LumosX ◆

<p align="center"><sub>✦ ICLR 2026 ✦</sub></p>

**LumosX: Relate Any Identities with Their Attributes for Personalized Video Generation**

### Showcase ✧ animated

<p align="center"><i>Identity-consistent · Subject-consistent personalized generation</i></p>

<table>
  <tr>
    <td align="center" width="50%"><b>✧ Identity consistency</b></td>
    <td align="center" width="50%"><b>✧ Subject consistency</b></td>
  </tr>
  <tr>
    <td align="center"><img src="LumosX/asserts/videos/Identity-Consistent%20Video%20Generation/LumosX/video_person2_101_1_sp1_480x832.gif" alt="LumosX identity-consistent demo" width="95%"/></td>
    <td align="center"><img src="LumosX/asserts/videos/Subject-Consistent%20Video%20Generation/LumosX/video_person2_001_sp1_480x832.gif" alt="LumosX subject-consistent demo" width="95%"/></td>
  </tr>
</table>

<sub>➜ Representative results from <code>LumosX/asserts/videos/</code> · more demos in <a href="LumosX/README.md">LumosX/README.md</a></sub>

- **Venue:** **ICLR 2026**
- **Summary:** We propose **LumosX**, a framework that advances both data and model design for personalized video generation. The data pipeline builds relational structure from captions and MLLM-derived priors; the model uses Relational Self-Attention and Relational Cross-Attention to encode subject–attribute dependencies. Companion evaluation resources live under **`LumosX/benchmark/`**.

**Quick links**

- **Model weights:** [Hugging Face · LumosX](https://huggingface.co/Alibaba-DAMO-Academy/LumosX)
- **Documentation:** [LumosX/README.md](LumosX/README.md) — installation, checkpoints, inference, and benchmark evaluation

---

## ◆ UniLumos ◆

<p align="center"><sub>✦ NeurIPS 2025 ✦</sub></p>

**UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback**

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
└── UniLumos/                 # NeurIPS 2025 · unified relighting + LumosBench/
    ├── README.md
    └── LumosBench/
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
```

---

## ✶ Citation ✶

If you use either project, please cite the corresponding paper. BibTeX entries are in the **Citation** section of each subproject `README.md`.

---

## ◈ Related work ◈

- Foundational video generation: **[Lumos](https://github.com/alibaba-damo-academy/Lumos)** (also referenced from the UniLumos documentation).
