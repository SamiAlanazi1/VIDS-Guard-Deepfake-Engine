# VIDS-Guard Deepfake Engine

<p align="left">
  <a href="https://doi.org/10.1016/j.iswa.2026.200664">
    <img src="https://img.shields.io/badge/Published%20in-Information%20Sciences%20%26%20Applications-blue" />
  </a>
  <a href="https://doi.org/10.5281/zenodo.17362749">
    <img src="https://img.shields.io/badge/Dataset%20DOI-VIDS--Guard-green" />
  </a>

</p>

---

## 🔍 Overview

**VIDS-Guard Deepfake Engine** is a **forensics-aware video deepfake detection system** designed for **robust real-world deployment**.

The framework goes beyond semantic detection by explicitly modelling **forensic artefacts** across multiple domains:

- Spatial residual noise (SRM)
- Chromatic inconsistencies (YCbCr)
- Frequency anomalies (FFT)
- Temporal inconsistencies (Transformer)

---

## 🧠 Core Idea

> Deepfakes are better detected through **forensic inconsistencies**, not just visual semantics.

VIDS-Guard implements a **multi-stream fusion pipeline** that captures complementary manipulation traces often ignored by conventional CNN-based approaches.

---

## 🖼️ Architecture

<p align="center">
  <img src="docs/architecture.jpg" width="700"/>
</p>

---

## ⚙️ Features

- Multi-stream forensic learning
- Temporal transformer for video-level prediction
- Cross-dataset generalisation capability
- Robust to compression and unseen manipulations
- Modular and extensible pipeline

---

## 📦 Dataset

This project uses the **VIDS-Guard Dataset (v1.0)**:

- 26,975 videos
- Balanced real/fake distribution
- Aggregated from 11 datasets
- Includes unseen external test set

📥 Download from Zenodo:
- https://doi.org/10.5281/zenodo.17362749  
- https://doi.org/10.5281/zenodo.17382113  

---

## 🔗 Related Publication

**Robust Video Deepfake Detection via Forensics-Aware Multi-Stream Learning**  
*Information Sciences & Applications, 2026*

- 📄 Paper: https://doi.org/10.1016/j.iswa.2026.200664
---

## 🚀 Getting Started

### 1. Clone repo
```bash
git clone https://github.com/SamiAlanazi1/VIDS-Guard-Deepfake-Engine.git
cd VIDS-Guard-Deepfake-Engine
---


