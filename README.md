# <div align="center">STRAP: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization</div>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2505.19547/)
[![GitHub stars](https://img.shields.io/github/stars/HoweyZ/STRAP?style=social)](https://github.com/HoweyZ/STRAP)

**Spatio-Temporal Pattern Retrieval Framework for Out-of-Distribution Generalization**

[📄 Paper](https://arxiv.org/abs/2505.19547) | [📊 Datasets](https://drive.google.com/drive/folders/1OiMLuFBdc56CLekileRjH0xyhDWuoC6C)

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🏗️ Repository Structure](#️-repository-structure)
- [🚀 Getting Started](#-getting-started)
- [🙏 Acknowledgements](#-acknowledgements)
- [📝 Citation](#-citation)
- [📜 License](#-license)

---

## ✨ Overview

**STRAP** is a novel spatio-temporal pattern retrieval framework designed to tackle the challenging problem of **out-of-distribution (OOD) generalization** in spatio-temporal forecasting tasks.

---

## 🏗️ Repository Structure

STRAP/
│
├── 📄 README.md                    # Project documentation
├── 📄 LICENSE                      # Apache 2.0 License
├── 📄 environment.yaml             # Conda environment configuration
├── 🚀 main.py                      # Main entry point for experiments
├── 🚀 stkec_main.py               # STKEC experiments entry point
├── 📜 run.sh                       # Batch experiment execution script
│
├── 📁 conf/                        # ⚙️ Configuration files
│   ├── AIR/                       # Air quality dataset configs
│   ├── ENERGY-Wind/               # Wind energy dataset configs
│   └── PEMS/                      # Traffic dataset configs
│       ├── strap.json
│       ├── ewc.json
│       └── ...
│
├── 📁 src/                         # 💻 Source code
│   ├── dataer/                    # Data loading and preprocessing
│   │   ├── ...
│   │
│   ├── model/                     # Model implementations
│   │   ├── ...             # Model components
│   │
│   └── trainer/                   # Training and evaluation
│       ├── ...
│
├── 📁 utils/                       # 🛠️ Utility functions
│   ├── ...
│
├── 📁 font/                        # Font files for visualization
├── 📁 log/                         # 📊 Training logs and checkpoints
└── 📁 data/                        # 💾 Dataset storage (create this)

---

## 🚀 Getting Started

### 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Conda** or **Miniconda** ([Download](https://www.anaconda.com/products/distribution))
- **NVIDIA GPU** with CUDA support (recommended)
- **Python 3.8+**

### 💻 Usage

```bash
# ENERGY-Wind, the same for other datasets.
bash run.sh
```

---

## 🙏 Acknowledgements

We would like to express our gratitude to:

- **EAC**: We thank the authors for their excellent work. Our implementation builds upon their codebase: [EAC Repository](https://github.com/Onedean/EAC)

---

## 📝 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{zhang2025strap,
  title={STRAP: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization},
  author={Zhang, Haoyu and Zhang, Wentao and Miao, Hao and Jiang, Xinke and Fang, Yuchen and Zhang, Yifan},
  journal={arXiv preprint arXiv:2505.19547},
  year={2025}
}
```
---

## 📜 License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.






