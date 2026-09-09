# <div align="center">STRAP: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization</div>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2505.19547/)
[![GitHub stars](https://img.shields.io/github/stars/HoweyZ/STRAP?style=social)](https://github.com/HoweyZ/STRAP)

[📄 Paper](https://arxiv.org/abs/2505.19547) | [📊 Datasets](https://drive.google.com/drive/folders/1OiMLuFBdc56CLekileRjH0xyhDWuoC6C)

</div>

---

## 📋 Table of Contents

- [✨ Overview](#-overview)
- [🏗️ Repository Structure](#️-repository-structure)
- [🚀 Getting Started](#-getting-started)
- [🙏 Acknowledgements](#-acknowledgements)
- [📝 Citation](#-citation)
- [🌟 Star History](#-star-history)

---

## ✨ Overview

Spatio-Temporal Graph Neural Networks (STGNNs) have emerged as a powerful tool for modeling dynamic graph-structured data across diverse domains. However, they often fail to generalize in Spatio-Temporal Out-of-Distribution (STOOD) scenarios, where both temporal dynamics and spatial structures evolve beyond the training distribution. To address this problem, we propose STRAP, which enhances model generalization by integrating retrieval-augmented learning into the STGNN continue learning pipeline. Extensive experiments across multiple real-world streaming graph datasets show that \methodname consistently outperforms state-of-the-art STGNN baselines on STOOD tasks, demonstrating its robustness, adaptability, and strong generalization capability without task-specific fine-tuning.
<div align="center">
  <img src="./framework.jpg" alt="STRAP Framework" width="800"/>
  <p><em>STRAP Framework Architecture</em></p>
</div>


---

## 🏗️ Repository Structure
```
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
```
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

## STRAP implementation and runtime

`main.py` uses `RAP_Model` in `src/model/model.py`. RAP passes its actual
STGNN/DCRNN/ASTGNN/TGCN backbone to `PatternExtractor` in
`src/model/pattern_features.py`; `src/model/STRAP.py` owns the three pattern
libraries, paired history, retrieval and learned fusion.

| Library | Key construction | Value construction |
| --- | --- | --- |
| Spatial | Degree groups and modularity communities: signal moments, degree statistics and graph structure | Mean backbone output on the same subgraph |
| Spatial | Negative/zero/positive node curvature, high-flow/fluctuation/bottleneck edge curvature, degree and 16-dimensional topology descriptor | Backbone neighborhood-center features; endpoint means for edges |
| Temporal | Each node's k-hop neighborhood: signal/graph statistics and level-2 db4 coefficient moments over consecutive samples | Mean backbone output on that neighborhood |
| Temporal | Level-4 db4 coefficient magnitudes for each node's input window | Backbone output for that actual input on a singleton graph |
| Spatiotemporal | Spectral clusters in overlapping time windows, with the same statistical and wavelet descriptors | Mean backbone output on the same cluster/window |
| Spatiotemporal | Top spatial–temporal pairs by cosine affinity, followed by pooled key outer products | Pooled outer products of the corresponding values |

Topology includes degree, clustering, closeness, betweenness, eigenvector
centrality and neighborhood degree statistics. It covers every connected
component. The topology embedding contributes to key matching and is trained
through retrieval weights. Node and edge curvature retain the original STRAP
formulas. db4 uses symmetric boundary extension, including short input windows.
No key constructor substitutes zeros or random features after an error.

`initialize_training_patterns` in `src/trainer/engine.py` builds each year's
libraries before training or EWC. It uses the latest consecutive training
windows (`pattern_build_samples`, default `batch_size`), reverses the repository's
newest-first dataset order into chronological order, and builds on the full
graph even in incremental years. Skipping parameter training for a year with no
new nodes still initializes that year's library. Prediction never builds a
library from validation/test data. Direct RAP users must call
`initialize_patterns(training_data, adj)` with chronological data before the
first prediction, or load a saved year with `set_year(year)`.

History is sampled by inverse key norm, with age decay and an adaptive merge
ratio. Every sampled index selects its key, value and topology together. Each
archive contains only its own year's selected current patterns, avoiding repeated
accumulation of ancestors. `history_ratio` defaults to 0.3; `history_limit`
defaults to 10000 per library. Rebuilding a year creates a new version.

- Active keys, distinct values and topology are model-device buffers. The same
  fixed Rademacher projection transforms queries and keys. Exact Euclidean
  search, per-library candidate selection, global top-k and distance-weighted
  aggregation stay on the device. Feature and retrieved-pattern embeddings
  are combined with `fusion_weight` (default 0.7).
- `use_spatial_lib`, `use_temporal_lib` and `use_spatiotemporal_lib` default to
  true. Their corresponding `*_dropout` settings apply during training. Cross
  patterns require all three libraries; independent spectral patterns require
  only the spatiotemporal library. `return_pattern_or_value` selects the
  retrieved payload and defaults to `value`.
- `k_neighbors` defaults to 50. Per-library retrieval counts retain defaults
  of 50000/50000/100000. Query/key search chunks default to 128/4096 and can be
  set with `retrieval_batch_size`/`retrieval_key_batch_size`. Gradients are
  computed only for selected distances, avoiding a dense retrieval autograd
  graph. There is no special bypass for batches larger than 256.
- `temporal_k_hop` defaults to 2; `max_neighbors` to 20; `max_cluster_nodes` to
  100. `time_window`/`time_overlap` default to 12/6. `spatial_clusters` defaults
  to automatic selection. `cross_pattern_count` and `curvature_pattern_count`
  default to 2000 (the latter per curvature category). Spatial groups retain
  a minimum size of 5; spectral groups retain a minimum size of 3. Counts per
  constructor are logged and saved as `source_counts`, including empty groups.
- Graph partitions, centralities and shortest paths use a single CPU graph
  snapshot at build time. Indices and structural descriptors are uploaded once
  and reused. Wavelets, numerical moments, curvature, outer products, sampling
  and retrieval use Torch on the input device. CPU graph preprocessing and
  library disk I/O remain explicit boundaries.

The full method stores versioned yearly `.pt` files and metadata in
`pattern_libraries/strap_full_v1/`. The previous simplified implementation's
caches and checkpoints are incompatible: start a new training run to generate
full-method artifacts. Loading remains strict; missing prediction libraries,
invalid configuration and extraction errors are reported directly.

Select CUDA with `--gpuid N`, or CPU explicitly with `--gpuid -1`. Tensor metrics,
PECPM history and drift histograms also remain on the selected device. Dataset
batches enter from host memory; CUDA loaders use pinning and non-blocking copies.
No additional test files are shipped. Full-dataset accuracy and GPU throughput
must be measured with the external datasets on a CUDA machine.

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

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HoweyZ/STRAP&type=date&legend=top-left)](https://www.star-history.com/#HoweyZ/STRAP&type=date&legend=top-left)

---





