# FlowLess-R: Geometry of Forgetting in Continual Learning

This repository contains the official implementation of **FlowLess-R**, a representation-space regularization method for continual learning based on representation flux.

The repository includes:

- continual learning algorithms
- representation geometry analysis
- replay regularization
- statistical evaluation
- visualization utilities

Legacy development code has been moved into dedicated `*_legacy` folders and is **not used** by the current implementation or the paper.

---

## Overview

FlowLess-R is a lightweight representation-space regularizer for replay-based continual learning. Rather than constraining network parameters, FlowLess-R penalizes excessive displacement of latent representations stored in the replay buffer, encouraging stable representation dynamics while preserving the plasticity required to learn new tasks.


## Benchmarks

The repository contains experiments on

- SplitMNIST
- SplitFashionMNIST
- SplitCIFAR10
- SplitTinyImageNet

## Project Workflow

The repository is organized around a simple pipeline: benchmark datasets are loaded, neural network models are trained using the FlowLess-R continual learning algorithms, and the resulting checkpoints and logs are analyzed to generate the figures and statistics reported in the paper.

```text
                           ┌───────────────────────────┐
                           │     src/datasets/         │
                           │ SplitMNIST               │
                           │ SplitFashionMNIST        │
                           │ SplitCIFAR10            │
                           │ SplitTinyImageNet       │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │      src/models/          │
                           │ MLP / ResNet-18           │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │     src/flowless/         │
                           │ FlowLess-R Training       │
                           │ ER / DER++ / ER-ACE       │
                           │ Hyperparameter Sweeps     │
                           └─────────────┬─────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  │                                             │
                  ▼                                             ▼
     ┌───────────────────────────┐               ┌───────────────────────────┐
     │    general_stats.py       │               │    src/investigation/     │
     │ Statistical Analysis      │               │ Representation Geometry   │
     │ Significance Tests        │               │ Flux, Density, Confidence │
     └─────────────┬─────────────┘               └─────────────┬─────────────┘
                   │                                           │
                   └──────────────────────┬────────────────────┘
                                          ▼
                           ┌───────────────────────────┐
                           │      Paper Figures        │
                           │       Tables             │
                           │      Final Results       │
                           └───────────────────────────┘
```

# Quick Start

## Clone the repository

```bash
git clone https://github.com/maksimkazanskii/FlowLess-R.git
cd FlowLess-R
```

## Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running Experiments

### SplitMNIST, SplitFashionMNIST and SplitCIFAR10

```bash
python -m src.flowless.regularizer_grid
```

Example:

```bash
python -m src.flowless.regularizer_grid \
    --dataset cifar10 \
    --memory_sizes 400 \
    --seeds 0
```

---

### SplitTinyImageNet

```bash
python -m src.flowless.regularizer_grid_tiny
```

Example:

```bash
python -m src.flowless.regularizer_grid_tiny \
    --dataset tinyimagenet \
    --memory_sizes 800 \
    --seeds 0
```

---

### Representation Geometry Analysis

```bash
python -m src.investigation.forgetting_flux
```

---

### Statistical Analysis

```bash
python -m src.flowless.general_stats
```
# Repository Structure

```
src/
├── datasets/
├── flowless/
├── investigation/
├── legacy/
└── models/

```

---

## src/datasets

This package provides the continual learning benchmark datasets used throughout the project. These modules are imported by the experiment scripts (e.g., `regularizer_grid.py` and `regularizer_grid_tiny.py`) to construct task sequences and data loaders.

| File | Description |
|------|-------------|
| `base.py` | Defines the base continual learning dataset interface, including the common API shared by all benchmarks (task iteration, train/test loaders, task metadata, etc.). |
| `datasets.py` | Utility for constructing the requested continual learning benchmark from the experiment configuration or command-line arguments. |
| `split_mnist.py` | Implements the SplitMNIST benchmark by partitioning MNIST into sequential tasks and providing the corresponding continual learning data loaders. |
| `split_fashion_mnist.py` | Implements the SplitFashionMNIST benchmark using the same continual learning protocol as SplitMNIST. |
| `split_cifar10.py` | Implements the SplitCIFAR10 benchmark, including dataset loading, preprocessing, task construction, and continual learning data loaders. |
| `split_tiny_imagenet.py` | Implements the SplitTinyImageNet benchmark, including image preprocessing, task construction, and continual learning data loaders. |
---

## src/models

This package contains the neural network architectures used throughout the continual learning experiments. The models are instantiated by the experiment scripts depending on the selected benchmark and provide the latent representations used by the FlowLess-R regularizer.

| File | Description |
|------|-------------|
| `mlp.py` | Implements the multilayer perceptron (MLP) architecture used for the SplitMNIST and SplitFashionMNIST benchmarks. The final hidden layer serves as the latent representation on which FlowLess-R regularization is applied. |
| `resnet.py` | Implements the ResNet-18 backbone used for the SplitCIFAR10 and SplitTinyImageNet benchmarks. The 512-dimensional feature representation produced after global average pooling is used as the latent representation for FlowLess-R regularization. |
---

## src/flowless

This package contains the complete implementation of FlowLess-R, including replay-based continual learning algorithms, hyperparameter sweeps, statistical analyses, and utilities used to reproduce the experiments reported in the paper.

### Training

| File | Description |
|------|-------------|
| `regularizer_grid.py` | Main FlowLess-R experiments (MNIST, FashionMNIST, CIFAR10) |
| `regularizer_grid_tiny.py` | TinyImageNet experiments |
| `DER_plusplus.py` | DER++ implementation with FlowLess-R |
| `DER_plusplus_resnet.py` | DER++ for ResNet models |
| `ER_ACE.py` | ER-ACE with FlowLess-R |
| `ER_ACE_tiny.py` | ER-ACE TinyImageNet implementation |

### Analysis

| File | Description |
|------|-------------|
| `general_stats.py` | Aggregate experimental results and statistical tests |
| `buffer_sweep_plots.py` | Replay buffer size plots |
| `regularizer_grid_stats_alpha.py` | Density-weighting analysis |
| `regularizer_layers.py` | Layer ablation experiments |
| `regularizer_layers_stats.py` | Layer ablation statistics |

---

## src/investigation

This package contains the scripts used to analyze representation dynamics and generate the figures presented in the paper, including representation flux, representation density, confidence evolution, and forgetting analyses.

| File | Description |
|------|-------------|
| `forgetting_flux.py` | Representation flux analysis |
| `splitmnist_geometry_buffer.py` | SplitMNIST geometry |
| `splitfashionmnist_geometry_buffer.py` | SplitFashionMNIST geometry |
| `cifar10_geometry_buffer.py` | SplitCIFAR10 geometry |
| `tiny_geometry_buffer.py` | SplitTinyImageNet geometry |
| `clean_the_npy.py` | Utility for cleaning cached analysis files |

---



## Reproducibility

Unless stated otherwise, all experiments use fixed random seeds and follow the evaluation protocol described in the paper.

**Random seeds**

- **SplitMNIST:** `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- **SplitFashionMNIST:** `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- **SplitCIFAR10:** `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- **SplitTinyImageNet:** `0, 1, 2, 5, 6`

Unless otherwise specified, all reported results in the paper are presented as the mean ± standard deviation over these independent continual learning runs.

# Paper

**Geometry of Forgetting: Representation Flux in Continual Learning**

This repository accompanies the paper and provides

- the complete implementation of FlowLess-R,
- scripts to reproduce all experiments,
- representation geometry analyses,
- statistical evaluation,
- generation of all figures and tables reported in the paper.
## License

This project is released under the MIT License.

## Citation

If you find this repository useful, please cite

```bibtex
@article{kazanskii2026flowless,
  title={Geometry of Forgetting: Representation Flux in Continual Learning},
  author={Kazanskii, Maksim A.},
  year={2026}
}
```