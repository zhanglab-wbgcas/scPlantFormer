# scPlantFormer

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#installation)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](#method-overview)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

scPlantFormer is a **transformer-based foundation workflow for plant single-cell multi-omics integration**.  
It is designed for:

- cross-dataset and cross-batch integration,
- cross-species transfer learning,
- cross-modality analysis (e.g., scRNA-seq ↔ scATAC-seq),
- cell-type representation learning for downstream annotation.

This repository currently provides core model utilities under `model/` and reproducible tutorials under `Tutorial/` for typical benchmarking and biological use cases.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Method Overview](#method-overview)
3. [Repository Layout](#repository-layout)
4. [Installation](#installation)
5. [Data Requirements](#data-requirements)
6. [Quick Start](#quick-start)
7. [Reproducibility & Reporting Checklist](#reproducibility--reporting-checklist)
8. [Recommended Figure/Result Exports](#recommended-figureresult-exports)
9. [Limitations](#limitations)
10. [Citation](#citation)
11. [Contact](#contact)

---

## Highlights

- **Transformer encoder backbone** for patchified gene expression inputs.
- **Configurable architecture** via environment variables and JSON-based model configuration helpers.
- **Data alignment utilities** for harmonizing query datasets to reference gene spaces.
- **Evaluation-ready functions** for embedding extraction, integration metrics, and scanpy visualizations.
- **Tutorial assets** that cover batch/tissue/species/modality integration scenarios.

---

## Method Overview

The core workflow is:

1. **Input standardization**: normalize and align gene features across datasets.
2. **Gene patch embedding**: split high-dimensional cell vectors into fixed-size patches (`gap`).
3. **Transformer representation learning**: train/fine-tune GPT-style blocks on patch sequences.
4. **Embedding extraction**: produce low-dimensional embeddings for each cell.
5. **Integration & evaluation**: optionally apply Sinkhorn transport and compute scIB/scanpy-based metrics.

This setup supports both **representation learning** and **transfer annotation** pipelines.

---

## Repository Layout

```text
scPlantFormer/
├── model/
│   ├── scplantFormer_model_mae.py   # Model, trainer, attention blocks
│   ├── data_utils_mae.py            # Data loading, patching, caching, preprocessing
│   ├── infer_utils_mae.py           # Inference-time alignment and embedding stream
│   ├── eval_utils_mae.py            # Metrics and visualization helpers
│   └── model_config_mae.py          # Config loading/saving and env overrides
├── Tutorial/
│   ├── Integration_batch.ipynb
│   ├── Integration_tissues.ipynb
│   ├── Integration_species.ipynb
│   ├── Integration_scRNA_scATAC.ipynb
│   ├── cross_dataset_cell-type_annoatation.py
│   └── inner_cell_type_annotation.py
└── Baseline_scLLM/
```

For details on scripts and notebooks, see:

- [`model/README.md`](./model/README.md)
- [`Tutorial/README.md`](./Tutorial/README.md)

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scanpy torch scikit-learn tqdm geomloss pot
```

Optional packages for full visualization/export support:

```bash
pip install pillow imageio
```

---

## Data Requirements

Expected primary data format: **`.h5ad` (AnnData)**.

Recommended fields:

- `adata.X`: expression or accessibility matrix (dense or sparse)
- `adata.obs['Celltype']` (or equivalent): cell labels for supervised evaluation
- `adata.obs['Condition']` (optional): batch/domain metadata
- `adata.var_names`: gene identifiers (preferably unique)

For transfer/inference, ensure the query data is aligned to either:

- a reference dataset’s feature space, or
- a curated gene order list (`gene_order.txt`).

---

## Quick Start

### 1) Set reproducibility-related environment variables

```bash
export PYTHONHASHSEED=0
export SCPLANT_MODEL_TYPE=gpt-nano
export SCPLANT_ACT=relu
export SCPLANT_PRETRAIN_CACHE=1
```

### 2) Run a tutorial workflow

- Open one of the notebooks in `Tutorial/` for end-to-end examples.
- For script-style experiments, start with `Tutorial/cross_dataset_cell-type_annoatation.py`.

### 3) Use model utilities in Python

```python
from model.eval_utils_mae import build_model
from model.infer_utils_mae import align_for_inference
```

---

## Reproducibility & Reporting Checklist

To match standards commonly expected by high-impact journals, report at minimum:

- dataset accession IDs and preprocessing protocol,
- QC thresholds and filtering criteria,
- model architecture (`n_layer`, `n_head`, `n_embd`, `gap`),
- training hyperparameters (epochs, batch size, LR, dropout, seed),
- hardware (GPU model, VRAM, CPU threads),
- software versions (`python`, `torch`, `scanpy`, `numpy`),
- statistical summary over **multiple random seeds**,
- full metric definitions and confidence intervals where applicable.

Also archive:

- exact run scripts,
- environment lock file,
- generated embeddings and figure outputs.

---

## Recommended Figure/Result Exports

Use scanpy-based UMAP/t-SNE figures and metrics JSON for publication supplements:

- UMAP/t-SNE panels by cell type and condition,
- batch mixing and biological conservation scores,
- confusion matrices for annotation transfer,
- ablation studies over `gap`, model size, and activation type.

---

## Limitations

- Performance depends strongly on gene-space alignment quality.
- Domain shift across species/platforms may require re-tuning and stronger regularization.
- Current repository focuses on utilities/tutorials; a fully packaged CLI is not yet provided.

---

## Citation

If you use scPlantFormer in academic work, please cite this repository and any associated manuscript/preprint when available.

```bibtex
@software{scplantformer,
  title   = {scPlantFormer: Transformer-based integration for plant single-cell omics},
  author  = {scPlantFormer contributors},
  year    = {2026},
  url     = {https://github.com/<your-org>/scPlantFormer}
}
```

---

## Contact

For issues, feature requests, or collaboration questions, please open a GitHub Issue in this repository.
