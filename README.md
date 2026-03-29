# scPlantFormer: Foundation-Style Representation Learning for Plant Single-Cell Omics Integration

> **Repository type**: research code + tutorials  
> **Primary use case**: cross-dataset cell-type annotation and integration for plant single-cell data  
> **Core modalities**: scRNA-seq, scATAC-seq

---

## 1. Project Overview

**scPlantFormer** is a Transformer-based toolkit for plant single-cell data analysis, focusing on:

1. **Cross-dataset cell-type annotation** under domain shifts (batch, tissue, species).
2. **Representation learning / pretraining-style workflows** for robust downstream transfer.
3. **Multi-modal integration** across scRNA-seq and scATAC-seq contexts.

The repository combines model utilities (`model/`) and runnable tutorials (`Tutorial/`) for practical experiments and benchmarking.

---

## 2. Scientific Positioning and Intended Contributions

This repository is structured to support manuscript-grade computational experiments in:

- **Cell identity transfer across heterogeneous datasets**.
- **Biological domain adaptation** for species and tissue shifts.
- **Modality bridging** between gene-expression and chromatin-accessibility profiles.

Potential manuscript-facing claims that can be investigated with this codebase:

- Improved annotation consistency under cross-study variation.
- Better latent representations for downstream classification.
- A practical workflow for plant-specific single-cell integration tasks.

> ⚠️ Please treat performance claims as **experiment-dependent**. Report metrics only after reproducing them on your own benchmark splits.

---

## 3. Repository Layout

```text
scPlantFormer/
├── README.md
├── README.zh-CN.md
├── model/
│   ├── model_config_mae.py
│   ├── data_utils_mae.py
│   ├── infer_utils_mae.py
│   ├── eval_utils_mae.py
│   └── scplantFormer_model_mae.py
├── Tutorial/
│   ├── Integration_batch.ipynb
│   ├── Integration_species.ipynb
│   ├── Integration_tissues.ipynb
│   ├── Integration_scRNA_scATAC.ipynb
│   ├── cross_dataset_cell-type_annoatation.py
│   └── inner_cell_type_annotation.py
└── Baseline_scLLM/
    └── scLLM
```

### Folder purpose

- **`model/`**: model architecture and utility modules (configuration, data handling, inference, evaluation).
- **`Tutorial/`**: notebooks/scripts for common scenarios (batch/species/tissue/modality integration, annotation tasks).
- **`Baseline_scLLM/`**: baseline-related artifact placeholder.

---

## 4. Environment and Dependencies

Recommended Python version: **3.9+**

Install core dependencies:

```bash
pip install torch scanpy scikit-learn geomloss tqdm numpy pandas
```

Optional best practices for reproducibility:

- Use a dedicated virtual environment (Conda/venv).
- Pin package versions in your own manuscript branch.
- Log package versions with `pip freeze > requirements.lock.txt`.

---

## 5. Quick Start (Reproducible Workflow)

### Step A — Prepare data

Prepare single-cell matrices and metadata in Scanpy-compatible formats (e.g., AnnData/h5ad), with consistent cell-type labels and train/test partition strategy.

### Step B — Select an experiment track

Use one of the tutorial entry points:

- Batch integration: `Tutorial/Integration_batch.ipynb`
- Cross-species integration: `Tutorial/Integration_species.ipynb`
- Cross-tissue integration: `Tutorial/Integration_tissues.ipynb`
- RNA–ATAC integration: `Tutorial/Integration_scRNA_scATAC.ipynb`
- Scripted annotation workflows:
  - `Tutorial/cross_dataset_cell-type_annoatation.py`
  - `Tutorial/inner_cell_type_annotation.py`

### Step C — Evaluate and report

Recommended manuscript-ready reporting items:

- Classification metrics: macro/micro F1, balanced accuracy, AUROC (if applicable).
- Integration quality: batch-mixing and biological conservation metrics.
- Stability: mean ± std over multiple random seeds.
- Runtime profile: wall-clock and GPU/CPU memory footprint.

---

## 6. Suggested Benchmarking Protocol (High-Impact Journal Style)

To align with high-standard publication expectations, include:

1. **Data transparency**: list all datasets, accession IDs, preprocessing filters.
2. **Strict split policy**: prevent information leakage across biological replicates.
3. **Baselines**: compare with classical ML and recent deep-learning methods.
4. **Ablation studies**: architecture, loss components, and modality effects.
5. **Robustness**: test across species/tissues/batches and low-label regimes.
6. **Statistical rigor**: confidence intervals or paired statistical tests.
7. **Interpretability**: analyze marker genes / attention patterns when possible.

---

## 7. Reproducibility Checklist

- [ ] Fixed random seeds and recorded all seed values.
- [ ] Logged software versions and hardware information.
- [ ] Shared preprocessing scripts and parameter settings.
- [ ] Saved train/validation/test splits.
- [ ] Reported mean ± std across repeated runs.
- [ ] Included failure cases and negative results where applicable.

---

## 8. Limitations

- Repository currently emphasizes workflow assets (scripts/notebooks) over packaging.
- No pinned dependency lockfile is provided by default.
- Dataset download and preprocessing pipelines may need adaptation for your local data schema.

---

## 9. Citation and Academic Use

If you use this repository in academic work:

1. Cite this codebase (GitHub URL + commit hash used in experiments).
2. Cite relevant baseline and dataset papers.
3. Report full experimental settings for reproducibility.

You may add a BibTeX entry once a formal paper or preprint is available.

---

## 10. Contact and Contributions

Contributions are welcome via issues and pull requests. For scientific contributions, please include:

- Problem statement and biological context.
- Reproducible script/notebook.
- Quantitative comparison with baselines.

