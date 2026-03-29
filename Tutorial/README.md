# Tutorial/ — Reproducible Analysis Workflows

This folder provides practical workflows for common integration and annotation settings.

## Notebooks

- `Integration_batch.ipynb`  
  Integrates datasets from different experimental batches.

- `Integration_tissues.ipynb`  
  Integrates cells across tissue contexts.

- `Integration_species.ipynb`  
  Cross-species transfer/integration workflow.

- `Integration_scRNA_scATAC.ipynb`  
  Cross-modality integration between transcriptomic and chromatin accessibility profiles.

## Python Scripts

- `cross_dataset_cell-type_annoatation.py`  
  Cross-dataset cell-type annotation pipeline.

- `inner_cell_type_annotation.py`  
  Intra-dataset (inner) cell-type annotation utilities.

## Suggested Use Order

1. Start with `Integration_batch.ipynb` to validate your environment.
2. Move to `Integration_tissues.ipynb` or `Integration_species.ipynb` based on your biological question.
3. Use `Integration_scRNA_scATAC.ipynb` for multimodal benchmarks.
4. Convert stable notebook settings into script-based runs for large-scale or repeated experiments.

## Reporting Checklist for Publication-Quality Results

For each tutorial-derived figure/table, record:

- notebook/script filename and commit hash,
- input dataset IDs and sample counts,
- preprocessing options and feature alignment settings,
- embedding dimensionality and model hyperparameters,
- all quantitative metrics and plotting parameters.
