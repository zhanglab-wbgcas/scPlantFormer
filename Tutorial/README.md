# Tutorial Guide

This folder contains runnable examples for typical single-cell integration and annotation scenarios.

## Notebooks
- `Integration_batch.ipynb`: integration across experimental batches.
- `Integration_species.ipynb`: integration across species.
- `Integration_tissues.ipynb`: integration across tissues.
- `Integration_scRNA_scATAC.ipynb`: integration across RNA and ATAC modalities.

## Scripts
- `cross_dataset_cell-type_annoatation.py`: cross-dataset annotation workflow.
- `inner_cell_type_annotation.py`: in-dataset/inner-type annotation workflow.

## Recommended reporting
For publication-grade experiments, report:
- exact train/validation/test split strategy,
- random seed list,
- preprocessing and filtering criteria,
- metric definitions and confidence intervals.
