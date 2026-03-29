# scPlantFormer

This project provides tools for cross-dataset cell-type annotation, pre-training models on multi-modal datasets (scRNA-seq, scATAC-seq), and integrating datasets across species, tissues, and experimental conditions. The repository includes Python scripts and Jupyter notebooks designed to facilitate the integration and analysis of large-scale biological data.

## Features
- Pre-training models on large-scale multi-modal datasets for improved cell-type prediction and annotation.
- Fine-grained cell-type annotation using custom attention-based deep learning models implemented in PyTorch.
- Integration notebooks for single-cell data across batches, species, tissues, and modalities (e.g., scRNA-seq and scATAC-seq).

## Project Structure

- `Pretrained_Model/`: Directory for storing models.
  - `Arabidopsis_all_Pretrained.pth`: Pre-trained model on all Arabidopsis scRNA-seq data.
  - `Flower_Pretrained.pth`: Pre-trained model on Arabidopsis flower scRNA-seq data.
  - `Leaf_Pretrained.pth`: Pre-trained model on Arabidopsis leaf scRNA-seq data.
  - `Root_Pretrained.pth`: Pre-trained model on Arabidopsis root scRNA-seq data.
  - `seed_Pretrained.pth`: Pre-trained model on Arabidopsis seed scRNA-seq data.

- `Tutorial/`: Notebooks and scripts for data integration tasks.
  - `Integration_batch.ipynb`: Integration of datasets across different experimental batches.
  - `Integration_scRNA_scATAC.ipynb`: Integration of scRNA-seq and scATAC-seq datasets.
  - `Integration_species.ipynb`: Integration across different species.
  - `Integration_tissues.ipynb`: Integration across different tissues.
  - `cross_dataset_cell-type_annoatation.py`: Cross-dataset cell-type annotation pipeline using machine learning classifiers.
  - `inner_cell_type_annotation.py`: Deep-learning-based inner cell-type annotation using attention layers.

- `model/`: Core model and utility modules.
  - `data_utils_mae.py`
  - `model_config_mae.py`
  - `scplantFormer_model_mae.py`
  - `infer_utils_mae.py`
  - `eval_utils_mae.py`

## Documentation

This repository includes Read the Docs / MkDocs configuration:
- `.readthedocs.yaml`
- `mkdocs.yml`
- `docs/index.md`

## Requirements

```txt
torch
scanpy
scikit-learn
geomloss
tqdm
numpy
pandas
```

## Build Documentation Locally

```bash
pip install -r docs/requirements.txt
mkdocs build --strict
```
