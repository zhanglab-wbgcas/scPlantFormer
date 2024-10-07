# scPlantFormer

This project provides tools for cross-dataset cell-type annotation, pre-training models on multi-modal datasets (scRNA-seq, scATAC-seq), and integrating datasets across species, tissues, and experimental conditions. The repository includes a set of Python scripts and Jupyter notebooks designed to facilitate the integration and analysis of large-scale biological data.

## Features
- Pre-training models on large-scale multi-modal datasets for improved cell-type prediction and annotation.
- Fine-grained cell-type annotation using custom attention-based deep learning models implemented in PyTorch.
- Jupyter notebooks for integration of single-cell data across different batches, species, tissues, and modalities (e.g., scRNA-seq and scATAC-seq).

## Project Structure

- `Pretrained_Model/`: Directory for storing models.
  - `Arabidopsis_all_Pretrained.pth`: Pre-trained model on All Arabidopsis scRNA-seq data.
  - `Flower_Pretrained.pth`: Pre-trained model on All Arabidopsis'flower scRNA-seq data.
  - `Leaf_Pretrained.pth`: Pre-trained model on All Arabidopsis'leaf scRNA-seq data.
  - `Root_Pretrained.pth`: Pre-trained model on All Arabidopsis'root scRNA-seq data.
  - `seed_Pretrained.pth`: Pre-trained model on All Arabidopsis'seed scRNA-seq data.

- `Tutorial/`: Jupyter notebooks for various data integration tasks:
  - `Integration_batch.ipynb`: Integration of datasets across different experimental batches.
  - `Integration_scRNA_scATAC.ipynb`: Integration of single-cell RNA-seq (scRNA-seq) and single-cell ATAC-seq (scATAC-seq) data.
  - `Integration_species.ipynb`: Integration of single-cell data across different species.
  - `Integration_tissues.ipynb`: Integration of single-cell data across different tissues.
  - `cross_dataset_cell-type_annotation.py`: Script for cross-dataset cell-type annotation using machine learning models (e.g., Logistic Regression, Decision Trees, VotingClassifier, StackingClassifier). This script processes large-scale biological data, performs classification, and logs the results.
  - `inner_cell_type_annotation.py`: Implements deep learning models for inner cell-type annotation using attention-based layers (e.g., `CausalSelfAttention`). The script leverages PyTorch and is designed for neural network-based analysis of single-cell data, including configuration management and random seed setting.


- `model/`: Python scripts for cell-type annotation and model training:
  - `pre_train_all.py`: Script for pre-training models across different single-cell datasets, using custom neural network layers for embedding and processing the data. It is optimized for large-scale data, including plant-specific models such as `scPlantGPT_v1`.

## Installation

To set up the project and install the required packages:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/project_name.git
   cd project_name
