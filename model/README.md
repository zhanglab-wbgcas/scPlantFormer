# Model Module Guide

This folder implements core components for the scPlantFormer workflow.

## Files
- `scplantFormer_model_mae.py`: model architecture definition.
- `model_config_mae.py`: centralized model/training configuration.
- `data_utils_mae.py`: data processing and loading helpers.
- `infer_utils_mae.py`: inference utilities.
- `eval_utils_mae.py`: evaluation utilities.

## Reproducibility tips
- Keep configuration snapshots for each run.
- Log model checkpoint hashes used in final figures/tables.
- Use multiple seeds and aggregate results.
