# model/ — Core Modeling Utilities

This directory contains the reusable core modules for scPlantFormer training, inference, and evaluation.

## Files

- `scplantFormer_model_mae.py`
  - Transformer backbone (`GPT`), attention blocks, training utilities, and seed/device helpers.
- `data_utils_mae.py`
  - Data sanitization, gene patch embedding, chunked loading, and on-disk cache helpers for pretraining.
- `infer_utils_mae.py`
  - Feature-space alignment helpers and streaming embedding inference.
- `eval_utils_mae.py`
  - Model builder, weight loading, embedding extraction, Sinkhorn transport, scIB metrics, and plot export tools.
- `model_config_mae.py`
  - Environment-driven configuration overrides and JSON config serialization.

## Typical Usage Pattern

1. Build or load model configuration (`model_config_mae.py`).
2. Prepare/patch data (`data_utils_mae.py`).
3. Train or load pretrained weights (`eval_utils_mae.py`).
4. Run embedding inference (`infer_utils_mae.py`).
5. Compute metrics and export figures (`eval_utils_mae.py`).

## Environment Variables (selected)

- `SCPLANT_MODEL_TYPE` (e.g., `gpt-nano`)
- `SCPLANT_N_LAYER`, `SCPLANT_N_HEAD`, `SCPLANT_N_EMBD`
- `SCPLANT_ACT` (`relu` or `gelu`)
- `SCPLANT_PRETRAIN_CACHE` (`1/0`)
- `SCPLANT_CACHE_DIR` (custom cache location)
- `SCPLANT_PRETRAIN_CHUNK` (chunk size for large matrices)
- `SCPLANT_DEVICE` (explicit device override)

## Notes for Manuscript-Grade Reproducibility

When reporting experiments, include:

- exact environment variable settings,
- model architecture and patching strategy (`gap`, number of patches),
- checkpoint hash/version and preprocessing signature,
- random seed strategy and number of repeated runs.
