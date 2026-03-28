from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import scipy.sparse as sp


LABEL_KEYS = ("Celltype", "CellType", "cell_type", "celltype")
CONDITION_KEYS = ("Condition", "condition")
DEFAULT_GENE_ORDER_PATH = (
    "/mnt/raid5/xujing/scmsPlant/01scPlant/scPlant/06Review/M10/type3_new/wgcna/gene_order.txt"
)


def align_to_reference(adata: sc.AnnData, ref_path: Path) -> Tuple[sc.AnnData, sc.AnnData]:
    ref = sc.read_h5ad(ref_path)
    adata.var_names_make_unique()
    if "features" in ref.var:
        feats = list(ref.var["features"].values)
    else:
        feats = list(ref.var_names.values)
    present = [f for f in feats if f in adata.var_names]
    missing = [f for f in feats if f not in adata.var_names]
    if not missing:
        adata = adata[:, feats]
        return adata, ref

    adata_present = adata[:, present]
    x = adata_present.X
    if not isinstance(x, np.ndarray):
        x = x.todense()
    x = np.asarray(x)
    out = np.zeros((adata.n_obs, len(feats)), dtype=x.dtype)
    feat_index = {f: i for i, f in enumerate(feats)}
    for j, feat in enumerate(present):
        out[:, feat_index[feat]] = x[:, j]
    adata_aligned = sc.AnnData(X=out, obs=adata.obs.copy(), var=pd.DataFrame(index=feats))
    return adata_aligned, ref


def align_to_gene_list(adata: sc.AnnData, gene_list: Iterable[str]) -> sc.AnnData:
    feats = list(gene_list)
    adata.var_names_make_unique()
    present = [f for f in feats if f in adata.var_names]
    missing = [f for f in feats if f not in adata.var_names]
    if not missing:
        return adata[:, feats]

    adata_present = adata[:, present]
    x = adata_present.X
    if not isinstance(x, np.ndarray):
        x = x.todense()
    x = np.asarray(x)
    out = np.zeros((adata.n_obs, len(feats)), dtype=x.dtype)
    feat_index = {f: i for i, f in enumerate(feats)}
    for j, feat in enumerate(present):
        out[:, feat_index[feat]] = x[:, j]
    return sc.AnnData(X=out, obs=adata.obs.copy(), var=pd.DataFrame(index=feats))


def load_gene_order(path: Path | None = None, env_key: str = "SCPLANT_GENE_ORDER") -> list[str]:
    if path is None:
        env_path = os.getenv(env_key, "").strip()
        path = Path(env_path) if env_path else Path(DEFAULT_GENE_ORDER_PATH)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing gene_order.txt: {path}")
    genes: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            g = line.strip()
            if g:
                genes.append(g)
    if not genes:
        raise ValueError(f"gene_order.txt is empty: {path}")
    return genes


def get_gene_list_from_ref(ref_path: Path) -> list[str]:
    ref = sc.read_h5ad(ref_path)
    if "features" in ref.var:
        return list(ref.var["features"].values)
    return list(ref.var_names.values)


def align_for_inference(
    adata: sc.AnnData,
    *,
    mode: str,
    ref_path: Path | None = None,
    gene_order: list[str] | None = None,
    gene_order_path: Path | None = None,
) -> tuple[sc.AnnData, list[str]]:
    if mode == "reference":
        if ref_path is None:
            raise ValueError("ref_path is required for mode='reference'")
        gene_list = get_gene_list_from_ref(ref_path)
    elif mode == "gene_order":
        gene_list = gene_order or load_gene_order(gene_order_path)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return align_to_gene_list(adata, gene_list), gene_list


def select_label_key(adata: sc.AnnData) -> str:
    for key in LABEL_KEYS:
        if key in adata.obs:
            return key
    raise KeyError(f"None of label keys found in obs: {LABEL_KEYS}")


def select_condition_key(adata: sc.AnnData) -> str | None:
    for key in CONDITION_KEYS:
        if key in adata.obs:
            return key
    return None


def gene_embedding_batch(x_batch: np.ndarray, gap: int) -> np.ndarray:
    single_cell_list = []
    for single_cell in x_batch:
        feature = []
        length = len(single_cell)
        for k in range(0, length, gap):
            if k + gap > length:
                a = single_cell[length - gap : length]
            else:
                a = single_cell[k : k + gap]
            feature.append(a)
        single_cell_list.append(np.asarray(feature))
    return np.asarray(single_cell_list)


def infer_embeddings_stream(
    model: torch.nn.Module,
    x: np.ndarray,
    gap: int,
    batch_cells: int = 128,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    emb_all = []
    for i in range(0, x.shape[0], batch_cells):
        x_batch = x[i : i + batch_cells]
        if sp.issparse(x_batch):
            x_batch = x_batch.toarray()
        x_patches = gene_embedding_batch(x_batch, gap)
        with torch.no_grad():
            x_t = torch.tensor(x_patches, dtype=torch.double, device=device)
            emb, _ = model.encode(x_t)
        emb_all.append(emb.cpu().numpy())
    return np.concatenate(emb_all, axis=0)


def compute_n_patches(n_features: int, gap: int) -> int:
    return int(math.ceil(n_features / gap))
