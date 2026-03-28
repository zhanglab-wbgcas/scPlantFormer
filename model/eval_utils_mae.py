from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from scplantFormer_model_mae import GPT, Trainer


def build_model(
    gap: int,
    n_patches: int,
    mod2_dim: int,
    dropout: float,
    h: int,
    loss1: float,
    model_type: str | None = "gpt-nano",
    n_layer: int | None = None,
    n_head: int | None = None,
    n_embd: int | None = None,
    act: str | None = "relu",
) -> GPT:
    model_config = GPT.get_default_config()
    if any(v is not None for v in (n_layer, n_head, n_embd)):
        if not all(v is not None for v in (n_layer, n_head, n_embd)):
            raise ValueError("n_layer, n_head, n_embd must be provided together.")
        model_config.model_type = None
        model_config.n_layer = n_layer
        model_config.n_head = n_head
        model_config.n_embd = n_embd
    else:
        model_config.model_type = model_type or "gpt-nano"
    model_config.vocab_size = gap
    model_config.block_size = n_patches
    if n_embd is None:
        model_config.n_embd = gap
    model_config.embd_pdrop = dropout
    model_config.resid_pdrop = dropout
    model_config.attn_pdrop = dropout
    model_config.loss1 = loss1
    model_config.h = h
    model_config.mod2_dim = mod2_dim
    model_config.act = (act or "relu").strip().lower()
    model = GPT(model_config)
    return model


def load_weights(model: GPT, weight_path: Path) -> None:
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)


def train_and_embed(
    model: GPT,
    x_patches: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
) -> np.ndarray:
    train_dataset = _make_dataset(x_patches, y)
    train_config = Trainer.get_default_config()
    train_config.epoch = epochs
    train_config.learning_rate = lr
    train_config.batch_size = batch_size
    trainer = Trainer(train_config, model, train_dataset)
    emb = trainer.run()
    return emb


def infer_embeddings(model: GPT, x_patches: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    emb_all: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, x_patches.shape[0], batch_size):
            x_batch = torch.tensor(x_patches[i : i + batch_size], dtype=torch.double, device=device)
            emb, _ = model.encode(x_batch)
            emb_all.append(emb.cpu().numpy())
    return np.concatenate(emb_all, axis=0)


def run_sinkhorn(
    emb: np.ndarray,
    batch_list: np.ndarray,
    reg: float,
) -> np.ndarray:
    import ot

    groups: Dict[str, np.ndarray] = {}
    for i, idx in enumerate(batch_list):
        if idx not in groups:
            groups[idx] = []
        groups[idx].append(emb[i])
    for k in groups:
        groups[k] = np.array(groups[k])

    unique_indices = list(groups.keys())
    counts = [len(groups[k]) for k in unique_indices]
    index_max = int(np.argmax(counts))

    def fit_transform_sinkhorn(xs, xt, reg_val=0.3):
        ot_sinkhorn = ot.da.LinearTransport(reg=reg_val)
        ot_sinkhorn.fit(Xs=xs, Xt=xt)
        return ot_sinkhorn.transform(Xs=xs)

    emb_max = groups[unique_indices[index_max]]
    for i, key in enumerate(unique_indices):
        if i != index_max:
            groups[key] = fit_transform_sinkhorn(groups[key], emb_max, reg_val=reg)

    emb_out = groups[unique_indices[0]]
    for key in unique_indices[1:]:
        emb_out = np.concatenate((emb_out, groups[key]), axis=0)
    return emb_out


def compute_scib_metrics(
    adata: sc.AnnData,
    embed_key: str,
    batch_key: str,
    label_key: str,
) -> Dict[str, float]:
    import scib

    sc.pp.neighbors(adata, use_rep=embed_key)
    metrics = scib.metrics.metrics_fast(adata, adata, batch_key, label_key, embed=embed_key)
    # scib.metrics.metrics_fast may return a (metrics x 1) DataFrame with metric names as index.
    # Downstream code expects a plain dict[str, float].
    if isinstance(metrics, pd.DataFrame):
        if metrics.shape[1] >= 1:
            return metrics.iloc[:, 0].to_dict()
        return {}
    if isinstance(metrics, pd.Series):
        return metrics.to_dict()
    return dict(metrics)


def save_metrics_json(path: Path, metrics: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _convert_png_to_jpg(png_path: Path) -> None:
    if not png_path.exists():
        return
    jpg_path = png_path.with_suffix(".jpg")
    try:
        from PIL import Image

        with Image.open(png_path) as img:
            rgb = img.convert("RGB")
            rgb.save(jpg_path, format="JPEG", quality=95)
    except Exception:
        try:
            import imageio.v2 as imageio

            img = imageio.imread(png_path)
            imageio.imwrite(jpg_path, img, quality=95)
        except Exception:
            pass


def _save_scanpy_plot(plot_fn, adata: sc.AnnData, color, save_stub: str) -> None:
    plot_fn(adata, color=color, wspace=1, save=f"{save_stub}.pdf")
    plot_fn(adata, color=color, wspace=1, save=f"{save_stub}.png")
    figdir = Path(sc.settings.figdir)
    for prefix in ("umap", "tsne"):
        png_path = figdir / f"{prefix}{save_stub}.png"
        _convert_png_to_jpg(png_path)


def save_umap_tsne(
    adata: sc.AnnData,
    embed_key: str,
    color_keys: Iterable[str],
    figdir: Path,
    prefix: str,
    color_name_map: Dict[str, str] | None = None,
) -> None:
    sc.settings.figdir = str(figdir)
    color_keys = list(color_keys)
    color_name_map = color_name_map or {}
    combined_colors = color_keys[:2] if len(color_keys) >= 2 else color_keys

    sc.pp.neighbors(adata, use_rep=embed_key)
    sc.tl.umap(adata)
    if combined_colors:
        _save_scanpy_plot(sc.pl.umap, adata, combined_colors, f"{prefix}_plot")
    for color in color_keys:
        suffix = color_name_map.get(color, color)
        _save_scanpy_plot(sc.pl.umap, adata, [color], f"{prefix}{suffix}_plot")

    if os.getenv("SCPLANT_SKIP_TSNE", "").lower() in {"1", "true", "yes"}:
        return

    sc.tl.tsne(adata, use_rep=embed_key)
    if combined_colors:
        _save_scanpy_plot(sc.pl.tsne, adata, combined_colors, f"{prefix}_tsne_plot")
    for color in color_keys:
        suffix = color_name_map.get(color, color)
        _save_scanpy_plot(sc.pl.tsne, adata, [color], f"{prefix}{suffix}_tsne_plot")


def save_tsne(
    adata: sc.AnnData,
    embed_key: str,
    color_keys: Iterable[str],
    figdir: Path,
    prefix: str,
    color_name_map: Dict[str, str] | None = None,
) -> None:
    if os.getenv("SCPLANT_SKIP_TSNE", "").lower() in {"1", "true", "yes"}:
        return
    sc.settings.figdir = str(figdir)
    color_keys = list(color_keys)
    color_name_map = color_name_map or {}
    combined_colors = color_keys[:2] if len(color_keys) >= 2 else color_keys

    sc.tl.tsne(adata, use_rep=embed_key)
    if combined_colors:
        _save_scanpy_plot(sc.pl.tsne, adata, combined_colors, f"{prefix}_tsne_plot")
    for color in color_keys:
        suffix = color_name_map.get(color, color)
        _save_scanpy_plot(sc.pl.tsne, adata, [color], f"{prefix}{suffix}_tsne_plot")


def ensure_categorical(adata: sc.AnnData, keys: Iterable[str]) -> None:
    for key in keys:
        if key not in adata.obs:
            continue
        vals = adata.obs[key].astype(str)
        cats = sorted(pd.unique(vals))
        adata.obs[key] = pd.Categorical(vals, categories=cats)


def filter_obs_keys(adata: sc.AnnData, keys: Iterable[str]) -> List[str]:
    return [key for key in keys if key in adata.obs]


def _make_dataset(x: np.ndarray, y: np.ndarray):
    from scplantFormer_model_mae import scDataSet

    return scDataSet(data=x, label=y)
