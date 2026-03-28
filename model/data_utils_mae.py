from __future__ import annotations

from typing import Optional, Tuple, Iterable
from pathlib import Path
import os
import time
import hashlib

import numpy as np
import scipy.sparse as sp
import scanpy as sc


def _to_dense(x):
    if not isinstance(x, np.ndarray):
        x = x.todense()
    return np.asarray(x)


def _sanitize_matrix(adata: sc.AnnData, clip_max: float = 700.0) -> None:
    X = adata.X
    if sp.issparse(X):
        data = X.data
        if data.size:
            bad = ~np.isfinite(data)
            if bad.any():
                data[bad] = 0
            if clip_max is not None and np.nanmax(np.abs(data)) > clip_max:
                data[data > clip_max] = clip_max
                data[data < -clip_max] = -clip_max
    else:
        X = np.asarray(X)
        bad = ~np.isfinite(X)
        if clip_max is not None and np.nanmax(np.abs(X)) > clip_max:
            X = np.clip(X, -clip_max, clip_max)
        if bad.any():
            X[bad] = 0
        adata.X = X


def gene_embedding(x: np.ndarray, gap: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"gene_embedding expects 2D array, got shape {x.shape}")
    num_cells, length = x.shape
    if length < gap:
        single_cell_list = []
        for single_cell in x:
            feature = []
            for k in range(0, length, gap):
                if k + gap > length:
                    a = single_cell[length - gap : length]
                else:
                    a = single_cell[k : k + gap]
                feature.append(a)
            feature = np.asarray(feature)
            single_cell_list.append(feature)
        single_cell_list = np.asarray(single_cell_list)
        print("single_cell_list.shape", single_cell_list.shape)
        return single_cell_list

    if length % gap == 0:
        single_cell_list = x.reshape(num_cells, length // gap, gap)
        print("single_cell_list.shape", single_cell_list.shape)
        return single_cell_list

    starts = np.arange(0, length - gap + 1, gap, dtype=np.int64)
    tail = length - gap
    if starts.size == 0 or starts[-1] != tail:
        starts = np.concatenate([starts, np.array([tail], dtype=np.int64)])
    idx = starts[:, None] + np.arange(gap, dtype=np.int64)[None, :]
    single_cell_list = np.take(x, idx, axis=1)
    print("single_cell_list.shape", single_cell_list.shape)
    return single_cell_list


def _embedding_index(length: int, gap: int):
    if length < gap:
        return None, 1, length
    if length % gap == 0:
        return None, length // gap, gap
    starts = np.arange(0, length - gap + 1, gap, dtype=np.int64)
    tail = length - gap
    if starts.size == 0 or starts[-1] != tail:
        starts = np.concatenate([starts, np.array([tail], dtype=np.int64)])
    idx = starts[:, None] + np.arange(gap, dtype=np.int64)[None, :]
    return idx, len(starts), gap


def build_patch_gene_index(n_genes: int, gap: int) -> np.ndarray:
    if n_genes < gap:
        return np.arange(n_genes, dtype=np.int64)[None, :]
    if n_genes % gap == 0:
        starts = np.arange(0, n_genes, gap, dtype=np.int64)
        idx = starts[:, None] + np.arange(gap, dtype=np.int64)[None, :]
        return idx
    idx, _, _ = _embedding_index(n_genes, gap)
    return idx


def _bool_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:d}h{minutes:02d}m{secs:05.2f}s"
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:d}m{secs:05.2f}s"
    return f"{seconds:.2f}s"


def _iter_chunks(total: int, chunk_size: int):
    if chunk_size <= 0 or chunk_size >= total:
        yield 0, total
        return
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        yield start, end


def _gene_order_tag(gene_order: Optional[Iterable[str]]) -> str:
    if not gene_order:
        return ""
    gene_list = list(gene_order)
    payload = "\n".join(gene_list).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:8]
    return f"_go{len(gene_list)}_{digest}"


def _pretrain_cache_path(
    path: str,
    gap: int,
    cache_dir: Optional[str],
    gene_order: Optional[Iterable[str]] = None,
) -> Path:
    src = Path(path)
    base = Path(cache_dir) if cache_dir else src.parent / ".scplant_cache"
    base.mkdir(parents=True, exist_ok=True)
    stat = src.stat()
    gene_tag = _gene_order_tag(gene_order)
    tag = f"{src.stem}_gap{gap}_mt{int(stat.st_mtime)}_sz{stat.st_size}{gene_tag}"
    return base / f"{tag}_pretrain"


def _pretrain_cache_files(
    path: str,
    gap: int,
    cache_dir: Optional[str],
    gene_order: Optional[Iterable[str]] = None,
):
    base = _pretrain_cache_path(path, gap, cache_dir, gene_order=gene_order)
    return base.with_name(base.name + "_x1.npy"), base.with_name(base.name + "_y1.npy")


def _load_dense_chunk(x, start: int, end: int) -> np.ndarray:
    chunk = x[start:end]
    if sp.issparse(chunk):
        chunk = chunk.toarray()
    return np.asarray(chunk)


_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "load_time": 0.0,
    "build_time": 0.0,
}


def _log_cache_stats(load_time: float, build_time: float, total_time: float):
    total = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
    hit_rate = (_CACHE_STATS["hits"] / total) if total else 0.0
    print(
        "[CacheStats] hits={}/{} rate={:.1f}% load={} build={} total={}".format(
            _CACHE_STATS["hits"],
            total,
            hit_rate * 100.0,
            _format_duration(load_time),
            _format_duration(build_time),
            _format_duration(total_time),
        )
    )


def prepare_pretrain_all_nomask(
    path: str,
    gap: int,
    cache_dir: Optional[str] = None,
    use_cache: Optional[bool] = None,
    gene_order: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    t_total = time.perf_counter()
    load_time = 0.0
    build_time = 0.0
    if use_cache is None:
        use_cache = _bool_env("SCPLANT_PRETRAIN_CACHE", True)
    if cache_dir is None:
        cache_dir = os.getenv("SCPLANT_CACHE_DIR", None)
    use_memmap = _bool_env("SCPLANT_PRETRAIN_MEMMAP", True)
    chunk_size = int(os.getenv("SCPLANT_PRETRAIN_CHUNK", "1024"))
    cache_x1 = None
    cache_y1 = None
    gene_list = list(gene_order) if gene_order is not None else None
    if use_cache:
        cache_x1, cache_y1 = _pretrain_cache_files(path, gap, cache_dir, gene_order=gene_list)

    x1 = None
    y1 = None
    if cache_x1 is not None and cache_y1 is not None and cache_x1.exists() and cache_y1.exists():
        t0 = time.perf_counter()
        mmap_mode = "r" if use_memmap else None
        x1 = np.load(cache_x1, mmap_mode=mmap_mode, allow_pickle=False)
        y1 = np.load(cache_y1, mmap_mode=mmap_mode, allow_pickle=False)
        load_time = time.perf_counter() - t0
        _CACHE_STATS["hits"] += 1
        _CACHE_STATS["load_time"] += load_time
        print(f"[Cache] loaded {cache_x1}")
    else:
        t0 = time.perf_counter()
        adata_mod1 = None
        try:
            adata_mod1 = sc.read_h5ad(path, backed="r")
        except Exception:
            adata_mod1 = sc.read_h5ad(path)

        x_data = adata_mod1.X
        n_cells, n_genes = adata_mod1.shape
        keep_cols = None
        new_positions = None
        if gene_list is not None:
            var_names = list(adata_mod1.var_names)
            indexer = {name: i for i, name in enumerate(var_names)}
            keep_cols = np.array(
                [indexer[g] for g in gene_list if g in indexer], dtype=np.int64
            )
            new_positions = np.array(
                [i for i, g in enumerate(gene_list) if g in indexer], dtype=np.int64
            )
            if keep_cols.size == 0:
                raise ValueError("No overlapping genes between dataset and gene_order.")
            missing = len(gene_list) - keep_cols.size
            extra = len(var_names) - keep_cols.size
            if missing:
                print(f"[Warn] gene_order missing {missing} genes; fill zeros.")
            if extra:
                print(f"[Info] dataset has {extra} extra genes not in gene_order; drop.")
            n_genes = len(gene_list)

        idx, n_patches, patch_len = _embedding_index(n_genes, gap)

        def _load_chunk(start: int, end: int) -> np.ndarray:
            chunk = x_data[start:end]
            if sp.issparse(chunk):
                if keep_cols is not None:
                    chunk = chunk[:, keep_cols]
                chunk = chunk.toarray()
            else:
                chunk = np.asarray(chunk)
                if keep_cols is not None:
                    chunk = chunk[:, keep_cols]
            return np.asarray(chunk)

        first_end = min(n_cells, chunk_size if chunk_size > 0 else n_cells)
        first_chunk = _load_chunk(0, first_end)
        dtype = first_chunk.dtype

        if use_cache and cache_x1 is not None and cache_y1 is not None:
            x1 = np.lib.format.open_memmap(cache_x1, mode="w+", dtype=dtype, shape=(n_cells, n_patches, patch_len))
            y1 = np.lib.format.open_memmap(cache_y1, mode="w+", dtype=dtype, shape=(n_cells, n_genes))
        else:
            x1 = np.empty((n_cells, n_patches, patch_len), dtype=dtype)
            y1 = np.empty((n_cells, n_genes), dtype=dtype)

        for start, end in _iter_chunks(n_cells, chunk_size):
            if start == 0:
                x_chunk = first_chunk
            else:
                x_chunk = _load_chunk(start, end)

            if gene_list is None:
                y_chunk = x_chunk
            else:
                y_chunk = np.zeros((end - start, n_genes), dtype=dtype)
                y_chunk[:, new_positions] = x_chunk

            y1[start:end] = y_chunk
            if n_genes < gap:
                x1[start:end] = gene_embedding(y_chunk, gap)
            elif idx is None and n_genes % gap == 0:
                x1[start:end] = y_chunk.reshape(end - start, n_patches, gap)
            else:
                x1[start:end] = np.take(y_chunk, idx, axis=1)

        if hasattr(x1, "flush"):
            x1.flush()
        if hasattr(y1, "flush"):
            y1.flush()

        if getattr(adata_mod1, "isbacked", False):
            try:
                adata_mod1.file.close()
            except Exception:
                pass

        build_time = time.perf_counter() - t0
        _CACHE_STATS["misses"] += 1
        _CACHE_STATS["build_time"] += build_time
        if use_cache and cache_x1 is not None and cache_y1 is not None:
            print(f"[Cache] saved {cache_x1}")

    total_time = time.perf_counter() - t_total
    _log_cache_stats(load_time, build_time, total_time)
    return x1, y1


def prepare_root_cross_dataset(path: str, ref_path: str, gap: int):
    adata_ref = sc.read_h5ad(ref_path)
    adata_mod1 = sc.read_h5ad(path)
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    adata_mod1 = adata_mod1[:, adata_ref.var["features"].values]
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["Dataset"].values
    return x1, y1, adata_mod1, batch_list


def prepare_infer_arrays(adata_mod1: sc.AnnData, gap: int, batch_key: str | None = None):
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = None
    if batch_key and batch_key in adata_mod1.obs:
        batch_list = adata_mod1.obs[batch_key].values
    return x1, y1, adata_mod1, batch_list


def prepare_root_cross_batch_from_adata(adata_mod1: sc.AnnData, adata_ref: sc.AnnData, gap: int):
    adata_mod1 = _prepare_hvg_align(adata_mod1, adata_ref)
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["experiments"].values
    return x1, y1, adata_mod1, batch_list


def prepare_leaf_cross_batch_from_adata(adata_mod1: sc.AnnData, adata_ref: sc.AnnData, gap: int):
    adata_mod1 = _prepare_hvg_align(adata_mod1, adata_ref)
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["experiments"].values
    return x1, y1, adata_mod1, batch_list


def _prepare_hvg_align(adata_mod1: sc.AnnData, adata_ref: sc.AnnData) -> sc.AnnData:
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    _sanitize_matrix(adata_mod1)
    for n_top in range(2000, adata_mod1.shape[1], 500):
        sc.pp.highly_variable_genes(adata_mod1, n_top_genes=n_top)
        adata_mod_temp = adata_mod1[:, adata_mod1.var.highly_variable]
        adata_mod1_con = adata_ref.concatenate(adata_mod_temp)
        n_top_con = adata_mod1_con.shape[1]
        if n_top_con > 2500:
            break

    print("adata_mod1_con", adata_mod1_con)
    print("adata_mod_temp", adata_mod_temp)

    adata_mod1_all = adata_ref.concatenate(adata_mod_temp, join="outer")
    adata_mod1 = adata_mod1_all[len(adata_ref) :]
    adata_mod1 = adata_mod1[:, adata_ref.var_names.values]
    return adata_mod1


def _prepare_align_without_hvg(adata_mod1: sc.AnnData, adata_ref: sc.AnnData) -> sc.AnnData:
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    _sanitize_matrix(adata_mod1)
    adata_mod1_all = adata_ref.concatenate(adata_mod1, join="outer")
    adata_mod1 = adata_mod1_all[len(adata_ref) :]
    adata_mod1 = adata_mod1[:, adata_ref.var_names.values]
    return adata_mod1


def _use_hvg_for_leaf_cross_tech() -> bool:
    no_hvg = os.getenv("SCPLANT_NO_HVG_LEAF_CROSS_TECH", "").strip().lower() in {"1", "true", "yes"}
    return not no_hvg


def prepare_root_cross_batch(path: str, ref_path: str, gap: int):
    adata_ref = sc.read_h5ad(ref_path)
    adata_mod1 = sc.read_h5ad(path)
    adata_mod1 = _prepare_hvg_align(adata_mod1, adata_ref)
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["experiments"].values
    return x1, y1, adata_mod1, batch_list


def prepare_leaf_cross_batch(path: str, ref_path: str, gap: int):
    adata_ref = sc.read_h5ad(ref_path)
    adata_mod1 = sc.read_h5ad(path)
    adata_mod1 = _prepare_hvg_align(adata_mod1, adata_ref)
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["experiments"].values
    return x1, y1, adata_mod1, batch_list


def prepare_root_cross_tech(path: str, ref_path: str, gap: int):
    adata_ref = sc.read_h5ad(ref_path)
    adata_mod1 = sc.read_h5ad(path)
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.filter_genes(adata_mod1, min_counts=1)
    adata_mod1_all = adata_ref.concatenate(adata_mod1, join="outer")
    adata_mod1 = adata_mod1_all[len(adata_ref) :]
    adata_mod1 = adata_mod1[:, adata_ref.var_names.values]
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["Dataset"].values
    return x1, y1, adata_mod1, batch_list


def prepare_root_cross_tech_from_adata(adata_mod1: sc.AnnData, adata_ref: sc.AnnData, gap: int):
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.filter_genes(adata_mod1, min_counts=1)
    adata_mod1_all = adata_ref.concatenate(adata_mod1, join="outer")
    adata_mod1 = adata_mod1_all[len(adata_ref) :]
    adata_mod1 = adata_mod1[:, adata_ref.var_names.values]
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["Dataset"].values
    return x1, y1, adata_mod1, batch_list


def prepare_leaf_cross_tech_rna(path: str, ref_path: str, gap: int):
    adata_ref = sc.read_h5ad(ref_path)
    adata_mod1 = sc.read_h5ad(path)
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    sc.pp.normalize_total(adata_mod1)
    sc.pp.log1p(adata_mod1)
    if _use_hvg_for_leaf_cross_tech():
        adata_mod1 = _prepare_hvg_align(adata_mod1, adata_ref)
    else:
        print("[Info] SCPLANT_NO_HVG_LEAF_CROSS_TECH enabled: skip HVG for leaf cross-tech RNA")
        adata_mod1 = _prepare_align_without_hvg(adata_mod1, adata_ref)
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["time"].values
    return x1, y1, adata_mod1, batch_list


def prepare_leaf_cross_tech_rna_from_adata(adata_mod1: sc.AnnData, adata_ref: sc.AnnData, gap: int):
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    sc.pp.normalize_total(adata_mod1)
    sc.pp.log1p(adata_mod1)
    if _use_hvg_for_leaf_cross_tech():
        adata_mod1 = _prepare_hvg_align(adata_mod1, adata_ref)
    else:
        print("[Info] SCPLANT_NO_HVG_LEAF_CROSS_TECH enabled: skip HVG for leaf cross-tech RNA")
        adata_mod1 = _prepare_align_without_hvg(adata_mod1, adata_ref)
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["time"].values
    return x1, y1, adata_mod1, batch_list


def prepare_leaf_cross_tech_atac(path: str, ref_path: str, gap: int):
    adata_ref = sc.read_h5ad(ref_path)
    adata_mod1 = sc.read_h5ad(path)
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    sc.pp.normalize_total(adata_mod1)
    sc.pp.log1p(adata_mod1)
    if _use_hvg_for_leaf_cross_tech():
        adata_mod1 = _prepare_hvg_align(adata_mod1, adata_ref)
    else:
        print("[Info] SCPLANT_NO_HVG_LEAF_CROSS_TECH enabled: skip HVG for leaf cross-tech ATAC")
        adata_mod1 = _prepare_align_without_hvg(adata_mod1, adata_ref)
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["time"].values
    return x1, y1, adata_mod1, batch_list


def prepare_leaf_cross_tech_atac_from_adata(adata_mod1: sc.AnnData, adata_ref: sc.AnnData, gap: int):
    adata_mod1.var_names_make_unique()
    adata_mod1.obs["domain_id"] = "0"
    sc.pp.normalize_total(adata_mod1)
    sc.pp.log1p(adata_mod1)
    if _use_hvg_for_leaf_cross_tech():
        adata_mod1 = _prepare_hvg_align(adata_mod1, adata_ref)
    else:
        print("[Info] SCPLANT_NO_HVG_LEAF_CROSS_TECH enabled: skip HVG for leaf cross-tech ATAC")
        adata_mod1 = _prepare_align_without_hvg(adata_mod1, adata_ref)
    x1 = _to_dense(adata_mod1.X)
    y1 = x1
    x1 = gene_embedding(x1, gap)
    batch_list = adata_mod1.obs["time"].values
    return x1, y1, adata_mod1, batch_list
