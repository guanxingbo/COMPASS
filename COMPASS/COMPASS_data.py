# -*- coding: utf-8 -*-
"""
COMPASS / SIMO: load three AnnData objects and apply the same preprocessing
as the first cells of COMPASS_method.ipynb.

Dependencies: anndata, pandas, numpy; optional scanpy import for environment checks.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

try:
    import scanpy as sc  # noqa: F401 — optional; matches notebook imports
except ImportError:
    sc = None


# --------------------------------------------------------------------------- #
# Global random seed (optional; same role as the first notebook cell)
# --------------------------------------------------------------------------- #
def set_global_seed(
    seed: int,
    deterministic_torch: bool = True,
    cudnn_benchmark: bool = False,
    verbose: bool = False,
) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                pass
        except Exception:
            pass
        torch.backends.cudnn.deterministic = bool(deterministic_torch)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if deterministic_torch and hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        if verbose:
            print(f"[seed] PyTorch not seeded: {e}")
    if verbose:
        print(
            f"[seed] Set global seed = {seed} | deterministic_torch={deterministic_torch} | cudnn_benchmark={cudnn_benchmark}"
        )


def simo_timer_start() -> float:
    """Wall-clock start for optional timing (same as ``simo_start_time = time.time()``)."""
    return time.time()


# --------------------------------------------------------------------------- #
# Name cleanup (var / obs)
# --------------------------------------------------------------------------- #
def _clean_index_to_str(idx):
    return pd.Index(pd.Index(idx).astype(str).str.strip())


def _normalize_var_names(adata_obj, obj_name: str):
    adata_obj.var_names = _clean_index_to_str(adata_obj.var_names)
    if adata_obj.var.index.name is not None and not isinstance(adata_obj.var.index.name, str):
        adata_obj.var.index.name = str(adata_obj.var.index.name)
    adata_obj.var_names_make_unique()
    dup_n = int(adata_obj.var_names.duplicated().sum())
    if dup_n > 0:
        raise ValueError(f"{obj_name}.var_names still has {dup_n} duplicate(s) after make_unique")
    return adata_obj


def _normalize_obs_names_sc(adata_obj, obj_name: str):
    adata_obj.obs_names = _clean_index_to_str(adata_obj.obs_names)
    adata_obj.obs_names_make_unique()
    dup_n = int(adata_obj.obs_names.duplicated().sum())
    if dup_n > 0:
        raise ValueError(f"{obj_name}.obs_names still has {dup_n} duplicate(s) after make_unique")
    return adata_obj


def _normalize_obs_names_spatial(adata_obj, obj_name: str):
    adata_obj.obs_names = _clean_index_to_str(adata_obj.obs_names)
    dup_mask = adata_obj.obs_names.duplicated()
    if dup_mask.any():
        dup_examples = adata_obj.obs_names[dup_mask].tolist()[:10]
        raise ValueError(
            f"{obj_name}.obs_names contains duplicates (unsafe for multimodal alignment). "
            f"Examples: {dup_examples}"
        )
    return adata_obj


def unify_celltype_column(sc_adata) -> None:
    if "CellType" in sc_adata.obs:
        pass
    elif "cell_type" in sc_adata.obs:
        sc_adata.obs["CellType"] = sc_adata.obs["cell_type"]
    elif "cellType" in sc_adata.obs:
        sc_adata.obs["CellType"] = sc_adata.obs["cellType"]
    else:
        raise KeyError("sc_adata.obs has no CellType / cell_type / cellType column")
    sc_adata.obs["CellType"] = sc_adata.obs["CellType"].astype(str).astype("category")


def basic_overlap_checks(adata, sc_adata, adt_adata, *, verbose: bool = True) -> Tuple[int, int]:
    common_genes = adata.var_names.intersection(sc_adata.var_names)
    common_spots = adata.obs_names.intersection(adt_adata.obs_names)
    if verbose:
        print(f"sc_adata:   n_obs={sc_adata.n_obs}, n_vars={sc_adata.n_vars}")
        print(f"adata:      n_obs={adata.n_obs}, n_vars={adata.n_vars}")
        print(f"adt_adata:  n_obs={adt_adata.n_obs}, n_vars={adt_adata.n_vars}")
        print(f"Common genes between ST RNA and scRNA: {len(common_genes)}")
        print(f"Common spots between ST RNA and 2nd modality: {len(common_spots)}")
    if len(common_genes) == 0:
        raise ValueError("No common genes between adata and sc_adata; downstream alignment will fail.")
    if len(common_spots) == 0:
        raise ValueError("No common obs_names between adata and adt_adata; multimodal spot merge will fail.")
    return len(common_genes), len(common_spots)


def load_and_preprocess(
    path_sc: str,
    path_st_rna: str,
    path_st_mod2: str,
    *,
    run_seed: Optional[int] = 123,
    simo_start: bool = False,
    verbose: bool = True,
) -> Tuple["ad.AnnData", "ad.AnnData", "ad.AnnData"]:
    """
    Read three h5ad files and apply notebook-consistent preprocessing.

    Returns
    -------
    sc_adata
        Reference single-cell RNA.
    adata
        Spatial transcriptome RNA (same name as in the notebook).
    adt_adata
        Second modality (ADT, ATAC, etc.; name kept from the notebook).
    """
    if run_seed is not None:
        set_global_seed(run_seed, deterministic_torch=True, cudnn_benchmark=False, verbose=verbose)

    sc_adata = ad.read_h5ad(path_sc)
    adata = ad.read_h5ad(path_st_rna)
    adt_adata = ad.read_h5ad(path_st_mod2)

    for obj_name, obj in [("sc_adata", sc_adata), ("adata", adata), ("adt_adata", adt_adata)]:
        _normalize_var_names(obj, obj_name)

    sc_adata = _normalize_obs_names_sc(sc_adata, "sc_adata")
    adata = _normalize_obs_names_spatial(adata, "adata")
    adt_adata = _normalize_obs_names_spatial(adt_adata, "adt_adata")

    unify_celltype_column(sc_adata)
    basic_overlap_checks(adata, sc_adata, adt_adata, verbose=verbose)

    if simo_start:
        globals()["_SIMO_START_TIME"] = simo_timer_start()

    return sc_adata, adata, adt_adata


__all__ = [
    "set_global_seed",
    "simo_timer_start",
    "load_and_preprocess",
    "unify_celltype_column",
    "basic_overlap_checks",
    "_normalize_var_names",
    "_normalize_obs_names_sc",
    "_normalize_obs_names_spatial",
]
