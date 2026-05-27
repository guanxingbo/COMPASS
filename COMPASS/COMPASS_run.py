# -*- coding: utf-8 -*-
"""
COMPASS_run: entry point — load data, run model / inference, then four figures.

Training code is not in this repository; implement ``run_compass_model`` with
logic ported from COMPASS_method.ipynb, or use h5ad files that already contain
the required embeddings and labels (see ``compass_outputs_ready``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA

from COMPASS_data import load_and_preprocess
from COMPASS_core import train_multimodal_and_predict_sc
from COMPASS_method import (
    assign_cell_locations_from_coord_strings,
    build_spot_type_prop_soft,
    hard_assign_cells_from_soft,
    map_cells_to_spots_dom_ct,
    plot_celltype_by_pred_domain,
    plot_cells_spatial,
    plot_spatial_pies,
)


def compass_outputs_ready(sc_adata, adata_prep) -> bool:
    """Return True if AnnData objects already hold fields needed for plotting."""
    need_sc_obsm = ("gae_latent", "pred_domain_proba")
    need_sc_obs = ("pred_domain", "CellType")
    need_st_obsm = ("gae_latent", "spatial")
    need_st_obs = ("domain",)
    for k in need_sc_obsm:
        if k not in sc_adata.obsm:
            return False
    for k in need_sc_obs:
        if k not in sc_adata.obs:
            return False
    for k in need_st_obsm:
        if k not in adata_prep.obsm:
            return False
    for k in need_st_obs:
        if k not in adata_prep.obs:
            return False
    return True


def run_compass_model(sc_adata, adata_prep, adt_adata, *, skip_if_ready: bool = True) -> None:
    """
    Train / infer COMPASS (GAE, domains, predictions). Port the notebook block here.

    After this returns, ``compass_outputs_ready(sc_adata, adata_prep)`` must be True
    unless you only rely on precomputed h5ad and ``skip_if_ready`` is True.
    """
    if skip_if_ready and compass_outputs_ready(sc_adata, adata_prep):
        return

    # --- 1) Ensure ST domain label is available under the expected key ---
    # Your simulations typically store ground-truth under `gt`.
    if "domain" not in adata_prep.obs:
        if "gt" in adata_prep.obs:
            adata_prep.obs["domain"] = adata_prep.obs["gt"]
        else:
            raise KeyError("adata_prep.obs must contain 'gt' (preferred) or 'domain'.")

    # --- 2) Train multimodal GAE on ST and infer domains for scRNA ---
    res = train_multimodal_and_predict_sc(
        adata_st=adata_prep,
        adata_sc=sc_adata,
        adata_mod2=adt_adata,
        st_label_key="domain",
        use_layer="log1p",
        mod2_kind="atac",
        k=6,
        sigma=None,
        hid_dim=128,
        z_dim=32,
        pca_dim=50,
        mod2_dim=50,
        epochs=200,
        batch_size_sc=512,
        sc_knn_k=15,
        sc_knn_sigma=None,
        pca_random_state=0,
        use_amp=False,
    )

    # ST embedding for downstream OT + figures
    adata_prep.obsm["gae_latent"] = res.z_st

    # sc embedding and domain probabilities
    sc_adata.obsm["gae_latent"] = res.z_sc
    sc_adata.obsm["pred_domain_proba"] = pd.DataFrame(
        res.proba_sc, index=sc_adata.obs_names, columns=res.classes
    )
    pred_idx = res.proba_sc.argmax(axis=1)
    sc_adata.obs["pred_domain"] = pd.Categorical.from_codes(pred_idx, res.classes)

    # Ensure CellType exists (downstream plots expect this name)
    if "CellType" not in sc_adata.obs:
        if "cell_type" in sc_adata.obs:
            sc_adata.obs["CellType"] = sc_adata.obs["cell_type"]
        elif "cellType" in sc_adata.obs:
            sc_adata.obs["CellType"] = sc_adata.obs["cellType"]
        else:
            raise KeyError("sc_adata.obs must contain 'CellType' (or 'cell_type'/'cellType').")
    sc_adata.obs["CellType"] = sc_adata.obs["CellType"].astype(str).astype("category")

    if not compass_outputs_ready(sc_adata, adata_prep):
        raise RuntimeError("COMPASS outputs are still missing required fields after run_compass_model().")


def run_four_figure_pipeline(
    sc_adata,
    adata_prep,
    *,
    proba_row_min: float = 0.1,
    st_domain_key: str = "domain",
    ot_eps: float = 0.05,
    ot_iters: int = 200,
) -> Tuple[object, object]:
    """
    Four figures in order: PCA → stacked bar → spot pies → cell scatter.

    Returns
    -------
    sc_adata_filtered, out_soft
        Filtered single-cell AnnData and cell coordinate frame from soft assignment.
    """
    # --- Fig. 1: PCA on gae_latent (filter cells by max pred_domain_proba) ---
    proba = sc_adata.obsm["pred_domain_proba"]
    if isinstance(proba, np.ndarray):
        proba_df = pd.DataFrame(proba, index=sc_adata.obs_names)
    else:
        proba_df = proba
    rowmax = proba_df.max(axis=1, skipna=True)
    keep = rowmax >= proba_row_min
    sc_adata_filtered = sc_adata[keep].copy()

    z = sc_adata_filtered.obsm["gae_latent"]
    pca = PCA(n_components=20, random_state=0)
    sc_adata_filtered.obsm["X_pca_gae"] = pca.fit_transform(z)
    for col in ("pred_domain", "CellType"):
        if col in sc_adata_filtered.obs:
            if not pd.api.types.is_categorical_dtype(sc_adata_filtered.obs[col]):
                sc_adata_filtered.obs[col] = sc_adata_filtered.obs[col].astype("category")
            sc.pl.embedding(
                sc_adata_filtered,
                basis="pca_gae",
                color=col,
                frameon=False,
                title=f"PCA (gae) – {col}",
            )

    # --- Fig. 2: stacked bar — x = pred_domain, stacks = CellType ---
    plot_celltype_by_pred_domain(
        sc_adata_filtered,
        ct_key="pred_domain",
        dom_key="CellType",
        sort_by_total=True,
        figsize=(10, 4),
        annotate=True,
    )

    # --- Fig. 3: spot-level soft pies ---
    p_full, _assigned, _spot_type_prop = map_cells_to_spots_dom_ct(
        sc_adata_filtered,
        adata_prep,
        emb_key_sc="gae_latent",
        emb_key_st="gae_latent",
        sc_domain_key="pred_domain",
        st_domain_key=st_domain_key,
        sc_type_key="CellType",
        metric="cosine",
        eps=ot_eps,
        iters=ot_iters,
        standardize=False,
        assignment_mode="argmax",
    )
    spot_type_prop_soft = build_spot_type_prop_soft(
        p_full,
        sc_adata_filtered,
        adata_prep,
        sc_domain_key="pred_domain",
        st_domain_key=st_domain_key,
        sc_type_key="CellType",
    )
    _ = plot_spatial_pies(
        adata_prep,
        spot_type_prop_soft,
        coord_key="spatial",
        radius=5,
        legend=True,
    )

    # --- Fig. 4: cell-level scatter (soft assignment + Poisson-disk layout) ---
    assigned_spot_for_cell_soft = hard_assign_cells_from_soft(
        p_full,
        sc_adata=sc_adata_filtered,
        st_adata=adata_prep,
        sc_domain_key="pred_domain",
        st_domain_key=st_domain_key,
        sc_type_key="CellType",
        p_eps=1e-12,
        random_state=0,
    )
    out_soft = assign_cell_locations_from_coord_strings(
        assigned_spot_for_cell_soft,
        st_adata=adata_prep,
        min_dist=None,
        k_nn=8,
        seed=0,
        support_mode="union_disks",
        support_radius_factor=1.1,
        clip_to_spot=True,
        clip_radius_factor=1.3,
        clip_radius_abs=None,
    )
    plot_cells_spatial(sc_adata_filtered, out_soft, color_keys=("pred_domain", "CellType"), s=5)

    return sc_adata_filtered, out_soft


def main(
    path_sc: Optional[Path] = None,
    path_st_rna: Optional[Path] = None,
    path_st_mod2: Optional[Path] = None,
) -> None:
    root = Path(__file__).resolve().parent
    path_sc = path_sc or root / "ref_RNA.h5ad"
    path_st_rna = path_st_rna or root / "simulation_rna_drop.h5ad"
    path_st_mod2 = path_st_mod2 or root / "simulation_atac.h5ad"

    sc_adata, adata, adt_adata = load_and_preprocess(
        str(path_sc),
        str(path_st_rna),
        str(path_st_mod2),
        run_seed=123,
        simo_start=False,
        verbose=True,
    )
    adata_prep = adata.copy()

    run_compass_model(sc_adata, adata_prep, adt_adata, skip_if_ready=True)
    run_four_figure_pipeline(sc_adata, adata_prep)


if __name__ == "__main__":
    main()
