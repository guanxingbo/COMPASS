#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import anndata as ad
import scipy.sparse


def _ensure_spatial(adata: ad.AnnData, key: str = "spatial") -> None:
    if key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm['{key}']. Please check your input file.")


def _align_by_obs_names(rna: ad.AnnData, atac: ad.AnnData, require_all: bool = False):
    rna_names = list(rna.obs_names)
    atac_set = set(atac.obs_names)
    common = [s for s in rna_names if s in atac_set]

    if require_all and len(common) != rna.n_obs:
        raise ValueError("RNA obs_names are not fully contained in ATAC obs_names.")
    if require_all and len(common) != atac.n_obs:
        raise ValueError("ATAC obs_names are not fully contained in RNA obs_names.")

    if len(common) == 0:
        raise ValueError("No common obs_names between RNA and ATAC.")

    rna2 = rna[common, :].copy()
    atac2 = atac[common, :].copy()
    return rna2, atac2


def _split_left_right_by_x(adata: ad.AnnData, spatial_key: str = "spatial"):
    _ensure_spatial(adata, spatial_key)
    spatial = adata.obsm[spatial_key]
    if spatial.shape[1] < 1:
        raise ValueError("spatial obsm must have at least 1 column (x).")
    x = np.asarray(spatial[:, 0]).astype(float)

    if spatial.shape[1] >= 2:
        y = np.asarray(spatial[:, 1]).astype(float)
    else:
        y = np.zeros_like(x)

    n = adata.n_obs
    # sort by x then y; left = first half by count, right = remaining
    idx_sorted = np.lexsort((y, x))
    left_idx = idx_sorted[: n // 2]
    right_idx = idx_sorted[n // 2:]
    return left_idx, right_idx


def _zero_x_rows(adata: ad.AnnData, idx: np.ndarray) -> None:
    """
    Hard drop signal: set adata.X[idx, :] = 0
    - Only modifies X (expression/accessibility matrix).
    - Does NOT modify coords or other annotations.
    """
    if idx.size == 0:
        return

    if scipy.sparse.issparse(adata.X):
        X_new = adata.X.tocsr(copy=True)
        X_new[idx, :] = 0
        X_new.eliminate_zeros()
        adata.X = X_new
    else:
        X_new = np.asarray(adata.X).copy()
        X_new[idx, :] = 0
        adata.X = X_new

DATA_PATH="./v2_rna_strong/"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-rna", default=DATA_PATH+"simulation_rna.h5ad", help="Path to base simulation_rna.h5ad (AnnData).")
    ap.add_argument("--base-atac", default=DATA_PATH+"simulation_atac.h5ad", help="Path to base simulation_atac.h5ad (AnnData).")
    ap.add_argument("--out-dir", default=DATA_PATH)
    ap.add_argument("--seed", type=int, default=221, help="Reserved for compatibility; not used in full-half zeroing.")
    ap.add_argument("--spatial-key", default="spatial", help="obsm key for spot coordinates.")
    ap.add_argument("--require-all-spots", action="store_true", help="Fail if obs_names do not fully match.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rna0 = ad.read_h5ad(args.base_rna)
    atac0 = ad.read_h5ad(args.base_atac)

    # Align spots by obs_names
    rna0, atac0 = _align_by_obs_names(rna0, atac0, require_all=args.require_all_spots)

    # Split left/right for each modality independently (based on their own coords).
    rna_left, rna_right = _split_left_right_by_x(rna0, args.spatial_key)
    atac_left, atac_right = _split_left_right_by_x(atac0, args.spatial_key)

    # Convention (hard complement):
    # - RNA: drop Right (set X rows to 0)
    # - ATAC: drop Left (set X rows to 0)
    rna = rna0.copy()
    atac = atac0.copy()

    _zero_x_rows(rna, rna_right)
    _zero_x_rows(atac, atac_left)

    out_rna = os.path.join(args.out_dir, "simulation_rna_drop.h5ad")
    out_atac = os.path.join(args.out_dir, "simulation_atac_drop.h5ad")

    rna.write_h5ad(out_rna)
    atac.write_h5ad(out_atac)

    print("wrote:")
    print(f"  RNA : {out_rna} (zeroed rows: {rna_right.size})")
    print(f"  ATAC: {out_atac} (zeroed rows: {atac_left.size})")


if __name__ == "__main__":
    main()