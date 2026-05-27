# -*- coding: utf-8 -*-
"""
Core model logic ported from Scenario4/multi_omics_S4.ipynb.

This file provides a minimal, script-friendly implementation of:
- multimodal (RNA + mod2) graph autoencoder training on spatial spots
- forward pass on single-cell data to get domain probabilities and embeddings

It intentionally focuses on producing the fields required by `COMPASS_run.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def _ensure_log1p_layer(adata, layer: str = "log1p", target_sum: float = 1e4) -> None:
    if layer in adata.layers:
        return
    # local import keeps scanpy optional unless training is called
    import scanpy as sc

    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=target_sum)
    sc.pp.log1p(tmp)
    adata.layers[layer] = tmp.X.copy()


def build_knn_graph(coords: np.ndarray, k: int = 8, sigma: Optional[float] = None) -> torch.Tensor:
    """
    Build normalized adjacency Â as torch sparse COO.
    """
    coords = np.asarray(coords, dtype=np.float32)
    N = int(coords.shape[0])
    if N <= 1:
        idx = np.vstack([np.arange(N), np.arange(N)])
        return torch.sparse_coo_tensor(idx, np.ones(N, np.float32), (N, N), dtype=torch.float32).coalesce()

    nbrs = NearestNeighbors(n_neighbors=int(k) + 1, metric="euclidean").fit(coords)
    dists, idx = nbrs.kneighbors(coords)

    rows = np.repeat(np.arange(N), int(k))
    cols = idx[:, 1:].ravel()
    if sigma is None:
        data = np.ones_like(rows, dtype=np.float32)
    else:
        data = np.exp(-(dists[:, 1:].ravel() ** 2) / (2.0 * float(sigma) ** 2)).astype(np.float32)

    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    A = A + A.T
    A.setdiag(1.0)

    deg = np.asarray(A.sum(1)).flatten()
    d_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    D_inv_sqrt = sp.diags(d_inv_sqrt.astype(np.float32))
    A_hat = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

    indices = np.vstack([A_hat.row, A_hat.col])
    A_hat_t = torch.sparse_coo_tensor(indices, A_hat.data, (N, N), dtype=torch.float32).coalesce()
    return A_hat_t


def build_batch_knn_subgraph_from_precomputed(
    batch_ids_np: np.ndarray,
    knn_idx_np: np.ndarray,
    knn_dist_np: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Build batch-local normalized Â from precomputed full kNN (numpy on CPU).
    """
    batch_ids_np = np.asarray(batch_ids_np)
    B = int(batch_ids_np.shape[0])
    if B <= 0:
        raise ValueError("empty batch")

    N_total = int(knn_idx_np.shape[0])
    pos = torch.full((N_total,), -1, dtype=torch.int32)
    batch_ids_t = torch.from_numpy(batch_ids_np.astype(np.int64, copy=False))
    pos[batch_ids_t] = torch.arange(B, dtype=torch.int32)

    neigh = torch.from_numpy(knn_idx_np[batch_ids_np].astype(np.int64, copy=False))
    cols = pos[neigh]
    valid = cols >= 0

    rows = torch.arange(B, dtype=torch.int64).unsqueeze(1).expand_as(cols)
    row_idx = rows[valid]
    col_idx = cols[valid].to(torch.int64)

    if (sigma is None) or (knn_dist_np is None):
        val = torch.ones_like(row_idx, dtype=torch.float32)
    else:
        d = torch.from_numpy(knn_dist_np[batch_ids_np].astype(np.float32, copy=False))
        d2 = (d[valid].to(torch.float32)) ** 2
        val = torch.exp(-d2 / (2.0 * float(sigma) ** 2))

    ii = torch.cat([row_idx, col_idx], dim=0)
    jj = torch.cat([col_idx, row_idx], dim=0)
    vv = torch.cat([val, val], dim=0)

    self_idx = torch.arange(B, dtype=torch.int64)
    ii = torch.cat([ii, self_idx], dim=0)
    jj = torch.cat([jj, self_idx], dim=0)
    vv = torch.cat([vv, torch.ones(B, dtype=torch.float32)], dim=0)

    A = torch.sparse_coo_tensor(torch.stack([ii, jj], dim=0), vv, (B, B), dtype=torch.float32).coalesce()

    deg = torch.zeros(B, dtype=torch.float32)
    deg.scatter_add_(0, A.indices()[0], A.values())
    d_inv_sqrt = torch.rsqrt(deg + 1e-8)
    norm_val = A.values() * d_inv_sqrt[A.indices()[0]] * d_inv_sqrt[A.indices()[1]]

    A_hat = torch.sparse_coo_tensor(A.indices(), norm_val, (B, B), dtype=torch.float32).coalesce()
    return A_hat.to(device)


def make_srt_template(adata_srt, use_layer: Optional[str] = None):
    X_srt_raw = _to_dense(adata_srt.layers[use_layer] if use_layer else adata_srt.X).astype(np.float32)
    genes_srt = pd.Index(adata_srt.var_names)
    mu_srt = X_srt_raw.mean(axis=0).astype(np.float32)
    sd_srt = (X_srt_raw.std(axis=0) + 1e-6).astype(np.float32)
    X_srt_std = ((X_srt_raw - mu_srt) / sd_srt).astype(np.float32)
    return X_srt_std, mu_srt, sd_srt, list(genes_srt)


def align_sc_to_srt_template(
    sc_adata,
    genes_srt: Sequence[str],
    mu_srt: np.ndarray,
    sd_srt: np.ndarray,
    *,
    sc_use_layer: Optional[str] = None,
    fill_missing: str = "srt_mean",
):
    X_sc_raw = _to_dense(sc_adata.layers[sc_use_layer] if sc_use_layer else sc_adata.X).astype(np.float32)
    genes_sc = pd.Index(sc_adata.var_names)

    # map_idx[j] = index in sc genes for srt gene j, or -1
    map_idx = genes_sc.get_indexer(pd.Index(genes_srt))
    present_mask = map_idx >= 0

    N_sc = int(X_sc_raw.shape[0])
    F_srt = int(len(genes_srt))
    if fill_missing == "zeros":
        base = np.zeros((N_sc, F_srt), dtype=np.float32)
    elif fill_missing == "srt_mean":
        base = np.broadcast_to(mu_srt.astype(np.float32, copy=False), (N_sc, F_srt)).copy()
    else:
        raise ValueError("fill_missing must be 'zeros' or 'srt_mean'")

    if present_mask.any():
        base[:, present_mask] = X_sc_raw[:, map_idx[present_mask]]

    X_sc_std = ((base - mu_srt) / sd_srt).astype(np.float32)
    return X_sc_std, present_mask


def _safe_n_components(n_obs: int, n_vars: int, requested_dim: int) -> int:
    if n_obs <= 1 or n_vars <= 1:
        return 1
    return int(max(1, min(int(requested_dim), int(n_obs - 1), int(n_vars - 1))))


def _preprocess_mod2_features(
    adata_mod2,
    *,
    mod2_kind: str = "atac",
    use_layer: Optional[str] = None,
    out_dim: int = 50,
    random_state: int = 0,
):
    """
    Minimal: ATAC -> TFIDF -> log1p -> TruncatedSVD(LSI) -> zscore.
    """
    mod2_kind = str(mod2_kind).lower()
    if mod2_kind != "atac":
        raise ValueError("This reference implementation currently supports mod2_kind='atac' only.")

    X_raw = adata_mod2.layers[use_layer] if (use_layer is not None) else adata_mod2.X
    if sp.issparse(X_raw):
        X_sp = X_raw.tocsr().astype(np.float32)
    else:
        X_sp = sp.csr_matrix(np.asarray(X_raw, dtype=np.float32))

    row_sum = np.asarray(X_sp.sum(axis=1)).ravel().astype(np.float32)
    row_sum[row_sum <= 0] = 1.0
    tf = sp.diags(1.0 / row_sum) @ X_sp

    df = np.asarray((X_sp > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log(1.0 + X_sp.shape[0] / (df + 1.0)).astype(np.float32)

    tfidf = (tf @ sp.diags(idf)).tocsr()
    if tfidf.nnz > 0:
        tfidf.data = np.log1p(tfidf.data * 1e4).astype(np.float32, copy=False)

    n_comp = _safe_n_components(tfidf.shape[0], tfidf.shape[1], int(out_dim))
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    X_feat = svd.fit_transform(tfidf).astype(np.float32, copy=False)

    mu_feat = X_feat.mean(axis=0, keepdims=True).astype(np.float32)
    sd_feat = (X_feat.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    X_feat = ((X_feat - mu_feat) / sd_feat).astype(np.float32, copy=False)

    meta = {"mod2_kind": "atac", "mod2_n_comp": int(n_comp), "mod2_lsi_mu": mu_feat.squeeze(0), "mod2_lsi_sd": sd_feat.squeeze(0)}
    return X_feat, meta


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = float(dropout)

    def forward(self, A_hat: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        X = F.dropout(X, p=self.dropout, training=self.training)
        XW = self.lin(X)
        if XW.dtype != torch.float32:
            XW = XW.float()
        if A_hat.dtype != torch.float32:
            A_hat = A_hat.float()
        H = torch.sparse.mm(A_hat, XW)
        return F.relu(H)


class GraphAutoencoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 64, z_dim: int = 16, n_classes: int = 5, dropout: float = 0.1, proj_dim: int = 32):
        super().__init__()
        self.enc1 = GCNLayer(in_dim, hid_dim, dropout=dropout)
        self.enc2 = GCNLayer(hid_dim, z_dim, dropout=dropout)
        self.dec_recon = nn.Linear(z_dim, in_dim)
        self.dec_cls = nn.Linear(z_dim, n_classes)
        self.proj = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(inplace=True), nn.Linear(z_dim, proj_dim))

    def forward(self, A_hat: torch.Tensor, X: torch.Tensor):
        h = self.enc1(A_hat, X)
        z = self.enc2(A_hat, h)
        x_hat = self.dec_recon(z)
        logits = self.dec_cls(z)
        z_proj = self.proj(z)
        return z, z_proj, x_hat, logits


class ResidualGatedFusion(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.mod2_proj = nn.Linear(z_dim, z_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(z_dim)

    def forward(self, z_rna: torch.Tensor, z_mod2: torch.Tensor) -> torch.Tensor:
        z_mod2 = self.mod2_proj(z_mod2)
        gate = self.gate(torch.cat([z_rna, z_mod2], dim=1))
        z_fused = z_rna + gate * z_mod2
        return self.norm(z_fused)


class SRTMultimodalWrapper(nn.Module):
    def __init__(self, F_rna: int, F_mod2: int, hid_dim: int, z_dim: int, n_classes: int, dropout: float = 0.1, proj_dim: int = 32):
        super().__init__()
        head_dim = max(int(n_classes), 2)
        self.rna = GraphAutoencoder(in_dim=F_rna, hid_dim=hid_dim, z_dim=z_dim, n_classes=head_dim, dropout=dropout, proj_dim=proj_dim)
        self.mod2 = GraphAutoencoder(in_dim=F_mod2, hid_dim=hid_dim, z_dim=z_dim, n_classes=head_dim, dropout=dropout, proj_dim=proj_dim)
        self.fuse = ResidualGatedFusion(z_dim)
        self.cls_fused = nn.Linear(z_dim, head_dim)
        self.cls_rna = nn.Linear(z_dim, head_dim)
        self.proj_fused = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(inplace=True), nn.Linear(z_dim, proj_dim))

    def forward_srt(self, A_hat: torch.Tensor, X_rna: torch.Tensor, X_mod2: torch.Tensor):
        z_rna, _, x_hat_rna, _ = self.rna(A_hat, X_rna)
        z_mod2, _, x_hat_mod2, _ = self.mod2(A_hat, X_mod2)
        z_fused = self.fuse(z_rna, z_mod2)
        logits_fused = self.cls_fused(z_fused)
        logits_rna = self.cls_rna(z_rna)
        z_proj_fused = self.proj_fused(z_fused)
        return z_fused, z_proj_fused, logits_fused, x_hat_rna, x_hat_mod2, z_rna, z_mod2, logits_rna

    def forward_sc(self, A_hat_sc: torch.Tensor, X_sc: torch.Tensor):
        z_sc, z_proj_sc, x_hat_sc, _ = self.rna(A_hat_sc, X_sc)
        logits_sc = self.cls_rna(z_sc)
        return z_sc, z_proj_sc, logits_sc, x_hat_sc


@dataclass
class CompassTrainResult:
    classes: list[str]
    z_st: np.ndarray
    z_sc: np.ndarray
    proba_sc: np.ndarray


def train_multimodal_and_predict_sc(
    *,
    adata_st,
    adata_sc,
    adata_mod2,
    st_label_key: str = "domain",
    use_layer: str = "log1p",
    mod2_kind: str = "atac",
    k: int = 6,
    sigma: Optional[float] = None,
    hid_dim: int = 128,
    z_dim: int = 32,
    pca_dim: int = 50,
    mod2_dim: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    batch_size_sc: int = 512,
    sc_knn_k: int = 15,
    sc_knn_sigma: Optional[float] = None,
    pca_random_state: int = 0,
    use_amp: bool = False,
) -> CompassTrainResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ensure_log1p_layer(adata_st, layer=use_layer)
    _ensure_log1p_layer(adata_sc, layer=use_layer)

    common_spots = adata_st.obs_names.intersection(adata_mod2.obs_names)
    if len(common_spots) == 0:
        raise ValueError("ST 与第二模态没有共同 obs_names（spots）。")

    idx_st = adata_st.obs_names.get_indexer(common_spots)
    idx_mod2 = adata_mod2.obs_names.get_indexer(common_spots)
    st_mm = adata_st[idx_st].copy()
    mod2_mm = adata_mod2[idx_mod2].copy()

    # template standardization + PCA features for RNA
    X_st_std, mu_srt, sd_srt, genes_srt = make_srt_template(st_mm, use_layer=use_layer)
    X_sc_std, _present = align_sc_to_srt_template(adata_sc, genes_srt, mu_srt, sd_srt, sc_use_layer=use_layer, fill_missing="srt_mean")

    n_comp_rna = int(min(int(pca_dim), X_st_std.shape[1] - 1)) if X_st_std.shape[1] > 1 else 1
    if sp.issparse(X_st_std):
        scaler = StandardScaler(with_mean=False)
        Xs = scaler.fit_transform(X_st_std)
        Xc = scaler.transform(X_sc_std)
        svd = TruncatedSVD(n_components=n_comp_rna, random_state=pca_random_state)
        X_st_feat = svd.fit_transform(Xs).astype(np.float32, copy=False)
        X_sc_feat = svd.transform(Xc).astype(np.float32, copy=False)
    else:
        pca = PCA(n_components=n_comp_rna, random_state=pca_random_state, svd_solver="randomized")
        X_st_feat = pca.fit_transform(np.asarray(X_st_std)).astype(np.float32, copy=False)
        X_sc_feat = pca.transform(np.asarray(X_sc_std)).astype(np.float32, copy=False)

    X_mod2_feat, _mod2_meta = _preprocess_mod2_features(mod2_mm, mod2_kind=mod2_kind, out_dim=mod2_dim, random_state=pca_random_state)

    X_rna = torch.from_numpy(np.asarray(X_st_feat, dtype=np.float32, order="C")).to(device=device, non_blocking=True)
    X_mod2 = torch.from_numpy(np.asarray(X_mod2_feat, dtype=np.float32, order="C")).to(device=device, non_blocking=True)

    X_sc_feat_cpu = np.asarray(X_sc_feat, dtype=np.float32, order="C")
    N_sc = int(X_sc_feat_cpu.shape[0])

    # sc kNN (for batch graph)
    nbrs_sc = NearestNeighbors(n_neighbors=int(sc_knn_k) + 1, metric="euclidean").fit(X_sc_feat_cpu)
    sc_dists_full, sc_idx_full = nbrs_sc.kneighbors(X_sc_feat_cpu)
    sc_knn_idx = sc_idx_full[:, 1:].astype(np.int64, copy=False)
    sc_knn_dist = sc_dists_full[:, 1:].astype(np.float32, copy=False)

    coords = np.asarray(st_mm.obsm["spatial"])
    A_hat = build_knn_graph(coords, k=int(k), sigma=sigma).to(device)

    gt = st_mm.obs[st_label_key].astype("category")
    mask_labeled = ~gt.isna().to_numpy()
    y_codes = gt.cat.codes.to_numpy().astype(np.int64)
    classes = list(gt.cat.categories)
    n_classes = int(len(classes))

    y = torch.tensor(y_codes, device=device, dtype=torch.long)
    mask = torch.tensor(mask_labeled, dtype=torch.bool, device=device)

    model = SRTMultimodalWrapper(
        F_rna=int(X_rna.shape[1]),
        F_mod2=int(X_mod2.shape[1]),
        hid_dim=int(hid_dim),
        z_dim=int(z_dim),
        n_classes=n_classes,
        dropout=0.1,
        proj_dim=32,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    use_amp = bool(use_amp and (device.type == "cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if batch_size_sc is None or int(batch_size_sc) >= N_sc:
        batch_size_sc = N_sc
    batch_size_sc = int(batch_size_sc)

    for _epoch in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        if batch_size_sc < N_sc:
            batch_sc_ids = np.random.choice(N_sc, size=batch_size_sc, replace=False).astype(np.int64, copy=False)
        else:
            batch_sc_ids = np.arange(N_sc, dtype=np.int64)

        A_sc_batch = build_batch_knn_subgraph_from_precomputed(
            batch_sc_ids, sc_knn_idx, knn_dist_np=sc_knn_dist, sigma=sc_knn_sigma, device=device
        )
        X_sc_batch = torch.from_numpy(X_sc_feat_cpu[batch_sc_ids]).to(device=device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            z_fused, z_proj_fused, logits_fused, x_hat_rna, x_hat_mod2, _z_rna, _z_mod2, _logits_rna = model.forward_srt(
                A_hat, X_rna, X_mod2
            )
            loss_recon_rna = F.mse_loss(x_hat_rna, X_rna)
            loss_recon_mod2 = F.mse_loss(x_hat_mod2, X_mod2)

            if mask.any():
                loss_cls = F.cross_entropy(logits_fused[mask], y[mask])
            else:
                loss_cls = torch.tensor(0.0, device=device)

            # Light regularization: graph smoothness on fused embedding
            z_smooth = z_fused.float() if z_fused.dtype != torch.float32 else z_fused
            A_hat_smooth = A_hat.float() if A_hat.dtype != torch.float32 else A_hat
            smooth = torch.sparse.mm(A_hat_smooth, z_smooth) - z_smooth
            loss_smooth = (smooth * smooth).sum() / max(int(z_smooth.shape[0]), 1)

            # sc forward (aux) to keep head usable
            z_sc_b, _zproj_sc_b, logits_sc_b, _xhat_sc_b = model.forward_sc(A_sc_batch, X_sc_batch)
            _ = z_sc_b  # keep graph; no extra loss here

            loss = loss_recon_rna + loss_recon_mod2 + loss_cls + 0.1 * loss_smooth

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    # ---- final forward: ST embedding + full sc probabilities + sc embedding ----
    model.eval()
    with torch.no_grad():
        z_fused, _z_proj_fused, logits_fused, _x_hat_rna, _x_hat_mod2, _z_rna, _z_mod2, _logits_rna = model.forward_srt(A_hat, X_rna, X_mod2)

        # full sc: identity graph is the most stable choice for publication scripts
        N = int(N_sc)
        ii = np.vstack([np.arange(N), np.arange(N)])
        A_sc = torch.sparse_coo_tensor(ii, np.ones(N, dtype=np.float32), (N, N), dtype=torch.float32).coalesce().to(device)
        X_sc_t = torch.from_numpy(X_sc_feat_cpu).to(device=device, non_blocking=True)
        z_sc, _z_proj_sc, logits_sc, _x_hat_sc = model.forward_sc(A_sc, X_sc_t)
        proba_sc = F.softmax(logits_sc, dim=1).detach().cpu().numpy().astype(np.float32)

    return CompassTrainResult(
        classes=[str(c) for c in classes],
        z_st=z_fused.detach().cpu().numpy().astype(np.float32),
        z_sc=z_sc.detach().cpu().numpy().astype(np.float32),
        proba_sc=proba_sc,
    )


