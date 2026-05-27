# -*- coding: utf-8 -*-
"""
COMPASS_method: utilities extracted from COMPASS_method.ipynb.

Sinkhorn OT matching, domain×cell-type bucketing, soft/hard assignment,
spatial visualization helpers, and bar plots. Upstream training must
populate ``sc_adata.obsm['gae_latent']``, ``sc_adata.obs['pred_domain']``,
``sc_adata.obs['CellType']``, ``sc_adata.obsm['pred_domain_proba']``, and on
``adata_prep`` (ST): ``obs['domain']``, ``obsm['spatial']``, ``obsm['gae_latent']``.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Wedge, Patch
from matplotlib.collections import PatchCollection
from matplotlib.path import Path

import torch
import torch.nn.functional as F
import scanpy as sc
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree, ConvexHull

# ---------------------------------------------------------------------------
# 1) Sinkhorn OT — soft matching in embedding space
# ---------------------------------------------------------------------------
def ot_match_prob(
    X,
    Y,
    metric: str = "cosine",
    eps: float = 0.05,
    iters: int = 80,
    row_mass=None,
    col_mass=None,
    standardize: bool = False,
    device: str = None,
    return_torch: bool = True,
):
    """
    X: (N, D) cells/features
    Y: (M, D) spots/features
    Returns P: (N, M) row-stochastic soft matching probabilities.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    N, D = Xt.shape
    M, D2 = Yt.shape
    assert D == D2, "Feature dims must match"

    if standardize:
        Z = torch.cat([Xt, Yt], dim=0)
        mu = Z.mean(dim=0, keepdim=True)
        sd = Z.std(dim=0, keepdim=True) + 1e-6
        Xt = (Xt - mu) / sd
        Yt = (Yt - mu) / sd

    if metric.lower() == "cosine":
        Xt_n = F.normalize(Xt, dim=1)
        Yt_n = F.normalize(Yt, dim=1)
        C = 1.0 - torch.matmul(Xt_n, Yt_n.T)
    elif metric.lower() == "euclidean":
        x2 = (Xt**2).sum(dim=1, keepdim=True)
        y2 = (Yt**2).sum(dim=1, keepdim=True).T
        C = x2 + y2 - 2.0 * Xt @ Yt.T
        C = torch.clamp(C, min=0.0)
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    K = torch.exp(-C / eps).clamp(min=1e-12)

    if row_mass is None:
        r = torch.ones(N, device=device)
    else:
        r = torch.as_tensor(row_mass, dtype=torch.float32, device=device)
        assert r.shape == (N,)

    if col_mass is None:
        c = torch.full((M,), float(N) / M, device=device)
    else:
        c = torch.as_tensor(col_mass, dtype=torch.float32, device=device)
        assert c.shape == (M,)
        c = c * (r.sum() / (c.sum() + 1e-12))

    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iters):
        Kv = K @ v + 1e-12
        u = r / Kv
        KTu = K.T @ u + 1e-12
        v = c / KTu

    P = (u[:, None] * K) * v[None, :]
    if row_mass is None:
        P = P / (P.sum(dim=1, keepdim=True) + 1e-12)

    return P if return_torch else P.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 2) Domain × cell-type buckets → OT, hard assignment, spot×cell-type proportions
# ---------------------------------------------------------------------------
def map_cells_to_spots_dom_ct(
    sc_adata,
    st_adata,
    emb_key_sc="gae_latent",
    emb_key_st="gae_latent",
    sc_domain_key="pred_domain",
    st_domain_key="gt",
    sc_type_key="CellType",
    metric="cosine",
    eps=0.05,
    iters=200,
    standardize=False,
    assignment_mode="argmax",
    random_state=0,
    keep_P_full=False,
    store_topk=None,
    device=None,
):
    rng = np.random.default_rng(random_state)

    Z_sc = np.asarray(sc_adata.obsm[emb_key_sc], dtype=np.float32)
    Z_st = np.asarray(st_adata.obsm[emb_key_st], dtype=np.float32)

    sc_dom = sc_adata.obs[sc_domain_key].astype("category")
    st_dom = st_adata.obs[st_domain_key].astype("category")
    sc_ct = sc_adata.obs[sc_type_key].astype("category")

    sc_index = sc_adata.obs_names
    st_index = st_adata.obs_names

    domains = [d for d in sc_dom.cat.categories if d in set(st_dom.cat.categories)]
    cell_types = list(sc_ct.cat.categories)
    if len(domains) == 0:
        raise ValueError("No overlapping domain labels between sc_adata and st_adata.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    P_stats = {}
    assigned_spot_for_cell = pd.Series(index=sc_index, dtype="object")

    for d in domains:
        idx_st = np.where(st_dom.values == d)[0]
        if len(idx_st) == 0:
            continue
        Y = Z_st[idx_st]

        for t in cell_types:
            mask_sc = (sc_dom.values == d) & (sc_ct.values == t)
            idx_sc = np.where(mask_sc)[0]
            if len(idx_sc) == 0:
                continue

            X = Z_sc[idx_sc]

            P = ot_match_prob(
                X,
                Y,
                metric=metric,
                eps=eps,
                iters=iters,
                row_mass=None,
                col_mass=None,
                standardize=standardize,
                device=device,
                return_torch=True,
            )

            col_sum = P.sum(dim=0).detach().cpu().numpy().astype(np.float32, copy=False)

            entry = {"col_sum": col_sum, "idx_st": idx_st.astype(np.int64, copy=False)}
            if keep_P_full:
                entry["P"] = P
            if store_topk is not None and int(store_topk) > 0:
                k = int(min(store_topk, P.shape[1]))
                v, ii = torch.topk(P, k=k, dim=1)
                entry["topk_idx"] = ii.detach().cpu().numpy().astype(np.int32, copy=False)
                entry["topk_val"] = v.detach().cpu().numpy().astype(np.float16, copy=False)

            P_stats[(d, t)] = entry

            if assignment_mode == "balanced":
                Pn = P.detach().cpu().numpy().astype(np.float32, copy=False)
                Nc, Ns = Pn.shape
                q, r = divmod(Nc, Ns)
                caps = np.full(Ns, q, dtype=int)
                order = np.arange(Ns)
                caps[order[:r]] += 1
                col_ids = np.repeat(np.arange(Ns), caps)
                P_exp = Pn[:, col_ids]
                cost = -np.log(P_exp + 1e-12)
                row_ind, col_ind = linear_sum_assignment(cost)
                spot_local = col_ids[col_ind]
            else:
                spot_local = P.argmax(dim=1).detach().cpu().numpy()

            mapped_global = pd.Index(st_index[idx_st][spot_local])
            assigned_spot_for_cell.iloc[idx_sc] = mapped_global.values

            if not keep_P_full:
                del P
                if device == "cuda":
                    torch.cuda.empty_cache()

    assigned_spot_for_cell = assigned_spot_for_cell.dropna()

    df_map = pd.DataFrame(
        {
            "cell": assigned_spot_for_cell.index,
            "spot": assigned_spot_for_cell.values,
            "CellType": sc_ct.reindex(assigned_spot_for_cell.index).astype("category").astype(str),
        }
    )
    counts = df_map.groupby(["spot", "CellType"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=cell_types, fill_value=0)

    spot_totals = counts.sum(axis=1).replace(0, np.nan)
    spot_type_prop = counts.div(spot_totals, axis=0).fillna(0.0)
    spot_type_prop = spot_type_prop.reindex(st_index, fill_value=0.0)

    for ct in spot_type_prop.columns:
        st_adata.obs[f"prop_{ct}"] = spot_type_prop[ct].astype(np.float32)
    st_adata.obs["mapped_cell_count"] = counts.sum(axis=1).reindex(st_index).fillna(0).astype(int)

    return P_stats, assigned_spot_for_cell, spot_type_prop


def build_spot_type_prop_soft(P_stats, sc_adata, st_adata, sc_domain_key="pred_domain", st_domain_key="domain", sc_type_key="CellType"):
    st_dom = st_adata.obs[st_domain_key].astype("category")
    st_index = st_adata.obs_names
    cell_types = list(sc_adata.obs[sc_type_key].astype("category").cat.categories)

    out = pd.DataFrame(0.0, index=st_index, columns=cell_types)

    for (d, t), entry in P_stats.items():
        if d not in st_dom.cat.categories:
            continue
        idx_st = entry.get("idx_st", None)
        if idx_st is None:
            idx_st = np.where(st_dom.values == d)[0]

        if isinstance(entry, dict) and "col_sum" in entry:
            mass = entry["col_sum"]
        elif isinstance(entry, dict) and "P" in entry:
            P = entry["P"]
            mass = P.detach().cpu().numpy().sum(axis=0)
        else:
            P = entry
            mass = P.detach().cpu().numpy().sum(axis=0)

        out.loc[st_index[idx_st], t] += mass

    row_sum = out.sum(axis=1).replace(0, np.nan)
    return out.div(row_sum, axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# 3) Hungarian matching — capacity-constrained hard assignment within each domain
# ---------------------------------------------------------------------------
def _largest_remainder_rounding(x, total=None):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    if total is None:
        total = int(np.rint(x.sum()))
    base = np.floor(x).astype(int)
    rem = total - base.sum()
    if rem < 0:
        base = np.round(x).astype(int)
        diff = total - base.sum()
        if diff == 0:
            return base
        frac = x - np.floor(x)
        order = np.argsort(frac)
        ptr = 0
        while diff != 0 and ptr < len(order):
            j = order[ptr] if diff > 0 else order[-(ptr + 1)]
            base[j] += 1 if diff > 0 else -1
            diff = total - base.sum()
            ptr += 1
        return np.clip(base, 0, None)
    if rem == 0:
        return base
    frac = x - np.floor(x)
    order = np.argsort(-frac)
    base[order[:rem]] += 1
    return base


def hard_assign_cells_from_soft(
    P_full,
    sc_adata,
    st_adata,
    sc_domain_key="pred_domain",
    st_domain_key="gt",
    sc_type_key="CellType",
    p_eps=1e-12,
    random_state=0,
):
    rng = np.random.default_rng(random_state)

    st_dom = st_adata.obs[st_domain_key].astype("category")
    sc_dom = sc_adata.obs[sc_domain_key].astype("category")
    sc_ct = sc_adata.obs[sc_type_key].astype("category")

    st_index = st_adata.obs_names
    sc_index = sc_adata.obs_names

    domains = [d for d in sc_dom.cat.categories if d in set(st_dom.cat.categories)]

    assigned_spot = pd.Series(index=sc_index, dtype=object)

    for d in domains:
        idx_st = np.where(st_dom.values == d)[0]
        Ns = len(idx_st)
        if Ns == 0:
            continue

        rows_all = []
        idx_sc_all = []

        for t in sc_ct.cat.categories:
            key = (d, t)
            if key not in P_full:
                continue

            P_dict = P_full[key]

            if "P" in P_dict:
                P = P_dict["P"]
                if hasattr(P, "detach"):
                    Pn = P.detach().cpu().numpy()
                else:
                    Pn = np.asarray(P, dtype=float)
            elif "col_sum" in P_dict:
                col_sum = np.asarray(P_dict["col_sum"], dtype=float)
                mask_sc = (sc_dom.values == d) & (sc_ct.values == t)
                idx_sc = np.where(mask_sc)[0]
                Nc_type = len(idx_sc)
                if Nc_type == 0:
                    continue
                Pn = np.tile(col_sum / Nc_type, (Nc_type, 1))
                row_sums = Pn.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                Pn = Pn / row_sums
            else:
                continue

            mask_sc = (sc_dom.values == d) & (sc_ct.values == t)
            idx_sc = np.where(mask_sc)[0]

            if Pn.shape[0] != len(idx_sc) or Pn.shape[1] != Ns:
                if Pn.shape[0] < len(idx_sc):
                    repeat_times = len(idx_sc) // Pn.shape[0] + 1
                    Pn = np.tile(Pn, (repeat_times, 1))[: len(idx_sc)]
                elif Pn.shape[0] > len(idx_sc):
                    Pn = Pn[: len(idx_sc)]
                if Pn.shape[1] < Ns:
                    padding = np.zeros((Pn.shape[0], Ns - Pn.shape[1]))
                    Pn = np.hstack([Pn, padding])
                elif Pn.shape[1] > Ns:
                    Pn = Pn[:, :Ns]

            rows_all.append(Pn)
            idx_sc_all.append(idx_sc)

        if not rows_all:
            continue

        P_all = np.vstack(rows_all)
        idx_sc_all = np.concatenate(idx_sc_all)
        Nc_dom = P_all.shape[0]

        col_mass = P_all.sum(axis=0)
        if col_mass.sum() <= 0:
            caps_total = _largest_remainder_rounding(np.ones(Ns) * (Nc_dom / max(Ns, 1)), total=Nc_dom)
        else:
            caps_total = _largest_remainder_rounding(col_mass, total=Nc_dom)

        col_ids = np.repeat(np.arange(Ns), caps_total)
        if col_ids.size != Nc_dom:
            raise RuntimeError("Capacity expansion size mismatch.")

        P_exp = P_all[:, col_ids]
        cost = -np.log(P_exp + p_eps)
        row_ind, col_ind = linear_sum_assignment(cost)
        spot_local = col_ids[col_ind]

        mapped_global = pd.Index(st_index[idx_st][spot_local])
        assigned_spot.iloc[idx_sc_all] = mapped_global.values

    assigned_spot = assigned_spot.dropna()
    return assigned_spot


# ---------------------------------------------------------------------------
# 4) Poisson-disk sampling — synthetic cell coordinates (notebook-aligned)
# ---------------------------------------------------------------------------
def poisson_disk_in_region(inside_fn, mins, maxs, r, k=30, seed=0):
    rng = np.random.default_rng(seed)
    cell = r / np.sqrt(2)
    nx = int(np.ceil((maxs[0] - mins[0]) / cell))
    ny = int(np.ceil((maxs[1] - mins[1]) / cell))
    grid = -np.ones((nx, ny), dtype=int)

    def gcoords(p):
        return (np.floor((p - mins) / cell)).astype(int)

    for _ in range(2000):
        p0 = rng.uniform(mins, maxs)
        if inside_fn(p0):
            break
    else:
        raise RuntimeError("Failed to seed inside support region.")

    samples = [p0]
    active = [0]
    gx, gy = gcoords(p0)
    grid[gx, gy] = 0

    def valid(p):
        if not inside_fn(p):
            return False
        gx, gy = gcoords(p)
        if gx < 0 or gy < 0 or gx >= nx or gy >= ny:
            return False
        i0, i1 = max(gx - 2, 0), min(gx + 3, nx)
        j0, j1 = max(gy - 2, 0), min(gy + 3, ny)
        for ix in range(i0, i1):
            for iy in range(j0, j1):
                s = grid[ix, iy]
                if s == -1:
                    continue
                if np.linalg.norm(samples[s] - p) < r:
                    return False
        return True

    while active:
        i = rng.choice(active)
        c = samples[i]
        found = False
        for _ in range(k):
            rad = rng.uniform(r, 2 * r)
            ang = rng.uniform(0, 2 * np.pi)
            cand = c + rad * np.array([np.cos(ang), np.sin(ang)])
            if valid(cand):
                samples.append(cand)
                active.append(len(samples) - 1)
                gx, gy = gcoords(cand)
                grid[gx, gy] = len(samples) - 1
                found = True
                break
        if not found:
            active.remove(i)
    return np.asarray(samples, dtype=np.float32)


def _parse_xy_str(arr_like):
    s = pd.Series(arr_like, dtype=str).str.strip().str.lower().str.replace("×", "x", regex=False)
    xy = np.vstack(s.str.split("x", n=1, expand=True).astype(float).to_numpy())
    return xy


def assign_cell_locations_from_coord_strings(
    assigned_spot_for_cell,
    st_adata=None,
    min_dist=None,
    k_nn=8,
    seed=0,
    support_mode: str = "union_disks",
    support_radius_factor: float = 1.1,
    clip_to_spot: bool = False,
    clip_radius_factor: float = 1.3,
    clip_radius_abs=None,
):
    rng = np.random.default_rng(seed)

    spot_str = assigned_spot_for_cell.astype(str).values
    unique_spots, inverse = np.unique(spot_str, return_inverse=True)
    cap = np.bincount(inverse)
    S = len(unique_spots)
    N = len(spot_str)

    if st_adata is not None:
        XY_spot = np.zeros((S, 2), dtype=np.float32)
        for i, spot in enumerate(unique_spots):
            if spot in st_adata.obs_names:
                idx = st_adata.obs_names.get_loc(spot)
                if "x" in st_adata.obs.columns and "y" in st_adata.obs.columns:
                    XY_spot[i, 0] = float(st_adata.obs["x"].iloc[idx])
                    XY_spot[i, 1] = float(st_adata.obs["y"].iloc[idx])
                elif "spatial" in st_adata.obsm:
                    XY_spot[i, :] = st_adata.obsm["spatial"][idx, :2]
                elif "X" in st_adata.obs.columns and "Y" in st_adata.obs.columns:
                    XY_spot[i, 0] = float(st_adata.obs["X"].iloc[idx])
                    XY_spot[i, 1] = float(st_adata.obs["Y"].iloc[idx])
                else:
                    XY_spot[i, 0] = i % 100
                    XY_spot[i, 1] = i // 100
            else:
                XY_spot[i, 0] = i % 100
                XY_spot[i, 1] = i // 100
    else:
        try:
            XY_spot = _parse_xy_str(unique_spots)
        except Exception:
            raise ValueError(
                "Could not parse spot coordinates; pass st_adata or use spot ids as 'x'/'y' strings."
            )

    if S > 1:
        tree_spot = KDTree(XY_spot)
        dists, _ = tree_spot.query(XY_spot, k=min(4, S))
        local_scale = dists[:, 1] if dists.shape[1] >= 2 else np.full(S, np.median(dists))
        global_scale = np.median(local_scale) if S > 1 else (np.max(XY_spot, axis=0) - np.min(XY_spot, axis=0)).mean()
    else:
        local_scale = np.array([1.0])
        global_scale = 1.0

    if support_mode == "convex_hull" or S < 3:
        if S >= 3:
            hull = ConvexHull(XY_spot)
            poly = XY_spot[hull.vertices]
            Ppoly = Path(poly)
            mins = poly.min(axis=0)
            maxs = poly.max(axis=0)

            def inside_fn(p):
                return Ppoly.contains_point(p)

        else:
            mins = XY_spot.min(axis=0)
            maxs = XY_spot.max(axis=0)
            pad = 0.05 * (maxs - mins + 1e-6)
            mins = mins - pad
            maxs = maxs + pad

            def inside_fn(p):
                return (mins[0] <= p[0] <= maxs[0]) and (mins[1] <= p[1] <= maxs[1])

    elif support_mode == "union_disks":
        R = support_radius_factor * global_scale
        mins = XY_spot.min(axis=0) - R
        maxs = XY_spot.max(axis=0) + R
        if S > 1:
            tree_spot = KDTree(XY_spot)

            def inside_fn(p):
                dist, _ = tree_spot.query(p, k=1)
                return dist <= R

        else:

            def inside_fn(p):
                return np.linalg.norm(p - XY_spot[0]) <= R

    else:
        raise ValueError(f"Unknown support_mode: {support_mode}")

    if min_dist is None:
        if support_mode == "convex_hull" and S >= 3:
            area = ConvexHull(XY_spot).volume
            min_dist = 0.9 * np.sqrt(area / (N * np.pi))
        else:
            min_dist = 0.6 * global_scale

    pts = poisson_disk_in_region(inside_fn, mins, maxs, r=min_dist, k=30, seed=seed)
    tries = 0
    while pts.shape[0] < N and tries < 5:
        min_dist *= 0.9
        pts = poisson_disk_in_region(inside_fn, mins, maxs, r=min_dist, k=30, seed=seed + tries + 1)
        tries += 1
    if pts.shape[0] > N:
        pts = pts[rng.choice(pts.shape[0], size=N, replace=False)]
    elif pts.shape[0] < N:
        extra = []
        for _ in range(200000):
            q = rng.uniform(mins, maxs)
            if inside_fn(q):
                extra.append(q)
            if len(extra) + pts.shape[0] >= N:
                break
        if extra:
            pts = np.vstack([pts, np.asarray(extra, dtype=np.float32)])

    if S > 1:
        tree = KDTree(XY_spot)
        kq = min(k_nn, S)
        _, knn_idx = tree.query(pts, k=kq)
        if knn_idx.ndim == 1:
            knn_idx = knn_idx[:, None]
    else:
        knn_idx = np.zeros((len(pts), 1), dtype=int)

    remaining = cap.copy()
    assigned_point_to_spot = -np.ones(N, dtype=int)
    for p in rng.permutation(N):
        for j in knn_idx[p]:
            if remaining[j] > 0:
                assigned_point_to_spot[p] = j
                remaining[j] -= 1
                break
        if assigned_point_to_spot[p] < 0:
            order = np.argsort(np.linalg.norm(XY_spot - pts[p], axis=1))
            for j in order:
                if remaining[j] > 0:
                    assigned_point_to_spot[p] = j
                    remaining[j] -= 1
                    break
    if (assigned_point_to_spot < 0).any() or remaining.sum() != 0:
        raise RuntimeError("Capacity assignment failed; try larger k_nn or smaller min_dist.")

    if clip_to_spot:
        if clip_radius_abs is not None:
            r_clip = np.full(S, float(clip_radius_abs), dtype=float)
        else:
            r_clip = clip_radius_factor * local_scale
        delta = pts - XY_spot[assigned_point_to_spot]
        dist = np.linalg.norm(delta, axis=1) + 1e-12
        r_allowed = r_clip[assigned_point_to_spot]
        over = dist > r_allowed
        if np.any(over):
            scale = (r_allowed[over] / dist[over])[:, None]
            pts[over] = XY_spot[assigned_point_to_spot[over]] + delta[over] * scale

    cell_xy = np.zeros((N, 2), dtype=np.float32)
    per_spot_points = {j: [] for j in range(S)}
    for p, j in enumerate(assigned_point_to_spot):
        per_spot_points[j].append(p)
    for j in range(S):
        pts_j = np.asarray(per_spot_points[j], int)
        cells_j = np.where(inverse == j)[0]
        m = min(len(pts_j), len(cells_j))
        perm = rng.permutation(m)
        cell_xy[cells_j[:m]] = pts[pts_j[perm]]

    return pd.DataFrame(cell_xy, index=assigned_spot_for_cell.index, columns=["x", "y"])


# ---------------------------------------------------------------------------
# 5) Figures: spatial pies, cell scatter, stacked bars
# ---------------------------------------------------------------------------
def plot_spatial_pies(
    st_adata,
    spot_type_prop,
    coord_key="spatial",
    radius=15,
    alpha=0.95,
    max_types_per_spot=None,
    celltype_colors=None,
    background="white",
    edgecolor="none",
    linewidth=0.0,
    legend=True,
    figsize=(8, 10),
):
    coords = np.asarray(st_adata.obsm[coord_key])[:, :2]
    spot_index = st_adata.obs_names
    prop = spot_type_prop.reindex(spot_index).fillna(0.0).copy()
    s = prop.sum(axis=1).replace(0, np.nan)
    prop = prop.div(s, axis=0).fillna(0.0)
    cell_types = prop.columns.tolist()

    if celltype_colors is None:
        cmap = plt.get_cmap("tab20", len(cell_types))
        celltype_colors = {ct: cmap(i) for i, ct in enumerate(cell_types)}
    colors = [celltype_colors[ct] for ct in cell_types]

    def wedges_for_row(center, proportions):
        vals = proportions.values.astype(float)
        if max_types_per_spot is not None and max_types_per_spot < len(vals):
            topk = np.argpartition(-vals, max_types_per_spot - 1)[:max_types_per_spot]
            mask = np.zeros_like(vals, dtype=bool)
            mask[topk] = True
            vals = np.where(mask, vals, 0.0)
            total = vals.sum()
            vals = vals / total if total > 0 else vals
        wedges = []
        start = 0.0
        for frac, col in zip(vals, colors):
            if frac <= 0:
                continue
            theta = frac * 360.0
            wedges.append(
                Wedge(
                    center,
                    r=radius,
                    theta1=start,
                    theta2=start + theta,
                    facecolor=col,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                )
            )
            start += theta
        return wedges

    patches = []
    for (x, y), (_, row) in zip(coords, prop.iterrows()):
        patches.extend(wedges_for_row((x, y), row))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(background)
    pc = PatchCollection(patches, match_original=True, alpha=alpha)
    ax.add_collection(pc)

    xmin, ymin = coords.min(axis=0) - radius * 1.2
    xmax, ymax = coords.max(axis=0) + radius * 1.2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Spot-wise cell-type proportions (pie charts)")

    if legend:
        handles = [Patch(facecolor=celltype_colors[ct], label=ct) for ct in cell_types]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, title="Cell type")

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_cells_spatial(
    sc_adata,
    df_xy,
    color_keys=("pred_domain", "CellType"),
    s=6,
    alpha=0.85,
    figsize=(6, 5),
    title_prefix="",
):
    idx = sc_adata.obs_names.intersection(df_xy.index)
    if len(idx) == 0:
        raise ValueError("No overlapping cell IDs between sc_adata and df_xy.")
    xy = df_xy.loc[idx, ["x", "y"]].astype(float).to_numpy()

    def _plot_one(color_key):
        if color_key not in sc_adata.obs:
            raise KeyError(f"'{color_key}' not found in sc_adata.obs")
        cats = sc_adata.obs.loc[idx, color_key].astype("category")
        categories = list(cats.cat.categories)
        base = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors)
        if len(categories) > len(base):
            rep = int(np.ceil(len(categories) / len(base)))
            palette = (base * rep)[: len(categories)]
        else:
            palette = base[: len(categories)]
        color_map = dict(zip(categories, palette))
        plt.figure(figsize=figsize)
        for cat in categories:
            m = (cats == cat).to_numpy()
            if not m.any():
                continue
            plt.scatter(xy[m, 0], xy[m, 1], s=s, alpha=alpha, c=[color_map[cat]], label=str(cat), edgecolors="none")
        plt.gca().set_aspect("equal", adjustable="datalim")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{title_prefix}{color_key}")
        ncat = len(categories)
        if ncat <= 12:
            plt.legend(frameon=False, markerscale=2)
        else:
            plt.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, fontsize=8)
        plt.tight_layout()
        plt.show()

    for key in color_keys:
        _plot_one(key)


def plot_celltype_by_pred_domain(
    adata,
    ct_key="CellType",
    dom_key="pred_domain",
    sort_by_total=True,
    figsize=(8, 4),
    annotate=False,
):
    if ct_key not in adata.obs or dom_key not in adata.obs:
        raise KeyError(f"'{ct_key}' or '{dom_key}' not found in adata.obs")

    ct = adata.obs[ct_key].astype("category")
    dom = adata.obs[dom_key].astype("category")
    M = pd.crosstab(ct, dom).astype(int)
    totals = M.sum(axis=1)
    if sort_by_total:
        M = M.loc[totals.sort_values(ascending=False).index]
        totals = totals.loc[M.index]

    K = M.shape[1]
    cmap = ListedColormap(plt.cm.tab20.colors[:K])
    dom_colors = {d: cmap(i) for i, d in enumerate(M.columns)}

    ax = plt.figure(figsize=figsize).gca()
    bottom = np.zeros(M.shape[0], dtype=float)
    x = np.arange(M.shape[0])
    for j, d in enumerate(M.columns):
        vals = M[d].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=dom_colors[d], label=str(d), width=0.8, linewidth=0)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in M.index], rotation=45, ha="right")
    ax.set_ylabel("Number of cells")
    ax.set_title(f"Cells per {ct_key} colored by {dom_key}")
    ax.legend(title=dom_key, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax.set_xlim(-0.6, len(x) - 0.4)

    if annotate:
        for i, total in enumerate(totals):
            ax.text(i, total + max(totals) * 0.01, str(total), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()


__all__ = [
    "ot_match_prob",
    "map_cells_to_spots_dom_ct",
    "build_spot_type_prop_soft",
    "hard_assign_cells_from_soft",
    "poisson_disk_in_region",
    "assign_cell_locations_from_coord_strings",
    "plot_spatial_pies",
    "plot_cells_spatial",
    "plot_celltype_by_pred_domain",
]

