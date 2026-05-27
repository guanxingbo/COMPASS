# -*- coding: utf-8 -*-
"""
给已有的“空转 ATAC (spots × peaks) .h5ad”追加噪声并保存。

输入：一个 h5ad（通常是 simulation_atac.h5ad），要求至少包含：
- adata.X: (n_spots, n_peaks) counts 矩阵（稀疏/稠密均可）
- adata.obsm["spatial"]: (n_spots, 2) spot 坐标（用于空间梯度）

输出：一个新的 h5ad（默认在原文件名后加 .noisy.h5ad）
"""

import argparse
import numpy as np
import scipy.sparse as sp
import anndata as ad


def _as_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.array(X, dtype=np.float32, copy=True)


def add_spatial_gradient_poisson_noise_atac(
    adata: ad.AnnData,
    poisson_scale: float = 2.5,
    spatial_gradient_coeff: float = 1.8,
    zero_protection: bool = True,
    seed: int = 221,
) -> ad.AnnData:
    """
    空间梯度 Poisson（加性噪声）：
    - 距离中心越远，噪声越大（由 spatial_gradient_coeff 控制）
    - zero_protection=True 时只在原本非零位置加噪声（不造新 peak）
    """
    if poisson_scale <= 0:
        return adata

    if "spatial" not in adata.obsm:
        raise ValueError('输入 adata 缺少 obsm["spatial"]，无法计算空间梯度噪声。')

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError('obsm["spatial"] 形状应为 (n_spots, 2)（或至少 2 列）。')

    center = coords.mean(axis=0)
    dist = np.sqrt(((coords - center) ** 2).sum(axis=1))
    dist_max = float(dist.max()) if float(dist.max()) > 0 else 1.0
    normalized = dist / dist_max  # [0, 1]

    X = _as_dense_float32(adata.X)
    rng = np.random.default_rng(seed)

    # 每个 spot 一个噪声强度（scalar），对该 spot 的所有 peaks 采样 Poisson(s)
    spot_scales = poisson_scale * (1.0 + normalized * (spatial_gradient_coeff - 1.0))

    if zero_protection:
        non_zero_mask = (X > 0)
    else:
        non_zero_mask = None

    noise = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        s = float(spot_scales[i])
        if s <= 0:
            continue
        spot_noise = rng.poisson(lam=s, size=X.shape[1]).astype(np.float32)
        if non_zero_mask is not None:
            spot_noise[~non_zero_mask[i]] = 0.0
        noise[i] = spot_noise

    X_noisy = np.maximum(X + noise, 0.0)
    adata.X = sp.csr_matrix(X_noisy)  # 保持为稀疏 counts
    return adata


def add_atac_peak_dropout(
    adata: ad.AnnData,
    dropout_rate: float = 0.10,
    seed: int = 222,
) -> ad.AnnData:
    """
    peak dropout：随机把一部分非零项置 0（不新增 peak）。
    """
    if dropout_rate <= 0:
        return adata

    X = adata.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)

    X_coo = X.tocoo()
    nnz = X_coo.data.shape[0]
    if nnz == 0:
        adata.X = X
        return adata

    rng = np.random.default_rng(seed)
    keep = rng.random(nnz) >= float(dropout_rate)

    X_drop = sp.coo_matrix(
        (X_coo.data[keep], (X_coo.row[keep], X_coo.col[keep])),
        shape=X_coo.shape,
    ).tocsr()

    adata.X = X_drop
    return adata


def add_ambient_atac(
    adata: ad.AnnData,
    ambient_fraction: float = 0.03,
    seed: int = 223,
) -> ad.AnnData:
    """
    ambient ATAC：按全局均值 profile，给每个 spot 叠加 Poisson 背景。
    这一步可能会在原本为 0 的 peak 上引入少量 counts（更接近“背景污染”）。
    """
    if ambient_fraction <= 0:
        return adata

    X = _as_dense_float32(adata.X)

    mean_profile = X.mean(axis=0)
    mean_total = float(mean_profile.sum()) + 1e-8

    rng = np.random.default_rng(seed)

    for i in range(X.shape[0]):
        spot_total = float(X[i].sum())
        if spot_total < 1:
            continue
        ambient_scale = float(ambient_fraction) * spot_total / mean_total
        ambient = mean_profile * ambient_scale
        X[i] += rng.poisson(lam=np.maximum(ambient, 0.0)).astype(np.float32)

    X = np.maximum(X, 0.0)
    adata.X = sp.csr_matrix(X)
    return adata


def parse_args():
    p = argparse.ArgumentParser(
        description="为已有的空转 ATAC .h5ad 添加噪声（空间梯度Poisson + dropout + ambient）并保存。"
    )
    p.add_argument("--input", "-i", required=True, help="输入 ATAC .h5ad（spots × peaks）")
    p.add_argument("--output", "-o", default=None, help="输出 .h5ad（默认：input + .noisy.h5ad）")

    p.add_argument("--poisson-scale", type=float, default=2.5)
    p.add_argument("--spatial-gradient-coeff", type=float, default=1.8)
    p.add_argument("--zero-protection", action="store_true", help="只在非零 peak 上加 Poisson（不造新 peak）")

    p.add_argument("--dropout-rate", type=float, default=0.10)
    p.add_argument("--ambient-fraction", type=float, default=0.03)

    p.add_argument("--seed-poisson", type=int, default=221)
    p.add_argument("--seed-dropout", type=int, default=222)
    p.add_argument("--seed-ambient", type=int, default=223)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output
    if out is None:
        out = args.input + ".noisy.h5ad"

    adata = ad.read_h5ad(args.input)

    # 依次叠加噪声：梯度 Poisson -> dropout -> ambient
    adata = add_spatial_gradient_poisson_noise_atac(
        adata,
        poisson_scale=args.poisson_scale,
        spatial_gradient_coeff=args.spatial_gradient_coeff,
        zero_protection=args.zero_protection,
        seed=args.seed_poisson,
    )
    adata = add_atac_peak_dropout(
        adata,
        dropout_rate=args.dropout_rate,
        seed=args.seed_dropout,
    )
    adata = add_ambient_atac(
        adata,
        ambient_fraction=args.ambient_fraction,
        seed=args.seed_ambient,
    )

    adata.write_h5ad(out)
    print(f"✅ 输出完成: {out}")


if __name__ == "__main__":
    main()