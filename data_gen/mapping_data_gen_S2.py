# -*- coding: utf-8 -*-
"""
空间转录组模拟 V2：多组学互补设计

与 V1 的核心差异：
1. 引入"RNA 易混淆细胞类型对"机制 —— 在 RNA 空间降低特定细胞类型对的区分度
2. 保持 ATAC 数据干净 —— ATAC 保留完整区分能力
3. 增大混合复杂度 —— 每个 spot 更多细胞类型、更低主导比例
4. 显著提高 RNA 特异性噪声 —— Poisson + 环境 RNA + HVG 稀释

设计目标：多组学 > 单组学 > 其他方法（如 Seurat）
"""

import math
import os
import random
from typing import Optional, List, Union, Dict, Tuple

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import scanpy as sc


# ====================== 基础空间图案生成函数（与 V1 相同）======================

def conway_maxwell_poisson(lambda_: int, nu: float, seed: Optional[int] = None) -> int:
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)

    lambda_ = int(lambda_)
    nu = float(nu)
    C = np.sum([(pow(lambda_, k) / math.factorial(k)) ** nu for k in range(1000)])
    u, sum_p, k = np.random.rand(), 0, 0
    while sum_p < u:
        sum_p += (pow(lambda_, k) / math.factorial(k)) ** nu / C
        k += 1
    return k - 1


def squares(base_size: int = 12) -> np.ndarray:
    A = np.zeros([base_size, base_size])
    offset = base_size // 6
    size = base_size // 3
    A[offset:offset + size, offset:offset + size] = 1
    A[offset + size * 2:offset + size * 3, offset:offset + size] = 1
    A[offset:offset + size, offset + size * 2:offset + size * 3] = 1
    A[offset + size * 2:offset + size * 3, offset + size * 2:offset + size * 3] = 1
    return A


def corners(base_size: int = 12) -> np.ndarray:
    B = np.zeros([base_size // 2, base_size // 2])
    for i in range(base_size // 2):
        B[i, i:] = 1
    A = np.flip(B, axis=1)
    AB = np.hstack((A, B))
    CD = np.flip(AB, axis=0)
    return np.vstack((AB, CD))


def scotland(base_size: int = 12) -> np.ndarray:
    A = np.eye(base_size)
    for i in range(base_size):
        A[-i - 1, i] = 1
    return A


def checkers(base_size: int = 12) -> np.ndarray:
    unit = base_size // 3
    A = np.zeros([unit, unit])
    B = np.ones([unit, unit])
    AB = np.hstack((A, B, A))
    BA = np.hstack((B, A, B))
    return np.vstack((AB, BA, AB))


def rings(base_size: int = 12) -> np.ndarray:
    A = np.zeros([base_size, base_size])
    center = base_size // 2
    inner_radius = base_size // 4
    outer_radius = base_size // 2 - 1
    y, x = np.ogrid[:base_size, :base_size]
    dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    A[(dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)] = 1
    return A


def gen_spatial_factors(shapes: List[str], base_size: int = 12) -> tuple:
    shape_funcs = {
        "squares": squares, "corners": corners, "scotland": scotland,
        "checkers": checkers, "rings": rings,
    }
    assert len(shapes) == 4
    shape1 = shape_funcs[shapes[0]](base_size=base_size)
    shape2 = shape_funcs[shapes[1]](base_size=base_size)
    shape3 = shape_funcs[shapes[2]](base_size=base_size)
    shape4 = shape_funcs[shapes[3]](base_size=base_size)

    total_size = base_size * 2
    region_matrix = np.zeros((total_size, total_size), dtype=int)
    region_mapping = {
        (0, 0): 4, (0, 1): 5, (1, 0): 6, (1, 1): 7,
        (2, 0): 0, (2, 1): 1, (3, 0): 2, (3, 1): 3,
    }
    quadrants = [
        (shape1, 0, 0), (shape2, 0, base_size),
        (shape3, base_size, 0), (shape4, base_size, base_size),
    ]
    for q_idx, (q_matrix, x_offset, y_offset) in enumerate(quadrants):
        for i in range(base_size):
            for j in range(base_size):
                region_matrix[x_offset + i, y_offset + j] = region_mapping[(q_idx, int(q_matrix[i, j]))]

    top_row = np.hstack((shape1, shape2))
    bottom_row = np.hstack((shape3, shape4))
    full_pattern = np.vstack((top_row, bottom_row))

    F = np.zeros((total_size * total_size, 8))
    for r_id in range(8):
        mask = region_matrix.flatten() == r_id
        F[mask, r_id] = 1
    return F, full_pattern, region_matrix


# ====================== 采样器类（V2 修改：支持显式 region→celltype 映射）======================

class SamplerV2:
    """
    与 V1 Sampler 的区别：
    - 新增 region_cell_types 参数，可显式控制每个 region 包含哪些细胞类型
    - 当提供 region_cell_types 时，忽略 cell_type_number 的随机分配逻辑
    """

    def __init__(
            self,
            reference: Union[mu.MuData, ad.AnnData],
            cell_type_key: str,
            num_spots: int,
            cell_number_mean: Union[int, list] = None,
            cell_number_nu: Union[float, list] = 20.0,
            cell_type_number: Union[int, list] = None,
            region_cell_types: Optional[Dict[int, List[str]]] = None,
            balance: Optional[str] = "balanced",
            poisson_noise_scale: float = 1.0,
            structured_shapes: List[str] = None,
            structured_base_size: int = 12,
            random_seed: int = 221,
    ):
        if structured_shapes is None:
            structured_shapes = ["squares", "corners", "scotland", "checkers"]
        if cell_number_mean is None:
            cell_number_mean = [10] * 8
        if cell_type_number is None:
            cell_type_number = [4] * 8

        self.reference = reference
        self.cell_type_key = cell_type_key
        self.num_spots = num_spots
        self.obs = reference.obs if isinstance(reference, ad.AnnData) else reference[list(reference.mod.keys())[0]].obs
        self.poisson_noise_scale = poisson_noise_scale
        self.structured_shapes = structured_shapes
        self.structured_base_size = structured_base_size
        self.n_regions = 8
        self.random_seed = random_seed
        self.region_cell_types = region_cell_types

        expected_num_spots = (structured_base_size * 2) ** 2
        assert self.num_spots == expected_num_spots

        self.cell_number_mean = np.array(cell_number_mean) if not isinstance(cell_number_mean, np.ndarray) else cell_number_mean
        self.cell_number_nu = np.ones(8) * cell_number_nu if isinstance(cell_number_nu, (int, float)) else np.array(cell_number_nu)
        self.cell_type_number = np.array(cell_type_number) if not isinstance(cell_type_number, np.ndarray) else cell_type_number

        if balance not in ["balanced", "unbalanced"]:
            raise ValueError('balance must be one of ["balanced", "unbalanced"].')
        self.init_sample_prob(balance=balance)

    def init_sample_prob(self, balance="unbalanced"):
        cell_counts = self.obs[self.cell_type_key].value_counts(normalize=True)
        if balance == "unbalanced":
            self.cluster_p = cell_counts
            self.cell_p = self.obs[self.cell_type_key].map(self.cluster_p).astype(float)
        elif balance == "balanced":
            self.cluster_p = pd.Series(1 / len(cell_counts), index=cell_counts.index)
            self.cell_p = 1 / self.obs[self.cell_type_key].map(cell_counts).astype(float) / len(cell_counts)
        self.clusters = self.obs[self.cell_type_key].cat.categories
        self.cluster_p = self.cluster_p[self.clusters]

    def define_regions(self):
        F, full_pattern, region_matrix = gen_spatial_factors(
            shapes=self.structured_shapes, base_size=self.structured_base_size
        )
        self.regions = region_matrix.flatten()
        self.region_matrix = region_matrix

    def sample_data(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        if self.region_cell_types is not None:
            used_clusters = {
                rid: np.array(cts) for rid, cts in self.region_cell_types.items()
            }
        else:
            all_clusters = self.cluster_p.index.tolist()
            np.random.shuffle(all_clusters)
            used_clusters = {}
            current_idx = 0
            for region_id in range(self.n_regions):
                n_needed = self.cell_type_number[region_id]
                if current_idx + n_needed <= len(all_clusters):
                    selected = all_clusters[current_idx:current_idx + n_needed]
                    current_idx += n_needed
                else:
                    remaining = len(all_clusters) - current_idx
                    selected = all_clusters[current_idx:] + all_clusters[:n_needed - remaining]
                    current_idx = n_needed - remaining
                used_clusters[region_id] = np.array(selected)

        self.define_regions()

        cell_count = []
        for idx, region_id in enumerate(self.regions):
            cell_num = conway_maxwell_poisson(
                self.cell_number_mean[region_id],
                self.cell_number_nu[region_id],
                seed=self.random_seed + idx,
            )
            cell_count.append(max(cell_num, 1))
        cell_count = np.array(cell_count)

        used_clusters_list = [used_clusters[region_id] for region_id in self.regions]
        params = list(zip(cell_count, used_clusters_list, self.regions))
        return self.sample_spots(params)

    def sample_spots(self, params):
        sample_exp = {"tmp": self.reference} if isinstance(self.reference, ad.AnnData) else self.reference.mod
        exp = {key: np.zeros((len(params), adata.shape[1])) for key, adata in sample_exp.items()}
        density = np.zeros((len(params), len(self.clusters)))
        sampled_cells_df = []

        for i, (num_cell, used_clusters, region_id) in enumerate(params):
            np.random.seed(self.random_seed + i)
            cluster_mask = self.obs[self.cell_type_key].isin(used_clusters).values
            p = self.cell_p[cluster_mask] / self.cell_p[cluster_mask].sum()
            sampled_cells = np.random.choice(self.obs.index[cluster_mask], size=num_cell, p=p)
            for key, adata in sample_exp.items():
                exp[key][i, :] = adata[sampled_cells, :].X.sum(axis=0)
            density[i, :] = self.obs.loc[sampled_cells, self.cell_type_key].value_counts().reindex(
                self.clusters, fill_value=0
            ).values
            sampled_cells_df.append({
                "region_id": [region_id] * num_cell,
                "cell_id": sampled_cells.tolist(),
                "cell_type": self.obs.loc[sampled_cells, self.cell_type_key].values.tolist(),
            })

        for key in exp.keys():
            exp[key] = add_poisson_noise(exp[key], self.poisson_noise_scale, seed=self.random_seed)
        return exp, density, pd.DataFrame(sampled_cells_df)

    def get_coords(self):
        grid_size = int(np.sqrt(self.num_spots))
        x = np.arange(0, grid_size)
        y = np.arange(0, grid_size)
        X, Y = np.meshgrid(x, y)
        return X, Y


# ====================== 噪声/工具函数（与 V1 相同）======================

def add_poisson_noise(data, noise_scale=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    data_arr = data.toarray() if scipy.sparse.issparse(data) else np.array(data)
    lambda_ = np.maximum(data_arr * noise_scale, 1e-6)
    noise = np.random.poisson(lambda_)
    noisy_data = data_arr + noise
    return scipy.sparse.csr_matrix(np.maximum(noisy_data, 0).astype(int))


def add_spatial_gradient_poisson_noise(mdata: mu.MuData, noise_config: dict) -> mu.MuData:
    main_seed = noise_config["noise_seed"]
    ref_mod = list(mdata.mod.keys())[0]
    spatial_coords = mdata[ref_mod].obsm['spatial'].copy()
    center = spatial_coords.mean(axis=0)
    distances = np.sqrt(np.sum((spatial_coords - center) ** 2, axis=1))
    dist_max = distances.max() if distances.max() > 0 else 1
    normalized_dist = distances / dist_max

    for mod_name in mdata.mod.keys():
        if mod_name not in noise_config:
            continue
        cfg = noise_config[mod_name]
        if not cfg["enable_noise"] or cfg["poisson_scale"] == 0.0:
            continue
        adata = mdata[mod_name]
        exp_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X.copy()
        np.random.seed(main_seed + hash(mod_name) % 1000)
        gradient_coeff = cfg.get("spatial_gradient_coeff", 3.0)
        spot_noise_scales = cfg["poisson_scale"] * (1 + normalized_dist * (gradient_coeff - 1))
        non_zero_mask = (exp_matrix > 0)
        noise = np.zeros_like(exp_matrix)
        for spot_idx in range(exp_matrix.shape[0]):
            current_scale = spot_noise_scales[spot_idx]
            if current_scale <= 0:
                continue
            spot_noise = np.random.poisson(current_scale, size=exp_matrix.shape[1])
            if cfg.get("zero_protection", False):
                spot_noise[~non_zero_mask[spot_idx]] = 0
            noise[spot_idx] = spot_noise
        exp_noisy = exp_matrix + noise
        mdata[mod_name].X = scipy.sparse.csr_matrix(np.maximum(exp_noisy, 0))
    return mdata


def compute_cell_type_hvgs(adata_ref, cell_type_col, top_n=200, min_disp=0.5):
    cell_type_hvgs = {}
    cell_types = adata_ref.obs[cell_type_col].cat.categories
    for ct in cell_types:
        adata_ct = adata_ref[adata_ref.obs[cell_type_col] == ct].copy()
        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)
        sc.pp.highly_variable_genes(adata_ct, min_mean=0.0125, max_mean=3, min_disp=min_disp, n_top_genes=top_n)
        hvgs = adata_ct.var_names[adata_ct.var.highly_variable].tolist()
        cell_type_hvgs[ct] = hvgs
    return cell_type_hvgs


def shuffle_cell_type_hvgs(mdata, shuffle_config, cell_type_hvgs, dominant_cell_type_col="gt"):
    if not shuffle_config["enable_shuffle"]:
        return mdata
    target_mods = shuffle_config["target_modality"]
    if isinstance(target_mods, str):
        target_mods = [target_mods]
    np.random.seed(shuffle_config["shuffle_seed"])
    dilution_factor = shuffle_config["hvg_dilution_factor"]
    for mod_name in target_mods:
        if mod_name not in mdata.mod:
            continue
        adata = mdata[mod_name]
        exp_matrix = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X.copy()
        n_spots = exp_matrix.shape[0]
        feature_names = adata.var_names.tolist()
        n_shuffle_spots = int(n_spots * shuffle_config["spot_proportion"])
        shuffle_spot_idx = np.random.choice(n_spots, n_shuffle_spots, replace=False)
        ref_mod = list(mdata.mod.keys())[0]
        dominant_cell_types = mdata[ref_mod].obs[dominant_cell_type_col].values
        for spot_idx in shuffle_spot_idx:
            dom_ct = dominant_cell_types[spot_idx]
            if dom_ct not in cell_type_hvgs or len(cell_type_hvgs[dom_ct]) == 0:
                continue
            hvgs = cell_type_hvgs[dom_ct]
            n_dilute = int(len(hvgs) * 0.8)
            if n_dilute <= 0:
                continue
            selected_hvgs = np.random.choice(hvgs, size=n_dilute, replace=False).tolist()
            hvg_indices = [feature_names.index(hvg) for hvg in selected_hvgs if hvg in feature_names]
            if not hvg_indices:
                continue
            exp_matrix[spot_idx, hvg_indices] = exp_matrix[spot_idx, hvg_indices] * dilution_factor
        mdata[mod_name].X = scipy.sparse.csr_matrix(np.maximum(exp_matrix, 0))
    return mdata


def get_spots_in_regions(coords, regions, spot_names):
    target_spots = []
    for (cx, cy, size) in regions:
        x_min, x_max = cx - size / 2, cx + size / 2
        y_min, y_max = cy - size / 2, cy + size / 2
        in_region = (
            (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
            (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
        )
        target_spots.extend([spot_names[i] for i in np.where(in_region)[0]])
    target_spots = list(set(target_spots))
    target_spots.sort(key=lambda x: spot_names.index(x))
    return target_spots


def sample_cell_coords_in_circle(cx, cy, radius, size, rng):
    coords = np.zeros((size, 2))
    n = 0
    while n < size:
        x = rng.uniform(cx - radius, cx + radius)
        y = rng.uniform(cy - radius, cy + radius)
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
            coords[n, 0], coords[n, 1] = x, y
            n += 1
    return coords


def build_cell_level_data(ref_adata_dict, spot_names, spatial_coords, spot_to_cells,
                          cell_type_col, spot_radius=5.0, cell_count_min=7,
                          cell_count_max=13, seed=221):
    rng = np.random.default_rng(seed)
    all_cell_ids, all_spot_ids, all_cell_types, all_coords = [], [], [], []
    for i, spot_name in enumerate(spot_names):
        cx, cy = float(spatial_coords[i, 0]), float(spatial_coords[i, 1])
        cells = spot_to_cells.get(spot_name, [])
        if len(cells) == 0:
            continue
        n_want = int(rng.integers(cell_count_min, cell_count_max + 1))
        chosen = rng.choice(cells, size=n_want, replace=(n_want > len(cells))).tolist()
        coords = sample_cell_coords_in_circle(cx, cy, spot_radius, len(chosen), rng)
        all_cell_ids.extend(chosen)
        all_spot_ids.extend([spot_name] * len(chosen))
        all_coords.append(coords)
        ref_obs = ref_adata_dict[list(ref_adata_dict.keys())[0]].obs
        for c in chosen:
            all_cell_types.append(ref_obs.loc[c, cell_type_col])
    all_coords = np.vstack(all_coords)
    mod_to_adata = {}
    for mod_name, ref_adata in ref_adata_dict.items():
        X_cell = ref_adata[all_cell_ids, :].X
        cell_index = [f"{sid}_{k}" for k, sid in enumerate(all_spot_ids)]
        obs = pd.DataFrame({"spot_id": all_spot_ids, "cell_type": all_cell_types}, index=cell_index)
        adata_cell = ad.AnnData(X_cell, obs=obs, var=ref_adata.var.copy())
        adata_cell.obsm["spatial"] = all_coords
        mod_to_adata[mod_name] = adata_cell
    return mod_to_adata


def generate_spatial_data_v2(
        reference, cell_type_key, num_spots=576,
        balance=None, cell_number_mean=None, cell_number_nu=20.0,
        cell_type_number=None,
        region_cell_types=None,
        poisson_noise_scale=1.0,
        structured_shapes=None, structured_base_size=12,
        random_seed=0,
) -> tuple:
    if structured_shapes is None:
        structured_shapes = ["squares", "corners", "scotland", "checkers"]

    sampler = SamplerV2(
        reference=reference, cell_type_key=cell_type_key, num_spots=num_spots,
        cell_number_mean=cell_number_mean, cell_number_nu=cell_number_nu,
        cell_type_number=cell_type_number,
        region_cell_types=region_cell_types,
        balance=balance, poisson_noise_scale=poisson_noise_scale,
        structured_shapes=structured_shapes, structured_base_size=structured_base_size,
        random_seed=random_seed,
    )
    exp, density, sampled_cells_df = sampler.sample_data()
    X, Y = sampler.get_coords()
    coords = np.vstack((X.flatten(), Y.flatten())).T * 10

    spatial_mod = {}
    for key, data in exp.items():
        spatial_ann = ad.AnnData(scipy.sparse.csr_matrix(data))
        if isinstance(reference, ad.AnnData):
            spatial_ann.var.index = reference.var_names
        else:
            spatial_ann.var.index = reference[key].var_names
        spatial_ann.obs["cell_count"] = density.sum(axis=1)
        spatial_ann.obs["region_id"] = sampler.regions
        spatial_ann.uns["density"] = pd.DataFrame(density, columns=sampler.clusters, index=spatial_ann.obs_names)
        spatial_ann.obsm["proportions"] = density / density.sum(axis=1)[:, None]
        spatial_ann.uns["proportion_names"] = sampler.clusters.values
        spatial_ann.obsm["spatial"] = coords.astype("int")
        spatial_mod[key] = spatial_ann

    if isinstance(reference, ad.AnnData):
        return spatial_mod["tmp"], sampled_cells_df
    return mu.MuData(spatial_mod), sampled_cells_df


# ======================================================================
# ************** V2 新增：RNA 退化机制（核心差异）*****************
# ======================================================================

def find_pair_discriminating_genes(
        ref_adata: ad.AnnData,
        cell_type_col: str,
        type_a: str,
        type_b: str,
        top_n: int = 500,
) -> np.ndarray:
    """
    在参考数据中找到最能区分 type_a 和 type_b 的 top_n 个基因的索引。
    使用简单的均值差异排序（比 Wilcoxon 快，对于模拟目的足够）。
    """
    mask_a = ref_adata.obs[cell_type_col] == type_a
    mask_b = ref_adata.obs[cell_type_col] == type_b

    X_a = ref_adata[mask_a].X
    X_b = ref_adata[mask_b].X

    if scipy.sparse.issparse(X_a):
        mean_a = np.asarray(X_a.mean(axis=0)).ravel()
        mean_b = np.asarray(X_b.mean(axis=0)).ravel()
    else:
        mean_a = X_a.mean(axis=0)
        mean_b = X_b.mean(axis=0)

    diff = np.abs(mean_a - mean_b)
    top_indices = np.argsort(-diff)[:top_n]
    return top_indices


def blur_rna_confusable_markers(
        sim_mdata: mu.MuData,
        ref_adata_dict: dict,
        confusable_pairs: List[Tuple[str, str]],
        cell_type_col: str,
        blur_config: dict,
        seed: int = 42,
) -> mu.MuData:
    """
    对 RNA 模态中易混淆细胞类型对的 marker genes 施加针对性噪声。

    对每个 confusable pair (A, B)：
    1. 在参考数据中找出区分 A/B 的 top marker genes
    2. 在 spot 级 RNA 数据中，对这些 genes 添加高斯噪声
       噪声幅度 = blur_strength × 基因表达标准差
    3. 同时将表达值向 A/B 均值中点偏移（减小差异信号）

    ATAC 数据完全不受影响。
    """
    rna_mod = "rna"
    if rna_mod not in sim_mdata.mod:
        rna_mod = list(sim_mdata.mod.keys())[0]

    ref_rna = ref_adata_dict[rna_mod]
    top_n = blur_config.get("top_n_genes", 500)
    blur_strength = blur_config.get("blur_strength", 0.5)

    rna_adata = sim_mdata[rna_mod]
    X = rna_adata.X.toarray() if scipy.sparse.issparse(rna_adata.X) else rna_adata.X.copy()

    np.random.seed(seed)

    for pair_idx, (type_a, type_b) in enumerate(confusable_pairs):
        mask_a = ref_rna.obs[cell_type_col] == type_a
        mask_b = ref_rna.obs[cell_type_col] == type_b
        if mask_a.sum() == 0 or mask_b.sum() == 0:
            print(f"  ⚠ 跳过 pair ({type_a}, {type_b})：参考数据中无对应细胞")
            continue

        disc_genes = find_pair_discriminating_genes(ref_rna, cell_type_col, type_a, type_b, top_n=top_n)
        print(f"  Pair ({type_a}, {type_b}): 对 {len(disc_genes)} 个 discriminating genes 施加模糊")

        X_a_mean = np.asarray(ref_rna[mask_a].X[:, disc_genes].mean(axis=0)).ravel()
        X_b_mean = np.asarray(ref_rna[mask_b].X[:, disc_genes].mean(axis=0)).ravel()
        pair_midpoint = (X_a_mean + X_b_mean) / 2

        for gene_local_idx, gene_global_idx in enumerate(disc_genes):
            gene_col = X[:, gene_global_idx]
            gene_std = np.std(gene_col[gene_col > 0]) if np.any(gene_col > 0) else 1.0

            noise = np.random.normal(0, blur_strength * gene_std, size=X.shape[0])
            shift = (pair_midpoint[gene_local_idx] - gene_col) * blur_strength * 0.3
            X[:, gene_global_idx] = gene_col + shift + noise

    X = np.maximum(X, 0)
    sim_mdata[rna_mod].X = scipy.sparse.csr_matrix(X.astype(np.float32))
    return sim_mdata


def add_ambient_rna(
        sim_mdata: mu.MuData,
        ambient_fraction: float = 0.2,
        seed: int = 42,
) -> mu.MuData:
    """
    向每个 spot 的 RNA 添加环境 RNA 噪声。
    每个 spot 按全局均值 profile 的一定比例叠加"背景表达"，
    稀释细胞类型特异性信号。仅影响 RNA，不影响 ATAC。
    """
    rna_mod = "rna"
    if rna_mod not in sim_mdata.mod:
        rna_mod = list(sim_mdata.mod.keys())[0]

    rna_adata = sim_mdata[rna_mod]
    X = rna_adata.X.toarray() if scipy.sparse.issparse(rna_adata.X) else rna_adata.X.copy()

    mean_profile = X.mean(axis=0)
    mean_total = mean_profile.sum()

    np.random.seed(seed)

    for i in range(X.shape[0]):
        spot_total = X[i].sum()
        if spot_total < 1:
            continue
        ambient_scale = ambient_fraction * spot_total / (mean_total + 1e-8)
        ambient = mean_profile * ambient_scale
        per_gene_noise = np.random.poisson(np.maximum(ambient, 0.01))
        X[i] += per_gene_noise

    sim_mdata[rna_mod].X = scipy.sparse.csr_matrix(np.maximum(X, 0).astype(np.float32))
    return sim_mdata


def add_rna_crosstalk_between_pairs(
        sim_mdata: mu.MuData,
        ref_adata_dict: dict,
        confusable_pairs: List[Tuple[str, str]],
        cell_type_col: str,
        crosstalk_fraction: float = 0.3,
        seed: int = 42,
) -> mu.MuData:
    """
    对每个 spot，如果同时包含 confusable pair 中的两个细胞类型，
    则在 RNA 层面进行"串扰"：将 A 的一部分 RNA 特征混入 B，反之亦然。

    实现方式：计算该 spot 中 pair 成员的比例，按比例加权交叉污染。
    仅影响 RNA。
    """
    rna_mod = "rna"
    if rna_mod not in sim_mdata.mod:
        rna_mod = list(sim_mdata.mod.keys())[0]

    ref_rna = ref_adata_dict[rna_mod]
    rna_adata = sim_mdata[rna_mod]
    X = rna_adata.X.toarray() if scipy.sparse.issparse(rna_adata.X) else rna_adata.X.copy()
    proportions = rna_adata.obsm["proportions"]
    prop_names = list(rna_adata.uns["proportion_names"])

    np.random.seed(seed)

    for type_a, type_b in confusable_pairs:
        if type_a not in prop_names or type_b not in prop_names:
            continue
        idx_a = prop_names.index(type_a)
        idx_b = prop_names.index(type_b)

        mask_a_ref = ref_rna.obs[cell_type_col] == type_a
        mask_b_ref = ref_rna.obs[cell_type_col] == type_b
        mean_a = np.asarray(ref_rna[mask_a_ref].X.mean(axis=0)).ravel()
        mean_b = np.asarray(ref_rna[mask_b_ref].X.mean(axis=0)).ravel()

        for i in range(X.shape[0]):
            prop_a = proportions[i, idx_a]
            prop_b = proportions[i, idx_b]
            if prop_a < 0.01 or prop_b < 0.01:
                continue
            spot_total = X[i].sum()
            if spot_total < 1:
                continue
            cross_amount = crosstalk_fraction * min(prop_a, prop_b) * spot_total
            diff = mean_a - mean_b
            diff_norm = diff / (np.abs(diff).sum() + 1e-8)
            noise_a_to_b = np.random.poisson(np.maximum(cross_amount * np.abs(diff_norm), 0.01))
            noise_sign = np.sign(diff_norm) * noise_a_to_b
            X[i] -= noise_sign * 0.5
            X[i] += np.random.normal(0, 0.1 * cross_amount / (X.shape[1] + 1), size=X.shape[1])

    X = np.maximum(X, 0)
    sim_mdata[rna_mod].X = scipy.sparse.csr_matrix(X.astype(np.float32))
    return sim_mdata


# ======================================================================
# ********************* V2 配置区 ************************************
# ======================================================================

RAW_DATA_PATH = './data_source'
OUTPUT_PATH = './data_v2'
os.makedirs(OUTPUT_PATH, exist_ok=True)

MODALITY_CONFIG = {
    "rna": "human_melanoma_RNA.h5ad",
    "atac": "human_melanoma_ATAC.h5ad",
}
CELL_TYPE_COLUMN = "cell_type"

# ---- V2 核心变化 1：RNA 高噪声，ATAC 零噪声 ----
GLOBAL_POISSON_NOISE = {
    "rna": {
        "enable_noise": True,
        "poisson_scale": 35,
        "zero_protection": True,
        "spatial_gradient_coeff": 4.0,
    },
    "atac": {
        "enable_noise": True,
        "poisson_scale": 0,
        "zero_protection": True,
        "spatial_gradient_coeff": 1.0,
    },
    "noise_seed": 221,
}

# ---- V2 核心变化 2：降低主导比例，增加混合复杂度 ----
SPACE_DISRUPTION_CONFIG = {
    "extreme_proportion": 0.5,
    "shuffle_target_types": True,
    "cell_count_min": 10,
    "cell_count_max": 20,
    "random_seed": 221,
}

# ---- V2 核心变化 3：更激进的 RNA HVG 稀释 ----
FEATURE_SHUFFLE_CONFIG = {
    "enable_shuffle": True,
    "target_modality": ["rna"],
    "spot_proportion": 1.0,
    "hvg_top_n": 300,
    "hvg_min_disp": 0.3,
    "hvg_dilution_factor": 0.08,
    "shuffle_seed": 221,
}

BASE_SIZE = 12
STRUCTURED_SHAPES = ["rings", "scotland", "corners", "squares"]
CELL_NUMBER_MEAN = [12] * 8

# ---- V2 核心变化 4：每个 region 更多细胞类型 ----
CELL_TYPE_NUMBER = [5, 5, 4, 4, 4, 5, 4, 4]

# ---- V2 核心变化 5：显式 region → celltype 映射（确保 confusable pairs 共存）----
# 会在 main() 中根据实际 cell type 名称动态构建
# 10 种细胞类型：T-CD4, T-CD8, T-reg, mDC, mono-mac, myeloid, pDC, plasma, tumour-1, tumour-2
# Confusable pairs: (T-CD4, T-reg), (tumour-1, tumour-2), (mDC, pDC)
CONFUSABLE_PAIRS_TEMPLATE = [
    ("T-CD4", "T-reg"),
    ("tumour-1", "tumour-2"),
    ("mDC", "pDC"),
]

# 每个 pair 必须在至少 3 个 region 中共存
REGION_CELL_TYPES_TEMPLATE = {
    0: ["T-CD4", "T-reg", "mDC", "mono-mac", "tumour-1"],
    1: ["T-CD4", "T-reg", "pDC", "plasma", "tumour-2"],
    2: ["tumour-1", "tumour-2", "mDC", "T-CD8"],
    3: ["tumour-1", "tumour-2", "pDC", "myeloid"],
    4: ["mDC", "pDC", "T-CD4", "mono-mac"],
    5: ["mDC", "pDC", "tumour-1", "T-CD8", "myeloid"],
    6: ["T-CD4", "T-reg", "tumour-1", "tumour-2"],
    7: ["mDC", "pDC", "plasma", "T-CD8"],
}

# ---- V2 核心变化 6：confusable pair RNA 模糊参数 ----
RNA_BLUR_CONFIG = {
    "top_n_genes": 500,
    "blur_strength": 0.6,
    "ambient_fraction": 0.25,
    "crosstalk_fraction": 0.35,
}

SECOND_REGIONS = [
    (48, 48, 48),
    (192, 48, 52),
    (48, 192, 50),
    (192, 192, 48),
    (120, 120, 55),
]

CELL_LEVEL_CONFIG = {
    "spot_radius": 5.0,
    "cell_count_min": 7,
    "cell_count_max": 13,
    "cell_level_seed": 221,
}


def resolve_cell_type_names(ref_adata, cell_type_col, template_names):
    """
    将模板中的 cell type 名称映射到参考数据中的实际名称。
    处理可能的命名差异（如 T-CD4 vs T_CD4）。
    """
    actual_names = ref_adata.obs[cell_type_col].cat.categories.tolist()
    name_map = {}
    for tmpl in template_names:
        if tmpl in actual_names:
            name_map[tmpl] = tmpl
        else:
            tmpl_lower = tmpl.lower().replace("-", "").replace("_", "")
            for actual in actual_names:
                actual_lower = actual.lower().replace("-", "").replace("_", "")
                if tmpl_lower == actual_lower:
                    name_map[tmpl] = actual
                    break
    return name_map


def main():
    print("=" * 70)
    print("V2 数据生成：多组学互补设计")
    print("=" * 70)

    print("\n正在加载多模态单细胞数据...")
    modal_data = {}
    ref_adata_dict = {}
    cell_type_hvgs_dict = {}

    for mod_name, mod_file in MODALITY_CONFIG.items():
        fpath = os.path.join(RAW_DATA_PATH, mod_file)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"数据文件不存在: {fpath}")
        adata = sc.read_h5ad(fpath)
        if 'counts' in adata.layers:
            adata.X = adata.layers['counts'].copy()
        else:
            adata.X = adata.X.copy()
        if mod_name == list(MODALITY_CONFIG.keys())[0]:
            adata.obs[CELL_TYPE_COLUMN] = adata.obs[CELL_TYPE_COLUMN].astype('category')
            ref_cell_types = adata.obs[CELL_TYPE_COLUMN].values
        else:
            adata.obs[CELL_TYPE_COLUMN] = ref_cell_types
        ref_adata_dict[mod_name] = adata
        modal_data[mod_name] = adata
        if mod_name in FEATURE_SHUFFLE_CONFIG["target_modality"]:
            cell_type_hvgs_dict[mod_name] = compute_cell_type_hvgs(
                adata.copy(), CELL_TYPE_COLUMN,
                top_n=FEATURE_SHUFFLE_CONFIG["hvg_top_n"],
                min_disp=FEATURE_SHUFFLE_CONFIG["hvg_min_disp"],
            )

    ref_mod = list(MODALITY_CONFIG.keys())[0]
    print(f"✅ RNA：{modal_data['rna'].n_obs} 细胞 × {modal_data['rna'].n_vars} 特征")
    print(f"✅ ATAC：{modal_data['atac'].n_obs} 细胞 × {modal_data['atac'].n_vars} 特征")

    # 解析 cell type 名称映射
    all_template_names = set()
    for cts in REGION_CELL_TYPES_TEMPLATE.values():
        all_template_names.update(cts)
    for a, b in CONFUSABLE_PAIRS_TEMPLATE:
        all_template_names.add(a)
        all_template_names.add(b)

    name_map = resolve_cell_type_names(ref_adata_dict[ref_mod], CELL_TYPE_COLUMN, all_template_names)
    print(f"\n细胞类型名称映射: {name_map}")

    region_cell_types = {
        rid: [name_map[n] for n in cts if n in name_map]
        for rid, cts in REGION_CELL_TYPES_TEMPLATE.items()
    }
    confusable_pairs = [
        (name_map[a], name_map[b])
        for a, b in CONFUSABLE_PAIRS_TEMPLATE
        if a in name_map and b in name_map
    ]

    print(f"Confusable pairs: {confusable_pairs}")
    for rid, cts in region_cell_types.items():
        print(f"  Region {rid}: {cts}")

    mdata = mu.MuData(modal_data)

    # ==================== 1) 第一轮结构化抽样 ====================
    sim_params = {
        "cell_type_key": CELL_TYPE_COLUMN,
        "num_spots": (BASE_SIZE * 2) ** 2,
        "balance": "unbalanced",
        "cell_number_mean": CELL_NUMBER_MEAN,
        "cell_number_nu": 25.0,
        "cell_type_number": CELL_TYPE_NUMBER,
        "region_cell_types": region_cell_types,
        "poisson_noise_scale": 0.0,
        "structured_shapes": STRUCTURED_SHAPES,
        "structured_base_size": BASE_SIZE,
        "random_seed": 42,
    }

    print("\n[1] 生成第一轮结构化空间数据...")
    simulation_mdata, sampled_cells_df_first = generate_spatial_data_v2(reference=mdata, **sim_params)
    sampled_cells_df_first['sampling_round'] = 1
    print(f"✅ 第一轮抽样完成，生成 {simulation_mdata['rna'].n_obs} 个 spots")

    # ==================== 2) 第二轮极端 cell type 扰动 ====================
    spatial_coords = simulation_mdata[ref_mod].obsm['spatial'].copy()
    spot_names = simulation_mdata[ref_mod].obs_names.tolist()
    target_spots = get_spots_in_regions(spatial_coords, SECOND_REGIONS, spot_names)
    target_indices = [spot_names.index(name) for name in target_spots]
    n_target = len(target_spots)

    print(f"\n[2] 筛选出 {n_target} 个第二轮目标区域 spots")

    if n_target > 0:
        main_seed = SPACE_DISRUPTION_CONFIG["random_seed"]
        np.random.seed(main_seed)
        random.seed(main_seed)

        for mod_name in MODALITY_CONFIG.keys():
            adata = simulation_mdata[mod_name]
            adata_X_lil = adata.X.tolil()
            adata_X_lil[target_indices] = 0
            adata.X = adata_X_lil.tocsr()
            adata.obs.loc[target_spots, 'cell_count'] = 0
            if 'proportions' in adata.obsm:
                adata.obsm['proportions'][target_indices] = 0

        all_cell_types = ref_adata_dict[ref_mod].obs[CELL_TYPE_COLUMN].cat.categories
        n_cell_types = len(all_cell_types)
        cell_type_to_cells = {}
        for ct in all_cell_types:
            cell_type_to_cells[ct] = ref_adata_dict[ref_mod].obs.index[
                ref_adata_dict[ref_mod].obs[CELL_TYPE_COLUMN] == ct
            ].tolist()

        if SPACE_DISRUPTION_CONFIG["shuffle_target_types"]:
            np.random.seed(main_seed + 500)
            random_cell_types = np.random.permutation(all_cell_types)
        else:
            random_cell_types = all_cell_types

        sampled_cells_df_second = []
        for spot_order_idx, spot_idx in enumerate(target_indices):
            spot_name = target_spots[spot_order_idx]
            sub_seed = main_seed + spot_order_idx + 1000
            np.random.seed(sub_seed)
            random.seed(sub_seed)

            cell_count = np.random.randint(
                SPACE_DISRUPTION_CONFIG["cell_count_min"],
                SPACE_DISRUPTION_CONFIG["cell_count_max"] + 1,
            )
            dominant_ct = np.random.choice(random_cell_types)
            dominant_count = int(cell_count * SPACE_DISRUPTION_CONFIG["extreme_proportion"])
            other_cts = [ct for ct in random_cell_types if ct != dominant_ct]
            other_count = cell_count - dominant_count
            other_selected = np.random.choice(other_cts, size=other_count, replace=True)
            selected_types = [dominant_ct] * dominant_count + list(other_selected)
            np.random.shuffle(selected_types)

            sampled_cells = []
            for ct in set(selected_types):
                ct_count = selected_types.count(ct)
                available_cells = cell_type_to_cells[ct]
                sampled_ct_cells = np.random.choice(
                    available_cells, size=ct_count,
                    replace=len(available_cells) < ct_count,
                )
                sampled_cells.extend(sampled_ct_cells)

            for mod_name in MODALITY_CONFIG.keys():
                adata_sim = simulation_mdata[mod_name]
                adata_ref = ref_adata_dict[mod_name]
                exp_matrix = adata_ref[sampled_cells, :].X
                exp_sum = exp_matrix.sum(axis=0).A1 if scipy.sparse.issparse(exp_matrix) else exp_matrix.sum(axis=0)
                adata_X_lil = adata_sim.X.tolil()
                adata_X_lil[spot_idx] = exp_sum.reshape(1, -1)
                adata_sim.X = adata_X_lil.tocsr()

            cell_type_counts = pd.Series(selected_types).value_counts()
            proportions = np.zeros(n_cell_types)
            for ct in cell_type_counts.index:
                if ct in all_cell_types:
                    proportions[all_cell_types.get_loc(ct)] = cell_type_counts[ct] / cell_count
            simulation_mdata[ref_mod].obs.loc[spot_name, 'cell_count'] = cell_count
            if 'proportions' in simulation_mdata[ref_mod].obsm:
                simulation_mdata[ref_mod].obsm['proportions'][spot_idx] = proportions

            sampled_cells_df_second.append({
                "spot_name": spot_name, "sampling_round": 2,
                "main_seed": main_seed, "sub_seed": sub_seed,
                "cell_count": cell_count,
                "selected_types": list(set(selected_types)),
                "cell_type_proportions": proportions.tolist(),
                "sampled_cells": sampled_cells,
            })

        simulation_mdata[ref_mod].obs['second_sampling'] = False
        simulation_mdata[ref_mod].obs.loc[target_spots, 'second_sampling'] = True
        sampled_cells_df_second = pd.DataFrame(sampled_cells_df_second)
        sampled_cells_df = pd.concat([sampled_cells_df_first, sampled_cells_df_second], ignore_index=True)
    else:
        sampled_cells_df = sampled_cells_df_first
        simulation_mdata[ref_mod].obs['second_sampling'] = False

    prop = simulation_mdata[ref_mod].obsm["proportions"]
    prop_names = simulation_mdata[ref_mod].uns.get("proportion_names", None)
    if prop_names is None:
        prop_names = ref_adata_dict[ref_mod].obs[CELL_TYPE_COLUMN].cat.categories.values
    prop_names = np.array(prop_names)
    dominant_idx = np.argmax(prop, axis=1)
    simulation_mdata[ref_mod].obs["gt"] = prop_names[dominant_idx]

    # ==================== 3) 空间梯度 Poisson 噪声（RNA 重、ATAC 无）====================
    print("\n[3] 添加空间梯度式泊松噪声（RNA 重噪声, ATAC 无噪声）...")
    simulation_mdata = add_spatial_gradient_poisson_noise(simulation_mdata, GLOBAL_POISSON_NOISE)

    # ==================== 4) RNA HVG 稀释（更激进）====================
    print("\n[4] RNA 高变基因稀释（dilution_factor={})...".format(
        FEATURE_SHUFFLE_CONFIG["hvg_dilution_factor"]))
    for mod_name in FEATURE_SHUFFLE_CONFIG["target_modality"]:
        if mod_name in cell_type_hvgs_dict:
            simulation_mdata = shuffle_cell_type_hvgs(
                simulation_mdata, FEATURE_SHUFFLE_CONFIG,
                cell_type_hvgs_dict[mod_name],
                dominant_cell_type_col="gt",
            )

    # ==================== 5) V2 独有：confusable pair marker 模糊 ====================
    print("\n[5] 🔑 V2: 对 confusable pairs 的 discriminating genes 施加 RNA 模糊...")
    simulation_mdata = blur_rna_confusable_markers(
        simulation_mdata, ref_adata_dict, confusable_pairs,
        CELL_TYPE_COLUMN, RNA_BLUR_CONFIG, seed=42,
    )

    # ==================== 6) V2 独有：环境 RNA 噪声 ====================
    print("\n[6] 🔑 V2: 添加环境 RNA 背景噪声 (ambient_fraction={})...".format(
        RNA_BLUR_CONFIG["ambient_fraction"]))
    simulation_mdata = add_ambient_rna(
        simulation_mdata,
        ambient_fraction=RNA_BLUR_CONFIG["ambient_fraction"],
        seed=221,
    )

    # ==================== 7) V2 独有：confusable pair RNA 串扰 ====================
    print("\n[7] 🔑 V2: confusable pair RNA 串扰 (crosstalk_fraction={})...".format(
        RNA_BLUR_CONFIG["crosstalk_fraction"]))
    simulation_mdata = add_rna_crosstalk_between_pairs(
        simulation_mdata, ref_adata_dict, confusable_pairs,
        CELL_TYPE_COLUMN,
        crosstalk_fraction=RNA_BLUR_CONFIG["crosstalk_fraction"],
        seed=333,
    )

    # ==================== 8) 构建 spot → cell 映射 ====================
    spot_to_cells = {}
    for i, spot_name in enumerate(spot_names):
        if n_target > 0 and spot_name in target_spots:
            row = sampled_cells_df[
                (sampled_cells_df["sampling_round"] == 2) &
                (sampled_cells_df["spot_name"] == spot_name)
            ].iloc[0]
            spot_to_cells[spot_name] = row["sampled_cells"]
        else:
            spot_to_cells[spot_name] = sampled_cells_df_first.iloc[i]["cell_id"]

    # ==================== 9) 生成细胞级空间数据 ====================
    print("\n[8] 生成细胞级空间数据...")
    cell_level_mdata_dict = build_cell_level_data(
        ref_adata_dict=ref_adata_dict,
        spot_names=spot_names,
        spatial_coords=spatial_coords,
        spot_to_cells=spot_to_cells,
        cell_type_col=CELL_TYPE_COLUMN,
        spot_radius=CELL_LEVEL_CONFIG["spot_radius"],
        cell_count_min=CELL_LEVEL_CONFIG["cell_count_min"],
        cell_count_max=CELL_LEVEL_CONFIG["cell_count_max"],
        seed=CELL_LEVEL_CONFIG["cell_level_seed"],
    )
    print(f"✅ 细胞级数据：共 {cell_level_mdata_dict[ref_mod].n_obs} 个细胞")

    # ==================== 10) 保存结果 ====================
    print("\n[9] 保存结果...")

    ref_rna_path = os.path.join(OUTPUT_PATH, 'ref_RNA.h5ad')
    cell_level_mdata_dict["rna"].copy().write_h5ad(ref_rna_path)
    print(f"✅ 单细胞参考 RNA: {ref_rna_path}")

    sim_rna_path = os.path.join(OUTPUT_PATH, 'simulation_rna.h5ad')
    simulation_mdata["rna"].write_h5ad(sim_rna_path)
    print(f"✅ 空转 RNA (含 V2 退化): {sim_rna_path}")

    sim_atac_path = os.path.join(OUTPUT_PATH, 'simulation_atac.h5ad')
    simulation_mdata["atac"].write_h5ad(sim_atac_path)
    print(f"✅ 空转 ATAC (干净无退化): {sim_atac_path}")

    # 导出 GT proportions CSV
    gt_prop = simulation_mdata[ref_mod].obsm["proportions"]
    gt_names = simulation_mdata[ref_mod].uns["proportion_names"]
    gt_df = pd.DataFrame(gt_prop, columns=gt_names, index=simulation_mdata[ref_mod].obs_names)
    coords_df = pd.DataFrame(
        simulation_mdata[ref_mod].obsm["spatial"],
        columns=["x", "y"],
        index=simulation_mdata[ref_mod].obs_names,
    )
    gt_export = pd.concat([coords_df, gt_df], axis=1)
    gt_export.index.name = "spot_id"
    gt_path = os.path.join(OUTPUT_PATH, 'gt_proportions.csv')
    gt_export.to_csv(gt_path)
    print(f"✅ 真值比例: {gt_path}")

    if cell_type_hvgs_dict:
        hvgs_df = pd.DataFrame([
            {'cell_type': ct, 'modality': mod, 'hvgs': ','.join(hvgs)}
            for mod, ct_hvgs in cell_type_hvgs_dict.items()
            for ct, hvgs in ct_hvgs.items()
        ])
        hvgs_path = os.path.join(OUTPUT_PATH, 'cell_type_hvgs_info.csv')
        hvgs_df.to_csv(hvgs_path, index=False)

    sampled_cells_path = os.path.join(OUTPUT_PATH, 'sampled_cells_info.csv')
    sampled_cells_df.to_csv(sampled_cells_path, index=False)

    # 导出 cell_coords_gt.csv（单细胞真值坐标 + 细胞类型）
    ref_rna_cell = cell_level_mdata_dict["rna"]
    cell_coords_gt = pd.DataFrame({
        "cell_id": ref_rna_cell.obs_names,
        "x": ref_rna_cell.obsm["spatial"][:, 0],
        "y": ref_rna_cell.obsm["spatial"][:, 1],
        "cellType": ref_rna_cell.obs["cell_type"].values,
    })
    cell_coords_gt_path = os.path.join(OUTPUT_PATH, 'cell_coords_gt.csv')
    cell_coords_gt.to_csv(cell_coords_gt_path, index=False)
    print(f"✅ 单细胞真值坐标: {cell_coords_gt_path}  ({len(cell_coords_gt)} 个细胞)")

    # 打印数据设计摘要
    print("\n" + "=" * 70)
    print("V2 数据设计摘要")
    print("=" * 70)
    print(f"  Confusable pairs: {confusable_pairs}")
    print(f"  每个 region 的细胞类型数: {[len(v) for v in region_cell_types.values()]}")
    print(f"  RNA Poisson noise scale: {GLOBAL_POISSON_NOISE['rna']['poisson_scale']}")
    print(f"  ATAC Poisson noise scale: {GLOBAL_POISSON_NOISE['atac']['poisson_scale']}")
    print(f"  RNA HVG dilution factor: {FEATURE_SHUFFLE_CONFIG['hvg_dilution_factor']}")
    print(f"  Confusable marker blur strength: {RNA_BLUR_CONFIG['blur_strength']}")
    print(f"  Ambient RNA fraction: {RNA_BLUR_CONFIG['ambient_fraction']}")
    print(f"  RNA crosstalk fraction: {RNA_BLUR_CONFIG['crosstalk_fraction']}")
    print(f"  极端比例 (second round): {SPACE_DISRUPTION_CONFIG['extreme_proportion']}")
    print("=" * 70)
    print("\n预期效果:")
    print("  🥇 多组学方法 (RNA+ATAC): ATAC 可弥补 RNA 中的混淆信号 → 最佳指标")
    print("  🥈 单组学方法 (RNA GNN):  空间图信息部分缓解混淆 → 中等指标")
    print("  🥉 Seurat (RNA label transfer): RNA 混淆 + 无空间图 → 最差指标")
    print("\n全部完成。")


if __name__ == "__main__":
    main()
