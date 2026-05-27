# -*- coding: utf-8 -*-
"""
空间转录组模拟：Spot 级 + 细胞级（每 spot 7-13 细胞、圆内随机坐标）
不依赖其他 py 文件，单文件运行。
"""

import math
import os
import random
from typing import Optional, List, Union

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import scipy
import scanpy as sc


# ====================== 基础空间图案生成函数 ======================
def conway_maxwell_poisson(lambda_: int, nu: float, seed: Optional[int] = None) -> int:
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        random.seed(seed)

    lambda_ = int(lambda_)
    nu = float(nu)

    # 归一化常数 C
    C = np.sum([(pow(lambda_, k) / math.factorial(k)) ** nu for k in range(1000)])

    # 采样
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


def gen_spatial_factors(
        shapes: List[str],
        base_size: int = 12
) -> tuple:
    """
    生成 4 个基本图案，并组合成 8 个 region 的 one-hot 空间因子矩阵 F
    """
    shape_funcs = {
        "squares": squares,
        "corners": corners,
        "scotland": scotland,
        "checkers": checkers,
        "rings": rings
    }

    assert len(shapes) == 4, "shapes 长度必须为 4。"
    for shape in shapes:
        assert shape in shape_funcs.keys(), f"不支持的形状：{shape}"

    shape1 = shape_funcs[shapes[0]](base_size=base_size)
    shape2 = shape_funcs[shapes[1]](base_size=base_size)
    shape3 = shape_funcs[shapes[2]](base_size=base_size)
    shape4 = shape_funcs[shapes[3]](base_size=base_size)

    total_size = base_size * 2
    region_matrix = np.zeros((total_size, total_size), dtype=int)

    # region 映射：0/1 区分背景/图案，4 个 quadrant 共 8 个 region id
    region_mapping = {
        (0, 0): 4, (0, 1): 5,
        (1, 0): 6, (1, 1): 7,
        (2, 0): 0, (2, 1): 1,
        (3, 0): 2, (3, 1): 3
    }

    quadrants = [
        (shape1, 0, 0),
        (shape2, 0, base_size),
        (shape3, base_size, 0),
        (shape4, base_size, base_size)
    ]

    for q_idx, (q_matrix, x_offset, y_offset) in enumerate(quadrants):
        for i in range(base_size):
            for j in range(base_size):
                global_i = x_offset + i
                global_j = y_offset + j
                pixel_val = int(q_matrix[i, j])
                region_matrix[global_i, global_j] = region_mapping[(q_idx, pixel_val)]

    top_row = np.hstack((shape1, shape2))
    bottom_row = np.hstack((shape3, shape4))
    full_pattern = np.vstack((top_row, bottom_row))

    F = np.zeros((total_size * total_size, 8))
    for r_id in range(8):
        mask = region_matrix.flatten() == r_id
        F[mask, r_id] = 1

    return F, full_pattern, region_matrix


# ====================== 采样器类 ======================
class Sampler:
    def __init__(
            self,
            reference: Union[mu.MuData, ad.AnnData],
            cell_type_key: str,
            num_spots: int,
            cell_number_mean: Union[int, list] = [6, 8, 6, 8, 7, 9, 7, 9],
            cell_number_nu: Union[float, list] = 20.0,
            cell_type_number: Union[int, list] = [4, 4, 4, 4, 4, 4, 4, 4],
            balance: Optional[str] = "balanced",
            poisson_noise_scale: float = 1.0,
            structured_shapes: List[str] = None,
            structured_base_size: int = 12,
            random_seed: int = 221
    ):
        if structured_shapes is None:
            structured_shapes = ["squares", "corners", "scotland", "checkers"]

        self.reference = reference
        self.cell_type_key = cell_type_key
        self.num_spots = num_spots
        self.obs = reference.obs if isinstance(reference, ad.AnnData) else reference[list(reference.mod.keys())[0]].obs

        self.poisson_noise_scale = poisson_noise_scale
        self.structured_shapes = structured_shapes
        self.structured_base_size = structured_base_size
        self.n_regions = 8

        self.random_seed = random_seed

        expected_num_spots = (structured_base_size * 2) ** 2
        assert self.num_spots == expected_num_spots, "num_spots 必须等于 (structured_base_size * 2) ** 2"

        if isinstance(cell_number_mean, int):
            self.cell_number_mean = np.ones(8, dtype=int) * cell_number_mean
        else:
            self.cell_number_mean = np.array(cell_number_mean)

        if isinstance(cell_number_nu, float):
            self.cell_number_nu = np.ones(8) * cell_number_nu
        else:
            self.cell_number_nu = np.array(cell_number_nu)

        if isinstance(cell_type_number, int):
            self.cell_type_number = np.ones(8, dtype=int) * cell_type_number
        else:
            self.cell_type_number = np.array(cell_type_number)

        if balance not in ["balanced", "unbalanced"]:
            raise ValueError('balance must be one of ["balanced", "unbalanced"].')

        self.init_sample_prob(balance=balance)

    def init_sample_prob(self, balance="unbalanced"):
        cell_counts = self.obs[self.cell_type_key].value_counts(normalize=True)

        if balance == "unbalanced":
            # 按原始比例抽样
            self.cluster_p = cell_counts
            self.cell_p = self.obs[self.cell_type_key].map(self.cluster_p).astype(float)
        elif balance == "balanced":
            # 每个 cell type 权重均等
            self.cluster_p = pd.Series(1 / len(cell_counts), index=cell_counts.index)
            self.cell_p = 1 / self.obs[self.cell_type_key].map(cell_counts).astype(float) / len(cell_counts)

        self.clusters = self.obs[self.cell_type_key].cat.categories
        self.cluster_p = self.cluster_p[self.clusters]

    def define_regions(self):
        F, full_pattern, region_matrix = gen_spatial_factors(
            shapes=self.structured_shapes,
            base_size=self.structured_base_size
        )
        self.regions = region_matrix.flatten()
        self.region_matrix = region_matrix

    def sample_data(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        # 将所有 cluster 随机打乱后，按 region 分配要用到的 cluster
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

        # 每个 spot 的细胞数
        cell_count = []
        for idx, region_id in enumerate(self.regions):
            spot_seed = self.random_seed + idx
            cell_num = conway_maxwell_poisson(
                self.cell_number_mean[region_id],
                self.cell_number_nu[region_id],
                seed=spot_seed
            )
            cell_count.append(max(cell_num, 1))
        cell_count = np.array(cell_count)

        # 每个 spot 可用的 cluster
        used_clusters_list = [used_clusters[region_id] for region_id in self.regions]

        params = list(zip(cell_count, used_clusters_list, self.regions))
        return self.sample_spots(params)

    def sample_spots(self, params):
        # 兼容 AnnData / MuData
        sample_exp = {"tmp": self.reference} if isinstance(self.reference, ad.AnnData) else self.reference.mod
        exp = {key: np.zeros((len(params), adata.shape[1])) for key, adata in sample_exp.items()}

        # density[spot, cluster]
        density = np.zeros((len(params), len(self.clusters)))

        sampled_cells_df = []

        for i, (num_cell, used_clusters, region_id) in enumerate(params):
            spot_seed = self.random_seed + i
            np.random.seed(spot_seed)

            # 仅从该 region 指定的 cell types 中采样
            cluster_mask = self.obs[self.cell_type_key].isin(used_clusters).values
            p = self.cell_p[cluster_mask] / self.cell_p[cluster_mask].sum()

            sampled_cells = np.random.choice(
                self.obs.index[cluster_mask],
                size=num_cell,
                p=p,
            )

            # 聚合表达
            for key, adata in sample_exp.items():
                exp[key][i, :] = adata[sampled_cells, :].X.sum(axis=0)

            # 记录 cell type 组成
            density[i, :] = self.obs.loc[sampled_cells, self.cell_type_key].value_counts().reindex(
                self.clusters, fill_value=0
            ).values

            sampled_cells_df.append({
                "region_id": [region_id] * num_cell,
                "cell_id": sampled_cells.tolist(),
                "cell_type": self.obs.loc[sampled_cells, self.cell_type_key].values.tolist(),
            })

        # 加泊松噪声（全局）
        for key in exp.keys():
            exp[key] = add_poisson_noise(exp[key], self.poisson_noise_scale, seed=self.random_seed)

        return exp, density, pd.DataFrame(sampled_cells_df)

    def get_coords(self):
        grid_size = int(np.sqrt(self.num_spots))
        x = np.arange(0, grid_size)
        y = np.arange(0, grid_size)
        X, Y = np.meshgrid(x, y)
        return X, Y


def add_poisson_noise(data, noise_scale=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if scipy.sparse.issparse(data):
        data_arr = data.toarray()
    else:
        data_arr = np.array(data)

    lambda_ = np.maximum(data_arr * noise_scale, 1e-6)
    noise = np.random.poisson(lambda_)
    noisy_data = data_arr + noise
    noisy_data = np.maximum(noisy_data, 0).astype(int)

    return scipy.sparse.csr_matrix(noisy_data)


def add_spatial_gradient_poisson_noise(mdata: mu.MuData, noise_config: dict) -> mu.MuData:
    """
    按与中心距离增加泊松噪声强度，形成空间梯度噪声。
    """
    main_seed = noise_config["noise_seed"]

    # 参考一个模态的空间坐标
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
        # 每个 spot 的噪声 scale
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


def compute_cell_type_hvgs(adata_ref: ad.AnnData, cell_type_col: str, top_n=200, min_disp=0.5) -> dict:
    """
    为每个 cell type 单独做 HVG 选择，返回 {cell_type: [hvgs]}。
    """
    cell_type_hvgs = {}
    cell_types = adata_ref.obs[cell_type_col].cat.categories

    for ct in cell_types:
        adata_ct = adata_ref[adata_ref.obs[cell_type_col] == ct].copy()
        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)
        sc.pp.highly_variable_genes(
            adata_ct,
            min_mean=0.0125,
            max_mean=3,
            min_disp=min_disp,
            n_top_genes=top_n
        )
        hvgs = adata_ct.var_names[adata_ct.var.highly_variable].tolist()
        cell_type_hvgs[ct] = hvgs

    return cell_type_hvgs


def shuffle_cell_type_hvgs(mdata: mu.MuData, shuffle_config: dict, cell_type_hvgs: dict,
                           dominant_cell_type_col="gt") -> mu.MuData:
    """
    在给定模态上，对每个 spot 主导 cell type 的 HVGs 做“稀释”（下调表达），
    以模拟特征扰动。
    """
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

        # 要进行 shuffle 的 spot 数量
        n_shuffle_spots = int(n_spots * shuffle_config["spot_proportion"])
        shuffle_spot_idx = np.random.choice(n_spots, n_shuffle_spots, replace=False)

        # dominant cell type 来源于 ref_mod（通常是 RNA）
        ref_mod = list(mdata.mod.keys())[0]
        dominant_cell_types = mdata[ref_mod].obs[dominant_cell_type_col].values

        for spot_idx in shuffle_spot_idx:
            dom_ct = dominant_cell_types[spot_idx]
            if dom_ct not in cell_type_hvgs or len(cell_type_hvgs[dom_ct]) == 0:
                continue

            hvgs = cell_type_hvgs[dom_ct]
            # 只取 80% 的 HVGs 做稀释
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


def get_spots_in_regions(coords: np.ndarray, regions: list, spot_names: list) -> list:
    """
    给定一组矩形区域 (cx, cy, size)，选出落在任意一个区域内的 spot 名称。
    coords: [n_spots, 2]
    regions: [(cx, cy, size), ...]
    """
    target_spots = []

    for (cx, cy, size) in regions:
        x_min, x_max = cx - size / 2, cx + size / 2
        y_min, y_max = cy - size / 2, cy + size / 2

        in_region = (
                (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
                (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
        )

        target_spots.extend([spot_names[i] for i in np.where(in_region)[0]])

    # 去重并按原顺序排序
    target_spots = list(set(target_spots))
    target_spots.sort(key=lambda x: spot_names.index(x))

    return target_spots


def generate_spatial_data(
        reference: Union[mu.MuData, ad.AnnData],
        cell_type_key: str,
        num_spots: int = 576,
        balance: Optional[str] = None,
        cell_number_mean: Union[int, list] = [6, 8, 6, 8, 7, 9, 7, 9],
        cell_number_nu: Union[float, list] = 20.0,
        cell_type_number: Union[int, list] = [4, 4, 4, 4, 4, 4, 4, 4],
        poisson_noise_scale: float = 1.0,
        structured_shapes: List[str] = None,
        structured_base_size: int = 12,
        random_seed: int = 0
) -> tuple:
    """
    以 reference（单细胞多模态）为基础，生成 spot 级多模态空间数据。
    返回：
    - 若 reference 为 MuData，则返回 (MuData, sampled_cells_df)
    - 若 reference 为 AnnData，则返回 (AnnData, sampled_cells_df)
    """
    if structured_shapes is None:
        structured_shapes = ["squares", "corners", "scotland", "checkers"]

    sampler = Sampler(
        reference=reference,
        cell_type_key=cell_type_key,
        num_spots=num_spots,
        cell_number_mean=cell_number_mean,
        cell_number_nu=cell_number_nu,
        cell_type_number=cell_type_number,
        balance=balance,
        poisson_noise_scale=poisson_noise_scale,
        structured_shapes=structured_shapes,
        structured_base_size=structured_base_size,
        random_seed=random_seed
    )

    exp, density, sampled_cells_df = sampler.sample_data()
    X, Y = sampler.get_coords()
    coords = np.vstack((X.flatten(), Y.flatten())).T * 10  # 放大坐标，便于可视化

    spatial_mod = {}

    for key, data in exp.items():
        spatial_ann = ad.AnnData(scipy.sparse.csr_matrix(data))

        if isinstance(reference, ad.AnnData):
            spatial_ann.var.index = reference.var_names
        else:
            spatial_ann.var.index = reference[key].var_names

        # 每个 spot 的细胞数量
        spatial_ann.obs["cell_count"] = density.sum(axis=1)
        # region id
        spatial_ann.obs["region_id"] = sampler.regions

        # density 和 proportion（真值）
        spatial_ann.uns["density"] = pd.DataFrame(
            density, columns=sampler.clusters, index=spatial_ann.obs_names
        )
        spatial_ann.obsm["proportions"] = density / density.sum(axis=1)[:, None]
        spatial_ann.uns["proportion_names"] = sampler.clusters.values

        # 空间坐标
        spatial_ann.obsm["spatial"] = coords.astype("int")

        spatial_mod[key] = spatial_ann

    if isinstance(reference, ad.AnnData):
        return spatial_mod["tmp"], sampled_cells_df

    return mu.MuData(spatial_mod), sampled_cells_df


# ====================== 细胞级：spot 内随机坐标 ======================
def sample_cell_coords_in_circle(cx: float, cy: float, radius: float, size: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """在圆心 (cx,cy)、半径 radius 的圆内均匀采样 size 个 (x,y)。"""
    coords = np.zeros((size, 2))
    n = 0
    while n < size:
        x = rng.uniform(cx - radius, cx + radius)
        y = rng.uniform(cy - radius, cy + radius)
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
            coords[n, 0], coords[n, 1] = x, y
            n += 1
    return coords


def build_cell_level_data(
        ref_adata_dict: dict,
        spot_names: list,
        spatial_coords: np.ndarray,
        spot_to_cells: dict,
        cell_type_col: str,
        spot_radius: float = 5.0,
        cell_count_min: int = 7,
        cell_count_max: int = 13,
        seed: int = 221,
) -> dict:
    """
    使用 spot → 细胞映射表，为每个 spot 随机生成 7–13 个细胞，
    并在以该 spot 为圆心的半径 spot_radius 的圆内均匀采样细胞坐标。
    返回一个 {mod_name: AnnData} 的字典，作为细胞级多模态数据。
    """
    rng = np.random.default_rng(seed)

    all_cell_ids = []
    all_spot_ids = []
    all_cell_types = []
    all_coords = []

    for i, spot_name in enumerate(spot_names):
        cx, cy = float(spatial_coords[i, 0]), float(spatial_coords[i, 1])

        cells = spot_to_cells.get(spot_name, [])
        if len(cells) == 0:
            continue

        n_want = int(rng.integers(cell_count_min, cell_count_max + 1))
        if n_want <= len(cells):
            chosen = rng.choice(cells, size=n_want, replace=False).tolist()
        else:
            chosen = rng.choice(cells, size=n_want, replace=True).tolist()

        coords = sample_cell_coords_in_circle(cx, cy, spot_radius, len(chosen), rng)

        all_cell_ids.extend(chosen)
        all_spot_ids.extend([spot_name] * len(chosen))
        all_coords.append(coords)

        # 使用第一个模态的 obs 获取 cell type
        ref_obs = ref_adata_dict[list(ref_adata_dict.keys())[0]].obs
        for c in chosen:
            all_cell_types.append(ref_obs.loc[c, cell_type_col])

    all_coords = np.vstack(all_coords)

    mod_to_adata = {}
    for mod_name, ref_adata in ref_adata_dict.items():
        X_cell = ref_adata[all_cell_ids, :].X
        cell_index = [f"{sid}_{k}" for k, sid in enumerate(all_spot_ids)]

        obs = pd.DataFrame({
            "spot_id": all_spot_ids,
            "cell_type": all_cell_types,
        }, index=cell_index)

        adata_cell = ad.AnnData(X_cell, obs=obs, var=ref_adata.var.copy())
        adata_cell.obsm["spatial"] = all_coords

        mod_to_adata[mod_name] = adata_cell

    return mod_to_adata


# ====================================================================================
# ********************************** 配置区 **************************************
# ====================================================================================
RAW_DATA_PATH = '../data_source'
OUTPUT_PATH = './data_v1'
os.makedirs(OUTPUT_PATH, exist_ok=True)

MODALITY_CONFIG = {
    "rna": "human_melanoma_RNA.h5ad",
    "atac": "human_melanoma_ATAC.h5ad"
}

CELL_TYPE_COLUMN = "cell_type"

GLOBAL_POISSON_NOISE = {
    "rna": {"enable_noise": True, "poisson_scale": 15, "zero_protection": True, "spatial_gradient_coeff": 3.0},
    "atac": {"enable_noise": True, "poisson_scale": 0, "zero_protection": True, "spatial_gradient_coeff": 3.0},
    "noise_seed": 221
}

SPACE_DISRUPTION_CONFIG = {
    "extreme_proportion": 0.8,
    "shuffle_target_types": True,  # 是否打乱目标 spot 的 cell type 分布
    "cell_count_min": 10,
    "cell_count_max": 20,
    "random_seed": 221
}

FEATURE_SHUFFLE_CONFIG = {
    "enable_shuffle": True,
    "target_modality": ["rna"],
    "spot_proportion": 1.0,
    "hvg_top_n": 200,
    "hvg_min_disp": 0.5,
    "hvg_dilution_factor": 0.2,
    "shuffle_seed": 221
}

BASE_SIZE = 12
STRUCTURED_SHAPES = ["rings", "corners", "scotland", "squares"]

CELL_NUMBER_MEAN = [10] * 8
CELL_TYPE_NUMBER = [4, 4, 2, 2, 2, 4, 2, 2]

# 第二轮采样的区域（在 spot 坐标空间）
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


def main():
    print("正在加载多模态单细胞数据...")
    modal_data = {}
    ref_adata_dict = {}
    cell_type_hvgs_dict = {}

    # 1) 读取单细胞多模态数据，构建 MuData
    for mod_name, mod_file in MODALITY_CONFIG.items():
        fpath = os.path.join(RAW_DATA_PATH, mod_file)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"数据文件不存在: {fpath}")

        adata = sc.read_h5ad(fpath)

        # 使用 counts 层作为原始计数
        if 'counts' in adata.layers:
            adata.X = adata.layers['counts'].copy()
        else:
            adata.X = adata.X.copy()

        # 第一个模态负责 cell type 定义
        if mod_name == list(MODALITY_CONFIG.keys())[0]:
            adata.obs[CELL_TYPE_COLUMN] = adata.obs[CELL_TYPE_COLUMN].astype('category')
            ref_cell_types = adata.obs[CELL_TYPE_COLUMN].values
        else:
            # 其它模态复用同一 cell type 标签（假定 cell 顺序对齐）
            adata.obs[CELL_TYPE_COLUMN] = ref_cell_types

        ref_adata_dict[mod_name] = adata
        modal_data[mod_name] = adata

        # 为目标模态计算每个 cell type 的 HVGs
        if mod_name in FEATURE_SHUFFLE_CONFIG["target_modality"]:
            cell_type_hvgs_dict[mod_name] = compute_cell_type_hvgs(
                adata.copy(),
                CELL_TYPE_COLUMN,
                top_n=FEATURE_SHUFFLE_CONFIG["hvg_top_n"],
                min_disp=FEATURE_SHUFFLE_CONFIG["hvg_min_disp"]
            )

    print(f"✅ RNA：{modal_data['rna'].n_obs} 细胞 × {modal_data['rna'].n_vars} 特征")
    print(f"✅ ATAC：{modal_data['atac'].n_obs} 细胞 × {modal_data['atac'].n_vars} 特征")

    mdata = mu.MuData(modal_data)

    # 2) 第一轮结构化抽样：生成全图的 spot 级多模态空间数据
    sim_params = {
        "cell_type_key": CELL_TYPE_COLUMN,
        "num_spots": (BASE_SIZE * 2) ** 2,
        "balance": "unbalanced",
        "cell_number_mean": CELL_NUMBER_MEAN,
        "cell_number_nu": 25.0,
        "cell_type_number": CELL_TYPE_NUMBER,
        "poisson_noise_scale": 0.0,  # 第一轮不加全局噪声；后面再统一加
        "structured_shapes": STRUCTURED_SHAPES,
        "structured_base_size": BASE_SIZE,
        "random_seed": 42
    }

    print("\n开始生成第一轮结构化空间数据...")
    simulation_mdata, sampled_cells_df_first = generate_spatial_data(reference=mdata, **sim_params)
    sampled_cells_df_first['sampling_round'] = 1
    print(f"✅ 第一轮抽样完成，生成 {simulation_mdata['rna'].n_obs} 个 spots")

    # 3) 第二轮：在指定区域做“极端” cell type 扰动
    ref_mod = list(MODALITY_CONFIG.keys())[0]  # 以第一个模态（RNA）为主
    spatial_coords = simulation_mdata[ref_mod].obsm['spatial'].copy()
    spot_names = simulation_mdata[ref_mod].obs_names.tolist()

    target_spots = get_spots_in_regions(spatial_coords, SECOND_REGIONS, spot_names)
    target_indices = [spot_names.index(name) for name in target_spots]
    n_target = len(target_spots)

    print(f"\n🎯 筛选出 {n_target} 个第二轮目标区域 spots")

    if n_target > 0:
        main_seed = SPACE_DISRUPTION_CONFIG["random_seed"]
        np.random.seed(main_seed)
        random.seed(main_seed)

        # 先将目标 spot 清零
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

        # 为每个 cell type 预先缓存细胞 id 列表
        cell_type_to_cells = {}
        for ct in all_cell_types:
            cell_type_to_cells[ct] = ref_adata_dict[ref_mod].obs.index[
                ref_adata_dict[ref_mod].obs[CELL_TYPE_COLUMN] == ct
            ].tolist()

        # 是否打乱 cell type 顺序
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
                SPACE_DISRUPTION_CONFIG["cell_count_max"] + 1
            )

            dominant_ct = np.random.choice(random_cell_types)
            dominant_count = int(cell_count * SPACE_DISRUPTION_CONFIG["extreme_proportion"])

            other_cts = [ct for ct in random_cell_types if ct != dominant_ct]
            other_count = cell_count - dominant_count

            other_selected = np.random.choice(other_cts, size=other_count, replace=True)
            selected_types = [dominant_ct] * dominant_count + list(other_selected)
            np.random.shuffle(selected_types)

            # 为每个 cell type 抽样具体细胞
            sampled_cells = []
            for ct in set(selected_types):
                ct_count = selected_types.count(ct)
                available_cells = cell_type_to_cells[ct]
                sampled_ct_cells = np.random.choice(
                    available_cells,
                    size=ct_count,
                    replace=len(available_cells) < ct_count
                )
                sampled_cells.extend(sampled_ct_cells)

            # 更新各模态在该 spot 的表达（聚合）
            for mod_name in MODALITY_CONFIG.keys():
                adata_sim = simulation_mdata[mod_name]
                adata_ref = ref_adata_dict[mod_name]
                exp_matrix = adata_ref[sampled_cells, :].X
                if scipy.sparse.issparse(exp_matrix):
                    exp_sum = exp_matrix.sum(axis=0).A1
                else:
                    exp_sum = exp_matrix.sum(axis=0)
                exp_sum_2d = exp_sum.reshape(1, -1)

                adata_X_lil = adata_sim.X.tolil()
                adata_X_lil[spot_idx] = exp_sum_2d
                adata_sim.X = adata_X_lil.tocsr()

            # 该 spot 的 cell type 组成（真值）
            cell_type_counts = pd.Series(selected_types).value_counts()
            proportions = np.zeros(n_cell_types)
            for ct in cell_type_counts.index:
                if ct in all_cell_types:
                    proportions[all_cell_types.get_loc(ct)] = cell_type_counts[ct] / cell_count

            simulation_mdata[ref_mod].obs.loc[spot_name, 'cell_count'] = cell_count
            if 'proportions' in simulation_mdata[ref_mod].obsm:
                simulation_mdata[ref_mod].obsm['proportions'][spot_idx] = proportions

            sampled_cells_df_second.append({
                "spot_name": spot_name,
                "sampling_round": 2,
                "main_seed": main_seed,
                "sub_seed": sub_seed,
                "cell_count": cell_count,
                "selected_types": list(set(selected_types)),
                "cell_type_proportions": proportions.tolist(),
                "sampled_cells": sampled_cells
            })

        simulation_mdata[ref_mod].obs['second_sampling'] = False
        simulation_mdata[ref_mod].obs.loc[target_spots, 'second_sampling'] = True

        sampled_cells_df_second = pd.DataFrame(sampled_cells_df_second)
        sampled_cells_df = pd.concat([sampled_cells_df_first, sampled_cells_df_second], ignore_index=True)
    else:
        sampled_cells_df = sampled_cells_df_first
        simulation_mdata[ref_mod].obs['second_sampling'] = False

    # ✅ 关键修复：无论是否第二轮，都用 proportions 全量生成 gt（从 uns['proportion_names'] 取名字）
    prop = simulation_mdata[ref_mod].obsm["proportions"]
    prop_names = simulation_mdata[ref_mod].uns.get("proportion_names", None)
    if prop_names is None:
        prop_names = ref_adata_dict[ref_mod].obs[CELL_TYPE_COLUMN].cat.categories.values
    prop_names = np.array(prop_names)

    dominant_idx = np.argmax(prop, axis=1)
    simulation_mdata[ref_mod].obs["gt"] = prop_names[dominant_idx]

    # 4) 在 spot 级数据上添加空间梯度噪声
    print("\n📢 开始添加空间梯度式泊松噪声...")
    simulation_mdata = add_spatial_gradient_poisson_noise(simulation_mdata, GLOBAL_POISSON_NOISE)

    # 5) 针对主导 cell type 的 HVGs 做特征稀释
    print("\n🔀 开始高变基因稀释...")
    for mod_name in FEATURE_SHUFFLE_CONFIG["target_modality"]:
        if mod_name in cell_type_hvgs_dict:
            simulation_mdata = shuffle_cell_type_hvgs(
                simulation_mdata,
                FEATURE_SHUFFLE_CONFIG,
                cell_type_hvgs_dict[mod_name],
                dominant_cell_type_col="gt",
            )

    # 6) 构建 spot → cell 的映射，用于生成细胞级空间数据
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

    # 7) 生成细胞级空间数据（单细胞参考用）
    print("\n📐 生成细胞级空间数据（每 spot 7–13 细胞，坐标在 spot 圆内）...")
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

    n_cells_total = cell_level_mdata_dict[ref_mod].n_obs
    print(f"✅ 细胞级数据：共 {n_cells_total} 个细胞")

    # 8) 保存结果
    print("\n💾 保存结果...")

    # 单细胞参考：只保存 RNA 模态
    ref_rna_adata = cell_level_mdata_dict["rna"].copy()
    ref_rna_path = os.path.join(OUTPUT_PATH, 'ref_RNA.h5ad')
    ref_rna_adata.write_h5ad(ref_rna_path)
    print(f"✅ 单细胞参考 RNA 已保存到: {ref_rna_path}")

    # spot 级模拟 RNA
    sim_rna_path = os.path.join(OUTPUT_PATH, 'simulation_rna.h5ad')
    simulation_mdata["rna"].write_h5ad(sim_rna_path)
    print(f"✅ 空转 RNA 模拟数据已保存到: {sim_rna_path}")

    # spot 级模拟 ATAC
    sim_atac_path = os.path.join(OUTPUT_PATH, 'simulation_atac.h5ad')
    simulation_mdata["atac"].write_h5ad(sim_atac_path)
    print(f"✅ 空转 ATAC 模拟数据已保存到: {sim_atac_path}")

    # 额外：保存每个 cell type 的 HVGs 信息（可选）
    if cell_type_hvgs_dict:
        hvgs_df = pd.DataFrame([
            {'cell_type': ct, 'modality': mod, 'hvgs': ','.join(hvgs)}
            for mod, ct_hvgs in cell_type_hvgs_dict.items()
            for ct, hvgs in ct_hvgs.items()
        ])
        hvgs_path = os.path.join(OUTPUT_PATH, 'cell_type_hvgs_info.csv')
        hvgs_df.to_csv(hvgs_path, index=False)
        print(f"✅ 每个 cell type 的 HVGs 信息已保存到: {hvgs_path}")

    sampled_cells_path = os.path.join(OUTPUT_PATH, 'sampled_cells_info.csv')
    sampled_cells_df.to_csv(sampled_cells_path, index=False)
    print(f"✅ 抽样细胞信息已保存到: {sampled_cells_path}")

    print("全部完成。")


if __name__ == "__main__":
    main()