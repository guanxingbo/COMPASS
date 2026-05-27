import numpy as np
import anndata as ad
def stratified_sample_single_cell(adata, celltype_col='CellType', sample_fraction=0.5, 
                                   output_path=None, random_state=42):
    """
    对单细胞数据进行分层抽样（均衡采样）
    针对每种细胞类型按比例抽取，避免采样偏差
    
    参数:
        adata: AnnData对象
        celltype_col: 细胞类型列名
        sample_fraction: 采样比例 (0.25 或 0.5)
        output_path: 输出h5ad文件路径，如果为None则不保存
        random_state: 随机种子，用于结果复现
    
    返回:
        抽样后的AnnData对象
    """
    print(f"开始分层抽样，采样比例: {sample_fraction*100}%")
    print(f"原始数据形状: {adata.shape}")
    
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"未找到细胞类型列 '{celltype_col}'")
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 获取所有细胞类型
    cell_types = adata.obs[celltype_col].unique()
    print(f"细胞类型数量: {len(cell_types)}")
    
    # 存储抽样后的索引
    sampled_indices = []
    
    for ct in cell_types:
        # 获取该细胞类型的所有细胞索引
        ct_mask = adata.obs[celltype_col] == ct
        ct_indices = np.where(ct_mask)[0]
        n_cells = len(ct_indices)
        
        # 计算抽样数量（至少保留1个细胞）
        n_sample = max(1, int(n_cells * sample_fraction))
        
        # 随机抽样（无放回）
        sampled = np.random.choice(ct_indices, size=n_sample, replace=False)
        sampled_indices.extend(sampled)
        
        print(f"  {ct}: {n_cells} -> {n_sample} 细胞")
    
    # 根据抽样索引提取子集
    sampled_indices = sorted(sampled_indices)
    adata_sampled = adata[sampled_indices].copy()
    
    print(f"抽样后数据形状: {adata_sampled.shape}")
    
    # 保存到文件
    if output_path is not None:
        adata_sampled.write_h5ad(output_path)
        print(f"抽样数据已保存至: {output_path}")
    
    return adata_sampled

DATA_ROOT = './'
if __name__ == "__main__":
    # 读取原始单细胞数据
    adata_sc = ad.read_h5ad(DATA_ROOT+"ref_RNA.h5ad")
    celltype_col = 'cell_type'
    output_path = DATA_ROOT+"sample_sc/ref_RNA"

    # 用列表控制采样比例
    sample_fractions = [0.20, 0.30, 0.40, 0.60, 0.80]
    # 每个比例生成 5 个不同随机结果（5 个随机种子）
    
    # random_seeds = [42, 123, 456, 789, 1024]  # 或: list(range(42, 42 + n_replicates))
    random_seeds = [1024]
    n_replicates = len(random_seeds)

    for frac in sample_fractions:
        pct = int(frac * 100)
        for rep, seed in enumerate(random_seeds):
            out_path = f"{output_path}_{pct}_rep{rep}.h5ad"
            stratified_sample_single_cell(
                adata_sc,
                celltype_col=celltype_col,
                sample_fraction=frac,
                output_path=out_path,
                random_state=seed
            )



# if __name__ == "__main__":
# 	# pass
#     # 读取原始单细胞数据
#     adata_sc = ad.read_h5ad("data/atac_rna/ref_RNA.h5ad")
#     celltype_col = 'celltype'
#     output_path = "data/atac_rna/ref_RNA"
#     # 25% 分层抽样
#     adata_25 = stratified_sample_single_cell(
#         adata_sc, 
#         celltype_col=celltype_col,
#         sample_fraction=0.25,
#         output_path=f"{output_path}_25.h5ad",
#         random_state=42
#     )

#     # 40% 分层抽样
#     adata_40 = stratified_sample_single_cell(
#         adata_sc, 
#         celltype_col=celltype_col,
#         sample_fraction=0.40,
#         output_path=f"{output_path}_40.h5ad",
#         random_state=42
#     )
    
#     # 50% 分层抽样
#     adata_50 = stratified_sample_single_cell(
#         adata_sc, 
#         celltype_col=celltype_col,
#         sample_fraction=0.50,
#         output_path=f"{output_path}_50.h5ad",
#         random_state=42
#     )

#     # 60% 分层抽样
#     adata_60 = stratified_sample_single_cell(
#         adata_sc, 
#         celltype_col=celltype_col,
#         sample_fraction=0.60,
#         output_path=f"{output_path}_60.h5ad",
#         random_state=42
#     )

#     # 75% 分层抽样
#     adata_75 = stratified_sample_single_cell(
#         adata_sc, 
#         celltype_col=celltype_col,
#         sample_fraction=0.75,
#         output_path=f"{output_path}_75.h5ad",
#         random_state=42
#     )

# # 25% 分层抽样
# adata_25 = stratified_sample_single_cell(
# 	adata_sc, 
# 	celltype_col='CellType',
# 	sample_fraction=0.25,
# 	output_path="path/to/single_cell_25pct.h5ad",
# 	random_state=42
# )
# adata_sc = ad.read_h5ad("./ref_RNA.h5ad")