# COMPASS

**Cross-omics Mapping for Precise Assembly of Spatial Single-cell atlases (COMPASS)** reconstructs cell-resolved spatial maps by integrating spot-level spatial omics with dissociated single-cell RNA-seq, optionally using a co-registered auxiliary spatial modality (for example chromatin accessibility) to sharpen cross-modal alignment. The method learns a shared latent representation, performs domain-aware cell-to-spot assignment via regularized optimal transport with discrete refinement, and refines sub-spot coordinates so mapped cells form a dense, minimally overlapping layout suitable for spatial analysis at single-cell resolution.

This repository contains a script-oriented Python implementation used alongside the simulation notebooks, together with utilities for visualization and mapping.


## Method overview (concise)

The peer-reviewed manuscript describes COMPASS in three coupled parts: (i) a domain-aware representation module that aligns spots and dissociated cells in a shared RNA-centered latent space while transferring auxiliary spatial omics information where applicable (Residual-Gated Cross-Omics Distillation, RG-COD); (ii) a mapping module that stratifies cells and spots by predicted domain and cell type and solves entropy-regularized optimal transport within compatible strata, followed by capacity-constrained hard assignment; (iii) a geometric refinement step (Poisson-disk Geometric Refinement, PDGR) that assigns continuous coordinates within the spot-supported tissue region while encouraging minimum separation and type-coherent neighborhoods.

## Relationship between this repository and the manuscript

The paper additionally reports hidden Markov random field (HMRF)–based tissue domain initialization and full benchmark protocols across simulations and real datasets. In this repository, the graph autoencoder training path in `COMPASS/COMPASS_core.py` is a **minimal, script-friendly** port derived from the Scenario 4 multimodal notebook. It expects **spatial domain labels** on the spot-level RNA object (for example in `adata.obs['gt']`, which the driver copies to `domain` when needed), which matches the bundled simulation-style workflow. For full parity with every analysis in the manuscript, use the scenario notebooks under `Scenario1`–`Scenario4` or extend `run_compass_model` in `COMPASS_run.py` as noted in that file’s docstring.

## Repository layout

| Path | Description |
|------|-------------|
| `COMPASS/` | Python modules: data loading (`COMPASS_data.py`), model training and embedding inference (`COMPASS_core.py`), optimal transport, assignment, refinement, and plotting (`COMPASS_method.py`), and the entry script `COMPASS_run.py`. |
| `Scenario1/` … `Scenario4/` | Jupyter notebooks for single-omic and multi-omic analyses across the four simulation designs described in the paper. |
| `data_gen/` | Scripts for generating synthetic spatial and reference data used in simulations. |

## Requirements

- Python 3.10 or newer is recommended (compatible with current `scanpy` / `anndata` stacks).
- Core packages: `numpy`, `pandas`, `anndata`, `scanpy`, `scikit-learn`, `scipy`, `matplotlib`, `torch`.

PyTorch may use CUDA if available; the optimal-transport routines in `COMPASS_method.py` will select CUDA when present. CPU execution is supported but slower for large tensors.

Install dependencies with your preferred environment manager, for example:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
pip install numpy pandas anndata scanpy scikit-learn scipy matplotlib torch
```

Adjust package versions to match your institutional or cluster policy.

## Input data (`AnnData`)

`COMPASS_data.load_and_preprocess` reads three HDF5-backed AnnData objects (`.h5ad`).

**General constraints**

- **Gene names** must be unique and overlap between spatial RNA and single-cell RNA (non-empty intersection), or loading raises an error.
- **Spot identifiers** (`obs_names`) must match between spatial RNA and the second spatial modality (non-empty intersection), and must be **unique** on spatial objects.

**Single-cell reference (`sc_adata`)**

- Cell type column: one of `CellType`, `cell_type`, or `cellType` (normalized internally to `CellType`).

**Spatial RNA (`adata`, spot-level)**

- `obsm['spatial']`: two-dimensional coordinates (used by the graph autoencoder and downstream figures).
- Domain supervision for training: `obs['domain']`, or ground-truth / pseudo-labels in `obs['gt']` (the driver maps `gt` to `domain` when `domain` is absent).

**Second modality on spots (`adt_adata`, for example ATAC)**

- Same spots as spatial RNA (aligned `obs_names`). Feature matrix is treated according to `mod2_kind` in `train_multimodal_and_predict_sc` (default `"atac"` in `COMPASS_run.py`).

**After `run_compass_model` (or equivalent precomputation)**

The figure pipeline expects, at minimum:

| Object | Field | Role |
|--------|--------|------|
| `sc_adata` | `obsm['gae_latent']` | Shared latent embedding |
| `sc_adata` | `obsm['pred_domain_proba']` | Domain probability matrix |
| `sc_adata` | `obs['pred_domain']` | Predicted domain labels |
| `sc_adata` | `obs['CellType']` | Cell types |
| `adata_prep` (ST) | `obsm['gae_latent']` | Spot latent embedding |
| `adata_prep` | `obsm['spatial']` | Spatial coordinates |
| `adata_prep` | `obs['domain']` | Domain labels for OT stratification |

If `compass_outputs_ready` is true, `run_compass_model(..., skip_if_ready=True)` skips training (useful for precomputed `.h5ad` files).

## Running the default pipeline

Default filenames are resolved relative to the `COMPASS/` directory (next to `COMPASS_run.py`): `ref_RNA.h5ad`, `simulation_rna_drop.h5ad`, `simulation_atac.h5ad`.

```bash
cd COMPASS
python COMPASS_run.py
```

The script loads data, sets the global seed via `load_and_preprocess(..., run_seed=123)`, runs the multimodal model when outputs are not already present, then executes `run_four_figure_pipeline`: PCA of the GAE latent (cells filtered by maximum domain probability), stacked bar chart of cell types by predicted domain, spot-level composition pies after soft assignment, and cell-level spatial scatter after geometric refinement.

`main()` in `COMPASS_run.py` currently accepts optional `Path` arguments in code but the `if __name__ == "__main__"` block invokes `main()` without command-line arguments. To use custom paths, import `main` from another module or edit the call, for example:

```python
from pathlib import Path
from COMPASS_run import main
main(Path("my_sc.h5ad"), Path("my_st_rna.h5ad"), Path("my_st_atac.h5ad"))
```

(Run with the `COMPASS` package directory on `PYTHONPATH`, or execute from within that directory.)

## Reproducibility

- `load_and_preprocess(..., run_seed=123)` invokes `set_global_seed` with deterministic PyTorch behavior where supported.
- Several steps fix random seeds in code (for example `PCA(..., random_state=0)` in `COMPASS_run.py`, Poisson-disk–style refinement with `seed=0`, and `pca_random_state=0` in `train_multimodal_and_predict_sc`). Slight numerical differences may still appear across hardware and library versions.

## License

No `LICENSE` file is included in this repository. Redistribution and reuse terms should be obtained from the authors or added here when decided.


<!-- ## Authors
Xuanwu Wang<sup>1,2#</sup>, Wenjun Fang<sup>1#</sup>, Yuqi Wang<sup>3#</sup>, Xingbo Guan<sup>2#</sup>, Nuo Li<sup>2</sup>, Yihao Bai<sup>4</sup>, Lixin Liang<sup>1</sup>, Heng Peng<sup>2</sup>, Wei Liu<sup>4*</sup>, Qishi Dong<sup>1*</sup>

Affiliations: <sup>1</sup>School of Artificial Intelligence, Shenzhen Technology University, Shenzhen 518118, China; <sup>2</sup>Department of Mathematics, Hong Kong Baptist University, Hong Kong; <sup>3</sup>Department of Statistics and Data Science, Beijing Normal–Hong Kong Baptist University, Zhuhai 519087, China; <sup>4</sup>School of Mathematics, Sichuan University, Chengdu 610065, China. -->


## Contact

For questions about the method or this code release, please contact the corresponding authors: Wei Liu ([liuwei8@scu.edu.cn](mailto:liuwei8@scu.edu.cn)), or Qishi Dong ([dongqishi@sztu.edu.cn](mailto:dongqishi@sztu.edu.cn)).
