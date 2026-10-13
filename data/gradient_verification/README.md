# Interior gradient verification

This folder holds reusable scripts for checking whether graph-neighborhood
least-squares estimates can recover the CFD spatial gradients stored in the
interior dataset.

Main script:

- `verify_interior_gradients.py`: sweeps sample ratios, estimates all nine
  velocity-gradient components plus the three pressure-gradient components,
  and writes metrics and plots to the dedicated sibling directory
  `<parent_of_interior_dir>/angioflow_gradient_verification/` by default.

What it does:

1. loads the full validated interior pool with `edge_index`, `grad_v`, and
   `grad_p`;
2. draws the same random interior subset rule used during training;
3. induces the subgraph on the sampled nodes;
4. fits a local least-squares gradient at each sampled point from its graph
   neighbors;
5. compares the estimate with CFD ground truth and saves plots/metrics.

The fit uses float64 arithmetic, inverse-distance-squared neighbor weighting,
local RMS coordinate scaling, dimensionless ridge regularization, and a 3D
stencil-conditioning check. These details matter because positions are stored
in metres while absolute pressure is approximately 1e5 Pa.

By default, verification evaluates 100 unique cases, balanced across requested
versions (approximately 50 AneuG-Flow and 50 AngioFlow), rather than scanning the full dataset.
It logs every completed case with elapsed time and an ETA; use `--log_every N`
to print every Nth case, or `--log_every 0` to disable per-case progress logs.

To create ground-truth/estimate/error pictures without recomputing aggregate
metrics, use `--visualize_only`; it renders every ratio from `--ratios` and
defaults to 50 cases per version. This saves both central-slab and full-3D views
for all velocity- and pressure-gradient components, plus gradient magnitudes,
under `visualizations/ratio_*/<version>_<case>_frame*/`. Complete folders are
skipped on rerun, so an interrupted visualization job resumes automatically.

Complete usage:

```bash
conda activate new
export MPLCONFIGDIR=/tmp/angioflow-mpl

# Aggregate metrics for 100 balanced cases.
python -m angioflow.data.gradient_verification.verify_interior_gradients \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --versions aneug_flow,angioflow --frames 0 \
  --ratios 0.05,0.1,0.2,0.4,0.8,1.0 \
  --max_cases 100 --max_points none --log_every 1

# All-ratio slab and 3D visualizations. Safe to rerun after interruption.
python -m angioflow.data.gradient_verification.verify_interior_gradients \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --versions aneug_flow,angioflow --frames 0 \
  --ratios 0.05,0.1,0.2,0.4,0.8,1.0 \
  --max_cases 100 --max_points none \
  --visualize_only --visualize_cases_per_version 50 \
  --max_plot_points 12000
```

Outputs are written to:

- `.../datasets/angioflow_gradient_verification/summary.json`
- `.../datasets/angioflow_gradient_verification/summary.csv`
- `.../datasets/angioflow_gradient_verification/relative_mse_vs_ratio.png`
- `.../datasets/angioflow_gradient_verification/component_relative_mse_heatmap.png`
- `.../datasets/angioflow_gradient_verification/ratio_*/representative_case/`
- `.../datasets/angioflow_gradient_verification/visualizations/ratio_*/v*_<case>_frame*/`
