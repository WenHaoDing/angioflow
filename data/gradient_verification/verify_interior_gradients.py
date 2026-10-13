"""Verify whether graph-neighborhood least squares can recover CFD gradients.

This script mirrors the training-time interior sampling logic:
  * load the full validated interior pool and the CFD gradient targets;
  * draw the same random point subset rule used by InteriorSubsample
    (sample_ratio + optional max_points cap);
  * induce the subgraph on those sampled points;
  * estimate local velocity/pressure gradients from graph neighborhoods with
    a least-squares fit;
  * compare the estimates against the CFD ground truth for all 12 gradient
    components (9 for velocity, 3 for pressure).

Outputs are written to a dedicated sibling of the interior dataset by default:
<parent_of_interior_dir>/angioflow_gradient_verification/.
  * metrics.json / metrics.csv with per-ratio summary statistics;
  * plots that show component-wise MSE vs. sample ratio;
  * per-ratio case figures for manual inspection of the spatial patterns.

Complete usage (run metrics first, then the resumable visualization sweep):
  conda activate new
  export MPLCONFIGDIR=/tmp/angioflow-mpl

  # 100-case balanced metric sweep (approximately 50 AneuG-Flow + 50 AngioFlow)
  python -m angioflow.data.gradient_verification.verify_interior_gradients \
    --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
    --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
    --versions aneug_flow,angioflow --frames 0 \
    --ratios 0.1,0.2,0.4,0.8,1.0 \
    --max_cases 100 --max_points none --log_every 1

  # Slab + 3D GT/estimate/error views for the same ratios and case balance
  python -m angioflow.data.gradient_verification.verify_interior_gradients \
    --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
    --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
    --versions aneug_flow,angioflow --frames 0 \
    --ratios 0.1,0.2,0.4,0.8,1.0 \
    --max_cases 100 --max_points none \
    --visualize_only --visualize_cases_per_version 50 \
    --max_plot_points 12000
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from angioflow.config.defaults import add_data_args
from angioflow.data.datasets import load_excluded_keys
from angioflow.data.datasets_interior import (
    AngioFlowWallInteriorDataset,
    AngioFlowWallInteriorDatasetT0,
)


VEL_GRAD_NAMES = [
    "dvx_dx", "dvx_dy", "dvx_dz",
    "dvy_dx", "dvy_dy", "dvy_dz",
    "dvz_dx", "dvz_dy", "dvz_dz",
]
PRESS_GRAD_NAMES = ["dp_dx", "dp_dy", "dp_dz"]
ALL_COMPONENT_NAMES = VEL_GRAD_NAMES + PRESS_GRAD_NAMES


def parse_optional_positive_int(text):
    if str(text).strip().lower() in {"none", "null", "unlimited", "off"}:
        return None
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer or 'none'")
    return value


@dataclass
class SampleResult:
    ratio: float
    sample_count: int
    edge_count: int
    mean_degree: float
    valid_count: int
    valid_fraction: float
    vel_mse: np.ndarray
    vel_mae: np.ndarray
    vel_rel_mse: np.ndarray
    press_mse: np.ndarray
    press_mae: np.ndarray
    press_rel_mse: np.ndarray


def get_args():
    parser = argparse.ArgumentParser(
        description="Sweep interior sample ratios and verify graph-based gradient estimates.")
    add_data_args(parser)
    parser.add_argument(
        "--ratios",
        default="0.05,0.1,0.2,0.4,0.8,1.0",
        help="comma-separated sample ratios to evaluate",
    )
    parser.add_argument(
        "--max_points",
        type=parse_optional_positive_int,
        default=None,
        help="optional sampled-point cap; default 'none' uses the true requested ratio",
    )
    parser.add_argument(
        "--min_neighbors",
        type=int,
        default=3,
        help="minimum outgoing neighbors required for a least-squares estimate",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=3e-2,
        help="dimensionless Tikhonov regularization after local coordinate scaling",
    )
    parser.add_argument(
        "--distance_power",
        type=float,
        default=2.0,
        help="weight each neighbor by inverse distance to this power (0 disables weighting)",
    )
    parser.add_argument(
        "--condition_tol",
        type=float,
        default=1e-6,
        help="reject a stencil when min/max eigenvalue of its Gram matrix is below this",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="base seed used to build the same random subset rule as training",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=100,
        help="number of unique cases to evaluate, balanced across requested versions",
    )
    parser.add_argument(
        "--plot_case_index",
        type=int,
        default=0,
        help="dataset-sample index used for the detailed pattern plots",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="print detailed progress every N completed cases (0 disables per-case logs)",
    )
    parser.add_argument(
        "--visualize_only",
        action="store_true",
        help="render representative gradient comparisons without recomputing aggregate metrics",
    )
    parser.add_argument(
        "--visualize_ratio",
        type=float,
        default=None,
        help="deprecated single-ratio override for --visualize_only; default uses --ratios",
    )
    parser.add_argument(
        "--visualize_ratios",
        default=None,
        help="comma-separated visualization ratios; defaults to the full --ratios list",
    )
    parser.add_argument(
        "--visualize_cases_per_version",
        type=int,
        default=50,
        help="representative cases rendered per version by --visualize_only",
    )
    parser.add_argument(
        "--max_plot_points",
        type=int,
        default=12000,
        help="maximum valid points shown in each 3D panel",
    )
    parser.add_argument(
        "--skip_existing_visualizations",
        action="store_true",
        default=True,
        help="skip complete case-ratio visualization folders when resuming",
    )
    parser.add_argument(
        "--no_skip_existing_visualizations",
        dest="skip_existing_visualizations",
        action="store_false",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help=("output directory root; defaults to "
              "<parent_of_interior_dir>/angioflow_gradient_verification"),
    )
    parser.add_argument(
        "--data_pool_ratio",
        type=float,
        default=0.16,
        help="on-disk interior pool fraction (used only if preprocessing is enabled)",
    )
    parser.add_argument(
        "--interior_dir",
        default=None,
        help="interior data directory; defaults to <save_dir>/interior",
    )
    parser.add_argument(
        "--freeze_dataset",
        action="store_true",
        default=True,
        help="use only already-processed cases and never process raw data",
    )
    parser.add_argument(
        "--no_freeze_dataset",
        dest="freeze_dataset",
        action="store_false",
    )
    return parser.parse_args()


def build_dataset(args, exclude_keys):
    ds_kwargs = dict(
        case_root_angioflow=args.case_root_angioflow,
        case_root_aneug_flow=args.case_root_aneug_flow,
        save_dir=args.save_dir,
        n_frames=args.n_frames,
        decimate_ratio=args.decimate_ratio,
        decimate_method=args.decimate_method,
        lap_k=args.lap_k,
        exclude_keys=exclude_keys,
        interior_dir=args.interior_dir,
        pool_ratio=args.data_pool_ratio,
        sample_ratio=1.0,
        sanity_first=0,
        freeze_dataset=args.freeze_dataset,
        load_edge_index=True,
        load_grad=True,
    )
    versions = args.versions.split(",") if args.versions else None
    if args.frames == "all":
        return AngioFlowWallInteriorDataset(**ds_kwargs)
    frames = [int(t) for t in args.frames.split(",")]
    return AngioFlowWallInteriorDatasetT0(**ds_kwargs, versions=versions, frames=frames)


def parse_ratios(text):
    ratios = [float(item) for item in text.split(",") if item.strip()]
    if not ratios:
        raise ValueError("--ratios must contain at least one value")
    if any(r <= 0 or r > 1 for r in ratios):
        raise ValueError("all sample ratios must lie in (0, 1]")
    return ratios


def make_ratio_dir(output_root, ratio):
    token = f"{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    path = os.path.join(output_root, f"ratio_{token}")
    os.makedirs(path, exist_ok=True)
    return path


def draw_training_subset(num_nodes, ratio, max_points, seed):
    k = max(1, int(round(num_nodes * ratio)))
    if max_points is not None:
        k = min(k, max_points)
    k = min(k, num_nodes)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(num_nodes, generator=generator)
    return perm[:k]


def induced_sample(wall, interior, ratio, max_points, seed):
    idx = draw_training_subset(interior.num_nodes, ratio, max_points, seed)
    sampled = interior.subgraph(idx)
    return wall, sampled


def select_case_indices(dataset, max_cases=100, seed=0):
    """Choose one sample per case, balanced across dataset versions."""
    first_index = {}
    for index, (version, case_id, _frame) in enumerate(dataset._index):
        first_index.setdefault((version, case_id), index)

    per_version = {}
    for (version, case_id), index in first_index.items():
        per_version.setdefault(version, []).append((case_id, index))

    rng = random.Random(seed)
    for items in per_version.values():
        items.sort()
        rng.shuffle(items)

    total_available = sum(len(items) for items in per_version.values())
    if max_cases is None or max_cases >= total_available:
        return sorted(index for items in per_version.values() for _, index in items)
    if max_cases < 1:
        raise ValueError("--max_cases must be positive or omitted")

    versions = sorted(per_version)
    quota, remainder = divmod(max_cases, len(versions))
    selected = []
    unused = []
    for version_index, version in enumerate(versions):
        target = quota + (version_index < remainder)
        items = per_version[version]
        selected.extend(index for _, index in items[:target])
        unused.extend(index for _, index in items[target:])

    # If one version has fewer cases than its quota, fill from the others.
    if len(selected) < max_cases:
        rng.shuffle(unused)
        selected.extend(unused[:max_cases - len(selected)])
    return sorted(selected)


def estimate_local_gradients(interior, ridge=3e-2, min_neighbors=3,
                             distance_power=2.0, condition_tol=1e-6):
    """Estimate dvi/dxj and dp/dxj with a conditioned, weighted local fit.

    Coordinates are scaled by each stencil's RMS radius before solving, making
    ``ridge`` dimensionless and independent of whether positions are expressed
    in metres or millimetres. The result is converted back to derivatives with
    respect to the original coordinates. Float64 arithmetic is intentional:
    pressure is absolute (~1e5 Pa), so neighbor differences lose accuracy in
    float32 even though the stored source data cannot regain discarded bits.
    """
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if distance_power < 0:
        raise ValueError("distance_power must be non-negative")
    if condition_tol < 0:
        raise ValueError("condition_tol must be non-negative")

    pos = interior.pos.double().cpu()
    vel = interior.vel.double().cpu()
    press = interior.press.double().cpu()
    edge_index = interior.edge_index.cpu()

    num_nodes = interior.num_nodes
    adjacency = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.t().tolist():
        if src != dst:
            adjacency[src].append(dst)

    est_v = torch.full((num_nodes, 9), float("nan"), dtype=torch.float64)
    est_p = torch.full((num_nodes, 3), float("nan"), dtype=torch.float64)
    valid = torch.zeros(num_nodes, dtype=torch.bool)
    eye = torch.eye(3, dtype=pos.dtype)

    for node_id, neighbors in enumerate(adjacency):
        if len(neighbors) < min_neighbors:
            continue

        nbr = torch.tensor(neighbors, dtype=torch.long)
        dx = pos[nbr] - pos[node_id]
        dv = vel[nbr] - vel[node_id]
        dp = press[nbr] - press[node_id]

        scale = dx.square().sum(dim=1).mean().sqrt()
        if not torch.isfinite(scale) or scale <= 0:
            continue
        x = dx / scale
        distance = x.norm(dim=1).clamp_min(torch.finfo(x.dtype).eps)
        sqrt_weight = distance.pow(-0.5 * distance_power).unsqueeze(1)
        x_weighted = x * sqrt_weight
        gram = x_weighted.T @ x_weighted
        eigenvalues = torch.linalg.eigvalsh(gram)
        if eigenvalues[-1] <= 0 or eigenvalues[0] <= condition_tol * eigenvalues[-1]:
            continue

        try:
            regularized = gram + ridge * eye
            vel_sol = torch.linalg.solve(regularized, x_weighted.T @ (dv * sqrt_weight)) / scale
            press_sol = torch.linalg.solve(regularized, x_weighted.T @ (dp * sqrt_weight)) / scale
        except RuntimeError:
            continue

        if not (torch.isfinite(vel_sol).all() and torch.isfinite(press_sol).all()):
            continue

        est_v[node_id] = vel_sol.T.reshape(-1)
        est_p[node_id] = press_sol[:, 0]
        valid[node_id] = True

    return est_v, est_p, valid


def summarize_components(est, truth, valid):
    if valid.sum() == 0:
        raise RuntimeError("no valid nodes found for the least-squares estimate")

    diff = est[valid] - truth[valid]
    mse = diff.pow(2).mean(dim=0)
    mae = diff.abs().mean(dim=0)
    denom = truth[valid].pow(2).mean(dim=0).clamp_min(1e-12)
    rel_mse = mse / denom
    return mse.numpy(), mae.numpy(), rel_mse.numpy(), int(valid.sum().item())


def summarize_graph(edge_index, num_nodes):
    if edge_index.numel() == 0:
        return 0.0, 0
    deg = torch.bincount(edge_index[0], minlength=num_nodes)
    return float(deg.float().mean().item()), int(edge_index.shape[1])


def render_component_grid(pos, truth, estimate, names, valid_mask, output_path, title, mode="field"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_mask = np.asarray(valid_mask, dtype=bool)
    ext = pos.max(0) - pos.min(0)
    axis = int(np.argmin(ext))
    slab_frac = 0.05
    slab_mask = np.abs(pos[:, axis] - np.median(pos[:, axis])) < slab_frac * float(ext[axis] if ext[axis] > 0 else 1.0)
    slab_mask &= valid_mask
    if slab_mask.sum() == 0:
        slab_mask = valid_mask.copy()
    coords = pos[slab_mask]
    proj_axes = [a for a in (0, 1, 2) if a != axis]

    n = len(names)
    rows = int(math.ceil(n / 3))
    fig = plt.figure(figsize=(18, 5 * rows))
    fig.suptitle(title, fontsize=14)

    for i, name in enumerate(names):
        ax = fig.add_subplot(rows, 3, i + 1)
        if mode == "error":
            values = np.abs((estimate - truth)[slab_mask, i])
            vmax = max(float(np.nanpercentile(values, 98)), 1e-12) if values.size else 1.0
            sc = ax.scatter(coords[:, proj_axes[0]], coords[:, proj_axes[1]], c=values,
                            s=7, cmap="magma", vmin=0.0, vmax=vmax, linewidths=0)
        else:
            values = truth[slab_mask, i] if mode == "truth" else estimate[slab_mask, i]
            if values.size:
                lim = max(float(np.nanpercentile(np.abs(values), 98)), 1e-12)
            else:
                lim = 1.0
            sc = ax.scatter(coords[:, proj_axes[0]], coords[:, proj_axes[1]], c=values,
                            s=7, cmap="RdBu_r", vmin=-lim, vmax=lim, linewidths=0)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, shrink=0.7)

    for j in range(n, rows * 3):
        ax = fig.add_subplot(rows, 3, j + 1)
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _safe_filename(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def render_field_comparison(pos, truth, estimate, valid_mask, name, output_dir,
                            signed=True, max_plot_points=12000, seed=0):
    """Save slab and full-3D GT/estimate/absolute-error comparisons for one field."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = np.asarray(pos)
    truth = np.asarray(truth)
    estimate = np.asarray(estimate)
    valid = np.asarray(valid_mask, dtype=bool)
    valid &= np.isfinite(truth) & np.isfinite(estimate)
    if not valid.any():
        raise RuntimeError(f"no finite valid points available to render {name}")

    error = np.abs(estimate - truth)
    joint = np.concatenate([truth[valid], estimate[valid]])
    if signed:
        field_lim = max(float(np.nanpercentile(np.abs(joint), 98)), 1e-12)
        field_vmin, field_vmax, field_cmap = -field_lim, field_lim, "RdBu_r"
    else:
        field_vmin = 0.0
        field_vmax = max(float(np.nanpercentile(joint, 98)), 1e-12)
        field_cmap = "viridis"
    error_max = max(float(np.nanpercentile(error[valid], 98)), 1e-12)

    ext = pos.max(axis=0) - pos.min(axis=0)
    slab_axis = int(np.argmin(ext))
    thickness = 0.05 * float(ext[slab_axis] if ext[slab_axis] > 0 else 1.0)
    slab = valid & (np.abs(pos[:, slab_axis] - np.median(pos[:, slab_axis])) < thickness)
    if not slab.any():
        slab = valid.copy()
    projection_axes = [axis for axis in (0, 1, 2) if axis != slab_axis]

    os.makedirs(output_dir, exist_ok=True)
    filename = _safe_filename(name)
    panels = (("Ground truth", truth, field_cmap, field_vmin, field_vmax),
              ("Estimate", estimate, field_cmap, field_vmin, field_vmax),
              ("Absolute error", error, "magma", 0.0, error_max))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (panel_name, values, cmap, vmin, vmax) in zip(axes, panels):
        scatter = ax.scatter(
            pos[slab, projection_axes[0]], pos[slab, projection_axes[1]],
            c=values[slab], s=7, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(panel_name)
        ax.set_xlabel("xyz"[projection_axes[0]])
        ax.set_ylabel("xyz"[projection_axes[1]])
        ax.set_aspect("equal")
        fig.colorbar(scatter, ax=ax, shrink=0.78)
    fig.suptitle(
        f"{name}: central {'xyz'[slab_axis]} slab ({int(slab.sum())} points)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(output_dir, f"{filename}_slab.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    indices = np.flatnonzero(valid)
    if max_plot_points is not None and len(indices) > max_plot_points:
        indices = np.random.default_rng(seed).choice(
            indices, size=max_plot_points, replace=False)
    fig = plt.figure(figsize=(20, 6.5))
    for panel_index, (panel_name, values, cmap, vmin, vmax) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, panel_index, projection="3d")
        scatter = ax.scatter(
            pos[indices, 0], pos[indices, 1], pos[indices, 2], c=values[indices],
            s=1.2, alpha=0.28, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0,
            depthshade=False)
        ax.set_title(panel_name)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_box_aspect(np.maximum(ext, 1e-12))
        ax.view_init(elev=20, azim=-60)
        fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.08)
    fig.suptitle(f"{name}: full 3D ({len(indices)} displayed points)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(output_dir, f"{filename}_3d.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def run_visualization_only(args, dataset, case_indices, output_root):
    if args.visualize_ratio is not None:
        visualization_ratios = [args.visualize_ratio]
    elif args.visualize_ratios is not None:
        visualization_ratios = parse_ratios(args.visualize_ratios)
    else:
        visualization_ratios = parse_ratios(args.ratios)
    if any(ratio <= 0 or ratio > 1 for ratio in visualization_ratios):
        raise ValueError("visualization ratios must lie in (0, 1]")
    if args.visualize_cases_per_version < 1:
        raise ValueError("--visualize_cases_per_version must be positive")
    if args.max_plot_points is not None and args.max_plot_points < 1:
        raise ValueError("--max_plot_points must be positive")

    chosen = {}
    for dataset_index in case_indices:
        version = dataset._index[dataset_index][0]
        if len(chosen.get(version, [])) < args.visualize_cases_per_version:
            chosen.setdefault(version, []).append(dataset_index)

    cases_per_ratio = sum(len(indices) for indices in chosen.values())
    total = len(visualization_ratios) * cases_per_ratio
    total_images = total * (len(VEL_GRAD_NAMES) + len(PRESS_GRAD_NAMES) + 2) * 2
    visualization_root = os.path.join(output_root, "visualizations")
    print(f"[visualize] ratios={visualization_ratios} cases_per_ratio={cases_per_ratio} "
          f"by_version={dict((v, len(i)) for v, i in chosen.items())}", flush=True)
    print(f"[visualize] jobs={total} expected_pngs={total_images} root={visualization_root}",
          flush=True)

    completed = 0
    visualization_start = time.perf_counter()
    for ratio in visualization_ratios:
        ratio_token = f"{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")
        ratio_root = os.path.join(visualization_root, f"ratio_{ratio_token}")
        for version in sorted(chosen):
            for dataset_index in chosen[version]:
                _version, case_id, frame = dataset._index[dataset_index]
                job_start = time.perf_counter()
                print(f"[visualize] job={completed + 1}/{total} ratio={ratio:g} "
                      f"{version}/{case_id} frame={frame} preparing", flush=True)
                case_dir = os.path.join(
                    ratio_root, f"{version}_{_safe_filename(str(case_id))}_frame{frame}")
                expected_pngs = (len(VEL_GRAD_NAMES) + len(PRESS_GRAD_NAMES) + 2) * 2
                metadata_path = os.path.join(case_dir, "metadata.json")
                if args.skip_existing_visualizations and os.path.exists(metadata_path):
                    existing_pngs = sum(
                        filename.endswith(".png")
                        for _root, _dirs, files in os.walk(case_dir)
                        for filename in files)
                    with open(metadata_path, encoding="utf-8") as metadata_file:
                        existing_metadata = json.load(metadata_file)
                    configuration_matches = (
                        existing_metadata.get("ratio") == ratio
                        and existing_metadata.get("max_points", "missing") == args.max_points
                        and existing_metadata.get("ridge") == args.ridge
                        and existing_metadata.get("distance_power") == args.distance_power
                        and existing_metadata.get("condition_tol") == args.condition_tol
                    )
                    if existing_pngs == expected_pngs and configuration_matches:
                        completed += 1
                        elapsed = time.perf_counter() - visualization_start
                        eta = elapsed / completed * (total - completed)
                        print(f"[visualize] job={completed}/{total} skip_complete={case_dir} "
                              f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)
                        continue
                print(f"[visualize] job={completed + 1}/{total} estimating gradients",
                      flush=True)
                wall, interior = dataset[dataset_index]
                _, sampled = induced_sample(
                    wall, interior, ratio, args.max_points, args.seed + dataset_index)
                est_v, est_p, valid = estimate_local_gradients(
                    sampled, ridge=args.ridge, min_neighbors=args.min_neighbors,
                    distance_power=args.distance_power, condition_tol=args.condition_tol)

                pos = sampled.pos.double().cpu().numpy()
                truth_v = sampled.grad_v.double().cpu().numpy()
                truth_p = sampled.grad_p.double().cpu().numpy()
                estimate_v = est_v.numpy()
                estimate_p = est_p.numpy()
                valid_np = valid.numpy()

                for component, name in enumerate(VEL_GRAD_NAMES):
                    render_field_comparison(
                        pos, truth_v[:, component], estimate_v[:, component], valid_np,
                        name, os.path.join(case_dir, "velocity_components"), signed=True,
                        max_plot_points=args.max_plot_points,
                        seed=args.seed + component)
                for component, name in enumerate(PRESS_GRAD_NAMES):
                    render_field_comparison(
                        pos, truth_p[:, component], estimate_p[:, component], valid_np,
                        name, os.path.join(case_dir, "pressure_components"), signed=True,
                        max_plot_points=args.max_plot_points,
                        seed=args.seed + 100 + component)

                render_field_comparison(
                    pos, np.linalg.norm(truth_v.reshape(-1, 3, 3), axis=(1, 2)),
                    np.linalg.norm(estimate_v.reshape(-1, 3, 3), axis=(1, 2)), valid_np,
                    "velocity_gradient_frobenius_norm", os.path.join(case_dir, "magnitudes"),
                    signed=False, max_plot_points=args.max_plot_points, seed=args.seed + 200)
                render_field_comparison(
                    pos, np.linalg.norm(truth_p, axis=1), np.linalg.norm(estimate_p, axis=1),
                    valid_np, "pressure_gradient_magnitude", os.path.join(case_dir, "magnitudes"),
                    signed=False, max_plot_points=args.max_plot_points, seed=args.seed + 201)

                with open(os.path.join(case_dir, "metadata.json"), "w", encoding="utf-8") as file:
                    json.dump({
                        "version": version, "case_id": case_id, "frame": frame,
                        "ratio": ratio, "sample_count": sampled.num_nodes,
                        "max_points": args.max_points,
                        "valid_count": int(valid.sum()),
                        "valid_fraction": float(valid.double().mean()),
                        "ridge": args.ridge, "distance_power": args.distance_power,
                        "condition_tol": args.condition_tol,
                    }, file, indent=2)
                # Matplotlib's 3D renderer retains sizeable cyclic object graphs;
                # release them between jobs so hundreds of cases do not grow RAM.
                gc.collect()
                completed += 1
                elapsed = time.perf_counter() - visualization_start
                eta = elapsed / completed * (total - completed)
                print(f"[visualize] job={completed}/{total} wrote={case_dir} "
                      f"points={sampled.num_nodes} valid={valid.double().mean():.3f} "
                      f"job_time={time.perf_counter() - job_start:.1f}s "
                      f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)
    print(f"[visualize] done: {visualization_root}", flush=True)


def render_ratio_summary(output_root, ratios, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio_vals = np.asarray(ratios, dtype=float)
    vel_mean_rel_mse = np.asarray([r.vel_rel_mse.mean() for r in results])
    press_mean_rel_mse = np.asarray([r.press_rel_mse.mean() for r in results])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ratio_vals, vel_mean_rel_mse, marker="o", label="velocity gradients")
    ax.plot(ratio_vals, press_mean_rel_mse, marker="o", label="pressure gradients")
    ax.set_yscale("log")
    ax.set_xlabel("sample ratio")
    ax.set_ylabel("mean component relative MSE")
    ax.set_title("Normalized gradient-estimation error vs. sample ratio")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_root, "relative_mse_vs_ratio.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    heatmap = np.stack(
        [np.concatenate([r.vel_rel_mse, r.press_rel_mse]) for r in results], axis=1)
    fig, ax = plt.subplots(figsize=(max(8, len(ratios) * 1.1), 8))
    im = ax.imshow(np.log10(np.clip(heatmap, 1e-20, None)), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_yticks(np.arange(len(ALL_COMPONENT_NAMES)))
    ax.set_yticklabels(ALL_COMPONENT_NAMES)
    ax.set_xlabel("sample ratio")
    ax.set_title("log10(component relative MSE) by sample ratio")
    fig.colorbar(im, ax=ax, label="log10(relative MSE)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_root, "component_relative_mse_heatmap.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def to_jsonable(result: SampleResult):
    return {
        "ratio": result.ratio,
        "sample_count": result.sample_count,
        "edge_count": result.edge_count,
        "mean_degree": result.mean_degree,
        "valid_count": result.valid_count,
        "valid_fraction": result.valid_fraction,
        "velocity": {
            "component_names": VEL_GRAD_NAMES,
            "mse": result.vel_mse.tolist(),
            "mae": result.vel_mae.tolist(),
            "rel_mse": result.vel_rel_mse.tolist(),
            "mean_mse": float(result.vel_mse.mean()),
        },
        "pressure": {
            "component_names": PRESS_GRAD_NAMES,
            "mse": result.press_mse.tolist(),
            "mae": result.press_mae.tolist(),
            "rel_mse": result.press_rel_mse.tolist(),
            "mean_mse": float(result.press_mse.mean()),
        },
    }


def main():
    args = get_args()
    ratios = parse_ratios(args.ratios)
    if args.log_every < 0:
        raise ValueError("--log_every must be non-negative")

    exclude_keys = load_excluded_keys(args.save_dir) if args.exclude_anomalies else set()
    dataset = build_dataset(args, exclude_keys)

    case_indices = select_case_indices(dataset, args.max_cases, args.seed)

    interior_root = args.interior_dir or os.path.join(args.save_dir, "interior")
    default_output_root = os.path.join(
        os.path.dirname(os.path.abspath(interior_root)), "angioflow_gradient_verification")
    output_root = args.output_root or default_output_root
    os.makedirs(output_root, exist_ok=True)

    selected_versions = {}
    for dataset_index in case_indices:
        version, _case_id, _frame = dataset._index[dataset_index]
        selected_versions[version] = selected_versions.get(version, 0) + 1
    total_jobs = len(ratios) * len(case_indices)
    completed_jobs = 0
    run_start = time.perf_counter()
    if args.visualize_only:
        print(
            f"[setup] mode=visualize_only candidate_cases={len(case_indices)} "
            f"by_version={selected_versions}",
            flush=True,
        )
    else:
        print(
            f"[setup] cases={len(case_indices)} by_version={selected_versions} "
            f"ratios={ratios} total_case_ratio_jobs={total_jobs}", flush=True)
    print(
        f"[setup] max_points={args.max_points} min_neighbors={args.min_neighbors} "
        f"ridge={args.ridge:g} distance_power={args.distance_power:g} "
        f"condition_tol={args.condition_tol:g}", flush=True)
    print(f"[setup] output_root={output_root}", flush=True)

    if args.visualize_only:
        run_visualization_only(args, dataset, case_indices, output_root)
        return

    summary = {
        "dataset": {
            "save_dir": args.save_dir,
            "interior_dir": args.interior_dir,
            "versions": args.versions,
            "frames": args.frames,
            "ratios": ratios,
            "max_points": args.max_points,
            "min_neighbors": args.min_neighbors,
            "ridge": args.ridge,
            "distance_power": args.distance_power,
            "condition_tol": args.condition_tol,
            "seed": args.seed,
            "max_cases": args.max_cases,
            "num_samples": len(case_indices),
        },
        "results": [],
    }

    results = []

    for ratio_index, ratio in enumerate(ratios):
        ratio_dir = make_ratio_dir(output_root, ratio)
        per_sample = []
        sample_sizes = []
        degrees = []
        valid_fractions = []

        selected_payload = None

        print(
            f"[ratio] {ratio_index + 1}/{len(ratios)} starting ratio={ratio:g} "
            f"cases={len(case_indices)}",
            flush=True,
        )
        for case_number, dataset_index in enumerate(case_indices, start=1):
            version, case_id, frame = dataset._index[dataset_index]
            case_start = time.perf_counter()
            should_log = (
                args.log_every > 0
                and (case_number == 1 or case_number % args.log_every == 0
                     or case_number == len(case_indices))
            )
            if should_log:
                print(
                    f"[progress] job={completed_jobs + 1}/{total_jobs} "
                    f"ratio={ratio:g} case={case_number}/{len(case_indices)} "
                    f"{version}/{case_id} frame={frame} loading",
                    flush=True,
                )
            sample_seed = args.seed + dataset_index
            wall, interior = dataset[dataset_index]
            wall, sampled = induced_sample(wall, interior, ratio, args.max_points, sample_seed)
            est_v, est_p, valid = estimate_local_gradients(
                sampled, ridge=args.ridge, min_neighbors=args.min_neighbors,
                distance_power=args.distance_power, condition_tol=args.condition_tol)

            vel_mse, vel_mae, vel_rel_mse, valid_count_v = summarize_components(
                est_v, sampled.grad_v.float().cpu(), valid)
            press_mse, press_mae, press_rel_mse, valid_count_p = summarize_components(
                est_p, sampled.grad_p.float().cpu(), valid)
            valid_count = min(valid_count_v, valid_count_p)
            mean_degree, edge_count = summarize_graph(sampled.edge_index.cpu(), sampled.num_nodes)

            sample_sizes.append(int(sampled.num_nodes))
            degrees.append(mean_degree)
            valid_fractions.append(float(valid.float().mean().item()))
            per_sample.append(
                SampleResult(
                    ratio=ratio,
                    sample_count=int(sampled.num_nodes),
                    edge_count=edge_count,
                    mean_degree=mean_degree,
                    valid_count=valid_count,
                    valid_fraction=float(valid.float().mean().item()),
                    vel_mse=vel_mse,
                    vel_mae=vel_mae,
                    vel_rel_mse=vel_rel_mse,
                    press_mse=press_mse,
                    press_mae=press_mae,
                    press_rel_mse=press_rel_mse,
                )
            )
            completed_jobs += 1
            if should_log:
                elapsed = time.perf_counter() - run_start
                rate = elapsed / completed_jobs
                eta = rate * (total_jobs - completed_jobs)
                print(
                    f"[progress] job={completed_jobs}/{total_jobs} done "
                    f"points={sampled.num_nodes} edges={edge_count} "
                    f"degree={mean_degree:.2f} valid={valid.float().mean().item():.3f} "
                    f"case_time={time.perf_counter() - case_start:.1f}s "
                    f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                    flush=True,
                )

            if dataset_index == args.plot_case_index:
                selected_payload = {
                    "ratio": ratio,
                    "dataset_index": dataset_index,
                    "sampled": sampled,
                    "est_v": est_v,
                    "est_p": est_p,
                    "valid": valid,
                }

        if not per_sample:
            continue

        vel_mse = np.mean([item.vel_mse for item in per_sample], axis=0)
        vel_mae = np.mean([item.vel_mae for item in per_sample], axis=0)
        vel_rel_mse = np.mean([item.vel_rel_mse for item in per_sample], axis=0)
        press_mse = np.mean([item.press_mse for item in per_sample], axis=0)
        press_mae = np.mean([item.press_mae for item in per_sample], axis=0)
        press_rel_mse = np.mean([item.press_rel_mse for item in per_sample], axis=0)

        result = SampleResult(
            ratio=ratio,
            sample_count=int(np.mean(sample_sizes)),
            edge_count=int(np.mean([item.edge_count for item in per_sample])),
            mean_degree=float(np.mean(degrees)),
            valid_count=int(np.mean([item.valid_count for item in per_sample])),
            valid_fraction=float(np.mean(valid_fractions)),
            vel_mse=vel_mse,
            vel_mae=vel_mae,
            vel_rel_mse=vel_rel_mse,
            press_mse=press_mse,
            press_mae=press_mae,
            press_rel_mse=press_rel_mse,
        )
        results.append(result)
        summary["results"].append(to_jsonable(result))

        with open(os.path.join(ratio_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(to_jsonable(result), f, indent=2)

        with open(os.path.join(ratio_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["component", "mse", "mae", "rel_mse"])
            for name, mse, mae, rel in zip(VEL_GRAD_NAMES, result.vel_mse, result.vel_mae, result.vel_rel_mse):
                writer.writerow([name, mse, mae, rel])
            for name, mse, mae, rel in zip(PRESS_GRAD_NAMES, result.press_mse, result.press_mae, result.press_rel_mse):
                writer.writerow([name, mse, mae, rel])

        print(
            f"[ratio={ratio:g}] samples={result.sample_count} edges={result.edge_count} "
            f"deg={result.mean_degree:.2f} valid={result.valid_fraction:.3f} "
            f"vel_rel_mse={result.vel_rel_mse.mean():.4g} "
            f"press_rel_mse={result.press_rel_mse.mean():.4g}",
            flush=True,
        )

        if selected_payload is not None:
            rep_dir = os.path.join(ratio_dir, "representative_case")
            os.makedirs(rep_dir, exist_ok=True)
            sampled = selected_payload["sampled"]
            est_v = selected_payload["est_v"].numpy()
            est_p = selected_payload["est_p"].numpy()
            valid = selected_payload["valid"].numpy()

            render_component_grid(
                sampled.pos.float().cpu().numpy(),
                sampled.grad_v.float().cpu().numpy(),
                est_v,
                VEL_GRAD_NAMES,
                valid,
                os.path.join(rep_dir, "vel_grad_truth.png"),
                f"Velocity gradients, truth, ratio={ratio:g}, sample={sampled.num_nodes}",
                mode="truth",
            )
            render_component_grid(
                sampled.pos.float().cpu().numpy(),
                sampled.grad_v.float().cpu().numpy(),
                est_v,
                VEL_GRAD_NAMES,
                valid,
                os.path.join(rep_dir, "vel_grad_estimate.png"),
                f"Velocity gradients, LS estimate, ratio={ratio:g}, sample={sampled.num_nodes}",
                mode="estimate",
            )
            render_component_grid(
                sampled.pos.float().cpu().numpy(),
                sampled.grad_v.float().cpu().numpy(),
                est_v,
                VEL_GRAD_NAMES,
                valid,
                os.path.join(rep_dir, "vel_grad_abs_err.png"),
                f"Velocity gradients, absolute error, ratio={ratio:g}",
                mode="error",
            )
            render_component_grid(
                sampled.pos.float().cpu().numpy(),
                sampled.grad_p.float().cpu().numpy(),
                est_p,
                PRESS_GRAD_NAMES,
                valid,
                os.path.join(rep_dir, "press_grad_truth.png"),
                f"Pressure gradients, truth, ratio={ratio:g}, sample={sampled.num_nodes}",
                mode="truth",
            )
            render_component_grid(
                sampled.pos.float().cpu().numpy(),
                sampled.grad_p.float().cpu().numpy(),
                est_p,
                PRESS_GRAD_NAMES,
                valid,
                os.path.join(rep_dir, "press_grad_estimate.png"),
                f"Pressure gradients, LS estimate, ratio={ratio:g}, sample={sampled.num_nodes}",
                mode="estimate",
            )
            render_component_grid(
                sampled.pos.float().cpu().numpy(),
                sampled.grad_p.float().cpu().numpy(),
                est_p,
                PRESS_GRAD_NAMES,
                valid,
                os.path.join(rep_dir, "press_grad_abs_err.png"),
                f"Pressure gradients, absolute error, ratio={ratio:g}",
                mode="error",
            )

            with open(os.path.join(rep_dir, "representative_case.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ratio": ratio,
                        "dataset_index": selected_payload["dataset_index"],
                        "sample_count": sampled.num_nodes,
                        "valid_fraction": float(valid.float().mean().item()),
                        "case_id": getattr(sampled, "case_id", None),
                        "version": getattr(sampled, "version", None),
                        "t": int(sampled.t.item()) if hasattr(sampled, "t") else None,
                    },
                    f,
                    indent=2,
                )

    if not results:
        raise RuntimeError("no ratios were evaluated")

    render_ratio_summary(output_root, ratios, results)

    with open(os.path.join(output_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_root, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ratio",
            "sample_count",
            "edge_count",
            "mean_degree",
            "valid_count",
            "valid_fraction",
            "vel_mean_mse",
            "press_mean_mse",
        ])
        for result in results:
            writer.writerow([
                result.ratio,
                result.sample_count,
                result.edge_count,
                result.mean_degree,
                result.valid_count,
                result.valid_fraction,
                float(result.vel_mse.mean()),
                float(result.press_mse.mean()),
            ])

    print(f"[done] wrote verification outputs to {output_root}")


if __name__ == "__main__":
    main()
