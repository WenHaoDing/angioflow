"""Visualize a trained interior field model's velocity/pressure predictions.

Fully driven by the checkpoint: every model and dataset hyperparameter (backbone
width, versions, frames, train/test split seed, ...) is read back from the "args"
dict save_ckpt() already stores, so the only required input is which run to load.
This also guarantees eval uses the *exact* test split the run was trained against.

Each selected case runs at FULL point-cloud resolution, not the training-time
sample_ratio/max_points subsample: the decoder treats every interior point
independently (cross-attention to geometry tokens + local surface-KNN gather, no
interior-interior edges — see models/interior/decoder.py), so inference has no
dependency on how points were subsampled during training. Pose is fixed (no random
rotation) so plots are reproducible.

Usage:
  conda activate new
  python -m angioflow.training.eval_interior \
    --run_name interior_hgps_v1v2_latent16_d64_b16_kl32 --which best --n_cases 20 \
    --device cuda:2

Outputs go to <ckpt_dir>/eval/<version>_<case_id>_frame<t>/: velocity-magnitude,
pressure-anomaly, and divergence comparisons (ground truth / prediction / absolute
error), each as a central-slab 2D view and a full 3D view, plus per-case
metrics.json and an aggregate summary.json/csv across all evaluated cases.

Divergence uses the same graph gradient stencil as the continuity loss
(models/interior/constraints.continuity_residual) on the full pool edge_index, both
on the predicted velocity and on the CFD ground-truth velocity — the latter is a
useful reference, since even "true" CFD velocity isn't exactly divergence-free
under a discrete stencil, so it shows the noise floor to judge the prediction
against rather than expecting it to hit exactly zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch_geometric.data import Batch

from angioflow.data.datasets import load_excluded_keys
from angioflow.data.gradient_verification.verify_interior_gradients import (
    _safe_filename, render_field_comparison)
from angioflow.data.normalize_interior import load_or_compute_interior_stats
from angioflow.models.interior.constraints import continuity_residual
from angioflow.training.train import _chain, load_waveform, split_by_case
from angioflow.training.train_interior import build_dataset, build_model
from angioflow.transform import CenterAndRandomRotate, InteriorNormalize


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", required=True)
    p.add_argument("--ckpt_root", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_checkpoints")
    p.add_argument("--which", choices=["best", "last"], default="best")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_cases", type=int, default=25, help="test cases visualized per version")
    p.add_argument("--max_plot_points", type=int, default=12000)
    p.add_argument("--seed", type=int, default=0, help="case-selection reproducibility")
    p.add_argument("--out_dir", default=None, help="default: <ckpt_dir>/eval")
    p.add_argument("--vtk", action="store_true", default=True,
                   help="also write a .vtp point cloud per case (open in ParaView)")
    p.add_argument("--no_vtk", dest="vtk", action="store_false")
    return p.parse_args()


def select_eval_cases(dataset, test_idx, n_cases, seed):
    """First `n_cases` test-set samples per version present, seeded-shuffled."""
    by_version = {}
    for i in test_idx:
        by_version.setdefault(dataset._index[i][0], []).append(i)
    rng = random.Random(seed)
    selected = []
    for version in sorted(by_version):
        idxs = by_version[version]
        rng.shuffle(idxs)
        selected.extend(idxs[:n_cases])
    return selected


def write_vtk_case(pos, vel_gt, vel_pr, press_gt, press_pr, div_gt, div_pr, valid_div, case_dir):
    """Point cloud + GT/prediction/error fields as .vtp, readable directly in ParaView."""
    import pyvista as pv

    cloud = pv.PolyData(pos)
    cloud.point_data["vel_gt"] = vel_gt
    cloud.point_data["vel_pred"] = vel_pr
    cloud.point_data["vel_error"] = vel_pr - vel_gt
    cloud.point_data["vel_gt_mag"] = np.linalg.norm(vel_gt, axis=1)
    cloud.point_data["vel_pred_mag"] = np.linalg.norm(vel_pr, axis=1)
    cloud.point_data["press_gt"] = press_gt
    cloud.point_data["press_pred"] = press_pr
    cloud.point_data["press_error"] = press_pr - press_gt
    cloud.point_data["divergence_gt"] = div_gt
    cloud.point_data["divergence_pred"] = div_pr
    cloud.point_data["divergence_valid"] = valid_div.astype(np.uint8)
    cloud.save(os.path.join(case_dir, "fields.vtp"))


@torch.no_grad()
def evaluate_case(model, dataset, index, norm, waveform, device, max_plot_points, out_root,
                  div_kwargs, write_vtk=True):
    version, case_id, frame = dataset._index[index]
    wall, interior = dataset[index]
    wall = Batch.from_data_list([wall]).to(device)
    interior = Batch.from_data_list([interior]).to(device)

    vel_pred, press_pred = model(wall, interior, waveform)

    pos_t = interior.pos * norm.pos_std.to(device)
    vel_gt_t = norm.inverse_vel(interior.vel)
    vel_pr_t = norm.inverse_vel(vel_pred)
    press_gt_t = norm.inverse_press(interior.press)
    press_pr_t = norm.inverse_press(press_pred)

    # same graph gradient stencil the continuity loss uses, on the full pool graph
    div_gt_t, valid_gt = continuity_residual(vel_gt_t, pos_t, interior.edge_index, **div_kwargs)
    div_pr_t, valid_pr = continuity_residual(vel_pr_t, pos_t, interior.edge_index, **div_kwargs)
    valid_div = (valid_gt & valid_pr).cpu().numpy()

    pos = pos_t.cpu().numpy()
    vel_gt = vel_gt_t.cpu().numpy()
    vel_pr = vel_pr_t.cpu().numpy()
    press_gt = press_gt_t.cpu().numpy().reshape(-1)
    press_pr = press_pr_t.cpu().numpy().reshape(-1)
    div_gt = div_gt_t.cpu().numpy()
    div_pr = div_pr_t.cpu().numpy()

    vel_gt_mag = np.linalg.norm(vel_gt, axis=1)
    vel_pr_mag = np.linalg.norm(vel_pr, axis=1)
    press_ref = float(press_gt.mean())   # absolute pressure has a large offset;
                                         # plot the (physically meaningful) anomaly
    valid = np.ones(pos.shape[0], dtype=bool)
    case_dir = os.path.join(out_root, f"{version}_{_safe_filename(str(case_id))}_frame{frame}")

    render_field_comparison(pos, vel_gt_mag, vel_pr_mag, valid, "velocity_magnitude",
                            case_dir, signed=False, max_plot_points=max_plot_points, seed=0)
    render_field_comparison(pos, press_gt - press_ref, press_pr - press_ref, valid,
                            "pressure_anomaly", case_dir, signed=True,
                            max_plot_points=max_plot_points, seed=1)
    render_field_comparison(pos, div_gt, div_pr, valid_div, "divergence",
                            case_dir, signed=True, max_plot_points=max_plot_points, seed=2)
    if write_vtk:
        write_vtk_case(pos, vel_gt, vel_pr, press_gt, press_pr, div_gt, div_pr, valid_div, case_dir)

    # relative L2 = ||pred - gt||_2 / ||gt||_2, pooled over points and components —
    # dimensionless, so vel/press (and AneuG-Flow/AngioFlow cases at different flow scales) are
    # comparable in a way the physical-unit RMSE above isn't. Pressure uses the
    # same anomaly as the plot: raw absolute pressure's ~1e5 Pa offset would
    # swamp ||gt||_2 and make the ratio meaningless otherwise.
    press_anom_gt, press_anom_pr = press_gt - press_ref, press_pr - press_ref
    vel_rel_l2 = float(np.linalg.norm(vel_pr - vel_gt) / np.linalg.norm(vel_gt))
    press_rel_l2 = float(np.linalg.norm(press_anom_pr - press_anom_gt) / np.linalg.norm(press_anom_gt))

    metrics = {
        "version": version, "case_id": case_id, "frame": frame, "n_points": int(pos.shape[0]),
        "vel_rmse_mps": float(np.sqrt(np.mean((vel_pr - vel_gt) ** 2))),
        "vel_mag_mae_mps": float(np.mean(np.abs(vel_pr_mag - vel_gt_mag))),
        "vel_rel_l2": vel_rel_l2,
        "press_rmse_pa": float(np.sqrt(np.mean((press_pr - press_gt) ** 2))),
        "press_mae_pa": float(np.mean(np.abs(press_pr - press_gt))),
        "press_rel_l2": press_rel_l2,
        "div_gt_mean_abs_per_s": float(np.mean(np.abs(div_gt[valid_div]))) if valid_div.any() else None,
        "div_pred_mean_abs_per_s": float(np.mean(np.abs(div_pr[valid_div]))) if valid_div.any() else None,
    }
    with open(os.path.join(case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    args = get_args()
    device = torch.device(args.device)
    ckpt_dir = os.path.join(args.ckpt_root, args.run_name)
    ckpt_path = os.path.join(ckpt_dir, f"{args.which}.pth")

    chk = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = argparse.Namespace(**chk["args"])
    # the decoder forward pass never needs edge_index (interior points are treated
    # independently), but the divergence visualization below does — force it on
    # regardless of what the run trained with. div_ridge/etc. may not exist on
    # checkpoints that predate --lambda_div, hence the getattr fallbacks.
    train_args.lambda_div = 1.0
    train_args.div_method = "graph"    # viz always wants the graph edge_index, regardless
                                        # of what --div_method the checkpoint trained with
    div_kwargs = dict(
        ridge=getattr(train_args, "div_ridge", 3e-2),
        power=getattr(train_args, "div_power", 2.0),
        min_neighbors=getattr(train_args, "div_min_neighbors", 3),
        cond_tol=getattr(train_args, "div_cond_tol", 1e-6),
    )

    model = build_model(train_args, device)
    model.load_state_dict(chk["model"])
    model.eval()
    print(f"[ckpt] {ckpt_path}  epoch={chk['epoch']}  best_loss={chk['best_loss']:.4g}")

    exclude_keys = load_excluded_keys(train_args.save_dir) if train_args.exclude_anomalies else set()
    dataset = build_dataset(train_args, exclude_keys)
    train_idx, test_idx = split_by_case(dataset, train_args.train_frac, seed=train_args.seed)
    print(f"[data] {len(dataset)} samples  train={len(train_idx)}  test={len(test_idx)}")

    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}
    stats = load_or_compute_interior_stats(
        os.path.join(train_args.save_dir, "interior_stats.pt"), dataset, train_keys,
        n_cases=train_args.stats_n_cases, frames_per_case=train_args.stats_frames_per_case,
        seed=train_args.seed)
    norm = InteriorNormalize(stats)
    # fixed pose (no random rotation), no subsampling -> full pool, reproducible
    dataset.transform = _chain(CenterAndRandomRotate(rotate=False), norm)

    waveform = load_waveform(train_args.waveform_path, length=1000, device=device)

    out_root = args.out_dir or os.path.join(ckpt_dir, "eval")
    os.makedirs(out_root, exist_ok=True)

    cases = select_eval_cases(dataset, test_idx, args.n_cases, args.seed)
    print(f"[eval] {len(cases)} cases -> {out_root}")

    all_metrics = []
    for i in cases:
        m = evaluate_case(model, dataset, i, norm, waveform, device, args.max_plot_points, out_root,
                          div_kwargs, write_vtk=args.vtk)
        all_metrics.append(m)
        print(f"  {m['version']}/{m['case_id']}  n={m['n_points']}  "
              f"vel_rmse={m['vel_rmse_mps']:.4g} m/s (rel_l2 {m['vel_rel_l2']:.3g})  "
              f"press_rmse={m['press_rmse_pa']:.4g} Pa (rel_l2 {m['press_rel_l2']:.3g})  "
              f"div_gt={m['div_gt_mean_abs_per_s']:.4g}  div_pred={m['div_pred_mean_abs_per_s']:.4g} 1/s")

    summary = {"checkpoint": ckpt_path, "epoch": chk["epoch"], "cases": all_metrics}
    for version in sorted({m["version"] for m in all_metrics}):
        vm = [m for m in all_metrics if m["version"] == version]
        summary[f"{version}_mean_vel_rmse_mps"] = float(np.mean([m["vel_rmse_mps"] for m in vm]))
        summary[f"{version}_mean_vel_rel_l2"] = float(np.mean([m["vel_rel_l2"] for m in vm]))
        summary[f"{version}_mean_press_rmse_pa"] = float(np.mean([m["press_rmse_pa"] for m in vm]))
        summary[f"{version}_mean_press_rel_l2"] = float(np.mean([m["press_rel_l2"] for m in vm]))
        summary[f"{version}_mean_div_gt_per_s"] = float(np.mean([m["div_gt_mean_abs_per_s"] for m in vm]))
        summary[f"{version}_mean_div_pred_per_s"] = float(np.mean([m["div_pred_mean_abs_per_s"] for m in vm]))
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_root, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"[done] wrote {out_root}")


if __name__ == "__main__":
    main()
