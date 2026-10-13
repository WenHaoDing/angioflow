"""Visualize a trained joint WSS + interior velocity/pressure model's predictions.

Mirrors eval_wss.py (WSS panels) and eval_interior.py (velocity/pressure panels),
combined for AngioJointField. Fully driven by the checkpoint's saved "args" dict —
the only required input is which run to load — and eval always uses the exact test
split the run was trained against, via train_joint.build_dataset/build_model.

Like eval_wss.py, each case is cropped with the SAME JointAneurysmCrop
(mask_low/mask_high/crop_aneurysm_only/crop_factor/crop_slack) the run trained
with: this model never saw more than a mask_high-deep ring around the aneurysm, so
evaluating on the full mesh would be an out-of-distribution input, not a fairer
test. Unlike training, the WALL query subsampling (wall_query_ratio) is forced OFF
here — model.query_ratio is set to 1.0 so every case is visualized at full wall
resolution, matching how eval_wss.py always evaluates at full node resolution. The
interior stream is NOT re-subsampled either (full validated pool, minus whatever
JointAneurysmCrop's ROI filter removes) — same "full point cloud" philosophy as
eval_interior.py, since the interior decoder treats points independently. Pose is
fixed (no random rotation) for reproducibility; the crop's random ring depth is
still randomized but seeded via --seed.

Usage:
  conda activate new
  python -m angioflow.training.eval_joint \
    --run_name joint_p1r015_0frame_v1v2_e108f96_gps6_hop2_tok64_wq50_is10_blk3_xf_mh020_pressmax0 \
    --which best --device cuda:2 --no_viz --full_test_set

Outputs go to <ckpt_dir>/eval/<version>_<case_id>_frame<t>/, WSS and velocity/
pressure visualizations saved as SEPARATE image files (not one combined figure,
unlike the training-time sanity render):
  wss_magnitude_{iso,xy,yz}.png, wss_normal_component_{iso,xy,yz}.png  (pyvista,
    same style as save_wss_sanity/eval_wss.py's render_wss_panel)
  velocity_magnitude_{slab,3d}.png, pressure_anomaly_{slab,3d}.png  (matplotlib,
    same style as eval_interior.py's render_field_comparison)
plus per-case metrics.json (WSS + velocity + pressure) and an aggregate
summary.json/csv across all evaluated cases. No divergence panel: the joint
dataset loads no interior edge_index (no physics constraints, per the joint
model's design) and the WSS decoder's tangential constraint replaces WSS-side
physics checks the way eval_wss.py's wss.n panel does.

--no_viz skips ALL rendering (pyvista/matplotlib/vtk) and per-case output
directories -- just the forward pass + metrics, much faster for a full-test-set
run. --full_test_set evaluates every test-set case instead of --n_cases per
version.

Regardless of --no_viz, a single-row run_summary.csv/.json is always written to
out_root: run_name + every training hyperparameter (arg_* columns, straight from
the checkpoint's saved args) + MSE/rel_L2 for WSS and velocity, computed
separately per version AND pooled combined -- meant for appending one row per
run into a spreadsheet across many runs.

That row is also upserted into a master leaderboard CSV at <ckpt_root>/
joint_eval_leaderboard.csv (eval_wss.py keeps its own separate
wss_eval_leaderboard.csv, since the two tasks have different metrics/args). A
run_name already present has its row REPLACED, not duplicated, and if that run's
run_summary.json already exists on disk (from a prior invocation), its numbers
are reused as-is and the evaluation is skipped entirely -- see eval_leaderboard.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch_geometric.data import Batch

from angioflow.data.datasets import load_excluded_keys
from angioflow.data.gradient_verification.verify_interior_gradients import (
    _safe_filename, render_field_comparison)
from angioflow.data.normalize import load_or_compute_stats
from angioflow.data.normalize_interior import load_or_compute_interior_stats
from angioflow.training.eval_interior import select_eval_cases
from angioflow.training.eval_leaderboard import load_existing_run_summary, upsert_leaderboard
from angioflow.training.eval_wss import VIZ_MAX_WSS_PA, _pyvista_stdio_guard, render_wss_panel
from angioflow.training.eval_wss import write_vtk_case as write_vtk_wall
from angioflow.training.train import _chain, load_waveform, split_by_case
from angioflow.training.train_joint import _WallOnly, build_dataset, build_model, interior_stats_path
from angioflow.transform import (CenterAndRandomRotate, JointAneurysmCrop, JointNormalize,
                                 PressureZeroMax, SymmetrizeEdges)
from angioflow.visualize.sanity import _make_surface_mesh


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", required=True)
    p.add_argument("--ckpt_root", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_checkpoints")
    p.add_argument("--which", choices=["best", "last"], default="best")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_cases", type=int, default=25, help="test cases visualized per version "
                   "(ignored when --full_test_set)")
    p.add_argument("--full_test_set", action="store_true",
                   help="evaluate every test-set case (per version) instead of a capped sample "
                        "-- for the run_summary metrics, not for eyeballing individual renders")
    p.add_argument("--max_plot_points", type=int, default=12000, help="interior point-cloud plot cap")
    p.add_argument("--seed", type=int, default=0, help="case-selection / crop reproducibility")
    p.add_argument("--out_dir", default=None, help="default: <ckpt_dir>/eval")
    p.add_argument("--vtk", action="store_true", default=True,
                   help="also write wall_fields.vtp + interior_fields.vtp per case (ParaView)")
    p.add_argument("--no_vtk", dest="vtk", action="store_false")
    p.add_argument("--viz", action="store_true", default=True,
                   help="render per-case images/vtk (default on)")
    p.add_argument("--no_viz", dest="viz", action="store_false",
                   help="skip ALL rendering/vtk/per-case dirs -- just run_summary.csv/.json, "
                        "much faster for --full_test_set")
    return p.parse_args()


def write_vtk_interior(pos, vel_gt, vel_pr, press_gt, press_pr, case_dir):
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
    cloud.save(os.path.join(case_dir, "interior_fields.vtp"))


@torch.no_grad()
def compute_case(model, dataset, index, norm, waveform, device):
    """Forward pass + all denormalized fields + metrics for one case. No I/O -- the
    caller decides whether to render/write anything. Returns (metrics_dict, fields)
    where `fields` carries the raw arrays render_case() needs."""
    version, case_id, frame = dataset._index[index]
    wall, interior = dataset[index]
    wall = Batch.from_data_list([wall]).to(device)
    interior = Batch.from_data_list([interior]).to(device)

    wss_pred, query_idx, vel_pred, press_pred = model(wall, interior, waveform)
    assert query_idx is None, "eval must run at full wall resolution (model.query_ratio=1.0)"

    wall_pos = (wall.pos * norm.pos_std.to(device)).cpu().numpy()
    normals = wall.normals.cpu().numpy()
    wss_gt = norm.inverse_y(wall.y).cpu().numpy()
    wss_pr = norm.inverse_y(wss_pred).cpu().numpy()
    edge_index = wall.edge_index.cpu().numpy()
    wss_gt_mag = np.linalg.norm(wss_gt, axis=1)
    wss_pr_mag = np.linalg.norm(wss_pr, axis=1)
    # physics check: WSS is tangential traction (wss . n == 0) -- see eval_wss.py
    wss_gt_normal = (wss_gt * normals).sum(-1)
    wss_pr_normal = (wss_pr * normals).sum(-1)

    int_pos = (interior.pos * norm.pos_std.to(device)).cpu().numpy()
    vel_gt = norm.inverse_vel(interior.vel).cpu().numpy()
    vel_pr = norm.inverse_vel(vel_pred).cpu().numpy()
    press_gt = norm.inverse_press(interior.press).cpu().numpy().reshape(-1)
    press_pr = norm.inverse_press(press_pred).cpu().numpy().reshape(-1)
    vel_gt_mag = np.linalg.norm(vel_gt, axis=1)
    vel_pr_mag = np.linalg.norm(vel_pr, axis=1)
    press_ref = float(press_gt.mean())   # absolute pressure has a large offset;
                                         # plot/rel_l2 use the (physically meaningful) anomaly

    wss_rel_l2 = float(np.linalg.norm(wss_pr - wss_gt) / np.linalg.norm(wss_gt))
    vel_rel_l2 = float(np.linalg.norm(vel_pr - vel_gt) / np.linalg.norm(vel_gt))
    press_anom_gt, press_anom_pr = press_gt - press_ref, press_pr - press_ref
    press_rel_l2 = float(np.linalg.norm(press_anom_pr - press_anom_gt) / np.linalg.norm(press_anom_gt))

    metrics = {
        "version": version, "case_id": case_id, "frame": frame,
        "n_wall_points": int(wall_pos.shape[0]), "n_interior_points": int(int_pos.shape[0]),
        "wss_mse_pa2_per_s2": float(np.mean((wss_pr - wss_gt) ** 2)),
        "wss_rmse_pa_per_s": float(np.sqrt(np.mean((wss_pr - wss_gt) ** 2))),
        "wss_mag_mae_pa_per_s": float(np.mean(np.abs(wss_pr_mag - wss_gt_mag))),
        "wss_rel_l2": wss_rel_l2,
        "wss_gt_normal_mean_abs_pa_per_s": float(np.mean(np.abs(wss_gt_normal))),
        "wss_pred_normal_mean_abs_pa_per_s": float(np.mean(np.abs(wss_pr_normal))),
        "vel_mse_m2_per_s2": float(np.mean((vel_pr - vel_gt) ** 2)),
        "vel_rmse_mps": float(np.sqrt(np.mean((vel_pr - vel_gt) ** 2))),
        "vel_mag_mae_mps": float(np.mean(np.abs(vel_pr_mag - vel_gt_mag))),
        "vel_rel_l2": vel_rel_l2,
        "press_rmse_pa": float(np.sqrt(np.mean((press_pr - press_gt) ** 2))),
        "press_mae_pa": float(np.mean(np.abs(press_pr - press_gt))),
        "press_rel_l2": press_rel_l2,
    }

    fields = dict(wall_pos=wall_pos, edge_index=edge_index, wss_gt=wss_gt, wss_pr=wss_pr,
                  wss_gt_mag=wss_gt_mag, wss_pr_mag=wss_pr_mag,
                  wss_gt_normal=wss_gt_normal, wss_pr_normal=wss_pr_normal,
                  int_pos=int_pos, vel_gt=vel_gt, vel_pr=vel_pr,
                  press_gt=press_gt, press_pr=press_pr, press_ref=press_ref)
    return metrics, fields


def render_case(metrics, fields, max_plot_points, out_root, write_vtk=True):
    version, case_id, frame = metrics["version"], metrics["case_id"], metrics["frame"]
    case_dir = os.path.join(out_root, f"{version}_{_safe_filename(str(case_id))}_frame{frame}")
    os.makedirs(case_dir, exist_ok=True)

    mesh = _make_surface_mesh(fields["wall_pos"], fields["edge_index"])
    with _pyvista_stdio_guard():
        render_wss_panel(mesh, fields["wss_gt_mag"], fields["wss_pr_mag"], "WSS (Pa/s)",
                         case_dir, "wss_magnitude", signed=False, max_abs=VIZ_MAX_WSS_PA)
        render_wss_panel(mesh, fields["wss_gt_normal"], fields["wss_pr_normal"], "WSS.n (Pa/s)",
                         case_dir, "wss_normal_component", signed=True, max_abs=VIZ_MAX_WSS_PA)
        if write_vtk:
            write_vtk_wall(mesh, fields["wss_gt"], fields["wss_pr"], case_dir)

    int_pos = fields["int_pos"]
    valid = np.ones(int_pos.shape[0], dtype=bool)
    press_ref = fields["press_ref"]
    render_field_comparison(int_pos, np.linalg.norm(fields["vel_gt"], axis=1),
                            np.linalg.norm(fields["vel_pr"], axis=1), valid, "velocity_magnitude",
                            case_dir, signed=False, max_plot_points=max_plot_points, seed=0)
    render_field_comparison(int_pos, fields["press_gt"] - press_ref, fields["press_pr"] - press_ref,
                            valid, "pressure_anomaly", case_dir, signed=True,
                            max_plot_points=max_plot_points, seed=1)
    if write_vtk:
        write_vtk_interior(int_pos, fields["vel_gt"], fields["vel_pr"],
                           fields["press_gt"], fields["press_pr"], case_dir)
    with open(os.path.join(case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    args = get_args()
    ckpt_dir = os.path.join(args.ckpt_root, args.run_name)
    out_root = args.out_dir or os.path.join(ckpt_dir, "eval")
    leaderboard_path = os.path.join(args.ckpt_root, "joint_eval_leaderboard.csv")

    existing_row = load_existing_run_summary(out_root)
    if existing_row is not None:
        status = upsert_leaderboard(existing_row, leaderboard_path)
        print(f"[leaderboard] reused {out_root}/run_summary.json -> {status} in "
              f"{leaderboard_path}, no re-evaluation")
        return

    device = torch.device(args.device)
    ckpt_path = os.path.join(ckpt_dir, f"{args.which}.pth")

    chk = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = argparse.Namespace(**chk["args"])

    exclude_keys = load_excluded_keys(train_args.save_dir) if train_args.exclude_anomalies else set()
    dataset = build_dataset(train_args, exclude_keys)
    train_idx, test_idx = split_by_case(dataset, train_args.train_frac, seed=train_args.seed)
    print(f"[data] {len(dataset)} samples  train={len(train_idx)}  test={len(test_idx)}")

    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}
    wss_stats = load_or_compute_stats(
        os.path.join(train_args.save_dir, "stats.pt"), dataset, train_keys,
        n_cases=train_args.stats_n_cases, frames_per_case=train_args.stats_frames_per_case,
        seed=train_args.seed)
    press_zero_max = getattr(train_args, "press_zero_max", False)   # absent on older checkpoints
    interior_stats = load_or_compute_interior_stats(
        interior_stats_path(train_args), dataset, train_keys,
        n_cases=train_args.stats_n_cases, frames_per_case=train_args.stats_frames_per_case,
        seed=train_args.seed, press_zero_max=press_zero_max)
    norm = JointNormalize(wss_stats, interior_stats)

    model = build_model(train_args, device)
    model.load_state_dict(chk["model"])
    model.eval()
    model.query_ratio = 1.0   # always evaluate WSS at full wall resolution, regardless
                              # of what wall_query_ratio the run trained with
    print(f"[ckpt] {ckpt_path}  epoch={chk['epoch']}  best_loss={chk['best_loss']:.4g}")

    # fixed pose, SAME crop the run trained with (else full-mesh input is OOD -- see
    # eval_wss.py), full interior pool (no re-subsampling -- see eval_interior.py),
    # SAME pressure convention the run trained with (must run first -- see PressureZeroMax)
    eval_transforms = [PressureZeroMax()] if press_zero_max else []
    eval_transforms += [
        JointAneurysmCrop(low=train_args.mask_low, high=train_args.mask_high,
                          aneurysm_only=train_args.crop_aneurysm_only,
                          factor=train_args.crop_factor, slack=train_args.crop_slack),
        CenterAndRandomRotate(rotate=False),
        norm,
        _WallOnly(SymmetrizeEdges()),
    ]
    dataset.transform = _chain(*eval_transforms)

    waveform = load_waveform(train_args.waveform_path, length=1000, device=device)

    out_root = args.out_dir or os.path.join(ckpt_dir, "eval")
    os.makedirs(out_root, exist_ok=True)

    if args.viz:
        import pyvista as pv
        with _pyvista_stdio_guard():
            pv.start_xvfb()

    if args.full_test_set:
        cases = list(test_idx)
    else:
        cases = select_eval_cases(dataset, test_idx, args.n_cases, args.seed)
    print(f"[eval] {len(cases)} cases -> {out_root}  (viz={args.viz})")
    # JointAneurysmCrop's random ring depth draws from the global torch RNG; fix it
    # here (after stats/case-selection, which use their own local RNGs) so the crop
    # each case gets is reproducible across runs of this script with the same --seed
    torch.manual_seed(args.seed)

    all_metrics = []
    for i in cases:
        m, fields = compute_case(model, dataset, i, norm, waveform, device)
        if args.viz:
            render_case(m, fields, args.max_plot_points, out_root, write_vtk=args.vtk)
        all_metrics.append(m)
        print(f"  {m['version']}/{m['case_id']}  wall={m['n_wall_points']} int={m['n_interior_points']}  "
              f"wss_rmse={m['wss_rmse_pa_per_s']:.4g} Pa/s (rel_l2 {m['wss_rel_l2']:.3g})  "
              f"vel_rmse={m['vel_rmse_mps']:.4g} m/s (rel_l2 {m['vel_rel_l2']:.3g})  "
              f"press_rmse={m['press_rmse_pa']:.4g} Pa (rel_l2 {m['press_rel_l2']:.3g})")

    # --- per-case detail (skipped when --no_viz: no per-case dirs exist to pair with) ---
    if args.viz:
        summary = {"checkpoint": ckpt_path, "epoch": chk["epoch"], "cases": all_metrics}
        for version in sorted({m["version"] for m in all_metrics}):
            vm = [m for m in all_metrics if m["version"] == version]
            for key in ("wss_rmse_pa_per_s", "wss_rel_l2", "vel_rmse_mps", "vel_rel_l2",
                        "press_rmse_pa", "press_rel_l2"):
                summary[f"{version}_mean_{key}"] = float(np.mean([m[key] for m in vm]))
        with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(out_root, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
            writer.writeheader()
            writer.writerows(all_metrics)

    # --- run-level summary: one row, run_name + every hyperparameter + metrics per
    # version AND pooled combined -- always written, this is the "just give me a csv"
    # output. Metric = mean of per-case values (case-equal-weighted), matching the
    # existing summary.json convention above, not a point-pooled average.
    metric_keys = ["wss_mse_pa2_per_s2", "wss_rel_l2", "vel_mse_m2_per_s2", "vel_rel_l2"]

    def _agg(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    run_row = {"run_name": args.run_name, "checkpoint": ckpt_path, "epoch": chk["epoch"],
              "which": args.which, "n_cases_evaluated": len(all_metrics)}
    for k, v in vars(train_args).items():
        run_row[f"arg_{k}"] = v
    groups = {"all": all_metrics}
    for version in sorted({m["version"] for m in all_metrics}):
        groups[version] = [m for m in all_metrics if m["version"] == version]
    for scope, rows in groups.items():
        for key in metric_keys:
            run_row[f"{scope}_{key}"] = _agg(rows, key)

    with open(os.path.join(out_root, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_row, f, indent=2, default=str)
    with open(os.path.join(out_root, "run_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(run_row.keys()))
        writer.writeheader()
        writer.writerow(run_row)
    print(f"[done] wrote {out_root}  (run_summary.csv: run_name + args + "
          f"{', '.join(sorted(groups))}-scoped metrics)")

    status = upsert_leaderboard(run_row, leaderboard_path)
    print(f"[leaderboard] {status} {args.run_name} in {leaderboard_path}")


if __name__ == "__main__":
    main()
