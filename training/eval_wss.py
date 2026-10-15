"""Visualize a trained WSS (wall shear stress) model's predictions.

Mirrors eval_interior.py: fully driven by the checkpoint's saved "args" dict, so
the only required input is which run to load, and eval always uses the exact
test split the run was trained against. Works for any model in MODEL_REGISTRY
(pointnet2, hierarchical_gps, wss_query_field) since build/load is dispatched
through the same get_model() the generic train.py uses.

Each case is cropped with the SAME AneurysmCrop(mask_low, mask_high, crop_aneurysm_only)
the run trained with (runs trained with --no_crop are evaluated on whole meshes instead;
--no_crop --crop_loss_mask runs get the whole mesh as input and are scored on the crop region,
with wss_rel_l2_full reporting the whole mesh too) (unlike eval_interior.py, which can skip its subsampling transform
since that decoder is resolution-independent by construction -- this graph-conv model
only ever trains on cropped subgraphs, so evaluating on the full mesh would be an
out-of-distribution input, not a harder/fairer test). Pose is fixed (no random rotation)
for reproducibility; the crop's random ring depth is still randomized but seeded via
--seed so it's reproducible across runs.

Usage:
  conda activate new
  python -m angioflow.training.eval_wss \
    --run_name hier_gps_p1_fullg_0frame_v1v2_e128_npre2_gps8_npost4_knn8_inter2_pr015_nobi --which best --n_cases 5 \
    --device cuda:2 --full_test_set --no_viz

  # runs trained on an extracted AngioFlow-PyG root (--extracted): metrics on the whole
  # test set, renders for 5 cases per dataset
  python -m angioflow.training.eval_wss --ckpt_root <AngioFlow-PyG_checkpoints> \
    --run_name <run> --which best --full_test_set --n_render 5 --device cuda:0

Outputs go to <ckpt_dir>/eval/<version>_<case_id>_frame<t>/: WSS-magnitude and
wall-normal-component (physics check: WSS should be tangential, so wss . n ~ 0)
comparisons rendered as GT / Pred / |Error| surface-mesh panels -- same style as
save_wss_sanity() during training -- plus per-case metrics.json and an aggregate
summary.json/csv across all evaluated cases.

--no_viz skips ALL rendering (pyvista/vtk) and per-case output directories -- just
the forward pass + metrics, much faster for a full-test-set run. --full_test_set
evaluates every test-set case instead of --n_cases per version; with it, --n_render N
renders only N seeded-random cases per version while metrics still cover every case.
summary.json/csv (per-case metrics + per-version mean/median/p90/std of relative L2,
over the whole evaluated region and over the aneurysm nodes only) are always written.

Regardless of --no_viz, a single-row run_summary.csv/.json is always written to
out_root: run_name + every training hyperparameter (arg_* columns, straight from
the checkpoint's saved args) + MSE/rel_L2 for WSS, computed separately per version
AND pooled combined -- meant for appending one row per run into a spreadsheet
across many runs (mirrors eval_joint.py's run_summary).

That row is also appended to a master leaderboard CSV at <ckpt_root>/
wss_eval_leaderboard.csv (one file across ALL wss/hierarchical_gps/pointnet2 runs;
eval_joint.py keeps its own separate joint_eval_leaderboard.csv, since the two
tasks have different metrics/args). A run_name already present there is never
re-added, and if that run's run_summary.json already exists on disk (from a prior
invocation of this script), its numbers are reused as-is and the evaluation is
skipped entirely -- see eval_leaderboard.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
from torch_geometric.data import Batch

from angioflow.data.datasets import AngioFlowWallDataset, load_excluded_keys
from angioflow.data.datasets_t0 import AngioFlowWallDatasetT0
from angioflow.data.gradient_verification.verify_interior_gradients import _safe_filename
from angioflow.data.normalize import load_or_compute_stats
from angioflow.training.eval_interior import select_eval_cases
from angioflow.training.eval_leaderboard import load_existing_run_summary, upsert_leaderboard
from angioflow.training.train import _chain, _unpack_pred, get_model, load_waveform, split_by_case
from angioflow.transform import AneurysmCrop, CenterAndRandomRotate, PosYNormalize, SymmetrizeEdges
from angioflow.visualize.sanity import _make_surface_mesh


@contextmanager
def _pyvista_stdio_guard():
    """VTK's off-screen rendering corrupts stdout/stderr at the OS fd level (see
    save_wss_sanity); save both fds before any pyvista call and restore them
    unconditionally after."""
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    py_stdout, py_stderr = sys.stdout, sys.stderr
    try:
        yield
    finally:
        os.dup2(stdout_fd, 1); os.close(stdout_fd)
        os.dup2(stderr_fd, 2); os.close(stderr_fd)
        sys.stdout, sys.stderr = py_stdout, py_stderr


VIZ_MAX_WSS_PA = 15.0  # color-scale clamp for visualization only; metrics stay uncapped


def render_wss_panel(mesh, gt_vals, pred_vals, unit_label, out_dir, field_name, signed=False,
                     camera_positions=("iso", "xy", "yz"), max_abs=VIZ_MAX_WSS_PA, zoom=1.6):
    """GT / Pred / |Error| surface-mesh comparison, one file per camera angle --
    same pyvista panel style as save_wss_sanity(), clamped to +/- max_abs so a
    handful of hot-spot outliers don't wash out the whole colormap."""
    import pyvista as pv

    gt_c = np.clip(gt_vals, -max_abs, max_abs) if signed else np.clip(gt_vals, 0.0, max_abs)
    pred_c = np.clip(pred_vals, -max_abs, max_abs) if signed else np.clip(pred_vals, 0.0, max_abs)
    err = np.abs(pred_c - gt_c)
    vmin, vmax = (-max_abs, max_abs) if signed else (0.0, max_abs)
    cols = [
        (gt_c,   f"GT {unit_label}",      "coolwarm", vmin, vmax),
        (pred_c, f"Pred {unit_label}",    "coolwarm", vmin, vmax),
        (err,    f"|Error| {unit_label}", "Oranges",  0.0, max(float(err.max()), 1e-12)),
    ]
    for cam in camera_positions:
        pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2400, 900))
        for col, (vals, title, cmap, lo, hi) in enumerate(cols):
            pl.subplot(0, col)
            m = mesh.copy()
            m.point_data["v"] = vals
            pl.add_mesh(m, scalars="v", cmap=cmap, clim=[lo, hi], show_scalar_bar=True,
                        scalar_bar_args={"title": title, "vertical": True})
            pl.add_title(title, font_size=11)
            pl.camera_position = cam
            pl.camera.zoom(zoom)
        pl.screenshot(os.path.join(out_dir, f"{field_name}_{cam}.png"), scale=2)
        pl.close()


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
    p.add_argument("--seed", type=int, default=0, help="case-selection reproducibility")
    p.add_argument("--out_dir", default=None, help="default: <ckpt_dir>/eval")
    p.add_argument("--vtk", action="store_true", default=True,
                   help="also write a .vtp surface mesh per case (open in ParaView)")
    p.add_argument("--no_vtk", dest="vtk", action="store_false")
    p.add_argument("--viz", action="store_true", default=True,
                   help="render per-case images/vtk (default on)")
    p.add_argument("--no_viz", dest="viz", action="store_false",
                   help="skip ALL rendering/vtk/per-case dirs -- just metrics, "
                        "much faster for --full_test_set")
    p.add_argument("--viz_crop_region", action="store_true",
                   help="for runs that take the whole mesh (--no_crop, with or without "
                        "--crop_loss_mask): render only the crop region (the run's mask_low/"
                        "mask_high, same seeded ring depth as the metrics). Metrics are unchanged; "
                        "fields.vtp stays whole-mesh with an in_crop_region point field")
    p.add_argument("--n_render", type=int, default=None,
                   help="with --full_test_set: render only this many seeded-random cases per "
                        "version (metrics still cover every case); default: render all")
    return p.parse_args()


def build_dataset(args, exclude_keys):
    versions = args.versions.split(",") if args.versions else None
    ds_kwargs = dict(
        case_root_angioflow=args.case_root_angioflow, case_root_aneug_flow=args.case_root_aneug_flow, save_dir=args.save_dir,
        n_frames=args.n_frames, decimate_ratio=args.decimate_ratio, decimate_method=args.decimate_method,
        lap_k=args.lap_k, exclude_keys=exclude_keys,
        extracted=getattr(args, "extracted", False),   # runs trained on an AngioFlow-PyG root
        add_flow_split=getattr(args, "add_flow_split", False),
        sanity_n_normal=0)                            # no data-sanity rendering during eval
    if args.frames == "all":
        return AngioFlowWallDataset(**ds_kwargs)
    frames = [int(t) for t in args.frames.split(",")]
    return AngioFlowWallDatasetT0(**ds_kwargs, versions=versions, frames=frames)


def write_vtk_case(mesh, wss_gt, wss_pr, case_dir, in_region=None):
    mesh = mesh.copy()
    if in_region is not None:
        mesh.point_data["in_crop_region"] = in_region.astype(np.uint8)
    mesh.point_data["wss_gt"] = wss_gt
    mesh.point_data["wss_pred"] = wss_pr
    mesh.point_data["wss_error"] = wss_pr - wss_gt
    mesh.point_data["wss_gt_mag"] = np.linalg.norm(wss_gt, axis=1)
    mesh.point_data["wss_pred_mag"] = np.linalg.norm(wss_pr, axis=1)
    mesh.save(os.path.join(case_dir, "fields.vtp"))


@torch.no_grad()
def compute_case(model, dataset, index, norm, waveform, device, viz_crop_region=False):
    """Forward pass + all denormalized fields + metrics for one case. No I/O -- the
    caller decides whether to render/write anything. Returns (metrics_dict, fields)
    where `fields` carries the raw arrays render_case() needs."""
    version, case_id, frame = dataset._index[index]
    data = dataset[index]
    batch = Batch.from_data_list([data]).to(device)

    pred, query_idx = _unpack_pred(model(batch, waveform))
    assert query_idx is None, "eval must run at full node resolution (model.eval())"

    pos = (batch.pos * norm.pos_std.to(device)).cpu().numpy()
    normals = batch.normals.cpu().numpy()
    wss_gt = norm.inverse_y(batch.y).cpu().numpy()
    wss_pr = norm.inverse_y(pred).cpu().numpy()
    edge_index = batch.edge_index.cpu().numpy()

    wss_gt_mag = np.linalg.norm(wss_gt, axis=1)
    wss_pr_mag = np.linalg.norm(wss_pr, axis=1)
    # physics check: WSS is tangential traction (wss . n == 0); the GT value isn't
    # exactly zero either since normals come from the decimated mesh, not exact
    # geometry -- it's the noise floor to judge the prediction's residual against
    wss_gt_normal = (wss_gt * normals).sum(-1)
    wss_pr_normal = (wss_pr * normals).sum(-1)

    # metrics region: the crop region for --crop_loss_mask runs (batch.loss_mask), else every node
    lm = batch.loss_mask.cpu().numpy() if getattr(batch, "loss_mask", None) is not None \
        else np.ones(len(wss_gt), dtype=bool)
    wss_rel_l2 = float(np.linalg.norm(wss_pr[lm] - wss_gt[lm]) / np.linalg.norm(wss_gt[lm]))
    wss_rel_l2_full = float(np.linalg.norm(wss_pr - wss_gt) / np.linalg.norm(wss_gt))
    # same, over the aneurysm nodes only (node_type == 1) -- the clinically relevant region
    an = (batch.node_type == 1).cpu().numpy()
    wss_rel_l2_aneurysm = (float(np.linalg.norm(wss_pr[an] - wss_gt[an]) / np.linalg.norm(wss_gt[an]))
                           if an.any() else None)
    metrics = {
        "version": version, "case_id": case_id, "frame": frame, "n_points": int(pos.shape[0]),
        "n_metric_points": int(lm.sum()),
        "wss_mse_pa2_per_s2": float(np.mean((wss_pr[lm] - wss_gt[lm]) ** 2)),
        "wss_rmse_pa_per_s": float(np.sqrt(np.mean((wss_pr[lm] - wss_gt[lm]) ** 2))),
        "wss_mag_mae_pa_per_s": float(np.mean(np.abs(wss_pr_mag[lm] - wss_gt_mag[lm]))),
        "wss_rel_l2": wss_rel_l2,
        "wss_rel_l2_full": wss_rel_l2_full,
        "wss_rel_l2_aneurysm": wss_rel_l2_aneurysm,
        "n_aneurysm_points": int(an.sum()),
        "wss_gt_normal_mean_abs_pa_per_s": float(np.mean(np.abs(wss_gt_normal))),
        "wss_pred_normal_mean_abs_pa_per_s": float(np.mean(np.abs(wss_pr_normal))),
    }
    # region to render (--viz_crop_region): viz_mask for plain --no_crop runs, loss_mask
    # for --crop_loss_mask runs; None = render everything the model saw
    vm = getattr(batch, "viz_mask", None)
    if vm is None and viz_crop_region:
        vm = getattr(batch, "loss_mask", None)
    fields = dict(pos=pos, edge_index=edge_index, wss_gt=wss_gt, wss_pr=wss_pr,
                  wss_gt_mag=wss_gt_mag, wss_pr_mag=wss_pr_mag,
                  wss_gt_normal=wss_gt_normal, wss_pr_normal=wss_pr_normal,
                  viz_mask=None if vm is None else vm.cpu().numpy())
    return metrics, fields


def _region_fields(fields, m):
    """Per-node render fields restricted to nodes in m; edges with both ends inside, reindexed."""
    new_idx = -np.ones(len(m), dtype=np.int64)
    new_idx[m] = np.arange(int(m.sum()))
    ei = fields["edge_index"]
    out = {k: v[m] for k, v in fields.items()
           if isinstance(v, np.ndarray) and k not in ("edge_index", "viz_mask") and len(v) == len(m)}
    out["edge_index"] = new_idx[ei[:, m[ei[0]] & m[ei[1]]]]
    return out


def render_case(metrics, fields, out_root, write_vtk=True):
    version, case_id, frame = metrics["version"], metrics["case_id"], metrics["frame"]
    case_dir = os.path.join(out_root, f"{version}_{_safe_filename(str(case_id))}_frame{frame}")
    os.makedirs(case_dir, exist_ok=True)
    mesh = _make_surface_mesh(fields["pos"], fields["edge_index"])
    vm = fields.get("viz_mask")
    shown = fields if vm is None else _region_fields(fields, vm)
    shown_mesh = mesh if vm is None else _make_surface_mesh(shown["pos"], shown["edge_index"])

    with _pyvista_stdio_guard():
        render_wss_panel(shown_mesh, shown["wss_gt_mag"], shown["wss_pr_mag"], "WSS (Pa/s)",
                         case_dir, "wss_magnitude", signed=False)
        render_wss_panel(shown_mesh, shown["wss_gt_normal"], shown["wss_pr_normal"], "WSS.n (Pa/s)",
                         case_dir, "wss_normal_component", signed=True)
        if write_vtk:
            write_vtk_case(mesh, fields["wss_gt"], fields["wss_pr"], case_dir, in_region=vm)

    with open(os.path.join(case_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main():
    args = get_args()
    ckpt_dir = os.path.join(args.ckpt_root, args.run_name)
    out_root = args.out_dir or os.path.join(ckpt_dir, "eval")
    leaderboard_path = os.path.join(args.ckpt_root, "wss_eval_leaderboard.csv")

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
    stats = load_or_compute_stats(
        os.path.join(train_args.save_dir, "stats.pt"), dataset, train_keys,
        n_cases=train_args.stats_n_cases, frames_per_case=train_args.stats_frames_per_case,
        seed=train_args.seed)
    norm = PosYNormalize(stats)

    model = get_model(train_args, stats, device)
    model.load_state_dict(chk["model"])
    model.eval()
    print(f"[ckpt] {ckpt_path}  epoch={chk['epoch']}  best_loss={chk['best_loss']:.4g}")

    # fixed pose (no random rotation) for reproducibility, but AneurysmCrop must still
    # run with the SAME mask_low/mask_high/crop_aneurysm_only the run trained with --
    # this model never saw more than a mask_high-deep ring around the aneurysm during
    # training (e.g. 0.25 here), so evaluating on the full uncropped mesh is a train/eval
    # distribution mismatch, not a fairer/harder test. SymmetrizeEdges runs after
    # cropping, same order as train.py, to symmetrize the smaller cropped edge set.
    # evaluate the way the run trained: --no_crop -> whole mesh; --no_crop --crop_loss_mask ->
    # whole mesh in, metrics on the (seeded) crop region via loss_mask; default -> cropped subgraph
    no_crop = getattr(train_args, "no_crop", False)
    loss_mask = getattr(train_args, "crop_loss_mask", False)
    crop = ([AneurysmCrop(low=train_args.mask_low, high=train_args.mask_high,
                          aneurysm_only=train_args.crop_aneurysm_only, as_loss_mask=loss_mask)]
            if (not no_crop or loss_mask) else [])
    if args.viz_crop_region and no_crop and not loss_mask:
        # plain --no_crop run: mark the crop region for rendering only (metrics stay whole-mesh)
        crop = [AneurysmCrop(low=train_args.mask_low, high=train_args.mask_high,
                             aneurysm_only=train_args.crop_aneurysm_only,
                             as_loss_mask=True, mask_attr="viz_mask")]
    dataset.transform = _chain(CenterAndRandomRotate(rotate=False), norm, *crop, SymmetrizeEdges())
    print("[eval] crop: " + ("whole mesh in, metrics on the crop region (--crop_loss_mask)" if loss_mask else
                             "none (whole mesh, --no_crop)" if no_crop else "cropped subgraph, as in training")
          + ("; renders show the crop region only" if args.viz_crop_region and no_crop else ""))

    waveform = load_waveform(train_args.waveform_path, length=1000, device=device)

    os.makedirs(out_root, exist_ok=True)

    if args.viz:
        import pyvista as pv
        with _pyvista_stdio_guard():
            pv.start_xvfb()

    if args.full_test_set:
        cases = list(test_idx)
        render_set = (set(select_eval_cases(dataset, test_idx, args.n_render, args.seed))
                      if args.n_render is not None else set(cases))
    else:
        cases = select_eval_cases(dataset, test_idx, args.n_cases, args.seed)
        render_set = set(cases)
    if not args.viz:
        render_set = set()
    print(f"[eval] {len(cases)} cases, rendering {len(render_set)} -> {out_root}")
    # AneurysmCrop draws its random ring depth from the global torch RNG; fix it here
    # (after stats/case-selection, which use their own local RNGs) so the crop each
    # case gets is reproducible across runs of this script with the same --seed
    torch.manual_seed(args.seed)

    all_metrics = []
    for i in cases:
        m, fields = compute_case(model, dataset, i, norm, waveform, device,
                                 viz_crop_region=args.viz_crop_region)
        if i in render_set:
            render_case(m, fields, out_root, write_vtk=args.vtk)
        all_metrics.append(m)
        print(f"  {m['version']}/{m['case_id']}  n={m['n_points']}  "
              f"wss_rmse={m['wss_rmse_pa_per_s']:.4g} Pa/s (rel_l2 {m['wss_rel_l2']:.3g})  "
              f"normal_gt={m['wss_gt_normal_mean_abs_pa_per_s']:.3g}  "
              f"normal_pred={m['wss_pred_normal_mean_abs_pa_per_s']:.3g} Pa/s")

    # --- per-case detail + per-version distribution of relative L2 (always written) ---
    summary = {"checkpoint": ckpt_path, "epoch": chk["epoch"], "cases": all_metrics}
    table = []
    for version in sorted({m["version"] for m in all_metrics}) + ["all"]:
        vm = all_metrics if version == "all" else [m for m in all_metrics if m["version"] == version]
        summary[f"{version}_mean_wss_rmse_pa_per_s"] = float(np.mean([m["wss_rmse_pa_per_s"] for m in vm]))
        summary[f"{version}_mean_wss_gt_normal_pa_per_s"] = float(
            np.mean([m["wss_gt_normal_mean_abs_pa_per_s"] for m in vm]))
        summary[f"{version}_mean_wss_pred_normal_pa_per_s"] = float(
            np.mean([m["wss_pred_normal_mean_abs_pa_per_s"] for m in vm]))
        row = {"version": version, "n_cases": len(vm)}
        for key in ("wss_rel_l2", "wss_rel_l2_aneurysm"):
            v = np.array([m[key] for m in vm if m.get(key) is not None])
            stats_k = {"mean": float(v.mean()), "median": float(np.median(v)),
                       "p90": float(np.percentile(v, 90)), "std": float(v.std())}
            for name, val in stats_k.items():
                summary[f"{version}_{name}_{key}"] = val
                row[f"{key}_{name}"] = val
        table.append(row)
    summary["rel_l2_table"] = table
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_root, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nrelative L2 on test cases (checkpoint epoch {chk['epoch']}):")
    print(f"  {'dataset':12s} {'n':>4s} | {'whole region: mean  median   p90':>34s} | "
          f"{'aneurysm only: mean  median   p90':>35s}")
    for r in table:
        print(f"  {r['version']:12s} {r['n_cases']:4d} | "
              f"{r['wss_rel_l2_mean']:18.4f} {r['wss_rel_l2_median']:7.4f} {r['wss_rel_l2_p90']:6.4f} | "
              f"{r['wss_rel_l2_aneurysm_mean']:19.4f} {r['wss_rel_l2_aneurysm_median']:7.4f} "
              f"{r['wss_rel_l2_aneurysm_p90']:6.4f}")

    # --- run-level summary: one row, run_name + every hyperparameter + metrics per
    # version AND pooled combined -- always written, mirrors eval_joint.py's
    # run_summary. Metric = mean of per-case values (case-equal-weighted).
    metric_keys = ["wss_mse_pa2_per_s2", "wss_rel_l2", "wss_rel_l2_aneurysm"]

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
