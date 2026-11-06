"""Check deploy/joint_velocity.py against the training pipeline, on AngioFlow TEST cases.

Each case is fed to the deploy path the way a new patient would arrive: the raw wall mesh
in millimetres at an arbitrary world offset, its focused mask and a point near the inlet.
  1. features: deploy wall features vs the stored training statics (no remeshing)
  2. velocity: deploy prediction vs the dataset + eval-transform path, on the same CFD
     interior points and the same crop -> must agree; both scored against the CFD velocity
  3. query points: 20% of the pool (training density) and sample_lumen_points as context
  4. remeshing: the CFD wall remeshed isotropically (as a segmentation surface would be)
Sanity PNGs go to --sanity_dir (default deploy/sanity/).
--measure_density prints the training pool density (points / mm^3 of lumen) for
TRAINING_POOL_PER_MM3.

    python -m angioflow.deploy.validate_joint_velocity --bundle <bundle.pt> --device cuda:0 --n_cases 6
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Batch

from angioflow.deploy.joint_velocity import (FOCUSED, INLET, OUTLET_0, OUTLET_1, JointVelocityPredictor,
                                             _capped_polydata, prepare_surface, save_velocity_sanity)
from angioflow.transform import joint_crop

joint_crop._INTERIOR_NODE_KEYS += ("keep_row",)              # let the crop carry row ids

ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG"
OFFSET_MM = np.array([31.0, -12.5, 87.0])        # arbitrary "scanner" position


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def measure_density(root, n=40):
    out = []
    for f in sorted(glob.glob(f"{root}/angioflow/static_angioflow_*.pt"))[:n]:
        s = torch.load(f, weights_only=False)
        cid = os.path.basename(f)[len("static_angioflow_"):-3]
        it = torch.load(f"{root}/interior/interior_static_angioflow_{cid}.pt", weights_only=False)
        vol = _capped_polydata(s["pos"].double().numpy(), s["faces"].numpy(), s["node_type"].numpy()).volume * 1e9
        out.append(len(it["pos"]) / vol)
    print(f"[density] pool points per mm^3 over {len(out)} AngioFlow cases: median {np.median(out):.1f} "
          f"(p10 {np.percentile(out, 10):.1f}, p90 {np.percentile(out, 90):.1f})")


def test_cases(pred, n):
    from angioflow.training.train import split_by_case
    from angioflow.training.train_joint import build_dataset, excluded_case_keys, test_versions
    a = pred.args
    ds = build_dataset(a, excluded_case_keys(a))
    _, test_idx = split_by_case(ds, a.train_frac, seed=a.seed, test_versions=test_versions(a))
    test_idx = [i for i in test_idx if ds._index[i][0] == "angioflow"]
    return ds, test_idx[:: max(1, len(test_idx) // n)][:n]


def dataset_path(pred, ds, idx, crop_p):
    """The training/eval transform chain with the crop depth fixed to crop_p."""
    from angioflow.transform import (CenterAndRandomRotate, JointAneurysmCrop, JointNormalize,
                                     PressureZeroMax, SymmetrizeEdges)
    from angioflow.training.train_joint import _WallOnly
    a = pred.args
    wall, interior = ds.get(idx)
    gt_world = interior.pos.numpy().astype(np.float64).copy()
    vel_gt = interior.vel.numpy().copy()
    interior.keep_row = torch.arange(interior.num_nodes)       # tracked through the crop
    st = {"pos_std": pred.pos_std, "vel_mean": pred.vel_mean, "vel_std": pred.vel_std,
          "press_mean": torch.zeros(1), "press_std": torch.ones(1)}
    chain = ([PressureZeroMax()] if a.press_zero_max else []) + [
        JointAneurysmCrop(low=crop_p, high=crop_p, factor=a.crop_factor, slack=a.crop_slack),
        CenterAndRandomRotate(rotate=False),
        JointNormalize({"y_mean": torch.zeros(3), "y_std": torch.ones(3)}, st),
        _WallOnly(SymmetrizeEdges())]
    for t in chain:
        wall, interior = t((wall, interior))
    with torch.no_grad():
        _, _, vel, _ = pred.model(Batch.from_data_list([wall]).to(pred.device),
                                  Batch.from_data_list([interior]).to(pred.device), pred.waveform)
    vel = (vel.float().cpu() * pred.vel_std + pred.vel_mean).numpy()
    return gt_world, vel_gt, interior.keep_row.numpy(), vel


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n_cases", type=int, default=6)
    p.add_argument("--crop_p", type=float, default=0.25)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--measure_density", action="store_true")
    p.add_argument("--skip_remesh", action="store_true")
    p.add_argument("--sanity_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity"),
                   help="per case: <case>_cfd_mesh.png (CFD points, with CFD velocity + error) and "
                        "<case>_deploy.png (remeshed surface + sampled lumen, prediction only)")
    args = p.parse_args()
    if args.measure_density:
        measure_density(args.root)
        return
    pred = JointVelocityPredictor.from_bundle(args.bundle, args.device)
    pred.args.save_dir = args.root
    ds, cases = test_cases(pred, args.n_cases)
    print(f"[bundle] {args.bundle}  epoch {pred.epoch}  ->  {len(cases)} AngioFlow test cases")
    rows = []
    for idx in cases:
        v, cid, _ = ds._index[idx]
        s = ds._load_static(v, cid)
        nt = s["node_type"].numpy()
        verts_mm = s["pos"].double().numpy() * 1e3 + OFFSET_MM
        inlet_mm = verts_mm[nt == INLET].mean(0)
        outlet_mm = [verts_mm[nt == k].mean(0) for k in (OUTLET_0, OUTLET_1) if (nt == k).any()]
        surf = prepare_surface(verts_mm, s["faces"].numpy(), units="mm", inlet_point=inlet_mm, outlet_points=outlet_mm, smooth=False,
                               focused_mask=nt == FOCUSED, remesh=False, lap_k=pred.lap_k)

        # 1. features vs the stored statics
        dot = (surf.cot_eigvec * s["cot_eigvec"][:, :pred.lap_k]).sum(0).abs()
        nrm = (surf.normals * s["normals"]).sum(1)
        g_ref = s["global_geo_dist"]
        feat = dict(geo=float((surf.geo_dist - s["geo_dist"]).abs().max()),
                    normal_cos_min=float(nrm.min()), eigval=float(((surf.cot_eigval - s["cot_eigval"][:pred.lap_k]).abs()
                                                                   / s["cot_eigval"][:pred.lap_k].abs().clamp_min(1e-6)).max()),
                    eigvec_dot_min=float(dot.min()),
                    gglobal=float(np.corrcoef(surf.global_geo_dist, g_ref)[0, 1]),
                    flow=float((surf.flow_split.view(1, 2) - torch.tensor(ds._flow_split[(v, cid)])).abs().max()))

        # 2. same CFD points, same crop: deploy vs dataset path
        gt_world, vel_gt, kept_rows, vel_ds = dataset_path(pred, ds, idx, args.crop_p)
        q_mm = gt_world * 1e3 + OFFSET_MM
        out = pred.predict(surf, query_points=q_mm, crop_p=args.crop_p)
        vel_dep, gt_k = out.velocity, vel_gt[out.query_index]
        save_velocity_sanity(surf, out, os.path.join(args.sanity_dir, f"{cid}_cfd_mesh.png"), vel_gt=gt_k,
                             title=f"{cid}: CFD wall, CFD interior points")
        # same crop as the dataset (its stored focused-region geodesic) -> must match exactly
        g_own = surf.global_geo_dist
        surf.global_geo_dist = s["global_geo_dist"]
        ex = pred.predict(surf, query_points=q_mm, crop_p=args.crop_p)
        surf.global_geo_dist = g_own
        common, i_ex, i_ds = np.intersect1d(ex.query_index, kept_rows, return_indices=True)
        ds_vs_dep = rel_l2(ex.velocity[i_ex], vel_ds[i_ds])
        n_diff = len(ex.query_index) + len(kept_rows) - 2 * len(common)

        # 3. 20% of the pool (training density) and a sampled lumen as context
        rng = np.random.default_rng(0)
        sub = np.sort(rng.choice(len(q_mm), size=len(q_mm) // 5, replace=False))
        o20 = pred.predict(surf, query_points=q_mm[sub], crop_p=args.crop_p)
        probe = np.sort(rng.choice(len(q_mm), size=len(q_mm) // 20, replace=False))
        samp = pred.predict(surf, crop_p=args.crop_p)
        mix_pts = np.concatenate([samp.points, q_mm[probe]])
        omix = pred.predict(surf, query_points=mix_pts, crop_p=args.crop_p)
        pm = omix.query_index >= len(samp.points)
        r = dict(case=cid, n_wall=int(out.wall_kept.sum()), n_q=len(out.query_index),
                 vel_full=rel_l2(vel_dep, gt_k), vel_full_dataset=rel_l2(vel_ds, vel_gt[kept_rows]),
                 ds_vs_deploy=ds_vs_dep, ds_vs_deploy_pts_differ=n_diff,
                 vel_20pct=rel_l2(o20.velocity, vel_gt[sub[o20.query_index]]),
                 n_sampled=len(samp.points),
                 vel_probe_in_sampled=rel_l2(omix.velocity[pm], vel_gt[probe[omix.query_index[pm] - len(samp.points)]]),
                 **feat)

        # 4. remeshed wall (new vertices, new features), scored on the CFD points
        if not args.skip_remesh:
            surf_r = prepare_surface(verts_mm, s["faces"].numpy(), units="mm", inlet_point=inlet_mm, outlet_points=outlet_mm, smooth=False,
                                     focused_mask=nt == FOCUSED, remesh=True, lap_k=pred.lap_k)
            o_r = pred.predict(surf_r, query_points=q_mm[sub], crop_p=args.crop_p)
            o_s = pred.predict(surf_r)
            save_velocity_sanity(surf_r, o_s, os.path.join(args.sanity_dir, f"{cid}_deploy.png"),
                                 title=f"{cid}: remeshed wall, sampled lumen points")
            r.update(remesh_nodes=surf_r.report["nodes"], remesh_edge_mm=round(surf_r.report["edge_mm"], 4),
                     vel_remeshed_20pct=rel_l2(o_r.velocity, vel_gt[sub[o_r.query_index]]))
        rows.append(r)
        print("  " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)

    print("\nmedian over cases:")
    for k in rows[0]:
        if isinstance(rows[0][k], float):
            print(f"  {k:22s} {np.nanmedian([r[k] for r in rows]):.4g}")


if __name__ == "__main__":
    main()
