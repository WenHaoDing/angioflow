"""REFERENCE END-TO-END: bare surface mesh -> graph features -> model -> physical fields.

This is the worked example for the 3D Slicer plug-in. It shows the complete path
with nothing hidden behind the training dataset classes:

    1. load a mesh + opening labels + segmentation mask   (what Slicer provides)
    2. SurfaceMeshToModelInput.build_static(...)          -> normals, edges,
                                                             per-opening geodesics,
                                                             cotangent eigenpairs
    3. SurfaceMeshToModelInput.build_pair(...)            -> crop, centre, normalise,
                                                             symmetrise edges
    4. Batch.from_data_list([...]) + model(wall, interior, waveform)   -> INFERENCE
    5. inverse_y / inverse_vel / inverse_press            -> back to Pa/s, m/s, Pa
    6. relative L2 against the CFD ground truth

Step 4 needs three things that are easy to get wrong, so note them:
  * both streams must be wrapped in `Batch.from_data_list` even for a single case --
    the model reads `wall.batch` / `interior.batch`, which a bare `Data` lacks;
  * `model.query_ratio = 1.0`, else the WSS head returns a random subset and
    `query_idx` tells you which nodes it scored (see AngioJointField.forward);
  * the waveform tensor is a required forward argument, loaded from the same file
    the run trained with (train_args.waveform_path).

conda activate new
PYTHONPATH=.:./angioflow python -m angioflow.deploy.run_inference
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torch_geometric.data import Batch

from angioflow.deploy.mesh_to_graph import SurfaceMeshToModelInput

DEFAULT_RUN = "joint_p1r015_0frame_v1v2_e108f96_gps6_hop2_tok64_wq50_is10_blk3_xf_mh020_pressmax0"
CKPT_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_checkpoints"
MOCK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_cases")


def rel_l2(pred, gt):
    return float(np.linalg.norm(pred - gt) / max(np.linalg.norm(gt), 1e-12))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", default=DEFAULT_RUN)
    p.add_argument("--ckpt_root", default=CKPT_ROOT)
    p.add_argument("--mock-dir", default=MOCK_DIR)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cot-backend", default="scipy", choices=["pytorch3d", "scipy"])
    args = p.parse_args()
    device = torch.device(args.device)

    # ---------------------------------------------------------------- checkpoint
    ckpt_dir = os.path.join(args.ckpt_root, args.run_name)
    chk = torch.load(os.path.join(ckpt_dir, "best.pth"), map_location=device, weights_only=False)
    targs = argparse.Namespace(**chk["args"])

    from angioflow.training.train import load_waveform
    from angioflow.training.train_joint import build_model, interior_stats_path

    model = build_model(targs, device)
    model.load_state_dict(chk["model"])
    model.eval()
    model.query_ratio = 1.0          # score EVERY wall node, not a random subset
    waveform = load_waveform(targs.waveform_path, length=1000, device=device)
    print(f"[ckpt] {args.run_name}  epoch={chk['epoch']}")

    # ---------------------------------------------------------------- normalisation
    # MUST be the same stats files the run trained with -- they set the position and
    # field scales, so a mismatch silently rescales every prediction.
    wss_stats = torch.load(os.path.join(targs.save_dir, "stats.pt"), weights_only=False)
    interior_stats = torch.load(interior_stats_path(targs), weights_only=False)
    pre = SurfaceMeshToModelInput(wss_stats, interior_stats, lap_k=targs.lap_k,
                                  cot_backend=args.cot_backend)
    print(f"[norm] pos_std={float(pre.pos_scalar):.6g}  wss_std={float(pre.y_std.flatten()[0]):.4g}  "
          f"vel_std={float(pre.vel_std.flatten()[0]):.4g}  press_std={float(pre.press_std.flatten()[0]):.4g}")

    files = sorted(glob.glob(os.path.join(args.mock_dir, "mock_*.pt")))
    if not files:
        raise SystemExit(f"no mock_*.pt in {args.mock_dir} -- run export_mock_cases.py first")
    print(f"[mock] {len(files)} case(s)  cot_backend={args.cot_backend}\n")

    rows = []
    for path in files:
        s = torch.load(path, map_location="cpu", weights_only=False)

        # -- 2. geometry features from mesh + opening labels ------------------
        outlets = [o.numpy() for o in s["outlet_nodes"] if len(o) > 0]
        static = pre.build_static(
            vertices=s["vertices"].numpy().astype(np.float64),
            faces=s["faces"].numpy().astype(np.int64),
            inlet_nodes=s["inlet_nodes"].numpy(),
            outlet_nodes=outlets)

        # -- 3. crop / centre / normalise / symmetrise -------------------------
        wall, interior = pre.build_pair(static, s["query_points"].numpy(),
                                        t=s["t"], crop_mask=s["crop_mask"])

        # -- 4. inference (note the Batch wrapping -- the model needs .batch) ---
        wall_b = Batch.from_data_list([wall]).to(device)
        int_b = Batch.from_data_list([interior]).to(device)
        with torch.no_grad():
            wss_pred, query_idx, vel_pred, press_pred = model(wall_b, int_b, waveform)
        assert query_idx is None, "expected full-resolution WSS (model.query_ratio = 1.0)"

        # -- 5. denormalise back to physical units ----------------------------
        wss_pr = pre.inverse_y(wss_pred).cpu().numpy()          # Pa/s
        vel_pr = pre.inverse_vel(vel_pred).cpu().numpy()        # m/s
        press_pr = pre.inverse_press(press_pred).cpu().numpy().reshape(-1)   # Pa

        # -- 6. score against CFD ground truth --------------------------------
        wss_gt = s["wss_gt"].numpy()
        vel_gt = s["vel_gt"].numpy()
        press_gt = s["press_gt"].numpy().reshape(-1)
        press_ref = float(press_gt.mean())    # pressure has an arbitrary offset; the
                                              # anomaly is what is physically meaningful
        r = {
            "case": f"{s['version']}/{s['case_id']}",
            "n_wall": len(wss_gt), "n_query": len(vel_gt),
            "wss_rel_l2": rel_l2(wss_pr, wss_gt),
            "vel_rel_l2": rel_l2(vel_pr, vel_gt),
            "press_rel_l2": rel_l2(press_pr - press_ref, press_gt - press_ref),
            "wss_rmse": float(np.sqrt(np.mean((wss_pr - wss_gt) ** 2))),
            "vel_rmse": float(np.sqrt(np.mean((vel_pr - vel_gt) ** 2))),
        }
        rows.append(r)
        print(f"  {r['case']:22s} wall={r['n_wall']:6d} q={r['n_query']:6d}  "
              f"wss_relL2={r['wss_rel_l2']:.3f}  vel_relL2={r['vel_rel_l2']:.3f}  "
              f"press_relL2={r['press_rel_l2']:.3f}")

    print("\n" + "=" * 64)
    print("MEAN over mock cases (compare against the training eval's ~0.35)")
    print("=" * 64)
    for k in ("wss_rel_l2", "vel_rel_l2", "press_rel_l2"):
        print(f"  {k:14s} {np.mean([r[k] for r in rows]):.4f}")
    for k in ("wss_rmse", "vel_rmse"):
        unit = "Pa/s" if k.startswith("wss") else "m/s"
        print(f"  {k:14s} {np.mean([r[k] for r in rows]):.4g} {unit}")


if __name__ == "__main__":
    main()
