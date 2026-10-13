"""FULL DEPLOYMENT REHEARSAL: smooth + remesh the surface, then predict, then look at it.

This is run_inference.py with the mesh regularizer switched ON, i.e. the actual
sequence the 3D Slicer plug-in will follow:

    segmentation surface
      -> MeshRegularizer            smooth to the reference roughness, remesh to
                                    DOWNSTREAM_EDGE_MM (0.125 mm = training resolution)
      -> re-map opening/crop labels onto the NEW vertices  (remeshing changes them all)
      -> build_static / build_pair  geometry features, crop, centre, normalise
      -> model                      WSS + velocity + pressure
      -> inverse_*                  back to Pa/s, m/s, Pa
      -> render                     predicted vs ground-truth WSS magnitude

WHY THE NUMBERS MOVE. Regularization deliberately changes the surface: it removes
voxel-scale roughness and resamples the triangulation, so the predicted field lives
on different vertices than the CFD ground truth and the two are compared by nearest
neighbour. A raised relative L2 versus the un-regularized run is therefore EXPECTED
and is not by itself evidence of a bug -- the question these renders answer is
whether the WSS PATTERN survives (high-shear bands, the low-shear sac), which is
what the plug-in is for.

conda activate new
PYTHONPATH=.:./angioflow python -m angioflow.deploy.run_regularized_inference
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Batch

from angioflow.deploy.mesh_to_graph import SurfaceMeshToModelInput
from angioflow.deploy.run_inference import CKPT_ROOT, DEFAULT_RUN, MOCK_DIR, rel_l2

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regularized_viz")


def render_wss(mesh_pos, faces, gt_mag, pred_mag, title, out_path):
    """Side-by-side WSS magnitude, ground truth vs prediction, shared colour scale."""
    import pyvista as pv

    # headless box: VTK here is built against X (no OSMesa), so without a virtual
    # framebuffer the render aborts with "bad X server connection"
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()

    cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    clim = [0.0, float(np.percentile(gt_mag, 99))]     # p99 so a few hot nodes
                                                        # do not flatten the map
    p = pv.Plotter(off_screen=True, window_size=(1500, 700), shape=(1, 2))
    for col, (scal, sub) in enumerate(((gt_mag, "ground truth (CFD)"),
                                       (pred_mag, "prediction (regularized mesh)"))):
        m = pv.PolyData(mesh_pos, cells)
        m["WSS"] = scal
        p.subplot(0, col)
        p.add_mesh(m, scalars="WSS", cmap="turbo", clim=clim,
                   scalar_bar_args={"title": "|WSS| (Pa/s)"})
        p.add_text(sub, font_size=10)
        p.show_axes()
    p.link_views()
    p.add_text(title, position="upper_edge", font_size=11)
    p.screenshot(out_path)
    p.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", default=DEFAULT_RUN)
    p.add_argument("--ckpt_root", default=CKPT_ROOT)
    p.add_argument("--mock-dir", default=MOCK_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cot-backend", default="scipy", choices=["pytorch3d", "scipy"])
    p.add_argument("--no-regularize", action="store_true",
                   help="skip smoothing/remeshing -- for an apples-to-apples baseline")
    p.add_argument("--time-budget", type=float, default=60.0,
                   help="MeshRegularizer wall-clock budget per case, seconds")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    chk = torch.load(os.path.join(args.ckpt_root, args.run_name, "best.pth"),
                     map_location=device, weights_only=False)
    targs = argparse.Namespace(**chk["args"])
    from angioflow.training.train import load_waveform
    from angioflow.training.train_joint import build_model, interior_stats_path

    model = build_model(targs, device)
    model.load_state_dict(chk["model"])
    model.eval()
    model.query_ratio = 1.0
    waveform = load_waveform(targs.waveform_path, length=1000, device=device)

    pre = SurfaceMeshToModelInput(
        torch.load(os.path.join(targs.save_dir, "stats.pt"), weights_only=False),
        torch.load(interior_stats_path(targs), weights_only=False),
        lap_k=targs.lap_k, cot_backend=args.cot_backend,
        regularize=not args.no_regularize,
        regularizer_kwargs=dict(verbose=False, time_budget_s=args.time_budget))

    print(f"[ckpt] {args.run_name} epoch={chk['epoch']}   regularize={not args.no_regularize}")
    print(f"[out ] {args.out_dir}\n")

    rows = []
    for path in sorted(glob.glob(os.path.join(args.mock_dir, "mock_*.pt"))):
        s = torch.load(path, map_location="cpu", weights_only=False)
        tag = f"{s['version']}_{s['case_id']}"
        outs = [o.numpy() for o in s["outlet_nodes"] if len(o) > 0]

        # crop_mask goes to build_static so it is re-mapped onto the remeshed vertices
        static = pre.build_static(
            s["vertices"].numpy().astype(np.float64),
            s["faces"].numpy().astype(np.int64),
            s["inlet_nodes"].numpy(), outs,
            crop_mask=s["crop_mask"].numpy())
        rep = static.get("regularizer_report")
        if rep is not None:
            status = rep.get("status", "?") if isinstance(rep, dict) else "?"
            print(f"  [{tag}] regularizer: {status}, "
                  f"{len(s['vertices'])} -> {len(static['pos'])} verts")

        wall, interior = pre.build_pair(static, s["query_points"].numpy(), t=s["t"])

        with torch.no_grad():
            wss_pred, qi, vel_pred, press_pred = model(
                Batch.from_data_list([wall]).to(device),
                Batch.from_data_list([interior]).to(device), waveform)

        wss_pr = pre.inverse_y(wss_pred).cpu().numpy()
        vel_pr = pre.inverse_vel(vel_pred).cpu().numpy()

        # The prediction lives on the REGULARIZED + CROPPED vertices; the CFD truth
        # lives on the original ones. Map by nearest neighbour IN THE ORIGINAL WORLD
        # FRAME (undo build_static's centring) so the two point sets are genuinely
        # co-located -- the regularized surface sits within a fraction of an edge
        # length of the original, so the transfer is well posed.
        keep = static["crop_mask"].numpy()
        world_all = (static["pos"] + static["center"]).numpy()   # regularized, world frame
        kept_world = world_all[keep]
        gt_world = s["vertices"].numpy().astype(np.float64)[s["crop_mask"].numpy()]
        _, nn = cKDTree(gt_world).query(kept_world, k=1)
        wss_gt_on_pred = s["wss_gt"].numpy()[nn]

        # faces of the cropped patch, re-indexed to the kept vertices (for rendering)
        remap = np.full(len(world_all), -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        f_all = static["faces"].numpy()
        f_kept = remap[f_all[keep[f_all].all(axis=1)]]

        r = {
            "case": tag,
            "n_pred": len(wss_pr), "n_gt": len(s["wss_gt"]),
            # align GT to the query points that actually survived the crop
            "wss_rel_l2": rel_l2(wss_pr, wss_gt_on_pred),
            "vel_rel_l2": rel_l2(vel_pr, s["vel_gt"].numpy()[interior.query_index.numpy()]),
        }
        rows.append(r)

        out_png = os.path.join(args.out_dir, f"wss_{tag}.png")
        render_wss(kept_world, f_kept,
                   np.linalg.norm(wss_gt_on_pred, axis=1), np.linalg.norm(wss_pr, axis=1),
                   f"{tag}   wss relL2={r['wss_rel_l2']:.3f}", out_png)
        print(f"  {tag:26s} wss_relL2={r['wss_rel_l2']:.3f}  vel_relL2={r['vel_rel_l2']:.3f}"
              f"  -> {os.path.basename(out_png)}")

    print("\n" + "=" * 60)
    print(f"MEAN  wss_rel_l2 = {np.mean([r['wss_rel_l2'] for r in rows]):.4f}   "
          f"vel_rel_l2 = {np.mean([r['vel_rel_l2'] for r in rows]):.4f}")
    print(f"images -> {args.out_dir}")


if __name__ == "__main__":
    main()
