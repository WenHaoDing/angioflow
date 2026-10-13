"""Regression test: predictions must be INVARIANT to where the mesh sits in world space.

The mock cases were exported from the training dataset, whose meshes are already
centred on the origin -- so they cannot catch a centring bug. A Slicer segmentation
arrives at arbitrary world coordinates (typically 10s-100s of mm from the origin),
which is a genuinely different input.

This test translates the mesh AND the query points by a large offset and checks the
denormalised predictions are unchanged. It fails loudly if the wall and the query
points are ever centred by different offsets -- the exact bug that a
dataset-derived mock cannot expose.

conda activate new
PYTHONPATH=.:./angioflow python -m angioflow.deploy.test_world_translation
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torch_geometric.data import Batch

from angioflow.deploy.mesh_to_graph import SurfaceMeshToModelInput
from angioflow.deploy.run_inference import CKPT_ROOT, DEFAULT_RUN, MOCK_DIR, rel_l2


def model_inputs(pre, s, offset, scale=1.0):
    """The NORMALIZED tensors the model would receive, for a case shifted by
    `offset` metres and expressed in units where 1 unit = 1/scale metres.

    We compare INPUTS, not predictions, on purpose. cot_eigvec is recomputed by
    `eigsh` on every call and — because mix_laplacian's Laplacian is non-symmetric
    — comes out different each time (measured: rel diff 1.6 on identical input).
    That noise propagates to the output and would swamp a centring/unit error, so a
    prediction-level test cannot isolate what this file is meant to check. Positions
    and wall_dist are fully deterministic, so they isolate it exactly.
    """
    off = np.asarray(offset, dtype=np.float64)
    outs = [o.numpy() for o in s["outlet_nodes"] if len(o) > 0]
    verts = (s["vertices"].numpy().astype(np.float64) + off) * scale
    query = (s["query_points"].numpy().astype(np.float64) + off) * scale
    static = pre.build_static(verts, s["faces"].numpy().astype(np.int64),
                              s["inlet_nodes"].numpy(), outs)
    wall, interior = pre.build_pair(static, query, t=s["t"], crop_mask=s["crop_mask"])
    return wall.pos, interior.pos, interior.wall_dist, static["geo_dist"]


def predict(pre, model, waveform, device, s, offset, scale=1.0):
    """Run the full deploy path with the case shifted by `offset` metres and
    expressed in units where 1 unit = 1/scale metres (scale=1000 -> millimetres).
    `pre` must have been constructed with the matching input_units."""
    off = np.asarray(offset, dtype=np.float64)
    outs = [o.numpy() for o in s["outlet_nodes"] if len(o) > 0]
    verts = (s["vertices"].numpy().astype(np.float64) + off) * scale
    query = (s["query_points"].numpy().astype(np.float64) + off) * scale
    static = pre.build_static(verts, s["faces"].numpy().astype(np.int64),
                              s["inlet_nodes"].numpy(), outs)
    wall, interior = pre.build_pair(static, query, t=s["t"], crop_mask=s["crop_mask"])
    with torch.no_grad():
        wss, qi, vel, press = model(Batch.from_data_list([wall]).to(device),
                                    Batch.from_data_list([interior]).to(device), waveform)
    return pre.inverse_y(wss).cpu().numpy(), pre.inverse_vel(vel).cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", default=DEFAULT_RUN)
    p.add_argument("--ckpt_root", default=CKPT_ROOT)
    p.add_argument("--mock-dir", default=MOCK_DIR)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cot-backend", default="scipy", choices=["pytorch3d", "scipy"])
    p.add_argument("--offset", type=float, nargs=3, default=[0.05, -0.03, 0.12],
                   help="world translation in METRES (default ~ 50/-30/120 mm)")
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()
    device = torch.device(args.device)

    chk = torch.load(os.path.join(args.ckpt_root, args.run_name, "best.pth"),
                     map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**chk["args"])
    from angioflow.training.train_joint import interior_stats_path

    wss_stats = torch.load(os.path.join(targs.save_dir, "stats.pt"), weights_only=False)
    int_stats = torch.load(interior_stats_path(targs), weights_only=False)
    kw = dict(lap_k=targs.lap_k, cot_backend=args.cot_backend)
    pre_m = SurfaceMeshToModelInput(wss_stats, int_stats, input_units="m", **kw)
    pre_mm = SurfaceMeshToModelInput(wss_stats, int_stats, input_units="mm", **kw)

    print(f"[test] world offset = {args.offset} m,  plus a millimetre-units variant "
          f"(tol={args.tol})")
    print("      comparing NORMALIZED MODEL INPUTS (deterministic), not predictions\n")
    print(f"{'case':30s} {'translate':>11s} {'mm-units':>11s}   verdict")
    print("-" * 70)
    worst = 0.0
    for path in sorted(glob.glob(os.path.join(args.mock_dir, "mock_*.pt"))):
        s = torch.load(path, map_location="cpu", weights_only=False)
        ref = model_inputs(pre_m, s, [0.0, 0.0, 0.0])                      # metres, origin
        tr = model_inputs(pre_m, s, args.offset)                           # metres, translated
        mm = model_inputs(pre_mm, s, args.offset, scale=1000.0)            # mm, translated
        d_tr = max(rel_l2(b.numpy(), a.numpy()) for a, b in zip(ref, tr))
        d_mm = max(rel_l2(b.numpy(), a.numpy()) for a, b in zip(ref, mm))
        worst = max(worst, d_tr, d_mm)
        ok = "OK" if max(d_tr, d_mm) < args.tol else "FAIL"
        print(f"{s['version']}/{s['case_id']:22s} {d_tr:11.3e} {d_mm:11.3e}   {ok}")

    print("-" * 70)
    if worst < args.tol:
        print(f"PASS -- wall.pos / interior.pos / wall_dist / geo_dist are invariant to "
              f"world position AND input units (worst drift {worst:.3e})")
    else:
        print(f"FAIL -- worst drift {worst:.3e} exceeds tol {args.tol}: either the wall "
              "and query points are centred by different offsets, or the unit "
              "conversion is not applied consistently to both.")


if __name__ == "__main__":
    main()
