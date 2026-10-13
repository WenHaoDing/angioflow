"""Head-to-head: deploy preprocessing vs the native training pipeline, same samples.

run_inference.py shows the deploy path works, but a good relative-L2 alone does not
prove the preprocessing is FAITHFUL -- an input that is subtly wrong can still score
plausibly. This script removes that doubt by predicting the same (case, frame) twice:

  A. NATIVE : the training dataset's own static tensors + the eval transform chain
  B. DEPLOY : SurfaceMeshToModelInput, from mesh + opening labels + crop mask only

Both use the IDENTICAL crop mask (read from the mock file), so any difference is
attributable to the feature computation, not to a different random crop. If the two
predictions agree, the deploy path reproduces training.

conda activate new
PYTHONPATH=.:./angioflow python -m angioflow.deploy.compare_with_native
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected

from angioflow.deploy.mesh_to_graph import SurfaceMeshToModelInput
from angioflow.deploy.run_inference import CKPT_ROOT, DEFAULT_RUN, MOCK_DIR, rel_l2


def native_pair(static, s, pre):
    """Build the model input the TRAINING way: reuse the precomputed static tensors,
    apply the same crop mask, then centre / normalise / symmetrise."""
    keep = s["crop_mask"]
    wall = Data(pos=static["pos"], edge_index=static["edge_index"],
                normals=static["normals"], geo_dist=static["geo_dist"],
                cot_eigvec=static["cot_eigvec"], cot_eigval=static["cot_eigval"],
                t=torch.tensor([s["t"]]))
    int_pos = s["query_points"].clone()
    wall_dist = torch.from_numpy(
        __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(
            static["pos"].numpy()).query(int_pos.numpy())[0]).float()
    wall = wall.subgraph(keep)

    c = wall.pos.mean(0, keepdim=True)
    wall.pos = (wall.pos - c) / pre.pos_std
    int_pos = (int_pos - c) / pre.pos_std
    wall_dist = wall_dist / pre.pos_scalar
    wall.edge_index = to_undirected(wall.edge_index, num_nodes=wall.num_nodes)
    interior = Data(pos=int_pos, wall_dist=wall_dist, t=torch.tensor([s["t"]]))
    return wall, interior


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

    wss_stats = torch.load(os.path.join(targs.save_dir, "stats.pt"), weights_only=False)
    interior_stats = torch.load(interior_stats_path(targs), weights_only=False)
    pre = SurfaceMeshToModelInput(wss_stats, interior_stats, lap_k=targs.lap_k,
                                  cot_backend=args.cot_backend)

    print(f"{'case':22s} {'NATIVE wss/vel':>18s} {'DEPLOY wss/vel':>18s} {'pred agree':>12s}")
    print("-" * 76)
    agg = {"n_wss": [], "n_vel": [], "d_wss": [], "d_vel": [], "a_wss": [], "a_vel": []}

    for path in sorted(glob.glob(os.path.join(args.mock_dir, "mock_*.pt"))):
        s = torch.load(path, map_location="cpu", weights_only=False)
        static_ref = torch.load(
            os.path.join(targs.save_dir, f"static_{s['version']}_{s['case_id']}.pt"),
            weights_only=False)

        outs = [o.numpy() for o in s["outlet_nodes"] if len(o) > 0]
        static_dep = pre.build_static(s["vertices"].numpy().astype(np.float64),
                                      s["faces"].numpy().astype(np.int64),
                                      s["inlet_nodes"].numpy(), outs)

        preds = {}
        for tag, (w, i) in {
                "native": native_pair(static_ref, s, pre),
                "deploy": pre.build_pair(static_dep, s["query_points"].numpy(),
                                         t=s["t"], crop_mask=s["crop_mask"])}.items():
            with torch.no_grad():
                wss, qi, vel, press = model(Batch.from_data_list([w]).to(device),
                                            Batch.from_data_list([i]).to(device), waveform)
            preds[tag] = (pre.inverse_y(wss).cpu().numpy(), pre.inverse_vel(vel).cpu().numpy())

        wss_gt, vel_gt = s["wss_gt"].numpy(), s["vel_gt"].numpy()
        n_wss, n_vel = rel_l2(preds["native"][0], wss_gt), rel_l2(preds["native"][1], vel_gt)
        d_wss, d_vel = rel_l2(preds["deploy"][0], wss_gt), rel_l2(preds["deploy"][1], vel_gt)
        # how close are the two PREDICTIONS to each other (0 = identical pipelines)
        a_wss = rel_l2(preds["deploy"][0], preds["native"][0])
        a_vel = rel_l2(preds["deploy"][1], preds["native"][1])
        for k, v in zip(agg, (n_wss, n_vel, d_wss, d_vel, a_wss, a_vel)):
            agg[k].append(v)
        print(f"{s['version']}/{s['case_id']:14s} {n_wss:8.3f} /{n_vel:7.3f} "
              f"{d_wss:9.3f} /{d_vel:7.3f} {a_wss:7.3f} /{a_vel:5.3f}")

    print("-" * 76)
    print(f"{'MEAN':22s} {np.mean(agg['n_wss']):8.3f} /{np.mean(agg['n_vel']):7.3f} "
          f"{np.mean(agg['d_wss']):9.3f} /{np.mean(agg['d_vel']):7.3f} "
          f"{np.mean(agg['a_wss']):7.3f} /{np.mean(agg['a_vel']):5.3f}")
    print("\n'pred agree' is deploy-vs-native prediction difference: small => the "
          "deploy preprocessing reproduces the training pipeline.")


if __name__ == "__main__":
    main()
