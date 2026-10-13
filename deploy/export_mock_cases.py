"""Export a few real test-set samples as MOCK 3D-Slicer inputs.

Each exported .pt pretends to be what the Slicer plug-in will hand us:

    vertices [N,3] (metres)   faces [F,3]   inlet_nodes / outlet_nodes (index arrays)
    query_points [P,3]        -- where velocity/pressure is wanted

...plus the CFD ground truth for that sample (wss / vel / press), used only to
score the reconstruction. The mesh is already CROPPED — matching deployment, where
Slicer's segmentation model delivers the cropped region — so the crop is baked into
the export rather than being something the deploy class has to redo.

Everything is stored in the RAW METRIC FRAME (metres, un-centred, un-normalised):
centring and normalisation are the deploy class's job, so if they were pre-applied
here the validation would not be testing them.

conda activate new
python -m angioflow.deploy.export_mock_cases --run_name <run> --n-cases 3
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from angioflow.data.datasets import load_excluded_keys
from angioflow.training.train import split_by_case
from angioflow.training.train_joint import build_dataset

DEFAULT_RUN = "joint_p1r015_0frame_v1v2_e108f96_gps6_hop2_tok64_wq50_is10_blk3_xf_mh020_pressmax0"
CKPT_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_checkpoints"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mock_cases")


def crop_mask(static, mask_low, mask_high, aneurysm_only, generator=None):
    """The keep-mask AneurysmCrop would produce (angioflow/transform/crop.py)."""
    g = static["global_geo_dist"]
    if aneurysm_only:
        return (static["node_type"] == 1) | (g == 0)
    p = torch.empty(()).uniform_(mask_low, mask_high, generator=generator)
    return (static["node_type"] == 1) | (g <= p * g.max())


def subset_mesh(pos, faces, keep):
    """Restrict a triangle mesh to `keep`, dropping faces that lose any vertex and
    remapping the survivors to compact indices."""
    keep = np.asarray(keep, dtype=bool)
    remap = np.full(len(pos), -1, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    fmask = keep[faces].all(axis=1)
    return pos[keep], remap[faces[fmask]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", default=DEFAULT_RUN)
    p.add_argument("--ckpt_root", default=CKPT_ROOT)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--n-cases", type=int, default=3,
                   help="cases to export PER version (AneuG-Flow and AngioFlow differ in difficulty, "
                        "so validating on one alone is misleading)")
    p.add_argument("--versions", nargs="+", default=["aneug_flow", "angioflow"], choices=["aneug_flow", "angioflow"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chk = torch.load(os.path.join(args.ckpt_root, args.run_name, "best.pth"),
                     map_location="cpu", weights_only=False)
    targs = argparse.Namespace(**chk["args"])
    print(f"[run] {args.run_name}  mask_low={targs.mask_low} mask_high={targs.mask_high} "
          f"aneurysm_only={targs.crop_aneurysm_only} press_zero_max={targs.press_zero_max}")

    exclude = load_excluded_keys(targs.save_dir) if targs.exclude_anomalies else set()
    dataset = build_dataset(targs, exclude)
    _, test_idx = split_by_case(dataset, targs.train_frac, seed=targs.seed)
    print(f"[data] {len(dataset)} samples, {len(test_idx)} test")

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    # take n_cases from EACH requested version -- AngioFlow is the harder task, so a
    # AneuG-Flow-only sample makes the model look better than it is
    picked, per_version = [], {v: 0 for v in args.versions}
    for i in test_idx:
        v = dataset._index[i][0]
        if v in per_version and per_version[v] < args.n_cases:
            picked.append(i)
            per_version[v] += 1
        if all(c >= args.n_cases for c in per_version.values()):
            break
    print(f"[pick] {dict(per_version)}")

    for n, i in enumerate(picked):
        version, cid, t = dataset._index[i]
        wall, interior = dataset[i]                     # raw frame, no transforms set
        static = dataset._load_static(version, cid)

        press = interior.press
        if targs.press_zero_max:                        # PressureZeroMax, on the FULL pool
            press = press - press.max()

        keep = crop_mask(static, targs.mask_low, targs.mask_high,
                         targs.crop_aneurysm_only, generator=gen)
        pos = static["pos"].numpy().astype(np.float64)
        faces = static["faces"].numpy().astype(np.int64)
        new_pos, new_faces = subset_mesh(pos, faces, keep.numpy())

        nt = static["node_type"].numpy()[keep.numpy()]
        # JointAneurysmCrop's interior rule: keep points whose local wall survived
        from scipy.spatial import cKDTree
        d_kept = cKDTree(new_pos).query(interior.pos.numpy())[0]
        alive = torch.from_numpy(d_kept) <= targs.crop_factor * interior.wall_dist + targs.crop_slack

        # NOTE the crop keeps only the aneurysm sac, which contains NO inlet/outlet
        # nodes. Training computes geo_dist and the cotangent eigenvectors on the
        # FULL mesh and only then restricts to the kept nodes, so those features
        # cannot be reproduced from the cropped patch alone (geo_dist would collapse
        # to the all-1.0 "opening absent" padding). We therefore export the FULL
        # mesh plus the segmentation mask — which is also what Slicer actually has:
        # the whole lumen surface, plus a segmentation of the region to crop to.
        nt_full = static["node_type"].numpy()
        sample = {
            # ---- what Slicer will provide ----
            "vertices": static["pos"].float(),                     # [N,3] metres, FULL mesh
            "faces": static["faces"],                              # [F,3]
            "inlet_nodes": torch.from_numpy(np.where(nt_full == 2)[0]),
            "outlet_nodes": [torch.from_numpy(np.where(nt_full == 3)[0]),
                             torch.from_numpy(np.where(nt_full == 4)[0])],
            "crop_mask": keep,                                     # [N] bool, segmentation output
            "query_points": interior.pos[alive].float(),           # [P,3] metres
            "t": int(t),
            # ---- ground truth, for scoring only ----
            "wss_gt": wall.y[keep].float(),                        # [N_kept,3] Pa/s
            "vel_gt": interior.vel[alive].float(),                 # [P,3] m/s
            "press_gt": press[alive].float(),                      # [P,1] Pa
            # ---- provenance ----
            "version": version, "case_id": cid,
            "node_type": static["node_type"],
        }
        name = f"mock_{version}_{cid}_t{t}.pt".replace("/", "_")
        torch.save(sample, os.path.join(args.out_dir, name))
        n_out = sum(int((nt_full == k).any()) for k in (3, 4))
        print(f"[{n+1}/{len(picked)}] {name}: full mesh {len(pos)} verts / {len(faces)} faces, "
              f"crop keeps {int(keep.sum())} verts, {int(alive.sum())} query pts, "
              f"inlet={int((nt_full==2).sum())} nodes, {n_out} outlet(s)")

    print(f"\n[done] wrote {len(picked)} mock case(s) -> {args.out_dir}")


if __name__ == "__main__":
    main()
