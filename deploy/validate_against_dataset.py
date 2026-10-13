"""Validate SurfaceMeshToModelInput against the real training preprocessing.

Takes real cases, throws away everything except what a 3D Slicer operator would
have (vertex positions, triangles, and which boundary loop is inlet / outlet),
rebuilds the graph features with the deploy class, and diffs them against the
`static_*.pt` the training pipeline actually wrote.

This is the check that the deployment path is numerically the same pipeline, not
merely a plausible-looking one.

conda activate new
python -m angioflow.deploy.validate_against_dataset
python -m angioflow.deploy.validate_against_dataset --n-cases 5 --cot-backend scipy
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from angioflow.data.dataset_tags import split_tag_cid
from angioflow.deploy.mesh_to_graph import (ANEURYSM, INLET, OUTLET_0, OUTLET_1,
                                            SurfaceMeshToModelInput)

SAVE_DIR = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full"
CASE_ROOTS = {"aneug_flow": "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data",
              "angioflow": "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"}


def _rel(a, b):
    """Relative L2 over FINITE entries only.

    geo_dist legitimately contains +inf at nodes unreachable from an opening (the
    training pipeline leaves them inf too), and inf-inf would poison the norm --
    so compare where both sides are finite and separately assert the inf PATTERN
    matches, which is the part that actually has to agree.
    """
    a, b = a.double(), b.double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    if (torch.isfinite(a) != torch.isfinite(b)).any():
        return float("nan")            # inf pattern disagrees -> genuine mismatch
    den = b[finite].norm().clamp_min(1e-12)
    return float((a[finite] - b[finite]).norm() / den)


def check_case(pre, save_dir, version, cid, verbose=True):
    """Rebuild one case from mesh+openings only; compare with its static_*.pt."""
    static_ref = torch.load(os.path.join(save_dir, f"static_{version}_{cid}.pt"),
                            weights_only=False)
    reg = torch.load(os.path.join(CASE_ROOTS[version], cid, "wall_mesh_registration.pt"),
                     weights_only=False)

    # ---- what the operator would give us, and nothing else ----
    vertices = reg["pos"].numpy().astype(np.float64)
    faces = reg["faces"].numpy().astype(np.int64)
    nt = reg["node_type"].numpy()
    inlet_nodes = np.where(nt == INLET)[0]
    outlet_nodes = [np.where(nt == OUTLET_0)[0]]
    if (nt == OUTLET_1).any():
        outlet_nodes.append(np.where(nt == OUTLET_1)[0])
    aneurysm_mask = (nt == ANEURYSM)

    static = pre.build_static(vertices, faces, inlet_nodes, outlet_nodes,
                              aneurysm_mask=aneurysm_mask)

    results = {}
    results["pos"] = _rel(static["pos"], static_ref["pos"])
    results["normals"] = _rel(static["normals"], static_ref["normals"])
    results["geo_dist"] = _rel(static["geo_dist"], static_ref["geo_dist"])
    results["cot_eigval"] = _rel(static["cot_eigval"], static_ref["cot_eigval"])
    # eigenvectors are sign/rotation ambiguous within degenerate subspaces --
    # compare the invariant subspace via |V^T V_ref| column norms instead of raw diff
    V, Vr = static["cot_eigvec"].double(), static_ref["cot_eigvec"].double()
    overlap = (V.T @ Vr).abs()
    results["cot_eigvec_subspace"] = float(overlap.diagonal().mean())

    e_new = static["edge_index"]
    e_ref = static_ref["edge_index"]
    same_edges = (e_new.shape == e_ref.shape) and bool((e_new == e_ref).all())
    results["edge_index_identical"] = same_edges
    results["node_type_identical"] = bool((static["node_type"] == static_ref["node_type"]).all())

    if verbose:
        print(f"\n{version}/{cid}  N={len(vertices)}  outlets={len(outlet_nodes)}")
        print(f"  pos           rel_err = {results['pos']:.3e}")
        print(f"  normals       rel_err = {results['normals']:.3e}")
        print(f"  geo_dist      rel_err = {results['geo_dist']:.3e}")
        print(f"  cot_eigval    rel_err = {results['cot_eigval']:.3e}")
        print(f"  cot_eigvec    mean |<v_i,v_ref_i>| = {results['cot_eigvec_subspace']:.4f}  (1.0 = identical)")
        print(f"  edge_index    identical = {results['edge_index_identical']}")
        print(f"  node_type     identical = {results['node_type_identical']}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save-dir", default=SAVE_DIR)
    p.add_argument("--n-cases", type=int, default=3)
    p.add_argument("--cot-backend", default="pytorch3d", choices=["pytorch3d", "scipy"])
    p.add_argument("--lap-k", type=int, default=8)
    p.add_argument("--interior-stats", default=None,
                   help="default: <save-dir>/interior_stats_zeromax.pt")
    args = p.parse_args()

    wss_stats = torch.load(os.path.join(args.save_dir, "stats.pt"), weights_only=False)
    interior_stats = torch.load(
        args.interior_stats or os.path.join(args.save_dir, "interior_stats_zeromax.pt"),
        weights_only=False)
    pre = SurfaceMeshToModelInput(wss_stats, interior_stats, lap_k=args.lap_k,
                                  cot_backend=args.cot_backend)

    # prefer cases whose raw registration is still on disk
    cases = []
    for path in sorted(glob.glob(os.path.join(args.save_dir, "static_*.pt"))):
        stem = os.path.basename(path)[len("static_"):-len(".pt")]
        version, cid = split_tag_cid(stem)
        if version not in CASE_ROOTS:
            continue
        if os.path.exists(os.path.join(CASE_ROOTS[version], cid, "wall_mesh_registration.pt")):
            cases.append((version, cid))
        if len(cases) >= args.n_cases:
            break

    if not cases:
        print("[error] no case found with both static_*.pt and wall_mesh_registration.pt")
        return

    print(f"[validate] cot_backend={args.cot_backend}  lap_k={args.lap_k}  {len(cases)} case(s)")
    all_res = [check_case(pre, args.save_dir, v, c) for v, c in cases]

    print("\n" + "=" * 60)
    print("SUMMARY (worst across cases)")
    print("=" * 60)
    for key in ("pos", "normals", "geo_dist", "cot_eigval"):
        print(f"  {key:14s} max rel_err = {max(r[key] for r in all_res):.3e}")
    print(f"  {'cot_eigvec':14s} min subspace overlap = "
          f"{min(r['cot_eigvec_subspace'] for r in all_res):.4f}")
    print(f"  {'edge_index':14s} all identical = {all(r['edge_index_identical'] for r in all_res)}")
    print(f"  {'node_type':14s} all identical = {all(r['node_type_identical'] for r in all_res)}")

    # end-to-end: build the actual model input pair
    print("\n[end-to-end] building a (wall, interior) pair for the first case ...")
    version, cid = cases[0]
    reg = torch.load(os.path.join(CASE_ROOTS[version], cid, "wall_mesh_registration.pt"),
                     weights_only=False)
    nt = reg["node_type"].numpy()
    outs = [np.where(nt == OUTLET_0)[0]]
    if (nt == OUTLET_1).any():
        outs.append(np.where(nt == OUTLET_1)[0])
    static = pre.build_static(reg["pos"].numpy().astype(np.float64),
                             reg["faces"].numpy().astype(np.int64),
                             np.where(nt == INLET)[0], outs, aneurysm_mask=(nt == ANEURYSM))
    # mock interior query points: jitter inside the bounding box of the aneurysm
    rng = np.random.default_rng(0)
    sac = reg["pos"].numpy()[nt == ANEURYSM]
    q = sac[rng.integers(0, len(sac), 2000)] + rng.normal(0, 2e-4, (2000, 3))
    wall, interior = pre.build_pair(static, q, t=0, crop=True, seed=0)
    print(f"  wall     : pos{tuple(wall.pos.shape)} edges{tuple(wall.edge_index.shape)} "
          f"geo_dist{tuple(wall.geo_dist.shape)} cot_eigvec{tuple(wall.cot_eigvec.shape)}")
    print(f"  interior : pos{tuple(interior.pos.shape)} wall_dist{tuple(interior.wall_dist.shape)}")
    print(f"  wall.pos  std={wall.pos.std():.3f} (normalized, ~O(1) expected)")


if __name__ == "__main__":
    main()
