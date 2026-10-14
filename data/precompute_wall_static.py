"""Precompute every geometry-only per-node field of the wall graph, once per case,
locally -- so the cloud extraction Job only has to slice WSS frames out of
wall_data.pt and never needs pyvista/pytorch3d.

Input: the Hub's own wall_mesh_registration.pt files, mirrored locally by
angioflow.huggingface.download_wall_mesh_registrations (read from the Hub, not the
local dataset mirrors, so the output matches exactly what the Job will pair it with).
Output: one wall_graph_static.pt per case, in a staging root that mirrors each repo's
layout, ready to upload back next to the registration it was computed from:
    <out_root>/<dataset>/<repo prefix>/<case>/wall_graph_static.pt
    <out_root>/<dataset>/_sanity/<case>/{mesh,geo_dist,cot_eigvec}.png

wall_graph_static.pt node order == registration `pos` order == wall_data.pt node order
(verified exactly equal on all three datasets), so WSS needs no remapping:
    pos             [N, 3]  float32  registration pos minus its mean (all N nodes)
    edge_index      [2, E]  int64    face edges + orphan kNN edges, unique, i < j
    faces           [F, 3]  int64    registration faces, unchanged
    node_type       [N]     int64    0=wall 1=focused 2=inlet 3=outlet_0 4=outlet_1
    flow_split      [2]     float32
    aneurysm_type   scalar  int64
    global_geo_dist [N]     float32  registration geo_dist / its max, in [0, 1]
    geo_dist        [N, 3]  float32  per-opening (inlet, outlet_0, outlet_1) geodesic
                                     distance, each in [0, 1]; absent opening -> 1.0
    normals         [N, 3]  float32  unit outward vertex normals
    cot_eigval      [K]     float32  ascending; K = lap_k (default 64)
    cot_eigvec      [N, K]  float32  each mode unit L2 norm, deterministic sign
    orphan_mask     [N]     bool     node in no face -> fields filled by kNN (below)
    meta            dict             config, source sha256, QC numbers

Cotangent Laplacian: the standard FEM form L v = lam M v (L = cot stiffness,
M = lumped barycentric vertex area), solved as a symmetric generalized problem by
shift-invert. Replaces graph.compute_pe_cot_eigenvector, which handed the
non-symmetric row-normalized I - D^-1 W to eigsh (a symmetric solver): its output
was not converged (residual ~ eigenvalue size) and changed run to run, so 8 modes
were not the first 8 of 64. This version is exact (residual ~1e-14), identical on
rerun, and its first k modes don't depend on how many were requested -- downstream
training can slice cot_eigvec[:, :k] for any k <= lap_k. M makes the modes
independent of mesh density, which differs between the three datasets.
Mode 0 is the constant vector (lam = 0).

Orphan vertices: nodes in `pos` used by no face (AneuG-Flow: 633/634 cases, 1-19
each; none in AngioFlow/AneuGv2). They are real wall nodes with real WSS (within ~2%
of their neighbours') lying ~0.5 median edge lengths from the surface, but the old
pipeline gave them inf geo_dist, zero normals and an isolated graph node. Here every
per-node field (normals, geo_dist, cot_eigvec) is computed on the face-referenced
surface only, then filled at orphans by inverse-distance-weighted kNN (k=3)
interpolation from their nearest surface vertices; the same 3 neighbours are added
to edge_index so orphans take part in message passing.

OMP_NUM_THREADS=1 keeps each worker's BLAS single-threaded (workers already run in parallel):
    OMP_NUM_THREADS=1 python -m angioflow.data.precompute_wall_static --cases_per_dataset 3  # smoke test
    OMP_NUM_THREADS=1 python -m angioflow.data.precompute_wall_static --workers 16          # everything
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import time
import traceback
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree

from angioflow.data.mesh import compute_vertex_normals, face_to_edge_index
from angioflow.data.per_opening_geo import per_opening_geo_dist

REG_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/hf_registrations"
OUT_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/hf_wall_graph_static"
DATASETS = ("aneug_flow", "angioflow", "aneugv2")
OUT_NAME = "wall_graph_static.pt"
LAP_K = 64
KNN_K = 3
SANITY_MODES = list(range(8)) + [15, 31, 63]
FORMAT_VERSION = 1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cot_stiffness_and_mass(pos, faces):
    """Symmetric cot weights W (w_ij = (cot a + cot b) / 2) and lumped barycentric mass."""
    p = pos[faces]
    e0, e1, e2 = p[:, 1] - p[:, 2], p[:, 2] - p[:, 0], p[:, 0] - p[:, 1]  # edge opposite vertex 0/1/2
    dbl_area = np.linalg.norm(np.cross(e1, e2), axis=1)
    dbl_area_safe = np.maximum(dbl_area, 1e-8 * np.median(dbl_area))
    cots = [-(e1 * e2).sum(1), -(e2 * e0).sum(1), -(e0 * e1).sum(1)]     # x 1/dbl_area below
    I = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])          # angle at v0 <-> edge (v1, v2)
    J = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    V = 0.5 * np.concatenate([c / dbl_area_safe for c in cots])
    N = len(pos)
    W = sp.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    W = (W + W.T).tocsr()
    mass = np.zeros(N)
    for c in range(3):
        np.add.at(mass, faces[:, c], dbl_area / 6.0)
    return W, mass, float(dbl_area.min() / np.median(dbl_area))


def cot_mass_eigs(pos, faces, k):
    """Smallest-k eigenpairs of L v = lam M v. Returns (lam [k], U [N, k], qc dict)."""
    W, mass, min_area_rel = cot_stiffness_and_mass(pos, faces)
    L = (sp.diags(np.asarray(W.sum(1)).ravel()) - W).tocsc()
    M = sp.diags(mass).tocsc()
    scale = L.diagonal().mean() / mass.mean()
    # fixed start vector: ARPACK otherwise draws a random one, leaving last-bit noise run to run
    v0 = np.random.default_rng(0).standard_normal(L.shape[0])
    lam, U = eigsh(L, k=k, M=M, sigma=-1e-6 * scale, which="LM", v0=v0)
    order = np.argsort(lam)
    lam, U = lam[order], U[:, order]
    MU = M @ U
    resid = np.linalg.norm(L @ U - MU * lam, axis=0) / (np.linalg.norm(MU, axis=0) * scale)
    U = U / np.linalg.norm(U, axis=0, keepdims=True)
    # deterministic sign: largest-magnitude entry of each mode positive
    U = U * np.sign(U[np.abs(U).argmax(0), np.arange(U.shape[1])])
    qc = dict(eig_max_resid=float(resid.max()), n_zero_eig=int((np.abs(lam) < 1e-8 * scale).sum()),
              n_neg_eig=int((lam < -1e-8 * scale).sum()), min_area_rel=min_area_rel)
    return lam, U, qc


def idw_fill(values_used, d, nn):
    """Inverse-distance-weighted mean over each orphan's k nearest surface vertices."""
    w = 1.0 / np.maximum(d, 1e-12)
    w = w / w.sum(1, keepdims=True)
    return np.einsum("ok,ok...->o...", w, values_used[nn])


def process_case(reg_path, lap_k=LAP_K, knn_k=KNN_K):
    reg = torch.load(reg_path, weights_only=False)
    pos_all = reg["pos"].numpy().astype(np.float64)
    faces = reg["faces"].numpy().astype(np.int64)
    node_type = reg["node_type"].numpy()
    N = len(pos_all)
    pos_c = pos_all - pos_all.mean(axis=0, keepdims=True)

    # face-referenced surface vs orphans
    used = np.unique(faces)
    orphan_mask = np.ones(N, bool)
    orphan_mask[used] = False
    orphans = np.flatnonzero(orphan_mask)
    remap = -np.ones(N, np.int64)
    remap[used] = np.arange(len(used))
    sub_pos, sub_faces = pos_c[used], remap[faces]
    sub_edges = face_to_edge_index(sub_faces)
    n_comp = connected_components(
        sp.coo_matrix((np.ones(sub_edges.shape[1]), sub_edges), shape=(len(used),) * 2),
        directed=False)[0]

    # per-node fields on the surface only
    t0 = time.time()
    lam, U, qc = cot_mass_eigs(sub_pos, sub_faces, lap_k)
    t_eig = time.time() - t0
    normals_sub = compute_vertex_normals(sub_pos, sub_faces)
    opening = per_opening_geo_dist(sub_pos, sub_edges, node_type[used])
    geo_sub = np.stack([opening.get(n, np.ones(len(used)))
                        for n in ("inlet", "outlet_0", "outlet_1")], axis=1)

    # scatter back to all N nodes, kNN-fill orphans
    cot_eigvec = np.zeros((N, lap_k)); cot_eigvec[used] = U
    normals = np.zeros((N, 3)); normals[used] = normals_sub
    geo_dist = np.ones((N, 3)); geo_dist[used] = geo_sub
    edge_index = face_to_edge_index(faces)
    if len(orphans):
        d, nn_sub = cKDTree(sub_pos).query(pos_c[orphans], k=knn_k)
        cot_eigvec[orphans] = idw_fill(U, d, nn_sub)
        n = idw_fill(normals_sub, d, nn_sub)
        normals[orphans] = n / np.linalg.norm(n, axis=1, keepdims=True)
        geo_dist[orphans] = idw_fill(geo_sub, d, nn_sub)
        knn_edges = np.sort(np.stack([np.repeat(orphans, knn_k), used[nn_sub].ravel()], 1), 1)
        all_edges = np.concatenate([edge_index.T, knn_edges])
        codes = np.unique(all_edges[:, 0] * N + all_edges[:, 1])
        edge_index = np.stack([codes // N, codes % N]).astype(np.int64)

    gg = reg["geo_dist"].float()
    out = {
        "pos":             torch.from_numpy(pos_c).float(),
        "edge_index":      torch.from_numpy(edge_index),
        "faces":           torch.from_numpy(faces),
        "node_type":       reg["node_type"].long(),
        "flow_split":      reg["flow_split"].float(),
        "aneurysm_type":   reg["aneurysm_type"].long(),
        "global_geo_dist": gg / gg.max().clamp_min(1e-8),
        "geo_dist":        torch.from_numpy(geo_dist).float(),
        "normals":         torch.from_numpy(normals).float(),
        "cot_eigval":      torch.from_numpy(lam).float(),
        "cot_eigvec":      torch.from_numpy(cot_eigvec).float(),
        "orphan_mask":     torch.from_numpy(orphan_mask),
    }
    qc.update(n_nodes=N, n_orphans=int(len(orphans)), n_components=int(n_comp), eig_time_s=round(t_eig, 2))
    bad = [k for k, v in out.items()
           if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
    if bad:
        raise ValueError(f"non-finite values in {bad}")
    out["meta"] = dict(format_version=FORMAT_VERSION, laplacian="cot_mass", lap_k=lap_k,
                       knn_k=knn_k, source_file="wall_mesh_registration.pt",
                       source_sha256=sha256_file(reg_path), qc=qc)
    return out


def render_sanity(static, out_dir, dataset, cid):
    from angioflow.visualize.data_sanity import (_cot_eigvec_figure, _geo_dist_figure,
                                                 _mesh_figure, _vis_mesh_and_remap)
    os.makedirs(out_dir, exist_ok=True)
    pos, faces = static["pos"].numpy(), static["faces"].numpy()
    vis_pos, vis_faces, idx = _vis_mesh_and_remap(pos, faces)
    _mesh_figure(vis_pos, vis_faces, os.path.join(out_dir, "mesh.png"), dataset, cid)
    _geo_dist_figure(vis_pos, vis_faces, static["geo_dist"].numpy()[idx],
                     static["global_geo_dist"].numpy()[idx],
                     os.path.join(out_dir, "geo_dist.png"), dataset, cid)
    k = static["cot_eigvec"].shape[1]
    _cot_eigvec_figure(vis_pos, vis_faces, static["cot_eigvec"].numpy()[idx],
                       static["cot_eigval"].numpy(), os.path.join(out_dir, "cot_eigvec.png"),
                       dataset, cid, mode_ids=[m for m in SANITY_MODES if m < k])


def _job(args):
    dataset, reg_path, out_root, redo, sanity = args
    rel = os.path.relpath(os.path.dirname(reg_path), os.path.join(REG_ROOT, dataset))
    cid = os.path.basename(rel)
    out_path = os.path.join(out_root, dataset, rel, OUT_NAME)
    try:
        if not redo and os.path.exists(out_path):
            meta = torch.load(out_path, weights_only=False)["meta"]
            if meta["source_sha256"] == sha256_file(reg_path):
                return dataset, cid, "skip", None
        t0 = time.time()
        static = process_case(reg_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp = out_path + ".tmp"
        torch.save(static, tmp)
        os.replace(tmp, out_path)
        if sanity:
            render_sanity(static, os.path.join(out_root, dataset, "_sanity", cid), dataset, cid)
        return dataset, cid, "ok", dict(static["meta"]["qc"], total_s=round(time.time() - t0, 1))
    except Exception:
        return dataset, cid, "fail", traceback.format_exc(limit=3)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reg_root", default=REG_ROOT)
    p.add_argument("--out_root", default=OUT_ROOT)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--cases", nargs="*", default=None, help="only these case ids")
    p.add_argument("--cases_per_dataset", type=int, default=None, help="smoke test: first N per dataset")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--sanity", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--redo", action="store_true", help="recompute even if output matches source sha256")
    return p.parse_args()


def main():
    args = parse_args()
    jobs = []
    for ds in args.datasets:
        paths = sorted(glob.glob(os.path.join(args.reg_root, ds, "*", "*", "wall_mesh_registration.pt")))
        if args.cases:
            paths = [p for p in paths if os.path.basename(os.path.dirname(p)) in set(args.cases)]
        if args.cases_per_dataset:
            paths = paths[:args.cases_per_dataset]
        print(f"[plan] {ds}: {len(paths)} case(s)")
        jobs += [(ds, p, args.out_root, args.redo, args.sanity) for p in paths]

    counts, failed = {}, []
    with Pool(args.workers) as pool:
        for ds, cid, status, info in pool.imap_unordered(_job, jobs):
            counts[(ds, status)] = counts.get((ds, status), 0) + 1
            if status == "ok":
                print(f"[ok] {ds}/{cid} {info}", flush=True)
            elif status == "fail":
                failed.append((ds, cid))
                print(f"[fail] {ds}/{cid}\n{info}", flush=True)
    print("\n[summary]", {f"{d}/{s}": n for (d, s), n in sorted(counts.items())})
    if failed:
        print("[failed]", failed)


if __name__ == "__main__":
    main()
