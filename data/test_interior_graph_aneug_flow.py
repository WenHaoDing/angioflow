"""One-case test of the surface-validated interior KNN graph (AneuG-Flow dataset).

AneuG-Flow has NO tetrahedral connectivity in blood_data.pt (unlike AngioFlow), so the tet-edge
geodesic validator can't be used. Instead we validate KNN candidates directly
against the CFD surface mesh (shape_remeshed_cfd.obj) by segment-mesh
intersection: an edge is invalid iff the open segment between its two interior
endpoints pierces a wall triangle (i.e. it leaves the lumen — the dome-to-adjacent
-artery shortcut we want to trim). Ray casting needs neither a watertight mesh
nor consistent normals, which matters because this mesh is neither.

Two unit/geometry facts for AneuG-Flow (handled below):
  * interior points are in metres, the .obj is in millimetres -> scale mesh x1e-3
  * geometry is static (interior coords identical across all 80 frames)

Pipeline mirrors the AngioFlow test so the two can be compared and later merged:
  1. FPS density-weighted pool (reused from test_interior_graph)
  2. Euclidean KNN (k=64) candidates on the pool
  3. validate by embree segment-mesh intersection
  4. simulate a training get(): draw SAMPLE_RATIO, keep both-endpoint edges

Outputs (angioflow/data/graph_sanity/):
  <case>_aneug_flow_pool_fps_slabs.png   FPS pool density (points only)
  <case>_aneug_flow_kept_slabs.png       training-subsampled validated graph
  <case>_aneug_flow_rejected.png         rejected (wall-crossing) edges over the mesh
  <case>_aneug_flow_{graph_sub,wall,rejected}.vtp  ParaView exports
"""
import os
import time

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import angioflow.data.test_interior_graph as tig
from angioflow.data.test_interior_graph import (
    sample_pool, sample_pool_fps, render_slabs, render_rejected,
    pv_polydata, OUT_DIR, K, POOL_RATIO, SAMPLE_RATIO,
)

CASE_DIR = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data/stable_246"
MESH_SCALE = 1e-3          # .obj is mm, interior points are metres
EPS_FRAC = 0.02            # shrink each segment end before casting (avoid grazing near-wall endpoints)


def load_interior_pos(blood):
    """Frame-0 interior coordinates (static across frames in AneuG-Flow)."""
    x = blood["x_coordinate"][0, :, 0].numpy().astype(np.float64)
    y = blood["y_coordinate"][0, :, 0].numpy().astype(np.float64)
    z = blood["z_coordinate"][0, :, 0].numpy().astype(np.float64)
    return np.stack([x, y, z], axis=1)


def knn_candidates(pool_pos, k):
    """Unique undirected KNN candidate edges on the pool."""
    _, nn_idx = cKDTree(pool_pos).query(pool_pos, k=k + 1)
    nn_idx = nn_idx[:, 1:]                                   # drop self
    P = len(pool_pos)
    src = np.repeat(np.arange(P), k)
    dst = nn_idx.ravel()
    e = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    return e[np.unique(e[:, 0] * P + e[:, 1], return_index=True)[1]]


def validate_segments(pool_pos, edges, mesh, eps_frac=EPS_FRAC):
    """Invalid iff the open segment i->j pierces the wall mesh.

    Cast one ray per edge from a slightly-inset start toward j; a first hit closer
    than the (inset) segment length means the segment left the lumen. Endpoints
    are inset by eps_frac*len so a near-wall endpoint sitting almost on the surface
    doesn't self-trigger.
    """
    o = pool_pos[edges[:, 0]]
    d = pool_pos[edges[:, 1]] - o
    L = np.linalg.norm(d, axis=1)
    dirs = d / L[:, None]
    eps = eps_frac * L
    origins = o + dirs * eps[:, None]
    seg = L - 2 * eps                                        # castable length

    locs, idx_ray, _ = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=False)                 # nearest hit per ray
    hit_dist = np.linalg.norm(locs - origins[idx_ray], axis=1)
    invalid = np.zeros(len(edges), dtype=bool)
    crossing = hit_dist < seg[idx_ray]
    invalid[idx_ray[crossing]] = True
    return invalid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    case = os.path.basename(CASE_DIR.rstrip("/"))
    t0 = time.time()

    blood = torch.load(os.path.join(CASE_DIR, "blood_data.pt"),
                       weights_only=False, mmap=True)
    int_pos = load_interior_pos(blood)
    mesh = trimesh.load(os.path.join(CASE_DIR, "shape_remeshed_cfd.obj"),
                        process=False)
    mesh.apply_scale(MESH_SCALE)                             # mm -> m
    print(f"[load] {len(int_pos)} interior pts, mesh {len(mesh.faces)} faces  "
          f"({time.time()-t0:.1f}s)")

    # wall distance from the surface (nearest mesh vertex — fine for weighting)
    wall_dist = cKDTree(mesh.vertices).query(int_pos)[0]

    pool_rand = sample_pool(wall_dist, POOL_RATIO, np.random.default_rng(0))
    pool = sample_pool_fps(int_pos, wall_dist, POOL_RATIO, np.random.default_rng(0))
    print(f"[pool] rand {len(pool_rand)} / fps {len(pool)} pts "
          f"({time.time()-t0:.1f}s)")
    for tag, pl_ in [("rand", pool_rand), ("fps", pool)]:
        nnd = cKDTree(int_pos[pl_]).query(int_pos[pl_], k=2)[0][:, 1]
        print(f"[uniformity:{tag}] nn-dist mean {nnd.mean():.2e}  "
              f"CV {nnd.std() / nnd.mean():.3f}")
    render_slabs(os.path.join(OUT_DIR, f"{case}_aneug_flow_pool_fps_slabs.png"),
                 int_pos[pool], np.empty((0, 2), np.int64),
                 f"{case}  AneuG-Flow POOL density-weighted FPS")

    pool_pos = int_pos[pool]
    edges = knn_candidates(pool_pos, K)
    print(f"[knn] {len(edges)} unique candidate edges ({time.time()-t0:.1f}s)")

    invalid = validate_segments(pool_pos, edges, mesh)
    kept = edges[~invalid]
    rejected = edges[invalid]
    print(f"[validate] kept {len(kept)}/{len(edges)} "
          f"({100*(~invalid).mean():.2f}%)  rejected {len(rejected)}  "
          f"({time.time()-t0:.1f}s)")

    # training-time subsample: draw SAMPLE_RATIO of the pool, keep both-in edges
    rng = np.random.default_rng(1)
    P = len(pool)
    sub = rng.permutation(P)[: int(P * SAMPLE_RATIO)]
    sampled = np.zeros(P, dtype=bool); sampled[sub] = True
    remap = np.full(P, -1, dtype=np.int64); remap[sub] = np.arange(len(sub))
    emask = sampled[kept[:, 0]] & sampled[kept[:, 1]]
    kept_sub = remap[kept[emask]]
    deg = np.bincount(kept_sub.ravel(), minlength=len(sub))
    print(f"[subsample] {len(sub)} pts, {len(kept_sub)} edges | degree "
          f"min {deg.min()} / mean {deg.mean():.1f} / max {deg.max()} | "
          f"deg<6: {(deg < 6).sum()} ({100*(deg < 6).mean():.2f}%)")

    import pyvista as pv
    faces = mesh.faces
    wall_mesh = pv.PolyData(
        mesh.vertices, np.column_stack([np.full(len(faces), 3), faces]).ravel())
    render_slabs(os.path.join(OUT_DIR, f"{case}_aneug_flow_kept_slabs.png"),
                 pool_pos[sub], kept_sub,
                 f"{case}  AneuG-Flow training-subsampled validated graph "
                 f"(mean deg {deg.mean():.1f})")
    render_rejected(os.path.join(OUT_DIR, f"{case}_aneug_flow_rejected.png"), wall_mesh,
                    pool_pos, rejected, np.empty((0, 2), np.int64),
                    f"{case}  AneuG-Flow rejected wall-crossing edges ({len(rejected)})")

    # ParaView exports
    sub_pts = pool_pos[sub]
    lines = np.column_stack([np.full(len(kept_sub), 2), kept_sub]).ravel()
    gp = pv_polydata(sub_pts, lines); gp["degree"] = deg.astype(np.float32)
    gp.save(os.path.join(OUT_DIR, f"{case}_aneug_flow_graph_sub.vtp"))
    wall_mesh.save(os.path.join(OUT_DIR, f"{case}_aneug_flow_wall.vtp"))
    if len(rejected):
        rl = np.column_stack([np.full(len(rejected), 2), rejected]).ravel()
        pv_polydata(pool_pos, rl).save(
            os.path.join(OUT_DIR, f"{case}_aneug_flow_rejected.vtp"))
    print(f"[ok] exports -> {OUT_DIR}/{case}_aneug_flow_*  ({time.time()-t0:.1f}s)")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()


"""
conda activate new
python -m angioflow.data.test_interior_graph_aneug_flow
"""
