"""One-case test of the tet-validated interior KNN graph.

Pipeline under test (what _process_case will eventually do):
  1. sample the interior pool exactly like _InteriorMixin._sample_pool
  2. Euclidean KNN (k=64) candidates on the pool
  3. validate each candidate edge against the tetrahedral connectivity
     (blood_data.pt "cells"): edge kept iff geodesic(i,j) <= GEO_RATIO * euclid(i,j)
  4. simulate a training get(): draw SAMPLE_RATIO of the pool, keep edges whose
     both endpoints survive, remap indices

Geodesics use a coarsening trick so we never run per-source Dijkstra on the
564k-node dense graph: one multi-source Dijkstra assigns every dense node to its
nearest pool seed, then each dense edge crossing two cells contributes a
pool-level edge weighted dist(u->seed_u) + w(u,v) + dist(v->seed_v). Shortest
paths on that coarse graph are an UPPER BOUND on the true dense geodesic, so
validation is conservative: it can over-reject borderline edges but can never
accept a wall-crossing shortcut.

Outputs (angioflow/data/graph_sanity/):
  <case>_kept.png      subsampled points + validated edges, 3 views
  <case>_rejected.png  rejected candidate edges in red, 3 views
"""
import heapq
import os
import time

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

CASE_DIR = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/B2o6VFmWVW_aneurysm1"
OUT_DIR = os.path.join(os.path.dirname(__file__), "graph_sanity")

K = 64                 # pool-level KNN candidates
POOL_RATIO = 0.16      # matches datasets_interior __main__
SAMPLE_RATIO = 0.25    # training-time draw from the pool
GEO_RATIO = 1.4        # keep edge iff geodesic <= GEO_RATIO * euclidean
LIMIT_FACTOR = 1.5     # Dijkstra truncation: LIMIT_FACTOR * (per-source 64th-NN dist)
NEAR_DECAY, FAR_FLOOR, MIN_WALL_DIST = 0.1, 0.25, 1e-4
MAX_RENDER_EDGES = 200_000


def sample_pool(wall_dist, pool_ratio, rng):
    # verbatim logic from _InteriorMixin._sample_pool (weighted-random baseline)
    valid = np.where(wall_dist >= MIN_WALL_DIST)[0]
    P = max(1, min(int(len(wall_dist) * pool_ratio), len(valid)))
    d = wall_dist[valid] / wall_dist.max()
    hist, edges = np.histogram(d, bins=32)
    inv_density = 1.0 / hist.clip(1)[np.clip(np.digitize(d, edges) - 1, 0, 31)]
    profile = FAR_FLOOR + (1.0 - FAR_FLOOR) * np.exp(-d / NEAR_DECAY)
    w = inv_density * profile
    return valid[rng.choice(len(valid), size=P, replace=False, p=w / w.sum())]


def _adaptive_fps(pts, r_local, P, rng, r_factor=3.0, cap_factor=8.0):
    """Density-weighted farthest-point sampling (blue noise with variable radius).

    Greedy FPS where the priority is dmin(i) / r_local(i): a point is picked when
    it is far from all chosen points *relative to its local target spacing*. Small
    r_local near the wall => that region tolerates tighter packing => stays denser,
    while every region is still evenly spread (no random clumping).

    Scales via a lazy max-heap + local ball-updates: selecting a point only
    shrinks dmin for points within R of it, so each step touches a local
    neighborhood, not all N. Heap entries carry the dmin they were pushed with; a
    popped entry whose dmin has since changed is stale and skipped.

    R = min(r_factor*dmin, cap) with cap ~ a few times the target spacing. Without
    the cap the early picks (dmin ~ domain size) each scan nearly all N; the cap
    is safe because a point farther than cap from every pick has true dmin > cap
    and so genuinely deserves high priority regardless of the exact value.
    """
    N = len(pts)
    tree = cKDTree(pts)
    # target spacing after downsampling M->P: candidate spacing * (M/P)^(1/3)
    cand_nn = tree.query(pts, k=2)[0][:, 1]
    s0 = np.median(cand_nn) * (N / P) ** (1.0 / 3.0)
    cap = cap_factor * s0

    dmin = np.full(N, np.inf)
    chosen = np.zeros(N, dtype=bool)
    selected = np.empty(P, dtype=np.int64)

    first = int(rng.integers(N))
    selected[0] = first
    chosen[first] = True
    dd = np.linalg.norm(pts - pts[first], axis=1)      # first update is global
    dmin = np.minimum(dmin, dd)
    dmin[first] = -1.0

    heap = [(-(dmin[i] / r_local[i]), i, dmin[i]) for i in range(N) if not chosen[i]]
    heapq.heapify(heap)

    n = 1
    while n < P and heap:
        neg_score, i, d_at_push = heapq.heappop(heap)
        if chosen[i] or d_at_push != dmin[i]:
            continue                                    # stale entry
        selected[n] = i
        chosen[i] = True
        n += 1
        R = min(r_factor * dmin[i], cap)
        dmin[i] = -1.0
        idx = np.asarray(tree.query_ball_point(pts[i], R), dtype=np.int64)
        if len(idx):
            nd = np.linalg.norm(pts[idx] - pts[i], axis=1)
            better = nd < dmin[idx]
            upd = idx[better]
            dmin[upd] = nd[better]
            for j, dj in zip(upd.tolist(), dmin[upd].tolist()):
                if not chosen[j]:
                    heapq.heappush(heap, (-(dj / r_local[j]), j, dj))
    return selected[:n]


def sample_pool_fps(int_pos, wall_dist, pool_ratio, rng):
    """Density-weighted FPS pool: near-wall boost preserved, no random clumping.

    Pre-samples a weighted-random candidate set (cheap, gives the near-wall
    density profile) then declumps it to the target count with adaptive FPS.
    r_local(d) ~ density(d)^(-1/3): the 3D spacing that realises the desired
    density, smaller near the wall.
    """
    valid = np.where(wall_dist >= MIN_WALL_DIST)[0]
    P = max(1, min(int(len(wall_dist) * pool_ratio), len(valid)))
    d = wall_dist[valid] / wall_dist.max()
    hist, edges = np.histogram(d, bins=32)
    inv_density = 1.0 / hist.clip(1)[np.clip(np.digitize(d, edges) - 1, 0, 31)]
    density = FAR_FLOOR + (1.0 - FAR_FLOOR) * np.exp(-d / NEAR_DECAY)
    w = inv_density * density

    # pre-reduce to ~oversample*P candidates so FPS runs on a smaller set
    M = min(len(valid), int(4.0 * P))
    cand = rng.choice(len(valid), size=M, replace=False, p=w / w.sum())
    r_local = density[cand] ** (-1.0 / 3.0)
    sel = _adaptive_fps(int_pos[valid[cand]], r_local, P, rng)
    return valid[cand[sel]]


def tet_edges(cells, pos):
    """Unique undirected tet edges + Euclidean lengths."""
    pairs = cells[:, [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]].reshape(-1, 2)
    pairs.sort(axis=1)
    key = pairs[:, 0] * len(pos) + pairs[:, 1]
    pairs = pairs[np.unique(key, return_index=True)[1]]
    w = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    return pairs, w


def coarse_pool_graph(pairs, w, pos, pool):
    """Pool-level graph whose shortest paths upper-bound dense geodesics."""
    N, P = len(pos), len(pool)
    A = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(N, N)).tocsr()
    # one multi-source Dijkstra: dist to nearest pool seed + which seed
    dist, _, seed = dijkstra(A, directed=False, indices=pool,
                             min_only=True, return_predecessors=True)
    rank = np.full(N, -1, dtype=np.int64)
    rank[pool] = np.arange(P)
    su, sv = rank[seed[pairs[:, 0]]], rank[seed[pairs[:, 1]]]
    cross = su != sv
    wc = dist[pairs[cross, 0]] + w[cross] + dist[pairs[cross, 1]]
    a = np.minimum(su[cross], sv[cross])
    b = np.maximum(su[cross], sv[cross])
    key, inv = np.unique(a * P + b, return_inverse=True)
    minw = np.full(len(key), np.inf)
    np.minimum.at(minw, inv, wc)
    ca, cb = key // P, key % P
    C = coo_matrix((np.concatenate([minw, minw]),
                    (np.concatenate([ca, cb]), np.concatenate([cb, ca]))),
                   shape=(P, P)).tocsr()
    return C


def validate_knn(C, pool_pos, k, chunk=256):
    """KNN candidates on the pool + geodesic upper bounds via the coarse graph."""
    P = len(pool_pos)
    nn_dist, nn_idx = cKDTree(pool_pos).query(pool_pos, k=k + 1)
    nn_dist, nn_idx = nn_dist[:, 1:], nn_idx[:, 1:]          # drop self
    limits = LIMIT_FACTOR * nn_dist[:, -1]

    geo = np.full((P, k), np.inf)
    for lo in range(0, P, chunk):
        hi = min(lo + chunk, P)
        D = dijkstra(C, directed=False, indices=np.arange(lo, hi),
                     limit=float(limits[lo:hi].max()))
        geo[lo:hi] = np.take_along_axis(D, nn_idx[lo:hi], axis=1)
        if lo % (chunk * 40) == 0:
            print(f"  dijkstra {lo}/{P}", flush=True)
    return nn_idx, nn_dist, geo


def exact_spot_check(pairs, w, pos, pool, edges, euclid, label, chunk=64):
    """Exact dense-graph geodesics for a sample of coarse-rejected edges.

    Measures how much the coarse upper bound over-rejects: reports the true
    geo/euclid ratio distribution and how many edges exact validation would keep.
    """
    N = len(pos)
    A = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(N, N)).tocsr()
    src_dense = pool[edges[:, 0]]
    dst_dense = pool[edges[:, 1]]
    ratios = np.full(len(edges), np.inf)
    for lo in range(0, len(edges), chunk):
        hi = min(lo + chunk, len(edges))
        lim = float((GEO_RATIO * 1.05) * euclid[lo:hi].max())
        D = dijkstra(A, directed=False, indices=src_dense[lo:hi], limit=lim)
        ratios[lo:hi] = D[np.arange(hi - lo), dst_dense[lo:hi]] / euclid[lo:hi]
    finite = ratios[np.isfinite(ratios)]
    kept_exact = (ratios <= GEO_RATIO).sum()
    print(f"[exact:{label}] n={len(edges)}  exact-valid {kept_exact} "
          f"({100 * kept_exact / len(edges):.1f}%)  "
          f"true ratio median {np.median(finite) if len(finite) else float('nan'):.2f}  "
          f"unreachable@limit {np.isinf(ratios).sum()}", flush=True)


def to_edge_index(nn_idx, mask):
    """Directed (i -> nn_idx[i,j]) pairs where mask, symmetrised + deduped."""
    P, k = nn_idx.shape
    src = np.repeat(np.arange(P), k)[mask.ravel()]
    dst = nn_idx.ravel()[mask.ravel()]
    e = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    e = e[np.unique(e[:, 0] * P + e[:, 1], return_index=True)[1]]
    return e


def _add_edges(pl, pts, e, color, lw, opacity, max_edges):
    if len(e) == 0:
        return
    if len(e) > max_edges:
        e = e[np.random.default_rng(0).choice(len(e), max_edges, replace=False)]
    lines = np.column_stack([np.full(len(e), 2), e]).ravel()
    pl.add_mesh(pv_polydata(pts, lines), color=color, line_width=lw,
                opacity=opacity)


def pv_polydata(pts, lines=None):
    import pyvista as pv
    return pv.PolyData(pts, lines=lines) if lines is not None else pv.PolyData(pts)


def render_slabs(path, pts, edges, title, slab_frac=0.04):
    """Thin-slab cross sections viewed face-on: the only way to see interior
    connectivity texture — full-cloud views saturate into a solid blob."""
    import pyvista as pv
    pv.start_xvfb()
    ext = pts.max(0) - pts.min(0)
    pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2700, 900))
    panels = []
    for axis in (0, 1, 2):
        thick = slab_frac * ext[axis]
        in_slab = np.abs(pts[:, axis] - np.median(pts[:, axis])) < thick
        emask = in_slab[edges[:, 0]] & in_slab[edges[:, 1]]
        panels.append((axis, in_slab, edges[emask]))
    for col, (axis, in_slab, e) in enumerate(panels):
        pl.subplot(0, col)
        pl.add_mesh(pv_polydata(pts[in_slab]), color="black", point_size=2.5,
                    render_points_as_spheres=True)
        # remap slab edges to slab-local point indices
        remap = np.cumsum(in_slab) - 1
        _add_edges(pl, pts[in_slab], remap[e], "steelblue", 1.0, 0.6,
                   MAX_RENDER_EDGES)
        pl.add_text(f"{title}  slab |{'xyz'[axis]}|<{slab_frac:.0%} "
                    f"({in_slab.sum()} pts, {len(e)} edges)",
                    position="upper_left", font_size=11)
        pl.camera_position = ["yz", "xz", "xy"][axis]
    pl.screenshot(path)
    pl.close()
    print(f"[ok] {path}")


def render_rejected(path, wall_mesh, pts, strong, borderline, title):
    import pyvista as pv
    pv.start_xvfb()
    pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2700, 900))
    for col, cam in enumerate(["xy", "xz", "yz"]):
        pl.subplot(0, col)
        pl.add_mesh(wall_mesh, color="white", opacity=0.06)
        _add_edges(pl, pts, borderline, "orange", 1.0, 0.15, 20_000)
        _add_edges(pl, pts, strong, "red", 2.0, 0.9, MAX_RENDER_EDGES)
        pl.add_text(f"{title}  [{cam}]", position="upper_left", font_size=11)
        pl.camera_position = cam
    pl.screenshot(path)
    pl.close()
    print(f"[ok] {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    case = os.path.basename(CASE_DIR.rstrip("/"))
    t0 = time.time()

    blood = torch.load(os.path.join(CASE_DIR, "blood_data.pt"),
                       weights_only=False, mmap=True)
    int_pos = blood["pos"].numpy().astype(np.float64)
    cells = blood["cells"].numpy()
    reg = torch.load(os.path.join(CASE_DIR, "wall_mesh_registration.pt"),
                     weights_only=False)
    wall_pos, wall_faces = reg["pos"].numpy().astype(np.float64), reg["faces"].numpy()
    print(f"[load] {len(int_pos)} interior pts, {len(cells)} tets  "
          f"({time.time()-t0:.1f}s)")

    wall_dist = cKDTree(wall_pos).query(int_pos)[0]
    pool_rand = sample_pool(wall_dist, POOL_RATIO, np.random.default_rng(0))
    print(f"[pool:rand] {len(pool_rand)} pts ({time.time()-t0:.1f}s)")
    pool = sample_pool_fps(int_pos, wall_dist, POOL_RATIO, np.random.default_rng(0))
    print(f"[pool:fps ] {len(pool)} pts ({time.time()-t0:.1f}s)")

    # uniformity metric: coefficient of variation of nearest-neighbour spacing.
    # lower = more even (blue-noise ~0.1-0.2; random/clumped ~0.5+).
    for tag, pl_ in [("rand", pool_rand), ("fps", pool)]:
        nnd = cKDTree(int_pos[pl_]).query(int_pos[pl_], k=2)[0][:, 1]
        print(f"[uniformity:{tag}] nn-dist mean {nnd.mean():.2e}  "
              f"CV {nnd.std() / nnd.mean():.3f}")
    # side-by-side pool density slabs (points only) to eyeball clumping
    render_slabs(os.path.join(OUT_DIR, f"{case}_pool_rand_slabs.png"),
                 int_pos[pool_rand], np.empty((0, 2), np.int64),
                 f"{case}  POOL weighted-random")
    render_slabs(os.path.join(OUT_DIR, f"{case}_pool_fps_slabs.png"),
                 int_pos[pool], np.empty((0, 2), np.int64),
                 f"{case}  POOL density-weighted FPS")

    pairs, w = tet_edges(cells, int_pos)
    print(f"[tet]  {len(pairs)} unique dense edges ({time.time()-t0:.1f}s)")

    C = coarse_pool_graph(pairs, w, int_pos, pool)
    print(f"[coarse] {C.nnz // 2} pool-level edges ({time.time()-t0:.1f}s)")

    pool_pos = int_pos[pool]
    nn_idx, nn_dist, geo = validate_knn(C, pool_pos, K)
    ratio = geo / nn_dist
    valid = ratio <= GEO_RATIO
    borderline_m = (ratio > GEO_RATIO) & (ratio <= 2.0)
    strong_m = ~valid & ~borderline_m           # ratio > 2 or unreachable
    print(f"[validate] kept {valid.sum()}/{valid.size} directed candidates "
          f"({100 * valid.mean():.2f}%) | borderline(1.4-2.0) "
          f"{borderline_m.sum()} | strong(>2/unreachable) {strong_m.sum()} "
          f"(inf: {np.isinf(ratio).sum()})  ({time.time()-t0:.1f}s)")
    finite = ratio[np.isfinite(ratio)]
    print(f"[ratio] p50 {np.percentile(finite, 50):.2f}  "
          f"p90 {np.percentile(finite, 90):.2f}  p99 {np.percentile(finite, 99):.2f}")

    kept = to_edge_index(nn_idx, valid)
    borderline = to_edge_index(nn_idx, borderline_m)
    strong = to_edge_index(nn_idx, strong_m)
    print(f"[edges] kept {len(kept)}  borderline {len(borderline)}  "
          f"strong {len(strong)} (undirected)")

    # exact dense-graph geodesics on a sample of each rejected class: how much
    # does the coarse upper bound over-reject?
    rng0 = np.random.default_rng(0)
    for e, label in [(borderline, "borderline"), (strong, "strong")]:
        if len(e) == 0:
            continue
        s = e[rng0.choice(len(e), min(200, len(e)), replace=False)]
        eu = np.linalg.norm(pool_pos[s[:, 0]] - pool_pos[s[:, 1]], axis=1)
        exact_spot_check(pairs, w, int_pos, pool, s, eu, label)

    # simulate training-time subsampling: draw SAMPLE_RATIO of the pool
    rng = np.random.default_rng(1)
    P = len(pool)
    sub = rng.permutation(P)[: int(P * SAMPLE_RATIO)]
    sampled = np.zeros(P, dtype=bool)
    sampled[sub] = True
    remap = np.full(P, -1, dtype=np.int64)
    remap[sub] = np.arange(len(sub))
    emask = sampled[kept[:, 0]] & sampled[kept[:, 1]]
    kept_sub = remap[kept[emask]]
    deg = np.bincount(kept_sub.ravel(), minlength=len(sub))
    print(f"[subsample] {len(sub)} pts, {len(kept_sub)} edges | degree "
          f"min {deg.min()} / mean {deg.mean():.1f} / max {deg.max()} | "
          f"deg<6: {(deg < 6).sum()} pts ({100 * (deg < 6).mean():.2f}%)")

    import pyvista as pv
    wall_mesh = pv.PolyData(
        wall_pos, np.column_stack([np.full(len(wall_faces), 3),
                                   wall_faces]).ravel())
    render_slabs(os.path.join(OUT_DIR, f"{case}_kept_slabs.png"),
                 pool_pos[sub], kept_sub,
                 f"{case}  training-subsampled validated graph "
                 f"(mean deg {deg.mean():.1f})")
    render_rejected(os.path.join(OUT_DIR, f"{case}_rejected.png"), wall_mesh,
                    pool_pos, strong, borderline,
                    f"{case}  rejected: strong red ({len(strong)}), "
                    f"borderline orange ({len(borderline)}, subsampled)")

    # ParaView exports for manual inspection: the training-subsampled graph as
    # line cells (+ degree as point data), and the wall surface for context
    sub_pts = pool_pos[sub]
    lines = np.column_stack([np.full(len(kept_sub), 2), kept_sub]).ravel()
    graph_poly = pv_polydata(sub_pts, lines)
    graph_poly["degree"] = deg.astype(np.float32)
    graph_poly.save(os.path.join(OUT_DIR, f"{case}_graph_sub.vtp"))
    wall_mesh.save(os.path.join(OUT_DIR, f"{case}_wall.vtp"))
    rej_lines = np.column_stack([
        np.full(len(strong) + len(borderline), 2),
        np.concatenate([strong, borderline]) if len(strong) else borderline,
    ]).ravel()
    rej_poly = pv_polydata(pool_pos, rej_lines)
    rej_poly["strong"] = np.concatenate([
        np.ones(len(strong), np.float32), np.zeros(len(borderline), np.float32)])
    rej_poly.save(os.path.join(OUT_DIR, f"{case}_rejected.vtp"))
    print(f"[ok] paraview exports -> {OUT_DIR}/{case}_{{graph_sub,wall,rejected}}.vtp")
    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()


"""
conda activate new
python -m angioflow.data.test_interior_graph
"""
