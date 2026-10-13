"""Wall dataset variants that also carry an interior flow-field point cloud.

Preprocessing (per case, in _process_case):
  * samples a pool of interior points via density-weighted FPS — denser near the
    wall (whose velocity gradients determine WSS) but evenly spread (blue noise),
  * builds a wall-respecting KNN graph over the pool: candidate edges are pruned
    so none crosses the vessel wall — AngioFlow via the tetrahedral connectivity in
    blood_data.pt (geodesic ≈ straight-line), AneuG-Flow by ray-casting each segment
    against the full CFD surface (shape_remeshed_cfd.obj),
  * saves per-frame velocity [P,3], pressure [P,1], the velocity-gradient tensor
    grad_v [P,9] (dvi/dxj, row-major) and pressure gradient grad_p [P,3].

get() returns a `(wall, interior)` tuple of two Data objects — wall node count N
and interior pool count P vary per case, so each stream gets its own batch vector.
PyG's DataLoader collates tuples element-wise: a batch arrives as
`(wall_batch, interior_batch)`. The interior Data uses `pos` as its node attribute
(plus edge_index, vel, press, grad_v, grad_p, wall_dist) and is returned in FULL —
the per-epoch random subsample + induced graph is an InteriorSubsampleGraph
transform, and CenterAndRandomRotate rotates the interior vectors/tensors with the
wall's frame. `dataset.transform` receives the pair, so transforms must accept it.
Cases without blood_data.pt are excluded.
"""
from __future__ import annotations

import heapq
import os
from functools import lru_cache

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from angioflow.data.datasets import AngioFlowWallDataset
from angioflow.data.datasets_t0 import AngioFlowWallDatasetT0

VEL_KEYS = ["x_velocity", "y_velocity", "z_velocity"]
PRESS_KEY = "absolute_pressure"
# velocity-gradient tensor, row-major dvi/dxj: [dvx/dx, dvx/dy, dvx/dz, dvy/dx, ...]
VGRAD_KEYS = [f"d{c}_velocity_d{x}" for c in "xyz" for x in "xyz"]
PGRAD_KEYS = ["dp_dx", "dp_dy", "dp_dz"]


# ----------------------------------------------------------------------------
# density-weighted farthest-point sampling (blue noise with a wall-distance
# dependent target spacing) — see angioflow/data/test_interior_graph.py for the
# prototype/derivation.
# ----------------------------------------------------------------------------
def _adaptive_fps(pts, r_local, P, rng, r_factor=3.0, cap_factor=8.0):
    """Greedy FPS with priority dmin(i)/r_local(i): points stay far apart relative
    to a local target spacing (small near the wall => denser there, evenly spread
    everywhere). Lazy heap + capped local ball-updates keep it near O(N log N)."""
    N = len(pts)
    tree = cKDTree(pts)
    cand_nn = tree.query(pts, k=2)[0][:, 1]
    s0 = np.median(cand_nn) * (N / P) ** (1.0 / 3.0)
    cap = cap_factor * s0

    dmin = np.full(N, np.inf)
    chosen = np.zeros(N, dtype=bool)
    selected = np.empty(P, dtype=np.int64)

    first = int(rng.integers(N))
    selected[0] = first
    chosen[first] = True
    dmin = np.minimum(dmin, np.linalg.norm(pts - pts[first], axis=1))
    dmin[first] = -1.0

    heap = [(-(dmin[i] / r_local[i]), i, dmin[i]) for i in range(N) if not chosen[i]]
    heapq.heapify(heap)

    n = 1
    while n < P and heap:
        _, i, d_at_push = heapq.heappop(heap)
        if chosen[i] or d_at_push != dmin[i]:
            continue                                     # stale entry
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


# ----------------------------------------------------------------------------
# validated pool graph — candidate KNN edges pruned so none crosses the wall.
# AngioFlow uses the tetrahedral connectivity in blood_data.pt (geodesic-through-the-mesh
# must match the straight-line distance). AneuG-Flow has no tets, so it ray-casts each
# candidate segment against the full CFD surface (shape_remeshed_cfd.obj). Both
# return a symmetric directed edge_index [2, E].
# ----------------------------------------------------------------------------
def _symmetric_edge_index(undirected):
    """[E,2] unique undirected -> [2, 2E] directed both ways (torch long)."""
    e = np.concatenate([undirected, undirected[:, ::-1]], axis=0)
    return torch.from_numpy(np.ascontiguousarray(e.T)).long()


def _knn_undirected(pool_pos, k):
    _, nn = cKDTree(pool_pos).query(pool_pos, k=k + 1)
    nn = nn[:, 1:]                                        # drop self
    P = len(pool_pos)
    src = np.repeat(np.arange(P), k)
    dst = nn.ravel()
    e = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    e = e[np.unique(e[:, 0] * P + e[:, 1], return_index=True)[1]]
    return e


def _validate_tet_geodesic(int_pos, cells, pool, k, geo_ratio, limit_factor):
    """AngioFlow: keep KNN edge iff geodesic(i,j) on the tet graph <= geo_ratio*euclid.

    Geodesics use a coarsening: one multi-source Dijkstra assigns every dense node
    to its nearest pool seed; dense edges crossing two cells give a pool-level
    graph whose shortest paths upper-bound the true geodesic (so validation is
    conservative — it never accepts a wall-crossing shortcut)."""
    pairs = cells[:, [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3]].reshape(-1, 2)
    pairs.sort(axis=1)
    pairs = pairs[np.unique(pairs[:, 0] * len(int_pos) + pairs[:, 1],
                            return_index=True)[1]]
    w = np.linalg.norm(int_pos[pairs[:, 0]] - int_pos[pairs[:, 1]], axis=1)

    N, P = len(int_pos), len(pool)
    A = coo_matrix((w, (pairs[:, 0], pairs[:, 1])), shape=(N, N)).tocsr()
    dist, _, seed = dijkstra(A, directed=False, indices=pool,
                             min_only=True, return_predecessors=True)
    rank = np.full(N, -1, dtype=np.int64)
    rank[pool] = np.arange(P)
    su, sv = rank[seed[pairs[:, 0]]], rank[seed[pairs[:, 1]]]
    cross = su != sv
    wc = dist[pairs[cross, 0]] + w[cross] + dist[pairs[cross, 1]]
    a = np.minimum(su[cross], sv[cross]); b = np.maximum(su[cross], sv[cross])
    key, inv = np.unique(a * P + b, return_inverse=True)
    minw = np.full(len(key), np.inf); np.minimum.at(minw, inv, wc)
    ca, cb = key // P, key % P
    C = coo_matrix((np.concatenate([minw, minw]),
                    (np.concatenate([ca, cb]), np.concatenate([cb, ca]))),
                   shape=(P, P)).tocsr()

    pool_pos = int_pos[pool]
    edges = _knn_undirected(pool_pos, k)
    nn_dist = np.linalg.norm(pool_pos[edges[:, 0]] - pool_pos[edges[:, 1]], axis=1)
    limits = limit_factor * cKDTree(pool_pos).query(pool_pos, k=k + 1)[0][:, -1]
    keep = np.zeros(len(edges), dtype=bool)
    chunk = 256
    for lo in range(0, P, chunk):
        hi = min(lo + chunk, P)
        D = dijkstra(C, directed=False, indices=np.arange(lo, hi),
                     limit=float(limits[lo:hi].max()))
        m = (edges[:, 0] >= lo) & (edges[:, 0] < hi)
        ei = edges[m]
        geo = D[ei[:, 0] - lo, ei[:, 1]]
        keep[m] = geo <= geo_ratio * nn_dist[m]
    return _symmetric_edge_index(edges[keep])


def _validate_surface(pool_pos, mesh, k, eps_frac):
    """AneuG-Flow: keep KNN edge iff the open segment i->j does not pierce the wall mesh
    (ray cast, embree). `mesh` is the full CFD surface already in the interior's
    coordinate frame. Endpoints are inset by eps_frac*len so a near-wall endpoint
    sitting almost on the surface does not self-trigger."""
    edges = _knn_undirected(pool_pos, k)
    o = pool_pos[edges[:, 0]]
    d = pool_pos[edges[:, 1]] - o
    L = np.linalg.norm(d, axis=1)
    dirs = d / L[:, None]
    eps = eps_frac * L
    origins = o + dirs * eps[:, None]
    seg = L - 2 * eps
    locs, idx_ray, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    hit = np.linalg.norm(locs - origins[idx_ray], axis=1)
    invalid = np.zeros(len(edges), dtype=bool)
    invalid[idx_ray[hit < seg[idx_ray]]] = True
    return _symmetric_edge_index(edges[~invalid])


class _InteriorMixin:
    def __init__(self, *args, pool_ratio=0.25, sample_ratio=0.5,
                 near_decay=0.1, far_floor=0.25, min_wall_dist=1e-4,
                 graph_k=64, geo_ratio=1.4, geo_limit_factor=1.5,
                 ray_eps_frac=0.02, mesh_scale=1e-3,
                 load_edge_index=True, load_grad=True,
                 interior_dir=None, sanity_first=50, **kwargs):
        self.pool_ratio = pool_ratio        # fraction of interior mesh nodes kept in the on-disk pool
        self.sample_ratio = sample_ratio    # fraction of the pool drawn per get() call
        self.near_decay = near_decay        # near-wall density boost length scale (fraction of max wall dist)
        self.far_floor = far_floor          # uniform floor of the density profile — coverage everywhere
        self.min_wall_dist = min_wall_dist  # drop points closer than this to the wall (mesh units = m; 1e-4 = 0.1 mm)
        self.graph_k = graph_k              # pool-level KNN candidate degree (before validation)
        self.geo_ratio = geo_ratio          # AngioFlow: keep edge iff geodesic <= geo_ratio * euclid
        self.geo_limit_factor = geo_limit_factor  # AngioFlow: Dijkstra truncation vs per-source k-th NN dist
        self.ray_eps_frac = ray_eps_frac    # AneuG-Flow: shrink each segment end before ray casting
        self.mesh_scale = mesh_scale        # AneuG-Flow: shape_remeshed_cfd.obj units -> interior units (mm -> m)
        # get() attaches these only when requested. The pool graph edge_index is
        # the bulk of the static file (~76 MB) and the velocity/pressure GRADIENTS
        # dominate the frame files; a baseline that predicts vel/press uses none of
        # them, so skipping them (with mmap loads) keeps those bytes off disk-read,
        # RAM cache, and the GPU. Turn back on for a gradient-supervised variant.
        self.load_edge_index = load_edge_index
        self.load_grad = load_grad
        self._interior_dir = interior_dir
        self.sanity_first = sanity_first    # render sanity checks for the first N processed cases PER VERSION
        self._sanity_done = {}              # per-version counter: {version: n_rendered}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _interior_pos(blood, version):
        """Raw interior point positions in the blood/wall coordinate frame.

        AngioFlow stores a static `pos`; AneuG-Flow stores per-frame coordinates that are
        identical across frames (static geometry), so frame 0 is representative.
        """
        if version == "angioflow":
            return blood["pos"].numpy().astype(np.float64)
        return np.stack([blood[f"{a}_coordinate"][0, :, 0].numpy() for a in "xyz"],
                        axis=1).astype(np.float64)

    def _build_pool_graph(self, version, int_pos, pool, blood, case_dir):
        """Tet-geodesic (AngioFlow) or surface ray-cast (AneuG-Flow) validated pool graph.

        Operates in the raw (un-centred) coordinate frame: pool positions and the
        validating geometry (tet cells / surface mesh) all live there.
        """
        if version == "angioflow":
            cells = blood["cells"].numpy()
            return _validate_tet_geodesic(int_pos, cells, pool, self.graph_k,
                                          self.geo_ratio, self.geo_limit_factor)
        # AneuG-Flow: full CFD surface — NOT the (cropped) registration mesh
        import trimesh
        obj = os.path.join(case_dir, "shape_remeshed_cfd.obj")
        mesh = trimesh.load(obj, process=False)
        mesh.apply_scale(self.mesh_scale)                # mm -> m (interior frame)
        return _validate_surface(int_pos[pool], mesh, self.graph_k, self.ray_eps_frac)

    @property
    def interior_dir(self):
        # interior data lives in its own folder — easy to inspect / wipe / rsync
        return self._interior_dir or os.path.join(self.save_dir, "interior")

    def _interior_static_path(self, version, cid):
        return os.path.join(self.interior_dir, f"interior_static_{version}_{cid}.pt")

    def _interior_frame_path(self, version, cid, t):
        return os.path.join(self.interior_dir, f"interior_{version}_{cid}_{t}.pt")

    def _case_complete(self, version, cid):
        return super()._case_complete(version, cid) and \
               os.path.exists(self._interior_static_path(version, cid)) and \
               all(os.path.exists(self._interior_frame_path(version, cid, t))
                   for t in range(self.n_frames))

    @property
    def processed_file_names(self):
        # absolute paths: interior_dir may live outside save_dir, and
        # os.path.join(processed_dir, <absolute>) resolves to the absolute path
        extra = [self._interior_static_path(v, cid) for (v, cid) in self.case_keys]
        extra += [self._interior_frame_path(v, cid, t)
                  for (v, cid) in self.case_keys for t in range(self.n_frames)]
        return super().processed_file_names + extra

    def _sample_pool(self, int_pos, wall_dist, rng):
        """Density-weighted farthest-point pool: denser near the wall, evenly
        spread (blue noise) everywhere — no random clumping.

        A weighted-random pre-sample gives the desired near-wall density profile
        (1/mesh-density neutralises the boundary-layer refinement; a smooth
        near_decay boost over a far_floor keeps mid-lumen coverage), then adaptive
        FPS thins it to the target count with a local target spacing
        r_local ~ density^(-1/3) so the near-wall boost survives declumping.
        """
        # drop the no-slip band: points closer than min_wall_dist carry ~zero
        # velocity and no usable signal
        valid = np.where(wall_dist >= self.min_wall_dist)[0]
        P = max(1, min(int(len(wall_dist) * self.pool_ratio), len(valid)))
        d = wall_dist[valid] / wall_dist.max()
        hist, edges = np.histogram(d, bins=32)
        inv_density = 1.0 / hist.clip(1)[np.clip(np.digitize(d, edges) - 1, 0, 31)]
        density = self.far_floor + (1.0 - self.far_floor) * np.exp(-d / self.near_decay)
        w = inv_density * density

        # pre-reduce to ~4P candidates (cheap, carries the density profile), then
        # declump with adaptive FPS on a local spacing ~ density^(-1/3)
        M = min(len(valid), int(4.0 * P))
        cand = rng.choice(len(valid), size=M, replace=False, p=w / w.sum())
        r_local = density[cand] ** (-1.0 / 3.0)
        sel = _adaptive_fps(int_pos[valid[cand]], r_local, P, rng)
        return valid[cand[sel]]

    def _process_case(self, version, cid, case_dir, reg_path):
        blood_path = os.path.join(case_dir, "blood_data.pt")
        if not os.path.exists(blood_path):
            raise FileNotFoundError("no blood_data.pt")
        # reuse existing wall files (static/wss) — only the interior part is new
        if not super()._case_complete(version, cid):
            super()._process_case(version, cid, case_dir, reg_path)

        blood = torch.load(blood_path, weights_only=False)
        # reg["pos"] is the (cropped) registered wall — used ONLY for the centring
        # offset, which the parent applies to the wall static; the interior must
        # replicate it so the two streams share a coordinate frame. It is NOT the
        # validating surface (that is the full CFD mesh, per-version, below).
        wall_pos = torch.load(reg_path, weights_only=False)["pos"].numpy().astype(np.float64)
        int_pos = self._interior_pos(blood, version)

        wall_dist = cKDTree(wall_pos).query(int_pos)[0]
        rng = np.random.default_rng(0)
        pool = self._sample_pool(int_pos, wall_dist, rng)
        os.makedirs(self.interior_dir, exist_ok=True)

        # validate the pool KNN graph in the raw frame, before centring
        edge_index = self._build_pool_graph(version, int_pos, pool, blood, case_dir)

        # parent centers wall pos on the un-decimated mesh mean — same offset here
        center = wall_pos.mean(0, keepdims=True)

        torch.save({
            "pos":        torch.from_numpy(int_pos[pool] - center).float(),  # [P, 3]
            "wall_dist":  torch.from_numpy(wall_dist[pool]).float(),         # [P]
            "edge_index": edge_index,                                        # [2, E] validated
        }, self._interior_static_path(version, cid))

        vel = torch.cat([blood[k][:, pool] for k in VEL_KEYS], dim=-1)     # [T, P, 3]
        press = blood[PRESS_KEY][:, pool]                                  # [T, P, 1]
        grad_v = torch.cat([blood[k][:, pool] for k in VGRAD_KEYS], dim=-1)  # [T, P, 9]
        grad_p = torch.cat([blood[k][:, pool] for k in PGRAD_KEYS], dim=-1)  # [T, P, 3]
        for t in range(self.n_frames):
            torch.save({"vel": vel[t].clone(), "press": press[t].clone(),
                        "grad_v": grad_v[t].clone(), "grad_p": grad_p[t].clone()},
                       self._interior_frame_path(version, cid, t))
        print(f"[ok] {version}/{cid}: interior pool {len(pool)}/{len(int_pos)} pts, "
              f"{edge_index.shape[1] // 2} validated edges "
              f"(decay={self.near_decay:.2f}, floor={self.far_floor:.2f})")

        # render sanity checks for the first few cases PER VERSION so sampling/
        # velocity can be examined (and processing aborted) before the whole
        # dataset is built — per-version so AneuG-Flow and AngioFlow both get coverage even
        # though all AneuG-Flow cases are processed before any AngioFlow case
        if self._sanity_done.get(version, 0) < self.sanity_first:
            interior_sanity(self.save_dir, self.interior_dir, version, cid,
                            n_frames=self.n_frames)
            self._sanity_done[version] = self._sanity_done.get(version, 0) + 1

    @lru_cache(maxsize=1024)
    def _load_interior_static(self, version, cid):
        # mmap: only the tensors get() actually touches (pos, wall_dist — and
        # edge_index if load_edge_index) are paged in; the ~76 MB edge_index stays
        # unread when unused, so the cache footprint is a few MB/case not ~76.
        return torch.load(self._interior_static_path(version, cid),
                          weights_only=False, mmap=True)

    def get(self, idx):
        wall = super().get(idx)
        version, cid, t = self._index[idx]
        s = self._load_interior_static(version, cid)
        frame = torch.load(self._interior_frame_path(version, cid, t),
                           weights_only=False, mmap=True)
        # Return the FULL validated pool; a downstream transform draws the per-epoch
        # random subsample. edge_index / grad_v / grad_p are attached only when
        # requested (see load_edge_index / load_grad) — with mmap, skipping them
        # means their bytes are never read from disk.
        interior = Data(
            pos=s["pos"],                    # [P, 3] — node attribute, gets its own batch vector
            vel=frame["vel"],                # [P, 3]
            press=frame["press"],            # [P, 1]
            wall_dist=s["wall_dist"],        # [P]
            t=torch.tensor([t]),
            version=version,
            case_id=cid,
        )
        if self.load_edge_index:
            interior.edge_index = s["edge_index"]   # [2, E] tet/surface-validated pool graph
        if self.load_grad:
            interior.grad_v = frame["grad_v"]       # [P, 9] dvi/dxj, row-major
            interior.grad_p = frame["grad_p"]       # [P, 3] dp/dxj
        return wall, interior


class AngioFlowWallInteriorDataset(_InteriorMixin, AngioFlowWallDataset):
    """All-frames wall dataset + interior point cloud.

    One sample per (case, t): `(wall Data, interior Data)` at the same frame t.
    Only cases with blood_data.pt are kept. get() returns the FULL validated pool;
    apply InteriorSubsampleGraph as a transform for the per-epoch subsample + graph.
    Extra kwargs: pool_ratio, near_decay, far_floor, graph_k, geo_ratio.
    """


class AngioFlowWallInteriorDatasetT0(_InteriorMixin, AngioFlowWallDatasetT0):
    """t=0 variant: one sample per case, interior fields taken at frame 0.

    Same constructor args as AngioFlowWallDatasetT0 plus the interior kwargs
    from _InteriorMixin. get() returns `(wall Data, interior Data)`.
    """


def interior_sanity(save_dir, interior_dir, version, cid, n_frames=80, fps=10,
                    max_speed=None, point_size=4.0, point_opacity=0.35, redo=False,
                    slab_frac=0.04, max_render_edges=150_000):
    """Two per-case sanity renders, written to interior_dir/sanity:

    1. graph_{version}_{cid}.png — the validated pool graph as thin slab cross
       sections (points + edges, one panel per axis). Eyeball check: connectivity
       stays inside the lumen and no edge bridges anatomically separate branches.
    2. vel_{version}_{cid}.gif — interior points coloured by |v| per frame over a
       translucent wall mesh. Eyeball check: fast core, ~0 at the wall (no-slip),
       pulsating with the waveform.
    """
    import pyvista as pv
    pv.start_xvfb()

    sanity_dir = os.path.join(interior_dir, "sanity")
    os.makedirs(sanity_dir, exist_ok=True)
    graph_path = os.path.join(sanity_dir, f"graph_{version}_{cid}.png")
    gif_path = os.path.join(sanity_dir, f"vel_{version}_{cid}.gif")
    if not redo and os.path.exists(graph_path) and os.path.exists(gif_path):
        print(f"[skip] {version}/{cid}: sanity renders exist")
        return

    s = torch.load(os.path.join(interior_dir, f"interior_static_{version}_{cid}.pt"),
                   weights_only=False)
    pos = s["pos"].numpy()
    edges = s["edge_index"].numpy().T                       # [2E, 2] directed
    edges = edges[edges[:, 0] < edges[:, 1]]                # -> unique undirected

    wall = torch.load(os.path.join(save_dir, f"static_{version}_{cid}.pt"), weights_only=False)
    faces = wall["faces"].numpy()
    cells = np.concatenate([np.full((len(faces), 1), 3), faces], axis=1).ravel()
    wall_mesh = pv.PolyData(wall["pos"].numpy(), cells)

    # --- graph PNG: thin slab cross sections viewed face-on. The full pool graph
    # is too dense to read, so render a deterministic 0.25 subsample (mirrors a
    # training draw): the slabs then show the connectivity texture and expose any
    # edge that would bridge disjoint cross sections (a wall-crossing shortcut).
    rng = np.random.default_rng(0)
    keep = np.sort(rng.permutation(len(pos))[: max(1, len(pos) // 4)])
    remap = np.full(len(pos), -1, np.int64); remap[keep] = np.arange(len(keep))
    kept = np.isin(np.arange(len(pos)), keep)
    esub = edges[kept[edges[:, 0]] & kept[edges[:, 1]]]
    esub = remap[esub]                                      # into subsample ids
    gpos = pos[keep]

    ext = gpos.max(0) - gpos.min(0)
    pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2700, 900))
    for col, axis in enumerate((0, 1, 2)):
        thick = slab_frac * ext[axis]
        m = np.abs(gpos[:, axis] - np.median(gpos[:, axis])) < thick
        pl.subplot(0, col)
        pl.add_mesh(pv.PolyData(gpos[m]), color="black", point_size=2.5,
                    render_points_as_spheres=True)
        emask = m[esub[:, 0]] & m[esub[:, 1]]
        es = (np.cumsum(m) - 1)[esub[emask]]                # remap to slab-local ids
        if len(es) > max_render_edges:
            es = es[rng.choice(len(es), max_render_edges, replace=False)]
        if len(es):
            lines = np.column_stack([np.full(len(es), 2), es]).ravel()
            pl.add_mesh(pv.PolyData(gpos[m], lines=lines), color="steelblue",
                        line_width=1.0, opacity=0.6)
        pl.add_text(f"{version}/{cid}  graph slab |{'xyz'[axis]}|<{slab_frac:.0%}  "
                    f"({m.sum()} pts, {int(emask.sum())} edges)",
                    position="upper_left", font_size=10)
        pl.camera_position = ["yz", "xz", "xy"][axis]
    pl.screenshot(graph_path)
    pl.close()
    print(f"[ok] graph png -> {graph_path}")

    # --- velocity GIF: |v| per frame ---
    mags = []
    for t in range(n_frames):
        f = torch.load(os.path.join(interior_dir, f"interior_{version}_{cid}_{t}.pt"),
                       weights_only=False)
        mags.append(np.linalg.norm(f["vel"].numpy(), axis=1))
    vmax = max_speed if max_speed is not None else float(np.percentile(np.concatenate(mags), 99))

    cloud = pv.PolyData(pos)
    cloud["speed"] = mags[0]
    pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
    pl.add_mesh(wall_mesh, color="white", opacity=0.12)
    # translucent points so near-wall (slow) points don't occlude the fast core —
    # the interior structure stays visible through the outer shell
    pl.add_mesh(cloud, scalars="speed", cmap="turbo", clim=[0.0, vmax],
                point_size=point_size, render_points_as_spheres=True,
                opacity=point_opacity,
                scalar_bar_args=dict(title="|v| (m/s)", n_labels=5))
    pl.add_text(f"frame 0/{n_frames}", position="upper_left", font_size=14, name="ftxt")
    pl.open_gif(gif_path, fps=fps)
    for t, mag in enumerate(mags):
        cloud["speed"] = mag                                     # in-place scalar update
        pl.add_text(f"frame {t}/{n_frames}", position="upper_left", font_size=14, name="ftxt")
        pl.write_frame()
    pl.close()
    print(f"[ok] {version}/{cid}: {n_frames} frames -> {gif_path}  |v| clim=[0, {vmax:.3f}]")


if __name__ == "__main__":
    # Build the interior dataset. Constructing the dataset processes every case
    # (idempotent — already-complete cases are skipped): FPS pool, validated pool
    # graph, and per-frame vel/press/grad_v/grad_p. Per-case graph + velocity
    # sanity renders are written for the first `sanity_first` cases PER VERSION
    # into interior_dir/sanity while processing.
    save_dir = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full"
    interior_dir = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior"
    ds = AngioFlowWallInteriorDataset(
        case_root_angioflow="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged",
        case_root_aneug_flow="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data",
        save_dir=save_dir,
        interior_dir=interior_dir,
        pool_ratio=0.16,   # fraction of interior nodes kept in the on-disk pool
        n_frames=80,
        decimate_method="fps",
        decimate_ratio=1.0,
        lap_k=8,
        sanity_first=50,   # graph + velocity sanity renders for the first 50 cases of EACH version
    )
    print(f"[dataset] {len(ds)} samples over {len(ds.case_keys)} cases with interior data")

    wall, interior = ds[0]
    print(f"[sample] {wall.version}/{wall.case_id}")
    print(f"  wall:     pos {tuple(wall.pos.shape)}  y {tuple(wall.y.shape)}")
    print(f"  interior: pos {tuple(interior.pos.shape)}  edge_index {tuple(interior.edge_index.shape)}  "
          f"vel {tuple(interior.vel.shape)}  grad_v {tuple(interior.grad_v.shape)}  "
          f"press {tuple(interior.press.shape)}  grad_p {tuple(interior.grad_p.shape)}  "
          f"wall_dist {tuple(interior.wall_dist.shape)}")

    # per-epoch subsample + induced graph is a transform, not baked into get()
    from angioflow.transform import InteriorSubsampleGraph
    _, sub = InteriorSubsampleGraph(sample_ratio=0.25)((wall, interior))
    print(f"  subsample(0.25): {sub.num_nodes} pts, {sub.edge_index.shape[1]} directed edges")

    # tuple batching smoke test: each stream gets its own batch vector
    from torch_geometric.loader import DataLoader
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    wb, ib = next(iter(loader))
    print(f"[batch]  wall {wb.num_graphs} graphs, {wb.num_nodes} nodes  |  "
          f"interior {ib.num_graphs} clouds, {ib.num_nodes} points")

    # ensure graph + velocity sanity renders exist for the first few cases
    for v, cid in ds.case_keys[:5]:
        interior_sanity(save_dir, interior_dir, v, cid, n_frames=80)


"""
conda activate new
python -m angioflow.data.datasets_interior


"""
