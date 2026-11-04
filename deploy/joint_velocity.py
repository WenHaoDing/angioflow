"""Velocity inference with the joint (WSS + velocity + pressure) model on a NEW vessel surface.

Input: an open triangulated lumen surface around one aneurysm (clipped at 1 inlet and 1-2
outlets), which surface vertices are the aneurysm "focused" region, and which opening is
the inlet. Output: blood velocity (m/s) at points inside the lumen near the aneurysm, at
frame 0 of the training inlet waveform.

    from angioflow.deploy.joint_velocity import JointVelocityPredictor, prepare_surface
    pred = JointVelocityPredictor.from_bundle("joint_velocity_bundle.pt", device="cuda:0")
    surf = prepare_surface(verts_mm, faces, units="mm", inlet_point=inlet_xyz_mm,
                           outlet_points=[out0_xyz_mm, out1_xyz_mm], focused_mask=focus_mask,
                           sac_points=dome_xyz_mm)      # smooths (AneuG regularizer) + remeshes
    out = pred.predict(surf)            # out.points [Q,3] (input units), out.velocity [Q,3] m/s

This replaces deploy/mesh_to_graph.py for the current data format (AngioFlow-PyG, v3):
MRI surfaces are first smoothed with AneuG's dome-normalized mesh regularizer (optional,
default on; needs the AneuG repo importable as `AneuG`), then remeshed to the training edge;
the wall features are computed with the SAME functions that built the training data --
cot_mass_eigs (generalized L v = lam M v, made dimensionless by A/4pi as the dataset does),
pyvista normals, per-opening geodesics, the (d0/d1)^2.45 flow-split rule -- not copies.
See deploy/JOINT_VELOCITY_HANDOFF.md for what the model expects and why.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_undirected

from angioflow.data.flow_split import flow_split_from_geometry
from angioflow.data.mesh import compute_vertex_normals, face_to_edge_index
from angioflow.data.per_opening_geo import per_opening_geo_dist
from angioflow.data.precompute_wall_static import cot_mass_eigs

FOCUSED, INLET, OUTLET_0, OUTLET_1 = 1, 2, 3, 4      # node_type codes (0 = plain wall)
UNIT_TO_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}

# Training wall meshes (AngioFlow-PyG v3, measured on 60 cases per dataset). AngioFlow is the
# real-patient set and the test set; its walls are the CFD surface meshes, used unremeshed.
TRAINING_MESH = dict(
    edge_mm=0.12,                 # median edge length: AngioFlow 0.123, AneuG-Flow 0.108, AneuGv2 0.106
    edge_mm_p10_p90=(0.108, 0.156),
    nodes=30_000,                 # median node count (p10-p90 ~22k-32k AngioFlow, ~28k-44k AneuG)
    area_mm2=405,                 # median surface area AngioFlow (p10-p90 229-627)
    extent_mm=20,                 # median largest bounding-box side (p10-p90 16-24 AngioFlow)
    focused_fraction=0.5,         # node_type==1 covers ~half the nodes
)
MIN_WALL_DIST_M = 1e-4            # training dropped interior points < 0.1 mm from the wall
# Training pool density: interior pool points per mm^3 of capped lumen volume, AngioFlow
# median over 40 cases, p10-p90 113-413 (validate_joint_velocity.py --measure_density).
TRAINING_POOL_PER_MM3 = 233.0
# Training fed the model a random 20% of the pool per step (interior_sample_ratio 0.2):
# the default query density. Velocity error was the same at full pool density
# (validate_joint_velocity.py), which only costs 5x the time and memory.
TRAINING_QUERY_PER_MM3 = 0.2 * TRAINING_POOL_PER_MM3


# ---------------------------------------------------------------------------- mesh helpers

def boundary_loops(faces):
    """Vertex-index arrays, one per boundary loop (opening) of an open triangle mesh."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    u, c = np.unique(e, axis=0, return_counts=True)
    b = u[c == 1]
    if len(b) == 0:
        return []
    verts, inv = np.unique(b, return_inverse=True)
    inv = inv.reshape(-1, 2)
    n, lab = connected_components(csr_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])),
                                             shape=(len(verts),) * 2), directed=False)
    return [verts[lab == k] for k in range(n)]


def remesh_isotropic(vertices_mm, faces, target_edge_mm=TRAINING_MESH["edge_mm"], iterations=10):
    """Isotropic remesh (pymeshlab) to the training edge length. Openings stay open."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(np.asarray(vertices_mm, np.float64), np.asarray(faces, np.int32)))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations, targetlen=pymeshlab.PureValue(target_edge_mm),
        maxsurfdist=pymeshlab.PureValue(0.5 * target_edge_mm))
    m = ms.current_mesh()
    return np.asarray(m.vertex_matrix(), np.float64), np.asarray(m.face_matrix(), np.int64)


def sac_span_mm(points_mm):
    """Aneurysm sac size as the dome-normalized regularizer defines it: the longest extent
    along the points' own principal axes (dome surface vertices or dome-label voxel centres)."""
    from AneuG.mesh_regularizer.dome_normalized import pca_span
    return pca_span(np.asarray(points_mm, np.float64))


def regularize_surface(vertices_mm, faces, sac_mm, time_budget_s=None, **kwargs):
    """Dome-normalized roughness regularization (AneuG mesh_regularizer.dome_normalized, the
    domenorm6.15 reference model): MRI/segmentation surfaces carry voxel-scale noise the CFD
    training walls do not have. The shape is scaled so its sac matches the 6.15 mm reference
    sac, smoothed toward the reference roughness with shrinkage compensation and the openings
    frozen, then scaled back. Same settings as `python -m mesh_regularizer.dome_normalized run`
    (no export remesh -- prepare_surface remeshes to the training edge afterwards).
    time_budget_s: wall-clock limit per case (None = no limit, ~1-3 min on a training-size
    case); when hit, report["status"] is "time_budget_exceeded" and the mesh smoothed so far
    is returned. Returns (vertices_mm, faces, report)."""
    import trimesh
    from AneuG.mesh_regularizer.dome_normalized import DomeNormalizedRegularizer
    opts = dict(export_edge=None, time_budget_s=time_budget_s, n_seeds=4000, verbose=False)
    opts.update(kwargs)
    reg = DomeNormalizedRegularizer(**opts)
    out, report = reg.forward(trimesh.Trimesh(np.asarray(vertices_mm, np.float64),
                                              np.asarray(faces, np.int64), process=False),
                              float(sac_mm), freeze_boundary=True)
    return np.asarray(out.vertices, np.float64), np.asarray(out.faces, np.int64), report


def mesh_report(vertices_m, faces):
    """Numbers to compare against TRAINING_MESH."""
    p, f = np.asarray(vertices_m) * 1e3, np.asarray(faces)
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1), axis=0)
    area = 0.5 * np.linalg.norm(np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]]), axis=1).sum()
    return dict(nodes=len(p), edge_mm=float(np.median(np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1))),
                area_mm2=float(area), extent_mm=float((p.max(0) - p.min(0)).max()),
                openings=len(boundary_loops(f)))


def _geodesic_to(pos, faces, sources):
    """Geodesic (mesh-edge Dijkstra) distance from every vertex to a vertex set."""
    e = face_to_edge_index(faces)
    w = np.linalg.norm(pos[e[0]] - pos[e[1]], axis=1)
    adj = csr_matrix((np.concatenate([w, w]), (np.concatenate(e[::-1]), np.concatenate(e))),
                     shape=(len(pos),) * 2)
    return dijkstra(adj, directed=False, indices=np.flatnonzero(sources), min_only=True)


# ---------------------------------------------------------------------------- the surface

@dataclass
class PreparedSurface:
    """A wall mesh with every per-node feature the model reads, in the training layout.

    pos is centred (wall mean removed) and in metres; `center` + `to_metres` map it back to
    the caller's frame. node_type: 0 wall, 1 focused, 2 inlet rim, 3/4 outlet rims."""
    pos: torch.Tensor                 # [N, 3] metres, centred
    faces: np.ndarray                 # [F, 3]
    node_type: torch.Tensor           # [N]
    normals: torch.Tensor             # [N, 3]
    geo_dist: torch.Tensor            # [N, 3] per opening (inlet, outlet_0, outlet_1), in [0, 1]
    global_geo_dist: torch.Tensor     # [N] to the focused region, in [0, 1] (crop only)
    cot_eigvec: torch.Tensor          # [N, k]
    cot_eigval: torch.Tensor          # [k] dimensionless (lam * A / 4pi)
    flow_split: torch.Tensor          # [2] (q_outlet0, q_outlet1)
    edge_index: torch.Tensor          # [2, E] face edges, i < j
    center: np.ndarray                # [1, 3] metres: caller frame -> centred frame
    to_metres: float
    report: dict = field(default_factory=dict)


def prepare_surface(vertices, faces, units="mm", inlet_point=None, outlet_points=None,
                    focused_mask=None, smooth=True, sac_mm=None, sac_points=None,
                    smooth_time_budget_s=None, smooth_kwargs=None, remesh=True, target_edge_mm=TRAINING_MESH["edge_mm"],
                    lap_k=8):
    """Raw open lumen surface -> PreparedSurface.

    vertices [N, 3] in `units` ('mm' for image/DICOM space), faces [F, 3] triangles.
    The surface must be OPEN at exactly 1 inlet + 1 or 2 outlets (boundary loops).
    inlet_point [3] (same units): any point near the inlet opening; the nearest loop wins.
    outlet_points [[3]] or [[3], [3]]: a point near outlet_0 (and outlet_1), same rule. The
        order is the caller's: it fixes which outlet is outlet_0 in geo_dist and flow_split.
    focused_mask [N] bool on the INPUT vertices: the aneurysm focused region (sac/dome, as
        clipped_reconstruction.ply marked it in training). Mapped onto the final vertices.
    smooth: dome-normalized roughness regularization first (regularize_surface) -- default on,
        for MRI/segmentation surfaces. Needs the sac size: sac_mm (longest PCA span of the
        sac, mm) or sac_points [M, 3] in `units` (dome vertices or dome-label voxel centres)
        to measure it from. smooth_time_budget_s caps the smoothing wall-clock time per case
        (None = run to convergence, typically 1-3 min). smooth_kwargs go to
        DomeNormalizedRegularizer (e.g.
        tolerance, stopping="relaxed"). Turn it off for meshes already CFD-smooth.
    remesh: isotropic remesh to target_edge_mm (the training density). Always applied after
        smoothing, whose own internal remesh is at a sac-dependent edge.
    """
    to_m = UNIT_TO_M[units] if isinstance(units, str) else float(units)
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    if focused_mask is None or inlet_point is None or not outlet_points:
        raise ValueError("focused_mask, inlet_point and outlet_points are required (the crop "
                         "and the boundary conditions depend on them)")
    focused_mask = np.asarray(focused_mask, bool)
    v_in_mm = v * to_m * 1e3
    v_mm, smooth_report = v_in_mm, None
    if smooth:
        if sac_mm is None:
            if sac_points is None:
                raise ValueError("smooth=True needs the aneurysm sac size: pass sac_mm, or "
                                 "sac_points to measure it from (or smooth=False)")
            sac_mm = sac_span_mm(np.asarray(sac_points, np.float64) * to_m * 1e3)
        v_mm, f, smooth_report = regularize_surface(v_in_mm, f, sac_mm, smooth_time_budget_s,
                                                    **(smooth_kwargs or {}))
    if remesh or smooth:
        v_mm, f = remesh_isotropic(v_mm, f, target_edge_mm)
    if v_mm is not v_in_mm:
        focused_mask = focused_mask[cKDTree(v_in_mm).query(v_mm)[1]]   # nearest input vertex
    pos = v_mm * 1e-3
    used = np.unique(f)
    if len(used) != len(pos):
        raise ValueError(f"{len(pos) - len(used)} vertices are in no face -- remove them first")

    loops = boundary_loops(f)
    if not 2 <= len(loops) <= 3:
        raise ValueError(f"surface has {len(loops)} openings; need 1 inlet + 1-2 outlets "
                         "(clip the vessel with planes; close any other holes)")
    if len(outlet_points) != len(loops) - 1:
        raise ValueError(f"{len(outlet_points)} outlet point(s) given but the surface has "
                         f"{len(loops) - 1} non-inlet opening(s)")
    cents = np.stack([pos[l].mean(0) for l in loops])
    pick = [int(np.argmin(np.linalg.norm(cents - np.asarray(pt, np.float64) * to_m, axis=1)))
            for pt in [inlet_point, *outlet_points]]
    if len(set(pick)) != len(pick):
        raise ValueError(f"inlet/outlet points map to openings {pick}: two points chose the same one")
    node_type = np.zeros(len(pos), np.int64)
    node_type[focused_mask] = FOCUSED
    for code, k in zip((INLET, OUTLET_0, OUTLET_1), pick):
        node_type[loops[k]] = code
    outlets = pick[1:]
    if not (node_type == FOCUSED).any():
        raise ValueError("focused_mask selects no surface vertex")

    center = pos.mean(0, keepdims=True)
    pc = pos - center
    lam, U, qc = cot_mass_eigs(pc, f, lap_k)
    area = 0.5 * np.linalg.norm(np.cross(pc[f[:, 1]] - pc[f[:, 0]], pc[f[:, 2]] - pc[f[:, 0]]), axis=1).sum()
    edge_index = face_to_edge_index(f)
    opening = per_opening_geo_dist(pc, edge_index, node_type)
    geo = np.stack([opening.get(n, np.ones(len(pc))) for n in ("inlet", "outlet_0", "outlet_1")], 1)
    g = _geodesic_to(pc, f, node_type == FOCUSED)
    if not np.isfinite(g).all() or not np.isfinite(geo).all():
        raise ValueError("mesh is not one connected surface (infinite geodesic distances)")
    rep = mesh_report(pos, f)
    rep.update(eig_max_resid=qc["eig_max_resid"], n_outlets=len(outlets),
               focused_fraction=float((node_type == FOCUSED).mean()))
    if smooth_report is not None:
        rep["smoothing"] = {k: smooth_report.get(k) for k in (
            "status", "rounds", "sac_span_mm", "alpha", "target_scales_used",
            "surface_deviation_vs_original_max_mm_real", "signed_excess_after")}
    return PreparedSurface(
        pos=torch.from_numpy(pc).float(), faces=f, node_type=torch.from_numpy(node_type),
        normals=torch.from_numpy(compute_vertex_normals(pc, f)).float(),
        geo_dist=torch.from_numpy(geo).float(),
        global_geo_dist=torch.from_numpy(g / max(g.max(), 1e-12)).float(),
        cot_eigvec=torch.from_numpy(U).float(),
        cot_eigval=torch.from_numpy(lam * area / (4 * np.pi)).float(),
        flow_split=torch.from_numpy(flow_split_from_geometry(pc, node_type)).float(),
        edge_index=torch.from_numpy(edge_index), center=center, to_metres=to_m, report=rep)


# ---------------------------------------------------------------------------- query points

def _capped_polydata(pos, faces, node_type):
    """Close every opening with a fan around its rim centroid (consistent orientation)."""
    import pyvista as pv
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(e, 1)
    u, inv, c = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bnd = e[c[inv] == 1]                                   # directed as in their face
    pts, tris = [pos], [faces]
    for t in (INLET, OUTLET_0, OUTLET_1):
        sel = bnd[node_type[bnd[:, 0]] == t]
        if len(sel) == 0:
            continue
        cid = sum(len(p) for p in pts)
        pts.append(pos[node_type == t].mean(0, keepdims=True))
        tris.append(np.stack([sel[:, 1], sel[:, 0], np.full(len(sel), cid)], 1))
    P, F = np.concatenate(pts), np.concatenate(tris)
    return pv.PolyData(P, np.concatenate([np.full((len(F), 1), 3), F], 1).ravel())


def sample_lumen_points(surf, n_points=None, density_per_mm3=None, region_mask=None, seed=0):
    """Points inside the lumen, distributed like the training interior pool: denser near the
    wall, evenly spread (the dataset's own density-weighted FPS), none within 0.1 mm of the
    wall. Returns [Q, 3] in the CENTRED metre frame (PreparedSurface.pos frame).

    region_mask [N] bool: only sample the lumen inside the bounding box of these wall nodes
    (predict passes the crop). Default count: the density the model trained at
    (TRAINING_QUERY_PER_MM3) x the lumen volume sampled."""
    import pyvista as pv
    from angioflow.data.datasets_interior import _InteriorMixin
    pos, nt = surf.pos.numpy().astype(np.float64), surf.node_type.numpy()
    closed = _capped_polydata(pos, surf.faces, nt)
    box = pos if region_mask is None else pos[np.asarray(region_mask, bool)]
    lo, hi = box.min(0), box.max(0)
    rng = np.random.default_rng(seed)

    def draw(n):
        x = rng.uniform(lo, hi, size=(n, 3))
        return x[pv.PolyData(x).select_enclosed_points(closed, check_surface=True)["SelectedPoints"].astype(bool)]

    cand = draw(200_000)                                  # also measures the lumen volume in the box
    vol_mm3 = np.prod(hi - lo) * 1e9 * len(cand) / 200_000
    if n_points is None:
        n_points = int(round((density_per_mm3 or TRAINING_QUERY_PER_MM3) * vol_mm3))
    while len(cand) < 4 * n_points:                       # FPS thins ~4x candidates
        cand = np.concatenate([cand, draw(200_000)])
    wall_dist = cKDTree(pos).query(cand)[0]
    pool = argparse.Namespace(min_wall_dist=MIN_WALL_DIST_M, near_decay=0.1, far_floor=0.25,
                              pool_ratio=n_points / len(cand))
    idx = _InteriorMixin._sample_pool(pool, cand, wall_dist, rng)
    return cand[idx]



# ---------------------------------------------------------------------------- the model

@dataclass
class VelocityResult:
    points: np.ndarray        # [Q, 3] in the caller's frame and units (as passed to prepare_surface)
    velocity: np.ndarray      # [Q, 3] m/s
    query_index: np.ndarray   # [Q] row of each result in the query_points passed (if any)
    wall_kept: np.ndarray     # [N] bool: surface vertices inside the crop


class JointVelocityPredictor:
    """The trained joint model + everything needed to run it outside the training repo's
    data folders (model weights, args, interior normalization stats, inlet waveform)."""

    def __init__(self, bundle, device="cpu"):
        from angioflow.training.train_joint import build_model
        self.args = argparse.Namespace(**bundle["args"])
        self.device = torch.device(device)
        self.model = build_model(self.args, self.device)
        self.model.load_state_dict(bundle["model"])
        self.model.eval()
        self.model.query_ratio = 1.0           # WSS at every wall node (no random subset)
        st = bundle["interior_stats"]
        self.pos_std = st["pos_std"].float()
        self.pos_scalar = float(st["pos_std"].flatten()[0])
        self.vel_mean, self.vel_std = st["vel_mean"].float(), st["vel_std"].float()
        self.waveform = bundle["waveform"].to(self.device)
        self.epoch = bundle.get("epoch")
        self.lap_k = self.args.lap_k

    @staticmethod
    def export_bundle(run_dir, out_path, which="best", data_root=None, waveform_path=None):
        """Pack <run_dir>/<which>.pth + its stats + waveform into one portable file."""
        from angioflow.training.train import load_waveform
        from angioflow.training.train_joint import interior_stats_path
        chk = torch.load(os.path.join(run_dir, f"{which}.pth"), map_location="cpu", weights_only=False)
        a = argparse.Namespace(**chk["args"])
        if data_root:
            a.save_dir = data_root
        wf = load_waveform(waveform_path or a.waveform_path, length=1000, device=torch.device("cpu"))
        bundle = dict(model=chk["model"], args=chk["args"], epoch=chk["epoch"],
                      best_loss=chk.get("best_loss"), run_name=a.run_name,
                      interior_stats=torch.load(interior_stats_path(a), weights_only=False),
                      waveform=wf, training_mesh=TRAINING_MESH)
        torch.save(bundle, out_path)
        return out_path

    @classmethod
    def from_bundle(cls, path, device="cpu"):
        return cls(torch.load(path, map_location="cpu", weights_only=False), device)

    def crop_mask(self, surf, crop_p=None):
        """Wall nodes kept: focused region + geodesic ring of depth crop_p x max distance.
        Training drew crop_p ~ U[mask_low, mask_high] = U[0, 0.25]; default = mask_high."""
        p = self.args.mask_high if crop_p is None else crop_p
        return (surf.node_type == FOCUSED) | (surf.global_geo_dist <= p * surf.global_geo_dist.max())

    @torch.no_grad()
    def predict(self, surf, query_points=None, crop_p=None, n_points=None, seed=0):
        """Velocity (m/s) inside the lumen around the aneurysm.

        query_points [Q, 3] in the caller's frame/units, or None to sample the lumen like the
        training data (sample_lumen_points). Points whose nearest wall was cropped away are
        dropped (as in training); `query_index` says which input rows survived.
        seed fixes both the lumen sampling and the model's pooling, so a call is reproducible;
        averaging a few seeds smooths the ~1% pooling noise."""
        if surf.cot_eigvec.shape[1] != self.lap_k:
            raise ValueError(f"surface has {surf.cot_eigvec.shape[1]} cot modes, model wants {self.lap_k}")
        keep = self.crop_mask(surf, crop_p)
        if query_points is None:
            q = sample_lumen_points(surf, n_points=n_points, region_mask=keep, seed=seed)
        else:
            q = np.asarray(query_points, np.float64) * surf.to_metres - surf.center
        q = torch.from_numpy(q).float()
        wall_dist = torch.from_numpy(cKDTree(surf.pos.numpy()).query(q.numpy())[0]).float()

        wall = Data(pos=surf.pos, edge_index=surf.edge_index, normals=surf.normals,
                    geo_dist=surf.geo_dist, cot_eigvec=surf.cot_eigvec, cot_eigval=surf.cot_eigval,
                    node_type=surf.node_type, t=torch.tensor([0])).subgraph(keep)
        # JointAneurysmCrop: keep a point only if its nearest wall survived the crop
        d_kept = torch.from_numpy(cKDTree(wall.pos.numpy()).query(q.numpy())[0]).float()
        alive = d_kept <= self.args.crop_factor * wall_dist + self.args.crop_slack
        q, wall_dist, qidx = q[alive], wall_dist[alive], torch.arange(len(alive))[alive]
        # CenterAndRandomRotate(rotate=False) on the cropped wall, then JointNormalize
        c = wall.pos.mean(0, keepdim=True)
        wall.pos = (wall.pos - c) / self.pos_std
        wall.edge_index = to_undirected(wall.edge_index, num_nodes=wall.num_nodes)
        if getattr(self.args, "add_flow_split", False):
            wall.bc_flow_split = surf.flow_split.view(1, 2)
        interior = Data(pos=(q - c) / self.pos_std, wall_dist=wall_dist / self.pos_scalar,
                        t=torch.tensor([0]))

        wb = Batch.from_data_list([wall]).to(self.device)
        ib = Batch.from_data_list([interior]).to(self.device)
        # the encoder's FPS pooling picks a random start point: ~1% run-to-run spread
        # unseeded, identical with a fixed seed (global RNG state is restored afterwards)
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(seed)
            _, _, vel, _ = self.model(wb, ib, self.waveform)
        vel = vel.float().cpu() * self.vel_std + self.vel_mean
        pts = (q.double().numpy() + surf.center) / surf.to_metres
        return VelocityResult(points=pts, velocity=vel.numpy(), query_index=qidx.numpy(),
                              wall_kept=keep.numpy())


# ---------------------------------------------------------------------------- sanity render

def save_velocity_sanity(surf, out, path, vel_gt=None, slab_mm=0.6, title=""):
    """One PNG to eyeball a deployment: the input labels (top-left: openings, focused region,
    crop) and |v| on three thin slabs through the crop, along its principal axes --
    prediction, plus CFD and |error| rows when vel_gt [Q, 3] (aligned with out) is given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wall = surf.pos.numpy() * 1e3                                   # centred mm
    nt = surf.node_type.numpy()
    pts = (out.points * surf.to_metres - surf.center) * 1e3
    kept = wall[out.wall_kept]
    c = kept.mean(0)
    axes = np.linalg.svd(kept - c, full_matrices=False)[2]          # rows: principal directions
    spd = np.linalg.norm(out.velocity, axis=1)
    rows = [("prediction", spd)]
    if vel_gt is not None:
        gt = np.linalg.norm(vel_gt, axis=1)
        rows += [("CFD", gt), ("|error|", np.linalg.norm(out.velocity - vel_gt, axis=1))]
    vmax = float(np.percentile(rows[-2 if vel_gt is not None else 0][1], 99.5))

    fig, ax = plt.subplots(1 + len(rows), 3, figsize=(13, 4.2 * (1 + len(rows))))
    # input row: wall nodes by label, in the plane of the two main axes
    w2 = (wall - c) @ axes[:2].T
    colours = {0: "#c8c8c8", FOCUSED: "#e8743b", INLET: "#2a7fd4", OUTLET_0: "#3aa655", OUTLET_1: "#a64ac9"}
    names = {0: "wall", FOCUSED: "focused", INLET: "inlet", OUTLET_0: "outlet_0", OUTLET_1: "outlet_1"}
    for k in (0, FOCUSED, INLET, OUTLET_0, OUTLET_1):
        m = nt == k
        if m.any():
            ax[0, 0].scatter(w2[m, 0], w2[m, 1], s=0.3 if k <= FOCUSED else 4, c=colours[k], label=names[k])
    ax[0, 0].legend(markerscale=6, fontsize=8, loc="best")
    ax[0, 0].set_title("labels (projection)")
    ax[0, 1].scatter(w2[~out.wall_kept, 0], w2[~out.wall_kept, 1], s=0.3, c="#dddddd")
    ax[0, 1].scatter(w2[out.wall_kept, 0], w2[out.wall_kept, 1], s=0.3, c="#333333")
    ax[0, 1].set_title(f"crop: {int(out.wall_kept.sum())} / {len(wall)} wall nodes")
    r = surf.report
    ax[0, 2].axis("off")
    ax[0, 2].text(0, 0.95, "\n".join([
        title, f"nodes {r['nodes']}  (training ~30k)", f"edge {r['edge_mm']:.3f} mm  (0.12)",
        f"area {r['area_mm2']:.0f} mm²  (~400)", f"extent {r['extent_mm']:.1f} mm  (~20)",
        f"outlets {r['n_outlets']}  flow split [{surf.flow_split[0]:.3f}, {surf.flow_split[1]:.3f}]",
        f"focused {r['focused_fraction']:.0%} of nodes  (~50%)",
        f"query points {len(pts)}  |v| max {spd.max():.3f} m/s"]), va="top", family="monospace", fontsize=9)

    for j in range(3):                                              # slab normal = axis j
        u, v = [a for a in range(3) if a != j]
        sel = np.abs((pts - c) @ axes[j]) < slab_mm / 2
        wsel = np.abs((kept - c) @ axes[j]) < slab_mm / 2
        P, W = (pts[sel] - c) @ axes[[u, v]].T, (kept[wsel] - c) @ axes[[u, v]].T
        for i, (name, val) in enumerate(rows, start=1):
            a = ax[i, j]
            a.scatter(W[:, 0], W[:, 1], s=1, c="#999999")
            hi = vmax if name != "|error|" else 0.5 * vmax
            sc = a.scatter(P[:, 0], P[:, 1], s=3, c=val[sel], cmap="turbo" if name != "|error|" else "magma",
                           vmin=0, vmax=hi)
            a.set_title(f"{name} |v|, slab normal to axis {j} ({sel.sum()} pts)", fontsize=9)
            fig.colorbar(sc, ax=a, fraction=0.046, label="m/s")
    for a in ax.ravel():
        a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
