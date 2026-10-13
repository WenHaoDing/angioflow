"""
Given a CFD dataset, create registration for the shape.
Registration file format:
wall_mesh_registration.pt = {
    "pos":           [N, 3]   float32  remeshed wall node positions
    "edge_index":    [2, E]   int64    undirected graph edges
    "faces":         [F, 3]   int64    triangle connectivity (for normals/laplacian)
    "node_type":     [N]      int64    0=wall, 1=focused, 2=inlet, 3=outlet_0, 4=outlet_1
    "flow_split":    [2]      float32  outlet flow split ratios
    "aneurysm_type": scalar   int64    aneurysm morphology type (0=typeB, ...)
    "geo_dist":      [N]      float32  graph-geodesic distance to GHD-overlap region
}
`pos` is exactly wall_data.pt's raw CFD point cloud (same N, same order, no remapping): ball-pivot
triangulates that cloud directly (1:1 vertex identity), then fill_small_holes patches the small
topology pinholes ball-pivoting leaves behind while the 3 real openings stay open. A handful of
remaining orphan nodes (untouched by any face) are stitched into `edge_index`/`geo_dist` via KNN
edges. `pos`/`faces` line up directly with wall_data.pt's per-frame WSS arrays — no extra index.
"""

from __future__ import annotations
import os
import glob
import json
from collections import defaultdict

import numpy as np
import torch
import open3d as o3d
import pyvista as pv
pv.start_xvfb()
from scipy.spatial import cKDTree

from angioflow.data.mesh import face_to_edge_index, visualize_zone, remesh_from_pointcloud, fill_small_holes
from angioflow.data.align import compute_geo_distance
from angioflow.data.registration_sanity import SANITY_ROOT, case_sanity_dir, save_registration_sanity



# ---------------------------------------------------------------------------
# GHD helpers (AneuG-Flow and AngioFlow checkpoint formats)
# ---------------------------------------------------------------------------

def get_ghd_chk(case_dir):
    if os.path.exists(os.path.join(case_dir, "checkpoint.npy")):
        chk = np.load(os.path.join(case_dir, "checkpoint.npy"), allow_pickle=True).item()
        chk = {k: torch.tensor(v) for k, v in chk.items()}
        return chk['ghd'], chk['mean_ghd'], chk['std_ghd'], chk['norm_canonical'], chk['cpcd_scale']
    else:
        chk = dict(np.load(os.path.join(case_dir, "ghd_coefficients.npz"), allow_pickle=True))
        chk = {k: torch.tensor(v) for k, v in chk.items()}
        with open(os.path.join(case_dir, "metrics.json")) as f:
            metrics = json.load(f)
        ghd = chk['phi'].view(-1, 3).float()
        return ghd, torch.zeros((1, 3)), torch.ones((1, 3)), 1.0, torch.exp(chk['log_scale']) * metrics['s_can']


def get_focused_mesh(case_dir, ghd_reconstruct):
    ghd, ghd_mean, ghd_std, norm_canonical, shape_scale = get_ghd_chk(case_dir)
    scale = (shape_scale * 0.001 * norm_canonical).view(1, 1)  # ghd_forward_as_Meshes expects [B, 1]
    return ghd_reconstruct.ghd_forward_as_Meshes(
        ghd.clone().unsqueeze(0), mean=ghd_mean, std=ghd_std, scale=scale)


# ---------------------------------------------------------------------------
# Mesh reconstruction and opening detection
# ---------------------------------------------------------------------------

def load_wall_pointcloud(wall_data_path):
    """Read wall coords from AneuG-Flow wall_data.pt at t=0 -> [N, 3] float64."""
    d = torch.load(wall_data_path, weights_only=False)
    x = d["x_coordinate"][0, :, 0]
    y = d["y_coordinate"][0, :, 0]
    z = d["z_coordinate"][0, :, 0]
    return torch.stack([x, y, z], dim=1).numpy().astype(np.float64)


def detect_openings(pts, faces):
    """Boundary-edge connected components -> list of (node_ids, centroid, approx_area)."""
    edge_count = defaultdict(int)
    for tri in faces:
        for i in range(3):
            edge_count[tuple(sorted([tri[i], tri[(i + 1) % 3]]))] += 1
    adj = defaultdict(set)
    for (a, b), c in edge_count.items():
        if c == 1:
            adj[a].add(b); adj[b].add(a)
    visited, openings = set(), []
    for start in list(adj):
        if start in visited:
            continue
        comp, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n); comp.add(n)
            stack.extend(adj[n] - visited)
        node_ids = np.array(sorted(comp))
        centroid = pts[node_ids].mean(axis=0)
        r = np.linalg.norm(pts[node_ids] - centroid, axis=1).mean()
        openings.append((node_ids, centroid, np.pi * r ** 2))
    return openings


def classify_openings(openings, inlet_centroid):
    """Return (inlet_nodes, outlet0_nodes, outlet1_nodes); outlet_0 is larger."""
    dists = [np.linalg.norm(o[1] - inlet_centroid) for o in openings]
    inlet_idx = int(np.argmin(dists))
    others = sorted([(i, o) for i, o in enumerate(openings) if i != inlet_idx],
                    key=lambda x: -x[1][2])
    assert len(others) == 2, f"Expected 2 outlets, got {len(others)}"
    return openings[inlet_idx][0], others[0][1][0], others[1][1][0]


def orphan_knn_edges(pos, faces, k=6):
    """KNN(k) edges from orphan nodes (not touched by any face) into the full point cloud.

    Ball-pivoting on the remeshed OBJ doesn't cover every raw CFD point; this stitches the
    leftover orphans into the graph so geo_dist/edge_index reach them too.
    """
    used = np.zeros(len(pos), dtype=bool)
    used[np.unique(faces)] = True
    orphan_ids = np.where(~used)[0]
    if len(orphan_ids) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    _, idx = cKDTree(pos).query(pos[orphan_ids], k=k + 1)  # k+1: includes self at dist 0
    nbrs = idx[:, 1:]
    edges = np.stack([np.repeat(orphan_ids, k), nbrs.ravel()], axis=1)
    return edges.astype(np.int64)


def verify_registration(case_dir, pts, edge_index, knn_edges):
    """Sanity-check: pos matches wall_data.pt exactly; edge_index includes all KNN stitch edges."""
    name = os.path.basename(case_dir)
    raw = load_wall_pointcloud(os.path.join(case_dir, "wall_data.pt"))
    pos_ok = np.array_equal(pts, raw)
    print(f"[verify] {name}: pos == wall_data.pt exactly: {'PASS' if pos_ok else 'FAIL'}")

    knn_sorted = set(map(tuple, np.sort(knn_edges, axis=1))) if len(knn_edges) > 0 else set()
    edge_set = set(map(tuple, edge_index.T))
    missing = knn_sorted - edge_set
    edges_ok = len(missing) == 0
    print(f"[verify] {name}: edge_index includes all {len(knn_sorted)} KNN stitch edges: "
          f"{'PASS' if edges_ok else f'FAIL ({len(missing)} missing)'}")

    return pos_ok and edges_ok


# ---------------------------------------------------------------------------
# Labelling and visualisation
# ---------------------------------------------------------------------------

def largest_connected_focused(focused_mask, faces):
    """Keep only the largest connected component of focused nodes; drop islands."""
    focused_ids = set(np.where(focused_mask)[0])
    adj = defaultdict(set)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            if a in focused_ids and b in focused_ids:
                adj[a].add(b); adj[b].add(a)
    visited, components = set(), []
    for start in focused_ids:
        if start in visited:
            continue
        comp, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n); comp.add(n)
            stack.extend(adj[n] - visited)
        components.append(comp)
    largest = max(components, key=len) if components else set()
    new_mask = np.zeros(len(focused_mask), dtype=bool)
    new_mask[list(largest)] = True
    return new_mask


def build_node_type(n_pts, focused_mask, inlet_nodes, outlet0_nodes, outlet1_nodes):
    labels = torch.zeros(n_pts, dtype=torch.long)
    labels[focused_mask] = 1
    labels[torch.from_numpy(inlet_nodes)] = 2
    labels[torch.from_numpy(outlet0_nodes)] = 3
    labels[torch.from_numpy(outlet1_nodes)] = 4
    return labels  # [N]


def orphan_fix_vis(pts, faces, knn_edges, out_path):
    """Save PNG: mesh surface (gray) + KNN stitch-edges for orphan nodes (yellow lines)."""
    cells = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1).ravel()
    pl = pv.Plotter(off_screen=True, window_size=(4800, 3600))
    pl.add_mesh(pv.PolyData(pts, cells), color="lightgray", show_edges=True, edge_color="black", line_width=0.3)
    if len(knn_edges) > 0:
        line_pts = np.empty((2 * len(knn_edges), 3))
        line_pts[0::2] = pts[knn_edges[:, 0]]
        line_pts[1::2] = pts[knn_edges[:, 1]]
        pl.add_lines(line_pts, color="yellow", width=0.3)
    pl.add_legend([["mesh surface", "lightgray"], ["KNN stitch edge", "yellow"]])
    pl.screenshot(out_path); pl.close()


# ---------------------------------------------------------------------------
# Per-case and batch processing
# ---------------------------------------------------------------------------

def process_case(case_dir, ghd_reconstruct, aneurysm_type=0, sanity_dir=None):
    name = os.path.basename(case_dir)
    print(f"[...] {name}: loading point cloud")
    wall_data_path = os.path.join(case_dir, "wall_data.pt")
    inlet_centroid_path = glob.glob(os.path.join(case_dir, "inlet_centroid*"))[0]
    flowsplit_path = os.path.join(case_dir, "flowsplit_ratio.txt")

    # pos = wall_data.pt's raw CFD point cloud exactly (same N, same order). Ball-pivot
    # triangulates that same cloud directly (1:1 vertex identity, no remapping), then
    # fill_holes patches the small topology pinholes ball-pivoting leaves behind while
    # the 3 real openings (much larger) are untouched.
    pts = load_wall_pointcloud(wall_data_path)
    _, faces = remesh_from_pointcloud(pts)
    pts, faces = fill_small_holes(pts, faces, hole_size=0.0003)  # 0.3mm

    _out_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(pts * 1000.0),  # metres -> mm
        o3d.utility.Vector3iVector(faces),
    )
    o3d.io.write_triangle_mesh(os.path.join(case_dir, "shape_remeshed_cfd.obj"), _out_mesh)

    # orphan raw points untouched by any face: stitch into the graph via KNN edges
    knn_edges = orphan_knn_edges(pts, faces, k=6)
    print(f"[...] {name}: pos={len(pts)} (raw CFD), faces={len(faces)}, "
          f"orphan KNN edges={len(knn_edges)} — saving preview")
    visualize_zone(pts, faces, os.path.join(case_dir, "remesh_preview.png"))

    edge_index = face_to_edge_index(faces)
    if len(knn_edges) > 0:
        extra = np.sort(knn_edges, axis=1)
        edge_index = np.unique(np.concatenate([edge_index.T, extra], axis=0), axis=0).T

    ic = np.loadtxt(inlet_centroid_path, delimiter=",", skiprows=1)
    inlet_centroid = ic.reshape(-1, 3).mean(axis=0)

    openings = detect_openings(pts, faces)
    print(f"[...] {name}: found {len(openings)} boundary openings")
    if len(openings) != 3:
        print(f"[fail] {name}: expected 3 openings, got {len(openings)} — skipping")
        return False
    inlet_nodes, outlet0_nodes, outlet1_nodes = classify_openings(openings, inlet_centroid)

    focused_Meshes = get_focused_mesh(case_dir, ghd_reconstruct)
    focused_pts = focused_Meshes.verts_packed().detach().cpu().numpy()
    focused_faces = focused_Meshes.faces_packed().cpu().numpy()
    fv = focused_pts[focused_faces]  # [F, 3, 3]
    max_edge = max(
        np.linalg.norm(fv[:, 0] - fv[:, 1], axis=1).max(),
        np.linalg.norm(fv[:, 1] - fv[:, 2], axis=1).max(),
        np.linalg.norm(fv[:, 2] - fv[:, 0], axis=1).max(),
    )
    thresh = max_edge * 2
    dists, _ = cKDTree(focused_pts).query(pts, k=1)
    focused_mask = largest_connected_focused(dists < thresh, faces)
    print(f"[...] {name}: focused thresh={thresh*1000:.2f}mm, focused={focused_mask.sum()}")

    print(f"[...] {name}: computing geodesic distance")
    geo_dist = compute_geo_distance(pts, faces, focused_Meshes, extra_edges=knn_edges)  # [N] float32

    node_type = build_node_type(len(pts), focused_mask, inlet_nodes, outlet0_nodes, outlet1_nodes)
    flow_split = torch.from_numpy(np.loadtxt(flowsplit_path).reshape(-1).astype(np.float32))  # [2]

    reg = {
        "pos":           torch.from_numpy(pts).float(),
        "edge_index":    torch.from_numpy(edge_index),
        "faces":         torch.from_numpy(faces),
        "node_type":     node_type,
        "flow_split":    flow_split,
        "aneurysm_type": torch.tensor(aneurysm_type, dtype=torch.long),
        "geo_dist":      geo_dist,
    }
    torch.save(reg, os.path.join(case_dir, "wall_mesh_registration.pt"))
    verify_registration(case_dir, pts, edge_index, knn_edges)

    if sanity_dir is not None:
        case_dir_sanity = case_sanity_dir(name, root=sanity_dir)
        save_registration_sanity(name, pts, faces, node_type, geo_dist, out_dir=case_dir_sanity)
        orphan_fix_vis(pts, faces, knn_edges, os.path.join(case_dir_sanity, "orphan_fix.png"))

    print(f"[ok] {os.path.basename(case_dir)}: {len(pts)} pts, focused={focused_mask.sum()}, "
          f"inlet={len(inlet_nodes)}, out0={len(outlet0_nodes)}, out1={len(outlet1_nodes)}")
    return True


def process_all(dataset_root, ghd_reconstruct, aneurysm_type=0, case_pattern="stable_*", sanity_dir=None):
    import traceback
    for case_dir in sorted(glob.glob(os.path.join(dataset_root, case_pattern))):
        name = os.path.basename(case_dir)
        try:
            process_case(case_dir, ghd_reconstruct, aneurysm_type=aneurysm_type, sanity_dir=sanity_dir)
        except Exception as e:
            print(f"[fail] {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    from AneuG.models.ghd_reconstruct import GHD_Reconstruct
    from AneuG.utils.utils import safe_load_mesh

    canonical_path = os.path.join('AneuG/checkpoints/alignment/canonical_typeB')
    canonical_Meshes = safe_load_mesh(os.path.join(canonical_path, 'part_aligned_updated.obj'))
    ghd_reconstruct = GHD_Reconstruct(canonical_Meshes, os.path.join(canonical_path, "canonical_typeB_144.pkl"))

    dataset_root = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data"
    sanity_dir = SANITY_ROOT
    process_all(dataset_root, ghd_reconstruct, aneurysm_type=0, sanity_dir=sanity_dir)


"""
conda activate new
python -m angioflow.data.register_cfd_shapes_aneug_flow


1. Git clone the AneuG-Net repo
2. In the same repo, clone the AneuG repo: git clone https://github.com/WenHaoDing/AneuG.git
3. cd AneuG
4. Download the canonical_typeB checkpoint from Google Drive and unzip: https://drive.google.com/file/d/1T0CZByyLtk7e-WKu2ZAjunuFosXYl0cH/view?usp=drive_link
5. Preprocess AneuG-Flow data, this only generates new data, won't contaminate the original data: python -m angioflow.data.register_cfd_shapes_aneug_flow
6. Run pyG dataset to preprocess graph: python -m angioflow.data.datasets_interior
6. Run training script:  /home/wenhao/AneuG-Net/angioflow/training/train_interior.py

"""
