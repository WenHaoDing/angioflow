"""Per-opening geodesic distance (one field per inlet / outlet) + sanity images.

Re-uses the registration's `node_type` labels (2=inlet, 3=outlet_0, 4=outlet_1)
to mark source nodes. Distances are multi-source Dijkstra over the wall mesh
graph, edge weights = Euclidean edge length.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pyvista as pv
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


OPENING_TYPES = {2: "inlet", 3: "outlet_0", 4: "outlet_1"}


def _build_graph(pos, edge_index):
    """Symmetric weighted CSR adjacency (edge weight = Euclidean edge length)."""
    i, j = edge_index
    w = np.linalg.norm(pos[i] - pos[j], axis=1)
    N = len(pos)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.concatenate([w, w])
    return csr_matrix((data, (rows, cols)), shape=(N, N))


def per_opening_geo_dist(pos, edge_index, node_type):
    """For each opening type in OPENING_TYPES, return [N] geodesic distance min-pooled
    over its source nodes and normalized to [0, 1] (per its own finite max).
    Accepts numpy arrays or torch tensors; skips absent openings.
    """
    pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos, dtype=np.float64)
    edge_index = np.asarray(edge_index.cpu() if hasattr(edge_index, "cpu") else edge_index)
    node_type = np.asarray(node_type.cpu() if hasattr(node_type, "cpu") else node_type)
    adj = _build_graph(pos, edge_index)

    out = {}
    for tid, name in OPENING_TYPES.items():
        src = np.where(node_type == tid)[0]
        if src.size == 0:
            continue
        d = dijkstra(adj, directed=False, indices=src, min_only=True)
        finite = np.isfinite(d)
        dmax = d[finite].max() if finite.any() else 1.0
        out[name] = d / max(dmax, 1e-8)                     # in [0, 1]; unreachable nodes stay inf
    return out


def _vis_dist(pos, faces, dist, out_path, title):
    cells = np.concatenate([np.full((len(faces), 1), 3), faces], axis=1).ravel()
    mesh = pv.PolyData(pos, cells)
    mesh["geo"] = dist
    pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
    pl.add_mesh(mesh, scalars="geo", cmap="viridis",
                scalar_bar_args=dict(title=title, n_labels=5))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pl.screenshot(out_path)
    pl.close()


def process_case(reg_path, sanity_dir, save_back=False):
    """Compute per-opening dists for one case; save PNG per opening.
    If `save_back=True`, also write the dict back into `reg` under key 'geo_dist_per_opening'.
    """
    case_name = os.path.basename(os.path.dirname(reg_path))
    reg = torch.load(reg_path, weights_only=False)
    pos = reg["pos"].numpy()
    faces = reg["faces"].numpy()
    dists = per_opening_geo_dist(reg["pos"], reg["edge_index"], reg["node_type"])
    for name, d in dists.items():
        png = os.path.join(sanity_dir, f"{case_name}_{name}.png")
        _vis_dist(pos, faces, d, png, title=f"geo {name}")
    print(f"[ok] {case_name}: {list(dists.keys())} -> {sanity_dir}")
    if save_back:
        reg["geo_dist_per_opening"] = {k: torch.from_numpy(v).float() for k, v in dists.items()}
        torch.save(reg, reg_path)
        print(f"[ok] {case_name}: wrote geo_dist_per_opening back to {reg_path}")
    return dists


def process_all(case_root, sanity_dir, save_back=False):
    pv.start_xvfb()
    os.makedirs(sanity_dir, exist_ok=True)
    for d in sorted(glob.glob(os.path.join(case_root, "*"))):
        reg_path = os.path.join(d, "wall_mesh_registration.pt")
        if not os.path.exists(reg_path):
            continue
        try:
            process_case(reg_path, sanity_dir, save_back=save_back)
        except Exception as e:
            print(f"[fail] {os.path.basename(d)}: {e}")


if __name__ == "__main__":
    process_all(
        case_root="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged",
        sanity_dir="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/_sanity/per_opening_geo",
        save_back=False,
    )


"""
conda activate new
python -m angioflow.data.per_opening_geo


"""
