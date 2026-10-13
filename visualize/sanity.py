"""Sanity-check visualisation: ground-truth vs predicted WSS magnitude on the mesh."""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import torch


def _triangles_from_edge_index(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """Recover triangle faces from the symmetric edge_index of a triangular mesh.

    For each undirected edge (i, j), any node k that is adjacent to both i and j
    and satisfies k > j forms a unique triangle [i, j, k].  This perfectly recovers
    the original faces as long as the mesh is triangular (which ours always is).
    """
    mask = edge_index[0] < edge_index[1]
    src, dst = edge_index[0, mask].tolist(), edge_index[1, mask].tolist()

    neighbors: list[set] = [set() for _ in range(n_nodes)]
    for i, j in zip(src, dst):
        neighbors[i].add(j)
        neighbors[j].add(i)

    triangles = []
    for i, j in zip(src, dst):
        for k in neighbors[i]:
            if k > j and k in neighbors[j]:
                triangles.append([i, j, k])

    return np.array(triangles, dtype=np.int64) if triangles else np.empty((0, 3), dtype=np.int64)


def _make_surface_mesh(pos: np.ndarray, edge_index: np.ndarray):
    """Build a pyvista PolyData surface mesh by recovering triangles from edge_index."""
    import pyvista as pv
    faces = _triangles_from_edge_index(edge_index, len(pos))
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    return pv.PolyData(pos, cells)


@torch.no_grad()
def save_wss_sanity(model, loader, waveform, device, stats, out_dir, epoch,
                    max_wss: float = 20.0, n_samples: int = 3):
    """Render GT vs predicted WSS magnitude as surface meshes for *n_samples* test graphs.

    Saves out_dir/sanity_epoch_{epoch:04d}_s{i}.png  (one file per sample).
    Node positions are shown in normalised coordinates — the 3-D shape is preserved.
    WSS magnitude is clamped to [0, max_wss] Pa/s to match the log-mag loss range.
    """
    os.makedirs(out_dir, exist_ok=True)

    y_mean = stats["y_mean"].cpu().numpy()   # [3]
    y_std  = stats["y_std"].cpu().numpy()    # [3]

    model.eval()
    samples = []
    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch, waveform)
        if isinstance(pred, tuple):
            # query-style models return (pred, query_idx); model.eval() above means
            # query_idx is always None here (full resolution — see AngioWSSQueryField)
            pred, _ = pred

        for g in range(batch.num_graphs):
            node_mask  = batch.batch == g
            first_node = node_mask.nonzero(as_tuple=False)[0, 0].item()

            edge_in_g = node_mask[batch.edge_index[0]]
            ei_local  = batch.edge_index[:, edge_in_g].cpu().numpy() - first_node

            pos_np = batch.pos[node_mask].cpu().numpy()
            gt, pd = batch.y[node_mask], pred[node_mask]
            gt_denorm = gt.cpu().numpy() * y_std + y_mean   # [N, 3] Pa/s, global xyz
            pd_denorm = pd.cpu().numpy() * y_std + y_mean   # [N, 3] Pa/s, global xyz
            gt_mag    = np.linalg.norm(gt_denorm, axis=1).clip(0, max_wss)
            pred_mag  = np.linalg.norm(pd_denorm, axis=1).clip(0, max_wss)
            err       = np.abs(gt_mag - pred_mag)
            label     = f"{batch.version[g]}/{batch.case_id[g]}"

            samples.append((pos_np, ei_local, gt_mag, pred_mag, err, label))
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break
    model.train()

    if not samples:
        print("[sanity] no samples — skipping", flush=True)
        return

    # VTK's off-screen rendering corrupts stdout/stderr at the OS fd level.
    # Save both fds before any pyvista call and restore them unconditionally after.
    _stdout_fd = os.dup(1)
    _stderr_fd = os.dup(2)
    _py_stdout = sys.stdout
    _py_stderr = sys.stderr
    try:
        import pyvista as pv
        pv.start_xvfb()

        for i, (pos, ei, gt_mag, pred_mag, err, label) in enumerate(samples):
            vmin = min(gt_mag.min(), pred_mag.min())
            vmax = max(gt_mag.max(), pred_mag.max())

            cols = [
                (gt_mag,   "GT (Pa/s)",      "coolwarm", vmin, vmax),
                (pred_mag, "Pred (Pa/s)",    "coolwarm", vmin, vmax),
                (err,      "|Error| (Pa/s)", "Oranges",  0.0,  err.max()),
            ]

            pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 750))
            pl.link_views()

            for col, (vals, title, cmap, lo, hi) in enumerate(cols):
                pl.subplot(0, col)
                mesh = _make_surface_mesh(pos, ei)
                mesh.point_data["wss"] = vals
                pl.add_mesh(mesh, scalars="wss", cmap=cmap, clim=[lo, hi],
                            show_scalar_bar=True,
                            scalar_bar_args={"title": title, "vertical": True})
                pl.add_title(f"{label}  {title}", font_size=11)

            pl.camera_position = "iso"
            out_path = os.path.join(out_dir, f"sanity_epoch_{epoch:04d}_s{i}.png")
            pl.screenshot(out_path)
            pl.close()
    finally:
        # Restore OS-level fds unconditionally so training stdout survives.
        os.dup2(_stdout_fd, 1);  os.close(_stdout_fd)
        os.dup2(_stderr_fd, 2);  os.close(_stderr_fd)
        sys.stdout = _py_stdout
        sys.stderr = _py_stderr

    for i in range(len(samples)):
        out_path = os.path.join(out_dir, f"sanity_epoch_{epoch:04d}_s{i}.png")
        print(f"[sanity] saved {out_path}", flush=True)
