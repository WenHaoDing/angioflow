"""Read and interpret CFD volume / surface meshes (cells, faces, boundary zones)."""

from __future__ import annotations
import os
import glob

import numpy as np
import meshio
import pyvista as pv
# pv.set_jupyter_backend("static")
# pv.start_xvfb()  # for off-screen rendering on headless servers (e.g. SLURM nodes without X11 forwarding)
import torch


def fix_encas_geometry_model(encas_path):
    """Patch ensight .encas: model line + TIME filename start number from actual .scl files."""
    with open(encas_path, "r") as f:
        lines = f.readlines()
    modified = False
    in_time = False
    for i, line in enumerate(lines):
        if line.strip().startswith("model:") and "ensight_files.geo" in line:
            if "model:  2" in line:
                lines[i] = 'model:  1  "ensight_files.geo"\n'
                modified = True
            continue
        if line.strip() == "TIME":
            in_time = True
            continue
        if in_time and line.strip().startswith("filename start number:"):
            scls = sorted(glob.glob(os.path.join(os.path.dirname(encas_path), "ensight_files*.scl*")))
            if scls:
                try:
                    start = int(os.path.basename(scls[0]).split(".")[0].replace("ensight_files", ""))
                    lines[i] = f"filename start number:      {start}\n"
                    modified = True
                except ValueError:
                    pass
            continue
    if modified:
        with open(encas_path, "w") as f:
            f.writelines(lines)
            
def compute_vertex_normals(points, faces):
    k = faces.shape[1]
    cells = np.concatenate([np.full((faces.shape[0], 1), k), faces], axis=1).ravel()
    mesh = pv.PolyData(points, cells).compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
    return np.asarray(mesh.point_data["Normals"], dtype=np.float64)


def summarize_zones(mesh: meshio.Mesh):
    print(f"Total points: {mesh.points.shape[0]}")
    for i, block in enumerate(mesh.cells):
        conn = np.asarray(block.data)
        print(f"zone {i} ({block.type}): {len(conn)} faces, {np.unique(conn).size} nodes")


def get_zone_mesh(mesh: meshio.Mesh, zone_id: int):
    """
    Returns:
        points       : [n_nodes, 3]   float64, node coords (zone-local subset of mesh.points)
        edge_index   : [2, n_edges]   int64,   undirected, deduped, LOCAL node ids
        parent_index : [n_nodes]      int64,   local -> global node id (row in mesh.points)
        faces        : [n_faces, k]   int64,   face connectivity in LOCAL ids (k=3 tri, 4 quad)
    """
    conn = np.asarray(mesh.cells[zone_id].data, dtype=np.int64)
    parent_index, inverse = np.unique(conn, return_inverse=True)
    faces = inverse.reshape(conn.shape)
    points = np.asarray(mesh.points, dtype=np.float64)[parent_index]
    return points, faces, parent_index

def face_to_edge_index(faces):
    k = faces.shape[1]
    edges = np.concatenate([faces[:, [i, (i + 1) % k]] for i in range(k)], axis=0)
    edges.sort(axis=1)
    # encode as scalar for fast dedup (avoids slow row-wise np.unique)
    n = int(faces.max()) + 1
    codes = np.unique(edges[:, 0].astype(np.int64) * n + edges[:, 1].astype(np.int64))
    return np.stack([codes // n, codes % n], axis=0).astype(np.int64)

def decimate_zone_mesh(points, faces, ratio=0.25):
    k = faces.shape[1]
    cells = np.concatenate([np.full((faces.shape[0], 1), k), faces], axis=1).ravel()
    mesh = pv.PolyData(points, cells).triangulate()
    dec = mesh.decimate_pro(reduction=1 - ratio, preserve_topology=True)
    new_points = np.asarray(dec.points, dtype=np.float64)
    new_faces = dec.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
    return new_points, new_faces

def remesh_from_pointcloud(points):
    """Ball-pivot surface reconstruction from a point cloud -> (pts [N,3], faces [F,3])."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float64)))
    avg = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=avg * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    radii = [avg * r for r in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]
    m = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.triangles, dtype=np.int64)


def fill_small_holes(pts, faces, hole_size):
    """Triangulate small boundary loops (radius < hole_size); leaves larger real openings
    (e.g. inlet/outlets) untouched.

    fill_holes() only adds faces, never moves/adds vertices — but Open3D 0.19's tensor
    round-trip returns garbage vertex positions, so we discard them and keep the exact
    input `pts` (verified: vertex count and index range are unchanged).
    """
    import open3d as o3d
    legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pts), o3d.utility.Vector3iVector(faces))
    filled = o3d.t.geometry.TriangleMesh.from_legacy(legacy).fill_holes(hole_size=hole_size).to_legacy()
    new_faces = np.asarray(filled.triangles, dtype=np.int64)
    assert len(filled.vertices) == len(pts), "fill_holes unexpectedly changed vertex count"
    return pts, new_faces


def test_remesh_from_msh(msh_path, out_path, zid=0):
    """Extract zone `zid` points from .msh, ball-pivoting remesh, save side-by-side PNG."""
    import open3d as o3d
    points, _, _ = get_zone_mesh(meshio.read(msh_path), zid)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    avg = float(np.mean(pcd.compute_nearest_neighbor_distance()))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=avg * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)
    radii = [avg * r for r in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)]
    m = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    new_pts = np.asarray(m.vertices, dtype=np.float64)
    new_faces = np.asarray(m.triangles, dtype=np.int64)
    cells = np.concatenate([np.full((new_faces.shape[0], 1), 3), new_faces], axis=1).ravel()
    pl = pv.Plotter(off_screen=True, shape=(1, 2))
    pl.subplot(0, 0); pl.add_text("input points"); pl.add_points(pv.PolyData(points), point_size=2)
    pl.subplot(0, 1); pl.add_text("remeshed")
    pl.add_mesh(pv.PolyData(new_pts, cells), show_edges=True, color="lightsteelblue",
                edge_color="black", line_width=0.3)
    pl.link_views(); pl.screenshot(out_path); pl.close()
    obj_path = os.path.splitext(out_path)[0] + ".obj"
    o3d.io.write_triangle_mesh(obj_path, m)
    print(f"[ok] {os.path.basename(msh_path)}: {len(points)} pts -> {len(new_faces)} tris ({obj_path})")


def save_wall_graph(msh_path, out_path, zid=0):
    """Load .msh, extract zone `zid` (default wall), save graph + coords to a .pt file."""
    mesh = meshio.read(msh_path)
    points, faces, parent_index = get_zone_mesh(mesh, zid)
    edge_index = face_to_edge_index(faces)
    data = {
        "pos": torch.from_numpy(points).float(),                    # [N, 3] node coords
        "edge_index": torch.from_numpy(edge_index),                 # [2, E]
        "faces": torch.from_numpy(faces),                           # [F, k] local face conn
        "parent_index": torch.from_numpy(parent_index),             # [N] local -> global node id
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(data, out_path)
    return data


def decimate_fps_ballpivot(points, faces, ratio=0.5, hole_size=0.0003):
    """Uniform decimation: FPS selects a spatially even subset of the original vertices,
    ball-pivot re-triangulates, fill_small_holes patches tiny gaps.

    Because FPS picks a subset of the original vertices, orig_idx lets you slice WSS /
    node attributes directly — no interpolation needed.

    Returns:
        new_pos   [M, 3]  float64  decimated positions (M ≈ N * ratio)
        new_faces [F, 3]  int64    new triangle connectivity
        orig_idx  [M]     int64    new_pos[i] == points[orig_idx[i]]
    """
    import torch
    from pytorch3d.ops import sample_farthest_points

    n_samples = max(int(len(points) * ratio), 3)
    pts_t = torch.from_numpy(np.asarray(points, dtype=np.float32)).unsqueeze(0)  # [1, N, 3]
    _, idx = sample_farthest_points(pts_t, K=n_samples)
    orig_idx = idx[0].cpu().numpy().astype(np.int64)

    new_pos = np.asarray(points, dtype=np.float64)[orig_idx]
    _, new_faces = remesh_from_pointcloud(new_pos)
    new_pos, new_faces = fill_small_holes(new_pos, new_faces, hole_size=hole_size)
    return new_pos, new_faces, orig_idx


def visualize_zone(points, faces, out_path, show_edges=True):
    k = faces.shape[1]
    cells = np.concatenate([np.full((faces.shape[0], 1), k), faces], axis=1).ravel()
    mesh = pv.PolyData(points, cells)
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(mesh, show_edges=show_edges, color="lightsteelblue", edge_color="black", line_width=0.3)
    pl.screenshot(out_path)
    pl.close()


if __name__ == "__main__":
    mock_cfd_mesh = "/home/wenhao/AneuG-Net/angioflow_debug/case_100/mesh.msh"
    test_remesh_from_msh(mock_cfd_mesh, "remesh_test.png", zid=0)
        



"""
conda activate new
python angioflow/data/mesh.py


rsync -av  \
  /media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_yiying/ \
  /media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/

"""