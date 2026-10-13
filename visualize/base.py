
from __future__ import annotations

import glob
import os

import numpy as np
import torch
import ensightreader
from scipy.spatial import cKDTree
from torch_geometric.data import Data, Dataset

import pyvista as pv
pv.start_xvfb()

from angioflow.data.mesh import decimate_zone_mesh, face_to_edge_index, fix_encas_geometry_model, compute_vertex_normals

PHYSICS_KEYS = ["x_wall_shear", "y_wall_shear", "z_wall_shear"]


def save_wss_snapshots(case_root, out_dir, timestep=40, decimate_ratio=0.25,
                       case_pattern="case_*", clim=None):
    os.makedirs(out_dir, exist_ok=True)
    for case_dir in sorted(glob.glob(os.path.join(case_root, case_pattern))):
        name = os.path.basename(case_dir)
        encas = os.path.join(case_dir, "Results", "ensight_files.encas")
        try:
            fix_encas_geometry_model(encas)
            case = ensightreader.read_case(encas)
            geofile = case.get_geometry_model()
            part = geofile.get_part_by_name("surface3")
            with open(geofile.file_path, "rb") as fp:
                coords = np.asarray(part.read_nodes(fp), dtype=np.float64)
                faces = np.concatenate(
                    [np.asarray(b.read_connectivity(fp), dtype=np.int64) - 1 for b in part.element_blocks],
                    axis=0,
                )
            new_pts, new_faces = decimate_zone_mesh(coords, faces, ratio=decimate_ratio)
            _, idx = cKDTree(coords).query(new_pts, k=1)

            cols = []
            for key in PHYSICS_KEYS:
                v = case.get_variable(key, timestep=timestep)
                with open(v.file_path, "rb") as fp:
                    cols.append(np.asarray(v.read_node_data(fp, part.part_id), dtype=np.float64))
            wss_mag = np.linalg.norm(np.stack(cols, axis=1)[idx], axis=1)

            k = new_faces.shape[1]
            cells = np.concatenate([np.full((new_faces.shape[0], 1), k), new_faces], axis=1).ravel()
            mesh = pv.PolyData(new_pts, cells)
            mesh["wss"] = wss_mag

            pl = pv.Plotter(off_screen=True)
            pl.add_mesh(mesh, scalars="wss", cmap="coolwarm", clim=clim, show_scalar_bar=True)
            pl.screenshot(os.path.join(out_dir, f"{name}_t{timestep}.png"))
            pl.close()
            print(f"[ok] {name}: wss range {wss_mag.min():.3f}..{wss_mag.max():.3f}")
        except Exception as e:
            print(f"[skip] {name}: {e}")


def save_geo_distance(coord_cfd, faces_cfd, geo_dist, ghd_Meshes, out_path, cmap="jet"):
    """Color the CFD mesh by geodesic distance and overlay the GHD mesh (translucent).

    matplotlib (no VTK) to avoid the Open3D/VTK conflict that segfaults in-process.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfd = np.asarray(coord_cfd, dtype=np.float64)
    fc = np.asarray(faces_cfd, dtype=np.int64)
    if geo_dist is None:
        gd = np.full(cfd.shape[0], np.inf, dtype=np.float64)
    else:
        gd = np.asarray(geo_dist.detach().cpu() if hasattr(geo_dist, "detach") else geo_dist, dtype=np.float64)
    ghd_v = ghd_Meshes.verts_packed().detach().cpu().numpy()
    ghd_f = ghd_Meshes.faces_packed().detach().cpu().numpy()

    face_val = gd[fc].mean(axis=1)             # per-triangle scalar
    face_val[~np.isfinite(face_val)] = np.nan  # disconnected -> shown as 'bad' color

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("lightgray")

    coll = ax.plot_trisurf(cfd[:, 0], cfd[:, 1], cfd[:, 2], triangles=fc,
                           cmap=cmap_obj, linewidth=0, antialiased=False)
    coll.set_array(face_val)
    coll.set_clim(np.nanmin(face_val), np.nanmax(face_val))
    fig.colorbar(coll, ax=ax, shrink=0.6, label="geodesic distance")

    # GHD mesh overlay as wireframe (transparent faces, visible edges) to see overlap clearly
    ax.plot_trisurf(ghd_v[:, 0], ghd_v[:, 1], ghd_v[:, 2], triangles=ghd_f,
                    color=(0, 0, 0, 0), edgecolor="black", linewidth=0.2)

    ax.set_box_aspect((1, 1, 1))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_ghd_cfd_overlay(cfd_pts, ghd_pts, out_path, point_size=2.0):
    """Render CFD vs GHD point clouds in one view to inspect alignment.

    Uses matplotlib (not VTK) to avoid an Open3D/VTK GL-context conflict that
    segfaults when rendering in the same process as align_GHD_coord_to_CFD.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfd = np.asarray(cfd_pts.detach().cpu() if hasattr(cfd_pts, "detach") else cfd_pts, dtype=np.float64)
    ghd = np.asarray(ghd_pts.detach().cpu() if hasattr(ghd_pts, "detach") else ghd_pts, dtype=np.float64)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(cfd[:, 0], cfd[:, 1], cfd[:, 2], s=point_size, c="red", label="CFD", depthshade=False)
    ax.scatter(ghd[:, 0], ghd[:, 1], ghd[:, 2], s=point_size, c="blue", label="GHD", depthshade=False)
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

