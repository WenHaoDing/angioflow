"""Extract Fluent CFD results via ensightreader: blood (interior volume) + wall zones."""


"""
Format:

blood_data.pt  — dict saved by extract_blood_data()
    pos            [N, 3]    float32   node positions from geometry
    pos_frame      [N, 3]    float32   node coords read from t=0 variables
    cells          [C, k]    int64     volumetric cell connectivity (0-indexed, k=4 for tets)
    <BLOOD_KEYS>   [T, N, 1] float32   per-key time series (T=n_frames, N=nodes)
        viscosity_lam, dp_dt, dp_dz, dp_dy, dp_dx,
        dz_velocity_dy, dy_velocity_dz, dz_velocity_dx,
        dx_velocity_dz, dy_velocity_dx, dx_velocity_dy,
        dz_velocity_dz, dy_velocity_dy, dx_velocity_dx,
        z_velocity, y_velocity, x_velocity, absolute_pressure

wall_data.pt  — dict[zone_name -> dict] saved by extract_wall_data()
    zones: "wall", "inlet", "outlet_0", "outlet_1"
    each zone:
        pos            [N, 3]    float32   node positions from geometry
        pos_frame      [N, 3]    float32   node coords read from t=0 variables
        faces          [F, k]    int64     surface face connectivity (0-indexed)
    wall zone only:
        z_wall_shear   [T, N, 1] float32
        y_wall_shear   [T, N, 1] float32
        x_wall_shear   [T, N, 1] float32
"""
from __future__ import annotations

import glob
import os
from itertools import combinations

import numpy as np
import torch
import ensightreader

from angioflow.data.mesh import face_to_edge_index, fix_encas_geometry_model


BLOOD_KEYS = [
    "viscosity_lam", "dp_dt", "dp_dz", "dp_dy", "dp_dx",
    "dz_velocity_dy", "dy_velocity_dz", "dz_velocity_dx",
    "dx_velocity_dz", "dy_velocity_dx", "dx_velocity_dy",
    "dz_velocity_dz", "dy_velocity_dy", "dx_velocity_dx",
    "z_velocity", "y_velocity", "x_velocity",
    "dynamic_pressure", "absolute_pressure",
]

WALL_KEYS = [
    "z_wall_shear", "y_wall_shear", "x_wall_shear", "wall_shear",
]

COORD_KEYS = ("x_coordinate", "y_coordinate", "z_coordinate")

WALL_PARTS = [
    ("wall", "surface3"),
    ("inlet", "surface4"),
    ("outlet_0", "surface5"),
    ("outlet_1", "surface6"), # 
]

_METADATA_FILES = [
    "flowsplit_ratio.txt",
    "geomagic_processed.obj",
    "ghd_coefficients.npz",
    "ghd_forward_fusion_info.npz",
    "ghd_merged_reconstruction.obj",
    "ghd_reconstructed.obj",
    "ghd_smoothed_reconstruction.obj",
    "inlet_centroids.csv",
    "landmarks.npz",
    "metrics.json",
    "branch_ranking.npy",
]

def _read_part_geometry(geofile, part):
    """Return (pos [N,3], cells [C,k]) with connectivity converted to 0-indexed."""
    with open(geofile.file_path, "rb") as fp:
        pos = np.asarray(part.read_nodes(fp), dtype=np.float64)
        cells = np.concatenate(
            [np.asarray(b.read_connectivity(fp), dtype=np.int64) - 1 for b in part.element_blocks],
            axis=0,
        )
    return pos, cells


def _cells_to_edge_index(cells):
    """All-pairs edges per cell (tet -> 6 edges), deduped and undirected. -> [2, E]."""
    k = cells.shape[1]
    edges = np.concatenate([cells[:, list(p)] for p in combinations(range(k), 2)], axis=0)
    edges.sort(axis=1)
    return np.unique(edges, axis=0).T.astype(np.int64)


def _read_frames(case, part, keys, n_frames):
    """Read each `key` over `n_frames` timesteps -> {key: [T, N, 1]}."""
    out = {}
    for key in keys:
        cols = []
        for t in range(n_frames):
            v = case.get_variable(key, timestep=t)
            with open(v.file_path, "rb") as fp:
                arr = np.asarray(v.read_node_data(fp, part.part_id), dtype=np.float64)
            cols.append(torch.from_numpy(arr).view(-1, 1).float())
        out[key] = torch.stack(cols, dim=0)
    return out


def _read_pos_frame(case, part):
    """Read xyz coord variables at t=0 -> [N, 3]. Sanity check vs static geometry."""
    cols = []
    for key in COORD_KEYS:
        v = case.get_variable(key, timestep=0)
        with open(v.file_path, "rb") as fp:
            cols.append(np.asarray(v.read_node_data(fp, part.part_id), dtype=np.float64))
    return torch.from_numpy(np.stack(cols, axis=1)).float()


def _print_data_shapes(label, data):
    print(f"[{label}]")
    if isinstance(data, dict) and isinstance(next(iter(data.values())), dict):
        for zone_name, zone in data.items():
            print(f"  {zone_name}")
            for k, v in zone.items():
                print(f"    {k:<25s} {tuple(v.shape)}")
    else:
        for k, v in data.items():
            print(f"  {k:<25s} {tuple(v.shape)}")


def extract_blood_data(encas_path, n_frames=80):
    case = ensightreader.read_case(encas_path)
    geofile = case.get_geometry_model()
    part = geofile.get_part_by_name("blood")
    pos, cells = _read_part_geometry(geofile, part)
    data = {
        "pos": torch.from_numpy(pos).float(),
        "pos_frame": _read_pos_frame(case, part),            # [N, 3] coords from t=0 vars
        "cells": torch.from_numpy(cells),                    # [C, k] volumetric cell conn
    }
    data.update(_read_frames(case, part, BLOOD_KEYS, n_frames))
    _print_data_shapes("blood", data)
    return data


def extract_wall_data(encas_path, n_frames=80):
    """Per-zone dict; only the wall zone carries WSS/coord time series."""
    case = ensightreader.read_case(encas_path)
    geofile = case.get_geometry_model()
    out = {}
    for zone_name, part_name in WALL_PARTS:
        part = geofile.get_part_by_name(part_name)
        if part is None:
            continue
        pos, faces = _read_part_geometry(geofile, part)
        zone = {
            "pos": torch.from_numpy(pos).float(),
            "pos_frame": _read_pos_frame(case, part),        # [N, 3] coords from t=0 vars
            "faces": torch.from_numpy(faces),                # [F, k]
        }
        if zone_name == "wall":
            zone.update(_read_frames(case, part, WALL_KEYS, n_frames))
        out[zone_name] = zone
    _print_data_shapes("wall", out)
    return out





def copy_case_metadata(case_dir, out_dir):
    """Copy metadata files from <case_dir>/Results/ into out_dir if they exist."""
    import shutil
    results_dir = os.path.join(case_dir, "Results")
    candidates = list(_METADATA_FILES) + [
        os.path.basename(p)
        for p in glob.glob(os.path.join(results_dir, "output*"))
    ]
    copied, missing = [], []
    for fname in candidates:
        src = os.path.join(results_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
            copied.append(fname)
        else:
            missing.append(fname)
    print(f"  metadata  copied={len(copied)}  missing={len(missing)}")


def process_case(case_dir, save_root, n_frames=80):
    """Write `<save_root>/<case_name>/blood_data.pt` and `wall_data.pt`."""
    encas = os.path.join(case_dir, "Results", "ensight_files.encas")
    if not os.path.exists(encas):
        print(f"[skip] {os.path.basename(case_dir)}: no .encas")
        return False
    fix_encas_geometry_model(encas)
    out_dir = os.path.join(save_root, os.path.basename(case_dir))
    os.makedirs(out_dir, exist_ok=True)
    blood_path = os.path.join(out_dir, "blood_data.pt")
    wall_path = os.path.join(out_dir, "wall_data.pt")
    if not os.path.exists(blood_path):
        torch.save(extract_blood_data(encas, n_frames), blood_path)
    if not os.path.exists(wall_path):
        torch.save(extract_wall_data(encas, n_frames), wall_path)
    copy_case_metadata(case_dir, out_dir)
    return True


def export_encase_to_pt(case_root, save_root, n_frames=80, case_pattern="case_*", redo=False):
    """
    🔥🔥🔥
    Export all cases under `case_root` matching `case_pattern` to `save_root` as .pt files.
    Skips cases where both .pt files already exist unless redo=True.
    """
    os.makedirs(save_root, exist_ok=True)
    for case_dir in sorted(glob.glob(os.path.join(case_root, case_pattern))):
        name = os.path.basename(case_dir)
        out_dir = os.path.join(save_root, name)
        if not redo and os.path.exists(os.path.join(out_dir, "blood_data.pt")) \
                    and os.path.exists(os.path.join(out_dir, "wall_data.pt")):
            print(f"[skip] {name}: already done")
            continue
        try:
            if process_case(case_dir, save_root, n_frames=n_frames):
                print(f"[ok] {name}")
        except Exception as e:
            print(f"[fail] {name}: {e}")

"""
Maintaince code session.

"""

_ZONE_STYLE = {
    "surface3": ("steelblue",  0.9,  "wall"),
    "surface4": ("red",        0.9,  "inlet"),
    "surface5": ("yellow",     0.9,  "outlet_0"),
    "surface6": ("black",      0.9,  "outlet_1"),
}
_VTK_TYPE = {3: 5, 4: 10, 8: 12}   # tri, tet, hex


def _to_polydata(pos, cells):
    import pyvista as pv
    k = cells.shape[1]
    flat = np.concatenate([np.full((len(cells), 1), k), cells], axis=1).ravel()
    return pv.PolyData(pos, flat)


def _to_ugrid(pos, cells):
    import pyvista as pv
    k = cells.shape[1]
    flat = np.concatenate([np.full((len(cells), 1), k), cells], axis=1).ravel()
    return pv.UnstructuredGrid(flat, np.full(len(cells), _VTK_TYPE[k]), pos)


def plot_case_mesh(case_dir):
    """
    🔥🔥
    Show a 1x2 interactive window: left=surface zones coloured by type, right=blood outer surface."""
    import pyvista as pv

    encas = os.path.join(case_dir, "Results", "ensight_files.encas")
    if not os.path.exists(encas):
        print(f"[skip] no .encas in {case_dir}")
        return

    fix_encas_geometry_model(encas)
    case = ensightreader.read_case(encas)
    geofile = case.get_geometry_model()

    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 800))
    pl.subplot(0, 0)
    pl.add_text("Surface zones", font_size=12)
    pl.subplot(0, 1)
    pl.add_text("Blood volume", font_size=12)

    for _, part in geofile.parts.items():
        pos, cells = _read_part_geometry(geofile, part)
        if part.part_name == "blood":
            pl.subplot(0, 1)
            grid = _to_ugrid(pos, cells)
            sliced = grid.slice(normal="z", origin=grid.center)
            pl.add_mesh(sliced, color="lightcoral",
                        show_edges=True, edge_color="white", line_width=0.5)
        elif part.part_name in _ZONE_STYLE:
            color, opacity, label = _ZONE_STYLE[part.part_name]
            pl.subplot(0, 0)
            pl.add_mesh(_to_polydata(pos, cells), color=color, opacity=opacity,
                        show_edges=True, edge_color="dimgray", line_width=0.3, label=label)

    pl.subplot(0, 0)
    pl.add_legend(size=(0.25, 0.25))
    pl.link_views()
    pl.show()


def print_parts(case_root, case_pattern="case_*"):
    """Print all parts and face counts for every case under case_root.

    Expects:  <case_root>/<case>/<Results>/ensight_files.encas
    """
    for case_dir in sorted(glob.glob(os.path.join(case_root, case_pattern))):
        encas = os.path.join(case_dir, "Results", "ensight_files.encas")
        if not os.path.exists(encas):
            print(f"{os.path.basename(case_dir)}: [no .encas]")
            continue
        print(f"\n=== {os.path.basename(case_dir)} ===")
        try:
            fix_encas_geometry_model(encas)
            case = ensightreader.read_case(encas)
            geofile = case.get_geometry_model()
            for part_id, part in geofile.parts.items():
                _, cells = _read_part_geometry(geofile, part)
                print(f"  part {part_id:3d}  {part.part_name:<30s}  faces={len(cells)}")
        except Exception as e:
            print(f"  [error] {e}")



def test_wall_coords_in_blood(encas_path):
    """Check that every wall zone node exists exactly in the blood node set."""
    from scipy.spatial import cKDTree

    fix_encas_geometry_model(encas_path)
    case = ensightreader.read_case(encas_path)
    geofile = case.get_geometry_model()

    blood_part = geofile.get_part_by_name("blood")
    blood_pos, _ = _read_part_geometry(geofile, blood_part)
    tree = cKDTree(blood_pos)

    for zone_name, part_name in WALL_PARTS:
        part = geofile.get_part_by_name(part_name)
        if part is None:
            continue
        wall_pos, _ = _read_part_geometry(geofile, part)
        dist, _ = tree.query(wall_pos, k=1, workers=-1)
        n_exact = int(np.sum(dist == 0.0))
        n_near  = int(np.sum(dist < 1e-6))
        print(f"  {zone_name:<12s}  nodes={len(wall_pos)}  "
              f"exact={n_exact}  near(<1e-6)={n_near}  max_dist={dist.max():.3e}")


if __name__ == "__main__":
    # case_root = "E:/AngioFlow/batch_2"
    # export_encase_to_pt(case_root, "H:/AngioFlow/processed", n_frames=80, case_pattern="case_*")
    ghd_chk = np.load("H:/AneuG-Flow/stable_4263/checkpoint.npy", allow_pickle=True).item()
    print(ghd_chk.keys())
    print(ghd_chk['ghd'].shape)
    print(ghd_chk['mean_ghd'].shape)
    print(ghd_chk['norm_canonical'])
"""
conda activate new
python -m angioflow.data.fluent_export


=== case_202 ===
Warning: unsupported section (E:/AngioFlow/batch_2\case_202\Results\ensight_files.encas:63), skipping
  part   2  blood                           faces=1695546
  part   3  surface3                        faces=43483
  part   4  surface4                        faces=1097
  part   5  surface5                        faces=834
  part   6  surface6                        faces=924

=== case_204 ===
Warning: unsupported section (E:/AngioFlow/batch_2\case_204\Results\ensight_files.encas:63), skipping
  part   2  blood                           faces=3883753
  part   3  surface3                        faces=64088
  part   4  surface4                        faces=3420
  part   5  surface5                        faces=2446

"""
