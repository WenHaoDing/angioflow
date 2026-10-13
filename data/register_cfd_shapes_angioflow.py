"""
Register CFD shapes for the AngioFlow dataset (exported via fluent_export.py).

AngioFlow wall_data.pt already has named zones — no remeshing or opening detection needed.
Zone nodes are matched back to the wall mesh via nearest-neighbour to assign node_type.

Focused area: loaded from clipped_reconstruction.ply (mm, rotated/shifted) found via
preprocess_dir_list, then ICP-aligned to the CFD mesh.

Registration file format (same as AneuG-Flow):
wall_mesh_registration.pt = {
    "pos":           [N, 3]   float32  wall node positions (metres)
    "edge_index":    [2, E]   int64    undirected graph edges
    "faces":         [F, 3]   int64    triangle connectivity
    "node_type":     [N]      int64    0=wall, 1=focused, 2=inlet, 3=outlet_0, 4=outlet_1
    "flow_split":    [2]      float32  outlet flow split ratios (or zeros if unavailable)
    "aneurysm_type": scalar   int64    aneurysm morphology type (looked up from
                                       endpoints_manual.npy in preprocess_dir_list,
                                       see load_aneurysm_type -- NOT a constant; a case
                                       is skipped, not silently mislabelled, if its
                                       endpoints_manual.npy can't be found anywhere in
                                       preprocess_dir_list)
    "geo_dist":      [N]      float32  geodesic dist to focused region (zeros if unavailable)
}

---------------------------------------------------------------------------
MANUAL-FIX WORKFLOW for cases the automatic focused-mesh alignment rejects
---------------------------------------------------------------------------
A case is skipped (not silently mislabelled) when find_ply_path finds no
clipped_reconstruction.ply, or align_focused_to_cfd's RANSAC+ICP can't reach
fitness >= min_fitness / mean_surf_dist <= max_mean_dist against it. Every such
case is logged to bad_cases_path (<sanity_dir>/bad_alignment_cases.txt).

1. RUN. process_all() finishes; export_failed_wall_meshes(bad_cases_path,
   dataset_root, sanity_dir=sanity_dir) then writes each skipped case's own
   CFD wall surface (wall_data.pt's pos+faces, already in METRES, already in
   the CFD frame -- no external file, no alignment needed to get here) to
       <sanity_dir>/0000_failed_wall_meshes/<case_name>.obj

2. FIX, BY HAND. Open each <case_name>.obj in a mesh editor (Blender, MeshLab,
   ...) and DELETE every face except the aneurysm-focused region -- i.e. clip
   the vessel wall down to just the sac/dome, the same region clipped_recon-
   struction.ply would normally mark. Do NOT import or align a different
   segmentation mesh here: because this .obj IS the CFD wall mesh itself, the
   clipped-out faces are already exactly where they belong, so the later
   automatic re-alignment (step 4) is applied to a mesh that is a literal
   subset of its own target and will trivially pass with fitness ~= 1.0 --
   redrawing the geometry from scratch would reintroduce the alignment problem
   this workflow exists to avoid. Export each result as
       <sanity_dir>/failed_wall_meshes_manually_fixed/<case_name>.<ext>
   (.obj/.ply/anything open3d reads; one file per case, same case_name stem).

3. IMPORT. import_manually_fixed_meshes(fixed_dir, dataset_root) converts each
   file to .ply and writes it to
       <dataset_root>/<case_name>/manually_fixed_clipped_reconstruction.ply
   This already happens automatically at the top of __main__ on every run, so
   simply dropping fixed files into failed_wall_meshes_manually_fixed/ and
   re-running the script is enough -- no separate step needed.

4. RE-RUN (redo=True, or just this case_pattern). find_ply_path checks for
   manually_fixed_clipped_reconstruction.ply FIRST, ahead of searching
   preprocess_dir_list, and treats it as already in metres (scale=1.0, no
   mm->m conversion) -- unlike clipped_reconstruction.ply from
   preprocess_dir_list, which is mm and needs *0.001. It still goes through
   align_focused_to_cfd (see step 2 for why that should be a near no-op) and
   can therefore still fail the fitness/distance thresholds if the manual clip
   was altered rather than just trimmed; if it fails again, the case re-enters
   bad_cases_path for another pass.
"""

from __future__ import annotations
import os
import glob

import numpy as np
import torch
import open3d as o3d
import pyvista as pv
pv.start_xvfb()
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree

from angioflow.data.mesh import face_to_edge_index
from angioflow.data.align import compute_geo_distance
from angioflow.data.register_cfd_shapes_aneug_flow import largest_connected_focused
from angioflow.data.registration_sanity import SANITY_ROOT, case_sanity_dir, save_registration_sanity


# ---------------------------------------------------------------------------
# Aneurysm type lookup
# ---------------------------------------------------------------------------

def load_aneurysm_type(case_name, preprocess_dir_list):
    """The case's true morphology type, from endpoints_manual.npy -- searched for
    across preprocess_dir_list in order, first hit wins.

    Returns None if no source has it. Callers must NOT default to a fixed value on
    None: an audit of this script's past output found EVERY ONE of the existing
    wall_mesh_registration.pt files held aneurysm_type=0, because a previous
    __main__ hardcoded that value for every case regardless of truth -- 238 of 447
    checkable cases (53%) were actually type 1 or 2. A case with no discoverable
    type should be skipped, exactly like a case with no focused mesh.
    """
    for root in preprocess_dir_list:
        path = os.path.join(root, case_name, "endpoints_manual.npy")
        if os.path.exists(path):
            try:
                return int(np.load(path, allow_pickle=True).item()["aneurysm_type"])
            except Exception as exc:
                print(f"  [warn] {case_name}: unreadable {path}: {exc}")
    return None


# ---------------------------------------------------------------------------
# Focused mesh lookup
# ---------------------------------------------------------------------------

def find_ply_path(case_dir, preprocess_dir_list):
    """Find the focused mesh PLY for a case, in priority order:
    1. manually_fixed_clipped_reconstruction.ply in case folder — already in metres
    2. clipped_reconstruction.ply in any preprocess dir — in mm, needs *0.001
    Returns (path, scale) where scale converts vertices to metres, or (None, None).
    """
    manual = os.path.join(case_dir, "manually_fixed_clipped_reconstruction.ply")
    if os.path.exists(manual):
        return manual, 1.0  # already in metres

    name = os.path.basename(case_dir)
    for root in preprocess_dir_list:
        for fname in ("clipped_reconstruction.ply", "merged_reconstruction_remeshed_clipped.obj"):
            candidate = os.path.join(root, name, fname)
            if os.path.exists(candidate):
                return candidate, 0.001  # mm -> m

    return None, None


# ---------------------------------------------------------------------------
# Focused mesh: load PLY + ICP align to CFD
# ---------------------------------------------------------------------------

def load_focused_ply(ply_path, scale=0.001):
    """Load PLY and apply scale to convert vertices to metres."""
    mesh = o3d.io.read_triangle_mesh(ply_path)
    pts = np.asarray(mesh.vertices, dtype=np.float64) * scale
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    return pts, faces


def align_focused_to_cfd(focused_pts, cfd_pts, max_dist=0.005):
    """
    RANSAC+FPFH global registration → point-to-plane ICP refinement.
    Global step handles arbitrary rotations; ICP refines to sub-voxel accuracy.
    Returns (aligned_pts [N,3], transformation [4,4], fitness, mean_surf_dist).
    mean_surf_dist: mean distance from each aligned focused vertex to its nearest CFD vertex.
    """
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(focused_pts))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cfd_pts))

    extent = src.get_axis_aligned_bounding_box().get_extent()
    voxel_size = float(np.linalg.norm(extent)) / 40.0

    src_down = src.voxel_down_sample(voxel_size)
    tgt_down = tgt.voxel_down_sample(voxel_size)

    normal_param = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    src_down.estimate_normals(normal_param)
    tgt_down.estimate_normals(normal_param)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max_dist, max_nn=30))

    feat_param = o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(src_down, feat_param)
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tgt_down, feat_param)

    ransac_dist = voxel_size * 1.5
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    icp = o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    R, t = icp.transformation[:3, :3], icp.transformation[:3, 3]
    aligned_pts = (R @ focused_pts.T).T + t

    mean_surf_dist = float(cKDTree(cfd_pts).query(aligned_pts, k=1)[0].mean())
    return aligned_pts, icp.transformation, icp.fitness, mean_surf_dist


def get_focused_mesh_angioflow(case_dir, preprocess_dir_list, cfd_pts,
                        min_fitness=0.95, max_mean_dist=0.00025):
    """
    Look up clipped_reconstruction.ply, convert mm->m, align to CFD mesh.
    Rejects if ICP fitness < min_fitness OR mean surface distance > max_mean_dist (2 mm).
    Returns (Meshes, reason) — Meshes is None if not found / alignment rejected.
    """
    name = os.path.basename(case_dir)
    ply_path, ply_scale = find_ply_path(case_dir, preprocess_dir_list)

    if ply_path is None:
        reason = "PLY not found (no manually_fixed_clipped_reconstruction.ply or clipped_reconstruction.ply)"
        print(f"  [warn] {name}: {reason}")
        return None, reason

    ply_source = os.path.basename(ply_path)
    focused_pts, focused_faces = load_focused_ply(ply_path, scale=ply_scale)
    aligned_pts, T, fitness, mean_dist = align_focused_to_cfd(focused_pts, cfd_pts)
    print(f"  [align] {name} ({ply_source})  fitness={fitness:.4f}  "
          f"mean_dist={mean_dist*1000:.2f}mm  T_t={T[:3,3].round(4)}")

    if fitness < min_fitness:
        reason = f"ICP fitness {fitness:.4f} < {min_fitness}"
        print(f"  [warn] {name}: {reason}")
        return None, reason

    if mean_dist > max_mean_dist:
        reason = f"mean surface dist {mean_dist*1000:.2f}mm > {max_mean_dist*1000:.0f}mm"
        print(f"  [warn] {name}: {reason}")
        return None, reason

    return Meshes(
        verts=[torch.from_numpy(aligned_pts).float()],
        faces=[torch.from_numpy(focused_faces).long()],
    ), None


# ---------------------------------------------------------------------------
# Node matching: map zone nodes -> wall mesh node indices
# ---------------------------------------------------------------------------

def match_zone_to_wall(wall_pts, zone_pts, tol=1e-6):
    """Return wall node indices nearest to each zone node.

    Boundary zones (inlet/outlet) are full caps; only their rim nodes coincide with
    the wall mesh's boundary loop, so most zone nodes are expected to go unmatched.
    """
    dists, idx = cKDTree(wall_pts).query(zone_pts, k=1)
    return idx[dists < tol]


def build_node_type(n_pts, focused_mask, inlet_idx, outlet0_idx, outlet1_idx):
    labels = torch.zeros(n_pts, dtype=torch.long)
    if focused_mask is not None:
        labels[focused_mask] = 1
    labels[torch.from_numpy(inlet_idx)] = 2
    labels[torch.from_numpy(outlet0_idx)] = 3
    labels[torch.from_numpy(outlet1_idx)] = 4
    return labels


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def alignment_vis(cfd_pts, cfd_faces, focused_pts, focused_faces, out_path):
    """Overlay sanity: CFD wall surface (translucent blue) + aligned focused mesh (solid red)."""
    cfd_cells = np.concatenate([np.full((cfd_faces.shape[0], 1), 3), cfd_faces], axis=1).ravel()
    focused_cells = np.concatenate([np.full((focused_faces.shape[0], 1), 3), focused_faces], axis=1).ravel()
    pl = pv.Plotter(off_screen=True, window_size=(2400, 1800))
    pl.add_mesh(pv.PolyData(cfd_pts, cfd_cells), color="steelblue", opacity=0.35, show_edges=False)
    pl.add_mesh(pv.PolyData(focused_pts, focused_cells), color="tomato", opacity=0.85, show_edges=False)
    pl.add_legend([["CFD wall", "steelblue"], ["focused (aligned)", "tomato"]])
    pl.screenshot(out_path); pl.close()


def _to_polydata(pos, faces):
    cells = np.concatenate([np.full((faces.shape[0], 1), faces.shape[1]), faces], axis=1).ravel()
    return pv.PolyData(pos, cells)


def sanity_check_case(case_dir, out_dir, fps=10):
    """Mesh PNG + WSS GIF + velocity GIF straight from the extracted wall_data.pt /
    blood_data.pt, independent of (and a stronger check than) the registration
    sanity renders above: those show node_type/geo_dist, derived quantities that can
    look fine even if the raw extracted fields are wrong. This looks at the raw
    fields themselves -- does the WSS/velocity time series actually look physical.

    Self-contained (no dependency on fluent_export.py's or AneuG's plot_case_mesh,
    which read a different source, the raw .encas Fluent case): everything here
    comes from the same two .pt files process_case itself reads.
    """
    name = os.path.basename(case_dir)
    os.makedirs(out_dir, exist_ok=True)

    wall_data = torch.load(os.path.join(case_dir, "wall_data.pt"), weights_only=False)
    blood_data = torch.load(os.path.join(case_dir, "blood_data.pt"), weights_only=False)
    wall = wall_data["wall"]

    pos_w = wall["pos"].numpy()
    faces_w = wall["faces"].numpy()
    wall_mesh = _to_polydata(pos_w, faces_w)

    # --- mesh PNG: plain wall surface, geometry-only sanity check ---
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    pl.add_mesh(wall_mesh, color="lightgrey", show_edges=True, edge_color="dimgray", line_width=0.3)
    pl.screenshot(os.path.join(out_dir, "mesh.png"))
    pl.close()

    # WSS magnitude  [T, N_wall]
    wss_mag = torch.sqrt(
        wall["x_wall_shear"][:, :, 0] ** 2 +
        wall["y_wall_shear"][:, :, 0] ** 2 +
        wall["z_wall_shear"][:, :, 0] ** 2
    ).numpy()
    T = wss_mag.shape[0]
    wss_clim = [0, float(np.percentile(wss_mag, 99))]

    # velocity magnitude  [T, N_blood]
    vel_mag = torch.sqrt(
        blood_data["x_velocity"][:, :, 0] ** 2 +
        blood_data["y_velocity"][:, :, 0] ** 2 +
        blood_data["z_velocity"][:, :, 0] ** 2
    ).numpy()
    vel_clim = [0, float(np.percentile(vel_mag, 99))]

    blood_pcd = pv.PolyData(blood_data["pos"].numpy())
    blood_pcd.point_data["vel"] = vel_mag[0]

    # --- WSS GIF ---
    wall_mesh.point_data["wss"] = wss_mag[0]
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    pl.open_gif(os.path.join(out_dir, "wss.gif"), fps=fps)
    pl.add_mesh(wall_mesh, scalars="wss", clim=wss_clim, cmap="jet",
                scalar_bar_args={"title": "WSS mag"})
    pl.camera.zoom(2.5)
    for t in range(T):
        wall_mesh.point_data["wss"] = wss_mag[t]
        pl.render()
        pl.write_frame()
    pl.close()

    # --- velocity GIF (transparent point cloud) ---
    pl = pv.Plotter(off_screen=True, window_size=(900, 700))
    pl.open_gif(os.path.join(out_dir, "vel.gif"), fps=fps)
    pl.add_points(blood_pcd, scalars="vel", clim=vel_clim, cmap="jet",
                  point_size=2, opacity=0.3,
                  scalar_bar_args={"title": "Vel mag (m/s)"})
    pl.camera.zoom(2.5)
    for t in range(T):
        blood_pcd.point_data["vel"] = vel_mag[t]
        pl.render()
        pl.write_frame()
    pl.close()

    print(f"[saved] {name}: mesh.png, wss.gif, vel.gif -> {out_dir}")


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def process_case(case_dir, preprocess_dir_list, sanity_dir=None, match_tol=1e-6):
    name = os.path.basename(case_dir)

    # checked first, before any of the expensive mesh work below: a case with no
    # discoverable type must not get silently mislabelled with a placeholder
    aneurysm_type = load_aneurysm_type(name, preprocess_dir_list)
    if aneurysm_type is None:
        reason = "aneurysm_type not found in any preprocess_dir_list source"
        print(f"[skip] {name}: {reason}")
        return reason

    if sanity_dir is not None:
        os.makedirs(sanity_dir, exist_ok=True)
    print(f"[...] {name}: loading wall_data.pt")
    wall_data = torch.load(os.path.join(case_dir, "wall_data.pt"), weights_only=False)

    if "faces" not in wall_data["wall"]:
        print(f"[skip] {name}: wall_data.pt has no mesh connectivity")
        return "no mesh connectivity"

    pts = wall_data["wall"]["pos"].numpy().astype(np.float64)
    faces = wall_data["wall"]["faces"].numpy().astype(np.int64)
    print(f"[...] {name}: wall mesh {len(pts)} verts, {len(faces)} faces")

    edge_index = face_to_edge_index(faces)

    empty_idx = np.zeros(0, dtype=np.int64)
    inlet_idx   = match_zone_to_wall(pts, wall_data["inlet"]["pos"].numpy(), tol=match_tol)
    outlet0_idx = (match_zone_to_wall(pts, wall_data["outlet_0"]["pos"].numpy(), tol=match_tol)
                   if "outlet_0" in wall_data else empty_idx)
    outlet1_idx = (match_zone_to_wall(pts, wall_data["outlet_1"]["pos"].numpy(), tol=match_tol)
                   if "outlet_1" in wall_data else empty_idx)
    print(f"[...] {name}: inlet={len(inlet_idx)}, out0={len(outlet0_idx)}, out1={len(outlet1_idx)}")

    # Focused area via PLY + ICP
    print(f"[...] {name}: finding focused mesh")
    focused_Meshes, align_reason = get_focused_mesh_angioflow(case_dir, preprocess_dir_list, pts)

    if focused_Meshes is None:
        print(f"[skip] {name}: focused mesh not found ({align_reason})")
        return align_reason

    focused_pts = focused_Meshes.verts_packed().detach().cpu().numpy()
    focused_faces_np = focused_Meshes.faces_packed().cpu().numpy()
    fv = focused_pts[focused_faces_np]
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
    geo_dist = compute_geo_distance(pts, faces, focused_Meshes)

    case_dir_sanity = case_sanity_dir(name, root=sanity_dir) if sanity_dir is not None else None
    if case_dir_sanity is not None:
        alignment_vis(pts, faces, focused_pts, focused_faces_np,
                      os.path.join(case_dir_sanity, "alignment.png"))

    node_type = build_node_type(len(pts), focused_mask, inlet_idx, outlet0_idx, outlet1_idx)

    flowsplit_path = os.path.join(case_dir, "flowsplit_ratio.txt")
    flow_split = (torch.from_numpy(np.loadtxt(flowsplit_path).reshape(-1).astype(np.float32))
                  if os.path.exists(flowsplit_path) else torch.zeros(2))

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
    print(f"[ok] {name}: saved wall_mesh_registration.pt")

    if case_dir_sanity is not None:
        save_registration_sanity(name, pts, faces, node_type, geo_dist, out_dir=case_dir_sanity)
        try:
            sanity_check_case(case_dir, case_dir_sanity)
        except Exception as exc:
            # raw-field sanity (mesh/WSS/velocity gifs) must never fail a case that
            # otherwise registered correctly -- report and move on
            print(f"  [warn] {name}: sanity_check_case failed: {exc}")

    return None


def process_all(dataset_root, preprocess_dir_list,
                case_pattern="*", sanity_dir=None,
                max_cases=None, bad_cases_path=None, redo=False):
    import traceback
    case_dirs = sorted(
        d for d in glob.glob(os.path.join(dataset_root, case_pattern))
        if os.path.isdir(d)
    )
    if max_cases is not None:
        case_dirs = case_dirs[:max_cases]

    bad_cases = []
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)
        if not redo and os.path.exists(os.path.join(case_dir, "wall_mesh_registration.pt")):
            print(f"[skip] {name}: already done")
            continue
        try:
            reason = process_case(case_dir, preprocess_dir_list, sanity_dir=sanity_dir)
            if reason is not None:
                bad_cases.append((name, reason))
        except Exception as e:
            print(f"[fail] {name}: {e}")
            traceback.print_exc()
            bad_cases.append((name, f"exception: {e}"))

    if bad_cases_path is not None:
        with open(bad_cases_path, "w") as f:
            for name, reason in bad_cases:
                f.write(f"{name}\t{reason}\n")
        print(f"\n[summary] {len(bad_cases)}/{len(case_dirs)} cases skipped/failed -> {bad_cases_path}")

    return bad_cases


def export_failed_wall_meshes(bad_cases_path, dataset_root, out_dir=None, sanity_dir=None):
    """Read failed-alignment case names from `bad_cases_path` and export each one's
    CFD wall surface as `<out_dir>/<case_name>.obj` for manual editing.
    `out_dir` defaults to `<sanity_dir>/0000_failed_wall_meshes`.

    Step 1 of the MANUAL-FIX WORKFLOW in this module's docstring -- read that
    before editing these .obj files: the fix is to CLIP this mesh down to the
    focused region, not to replace it with different geometry.
    """
    if out_dir is None:
        assert sanity_dir is not None, "either out_dir or sanity_dir must be given"
        out_dir = os.path.join(sanity_dir, "0000_failed_wall_meshes")
    os.makedirs(out_dir, exist_ok=True)

    with open(bad_cases_path, "r") as f:
        names = [line.split("\t", 1)[0].strip() for line in f if line.strip()]

    for name in names:
        wall_pt = os.path.join(dataset_root, name, "wall_data.pt")
        if not os.path.exists(wall_pt):
            print(f"[skip] {name}: no wall_data.pt at {wall_pt}")
            continue
        wd = torch.load(wall_pt, weights_only=False)
        if "wall" not in wd or "faces" not in wd["wall"]:
            print(f"[skip] {name}: missing wall mesh")
            continue
        pts = wd["wall"]["pos"].numpy().astype(np.float64)
        faces = wd["wall"]["faces"].numpy().astype(np.int64)
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(pts),
            o3d.utility.Vector3iVector(faces),
        )
        out_path = os.path.join(out_dir, f"{name}.obj")
        o3d.io.write_triangle_mesh(out_path, m)
        print(f"[ok] {name} -> {out_path}")


def import_manually_fixed_meshes(fixed_dir, dataset_root):
    """Copy `<fixed_dir>/<case_name>.<ext>` into
    `<dataset_root>/<case_name>/manually_fixed_clipped_reconstruction.ply`.
    Uses o3d read+write so source can be .obj/.ply/etc — output is always .ply.

    Step 3 of the MANUAL-FIX WORKFLOW in this module's docstring. find_ply_path
    treats the file this writes as already in METRES (scale=1.0) -- unlike
    clipped_reconstruction.ply from preprocess_dir_list, which is millimetres.
    """
    for path in sorted(glob.glob(os.path.join(fixed_dir, "*"))):
        name = os.path.splitext(os.path.basename(path))[0]
        case_dir = os.path.join(dataset_root, name)
        if not os.path.isdir(case_dir):
            print(f"[skip] {name}: case dir not found in {dataset_root}")
            continue
        m = o3d.io.read_triangle_mesh(path)
        if len(m.vertices) == 0:
            print(f"[skip] {name}: empty / unreadable mesh at {path}")
            continue
        dst = os.path.join(case_dir, "manually_fixed_clipped_reconstruction.ply")
        o3d.io.write_triangle_mesh(dst, m)
        print(f"[ok] {name} -> {dst}")


if __name__ == "__main__":
    dataset_root = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
    sanity_dir   = SANITY_ROOT

    preprocess_dir_list = [
        '/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS',
        # AneuX_stable, not the bare "AneuX" sibling: AneuX_stable is the more
        # complete, consolidated source (344 vs 325 entries) -- in particular it
        # has the 20 cases whose geometry folders were missing from "AneuX" and had
        # to be tracked down into AneuX_stable separately (see
        # angioflow/data/fix_registry_aneurysm_type.py's docstring for the story).
        '/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_stable',
    ]
    bad_cases_path = os.path.join(sanity_dir, "bad_alignment_cases.txt")
    import_manually_fixed_meshes(
        os.path.join(sanity_dir, "failed_wall_meshes_manually_fixed"),
        dataset_root,
    )
    process_all(dataset_root, preprocess_dir_list,
                sanity_dir=sanity_dir,
                bad_cases_path=bad_cases_path, redo=True)
    export_failed_wall_meshes(bad_cases_path, dataset_root, sanity_dir=sanity_dir)


"""
conda activate new
python -m angioflow.data.register_cfd_shapes_angioflow

"""
