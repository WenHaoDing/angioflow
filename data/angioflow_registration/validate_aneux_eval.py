"""Independent validation of a dataset built by build_aneux_type0_eval.py.

Read-only: it never touches the raw case data or the assembled .pth, it only
writes images/CSV into a subfolder of OUTPUT_ROOT.

Two questions, each answered against an external reference rather than against
the builder's own code path:

TEST 1 -- is the stored mesh the right SIZE?
    mesh_data stores the GHD surface in the CANONICAL frame (metres): canonical
    shape scaled by this case's true physical size (s * s_can * 1e-3), with no
    rotation/translation. Converting to mm and applying the alignment gives

        world_mm = canonical_mm @ R_eq.T + t_eq

    -- a PURE RIGID transform, because the canonical mesh already carries
    s_equivalent (see alignment.ghd_world_alignment). That is what makes this a
    size test: if the stored size were wrong (a missing 1e-3, a dropped s_can),
    no rotation+translation could put it on the segmentation surface. We overlay
    against the case's own segmentation .obj (millimetres, CFD world frame) and
    report the residual isotropic scale, which must come out ~1.0.

TEST 2 -- is the WSS TANGENT to the wall?
    Wall shear stress is a traction in the surface plane, so wss . n = 0 up to
    discretisation. A non-zero dot means the WSS sitting on a vertex did not
    come from that point of the wall.

    NOTE both channels must be brought into the SAME frame before dotting, and
    the tensor as written does not satisfy that -- see the report this prints.
    We therefore evaluate three ways: as-stored, after rotating WSS into the
    canonical frame (wss @ R_eq), and on the raw CFD data itself (wall_data.pt
    WSS vs normals of the CFD wall mesh) as the achievable floor.

conda activate new
PYTHONPATH=.:./angioflow python -m angioflow.data.angioflow_registration.validate_aneux_eval
"""
import argparse
import os

# Must be set before VTK/libGL is loaded by the first pyvista import. Not
# conditioned on DISPLAY: over SSH, DISPLAY is set (localhost:10.0) but the
# forwarded server offers no GLX, so VTK aborts creating a context there just as
# it does with no DISPLAY at all. Set AF_USE_GPU=1 to opt out on a GPU desktop.
if not os.environ.get("AF_USE_GPU"):
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .build_aneux_type0_eval import CASE_ROOT, GHD_ROOT, OUTPUT_ROOT
from .registration import build_ghd_dir_index, load_ghd_affine, MM_TO_M

STEADY_NORM = ("/media/yaplab2/wd8tb/wenhao/datasets/AneuG_Flow_stable/AneuG_processed/"
               "assembled_registered_steady_data_1k.pth")
# AneuX cases have no geomagic_processed.obj -- its absence is what marks them as
# AneuX in the registry. merged_reconstruction_remeshed.obj is the same thing for
# them: the segmentation surface in millimetres, in the CFD world frame.
SEG_OBJ_NAMES = ("geomagic_processed.obj", "merged_reconstruction_remeshed.obj")
WSS_CH = ("wss_x", "wss_y", "wss_z")
NRM_CH = ("x_normal", "y_normal", "z_normal")


def load_norm(path):
    st = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    n = st["tensor_norm"]
    return {k: (n["mean"].flatten()[i].item(), n["std"].flatten()[i].item())
            for i, k in enumerate(st["label"])}


def denorm(t, labels, norm, names):
    """Pull `names` out of a stored (normalized) frame, back in physical units."""
    li = {k: i for i, k in enumerate(labels)}
    out = t[:, [li[n] for n in names]].clone().float()
    if norm is not None:
        for j, k in enumerate(names):
            if k in norm:
                m, s = norm[k]
                out[:, j] = out[:, j] * (s + 1e-5) + m
    return out


def seg_obj_path(case_dir):
    for name in SEG_OBJ_NAMES:
        p = os.path.join(case_dir, name)
        if os.path.exists(p):
            return p
    return None


def rigid_to_world_mm(canonical_m, affine):
    """Canonical metres -> CFD world millimetres. Rotation+translation ONLY: the
    canonical verts already carry s_equivalent, so no scaling happens here."""
    R_eq = affine["R_frame"].T @ affine["R"]
    t_eq = affine["s_can"] * affine["t_vec"] @ affine["R_frame"] + affine["neck_pt"]
    return np.asarray(canonical_m, dtype=np.float64) / MM_TO_M @ R_eq.T + t_eq


def residual_scale(src, dst_tree, dst_pts):
    """Extra isotropic scale that would best fit `src` onto its nearest points in
    `dst`. 1.0 = size already correct; 1e-3 / 1e3 = a unit bug."""
    _, nn = dst_tree.query(src, k=1)
    a = src - src.mean(0)
    b = dst_pts[nn] - dst_pts[nn].mean(0)
    return float((a * b).sum() / (a * a).sum())


def cfd_wall_tangency(case_dir):
    """Control: |cos| between the RAW CFD WSS and the RAW CFD wall normals. This
    is the floor our registration could ever reach -- it is not exactly 0 because
    vertex normals of a discrete mesh are not the true surface normal."""
    w = torch.load(os.path.join(case_dir, "wall_data.pt"))["wall"]
    pos = w["pos"].double().numpy()
    faces = w["faces"].long().numpy()
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)                      # area-weighted face normal
    vn = np.zeros_like(pos)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30
    wss = torch.cat([w["x_wall_shear"], w["y_wall_shear"], w["z_wall_shear"]], -1)[0].double().numpy()
    wss /= np.linalg.norm(wss, axis=1, keepdims=True) + 1e-30
    return np.abs((vn * wss).sum(1))


_XVFB = [False]


def _headless():
    """Render into a virtual framebuffer (see the LIBGL note at the top of this
    file). Done once, and not conditioned on DISPLAY -- a forwarded X display is
    present over SSH but cannot serve GLX."""
    import pyvista as pv
    if not _XVFB[0] and not os.environ.get("AF_USE_GPU"):
        pv.start_xvfb()
        _XVFB[0] = True


def render_overlay(mesh_mm, faces, seg, title, out_path):
    import pyvista as pv
    _headless()
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    m = pv.PolyData(mesh_mm, cells)
    p = pv.Plotter(off_screen=True, window_size=(1600, 800), shape=(1, 2))
    for col, cpos in enumerate(("xy", "xz")):
        p.subplot(0, col)
        p.add_mesh(seg, color="lightgray", opacity=0.35, label="segmentation .obj")
        p.add_mesh(m, color="crimson", style="wireframe", line_width=1,
                   label="dataset mesh (canonical, rigidly placed)")
        p.camera_position = cpos
        p.add_legend(size=(0.42, 0.12))
        p.show_axes()
    p.add_text(title, position="upper_edge", font_size=10)
    p.screenshot(out_path)
    p.close()


def render_tangency(mesh_mm, faces, cos_stored, cos_rot, title, out_path):
    import pyvista as pv
    _headless()
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    p = pv.Plotter(off_screen=True, window_size=(1600, 800), shape=(1, 2))
    for col, (c, sub) in enumerate(((cos_stored, "as stored in .pth"),
                                    (cos_rot, "after wss @ R_eq (canonical frame)"))):
        m = pv.PolyData(mesh_mm, cells)
        m["|cos|"] = c
        p.subplot(0, col)
        p.add_mesh(m, scalars="|cos|", cmap="inferno", clim=[0, 1],
                   scalar_bar_args={"title": "|cos(wss, n)|"})
        p.add_text(f"{sub}\nmean={c.mean():.3f}", position="lower_left", font_size=9)
        p.show_axes()
    p.link_views()
    p.add_text(title, position="upper_edge", font_size=10)
    p.screenshot(out_path)
    p.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--assembled", default=os.path.join(
        OUTPUT_ROOT, "assembled_registered_data_AneuX_type0.pth"))
    p.add_argument("--case-root", default=CASE_ROOT)
    p.add_argument("--ghd-root", default=GHD_ROOT)
    p.add_argument("--steady-norm", default=STEADY_NORM,
                   help="source of the mean/std the builder normalized with")
    p.add_argument("--out-dir", default=None, help="default: <OUTPUT_ROOT>/validation_<tag>")
    p.add_argument("--n-overlay", type=int, default=4, help="cases to render (test 1)")
    p.add_argument("--n-tangency", type=int, default=0,
                   help="cases to score for tangency; 0 = all")
    p.add_argument("--frame", type=int, default=0, help="time frame to score")
    return p.parse_args()


def main():
    args = parse_args()
    tag = os.path.basename(args.assembled).replace("assembled_registered_data_", "").replace(".pth", "")
    out_dir = args.out_dir or os.path.join(OUTPUT_ROOT, f"validation_{tag}")
    os.makedirs(out_dir, exist_ok=True)

    data = torch.load(args.assembled, map_location="cpu", mmap=True, weights_only=False)
    md, rl = data["mesh_data"], data["registered_data_list"]
    if "cases" not in md:
        raise SystemExit("per-type nested build; pass a single-type dataset")
    meshes = md["subdivided_ghd_Meshes"]
    faces = meshes.faces_list()[0].numpy()
    ghd_index = build_ghd_dir_index(args.ghd_root)
    labels = rl[0]["labels"]

    # is the stored tensor normalized? compare its x/y/z against the mesh verts
    xyz0 = rl[0]["tensor"][args.frame][:, :3]
    normalized = (xyz0 - meshes.verts_list()[0]).abs().max().item() > 1e-6
    norm = load_norm(args.steady_norm) if normalized else None
    print(f"[data] {len(md['cases'])} cases, tensor {'NORMALIZED' if normalized else 'raw'}"
          f"{' (denormalizing with ' + os.path.basename(args.steady_norm) + ')' if normalized else ''}")
    print(f"[out ] {out_dir}\n")

    # ---------------------------------------------------------------- TEST 1
    print("=" * 78)
    print("TEST 1  mesh size: dataset mesh rigidly placed onto the segmentation .obj")
    print("=" * 78)
    print(f"{'case':10s} {'ref .obj':34s} {'nn_mean':>8s} {'nn_p95':>8s} {'resid_scale':>12s}")
    size_rows = []
    for i in range(min(args.n_overlay, len(md["cases"]))):
        case = md["cases"][i]
        case_dir = os.path.join(args.case_root, case)
        obj = seg_obj_path(case_dir)
        if obj is None or case not in ghd_index:
            print(f"{case:10s} -- no segmentation .obj / no GHD fit, skipped")
            continue
        import pyvista as pv
        seg = pv.read(obj)
        seg_pts = np.asarray(seg.points, dtype=np.float64)
        world_mm = rigid_to_world_mm(meshes.verts_list()[i].numpy(), load_ghd_affine(ghd_index[case]))
        tree = cKDTree(seg_pts)
        nn, _ = tree.query(world_mm, k=1)
        s_res = residual_scale(world_mm, tree, seg_pts)
        size_rows.append((case, os.path.basename(obj), nn.mean(), np.percentile(nn, 95), s_res))
        print(f"{case:10s} {os.path.basename(obj):34s} {nn.mean():8.3f} "
              f"{np.percentile(nn, 95):8.3f} {s_res:12.4f}")
        render_overlay(world_mm, faces, seg,
                       f"{case}   nn_mean={nn.mean():.3f} mm   residual scale={s_res:.4f}",
                       os.path.join(out_dir, f"size_overlay_{case}.png"))
    print("\ndistances in mm; residual scale ~1.0 means the stored size is correct\n")

    # ---------------------------------------------------------------- TEST 2
    print("=" * 78)
    print("TEST 2  WSS tangency: |cos(wss, n)| should be ~0 on a correct registration")
    print("=" * 78)
    n_t = len(rl) if args.n_tangency == 0 else min(args.n_tangency, len(rl))
    print(f"{'case':10s} {'as-stored':>10s} {'wss@R_eq':>10s} {'raw CFD':>10s}")
    rows = []
    for i in range(n_t):
        case = rl[i]["case"]
        if case not in ghd_index:
            continue
        t = rl[i]["tensor"][args.frame]
        n = F.normalize(denorm(t, labels, None, NRM_CH), dim=-1)   # normals pass through the norm
        w = denorm(t, labels, norm, WSS_CH)
        a = load_ghd_affine(ghd_index[case])
        R_eq = a["R_frame"].T @ a["R"]
        w_can = torch.from_numpy(w.numpy().astype(np.float64) @ R_eq).float()
        c_stored = (n * F.normalize(w, dim=-1)).sum(-1).abs().numpy()
        c_rot = (n * F.normalize(w_can, dim=-1)).sum(-1).abs().numpy()
        raw = cfd_wall_tangency(os.path.join(args.case_root, case))
        rows.append((case, c_stored.mean(), c_rot.mean(), raw.mean()))
        print(f"{case:10s} {c_stored.mean():10.4f} {c_rot.mean():10.4f} {raw.mean():10.4f}")
        if i < args.n_overlay and case in ghd_index:
            world_mm = rigid_to_world_mm(meshes.verts_list()[i].numpy(), a)
            render_tangency(world_mm, faces, c_stored, c_rot,
                            f"{case}   WSS tangency (raw CFD floor = {raw.mean():.3f})",
                            os.path.join(out_dir, f"wss_tangency_{case}.png"))

    csv = os.path.join(out_dir, "validation_report.csv")
    with open(csv, "w") as f:
        f.write("case,cos_as_stored,cos_after_rotation,cos_raw_cfd\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]:.5f},{r[2]:.5f},{r[3]:.5f}\n")
        f.write("\ncase,seg_obj,nn_mean_mm,nn_p95_mm,residual_scale\n")
        for r in size_rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.5f}\n")

    m = np.array([[r[1], r[2], r[3]] for r in rows])
    print("\n" + "=" * 78)
    print(f"MEAN over {len(rows)} cases   as-stored={m[:,0].mean():.4f}   "
          f"wss@R_eq={m[:,1].mean():.4f}   raw CFD floor={m[:,2].mean():.4f}")
    print(f"report -> {csv}\nimages -> {out_dir}")


if __name__ == "__main__":
    main()
