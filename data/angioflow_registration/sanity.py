"""Visual sanity checks for assemble_registered_transient_data output: render
the reconstructed mesh colored by wall shear stress magnitude so the CFD->mesh
physics registration can be spot-checked by eye, and overlay all cases of a
type to confirm they share the same canonical position/orientation.
"""

import os

# Must precede the first pyvista/VTK import. Not conditioned on DISPLAY: over SSH
# DISPLAY is set (localhost:10.0) but that forwarded server offers no GLX, so VTK
# aborts creating a context. Set AF_USE_GPU=1 to opt out on a GPU desktop.
if not os.environ.get("AF_USE_GPU"):
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import pyvista as pv


def _meshes2mesh(mesh):
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    return pv.PolyData(verts, np.hstack([np.full((faces.shape[0], 1), 3), faces]))


def _iter_types(mesh_data, registered_data_list):
    """Yield (atype, mesh_data, items). Single-type builds are stored FLAT (to match
    the old pipeline's layout), multi-type builds stay keyed by aneurysm type."""
    if isinstance(registered_data_list, dict):
        for atype in mesh_data:
            yield atype, mesh_data[atype], registered_data_list[atype]
    else:
        yield None, mesh_data, registered_data_list


def save_wss_sanity_pictures(mesh_data, registered_data_list, out_dir, time_reduce="mean", cmap="turbo"):
    """For each aneurysm_type/case, render the (canonical-frame) mesh colored by
    wall shear stress magnitude (time-reduced across frames) and save a PNG to
    out_dir.

    time_reduce: "mean", "max", or a frame index (int) to visualize a single frame.
    """
    os.makedirs(out_dir, exist_ok=True)
    pv.set_jupyter_backend('static')
    try:
        pv.start_xvfb()
    except Exception:
        pass

    def _reduce(wss):
        if time_reduce == "mean":
            return wss.mean(0).numpy()
        if time_reduce == "max":
            return wss.max(0).values.numpy()
        return wss[int(time_reduce)].numpy()

    saved = []
    for atype, md, items in _iter_types(mesh_data, registered_data_list):
        for i, item in enumerate(items):
            wss_idx = item["labels"].index("wss")
            mesh = _meshes2mesh(md["subdivided_ghd_Meshes"][i])
            mesh["wss"] = _reduce(item["tensor"][..., wss_idx])

            p = pv.Plotter(off_screen=True, window_size=(700, 700))
            p.add_mesh(mesh, scalars="wss", cmap=cmap, show_edges=False, scalar_bar_args={"title": "WSS (Pa)"})
            p.add_text(f"{item['case']}  (type {atype}, dataset={item['dataset_source']})", font_size=10)
            p.show_axes()
            write_path = os.path.join(out_dir, f"{item['case']}_type{atype}_wss.png")
            p.screenshot(write_path)
            p.close()
            saved.append(write_path)

    return saved


def save_cfd_vs_registered_wss(data, case_root, ghd_root, registry_path, out_dir,
                               given_norm=None, n_subdivide=1, limit=None,
                               time_reduce="mean", cmap="turbo"):
    """Side by side, in the SAME world frame and on ONE shared colour scale:

      left   raw CFD wall mesh, coloured by wall_data.pt's own |WSS|
      right  the registered mesh, coloured by the |WSS| read back out of the
             assembled dataset

    This is the end-to-end check on the knn transfer: the left panel is the
    source of truth and never passes through the pipeline, so if the high-shear
    bands and the low-shear sac do not land in the same places on the right, the
    registration moved the physics. The registered mesh is re-placed into the
    world frame (reconstruct_case_meshes) purely so both panels share a camera --
    the dataset itself stores it in canonical space.
    """
    import torch
    from angioflow.data.angioflow_registration import MultiCanonicalGHDReconstruct, load_registry, build_ghd_dir_index
    from angioflow.data.angioflow_registration.alignment import DEFAULT_CANONICAL_ROOT
    from angioflow.data.angioflow_registration.registration import (
        reconstruct_case_meshes, resolve_case_paths, _align_norm_to_labels,
    )

    os.makedirs(out_dir, exist_ok=True)
    pv.set_jupyter_backend("static")
    try:
        pv.start_xvfb()
    except Exception:
        pass

    def _reduce(a):
        if time_reduce == "mean":
            return a.mean(0)
        if time_reduce == "max":
            return a.max(0).values
        return a[int(time_reduce)]

    registry = load_registry(registry_path)
    ghd_index = build_ghd_dir_index(ghd_root)
    recon = MultiCanonicalGHDReconstruct(canonical_root=DEFAULT_CANONICAL_ROOT, device="cpu")

    saved = []
    for _, md, items in _iter_types(data["mesh_data"], data["registered_data_list"]):
        for item in items:
            if limit is not None and len(saved) >= limit:
                break
            case = item["case"]
            case_dir = os.path.join(case_root, case)
            entry = registry[case]
            ghd_dir, _ = resolve_case_paths(case, case_dir, item["dataset_source"], ghd_index=ghd_index)
            rec = reconstruct_case_meshes(recon, ghd_dir, entry["aneurysm_type"], n_subdivide=n_subdivide)
            world = rec["subdivided_world"]

            # dataset side -- undo the normalization the builder applied, so the
            # two panels are in the same physical units and can share a colour bar
            wss_idx = item["labels"].index("wss")
            reg_wss = _reduce(item["tensor"][..., wss_idx])
            if given_norm is not None:
                mean, std = _align_norm_to_labels(given_norm, item["labels"])
                reg_wss = reg_wss * (std[wss_idx] + 1e-5) + mean[wss_idx]

            wall = torch.load(os.path.join(case_dir, "wall_data.pt"))["wall"]
            cfd_wss = _reduce(wall["wall_shear"]).squeeze(-1)

            cfd = pv.PolyData(wall["pos"].numpy() * 1000.0,
                              np.hstack([np.full((wall["faces"].shape[0], 1), 3), wall["faces"].numpy()]))
            cfd["wss"] = cfd_wss.numpy()
            reg = pv.PolyData(world.verts_packed().numpy() * 1000.0,
                              np.hstack([np.full((world.faces_packed().shape[0], 1), 3),
                                         world.faces_packed().numpy()]))
            reg["wss"] = reg_wss.numpy()

            # p1..p99 of the SOURCE data, shared, so the comparison is of pattern
            # and level rather than of two independently stretched colour maps
            clim = [float(np.percentile(cfd["wss"], 1)), float(np.percentile(cfd["wss"], 99))]
            p = pv.Plotter(off_screen=True, window_size=(1500, 750), shape=(1, 2))
            for col, (m, sub) in enumerate(((cfd, "raw CFD wall (wall_data.pt)"),
                                            (reg, "registered mesh (assembled dataset)"))):
                p.subplot(0, col)
                p.add_mesh(m, scalars="wss", cmap=cmap, clim=clim,
                           scalar_bar_args={"title": "|WSS| (Pa)"})
                p.add_text(f"{sub}\nmean {float(np.mean(m['wss'])):.3f}", position="lower_left", font_size=9)
                p.show_axes()
            p.link_views()
            p.add_text(f"{case}  ({item['dataset_source']}, {time_reduce} over frames)",
                       position="upper_edge", font_size=10)
            write_path = os.path.join(out_dir, f"cfd_vs_registered_{case}.png")
            p.screenshot(write_path)
            p.close()
            saved.append(write_path)
    return saved


_OVERLAY_PALETTE = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "yellow", "brown", "gray"]


def save_canonical_alignment_overlay(mesh_data, out_dir):
    """One picture per type with every case's canonical-frame mesh overlaid
    (translucent) -- confirms all cases share the same position/orientation
    (only differing by their true per-patient isotropic scale)."""
    os.makedirs(out_dir, exist_ok=True)
    pv.set_jupyter_backend('static')
    try:
        pv.start_xvfb()
    except Exception:
        pass

    saved = []
    per_type = mesh_data if "cases" not in mesh_data else {None: mesh_data}
    for atype, md in per_type.items():
        n_cases = len(md["cases"])
        p = pv.Plotter(off_screen=True, window_size=(700, 700))
        for i, case in enumerate(md["cases"]):
            mesh = _meshes2mesh(md["subdivided_ghd_Meshes"][i])
            color = _OVERLAY_PALETTE[i % len(_OVERLAY_PALETTE)]
            p.add_mesh(mesh, color=color, opacity=0.35, show_edges=False, label=case)
        p.add_legend()
        p.add_text(f"type {atype}: {n_cases} cases overlaid (canonical frame)", font_size=10)
        p.show_axes()
        write_path = os.path.join(out_dir, f"type{atype}_canonical_overlay.png")
        p.screenshot(write_path)
        p.close()
        saved.append(write_path)
    return saved
