"""Eyeball + quantify the knn_interpolate step: is the reconstructed GHD mesh
actually sitting on the CFD wall surface when we transfer WSS onto it?

register_wall_physics() does
    knn_interpolate(wss, pos_cfd, mesh_verts_world)
so both point sets must occupy the same WORLD-frame region. If they don't, every
mesh vertex pulls WSS from the wrong part of the vessel (or extrapolates from far
away) and the physics is silently wrong even though the canonical-frame geometry
still looks perfect.

Reports, per case:
  - centroid offset between mesh and CFD cloud
  - distance from each mesh vertex to its NEAREST CFD point (this is what
    k=3 interpolation actually averages over). For a good registration this
    should be well under the CFD mesh's own edge length.
and renders the two overlaid (mesh surface + CFD points).

conda activate new
python -m angioflow.data.angioflow_registration.check_knn_registration
python -m angioflow.data.angioflow_registration.check_knn_registration --datasets ImperialNHS --limit 3
"""
import argparse
import os

import numpy as np
import torch

from angioflow.data.angioflow_registration import MultiCanonicalGHDReconstruct, load_registry, build_ghd_dir_index
from angioflow.data.angioflow_registration.registration import (
    reconstruct_case_meshes, resolve_case_paths, DEFAULT_REGISTRY_PATH,
)
from angioflow.data.angioflow_registration.alignment import DEFAULT_CANONICAL_ROOT

CASE_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
GHD_ROOT = "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/fitting_results_NewAneuX/Final"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knn_check")


def _render(case, dataset, mesh_mm, faces, cfd_mm, nn, out_dir):
    import pyvista as pv
    surf = pv.PolyData(mesh_mm, np.hstack([np.full((faces.shape[0], 1), 3), faces]))
    surf["nn_dist_mm"] = nn
    p = pv.Plotter(off_screen=True, window_size=(900, 800), shape=(1, 2))
    p.subplot(0, 0)
    p.add_mesh(surf, color="green", opacity=0.55, label="GHD mesh (world)")
    p.add_points(cfd_mm, color="red", point_size=2.0, opacity=0.35, label="CFD wall points")
    p.add_legend()
    p.add_text(f"{case}\nknn target vs CFD cloud", font_size=9)
    p.show_axes()
    p.subplot(0, 1)
    p.add_mesh(surf, scalars="nn_dist_mm", cmap="turbo",
               scalar_bar_args={"title": "dist to nearest CFD pt (mm)"})
    p.add_text(f"per-vertex distance to CFD surface\nmean {nn.mean():.3f} mm, max {nn.max():.3f} mm", font_size=9)
    p.show_axes()
    p.link_views()
    out = os.path.join(out_dir, f"knn_{dataset}_{case}.png")
    p.screenshot(out)
    p.close()
    return out


def save_knn_figures(data, case_root, ghd_root, registry_path, out_dir, n_subdivide=1, limit=None):
    """Render the knn-registration check for every case KEPT in an assembled dataset."""
    import pyvista as pv
    from scipy.spatial import cKDTree
    pv.set_jupyter_backend("static")
    try:
        pv.start_xvfb()
    except Exception:
        pass
    os.makedirs(out_dir, exist_ok=True)

    md = data["mesh_data"]
    md = md if "cases" in md else next(iter(md.values()))
    registry = load_registry(registry_path)
    ghd_index = build_ghd_dir_index(ghd_root)
    recon = MultiCanonicalGHDReconstruct(canonical_root=DEFAULT_CANONICAL_ROOT, device="cpu")

    n = 0
    for case, source in zip(md["cases"], md["dataset_source"]):
        if limit is not None and n >= limit:
            break
        case_dir = os.path.join(case_root, case)
        entry = registry[case]
        ghd_dir, _ = resolve_case_paths(case, case_dir, source, ghd_index=ghd_index)
        rec = reconstruct_case_meshes(recon, ghd_dir, entry["aneurysm_type"], n_subdivide=n_subdivide)
        world = rec["subdivided_world"]
        mesh_mm = world.verts_packed().numpy() * 1000.0
        cfd_mm = torch.load(os.path.join(case_dir, "wall_data.pt"))["wall"]["pos"].numpy() * 1000.0
        nn, _ = cKDTree(cfd_mm).query(mesh_mm, k=1)
        _render(case, source, mesh_mm, world.faces_packed().numpy(), cfd_mm, nn, out_dir)
        n += 1
    return n


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-root", default=CASE_ROOT)
    p.add_argument("--ghd-root", default=GHD_ROOT)
    p.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--datasets", nargs="+", default=["AneuX"], choices=["AneuX", "ImperialNHS"])
    p.add_argument("--aneurysm-types", nargs="+", type=int, default=[0])
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--n-subdivide", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import pyvista as pv
    pv.set_jupyter_backend("static")
    try:
        pv.start_xvfb()
    except Exception:
        pass

    registry = load_registry(args.registry)
    ghd_index = build_ghd_dir_index(args.ghd_root)
    recon = MultiCanonicalGHDReconstruct(canonical_root=DEFAULT_CANONICAL_ROOT, device="cpu")

    print(f"{'case':28s} {'centroid off':>13s} {'NN mean':>9s} {'NN p95':>9s} {'NN max':>9s}   (mm)")
    print("-" * 78)

    done = 0
    for case in sorted(os.listdir(args.case_root)):
        if done >= args.limit:
            break
        case_dir = os.path.join(args.case_root, case)
        entry = registry.get(case)
        if not entry or entry["dataset"] not in args.datasets:
            continue
        if entry["aneurysm_type"] not in args.aneurysm_types:
            continue
        if not os.path.exists(os.path.join(case_dir, "wall_data.pt")):
            continue
        ghd_dir, _ = resolve_case_paths(case, case_dir, entry["dataset"], ghd_index=ghd_index)
        if not ghd_dir or not os.path.exists(os.path.join(ghd_dir, "ghd_coefficients.npz")):
            continue

        rec = reconstruct_case_meshes(recon, ghd_dir, entry["aneurysm_type"], n_subdivide=args.n_subdivide)
        world = rec["subdivided_world"]
        mesh_mm = world.verts_packed().numpy() * 1000.0
        faces = world.faces_packed().numpy()
        cfd_mm = torch.load(os.path.join(case_dir, "wall_data.pt"))["wall"]["pos"].numpy() * 1000.0

        # nearest CFD point for every mesh vertex -- what knn_interpolate(k=3) averages
        from scipy.spatial import cKDTree
        nn, _ = cKDTree(cfd_mm).query(mesh_mm, k=1)
        off = np.linalg.norm(mesh_mm.mean(0) - cfd_mm.mean(0))
        print(f"{case:28s} {off:10.3f}    {nn.mean():9.3f} {np.percentile(nn,95):9.3f} {nn.max():9.3f}")

        surf = pv.PolyData(mesh_mm, np.hstack([np.full((faces.shape[0], 1), 3), faces]))
        surf["nn_dist_mm"] = nn
        p = pv.Plotter(off_screen=True, window_size=(900, 800), shape=(1, 2))
        p.subplot(0, 0)
        p.add_mesh(surf, color="green", opacity=0.55, label="GHD mesh (world)")
        p.add_points(cfd_mm, color="red", point_size=2.0, opacity=0.35, label="CFD wall points")
        p.add_legend()
        p.add_text(f"{case}\nknn target vs CFD cloud", font_size=9)
        p.show_axes()
        p.subplot(0, 1)
        p.add_mesh(surf, scalars="nn_dist_mm", cmap="turbo",
                   scalar_bar_args={"title": "dist to nearest CFD pt (mm)"})
        p.add_text("per-vertex distance to CFD surface", font_size=9)
        p.show_axes()
        p.link_views()
        out = os.path.join(args.out_dir, f"knn_{entry['dataset']}_{case}.png")
        p.screenshot(out)
        p.close()
        done += 1

    print(f"\nSaved {done} figure(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
