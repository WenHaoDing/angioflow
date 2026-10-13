"""Sanity-visualise the precomputed spatial gradients in blood_data.pt.

Both dataset versions already store, per frame per interior point, the full
velocity-gradient tensor and the pressure gradient:

  velocity: d{x,y,z}_velocity_d{x,y,z}   (9 components, dvi/dxj)
  pressure: dp_d{x,y,z}                   (3 components)

We render a thin slab cross-section coloured by each of the 12 fields (diverging
map centred at 0, symmetric clim at the 98th abs-percentile) at the peak-flow
frame, so the extraction can be eyeballed: dvi/dxj should show wall-shear
structure (large near the wall, ~0 in the core), dp/dx.. should be smooth.

One figure per case (4x3 panels), saved next to the graph sanity renders.
"""
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree  # noqa: F401  (kept parallel to sibling scripts)

OUT_DIR = os.path.join(os.path.dirname(__file__), "graph_sanity")

CASES = [
    ("angioflow", "B2o6VFmWVW_aneurysm1",
     "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/B2o6VFmWVW_aneurysm1"),
    ("aneug_flow", "stable_246",
     "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data/stable_246"),
]

VGRAD = [[f"d{c}_velocity_d{x}" for x in "xyz"] for c in "xyz"]   # rows dvi, cols dxj
PGRAD = ["dp_dx", "dp_dy", "dp_dz"]
VEL = ["x_velocity", "y_velocity", "z_velocity"]


def load_pos(blood, version):
    if version == "angioflow":
        return blood["pos"].numpy().astype(np.float64)
    # AneuG-Flow: static coords stored per-frame; frame 0 is representative
    return np.stack([blood[f"{a}_coordinate"][0, :, 0].numpy() for a in "xyz"],
                    axis=1).astype(np.float64)


def peak_frame(blood, n_probe=4000, seed=0):
    """Frame with the largest mean speed (probed on a random point subset)."""
    N = blood[VEL[0]].shape[1]
    idx = np.random.default_rng(seed).choice(N, size=min(n_probe, N), replace=False)
    idx = np.sort(idx)
    sq = 0.0
    for k in VEL:
        v = blood[k][:, idx, 0].numpy()          # [T, n_probe]
        sq = sq + v ** 2
    return int(np.sqrt(sq).mean(axis=1).argmax())


def render_case(version, cid, case_dir):
    import pyvista as pv
    pv.start_xvfb()
    t0 = time.time()
    blood = torch.load(os.path.join(case_dir, "blood_data.pt"),
                       weights_only=False, mmap=True)
    pos = load_pos(blood, version)
    t = peak_frame(blood)
    print(f"[{version}/{cid}] {len(pos)} pts, peak-flow frame {t} "
          f"({time.time()-t0:.1f}s)")

    # slab perpendicular to the thinnest axis => broadest cross-section
    ext = pos.max(0) - pos.min(0)
    axis = int(np.argmin(ext))
    thick = 0.05 * ext[axis]
    in_slab = np.abs(pos[:, axis] - np.median(pos[:, axis])) < thick
    sp = pos[in_slab]
    print(f"[{version}/{cid}] slab axis {'xyz'[axis]}: {in_slab.sum()} pts")

    fields = [(f"dv{'xyz'[r]}/d{'xyz'[c]}", VGRAD[r][c]) for r in range(3) for c in range(3)]
    fields += [(f"dp/d{'xyz'[c]}", PGRAD[c]) for c in range(3)]

    pl = pv.Plotter(shape=(4, 3), off_screen=True, window_size=(1800, 2200))
    for n, (label, key) in enumerate(fields):
        val = blood[key][t, :, 0].numpy()[in_slab]
        lim = np.percentile(np.abs(val), 98)
        lim = lim if lim > 0 else 1.0
        pl.subplot(n // 3, n % 3)
        cloud = pv.PolyData(sp)
        cloud["g"] = val
        pl.add_mesh(cloud, scalars="g", cmap="RdBu_r", clim=[-lim, lim],
                    point_size=4.0, render_points_as_spheres=True,
                    scalar_bar_args=dict(title=label, n_labels=3, fmt="%.1e"))
        pl.add_text(label, position="upper_left", font_size=9)
        pl.camera_position = ["yz", "xz", "xy"][axis]
        print(f"  {label:9s} ({key}): "
              f"[{val.min():+.2e}, {val.max():+.2e}] clim +/-{lim:.2e}")

    path = os.path.join(OUT_DIR, f"{cid}_{version}_gradients.png")
    pl.screenshot(path)
    pl.close()
    print(f"[ok] {path}  ({time.time()-t0:.1f}s)\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for version, cid, case_dir in CASES:
        render_case(version, cid, case_dir)


if __name__ == "__main__":
    main()


"""
conda activate new
python -m angioflow.data.test_gradients
"""
