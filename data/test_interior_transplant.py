"""Comprehensive post-transplant sanity check of datasets_interior.py.

Processes one AngioFlow and one AneuG-Flow case through the REAL dataset code path, then:

  1. numeric: apply CenterAndRandomRotate to the (wall, interior) pair and verify
       vel   -> R v
       grad_p-> R g
       grad_v-> R G R^T           (independent matmul, catches index bugs)
     plus a discriminating physical check: the convective acceleration
     (v·∇)v = G v must rotate as R (G v) — true for R G R^T with R v, but NOT
     for wrong index orders, so it pins down the exact tensor law.

  2. images (angioflow/data/graph_sanity/):
     <cid>_<ver>_align.png     wall points (grey) + interior pool (coloured by
                               wall_dist), 3 views — must occupy the same place.
     <cid>_<ver>_gradrot.png   interior coloured by rotation-INVARIANTS ∇·v and
                               ‖∇v‖_F, before vs after rotation — colours must be
                               identical (only the cloud orientation changes).
     <cid>_<ver>_graph.png     kept graph after the InteriorSubsampleGraph
                               transform (0.25), slab cross sections.
"""
import os

import numpy as np
import torch

from angioflow.data.datasets_interior import AngioFlowWallInteriorDataset
from angioflow.transform import CenterAndRandomRotate, InteriorSubsampleGraph
from angioflow.transform.rotate import _random_rotation_matrices

OUT_DIR = os.path.join(os.path.dirname(__file__), "graph_sanity")
SAVE_DIR = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full"
SCRATCH_INT = "/tmp/claude-1000/interior_transplant_check"
FRAME = 11    # peak-flow frame (from test_gradients.py)

CASES = [
    ("angioflow", "B2o6VFmWVW_aneurysm1",
     "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"),
    ("aneug_flow", "stable_246",
     "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data"),
]


def rel_err(a, b):
    return (a - b).abs().max().item() / b.abs().max().clamp_min(1e-12).item()


def check_rotation(wall, interior):
    """Verify the rotation transform's vector/tensor laws against manual formulas
    computed with the SAME R (reproduced by seeding)."""
    orig = interior.clone()
    seed = 12345
    torch.manual_seed(seed)
    w2, i2 = CenterAndRandomRotate(rotate=True)((wall.clone(), interior.clone()))
    torch.manual_seed(seed)
    R = _random_rotation_matrices(1, dtype=orig.pos.dtype)[0]           # same R

    exp_vel = orig.vel @ R.T
    exp_gp = orig.grad_p @ R.T
    G = orig.grad_v.view(-1, 3, 3)
    exp_gv = torch.einsum("ip,kpq,jq->kij", R, G, R).reshape(-1, 9)

    # discriminating check: convective acceleration (v·∇)v = G v rotates as R(Gv)
    conv_orig = torch.einsum("kij,kj->ki", G, orig.vel)
    conv_rot = torch.einsum("kij,kj->ki", i2.grad_v.view(-1, 3, 3), i2.vel)
    exp_conv = conv_orig @ R.T

    checks = {
        "vel = R v":            rel_err(i2.vel, exp_vel),
        "grad_p = R g":         rel_err(i2.grad_p, exp_gp),
        "grad_v = R G R^T":     rel_err(i2.grad_v, exp_gv),
        "(v.grad)v = R(Gv)":    rel_err(conv_rot, exp_conv),
        "||G|| invariant":      rel_err(i2.grad_v.view(-1, 3, 3).norm(dim=(1, 2)),
                                        G.norm(dim=(1, 2))),
        "div invariant":        rel_err(i2.grad_v.view(-1, 3, 3).diagonal(dim1=1, dim2=2).sum(1),
                                        G.diagonal(dim1=1, dim2=2).sum(1)),
    }
    print("  rotation-law relative errors (want ~1e-6, float32):")
    ok = True
    for name, e in checks.items():
        flag = "ok" if e < 1e-4 else "FAIL"
        ok = ok and e < 1e-4
        print(f"    {name:22s} {e:.2e}  [{flag}]")
    return ok


def render_align(path, wall_pos, int_pos, wall_dist, title):
    import pyvista as pv
    pv.start_xvfb()
    pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2400, 800))
    for col, cam in enumerate(["xy", "xz", "yz"]):
        pl.subplot(0, col)
        pl.add_mesh(pv.PolyData(wall_pos), color="lightgrey", point_size=3.0,
                    render_points_as_spheres=True, opacity=0.5)
        cloud = pv.PolyData(int_pos)
        cloud["wd"] = wall_dist
        pl.add_mesh(cloud, scalars="wd", cmap="turbo", point_size=3.5,
                    render_points_as_spheres=True,
                    scalar_bar_args=dict(title="wall dist", n_labels=3))
        pl.add_text(f"{title}  [{cam}]", position="upper_left", font_size=11)
        pl.camera_position = cam
    pl.screenshot(path); pl.close()
    print(f"  [ok] {path}")


def render_gradrot(path, pos0, posR, gv0, gvR, title):
    """Two invariants (div, Frobenius) before vs after rotation."""
    import pyvista as pv
    pv.start_xvfb()
    def div(g):  return g.reshape(-1, 3, 3)[:, [0, 1, 2], [0, 1, 2]].sum(1)
    def frob(g): return np.linalg.norm(g, axis=1)
    rows = [("div v", div(gv0), div(gvR)), ("||grad v||", frob(gv0), frob(gvR))]
    pl = pv.Plotter(shape=(2, 2), off_screen=True, window_size=(1600, 1600))
    for r, (name, s0, sR) in enumerate(rows):
        lim = np.percentile(np.abs(s0), 98) or 1.0
        for c, (pts, s, tag) in enumerate([(pos0, s0, "before"), (posR, sR, "after")]):
            pl.subplot(r, c)
            cl = pv.PolyData(pts); cl["s"] = s
            pl.add_mesh(cl, scalars="s", cmap="RdBu_r", clim=[-lim, lim],
                        point_size=4.0, render_points_as_spheres=True,
                        scalar_bar_args=dict(title=f"{name} {tag}", n_labels=3, fmt="%.1e"))
            pl.add_text(f"{name}  {tag}", position="upper_left", font_size=11)
    pl.screenshot(path); pl.close()
    print(f"  [ok] {path}  (colours must match before/after)")


def render_graph(path, pos, edge_index, title, slab_frac=0.05):
    import pyvista as pv
    pv.start_xvfb()
    e = edge_index.T.numpy()
    ext = pos.max(0) - pos.min(0)
    pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2700, 900))
    for col, ax in enumerate((0, 1, 2)):
        thick = slab_frac * ext[ax]
        m = np.abs(pos[:, ax] - np.median(pos[:, ax])) < thick
        pl.subplot(0, col)
        pl.add_mesh(pv.PolyData(pos[m]), color="black", point_size=2.5,
                    render_points_as_spheres=True)
        emask = m[e[:, 0]] & m[e[:, 1]]
        remap = np.cumsum(m) - 1
        es = remap[e[emask]]
        if len(es):
            lines = np.column_stack([np.full(len(es), 2), es]).ravel()
            pl.add_mesh(pv.PolyData(pos[m], lines=lines), color="steelblue",
                        line_width=1.0, opacity=0.6)
        pl.add_text(f"{title}  |{'xyz'[ax]}|<{slab_frac:.0%}", position="upper_left",
                    font_size=11)
        pl.camera_position = ["yz", "xz", "xy"][ax]
    pl.screenshot(path); pl.close()
    print(f"  [ok] {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_ok = True
    for version, cid, root in CASES:
        print(f"\n===== {version}/{cid} =====")
        ds = AngioFlowWallInteriorDataset(
            case_root_aneug_flow=root if version == "aneug_flow" else None,
            case_root_angioflow=root if version == "angioflow" else None,
            case_pattern={"aneug_flow": cid, "angioflow": cid},
            save_dir=SAVE_DIR, interior_dir=SCRATCH_INT,
            pool_ratio=0.16, n_frames=80,
            decimate_method="fps", decimate_ratio=1.0, lap_k=8,
        )
        idx = ds._index.index((version, cid, FRAME))
        wall, interior = ds[idx]
        print(f"  interior: pos {tuple(interior.pos.shape)}  edge_index "
              f"{tuple(interior.edge_index.shape)}  vel {tuple(interior.vel.shape)}  "
              f"grad_v {tuple(interior.grad_v.shape)}  grad_p {tuple(interior.grad_p.shape)}")
        # alignment: interior pool should sit inside the wall bbox
        wp = wall.pos.numpy(); ip = interior.pos.numpy()
        print(f"  wall  bbox {np.round(wp.min(0),4)}..{np.round(wp.max(0),4)}")
        print(f"  inter bbox {np.round(ip.min(0),4)}..{np.round(ip.max(0),4)}")

        all_ok &= check_rotation(wall, interior)

        # --- images ---
        render_align(os.path.join(OUT_DIR, f"{cid}_{version}_align.png"),
                     wp, ip, interior.wall_dist.numpy(), f"{cid} {version} wall+interior")

        seed = 999
        torch.manual_seed(seed)
        _, iR = CenterAndRandomRotate(rotate=True)((wall.clone(), interior.clone()))
        render_gradrot(os.path.join(OUT_DIR, f"{cid}_{version}_gradrot.png"),
                       interior.pos.numpy(), iR.pos.numpy(),
                       interior.grad_v.numpy(), iR.grad_v.numpy(),
                       f"{cid} {version}")

        sub = InteriorSubsampleGraph(sample_ratio=0.25)((wall.clone(), interior.clone()))[1]
        deg = torch.bincount(sub.edge_index[0], minlength=sub.num_nodes)
        print(f"  subsample: {sub.num_nodes} pts, {sub.edge_index.shape[1]} directed "
              f"edges, degree min {int(deg.min())}/mean {deg.float().mean():.1f}/max {int(deg.max())}")
        render_graph(os.path.join(OUT_DIR, f"{cid}_{version}_graph.png"),
                     sub.pos.numpy(), sub.edge_index, f"{cid} {version} subsampled graph")

    print(f"\n[{'ALL OK' if all_ok else 'FAILURES'}] rotation numeric checks")


if __name__ == "__main__":
    main()


"""
conda activate new
python -m angioflow.data.test_interior_transplant
"""
