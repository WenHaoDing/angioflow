"""Paper-figure panels for the AngioJointField schematic: input mesh, the HierGPS
pooling hierarchy, and the two predicted fields against ground truth.

Every panel is a SEPARATE transparent PNG rendered from the SAME camera, so they can
be dropped into a figure and will line up. Panels:

  01_surface_mesh              cropped wall mesh, triangulation visible (encoder input)
  02_pool_L{k}                 level k of the HierGPS hierarchy: FPS points + the kNN
                               graph rebuilt on them, over a ghost of the input surface
  03_wss_{gt,pred}             |WSS| on the wall, shared fixed colour scale (--wss_max)
  04_velocity_{gt,pred}        |u| on the interior point cloud, shared colour scale
  05_pressure_{gt,pred}        pressure anomaly (only with --pressure)

The hierarchy is reproduced with the SAME ops and hyperparameters the trained encoder
uses (fps -> knn_graph -> to_undirected, at the checkpoint's pool_ratio / pool_knn_k /
n_pool / bidirectional), so what is drawn is the graph the model actually pools over,
not an illustration of one. FPS picks a random start, so --seed fixes which points are
chosen; the COUNT per level is fixed by pool_ratio either way.

The case is cropped, posed and normalised exactly as angioflow.training.eval_joint does
-- same JointAneurysmCrop the run trained with, no random rotation -- so the geometry in
the figure is the geometry the model saw.

  conda activate new
  PYTHONPATH=. python -m angioflow.visualize.schematic.render_schematic \
      --run_name joint_p1r015_0frame_v1v2_e108f96_gps6_hop2_tok64_wq50_is10_blk3_xf_mh020_pressmax0 \
      --device cuda:0
  ... --case_id 0m5cVmE8R1 --view xz --zoom 1.3      # a specific case / angle
"""
from __future__ import annotations

import argparse
import json
import os

# before the first pyvista/VTK import: no hardware GL here, and over SSH DISPLAY is
# set but the forwarded server has no GLX, so VTK aborts creating a context
if not os.environ.get("AF_USE_GPU"):
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import fps, knn_graph
from torch_geometric.utils import to_undirected

from angioflow.data.datasets import load_excluded_keys
from angioflow.data.normalize import load_or_compute_stats
from angioflow.data.normalize_interior import load_or_compute_interior_stats
from angioflow.training.train import _chain, load_waveform, split_by_case
from angioflow.training.train_joint import _WallOnly, build_dataset, build_model, interior_stats_path
from angioflow.transform import (CenterAndRandomRotate, JointAneurysmCrop, JointNormalize,
                                 PressureZeroMax, SymmetrizeEdges)
from angioflow.visualize.sanity import _make_surface_mesh

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
DEFAULT_RUN = "joint_p1r015_0frame_v1v2_e108f96_gps6_hop2_tok64_wq50_is10_blk3_xf_mh020_pressmax0"


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_name", default=DEFAULT_RUN)
    p.add_argument("--ckpt_root", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_checkpoints")
    p.add_argument("--which", choices=["best", "last"], default="best")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--case_id", default=None, help="substring of the case id; default: first test case")
    p.add_argument("--frame", type=int, default=None, help="restrict to this time frame")
    p.add_argument("--seed", type=int, default=0, help="crop depth + FPS start")
    p.add_argument("--view", default="iso", choices=["iso", "xy", "xz", "yz"])
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--window", type=int, nargs=2, default=[1400, 1400])
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--wss_max", type=float, default=12.0, help="upper end of the shared |WSS| colour scale")
    p.add_argument("--point_size", type=float, default=7.0, help="interior cloud point size")
    p.add_argument("--max_points", type=int, default=20000, help="interior cloud plot cap")
    p.add_argument("--pressure", action="store_true", help="also render pressure anomaly panels")
    p.add_argument("--colorbar", action="store_true", default=True)
    p.add_argument("--no_colorbar", dest="colorbar", action="store_false")
    p.add_argument("--opaque", action="store_true", help="white background instead of transparent")
    p.add_argument("--vtk", action="store_true", help="also dump .vtp/.vtk for re-rendering elsewhere")
    return p.parse_args()


@torch.no_grad()
def forward_case(model, dataset, index, norm, waveform, device):
    """One case through the model. Returns (info, fields, wall) -- `wall` is kept so the
    hierarchy is pooled over the SAME crop that produced the fields (the crop draws a
    random ring depth, so re-fetching the sample would give a different mesh)."""
    version, case_id, frame = dataset._index[index]
    wall, interior = dataset[index]
    wall_b = Batch.from_data_list([wall]).to(device)
    int_b = Batch.from_data_list([interior]).to(device)

    wss_pred, query_idx, vel_pred, press_pred = model(wall_b, int_b, waveform)
    assert query_idx is None, "need full wall resolution: set model.query_ratio = 1.0"

    pos_std = norm.pos_std.to(device)
    fields = dict(
        wall_pos=(wall_b.pos * pos_std).cpu().numpy(),
        edge_index=wall_b.edge_index.cpu().numpy(),
        wss_gt=norm.inverse_y(wall_b.y).cpu().numpy(),
        wss_pr=norm.inverse_y(wss_pred).cpu().numpy(),
        int_pos=(int_b.pos * pos_std).cpu().numpy(),
        vel_gt=norm.inverse_vel(int_b.vel).cpu().numpy(),
        vel_pr=norm.inverse_vel(vel_pred).cpu().numpy(),
        press_gt=norm.inverse_press(int_b.press).cpu().numpy().reshape(-1),
        press_pr=norm.inverse_press(press_pred).cpu().numpy().reshape(-1),
    )
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    info = {
        "version": version, "case_id": str(case_id), "frame": int(frame),
        "n_wall": int(fields["wall_pos"].shape[0]),
        "n_interior": int(fields["int_pos"].shape[0]),
        "wss_rel_l2": rel(fields["wss_pr"], fields["wss_gt"]),
        "vel_rel_l2": rel(fields["vel_pr"], fields["vel_gt"]),
    }
    return info, fields, wall_b


def pool_levels(wall, n_pool, ratio, knn_k, bidirectional, pos_std):
    """Replay GeoHierGPSEncoder.encode_tokens_with_pos's pooling: n_pool x
    (FPS -> kNN graph -> optional symmetrise). Returns [(pos_mm, edge_index), ...] for
    each COARSE level, positions denormalised to match the other panels."""
    pos, batch = wall.pos, wall.batch
    levels = []
    for _ in range(n_pool):
        perm = fps(pos, batch, ratio=ratio)
        pos, batch = pos[perm], batch[perm]
        ei = knn_graph(pos, k=knn_k, batch=batch, loop=False)
        if bidirectional:
            ei = to_undirected(ei, num_nodes=pos.size(0))
        levels.append(((pos * pos_std.to(pos.device)).cpu().numpy(), ei.cpu().numpy()))
    return levels


def _plotter(args):
    import pyvista as pv
    p = pv.Plotter(off_screen=True, window_size=tuple(args.window))
    if args.opaque:
        p.set_background("white")
    return p


def _shoot(p, args, cpos, path):
    p.camera_position = cpos
    p.screenshot(path, transparent_background=not args.opaque)
    p.close()
    print(f"  -> {os.path.basename(path)}")


def base_camera(mesh, args):
    """One camera for every panel, derived from the input mesh so the coarse levels and
    the point clouds are framed identically."""
    p = _plotter(args)
    p.add_mesh(mesh)
    p.camera_position = args.view
    p.reset_camera()
    p.camera.zoom(args.zoom)
    cpos = p.camera_position
    p.close()
    return cpos


def render_mesh_panel(mesh, args, cpos, path):
    p = _plotter(args)
    p.add_mesh(mesh, color="#bfd3e6", show_edges=True, edge_color="#2b4a6f",
               line_width=0.6, smooth_shading=True)
    _shoot(p, args, cpos, path)


def render_pool_panel(mesh, pos, edge_index, args, cpos, path):
    """Coarse level over a ghost of the input surface, so the reader sees WHERE the
    retained points sit. Edges are drawn as explicit 2-point cells (the kNN graph is
    not a triangulation, so it cannot go through PolyData faces)."""
    import pyvista as pv
    ei = edge_index[:, edge_index[0] < edge_index[1]]          # one line per undirected pair
    lines = np.hstack([np.full((ei.shape[1], 1), 2), ei.T]).ravel()
    graph = pv.PolyData(pos, lines=lines)
    p = _plotter(args)
    p.add_mesh(mesh, color="#c8c8c8", opacity=0.18, smooth_shading=True)
    p.add_mesh(graph, color="black", line_width=1.6)
    p.add_mesh(pv.PolyData(pos), color="#1f4fd8", point_size=args.point_size + 3,
               render_points_as_spheres=True)
    _shoot(p, args, cpos, path)


def render_scalar_mesh(mesh, values, clim, title, args, cpos, path):
    p = _plotter(args)
    m = mesh.copy()
    m[title] = values
    p.add_mesh(m, scalars=title, cmap=args.cmap, clim=clim, smooth_shading=True,
               show_scalar_bar=args.colorbar,
               scalar_bar_args=dict(title=title, n_labels=3, title_font_size=22,
                                    label_font_size=18, color="black"))
    _shoot(p, args, cpos, path)


def render_scalar_cloud(pos, values, clim, title, args, cpos, path):
    import pyvista as pv
    cloud = pv.PolyData(pos)
    cloud[title] = values
    p = _plotter(args)
    p.add_mesh(cloud, scalars=title, cmap=args.cmap, clim=clim,
               point_size=args.point_size, render_points_as_spheres=True,
               show_scalar_bar=args.colorbar,
               scalar_bar_args=dict(title=title, n_labels=3, title_font_size=22,
                                    label_font_size=18, color="black"))
    _shoot(p, args, cpos, path)


def subsample(n, cap, seed):
    if n <= cap:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, cap, replace=False)


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = os.path.join(args.ckpt_root, args.run_name)
    chk = torch.load(os.path.join(ckpt_dir, f"{args.which}.pth"), map_location=device,
                     weights_only=False)
    targs = argparse.Namespace(**chk["args"])

    exclude = load_excluded_keys(targs.save_dir) if targs.exclude_anomalies else set()
    dataset = build_dataset(targs, exclude)
    train_idx, test_idx = split_by_case(dataset, targs.train_frac, seed=targs.seed)
    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}

    wss_stats = load_or_compute_stats(
        os.path.join(targs.save_dir, "stats.pt"), dataset, train_keys,
        n_cases=targs.stats_n_cases, frames_per_case=targs.stats_frames_per_case, seed=targs.seed)
    press_zero_max = getattr(targs, "press_zero_max", False)
    interior_stats = load_or_compute_interior_stats(
        interior_stats_path(targs), dataset, train_keys, n_cases=targs.stats_n_cases,
        frames_per_case=targs.stats_frames_per_case, seed=targs.seed, press_zero_max=press_zero_max)
    norm = JointNormalize(wss_stats, interior_stats)

    model = build_model(targs, device)
    model.load_state_dict(chk["model"])
    model.eval()
    model.query_ratio = 1.0

    # identical to eval_joint: same crop the run trained with, fixed pose
    transforms = [PressureZeroMax()] if press_zero_max else []
    transforms += [
        JointAneurysmCrop(low=targs.mask_low, high=targs.mask_high,
                          aneurysm_only=targs.crop_aneurysm_only,
                          factor=targs.crop_factor, slack=targs.crop_slack),
        CenterAndRandomRotate(rotate=False),
        norm,
        _WallOnly(SymmetrizeEdges()),
    ]
    dataset.transform = _chain(*transforms)
    waveform = load_waveform(targs.waveform_path, length=1000, device=device)

    picks = [i for i in test_idx
             if (args.case_id is None or args.case_id in str(dataset._index[i][1]))
             and (args.frame is None or dataset._index[i][2] == args.frame)]
    if not picks:
        raise SystemExit(f"no test case matches case_id={args.case_id} frame={args.frame}")
    index = picks[0]

    import pyvista as pv
    pv.start_xvfb()

    info, fields, wall = forward_case(model, dataset, index, norm, waveform, device)
    tag = f"{info['version']}_{info['case_id']}_frame{info['frame']}"
    print(f"[case] {tag}  wall={info['n_wall']} pts  interior={info['n_interior']} pts")
    print(f"[err ] wss rel_L2={info['wss_rel_l2']:.4f}  vel rel_L2={info['vel_rel_l2']:.4f}")

    mesh = _make_surface_mesh(fields["wall_pos"], fields["edge_index"])
    cpos = base_camera(mesh, args)

    render_mesh_panel(mesh, args, cpos, os.path.join(args.out_dir, "01_surface_mesh.png"))

    levels = pool_levels(wall, targs.n_pool, targs.pool_ratio, targs.pool_knn_k,
                         getattr(targs, "bidirectional", True), norm.pos_std)
    for k, (pos, ei) in enumerate(levels, start=1):
        print(f"[pool] level {k}: {len(pos)} points, {ei.shape[1] // 2} kNN edges")
        render_pool_panel(mesh, pos, ei, args, cpos,
                          os.path.join(args.out_dir, f"02_pool_L{k}.png"))

    wss_gt = np.linalg.norm(fields["wss_gt"], axis=1)
    wss_pr = np.linalg.norm(fields["wss_pr"], axis=1)
    clim = [0.0, args.wss_max]                         # fixed, shared: the two panels
    for name, v in (("gt", wss_gt), ("pred", wss_pr)):  # must be comparable by eye
        render_scalar_mesh(mesh, v, clim, "|WSS| (Pa)", args, cpos,
                           os.path.join(args.out_dir, f"03_wss_{name}.png"))

    sel = subsample(fields["int_pos"].shape[0], args.max_points, args.seed)
    ipos = fields["int_pos"][sel]
    vel_gt = np.linalg.norm(fields["vel_gt"], axis=1)[sel]
    vel_pr = np.linalg.norm(fields["vel_pr"], axis=1)[sel]
    clim = [0.0, float(np.percentile(vel_gt, 99))]
    for name, v in (("gt", vel_gt), ("pred", vel_pr)):
        render_scalar_cloud(ipos, v, clim, "|u| (m/s)", args, cpos,
                            os.path.join(args.out_dir, f"04_velocity_{name}.png"))

    if args.pressure:
        ref = float(fields["press_gt"].mean())        # absolute pressure has a big offset
        pg, pp = fields["press_gt"][sel] - ref, fields["press_pr"][sel] - ref
        a = float(np.percentile(np.abs(pg), 99))
        for name, v in (("gt", pg), ("pred", pp)):
            render_scalar_cloud(ipos, v, [-a, a], "p anomaly (Pa)", args, cpos,
                                os.path.join(args.out_dir, f"05_pressure_{name}.png"))

    if args.vtk:
        mesh["wss_gt"], mesh["wss_pred"] = wss_gt, wss_pr
        mesh.save(os.path.join(args.out_dir, "wall.vtp"))
        for k, (pos, ei) in enumerate(levels, start=1):
            e = ei[:, ei[0] < ei[1]]
            pv.PolyData(pos, lines=np.hstack([np.full((e.shape[1], 1), 2), e.T]).ravel()).save(
                os.path.join(args.out_dir, f"pool_L{k}.vtp"))
        cloud = pv.PolyData(fields["int_pos"])
        cloud["vel_gt"] = np.linalg.norm(fields["vel_gt"], axis=1)
        cloud["vel_pred"] = np.linalg.norm(fields["vel_pr"], axis=1)
        cloud.save(os.path.join(args.out_dir, "interior.vtp"))

    meta = dict(info, run_name=args.run_name, view=args.view, zoom=args.zoom, seed=args.seed,
                n_pool=targs.n_pool, pool_ratio=targs.pool_ratio, pool_knn_k=targs.pool_knn_k,
                level_sizes=[int(len(p)) for p, _ in levels],
                camera=[list(map(float, v)) for v in cpos])
    with open(os.path.join(args.out_dir, "panels.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()
