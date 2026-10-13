"""Train the joint WSS + interior velocity/pressure model (AngioJointField).

Mirrors train.py (WSS side) and train_interior.py (interior side):
  * dataset: AngioFlowWallInterior[T0] — (wall, interior) pairs, same files both
    single-task pipelines already use; nothing new on disk;
  * transforms: JointAneurysmCrop (wall ROI crop + matching interior filter),
    InteriorSubsample (per-step interior draw), CenterAndRandomRotate (shared R),
    JointNormalize (one canonical pos scale), SymmetrizeEdges on the wall;
  * model: one shared HierGPS encoder -> two query-style heads coupled by
    per-block cross-field attention (models/joint);
  * loss = WSS terms (logmag + huber + dir, exactly train.py's; smooth only at
    full query resolution) + lambda_vel*huber(vel) + lambda_press*huber(press).
    No physics constraints (no div loss; wall_gate off by default; the WSS
    tangential projection stays on by default but --no_tangential disables it).
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from angioflow.config.defaults import (add_data_args, add_encoder_args, add_optim_args,
                                       add_schedule_args, add_misc_args)
from angioflow.data.datasets import load_excluded_keys
from angioflow.data.datasets_interior import (AngioFlowWallInteriorDataset,
                                              AngioFlowWallInteriorDatasetT0)
from angioflow.data.normalize import load_or_compute_stats
from angioflow.data.normalize_interior import load_or_compute_interior_stats
from angioflow.losses import log_mag_loss, tv_smooth_loss, directional_loss
from angioflow.models.joint import build_joint_field
from angioflow.transform import (CenterAndRandomRotate, InteriorSubsample,
                                 JointAneurysmCrop, JointNormalize, PressureZeroMax,
                                 SymmetrizeEdges)
from angioflow.training.train import (
    _chain, _pointwise_loss, load_waveform, maybe_resume, save_ckpt, split_by_case,
    build_scheduler, apply_lr_warmup, step_scheduler, save_run_config,
)


def add_model_args(parser):
    g = parser.add_argument_group("model:joint_field")
    g.add_argument("--dim_emb",    type=int, default=108, help="HierGPS encoder / token width")
    g.add_argument("--dim_field",  type=int, default=64,  help="joint decoder width (both heads)")
    g.add_argument("--n_blocks",   type=int, default=2,   help="joint fusion blocks (global + local + cross-field)")
    g.add_argument("--xattn_heads", type=int, default=4,  help="heads in token & cross-field attention")
    g.add_argument("--local_heads", type=int, default=4,  help="heads in the surface GATv2 local branch")
    g.add_argument("--k_local",    type=int, default=16,  help="wall neighbours per interior query")
    g.add_argument("--k_field",    type=int, default=16,
                   help="k for the BIDIRECTIONAL local state fusion (LocalFieldGather): "
                        "each wall query attends to its k nearest interior queries' "
                        "velocity states (WSS = mu*du/dn is a function of exactly that "
                        "boundary-layer field) AND each interior query attends to its k "
                        "nearest wall queries' WSS states (causality is one-way, inference "
                        "is not); 0 disables both directions")
    g.add_argument("--cross_field", action="store_true", default=True,
                   help="stage-3 GLOBAL surface<->interior cross-attention (all-to-all, "
                        "learned weighting -- catches long-range coupling the local "
                        "k_field branch can't, e.g. a jet impinging on a distant wall "
                        "patch); disable for a local-only ablation")
    g.add_argument("--no_cross_field", dest="cross_field", action="store_false")
    g.add_argument("--local_hops", type=int, default=2,   help="surface local branch k-hop radius (main GPU knob)")
    g.add_argument("--n_freqs",    type=int, default=6)
    g.add_argument("--geometry_token_pool", choices=["none", "latent"], default="latent")
    g.add_argument("--n_geometry_tokens", type=int, default=64)
    g.add_argument("--token_pool_heads", type=int, default=4)
    g.add_argument("--token_pool_layers", type=int, default=1)
    g.add_argument("--wall_query_ratio", type=float, default=0.25,
                   help="fraction of (cropped) WALL nodes decoded/supervised as WSS "
                        "queries per step, in BOTH train and eval (unbiased metric "
                        "estimate at a fraction of the GPU; sanity renders still run "
                        "full-resolution). 1.0 = every node")
    g.add_argument("--wall_query_max_points", type=int, default=None,
                   help="cap on WSS queries/graph (ceiling on wall_query_ratio)")
    g.add_argument("--interior_sample_ratio", type=float, default=0.1,
                   help="fraction of the (cropped) INTERIOR pool drawn per sample "
                        "(dataset transform, also applies at eval); the interior can be "
                        "sparse -- 0.1 is typically enough")
    g.add_argument("--interior_max_points", type=int, default=None,
                   help="cap on interior points/graph (ceiling on interior_sample_ratio)")
    g.add_argument("--tangential", action="store_true", default=True,
                   help="project WSS onto the tangent plane (parameter-free identity)")
    g.add_argument("--no_tangential", dest="tangential", action="store_false")
    g.add_argument("--wall_gate", action="store_true", default=False,
                   help="no-slip velocity gate (off: no physics constraints)")
    g.add_argument("--press_zero_max", action="store_true", default=False,
                   help="re-reference interior pressure to each case's own maximum "
                        "(p -> p - p.max(), so every value is <=0) instead of z-scoring "
                        "around Fluent's absolute-pressure convention (~1e5 Pa, an "
                        "arbitrary offset unrelated to the flow) -- see "
                        "transform.PressureZeroMax. Uses a SEPARATE stats file "
                        "(interior_stats_zeromax.pt) so this never touches the shared "
                        "interior_stats.pt train_interior.py depends on")
    # crop (matches train.py's AneurysmCrop args + the interior filter)
    g.add_argument("--crop_factor", type=float, default=1.5,
                   help="interior point kept iff dist-to-kept-wall <= factor*wall_dist + slack")
    g.add_argument("--crop_slack", type=float, default=5e-4)
    # HierGPS backbone
    g.add_argument("--n_pre",      type=int, default=2)
    g.add_argument("--n_gps",      type=int, default=4)
    g.add_argument("--n_pool",     type=int, default=2)
    g.add_argument("--n_inter",    type=int, default=0)
    g.add_argument("--pool_ratio", type=float, default=0.25)
    g.add_argument("--pool_knn_k", type=int, default=6)
    g.add_argument("--n_heads",    type=int, default=4)
    g.add_argument("--attn_type",  default="performer")
    g.add_argument("--grad_ckpt",  action="store_true")
    g.add_argument("--bidirectional", action="store_true", default=True)
    g.add_argument("--no_bidirectional", dest="bidirectional", action="store_false")
    # loss
    g.add_argument("--pointwise_loss", choices=["huber", "mse"], default="huber")
    g.add_argument("--lambda_huber",  type=float, default=5.0, help="WSS pointwise weight")
    g.add_argument("--lambda_logmag", type=float, default=0.1)
    g.add_argument("--lambda_dir",    type=float, default=0.1)
    g.add_argument("--lambda_smooth", type=float, default=0.1,
                   help="WSS TV smoothness; only applies at full query resolution")
    g.add_argument("--lambda_vel",    type=float, default=1.0)
    g.add_argument("--lambda_press",  type=float, default=1.0)
    g.add_argument("--max_wss",       type=float, default=35.0)
    g.add_argument("--log_eps",       type=float, default=0.1)
    g.add_argument("--sanity_every", type=int, default=50,
                   help="render pred-vs-GT WSS + velocity every N epochs (0 disables)")
    # data plumbing (interior side, mirrors train_interior.py)
    g.add_argument("--data_pool_ratio", type=float, default=0.16)
    g.add_argument("--interior_dir", default=None)
    g.add_argument("--freeze_dataset", action="store_true", default=True)
    g.add_argument("--no_freeze_dataset", dest="freeze_dataset", action="store_false")
    return parser


def get_args():
    p = argparse.ArgumentParser()
    add_data_args(p); add_encoder_args(p); add_optim_args(p); add_schedule_args(p)
    add_misc_args(p); add_model_args(p)
    # joint task logs to its own project; lr: the shared 1e-3 default destabilizes THIS
    # model at batch 4-5 (loss rises exactly as warmup passes ~6e-4, verified in the
    # first xf run) while 3e-4 learns cleanly (single-batch overfit: 0.92 -> 0.09)
    p.set_defaults(wandb_project="AngioFlowJoint", lr=3e-4)
    return p.parse_args()


def build_dataset(args, exclude_keys):
    ds_kwargs = dict(
        case_root_angioflow=args.case_root_angioflow, case_root_aneug_flow=args.case_root_aneug_flow,
        save_dir=args.save_dir, n_frames=args.n_frames,
        decimate_ratio=args.decimate_ratio, decimate_method=args.decimate_method,
        lap_k=args.lap_k, exclude_keys=exclude_keys,
        interior_dir=args.interior_dir, pool_ratio=args.data_pool_ratio,
        sample_ratio=args.interior_sample_ratio, sanity_first=0,
        freeze_dataset=args.freeze_dataset,
        load_edge_index=False, load_grad=False,        # no physics, no interior graph
    )
    versions = args.versions.split(",") if args.versions else None
    if args.frames == "all":
        return AngioFlowWallInteriorDataset(**ds_kwargs)
    frames = [int(t) for t in args.frames.split(",")]
    return AngioFlowWallInteriorDatasetT0(**ds_kwargs, versions=versions, frames=frames)


def interior_stats_path(args):
    """Separate file when --press_zero_max is on, so it never overwrites/reads the
    shared interior_stats.pt that train_interior.py's absolute-pressure convention
    depends on -- the two conventions are not interchangeable (see PressureZeroMax)."""
    name = "interior_stats_zeromax.pt" if getattr(args, "press_zero_max", False) else "interior_stats.pt"
    return os.path.join(args.save_dir, name)


class _WallOnly:
    """Adapt a single-graph transform to the (wall, interior) pair."""

    def __init__(self, t):
        self.t = t

    def __call__(self, data):
        wall, interior = data
        return self.t(wall), interior


def joint_loss(wss, query_idx, vel, press, wall, interior, args, y_mean, y_std, huber_delta):
    """WSS terms exactly as train.py's + pointwise vel/press. Returns
    (loss, dict-of-components) — components as floats for logging."""
    if query_idx is None:
        y, normals, edge_index = wall.y, wall.normals, wall.edge_index
    else:
        y, normals, edge_index = wall.y[query_idx], wall.normals[query_idx], None
    l_logmag = log_mag_loss(wss, y, y_mean, y_std, args.log_eps, args.max_wss)
    l_huber = _pointwise_loss(wss, y, args.pointwise_loss, huber_delta)
    l_smooth = tv_smooth_loss(wss, edge_index) if edge_index is not None else wss.new_zeros(())
    l_dir = directional_loss(wss, normals, y_mean, y_std)
    l_vel = _pointwise_loss(vel, interior.vel, args.pointwise_loss, huber_delta)
    l_press = _pointwise_loss(press, interior.press, args.pointwise_loss, huber_delta)
    loss = (args.lambda_logmag * l_logmag + args.lambda_huber * l_huber
            + args.lambda_smooth * l_smooth + args.lambda_dir * l_dir
            + args.lambda_vel * l_vel + args.lambda_press * l_press)
    comps = dict(logmag=l_logmag.item(), huber=l_huber.item(), smooth=l_smooth.item(),
                 dir=l_dir.item(), vel=l_vel.item(), press=l_press.item())
    return loss, comps


@torch.no_grad()
def evaluate(model, loader, waveform, device, args, y_mean, y_std, huber_delta, norm):
    model.eval()
    acc = defaultdict(float)
    n = 0
    for wall, interior in loader:
        wall, interior = wall.to(device), interior.to(device)
        wss, query_idx, vel, press = model(wall, interior, waveform)
        loss, comps = joint_loss(wss, query_idx, vel, press, wall, interior, args,
                                 y_mean, y_std, huber_delta)
        b = wall.num_graphs
        acc["loss"] += loss.item() * b
        for k, v in comps.items():
            acc[k] += v * b
        # denormalized MSEs in physical units (WSS on whatever subset was queried)
        y = wall.y if query_idx is None else wall.y[query_idx]
        acc["mse_wss"] += ((wss - y) * y_std).pow(2).mean().item() * b
        acc["mse_vel"] += ((vel - interior.vel) * norm.vel_std.to(device)).pow(2).mean().item() * b
        n += b
    model.train()
    n = max(n, 1)
    return {k: v / n for k, v in acc.items()}


def _tight_3d(ax, pts, margin=0.02):
    """Zoom a 3D axis right onto the point cloud: tight limits (2% margin) and a
    true-aspect box, instead of matplotlib's default padded cube."""
    lo, hi = pts.min(0), pts.max(0)
    span = (hi - lo).clip(min=1e-9)
    lo, hi = lo - margin * span, hi + margin * span
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_axis_off()


@torch.no_grad()
def save_joint_sanity(model, wall, interior, waveform, device, norm, out_dir, epoch,
                      max_wss=35.0, point_alpha=0.5, slab_frac=0.06, dpi=150):
    """Both fields of one test case, pred vs GT, in a single figure:

      row 0 — WSS 3D clouds on the (cropped) wall: pred | GT | |error|;
      row 1 — |velocity| PRED: 3D cloud over faint wall points + three slab sections;
      row 2 — |velocity| GT:   same layout, shared colour scale.

    Combines train.py's WSS sanity and train_interior.py's interior sanity, rendered
    with matplotlib (Agg): VTK off-screen collides with the live CUDA context during
    training and swallows stdout fds. 3D panels are high-dpi and tightly zoomed onto
    the cloud (_tight_3d) rather than matplotlib's default padded view.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    was_training = model.training
    model.eval()
    wall, interior = wall.to(device), interior.to(device)
    # the surface render needs a prediction at EVERY wall node, so temporarily lift
    # the query subsampling (eval otherwise queries wall_query_ratio like training);
    # no_grad + once-every-sanity_every epochs keeps the full-res pass affordable
    old_ratio = model.query_ratio
    model.query_ratio = 1.0
    try:
        wss_pred, _, vel_pred, _ = model(wall, interior, waveform)
    finally:
        model.query_ratio = old_ratio
    if was_training:
        model.train()

    wm, im = (wall.batch == 0), (interior.batch == 0)
    wp = wall.pos[wm].cpu().numpy()
    ip = interior.pos[im].cpu().numpy()
    wss_gt = norm.inverse_y(wall.y[wm]).norm(dim=-1).clamp(0, max_wss).cpu().numpy()
    wss_pr = norm.inverse_y(wss_pred[wm]).norm(dim=-1).clamp(0, max_wss).cpu().numpy()
    wss_err = np.abs(wss_gt - wss_pr)
    vgt = norm.inverse_vel(interior.vel[im]).norm(dim=-1).cpu().numpy()
    vpr = norm.inverse_vel(vel_pred[im]).norm(dim=-1).cpu().numpy()
    wmax = float(np.percentile(wss_gt, 99)) or 1.0
    vmax = float(np.percentile(vgt, 99)) or 1.0
    ext = ip.max(0) - ip.min(0)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"joint_epoch{epoch:04d}.png")

    fig = plt.figure(figsize=(26, 19))
    # row 0: WSS on the wall (3D, zoomed)
    for c, (name, mag, cmap, hi) in enumerate([("WSS pred", wss_pr, "coolwarm", wmax),
                                               ("WSS GT", wss_gt, "coolwarm", wmax),
                                               ("WSS |err|", wss_err, "Oranges", wss_err.max() or 1.0)]):
        ax = fig.add_subplot(3, 4, c + 1, projection="3d")
        sc = ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c=mag, cmap=cmap, vmin=0, vmax=hi,
                        s=10, alpha=0.9, linewidths=0)
        _tight_3d(ax, wp)
        ax.set_title(name, fontsize=11)
        fig.colorbar(sc, ax=ax, shrink=0.55, label="Pa/s")

    # rows 1-2: interior |velocity| pred / GT — 3D cloud + slab sections
    sc = None
    for r, (name, mag) in enumerate([("vel pred", vpr), ("vel GT", vgt)]):
        ax = fig.add_subplot(3, 4, (r + 1) * 4 + 1, projection="3d")
        ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c="lightgrey", s=2, alpha=0.10, linewidths=0)
        ax.scatter(ip[:, 0], ip[:, 1], ip[:, 2], c=mag, cmap="turbo", vmin=0, vmax=vmax,
                   s=8, alpha=point_alpha, linewidths=0)
        _tight_3d(ax, ip)
        ax.set_title(f"{name}  3D", fontsize=11)
        for axis in (0, 1, 2):
            ax = fig.add_subplot(3, 4, (r + 1) * 4 + 2 + axis)
            s = np.abs(ip[:, axis] - np.median(ip[:, axis])) < slab_frac * ext[axis]
            oa = [a for a in (0, 1, 2) if a != axis]
            sc = ax.scatter(ip[s][:, oa[0]], ip[s][:, oa[1]], c=mag[s], cmap="turbo",
                            vmin=0, vmax=vmax, s=10, alpha=0.9, linewidths=0)
            ax.set_title(f"{name}  slab |{'xyz'[axis]}|", fontsize=10)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(hspace=0.08, wspace=0.08, right=0.9)
    if sc is not None:
        cax = fig.add_axes([0.92, 0.08, 0.012, 0.5])   # dedicated strip right of the grid
        fig.colorbar(sc, cax=cax, label="|v| (m/s)")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[sanity] {path}  WSS clim=[0,{wmax:.1f}] Pa/s  |v| clim=[0,{vmax:.3f}] m/s", flush=True)


def build_model(args, device):
    model = build_joint_field(
        n_cot=args.lap_k, dim_emb=args.dim_emb, dim_field=args.dim_field,
        n_blocks=args.n_blocks, xattn_heads=args.xattn_heads, local_heads=args.local_heads,
        k_local=args.k_local, k_field=args.k_field, cross_field=args.cross_field,
        local_hops=args.local_hops, n_freqs=args.n_freqs,
        dim_te=args.dim_te, dim_wfe=args.dim_wfe,
        geometry_token_pool=args.geometry_token_pool,
        n_geometry_tokens=args.n_geometry_tokens,
        token_pool_heads=args.token_pool_heads, token_pool_layers=args.token_pool_layers,
        query_ratio=args.wall_query_ratio, query_max_points=args.wall_query_max_points,
        tangential=args.tangential, wall_gate=args.wall_gate,
        # backbone kwargs
        n_geo_dist=args.n_geo_dist, dim_pe=args.dim_pe, n_pre=args.n_pre, n_gps=args.n_gps,
        n_pool=args.n_pool, n_inter=args.n_inter, pool_ratio=args.pool_ratio,
        pool_knn_k=args.pool_knn_k, n_heads=args.n_heads, attn_type=args.attn_type,
        pe_n_heads=args.pe_n_heads, pe_n_layers=args.pe_n_layers, pe_use_mlp=args.pe_use_mlp,
        grad_ckpt=args.grad_ckpt, bidirectional=args.bidirectional)
    return model.to(device)


def save_script_copy(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(ckpt_dir, os.path.basename(__file__)))


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    exclude_keys = load_excluded_keys(args.save_dir) if args.exclude_anomalies else set()
    dataset = build_dataset(args, exclude_keys)
    train_idx, test_idx = split_by_case(dataset, args.train_frac, seed=args.seed)
    print(f"[data] {len(dataset)} samples  train={len(train_idx)}  test={len(test_idx)}")

    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}
    # stats.pt / interior_stats.pt already exist from the single-task pipelines;
    # interior_stats_zeromax.pt (--press_zero_max) is computed fresh on first use
    wss_stats = load_or_compute_stats(
        os.path.join(args.save_dir, "stats.pt"), dataset, train_keys,
        n_cases=args.stats_n_cases, frames_per_case=args.stats_frames_per_case, seed=args.seed)
    interior_stats = load_or_compute_interior_stats(
        interior_stats_path(args), dataset, train_keys,
        n_cases=args.stats_n_cases, frames_per_case=args.stats_frames_per_case, seed=args.seed,
        press_zero_max=args.press_zero_max)
    norm = JointNormalize(wss_stats, interior_stats)
    y_mean = wss_stats["y_mean"].to(device)
    y_std = wss_stats["y_std"].to(device)
    huber_delta = 1.0

    transforms = []
    if args.press_zero_max:
        # must run FIRST, on the full untouched interior pool -- see PressureZeroMax
        transforms.append(PressureZeroMax())
    transforms += [
        JointAneurysmCrop(low=args.mask_low, high=args.mask_high,
                          aneurysm_only=args.crop_aneurysm_only,
                          factor=args.crop_factor, slack=args.crop_slack),
        InteriorSubsample(sample_ratio=args.interior_sample_ratio,
                          max_points=args.interior_max_points),
        CenterAndRandomRotate(rotate=True),
        norm,
        _WallOnly(SymmetrizeEdges()),
    ]
    dataset.transform = _chain(*transforms)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)

    waveform = load_waveform(args.waveform_path, length=1000, device=device)
    model = build_model(args, device)
    print(f"[model] joint_field/hgps  {sum(p.numel() for p in model.parameters()):,} parameters")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = build_scheduler(optim, args)

    ckpt_dir = os.path.join(args.ckpt_root, args.run_name)
    save_script_copy(ckpt_dir)
    save_run_config(ckpt_dir, args)
    ckpt_last = os.path.join(ckpt_dir, "last.pth")
    ckpt_best = os.path.join(ckpt_dir, "best.pth")
    epoch_start, best_loss = 0, float("inf")
    if args.resume:
        epoch_start, best_loss = maybe_resume(ckpt_dir, model, optim, sched)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                   config=vars(args), resume="allow", id=args.wandb_run_id)

    for epoch in range(epoch_start, args.total_epochs):
        model.train()
        warming_up = apply_lr_warmup(optim, args, epoch)
        if hasattr(model, "redraw_projection"):
            model.redraw_projection.redraw_projections()
        tr = defaultdict(float)
        n = 0
        gnorm_sum, gnorm_max, n_clipped, n_steps = 0.0, 0.0, 0, 0
        for wall, interior in train_loader:
            wall, interior = wall.to(device), interior.to(device)
            if (torch.isnan(wall.y).any() or torch.isnan(interior.vel).any()
                    or torch.isnan(interior.press).any()):
                print("[warn] NaN in targets — skipping batch")
                continue
            optim.zero_grad()
            wss, query_idx, vel, press = model(wall, interior, waveform)
            loss, comps = joint_loss(wss, query_idx, vel, press, wall, interior, args,
                                     y_mean, y_std, huber_delta)
            if torch.isnan(loss):
                print("[warn] NaN loss — skipping batch"); optim.zero_grad(); continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            gnorm_sum += gnorm.item()
            gnorm_max = max(gnorm_max, gnorm.item())
            n_clipped += gnorm.item() > args.grad_clip_norm
            n_steps += 1
            optim.step()
            b = wall.num_graphs
            tr["loss"] += loss.item() * b
            for k, v in comps.items():
                tr[k] += v * b
            n += b
        n = max(n, 1)
        gnorm_mean = gnorm_sum / max(n_steps, 1)
        clip_frac = n_clipped / max(n_steps, 1)

        te = evaluate(model, test_loader, waveform, device, args, y_mean, y_std,
                      huber_delta, norm)
        te_loss = te["loss"]
        if not warming_up:
            step_scheduler(sched, args, te_loss)
        lr_now = optim.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train_loss {tr['loss']/n:.4g} "
              f"(wss: logmag {tr['logmag']/n:.4g} huber {tr['huber']/n:.4g} | "
              f"vel {tr['vel']/n:.4g} press {tr['press']/n:.4g})  "
              f"test_loss {te_loss:.4g} (logmag {te['logmag']:.4g} huber {te['huber']:.4g} "
              f"vel {te['vel']:.4g} press {te['press']:.4g})  "
              f"test_mse_wss {te['mse_wss']:.4g} (Pa/s)²  test_mse_vel {te['mse_vel']:.4g} (m/s)²  "
              f"lr {lr_now:.2e}  gnorm {gnorm_mean:.3g} (max {gnorm_max:.3g}, clipped {clip_frac:.0%})",
              flush=True)

        if args.wandb:
            wandb.log({**{f"train_{k}": v / n for k, v in tr.items()},
                       **{f"test_{k}": v for k, v in te.items()},
                       "lr": lr_now, "grad_norm_mean": gnorm_mean,
                       "grad_norm_max": gnorm_max, "grad_clip_frac": clip_frac}, step=epoch)

        if args.sanity_every and epoch % args.sanity_every == 0:
            try:
                # fresh test batch each time (test_loader is shuffled) -> a different
                # shape is rendered every sanity epoch
                sw, si = next(iter(test_loader))
                save_joint_sanity(model, sw, si, waveform, device, norm,
                                  os.path.join(ckpt_dir, "sanity"), epoch,
                                  max_wss=args.max_wss)
            except Exception as e:                        # a render hiccup must not kill training
                print(f"[sanity] render failed at epoch {epoch}: {e}", flush=True)

        save_ckpt(ckpt_last, model, optim, sched, epoch, best_loss, args)
        if epoch % 100 == 0:
            save_ckpt(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"),
                      model, optim, sched, epoch, best_loss, args)
        if te_loss < best_loss:
            best_loss = te_loss
            save_ckpt(ckpt_best, model, optim, sched, epoch, best_loss, args)
            print(f"  ↳ new best @ epoch {epoch} (test_loss={best_loss:.4g})")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main(get_args())


"""
# ---------------------------------------------------------------------------
# Joint baselines, ablation pair: WITH the global cross-field exchange (stage 3)
# vs local-only state fusion (--no_cross_field; k_field kNN attention remains).
# All loss weights explicit. Batch sizes VERIFIED via 2-full-epoch dry-runs on a
# clean 24GB GPU (train + full-resolution eval both covered):
#   cross-field arm:  batch_size 4 -> peak 22.1GB (~1.5GB headroom; the global
#                     dense surface<->interior attention costs ~7.6GB at b4 --
#                     do NOT raise this batch size)
#   local-only arm:   batch_size 6 -> peak 21.4GB (~2.2GB headroom; b4 was only
#                     14.5GB, so 6 reclaims the cross-attention savings)
# ---------------------------------------------------------------------------
python -m angioflow.training.train_joint \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name joint_p2_0frame_v1v2_e108f64_gps4_hop2_tok32_wq25_is10_xf \
  --versions aneug_flow,angioflow \
  --decimate_ratio 1.0 \
  --dim_emb 108 --dim_field 64 \
  --n_pre 2 --n_gps 4 --n_pool 2 --pool_ratio 0.25 --pool_knn_k 8 --n_inter 1 \
  --geometry_token_pool latent --n_geometry_tokens 32 \
  --n_blocks 2 --xattn_heads 4 --local_hops 2 --local_heads 4 --k_local 16 --k_field 16 \
  --wall_query_ratio 0.25 --interior_sample_ratio 0.1 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_logmag 0.1 --lambda_smooth 0.1 \
  --lambda_dir 0.1 --lambda_vel 1.0 --lambda_press 1.0 \
  --grad_ckpt \
  --lr 3e-4 \
  --total_epochs 300 --batch_size 5 --device cuda:1 --wandb --wandb_run_id 4k7l2ock \
  --grad_clip_norm 3.0

# local-only ablation: no global cross-field attention, bidirectional k_field kNN
# state fusion only
python -m angioflow.training.train_joint \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name joint_p2_0frame_v1v2_e108f64_gps4_hop2_tok32_wq25_is10_noxf \
  --versions aneug_flow,angioflow \
  --decimate_ratio 1.0 \
  --dim_emb 108 --dim_field 64 \
  --n_pre 2 --n_gps 4 --n_pool 2 --pool_ratio 0.25 --pool_knn_k 8 --n_inter 1 \
  --geometry_token_pool latent --n_geometry_tokens 32 \
  --n_blocks 2 --xattn_heads 4 --local_hops 2 --local_heads 4 --k_local 16 --k_field 16 \
  --no_cross_field \
  --wall_query_ratio 0.25 --interior_sample_ratio 0.1 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_logmag 0.1 --lambda_smooth 0.1 \
  --lambda_dir 0.1 --lambda_vel 1.0 --lambda_press 1.0 \
  --grad_ckpt \
  --lr 3e-4 \
  --total_epochs 300 --batch_size 6 --device cuda:2 --wandb


python -m angioflow.training.train_joint \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name joint_p2_0frame_v1v2_e108f64_gps4_hop2_tok32_wq25_is10_xf \
  --versions aneug_flow,angioflow \
  --decimate_ratio 1.0 \
  --dim_emb 108 --dim_field 64 \
  --n_pre 2 --n_gps 4 --n_pool 2 --pool_ratio 0.25 --pool_knn_k 8 --n_inter 1 \
  --geometry_token_pool latent --n_geometry_tokens 32 \
  --n_blocks 2 --xattn_heads 4 --local_hops 2 --local_heads 4 --k_local 16 --k_field 16 \
  --wall_query_ratio 0.25 --interior_sample_ratio 0.1 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_logmag 0.1 --lambda_smooth 0.1 \
  --lambda_dir 0.1 --lambda_vel 1.0 --lambda_press 1.0 \
  --grad_ckpt \
  --lr 3e-4 \
  --total_epochs 300 --batch_size 4 --device cuda:1 --wandb --wandb_run_id 4k7l2ock \
  --grad_clip_norm 3.0

# ---------------------------------------------------------------------------
# High-capacity xf variant (NOT dry-run verified): decoder width 64->96 and
# blocks 2->3 (x2.25 on the dominant decoder cost), plus near-free encoder
# capacity (gps 4->6, tokens 32->64). At batch_size 4 the extrapolation from the
# observed 16GB @ b5 predicts ~24GB -- OOM-risky. Eval now queries wall_query_ratio
# too (no full-resolution eval spike; only the every-50-epochs sanity render runs
# full-res, under no_grad). If it OOMs, fall back in this order: --n_blocks 2
# (keeps the width gain), then --batch_size 3.
# ---------------------------------------------------------------------------
python -m angioflow.training.train_joint \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name joint_p1r015_0frame_v1v2_e108f96_gps6_hop2_tok64_wq25_is10_blk3_xf \
  --versions aneug_flow,angioflow \
  --decimate_ratio 1.0 \
  --dim_emb 108 --dim_field 96 \
  --n_pre 2 --n_gps 6 --n_pool 1 --pool_ratio 0.15 --pool_knn_k 8 --n_inter 1 \
  --geometry_token_pool latent --n_geometry_tokens 64 \
  --n_blocks 3 --xattn_heads 4 --local_hops 2 --local_heads 4 --k_local 16 --k_field 16 \
  --wall_query_ratio 0.25 --interior_sample_ratio 0.1 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_logmag 0.1 --lambda_smooth 0.1 \
  --lambda_dir 0.1 --lambda_vel 1.0 --lambda_press 1.0 \
  --grad_ckpt \
  --lr 3e-4 \
  --total_epochs 300 --batch_size 4 --device cuda:1 --wandb \
  --grad_clip_norm 3.0

# ---------------------------------------------------------------------------
# Same config as the blk3_xf baseline above (this one trains well: clip% decayed
# 47%->2% over 167 epochs, test_loss 0.78->0.19, still setting new bests -- see
# session notes), single factor changed: a more aggressive ROI crop. AneurysmCrop
# always keeps the aneurysm (node_type==1) plus a random geodesic ring up to
# p*max(geo_dist), p ~ U[mask_low, mask_high] -- mask_high 0.25 -> 0.10 roughly
# halves the deepest ring the model ever sees, so both the wall AND (via
# JointAneurysmCrop's consistency filter) the interior pool shrink together.
# Cheaper per case (smaller graphs), so batch_size could likely go higher than 4,
# but left unchanged here to isolate crop depth as the only variable -- verify
# GPU headroom yourself before bumping it. For the most aggressive option (no
# ring at all, deterministic aneurysm-only every sample) use --crop_aneurysm_only
# instead of lowering mask_high further.
# ---------------------------------------------------------------------------
python -m angioflow.training.train_joint \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name joint_p1r015_0frame_v1v2_e108f96_gps6_hop2_tok64_wq50_is10_blk3_xf_mh020_pressmax0 \
  --versions aneug_flow,angioflow \
  --decimate_ratio 1.0 \
  --mask_low 0.0 --mask_high 0.01 \
  --dim_emb 108 --dim_field 96 \
  --n_pre 2 --n_gps 6 --n_pool 1 --pool_ratio 0.15 --pool_knn_k 8 --n_inter 1 \
  --geometry_token_pool latent --n_geometry_tokens 64 \
  --n_blocks 3 --xattn_heads 4 --local_hops 2 --local_heads 4 --k_local 16 --k_field 16 \
  --wall_query_ratio 0.5 --interior_sample_ratio 0.2 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_logmag 0.1 --lambda_smooth 0.1 \
  --lambda_dir 0.1 --lambda_vel 1.0 --lambda_press 1.0 \
  --grad_ckpt \
  --lr 3e-4 \
  --total_epochs 300 --batch_size 4 --device cuda:0 --wandb --wandb_run_id eqy8bvnz\
  --grad_clip_norm 3.0 --press_zero_max
"""
