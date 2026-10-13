"""Train the interior velocity/pressure field baselines (HierGPS / PointNet backbone).

Mirrors angioflow/training/train.py, but for the interior task:
  * dataset returns (wall, interior) pairs (AngioFlowWallInterior[T0]);
  * transforms are tuple-aware: CenterAndRandomRotate (shared R over wall+interior),
    InteriorSubsampleGraph (per-epoch pool subsample), InteriorNormalize;
  * model is AngioInteriorField (geometry backbone → coarse tokens → interior
    cross-attention → vel/press); no WSS anywhere;
  * loss = lambda_vel · huber/mse(vel) + lambda_press · huber/mse(press), plus an
    optional lambda_div · continuity residual (div(vel_pred) via a local graph
    gradient stencil, see models.interior.constraints), in normalized units.

Two baselines via --backbone {hgps, pointnet}.
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

from angioflow.config.defaults import add_data_args, add_encoder_args, add_optim_args, add_misc_args
from angioflow.data.datasets import load_excluded_keys
from angioflow.data.datasets_interior import (AngioFlowWallInteriorDataset,
                                              AngioFlowWallInteriorDatasetT0)
from angioflow.data.normalize_interior import load_or_compute_interior_stats
from angioflow.models.interior import build_interior_field
from angioflow.models.interior.constraints import (append_synthetic_offsets,
                                                    build_synthetic_offsets,
                                                    continuity_residual,
                                                    synthetic_continuity_residual)
from angioflow.transform import (CenterAndRandomRotate, InteriorNormalize, InteriorSubsample,
                                 InteriorSubsampleGraph)
# generic, task-neutral helpers (not model pieces) reused from the WSS trainer
from angioflow.training.train import (
    _chain, load_waveform, maybe_resume, save_ckpt, split_by_case,
    build_scheduler, apply_lr_warmup, step_scheduler, save_run_config,
)


def add_model_args(parser):
    g = parser.add_argument_group("model:interior_field")
    g.add_argument("--backbone",   choices=["hgps", "pointnet"], default="hgps")
    g.add_argument("--dim_emb",    type=int, default=128, help="geometry encoder / token width")
    g.add_argument("--dim_field",  type=int, default=128, help="cross-attention decoder width")
    g.add_argument("--n_blocks",   type=int, default=2,   help="two-scale fusion blocks (global+local)")
    g.add_argument("--xattn_heads", type=int, default=4,  help="heads in the global cross-attention")
    g.add_argument("--k_local",    type=int, default=16,  help="surface neighbours per interior query (local branch)")
    g.add_argument("--n_freqs",    type=int, default=6,   help="Fourier frequencies for query pos")
    g.add_argument("--geometry_token_pool", choices=["none", "latent"], default="none",
                   help="global tokens: variable downsampled graph (none) or fixed learned latents")
    g.add_argument("--n_geometry_tokens", type=int, default=64,
                   help="fixed tokens per graph when --geometry_token_pool latent")
    g.add_argument("--token_pool_heads", type=int, default=4)
    g.add_argument("--token_pool_layers", type=int, default=1)
    g.add_argument("--wall_gate", action="store_true", default=True,
                   help="hard no-slip constraint: |velocity| -> 0 at the wall via wall_dist")
    g.add_argument("--no_wall_gate", dest="wall_gate", action="store_false")
    g.add_argument("--sample_ratio", type=float, default=0.25,
                   help="fraction of the interior pool drawn per sample (training subsample)")
    g.add_argument("--max_points", type=int, default=None,
                   help="cap on interior points/graph (ceiling on sample_ratio; bounds memory on big pools)")
    g.add_argument("--data_pool_ratio", type=float, default=0.16,
                   help="on-disk pool fraction the dataset was built with (only used if reprocessing)")
    g.add_argument("--interior_dir", default=None, help="interior data dir (default: <save_dir>/interior)")
    g.add_argument("--freeze_dataset", action="store_true", default=True,
                   help="use only already-processed cases; NEVER (re)process raw data during training")
    g.add_argument("--no_freeze_dataset", dest="freeze_dataset", action="store_false")
    # HierGPS backbone
    g.add_argument("--n_pre",      type=int, default=2)
    g.add_argument("--n_gps",      type=int, default=4)
    g.add_argument("--n_pool",     type=int, default=2,   help="FPS pool levels for the token graph")
    g.add_argument("--pool_ratio", type=float, default=0.25)
    g.add_argument("--pool_knn_k", type=int, default=6)
    g.add_argument("--n_heads",    type=int, default=4)
    g.add_argument("--attn_type",  default="performer")
    # PointNet backbone
    g.add_argument("--base_ch",    type=int, default=64)
    # loss
    g.add_argument("--pointwise_loss", choices=["huber", "mse"], default="huber")
    g.add_argument("--lambda_vel",   type=float, default=1.0)
    g.add_argument("--lambda_press", type=float, default=1.0)
    g.add_argument("--lambda_div", type=float, default=0.0,
                   help="incompressible continuity (div-free) physics loss weight; 0 disables "
                        "it and keeps the graph-free baseline data path")
    g.add_argument("--div_method", choices=["graph", "tet"], default="graph",
                   help="continuity-residual estimator: 'graph' = WLS stencil over the "
                        "precomputed pool edge_index (cheap, no extra model queries, default); "
                        "'tet' = fresh synthetic tetrahedra around a random subset of query "
                        "points, queried through one extra combined forward pass -- more "
                        "accurate once K (--div_k_offsets) is large enough, see "
                        "angioflow/data/gradient_estimation/README.md")
    # graph method
    g.add_argument("--div_ridge", type=float, default=3e-2)
    g.add_argument("--div_power", type=float, default=2.0)
    g.add_argument("--div_min_neighbors", type=int, default=3)
    g.add_argument("--div_cond_tol", type=float, default=1e-6)
    # tet method
    g.add_argument("--div_n_centers", type=int, default=128,
                   help="[tet] max synthetic-tetrahedra centers per graph per step -- the "
                        "main GPU-cost knob (extra query rows/graph = n_centers*k_offsets)")
    g.add_argument("--div_k_offsets", type=int, default=24,
                   help="[tet] offset vertices per element (>=3; 4 = canonical regular "
                        "tetrahedron, else an isotropic Fibonacci-sphere spread); empirically "
                        "K>=16 is needed to beat the graph stencil, K>=24 to beat it reliably")
    g.add_argument("--div_radius", type=float, default=0.12,
                   help="[tet] offset radius in the SAME normalized frame as interior.pos "
                        "(pos/pos_std) -- default ~5x the typical normalized point spacing, "
                        "per interior_stats.pt and the offline radius sweep's optimum")
    g.add_argument("--div_wall_margin", type=float, default=2.0,
                   help="[tet] a center is only used if wall_dist > this many radii, so no "
                        "offset vertex of its element can cross the wall")
    g.add_argument("--div_wall_percentile", type=float, default=0.75,
                   help="[tet] centers are farthest-point-sampled (spatially spread, not "
                        "clustered) from only the top fraction of eligible points by wall "
                        "distance -- 0.75 keeps everything above the 25th percentile (excludes "
                        "only the nearest-wall quarter), biasing toward the bulk-volume "
                        "interior (dome core, parent-vessel core) over the boundary layer")
    # lr schedule
    g.add_argument("--lr_schedule", choices=["cosine", "reduce_on_plateau"], default="cosine",
                   help="'cosine' = current CosineAnnealingLR(T_max=total_epochs), unchanged "
                        "default; 'reduce_on_plateau' holds LR constant until test_loss stops "
                        "improving for --lr_plateau_patience epochs, then cuts it by "
                        "--lr_plateau_factor -- lets a productive LR 'dwell' as long as it's "
                        "actually still helping, instead of a fixed schedule guessing epoch "
                        "boundaries")
    g.add_argument("--lr_warmup_epochs", type=int, default=5,
                   help="linear LR warmup 0->lr over this many epochs before either schedule "
                        "kicks in (0 disables)")
    g.add_argument("--lr_plateau_factor", type=float, default=0.5,
                   help="[reduce_on_plateau] LR multiplier applied when test_loss plateaus")
    g.add_argument("--lr_plateau_patience", type=int, default=15,
                   help="[reduce_on_plateau] epochs with no test_loss improvement before "
                        "cutting LR")
    g.add_argument("--sanity_every", type=int, default=10,
                   help="render predicted-vs-GT interior velocity every N epochs (0 disables)")
    return parser


def get_args():
    p = argparse.ArgumentParser()
    add_data_args(p); add_encoder_args(p); add_optim_args(p); add_misc_args(p); add_model_args(p)
    return p.parse_args()


def build_dataset(args, exclude_keys):
    ds_kwargs = dict(
        case_root_angioflow=args.case_root_angioflow, case_root_aneug_flow=args.case_root_aneug_flow,
        save_dir=args.save_dir, n_frames=args.n_frames,
        decimate_ratio=args.decimate_ratio, decimate_method=args.decimate_method,
        lap_k=args.lap_k, exclude_keys=exclude_keys,
        interior_dir=args.interior_dir, pool_ratio=args.data_pool_ratio,
        sample_ratio=args.sample_ratio, sanity_first=0,   # no sanity renders during training
        freeze_dataset=args.freeze_dataset,               # never (re)process raw data here
        # pool edge_index only for the graph continuity estimator; grad_v/grad_p stay unused
        load_edge_index=(args.lambda_div > 0 and getattr(args, "div_method", "graph") == "graph"),
        load_grad=False,
    )
    versions = args.versions.split(",") if args.versions else None
    if args.frames == "all":
        return AngioFlowWallInteriorDataset(**ds_kwargs)
    frames = [int(t) for t in args.frames.split(",")]
    return AngioFlowWallInteriorDatasetT0(**ds_kwargs, versions=versions, frames=frames)


def _pointwise(pred, y, kind, delta=1.0):
    return F.mse_loss(pred, y) if kind == "mse" else F.huber_loss(pred, y, delta=delta)


def field_loss(vel_pred, press_pred, interior, kind, lam_vel, lam_press, lam_div=0.0,
               div_kwargs=None, divergence=None):
    """`divergence`: precomputed [P] residual tensor (e.g. from the synthetic-tet
    estimator) to use directly; if None and lam_div>0, falls back to computing it from
    interior.edge_index via the graph stencil (constraints.continuity_residual) — the
    original, still-default behaviour.
    """
    lv = _pointwise(vel_pred, interior.vel, kind)
    lp = _pointwise(press_pred, interior.press, kind)
    ld = vel_pred.new_zeros(())
    if lam_div > 0:
        if divergence is None:
            divergence, valid = continuity_residual(vel_pred, interior.pos, interior.edge_index,
                                                     **(div_kwargs or {}))
            divergence = divergence[valid]
        if divergence.numel() > 0:
            ld = divergence.square().mean()
    loss = lam_vel * lv + lam_press * lp + lam_div * ld
    return loss, lv.item(), lp.item(), ld.item()


def run_model(model, wall, interior, waveform, args, div_kwargs=None):
    """model(wall, interior, waveform) + the full field loss in one call — the shared
    extension point for train_interior.py and train_interior_ddp.py's training loops
    (and evaluate() below). Owns the --div_method dispatch:
      'graph' — untouched original path: one forward pass on `interior`, divergence from
                interior.edge_index via the graph stencil.
      'tet'   — builds synthetic offset elements around a subset of `interior`'s own
                points (constraints.build_synthetic_offsets), appends them to `interior`
                for a SINGLE combined forward pass (constraints.append_synthetic_offsets),
                then slices the original points back out. No extra pass over the geometry
                backbone — only the (cheap) query-side decoder sees the extra rows.
    """
    if args.lambda_div > 0 and args.div_method == "tet":
        offsets = build_synthetic_offsets(
            interior.pos, interior.wall_dist, interior.batch,
            args.div_n_centers, args.div_k_offsets, args.div_radius, args.div_wall_margin,
            args.div_wall_percentile)
        if offsets is None:
            vel_pred, press_pred = model(wall, interior, waveform)
            divergence = vel_pred.new_zeros(0)
        else:
            combined, orig_idx, offset_idx = append_synthetic_offsets(interior, offsets)
            vel_c, press_c = model(wall, combined, waveform)
            vel_pred, press_pred = vel_c[orig_idx], press_c[orig_idx]
            offset_vel = vel_c[offset_idx].reshape(offsets.center_idx.numel(), offsets.k, -1)
            divergence = synthetic_continuity_residual(vel_pred[offsets.center_idx], offset_vel, offsets)
        loss, lv, lp, ld = field_loss(vel_pred, press_pred, interior, args.pointwise_loss,
                                      args.lambda_vel, args.lambda_press, args.lambda_div,
                                      divergence=divergence)
    else:
        vel_pred, press_pred = model(wall, interior, waveform)
        loss, lv, lp, ld = field_loss(vel_pred, press_pred, interior, args.pointwise_loss,
                                      args.lambda_vel, args.lambda_press, args.lambda_div, div_kwargs)
    return vel_pred, press_pred, loss, lv, lp, ld


@torch.no_grad()
def evaluate(model, loader, waveform, device, norm, args, div_kwargs=None):
    model.eval()
    tot, tv, tp, td, mv, mp, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for wall, interior in loader:
        wall, interior = wall.to(device), interior.to(device)
        vel_pred, press_pred, loss, lv, lp, ld = run_model(model, wall, interior, waveform,
                                                            args, div_kwargs)
        b = wall.num_graphs
        tot += loss.item() * b; tv += lv * b; tp += lp * b; td += ld * b
        # denormalized MSE (physical units): vel in m/s, press in Pa
        mv += ((vel_pred - interior.vel) * norm.vel_std.to(device)).pow(2).mean().item() * b
        mp += ((press_pred - interior.press) * norm.press_std.to(device)).pow(2).mean().item() * b
        n += b
    model.train()
    n = max(n, 1)
    return tot / n, tv / n, tp / n, td / n, mv / n, mp / n


def save_script_copy(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(ckpt_dir, os.path.basename(__file__)))


@torch.no_grad()
def save_interior_sanity(model, wall, interior, waveform, device, norm, out_dir, epoch,
                         point_alpha=0.35, slab_frac=0.06):
    """Predicted vs GT interior |velocity| for one test case: a translucent 3D view
    (interior points over faint wall points — also an alignment check) plus three
    slab cross-sections. Rows = predicted / ground-truth, shared colour scale.

    Rendered with matplotlib (Agg) rather than pyvista/VTK: VTK off-screen creates
    an OpenGL context that collides with the live CUDA context during training and
    also captures the process's stdout fds (swallowing later logs). matplotlib is
    pure-CPU and does neither.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    was_training = model.training
    model.eval()
    wall, interior = wall.to(device), interior.to(device)
    vel_pred, _ = model(wall, interior, waveform)
    if was_training:
        model.train()

    im = (interior.batch == 0)
    wm = (wall.batch == 0)
    pos = interior.pos[im].cpu().numpy()
    wp = wall.pos[wm].cpu().numpy()
    vgt = norm.inverse_vel(interior.vel[im]).norm(dim=-1).cpu().numpy()   # physical m/s
    vpr = norm.inverse_vel(vel_pred[im]).norm(dim=-1).cpu().numpy()
    vmax = float(np.percentile(vgt, 99)) or 1.0
    ext = (pos.max(0) - pos.min(0))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"vel_epoch{epoch:04d}.png")

    fig = plt.figure(figsize=(24, 12))
    sc = None
    for r, (name, mag) in enumerate([("pred", vpr), ("GT", vgt)]):
        # col 0: translucent 3D cloud over faint wall points (alignment + field)
        ax = fig.add_subplot(2, 4, r * 4 + 1, projection="3d")
        ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c="lightgrey", s=2, alpha=0.10, linewidths=0)
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=mag, cmap="turbo", vmin=0, vmax=vmax,
                   s=6, alpha=point_alpha, linewidths=0)
        ax.set_title(f"{name}  3D (transparent)", fontsize=10); ax.set_axis_off()
        # cols 1-3: slab cross-sections (project onto the other two axes)
        for axis in (0, 1, 2):
            ax = fig.add_subplot(2, 4, r * 4 + 2 + axis)
            s = np.abs(pos[:, axis] - np.median(pos[:, axis])) < slab_frac * ext[axis]
            oa = [a for a in (0, 1, 2) if a != axis]
            sc = ax.scatter(pos[s][:, oa[0]], pos[s][:, oa[1]], c=mag[s], cmap="turbo",
                            vmin=0, vmax=vmax, s=8, alpha=0.85, linewidths=0)
            ax.set_title(f"{name}  slab |{'xyz'[axis]}|", fontsize=10)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    if sc is not None:
        fig.colorbar(sc, ax=fig.axes, shrink=0.5, label="|v| (m/s)")
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    print(f"[sanity] {path}  |v| clim=[0, {vmax:.3f}] m/s", flush=True)


def build_model(args, device):
    # common decoder args; n_geo_dist is a backbone arg (goes to bb_kwargs)
    common = dict(dim_emb=args.dim_emb, dim_field=args.dim_field, n_blocks=args.n_blocks,
                  xattn_heads=args.xattn_heads, k_local=args.k_local, n_freqs=args.n_freqs,
                  dim_te=args.dim_te, dim_wfe=args.dim_wfe, n_geo_dist=args.n_geo_dist,
                  geometry_token_pool=args.geometry_token_pool,
                  n_geometry_tokens=args.n_geometry_tokens,
                  token_pool_heads=args.token_pool_heads,
                  token_pool_layers=args.token_pool_layers, wall_gate=args.wall_gate)
    if args.backbone == "hgps":
        model = build_interior_field(
            "hgps", n_cot=args.lap_k, dim_pe=args.dim_pe, n_pre=args.n_pre, n_gps=args.n_gps,
            n_pool=args.n_pool, pool_ratio=args.pool_ratio, pool_knn_k=args.pool_knn_k,
            n_heads=args.n_heads, attn_type=args.attn_type, pe_n_heads=args.pe_n_heads,
            pe_n_layers=args.pe_n_layers, pe_use_mlp=args.pe_use_mlp, **common)
    else:
        model = build_interior_field("pointnet", base_ch=args.base_ch, **common)
    return model.to(device)


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    exclude_keys = load_excluded_keys(args.save_dir) if args.exclude_anomalies else set()
    dataset = build_dataset(args, exclude_keys)
    train_idx, test_idx = split_by_case(dataset, args.train_frac, seed=args.seed)
    print(f"[data] {len(dataset)} samples  train={len(train_idx)}  test={len(test_idx)}")

    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}
    stats = load_or_compute_interior_stats(
        os.path.join(args.save_dir, "interior_stats.pt"), dataset, train_keys,
        n_cases=args.stats_n_cases, frames_per_case=args.stats_frames_per_case, seed=args.seed)
    norm = InteriorNormalize(stats)

    div_kwargs = dict(ridge=args.div_ridge, power=args.div_power,
                      min_neighbors=args.div_min_neighbors, cond_tol=args.div_cond_tol)
    subsample = (InteriorSubsampleGraph(sample_ratio=args.sample_ratio, max_points=args.max_points)
                if (args.lambda_div > 0 and args.div_method == "graph") else
                InteriorSubsample(sample_ratio=args.sample_ratio, max_points=args.max_points))
    dataset.transform = _chain(
        subsample,
        CenterAndRandomRotate(rotate=True),                 # shared centroid + R over wall & interior
        norm,
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)

    waveform = load_waveform(args.waveform_path, length=1000, device=device)
    model = build_model(args, device)
    token_desc = ("downsampled_graph" if args.geometry_token_pool == "none" else
                  f"{args.n_geometry_tokens}_latent_tokens")
    print(f"[model] interior_field/{args.backbone}  global={token_desc}  "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

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
        tot, tv, tp, td, n = 0.0, 0.0, 0.0, 0.0, 0
        for wall, interior in train_loader:
            wall, interior = wall.to(device), interior.to(device)
            if torch.isnan(interior.vel).any() or torch.isnan(interior.press).any():
                print("[warn] NaN in interior targets — skipping batch")
                continue
            optim.zero_grad()
            vel_pred, press_pred, loss, lv, lp, ld = run_model(model, wall, interior, waveform,
                                                                args, div_kwargs)
            if torch.isnan(loss):
                print("[warn] NaN loss — skipping batch"); optim.zero_grad(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optim.step()
            b = wall.num_graphs
            tot += loss.item() * b; tv += lv * b; tp += lp * b; td += ld * b; n += b
        n = max(n, 1)

        te_loss, te_v, te_p, te_d, te_mv, te_mp = evaluate(
            model, test_loader, waveform, device, norm, args, div_kwargs)
        if not warming_up:
            step_scheduler(sched, args, te_loss)
        lr_now = optim.param_groups[0]["lr"]
        div_desc = f"  train_div {td/n:.4g}  test_div {te_d:.4g}" if args.lambda_div > 0 else ""
        print(f"epoch {epoch:3d}  train_loss {tot/n:.4g} (vel {tv/n:.4g} press {tp/n:.4g})  "
              f"test_loss {te_loss:.4g} (vel {te_v:.4g} press {te_p:.4g}){div_desc}  "
              f"test_mse_vel {te_mv:.4g} (m/s)²  test_mse_press {te_mp:.4g} (Pa)²  lr {lr_now:.2e}",
              flush=True)

        if args.wandb:
            wandb.log({"train_loss": tot / n, "train_vel": tv / n, "train_press": tp / n,
                       "train_div": td / n,
                       "test_loss": te_loss, "test_vel": te_v, "test_press": te_p, "test_div": te_d,
                       "test_mse_vel": te_mv, "test_mse_press": te_mp, "lr": lr_now}, step=epoch)

        if args.sanity_every and epoch % args.sanity_every == 0:
            try:
                # fresh test batch each time (test_loader is shuffled) -> a different
                # shape is rendered every sanity epoch
                sw, si = next(iter(test_loader))
                save_interior_sanity(model, sw, si, waveform, device, norm,
                                     os.path.join(ckpt_dir, "sanity"), epoch)
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
conda activate new

python -m angioflow.training.train_interior --backbone hgps \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name interior_hgps_v1v2_latent64_d64_b16_hc \
  --versions aneug_flow,angioflow --frames 0 \
  --decimate_ratio 1.0 \
  --sample_ratio 0.1 --max_points 10000 \
  --dim_emb 64 --dim_field 64 \
  --n_pool 2 --pool_ratio 0.25 \
  --n_gps 4 --n_blocks 2 --k_local 16 \
  --geometry_token_pool latent \
  --n_geometry_tokens 64 \
  --token_pool_heads 4 \
  --token_pool_layers 1 \
  --lambda_vel 1.0 --lambda_press 1.0 \
  --device cuda:0 --batch_size 16 --num_workers 4 \
  --total_epochs 500 --sanity_every 25 \
  --wandb --wandb_project AngioFlow_velocity



python -m angioflow.training.train_interior --backbone hgps \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name interior_hgps_v1v2_graph_d64_b16_hc \
  --versions aneug_flow,angioflow --frames 0 \
  --decimate_ratio 1.0 \
  --sample_ratio 0.1 --max_points 10000 \
  --dim_emb 64 --dim_field 64 \
  --n_pool 2 --pool_ratio 0.25 \
  --n_gps 4 --n_blocks 2 --k_local 16 \
  --geometry_token_pool none \
  --lambda_vel 1.0 --lambda_press 1.0 \
  --device cuda:0 --batch_size 16 --num_workers 4 \
  --total_epochs 500 --sanity_every 25 \
  --wandb --wandb_project AngioFlow_velocity

  
# ablation: same everything, + continuity loss
python -m angioflow.training.train_interior --backbone hgps \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name interior_hgps_v1v2_graph_d64_b16_hc_div \
  --versions aneug_flow,angioflow --frames 0 \
  --decimate_ratio 1.0 \
  --sample_ratio 0.1 --max_points 10000 \
  --dim_emb 64 --dim_field 64 \
  --n_pool 2 --pool_ratio 0.25 \
  --n_gps 4 --n_blocks 2 --k_local 16 \
  --geometry_token_pool none \
  --lambda_vel 1.0 --lambda_press 0.0 \
  --lambda_div 0.25 \
  --device cuda:2 --batch_size 16 --num_workers 4 \
  --total_epochs 500 --sanity_every 25 \
  --wandb --wandb_project AngioFlow_velocity

python -m angioflow.training.train_interior --backbone hgps \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name interior_hgps_v1v2_graph_d96_b8_k32_pre2_gps6_fusion3_pool2 \
  --versions aneug_flow,angioflow --frames 0 \
  --decimate_ratio 1.0 \
  --sample_ratio 0.1 --max_points 10000 \
  --dim_emb 96 --dim_field 96 \
  --n_pre 2 \
  --n_pool 2 --pool_ratio 0.25 --pool_knn_k 8 \
  --n_gps 6 --n_heads 4 \
  --n_blocks 3 --xattn_heads 4 \
  --k_local 32 \
  --geometry_token_pool none \
  --lambda_vel 1.0 --lambda_press 1.0 \
  --device cuda:2 \
  --batch_size 8 --num_workers 4 \
  --total_epochs 500 --sanity_every 25 \
  --wandb --wandb_project AngioFlow_velocity


conda activate new

python -m angioflow.training.train_interior --backbone hgps \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name interior_hgps_v1v2_graph_div_r025_d48_b6_div \
  --versions aneug_flow,angioflow --frames 0 \
  --decimate_ratio 1.0 \
  --sample_ratio 0.25 --max_points 30000 \
  --dim_emb 48 --dim_field 48 \
  --n_pool 2 --pool_ratio 0.25 \
  --n_gps 4 --n_blocks 2 --k_local 16 \
  --geometry_token_pool none \
  --lambda_vel 1.0 --lambda_press 1.0 \
  --lambda_div 1.0 \
  --device cuda:0 --batch_size 6 --num_workers 4 \
  --total_epochs 500 --sanity_every 25 \
  --wandb --wandb_project AngioFlow_velocity


# ablation: small n_geometry_tokens (16, vs. latent64's 64) to cut the global
# cross-attention's GPU cost, k_local raised 16->32 to compensate the lost geometric
# detail with a stronger local surface-gather branch. Pair: no continuity / tet continuity.
python -m angioflow.training.train_interior --backbone hgps \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name interior_hgps_v1v2_latent16_d64_b16_kl32 \
  --versions aneug_flow,angioflow --frames 0 \
  --decimate_ratio 1.0 \
  --sample_ratio 0.1 --max_points 10000 \
  --dim_emb 64 --dim_field 64 \
  --n_pool 2 --pool_ratio 0.25 \
  --n_gps 4 --n_blocks 2 --k_local 32 \
  --geometry_token_pool latent \
  --n_geometry_tokens 16 \
  --token_pool_heads 4 \
  --token_pool_layers 1 \
  --lambda_vel 1.0 --lambda_press 0.0 \
  --device cuda:0 --batch_size 12 --num_workers 4 \
  --total_epochs 500 --sanity_every 25 \
  --lr_schedule reduce_on_plateau \
  --wandb --wandb_project AngioFlow_velocity --wandb_run_id yt89n96i

python -m angioflow.training.train_interior --backbone hgps \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --interior_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior \
  --run_name interior_hgps_v1v2_latent16_d64_b16_kl32_divtet \
  --versions aneug_flow,angioflow --frames 0 \
  --decimate_ratio 1.0 \
  --sample_ratio 0.1 --max_points 10000 \
  --dim_emb 64 --dim_field 64 \
  --n_pool 2 --pool_ratio 0.25 \
  --n_gps 4 --n_blocks 2 --k_local 32 \
  --geometry_token_pool latent \
  --n_geometry_tokens 16 \
  --token_pool_heads 4 \
  --token_pool_layers 1 \
  --lambda_vel 1.0 --lambda_press 0.0 \
  --lambda_div 0.5 --div_method tet \
  --div_n_centers 128 --div_k_offsets 24 --div_radius 0.12 --div_wall_margin 2.0 \
  --div_wall_percentile 0.75 \
  --device cuda:1 --batch_size 8 --num_workers 4 \
  --total_epochs 500 --sanity_every 25 \
  --lr_schedule reduce_on_plateau \
  --wandb --wandb_project AngioFlow_velocity --wandb_run_id 0u8h97i0
  # batch_size 16 (matching the no-continuity companion above) OOM'd on a 24GB 3090 Ti
  # at k_local=32 + div_n_centers=128/div_k_offsets=24; batch_size=8 fit with ~7GB headroom
  # (16.4/23.7GB used) in a 2-epoch debug run -- retune from there if you change any of
  # k_local/div_n_centers/div_k_offsets/sample_ratio/max_points.

"""
