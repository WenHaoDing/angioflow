"""Train PointNet++ WSS prediction with ground-truth interior velocity injection.

This is a self-contained experiment entrypoint.  It intentionally does not touch
``angioflow.training.train`` or the global model registry while the design is
still being tested.

Dataset contract
----------------
Uses ``AngioFlowWallInteriorDataset`` / ``AngioFlowWallInteriorDatasetT0`` which
return ``(wall_data, interior_data)``.  PyG batches these element-wise, so the
training loop receives ``(wall_batch, interior_batch)`` and calls::

    pred = model(wall_batch, interior_batch, waveform)

The ground-truth interior velocity is an input at validation/test time as well.
That makes this an oracle conditional experiment: it measures whether interior
velocity helps the WSS branch, not whether velocity can be inferred from geometry.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
os.environ.setdefault(
    "WANDB_API_KEY",
    "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",
)
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from angioflow.config.defaults import (
    add_data_args,
    add_encoder_args,
    add_loss_args,
    add_misc_args,
    add_optim_args,
)
from angioflow.data.datasets import load_excluded_keys
from angioflow.data.datasets_interior import (
    AngioFlowWallInteriorDataset,
    AngioFlowWallInteriorDatasetT0,
)
from angioflow.data.normalize import load_or_compute_stats
from angioflow.losses import directional_loss, log_mag_loss, tv_smooth_loss
from angioflow.models.pointnet2_teacher_velocity import AngioPointNet2TeacherVelocity
from angioflow.transform import AneurysmCrop
from angioflow.transform.rotate import _random_rotation_matrices


def _chain(*transforms):
    def _apply(sample):
        for t in transforms:
            sample = t(sample)
        return sample
    return _apply


class PairCenterAndRandomRotate:
    """Apply one centering/rotation transform consistently to wall and interior.

    Wall/interior positions must stay in the same coordinate system for kNN
    injection.  Interior velocity, wall normals, and WSS rotate as vectors.
    """

    def __init__(self, rotate: bool = True):
        self.rotate = rotate

    def __call__(self, sample):
        wall, interior = sample
        center = wall.pos.mean(0, keepdim=True)
        wall.pos = wall.pos - center
        interior.pos = interior.pos - center

        if self.rotate:
            R = _random_rotation_matrices(
                1, device=wall.pos.device, dtype=wall.pos.dtype
            )[0]
            wall.pos = wall.pos @ R.T
            wall.normals = wall.normals @ R.T
            wall.y = wall.y @ R.T
            interior.pos = interior.pos @ R.T
            interior.vel = interior.vel @ R.T
        return wall, interior


class PairPosYNormalize:
    """Normalize wall pos/y and interior pos with the same wall-derived stats."""

    def __init__(self, stats):
        self.pos_mean = stats["pos_mean"]
        self.pos_std = stats["pos_std"]
        self.y_mean = stats["y_mean"]
        self.y_std = stats["y_std"]
        self.pos_scale = stats["pos_std"].mean().clamp_min(1e-8)

    def __call__(self, sample):
        wall, interior = sample
        wall.pos = (wall.pos - self.pos_mean) / self.pos_std
        wall.y = (wall.y - self.y_mean) / self.y_std
        interior.pos = (interior.pos - self.pos_mean) / self.pos_std
        interior.wall_dist = interior.wall_dist / self.pos_scale
        return wall, interior


class PairWallCrop:
    """Crop only the wall graph; keep the full sampled interior cloud."""

    def __init__(self, low=0.0, high=0.25, aneurysm_only=False):
        self.wall_crop = AneurysmCrop(low=low, high=high, aneurysm_only=aneurysm_only)

    def __call__(self, sample):
        wall, interior = sample
        return self.wall_crop(wall), interior


def split_by_case(dataset, train_frac=0.8, seed=0):
    # Stratified by version, matching the general training script.
    per_version = defaultdict(list)
    for v, c, _ in dataset._index:
        per_version[v].append((v, c))

    rng = random.Random(seed)
    train_keys = set()
    for v in sorted(per_version):
        keys = sorted(set(per_version[v]))
        rng.shuffle(keys)
        n_train = int(len(keys) * train_frac)
        train_keys.update(keys[:n_train])
        print(f"[split] {v}: {n_train} train  {len(keys)-n_train} test")

    train_idx, test_idx = [], []
    for i, (v, c, _) in enumerate(dataset._index):
        (train_idx if (v, c) in train_keys else test_idx).append(i)
    return train_idx, test_idx


def load_waveform(path, length=1000, device="cuda"):
    raw = torch.from_numpy(np.loadtxt(path, delimiter=",")).float()
    raw = raw[: len(raw) // 5]
    wf = F.interpolate(
        raw.view(1, 1, -1),
        size=(length,),
        mode="linear",
        align_corners=True,
    ).permute(0, 2, 1)
    return wf.to(device)


def _pointwise_loss(pred, y, kind, huber_delta):
    if kind == "mse":
        return F.mse_loss(pred, y)
    return F.huber_loss(pred, y, delta=huber_delta)


def compute_velocity_scale(dataset, train_keys, n_cases=100, frames_per_case=8, seed=0):
    """Scalar RMS over GT interior velocity components from sampled train cases."""
    rng = random.Random(seed)
    keys = list(train_keys)
    rng.shuffle(keys)
    sample = keys[: min(n_cases, len(keys))]

    sum_sq = 0.0
    count = 0
    for v, cid in sample:
        frame_ids = rng.sample(
            range(dataset.n_frames),
            min(frames_per_case, dataset.n_frames),
        )
        for t in frame_ids:
            frame = torch.load(dataset._interior_frame_path(v, cid, t), weights_only=False)
            vel = frame["vel"].float()
            sum_sq += vel.pow(2).sum().item()
            count += vel.numel()

    if count == 0:
        return 1.0
    return float(max((sum_sq / count) ** 0.5, 1e-8))


def load_or_compute_velocity_scale(path, dataset, train_keys, n_cases=100,
                                   frames_per_case=8, seed=0):
    if os.path.exists(path):
        obj = torch.load(path, weights_only=False)
        scale = float(obj["velocity_scale"])
        print(f"[stats] loaded velocity_scale={scale:.4g} from {path}")
        return scale

    scale = compute_velocity_scale(
        dataset,
        train_keys,
        n_cases=n_cases,
        frames_per_case=frames_per_case,
        seed=seed,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "velocity_scale": scale,
            "n_cases": min(n_cases, len(train_keys)),
            "n_frames_per_case": frames_per_case,
        },
        path,
    )
    print(f"[stats] computed velocity_scale={scale:.4g} -> saved to {path}")
    return scale


@torch.no_grad()
def evaluate(model, loader, waveform, device, y_mean, y_std, log_eps, max_wss,
             huber_delta, pointwise_kind="huber"):
    model.eval()
    tot_logmag, tot_huber, tot_mse, n = 0.0, 0.0, 0.0, 0
    for wall, interior in loader:
        wall = wall.to(device)
        interior = interior.to(device)
        pred = model(wall, interior, waveform)
        tot_logmag += log_mag_loss(pred, wall.y, y_mean, y_std, log_eps, max_wss).item() * wall.num_graphs
        tot_huber += _pointwise_loss(pred, wall.y, pointwise_kind, huber_delta).item() * wall.num_graphs
        tot_mse += ((pred - wall.y) * y_std).pow(2).mean().item() * wall.num_graphs
        n += wall.num_graphs
    model.train()
    return tot_logmag / max(n, 1), tot_huber / max(n, 1), tot_mse / max(n, 1)


@torch.no_grad()
def save_wss_sanity_teacher(model, loader, waveform, device, stats, out_dir, epoch,
                            max_wss: float = 20.0, n_samples: int = 3):
    """Paired-dataset version of visualize.sanity.save_wss_sanity."""
    import sys

    from angioflow.visualize.sanity import _make_surface_mesh

    os.makedirs(out_dir, exist_ok=True)
    y_mean = stats["y_mean"].cpu().numpy()
    y_std = stats["y_std"].cpu().numpy()

    model.eval()
    samples = []
    for wall, interior in loader:
        wall = wall.to(device)
        interior = interior.to(device)
        pred = model(wall, interior, waveform)

        for g in range(wall.num_graphs):
            node_mask = wall.batch == g
            first_node = node_mask.nonzero(as_tuple=False)[0, 0].item()
            edge_in_g = node_mask[wall.edge_index[0]]
            ei_local = wall.edge_index[:, edge_in_g].cpu().numpy() - first_node

            pos_np = wall.pos[node_mask].cpu().numpy()
            gt, pd = wall.y[node_mask], pred[node_mask]

            gt_denorm = gt.cpu().numpy() * y_std + y_mean
            pd_denorm = pd.cpu().numpy() * y_std + y_mean
            gt_mag = np.linalg.norm(gt_denorm, axis=1).clip(0, max_wss)
            pred_mag = np.linalg.norm(pd_denorm, axis=1).clip(0, max_wss)
            err = np.abs(gt_mag - pred_mag)
            label = f"{wall.version[g]}/{wall.case_id[g]}"
            samples.append((pos_np, ei_local, gt_mag, pred_mag, err, label))
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break
    model.train()

    if not samples:
        print("[sanity] no samples — skipping", flush=True)
        return

    # PyVista off-screen rendering can corrupt stdout/stderr; mirror the general
    # sanity helper's fd preservation.
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    py_stdout = sys.stdout
    py_stderr = sys.stderr
    try:
        import pyvista as pv

        pv.start_xvfb()
        for i, (pos, ei, gt_mag, pred_mag, err, label) in enumerate(samples):
            vmin = min(gt_mag.min(), pred_mag.min())
            vmax = max(gt_mag.max(), pred_mag.max())
            cols = [
                (gt_mag, "GT (Pa/s)", "coolwarm", vmin, vmax),
                (pred_mag, "Pred (Pa/s)", "coolwarm", vmin, vmax),
                (err, "|Error| (Pa/s)", "Oranges", 0.0, err.max()),
            ]

            pl = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2100, 750))
            pl.link_views()
            for col, (vals, title, cmap, lo, hi) in enumerate(cols):
                pl.subplot(0, col)
                mesh = _make_surface_mesh(pos, ei)
                mesh.point_data["wss"] = vals
                pl.add_mesh(
                    mesh,
                    scalars="wss",
                    cmap=cmap,
                    clim=[lo, hi],
                    show_scalar_bar=True,
                    scalar_bar_args={"title": title, "vertical": True},
                )
                pl.add_title(f"{label}  {title}", font_size=11)
            pl.camera_position = "iso"
            out_path = os.path.join(out_dir, f"sanity_epoch_{epoch:04d}_s{i}.png")
            pl.screenshot(out_path)
            pl.close()
    finally:
        os.dup2(stdout_fd, 1)
        os.close(stdout_fd)
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        sys.stdout = py_stdout
        sys.stderr = py_stderr

    for i in range(len(samples)):
        out_path = os.path.join(out_dir, f"sanity_epoch_{epoch:04d}_s{i}.png")
        print(f"[sanity] saved {out_path}", flush=True)


def save_ckpt(path, model, optim, sched, epoch, best_loss, args=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "args": vars(args) if args is not None else None,
        },
        path,
    )


def save_script_copy(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(ckpt_dir, os.path.basename(__file__)))


def maybe_resume(ckpt_dir, model, optim, sched):
    candidates = [os.path.join(ckpt_dir, "last.pth")]
    epoch_ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")),
        key=lambda p: int(os.path.basename(p).replace("epoch_", "").replace(".pth", "")),
    )
    if epoch_ckpts:
        candidates.append(epoch_ckpts[-1])
    for path in candidates:
        if os.path.exists(path):
            chk = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(chk["model"])
            optim.load_state_dict(chk["optim"])
            sched.load_state_dict(chk["sched"])
            print(f"[resume] {path} @ epoch {chk['epoch']+1}, best={chk['best_loss']:.4g}")
            return chk["epoch"] + 1, chk["best_loss"]
    return 0, float("inf")


def add_interior_data_args(parser):
    g = parser.add_argument_group("interior dataset")
    g.add_argument("--pool_ratio", type=float, default=0.25)
    g.add_argument("--sample_ratio", type=float, default=0.5)
    g.add_argument("--near_decay", type=float, default=0.1)
    g.add_argument("--far_floor", type=float, default=0.25)
    g.add_argument("--min_wall_dist", type=float, default=1e-4)
    g.add_argument("--interior_dir",
                   default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior")
    g.add_argument("--interior_sanity_first", type=int, default=0)
    g.add_argument("--freeze_dataset", action="store_true",
                   help="only use cases already fully processed on disk; never "
                        "process new/incomplete cases (e.g. mid-download)")
    g.add_argument("--stats_path", default=None,
                   help="default: <save_dir>/stats_interior_teacher_velocity.pt")
    g.add_argument("--velocity_stats_path", default=None,
                   help="default: <save_dir>/stats_interior_velocity.pt")
    return parser


def add_teacher_pointnet_args(parser):
    g = parser.add_argument_group("model:pointnet2_teacher_velocity")
    g.add_argument("--dim_emb", type=int, default=64)
    g.add_argument("--base_ch", type=int, default=128,
                   help="SA/FP channel width base (c1=base_ch, c2=2x, c3=8x)")
    g.add_argument("--sa_ratio1", type=float, default=0.25, help="FPS ratio for SA1")
    g.add_argument("--sa_ratio2", type=float, default=0.25, help="FPS ratio for SA2")
    g.add_argument("--sa_r1", type=float, default=0.3, help="ball radius for SA1")
    g.add_argument("--sa_r2", type=float, default=0.5, help="ball radius for SA2")
    g.add_argument("--injection_hidden", type=int, default=128)
    g.add_argument("--injection_k", type=int, default=32)
    g.add_argument("--velocity_scale", type=float, default=0.0,
                   help="GT velocity scale; <=0 computes RMS from train interior frames")
    g.add_argument("--sanity_every", type=int, default=50,
                   help="render WSS sanity images every N epochs; <=0 disables")
    return parser


def get_args():
    p = argparse.ArgumentParser()
    add_data_args(p)
    add_encoder_args(p)
    add_loss_args(p)
    add_optim_args(p)
    add_misc_args(p)
    add_interior_data_args(p)
    add_teacher_pointnet_args(p)
    return p.parse_args()


def build_model(args, velocity_scale, device):
    return AngioPointNet2TeacherVelocity(
        n_geo_dist=args.n_geo_dist,
        dim_te=args.dim_te,
        dim_wfe=args.dim_wfe,
        dim_emb=args.dim_emb,
        dim_out=3,
        base_ch=args.base_ch,
        sa_ratios=(args.sa_ratio1, args.sa_ratio2),
        sa_radii=(args.sa_r1, args.sa_r2),
        injection_hidden=args.injection_hidden,
        injection_k=args.injection_k,
        velocity_scale=velocity_scale,
    ).to(device)


def build_dataset(args, exclude_keys):
    versions = args.versions.split(",") if args.versions else None
    ds_kwargs = dict(
        case_root_angioflow=args.case_root_angioflow,
        case_root_aneug_flow=args.case_root_aneug_flow,
        save_dir=args.save_dir,
        n_frames=args.n_frames,
        decimate_ratio=args.decimate_ratio,
        decimate_method=args.decimate_method,
        lap_k=args.lap_k,
        exclude_keys=exclude_keys,
        pool_ratio=args.pool_ratio,
        sample_ratio=args.sample_ratio,
        near_decay=args.near_decay,
        far_floor=args.far_floor,
        min_wall_dist=args.min_wall_dist,
        interior_dir=args.interior_dir,
        sanity_first=args.interior_sanity_first,
        freeze_dataset=args.freeze_dataset,
    )
    if args.frames == "all":
        dataset = AngioFlowWallInteriorDataset(**ds_kwargs)
    else:
        frames = [int(t) for t in args.frames.split(",")]
        dataset = AngioFlowWallInteriorDatasetT0(
            **ds_kwargs,
            versions=versions,
            frames=frames,
        )
    return dataset


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    exclude_keys = load_excluded_keys(args.save_dir) if args.exclude_anomalies else set()
    dataset = build_dataset(args, exclude_keys)
    train_idx, test_idx = split_by_case(dataset, args.train_frac, seed=args.seed)
    print(f"[data] {len(dataset)} samples  train={len(train_idx)}  test={len(test_idx)}")

    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}
    stats_path = args.stats_path or os.path.join(args.save_dir, "stats_interior_teacher_velocity.pt")
    stats = load_or_compute_stats(
        stats_path,
        dataset,
        train_keys,
        n_cases=args.stats_n_cases,
        frames_per_case=args.stats_frames_per_case,
        seed=args.seed,
    )
    print(f"[stats] pos_std={stats['pos_std'].mean().item():.4f}")

    if args.velocity_scale > 0:
        velocity_scale = args.velocity_scale
        print(f"[stats] using CLI velocity_scale={velocity_scale:.4g}")
    else:
        velocity_stats_path = args.velocity_stats_path or os.path.join(
            args.save_dir, "stats_interior_velocity.pt"
        )
        velocity_scale = load_or_compute_velocity_scale(
            velocity_stats_path,
            dataset,
            train_keys,
            n_cases=args.stats_n_cases,
            frames_per_case=args.stats_frames_per_case,
            seed=args.seed,
        )

    y_mean = stats["y_mean"].to(device)
    y_std = stats["y_std"].to(device)
    huber_delta = 1.0
    print(
        f"[train] pointnet2_teacher_velocity  log-mag eps={args.log_eps:.2f} "
        f"max={args.max_wss:.1f} Pa/s  pointwise={args.pointwise_loss}  "
        f"velocity_scale={velocity_scale:.4g}"
    )

    transforms = [PairCenterAndRandomRotate(rotate=True)]
    transforms += [
        PairPosYNormalize(stats),
        PairWallCrop(
            low=args.mask_low,
            high=args.mask_high,
            aneurysm_only=args.crop_aneurysm_only,
        ),
    ]
    dataset.transform = _chain(*transforms)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    waveform = load_waveform(args.waveform_path, length=1000, device=device)
    model = build_model(args, velocity_scale, device)
    print(f"[model] pointnet2_teacher_velocity  {sum(p.numel() for p in model.parameters()):,} parameters")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=args.total_epochs,
        eta_min=args.lr * args.lr_end_factor,
    )

    ckpt_dir = os.path.join(args.ckpt_root, args.run_name)
    save_script_copy(ckpt_dir)
    ckpt_last = os.path.join(ckpt_dir, "last.pth")
    ckpt_best = os.path.join(ckpt_dir, "best.pth")
    epoch_start, best_loss = 0, float("inf")
    if args.resume:
        epoch_start, best_loss = maybe_resume(ckpt_dir, model, optim, sched)

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args) | {"resolved_velocity_scale": velocity_scale},
            resume="allow",
            id=args.wandb_run_id,
        )

    for epoch in range(epoch_start, args.total_epochs):
        model.train()
        tot_logmag, tot_huber, tot_smooth, tot_dir, n = 0.0, 0.0, 0.0, 0.0, 0
        gnorm_sum, gnorm_max, n_clipped, n_steps = 0.0, 0.0, 0, 0

        for wall, interior in train_loader:
            wall = wall.to(device)
            interior = interior.to(device)
            if (
                torch.isnan(wall.pos).any()
                or torch.isnan(wall.y).any()
                or torch.isnan(interior.pos).any()
                or torch.isnan(interior.vel).any()
            ):
                print("[warn] NaN in input/target — skipping batch")
                continue

            optim.zero_grad()
            pred = model(wall, interior, waveform)
            loss_logmag = log_mag_loss(pred, wall.y, y_mean, y_std, args.log_eps, args.max_wss)
            loss_huber = _pointwise_loss(pred, wall.y, args.pointwise_loss, huber_delta)
            loss_smooth = tv_smooth_loss(pred, wall.edge_index)
            loss_dir = directional_loss(pred, wall.normals, y_mean, y_std)
            loss = (
                args.lambda_logmag * loss_logmag
                + args.lambda_huber * loss_huber
                + args.lambda_smooth * loss_smooth
                + args.lambda_dir * loss_dir
            )
            if torch.isnan(loss):
                print(
                    f"[warn] NaN loss — pred nan={torch.isnan(pred).sum()}, "
                    f"y nan={torch.isnan(wall.y).sum()}, skipping"
                )
                optim.zero_grad()
                continue

            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip_norm,
            )
            gnorm_sum += gnorm.item()
            gnorm_max = max(gnorm_max, gnorm.item())
            n_clipped += gnorm.item() > args.grad_clip_norm
            n_steps += 1
            optim.step()

            tot_logmag += loss_logmag.item() * wall.num_graphs
            tot_huber += loss_huber.item() * wall.num_graphs
            tot_smooth += loss_smooth.item() * wall.num_graphs
            tot_dir += loss_dir.item() * wall.num_graphs
            n += wall.num_graphs

        train_logmag = tot_logmag / max(n, 1)
        train_huber = tot_huber / max(n, 1)
        train_smooth = tot_smooth / max(n, 1)
        train_dir = tot_dir / max(n, 1)
        gnorm_mean = gnorm_sum / max(n_steps, 1)
        clip_frac = n_clipped / max(n_steps, 1)
        sched.step()

        test_logmag, test_huber, test_mse_denorm = evaluate(
            model,
            test_loader,
            waveform,
            device,
            y_mean,
            y_std,
            args.log_eps,
            args.max_wss,
            huber_delta,
            pointwise_kind=args.pointwise_loss,
        )
        lr_now = optim.param_groups[0]["lr"]
        print(
            f"epoch {epoch:3d}  "
            f"train_logmag {train_logmag:.4g}  train_huber {train_huber:.4g}  "
            f"train_smooth {train_smooth:.4g}  train_dir {train_dir:.4g} (Pa/s)  "
            f"test_logmag {test_logmag:.4g}  test_huber {test_huber:.4g}  "
            f"test_mse_denorm {test_mse_denorm:.4g} (Pa/s)²  lr {lr_now:.2e}  "
            f"gnorm {gnorm_mean:.3g} (max {gnorm_max:.3g}, clipped {clip_frac:.0%})",
            flush=True,
        )

        if args.wandb:
            wandb.log(
                {
                    "train_logmag": train_logmag,
                    "train_huber": train_huber,
                    "train_smooth": train_smooth,
                    "train_dir": train_dir,
                    "test_logmag": test_logmag,
                    "test_huber": test_huber,
                    "test_mse_denorm": test_mse_denorm,
                    "lr": lr_now,
                    "grad_norm_mean": gnorm_mean,
                    "grad_norm_max": gnorm_max,
                    "grad_clip_frac": clip_frac,
                },
                step=epoch,
            )

        if args.sanity_every > 0 and epoch % args.sanity_every == 0:
            sanity_dir = os.path.join(ckpt_dir, "sanity")
            save_wss_sanity_teacher(
                model,
                test_loader,
                waveform,
                device,
                stats,
                sanity_dir,
                epoch,
                max_wss=args.max_wss,
            )

        test_loss = test_logmag + test_huber
        save_ckpt(ckpt_last, model, optim, sched, epoch, best_loss, args)
        if epoch % 100 == 0:
            save_ckpt(
                os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"),
                model,
                optim,
                sched,
                epoch,
                best_loss,
                args,
            )
        if test_loss < best_loss:
            best_loss = test_loss
            save_ckpt(ckpt_best, model, optim, sched, epoch, best_loss, args)
            print(f"  ↳ new best @ epoch {epoch} (test_loss={best_loss:.4g})")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main(get_args())


"""
Example:

conda activate new
python -m angioflow.training.train --model pointnet2 \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full \
  --run_name teacher_velocity_baseline_pointnet2_fullg_v2only_fix \
  --versions angioflow \
  --decimate_ratio 1.0 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_smooth 0.1 --lambda_logmag 1 --lambda_dir 0.1 \
  --total_epochs 500 --wandb





"""
