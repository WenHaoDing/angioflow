"""Train AngioPointNet2 on AngioFlowWallDatasetT0 to predict per-node WSS at t=0.

Single-GPU. Same data pipeline as train_transformer_t0.py; model and args differ.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
os.environ['WANDB_API_KEY'] = "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"
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from angioflow.data.datasets import load_excluded_keys
from angioflow.data.datasets_t0 import AngioFlowWallDatasetT0
from angioflow.data.datasets import AngioFlowWallDataset
from angioflow.data.normalize import load_or_compute_stats
from angioflow.models.pointnet2 import AngioPointNet2
from angioflow.losses import log_mag_loss, tv_smooth_loss, directional_loss
from angioflow.transform import AneurysmCrop, CenterAndRandomRotate, PosYNormalize
from angioflow.visualize.sanity import save_wss_sanity


def _chain(*transforms):
    def _apply(data):
        for t in transforms:
            data = t(data)
        return data
    return _apply


def split_by_case(dataset, train_frac=0.8, seed=0):
    # Stratified by version: split each version 80/20 independently so the
    # test set always contains cases from every version present in the dataset.
    from collections import defaultdict
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
    wf = F.interpolate(raw.view(1, 1, -1), size=(length,),
                       mode="linear", align_corners=True).permute(0, 2, 1)
    return wf.to(device)


def _pointwise_loss(pred, y, kind, huber_delta):
    if kind == "mse":
        return F.mse_loss(pred, y)
    return F.huber_loss(pred, y, delta=huber_delta)


@torch.no_grad()
def evaluate(model, loader, waveform, device, y_mean, y_std, log_eps, max_wss, huber_delta,
             pointwise_kind="huber"):
    model.eval()
    tot_logmag, tot_huber, tot_mse, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, waveform)
        tot_logmag += log_mag_loss(pred, batch.y, y_mean, y_std, log_eps, max_wss).item() * batch.num_graphs
        tot_huber  += _pointwise_loss(pred, batch.y, pointwise_kind, huber_delta).item() * batch.num_graphs
        tot_mse    += ((pred - batch.y) * y_std).pow(2).mean().item() * batch.num_graphs
        n += batch.num_graphs
    model.train()
    return tot_logmag / max(n, 1), tot_huber / max(n, 1), tot_mse / max(n, 1)


def save_ckpt(path, model, optim, sched, epoch, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }, path)


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


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    exclude_keys = load_excluded_keys(args.save_dir) if args.exclude_anomalies else set()
    versions = args.versions.split(",") if args.versions else None
    dataset = AngioFlowWallDatasetT0(
        case_root_angioflow=args.case_root_angioflow,
        case_root_aneug_flow=args.case_root_aneug_flow,
        save_dir=args.save_dir,
        n_frames=args.n_frames,
        decimate_ratio=args.decimate_ratio,
        decimate_method=args.decimate_method,
        lap_k=args.lap_k,
        exclude_keys=exclude_keys,
        versions=versions,
    )
    train_idx, test_idx = split_by_case(dataset, args.train_frac, seed=args.seed)
    print(f"[data] {len(dataset)} samples  train={len(train_idx)}  test={len(test_idx)}")

    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}
    stats = load_or_compute_stats(
        os.path.join(args.save_dir, "stats.pt"),
        dataset, train_keys,
        n_cases=args.stats_n_cases, frames_per_case=args.stats_frames_per_case,
        seed=args.seed,
    )
    print(f"[stats] pos_std={stats['pos_std'].mean().item():.4f}  "
          f"(sa_r1={args.sa_r1:.3f}, sa_r2={args.sa_r2:.3f} in normalised units)")
    y_mean = stats['y_mean'].to(device)   # [3]
    y_std  = stats['y_std'].to(device)    # [3]
    # delta=1.0 in log space: errors > e^1 ≈ 2.7× multiplicative factor get linear gradient
    huber_delta = 1.0
    print(f"[train] log-mag loss  eps={args.log_eps:.2f} Pa/s  max={args.max_wss:.1f} Pa/s  "
          f"huber_delta={huber_delta:.2f} (log scale)")
    dataset.transform = _chain(
        CenterAndRandomRotate(rotate=True),
        PosYNormalize(stats),
        AneurysmCrop(low=args.mask_low, high=args.mask_high,
                     aneurysm_only=args.crop_aneurysm_only),
    )

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    waveform = load_waveform(args.waveform_path, length=1000, device=device)

    model = AngioPointNet2(
        n_geo_dist=args.n_geo_dist,
        dim_te=args.dim_te,
        dim_wfe=args.dim_wfe,
        dim_emb=args.dim_emb,
        dim_out=3,
        base_ch=args.base_ch,
        sa_ratios=(args.sa_ratio1, args.sa_ratio2),
        sa_radii=(args.sa_r1, args.sa_r2),
    ).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # cosine: slow decay at start and end — more epochs spent at small lr to converge
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.total_epochs, eta_min=args.lr * args.lr_end_factor)

    ckpt_dir  = os.path.join(args.ckpt_root, args.run_name)
    ckpt_last = os.path.join(ckpt_dir, "last.pth")
    ckpt_best = os.path.join(ckpt_dir, "best.pth")
    epoch_start, best_loss = 0, float("inf")
    if args.resume:
        epoch_start, best_loss = maybe_resume(ckpt_dir, model, optim, sched)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args),
                   resume="allow", id=args.wandb_run_id)

    for epoch in range(epoch_start, args.total_epochs):
        model.train()
        tot_logmag, tot_huber, tot_smooth, tot_dir, n = 0.0, 0.0, 0.0, 0.0, 0
        gnorm_sum, gnorm_max, n_clipped, n_steps = 0.0, 0.0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            if torch.isnan(batch.pos).any() or torch.isnan(batch.y).any():
                print("[warn] NaN in input/target — skipping batch")
                continue
            optim.zero_grad()
            pred = model(batch, waveform)
            loss_logmag = log_mag_loss(pred, batch.y, y_mean, y_std, args.log_eps, args.max_wss)
            loss_huber  = _pointwise_loss(pred, batch.y, args.pointwise_loss, huber_delta)
            loss_smooth = tv_smooth_loss(pred, batch.edge_index)
            loss_dir    = directional_loss(pred, batch.normals, y_mean, y_std)
            loss = args.lambda_logmag * loss_logmag + args.lambda_huber * loss_huber \
                 + args.lambda_smooth * loss_smooth + args.lambda_dir * loss_dir
            if torch.isnan(loss):
                print(f"[warn] NaN loss — pred nan={torch.isnan(pred).sum()}, "
                      f"y nan={torch.isnan(batch.y).sum()}, skipping")
                optim.zero_grad()
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            gnorm_sum += gnorm.item()
            gnorm_max = max(gnorm_max, gnorm.item())
            n_clipped += gnorm.item() > 1.0
            n_steps += 1
            optim.step()
            tot_logmag += loss_logmag.item() * batch.num_graphs
            tot_huber  += loss_huber.item()  * batch.num_graphs
            tot_smooth += loss_smooth.item() * batch.num_graphs
            tot_dir    += loss_dir.item()    * batch.num_graphs
            n += batch.num_graphs
        train_logmag = tot_logmag / max(n, 1)
        train_huber  = tot_huber  / max(n, 1)
        train_smooth = tot_smooth / max(n, 1)
        train_dir    = tot_dir    / max(n, 1)
        gnorm_mean = gnorm_sum / max(n_steps, 1)
        clip_frac  = n_clipped / max(n_steps, 1)
        sched.step()

        test_logmag, test_huber, test_mse_denorm = evaluate(
            model, test_loader, waveform, device,
            y_mean, y_std, args.log_eps, args.max_wss, huber_delta,
            pointwise_kind=args.pointwise_loss)
        lr_now = optim.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  "
              f"train_logmag {train_logmag:.4g}  train_huber {train_huber:.4g}  train_smooth {train_smooth:.4g}  "
              f"train_dir {train_dir:.4g} (Pa/s)  "
              f"test_logmag {test_logmag:.4g}  test_huber {test_huber:.4g}  "
              f"test_mse_denorm {test_mse_denorm:.4g} (Pa/s)²  lr {lr_now:.2e}  "
              f"gnorm {gnorm_mean:.3g} (max {gnorm_max:.3g}, clipped {clip_frac:.0%})",
              flush=True)

        if args.wandb:
            wandb.log({"train_logmag": train_logmag, "train_huber": train_huber,
                       "train_smooth": train_smooth, "train_dir": train_dir,
                       "test_logmag": test_logmag, "test_huber": test_huber,
                       "test_mse_denorm": test_mse_denorm, "lr": lr_now,
                       "grad_norm_mean": gnorm_mean, "grad_norm_max": gnorm_max,
                       "grad_clip_frac": clip_frac}, step=epoch)

        if epoch % 50 == 0:
            sanity_dir = os.path.join(ckpt_dir, "sanity")
            save_wss_sanity(model, test_loader, waveform, device, stats, sanity_dir, epoch,
                            max_wss=args.max_wss)

        test_loss = test_logmag + test_huber
        save_ckpt(ckpt_last, model, optim, sched, epoch, best_loss)
        if epoch % 100 == 0:
            save_ckpt(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"),
                      model, optim, sched, epoch, best_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            save_ckpt(ckpt_best, model, optim, sched, epoch, best_loss)
            print(f"  ↳ new best @ epoch {epoch} (test_loss={best_loss:.4g})")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--case_root_angioflow", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged")
    p.add_argument("--case_root_aneug_flow", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data")
    p.add_argument("--save_dir",     default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_v2")
    p.add_argument("--n_frames", type=int, default=80)
    p.add_argument("--decimate_ratio", type=float, default=0.5)
    p.add_argument("--decimate_method", choices=["quadric", "fps"], default="quadric")
    p.add_argument("--lap_k", type=int, default=8)
    p.add_argument("--waveform_path", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/_meta/waveform_yiying.txt")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--exclude_anomalies", action="store_true", default=True)
    p.add_argument("--no_exclude_anomalies", dest="exclude_anomalies", action="store_false")
    p.add_argument("--versions", default=None, help="comma-separated versions to use, e.g. 'aneug_flow' or 'aneug_flow,angioflow'")
    p.add_argument("--mask_low",  type=float, default=0.0)
    p.add_argument("--mask_high", type=float, default=0.25)
    p.add_argument("--crop_aneurysm_only", action="store_true",
                   help="deterministic crop to the aneurysm region (global_geo_dist == 0); "
                        "disables the random parent-vessel ring")
    p.add_argument("--stats_n_cases", type=int, default=100)
    p.add_argument("--stats_frames_per_case", type=int, default=8)
    # model
    p.add_argument("--n_geo_dist",  type=int,   default=3)
    p.add_argument("--dim_pe",      type=int,   default=8)
    p.add_argument("--dim_te",      type=int,   default=8)
    p.add_argument("--dim_wfe",     type=int,   default=8)
    p.add_argument("--dim_emb",     type=int,   default=64)
    p.add_argument("--base_ch",     type=int,   default=128,   help="SA/FP channel width base (c1=base_ch, c2=2x, c3=8x)")
    p.add_argument("--sa_ratio1",   type=float, default=0.25, help="FPS ratio for SA1")
    p.add_argument("--sa_ratio2",   type=float, default=0.25, help="FPS ratio for SA2")
    p.add_argument("--sa_r1",       type=float, default=0.3,  help="ball radius for SA1 (normalised units)")
    p.add_argument("--sa_r2",       type=float, default=0.5,  help="ball radius for SA2 (normalised units)")
    p.add_argument("--pe_n_heads",  type=int,   default=2)
    p.add_argument("--pe_n_layers", type=int,   default=1)
    # loss
    p.add_argument("--max_wss",       type=float, default=35.0, help="WSS magnitude cap (Pa/s) for log-mag loss")
    p.add_argument("--log_eps",       type=float, default=0.1,  help="WSS magnitude floor (Pa/s) to avoid log(0)")
    p.add_argument("--pointwise_loss", choices=["huber", "mse"], default="huber",
                   help="pointwise regression loss on normalized predictions")
    p.add_argument("--lambda_huber",  type=float, default=5.0,
                   help="weight on the pointwise (huber or mse) loss")
    p.add_argument("--lambda_smooth", type=float, default=0.1,
                   help="weight for TV smoothness regulariser on predictions")
    p.add_argument("--lambda_logmag", type=float, default=0.1,
                   help="weight for log-magnitude loss on predictions")
    p.add_argument("--lambda_dir",    type=float, default=0.01,
                   help="weight for directional loss: |wss_pred · n| in Pa/s "
                        "(WSS must be tangent to the wall)")
    # optim
    p.add_argument("--lr",             type=float, default=0.001)
    p.add_argument("--lr_end_factor",  type=float, default=0.05)
    p.add_argument("--total_epochs",   type=int,   default=500)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--num_workers",    type=int,   default=4)
    # misc
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--run_name",     default="pointnet2_t0")
    p.add_argument("--ckpt_root",    default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_checkpoints",
                   help="root dir for checkpoints; run saves to <ckpt_root>/<run_name>/")
    p.add_argument("--wandb",         action="store_true")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--wandb_project", default="AngioFlow")
    p.add_argument("--wandb_run_id",  default=None, help="existing run ID to re-attach to (find it in the wandb UI URL)")
    p.add_argument("--resume",       action="store_true",  default=True)

    args = p.parse_args()
    main(args)


"""
conda activate new
export WANDB_API_KEY=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

python -m angioflow.training.train_pointnet2_t0 \
  --run_name pointnet2_t0_aneurysmonly \
  --versions aneug_flow,angioflow \
  --crop_aneurysm_only \
  --pointwise_loss huber \
  --lambda_huber 5.0 --lambda_smooth 0.1 --lambda_logmag 0.1 --lambda_dir 0.0\
  --total_epochs 500 \
  --wandb

test AneuG-Flow only

python -m angioflow.training.train_pointnet2_t0 \
  --run_name pointnet2_t0_v1only \
  --versions aneug_flow \
  --crop_aneurysm_only \
  --pointwise_loss huber \
  --lambda_huber 5.0 --lambda_smooth 0.1 --lambda_logmag 0.1 --lambda_dir 0.0 \
  --total_epochs 500 \
  --wandb

  
  
  python -m angioflow.training.train_pointnet2_t0 \
  --run_name pointnet2_t0_v2only \
  --versions angioflow \
  --crop_aneurysm_only \
  --pointwise_loss huber \
  --lambda_huber 5.0 --lambda_smooth 0.1 --lambda_logmag 0.1 --lambda_dir 0.0 \
  --total_epochs 500 \
  --wandb


"""
