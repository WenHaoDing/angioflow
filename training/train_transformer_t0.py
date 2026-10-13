"""Train AngioGraphGPS on AngioFlowWallDatasetT0 to predict per-node WSS at t=0.

Single-GPU. Same data pipeline as train_pointnet2_t0.py; model and args differ.
"""
from __future__ import annotations

import argparse
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
from angioflow.data.normalize import load_or_compute_stats
from angioflow.losses import log_mag_huber_loss
from angioflow.models.transformer import AngioGraphGPS
from angioflow.transform import AneurysmCrop, CenterAndRandomRotate, PosYNormalize


def _chain(*transforms):
    def _apply(data):
        for t in transforms:
            data = t(data)
        return data
    return _apply


def split_by_case(dataset, train_frac=0.8, seed=0):
    keys = sorted({(v, c) for (v, c, _) in dataset._index})
    rng = random.Random(seed)
    rng.shuffle(keys)
    n_train = int(len(keys) * train_frac)
    train_keys = set(keys[:n_train])
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


@torch.no_grad()
def evaluate(model, loader, waveform, device, huber_delta):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, waveform)
        loss = F.huber_loss(pred, batch.y, delta=huber_delta)
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    model.train()
    return total / max(n, 1)


def save_ckpt(path, model, optim, sched, epoch, best_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
    }, path)


def maybe_resume(path, model, optim, sched):
    if not os.path.exists(path):
        return 0, float("inf")
    chk = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(chk["model"])
    optim.load_state_dict(chk["optim"])
    sched.load_state_dict(chk["sched"])
    print(f"[resume] {path} @ epoch {chk['epoch']+1}, best={chk['best_loss']:.4g}")
    return chk["epoch"] + 1, chk["best_loss"]


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
    print(f"[stats] pos_std={stats['pos_std'].mean().item():.4f}")
    # y is z-scored by PosYNormalize, so normalized y ~ N(0,1); 1.0 sigma is the natural Huber threshold
    huber_delta = 1.0
    print(f"[train] huber_delta={huber_delta:.4f} (y is z-scored to unit variance)")
    dataset.transform = _chain(
        CenterAndRandomRotate(rotate=True),
        PosYNormalize(stats),
        AneurysmCrop(low=args.mask_low, high=args.mask_high),
    )

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    waveform = load_waveform(args.waveform_path, length=1000, device=device)

    model = AngioGraphGPS(
        n_cot=args.lap_k, node_dim_in=6, n_geo_dist=args.n_geo_dist,
        dim_pe=args.dim_pe, dim_te=args.dim_te, dim_wfe=args.dim_wfe,
        dim_emb=args.dim_emb, n_heads=args.n_heads,
        num_layers=args.n_gps_layers, dim_out=3,
        attn_type=args.attn_type, attn_kwargs=args.attn_kwargs,
        pe_n_heads=args.pe_n_heads, pe_n_layers=args.pe_n_layers, pe_use_mlp=args.pe_use_mlp,
        edge_norm_dict={"mean": stats["edge_mean"], "std": stats["edge_std"]},
    ).to(device)
    print(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1.0, end_factor=args.lr_end_factor, total_iters=args.total_epochs)

    ckpt_dir  = os.path.join("checkpoints", args.run_name)
    ckpt_last = os.path.join(ckpt_dir, "last.pth")
    ckpt_best = os.path.join(ckpt_dir, "best.pth")
    epoch_start, best_loss = 0, float("inf")
    if args.resume:
        epoch_start, best_loss = maybe_resume(ckpt_last, model, optim, sched)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args), resume="allow")

    for epoch in range(epoch_start, args.total_epochs):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            if torch.isnan(batch.pos).any() or torch.isnan(batch.y).any():
                print("[warn] NaN in input/target — skipping batch")
                continue
            optim.zero_grad()
            pred = model(batch, waveform)
            loss = F.huber_loss(pred, batch.y, delta=huber_delta)
            if torch.isnan(loss):
                print(f"[warn] NaN loss — pred nan={torch.isnan(pred).sum()}, skipping")
                optim.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        train_loss = total / max(n, 1)
        sched.step()

        test_loss = evaluate(model, test_loader, waveform, device, huber_delta)
        lr_now = optim.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  train {train_loss:.4g}  test {test_loss:.4g}  lr {lr_now:.2e}")

        if args.wandb:
            wandb.log({"train_mse": train_loss, "test_mse": test_loss, "lr": lr_now}, step=epoch)

        save_ckpt(ckpt_last, model, optim, sched, epoch, best_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            save_ckpt(ckpt_best, model, optim, sched, epoch, best_loss)
            print(f"  ↳ new best @ epoch {epoch} (test_mse={best_loss:.4g})")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--case_root_angioflow", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged")
    p.add_argument("--case_root_aneug_flow", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data")
    p.add_argument("--save_dir",     default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_v2")
    p.add_argument("--n_frames",     type=int,   default=80)
    p.add_argument("--decimate_ratio", type=float, default=0.5)
    p.add_argument("--decimate_method", choices=["quadric", "fps"], default="quadric")
    p.add_argument("--lap_k",        type=int,   default=8)
    p.add_argument("--waveform_path", default="/home/wenhao/datasets/AneuG_processed/waveform_yiying.txt")
    p.add_argument("--train_frac",   type=float, default=0.8)
    p.add_argument("--exclude_anomalies", action="store_true")
    p.add_argument("--versions",     default=None, help="comma-separated versions, e.g. 'aneug_flow' or 'aneug_flow,angioflow'")
    p.add_argument("--mask_low",     type=float, default=0.0)
    p.add_argument("--mask_high",    type=float, default=0.1)
    p.add_argument("--stats_n_cases", type=int,  default=100)
    p.add_argument("--stats_frames_per_case", type=int, default=8)
    # model
    p.add_argument("--n_geo_dist",   type=int,   default=3)
    p.add_argument("--dim_pe",       type=int,   default=8)
    p.add_argument("--dim_te",       type=int,   default=8)
    p.add_argument("--dim_wfe",      type=int,   default=8)
    p.add_argument("--dim_emb",      type=int,   default=8)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--n_gps_layers", type=int,   default=2)
    p.add_argument("--pe_n_heads",   type=int,   default=2)
    p.add_argument("--pe_n_layers",  type=int,   default=2)
    p.add_argument("--pe_use_mlp",   action="store_true", help="use MLP instead of transformer in CotPE")
    p.add_argument("--attn_type",    default="performer")
    p.add_argument("--attn_kwargs",  default=None)
    # optim
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--lr_end_factor", type=float, default=0.05)
    p.add_argument("--total_epochs", type=int,   default=500)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--num_workers",  type=int,   default=4)
    # misc
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--run_name",     default="transformer_t0")
    p.add_argument("--resume",       action="store_true", default=False)
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_project", default="AngioFlow")

    args = p.parse_args()
    main(args)


"""
conda activate new
python -m angioflow.training.train_transformer_t0 --run_name transformer_t0_v1 --versions aneug_flow --wandb
"""
