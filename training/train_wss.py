"""Train AngioGraphGPS on AngioFlowWallDataset to predict per-node WSS.

Single-GPU MVP. DDP wiring is intentionally absent; add it later by wrapping
the train loop in `mp.spawn` and the model in `DDP` once the single-GPU run
is verified.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from angioflow.data.datasets import AngioFlowWallDataset, load_excluded_keys
from angioflow.data.normalize import load_or_compute_stats
from angioflow.transform import PosYNormalize, AneurysmLossMask, CenterAndRandomRotate


def _chain(*transforms):
    def _apply(data):
        for t in transforms:
            data = t(data)
        return data
    return _apply
from angioflow.models.transformer import AngioGraphGPS


def split_by_case(dataset, train_frac=0.8, seed=0):
    """Per-case holdout. Returns (train_idx, test_idx) into dataset._index."""
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
    """Load waveform .txt, keep first 1/5, interp to `length`. -> [1, length, 1]."""
    raw = torch.from_numpy(np.loadtxt(path, delimiter=",")).float()
    raw = raw[: len(raw) // 5]
    wf = F.interpolate(raw.view(1, 1, -1), size=(length,),
                       mode="linear", align_corners=True).permute(0, 2, 1)
    return wf.to(device)


@torch.no_grad()
def evaluate(model, loader, waveform, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, waveform)
        m = batch.loss_mask                                # respect mask at eval too (only aneurysm if mask_high=0)
        loss = F.mse_loss(pred[m], batch.y[m])
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
    dataset = AngioFlowWallDataset(
        case_root_angioflow=args.case_root_angioflow,
        case_root_aneug_flow=args.case_root_aneug_flow,
        save_dir=args.save_dir,
        n_frames=args.n_frames,
        decimate_ratio=args.decimate_ratio,
        decimate_method=args.decimate_method,
        lap_k=args.lap_k,
        exclude_keys=exclude_keys,
    )
    train_idx, test_idx = split_by_case(dataset, args.train_frac, seed=args.seed)
    print(f"[data] {len(dataset)} samples; train={len(train_idx)}, test={len(test_idx)}")

    # normalization stats — computed on train cases only, cached to <save_dir>/stats.pt
    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}
    stats = load_or_compute_stats(
        os.path.join(args.save_dir, "stats.pt"),
        dataset, train_keys,
        n_cases=args.stats_n_cases, frames_per_case=args.stats_frames_per_case,
        seed=args.seed,
    )
    dataset.transform = _chain(
        CenterAndRandomRotate(rotate=True),                    # center per graph; random SO(3)
        PosYNormalize(stats),                                  # isotropic z-score (rotation-invariant)
        AneurysmLossMask(low=args.mask_low, high=args.mask_high),
    )

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    waveform = load_waveform(args.waveform_path, length=1000, device=device)

    model = AngioGraphGPS(
        n_cot=args.lap_k, node_dim_in=6, n_node_types=args.n_node_types,
        dim_pe=args.dim_pe, dim_te=args.dim_te, dim_wfe=args.dim_wfe,
        dim_emb=args.dim_emb, n_heads=args.n_heads,
        num_layers=args.n_gps_layers, dim_out=3,
        attn_type=args.attn_type, attn_kwargs=args.attn_kwargs,
        pe_n_heads=args.pe_n_heads, pe_n_layers=args.pe_n_layers,
        edge_norm_dict={"mean": stats["edge_mean"], "std": stats["edge_std"]},
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LinearLR(
        optim, start_factor=1.0, end_factor=args.lr_end_factor, total_iters=args.total_epochs)

    ckpt_dir = os.path.join("checkpoints", args.run_name)
    ckpt_last = os.path.join(ckpt_dir, "last.pth")
    ckpt_best = os.path.join(ckpt_dir, "best.pth")
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
            optim.zero_grad()
            pred = model(batch, waveform)
            m = batch.loss_mask
            loss = F.mse_loss(pred[m], batch.y[m])
            loss.backward()
            optim.step()
            total += loss.item() * batch.num_graphs
            n += batch.num_graphs
        train_loss = total / max(n, 1)
        sched.step()

        test_loss = evaluate(model, test_loader, waveform, device)
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
    p.add_argument("--save_dir",     default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_v1")
    p.add_argument("--n_frames", type=int, default=80)
    p.add_argument("--decimate_ratio", type=float, default=0.5)
    p.add_argument("--decimate_method", choices=["quadric", "fps"], default="quadric")
    p.add_argument("--lap_k", type=int, default=3)
    p.add_argument("--waveform_path", default="/home/wenhao/datasets/AneuG_processed/waveform_yiying.txt")
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--exclude_anomalies", action="store_true",
                   help="drop cases listed in <save_dir>/anomalies.json")
    p.add_argument("--mask_low", type=float, default=0.0,
                   help="lower bound for per-graph parent-vessel keep ratio (loss mask)")
    p.add_argument("--mask_high", type=float, default=0.25,
                   help="upper bound; aneurysm (geo_dist==0) is always kept. Set both to 1.0 for full graph.")
    p.add_argument("--stats_n_cases", type=int, default=100)
    p.add_argument("--stats_frames_per_case", type=int, default=8)
    # model
    p.add_argument("--n_node_types", type=int, default=5,
                   help="0=wall, 1=focused/aneurysm, 2=inlet, 3=outlet_0, 4=outlet_1")
    p.add_argument("--dim_pe", type=int, default=16)
    p.add_argument("--dim_te", type=int, default=8)
    p.add_argument("--dim_wfe", type=int, default=8)
    p.add_argument("--dim_emb", type=int, default=32)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_gps_layers", type=int, default=2)
    p.add_argument("--pe_n_heads", type=int, default=2)
    p.add_argument("--pe_n_layers", type=int, default=2)
    p.add_argument("--attn_type", default="performer")
    p.add_argument("--attn_kwargs", default=None)
    # optim
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr_end_factor", type=float, default=0.05,
                   help="final lr as a fraction of initial lr; linearly interpolated over total_epochs")
    p.add_argument("--total_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=4)
    # misc
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", default="run0")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_entity", default=None,
                   help="wandb username or team — runs go to YOUR workspace on shared accounts. "
                        "Defaults to whoever ran `wandb login` last (risky on shared OS accounts).")
    p.add_argument("--wandb_project", default="AngioFlow-WSS")

    args = p.parse_args()
    main(args)


"""
conda activate new
export WANDB_API_KEY=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
python -m angioflow.training.train_wss --run_name dbg0
python -m angioflow.training.train_wss --run_name v0 --wandb


"""
