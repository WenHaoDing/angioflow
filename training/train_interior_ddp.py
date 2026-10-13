"""Distributed (DDP) launcher for the interior field baselines.

train_interior.py is untouched — this script imports its model/loss/dataset
building blocks (build_dataset, build_model, field_loss, evaluate,
save_interior_sanity, get_args: same CLI flags, same defaults) and adds only the
distributed-specific wiring: process-group setup, DistributedSampler, DDP wrapping,
rank-0 guards on logging/checkpointing/sanity renders, and a few explicit
synchronization points that single-process training didn't need:

  * NaN-skip is all-reduced (MAX) across ranks before any rank decides to skip a
    step — one process silently `continue`-ing while others call `.backward()`
    would deadlock the collective all-reduce.
  * epoch-level loss accumulators (tot/tv/tp/td/n) are summed across ranks before
    printing/logging, so numbers reflect the whole epoch, not one shard.
  * lambda_div is a plain constant (args.lambda_div, identical CLI input on every
    rank), so — unlike an adaptively-tuned weight would need — it needs no
    cross-rank synchronization at all.
  * the HierGPS performer backbone's `redraw_projections()` draws from the default
    RNG; since every rank calls `torch.manual_seed(args.seed)` with the same seed
    and nothing rank-specific consumes that generator beforehand, replicas draw
    identical projections without an explicit broadcast. If a future change adds
    rank-dependent randomness ahead of it, that assumption breaks silently — an
    explicit broadcast of the projection buffers would be the fix.

Launch (same flags as train_interior.py, just a different entry point):
  torchrun --nproc_per_node=<N> -m angioflow.training.train_interior_ddp \\
    --backbone hgps --save_dir ... --interior_dir ... --run_name ... \\
    --sample_ratio 0.25 --max_points 30000 --dim_emb 48 --dim_field 48 \\
    --batch_size 6 --lambda_div 1.0 --total_epochs 500 ...
"""
from __future__ import annotations

import os
import random
import shutil

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from angioflow.data.datasets import load_excluded_keys
from angioflow.data.normalize_interior import load_or_compute_interior_stats
from angioflow.transform import (CenterAndRandomRotate, InteriorNormalize, InteriorSubsample,
                                 InteriorSubsampleGraph)
from angioflow.training.train import _chain, load_waveform, maybe_resume, save_ckpt, split_by_case
from angioflow.training.train_interior import (
    apply_lr_warmup, build_dataset, build_model, build_scheduler, evaluate, get_args, run_model,
    save_interior_sanity, save_run_config, step_scheduler,
)


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def is_main(rank):
    return rank == 0


def save_script_copy(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    shutil.copy(__file__, os.path.join(ckpt_dir, os.path.basename(__file__)))


def all_reduce_sum(*values, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def any_rank_true(flag, device):
    t = torch.tensor([1 if flag else 0], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())


def main(args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    exclude_keys = load_excluded_keys(args.save_dir) if args.exclude_anomalies else set()
    dataset = build_dataset(args, exclude_keys)
    train_idx, test_idx = split_by_case(dataset, args.train_frac, seed=args.seed)
    if is_main(rank):
        print(f"[data] {len(dataset)} samples  train={len(train_idx)}  test={len(test_idx)}  "
              f"world_size={world_size}")

    train_keys = {(v, c) for (v, c, _) in (dataset._index[i] for i in train_idx)}
    stats_path = os.path.join(args.save_dir, "interior_stats.pt")
    if is_main(rank) and not os.path.exists(stats_path):
        load_or_compute_interior_stats(stats_path, dataset, train_keys, n_cases=args.stats_n_cases,
                                       frames_per_case=args.stats_frames_per_case, seed=args.seed)
    dist.barrier()   # other ranks wait until rank 0 has written interior_stats.pt
    stats = load_or_compute_interior_stats(stats_path, dataset, train_keys, n_cases=args.stats_n_cases,
                                           frames_per_case=args.stats_frames_per_case, seed=args.seed)
    norm = InteriorNormalize(stats)

    div_kwargs = dict(ridge=args.div_ridge, power=args.div_power,
                      min_neighbors=args.div_min_neighbors, cond_tol=args.div_cond_tol)
    subsample = (InteriorSubsampleGraph(sample_ratio=args.sample_ratio, max_points=args.max_points)
                if (args.lambda_div > 0 and args.div_method == "graph") else
                InteriorSubsample(sample_ratio=args.sample_ratio, max_points=args.max_points))
    dataset.transform = _chain(subsample, CenterAndRandomRotate(rotate=True), norm)

    train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                       shuffle=True, seed=args.seed)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=test_sampler,
                             num_workers=args.num_workers, pin_memory=True)

    waveform = load_waveform(args.waveform_path, length=1000, device=device)
    model = build_model(args, device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = build_scheduler(optim, args)

    ckpt_dir = os.path.join(args.ckpt_root, args.run_name)
    ckpt_last = os.path.join(ckpt_dir, "last.pth")
    ckpt_best = os.path.join(ckpt_dir, "best.pth")
    epoch_start, best_loss = 0, float("inf")
    if args.resume:
        # every rank reads the same checkpoint file (read-only, no race); DDP's
        # constructor below broadcasts rank 0's params to all ranks regardless.
        epoch_start, best_loss = maybe_resume(ckpt_dir, model, optim, sched)

    model = DDP(model, device_ids=[local_rank])

    if is_main(rank):
        save_script_copy(ckpt_dir)
        save_run_config(ckpt_dir, args)
        token_desc = ("downsampled_graph" if args.geometry_token_pool == "none" else
                      f"{args.n_geometry_tokens}_latent_tokens")
        print(f"[model] interior_field/{args.backbone}  global={token_desc}  "
              f"{sum(p.numel() for p in model.parameters()):,} parameters")

    if args.wandb and is_main(rank):
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.run_name,
                   config=vars(args), resume="allow", id=args.wandb_run_id)

    for epoch in range(epoch_start, args.total_epochs):
        model.train()
        warming_up = apply_lr_warmup(optim, args, epoch)
        train_sampler.set_epoch(epoch)   # re-shuffle differently every epoch
        test_sampler.set_epoch(epoch)    # DistributedSampler is otherwise deterministic per
                                          # epoch index, so the sanity render below would draw
                                          # the same "fresh" batch every time without this
        if hasattr(model.module, "redraw_projection"):
            model.module.redraw_projection.redraw_projections()
        tot, tv, tp, td, n = 0.0, 0.0, 0.0, 0.0, 0
        for wall, interior in train_loader:
            wall, interior = wall.to(device), interior.to(device)
            has_nan = bool(torch.isnan(interior.vel).any() or torch.isnan(interior.press).any())
            if any_rank_true(has_nan, device):
                if is_main(rank):
                    print("[warn] NaN in interior targets on some rank — skipping batch")
                continue
            optim.zero_grad()
            vel_pred, press_pred, loss, lv, lp, ld = run_model(model, wall, interior, waveform,
                                                                args, div_kwargs)
            if any_rank_true(bool(torch.isnan(loss)), device):
                if is_main(rank):
                    print("[warn] NaN loss on some rank — skipping batch")
                optim.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optim.step()
            b = wall.num_graphs
            tot += loss.item() * b; tv += lv * b; tp += lp * b; td += ld * b; n += b
        tot, tv, tp, td, n = all_reduce_sum(tot, tv, tp, td, n, device=device)
        n = max(n, 1)
        sched.step()

        te_loss, te_v, te_p, te_d, te_mv, te_mp = evaluate(
            model, test_loader, waveform, device, norm, args, div_kwargs)
        te_loss, te_v, te_p, te_d, te_mv, te_mp = all_reduce_sum(
            te_loss, te_v, te_p, te_d, te_mv, te_mp, device=device)
        te_loss, te_v, te_p, te_d, te_mv, te_mp = (
            x / world_size for x in (te_loss, te_v, te_p, te_d, te_mv, te_mp))

        if is_main(rank):
            lr_now = optim.param_groups[0]["lr"]
            div_desc = f"  train_div {td/n:.4g}  test_div {te_d:.4g}" if args.lambda_div > 0 else ""
            print(f"epoch {epoch:3d}  train_loss {tot/n:.4g} (vel {tv/n:.4g} press {tp/n:.4g})  "
                  f"test_loss {te_loss:.4g} (vel {te_v:.4g} press {te_p:.4g}){div_desc}  "
                  f"test_mse_vel {te_mv:.4g} (m/s)²  test_mse_press {te_mp:.4g} (Pa)²  lr {lr_now:.2e}",
                  flush=True)
            if args.wandb:
                import wandb
                wandb.log({"train_loss": tot / n, "train_vel": tv / n, "train_press": tp / n,
                           "train_div": td / n,
                           "test_loss": te_loss, "test_vel": te_v, "test_press": te_p, "test_div": te_d,
                           "test_mse_vel": te_mv, "test_mse_press": te_mp, "lr": lr_now}, step=epoch)

            if args.sanity_every and epoch % args.sanity_every == 0:
                try:
                    sw, si = next(iter(test_loader))
                    save_interior_sanity(model, sw, si, waveform, device, norm,
                                         os.path.join(ckpt_dir, "sanity"), epoch)
                except Exception as e:
                    print(f"[sanity] render failed at epoch {epoch}: {e}", flush=True)

            save_ckpt(ckpt_last, model.module, optim, sched, epoch, best_loss, args)
            if epoch % 100 == 0:
                save_ckpt(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"),
                          model.module, optim, sched, epoch, best_loss, args)
            if te_loss < best_loss:
                best_loss = te_loss
                save_ckpt(ckpt_best, model.module, optim, sched, epoch, best_loss, args)
                print(f"  ↳ new best @ epoch {epoch} (test_loss={best_loss:.4g})")
        # keep every rank's best_loss consistent for a future resume, even though
        # only rank 0 writes it to disk
        best_loss = min(best_loss, te_loss)

    if args.wandb and is_main(rank):
        import wandb
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main(get_args())

"""
conda activate new

CUDA_VISIBLE_DEVICES=1,0 torchrun --nproc_per_node=2 -m angioflow.training.train_interior_ddp \
  --backbone hgps \
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
  --lambda_vel 1.0 --lambda_press 0.0 \
  --lambda_div 0.25 \
  --batch_size 6 --num_workers 4 \
  --total_epochs 500 --sanity_every 25 \
  --wandb --wandb_project AngioFlow_velocity

"""

