"""General training script — model architecture is a CLI choice, not a separate file.

Data pipeline, transforms, loss, optimizer, logging, and checkpointing are shared
across every registered model (see `angioflow/config/models/`). Add a new
architecture by dropping a `<name>.py` config module there (with `add_args` and
`build_model`) and registering it in `angioflow/config/models/__init__.py` —
no changes needed here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
import time
os.environ.setdefault(
    "WANDB_API_KEY",
    "delete",
)
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from angioflow.config.defaults import (
    add_data_args, add_encoder_args, add_loss_args, add_optim_args, add_misc_args,
    add_schedule_args,
)
from angioflow.config.models import get_model_config, MODEL_REGISTRY
from angioflow.data.datasets import AngioFlowWallDataset, load_excluded_keys
from angioflow.data.datasets_t0 import AngioFlowWallDatasetT0
from angioflow.data.normalize import load_or_compute_stats
from angioflow.losses import log_mag_loss, tv_smooth_loss, directional_loss
from angioflow.transform import AneurysmCrop, CenterAndRandomRotate, PosYNormalize, SymmetrizeEdges
from angioflow.visualize.sanity import save_loss_curves, save_wss_sanity


def _chain(*transforms):
    def _apply(data):
        for t in transforms:
            data = t(data)
        return data
    return _apply


def split_by_case(dataset, train_frac=0.8, seed=0):
    # Stratified by version: split each version 80/20 independently so the
    # test set always contains cases from every version present in the dataset.
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


def _unpack_pred(out):
    """Most models return a plain [N, 3] tensor, aligned 1:1 with the input batch's
    nodes. Query-style models (e.g. wss_query_field) may instead predict only a random
    SUBSET of nodes for GPU efficiency, and return (pred, query_idx) — query_idx is
    None whenever that model ran at full resolution (e.g. eval/inference)."""
    return out if isinstance(out, tuple) else (out, None)


def _slice_for_query(batch, query_idx):
    """y/normals sliced to whatever `pred` actually covers; edge_index is None when
    subsampling is active — a random query subset has no meaningful local-adjacency
    structure for tv_smooth_loss to regularise (that's the geometry encoder's / the
    model's own LocalMeshGather's job, both of which still see the FULL mesh
    internally — this is purely about the LOSS side). query_idx=None (every existing
    model, plus query-style models outside training) means no slicing at all."""
    if query_idx is None:
        return batch.y, batch.normals, batch.edge_index
    return batch.y[query_idx], batch.normals[query_idx], None


@torch.no_grad()
def evaluate(model, loader, waveform, device, y_mean, y_std, log_eps, max_wss, huber_delta,
             pointwise_kind="huber", max_batches=None):
    model.eval()
    tot_logmag, tot_huber, tot_mse, n = 0.0, 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = batch.to(device)
        pred, query_idx = _unpack_pred(model(batch, waveform))
        y, _, _ = _slice_for_query(batch, query_idx)
        tot_logmag += log_mag_loss(pred, y, y_mean, y_std, log_eps, max_wss).item() * batch.num_graphs
        tot_huber  += _pointwise_loss(pred, y, pointwise_kind, huber_delta).item() * batch.num_graphs
        tot_mse    += ((pred - y) * y_std).pow(2).mean().item() * batch.num_graphs
        n += batch.num_graphs
    model.train()
    return tot_logmag / max(n, 1), tot_huber / max(n, 1), tot_mse / max(n, 1)


def save_ckpt(path, model, optim, sched, epoch, best_loss, args=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        # full training config: lets eval scripts rebuild the model without
        # having to re-specify hyperparameters by hand
        "args": vars(args) if args is not None else None,
    }, path)


def build_scheduler(optim, args):
    if args.lr_schedule == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=args.lr_plateau_factor, patience=args.lr_plateau_patience,
            threshold=1e-4, min_lr=args.lr * args.lr_end_factor)
    if args.lr_schedule == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1.0, end_factor=args.lr_end_factor, total_iters=args.total_epochs)
    if args.lr_schedule == "step":
        return torch.optim.lr_scheduler.StepLR(
            optim, step_size=max(1, round(0.2 * args.total_epochs)), gamma=0.5)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.total_epochs, eta_min=args.lr * args.lr_end_factor)


def apply_lr_warmup(optim, args, epoch):
    """Linear LR warmup 0->args.lr over the first args.lr_warmup_epochs epochs. Returns
    True while warmup is active, so the caller skips stepping `sched` this epoch (stepping
    both warmup and the schedule together would double-apply a change)."""
    if epoch >= args.lr_warmup_epochs:
        return False
    for pg in optim.param_groups:
        pg["lr"] = args.lr * (epoch + 1) / args.lr_warmup_epochs
    return True


def step_scheduler(sched, args, te_loss):
    if args.lr_schedule == "reduce_on_plateau":
        sched.step(te_loss)
    else:
        sched.step()


def save_run_config(ckpt_dir, args):
    """Human-readable copy of the run's CLI args, written once at launch (before any
    checkpoint exists). save_ckpt() also embeds args in every .pth, but that requires
    torch.load-ing a multi-GB checkpoint just to inspect a run's config -- this is the
    same information as a plain, greppable/diffable JSON file next to the run."""
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)


def save_script_copy(ckpt_dir):
    """Copy this training script into the checkpoint dir so the exact code that
    produced a run's checkpoints stays alongside them (config lives in per-model
    files, but this pins down the training loop / dispatch logic itself)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    dst = os.path.join(ckpt_dir, os.path.basename(__file__))
    shutil.copy(__file__, dst)


HISTORY_FIELDS = ["epoch", "train_logmag", "train_huber", "train_smooth", "train_dir",
                  "test_logmag", "test_huber", "test_mse_denorm", "test_loss", "lr",
                  "grad_norm_mean", "grad_norm_max", "grad_clip_frac", "epoch_time_s"]


def init_history(path, epoch_start):
    """<ckpt_dir>/history.csv, one row per epoch. On resume, keep only rows before
    epoch_start so a restarted run never carries duplicate or stale epochs."""
    rows = []
    if epoch_start > 0 and os.path.exists(path):
        with open(path) as f:
            rows = [r for r in csv.DictReader(f) if int(r["epoch"]) < epoch_start]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        w.writeheader()
        w.writerows(rows)


def append_history(path, row):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writerow(row)


def maybe_resume(ckpt_dir, model, optim, sched):
    """Try last.pth first; fall back to the latest epoch_XXXX.pth if missing."""
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


def get_args():
    """Two-stage parse: first pull `--model` to know which config module's args
    to add, then build the full parser and do the real parse.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=list(MODEL_REGISTRY), required=True)
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre])
    add_data_args(p)
    add_encoder_args(p)
    add_loss_args(p)
    add_optim_args(p)
    add_schedule_args(p)
    add_misc_args(p)
    get_model_config(known.model).add_args(p)
    p.add_argument("--sanity_every", type=int, default=50,
                   help="render WSS sanity snapshots every N epochs (epoch 0 included)")
    p.add_argument("--ckpt_every", type=int, default=100,
                   help="save an extra epoch_XXXX.pth every N epochs (last.pth / best.pth are "
                        "written every epoch / on improvement regardless)")
    p.add_argument("--max_steps", type=int, default=None,
                   help="smoke test: stop after this many training steps (and evaluate on as many "
                        "test batches), print per-step loss/time/GPU memory, render one sanity "
                        "snapshot, save no checkpoints")
    return p.parse_args()


def get_model(args, stats, device):
    return get_model_config(args.model).build_model(args, stats, device)


def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    exclude_keys = load_excluded_keys(args.save_dir) if args.exclude_anomalies else set()
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
        extracted=args.extracted,
    )
    if args.frames == "all":
        # note: the full dataset has no `versions` filter — all versions are used
        dataset = AngioFlowWallDataset(**ds_kwargs)
    else:
        frames = [int(t) for t in args.frames.split(",")]
        dataset = AngioFlowWallDatasetT0(**ds_kwargs, versions=versions, frames=frames)
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
    y_mean = stats['y_mean'].to(device)
    y_std  = stats['y_std'].to(device)
    # delta=1.0 in log space: errors > e^1 ≈ 2.7× multiplicative factor get linear gradient
    huber_delta = 1.0
    print(f"[train] model={args.model}  log-mag loss eps={args.log_eps:.2f} max={args.max_wss:.1f} Pa/s  "
          f"pointwise={args.pointwise_loss}")

    transforms = [
        CenterAndRandomRotate(rotate=True),
        PosYNormalize(stats),
        AneurysmCrop(low=args.mask_low, high=args.mask_high,
                     aneurysm_only=args.crop_aneurysm_only),
        SymmetrizeEdges(),   # after cropping: symmetrizes the smaller, already-cropped edge set
    ]
    dataset.transform = _chain(*transforms)

    train_loader = DataLoader(Subset(dataset, train_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(Subset(dataset, test_idx),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    waveform = load_waveform(args.waveform_path, length=1000, device=device)

    model = get_model(args, stats, device)
    print(f"[model] {args.model}  {sum(p.numel() for p in model.parameters()):,} parameters")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = build_scheduler(optim, args)

    ckpt_dir  = os.path.join(args.ckpt_root, args.run_name)
    save_script_copy(ckpt_dir)
    save_run_config(ckpt_dir, args)
    ckpt_last = os.path.join(ckpt_dir, "last.pth")
    ckpt_best = os.path.join(ckpt_dir, "best.pth")
    epoch_start, best_loss = 0, float("inf")
    if args.resume:
        epoch_start, best_loss = maybe_resume(ckpt_dir, model, optim, sched)
    history_path = os.path.join(ckpt_dir, "history.csv")
    init_history(history_path, epoch_start)

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=args.run_name, config=vars(args),
                   resume="allow", id=args.wandb_run_id)

    for epoch in range(epoch_start, args.total_epochs):
        t_epoch = time.perf_counter()
        model.train()
        warming_up = apply_lr_warmup(optim, args, epoch)
        if hasattr(model, "redraw_projection"):
            model.redraw_projection.redraw_projections()
        tot_logmag, tot_huber, tot_smooth, tot_dir, n = 0.0, 0.0, 0.0, 0.0, 0
        gnorm_sum, gnorm_max, n_clipped, n_steps = 0.0, 0.0, 0, 0
        for batch in train_loader:
            t_step = time.perf_counter()
            batch = batch.to(device)
            if torch.isnan(batch.pos).any() or torch.isnan(batch.y).any():
                print("[warn] NaN in input/target — skipping batch")
                continue
            optim.zero_grad()
            pred, query_idx = _unpack_pred(model(batch, waveform))
            y, normals, edge_index = _slice_for_query(batch, query_idx)
            loss_logmag = log_mag_loss(pred, y, y_mean, y_std, args.log_eps, args.max_wss)
            loss_huber  = _pointwise_loss(pred, y, args.pointwise_loss, huber_delta)
            loss_smooth = tv_smooth_loss(pred, edge_index) if edge_index is not None else pred.new_zeros(())
            loss_dir = directional_loss(pred, normals, y_mean, y_std)
            loss = args.lambda_logmag * loss_logmag + args.lambda_huber * loss_huber \
                 + args.lambda_smooth * loss_smooth + args.lambda_dir * loss_dir
            if torch.isnan(loss):
                print(f"[warn] NaN loss — pred nan={torch.isnan(pred).sum()}, "
                      f"y nan={torch.isnan(y).sum()}, skipping")
                optim.zero_grad()
                continue
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            gnorm_sum += gnorm.item()
            gnorm_max = max(gnorm_max, gnorm.item())
            n_clipped += gnorm.item() > args.grad_clip_norm
            n_steps += 1
            optim.step()
            if args.max_steps:
                torch.cuda.synchronize(device)
                print(f"[step {n_steps:3d}] loss {loss.item():.4g}  {time.perf_counter() - t_step:.2f}s  "
                      f"nodes {batch.num_nodes}  peak GPU {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB",
                      flush=True)
            tot_logmag += loss_logmag.item() * batch.num_graphs
            tot_huber  += loss_huber.item()  * batch.num_graphs
            tot_smooth += loss_smooth.item() * batch.num_graphs
            tot_dir    += loss_dir.item()    * batch.num_graphs
            n += batch.num_graphs
            if args.max_steps and n_steps >= args.max_steps:
                break
        train_logmag = tot_logmag / max(n, 1)
        train_huber  = tot_huber  / max(n, 1)
        train_smooth = tot_smooth / max(n, 1)
        train_dir    = tot_dir    / max(n, 1)
        gnorm_mean = gnorm_sum / max(n_steps, 1)
        clip_frac  = n_clipped / max(n_steps, 1)

        test_logmag, test_huber, test_mse_denorm = evaluate(
            model, test_loader, waveform, device,
            y_mean, y_std, args.log_eps, args.max_wss, huber_delta,
            pointwise_kind=args.pointwise_loss, max_batches=args.max_steps)
        # weighted the same as the training loss's logmag/huber terms, so the scheduler
        # and best-checkpoint selection track what training actually optimizes, not an
        # arbitrary 1:1 sum (huber is weighted 50x higher than logmag by default)
        test_loss = args.lambda_logmag * test_logmag + args.lambda_huber * test_huber
        if not warming_up:
            step_scheduler(sched, args, test_loss)
        lr_now = optim.param_groups[0]["lr"]
        print(f"epoch {epoch:3d}  "
              f"train_logmag {train_logmag:.4g}  train_huber {train_huber:.4g}  train_smooth {train_smooth:.4g}  "
              f"train_dir {train_dir:.4g} (Pa/s)  "
              f"test_logmag {test_logmag:.4g}  test_huber {test_huber:.4g}  "
              f"test_mse_denorm {test_mse_denorm:.4g} (Pa/s)²  lr {lr_now:.2e}  "
              f"gnorm {gnorm_mean:.3g} (max {gnorm_max:.3g}, clipped {clip_frac:.0%})",
              flush=True)
        append_history(history_path, dict(
            epoch=epoch, train_logmag=train_logmag, train_huber=train_huber, train_smooth=train_smooth,
            train_dir=train_dir, test_logmag=test_logmag, test_huber=test_huber,
            test_mse_denorm=test_mse_denorm, test_loss=test_loss, lr=lr_now,
            grad_norm_mean=gnorm_mean, grad_norm_max=gnorm_max, grad_clip_frac=clip_frac,
            epoch_time_s=round(time.perf_counter() - t_epoch, 1)))
        save_loss_curves(history_path, os.path.join(ckpt_dir, "sanity", "loss_curves.png"))
        if args.max_steps:      # also exercise the sanity renderer, but save no checkpoints
            sanity_dir = os.path.join(ckpt_dir, "sanity")
            save_wss_sanity(model, test_loader, waveform, device, stats, sanity_dir, epoch,
                            max_wss=args.max_wss)
            print(f"[smoke] {n_steps} step(s) + {args.max_steps} test batch(es) done; "
                  f"sanity -> {sanity_dir}; no checkpoints saved")
            return

        if args.wandb:
            wandb.log({"train_logmag": train_logmag, "train_huber": train_huber,
                       "train_smooth": train_smooth, "train_dir": train_dir,
                       "test_logmag": test_logmag, "test_huber": test_huber,
                       "test_mse_denorm": test_mse_denorm, "lr": lr_now,
                       "grad_norm_mean": gnorm_mean, "grad_norm_max": gnorm_max,
                       "grad_clip_frac": clip_frac}, step=epoch)

        if epoch % args.sanity_every == 0:
            sanity_dir = os.path.join(ckpt_dir, "sanity")
            save_wss_sanity(model, test_loader, waveform, device, stats, sanity_dir, epoch,
                            max_wss=args.max_wss)

        save_ckpt(ckpt_last, model, optim, sched, epoch, best_loss, args)
        if epoch % args.ckpt_every == 0:
            save_ckpt(os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"),
                      model, optim, sched, epoch, best_loss, args)
        if test_loss < best_loss:
            best_loss = test_loss
            save_ckpt(ckpt_best, model, optim, sched, epoch, best_loss, args)
            print(f"  ↳ new best @ epoch {epoch} (test_loss={best_loss:.4g})")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main(get_args())


"""
conda activate new

python -m angioflow.training.train --model hierarchical_gps \
  --extracted --save_dir /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG \
  --ckpt_root /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints \
  --run_name hier_gps_p1_e128_npre2_gps8_npost4_knn8_inter2_pr015_aneugflow_angioflow \
  --versions aneug_flow,angioflow --frames 0 --train_frac 0.9 --decimate_ratio 1.0 \
  --attn_type performer --dim_emb 128 --dim_pe 16 --dim_te 8 --dim_wfe 8 \
  --n_pre 2 --n_gps 8 --n_post 4 --n_pool 1 --pool_ratio 0.15 --pool_knn_k 8 --n_inter 2 --n_heads 4 \
  --pe_use_mlp --pe_n_heads 2 --pe_n_layers 2 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_smooth 0.0 --lambda_logmag 0.1 --lambda_dir 0.1 \
  --max_wss 35 --log_eps 0.1 --grad_clip_norm 3.0 \
  --lr 1e-3 --lr_schedule reduce_on_plateau --lr_plateau_factor 0.5 --lr_plateau_patience 15 --lr_warmup_epochs 5 --lr_end_factor 0.01 \
  --mask_low 0.0 --mask_high 0.25 --grad_ckpt --seed 0 \
  --batch_size 24 --total_epochs 500 --sanity_every 25 --device cuda:3 --wandb



"""
