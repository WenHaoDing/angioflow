"""LR probe for warm-starting the joint model on ALL frames from a frame-0 checkpoint.

Question: does a warm start at a given peak LR learn the other frames of the cardiac cycle
without wiping out what the frame-0 model knows? For each --lrs value, from the same
--init_from weights with a fresh AdamW (exactly what train_joint.py --init_from does):
  - train --steps steps on every frame of the training cases (same transforms and loss as
    train_joint.py), linear warmup over --warmup_steps, then constant LR;
  - every --eval_every steps (and at step 0 = the frame-0 model zero-shot) evaluate a FIXED,
    seeded set: --eval_cases test cases x each of --eval_frames, reported per frame
    (frame 0 separately: forgetting shows there first).
Writes <out>/probe.csv (lr, step, frame, loss terms), <out>/probe.json and <out>/probe.png.

All other flags are train_joint.py's (the run's own config) with --frames all, e.g.:
    python -m angioflow.training.lr_probe_joint --init_from <run>/best.pth --lrs 3e-5,1e-4,3e-4 \\
        --steps 1000 --out <dir> -- <train_joint.py flags of the run> --frames all --device cuda:3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init_from", required=True)
    p.add_argument("--lrs", default="3e-5,1e-4,3e-4")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--eval_frames", default="0,10,20,30,40,50,60,70")
    p.add_argument("--eval_cases", type=int, default=40, help="test cases evaluated (evenly spread)")
    p.add_argument("--out", required=True)
    own, rest = p.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    import angioflow.training.train_joint as TJ
    sys.argv = ["train_joint"] + rest
    return own, TJ.get_args(), TJ


def main():
    own, args, TJ = parse()
    from angioflow.data.normalize import load_or_compute_stats
    from angioflow.data.normalize_interior import load_or_compute_interior_stats
    from angioflow.training.train import _chain, load_waveform, split_by_case
    from angioflow.transform import (CenterAndRandomRotate, InteriorSubsample, JointAneurysmCrop,
                                     JointNormalize, PressureZeroMax, SymmetrizeEdges)
    assert args.frames == "all", "the probe trains on every frame: pass --frames all"
    os.makedirs(own.out, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ds = TJ.build_dataset(args, TJ.excluded_case_keys(args))
    train_idx, test_idx = split_by_case(ds, args.train_frac, seed=args.seed, test_versions=TJ.test_versions(args))
    train_keys = {(v, c) for (v, c, _) in (ds._index[i] for i in train_idx)}
    wss_stats = load_or_compute_stats(os.path.join(args.save_dir, "stats.pt"), ds, train_keys,
                                      n_cases=args.stats_n_cases, frames_per_case=args.stats_frames_per_case,
                                      seed=args.seed)
    int_stats = load_or_compute_interior_stats(TJ.interior_stats_path(args), ds, train_keys,
                                               n_cases=args.stats_n_cases,
                                               frames_per_case=args.stats_frames_per_case,
                                               seed=args.seed, press_zero_max=args.press_zero_max)
    norm = JointNormalize(wss_stats, int_stats)
    y_mean, y_std = wss_stats["y_mean"].to(device), wss_stats["y_std"].to(device)
    tf = ([PressureZeroMax()] if args.press_zero_max else []) + [
        JointAneurysmCrop(low=args.mask_low, high=args.mask_high, aneurysm_only=args.crop_aneurysm_only,
                          factor=args.crop_factor, slack=args.crop_slack),
        InteriorSubsample(sample_ratio=args.interior_sample_ratio, max_points=args.interior_max_points),
        CenterAndRandomRotate(rotate=True), norm, TJ._WallOnly(SymmetrizeEdges())]
    ds.transform = _chain(*tf)

    # fixed eval set: evenly spread test cases, at each eval frame
    frames = [int(t) for t in own.eval_frames.split(",")]
    test_cases = sorted({ds._index[i][:2] for i in test_idx})
    pick = set(test_cases[:: max(1, len(test_cases) // own.eval_cases)][:own.eval_cases])
    eval_sets = {t: [i for i in test_idx if ds._index[i][:2] in pick and ds._index[i][2] == t] for t in frames}
    n_frames = {ds._index[i][2] for i in train_idx}
    print(f"[probe] train: {len(train_idx)} samples ({len(train_keys)} cases x up to {len(n_frames)} frames); "
          f"eval: {len(pick)} test cases x frames {frames}", flush=True)
    waveform = load_waveform(args.waveform_path, length=1000, device=device)
    init = torch.load(own.init_from, map_location="cpu", weights_only=False)
    print(f"[probe] init {own.init_from} (epoch {init.get('epoch')}, best {init.get('best_loss')})", flush=True)

    def evaluate(model):
        out = {}
        for t, idx in eval_sets.items():
            torch.manual_seed(1234 + t)                 # same crops / subsamples / rotations every time
            ld = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            with torch.no_grad():
                out[t] = TJ.evaluate(model, ld, waveform, device, args, y_mean, y_std, 1.0, norm)
        return out

    rows = []
    path_csv = os.path.join(own.out, "probe.csv")
    keys = ["loss", "logmag", "huber", "vel", "press", "mse_wss", "mse_vel"]

    def log(lr, step, ev, train_loss, secs):
        for t, e in ev.items():
            rows.append(dict(lr=lr, step=step, frame=t, train_loss=train_loss, secs=round(secs, 1),
                             **{k: e.get(k, float("nan")) for k in keys}))
        with open(path_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        f0 = ev[frames[0]]["loss"]
        oth = np.mean([ev[t]["loss"] for t in frames if t != frames[0]])
        print(f"  lr {lr:.0e} step {step:5d}  test loss frame {frames[0]} {f0:.4f}  other frames {oth:.4f}  "
              f"(vel f0 {ev[frames[0]]['vel']:.4f} other {np.mean([ev[t]['vel'] for t in frames if t != frames[0]]):.4f})"
              f"  train {train_loss:.4f}  {secs:.0f}s", flush=True)

    model = TJ.build_model(args, device)
    model.load_state_dict(init["model"])
    t0 = time.perf_counter()
    log(0.0, 0, evaluate(model), float("nan"), time.perf_counter() - t0)     # zero-shot baseline
    for lr in [float(x) for x in own.lrs.split(",")]:
        torch.manual_seed(args.seed)
        model = TJ.build_model(args, device)
        model.load_state_dict(init["model"])
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        loader = DataLoader(Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True)
        it, step, run, t0 = iter(loader), 0, [], time.perf_counter()
        while step < own.steps:
            try:
                wall, interior = next(it)
            except StopIteration:
                it = iter(loader); continue
            wall, interior = wall.to(device), interior.to(device)
            for pg in optim.param_groups:
                pg["lr"] = lr * min(1.0, (step + 1) / own.warmup_steps)
            optim.zero_grad()
            wss, qi, vel, press = model(wall, interior, waveform)
            loss, _ = TJ.joint_loss(wss, qi, vel, press, wall, interior, args, y_mean, y_std, 1.0)
            if torch.isnan(loss):
                print("[warn] NaN loss, batch skipped", flush=True); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            optim.step()
            run.append(loss.item()); step += 1
            if step % own.eval_every == 0 or step == own.steps:
                log(lr, step, evaluate(model), float(np.mean(run[-own.eval_every:])), time.perf_counter() - t0)
                model.train()
        del model, optim
        torch.cuda.empty_cache()
    with open(os.path.join(own.out, "probe.json"), "w") as f:
        json.dump(dict(init=own.init_from, lrs=own.lrs, steps=own.steps, warmup_steps=own.warmup_steps,
                       eval_frames=frames, eval_cases=sorted(map(list, pick)), rows=rows), f, indent=1)
    plot(rows, frames, os.path.join(own.out, "probe.png"))
    print(f"[probe] wrote {own.out}", flush=True)


def plot(rows, frames, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = {r["frame"]: r for r in rows if r["step"] == 0}
    lrs = sorted({r["lr"] for r in rows if r["lr"] > 0})
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for lr in lrs:
        rs = [r for r in rows if r["lr"] == lr]
        steps = sorted({r["step"] for r in rs})
        get = lambda s, t, k: next(r[k] for r in rs if r["step"] == s and r["frame"] == t)
        s0 = [0] + steps
        f0 = [base[frames[0]]["loss"]] + [get(s, frames[0], "loss") for s in steps]
        ot = [np.mean([base[t]["loss"] for t in frames[1:]])] + [np.mean([get(s, t, "loss") for t in frames[1:]]) for s in steps]
        ax[0].plot(s0, f0, "o-", label=f"lr {lr:.0e}")
        ax[1].plot(s0, ot, "o-", label=f"lr {lr:.0e}")
        last = max(steps)
        ax[2].plot(frames, [get(last, t, "loss") for t in frames], "o-", label=f"lr {lr:.0e} @ {last}")
    ax[2].plot(frames, [base[t]["loss"] for t in frames], "k--", label="frame-0 model (zero-shot)")
    for a, t in zip(ax, (f"test loss, frame {frames[0]} (forgetting?)", "test loss, other frames (learning?)",
                         "test loss per frame, end of probe")):
        a.set_title(t); a.legend(fontsize=8); a.grid(alpha=0.3)
    ax[0].set_xlabel("step"); ax[1].set_xlabel("step"); ax[2].set_xlabel("frame")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
