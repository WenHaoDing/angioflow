"""Normalization stats for the interior velocity/pressure task.

Separate from angioflow/data/normalize.py (the WSS stats) so the two tasks don't
share mutable code. Computes, over a sampled train subset:
  pos_std    — isotropic scalar (rotation-invariant), from the surface statics;
               used to z-score BOTH wall.pos and interior.pos (shared frame) and
               to scale interior.wall_dist (a distance in the same frame).
  vel_std    — isotropic scalar, zero mean (velocity is a vector field, so a
               downstream random SO(3) rotation must leave normalized values
               unchanged — same convention as WSS `y`).
  press_mean/std — scalar (pressure is a rotation-invariant scalar with a large
               absolute offset, so it keeps a real mean).
"""
from __future__ import annotations

import os
import random

import torch


def compute_interior_stats(dataset, train_keys, n_cases=100, frames_per_case=8, seed=0,
                           press_zero_max=False):
    """press_zero_max: match transform.PressureZeroMax's convention (p -> p - p.max(),
    per (case, frame) sample, over the FULL frame -- same population this reads).
    Forces press_mean=0 rather than computing it, so a normalized value of 0 always
    decodes back to exactly "this case's own peak pressure", the same zero-mean-by-
    construction convention already used for vel_mean. Off by default -- existing
    callers (train_interior.py) are unaffected."""
    rng = random.Random(seed)
    keys = list(train_keys)
    rng.shuffle(keys)
    sample = keys[:min(n_cases, len(keys))]

    pos_chunks, vel_chunks, press_chunks = [], [], []
    for (v, cid) in sample:
        s = torch.load(dataset._static_path(v, cid), weights_only=False)          # surface static
        pos_chunks.append(s["pos"].float() - s["pos"].float().mean(0, keepdim=True))
        for t in rng.sample(range(dataset.n_frames), min(frames_per_case, dataset.n_frames)):
            f = torch.load(dataset._interior_frame_path(v, cid, t), weights_only=False)
            vel_chunks.append(f["vel"].float())
            press = f["press"].float()
            if press_zero_max:
                press = press - press.max()
            press_chunks.append(press)

    pos = torch.cat(pos_chunks, dim=0)
    vel = torch.cat(vel_chunks, dim=0)
    press = torch.cat(press_chunks, dim=0)

    pos_std = pos.std().clamp_min(1e-8).expand(3).clone()
    vel_std = vel.std().clamp_min(1e-8).expand(3).clone()
    press_mean = torch.zeros(1) if press_zero_max else press.mean().view(1).clone()
    press_std = press.std().clamp_min(1e-8).view(1).clone()

    return {
        "pos_mean":   torch.zeros(3),
        "pos_std":    pos_std,
        "vel_mean":   torch.zeros(3),
        "vel_std":    vel_std,
        "press_mean": press_mean,
        "press_std":  press_std,
        "press_zero_max": press_zero_max,
        "n_cases":    len(sample),
        "n_frames_per_case": frames_per_case,
    }


def load_or_compute_interior_stats(path, dataset, train_keys, n_cases=100,
                                   frames_per_case=8, seed=0, press_zero_max=False):
    if os.path.exists(path):
        stats = torch.load(path, weights_only=False)
        print(f"[stats] loaded interior stats from {path} "
              f"(n_cases={stats.get('n_cases')}, vel_std={stats['vel_std'][0].item():.4g}, "
              f"press_mean={stats['press_mean'].item():.4g}, press_std={stats['press_std'].item():.4g}, "
              f"press_zero_max={stats.get('press_zero_max', False)})")
        return stats
    stats = compute_interior_stats(dataset, train_keys, n_cases, frames_per_case, seed,
                                   press_zero_max=press_zero_max)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(stats, path)
    print(f"[stats] computed interior stats over {stats['n_cases']} cases × "
          f"{frames_per_case} frames: pos_std={stats['pos_std'][0].item():.4g}, "
          f"vel_std={stats['vel_std'][0].item():.4g}, press_std={stats['press_std'].item():.4g}, "
          f"press_zero_max={press_zero_max} -> saved to {path}")
    return stats
