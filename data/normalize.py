"""Per-component pos/y/edge normalization stats over a sampled train subset."""
from __future__ import annotations

import os
import random

import torch


def compute_pos_y_stats(dataset, train_keys, n_cases=100, frames_per_case=8, seed=0):
    """Sample `n_cases` train cases × `frames_per_case` random frames; return stats dict.

    Conventions (rotation-invariant):
      - `pos` is per-case-centered (centroid subtracted) before any statistics.
      - Stats are made isotropic across x/y/z so a downstream random SO(3) rotation
        leaves the normalized values unchanged. `pos_mean` is therefore zeros(3),
        `pos_std` is a single scalar broadcast across components; same for `y`.
      - Edge stats are computed on (centered/normalized pos, raw normals) and
        grouped per 3-feature block (|Δpos|, |Δnormal|, midpoint(pos), midpoint(normal))
        so each block shares one scalar mean/std — also rotation-invariant.
    """
    rng = random.Random(seed)
    keys = list(train_keys)
    rng.shuffle(keys)
    sample = keys[:min(n_cases, len(keys))]

    pos_chunks, y_chunks, statics = [], [], []
    for (v, cid) in sample:
        s = torch.load(dataset._static_path(v, cid), weights_only=False)
        pos_c = s["pos"].float() - s["pos"].float().mean(0, keepdim=True)        # center per case
        pos_chunks.append(pos_c)
        statics.append({"pos_c": pos_c, "normals": s["normals"].float(), "edge_index": s["edge_index"]})
        t_idxs = rng.sample(range(dataset.n_frames),
                            min(frames_per_case, dataset.n_frames))
        for t in t_idxs:
            y_chunks.append(torch.load(dataset._wss_path(v, cid, t), weights_only=False))
    pos = torch.cat(pos_chunks, dim=0)                                            # [N_tot, 3]
    y   = torch.cat(y_chunks,   dim=0).float()                                    # [M_tot, 3]

    pos_std_scalar = pos.std().clamp_min(1e-8)
    y_std_scalar   = y.std().clamp_min(1e-8)
    pos_mean = torch.zeros(3)
    y_mean   = torch.zeros(3)
    pos_std  = pos_std_scalar.expand(3).clone()
    y_std    = y_std_scalar.expand(3).clone()

    # edge stats: groups of 3 features share a mean/std (isotropic per group)
    edge_attr_chunks = []
    for s in statics:
        x = torch.cat([s["pos_c"] / pos_std_scalar, s["normals"]], dim=-1)        # [N, 6]
        i, j = s["edge_index"]
        edge_attr_chunks.append(torch.cat([(x[i] - x[j]).abs(),
                                           (x[i] + x[j]) / 2.0], dim=-1))         # [E, 12]
    edge_attr = torch.cat(edge_attr_chunks, dim=0)
    edge_mean = torch.zeros(12)
    edge_std  = torch.zeros(12)
    for a, b in [(0, 3), (3, 6), (6, 9), (9, 12)]:                                # |Δpos|, |Δn|, mid(pos), mid(n)
        block = edge_attr[:, a:b]
        edge_mean[a:b] = block.mean()
        edge_std[a:b]  = block.std().clamp_min(1e-8)

    return {
        "pos_mean":  pos_mean,
        "pos_std":   pos_std,
        "y_mean":    y_mean,
        "y_std":     y_std,
        "edge_mean": edge_mean,
        "edge_std":  edge_std,
        "n_cases":   len(sample),
        "n_frames_per_case": frames_per_case,
    }


def load_or_compute_stats(path, dataset, train_keys, n_cases=100, frames_per_case=8, seed=0):
    """Load stats from `path` if it exists, else compute + save."""
    if os.path.exists(path):
        stats = torch.load(path, weights_only=False)
        print(f"[stats] loaded from {path} "
              f"(n_cases={stats.get('n_cases')}, y_std={stats['y_std'].tolist()})")
        return stats
    stats = compute_pos_y_stats(dataset, train_keys, n_cases, frames_per_case, seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(stats, path)
    print(f"[stats] computed over {stats['n_cases']} cases × {frames_per_case} frames; "
          f"y_std={stats['y_std'].tolist()} -> saved to {path}")
    return stats
