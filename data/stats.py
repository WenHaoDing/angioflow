"""Helpers to compute normalization stats (pos, WSS, edge attrs) on a calibration set."""
from __future__ import annotations

import torch


def edge_attr(pos, normal, edge_index):
    """Same edge feature the model sees: cat(|Δ(pos,normal)|, centre)."""
    x = torch.cat([pos, normal], dim=-1)
    i, j = edge_index
    return torch.cat([(x[i] - x[j]).abs(), (x[i] + x[j]) / 2.0], dim=-1)        # [E, 2*node_dim_in]


def compute_stats(dataset, calib_cases, frame_ids):
    """Iterate calibration cases × frames, return a dict of normalization stats.

    Stats are computed on centred pos (so they match what the model sees after the
    Center transform). Random rotation isn't applied — it preserves |Δ| and is
    mean-zero on pos/y, so per-channel variance is unaffected in expectation.
    """
    pos_chunks, y_chunks, edge_chunks = [], [], []
    for ci in calib_cases:
        static = dataset._load_static(int(ci))
        pos = static["pos"] - static["pos"].mean(dim=0, keepdim=True)
        pos_chunks.append(pos)
        edge_chunks.append(edge_attr(pos, static["normal"], static["edge_index"]))
        for t in frame_ids:
            y_chunks.append(torch.load(dataset._wss_path(int(ci), int(t))))

    pos = torch.cat(pos_chunks, dim=0)
    y = torch.cat(y_chunks, dim=0)
    edges = torch.cat(edge_chunks, dim=0)
    return {
        "pos_std": pos.std().item(),                # isotropic scalar (direction-preserving scaling)
        "y_std": y.std().item(),                    # isotropic scalar; no y_mean -> no direction shift
        "edge_mean": edges.mean(dim=0),             # per-channel ok: edge features aren't a vector field
        "edge_std": edges.std(dim=0),
    }
