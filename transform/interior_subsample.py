"""Graph-free random subsample of the interior pool.

The baseline field models are query-style: they read each interior point
independently (position + wall_dist → cross-attention), so they need a random
subset of the pool per step, NOT the induced graph. This transform slices the
node attributes present on the interior Data and touches nothing else — in
particular it never materialises the pool `edge_index`, so with mmap loads those
bytes are never read from disk. (Use InteriorSubsampleGraph instead when a variant
actually consumes the interior graph, e.g. a gradient-supervised loss.)

Operates on the (wall, interior) pair; wall passes through untouched.
"""
from __future__ import annotations

import torch


class InteriorSubsample:
    """Draw a random subset of the interior pool: sample_ratio of the points, but
    never more than max_points (a CAP). The ratio gives natural per-case variation
    while the cap bounds the largest pools so memory stays predictable — e.g.
    ratio 0.1 keeps ~7.8k on a median case but caps a 12.5k draw down to max_points.
    """

    NODE_KEYS = ("pos", "vel", "press", "grad_v", "grad_p", "wall_dist")

    def __init__(self, sample_ratio=0.1, max_points=None, min_points=1):
        self.sample_ratio = sample_ratio
        self.max_points = max_points
        self.min_points = min_points

    def __call__(self, data):
        if isinstance(data, tuple):
            wall, interior = data
            return wall, self._apply(interior)
        return self._apply(data)

    def _apply(self, d):
        P = d.num_nodes
        k = max(self.min_points, int(round(P * self.sample_ratio)))
        if self.max_points is not None:
            k = min(k, self.max_points)          # cap the largest pools
        k = min(k, P)
        if k >= P:
            return d
        idx = torch.randperm(P)[:k]
        for key in self.NODE_KEYS:
            if key in d:
                d[key] = d[key][idx]
        return d
