"""Crop the graph to the aneurysm plus a random geodesic ring around it."""
from __future__ import annotations

import torch


class AneurysmCrop:
    """Crop to `node_type == 1` (aneurysm) plus a random ring of surrounding nodes.

    Rule (per single-graph `Data`; relies on PyG's `Data.subgraph`):
      - always keep nodes with `node_type == 1`
      - sample `p ~ U[low, high]`; additionally keep nodes with
        `geo_dist <= p * geo_dist.max()`
    p=0 -> aneurysm only; p=1 -> whole graph; p=0.25 -> aneurysm + ~25%-deep ring.

    `aneurysm_only=True` disables the random ring entirely: keep exactly the
    aneurysm region (`node_type == 1` ∪ `global_geo_dist == 0`) — deterministic,
    so the same case always yields the same graph.

    Use as a *pre-batch* transform (set on `dataset.transform`) — `Data.subgraph`
    handles the index/edge remapping for you.
    """

    def __init__(self, low=0.0, high=0.25, aneurysm_only=False):
        assert 0.0 <= low <= high <= 1.0
        self.low = low
        self.high = high
        self.aneurysm_only = aneurysm_only

    def __call__(self, data):
        if hasattr(data, "batch") and data.batch is not None:
            raise RuntimeError("AneurysmCrop must run pre-batch (set on dataset.transform).")
        g = data.global_geo_dist
        if self.aneurysm_only:
            keep = (data.node_type == 1) | (g == 0)
        else:
            p = torch.empty((), device=g.device).uniform_(self.low, self.high)
            tau = p * g.max()
            keep = (data.node_type == 1) | (g <= tau)
        return data.subgraph(keep)
