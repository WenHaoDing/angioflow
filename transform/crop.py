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

    `as_loss_mask=True` keeps the whole graph and attaches the same region as
    `data.loss_mask` ([N] bool) instead -- the model sees the full mesh, the loss is
    taken only where the crop would have kept nodes (train.py --no_crop --crop_loss_mask).
    One code path for both, so the masked region is exactly the crop region.
    `mask_attr` names that attribute (eval_wss.py uses "viz_mask" to mark the region for
    rendering only, without turning it into a metrics/loss region).
    """

    def __init__(self, low=0.0, high=0.25, aneurysm_only=False, as_loss_mask=False,
                 mask_attr="loss_mask"):
        assert 0.0 <= low <= high <= 1.0
        self.low = low
        self.high = high
        self.aneurysm_only = aneurysm_only
        self.as_loss_mask = as_loss_mask
        self.mask_attr = mask_attr

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
        if self.as_loss_mask:
            setattr(data, self.mask_attr, keep)
            return data
        return data.subgraph(keep)
