"""Per-graph random loss mask favouring the aneurysm region."""
from __future__ import annotations

import torch


class AneurysmLossMask:
    """Attach `data.loss_mask: [N] bool`.

    For each graph independently:
      - sample `p ~ U[low, high]` (one draw per graph)
      - compute `tau = p * geo_dist.max()` on THAT graph
      - keep nodes with `geo_dist <= tau`
    So p=0 -> aneurysm only, p=1 -> whole graph, intermediate p -> aneurysm
    plus a geodesic ring of relative depth p.
    Works on either a single `Data` or a batched `Batch` (uses `data.batch`).
    """

    def __init__(self, low=0.0, high=0.25):
        assert 0.0 <= low <= high <= 1.0
        self.low = low
        self.high = high

    def __call__(self, data):
        geo = data.global_geo_dist
        batch = getattr(data, "batch", None)
        if batch is None:
            p = torch.empty((), device=geo.device).uniform_(self.low, self.high)
            tau = p * geo.max()                                       # scalar
        else:
            B = int(batch.max()) + 1
            p = torch.empty(B, device=geo.device).uniform_(self.low, self.high)
            gmax = torch.zeros(B, dtype=geo.dtype, device=geo.device)
            gmax = gmax.scatter_reduce(0, batch, geo, reduce="amax", include_self=False)
            tau = (p * gmax)[batch]                                   # [N], per-node tau
        data.loss_mask = geo <= tau
        return data
