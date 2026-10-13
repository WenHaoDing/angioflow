"""On-the-fly graph transforms."""
from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import subgraph


class Normalize(BaseTransform):
    """Center pos per-graph + scale by scalar pos_std; scale y by scalar y_std.

    Both scalings are isotropic so the geometric direction of pos and the WSS vector
    direction are exactly preserved. No mean is subtracted from y (a scalar shift
    would still rotate the WSS direction by (m, m, m)).
    """

    def __init__(self, pos_std: float | None = None, y_std: float | None = None):
        super().__init__()
        self.pos_std = pos_std
        self.y_std = y_std

    def forward(self, data: Data) -> Data:
        data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
        if self.pos_std:
            data.pos = data.pos / self.pos_std
        if self.y_std:
            data.y = data.y / self.y_std
        return data


def _random_so3(device=None, dtype=torch.float32):
    """Uniform-random rotation in SO(3) via a normalized random quaternion."""
    q = torch.randn(4, device=device, dtype=dtype)
    q = q / q.norm()
    w, x, y, z = q
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z),   2 * (x * y - z * w),       2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w),       1 - 2 * (x * x + z * z),   2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w),       2 * (y * z + x * w),       1 - 2 * (x * x + y * y)]),
    ])


class RandomRotation(BaseTransform):
    """Apply a uniform-random SO(3) rotation to all listed vector fields (pos, normal, y by default)."""

    def __init__(self, keys=("pos", "normal", "y")):
        super().__init__()
        self.keys = keys

    def forward(self, data: Data) -> Data:
        R = _random_so3(device=data.pos.device, dtype=data.pos.dtype)
        for k in self.keys:
            v = getattr(data, k, None)
            if v is not None:
                data[k] = v @ R.T
        return data


class GeoDistCrop(BaseTransform):
    """Keep all geo_dist==0 (dome/overlap) nodes + the smallest p-fraction (by geo_dist) of the rest.

    p ~ Uniform(p_min, p_max) per call. Contiguous, vessel-following, variable N.
    """

    def __init__(self, p_min=0.1, p_max=0.3, zero_tol=1e-8):
        super().__init__()
        self.p_min, self.p_max, self.zero_tol = p_min, p_max, zero_tol

    def forward(self, data: Data) -> Data:
        g = data.geo_dist
        seed = g <= self.zero_tol                                  # dome / overlap nodes
        rest = (~seed).nonzero(as_tuple=False).view(-1)
        p = float(torch.empty(1).uniform_(self.p_min, self.p_max))
        k = int(round(p * rest.numel()))
        keep_rest = rest[torch.argsort(g[rest])[:k]] if k > 0 else rest[:0]
        mask = seed.clone()
        mask[keep_rest] = True

        n_old = data.num_nodes
        edge_index, _ = subgraph(mask, data.edge_index, relabel_nodes=True, num_nodes=n_old)
        out = data.__class__()
        for key, val in data:
            if key == "edge_index":
                continue
            if torch.is_tensor(val) and val.dim() >= 1 and val.size(0) == n_old:
                val = val[mask]
            out[key] = val
        out.edge_index = edge_index
        return out
