"""PointNet++ for per-node WSS prediction on variable-size aneurysm meshes.

Same forward(data, waveform) interface as AngioGraphGPS.

Node features: normals [3] + geo_dist [n_geo_dist] + CotPE [dim_pe]
             + time_enc [dim_te] + waveform_enc [dim_wfe]

Architecture:
  SA1  (FPS + ball query, local) -> N/4  points
  SA2  (FPS + ball query, local) -> N/16 points
  SA3  (global max pool)         -> 1    per graph
  FP3  (kNN interp, 1->N/16)
  FP2  (kNN interp, N/16->N/4)
  FP1  (kNN interp, N/4->N)
  MLP head -> [N, dim_out]

"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, knn_interpolate, radius

from angioflow.models.transformer import TimeEmbedder, WaveformEmbedder


class _SAModule(nn.Module):
    """Set Abstraction: FPS + ball query + PointNetConv."""

    def __init__(self, ratio: float, r: float, mlp_channels: list[int]):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(MLP(mlp_channels), add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv((x, x[idx] if x is not None else None), (pos, pos[idx]), edge_index)
        return x, pos[idx], batch[idx]


class _GlobalSAModule(nn.Module):
    """Global Set Abstraction: per-point MLP then max pool -> 1 vector per graph."""

    def __init__(self, mlp_channels: list[int]):
        super().__init__()
        self.mlp = MLP(mlp_channels)

    def forward(self, x, pos, batch):
        x = self.mlp(torch.cat([x, pos], dim=-1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class _FPModule(nn.Module):
    """Feature Propagation: kNN interpolation + skip connection + MLP."""

    def __init__(self, k: int, mlp_channels: list[int]):
        super().__init__()
        self.k = k
        self.mlp = MLP(mlp_channels)

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=-1)
        return self.mlp(x), pos_skip, batch_skip


class AngioPointNet2(nn.Module):
    """PointNet++ predicting per-node WSS [N, 3] from aneurysm mesh data."""

    def __init__(
        self,
        n_geo_dist: int = 3,
        dim_te: int = 8,
        dim_wfe: int = 8,
        dim_emb: int = 64,
        dim_out: int = 3,
        base_ch: int = 64,
        sa_ratios: tuple[float, float] = (0.25, 0.25),
        sa_radii: tuple[float, float] = (0.3, 0.5),
    ):
        super().__init__()
        self.t_enc   = TimeEmbedder(dim_te)
        self.wf_enc  = WaveformEmbedder(dim_wfe)

        # feat_dim: normals(3) + geo_dist(n_geo_dist) + te(dim_te) + wfe(dim_wfe)
        fd = 3 + n_geo_dist + dim_te + dim_wfe
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 8

        # SA layers — PointNetConv local_nn input is feat_dim + 3 (appends pos_j - pos_i)
        self.sa1 = _SAModule(sa_ratios[0], sa_radii[0], [fd + 3,  c1 // 2, c1 // 2, c1])
        self.sa2 = _SAModule(sa_ratios[1], sa_radii[1], [c1 + 3,  c1,      c1,      c2])
        self.sa3 = _GlobalSAModule(                      [c2 + 3,  c2,      c2 * 2,  c3])

        # FP layers — MLP input = interpolated_dim + skip_dim
        self.fp3 = _FPModule(3, [c3 + c2,  c2, c2])
        self.fp2 = _FPModule(3, [c2 + c1,  c2, c1])
        self.fp1 = _FPModule(3, [c1 + fd,  c1, c1, dim_emb])

        self.head = nn.Sequential(
            nn.Linear(dim_emb, dim_emb // 2), nn.ReLU(),
            nn.Linear(dim_emb // 2, dim_out),
        )

    def _encode_nodes(self, data, waveform):
        t_enc  = self.t_enc(data.t)[data.batch]                   # [N, dim_te]
        wf_enc = self.wf_enc(data.t, waveform)[data.batch]        # [N, dim_wfe]
        # geo_dist can have inf for geodesically disconnected nodes after crop
        geo_dist = data.geo_dist.nan_to_num(nan=0.0, posinf=1.0)
        self._nan_check("normals", data.normals)
        self._nan_check("geo_dist", data.geo_dist)
        self._nan_check("t_enc", t_enc)
        self._nan_check("wf_enc", wf_enc)
        return torch.cat([data.normals, geo_dist, t_enc, wf_enc], dim=-1)  # [N, fd]

    @staticmethod
    def _nan_check(name, t):
        n = torch.isnan(t).sum().item()
        if n:
            finite = t[~torch.isnan(t)]
            stats = f"  min={finite.min().item():.3g}  max={finite.max().item():.3g}" if finite.numel() else "  (all NaN)"
            print(f"[NaN] {name}: {n}/{t.numel()}  shape={tuple(t.shape)}{stats}")
        return n > 0

    def forward(self, data, waveform):
        x0, pos0, b0 = self._encode_nodes(data, waveform), data.pos, data.batch
        self._nan_check("x0 (encode)", x0)
        self._nan_check("pos0", pos0)

        x1, pos1, b1 = self.sa1(x0, pos0, b0)
        if self._nan_check("x1 (sa1)", x1):
            self._nan_check("  x0 into sa1", x0)
        x2, pos2, b2 = self.sa2(x1, pos1, b1)
        self._nan_check("x2 (sa2)", x2)
        x3, pos3, b3 = self.sa3(x2, pos2, b2)
        self._nan_check("x3 (sa3)", x3)

        x, pos2, b2 = self.fp3(x3, pos3, b3, x2, pos2, b2)
        self._nan_check("x (fp3)", x)
        x, pos1, b1 = self.fp2(x,  pos2, b2, x1, pos1, b1)
        self._nan_check("x (fp2)", x)
        x, _,    _  = self.fp1(x,  pos1, b1, x0, pos0, b0)
        self._nan_check("x (fp1)", x)

        out = self.head(x)
        self._nan_check("out (head)", out)
        return out                                                 # [N, dim_out]
