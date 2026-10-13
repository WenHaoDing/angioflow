"""Geometry-only surface encoders for the interior flow-field baselines.

Each encoder consumes the SURFACE graph (no time/waveform — geometry is static,
so it is run once per case) and returns a small set of coarse tokens that carry
global geometry context:

    encode_tokens(wall) -> (tokens [T, dim_emb], token_batch [T])

Local wall detail is handled separately by the decoder's LocalSurfaceGather
(which reads raw wall pos/normals), so these encoders only provide the GLOBAL
scale. Two backbones give two baselines: HierGPS (mesh) and PointNet++ (points).

Kept independent of the WSS models; only pure leaf utilities (CotPE,
RedrawProjection) are imported read-only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import (GINEConv, GPSConv, MLP, PointNetConv, fps,
                                global_max_pool, knn_graph, knn_interpolate, radius)
from torch_geometric.utils import to_undirected

from angioflow.models.transformer import CotPE, RedrawProjection


class _GeoNodeEncoder(nn.Module):
    """Geometry-only node features: pos, normals, geo_dist + cotangent PE. No time."""

    def __init__(self, n_cot, node_dim_in=6, n_geo_dist=3, dim_pe=16, dim_emb=128,
                 pe_n_heads=2, pe_n_layers=2, pe_use_mlp=True):
        super().__init__()
        self.cot_pe = CotPE(n_cot, dim_pe=dim_pe, n_heads=pe_n_heads,
                            n_layers=pe_n_layers, use_mlp=pe_use_mlp)
        self.lin = nn.Linear(node_dim_in + n_geo_dist, dim_emb - dim_pe)

    def forward(self, data):
        B = int(data.batch.max()) + 1
        k = data.cot_eigvec.size(1)
        eigval = data.cot_eigval.view(B, k)[data.batch]
        pe = self.cot_pe(data.cot_eigvec, eigval, self.training)
        geo = data.geo_dist.nan_to_num(nan=0.0, posinf=1.0)
        x = torch.cat([data.pos, data.normals, geo], dim=-1)
        return torch.cat([self.lin(x), pe], dim=-1)


class _GeoEdgeEncoder(nn.Module):
    """Simple geometric edge features: |Δ(pos,normals)| + edge centre -> dim_emb."""

    def __init__(self, dim_emb=128):
        super().__init__()
        self.lin = nn.Linear(12, dim_emb)

    def forward(self, data):
        x = torch.cat([data.pos, data.normals], dim=-1)
        i, j = data.edge_index
        grad = (x[i] - x[j]).abs()
        centre = (x[i] + x[j]) / 2.0
        return self.lin(torch.cat([grad, centre], dim=-1))


class GeoHierGPSEncoder(nn.Module):
    """Hierarchical GPS as a static geometry encoder → coarse GPS tokens.

    pre-convs (GINEConv) → [FPS pool → kNN graph → inter-convs] × n_pool → GPS.

    bidirectional (default True): a kNN graph is inherently directed (A being B's
    nearest point doesn't make B A's nearest point) — confirmed empirically (~30% of
    edges in a typical kNN graph have no reverse pair). Left un-symmetrized,
    inter_convs/gps_convs at each coarse level would only aggregate from about half
    of a node's true nearest neighbours. This only covers the coarse-level graph
    rebuilt HERE, inside encode_tokens_with_pos (it doesn't exist until FPS pooling
    runs, so a pre-processing transform can't reach it) — the full-resolution
    data.edge_index (pre_convs) needs the same fix applied one level up, at the
    dataset-transform stage (see angioflow.transform.SymmetrizeEdges; not currently
    wired into train_interior.py's pipeline).
    """

    def __init__(self, n_cot, dim_emb=128, n_geo_dist=3, dim_pe=16, n_pre=2, n_gps=4,
                 n_pool=2, n_inter=0, pool_ratio=0.25, pool_knn_k=6, n_heads=4,
                 attn_type="performer", pe_n_heads=2, pe_n_layers=2, pe_use_mlp=True,
                 grad_ckpt=False, bidirectional=True):
        super().__init__()
        self.dim_emb = dim_emb
        self.n_pool = n_pool
        self.pool_ratio = pool_ratio
        self.pool_knn_k = pool_knn_k
        self.grad_ckpt = grad_ckpt
        self.bidirectional = bidirectional
        self.node_encoder = _GeoNodeEncoder(n_cot, dim_emb=dim_emb, n_geo_dist=n_geo_dist,
                                            dim_pe=dim_pe, pe_n_heads=pe_n_heads,
                                            pe_n_layers=pe_n_layers, pe_use_mlp=pe_use_mlp)
        self.edge_encoder = _GeoEdgeEncoder(dim_emb)

        def _gine():
            return GINEConv(MLP([dim_emb, dim_emb * 2, dim_emb], act="relu"), edge_dim=dim_emb)

        self.pre_convs = ModuleList([_gine() for _ in range(n_pre)])
        self.pre_norms = ModuleList([nn.LayerNorm(dim_emb) for _ in range(n_pre)])
        self.inter_convs = ModuleList([
            ModuleList([_gine() for _ in range(n_inter)]) for _ in range(n_pool)])
        self.inter_norms = ModuleList([
            ModuleList([nn.LayerNorm(dim_emb) for _ in range(n_inter)]) for _ in range(n_pool)])
        self.gps_convs = ModuleList()
        for _ in range(n_gps):
            mlp = Sequential(Linear(dim_emb, dim_emb), ReLU(), Linear(dim_emb, dim_emb))
            self.gps_convs.append(GPSConv(dim_emb, GINEConv(mlp), heads=n_heads, attn_type=attn_type))
        self.redraw_projection = RedrawProjection(
            self.gps_convs, redraw_interval=1000 if attn_type == "performer" else None)

    def encode_tokens_with_pos(self, data):
        x = self.node_encoder(data)
        ea = self.edge_encoder(data)
        for conv, norm in zip(self.pre_convs, self.pre_norms):
            x = norm(x + conv(x, data.edge_index, ea))

        pos, batch = data.pos, data.batch
        ei, eac = data.edge_index, ea
        for convs, norms in zip(self.inter_convs, self.inter_norms):
            perm = fps(pos, batch, ratio=self.pool_ratio)
            x, pos, batch = x[perm], pos[perm], batch[perm]
            ei = knn_graph(pos, k=self.pool_knn_k, batch=batch, loop=False)
            if self.bidirectional:
                ei = to_undirected(ei, num_nodes=pos.size(0))
            eac = x.new_zeros(ei.size(1), self.dim_emb)
            for conv, norm in zip(convs, norms):
                x = norm(x + conv(x, ei, eac))

        for conv in self.gps_convs:
            if self.grad_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(conv, x, ei, batch, edge_attr=eac, use_reentrant=False)
            else:
                x = conv(x, ei, batch, edge_attr=eac)
        return x, batch, pos

    def encode_tokens(self, data):
        tokens, batch, _ = self.encode_tokens_with_pos(data)
        return tokens, batch


class _SA(nn.Module):
    """PointNet++ set abstraction: FPS + ball query + PointNetConv."""

    def __init__(self, ratio, r, mlp):
        super().__init__()
        self.ratio, self.r = ratio, r
        self.conv = PointNetConv(MLP(mlp), add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        ei = torch.stack([col, row], dim=0)
        x = self.conv((x, x[idx] if x is not None else None), (pos, pos[idx]), ei)
        return x, pos[idx], batch[idx]


class _GlobalSA(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = MLP(mlp)

    def forward(self, x, pos, batch):
        x = self.mlp(torch.cat([x, pos], dim=-1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class _FP(nn.Module):
    def __init__(self, k, mlp):
        super().__init__()
        self.k = k
        self.mlp = MLP(mlp)

    def forward(self, x, pos, batch, xs, ps, bs):
        x = knn_interpolate(x, pos, ps, batch, bs, k=self.k)
        if xs is not None:
            x = torch.cat([x, xs], dim=-1)
        return self.mlp(x), ps, bs


class GeoPointNetEncoder(nn.Module):
    """PointNet++ as a static geometry encoder → coarse tokens (with global context).

    SA1 → SA2 → global SA3 → FP3 back to the SA2 level.
    """

    def __init__(self, dim_emb=128, n_geo_dist=3, base_ch=64,
                 sa_ratios=(0.25, 0.25), sa_radii=(0.3, 0.5)):
        super().__init__()
        self.dim_emb = dim_emb
        fd = 3 + n_geo_dist
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 8
        self.sa1 = _SA(sa_ratios[0], sa_radii[0], [fd + 3, c1 // 2, c1 // 2, c1])
        self.sa2 = _SA(sa_ratios[1], sa_radii[1], [c1 + 3, c1, c1, c2])
        self.sa3 = _GlobalSA([c2 + 3, c2, c2 * 2, c3])
        self.fp3 = _FP(3, [c3 + c2, c2, c2])
        self.token_out = nn.Linear(c2, dim_emb)

    def _geo(self, data):
        geo = data.geo_dist.nan_to_num(nan=0.0, posinf=1.0)
        return torch.cat([data.normals, geo], dim=-1)

    def encode_tokens_with_pos(self, data):
        x0, pos0, b0 = self._geo(data), data.pos, data.batch
        x1, pos1, b1 = self.sa1(x0, pos0, b0)
        x2, pos2, b2 = self.sa2(x1, pos1, b1)
        x3, pos3, b3 = self.sa3(x2, pos2, b2)
        x, pos2, b2 = self.fp3(x3, pos3, b3, x2, pos2, b2)
        return self.token_out(x), b2, pos2

    def encode_tokens(self, data):
        tokens, batch, _ = self.encode_tokens_with_pos(data)
        return tokens, batch
