"""Two-scale interior field decoder.

Each interior query point is refined by a stack of fusion blocks that alternate:
  * GLOBAL — cross-attention to the encoder's coarse tokens (bulk-flow context);
  * LOCAL  — gather the k nearest SURFACE nodes and embed the local wall geometry
             (relative position, normal, offset) via an edge MLP + pool. Pure
             geometry: no surface encoder features, no interior↔interior coupling.
  * FFN.
Interleaving (not one-shot) lets the query ask a sharper local question after
absorbing global context, and vice-versa. Cardiac phase enters here, on the query
side, so the surface encoder stays static.

Modular by design — QueryEmbed / GlobalCrossAttn / LocalSurfaceGather / FusionBlock
are independent nn.Modules so future variants can recombine them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import knn
from torch_geometric.utils import scatter, to_dense_batch

from angioflow.models.interior.constraints import WallNoSlipGate
from angioflow.models.transformer import TimeEmbedder, WaveformEmbedder


def fourier_encode(pos, n_freqs=6):
    """[..., 3] -> [..., 3 + 6*n_freqs]; pos assumed normalised (~O(1))."""
    freqs = (2.0 ** torch.arange(n_freqs, device=pos.device, dtype=pos.dtype)) * math.pi
    ang = pos.unsqueeze(-1) * freqs
    return torch.cat([pos, ang.sin().flatten(-2), ang.cos().flatten(-2)], dim=-1)


class QueryEmbed(nn.Module):
    """Interior query -> initial token: Fourier(pos) + wall_dist + time + waveform."""

    def __init__(self, dim, n_freqs=6, dim_te=8, dim_wfe=8):
        super().__init__()
        self.n_freqs = n_freqs
        self.t_enc = TimeEmbedder(dim_te)
        self.wf_enc = WaveformEmbedder(dim_wfe)
        in_dim = 3 + 6 * n_freqs + 1 + dim_te + dim_wfe
        self.mlp = Sequential(Linear(in_dim, dim), ReLU(), Linear(dim, dim))

    def forward(self, interior, waveform):
        t_enc = self.t_enc(interior.t)[interior.batch]
        wf_enc = self.wf_enc(interior.t, waveform)[interior.batch]
        wd = interior.wall_dist
        wd = wd.unsqueeze(-1) if wd.dim() == 1 else wd
        return self.mlp(torch.cat([fourier_encode(interior.pos, self.n_freqs), wd, t_enc, wf_enc], dim=-1))


class GlobalCrossAttn(nn.Module):
    """Pre-LN cross-attention from (flat, batched) queries to dense coarse tokens.

    Queries stay flat [K, D] with a batch vector; converted to a padded [B, Kmax, D]
    only for the attention op, then scattered back to flat.
    """

    def __init__(self, dim, n_heads):
        super().__init__()
        self.ln_q = LayerNorm(dim)
        self.ln_kv = LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, q, q_batch, kv_dense, kv_mask):
        q_dense, q_mask = to_dense_batch(self.ln_q(q), q_batch)          # [B, Kmax, D]
        a, _ = self.attn(q_dense, self.ln_kv(kv_dense), self.ln_kv(kv_dense),
                         key_padding_mask=~kv_mask, need_weights=False)
        return a[q_mask]                                                 # flat, aligned to q


class LocalSurfaceGather(nn.Module):
    """Per interior query: kNN to SURFACE nodes → edge MLP over local wall geometry → pool.

    Edge features (per query i, surface neighbour s):
      q_i        current query state (so the gather refines with the block)
      Δx         = x_s − x_i   (direction + distance to the wall patch)
      |Δx|       distance to the patch
      Δx·n_s     offset along the wall normal (perpendicular vs. grazing)
      n_s        local wall orientation
      wall_dist  the query's distance to the nearest wall
    kNN indices are computed once (static geometry) and reused across blocks.
    """

    def __init__(self, dim, k=16, reduce="mean"):
        super().__init__()
        self.k = k
        self.reduce = reduce
        self.ln = LayerNorm(dim)
        edge_in = dim + 3 + 1 + 1 + 3 + 1
        self.edge_mlp = Sequential(Linear(edge_in, dim * 2), ReLU(), Linear(dim * 2, dim))

    def forward(self, q, q_pos, q_wall_dist, q_batch, surf_pos, surf_normal, surf_batch, knn_idx):
        if knn_idx is None:
            assign = knn(surf_pos, q_pos, self.k, surf_batch, q_batch)   # [2, K*k]
            knn_idx = (assign[0], assign[1])                             # (query idx, surface idx)
        row, col = knn_idx
        qn = self.ln(q)
        dx = surf_pos[col] - q_pos[row]
        dist = dx.norm(dim=-1, keepdim=True)
        n_s = surf_normal[col]
        dn = (dx * n_s).sum(-1, keepdim=True)
        wd = q_wall_dist[row]
        wd = wd.unsqueeze(-1) if wd.dim() == 1 else wd
        e = self.edge_mlp(torch.cat([qn[row], dx, dist, dn, n_s, wd], dim=-1))
        agg = scatter(e, row, dim=0, dim_size=q.size(0), reduce=self.reduce)
        return agg, knn_idx


class FusionBlock(nn.Module):
    """One refinement step: global cross-attn + local surface gather + FFN (residual)."""

    def __init__(self, dim, n_heads=4, k_local=16, local_reduce="mean"):
        super().__init__()
        self.gattn = GlobalCrossAttn(dim, n_heads)
        self.local = LocalSurfaceGather(dim, k=k_local, reduce=local_reduce)
        self.ln_f = LayerNorm(dim)
        self.ffn = Sequential(Linear(dim, dim * 2), ReLU(), Linear(dim * 2, dim))

    def forward(self, q, q_batch, kv_dense, kv_mask, geom, knn_idx):
        q = q + self.gattn(q, q_batch, kv_dense, kv_mask)
        loc, knn_idx = self.local(q, *geom, knn_idx=knn_idx)
        q = q + loc
        q = q + self.ffn(self.ln_f(q))
        return q, knn_idx


class TwoScaleInteriorDecoder(nn.Module):
    """QueryEmbed → n_blocks × FusionBlock (global + local) → vel/press head."""

    def __init__(self, token_dim, dim=128, n_blocks=2, n_heads=4, k_local=16,
                 n_freqs=6, dim_te=8, dim_wfe=8, dim_vel=3, dim_press=1, local_reduce="mean",
                 wall_gate=True):
        super().__init__()
        self.dim_vel = dim_vel
        self.query_embed = QueryEmbed(dim, n_freqs, dim_te, dim_wfe)
        self.token_proj = Linear(token_dim, dim)
        self.blocks = ModuleList([
            FusionBlock(dim, n_heads, k_local, local_reduce) for _ in range(n_blocks)])
        self.head = Sequential(Linear(dim, dim // 2), ReLU(), Linear(dim // 2, dim_vel + dim_press))
        # hard no-slip constraint: |v| -> 0 at the wall (pressure is not gated)
        self.wall_gate = WallNoSlipGate() if wall_gate else None

    def forward(self, interior, waveform, tokens, token_batch, surf_pos, surf_normal, surf_batch):
        q = self.query_embed(interior, waveform)
        kv_dense, kv_mask = to_dense_batch(self.token_proj(tokens), token_batch)
        geom = (interior.pos, interior.wall_dist, interior.batch, surf_pos, surf_normal, surf_batch)
        knn_idx = None
        for blk in self.blocks:
            q, knn_idx = blk(q, interior.batch, kv_dense, kv_mask, geom, knn_idx)
        out = self.head(q)
        vel, press = out[:, :self.dim_vel], out[:, self.dim_vel:]
        if self.wall_gate is not None:
            vel = self.wall_gate(vel, interior.wall_dist)
        return vel, press
