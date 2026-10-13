"""Joint two-field decoder: WSS on the wall + velocity/pressure in the lumen, refined
together, VGGT-style.

Both query sets are decoded query-point-style against ONE shared set of geometry
tokens, and — the joint part — the two query populations exchange information every
block via bidirectional cross-attention (surface queries attend to the interior set
and vice versa). Physically: WSS is the wall trace of the interior velocity gradient,
so each field carries information the other's supervision never sees directly.

One JointFusionBlock =
  1. GLOBAL      — each query set cross-attends to the shared geometry tokens;
  2. LOCAL       — surface queries: GATv2 over the k-hop mesh neighbourhood
                   (wss.LocalMeshGather); interior queries: kNN gather of local wall
                   geometry (interior.LocalSurfaceGather);
  3. CROSS-FIELD — surface <-> interior bidirectional cross-attention, computed from
                   the SAME snapshot of both sets (symmetric, no ordering artifact);
  4. FFN         — per set.
Repeated n_blocks times (2-3), like VGGT's alternating frame/global attention.

Every attention/gather/embedding module is imported from the existing WSS and
interior decoders — nothing here reimplements them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, ModuleList, ReLU, Sequential
from torch_geometric.utils import to_dense_batch

from torch_geometric.nn import GATv2Conv, knn

from angioflow.models.interior.constraints import WallNoSlipGate
from angioflow.models.interior.decoder import GlobalCrossAttn, LocalSurfaceGather, QueryEmbed
from angioflow.models.wss.query_decoder import (LocalMeshGather, WSSQueryEmbed,
                                                k_hop_edges, query_local_edges)


class LocalFieldGather(nn.Module):
    """Per destination query: attention over its k nearest SOURCE queries' current
    states — the local state-fusion channel between the two fields, one direction
    per instance:

      surface <- interior (normal_side="dst"): each wall query reads the evolving
        velocity representations right next to it. WSS is mu * du/dn, a function of
        exactly that boundary-layer field, so the physically sharpest signal is
        hard-wired rather than left for global attention to discover.
      interior <- surface (normal_side="src"): the mirror image — a near-wall point
        reads the neighbouring WSS states; a confidently-high WSS estimate is direct
        evidence that the local velocity gradient must be steep. Causality is one-way
        but INFERENCE is not: joint estimation is about consistency between two views
        of the same flow, so information flows both ways.

    States (not raw geometry) are gathered, because the other field's estimate is
    exactly what its queries are busy computing. Relative geometry rides along as edge
    features: dx, |dx| and dx.n — the offset along the WALL normal (the destination's
    own normal for surface queries, the source neighbour's for interior queries),
    i.e. the direction the WSS-defining gradient is taken in.

    Bipartite GATv2 (dynamic per-query attention), same head convention as
    LocalMeshGather. kNN indices are static geometry — computed once per forward and
    cached across blocks; states are re-read fresh each block.
    """

    def __init__(self, dim, k=16, n_heads=4, normal_side="dst"):
        super().__init__()
        assert normal_side in ("dst", "src")
        self.k = k
        self.normal_side = normal_side
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        head_dim = max(1, dim // n_heads)
        self.gat = GATv2Conv((dim, dim), head_dim, heads=n_heads, concat=True,
                             add_self_loops=False, edge_dim=5)
        self.out_proj = Linear(head_dim * n_heads, dim)

    def forward(self, q_dst, q_src, dst_pos, src_pos, normals, dst_batch, src_batch, knn_idx):
        if knn_idx is None:
            assign = knn(src_pos, dst_pos, self.k, src_batch, dst_batch)   # [2, Q_dst*k]
            knn_idx = (assign[0], assign[1])                               # (dst idx, src idx)
        row, col = knn_idx
        dx = src_pos[col] - dst_pos[row]
        dist = dx.norm(dim=-1, keepdim=True)
        n = normals[row] if self.normal_side == "dst" else normals[col]
        dn = (dx * n).sum(-1, keepdim=True)
        edge_attr = torch.cat([dx, dist, dn], dim=-1)                      # [E, 5]
        out = self.gat((self.ln_kv(q_src), self.ln_q(q_dst)),
                       torch.stack([col, row]), edge_attr=edge_attr)
        return self.out_proj(out), knn_idx


class JointFusionBlock(nn.Module):
    """One refinement step over BOTH query sets (see module docstring)."""

    def __init__(self, dim, n_heads=4, local_heads=4, k_local=16, k_field=16,
                 cross_field=True):
        super().__init__()
        self.gattn_s = GlobalCrossAttn(dim, n_heads)
        self.gattn_i = GlobalCrossAttn(dim, n_heads)
        self.local_s = LocalMeshGather(dim, n_heads=local_heads)
        self.local_i = LocalSurfaceGather(dim, k=k_local)
        # bidirectional local state fusion (k_field=0 disables both directions):
        # wall queries <- k nearest interior states, and interior <- k nearest wall states
        if k_field > 0:
            self.field_s = LocalFieldGather(dim, k=k_field, n_heads=local_heads, normal_side="dst")
            self.field_i = LocalFieldGather(dim, k=k_field, n_heads=local_heads, normal_side="src")
        else:
            self.field_s = self.field_i = None
        # global cross-field exchange (stage 3); cross_field=False disables it so
        # local-only vs global-only vs both can be ablated cleanly
        if cross_field:
            self.cross_s = GlobalCrossAttn(dim, n_heads)   # surface queries <- interior set
            self.cross_i = GlobalCrossAttn(dim, n_heads)   # interior queries <- surface set
        else:
            self.cross_s = self.cross_i = None
        self.ln_s = LayerNorm(dim)
        self.ln_i = LayerNorm(dim)
        self.ffn_s = Sequential(Linear(dim, dim * 2), ReLU(), Linear(dim * 2, dim))
        self.ffn_i = Sequential(Linear(dim, dim * 2), ReLU(), Linear(dim * 2, dim))

    def forward(self, qs, qs_batch, qi, qi_batch, tok_dense, tok_mask,
                s_geom, i_geom, local_ei, knn_idx, field_idx, field_idx_rev):
        # 1. global: both sets read the shared geometry tokens
        qs = qs + self.gattn_s(qs, qs_batch, tok_dense, tok_mask)
        qi = qi + self.gattn_i(qi, qi_batch, tok_dense, tok_mask)

        # 2a. local geometry branches (unchanged from the single-task decoders)
        qs = qs + self.local_s(qs, *s_geom, local_ei)
        loc, knn_idx = self.local_i(qi, *i_geom, knn_idx=knn_idx)
        qi = qi + loc

        # 2b. bidirectional local state fusion — symmetric snapshot (both directions
        # read the same post-2a states, so neither sees the other's 2b update)
        if self.field_s is not None:
            s_pos, s_normals = s_geom[0], s_geom[1]
            i_pos, i_batch = i_geom[0], i_geom[2]
            f_s, field_idx = self.field_s(qs, qi, s_pos, i_pos, s_normals,
                                          qs_batch, i_batch, field_idx)
            f_i, field_idx_rev = self.field_i(qi, qs, i_pos, s_pos, s_normals,
                                              i_batch, qs_batch, field_idx_rev)
            qs = qs + f_s
            qi = qi + f_i

        # 3. global cross-field: symmetric snapshot, both directions
        if self.cross_s is not None:
            qs_dense, qs_mask = to_dense_batch(qs, qs_batch)
            qi_dense, qi_mask = to_dense_batch(qi, qi_batch)
            qs = qs + self.cross_s(qs, qs_batch, qi_dense, qi_mask)
            qi = qi + self.cross_i(qi, qi_batch, qs_dense, qs_mask)

        # 4. FFN
        qs = qs + self.ffn_s(self.ln_s(qs))
        qi = qi + self.ffn_i(self.ln_i(qi))
        return qs, qi, knn_idx, field_idx, field_idx_rev


class JointTwoFieldDecoder(nn.Module):
    """Two query embeddings -> n_blocks x JointFusionBlock -> two heads.

    Outputs: wss [Q_s, 3] (optionally tangentially projected), vel [Q_i, 3],
    press [Q_i, 1] (velocity optionally no-slip gated — off by default, no physics).
    """

    def __init__(self, token_dim, dim=128, n_blocks=2, n_heads=4, local_heads=4,
                 k_local=16, k_field=16, cross_field=True, local_hops=2, n_freqs=6,
                 dim_te=8, dim_wfe=8, tangential=True, wall_gate=False):
        super().__init__()
        self.local_hops = local_hops
        self.tangential = tangential
        self.embed_s = WSSQueryEmbed(dim, n_freqs, dim_te, dim_wfe)
        self.embed_i = QueryEmbed(dim, n_freqs, dim_te, dim_wfe)
        self.token_proj = Linear(token_dim, dim)
        self.blocks = ModuleList([
            JointFusionBlock(dim, n_heads, local_heads, k_local, k_field, cross_field)
            for _ in range(n_blocks)])
        self.head_s = Sequential(Linear(dim, dim // 2), ReLU(), Linear(dim // 2, 3))
        self.head_i = Sequential(Linear(dim, dim // 2), ReLU(), Linear(dim // 2, 4))
        self.wall_gate = WallNoSlipGate() if wall_gate else None

    def forward(self, wall, interior, waveform, tokens, token_batch, query_idx=None):
        expanded_ei = k_hop_edges(wall.edge_index, wall.num_nodes, self.local_hops)
        if query_idx is None:
            s_pos, s_normals, s_batch = wall.pos, wall.normals, wall.batch
            local_ei = expanded_ei
        else:
            s_pos, s_normals, s_batch = wall.pos[query_idx], wall.normals[query_idx], wall.batch[query_idx]
            local_ei = query_local_edges(expanded_ei, query_idx, wall.num_nodes)

        qs = self.embed_s(s_pos, s_normals, s_batch, wall.t, waveform)
        qi = self.embed_i(interior, waveform)
        tok_dense, tok_mask = to_dense_batch(self.token_proj(tokens), token_batch)
        # neighbours for the surface local branch come from the FULL wall (queried or
        # not); the interior local branch gathers from the full wall too
        s_geom = (s_pos, s_normals, wall.pos, wall.normals)
        i_geom = (interior.pos, interior.wall_dist, interior.batch,
                  wall.pos, wall.normals, wall.batch)

        knn_idx = None        # interior->wall geometry kNN: static, cached across blocks
        field_idx = None      # wall<-interior state kNN: likewise
        field_idx_rev = None  # interior<-wall state kNN: likewise
        for blk in self.blocks:
            qs, qi, knn_idx, field_idx, field_idx_rev = blk(
                qs, s_batch, qi, interior.batch, tok_dense, tok_mask,
                s_geom, i_geom, local_ei, knn_idx, field_idx, field_idx_rev)

        wss = self.head_s(qs)
        if self.tangential:
            wss = wss - (wss * s_normals).sum(dim=-1, keepdim=True) * s_normals
        out_i = self.head_i(qi)
        vel, press = out_i[:, :3], out_i[:, 3:]
        if self.wall_gate is not None:
            vel = self.wall_gate(vel, interior.wall_dist)
        return wss, vel, press
