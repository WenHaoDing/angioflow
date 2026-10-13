"""AngioHierarchicalGPS: multi-level GNN encoder → GPS transformer → multi-level decoder.

Down path:  full-res pre-convs → [FPS pool → inter-convs] × n_pool → GPS (coarsest)
Up path:    GPS → [unpool + skip → inter-convs] × n_pool → full-res post-convs → head

pool_ratio is applied at every level, so total compression = pool_ratio ** n_pool.
For a 2× larger graph with n_pool=2, pool_ratio=0.5 keeps the GPS graph the same
absolute size as the original n_pool=1, pool_ratio=0.25 setting.

Design notes (encoder vs. decoder aggregation):
  - GINEConv (sum aggregation) is used only where discriminative power matters:
    building the full-res encoding (pre_convs) and the coarse representation
    (inter_down_convs, GPS's internal local-conv branch). Sum aggregation is not
    normalised by neighbour count, so on a *free* (non-registered) mesh with
    variable local triangle density, it makes predictions depend on discretisation
    density — not just on the underlying field. That's fine for building features,
    but wrong for reconstructing a spatially smooth field.
  - SAGEConv (mean aggregation) is used for reconstruction: inter_up_convs and
    post_convs. Mean aggregation is degree-invariant, so the decoder's smoothing
    behaviour doesn't depend on local mesh density — this is what actually fixes
    the "blocky" full-res output (knn_interpolate alone only fixes the *unpool*
    step; the post-unpool full-res conv stack can still reintroduce density-
    dependent artefacts).

Pooling: FPS (farthest-point sampling) instead of SAGPooling. FPS gives
deterministic, uniform spatial coverage of the coarse level — no learned score
that can concentrate coarse nodes unevenly (e.g. all in the aneurysm, starving
the rest of the surface of global context). Since FPS explicitly maximises
inter-point distance, the original mesh edges essentially never connect two
FPS-selected nodes, so the coarse-level graph is rebuilt via a plain kNN graph
on the sampled positions (no learned edge features at this synthetic level —
edge_attr is zeros, GINEConv there is just sum aggregation over the kNN graph).

bidirectional (default True): a kNN graph is inherently directed (A being B's
nearest point doesn't make B A's nearest point) — confirmed empirically (~30%
of edges in a typical kNN graph have no reverse pair). Left un-symmetrized,
inter_down_convs/gps_convs/inter_up_convs at this coarse level would each only
aggregate from about half of a node's true nearest neighbours. This only covers
the coarse-level graph rebuilt HERE, inside forward() (it doesn't exist until FPS
pooling runs, so a pre-processing transform can't reach it) — the full-resolution
data.edge_index (pre_convs, post_convs, final_smooth) is expected to already be
symmetric via the angioflow.transform.SymmetrizeEdges dataset transform.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GATConv, GINEConv, GPSConv, MLP, SAGEConv, fps, knn_graph, knn_interpolate
from torch_geometric.utils import to_undirected

from angioflow.models.transformer import EdgeEncoder, NodeEncoder, RedrawProjection


class AngioHierarchicalGPS(nn.Module):
    def __init__(
        self,
        n_cot: int,
        node_dim_in: int = 6,
        n_geo_dist: int = 3,
        dim_pe: int = 16,
        dim_te: int = 8,
        dim_wfe: int = 8,
        dim_emb: int = 128,
        n_pre: int = 2,
        n_gps: int = 4,
        n_post: int = 2,
        n_pool: int = 1,
        n_inter: int = 0,
        pool_ratio: float = 0.25,
        pool_knn_k: int = 6,
        n_heads: int = 4,
        dim_out: int = 3,
        pe_n_heads: int = 2,
        pe_n_layers: int = 2,
        pe_use_mlp: bool = True,
        edge_norm_dict=None,
        attn_type: str = "performer",
        grad_ckpt: bool = False,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.bidirectional = bidirectional
        self.n_pool = n_pool
        self.pool_ratio = pool_ratio
        self.pool_knn_k = pool_knn_k
        self.dim_emb = dim_emb

        self.node_encoder = NodeEncoder(
            n_cot,
            node_dim_in=node_dim_in,
            n_geo_dist=n_geo_dist,
            dim_pe=dim_pe,
            dim_te=dim_te,
            dim_wfe=dim_wfe,
            dim_emb=dim_emb,
            pe_n_heads=pe_n_heads,
            pe_n_layers=pe_n_layers,
            pe_use_mlp=pe_use_mlp,
        )
        self.edge_encoder = EdgeEncoder(
            n_cot, dim_emb=dim_emb, edge_norm_dict=edge_norm_dict,
        )

        def _gine():
            return GINEConv(MLP([dim_emb, dim_emb * 2, dim_emb], act="relu"), edge_dim=dim_emb)

        def _sage():
            return SAGEConv(dim_emb, dim_emb, aggr="mean")

        # Full-resolution encoder (sum-aggregation: discriminative feature building)
        self.pre_convs = nn.ModuleList([_gine() for _ in range(n_pre)])
        self.pre_norms = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(n_pre)])

        # Multi-level FPS pool + intermediate convs (down path: after each pool).
        # FPS has no learnable parameters, so there's nothing to store per level
        # besides the (shared) inter-convs.
        self.inter_down_convs = nn.ModuleList([
            nn.ModuleList([_gine() for _ in range(n_inter)]) for _ in range(n_pool)
        ])
        self.inter_down_norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(n_inter)]) for _ in range(n_pool)
        ])

        # GPS transformer at coarsest level
        self.gps_convs = nn.ModuleList()
        for _ in range(n_gps):
            mlp = Sequential(Linear(dim_emb, dim_emb), ReLU(), Linear(dim_emb, dim_emb))
            self.gps_convs.append(
                GPSConv(dim_emb, GINEConv(mlp), heads=n_heads, attn_type=attn_type)
            )

        # Multi-level unpool + intermediate convs (up path: after each unpool).
        # Mean aggregation: reconstruction should smooth, not discriminate.
        self.inter_up_convs = nn.ModuleList([
            nn.ModuleList([_sage() for _ in range(n_inter)]) for _ in range(n_pool)
        ])
        self.inter_up_norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(n_inter)]) for _ in range(n_pool)
        ])

        # Full-resolution decoder (mean aggregation — see module docstring)
        self.post_convs = nn.ModuleList([_sage() for _ in range(n_post)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(n_post)])

        self.head = Sequential(
            Linear(dim_emb, dim_emb // 2), ReLU(),
            Linear(dim_emb // 2, dim_out),
        )
        # Final graph-smoothing layer: attention-weighted (not fixed-mean) aggregation
        # over the mesh. A plain mean here (the old SAGEConv) applies the same uniform
        # blur everywhere, which — stacked on top of the already mean-aggregating
        # decoder above — over-smooths real high-frequency signal (WSS hot spots).
        # GATConv learns per-edge attention weights, so it can attend mostly to itself
        # (near-identity) where the local field is genuinely peaked, and spread out
        # where it's flat, instead of a fixed blur radius everywhere.
        self.final_smooth = GATConv(dim_out, dim_out, heads=1, concat=False)

        self.redraw_projection = RedrawProjection(
            self.gps_convs,
            redraw_interval=1000 if attn_type == "performer" else None,
        )

    def forward(self, data, waveform):
        x = self.node_encoder(data, waveform)   # [N, dim_emb]
        edge_attr = self.edge_encoder(data)      # [E, dim_emb]

        # Full-resolution encoder
        for conv, norm in zip(self.pre_convs, self.pre_norms):
            x = norm(x + conv(x, data.edge_index, edge_attr))

        # Multi-level down-sampling via FPS.
        # per-level state (index 0 = full-res; pool appends level k+1, a node subset of level k)
        all_pos = [data.pos]
        all_batches = [data.batch]
        # all_eis[lvl] is the edge_index actually valid for all_pos[lvl]'s node set:
        # level 0 is the real full-res mesh; levels 1..n_pool are the synthetic kNN
        # graphs rebuilt after each FPS pool (needed by inter_up_convs at that level).
        all_eis = [data.edge_index]
        x_skips = []
        cur_batch = data.batch

        for down_convs, down_norms in zip(self.inter_down_convs, self.inter_down_norms):
            x_skips.append(x)
            perm = fps(all_pos[-1], cur_batch, ratio=self.pool_ratio)
            x = x[perm]
            pos_coarse = all_pos[-1][perm]
            cur_batch = cur_batch[perm]
            all_pos.append(pos_coarse)
            all_batches.append(cur_batch)

            # FPS maximises inter-point distance, so original mesh edges essentially
            # never connect two selected nodes — rebuild a plain kNN graph instead.
            # No real edge features exist at this synthetic level; zeros are fine
            # for GINEConv (it degenerates to sum-aggregation over the kNN graph).
            cur_ei = knn_graph(pos_coarse, k=self.pool_knn_k, batch=cur_batch, loop=False)
            if self.bidirectional:
                cur_ei = to_undirected(cur_ei, num_nodes=pos_coarse.size(0))
            cur_ea = x.new_zeros(cur_ei.size(1), self.dim_emb)
            all_eis.append(cur_ei)

            for conv, norm in zip(down_convs, down_norms):
                x = norm(x + conv(x, cur_ei, cur_ea))

        # GPS at coarsest level
        for conv in self.gps_convs:
            if self.grad_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(conv, x, cur_ei, cur_batch,
                               edge_attr=cur_ea, use_reentrant=False)
            else:
                x = conv(x, cur_ei, cur_batch, edge_attr=cur_ea)

        # Multi-level up-sampling (reverse order).
        # PointNet++ FP-style unpooling: every fine node gets an inverse-distance
        # blend of its 3 nearest coarse nodes. Unlike perm-scatter (zeros at
        # non-selected nodes), coarse/global information covers the whole surface
        # smoothly — no Voronoi-block artefacts.
        for lvl in reversed(range(self.n_pool)):
            x_up = knn_interpolate(x, all_pos[lvl + 1], all_pos[lvl],
                                   batch_x=all_batches[lvl + 1],
                                   batch_y=all_batches[lvl], k=3)
            x = x_up + x_skips[lvl]
            for conv, norm in zip(self.inter_up_convs[lvl], self.inter_up_norms[lvl]):
                x = norm(x + conv(x, all_eis[lvl]))

        # Full-resolution decoder
        for conv, norm in zip(self.post_convs, self.post_norms):
            x = norm(x + conv(x, data.edge_index))

        pred = self.head(x)                                # [N, dim_out]
        pred = self.final_smooth(pred, data.edge_index)    # adaptive local smoothing

        return self.tangential_constraint(pred, data)

    def tangential_constraint(self, pred, data):
        """Hard physics constraint: WSS is the tangential traction, so wss · n = 0 —
        project out data.normals (rotated consistently with pos/y; commutes with the
        isotropic zero-mean z-scoring, see angioflow.data.normalize)."""
        n = data.normals
        return pred - (pred * n).sum(dim=-1, keepdim=True) * n
