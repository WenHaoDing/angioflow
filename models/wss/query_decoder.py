"""Query-point-style WSS decoder — mirrors interior/decoder.py's TwoScaleInteriorDecoder.

Each wall point is refined by a stack of fusion blocks that alternate:
  * GLOBAL — cross-attention to the geometry encoder's coarse tokens (bulk-shape context);
  * LOCAL  — GATv2 attention over the point's k-hop mesh neighbourhood (LocalMeshGather),
             for high-frequency detail (WSS hot spots) the coarse token set has no
             mechanism to resolve;
  * FFN.
Unlike the interior decoder's LocalSurfaceGather (interior points gathering from a
DIFFERENT point set, the wall surface), here the query points ARE the wall points, so
the local branch gathers from (a k-hop expansion of) data.edge_index directly — no kNN
reconstruction needed.

Supports predicting only a random SUBSET of the wall's nodes (`query_idx`), so training
can control GPU cost independently of the geometry encoder's resolution: the encoder
always sees the FULL mesh (best tokens), only the DECODER's query count shrinks. A
queried point's local-gather neighbours are NOT restricted to the query set either —
LocalMeshGather's GATv2 layer is bipartite (neighbours need no accumulated/learned state,
just a cheap fresh projection of their own raw pos/normal — see LocalMeshGather), so
neighbours are looked up in the full mesh regardless of which points are queried.

GlobalCrossAttn and fourier_encode are imported, not reimplemented — both are already
generic (dim/heads only; no interior-specific assumptions).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, ModuleList, ReLU, Sequential
from torch_geometric import EdgeIndex
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import coalesce, remove_self_loops, scatter, to_dense_batch, to_undirected

from angioflow.models.interior.decoder import GlobalCrossAttn, fourier_encode
from angioflow.models.transformer import TimeEmbedder, WaveformEmbedder


def k_hop_edges(edge_index, num_nodes, k):
    """Union of edges reachable within 1..k hops of the base (1-hop) mesh edge_index,
    deduplicated (self-loops removed). k=1 returns the symmetrized input, coalesced.
    Verified against a grid graph's exact analytic hop counts (see git history for the
    check): widens LocalMeshGather's receptive field while staying strictly confined to
    REAL mesh connectivity — unlike a spatial kNN, this can never pull in a point that's
    close in 3D but far along the actual surface (e.g. across a thin flow-divider ridge
    at a bifurcation, exactly where WSS hot spots live).

    edge_index is symmetrized first: angioflow.data.mesh.face_to_edge_index stores each
    undirected mesh edge ONCE (i<j only, to halve on-disk size), not both directions —
    confirmed empirically (100% of a real case's edges had no reverse pair present). Every
    hop-composition step below needs the true (both-directions) adjacency, or a node only
    ever gathers from the half of its real neighbours where it happens to be the larger
    index, and the "1-hop" base case would itself already be wrong (degree ~3 instead of
    the true ~6 for a closed triangulated surface)."""
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    if k <= 1:
        return coalesce(edge_index, num_nodes=num_nodes)
    ei = EdgeIndex(edge_index, sparse_size=(num_nodes, num_nodes)).sort_by("row")[0]
    cur = ei
    all_edges = edge_index
    for _ in range(k - 1):
        cur = cur.matmul(ei)[0].as_tensor()
        cur, _ = remove_self_loops(cur)
        all_edges = torch.cat([all_edges, cur], dim=1)
        cur = EdgeIndex(cur, sparse_size=(num_nodes, num_nodes)).sort_by("row")[0]
    return coalesce(all_edges, num_nodes=num_nodes)


def query_local_edges(edge_index, query_idx, n_full):
    """Restrict edge_index to edges whose SOURCE (the point receiving the local-gather
    message) is in query_idx, remapped to query_idx's local [0, Q) indexing. The
    destination (neighbour) index is left as-is — an index into the FULL mesh — so a
    query point can gather from any real mesh neighbour, queried or not."""
    row, col = edge_index
    remap = row.new_full((n_full,), -1)
    remap[query_idx] = torch.arange(query_idx.numel(), device=query_idx.device)
    keep = remap[row] >= 0
    return torch.stack([remap[row[keep]], col[keep]])


class WSSQueryEmbed(nn.Module):
    """Wall-point query -> initial token: Fourier(pos) + normals + time + waveform.

    No wall_dist term (unlike interior's QueryEmbed) — these query points ARE the
    wall, so distance-to-wall is trivially zero. `normals` is included instead: WSS is
    a tangential-traction quantity, so the point's own orientation is directly relevant
    at the per-query level (in addition to whatever normal-informed detail the local
    mesh-gather branch and the geometry tokens separately carry).
    """

    def __init__(self, dim, n_freqs=6, dim_te=8, dim_wfe=8):
        super().__init__()
        self.n_freqs = n_freqs
        self.t_enc = TimeEmbedder(dim_te)
        self.wf_enc = WaveformEmbedder(dim_wfe)
        in_dim = 3 + 6 * n_freqs + 3 + dim_te + dim_wfe
        self.mlp = Sequential(Linear(in_dim, dim), ReLU(), Linear(dim, dim))

    def forward(self, pos, normals, batch, t, waveform):
        t_enc = self.t_enc(t)[batch]
        wf_enc = self.wf_enc(t, waveform)[batch]
        return self.mlp(torch.cat([fourier_encode(pos, self.n_freqs), normals, t_enc, wf_enc], dim=-1))


class LocalMeshGather(nn.Module):
    """Per query point: bipartite GATv2 attention over its k-hop mesh neighbourhood.

    GATv2 (not GATv1): v1's attention is *static* — the ranking of which neighbour gets
    weighted highest doesn't depend on the query, only on a fixed learned combination.
    v2 fixes this with dynamic, query-dependent attention — exactly what "learn which of
    my neighbours matter" needs. Same PyG interface either way (edge_attr, bipartite
    tuples), so this is a free upgrade over v1, not a added-complexity tradeoff.

    Bipartite so neighbours need no accumulated/learned state: a neighbour's GAT "content"
    feature is a cheap, fresh projection of its own raw (pos, normal) (`neighbor_embed`)
    — computed on demand for whichever nodes the (k-hop) edge_index actually references,
    NOT requiring that neighbour to have been processed as a query itself. That's what
    keeps this compatible with query-point subsampling unchanged: a non-queried point can
    still be gathered from, it just never gets its own prediction.

    Relative geometry (offset direction/distance to the query, not absolute position, so
    the local pattern stays translation-consistent) is passed as GATv2's edge_attr, which
    natively folds into both the attention score AND the message — distinct from
    `neighbor_embed`, which is the neighbour's own absolute content feature.
    """

    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.neighbor_embed = Sequential(Linear(6, dim), ReLU(), Linear(dim, dim))
        self.ln_q = LayerNorm(dim)
        # concat=True with out_channels=dim//n_heads: each head computes a SLICE of the
        # width (standard multi-head convention, like GlobalCrossAttn's MultiheadAttention)
        # -- concat=False with out_channels=dim (the first version of this) made every head
        # compute the FULL width before averaging, ~n_heads x more expensive, and OOM'd at
        # full mesh resolution. out_proj guarantees exact `dim` output regardless of whether
        # dim divides evenly by n_heads (heads*(dim//n_heads) can undershoot by a few channels).
        head_dim = max(1, dim // n_heads)
        self.gat = GATv2Conv((dim, dim), head_dim, heads=n_heads, concat=True,
                             add_self_loops=False, edge_dim=8)
        self.out_proj = Linear(head_dim * n_heads, dim)

    def forward(self, q, query_pos, query_normals, neighbor_pos, neighbor_normals, edge_index):
        row, col = edge_index                                # row: local query idx [0,Q); col: FULL-mesh idx
        uniq_col, inv = col.unique(return_inverse=True)        # embed only the neighbours actually referenced
        nb_feat = torch.cat([neighbor_pos[uniq_col], neighbor_normals[uniq_col]], dim=-1)
        x_src = self.neighbor_embed(nb_feat)                    # [n_uniq, dim] -- neighbours' own content
        x_dst = self.ln_q(q)                                     # [Q, dim] -- queries' current state

        dx = neighbor_pos[col] - query_pos[row]
        dist = dx.norm(dim=-1, keepdim=True)
        dn = (dx * query_normals[row]).sum(-1, keepdim=True)
        edge_attr = torch.cat([dx, dist, dn, neighbor_normals[col]], dim=-1)   # [E, 8]

        gat_edge_index = torch.stack([inv, row])                 # source=neighbour(local), target=query
        return self.out_proj(self.gat((x_src, x_dst), gat_edge_index, edge_attr=edge_attr))


class LocalMeshGatherEdgeMLP(nn.Module):
    """Original (pre-GAT) local aggregation: per-edge MLP + mean-pool, no attention.

    Cheaper than LocalMeshGather because the per-edge MLP call is a single dense matmul
    over all edges at once (no attention-score pass, no softmax, no heads multiplier) —
    but it can't weigh neighbours differently depending on the query's own state, and
    every neighbour along an edge is always averaged with equal weight regardless of
    how (ir)relevant it is. Kept as a `local_module="edge_mlp"` option alongside the
    default GAT design for a direct GPU-cost/accuracy comparison, not because it's
    expected to be strictly better or worse.
    """

    def __init__(self, dim):
        super().__init__()
        self.ln_q = LayerNorm(dim)
        self.edge_mlp = Sequential(Linear(dim + 8, dim * 2), ReLU(), Linear(dim * 2, dim))

    def forward(self, q, query_pos, query_normals, neighbor_pos, neighbor_normals, edge_index):
        row, col = edge_index
        dx = neighbor_pos[col] - query_pos[row]
        dist = dx.norm(dim=-1, keepdim=True)
        dn = (dx * query_normals[row]).sum(-1, keepdim=True)
        edge_geom = torch.cat([dx, dist, dn, neighbor_normals[col]], dim=-1)   # [E, 8]
        e = self.edge_mlp(torch.cat([self.ln_q(q)[row], edge_geom], dim=-1))
        return scatter(e, row, dim=0, dim_size=q.size(0), reduce="mean")


class WSSFusionBlock(nn.Module):
    """One refinement step: global cross-attn + local mesh gather + FFN (residual)."""

    def __init__(self, dim, n_heads=4, local_heads=4, local_module="gat"):
        super().__init__()
        self.gattn = GlobalCrossAttn(dim, n_heads)
        if local_module == "gat":
            self.local = LocalMeshGather(dim, n_heads=local_heads)
        elif local_module == "edge_mlp":
            self.local = LocalMeshGatherEdgeMLP(dim)
        else:
            raise ValueError(f"unknown local_module '{local_module}' (want 'gat' or 'edge_mlp')")
        self.ln_f = LayerNorm(dim)
        self.ffn = Sequential(Linear(dim, dim * 2), ReLU(), Linear(dim * 2, dim))

    def forward(self, q, q_batch, kv_dense, kv_mask, query_pos, query_normals,
                neighbor_pos, neighbor_normals, edge_index):
        q = q + self.gattn(q, q_batch, kv_dense, kv_mask)
        q = q + self.local(q, query_pos, query_normals, neighbor_pos, neighbor_normals, edge_index)
        q = q + self.ffn(self.ln_f(q))
        return q


class TwoScaleWSSDecoder(nn.Module):
    """QueryEmbed -> n_blocks x FusionBlock (global + local) -> WSS head + tangential constraint."""

    def __init__(self, token_dim, dim=128, n_blocks=2, n_heads=4, n_freqs=6,
                 dim_te=8, dim_wfe=8, dim_out=3, local_hops=2, local_heads=4,
                 local_module="gat"):
        super().__init__()
        self.local_hops = local_hops
        self.query_embed = WSSQueryEmbed(dim, n_freqs, dim_te, dim_wfe)
        self.token_proj = Linear(token_dim, dim)
        self.blocks = ModuleList([
            WSSFusionBlock(dim, n_heads, local_heads, local_module) for _ in range(n_blocks)])
        self.head = Sequential(Linear(dim, dim // 2), ReLU(), Linear(dim // 2, dim_out))

    def forward(self, data, waveform, tokens, token_batch, query_idx=None):
        # k-hop expansion happens ONCE on the full mesh (real topology, no wall/branch-
        # crossing risk), THEN restricted to the query subset if subsampling is active —
        # reused across all blocks (static per forward call, doesn't depend on q).
        expanded_ei = k_hop_edges(data.edge_index, data.num_nodes, self.local_hops)

        if query_idx is None:
            q_pos, q_normals, q_batch = data.pos, data.normals, data.batch
            local_ei = expanded_ei
        else:
            q_pos, q_normals, q_batch = data.pos[query_idx], data.normals[query_idx], data.batch[query_idx]
            local_ei = query_local_edges(expanded_ei, query_idx, data.num_nodes)

        q = self.query_embed(q_pos, q_normals, q_batch, data.t, waveform)
        kv_dense, kv_mask = to_dense_batch(self.token_proj(tokens), token_batch)
        for blk in self.blocks:
            q = blk(q, q_batch, kv_dense, kv_mask, q_pos, q_normals, data.pos, data.normals, local_ei)
        pred = self.head(q)
        return self.tangential_constraint(pred, q_normals)

    def tangential_constraint(self, pred, normals):
        """Hard physics constraint: WSS is the tangential traction, so wss . n = 0. Same
        mechanism as AngioHierarchicalGPS.tangential_constraint (kept identical so
        train.py's loss/eval code needs no changes for this model)."""
        return pred - (pred * normals).sum(dim=-1, keepdim=True) * normals
