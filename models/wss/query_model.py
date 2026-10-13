"""AngioWSSQueryField = geometry encoder (surface -> coarse tokens) + query-point-style
WSS decoder. Mirrors angioflow/models/interior/model.py's AngioInteriorField exactly:
same geometry backbones (reused directly, unmodified), same optional latent-token
pooling. Deliberately structured this way so a future two-branch model (WSS + interior
velocity) can share one geometry backbone with two decoder heads.

    forward(data, waveform) -> (pred [Q, 3], query_idx)

The geometry encoder always sees the FULL mesh (best tokens); only the DECODER's query
count is controllable, via `query_ratio`/`query_max_points` — GPU cost independent of
encoder resolution. Subsampling is active only during training (`self.training`):
`query_idx` is None (== every wall node, Q == N) whenever the model is in eval mode, or
whenever query_ratio>=1 with no cap, so evaluation/inference/sanity-rendering always run
at full resolution and stay directly comparable to the other (always-full-resolution)
baselines. angioflow.training.train's loss/eval code is written to unpack this
(pred, query_idx) tuple return -- see train.py's `_unpack_pred`/`_slice_for_query`.

Plugs into the existing angioflow.training.train registry — see
angioflow/config/models/wss_query_field.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from angioflow.models.interior.encoders import GeoHierGPSEncoder, GeoPointNetEncoder
from angioflow.models.interior.token_pool import GeometryLatentResampler
from angioflow.models.wss.query_decoder import TwoScaleWSSDecoder


def sample_query_idx(batch, ratio, max_points=None, min_points=1):
    """Up to `ratio` fraction of nodes per graph (uniform random, no replacement),
    capped at max_points. Returns None (== "use every node") when ratio>=1 and no cap
    is set, so callers can cheaply skip all subsetting in the common full-resolution case."""
    if ratio >= 1.0 and max_points is None:
        return None
    idx_list = []
    for b in batch.unique(sorted=False).tolist():
        node_idx = (batch == b).nonzero(as_tuple=True)[0]
        k = max(min_points, int(round(node_idx.numel() * ratio)))
        if max_points is not None:
            k = min(k, max_points)
        k = min(k, node_idx.numel())
        if k < node_idx.numel():
            perm = torch.randperm(node_idx.numel(), device=node_idx.device)[:k]
            node_idx = node_idx[perm]
        idx_list.append(node_idx)
    return torch.cat(idx_list)


class AngioWSSQueryField(nn.Module):
    def __init__(self, backbone, token_dim, dim_field=128, n_blocks=2, n_heads=4,
                 n_freqs=6, dim_te=8, dim_wfe=8, local_hops=2, local_heads=4,
                 local_module="gat", geometry_token_pool="none", n_geometry_tokens=64,
                 token_pool_heads=4, token_pool_layers=1, dim_out=3,
                 query_ratio=1.0, query_max_points=None):
        super().__init__()
        self.backbone = backbone
        self.query_ratio = query_ratio
        self.query_max_points = query_max_points
        if geometry_token_pool == "none":
            self.token_pool = None
        elif geometry_token_pool == "latent":
            self.token_pool = GeometryLatentResampler(
                token_dim, n_tokens=n_geometry_tokens, n_heads=token_pool_heads,
                n_layers=token_pool_layers)
        else:
            raise ValueError(
                f"unknown geometry_token_pool '{geometry_token_pool}' (want 'none' or 'latent')")
        self.decoder = TwoScaleWSSDecoder(
            token_dim, dim=dim_field, n_blocks=n_blocks, n_heads=n_heads, n_freqs=n_freqs,
            dim_te=dim_te, dim_wfe=dim_wfe, dim_out=dim_out, local_hops=local_hops,
            local_heads=local_heads, local_module=local_module)
        if hasattr(backbone, "redraw_projection"):          # performer bookkeeping
            self.redraw_projection = backbone.redraw_projection

    def forward(self, data, waveform):
        if self.token_pool is None:
            tokens, token_batch = self.backbone.encode_tokens(data)
        else:
            tokens, token_batch, token_pos = self.backbone.encode_tokens_with_pos(data)
            tokens, token_batch = self.token_pool(tokens, token_pos, token_batch)
        query_idx = (sample_query_idx(data.batch, self.query_ratio, self.query_max_points)
                    if self.training else None)
        pred = self.decoder(data, waveform, tokens, token_batch, query_idx)
        return pred, query_idx


def build_wss_query_field(backbone="hgps", n_cot=8, dim_emb=128, dim_field=128,
                          n_blocks=2, xattn_heads=4, n_freqs=6, dim_te=8, dim_wfe=8,
                          local_hops=2, local_heads=4, local_module="gat",
                          geometry_token_pool="none",
                          n_geometry_tokens=64, token_pool_heads=4, token_pool_layers=1,
                          dim_out=3, query_ratio=1.0, query_max_points=None, **bb_kwargs):
    """Build a WSS baseline. `xattn_heads` are the decoder's global cross-attention
    heads; bb_kwargs go to the geometry backbone (e.g. HierGPS `n_heads` = GPS heads —
    kept distinct)."""
    if backbone == "hgps":
        bb = GeoHierGPSEncoder(n_cot, dim_emb=dim_emb, **bb_kwargs)
    elif backbone == "pointnet":
        bb = GeoPointNetEncoder(dim_emb=dim_emb, **bb_kwargs)
    else:
        raise ValueError(f"unknown backbone '{backbone}' (want 'hgps' or 'pointnet')")
    return AngioWSSQueryField(bb, token_dim=dim_emb, dim_field=dim_field, n_blocks=n_blocks,
                              n_heads=xattn_heads, n_freqs=n_freqs, dim_te=dim_te,
                              dim_wfe=dim_wfe, local_hops=local_hops, local_heads=local_heads,
                              local_module=local_module,
                              geometry_token_pool=geometry_token_pool,
                              n_geometry_tokens=n_geometry_tokens,
                              token_pool_heads=token_pool_heads,
                              token_pool_layers=token_pool_layers, dim_out=dim_out,
                              query_ratio=query_ratio, query_max_points=query_max_points)
