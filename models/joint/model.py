"""AngioJointField = one shared HierGPS geometry encoder + joint two-field decoder.

    forward(wall, interior, waveform) -> (wss, query_idx, vel, press)

wss covers the wall's WSS query set — a random `query_ratio` subset in BOTH train and
eval (query_idx is None only at query_ratio>=1, i.e. full resolution); vel/press cover
every interior point in the batch (the interior subsample is a dataset transform,
matching train_interior.py).

The encoder runs ONCE per case and feeds both heads — the point of the joint model.
"""
from __future__ import annotations

import torch.nn as nn

from angioflow.models.interior.encoders import GeoHierGPSEncoder
from angioflow.models.interior.token_pool import GeometryLatentResampler
from angioflow.models.joint.decoder import JointTwoFieldDecoder
from angioflow.models.wss.query_model import sample_query_idx


class AngioJointField(nn.Module):
    def __init__(self, backbone, token_dim, dim_field=128, n_blocks=2, n_heads=4,
                 local_heads=4, k_local=16, k_field=16, cross_field=True,
                 local_hops=2, n_freqs=6, dim_te=8, dim_wfe=8,
                 geometry_token_pool="none", n_geometry_tokens=64,
                 token_pool_heads=4, token_pool_layers=1,
                 query_ratio=1.0, query_max_points=None, tangential=True, wall_gate=False):
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
        self.decoder = JointTwoFieldDecoder(
            token_dim, dim=dim_field, n_blocks=n_blocks, n_heads=n_heads,
            local_heads=local_heads, k_local=k_local, k_field=k_field,
            cross_field=cross_field, local_hops=local_hops, n_freqs=n_freqs,
            dim_te=dim_te, dim_wfe=dim_wfe,
            tangential=tangential, wall_gate=wall_gate)
        if hasattr(backbone, "redraw_projection"):          # performer bookkeeping
            self.redraw_projection = backbone.redraw_projection

    def forward(self, wall, interior, waveform):
        if self.token_pool is None:
            tokens, token_batch = self.backbone.encode_tokens(wall)
        else:
            tokens, token_batch, token_pos = self.backbone.encode_tokens_with_pos(wall)
            tokens, token_batch = self.token_pool(tokens, token_pos, token_batch)
        # unlike AngioWSSQueryField, subsampling applies in BOTH train and eval: eval
        # at full wall resolution costs 1/query_ratio x the decoder memory for a more
        # precise metric estimate that isn't worth the GPU (the subset estimate is
        # unbiased). Full resolution when query_ratio>=1, or by temporarily setting
        # query_ratio=1.0 (see train_joint.save_joint_sanity, which needs every node).
        query_idx = sample_query_idx(wall.batch, self.query_ratio, self.query_max_points)
        wss, vel, press = self.decoder(wall, interior, waveform, tokens, token_batch, query_idx)
        return wss, query_idx, vel, press


def build_joint_field(n_cot=8, dim_emb=128, dim_field=128, n_blocks=2, xattn_heads=4,
                      local_heads=4, k_local=16, k_field=16, cross_field=True,
                      local_hops=2, n_freqs=6,
                      dim_te=8, dim_wfe=8, geometry_token_pool="none",
                      n_geometry_tokens=64, token_pool_heads=4, token_pool_layers=1,
                      query_ratio=1.0, query_max_points=None, tangential=True,
                      wall_gate=False, **bb_kwargs):
    """HierGPS-backed joint model (the geometry backbone is fixed to HierGPS by
    design). bb_kwargs go to GeoHierGPSEncoder (n_pre, n_gps, n_pool, pool_ratio, ...)."""
    bb = GeoHierGPSEncoder(n_cot, dim_emb=dim_emb, **bb_kwargs)
    return AngioJointField(bb, token_dim=dim_emb, dim_field=dim_field, n_blocks=n_blocks,
                           n_heads=xattn_heads, local_heads=local_heads, k_local=k_local,
                           k_field=k_field, cross_field=cross_field,
                           local_hops=local_hops, n_freqs=n_freqs, dim_te=dim_te,
                           dim_wfe=dim_wfe, geometry_token_pool=geometry_token_pool,
                           n_geometry_tokens=n_geometry_tokens,
                           token_pool_heads=token_pool_heads,
                           token_pool_layers=token_pool_layers, query_ratio=query_ratio,
                           query_max_points=query_max_points, tangential=tangential,
                           wall_gate=wall_gate)
