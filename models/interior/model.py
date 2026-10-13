"""Assembly of the interior velocity/pressure field baselines.

AngioInteriorField = geometry encoder (surface → coarse tokens) + two-scale
interior decoder (global cross-attention to tokens + local surface-geometry gather).

    forward(wall, interior, waveform) -> (vel [K,3], press [K,1])

The two baselines share the decoder and differ only in the geometry backbone.
"""
from __future__ import annotations

import torch.nn as nn

from angioflow.models.interior.decoder import TwoScaleInteriorDecoder
from angioflow.models.interior.encoders import GeoHierGPSEncoder, GeoPointNetEncoder
from angioflow.models.interior.token_pool import GeometryLatentResampler


class AngioInteriorField(nn.Module):
    def __init__(self, backbone, token_dim, dim_field=128, n_blocks=2, n_heads=4,
                 k_local=16, n_freqs=6, dim_te=8, dim_wfe=8, local_reduce="mean",
                 geometry_token_pool="none", n_geometry_tokens=64,
                 token_pool_heads=4, token_pool_layers=1, wall_gate=True):
        super().__init__()
        self.backbone = backbone
        if geometry_token_pool == "none":
            self.token_pool = None
        elif geometry_token_pool == "latent":
            self.token_pool = GeometryLatentResampler(
                token_dim, n_tokens=n_geometry_tokens, n_heads=token_pool_heads,
                n_layers=token_pool_layers)
        else:
            raise ValueError(
                f"unknown geometry_token_pool '{geometry_token_pool}' (want 'none' or 'latent')")
        self.decoder = TwoScaleInteriorDecoder(
            token_dim, dim=dim_field, n_blocks=n_blocks, n_heads=n_heads, k_local=k_local,
            n_freqs=n_freqs, dim_te=dim_te, dim_wfe=dim_wfe, local_reduce=local_reduce,
            wall_gate=wall_gate)
        if hasattr(backbone, "redraw_projection"):          # performer bookkeeping
            self.redraw_projection = backbone.redraw_projection

    def forward(self, wall, interior, waveform):
        if self.token_pool is None:
            tokens, token_batch = self.backbone.encode_tokens(wall)
        else:
            tokens, token_batch, token_pos = self.backbone.encode_tokens_with_pos(wall)
            tokens, token_batch = self.token_pool(tokens, token_pos, token_batch)
        return self.decoder(interior, waveform, tokens, token_batch,
                            wall.pos, wall.normals, wall.batch)


def build_interior_field(backbone="hgps", n_cot=8, dim_emb=128, dim_field=128,
                         n_blocks=2, xattn_heads=4, k_local=16, n_freqs=6,
                         dim_te=8, dim_wfe=8, local_reduce="mean",
                         geometry_token_pool="none", n_geometry_tokens=64,
                         token_pool_heads=4, token_pool_layers=1, wall_gate=True,
                         **bb_kwargs):
    """Build a velocity/pressure baseline.

    `xattn_heads` are the decoder's global cross-attention heads; bb_kwargs go to
    the geometry backbone (e.g. HierGPS `n_heads` = GPS heads — kept distinct).
    """
    if backbone == "hgps":
        bb = GeoHierGPSEncoder(n_cot, dim_emb=dim_emb, **bb_kwargs)
    elif backbone == "pointnet":
        bb = GeoPointNetEncoder(dim_emb=dim_emb, **bb_kwargs)
    else:
        raise ValueError(f"unknown backbone '{backbone}' (want 'hgps' or 'pointnet')")
    return AngioInteriorField(bb, token_dim=dim_emb, dim_field=dim_field, n_blocks=n_blocks,
                              n_heads=xattn_heads, k_local=k_local, n_freqs=n_freqs,
                              dim_te=dim_te, dim_wfe=dim_wfe, local_reduce=local_reduce,
                              geometry_token_pool=geometry_token_pool,
                              n_geometry_tokens=n_geometry_tokens,
                              token_pool_heads=token_pool_heads,
                              token_pool_layers=token_pool_layers, wall_gate=wall_gate)
