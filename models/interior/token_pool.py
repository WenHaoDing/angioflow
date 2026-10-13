"""Fixed-budget pooling of variable-size surface-geometry token sets."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class _LatentPoolBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.latent_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, latents, source, source_mask):
        update, _ = self.cross_attn(
            self.latent_norm(latents), self.source_norm(source), self.source_norm(source),
            key_padding_mask=~source_mask, need_weights=False)
        latents = latents + update
        update, _ = self.self_attn(
            self.self_norm(latents), self.self_norm(latents), self.self_norm(latents),
            need_weights=False)
        latents = latents + update
        return latents + self.ffn(self.ffn_norm(latents))


class GeometryLatentResampler(nn.Module):
    """Compress graph-derived features into a fixed number of abstract tokens.

    Learned latent queries cross-attend to variable-length surface features. A
    positional MLP is added to the source features so the resampler can retain
    where geometry occurs, rather than only which patterns are present.
    """

    def __init__(self, dim: int, n_tokens: int = 64, n_heads: int = 4,
                 n_layers: int = 1):
        super().__init__()
        if n_tokens < 1:
            raise ValueError("n_tokens must be positive")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        if dim % n_heads:
            raise ValueError(f"token dimension {dim} must be divisible by {n_heads} heads")
        self.n_tokens = n_tokens
        self.latents = nn.Parameter(torch.empty(n_tokens, dim))
        nn.init.normal_(self.latents, std=1.0 / math.sqrt(dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [_LatentPoolBlock(dim, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens, token_pos, token_batch):
        if tokens.size(0) != token_pos.size(0) or tokens.size(0) != token_batch.size(0):
            raise ValueError("tokens, token_pos and token_batch must have equal length")
        source, source_mask = to_dense_batch(
            tokens + self.pos_embed(token_pos), token_batch)
        batch_size = source.size(0)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        for block in self.blocks:
            latents = block(latents, source, source_mask)
        latents = self.out_norm(latents)
        pooled_batch = torch.arange(batch_size, device=tokens.device).repeat_interleave(
            self.n_tokens)
        return latents.reshape(batch_size * self.n_tokens, -1), pooled_batch
