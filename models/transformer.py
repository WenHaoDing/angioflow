"""Graph-transformer (GPS) for per-node WSS on variable-size cropped meshes.

Adapted from the fixed-topology GraphGPS: GHD positional encoding is dropped, the
cotangent-Laplacian eigenpairs are used as genuine per-node PE, and all graph-level
broadcasting is done by `[data.batch]` indexing so graphs may have different N.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch_geometric.nn import GINEConv, GPSConv
from torch_geometric.nn.attention import PerformerAttention


class TimeEmbedder(nn.Module):
    """Sinusoidal frame-index embedding -> [B, dim_te]."""

    def __init__(self, dim_te=8):
        super().__init__()
        self.dim_te = dim_te
        self.mlp = Sequential(Linear(dim_te, 2 * dim_te), ReLU(), Linear(2 * dim_te, dim_te))

    def forward(self, t, max_period=100):
        t = t.view(-1).float()                                  # [B]
        half = self.dim_te // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim_te % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)                                    # [B, dim_te]


class WaveformEmbedder(nn.Module):
    """Encode the (shared) inflow waveform, sample it at each frame -> [B, dim_wfe]."""

    def __init__(self, dim_wfe=8, kernel_size=32, T=80):
        super().__init__()
        self.T = T
        self.bn = nn.BatchNorm1d(3)
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size, stride=2, padding=kernel_size // 2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size, stride=2, padding=kernel_size // 2), nn.ReLU(),
            nn.Conv1d(64, dim_wfe, kernel_size, stride=2, padding=kernel_size // 2), nn.ReLU(),
        )

    def _deriv(self, wf):                                       # wf [1, 1, W]
        dt = 0.8 / wf.shape[2]
        v = F.pad(torch.diff(wf, dim=2) / dt, (0, 1), mode="replicate")
        a = F.pad(torch.diff(v, dim=2) / dt, (0, 1), mode="replicate")
        return torch.cat([wf, v, a], dim=1)                     # [1, 3, W]

    def forward(self, t, waveform):                            # waveform [1, W, 1]
        assert waveform.shape[0] == 1, "single shared waveform expected"
        cond = self.conv(self.bn(self._deriv(waveform.permute(0, 2, 1))))
        cond = F.interpolate(cond, size=self.T, mode="linear", align_corners=True).transpose(1, 2)  # [1, T, dim_wfe]
        t = t.view(-1)
        mask = (t >= 0) & (t < self.T)
        enc = cond[0, t.clamp(0, self.T - 1), :]                # [B, dim_wfe]
        enc = enc.clone()
        enc[~mask] = 0.0
        return enc


class CotPE(nn.Module):
    """Per-node positional encoding from cotangent-Laplacian eigenpairs.

    Args:
        use_mlp: if True, replace the Transformer encoder with a shared MLP applied
                 independently per eigenmode (faster, no cross-mode attention).
    """

    def __init__(self, n_cot, dim_pe=32, n_heads=4, n_layers=4, use_mlp=False):
        super().__init__()
        self.use_mlp = use_mlp
        self.bn = nn.BatchNorm1d(n_cot)
        self.proj = nn.Linear(2, dim_pe)                        # (eigvec, eigval) -> dim_pe
        if use_mlp:
            layers = []
            for _ in range(n_layers):
                layers += [nn.Linear(dim_pe, dim_pe * 4), nn.ReLU(),
                           nn.Linear(dim_pe * 4, dim_pe), nn.ReLU()]
            self.encoder = nn.Sequential(*layers)
        else:
            layer = nn.TransformerEncoderLayer(d_model=dim_pe, nhead=n_heads,
                                               dim_feedforward=dim_pe * 4, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, eigvec, eigval, training):               # eigvec/eigval [N, k]
        if training:                                            # random sign flip per mode
            sign = torch.randint(0, 2, (1, eigvec.size(1)), device=eigvec.device) * 2 - 1
            eigvec = eigvec * sign
        lap = self.bn(torch.stack([eigvec, eigval], dim=-1))    # [N, k, 2]
        x = self.proj(lap)                                      # [N, k, dim_pe]
        if self.use_mlp:
            N, k, d = x.shape
            x = self.encoder(x.reshape(N * k, d)).reshape(N, k, d)
        else:
            x = self.encoder(x)
        return x.sum(dim=1)                                     # [N, dim_pe]


class NodeEncoder(nn.Module):
    """Fuse node features (pos, normals, geo_dist), time, waveform,
    and cotangent PE -> [N, dim_emb]."""

    def __init__(self, n_cot, node_dim_in=6, n_geo_dist=3, dim_pe=32, dim_te=8, dim_wfe=8,
                 dim_emb=64, pe_n_heads=4, pe_n_layers=4, pe_use_mlp=False):
        super().__init__()
        self.cot_pe = CotPE(n_cot, dim_pe=dim_pe, n_heads=pe_n_heads, n_layers=pe_n_layers,
                            use_mlp=pe_use_mlp)
        self.t_enc = TimeEmbedder(dim_te)
        self.wf_enc = WaveformEmbedder(dim_wfe)
        self.lin = nn.Linear(node_dim_in + n_geo_dist + dim_te + dim_wfe, dim_emb - dim_pe)

    def forward(self, data, waveform):
        B = int(data.batch.max()) + 1
        k = data.cot_eigvec.size(1)
        eigval = data.cot_eigval.view(B, k)[data.batch]         # [N, k], per-graph broadcast
        pe = self.cot_pe(data.cot_eigvec, eigval, self.training)  # [N, dim_pe]

        geo_dist = data.geo_dist.nan_to_num(nan=0.0, posinf=1.0)
        x = torch.cat([data.pos, data.normals, geo_dist], dim=-1)  # [N, node_dim_in + n_geo_dist]
        t_enc = self.t_enc(data.t)[data.batch]                  # [N, dim_te]
        wf_enc = self.wf_enc(data.t, waveform)[data.batch]      # [N, dim_wfe]
        h = self.lin(torch.cat([x, t_enc, wf_enc], dim=-1))     # [N, dim_emb - dim_pe]
        return torch.cat([h, pe], dim=-1)                       # [N, dim_emb]


class EdgeEncoder(nn.Module):
    """Edge features: (pos, normals) Euclidean gradients + cotangent Laplacian
    eigenvector gradients -> [E, dim_emb].

    Euclidean:  |Δ(pos,normals)| and centre  -> Linear -> euclidean_dim_emb
    Cot:        BN + |Δ(cot_eigvec)|          -> Linear -> cot_dim_emb
    Fuse:       cat(euclidean, cot)            -> Linear -> dim_emb
    """

    def __init__(self, n_cot, euc_dim=6, enc_centre=True,
                 euclidean_dim_emb=32, cot_dim_emb=16, dim_emb=64,
                 edge_norm_dict=None):
        super().__init__()
        self.enc_centre = enc_centre
        self.cot_bn = nn.BatchNorm1d(n_cot)
        euc_in = euc_dim * 2 if enc_centre else euc_dim
        self.euclidean_encoder = nn.Linear(euc_in, euclidean_dim_emb)
        self.cot_encoder = nn.Linear(n_cot, cot_dim_emb)
        self.fuse = nn.Linear(euclidean_dim_emb + cot_dim_emb, dim_emb)
        if edge_norm_dict is not None:
            self.register_buffer("edge_mean", edge_norm_dict["mean"], persistent=False)
            self.register_buffer("edge_std", edge_norm_dict["std"], persistent=False)

    def forward(self, data):
        x = torch.cat([data.pos, data.normals], dim=-1)  # [N, 6]
        i, j = data.edge_index
        grad = torch.abs(x[i] - x[j])
        if self.enc_centre:
            euc_attr = torch.cat([grad, (x[i] + x[j]) / 2.0], dim=-1)
        else:
            euc_attr = grad
        if hasattr(self, "edge_mean"):
            euc_attr = (euc_attr - self.edge_mean) / (self.edge_std + 1e-6)
        euc_emb = self.euclidean_encoder(euc_attr)                       # [E, euclidean_dim_emb]
        cot = self.cot_bn(data.cot_eigvec)                               # [N, n_cot]
        cot_emb = self.cot_encoder(torch.abs(cot[i] - cot[j]))          # [E, cot_dim_emb]
        return self.fuse(torch.cat([euc_emb, cot_emb], dim=-1))         # [E, dim_emb]


class AngioGraphGPS(nn.Module):
    """GPS graph transformer predicting per-node WSS on variable-size meshes."""

    def __init__(self, n_cot=16, node_dim_in=6, n_geo_dist=3, dim_pe=32, dim_te=8, dim_wfe=8,
                 dim_emb=64, n_heads=4, num_layers=6, dim_out=3,
                 attn_type="performer", attn_kwargs: Dict[str, Any] = None,
                 pe_n_heads=4, pe_n_layers=4, pe_use_mlp=False, edge_norm_dict=None):
        super().__init__()
        self.node_encoder = NodeEncoder(n_cot, node_dim_in=node_dim_in, n_geo_dist=n_geo_dist,
                                        dim_pe=dim_pe, dim_te=dim_te, dim_wfe=dim_wfe, dim_emb=dim_emb,
                                        pe_n_heads=pe_n_heads, pe_n_layers=pe_n_layers,
                                        pe_use_mlp=pe_use_mlp)
        self.edge_encoder = EdgeEncoder(n_cot, dim_emb=dim_emb, edge_norm_dict=edge_norm_dict)
        self.convs = ModuleList()
        for _ in range(num_layers):
            mlp = Sequential(Linear(dim_emb, dim_emb), ReLU(), Linear(dim_emb, dim_emb))
            self.convs.append(GPSConv(dim_emb, GINEConv(mlp), heads=n_heads,
                                      attn_type=attn_type, attn_kwargs=attn_kwargs))
        self.mlp = Sequential(
            Linear(dim_emb, dim_emb // 2), ReLU(),
            Linear(dim_emb // 2, dim_emb // 4), ReLU(),
            Linear(dim_emb // 4, dim_out),
        )
        self.redraw_projection = RedrawProjection(
            self.convs, redraw_interval=1000 if attn_type == "performer" else None)

    def forward(self, data, waveform):
        edge_attr = self.edge_encoder(data)                     # [E, dim_emb]
        x = self.node_encoder(data, waveform)                   # [N, dim_emb]
        for conv in self.convs:
            x = conv(x, data.edge_index, data.batch, edge_attr=edge_attr)
        return self.mlp(x)                                      # [N, dim_out]


class RedrawProjection:
    """Periodically redraw Performer projection matrices during training."""

    def __init__(self, model: nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            for m in self.model.modules():
                if isinstance(m, PerformerAttention):
                    m.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
