"""Physics constraints as reusable modules (usable by any interior field model)."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import fps
from torch_geometric.utils import scatter


class WallNoSlipGate(nn.Module):
    """Hard no-slip constraint for a vector field via distance to the wall.

    Multiplies a raw vector field by a gate g(d) = 1 - exp(-d / tau) of the
    (normalised) wall distance d, so the field is forced to 0 at the wall (d -> 0,
    no-slip) and relaxes to the raw prediction in the core (d >> tau). g is linear
    near the wall (g ≈ d/tau), matching the boundary-layer velocity ramp
    ∂u/∂n = WSS/μ. tau — the boundary-layer thickness in the same units as the
    wall distance the model is fed — is learned (softplus of a log-space
    parameter, so it stays positive).

    Usage:
        self.gate = WallNoSlipGate()
        vel = self.gate(vel, interior.wall_dist)   # [K, 3] * [K, 1]
    """

    def __init__(self, tau0: float = 0.15):
        super().__init__()
        # store softplus^{-1}(tau0) so softplus(log_tau) == tau0 at init
        inv = math.log(math.expm1(tau0)) if tau0 > 0 else -5.0
        self.log_tau = nn.Parameter(torch.tensor(float(inv)))

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self.log_tau)

    def gate(self, wall_dist: torch.Tensor) -> torch.Tensor:
        """Return the per-point scalar gate g(d) in [0, 1), shape [..., 1]."""
        wd = wall_dist.unsqueeze(-1) if wall_dist.dim() == 1 else wall_dist
        return 1.0 - torch.exp(-wd / self.tau)

    def forward(self, vec: torch.Tensor, wall_dist: torch.Tensor) -> torch.Tensor:
        return vec * self.gate(wall_dist)


@dataclass(frozen=True)
class GradientStencil:
    """A fixed per-point weighted-least-squares gradient operator.

    For center i with neighbours j, grad(f)_i = sum_j coeff[e] * (f_j - f_i). Built
    once from geometry (pos, edge_index) — never from a predicted field — so it is
    a plain linear map: differentiating a field through it is a single scatter-add,
    with no solve or eigendecomposition in the backward graph.
    """
    edge_index: torch.Tensor   # [2, E] directed center -> neighbour, self-loops removed
    coeff: torch.Tensor        # [E, 3]
    valid: torch.Tensor        # [N] bool, False where the local stencil is unusable


def build_gradient_stencil(pos, edge_index, ridge=3e-2, power=2.0, min_neighbors=3, cond_tol=1e-6):
    """Fit grad(f)_i by local Taylor expansion: f_j - f_i ~= grad(f)_i . (x_j - x_i).

    Weighted least squares over the neighbours of i, weight = |x_j - x_i|^-power.
    Coordinates are rescaled by each point's local RMS neighbour distance so `ridge`
    is dimensionless; a point is `valid` only if it has >= min_neighbors and its 3x3
    normal-equations matrix isn't near-singular (eigenvalue ratio > cond_tol), e.g.
    a near-planar or sparse stencil.
    """
    center, neighbor = edge_index
    keep = center != neighbor          # a self edge has dx=0 and blows up the 1/distance weight
    center, neighbor = center[keep], neighbor[keep]
    n = pos.size(0)

    with torch.no_grad():
        dx = pos[neighbor] - pos[center]
        degree = torch.bincount(center, minlength=n)
        scale = (scatter(dx.square().sum(-1), center, dim_size=n, reduce="sum")
                / degree.clamp_min(1)).sqrt().clamp_min(1e-12)

        x = dx / scale[center, None]
        weight = x.norm(dim=-1).clamp_min(1e-12).pow(-power)
        wx = weight[:, None] * x

        gram = scatter(wx[:, :, None] * x[:, None, :], center, dim_size=n, reduce="sum")
        eigenvalues = torch.linalg.eigvalsh(gram)
        condition = eigenvalues[:, 0] / eigenvalues[:, -1].clamp_min(1e-12)
        valid = (degree >= min_neighbors) & (eigenvalues[:, -1] > 0) & (condition > cond_tol)

        eye = torch.eye(3, device=pos.device, dtype=pos.dtype).expand(n, 3, 3)
        gram = torch.where(valid[:, None, None], gram + ridge * eye, eye)
        inverse = torch.linalg.solve(gram, eye)
        coeff = torch.einsum("nij,nj->ni", inverse[center], wx) / scale[center, None]
        coeff = coeff * valid[center, None]

    return GradientStencil(edge_index=torch.stack([center, neighbor]), coeff=coeff, valid=valid)


def apply_gradient_stencil(field, stencil: GradientStencil):
    """grad(field): [N] -> [N, 3], or [N, C] -> [N, C, 3]. Differentiable in `field`."""
    is_scalar = field.dim() == 1
    f = field[:, None] if is_scalar else field
    center, neighbor = stencil.edge_index
    df = f[neighbor] - f[center]                                        # [E, C]
    contribution = stencil.coeff[:, :, None] * df[:, None, :]            # [E, 3, C]
    grad = scatter(contribution, center, dim=0, dim_size=f.size(0), reduce="sum").transpose(1, 2)
    return grad[:, 0] if is_scalar else grad                             # [N, 3] / [N, C, 3]


def continuity_residual(vel, pos, edge_index, ridge=3e-2, power=2.0, min_neighbors=3, cond_tol=1e-6):
    """Incompressible continuity div(v) = tr(grad v) at each point, plus the stencil's
    validity mask. `pos`/`vel` may be in any consistently-scaled frame (e.g. the
    isotropically normalized training frame): rescaling both by a scalar leaves
    div(v) proportional to its physical value, so the zero-residual constraint is
    unaffected by normalization.
    """
    stencil = build_gradient_stencil(pos, edge_index, ridge, power, min_neighbors, cond_tol)
    grad_v = apply_gradient_stencil(vel, stencil)                        # [N, 3, 3]
    return grad_v.diagonal(dim1=1, dim2=2).sum(-1), stencil.valid


# ---------------------------------------------------------------------------
# Synthetic-tetrahedron continuity: an alternative to the graph stencil above.
#
# The graph stencil differentiates the field with whatever local point density the
# training-time pool subsample happens to provide — often sparse (degree ~17 at the
# default sample_ratio=0.25) and never denser than the precomputed on-disk pool graph.
# Since interior query points are independent in AngioInteriorField's decoder (no
# interior-interior coupling — see decoder.py), the model can just as well be queried
# at brand-new positions that were never part of any graph: K freshly-generated offset
# points around a chosen center, forming a tetrahedron-like "element", give a
# least-squares local-gradient estimate exactly analogous to Green-Gauss reconstruction
# (identical to it for K=4, more noise-averaging redundancy for K>4). An empirical study
# over 10 real CFD cases (angioflow/data/gradient_estimation/) found this beats the graph
# stencil once K is large enough (consistently by K=24, always by K=64).
# ---------------------------------------------------------------------------

def _regular_tetrahedron_directions(device=None, dtype=torch.float32):
    d = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                      [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], device=device, dtype=dtype)
    return d / d.norm(dim=-1, keepdim=True)


def _fibonacci_sphere_directions(k, device=None, dtype=torch.float32):
    """k well-spread unit vectors on the sphere; the i+0.5 offset avoids the pole
    collapse a bare integer index would cause at small k."""
    i = torch.arange(k, device=device, dtype=dtype) + 0.5
    y = 1 - 2 * i / k
    r = (1 - y * y).clamp_min(0).sqrt()
    golden_angle = math.pi * (3 - math.sqrt(5))
    theta = golden_angle * i
    return torch.stack([torch.cos(theta) * r, y, torch.sin(theta) * r], dim=-1)


def tet_offset_directions(k, device=None, dtype=torch.float32):
    """k>=3 unit offset directions for a synthetic gradient-estimation element: the
    canonical regular tetrahedron at k=4, an isotropic Fibonacci-sphere spread otherwise."""
    if k < 3:
        raise ValueError("need at least 3 offsets for a well-posed 3D gradient")
    return _regular_tetrahedron_directions(device, dtype) if k == 4 else \
        _fibonacci_sphere_directions(k, device, dtype)


def _random_rotation(n, device=None, dtype=torch.float32, generator=None):
    """n Haar-uniform SO(3) matrices, [n, 3, 3] (QR trick, Mezzadri 2007) — each
    synthetic element gets its own random orientation so they don't all share the same
    axes relative to e.g. the query Fourier positional encoding."""
    a = torch.randn(n, 3, 3, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(a)
    q = q * torch.diagonal(r, dim1=-2, dim2=-1).sign().unsqueeze(-2)
    q = q.clone()
    q[..., :, -1] = q[..., :, -1] * torch.linalg.det(q).sign().unsqueeze(-1)
    return q


def sample_tet_centers(pos, wall_dist, batch, n_per_graph, min_wall_dist, wall_percentile=0.75):
    """Up to n_per_graph indices per graph: first restrict to the top `wall_percentile`
    fraction of eligible points (wall_dist > min_wall_dist) by wall distance — e.g. 0.75
    keeps everything above the 25th percentile of wall distance, excluding only the
    nearest-wall quarter. Biases toward the physically larger bulk-volume interior
    (aneurysm dome core, parent-vessel core) over the thin near-wall boundary layer, which
    WallNoSlipGate already constrains directly. Then farthest-point-sample down to
    n_per_graph *within* that candidate pool (not plain top-K by wall_dist alone), so
    centers spread across the bulk volume instead of clustering around a single deepest
    point (e.g. the dome apex). A graph with fewer eligible points just contributes all
    of them."""
    idx_list = []
    for b in batch.unique(sorted=False).tolist():
        m = (batch == b) & (wall_dist > min_wall_dist)
        idx = m.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        n_keep = max(min(n_per_graph, idx.numel()), int(idx.numel() * wall_percentile))
        if idx.numel() > n_keep:
            top = torch.topk(wall_dist[idx], n_keep).indices
            idx = idx[top]
        if idx.numel() > n_per_graph:
            sel = fps(pos[idx], ratio=n_per_graph / idx.numel())
            idx = idx[sel[:n_per_graph]]
        idx_list.append(idx)
    if not idx_list:
        return wall_dist.new_zeros(0, dtype=torch.long)
    return torch.cat(idx_list)


@dataclass(frozen=True)
class SyntheticOffsets:
    """K freshly-generated query points around each of P chosen interior centers,
    ready to be appended to an `interior` batch for one combined model forward pass."""
    pos: torch.Tensor          # [P*K, 3] flattened offset positions
    wall_dist: torch.Tensor    # [P*K] each offset inherits its center's wall_dist (approx:
                                # the offset itself is at most `radius` from a point already
                                # known to be > min_wall_dist_factor*radius from the wall)
    batch: torch.Tensor        # [P*K]
    center_idx: torch.Tensor   # [P] indices into the ORIGINAL interior batch
    center_pos: torch.Tensor   # [P, 3]
    k: int


def build_synthetic_offsets(pos, wall_dist, batch, n_centers, k_offsets, radius,
                            min_wall_dist_factor=2.0, wall_percentile=0.75, generator=None):
    """Pick <=n_centers eligible interior points per graph (wall_dist > min_wall_dist_factor
    * radius, so no offset vertex can cross the wall; farthest-point-sampled within the
    deepest `wall_percentile` fraction of those — see sample_tet_centers) and generate
    k_offsets isotropic offset positions around each. `radius` must be in the SAME
    (normalized) frame as `pos`/`wall_dist` — e.g. training's already-normalized
    interior.pos/interior.wall_dist, not physical units (same convention as
    continuity_residual above). Returns None if no point in the batch is far enough from
    the wall.
    """
    center_idx = sample_tet_centers(pos, wall_dist, batch, n_centers,
                                    min_wall_dist_factor * radius, wall_percentile)
    if center_idx.numel() == 0:
        return None
    device, dtype = pos.device, pos.dtype
    center_pos = pos[center_idx]
    directions = tet_offset_directions(k_offsets, device, dtype)
    rot = _random_rotation(center_pos.size(0), device, dtype, generator)
    dirs = torch.einsum("pij,kj->pki", rot, directions)                # [P, K, 3]
    offset_pos = center_pos.unsqueeze(1) + radius * dirs                # [P, K, 3]
    p, k = offset_pos.shape[:2]
    return SyntheticOffsets(
        pos=offset_pos.reshape(p * k, 3), wall_dist=wall_dist[center_idx].repeat_interleave(k),
        batch=batch[center_idx].repeat_interleave(k), center_idx=center_idx,
        center_pos=center_pos, k=k)


def append_synthetic_offsets(interior, offsets: SyntheticOffsets):
    """Combine `interior` with `offsets` into one batch ready for a single model
    forward pass. Query points are independent in the decoder (global cross-attn has no
    query-query interaction, local surface-gather is per-query kNN), so appending more
    query rows would leave the original rows' predictions unchanged IF PyG's `batch`
    stayed non-decreasing — which a naive concat does NOT guarantee for >1 graph
    (original rows already cycle through every graph id, so offsets appended after them
    cycle through 0..B-1 a second time). `to_dense_batch` (used by the decoder's global
    cross-attention) silently mis-assigns rows to the wrong graph when `batch` isn't
    sorted, so this stable-sorts the combined batch and returns index tensors to map
    predictions back.

    Returns (combined, orig_idx, offset_idx): combined[orig_idx] recovers interior's
    original row order; combined[offset_idx] recovers offsets.pos's row order.
    """
    orig_n = interior.pos.size(0)
    all_pos = torch.cat([interior.pos, offsets.pos], dim=0)
    all_wall_dist = torch.cat([interior.wall_dist, offsets.wall_dist], dim=0)
    all_batch = torch.cat([interior.batch, offsets.batch], dim=0)
    order = torch.argsort(all_batch, stable=True)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)

    combined = copy.copy(interior)
    combined.pos, combined.wall_dist, combined.batch = all_pos[order], all_wall_dist[order], all_batch[order]
    return combined, inv[:orig_n], inv[orig_n:]


def synthetic_continuity_residual(center_vel, offset_vel, offsets: SyntheticOffsets):
    """div(v) at each synthetic element's center. `center_vel`: [P, 3] (vel_pred at
    offsets.center_idx); `offset_vel`: [P, K, 3] (vel_pred at offsets.pos, reshaped —
    same [center-major, offset-minor] order as offsets.center_idx/offsets.pos). Always
    well-posed for k>=3 by construction — unlike the graph stencil, no `valid` mask needed."""
    p, k = offsets.center_idx.numel(), offsets.k
    offset_pos = offsets.pos.reshape(p, k, 3)
    dx = offset_pos - offsets.center_pos.unsqueeze(1)
    dv = offset_vel - center_vel.unsqueeze(1)
    grad = torch.linalg.lstsq(dx, dv).solution.transpose(1, 2)          # [P, 3, 3]
    return grad.diagonal(dim1=1, dim2=2).sum(-1)                        # [P]
