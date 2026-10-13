"""Local gradient estimators built from freshly-generated points, not a fixed
precomputed graph. See README.md for why.

For a locally-linear field, least-squares over K>=3 offset "element" vertices
around a center is exactly the Green-Gauss / FEM shape-function-derivative
gradient: K=3 is the minimal, exactly-determined (zero-redundancy) case; a
regular tetrahedron (K=4, center at the centroid rather than a vertex) adds one
redundant equation almost for free; larger K (Fibonacci-sphere directions) adds
more noise-averaging redundancy at the cost of more model queries.
"""
from __future__ import annotations

import math

import torch


def regular_tetrahedron_directions(device=None, dtype=torch.float64):
    """4 unit vectors to a regular tetrahedron's vertices, centered at the
    origin (so a probe using these is the tet's centroid, not one of its
    vertices) — the canonical, maximally-isotropic K=4 element."""
    d = torch.tensor([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                      [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], device=device, dtype=dtype)
    return d / d.norm(dim=-1, keepdim=True)


def fibonacci_sphere_directions(k, device=None, dtype=torch.float64):
    """k well-spread unit vectors on the sphere (Fibonacci spiral) — a general
    K>=3 element; approximately tetrahedral at k=4, more isotropic coverage
    (and more noise-averaging redundancy) as k grows.

    Uses i+0.5 (not i) as the standard Fibonacci-lattice offset — with a bare
    integer i, small k places points exactly at the poles (e.g. k=3 puts two of
    three points on the same axis, nearly collinear, which is catastrophically
    ill-conditioned for the 3x3 solve in `local_gradient`)."""
    if k < 3:
        raise ValueError("need at least 3 offsets for a well-posed 3D gradient")
    i = torch.arange(k, device=device, dtype=dtype) + 0.5
    y = 1 - 2 * i / k
    r = (1 - y * y).clamp_min(0).sqrt()
    golden_angle = math.pi * (3 - math.sqrt(5))
    theta = golden_angle * i
    x = torch.cos(theta) * r
    z = torch.sin(theta) * r
    return torch.stack([x, y, z], dim=-1)


def random_rotation(n, device=None, dtype=torch.float64, generator=None):
    """n Haar-uniform SO(3) matrices, [n, 3, 3] (QR trick, Mezzadri 2007) —
    same construction as transform.rotate._random_rotation_matrices."""
    a = torch.randn(n, 3, 3, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(a)
    d = torch.diagonal(r, dim1=-2, dim2=-1).sign()
    q = q * d.unsqueeze(-2)
    det = torch.linalg.det(q)
    q = q.clone()
    q[..., :, -1] = q[..., :, -1] * det.sign().unsqueeze(-1)
    return q


def offset_positions(center_pos, directions, radius, rotate=True, generator=None):
    """[P, 3] centers + [K, 3] unit directions -> [P, K, 3] synthetic offset
    positions, each probe's element independently rotated if `rotate` (avoids
    every element sharing the same orientation relative to e.g. the Fourier
    positional encoding's axes)."""
    p, k = center_pos.size(0), directions.size(0)
    dirs = directions.to(center_pos.device, center_pos.dtype)
    if rotate:
        rot = random_rotation(p, device=center_pos.device, dtype=center_pos.dtype,
                              generator=generator)
        dirs = torch.einsum("pij,kj->pki", rot, dirs)      # [P, K, 3]
    else:
        dirs = dirs.expand(p, k, 3)
    return center_pos.unsqueeze(1) + radius * dirs


def local_gradient(center_val, offset_val, center_pos, offset_pos):
    """Least-squares gradient from a probe + its K>=3 offset element vertices.

    center_val: [P, C], offset_val: [P, K, C]; center_pos: [P, 3], offset_pos: [P, K, 3]
    Returns grad: [P, C, 3] with grad[p] @ (x_k - x_p) ~= val_k[p] - val_p.
    Differentiable in center_val/offset_val; positions are geometry (no_grad
    upstream is the caller's responsibility, same convention as constraints.py).
    """
    dx = offset_pos - center_pos.unsqueeze(1)               # [P, K, 3]
    dv = offset_val - center_val.unsqueeze(1)                # [P, K, C]
    grad = torch.linalg.lstsq(dx, dv).solution                # [P, 3, C]
    return grad.transpose(1, 2)                               # [P, C, 3]


def divergence(vel_grad):
    """[P, 3, 3] -> [P] trace (sum of diagonal), i.e. tr(grad v) = div(v)."""
    return vel_grad.diagonal(dim1=-2, dim2=-1).sum(-1)
