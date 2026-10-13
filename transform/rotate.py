"""Per-graph centering + uniform random SO(3) rotation."""
from __future__ import annotations

import torch


def _random_rotation_matrices(n, device=None, dtype=torch.float32):
    """Haar-uniform SO(3) rotation matrices via QR (Mezzadri 2007). Shape [n, 3, 3]."""
    A = torch.randn(n, 3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # make Q uniform on O(3): multiply columns by sign of R's diagonal
    d = torch.diagonal(R, dim1=-2, dim2=-1).sign()
    Q = Q * d.unsqueeze(-2)
    # restrict to SO(3): flip last column if det < 0
    det = torch.linalg.det(Q)
    Q[..., :, -1] = Q[..., :, -1] * det.sign().unsqueeze(-1)
    return Q


class CenterAndRandomRotate:
    """Centre `data.pos` at its centroid, then apply a random SO(3) rotation
    consistently to `data.pos`, `data.normals`, and `data.y` (treated as 3-vectors).
    Per-graph in batched mode (uses `data.batch`).
    """

    def __init__(self, rotate=True):
        self.rotate = rotate

    def _rotate_vec(self, vec, R):
        # vec: [N, 3], R: [N, 3, 3] -> [N, 3]
        return torch.einsum("nij,nj->ni", R, vec)

    def __call__(self, data):
        # interior dataset: (wall, interior) pair — centre + rotate BOTH streams
        # with the SAME centroid and R (they are the same geometry) so they stay
        # aligned. The centroid is the wall's, matching how the wall static was
        # centred at preprocessing.
        if isinstance(data, tuple):
            return self._call_pair(*data)

        pos = data.pos
        batch = getattr(data, "batch", None)

        if batch is None:
            # single graph
            data.pos = pos - pos.mean(0, keepdim=True)
            if self.rotate:
                R = _random_rotation_matrices(1, device=pos.device, dtype=pos.dtype)[0]   # [3, 3]
                data.pos = data.pos @ R.T
                if hasattr(data, "normals"):
                    data.normals = data.normals @ R.T
                if hasattr(data, "y"):
                    data.y = data.y @ R.T
            return data

        # batched: per-graph centroid and rotation
        B = int(batch.max()) + 1
        sums = torch.zeros(B, 3, dtype=pos.dtype, device=pos.device)
        sums.index_add_(0, batch, pos)
        counts = torch.bincount(batch, minlength=B).to(pos.dtype).clamp_min(1).unsqueeze(-1)
        centroid = sums / counts                                       # [B, 3]
        data.pos = pos - centroid[batch]

        if self.rotate:
            R = _random_rotation_matrices(B, device=pos.device, dtype=pos.dtype)   # [B, 3, 3]
            Rn = R[batch]                                              # [N, 3, 3]
            data.pos = self._rotate_vec(data.pos, Rn)
            if hasattr(data, "normals"):
                data.normals = self._rotate_vec(data.normals, Rn)
            if hasattr(data, "y"):
                data.y = self._rotate_vec(data.y, Rn)
        return data

    def _call_pair(self, wall, interior):
        """Centre + rotate a (wall, interior) pair with a shared centroid and R.

        Applied per-sample (pre-batch), so both are single graphs. Interior
        velocity and pressure-gradient rotate as vectors (R·v, R·g); the velocity
        gradient is a rank-2 tensor and rotates as R·G·Rᵀ (G[k] = ∂v_i/∂x_j).
        """
        c = wall.pos.mean(0, keepdim=True)
        wall.pos = wall.pos - c
        interior.pos = interior.pos - c
        if not self.rotate:
            return wall, interior

        R = _random_rotation_matrices(1, device=wall.pos.device, dtype=wall.pos.dtype)[0]
        wall.pos = wall.pos @ R.T
        if getattr(wall, "normals", None) is not None:
            wall.normals = wall.normals @ R.T
        if getattr(wall, "y", None) is not None:      # interior task carries no WSS y
            wall.y = wall.y @ R.T

        interior.pos = interior.pos @ R.T
        interior.vel = interior.vel @ R.T
        # gradients are optional (a baseline that predicts only vel/press drops them)
        if getattr(interior, "grad_p", None) is not None:
            interior.grad_p = interior.grad_p @ R.T
        if getattr(interior, "grad_v", None) is not None:
            G = interior.grad_v.view(-1, 3, 3)                   # [P, 3, 3]
            interior.grad_v = torch.einsum("ip,kpq,jq->kij", R, G, R).reshape(-1, 9)
        return wall, interior
