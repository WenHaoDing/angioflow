"""Regression losses, pattern losses, and physics-informed loss terms."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def log_mag_loss(pred, y, y_mean, y_std, log_eps: float, max_wss: float) -> torch.Tensor:
    """MSE on log-clamped WSS magnitude (Pa/s).

    Denormalises pred/y to Pa/s, computes log(clamp(||WSS||, log_eps, max_wss)),
    then applies MSE in that log space.

    Gives equal gradient weight to proportional errors across [log_eps, max_wss] Pa/s.
    Regions above max_wss are saturated → zero loss (model not penalised for imprecision
    at high-WSS regions such as the neck).  The caller should add a Huber term in
    normalised space to prevent unbounded raw predictions.

    Args:
        pred:    model output in normalised space  [N, 3]
        y:       ground truth in normalised space  [N, 3]
        y_mean:  per-component mean used to normalise y  [3]
        y_std:   per-component std  used to normalise y  [3]
        log_eps: floor magnitude (Pa/s) — prevents log(0)
        max_wss: ceiling magnitude (Pa/s) — above this, loss = 0
    """
    y_raw = y    * y_std + y_mean                                      # [N, 3] Pa/s
    p_raw = pred * y_std + y_mean                                      # [N, 3] Pa/s
    gt_logmag = torch.log(y_raw.norm(dim=-1).clamp(log_eps, max_wss)) # [N]
    pd_logmag = torch.log(p_raw.norm(dim=-1).clamp(log_eps, max_wss)) # [N]
    # MSE in log space: clamping already bounds the range, no outliers to guard against here
    return F.mse_loss(pd_logmag, gt_logmag)


def smooth_loss(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Dirichlet energy on predictions (L2 on edge differences).

    Encourages neighbouring nodes to predict similar values. L2 heavily penalises
    long-wavelength variation but is weak on per-triangle speckle. Prefer
    `tv_smooth_loss` if you want to kill speckle.

    Returns 0 (not NaN) when edge_index has no edges — e.g. a random query-point
    subset (see wss_query_field) whose induced subgraph happens to have no
    mutually-adjacent pairs this step; `.mean()` on an empty tensor is NaN otherwise,
    which would poison the whole loss.
    """
    i, j = edge_index
    if i.numel() == 0:
        return pred.new_zeros(())
    return (pred[i] - pred[j]).pow(2).mean()


def tv_smooth_loss(pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Total-variation smoothness (L1 on edge differences).

    Much stronger than Dirichlet at eliminating per-triangle jitter without
    over-smoothing large-scale features (e.g. the WSS hot spot at the neck).
    Standard TV vector norm: sum of edge-wise 3-vector magnitudes, meaned.

    Returns 0 (not NaN) when edge_index has no edges — see smooth_loss.
    """
    i, j = edge_index
    if i.numel() == 0:
        return pred.new_zeros(())
    return (pred[i] - pred[j]).norm(dim=-1).mean()


def directional_loss(pred, normals, y_mean, y_std) -> torch.Tensor:
    """Physics prior: WSS is the tangential traction, so wss · n = 0 on the wall.

    Denormalises the prediction and penalises the magnitude of its surface-normal
    component (L1, robust to hot spots). Needs no ground truth — pure consistency
    with the surface geometry. Units: Pa/s (raw WSS scale).

    Note the floor: `normals` are estimated from the decimated mesh, so even the
    CFD ground truth has a small non-zero residual under this loss — don't expect
    (or force) it to reach exactly zero.

    Args:
        pred:    model output in normalised space  [N, 3]
        normals: unit surface normals              [N, 3]
        y_mean:  per-component mean used to normalise y  [3]
        y_std:   per-component std  used to normalise y  [3]
    """
    p_raw = pred * y_std + y_mean                     # [N, 3] Pa/s
    return (p_raw * normals).sum(dim=-1).abs().mean()
