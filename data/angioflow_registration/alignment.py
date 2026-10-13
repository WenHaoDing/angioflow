"""Canonical/GHD-space <-> CFD world-frame affine alignment.

Copied from AneuGv2's dataset/preprocess.py: the canonical->world formula only
needs the per-case fitting affine (R from w_rot, s = exp(log_scale), t_vec) plus
the per-case landmark frame (s_can, R_frame, neck_pt_world) -- no mesh-to-mesh
ICP registration is needed for this step.
"""

from pathlib import Path

import numpy as np

# Canonical mesh assets (mesh.obj / eigenvectors.npy / mesh_trimmed.obj per type)
# live in the AneuG checkout (formerly a sibling "AneuGv2" directory, now vendored
# under angioflow/); these are data assets, not code, so we reference them by path
# rather than copying the files into angioflow.
DEFAULT_CANONICAL_ROOT = Path(__file__).resolve().parents[2] / "AneuG" / "dataset" / "canonical"

CANONICAL_BY_TYPE = {
    0: ("bifurcated", DEFAULT_CANONICAL_ROOT / "Bifurcated"),
    1: ("sidewall", DEFAULT_CANONICAL_ROOT / "Sidewall"),
    2: ("sidewall", DEFAULT_CANONICAL_ROOT / "Sidewall"),
}


def ghd_world_alignment(data, R, s, t_vec, s_can, R_frame, neck_pt, style='c2w'):
    """
    canonical -> world:
    equivalent formula:
    w = c @ R_equivalent.T * s_equivalent + t_equivalent
    c = (w - t_equivalent) / s_equivalent @ R_equivalent
    R_equivalent = (R_frame.T @ R)
    s_equivalent = s * s_can
    t_equivalent = (t_vec @ R_frame) * s_can + neck_pt
    reverse ->
    c = (w - t_equivalent) / s_equivalent @ R_equivalent
    c* = (w - t_equivalent) / s_equivalent @ R_equivalent * s_equivalent
    c* = (w - t_equivalent) @ R_equivalent
    """
    if data.__class__.__module__.startswith("torch"):
        data = data.detach().cpu().numpy()

    R_equivalent = R_frame.T @ R
    t_equivalent = s_can * t_vec @ R_frame + neck_pt
    s_equivalent = s * s_can

    if style == "c2w":
        return data @ R_equivalent.T * s_equivalent + t_equivalent
    if style == "w2c":
        return (data - t_equivalent) / s_equivalent @ R_equivalent
    if style == "w2c_no_scale":
        return (data - t_equivalent) @ R_equivalent
    raise ValueError(f"Unknown alignment style: {style}")


def _so3_exp_map_numpy(w):
    """Rodrigues exponential map: axis-angle vector -> SO(3) matrix."""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = w / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
