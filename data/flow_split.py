"""Outlet flow split computed from geometry, on the fly -- the rule the CFD outlet
boundary conditions were set with (write_flowsplit_ratio in the CFD setup):

    Q_0 / Q_1 = (d_0 / d_1) ** 2.45,   d = equivalent diameter sqrt(4 A / pi) of each outlet cap

normalized so q_0 + q_1 = 1. Index 0 is outlet_0 (node_type 3), index 1 is outlet_1
(node_type 4) of the registration. AneuGv2 and AngioFlow follow this rule exactly.
AneuG-Flow does not: its splits come from its flowsplit_ratio.txt (not reproducible from
outlet size), and in ~11% of cases that file lists them in the opposite outlet order --
generate_flow_split.py resolves both into one table the dataset reads.

The cap area comes from the outlet's rim nodes in the static graph: the rim loop is
projected onto its best-fit plane, ordered by angle around its centroid, and its
enclosed area taken with the shoelace formula. Needs the full mesh (run before any
crop, which usually removes the outlets). Single-outlet cases get [1, 0].

measured_outlet0_share() reads the split a simulation actually applied off its own
velocity field: for each outlet, |mean axial velocity| x cap area over the interior
points in a thin disc just inside the cap. A rough flux, but both outlets carry the same
bias, so the ratio holds: on AneuGv2 / AngioFlow it matches the stored split to a median
of 0.003-0.004. generate_flow_split.py uses it to put AneuG-Flow's stored splits in the
registration's outlet order.
"""
from __future__ import annotations

import numpy as np

FLOW_SPLIT_EXPONENT = 2.45
INLET, OUTLET_0, OUTLET_1 = 2, 3, 4


def rim_cap_area(pos, rim_mask):
    """Area enclosed by an opening's rim loop (pos [N, 3], rim_mask [N] bool)."""
    p = np.asarray(pos, dtype=np.float64)[np.asarray(rim_mask, dtype=bool)]
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)       # vt[0], vt[1] span the cap plane
    x, y = (p - c) @ vt[0], (p - c) @ vt[1]
    order = np.argsort(np.arctan2(y, x))
    x, y = x[order], y[order]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def flow_split_from_geometry(pos, node_type, exponent=FLOW_SPLIT_EXPONENT):
    """[q_outlet0, q_outlet1] (float64, sums to 1) from the outlet cap sizes; [1, 0] if
    the case has a single outlet."""
    node_type = np.asarray(node_type)
    if not (node_type == OUTLET_1).any():
        return np.array([1.0, 0.0])
    d0, d1 = (np.sqrt(4.0 * rim_cap_area(pos, node_type == k) / np.pi) for k in (OUTLET_0, OUTLET_1))
    ratio = (d0 / d1) ** exponent
    q = np.array([ratio / (1.0 + ratio), 1.0 / (1.0 + ratio)])
    return q / q.sum()


def cap_frame(pos, rim_mask):
    """(centroid, unit normal, mean radius) of an opening, from its rim nodes."""
    p = np.asarray(pos, dtype=np.float64)[np.asarray(rim_mask, dtype=bool)]
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    return c, vt[-1], float(np.linalg.norm(p - c, axis=1).mean())


def _outlet_flow(interior_pos, interior_vel, c, n, r, min_points=20):
    d = interior_pos - c
    axial = d @ n
    radial = np.linalg.norm(d - axial[:, None] * n, axis=1)
    sel = (np.abs(axial) < 0.6 * r) & (radial < 0.8 * r)
    if sel.sum() < min_points:
        return np.nan
    return abs(float((interior_vel[sel] @ n).mean())) * np.pi * r ** 2


def measured_outlet0_share(pos, node_type, interior_pos, interior_vel):
    """Share of the outflow leaving through outlet_0 in a simulated velocity field, or nan
    if it can't be measured (single outlet, or too few interior points near a cap).
    pos / interior_pos must share one frame (they do in the extracted static files)."""
    node_type = np.asarray(node_type)
    if not (node_type == OUTLET_1).any():
        return np.nan
    ipos, vel = np.asarray(interior_pos, dtype=np.float64), np.asarray(interior_vel, dtype=np.float64)
    q = [_outlet_flow(ipos, vel, *cap_frame(pos, node_type == k)) for k in (OUTLET_0, OUTLET_1)]
    return np.nan if np.isnan(q).any() or sum(q) <= 0 else q[0] / sum(q)
