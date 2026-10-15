"""Outlet flow split computed from geometry, on the fly -- the rule the CFD outlet
boundary conditions were set with (write_flowsplit_ratio in the CFD setup):

    Q_0 / Q_1 = (d_0 / d_1) ** 2.45,   d = equivalent diameter sqrt(4 A / pi) of each outlet cap

normalized so q_0 + q_1 = 1. Index 0 is outlet_0 (node_type 3), index 1 is outlet_1
(node_type 4) -- the same order as the stored flow_split (verified against the CFD
outlet flows on all three datasets).

The cap area comes from the outlet's rim nodes in the static graph: the rim loop is
projected onto its best-fit plane, ordered by angle around its centroid, and its
enclosed area taken with the shoelace formula. Needs the full mesh (run before any
crop, which usually removes the outlets). Single-outlet cases get [1, 0].
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
