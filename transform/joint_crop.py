"""ROI crop for the joint (wall, interior) task: crop the wall to the aneurysm +
random ring (exactly AneurysmCrop), then drop interior points whose local wall
context did not survive the crop.

Keep rule for an interior point: dist(point, kept wall) <= factor * wall_dist + slack,
where wall_dist is its stored distance to the FULL wall. If the point's true nearest
wall patch survived, the two distances are nearly equal and it stays; if its wall
neighbourhood was cropped away, dist-to-kept-wall jumps and it is dropped. This keeps
both prediction targets focused on the ROI and guarantees every surviving interior
point still has genuine local wall geometry for LocalSurfaceGather to read.

Run FIRST in the transform chain, in the raw metric frame (both distances are
computed against wall nodes in the same frame, so the ratio test is well posed).
"""
from __future__ import annotations

import torch
from scipy.spatial import cKDTree

from angioflow.transform.crop import AneurysmCrop

_INTERIOR_NODE_KEYS = ("pos", "vel", "press", "grad_v", "grad_p", "wall_dist")


class JointAneurysmCrop:
    def __init__(self, low=0.0, high=0.25, aneurysm_only=False,
                 factor=1.5, slack=5e-4):
        self.wall_crop = AneurysmCrop(low=low, high=high, aneurysm_only=aneurysm_only)
        self.factor = factor
        self.slack = slack          # metres; absorbs discretisation noise near the wall

    def __call__(self, data):
        wall, interior = data
        wall = self.wall_crop(wall)
        d_kept = cKDTree(wall.pos.numpy()).query(interior.pos.numpy())[0]
        keep = torch.from_numpy(d_kept) <= self.factor * interior.wall_dist + self.slack
        if keep.all():
            return wall, interior
        for key in _INTERIOR_NODE_KEYS:
            if key in interior:
                interior[key] = interior[key][keep]
        return wall, interior
