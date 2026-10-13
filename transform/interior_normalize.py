"""Z-score normalization for the (wall, interior) interior-field task.

Separate from angioflow/transform/normalize.py (WSS PosYNormalize) so the tasks
don't share mutable code. Applied AFTER CenterAndRandomRotate (so pos is already
centred/rotated) and after InteriorSubsampleGraph. All stats are isotropic /
scalar, so they commute with the random rotation.

  wall.pos, interior.pos  /= pos_std      (shared frame)
  interior.wall_dist      /= pos_std      (a distance in the same frame)
  interior.vel            = (vel - 0) / vel_std
  interior.press          = (press - press_mean) / press_std
"""
from __future__ import annotations


class InteriorNormalize:
    """PyG `transform` callable on the (wall, interior) pair."""

    def __init__(self, stats):
        self.pos_std = stats["pos_std"]
        self.pos_scalar = stats["pos_std"][0]
        self.vel_mean = stats["vel_mean"]
        self.vel_std = stats["vel_std"]
        self.press_mean = stats["press_mean"]
        self.press_std = stats["press_std"]

    def __call__(self, data):
        wall, interior = data
        wall.pos = wall.pos / self.pos_std
        interior.pos = interior.pos / self.pos_std
        interior.wall_dist = interior.wall_dist / self.pos_scalar
        interior.vel = (interior.vel - self.vel_mean) / self.vel_std
        interior.press = (interior.press - self.press_mean) / self.press_std
        return wall, interior

    def inverse_vel(self, vel):
        return vel * self.vel_std.to(vel.device) + self.vel_mean.to(vel.device)

    def inverse_press(self, press):
        return press * self.press_std.to(press.device) + self.press_mean.to(press.device)
