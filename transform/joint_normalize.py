"""Normalization for the joint (wall, interior) task — one canonical position scale.

Combines the two single-task normalizers without letting them fight over wall.pos:
  * positions (wall + interior) and wall_dist scale by the INTERIOR stats' pos_std
    (isotropic scalar — the interior convention, safe under random rotation);
  * wall.y (WSS) z-scores with the WSS stats' y_mean/y_std (as PosYNormalize);
  * interior.vel / interior.press z-score with the interior stats (as InteriorNormalize).

Apply AFTER CenterAndRandomRotate, like both parents. Both stats dicts are the
existing on-disk files (stats.pt / interior_stats.pt) — nothing new is computed.
"""
from __future__ import annotations


class JointNormalize:
    def __init__(self, wss_stats, interior_stats):
        self.pos_std = interior_stats["pos_std"]
        self.pos_scalar = interior_stats["pos_std"][0]
        self.y_mean = wss_stats["y_mean"]
        self.y_std = wss_stats["y_std"]
        self.vel_mean = interior_stats["vel_mean"]
        self.vel_std = interior_stats["vel_std"]
        self.press_mean = interior_stats["press_mean"]
        self.press_std = interior_stats["press_std"]

    def __call__(self, data):
        wall, interior = data
        wall.pos = wall.pos / self.pos_std
        wall.y = (wall.y - self.y_mean) / self.y_std
        interior.pos = interior.pos / self.pos_std
        interior.wall_dist = interior.wall_dist / self.pos_scalar
        interior.vel = (interior.vel - self.vel_mean) / self.vel_std
        interior.press = (interior.press - self.press_mean) / self.press_std
        return wall, interior

    def inverse_y(self, y):
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)

    def inverse_vel(self, vel):
        return vel * self.vel_std.to(vel.device) + self.vel_mean.to(vel.device)

    def inverse_press(self, press):
        return press * self.press_std.to(press.device) + self.press_mean.to(press.device)
