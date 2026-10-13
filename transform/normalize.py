"""Z-score normalization of `data.pos` and `data.y`."""
from __future__ import annotations


class PosYNormalize:
    """PyG `transform` callable; z-scores `data.pos` and `data.y` using precomputed stats."""

    def __init__(self, stats):
        self.pos_mean = stats["pos_mean"]
        self.pos_std = stats["pos_std"]
        self.y_mean = stats["y_mean"]
        self.y_std = stats["y_std"]

    def __call__(self, data):
        data.pos = (data.pos - self.pos_mean) / self.pos_std
        data.y = (data.y - self.y_mean) / self.y_std
        return data

    def inverse_y(self, y):
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)
