"""Loss functions for CFD field regression."""
from angioflow.losses.losses import log_mag_loss, smooth_loss, tv_smooth_loss, directional_loss

__all__ = ["log_mag_loss", "smooth_loss", "tv_smooth_loss", "directional_loss"]
