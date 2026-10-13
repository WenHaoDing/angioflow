"""Interior velocity/pressure field baselines (geometry encoder + two-scale decoder).

Public API:
  build_interior_field(backbone="hgps"|"pointnet", ...) -> AngioInteriorField
  AngioInteriorField.forward(wall, interior, waveform) -> (vel, press)

Building blocks (for future variants) live in encoders.py / decoder.py.
"""
from angioflow.models.interior.model import AngioInteriorField, build_interior_field
from angioflow.models.interior.token_pool import GeometryLatentResampler

__all__ = ["AngioInteriorField", "GeometryLatentResampler", "build_interior_field"]
