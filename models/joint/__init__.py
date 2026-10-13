"""Joint WSS + interior velocity/pressure prediction (one shared HierGPS encoder,
two query-style heads coupled by per-block cross-field attention).

Public API:
  build_joint_field(...) -> AngioJointField
  AngioJointField.forward(wall, interior, waveform) -> (wss, query_idx, vel, press)
"""
from angioflow.models.joint.model import AngioJointField, build_joint_field

__all__ = ["AngioJointField", "build_joint_field"]
