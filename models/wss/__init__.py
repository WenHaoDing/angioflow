"""Wall-shear-stress (WSS) surface prediction baselines.

  hierarchical_gps.py  — AngioHierarchicalGPS: graph U-Net (FPS pool/unpool + GPS)
  pointnet2.py          — AngioPointNet2
  query_decoder.py       — TwoScaleWSSDecoder: query-point-style decoder building blocks
  query_model.py          — AngioWSSQueryField, build_wss_query_field

query_model/query_decoder deliberately mirror angioflow/models/interior/ (geometry
backbone -> coarse tokens -> query cross-attention), reusing its encoders directly, so a
future two-branch WSS+velocity model can share one geometry backbone with two heads.
"""
from angioflow.models.wss.query_model import AngioWSSQueryField, build_wss_query_field

__all__ = ["AngioWSSQueryField", "build_wss_query_field"]
