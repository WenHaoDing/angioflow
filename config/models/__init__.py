"""Model registry: name -> config module exposing `add_args(parser)` and
`build_model(args, stats, device)`.
"""
from __future__ import annotations

from angioflow.config.models import pointnet2, hierarchical_gps, wss_query_field

MODEL_REGISTRY = {
    "pointnet2": pointnet2,
    "hierarchical_gps": hierarchical_gps,
    "wss_query_field": wss_query_field,
}


def get_model_config(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
