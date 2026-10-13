"""Symmetrize a Data object's edge_index — as a dataset transform, not a model concern.

angioflow.data.mesh.face_to_edge_index stores each undirected mesh edge ONCE
(canonicalised i<j, to halve on-disk size), not both directions — confirmed
empirically on real data (100% of a checked case's edges had no reverse pair
present). Any graph conv reading data.edge_index directly (GINEConv/SAGEConv/
GATConv in AngioHierarchicalGPS and AngioGraphGPS, pre_convs in GeoHierGPSEncoder,
...) only aggregates from the half of a node's true neighbours where it happens to
be the larger index, unless this is applied first.

Does NOT fix the coarse-level kNN graphs models like AngioHierarchicalGPS /
GeoHierGPSEncoder rebuild internally after FPS pooling — those don't exist until
the model's forward pass (built from pooled positions computed at that point), so
a pre-processing transform can't reach them; each model's own knn_graph rebuild
still needs its own to_undirected call (see their `bidirectional` option).
"""
from __future__ import annotations

from torch_geometric.utils import to_undirected


class SymmetrizeEdges:
    """data.edge_index -> both directions, deduplicated. No-op if already symmetric."""

    def __call__(self, data):
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        return data
