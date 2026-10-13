"""Training-time interior subsample + induced graph, as a dataset transform.

The interior pool carries a tet-validated `edge_index` (built once at
preprocessing: KNN candidates whose geodesic-through-the-mesh matches their
straight-line distance, so no edge crosses the wall into a disconnected branch).
That graph is geometry-only and shared across all frames, so it lives on the pool.

Each __call__ draws a random `sample_ratio` of the pool (free per-epoch
augmentation) and returns the induced subgraph: only edges with BOTH endpoints
sampled, relabelled to the subsample's local indices. No KNN and no validation
at train time — just a mask + gather — so the graph stays topologically correct
by construction and the per-batch cost is negligible.

The interior dataset passes the whole `(wall, interior)` pair to `dataset.transform`,
so __call__ accepts either that pair (wall passes through untouched) or a lone
interior Data (composability / testing).
"""
from __future__ import annotations

import torch


class InteriorSubsampleGraph:
    """Random pool subsample + induced validated graph.

    Args:
        sample_ratio: fraction of the interior pool drawn each call. With a
            pool built at k=64 KNN, ratio 0.25 gives mean node degree ~16.
        max_points: cap on drawn points, same role as InteriorSubsample's cap —
            bounds memory (and, for graph variants, edge count) on the largest pools.
        min_points: never draw fewer than this many points (guards tiny pools).
    """

    def __init__(self, sample_ratio=0.25, max_points=None, min_points=1):
        self.sample_ratio = sample_ratio
        self.max_points = max_points
        self.min_points = min_points

    def __call__(self, data):
        if isinstance(data, tuple):
            wall, interior = data
            return wall, self._apply(interior)
        return self._apply(data)

    def _apply(self, d):
        if not hasattr(d, "edge_index") or d.edge_index is None:
            raise ValueError(
                "InteriorSubsampleGraph needs a precomputed pool `edge_index` on "
                "the interior Data; run the graph preprocessing first.")
        P = d.num_nodes
        k = max(self.min_points, int(round(P * self.sample_ratio)))
        if self.max_points is not None:
            k = min(k, self.max_points)
        k = min(k, P)
        if k >= P:
            return d                                    # eval / full-pool: no-op
        idx = torch.randperm(P, device=d.edge_index.device)[:k]
        # Data.subgraph filters edges to both-endpoints-in-subset, relabels them,
        # and slices every node-level attribute (pos/vel/press/wall_dist/grads/…)
        # and edge_attr — field-agnostic, so new fields need no change here.
        return d.subgraph(idx)


if __name__ == "__main__":
    # self-test on a 3D grid graph: the induced subsample must (1) introduce no
    # edge that wasn't in the original graph, (2) relabel consistently with pos,
    # (3) carry all node fields through, (4) leave the pair's wall element alone.
    from torch_geometric.data import Data

    g = 12
    xyz = torch.stack(torch.meshgrid(torch.arange(g), torch.arange(g),
                                     torch.arange(g), indexing="ij"), -1)
    pos = xyz.reshape(-1, 3).float()
    P = pos.shape[0]
    # 6-neighbour grid edges (undirected, both directions)
    idx3 = torch.arange(P).reshape(g, g, g)
    edges = []
    for ax in range(3):
        a = idx3.narrow(ax, 0, g - 1).reshape(-1)
        b = idx3.narrow(ax, 1, g - 1).reshape(-1)
        edges += [torch.stack([a, b]), torch.stack([b, a])]
    edge_index = torch.cat(edges, dim=1)
    orig = {(int(i), int(j)) for i, j in edge_index.t().tolist()}

    interior = Data(pos=pos, edge_index=edge_index,
                    vel=torch.randn(P, 3), press=torch.randn(P, 1),
                    wall_dist=torch.rand(P), grad=torch.randn(P, 9),
                    version="vX", case_id="c0")
    wall = Data(pos=torch.randn(50, 3))

    tf = InteriorSubsampleGraph(sample_ratio=0.25)
    torch.manual_seed(0)
    w2, sub = tf((wall, interior))

    # (4) wall untouched
    assert w2 is wall, "wall element must pass through"
    # subsample size
    assert abs(sub.num_nodes - round(P * 0.25)) <= 1, sub.num_nodes
    # (3) node fields carried + resized
    for f in ("pos", "vel", "press", "wall_dist", "grad"):
        assert sub[f].shape[0] == sub.num_nodes, f
    assert sub.version == "vX" and sub.case_id == "c0"
    # (1) no fabricated edges: every subgraph edge maps back to an original edge.
    # recover original ids from the position round-trip (grid pos are unique ints)
    key = {tuple(p): n for n, p in enumerate(pos.round().int().tolist())}
    back = [key[tuple(p)] for p in sub.pos.round().int().tolist()]
    for i, j in sub.edge_index.t().tolist():
        assert (back[i], back[j]) in orig, f"fabricated edge {back[i]}->{back[j]}"
    # degree stats
    deg = torch.bincount(sub.edge_index[0], minlength=sub.num_nodes)
    print(f"[ok] pool {P} -> subsample {sub.num_nodes} pts, "
          f"{sub.edge_index.shape[1]} directed edges, "
          f"degree min {int(deg.min())} / mean {deg.float().mean():.1f} / "
          f"max {int(deg.max())}")
    # full-pool no-op
    assert tf.__class__(sample_ratio=1.0)(interior) is interior
    print("[ok] all invariants hold")
