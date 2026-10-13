"""Shared sanity-rendering for the three register_cfd_shapes_*.py scripts (AneuG-Flow, AngioFlow,
aneugv2) -- all three write the same wall_mesh_registration.pt schema (pos/faces/
node_type/geo_dist, node_type 0=wall,1=focused,2=inlet,3=outlet_0,4=outlet_1, single-
channel geo_dist), so one matplotlib renderer covers all three instead of each script
keeping its own pyvista version.

register_cfd_shapes_aneugv2.py does NOT import this: it runs as a self-contained
Hugging Face Job with no local `angioflow` import (see that file's own docstring), so
it keeps its own inlined copy of _trisurf_fig/mesh.png/registration.png/geo_dist.png --
kept in sync with this one by hand. This module is the shared/canonical version for
AneuG-Flow and AngioFlow, which already import other angioflow.data helpers freely.

All sanity output -- every version, every script -- lands under one shared local root,
organized by case, so the whole registration stage can be eyeballed from one place:
    SANITY_ROOT/<case>/mesh.png
    SANITY_ROOT/<case>/registration.png
    SANITY_ROOT/<case>/geo_dist.png
(plus each script's own extra diagnostics in that same per-case folder: AneuG-Flow's
orphan_fix.png, AngioFlow's alignment.png/mesh+wss.gif+vel.gif from sanity_check_case.)
"""
from __future__ import annotations

import os

import numpy as np

SANITY_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_ghd_graphs/_sanity"


def case_sanity_dir(case_name, root=SANITY_ROOT):
    d = os.path.join(root, case_name)
    os.makedirs(d, exist_ok=True)
    return d


def _trisurf_fig(pos, faces, scalars=None, cmap="viridis", clim=None, title=None, highlight=None):
    """One static mesh panel. matplotlib only -- no pyvista/VTK/xvfb needed for a QA PNG.

    highlight: optional [(mask, color, label), ...] -- scattered ON TOP of the mesh in
    a bold marker, for regions too few nodes (inlet/outlet rims) to read from face
    colour alone."""
    import matplotlib.pyplot as plt
    lo, hi = pos.min(0), pos.max(0)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    if scalars is None:
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=faces,
                        color="lightgrey", linewidth=0.1, edgecolor="dimgrey", antialiased=False)
    else:
        tri = ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=faces,
                              cmap=cmap, linewidth=0, antialiased=False)
        tri.set_array(scalars[faces].mean(axis=1))
        tri.set_clim(*clim) if clim is not None else tri.autoscale()
        fig.colorbar(tri, ax=ax, shrink=0.5)
    if highlight:
        for mask, color, label in highlight:
            if mask.any():
                ax.scatter(pos[mask, 0], pos[mask, 1], pos[mask, 2], color=color, s=50,
                          edgecolors="black", linewidths=0.6, label=f"{label} ({int(mask.sum())})",
                          depthshade=False)
        ax.legend(loc="upper left", fontsize=8, markerscale=1.2)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo).clip(min=1e-9))
    ax.set_axis_off()
    return fig


def save_registration_sanity(case, pos, faces, node_type, geo_dist, out_dir=None):
    """mesh.png (plain geometry), registration.png (node_type, inlet/outlet/outlet_1
    highlighted), geo_dist.png -- the standardized 3-image set for ANY
    register_cfd_shapes_*.py script's wall_mesh_registration.pt (same schema in all
    three). node_type/geo_dist accept either a torch tensor or a numpy array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or case_sanity_dir(case)
    nt = node_type.numpy() if hasattr(node_type, "numpy") else np.asarray(node_type)
    gd = geo_dist.numpy() if hasattr(geo_dist, "numpy") else np.asarray(geo_dist)

    fig = _trisurf_fig(pos, faces, title=f"{case}: wall mesh ({len(pos)} nodes)")
    fig.savefig(os.path.join(out_dir, "mesh.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    highlight = [(nt == 2, "red", "inlet"), (nt == 3, "blue", "outlet_0"), (nt == 4, "lime", "outlet_1")]
    fig = _trisurf_fig(pos, faces, nt.astype(float), "tab10", (0, 4),
                       f"{case}: node_type (0=wall 1=focused 2=inlet 3=out0 4=out1)",
                       highlight=highlight)
    fig.savefig(os.path.join(out_dir, "registration.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    fig = _trisurf_fig(pos, faces, gd, "viridis", None, f"{case}: geo_dist (m)")
    fig.savefig(os.path.join(out_dir, "geo_dist.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    return out_dir
