"""Make merged_reconstruction_remeshed.obj for AneuX cases that only have
merged_reconstruction.ply, so regularize_angioflow_fix.py can smooth them like the rest.

The existing *_remeshed.obj files are the .ply isotropically remeshed to a ~0.132 mm
median edge (max ~0.2 mm), with the same openings, area and bounds; the tool that made
them is not in this repo. This reproduces that with mesh_regularizer.remesh (pymeshlab
isotropic remeshing, calibrated so the achieved edge matches the request).

--validate remeshes the .ply of cases that DO have a remeshed .obj and compares the result
with it (edge lengths, area, openings, surface distance), to check the reproduction.
A remeshed case gets merged_reconstruction_remeshed.obj + remesh_info.json (how it was
made -- so it is never mistaken for an original).

Needs the vmtk_autogen env:
    conda activate vmtk_autogen
    python debugging/remesh_aneux_merged.py --validate
    python debugging/remesh_aneux_merged.py --cases_file debugging/file_lists/aneux_no_remeshed_obj.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ANEUG = "/media/yaplab2/HDD Storage/wenhao/AneuG"
ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/angioflow_fix"
REFERENCE_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_stable"
SRC, OUT = "merged_reconstruction.ply", "merged_reconstruction_remeshed.obj"
TARGET_EDGE = 0.132          # mm: median edge of the existing remeshed .obj files
MAX_SHIFT = 1.0              # mm an opening centroid may move (the volume mesher's limit)


def load(path):
    import trimesh
    return trimesh.load(path, process=True)        # merges duplicate vertices


def describe(m):
    """Median/max edge, area, number of openings, opening centroids."""
    import trimesh
    e = m.edges_unique_length
    boundary = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
    edges = m.edges_sorted[boundary]
    # connected boundary loops = openings
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    loops = {}
    for v in np.unique(edges):
        loops.setdefault(find(v), []).append(v)
    cents = [m.vertices[v].mean(0) for v in loops.values()]
    return dict(edge_median=float(np.median(e)), edge_max=float(e.max()), area=float(m.area),
                openings=len(loops), centroids=cents)


def remesh(m):
    sys.path.insert(0, ANEUG)
    from mesh_regularizer.mesh_regularizer import remesh as mr_remesh
    return mr_remesh(m, TARGET_EDGE, backend="pymeshlab", calibrate=True)


def compare(a, b):
    import trimesh
    da, db = describe(a), describe(b)
    shift = (max(min(np.linalg.norm(c - d) for d in db["centroids"]) for c in da["centroids"])
             if da["openings"] == db["openings"] else float("nan"))
    _, dist, _ = trimesh.proximity.closest_point(a, b.vertices[np.random.default_rng(0).choice(
        len(b.vertices), min(5000, len(b.vertices)), replace=False)])
    return da, db, shift, float(dist.mean()), float(dist.max())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--cases_file", default=None, help="cases to remesh (one per line)")
    p.add_argument("--validate", action="store_true",
                   help=f"compare against existing {OUT} files in {REFERENCE_ROOT} instead")
    p.add_argument("--n_validate", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.validate:
        cases = sorted(c for c in os.listdir(REFERENCE_ROOT)
                       if os.path.exists(os.path.join(REFERENCE_ROOT, c, OUT)))
        cases = [cases[i] for i in np.linspace(0, len(cases) - 1, args.n_validate).astype(int)]
        for c in cases:
            ply = load(os.path.join(REFERENCE_ROOT, c, SRC))
            ours, theirs = remesh(ply), load(os.path.join(REFERENCE_ROOT, c, OUT))
            do, dt, shift, dmean, dmax = compare(ours, theirs)
            dp = describe(ply)
            print(f"{c}\n   ply      edge med {dp['edge_median']:.3f} max {dp['edge_max']:.3f}  area {dp['area']:.1f}  openings {dp['openings']}"
                  f"\n   existing edge med {dt['edge_median']:.3f} max {dt['edge_max']:.3f}  area {dt['area']:.1f}  openings {dt['openings']}"
                  f"\n   ours     edge med {do['edge_median']:.3f} max {do['edge_max']:.3f}  area {do['area']:.1f}  openings {do['openings']}"
                  f"  | opening centroid shift {shift:.3f} mm, existing->ours distance mean {dmean:.4f} max {dmax:.3f} mm", flush=True)
        return

    cases = [l.strip() for l in open(args.cases_file) if l.strip()]
    for c in cases:
        d = os.path.join(args.root, c)
        out = os.path.join(d, OUT)
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {c}: {OUT} exists"); continue
        ply = load(os.path.join(d, SRC))
        m = remesh(ply)
        dm, dp, shift, dmean, dmax = compare(m, ply)        # (remeshed, ply)
        if dm["openings"] != dp["openings"] or not shift <= MAX_SHIFT:
            print(f"[FAIL] {c}: openings {dp['openings']} -> {dm['openings']}, shift {shift:.3f} mm -- not written", flush=True)
            continue
        m.export(out)
        info = dict(source=SRC, method="mesh_regularizer.remesh(backend='pymeshlab', calibrate=True)",
                    target_edge_mm=TARGET_EDGE, edge_median_mm=dm["edge_median"], edge_max_mm=dm["edge_max"],
                    area_before=dp["area"], area_after=dm["area"], openings=dm["openings"],
                    opening_centroid_shift_mm=shift, made_by="angioflow/debugging/remesh_aneux_merged.py")
        with open(os.path.join(d, "remesh_info.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(f"[ok] {c}: edge med {dm['edge_median']:.3f} max {dm['edge_max']:.3f}, area {dp['area']:.1f} -> {dm['area']:.1f}, "
              f"openings {dm['openings']}, centroid shift {shift:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
