"""Does AneuG-Flow's flowsplit_ratio.txt follow the outlet-size rule?

The CFD setup's rule (AneuG/cfd_mesher/volume_mesh.py write_flowsplit_ratio):
    Q_0 / Q_1 = (d_0 / d_1) ** 2.45,   d = sqrt(4 A / pi) of each outlet cap
AneuGv2 and AngioFlow follow it exactly; AneuG-Flow did not when the outlet areas were
taken from the registration rims. Ground truth here is each case's flowsplit_ratio.txt
("q_outlet0 q_outlet1", same order as the registration's outlet_0 / outlet_1).

For every case, each outlet's cap area is measured on every surface available locally:
    rim                     registration rims (node_type 3 / 4) -- the CFD wall's openings
    shape.obj               original generated surface
    shape_remeshed.obj      remeshed surface
    shape_remeshed_cfd.obj  the surface the CFD wall was built from (same vertices as rim)
(loops matched to inlet / outlet_0 / outlet_1 by nearest centroid). For each surface it
reports the split the rule predicts vs the file, and the exponent the file implies.

It also measures each daughter branch's radius AWAY from its outlet, at several geodesic
distances up the branch (ring_<d>mm: the wall nodes at that distance from the outlet rim
form a ring; radius = RMS distance to the ring centroid, area = pi r^2). If the file was
computed from vessel size somewhere other than the cut outlets (e.g. near the
bifurcation, where Murray's law applies), the rule fits best at that distance.

    python debugging/aneug_flow_flowsplit.py                              # all cases
    python debugging/aneug_flow_flowsplit.py --n_cases 100 --out /tmp/x.csv   # seeded random sample
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

CASE_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data"
EXPONENT = 2.45
SURFACES = ("shape.obj", "shape_remeshed.obj", "shape_remeshed_cfd.obj")
RING_MM = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)       # geodesic distance up each branch
RING_HALF_WIDTH_MM = 0.15
OPENINGS = {"inlet": 2, "outlet_0": 3, "outlet_1": 4}


def load_obj(path):
    V, F = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                V.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.array(V), np.array(F, dtype=np.int64)


def boundary_loops(faces):
    """Ordered vertex loops along edges used by exactly one face."""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    u, c = np.unique(e, axis=0, return_counts=True)
    adj = {}
    for a, b in u[c == 1]:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        loops.append(np.array(loop))
    return loops


def loop_area(pts):
    """Area of an ordered (near-planar) 3D loop -- Newell's vector area."""
    p = pts - pts.mean(0)
    return 0.5 * np.linalg.norm(np.cross(p, np.roll(p, -1, axis=0)).sum(0))


def rim_area(pos, mask):
    """Area enclosed by an unordered rim: order by angle in its best-fit plane first."""
    p = pos[mask]
    c = p.mean(0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    ang = np.arctan2((p - c) @ vt[1], (p - c) @ vt[0])
    return loop_area(p[np.argsort(ang)])


def ring_areas(pos, faces, mask, dists):
    """pi r^2 of the wall ring at each geodesic distance from an outlet rim (nan if empty)."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    w = np.linalg.norm(pos[e[:, 0]] - pos[e[:, 1]], axis=1)
    n = len(pos)
    g = coo_matrix((np.concatenate([w, w]), (np.concatenate([e[:, 0], e[:, 1]]),
                                             np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n)).tocsr()
    d = dijkstra(g, directed=False, indices=np.flatnonzero(mask), min_only=True)
    out = []
    for dd in dists:
        ring = pos[np.abs(d - dd) < RING_HALF_WIDTH_MM]
        if len(ring) < 8:
            out.append(np.nan)
            continue
        r = np.sqrt(((ring - ring.mean(0)) ** 2).sum(1).mean())
        out.append(np.pi * r ** 2)
    return out


def split_from_areas(a0, a1, exponent=EXPONENT):
    d0, d1 = np.sqrt(4 * a0 / np.pi), np.sqrt(4 * a1 / np.pi)
    r = (d0 / d1) ** exponent
    return r / (1 + r)


def implied_exponent(q0, a0, a1):
    """Exponent k with q0/q1 = (d0/d1)^k -- ill-defined when the outlets are near equal size."""
    ld = 0.5 * np.log(a0 / a1)
    return np.log(q0 / (1 - q0)) / ld if abs(ld) > 0.02 else np.nan


def analyse_case(case_dir):
    q_file = np.loadtxt(os.path.join(case_dir, "flowsplit_ratio.txt")).reshape(-1)
    q0 = q_file[0] / q_file.sum()
    reg = torch.load(os.path.join(case_dir, "wall_mesh_registration.pt"), weights_only=False)
    pos = reg["pos"].numpy().astype(np.float64) * 1e3              # m -> mm (obj units)
    nt = reg["node_type"].numpy()
    cent = {k: pos[nt == v].mean(0) for k, v in OPENINGS.items()}

    row = {"case": os.path.basename(case_dir), "q0_file": q0}
    areas = {"rim": {k: rim_area(pos, nt == v) for k, v in OPENINGS.items()}}
    for name in SURFACES:
        path = os.path.join(case_dir, name)
        if not os.path.exists(path):
            continue
        V, F = load_obj(path)
        loops = boundary_loops(F)
        areas[name] = {}
        for k, c in cent.items():                                  # nearest loop to each opening
            i = int(np.argmin([np.linalg.norm(V[l].mean(0) - c) for l in loops]))
            areas[name][k] = loop_area(V[loops[i]])
            row[f"{name}_{k}_match_dist_mm"] = float(np.linalg.norm(V[loops[i]].mean(0) - c))
    faces = reg["faces"].numpy()
    rings = {k: ring_areas(pos, faces, nt == OPENINGS[k], RING_MM) for k in ("outlet_0", "outlet_1")}
    for i, dd in enumerate(RING_MM):
        areas[f"ring_{dd:g}mm"] = {"outlet_0": rings["outlet_0"][i], "outlet_1": rings["outlet_1"][i]}
    for src, a in areas.items():
        row[f"{src}_area0_mm2"], row[f"{src}_area1_mm2"] = a["outlet_0"], a["outlet_1"]
        row[f"{src}_q0_rule"] = split_from_areas(a["outlet_0"], a["outlet_1"])
        row[f"{src}_k_implied"] = implied_exponent(q0, a["outlet_0"], a["outlet_1"])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case_root", default=CASE_ROOT)
    p.add_argument("--n_cases", type=int, default=None, help="random sample of this many cases (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="per-case CSV (default: print the summary only)")
    args = p.parse_args()

    dirs = sorted(d for d in glob.glob(os.path.join(args.case_root, "stable_*"))
                  if os.path.exists(os.path.join(d, "flowsplit_ratio.txt")))
    if args.n_cases:
        dirs = sorted(np.random.default_rng(args.seed).choice(dirs, size=min(args.n_cases, len(dirs)), replace=False))
    rows = []
    for i, d in enumerate(dirs):
        try:
            rows.append(analyse_case(d))
        except Exception as e:
            print(f"[skip] {os.path.basename(d)}: {e}")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(dirs)}", flush=True)
    print(f"\n{len(rows)} cases with flowsplit_ratio.txt (ground truth), rule exponent {EXPONENT}")

    q = np.array([r["q0_file"] for r in rows])
    print(f"file q_outlet0: min {q.min():.3f} median {np.median(q):.3f} max {q.max():.3f}")
    print(f"\n{'surface':24s} | rule vs file: median|d|   p90|d|  within 0.01  corr  | implied exponent: median  IQR")
    for src in ("rim",) + SURFACES + tuple(f"ring_{dd:g}mm" for dd in RING_MM):
        if f"{src}_q0_rule" not in rows[0]:
            continue
        r = np.array([x[f"{src}_q0_rule"] for x in rows])
        k = np.array([x[f"{src}_k_implied"] for x in rows])
        ok = np.isfinite(r)
        r, qq, k = r[ok], q[ok], k[np.isfinite(k)]
        d = np.abs(r - qq)
        print(f"{src:24s} | {np.median(d):21.4f} {np.percentile(d, 90):8.4f} {np.mean(d < 0.01):10.0%} "
              f"{np.corrcoef(r, qq)[0, 1]:+6.3f} | {np.median(k):23.3f}  {np.percentile(k, 25):.2f}-{np.percentile(k, 75):.2f}"
              + (f"   (n={ok.sum()})" if ok.sum() < len(q) else ""))
    worst = sorted(rows, key=lambda x: -abs(x["rim_q0_rule"] - x["q0_file"]))[:8]
    print("\nlargest disagreements (rim areas):")
    for x in worst:
        print(f"  {x['case']:12s} file {x['q0_file']:.3f}  rule {x['rim_q0_rule']:.3f} | "
              f"A0 {x['rim_area0_mm2']:.2f} A1 {x['rim_area1_mm2']:.2f} mm²  implied k {x['rim_k_implied']:.2f}")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-case table -> {args.out}")


if __name__ == "__main__":
    main()
