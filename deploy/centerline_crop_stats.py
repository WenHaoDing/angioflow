"""How much vessel the joint model sees around the aneurysm, measured along centerlines.

New geometry must be clipped to a similar extent before deployment, so this measures it on the
training cases (CFD wall meshes of the extracted AngioFlow-PyG root), separately for each
aneurysm_type (0 bifurcation, 1 sidewall, 2 AngioFlow's third type), per inlet->outlet path:
  upstream    centerline length from the aneurysm anchor back to where the kept wall ends
  downstream  centerline length from the anchor out to where the kept wall ends, per outlet
for the full mesh and for the training crop (focused region + geodesic ring p x max, p in
U[0, 0.25]: reported at 5 evenly spaced p and pooled over them = the training distribution), in mm and in vessel diameters (2 x inscribed-sphere
radius: parent artery near the inlet, each branch near its outlet).

Anchor: the dome apex is the focused-region wall vertex farthest from every centerline; the
anchor is the centerline point (any path) nearest to the apex, and each path is measured from
its own point nearest to that anchor -- the anchor itself on the path(s) passing the aneurysm,
about the bifurcation on a branch that leaves before it. A centerline point counts as kept when
at least half of the wall vertices of its cross-section (within 1.3 r + 0.2 mm) are kept; the
kept extent is the contiguous kept run through the anchor.

Three stages, two envs (vmtk has no torch, the data needs it). Outputs go to --out (never the
data root):
    python -m angioflow.deploy.centerline_crop_stats export --out <dir>          # env new
    ~/miniconda3/envs/vmtk310/bin/python deploy/centerline_crop_stats.py centerlines --out <dir> --workers 12
    python -m angioflow.deploy.centerline_crop_stats stats --out <dir>           # env new
    python -m angioflow.deploy.centerline_crop_stats render --out <dir> --limit 4  # sanity PNGs
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np

ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG"
OUT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_centerlines"
DATASETS = ("angioflow", "aneug_flow", "aneugv2")
FOCUSED, INLET, OUTLET_0, OUTLET_1 = 1, 2, 3, 4
CROP_P = (0.0, 0.0625, 0.125, 0.1875, 0.25)   # evenly over training's p ~ U[0, 0.25]
END_MM = 2.0                       # vessel diameter = median 2r over this length at a path end


# ------------------------------------------------------------------ 1. export (env new)

def export(args):
    import torch
    for tag in args.datasets:
        files = sorted(glob.glob(f"{args.root}/{tag}/static_{tag}_*.pt"))[: args.limit or None]
        os.makedirs(f"{args.out}/mesh/{tag}", exist_ok=True)
        for f in files:
            cid = os.path.basename(f)[len(f"static_{tag}_"):-3]
            dst = f"{args.out}/mesh/{tag}/{cid}.npz"
            if os.path.exists(dst):
                continue
            s = torch.load(f, weights_only=False)
            nt = s["node_type"].numpy()
            pos = s["pos"].double().numpy() * 1e3
            ends = [pos[nt == k].mean(0) for k in (INLET, OUTLET_0, OUTLET_1) if (nt == k).any()]
            np.savez(dst, pos_mm=pos, faces=s["faces"].numpy().astype(np.int64), node_type=nt,
                     aneurysm_type=int(s["aneurysm_type"]),
                     global_geo_dist=s["global_geo_dist"].numpy(), endpoints=np.stack(ends))
        print(f"[export] {tag}: {len(files)} cases -> {args.out}/mesh/{tag}", flush=True)


# ------------------------------------------------------------------ 2. centerlines (env vmtk310)

def _centerlines_case(job):
    src, dst = job
    if os.path.exists(dst):
        return dst, "skip"
    try:
        import pyvista as pv
        from vmtk import vmtkscripts
        from vtk.util.numpy_support import vtk_to_numpy
        d = np.load(src)
        f = d["faces"]
        surf = pv.PolyData(d["pos_mm"], np.hstack([np.full((len(f), 1), 3), f]).ravel())
        cap = vmtkscripts.vmtkSurfaceCapper()
        cap.Surface, cap.Method, cap.Interactive = surf, "centerpoint", 0
        cap.Execute()
        ep = d["endpoints"]
        cl = vmtkscripts.vmtkCenterlines()               # AneuG geometry/vmtk_centerlines.py settings
        cl.Surface = cap.Surface
        cl.SeedSelectorName = "pointlist"
        cl.SourcePoints = ep[0].tolist()
        cl.TargetPoints = [x for pt in ep[1:] for x in pt.tolist()]
        cl.AppendEndPoints = 1
        cl.Execute()
        rs = vmtkscripts.vmtkCenterlineResampling()
        rs.Centerlines, rs.Length = cl.Centerlines, 0.1
        rs.Execute()
        sm = vmtkscripts.vmtkCenterlineSmoothing()
        sm.Centerlines, sm.NumberOfSmoothingIterations, sm.SmoothFactor = rs.Centerlines, 100, 0.1
        sm.Execute()
        poly = sm.Centerlines
        pts = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
        rad = vtk_to_numpy(poly.GetPointData().GetArray("MaximumInscribedSphereRadius")).astype(float)
        out = {}
        for i in range(poly.GetNumberOfCells()):
            ids = poly.GetCell(i).GetPointIds()
            idx = np.array([ids.GetId(k) for k in range(ids.GetNumberOfIds())])
            p = pts[idx]
            if np.linalg.norm(p[0] - ep[0]) > np.linalg.norm(p[-1] - ep[0]):
                idx = idx[::-1]                            # inlet first
            k = int(np.argmin(np.linalg.norm(ep[1:] - pts[idx[-1]], axis=1)))   # which outlet
            out[f"path{k}_pts"], out[f"path{k}_r"] = pts[idx], rad[idx]
        np.savez(dst, n_paths=len(ep) - 1, **out)
        return dst, "ok"
    except Exception:
        return dst, "FAIL " + traceback.format_exc(limit=2).replace("\n", " | ")


def centerlines(args):
    jobs = []
    for tag in args.datasets:
        os.makedirs(f"{args.out}/centerline/{tag}", exist_ok=True)
        for src in sorted(glob.glob(f"{args.out}/mesh/{tag}/*.npz"))[: args.limit or None]:
            jobs.append((src, f"{args.out}/centerline/{tag}/{os.path.basename(src)}"))
    n_fail = 0
    with Pool(args.workers) as pool:
        for i, (dst, status) in enumerate(pool.imap_unordered(_centerlines_case, jobs), 1):
            if status.startswith("FAIL"):
                n_fail += 1
                print(f"[fail] {dst}: {status}", flush=True)
            if i % 50 == 0 or i == len(jobs):
                print(f"[centerlines] {i}/{len(jobs)} done, {n_fail} failed", flush=True)


# ------------------------------------------------------------------ 3. stats (env new)

def case_stats(mesh_npz, cl_npz):
    from scipy.spatial import cKDTree
    d, c = np.load(mesh_npz), np.load(cl_npz)
    pos, nt, g = d["pos_mm"], d["node_type"], d["global_geo_dist"]
    paths = [(c[f"path{k}_pts"], c[f"path{k}_r"]) for k in range(int(c["n_paths"])) if f"path{k}_pts" in c]
    if not paths:
        raise ValueError("no centerline paths")
    allpts = np.concatenate([p for p, _ in paths])
    foc = np.flatnonzero(nt == FOCUSED)
    dist_cl = cKDTree(allpts).query(pos[foc])[0]
    apex = pos[foc[np.argmax(dist_cl)]]
    anchor = allpts[cKDTree(allpts).query(apex)[1]]
    tree = cKDTree(pos)
    out = dict(dome_height_mm=float(dist_cl.max()), paths=[])
    for k, (p, r) in enumerate(paths):
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        a = int(np.argmin(np.linalg.norm(p - anchor, axis=1)))
        nbrs = tree.query_ball_point(p, 1.3 * r + 0.2)
        d_in = 2 * np.median(r[s <= END_MM])
        d_out = 2 * np.median(r[s >= s[-1] - END_MM])
        rec = dict(outlet=k, length_mm=float(s[-1]), anchor_mm=float(s[a]),
                   d_parent_mm=float(d_in), d_branch_mm=float(d_out))
        for name, p_crop in [("full", None)] + [(f"p{pc:g}", pc) for pc in CROP_P]:
            keep = np.ones(len(pos), bool) if p_crop is None else (nt == FOCUSED) | (g <= p_crop * g.max())
            kept = np.array([len(n) > 0 and keep[n].mean() >= 0.5 for n in nbrs])
            if not kept[a]:
                rec[name] = dict(up_mm=0.0, down_mm=0.0)
                continue
            lo = a - np.argmin(kept[a::-1]) + 1 if not kept[: a + 1].all() else 0
            hi = a + np.argmin(kept[a:]) - 1 if not kept[a:].all() else len(p) - 1
            rec[name] = dict(up_mm=float(s[a] - s[lo]), down_mm=float(s[hi] - s[a]),
                             up_diam=float((s[a] - s[lo]) / d_in), down_diam=float((s[hi] - s[a]) / d_out))
        out["paths"].append(rec)
    return out


def _stats_job(job):
    tag, cid, m, cl, root = job
    try:
        r = case_stats(m, cl)
    except Exception as e:
        r = {"error": str(e)}
    d = np.load(m)
    if "aneurysm_type" in d:
        r["aneurysm_type"] = int(d["aneurysm_type"])
    else:                                       # meshes exported before the type was added
        import torch
        r["aneurysm_type"] = int(torch.load(f"{root}/{tag}/static_{tag}_{cid}.pt",
                                            weights_only=False)["aneurysm_type"])
    return tag, cid, r


def _q(x):
    x = np.asarray(x, float)
    return f"{np.median(x):6.2f} [{np.percentile(x, 10):5.2f}, {np.percentile(x, 90):5.2f}]"


def _summary(label, rows):
    """Lines for one group of cases. up: the longest path's (the trunk through the aneurysm);
    down: every outlet path pooled."""
    ok = [r for r in rows if "error" not in r]
    n_out = np.bincount([len(r["paths"]) for r in ok], minlength=3)[1:]
    lines = [f"== {label}: {len(ok)} cases ({len(rows) - len(ok)} failed; "
             f"{n_out[0]} single-, {n_out[1]} two-outlet)",
             f"  parent diameter mm {_q([r['paths'][0]['d_parent_mm'] for r in ok])}   "
             f"branch diameter mm {_q([p['d_branch_mm'] for r in ok for p in r['paths']])}   "
             f"dome height mm {_q([r['dome_height_mm'] for r in ok])}"]
    crops = [f"p{pc:g}" for pc in CROP_P]
    for label, names in [("full", ["full"])] + [(n, [n]) for n in crops] + [("train", crops)]:
        up = [max(p[n]["up_mm"] for p in r["paths"]) for r in ok for n in names]
        upd = [max(p[n].get("up_diam", 0) for p in r["paths"]) for r in ok for n in names]
        dn = [p[n]["down_mm"] for r in ok for p in r["paths"] for n in names]
        dnd = [p[n].get("down_diam", 0) for r in ok for p in r["paths"] for n in names]
        lines.append(f"  {label:7s} up {_q(up)} mm = {_q(upd)} D_parent | down {_q(dn)} mm = {_q(dnd)} D_branch")
    return lines


def stats(args):
    jobs = []
    for tag in args.datasets:
        for cl in sorted(glob.glob(f"{args.out}/centerline/{tag}/*.npz"))[: args.limit or None]:
            cid = os.path.basename(cl)[:-4]
            jobs.append((tag, cid, f"{args.out}/mesh/{tag}/{cid}.npz", cl, args.root))
    with Pool(args.workers) as pool:
        res = pool.map(_stats_job, jobs)
    table = {}
    for tag, cid, r in res:
        table.setdefault(tag, {})[cid] = r
    with open(f"{args.out}/centerline_crop_stats.json", "w") as f:
        json.dump(table, f, indent=1)
    lines = [f"per-case results: {args.out}/centerline_crop_stats.json",
             "median [p10, p90]. up = anchor -> inlet along the trunk; down = anchor -> each outlet "
             "(outlet paths pooled). full = whole training mesh; pX = training crop at depth p=X; "
             "train = pooled over the 5 depths = what training feeds the model (p ~ U[0, 0.25]). D = vessel diameter at that end (2 x inscribed radius)."]
    for at in sorted({r["aneurysm_type"] for t in table.values() for r in t.values()}):
        tags = [tag for tag in table if any(r["aneurysm_type"] == at for r in table[tag].values())]
        lines += [""] + _summary(f"type {at}, all datasets",
                                 [r for tag in tags for r in table[tag].values() if r["aneurysm_type"] == at])
        if len(tags) > 1:
            for tag in tags:
                lines += _summary(f"type {at}, {tag}",
                                  [r for r in table[tag].values() if r["aneurysm_type"] == at])
    text = "\n".join(lines)
    with open(f"{args.out}/centerline_crop_stats.txt", "w") as f:
        f.write(text + "\n")
    print(text)

def render(args):
    """PNG per case: wall (light) with the p=0.25 crop (dark), centerlines coloured by kept
    (green) / dropped (red) at p=0.25, the dome apex (star) and each path's anchor (circle)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial import cKDTree
    os.makedirs(f"{args.out}/sanity", exist_ok=True)
    for tag in args.datasets:
        for cl in sorted(glob.glob(f"{args.out}/centerline/{tag}/*.npz"))[: args.limit or 4]:
            cid = os.path.basename(cl)[:-4]
            d, c = np.load(f"{args.out}/mesh/{tag}/{cid}.npz"), np.load(cl)
            pos, nt, g = d["pos_mm"], d["node_type"], d["global_geo_dist"]
            keep = (nt == FOCUSED) | (g <= CROP_P[-1] * g.max())
            ax3 = np.linalg.svd(pos - pos.mean(0), full_matrices=False)[2][:2]
            P = (pos - pos.mean(0)) @ ax3.T
            paths = [(c[f"path{k}_pts"], c[f"path{k}_r"]) for k in range(int(c["n_paths"])) if f"path{k}_pts" in c]
            foc = np.flatnonzero(nt == FOCUSED)
            allpts = np.concatenate([p for p, _ in paths])
            apex = pos[foc[np.argmax(cKDTree(allpts).query(pos[foc])[0])]]
            anchor = allpts[cKDTree(allpts).query(apex)[1]]
            tree = cKDTree(pos)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.scatter(P[~keep, 0], P[~keep, 1], s=0.2, c="#dddddd")
            ax.scatter(P[keep, 0], P[keep, 1], s=0.2, c="#9a9a9a")
            for p, r in paths:
                kept = np.array([len(n) > 0 and keep[n].mean() >= 0.5 for n in tree.query_ball_point(p, 1.3 * r + 0.2)])
                Q = (p - pos.mean(0)) @ ax3.T
                ax.scatter(Q[:, 0], Q[:, 1], s=2, c=np.where(kept, "#1b9e3a", "#d62728"))
                a = int(np.argmin(np.linalg.norm(p - anchor, axis=1)))
                ax.scatter(*Q[a], s=60, facecolors="none", edgecolors="k")
            A = (apex - pos.mean(0)) @ ax3.T
            ax.scatter(*A, s=150, marker="*", c="#ff7f0e", edgecolors="k")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{tag}/{cid}: crop p=0.25 dark, centerline kept green / dropped red", fontsize=9)
            fig.tight_layout(); fig.savefig(f"{args.out}/sanity/{tag}_{cid}.png", dpi=100); plt.close(fig)
    print(f"[render] -> {args.out}/sanity")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["export", "centerlines", "stats", "render"])
    p.add_argument("--root", default=ROOT)
    p.add_argument("--out", default=OUT)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--limit", type=int, default=0, help="first N cases per dataset (0 = all)")
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    {"export": export, "centerlines": centerlines, "stats": stats, "render": render}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
