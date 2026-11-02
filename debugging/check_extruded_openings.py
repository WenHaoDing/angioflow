"""Check the openings of hand-extruded geomagic_processed.obj surfaces BEFORE volume
meshing, and write a fusion npz the volume mesher will label them correctly with.

Why. AneuG/cfd_mesher/generate_cfd_volume_meshes.py labels the mesh caps with
volume_mesh.relabel_inlet_outlets_v2: the reference for each branch is the LAST point of
its centerline in the fusion npz (cpcd_glo[i][-1], which sits on the original cap), and
the caps are taken greedily -- inlet first, then outlet 1, outlet 2 -- each grabbing the
NEAREST remaining cap centroid. Extruding an opening moves its cap several mm away from
that point (3.6-7.7 mm measured on these cases), the same scale as the gap between
openings, so a reference can grab the wrong cap: inlet/outlet or the two outlets swapped,
silently -- which also swaps the flow split and puts the inflow on the wrong face.

What this does, per case under --root:
  1. reads geomagic_processed.obj the way the mesher does (volume_mesh.read_surface) and
     finds its openings with the mesher's own boundary extractor;
  2. COUNT: openings must equal the npz branch count (2 or 3), as the mesher requires;
  3. MATCH (robust): each branch's centerline is extended past its end along its end
     tangent (cpcd_glo_tangent); a cap belongs to the branch whose extended centerline it
     sits on. Cost = distance of the cap centroid from that ray (+ a penalty if the cap is
     behind the branch end); one global optimal assignment (Hungarian), and its margin
     over the next-best assignment;
  4. compares with what the mesher's greedy rule would pick from the ORIGINAL npz;
  5. writes <case>/fusion_info_extruded.npz: the original npz with each branch's
     centerline continued (0.278 mm spacing, as stored) up to its matched cap, so its last
     point IS the cap centroid -- then re-runs the mesher's greedy rule against it to
     prove it now picks the robust assignment;
  6. renders <root>/sanity/<status>/<case>.png: three views of the surface, centerlines
     and caps labelled inlet / outlet1 / outlet2 in the order the mesher will use
     (plus <root>/sanity/opening_check.csv). The volume mesher would list sanity/ as a
     case and skip it ("no geomagic_processed.obj") -- harmless, or pass --exclude sanity.

Status:
  ok     counts match; robust and greedy (original npz) agree; nothing odd
  fixed  counts match; greedy on the original npz would mislabel -- the written npz fixes it
  check  counts match but something is unusual (a cap far off its branch's axis, an
         ambiguous assignment, a cap behind its branch end, a cap plane oblique to the
         branch) -- look at the png
  fail   opening count != branch count (or no npz / surface) -- the mesher would stop;
         no npz written
Then mesh with the written npz for every non-fail case:
    python -m cfd_mesher.generate_cfd_volume_meshes --surface_root <root> \\
        --npz_filename fusion_info_extruded.npz ...

Needs the vmtk_autogen env:
    conda activate vmtk_autogen
    python debugging/check_extruded_openings.py
    python debugging/check_extruded_openings.py --cases 0m5cVmE8R1_aneurysm1 C0064_cut2
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys

import numpy as np

ANEUG = "/media/yaplab2/HDD Storage/wenhao/AneuG"
ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/angioflow_fix"
SURFACE = "geomagic_processed.obj"
NPZ_CANDIDATES = ("ghd_forward_fusion_info.npz", "forward_fusion_info.npz")   # ImperialNHS, AneuX
OUT_NPZ = "fusion_info_extruded.npz"
SANITY_DIR = "sanity"    # <root>/sanity/<status>/<case>.png -- not a case folder; skipped when listing cases
SPACING = 0.278          # mm, point spacing of the stored centerlines
BEHIND_PENALTY = 5.0     # cost per mm a cap sits behind its branch end
PERP_WARN = 1.5          # mm off the extended centerline
BEHIND_WARN = 1.0        # mm behind the branch end (trimmed instead of extruded?)
ANGLE_WARN = 35.0        # deg between cap plane normal and branch tangent
MARGIN_WARN = 3.0        # mm: next-best assignment must cost at least this much more
ROLES = ("inlet", "outlet1", "outlet2")
FIELDS = ["case", "status", "npz", "n_branches", "n_openings", "greedy_agrees", "margin_mm",
          "extrusion_mm", "perp_mm", "angle_deg", "notes"]


def _mesher():
    sys.path.insert(0, ANEUG)
    from cfd_mesher import volume_mesh as vm
    return vm


def openings(vm, surface):
    """Mesher's own boundary extractor: centroid, area, plane normal per opening."""
    out = []
    for b in vm.get_open_boundary_info_from_surface(surface):
        out.append(dict(centroid=np.asarray(b["centroid"], float), area=b["area"]))
    # plane normals from the boundary points
    from vmtk import vtkvmtk
    ext = vtkvmtk.vtkvmtkPolyDataBoundaryExtractor()
    ext.SetInputData(surface); ext.Update()
    loops = ext.GetOutput()
    for i in range(loops.GetNumberOfCells()):
        pts = vm._points_from_vtk_cell(loops.GetCell(i))
        q = pts - pts.mean(0)
        out[i]["normal"] = np.linalg.svd(q, full_matrices=False)[2][-1]
        out[i]["radius"] = float(np.linalg.norm(q, axis=1).mean())
    return out


def greedy(refs, caps):
    """The mesher's rule: refs in order, each takes the nearest remaining cap."""
    used, pick = set(), []
    for r in refs:
        j = min((j for j in range(len(caps)) if j not in used), key=lambda j: np.linalg.norm(r - caps[j]))
        used.add(j); pick.append(j)
    return pick


def ray_cost(end, tangent, c):
    v = c - end
    along = float(v @ tangent)
    perp = float(np.linalg.norm(v - along * tangent))
    return perp + BEHIND_PENALTY * max(0.0, -along), along, perp


def check_case(vm, case, root, write=True):
    d = os.path.join(root, case)
    row = dict(case=case)
    npz_name = next((n for n in NPZ_CANDIDATES if os.path.exists(os.path.join(d, n))), None)
    if npz_name is None or not os.path.exists(os.path.join(d, SURFACE)):
        return {**row, "status": "fail", "notes": f"missing {SURFACE if npz_name else 'fusion npz'}"}, None
    row["npz"] = npz_name
    z = np.load(os.path.join(d, npz_name), allow_pickle=True)
    cp = [np.asarray(b, float) for b in z["cpcd_glo"]]
    tg = [np.asarray(t, float) for t in z["cpcd_glo_tangent"]]
    ends = [b[-1] for b in cp]
    tans = [t[-1] / np.linalg.norm(t[-1]) for t in tg]
    surface = vm.read_surface(os.path.join(d, SURFACE))
    ops = openings(vm, surface)
    row.update(n_branches=len(cp), n_openings=len(ops))
    if len(ops) != len(cp):
        return {**row, "status": "fail",
                "notes": f"{len(ops)} openings on {SURFACE} vs {len(cp)} branches in {npz_name}"}, (surface, ops, cp, None)
    caps = [o["centroid"] for o in ops]

    # robust: best global assignment of caps to branches by extended-centerline distance
    n = len(cp)
    cost = np.array([[ray_cost(ends[i], tans[i], caps[j])[0] for j in range(n)] for i in range(n)])
    perms = sorted(itertools.permutations(range(n)), key=lambda p: sum(cost[i, p[i]] for i in range(n)))
    best = list(perms[0])
    margin = (sum(cost[i, perms[1][i]] for i in range(n)) - sum(cost[i, best[i]] for i in range(n))
              if len(perms) > 1 else np.inf)
    along = [ray_cost(ends[i], tans[i], caps[best[i]])[1] for i in range(n)]
    perp = [ray_cost(ends[i], tans[i], caps[best[i]])[2] for i in range(n)]
    angle = [float(np.degrees(np.arccos(min(1.0, abs(ops[best[i]]["normal"] @ tans[i]))))) for i in range(n)]
    g = greedy(ends, caps)
    agrees = g == best
    row.update(greedy_agrees=agrees, margin_mm=round(float(margin), 2),
               extrusion_mm=" ".join(f"{a:.2f}" for a in along),
               perp_mm=" ".join(f"{p:.2f}" for p in perp),
               angle_deg=" ".join(f"{a:.0f}" for a in angle))
    notes = []
    for i in range(n):
        if perp[i] > PERP_WARN:
            notes.append(f"{ROLES[i]} cap {perp[i]:.1f} mm off its branch axis")
        if along[i] < -BEHIND_WARN:
            notes.append(f"{ROLES[i]} cap {-along[i]:.1f} mm behind its branch end")
        if angle[i] > ANGLE_WARN:
            notes.append(f"{ROLES[i]} cap plane {angle[i]:.0f} deg oblique to the branch")
    if margin < MARGIN_WARN:
        notes.append(f"assignment margin only {margin:.1f} mm")
    if not agrees:
        notes.append("mesher's greedy rule on the original npz would mislabel: "
                     + ", ".join(f"{ROLES[i]}->cap{g[i]} (should be cap{best[i]})" for i in range(n) if g[i] != best[i]))
    status = "check" if any("off its" in s or "behind" in s or "oblique" in s or "margin" in s for s in notes) \
        else ("ok" if agrees else "fixed")

    if write:
        new_cp, new_tg = [], []
        for i in range(n):
            target = caps[best[i]]
            seg = target - ends[i]
            L = float(np.linalg.norm(seg))
            k = int(np.ceil(L / SPACING))
            extra = [ends[i] + seg * (s / k) for s in range(1, k + 1)] if k else []
            dirv = seg / L if L > 1e-9 else tans[i]
            new_cp.append(np.vstack([cp[i]] + extra) if extra else cp[i].copy())
            new_tg.append(np.vstack([tg[i]] + [dirv] * len(extra)) if extra else tg[i].copy())
        data = {k: z[k] for k in z.files}
        obj = lambda lst: np.array(lst + [None], dtype=object)[:-1]          # keep a 1-D object array
        data["cpcd_glo"], data["cpcd_glo_tangent"] = obj(new_cp), obj(new_tg)
        np.savez(os.path.join(d, OUT_NPZ), **data)
        # prove the mesher's own rule now picks the robust assignment from the written npz
        zz = np.load(os.path.join(d, OUT_NPZ), allow_pickle=True)
        if greedy([np.asarray(b, float)[-1] for b in zz["cpcd_glo"]], caps) != best:
            status, notes = "fail", notes + ["written npz still mislabels under the greedy rule"]
    return {**row, "status": status, "notes": "; ".join(notes)}, (surface, ops, cp, best)


def render(case, root, out_dir, row, geo):
    import pyvista as pv
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    surface, ops, cp, best = geo
    colors = ("red", "limegreen", "royalblue")
    # three views side by side -- one view can hide a cap behind the vessel
    pl = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(2700, 900), border=False)
    for k, (view, name) in enumerate((("iso", "isometric"), ("xz", "front (xz)"), ("yz", "side (yz)"))):
        pl.subplot(0, k)
        pl.add_mesh(surface, color="lightgrey", opacity=0.35)
        for i, b in enumerate(cp):
            pl.add_mesh(pv.Spline(b, max(len(b), 2)) if len(b) > 1 else pv.PolyData(b), color=colors[i], line_width=4)
        for j, o in enumerate(ops):
            role = ROLES[best.index(j)] if best is not None and j in best else f"cap{j}"
            col = colors[best.index(j)] if best is not None and j in best else "black"
            pl.add_mesh(pv.Sphere(radius=max(0.4, 0.25 * o["radius"]), center=o["centroid"]), color=col)
            pl.add_point_labels([o["centroid"]], [role], font_size=30, text_color=col, shape_opacity=0.6,
                                always_visible=True)
        if k == 0:
            pl.add_text(f"{case}  [{row['status']}]  cap offset along branch {row.get('extrusion_mm', '')} mm\n"
                        f"{row.get('notes', '')[:150]}", font_size=22, position="upper_left")
        else:
            pl.add_text(name, font_size=22, position="upper_left")
        pl.view_isometric() if view == "iso" else (pl.view_xz() if view == "xz" else pl.view_yz())
    os.makedirs(os.path.join(out_dir, row["status"]), exist_ok=True)
    pl.screenshot(os.path.join(out_dir, row["status"], f"{case}.png")); pl.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--out", default=None,
                   help=f"renders (<out>/<status>/<case>.png) + summary csv (default: <root>/{SANITY_DIR})")
    p.add_argument("--no_render", action="store_true")
    p.add_argument("--no_write", action="store_true", help=f"check only, don't write {OUT_NPZ}")
    args = p.parse_args()
    out = args.out or os.path.join(args.root, SANITY_DIR)
    os.makedirs(out, exist_ok=True)
    vm = _mesher()
    cases = args.cases or sorted(c for c in os.listdir(args.root)
                                 if os.path.isdir(os.path.join(args.root, c)) and c != SANITY_DIR)
    rows = []
    for c in cases:
        try:
            row, geo = check_case(vm, c, args.root, write=not args.no_write)
        except Exception as e:
            row, geo = dict(case=c, status="fail", notes=f"error: {e!r}"), None
        rows.append(row)
        print(f"[{row['status']:5s}] {c:40s} {row.get('notes', '')}", flush=True)
        if geo is not None and not args.no_render:
            try:
                render(c, args.root, out, row, geo)
            except Exception as e:
                print(f"        render failed: {e!r}", flush=True)
    with open(os.path.join(out, "opening_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"-> {os.path.join(out, 'opening_check.csv')}  (pngs by status under {out}/)")


if __name__ == "__main__":
    main()
