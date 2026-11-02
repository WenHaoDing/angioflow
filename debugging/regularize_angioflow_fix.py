"""Smooth the CFD-rerun surfaces with the mesh regularizer, exactly as
AneuG/cfd_mesher/generate_cfd_volume_meshes.py does before volume meshing.

Per case folder under --root (default cfd/angioflow_fix):
    ghd_smoothed_reconstruction.obj     input surface (ImperialNHS cases)
    merged_reconstruction_remeshed.obj  input surface where the above is absent (AneuX cases)
    reg_mesh.obj                       output: the regularized surface, or the input
                                        unchanged if the regularizer did not act on it
                                        (see the summary's status column)
    reg_mesh.vtp                        the regularizer's own output (regularized cases)
    reg_mesh_regularization.json        its report -- deliberately NOT regularization.json,
                                        which the volume mesher caches and would reuse
                                        for a different input

Same call and the same settings as the mesher (its build_parser defaults, read from the
module so they stay in sync): volume_mesh.regularize_surface_if_needed, dome-normalized
with the case's sac size from ImperialNHS_v2/<case>/dome_size.npy, snapping to the next
level below where the surface measures (-1 -0.5 -0.25 0 0.25 SD), the 'relaxed' stopping
preset, and the regularized surface rejected if an opening moves more than 1 mm. A case
with no sac size uses the millimetre model at 0.25 SD. Status per case:
    regularized   smoothed; reg_mesh.obj is the smoothed surface
    none          already below the lowest level; reg_mesh.obj = the input
    rejected      smoothing moved an opening / changed the openings; reg_mesh.obj = the input
    no_source     none of the input surfaces in the folder
    failed        exception (in the summary)
Summary: <root>/reg_mesh_summary.csv.

Needs the vmtk_autogen env (mesh_regularizer, trimesh, pyvista, vmtk):
    conda activate vmtk_autogen
    python debugging/regularize_angioflow_fix.py --workers 8
    python debugging/regularize_angioflow_fix.py --cases 0m5cVmE8R1_aneurysm1
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

ANEUG = "/media/yaplab2/HDD Storage/wenhao/AneuG"
ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/angioflow_fix"
SOURCES = ("ghd_smoothed_reconstruction.obj",       # ImperialNHS
           "merged_reconstruction_remeshed.obj")    # AneuX
OUT = "reg_mesh.obj"
FIELDS = ["case", "status", "source", "mode", "dome_sac_mm", "roughness_before_sd", "target_sd",
          "cap_shift_max_mm", "rounds", "seconds", "reason"]


def run_case(job):
    case, root, overwrite = job
    d = os.path.join(root, case)
    out = os.path.join(d, OUT)
    row = dict(case=case)
    name = next((s for s in SOURCES if os.path.exists(os.path.join(d, s))), None)
    if name is None:
        return {**row, "status": "no_source", "reason": f"none of {SOURCES}"}
    src = os.path.join(d, name)
    row["source"] = name
    if os.path.exists(out) and not overwrite:
        return {**row, "status": "exists", "reason": f"{OUT} exists (--overwrite to redo)"}
    sys.path.insert(0, ANEUG)
    import trimesh
    from cfd_mesher import volume_mesh as vm
    from cfd_mesher.generate_cfd_volume_meshes import _dome_sac_mm, build_parser
    a = build_parser().parse_args([])                    # the mesher's own defaults
    sac = _dome_sac_mm(case, a.dome_size_root) if a.dome_normalized else None
    t0 = time.time()
    try:
        used, report = vm.regularize_surface_if_needed(
            surface_file=src, output_file=os.path.join(d, "reg_mesh.vtp"),
            report_path=os.path.join(d, "reg_mesh_regularization.json"),
            tolerance=a.dome_tolerance if sac else a.regularizer_tolerance,
            remesher=a.regularizer_remesher, max_cap_shift=a.max_cap_shift,
            overwrite=overwrite, dome_sac_mm=sac, stopping=a.regularizer_stopping,
            snap_levels=a.dome_snap_levels if sac else None,
            fallback_tolerance=a.regularizer_tolerance)
        surf = vm.read_surface(used)                     # smoothed vtp, or the merged input
        faces = surf.faces.reshape(-1, 4)[:, 1:]
        trimesh.Trimesh(surf.points, faces, process=False).export(out)
    except Exception as e:
        return {**row, "status": "failed", "seconds": round(time.time() - t0), "reason": repr(e)}
    shifts = report.get("cap_shift_mm") or []
    return {**row, "status": report["action"], "mode": report["mode"], "dome_sac_mm": sac,
            "roughness_before_sd": round(report["assessment"]["worst_target_excess_sd"], 3),
            "target_sd": report["tolerance"],
            "cap_shift_max_mm": round(max(shifts), 3) if shifts else "",
            "rounds": (report.get("forward") or {}).get("rounds", ""),
            "seconds": round(time.time() - t0), "reason": report.get("reason", "")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--cases", nargs="*", default=None, help="default: every folder under --root")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true", help=f"redo cases that already have {OUT}")
    args = p.parse_args()
    cases = args.cases or sorted(c for c in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, c)) and c != "sanity")
    jobs = [(c, args.root, args.overwrite) for c in cases]
    print(f"{len(cases)} case(s) under {args.root}, {args.workers} worker(s)", flush=True)

    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_case, jobs), 1):
            rows.append(r)
            if r["status"] not in ("no_source", "exists"):
                print(f"[{i}/{len(jobs)}] {r['case']}: {r['status']}"
                      + (f" ({r.get('mode')}, {r.get('roughness_before_sd')} SD -> target {r.get('target_sd')}, "
                         f"{r.get('seconds')} s)" if r["status"] != "failed" else f" {r['reason']}"), flush=True)

    summary = os.path.join(args.root, "reg_mesh_summary.csv")
    old = {}
    if os.path.exists(summary):                          # keep rows of cases not run this time
        with open(summary) as f:
            old = {r["case"]: r for r in csv.DictReader(f)}
    old.update({r["case"]: r for r in rows if r["status"] != "exists"})
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(old[c] for c in sorted(old))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"\n-> {summary}")
    for r in rows:
        if r["status"] == "no_source":
            print(f"  no input surface: {r['case']}")


if __name__ == "__main__":
    main()
