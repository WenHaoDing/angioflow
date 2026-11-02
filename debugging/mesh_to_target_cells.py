"""Volume-mesh every case to a target cell count (~2.75 M), with the UNMODIFIED
AneuG/cfd_mesher/generate_cfd_volume_meshes.py, one mesher run per case.

The mesher picks a single edge length from shape volume/area against references tuned
on AneuGv2; on the extruded angioflow_fix surfaces that gave a 2.44 M mean (1.72-3.05 M).
Here each case gets its own edge instead. Fitted on the first 39 meshes:

    cells ~= 158.4 * A / e^2         (A = open-surface wall area, mm^2; e = edge, mm)
            mean error 4.8 %, max 14 % -- a volume term (even the true capped volume)
            does not help: the boundary layer + near-wall tets scale with the wall.
so  e = sqrt(158.4 * A / target).

Per case: e is passed as --base_edge with --ref_volume/--ref_area huge (the mesher's
adaptive coarsening never triggers, so it meshes at exactly e). The cell count is read
from the mesher's own "Mesh stats" line; outside [--low, --high] the case is re-meshed
(--overwrite) at e * sqrt(cells / target) (cells ~ 1/e^2), up to --max_attempts. A case
already meshed with >= --low cells is kept (only meshes below the band are redone;
--redo_all to remesh everything).

Fixed mesher options: --surface_root = --save_root = --root (in place; the config default
writes to cfd/AneuGv2), --npz_filename fusion_info_extruded.npz, --no-regularize. The
surface it meshes is the open geomagic_processed.obj (mesher default); A is measured on
that same open surface.

    conda activate vmtk_autogen
    python debugging/mesh_to_target_cells.py --dry_run         # print per-case edges, mesh nothing
    python debugging/mesh_to_target_cells.py --workers 5
Outputs: <logs>/<case>.log (all attempts), <logs>/mesh_to_target.csv, <logs>/TARGET_DONE.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ANEUG = "/media/yaplab2/HDD Storage/wenhao/AneuG"
ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/angioflow_fix"
LOGS = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/angioflow_fix_meshing_logs/target"
K_AREA = 158.4
STATS = re.compile(r"Mesh stats: (\d+) nodes, (\d+) volume cells, (\d+) surface faces")
FIELDS = ["case", "area_mm2", "status", "cells_M", "edge_mm", "attempts", "history", "seconds", "error"]


def wall_area(case, root):
    sys.path.insert(0, ANEUG)
    from cfd_mesher import volume_mesh as vm
    return float(vm.read_surface(os.path.join(root, case, "geomagic_processed.obj")).area)


def existing_cells(case, root):
    p = os.path.join(root, case, "mesh.vtu")
    if not (os.path.exists(p) and os.path.exists(os.path.join(root, case, "mesh.msh"))):
        return None
    import vtk
    r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(p); r.Update()
    g = r.GetOutput()
    return sum(g.GetCellType(i) in (10, 13, 12, 14) for i in range(g.GetNumberOfCells()))  # tet, wedge, hex, pyramid


def run_mesher(case, edge, args, attempt):
    cmd = [sys.executable, "-u", "-m", "cfd_mesher.generate_cfd_volume_meshes",
           "--surface_root", args.root, "--save_root", args.root,
           "--npz_filename", "fusion_info_extruded.npz", "--no-regularize",
           "--base_edge", f"{edge:.4f}", "--ref_volume", "1e12", "--ref_area", "1e12",
           "--overwrite", "--cases", case, "--run_tag", f"target_{case}"]
    if args.mesh_timeout:
        cmd += ["--mesh_timeout", str(args.mesh_timeout)]
    with open(os.path.join(args.logs, f"{case}.log"), "a") as log:
        log.write(f"\n===== attempt {attempt}: edge {edge:.4f} mm  ({time.strftime('%F %T')})\n{' '.join(cmd)}\n")
        log.flush()
        subprocess.run(cmd, cwd=ANEUG, stdout=log, stderr=subprocess.STDOUT)
    text = open(os.path.join(args.logs, f"{case}.log"), errors="ignore").read().split(f"===== attempt {attempt}:")[-1]
    m = STATS.findall(text)
    failed = re.findall(r"^\s+failed: (.*)$", text, re.M)
    return (int(m[-1][1]) if m else None), (failed[-1] if failed else None)


def do_case(case, area, args):
    t0 = time.time()
    row = dict(case=case, area_mm2=round(area, 1))
    if not args.redo_all:
        n = existing_cells(case, args.root)
        if n is not None and n >= args.low:              # existing mesh: redo only if below the band
            return {**row, "status": "kept", "cells_M": round(n / 1e6, 3), "attempts": 0,
                    "history": f"existing {n / 1e6:.2f}M"}
    edge = math.sqrt(K_AREA * area / args.target)
    history = []
    for attempt in range(1, args.max_attempts + 1):
        n, err = run_mesher(case, edge, args, attempt)
        if n is None:
            return {**row, "status": "failed", "edge_mm": round(edge, 4), "attempts": attempt,
                    "history": " ".join(history), "seconds": round(time.time() - t0), "error": err or "no Mesh stats line"}
        history.append(f"{edge:.4f}->{n / 1e6:.2f}M")
        if args.low <= n <= args.high:
            break
        edge = edge * math.sqrt(n / args.target)
    status = "ok" if args.low <= n <= args.high else ("below_floor" if n < args.floor else "out_of_band")
    return {**row, "status": status, "cells_M": round(n / 1e6, 3), "edge_mm": history[-1].split("->")[0],
            "attempts": len(history), "history": " ".join(history), "seconds": round(time.time() - t0)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--logs", default=LOGS)
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--target", type=float, default=2.75e6)
    p.add_argument("--low", type=float, default=2.5e6, help="accept band low")
    p.add_argument("--high", type=float, default=3.0e6, help="accept band high")
    p.add_argument("--floor", type=float, default=2.25e6, help="hard minimum (reported)")
    p.add_argument("--max_attempts", type=int, default=3)
    p.add_argument("--mesh_timeout", type=float, default=None,
                   help="passed to the mesher (its default: 2700 s per vmtkmeshgenerator run)")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--redo_all", action="store_true", help="remesh even cases already in the band")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    os.makedirs(args.logs, exist_ok=True)
    cases = args.cases or sorted(c for c in os.listdir(args.root)
                                 if os.path.isdir(os.path.join(args.root, c)) and c != "sanity")
    areas = {c: wall_area(c, args.root) for c in cases}
    if args.dry_run:
        for c in cases:
            e = math.sqrt(K_AREA * areas[c] / args.target)
            print(f"{c:40s} A {areas[c]:7.1f} mm2  edge {e:.4f} mm")
        return
    print(f"{len(cases)} case(s), {args.workers} workers, target {args.target / 1e6:.2f}M "
          f"(accept {args.low / 1e6:.2f}-{args.high / 1e6:.2f}M)", flush=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(do_case, c, areas[c], args): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:
                r = dict(case=futs[f], status="failed", error=repr(e))
            rows.append(r)
            print(f"[{i}/{len(cases)}] {r['case']}: {r['status']} {r.get('cells_M', '')}M  {r.get('history', '')} "
                  f"{r.get('error', '') or ''}", flush=True)
            with open(os.path.join(args.logs, "mesh_to_target.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader()
                w.writerows(sorted(rows, key=lambda x: x["case"]))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    msg = "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    print("\n" + msg)
    with open(os.path.join(args.logs, "TARGET_DONE"), "w") as f:
        f.write(time.strftime("%F %T") + "  " + msg + "\n")


if __name__ == "__main__":
    main()
