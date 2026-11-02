# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch==2.4.0",
#     "numpy<2.4",
#     "huggingface_hub>=1.8",
#     "pyvista==0.42.3",
#     "vtk==9.3.1",
#     "imageio>=2.30",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Hugging Face Job: PyVista sanity renders for every CFD case of a dataset on the Hub.

A self-contained uv script -- submit it from your machine; everything runs on Hugging Face
(nothing is downloaded locally, no repo clone):

    hf jobs uv run --flavor cpu-upgrade --timeout 6h -s HF_TOKEN -d \\
        huggingface/render_sanity_hf.py --repo_id whding123/AngioFlow --execute --yes
    hf jobs uv run --flavor cpu-upgrade --timeout 6h -s HF_TOKEN -d \\
        huggingface/render_sanity_hf.py --repo_id whding123/AneuGv2 --execute --yes

Works for any dataset laid out as <src_prefix>/<case>/{wall_data.pt, blood_data.pt,
wall_mesh_registration.pt} (whding123/AngioFlow, whding123/AneuGv2: AngioFlow-style
wall_data.pt with the wall zone under "wall", blood_data.pt with "pos" + x/y/z_velocity).

Per case, written to <out_prefix>/<case>/ of the SAME dataset (default sanity/; never under
the source cfd_data/ -- an overlapping --out_prefix is refused):
    mesh.png          wall mesh (from the registration)
    registration.png  node_type: wall / focused / inlet / outlet_0 / outlet_1 (rims marked)
    geo_dist.png      geodesic distance to the focused region (mm)
    wss.gif           |WSS| on the CFD wall, every frame
    vel.gif           |velocity| on the blood point cloud, every frame
    info.json         counts + automatic checks (see check_case), e.g. registration_stale:
                      the registration's wall is not the current wall_data.pt's wall
and one table per Job, <out_prefix>/_summary/<dataset>_job<i>.csv, with every case's checks.

Rendering: VTK's OSMesa wheel (vtk-osmesa, wheels.vtk.org) -- software OpenGL, no X server,
no apt packages -- swapped in for the stock vtk at start-up (the script then re-executes
itself). Tested on HF (debugging/hf_pyvista_test.py): identical images to Xvfb, ~35 s/case.
With no X display needed, --workers cases render in parallel in one Job. --backend xvfb
uses the stock vtk + pv.start_xvfb() instead (local tests).

Skips cases whose <out_prefix>/<case>/ is already complete (--redo to re-render), so a rerun
continues; cases without wall_mesh_registration.pt are skipped and listed (register them
first, then rerun). --num_jobs/--job_index split the cases over several Jobs.
Without --execute --yes it only prints the plan.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC_PREFIX = "cfd_data"
OUT_PREFIX = "sanity"
OUTPUTS = ["mesh.png", "registration.png", "geo_dist.png", "wss.gif", "vel.gif", "info.json"]
NODE_TYPES = {0: "wall", 1: "focused", 2: "inlet", 3: "outlet_0", 4: "outlet_1"}
NODE_COLORS = ["lightgrey", "orange", "red", "blue", "lime"]
VTK_WHEELS = "https://wheels.vtk.org"
WORK_DIR = "/tmp/render_sanity"
MAX_RETRIES, RETRY_DELAY = 5, 10
# focused nodes lie within ~2 edge lengths of the focused mesh, so their geo_dist is small but
# not 0: up to ~2 mm on AngioFlow, 2-4 mm on AneuGv2 (coarser dome.obj; p99 3.95 mm over 144
# cases); far beyond that = measured from the wrong region (one AneuGv2 case: 13.9 mm)
FOCUSED_GEO_MAX_M = 5e-3


def with_retry(fn, what):
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            print(f"[retry] {what}: attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(max(delay, 60) if ("429" in str(e) or "rate limit" in str(e).lower()) else delay)
            delay *= 2


def ensure_osmesa():
    """Replace the stock vtk with vtk-osmesa in this uv environment, then re-execute."""
    if os.environ.get("RENDER_SANITY_OSMESA") == "1":
        return
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("[abort] --backend osmesa needs `uv` on PATH (hf jobs uv run has it)")
    py = sys.executable
    cmd = (f"{uv} pip uninstall --python {py} vtk && "
           f"{uv} pip install --python {py} --extra-index-url {VTK_WHEELS} vtk-osmesa==9.3.1")
    print(f"[setup] {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)
    os.environ["RENDER_SANITY_OSMESA"] = "1"
    os.execv(py, [py, os.path.abspath(__file__)] + sys.argv[1:])


# ---------------------------------------------------------------------------
# one case (runs in a worker process)
# ---------------------------------------------------------------------------

def check_case(reg, wall, wss, vel):
    """Counts + flags. A flag means: look at this case's images."""
    import numpy as np
    nt, gd = reg["node_type"].numpy(), reg["geo_dist"].numpy()
    counts = np.bincount(nt, minlength=5)[:5].tolist()
    wall_pos = wall["pos"].numpy()
    same_wall = reg["pos"].shape == wall["pos"].shape
    pos_err = float(np.abs(reg["pos"].numpy() - wall_pos).max()) if same_wall else None
    flags = []
    if not same_wall or pos_err > 1e-6:
        flags.append("registration_stale")          # registration built from another wall mesh
    for k in (1, 2, 3):
        if counts[k] == 0:
            flags.append(f"no_{NODE_TYPES[k]}")
    if not np.isfinite(gd).all():
        flags.append("geo_dist_nonfinite")
    elif gd.max() <= 0:
        flags.append("geo_dist_all_zero")
    elif counts[1] and float(gd[nt == 1].max()) > FOCUSED_GEO_MAX_M:
        flags.append("geo_dist_large_on_focused")   # measured from somewhere else than the focused region
    if not np.isfinite(wss).all():
        flags.append("wss_nonfinite")
    if not np.isfinite(vel).all():
        flags.append("vel_nonfinite")
    return {
        "n_wall_nodes": int(len(wall_pos)), "n_reg_nodes": int(reg["pos"].shape[0]),
        "reg_wall_max_pos_diff_m": pos_err,
        **{f"n_{NODE_TYPES[k]}": counts[k] for k in range(5)},
        "geo_dist_max_mm": float(np.nanmax(gd) * 1000),
        "geo_dist_max_on_focused_mm": float(gd[nt == 1].max() * 1000) if counts[1] else None,
        "aneurysm_type": int(reg["aneurysm_type"]) if "aneurysm_type" in reg else None,
        "flow_split": [round(float(x), 4) for x in reg["flow_split"]] if "flow_split" in reg else None,
        "wss_frames": int(wss.shape[0]), "wss_p99_pa": float(np.nanpercentile(wss, 99)),
        "vel_frames": int(vel.shape[0]), "vel_points": int(vel.shape[1]),
        "vel_p99_ms": float(np.nanpercentile(vel, 99)),
        "flags": flags,
    }


def render_case(repo_id, case, args_d):
    """Download, render, return {"case", "out_dir", "info"} or {"case", "error"}."""
    import numpy as np
    import torch
    import pyvista as pv
    if args_d["backend"] == "xvfb":
        pv.start_xvfb()
    pv.OFF_SCREEN = True
    from huggingface_hub import hf_hub_download

    t0 = time.time()
    dl = os.path.join(args_d["work_dir"], "dl", case)
    out_dir = os.path.join(args_d["work_dir"], "out", case)
    shutil.rmtree(dl, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    try:
        paths = {}
        for name in ["wall_mesh_registration.pt", "wall_data.pt", "blood_data.pt"]:
            paths[name] = with_retry(lambda: hf_hub_download(
                repo_id, f"{args_d['src_prefix']}/{case}/{name}", repo_type="dataset",
                token=args_d["token"], local_dir=dl), f"download {case}/{name}")
        t_dl = time.time() - t0

        reg = torch.load(paths["wall_mesh_registration.pt"], weights_only=False)
        wall = torch.load(paths["wall_data.pt"], weights_only=False)["wall"]
        wss = torch.sqrt(wall["x_wall_shear"][:, :, 0] ** 2 + wall["y_wall_shear"][:, :, 0] ** 2
                         + wall["z_wall_shear"][:, :, 0] ** 2).numpy()
        blood = torch.load(paths["blood_data.pt"], weights_only=False, mmap=True)
        vel = torch.sqrt(blood["x_velocity"][:, :, 0] ** 2 + blood["y_velocity"][:, :, 0] ** 2
                         + blood["z_velocity"][:, :, 0] ** 2).numpy()
        blood_pos = blood["pos"].numpy()
        del blood
        info = check_case(reg, wall, wss, vel)

        def polydata(pos, faces):
            faces = np.asarray(faces)
            return pv.PolyData(np.asarray(pos), np.hstack([np.full((len(faces), 1), 3), faces]).ravel())

        pos, faces = reg["pos"].numpy(), reg["faces"].numpy()
        nt, gd = reg["node_type"].numpy(), reg["geo_dist"].numpy()

        pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
        pl.add_mesh(polydata(pos, faces), color="lightgrey", show_edges=True, edge_color="dimgray", line_width=0.3)
        pl.add_text(f"{case}: wall mesh ({len(pos)} nodes, {len(faces)} faces)", font_size=10)
        pl.screenshot(os.path.join(out_dir, "mesh.png")); pl.close()

        m = polydata(pos, faces)
        m.point_data["node_type"] = nt
        pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
        pl.add_mesh(m, scalars="node_type", cmap=NODE_COLORS, clim=[-0.5, 4.5], n_colors=5,
                    annotations={k: v for k, v in NODE_TYPES.items()},
                    scalar_bar_args={"title": "node_type", "n_labels": 0})
        for k in (2, 3, 4):
            if (nt == k).any():
                pl.add_points(pos[nt == k], color=NODE_COLORS[k], point_size=9, render_points_as_spheres=True)
        counts = ", ".join(f"{NODE_TYPES[k]} {info['n_' + NODE_TYPES[k]]}" for k in range(5))
        pl.add_text(f"{case}: {counts}" + (f"\nFLAGS: {', '.join(info['flags'])}" if info["flags"] else ""),
                    font_size=10)
        pl.screenshot(os.path.join(out_dir, "registration.png")); pl.close()

        m = polydata(pos, faces)
        m.point_data["geo_dist_mm"] = gd * 1000
        pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
        pl.add_mesh(m, scalars="geo_dist_mm", cmap="viridis", scalar_bar_args={"title": "geo_dist (mm)"})
        pl.add_text(f"{case}: geodesic distance to the focused region", font_size=10)
        pl.screenshot(os.path.join(out_dir, "geo_dist.png")); pl.close()

        m = polydata(wall["pos"].numpy(), wall["faces"].numpy())
        m.point_data["wss"] = wss[0]
        pl = pv.Plotter(off_screen=True, window_size=(900, 700))
        pl.open_gif(os.path.join(out_dir, "wss.gif"), fps=10)
        pl.add_mesh(m, scalars="wss", clim=[0, info["wss_p99_pa"]], cmap="jet",
                    scalar_bar_args={"title": "|WSS| (Pa)"})
        pl.add_text(f"{case}: |WSS|", font_size=10)
        for t in range(wss.shape[0]):
            m.point_data["wss"] = wss[t]
            pl.render(); pl.write_frame()
        pl.close()

        cloud = pv.PolyData(blood_pos)
        cloud.point_data["vel"] = vel[0]
        pl = pv.Plotter(off_screen=True, window_size=(900, 700))
        pl.open_gif(os.path.join(out_dir, "vel.gif"), fps=10)
        pl.add_points(cloud, scalars="vel", clim=[0, info["vel_p99_ms"]], cmap="jet",
                      point_size=2, opacity=0.3, scalar_bar_args={"title": "|velocity| (m/s)"})
        pl.add_text(f"{case}: |velocity| ({vel.shape[1]} points)", font_size=10)
        for t in range(vel.shape[0]):
            cloud.point_data["vel"] = vel[t]
            pl.render(); pl.write_frame()
        pl.close()

        info.update(case=case, seconds_download=round(t_dl, 1), seconds_total=round(time.time() - t0, 1))
        with open(os.path.join(out_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=1)
        return {"case": case, "out_dir": out_dir, "info": info}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"case": case, "error": repr(e)}
    finally:
        shutil.rmtree(dl, ignore_errors=True)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default="whding123/AngioFlow")
    p.add_argument("--src_prefix", default=SRC_PREFIX)
    p.add_argument("--out_prefix", default=OUT_PREFIX)
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--redo", action="store_true", help="re-render cases already complete on the Hub")
    p.add_argument("--workers", type=int, default=8, help="cases rendered in parallel")
    p.add_argument("--commit_every", type=int, default=20, help="cases per upload commit")
    p.add_argument("--num_jobs", type=int, default=1)
    p.add_argument("--job_index", type=int, default=0)
    p.add_argument("--backend", choices=["osmesa", "xvfb"], default="osmesa")
    p.add_argument("--work_dir", default=WORK_DIR)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true")
    p.add_argument("--yes", action="store_true", help="with --execute: render + upload")
    p.add_argument("--no_upload", action="store_true", help="with --execute: render only, keep <work_dir>/out")
    args = p.parse_args()
    src, out = args.src_prefix.strip("/"), args.out_prefix.strip("/")
    if not out or out == src or out.startswith(src + "/") or src.startswith(out + "/"):
        p.error(f"--out_prefix {out!r} must not be, contain or lie inside --src_prefix {src!r}")
    args.src_prefix, args.out_prefix = src, out
    return args


def main():
    args = parse_args()
    if args.execute and args.backend == "osmesa":
        ensure_osmesa()
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=args.token)
    files = with_retry(lambda: api.list_repo_files(args.repo_id, repo_type="dataset"), "list files")
    src, out = {}, {}
    for f in files:
        parts = f.split("/")
        if len(parts) == 3 and parts[0] == args.src_prefix and not parts[1].startswith("_"):
            src.setdefault(parts[1], set()).add(parts[2])
        elif len(parts) == len(args.out_prefix.split("/")) + 2 and f.startswith(args.out_prefix + "/"):
            out.setdefault(parts[-2], set()).add(parts[-1])
    cases = sorted(c for c, s in src.items() if {"wall_data.pt", "blood_data.pt"} <= s)
    if args.cases:
        cases = [c for c in cases if c in set(args.cases)]
    cases = cases[args.job_index::args.num_jobs]
    unregistered = [c for c in cases if "wall_mesh_registration.pt" not in src[c]]
    done = [c for c in cases if set(OUTPUTS) <= out.get(c, set())]
    todo = [c for c in cases if c not in unregistered and (args.redo or c not in done)]
    tag = args.repo_id.split("/")[-1]
    print(f"[plan] {args.repo_id}: job {args.job_index}/{args.num_jobs}: {len(cases)} case(s); "
          f"{len(unregistered)} without wall_mesh_registration.pt (skipped); {len(done)} already rendered"
          f"{' (redone)' if args.redo else ' (skipped)'}; {len(todo)} to render -> "
          f"{args.repo_id}/{args.out_prefix}/<case>/  ({args.workers} workers, backend {args.backend})", flush=True)
    if unregistered:
        print(f"[plan] unregistered: {unregistered}", flush=True)
    if not (args.execute and (args.yes or args.no_upload)) or not todo:
        print("[plan only] add --execute --yes to render + upload (--execute --no_upload: render only)")
        return

    ncpu = os.cpu_count() or 1
    os.environ.setdefault("LP_NUM_THREADS", str(max(1, ncpu // args.workers)))   # Mesa llvmpipe threads
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    args_d = {"backend": args.backend, "work_dir": args.work_dir, "token": args.token,
              "src_prefix": args.src_prefix}
    upload = not args.no_upload
    pending, rows, n_up, t_run = [], [], 0, time.time()

    def flush():
        nonlocal pending, n_up
        if not pending or not upload:
            return
        ops = [CommitOperationAdd(path_in_repo=f"{args.out_prefix}/{r['case']}/{name}",
                                  path_or_fileobj=os.path.join(r["out_dir"], name))
               for r in pending for name in OUTPUTS]
        with_retry(lambda: api.create_commit(args.repo_id, repo_type="dataset", operations=ops,
                                             commit_message=f"sanity renders for {len(pending)} case(s)"),
                   f"commit {len(pending)} case(s)")
        for r in pending:
            shutil.rmtree(r["out_dir"], ignore_errors=True)
        n_up += len(pending)
        print(f"[upload] {len(pending)} case(s) -> {args.repo_id}/{args.out_prefix}/ ({n_up} so far)", flush=True)
        pending = []

    try:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = {ex.submit(render_case, args.repo_id, c, args_d): c for c in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    r = fut.result()
                except Exception as e:                  # worker crashed outright
                    r = {"case": futs[fut], "error": repr(e)}
                if "error" in r:
                    rows.append({"case": r["case"], "error": r["error"]})
                    print(f"[{i}/{len(todo)}] {r['case']}: FAILED {r['error']}", flush=True)
                    continue
                info = r["info"]
                rows.append({**{k: v for k, v in info.items() if k != "flags"}, "flags": " ".join(info["flags"])})
                print(f"[{i}/{len(todo)}] {r['case']}: {info['seconds_total']:.0f}s"
                      f"{'  FLAGS ' + ' '.join(info['flags']) if info['flags'] else ''}", flush=True)
                pending.append(r)
                if len(pending) >= args.commit_every:
                    flush()
    finally:
        flush()
        n_rendered = sum(not r.get("error") for r in rows)          # this run only (before merging)
        n_flag = sum(bool(r.get("flags")) for r in rows)
        n_fail = len(rows) - n_rendered
        if rows and upload:
            # keep the rows of cases an earlier run rendered and this one didn't (if still in the dataset)
            name = f"{args.out_prefix}/_summary/{tag}_job{args.job_index}.csv"
            if name in files:
                from huggingface_hub import hf_hub_download
                done_now = {r["case"] for r in rows}
                old = with_retry(lambda: hf_hub_download(args.repo_id, name, repo_type="dataset", token=args.token,
                                                         local_dir=os.path.join(args.work_dir, "old_summary")),
                                 "download old summary")
                with open(old) as f:
                    rows += [r for r in csv.DictReader(f) if r["case"] not in done_now and r["case"] in src]
        if rows:
            path = os.path.join(args.work_dir, f"{tag}_job{args.job_index}.csv")
            keys = ["case", "flags", "error"] + sorted({k for r in rows for k in r} - {"case", "flags", "error"})
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(sorted(rows, key=lambda r: r["case"]))
            if upload:
                with_retry(lambda: api.upload_file(path_or_fileobj=path, repo_id=args.repo_id, repo_type="dataset",
                                                   path_in_repo=f"{args.out_prefix}/_summary/{os.path.basename(path)}",
                                                   commit_message="sanity summary"), "upload summary")
        print(f"\n[summary] {n_rendered} rendered ({n_up} uploaded), {n_flag} flagged, {n_fail} failed  "
              f"[{(time.time() - t_run) / 60:.1f} min]  table: {args.out_prefix}/_summary/{tag}_job{args.job_index}.csv",
              flush=True)


if __name__ == "__main__":
    main()
