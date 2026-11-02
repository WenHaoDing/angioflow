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
"""Hugging Face Job TEST: can PyVista render off-screen on HF Jobs, and how?

Tries the two standard ways to give VTK an OpenGL context on a headless machine, each in
its own child process, on one real AngioFlow case (wall_data.pt + blood_data.pt +
wall_mesh_registration.pt from whding123/AngioFlow cfd_data/<case>/):

  osmesa  VTK's vtk-osmesa wheel (wheels.vtk.org) swapped in for vtk: software rendering,
          no X server, no apt packages. Run FIRST, before anything is apt-installed, so it
          proves it needs nothing from the system.
  xvfb    the stock vtk wheel + apt-get install xvfb libgl1 libxrender1 + pv.start_xvfb()

Each backend renders the full sanity set in the style of register_cfd_shapes_angioflow.py's
sanity_check_case / registration_sanity (PyVista versions):
  mesh.png, registration.png (node_type + inlet/outlet rims), geo_dist.png,
  wss.gif (|WSS|, all frames), vel.gif (|velocity| point cloud, all frames)
with per-item timing and a blank-image check, and the outputs + a report.json are uploaded
to whding123/AngioFlow under _sanity_test/<backend>/<case>/ (nothing under cfd_data/).

    hf jobs uv run --flavor cpu-upgrade --timeout 1h -s HF_TOKEN -d debugging/hf_pyvista_test.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ID = "whding123/AngioFlow"
CASE = "0HRrQPIFOr_aneurysm1"
WORK = "/tmp/pv_test"
OUT_PREFIX = "_sanity_test"
VTK_WHEELS = "https://wheels.vtk.org"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
    print("\n".join("  " + l for l in tail), flush=True)
    return r.returncode


# ---------------------------------------------------------------------------
# child: render one case with one backend
# ---------------------------------------------------------------------------

def render(backend, case_dir, out_dir):
    import numpy as np
    import torch
    report = {"backend": backend, "items": {}}
    t0 = time.time()
    import vtk
    import pyvista as pv
    report["vtk"] = vtk.VTK_VERSION
    report["pyvista"] = pv.__version__
    if backend == "xvfb":
        pv.start_xvfb()
    pv.OFF_SCREEN = True
    os.makedirs(out_dir, exist_ok=True)

    def item(name, fn):
        t = time.time()
        try:
            info = fn() or {}
            report["items"][name] = {"ok": True, "seconds": round(time.time() - t, 1), **info}
        except Exception as e:
            import traceback
            traceback.print_exc()
            report["items"][name] = {"ok": False, "seconds": round(time.time() - t, 1), "error": repr(e)}
        print(f"[{backend}] {name}: {report['items'][name]}", flush=True)

    reg = torch.load(os.path.join(case_dir, "wall_mesh_registration.pt"), weights_only=False)
    pos, faces = reg["pos"].numpy(), reg["faces"].numpy()
    nt, gd = reg["node_type"].numpy(), reg["geo_dist"].numpy()
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()

    def mesh():
        return pv.PolyData(pos, cells)

    def shot(pl, name):
        img = pl.screenshot(os.path.join(out_dir, name), return_img=True)
        report.setdefault("render_window", pl.ren_win.GetClassName())
        pl.close()
        return {"shape": list(img.shape), "non_blank": bool(img.std() > 1.0)}

    def mesh_png():
        pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
        pl.add_mesh(mesh(), color="lightgrey", show_edges=True, edge_color="dimgray", line_width=0.3)
        pl.add_text(f"{CASE}: wall mesh ({len(pos)} nodes, {len(faces)} faces)", font_size=10)
        return shot(pl, "mesh.png")

    def registration_png():
        m = mesh()
        m.point_data["node_type"] = nt
        pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
        pl.add_mesh(m, scalars="node_type", cmap=["lightgrey", "orange", "red", "blue", "lime"],
                    clim=[-0.5, 4.5], n_colors=5, categories=True,
                    scalar_bar_args={"title": "0 wall 1 focused 2 inlet 3 out0 4 out1", "n_labels": 5})
        for k, color in [(2, "red"), (3, "blue"), (4, "lime")]:
            if (nt == k).any():
                pl.add_points(pos[nt == k], color=color, point_size=9, render_points_as_spheres=True)
        counts = np.bincount(nt, minlength=5).tolist()
        pl.add_text(f"{CASE}: node_type counts {counts}", font_size=10)
        return {**shot(pl, "registration.png"), "counts": counts}

    def geo_png():
        m = mesh()
        m.point_data["geo_dist_mm"] = gd * 1000
        pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
        pl.add_mesh(m, scalars="geo_dist_mm", cmap="viridis", scalar_bar_args={"title": "geo_dist (mm)"})
        pl.add_text(f"{CASE}: geodesic distance to the focused region", font_size=10)
        return shot(pl, "geo_dist.png")

    def wss_gif():
        wall = torch.load(os.path.join(case_dir, "wall_data.pt"), weights_only=False)["wall"]
        wss = torch.sqrt(wall["x_wall_shear"][:, :, 0] ** 2 + wall["y_wall_shear"][:, :, 0] ** 2
                         + wall["z_wall_shear"][:, :, 0] ** 2).numpy()
        m = pv.PolyData(wall["pos"].numpy(), np.hstack([np.full((len(wall["faces"]), 1), 3),
                                                       wall["faces"].numpy()]).ravel())
        m.point_data["wss"] = wss[0]
        pl = pv.Plotter(off_screen=True, window_size=(900, 700))
        pl.open_gif(os.path.join(out_dir, "wss.gif"), fps=10)
        pl.add_mesh(m, scalars="wss", clim=[0, float(np.percentile(wss, 99))], cmap="jet",
                    scalar_bar_args={"title": "|WSS| (Pa)"})
        for t in range(wss.shape[0]):
            m.point_data["wss"] = wss[t]
            pl.render()
            pl.write_frame()
        img = pl.screenshot(return_img=True)
        pl.close()
        return {"frames": int(wss.shape[0]), "non_blank": bool(img.std() > 1.0)}

    def vel_gif():
        blood = torch.load(os.path.join(case_dir, "blood_data.pt"), weights_only=False, mmap=True)
        vel = torch.sqrt(blood["x_velocity"][:, :, 0] ** 2 + blood["y_velocity"][:, :, 0] ** 2
                         + blood["z_velocity"][:, :, 0] ** 2).numpy()
        cloud = pv.PolyData(blood["pos"].numpy())
        cloud.point_data["vel"] = vel[0]
        pl = pv.Plotter(off_screen=True, window_size=(900, 700))
        pl.open_gif(os.path.join(out_dir, "vel.gif"), fps=10)
        pl.add_points(cloud, scalars="vel", clim=[0, float(np.percentile(vel, 99))], cmap="jet",
                      point_size=2, opacity=0.3, scalar_bar_args={"title": "|velocity| (m/s)"})
        for t in range(vel.shape[0]):
            cloud.point_data["vel"] = vel[t]
            pl.render()
            pl.write_frame()
        img = pl.screenshot(return_img=True)
        pl.close()
        return {"frames": int(vel.shape[0]), "points": int(vel.shape[1]), "non_blank": bool(img.std() > 1.0)}

    for name, fn in [("mesh.png", mesh_png), ("registration.png", registration_png),
                     ("geo_dist.png", geo_png), ("wss.gif", wss_gif), ("vel.gif", vel_gif)]:
        item(name, fn)
    report["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)


def run_child(backend, case_dir, out_dir, timeout=1500):
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    t = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--render", backend,
                            "--case_dir", case_dir, "--out_dir", out_dir], env=env, timeout=timeout)
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"[{backend}] child exit {code} after {time.time() - t:.0f}s", flush=True)
    path = os.path.join(out_dir, "report.json")
    if os.path.exists(path):
        with open(path) as f:
            return {**json.load(f), "exit": code}
    os.makedirs(out_dir, exist_ok=True)
    rep = {"backend": backend, "exit": code, "error": "no report (crashed before finishing)"}
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    return rep


# ---------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------

def main():
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ.get("HF_TOKEN")
    print(f"[env] python {sys.version.split()[0]}  uid {os.getuid()}  cpus {os.cpu_count()}  "
          f"DISPLAY={os.environ.get('DISPLAY')}", flush=True)
    sh("cat /etc/os-release | head -2; free -g | head -2")

    case_dir = os.path.join(WORK, "case")
    t = time.time()
    for name in ["wall_mesh_registration.pt", "wall_data.pt", "blood_data.pt"]:
        hf_hub_download(REPO_ID, f"cfd_data/{CASE}/{name}", repo_type="dataset", token=token, local_dir=WORK)
    os.makedirs(case_dir, exist_ok=True)
    for name in os.listdir(os.path.join(WORK, "cfd_data", CASE)):
        shutil.move(os.path.join(WORK, "cfd_data", CASE, name), os.path.join(case_dir, name))
    print(f"[download] {CASE} in {time.time() - t:.0f}s", flush=True)

    results, setup = {}, {}
    out_root = os.path.join(WORK, "out")

    # 1. OSMesa wheel, before any apt package exists
    t = time.time()
    setup["osmesa_install_exit"] = sh(f"uv pip uninstall --python {sys.executable} vtk && "
                                      f"uv pip install --python {sys.executable} --extra-index-url {VTK_WHEELS} "
                                      f"vtk-osmesa==9.3.1")
    setup["osmesa_install_s"] = round(time.time() - t)
    results["osmesa"] = run_child("osmesa", case_dir, os.path.join(out_root, "osmesa", CASE))

    # 2. stock vtk + Xvfb
    t = time.time()
    setup["vtk_reinstall_exit"] = sh(f"uv pip uninstall --python {sys.executable} vtk-osmesa && "
                                     f"uv pip install --python {sys.executable} vtk==9.3.1")
    setup["apt_exit"] = sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
                           "apt-get install -y -qq xvfb libgl1 libxrender1 > /dev/null")
    setup["xvfb_install_s"] = round(time.time() - t)
    results["xvfb"] = run_child("xvfb", case_dir, os.path.join(out_root, "xvfb", CASE))

    summary = {"case": CASE, "setup": setup, "results": results}
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n[summary]\n" + json.dumps(summary, indent=1), flush=True)
    HfApi(token=token).upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=out_root,
                                     path_in_repo=OUT_PREFIX, commit_message="PyVista-on-HF-Jobs render test")
    print(f"[upload] -> {REPO_ID}/{OUT_PREFIX}/", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--render", choices=["osmesa", "xvfb"])
    p.add_argument("--case_dir")
    p.add_argument("--out_dir")
    a = p.parse_args()
    if a.render:
        render(a.render, a.case_dir, a.out_dir)
    else:
        main()
