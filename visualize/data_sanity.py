"""Sanity renders for EXTRACTED graph data -- the files the dataset classes read
(static_* / wss_* wall graphs, interior_static_* / interior_* interior pools).
Local machine only: matplotlib, plus VTK decimation for the wall previews.

Which cases are rendered: not all of them (~1,760 cases x ~20 s), but every OUTLIER
plus a few NORMAL cases per dataset, picked from the flow distribution so problems
surface first:
  * metric per case: peak |WSS| at t=0, and peak |v| at t=0 when interior data is
    present -- where blow-ups and unit mix-ups show up. Cached per dataset in
    metrics.json, so later calls only measure new cases.
  * outlier: non-finite values, or |robust z| > z_thresh on log10 of a metric
    (median/MAD, per dataset -- flow scales differ between datasets).
  * normal: n_normal non-outliers at evenly spaced quantiles of the WSS metric, so
    the whole range is seen, not just the median.

Outputs under <sanity_root>/<tag>/ (dataset classes: <save_dir>/sanity/<tag>/):
    distribution.png  outliers.json  metrics.json
    <case>/mesh.png  geo_dist.png  cot_eigvec.png  wss.png  wss_trace.png
    <case>/interior.png  interior_stats.png                (interior data only)
A case whose files all exist is skipped unless redo=True.

    python -m angioflow.visualize.data_sanity --save_dir <root>                  # outliers + 5 normal per dataset
    python -m angioflow.visualize.data_sanity --save_dir <root> --all --workers 16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import torch
from scipy.spatial import cKDTree

SANITY_MODES = list(range(8)) + [15, 31, 63]       # cot_eigvec modes drawn (those < k)
WSS_FRAMES = [0, 16, 32, 48, 64, 79]               # filmstrip frames (those < n_frames)
WALL_FILES = ["mesh.png", "geo_dist.png", "cot_eigvec.png", "wss.png", "wss_trace.png"]
INTERIOR_FILES = ["interior.png", "interior_stats.png"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------------
# wall figures (moved from data/datasets.py)
# ----------------------------------------------------------------------------
def _trisurf_panel(ax, pos, faces, scalars, cmap, clim, title):
    """One 3D trisurf subplot colored by a per-node scalar field."""
    tri = ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=faces,
                          cmap=cmap, linewidth=0, antialiased=False)
    tri.set_array(scalars[faces].mean(axis=1))
    tri.set_clim(*clim)
    ax.figure.colorbar(tri, ax=ax, shrink=0.5)
    lo, hi = pos.min(0), pos.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo).clip(min=1e-9))
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def _geo_dist_figure(pos, faces, geo_dist, global_geo_dist, out_path, version, cid):
    """inlet/outlet_0/outlet_1/global geo_dist, one panel each, single figure."""
    plt = _plt()
    names = ["inlet", "outlet_0", "outlet_1", "global"]
    fields = [geo_dist[:, 0], geo_dist[:, 1], geo_dist[:, 2], global_geo_dist]
    fig = plt.figure(figsize=(5 * len(names), 5))
    for i, (name, field) in enumerate(zip(names, fields)):
        ax = fig.add_subplot(1, len(names), i + 1, projection="3d")
        _trisurf_panel(ax, pos, faces, field, "viridis", (0, 1), f"geo_dist -> {name}")
    fig.suptitle(f"{version}/{cid}: geo_dist", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _cot_eigvec_figure(pos, faces, cot_eigvec, cot_eigval, out_path, version, cid, mode_ids=None):
    """Cotangent-Laplacian eigenvector modes, one panel each, single figure.
    Signed basis functions -> symmetric coolwarm scale around 0.
    mode_ids: subset of mode indices to draw (default: all)."""
    plt = _plt()
    mode_ids = list(range(cot_eigvec.shape[1])) if mode_ids is None else list(mode_ids)
    n_modes = len(mode_ids)
    fig = plt.figure(figsize=(5 * n_modes, 5))
    for i, mi in enumerate(mode_ids):
        vals = cot_eigvec[:, mi]
        clim = float(np.abs(vals).max()) or 1.0
        ax = fig.add_subplot(1, n_modes, i + 1, projection="3d")
        _trisurf_panel(ax, pos, faces, vals, "coolwarm", (-clim, clim),
                       f"mode {mi} (λ={cot_eigval[mi]:.3g})")
    fig.suptitle(f"{version}/{cid}: cot_eigvec", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _vis_mesh_and_remap(pos, faces, max_faces=3000):
    """Decimate a disposable copy for sanity rendering only -- the stored data is
    untouched. matplotlib's 3D trisurf cost scales with face count; full res (tens
    of thousands of faces) takes seconds per panel. Values are sampled at each
    decimated vertex's nearest original node."""
    from angioflow.data.mesh import decimate_zone_mesh
    if len(faces) <= max_faces:
        return pos, faces, np.arange(len(pos))
    vis_pos, vis_faces = decimate_zone_mesh(pos, faces, ratio=max_faces / len(faces))
    _, orig_idx = cKDTree(pos).query(vis_pos, k=1)
    return vis_pos, vis_faces, orig_idx


def _mesh_figure(pos, faces, out_path, version, cid):
    """Static mesh, edges visible."""
    plt = _plt()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=faces,
                    color="lightsteelblue", linewidth=0.1, edgecolor="dimgrey", antialiased=False)
    lo, hi = pos.min(0), pos.max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo).clip(min=1e-9))
    ax.set_axis_off()
    ax.set_title(f"{version}/{cid}  {len(pos)} nodes  {len(faces)} faces", fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _wss_filmstrip_figure(pos, faces, wss_mag, frames, out_path, version, cid):
    """|WSS| at a few frames on one shared colour scale (0 .. p99, so a single spike
    saturates instead of washing out every other frame)."""
    plt = _plt()
    vmax = float(np.percentile(wss_mag, 99)) or 1.0
    fig = plt.figure(figsize=(5 * len(frames), 5))
    for i, t in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), i + 1, projection="3d")
        _trisurf_panel(ax, pos, faces, wss_mag[i], "inferno", (0, vmax),
                       f"t={t}  max={wss_mag[i].max():.3g}")
    fig.suptitle(f"{version}/{cid}: |WSS| (colour scale 0-{vmax:.3g}, p99)", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _trace_figure(mag, out_path, title, ylabel):
    """max / p99 / mean / median of a per-node magnitude over time. mag: [T, N]."""
    plt = _plt()
    t = np.arange(len(mag))
    fig, ax = plt.subplots(figsize=(8, 4))
    for f, name in [(lambda m: m.max(1), "max"), (lambda m: np.percentile(m, 99, axis=1), "p99"),
                    (lambda m: m.mean(1), "mean"), (lambda m: np.median(m, axis=1), "median")]:
        ax.plot(t, f(mag), label=name)
    ax.set_xlabel("frame"); ax.set_ylabel(ylabel); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# interior figures
# ----------------------------------------------------------------------------
def _interior_slab_figure(pos, vel_mag, wall_dist, t_peak, out_path, version, cid, slab_frac=0.04):
    """Thin slabs through the centre along x/y/z, viewed face-on: pool points coloured
    by |v| at the peak-flow frame (top row) and by wall distance (bottom row)."""
    plt = _plt()
    centre = np.median(pos, axis=0)
    ext = pos.max(0) - pos.min(0)
    vmax = float(np.percentile(vel_mag, 99)) or 1.0
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for a in range(3):
        m = np.abs(pos[:, a] - centre[a]) < slab_frac * ext[a]
        u, v = [i for i in range(3) if i != a]
        for row, (vals, cmap, clim, name) in enumerate([(vel_mag, "jet", (0, vmax), f"|v| t={t_peak}"),
                                                         (wall_dist, "viridis", (0, wall_dist.max()), "wall_dist")]):
            ax = axes[row, a]
            sc = ax.scatter(pos[m, u], pos[m, v], c=vals[m], s=1.5, cmap=cmap, vmin=clim[0], vmax=clim[1])
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{'xyz'[a]}-slab ({m.sum()} pts): {name}", fontsize=9)
            fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.suptitle(f"{version}/{cid}: interior pool ({len(pos)} pts)", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _interior_stats_figure(wall_dist, vel_mag, out_path, version, cid):
    """Pool wall-distance histogram (near-wall density boost) + |v| over time."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(wall_dist * 1e3, bins=60, color="steelblue")
    axes[0].set_xlabel("wall distance (mm)"); axes[0].set_ylabel("pool points")
    axes[0].set_title("pool density vs wall distance", fontsize=10)
    t = np.arange(len(vel_mag))
    for f, name in [(lambda m: m.max(1), "max"), (lambda m: np.percentile(m, 99, axis=1), "p99"),
                    (lambda m: m.mean(1), "mean")]:
        axes[1].plot(t, f(vel_mag), label=name)
    axes[1].set_xlabel("frame"); axes[1].set_ylabel("|v| (m/s)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title("|v| over time", fontsize=10)
    fig.suptitle(f"{version}/{cid}", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# per-case renderers (concrete paths, so they can run in worker processes)
# ----------------------------------------------------------------------------
def _existing(paths):
    return [(t, p) for t, p in enumerate(paths) if os.path.exists(p)]


def render_wall_case(static_path, wss_paths, out_dir, version, cid):
    """mesh / geo_dist / cot_eigvec / wss filmstrip / wss trace for one case."""
    os.makedirs(out_dir, exist_ok=True)
    static = torch.load(static_path, weights_only=False)
    pos, faces = static["pos"].numpy(), static["faces"].numpy()
    vis_pos, vis_faces, idx = _vis_mesh_and_remap(pos, faces)
    _mesh_figure(vis_pos, vis_faces, os.path.join(out_dir, "mesh.png"), version, cid)
    _geo_dist_figure(vis_pos, vis_faces, static["geo_dist"].numpy()[idx],
                     static["global_geo_dist"].numpy()[idx],
                     os.path.join(out_dir, "geo_dist.png"), version, cid)
    k = static["cot_eigvec"].shape[1]
    _cot_eigvec_figure(vis_pos, vis_faces, static["cot_eigvec"].numpy()[idx],
                       static["cot_eigval"].numpy(), os.path.join(out_dir, "cot_eigvec.png"),
                       version, cid, mode_ids=[m for m in SANITY_MODES if m < k])
    frames = _existing(wss_paths)
    if frames:
        mag = np.stack([torch.load(p, weights_only=False).float().norm(dim=1).numpy() for _, p in frames])
        ts = [t for t, _ in frames]
        i_peak = int(mag.mean(1).argmax())             # systole -- usually between fixed frames
        strip = sorted({i for i, t in enumerate(ts) if t in WSS_FRAMES} | {i_peak})
        _wss_filmstrip_figure(vis_pos, vis_faces, mag[strip][:, idx], [ts[i] for i in strip],
                              os.path.join(out_dir, "wss.png"), version, cid)
        _trace_figure(mag, os.path.join(out_dir, "wss_trace.png"),
                      f"{version}/{cid}: |WSS| over {len(ts)} frames", "|WSS| (Pa)")


def render_interior_case(static_path, frame_paths, out_dir, version, cid):
    """interior slabs + pool density / |v| trace for one case."""
    os.makedirs(out_dir, exist_ok=True)
    s = torch.load(static_path, weights_only=False, mmap=True)
    pos, wall_dist = s["pos"].numpy(), s["wall_dist"].numpy()
    frames = _existing(frame_paths)
    vel_mag = np.stack([torch.load(p, weights_only=False, mmap=True)["vel"].float().norm(dim=1).numpy()
                        for _, p in frames])
    i_peak = int(vel_mag.mean(1).argmax())
    _interior_slab_figure(pos, vel_mag[i_peak], wall_dist, frames[i_peak][0],
                          os.path.join(out_dir, "interior.png"), version, cid)
    _interior_stats_figure(wall_dist, vel_mag, os.path.join(out_dir, "interior_stats.png"), version, cid)


def _render_job(job):
    """(version, cid, out_dir, wall_paths | None, interior_paths | None) -> error str | None."""
    version, cid, out_dir, wall, interior = job
    try:
        if wall:
            render_wall_case(*wall, out_dir, version, cid)
        if interior:
            render_interior_case(*interior, out_dir, version, cid)
        return None
    except Exception as e:
        return f"{version}/{cid}: {e!r}"


# ----------------------------------------------------------------------------
# case selection: outliers + spread of normal cases
# ----------------------------------------------------------------------------
def _peak(t):
    t = t.float()
    return float(t.norm(dim=1).max()) if torch.isfinite(t).all() else float("nan")


def case_metrics(cases, wss_path, interior_frame_path=None, cache_path=None):
    """{cid: {"wss_max_t0": float, ["vel_max_t0": float]}}; NaN marks non-finite data.
    wss_path(cid, t) / interior_frame_path(cid, t) -> file path. Cached in cache_path."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    changed = False
    for cid in cases:
        m = cache.get(cid, {})
        if "wss_max_t0" not in m:
            m["wss_max_t0"] = _peak(torch.load(wss_path(cid, 0), weights_only=False))
        if interior_frame_path and "vel_max_t0" not in m:
            m["vel_max_t0"] = _peak(torch.load(interior_frame_path(cid, 0), weights_only=False, mmap=True)["vel"])
        if cache.get(cid) != m:
            cache[cid], changed = m, True
    if cache_path and changed:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=1)
    return {cid: cache[cid] for cid in cases}


def robust_z(x):
    """Modified z-score (Iglewicz-Hoaglin): 0.6745 * (x - median) / MAD."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 0.6745 * (x - med) / max(mad, 1e-12)


def select_cases(metrics, n_normal=5, z_thresh=3.5):
    """-> (normal [cid], outliers {cid: [reason]}). Outliers: non-finite metric, or
    |robust z| > z_thresh of log10(metric). Normal: n_normal non-outliers at evenly
    spaced quantiles of wss_max_t0."""
    outliers = {}
    for key in ("wss_max_t0", "vel_max_t0"):
        vals = [(c, m[key]) for c, m in sorted(metrics.items()) if key in m]
        good = []
        for c, v in vals:
            if v is None or not np.isfinite(v) or v <= 0:
                outliers.setdefault(c, []).append(f"{key} non-finite/zero")
            else:
                good.append((c, v))
        if len(good) >= 3:
            z = robust_z(np.log10([v for _, v in good]))
            for (c, v), zi in zip(good, z):
                if abs(zi) > z_thresh:
                    outliers.setdefault(c, []).append(f"{key}={v:.4g} (z={zi:+.1f})")
    pool = sorted((m["wss_max_t0"], c) for c, m in metrics.items() if c not in outliers)
    if not pool or n_normal <= 0:
        return [], outliers
    picks = np.unique(np.round(np.linspace(0, len(pool) - 1, min(n_normal, len(pool)))).astype(int))
    return [pool[i][1] for i in picks], outliers


def _distribution_figure(metrics, normal, outliers, out_path, version, z_thresh):
    plt = _plt()
    keys = [k for k in ("wss_max_t0", "vel_max_t0") if any(k in m for m in metrics.values())]
    fig, axes = plt.subplots(1, len(keys), figsize=(7 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        pts = [(c, m[key]) for c, m in metrics.items() if key in m and np.isfinite(m[key]) and m[key] > 0]
        lv = np.log10([v for _, v in pts])
        ax.hist(lv, bins=40, color="lightgrey", edgecolor="grey")
        for c, v in pts:
            if c in outliers:
                ax.axvline(np.log10(v), color="red", lw=1)
            elif c in normal:
                ax.axvline(np.log10(v), color="green", lw=1, ls="--")
        ax.set_xlabel(f"log10 {key}"); ax.set_ylabel("cases")
        ax.set_title(f"{key}: {len(pts)} cases, red = outlier (|z|>{z_thresh}), green = normal sample", fontsize=9)
    fig.suptitle(f"{version}: {len(metrics)} cases, {len(outliers)} outlier(s)", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_dataset_sanity(sanity_root, version, cases, static_path, wss_path, n_frames,
                          interior=None, n_normal=5, z_thresh=3.5, render_all=False,
                          redo=False, workers=1, verbose=True):
    """Select (or take all) cases of one dataset and render whatever sanity is missing.

    static_path(cid) / wss_path(cid, t): wall file paths. interior: None, or
    {"static": f(cid), "frame": f(cid, t)} for the interior files.
    Returns {"normal": [...], "outliers": {...}, "rendered": [...], "failed": [...]}."""
    out_root = os.path.join(sanity_root, version)
    os.makedirs(out_root, exist_ok=True)
    metrics = case_metrics(cases, wss_path, interior["frame"] if interior else None,
                           cache_path=os.path.join(out_root, "metrics.json"))
    normal, outliers = select_cases(metrics, n_normal, z_thresh)
    with open(os.path.join(out_root, "outliers.json"), "w") as f:
        json.dump(dict(z_thresh=z_thresh, outliers=outliers, normal=normal), f, indent=1)
    _distribution_figure(metrics, normal, outliers, os.path.join(out_root, "distribution.png"),
                         version, z_thresh)

    targets = list(cases) if render_all else sorted(set(normal) | set(outliers))
    expected = WALL_FILES + (INTERIOR_FILES if interior else [])
    jobs = []
    for cid in targets:
        out_dir = os.path.join(out_root, cid)
        if not redo and all(os.path.exists(os.path.join(out_dir, f)) for f in expected):
            continue
        wall = (static_path(cid), [wss_path(cid, t) for t in range(n_frames)])
        inter = ((interior["static"](cid), [interior["frame"](cid, t) for t in range(n_frames)])
                 if interior else None)
        jobs.append((version, cid, out_dir, wall, inter))
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            errors = pool.map(_render_job, jobs, chunksize=1)
    else:
        errors = [_render_job(j) for j in jobs]
    failed = [e for e in errors if e]
    if verbose:
        print(f"[sanity] {version}: {len(cases)} cases, {len(outliers)} outlier(s), {len(normal)} normal "
              f"sample(s); rendered {len(jobs) - len(failed)} new -> {out_root}"
              + (f"; {len(failed)} failed: {failed[:3]}" if failed else ""))
        for c, why in outliers.items():
            print(f"  [outlier] {version}/{c}: {'; '.join(why)}")
    return dict(normal=normal, outliers=outliers, rendered=[j[1] for j in jobs], failed=failed)


def wss_sanity_video(save_dir, version, cid, n_frames=80, out_dir=None, redo=False):
    """Wall sanity for one case of a per-version-dirs root, into
    <save_dir>/sanity/<version>/<cid>/ (kept for detect_polluted_cases(render=True))."""
    out_dir = out_dir or os.path.join(save_dir, "sanity", version, cid)
    if not redo and all(os.path.exists(os.path.join(out_dir, f)) for f in WALL_FILES):
        return
    vdir = os.path.join(save_dir, version)
    render_wall_case(os.path.join(vdir, f"static_{version}_{cid}.pt"),
                     [os.path.join(vdir, f"wss_{version}_{cid}_{t}.pt") for t in range(n_frames)],
                     out_dir, version, cid)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save_dir", required=True, help="root with <tag>/static_*.pt (+ interior/) -- unzipped layout")
    p.add_argument("--interior_dir", default=None, help="default: <save_dir>/interior if it exists")
    p.add_argument("--datasets", nargs="*", default=None, help="tags (default: every <tag>/ folder found)")
    p.add_argument("--n_frames", type=int, default=80)
    p.add_argument("--n_normal", type=int, default=5)
    p.add_argument("--z", type=float, default=3.5)
    p.add_argument("--all", action="store_true", help="render every case, not just outliers + normal samples")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    interior_dir = args.interior_dir or os.path.join(args.save_dir, "interior")
    tags = args.datasets or sorted(d for d in os.listdir(args.save_dir)
                                   if glob.glob(os.path.join(args.save_dir, d, f"static_{d}_*.pt")))
    for tag in tags:
        vdir = os.path.join(args.save_dir, tag)
        cases = sorted(os.path.basename(p)[len(f"static_{tag}_"):-len(".pt")]
                       for p in glob.glob(os.path.join(vdir, f"static_{tag}_*.pt")))
        has_interior = bool(glob.glob(os.path.join(interior_dir, f"interior_static_{tag}_*.pt")))
        if has_interior:   # only cases that have both
            cases = [c for c in cases if os.path.exists(os.path.join(interior_dir, f"interior_static_{tag}_{c}.pt"))]
        interior = (dict(static=lambda c, tag=tag: os.path.join(interior_dir, f"interior_static_{tag}_{c}.pt"),
                         frame=lambda c, t, tag=tag: os.path.join(interior_dir, f"interior_{tag}_{c}_{t}.pt"))
                    if has_interior else None)
        render_dataset_sanity(os.path.join(args.save_dir, "sanity"), tag, cases,
                              static_path=lambda c, tag=tag, vdir=vdir: os.path.join(vdir, f"static_{tag}_{c}.pt"),
                              wss_path=lambda c, t, tag=tag, vdir=vdir: os.path.join(vdir, f"wss_{tag}_{c}_{t}.pt"),
                              n_frames=args.n_frames, interior=interior, n_normal=args.n_normal,
                              z_thresh=args.z, render_all=args.all, redo=args.redo, workers=args.workers)


if __name__ == "__main__":
    main()
