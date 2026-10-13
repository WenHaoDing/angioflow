"""Tools for detecting poorly-converged CFD cases via WSS smoothness metrics."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

from angioflow.data.dataset_tags import split_tag_cid


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_case_wss_gif(
    save_dir: str,
    version: str,
    case_id: str,
    out_path: str | None = None,
    max_wss: float = 30.0,
    fps: int = 10,
) -> str:
    """Render per-frame WSS magnitude for one case as a GIF on the surface mesh."""
    static_path = os.path.join(save_dir, f"static_{version}_{case_id}.pt")
    if not os.path.exists(static_path):
        raise FileNotFoundError(static_path)

    static = torch.load(static_path, weights_only=False)
    pos    = static["pos"].numpy().astype(np.float32)
    faces  = static["faces"].numpy().astype(np.int64)

    wss_paths = sorted(
        glob.glob(os.path.join(save_dir, f"wss_{version}_{case_id}_*.pt")),
        key=lambda p: int(p.rsplit("_", 1)[-1].replace(".pt", "")),
    )
    if not wss_paths:
        raise FileNotFoundError(f"No WSS frames found for {version}/{case_id} in {save_dir}")

    wss_frames = [
        torch.load(p, weights_only=False).numpy().astype(np.float32)
        for p in wss_paths
    ]

    if out_path is None:
        out_dir = os.path.join(save_dir, "sanity")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"wss_{version}_{case_id}.gif")

    _fd1 = os.dup(1); _fd2 = os.dup(2)
    _py_out = sys.stdout; _py_err = sys.stderr
    try:
        import pyvista as pv
        pv.start_xvfb()

        cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
        mesh  = pv.PolyData(pos, cells)

        pl = pv.Plotter(off_screen=True, window_size=(900, 700))
        pl.open_gif(out_path, fps=fps)

        for t, wss in enumerate(wss_frames):
            mag = np.linalg.norm(wss, axis=1).clip(0, max_wss)
            mesh.point_data["wss"] = mag
            pl.clear()
            pl.add_mesh(mesh, scalars="wss", cmap="coolwarm",
                        clim=[0, max_wss], show_scalar_bar=True,
                        scalar_bar_args={"title": "WSS (Pa/s)", "vertical": True})
            pl.add_title(f"{version}/{case_id}  frame {t:02d}", font_size=12)
            pl.camera_position = "iso"
            pl.write_frame()

        pl.close()
    finally:
        os.dup2(_fd1, 1); os.close(_fd1)
        os.dup2(_fd2, 2); os.close(_fd2)
        sys.stdout = _py_out; sys.stderr = _py_err

    print(f"[pollution_detect] saved {out_path}", flush=True)
    return out_path


# ---------------------------------------------------------------------------
# Smoothness metrics
# ---------------------------------------------------------------------------

def _spatial_roughness(mag: np.ndarray, edge_index: np.ndarray,
                       max_wss: float = 50.0) -> float:
    """Normalised Dirichlet energy: Σ_{edges} (mag_i - mag_j)² / Σ_i mag_i².

    WSS is clamped to [0, max_wss] before scoring so extreme outlier values
    don't dominate the numerator while the denominator stays bounded.
    """
    mag = mag.clip(0.0, max_wss)
    i, j = edge_index[0], edge_index[1]
    mask = i < j          # count each undirected edge once
    diff_sq = (mag[i[mask]] - mag[j[mask]]) ** 2
    sig_sq  = (mag ** 2).sum()
    if sig_sq < 1e-12:
        return 0.0
    return float(diff_sq.sum() / sig_sq)


def _temporal_roughness(mags: list[np.ndarray]) -> float:
    """Mean frame-to-frame L2 change normalised by mean signal energy.

    High values indicate the WSS field oscillates randomly between frames —
    a hallmark of non-converged transient simulations.
    """
    if len(mags) < 2:
        return 0.0
    changes  = [((mags[t + 1] - mags[t]) ** 2).mean() for t in range(len(mags) - 1)]
    energies = [(m ** 2).mean() for m in mags]
    denom = np.mean(energies)
    if denom < 1e-12:
        return 0.0
    return float(np.mean(changes) / denom)


def score_case(save_dir: str, version: str, case_id: str,
               max_wss: float = 50.0) -> dict:
    """Compute spatial and temporal roughness scores for one case.

    WSS is clamped to [0, max_wss] Pa before scoring.

    Returns:
        dict with keys:
          spatial_roughness  — mean Dirichlet energy over frames (lower = smoother)
          temporal_roughness — mean frame-to-frame change / energy (lower = steadier)
          n_frames           — number of frames loaded
    """
    static = torch.load(
        os.path.join(save_dir, f"static_{version}_{case_id}.pt"), weights_only=False,
    )
    edge_index = static["edge_index"].numpy()

    wss_paths = sorted(
        glob.glob(os.path.join(save_dir, f"wss_{version}_{case_id}_*.pt")),
        key=lambda p: int(p.rsplit("_", 1)[-1].replace(".pt", "")),
    )
    mags = [
        np.linalg.norm(
            torch.load(p, weights_only=False).numpy().astype(np.float32), axis=1
        )
        for p in wss_paths
    ]

    spatial  = float(np.mean([_spatial_roughness(m, edge_index, max_wss) for m in mags]))
    temporal = _temporal_roughness(mags)

    return {
        "spatial_roughness":  round(spatial,  6),
        "temporal_roughness": round(temporal, 6),
        "n_frames":           len(mags),
    }


def plot_score_histogram(scores: dict, out_path: str, max_wss: float = 50.0) -> None:
    """Save a histogram of spatial_roughness across all cases."""
    import matplotlib.pyplot as plt

    values = [s["spatial_roughness"] for s in scores.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Spatial roughness (Dirichlet energy, WSS clamped to "
                  f"0–{max_wss} Pa)", fontsize=11)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_title("WSS spatial roughness distribution", fontsize=13)
    ax.axvline(np.median(values), color="orange", linestyle="--",
               label=f"median = {np.median(values):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[pollution_detect] histogram saved to {out_path}", flush=True)


def scan_all_cases(
    save_dir: str,
    out_json: str | None = None,
    hist_path: str | None = None,
    max_wss: float = 50.0,
    verbose: bool = True,
) -> dict[tuple[str, str], dict]:
    """Score every case in save_dir and optionally save results to JSON + histogram.

    Args:
        save_dir:  directory containing static_*.pt and wss_*.pt files
        out_json:  if given, write scores to this path
        hist_path: if given, save a histogram PNG to this path
        max_wss:   WSS clamped to this value before scoring
        verbose:   print progress

    Returns:
        dict mapping (version, case_id) -> score dict
    """
    static_paths = sorted(glob.glob(os.path.join(save_dir, "static_*.pt")))
    results = {}

    for path in static_paths:
        stem = os.path.basename(path)[len("static_"):-len(".pt")]  # aneug_flow_stable_104
        version, case_id = split_tag_cid(stem)
        if version is None:
            continue
        try:
            scores = score_case(save_dir, version, case_id, max_wss=max_wss)
            results[(version, case_id)] = scores
            if verbose:
                print(f"[{version}/{case_id}]  "
                      f"spatial={scores['spatial_roughness']:.4f}  "
                      f"temporal={scores['temporal_roughness']:.4f}  "
                      f"frames={scores['n_frames']}", flush=True)
        except Exception as e:
            if verbose:
                print(f"[{version}/{case_id}] ERROR: {e}", flush=True)

    if out_json is not None:
        serialisable = {f"{v}_{c}": s for (v, c), s in results.items()}
        with open(out_json, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"[pollution_detect] scores saved to {out_json}", flush=True)

    if hist_path is not None and results:
        plot_score_histogram(results, hist_path, max_wss=max_wss)

    return results


def save_cfd_anomalies(
    save_dir: str,
    spatial_threshold: float = 0.25,
    max_wss: float = 50.0,
    verbose: bool = True,
) -> str:
    """Scan all cases in save_dir, flag non-converged ones, and write cfd_anomalies.json.

    Returns the path to the written JSON file.
    """
    scores = scan_all_cases(save_dir, max_wss=max_wss, verbose=verbose)
    flagged = flag_polluted(scores, spatial_threshold=spatial_threshold)
    out_path = os.path.join(save_dir, "cfd_anomalies.json")
    with open(out_path, "w") as f:
        json.dump([{"version": v, "cid": c} for v, c in sorted(flagged)], f, indent=2)
    print(f"[pollution_detect] cfd_anomalies.json: {len(flagged)} flagged → {out_path}", flush=True)
    return out_path


def flag_polluted(
    scores: dict,
    spatial_threshold: float  = 0.05,
    temporal_threshold: float = 0.10,
) -> list[tuple[str, str]]:
    """Return cases that exceed either roughness threshold.

    Suggested starting thresholds — calibrate by comparing known-good and
    known-bad cases (e.g. aneug_flow/stable_104 is known bad).
    """
    flagged = []
    for (version, case_id), s in scores.items():
        if (s["spatial_roughness"]  > spatial_threshold or
                s["temporal_roughness"] > temporal_threshold):
            flagged.append((version, case_id))
    return flagged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    # visualise a single case
    vis = sub.add_parser("vis")
    vis.add_argument("--save_dir", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_v2")
    vis.add_argument("--version",  default="aneug_flow")
    vis.add_argument("--case_id",  default="stable_104")
    vis.add_argument("--out_path", default=None)
    vis.add_argument("--max_wss",  type=float, default=30.0)
    vis.add_argument("--fps",      type=int,   default=10)

    # score a single case
    sc = sub.add_parser("score")
    sc.add_argument("--save_dir", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_v2")
    sc.add_argument("--version",  default="aneug_flow")
    sc.add_argument("--case_id",  default="stable_104")
    sc.add_argument("--max_wss",  type=float, default=50.0)

    # scan all cases
    scan = sub.add_parser("scan")
    scan.add_argument("--save_dir",   default="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_v2")
    scan.add_argument("--out_json",   default="/home/wenhao/AneuG-Net/pollution_scores.json")
    scan.add_argument("--hist_path",  default="/home/wenhao/AneuG-Net/pollution_histogram.png")
    scan.add_argument("--max_wss",    type=float, default=50.0)
    scan.add_argument("--spatial_threshold",  type=float, default=0.10)
    scan.add_argument("--temporal_threshold", type=float, default=0.10)

    args = p.parse_args()

    if args.cmd == "vis":
        visualize_case_wss_gif(
            args.save_dir, args.version, args.case_id,
            out_path=args.out_path, max_wss=args.max_wss, fps=args.fps,
        )
    elif args.cmd == "score":
        s = score_case(args.save_dir, args.version, args.case_id, max_wss=args.max_wss)
        print(s)
    elif args.cmd == "scan":
        scores = scan_all_cases(args.save_dir, out_json=args.out_json,
                                hist_path=args.hist_path, max_wss=args.max_wss)
        flagged = flag_polluted(scores,
                                spatial_threshold=args.spatial_threshold,
                                temporal_threshold=args.temporal_threshold)
        print(f"\nFlagged {len(flagged)} cases:")
        for v, c in sorted(flagged):
            print(f"  {v}/{c}")
    else:
        p.print_help()


"""


"""