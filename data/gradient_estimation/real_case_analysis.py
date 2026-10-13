"""Real-data test with a known reference: synthesize small tetrahedral elements
around real probe points, but since raw CFD ground truth only exists at fixed
mesh nodes (unlike the model, which is continuously queryable), interpolate the
ground-truth velocity onto the fresh synthetic vertices with knn_interpolate,
then estimate divergence the same way as everywhere else in this folder
(algorithms.local_gradient). Compared against Fluent's own stored grad_v trace
(a real, known reference — this is what neither synthetic_benchmark.py, which
has ground truth but an artificial field, nor a model-probe test, which has real
geometry but no known reference, can give on their own) and against the
existing graph-KNN-WLS stencil on the same probes.

Probes are filtered by wall distance > 2x the largest element radius in the
sweep, computed fresh from wall_data.pt (not the interior dataset's cached
wall_dist, which was measured against the registered/cropped wall mesh used for
the WSS task — the raw wall zone is the more complete, direct reference) —
conservative on purpose: continuity accuracy right at the wall isn't the point
here, the dome/lumen interior is.

Usage:
  conda activate new
  export MPLCONFIGDIR=/tmp/angioflow-mpl
  python -m angioflow.data.gradient_estimation.real_case_analysis
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.nn import knn_interpolate

from angioflow.data.datasets_interior import AngioFlowWallInteriorDatasetT0
from angioflow.data.gradient_estimation import algorithms as ga
from angioflow.data.gradient_verification.verify_interior_gradients import render_field_comparison
from angioflow.models.interior.constraints import continuity_residual

OUT_DIR = os.path.join(os.path.dirname(__file__), "real_case_results")


def build_dataset():
    return AngioFlowWallInteriorDatasetT0(
        case_root_angioflow="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged",
        case_root_aneug_flow="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data",
        save_dir="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_full",
        interior_dir="/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_interior",
        n_frames=80, decimate_ratio=1.0, decimate_method="quadric", lap_k=8,
        pool_ratio=0.16, sample_ratio=1.0, sanity_first=0,
        freeze_dataset=True, load_edge_index=True, load_grad=True,
        versions=["aneug_flow", "angioflow"], frames=[0],
    )


def fresh_wall_dist(dataset, version, case_id, query_pos):
    """Recompute distance to the raw wall_data.pt surface (not the interior
    dataset's cached wall_dist, which used the registered/cropped wall mesh)."""
    case_dir = dataset.case_dir_by_key[(version, case_id)]
    reg = torch.load(os.path.join(case_dir, "wall_mesh_registration.pt"), weights_only=False)
    center = reg["pos"].numpy().astype(np.float64).mean(0, keepdims=True)

    wall_data = torch.load(os.path.join(case_dir, "wall_data.pt"), weights_only=False)
    if version == "angioflow":
        wall_pos = wall_data["wall"]["pos"].numpy().astype(np.float64)
    else:
        wall_pos = np.stack([wall_data[f"{a}_coordinate"][0, :, 0].numpy() for a in "xyz"],
                            axis=1).astype(np.float64)
    wall_pos = wall_pos - center
    return cKDTree(wall_pos).query(query_pos.astype(np.float64))[0]


def _adaptive_max_radius_factor(wd, spacing, n_probes, safety_multiple=1.3,
                                min_factor=3.0, max_factor=9.0):
    """Largest radius (as a multiple of `spacing`, within [min_factor, max_factor])
    for which wall_dist > 2*radius still leaves >= n_probes*safety_multiple points.

    Vessel depth varies case to case (a smaller/shallower vessel has far less
    "safe" dome interior at a given radius) -- a fixed multiplier that works for
    one case can leave zero safe points for another (confirmed: stable_0 was
    fine at factor=9, stable_1 had 0 safe points at that same factor)."""
    target = int(np.ceil(n_probes * safety_multiple))
    sorted_wd, _ = torch.sort(wd, descending=True)
    threshold = sorted_wd[min(target, sorted_wd.numel()) - 1].item()
    factor = (threshold / 2.0) / spacing
    return float(max(min_factor, min(factor, max_factor)))


def _resid_stats(est, truth):
    """RMSE alongside MAE/median: RMSE (squares residuals) is disproportionately
    sensitive to a handful of outlier points -- and both the estimators and
    Fluent's own stored grad_v have some (CFD reconstruction artifacts, not real
    physics) -- so MAE/median better reflect typical, majority-of-points accuracy."""
    resid = (est - truth).abs()
    return dict(rmse=torch.sqrt(torch.mean(resid ** 2)).item(),
               mae=resid.mean().item(), median=resid.median().item())


def run_case(dataset, index, n_probes=500, k_interp=8, seed=0):
    version, case_id, frame = dataset._index[index]
    interior = dataset[index][1]
    pos = interior.pos.double()
    vel = interior.vel.double()
    grad_v = interior.grad_v.double()
    edge_index = interior.edge_index
    n_pool = pos.shape[0]
    print(f"\n[case] {version}/{case_id} frame={frame}  pool points={n_pool}")

    div_fluent_full = grad_v.view(-1, 3, 3).diagonal(dim1=1, dim2=2).sum(-1)

    div_graph_full, valid_graph_full = continuity_residual(
        vel, pos, edge_index, ridge=3e-2, power=2.0, min_neighbors=3, cond_tol=1e-6)

    spacing = float(cKDTree(pos.numpy()).query(pos.numpy(), k=2)[0][:, 1].mean())
    wd = torch.from_numpy(fresh_wall_dist(dataset, version, case_id, pos.numpy()))
    print(f"  point spacing~{spacing:.3g} m  wall_dist percentiles (m) "
          f"[10/50/90/99]={np.percentile(wd.numpy(), [10, 50, 90, 99]).round(6)}")

    # cap the radius sweep so the 2x-radius safety margin still leaves a
    # meaningful chunk of the pool as valid probes: real vessel radii are only
    # a few mm, and this is a hard geometric ceiling, not an arbitrary choice —
    # and it's case-specific (a shallower vessel has far less safe interior at
    # a given radius), so the ceiling is computed per case, not a fixed factor.
    max_factor = _adaptive_max_radius_factor(wd, spacing, n_probes)
    radii = np.geomspace(spacing * 0.5, spacing * max_factor, 10)
    max_radius = float(radii[-1])
    safe = wd > 2.0 * max_radius
    print(f"  radii [{radii[0]:.2g}, {radii[-1]:.2g}] m  "
          f"safe (wall_dist > {2*max_radius:.2g} m): {int(safe.sum())}/{n_pool}")
    if safe.sum() < n_probes:
        raise RuntimeError(f"only {int(safe.sum())} points pass the wall-distance filter "
                          f"(need {n_probes}) — shrink the radius sweep further for this case")

    torch.manual_seed(seed)
    safe_idx = safe.nonzero(as_tuple=True)[0]
    probe_idx = safe_idx[torch.randperm(safe_idx.numel())[:n_probes]]
    p = probe_idx.numel()

    center_pos = pos[probe_idx]
    center_vel = vel[probe_idx]
    div_fluent = div_fluent_full[probe_idx]
    div_graph = div_graph_full[probe_idx]
    valid_graph = valid_graph_full[probe_idx]

    graph_stats = _resid_stats(div_graph[valid_graph], div_fluent[valid_graph])
    print(f"  existing graph stencil vs Fluent grad_v @ probes: RMSE={graph_stats['rmse']:.4g}  "
          f"MAE={graph_stats['mae']:.4g}  median={graph_stats['median']:.4g} 1/s "
          f"({int(valid_graph.sum())}/{p} valid)")

    ks = [4, 16]
    rows = []
    for k in ks:
        dirs = ga.fibonacci_sphere_directions(k, dtype=torch.float64) if k != 4 else \
            ga.regular_tetrahedron_directions(dtype=torch.float64)
        for radius in radii:
            offsets = ga.offset_positions(center_pos, dirs, float(radius), rotate=True)
            offset_vel = knn_interpolate(vel, pos, offsets.reshape(-1, 3), k=k_interp).reshape(p, -1, 3)
            grad = ga.local_gradient(center_vel, offset_vel, center_pos, offsets)
            div_est = ga.divergence(grad)
            stats = _resid_stats(div_est, div_fluent)
            rows.append(dict(k=k, radius=float(radius), **stats))
            print(f"  synthetic tet K={k:2d} r={radius:.2e} m  RMSE={stats['rmse']:.4g}  "
                  f"MAE={stats['mae']:.4g} 1/s")

    return dict(version=version, case_id=case_id, n_pool=n_pool, n_probes=p, spacing=spacing,
               graph_rmse=graph_stats["rmse"], graph_mae=graph_stats["mae"], sweep=rows,
               # for the spatial/parity plot at a representative radius
               div_fluent=div_fluent, div_graph=div_graph, valid_graph=valid_graph,
               center_pos=center_pos, center_vel=center_vel, pos=pos, vel=vel, radii=radii)


def plot_case(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"{result['version']}_{result['case_id']}"
    ks = sorted({r["k"] for r in result["sweep"]})

    # RMSE (left) vs MAE (right): RMSE squares residuals, so it's pulled
    # disproportionately by a handful of outlier points (both the estimators'
    # and Fluent's own grad_v have some — see the residual-percentile check);
    # MAE better reflects accuracy on the majority of points.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = plt.get_cmap("viridis")
    for ax, metric, label in [(axes[0], "rmse", "RMSE"), (axes[1], "mae", "MAE")]:
        for i, k in enumerate(ks):
            rows = sorted((r for r in result["sweep"] if r["k"] == k), key=lambda r: r["radius"])
            ax.plot([r["radius"] for r in rows], [r[metric] for r in rows], "-o", ms=3,
                   color=cmap(i / max(len(ks) - 1, 1)), label=f"synthetic tet, K={k}")
        ax.axhline(result[f"graph_{metric}"], color="black", ls="--", label="existing graph stencil")
        ax.axvline(result["spacing"], color="gray", ls=":", label="mean point spacing")
        ax.set_xscale("log"); ax.set_yscale("log")
        # explicit tick at every tested radius: the default log-scale ticker
        # only labels power-of-10 positions, and this sweep spans about one
        # decade, so it was showing a single "1e-4" label and nothing else
        ax.set_xticks(result["radii"])
        ax.set_xticklabels([f"{r:.2g}" for r in result["radii"]], rotation=45, fontsize=6)
        ax.minorticks_off()
        ax.set_xlabel("element radius (m)")
        ax.set_ylabel(f"divergence {label} vs. Fluent's own grad_v (1/s)")
        ax.set_title(label)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)
    fig.suptitle(f"{result['version']}/{result['case_id']}: accuracy vs. Fluent reference")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, f"{name}_radius_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path}")


def plot_parity(result, k_offsets=16, k_interp=8):
    """Parity + shared-scale spatial error, same visual language as
    synthetic_benchmark.plot_spatial: best synthetic-tet setting found (K,
    largest swept radius) vs. the existing graph stencil, both against
    Fluent's own grad_v trace as the real known reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    name = f"{result['version']}_{result['case_id']}"
    radius = float(result["radii"][-1])
    center_pos, center_vel = result["center_pos"], result["center_vel"]
    pos, vel = result["pos"], result["vel"]
    div_fluent = result["div_fluent"]

    dirs = ga.fibonacci_sphere_directions(k_offsets, dtype=torch.float64)
    offsets = ga.offset_positions(center_pos, dirs, radius, rotate=True)
    p = center_pos.shape[0]
    offset_vel = knn_interpolate(vel, pos, offsets.reshape(-1, 3), k=k_interp).reshape(p, -1, 3)
    grad = ga.local_gradient(center_vel, offset_vel, center_pos, offsets)
    div_tet = ga.divergence(grad)

    div_graph, valid_graph = result["div_graph"], result["valid_graph"]
    lim = float(div_fluent.abs().quantile(0.98))
    err_tet = (div_tet - div_fluent).abs()
    err_graph = (div_graph - div_fluent).abs()
    shared_vmax = float(max(err_tet.quantile(0.98), err_graph[valid_graph].quantile(0.98)))
    pos_np = center_pos.numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9.2))
    panels = [(f"synthetic tet K={k_offsets}, r={radius:.2e} m", div_tet, err_tet,
              torch.ones_like(div_fluent, dtype=torch.bool)),
             ("existing graph stencil", div_graph, err_graph, valid_graph)]
    for row, (title, div_est, err, valid) in enumerate(panels):
        rmse = torch.sqrt(torch.mean((div_est[valid] - div_fluent[valid]) ** 2)).item()
        ax = axes[row, 0]
        ax.scatter(div_fluent[valid].numpy(), div_est[valid].numpy(), s=4, alpha=0.35, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="perfect (y=x)")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("Fluent grad_v trace (1/s)"); ax.set_ylabel("estimate (1/s)")
        ax.set_title(f"{title} — parity (RMSE={rmse:.3g})", fontsize=10)
        ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper left")

        ax = axes[row, 1]
        sc = ax.scatter(pos_np[valid.numpy(), 0], pos_np[valid.numpy(), 1],
                        c=err[valid].numpy(), s=10, cmap="magma", vmin=0.0, vmax=shared_vmax,
                        linewidths=0)
        ax.set_title(f"{title} — |estimate - Fluent| (shared scale)", fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle(f"{result['version']}/{result['case_id']}: accuracy vs. Fluent's own grad_v",
                fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, f"{name}_parity.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path}")


def plot_full_shape(dataset, index, k_offsets=16, k_interp=8, radius_factor=5.0,
                    max_plot_points=12000):
    """Estimated divergence across the whole safe (wall-filtered) dome
    interior, GT/estimate/error slab+3D — same render_field_comparison format
    as every other divergence_slab.png in this project, so it's directly
    comparable to the ones already looked at. If continuity holds, divergence
    should read ~0 everywhere here; this shows where it doesn't, for both the
    synthetic-tet method and the existing graph stencil, against the same
    Fluent grad_v reference."""
    version, case_id, frame = dataset._index[index]
    interior = dataset[index][1]
    pos = interior.pos.double()
    vel = interior.vel.double()
    grad_v = interior.grad_v.double()
    edge_index = interior.edge_index
    n_pool = pos.shape[0]

    div_fluent_full = grad_v.view(-1, 3, 3).diagonal(dim1=1, dim2=2).sum(-1)
    div_graph_full, valid_graph_full = continuity_residual(
        vel, pos, edge_index, ridge=3e-2, power=2.0, min_neighbors=3, cond_tol=1e-6)

    spacing = float(cKDTree(pos.numpy()).query(pos.numpy(), k=2)[0][:, 1].mean())
    radius = spacing * radius_factor    # same radius as the best point found in run_case's sweep
    wd = torch.from_numpy(fresh_wall_dist(dataset, version, case_id, pos.numpy()))
    safe = wd > 2.0 * radius
    print(f"\n[{version}/{case_id}] full-shape divergence @ radius={radius:.2e} m "
          f"(K={k_offsets}, k_interp={k_interp}): {int(safe.sum())}/{n_pool} safe (dome-interior) points")

    safe_idx = safe.nonzero(as_tuple=True)[0]
    center_pos, center_vel = pos[safe_idx], vel[safe_idx]
    p = center_pos.shape[0]

    dirs = ga.fibonacci_sphere_directions(k_offsets, dtype=torch.float64)
    offsets = ga.offset_positions(center_pos, dirs, radius, rotate=True)
    offset_vel = knn_interpolate(vel, pos, offsets.reshape(-1, 3), k=k_interp).reshape(p, -1, 3)
    div_tet = ga.divergence(ga.local_gradient(center_vel, offset_vel, center_pos, offsets))

    div_fluent = div_fluent_full[safe_idx]
    div_graph = div_graph_full[safe_idx]
    valid_graph = valid_graph_full[safe_idx]

    case_dir = os.path.join(OUT_DIR, f"{version}_{case_id}_full_shape")
    pos_np = center_pos.numpy()
    render_field_comparison(pos_np, div_fluent.numpy(), div_tet.numpy(),
                            np.ones(p, dtype=bool), "divergence_synthetic_tet",
                            case_dir, signed=True, max_plot_points=max_plot_points, seed=0)
    render_field_comparison(pos_np, div_fluent.numpy(), div_graph.numpy(),
                            valid_graph.numpy(), "divergence_graph_stencil",
                            case_dir, signed=True, max_plot_points=max_plot_points, seed=1)
    print(f"[ok] wrote {case_dir}")
    return case_dir


def training_regime_reference(dataset, index, safe_mask, sample_ratio=0.25, seed=0):
    """Reproduce InteriorSubsampleGraph(sample_ratio) exactly (random node
    subset + induced subgraph, over the WHOLE pool — matching what training
    actually does, no wall filtering), then report accuracy vs Fluent
    restricted to whichever of those points also fall in the wall-safe (dome
    interior) set, for a fair comparison against the synthetic-tet sweep."""
    interior = dataset[index][1]
    n_pool = interior.pos.shape[0]
    torch.manual_seed(seed)
    k = max(1, round(n_pool * sample_ratio))
    idx = torch.randperm(n_pool)[:k]
    sub = interior.subgraph(idx)
    mean_degree = sub.edge_index.shape[1] / sub.pos.shape[0]

    div_fluent_sub = sub.grad_v.double().view(-1, 3, 3).diagonal(dim1=1, dim2=2).sum(-1)
    div_sub, valid_sub = continuity_residual(
        sub.vel.double(), sub.pos.double(), sub.edge_index,
        ridge=3e-2, power=2.0, min_neighbors=3, cond_tol=1e-6)

    keep = valid_sub & safe_mask[idx]
    stats = _resid_stats(div_sub[keep], div_fluent_sub[keep])
    print(f"  [training regime] sample_ratio={sample_ratio} degree~{mean_degree:.1f}  "
          f"{int(keep.sum())}/{k} safe+valid  RMSE={stats['rmse']:.4g}  MAE={stats['mae']:.4g} 1/s")
    return dict(sample_ratio=sample_ratio, mean_degree=mean_degree, **stats)


def plot_k_sweep(dataset, index, k_interp=8, n_probes=500, seed=0,
                 ks=(4, 8, 16, 24, 32, 48, 64)):
    """RMSE and MAE vs. Fluent, as a function of K -- with radius
    independently re-optimized (swept and minimized) for EACH K, not one
    shared radius assumed to suit all of them. A fixed shared radius (what an
    earlier version of this did) understates lower-K settings, since K=4's own
    best radius is smaller than K=16's -- see the radius-sweep plot."""
    version, case_id, frame = dataset._index[index]
    interior = dataset[index][1]
    pos = interior.pos.double()
    vel = interior.vel.double()
    grad_v = interior.grad_v.double()
    edge_index = interior.edge_index
    print(f"\n[{version}/{case_id}] K-convergence sweep (best radius searched per K)")

    div_fluent_full = grad_v.view(-1, 3, 3).diagonal(dim1=1, dim2=2).sum(-1)
    div_graph_full, valid_graph_full = continuity_residual(
        vel, pos, edge_index, ridge=3e-2, power=2.0, min_neighbors=3, cond_tol=1e-6)

    spacing = float(cKDTree(pos.numpy()).query(pos.numpy(), k=2)[0][:, 1].mean())
    wd = torch.from_numpy(fresh_wall_dist(dataset, version, case_id, pos.numpy()))
    # same adaptive-ceiling logic as run_case -- a fixed factor left zero safe
    # points for shallower-vessel cases (confirmed: stable_1 failed at factor=9)
    max_factor = _adaptive_max_radius_factor(wd, spacing, n_probes)
    radii = np.geomspace(spacing * 0.5, spacing * max_factor, 10)
    max_radius = float(radii[-1])
    safe = wd > 2.0 * max_radius
    if safe.sum() < n_probes:
        raise RuntimeError(f"only {int(safe.sum())} safe points at max radius {max_radius:.2e} m "
                          f"(need {n_probes})")

    torch.manual_seed(seed)
    safe_idx = safe.nonzero(as_tuple=True)[0]
    probe_idx = safe_idx[torch.randperm(safe_idx.numel())[:n_probes]]
    p = probe_idx.numel()
    center_pos, center_vel = pos[probe_idx], vel[probe_idx]
    div_fluent = div_fluent_full[probe_idx]

    valid_fp = valid_graph_full[probe_idx]
    full_pool_stats = _resid_stats(div_graph_full[probe_idx][valid_fp], div_fluent[valid_fp])
    print(f"  [full pool] degree~64  RMSE={full_pool_stats['rmse']:.4g}  "
          f"MAE={full_pool_stats['mae']:.4g} 1/s")

    training = training_regime_reference(dataset, index, safe, sample_ratio=0.25, seed=seed)

    rows = []
    for k in ks:
        dirs = ga.fibonacci_sphere_directions(k, dtype=torch.float64) if k != 4 else \
            ga.regular_tetrahedron_directions(dtype=torch.float64)
        per_radius = []
        for radius in radii:
            offsets = ga.offset_positions(center_pos, dirs, float(radius), rotate=True)
            offset_vel = knn_interpolate(vel, pos, offsets.reshape(-1, 3), k=k_interp).reshape(p, -1, 3)
            grad = ga.local_gradient(center_vel, offset_vel, center_pos, offsets)
            div_est = ga.divergence(grad)
            per_radius.append(dict(radius=float(radius), **_resid_stats(div_est, div_fluent)))
        best = min(per_radius, key=lambda r: r["rmse"])
        rows.append(dict(k=k, best_radius=best["radius"], rmse=best["rmse"], mae=best["mae"],
                         median=best["median"]))
        print(f"  K={k:3d}  best radius={best['radius']:.2e} m  RMSE={best['rmse']:.4g}  "
              f"MAE={best['mae']:.4g} 1/s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, label in [(axes[0], "rmse", "RMSE"), (axes[1], "mae", "MAE")]:
        ax.plot([r["k"] for r in rows], [r[metric] for r in rows], "-o", color="tab:blue",
               label="synthetic tet (best radius per K)")
        ax.axhline(training[metric], color="tab:orange", ls="--",
                  label=f"current training (ratio=0.25, degree~{training['mean_degree']:.0f})")
        ax.axhline(full_pool_stats[metric], color="black", ls=":", label="full pool (degree~64)")
        ax.set_xlabel("K (offsets per element)")
        ax.set_ylabel(f"divergence {label} vs. Fluent grad_v (1/s)")
        ax.set_title(label); ax.grid(True, ls=":", alpha=0.4); ax.legend(fontsize=8)

    fig.suptitle(f"{version}/{case_id}: K-convergence, radius independently optimized per K "
                f"(swept {radii[0]:.2g}-{radii[-1]:.2g} m)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, f"{version}_{case_id}_k_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path}")
    return rows, training, full_pool_stats


def main(n_cases_per_version=5):
    dataset = build_dataset()
    by_version = {}
    for i, (v, cid, t) in enumerate(dataset._index):
        by_version.setdefault(v, []).append(i)
    chosen = by_version["aneug_flow"][:n_cases_per_version] + by_version["angioflow"][:n_cases_per_version]

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    for index in chosen:
        result = run_case(dataset, index)
        plot_case(result)
        plot_parity(result)
        plot_full_shape(dataset, index)
        plot_k_sweep(dataset, index)
        summary.append({k: v for k, v in result.items()
                        if k not in ("div_fluent", "div_graph", "valid_graph",
                                     "center_pos", "center_vel", "pos", "vel", "radii")})
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
