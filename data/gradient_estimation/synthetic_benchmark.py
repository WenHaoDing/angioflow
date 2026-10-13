"""Controlled comparison: a known analytic vector field with injectable i.i.d.
per-point noise (mimicking the CFD export/interpolation noise diagnosed
elsewhere — see README.md), so each estimator's error is measured directly
against a known true divergence rather than inferred from real data.

Field: v(x) = A x + b + [sin(k0 x0), sin(k1 x1), sin(k2 x2)]  (per-component sine
adds real curvature -> nonzero truncation error at larger radii, so the sweep
below shows the actual bias/variance trade-off, not just a variance sweep).
True divergence: tr(A) + k0 cos(k0 x0) + k1 cos(k1 x1) + k2 cos(k2 x2).

Compares, at matched degree K and a sweep of element radius:
  * isotropic offsets   (Fibonacci-sphere directions, well-conditioned by construction)
  * naive offsets       (independent random directions, no conditioning guarantee —
                         stand-in for what a real point cloud's nearest neighbors
                         would give you, not deliberately spread out)
across a couple of realistic noise levels, to see whether deliberately isotropic
elements actually buy anything over "whatever neighbors happen to be nearby",
and to find the radius/K sweet spot (numerical-differentiation V-curve: too large
a radius loses accuracy to field curvature, too small loses it to noise).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from angioflow.data.gradient_estimation import algorithms as ga

OUT_DIR = os.path.join(os.path.dirname(__file__), "synthetic_results")


def true_field_and_divergence(x, A, freqs):
    """x: [..., 3] -> (v: [..., 3], div: [...])"""
    lin = x @ A.T
    sinus = torch.sin(x * freqs)                       # [..., 3], elementwise
    v = lin + sinus
    div = torch.diagonal(A).sum() + (freqs * torch.cos(x * freqs)).sum(-1)
    return v, div


def estimate(centers, directions_fn, k, radius, A, freqs, noise_std, rotate, generator):
    dirs = directions_fn(k, device=centers.device, dtype=centers.dtype) \
        if directions_fn is not ga.regular_tetrahedron_directions else \
        directions_fn(device=centers.device, dtype=centers.dtype)
    offsets = ga.offset_positions(centers, dirs, radius, rotate=rotate, generator=generator)

    center_val, _ = true_field_and_divergence(centers, A, freqs)
    offset_val, _ = true_field_and_divergence(offsets, A, freqs)
    if noise_std > 0:
        center_val = center_val + noise_std * torch.randn_like(center_val)
        offset_val = offset_val + noise_std * torch.randn_like(offset_val)

    grad = ga.local_gradient(center_val, offset_val, centers, offsets)
    return ga.divergence(grad)


def naive_random_directions(k, device=None, dtype=torch.float64):
    d = torch.randn(k, 3, device=device, dtype=dtype)
    return d / d.norm(dim=-1, keepdim=True)


def run():
    torch.manual_seed(0)
    dtype = torch.float64
    device = torch.device("cpu")
    P = 2000

    # domain/field scale chosen to resemble the real regime: point spacing
    # ~1e-4 m (measured), so put curvature on a ~1e-3 m length scale (an order
    # of magnitude larger than typical spacing, like real flow features are).
    spacing = 1.2e-4
    domain = spacing * 40
    centers = (torch.rand(P, 3, dtype=dtype, device=device) - 0.5) * domain

    A = torch.randn(3, 3, dtype=dtype, device=device) * 50.0
    length_scale = 1.5e-3
    freqs = torch.full((3,), 2 * np.pi / length_scale, dtype=dtype, device=device)

    _, true_div = true_field_and_divergence(centers, A, freqs)
    true_div_scale = true_div.std().item()

    # noise levels as a fraction of the true field's local value range —
    # chosen to bracket the ~7-17x stencil-degradation regime measured on
    # real CFD data, not to reproduce it exactly (this is a controlled probe,
    # not a re-derivation of the earlier real-data diagnostic).
    field_val, _ = true_field_and_divergence(centers, A, freqs)
    field_scale = field_val.std().item()
    noise_levels = {"none": 0.0, "low": 0.01 * field_scale, "high": 0.05 * field_scale}

    radii = np.geomspace(spacing * 0.3, spacing * 60, 14)
    # K=3 dropped from the generic sweep: a smooth spiral formula forced down
    # to only 3 points isn't guaranteed well-conditioned (verified — it isn't,
    # for this one), unlike the hand-picked K=4 regular tetrahedron below,
    # which is well-conditioned by construction for any K>=4.
    ks = [4, 8, 16, 32]

    results = []   # rows: noise_name, offset_kind, k, radius, rmse, rmse_rel
    generator = torch.Generator(device=device).manual_seed(0)

    for noise_name, noise_std in noise_levels.items():
        for offset_kind, dir_fn in [("isotropic", ga.fibonacci_sphere_directions),
                                    ("naive", naive_random_directions)]:
            for k in ks:
                for radius in radii:
                    torch.manual_seed(1)   # same noise draw across methods at a given setting
                    est_div = estimate(centers, dir_fn, k, float(radius), A, freqs,
                                       noise_std, rotate=True, generator=generator)
                    rmse = torch.sqrt(torch.mean((est_div - true_div) ** 2)).item()
                    results.append(dict(noise=noise_name, offsets=offset_kind, k=k,
                                        radius=float(radius), rmse=rmse,
                                        rmse_rel=rmse / true_div_scale))
        # regular tetrahedron (k=4 exact canonical case) evaluated separately
        for radius in radii:
            torch.manual_seed(1)
            est_div = estimate(centers, ga.regular_tetrahedron_directions, 4, float(radius),
                               A, freqs, noise_std, rotate=True, generator=generator)
            rmse = torch.sqrt(torch.mean((est_div - true_div) ** 2)).item()
            results.append(dict(noise=noise_name, offsets="regular_tet", k=4,
                                radius=float(radius), rmse=rmse, rmse_rel=rmse / true_div_scale))

    print(f"true divergence std (signal scale): {true_div_scale:.4g}")
    print(f"field value std: {field_scale:.4g}  point spacing: {spacing:.3g} m  "
          f"curvature length scale: {length_scale:.3g} m")
    context = dict(centers=centers, A=A, freqs=freqs, spacing=spacing,
                  noise_high=noise_levels["high"])
    return results, context


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    noises = sorted({r["noise"] for r in results}, key=lambda n: {"none": 0, "low": 1, "high": 2}[n])
    ks = sorted({r["k"] for r in results if r["offsets"] != "regular_tet"})
    radii = sorted({r["radius"] for r in results})

    fig, axes = plt.subplots(1, len(noises), figsize=(6 * len(noises), 5), sharey=True)
    if len(noises) == 1:
        axes = [axes]
    cmap = plt.get_cmap("viridis")
    for ax, noise in zip(axes, noises):
        for offset_kind, style in [("isotropic", "-"), ("naive", "--")]:
            for i, k in enumerate(ks):
                rows = sorted((r for r in results if r["noise"] == noise
                              and r["offsets"] == offset_kind and r["k"] == k),
                             key=lambda r: r["radius"])
                if not rows:
                    continue
                ax.plot([r["radius"] for r in rows], [r["rmse_rel"] for r in rows], style,
                       color=cmap(i / max(len(ks) - 1, 1)), label=f"{offset_kind} K={k}",
                       alpha=0.85)
        tet_rows = sorted((r for r in results if r["noise"] == noise and r["offsets"] == "regular_tet"),
                          key=lambda r: r["radius"])
        ax.plot([r["radius"] for r in tet_rows], [r["rmse_rel"] for r in tet_rows], "k-",
               linewidth=2.5, label="regular tet (K=4, canonical)")
        ax.set_xscale("log"); ax.set_yscale("log")
        # explicit tick at every tested radius, since the default log ticker
        # only labels power-of-10 positions and can leave a span like this
        # with just one (or zero) labeled ticks
        ax.set_xticks(radii)
        ax.set_xticklabels([f"{r:.2g}" for r in radii], rotation=45, fontsize=6)
        ax.minorticks_off()
        ax.set_xlabel("element radius (m)")
        ax.set_title(f"noise = {noise}")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel("divergence RMSE / true divergence std")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Synthetic benchmark: local-element gradient error vs. element radius")
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    path = os.path.join(OUT_DIR, "radius_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path}")


def plot_spatial(centers, A, freqs, radius, noise_std, seed=1):
    """Divergence accuracy for a handful of representative configs at one
    (radius, noise) setting: a parity plot (estimated vs. true divergence —
    the direct answer to "how accurate is it") and a spatial abs-error map
    using ONE shared color scale across all configs, so worse-vs-better is
    visible at a glance instead of hidden behind per-panel auto-scaling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, true_div = true_field_and_divergence(centers, A, freqs)
    pos = centers.numpy()
    true_div_np = true_div.numpy()

    configs = [
        ("naive, K=4", naive_random_directions, 4),
        ("regular tet, K=4 (canonical)", ga.regular_tetrahedron_directions, 4),
        ("isotropic, K=8", ga.fibonacci_sphere_directions, 8),
        ("isotropic, K=32", ga.fibonacci_sphere_directions, 32),
    ]

    est_divs, errs = [], []
    for name, dir_fn, k in configs:
        torch.manual_seed(seed)
        generator = torch.Generator(device=centers.device).manual_seed(0)
        est_div = estimate(centers, dir_fn, k, radius, A, freqs, noise_std,
                           rotate=True, generator=generator)
        est_divs.append(est_div)
        errs.append((est_div - true_div).abs())

    # one shared error scale across ALL configs (98th pct of the worst config),
    # so a quick visual scan shows the accuracy ordering the RMSE numbers give
    shared_err_vmax = float(max(e.quantile(0.98) for e in errs))
    lim = float(true_div.abs().quantile(0.98))

    n = len(configs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.6 * n))
    for row, ((name, _, _), est_div, err) in enumerate(zip(configs, est_divs, errs)):
        rmse = torch.sqrt((err ** 2).mean()).item()

        ax = axes[row, 0]
        ax.scatter(true_div_np, est_div.numpy(), s=4, alpha=0.35, linewidths=0)
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="perfect (y=x)")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("true divergence"); ax.set_ylabel("estimated divergence")
        ax.set_title(f"{name} — parity (RMSE={rmse:.3g})", fontsize=10)
        ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper left")

        ax = axes[row, 1]
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=err.numpy(), s=8, cmap="magma",
                        vmin=0.0, vmax=shared_err_vmax, linewidths=0)
        ax.set_title(f"{name} — |estimate - true| (shared scale)", fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(sc, ax=ax, shrink=0.8)
    fig.suptitle(f"Divergence accuracy — element radius={radius:.2e} m, noise_std={noise_std:.3g}",
                fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "spatial_comparison.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {path}")


if __name__ == "__main__":
    results, context = run()
    plot(results)
    # sweet-spot radius from the "high" noise curve (~1-2x point spacing) — the
    # regime where truncation and noise error roughly balance
    plot_spatial(context["centers"], context["A"], context["freqs"],
                radius=context["spacing"] * 1.5, noise_std=context["noise_high"])
    import json
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[done] {len(results)} rows -> {OUT_DIR}")
