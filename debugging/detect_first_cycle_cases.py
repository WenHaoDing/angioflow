"""Which AngioFlow cases' 80 extracted frames come from the FIRST CFD cycle instead of the
second?

Every case was simulated for 2 cardiac cycles and should have been exported from the
second, where the flow is periodic: the last frame (79) leads smoothly back into the
first (0), one ordinary time step apart. A case exported from the first cycle starts
from the solver's initial condition, so frame 0 does not continue frame 79 -- the
wrap-around 79 -> 0 is a jump much larger than the case's ordinary frame-to-frame step.

Per case and per field (interior velocity, wall WSS), with F_t the field at frame t:
    step_t     = rms(F_{t+1} - F_t),  t = 0..78      ordinary frame-to-frame change
    wrap       = rms(F_0 - F_79)                      the 79 -> 0 step
    jump_local = wrap / median(step_74..78)           the wrap vs the steps right before it:
                                                      ~1 when the derivative is continuous
    curv_local = rms(F_0 - (2 F_79 - F_78)) / median(rms(F_{t+1} - 2 F_t + F_{t-1}), t = 75..78)
                                                      the same for the second difference
    jump       = wrap / median(step_t)                (info) vs the whole cycle's median step
    rel_wrap   = wrap / rms(F)                        (info) frames 0 vs 79 relative to the field
The LOCAL ratios are the primary metrics: all datasets share the waveform phase --
frames 75-79 are calm late diastole, frames 0-4 the systolic upstroke -- so a whole-
cycle median (dominated by systole) understates a jump out of diastole (e.g. a clear
cycle-1 case at jump 1.8 but jump_local ~14; good cases sit at jump_local ~1).

No threshold is assumed; two references set them:
    good reference  AneuG-Flow -- every case exported from cycle 2
    bad reference   AneuGv2, its first 300 cases (case-id order) -- all exported from
                    cycle 1 (later AneuGv2 cases include correct ones, so they are not used;
                    --bad_ref_cases N, or --bad_reference none once AneuGv2 is re-exported)
For each primary metric (velocity jump_local and curv_local, WSS jump_local):
    good threshold  = good-reference p99
    bad threshold   = the larger of the bad-reference p1 and --bad_factor (2) x the
                      good-reference max (the floor keeps a fishy band between the two)
Target cases (AngioFlow):
    good   every primary metric <= its good threshold
    bad    any primary metric   >= its bad threshold
    fishy  everything else -> decide by eye from the renders
A case without interior data uses its WSS metric only.

Outputs under --out (default <root>_cycle_check/):
    metrics.csv            one row per case (references + target): metrics + class; a
                           resumable cache -- reruns only compute new cases
    classes.json           target cases by class, worst first
    distribution.png       histograms of the primary metrics for the references and the
                           target, thresholds marked
    render/<case>/         vel.gif (pyvista: interior |v| over all 80 frames, looping, so a
                           cycle-1 export jumps at the loop point) + steps.png (per-frame
                           step curve incl. the wrap, against both reference bands)

    python debugging/detect_first_cycle_cases.py --workers 8                    # renders fishy cases
    python debugging/detect_first_cycle_cases.py --render flagged --max_render 20
    python debugging/detect_first_cycle_cases.py --max_cases 15 --max_ref_cases 15 --bad_ref_cases 15 --render none
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np
import torch

ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG"
TARGET = "angioflow"
GOOD_REF = "aneug_flow"
BAD_REF = "aneugv2"
N_FRAMES = 80
FIELDS = ("vel", "wss")
METRICS = ("jump_local", "curv_local", "jump", "rel_wrap")
PRIMARY = (("vel", "jump_local"), ("vel", "curv_local"), ("wss", "jump_local"))
LOCAL = 5               # steps right before the wrap (74..78) that set its local scale
CSV_FIELDS = ["dataset", "case"] + [f"{f}_{m}" for f in FIELDS for m in METRICS] + ["class"]


def case_ids(root, tag):
    prefix = f"static_{tag}_"
    return sorted(f[len(prefix):-len(".pt")] for f in os.listdir(os.path.join(root, tag))
                  if f.startswith(prefix) and f.endswith(".pt"))


def load_field(root, tag, cid, field, n_frames=N_FRAMES):
    """[T, M, 3] float32, or None if any frame is missing."""
    if field == "wss":
        paths = [os.path.join(root, tag, f"wss_{tag}_{cid}_{t}.pt") for t in range(n_frames)]
    else:
        paths = [os.path.join(root, "interior", f"interior_{tag}_{cid}_{t}.pt") for t in range(n_frames)]
    if not all(os.path.exists(p) for p in paths):
        return None
    frames = []
    for p in paths:
        x = torch.load(p, weights_only=False, mmap=True)
        frames.append((x["vel"] if field == "vel" else x).numpy().astype(np.float32))
    return np.stack(frames)


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def periodicity(F):
    """Metrics of one field series F [T, M, 3] (see module docstring), the step curve,
    and the wrap step."""
    steps = np.array([_rms(F[t + 1] - F[t]) for t in range(len(F) - 1)])
    curv_in = np.array([_rms(F[t + 1] - 2 * F[t] + F[t - 1]) for t in range(1, len(F) - 1)])
    wrap = _rms(F[0] - F[-1])
    curv_wrap = _rms(F[0] - (2 * F[-1] - F[-2]))
    eps = 1e-12
    return dict(jump_local=wrap / max(np.median(steps[-LOCAL:]), eps),
                curv_local=curv_wrap / max(np.median(curv_in[-(LOCAL - 1):]), eps),
                jump=wrap / max(np.median(steps), eps),
                rel_wrap=wrap / max(_rms(F), eps)), steps, wrap


def case_metrics(job):
    root, tag, cid = job
    row = dict(dataset=tag, case=cid)
    for field in FIELDS:
        F = load_field(root, tag, cid, field)
        m = periodicity(F)[0] if F is not None else {k: "" for k in METRICS}
        row.update({f"{field}_{k}": v for k, v in m.items()})
    return row


def read_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {(r["dataset"], r["case"]): r for r in csv.DictReader(f)}


def value(row, field, metric):
    v = row.get(f"{field}_{metric}", "")
    return float(v) if v not in ("", None) else None


def ref_values(rows, tag, field, metric):
    return np.array([v for r in rows if r["dataset"] == tag for v in [value(r, field, metric)] if v is not None])


def thresholds(rows, bad_ref, bad_factor):
    """good = good-reference p99; bad = the larger of the bad-reference p1 and
    bad_factor x the good-reference max."""
    th = {}
    for field, metric in PRIMARY:
        good = ref_values(rows, GOOD_REF, field, metric)
        if len(good) == 0:
            raise SystemExit(f"[abort] no {GOOD_REF} values for {field} {metric}")
        t = dict(good=float(np.percentile(good, 99)), good_max=float(good.max()),
                 good_median=float(np.median(good)), n_good=len(good))
        floor = bad_factor * t["good_max"]
        t.update(bad=floor, bad_rule=f"{bad_factor:g} x {GOOD_REF} max")
        bad = ref_values(rows, bad_ref, field, metric) if bad_ref else np.array([])
        if len(bad):
            t.update(bad_p1=float(np.percentile(bad, 1)), bad_median=float(np.median(bad)), n_bad=len(bad),
                     n_bad_in_good_range=int((bad <= t["good_max"]).sum()))
            if t["bad_p1"] > floor:
                t.update(bad=t["bad_p1"], bad_rule=f"{bad_ref} p1")
        th[(field, metric)] = t
    return th


def classify(row, th):
    vals = {(f, m): value(row, f, m) for f, m in PRIMARY}
    vals = {k: v for k, v in vals.items() if v is not None}
    if not vals:
        return "missing"
    if any(v >= th[k]["bad"] for k, v in vals.items()):
        return "bad"
    if all(v <= th[k]["good"] for k, v in vals.items()):
        return "good"
    return "fishy"


def severity(row):
    return value(row, "vel", "jump_local") or value(row, "wss", "jump_local") or 0.0


def plot_distribution(rows, th, tags, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(PRIMARY), figsize=(6 * len(PRIMARY), 4.5))
    for ax, (field, metric) in zip(axes, PRIMARY):
        vals = {t: ref_values(rows, t, field, metric) for t in tags}
        allv = np.concatenate([v for v in vals.values() if len(v)])
        bins = np.logspace(np.log10(max(allv.min(), 1e-6)), np.log10(allv.max()), 60)
        for t, v in vals.items():
            role = {GOOD_REF: "good ref", TARGET: "target"}.get(t, "bad ref")
            ax.hist(v, bins=bins, histtype="step", lw=1.5, label=f"{t} ({role}, n={len(v)})")
        t = th[(field, metric)]
        ax.axvline(t["good"], color="green", ls="--", label=f"good <= {t['good']:.3g}")
        ax.axvline(t["bad"], color="red", ls="--", label=f"bad >= {t['bad']:.3g} ({t['bad_rule']})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{field} {metric}"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def step_band(root, tag, cases, n=30):
    """Per-frame normalised step curves (velocity) of up to n reference cases."""
    curves = []
    for cid in cases[:n]:
        F = load_field(root, tag, cid, "vel")
        if F is not None:
            _, steps, wrap = periodicity(F)
            curves.append(np.append(steps, wrap) / np.median(steps))
    return np.array(curves)


def render_case(root, cid, row, bands, out_dir, fps=10):
    """vel.gif: interior points coloured by |v| over a translucent wall, all 80 frames,
    looping (a cycle-1 export jumps at the loop point); steps.png: the step curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyvista as pv
    if not os.environ.get("DISPLAY"):
        pv.start_xvfb()
    tag = TARGET
    os.makedirs(out_dir, exist_ok=True)
    fmt = lambda f, m: f"{value(row, f, m):.2f}" if value(row, f, m) is not None else "n/a"
    title = (f"{tag}/{cid}  [{row['class']}]  vel jump_local {fmt('vel', 'jump_local')}  "
             f"curv_local {fmt('vel', 'curv_local')}  wss jump_local {fmt('wss', 'jump_local')}")

    fig, ax = plt.subplots(figsize=(11, 4))
    for (ref, curves), color in zip(bands.items(), ("green", "red")):
        if len(curves):
            lo, med, hi = np.percentile(curves, [5, 50, 95], axis=0)
            x = np.arange(len(med))
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, label=f"{ref} vel 5-95%")
            ax.plot(x, med, color=color, lw=0.8)
    for field, color in (("vel", "tab:blue"), ("wss", "tab:orange")):
        F = load_field(root, tag, cid, field)
        if F is None:
            continue
        _, steps, wrap = periodicity(F)
        s = np.append(steps, wrap) / np.median(steps)
        ax.plot(np.arange(len(s)), s, color=color, label=f"this case: {field}")
        ax.scatter([len(s) - 1], [s[-1]], color=color, zorder=3)
    ax.set_xlabel("frame step t -> t+1  (last point: wrap 79 -> 0)")
    ax.set_ylabel("step / median step"); ax.set_yscale("log"); ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "steps.png"), dpi=110); plt.close(fig)

    V = load_field(root, tag, cid, "vel")
    if V is None:
        return
    ipos = torch.load(os.path.join(root, "interior", f"interior_static_{tag}_{cid}.pt"),
                      weights_only=False)["pos"].numpy()
    wall = torch.load(os.path.join(root, tag, f"static_{tag}_{cid}.pt"), weights_only=False, mmap=True)
    faces = wall["faces"].numpy()
    wall_mesh = pv.PolyData(wall["pos"].numpy(), np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    speed = np.linalg.norm(V, axis=-1)
    cloud = pv.PolyData(ipos)
    cloud["speed"] = speed[0]
    pl = pv.Plotter(off_screen=True, window_size=(1000, 800))
    pl.add_mesh(wall_mesh, color="white", opacity=0.12)
    pl.add_mesh(cloud, scalars="speed", cmap="turbo", clim=[0.0, float(np.percentile(speed, 99))],
                point_size=4.0, render_points_as_spheres=True, opacity=0.35,
                scalar_bar_args=dict(title="|v| (m/s)", n_labels=5))
    pl.add_text(title, position="upper_edge", font_size=8, name="title")
    pl.open_gif(os.path.join(out_dir, "vel.gif"), fps=fps)
    for t in range(len(speed)):
        cloud["speed"] = speed[t]
        pl.add_text(f"frame {t}/{len(speed) - 1}" + ("   <- loops back to 0" if t == len(speed) - 1 else ""),
                    position="lower_left", font_size=12, name="ftxt")
        pl.write_frame()
    pl.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT, help="extracted AngioFlow-PyG root")
    p.add_argument("--out", default=None, help="default: <root>_cycle_check")
    p.add_argument("--bad_reference", default=BAD_REF,
                   help=f"known cycle-1 dataset for the bad threshold ('none' to use --bad_factor only)")
    p.add_argument("--bad_factor", type=float, default=2.0,
                   help=f"floor of the bad threshold: this x the {GOOD_REF} max (keeps a fishy band "
                        f"between the good and bad thresholds)")
    p.add_argument("--max_cases", type=int, default=None, help=f"first N {TARGET} cases (quick look)")
    p.add_argument("--max_ref_cases", type=int, default=200,
                   help=f"{GOOD_REF} cases used (evenly spaced; enough for p99)")
    p.add_argument("--bad_ref_cases", type=int, default=300,
                   help=f"the first N bad-reference cases in case-id order (all known cycle-1)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--render", choices=["none", "fishy", "bad", "flagged"], default="fishy",
                   help="which target cases get vel.gif + steps.png (flagged = fishy + bad)")
    p.add_argument("--max_render", type=int, default=None, help="cap on cases rendered (worst first)")
    p.add_argument("--redo", action="store_true", help="recompute metrics / re-render even if cached")
    args = p.parse_args()
    bad_ref = None if args.bad_reference.lower() == "none" else args.bad_reference
    out = args.out or args.root.rstrip("/") + "_cycle_check"
    os.makedirs(out, exist_ok=True)
    cache_path = os.path.join(out, "metrics.csv")

    good_cids = case_ids(args.root, GOOD_REF)
    if args.max_ref_cases and len(good_cids) > args.max_ref_cases:
        good_cids = [good_cids[i] for i in np.linspace(0, len(good_cids) - 1, args.max_ref_cases).astype(int)]
    jobs = [(args.root, GOOD_REF, c) for c in good_cids]
    if bad_ref:
        jobs += [(args.root, bad_ref, c) for c in case_ids(args.root, bad_ref)[:args.bad_ref_cases]]
    jobs += [(args.root, TARGET, c) for c in case_ids(args.root, TARGET)[:args.max_cases]]

    cache = {} if args.redo else read_cache(cache_path)
    todo = [j for j in jobs if (j[1], j[2]) not in cache]
    print(f"[plan] {len(jobs)} case(s), {len(jobs) - len(todo)} cached, {len(todo)} to compute "
          f"({args.workers} workers)", flush=True)
    if todo:
        new_file = args.redo or not os.path.exists(cache_path)
        with open(cache_path, "w" if new_file else "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            with Pool(args.workers) as pool:
                for i, row in enumerate(pool.imap_unordered(case_metrics, todo), 1):
                    w.writerow(row); fh.flush()
                    cache[(row["dataset"], row["case"])] = row
                    if i % 50 == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)}", flush=True)

    rows = [cache[(j[1], j[2])] for j in jobs]
    th = thresholds(rows, bad_ref, args.bad_factor)
    print(f"\nthresholds (good ref {GOOD_REF}: cycle 2" + (f"; bad ref {bad_ref}: cycle 1)" if bad_ref else ")"))
    for (field, metric), t in th.items():
        print(f"  {field:3s} {metric:10s} {GOOD_REF}: median {t['good_median']:.3g} p99 {t['good']:.3g} "
              f"max {t['good_max']:.3g}"
              + (f" | {bad_ref}: median {t['bad_median']:.3g}, {t['n_bad_in_good_range']}/{t['n_bad']} "
                 f"inside the {GOOD_REF} range" if "n_bad" in t else "")
              + f"\n      -> good <= {t['good']:.3g}, bad >= {t['bad']:.3g} ({t['bad_rule']})")
    if bad_ref:
        looks_ok = [r["case"] for r in rows if r["dataset"] == bad_ref and classify(r, th) == "good"]
        if looks_ok:
            print(f"  [warn] {len(looks_ok)} bad-reference case(s) look periodic -- check --bad_ref_cases: "
                  f"{looks_ok[:10]}" + (" ..." if len(looks_ok) > 10 else ""))

    target = [r for r in rows if r["dataset"] == TARGET]
    for r in rows:
        r["class"] = classify(r, th)                  # references too: a check that they land where expected
    for tag in [GOOD_REF] + ([bad_ref] if bad_ref else []):
        c = [r["class"] for r in rows if r["dataset"] == tag]
        print(f"  check {tag:10s} " + "  ".join(f"{k} {c.count(k)}" for k in ("good", "fishy", "bad")))
    classes = {k: [r["case"] for r in sorted(target, key=severity, reverse=True) if r["class"] == k]
               for k in ("bad", "fishy", "good", "missing")}
    print(f"\n{TARGET}: " + "  ".join(f"{k} {len(v)}" for k, v in classes.items()))

    with open(cache_path, "w", newline="") as fh:            # rewrite with the classes
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader(); w.writerows(sorted(cache.values(), key=lambda r: (r["dataset"], r["case"])))
    with open(os.path.join(out, "classes.json"), "w") as f:
        json.dump({"dataset": TARGET, "good_reference": GOOD_REF, "bad_reference": bad_ref,
                   "thresholds": {f"{f}_{m}": t for (f, m), t in th.items()}, **classes}, f, indent=1)
    plot_distribution(rows, th, [GOOD_REF] + ([bad_ref] if bad_ref else []) + [TARGET],
                      os.path.join(out, "distribution.png"))
    print(f"-> {cache_path}\n-> {os.path.join(out, 'classes.json')}\n-> {os.path.join(out, 'distribution.png')}")

    if args.render != "none":
        want = {"fishy": {"fishy"}, "bad": {"bad"}, "flagged": {"fishy", "bad"}}[args.render]
        sel = sorted((r for r in target if r["class"] in want), key=severity, reverse=True)[:args.max_render]
        if sel:
            bands = {ref: step_band(args.root, ref, [r["case"] for r in rows if r["dataset"] == ref])
                     for ref in [GOOD_REF] + ([bad_ref] if bad_ref else [])}
        for r in sel:
            d = os.path.join(out, "render", r["case"])
            if os.path.exists(os.path.join(d, "vel.gif")) and not args.redo:
                continue
            render_case(args.root, r["case"], r, bands, d)
            print(f"[render] {r['case']} ({r['class']}) -> {d}", flush=True)


if __name__ == "__main__":
    main()
