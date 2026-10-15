"""Do AneuGv2 / AngioFlow simulations actually run with the outlet-size flow split?

For every two-outlet case: the split the CFD setup's rule predicts from the outlet caps,
Q0/Q1 = (d0/d1)^2.45 (data/flow_split.py flow_split_from_geometry), vs the outlet_0 share
the simulation actually produced, read off its frame-0 velocity field
(measured_outlet0_share). Reports agreement, whether a REVERSED order fits better (as in
~11% of AneuG-Flow cases), and the stored flowsplit_ratio.txt where a case has one.
AneuG-Flow can be included as a contrast.

    python debugging/verify_flow_split_formula.py                       # angioflow + aneugv2
    python debugging/verify_flow_split_formula.py --datasets aneug_flow angioflow aneugv2 --out /tmp/v.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import torch

from angioflow.data.flow_split import OUTLET_1, flow_split_from_geometry, measured_outlet0_share

ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--datasets", nargs="+", default=["angioflow", "aneugv2"])
    p.add_argument("--out", default=None, help="per-case CSV")
    args = p.parse_args()

    rows = []
    for tag in args.datasets:
        for path in sorted(glob.glob(os.path.join(args.root, tag, f"static_{tag}_*.pt"))):
            cid = os.path.basename(path)[len(f"static_{tag}_"):-len(".pt")]
            s = torch.load(path, weights_only=False, mmap=True)
            nt = s["node_type"].numpy()
            if not (nt == OUTLET_1).any():
                continue
            it = torch.load(os.path.join(args.root, "interior", f"interior_static_{tag}_{cid}.pt"),
                            weights_only=False, mmap=True)
            vel = torch.load(os.path.join(args.root, "interior", f"interior_{tag}_{cid}_0.pt"),
                             weights_only=False, mmap=True)["vel"].numpy()
            meas = measured_outlet0_share(s["pos"].numpy(), nt, it["pos"].numpy(), vel)
            fs = s["flow_split"].double().numpy().reshape(-1)[:2]
            rows.append(dict(dataset=tag, case=cid,
                             formula_q0=float(flow_split_from_geometry(s["pos"].numpy(), nt)[0]),
                             measured_q0=meas,
                             stored_q0=float(fs[0] / fs.sum()) if fs.sum() > 0 else np.nan,
                             aneurysm_type=int(s["aneurysm_type"])))

    print(f"{'dataset':10s} {'n':>4s} | formula vs simulation: median|d|   p95|d|    max|d|  corr   "
          f"| reversed order fits better | >0.03 off")
    for tag in args.datasets:
        r = [x for x in rows if x["dataset"] == tag and np.isfinite(x["measured_q0"])]
        if not r:
            continue
        f = np.array([x["formula_q0"] for x in r]); m = np.array([x["measured_q0"] for x in r])
        d = np.abs(f - m)
        rev = np.abs((1 - f) - m) < d
        uneven = np.abs(f - 0.5) >= 0.02
        print(f"{tag:10s} {len(r):4d} | {np.median(d):22.4f} {np.percentile(d, 95):8.4f} {d.max():9.4f} "
              f"{np.corrcoef(f, m)[0, 1]:+6.3f} | {int((rev & uneven).sum()):3d} of {int(uneven.sum())} uneven"
              f"{'':12s}| {int((d > 0.03).sum())}")
        has = [x for x in r if np.isfinite(x["stored_q0"])]
        if has:
            sd = np.array([abs(x["formula_q0"] - x["stored_q0"]) for x in has])
            print(f"{'':10s}      with a stored file ({len(has)}): formula vs file max|d| {sd.max():.4f}; "
                  f"without ({len(r) - len(has)}): formula vs simulation median|d| "
                  f"{np.median([abs(x['formula_q0'] - x['measured_q0']) for x in r if not np.isfinite(x['stored_q0'])]) if len(r) > len(has) else float('nan'):.4f}")
        worst = sorted(r, key=lambda x: -abs(x["formula_q0"] - x["measured_q0"]))[:5]
        for x in worst:
            print(f"{'':10s}      worst: {x['case']:30s} type {x['aneurysm_type']} formula {x['formula_q0']:.3f} "
                  f"measured {x['measured_q0']:.3f} stored {x['stored_q0']:.3f}")
    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-case table -> {args.out}")


if __name__ == "__main__":
    main()
