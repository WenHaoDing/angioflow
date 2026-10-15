"""Write the outlet flow-split boundary condition of every case, one file per dataset,
<root>/<tag>/flow_split.json, in the registration's outlet order -- what the dataset feeds
the model with add_flow_split=True ([q_outlet0, q_outlet1], q_outlet0 for node_type 3).

Run once per dataset on an extracted AngioFlow-PyG root (wall statics + interior data).
How a dataset's split is obtained is set in METHODS; a dataset not listed there (e.g. a
new one) uses the outlet-size formula -- override with --method:

    aneug_flow   the case's flowsplit_ratio.txt values (copied into its registration /
                 static as flow_split -- the ground truth; they can't be recomputed from
                 outlet size). The file's order does not always match the registration's
                 outlets (~11% of cases are reversed), so the outlet_0 share the simulation
                 actually applied is measured from its frame-0 velocity field, and the pair
                 is swapped when the measurement matches the reversed order.
    angioflow    the size rule the CFD setup used, Q0/Q1 = (d0/d1)^2.45 from the outlet caps
    aneugv2      (reproduces every stored split to <= 0.0013 and fills the missing ones).
    single outlet [1, 0].

Each entry records where its value came from (file / file_swapped / file_unverified /
formula / single_outlet) and, for aneug_flow, the measured share and its distance to the
chosen order, so every decision can be checked:

    {"dataset": "aneug_flow", "method": "file_order_checked", "exponent": 2.45, ...,
     "cases": {"stable_0": {"q": [0.662, 0.338], "source": "file", "measured_q0": 0.66, ...}, ...}}

    python -m angioflow.data.generate_flow_split                          # every dataset in the root
    python -m angioflow.data.generate_flow_split --datasets aneugv2
    python -m angioflow.data.generate_flow_split --datasets <new_tag> --method file_order_checked
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from angioflow.data.dataset_tags import DATASET_TAGS
from angioflow.data.flow_split import (FLOW_SPLIT_EXPONENT, OUTLET_1, flow_split_from_geometry,
                                       measured_outlet0_share)

ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG"
# how each dataset's split is obtained; unlisted datasets default to "formula"
#   file_order_checked  the stored flowsplit_ratio.txt values, put in the registration's outlet
#                       order by measuring which order the simulation applied (needs interior data)
#   formula             Q0/Q1 = (d0/d1)^2.45 from the outlet caps (verified against the simulations
#                       of AngioFlow / AneuGv2: debugging/verify_flow_split_formula.py)
METHODS = {"aneug_flow": "file_order_checked", "angioflow": "formula", "aneugv2": "formula"}
NEAR_EVEN = 0.02      # |q0 - 0.5| below this: order can't be told from the flow, and barely matters


def file_order_checked_entry(static, root, tag, cid):
    stored = static["flow_split"].double().numpy().reshape(-1)[:2]
    q = stored / stored.sum()
    frame = os.path.join(root, "interior", f"interior_{tag}_{cid}_0.pt")
    meas = np.nan
    if os.path.exists(frame):
        it = torch.load(os.path.join(root, "interior", f"interior_static_{tag}_{cid}.pt"),
                        weights_only=False, mmap=True)
        vel = torch.load(frame, weights_only=False, mmap=True)["vel"].numpy()
        meas = measured_outlet0_share(static["pos"].numpy(), static["node_type"].numpy(),
                                      it["pos"].numpy(), vel)
    entry = {"stored_q0": float(q[0]), "measured_q0": None if np.isnan(meas) else float(meas)}
    if np.isnan(meas):
        entry.update(q=[float(q[0]), float(q[1])], source="file_unverified")
    elif abs(meas - q[1]) < abs(meas - q[0]):
        entry.update(q=[float(q[1]), float(q[0])], source="file_swapped")
    else:
        entry.update(q=[float(q[0]), float(q[1])], source="file")
    if entry["measured_q0"] is not None:
        entry["measured_minus_q0"] = float(meas - entry["q"][0])
        entry["near_even"] = bool(abs(q[0] - 0.5) < NEAR_EVEN)
    return entry


def formula_entry(static):
    nt = static["node_type"].numpy()
    q = flow_split_from_geometry(static["pos"].numpy(), nt)
    return {"q": [float(q[0]), float(q[1])],
            "source": "formula" if (nt == OUTLET_1).any() else "single_outlet"}


def flow_split_file(root, tag, per_version_dirs=True):
    """Where a dataset's table lives -- next to its graphs (also used by the dataset class)."""
    return (os.path.join(root, tag, "flow_split.json") if per_version_dirs
            else os.path.join(root, f"flow_split_{tag}.json"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT, help="extracted AngioFlow-PyG root")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="dataset tags (default: every <tag>/ folder in the root with static files)")
    p.add_argument("--method", choices=["formula", "file_order_checked"], default=None,
                   help="override METHODS for the selected datasets")
    args = p.parse_args()
    tags = args.datasets or sorted(t for t in DATASET_TAGS
                                   if glob.glob(os.path.join(args.root, t, f"static_{t}_*.pt")))

    for tag in tags:
        method = args.method or METHODS.get(tag, "formula")
        paths = sorted(glob.glob(os.path.join(args.root, tag, f"static_{tag}_*.pt")))
        print(f"[{tag}] {len(paths)} cases, method {method}", flush=True)
        cases = {}
        for i, path in enumerate(paths):
            cid = os.path.basename(path)[len(f"static_{tag}_"):-len(".pt")]
            s = torch.load(path, weights_only=False, mmap=True)
            cases[cid] = (file_order_checked_entry(s, args.root, tag, cid) if method == "file_order_checked"
                          else formula_entry(s))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(paths)}", flush=True)

        out = flow_split_file(args.root, tag)
        with open(out, "w") as f:
            json.dump({"dataset": tag, "method": method, "exponent": FLOW_SPLIT_EXPONENT,
                       "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "order": "q = [q_outlet0, q_outlet1], outlet_0 = node_type 3 of the registration",
                       "cases": cases}, f, indent=1)
        src = {s_: sum(e["source"] == s_ for e in cases.values()) for s_ in
               ("file", "file_swapped", "file_unverified", "formula", "single_outlet")}
        print(f"  wrote {len(cases)} cases -> {out}: " + "  ".join(f"{k} {v}" for k, v in src.items() if v))
        if method == "file_order_checked":
            res = np.array([abs(e["measured_minus_q0"]) for e in cases.values() if "measured_minus_q0" in e])
            sw_even = sum(e["source"] == "file_swapped" and e.get("near_even") for e in cases.values())
            print(f"  measured vs chosen order: median |d| {np.median(res):.4f}, p99 {np.percentile(res, 99):.4f}, "
                  f"max {res.max():.4f} | swaps decided on a near-even split (<{NEAR_EVEN} from 0.5): {sw_even}")


if __name__ == "__main__":
    main()
