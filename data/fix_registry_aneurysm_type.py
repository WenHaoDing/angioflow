"""Fix angioflowv2_merged/_meta/case_registry.xlsx's aneurysm_type (and, where the
evidence contradicts it, dataset) column against ground truth.

GROUND TRUTH is each case's own endpoints_manual.npy, found by searching its
geometry folder across all known sources (first hit wins; verified the two
ImperialNHS roots agree on every case present in both):

    AneuX        /media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_stable
    ImperialNHS  /media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS
    ImperialNHS  /media/yaplab2/HDD Storage/wenhao/AneuSeg/PostTr_debug/ImperialNHS

WHY THE REGISTRY NEEDS THIS. An audit of the 467 cases that already have a
wall_mesh_registration.pt found every single one stored aneurysm_type=0 --
register_cfd_shapes_angioflow.py's __main__ hardcodes it instead of looking it
up. Separately, 20 AneuX cases were labelled "ImperialNHS" in the registry (the
dataset column is inferred from geomagic_processed.obj presence in
build_case_registry.py, which those 20 happen to also have) and so had no
aneurysm_type at all until their AneuX geometry folders were tracked down and
consolidated into AneuX_stable.

EVERY row is checked, not just ones with a CFD folder that currently exists under
angioflowv2_merged: dataset/aneurysm_type describe the case's identity, not whether
CFD data happens to be present right now, so a geometry-only row (e.g. C0065_cut2,
backfilled into AneuX/AneuX_stable later but never gated on having CFD data) still
gets corrected.

This script fixes ONLY case_registry.xlsx. It does not touch wall_mesh_
registration.pt (see fix_wall_registration_aneurysm_type.py for that, run
AFTER this one gets the registry right) or the dataset itself.

Dry run by default: prints what would change and does not write anything.
--execute --yes writes the fix, after backing up the registry and (b) saving a
JSON log of every change (old value, new value, source file) so it is auditable
and reversible.

    python -m angioflow.data.fix_registry_aneurysm_type                  # dry run
    python -m angioflow.data.fix_registry_aneurysm_type --execute --yes  # apply
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import shutil

import numpy as np
import pandas as pd

MERGED_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
REGISTRY_PATH = os.path.join(MERGED_ROOT, "_meta", "case_registry.xlsx")
SOURCES = [
    ("AneuX", "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_stable"),
    ("ImperialNHS", "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"),
    ("ImperialNHS", "/media/yaplab2/HDD Storage/wenhao/AneuSeg/PostTr_debug/ImperialNHS"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", default=REGISTRY_PATH)
    p.add_argument("--merged_root", default=MERGED_ROOT)
    p.add_argument("--execute", action="store_true", help="actually write (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def ground_truth(case, sources):
    """First source whose geometry folder has this case, with its endpoints_manual.npy
    content. Returns (dataset, aneurysm_type, source_path) or (None, None, None)."""
    for dataset, root in sources:
        path = os.path.join(root, case, "endpoints_manual.npy")
        if os.path.isfile(path):
            try:
                d = np.load(path, allow_pickle=True).item()
                return dataset, int(d["aneurysm_type"]), path
            except Exception as exc:
                print(f"  [warn] {case}: unreadable {path}: {exc}")
    return None, None, None


def plan(args):
    reg = pd.read_excel(args.registry)
    reg["case"] = reg["case"].astype(str)

    rows = []
    for i, r in reg.iterrows():
        case = r["case"]
        # NOTE: every row is checked, not just ones with a CURRENT CFD folder in
        # merged_root -- dataset/aneurysm_type are properties of the case's identity
        # (which geometry source it actually came from), not of whether CFD data
        # happens to exist right now. A row like C0065_cut2 had no CFD folder yet
        # still carried a wrong "ImperialNHS" label inferred by build_case_registry.py
        # from geomagic_processed.obj presence -- leaving it unchecked just because its
        # CFD folder was absent left that wrong label sitting in the registry.
        cur_type = None if pd.isna(r.get("aneurysm_type")) else int(r["aneurysm_type"])
        cur_dataset = None if pd.isna(r.get("dataset")) else str(r["dataset"])

        gt_dataset, gt_type, src = ground_truth(case, SOURCES)
        if gt_type is None:
            status = "no_ground_truth"
        elif cur_type == gt_type and cur_dataset == gt_dataset:
            status = "already_correct"
        elif cur_type is None:
            status = "filled_missing"
        else:
            status = "corrected"
        rows.append({"index": i, "case": case, "cur_dataset": cur_dataset, "cur_type": cur_type,
                    "gt_dataset": gt_dataset, "gt_type": gt_type, "source": src, "status": status})
    return reg, rows


def main():
    args = parse_args()
    reg, rows = plan(args)
    by_status = collections.Counter(r["status"] for r in rows)
    print(f"[scan] {len(rows)} case(s) in {args.registry}")
    print(f"       {dict(by_status)}\n")

    changed = [r for r in rows if r["status"] in ("filled_missing", "corrected")]
    dataset_changed = [r for r in changed if r["cur_dataset"] != r["gt_dataset"]]
    print(f"aneurysm_type will change for {len(changed)} case(s) "
          f"({sum(r['status']=='filled_missing' for r in changed)} filled in, "
          f"{sum(r['status']=='corrected' for r in changed)} corrected)")
    print(f"dataset column will ALSO change for {len(dataset_changed)} of those\n")

    if dataset_changed:
        print(f"{'case':26s} {'dataset: old -> new':28s} {'type: old -> new'}")
        print("-" * 90)
        for r in dataset_changed:
            print(f"{r['case']:26s} {str(r['cur_dataset']):12s} -> {r['gt_dataset']:12s}   "
                 f"{str(r['cur_type']):>4s} -> {r['gt_type']}")
        print()

    same_dataset_changed = [r for r in changed if r not in dataset_changed]
    if same_dataset_changed:
        print(f"type-only changes (dataset unchanged), first 20 of {len(same_dataset_changed)}:")
        for r in same_dataset_changed[:20]:
            print(f"   {r['case']:26s} {str(r['cur_type']):>4s} -> {r['gt_type']}")
        print()

    no_gt = [r for r in rows if r["status"] == "no_ground_truth"]
    if no_gt:
        print(f"no ground truth found for {len(no_gt)} case(s) (left unchanged): "
              f"{[r['case'] for r in no_gt]}\n")

    if not args.execute:
        print("[dry run only] pass --execute --yes to write the fix.")
        return
    if not args.yes:
        print("[dry run only] --execute given without --yes -- not touching anything.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.registry + f".bak.{stamp}"
    shutil.copy(args.registry, backup)

    for r in changed:
        reg.at[r["index"], "aneurysm_type"] = r["gt_type"]
        if r["cur_dataset"] != r["gt_dataset"]:
            reg.at[r["index"], "dataset"] = r["gt_dataset"]
    reg.to_excel(args.registry, index=False)

    log_path = os.path.join(os.path.dirname(args.registry), f"fix_aneurysm_type_{stamp}.json")
    with open(log_path, "w") as fh:
        json.dump({"started": stamp, "registry": args.registry, "backup": backup,
                   "changed": changed, "no_ground_truth": no_gt}, fh, indent=2)

    print(f"[done] {len(changed)} row(s) updated in {args.registry}")
    print(f"[backup] {backup}")
    print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
