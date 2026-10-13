"""Rename the SIMPLIFIED case folders in angioflowv2_merged back to their FULL
(original fitting) names, so register_cfd_shapes_angioflow.py's preprocess_dir_list
lookups and any other full-name-keyed tool can find them directly, with no
simplify_name() gymnastics.

Why this exists: the dataset currently mixes short ids ("C0064", "ANSYS09") with a
few full ones ("C0064_cut2") picked up by the recent rsync. register_cfd_shapes_
angioflow.py's AneuSeg registry speaks in full names (original_name), so short
folders are invisible to it ("case dir not found").

THE MAPPING, and why it needs a dry run. There is no file inside angioflowv2_merged
that records a case's full name -- the CFD dataset's own case_registry.xlsx
("original_name" column) just duplicates the short id, it is not the answer. The
only place full names actually live is the GHD fitting tree
(AneuG/runtime_dataset/assembled/<AneuX|ImperialNHS>/<full_name>/), so a short
folder is matched to whichever full name SIMPLIFIES to it. That mapping can be
wrong two ways, both reported rather than guessed:

  ambiguous   several full names simplify to the same short id (e.g. SNF365 <-
              two different patients' fits). Never renamed.
  ok          exactly one full name simplifies to this short id -- but a string
              match alone does not prove they are the SAME PHYSICAL CASE, only
              that no other candidate collides. Where place_v2_by_name.py has
              already placed this case (placement_type0/placement_manifest.csv),
              its geometry check (the fitted surface actually landing on this
              case's own CFD wall, to <0.3 mm) is real content verification and
              is shown alongside the rename; where it hasn't, the rename is
              still only a name match and is flagged UNVERIFIED.

Nothing is renamed without --execute, and even then --execute alone prints the
same dry-run table and stops -- pass --execute --yes to actually touch the
filesystem. Besides the folder, --execute also updates angioflowv2_merged/_meta/
case_registry.xlsx's "case" column (every downstream script keys off it) and any
matching files in angioflowv2_merged/_sanity/, both with a .bak left behind.

    python -m angioflow.data.rename_angioflow_merged_to_full_names                  # dry run
    python -m angioflow.data.rename_angioflow_merged_to_full_names --execute --yes  # do it
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, os.path.join(REPO, "angioflow")):
    if p not in sys.path:
        sys.path.insert(0, p)

from angioflow.data.rename_geometry_folder import simplify_name  # noqa: E402

MERGED_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
ASSEMBLED_ROOT = "/media/yaplab2/HDD Storage/wenhao/AneuG/runtime_dataset/assembled"
REGISTRY_PATH = os.path.join(MERGED_ROOT, "_meta", "case_registry.xlsx")
PLACEMENT_MANIFEST = os.path.join(
    REPO, "ood_fix", "cfd_data_factory", "placement_type0", "placement_manifest.csv")
DATASETS = ("AneuX", "ImperialNHS")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged_root", default=MERGED_ROOT)
    p.add_argument("--assembled_root", default=ASSEMBLED_ROOT)
    p.add_argument("--registry", default=REGISTRY_PATH)
    p.add_argument("--placement_manifest", default=PLACEMENT_MANIFEST)
    p.add_argument("--execute", action="store_true", help="actually rename (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def full_name_candidates(assembled_root):
    """simplified name -> [(dataset, full_name), ...] from the GHD fitting tree."""
    cands = collections.defaultdict(list)
    for ds in DATASETS:
        root = os.path.join(assembled_root, ds)
        if not os.path.isdir(root):
            continue
        for n in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, n)):
                cands[simplify_name(n)].append((ds, n))
    return cands


def load_placement_info(path):
    """CFD case -> manifest row, for the geometric-verification column."""
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return {r["case"]: r for r in csv.DictReader(fh)}


def plan(args):
    cands = full_name_candidates(args.assembled_root)
    all_full_names = {n for v in cands.values() for _, n in v}
    placement = load_placement_info(args.placement_manifest)

    current = sorted(d for d in os.listdir(args.merged_root)
                     if os.path.isdir(os.path.join(args.merged_root, d)) and not d.startswith((".", "_")))

    rows = []
    for name in current:
        if name in all_full_names:
            rows.append({"current": name, "full_name": name, "status": "already_full"})
            continue
        c = cands.get(name, [])
        if not c:
            rows.append({"current": name, "full_name": "", "status": "no_candidate"})
        elif len(c) > 1:
            rows.append({"current": name, "full_name": "", "status": "ambiguous",
                        "candidates": "; ".join(f"{ds}/{n}" for ds, n in c)})
        else:
            ds, full = c[0]
            pr = placement.get(name)
            if pr and pr.get("status") == "placed":
                verify = f"VERIFIED geometry: median {float(pr['median_mm']):.3f} mm, " \
                        f"coverage {100 * float(pr['coverage']):.0f}%"
            elif pr:
                verify = f"placement status={pr['status']} (not a confirmed match)"
            else:
                verify = "UNVERIFIED (name match only, no placement check)"
            rows.append({"current": name, "full_name": full, "status": "ok",
                        "dataset": ds, "verify": verify})
    return rows


def main():
    args = parse_args()
    rows = plan(args)
    by_status = collections.Counter(r["status"] for r in rows)
    print(f"[scan] {len(rows)} case folder(s) in {args.merged_root}")
    print(f"       {dict(by_status)}\n")

    ok = [r for r in rows if r["status"] == "ok"]
    verified = [r for r in ok if r["verify"].startswith("VERIFIED")]
    unverified = [r for r in ok if r["verify"].startswith("UNVERIFIED")]
    print(f"proposed renames: {len(ok)}  ({len(verified)} geometry-verified, "
          f"{len(unverified)} name-match only)\n")

    print(f"{'current':26s} -> {'full name':36s} {'verification'}")
    print("-" * 110)
    for r in sorted(ok, key=lambda r: r["verify"].startswith("UNVERIFIED")):
        print(f"{r['current']:26s} -> {r['full_name']:36s} {r['verify']}")

    amb = [r for r in rows if r["status"] == "ambiguous"]
    if amb:
        print(f"\nAMBIGUOUS (not renamed, needs a human decision) -- {len(amb)}:")
        for r in amb:
            print(f"   {r['current']:20s} candidates: {r['candidates']}")

    no_cand = [r for r in rows if r["status"] == "no_candidate"]
    if no_cand:
        print(f"\nno full name found for {len(no_cand)} folder(s) (kept as-is): "
              f"{[r['current'] for r in no_cand][:15]}"
              f"{' ...' if len(no_cand) > 15 else ''}")

    if not args.execute:
        print("\n[dry run only] pass --execute --yes to actually rename.")
        return
    if not args.yes:
        print("\n[dry run only] --execute given without --yes -- not touching anything.")
        return

    # ---- apply --------------------------------------------------------------
    print(f"\n[execute] renaming {len(ok)} folder(s) ...")
    renamed = {}
    for r in ok:
        src = os.path.join(args.merged_root, r["current"])
        dst = os.path.join(args.merged_root, r["full_name"])
        if os.path.exists(dst):
            print(f"  [skip] {r['current']} -> {r['full_name']}: target already exists")
            continue
        os.rename(src, dst)
        renamed[r["current"]] = r["full_name"]
        print(f"  [ok] {r['current']} -> {r['full_name']}")

    sanity_dir = os.path.join(args.merged_root, "_sanity")
    if os.path.isdir(sanity_dir):
        import glob
        for old, new in renamed.items():
            for f in glob.glob(os.path.join(sanity_dir, old)) + \
                     glob.glob(os.path.join(sanity_dir, f"{old}_*")) + \
                     glob.glob(os.path.join(sanity_dir, f"{old}.*")):
                suffix = os.path.basename(f)[len(old):]
                nf = os.path.join(sanity_dir, new + suffix)
                if not os.path.exists(nf):
                    os.rename(f, nf)

    if renamed and os.path.isfile(args.registry):
        import pandas as pd
        shutil.copy(args.registry, args.registry + ".bak")
        df = pd.read_excel(args.registry)
        df["case"] = df["case"].astype(str).map(lambda c: renamed.get(c, c))
        df.to_excel(args.registry, index=False)
        print(f"\n[registry] updated {args.registry} ({len(renamed)} case ids), "
              f"backup at {args.registry}.bak")

    print(f"\n[done] {len(renamed)} folder(s) renamed.")


if __name__ == "__main__":
    main()
