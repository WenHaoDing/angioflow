"""Rename AngioFlow CFD case folders from case_X to their original names via the registry.

Run this ONCE before register_cfd_shapes_angioflow.py.  After renaming, the
folder name is the canonical case identifier and no further registry lookups are
needed during processing.
"""

from __future__ import annotations
import os
import glob


def rename_cases(dataset_root, registry_df, dry_run=False):
    """Rename every case_X folder in dataset_root to its original_name from registry_df.

    Skips entries where the target folder already exists (idempotent).
    Returns {old_name: new_name} for all processed entries.
    """
    renamed = {}
    for case_dir in sorted(glob.glob(os.path.join(dataset_root, "case_*"))):
        name = os.path.basename(case_dir)
        try:
            case_idx = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        row = registry_df[registry_df["index"] == case_idx]
        if row.empty:
            print(f"[warn] {name}: not in registry, skipping")
            continue
        original_name = str(row.iloc[0]["original_name"])
        new_dir = os.path.join(dataset_root, original_name)
        if os.path.exists(new_dir):
            print(f"[skip] {name} -> {original_name}: target already exists")
            renamed[name] = original_name
            continue
        print(f"{'[dry]' if dry_run else '[rename]'} {name} -> {original_name}")
        if not dry_run:
            os.rename(case_dir, new_dir)
        renamed[name] = original_name
    print(f"\n[done] {len(renamed)} folders {'(dry run)' if dry_run else 'renamed'}")
    return renamed


if __name__ == "__main__":
    import pandas as pd

    dataset_root  = "/media/yaplab2/Elements/AngioFlow/processed"
    registry_path = "/media/yaplab2/HDD Storage/wenhao/AneuSeg/cfd_meshing/case_registry.xlsx"

    registry_df = pd.read_excel(registry_path)
    rename_cases(dataset_root, registry_df, dry_run=False)


"""
conda activate new
python -m angioflow.data.rename_angioflow_cases

"""
