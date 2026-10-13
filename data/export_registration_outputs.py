"""Copy the per-case files register_cfd_shapes_aneug_flow.py writes into a
separate destination tree, one subfolder per case (same case name preserved),
for handing off to a collaborator without them having to re-run registration.

Usage:
  conda activate new
  python -m angioflow.data.export_registration_outputs \
    --dataset_root /media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data \
    --dest_root /media/yaplab2/wd8tb/wenhao/datasets/angioflowv1_registration_export

cd /media/yaplab2/wd8tb/wenhao/datasets && zip -r angioflowv1_registration_export.zip angioflowv1_registration_export


"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

FILES_TO_COPY = ["wall_mesh_registration.pt", "shape_remeshed_cfd.obj", "remesh_preview.png"]


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", default="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data")
    p.add_argument("--dest_root", required=True)
    p.add_argument("--case_pattern", default="stable_*")
    return p.parse_args()


def main():
    args = get_args()
    case_dirs = sorted(d for d in glob.glob(os.path.join(args.dataset_root, args.case_pattern))
                       if os.path.isdir(d))
    print(f"[found] {len(case_dirs)} case dirs under {args.dataset_root}")

    n_copied, n_missing = 0, 0
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)
        dest_case_dir = os.path.join(args.dest_root, name)
        for fname in FILES_TO_COPY:
            src = os.path.join(case_dir, fname)
            if not os.path.exists(src):
                print(f"[missing] {name}/{fname}")
                n_missing += 1
                continue
            os.makedirs(dest_case_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dest_case_dir, fname))
            n_copied += 1

    print(f"[done] copied {n_copied} files ({n_missing} missing) -> {args.dest_root}")


if __name__ == "__main__":
    main()
