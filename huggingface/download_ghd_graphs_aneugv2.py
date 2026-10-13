"""Download whding123/AneuGv2's ghd_graphs/ghd_graphs_aneugv2.zip (built by the
cloud Job run of ood_fix/cfd_data_factory/extract_ghd_graphs_aneugv2.py) and unzip
it into the shared local GHD-graph training dataset's AneuGv2/ subfolder.

The zip's own internal layout already matches that subfolder exactly --
static_aneugv2_<case>.pt / wss_aneugv2_<case>_<t>.pt / extraction_manifest.csv at
its root, sanity/ as a subfolder -- the same convention
ood_fix/cfd_data_factory/extract_wss_dataset.py's AneuGv1/ and Angioflow/
subfolders use and ood_fix/dataset.py's VERSION_SUBDIR reads (see that file), so
extraction here is a direct unzip, no restructuring.

Extraction MERGES into --dest rather than clearing it first: a file already there
with the same name is overwritten, anything else already there is left alone (so
this is safe to rerun against a newer zip, or point at a --dest that also holds a
previous local-run's output).

    python -m angioflow.huggingface.download_ghd_graphs_aneugv2 --dry_run
    python -m angioflow.huggingface.download_ghd_graphs_aneugv2
"""
from __future__ import annotations

import argparse
import os
import zipfile

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "whding123/AneuGv2"
REPO_TYPE = "dataset"
REMOTE_PATH = "ghd_graphs/ghd_graphs_aneugv2.zip"
DEST_DEFAULT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_ghd_graphs/AneuGv2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--remote_path", default=REMOTE_PATH)
    p.add_argument("--dest", default=DEST_DEFAULT)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="defaults to $HF_TOKEN, else huggingface_hub's cached login")
    p.add_argument("--dry_run", action="store_true", help="show what would happen and stop")
    return p.parse_args()


def main():
    args = parse_args()
    api = HfApi(token=args.token)

    info = next((f for f in api.list_repo_tree(
                    repo_id=args.repo_id, path_in_repo=os.path.dirname(args.remote_path),
                    repo_type=REPO_TYPE, recursive=True)
                if f.path == args.remote_path), None)
    if info is None:
        print(f"[fail] {args.remote_path} not found in {args.repo_id} -- "
              f"has the extraction Job finished and uploaded it yet?")
        return
    print(f"[plan] {args.remote_path}  ({info.size / 1e6:.1f} MB)  -> {args.dest}")
    if args.dry_run:
        print("\n[dry run only] rerun without --dry_run to download + unzip.")
        return

    zip_path = hf_hub_download(repo_id=args.repo_id, repo_type=REPO_TYPE, token=args.token,
                               filename=args.remote_path)
    print(f"[download] cached at {zip_path}")

    os.makedirs(args.dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(args.dest)
    n_static = sum(1 for n in names if n.startswith("static_"))
    n_sanity = sum(1 for n in names if n.startswith("sanity/"))
    print(f"[done] extracted {len(names)} file(s) ({n_static} case(s), "
          f"{n_sanity} sanity file(s)) -> {args.dest}")


if __name__ == "__main__":
    main()
