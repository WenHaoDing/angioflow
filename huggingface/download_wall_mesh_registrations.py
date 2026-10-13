"""Download every case's wall_mesh_registration.pt from the three CFD datasets on the
Hub into one local staging root -- the input for precomputing per-case static graph
fields (normals, geodesic distances, cotangent-Laplacian eigenbasis) locally, where
pytorch3d/pyvista are available, before uploading them back next to each case.

Reads from the Hub rather than the local dataset mirrors on purpose: the Hub is the
source of truth that the cloud extraction Job will read, so the precomputed fields
must be derived from exactly those files.

Layout mirrors each repo's own paths under a per-dataset folder:
    <dest>/aneug_flow/transient_data/<case>/wall_mesh_registration.pt
    <dest>/angioflow/cfd_data/<case>/wall_mesh_registration.pt
    <dest>/aneugv2/cfd_data/<case>/wall_mesh_registration.pt

Resumable: snapshot_download keeps per-file metadata under <dest>/<dataset>/.cache/,
so a rerun skips files already downloaded and unchanged on the Hub, and re-fetches
any that were updated there since.

    python -m angioflow.huggingface.download_wall_mesh_registrations --dry_run
    python -m angioflow.huggingface.download_wall_mesh_registrations
    python -m angioflow.huggingface.download_wall_mesh_registrations --datasets aneugv2
"""
from __future__ import annotations

import argparse
import fnmatch
import os

from huggingface_hub import HfApi, snapshot_download

REPO_TYPE = "dataset"
# dataset tag -> (repo_id, registration path pattern inside that repo)
DATASETS = {
    "aneug_flow": ("whding123/AneuG-Flow", "transient_data/*/wall_mesh_registration.pt"),
    "angioflow":  ("whding123/AngioFlow",  "cfd_data/*/wall_mesh_registration.pt"),
    "aneugv2":    ("whding123/AneuGv2",    "cfd_data/*/wall_mesh_registration.pt"),
}
DEST_DEFAULT = "/media/yaplab2/wd8tb/wenhao/datasets/hf_registrations"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--dest", default=DEST_DEFAULT)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="defaults to $HF_TOKEN, else huggingface_hub's cached login")
    p.add_argument("--dry_run", action="store_true", help="count matching files on the Hub and stop")
    return p.parse_args()


def main():
    args = parse_args()
    api = HfApi(token=args.token)

    for name in args.datasets:
        repo_id, pattern = DATASETS[name]
        local_dir = os.path.join(args.dest, name)
        if args.dry_run:
            remote = [f for f in api.list_repo_files(repo_id, repo_type=REPO_TYPE)
                      if fnmatch.fnmatch(f, pattern)]
            print(f"[plan] {name}: {len(remote)} registration(s) in {repo_id} -> {local_dir}")
            continue
        snapshot_download(repo_id, repo_type=REPO_TYPE, token=args.token,
                          allow_patterns=[pattern], local_dir=local_dir,
                          max_workers=args.max_workers)
        n_local = sum(1 for root, _, files in os.walk(local_dir)
                      if "wall_mesh_registration.pt" in files and "/.cache" not in root)
        print(f"[done] {name}: {n_local} registration(s) in {local_dir}")

    if args.dry_run:
        print("\n[dry run only] rerun without --dry_run to download.")


if __name__ == "__main__":
    main()
