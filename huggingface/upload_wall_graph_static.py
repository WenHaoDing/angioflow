"""Upload the per-case wall_graph_static.pt files written by
angioflow.data.precompute_wall_static to each dataset on the Hub, next to the
wall_mesh_registration.pt they were computed from:

    whding123/AneuG-Flow  transient_data/<case>/wall_graph_static.pt
    whding123/AngioFlow   cfd_data/<case>/wall_graph_static.pt
    whding123/AneuGv2     cfd_data/<case>/wall_graph_static.pt

The local staging root already mirrors each repo's layout (<root>/<dataset>/<prefix>/
<case>/wall_graph_static.pt), so a file's repo path is just its path under
<root>/<dataset>/. Sanity PNGs (<root>/<dataset>/_sanity/) are NOT uploaded:
AneuGv2's cfd_data/_sanity/<case>/ already holds the registration's own geo_dist.png,
which a same-named upload here would overwrite.

Only uploads a case whose registration is still the one the file was computed from
(the file's meta.source_sha256 == the Hub registration's LFS sha256), and skips files
whose content is already on the Hub unchanged -- so a rerun only sends what's missing
or changed. Batched commits with retry, like upload_wall_mesh_registration.py.

    python -m angioflow.huggingface.upload_wall_graph_static                   # dry run
    python -m angioflow.huggingface.upload_wall_graph_static --execute --yes   # upload all
    python -m huggingface.upload_wall_graph_static --datasets aneugv2 --execute --yes
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import time

import torch
from huggingface_hub import CommitOperationAdd, HfApi

REPO_TYPE = "dataset"
DATASETS = {  # dataset tag -> (repo_id, case-folder prefix inside the repo)
    "aneug_flow": ("whding123/AneuG-Flow", "transient_data"),
    "angioflow":  ("whding123/AngioFlow",  "cfd_data"),
    "aneugv2":    ("whding123/AneuGv2",    "cfd_data"),
}
LOCAL_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/hf_wall_graph_static"
FNAME = "wall_graph_static.pt"
REG_FNAME = "wall_mesh_registration.pt"
CHUNK_SIZE = 40          # ~13MB/file * 40 ~= 520MB/commit
MAX_RETRIES = 5
RETRY_DELAY = 10         # seconds; doubles each retry


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--local_root", default=LOCAL_ROOT)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true", help="actually upload (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def plan_dataset(api, name, local_root):
    """Return (to_upload [(repo_path, local_path)], report dict)."""
    repo_id, prefix = DATASETS[name]
    local = {os.path.relpath(p, os.path.join(local_root, name)): p
             for p in glob.glob(os.path.join(local_root, name, prefix, "*", FNAME))}
    # remote LFS sha256 of every registration and every already-uploaded static file
    remote = {f.path: f.lfs.sha256 for f in api.list_repo_tree(
                  repo_id, path_in_repo=prefix, repo_type=REPO_TYPE, recursive=True)
              if getattr(f, "lfs", None) is not None
              and os.path.basename(f.path) in (FNAME, REG_FNAME)}

    to_upload, unchanged, stale, no_reg = [], [], [], []
    for repo_path, local_path in sorted(local.items()):
        reg_path = os.path.join(os.path.dirname(repo_path), REG_FNAME)
        if reg_path not in remote:
            no_reg.append(repo_path)
            continue
        if torch.load(local_path, weights_only=False)["meta"]["source_sha256"] != remote[reg_path]:
            stale.append(repo_path)
            continue
        if remote.get(repo_path) == sha256_file(local_path):
            unchanged.append(repo_path)
            continue
        to_upload.append((repo_path, local_path))
    missing = sorted({os.path.dirname(p) for p in remote if p.endswith(REG_FNAME)}
                     - {os.path.dirname(p) for p in local})
    return to_upload, dict(local=len(local), unchanged=len(unchanged), stale=stale,
                           no_registration_on_hub=no_reg, registration_but_no_local_file=missing)


def commit_chunks(api, repo_id, items, name):
    chunks = [items[i:i + CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]
    for ci, chunk in enumerate(chunks, 1):
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for r, l in chunk]
        delay = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                api.create_commit(repo_id=repo_id, repo_type=REPO_TYPE, operations=ops,
                                  commit_message=f"Add {FNAME} for {len(ops)} case(s) "
                                                 f"(batch {ci}/{len(chunks)})")
                print(f"[{name} {ci}/{len(chunks)}] committed {len(ops)} file(s)", flush=True)
                break
            except Exception as e:
                rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
                wait = max(delay, 60) if rate_limited else delay
                print(f"[{name} {ci}/{len(chunks)}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(wait)
                    delay *= 2
                else:
                    raise


def main():
    args = parse_args()
    api = HfApi(token=args.token)
    plans = {}
    for name in args.datasets:
        to_upload, rep = plan_dataset(api, name, args.local_root)
        plans[name] = to_upload
        print(f"[plan] {name} -> {DATASETS[name][0]}: {len(to_upload)} to upload, "
              f"{rep['unchanged']} already on the Hub unchanged (of {rep['local']} local)")
        for key in ("stale", "no_registration_on_hub", "registration_but_no_local_file"):
            if rep[key]:
                print(f"  [warn] {key}: {len(rep[key])} {rep[key][:5]}{'...' if len(rep[key]) > 5 else ''}")

    if not args.execute:
        print("\n[dry run only] pass --execute --yes to upload.")
        return
    if not args.yes:
        print("\n[dry run only] --execute given without --yes -- not touching anything.")
        return
    for name, to_upload in plans.items():
        if to_upload:
            commit_chunks(api, DATASETS[name][0], to_upload, name)
    print("\n[done]")


if __name__ == "__main__":
    main()
