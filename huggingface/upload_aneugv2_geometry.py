"""Upload AneuGv2's local geometry-stage case folders to whding123/AneuGv2's geometry/
on the Hub -- the CFD simulation outputs (wall_data.pt/blood_data.pt) for this dataset
live remotely only (cfd_data/, uploaded separately, e.g. via an HPC/Jobs pipeline); this
script carries over everything needed to BUILD wall_mesh_registration.pt against them
later (as a Hugging Face Jobs script, reading both prefixes remotely) except the two
files that pipeline doesn't need: mesh.msh/mesh.vtu are the volume-mesh inputs already
consumed to produce wall_data.pt/blood_data.pt, and dominate case size (~97% of it,
229MB+50MB out of ~286MB for a sample case) for no benefit once registration only reads
geometry (GHD fit, openings, centerlines, ...), never the volume mesh itself.

Only directories named cohort_* under --local_root count as cases -- the root also has
loose log files (volume_mesh_log_*.txt, case_list.txt/xlsx, ...) and a stray .cache/
from a prior huggingface_hub download, none of which are per-case data.

Batched commits, CASE_CHUNK cases each (not one commit per case, not one giant commit):
650 cases as 650 separate commits would hammer the Hub's hourly commit-rate limit --
the exact incident upload_angioflow.py's own docstring describes (615 cases, fixed by
batching). Batching also keeps a dropped connection cheap to retry -- only the current
chunk is lost, not the whole upload -- same reasoning as upload_wall_mesh_registration.py.
Skip-already-uploaded + retry-with-backoff logic mirrors upload_angioflow.py.

    python -m angioflow.huggingface.upload_aneugv2_geometry                  # dry run
    python -m angioflow.huggingface.upload_aneugv2_geometry --execute --yes  # apply
"""
from __future__ import annotations

import argparse
import os
import time

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "whding123/AneuGv2"
LOCAL_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/cfd/AneuGv2"
PATH_IN_REPO_PREFIX = "geometry"
SKIP_FILES = {"mesh.msh", "mesh.vtu"}
CASE_CHUNK = 100    # cases per commit (~20MB/case non-mesh -> ~400MB/commit)
MAX_RETRIES = 5
RETRY_DELAY = 10   # seconds; doubles each retry (important for 429 rate-limit backoff)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--local_root", default=LOCAL_ROOT)
    p.add_argument("--prefix", default=PATH_IN_REPO_PREFIX)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true", help="actually upload (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def _case_dirs(local_root):
    return sorted(d for d in os.listdir(local_root)
                 if d.startswith("cohort_") and os.path.isdir(os.path.join(local_root, d)))


def _remote_sizes(api, repo_id, prefix):
    try:
        return {item.path: item.size for item in
                api.list_repo_tree(repo_id=repo_id, path_in_repo=prefix, recursive=True, repo_type="dataset")
                if hasattr(item, "size")}
    except Exception as e:
        print(f"[scan] could not list existing files ({prefix}/ may not exist yet): {e}")
        return {}


def _case_fully_uploaded(case_dir, case_name, prefix, remote_sizes):
    for fname in os.listdir(case_dir):
        if fname in SKIP_FILES or not os.path.isfile(os.path.join(case_dir, fname)):
            continue
        path_in_repo = f"{prefix}/{case_name}/{fname}"
        if remote_sizes.get(path_in_repo) != os.path.getsize(os.path.join(case_dir, fname)):
            return False
    return True


def main():
    args = parse_args()
    api = HfApi(token=args.token)
    cases = _case_dirs(args.local_root)
    print(f"[scan] {len(cases)} case(s) under {args.local_root}")

    remote_sizes = _remote_sizes(api, args.repo_id, args.prefix)
    todo = [c for c in cases
           if not _case_fully_uploaded(os.path.join(args.local_root, c), c, args.prefix, remote_sizes)]
    print(f"[plan] {len(todo)} case(s) to upload, {len(cases) - len(todo)} already up to date")
    if not args.execute:
        print("\n[dry run only] pass --execute --yes to upload.")
        print("would upload (first 10):", todo[:10])
        return
    if not args.yes:
        print("\n[dry run only] --execute given without --yes -- not touching anything.")
        return

    chunks = [todo[i:i + CASE_CHUNK] for i in range(0, len(todo), CASE_CHUNK)]
    print(f"[upload] {len(todo)} case(s) in {len(chunks)} commit(s) of up to {CASE_CHUNK} cases each")
    for i, chunk in enumerate(chunks, 1):
        ops = [
            CommitOperationAdd(path_in_repo=f"{args.prefix}/{case}/{fname}",
                               path_or_fileobj=os.path.join(args.local_root, case, fname))
            for case in chunk
            for fname in os.listdir(os.path.join(args.local_root, case))
            if fname not in SKIP_FILES and os.path.isfile(os.path.join(args.local_root, case, fname))
        ]
        delay = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=ops,
                                  commit_message=f"Add geometry for {len(chunk)} case(s) "
                                                 f"(batch {i}/{len(chunks)})")
                print(f"[{i}/{len(chunks)}] committed {len(chunk)} case(s), {len(ops)} file(s)")
                break
            except Exception as e:
                rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
                wait = max(delay, 60) if rate_limited else delay
                print(f"[{i}/{len(chunks)}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    print(f"    retrying in {wait}s...")
                    time.sleep(wait)
                    delay *= 2
                else:
                    raise
    print(f"\n[done] uploaded {len(todo)} case(s) to {args.repo_id}/{args.prefix} across {len(chunks)} commit(s)")


if __name__ == "__main__":
    main()
