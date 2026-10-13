"""Upload local wall_mesh_registration.pt files to the matching case folders in
whding123/AneuG-Flow's transient_data/ on the Hub, in a single commit.

Case-name check first: the Hub repo had its 96 CFD-anomaly cases removed (see
remove_aneug_flow_anomalies.py) while all 730 local case folders still have the file, so the
two sides are NOT 1:1 -- only cases present on BOTH sides get uploaded. A case local-
only (one of the removed anomalies) is skipped on purpose, not an error; a case
remote-only (shouldn't happen -- every local case has the file) would mean the file
is genuinely missing locally and is reported, not silently skipped.

    python -m angioflow.huggingface.upload_wall_mesh_registration                  # dry run
    python -m angioflow.huggingface.upload_wall_mesh_registration --execute --yes  # apply
"""
from __future__ import annotations

import argparse
import os
import time

from huggingface_hub import CommitOperationAdd, HfApi

REPO_ID = "whding123/AneuG-Flow"
PREFIX = "transient_data"
LOCAL_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data"
FNAME = "wall_mesh_registration.pt"
CHUNK_SIZE = 80          # ~4MB/file * 80 ~= 330MB/commit -- small enough that a dropped
                         # connection only costs one chunk's retry, not the whole upload
MAX_RETRIES = 5
RETRY_DELAY = 10         # seconds; doubles each retry (matches upload_angioflow.py)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--local_root", default=LOCAL_ROOT)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true", help="actually upload (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def main():
    args = parse_args()
    local_cases = {d for d in os.listdir(args.local_root)
                   if os.path.isfile(os.path.join(args.local_root, d, FNAME))}
    print(f"[scan] {len(local_cases)} local case(s) under {args.local_root} have {FNAME}")

    api = HfApi(token=args.token)
    remote_files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    remote_cases = {f.split("/", 2)[1] for f in remote_files if f.startswith(f"{args.prefix}/")}
    print(f"[scan] {len(remote_cases)} case(s) under {args.repo_id}/{args.prefix}")

    both = sorted(local_cases & remote_cases)
    local_only = sorted(local_cases - remote_cases)
    remote_only = sorted(remote_cases - local_cases)
    print(f"\n[plan] {len(both)} case(s) on both sides -> will upload {FNAME}")
    print(f"[plan] {len(local_only)} local-only (not on the Hub -- skipped, not uploaded): "
          f"{local_only[:10]}{'...' if len(local_only) > 10 else ''}")
    if remote_only:
        print(f"[warn] {len(remote_only)} case(s) on the Hub with NO local {FNAME} "
              f"(unexpected -- every local case should have it): {remote_only}")

    if not args.execute:
        print("\n[dry run only] pass --execute --yes to upload.")
        return
    if not args.yes:
        print("\n[dry run only] --execute given without --yes -- not touching anything.")
        return

    chunks = [both[i:i + CHUNK_SIZE] for i in range(0, len(both), CHUNK_SIZE)]
    print(f"\n[upload] {len(both)} case(s) in {len(chunks)} commit(s) of up to {CHUNK_SIZE} each")
    for i, chunk in enumerate(chunks, 1):
        ops = [CommitOperationAdd(path_in_repo=f"{args.prefix}/{c}/{FNAME}",
                                  path_or_fileobj=os.path.join(args.local_root, c, FNAME))
               for c in chunk]
        delay = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=ops,
                                  commit_message=f"Add {FNAME} to {len(chunk)} case(s) "
                                                 f"(batch {i}/{len(chunks)})")
                print(f"  [{i}/{len(chunks)}] committed {len(chunk)} case(s)")
                break
            except Exception as e:
                rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
                wait = max(delay, 60) if rate_limited else delay
                print(f"  [{i}/{len(chunks)}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    print(f"    retrying in {wait}s...")
                    time.sleep(wait)
                    delay *= 2
                else:
                    raise
    print(f"\n[done] uploaded {FNAME} for {len(both)} case(s) across {len(chunks)} commit(s)")


if __name__ == "__main__":
    main()
