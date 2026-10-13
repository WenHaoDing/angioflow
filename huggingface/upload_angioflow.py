"""Upload/manage the AngioFlow dataset on the Hugging Face Hub.

SECURITY: an HF token used to be hardcoded in this file and was committed to git history
(pushed to GitHub) -- if that token is still live, revoke it at
https://huggingface.co/settings/tokens and issue a new one. This script now reads the
token from the HF_TOKEN environment variable only; nothing secret lives in source.

One commit per CASE DIRECTORY (not per file): the original per-file `upload_file` loop
hit the Hub's hourly commit-rate limit on a 615-case, 7260-file dataset. Batching to one
`upload_folder` commit per case directory cuts that ~12x and is still resumable -- a case
already fully present on the remote (matching file set + sizes under
cfd_data/<case>/) is skipped without an API call, so an interrupted run just continues
from wherever it left off. `upload_large_folder` (the Hub's own resumable/rate-limit-aware
uploader) was considered instead, but as of huggingface_hub==1.20.1 it has no
`path_in_repo` param -- it can only target the repo root, not a cfd_data/ subfolder -- so
it isn't used here.

A case is any top-level dir holding wall_data.pt; _meta/, _sanity/, _logs/, processed/
etc. are not uploaded.

--wipe deletes every file in the repo except .gitattributes. Without --yes it only
prints what it would delete.

Usage:
  conda activate new
  export HF_TOKEN=hf_...
  python angioflow/huggingface/upload_angioflow.py --wipe              # dry run
  python angioflow/huggingface/upload_angioflow.py --wipe --yes --upload
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi

LOCAL_BASE_PATH = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
REPO_ID = "whding123/AngioFlow"
REPO_TYPE = "dataset"
PATH_IN_REPO_PREFIX = "cfd_data"   # each future dataset variant gets its own top-level folder
KEEP_ON_WIPE = {".gitattributes"}
WIPE_CHUNK = 2000   # delete ops per commit
MAX_RETRIES = 5
RETRY_DELAY = 10   # seconds; doubles each retry (important for 429 rate-limit backoff)


def _token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Set HF_TOKEN (export HF_TOKEN=hf_...) -- the token is no longer hardcoded "
            "here (a previous version leaked one to git history; rotate it if still live).")
    return token


def wipe_repo_contents(repo_id=REPO_ID, repo_type=REPO_TYPE, yes=False):
    """Delete every file in the repo except KEEP_ON_WIPE, in WIPE_CHUNK-sized commits --
    one delete op per commit would reintroduce the commit-rate problem."""
    api = HfApi(token=_token())
    files = [f for f in api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
             if f not in KEEP_ON_WIPE]
    if not files:
        print("[wipe] repo already empty")
        return
    top = sorted({f.split("/")[0] for f in files})
    print(f"[wipe] {len(files)} files in {repo_id} (top-level: {top})")
    if not yes:
        print("[wipe] dry run -- pass --yes to delete")
        return
    for i in range(0, len(files), WIPE_CHUNK):
        chunk = files[i:i + WIPE_CHUNK]
        api.create_commit(repo_id=repo_id, repo_type=repo_type,
                          operations=[CommitOperationDelete(path_in_repo=f) for f in chunk],
                          commit_message=f"Remove {len(chunk)} existing files")
        print(f"[wipe] deleted {i + len(chunk)}/{len(files)}")


def _case_dirs(local_base_path):
    # a case is a dir with wall_data.pt -- skips _meta/, _sanity/, _logs/, processed/, .vscode/
    return sorted(p for p in Path(local_base_path).iterdir()
                 if p.is_dir() and (p / "wall_data.pt").is_file())


def _remote_sizes(api, repo_id, repo_type, path_in_repo_prefix):
    return {
        item.path: item.size
        for item in api.list_repo_tree(repo_id=repo_id, path_in_repo=path_in_repo_prefix,
                                       recursive=True, repo_type=repo_type)
        if hasattr(item, "size")
    }


def _case_fully_uploaded(case_dir, path_in_repo_prefix, remote_sizes):
    for local_file in case_dir.rglob("*"):
        if not local_file.is_file():
            continue
        path_in_repo = f"{path_in_repo_prefix}/{case_dir.name}/{local_file.relative_to(case_dir)}"
        if remote_sizes.get(path_in_repo) != local_file.stat().st_size:
            return False
    return True


def upload_missing_case_dirs(local_base_path=LOCAL_BASE_PATH, repo_id=REPO_ID,
                             repo_type=REPO_TYPE, path_in_repo_prefix=PATH_IN_REPO_PREFIX):
    api = HfApi(token=_token())
    try:
        remote_sizes = _remote_sizes(api, repo_id, repo_type, path_in_repo_prefix)
    except Exception as e:
        print(f"[upload] could not list existing files ({path_in_repo_prefix}/ may not exist yet): {e}")
        remote_sizes = {}

    case_dirs = _case_dirs(local_base_path)
    print(f"[upload] {len(case_dirs)} case directories found under {local_base_path}")

    for case_dir in case_dirs:
        if _case_fully_uploaded(case_dir, path_in_repo_prefix, remote_sizes):
            print(f"[skip] {case_dir.name} (already up to date)")
            continue

        path_in_repo = f"{path_in_repo_prefix}/{case_dir.name}"
        print(f"[upload] {case_dir} -> {path_in_repo}")
        delay = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                api.upload_folder(repo_id=repo_id, repo_type=repo_type,
                                  folder_path=str(case_dir), path_in_repo=path_in_repo,
                                  commit_message=f"Add {case_dir.name}")
                break
            except Exception as e:
                rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
                wait = max(delay, 60) if rate_limited else delay
                print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    print(f"  retrying in {wait}s...")
                    time.sleep(wait)
                    delay *= 2
                else:
                    raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wipe", action="store_true", help="delete all existing files in the repo first")
    p.add_argument("--yes", action="store_true", help="confirm --wipe; without it --wipe is a dry run")
    p.add_argument("--upload", action="store_true", help="upload local_base_path -> cfd_data/")
    args = p.parse_args()
    if not (args.wipe or args.upload):
        p.error("pass --wipe and/or --upload")
    if args.wipe and args.upload and not args.yes:
        p.error("--wipe --upload needs --yes (otherwise the wipe is a dry run and old files stay)")
    if args.wipe:
        wipe_repo_contents(yes=args.yes)
    if args.upload:
        upload_missing_case_dirs()


"""
conda activate new
export HF_TOKEN=hf_...
python angioflow/huggingface/upload_angioflow.py --wipe              # dry run
python angioflow/huggingface/upload_angioflow.py --wipe --yes --upload
"""
