"""Download only `raw_data_fixed/` from the whding123/AngioFlow Hugging Face dataset
(not the whole repo, which also has a `raw_data/` top-level folder -- 9345 files total
vs 2085 under raw_data_fixed/'s 107 cases).

Case folders land directly under --dest (the `raw_data_fixed/` prefix itself is not
kept locally), matching every other dataset root in this project (angioflowv2_merged/
<case>, AneuX_stable/<case>, ...) rather than adding a redundant extra directory level.

Uses `snapshot_download(..., allow_patterns=...)`, which only transfers files under the
given prefix and resumes a partial download on rerun (unlike huggingface/download_data.py's
hand-rolled wget+progress-log approach) -- no retry/resume logic needed here.

Auth: no token is hardcoded (a hardcoded token in this repo's history was leaked --
see huggingface/upload_angioflow.py). Defaults to huggingface_hub's normal resolution
(the machine's cached `huggingface-cli login` token); pass --token or set HF_TOKEN to
override.

    python -m angioflow.data.download_imperialnhs_fix --dry_run
    python -m angioflow.data.download_imperialnhs_fix
"""
from __future__ import annotations

import argparse
import os
import shutil

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "whding123/AngioFlow"
REPO_TYPE = "dataset"
REMOTE_PREFIX = "raw_data_fixed"
DEST_DEFAULT = "/media/yaplab2/wd8tb/wenhao/datasets/ImperialNHS_fix"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--prefix", default=REMOTE_PREFIX)
    p.add_argument("--dest", default=DEST_DEFAULT)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="defaults to $HF_TOKEN, else huggingface_hub's cached login")
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--dry_run", action="store_true", help="list what would be downloaded and stop")
    return p.parse_args()


def flatten(staging_root, prefix, dest):
    """Move <staging_root>/<prefix>/<case>/... up to <dest>/<case>/..., merging into
    any case folder a previous partial run already placed directly under dest."""
    prefixed = os.path.join(staging_root, prefix)
    if not os.path.isdir(prefixed):
        return 0
    n = 0
    for case in sorted(os.listdir(prefixed)):
        src = os.path.join(prefixed, case)
        dst = os.path.join(dest, case)
        if not os.path.isdir(src):
            continue
        if os.path.isdir(dst):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src)
        else:
            shutil.move(src, dst)
        n += 1
    # staging_root IS dest (see main()), so only the now-empty prefix dir needs removing
    try:
        os.rmdir(prefixed)
    except OSError:
        pass   # not empty -- some non-case file sat directly under the prefix; leave it
    return n


def main():
    args = parse_args()
    api = HfApi(token=args.token)

    files = [f for f in api.list_repo_files(repo_id=args.repo_id, repo_type=REPO_TYPE)
            if f.startswith(f"{args.prefix}/")]
    cases = sorted({f.split("/", 2)[1] for f in files if "/" in f[len(args.prefix) + 1:]})
    print(f"[scan] {len(files)} file(s) across {len(cases)} case(s) under "
          f"{args.repo_id}/{args.prefix}")
    if args.dry_run:
        print(f"would download to: {args.dest}")
        for c in cases[:20]:
            print(f"   {c}")
        if len(cases) > 20:
            print(f"   ... and {len(cases) - 20} more")
        return

    os.makedirs(args.dest, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id, repo_type=REPO_TYPE, local_dir=args.dest,
        allow_patterns=[f"{args.prefix}/*"], token=args.token,
        max_workers=args.max_workers,
    )
    n = flatten(args.dest, args.prefix, args.dest)
    # snapshot_download's own bookkeeping (blobs/locks), not part of the dataset
    cache_dir = os.path.join(args.dest, ".cache")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
    print(f"[done] {n} case folder(s) in {args.dest}")


if __name__ == "__main__":
    main()
