"""Remove specific cases from a Hugging Face dataset repo, in a single commit.

Generic by design: the caller supplies repo_id/prefix/cases, so the same function
works for any dataset (not just whding123/AneuG-Flow's transient_data/) -- a separate
main script will own the per-dataset specifics (which repo, which prefix, where the
case list comes from) and call remove_cases_from_dataset() with them.

Auth: no token hardcoded (a hardcoded token in this repo's git history was leaked --
see upload_angioflow.py). Pass token=None (the default) to use huggingface_hub's normal
resolution (the machine's cached `huggingface-cli login` token), or pass one explicitly.
"""
from __future__ import annotations

from huggingface_hub import CommitOperationDelete, HfApi


def remove_cases_from_dataset(repo_id, prefix, cases, repo_type="dataset", token=None,
                              execute=False, yes=False):
    """Delete every file under <prefix>/<case>/ for each case in `cases`, in ONE commit
    (one commit per file would risk the Hub's hourly commit-rate limit on a large case
    list -- same reasoning as upload_angioflow.py's wipe_repo_contents()).

    Dry run by default: prints the plan and returns without writing anything. Actually
    deleting needs BOTH execute=True and yes=True (two separate flags, not one, so a
    caller can't delete by only setting one) -- matches every other destructive script
    in this project.

    Returns {case: [repo-relative file paths]} for cases found under prefix/, so a
    caller can inspect or log what happened (or would happen, on a dry run).
    """
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    cases = set(cases)

    by_case = {}
    for f in files:
        if not f.startswith(f"{prefix}/"):
            continue
        case = f.split("/", 2)[1]
        if case in cases:
            by_case.setdefault(case, []).append(f)
    missing = sorted(cases - by_case.keys())
    n_files = sum(len(v) for v in by_case.values())

    print(f"[plan] {len(by_case)} case(s) found under {repo_id}/{prefix} ({n_files} file(s) total); "
          f"{len(missing)} already absent remotely")
    for c in sorted(by_case):
        print(f"   {c}  ({len(by_case[c])} files)")
    if missing:
        print(f"already absent (skipped): {missing}")

    if not execute:
        print("\n[dry run only] pass execute=True, yes=True to delete.")
        return by_case
    if not yes:
        print("\n[dry run only] execute=True given without yes=True -- not touching anything.")
        return by_case

    ops = [CommitOperationDelete(path_in_repo=f) for fs in by_case.values() for f in fs]
    api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=ops,
                      commit_message=f"Remove {len(by_case)} case(s) ({n_files} files) from {prefix}/")
    print(f"\n[done] deleted {n_files} file(s) across {len(by_case)} case(s) in a single commit")
    return by_case
