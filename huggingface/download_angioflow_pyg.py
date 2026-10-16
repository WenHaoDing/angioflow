"""Download the extracted training graphs from whding123/AngioFlow-PyG and unzip them
into one local root -- the layout the dataset classes read:

    <dest>/<tag>/static_<tag>_<case>.pt, wss_<tag>_<case>_<t>.pt, manifest.json   (wall)
    <dest>/interior/interior_static_<tag>_<case>.pt, interior_<tag>_<case>_<t>.pt  (interior)

Hub sources (built by extract_pyg_stable.py / extract_interior_pyg_stable.py):
    wall           wall/<tag>.zip                    (assembled; one zip per dataset)
    interior_lean  interior_lean/<tag>/part_*.zip    (vel + press only)
    interior       interior/<tag>/part_*.zip         (vel + press + gradients)
    flow_split     flow_split/<tag>/flow_split.json   (per-dataset outlet flow-split table,
                                                     written by angioflow.data.generate_flow_split)
interior_lean and interior unzip to the same file names -- download one of them.
flow_split tables are small and always re-fetched, to <dest>/<tag>/flow_split.json -- where
the dataset class reads them with add_flow_split=True.

One archive at a time: download -> unzip -> verify every case in its manifest/part JSON
has its static file and all frames -> delete the zip. Resumable: each extracted archive
is recorded under <dest>/_download/done/ with the Hub file's sha256, so a rerun skips it
(and re-fetches it if the file on the Hub changed).

--frames 0[,8,...] unzips only those frames (plus every static file and the manifest) --
e.g. ~22 GB of the 79 GB of wall zips for frame 0 -- what training on chosen frames
needs (training/train_hf.py). The zips are still downloaded whole, one at a time.
A rerun asking for frames that an earlier run did not unzip fetches that archive again.

    python -m angioflow.huggingface.download_angioflow_pyg --dry_run
    python -m angioflow.huggingface.download_angioflow_pyg                         # wall, all datasets
    python -m angioflow.huggingface.download_angioflow_pyg --frames 0               # wall, frame 0 only
    python -m angioflow.huggingface.download_angioflow_pyg --what interior_lean --datasets aneugv2
    python -m angioflow.huggingface.download_angioflow_pyg --what flow_split
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "whding123/AngioFlow-PyG"
REPO_TYPE = "dataset"
DATASETS = ("aneug_flow", "angioflow", "aneugv2")
DEST_DEFAULT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG"


def remote_archives(api, repo, what, tag):
    """[(repo path, size, sha256, part JSON path or None)] to fetch for one dataset."""
    if what == "wall":
        info = api.get_paths_info(repo, [f"wall/{tag}.zip"], repo_type=REPO_TYPE)
        if not info:
            raise SystemExit(f"[abort] wall/{tag}.zip not on {repo} -- run extract_pyg_stable --assemble first")
        return [(info[0].path, info[0].size, info[0].lfs.sha256, None)]
    files = {f.path: f for f in api.list_repo_tree(repo, path_in_repo=f"{what}/{tag}", repo_type=REPO_TYPE)}
    zips = sorted(p for p in files if p.endswith(".zip"))
    if not zips:
        raise SystemExit(f"[abort] no parts under {what}/{tag} on {repo}")
    return [(p, files[p].size, files[p].lfs.sha256, p[:-len(".zip")] + ".json") for p in zips]


def expected_files(what, tag, cases_frames, frames=None):
    """Paths (relative to dest) every case must have after unzipping (frames: only those)."""
    out = []
    for cid, n in cases_frames.items():
        ts = range(n) if frames is None else frames
        if what == "wall":
            out.append(f"{tag}/static_{tag}_{cid}.pt")
            out += [f"{tag}/wss_{tag}_{cid}_{t}.pt" for t in ts]
        else:
            out.append(f"interior/interior_static_{tag}_{cid}.pt")
            out += [f"interior/interior_{tag}_{cid}_{t}.pt" for t in ts]
    return out


def frame_of(name):
    """Frame index of a per-frame member (wss_<tag>_<case>_<t>.pt / interior_<tag>_<case>_<t>.pt),
    None for static files, manifests and anything else."""
    base = os.path.basename(name)
    if not base.endswith(".pt") or not base.startswith(("wss_", "interior_")) or base.startswith("interior_static_"):
        return None
    return int(base[:-len(".pt")].rsplit("_", 1)[1])


def fetch_one(api, args, what, tag, path, sha, part_json):
    dl = os.path.join(args.dest, "_download")
    local = hf_hub_download(args.repo_id, path, repo_type=REPO_TYPE, token=args.token, local_dir=dl)
    with zipfile.ZipFile(local) as zf:
        if what == "wall":
            manifest = json.loads(zf.read(f"{tag}/manifest.json"))
            cases = {c: r["n_frames"] for c, r in manifest["records"].items()}
        else:
            with open(hf_hub_download(args.repo_id, part_json, repo_type=REPO_TYPE,
                                      token=args.token, local_dir=dl)) as f:
                part = json.load(f)
            cases = {c: part["records"][c]["n_frames"] for c in part["cases"]}
        members = None
        if args.frames is not None:
            members = [n for n in zf.namelist() if frame_of(n) is None or frame_of(n) in args.frames]
        zf.extractall(args.dest, members=members)
    missing = [p for p in expected_files(what, tag, cases, args.frames)
               if not os.path.exists(os.path.join(args.dest, p))]
    if missing:
        raise SystemExit(f"[abort] {path}: {len(missing)} expected file(s) missing after unzip, e.g. {missing[:3]}")
    if not args.keep_zips:
        os.remove(local)
    done_path = os.path.join(dl, "done", path.replace("/", "__") + ".json")
    os.makedirs(os.path.dirname(done_path), exist_ok=True)
    with open(done_path, "w") as f:
        json.dump(dict(path=path, sha256=sha, n_cases=len(cases), frames=args.frames), f)
    return len(cases)


def is_done(dest, path, sha, frames=None):
    """Extracted from this exact Hub file, with every frame asked for now (frames None = all)."""
    done_path = os.path.join(dest, "_download", "done", path.replace("/", "__") + ".json")
    if not os.path.exists(done_path):
        return False
    with open(done_path) as f:
        done = json.load(f)
    if done.get("sha256") != sha:
        return False
    had = done.get("frames")            # None: all frames were unzipped
    return had is None or (frames is not None and set(frames) <= set(had))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", nargs="+", default=["wall"],
                   choices=["wall", "interior_lean", "interior", "flow_split"])
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--dest", default=DEST_DEFAULT)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--keep_zips", action="store_true", help="keep each zip after unzipping")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--dry_run", action="store_true", help="show what would be downloaded and stop")
    p.add_argument("--frames", default=None,
                   help="comma-separated frame indices to unzip, e.g. 0 or 0,8,16 (default: all)")
    args = p.parse_args(argv)
    if "interior_lean" in args.what and "interior" in args.what:
        p.error("interior_lean and interior unzip to the same files -- pick one")
    if args.frames is not None:
        args.frames = sorted({int(t) for t in args.frames.split(",")})
    return args


def fetch_flow_split(args, tag):
    """Small per-dataset table -> <dest>/<tag>/flow_split.json (always re-fetched)."""
    local = hf_hub_download(args.repo_id, f"flow_split/{tag}/flow_split.json", repo_type=REPO_TYPE,
                            token=args.token, local_dir=os.path.join(args.dest, "_download"))
    out = os.path.join(args.dest, tag, "flow_split.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copyfile(local, out)
    with open(out) as f:
        n = len(json.load(f)["cases"])
    print(f"[ok] flow_split/{tag}/flow_split.json: {n} case(s) -> {out}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    api = HfApi(token=args.token)
    if "flow_split" in args.what:
        if not args.dry_run:
            for tag in args.datasets:
                fetch_flow_split(args, tag)
        else:
            print(f"[plan] flow_split tables for {args.datasets}")
        args.what = [w for w in args.what if w != "flow_split"]
        if not args.what:
            return
    plan = []
    for what in args.what:
        for tag in args.datasets:
            for path, size, sha, part_json in remote_archives(api, args.repo_id, what, tag):
                plan.append((what, tag, path, size, sha, part_json, is_done(args.dest, path, sha, args.frames)))
    todo = [x for x in plan if not x[6]]
    biggest = max((x[3] for x in todo), default=0)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.dest)) if not os.path.exists(args.dest)
                             else args.dest).free
    print(f"[plan] {len(plan)} archive(s), {len(plan) - len(todo)} already extracted, {len(todo)} to fetch "
          f"({sum(x[3] for x in todo) / 1e9:.1f} GB) -> {args.dest}")
    for what, tag, path, size, _, _, done in plan:
        print(f"  {'done' if done else 'todo'}  {path}  {size / 1e9:.2f} GB")
    need = sum(x[3] for x in todo) + biggest           # extracted data + one zip in flight
    #                                                    (an upper bound with --frames)
    print(f"[disk] need ~{need / 1e9:.0f} GB, free {free / 1e9:.0f} GB")
    if args.dry_run:
        print("\n[dry run only] rerun without --dry_run to download.")
        return
    if need > free:
        raise SystemExit("[abort] not enough free disk space")

    os.makedirs(args.dest, exist_ok=True)
    for what, tag, path, size, sha, part_json, _ in todo:
        n = fetch_one(api, args, what, tag, path, sha, part_json)
        print(f"[ok] {path}: {n} case(s) extracted and verified", flush=True)
    print(f"\n[done] {args.dest}")


if __name__ == "__main__":
    main()
