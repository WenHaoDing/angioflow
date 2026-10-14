"""Build the training graph dataset for one CFD dataset on a Hugging Face Job and
upload it to the public dataset whding123/AngioFlow-PyG (created on the first
upload if missing) as a single zip:

    wall/<tag>.zip      tag = aneug_flow | angioflow | aneugv2

Unzipped, every zip lands in its own per-dataset folder -- the per_version_dirs
layout angioflow.data.datasets.AngioFlowWallDataset reads -- so all three can be
unzipped into the same root:
    <tag>/static_<tag>_<case>.pt        static graph (all wall_graph_static.pt fields
                                        but meta: pos, edge_index, faces, node_type,
                                        flow_split, aneurysm_type, global_geo_dist,
                                        geo_dist, normals, cot_eigval, cot_eigvec
                                        [N, 64], orphan_mask)
    <tag>/wss_<tag>_<case>_<t>.pt       [N, 3] float32 wall shear stress, t < n_frames
    <tag>/manifest.json                 config + one record per case

Unlike datasets.py (a local class for processing + training), this script computes
nothing geometric: every geometry field was precomputed locally by
angioflow.data.precompute_wall_static and uploaded next to each case's registration
(huggingface/upload_wall_graph_static.py). Here it is only paired with the case's
wall_data.pt and the WSS frames are sliced out, so the Job needs no pytorch3d/pyvista
-- only torch, numpy, huggingface_hub, and angioflow.data.dataset_tags (dependency-
free). Node order of wall_graph_static.pt == registration pos == wall_data.pt (checked
exactly on all three datasets), so WSS is used as-is.

Input is the source dataset MOUNTED read-only into the Job, not downloaded:
    hf jobs run ... -v hf://datasets/whding123/AneuGv2:/data ...  -> --data_root /data
Each case is checked before use: its wall_graph_static.pt must have been computed from
the registration that is next to it (meta.source_sha256), and wall_data.pt must have
>= n_frames frames with the same node count.

Resumable in two stages, so a timeout never loses more than one part:
  1. extract (default): cases are processed in parts of --part_size; each part is
     uploaded as wall/_parts/<tag>/part_NNNN.zip plus a small
     part_NNNN.json listing its cases. A rerun reads those JSONs and only processes
     cases not in any part yet.
  2. --assemble: streams every part into one <tag>.zip (one part on disk at a time),
     adds manifest.json, uploads it. --cleanup_parts then deletes _parts/<tag>/.
Disk: the final zip (~20-31 GB per dataset) plus one part -- fits the Job's 50 GB.

Without --execute nothing is uploaded: extract builds only the first part locally in
--scratch (a smoke test), assemble only prints the plan.

    # on a Job (repo cloned to /work/angioflow, run from /work):
    python -m angioflow.huggingface.extract_pyg_stable --dataset aneugv2 --data_root /data --execute --yes
    python -m angioflow.huggingface.extract_pyg_stable --dataset aneugv2 --assemble --execute --yes
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
import zipfile

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from angioflow.data.dataset_tags import DATASET_TAGS

DATASET_PREFIX = {"aneug_flow": "transient_data", "angioflow": "cfd_data", "aneugv2": "cfd_data"}
OUT_REPO = "whding123/AngioFlow-PyG"
OUT_PREFIX = "wall"
REPO_TYPE = "dataset"
STATIC_NAME = "wall_graph_static.pt"
REG_NAME = "wall_mesh_registration.pt"
WALL_NAME = "wall_data.pt"
PHYSICS_KEYS = ["x_wall_shear", "y_wall_shear", "z_wall_shear"]
N_FRAMES = 80
PART_SIZE = 25           # ~50 MB/case -> ~1.2 GB per part
MAX_RETRIES = 5
RETRY_DELAY = 10         # seconds; doubles each retry


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def ensure_repo(api, repo_id):
    """Create the output dataset (public) if it doesn't exist yet."""
    api.create_repo(repo_id, repo_type=REPO_TYPE, private=False, exist_ok=True)


def with_retry(fn, what):
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as e:
            rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
            print(f"[retry] {what}: attempt {attempt}/{MAX_RETRIES} failed: {e}", flush=True)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(max(delay, 60) if rate_limited else delay)
            delay *= 2


def wss_frames(wall_data, n_frames):
    """[T, N, 3] float32. AneuG-Flow wall_data.pt is a flat dict; AngioFlow/AneuGv2
    nest the same keys under the "wall" zone."""
    zone = wall_data["wall"] if "wall" in wall_data else wall_data
    T = zone[PHYSICS_KEYS[0]].shape[0]
    if T < n_frames:
        raise ValueError(f"wall_data.pt has {T} frames < n_frames={n_frames}")
    return np.stack([zone[k][:n_frames, :, 0].numpy() for k in PHYSICS_KEYS], axis=-1).astype(np.float32)


def discover_cases(case_root):
    cases, missing = [], {}
    for cid in sorted(os.listdir(case_root)):
        d = os.path.join(case_root, cid)
        if cid.startswith("_") or not os.path.isdir(d):
            continue
        absent = [f for f in (STATIC_NAME, REG_NAME, WALL_NAME) if not os.path.isfile(os.path.join(d, f))]
        if absent:
            missing[cid] = absent
        else:
            cases.append(cid)
    return cases, missing


def _torch_bytes(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def add_case(zf, tag, cid, case_dir, n_frames):
    """Write one case's static + WSS frames into an open zip; return its manifest record."""
    static = torch.load(os.path.join(case_dir, STATIC_NAME), weights_only=False)
    meta = static.pop("meta")
    reg_sha = sha256_file(os.path.join(case_dir, REG_NAME))
    if meta["source_sha256"] != reg_sha:
        raise ValueError(f"{STATIC_NAME} was computed from a different {REG_NAME} -- "
                         f"rerun precompute_wall_static + upload_wall_graph_static for this case")
    wss = wss_frames(torch.load(os.path.join(case_dir, WALL_NAME), weights_only=False), n_frames)
    N = static["pos"].shape[0]
    if wss.shape[1] != N:
        raise ValueError(f"wall_data.pt has {wss.shape[1]} nodes, {STATIC_NAME} has {N}")

    zf.writestr(f"{tag}/static_{tag}_{cid}.pt", _torch_bytes(static))
    for t in range(n_frames):
        zf.writestr(f"{tag}/wss_{tag}_{cid}_{t}.pt", _torch_bytes(torch.from_numpy(wss[t].copy())))
    return dict(n_nodes=int(N), n_frames=n_frames, n_orphans=int(static["orphan_mask"].sum()),
                lap_k=int(meta["lap_k"]), laplacian=meta["laplacian"],
                registration_sha256=reg_sha,
                wss_max_t0=float(np.linalg.norm(wss[0], axis=1).max()))


def parts_on_hub(api, tag, out_repo, out_prefix):
    """{part_index: [case ids]} from the part JSONs already uploaded."""
    folder = f"{out_prefix}/_parts/{tag}"
    try:
        files = [f.path for f in api.list_repo_tree(out_repo, path_in_repo=folder, repo_type=REPO_TYPE)]
    except Exception:
        return {}
    parts = {}
    for path in files:
        if path.endswith(".json"):
            with open(hf_hub_download(out_repo, path, repo_type=REPO_TYPE, token=api.token)) as f:
                parts[int(os.path.basename(path)[len("part_"):-len(".json")])] = list(json.load(f)["cases"])
    return parts


def extract(args, api):
    tag = args.dataset
    case_root = os.path.join(args.data_root, DATASET_PREFIX[tag])
    cases, missing = discover_cases(case_root)
    print(f"[scan] {tag}: {len(cases)} complete case(s) under {case_root}")
    if missing:
        print(f"[warn] {len(missing)} case(s) missing files, skipped: {dict(list(missing.items())[:5])}")
    if args.max_cases:
        cases = cases[:args.max_cases]

    parts = parts_on_hub(api, tag, args.out_repo, args.out_prefix)
    done = {c for cs in parts.values() for c in cs}
    todo = [c for c in cases if c not in done]
    next_idx = max(parts, default=-1) + 1
    batches = [todo[i:i + args.part_size] for i in range(0, len(todo), args.part_size)]
    print(f"[plan] {len(done)} case(s) already in {len(parts)} part(s) on the Hub; "
          f"{len(todo)} to process in {len(batches)} part(s) of up to {args.part_size}")
    if not args.execute or not args.yes:
        batches = batches[:1]
        print("[local only] no --execute --yes: building the first part in --scratch, no upload")

    os.makedirs(args.scratch, exist_ok=True)
    if args.execute and args.yes:
        ensure_repo(api, args.out_repo)
    failed = {}
    for bi, batch in enumerate(batches):
        idx = next_idx + bi
        zip_path = os.path.join(args.scratch, f"part_{idx:04d}.zip")
        records = {}
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for cid in batch:
                t0 = time.time()
                try:
                    records[cid] = add_case(zf, tag, cid, os.path.join(case_root, cid), args.n_frames)
                    print(f"[ok] {tag}/{cid} N={records[cid]['n_nodes']} ({time.time() - t0:.1f}s)", flush=True)
                except Exception as e:
                    failed[cid] = repr(e)
                    print(f"[fail] {tag}/{cid}: {e}", flush=True)
        part_meta = dict(dataset=tag, part=idx, cases=sorted(records), records=records,
                         n_frames=args.n_frames, git_commit=git_commit())
        json_path = zip_path[:-len(".zip")] + ".json"
        with open(json_path, "w") as f:
            json.dump(part_meta, f, indent=1)
        print(f"[part {idx:04d}] {len(records)} case(s), {os.path.getsize(zip_path) / 1e9:.2f} GB -> {zip_path}")
        if not args.execute or not args.yes:
            break
        if records:
            folder = f"{args.out_prefix}/_parts/{tag}"
            ops = [CommitOperationAdd(path_in_repo=f"{folder}/part_{idx:04d}.zip", path_or_fileobj=zip_path),
                   CommitOperationAdd(path_in_repo=f"{folder}/part_{idx:04d}.json", path_or_fileobj=json_path)]
            with_retry(lambda: api.create_commit(
                repo_id=args.out_repo, repo_type=REPO_TYPE, operations=ops,
                commit_message=f"{tag}: PyG stable part {idx:04d} ({len(records)} case(s))"),
                f"upload part {idx:04d}")
            print(f"[part {idx:04d}] uploaded", flush=True)
        os.remove(zip_path)
        os.remove(json_path)
    if failed:
        print(f"\n[failed] {len(failed)} case(s): {failed}")


def assemble(args, api):
    tag = args.dataset
    folder = f"{args.out_prefix}/_parts/{tag}"
    parts = parts_on_hub(api, tag, args.out_repo, args.out_prefix)
    all_cases = [c for i in sorted(parts) for c in parts[i]]
    dupes = sorted({c for c in all_cases if all_cases.count(c) > 1})
    print(f"[plan] {len(parts)} part(s), {len(all_cases)} case(s) -> {args.out_prefix}/{tag}.zip")
    if dupes:
        raise SystemExit(f"[abort] case(s) in more than one part: {dupes[:10]}")
    if not args.execute or not args.yes:
        print("[dry run only] pass --execute --yes to assemble + upload.")
        return

    os.makedirs(args.scratch, exist_ok=True)
    final_path = os.path.join(args.scratch, f"{tag}.zip")
    manifest = dict(dataset=tag, n_cases=len(all_cases), n_frames=None, records={},
                    created=time.strftime("%Y-%m-%d %H:%M:%S"), parts=sorted(parts),
                    source_repo={"aneug_flow": "whding123/AneuG-Flow", "angioflow": "whding123/AngioFlow",
                                 "aneugv2": "whding123/AneuGv2"}[tag])
    with zipfile.ZipFile(final_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:
        for idx in sorted(parts):
            part_dir = os.path.join(args.scratch, "_dl")
            zip_local = hf_hub_download(args.out_repo, f"{folder}/part_{idx:04d}.zip", repo_type=REPO_TYPE,
                                        token=api.token, local_dir=part_dir)
            json_local = hf_hub_download(args.out_repo, f"{folder}/part_{idx:04d}.json", repo_type=REPO_TYPE,
                                         token=api.token, local_dir=part_dir)
            with open(json_local) as f:
                pm = json.load(f)
            manifest["records"].update(pm["records"])
            manifest["n_frames"] = pm["n_frames"]
            with zipfile.ZipFile(zip_local) as zin:
                for info in zin.infolist():
                    with zin.open(info) as src, zout.open(info.filename, "w", force_zip64=True) as dst:
                        shutil.copyfileobj(src, dst, 16 << 20)
            shutil.rmtree(part_dir)          # keep only one part on disk at a time
            print(f"[assemble] part {idx:04d} ({len(pm['cases'])} case(s))", flush=True)
        manifest["git_commit"] = git_commit()
        zout.writestr(f"{tag}/manifest.json", json.dumps(manifest, indent=1))
    print(f"[assemble] {final_path}: {os.path.getsize(final_path) / 1e9:.2f} GB")

    with_retry(lambda: api.upload_file(path_or_fileobj=final_path, path_in_repo=f"{args.out_prefix}/{tag}.zip",
                                       repo_id=args.out_repo, repo_type=REPO_TYPE,
                                       commit_message=f"{tag}: PyG stable dataset ({len(all_cases)} cases)"),
               "upload final zip")
    print(f"[done] uploaded {args.out_prefix}/{tag}.zip")
    os.remove(final_path)
    if args.cleanup_parts:
        with_retry(lambda: api.delete_folder(folder, repo_id=args.out_repo, repo_type=REPO_TYPE,
                                             commit_message=f"{tag}: remove PyG stable parts (assembled)"),
                   "delete parts")
        print(f"[cleanup] deleted {folder}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_TAGS))
    p.add_argument("--data_root", default="/data", help="mount point of the source dataset repo")
    p.add_argument("--out_repo", default=OUT_REPO)
    p.add_argument("--out_prefix", default=OUT_PREFIX)
    p.add_argument("--scratch", default="/tmp/pyg_stable")
    p.add_argument("--n_frames", type=int, default=N_FRAMES)
    p.add_argument("--part_size", type=int, default=PART_SIZE)
    p.add_argument("--max_cases", type=int, default=None, help="smoke test: only the first N cases")
    p.add_argument("--assemble", action="store_true", help="stage 2: merge uploaded parts into <tag>.zip")
    p.add_argument("--cleanup_parts", action="store_true", help="with --assemble: delete _parts/<tag>/ after upload")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true", help="actually upload (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute")
    return p.parse_args()


def main():
    args = parse_args()
    api = HfApi(token=args.token)
    (assemble if args.assemble else extract)(args, api)


if __name__ == "__main__":
    main()
