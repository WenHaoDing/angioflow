"""Build the interior (blood flow) training data for one CFD dataset on a Hugging Face
Job and upload it to the public dataset whding123/AngioFlow-PyG (created on the
first upload if missing), next to the wall graphs that extract_pyg_stable.py builds:

    interior/<tag>/part_jJJ_NNNN.zip   ~50 cases (~20 GB) each
    interior/<tag>/part_jJJ_NNNN.json  cases + per-case records
    interior/<tag>/manifest.json       written by --finalize

--lean stores only vel + press per frame (no gradients; ~0.11 GB/case instead of
~0.44) under interior_lean/<tag>/ instead (150-case parts). Training that doesn't
supervise gradients (load_grad=False, e.g. train_joint.py) needs only the lean data.
"lean" is part of each part's settings, so the two variants can never be mixed.

Every part unzips into the same flat folder, which is the default interior_dir of
angioflow.data.datasets_interior (<save_dir>/interior/) -- so unzipping the wall zips
and all interior parts into one save_dir gives a root the existing loader reads:
    interior/interior_static_<tag>_<case>.pt   {"pos": [P, 3], "wall_dist": [P]}
    interior/interior_<tag>_<case>_<t>.pt      {"vel": [P, 3], "press": [P, 1],
                                                "grad_v": [P, 9], "grad_p": [P, 3]}
                                               (--lean: vel + press only)
No pool graph (edge_index): train_joint.py doesn't load it, and AneuG-Flow's
validation needs shape_remeshed_cfd.obj, which isn't on the Hub. Use the loader
with load_edge_index=False.

The interior sampling is a COPY of datasets_interior.py's (_adaptive_fps,
_InteriorMixin._sample_pool / _interior_pos / the interior half of _process_case),
not an import: that module pulls in pytorch3d/pyvista through datasets.py. Keep the
two in sync by hand. Defaults match how angioflow_pyg_interior was built
(pool_ratio=0.16, near_decay=0.1, far_floor=0.25, min_wall_dist=1e-4, seed 0), and
the pool is deterministic per case, so a rerun reproduces it exactly.
Interior positions: AngioFlow/AneuGv2 blood_data.pt store a static "pos";
AneuG-Flow stores per-frame x/y/z_coordinate (identical across frames).
Centring uses the registration pos mean -- the same offset as the wall static, so
wall and interior share one frame.

Input is the source dataset MOUNTED read-only (-v hf://datasets/<repo>:/data):
<data_root>/<prefix>/<case>/{blood_data.pt, wall_mesh_registration.pt}. Each case
reads a 3-3.6 GB blood_data.pt, so a dataset can be split over parallel Jobs:
--num_jobs N --job_index i takes the cases whose crc32(case id) % N == i, and names
its parts part_j<i>_*, so concurrent Jobs never collide. A rerun (any job split)
skips every case already listed in an uploaded part JSON.

Without --execute nothing is uploaded: the first part is built locally in --scratch.

    python -m angioflow.huggingface.extract_interior_pyg_stable --dataset aneugv2 --max_cases 2   # smoke test
    python -m angioflow.huggingface.extract_interior_pyg_stable --dataset aneugv2 --lean --execute --yes
    python -m angioflow.huggingface.extract_interior_pyg_stable --dataset aneugv2 --num_jobs 4 --job_index 0 --execute --yes
    python -m angioflow.huggingface.extract_interior_pyg_stable --dataset aneugv2 --finalize --execute --yes
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
import zlib

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError, RepositoryNotFoundError
from scipy.spatial import cKDTree

from angioflow.data.dataset_tags import DATASET_TAGS

DATASET_PREFIX = {"aneug_flow": "transient_data", "angioflow": "cfd_data", "aneugv2": "cfd_data"}
SOURCE_REPO = {"aneug_flow": "whding123/AneuG-Flow", "angioflow": "whding123/AngioFlow",
               "aneugv2": "whding123/AneuGv2"}                # for --blood_read download
OUT_REPO = "whding123/AngioFlow-PyG"
OUT_PREFIX = "interior"
OUT_PREFIX_LEAN = "interior_lean"
REPO_TYPE = "dataset"
BLOOD_NAME = "blood_data.pt"
REG_NAME = "wall_mesh_registration.pt"
ZIP_DIR = "interior"            # every entry lives under interior/ inside each part zip
N_FRAMES = 80
PART_SIZE = 50                  # ~440 MB/case -> ~22 GB per part
PART_SIZE_LEAN = 150            # ~110 MB/case (--lean) -> ~16 GB per part
MAX_RETRIES = 5
RETRY_DELAY = 10                # seconds; doubles each retry

# same keys as datasets_interior.py
VEL_KEYS = ["x_velocity", "y_velocity", "z_velocity"]
PRESS_KEY = "absolute_pressure"
VGRAD_KEYS = [f"d{c}_velocity_d{x}" for c in "xyz" for x in "xyz"]   # row-major dvi/dxj
PGRAD_KEYS = ["dp_dx", "dp_dy", "dp_dz"]
# part settings that must agree across every part of one dataset
CONFIG_KEYS = ("lean", "n_frames", "pool_ratio", "near_decay", "far_floor", "min_wall_dist", "seed")


# ----------------------------------------------------------------------------
# copied from datasets_interior.py -- keep in sync
# ----------------------------------------------------------------------------
def _adaptive_fps(pts, r_local, P, rng, r_factor=3.0, cap_factor=8.0):
    """Greedy FPS with priority dmin(i)/r_local(i): points stay far apart relative
    to a local target spacing (small near the wall => denser there, evenly spread
    everywhere). Lazy heap + capped local ball-updates keep it near O(N log N)."""
    N = len(pts)
    tree = cKDTree(pts)
    cand_nn = tree.query(pts, k=2)[0][:, 1]
    s0 = np.median(cand_nn) * (N / P) ** (1.0 / 3.0)
    cap = cap_factor * s0

    dmin = np.full(N, np.inf)
    chosen = np.zeros(N, dtype=bool)
    selected = np.empty(P, dtype=np.int64)

    first = int(rng.integers(N))
    selected[0] = first
    chosen[first] = True
    dmin = np.minimum(dmin, np.linalg.norm(pts - pts[first], axis=1))
    dmin[first] = -1.0

    heap = [(-(dmin[i] / r_local[i]), i, dmin[i]) for i in range(N) if not chosen[i]]
    heapq.heapify(heap)

    n = 1
    while n < P and heap:
        _, i, d_at_push = heapq.heappop(heap)
        if chosen[i] or d_at_push != dmin[i]:
            continue                                     # stale entry
        selected[n] = i
        chosen[i] = True
        n += 1
        R = min(r_factor * dmin[i], cap)
        dmin[i] = -1.0
        idx = np.asarray(tree.query_ball_point(pts[i], R), dtype=np.int64)
        if len(idx):
            nd = np.linalg.norm(pts[idx] - pts[i], axis=1)
            better = nd < dmin[idx]
            upd = idx[better]
            dmin[upd] = nd[better]
            for j, dj in zip(upd.tolist(), dmin[upd].tolist()):
                if not chosen[j]:
                    heapq.heappush(heap, (-(dj / r_local[j]), j, dj))
    return selected[:n]


def sample_pool(int_pos, wall_dist, rng, pool_ratio, near_decay, far_floor, min_wall_dist):
    """_InteriorMixin._sample_pool with its attributes as arguments."""
    valid = np.where(wall_dist >= min_wall_dist)[0]
    P = max(1, min(int(len(wall_dist) * pool_ratio), len(valid)))
    d = wall_dist[valid] / wall_dist.max()
    hist, edges = np.histogram(d, bins=32)
    inv_density = 1.0 / hist.clip(1)[np.clip(np.digitize(d, edges) - 1, 0, 31)]
    density = far_floor + (1.0 - far_floor) * np.exp(-d / near_decay)
    w = inv_density * density

    M = min(len(valid), int(4.0 * P))
    cand = rng.choice(len(valid), size=M, replace=False, p=w / w.sum())
    r_local = density[cand] ** (-1.0 / 3.0)
    sel = _adaptive_fps(int_pos[valid[cand]], r_local, P, rng)
    return valid[cand[sel]]


def interior_pos(blood):
    """Raw interior positions [N, 3] float64. Static "pos" (AngioFlow, AneuGv2) or
    per-frame coordinates identical across frames (AneuG-Flow)."""
    if "pos" in blood:
        return blood["pos"].numpy().astype(np.float64)
    return np.stack([blood[f"{a}_coordinate"][0, :, 0].numpy() for a in "xyz"],
                    axis=1).astype(np.float64)
# ----------------------------------------------------------------------------


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


def _torch_bytes(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def load_blood(path):
    # mmap pages in only the keys used below (skips viscosity, dp_dt, dynamic_pressure)
    try:
        return torch.load(path, weights_only=False, mmap=True)
    except Exception:
        return torch.load(path, weights_only=False)


def fetch_blood(tag, cid, case_dir, args):
    """-> (path to torch.load, path to delete afterwards or None).
    mount:    memory-map straight through the mount. Gathering ~80k scattered pool
              points then turns into many small remote page reads -- slow on a network
              mount (~5-6 min/case observed on HF vs ~40 s locally).
    copy:     one sequential copy of the file to scratch first, then mmap it locally.
    download: hf_hub_download from the source repo to scratch (parallel chunk fetch)."""
    src = os.path.join(case_dir, BLOOD_NAME)
    if args.blood_read == "mount":
        return src, None
    tmp_dir = os.path.join(args.scratch, "_blood")
    os.makedirs(tmp_dir, exist_ok=True)
    if args.blood_read == "copy":
        dst = os.path.join(tmp_dir, f"{cid}_{BLOOD_NAME}")
        with open(src, "rb") as f, open(dst, "wb") as g:
            shutil.copyfileobj(f, g, 64 << 20)
        return dst, dst
    path = with_retry(lambda: hf_hub_download(SOURCE_REPO[tag], f"{DATASET_PREFIX[tag]}/{cid}/{BLOOD_NAME}",
                                              repo_type=REPO_TYPE, token=args.token, local_dir=tmp_dir),
                      f"download {cid}/{BLOOD_NAME}")
    return path, path


def build_case(tag, cid, case_dir, args):
    """One case's zip entries [(name, bytes)] + manifest record. Everything is built in
    memory first (~0.1-0.4 GB), so a case that fails part-way (e.g. a missing gradient
    key) leaves nothing in the part. record["timing_s"] has seconds per stage."""
    timing, t = {}, time.perf_counter()

    def lap(name):
        nonlocal t
        now = time.perf_counter()
        timing[name] = round(now - t, 1)
        t = now

    reg_path = os.path.join(case_dir, REG_NAME)
    wall_pos = torch.load(reg_path, weights_only=False)["pos"].numpy().astype(np.float64)
    blood_path, cleanup = fetch_blood(tag, cid, case_dir, args)
    lap("fetch")
    try:
        blood = load_blood(blood_path)
        T = blood[VEL_KEYS[0]].shape[0]
        if T < args.n_frames:
            raise ValueError(f"{BLOOD_NAME} has {T} frames < n_frames={args.n_frames}")
        int_pos = interior_pos(blood)
        lap("load")

        wall_dist = cKDTree(wall_pos).query(int_pos)[0]
        rng = np.random.default_rng(args.seed)
        pool = sample_pool(int_pos, wall_dist, rng, args.pool_ratio, args.near_decay,
                           args.far_floor, args.min_wall_dist)
        pool_t = torch.from_numpy(pool)
        center = wall_pos.mean(0, keepdims=True)
        lap("sample")

        n = args.n_frames
        vel = torch.cat([blood[k][:n, pool_t] for k in VEL_KEYS], dim=-1)       # [T, P, 3]
        press = blood[PRESS_KEY][:n, pool_t]                                    # [T, P, 1]
        if args.lean:                     # gradient keys are never read
            frames = [{"vel": vel[t_].clone(), "press": press[t_].clone()} for t_ in range(n)]
        else:
            grad_v = torch.cat([blood[k][:n, pool_t] for k in VGRAD_KEYS], dim=-1)  # [T, P, 9]
            grad_p = torch.cat([blood[k][:n, pool_t] for k in PGRAD_KEYS], dim=-1)  # [T, P, 3]
            frames = [{"vel": vel[t_].clone(), "press": press[t_].clone(),
                       "grad_v": grad_v[t_].clone(), "grad_p": grad_p[t_].clone()} for t_ in range(n)]
        del blood
        lap("slice")
    finally:
        if cleanup and os.path.exists(cleanup):
            os.remove(cleanup)

    entries = [(f"{ZIP_DIR}/interior_static_{tag}_{cid}.pt", _torch_bytes({
        "pos":       torch.from_numpy(int_pos[pool] - center).float(),   # [P, 3]
        "wall_dist": torch.from_numpy(wall_dist[pool]).float(),          # [P]
    }))]
    entries += [(f"{ZIP_DIR}/interior_{tag}_{cid}_{t_}.pt", _torch_bytes(f)) for t_, f in enumerate(frames)]
    lap("serialize")
    return entries, dict(n_pool=int(len(pool)), n_interior=int(len(int_pos)), n_frames=n,
                         registration_sha256=sha256_file(reg_path),
                         vel_max_t0=float(vel[0].norm(dim=1).max()), timing_s=timing)


def discover_cases(case_root):
    cases, missing = [], {}
    for cid in sorted(os.listdir(case_root)):
        d = os.path.join(case_root, cid)
        if cid.startswith("_") or not os.path.isdir(d):
            continue
        absent = [f for f in (BLOOD_NAME, REG_NAME) if not os.path.isfile(os.path.join(d, f))]
        if absent:
            missing[cid] = absent
        else:
            cases.append(cid)
    return cases, missing


PART_RE = re.compile(r"part_j(\d+)_(\d+)\.json$")


def parts_on_hub(api, folder, out_repo):
    """{part file stem: part JSON} for every part already uploaded. Only a confirmed-
    missing repo or folder means "no parts yet"; any other listing error is retried, then
    raised -- never read as empty, or a resume would renumber and overwrite existing parts."""
    def _list():
        try:
            return [f.path for f in api.list_repo_tree(out_repo, path_in_repo=folder, repo_type=REPO_TYPE)]
        except (RepositoryNotFoundError, RemoteEntryNotFoundError):
            return []
    parts = {}
    for path in with_retry(_list, f"list {folder}"):
        if PART_RE.search(path):
            local = with_retry(lambda: hf_hub_download(out_repo, path, repo_type=REPO_TYPE, token=api.token),
                               f"download {path}")
            with open(local) as f:
                parts[os.path.basename(path)[:-len(".json")]] = json.load(f)
    return parts


def check_config(parts, config, where):
    """Abort if any existing part was built with settings other than `config` -- a resume
    must not mix configurations in one dataset."""
    bad = {s: {k: p.get(k) for k in config} for s, p in parts.items()
           if any(p.get(k) != v for k, v in config.items())}
    if bad:
        raise SystemExit(f"[abort] {where}: part(s) built with settings other than {config}: "
                         f"{dict(list(bad.items())[:5])} -- rerun with the same settings or a new --out_prefix")


def assert_free(api, repo, paths):
    """Never overwrite: abort if any target path already exists on the Hub."""
    taken = [p for p in paths if with_retry(lambda: api.file_exists(repo, p, repo_type=REPO_TYPE), f"check {p}")]
    if taken:
        raise SystemExit(f"[abort] refusing to overwrite existing {taken}")


def extract(args, api):
    tag = args.dataset
    folder = f"{args.out_prefix}/{tag}"
    case_root = os.path.join(args.data_root, DATASET_PREFIX[tag])
    cases, missing = discover_cases(case_root)
    print(f"[scan] {tag}: {len(cases)} case(s) with {BLOOD_NAME} + {REG_NAME} under {case_root}")
    if missing:
        print(f"[warn] {len(missing)} case(s) missing files, skipped: {dict(list(missing.items())[:5])}")
    mine = [c for c in cases if zlib.crc32(c.encode()) % args.num_jobs == args.job_index]
    if args.max_cases:
        mine = mine[:args.max_cases]

    parts = parts_on_hub(api, folder, args.out_repo)
    check_config(parts, {k: getattr(args, k) for k in CONFIG_KEYS}, folder)
    done = {c for p in parts.values() for c in p["cases"]}
    todo = [c for c in mine if c not in done]
    own = [int(PART_RE.search(s + ".json")[2]) for s in parts
           if int(PART_RE.search(s + ".json")[1]) == args.job_index]
    next_idx = max(own, default=-1) + 1
    batches = [todo[i:i + args.part_size] for i in range(0, len(todo), args.part_size)]
    print(f"[plan] job {args.job_index}/{args.num_jobs}: {len(mine)} case(s) assigned, "
          f"{len(mine) - len(todo)} already uploaded, {len(todo)} to process in {len(batches)} "
          f"part(s) of up to {args.part_size}")
    if not args.execute or not args.yes:
        batches = batches[:1]
        print("[local only] no --execute --yes: building the first part in --scratch, no upload")

    os.makedirs(args.scratch, exist_ok=True)
    if args.execute and args.yes:
        ensure_repo(api, args.out_repo)
    failed = {}
    for bi, batch in enumerate(batches):
        stem = f"part_j{args.job_index:02d}_{next_idx + bi:04d}"
        zip_path = os.path.join(args.scratch, stem + ".zip")
        records = {}
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for cid in batch:
                t0 = time.time()
                try:
                    entries, record = build_case(tag, cid, os.path.join(case_root, cid), args)
                except Exception as e:
                    failed[cid] = repr(e)
                    print(f"[fail] {tag}/{cid}: {e}", flush=True)
                    continue
                for name, data in entries:      # write errors (e.g. disk full) stop the Job
                    zf.writestr(name, data)
                del entries
                records[cid] = record
                print(f"[ok] {tag}/{cid} pool {record['n_pool']}/{record['n_interior']} "
                      f"({time.time() - t0:.1f}s: {record['timing_s']})", flush=True)
        part_meta = dict(dataset=tag, part=stem, cases=sorted(records), records=records,
                         **{k: getattr(args, k) for k in CONFIG_KEYS}, git_commit=git_commit())
        json_path = os.path.join(args.scratch, stem + ".json")
        with open(json_path, "w") as f:
            json.dump(part_meta, f, indent=1)
        print(f"[{stem}] {len(records)} case(s), {os.path.getsize(zip_path) / 1e9:.2f} GB -> {zip_path}")
        if not args.execute or not args.yes:
            break
        if records:
            assert_free(api, args.out_repo, [f"{folder}/{stem}.zip", f"{folder}/{stem}.json"])
            ops = [CommitOperationAdd(path_in_repo=f"{folder}/{stem}.zip", path_or_fileobj=zip_path),
                   CommitOperationAdd(path_in_repo=f"{folder}/{stem}.json", path_or_fileobj=json_path)]
            with_retry(lambda: api.create_commit(
                repo_id=args.out_repo, repo_type=REPO_TYPE, operations=ops,
                commit_message=f"{tag}: interior {stem} ({len(records)} case(s))"), f"upload {stem}")
            print(f"[{stem}] uploaded", flush=True)
        os.remove(zip_path)
        os.remove(json_path)
    if failed:
        print(f"\n[failed] {len(failed)} case(s): {failed}")


def finalize(args, api):
    """Write manifest.json listing every part and case (no data is moved)."""
    tag = args.dataset
    folder = f"{args.out_prefix}/{tag}"
    parts = parts_on_hub(api, folder, args.out_repo)
    if not parts:
        raise SystemExit(f"[abort] no parts under {folder} -- nothing to finalize")
    config = {k: parts[min(parts)][k] for k in CONFIG_KEYS}
    check_config(parts, config, folder)
    all_cases = [c for p in parts.values() for c in p["cases"]]
    dupes = sorted({c for c in all_cases if all_cases.count(c) > 1})
    print(f"[plan] {len(parts)} part(s), {len(all_cases)} case(s) -> {folder}/manifest.json")
    if dupes:
        raise SystemExit(f"[abort] case(s) in more than one part: {dupes[:10]}")
    manifest = dict(dataset=tag, n_cases=len(all_cases), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                    parts={s: p["cases"] for s, p in sorted(parts.items())},
                    records={c: r for p in parts.values() for c, r in p["records"].items()},
                    config=config)
    if not args.execute or not args.yes:
        print("[dry run only] pass --execute --yes to upload manifest.json.")
        return
    with_retry(lambda: api.upload_file(path_or_fileobj=json.dumps(manifest, indent=1).encode(),
                                       path_in_repo=f"{folder}/manifest.json", repo_id=args.out_repo,
                                       repo_type=REPO_TYPE,
                                       commit_message=f"{tag}: interior manifest ({len(all_cases)} cases)"),
               "upload manifest")
    print(f"[done] uploaded {folder}/manifest.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_TAGS))
    p.add_argument("--data_root", default="/data", help="mount point of the source dataset repo")
    p.add_argument("--out_repo", default=OUT_REPO)
    p.add_argument("--out_prefix", default=None,
                   help=f"default: {OUT_PREFIX_LEAN} with --lean, else {OUT_PREFIX}")
    p.add_argument("--scratch", default="/tmp/interior_pyg_stable")
    p.add_argument("--n_frames", type=int, default=N_FRAMES)
    p.add_argument("--part_size", type=int, default=None,
                   help=f"cases per part; default {PART_SIZE_LEAN} with --lean, else {PART_SIZE}")
    p.add_argument("--lean", action="store_true", help="store only vel + press (no gradients)")
    p.add_argument("--blood_read", choices=["copy", "download", "mount"], default="copy",
                   help="how blood_data.pt is read: copy to scratch first (default), hf_hub_download "
                        "from the source repo, or mmap straight through the mount (slow)")
    p.add_argument("--num_jobs", type=int, default=1, help="split the dataset over this many parallel Jobs")
    p.add_argument("--job_index", type=int, default=0, help="which split this Job runs (0..num_jobs-1)")
    p.add_argument("--max_cases", type=int, default=None, help="smoke test: only the first N assigned cases")
    # interior pool sampling -- defaults = how angioflow_pyg_interior was built
    p.add_argument("--pool_ratio", type=float, default=0.16)
    p.add_argument("--near_decay", type=float, default=0.1)
    p.add_argument("--far_floor", type=float, default=0.25)
    p.add_argument("--min_wall_dist", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--finalize", action="store_true", help="write manifest.json from the uploaded parts")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true", help="actually upload (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute")
    args = p.parse_args()
    assert 0 <= args.job_index < args.num_jobs, "--job_index must be in [0, --num_jobs)"
    if args.out_prefix is None:
        args.out_prefix = OUT_PREFIX_LEAN if args.lean else OUT_PREFIX
    if args.part_size is None:
        args.part_size = PART_SIZE_LEAN if args.lean else PART_SIZE
    return args


def main():
    args = parse_args()
    api = HfApi(token=args.token)
    (finalize if args.finalize else extract)(args, api)


if __name__ == "__main__":
    main()
