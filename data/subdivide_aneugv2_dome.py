"""Precompute the subdivided, uncapped AneuGv2 wall mesh -- using the REAL
pytorch3d.ops.SubdivideMeshes, the exact same call
ood_fix/cfd_data_factory/extract_wss_dataset.py uses for AneuG-Flow/AngioFlow -- and upload the
result to geometry/<case>/dome_subdivided.npz on the Hub.

WHY THIS EXISTS. extract_ghd_graphs_aneugv2.py runs as a self-contained Hugging
Face Job (no local import, no pytorch3d -- see that file's own docstring), so it
originally reimplemented subdivision in plain numpy. That reimplementation is
mathematically equivalent to pytorch3d's (verified: same midpoint-per-edge,
4-way-split topology), but "mathematically equivalent, independently written" is
exactly the pattern that already produced two REAL, silent bugs this session
(the cotangent Laplacian and the WSS knn-interpolation weighting both had two
independently-written "equivalent" implementations quietly diverge). Rather than
risk a third, this script does the ONE thing that actually needs pytorch3d --
subdivision -- locally, once per case, with the literal same library call AneuG-Flow/AngioFlow
uses. extract_ghd_graphs_aneugv2.py then just DOWNLOADS the result; it carries no
subdivision code of its own at all.

THE GHD EIGENBASIS RIDES ALONG. SubdivideMeshes(meshes, feats=...) subdivides
arbitrary per-vertex features with the mesh (same midpoint averaging) -- so the
per-type canonical eigenbasis (assets/ghd_eigenvectors_{bifurcated,sidewall}.npy,
selected down to this case's wall vertices) is subdivided in the SAME call as the
mesh itself, not a separate step. All 144 modes are kept and L2-normalized per
column here (not just the first --n_ghd), so a later extraction run can pick any
--n_ghd without re-running this script.

Nothing in this script runs as a Job -- it's a normal local script (pytorch3d,
huggingface_hub), the same category as angioflow/huggingface/upload_*.py.

    python -m angioflow.data.subdivide_aneugv2_dome --dry_run
    python -m angioflow.data.subdivide_aneugv2_dome --cases cohort_650_0000_t0
    python -m angioflow.data.subdivide_aneugv2_dome --execute --yes
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import Meshes

REPO_ID = "whding123/AneuGv2"
GEO_PREFIX = "geometry"
CFD_PREFIX = "cfd_data"
ASSETS = {0: "assets/ghd_eigenvectors_bifurcated.npy", 1: "assets/ghd_eigenvectors_sidewall.npy"}
MM_TO_M = 1e-3
OUT_NAME = "dome_subdivided.npz"
BATCH_SIZE = 20
MAX_RETRIES = 5
RETRY_DELAY = 10


def load_obj(path):
    """Minimal Wavefront .obj reader, IN FILE ORDER -- copied from
    extract_ghd_graphs_aneugv2.py (not imported -- that script must stay
    self-contained for its own, separate reason; this one just avoids the
    dependency for its own sake)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def uncap_mesh(V, F, wall_vertex_mask, wall_face_mask):
    """Copied from extract_ghd_graphs_aneugv2.py -- identical logic, kept in sync
    by hand since that script can't import this one either."""
    keep_idx = np.where(wall_vertex_mask)[0]
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    F_wall = remap[F[wall_face_mask]]
    assert (F_wall >= 0).all(), "wall_face_mask kept a face with a capped vertex"
    return V[keep_idx], F_wall, keep_idx


def process_case(case_dir, eigvec_cache, token, n_subdivide):
    import json
    with open(os.path.join(case_dir, "meta.json")) as f:
        atype = int(json.load(f)["aneurysm_type"])
    if atype not in ASSETS:
        return None, f"no GHD eigenbasis asset for aneurysm_type={atype}"

    V, F = load_obj(os.path.join(case_dir, "dome.obj"))
    V = V * MM_TO_M
    z = np.load(os.path.join(case_dir, "cap_masks.npz"), allow_pickle=True)
    if len(V) != int(z["dome_n_vertices"]) or len(F) != int(z["dome_n_faces"]):
        return None, "dome.obj vertex/face count disagrees with cap_masks.npz"

    V_wall, F_wall, keep_idx = uncap_mesh(V, F, z["wall_vertex_mask"], z["wall_face_mask"])

    if atype not in eigvec_cache:
        path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", token=token,
                               filename=ASSETS[atype])
        eigvec_cache[atype] = np.load(path).astype(np.float64)   # [N_dome, 144]
    full_eigvec = eigvec_cache[atype]
    if full_eigvec.shape[0] != len(V):
        return None, f"eigenbasis has {full_eigvec.shape[0]} rows, dome.obj has {len(V)} vertices"
    E_coarse = full_eigvec[keep_idx]                              # [n_keep, 144]

    mesh = Meshes(verts=[torch.from_numpy(V_wall).float()],
                  faces=[torch.from_numpy(F_wall).long()])
    feats = torch.from_numpy(E_coarse).float()
    for _ in range(n_subdivide):
        mesh, feats = SubdivideMeshes()(mesh, feats=feats)

    V_fine = mesh.verts_packed().numpy().astype(np.float32)
    F_fine = mesh.faces_packed().numpy().astype(np.int64)
    E_fine = feats.numpy().astype(np.float64)
    E_fine = E_fine / (np.linalg.norm(E_fine, axis=0, keepdims=True) + 1e-12)

    return {
        "pos": V_fine, "faces": F_fine.astype(np.int32),
        "ghd_eigvec": E_fine.astype(np.float32),   # [N_fine, 144], unit-norm per column
        "n_subdivide": n_subdivide, "aneurysm_type": atype,
    }, None


def discover_cases(api):
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    cfd_cases = {f.split("/", 2)[1] for f in files if f.startswith(f"{CFD_PREFIX}/")
                and f.endswith("/wall_data.pt")}
    cap_cases = {f.split("/", 2)[1] for f in files if f.startswith(f"{GEO_PREFIX}/")
                and f.endswith("/cap_masks.npz")}
    done = {f.split("/", 2)[1] for f in files if f.startswith(f"{GEO_PREFIX}/")
           and f.endswith(f"/{OUT_NAME}")}
    return sorted(cfd_cases & cap_cases), done


def download_case(api, case, scratch, token):
    import shutil
    case_dir = os.path.join(scratch, case)
    os.makedirs(case_dir, exist_ok=True)
    for fname in ("dome.obj", "cap_masks.npz", "meta.json"):
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset", token=token,
                        filename=f"{GEO_PREFIX}/{case}/{fname}", local_dir=scratch)
        shutil.move(os.path.join(scratch, GEO_PREFIX, case, fname), os.path.join(case_dir, fname))
    return case_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--n_subdivide", type=int, default=1)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--redo", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--execute", action="store_true", help="actually process + upload (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def main():
    args = parse_args()
    api = HfApi(token=args.token)
    cases, done = discover_cases(api)
    if args.cases:
        cases = [c for c in cases if c in set(args.cases)]
    todo = cases if args.redo else [c for c in cases if c not in done]
    print(f"[plan] {len(cases)} case(s) with dome.obj+cap_masks.npz+wall_data.pt; "
          f"{len(done)} already have {OUT_NAME}; processing {len(todo)}")
    if args.dry_run or not args.execute:
        print("cases (first 20):", todo[:20])
        print("\n[dry run only] pass --execute --yes to process + upload.")
        return
    if not args.yes:
        print("\n[dry run only] --execute given without --yes -- not touching anything.")
        return

    eigvec_cache = {}
    batches = [todo[i:i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    print(f"[run] {len(todo)} case(s) in {len(batches)} batch(es) of up to {args.batch_size}")

    for bi, batch in enumerate(batches, 1):
        ops = []
        with tempfile.TemporaryDirectory() as scratch:
            for case in batch:
                t0 = time.time()
                try:
                    case_dir = download_case(api, case, scratch, args.token)
                    result, reason = process_case(case_dir, eigvec_cache, args.token, args.n_subdivide)
                    if result is None:
                        print(f"[skip] {case}: {reason}")
                        continue
                    out_path = os.path.join(case_dir, OUT_NAME)
                    np.savez(out_path, **result)
                    ops.append(CommitOperationAdd(path_in_repo=f"{GEO_PREFIX}/{case}/{OUT_NAME}",
                                                  path_or_fileobj=out_path))
                    print(f"[ok] {case}: {len(result['pos'])} nodes ({time.time() - t0:.1f}s)")
                except Exception as e:
                    print(f"[fail] {case}: {e}")

            if not ops:
                continue
            delay = RETRY_DELAY
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=ops,
                                      commit_message=f"Add {OUT_NAME} for {len(ops)} case(s) "
                                                     f"(batch {bi}/{len(batches)})")
                    print(f"[{bi}/{len(batches)}] committed {len(ops)} case(s)")
                    break
                except Exception as e:
                    rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
                    wait = max(delay, 60) if rate_limited else delay
                    print(f"[{bi}/{len(batches)}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(wait); delay *= 2
                    else:
                        raise
    print("\n[done]")


if __name__ == "__main__":
    main()
