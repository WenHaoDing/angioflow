"""data/register_cfd_shapes_angioflow.py, on the fly against whding123/AngioFlow: for each
case, download its CFD wall from the Hub, register it, upload the registration, delete
the download -- for cases re-meshed/re-exported on the Hub that have no local copy.

Per case:
  1. download   cfd_data/<case>/wall_data.pt (+ flowsplit_ratio.txt, if the Hub has it)
                into a temporary case folder <work_dir>/cfd_data/<case>/
  2. register   the ORIGINAL script's process_case on that folder -- identical logic:
                aneurysm type from endpoints_manual.npy, focused mesh from the local
                geometry folders (PREPROCESS_DIRS) RANSAC+ICP-aligned to the CFD wall,
                zone matching, geo_dist. Its sanity renders go to the usual SANITY_ROOT/<case>/
                (alignment/mesh/registration/geo_dist PNGs; the raw-field WSS/velocity GIFs
                need the ~3 GB blood_data.pt, which is not downloaded, so that step is off)
  3. upload     wall_mesh_registration.pt -> Hub cfd_data/<case>/, batched --commit_every
                cases per commit (and whatever is pending when the run ends or fails)
  4. clean up   the temporary case folder

Why local and not a Job: the aneurysm types and focused meshes come from the local geometry
folders (data/geometry/ImperialNHS, AneuX_stable), which are not on the Hub, and the
alignment needs open3d + pytorch3d. Run it in the `new` env.

Manual fixes (the original's MANUAL-FIX WORKFLOW): if a case has a hand-clipped focused
mesh -- <SANITY_ROOT>/failed_wall_meshes_manually_fixed/<case>.<ext> first, else the local
mirror's <LOCAL_ROOT>/<case>/manually_fixed_clipped_reconstruction.ply -- it is copied into
the temporary case folder, where process_case prefers it, as in the original. A clip cut
from an OLD mesh still goes through the same ICP fitness/distance checks against the new one.
A case that is skipped or fails is listed in <SANITY_ROOT>/bad_alignment_cases_hf.txt (kept
across runs: a rerun replaces only the lines of the cases it processed) and
its NEW wall mesh is exported to <SANITY_ROOT>/0000_failed_wall_meshes/<case>.obj, ready to
clip; drop the clipped file into failed_wall_meshes_manually_fixed/ and rerun with --cases.

The local mirror (LOCAL_ROOT) is never written to: its wall_data.pt files are the old exports.

Default selection: every case whose Hub folder has wall_data.pt but no
wall_mesh_registration.pt -- so a rerun picks up where the last one stopped. --cases
limits it; --redo also re-registers cases that already have one (their wall_graph_static.pt
then no longer matches and must be rebuilt: precompute_wall_static + upload_wall_graph_static).

    conda activate new
    python -m angioflow.huggingface.register_cfd_shapes_angioflow_hf                     # plan only
    python -m angioflow.huggingface.register_cfd_shapes_angioflow_hf --cases C0065_cut2 --execute --no_upload
    python -m angioflow.huggingface.register_cfd_shapes_angioflow_hf --execute --yes      # process + upload
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import time

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

import angioflow.data.register_cfd_shapes_angioflow as reg_mod
from angioflow.data.register_cfd_shapes_angioflow import load_aneurysm_type, process_case
from angioflow.data.registration_sanity import SANITY_ROOT
from angioflow.huggingface.extract_pyg_stable import with_retry

REPO_ID = "whding123/AngioFlow"
REPO_TYPE = "dataset"
PREFIX = "cfd_data"
FNAME = "wall_mesh_registration.pt"
LOCAL_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
WORK_DIR = "/media/yaplab2/wd8tb/wenhao/datasets/_angioflow_registration_hf_tmp"
# same sources, same order as data/register_cfd_shapes_angioflow.py's __main__
PREPROCESS_DIRS = [
    "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS",
    "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_stable",
]
MANUAL_NAME = "manually_fixed_clipped_reconstruction.ply"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--cases", nargs="*", default=None, help="only these cases (default: all lacking a registration)")
    p.add_argument("--redo", action="store_true", help="also re-register cases that already have one on the Hub")
    p.add_argument("--work_dir", default=WORK_DIR, help="temporary case folders + pending uploads")
    p.add_argument("--sanity_dir", default=SANITY_ROOT)
    p.add_argument("--local_root", default=LOCAL_ROOT, help="local mirror, read only for manual fixes")
    p.add_argument("--commit_every", type=int, default=20, help="cases per upload commit")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--execute", action="store_true", help="process the cases (without it: plan only)")
    p.add_argument("--yes", action="store_true", help="with --execute: also upload")
    p.add_argument("--no_upload", action="store_true",
                   help="with --execute: keep the results in <work_dir>/results/<case>/ instead of uploading")
    return p.parse_args()


def hub_cases(api, repo):
    """{case: set of file names directly under cfd_data/<case>/}."""
    files = with_retry(lambda: api.list_repo_files(repo, repo_type=REPO_TYPE), f"list {repo}")
    out = {}
    for f in files:
        parts = f.split("/")
        if len(parts) == 3 and parts[0] == PREFIX and not parts[1].startswith("_"):
            out.setdefault(parts[1], set()).add(parts[2])
    return {c: s for c, s in out.items() if "wall_data.pt" in s}


def manual_fix_source(case, args):
    fixed = sorted(glob.glob(os.path.join(args.sanity_dir, "failed_wall_meshes_manually_fixed", f"{case}.*")))
    if fixed:
        return fixed[0]
    local = os.path.join(args.local_root, case, MANUAL_NAME)
    return local if os.path.exists(local) else None


def install_manual_fix(src, case_dir):
    """As the original's import_manually_fixed_meshes: read anything open3d reads, write .ply."""
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(src)
    if len(m.vertices) == 0:
        print(f"  [warn] empty / unreadable manual fix {src} -- ignored")
        return
    o3d.io.write_triangle_mesh(os.path.join(case_dir, MANUAL_NAME), m)


def export_failed_wall(case_dir, out_dir):
    """As the original's export_failed_wall_meshes, from the NEW (downloaded) wall_data.pt."""
    import open3d as o3d
    wd = torch.load(os.path.join(case_dir, "wall_data.pt"), weights_only=False)
    if "wall" not in wd or "faces" not in wd["wall"]:
        return None
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(wd["wall"]["pos"].numpy().astype(np.float64)),
        o3d.utility.Vector3iVector(wd["wall"]["faces"].numpy().astype(np.int64)))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{os.path.basename(case_dir)}.obj")
    o3d.io.write_triangle_mesh(path, m)
    return path


class Uploader:
    """Collects finished registrations and commits them --commit_every at a time."""

    def __init__(self, api, args):
        self.api, self.args = api, args
        self.pending = []           # (case, local path)
        self.n_uploaded = 0

    def add(self, case, path):
        self.pending.append((case, path))
        if len(self.pending) >= self.args.commit_every:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        ops = [CommitOperationAdd(path_in_repo=f"{PREFIX}/{c}/{FNAME}", path_or_fileobj=p) for c, p in self.pending]
        names = [c for c, _ in self.pending]
        with_retry(lambda: self.api.create_commit(
            self.args.repo_id, repo_type=REPO_TYPE, operations=ops,
            commit_message=f"{FNAME} for {len(ops)} re-exported case(s) (register_cfd_shapes_angioflow_hf)"),
            f"commit {len(ops)} registration(s)")
        for _, p in self.pending:
            os.remove(p)
        self.n_uploaded += len(ops)
        print(f"[upload] {len(ops)} registration(s) -> {self.args.repo_id}/{PREFIX}/: {', '.join(names)}", flush=True)
        self.pending = []


def main():
    args = parse_args()
    api = HfApi(token=args.token)
    on_hub = hub_cases(api, args.repo_id)
    if args.cases:
        unknown = sorted(set(args.cases) - set(on_hub))
        if unknown:
            raise SystemExit(f"[abort] not on {args.repo_id}/{PREFIX} (or no wall_data.pt): {unknown}")
        cases = sorted(set(args.cases))
    else:
        cases = sorted(on_hub)
    have = [c for c in cases if FNAME in on_hub[c]]
    todo = cases if args.redo else [c for c in cases if FNAME not in on_hub[c]]

    no_type = [c for c in todo if load_aneurysm_type(c, PREPROCESS_DIRS) is None]
    manual = {c: manual_fix_source(c, args) for c in todo}
    print(f"[plan] {len(on_hub)} case(s) on {args.repo_id}/{PREFIX}; {len(cases)} selected, {len(have)} already "
          f"registered{' (redone: --redo)' if args.redo else ' (skipped)'}; {len(todo)} to register")
    print(f"[plan] {sum('flowsplit_ratio.txt' in on_hub[c] for c in todo)} of them have flowsplit_ratio.txt on "
          f"the Hub (else flow_split = zeros, as in the original)")
    print(f"[plan] {sum(v is not None for v in manual.values())} have a manual focused-mesh fix: "
          f"{[c for c, v in manual.items() if v]}")
    if no_type:
        print(f"[plan] {len(no_type)} have no aneurysm type in PREPROCESS_DIRS and will be skipped: {no_type}")
    upload = args.execute and args.yes and not args.no_upload
    print(f"[plan] upload: {upload}  (sanity -> {args.sanity_dir}/<case>/)")
    if not args.execute or not todo:
        print("\n[plan only] add --execute --yes to process + upload (--execute --no_upload to keep results locally)")
        return
    if not args.yes and not args.no_upload:
        print("\n[plan only] --execute needs --yes to upload, or --no_upload")
        return

    reg_mod.sanity_check_case = lambda *a, **k: None    # needs blood_data.pt (see docstring)
    pending_dir = os.path.join(args.work_dir, "pending" if upload else "results")
    uploader = Uploader(api, args) if upload else None
    bad, done, n_ok, t_run = [], set(), 0, time.time()
    bad_path = os.path.join(args.sanity_dir, "bad_alignment_cases_hf.txt")
    earlier = []
    if os.path.exists(bad_path):
        with open(bad_path) as f:
            earlier = [tuple(line.rstrip("\n").split("\t", 1)) for line in f if "\t" in line]
    try:
        for i, case in enumerate(todo, 1):
            t0 = time.time()
            print(f"\n===== [{i}/{len(todo)}] {case}", flush=True)
            case_dir = os.path.join(args.work_dir, PREFIX, case)
            shutil.rmtree(case_dir, ignore_errors=True)
            try:
                for name in ["wall_data.pt"] + (["flowsplit_ratio.txt"] if "flowsplit_ratio.txt" in on_hub[case] else []):
                    with_retry(lambda: hf_hub_download(args.repo_id, f"{PREFIX}/{case}/{name}", repo_type=REPO_TYPE,
                                                       token=args.token, local_dir=args.work_dir),
                               f"download {case}/{name}")
                if manual[case]:
                    install_manual_fix(manual[case], case_dir)
                    print(f"  [manual] focused mesh from {manual[case]}")
                reason = process_case(case_dir, PREPROCESS_DIRS, sanity_dir=args.sanity_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                reason = f"exception: {e}"
            if reason is None:
                out = os.path.join(pending_dir, case, FNAME)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.move(os.path.join(case_dir, FNAME), out)
                n_ok += 1
                if uploader:
                    uploader.add(case, out)
            else:
                bad.append((case, reason))
                obj = None
                if os.path.exists(os.path.join(case_dir, "wall_data.pt")):
                    obj = export_failed_wall(case_dir, os.path.join(args.sanity_dir, "0000_failed_wall_meshes"))
                print(f"[bad] {case}: {reason}" + (f"  (wall mesh -> {obj})" if obj else ""), flush=True)
            done.add(case)
            shutil.rmtree(case_dir, ignore_errors=True)
            print(f"[time] {case}: {time.time() - t0:.0f}s", flush=True)
    finally:
        if uploader:
            uploader.flush()
        with open(bad_path, "w") as f:
            f.writelines(f"{c}\t{r}\n" for c, r in [x for x in earlier if x[0] not in done] + bad)
        print(f"\n[summary] {n_ok} registered ({uploader.n_uploaded if uploader else 0} uploaded"
              f"{'' if uploader else f'; results in {pending_dir}'}), {len(bad)} skipped/failed -> {bad_path}  "
              f"[{(time.time() - t_run) / 60:.1f} min]")


if __name__ == "__main__":
    main()
