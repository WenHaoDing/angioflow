"""train_joint.py on a Hugging Face Job: same model, same CLI flags, same training loop
(it calls train_joint.main), with train_hf.py's Job plumbing around it --

  data in       from whding123/AngioFlow-PyG, unzipped to local disk, only the static files
                + the frames trained on (--frames 0): wall graphs (wall/<tag>.zip) and the
                interior point clouds (interior_lean/<tag>/part_*.zip, vel + press -- all
                train_joint reads), plus each dataset's flow_split.json with --add_flow_split
                (huggingface/download_angioflow_pyg.py). All three datasets at frame 0:
                ~260 GB of zips fetched one at a time, ~25 GB kept; ~25 min on an L40S Job.
  run inputs    stats.pt, the interior stats and the inlet waveform, which are not in the
                public dataset, from _meta/ of the PRIVATE checkpoint repo (--ckpt_repo).
                The interior stats file is interior_stats_zeromax.pt with --press_zero_max,
                else interior_stats.pt. Upload once:
                    hf upload whding123/AngioFlow-checkpoints <local interior_stats_zeromax.pt> _meta/interior_stats_zeromax.pt --private
                A missing stats file is recomputed from the training split (as train_joint.py
                does); a missing waveform aborts.
  checkpoints   the run folder is pushed to <ckpt_repo>/<run_name>/ every --push_every_min
  + resume      minutes and at the end, and a rerun of the same command resumes from it --
                exactly as train_hf.py (see its docstring).

Nothing that processes raw meshes (pytorch3d / pyvista / trimesh / AneuG) is imported. The
dataset's data-sanity renders are skipped (--no_data_sanity is forced); the per-epoch joint
sanity figures and loss curves are matplotlib only, so no Xvfb is needed. --max_steps is a
smoke test (e.g. to size --batch_size on the Job's GPU): nothing is pushed.

Launch (from a machine with the hf CLI logged in; the Job clones the PUSHED GitHub repo):

    hf jobs run --flavor rtx-pro-6000 --timeout 24h --secrets HF_TOKEN -d \\
      python:3.11 bash -c 'set -e
        git clone --depth 1 https://github.com/WenHaoDing/angioflow.git /work/angioflow
        git -C /work/angioflow log --oneline -1
        pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
        pip install -q numpy==2.3.5 scipy==1.16.3 huggingface_hub==1.20.1 \\
            torch_geometric==2.7.0 matplotlib==3.9.1
        pip install -q torch_scatter==2.1.2 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
        cd /work && exec python -m angioflow.training.train_joint_hf --run_name <run> ... <train_joint.py flags>'

torch 2.7 + CUDA 12.8 because the RTX PRO 6000 (and any Blackwell GPU) has no kernels in
torch 2.4's cu121 build; on an L40S / A100 / H200 the torch-2.4.0+cu121 pair works too.

--save_dir / --ckpt_root / --interior_dir / --waveform_path / --extracted are set by this
script (any given are ignored); every other train_joint.py flag passes through unchanged.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import sys
import time

from huggingface_hub import HfApi, snapshot_download

import angioflow.training.train_joint as joint_mod
from angioflow.data.datasets import AngioFlowWallDataset
from angioflow.data.dataset_tags import DATASET_TAGS
from angioflow.huggingface import download_angioflow_pyg
from angioflow.training.train_hf import (CKPT_REPO, DATA_REPO, META_PREFIX, RESUME_FREE_KEYS,
                                         HubPusher, check_resume_config, fetch_meta,
                                         run_files_on_hub)

WORK_DIR = "/tmp/angioflow_joint_hf"
JOINT_RESUME_FREE_KEYS = RESUME_FREE_KEYS | {"interior_dir", "no_data_sanity"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="All other flags go to train_joint.py "
                                       "(python -m angioflow.training.train_joint -h).")
    p.add_argument("--ckpt_repo", default=CKPT_REPO, help="private model repo for run inputs (_meta/) and runs")
    p.add_argument("--data_repo", default=DATA_REPO)
    p.add_argument("--stats_file", default="stats.pt",
                   help=f"file under {META_PREFIX}/ of --ckpt_repo used as <save_dir>/stats.pt")
    p.add_argument("--work_dir", default=WORK_DIR, help="local data/ and ckpt/ go here")
    p.add_argument("--push_every_min", type=float, default=20.0)
    p.add_argument("--skip_download", action="store_true",
                   help="data already in <work_dir>/data (e.g. a local test): don't download")
    p.add_argument("--no_push", action="store_true", help="never write to the Hub (local test)")
    p.add_argument("--allow_config_change", action="store_true",
                   help="resume even if the Hub run's args.json differs from these settings")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    own, rest = p.parse_known_args()

    argv, sys.argv = sys.argv, [sys.argv[0]] + rest
    try:
        targs = joint_mod.get_args()
    finally:
        sys.argv = argv
    return own, targs


def main():
    own, targs = parse_args()
    smoke = bool(targs.max_steps)
    push = not (own.no_push or smoke)
    api = HfApi(token=own.token)
    data_dir = os.path.join(own.work_dir, "data")
    targs.extracted = True
    targs.save_dir = data_dir
    targs.interior_dir = None                       # = <save_dir>/interior, where the download unzips
    targs.ckpt_root = os.path.join(own.work_dir, "ckpt")
    targs.no_data_sanity = True
    ckpt_dir = os.path.join(targs.ckpt_root, targs.run_name)
    print(f"[hf] run {targs.run_name}  data -> {data_dir}  ckpt -> {ckpt_dir}  "
          f"push -> {own.ckpt_repo + '/' + targs.run_name if push else 'off' + (' (smoke test)' if smoke else '')}",
          flush=True)
    if push:            # fail now, not after hours of training, if the token can't write there
        api.create_repo(own.ckpt_repo, private=True, exist_ok=True)

    # data: wall graphs + interior point clouds (+ flow-split tables)
    tags = targs.versions.split(",") if targs.versions else list(DATASET_TAGS)
    if not own.skip_download:
        t0 = time.time()
        argv = ["--what", "wall", "interior_lean"] + (["flow_split"] if targs.add_flow_split else []) + \
               ["--datasets", *tags, "--dest", data_dir, "--repo_id", own.data_repo]
        if targs.frames != "all":
            argv += ["--frames", targs.frames]
        if own.token:
            argv += ["--token", own.token]
        download_angioflow_pyg.main(argv)
        print(f"[hf] data ready in {(time.time() - t0) / 60:.1f} min", flush=True)

    # run inputs not in the public dataset
    interior_stats = os.path.basename(joint_mod.interior_stats_path(targs))
    for name, dest in [(own.stats_file, os.path.join(data_dir, "stats.pt")),
                       (interior_stats, os.path.join(data_dir, interior_stats))]:
        if os.path.exists(dest):
            continue
        if fetch_meta(own, name, dest):
            print(f"[hf] {os.path.basename(dest)} <- {own.ckpt_repo}/{META_PREFIX}/{name}")
        else:
            print(f"[hf] [warn] no {META_PREFIX}/{name} on {own.ckpt_repo}: recomputed from the "
                  f"training split -- it will not match a local run's {os.path.basename(dest)}", flush=True)
    if not os.path.exists(targs.waveform_path):
        local = os.path.join(own.work_dir, "meta", os.path.basename(targs.waveform_path))
        if not fetch_meta(own, os.path.basename(targs.waveform_path), local):
            raise SystemExit(f"[abort] waveform {os.path.basename(targs.waveform_path)} not under "
                             f"{own.ckpt_repo}/{META_PREFIX}/ -- upload it (see train_hf.py's docstring)")
        targs.waveform_path = local
        print(f"[hf] waveform <- {own.ckpt_repo}/{META_PREFIX}/{os.path.basename(local)}")

    # resume an existing run
    on_hub = run_files_on_hub(api, own.ckpt_repo, targs.run_name)
    if on_hub:
        snapshot_download(own.ckpt_repo, allow_patterns=[f"{targs.run_name}/*"], local_dir=targs.ckpt_root,
                          token=own.token)
        print(f"[hf] resuming: {len(on_hub)} file(s) of {own.ckpt_repo}/{targs.run_name} downloaded", flush=True)
    check_resume_config(ckpt_dir, targs, own.allow_config_change, JOINT_RESUME_FREE_KEYS)
    if targs.wandb and not targs.wandb_run_id:
        id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
        if os.path.exists(id_path):
            with open(id_path) as f:
                targs.wandb_run_id = f.read().strip()
        else:
            import wandb
            targs.wandb_run_id = wandb.util.generate_id()
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(id_path, "w") as f:
                f.write(targs.wandb_run_id)
        print(f"[hf] W&B run id {targs.wandb_run_id}")

    # small /dev/shm (Docker default 64 MB) kills DataLoader workers mid-epoch
    shm = shutil.disk_usage("/dev/shm").total if os.path.isdir("/dev/shm") else 0
    print(f"[hf] /dev/shm {shm / 2**30:.1f} GiB, {os.cpu_count()} CPUs, accelerator {os.environ.get('ACCELERATOR')}")
    if targs.num_workers and shm < 2**30:
        print(f"[hf] [warn] /dev/shm < 1 GiB: num_workers {targs.num_workers} -> 0")
        targs.num_workers = 0

    AngioFlowWallDataset.render_sanity = lambda self, *a, **k: None

    pusher = HubPusher(api, own.ckpt_repo, targs.run_name, ckpt_dir,
                       os.path.join(own.work_dir, "push", targs.run_name), own.push_every_min) if push else None
    # SIGTERM -> unwind so the final push still runs (needs `exec python`; see train_hf.py)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        joint_mod.main(targs, epoch_end_hook=pusher.epoch_end if pusher else None)
    finally:
        if pusher:
            pusher.final()


if __name__ == "__main__":
    main()
