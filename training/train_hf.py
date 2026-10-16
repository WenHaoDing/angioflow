"""train.py on a Hugging Face Job: same model, same CLI flags, same training loop
(it calls train.main), plus what a Job needs around it --

  data in       wall graphs from whding123/AngioFlow-PyG, unzipped to local disk --
                only the static files + the frames trained on (--frames 0: ~22 GB of
                the 79 GB of zips) -- plus each dataset's flow_split.json with
                --add_flow_split (huggingface/download_angioflow_pyg.py)
  run inputs    stats.pt and the inlet waveform, which are not in the public dataset,
                from _meta/ of the PRIVATE checkpoint repo (--ckpt_repo). Upload once:
                    hf upload whding123/AngioFlow-checkpoints <local stats.pt> _meta/stats.pt --private
                    hf upload whding123/AngioFlow-checkpoints <waveform_yiying.txt> _meta/waveform_yiying.txt --private
                A missing stats file is recomputed from the training split (as train.py
                does); a missing waveform aborts.
  checkpoints   the run folder (last/best/epoch_XXXX.pth, history.csv, args.json,
                sanity/) is pushed to <ckpt_repo>/<run_name>/ every --push_every_min
                minutes (after that epoch's checkpoints are written, uploaded in the
                background) and at the end. A Job's disk is discarded when it ends, so
                a timeout loses at most the minutes since the last push.
  resume        if <ckpt_repo>/<run_name>/ already exists, it is downloaded first and
                train.py resumes from its last.pth -- rerun the same command (or launch
                with --attempts N) to continue after a timeout or crash. Refuses if the
                stored args.json was trained with other settings (--allow_config_change).
                The W&B run id is kept in the run folder, so a resumed run continues
                the same W&B run.

Code that only processes raw meshes (pytorch3d / pyvista / AneuG) is never imported.
The dataset's data-sanity renders are skipped (already rendered locally); the per-epoch
WSS sanity snapshots need pyvista + Xvfb and are skipped with a warning if they fail.
--max_steps is a smoke test: nothing is pushed.

Launch (from a machine with the hf CLI logged in; the Job clones the PUSHED GitHub repo):

    hf jobs run --flavor l40sx1 --timeout 24h --secrets HF_TOKEN --secrets WANDB_API_KEY -d \\
      python:3.11 bash -c 'set -e
        apt-get update -qq && apt-get install -y -qq xvfb libgl1 libxrender1 > /dev/null
        git clone --depth 1 https://github.com/WenHaoDing/angioflow.git /work/angioflow
        git -C /work/angioflow log --oneline -1
        pip install -q torch==2.4.0 numpy==2.3.5 scipy==1.16.3 huggingface_hub==1.20.1 \\
            torch_geometric==2.7.0 matplotlib==3.9.1 wandb==0.23.1 pyvista==0.42.3 vtk==9.3.1
        pip install -q torch_scatter==2.1.2 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
        cd /work && exec python -m angioflow.training.train_hf --model hierarchical_gps --run_name <run> ... <train.py flags>'

--save_dir / --ckpt_root / --waveform_path / --extracted are set by this script (any
given are ignored); every other train.py flag passes through unchanged. Follow with
`hf jobs logs -f <job_id>`; fetch the run with
`hf download whding123/AngioFlow-checkpoints --include "<run>/*" --local-dir <ckpt_root>`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.hf_api import RepoFile

import angioflow.training.train as train_mod
from angioflow.data.datasets import AngioFlowWallDataset
from angioflow.data.dataset_tags import DATASET_TAGS
from angioflow.huggingface import download_angioflow_pyg
from angioflow.huggingface.extract_pyg_stable import with_retry

CKPT_REPO = "whding123/AngioFlow-checkpoints"
DATA_REPO = "whding123/AngioFlow-PyG"
META_PREFIX = "_meta"
WORK_DIR = "/tmp/angioflow_hf"
# args.json keys a resumed run may change: paths/device of this machine, run length,
# logging cadence, loader workers
RESUME_FREE_KEYS = {"save_dir", "ckpt_root", "waveform_path", "device", "wandb_run_id", "total_epochs",
                    "sanity_every", "ckpt_every", "num_workers", "wandb", "max_steps", "resume"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="All other flags go to train.py (python -m angioflow.training.train --model <m> -h).")
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
        targs = train_mod.get_args()
    finally:
        sys.argv = argv
    return own, targs


def run_files_on_hub(api, repo, run):
    """Files of <repo>/<run>/; [] only if the repo or the folder is confirmed missing."""
    def _list():
        try:
            return [f.path for f in api.list_repo_tree(repo, path_in_repo=run, recursive=True)
                    if isinstance(f, RepoFile)]
        except (RepositoryNotFoundError, EntryNotFoundError):
            return []
    return with_retry(_list, f"list {repo}/{run}")


def fetch_meta(own, name, dest):
    """<ckpt_repo>/_meta/<name> -> dest; False if it isn't there (not retried)."""
    def _get():
        try:
            return hf_hub_download(own.ckpt_repo, f"{META_PREFIX}/{name}", token=own.token,
                                   local_dir=os.path.join(own.work_dir, "_meta_dl"))
        except (RepositoryNotFoundError, EntryNotFoundError):
            return None
    local = with_retry(_get, f"download {META_PREFIX}/{name}")
    if local is None:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(local, dest)
    return True


def check_resume_config(ckpt_dir, targs, allow):
    path = os.path.join(ckpt_dir, "args.json")
    if not os.path.exists(path):
        return
    with open(path) as f:
        old = json.load(f)
    new = json.loads(json.dumps(vars(targs), default=str))
    diff = {k: (old.get(k, "<absent>"), new.get(k, "<absent>")) for k in sorted(set(old) | set(new))
            if k not in RESUME_FREE_KEYS and old.get(k, "<absent>") != new.get(k, "<absent>")}
    if diff:
        msg = "\n".join(f"  {k}: run={a!r}  now={b!r}" for k, (a, b) in diff.items())
        if not allow:
            raise SystemExit(f"[abort] {path} was trained with other settings:\n{msg}\n"
                             f"use a new --run_name, or --allow_config_change to resume anyway")
        print(f"[resume] settings changed (--allow_config_change):\n{msg}")


class HubPusher:
    """Pushes the run folder to <repo>/<run>/. Each push first copies the folder to a
    staging dir in the training thread -- right after the checkpoints are written, so the
    copy is never half-saved -- then uploads it in a background thread; a push due while
    the last one is still uploading waits for the next epoch."""

    def __init__(self, api, repo, run, ckpt_dir, staging, every_min):
        self.api, self.repo, self.run, self.ckpt_dir = api, repo, run, ckpt_dir
        self.staging, self.every_s = staging, every_min * 60
        self.last = time.time()
        self.thread = None
        self.epoch = None

    def _stage(self):
        shutil.rmtree(self.staging, ignore_errors=True)
        shutil.copytree(self.ckpt_dir, self.staging, ignore=shutil.ignore_patterns("*.tmp"))

    def _upload(self, epoch):
        with_retry(lambda: self.api.upload_folder(
            repo_id=self.repo, folder_path=self.staging, path_in_repo=self.run,
            commit_message=f"{self.run}: through epoch {epoch}"), f"push {self.run}")
        print(f"[push] {self.repo}/{self.run} through epoch {epoch}", flush=True)

    def _upload_bg(self, epoch):
        try:
            self._upload(epoch)
        except Exception as e:          # the next push retries; the final push raises
            print(f"[push] failed (epoch {epoch}), will retry at the next push: {e}", flush=True)

    def epoch_end(self, epoch, ckpt_dir):
        self.epoch = epoch
        if time.time() - self.last < self.every_s or (self.thread and self.thread.is_alive()):
            return
        self._stage()
        self.last = time.time()
        self.thread = threading.Thread(target=self._upload_bg, args=(epoch,), daemon=True)
        self.thread.start()

    def final(self):
        if self.thread:
            self.thread.join()
        if not os.path.isdir(self.ckpt_dir):
            return
        self._stage()
        self._upload(self.epoch if self.epoch is not None else "start")


def skip_failing_sanity(fn):
    """The per-epoch WSS snapshots render with pyvista + Xvfb; if that fails on the Job,
    warn once and stop trying instead of killing the run."""
    state = {"off": False}

    def wrapped(*a, **k):
        if state["off"]:
            return
        try:
            return fn(*a, **k)
        except Exception as e:
            state["off"] = True
            print(f"[sanity] WSS snapshot rendering failed, disabled for this run: {e!r}", flush=True)
    return wrapped


def main():
    own, targs = parse_args()
    smoke = bool(targs.max_steps)
    push = not (own.no_push or smoke)
    api = HfApi(token=own.token)
    data_dir = os.path.join(own.work_dir, "data")
    targs.extracted = True
    targs.save_dir = data_dir
    targs.ckpt_root = os.path.join(own.work_dir, "ckpt")
    ckpt_dir = os.path.join(targs.ckpt_root, targs.run_name)
    print(f"[hf] run {targs.run_name}  data -> {data_dir}  ckpt -> {ckpt_dir}  "
          f"push -> {own.ckpt_repo + '/' + targs.run_name if push else 'off' + (' (smoke test)' if smoke else '')}",
          flush=True)
    if push:            # fail now, not after hours of training, if the token can't write there
        api.create_repo(own.ckpt_repo, private=True, exist_ok=True)

    # data: wall graphs (+ flow-split tables)
    tags = targs.versions.split(",") if targs.versions else list(DATASET_TAGS)
    if not own.skip_download:
        t0 = time.time()
        argv = ["--what", "wall"] + (["flow_split"] if targs.add_flow_split else []) + \
               ["--datasets", *tags, "--dest", data_dir, "--repo_id", own.data_repo]
        if targs.frames != "all":
            argv += ["--frames", targs.frames]
        if own.token:
            argv += ["--token", own.token]
        download_angioflow_pyg.main(argv)
        print(f"[hf] data ready in {(time.time() - t0) / 60:.1f} min", flush=True)

    # run inputs not in the public dataset
    if not os.path.exists(os.path.join(data_dir, "stats.pt")):
        if fetch_meta(own, own.stats_file, os.path.join(data_dir, "stats.pt")):
            print(f"[hf] stats <- {own.ckpt_repo}/{META_PREFIX}/{own.stats_file}")
        else:
            print(f"[hf] [warn] no {META_PREFIX}/{own.stats_file} on {own.ckpt_repo}: stats are recomputed "
                  f"from the training split -- they will not match a local run's stats.pt", flush=True)
    if not os.path.exists(targs.waveform_path):
        local = os.path.join(own.work_dir, "meta", os.path.basename(targs.waveform_path))
        if not fetch_meta(own, os.path.basename(targs.waveform_path), local):
            raise SystemExit(f"[abort] waveform {os.path.basename(targs.waveform_path)} not under "
                             f"{own.ckpt_repo}/{META_PREFIX}/ -- upload it (see this script's docstring)")
        targs.waveform_path = local
        print(f"[hf] waveform <- {own.ckpt_repo}/{META_PREFIX}/{os.path.basename(local)}")

    # resume an existing run
    on_hub = run_files_on_hub(api, own.ckpt_repo, targs.run_name)
    if on_hub:
        snapshot_download(own.ckpt_repo, allow_patterns=[f"{targs.run_name}/*"], local_dir=targs.ckpt_root,
                          token=own.token)
        print(f"[hf] resuming: {len(on_hub)} file(s) of {own.ckpt_repo}/{targs.run_name} downloaded", flush=True)
    check_resume_config(ckpt_dir, targs, own.allow_config_change)
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
    train_mod.save_wss_sanity = skip_failing_sanity(train_mod.save_wss_sanity)

    pusher = HubPusher(api, own.ckpt_repo, targs.run_name, ckpt_dir,
                       os.path.join(own.work_dir, "push", targs.run_name), own.push_every_min) if push else None
    # if the Job is stopped with SIGTERM (needs `exec python` in the launch command: as PID 1,
    # bash would not pass it on), unwind so the final push still runs -- best effort.
    # Checkpoints are saved via rename, so an interrupted save never reaches the Hub.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        train_mod.main(targs, epoch_end_hook=pusher.epoch_end if pusher else None)
    finally:
        if pusher:
            pusher.final()


if __name__ == "__main__":
    main()
