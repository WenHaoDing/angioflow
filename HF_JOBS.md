# Hugging Face Jobs: training, data, checkpoints

How this project trains on Hugging Face Jobs: what lives in which Hub repo, how a training Job is launched, resumed and monitored, how to write a Job wrapper for a new trainer, and the problems already hit. For the project overview, local training and evaluation, see [AGENT_GUIDE.md](AGENT_GUIDE.md).

## Ground rules

- **Never launch a GPU Job yourself.** Write the full `hf jobs run ...` command and give it to the user. CPU Jobs need the user's go-ahead too. Cancelling, re-running and uploading checkpoints are also the user's call.
- **Read-only commands are fine:** `hf jobs ps`, `logs`, `stats`, `inspect`, `hardware`, and listing repo files.
- **A Job clones the pushed `master` from GitHub.** Commit when asked; the user pushes, because this shell has no GitHub credentials. Check with `git fetch && git status -sb` that `master` is not ahead of `origin/master` before handing over a command that depends on new code.
- **Smoke-test first.** Use `--max_steps N` for a short run that pushes nothing.

## The Hub repos

| Repo | Type | Visibility | Content |
|---|---|---|---|
| `whding123/AngioFlow-PyG` | dataset | public | Extracted training data: wall graphs, lumen point clouds and flow-split tables. Layout in [AGENT_GUIDE.md](AGENT_GUIDE.md#what-angioflow-pyg-contains) |
| `whding123/AngioFlow-checkpoints` | model | **private** | `_meta/`: run inputs that are not in the dataset. `_init/`: checkpoints that warm-start new runs. `<run_name>/`: one folder per training run |
| `whding123/AngioFlow`, `AneuGv2`, `AneuG-Flow` | dataset | | Source CFD data; only needed to rebuild the training data |

Files in `_meta/` (training normalises with these, so they must match the ones the run was meant to use):

| File | What |
|---|---|
| `stats.pt` | WSS and position normalisation. Identical to the local root's `stats.pt` (sha256 checked 2026-11-06) |
| `interior_stats_zeromax.pt` | Velocity/pressure normalisation for `--press_zero_max` |
| `stats_aneugflow_angioflow.pt` | Older stats for runs on aneug_flow + angioflow only |
| `waveform_yiying.txt` | Inlet waveform; 5 cycles, of which training uses the first. See "Time" below |

## The `hf` CLI

`hf` is on the PATH (version 2.x; version 1.8 rejects the `rtx-pro-6000` flavour). It is logged in as `whding123`.

| Command | Use |
|---|---|
| `hf jobs hardware` | Flavours with their vCPUs, RAM, disk, GPU and price |
| `hf jobs run --flavor <f> --timeout 48h --secrets HF_TOKEN -d python:3.11 bash -c '...'` | Start a Job, detached (`-d`) |
| `hf jobs ps` | Running Jobs, with ids and run times |
| `hf jobs logs <id>` (`-f` to follow) | Job output. Pipe it through `grep -v "B/s\]"` to hide download progress bars |
| `hf jobs stats <id>` | One snapshot of CPU %, RAM used/total and GPU use/memory |
| `hf jobs inspect <id>` | Status (`RUNNING`, `CANCELED`, …), flavour, command, timings |
| `hf jobs cancel <id>` | Stop a Job. SIGTERM reaches the trainer, which pushes once more before exiting |
| `hf jobs ssh <id>` | Shell into a Job, but only if it was started with `--ssh` and the user's SSH key is registered on the Hub |
| `hf upload <repo> <local file> <path in repo>` | Upload one file |
| `hf download <repo> --include "<run>/*" --local-dir <dir>` | Fetch a run folder |

From Python, `huggingface_hub.HfApi().list_repo_files("whding123/AngioFlow-checkpoints")` lists what is on the Hub. `get_paths_info(...)` gives each file's size and sha256, so you can compare it with a local file without downloading it.

### Hardware

Training uses `rtx-pro-6000`: **23 vCPUs, 256 GB RAM, 1× RTX PRO 6000 (96 GB)**, about $2.75/h.

- **Software:** the card is Blackwell, so it needs **torch 2.7.0 + cu128** and the PyG wheels built for torch 2.7.0+cu128. torch 2.4/cu121 has no kernels for it.
- **Disk:** `/tmp` reported 13–14 TB free, so all 80 frames (about 120 GB unpacked) fit easily.
- **What the Job reports:** inside the Job, `os.cpu_count()` and `/dev/shm` show the **host** (192 CPUs, about 2 TB of shared memory), not the flavour's 23 vCPUs. Size `--num_workers` from `hf jobs hardware`, not from those log lines.

## Training a model on a Job

The wrappers [training/train_joint_hf.py](training/train_joint_hf.py) (joint WSS + velocity + pressure model) and [training/train_hf.py](training/train_hf.py) (WSS-only) take every flag of the local trainer and call its `main()`. Around it they:

1. **Download the data** for `--versions` and `--frames` from AngioFlow-PyG, by calling `huggingface/download_angioflow_pyg.py`. This goes into `/tmp/angioflow_joint_hf/data`. All three datasets at all frames is 262 GB of zips, and took **8.3 min** on `rtx-pro-6000` (about 600 MB/s). With `--frames 0` the same zips are fetched but only frame 0 is unpacked.
2. **Fetch the run inputs** (`stats.pt`, the interior stats and the waveform) from `_meta/` on the checkpoint repo. A missing stats file is recomputed from the training split, with a warning; a missing waveform aborts the Job.
3. **Fetch the starting weights:** with `--init_ckpt _init/<name>.pth`, that file is downloaded and passed as `--init_from` (new optimizer and scheduler; ignored on resume).
4. **Resume** if `<ckpt_repo>/<run_name>/` already exists: it is downloaded and training continues from its `last.pth`.
5. **Train**, pushing the run folder to `<ckpt_repo>/<run_name>/` after an epoch once 20 min have passed since the last push (`--push_every_min`), and once more at the end, including after a timeout or `hf jobs cancel`.

The wrapper sets `--save_dir`, `--ckpt_root`, `--interior_dir`, `--waveform_path` and `--extracted` itself; any you pass are ignored. It also forces `--no_data_sanity`.

### The launch command

```bash
hf jobs run --flavor rtx-pro-6000 --timeout 48h --secrets HF_TOKEN -d \
  python:3.11 bash -c 'set -e
    git clone --depth 1 https://github.com/WenHaoDing/angioflow.git /work/angioflow
    git -C /work/angioflow log --oneline -1
    pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -q numpy==2.3.5 scipy==1.16.3 huggingface_hub==1.20.1 torch_geometric==2.7.0 matplotlib==3.9.1
    pip install -q torch_scatter==2.1.2 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
    cd /work && exec python -m angioflow.training.train_joint_hf --run_name <run_name> <train_joint.py flags>'
```

- **`exec python`** makes Python the Job's main process, so `hf jobs cancel` and timeouts send SIGTERM to the trainer itself. The wrapper turns SIGTERM into a normal exit, which runs the final push. Without `exec`, the shell gets the signal and the last minutes of training are lost.
- **`git log --oneline -1`** prints the commit the Job runs. Check it in the log.
- **W&B:** add `wandb` to the pip line, `--secrets WANDB_API_KEY` to `hf jobs run`, and `--wandb` to the trainer flags. The W&B run id is saved in the run folder, so a resumed Job continues the same W&B run.

### Example: the all-frames joint run (2026-11-06)

Warm-started from the frame-0 model, trained on all 80 frames:

```
--run_name joint_v3data_allframes_k8_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs_b16_lr4e-4_wu300_p5_initft
--init_ckpt _init/joint_v3data_0frame_fs_b4_ft_p5_best.pth
--versions aneug_flow,angioflow,aneugv2 --frames all --train_frames_per_case 8
--test_frames 0,10,20,30,40,50,60,70 --exclude_cases C0064_cut2 --test_versions angioflow
--batch_size 16 --num_workers 20 --lr 4e-4 --lr_warmup_steps 300 --lr_plateau_patience 5 --total_epochs 100
<plus the frame-0 run's model and loss flags>
```

- **`--train_frames_per_case K`:** each epoch trains on K random frames of every case. The draw depends only on the epoch number, so a resumed run sees the same epochs. With K=8 an epoch is 13,656 samples, 854 steps at batch 16, about 1 h.
- **`--test_frames`:** fixed frames the test cases are scored at. The test loss is their mean. Frame 0 is also logged on its own (`test_f0_*` in `history.csv`), and every frame is in `test_frames.csv`.
- **`--lr_warmup_steps`:** warmup counted in optimizer steps, replacing `--lr_warmup_epochs`.
- **Peak LR:** chosen with [training/lr_probe_joint.py](training/lr_probe_joint.py), which runs 1,000 steps from the frame-0 checkpoint at several LRs and tracks frame-0 and other-frame test loss. At batch 4, 1e-4 and 3e-4 tied, and 3e-5 made frame 0 worst. Batch 16 then used 4e-4. This model's loss has risen at LRs above about 6e-4.
- **Batch size:** batch 24 used 88–95 GB of the 96 GB of GPU memory, so the run uses batch 16.

### Smoke test

Append `--max_steps 20` (or so) and use `--timeout 2h`. Each step prints its loss, time and `peak GPU … GB`. Nothing is pushed. To size the batch quickly, use `--frames 0` without `--train_frames_per_case` / `--test_frames`. Samples are the same size at every frame, and only frame 0 gets unpacked.

### Warm-starting from a checkpoint

Upload the checkpoint under `_init/` with a name that says what it is, then pass `--init_ckpt`:

```bash
hf upload whding123/AngioFlow-checkpoints <local run>/best.pth _init/<descriptive_name>.pth
```

Check the `[init] … (its epoch N, best=…)` line in the Job log. It tells you which checkpoint was actually loaded. An upload made while a local fine-tune is still running gets that run's best epoch *so far*.

### Resuming

Re-submit the **same command with the same `--run_name`**. The Job downloads `<run_name>/` from the Hub and continues from its `last.pth`. In the log, look for:

- `[hf] resuming: N file(s) … downloaded`
- `[resume] …/last.pth @ epoch E, best=…`

Model, optimizer, scheduler and history all come back. Work since the last push is lost, so at most one partial epoch.

The wrapper compares the stored `args.json` with the new flags and aborts if a setting differs. Settings that may change on resume are listed in `RESUME_FREE_KEYS` ([training/train_hf.py](training/train_hf.py)) and `JOINT_RESUME_FREE_KEYS` ([training/train_joint_hf.py](training/train_joint_hf.py)):

- `num_workers`, `total_epochs`, `ckpt_every`, `sanity_every`, `wandb`, `device`, paths;
- `loader_start_method` and `loader_timeout`.

To resume despite another difference, pass `--allow_config_change`. To start over instead, use a new `--run_name`.

`--lr_plateau_patience` and `--resume_lr` override the checkpoint's values on resume (`apply_resume_overrides` in `train_joint.py`).

### Getting results back

```bash
hf download whding123/AngioFlow-checkpoints --include "<run_name>/*" \
  --local-dir /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints
```

The run folder holds:

- `last.pth`, `best.pth` and `epoch_XXXX.pth` (every `--ckpt_every` epochs);
- `history.csv` and `test_frames.csv`;
- `args.json`;
- a copy of `train_joint.py`;
- `sanity/`, with the loss curves and sample renders.

## Downloading the data yourself

Locally, or from any new script:

```bash
cd "/media/yaplab2/HDD Storage/wenhao" && ~/miniconda3/envs/new/bin/python -m angioflow.huggingface.download_angioflow_pyg \
  --what wall interior_lean flow_split --datasets aneug_flow angioflow aneugv2 --dest <dir> [--frames 0] [--dry_run]
```

The download is resumable: each extracted archive is recorded with its Hub sha256, and only new or changed archives are fetched again. One zip at a time is downloaded, unpacked, verified against its manifest, then deleted. From Python, call `download_angioflow_pyg.main([...same argv...])`. The dataset is public, so no token is needed for it. The checkpoint repo is private and needs `HF_TOKEN`; on a Job, `--secrets HF_TOKEN` provides it.

## Writing a Job wrapper for a new trainer

Copy the pattern of `train_joint_hf.py`. A new wrapper should:

1. **Pass flags through to the trainer.** Parse the wrapper's own flags with `parse_known_args()`. Hand the rest to the trainer's `get_args()` by temporarily replacing `sys.argv`, so the Job accepts exactly the local trainer's flags.
2. **Override the paths:** set `save_dir`, `ckpt_root` etc. to `/tmp/...`, and set `extracted=True`.
3. **Download** with `download_angioflow_pyg.main(argv)`, passing `--frames` unless training on all frames.
4. **Fetch run inputs** with `fetch_meta` (in `train_hf.py`), and starting weights with `hf_hub_download(ckpt_repo, init_ckpt)`.
5. **Resume:** use `run_files_on_hub`, then `snapshot_download(allow_patterns=[f"{run}/*"])`, then `check_resume_config`.
6. **Push:** create a `HubPusher(api, repo, run, ckpt_dir, staging, every_min)` and give the trainer `epoch_end_hook=pusher.epoch_end`. The trainer's `main(args, epoch_end_hook=None)` must call the hook after each epoch's checkpoints are written. Call `pusher.final()` in a `finally:`.
7. **Handle SIGTERM:** `signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))`, so a timeout or cancel unwinds into that `finally:`.
8. **Push nothing in smoke tests:** no `HubPusher` when `--max_steps` is set.
9. **Start DataLoader workers with forkserver** (next section).

## Problems already hit

### The Job "hangs": GPU at 0%, no log output for hours (fixed 2026-11-06)

**Symptom.** The all-frames run finished epochs 0–6 (about 1 h each) and then logged nothing for 3 h with the GPU idle. `hf jobs cancel` still produced a final `[push] … through epoch 6`.

**Diagnosis.**
- **The main process was alive and running Python:** it handled SIGTERM and pushed. A GPU or driver hang blocks inside CUDA calls, and the signal handler would never have run.
- **No worker had died:** PyTorch raises `DataLoader worker … exited unexpectedly` within seconds when one does.

So the main process was waiting on a DataLoader worker that was alive but stuck.

**Cause.** DataLoader workers are created by `fork()` at the start of every epoch and every evaluation. `HubPusher` uploads in a background thread, started right after the epoch's checkpoints are written, which is exactly when the next epoch forks its workers. If another thread holds a lock at the moment of a fork (in malloc, OpenSSL, the hf_xet upload runtime, …), the child inherits the lock in its held state and blocks forever. It happens rarely and at random. Per-frame test loaders made it more likely: there were 8 of them, about 180 forks per epoch.

**Fix** (commit `7780b73`, in `train_joint.py`):

- **`--loader_start_method forkserver` (default).** Workers fork from a clean helper process, so threads in the training process no longer matter. Forkserver sends the dataset to each worker pickled, which needed two changes:
  - The transform chain is a class (`_Chain` in `train.py`), not a closure, because closures don't pickle.
  - The flow-split table holds plain floats. With one tensor per case, every tensor was sent as a file descriptor, and forkserver failed with `ValueError: too many fds`. **Keep tensors off dataset objects.**
- **`--loader_timeout 900` (seconds per batch).** If a worker ever does get stuck:
  - in training: `[warn] train: DataLoader timed out … epoch N ends early`, and the next epoch starts fresh workers;
  - in evaluation: one retry, then an error.

  A stall costs at most 15 min instead of the rest of the Job.
- **One test loader for all test frames.** A `FrameBatchSampler` keeps each batch to a single frame, and `evaluate(..., per_frame=...)` reports each frame from that one loader.
- **The WSS-only trainer:** `train.py` still forks per epoch and `train_hf.py` uses the same threaded pusher. Give it the same loader settings before long Jobs.

### RAM "near 256 GB" while downloading or training

`hf jobs stats` reports the container's memory **including the Linux file cache**. Downloading and unpacking about 260 GB pushes that cache up to the limit, and it stays there. The kernel hands the memory back whenever a process asks for it.

- **Not a problem:** RAM sitting flat at about 256.0 GB while the GPU stays busy.
- **A real problem:** the Job dies with `OOMKilled` / exit 137, or a `DataLoader worker … killed`.

What our processes actually use:
- **Each DataLoader worker** caches static geometry: about 1.3 GB at start plus about 7 MB per case it has loaded.
- **Workers restart every epoch**, so their caches reset.
- **20 workers at 8 frames per case** come to about 100 GB at the end of an epoch.

### GPU memory

`hf jobs stats` shows memory that PyTorch has *reserved*, which can be above what training actually needs. The random ROI crop makes graph sizes vary a lot (`--mask_high 0.25`), so the peak arrives late and at random. Keep the smoke test's `peak GPU` below about 85 GB on the 96 GB card.

### Others

- **Host-RAM leak in `k_hop_edges`** (fixed): see [AGENT_GUIDE.md](AGENT_GUIDE.md#known-pitfalls).
- **Shared memory:** a container `/dev/shm` under 1 GiB makes the wrappers set `--num_workers 0`, with a warning. On `rtx-pro-6000` it is large.
- **Stale stats:** after the training data changes, recompute `stats.pt` / `interior_stats_zeromax.pt` and re-upload them to `_meta/`. Otherwise Jobs normalise with the old ones.

## Time and the waveform (for inference at other frames)

- **The model's inputs:** `model(wall, interior, waveform)`.
- **Frame index:** `wall.t` and `interior.t` hold the frame index, 0–79.
- **Waveform:** the same 1,000-point inlet curve for every case. `training.train.load_waveform` builds it from the first of the 5 cycles in `waveform_yiying.txt`.
- **Getting another frame:** to predict frame `t`, keep the geometry and the waveform, and set both `t` to `t`.
- **Checkpoints:** frame-0 checkpoints only work at `t = 0`. For other frames, use an all-frames run.
- **Details:** `TimeEmbedder` and `WaveformEmbedder` in [models/transformer.py](models/transformer.py).
