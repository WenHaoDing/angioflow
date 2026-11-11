# AngioFlow: project setup and resuming training on a new workstation

## What this project does

AngioFlow is a deep-learning surrogate for CFD in intracranial aneurysms. Given a vessel geometry (a wall surface mesh around an aneurysm) and the inlet waveform, one network predicts, at any of the 80 frames of a cardiac cycle:

- **wall shear stress (WSS)** on the vessel wall;
- **velocity and pressure** at points inside the lumen.

The model is the **joint model** ([training/train_joint.py](training/train_joint.py)):

- a hierarchical graph-transformer encoder (HierGPS) reads the wall mesh;
- query decoders predict WSS at wall nodes and velocity/pressure at lumen points, with attention across the two fields;
- time enters as the frame index (0–79) and a shared inlet waveform (`waveform_yiying.txt`: 5 cycles, training uses the first);
- the outlet flow split is an extra input (`--add_flow_split`).

It is trained on three CFD datasets, 1,829 cases in total:

| Dataset | Tag | What |
|---|---|---|
| AneuG-Flow | `aneug_flow` | old synthetic shapes |
| AngioFlow | `angioflow` | real patient shapes; holds the test split |
| AneuGv2 | `aneugv2` | new synthetic shapes |

[AneuG](https://github.com/WenHaoDing/AneuG) is the sibling repository that generates the synthetic aneurysm shapes and holds the mesh tools (registration, mesh regulariser). **Training does not import it.** You need it for deployment on new surfaces ([deploy/](deploy/)) and for rebuilding the training data.

More detail: [AGENT_GUIDE.md](AGENT_GUIDE.md) (data, model inputs, local training, evaluation), [HF_JOBS.md](HF_JOBS.md) (Hugging Face Jobs, known problems), [DATASETS.md](DATASETS.md).

## Where things live

| What | Where |
|---|---|
| Code | GitHub `WenHaoDing/angioflow` (this repo) and `WenHaoDing/AneuG` |
| Training data | Hub dataset `whding123/AngioFlow-PyG` (public). Downloaded by the training wrapper |
| Normalisation stats and waveform | `_meta/` of the Hub model repo `whding123/AngioFlow-checkpoints` (**private**). Fetched by the wrapper |
| Warm-start weights | `_init/` of the same repo |
| Runs (checkpoints, `history.csv`, `args.json`) | `whding123/AngioFlow-checkpoints/<run_name>/` |

The Hub run folder is the source of truth. A run can be continued on any machine that can read and write that repo.

## 1. Clone

The code is imported as the package `angioflow`, so clone both repos into the **same parent folder** and always run Python from that parent folder.

```bash
mkdir -p ~/work && cd ~/work                      # any parent folder; it must be on sys.path
git clone https://github.com/WenHaoDing/angioflow.git
git clone -b vessel-conditioned https://github.com/WenHaoDing/AneuG.git   # only for deploy / data rebuilds
```

- **AneuG branch:** `vessel-conditioned` (bad2a62 or later) or `v2`. Deployment smoothing needs `mesh_regularizer/dome_normalized.py`, and both branches have it. It is imported as a top-level `AneuG` package, which is why it sits next to `angioflow`.
- **Private repos:** if a clone asks for credentials, run `gh auth login` first or use the SSH URL (`git@github.com:WenHaoDing/<repo>.git`).

## 2. Environment

This pinned set is what the Hugging Face Jobs run on (tested on an RTX PRO 6000):

```bash
conda create -n angioflow python=3.11 -y && conda activate angioflow
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.3.5 scipy==1.16.3 huggingface_hub==1.20.1 torch_geometric==2.7.0 matplotlib==3.9.1
pip install torch_scatter==2.1.2 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install wandb          # optional, only with --wandb
```

- **CUDA:** torch 2.7 + cu128 needs a recent NVIDIA driver (the 570 series or newer). Blackwell cards (RTX PRO 6000, RTX 50xx, B200) need it.
- **Older driver, non-Blackwell card:** torch 2.4.0 also works. Use the matching PyG wheels from `https://data.pyg.org/whl/torch-2.4.0+cu121.html` (or `+cu118`). The yaplab2 machine runs torch 2.4.0 with cu118 wheels.
- **Deployment or data rebuilds** also need pytorch3d, trimesh and pyvista. See `AneuG/environment.yml`.

Check that it works:

```bash
cd ~/work && python -c "import torch, torch_geometric, torch_scatter, torch_cluster, angioflow.training.train_joint; print(torch.__version__, torch.cuda.is_available())"
```

### Hub login

The checkpoint repo is private, and resuming writes to it. Log in with a token that has write access to `whding123/AngioFlow-checkpoints`:

```bash
hf auth login           # or: export HF_TOKEN=hf_...
hf auth whoami          # should print whding123
```

## 3. Hardware and disk

- **GPU memory:**
  - batch 4 peaks at about 20 GB;
  - batch 24 used 88–95 GB on a 96 GB card;
  - so plan on **at least 80 GB for batch 16**, and measure it with the smoke test below.
- **Smaller GPU:** see "Changing the batch size" at the end.
- **Disk:**
  - all three datasets at all 80 frames take about **120 GB** once unpacked;
  - the download fetches about 262 GB of zips, one at a time, each deleted after unpacking;
  - leave about 200 GB free under `--work_dir`.
- **CPU and RAM:** the run used 20 DataLoader workers on 23 vCPUs and 256 GB RAM. Set `--num_workers` to roughly your CPU core count minus 2, and lower it if RAM runs short.

## 4. Current state of training (2026-11-11)

**The run to continue** is the all-frames joint model, warm-started from the frame-0 model:

```
joint_v3data_allframes_k8_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs_b16_lr4e-4_wu300_p5_initft
```

- **Where it stopped:** on a Hugging Face Job, with the last push "through epoch 22" (2026-11-07). The best test loss is 0.5957 (epoch 21); frame 0 alone scores 0.156.
- **LR:** the plateau scheduler had halved it from 4e-4 to 2e-4 at epoch 19. The scheduler state is in `last.pth`, so this carries over.
- **Length:** each epoch trains on 8 random frames of every case (13,656 samples, 854 steps at batch 16) and tests the AngioFlow test cases at frames 0, 10, …, 70. The run ends after 100 epochs.
- **No Job is running it now.** Never run the same `run_name` on two machines at once: both would push to the same Hub folder.

**The frame-0 fine-tune** (`joint_v3data_0frame_…_angiotest`) is still running locally on yaplab2 (tmux `joint_ft_p5`, near its final epoch 349). Its checkpoints are only on yaplab2's disk. To continue it elsewhere, see "Moving a local-only run" at the end.

## 5. Resume (or start) training

Training runs through [training/train_joint_hf.py](training/train_joint_hf.py), which wraps `train_joint.py` and takes the same flags. It works the same on a workstation as on a Hugging Face Job:

1. downloads the data into `<work_dir>/data`;
2. fetches the stats and the waveform from `_meta/`;
3. if `<run_name>/` exists on the Hub, downloads it and **resumes from its `last.pth`** (model, optimizer, scheduler, history). Otherwise it starts a new run, using `--init_ckpt` weights if given;
4. trains, pushing the run folder back to the Hub every 20 minutes and once more on exit (also on Ctrl-C or `kill`).

So **the same command starts a run and resumes it.** Before resuming, it compares the flags with the run's `args.json` and aborts if a training setting differs. Paths, `--device`, `--num_workers`, `--total_epochs`, `--work_dir` and the init checkpoint's local path may differ.

Save this as `~/work/run_allframes.sh` and edit the three lines at the top:

```bash
#!/bin/bash
# Resume (or start) the all-frames joint run. Rerun the same script to resume after any stop.
set -e
WORK_DIR=/data/angioflow_work      # needs ~200 GB free; data/, ckpt/, push/ go here
DEVICE=cuda:0
WORKERS=20                         # ~ CPU cores - 2; lower if RAM runs short

RUN=joint_v3data_allframes_k8_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs_b16_lr4e-4_wu300_p5_initft
cd ~/work                          # the folder that holds angioflow/
exec python -m angioflow.training.train_joint_hf \
  --work_dir "$WORK_DIR" --device "$DEVICE" --num_workers "$WORKERS" \
  --run_name "$RUN" --init_ckpt _init/joint_v3data_0frame_fs_b4_ft_p5_best.pth \
  --versions aneug_flow,angioflow,aneugv2 --frames all --train_frames_per_case 8 \
  --test_frames 0,10,20,30,40,50,60,70 --test_versions angioflow --exclude_cases C0064_cut2 \
  --add_flow_split --press_zero_max --decimate_ratio 1.0 --mask_low 0.0 --mask_high 0.25 \
  --dim_emb 108 --dim_field 96 --n_pre 2 --n_gps 6 --n_pool 1 --pool_ratio 0.15 --pool_knn_k 8 --n_inter 1 \
  --geometry_token_pool latent --n_geometry_tokens 64 \
  --n_blocks 3 --xattn_heads 4 --local_hops 2 --local_heads 4 --k_local 16 --k_field 16 \
  --wall_query_ratio 0.5 --interior_sample_ratio 0.2 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_logmag 0.1 --lambda_smooth 0.1 \
  --lambda_dir 0.1 --lambda_vel 1.0 --lambda_press 1.0 \
  --grad_ckpt --lr 4e-4 --lr_warmup_steps 300 --lr_plateau_patience 5 --grad_clip_norm 3.0 \
  --total_epochs 100 --batch_size 16 --sanity_every 5 --ckpt_every 10 "$@"
```

These flags reproduce the run's `args.json` exactly: checked on 2026-11-11 with the wrapper's own config check, for both the default and a different `--work_dir`. `--init_ckpt` only matters for a new run; on resume it is downloaded but not loaded.

### Step 1: smoke test (memory check, nothing pushed)

```bash
bash ~/work/run_allframes.sh --run_name smoke_allframes_b16 --max_steps 20
```

- **What it does:**
  - downloads the full data on the first run, which takes the longest; later runs skip archives already unpacked;
  - starts a throwaway run from the init weights and trains 20 steps.
- **What to check:**
  - each step prints its loss, time and `peak GPU … GB`;
  - the run ends with `[smoke] … peak GPU X GB; no checkpoints saved`.
- **Afterwards:** delete `<WORK_DIR>/ckpt/smoke_allframes_b16`.

It uses a different `--run_name` on purpose, so it cannot touch the real run's local files.

### Step 2: resume

Run it in tmux, so it survives a closed terminal:

```bash
tmux new -s allframes
bash ~/work/run_allframes.sh 2>&1 | tee -a ~/work/allframes.log
# detach: Ctrl-b d      reattach: tmux attach -t allframes
```

In the first minutes of the log, look for:

- `[hf] resuming: N file(s) of whding123/AngioFlow-checkpoints/<run> downloaded`
- `[resume] …/last.pth @ epoch 23, best=0.5957`

If you see `[init] model weights from …` instead, the run was **not** found on the Hub and a new run started. Check the run name and the Hub login.

**To stop:** press Ctrl-C in the tmux window, or `kill <pid>`. The wrapper pushes once more before exiting. Only work since the last push is lost, so at most part of one epoch. **To continue:** run the same command again.

### Starting a new run instead

Use the same script with a new `--run_name` (the last one given wins). Change any training flags you like. With `--init_ckpt _init/<name>.pth` it starts from those weights; without it, from scratch. To make a checkpoint available as a warm start:

```bash
hf upload whding123/AngioFlow-checkpoints <run folder>/best.pth _init/<descriptive_name>.pth
```

The same wrapper also runs on a Hugging Face Job: see [HF_JOBS.md](HF_JOBS.md) for the `hf jobs run` command. Launching Jobs is the user's call.

## 6. Monitoring and results

| What | Where |
|---|---|
| Per-epoch metrics | `<WORK_DIR>/ckpt/<run>/history.csv`. `test_loss` is the mean over the test frames; `test_f0_*` is frame 0 alone |
| Per-frame test metrics | `test_frames.csv` |
| Loss curves, sample renders | `sanity/` (every 5 epochs) |
| Checkpoints | `last.pth` (every epoch), `best.pth` (best test loss), `epoch_XXXX.pth` (every 10 epochs) |

Everything is pushed to the Hub. From any machine:

```bash
hf download whding123/AngioFlow-checkpoints --include "<run>/*" --local-dir <dir>
```

To evaluate a checkpoint, see section 4 of [AGENT_GUIDE.md](AGENT_GUIDE.md) (`angioflow.training.eval_joint`).

## 7. Things that go wrong

- **`ModuleNotFoundError: angioflow`:** Python is not running from the parent folder. `cd ~/work` first, or `export PYTHONPATH=~/work`.
- **`[abort] … was trained with other settings`:** a flag differs from the run's `args.json`. The message lists which. Fix the flag; don't pass `--allow_config_change` unless you mean to change the run.
- **`401` / `403` / repo not found:** the Hub token can't read or write `whding123/AngioFlow-checkpoints`.
- **The run stalls with the GPU at 0%:** a DataLoader worker hung. This was fixed on 2026-11-06 (workers start via forkserver; a 900 s batch timeout ends a stuck epoch early with a warning). See [HF_JOBS.md](HF_JOBS.md#the-job-hangs-gpu-at-0-no-log-output-for-hours-fixed-2026-11-06).
- **RAM looks full:** most of it is the OS page cache from reading the data files, not a leak. Real pressure shows as workers being killed. Lower `--num_workers` then.
- **`/dev/shm < 1 GiB` warning** (in Docker): the wrapper falls back to `num_workers 0`, which is very slow. Start the container with `--shm-size=16g` or more.
- **Out of GPU memory:** see the next section.

### Changing the batch size

`--batch_size` is a training setting, so the resume aborts if you change it. If batch 16 does not fit:

- **Better:** use a GPU with enough memory, or run on a Hugging Face Job (`rtx-pro-6000`).
- **Otherwise:** resume with `--batch_size 8 --allow_config_change --resume_lr 1e-4` (the LR halved with the batch). The learning dynamics change from that point, so note it when reporting results.

### Moving a local-only run

A run trained without the wrapper (plain `train_joint.py`, like the frame-0 fine-tune on yaplab2) exists only on that machine. To continue it elsewhere:

1. upload its folder: `hf upload whding123/AngioFlow-checkpoints <ckpt_root>/<run> <run>`;
2. on the new machine, run the wrapper with the **same flags** as the original run (they are in its `args.json`) and the same `--run_name`.

The wrapper then resumes it like any Hub run.
