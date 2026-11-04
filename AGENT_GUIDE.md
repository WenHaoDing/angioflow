# Agent guide: data, training and evaluation

This guide covers the joint model, which predicts WSS on the vessel wall together with velocity and pressure in the lumen. It explains where the data lives, how to download it, and which script runs each step. To rebuild the training data after CFD cases change, see [data/UPDATING_CFD_CASES.md](data/UPDATING_CFD_CASES.md). For dataset naming and history, see [DATASETS.md](DATASETS.md).

## Ground rules

- **Never launch Hugging Face GPU Jobs yourself.** Write the `hf jobs run ...` command and give it to the user. CPU Jobs need the user's go-ahead too.
- **Local GPUs:**
  - Never use GPU 2, which overheats.
  - Use the GPU the user names, and check it is free first (`nvidia-smi`). GPU 0 is often busy with other people's jobs.
- **Keep local tests short.** Use `--max_steps` smoke tests and few steps. Don't run full epochs to test something.
- **Commits:**
  - Commit when asked.
  - The user pushes. This shell has no GitHub credentials.
  - Jobs clone the **pushed** `master`, so code must be pushed before a Job can use it.

## Environment

- Run everything as a package from the **parent** folder:
  ```bash
  cd "/media/yaplab2/HDD Storage/wenhao" && ~/miniconda3/envs/new/bin/python -m angioflow.<module> ...
  ```
  The `new` conda env has torch 2.4 and PyG. The repo is imported as `angioflow`.
- **The `hf` CLI** lives in its own venv (`~/.hf-cli/venv`, version 2.4.0).
  - Version 1.8 rejects the `rtx-pro-6000` flavour, so don't downgrade.
  - Don't run Python scripts with that venv's interpreter: it has no numpy.

## Where things live

| What | Where |
|---|---|
| Source CFD data | Hub datasets `whding123/AngioFlow`, `whding123/AneuGv2`, `whding123/AneuG-Flow` (`cfd_data/<case>/`) |
| Extracted training data | Hub dataset `whding123/AngioFlow-PyG` (**public**) |
| Run inputs and checkpoints | Hub model repo `whding123/AngioFlow-checkpoints` (**private**). Run inputs are in `_meta/`, runs in `<run_name>/` |
| Local training root | `/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG` |
| Local checkpoints | `/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints/<run_name>/` for runs on the new data. Pass `--ckpt_root` to training and to `eval_joint`, because the code default is the older `angioflow_checkpoints/` |
| Inlet waveform | `/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/_meta/waveform_yiying.txt` (also `_meta/waveform_yiying.txt` on the checkpoint repo) |

The dataset tags in code are `aneug_flow`, `angioflow` and `aneugv2`. Old roots that still use `v1`/`v2` file names can't be read by the current code.

### What `AngioFlow-PyG` contains

| Path | Content |
|---|---|
| `wall/<tag>.zip` | Wall graphs: `static_<tag>_<case>.pt` (positions, normals, edges, node_type, geodesic distances, Laplacian eigenbasis) and `wss_<tag>_<case>_<t>.pt` for frames t = 0–79 |
| `interior_lean/<tag>/part_j00_NNNN.zip` (+ `.json`) | Lumen point clouds: `interior_static_<tag>_<case>.pt` (positions, wall_dist) and `interior_<tag>_<case>_<t>.pt` (velocity, pressure). About 150 cases per part |
| `flow_split/<tag>/flow_split.json` | Outlet flow split per case, `[q_outlet0, q_outlet1]`, in the registration's outlet order |

Case counts as of 2026-11-02: aneug_flow 634, angioflow 606, aneugv2 589, so 1,829 in total. Two AngioFlow cases are missing from the wall data: `B2o6VFmWVW_aneurysm1` and `BcXeCx3voF_aneurysm1`. They were re-exported after registration, and the wall extraction rejected them because their node counts no longer match. Their interior data exists but goes unused, because cases are discovered from the wall files.

## 1. Download the data (local)

The download is resumable. Each archive is checked against its Hub sha256, so rerunning fetches only new or changed archives.

```bash
cd "/media/yaplab2/HDD Storage/wenhao" && ~/miniconda3/envs/new/bin/python -m angioflow.huggingface.download_angioflow_pyg \
  --what wall interior_lean flow_split            # all datasets, all 80 frames (~264 GB on disk)
```

- `--datasets angioflow aneugv2` limits which datasets are fetched.
- `--frames 0` unzips frame 0 only. The zips are still downloaded whole, one at a time.
- `--dry_run` prints the plan and disk check, then stops.
- `--dest <dir>` overrides the default destination, the local root above.

The root then looks like `<root>/<tag>/{static_*,wss_*,flow_split.json}` and `<root>/interior/{interior_static_*,interior_*}`.

### Two files the download does not provide

The download does not include the normalisation stats. Put them in the root:

- `stats.pt` (WSS and positions)
- `interior_stats_zeromax.pt` (velocity and pressure, for `--press_zero_max`)

Either fetch the current versions from `_meta/` on the checkpoint repo, or let the first training run compute and save them. Each case contributes 8 random frames out of all 80, regardless of `--frames`, so one pair serves both frame-0 and all-frames runs. Computing them needs all frames on disk. The `_meta/` copies were computed from the full local root over all cases on 2026-11-02.

### Keep anomaly lists out of the root

Don't put `anomalies.json` or `cfd_anomalies.json` in the root. Training excludes every case listed in them by default (`--exclude_anomalies`), and the user has chosen to train on all cases. A colleague is checking every case by eye instead.

## 2. Train locally (frame 0)

The current config is three datasets, frame 0, the cross-field decoder, a crop of up to 0.25 and the flow-split input:

```bash
cd "/media/yaplab2/HDD Storage/wenhao" && ~/miniconda3/envs/new/bin/python -m angioflow.training.train_joint \
  --save_dir /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG --extracted --no_data_sanity \
  --ckpt_root /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints \
  --run_name joint_v3data_0frame_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs \
  --versions aneug_flow,angioflow,aneugv2 --frames 0 --exclude_cases C0064_cut2 \
  --add_flow_split --press_zero_max \
  --decimate_ratio 1.0 \
  --mask_low 0.0 --mask_high 0.25 \
  --dim_emb 108 --dim_field 96 \
  --n_pre 2 --n_gps 6 --n_pool 1 --pool_ratio 0.15 --pool_knn_k 8 --n_inter 1 \
  --geometry_token_pool latent --n_geometry_tokens 64 \
  --n_blocks 3 --xattn_heads 4 --local_hops 2 --local_heads 4 --k_local 16 --k_field 16 \
  --wall_query_ratio 0.5 --interior_sample_ratio 0.2 \
  --pointwise_loss huber --lambda_huber 5.0 --lambda_logmag 0.1 --lambda_smooth 0.1 \
  --lambda_dir 0.1 --lambda_vel 1.0 --lambda_press 1.0 \
  --grad_ckpt --lr 3e-4 --grad_clip_norm 3.0 \
  --total_epochs 300 --batch_size 4 --device cuda:1 --wandb
```

### Flags that matter

- `--extracted`: `--save_dir` is a downloaded AngioFlow-PyG root. The interior is read from `<save_dir>/interior/`.
- `--frames 0`: the default. Pass `all` for every frame, or a list such as `0,20,40`.
- `--exclude_cases`: cases to leave out of both train and test, comma-separated, as `<case>` or `<tag>/<case>`.
  - It is saved in `args.json`, so `eval_joint` drops the same cases.
  - A name that matches nothing aborts the run.
  - `C0064_cut2` (AngioFlow) is excluded by the user's decision of 2026-11-02.
- `--add_flow_split`: needs `<root>/<tag>/flow_split.json` for every dataset used. The run aborts if a case is missing from a table.
- `--mask_low` and `--mask_high`: each sample keeps the aneurysm (`node_type == 1`) plus a ring of parent vessel. The ring's geodesic depth is `p × max(geo_dist)` with `p ~ U[mask_low, mask_high]`, so a higher value means a bigger graph. The interior is filtered to match the kept wall.
- `--max_steps N`: smoke test. It prints per-step loss, time and peak GPU memory, runs N test batches, renders one sanity figure and saves no checkpoints.

### Memory and resuming

- **Memory:** batch size 4 peaked at 20.3 GB on a 24 GB card over 60 steps. If it runs out of memory, use `--batch_size 3`.
- **Resuming** is the default. A rerun with the same `--run_name` continues from `last.pth`.

### Model inputs and outputs

- **Inputs:**
  - wall positions, normals and mesh edges;
  - per-opening geodesic distance;
  - the Laplacian eigenvector encoding;
  - interior positions and wall_dist;
  - the frame index;
  - the shared inlet waveform;
  - the flow split.
- **Never fed to the model:** dataset name, case ID, `aneurysm_type`. `node_type` is used only for the crop.
- **Outputs:**
  - tangential WSS on 50% of the wall nodes;
  - velocity and pressure at the sampled interior points. Pressure is shifted so its per-case maximum is 0.
- **Split:** 80/20 by case within each dataset, seed 0.

### Aneurysm types

`aneurysm_type` is stored in every case's static as a scalar. It's never a model input.

| Type | Geometry | Outlets | Cases (AngioFlow / AneuG-Flow / AneuGv2) |
|---|---|---|---|
| 0 | Bifurcation aneurysm | 2 | 316 / 634 / 299 |
| 1 | Sidewall aneurysm | 1 (3 AngioFlow cases have 2) | 201 / 0 / 290 |
| 2 | Sidewall aneurysm with a bifurcation just downstream of the dome. The simulation kept both downstream branches, so it has two outlets. | 2 (2 cases have 1) | 89 / 0 / 0 |

### How much vessel the model sees around the aneurysm

**How the crop works:** the model never sees a whole training mesh. Each sample is cropped to the aneurysm's focused region (`node_type == 1`) plus a band of vessel around it. The band's depth is drawn at random for every sample, from no band at all up to 25% of the mesh's largest geodesic distance from the focused region (`--mask_low 0 --mask_high 0.25`).

**How it was measured:** [deploy/centerline_crop_stats.py](deploy/centerline_crop_stats.py) runs vmtk centerlines on all 606 AngioFlow CFD wall meshes, the real-patient set. It measures vessel length along the centerline, starting from the point next to the aneurysm (the centerline point nearest the dome apex):
- **upstream:** toward the inlet;
- **downstream:** out along each outlet branch.

Values are median [10th–90th percentile]. D is the vessel diameter at that end: the parent artery for upstream, the branch for downstream.

| Type | | Whole mesh | Cropped, smallest band | Cropped, typical | Cropped, largest band |
|---|---|---|---|---|---|
| 0 | upstream | 19.3 mm [13.2, 23.9], 6.7 D | 7.2 mm, 2.6 D | 8.2 mm [5.0, 11.9], 3.0 D | 9.5 mm, 3.5 D |
| 0 | downstream | 10.6 mm [7.7, 13.0], 5.6 D | 5.2 mm, 2.8 D | 6.2 mm [3.3, 9.0], 3.3 D | 7.6 mm, 4.0 D |
| 1 | upstream | 21.0 mm [12.8, 27.4], 5.5 D | 7.0 mm, 1.9 D | 8.2 mm [4.5, 14.1], 2.3 D | 10.3 mm, 2.8 D |
| 1 | downstream | 13.7 mm [10.5, 19.8], 4.4 D | 6.5 mm, 2.0 D | 7.6 mm [3.7, 12.0], 2.4 D | 9.6 mm, 2.9 D |
| 2 | upstream | 18.0 mm [13.8, 23.2], 5.0 D | 7.6 mm, 2.2 D | 8.8 mm [5.3, 13.4], 2.5 D | 10.2 mm, 2.8 D |
| 2 | downstream | 14.4 mm [10.3, 20.1], 6.0 D | 9.4 mm, 4.1 D | 10.6 mm [6.5, 17.1], 4.6 D | 11.8 mm, 5.2 D |

"Typical" pools every band depth training draws.

**Vessel sizes (median):**

| Type | Parent diameter | Branch diameter | Dome height |
|---|---|---|---|
| 0 | 2.6 mm | 1.9 mm | 4.9 mm |
| 1 | 3.9 mm | 3.3 mm | 6.0 mm |
| 2 | 3.5 mm | 2.3 mm | 6.7 mm |

**Clipping a new mesh for deployment** ([deploy/JOINT_VELOCITY_HANDOFF.md](deploy/JOINT_VELOCITY_HANDOFF.md)):
- Clip it to the **whole-mesh** extent for its type: about 5–7 parent diameters upstream and 4–6 branch diameters downstream.
- Don't clip it to the cropped extent. The deploy code applies the training crop itself, and the band's depth scales with the mesh's own size. So a mesh clipped like the training meshes ends up with a training-like crop.
- A much longer or much shorter mesh changes what the model sees.

**Outputs** are in `/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_centerlines/`:
- `centerline_crop_stats.txt`: the summary, with intermediate band depths;
- `centerline_crop_stats.json`: per-case and per-path numbers;
- `centerline/`: vmtk paths;
- `sanity/`: renders.

The extraction was stopped after 276 of the 634 AneuG-Flow cases, because only AngioFlow is needed. The script can resume it with `--datasets aneug_flow aneugv2`.

### Run folder

In `<ckpt_root>/<run_name>/`:

| File | Written |
|---|---|
| `last.pth` | every epoch |
| `best.pth` | when the test loss improves |
| `epoch_0000.pth`, `epoch_0100.pth`, … | every 100 epochs |
| `sanity/` | every 50 epochs |
| `history.csv`, `args.json` | throughout |

With `--wandb`, the run logs to the `AngioFlowJoint` W&B project.

## 3. Train on a Hugging Face Job

[training/train_joint_hf.py](training/train_joint_hf.py) wraps `train_joint.main` and takes the same flags. The Job:

1. downloads the data for `--versions` and `--frames` from AngioFlow-PyG;
2. fetches `stats.pt`, the interior stats and the waveform from `_meta/` on the checkpoint repo;
3. trains, pushing the run folder to `whding123/AngioFlow-checkpoints/<run_name>/` every 20 minutes (`--push_every_min`) and once more at the end, including on timeout;
4. resumes from the Hub copy when the same command is rerun.

`--save_dir`, `--interior_dir` and `--extracted` are set by the wrapper.

### Hardware

- **Flavour:** `rtx-pro-6000` (98 GB GPU, about $2.75/h).
- **Software:** it's a Blackwell card, so it needs **torch 2.7 + cu128** and PyG wheels built for it. torch 2.4/cu121 has no kernels for it.
- **Disk:** Job `/tmp` had 14 TB free, so all-frames training fits.

```bash
hf jobs run --flavor rtx-pro-6000 --timeout 24h --secrets HF_TOKEN -d python:3.11 bash -c 'set -e
  git clone --depth 1 https://github.com/WenHaoDing/angioflow.git /work/angioflow
  git -C /work/angioflow log --oneline -1
  pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -q numpy==2.3.5 scipy==1.16.3 huggingface_hub==1.20.1 torch_geometric==2.7.0 matplotlib==3.9.1
  pip install -q torch_scatter==2.1.2 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
  cd /work && exec python -m angioflow.training.train_joint_hf --run_name <run> <train_joint.py flags>'
```

### W&B on a Job

- Add `wandb` to the pip line and `--secrets WANDB_API_KEY` to the `hf jobs run` command.
- The W&B run id is stored in the run folder, so a resumed Job continues the same W&B run.

### Smoke tests and frames

- **`--max_steps N` never pushes.** Use it first, and for sizing the batch.
- **Frame-0 vs all-frames:** the frame-0 Job downloads about 233 GB of zips and keeps only frame 0. An all-frames Job (`--frames all`) keeps everything.
- **Epochs with all frames:** one epoch then holds about 80× as many samples, so lower `--total_epochs`.
- **`--frames all` with extracted data hasn't been smoke-tested yet.** Test it before a long run.

## 4. Evaluate

```bash
cd "/media/yaplab2/HDD Storage/wenhao" && ~/miniconda3/envs/new/bin/python -m angioflow.training.eval_joint \
  --ckpt_root /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints \
  --run_name <run_name> --which best --device cuda:1 --full_test_set --no_viz
```

- Everything comes from the checkpoint's saved args, including the same test split and crop.
- Outputs go to `<ckpt_dir>/eval/`: per-case `metrics.json`, plus `summary.json` and `summary.csv`.
- Drop `--no_viz` to also render the figures.

## 5. Other entry points

| Script | Use |
|---|---|
| `training/train.py`, `training/train_hf.py` | WSS-only model (wall graphs only), local and Job versions |
| `training/eval_wss.py`, `training/eval_interior.py`, `training/eval_leaderboard.py` | Single-task evaluation and the run leaderboard |
| `data/generate_flow_split.py` | Rebuild `flow_split.json` from a local root: `--root <root> --datasets angioflow aneugv2`. Then upload it to `flow_split/<tag>/flow_split.json` on AngioFlow-PyG |
| `huggingface/render_sanity_hf.py` | PyVista sanity renders (registration, geo_dist, WSS and velocity GIFs) as a uv Job, written to `sanity/<case>/` of the source dataset |
| [data/UPDATING_CFD_CASES.md](data/UPDATING_CFD_CASES.md) | Full pipeline from re-exported CFD cases to new training data: registration, wall statics, extraction Jobs, assembly, download, flow split |
| [deploy/JOINT_VELOCITY_HANDOFF.md](deploy/JOINT_VELOCITY_HANDOFF.md) | Run the joint model on a new vessel surface (e.g. from a segmentation) to get velocity: `deploy/joint_velocity.py`. The other `deploy/` scripts are for an older model and data format. |

### Preparing a new surface (deployment)

`deploy/joint_velocity.prepare_surface` turns an MRI or segmentation surface into the model's input in three steps:

1. **Smooth** (on by default) with AneuG's mesh regularizer. This is the **dome-normalized** variant, `AneuG/mesh_regularizer/dome_normalized.py`, with the reference model `reference_models/reference_roughness__domenorm6.15__t0.3-…-1.65__g2.npz`; that's the model to use, not the older mm-based ones.
   - **Needs:** the sac size (`sac_mm`, or `sac_points` to measure it from) and the AneuG repo importable as `AneuG`.
   - **Time:** 1–4 min per case. `smooth_time_budget_s` caps it; `None` runs to convergence, which is fine for offline batches.
   - **Turn off** with `smooth=False` for meshes that are already CFD-smooth.
2. **Remesh** to the 0.12 mm training edge with pymeshlab.
3. **Compute the wall features** with the training pipeline's own functions.

Clip the surface first, using the centerline statistics under "How much vessel the model sees around the aneurysm". Details and checks are in [deploy/JOINT_VELOCITY_HANDOFF.md](deploy/JOINT_VELOCITY_HANDOFF.md).

## Known pitfalls

- **Local GPU 1 has faulty VRAM; don't train on it.**
  - **Symptom:** big runs hit a deterministic `indexSelectLargeIndex ... srcIndex < srcSelectDimSize` assert in the decoder's GATv2 gather, even though the inputs were valid. Replaying the batch in a fresh process was clean, because the replay never allocated far enough into memory to reach the bad block.
  - **Cause:** hardware. A VRAM pattern test found bit errors about 12.5 GB into GPU 1's memory.
  - **Wrong turn:** commit `18d1601` blamed cuSPARSE and moved the `k_hop_edges` hop products to the CPU.
  - **If that assert comes back:** run a VRAM pattern test on that GPU before blaming the code or data.
- **Host-RAM leak in `k_hop_edges`, fixed.**
  - **Symptom:** joint runs were OOM-killed after a few hours, both locally and on HF (exit 137, `OOMKilled`). The main process grew by about 14 GB/h.
  - **Cause:** the CPU sparse-CSR matrix multiply (`EdgeIndex.matmul`) that `18d1601` introduced leaks host RAM, about 7.5 MB per call.
  - **Fix:** `k_hop_edges` now uses plain index operations on the graph's own device. It is edge-for-edge identical to the old version, takes about 10 ms per call on the GPU, and doesn't leak.
- **Stale extraction outputs.** The extraction Jobs skip cases already in uploaded parts, so a re-exported case keeps its old output. See "Things to know" in `UPDATING_CFD_CASES.md`.
- **Old tables after re-registration.** After a registration changes, `flow_split.json` must be regenerated. An old table silently gives the wrong outlet order for cases whose outlets were relabelled.
- **Stats after data changes.** The stats in `_meta/` must be recomputed and re-uploaded whenever the training data changes.
