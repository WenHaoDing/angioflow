# Joint frame-0 model: warm-started run on an HF Job

This starts a **new run** on an RTX PRO 6000 Job. Its weights are initialised from the local run's `best.pth`, and it has a fresh optimizer, scheduler and warmup. The local run is
`joint_v3data_0frame_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs` on GPU 3.

| | Local run | Warm-started Job |
|---|---|---|
| Batch size | 4 | **16** |
| Learning rate | 3e-4 | **6e-4**, after a 5-epoch linear warmup from the loaded weights |
| Plateau patience | 15 | **20** (factor 0.5, floor 6e-6) |
| Test set | 80/20 of every dataset (366 cases) | **AngioFlow only**: the same 121 cases as the local run |
| Training set | 1,462 cases | **1,707**: all aneug_flow (634) and aneugv2 (589), plus 484 AngioFlow |

AneuG-Flow and AneuGv2 are synthetic, so all their cases train. `--test_versions angioflow` keeps AngioFlow's test cases identical to the local run's: every dataset is still shuffled in the same order. None of them was a training case of the local run, so the warm start doesn't leak test data. This was checked on 2 Nov with 121 identical cases and 0 overlap.

Everything else is unchanged: the data, model, crop and losses, and `C0064_cut2` stays excluded.

**The test loss now covers AngioFlow only.** It isn't comparable with the local run's test loss, which also included synthetic cases. `best.pth` is now chosen on AngioFlow.

## 0. Push the code (before launching)

These scripts change: `train.py`, `train_joint.py`, `train_joint_hf.py` and `eval_joint.py`.
- `--test_versions` and `--init_from` are new.
- `train_joint_hf.py` gains `--init_ckpt`.
- `--resume_lr` and the plateau patience/factor override on resume are included too.

```bash
cd "/media/yaplab2/HDD Storage/wenhao/angioflow" && git add training/train.py training/train_joint.py \
  training/train_joint_hf.py training/eval_joint.py training/WARMSTART_JOINT_ON_HF.md \
  && git commit -m "joint: --test_versions, --init_from/--init_ckpt warm start, --resume_lr" \
  && git push origin master
```

## 1. Upload the local best checkpoint

`best.pth` keeps improving, so upload it when you stop the local run, after an epoch line has printed. The local run can also simply keep going; the Job doesn't need it stopped.

```bash
hf upload whding123/AngioFlow-checkpoints \
  /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints/joint_v3data_0frame_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs/best.pth \
  _init/joint_v3data_0frame_fs_b4_best.pth
```

## 2. Launch the Job

```bash
export WANDB_API_KEY=<your key>     # --secrets WANDB_API_KEY reads it from the shell
```

```bash
hf jobs run --flavor rtx-pro-6000 --timeout 24h --secrets HF_TOKEN --secrets WANDB_API_KEY -d python:3.11 bash -c 'set -e
  git clone --depth 1 https://github.com/WenHaoDing/angioflow.git /work/angioflow
  git -C /work/angioflow log --oneline -1
  pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  pip install -q numpy==2.3.5 scipy==1.16.3 huggingface_hub==1.20.1 torch_geometric==2.7.0 matplotlib==3.9.1 wandb==0.23.1
  pip install -q torch_scatter==2.1.2 torch_cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
  cd /work && exec python -m angioflow.training.train_joint_hf \
    --run_name joint_v3data_0frame_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs_warm_b16_lr6e-4_p20_angiotest \
    --init_ckpt _init/joint_v3data_0frame_fs_b4_best.pth \
    --versions aneug_flow,angioflow,aneugv2 --frames 0 --exclude_cases C0064_cut2 --test_versions angioflow \
    --add_flow_split --press_zero_max --decimate_ratio 1.0 --mask_low 0.0 --mask_high 0.25 \
    --dim_emb 108 --dim_field 96 --n_pre 2 --n_gps 6 --n_pool 1 --pool_ratio 0.15 --pool_knn_k 8 --n_inter 1 \
    --geometry_token_pool latent --n_geometry_tokens 64 \
    --n_blocks 3 --xattn_heads 4 --local_hops 2 --local_heads 4 --k_local 16 --k_field 16 \
    --wall_query_ratio 0.5 --interior_sample_ratio 0.2 \
    --pointwise_loss huber --lambda_huber 5.0 --lambda_logmag 0.1 --lambda_smooth 0.1 \
    --lambda_dir 0.1 --lambda_vel 1.0 --lambda_press 1.0 \
    --grad_ckpt --grad_clip_norm 3.0 --num_workers 8 \
    --batch_size 16 --lr 6e-4 --lr_plateau_patience 20 --total_epochs 300 \
    --wandb'
```

This starts a new W&B run. Its id is saved in the run folder, so resumed Jobs continue it.

## 3. What the Job log should show

After about 10 minutes of download and dataset build, the log should show:

1. The pushed commit on the first line.
2. `[hf] init weights <- whding123/AngioFlow-checkpoints/_init/joint_v3data_0frame_fs_b4_best.pth`.
3. `[split] aneug_flow: 634 train  0 test`, `aneugv2: 589 train  0 test`, `angioflow: 484 train  121 test`, then `[data] 1828 samples  train=1707  test=121`.
4. `[init] model weights from .../joint_v3data_0frame_fs_b4_best.pth (its epoch N, best=...)`.
5. `epoch   0` with `lr 1.20e-04`, rising to `6.00e-04` at epoch 4. The losses should start near the local model's level, not from scratch.

**If something looks wrong:**
- **Loss climbs or clipping spikes:** if, once warmup reaches 6e-4, the loss climbs or the `clipped` percentage spikes, cancel the Job (`hf jobs cancel <id>`) and relaunch with a lower `--lr`, for example 4e-4.
- **Relaunching after a cancel:** use a new `--run_name`, or delete the run's folder on the Hub. Otherwise the relaunch resumes the cancelled run instead of starting fresh.

## 4. Time, cost, timeout

Each epoch is about 6.5 min (1,707 training cases at ~4.8 cases/s, plus the AngioFlow test). So 300 epochs take about 32 h, roughly $90. That's more than one 24 h Job, so expect one rerun.

The run is pushed to the Hub every 20 min and on timeout. **Rerun the exact same command** to continue. It resumes from the run's own `last.pth`, and `--init_ckpt` / `--init_from` are then ignored, so the weights are not re-initialised.

## 5. Fetch and evaluate

```bash
hf download whding123/AngioFlow-checkpoints \
  --include "joint_v3data_0frame_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs_warm_b16_lr6e-4_p20_angiotest/*" \
  --local-dir /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints
```

`eval_joint` reads `--test_versions` from the checkpoint's args, so it evaluates on the same 121 AngioFlow test cases. Pass `--ckpt_root /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints`.
