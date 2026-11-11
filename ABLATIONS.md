# Ablation design for the AngioFlow paper

This document proposes the experiments for the paper's ablation section, from a literature review done on 2026-11-10. It has two tables:

- **Table 1 (comparison with other methods):** our model against general ML backbones and against published hemodynamics surrogates, under one protocol.
- **Table 2 (our own design choices):**
  - predicting WSS only vs WSS + flow;
  - predicting the whole cardiac cycle directly vs predicting frame 0 and then stepping through the cycle with Hachem's group's models (MeshGraphNet-style rollout);
  - with and without a steady-state CFD prior as input, for both WSS-only and WSS + flow.

An optional Table 3 (training data and components) is sketched in section 6. Section 7 lists what exists in the repo and what still has to be built, and section 8 gives the compute plan.

## 1. What is being tested

Our model maps **geometry to flow**:
- **Input:** the cropped wall surface mesh around the aneurysm and a lumen point cloud. The shared inlet waveform is used, plus the frame index t for whole-cycle runs.
- **Encoder:** one HierGPS geometry encoder (graph network + GPS transformer over pooled tokens).
- **Decoders:** two query decoders: WSS on wall nodes, velocity and pressure on lumen points. They are coupled by cross-field attention.
- **Time:** handled **directly**. The frame index and waveform features condition every query, so any frame is one forward pass with no time stepping (see `HF_JOBS.md`, "Time and the waveform").
- **Training data:**
  - two synthetic datasets: AneuG-Flow (634 cases) and AneuGv2 (589);
  - the real-patient AngioFlow dataset (605 cases, 121 of them held out for test).

The paper's claims that the ablations must support:

1. The architecture beats standard geometric-learning backbones and published hemodynamics surrogates on the same data (Table 1).
2. Predicting lumen flow together with WSS helps, or at least doesn't hurt, WSS (Table 2a).
3. Direct whole-cycle prediction is better and faster than predicting one frame and stepping through the cycle (Table 2b).
4. How much a cheap steady CFD solution adds as prior information, against what it costs in time (Table 2c).

## 2. Literature review

**Method.**
- **Tool:** the `ml-lit-search` skill, which queried OpenAlex and arXiv (Semantic Scholar was rate-limited from this network).
- **Searches:** 19 queries in three groups: vascular/aneurysm surrogates, ML backbones for mesh/geometry fields, and time handling. Papers from 2018 on, about 560 unique papers in total.
- **Screening:** by abstract.
- **Extra reading:** Hachem's group's paper (arXiv 2512.09013), ICML paper (2605.01542) and code (`graph-physics`) were read in full.

Nothing below comes from memory alone.

### 2.1 Closest works: learned surrogates for vascular flow

| Paper | What it does | Role in our ablations |
|---|---|---|
| [Garnier, Jeken Rico, Lannelongue, … (Hachem group), *Graph Deep Learning for Intracranial Aneurysm Blood Flow Simulation and Risk Assessment*, arXiv 2025](https://arxiv.org/abs/2512.09013) | **Autoregressive** masked graph transformer (51M parameters) on **tetrahedral volume meshes** (~250k nodes), Δt = 10 ms over a ~0.8 s cycle (~80 steps, the same time grid as ours). Predicts velocity; WSS/TAWSS/OSI are derived from it. Trained on AnXplore (101 ICA aneurysms) and fine-tuned on 13 patient geometries with several waveforms. <1 min per cycle on an A100. They tried Transolver, which underperformed. | **Main domain baseline** (Table 1b), and the published form of the "step through the cycle" approach of Table 2b |
| [Lannelongue, Garnier, Jeken Rico, Goetz, *Physics constrained GNN for real time prediction of intracranial aneurysm hemodynamics*, npj Digital Medicine 2026](https://doi.org/10.1038/s41746-026-02404-z) | Physics-constrained GNN predicting full 3D, time-resolved fields through the cycle; generalises to new inflow conditions and geometries. Releases a 105-geometry benchmark. | Cite; its benchmark is the candidate for the external check |
| [Garnier, Lannelongue, Hachem, *Mesh Based Simulations with Spatial and Temporal awareness*, ICML 2026](https://arxiv.org/abs/2605.01542) | Shows that plain autoregressive rollout is a forward-Euler scheme and replaces it with a predictor–corrector "Temporal Correction" (cross-attention). Tested with MeshGraphNet, Transolver and a Transformer, on an aneurysm dataset among others. | Use their Temporal Correction as the **strongest** rollout variant in Table 2b, so the step-through-the-cycle approach is not a strawman |
| [Garnier, Lannelongue, Viquerat, Hachem, *Training transformers for mesh-based simulations*, Neural Comput. Appl. 2026](https://arxiv.org/abs/2508.18051) | Graph transformer with the adjacency matrix as attention mask; beats MeshGraphNet on 3D CFD. Code: [graph-physics](https://github.com/DonsetPG/graph-physics), which has MeshGraphNet, Transformer and Transolver, noise injection, and aneurysm data loaders (PyTorch + PyG + Lightning; also needs DGL). | Reference implementation for the MeshGraphNet and Hachem-style rows |
| [Bai, Zhou, Miyata, Shih, *STEP: … WSS throughout the cardiac cycle in MCA bifurcation aneurysms*, bioRxiv 2026](https://doi.org/10.64898/2026.10.12.758700) | **Trained on AneuG-Flow, one of our datasets.** LaB-GATr backbone pretrained on 11,656 steady cases, then fine-tuned on 449 transient cases (72 val / 57 test). 80 time points with waveform conditioning. Whole-cycle WSS vector MAE 1.688 Pa; TAWSS MAE 1.04 Pa; OSI MAE 0.0063; 4.2 s for 80 frames. They note a shared inlet waveform and train/test geometry overlap as limits. | Direct competitor on overlapping data. Its backbone becomes the LaB-GATr row of Table 1. Its numbers are **not** comparable with ours (different split and metric); see section 9 |
| [Sheng, **Ding**, Roi, Yeo, … *Real-Time Pulsatile Flow Prediction … using a Graph Transformer and Steady-Flow Data Augmentation*, arXiv 2026](https://arxiv.org/abs/2601.19876) | **Our group's previous paper.** GPS Graph Transformer with time and waveform conditioning; full-cycle WSS on aneurysm surface meshes; steady-CFD augmentation. Its model is `AngioGraphGPS` in `models/transformer.py`. | **"Our previous model" row** in Table 1b. It shows what the new architecture and flow prediction add |
| [Suk, Imre, Wolterink, *LaB-GATr: Geometric Algebra Transformers for Large Biomedical Surface and Volume Meshes*, MICCAI 2024](https://arxiv.org/abs/2403.07536) | E(3)-equivariant transformer with geometric tokenisation; state of the art on cardiovascular hemodynamics tasks; meshes up to 200k vertices; public code. | Domain-specific backbone (Table 1a) |
| [Rygiel, Suk, Yeung, Brüne, *When Does Equivariance Help? Canonical Alignment in Neural Fluid Surrogates*, 2026](https://arxiv.org/abs/2605.18816) | AB-GATr, an equivariant surrogate for **coupled surface + volume** fields. Across three hemodynamic benchmarks, equivariance beats rotation augmentation, whose benefit is inconsistent. | Closest in spirit to our joint WSS + flow output. Also motivates the rotation-robustness check in section 6 (we use rotation augmentation) |
| [Suk et al., *Mesh neural networks for SE(3)-equivariant hemodynamics estimation on the artery wall*, Comput. Biol. Med. 2024](https://arxiv.org/abs/2212.05023) | Equivariant mesh convolution for WSS on synthetic coronary arteries (7.6% approximation error). | Cite; covered by LaB-GATr |
| [Suk et al., *Physics-informed GNNs for flow field estimation in carotid arteries*, Med. Image Anal. 2026](https://arxiv.org/abs/2408.07110) | PointNet++ with steerable layers and physics priors, trained on 4D-flow MRI. | Cite (lumen velocity from geometry) |
| [Suk et al., *Deep vectorised operators for pulsatile hemodynamics … from a steady-state prior*, CMPB 2025](https://arxiv.org/abs/2410.11920) | Pulsatile velocity and pressure in 74 stenotic coronaries, conditioned on a steady solution. | **Closest published analogue of our steady-prior variant** (Table 2c): steady CFD given as input at inference |
| [Li, Liu, Yang, Wang, *A deep learning-based rapid strategy for computing WSS in intracranial aneurysms*, Phys. Fluids 2025](https://doi.org/10.1063/5.0267607) | Dual-path PointNet for WSS on 2,000 idealised aneurysms. | Represented by the PointNet++ row |
| [Li et al., *Prediction of 3D cardiovascular hemodynamics before and after CABG via deep learning*, Commun. Biol. 2021](https://doi.org/10.1038/s42003-020-01638-1) | Point-cloud network for 3D velocity and pressure fields, 600× faster than CFD. | Cite (point-based volume flow) |
| [Faisal et al., *Rapid WSS prediction for aortic aneurysms using deep learning (MultiViewUNet)*, MBEC 2025](https://doi.org/10.1007/s11517-025-03311-3) | Maps AAA geometry to multi-view images; U-Net predicts TAWSS; 23 real + 230 synthetic cases. | **Aorta reference.** Not re-run: it needs a fixed aortic topology and image projections, and predicts TAWSS only |
| [Liang, Mao, Sun, *A feasibility study of deep learning for predicting hemodynamics of human thoracic aorta*, J. Biomech. 2019](https://doi.org/10.1016/j.jbiomech.2019.109544) | Early aorta work (shape encoding + network). Title only; the abstract was not available in the search. | **Aorta reference.** Not re-run: it relies on point correspondence between shapes, which our free meshes don't have |
| [D'Inverno et al., *Mesh-informed ROMs for aneurysm rupture risk*, JCAM 2025](https://arxiv.org/abs/2410.03802); [POD + ML for cerebral saccular aneurysms, Sci. Rep. 2025](https://doi.org/10.1038/s41598-025-17823-3) | Reduced-order and POD surrogates for one patient or one geometry family. | Cite as the non-geometry-general alternative; not comparable |

### 2.2 ML backbones for fields on irregular geometry

| Paper | Why it's a candidate |
|---|---|
| [Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks* (MeshGraphNet), ICLR 2021](https://arxiv.org/abs/2010.03409) | The standard mesh GNN, used both as a direct field predictor (Table 1a) and as the time stepper (Table 2b) |
| [Wu et al., *Transolver: A Fast Transformer Solver for PDEs on General Geometries*, ICML 2024](https://arxiv.org/abs/2402.02366) ([code](https://github.com/thuml/Transolver)); [Transolver++, 2025](https://arxiv.org/abs/2502.02414) | Leading transformer for fields on arbitrary point sets. It handles surface and volume points in one model, the same setting as ours |
| [Li et al., *Geometry-Informed Neural Operator* (GINO), NeurIPS 2023](https://arxiv.org/abs/2309.00583) | Neural-operator family for large 3D geometry problems; optional row |
| [Alkin et al., *Universal Physics Transformers*, NeurIPS 2024](https://arxiv.org/abs/2402.12365); [AB-UPT, 2025](https://arxiv.org/abs/2502.09692) | Anchor-token transformers for automotive surface + volume CFD; architecturally close to our token pooling and query decoders. Optional row; check that code is available |
| [Kashefi et al., *A point-cloud deep learning framework for … fluid flow fields on irregular geometries*, Phys. Fluids 2021](https://arxiv.org/abs/2010.09469) | Point-cloud baseline lineage. PointNet++ already exists in this repo |
| [Hao et al., GNOT, 2023](https://arxiv.org/abs/2302.14376); [GAOT, NeurIPS 2025](https://arxiv.org/abs/2505.18781); [Geom-DeepONet, CMAME 2024](https://arxiv.org/abs/2403.14788) | Further operator-learning options; cite, don't run |

### 2.3 Time handling: rollout vs direct prediction

| Paper | Relevance |
|---|---|
| Garnier et al. 2025 (above) | Published aneurysm surrogate that **steps through time** |
| Garnier et al., ICML 2026 (above) | Rollout = forward Euler; error grows over long horizons; a predictor–corrector reduces it |
| [Lippe et al., *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers*, NeurIPS 2023](https://arxiv.org/abs/2308.05732) | Standard reference for rollout error accumulation |
| Sheng, Ding et al. 2026; STEP 2026 | **Direct** time conditioning (frame index / waveform embedding), as in our model |

### 2.4 Where our work sits (claims to verify, not to assume)

- **Volume vs surface.** Hachem's group predicts *volume* velocity and derives WSS from it; STEP and our earlier paper predict *wall* WSS only. Learned surrogates that predict **wall WSS and lumen flow jointly, with explicit coupling**, are rare. AB-GATr (2026) is the closest, and it is a backbone paper, not an aneurysm study.
- **Data.** Most aneurysm surrogates train on synthetic or idealised geometries (STEP, Li 2025), or on about 100 patients (AnXplore). Ours trains on 1,228 synthetic cases plus 484 real patients and tests on 121 unseen real patients.
- **Time.** The literature is split between stepping through the cycle (Hachem) and direct conditioning (STEP, our earlier paper). **No paper we found compares the two on the same data.** That makes Table 2b a contribution in its own right.

## 3. Shared evaluation protocol (both tables)

1. **Fix model selection first.**
   - **The problem:** `train_joint.py` uses the *test* loss both for `best.pth` and for the LR scheduler. A reviewer will flag that.
   - **The fix:** add a validation split (e.g. 10% of the 484 AngioFlow training cases, by case, seeded). Select the checkpoint and drive the scheduler on validation, and report test once.
   - **Existing runs:** they must be re-selected (or re-run) under this rule before their numbers go into the paper.
2. **One split for every row.**
   - **Test:** the same 121 AngioFlow cases (`--test_versions angioflow`, seed 0). AneuG-Flow and AneuGv2 are training-only.
   - **Not comparable:** the existing WSS-only result (0.408 relative L2) used a different split (61 test cases). It must be re-run on this one.
3. **Same input region and features.**
   - **Region:** every model sees the same aneurysm crop. At evaluation the crop is seeded and the wall is predicted at full resolution, as in `eval_joint.py`.
   - **Features:** baselines get the same per-node features (position, normal, geodesic distances, Laplacian eigenvectors) and the same time/waveform embedding. Table 1 then isolates the **architecture**, and Table 2 the **task** and **time strategy**.
4. **Same budget.**
   - **Training:** equal number of training samples seen; batch size as large as memory allows.
   - **Learning rate:** a short sweep per model (1e-4, 3e-4, 1e-3) picked on validation.
   - **Report:** parameter count, training GPU-hours, and inference time per case for the full cycle.
5. **Metrics.** All per case, then averaged over the 121 test cases.

   | Metric | Definition | Notes |
   |---|---|---|
   | WSS relative L2 | ‖pred − CFD‖ / ‖CFD‖ over wall nodes, WSS vectors | at frame 0, at peak systole (the highest-inflow frame; frame ~10 had the largest error in the LR test), and averaged over all 80 frames |
   | Velocity relative L2 | same, over lumen points | same three time points |
   | Pressure relative L2 | on the pressure anomaly | joint models only |
   | TAWSS, OSI, RRT error | MAE, plus relative error for TAWSS, from all 80 predicted frames | the clinically used indices; whole-cycle models only |
   | Sac-only WSS error and low-WSS area | WSS error restricted to aneurysm nodes (`node_type == 1`); fraction of the sac below a TAWSS threshold, vs CFD | what clinicians look at |
   | Inference time | full cycle, one case, same GPU | |

6. **Uncertainty.**
   - **Per-case differences:** compare each baseline with ours case by case, and give a bootstrap 95% interval for the mean difference and the number of cases where ours is better (e.g. 104/121).
   - **Seeds:** 3 seeds for our model and the strongest baseline; 1 seed elsewhere, stated in the caption.
   - **Breakdown:** by aneurysm type (bifurcation / sidewall / sidewall near bifurcation), in an appendix table.

## 4. Table 1: comparison with other methods

**Setting.** All rows are trained on the full training set. Rows run on the **whole cycle** where the budget allows; otherwise the table is split into a frame-0 part (all rows) and a whole-cycle part (ours plus the two best baselines). See the tiers in section 8.

### 4a. ML backbones in our pipeline

Same data, crop, features, time conditioning, loss and output heads; only the network changes.

| Row | Implementation | Why |
|---|---|---|
| PointNet++ | Exists: `GeoPointNetEncoder` (interior model `--backbone pointnet`) and `models/wss/pointnet2.py` | Classic point-based baseline; stands in for the PointNet-family vascular papers (Li 2021, Li 2025, Kashefi 2021) |
| MeshGraphNet (direct, not rolled out) | New: Hachem's group's MeshGraphNet implementation from `graph-physics`, on the wall mesh + lumen kNN graph, predicting each frame directly from geometry + time conditioning | The standard mesh GNN. Using their code keeps one implementation shared with the rollout rows of Table 2b |
| Transolver | New: wrap the official code (`thuml/Transolver`), wall and lumen points as one set | Strongest general geometry transformer |
| LaB-GATr | New: official code (Suk et al.) | Equivariant, built for biomedical meshes; STEP's backbone |
| *(optional)* AB-UPT or GINO | New | Anchor-token / operator families; add only if budget allows |
| **Ours (HierGPS encoder + coupled query decoders)** | Exists | |

### 4b. Published hemodynamics surrogates

| Row | How to run it fairly | Caveat |
|---|---|---|
| Graph Transformer, our earlier paper (Sheng, Ding et al. 2026), WSS only | `AngioGraphGPS` exists in `models/transformer.py`; its trainer (`train_transformer_t0.py`) needs porting to the current data layout | WSS columns only |
| Autoregressive graph transformer (Hachem group, Garnier et al. 2025) | Train their `graph-physics` transformer on our data (wall mesh + lumen kNN graph), one step per 10 ms frame. Two variants: start from the **CFD** frame 0 (as published) and from **our predicted** frame 0. | Their method uses a tetrahedral volume mesh; ours has lumen point clouds, so a kNN graph replaces mesh edges. Say so in the caption |
| STEP / LaB-GATr (Bai et al. 2026) | Covered by the LaB-GATr row in 4a (same backbone, same conditioning). Add steady-flow pretraining only if we have steady CFD for our cases | Their published numbers come from AneuG-Flow with another split: quote them in the text, not in the table |
| Aorta methods (MultiViewUNet, Liang 2019) | Not re-run (fixed topology / point correspondence) | Explain in one sentence in the paper why they don't apply to free aneurysm meshes |

Table layout (numbers to fill in):

| Method | Params | Train GPU-h | WSS rel. L2 (cycle / peak) | Velocity rel. L2 (cycle / peak) | Pressure rel. L2 | TAWSS MAE | OSI MAE | Inference (s/cycle) |
|---|---|---|---|---|---|---|---|---|
| PointNet++ | | | | | | | | |
| MeshGraphNet | | | | | | | | |
| Transolver | | | | | | | | |
| LaB-GATr | | | | | | | | |
| Graph Transformer (ours, 2026) | | | | — | — | | | |
| AR graph transformer (Garnier et al.), CFD start | | | | | — | | | |
| AR graph transformer (Garnier et al.), predicted start | | | | | — | | | |
| **Ours** | 2.74M | | | | | | | |

**Optional external check.** AnXplore (101 aneurysms, public: https://github.com/aurelegoetz/AnXplore) is Hachem's group's data.
- **Value:** training and testing our model on it would answer "does it work on someone else's data?".
- **Cost:** high. It needs a conversion pipeline: tetrahedral volume mesh → our wall graph, lumen points, geodesic features, Laplacian eigenvectors and flow split.
- **Comparison caveat:** their published errors are velocity RMSE in mm/s, so ours would have to be reported in the same units.
- **Recommendation:** keep it for a revision unless a reviewer is likely to insist.

## 5. Table 2: our model's design choices

All rows use our architecture and budget. Only the factor named in each row changes.

### 5a. What to predict: WSS only vs WSS + flow

| Row | Outputs | Run | Status |
|---|---|---|---|
| WSS only | WSS | `train.py --model wss_query_field --backbone hgps` (the same HierGPS encoder, with its own WSS query decoder), or the joint model with `--lambda_vel 0 --lambda_press 0` and the interior head removed | Exists; needs a run on the shared split |
| Flow only | velocity + pressure | `train_interior.py --backbone hgps` | Exists; needs a run |
| Joint, no coupling | all three | `train_joint.py --no_cross_field` | Exists; needs a run |
| **Joint (ours)** | all three | current run | Exists |

**How to read the outcomes.**
- **Joint beats WSS-only on WSS:** flow supervision helps WSS (the cleanest finding).
- **Joint with no coupling falls between the two:** the cross-field attention, not just the shared encoder, carries the gain.
- **Joint is merely equal on WSS:** the claim becomes "one model gives both outputs at no cost to WSS", which is still worth stating.

Run Table 2a at **frame 0 first**: four runs of about 3 days each on a local 24 GB GPU (section 8). Repeat on the whole cycle only for the two most informative rows (WSS-only vs joint).

### 5b. Time: predict any frame directly vs predict frame 0 and step through the cycle

| Row | How frames 1–79 are produced | What it isolates |
|---|---|---|
| Frame-0 model, applied to every frame | No time training (the LR test already showed it fails: loss 4.7 at frame 10 vs 0.15 at frame 0) | Lower bound: time must be learned |
| Quasi-steady scaling | frame-0 prediction × waveform scaling between frames, e.g. WSS(t) = WSS(0)·Q(t)/Q(0). With the steady CFD code (5c), also the *CFD* version: steady solution × Q(t)/Q̄ | A cheap physics baseline with no learned time. If it is close to the learned models, the time modelling adds little |
| Frame 0 + Hachem-group rollout, **from CFD frame 0** | Their models from `graph-physics`, trained on frame pairs (t → t+1) with their noise injection, given the waveform at t and t+1; started from the true frame 0. Two steppers: their **MeshGraphNet**, and their **masked graph transformer** (the aneurysm model of Garnier et al. 2025) | Rollout error alone (an optimistic upper bound for this approach) |
| **Frame 0 + Hachem-group rollout, from our predicted frame 0** | Same two steppers, started from our frame-0 model's output | **The two-stage approach as it would really be used** |
| Same, with Temporal Correction (Garnier et al., ICML 2026) | Their predictor–corrector version of the transformer stepper | Gives the rollout approach its best published form |
| **Direct whole-cycle (ours)** | The frame index and waveform condition the model; all 80 frames come from independent forward passes from geometry | |
| *(optional)* Direct, from scratch vs warm-started from frame 0 | Same as ours, different initialisation | Mirrors STEP's "steady pretraining helps 28%" finding; ours is warm-started |

**Extra outputs for 5b (a figure, not just the table):**
- **Error over the cycle:** relative L2 vs frame index for each row. Rollout error should grow with every step, while direct prediction should track the waveform's difficulty (systole hardest).
- **Cycle closure:** for the rollout rows, the difference between predicted frame 79→80 and frame 0. A periodic flow should close the loop.
- **Inference time:** the rollout needs 80 *sequential* steps; direct prediction can batch all 80 frames at once.

**Details to get right.**
- **Use their code, not a re-implementation.** The rollout rows use `graph-physics` as published, with their training recipe (one-step targets, noise injection) and their default hyperparameters scaled to our graph size. Only the data loader is ours. A reviewer then can't argue the baseline was weakened.
- **What the stepper predicts:** the change from frame t to t+1, given the waveform at both frames.
- **Its graph:** wall mesh edges plus a lumen kNN graph, and wall–lumen kNN edges so that WSS and velocity exchange information. Their published model uses a tetrahedral volume mesh; our lumen is a point cloud, so the kNN graph stands in for mesh edges (say so in the caption).
- **Frame alignment:** the same 80-frame, 10 ms grid as the data (which matches their Δt). Frame 0 is near diastole.
- **WSS-only variant:** the steppers also run with wall fields only, so that Table 2b can be filled for both the WSS-only and the joint task.

### 5c. Steady-state CFD prior: with and without, for WSS only and WSS + flow

A colleague has written a steady CFD solver that runs in Python; it will be wired in later. The steady solution becomes an **extra input**. The model still predicts the pulsatile field at every frame.

| Row | Task | Prior input | What it isolates |
|---|---|---|---|
| WSS only | WSS | — | (from 5a) |
| WSS only + steady prior | WSS | steady WSS vector on each wall node | Value of the prior for WSS |
| Joint (ours) | WSS + velocity + pressure | — | (from 5a) |
| **Joint + steady prior** | WSS + velocity + pressure | steady WSS on wall nodes; steady velocity and pressure on lumen points | Value of the prior when flow is also predicted |
| *(physics reference)* Steady CFD × waveform scaling | WSS (+ flow) | steady solution scaled by Q(t)/Q̄ | What the prior alone gives, without learning (also in 5b) |

**Design choices to fix before the runs.**
- **Steady boundary condition:** one steady solve per case, at the mean inflow of the shared waveform (Q̄). Use the same outlet split as the transient CFD.
- **How the prior enters:**
  - **As an input feature (default):** concatenate the normalised steady fields to the node and query features. It's the smallest change: the encoder sees the steady WSS on the wall, and the query embeddings see the steady velocity and pressure at each lumen point.
  - **As a residual (variant):** predict the difference between the transient field and the steady field × Q(t)/Q̄. Run this only if the input-feature version helps, to see whether the model then has less to learn.
- **Mesh for the steady solve:** reuse each case's original CFD volume mesh. `blood_data.pt` stores it: `pos` (nodes) and `cells` (tetrahedra, e.g. 259k nodes / 1.54M tets in one case). The wall registration nodes coincide exactly with its boundary nodes, and its leftover boundary triangles are the inlet/outlet caps, named from the rim labels in `node_type`. Export it as a `.msh` with wall/inlet/outlet zones for the colleague's solver.
- **Mapping onto our points:** with the original mesh, the steady WSS lands exactly on our wall nodes, with no interpolation. The lumen points are probably a subset of the volume nodes too (to verify); otherwise use barycentric interpolation inside the tetrahedra.
- **Normalisation:** the prior gets its own statistics (as `stats.pt` does for the targets), computed on training cases only.
- **Time grid:** whole-cycle. Run each pair once at frame 0 first, if the budget is tight.

**Report alongside accuracy:**
- **Cost:** a prior-informed model needs a steady CFD solve per new patient, so its inference time is (steady solve + network), not network only. Put both in the table: the pitch is accuracy per second, against transient CFD.
- **Robustness:** error when the steady solution comes from a coarser mesh or a cheaper solver setting. A prior that has to be perfect is not a cheap prior.

**How this differs from the published steady-flow ideas:**
- **STEP:** *pretraining* on steady cases, with no steady solve at inference.
- **Our earlier paper (Sheng, Ding et al.):** steady cases as *augmentation*.
- **Suk et al. 2025 (deep vectorised operators):** steady solution as an *input at inference*, the same idea as this table, shown on coronaries.

Table 2c tests the input version. Pretraining on steady cases (STEP-style) is a natural extra row if the colleague's solver can produce enough cases cheaply.

## 6. Optional Table 3: training data and components

Only if space and budget allow; otherwise an appendix.

| Row | Question |
|---|---|
| Real patients only (no AneuG-Flow / AneuGv2) | How much do the synthetic datasets help on real-patient test cases? Likely the most interesting extra result for clinical readers |
| Crop depth (`--mask_high` 0.10 / 0.25 / full vessel) | How much surrounding vessel does the model need? |
| No geometry-token pooling (`--geometry_token_pool none`) | Value of the token resampler |
| `--local_hops` 1 vs 2 | Local receptive field |
| Rotation robustness: test cases rotated at random | We use rotation augmentation, not an equivariant network. AB-GATr reports augmentation is inconsistent, so check that our errors don't change when cases are rotated |

## 7. What exists and what must be built

| Item | Status | Effort |
|---|---|---|
| Validation split + selection on validation | **Missing.** `best.pth` and the scheduler use test loss | small: `split_by_case` + loader + selection code |
| Whole-cycle evaluation (all 80 frames; TAWSS/OSI/RRT; sac-only; error-vs-frame) | **Missing.** `eval_joint.py` reports per-frame relative L2 but has no cycle indices | medium |
| WSS-only, flow-only, joint without coupling | Exist (`train.py`, `train_interior.py`, `--no_cross_field`) | runs only |
| PointNet++ backbone | Exists for WSS-only and flow-only; not wired into the joint model | small–medium |
| Hachem-group models (`graph-physics`: MeshGraphNet, masked graph transformer, Temporal Correction) | **Missing.** Needs a data loader from our wall graphs + lumen points to their format, and their environment (PyTorch, PyG, Lightning, DGL) | medium–large: one adapter serves the direct MeshGraphNet row (Table 1a), the AR transformer rows (Table 1b) and all rollout rows (Table 2b) |
| Transolver, LaB-GATr wrappers | **Missing** | medium each (official code; adapt inputs and heads) |
| Earlier Graph Transformer (`AngioGraphGPS`) | Model exists; its trainer targets the old data layout | small–medium |
| Steady-state prior | **Solver exists** (colleague's Python code). Missing: exporting each case's CFD volume mesh from `blood_data.pt` to `.msh` with wall/inlet/outlet zones; running the solver on all 1,828 cases; storing the steady fields next to the training data (and on the HF dataset); an input option in `train_joint.py` / `train.py` | medium, plus solver compute for every case (test cases included) |
| Quasi-steady scaling baseline | **Missing** | small (needs inflow Q(t) from the waveform) |
| AnXplore conversion (optional) | **Missing** | large |

## 8. Compute plan

Measured costs:
- **Whole-cycle joint run:** about 1 h per epoch (8 frames per case, batch 16, RTX PRO 6000 on HF, about $2.75/h). The current run is set to 100 epochs (≈100 h, ≈$275).
- **Frame-0 joint run:** about 16 min per epoch on a local 24 GB GPU at batch 4.

| Tier | Runs | Estimated cost |
|---|---|---|
| **A (needed for the tables)** | Re-select existing runs on validation · Table 2a at frame 0 (WSS-only, flow-only, joint without coupling) · Table 1a at frame 0 (PointNet++, MeshGraphNet, Transolver, LaB-GATr) · Table 2b (Hachem-group steppers trained once; rollouts from CFD and from predicted frame 0; quasi-steady scaling) · Table 2c (steady solves for all cases; WSS-only and joint, each with the prior) · earlier Graph Transformer at frame 0 | ~10 local runs of 2–3 days, plus two stepper runs and the steady solves. Feasible on GPUs 0/3 in about 3–4 weeks, or faster on HF |
| **B (strengthens claims)** | Whole-cycle runs for the 2 best baselines and WSS-only · whole-cycle runs of the two steady-prior rows · 3 seeds for ours and the best baseline · Temporal Correction stepper | ~6–8 HF runs at 50–100 h each (~$1,000–2,200) |
| **C (optional)** | Table 3 rows · AnXplore external test · warm-start vs scratch | as budget allows |

Order: validation split and cycle metrics → Table 2a (frame 0) → `graph-physics` adapter and Table 2b → Table 1a (frame 0) → steady prior wired in, Table 2c → tier B. The steady solves for all 1,828 cases can run in parallel with everything else as soon as the solver is wired in, since they need no GPU work from us.

## 9. Pitfalls

- **Test-set selection** (section 3.1). Every reported number must come from a validation-selected checkpoint.
- **STEP comparison.** STEP is trained and tested on AneuG-Flow, which we use for *training*. Our model has seen those geometries, so never evaluate it on STEP's test cases. STEP's own paper also notes geometry overlap across its stages. Quote STEP's numbers only as context.
- **Different flow representations.** Hachem's group uses tetrahedral volume meshes; we use lumen point clouds. When their method runs on our data, a kNN graph replaces the mesh. State this, and don't claim their published accuracy was reproduced.
- **Comparable metrics.** Relative L2 (ours, `eval_joint.py`), "maximum-based relative L2" (our earlier paper), MAE in Pa (STEP) and velocity RMSE in mm/s (Hachem) are all different. Report relative L2 *and* MAE in physical units, so readers can relate our numbers to each paper.
- **Same waveform.** All our cases share one inlet waveform, as STEP notes for itself. The whole-cycle model learns time, not a response to different waveforms, so don't claim generalisation to new inflow conditions. (Hachem's model was trained with several waveforms.)
- **Crop.** Every model, baselines included, must see the same aneurysm crop. Otherwise, differences in region size masquerade as differences in accuracy.
- **Steady prior on test cases.** Run the steady solve for test cases with the same settings as for training cases. Reusing the transient CFD's *mesh* is fine, since it is geometry only. Never start from the transient *solution* (initial fields, restart files), or test information leaks into the input. For a new patient, the inference-time cost must include volume meshing, since no CFD mesh exists yet.
- **Steady prior and the time claim.** A prior-informed model is no longer "geometry only". Present it as a separate operating point (more accurate, needs a steady solve), not as a replacement for the geometry-only model.

## Coverage of the review

- **Queries:** 19 over OpenAlex and arXiv, in three groups:
  - vascular and aneurysm surrogates (6);
  - backbones for fields on irregular geometry (7);
  - time handling and follow-ups (6).
- **Years:** 2018–2026.
- **Missing source:** Semantic Scholar was rate-limited throughout.
- **Read in full:** Garnier et al. (arXiv 2512.09013; partly from the HTML version) and their ICML 2026 paper, plus the `graph-physics` README.
- **Not checked:** the npj paper's full text, including where its 105-geometry benchmark is hosted. Check that before citing it as public.
- **Raw search results:** the JSON files from this review are in the session scratchpad (`lit_domain.json`, `lit_ml.json`, `lit_time.json`). Ask if you want them kept with the paper materials.
