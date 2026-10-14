# Review brief: HF graph-extraction scripts (temporary — do not commit)

You are reviewing two new Hugging Face Job scripts that build the training graph
data, plus the local code they depend on and feed. Goal: find anything that would
make the extracted data **wrong**, or that would break or silently corrupt **model
training later today**. Report findings; don't fix them unless asked.

- `huggingface/extract_pyg_stable.py` — wall graphs (static + 80 WSS frames)
- `huggingface/extract_interior_pyg_stable.py` — interior pools (velocity/pressure)

## Ground rules

- **Do not upload or modify anything on the Hub.** Never pass `--execute --yes`.
  Without those flags both scripts only build their first part locally in
  `--scratch` (read-only Hub listing is fine).
- Don't delete or overwrite data under `/media/yaplab2/wd8tb/wenhao/datasets/`.
  Write test output to a scratch/temp directory.
- Python env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate new`.
  Run modules from the repo's parent: `cd "/media/yaplab2/HDD Storage/wenhao"`
  (or `PYTHONPATH=..` from inside the repo).
- Dataset tags (naming rule): `aneug_flow` (whding123/AneuG-Flow, synthetic),
  `angioflow` (whding123/AngioFlow, real patients), `aneugv2` (whding123/AneuGv2).
  Never `v1`/`v2` in new code. `data/dataset_tags.py` maps legacy tags on read only.

## Pipeline

```
Hub source repos (CFD per case: wall_data.pt, blood_data.pt, wall_mesh_registration.pt)
  │  huggingface/download_wall_mesh_registrations.py     (local mirror of registrations)
  ▼  data/precompute_wall_static.py   LOCAL, needs pyvista  → wall_graph_static.pt per case
  │  huggingface/upload_wall_graph_static.py   → next to each registration on the Hub (DONE, all 1,760)
  ▼
HF Jobs (mount source repo read-only at /data, clone this repo, pip torch/numpy/scipy/huggingface_hub)
  ├─ extract_pyg_stable.py          → whding123/AngioFlow-PyG  wall/<tag>.zip  (via wall/_parts/<tag>/)
  └─ extract_interior_pyg_stable.py → whding123/AngioFlow-PyG  interior/<tag>/part_jJJ_NNNN.zip + manifest.json
  ▼
Local: download + unzip everything into ONE root → train locally
  <root>/<tag>/static_<tag>_<cid>.pt, wss_<tag>_<cid>_<t>.pt, manifest.json
  <root>/interior/interior_static_<tag>_<cid>.pt, interior_<tag>_<cid>_<t>.pt
```

Case counts with registrations: aneug_flow 634, angioflow 607, aneugv2 519
(31+ more AneuGv2 cases still being uploaded, not yet registered — expected warnings).

## What each output file contains

**Wall static** `static_<tag>_<cid>.pt` = `wall_graph_static.pt` minus `meta`:
`pos [N,3]` (registration pos minus its mean over all N nodes), `edge_index [2,E]`
(unique i<j: face edges + orphan kNN edges), `faces [F,3]`, `node_type [N]`,
`flow_split [2]`, `aneurysm_type`, `global_geo_dist [N]`, `geo_dist [N,3]`,
`normals [N,3]`, `cot_eigval [64]`, `cot_eigvec [N,64]`, `orphan_mask [N]`.
**Wall frame** `wss_<tag>_<cid>_<t>.pt` = `[N,3]` float32, t = 0..79.

**Interior static** `interior_static_<tag>_<cid>.pt` = `{"pos": [P,3], "wall_dist": [P]}`
(no `edge_index`). **Interior frame** `interior_<tag>_<cid>_<t>.pt` =
`{"vel": [P,3], "press": [P,1], "grad_v": [P,9], "grad_p": [P,3]}`, t = 0..79.

## Claims to verify (and how they were checked so far)

| Claim | Checked by |
|---|---|
| Node order of `wall_graph_static.pt` == registration `pos` == `wall_data.pt` (WSS sliced with no remap) | positions/faces exactly equal on 5 aneug_flow, 5 angioflow, 2 aneugv2 cases |
| Wall fields other than cot/orphans identical to old `datasets.py` `_process_case` | exact equality on 2 cases (`pos, faces, node_type, flow_split, aneurysm_type, global_geo_dist`; `geo_dist`, `normals` at non-orphan nodes) |
| WSS frames identical to old pipeline | bit-identical at t=0,41,79, aneug_flow + angioflow |
| Interior pool/fields identical to `data/datasets_interior.py` | same-input comparison on 1 case/dataset; bit-identical to old `angioflow_pyg_interior` files for aneug_flow `stable_0`, `stable_139`, `stable_4277` (all 80 frames). **Use only AneuG-Flow for old-file comparisons** — some AngioFlow CFD cases were updated since. |
| Static checksum guard rejects a static computed from a different registration | tested |
| Wall `--assemble` merges parts byte-identically, with manifest | tested offline (Hub calls faked) |
| Extracted data loads through the real classes | `AngioFlowWallInteriorDatasetT0(per_version_dirs=True, load_edge_index=False, freeze_dataset=True)` on 3 aneug_flow cases |
| Job scripts import without pytorch3d/pyvista/vtk/meshio/trimesh/matplotlib/AneuG | import with those blocked |
| Smoke Job on HF (clone, pip, mount, 3 aneugv2 cases, dry run) | passed for wall script; **interior smoke Job not run yet** |

Re-verify anything you can; independent checks are the point.

## Intentional differences from the old pipeline

1. **Cotangent Laplacian:** old `graph.compute_pe_cot_eigenvector` gave `eigsh` a
   non-symmetric matrix (non-converged, different every run). New
   `precompute_wall_static.cot_mass_eigs`: `L v = λ M v` (cot stiffness, lumped
   mass), exact, deterministic, 64 modes, unit-L2 + fixed sign. **Eigenvalues are
   ~1e4 (1/m²) vs ~1e-4 before.** `AngioFlowWallDataset._load_static` slices to
   `lap_k` before caching.
2. **Orphan vertices** (in `pos`, in no face; AneuG-Flow only, 4,546 nodes in 633
   cases): normals / geo_dist / eigvecs kNN-filled (k=3, inverse distance), 3 kNN
   edges each, `orphan_mask`. Old pipeline left `geo_dist=inf`, zero normals.
3. **Interior has no `edge_index`** (graph needs `shape_remeshed_cfd.obj` for
   AneuG-Flow, not on the Hub). Interior sampling uses `pool_ratio=0.16`,
   `near_decay=0.1`, `far_floor=0.25`, `min_wall_dist=1e-4`, seed 0.
4. Old checkpoints are **not** compatible with the new graphs (eigenpairs differ).

## Known gaps — check these against training today

1. **`per_version_dirs`:** new wall files live in `<root>/<tag>/`. **No training
   script passes `per_version_dirs=True`** (e.g. `training/train_joint.py`
   `build_dataset`), so they'd look in `<root>/` and find nothing.
2. **Case discovery:** `AngioFlowWallDataset.__init__` lists *source* case folders
   (`case_roots` + `case_pattern`), not extracted files. Cases without a local
   source folder (all AneuGv2) are invisible; `aneugv2` isn't in `case_roots` at all.
3. **Interior `edge_index`:** loaders must use `load_edge_index=False`.
   `train_joint.py` does; `train_interior.py` sets it True when
   `lambda_div > 0` with `div_method == "graph"` → would fail on new data.
4. **Startup cost:** `_rebuild_successful_index` fully loads every static only to
   compute unused counts; with 64-mode statics that's ~25 GB read per construction.
5. **Defaults:** `config/defaults.py` defaults `--decimate_ratio 0.5 --decimate_method quadric`;
   extracted data is full resolution (recent runs pass `1.0`). With `freeze_dataset`
   no reprocessing happens, but confirm nothing else depends on the ratio.
6. **Sanity hook:** dataset construction now calls `render_sanity()` (renders
   outliers + 5 normal cases per dataset into `<save_dir>/sanity/`, rank 0 only;
   `sanity_n_normal=0` disables). Confirm it can't slow or break a training start.
7. `cot_eigval` scale change (intentional difference 1) vs any model
   that feeds eigenvalues in as features (e.g. `models/transformer.py` `cot_pe`)
   and any normalization stats computed on them.
8. Interior `press` is absolute pressure (`absolute_pressure` key), same as before;
   `--press_zero_max` in `train_joint.py` handles re-referencing.

## Suggested checks

- Read both scripts end to end; look for off-by-one in frame ranges, wrong key
  names, dtype changes, tensors saved as views (whole storage saved), zip paths,
  resumability logic (`parts_on_hub`, part naming, `--num_jobs/--job_index` split
  by `crc32(case_id) % N`), and what happens on partial failure.
- Build a small root locally (no upload): make a folder that mimics the mounted repo
  (`<dir>/transient_data/<case>/` with symlinks to `wall_graph_static.pt`,
  `wall_mesh_registration.pt`, `wall_data.pt`, `blood_data.pt`), run both scripts
  with `--data_root <dir>` (no `--execute`), unzip both parts into one root.
  Sources: `wall_graph_static.pt` → `/media/yaplab2/wd8tb/wenhao/datasets/hf_wall_graph_static/aneug_flow/transient_data/<case>/`;
  registrations → `.../hf_registrations/aneug_flow/transient_data/<case>/`;
  `wall_data.pt`, `blood_data.pt` → `.../angioflowv1/transient_data/<case>/`.
- Point `training/train_joint.py`'s dataset construction at that root and see what
  breaks (gaps 1-2 expected); ideally run one training step.
- Sanity CLI on the root: `python -m angioflow.visualize.data_sanity --save_dir <root>`.

## Key files

`huggingface/extract_pyg_stable.py`, `huggingface/extract_interior_pyg_stable.py`,
`data/precompute_wall_static.py`, `data/datasets.py` (`_load_static`, `render_sanity`,
`__init__` discovery), `data/datasets_t0.py`, `data/datasets_interior.py`,
`data/dataset_tags.py`, `visualize/data_sanity.py`, `training/train_joint.py`,
`config/defaults.py`, `to_do.md`.
