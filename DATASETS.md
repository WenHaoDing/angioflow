# Datasets

Three Hugging Face dataset repos, all under `whding123/`. **Use the dataset
name directly going forward** (`AneuG-Flow`, `AngioFlow`, `AneuGv2`) — the old
`v1`/`v2` convention from `AneuG-Net` is retired in this repo. It was a binary
label for two dataset families; `AneuGv2` is a third family that doesn't fit
that split, so the binary naming stops being useful here.

In code the tags are `aneug_flow` (AneuG-Flow) and `angioflow` (AngioFlow),
defined in [data/dataset_tags.py](data/dataset_tags.py): kwargs
`case_root_aneug_flow` / `case_root_angioflow`, `--versions aneug_flow,angioflow`,
`Data.version`, and new processed filenames (`static_aneug_flow_<cid>.pt`,
`wss_angioflow_<cid>_<t>.pt`, per-dataset subfolders `<save_dir>/aneug_flow/`).
Processed roots written before the rename (`angioflow_pyg_full`, `_v1`, `_v2`,
`_interior`, and the 30-case smoke test in `angioflow_pyg_stable/{v1,v2}/`) still
use `static_v1_*` / `static_v2_*` names and are **not readable** by the current
code — regenerate rather than read them. The only legacy shim is
`normalize_tag()`, which maps `"v1"`/`"v2"` when *reading* tags back from
existing JSONs (`cfd_anomalies.json`) and old checkpoints' saved `--versions`
args. Source dataset folder names on disk (`angioflowv1/`, `angioflowv2_merged/`)
are unchanged.

## `whding123/AneuG-Flow` — old synthetic dataset

Old `v1`. Fully synthetic (AneuG-generated) geometries + CFD.

- Local mirror: `/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data`
- Case ids: `stable_<n>` — 634 cases after cleanup (96 non-converged cases
  removed 2026-10-13, see [HANDOFF.md](HANDOFF.md); matches the Hub 1:1 now)
- `transient_data/<case>/`: `blood_data.pt`, `checkpoint.npy`,
  `flowsplit_ratio.txt`, `inlet_centroids.csv`, `shape.obj`,
  `shape_remeshed.obj`, `wall_data.pt`, `wall_mesh_registration.pt`
- `wall_data.pt` is a **flat dict** — `wall_data["x_wall_shear"]` etc. directly,
  `[T, N, 1]` each, no nested zone key
- Also on the Hub but not case-structured: `processed_data/` (precomputed
  registered-data `.pth` bundles + an old graph-encoder checkpoint),
  `steady_data/raw_data.pth`, `cfd/waveform.txt`, `Geometries.zip`

## `whding123/AngioFlow` — real patient dataset

Old `v2`. Real patient geometries + CFD (not synthetic).

- Local mirror: `/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged`
- Case ids: hash + aneurysm index, e.g. `0HRrQPIFOr_aneurysm1` (one patient
  can have multiple aneurysms, each its own case) — 609 cases with CFD on the Hub
- `cfd_data/<case>/`: `blood_data.pt`, `branch_ranking.npy`,
  `geomagic_processed.obj`, `ghd_merged_reconstruction.obj`,
  `ghd_reconstructed.obj`, `ghd_smoothed_reconstruction.obj`,
  `inlet_centroids.csv`, `output/` (raw solver output dir), `wall_data.pt`,
  `wall_mesh_registration.pt`
- `wall_data.pt` is **nested** — `wall_data["wall"]["x_wall_shear"]` etc.
  (different convention from AneuG-Flow's flat dict — both extraction/registration
  code paths already branch on this per-version)
- Also has `ghd_graphs/ghd_graphs.zip` (GHD-surface extraction product, same
  family of pipeline as AneuGv2's, run against real-patient geometry)

## `whding123/AneuGv2` — new synthetic dataset

Newer synthetic generation, different pipeline from AneuG-Flow (GHD-eigenbasis
reconstruction rather than direct wall-mesh registration). Produced/consumed by
`ood_fix/cfd_data_factory/extract_ghd_graphs_aneugv2.py` (version tag
`"aneugv2"` in that script's output filenames).

- Case ids: `cohort_<N>_<idx>_t<k>`, e.g. `cohort_650_0000_t0`
- 650 cases have `geometry/`, only 519 also have `cfd_data/` (CFD not
  finished/uploaded for every registered geometry yet)
- `geometry/<case>/`: `cap_masks.npz`, `centerlines.npy`, `dome.obj`,
  `dome_subdivided.npz` (uncapped + subdivided, carries the GHD eigenbasis —
  produced locally by `angioflow/data/subdivide_aneugv2_dome.py`, needs
  pytorch3d, **not** run inside the HF Job itself), `flowsplit_ratio.txt`,
  `geomagic_processed.obj`, `ghd.npy`, `ghd_forward_fusion_info.npz`,
  `ghd_surface.vtp`, `inlet_centroids.csv`, `jnl_transient`, `mesh.obj`,
  `meta.json` (has `aneurysm_type`), `openings.npy`,
  `parabolic_vinlet_udf_full.c`, `regularization.json`,
  `udf_waveform_5cycles.txt`, `uncap_sanity.png`, `z.npy`
- `cfd_data/<case>/`: `blood_data.pt`, `flowsplit_ratio.txt`,
  `geomagic_processed.obj`, `ghd_forward_fusion_info.npz`,
  `inlet_centroids.csv`, `output/`, `wall_data.pt` (nested under `"wall"`,
  same convention as AngioFlow), `wall_mesh_registration.pt`
- `cfd_data/_sanity/<case>/`: QC renders from an earlier registration step
  (`geo_dist.png`, `mesh.png`, `registration.png`, `vel.gif`, `wss.gif`) —
  **not** the same as `extract_ghd_graphs_aneugv2.py`'s own sanity output,
  which goes into `ghd_graphs/`'s zip instead
- `assets/`: `ghd_eigenvectors_bifurcated.npy`, `ghd_eigenvectors_sidewall.npy`
  — canonical GHD eigenbases shared across cases, selected per-case by
  aneurysm/vessel type
- `ghd_graphs/`: `raw/<case>/static_aneugv2_<case>.pt` +
  `wss_aneugv2_<case>_<t>.pt` (per-case, resumable across separate Job runs),
  plus the final bundled `ghd_graphs_aneugv2.zip`
- Two-stage pipeline, see that script's own module docstring for the full
  reasoning: (1) `subdivide_aneugv2_dome.py` locally (pytorch3d) →
  `dome_subdivided.npz`; (2) `extract_ghd_graphs_aneugv2.py` as an actual
  `hf jobs uv run` job (self-contained — no local `angioflow`/`ood_fix` import,
  since the Job runs in a fresh environment)
