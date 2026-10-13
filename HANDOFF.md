# Handoff: AneuG-Flow/AngioFlow CFD graph re-extraction (angioflow_pyg_stable)

This copy of `angioflow/` was split out from `AneuG-Net` on 2026-10-13 to become
its own repo. This file summarizes an in-progress task so an agent picking up
here doesn't have to re-derive context.

## Task

Re-extract AneuG-Flow + AngioFlow CFD wall-shear-stress graphs (via `AngioFlowWallDataset` in
[data/datasets.py](data/datasets.py)) into a new output root with a different
on-disk layout, after `angioflowv2_merged` (AngioFlow source) was updated with new cases.

- AneuG-Flow source: `/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data`
  (case dirs `stable_*`)
- AngioFlow source: `/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged`
- New output root: `/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_stable`
- Config (unchanged from the old `angioflow_pyg_full` root): `n_frames=80`,
  `decimate_method="fps"`, `decimate_ratio=1.0`, `lap_k=8`

## Layout change: per-dataset subfolders

Old roots (`angioflow_pyg_full`, `_v1`, `_v2`) are **flat**:
`<save_dir>/static_v1_<cid>.pt`, `<save_dir>/wss_v1_<cid>_<t>.pt`, etc., both
datasets' files mixed together in one directory.

The new root organizes by dataset: `<save_dir>/aneug_flow/static_aneug_flow_<cid>.pt`,
`<save_dir>/angioflow/wss_angioflow_<cid>_<t>.pt`. (Dataset tags were renamed from
`v1`/`v2` on 2026-10-13 -- see [DATASETS.md](DATASETS.md).)

This is implemented as an **opt-in constructor flag**, `per_version_dirs=False`
by default, so callers that use flat roots (`train_joint.py`, `train_interior.py`,
`train.py`, `datasets_interior.py`, ...) keep the flat layout — only the new `angioflow_pyg_stable` extraction in
`datasets.py`'s `__main__` block passes `per_version_dirs=True`. Do not flip
the default; see `_version_dir()` in `data/datasets.py` for why.

Also added `max_cases_per_version=None` (opt-in), used to smoke-test the
extraction on ~30 cases/version before committing to the full run.

## Status so far

1. **Smoke-tested**: 30 cases/version extracted successfully into
   `angioflow_pyg_stable/{v1,v2}/`. Sanity renders (see below) looked correct.
   **These predate the tag rename** (`static_v1_*` names), so the current code
   won't see them as done -- the full run will redo them under
   `{aneug_flow,angioflow}/`; delete `angioflow_pyg_stable/{v1,v2}/` afterwards.
2. **AneuG-Flow source cleanup done**: removed 96 case dirs from
   `angioflowv1/transient_data` that are flagged as non-converged CFD in
   `angioflowv2_merged/_meta/cfd_anomalies.json` (confirmed exact 96/96 match
   against cases missing from the `whding123/AneuG-Flow` HF dataset mirror —
   those 96 were deliberately excluded from the HF upload for the same reason).
   Freed ~318G. Local `transient_data` now has exactly 634 `stable_*` cases,
   matching the HF mirror 1:1.
3. **Not yet run**: the full extraction (all 634 AneuG-Flow + whatever's in
   `angioflowv2_merged`) into `angioflow_pyg_stable`. The smoke test only
   covers 30/version; everything else still needs processing. Just rerun
   `datasets.py`'s `__main__` (same config, no `max_cases_per_version` cap).
   Already-done cases under the *current* tags are skipped via `_case_complete`;
   the pre-rename smoke-test files are not (see 1.).

## Sanity rendering: pyvista dropped, matplotlib only

Originally used pyvista for `wss_sanity_video()`. Reasoning for the rewrite:
extraction is eventually moving to HF Jobs, and pyvista needs a working
X/EGL/OSMesa backend for off-screen rendering — **verified broken** on HF
Jobs' default `cpu-basic` `uv run` container (two test jobs: one crashed on
`pv.start_xvfb()` not existing in that pyvista version, the retry without it
hung with VTK warnings about missing X/EGL/OSMesa until the timeout killed it;
nothing ever uploaded). A custom Docker image with `libosmesa6` installed
could probably fix this, but wasn't pursued — see `--image` flag on
`hf jobs uv run` if that's revisited later.

Current `wss_sanity_video()` in `data/datasets.py`:
- `mesh.png` — plain mesh, edges visible (matplotlib `plot_trisurf`)
- `geo_dist.png` — **one figure**, 4 panels (inlet / outlet_0 / outlet_1 / global)
- `cot_eigvec.png` — **one figure**, one panel per cotangent-Laplacian mode
- **No WSS animation.** A pyvista GIF took ~47s/case; the matplotlib
  equivalent (rebuild-free, recolor-only per frame) took ~9 minutes/case on a
  full-res mesh — rejected as unacceptable. Dropped entirely rather than kept
  slow; `wss_sanity_video(..., n_frames=...)` still takes `n_frames` for
  signature compatibility but no longer uses it for an animation.
- All three figures render against a **decimated throwaway copy** of the mesh
  (`_vis_mesh_and_remap()`, `max_faces=3000` via `decimate_zone_mesh` —
  pure VTK geometry filter, not rendering, so unaffected by the HF off-screen
  problem), not the full-res stored data. This cut total render time per case
  from ~65s to ~16s for the three static figures. The stored `static_*.pt` /
  `wss_*.pt` files themselves are untouched, full resolution as before.

If someone wants WSS visualized again later: either (a) a static multi-panel
"filmstrip" of ~6-8 representative frames in one figure (same pattern as
`geo_dist`/`cot_eigvec`, should run in well under a minute), or (b) revisit
the custom-Docker-image route for pyvista on HF Jobs.

## Running from this repo

Needs `AneuG` importable as a top-level package. The local copy at
`angioflow/AneuG/` (this repo's own `AneuG/`, i.e. `<this_repo>/AneuG/`) is
the active one — not `/home/wenhao/AneuG` (older, external). When running
scripts here, add both the parent of this repo and this repo's own root to
`sys.path` (mirrors how it's done from `AneuG-Net`, where
`sys.path.insert(0, ".../AneuG-Net/angioflow")` was needed before
`sys.path.insert(0, ".../AneuG-Net")`), e.g. from this repo's parent dir:

```python
import sys
sys.path.insert(0, "/media/yaplab2/HDD Storage/wenhao/angioflow")
sys.path.insert(0, "/media/yaplab2/HDD Storage/wenhao")
```

Conda env: `new` (`source <miniconda3>/etc/profile.d/conda.sh && conda activate new`).

This repo is a plain directory copy made via `cp -a` from `AneuG-Net/angioflow`
on 2026-10-13 — not yet git-initialized. No cross-package imports on
`ood_fix/` or anything else outside `angioflow/` were found, so it should be
self-contained.
