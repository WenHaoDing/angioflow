# From updated CFD cases to training data

When cases in `whding123/AngioFlow` or `whding123/AneuGv2` (`cfd_data/<case>/`) get new CFD
results, these files have to be rebuilt, in this order. **Job** = runs on Hugging Face Jobs;
**local** = runs on this machine.

| # | Script | Runs | Produces |
|---|---|---|---|
| 1 | `huggingface/register_cfd_shapes_angioflow_hf.py` (AngioFlow) | local, on the fly (downloads `wall_data.pt`, registers, uploads; needs the local geometry folders + `new` env) | Hub `cfd_data/<case>/wall_mesh_registration.pt` for every case lacking one |
| 1  | `huggingface/register_cfd_shapes_aneugv2.py` (AneuGv2) | **Job** (`hf jobs uv run`) | Hub `cfd_data/<case>/wall_mesh_registration.pt` + `cfd_data/_sanity/<case>/` |
| 2 | `huggingface/download_wall_mesh_registrations.py` | local | local copy of every Hub registration (`hf_registrations/`) |
| 3 | `data/precompute_wall_static.py` | local | `wall_graph_static.pt` per case (normals, geodesic distances, Laplacian eigenbasis) |
| 4 | `huggingface/upload_wall_graph_static.py` | local | Hub `cfd_data/<case>/wall_graph_static.pt` |
| 5 | `huggingface/extract_pyg_stable.py` (`--assemble` after) | **Job** | `whding123/AngioFlow-PyG` → `wall/<dataset>.zip` (wall graphs + WSS) |
| 6 | `huggingface/extract_interior_pyg_stable.py --lean` (`--finalize` after) | **Job** | `whding123/AngioFlow-PyG` → `interior_lean/<dataset>/` (velocity + pressure) |
| 7 | `huggingface/download_angioflow_pyg.py` | local | local `AngioFlow-PyG/` training root |
| 8 | `data/generate_flow_split.py` + `hf upload` | local | `AngioFlow-PyG/<dataset>/flow_split.json` (+ Hub `flow_split/<dataset>/`) |

Then train (`training/train.py` or `train_joint.py` with `--extracted`; on a Job, `train_hf.py` or
`train_joint_hf.py`, which download the data themselves).

## Things to know

- **Updating a case wipes its derived files.** A re-exported case folder on the Hub no longer
  has its `wall_mesh_registration.pt` and `wall_graph_static.pt`, so it needs steps 1–4 again.
- **Steps 5 and 6 skip cases they have already extracted.** To pick up updated cases, delete the
  dataset's old output on AngioFlow-PyG first (`wall/_parts/<dataset>/` + `wall/<dataset>.zip`,
  and `interior_lean/<dataset>/`), then rerun.
- **Jobs run the code on GitHub:** steps 5 and 6 clone the repo, so push local changes first.
  Step 1 for AneuGv2 is a single uploaded uv script, so it needs no push.
- **AngioFlow step 1 never reads the local mirror's `wall_data.pt`** (stale for re-exported
  cases); `data/register_cfd_shapes_angioflow.py` is the same registration on local files only.
  Don't use `huggingface/upload_angioflow.py` for registrations: it re-uploads whole case folders,
  including the possibly stale local `blood_data.pt`.
- **New exports carry `frames.json`** with `"cycle": 2`, meaning the correct second cycle.

## Status (2026-10-29)

| Dataset | Re-exported (have `frames.json`) | Lacking registration + static |
|---|---|---|
| AngioFlow | 6 of 609 | those 6, plus `SURRrgDljJ_aneurysm2` and `eM6xqoqoo1_aneurysm1` (never registered) |
| AneuGv2 | 444 of 590 | those 444 (the other 146 are still the old exports) |
