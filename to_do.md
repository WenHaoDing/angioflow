# To do

## Inference: match the new precomputed graph features (not started)

Graphs built by `data/precompute_wall_static.py` → `huggingface/extract_pyg_stable.py`
(`Angioflow_PyG_stable/`) differ from the old pipeline in the cotangent-Laplacian
eigenpairs and at orphan nodes. Any model trained on them needs
`deploy/mesh_to_graph.py` to compute its inputs the same way, or deploy-time inputs
won't match training.

- **Eigenpairs:** `compute_cot_eigenpairs` (both `pytorch3d` and `scipy` backends)
  still hands the non-symmetric row-normalized Laplacian to `eigsh` (a symmetric
  solver), so its output isn't converged and changes run to run. Replace it with
  `cot_mass_eigs` from `data/precompute_wall_static.py`: generalized `L v = λ M v`
  (cot stiffness, lumped mass), unit-L2 modes, largest-|entry|-positive sign,
  fixed ARPACK `v0`. Import it rather than copying, so there is one implementation.
- **Eigenvalue scale:** new `cot_eigval` is in 1/m² (~1e4), old was ~1e-4. Check
  every use of `cot_eigval` as a feature (e.g. `models/transformer.py` `cot_pe`).
- **Orphan vertices** (in `pos`, in no face): compute normals / geo_dist /
  eigenvectors on the face-referenced surface, kNN-fill (k=3, inverse distance)
  at orphans, add the kNN edges -- same as `precompute_wall_static.process_case`.
  Ideally deploy calls that function's pieces directly.
- **Validation:** update `deploy/validate_against_dataset.py` to compare deploy
  output against `wall_graph_static.pt` / the new `static_*.pt` (expect exact
  eigenpairs now, not just subspace overlap).
- Checkpoints trained on the old graphs are not compatible with the new ones.

## Training: use the new graphs (not started)

- ~~Slice eigen features to `lap_k`~~ -- done: `AngioFlowWallDataset._load_static`
  keeps `cot_eigvec[:, :lap_k].clone()` / `cot_eigval[:lap_k]` before caching (not a
  transform: the per-worker `lru_cache` would otherwise hold all 64 modes, ~9 MB/case).
  Raises if a static has fewer than `lap_k` modes.
- **AneuGv2 in the loader:** `AngioFlowWallDataset` only knows the
  `aneug_flow` / `angioflow` case roots; add `aneugv2`.
