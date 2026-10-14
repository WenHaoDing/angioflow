# HF graph extraction review findings

Review date: 2026-10-14. Scope: `REVIEW_hf_extraction.md`, both extraction scripts, and their local dataset/training consumers.

**Recommendation: request changes before training on the new extracted root.**

## Findings

### 1. [P1] Training cannot read the extracted wall layout

Location: [training/train_joint.py:150](training/train_joint.py#L150).

`build_dataset` omits `per_version_dirs=True`, while the wall archives extract into `<root>/<tag>/`. With one real extracted case, the default constructor returned **0 samples**; enabling that option returned **1 valid sample**. Training entry points need to expose or select the new layout.

### 2. [P1] AneuGv2 remains undiscoverable

Location: [data/datasets.py:60](data/datasets.py#L60).

Case discovery depends exclusively on local source folders. `case_roots` contains only `aneug_flow` and `angioflow`, so even after fixing the layout, extracted AneuGv2 cases cannot enter training. Frozen datasets should discover cases from extracted files or manifests, including the `aneugv2` tag.

### 3. [P1] Hub listing failures can overwrite existing parts

Locations: [huggingface/extract_pyg_stable.py:172](huggingface/extract_pyg_stable.py#L172), [huggingface/extract_interior_pyg_stable.py:262](huggingface/extract_interior_pyg_stable.py#L262).

Both `parts_on_hub` functions catch every listing exception and return `{}`. A simulated service failure reproduced this behavior. A resume then starts numbering at zero and can overwrite existing ZIP/JSON pairs, losing previously extracted cases. Assembly or finalization can likewise publish an empty result after a listing failure.

Only a confirmed missing destination should mean “no parts”; transient, authentication, and other listing failures should retry or abort.

### 4. [P1] The new eigenvalue scale suppresses spatial eigenvector features in CotPE

Location: [models/transformer.py:98](models/transformer.py#L98).

`CotPE` stacks eigenvector and eigenvalue values into `[N, k, 2]` and applies `BatchNorm1d(k)`. Each mode's normalization statistics therefore combine both feature types. The large new eigenvalues dominate the normalization and substantially suppress spatial variation in the eigenvector feature.

On AneuG-Flow `stable_0`, the nonconstant eigenvectors among the first eight modes had spatial standard deviation around **0.0053** before BatchNorm and only **6.5e-8–1.1e-6** afterward. This is a measured feature-scaling issue; its effect on final training quality was not benchmarked.

Normalize the two features separately or introduce an appropriate eigenvalue scaling convention before training models using this encoder.

### 5. [P2] Resume accepts incompatible extraction configurations

Locations: [huggingface/extract_interior_pyg_stable.py:286](huggingface/extract_interior_pyg_stable.py#L286), [huggingface/extract_interior_pyg_stable.py:354](huggingface/extract_interior_pyg_stable.py#L354).

Resume skips cases solely by case ID. Changing the seed, sampling parameters, or frame count produces a mixture of old and new configurations under the same dataset destination. Finalization nevertheless reports the first part's configuration for the whole dataset. Wall extraction similarly does not validate existing parts' frame counts before resume or assembly.

Validate all existing part configurations before resuming or finalizing, and reject incompatible settings or use a separate destination.

### 6. [P2] Failed interior cases leave untracked files inside parts

Location: [huggingface/extract_interior_pyg_stable.py:225](huggingface/extract_interior_pyg_stable.py#L225).

The script writes static data into the part before accessing all required physics fields. A synthetic case missing `dx_velocity_dx` raised `KeyError` after leaving `interior_static_aneug_flow_bad.pt` in the ZIP. The extraction loop excludes the failed case from its manifest but cannot remove its already-written entries. If other cases succeed, the part can still be uploaded with those untracked files.

Later retries can create overlapping entries across parts, making unzip order matter. For example, an earlier failed attempt's static can overwrite a successful retry's static; with different sampling settings, that can pair the wrong positions with the retry's frame fields. Stage each case completely before adding its files to the part.

### 7. [P1, conditional] Graph-divergence training fails on these interior files

Location: [training/train_interior.py:163](training/train_interior.py#L163).

When `lambda_div > 0` and `div_method="graph"`, the training script enables `load_edge_index`. The new interior statics omit that field, so sample loading raises `KeyError`. Joint training already disables interior edge loading.

Reject the incompatible configuration early or provide a supported alternative.

## Independent validation

Tests used the `new` Python environment and wrote output only under `/tmp`. No Hub calls were used to upload or modify data; listing-failure tests used a fake API.

- Extracted all 80 wall and interior frames for AneuG-Flow `stable_0`, using read-only source files linked into a temporary mount layout.
- All 80 extracted WSS frames were bit-identical to the source WSS slices.
- Extracted wall faces were exactly equal to registration faces.
- Interior `pos` and `wall_dist` were bit-identical to the old local `interior_static_v1_stable_0.pt` fields.
- All 80 interior frames were bit-identical to the corresponding old local `interior_v1_stable_0_<t>.pt` files for every exported field. Legacy tags were used only to read existing comparison files.
- The combined extracted root loaded through `AngioFlowWallInteriorDatasetT0` with `per_version_dirs=True`, `freeze_dataset=True`, `load_edge_index=False`, `sanity_n_normal=0`, and `lap_k=8`.
- The loaded sample contained 35,488 wall nodes and 85,725 interior points; loaded interior velocity values were finite.
- Omitting `per_version_dirs=True` returned an empty dataset for that same root.
- Both extraction scripts and the actual interior dataset class imported successfully in the local environment.
- Fake listing failures returned empty part maps in both scripts.
- A missing interior gradient field left a static ZIP entry after case failure.
- Directly measured CotPE's BatchNorm suppression of spatial eigenvector variation using the extracted wall static.

Scratch artifacts: `/tmp/hf-review-n3q7rlgs`. Reproduction scripts: `/tmp/review_hf_checks.py` and `/tmp/review_hf_compare.py`.

## Limits of this review

No HF interior Job, full training step, or all-dataset validation was run. The default sanity rendering hook was disabled during the loading test, so its startup cost and failure behavior were not runtime-validated. The numerical cotangent eigensolver was inspected but its full eigenpair correctness and determinism were not independently rerun. Successful comparisons on one AneuG-Flow case do not establish correctness across all cases or datasets.

The review made no fixes to implementation files, modified no source datasets, and made no Hub changes.
