# Presentation site

`site/index.html` is the clinician-facing results page (published as a claude.ai artifact). It reads its data from `site/data/`, which is gitignored and rebuilt by:

```bash
# from the parent of angioflow/, conda env "new"; GPU 1 needs the memory cap
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PWD" python -m angioflow.presentation.export_showcase --stage all --mem_frac 0.3
cp    /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_showcase/summary.json angioflow/presentation/site/data/
cp -r /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_showcase/viz        angioflow/presentation/site/data/
python -m angioflow.presentation.export_aneug --cases cohort_650_0002_t0,cohort_650_0000_t0,cohort_650_0336_t1,cohort_650_0327_t1
cp -r /media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_showcase/aneug      angioflow/presentation/site/data/
# artifacts don't serve raw binary: the page fetches each case as base64 text
for f in angioflow/presentation/site/data/{viz,aneug}/*.bin; do base64 -w0 "$f" > "${f%.bin}.b64.txt"; done
```

Publish `index.html` with every `data/**` file except the `.bin` files as supporting files.

- **Metrics:** both models on every AngioFlow case. Each model's own train/test split, evaluated like `training/eval_wss.py` and `training/eval_joint.py`. The output is `metrics_<model>.csv` and `summary.json`.
- **3D cases:** `--n_viz_per_type` test cases per aneurysm type that neither model trained on, closest to the type's median error. Each case is one binary file. `viz/cases.json` gives its layout.
- **Newer checkpoint:** to refresh the numbers, rerun with `--stage all`. To refresh only the 3D cases (which reuses the CSVs), rerun with `--stage viz`.
