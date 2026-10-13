"""Assemble a real-patient eval dataset: cases are GHD-reconstructed into a
shared canonical frame (so all shapes are directly comparable/overlaid, only
differing by true patient scale), with CFD wall shear stress registered onto
each mesh via a transient world-frame alignment step (see
registration.reconstruct_case_meshes for the split).

dataset source and aneurysm_type both come from case_registry.xlsx -- see
angioflow/data/build_case_registry.py. The GHD fitting folder keeps its
original case names and is indexed via simplify_name(), so nothing there needs
renaming.

Feeds the same downstream consumers as the original (synthetic-data)
new_version/loaders.py:assemble_registered_transient_data --
new_version/datasets_wss_optimized.py's GraphEncoder / RegisteredMultiTaskImageDataset,
as used by paper/eval_whole_cycle.ipynb -- hence the exact mesh_data /
registered_data_list schema match (units: meters, matching that pipeline).

conda activate new
python -m angioflow.data.angioflow_registration.build_aneux_type0_eval                       # AneuX, type 0
python -m angioflow.data.angioflow_registration.build_aneux_type0_eval --datasets AneuX ImperialNHS
python -m angioflow.data.angioflow_registration.build_aneux_type0_eval --aneurysm-types 1 2 --tag sidewall

reference dataset: /media/yaplab2/wd8tb/wenhao/datasets/AneuG_Flow_stable/AneuG_processed/assembled_registered_data_1k_v5.pth
reference code for generating dataset: assemble_registered_transient_data in /home/wenhao/AneuG-Net/new_version/loaders.py
"""
import argparse
import os

import torch

from angioflow.data.angioflow_registration.registration import (
    assemble_registered_transient_data, load_given_norm, DEFAULT_REGISTRY_PATH,
)
from angioflow.data.angioflow_registration.sanity import (
    save_wss_sanity_pictures, save_canonical_alignment_overlay, save_cfd_vs_registered_wss,
)

CASE_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
GHD_ROOT = "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/fitting_results_NewAneuX/Final"
OUTPUT_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/aneux_type0_eval"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case-root", default=CASE_ROOT, help="root holding per-case wall_data.pt folders")
    p.add_argument("--ghd-root", default=GHD_ROOT, help="AneuX GHD fitting root (original case names)")
    p.add_argument("--registry", default=DEFAULT_REGISTRY_PATH, help="case_registry.xlsx path")
    p.add_argument("--output-root", default=OUTPUT_ROOT)
    p.add_argument("--tag", default=None, help="suffix for the output .pth / sanity dir (default: derived from filters)")
    p.add_argument("--datasets", nargs="+", default=["AneuX"], choices=["AneuX", "ImperialNHS"],
                   help="dataset sources to include (default: AneuX only)")
    p.add_argument("--aneurysm-types", nargs="+", type=int, default=[0],
                   help="aneurysm types to include; 0=bifurcated, 1/2=sidewall (default: 0)")
    p.add_argument("--limit", type=int, default=None, help="stop after N cases (quick smoke builds)")
    p.add_argument("--max-align-mean-mm", type=float, default=0.5,
                   help="drop a case if its mean mesh->CFD nearest-point distance exceeds this (0 disables)")
    p.add_argument("--max-align-max-mm", type=float, default=0.5,
                   help="drop a case if its WORST mesh->CFD nearest-point distance exceeds this (0 disables)")
    p.add_argument("--n-subdivide", type=int, default=1)
    p.add_argument("--downsampling-factors", nargs="+", type=int, default=[4, 4])
    p.add_argument("--given-norm-from", default=None, metavar="STEADY_PTH",
                   help="assembled steady .pth whose tensor_norm normalizes the output; "
                        "required for feeding models trained on the old dataset, which was saved normalized")
    p.add_argument("--overwrite", action="store_true", help="rebuild even if the output .pth already exists")
    p.add_argument("--no-sanity", action="store_true", help="skip WSS / alignment sanity renders")
    p.add_argument("--n-wss-compare", type=int, default=8,
                   help="cases to render raw-CFD vs registered WSS for (0 = all)")
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: cuda:0 if available)")
    return p.parse_args()


def main():
    args = parse_args()
    tag = args.tag or f"{'_'.join(args.datasets)}_type{''.join(map(str, args.aneurysm_types))}"
    os.makedirs(args.output_root, exist_ok=True)
    assembled_path = os.path.join(args.output_root, f"assembled_registered_data_{tag}.pth")
    sanity_dir = os.path.join(args.output_root, f"sanity_{tag}")

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    given_norm = load_given_norm(args.given_norm_from) if args.given_norm_from else None
    print(f"[build] datasets={args.datasets} types={args.aneurysm_types} "
          f"normalize={given_norm is not None} -> {assembled_path}")

    data = assemble_registered_transient_data(
        args.case_root, ghd_root=args.ghd_root, registry_path=args.registry,
        aneurysm_types=args.aneurysm_types, datasets=args.datasets, limit=args.limit,
        n_subdivide=args.n_subdivide, downsampling_factors=tuple(args.downsampling_factors),
        normalize=given_norm is not None, given_norm=given_norm,
        max_align_mean_mm=args.max_align_mean_mm or None,
        max_align_max_mm=args.max_align_max_mm or None,
        assembled_write_path=assembled_path, overwrite_assembled=args.overwrite,
        device=device,
    )

    print(f"\nWrote assembled data to {assembled_path}")
    md_all = data["mesh_data"]
    md_all = {None: md_all} if "cases" in md_all else md_all  # single-type builds are flat
    for atype, md in md_all.items():
        sources = {}
        for s in md["dataset_source"]:
            sources[s] = sources.get(s, 0) + 1
        print(f"  type {atype}: {len(md['cases'])} cases {sources}")
        print(f"    {md['cases']}")
    print("Skipped: " + ", ".join(f"{k}={len(v)}" for k, v in data["skipped"].items()))
    for case in data["skipped"]["misaligned"]:
        a = data["alignment"][case]
        print(f"  [misaligned] {case}: NN mean {a['mean']:.3f} mm, max {a['max']:.3f} mm")

    # alignment QC report (every case that got as far as the knn step, kept or not)
    os.makedirs(sanity_dir, exist_ok=True)
    report = os.path.join(sanity_dir, "alignment_report.csv")
    kept = set(data["mesh_data"]["cases"]) if "cases" in data["mesh_data"] else set()
    with open(report, "w") as f:
        f.write("case,kept,nn_mean_mm,nn_p95_mm,nn_max_mm,centroid_offset_mm\n")
        for case, a in sorted(data["alignment"].items(), key=lambda kv: -kv[1]["mean"]):
            f.write(f"{case},{case in kept},{a['mean']:.4f},{a['p95']:.4f},{a['max']:.4f},{a['centroid_offset']:.4f}\n")
    print(f"Wrote alignment QC report -> {report}")

    if not args.no_sanity:
        saved = save_wss_sanity_pictures(data["mesh_data"], data["registered_data_list"], sanity_dir)
        overlays = save_canonical_alignment_overlay(data["mesh_data"], sanity_dir)
        print(f"Saved {len(saved)} WSS + {len(overlays)} overlay images to {sanity_dir}")
        from angioflow.data.angioflow_registration.check_knn_registration import save_knn_figures
        knn_dir = os.path.join(sanity_dir, "knn_check")
        n_knn = save_knn_figures(data, args.case_root, args.ghd_root, args.registry, knn_dir,
                                 n_subdivide=args.n_subdivide)
        print(f"Saved {n_knn} knn-registration figures to {knn_dir}")

        # raw CFD |WSS| vs the |WSS| read back out of the assembled dataset
        cmp_dir = os.path.join(sanity_dir, "cfd_vs_registered")
        n_cmp = save_cfd_vs_registered_wss(data, args.case_root, args.ghd_root, args.registry,
                                           cmp_dir, given_norm=given_norm,
                                           n_subdivide=args.n_subdivide,
                                           limit=args.n_wss_compare or None)
        print(f"Saved {len(n_cmp)} CFD-vs-registered WSS figures to {cmp_dir}")
    return data


if __name__ == "__main__":
    main()
