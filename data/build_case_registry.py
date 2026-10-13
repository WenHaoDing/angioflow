"""Build a logging registry for angioflowv2_merged: AneuX vs ImperialNHS, plus
the original (pre-truncation) case name for AneuX cases, looked up by matching
against the folder names in PostTr_debug/AneuX.

A case is classified as ImperialNHS if its folder contains geomagic_processed.obj
(only ImperialNHS cases carry that file), otherwise AneuX.
"""

import os
import re
import numpy as np
import pandas as pd

MERGED_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged"
ANEUX_POST_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_27072026"
IMPERIALNHS_TYPE_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"
OUTPUT_PATH = os.path.join(MERGED_ROOT, "case_registry.xlsx")


def _load_aneurysm_type(endpoints_path):
    if not os.path.exists(endpoints_path):
        return "N/A"
    try:
        return int(np.load(endpoints_path, allow_pickle=True).item()["aneurysm_type"])
    except Exception:
        return "N/A"


def _is_case_dir(path):
    return os.path.exists(os.path.join(path, "wall_data.pt")) or os.path.exists(os.path.join(path, "blood_data.pt"))


def _match_aneux_original_name(case_name, post_names):
    for name in post_names:
        if name.startswith(case_name):
            return name
    # ANSYS cases were reformatted (e.g. "ANSYS09" from "ANSYS_UNIGE_09_cut1");
    # fall back to matching the trailing digits against an "ANSYS_UNIGE_<digits>_" folder.
    m = re.match(r"ANSYS0*([0-9]+)$", case_name)
    if m:
        digits = m.group(1)
        for name in post_names:
            if re.match(rf"ANSYS_UNIGE_0*{digits}_", name):
                return name
    return None


def build_registry(merged_root=MERGED_ROOT, aneux_post_root=ANEUX_POST_ROOT):
    post_names = sorted(os.listdir(aneux_post_root))

    rows = []
    for case in sorted(os.listdir(merged_root)):
        case_path = os.path.join(merged_root, case)
        if not os.path.isdir(case_path) or not _is_case_dir(case_path):
            continue

        dataset = "ImperialNHS" if os.path.exists(os.path.join(case_path, "geomagic_processed.obj")) else "AneuX"

        original_name = ""
        post_dir = ""
        if dataset == "AneuX":
            matched = _match_aneux_original_name(case, post_names)
            if matched is not None:
                original_name = matched
                post_dir = os.path.join(aneux_post_root, matched)
            aneurysm_type = _load_aneurysm_type(os.path.join(post_dir, "endpoints_manual.npy")) if post_dir else "N/A"
        else:
            original_name = case  # ImperialNHS folders are already named by original_name
            aneurysm_type = _load_aneurysm_type(os.path.join(IMPERIALNHS_TYPE_ROOT, case, "endpoints_manual.npy"))

        rows.append({
            "case": case,
            "dataset": dataset,
            "original_name": original_name,
            "aneurysm_type": aneurysm_type,
            "case_dir": case_path,
            "post_dir": post_dir,
        })

    df = pd.DataFrame(rows, columns=["case", "dataset", "original_name", "aneurysm_type", "case_dir", "post_dir"])
    df.to_excel(OUTPUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_registry()
    print(f"Wrote {len(df)} cases to {OUTPUT_PATH}")
    print(df["dataset"].value_counts())
    aneux = df[df["dataset"] == "AneuX"]
    print(f"AneuX cases matched to an original name: {(aneux['original_name'] != '').sum()} / {len(aneux)}")
    print(f"Cases with a known aneurysm_type: {(df['aneurysm_type'] != 'N/A').sum()} / {len(df)}")

"""
conda activate new
python -m angioflow.data.build_case_registry

"""
