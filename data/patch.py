"""
Patch raw CFD dataset with necessary checkpoint files.
"""
import glob
import os
import shutil

import pandas as pd


class GHD_CHKPatcher(object):
    def __init__(self,
                 ghd_root,
                 cfd_root,
                 case_manager_path,
                 file_list=("ghd_coefficients.npz", "landmarks.npz", "metrics.json"),
                 ):
        self.ghd_roots = [ghd_root] if isinstance(ghd_root, str) else list(ghd_root)
        self.cfd_root = cfd_root
        self.file_list = list(file_list)
        self.case_manager = pd.read_excel(
            case_manager_path,
            dtype={"index": int, "original_name": str, "case_dir": str},
        )
        self._name_by_index = dict(zip(self.case_manager["index"], self.case_manager["original_name"]))

    def patch_files(self):
        for cfd_case in sorted(glob.glob(os.path.join(self.cfd_root, "case_*"))):
            name = os.path.basename(cfd_case)
            try:
                idx = int(name.split("_")[1])
            except (IndexError, ValueError):
                print(f"[skip] {name}: cannot parse index")
                continue

            original_name = self._name_by_index.get(idx)
            if original_name is None:
                print(f"[skip] {name}: index {idx} not in case manager")
                continue

            ghd_case = next(
                (os.path.join(r, original_name) for r in self.ghd_roots
                 if os.path.isdir(os.path.join(r, original_name))),
                None,
            )
            if ghd_case is None:
                print(f"[skip] {name}: GHD dir for {original_name} not found in any ghd_root")
                continue

            dst_dir = os.path.join(cfd_case, "Results")
            if not os.path.isdir(dst_dir):
                print(f"[skip] {name}: Results dir missing at {dst_dir}")
                continue
            for fname in self.file_list:
                src = os.path.join(ghd_case, fname)
                if not os.path.isfile(src):
                    print(f"[miss] {name}: {fname} not in {ghd_case}")
                    continue
                shutil.copy2(src, os.path.join(dst_dir, fname))
            # log full mapping so wrong-dir copies are easy to spot
            print(f"[ok] cfd={name} idx={idx} xlsx_name={original_name!r} "
                  f"found_folder={os.path.basename(ghd_case)!r} src_dir={ghd_case}")


if __name__ == "__main__":
    patcher = GHD_CHKPatcher(
        ghd_root=["/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/ImperialNHS",
                  "/media/yaplab2/HDD Storage/almaha/Aneu_GHD/Fitting_Results_Final/AnueX_1",
                  ],
        cfd_root="/home/wenhao/AneuG-Net/angioflow_debug",
        case_manager_path="/media/yaplab2/HDD Storage/wenhao/AneuSeg/cfd_meshing/case_registry.xlsx",
    )
    patcher.patch_files()



"""
conda activate new
python -m angioflow.data.patch

"""
