"""
Retrieve wall graphs from Fluent .msh files and save them as PyTorch .pt files for later use in training or evaluation.

Retrieve graphs:
retrieve_wall_graphs



"""
from __future__ import annotations

import glob
import os

from angioflow.data.mesh import save_wall_graph


def retrieve_wall_graphs(case_root, save_dir, zid=0, case_pattern="*", msh_glob="*.msh", overwrite=False):
    """For each case subfolder under `case_root`, find its .msh and save the wall graph
    as `<save_dir>/<case_name>.pt`.
    """
    os.makedirs(save_dir, exist_ok=True)
    for case_dir in sorted(glob.glob(os.path.join(case_root, case_pattern))):
        if not os.path.isdir(case_dir):
            continue
        name = os.path.basename(case_dir)
        out_path = os.path.join(save_dir, f"{name}.pt")
        if not overwrite and os.path.exists(out_path):
            print(f"[skip] {name}: already exists")
            continue
        msh_files = sorted(glob.glob(os.path.join(case_dir, msh_glob)))
        if not msh_files:
            print(f"[skip] {name}: no .msh found")
            continue
        try:
            save_wall_graph(msh_files[0], out_path, zid=zid)
            print(f"[ok] {name} <- {os.path.basename(msh_files[0])}")
        except Exception as e:
            print(f"[fail] {name}: {e}")


if __name__ == "__main__":
    retrieve_wall_graphs(
        case_root="/home/wenhao/AneuG-Net/angioflow_debug",
        save_dir="/home/wenhao/AneuG-Net/angioflow_debug/wall_graphs",
    )


"""
conda activate new
python -m angioflow.data.retrieve_graph_fluent

"""
