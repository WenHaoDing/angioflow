"""Build a renamed mirror of an AneuX geometry folder using simplify_name()."""
from __future__ import annotations

import os
import re
import shutil
import glob


def simplify_name(folder_name: str) -> str:
    """Map an original case folder name to its simplified output name.

    C0002_cut2                          -> C0002
    ANSYS_UNIGE_09_cut1                 -> ANSYS09
    SNF00000036_02_2_cut2               -> SNF036
    UPF_P0272.00_ID1_cut1               -> UPF272
    USFD_0005_cut1                      -> USFD005
    USFD_UNIGE_0034_cut1                -> USFD_U034
    p043_HAARCREcDAAQDQcbHgANDRQM_cut2  -> p043
    Anything else                       -> input unchanged
    """
    m = re.search(r"SNF0*(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"SNF{int(m.group(1)):03d}"
    m = re.search(r"UPF_P0*(\d+)\.", folder_name, re.IGNORECASE)
    if m:
        return f"UPF{int(m.group(1)):03d}"
    m = re.match(r"(p\d+)_", folder_name, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"ANSYS_UNIGE_0*(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"ANSYS{int(m.group(1)):02d}"
    m = re.match(r"USFD_UNIGE_0*(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"USFD_U{int(m.group(1)):03d}"
    m = re.match(r"USFD_0*(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"USFD{int(m.group(1)):03d}"
    # Fallback: drop trailing _cutN suffix if present, else keep original.
    m = re.match(r"(.+?)(_cut\d+)$", folder_name)
    if m:
        return m.group(1)
    return folder_name


def build_renamed_mirror(src_root, dst_root, mode="symlink"):
    """Build `dst_root` mirroring `src_root` with each subfolder renamed via simplify_name.
    mode: "symlink" (default, cheap), "copy" (shutil.copytree), or "rename" (in-place mv).
    Collisions (multiple src dirs mapping to the same name) are reported and skipped.
    """
    assert mode in ("symlink", "copy", "rename")
    if mode != "rename":
        os.makedirs(dst_root, exist_ok=True)

    src_dirs = sorted(d for d in glob.glob(os.path.join(src_root, "*")) if os.path.isdir(d))
    target_to_src = {}
    for d in src_dirs:
        target_to_src.setdefault(simplify_name(os.path.basename(d)), []).append(d)

    n_done, n_skip = 0, 0
    for new_name, sources in target_to_src.items():
        if len(sources) > 1:
            print(f"[skip] collision -> {new_name}: {[os.path.basename(s) for s in sources]}")
            n_skip += 1
            continue
        src = sources[0]
        if mode == "rename":
            dst = os.path.join(src_root, new_name)
        else:
            dst = os.path.join(dst_root, new_name)
        if mode == "rename" and os.path.basename(src) == new_name:
            continue                                 # already named correctly, nothing to do in-place
        if os.path.exists(dst) or os.path.islink(dst):
            print(f"[skip] {new_name}: target exists")
            n_skip += 1
            continue
        if mode == "symlink":
            os.symlink(os.path.abspath(src), dst)
        elif mode == "copy":
            shutil.copytree(src, dst)
        else:
            os.rename(src, dst)
        n_done += 1
        print(f"[{mode}] {os.path.basename(src)} -> {new_name}")

    print(f"\n[done] {n_done} renamed, {n_skip} skipped, {len(src_dirs)} total")


if __name__ == "__main__":
    build_renamed_mirror(
        src_root="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX",
        dst_root="/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/AneuX_renamed",
        mode="symlink",
    )


"""
conda activate new
python -m angioflow.data.rename_geometry_folder

"""