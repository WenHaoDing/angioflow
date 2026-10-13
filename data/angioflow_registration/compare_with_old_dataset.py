"""Validate that the new (real-patient) eval dataset lands in the SAME space as
the old synthetic training dataset -- checking what the MODEL actually consumes.

This matters because the whole point of the new dataset is out-of-distribution
evaluation of models trained on the old one: if the inputs sit in a different
position/orientation/scale, the model sees garbage and the metrics are
meaningless, regardless of whether the WSS values themselves are correct.

IMPORTANT: this compares the stored TENSOR channels (x/y/z + normals), not the
stored meshes. An earlier version of this script plotted
mesh_data['subdivided_ghd_Meshes'], which was always canonical-frame and always
correct -- so it looked fine even while the tensor's x/y/z channels were being
written in world frame. The model reads the tensor (args.x_labels =
['x','y','z','x_normal','y_normal','z_normal']), so the tensor is what must be
validated.

Both datasets are stored ALREADY NORMALIZED with the steady set's tensor_norm,
so we denormalize both before comparing in physical units (meters).

conda activate new
python -m angioflow.data.angioflow_registration.compare_with_old_dataset
"""
import os

import numpy as np
import torch

OLD_DATASET = "/media/yaplab2/wd8tb/wenhao/datasets/AneuG_Flow_stable/AneuG_processed/assembled_registered_data_1k_v5.pth"
NEW_DATASET = "/media/yaplab2/wd8tb/wenhao/datasets/aneux_type0_eval/assembled_registered_data_AneuX_type0.pth"
STEADY = "/media/yaplab2/wd8tb/wenhao/datasets/AneuG_Flow_stable/AneuG_processed/assembled_registered_steady_data_1k_v4.pth"
# renders live next to the eval dataset they check
OUT_DIR = "/media/yaplab2/wd8tb/wenhao/datasets/aneux_type0_eval/alignment_check"

N_PAIRS = 5
SEED = 0        # random case sampling; change for a different draw
ANEURYSM_TYPE = 0
XYZ = ["x", "y", "z"]
NRM = ["x_normal", "y_normal", "z_normal"]


def _denorm(tensor, labels, norm):
    """Undo the stored per-channel normalization, matching channels by name."""
    lut = {n: i for i, n in enumerate(norm["label"])}
    out = tensor.clone()
    for i, name in enumerate(labels):
        if name in lut:
            out[..., i] = out[..., i] * norm["std"][lut[name]] + norm["mean"][lut[name]]
    return out


def _channels(item, norm, names):
    labels = item["labels"]
    frame0 = _denorm(item["tensor"][0], labels, norm)
    return frame0[:, [labels.index(n) for n in names]]


def _fmt(t):
    return "[" + ", ".join(f"{v:+.5f}" for v in np.asarray(t).tolist()) + "]"


def _pv(points, faces):
    import pyvista as pv
    return pv.PolyData(np.asarray(points), np.hstack([np.full((faces.shape[0], 1), 3), faces]))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    steady = torch.load(STEADY, map_location="cpu", mmap=True)
    norm = {"mean": steady["tensor_norm"]["mean"].flatten(),
            "std": steady["tensor_norm"]["std"].flatten(),
            "label": list(steady["label"])}

    old = torch.load(OLD_DATASET, map_location="cpu", mmap=True)
    new = torch.load(NEW_DATASET, map_location="cpu")

    old_items = old["registered_data_list"]
    old_faces = old["mesh_data"]["faces_list"][0].numpy()
    # single-type builds are stored flat, exactly like the old dataset
    new_md, new_rl = new["mesh_data"], new["registered_data_list"]
    if isinstance(new_rl, dict):
        new_md, new_rl = new_md[ANEURYSM_TYPE], new_rl[ANEURYSM_TYPE]
    new_items = new_rl
    new_faces = new_md["faces_list"][0].numpy()

    # randomly sample one case from each dataset per pair
    rng = np.random.default_rng(SEED)
    n = min(N_PAIRS, len(new_items))
    old_pick = rng.choice(len(old_items), size=n, replace=False)
    new_pick = rng.choice(len(new_items), size=n, replace=False)

    print("=" * 72)
    print("TENSOR CHANNELS, denormalized to meters -- this is what the model reads")
    print("=" * 72)

    for tag, items, idxs in (("OLD", old_items, old_pick), ("NEW", new_items, new_pick)):
        xyz = torch.stack([_channels(items[int(i)], norm, XYZ) for i in idxs])
        nrm = torch.stack([_channels(items[int(i)], norm, NRM) for i in idxs])
        cen = xyz.mean(dim=1)
        print(f"\n{tag}  ({[items[int(i)]['case'] for i in idxs]})")
        print(f"  xyz centroid mean : {_fmt(cen.mean(0))}")
        print(f"  xyz vertex std    : {_fmt(xyz.reshape(-1, 3).std(0))}")
        print(f"  bbox extent mean  : {_fmt((xyz.max(1).values - xyz.min(1).values).mean(0))}")
        print(f"  normal magnitude  : {_fmt(nrm.reshape(-1, 3).norm(dim=-1).mean().unsqueeze(0))} (should be ~1)")

    # ---- pairwise renders from the TENSOR coordinates ----
    import pyvista as pv
    pv.set_jupyter_backend("static")
    try:
        pv.start_xvfb()
    except Exception:
        pass

    print()
    for k in range(n):
        o_item, n_item = old_items[int(old_pick[k])], new_items[int(new_pick[k])]
        o_xyz = _channels(o_item, norm, XYZ).numpy()
        n_xyz = _channels(n_item, norm, XYZ).numpy()

        p = pv.Plotter(off_screen=True, window_size=(800, 800))
        p.add_mesh(_pv(o_xyz, old_faces), color="blue", opacity=0.40, label=f"old: {o_item['case']}")
        p.add_mesh(_pv(n_xyz, new_faces), color="red", opacity=0.40, label=f"new: {n_item['case']}")
        p.add_legend()
        p.add_text(f"TENSOR x/y/z (denormalized)\nold {o_item['case']}  vs  new {n_item['case']}", font_size=9)
        p.show_axes()
        p.show_bounds(grid="front", location="outer", all_edges=True)
        out = os.path.join(OUT_DIR, f"tensor_pair{k + 1}_{o_item['case']}_vs_{n_item['case']}.png")
        p.screenshot(out)
        p.close()
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
