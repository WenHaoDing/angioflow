"""Export AneuG v2 generation examples for the presentation site (CPU, a few seconds).

For a few cases of AneuG's generated cohort (runtime_dataset/generation/cohort_650) this
re-runs the graph-transformer morphology sensor exactly as gen_merged_cohort.py did
(SensorUncapper.uncap on the case's ghd.npy, the deterministic mean) and writes:
  <out>/aneug/<case>.bin
    verts f32[nv,3] mm        the generated aneurysm complex (dome + vessel stubs)
    faces u32[nf,3]           its closed template surface
    label u8[ceil4(nv)]       0 vessel wall, 1 aneurysm dome, 2 inlet cap, 3 outlet 1 cap, 4 outlet 2 cap
    keep  u8[ceil4(nf)]       1 = face kept after uncapping (faces wholly inside a cap are cut)
    curve f32[nb,128,3] mm    the branch transformer's raw centerlines, generation order
                              (inlet first, then the outlets: one token sequence)
  <out>/aneug/cases.json      per case: type, nv, nf, nb, n per curve
Frame: complex centroid at the origin, mm.

    python -m angioflow.presentation.export_aneug --cases cohort_650_0000_t0,cohort_650_0325_t1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ANEUG = "/media/yaplab2/HDD Storage/wenhao/AneuG"
COHORT = f"{ANEUG}/runtime_dataset/generation/cohort_650"
OUT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_showcase"


def pad4(a):
    out = np.zeros(-(-len(a) // 4) * 4, np.uint8)
    out[:len(a)] = a
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", required=True, help="comma-separated cohort_650 case folders")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    sys.path.insert(0, ANEUG)                       # AneuG's own top-level packages (models, utils, ...)
    from models.morphoformer import SensorUncapper
    from models.multi_canonical_ghd_reconstruct import MultiCanonicalGHDReconstruct

    multi = MultiCanonicalGHDReconstruct(f"{ANEUG}/dataset/canonical", device="cpu")
    sensors = {}
    odir = os.path.join(args.out, "aneug")
    os.makedirs(odir, exist_ok=True)
    manifest = []
    for case in args.cases.split(","):
        cdir = os.path.join(COHORT, case)
        meta = json.load(open(os.path.join(cdir, "meta.json")))
        t = int(meta["aneurysm_type"])
        if meta["sensor"] not in sensors:
            sensors[meta["sensor"]] = SensorUncapper.from_checkpoint(meta["sensor"], multi, device="cpu")
        u = sensors[meta["sensor"]].uncap(np.load(os.path.join(cdir, "ghd.npy")), t)
        verts = u["verts"].astype(np.float64)
        faces = multi.get(t).canonical_Meshes.faces_packed().cpu().numpy()
        label = np.zeros(len(verts), np.uint8)
        if u["dome"] is not None:
            label[u["dome"]] = 1
        for b, cap in enumerate(u["cap_masks"]):
            label[cap] = 2 + b
        keep = ~u["cap_masks"].any(0)[faces].all(1)
        curves = [np.asarray(c, np.float64) for c in np.load(os.path.join(cdir, "centerlines.npy"), allow_pickle=True)]
        curves = [c for c in curves if len(c)]
        ctr = verts.mean(0)
        # the cohort's centerlines start at the sensor's start points: check this rerun matches
        gap = max(float(np.linalg.norm(c[0] - s)) for c, s in zip(curves, u["start_points"]))
        n = max(len(c) for c in curves)
        cv = np.full((len(curves), n, 3), np.nan, np.float32)
        for b, c in enumerate(curves):
            cv[b, :len(c)] = c - ctr
        with open(os.path.join(odir, f"{case}.bin"), "wb") as f:
            for a in ((verts - ctr).astype(np.float32), faces.astype(np.uint32), pad4(label), pad4(keep), cv):
                f.write(np.ascontiguousarray(a).tobytes())
        manifest.append(dict(case=case, aneurysm_type=t, nv=len(verts), nf=len(faces), nb=len(curves), n=n,
                             n_per_curve=[len(c) for c in curves],
                             arc_mm=[float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) for c in curves],
                             label_counts=np.bincount(label, minlength=5).tolist(), faces_cut=int((~keep).sum())))
        print(f"  {case} type {t}: nv={len(verts)} caps={[int(c.sum()) for c in u['cap_masks']]} "
              f"dome={int((label == 1).sum())} cut={int((~keep).sum())} curves={manifest[-1]['n_per_curve']} "
              f"start gap={gap:.4f} mm", flush=True)
    with open(os.path.join(odir, "cases.json"), "w") as f:
        json.dump({"layout": ["verts f32 nv*3", "faces u32 nf*3", "label u8 ceil4(nv)", "keep u8 ceil4(nf)",
                              "curve f32 nb*n*3 (NaN-padded)"], "cases": manifest}, f, indent=1)
    print(f"wrote {odir}")


if __name__ == "__main__":
    main()
