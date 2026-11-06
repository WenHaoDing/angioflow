"""Export AneuG's vessel-conditioned generation results for the second presentation page.

Reads an ostium inference run (AneuG scripts/ostium/run_inference.py + run_merge.py) and the
graph folder it ran on; nothing is re-run. Per case, in the graph/mesh frame (mm), centred on
the patch's artery centroid:
  <out>/vessel/<case>.bin
    vessel_v f32[nv,3], vessel_f u32[nf,3]     healthy artery surface (marching cubes of patch_seg > 0)
    graph f32[N,3], score f32[N]               1D graph points and the site model's score (normalised to max 1)
    edges u32[N-1,2]                           graph edges (each once)
  <out>/vessel/<case>_s<k>.bin, per shown sample k
    ring f32[64,3]                             generated ostium (neck) ring
    cx_v f32[nc,3], cx_f u32[ncf,3], p_dome f32[nc]   generated complex (dome + parent vessel), dome probability
    mg_v f32[nm,3], mg_f u32[nmf,3], mg_lab u8[ceil4(nm)]   merged voxel mask surface, 1 artery / 2 new aneurysm
  <out>/vessel/cases.json
Samples shown: merged and accepted, without merge flags, preferring ones whose inference
checks were "ok"; --n_samples per case.

    python -m angioflow.presentation.export_vessel_cond --cases 1KPkS2Uocy_healthy,JFtZYj7T3t_healthy
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

INFER = "/media/yaplab2/HDD Storage/wenhao/AneuG/runtime_infer/ostium/sitetest_model"
GRAPHS = "/media/yaplab2/HDD Storage/wenhao/image-generation/debugging/graph1d_aneug_test"
OUT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_showcase"


def pad4(a):
    out = np.zeros(-(-len(a) // 4) * 4, np.uint8)
    out[:len(a)] = a
    return out


def surface(mask_zyx, origin, spacing, sigma=0.8):
    """Smooth surface of a (z, y, x) mask in mm: marching cubes at 0.5 of the blurred mask."""
    m = np.pad(mask_zyx.astype(np.float32), 2)
    v, f, _, _ = marching_cubes(gaussian_filter(m, sigma), 0.5)
    xyz = (v[:, ::-1] - 2) * spacing + origin                    # (z, y, x) voxel -> (x, y, z) mm
    return xyz, f[:, ::-1].copy()                                 # flip winding with the axis swap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", required=True)
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--infer", default=INFER)
    p.add_argument("--graphs", default=GRAPHS)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    odir = os.path.join(a.out, "vessel")
    os.makedirs(odir, exist_ok=True)
    merges = {(r["case"], r["sample"]): r for r in json.load(open(os.path.join(a.infer, "merge_summary.json")))}
    manifest = []
    for case in a.cases.split(","):
        gdir, idir = os.path.join(a.graphs, case), os.path.join(a.infer, case)
        meta = json.load(open(os.path.join(gdir, "meta.json")))
        origin, spacing = np.array(meta["origin_xyz"]), np.array(meta["spacing_xyz"])
        g = np.load(os.path.join(gdir, "graph.npz"))
        seg = np.load(os.path.join(gdir, "patch_seg.npy"))
        vv, vf = surface(seg == 1, origin, spacing)
        ctr = vv.mean(0)
        score = np.load(os.path.join(idir, "site_scores.npy"))[0]
        score = score / score.max()
        ei = g["edge_index"]
        edges = ei[:, ei[0] < ei[1]].T
        with open(os.path.join(odir, f"{case}.bin"), "wb") as f:
            for x in ((vv - ctr).astype(np.float32), vf.astype(np.uint32), (g["pos"] - ctr).astype(np.float32),
                      score.astype(np.float32), edges.astype(np.uint32)):
                f.write(np.ascontiguousarray(x).tobytes())
        info = json.load(open(os.path.join(idir, "case.json")))
        ok = {s["sample"] for s in info["samples"] if s["flags"] in ([], ["ok"])}
        cand = [k for (c, k), r in sorted(merges.items()) if c == case and r.get("accepted") and not r.get("flags")]
        cand = sorted(cand, key=lambda k: (k not in ok, k))[:a.n_samples]
        samples = []
        for k in cand:
            s = np.load(os.path.join(idir, f"sample_{k:02d}.npz"))
            mg = np.load(os.path.join(idir, f"merged_{k:02d}.npz"))["seg"]
            mv, mf = surface(mg > 0, origin, spacing)
            # label each surface vertex by the nearest labelled voxel
            zyx = np.argwhere(mg > 0)
            lab = mg[tuple(zyx.T)][cKDTree(zyx[:, ::-1] * spacing + origin).query(mv)[1]]
            with open(os.path.join(odir, f"{case}_s{k}.bin"), "wb") as f:
                for x in ((s["ring_mm"] - ctr).astype(np.float32), (s["verts_mm"] - ctr).astype(np.float32),
                          s["faces"].astype(np.uint32), s["p_dome"].astype(np.float32),
                          (mv - ctr).astype(np.float32), mf.astype(np.uint32), pad4(lab.astype(np.uint8))):
                    f.write(np.ascontiguousarray(x).tobytes())
            r = merges[(case, k)]
            samples.append(dict(sample=k, nr=len(s["ring_mm"]), nc=len(s["verts_mm"]), ncf=len(s["faces"]),
                                nm=len(mv), nmf=len(mf), atype=int(s["atype"]),
                                site=int(s["site"]), sac_volume_mm3=r["sac_volume_mm3"], inference_ok=k in ok))
        manifest.append(dict(case=case, group=meta["group"], nv=len(vv), nf=len(vf), N=len(g["pos"]),
                             ne=len(edges), n_generated=len(info["samples"]),
                             n_accepted=sum(1 for (c, _), r in merges.items() if c == case and r.get("accepted")),
                             samples=samples))
        print(f"  {case}: vessel {len(vv)} v, graph {len(g['pos'])} pts, samples {cand} "
              f"(ok {sorted(ok & set(cand))})", flush=True)
    with open(os.path.join(odir, "cases.json"), "w") as f:
        json.dump({"cases": manifest}, f, indent=1)
    print(f"wrote {odir}")


if __name__ == "__main__":
    main()
