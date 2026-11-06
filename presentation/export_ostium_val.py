"""Export the ostium model's site + neck predictions on its validation cases (vessel page, CPU).

The model computation is AneuG's scripts/ostium/visualize_predictions.render_sample (its "full
chain"), on the run's own held-out cases without augmentation: the site distribution over the
graph points; --k sites drawn from it, with the neck the model draws at each (as at inference).
Per case writes <out>/ostium_val/<i>.bin, positions in mm, centred on the graph's mean point:
  pos f32[N,3]  prob f32[N]  ring_true f32[64,3]  site f32[k,3]  ring f32[k,64,3]
and ostium_val/cases.json (case, N, k, distance of each sampled site to the true one). Seeded per case.

    python -m angioflow.presentation.export_ostium_val
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ANEUG = "/media/yaplab2/HDD Storage/wenhao/AneuG"
RUN = f"{ANEUG}/runtime_train/ostium_model/abl_mlp_crop_rot15_selfrim_graph_mix_morph_w2_drop"
OUT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_showcase"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=f"{RUN}/epoch_0500.pth")
    p.add_argument("--k", type=int, default=5, help="sites drawn per case")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    sys.path[:0] = [ANEUG, f"{ANEUG}/scripts/train"]
    os.chdir(ANEUG)                                  # AneuG resolves some paths from the repo root
    import torch
    from torch_geometric.utils import softmax
    import train_ostium_model as T
    from dataset.ostium_dataset import OstiumGraphDataset, make_loader
    from models.ostium_model import MorphOstiumModel, OstiumModel

    dev = torch.device("cpu")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    args = ck["args"]
    kw = dict(d=args["d"], mode=args["mode"], with_complex=not args.get("no_complex", False),
              norm_tokens=bool(args.get("norm_tokens", 0)), flow_h=args.get("flow_h", 256),
              flow_blocks=args.get("flow_blocks", 4))
    if args.get("morph_cond"):
        ref = json.load(open(f"{ANEUG}/dataset/ostium_morph_reference.json"))
        model = MorphOstiumModel(morph_mean=ref["mean"], morph_std=ref["std"], **kw)
    else:
        model = OstiumModel(**kw)
    model.load_state_dict(ck["model"])
    model.eval()
    ys = T.Stats(torch.zeros(2, 33)); ys.mean, ys.std = ck["y_stats"]; ys.to(dev)
    vnames = [r["case"] for r in json.load(open(args["val_cases"]))["cases"]]
    ds = OstiumGraphDataset("val", augment=False, complex_targets=False, val_cases=vnames, point_frames=True,
                            morph=bool(args.get("morph_cond")), seed=args.get("seed", 0))
    unit = ds.unit
    odir = os.path.join(a.out, "ostium_val")
    os.makedirs(odir, exist_ok=True)
    for f in os.listdir(odir):
        os.remove(os.path.join(odir, f))
    manifest = []
    with torch.no_grad():
        for i, b in enumerate(make_loader(ds, 1, False, 0)):
            torch.manual_seed(a.seed + i)
            enc = model.enc(b)
            pr = softmax(model.site_logits(enc, b), b.batch)
            sites, rings = [], []
            for _ in range(a.k):
                idx = T.sample_sites(pr, b.batch, 1)
                tg, sc = b.tangent_pts[idx], b.scale_pts[idx]
                y = ys.inv(model.ost.sample(model.ostium_condition(enc, b, idx, tg, sc)))
                sites.append(b.pos[idx][0].numpy() * unit)
                rings.append(T.decode_ring(y, b.pos[idx], tg, sc)[0].numpy() * unit)
            pos = b.pos.numpy() * unit
            ring_true = b.ring_img[0].numpy() * unit
            with open(os.path.join(odir, f"{i:02d}.bin"), "wb") as f:
                for x in (pos, pr.numpy(), ring_true, np.stack(sites), np.stack(rings)):
                    f.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
            st = pos[int(b.site_true[0])]
            d = [float(np.linalg.norm(s - st)) for s in sites]
            manifest.append(dict(i=i, case=b.case[0], atype=int(b.atype[0]), N=len(pos), k=a.k, site_dist_mm=d,
                                 real_neck_span_mm=float(np.linalg.norm(ring_true[:, None] - ring_true[None], axis=2).max())))
            print(f"  {i:02d} {b.case[0]}: sampled sites {np.round(d, 1)} mm from the real one", flush=True)
    with open(os.path.join(odir, "cases.json"), "w") as f:
        json.dump(dict(run=os.path.basename(os.path.dirname(a.ckpt)), epoch=int(ck.get("epoch", 0)), cases=manifest), f, indent=1)
    print(f"wrote {odir}")


if __name__ == "__main__":
    main()
