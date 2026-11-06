"""Export metrics + 3D viewer data for the presentation site (presentation/site/).

Two models, AngioFlow cases only:
  wss    the WSS-only hierarchical GPS model         (WSS on the wall)
  joint  the dual-branch joint model                  (WSS on the wall + velocity in the lumen)

Stage "metrics": every AngioFlow case (train and test) through each model, evaluated exactly
like training/eval_wss.py and training/eval_joint.py (fixed pose, the run's own random crop
depth seeded with --seed, full wall resolution). Each model is scored on its OWN split
(WSS-only: 10% AngioFlow test, joint: 20%). Writes metrics_<model>.csv and summary.json
(mean and median per-case relative L2, by split and aneurysm type).

Stage "viz": --n_viz_per_type test cases per aneurysm type that NEITHER model trained on,
picked as the cases closest to their type's median error (not the best ones). Both models
run at the deepest training crop (p = mask_high = 0.25) so they predict on the same wall
nodes. Per case writes <out>/viz/<case>.bin and appends to viz/cases.json:
  verts f32[nv,3] mm   faces u32[nf,3]   wss_cfd / wss_wssonly / wss_joint f32[nv] (|WSS|, Pa)
  focused u8[nv] (padded to 4 bytes)     pts f32[np,3] mm   vel_cfd / vel_joint f32[np,3] m/s
Frame: cropped-wall centroid at the origin, aneurysm pointing +y.

    CUDA_VISIBLE_DEVICES=1 python -m angioflow.presentation.export_showcase --stage all --mem_frac 0.3
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch_geometric.data import Batch

CKPT_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_checkpoints"
RUNS = {
    "wss": "hier_gps_p1_e128_npre2_gps8_npost4_knn8_inter2_pr015_v3data_flowsplit",
    "joint": "joint_v3data_0frame_v123_e108f96_gps6_hop2_tok64_wq50_is20_blk3_xf_mh025_pressmax0_fs_warm_b4_lr3e-4_p20_angiotest",
}
DATA_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG"
WAVEFORM = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/_meta/waveform_yiying.txt"
OUT = "/media/yaplab2/wd8tb/wenhao/datasets/AngioFlow-PyG_showcase"
TYPE_NAMES = {0: "bifurcation", 1: "sidewall", 2: "sidewall, near bifurcation"}


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


class AddNodeId:
    """Tag wall nodes with their row in the stored static, so it survives the crop."""

    def __call__(self, data):
        wall = data[0] if isinstance(data, tuple) else data
        wall.node_id = torch.arange(wall.num_nodes)
        return data


class Model:
    """One trained run, its dataset (all versions, so the split is reproduced) and transforms."""

    def __init__(self, kind, device, which="best"):
        from angioflow.training.train import _chain, excluded_case_keys, load_waveform, split_by_case
        self.kind, self.device, self._chain = kind, device, _chain
        ckpt_path = os.path.join(CKPT_ROOT, RUNS[kind], f"{which}.pth")
        chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = argparse.Namespace(**chk["args"])
        a.save_dir, a.waveform_path = DATA_ROOT, WAVEFORM      # HF runs saved /tmp paths
        self.args, self.epoch = a, int(chk["epoch"])
        from angioflow.data.normalize import load_or_compute_stats
        if kind == "wss":
            from angioflow.training.eval_wss import build_dataset
            from angioflow.training.train import get_model
            from angioflow.transform import PosYNormalize
            self.ds = build_dataset(a, excluded_case_keys(a))
            tr, te = split_by_case(self.ds, a.train_frac, seed=a.seed)
        else:
            from angioflow.data.normalize_interior import load_or_compute_interior_stats
            from angioflow.training.train_joint import build_dataset, build_model, interior_stats_path, test_versions
            from angioflow.transform import JointNormalize
            self.ds = build_dataset(a, excluded_case_keys(a))
            tr, te = split_by_case(self.ds, a.train_frac, seed=a.seed, test_versions=test_versions(a))
        train_keys = {self.ds._index[i][:2] for i in tr}
        stats = load_or_compute_stats(os.path.join(a.save_dir, "stats.pt"), self.ds, train_keys,
                                      n_cases=a.stats_n_cases, frames_per_case=a.stats_frames_per_case, seed=a.seed)
        if kind == "wss":
            self.norm = PosYNormalize(stats)
            self.model = get_model(a, stats, device)
        else:
            self.press_zero_max = getattr(a, "press_zero_max", False)
            ist = load_or_compute_interior_stats(interior_stats_path(a), self.ds, train_keys,
                                                 n_cases=a.stats_n_cases, frames_per_case=a.stats_frames_per_case,
                                                 seed=a.seed, press_zero_max=self.press_zero_max)
            self.norm = JointNormalize(stats, ist)
            self.model = build_model(a, device)
        self.model.load_state_dict(chk["model"])
        self.model.eval()
        if kind == "joint":
            self.model.query_ratio = 1.0
        self.waveform = load_waveform(a.waveform_path, length=1000, device=device)
        self.split = {}
        for idxs, name in ((tr, "train"), (te, "test")):
            for i in idxs:
                v, c, _ = self.ds._index[i]
                if v == "angioflow":
                    self.split[c] = (i, name)
        self.set_crop(None)
        print(f"[{kind}] {ckpt_path} epoch {self.epoch}: AngioFlow "
              f"{sum(s == 'train' for _, s in self.split.values())} train / "
              f"{sum(s == 'test' for _, s in self.split.values())} test", flush=True)

    def set_crop(self, p):
        """p=None: the run's own random crop depth U[mask_low, mask_high]; else fixed depth p."""
        from angioflow.transform import AneurysmCrop, CenterAndRandomRotate, SymmetrizeEdges
        a = self.args
        lo, hi = (a.mask_low, a.mask_high) if p is None else (p, p)
        if self.kind == "wss":
            crop = AneurysmCrop(low=lo, high=hi, aneurysm_only=a.crop_aneurysm_only)
            self.ds.transform = self._chain(AddNodeId(), CenterAndRandomRotate(rotate=False), self.norm,
                                            crop, SymmetrizeEdges())
        else:
            from angioflow.training.train_joint import _WallOnly
            from angioflow.transform import JointAneurysmCrop, PressureZeroMax
            chain = [AddNodeId()] + ([PressureZeroMax()] if self.press_zero_max else []) + [
                JointAneurysmCrop(low=lo, high=hi, aneurysm_only=a.crop_aneurysm_only,
                                  factor=a.crop_factor, slack=a.crop_slack),
                CenterAndRandomRotate(rotate=False), self.norm, _WallOnly(SymmetrizeEdges())]
            self.ds.transform = self._chain(*chain)

    @torch.no_grad()
    def run(self, idx):
        """Denormalized fields for one case: wall node ids, WSS gt/pred [n,3] (Pa), and for the
        joint model interior pos (static frame, m) + velocity gt/pred [m,3] (m/s)."""
        from angioflow.training.train import _unpack_pred
        d = self.device
        if self.kind == "wss":
            wall = Batch.from_data_list([self.ds[idx]]).to(d)
            pred, q = _unpack_pred(self.model(wall, self.waveform))
            assert q is None
            out = dict(node_id=wall.node_id.cpu().numpy(),
                       wss_gt=self.norm.inverse_y(wall.y).cpu().numpy(),
                       wss_pr=self.norm.inverse_y(pred).cpu().numpy())
            return out
        wall, interior = self.ds[idx]
        wall = Batch.from_data_list([wall]).to(d)
        interior = Batch.from_data_list([interior]).to(d)
        wss, q, vel, _ = self.model(wall, interior, self.waveform)
        assert q is None
        ps = self.norm.pos_std.to(d)
        return dict(node_id=wall.node_id.cpu().numpy(),
                    wall_pos=(wall.pos * ps).cpu().numpy(), int_pos=(interior.pos * ps).cpu().numpy(),
                    wss_gt=self.norm.inverse_y(wall.y).cpu().numpy(), wss_pr=self.norm.inverse_y(wss).cpu().numpy(),
                    vel_gt=self.norm.inverse_vel(interior.vel).cpu().numpy(),
                    vel_pr=self.norm.inverse_vel(vel).cpu().numpy())

    def free(self):
        del self.model
        torch.cuda.empty_cache()


def case_type(ds, cid):
    return int(ds._load_static("angioflow", cid)["aneurysm_type"])


# ---------------------------------------------------------------- metrics
def stage_metrics(args, device):
    for kind in args.models:
        m = Model(kind, device)
        torch.manual_seed(args.seed)          # crop depth + FPS start, as the eval scripts do
        rows = []
        items = sorted(m.split.items())[:args.limit]
        for n, (cid, (idx, split)) in enumerate(items):
            o = m.run(idx)
            r = dict(model=kind, epoch=m.epoch, case=cid, split=split, aneurysm_type=case_type(m.ds, cid),
                     n_wall=len(o["node_id"]), wss_rel_l2=rel_l2(o["wss_pr"], o["wss_gt"]))
            if kind == "joint":
                r.update(n_interior=len(o["vel_gt"]), vel_rel_l2=rel_l2(o["vel_pr"], o["vel_gt"]))
            rows.append(r)
            if n % 50 == 0:
                print(f"  [{kind}] {n}/{len(m.split)} {cid} {split} " +
                      " ".join(f"{k}={v:.3f}" for k, v in r.items() if k.endswith("rel_l2")), flush=True)
        path = os.path.join(args.out, f"metrics_{kind}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[{kind}] wrote {path}", flush=True)
        m.free()
    summarize(args.out)


def summarize(out):
    summary = {"type_names": TYPE_NAMES, "models": {}}
    for kind in RUNS:
        path = os.path.join(out, f"metrics_{kind}.csv")
        if not os.path.exists(path):
            continue
        rows = list(csv.DictReader(open(path)))
        keys = [k for k in rows[0] if k.endswith("rel_l2")]
        groups = {}
        for split in ("train", "test"):
            for t in ["all", "0", "1", "2"]:
                sel = [r for r in rows if r["split"] == split and (t == "all" or r["aneurysm_type"] == t)]
                g = {"n": len(sel)}
                for k in keys:
                    v = np.array([float(r[k]) for r in sel])
                    if len(v):
                        g[k] = {"mean": float(v.mean()), "median": float(np.median(v)),
                                "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}
                groups[f"{split}/{t}"] = g
        summary["models"][kind] = {"run": RUNS[kind], "epoch": int(rows[0]["epoch"]), "groups": groups}
        print(f"\n[{kind}] epoch {rows[0]['epoch']}  mean (median) per-case relative L2")
        for g, v in groups.items():
            print(f"  {g:8s} n={v['n']:4d}  " + "  ".join(
                f"{k}={v[k]['mean']:.3f} ({v[k]['median']:.3f})" for k in keys if k in v))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


# ---------------------------------------------------------------- viz
def pick_cases(out, n_per_type):
    """Test cases unseen by both models, closest to their type's median error."""
    rows = {k: {r["case"]: r for r in csv.DictReader(open(os.path.join(out, f"metrics_{k}.csv")))} for k in RUNS}
    both = [c for c in rows["wss"] if rows["wss"][c]["split"] == "test" and rows["joint"][c]["split"] == "test"]
    picks = []
    for t in ("0", "1", "2"):
        cs = [c for c in both if rows["wss"][c]["aneurysm_type"] == t]
        if not cs:
            continue
        errs = np.array([[float(rows["wss"][c]["wss_rel_l2"]), float(rows["joint"][c]["vel_rel_l2"])] for c in cs])
        z = np.abs(errs / np.median(errs, 0) - 1).sum(1)        # distance from the type median
        picks += [(c, int(t)) for c in np.array(cs)[np.argsort(z)[:n_per_type]]]
    return picks, rows


def aneurysm_up(points, focused):
    """Rotation taking (focused centroid - wall centroid) to +y."""
    a = points[focused].mean(0) - points.mean(0)
    a /= np.linalg.norm(a)
    y = np.array([0.0, 1.0, 0.0])
    v, c = np.cross(a, y), float(a @ y)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def stage_viz(args, device):
    picks, rows = pick_cases(args.out, args.n_viz_per_type)
    print(f"[viz] {len(picks)} cases: {picks}", flush=True)
    res = {c: {} for c, _ in picks}
    for kind in RUNS:
        m = Model(kind, device)
        m.set_crop(m.args.mask_high)
        for c, _ in picks:
            torch.manual_seed(args.seed)
            res[c][kind] = m.run(m.split[c][0])
        ds = m.ds
        m.free()
    vdir = os.path.join(args.out, "viz")
    os.makedirs(vdir, exist_ok=True)
    rng = np.random.default_rng(0)
    manifest = []
    for c, t in picks:
        w, j = res[c]["wss"], res[c]["joint"]
        assert np.array_equal(w["node_id"], j["node_id"]), f"{c}: the two crops differ"
        s = ds._load_static("angioflow", c)
        pos_all = s["pos"].double().numpy()
        nid = j["node_id"]
        keep = np.zeros(len(pos_all), bool)
        keep[nid] = True
        remap = -np.ones(len(pos_all), np.int64)
        remap[nid] = np.arange(len(nid))
        faces = s["faces"].numpy()
        faces = remap[faces[keep[faces].all(1)]]
        wall = pos_all[nid]
        # the joint chain centred on its cropped wall: undo that shift for the interior points
        shift = wall.mean(0) - j["wall_pos"].mean(0)
        inner = j["int_pos"].astype(np.float64) + shift
        focused = s["node_type"].numpy()[nid] == 1
        R = aneurysm_up(wall, focused)
        ctr = wall.mean(0)
        verts = ((wall - ctr) @ R.T * 1e3).astype(np.float32)
        sel = np.sort(rng.choice(len(inner), size=min(args.n_points, len(inner)), replace=False))
        pts = ((inner[sel] - ctr) @ R.T * 1e3).astype(np.float32)
        vel_cfd = (j["vel_gt"][sel] @ R.T).astype(np.float32)
        vel_joint = (j["vel_pr"][sel] @ R.T).astype(np.float32)
        mag = lambda x: np.linalg.norm(x, axis=1).astype(np.float32)
        wss = dict(wss_cfd=mag(w["wss_gt"]), wss_wssonly=mag(w["wss_pr"]), wss_joint=mag(j["wss_pr"]))
        foc = np.zeros(-(-len(nid) // 4) * 4, np.uint8)
        foc[:len(nid)] = focused
        parts = [verts, faces.astype(np.uint32), wss["wss_cfd"], wss["wss_wssonly"], wss["wss_joint"], foc,
                 pts, vel_cfd, vel_joint]
        with open(os.path.join(vdir, f"{c}.bin"), "wb") as f:
            for p in parts:
                f.write(np.ascontiguousarray(p).tobytes())
        manifest.append(dict(
            case=c, aneurysm_type=t, nv=len(verts), nf=len(faces), np=len(pts),
            n_interior_full=len(inner),
            extent_mm=float(np.ptp(verts, 0).max()),
            wss_p99=float(np.percentile(wss["wss_cfd"], 99)), vel_p99=float(np.percentile(mag(vel_cfd), 99)),
            # errors on this crop (p = 0.25, full interior), and the case's metrics-stage numbers
            wss_rel_l2_wssonly=rel_l2(w["wss_pr"], w["wss_gt"]), wss_rel_l2_joint=rel_l2(j["wss_pr"], j["wss_gt"]),
            vel_rel_l2_joint=rel_l2(j["vel_pr"], j["vel_gt"]),
            metrics_stage=dict(wss_rel_l2_wssonly=float(rows["wss"][c]["wss_rel_l2"]),
                               wss_rel_l2_joint=float(rows["joint"][c]["wss_rel_l2"]),
                               vel_rel_l2_joint=float(rows["joint"][c]["vel_rel_l2"]))))
        print(f"  {c} type {t}: nv={len(verts)} nf={len(faces)} np={len(pts)}/{len(inner)}  "
              f"wss {manifest[-1]['wss_rel_l2_wssonly']:.3f}/{manifest[-1]['wss_rel_l2_joint']:.3f}  "
              f"vel {manifest[-1]['vel_rel_l2_joint']:.3f}", flush=True)
    with open(os.path.join(vdir, "cases.json"), "w") as f:
        json.dump({"layout": ["verts f32 nv*3", "faces u32 nf*3", "wss_cfd f32 nv", "wss_wssonly f32 nv",
                              "wss_joint f32 nv", "focused u8 ceil4(nv)", "pts f32 np*3",
                              "vel_cfd f32 np*3", "vel_joint f32 np*3"],
                   "cases": manifest}, f, indent=1)
    print(f"[viz] wrote {vdir}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["metrics", "viz", "all", "summary"], default="all")
    p.add_argument("--models", default="wss,joint")
    p.add_argument("--out", default=OUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mem_frac", type=float, default=None,
                   help="cap the CUDA allocator at this fraction of the device (GPU 1: <=0.3)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_viz_per_type", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="metrics on the first N cases only (smoke test)")
    p.add_argument("--n_points", type=int, default=80000, help="lumen points kept per viz case")
    args = p.parse_args()
    args.models = args.models.split(",")
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    if args.mem_frac and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac, device)
    if args.stage == "summary":
        summarize(args.out)
    if args.stage in ("metrics", "all"):
        stage_metrics(args, device)
    if args.stage in ("viz", "all"):
        stage_viz(args, device)


if __name__ == "__main__":
    main()
