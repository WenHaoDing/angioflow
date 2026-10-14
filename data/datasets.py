"""PyG Datasets for AneuG CFD cases."""
from __future__ import annotations
import glob
import os
from functools import lru_cache

import numpy as np
import torch
from scipy.spatial import cKDTree

from torch_geometric.data import Data, Dataset

from angioflow.data.graph import compute_pe_cot_eigenvector
from angioflow.data.per_opening_geo import per_opening_geo_dist
from angioflow.data.mesh import (face_to_edge_index, decimate_zone_mesh,
                                  decimate_fps_ballpivot, compute_vertex_normals)
from angioflow.data.dataset_tags import normalize_tag, split_tag_cid

PHYSICS_KEYS = ["x_wall_shear", "y_wall_shear", "z_wall_shear"]


class AngioFlowWallDataset(Dataset):
    """One Data per (case, timestep). Geometry/labels come from wall_mesh_registration.pt;
    WSS time series come from wall_data.pt (AneuG-Flow: flat dict, AngioFlow: nested under "wall" zone).
    Both line up exactly with registration `pos` — no remapping needed.
    """

    def __init__(self, case_root_angioflow=None, case_root_aneug_flow=None, save_dir=None, n_frames=80,
                 lap_k=3, decimate_ratio=0.5, decimate_method="quadric",
                 case_pattern={"aneug_flow": "stable_*", "angioflow": "*"},
                 transform=None, pre_transform=None, redo=False, exclude_keys=None,
                 freeze_dataset=False, per_version_dirs=False, max_cases_per_version=None,
                 sanity_n_normal=5, sanity_z=3.5):
        assert save_dir is not None, "save_dir is required"
        assert decimate_method in ("quadric", "fps"), "decimate_method must be 'quadric' or 'fps'"
        self.case_roots = {"aneug_flow": case_root_aneug_flow, "angioflow": case_root_angioflow}
        self.case_pattern = case_pattern
        self.save_dir = save_dir
        # per_version_dirs: write/read static_*/wss_*.pt under <save_dir>/<version>/
        # instead of flat in <save_dir>. Opt-in only — existing save_dirs
        # (angioflow_pyg_full, angioflow_pyg_v1, angioflow_pyg_v2) are flat on disk; flipping the default
        # would make every already-processed case there look incomplete.
        self.per_version_dirs = per_version_dirs
        self.n_frames = n_frames
        self.lap_k = lap_k
        self.decimate_ratio = decimate_ratio
        self.decimate_method = decimate_method
        self.redo = redo
        self.exclude_keys = {tuple(k) for k in (exclude_keys or [])}
        # freeze_dataset: never call _process_case, even for newly-discovered case
        # dirs (e.g. mid-download) — only cases already fully processed on disk
        # are kept, so the dataset is guaranteed not to touch raw sim data.
        self.freeze_dataset = freeze_dataset
        # sanity renders (angioflow.visualize.data_sanity) for every outlier + sanity_n_normal
        # spread-out normal cases per dataset, only where missing; 0/None disables.
        self.sanity_n_normal = sanity_n_normal
        self.sanity_z = sanity_z

        # case_dir_by_key[(version, cid)] = case_dir.  cid = folder basename (str).
        self.case_dir_by_key = {}
        for version, root in self.case_roots.items():
            if root is None:
                continue
            for d in sorted(glob.glob(os.path.join(root, self.case_pattern[version]))):
                if not os.path.isdir(d):
                    continue
                key = (version, self._parse_case_id(d))
                if key in self.exclude_keys:
                    continue
                self.case_dir_by_key[key] = d
        if self.exclude_keys:
            print(f"[exclude] dropping {len(self.exclude_keys)} cases via exclude_keys")
        if max_cases_per_version is not None:
            counts = {}
            limited = {}
            for key, d in self.case_dir_by_key.items():
                version = key[0]
                if counts.get(version, 0) < max_cases_per_version:
                    limited[key] = d
                    counts[version] = counts.get(version, 0) + 1
            self.case_dir_by_key = limited
            print(f"[limit] capped to {max_cases_per_version} cases/version: "
                  f"{ {v: c for v, c in counts.items()} }")
        if self.freeze_dataset:
            n_before = len(self.case_dir_by_key)
            self.case_dir_by_key = {k: d for k, d in self.case_dir_by_key.items()
                                    if self._case_complete(*k)}
            print(f"[freeze] keeping {len(self.case_dir_by_key)}/{n_before} already-processed "
                  f"cases; new/incomplete cases will not be processed")
        self.case_keys = sorted(self.case_dir_by_key.keys())
        self.cases = [self.case_dir_by_key[k] for k in self.case_keys]

        os.makedirs(save_dir, exist_ok=True)
        if self.per_version_dirs:
            for version in self.case_roots:
                os.makedirs(os.path.join(save_dir, version), exist_ok=True)
        self._index = [(v, cid, t) for (v, cid) in self.case_keys for t in range(n_frames)]
        super().__init__(root=case_root_angioflow or case_root_aneug_flow,
                         transform=transform, pre_transform=pre_transform)
        self._rebuild_successful_index()
        # rank 0 only: DDP ranks each build the dataset and would race on the same files
        if self.sanity_n_normal and int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0:
            self.render_sanity()

    def render_sanity(self, render_all=False, redo=False, workers=1):
        """Render missing sanity images into <save_dir>/sanity/<version>/ -- see
        angioflow.visualize.data_sanity for what is selected and drawn."""
        from angioflow.visualize.data_sanity import render_dataset_sanity
        for version in sorted({v for v, _ in self.case_keys}):
            render_dataset_sanity(
                os.path.join(self.save_dir, "sanity"), version,
                [cid for v, cid in self.case_keys if v == version],
                static_path=lambda cid, v=version: self._static_path(v, cid),
                wss_path=lambda cid, t, v=version: self._wss_path(v, cid, t),
                n_frames=self.n_frames, interior=self._sanity_interior(version),
                n_normal=self.sanity_n_normal or 0, z_thresh=self.sanity_z,
                render_all=render_all, redo=redo, workers=workers)

    def _sanity_interior(self, version):
        """Interior file paths for render_sanity; None = wall-only dataset."""
        return None

    @staticmethod
    def _parse_case_id(case_dir):
        return os.path.basename(case_dir)

    @property
    def processed_dir(self):
        return self.save_dir

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        def _rel(version, name):
            return f"{version}/{name}" if self.per_version_dirs else name
        statics = [_rel(v, f"static_{v}_{cid}.pt") for (v, cid) in self.case_keys]
        wss = [_rel(v, f"wss_{v}_{cid}_{t}.pt") for (v, cid, t) in self._index]
        return statics + wss

    def _version_dir(self, version):
        return os.path.join(self.save_dir, version) if self.per_version_dirs else self.save_dir

    def _static_path(self, version, cid):
        return os.path.join(self._version_dir(version), f"static_{version}_{cid}.pt")

    def _wss_path(self, version, cid, t):
        return os.path.join(self._version_dir(version), f"wss_{version}_{cid}_{t}.pt")

    def download(self):
        pass

    def _case_complete(self, version, cid):
        return os.path.exists(self._static_path(version, cid)) and \
               all(os.path.exists(self._wss_path(version, cid, t)) for t in range(self.n_frames))

    def process(self):
        if self.freeze_dataset:
            return
        for (version, cid) in self.case_keys:
            if not self.redo and self._case_complete(version, cid):
                continue
            case_dir = self.case_dir_by_key[(version, cid)]
            reg_path = os.path.join(case_dir, "wall_mesh_registration.pt")
            if not os.path.exists(reg_path):
                print(f"[skip] {version}/{cid}: no wall_mesh_registration.pt")
                continue
            try:
                self._process_case(version, cid, case_dir, reg_path)
            except Exception as e:
                print(f"[skip] {version}/{cid}: {e}")

    def _rebuild_successful_index(self):
        """Keep only cases with a complete processed static + WSS frame set."""
        case_keys, cases, index = [], [], []
        for (version, cid) in self.case_keys:
            if not self._case_complete(version, cid):
                continue
            s = torch.load(self._static_path(version, cid), weights_only=False)
            n_nodes = s["pos"].shape[0]
            n_edges = s["edge_index"].shape[1] // 2
            # print(f"[loaded] {version}/{cid}: {n_nodes} nodes, {n_edges} edges")
            case_keys.append((version, cid))
            cases.append(self.case_dir_by_key[(version, cid)])
            index.extend((version, cid, t) for t in range(self.n_frames))
        self.case_keys = case_keys
        self.cases = cases
        self._index = index

    def _process_case(self, version, cid, case_dir, reg_path):
        reg = torch.load(reg_path, weights_only=False)
        pos = reg["pos"].numpy().astype(np.float64)
        faces = reg["faces"].numpy().astype(np.int64)

        # surface normals on full-res mesh (before decimation)
        normals = compute_vertex_normals(pos, faces)  # [N, 3]

        # decimate mesh; orig_idx maps each new vertex to its nearest original vertex
        if self.decimate_ratio >= 1.0:
            new_pos, new_faces = pos, faces
            orig_idx = np.arange(len(pos), dtype=np.int64)
        elif self.decimate_method == "fps":
            new_pos, new_faces, orig_idx = decimate_fps_ballpivot(
                pos, faces, ratio=self.decimate_ratio)
        else:
            new_pos, new_faces = decimate_zone_mesh(pos, faces, ratio=self.decimate_ratio)
            _, orig_idx = cKDTree(pos).query(new_pos, k=1)
            orig_idx = orig_idx.astype(np.int64)
        orig_idx_t = torch.from_numpy(orig_idx)

        # centralize per-case (pre-transform) on the un-decimated mesh mean so the
        # offset is deterministic and recoverable from reg["pos"] regardless of decimation
        new_pos = new_pos - pos.mean(axis=0, keepdims=True)

        new_edge_index = face_to_edge_index(new_faces)
        cot_eigval, cot_eigvec = compute_pe_cot_eigenvector(new_pos, new_faces, k=self.lap_k)
        if not (torch.isfinite(cot_eigval).all() and torch.isfinite(cot_eigvec).all()):
            raise ValueError("non-finite cot Laplacian eigenpairs (likely degenerate mesh)")

        n_orig = len(pos)
        n_nodes = len(new_pos)
        n_edges = new_edge_index.shape[1] // 2
        print(f"[ok] {version}/{cid}: {n_orig}→{n_nodes} nodes, {n_edges} edges "
              f"(decimate_ratio={self.decimate_ratio})")

        node_type_dec = reg["node_type"][orig_idx_t]                        # [N]
        # global_geo_dist: original distance from focused region, normalized to [0, 1]
        gg = reg["geo_dist"][orig_idx_t].float()
        global_geo_dist = gg / gg.max().clamp_min(1e-8)
        # geo_dist: per-opening (inlet, outlet_0, outlet_1) min-distance, each in [0, 1],
        # absent openings padded with 1.0 (semantically "infinitely far / not present")
        opening_dists = per_opening_geo_dist(new_pos, new_edge_index, node_type_dec.numpy())
        N = len(new_pos)
        geo_dist = torch.stack([
            torch.from_numpy(opening_dists.get("inlet",    np.ones(N))).float(),
            torch.from_numpy(opening_dists.get("outlet_0", np.ones(N))).float(),
            torch.from_numpy(opening_dists.get("outlet_1", np.ones(N))).float(),
        ], dim=-1)                                                          # [N, 3]

        static = {
            "pos":             torch.from_numpy(new_pos).float(),       # [N, 3]
            "edge_index":      torch.from_numpy(new_edge_index),        # [2, E]
            "faces":           torch.from_numpy(new_faces),             # [F, 3]
            "node_type":       node_type_dec,                           # [N]
            "flow_split":      reg["flow_split"],                       # [2]
            "aneurysm_type":   reg["aneurysm_type"],                    # scalar
            "global_geo_dist": global_geo_dist,                         # [N]   in [0, 1]
            "geo_dist":        geo_dist,                                # [N, 3] each col in [0, 1]
            "normals":         torch.from_numpy(normals[orig_idx]).float(),  # [N, 3]
            "cot_eigval":      cot_eigval,                              # [k]
            "cot_eigvec":      cot_eigvec,                              # [N, k]
        }
        torch.save(static, self._static_path(version, cid))

        wall_data = torch.load(os.path.join(case_dir, "wall_data.pt"), weights_only=False)
        wss_frame = {"aneug_flow": self._wss_frame_aneug_flow,
                     "angioflow": self._wss_frame_angioflow}[version]
        for t in range(self.n_frames):
            wss = wss_frame(wall_data, t)[orig_idx_t]  # [N_dec, 3]
            torch.save(wss, self._wss_path(version, cid, t))

    @staticmethod
    def _wss_frame_aneug_flow(wall_data, t):
        """AneuG-Flow wall_data.pt: flat dict, [T, N, 1] keys, same node order as registration pos."""
        cols = [wall_data[key][t, :, 0].numpy() for key in PHYSICS_KEYS]
        return torch.from_numpy(np.stack(cols, axis=1)).float()

    @staticmethod
    def _wss_frame_angioflow(wall_data, t):
        """AngioFlow wall_data.pt: nested under "wall" zone, same node order as registration pos."""
        cols = [wall_data["wall"][key][t, :, 0].numpy() for key in PHYSICS_KEYS]
        return torch.from_numpy(np.stack(cols, axis=1)).float()

    @lru_cache(maxsize=2048)
    def _load_static(self, version, cid):
        s = torch.load(self._static_path(version, cid), weights_only=False)
        # Precomputed statics (whding123/AngioFlow-PyG) carry 64 cot modes; keep only lap_k,
        # here and not in a transform: this cache holds every static per DataLoader
        # worker, and a transform would only run after it (~9 MB/case at 64 modes).
        # clone() so the full 64-column tensor is freed rather than kept behind a view.
        k = s["cot_eigvec"].shape[1]
        if k < self.lap_k:
            raise ValueError(f"{version}/{cid}: static has {k} cot modes < lap_k={self.lap_k}")
        if k > self.lap_k:
            s["cot_eigvec"] = s["cot_eigvec"][:, :self.lap_k].clone()
            s["cot_eigval"] = s["cot_eigval"][:self.lap_k].clone()
        return s

    def len(self):
        return len(self._index)

    def get(self, idx):
        version, cid, t = self._index[idx]
        s = self._load_static(version, cid)
        wss = torch.load(self._wss_path(version, cid, t), weights_only=False)
        return Data(
            pos=s["pos"],               # [N, 3]
            edge_index=s["edge_index"], # [2, E]
            y=wss,                      # [N, 3]
            normals=s["normals"],       # [N, 3]
            node_type=s["node_type"],   # [N]
            geo_dist=s["geo_dist"],     # [N, 3] per-opening (inlet, outlet_0, outlet_1)
            global_geo_dist=s["global_geo_dist"],  # [N] in [0, 1]
            cot_eigvec=s["cot_eigvec"], # [N, k]
            cot_eigval=s["cot_eigval"], # [k]
            flow_split=s["flow_split"], # [2]
            aneurysm_type=s["aneurysm_type"],  # scalar
            t=torch.tensor([t]),
            version=version,            # str — batched as list by PyG
            case_id=cid,                # str — batched as list by PyG
        )


def detect_polluted_cases(save_dir, threshold=3.5, render=False):
    """Flag cases whose t=0 max |WSS| is an outlier (modified-z via median/MAD).

    Loads only `wss_*_0.pt` (one tensor per case), computes max ||WSS||, and
    reports any case with |z| > threshold (3.5 ≈ Iglewicz–Hoaglin convention).
    Useful for catching simulation blow-ups or unit-mismatched cases.
    """
    import numpy as np
    cases = []
    for path in sorted(glob.glob(os.path.join(save_dir, "*", "wss_*_0.pt"))):
        stem = os.path.basename(path)[len("wss_"):-len("_0.pt")]  # e.g. angioflow_C0001_cut2
        v, cid = split_tag_cid(stem)
        if v is None:
            continue
        wss = torch.load(path, weights_only=False)
        cases.append((v, cid, wss.norm(dim=-1).max().item()))

    if not cases:
        print("[detect] no wss_*_0.pt found in", save_dir)
        return []

    mags = np.array([m for _, _, m in cases])
    median = np.median(mags)
    mad = np.median(np.abs(mags - median)) + 1e-8
    z = 0.6745 * (mags - median) / mad

    flagged = sorted(
        [(v, cid, mag, float(zi)) for (v, cid, mag), zi in zip(cases, z) if abs(zi) > threshold],
        key=lambda x: -abs(x[3]),
    )
    print(f"[detect] {len(cases)} cases, median max|WSS|={median:.3f}, MAD={mad:.3f}")
    for v, cid, mag, zi in flagged:
        print(f"  [outlier] {v}/{cid}: max|WSS|={mag:.3f}, z={zi:+.2f}")
        if render:
            from angioflow.visualize.data_sanity import wss_sanity_video
            wss_sanity_video(save_dir, v, cid, n_frames=80)
    print(f"[detect] {len(flagged)}/{len(cases)} flagged (|z|>{threshold})")

    import json
    out_path = os.path.join(save_dir, "anomalies.json")
    with open(out_path, "w") as f:
        json.dump([{"version": v, "cid": cid, "max_wss": mag, "z": zi}
                   for v, cid, mag, zi in flagged], f, indent=2)
    print(f"[detect] wrote {out_path}")
    return flagged


def load_excluded_keys(save_dir):
    """Merge anomalies.json (WSS outliers) and cfd_anomalies.json (non-converged CFD)
    from save_dir and return a set of (version, cid) keys for exclude_keys=."""
    import json
    keys = set()
    for fname in ("anomalies.json", "cfd_anomalies.json"):
        path = os.path.join(save_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            items = json.load(f)
        keys.update((normalize_tag(d["version"]), d["cid"]) for d in items)
    return keys


if __name__ == "__main__":
    save_dir = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_pyg_stable"
    ds = AngioFlowWallDataset(
        case_root_aneug_flow="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv1/transient_data",
        case_root_angioflow="/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged",
        lap_k=8,
        save_dir=save_dir,
        n_frames=80,
        decimate_method='fps',
        decimate_ratio=1.0,
        per_version_dirs=True,
    )

    from collections import defaultdict
    per_version = defaultdict(list)
    for v, cid in ds.case_keys:
        per_version[v].append(cid)
    for v, cids in per_version.items():
        print(f"[dataset] {v}: {len(cids)} cases — {cids[:5]}{'...' if len(cids) > 5 else ''}")

    detect_polluted_cases(save_dir, threshold=5, render=True)

    from angioflow.data.pollution_detect import save_cfd_anomalies
    save_cfd_anomalies(save_dir)





"""
conda activate new
python -m angioflow.data.datasets

"""