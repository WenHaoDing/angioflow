# /// script
# dependencies = ["huggingface_hub", "numpy", "scipy", "torch", "matplotlib"]
# ///
"""Register CFD wall shapes for the AneuGv2 dataset, reading/writing whding123/AneuGv2
on the Hub directly -- meant to run as a Hugging Face Job (hf jobs uv run), not as a
module in this repo: it's a single self-contained file with no local `angioflow`
import, since a Job runs in a fresh environment on HF's own infrastructure where this
repo doesn't exist. The dependency header above is what `uv` reads to install exactly
these packages there -- no requirements.txt, no pre-baked image.

hf jobs uv run --flavor cpu-upgrade --timeout 6h --secrets HF_TOKEN \
    angioflow/huggingface/register_cfd_shapes_aneugv2.py -- --execute --yes



No pytorch3d: unlike register_cfd_shapes_angioflow.py (whose focused-region step
needs ICP to align a separately-aligned PLY against the CFD wall, hence pytorch3d's
Meshes + angioflow.data.align.compute_geo_distance), this dataset's geometry/<case>/
dome.obj is ALREADY co-registered with cfd_data/<case>/wall_data.pt's CFD frame -- the
only difference is units (dome.obj/mesh.obj/geomagic_processed.obj are in mm, wall_data
is in m). Verified empirically on 3 real cases: after a plain *0.001 scale (no rotation,
no translation, no ICP), the nearest-vertex distance from every wall node actually inside
the dome to dome.obj's surface is a few microns to a few tens of microns -- a rotation/
translation misalignment of any real size would show up as a nonzero floor across ALL
of those points, and there isn't one. So the whole focused-region + geo_dist step is
just a KD-tree threshold + Dijkstra, no pytorch3d, no open3d, no ICP.

Registration file format (same fields as AneuG-Flow/AngioFlow's wall_mesh_registration.pt):
    "pos":           [N, 3]   float32  wall node positions (metres)
    "edge_index":    [2, E]   int64    undirected graph edges
    "faces":         [F, 3]   int64    triangle connectivity
    "node_type":     [N]      int64    0=wall, 1=focused (dome), 2=inlet, 3=outlet_0, 4=outlet_1
    "flow_split":    [2]      float32  outlet flow split ratios (zeros if unavailable)
    "aneurysm_type": scalar   int64    from geometry/<case>/meta.json directly (every
                                       AneuGv2 case has it -- no multi-source search
                                       like AngioFlow's load_aneurysm_type needed)
    "geo_dist":      [N]      float32  geodesic distance to the dome (focused) region

Written back to the Hub, batched BATCH_SIZE cases per commit (not one per case, not one
commit for everything -- see upload_aneugv2_geometry.py's docstring for why: this exact
rate-limit incident already happened once this session). Everything lives under
cfd_data/<case>/ -- wall_mesh_registration.pt belongs next to the wall_data.pt/
blood_data.pt it was computed from, not in geometry/ (that holds the pre-mesh inputs
this script reads, not its own output):
    cfd_data/<case>/wall_mesh_registration.pt
    cfd_data/_sanity/<case>/mesh.png, registration.png, geo_dist.png   (--no-sanity to skip)
    cfd_data/_sanity/<case>/wss.gif, vel.gif   (every case; --gif_frames caps frames/gif,
                                               default 10 of 80 -- NOT case count: every
                                               case still downloads blood_data.pt, 3+GB)

Every sanity file is ALSO copied to LOCAL_SANITY_ROOT/<case>/ (same filenames) --
standardized, shared with register_cfd_shapes_aneug_flow.py/register_cfd_shapes_angioflow.py's own sanity
output (angioflow.data.registration_sanity.save_registration_sanity produces the same
mesh.png/registration.png/geo_dist.png; this script can't import that module -- see
"No pytorch3d" above -- so it keeps its own copy of the same rendering, inlined). A
real cloud Job run has no local disk for that copy to land on, so it silently no-ops
there -- the Hub upload above is what a real job run actually produces.

For local testing (plain `python`, not `hf jobs uv run`) this needs the same packages
on PYTHONPATH -- `pip install huggingface_hub numpy scipy torch matplotlib` -- since the
uv dependency header is only read by `uv`/the Job runner, not by python directly:

    python register_cfd_shapes_aneugv2.py --cases cohort_650_0000_t0   # one case, local
    python register_cfd_shapes_aneugv2.py --dry_run                    # plan the full run
    python register_cfd_shapes_aneugv2.py --execute --yes              # process + upload all
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

REPO_ID = "whding123/AneuGv2"
CFD_PREFIX = "cfd_data"
GEO_PREFIX = "geometry"
# Duplicated from angioflow.data.registration_sanity.SANITY_ROOT, not imported: this
# script runs as a self-contained Hugging Face Job (see module docstring) with no
# local `angioflow` import. Only reachable when run locally for testing -- a real
# cloud Job container has no /media/yaplab2 mount, so the copy below silently no-ops
# there and the Hub upload (below) remains the only sanity output for a real run.
LOCAL_SANITY_ROOT = "/media/yaplab2/wd8tb/wenhao/datasets/angioflow_ghd_graphs/_sanity"
MM_TO_M = 1e-3
MATCH_TOL = 1e-6          # inlet/outlet zone -> wall node matching (metres)
BATCH_SIZE = 20
MAX_RETRIES = 5
RETRY_DELAY = 10


# ---------------------------------------------------------------------------
# Plain-numpy geometry helpers (no pytorch3d/open3d/trimesh)
# ---------------------------------------------------------------------------

def face_to_edge_index(faces):
    """Copied from angioflow.data.mesh (not imported -- see module docstring)."""
    k = faces.shape[1]
    edges = np.concatenate([faces[:, [i, (i + 1) % k]] for i in range(k)], axis=0)
    edges.sort(axis=1)
    n = int(faces.max()) + 1
    codes = np.unique(edges[:, 0].astype(np.int64) * n + edges[:, 1].astype(np.int64))
    return np.stack([codes // n, codes % n], axis=0).astype(np.int64)


def load_obj(path):
    """Minimal Wavefront .obj reader: vertices + triangle faces only, nothing fancy --
    dome.obj has no normals/UVs/materials worth parsing."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def largest_connected_focused(focused_mask, faces):
    """Keep only the largest connected component of focused nodes; drop islands.
    Copied from register_cfd_shapes_aneug_flow.py rather than imported -- that
    module's own top-level import of angioflow.data.align pulls in pytorch3d."""
    focused_ids = set(np.where(focused_mask)[0])
    adj = defaultdict(set)
    for tri in faces:
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            if a in focused_ids and b in focused_ids:
                adj[a].add(b); adj[b].add(a)
    visited, components = set(), []
    for start in focused_ids:
        if start in visited:
            continue
        comp, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n); comp.add(n)
            stack.extend(adj[n] - visited)
        components.append(comp)
    largest = max(components, key=len) if components else set()
    mask = np.zeros(len(focused_mask), dtype=bool)
    mask[list(largest)] = True
    return mask


def match_zone_to_wall(wall_pts, zone_pts, tol=MATCH_TOL):
    """Wall node indices nearest to each zone node (inlet/outlet caps only coincide
    with the wall mesh along their rim; most zone nodes go unmatched, by design)."""
    dists, idx = cKDTree(wall_pts).query(zone_pts, k=1)
    return idx[dists < tol]


def build_node_type(n_pts, focused_mask, inlet_idx, outlet0_idx, outlet1_idx):
    labels = torch.zeros(n_pts, dtype=torch.long)
    labels[focused_mask] = 1
    labels[torch.from_numpy(inlet_idx)] = 2
    labels[torch.from_numpy(outlet0_idx)] = 3
    labels[torch.from_numpy(outlet1_idx)] = 4
    return labels


def _max_edge_len(pts, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    return float(np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1).max())


def compute_geo_distance(coord_cfd, faces_cfd, dome_verts, dome_faces):
    """Graph-geodesic distance [N] from each CFD wall node to the dome (focused) region.
    Same algorithm as angioflow.data.align.compute_geo_distance, just taking plain
    (verts, faces) arrays instead of a pytorch3d Meshes object -- dome_verts/dome_faces
    are already in the CFD frame (see module docstring), nothing to reconstruct."""
    tol = max(_max_edge_len(coord_cfd, faces_cfd), _max_edge_len(dome_verts, dome_faces))
    d_to_dome, _ = cKDTree(dome_verts).query(coord_cfd, k=1)
    seed = np.where(d_to_dome < tol)[0]
    assert seed.size > 0, "no CFD nodes overlap dome.obj -- check units/alignment"

    N = coord_cfd.shape[0]
    e = np.concatenate([faces_cfd[:, [0, 1]], faces_cfd[:, [1, 2]], faces_cfd[:, [2, 0]]], axis=0)
    w = np.linalg.norm(coord_cfd[e[:, 0]] - coord_cfd[e[:, 1]], axis=1)
    rows = np.concatenate([e[:, 0], e[:, 1], np.full(seed.size, N), seed])
    cols = np.concatenate([e[:, 1], e[:, 0], seed, np.full(seed.size, N)])
    data = np.concatenate([w, w, np.zeros(seed.size), np.zeros(seed.size)])
    adj = csr_matrix((data, (rows, cols)), shape=(N + 1, N + 1))
    geo_dist = dijkstra(adj, indices=N)[:N]
    return torch.from_numpy(geo_dist).float()


# ---------------------------------------------------------------------------
# Per-case processing
# ---------------------------------------------------------------------------

def process_case(case_dir):
    """case_dir must already have wall_data.pt (from cfd_data/) and dome.obj + meta.json
    (from geometry/) downloaded into it. Returns the registration dict, or (None, reason)
    on a skip."""
    wall_data = torch.load(os.path.join(case_dir, "wall_data.pt"), weights_only=False)
    if "faces" not in wall_data["wall"]:
        return None, "wall_data.pt has no mesh connectivity"

    pts = wall_data["wall"]["pos"].numpy().astype(np.float64)
    faces = wall_data["wall"]["faces"].numpy().astype(np.int64)
    edge_index = face_to_edge_index(faces)

    empty_idx = np.zeros(0, dtype=np.int64)
    inlet_idx = match_zone_to_wall(pts, wall_data["inlet"]["pos"].numpy())
    outlet0_idx = (match_zone_to_wall(pts, wall_data["outlet_0"]["pos"].numpy())
                  if "outlet_0" in wall_data else empty_idx)
    outlet1_idx = (match_zone_to_wall(pts, wall_data["outlet_1"]["pos"].numpy())
                  if "outlet_1" in wall_data else empty_idx)

    dome_verts, dome_faces = load_obj(os.path.join(case_dir, "dome.obj"))
    dome_verts = dome_verts * MM_TO_M

    thresh = _max_edge_len(dome_verts, dome_faces) * 2
    dists, _ = cKDTree(dome_verts).query(pts, k=1)
    focused_mask = largest_connected_focused(dists < thresh, faces)

    geo_dist = compute_geo_distance(pts, faces, dome_verts, dome_faces)
    node_type = build_node_type(len(pts), focused_mask, inlet_idx, outlet0_idx, outlet1_idx)

    with open(os.path.join(case_dir, "meta.json")) as f:
        aneurysm_type = int(json.load(f)["aneurysm_type"])

    flowsplit_path = os.path.join(case_dir, "flowsplit_ratio.txt")
    flow_split = (torch.from_numpy(np.loadtxt(flowsplit_path).reshape(-1).astype(np.float32))
                 if os.path.exists(flowsplit_path) else torch.zeros(2))

    reg = {
        "pos": torch.from_numpy(pts).float(),
        "edge_index": torch.from_numpy(edge_index),
        "faces": torch.from_numpy(faces),
        "node_type": node_type,
        "flow_split": flow_split,
        "aneurysm_type": torch.tensor(aneurysm_type, dtype=torch.long),
        "geo_dist": geo_dist,
    }
    return reg, None


def _trisurf_fig(pos, faces, scalars=None, cmap="viridis", clim=None, title=None, highlight=None):
    """One static mesh panel. matplotlib only (see save_static_sanity docstring).

    highlight: optional [(mask, color, label), ...] -- scattered ON TOP of the mesh in
    a bold marker, for regions too few nodes (inlet/outlet rims: tens of nodes out of
    tens of thousands) to read from face colour alone."""
    import matplotlib.pyplot as plt
    lo, hi = pos.min(0), pos.max(0)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    if scalars is None:
        ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=faces,
                        color="lightgrey", linewidth=0.1, edgecolor="dimgrey", antialiased=False)
    else:
        tri = ax.plot_trisurf(pos[:, 0], pos[:, 1], pos[:, 2], triangles=faces,
                              cmap=cmap, linewidth=0, antialiased=False)
        tri.set_array(scalars[faces].mean(axis=1))   # per-face colour = mean of its 3 verts
        if clim is not None:
            tri.set_clim(*clim)
        fig.colorbar(tri, ax=ax, shrink=0.5)
    if highlight:
        for mask, color, label in highlight:
            if mask.any():
                ax.scatter(pos[mask, 0], pos[mask, 1], pos[mask, 2], color=color, s=50,
                          edgecolors="black", linewidths=0.6, label=f"{label} ({int(mask.sum())})",
                          depthshade=False)
        ax.legend(loc="upper left", fontsize=8, markerscale=1.2)
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo).clip(min=1e-9))
    ax.set_axis_off()
    return fig


def save_static_sanity(case, reg, out_dir):
    """mesh.png (plain geometry) + registration.png (node_type) + geo_dist.png --
    matplotlib trisurf, not pyvista/VTK: a Job's container shouldn't need a real or
    virtual GL context just to save QA PNGs. Cheap: reuses fields process_case already
    has in memory, downloads nothing extra."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos, faces = reg["pos"].numpy(), reg["faces"].numpy()
    fig = _trisurf_fig(pos, faces, title=f"{case}: wall mesh")
    fig.savefig(os.path.join(out_dir, "mesh.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    nt = reg["node_type"].numpy()
    highlight = [(nt == 2, "red", "inlet"), (nt == 3, "blue", "outlet_0"), (nt == 4, "lime", "outlet_1")]
    fig = _trisurf_fig(pos, faces, nt.astype(float), "tab10", (0, 4),
                       f"{case}: node_type (0=wall 1=focused 2=inlet 3=out0 4=out1)",
                       highlight=highlight)
    fig.savefig(os.path.join(out_dir, "registration.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

    fig = _trisurf_fig(pos, faces, reg["geo_dist"].numpy(), "viridis", None, f"{case}: geo_dist (m)")
    fig.savefig(os.path.join(out_dir, "geo_dist.png"), dpi=110, bbox_inches="tight"); plt.close(fig)


def _save_gif(frames_rgba, out_path, fps):
    from PIL import Image
    images = [Image.fromarray(f) for f in frames_rgba]
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   duration=1000 // fps, loop=0)


def save_wss_gif(case, case_dir, reg, out_path, fps=5, max_frames=10):
    """Animated WSS magnitude straight from wall_data.pt's raw per-frame fields --
    independent of (stronger check than) registration.png/geo_dist.png, which show
    quantities *derived* from this script's own logic and can look fine even if the
    raw CFD fields were parsed wrong. Reuses one Poly3DCollection across frames (only
    the per-face colour array changes) instead of rebuilding the mesh each frame --
    matplotlib has no GPU path here, so redrawing 30-60k faces is already slow; capped
    to the first max_frames (of 80) rather than every frame, every case, for that reason."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    wall = torch.load(os.path.join(case_dir, "wall_data.pt"), weights_only=False)["wall"]
    wss_mag = torch.sqrt(wall["x_wall_shear"][:, :, 0] ** 2 + wall["y_wall_shear"][:, :, 0] ** 2
                         + wall["z_wall_shear"][:, :, 0] ** 2).numpy()
    T = min(max_frames, wss_mag.shape[0])
    clim = (0.0, float(np.percentile(wss_mag, 99)))

    pos, faces = reg["pos"].numpy(), reg["faces"].numpy()
    lo, hi = pos.min(0), pos.max(0)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    coll = Poly3DCollection(pos[faces], cmap="jet")
    coll.set_clim(*clim)
    ax.add_collection3d(coll)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo).clip(min=1e-9))
    ax.set_axis_off()
    ax.set_title(f"{case}: |WSS| (Pa)", fontsize=10)
    fig.colorbar(plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(*clim)), ax=ax, shrink=0.5)
    fig.canvas.draw()

    frames = []
    for t in range(T):
        coll.set_array(wss_mag[t][faces].mean(axis=1))
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    _save_gif(frames, out_path, fps)


def save_vel_gif(case, blood_path, out_path, fps=5, max_points=20000, max_frames=10):
    """Animated velocity magnitude point cloud from blood_data.pt -- runs for every
    case (blood_data.pt, 3+ GB/case, is downloaded regardless now), subsampled to
    max_points (a bare scatter of the full ~400k-point cloud, redrawn in software
    rendering, is punishingly slow) and capped to the first max_frames (of 80)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blood = torch.load(blood_path, weights_only=False)
    pos = blood["pos"].numpy()
    if len(pos) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pos), max_points, replace=False)
    else:
        idx = slice(None)
    pos = pos[idx]
    vel_mag = torch.sqrt(blood["x_velocity"][:, :, 0] ** 2 + blood["y_velocity"][:, :, 0] ** 2
                         + blood["z_velocity"][:, :, 0] ** 2).numpy()[:, idx]
    T = min(max_frames, vel_mag.shape[0])
    clim = (0.0, float(np.percentile(vel_mag, 99)))

    lo, hi = pos.min(0), pos.max(0)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=vel_mag[0], cmap="jet",
                    vmin=clim[0], vmax=clim[1], s=2, alpha=0.3, linewidths=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect((hi - lo).clip(min=1e-9))
    ax.set_axis_off()
    ax.set_title(f"{case}: |velocity| (m/s), {len(pos)}-pt sample", fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.5)
    fig.canvas.draw()

    frames = []
    for t in range(T):
        sc.set_array(vel_mag[t])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    _save_gif(frames, out_path, fps)


# ---------------------------------------------------------------------------
# Hub driver
# ---------------------------------------------------------------------------

def discover_cases(api, repo_id, redo=False):
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    cfd_cases = {f.split("/", 2)[1] for f in files if f.startswith(f"{CFD_PREFIX}/")}
    geo_cases = {f.split("/", 2)[1] for f in files if f.startswith(f"{GEO_PREFIX}/")}
    cases = cfd_cases & geo_cases
    if not redo:
        done = {f.split("/", 2)[1] for f in files
                if f.startswith(f"{CFD_PREFIX}/") and f.endswith("/wall_mesh_registration.pt")}
        cases -= done
    return sorted(cases)


def download_case(api, repo_id, case, scratch):
    """blood_data.pt is always fetched now: every case gets a vel.gif, not just a
    sample (3+ GB/case download -- see module docstring for the bandwidth tradeoff).

    Every aneurysm_type is processed: process_case's focused region comes from the
    case's own dome.obj, nothing type-specific. meta.json is only fetched because
    process_case stores its aneurysm_type in the registration.
    Returns (case_dir, None) on success, (None, reason) to skip -- same convention
    as process_case."""
    case_dir = os.path.join(scratch, case)
    os.makedirs(case_dir, exist_ok=True)

    hf_hub_download(repo_id=repo_id, repo_type="dataset", token=api.token,
                    filename=f"{GEO_PREFIX}/{case}/meta.json", local_dir=scratch)
    shutil.move(os.path.join(scratch, GEO_PREFIX, case, "meta.json"), os.path.join(case_dir, "meta.json"))

    for fname in ("wall_data.pt", "blood_data.pt"):
        hf_hub_download(repo_id=repo_id, repo_type="dataset", token=api.token,
                        filename=f"{CFD_PREFIX}/{case}/{fname}", local_dir=scratch)
        shutil.move(os.path.join(scratch, CFD_PREFIX, case, fname), os.path.join(case_dir, fname))
    for fname in ("dome.obj", "flowsplit_ratio.txt"):
        try:
            hf_hub_download(repo_id=repo_id, repo_type="dataset", token=api.token,
                            filename=f"{GEO_PREFIX}/{case}/{fname}", local_dir=scratch)
            shutil.move(os.path.join(scratch, GEO_PREFIX, case, fname), os.path.join(case_dir, fname))
        except EntryNotFoundError:
            if fname != "flowsplit_ratio.txt":
                raise
            # process_case already tolerates this missing (falls back to zeros(2)) --
            # the download-side guard just needs to not crash the case over it
    return case_dir, None
    return case_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo_id", default=REPO_ID)
    p.add_argument("--cases", nargs="*", default=None, help="explicit case list; default: all overlapping cases")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--sanity", action=argparse.BooleanOptionalAction, default=True,
                   help="save + upload mesh/registration/geo_dist PNGs + wss.gif/vel.gif per "
                        "case (default on). Every case downloads blood_data.pt (3+GB) for "
                        "vel.gif now -- see module docstring for the bandwidth tradeoff.")
    p.add_argument("--gif_frames", type=int, default=10,
                   help="frames per gif (of 80 available) -- matplotlib has no GPU path, so "
                        "rendering all 80 for every case would be very slow")
    p.add_argument("--redo", action="store_true",
                   help="reprocess cases that already have cfd_data/<case>/wall_mesh_registration.pt "
                        "(default: skip them -- a rerun only picks up new/unregistered cases)")
    p.add_argument("--dry_run", action="store_true", help="list cases and stop")
    p.add_argument("--execute", action="store_true", help="actually process + upload (needs --yes too)")
    p.add_argument("--yes", action="store_true", help="confirm --execute; without it, --execute still only prints")
    return p.parse_args()


def main():
    args = parse_args()
    api = HfApi(token=args.token)
    cases = args.cases or discover_cases(api, args.repo_id, redo=args.redo)
    print(f"[plan] {len(cases)} case(s) with both {CFD_PREFIX}/ and {GEO_PREFIX}/ data")
    if args.dry_run or not args.execute:
        print("cases (first 20):", cases[:20])
        print("\n[dry run only] pass --execute --yes to process + upload.")
        return
    if not args.yes:
        print("\n[dry run only] --execute given without --yes -- not touching anything.")
        return

    batches = [cases[i:i + args.batch_size] for i in range(0, len(cases), args.batch_size)]
    print(f"[run] {len(cases)} case(s) in {len(batches)} batch(es) of up to {args.batch_size}")

    for bi, batch in enumerate(batches, 1):
        ops = []
        with tempfile.TemporaryDirectory() as scratch:
            for case in batch:
                t0 = time.time()
                try:
                    case_dir, dl_reason = download_case(api, args.repo_id, case, scratch)
                    if case_dir is None:
                        print(f"[skip] {case}: {dl_reason}")
                        continue
                    reg, reason = process_case(case_dir)
                    if reg is None:
                        print(f"[skip] {case}: {reason}")
                        continue
                    out_path = os.path.join(case_dir, "wall_mesh_registration.pt")
                    torch.save(reg, out_path)
                    ops.append(CommitOperationAdd(
                        path_in_repo=f"{CFD_PREFIX}/{case}/wall_mesh_registration.pt",
                        path_or_fileobj=out_path))
                    if args.sanity:
                        sanity_dir = os.path.join(case_dir, "_sanity")
                        os.makedirs(sanity_dir, exist_ok=True)
                        try:
                            save_static_sanity(case, reg, sanity_dir)
                            save_wss_gif(case, case_dir, reg, os.path.join(sanity_dir, "wss.gif"),
                                        max_frames=args.gif_frames)
                            save_vel_gif(case, os.path.join(case_dir, "blood_data.pt"),
                                        os.path.join(sanity_dir, "vel.gif"), max_frames=args.gif_frames)
                            for fname in os.listdir(sanity_dir):
                                ops.append(CommitOperationAdd(
                                    path_in_repo=f"{CFD_PREFIX}/_sanity/{case}/{fname}",
                                    path_or_fileobj=os.path.join(sanity_dir, fname)))
                            try:
                                local_case_dir = os.path.join(LOCAL_SANITY_ROOT, case)
                                os.makedirs(local_case_dir, exist_ok=True)
                                for fname in os.listdir(sanity_dir):
                                    shutil.copy2(os.path.join(sanity_dir, fname),
                                                os.path.join(local_case_dir, fname))
                            except OSError:
                                pass   # no local disk access -- running in a real cloud Job
                        except Exception as exc:
                            # a render failure must not lose an otherwise-good registration
                            print(f"[sanity fail] {case}: {exc}")
                    print(f"[ok] {case} ({time.time() - t0:.1f}s)")
                except Exception as e:
                    print(f"[fail] {case}: {e}")
                finally:
                    # CPU Upgrade (32GB RAM, 50GB ephemeral storage) can't hold a whole
                    # batch's downloaded inputs at once -- blood_data.pt alone is 3+GB/case.
                    # Drop this case's bulky DOWNLOADED inputs now; the small OUTPUTS
                    # (wall_mesh_registration.pt, sanity pngs/gifs) stay until the batch
                    # commits below, since those are only a few MB total even x20 cases.
                    for fname in ("wall_data.pt", "blood_data.pt"):
                        fpath = os.path.join(scratch, case, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)

            if not ops:
                continue
            delay = RETRY_DELAY
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=ops,
                                      commit_message=f"Add wall_mesh_registration.pt for "
                                                     f"{len(ops)} case(s) (batch {bi}/{len(batches)})")
                    print(f"[{bi}/{len(batches)}] committed {len(ops)} case(s)")
                    break
                except Exception as e:
                    rate_limited = "429" in str(e) or "rate limit" in str(e).lower()
                    wait = max(delay, 60) if rate_limited else delay
                    print(f"[{bi}/{len(batches)}] attempt {attempt}/{MAX_RETRIES} failed: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(wait); delay *= 2
                    else:
                        raise
    print("\n[done]")


if __name__ == "__main__":
    main()
