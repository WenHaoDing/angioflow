"""Dataset-aware GHD <-> CFD registration for AngioFlow (AneuX + ImperialNHS) data.

All coordinates are stored in METERS, matching the original (synthetic-data)
new_version/loaders.py pipeline's convention (its `scale` includes a 0.001
mm->m factor) -- landmarks.npz/metrics.json/ghd_coefficients.npz are natively
millimeter-scale, so this module converts to meters right before storage.

case_registry.xlsx (angioflow/data/build_case_registry.py) is the single source
of truth for each case's dataset source and aneurysm_type -- no per-case file
sniffing. Two dataset sources are handled:

- "ImperialNHS": ghd_coefficients.npz / landmarks.npz / metrics.json are
  colocated with wall_data.pt in the case_dir itself (e.g. an *_extracted
  folder produced by export_batch_pt.py), alongside geomagic_processed.obj.
- "AneuX": those GHD fitting files live under a separate ghd_root tree that
  keeps the ORIGINAL case names (C0002_cut2, SNF00000036_02_2_cut2, ...) and
  must not be renamed. We index it with the same simplify_name() the geometry
  mirror uses (see angioflow/data/rename_geometry_folder.py), which maps those
  onto the registry's short case ids collision-free.

Both sources reconstruct through the type-aware MultiCanonicalGHDReconstruct
(bifurcated vs sidewall canonical, picked by aneurysm_type). Two frames of that
reconstruction are used for two different purposes, mirroring the original
new_version/loaders.py:record_mesh_upsampling + register_transient_raw_data_to_image_4k
split:

- Canonical frame (mesh_data['subdivided_ghd_Meshes'] etc.): canonical + GHD
  offset, scaled isotropically by this case's true physical size (s * s_can)
  but with NO rotation/translation applied. Every case shares the same
  position/orientation here, so shapes are directly comparable/overlaid across
  cases -- this is what a shared-topology GNN trains on, and what the old
  pipeline stored (scale-only, canonical-space).
- World frame (used only transiently inside reconstruct_case_meshes /
  register_wall_physics): the full affine (R, s, t_vec, s_can, R_frame,
  neck_pt) alignment into the real CFD frame. This is needed ONLY as the
  knn_interpolate target so each vertex gets the physically-correct CFD wall
  shear stress -- no mesh-to-mesh ICP needed for that step. NOTHING is stored
  in this frame: the sampled WSS vectors are rotated straight back to canonical
  (R_equivalent) so that every channel of the registered tensor -- coordinates,
  normals and WSS -- lives in one frame. Skipping that rotation leaves the WSS
  pointing in world directions next to canonical normals, which destroys the
  wss . n = 0 tangency the field must satisfy.

For ImperialNHS cases, an optional geomagic_processed.obj + CFD point-cloud ICP
refinement (get_cfd_to_world_transform) can be layered on top of the world-frame
step when `refine_imperialnhs_with_icp=True`, matching the old export_batch_pt.py
flow.
"""

import os
import json

import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.ops import SubdivideMeshes
from pytorch3d.structures import join_meshes_as_batch
from torch_geometric.nn.unpool import knn_interpolate

import pandas as pd

from angioflow.data.rename_geometry_folder import simplify_name

from .mesh_utils import clean_meshes, get_trimmed_indice, get_downsampling_edge_index
from .ghd_reconstruct import MultiCanonicalGHDReconstruct
from .alignment import ghd_world_alignment, _so3_exp_map_numpy, DEFAULT_CANONICAL_ROOT
from .cfd_alignment import get_cfd_to_world_transform

IMPERIALNHS_TYPE_ROOT = "/media/yaplab2/wd8tb/wenhao/angioflow/data/geometry/ImperialNHS"


# ---------------------------------------------------------------------------
# Dataset source detection and path resolution
# ---------------------------------------------------------------------------

DEFAULT_REGISTRY_PATH = "/media/yaplab2/wd8tb/wenhao/datasets/angioflowv2_merged/_meta/case_registry.xlsx"


def load_registry(registry_path=DEFAULT_REGISTRY_PATH):
    """case -> {"dataset", "aneurysm_type", "original_name"} from case_registry.xlsx.

    The registry is the single source of truth for both dataset source and
    aneurysm_type -- no per-case file sniffing (endpoints_manual.npy /
    landmarks.npz / geomagic_processed.obj) is done anymore. Rows with a blank
    aneurysm_type get None and are skipped by the assembler.
    """
    df = pd.read_excel(registry_path)
    registry = {}
    for row in df.itertuples(index=False):
        atype = getattr(row, "aneurysm_type", None)
        registry[str(row.case)] = {
            "dataset": str(row.dataset),
            "aneurysm_type": None if pd.isna(atype) else int(atype),
            "original_name": None if pd.isna(row.original_name) else str(row.original_name),
        }
    return registry


def build_ghd_dir_index(ghd_root, return_collisions=False):
    """simplified case name -> GHD fitting folder path.

    The GHD fitting folder keeps its ORIGINAL case names (C0002_cut2,
    SNF00000036_02_2_cut2, ...) and must not be renamed, so we index it by
    applying the same simplify_name() the geometry mirror uses.

    simplify_name() is NOT injective on every fitting root: e.g.
    ANSYS_UNIGE_30_612_cut1 and ANSYS_UNIGE_30_614_cut1 both map to "ANSYS30".
    Silently keeping one would attach a DIFFERENT patient's GHD fit to the case's
    CFD data -- observed as a ~20 mm mesh/CFD mismatch, i.e. physically wrong WSS
    with no obvious symptom in canonical space. Ambiguous names are therefore
    dropped from the index and reported instead of guessed.
    """
    candidates = {}
    for name in sorted(os.listdir(ghd_root)):
        path = os.path.join(ghd_root, name)
        if os.path.isdir(path):
            candidates.setdefault(simplify_name(name), []).append(path)

    index = {k: v[0] for k, v in candidates.items() if len(v) == 1}
    collisions = {k: v for k, v in candidates.items() if len(v) > 1}
    if collisions:
        print(f"[build_ghd_dir_index] {len(collisions)} ambiguous case name(s) excluded "
              f"(several fitting folders map to the same id): "
              + "; ".join(f"{k} <- {[os.path.basename(p) for p in v]}" for k, v in collisions.items()))
    return (index, collisions) if return_collisions else index


def resolve_case_paths(case, case_dir, dataset_source, ghd_index=None):
    """Returns (ghd_dir, geomagic_path); ghd_dir is None when no GHD fit exists."""
    if dataset_source == "AneuX":
        if ghd_index is None:
            raise ValueError("AneuX cases require a ghd_index (see build_ghd_dir_index)")
        return ghd_index.get(case), None

    # ImperialNHS: ghd files are colocated with wall_data.pt in case_dir itself.
    geomagic_path = os.path.join(case_dir, "geomagic_processed.obj")
    return case_dir, geomagic_path


# ---------------------------------------------------------------------------
# GHD reconstruction + affine alignment (type-aware)
# ---------------------------------------------------------------------------

_type_mask_cache = {}
_type_template_cache = {}


def get_type_mask(multi_recon, atype):
    """Boolean vertex mask (True = keep) removing the inlet/outlet cap vertices
    from the canonical mesh of this aneurysm type, via nearest-vertex
    correspondence against that type's mesh_trimmed.obj. Cached per type."""
    from AneuG.utils.utils import safe_load_mesh

    if atype not in _type_mask_cache:
        spec = multi_recon.specs[int(atype)]
        raw_canonical = safe_load_mesh(str(spec.mesh_path))
        raw_trimmed = safe_load_mesh(os.path.join(str(spec.root), "mesh_trimmed.obj"))
        _type_mask_cache[atype] = get_trimmed_indice(raw_trimmed, raw_canonical)
    return _type_mask_cache[atype]


def load_ghd_affine(ghd_dir):
    ghd_npz = np.load(os.path.join(ghd_dir, "ghd_coefficients.npz"))
    with open(os.path.join(ghd_dir, "metrics.json")) as f:
        s_can = float(json.load(f)["s_can"])
    landmarks = np.load(os.path.join(ghd_dir, "landmarks.npz"))
    return {
        "phi": torch.from_numpy(ghd_npz["phi"]).view(1, -1, 3).float(),
        "R": _so3_exp_map_numpy(ghd_npz["w_rot"]),
        "s": float(np.exp(float(ghd_npz["log_scale"]))),
        "t_vec": ghd_npz["t_vec"],
        "s_can": s_can,
        "R_frame": landmarks["R_frame"],
        "neck_pt": landmarks["neck_pt_world"],
    }


MM_TO_M = 1e-3


def reconstruct_case_meshes(multi_recon, ghd_dir, aneurysm_type, n_subdivide=1):
    """Reconstruct via the type-aware canonical and trim caps, then return both
    frames needed downstream, both in METERS:

    - cleaned_canonical / subdivided_canonical: canonical + GHD offset, scaled
      isotropically by this case's true size (shape_scale = s * s_can * 1e-3,
      the same 0.001 mm->m factor the old pipeline's "scale" used), NO
      rotation/translation. Shared position/orientation across every case of
      this type -- goes into mesh_data.
    - subdivided_world: the same subdivided mesh, fully aligned (R, s, t_vec,
      s_can, R_frame, neck_pt) into the CFD/world frame. Used only as the
      knn_interpolate target in register_wall_physics, never stored directly.
    """
    affine = load_ghd_affine(ghd_dir)
    mask = get_type_mask(multi_recon, aneurysm_type)
    phi = affine["phi"].to(multi_recon.device)

    ghd_Meshes = multi_recon.forward_as_meshes(phi, aneurysm_type=aneurysm_type, denormalize_shape=False)
    cleaned_Meshes = clean_meshes(ghd_Meshes, mask)
    subdivided_Meshes = cleaned_Meshes
    for _ in range(n_subdivide):
        subdivided_Meshes = SubdivideMeshes()(subdivided_Meshes)

    verts_world_mm = ghd_world_alignment(
        subdivided_Meshes.verts_packed().detach().cpu().numpy(), R=affine["R"], s=affine["s"], t_vec=affine["t_vec"],
        s_can=affine["s_can"], R_frame=affine["R_frame"], neck_pt=affine["neck_pt"], style="c2w",
    )
    subdivided_world = subdivided_Meshes.update_padded(
        (torch.from_numpy(verts_world_mm).float() * MM_TO_M).unsqueeze(0).to(subdivided_Meshes.device)
    )

    shape_scale = torch.tensor([[affine["s"] * affine["s_can"] * MM_TO_M]])  # [1, 1], matches old "shape_scale" convention
    cleaned_canonical = cleaned_Meshes.update_padded(cleaned_Meshes.verts_padded() * shape_scale.to(cleaned_Meshes.device))
    subdivided_canonical = subdivided_Meshes.update_padded(subdivided_Meshes.verts_padded() * shape_scale.to(subdivided_Meshes.device))

    return {
        "cleaned_canonical": cleaned_canonical,
        "subdivided_canonical": subdivided_canonical,
        "subdivided_world": subdivided_world,
        "shape_scale": shape_scale,
        "phi": affine["phi"],
        # world -> canonical rotation for VECTOR fields: c = w @ R_equivalent
        # (ghd_world_alignment's c2w is w = c @ R_equivalent.T * s + t, and
        # R_equivalent is orthogonal). Needed because CFD vectors are sampled in
        # the world frame but stored alongside canonical-frame geometry.
        "R_eq": affine["R_frame"].T @ affine["R"],
    }


def get_type_template_mesh(multi_recon, atype, n_subdivide=1):
    """Zero-deformation mesh for this type, used only to derive the (shared,
    per-case-independent) downsampling/pooling hierarchy from its topology."""
    if atype not in _type_template_cache:
        mask = get_type_mask(multi_recon, atype)
        recon = multi_recon.get(atype)
        zero_phi = torch.zeros(1, recon.GHD_eigvec.shape[1], 3, device=recon.canonical_Meshes.device)
        template = multi_recon.forward_as_meshes(zero_phi, aneurysm_type=atype, denormalize_shape=False)
        template = clean_meshes(template, mask)
        for _ in range(n_subdivide):
            template = SubdivideMeshes()(template)
        _type_template_cache[atype] = template
    return _type_template_cache[atype]


# ---------------------------------------------------------------------------
# CFD -> mesh physics registration
# ---------------------------------------------------------------------------

def _refine_cfd_with_icp(pos_cfd_m, wss, geomagic_path):
    """Optional extra rigid registration step for ImperialNHS: the raw CFD mesh
    frame can differ from the geomagic-processed segmentation frame that GHD was
    fitted against. Estimate that transform via ICP (geomagic_processed.obj is
    millimeter-scale) and apply it to the CFD points (coord) and WSS vectors
    (rotation only) before interpolation. Returns points back in meters."""
    import pyvista as pv

    world_mesh = pv.read(geomagic_path)
    coord_world_mm = torch.from_numpy(np.asarray(world_mesh.points)).float()
    R_cfd2world, t_cfd2world = get_cfd_to_world_transform(pos_cfd_m * 1000.0, coord_world_mm)
    pos_cfd_m = (pos_cfd_m * 1000.0 @ R_cfd2world.T + t_cfd2world) * MM_TO_M
    wss = wss @ R_cfd2world.T  # vectors: rotate only, no translation (unit-independent)
    return pos_cfd_m, wss


def alignment_error_mm(mesh_verts_world_m, pos_cfd_m):
    """Distance (mm) from each mesh vertex to its NEAREST CFD wall point.

    This is the quantity knn_interpolate(k=3) actually averages over, so it is
    the direct measure of whether the WSS transfer is physically meaningful. A
    correct registration sits well under the CFD mesh's own edge length; a
    mismatched GHD fit shows up here as millimetres of error while canonical-frame
    geometry still looks perfect.
    """
    from scipy.spatial import cKDTree

    def _np(x):
        return (x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)) * 1000.0

    mesh_mm, cfd_mm = _np(mesh_verts_world_m), _np(pos_cfd_m)
    nn, _ = cKDTree(cfd_mm).query(mesh_mm, k=1)
    return {"mean": float(nn.mean()), "p95": float(np.percentile(nn, 95)), "max": float(nn.max()),
            "centroid_offset": float(np.linalg.norm(mesh_mm.mean(0) - cfd_mm.mean(0)))}


def register_wall_physics(case_dir, mesh_verts_world_m, mesh_verts_canonical_m,
                          mesh_normals_canonical=None, raw_file="wall_data.pt",
                          insert_normals=True, icp_refine=None, device=None,
                          max_align_mean_mm=None, max_align_max_mm=None,
                          wss_rotation=None):
    """CFD wall_data.pt (AngioFlow schema: wall.pos [N,3] in meters, wall.{x,y,z}_wall_shear
    and wall.wall_shear as [T,N,1]) -> physics registered onto the aligned mesh's
    vertices via a single knn_interpolate call (geometry is static across time, so
    all T frames are packed as extra channels instead of repeating the KNN query).

    Two frames are involved and must not be mixed up:
      - mesh_verts_world_m is ONLY the knn_interpolate target, so each vertex picks
        up the physically correct CFD wall shear stress.
      - the stored x/y/z and normal channels are the CANONICAL-frame ones, because
        that is what the old pipeline stored and therefore what the model consumes
        as input features (verified: denormalizing the old v5 dataset's x/y/z and
        normals reproduces its canonical mesh coords/normals exactly). Storing
        world-frame coords here would feed the model patient-specific absolute
        positions it was never trained on.

    `wss_rotation` (= R_equivalent) rotates the interpolated WSS VECTORS out of
    the CFD world frame and into that same canonical frame. It is not optional in
    practice: WSS is a surface traction, so wss . n = 0, and leaving the vectors
    in the world frame while the normals are canonical breaks that relation --
    measured as mean |cos(wss, n)| = 0.50 (i.e. no better than random directions)
    versus 0.09 once rotated, against a 0.002 floor set by the raw CFD data. The
    scalar magnitude channel is frame-independent and is left alone.

    Returns (images, labels, align). `images` is None when the QC thresholds are
    set and the mesh does not actually sit on the CFD wall -- in that case the
    interpolated WSS would come from the wrong part of the vessel, so the case
    must be dropped rather than silently kept.
    """
    device = device or torch.device("cpu")
    wall = torch.load(os.path.join(case_dir, raw_file))["wall"]
    pos_cfd_m = wall["pos"].float()  # already meters, matching mesh_verts_world_m
    wss = torch.cat([wall["x_wall_shear"], wall["y_wall_shear"], wall["z_wall_shear"], wall["wall_shear"]], dim=-1).float()
    T, N_cfd, C = wss.shape

    if icp_refine is not None:
        pos_cfd_m, wss = icp_refine(pos_cfd_m, wss)

    align = alignment_error_mm(mesh_verts_world_m, pos_cfd_m)
    if ((max_align_mean_mm is not None and align["mean"] > max_align_mean_mm) or
            (max_align_max_mm is not None and align["max"] > max_align_max_mm)):
        return None, None, align

    x_all = wss.permute(1, 0, 2).reshape(N_cfd, T * C)
    pos_y = mesh_verts_world_m.to(device)
    images = knn_interpolate(x_all.to(device), pos_cfd_m.to(device), pos_y, k=3)
    images = images.view(-1, T, C).permute(1, 0, 2).contiguous()  # [T, N_mesh, C]

    if wss_rotation is not None:
        # channels are [wss_x, wss_y, wss_z, |wss|]: rotate the vector, leave the
        # magnitude. Applied after interpolation rather than before -- rotation is
        # linear and knn_interpolate is a weighted mean, so the result is identical
        # and this way it costs one small matmul instead of one per CFD point.
        R = torch.as_tensor(np.asarray(wss_rotation), dtype=images.dtype, device=images.device)
        images[..., :3] = images[..., :3] @ R

    T_out = images.shape[0]
    # stored geometry channels are CANONICAL frame (see docstring), not pos_y
    xyz = mesh_verts_canonical_m.to(images.device).unsqueeze(0).expand(T_out, -1, -1)
    images = torch.cat([xyz, images], dim=-1)
    labels = ["x", "y", "z", "wss_x", "wss_y", "wss_z", "wss"]
    if insert_normals:
        assert mesh_normals_canonical is not None, "insert_normals=True requires mesh_normals_canonical"
        normals = mesh_normals_canonical.to(images.device).unsqueeze(0).expand(T_out, -1, -1)
        images = torch.cat([images[..., :3], normals, images[..., 3:]], dim=-1)
        labels = labels[:3] + ["x_normal", "y_normal", "z_normal"] + labels[3:]
    return images.cpu(), labels, align


# ---------------------------------------------------------------------------
# Normalization (must match what the model was trained on)
# ---------------------------------------------------------------------------

def load_given_norm(steady_path):
    """Pull {'mean','std','label'} out of an assembled steady dataset .pth.

    The old pipeline saved its transient tensors ALREADY normalized with the
    steady set's per-channel stats, so a model trained on it expects the same
    normalization here.
    """
    steady = torch.load(steady_path, map_location="cpu", mmap=True)
    norm = steady["tensor_norm"]
    return {"mean": norm["mean"].flatten().clone(),
            "std": norm["std"].flatten().clone(),
            "label": list(steady["label"])}


def _align_norm_to_labels(given_norm, labels):
    """Reorder/broaden a given_norm onto `labels` by NAME, not position.

    Our tensors carry an extra 'wss' magnitude channel the old pipeline did not,
    so the 9-channel norm cannot be applied positionally. Channels absent from
    the norm are passed through untouched (mean 0, std 1).
    """
    if "label" not in given_norm:
        mean, std = given_norm["mean"].flatten(), given_norm["std"].flatten()
        if len(mean) != len(labels):
            raise ValueError(f"given_norm has {len(mean)} channels but tensor has {len(labels)}; "
                             "supply a 'label' list so channels can be matched by name")
        return mean, std

    lut = {name: i for i, name in enumerate(given_norm["label"])}
    mean = torch.zeros(len(labels))
    std = torch.ones(len(labels))
    for i, name in enumerate(labels):
        if name in lut:
            mean[i] = given_norm["mean"].flatten()[lut[name]]
            std[i] = given_norm["std"].flatten()[lut[name]]
    return mean, std


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def assemble_registered_transient_data(
    case_root, ghd_root=None,
    registry_path=DEFAULT_REGISTRY_PATH,
    aneurysm_types=None,               # e.g. [0] or [1, 2]; None = keep every type found
    datasets=None,                     # e.g. ["AneuX"]; None = keep every dataset source
    limit=None,                        # stop after this many successfully processed cases
    n_subdivide=1, downsampling_factors=(4, 4),
    raw_file="wall_data.pt", insert_normals=True,
    max_align_mean_mm=0.5, max_align_max_mm=0.5,
    refine_imperialnhs_with_icp=False,
    canonical_root=DEFAULT_CANONICAL_ROOT,
    assembled_write_path=None, overwrite_assembled=False,
    normalize=True, given_norm=None,
    device=None,
):
    """Per-case: look up dataset source + aneurysm_type in the registry,
    reconstruct+align the GHD surface mesh for that type, and register CFD wall
    physics onto it. Cases are grouped by aneurysm_type (each type has its own
    canonical topology, so mesh_data/downsampling hierarchy are built once per
    type, not globally).
    """
    if assembled_write_path is not None and os.path.exists(assembled_write_path) and not overwrite_assembled:
        return torch.load(assembled_write_path)

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    multi_recon = MultiCanonicalGHDReconstruct(canonical_root=canonical_root, device=device)
    registry = load_registry(registry_path)
    ghd_index, ghd_collisions = (build_ghd_dir_index(ghd_root, return_collisions=True)
                                 if ghd_root is not None else (None, {}))

    # atype -> per-case lists; mirrors record_mesh_upsampling's per-case accumulation
    by_type = {}
    skipped = {"not_in_registry": [], "no_ghd_files": [], "no_type": [],
               "type_filtered": [], "dataset_filtered": [], "ambiguous_ghd": [], "misaligned": []}
    alignment = {}  # case -> nearest-CFD-point distance stats (mm), kept for auditing

    case_names = sorted(d for d in os.listdir(case_root) if os.path.isdir(os.path.join(case_root, d)))
    for case in tqdm(case_names):
        case_dir = os.path.join(case_root, case)
        if not os.path.exists(os.path.join(case_dir, raw_file)):
            continue

        entry = registry.get(case)
        if entry is None:
            skipped["not_in_registry"].append(case)
            continue

        if datasets is not None and entry["dataset"] not in datasets:
            skipped["dataset_filtered"].append(case)
            continue

        aneurysm_type = entry["aneurysm_type"]
        if aneurysm_type is None:
            skipped["no_type"].append(case)
            continue
        if aneurysm_types is not None and aneurysm_type not in aneurysm_types:
            skipped["type_filtered"].append(case)
            continue

        dataset_source = entry["dataset"]
        ghd_dir, geomagic_path = resolve_case_paths(case, case_dir, dataset_source, ghd_index=ghd_index)
        if ghd_dir is None and case in ghd_collisions:
            skipped["ambiguous_ghd"].append(case)
            continue
        if ghd_dir is None or not os.path.exists(os.path.join(ghd_dir, "ghd_coefficients.npz")):
            skipped["no_ghd_files"].append(case)
            continue

        recon = reconstruct_case_meshes(multi_recon, ghd_dir, aneurysm_type, n_subdivide=n_subdivide)

        icp_refine = None
        if dataset_source == "ImperialNHS" and refine_imperialnhs_with_icp and geomagic_path and os.path.exists(geomagic_path):
            icp_refine = lambda pos, wss, _geomagic_path=geomagic_path: _refine_cfd_with_icp(pos, wss, _geomagic_path)

        # register physics using the WORLD-frame mesh only as the knn_interpolate
        # target, so each vertex samples the physically correct CFD wall shear.
        # Everything then stored -- x/y/z, normals AND the WSS vectors (via
        # wss_rotation) -- is expressed in the CANONICAL frame, so the tensor is
        # internally consistent and wss stays tangent to the stored normals.
        images, labels, align = register_wall_physics(
            case_dir,
            recon["subdivided_world"].verts_packed(),
            recon["subdivided_canonical"].verts_packed(),
            mesh_normals_canonical=recon["subdivided_canonical"].verts_normals_packed(),
            raw_file=raw_file, insert_normals=insert_normals, icp_refine=icp_refine, device=device,
            max_align_mean_mm=max_align_mean_mm, max_align_max_mm=max_align_max_mm,
            wss_rotation=recon["R_eq"],
        )
        alignment[case] = align
        if images is None:
            skipped["misaligned"].append(case)
            print(f"[skip] {case}: mesh not on CFD wall "
                  f"(NN mean {align['mean']:.3f} mm, max {align['max']:.3f} mm, "
                  f"centroid off {align['centroid_offset']:.3f} mm)")
            continue

        bucket = by_type.setdefault(aneurysm_type, {
            "cases": [], "dataset_source": [], "cleaned_meshes": [], "subdivided_meshes": [],
            "phi": [], "shape_scale": [], "tensors": [], "labels": labels,
        })
        bucket["cases"].append(case)
        bucket["dataset_source"].append(dataset_source)
        bucket["cleaned_meshes"].append(recon["cleaned_canonical"].cpu())
        bucket["subdivided_meshes"].append(recon["subdivided_canonical"].cpu())
        bucket["phi"].append(recon["phi"])
        bucket["shape_scale"].append(recon["shape_scale"])
        bucket["tensors"].append(images)

        if limit is not None and sum(len(b["cases"]) for b in by_type.values()) >= limit:
            break

    mesh_data = {}
    registered_data_list = {}
    for atype, bucket in by_type.items():
        # downsampling/pooling hierarchy is topology-only and shared across all
        # cases of this type, so it's derived once from a zero-deformation
        # template rather than per case (same approximation the old pipeline
        # made using cases[0]'s topology).
        template = get_type_template_mesh(multi_recon, atype, n_subdivide=n_subdivide)
        idx_list, edge_index_list, faces_list = get_downsampling_edge_index(template[0], factors=list(downsampling_factors))

        mesh_data[atype] = {
            "root": [case_root for _ in bucket["cases"]],
            "cases": bucket["cases"],
            "dataset_source": bucket["dataset_source"],
            "cleaned_ghd_Meshes": join_meshes_as_batch(bucket["cleaned_meshes"]),
            "subdivided_ghd_Meshes": join_meshes_as_batch(bucket["subdivided_meshes"]),
            "idx_list": idx_list,
            "edge_index_list": edge_index_list,
            "faces_list": faces_list,
            "ds_factors": list(downsampling_factors),
            "ghd": torch.stack(bucket["phi"], dim=0).squeeze(1),  # [B, num_coeffs, 3]
            # AngioFlow checkpoints carry no per-coefficient GHD normalization (see get_ghd_chk);
            # kept for interface parity with datasets_wss_optimized.GraphEncoder.
            "ghd_mean": torch.zeros((1, 3)),
            "ghd_std": torch.ones((1, 3)),
            "norm_canonical": 1.0,
            "shape_scale": torch.cat(bucket["shape_scale"], dim=0),  # [B, 1]
            "mask": get_type_mask(multi_recon, atype),
        }

        tensors = bucket["tensors"]
        if normalize:
            if given_norm is None:
                raise ValueError("normalize=True requires given_norm (see load_given_norm)")
            mean, std = _align_norm_to_labels(given_norm, bucket["labels"])
            tensors = [(t - mean.to(t.device)) / (std.to(t.device) + 1e-5) for t in tensors]
        registered_data_list[atype] = [
            {"tensor": tensor, "labels": bucket["labels"], "case": case, "dataset_source": ds}
            for tensor, case, ds in zip(tensors, bucket["cases"], bucket["dataset_source"])
        ]

    # Match the old pipeline's on-disk layout EXACTLY when a single aneurysm type
    # is present (the eval case): mesh_data is a flat dict and registered_data_list
    # a flat list, which is what datasets_wss_optimized.py's GraphEncoder /
    # Registered*ImageDataset iterate. Only keep the per-type nesting when several
    # types are mixed, since those have different canonical topologies and cannot
    # share one mesh_data.
    if len(mesh_data) == 1:
        (atype,) = mesh_data
        mesh_data, registered_data_list = mesh_data[atype], registered_data_list[atype]

    data = {"mesh_data": mesh_data, "registered_data_list": registered_data_list,
            "skipped": skipped, "alignment": alignment}
    if assembled_write_path is not None:
        torch.save(data, assembled_write_path)
    return data
