"""Turn a bare open surface mesh into the exact (wall, interior) graph pair the
trained AngioJointField model consumes -- the deployment-side replacement for the
whole offline preprocessing + dataset + transform stack.

WHY THIS EXISTS
---------------
In training, a sample reaches the model through three separate stages:

  1. angioflow/data/register_cfd_shapes_angioflow_v*.py  -> wall_mesh_registration.pt
     (remesh the CFD point cloud, label openings, locate the aneurysm)
  2. angioflow/data/datasets.py::AngioFlowWallDataset._process_case -> static_*.pt
     (normals, edge_index, per-opening geodesics, cotangent Laplacian eigenpairs)
  3. the eval transform chain in angioflow/training/eval_joint.py
     (crop -> centre -> normalize -> symmetrize edges)

At deployment none of that exists: 3D Slicer hands us a triangulated lumen surface
and the operator tells us which boundary loop is the inlet and which are outlets.
This module reproduces stages 2 and 3 from that input alone, and is validated to
reproduce stage 2's on-disk tensors bit-for-bit (see validate_against_dataset.py).

WHAT THE MODEL ACTUALLY READS
-----------------------------
Confirmed by grepping the encoder/decoder (angioflow/models/interior/encoders.py,
angioflow/models/joint/decoder.py) -- the forward pass touches ONLY:

  wall:      pos, normals, edge_index, geo_dist, cot_eigvec, cot_eigval, batch, t
  interior:  pos, wall_dist, batch

Everything else the training dataset carries is NOT a model input:
  * `y`               -- the WSS target (absent at inference, obviously)
  * `node_type`       -- consumed only by AneurysmCrop
  * `global_geo_dist` -- consumed only by AneurysmCrop
  * `flow_split`      -- never read by the model (surprisingly: the outlet split is
                         a CFD boundary condition that never reaches the network)
  * `aneurysm_type`   -- never read by the model

So `geo_dist` here is the PER-OPENING field ([N,3] = inlet / outlet_0 / outlet_1),
which is exactly computable once the operator identifies the openings. It must not
be confused with `global_geo_dist` ([N], distance to the aneurysm) which is a
cropping input only -- see the CROPPING note below.

CROPPING (read this before trusting an output)
----------------------------------------------
The shipped joint runs train with mask_high ~= 0.01, i.e. every training sample was
the aneurysm region plus a ~1%-deep geodesic ring -- NOT a whole vessel tree. Feeding
the full Slicer mesh is therefore an out-of-distribution input, and eval_joint.py
says as much ("evaluating on the full mesh would be an out-of-distribution input,
not a fairer test"). Reproducing the crop needs the aneurysm region, which in
training came from GHD registration and cannot be recovered from mesh + openings.

`aneurysm_mask` (or `aneurysm_seed` + `aneurysm_radius`) is therefore an explicit
input to this class. Supply it and the crop matches training; omit it and you get
`crop='none'`, which is honest but out-of-distribution.

UNITS: positions are METRES throughout (raw CFD frame), matching wall_data.pt.

DEPENDENCIES: compute_pe_cot_eigenvector pulls in pytorch3d via AneuG's
mix_laplacian. `cot_backend="scipy"` selects a dependency-free reimplementation
(numpy/scipy only) that is validated to agree with the pytorch3d path -- use it for
the Slicer transplant, where pytorch3d is not realistic.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# node_type codes -- must match register_cfd_shapes_angioflow_v*.py
WALL, ANEURYSM, INLET, OUTLET_0, OUTLET_1 = 0, 1, 2, 3, 4
OPENING_ORDER = (INLET, OUTLET_0, OUTLET_1)   # column order of geo_dist [N, 3]


# ---------------------------------------------------------------------------
# geometry primitives (mirrors angioflow/data/mesh.py, kept dependency-light)
# ---------------------------------------------------------------------------

def compute_vertex_normals(points, faces, backend="vtk"):
    """Vertex normals.

    backend='vtk' reproduces training EXACTLY: angioflow.data.mesh uses
    pyvista's compute_normals(auto_orient_normals=True), whose consistent
    outward orientation a plain area-weighted sum does not reproduce (~5%
    relative difference measured). VTK ships inside 3D Slicer, so this is also
    the portable choice there.

    backend='numpy' is the dependency-free area-weighted fallback; it is CLOSE
    but not identical, so prefer 'vtk' unless VTK is genuinely unavailable.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if backend == "vtk":
        import pyvista as pv
        cells = np.concatenate([np.full((faces.shape[0], 1), faces.shape[1]), faces],
                               axis=1).ravel()
        mesh = pv.PolyData(points, cells).compute_normals(
            point_normals=True, cell_normals=False, auto_orient_normals=True)
        return np.asarray(mesh.point_data["Normals"], dtype=np.float64)
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)                    # area-weighted face normal
    vn = np.zeros_like(points)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.clip(norms, 1e-20, None)


def face_to_edge_index(faces):
    """Undirected mesh edges stored ONCE as i<j -- byte-identical convention to
    angioflow.data.mesh.face_to_edge_index. SymmetrizeEdges expands it later."""
    faces = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    e = np.unique(e, axis=0)
    return e.T.copy()                                   # [2, E]


def per_opening_geo_dist(pos, edge_index, node_type):
    """[N, 3] geodesic distance to (inlet, outlet_0, outlet_1).

    Multi-source Dijkstra over the mesh graph with Euclidean edge weights, each
    column min-pooled over its opening's nodes and scaled to [0, 1] by its own
    finite max. An ABSENT opening (e.g. single-outlet cases) is filled with 1.0,
    meaning "infinitely far / not present" -- same padding datasets.py uses, so
    single- and double-outlet cases share one tensor layout.
    """
    pos = np.asarray(pos, dtype=np.float64)
    edge_index = np.asarray(edge_index)
    node_type = np.asarray(node_type)
    N = len(pos)

    i, j = edge_index
    w = np.linalg.norm(pos[i] - pos[j], axis=1)
    adj = csr_matrix((np.concatenate([w, w]),
                      (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(N, N))

    cols = []
    for tid in OPENING_ORDER:
        src = np.where(node_type == tid)[0]
        if src.size == 0:
            cols.append(np.ones(N))
            continue
        d = dijkstra(adj, directed=False, indices=src, min_only=True)
        finite = np.isfinite(d)
        dmax = d[finite].max() if finite.any() else 1.0
        cols.append(d / max(dmax, 1e-8))
    return torch.from_numpy(np.stack(cols, axis=-1)).float()


def geodesic_from_nodes(pos, edge_index, source_nodes):
    """[N] geodesic distance to a node set, normalized to [0, 1]. Used to build
    `global_geo_dist` for cropping when the operator marks the aneurysm."""
    pos = np.asarray(pos, dtype=np.float64)
    edge_index = np.asarray(edge_index)
    N = len(pos)
    i, j = edge_index
    w = np.linalg.norm(pos[i] - pos[j], axis=1)
    adj = csr_matrix((np.concatenate([w, w]),
                      (np.concatenate([i, j]), np.concatenate([j, i]))), shape=(N, N))
    d = dijkstra(adj, directed=False, indices=np.asarray(source_nodes), min_only=True)
    finite = np.isfinite(d)
    dmax = d[finite].max() if finite.any() else 1.0
    return torch.from_numpy(d / max(dmax, 1e-8)).float()


# ---------------------------------------------------------------------------
# cotangent Laplacian eigenpairs
# ---------------------------------------------------------------------------

def _cot_laplacian_scipy(points, faces):
    """Row-normalized cotangent Laplacian as a scipy COO matrix.

    Portable reimplementation of AneuG's mix_laplacian cotangent branch (the only
    branch compute_pe_cot_eigenvector uses):
        W_ij  = cotangent weight of edge (i,j), symmetrized
        W_ij <- W_ij / sum_k W_ik                      (row-normalize)
        L     = (1 + 1e-6) I - W
    No pytorch3d / torch_geometric -- safe for the 3D Slicer transplant.
    """
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    N = len(points)

    # --- pytorch3d.ops.cot_laplacian, ported exactly ---
    # NOTE the area must be the HERON area of the triangle, not |cross| (= 2*area),
    # and there is no extra 1/2: cot_a = (b^2 + c^2 - a^2) / (4 * area). Getting
    # either wrong changes the eigenvectors enough to matter (validated: this port
    # agrees with pytorch3d to ~3e-7 relative, i.e. float32 noise).
    v0, v1, v2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    a = np.linalg.norm(v1 - v2, axis=1)
    b = np.linalg.norm(v0 - v2, axis=1)
    c = np.linalg.norm(v0 - v1, axis=1)
    s = 0.5 * (a + b + c)
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 1e-12, None))
    cota = (b ** 2 + c ** 2 - a ** 2) / area / 4.0
    cotb = (a ** 2 + c ** 2 - b ** 2) / area / 4.0
    cotc = (a ** 2 + b ** 2 - c ** 2) / area / 4.0

    # each cot sits on the edge OPPOSITE its vertex; symmetrize as pytorch3d's L += L.t()
    ii = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    jj = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    ww = np.concatenate([cota, cotb, cotc])
    W = coo_matrix((ww, (ii, jj)), shape=(N, N)).tocsr()
    W = W + W.T

    # mix_laplacian then row-normalizes (to_undirected's doubling cancels here) and
    # builds L = (1 + 1e-6) I - W_norm -- deliberately NON-symmetric.
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    Dinv = csr_matrix((1.0 / np.clip(rowsum, 1e-20, None), (np.arange(N), np.arange(N))),
                      shape=(N, N))
    Wn = Dinv @ W
    eye = csr_matrix((np.full(N, 1.0 + 1e-6), (np.arange(N), np.arange(N))), shape=(N, N))
    return (eye - Wn).tocoo()


def compute_cot_eigenpairs(points, faces, k=8, backend="pytorch3d"):
    """(cot_eigval [k], cot_eigvec [N, k]) -- the smallest-magnitude eigenpairs.

    backend='pytorch3d' calls the EXACT function the training pipeline used
    (angioflow.data.graph.compute_pe_cot_eigenvector); 'scipy' uses the portable
    reimplementation above. Both are validated to agree.
    """
    if backend == "pytorch3d":
        from angioflow.data.graph import compute_pe_cot_eigenvector
        return compute_pe_cot_eigenvector(np.asarray(points, dtype=np.float64),
                                          np.asarray(faces, dtype=np.int64), k=k)
    if backend != "scipy":
        raise ValueError(f"unknown cot_backend {backend!r} (want 'pytorch3d' or 'scipy')")
    lap = _cot_laplacian_scipy(points, faces)
    val, vec = eigsh(lap, k=k, which="SM")
    return torch.from_numpy(val).float(), torch.from_numpy(vec).float()


# ---------------------------------------------------------------------------
# mesh regularization (roughness + resolution matching)
# ---------------------------------------------------------------------------

def regularize_surface(vertices_mm, faces, labels=None, regularizer=None, **kwargs):
    """Smooth + remesh a segmentation surface to the training data's style.

    A Slicer segmentation carries voxel-scale roughness and sits at the acquisition's
    own resolution (~0.4-0.5 mm edges); the training meshes are ~0.125 mm and much
    smoother (DOWNSTREAM_EDGE_MM, measured from angioflowv2_merged -- the very
    dataset this model trained on). A graph network tolerates some resolution
    change, but there is no reason to spend accuracy on it.

    Uses AneuG.mesh_regularizer.MeshRegularizer, which works in MILLIMETRES --
    hence `vertices_mm`.

    REMESHING CHANGES THE VERTEX SET: count, order and positions all differ
    afterwards, so any per-vertex label (inlet / outlet / crop mask) supplied for
    the ORIGINAL mesh is meaningless against the new one. `labels` are therefore
    re-mapped by nearest neighbour onto the regularized vertices and returned
    alongside it -- caller-side indices must be rebuilt from these, never reused.

    labels: dict name -> [N] array (bool mask or int per-vertex label)
    returns (vertices_mm [M,3], faces [F,3], mapped_labels, report)
    """
    import trimesh
    from scipy.spatial import cKDTree

    if regularizer is None:
        from AneuG.mesh_regularizer import MeshRegularizer
        regularizer = MeshRegularizer(**kwargs)

    src = trimesh.Trimesh(vertices=np.asarray(vertices_mm, dtype=np.float64),
                          faces=np.asarray(faces, dtype=np.int64), process=False)
    out, report = regularizer.forward(src)
    new_v = np.asarray(out.vertices, dtype=np.float64)
    new_f = np.asarray(out.faces, dtype=np.int64)

    mapped = {}
    if labels:
        # nearest ORIGINAL vertex for each NEW vertex -- the surfaces are within a
        # fraction of an edge length of each other, so nearest-neighbour transfer is
        # well posed for region labels. Openings stay openings because the boundary
        # is frozen during smoothing (freeze_boundary=True).
        _, idx = cKDTree(np.asarray(vertices_mm, dtype=np.float64)).query(new_v, k=1)
        for name, arr in labels.items():
            mapped[name] = np.asarray(arr)[idx]
    return new_v, new_f, mapped, report


# ---------------------------------------------------------------------------
# the preprocessor
# ---------------------------------------------------------------------------

class SurfaceMeshToModelInput:
    """Bare surface mesh (+ opening labels) -> model-ready (wall, interior) pair.

    Parameters mirror the training run's args; defaults match the shipped joint
    checkpoints (lap_k=8, n_geo_dist=3, decimate_ratio=1.0 i.e. no decimation).

    stats: the two on-disk stat dicts the run normalized with --
        wss_stats      = torch.load(<save_dir>/stats.pt)
        interior_stats = torch.load(<save_dir>/interior_stats[_zeromax].pt)
    Pass the SAME files the checkpoint trained against; they set the position and
    field scales, so a mismatch silently rescales every prediction.

    input_units: units of the INCOMING vertices / query_points. The model and all
        the dataset statistics are in METRES, but 3D Slicer (like DICOM) works in
        MILLIMETRES, so a Slicer segmentation must be scaled by 1e-3 on the way in.
        Getting this wrong is silent: the mesh is simply 1000x too big, every
        normalized coordinate blows up, and the model returns confident nonsense.
        Accepts 'm', 'mm', 'cm', or a float metres-per-unit factor.

        OUTPUTS are unaffected -- WSS (Pa/s), velocity (m/s) and pressure (Pa) are
        physical quantities, so they come back in SI regardless of the input units.
    """

    _UNIT_TO_M = {"m": 1.0, "metre": 1.0, "meter": 1.0,
                  "mm": 1e-3, "millimetre": 1e-3, "millimeter": 1e-3,
                  "cm": 1e-2, "centimetre": 1e-2, "centimeter": 1e-2}

    def __init__(self, wss_stats, interior_stats, lap_k=8,
                 cot_backend="pytorch3d", mask_low=0.0, mask_high=0.01,
                 aneurysm_only=False, input_units="m", regularize=False,
                 regularizer_kwargs=None):
        self.to_metres = self._parse_units(input_units)
        self.input_units = input_units
        self.regularize = regularize
        self.regularizer_kwargs = regularizer_kwargs
        self.lap_k = lap_k
        self.cot_backend = cot_backend
        self.mask_low = mask_low
        self.mask_high = mask_high
        self.aneurysm_only = aneurysm_only
        # JointNormalize: positions use the INTERIOR pos_std for both streams
        self.pos_std = interior_stats["pos_std"]
        self.pos_scalar = interior_stats["pos_std"][0]
        self.y_mean, self.y_std = wss_stats["y_mean"], wss_stats["y_std"]
        self.vel_mean, self.vel_std = interior_stats["vel_mean"], interior_stats["vel_std"]
        self.press_mean, self.press_std = interior_stats["press_mean"], interior_stats["press_std"]

    @classmethod
    def _parse_units(cls, units):
        if isinstance(units, (int, float)):
            return float(units)
        key = str(units).strip().lower()
        if key not in cls._UNIT_TO_M:
            raise ValueError(f"unknown input_units {units!r}; use one of "
                             f"{sorted(cls._UNIT_TO_M)} or a metres-per-unit float")
        return cls._UNIT_TO_M[key]

    # -- stage 2: geometry features (reproduces datasets.py::_process_case) ----

    def build_static(self, vertices, faces, inlet_nodes, outlet_nodes,
                     aneurysm_mask=None, crop_mask=None, regularize=None,
                     regularizer=None):
        """Everything derivable from mesh + opening labels, in the RAW metric frame.

        vertices      [N, 3] float, in `input_units` (converted to metres here)
        faces         [F, 3] int
        inlet_nodes   indices of the inlet boundary loop
        outlet_nodes  sequence of index arrays, one per outlet (1 or 2 supported)
        aneurysm_mask [N] bool or None -- marks the aneurysm sac (cropping only)
        crop_mask     [N] bool or None -- Slicer's segmentation; pass it HERE (not
                      just to build_pair) when regularizing, so it is re-mapped onto
                      the remeshed vertices. The mapped copy is returned in
                      static["crop_mask"].
        regularize    smooth + remesh to the training resolution first (default:
                      the class-level setting). Required for real segmentation
                      surfaces; leave off for meshes already at training style.
        """
        # everything downstream (pos_std, crop_slack, wall_dist) is in METRES
        pos = np.asarray(vertices, dtype=np.float64) * self.to_metres
        faces = np.asarray(faces, dtype=np.int64)
        N = len(pos)

        node_type = np.zeros(N, dtype=np.int64)
        if aneurysm_mask is not None:
            node_type[np.asarray(aneurysm_mask, dtype=bool)] = ANEURYSM
        node_type[np.asarray(inlet_nodes, dtype=np.int64)] = INLET
        for slot, out_nodes in enumerate(outlet_nodes):
            if slot > 1:
                raise ValueError("at most two outlets are supported (outlet_0, outlet_1)")
            node_type[np.asarray(out_nodes, dtype=np.int64)] = OUTLET_0 + slot

        regularize = self.regularize if regularize is None else regularize
        reg_report = None
        if regularize:
            # MeshRegularizer speaks millimetres; everything else here is metres
            labels = {"node_type": node_type}
            if crop_mask is not None:
                labels["crop_mask"] = np.asarray(crop_mask, dtype=bool)
            new_mm, faces, mapped, reg_report = regularize_surface(
                pos * 1e3, faces, labels=labels, regularizer=regularizer,
                **(self.regularizer_kwargs or {}))
            pos = new_mm * 1e-3
            node_type = mapped["node_type"].astype(np.int64)
            crop_mask = mapped.get("crop_mask")
            N = len(pos)

        normals = compute_vertex_normals(pos, faces)
        edge_index = face_to_edge_index(faces)
        cot_eigval, cot_eigvec = compute_cot_eigenpairs(
            pos, faces, k=self.lap_k, backend=self.cot_backend)
        if not (torch.isfinite(cot_eigval).all() and torch.isfinite(cot_eigvec).all()):
            raise ValueError("non-finite cotangent Laplacian eigenpairs (degenerate mesh?)")

        geo_dist = per_opening_geo_dist(pos, edge_index, node_type)

        # Centring, stage 1 of 2. datasets.py centres the stored wall on the FULL-RES
        # mesh mean, and datasets_interior.py subtracts THE SAME offset from the
        # interior points so both streams share one coordinate frame. We must record
        # that offset and apply it to the query points in build_pair -- a Slicer mesh
        # arrives at arbitrary world coordinates, so skipping it would leave the wall
        # centred and the query points still out at their world position.
        center = pos.mean(axis=0, keepdims=True)
        centred = pos - center

        static = {
            "center": torch.from_numpy(center).float(),      # [1, 3] world -> centred
            "pos": torch.from_numpy(centred).float(),
            "edge_index": torch.from_numpy(edge_index),
            "faces": torch.from_numpy(faces),
            "node_type": torch.from_numpy(node_type),
            "normals": torch.from_numpy(normals).float(),
            "geo_dist": geo_dist,
            "cot_eigval": cot_eigval,
            "cot_eigvec": cot_eigvec,
        }
        if aneurysm_mask is not None:
            src = np.where(node_type == ANEURYSM)[0]
            static["global_geo_dist"] = geodesic_from_nodes(pos, edge_index, src)
        if crop_mask is not None:
            static["crop_mask"] = torch.as_tensor(np.asarray(crop_mask), dtype=torch.bool)
        if reg_report is not None:
            static["regularizer_report"] = reg_report
        return static

    # -- stage 3: the eval transform chain ------------------------------------

    def build_pair(self, static, query_points, t=0, crop_mask=None, crop=False, seed=None):
        """(wall Data, interior Data) ready for AngioJointField.forward.

        Applies, in the SAME order as eval_joint.py:
          JointAneurysmCrop -> CenterAndRandomRotate(rotate=False)
              -> JointNormalize -> SymmetrizeEdges

        crop_mask [N] bool -- the region to keep, i.e. Slicer's segmentation output.
        This is the DEPLOYMENT path: `static` must describe the FULL mesh, because
        training computes geo_dist and the cotangent eigenvectors globally and only
        then restricts them to the kept nodes. Passing an already-cropped mesh into
        build_static instead would silently change both features (the crop contains
        no inlet/outlet nodes at all, so geo_dist would collapse to its all-1.0
        "opening absent" padding).

        crop=True instead derives the mask from `global_geo_dist` the way training's
        AneurysmCrop does -- only useful for reproducing training-time randomness.

        query_points [P, 3] metres, in the same raw frame as `vertices`: the
        interior locations to predict velocity/pressure at. `wall_dist` is measured
        against the FULL (uncropped) wall, exactly as datasets_interior.py does.
        """
        wall_pos = static["pos"].clone()
        # same unit conversion build_static applied to the vertices
        int_pos = torch.as_tensor(
            np.asarray(query_points, dtype=np.float64) * self.to_metres).float()
        # bring the query points into the SAME frame build_static put the wall in.
        # `query_points` arrive in raw world coordinates (Slicer), whereas static["pos"]
        # has already had `center` removed -- without this the two streams are offset
        # by the mesh centroid and every distance/position the model sees is wrong.
        int_pos = int_pos - static["center"]

        # wall_dist against the FULL wall, in metres (pre-crop, pre-normalize).
        # Translation-invariant, so it does not matter that both are now centred.
        wall_dist = torch.from_numpy(
            cKDTree(wall_pos.numpy()).query(int_pos.numpy())[0]).float()

        # a crop mask carried on `static` has already been re-mapped onto the
        # regularized vertices; a mask passed in here refers to the ORIGINAL mesh and
        # would be wrong after remeshing, so the stored one wins.
        if crop_mask is None and "crop_mask" in static:
            crop_mask = static["crop_mask"]
        elif crop_mask is not None and "crop_mask" in static:
            crop_mask = static["crop_mask"]

        keep = None
        if crop_mask is not None:
            keep = torch.as_tensor(np.asarray(crop_mask), dtype=torch.bool)
        elif crop:
            if "global_geo_dist" not in static:
                raise ValueError(
                    "crop=True needs an aneurysm region: pass aneurysm_mask to build_static, "
                    "or pass crop_mask=<segmentation>, or leave both off to run the full mesh")
            g = static["global_geo_dist"]
            if self.aneurysm_only:
                keep = (static["node_type"] == ANEURYSM) | (g == 0)
            else:
                gen = None
                if seed is not None:
                    gen = torch.Generator().manual_seed(seed)
                p = torch.empty(()).uniform_(self.mask_low, self.mask_high, generator=gen)
                keep = (static["node_type"] == ANEURYSM) | (g <= p * g.max())

        wall = Data(
            pos=wall_pos,
            edge_index=static["edge_index"],
            normals=static["normals"],
            geo_dist=static["geo_dist"],
            cot_eigvec=static["cot_eigvec"],
            cot_eigval=static["cot_eigval"],
            node_type=static["node_type"],
            t=torch.tensor([t]),
        )
        query_index = torch.arange(int_pos.shape[0])
        if keep is not None:
            wall = wall.subgraph(keep)
            # drop query points whose local wall context did not survive (JointAneurysmCrop)
            d_kept = cKDTree(wall.pos.numpy()).query(int_pos.numpy())[0]
            alive = torch.from_numpy(d_kept) <= 1.5 * wall_dist + 5e-4
            int_pos, wall_dist = int_pos[alive], wall_dist[alive]
            query_index = query_index[alive]

        # centre BOTH streams on the (possibly cropped) wall centroid -- CenterAndRandomRotate
        c = wall.pos.mean(0, keepdim=True)
        wall.pos = wall.pos - c
        int_pos = int_pos - c

        # JointNormalize
        wall.pos = wall.pos / self.pos_std
        int_pos = int_pos / self.pos_std
        wall_dist = wall_dist / self.pos_scalar

        # SymmetrizeEdges (wall only)
        wall.edge_index = to_undirected(wall.edge_index, num_nodes=wall.num_nodes)

        # query_index maps each surviving interior row back to its row in the caller's
        # `query_points`. The crop silently drops points whose wall neighbourhood was
        # removed, so WITHOUT this the caller cannot tell which query it got an answer
        # for -- essential both for scoring against ground truth and for writing results
        # back onto the right Slicer points.
        interior = Data(pos=int_pos, wall_dist=wall_dist, t=torch.tensor([t]),
                        query_index=query_index)
        return wall, interior

    # -- denormalization (mirrors JointNormalize.inverse_*) -------------------

    def inverse_y(self, y):
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)

    def inverse_vel(self, vel):
        return vel * self.vel_std.to(vel.device) + self.vel_mean.to(vel.device)

    def inverse_press(self, press):
        return press * self.press_std.to(press.device) + self.press_mean.to(press.device)
