"""Build Graph Features"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch
import ensightreader
from scipy.spatial import cKDTree
from torch_geometric.data import Data, Dataset
from pytorch3d.structures import Meshes
import pyvista as pv
from AneuG.ghd.base.graph_harmonic_deformation import mix_laplacian
from scipy.sparse.linalg import eigsh

from angioflow.data.mesh import decimate_zone_mesh, face_to_edge_index, fix_encas_geometry_model, compute_vertex_normals

BC_PARTS = {1: ["surface4"], 2: ["surface5", "surface6"]}  # label -> ensight part names


# def compute_bc_labels(decimated_pts, geofile, tol=1e-6):
#     labels = np.zeros(decimated_pts.shape[0], dtype=np.int64)
#     with open(geofile.file_path, "rb") as fp:
#         for label, names in BC_PARTS.items():
#             for name in names:
#                 part = geofile.get_part_by_name(name)
#                 if part is None:
#                     continue
#                 ring_pts = np.asarray(part.read_nodes(fp), dtype=np.float64)
#                 d, _ = cKDTree(ring_pts).query(decimated_pts, k=1)
#                 labels[d < tol] = label
#     return labels

def to_Meshes(pts, faces):
    verts = torch.from_numpy(pts).float().unsqueeze(0)  # [1, N, 3]
    faces = torch.from_numpy(faces).long().unsqueeze(0)  # [1, F, 3]
    return Meshes(verts=verts, faces=faces)

def compute_pe_cot_eigenvector(pts, faces, k=6):
    """
    PE: cotangent eigenvectors
    
    """
    # build Meshes
    meshes = to_Meshes(pts, faces)
    cotlap_np, _, _ = mix_laplacian(meshes)
    cot_eigval_, cot_eigvec_ = eigsh(cotlap_np, k=k, which='SM')
    cot_eigval = torch.from_numpy(cot_eigval_).float() # [k]
    cot_eigvec = torch.from_numpy(cot_eigvec_).float() # [N, k]
    return cot_eigval, cot_eigvec
