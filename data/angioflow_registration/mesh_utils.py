"""Mesh helpers for trimming cap vertices and building the GNN downsampling
hierarchy. Copied from AneuG-Net's new_version/loaders.py.
"""

import numpy as np
import torch
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes

from .mesh_ops import generate_transform_matrices, Mesh


def clean_meshes(meshes: Meshes, keep_indices: torch.Tensor) -> Meshes:
    """
    clip meshes and rearrange node indices in faces
    keep_indices: a tensor of indices of vertices to keep
    """
    assert meshes._N == len(meshes.verts_padded()), "Meshes must be batched"
    B = meshes._N
    device = meshes.device
    keep_indices = keep_indices.to(device)

    # Create a mapping from old vertex indices to new indices
    old_to_new = -torch.ones((meshes.num_verts_per_mesh()[0],), dtype=torch.long, device=device)
    old_to_new[keep_indices] = torch.arange(len(keep_indices), device=device)
    new_verts = []
    new_faces = []

    for b in range(B):
        verts = meshes.verts_padded()[b]
        faces = meshes.faces_padded()[b]
        # Filter vertices
        verts = verts[keep_indices]
        new_verts.append(verts)
        # Mask faces: keep only faces where all 3 vertices are in keep_indices
        face_mask = (old_to_new[faces].min(dim=-1).values >= 0)
        faces = faces[face_mask]
        # Remap face indices
        remapped_faces = old_to_new[faces]
        new_faces.append(remapped_faces)
    # Pad to batch again
    new_mesh = Meshes(verts=new_verts, faces=new_faces)
    return new_mesh


def get_trimmed_indice(trimmed_Meshes, canonical_Meshes):
    dists = knn_points(canonical_Meshes.verts_padded(), trimmed_Meshes.verts_padded(), K=1, return_nn=False)[0]
    verts2trim = torch.where(dists.view(-1) > 1e-5)[0]
    print("Trimmed vertices:", verts2trim.shape[0])
    total_verts = canonical_Meshes.verts_padded().shape[1]
    mask = torch.ones(total_verts, dtype=torch.bool, device=canonical_Meshes.device)
    mask[verts2trim] = False
    return torch.where(mask)[0]


def get_downsampling_edge_index(mesh: Meshes, factors=[4, 4]):
    mesh = Mesh(
        v=mesh.verts_packed().cpu().detach().numpy(),
        f=mesh.faces_packed().cpu().detach().numpy())
    M, _, _, _ = generate_transform_matrices(mesh, factors=factors, compute_upsampling=False)
    idx_list = get_mesh_downsampling_idx_batch(M)
    edge_index_list, faces_list = get_edge_index_list(M)
    return idx_list, edge_index_list, faces_list


def get_mesh_downsampling_idx_batch(M, atol=1e-6):
    idx_list = []
    # Loop through meshes, starting from the second mesh
    for i in range(1, len(M)):
        v_src = torch.tensor(M[i - 1].v, dtype=torch.float32)  # vertices of the previous mesh
        v_tgt = torch.tensor(M[i].v, dtype=torch.float32)   # vertices of the current mesh
        # Compute pairwise distances
        diff = v_src.unsqueeze(0) - v_tgt.unsqueeze(1)  # (N_tgt, N_src, 3)
        dists = torch.norm(diff, dim=2)                 # (N_tgt, N_src)
        # Find nearest source vertex for each target vertex
        min_dists, idx = dists.min(dim=1)               # (N_tgt,)
        # Check if any distance exceeds the tolerance
        if (min_dists > atol).any():
            raise ValueError("Some vertices in mesh_tgt do not match any vertex in mesh_src within tolerance.")
        idx_list.append(idx)
    return idx_list


def get_edge_index_list(M):
    edge_index_list = []
    faces_list = []
    for mesh in M:
        faces = torch.tensor(mesh.f.astype(np.int64), dtype=torch.long)  # (F, 3)
        edges = torch.cat([
            faces[:, [0, 1]],  # edge (i, j)
            faces[:, [1, 2]],  # edge (j, k)
            faces[:, [2, 0]]   # edge (k, i)
        ], dim=0)  # (3 * F, 2)
        edges = edges.sort(dim=1)[0]
        edges = torch.unique(edges, dim=0)  # (E, 2)
        # Add edge_index to the list
        edge_index_list.append(edges.t())  # (2, E)
        faces_list.append(faces)
    return edge_index_list, faces_list
