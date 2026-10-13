"""GHD reconstruction (single canonical template).

Copied from AneuGv2's models/ghd_reconstruct.py -- this version's
Graph_Harmonic_Deform supports the .npy eigenvector files AneuGv2's canonical
assets use, unlike AneuG's own (pickle-only) copy. Trimmed to __init__,
normalize, and ghd_forward_as_Meshes -- the only methods
MultiCanonicalGHDReconstruct needs.
"""

import torch
from pytorch3d.structures import Meshes

from .graph_harmonic_deformation import Graph_Harmonic_Deform


class GHD_Reconstruct(object):
    """
    Class for GHD reconstruction
    * This class stores the normalized canonical instead of the original
    * For forward_as_Meshes, denormalize_shape will return shape as the state just after alignment
    """

    def __init__(self, canonical_Meshes: Meshes, eigen_chk: str, num_Basis=12 ** 2, device=torch.device('cpu')):
        self.canonical_Meshes = canonical_Meshes.to(device)
        # normalize
        self.norm_canonical, self.canonical_Meshes = self.normalize()
        self.forward_norm_canonical = self.canonical_Meshes.verts_packed().norm(dim=-1).max()
        # GHD object
        self.canonical_ghd = Graph_Harmonic_Deform(base_shape=self.canonical_Meshes, num_Basis=num_Basis, eigen_chk=eigen_chk)
        # deformed shape
        self.deformed_Meshes = None
        self.GHD_eigvec = self.canonical_ghd.GBH_eigvec.to(device)

    def normalize(self):
        """
        Normalize the vertices to the canonical mesh
        """
        norm_canonical = torch.max(torch.norm(self.canonical_Meshes.verts_packed(), dim=-1)).detach().item() * 1.10
        norm_canonical *= 2.50
        canonical_Meshes = self.canonical_Meshes.update_padded(self.canonical_Meshes.verts_padded() / norm_canonical)
        return norm_canonical, canonical_Meshes

    def ghd_forward_as_Meshes(self, ghd: torch.Tensor, denormalize_shape=False, trimmed_faces=None, mean=None, std=None, scale=None):
        """
        we train our model in fitting env, set denormalize_shape=True to
        return the actual size
        """
        if mean is not None and std is not None:
            ghd = ghd * std.to(ghd.device) + mean.to(ghd.device)
        B = ghd.shape[0]
        ghd = ghd.reshape(B, -1, 3)
        Meshes_ = Meshes(verts=self.canonical_Meshes.verts_padded().repeat(B, 1, 1),
                         faces=self.canonical_Meshes.faces_padded().repeat(B, 1, 1)).to(ghd.device)
        offset = torch.einsum('nm,bmc->bnc', self.GHD_eigvec, ghd)  # [B, N, 3]
        verts_updated_padded = self.canonical_Meshes.verts_padded() + offset
        if denormalize_shape:
            verts_updated_padded = verts_updated_padded * self.norm_canonical
        if scale is not None:
            assert scale.dim() == 2
            verts_updated_padded = verts_updated_padded * scale.unsqueeze(-1)
        # update Meshes
        Meshes_ = Meshes_.update_padded(verts_updated_padded)
        if trimmed_faces is not None:
            faces = trimmed_faces.unsqueeze(0).repeat(B, 1, 1).to(Meshes_.device)
            Meshes_ = Meshes(verts=Meshes_.verts_padded(), faces=faces)
        return Meshes_
