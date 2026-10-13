"""Graph Harmonic Deformation base class (reconstruction only).

Copied from AneuGv2's ghd/base/graph_harmonic_deformation.py. AneuG's own copy
of this class only supports pickle eigen checkpoints; AneuGv2's canonical
assets ship eigenvectors.npy instead, so this AneuGv2-compatible version is copied
in rather than imported from the (independent) AneuG package.

Trimmed: dropped Graph_Harmonic_Deform_opening_alignment_dynamic, which is
fitting-only and needs the heavy ghd.fitting.registration import chain -- not
used for plain reconstruction.
"""

import pickle

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian, norm_laplacian
from pytorch3d.transforms import axis_angle_to_matrix
from torch_geometric.utils import to_undirected, get_laplacian, add_self_loops
from torch_scatter import scatter
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix


def mix_laplacian(base_shape: Meshes):
    device = base_shape.device
    cotweight, _ = cot_laplacian(base_shape.verts_packed(), base_shape.faces_packed())
    connection = cotweight.coalesce().indices()
    cotweight_value = cotweight.coalesce().values()
    connection, cotweight_value = to_undirected(connection, edge_attr=cotweight_value)
    normweight = norm_laplacian(base_shape.verts_packed(), base_shape.edges_packed())
    connection_1 = normweight.coalesce().indices()
    normweight_value = normweight.coalesce().values()
    _, normweight = to_undirected(connection_1, edge_attr=normweight_value)
    normweight_mean = scatter(normweight, connection[0], dim=0, reduce='add')
    cotweight_mean = scatter(cotweight_value, connection[0], dim=0, reduce='add')
    normweight_value = normweight_value / normweight_mean[connection[0]]
    cotweight_value = cotweight_value / cotweight_mean[connection[0]]
    norlap = add_self_loops(connection, edge_attr=normweight_value, fill_value='add', num_nodes=base_shape.verts_packed().shape[0])
    norlap_adj = add_self_loops(connection, edge_attr=normweight_value, fill_value=0, num_nodes=base_shape.verts_packed().shape[0])
    norlap = torch.sparse_coo_tensor(indices=norlap[0], values=norlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    norlap_adj = torch.sparse_coo_tensor(indices=norlap_adj[0], values=norlap_adj[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    norlap = norlap - 2 * norlap_adj
    norlap_np = coo_matrix((norlap.coalesce().values().cpu().numpy(), (norlap.coalesce().indices()[0].cpu().numpy(), norlap.coalesce().indices()[1].cpu().numpy())))
    cotlap = add_self_loops(connection, edge_attr=-cotweight_value, fill_value=1 + 1e-6, num_nodes=base_shape.verts_packed().shape[0])
    cotlap = torch.sparse_coo_tensor(indices=cotlap[0], values=cotlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    cotlap_np = coo_matrix((cotlap.coalesce().values().cpu().numpy(), (cotlap.coalesce().indices()[0].cpu().numpy(), cotlap.coalesce().indices()[1].cpu().numpy())))
    standerlap = get_laplacian(connection, num_nodes=base_shape.verts_packed().shape[0], dtype=torch.float32, normalization='sym')
    standerlap = torch.sparse_coo_tensor(indices=standerlap[0], values=standerlap[1], size=(base_shape.verts_packed().shape[0], base_shape.verts_packed().shape[0]), dtype=torch.float32, device=device)
    standerlap_np = coo_matrix((standerlap.coalesce().values().cpu().numpy(), (standerlap.coalesce().indices()[0].cpu().numpy(), standerlap.coalesce().indices()[1].cpu().numpy())))
    return cotlap_np, norlap_np, standerlap_np


class Graph_Harmonic_Deform(nn.Module):
    def __init__(self, base_shape: Meshes, num_Basis=6 * 6 + 1, mix_lap_weight=[1, 0.1, 0.1], eigen_chk=None):
        super(Graph_Harmonic_Deform, self).__init__()
        self.device = base_shape.device
        self.base_shape = base_shape
        self.cotlap_np, self.norlap_np, self.standerlap_np = mix_laplacian(base_shape)
        self.mix_lap = mix_lap_weight[0] * self.cotlap_np + mix_lap_weight[1] * self.norlap_np + mix_lap_weight[2] * self.standerlap_np
        if eigen_chk is not None:
            if str(eigen_chk).endswith(".npy"):
                eigvec = np.load(eigen_chk)
                self.GBH_eigvec = torch.from_numpy(eigvec).to(self.device).float()
                self.GBH_eigval = torch.zeros((1, self.GBH_eigvec.shape[-1]), device=self.device)
            else:
                with open(eigen_chk, 'rb') as f:
                    chk = pickle.load(f)
                for key_ in chk.keys():
                    setattr(self, key_, chk[key_].to(self.device))
        else:
            self.GBH_eigval, self.GBH_eigvec = eigsh(self.mix_lap, k=num_Basis, which='SM')
            self.GBH_eigvec = torch.from_numpy(self.GBH_eigvec).to(base_shape.device).float()
            self.GBH_eigval = torch.from_numpy(self.GBH_eigval).to(base_shape.device).float().unsqueeze(0)
        self.deformation_param = nn.Parameter(torch.zeros((num_Basis, 3), dtype=torch.float32, device=base_shape.device), requires_grad=True)
        self.reset_affine_param()

    def to(self, device):
        self.device = device
        self.base_shape = self.base_shape.to(device)
        self.GBH_eigvec = self.GBH_eigvec.to(device)
        self.GBH_eigval = self.GBH_eigval.to(device)
        self.deformation_param = nn.Parameter(self.deformation_param.to(device))
        self.R = nn.Parameter(self.R.to(device))
        self.s = nn.Parameter(self.s.to(device))
        self.T = nn.Parameter(self.T.to(device))
        return self

    def reset_affine_param(self):
        self.R = nn.Parameter(torch.zeros(1, 3, device=self.device))
        self.s = nn.Parameter(torch.tensor([1.], device=self.device).unsqueeze(0))
        self.T = nn.Parameter(torch.zeros(1, 3, device=self.device))

    def ghb_coefficient_recover(self, GHB_coefficient):
        assert GHB_coefficient.shape[0] == self.GBH_eigvec.shape[-1]
        try:
            n = GHB_coefficient.shape[1]
        except Exception:
            n = 1
            GHB_coefficient = GHB_coefficient.unsqueeze(-1)
        return self.GBH_eigvec.matmul(GHB_coefficient)

    def project_to_ghb_eig(self, input_shape):
        assert input_shape.shape[0] == self.GBH_eigvec.shape[0]
        return self.GBH_eigvec.transpose(-1, -2).matmul(input_shape)

    def forward(self, GHB_coefficient=None):
        if GHB_coefficient is None:
            GHB_coefficient = self.deformation_param
        deformation = self.ghb_coefficient_recover(GHB_coefficient)
        output_shape = self.base_shape.offset_verts(deformation)
        R_matrix = axis_angle_to_matrix(self.R)
        output_shape = output_shape.update_padded((output_shape.verts_padded() @ R_matrix.transpose(-1, -2) * self.s + self.T).float())
        return output_shape
