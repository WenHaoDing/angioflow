"""
Align GHD data (in world coordinates) to a CFD coordinate.
"""

import os
import numpy as np
import torch
import json
from AneuG.models.ghd_reconstruct import GHD_Reconstruct
import open3d as o3d
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from pytorch3d.structures import Meshes


def get_alignment_components(ghd_chk_path, landmarks_chk_path, metrics_chk_path=None):
    ghd_chk = np.load(ghd_chk_path, allow_pickle=True)
    phi= torch.from_numpy(ghd_chk['phi']).view(-1, 3)
    w_rot = torch.from_numpy(ghd_chk['w_rot']).view(-1, 3)
    s = torch.tensor(float(np.exp(ghd_chk['log_scale'])))
    t_vec = torch.from_numpy(ghd_chk['t_vec']).view(-1, 3)
    R = _so3_exp_map(w_rot)
    with open(metrics_chk_path, "r") as f:
        metrics = json.load(f)
    landmarks = np.load(landmarks_chk_path)
    s_can = metrics['s_can']
    R_frame = torch.from_numpy(landmarks['R_frame']).float()
    neck_pt = torch.from_numpy(landmarks['neck_pt_world']).float()
    return phi, R, s, t_vec, s_can, R_frame, neck_pt


def _so3_exp_map(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map: axis-angle vector → SO(3) matrix."""
    theta = torch.norm(w) + 1e-12
    k = (w / theta).view(-1)
    K = torch.stack([
        torch.stack([torch.tensor(0., device=w.device),  -k[2],  k[1]]),
        torch.stack([ k[2],  torch.tensor(0., device=w.device), -k[0]]),
        torch.stack([-k[1],   k[0], torch.tensor(0., device=w.device)]),
    ])
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def get_rigid_transform(coord_src, coord_tgt, voxel_size=None, return_matrix=False):
    """
    Rigid transform (R, t) mapping coord_src -> coord_tgt frame via
    voxel downsample -> FPFH -> RANSAC -> point-to-plane ICP.

    coord_src, coord_tgt : [N, 3] or [T, N, 3] (torch.Tensor or np.ndarray)
    Returns (R, t) such that p_tgt ≈ p_src @ R.T + t, or a 4x4 matrix if return_matrix.
    """

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3:
            x = x[0]
        return np.ascontiguousarray(x, dtype=np.float64)

    src_np = _to_np(coord_src)
    tgt_np = _to_np(coord_tgt)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_np)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tgt_np)

    if voxel_size is None:
        extent = tgt.get_axis_aligned_bounding_box().get_extent()
        voxel_size = float(np.linalg.norm(extent)) / 50.0

    src_down = src.voxel_down_sample(voxel_size)
    tgt_down = tgt.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2.0
    normal_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    src_down.estimate_normals(normal_param)
    tgt_down.estimate_normals(normal_param)
    tgt.estimate_normals(normal_param)  # needed for point-to-plane ICP

    radius_feature = voxel_size * 5.0
    feat_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(src_down, feat_param)
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(tgt_down, feat_param)

    ransac_dist = voxel_size * 1.5
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    icp_dist = voxel_size * 0.4
    icp_result = o3d.pipelines.registration.registration_icp(
        src, tgt, icp_dist, ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    T = np.asarray(icp_result.transformation, dtype=np.float64)
    print(f"Global reg fitness/inlier-rmse: {ransac_result.fitness:.4f} / {ransac_result.inlier_rmse:.6f}")
    print(f"ICP fitness/inlier-rmse:         {icp_result.fitness:.4f} / {icp_result.inlier_rmse:.6f}")

    if return_matrix:
        return torch.from_numpy(T).float()
    R = torch.from_numpy(T[:3, :3].copy()).float()
    t = torch.from_numpy(T[:3, 3].copy()).float()
    return R, t


def align_GHD_coord_to_CFD_(phi, R, s, t_vec, s_can, R_frame, neck_pt,
                           R_world2cfd, t_world2cfd,
                           unit_factor = 0.001,  # mm -> m
                           ghd_reconstruct: GHD_Reconstruct=None):
    """
    canonical -> world:
    equivalent formula:
    w = c @ R_equivalent.T * s_equivalent + t_equivalent
    c = (w - t_equivalent) / s_equivalent @ R_equivalent
    R_equivalent = (R_frame.T @ R)
    s_equivalent = s * s_can
    t_equivalent = (t_vec @ R_frame) * s_can + neck_pt
    """
    # compute equivalent transform
    R_equivalent = R_frame.T @ R
    t_equivalent = s_can * t_vec @ R_frame + neck_pt
    s_equivalent = s * s_can
    # GHD reconstructed shape -> CFD shape
    ghd_coord = ghd_reconstruct.ghd_forward_as_Meshes(phi.unsqueeze(0)).verts_packed() # [N, 3]
    ghd_coord = ghd_coord @ R_equivalent.T * s_equivalent + t_equivalent
    ghd_coord = ghd_coord @ R_world2cfd.T + t_world2cfd
    ghd_coord = ghd_coord * unit_factor
    return ghd_coord

def align_GHD_coord_to_CFD(case_dir, cfd_coord, ghd_reconstruct: GHD_Reconstruct=None):
    # get alignment components (AneuGv2 format)
    ghd_chk_path = os.path.join(case_dir, "ghd_coefficients.npz")
    landmarks_chk_path = os.path.join(case_dir, "landmarks.npz")
    metrics_chk_path = os.path.join(case_dir, "metrics.json")
    world_mesh = pv.read(os.path.join(case_dir, "geomagic_processed.obj"))
    phi, R, s, t_vec, s_can, R_frame, neck_pt = get_alignment_components(ghd_chk_path, landmarks_chk_path, metrics_chk_path)
    # Note: world -> CFD transform is caused by software shifting and must be resolved with geomagic_processed.obj.
    # world_mesh is in mm; cfd_coord is in m. Register in mm space (cfd_coord * 1000), so the resulting
    # transform lives in mm and the unit_factor=0.001 below brings the GHD coord back to m to match cfd_coord.
    R_world2cfd, t_world2cfd = get_rigid_transform(world_mesh.points, cfd_coord * 1000.0)
    # align GHD coord to CFD coord
    aligned_ghd_coord = align_GHD_coord_to_CFD_(phi, R, s, t_vec, s_can, R_frame, neck_pt,
                           R_world2cfd, t_world2cfd,
                           unit_factor=0.001,
                           ghd_reconstruct=ghd_reconstruct)
    ghd_Meshes = Meshes(verts=aligned_ghd_coord.unsqueeze(0), faces=ghd_reconstruct.canonical_Meshes.faces_padded())
    return ghd_Meshes

def align_GHD_coord_to_CFD_aneug_flow(case_dir, cfd_coord, ghd_reconstruct: GHD_Reconstruct=None):
    ghd_chk_path = os.path.join(case_dir, 'checkpoint.npy')
    ghd_chk = np.load(ghd_chk_path, allow_pickle=True).item()
    phi = torch.from_numpy(ghd_chk['ghd']).view(-1, 3)
    mean_phi = torch.from_numpy(ghd_chk['mean_ghd']).view(-1, 3)
    std_phi = torch.from_numpy(ghd_chk['std_ghd']).view(-1, 3)
    s = ghd_chk['cpcd_scale']
    norm_canonical = torch.from_numpy(ghd_chk['norm_canonical']).float()
    phi = phi * std_phi + mean_phi
    phi = phi.unsqueeze(0)
    ghd_Meshes = ghd_reconstruct.ghd_forward_as_Meshes(phi).to(cfd_coord.device)
    ghd_Meshes = Meshes(verts=ghd_Meshes.verts_padded() * 0.001 * s * norm_canonical, faces=ghd_reconstruct.canonical_Meshes.faces_padded())
    return ghd_Meshes



def _max_edge_len(pts, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    return float(np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1).max())


def compute_geo_distance(coord_cfd, faces_cfd, ghd_Meshes, extra_edges=None):
    """Graph-geodesic distance [N] from each CFD node to the GHD-overlap region.

    Seed = CFD nodes within one (max) edge length of a GHD vertex; distance is then
    multi-source Dijkstra over the CFD mesh (edge weights = Euclidean edge length).
    `extra_edges` ([M, 2] int64, optional) are unioned into the graph — e.g. KNN edges
    connecting orphan nodes (not touched by any face) into the rest of the mesh.
    """
    coord_cfd = np.asarray(coord_cfd, dtype=np.float64)
    faces_cfd = np.asarray(faces_cfd, dtype=np.int64)
    ghd_verts = ghd_Meshes.verts_packed().detach().cpu().numpy().astype(np.float64)
    ghd_faces = ghd_Meshes.faces_packed().detach().cpu().numpy().astype(np.int64)

    # seed: CFD nodes overlapping the GHD surface
    tol = max(_max_edge_len(coord_cfd, faces_cfd), _max_edge_len(ghd_verts, ghd_faces))
    d_to_ghd, _ = cKDTree(ghd_verts).query(coord_cfd, k=1)
    seed = np.where(d_to_ghd < tol)[0]
    assert seed.size > 0, "no CFD nodes overlap the GHD mesh — check alignment"

    # CFD mesh graph (undirected, weighted by edge length) + super-source linking all seeds
    N = coord_cfd.shape[0]
    e = np.concatenate([faces_cfd[:, [0, 1]], faces_cfd[:, [1, 2]], faces_cfd[:, [2, 0]]], axis=0)
    if extra_edges is not None and len(extra_edges) > 0:
        e = np.concatenate([e, np.asarray(extra_edges, dtype=np.int64)], axis=0)
    w = np.linalg.norm(coord_cfd[e[:, 0]] - coord_cfd[e[:, 1]], axis=1)
    rows = np.concatenate([e[:, 0], e[:, 1], np.full(seed.size, N), seed])
    cols = np.concatenate([e[:, 1], e[:, 0], seed, np.full(seed.size, N)])
    data = np.concatenate([w, w, np.zeros(seed.size), np.zeros(seed.size)])
    adj = csr_matrix((data, (rows, cols)), shape=(N + 1, N + 1))

    geo_dist = dijkstra(adj, indices=N)[:N]  # [N], inf for nodes in disconnected components
    return torch.from_numpy(geo_dist).float()



if __name__ == "__main__":
    mock_dir = '/media/yaplab2/Elements1/AngioFlow/CFDv2'
