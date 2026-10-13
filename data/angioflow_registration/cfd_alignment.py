"""Optional CFD-mesh <-> world-mesh rigid registration (ICP), used only for
ImperialNHS refinement when a geomagic_processed.obj reference mesh is
available. Copied from AneuG-Net's new_version/export_batch_pt.py.
"""

import numpy as np
import torch


def get_cfd_to_world_transform(coord_cfd, coord_world, voxel_size=None, return_matrix=False):
    """
    Estimate the rigid transform (rotation + translation, no scaling) that maps
    points in `coord_cfd` to the frame of `coord_world`. The two clouds are assumed
    to be sampled on the same surface but at different (overlapping) locations.

    Pipeline: voxel downsample -> FPFH features -> RANSAC global registration ->
    point-to-plane ICP refinement on the full clouds.

    Parameters
    ----------
    coord_cfd : torch.Tensor or np.ndarray, shape [N, 3] or [T, N, 3]
    coord_world : torch.Tensor or np.ndarray, shape [M, 3] or [T, M, 3]
    voxel_size : float, optional. Defaults to ~1/50 of the target bbox diagonal.
    return_matrix : if True, return the 4x4 transform; else return (R, t).

    Returns
    -------
    (R, t) with R: (3, 3) and t: (3,) torch.float tensors, satisfying
    p_world ~= p_cfd @ R.T + t. If return_matrix=True, returns a (4, 4) tensor.
    """
    import open3d as o3d

    def _to_np(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3:
            x = x[0]
        return np.ascontiguousarray(x, dtype=np.float64)

    src_np = _to_np(coord_cfd)
    tgt_np = _to_np(coord_world)

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
