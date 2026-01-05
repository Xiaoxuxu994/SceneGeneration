import numpy as np
import open3d as o3d

from lib.metric import se3_to_relative_pose_error, calculate_auc, umeyama, accuracy, completion
from lib.evo_utils import get_tum_poses, eval_metrics
from lib.depth import depth_evaluation


def eval_angle(pred_extrinsics, gt_extrinsics, threshold=30):
    """evaluate relative angles in SE(3) predictions.

    Args:
        pred_extrinsics (Torch.tensor): OpenCV-style world-from-camera batched 4x4 matrices, (S, 4, 4)
        gt_extrinsics (Torch.tensor): OpenCV-style world-from-camera batched 4x4 matrices, (S, 4, 4)
        
        all extrinsics must be aligned to the first frame and averaged by the average metric-scale depth
        
    """
    
    assert pred_extrinsics.ndim == 3, "wrong input dim. expected 3, got {}".format(pred_extrinsics.dim)
    assert pred_extrinsics.shape[1] == 4 and pred_extrinsics.shape[2] == 4, \
        "pred_extrinsics must be homogeneous matrices in shape (S, 4, 4)"

    assert gt_extrinsics.ndim == 3, "wrong input dim. expected 3, got {}".format(gt_extrinsics.dim)
    assert gt_extrinsics.shape[1] == 4 and gt_extrinsics.shape[2] == 4, \
        "gt_extrinsics must be homogeneous matrices in shape (S, 4, 4)"
    
    S, _, _ = pred_extrinsics.shape
    
    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_extrinsics, gt_extrinsics, S)
    rra_acc = (rel_rangle_deg <= threshold).sum() / rel_rangle_deg.shape[0]
    rta_acc = (rel_tangle_deg <= threshold).sum() / rel_tangle_deg.shape[0]
    auc_at_threshold = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=threshold).item()
    
    return rra_acc.item(), rta_acc.item(), auc_at_threshold

def eval_trajectory(pred_extrinsics, gt_extrinsics):
    """evaluate pose trajectories.

    Args:
        pred_extrinsics (torch.tensor): OpenCV-style world-from-camera batched 4x4 matrices, (S, 4, 4)
        gt_extrinsics (torch.tensor): OpenCV-style world-from-camera batched 4x4 matrices, (S, 4, 4)
        
    """
    pred_traj = get_tum_poses(pred_extrinsics)
    gt_traj = get_tum_poses(gt_extrinsics)
    
    ate, rpe_trans, rpe_rot = eval_metrics(pred_traj, gt_traj)
    
    return ate, rpe_trans, rpe_rot

def eval_depth_map(pred_depths, gt_depths, valid_masks, 
        align_with_lstsq=False,
        align_with_lad=False,
        align_with_lad2=False,
        align_with_scale=False, 
        device="npu:0"):
    """evaluate depth maps

    Args:
        pred_depths (torch.tensor or numpy.array): depth maps in shape (S, H, W)
        gt_depths (torch.tensor or numpy.array): depth maps in shape (S, H, W)
        valid_masks (torch.tensor or numpy.array): depth masks in shape (S, H, W)
        
        align_with_lstsq (bool, optional): using least square alignment or not. Defaults to False.
        align_with_lad (bool, optional): using scale alignment or not. Defaults to False.
        align_with_lad2 (bool, optional): using scale and shift alignment or not. Defaults to False.
        align_with_scale (bool, optional): using scale alignment or not. Defaults to False.
        -> if none of the above is set to True, use median alignment
        
        device (str, optional): device to run alignment. Defaults to "npu:0".

    Returns:
        dict: depth evaluation results
    """
    depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
        pred_depths, 
        gt_depths, 
        custom_mask=valid_masks,
        align_with_lstsq=align_with_lstsq,
        align_with_lad=align_with_lad,
        align_with_lad2=align_with_lad2,
        align_with_scale=align_with_scale,
        device=device
    )
    
    return depth_results

def eval_point_map(pred_pts, gt_pts, valid_masks, threshold=0.1):
    """evaluate predicted point maps in world coordinates.

    Args:
        pred_pts (numpy.array): pixel-aligned point maps, (S, H, W, 3)
        gt_pts (numpy.array): pixel-aligned point maps, (S, H, W, 3)
        valid_masks (numpy.array): pixel-aligned masks, (S, H, W)
        
        all point maps must be aligned to the first frame and averaged by the average metric-scale depth
        
    """
    
    assert pred_pts.shape == gt_pts.shape
    assert pred_pts.shape[:-1] == valid_masks.shape
    
    S, H, W, _ = pred_pts.shape
    
    pred_pts_flatten = pred_pts.reshape((-1, 3))
    gt_pts_flatten = gt_pts.reshape((-1, 3))
    valid_masks_flatten = valid_masks.reshape((-1,))
    
    # rough point cloud alignment
    c, R, t = umeyama(pred_pts_flatten[valid_masks_flatten].T, gt_pts_flatten[valid_masks_flatten].T)
    
    # align predicted point clouds to the GTs
    pred_pts = pred_pts.reshape((S, -1, 3))
    R = np.repeat(R[None, :], S, axis=0)
    t = np.repeat(t[None, :], S, axis=0)

    pred_pts = pred_pts @ R
    pred_pts = pred_pts + np.transpose(t, (0, 2, 1))
    pred_pts = pred_pts.reshape((S, H, W, 3))
    
    # filter invalid points
    pred_pts = pred_pts[valid_masks].reshape(-1, 3)
    gt_pts = gt_pts[valid_masks].reshape(-1, 3)

    # convert to Open3D point clouds
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_pts)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
    
    # ICP align refinement
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd,
        pcd_gt,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    transformation = reg_p2p.transformation
    pcd = pcd.transform(transformation)
    
    # 9. estimate normals
    pcd.estimate_normals()
    pcd_gt.estimate_normals()
    pred_normal = np.asarray(pcd.normals)
    gt_normal = np.asarray(pcd_gt.normals)
    
    # compute metrics
    acc, acc_med, nc1, nc1_med = accuracy(
        pcd_gt.points, pcd.points, gt_normal, pred_normal
    )
    comp, comp_med, nc2, nc2_med = completion(
        pcd_gt.points, pcd.points, gt_normal, pred_normal
    )
    
    return acc, acc_med, nc1, nc1_med, comp, comp_med, nc2, nc2_med
    