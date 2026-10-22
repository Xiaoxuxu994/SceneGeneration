import numpy as np
import torch
import torch.nn.functional as F
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from scipy.spatial.transform import Rotation
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import PoseRelation, Unit
from evo.core import sync
from copy import deepcopy
from scipy.optimize import minimize

import logging
logging.getLogger("evo").setLevel(logging.WARNING)
logging.getLogger("evo.main_rpe").setLevel(logging.WARNING)
logging.getLogger("evo.main_ape").setLevel(logging.WARNING)


# evo changed keyword names across versions. Display-only names (ref_name / est_name) are dropped when the
# installed version does not accept them; any other unsupported keyword raises, so the metric never silently
# changes meaning (e.g. align / correct_scale / delta).
_EVO_DISPLAY_ONLY_KWARGS = {"ref_name", "est_name"}


def _evo_call(fn, *args, **kwargs):
    import inspect
    params = inspect.signature(fn).parameters
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        unsupported = [k for k in kwargs if k not in params]
        bad = [k for k in unsupported if k not in _EVO_DISPLAY_ONLY_KWARGS]
        if bad:
            raise TypeError(f"{fn.__module__}.{fn.__name__} does not accept {bad}; check the installed evo version")
        kwargs = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **kwargs)

def build_pair_index(N, B=1):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]

    return i1, i2


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    #########
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    # tvec_gt, tvec_pred (B, 3,)
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def calculate_auc(r_error, t_error, max_threshold=30, return_list=False):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

    :param r_error: torch.Tensor representing R error values (Degree).
    :param t_error: torch.Tensor representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error tensors along a new axis
    error_matrix = torch.stack((r_error, t_error), dim=1)

    # Compute the maximum error value for each pair
    max_errors, _ = torch.max(error_matrix, dim=1)

    # Calculate histogram of maximum error values
    histogram = torch.histc(
        max_errors, bins=max_threshold + 1, min=0, max=max_threshold
    )

    # Normalize the histogram
    num_pairs = float(max_errors.size(0))
    normalized_histogram = histogram / num_pairs

    if return_list:
        return (
            torch.cumsum(normalized_histogram, dim=0).mean(),
            normalized_histogram,
        )
    # Compute and return the cumulative sum of the normalized histogram
    return torch.cumsum(normalized_histogram, dim=0).mean()


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    # loss_dict = {}

    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses between pairs
    # We use closed_form_inverse to avoid potential numerical loss by torch.inverse()
    # This is possible because of SE3
    # relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(
    #     gt_se3[pair_idx_i2]
    # )
    # relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(
    #     pred_se3[pair_idx_i2]
    # )
    relative_pose_gt = gt_se3[pair_idx_i2].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i1])
    )
    relative_pose_pred = pred_se3[pair_idx_i2].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i1])
    )
    
    # Compute the difference in rotation and translation
    # between the ground truth and predicted relative camera poses
    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )


    return rel_rangle_deg, rel_tangle_deg


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


def to_numpy(x):
    return todevice(x, "numpy")


def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation

    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    c2w = to_numpy(c2w)
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose


def get_tum_poses(poses):
    """
    poses: list of 4x4 arrays
    """
    tt = np.arange(len(poses)).astype(float)
    tum_poses = [c2w_to_tumpose(p) for p in poses]
    tum_poses = np.stack(tum_poses, 0)
    return [tum_poses, tt]


def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple) or isinstance(args, list):
        traj, tstamps = args
        return PoseTrajectory3D(
            positions_xyz=traj[:, :3],
            orientations_quat_wxyz=traj[:, 3:],
            timestamps=tstamps,
        )
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


def eval_metrics(pred_traj, gt_traj=None, seq="", filename="", sample_stride=1, verbose=False):

    if sample_stride > 1:
        pred_traj[0] = pred_traj[0][::sample_stride]
        pred_traj[1] = pred_traj[1][::sample_stride]
        if gt_traj is not None:
            updated_gt_traj = []
            updated_gt_traj.append(gt_traj[0][::sample_stride])
            updated_gt_traj.append(gt_traj[1][::sample_stride])
            gt_traj = updated_gt_traj

    pred_traj = make_traj(pred_traj)

    if gt_traj is not None:
        gt_traj = make_traj(gt_traj)

        if pred_traj.timestamps.shape[0] == gt_traj.timestamps.shape[0]:
            pred_traj.timestamps = gt_traj.timestamps
        else:
            print(pred_traj.timestamps.shape[0], gt_traj.timestamps.shape[0])

        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    # ATE
    traj_ref = gt_traj
    traj_est = pred_traj

    ate_result = _evo_call(
        main_ape.ape,
        traj_ref,
        traj_est,
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=True,
        correct_scale=True,
    )

    ate = ate_result.stats["rmse"]
    # print(ate_result.np_arrays['error_array'])
    # exit()

    # RPE rotation and translation
    delta_list = [1]
    rpe_rots, rpe_transs = [], []
    for delta in delta_list:
        rpe_rots_result = _evo_call(
            main_rpe.rpe,
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

        rot = rpe_rots_result.stats["rmse"]
        rpe_rots.append(rot)

    for delta in delta_list:
        rpe_transs_result = _evo_call(
            main_rpe.rpe,
            traj_ref,
            traj_est,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=True,
            correct_scale=True,
            delta=delta,
            delta_unit=Unit.frames,
            rel_delta_tol=0.01,
            all_pairs=True,
        )

        trans = rpe_transs_result.stats["rmse"]
        rpe_transs.append(trans)

    rpe_trans, rpe_rot = np.mean(rpe_transs), np.mean(rpe_rots)
    if len(filename) > 0:
        with open(filename, "w+") as f:
            f.write(f"Seq: {seq} \n\n")
            f.write(f"{ate_result}")
            f.write(f"{rpe_rots_result}")
            f.write(f"{rpe_transs_result}")

    if verbose:
        print(f"Save results to {filename}")
    return ate, rpe_trans, rpe_rot


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


def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.clone().cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.clone().cpu().numpy().reshape(-1)

    initial_params = [s, t]  # s = 1, t = 0

    result = minimize(
        absolute_error_loss,
        initial_params,
        args=(predicted_depth_np, ground_truth_depth_np),
    )

    s, t = result.x
    return s, t


def absolute_value_scaling2(
    predicted_depth,
    ground_truth_depth,
    s_init=1.0,
    t_init=0.0,
    lr=1e-4,
    max_iters=1000,
    tol=1e-6,
):
    # Initialize s and t as torch tensors with requires_grad=True
    s = torch.tensor(
        [s_init],
        requires_grad=True,
        device=predicted_depth.device,
        dtype=predicted_depth.dtype,
    )
    t = torch.tensor(
        [t_init],
        requires_grad=True,
        device=predicted_depth.device,
        dtype=predicted_depth.dtype,
    )

    optimizer = torch.optim.Adam([s, t], lr=lr)

    prev_loss = None

    for i in range(max_iters):
        optimizer.zero_grad()

        # Compute predicted aligned depth
        predicted_aligned = s * predicted_depth + t

        # Compute absolute error
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)

        # Compute loss
        # loss = torch.sum(abs_error)
        loss = torch.mean(abs_error)
        print(loss)
        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Check convergence
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()


def depth_evaluation(
    predicted_depth_original,
    ground_truth_depth_original,
    max_depth=80,
    custom_mask=None,
    post_clip_min=None,
    post_clip_max=None,
    pre_clip_min=None,
    pre_clip_max=None,
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    metric_scale=False,
    lr=1e-4,
    max_iters=1000,
    device="cpu",
    align_with_scale=False,
    disp_input=False,
    return_full=False
):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.

    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.

    Returns:
        dict: A dictionary containing the evaluation metrics.
        # torch.Tensor: The depth error parity map.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)
        assert custom_mask.shape == ground_truth_depth_original.shape

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)

        # if custom_mask is not None:
        #     custom_mask = custom_mask.view(-1, w)
        #     custom_mask = custom_mask.to(device).float()
        if custom_mask is not None:
            if custom_mask.dtype != torch.bool:
                custom_mask = custom_mask != 0
            
            # flatten the masks to 2d
            custom_mask = custom_mask.view(-1, w)

    # put to device
    predicted_depth_original = predicted_depth_original.to(device).float()
    ground_truth_depth_original = ground_truth_depth_original.to(device).float()
    
    if custom_mask is not None:
        custom_mask = custom_mask.to(device)

    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (
            ground_truth_depth_original < max_depth
        )
    else:
        mask = ground_truth_depth_original > 0

    ## add custom mask
    if custom_mask is not None:
        # assert custom_mask.shape == ground_truth_depth_original.shape
        mask = mask & custom_mask

    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input:  # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    if metric_scale:
        predicted_depth = predicted_depth
    elif align_with_lstsq:
        # Convert to numpy for lstsq
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)

        # Add a column of ones for the shift term
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])

        # Solve for scale (s) and shift (t) using least squares
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]

        # convert to torch tensor
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)

        # Apply scale and shift
        # predicted_depth = s * predicted_depth + t
        predicted_depth *= s
        predicted_depth += t
    elif align_with_lad:
        s, t = absolute_value_scaling(
            predicted_depth,
            ground_truth_depth,
            s=torch.median(ground_truth_depth) / torch.median(predicted_depth),
        )
        # predicted_depth = s * predicted_depth + t
        predicted_depth *= s
        predicted_depth += t
    elif align_with_lad2:
        s_init = (
            torch.median(ground_truth_depth) / torch.median(predicted_depth)
        ).item()
        s, t = absolute_value_scaling2(
            predicted_depth,
            ground_truth_depth,
            s_init=s_init,
            lr=lr,
            max_iters=max_iters,
        )
        # predicted_depth = s * predicted_depth + t
        predicted_depth *= s
        predicted_depth += t
    elif align_with_scale:
        # Compute initial scale factor 's' using the closed-form solution (L2 norm)
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method
        for _ in range(10):
            # Compute residuals between scaled predictions and ground truth
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = (
                residuals.abs() + 1e-8
            )  # Add small constant to avoid division by zero

            # Compute weights inversely proportional to the residuals
            weights = 1.0 / abs_residuals

            # Update 's' using weighted sums
            weighted_dot_pred_gt = torch.sum(
                weights * predicted_depth * ground_truth_depth
            )
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth**2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        # Optionally clip 's' to prevent extreme scaling
        s = s.clamp(min=1e-3)

        # Detach 's' if you want to stop gradients from flowing through it
        s = s.detach()

        # Apply the scale factor to the predicted depth
        # predicted_depth = s * predicted_depth
        predicted_depth *= s

    else:
        # Align the predicted depth with the ground truth using median scaling
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    # if custom_mask is not None:
    #     assert custom_mask.shape == ground_truth_depth_original.shape
    #     mask_within_mask = custom_mask.cpu()[mask]
    #     predicted_depth = predicted_depth[mask_within_mask]
    #     ground_truth_depth = ground_truth_depth[mask_within_mask]
    
    # Calculate the metrics
    abs_rel = torch.mean(
        torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth
    ).item()
    sq_rel = torch.mean(
        ((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth
    ).item()
    
    # print(predicted_depth.min().item(), predicted_depth.max().item(), ground_truth_depth.min().item(), ground_truth_depth.max().item(), abs_rel)

    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()

    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(
        torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)
    ).item()

    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(
        predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth
    )
    threshold_0 = torch.mean((max_ratio < 1.0).float()).item()
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25**2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25**3).float()).item()

    # num_valid_pixels = (
    #     torch.sum(mask).item()
    #     if custom_mask is None
    #     else torch.sum(mask_within_mask).item()
    # )
    num_valid_pixels = torch.sum(mask).item()

    if num_valid_pixels == 0:
        (
            abs_rel,
            sq_rel,
            rmse,
            log_rmse,
            threshold_0,
            threshold_1,
            threshold_2,
            threshold_3,
        ) = (0, 0, 0, 0, 0, 0, 0, 0)

    results = {
        "Abs Rel": abs_rel,
        "Sq Rel": sq_rel,
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "δ < 1.": threshold_0,
        "δ < 1.25": threshold_1,
        "δ < 1.25^2": threshold_2,
        "δ < 1.25^3": threshold_3,
        "valid_pixels": num_valid_pixels,
    }

    if not return_full:
        return results
    
    # Compute the depth error parity map
    if metric_scale:
        predicted_depth_original = predicted_depth_original
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    elif align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input:
            predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = (
            torch.abs(predicted_depth_original - ground_truth_depth_original)
            / ground_truth_depth_original
        )

    # Reshape the depth_error_parity_map back to the original image size
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(
        mask, depth_error_parity_map, depth_error_parity_map_full
    )

    predict_depth_map_full = predicted_depth_original
    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(
        mask, ground_truth_depth_original, gt_depth_map_full
    )

    return (
        results,
        depth_error_parity_map_full,
        predict_depth_map_full,
        gt_depth_map_full,
    )


def eval_depth_map(
    pred_depths, gt_depths, valid_masks, 
    align_with_lstsq=False,
    align_with_lad=False,
    align_with_lad2=False,
    align_with_scale=False, 
    device="npu"):
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
    # depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
    depth_results = depth_evaluation(
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