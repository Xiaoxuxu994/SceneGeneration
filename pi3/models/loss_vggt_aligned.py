import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge
from ..utils.alignment import align_points_scale

from datasets import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__
from einops import rearrange
# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

def huber_loss(pred, target, delta=1.0, reduction='mean'):
    diff = pred - target
    abs_diff = diff.abs()

    loss = torch.where(
        abs_diff < delta,
        0.5 * diff.pow(2),
        delta * (abs_diff - 0.5 * delta),
    )

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(
        self,
        local_align_res=4096,
        train_conf=False,
        expected_dist_thresh=0.02,
        w_unproj_global=0.0,
        w_normalized_loss=0.0,
        normalize_use_local=False,
        normalize_penalize_zero=False,
        use_scale_align=False,
    ):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')

        self.train_conf = train_conf
        if self.train_conf:
            self.prepare_segformer()
            self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()
            self.expected_dist_thresh = expected_dist_thresh
        self.w_unproj_global = w_unproj_global
        self.w_normalized_loss = w_normalized_loss
        self.normalize_use_local = normalize_use_local
        self.normalize_penalize_zero=normalize_penalize_zero
        self.use_scale_align = use_scale_align

    def prepare_segformer(self):
        from pi3.models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda()

    def predict_sky_mask(self, imgs):
        with torch.no_grad():
            output = self.segformer.inference_(imgs)
            output = output == 2
        return output

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def mat_mul_vec_by_conv(self, vec: torch.Tensor, mat: torch.Tensor):
        """Implement multiplicatoin between matrix and vector by convolution.

        Args:
            vec (torch.Tensor): Tensor of shape (B, C, H, W) or (B, H, W, C) or (B, V, C, H, W) or (B, V, H, W, C)
            mat (torch.Tensor): Tensor of shape (B, C, C) or (B, V, C, C)
        
        Returns:
            torch.Tensor: A tensor with the same shape of vec.
        """
        assert vec.ndim == 5 and vec.shape[:2] == mat.shape[:2] or vec.ndim == 4 and vec.shape[:1] == mat.shape[:1], \
            "mat shape: {}, vec shape: {}".format(mat.shape, vec.shape)
        assert mat.shape[-2] == mat.shape[-1] and (mat.shape[-1] == vec.shape[-1] or mat.shape[-1] == vec.shape[-3]), \
            "mat shape: {}, vec shape: {}".format(mat.shape, vec.shape)
        vec_shape = vec.shape
        mat_shape = mat.shape
        vec = vec.reshape(-1, *vec_shape[-3:])
        mat = mat.reshape(-1, *mat_shape[-2:])
        shuffle = False
        if vec.shape[-1] == mat.shape[-1]:
            shuffle = True
            vec = rearrange(vec, "b h w c -> b c h w")
        b, d, h, w = vec.shape
        ret = torch.nn.functional.conv2d(vec.reshape(1, b * d, h, w), mat.reshape(b * d, d, 1, 1), groups=b)
        ret = ret.reshape(b, d, h, w)
        if shuffle:
            ret = rearrange(ret, "b c h w -> b h w c")
        ret = ret.reshape(*vec_shape)
        return ret

    def calc_extr_inv(self, extr: torch.Tensor):
        assert extr.shape[-2] == 4 and extr.shape[-1] == 4

        rot = extr[..., :3, :3]
        trans = extr[..., :3, 3:]

        ret = torch.zeros_like(extr)
        ret[..., :3, :3] = rot.transpose(-2, -1)
        ret[..., :3, 3:] = torch.matmul(rot.transpose(-2, -1), -trans)
        ret[..., 3, 3] = ret[..., 3, 3] + 1.0

        return ret

    def pad_one(self, x: torch.Tensor, dim: int):
        shape = list(x.shape)
        shape[dim] = 1
        return torch.cat([x, torch.ones(*shape, dtype=x.dtype, device=x.device)], dim=dim)


    def localpoint2global(self, points: torch.Tensor, extrs: torch.Tensor):
        """Align local point map under the global coordinate.

        Args:
            points (torch.Tensor): Local point map, tensor of shape (..., h, w, 3).
            extrs (torch.Tensor): World-to-camera transform, tensor of shape (..., 4, 4)
        """
        points_world_homo = self.mat_mul_vec_by_conv(self.pad_one(points, dim=-1), self.calc_extr_inv(extrs))
        points_world = points_world_homo[..., :3] / points_world_homo[..., -1:]
        return points_world

    def normalize_loss(self, pred, valid_masks=None):
        pred_local_pts = pred.get("local_points_raw", pred["local_points"])

        if pred_local_pts.ndim == 5 and pred_local_pts.shape[2] == 3:
            pred_local_pts = rearrange(pred_local_pts, "b v c h w -> b v h w c")

        B = pred_local_pts.shape[0]

        if self.normalize_use_local:
            point_norm = pred_local_pts.norm(dim=-1)
        else:
            camera_poses = pred.get("camera_poses_raw", pred["camera_poses"])
            pred_w2c = se3_inverse(camera_poses)
            pred_global_pts = self.localpoint2global(pred_local_pts, pred_w2c)
            point_norm = pred_global_pts.norm(dim=-1)

        if valid_masks is not None:
            valid = valid_masks.float()
            if valid.ndim == 5 and valid.shape[2] == 1:
                valid = valid[:, :, 0]

            point_scale = (point_norm * valid).reshape(B, -1).sum(dim=-1) / (
                valid.reshape(B, -1).sum(dim=-1) + 1e-8
            )
        else:
            point_scale = point_norm.reshape(B, -1).mean(dim=-1)

        loss = (point_scale - 1.0).abs().mean()

        if self.normalize_penalize_zero:
            loss = loss + (point_scale + 1.0 / (point_scale + 1e-8)).mean()

        return loss

    def noraml_loss(self, points, gt_points, mask):
        not_edge = ~depth_edge(gt_points[..., 2], rtol=0.03)
        mask = torch.logical_and(mask, not_edge)

        leftup, rightup, leftdown, rightdown = points[..., :-1, :-1, :], points[..., :-1, 1:, :], points[..., 1:, :-1, :], points[..., 1:, 1:, :]
        upxleft = torch.cross(rightup - rightdown, leftdown - rightdown, dim=-1)
        leftxdown = torch.cross(leftup - rightup, rightdown - rightup, dim=-1)
        downxright = torch.cross(leftdown - leftup, rightup - leftup, dim=-1)
        rightxup = torch.cross(rightdown - leftdown, leftup - leftdown, dim=-1)

        gt_leftup, gt_rightup, gt_leftdown, gt_rightdown = gt_points[..., :-1, :-1, :], gt_points[..., :-1, 1:, :], gt_points[..., 1:, :-1, :], gt_points[..., 1:, 1:, :]
        gt_upxleft = torch.cross(gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1)
        gt_leftxdown = torch.cross(gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1)
        gt_downxright = torch.cross(gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1)
        gt_rightxup = torch.cross(gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1)

        mask_leftup, mask_rightup, mask_leftdown, mask_rightdown = mask[..., :-1, :-1], mask[..., :-1, 1:], mask[..., 1:, :-1], mask[..., 1:, 1:]
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown

        MIN_ANGLE, MAX_ANGLE, BETA_RAD = math.radians(1), math.radians(90), math.radians(3)

        loss = mask_upxleft * _smooth(angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_leftxdown * _smooth(angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_downxright * _smooth(angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD) \
                + mask_rightxup * _smooth(angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE), beta=BETA_RAD)

        loss = loss.mean() / (4 * max(points.shape[-3:-1]))

        return loss

    def forward(self, pred, gt):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        if self.use_scale_align:
            with torch.no_grad():
                xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
                xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
                xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

                S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
                S_opt_local[S_opt_local <= 0] *= -1
        else:
            S_opt_local = torch.ones(B, device=pred_local_pts.device, dtype=pred_local_pts.dtype)

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

        # conf loss
        if self.train_conf:
            pred_conf = pred['conf']

            # probability loss
            valid = local_pts_loss.detach().mean(-1, keepdims=True) < self.expected_dist_thresh
            local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks], valid.float())

            sky_mask = self.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
            sky_mask[valid_masks] = False
            if sky_mask.sum() == 0:
                sky_mask_loss = 0.0 * aligned_local_pts.mean()
            else:
                sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
            
            final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
            details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()

        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
            final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # [Optional] Global Point Loss
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_pts = gt['global_points']

            pred_global_pts = pred['global_points']
            if self.use_scale_align:
                pred_global_pts = pred_global_pts * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        # [Optional] Unprojected Global Point Loss
        if self.w_unproj_global != 0.0:
            ## Global Point Loss, compare the points globally in the coordinate of the first camera. 
            gt_pts = gt['global_points']

            if 'points' in pred and pred['points'] is not None:
                pred_global_pts = pred['points']
            else:
                pred_w2c = se3_inverse(pred['camera_poses'])
                pred_global_pts = self.localpoint2global(pred_local_pts, pred_w2c)

                pred_trans_to_firstF = torch.cat([pred_w2c[:, :1] for _ in range(N)], dim=1) # b v 4 4
                pred_global_pts = self.mat_mul_vec_by_conv(self.pad_one(pred_global_pts, dim=-1), pred_trans_to_firstF)[..., :3] # b v h w 3
            if self.use_scale_align:
                pred_global_pts = pred_global_pts * S_opt_local.view(B, 1, 1, 1, 1)
            global_pts_loss = self.criteria_local(pred_global_pts[valid_masks].float(), 
                                                gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]

            final_loss += self.w_unproj_global * global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()

        # [Optional] Normalize Loss
        if self.w_normalized_loss != 0.0:
            normalized_loss = self.normalize_loss(pred, valid_masks)
            final_loss += self.w_normalized_loss * normalized_loss
            details['normalized_loss'] = normalized_loss

        return final_loss, details, S_opt_local


# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
        if scale is not None:
            pred_pose_align[..., :3, 3] *=  scale.view(B, 1, 1)
        
        pred_w2c = se3_inverse(pred_pose_align)
        gt_w2c = se3_inverse(gt_pose)
        
        pred_w2c_exp = pred_w2c.unsqueeze(2)
        pred_pose_exp = pred_pose_align.unsqueeze(1)
        
        gt_w2c_exp = gt_w2c.unsqueeze(2)
        gt_pose_exp = gt_pose.unsqueeze(1)
        
        pred_rel_all = torch.matmul(pred_w2c_exp, pred_pose_exp)
        gt_rel_all = torch.matmul(gt_w2c_exp, gt_pose_exp)

        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)

        t_pred = pred_rel_all[..., :3, 3][:, mask, ...]
        R_pred = pred_rel_all[..., :3, :3][:, mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][:, mask, ...]
        R_gt = gt_rel_all[..., :3, :3][:, mask, ...]

        # trans_loss = F.huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        trans_loss = huber_loss(t_pred, t_gt, reduction='mean', delta=0.1)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        )
        
        total_loss = self.alpha * trans_loss + rot_loss

        return total_loss, dict(trans_loss=trans_loss, rot_loss=rot_loss)

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

class Pi3Loss(nn.Module):
    def __init__(
        self,
        train_conf=False,
        w_unproj_global=0.0,
        w_normalized_loss=0.0,
        normalize_use_local=False,
        normalize_penalize_zero=False,
        w_depth_shape_loss=0.0,
        use_scale_align=False,

    ):
        super().__init__()

        self.point_loss = PointLoss(
            train_conf=train_conf,
            w_unproj_global=w_unproj_global,
            w_normalized_loss=w_normalized_loss,
            normalize_use_local=normalize_use_local,
            normalize_penalize_zero=normalize_penalize_zero,
            use_scale_align=use_scale_align
        )

        self.camera_loss = CameraLoss()

    def prepare_gt(self, gt, device):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1).to(device, non_blocking=True)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1).to(device, non_blocking=True).bool()
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1).to(device, non_blocking=True)
        imgs = torch.stack([view['img'] for view in gt], dim=1).to(device, non_blocking=True)

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        dataset_names = gt[0]['dataset']

        return dict(
            imgs = imgs,
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            dataset_names=dataset_names
        )
    
    def normalize_pred(self, pred, gt):

        pred = pred.copy()

        pred["local_points_raw"] = pred["local_points"]
        pred["camera_poses_raw"] = pred["camera_poses"]

        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        if 'global_points' in pred and pred['global_points'] is not None:
            # pred['global_points'] /= norm_factor[..., None, None, None, None]
            pred['global_points'] = pred['global_points'] / norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = local_points
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def canonicalize_pred(self, pred):

        pred = pred.copy()

        pred["local_points_raw"] = pred["local_points"]
        pred["camera_poses_raw"] = pred["camera_poses"]

        local_points = pred['local_points']
        camera_poses = pred['camera_poses']

        w2c_target = se3_inverse(camera_poses[:, 0])
        camera_poses = torch.einsum('bij, bnjk -> bnik', w2c_target, camera_poses)
        points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        pred['camera_poses'] = camera_poses
        pred['points'] = points

        return pred

    def forward(self, pred, gt_raw):
        device = pred['local_points'].device
        gt = self.prepare_gt(gt_raw,device)
        pred = self.canonicalize_pred(pred)

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss * 0.1
        details.update(camera_loss_details)

        return final_loss, details