import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
from collections import defaultdict
import math

from models.moge.utils3d.torch.utils import depth_edge, normal_edge
from utils.geometry import homogenize_points, se3_inverse, get_gt_warp, get_pixel, align_sequences_opencv_c2w
from utils.alignment import align_points_scale
from dataset.base.transforms import *
from utils.vis_utils import tensor_to_pil, visualize_depth, write_ply, visualize_normals
from models.dinov2.models.teacher_mv import compute_normals_robust

from dataset import __HIGH_QUALITY_DATASETS__, __MIDDLE_QUALITY_DATASETS__, __METRIC_DATASETS__, __TEACH_DATASETS__

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

# ---------------------------------------------------------------------------
# PointLoss: Scale-invariant Local Pointmap
# ---------------------------------------------------------------------------

class PointLoss(nn.Module):
    def __init__(self, local_align_res=4096):
        super().__init__()
        self.local_align_res = local_align_res
        self.criteria_local = nn.L1Loss(reduction='none')
        self.conf_loss_fn = torch.nn.BCEWithLogitsLoss()

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

    def forward(self, pred, gt, father_cls=None):
        pred_local_pts = pred['local_points']
        B, N, H, W, _ = pred_local_pts.shape

        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        weights = gt_local_pts[..., 2]
        weights = weights.clamp_min(0.1 * weighted_mean(weights, valid_masks, dim=(-2, -1), keepdim=True))
        weights = 1 / (weights + 1e-6)

        gt['weights'] = weights
        
        details = dict()
        final_loss = 0.0

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        details = {}
        final_loss = 0.0

        aligned_pred_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts
        pred['aligned_local_points'] = aligned_pred_local_pts

        # local point loss
        local_pts_loss = self.criteria_local(aligned_pred_local_pts[valid_masks].float(), gt_local_pts[valid_masks].float()) * weights[valid_masks].float()[..., None]
        details['local_pts_loss'] = local_pts_loss.mean()

        pix = torch.from_numpy(get_pixel(H, W).T.reshape(H, W, 3)).to(gt_local_pts.device).float()[:, :].repeat(B, N, 1, 1, 1)
        gt_rays = torch.einsum('bnij, bnhwj -> bnhwi', torch.inverse(gt['camera_intrinsics']), pix)[..., :2]
        rays_loss = F.l1_loss(pred['xy'], gt_rays)
        final_loss += rays_loss
        details['rays_loss'] = rays_loss

        depth_loss = local_pts_loss[..., 2].mean()
        details['depth_loss'] = depth_loss
        final_loss += depth_loss

        # Sparse depth loss
        use_depth_mask = pred['use_depth_mask'] if pred['use_depth_mask'] is not None else torch.zeros((B, N), device=gt_local_pts.device, dtype=torch.bool)
        sparse_depth_masks = torch.logical_and(valid_masks, gt['sparse_depth_masks'])
        sparse_depth_masks[~use_depth_mask] = 0
        if sparse_depth_masks.sum() > 0:
            sparse_depth_loss = (self.criteria_local(aligned_pred_local_pts[..., 2][sparse_depth_masks].float(), gt_local_pts[..., 2][sparse_depth_masks].float()) * weights[sparse_depth_masks].float()).mean()
        else:
            sparse_depth_loss = 0.0 * aligned_pred_local_pts.mean()
        details['sparse_depth_loss'] = sparse_depth_loss
        final_loss += sparse_depth_loss        

        # normal loss
        normal_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] in __HIGH_QUALITY_DATASETS__ + __MIDDLE_QUALITY_DATASETS__]
        if len(normal_batch_id) == 0:
            normal_loss =  0.0 * aligned_pred_local_pts.mean()
        else:
            normal_loss = self.noraml_loss(aligned_pred_local_pts[normal_batch_id], gt_local_pts[normal_batch_id], valid_masks[normal_batch_id])
        final_loss += normal_loss.mean()
        details['normal_loss'] = normal_loss.mean()

        # Metric loss
        metric_batch = [i for i, x in enumerate(gt['dataset_names']) if x in __METRIC_DATASETS__]
        if len(metric_batch) > 0:
            gt_metric = S_opt_local[metric_batch] * gt['norm_factor'][metric_batch] / pred['norm_factor'][metric_batch]
            metric_loss = (torch.log1p(pred['metric'][metric_batch]) - torch.log1p(gt_metric.view(len(metric_batch), 1)))**2
            metric_loss = metric_loss.mean()
        else:
            metric_loss = 0.0 * pred['metric'].mean()
        final_loss += metric_loss.mean()
        details['metric_loss'] = metric_loss.mean()

        # probability loss
        pred_conf = pred['conf']
        scale = 50.0  # 对应 beta = 0.02
        pts_error = local_pts_loss.detach().mean(-1, keepdim=True)
        target_conf = torch.exp(-scale * pts_error)
        local_conf_loss = self.conf_loss_fn(pred_conf[valid_masks].float(), target_conf.float())

        sky_mask = father_cls.predict_sky_mask(gt['imgs'].reshape(B*N, 3, H, W)).reshape(B, N, H, W)
        sky_mask[valid_masks] = False
        if sky_mask.sum() == 0:
            sky_mask_loss = 0.0 * pred_conf.mean()
        else:
            sky_mask_loss = self.conf_loss_fn(pred_conf[sky_mask], torch.zeros_like(pred_conf[sky_mask]))
        
        final_loss += 0.05 * (local_conf_loss + sky_mask_loss)
        details['local_conf_loss'] = (local_conf_loss + sky_mask_loss)

        return final_loss, details, S_opt_local

# ---------------------------------------------------------------------------
# CameraLoss: Affine-invariant Camera Pose
# ---------------------------------------------------------------------------

class CameraLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_weight = 10
        self.rot_weight = 0.1

        self.criteria_points = nn.L1Loss(reduction='none')

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
        # return R_err.mean()         # [0, 3.14]
        return R_err         # [0, 3.14]
    
    def forward(self, pred, gt, scale):
        B = gt['camera_poses'].shape[0]

        pred_pose = pred['camera_poses']
        gt_pose = gt['camera_poses']

        aligned_pred_local_points = pred['aligned_local_points']
        gt_local_points = gt['local_points']
        valid_masks = gt['valid_masks']
        weights = gt['weights']

        B, N, _, _ = pred_pose.shape

        pred_pose_align = pred_pose.clone()
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
        mask = mask[None].repeat(B, 1, 1)

        t_pred = pred_rel_all[..., :3, 3][mask, ...]
        R_pred = pred_rel_all[..., :3, :3][mask, ...]
        
        t_gt = gt_rel_all[..., :3, 3][mask, ...]
        R_gt = gt_rel_all[..., :3, :3][mask, ...]

        trans_loss = F.huber_loss(t_pred, t_gt, reduction='none', delta=0.1).reshape(B, N, (N-1), 3)
        
        rot_loss = self.rot_ang_loss(
            R_pred.reshape(-1, 3, 3), 
            R_gt.reshape(-1, 3, 3)
        ).reshape(B, N, (N-1))

        use_pose_mask = pred['use_pose_mask'] if pred['use_pose_mask'] is not None else torch.zeros((B, N), device=pred_pose.device, dtype=torch.bool)
        use_pose_mask = use_pose_mask.unsqueeze(2) & use_pose_mask.unsqueeze(1)
        use_pose_mask = use_pose_mask[mask].reshape(B, N, N-1)
        
        details = {}
        final_loss = 0.0
        if use_pose_mask.sum() > 0:
            details['trans_loss_prior'] = trans_loss[use_pose_mask].mean()
            details['rot_loss_prior'] = rot_loss[use_pose_mask].mean()

            final_loss += self.trans_weight * trans_loss[use_pose_mask].mean() + self.rot_weight * rot_loss[use_pose_mask].mean()
        else:
            details['trans_loss_prior'] = 0.0 * trans_loss.mean()
            details['rot_loss_prior'] = 0.0 * trans_loss.mean()

        if (~use_pose_mask).sum() > 0:
            # only supervise edge
            non_edge_mask = gt['overlap_matrix'] < 0.01
            edge_weight = torch.ones_like(mask, device=gt_pose.device, dtype=torch.float32)
            edge_weight[non_edge_mask] = 0.1
            edge_weight = edge_weight[mask].reshape(B, N, N-1)
            reweight_trans_loss = trans_loss*edge_weight[..., None]
            reweight_rot_loss = rot_loss*edge_weight

            details['trans_loss'] = trans_loss[~use_pose_mask].mean()
            details['rot_loss'] = rot_loss[~use_pose_mask].mean()

            details['edge_trans_loss'] = reweight_trans_loss[~use_pose_mask].mean()
            details['edge_rot_loss'] = reweight_rot_loss[~use_pose_mask].mean()
            
            final_loss += self.trans_weight * trans_loss[~use_pose_mask].mean() + self.rot_weight * rot_loss[~use_pose_mask].mean()
            # final_loss += self.trans_weight * reweight_trans_loss[~use_pose_mask].mean() + self.rot_weight * reweight_rot_loss[~use_pose_mask].mean()
        else:
            details['trans_loss'] = 0.0 * trans_loss.mean()
            details['rot_loss'] = 0.0 * trans_loss.mean()

        # # -------------------------  Unproject loss  -------------------------
        # # aligned_local_pts, gt_pts_cam: B x N x H x W x 3
        # pred_rel_pts = torch.einsum('bnmij, bmhwj -> bnmhwi', pred_rel_all, homogenize_points(aligned_pred_local_points))[..., :3]
        # gt_rel_pts = torch.einsum('bnmij, bmhwj -> bnmhwi', gt_rel_all, homogenize_points(gt_local_points))[..., :3]
        # B, N, _, H, W, _ = gt_rel_pts.shape

        # identity_mask = ~torch.eye(N, dtype=torch.bool, device=gt_rel_pts.device) # Shape: N x N, 对角线为 False
        # identity_mask = identity_mask.view(1, N, N, 1, 1).expand(B, N, N, H, W)

        # user_mask_expanded = valid_masks.unsqueeze(1).expand(B, N, N, H, W)
        # weights_expanded = weights.unsqueeze(1).expand(B, N, N, H, W)
        # final_mask = identity_mask & user_mask_expanded & gt['overlap_mask']

        # unproject_loss_all_points = self.criteria_points(pred_rel_pts.float(), gt_rel_pts.float()).mean(dim=-1)
        # weighted_point_loss = unproject_loss_all_points * weights_expanded.float()
        # weighted_point_loss[~final_mask] = 0.0
        # sum_point_loss_ij = weighted_point_loss.sum(dim=(-1, -2))
        # num_valid_points_ij = final_mask.sum(dim=(-1, -2)).float()
        # avg_unproject_loss_ij = sum_point_loss_ij / (num_valid_points_ij + 1e-6) # Shape (B, N, N)

        # final_loss += avg_unproject_loss_ij.mean() * 0.5
        # details['unproject_loss'] = avg_unproject_loss_ij.mean()

        return final_loss, details
    
    
# ---------------------------------------------------------------------------
# Flow Loss
# ---------------------------------------------------------------------------
class FlowLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def get_gt_flow(self, order, depths, poses, K, masks, target_H, target_W):
        B, N, H, W = depths.shape

        ref_depths = depths.reshape(B*N, H, W)
        ref_poses = poses.reshape(B*N, 4, 4)
        ref_K = K.reshape(B*N, 3, 3)
        arange_b = torch.arange(B)[:, None].to(order.device)
        src_depths = depths[arange_b, order].reshape(B*N, H, W)
        src_poses = poses[arange_b, order].reshape(B*N, 4, 4)
        src_K = K[arange_b, order].reshape(B*N, 3, 3)

        poses_ = torch.einsum('bij, bjk -> bik', torch.inverse(src_poses), ref_poses)

        x2, prob = get_gt_warp(ref_depths, src_depths, poses_, ref_K, src_K, relative_depth_error_threshold=0.01, H=target_H, W=target_W)

        ## Visualize
# b, i = 1, 3
# idx = order[b, i]
# from utils.basic import tensor_to_pil
# colors = torch.stack([inverse_CustomNorm(view['img']) for view in gt], dim=1)
# colors = resize_tensor(colors.permute(0, 1, 3, 4, 2), (H, W)).permute(0, 1, 4, 2, 3)
# warp_img = F.grid_sample(colors[b, idx][None], x2.reshape(B, N, H, W, 2)[b, i][None].float(), mode="bilinear", align_corners=False)
# warp_img[prob.reshape(B, N, H, W)[b, i, None].repeat(1, 3, 1, 1) < 0.5] = 1
# warp_img = torch.cat([colors[b, i], colors[b, idx], warp_img[0]], dim=-1)
# tensor_to_pil(warp_img).save('test.png')

        return x2.float(), prob

    def dense_flow_loss(self, pred_flow, pred_prob, gt_flow, gt_prob):
        B, N, H, W, _ = gt_flow.shape
        gt_flow = gt_flow.reshape(B*N, H, W, -1)
        gt_prob = gt_prob.reshape(B*N, H, W)
        pred_flow = pred_flow.reshape(B*N, H, W, -1)
        pred_prob = pred_prob.reshape(B*N, H, W)

        gt_flow = torch.clamp(gt_flow, -1, 1)
        
        reg_mask = gt_prob > 0.99
        # reg_loss = F.l1_loss(pred_flow, gt_flow, reduction  = 'none')[reg_mask]

        # robust loss
        # c = 1e-4
        c = 0.06 / max(H, W)
        alpha = 0.5
        x = (pred_flow - gt_flow).norm(dim=-1)[reg_mask]
        reg_loss = (c**alpha) * ((x / c)**2 + 1)**(alpha / 2) 

        certainty_loss = F.binary_cross_entropy_with_logits(pred_prob, gt_prob, reduction='none').mean()

        if not torch.any(reg_loss):
            reg_loss = (pred_flow.mean() * 0.0)  # Prevent issues where prob is 0 everywhere

        return reg_loss.mean(), certainty_loss

    def forward(self, pred, gt, gt_raw):
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        gt_pose = gt['camera_poses']
        gt_intrs = gt['camera_intrinsics']

        B, N, H, W, _ = gt_local_pts.shape
        order_flow = pred['order_flow']

        gt_flow, gt_prob = self.get_gt_flow(order_flow, gt_local_pts[..., 2], gt_pose, gt_intrs, valid_masks, target_H=H, target_W=W)       # dense flow
        gt_flow = gt_flow.reshape(B, N, *gt_flow.shape[1:])
        gt_prob = gt_prob.reshape(B, N, *gt_prob.shape[1:])
        # GameNewDynamic

        if 'flow' in gt_raw[0]:
            flow_batch = [ii for ii, x in enumerate(gt_raw[0]['flow_target']) if x is not None]

            flow_target = []
            flow_prob = []
            for flow_batch_id in flow_batch:
                if torch.is_tensor(gt_raw[0]['flow'][flow_batch_id]):
                    flow_target.append(torch.stack([view['flow'][flow_batch_id] for view in gt_raw], dim=0))
                else:
                    flow_target.append(torch.stack([torch.from_numpy(view['flow'][flow_batch_id]).to(gt_flow.device) for view in gt_raw], dim=0))

                if torch.is_tensor(gt_raw[0]['flow_prob'][flow_batch_id]):
                    flow_prob.append(torch.stack([view['flow_prob'][flow_batch_id] for view in gt_raw], dim=0))
                else:
                    flow_prob.append(torch.stack([torch.from_numpy(view['flow_prob'][flow_batch_id]).to(gt_flow.device) for view in gt_raw], dim=0))
            flow_target = torch.stack(flow_target, dim=0)
            flow_prob = torch.stack(flow_prob, dim=0)
            
            gt_flow[flow_batch] = flow_target
            gt_prob[flow_batch] = flow_prob.float()

# b, i = 1, 3
# idx = order_flow[b, i]
# colors = inverse_CustomNorm(gt['imgs'])
# warp_img = F.grid_sample(colors[b, idx][None], gt_flow.reshape(B, N, H, W, 2)[b, i][None].float(), mode="bilinear", align_corners=False)
# warp_img[gt_prob.reshape(B, N, H, W)[b, i, None].repeat(1, 3, 1, 1) < 0.5] = 1
# warp_img = torch.cat([colors[b, i], colors[b, idx], warp_img[0]], dim=-1)
# tensor_to_pil(warp_img).save('test.png')

        # static_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] not in ['GameNewDynamic', 'GameTrack']]
        static_batch_id = [i for i in range(len(gt['dataset_names'])) if gt['dataset_names'][i] not in []]
        final_loss = 0.0
        if len(static_batch_id) > 0:
            flow_loss, flow_prob_loss = self.dense_flow_loss(pred['flow'][static_batch_id], pred['flow_prob'][static_batch_id], gt_flow[static_batch_id], gt_prob[static_batch_id])

            final_loss += flow_loss * 1.0 * 0.25           # L1 loss
            final_loss += flow_prob_loss * 0.01 * 0.25
        else:
            flow_loss = 0.0 * pred['flow'].mean()
            flow_prob_loss = 0.0 * pred['flow_prob'].mean()
            final_loss += (flow_loss + flow_prob_loss)

        return final_loss, dict(flow_loss=flow_loss, flow_prob_loss=flow_prob_loss)

# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

from models.dinov2.models.rav_sym_test9_flow_v2 import PerceptualLoss
class Pi3XLoss(nn.Module):
    def __init__(
        self,
        use_teacher=True,
        use_pred_normalize=True,
    ):
        super().__init__()
        self.point_loss = PointLoss()
        self.camera_loss = CameraLoss()
        self.flow_loss = FlowLoss()

        self.use_pred_normalize = use_pred_normalize

        if use_teacher:
            self.prepare_teacher()
            self.prepare_segformer()

    def _init_frozen_module(self, module):
        """Helper method to initialize and freeze a module's parameters."""
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
        return module

    def prepare_segformer(self):
        from models.segformer.model import EncoderDecoder
        self.segformer = EncoderDecoder()
        self.segformer.load_state_dict(torch.load('ckpts/segformer.b0.512x512.ade.160k.pth', map_location=torch.device('cpu'), weights_only=False)['state_dict'])
        self.segformer = self.segformer.cuda().eval()

    def predict_sky_mask(self, imgs, chunk_size=8):
        # with torch.no_grad():
        #     output = self.segformer.inference_(imgs)
        #     output = output == 2
        # return output

        outputs = []
        N = len(imgs)
        with torch.no_grad():
            for i in range(0, N, chunk_size):
                chunk_imgs = imgs[i : i + chunk_size]
                chunk_output = self.segformer.inference_(chunk_imgs)
                chunk_output = chunk_output == 2
                outputs.append(chunk_output)
        return torch.cat(outputs, dim=0)

    def prepare_teacher(self):
        from models.dinov2.models.teacher_mv import TeacherMV
        # if mode == 'low_res':
        #     self.teacher = TeacherMV(ckpts='outputs/pi3x_teacher_new_2025-10-24_18-55-54/ckpts/checkpoint_39/pytorch_model.bin').cuda().eval()
        # else:
        self.teacher = TeacherMV(ckpts='outputs/pi3x_teacher_new_hres_2025-10-26_15-35-23/ckpts/checkpoint_49/pytorch_model.bin').cuda().eval()

    def teach(self, gt, ref_idxs=None):
        teach_dataset = __TEACH_DATASETS__

        teach_batch_id = [i for i, x in enumerate(gt['dataset_names']) if x in teach_dataset]

        if len(teach_batch_id) == 0:
            return gt
        imgs = gt['imgs'][teach_batch_id]
        B, N, _, H, W = imgs.shape
        device = imgs.device

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            result = self.teacher(imgs, gt['camera_poses'][teach_batch_id], gt['camera_intrinsics'][teach_batch_id])

        ## using predicted camera
        new_depth = result['local_points']
        pix = torch.from_numpy(get_pixel(H, W).T.reshape(H, W, 3)).to(imgs.device).float()[:, :].repeat(B, N, 1, 1, 1)
        gt_rays = torch.einsum('bnij, bnhwj -> bnhwi', torch.inverse(gt['camera_intrinsics'][teach_batch_id]), pix)[..., :2]
        pred_pts_cam = torch.cat([gt_rays * new_depth, new_depth], dim=-1)

        gt['local_points'][teach_batch_id] = pred_pts_cam
        if ref_idxs is None:
            w2c_target = se3_inverse(result['camera_poses'][:, 0])
        else:
            w2c_target = se3_inverse(result['camera_poses'][torch.arange(B, device=device), ref_idxs])
        result['camera_poses'] = torch.einsum('bij, bnjk -> bnik', w2c_target, result['camera_poses'])
        gt['camera_poses'][teach_batch_id] = result['camera_poses']
        gt['global_points'][teach_batch_id] = torch.einsum('bnij, bnhwj -> bnhwi', result['camera_poses'], homogenize_points(pred_pts_cam))[..., :3]

        sky_mask = self.predict_sky_mask(imgs.reshape(B*N, 3, H, W)).reshape(B, N, H, W)
        is_all_sky = sky_mask.all(dim=-1).all(dim=-1)
        is_all_sky_expanded = is_all_sky.unsqueeze(-1).unsqueeze(-1)
        sky_mask = torch.where(
            is_all_sky_expanded,
            torch.zeros_like(sky_mask),  # 纠正为 "全不是天空" (False)
            sky_mask                     # 保留原始值
        )
        edge = depth_edge(new_depth, rtol=0.03)[..., 0]
        gt['valid_masks'][teach_batch_id] = torch.logical_and(~sky_mask, ~edge)

        all_pts = gt['local_points'][teach_batch_id].clone()
        all_pts[~gt['valid_masks'][teach_batch_id]] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (gt['valid_masks'][teach_batch_id].float().sum(dim=[-1, -2, -3]) + 1e-8)
        norm_factor[gt['valid_masks'][teach_batch_id].float().sum(dim=[-1, -2, -3]) == 0] = 1

        gt['global_points'][teach_batch_id] /= norm_factor.view(B, 1, 1, 1, 1)
        gt['local_points'][teach_batch_id] /= norm_factor.view(B, 1, 1, 1, 1)
        gt['camera_poses'][teach_batch_id, ..., :3, 3] /= norm_factor.view(B, 1, 1)
        gt['norm_factor'][teach_batch_id] *= norm_factor

        if 'normal' in gt:
            gt['normal_batch_id'] = sorted(teach_batch_id + gt['normal_batch_id'])
            gt['normal'][teach_batch_id] = F.normalize(result['normal'], p=2, dim=-1, eps=1e-8)
            gt['normal_mask'][teach_batch_id] = gt['valid_masks'][teach_batch_id]

        return gt

    def prepare_gt(self, gt, ref_idxs=None):
        gt_pts = torch.stack([view['pts3d'] for view in gt], dim=1)
        masks = torch.stack([view['valid_mask'] for view in gt], dim=1)
        poses = torch.stack([view['camera_pose'] for view in gt], dim=1)
        imgs = torch.stack([view['img'] for view in gt], dim=1)

        device = imgs.device
        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        if ref_idxs is None:
            w2c_target = se3_inverse(poses[:, 0])
        else:
            w2c_target = se3_inverse(poses[torch.arange(B, device=device), ref_idxs])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_local_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            gt_local_pts[valid_batch] = gt_local_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        dataset_names = gt[0]['dataset']
        gt_intrs = torch.stack([view['camera_intrinsics'] for view in gt], dim=1)

        sparse_depth_masks = torch.stack([view['sparse_depth'] for view in gt], dim=1) > 0

        ret = dict(
            imgs=imgs,
            global_points=gt_pts,
            local_points=gt_local_pts,
            sparse_depth_masks=sparse_depth_masks,
            valid_masks=masks,
            camera_poses=poses,
            camera_intrinsics=gt_intrs,
            dataset_names=dataset_names,
            norm_factor=norm_factor,
            overlap_mask=gt[0]['overlap_mask'] if 'overlap_mask' in gt[0] else None,
            overlap_matrix=gt[0]['overlap_matrix'] if 'overlap_matrix' in gt[0] else None
        )

        return ret
    
    def normalize_pred(self, pred, gt):
        masks = gt['valid_masks']
        local_points = pred['local_points']

        B, N, H, W, _ = local_points.shape
        
        if self.use_pred_normalize:
            # normalize predict points
            all_pts = local_points.clone()
            all_pts[~masks] = 0
            all_pts = all_pts.reshape(B, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
            norm_factor[masks.sum([-1, -2, -3]) == 0] = 1

            local_points  = local_points / norm_factor[..., None, None, None, None]

            pred['local_points'] = pred['local_points'] / norm_factor[..., None, None, None, None]
            pred['camera_poses'][..., :3, 3] /= norm_factor.view(B, 1, 1)
            pred['norm_factor'] = norm_factor
        else:
            pred['norm_factor'] = torch.ones((B,), device=local_points.device)

        return pred

    def forward(self, pred, gt_raw):
        gt = self.prepare_gt(gt_raw, ref_idxs=pred['ref_idxs'] if 'ref_idxs' in pred else None)
        if hasattr(self, 'teacher'):
            gt = self.teach(gt, ref_idxs=pred['ref_idxs'] if 'ref_idxs' in pred else None)
        pred = self.normalize_pred(pred, gt)

# colors = torch.stack([inverse_CustomNorm(view['img']) for view in gt_raw], dim=1)
# B, N, _, H, W = colors.shape 
# bb = 0
# global_pts = torch.einsum('bnij, bnhwj -> bnhwi', pred['camera_poses'], homogenize_points(pred['local_points']))[..., :3]
# write_ply(global_pts[bb], colors[bb], 'test.ply')

        final_loss = 0.0
        details = dict()

        # Local Point Loss
        point_loss, point_loss_details, scale = self.point_loss(pred, gt, self)
        final_loss += point_loss
        details.update(point_loss_details)

        # Camera Loss
        camera_loss, camera_loss_details = self.camera_loss(pred, gt, scale)
        final_loss += camera_loss
        details.update(camera_loss_details)

        # # Flow Loss
        # flow_loss, flow_loss_details = self.flow_loss(pred, gt, gt_raw)
        # final_loss += flow_loss
        # details.update(flow_loss_details)

        return final_loss, details
    