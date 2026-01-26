import torch
import numpy as np

from typing import Tuple, Optional, List
from einops import rearrange

from .platform import DEVICE_NAME
from utils.log_utils import save_pcds_as_ply, depth2pointmap, pointmap2pixel, Depth2PointMap
from utils.common import WeightedMAD, calc_extr_inv, calc_intr_inv


class ClampLower(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, lower_bound: float):
        return input.clamp(min=lower_bound)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None

class BoundForArcCos(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        return input.clamp(min=-1+1e-5, max=1-1e-5)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


def mse_loss(x: torch.Tensor, y: torch.Tensor, revert_input=False, return_map=False) -> torch.Tensor:
    """Mean squared error loss.

    Args:
        x (torch.Tensor): Input, tensor of shape (B, V, C, H, W)
        y (torch.Tensor): Input, tensor of shape (B, V, C, H, W)

    Returns:
        torch.Tensor: Loss tensor.
    """
    if revert_input:
        assert x.min() >= 0 and y.min() >= 0, "Input should be no less than 0, but found x.min={} and y.min={}".format(x.min(), y.min())
        x, y = 1 / (x + 0), 1 / (y + 0)
    if return_map:
        return (x - y).square()
    return (x - y).square().mean()


def l1_loss(x: torch.Tensor, y: torch.Tensor, revert_input=False) -> torch.Tensor:
    # assert x.min() >= 0 and y.min() >= 0, "Input should be no less than 0, but found x.min={} and y.min={}".format(x.min(), y.min())
    if revert_input:
        x, y = 1 / (x + 0), 1 / (y + 0)
    return (x - y).abs().mean()


def calc_pixel_grad_map(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    assert x.shape[-3] == 1, "The input of pixel gradient calculation should be a depth map."
    H, W = x.shape[-2:]
    step = 1
    maps = []
    while step < H and step < W:
        _x = x[..., ::step, ::step]
        hori_grad = _x[..., :-1, :-1] - _x[..., :-1, 1:] # ..., 1, H-1, W-1
        vert_grad = _x[..., :-1, :-1] - _x[..., 1:, :-1] # ..., 1, H-1, W-1
        norm_grad = (hori_grad.square() + vert_grad.square() + eps).sqrt() + eps # ..., 1, H-1, W-1
        maps.append(torch.cat([hori_grad / norm_grad, vert_grad / norm_grad], dim=-3)) # Concatinate by channel.
        step *= 2
    return maps


def depth_shape_loss(x: torch.Tensor, y: torch.Tensor, 
                     conf_map: torch.Tensor=None, alpha=0.05) -> torch.Tensor:
    """Calculate depth shape loss.

    Args:
        x (torch.Tensor): Depth map of prediction, tensor of shape (N, C, H, W).
        y (torch.Tensor): Depth map of ground truth, tensor of shape (N, C, H, W).
        conf_map (torch.Tensor): Predicted confidence map of depth maps, tensor of shape (N, C, H, W).
        intrinsics (torch.Tensor): Camera intrinsics of each view, tensor of shape (N, 3, 3)

    Returns:
        torch.Tensor: Loss value.
    """
    assert x.shape == y.shape, "The shape of two input tensors must be equivalent, receiving x {} and y {}.".format(x.shape, y.shape)
    if x.ndim == 5:
        x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        y = y.reshape(y.shape[0] * y.shape[1], *y.shape[2:])
        if conf_map is not None:
            conf_map = conf_map.reshape(*x.shape)
    
    weighted_mad = WeightedMAD()
    x = weighted_mad(x.reshape(*x.shape[:-3], -1), 
                     conf_map.reshape(*x.shape[:-3], -1).detach() if conf_map is not None else None).reshape(*x.shape)
    y = weighted_mad(y.reshape(*y.shape[:-3], -1), 
                     conf_map.reshape(*x.shape[:-3], -1).detach() if conf_map is not None else None).reshape(*y.shape)
    scale_invariant_loss_map = (x - y).abs()
    if conf_map is None:
        scale_invariant_loss_value = scale_invariant_loss_map.mean()
    else:
        scale_invariant_loss_value = (conf_map * scale_invariant_loss_map - alpha * torch.log(conf_map)).mean()
    
    x_grad_maps = calc_pixel_grad_map(x)
    y_grad_maps = calc_pixel_grad_map(y)
    depth_grad_loss_maps = [(x_grad_map - y_grad_map).abs() for x_grad_map, y_grad_map in zip(x_grad_maps, y_grad_maps)]
    # if conf_map is None:
    depth_grad_loss_value = torch.cat([loss_map.mean()[None] for loss_map in depth_grad_loss_maps]).mean()
    # depth_grad_loss_value = depth_grad_loss_maps[0].mean()
    # else:
    #     depth_grad_loss_value = ((conf_map[..., :-1, :-1] * depth_grad_loss_maps[0]).mean() - alpha * torch.log(conf_map[..., :-1, :-1])).mean()
    
    depth_shape_loss_value = (scale_invariant_loss_value + depth_grad_loss_value) / 2
    
    return depth_shape_loss_value #, s, t


@torch.no_grad()
def procoss_matching_for_images(images: torch.Tensor, matching_model):
    """Calculate matching for adjacent images.

    Args:
        images (torch.Tensor): Images, tensor of shape (B, V, C, H, W)
    """
    
    if matching_model.cfg.name == "CoTracker":
        assert images.ndim == 5
        tracks, vis = matching_model.infer(images) # B V N 2, B V N 1
        return {
            "type": "CoTracker", 
            "matches": tracks, 
            "visibilities": vis, 
        }
    else:
        raise NotImplementedError(f"Matching model type {matching_model.cfg.name} is not implemented.")


def camera_basic_mat_loss(extrs: torch.Tensor, intrs: torch.Tensor, bmatches: torch.Tensor):
    """Basic matrix constraint for camera poses.

    Args:
        extrs (torch.Tensor): Tensor of shape (..., V, 4, 4)
        intrs (torch.Tensor): Tensor of shape (..., V, 3, 3)
        bmatches (torch.Tensor): Tensor of shape (..., V, k, 2)

    Returns:
        _type_: _description_
    """
    T_x = torch.zeros_like(extrs[..., :3, :3]) # ... 3 3
    T_x[..., 0, 1] = T_x[..., 0, 1] - extrs[..., 2, 3]
    T_x[..., 1, 0] = T_x[..., 1, 0] + extrs[..., 2, 3]
    T_x[..., 0, 2] = T_x[..., 0, 2] + extrs[..., 1, 3]
    T_x[..., 2, 0] = T_x[..., 2, 0] - extrs[..., 1, 3]
    T_x[..., 1, 2] = T_x[..., 1, 2] - extrs[..., 0, 3]
    T_x[..., 2, 1] = T_x[..., 2, 1] + extrs[..., 0, 3]
    intr_inv = calc_intr_inv(intrs)
    basic_matrices = intr_inv.transpose(-1, -2) @ T_x @ extrs[..., :3, :3] @ intr_inv # ... 3 3
    
    bhomo_pixels = torch.cat([bmatches, torch.ones_like(bmatches[..., :1])], dim=-1)[..., None] # ... k' 3 1
    
    return(bhomo_pixels.transpose(-1, -2) @ basic_matrices.unsqueeze(-3) @ bhomo_pixels).abs().mean()


def compute_angle_loss(unnormed_points: torch.Tensor, cam_cents: torch.Tensor):
    """Compute matching angle loss as 3d correspondence loss.
    Args:
        unnormed_points (torch.Tensor): Tensor of shape (B, V, K, 3)
        cam_cents (torch.Tensor): Tensor of shape (B, V, 1, 3)
    Returns:
        Tensor of shape (B, V, V, K), the exhaustive angle loss map.
    """
    vec_mat  = unnormed_points.unsqueeze(2) - cam_cents.unsqueeze(1) # b v v k 3, points from view v_i <- center of view v_j
    cent_mat = (unnormed_points - cam_cents).unsqueeze(1).expand_as(vec_mat).detach() # b v v k 3
    
    dot_product = (vec_mat * cent_mat).sum(dim=-1)
    norm_vec    = vec_mat.norm(dim=-1, p=2)
    norm_cent   = cent_mat.norm(dim=-1, p=2)
    cossim      = dot_product / (norm_vec * norm_cent + 1e-8)
    
    return cossim # b v v k


def matching_error_3D(depth: torch.Tensor, 
                      extrs: torch.Tensor, intrs: torch.Tensor, 
                      matching_info: dict,  view_per_batch: int, 
                      conf: torch.Tensor=None, 
                      device: torch.device=DEVICE_NAME, eps=1e-2, alpha=0.01, depth_is_z=True):
    """Compute 3d error between two group of (depths, cameras).

    Args:
        depth (torch.Tensor): Depths, tensor of shape (B, 1, H, W).
        extrs (torch.Tensor): Extrinsics, tensor of shape (B, 4, 4). world to camera.
        intrs (torch.Tensor): Intrinsics, tensor of shape (B, 3, 3).
    """

    H, W = depth.shape[-2:]
    
    weighted_mad = WeightedMAD()
    
    points_world = depth2pointmap(depth, extrs, intrs, depth_is_z=depth_is_z) # b 3 h w
    camera_centers = calc_extr_inv(extrs)[..., :3, -1].unsqueeze(1) # b 1 3
    
    def samp_points_by_pixel(points: torch.Tensor, pixels: torch.Tensor):
        """_summary_

        Args:
            points (torch.Tensor): Tensor of shape (b, c, h, w)
            pixels (torch.Tensor): Tensor of shape (b, k, 2)

        Returns:
            _type_: _description_
        """
        assert points.ndim == 4 and pixels.ndim == 3
        pixels = pixels.unsqueeze(1) # b 1 k 2
        # image_size = torch.tensor([W, H], device=device)[None, None, None]
        # pixels = (pixels + 0.5) / image_size * 2 - 1
        pixels = pixels + 0.5
        pixels[..., 0] = pixels[..., 0] / W
        pixels[..., 1] = pixels[..., 1] / H
        pixels = pixels * 2 - 1
        samped_points = torch.nn.functional.grid_sample(points, pixels, mode="bilinear", padding_mode="border") # b c 1 k
        samped_points = samped_points.squeeze(dim=2).permute(0, 2, 1) # b k c
        return samped_points
    
    assert len(depth) % view_per_batch == 0
    batch_size = len(depth) // view_per_batch
    
    errors_3d_cos = []
    errors_camera = []
    tracks = matching_info["matches"] # b v k 2, b v k 2
    vis    = matching_info["visibilities"] # b v k, b v k
    V = tracks.shape[1]
    
    mask = torch.all(vis, dim=1) # b k
    points = samp_points_by_pixel(points_world, tracks.reshape(-1, *tracks.shape[-2:])) # bv k c
    points = points.reshape(-1, view_per_batch, *points.shape[-2:]) # b v k c
    cam_cents = camera_centers.reshape(-1, view_per_batch, *camera_centers.shape[1:]) # b v 1 3
    unnormed_points = points.clone()
    normed_points = None
    
    cossim_map = compute_angle_loss(unnormed_points, cam_cents) # b v v k
    cossim_map = torch.acos(BoundForArcCos.apply(cossim_map)) # b v v k
    # cossim_map = (1 - cossim_map) / 2
    
    c = None
    if conf is not None:
        c = samp_points_by_pixel(conf, tracks.reshape(-1, *tracks.shape[-2:])) # b v k 1
        # c0 = c0.reshape(-1, view_per_batch, *c0.shape[1:]).squeeze(-1) # b v k
        # cossim_map = cossim_map * c0.unsqueeze(2) - alpha * torch.log(c0.unsqueeze(2).expand_as(cossim_map)) # b v v k
        # _, v, k, _ = points.shape
        c = c.reshape(-1, view_per_batch, *c.shape[1:]).expand_as(points) # b v k 3
        normed_points = weighted_mad(rearrange(points, "b v k c -> b c (v k)"), rearrange(c, "b v k c -> b c (v k)"))
        normed_points = rearrange(normed_points, "b c (v k) -> b v k c", v=view_per_batch)
    
    cossim_loss = rearrange(cossim_map, "b v vv k -> b k (v vv)") # b k v*v
    cossim_loss = rearrange(cossim_loss[..., :-1], "b k (v vv) -> b k v vv", v=view_per_batch-1)[..., 1:] # b k v-1 v
    cossim_loss = rearrange(cossim_loss, "b k v vv -> b k (v vv)").mean(dim=-1)
    
    cossim_loss = (cossim_loss * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8) # b
    ## Currently we filter invalid pairs in the dataloader. So these two lines can be commented out. 
    valid_batch_mask = mask.sum(dim=-1) > 16 # b
    cossim_loss = (cossim_loss * valid_batch_mask).sum() / (valid_batch_mask.sum() + 1e-8) # tensor
    
    errors_3d_cos.append(cossim_loss[None])
    
    # camera_basic_mat_loss = camera_basic_mat_loss(bextrs, bintrs, tracks0)
    # errors_camera.append(camera_basic_mat_loss[None])
    # else:
    #     raise NotImplementedError("Matching model type {} is not implemented.".format(matching_info["type"]))
    
    flag = False
    if len(errors_3d_cos) > 0:
        errors_3d_cos = torch.cat(errors_3d_cos).mean()
        flag = True
    else:
        errors_3d_cos = torch.tensor(0.0)
    if len(errors_camera) > 0:
        errors_camera = torch.cat(errors_camera).mean()
    else:
        errors_camera = torch.tensor(0.0)
    return errors_3d_cos + errors_camera if flag else None


class LossFunction(torch.nn.Module):

    def __init__(self, cfg, matching_model):
        super().__init__()
        self.cfg = cfg
        self.matching_model = matching_model
        self.enable_conf    = cfg.enable_conf
        self.conf_alpha     = cfg.conf_alpha
        self.enable_slide   = cfg.enable_slide
        self.pred_z_depth   = cfg.pred_z_depth
        self.eye3           = torch.eye(3, device=DEVICE_NAME)
        self.use_pred_intr  = cfg.use_pred_intr
        
    def ds_loss(self, output_dict, data, enable_conf=False, eps=1e-12):
        mono_depths = output_dict["mono_depths"].detach()
        depths = output_dict["depths"]
        input_masks = torch.ones_like(output_dict["depths_conf"]) if data["input_depth_masks"] is None else data["input_depth_masks"]
        input_masks = ClampLower.apply(input_masks, 0.01)
        ds_loss = depth_shape_loss(depths, mono_depths, 
            conf_map=output_dict["depths_conf"] * input_masks if enable_conf else None, alpha=self.conf_alpha)
        output_dict["depths_conf"] = output_dict["depths_conf"] * input_masks
        return ds_loss
        
    def geometric_loss(self, output_dict, data, enable_conf=False, detach_camera=False, depth_is_z=True, use_pred_intr=False):
        
        B, V, C, H, W = output_dict["depths"].shape
        
        # # Use depth reprojection error as geometric loss.
        depths_src, extrs_src, intrs_src = [], [], []
        
        depths_src = output_dict["scaled_depths"]
        confs_src  = output_dict["depths_conf"].permute(0, 1, 4, 2, 3)
        extrs_src  = output_dict["camera_extrs"] if output_dict.get("camera_extrs", None) is not None \
            else data["input_viewmats"]
        if not use_pred_intr:
            intrs_src = data["input_Ks"]
        else:
            fovs_pred = output_dict["camera_fovs"] # b v 2
            focal_pred = 0.5 / torch.tan(fovs_pred / 2)
            intrs_src = self.eye3[None, None].expand(B, V, -1, -1).clone()
            intrs_src[..., :2, :] = intrs_src[..., :2, :] * focal_pred[..., None]
            intrs_src[..., 0, 2] = 0.5
            intrs_src[..., 1, 2] = 0.5
        
        depths_src = depths_src.reshape(B, V, C, H, W)
        confs_src  = confs_src.reshape(B, V, C, H, W)
        extrs_src  = extrs_src.reshape(B, V, 4, 4)
        if detach_camera:
            extrs_src = extrs_src.detach()
        intrs_src  = intrs_src.reshape(B, V, 3, 3).detach()
        
        tracks0 = data["input_tracks0"] # B P K 2
        tracks1 = data["input_tracks1"] # B P K 2
        visibs0 = data["input_visibs0"] # B P K
        visibs1 = data["input_visibs1"] # B P K
        matching_info = dict(
            matches=torch.stack([rearrange(tracks0, "b p k c -> (b p) k c"), rearrange(tracks1, "b p k c -> (b p) k c")], dim=1), # (b p) 2 k c
            visibilities=torch.stack([rearrange(visibs0, "b p k -> (b p) k"), rearrange(visibs1, "b p k -> (b p) k")], dim=1), # (b p) 2 k
            matched_views0=data["input_tracks_view_id0"], # B P
            matched_views1=data["input_tracks_view_id1"], # B P
        )
        
        paired_depths, paired_confs, paired_extrs, paired_intrs = [], [], [], []
        for depths, confs, extrs, intrs, pairs_x, pairs_y in zip(
            depths_src, confs_src, extrs_src, intrs_src, data["input_tracks_view_id0"], data["input_tracks_view_id1"]):
            paired_depths.append(torch.stack([depths[pairs_x], depths[pairs_y]], dim=1)) # <- P 2 C H W
            paired_confs.append(torch.stack([confs[pairs_x], confs[pairs_y]], dim=1))    # <- P 2 1 H W
            paired_extrs.append(torch.stack([extrs[pairs_x], extrs[pairs_y]], dim=1))    # <- P 2 4 4
            paired_intrs.append(torch.stack([intrs[pairs_x], intrs[pairs_y]], dim=1))    # <- P 2 3 3
        paired_depths = torch.stack(paired_depths).reshape(-1, 2, C, H, W)
        paired_confs = torch.stack(paired_confs).reshape(-1, 2, C, H, W)
        paired_extrs = torch.stack(paired_extrs).reshape(-1, 2, 4, 4)
        paired_intrs = torch.stack(paired_intrs).reshape(-1, 2, 3, 3)
        
        matching_error = matching_error_3D(depth=paired_depths.view(-1, C, H, W), 
                                           extrs=paired_extrs.view(-1, 4, 4), 
                                           intrs=paired_intrs.view(-1, 3, 3), 
                                           matching_info=matching_info, 
                                           view_per_batch=2, 
                                           conf=paired_confs.view(-1, C, H, W) if enable_conf else None, 
                                           depth_is_z=depth_is_z)
                
        return matching_error, [matching_info, ]
    
    
    def relative_camera_loss(self, output_dict: dict[str, torch.Tensor], data: dict[str, torch.Tensor]):
        extrs_gt   = data["input_viewmats"] # b v 4 4
        extrs_pred = output_dict["camera_extrs"] # b v 4 4
        
        gt_t  = extrs_gt[..., :3, 3] # b v 3
        gt_t_magnitude = gt_t.norm(dim=-1).mean(dim=-1)[:, None, None] # b 1 1
        gt_t  = gt_t / (gt_t_magnitude + 1e-8)
        gt_r  = extrs_gt[..., :3, :3] # b v 3 3
        
        pred_t = extrs_pred[..., :3, 3] # b v 3
        pred_r = extrs_pred[..., :3, :3] # b v 3 3
        
        gt_rel_rot   = gt_r.unsqueeze(1) @ gt_r.unsqueeze(2).transpose(-1, -2) # b v v 3 3
        pred_rel_rot = pred_r.unsqueeze(1) @ pred_r.unsqueeze(2).transpose(-1, -2) # b v v 3 3
        rel_rot_inv_mul = (gt_rel_rot.transpose(-1, -2) @ pred_rel_rot) # b v v 3 3
        trace = rel_rot_inv_mul[..., 0, 0] + rel_rot_inv_mul[..., 1, 1] + rel_rot_inv_mul[..., 2, 2]
        
        L_r = torch.acos(BoundForArcCos.apply((trace - 1) / 2)).mean()
        L_t = torch.nn.functional.huber_loss(pred_t, gt_t, reduction="mean", delta=1.0)
        
        return L_r + 100.0 * L_t
        
    
    def intr_loss(self, output_dict, data):
        pred_fov = output_dict["camera_fovs"] # b v 2
        gt_focal = torch.stack([data["input_Ks"][..., 0, 0], data["input_Ks"][..., 1, 1]], dim=-1) # b v 2
        gt_fov = torch.atan2(0.5 * torch.ones_like(gt_focal), gt_focal) * 2 # b v 2
        return (pred_fov - gt_fov).abs().mean()
    
    
    def forward(self, output_dict, data, eps=1e-12):
        ### Compute loss weight
        w_ds        = self.cfg.depth_shape.w if self.cfg.get("depth_shape", None) is not None else 0.0
        w_geometric = self.cfg.geometric.w if self.cfg.get("geometric", None) is not None else 0.0
        w_intr      = self.cfg.camera.w if self.cfg.get("camera", None) is not None else 0.0
        
        ### Compute loss value
        ds_loss_value = self.ds_loss(output_dict, data, enable_conf=self.enable_conf, eps=eps) if w_ds > 1e-8 else torch.tensor(0.0)
        geometric_loss_value, matching_info = self.geometric_loss(output_dict, data, enable_conf=False, 
                                                                  depth_is_z=self.pred_z_depth, use_pred_intr=self.use_pred_intr)
        intr_loss_value = self.intr_loss(output_dict, data)
        
        total_loss = w_ds * ds_loss_value + w_geometric * geometric_loss_value + w_intr * intr_loss_value
        
        ### Return loss information
        loss_dict = {
            "total":         total_loss, 
            "depth_shape":   ds_loss_value, 
            "geometric":     geometric_loss_value, 
            "intr_fov":      intr_loss_value, 
            "matching_info": matching_info, 
        }
        return loss_dict