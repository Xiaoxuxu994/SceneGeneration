import torch
import numpy as np

from typing import Tuple, Optional, List
from einops import rearrange

from .platform import DEVICE_NAME
from utils.log_utils import depth2pointmap
from utils.common import WeightedMAD, calc_extr_inv, calc_intr_inv


class Clamp_lower(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        return input.clamp(min=1e-6)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

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
                      device: torch.device=DEVICE_NAME, eps=1e-2, alpha=0.01):
    """Compute chamfer disance error between two group of (depths, cameras).

    Args:
        depth (torch.Tensor): Depths, tensor of shape (B, 1, H, W).
        extrs (torch.Tensor): Extrinsics, tensor of shape (B, 4, 4). world to camera.
        intrs (torch.Tensor): Intrinsics, tensor of shape (B, 3, 3).
        matches (tuple[torch.Tensor, torch.Tensor]): Matched points in each pair of images.
    """

    H, W = depth.shape[-2:]
    pixel_normalizer = torch.tensor([W, H, 1], dtype=torch.float, device=device)
    
    weighted_mad = WeightedMAD()
    
    points_world = depth2pointmap(depth, extrs, intrs).permute(0, 3, 1, 2) # b 3 h w
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
        image_size = torch.tensor([W, H], device=device)[None, None, None]
        pixels = (pixels + 0.5) / image_size * 2 - 1
        samped_points = torch.nn.functional.grid_sample(points, pixels, mode="bilinear", padding_mode="border") # b c 1 k
        samped_points = samped_points.squeeze(dim=2).permute(0, 2, 1) # b k c
        return samped_points
    
    assert len(depth) % view_per_batch == 0
    batch_size = len(depth) // view_per_batch
    
    errors_3d_cos = []
    errors_camera = []
    errors_3d_dist = []
    if matching_info["type"] == "CoTracker":
        tracks = matching_info["matches"] # b v k 2, b v k 2
        vis    = matching_info["visibilities"] # b v k, b v k
        V = tracks.shape[1]
        
        mask = torch.all(vis, dim=1) # b k
        points = samp_points_by_pixel(points_world, tracks.reshape(-1, *tracks.shape[-2:])) # bv k c
        points = points.reshape(-1, view_per_batch, *points.shape[-2:]) # b v k c
        cam_cents = camera_centers.reshape(-1, view_per_batch, *camera_centers.shape[1:]) # b v 1 3
        bextrs = extrs.reshape(-1, view_per_batch, *extrs.shape[1:]) # b v 4 4
        bintrs = intrs.reshape(-1, view_per_batch, *intrs.shape[1:]) # b v 3 3
        unnormed_points = points.clone()
        normed_points = None
        
        cossim_map = compute_angle_loss(unnormed_points, cam_cents) # b v v k
        cossim_map = torch.acos(BoundForArcCos.apply(cossim_map)) # b v v k
        # cossim_map = (1 - cossim_map) / 2
        
        c = None
        if conf is not None:
            c = samp_points_by_pixel(conf, tracks.reshape(-1, *tracks.shape[-2:])) # b v k 1
            
            # c0 = c0.reshape(-1, view_per_batch, *c0.shape[1:]).squeeze(-1) # b v k
            # cossim_map = cossim_map * c0.unsqueeze(2) - 0.01 * torch.log(c0.unsqueeze(2).expand_as(cossim_map)) # b v v k
            
            _, v, k, _ = points.shape
            c = c.reshape(-1, view_per_batch, *c.shape[1:]).expand_as(points) # b v k 3
            normed_points = weighted_mad(rearrange(points, "b v k c -> b c (v k)"), rearrange(c, "b v k c -> b c (v k)"))
            normed_points = rearrange(normed_points, "b c (v k) -> b v k c", v=v)
        
        cossim_loss = rearrange(cossim_map, "b v vv k -> b k (v vv)") # b k v*v
        cossim_loss = rearrange(cossim_loss[..., :-1], "b k (v vv) -> b k v vv", v=V-1)[..., 1:] # b k v-1 v
        cossim_loss = rearrange(cossim_loss, "b k v vv -> b k (v vv)").mean(dim=-1)
        
        cossim_loss = (cossim_loss * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8) # b
        valid_batch_mask = mask.sum(dim=-1) > 64 # b
        cossim_loss = (cossim_loss * valid_batch_mask).sum() / (valid_batch_mask.sum() + 1e-8) # tensor
        
        errors_3d_cos.append(cossim_loss[None])
        
        # camera_basic_mat_loss = camera_basic_mat_loss(bextrs, bintrs, tracks0)
        # errors_camera.append(camera_basic_mat_loss[None])
    else:
        raise NotImplementedError("Matching model type {} is not implemented.".format(matching_info["type"]))
    
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
    if len(errors_3d_dist) > 0:
        errors_3d_dist = torch.cat(errors_3d_dist).mean()
    else:
        errors_3d_dist = torch.tensor(0.0)
    return errors_3d_cos + errors_camera + 0.1 * errors_3d_dist if flag else None


class LossFunction(torch.nn.Module):

    def __init__(self, cfg, matching_model):
        super().__init__()
        self.cfg = cfg
        self.matching_model = matching_model
        self.enable_conf    = cfg.enable_conf
        self.conf_alpha     = cfg.conf_alpha
        self.enable_slide   = cfg.enable_slide
        
    def ds_loss(self, output_dict, data, enable_conf=False, eps=1e-12):
        mono_depths = output_dict["mono_depths"].detach()
        depths = output_dict["depths"]
        ds_loss = depth_shape_loss(depths, mono_depths, 
            conf_map=output_dict["depths_conf"] if enable_conf else None, alpha=self.conf_alpha)
        return ds_loss
    
    def geometric_loss(self, output_dict, data, enable_conf=False, detach_camera=False):
        
        B, V, C, H, W = output_dict["depths"].shape
        errors = []
        
        # # Use depth reprojection error as geometric loss.
        depths_src, extrs_src, intrs_src = [], [], []
        
        depths_src = output_dict["scaled_depths"]
        conf_src   = output_dict["depths_conf"].permute(0, 1, 4, 2, 3)
        extrs_src  = output_dict["camera_extrs"] if output_dict.get("camera_extrs", None) is not None \
            else data["input_viewmats"]
        intrs_src  = data["input_Ks"]
        
        depths_src = depths_src.reshape(B, V, C, H, W)
        confs_src  = conf_src.reshape(B, V, C, H, W)
        extrs_src  = extrs_src.reshape(B, V, 4, 4)
        if detach_camera:
            extrs_src = extrs_src.detach()
        intrs_src  = intrs_src.reshape(B, V, 3, 3).detach()
        
        segment_length = 2 if self.enable_slide else V
        logV = 0
        matching_infos = []
        while segment_length <= V:
            assert V % segment_length == 0
            logV += 1
            # layer_image = data["input_images"].reshape(-1, segment_length, 3, H, W)
            layer_image = torch.cat([data["input_images"][:, idx * (segment_length // 2): idx * (segment_length // 2) + segment_length] \
                for idx in range(V // (segment_length // 2) - 1)], dim=1)
            layer_image = layer_image.reshape(-1, segment_length, 3, H, W)
            
            if data["matching_infos"] is None:
                # raise NotImplementedError("Online matching pseudo labeling has been deprecated, please first generate pseudo matching offline.")
                layer_matching_info = procoss_matching_for_images(layer_image, self.matching_model) # b*(v/s) s k 2
            else:
                # layer_matching_info = data["matching_infos"]["layer{}".format(logV - 1)]
                layer_matching_info = data["matching_infos"]["layer{}".format(logV)]
                layer_matching_info["matches"] = layer_matching_info["matches"][:, ::2, ...]
                layer_matching_info["visibilities"] = layer_matching_info["visibilities"][:, ::2, ...]
            matching_infos.append(layer_matching_info)
            
            depths_flatten = torch.cat([depths_src[:, idx * (segment_length // 2): idx * (segment_length // 2) + segment_length] \
                for idx in range(V // (segment_length // 2) - 1)], dim=1).reshape(-1, C, H, W).contiguous()
            confs_flatten = torch.cat([confs_src[:, idx * (segment_length // 2): idx * (segment_length // 2) + segment_length] \
                for idx in range(V // (segment_length // 2) - 1)], dim=1).reshape(-1, C, H, W).contiguous()
            extrs_flatten  = torch.cat([extrs_src[:, idx * (segment_length // 2): idx * (segment_length // 2) + segment_length] \
                for idx in range(V // (segment_length // 2) - 1)], dim=1).reshape(-1, 4, 4).contiguous()
            intrs_flatten  = torch.cat([intrs_src[:, idx * (segment_length // 2): idx * (segment_length // 2) + segment_length] \
                for idx in range(V // (segment_length // 2) - 1)], dim=1).reshape(-1, 3, 3).contiguous()
            
            view_per_batch = segment_length
            
            matching_error = matching_error_3D(depth=depths_flatten, 
                                               extrs=extrs_flatten, intrs=intrs_flatten, 
                                               matching_info=layer_matching_info, view_per_batch=view_per_batch, 
                                               conf=confs_flatten.detach() if enable_conf else None) * (V / segment_length * 2 - 1)
            if matching_error is not None:
                errors.append(matching_error[None])
            segment_length *= 2
        
        # return torch.cat(errors).mean(), matching_infos
        return torch.cat(errors).sum() / (2 * (V - 1) - logV), matching_infos
    
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
        
    
    def forward(self, output_dict, data, eps=1e-12):
        ### Compute loss weight
        w_ds        = self.cfg.depth_shape.w if self.cfg.get("depth_shape", None) is not None else 0.0
        w_geometric = self.cfg.geometric.w if self.cfg.get("geometric", None) is not None else 0.0
        
        ### Compute loss value
        ds_loss_value = self.ds_loss(output_dict, data, enable_conf=self.enable_conf, eps=eps) if w_ds > 1e-8 else torch.tensor(0.0)
        geometric_loss_value, matching_info = self.geometric_loss(output_dict, data, enable_conf=False) \
            if w_geometric > 1e-8 else torch.tensor(0.0)
        
        total_loss = w_ds * ds_loss_value + w_geometric * geometric_loss_value
        
        ### Return loss information
        loss_dict = {
            "total":         total_loss, 
            "depth_shape":   ds_loss_value, 
            "geometric":     geometric_loss_value, 
            "matching_info": matching_info, 
        }
        return loss_dict