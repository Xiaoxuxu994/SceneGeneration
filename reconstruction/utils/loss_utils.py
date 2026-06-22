import torch

from einops import rearrange

from .platform import DEVICE_NAME
from utils.common import WeightedMAD, calc_extr_inv, intr2ray, localpoint2global, mat_mul_vec_by_conv, pad_one, normalize_trans, get_norm_map
from submodules.pi3.utils.alignment import align_points_scale


class ClampLower(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, lower_bound: float):
        if lower_bound is not None:
            return input.clamp(min=lower_bound)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None
    
class ClampUpper(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, upper_bound: float):
        if upper_bound is not None:
            return input.clamp(max=upper_bound)
        return input

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

class GradBalancerForCamera(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        # balanced_grad = grad_output.clone()
        # balanced_grad[..., :3, 3] = grad_output[..., :3, 3] * 1.0
        # return balanced_grad
        # return grad_output * 100.0
        return grad_output


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


def weighted_mean(x: torch.Tensor, w: torch.Tensor=None, dim=None, keepdim: bool=False, eps: float=1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def calc_pixel_grad_map(x: torch.Tensor, w: torch.Tensor, eps=1e-6, mode="vert", max_layer=4) -> torch.Tensor:
    assert x.shape[-3] == 1, "The input of pixel gradient calculation should be a depth map."
    assert mode in ["vert", "hori"]
    H, W = x.shape[-2:]
    step = 1
    maps, weights = [], []
    # while step < H and step < W:
    #     _x = x[..., ::step, ::step]
    #     hori_grad = _x[..., :-1, :-1] - _x[..., :-1, 1:] # ..., 1, H-1, W-1
    #     vert_grad = _x[..., :-1, :-1] - _x[..., 1:, :-1] # ..., 1, H-1, W-1
    #     norm_grad = (hori_grad.square() + vert_grad.square() + eps).sqrt() + eps # ..., 1, H-1, W-1
    #     maps.append(torch.cat([hori_grad / norm_grad, vert_grad / norm_grad], dim=-3)) # Concatinate by channel.
    #     _w = w[..., ::step, ::step]
    #     total_grad_w = _w[..., :-1, :-1] * _w[..., :-1, 1:] * _w[..., 1:, :-1]
    #     weights.append(total_grad_w)
    #     step *= 2
    while step < H and step < W and step < 2**max_layer:
        _x = x[..., ::step, ::step]
        if mode == "hori":
            grad = _x[..., :-1, :-1] - _x[..., :-1, 1:] # ..., 1, H-1, W-1
        else:
            grad = _x[..., :-1, :-1] - _x[..., 1:, :-1] # ..., 1, H-1, W-1
        maps.append(grad)
        _w = w[..., ::step, ::step]
        if mode == "hori":
            grad_w = _w[..., :-1, :-1] * _w[..., :-1, 1:]
        else:
            grad_w = _w[..., :-1, :-1] * _w[..., 1:, :-1]
        weights.append(grad_w)
        step *= 2
    return maps, weights


def depth_shape_loss(x: torch.Tensor, y: torch.Tensor, 
                     conf_map: torch.Tensor=None, valid_map: torch.Tensor=None, 
                     alpha=0.05, calc_grad_loss=True, norm_depth=True, prefix="depth", conf4grad=False) -> torch.Tensor:
    """Calculate depth shape loss.

    Args:
        x (torch.Tensor): Depth map of prediction, tensor of shape (N, C, H, W).
        y (torch.Tensor): Depth map of ground truth, tensor of shape (N, C, H, W).
        conf_map (torch.Tensor): Predicted confidence map of depth maps, tensor of shape (N, C, H, W).

    Returns:
        torch.Tensor: Loss value.
    """
    assert x.shape == y.shape, "The shape of two input tensors must be equivalent, receiving x {} and y {}.".format(x.shape, y.shape)
    if x.ndim == 5:
        B, V = x.shape[:2]
        x = x.reshape(B * V, *x.shape[2:])
        y = y.reshape(B * V, *y.shape[2:])
        if conf_map is not None:
            conf_map = conf_map.reshape(B * V, *conf_map.shape[2:])
        if valid_map is not None:
            valid_map = valid_map.reshape(B * V, *valid_map.shape[2:])
    
    detail_info = {}
    if norm_depth:
        weighted_mad = WeightedMAD()
        # x = weighted_mad(x.reshape(*x.shape[:-3], -1), 
        #                 conf_map.reshape(*x.shape[:-3], -1).detach() if conf_map is not None else None).reshape(*x.shape)
        # y = weighted_mad(y.reshape(*y.shape[:-3], -1), 
        #                 conf_map.reshape(*x.shape[:-3], -1).detach() if conf_map is not None else None).reshape(*y.shape)
        x = weighted_mad(x.reshape(*x.shape[:-3], -1), 
                        valid_map.reshape(*x.shape[:-3], -1).detach() if valid_map is not None else None).reshape(*x.shape)
        y = weighted_mad(y.reshape(*y.shape[:-3], -1), 
                        valid_map.reshape(*x.shape[:-3], -1).detach() if valid_map is not None else None).reshape(*y.shape)
    scale_invariant_loss_map = (x - y).abs().mean(dim=-3, keepdim=True)
    
    ### Regression loss
    # depth_weight = 1 + 1 / (y.clamp(min=0.1*weighted_mean(y, valid_map, dim=(-2, -1), keepdim=True)) + 1e-8)
    # if not norm_depth:
    detail_info["{}_regression".format(prefix)] = (scale_invariant_loss_map * valid_map / (valid_map.sum() + 1e-8)).sum()
    
    ### Confidence loss
    if conf_map is not None:
        scale_invariant_loss_value = ((conf_map * scale_invariant_loss_map - alpha * torch.log(conf_map)) 
                                      * valid_map / (valid_map.sum() + 1e-8)
                                      ).sum()
        detail_info["{}_confidence".format(prefix)] = scale_invariant_loss_value
    
    ### Gradient loss
    if calc_grad_loss:
        # x_grad_maps, grad_weights = calc_pixel_grad_map(x, conf_map)
        # y_grad_maps, grad_weights = calc_pixel_grad_map(y, conf_map)
        # depth_grad_loss_maps = [(x_grad_map - y_grad_map).abs().mean(dim=-3, keepdim=True) * weight / (weight.sum() + 1e-8)
        #                         for x_grad_map, y_grad_map, weight in zip(x_grad_maps, y_grad_maps, grad_weights)]
        # depth_grad_loss_value = torch.cat([loss_map.sum()[None] for loss_map in depth_grad_loss_maps]).mean()
        # depth_shape_loss_value = (scale_invariant_loss_value + depth_grad_loss_value) / 2
        grad_weight = valid_map * conf_map if conf4grad and conf_map is not None else valid_map
        x_hori_grad_maps, hori_grad_weights = calc_pixel_grad_map(x, grad_weight, mode="hori")
        x_vert_grad_maps, vert_grad_weights = calc_pixel_grad_map(x, grad_weight, mode="vert")
        y_hori_grad_maps, hori_grad_weights = calc_pixel_grad_map(y, grad_weight, mode="hori")
        y_vert_grad_maps, vert_grad_weights = calc_pixel_grad_map(y, grad_weight, mode="vert")
        layer_cnt = len(x_hori_grad_maps)
        depth_grad_loss_values = []
        for layer_idx in range(layer_cnt):
            hori_grad_loss_map = (x_hori_grad_maps[layer_idx] - y_hori_grad_maps[layer_idx]).abs().mean(dim=-3, keepdim=True) \
                * hori_grad_weights[layer_idx] / (hori_grad_weights[layer_idx].sum() + 1e-8)
            vert_grad_loss_map = (x_vert_grad_maps[layer_idx] - y_vert_grad_maps[layer_idx]).abs().mean(dim=-3, keepdim=True) \
                * vert_grad_weights[layer_idx] / (vert_grad_weights[layer_idx].sum() + 1e-8)
            depth_grad_loss_values.append(hori_grad_loss_map.sum()[None] + vert_grad_loss_map.sum()[None])
        depth_grad_loss_value = torch.cat(depth_grad_loss_values).mean()
        detail_info["{}_grad".format(prefix)] = depth_grad_loss_value
    
    final_value = 0.0
    for k in detail_info.keys():
        final_value = final_value + detail_info[k]
    
    return final_value, detail_info


def samp_points_by_pixel(points: torch.Tensor, pixels: torch.Tensor, mode="bilinear", padding_mode="border"):
    """_summary_

    Args:
        points (torch.Tensor): Tensor of shape (b, c, h, w)
        pixels (torch.Tensor): Tensor of shape (b, k, 2)

    Returns:
        _type_: _description_
    """
    H, W = points.shape[-2:]
    assert points.ndim == 4 and pixels.ndim == 3
    pixels = pixels.unsqueeze(1) # b 1 k 2
    pixels = pixels + 0.5
    pixels[..., 0] = pixels[..., 0] / W
    pixels[..., 1] = pixels[..., 1] / H
    pixels = pixels * 2 - 1
    samped_points = torch.nn.functional.grid_sample(points, pixels, mode=mode, padding_mode=padding_mode, align_corners=False) # b c 1 k
    samped_points = samped_points.squeeze(dim=2).permute(0, 2, 1) # b k c
    return samped_points


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
    
    # dot_product = (vec_mat * cent_mat).sum(dim=-1)
    # norm_vec    = vec_mat.norm(dim=-1, p=2)
    # norm_cent   = cent_mat.norm(dim=-1, p=2)
    # cossim      = dot_product / (norm_vec * norm_cent + 1e-8)
    
    vec_mat  = vec_mat / (vec_mat.norm(dim=-1, keepdim=True) + 1e-8)
    cent_mat = cent_mat / (cent_mat.norm(dim=-1, keepdim=True) + 1e-8)
    cossim   = (vec_mat * cent_mat).sum(dim=-1)
    
    return cossim # b v v k


def matching_error_3D(points: torch.Tensor, 
                      extrs: torch.Tensor, 
                      matching_info: dict,  view_per_batch: int, filter_mask=None):
    """Compute 3d error between two group of (depths, cameras).

    Args:
        points (torch.Tensor): Points in world camera, tensor of shape (B*V*2, H, W, 3).
        extrs (torch.Tensor): Extrinsics, tensor of shape (B*V*2, 4, 4). world to camera.
        matching_info (dict): Containing tracks, a tensor of shape (B*V, 2, K, 2), and visibilities, a tensor of shape (B*V, 2, K).
        depth_is_z (bool): If the depth maps are z-depths. False if they are ray-depths. 
        enable_depth_weight (bool): If weight the matching error by pixel depth.
    """
    points = points.permute(0, 3, 1, 2) # b*v*2 3 h w
    camera_centers = calc_extr_inv(extrs)[..., :3, -1].unsqueeze(1) # b*v*2 1 3
    
    assert len(points) % view_per_batch == 0
    
    tracks = matching_info["matches"] # b*v 2 k 2
    vis = matching_info["visibilities"] # b*v 2 k
    mask = torch.all(vis, dim=1) # b*v k
    if filter_mask is not None:
        mask = mask * filter_mask
    
    points = samp_points_by_pixel(points, tracks.reshape(-1, *tracks.shape[-2:])) # b*v*2 k c
    points = points.reshape(-1, view_per_batch, *points.shape[-2:]) # b*v 2 k c
    cam_cents = camera_centers.reshape(-1, view_per_batch, *camera_centers.shape[1:]) # b*v 2 1 3
    unnormed_points = points.clone()
    
    cossim_map = compute_angle_loss(unnormed_points, cam_cents) # b*v 2 2 k
    cossim_map = torch.acos(BoundForArcCos.apply(cossim_map)) # b*v 2 2 k
    cossim_loss = rearrange(cossim_map, "b v vv k -> b k (v vv)") # b*v k 2*2
    cossim_loss = rearrange(cossim_loss[..., :-1], "b k (v vv) -> b k v vv", v=view_per_batch-1)[..., 1:] # b*v k 2-1 2
    cossim_loss = rearrange(cossim_loss, "b k v vv -> b k (v vv)").mean(dim=-1) # b*v k
    cossim_loss = (cossim_loss * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8) # b*v
    
    return cossim_loss.mean()


def matching_error_3D_normal(points: torch.Tensor,
                             matching_info: dict,  view_per_batch: int):
    """Compute 3d error between two group of (depths, cameras).

    Args:
        points (torch.Tensor): Points in world camera, tensor of shape (B*V*2, H, W, 3).
        extrs (torch.Tensor): Extrinsics, tensor of shape (B*V*2, 4, 4). world to camera.
        matching_info (dict): Containing tracks, a tensor of shape (B*V, 2, K, 2), and visibilities, a tensor of shape (B*V, 2, K).
        depth_is_z (bool): If the depth maps are z-depths. False if they are ray-depths. 
        enable_depth_weight (bool): If weight the matching error by pixel depth.
    """
    points = points.permute(0, 3, 1, 2) # b 3 h w
    norms = get_norm_map(points) # b 3 h w    
    
    assert len(points) % view_per_batch == 0
    tracks = matching_info["matches"] # b*v 2 k 2
    norms = samp_points_by_pixel(norms, tracks.reshape(-1, *tracks.shape[-2:]), mode="nearest") # bv k c
    norms = norms.reshape(-1, view_per_batch, *norms.shape[-2:]) # b v k c
    norm_loss = (1 - (norms[..., 0, :, :] * norms[..., 1, :, :]).sum(dim=-1).mean()) / 2
    
    return norm_loss


def matching_error_3D_dist(points: torch.Tensor, 
                           matching_info: dict,  view_per_batch: int, normalize_factor: torch.Tensor=None, with_sq=False):
    """Compute 3d error between two group of (depths, cameras).

    Args:
        points (torch.Tensor): Points in world camera, tensor of shape (B*V*2, H, W, 3).
        extrs (torch.Tensor): Extrinsics, tensor of shape (B*V*2, 4, 4). world to camera.
        matching_info (dict): Containing tracks, a tensor of shape (B*V, 2, K, 2), and visibilities, a tensor of shape (B*V, 2, K).
        depth_is_z (bool): If the depth maps are z-depths. False if they are ray-depths. 
        enable_depth_weight (bool): If weight the matching error by pixel depth.
    """
    points = points.permute(0, 3, 1, 2) # b 3 h w
    assert len(points) % view_per_batch == 0
    
    tracks = matching_info["matches"] # b*v 2 k 2
    vis = matching_info["visibilities"] # b*v 2 k
    mask = torch.all(vis, dim=1) # b*v k
    points = samp_points_by_pixel(points, tracks.reshape(-1, *tracks.shape[-2:])) # bv k c
    points = points.reshape(-1, view_per_batch, *points.shape[-2:]) # bv 2 k c
    points_offset = (points[..., 0, :, :] - points[..., 1, :, :]).norm(dim=-1) # bv k
    self_filtered_mask = None
    if normalize_factor is not None:
        # normalize_factor is a tensor of shape B
        v = points_offset.shape[0] // normalize_factor.shape[0]
        k = points_offset.shape[-1]
        normalize_factor = normalize_factor.reshape(-1, 1, 1).expand(-1, v, k).reshape(-1, k) + 1e-4 # bv k
        self_filtered_mask = ((points_offset / normalize_factor) < 2e-2) # bv k
        mask = mask * self_filtered_mask.detach()
    
    if not with_sq:
        dist_loss = points_offset # bv k
    else:
        dist_loss = points_offset + points_offset.square() # bv k
    dist_loss = ((dist_loss * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)).mean()
    
    return dist_loss.mean(), self_filtered_mask


class LossFunction(torch.nn.Module):

    def __init__(self, cfg, matching_model):
        super().__init__()
        self.cfg = cfg
        self.matching_model = matching_model
        self.enable_conf    = cfg.enable_conf
        self.conf_alpha     = cfg.conf_alpha
        self.enable_slide   = cfg.enable_slide
        self.eye3           = torch.eye(3, device=DEVICE_NAME)
        self.enable_alternate_geometric_opt = cfg.enable_alternate_geometric_opt
        self.opt_flag       = False
        self.norm_depth     = cfg.depth_shape.get("norm_depth", True)
        self.dl_detach_depth = cfg.depth_shape.get("detach_depth", False)
        self.local_align_res = 4096
        self.point_criteria  = torch.nn.L1Loss(reduction='none')
        self.full_weak_exclusive = cfg.get("full_weak_exclusive", False)
        self.norm_dist_loss = cfg.dist.get("norm_dist", False)
        self.norm_dist_ps_loss = cfg.dist.get("norm_dist_ps", False)
        self.filter_match_by_pred = cfg.get("filter_match_by_pred", False)
    
    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = torch.nn.functional.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def normalize_pred(self, pred, gt):
        local_points = rearrange(pred['local_points'], "b v c h w -> b v h w c")
        camera_poses = pred['camera_extrs']
        B, N, H, W, _ = local_points.shape
        masks = (gt['input_depth_masks'] > 1e-2)[..., 0, :, :] # b v h w

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]

        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)

        pred['local_points'] = rearrange(local_points, "b v h w c -> b v c h w")
        pred['camera_poses'] = camera_poses_normalized

        return pred

    def point_loss(self, output_dict, data):

        pred_local_pts = rearrange(output_dict['local_points'], "b v c h w -> b v h w c")
        gt_local_pts = data['input_points_local'] # b v h w 3
        valid_masks = data['input_depth_masks'] > 1e-2 # b v 1 h w
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape
        valid_masks = valid_masks.reshape(B, N, H, W)
        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)
        # Compute scales to align prediction to GT.
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), 
                                              valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), 
                                            valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), 
                                                 valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]
            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        ## Local point loss
        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts
        local_pts_loss = self.point_criteria(aligned_local_pts[valid_masks].float(), 
                                             gt_local_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]
        final_loss += local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()
        
        ## Global Point Loss, compare the points globally in the coordinate of the first camera. 
        gt_pts = data['input_points']
        pred_global_pts = localpoint2global(pred_local_pts, output_dict["camera_extrs"])
        # pred_global_pts = localpoint2global(pred_local_pts.detach(), output_dict["camera_extrs"])
        pred_trans_to_firstF = torch.cat([output_dict['camera_extrs'][:, :1] for _ in range(N)], dim=1) # b v 4 4
        pred_global_pts = mat_mul_vec_by_conv(pad_one(pred_global_pts, dim=-1), pred_trans_to_firstF)[..., :3] # b v h w 3
        pred_global_pts = pred_global_pts * S_opt_local.view(B, 1, 1, 1, 1) #.detach()
        global_pts_loss = self.point_criteria(pred_global_pts[valid_masks].float(), 
                                              gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]
        final_loss += global_pts_loss.mean()
        details['global_pts_loss'] = global_pts_loss.mean()

        return final_loss, details, S_opt_local
    
    def absolute_point_loss(self, output_dict, data, enable_conf=True):
        pred_local_pts = rearrange(output_dict['local_points'], "b v c h w -> b v h w c")
        pred_global_pts = localpoint2global(pred_local_pts, output_dict["camera_extrs"])
        pred_global_pts = rearrange(pred_local_pts, "b v h w c -> b v c h w")
        gt_global_pts = rearrange(data['input_points'], "b v h w c -> b v c h w")
        conf_map = output_dict["depths_conf"] if enable_conf else None
        valid_map = torch.ones_like(output_dict["depths_conf"]) if data["input_depth_masks"] is None else data["input_depth_masks"]
        return depth_shape_loss(pred_global_pts, gt_global_pts, 
                                conf_map=conf_map, valid_map=valid_map, alpha=self.conf_alpha, norm_depth=False, calc_grad_loss=False, prefix="point")
    
    def ds_loss(self, output_dict, data, enable_conf=False, lower=None):
        depths = output_dict["depths"]
        if self.dl_detach_depth:
            depths = depths.detach()
        input_depths = data["input_depths"].detach()
        input_masks  = torch.ones_like(output_dict["depths_conf"]) if data["input_depth_masks"] is None else data["input_depth_masks"]
        ds_loss = depth_shape_loss(depths, input_depths, 
            conf_map=output_dict["depths_conf"] if enable_conf else None, 
            valid_map=input_masks, 
            alpha=self.conf_alpha, norm_depth=self.norm_depth, conf4grad=self.cfg.depth_shape.get("conf4grad", True))
        return ds_loss
    
    def scale_points(self, output_dict, data, S_opt_local=None):
        points_src = output_dict["local_points_detach_xy"]
        extrs_src  = output_dict["camera_extrs"] if output_dict.get("camera_extrs", None) is not None \
            else data["input_viewmats"]
        B = points_src.shape[0]
        if S_opt_local is not None:
            points_src = points_src * S_opt_local.view(B, 1, 1, 1, 1)
            extrs_src_tmp = extrs_src.clone()
            extrs_src_tmp[..., :3, 3] = extrs_src[..., :3, 3] * S_opt_local.view(B, 1, 1)
            extrs_src = extrs_src_tmp.clone()
        output_dict["local_points_detach_xy_scaled"] = points_src.clone()
        output_dict["camera_extrs_scaled"] = extrs_src.clone()
    
    def geometric_loss(self, output_dict, data, detach_camera=False, detach_depth=False, use_scaled_points=False, with_sq=False):
        B, _, _, H, W = output_dict["depths"].shape
        
        points_src = output_dict["local_points_detach_xy"] if not use_scaled_points else output_dict["local_points_detach_xy_scaled"]
        if output_dict.get("camera_extrs", None) is None:
            extrs_src = data["input_viewmats"]
        else:
            extrs_src = output_dict["camera_extrs"] if not use_scaled_points else output_dict["camera_extrs_scaled"]
            extrs_src = GradBalancerForCamera.apply(extrs_src)
        if detach_depth:
            points_src = points_src.detach()
        if detach_camera:
            extrs_src = extrs_src.detach()
        normalize_factor = points_src.norm(dim=-3).reshape(B, -1).mean(dim=-1) # B
        points_src = localpoint2global(rearrange(points_src, "b v c h w -> b v h w c"), extrs_src) # b v h w 3

        if self.norm_dist_ps_loss:
            points_src = points_src / (normalize_factor.view(B, 1, 1, 1, 1) + 1e-12)
            extrs_src_temp = extrs_src.clone()
            extrs_src_temp[..., :3, 3] = extrs_src[..., :3, 3] / (normalize_factor.view(B, 1, 1) + 1e-12)
            extrs_src = extrs_src_temp.clone()
        
        tracks0 = data["input_tracks0"] # B P K 2
        tracks1 = data["input_tracks1"] # B P K 2
        visibs0 = data["input_visibs0"] # B P K
        visibs1 = data["input_visibs1"] # B P K
        matching_info = dict(
            matches=torch.stack([rearrange(tracks0, "b p k c -> (b p) k c"), rearrange(tracks1, "b p k c -> (b p) k c")], dim=1), # (b p) 2 k c
            visibilities=torch.stack([rearrange(visibs0, "b p k -> (b p) k"), rearrange(visibs1, "b p k -> (b p) k")], dim=1), # (b p) 2 k
            matched_views0=data["input_tracks_view_id0"], 
            matched_views1=data["input_tracks_view_id1"], 
        )
        
        paired_points, paired_extrs = [], []
        for points, extrs, pairs_x, pairs_y in zip(
            points_src, extrs_src, data["input_tracks_view_id0"], data["input_tracks_view_id1"]):
            paired_points.append(torch.stack([points[pairs_x], points[pairs_y]], dim=1)) # <- P 2 H W 3
            paired_extrs.append(torch.stack([extrs[pairs_x], extrs[pairs_y]], dim=1))    # <- P 2 4 4
        paired_points = torch.stack(paired_points).reshape(-1, 2, H, W, 3)
        paired_extrs = torch.stack(paired_extrs).reshape(-1, 2, 4, 4)
        
        matching_error_dist, filter_mask = matching_error_3D_dist(points=paired_points.reshape(-1, H, W, 3), 
                                                     matching_info=matching_info, 
                                                     view_per_batch=2, 
                                                     normalize_factor=normalize_factor if self.filter_match_by_pred else None, 
                                                     with_sq=with_sq)
        matching_error = matching_error_3D(points=paired_points.reshape(-1, H, W, 3), 
                                           extrs=paired_extrs.reshape(-1, 4, 4), 
                                           matching_info=matching_info, 
                                           view_per_batch=2, 
                                           filter_mask=filter_mask)
        matching_error_norm = matching_error_3D_normal(points=paired_points.reshape(-1, H, W, 3), 
                                                       matching_info=matching_info, 
                                                       view_per_batch=2)
        
        return matching_error, matching_error_norm, matching_error_dist, [matching_info, ]
    
    
    def relative_camera_loss(self, output_dict: dict[str, torch.Tensor], data: dict[str, torch.Tensor], S_opt_local: torch.Tensor):
        def huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float):
            mask = ((x - y).abs() < delta).to(dtype=x.dtype)
            return (0.5 * (x - y).square() * mask + delta * ((x - y).abs() - 0.5 * delta) * (1 - mask)).mean()
        
        if S_opt_local is not None:
            extrs_gt = data["input_viewmats"]      # b v 4 4
            extrs_pr = output_dict["camera_extrs"].clone() # b v 4 4
            extrs_pr[..., :3, 3] = output_dict["camera_extrs"][..., :3, 3] * S_opt_local[..., None, None]
        else:
            extrs_gt = normalize_trans(data["input_viewmats"], detach_norm_factor=True)      # b v 4 4
            extrs_pr = normalize_trans(output_dict["camera_extrs"], detach_norm_factor=False) # b v 4 4
        B, V = extrs_gt.shape[:2]
        
        w2c_gt_exp = extrs_gt.unsqueeze(1) # b 1 v 4 4
        w2c_pr_exp = extrs_pr.unsqueeze(1) # b 1 v 4 4
        c2w_gt_exp = calc_extr_inv(extrs_gt).unsqueeze(2) # b v 1 4 4
        c2w_pr_exp = calc_extr_inv(extrs_pr).unsqueeze(2) # b v 1 4 4
        
        rel_gt_pose = w2c_gt_exp @ c2w_gt_exp # b v v 4 4
        rel_pr_pose = w2c_pr_exp @ c2w_pr_exp # b v v 4 4
        rel_gt_pose = rel_gt_pose.reshape(B, -1, 4, 4)[:, 1:].reshape(B, V-1, V+1, 4, 4)[:, :, :-1].reshape(B, -1, 4, 4) # b (v-1)v 4 4
        rel_pr_pose = rel_pr_pose.reshape(B, -1, 4, 4)[:, 1:].reshape(B, V-1, V+1, 4, 4)[:, :, :-1].reshape(B, -1, 4, 4) # b (v-1)v 4 4
        
        t_rel_gt = rel_gt_pose[..., :3, 3]  # b (v-1)v 3
        r_rel_gt = rel_gt_pose[..., :3, :3] # b (v-1)v 3 3
        t_rel_pr = rel_pr_pose[..., :3, 3]  # b (v-1)v 3
        r_rel_pr = rel_pr_pose[..., :3, :3] # b (v-1)v 3 3
        
        rel_rot_inv_mul = (r_rel_pr.transpose(-1, -2) @ r_rel_gt) # b (v-1)v 3 3
        trace = rel_rot_inv_mul[..., 0, 0] + rel_rot_inv_mul[..., 1, 1] + rel_rot_inv_mul[..., 2, 2] # b (v-1)v
        L_r = torch.acos(BoundForArcCos.apply((trace - 1) / 2)).mean()
        L_t = huber_loss(t_rel_pr, t_rel_gt, delta=0.1)
        
        return L_r + 100 * L_t, L_r, L_t
    
    def absolute_camera_loss(self, output_dict: dict[str, torch.Tensor], data: dict[str, torch.Tensor]):
        extrs_gt = data["input_viewmats"] # b v 4 4
        extrs_pr = output_dict["camera_extrs"] # b v 4 4
        rot_gt = extrs_gt[..., :3, :3] # b v 3 3
        rot_pr = extrs_pr[..., :3, :3] # b v 3 3
        trans_gt = extrs_gt[..., :3, 3:] # b v 3 3
        trans_pr = extrs_pr[..., :3, 3:] # b v 3 3
        L_r = l1_loss(rot_gt, rot_pr)
        L_t = l1_loss(trans_gt, trans_pr)
        return L_r + L_t, L_r, L_t
    
    def intr_loss(self, output_dict, data):
        ## Constrain point maps with intrinsics.
        pr_local_points = output_dict["local_points_detach_z"].clone() # b v 3 h w
        _, _, _, H, W = pr_local_points.shape
        pr_ray_dirs = pr_local_points / (pr_local_points.norm(dim=-3, keepdim=True) + 1e-12) # b v 3 h w
        gt_ray_dirs = intr2ray(data["input_Ks"], H, W) # b v 3 h w
        gt_ray_dirs = gt_ray_dirs / (gt_ray_dirs.norm(dim=-3, keepdim=True) + 1e-12) # b v 3 h w
        ray_dirs_dot = (1 - (pr_ray_dirs * gt_ray_dirs).sum(dim=-3).mean()) / 2
        return ray_dirs_dot
    
    
    def normalize_loss(self, output_dict, data):
        ## Access local points
        pr_local_points = rearrange(output_dict["local_points"], "b v c h w -> b v h w c") # b v h w c
        # pr_local_points = rearrange(output_dict["local_points_detach_xy"], "b v c h w -> b v h w c") # b v h w c
        # pr_local_points = rearrange(output_dict["local_points_detach_z"], "b v c h w -> b v h w c") # b v h w c
        B, N = pr_local_points.shape[:2]
        ## Transform local points to global points
        # pr_global_points = localpoint2global(pr_local_points, output_dict["camera_extrs"]) # b v h w c
        pr_global_points = pr_local_points # b v h w c
        # ## Transform global points to the coordinate of first frame
        # pred_trans_to_firstF = torch.cat([output_dict['camera_extrs'][:, :1] for _ in range(N)], dim=1) # b v 4 4
        # pr_global_points = mat_mul_vec_by_conv(pad_one(pr_global_points, dim=-1), pred_trans_to_firstF)[..., :3] # b v h w 3
        ## Calculate the overall scale.
        valid_mask = data["input_depth_masks"].reshape(B, -1)
        pr_global_points_dist = pr_global_points.norm(dim=-1).reshape(B, -1)
        point_scale = (pr_global_points_dist * 
                       valid_mask / (valid_mask.sum(dim=-1, keepdim=True) + 1e-8)).sum(dim=-1).mean()
        # point_scale = pr_global_points_dist.mean(dim=-1)
        return ((point_scale - 1).abs().mean() + 
                (point_scale + 1 / (point_scale + 1e-8) - 2).mean()
                ) * 0.5
        # return (point_scale - 1).abs().mean()
    
    def forward(self, output_dict, data):
        ### Compute loss weight
        w_ds        = self.cfg.depth_shape.w if self.cfg.get("depth_shape", None) is not None else 0.0
        w_geometric = self.cfg.geometric.w if self.cfg.get("geometric", None) is not None else 0.0
        w_normal    = self.cfg.normal.w if self.cfg.get("normal", None) is not None else 0.0
        w_dist      = self.cfg.dist.w if self.cfg.get("dist", None) is not None else 0.0
        w_intr      = self.cfg.camera_intr.w if self.cfg.get("camera_intr", None) is not None else 0.0
        w_extr      = self.cfg.camera_extr.w if self.cfg.get("camera_extr", None) is not None else 0.0
        w_point     = self.cfg.point.w if self.cfg.get("point", None) is not None else 0.0
        w_normalize = self.cfg.normalize.w if self.cfg.get("normalize", None) is not None else 0.0
        
        # output_dict = self.normalize_pred(output_dict, data)
        # Transform camera prediction into the first frame
        # output_dict["camera_extrs"] = output_dict["camera_extrs"] @ calc_extr_inv(output_dict["camera_extrs"][:, :1])
        
        ### Compute loss value
        ## Depth shape loss
        ds_loss_value, depth_loss_details = self.ds_loss(output_dict, data, enable_conf=self.enable_conf, lower=1e-3)
        
        ## Point loss, disabled for self-sup learning
        if data.get("input_points_local", None) is not None:
            if not self.cfg.point.get("is_absolute", False):
                point_loss, point_loss_details, S_opt_local = self.point_loss(output_dict, data)
            else:
                point_loss, point_loss_details = self.absolute_point_loss(output_dict, data, enable_conf=self.enable_conf)
            if self.full_weak_exclusive:
                w_geometric = 0.0
                w_dist = 0.0
            loss_scale = 1.0 # supervised data
            with_sq = True
        else:
            point_loss, S_opt_local = torch.tensor(0.0), None
            w_extr = 0.0
            w_point = 0.0
            w_normalize = 0.0
            loss_scale = 0.1 # unsupervised data
            loss_scale = 1.0 # unsupervised data
            with_sq = False
        
        ## Intrinsic loss
        intr_loss_value = self.intr_loss(output_dict, data)
        
        ## Extrinsic loss, disabled for self-sup learning
        if not self.cfg.camera_extr.get("is_absolute", False):
            extr_loss_value, extr_r_loss_value, extr_t_loss_value = self.relative_camera_loss(output_dict, data, S_opt_local)
        else:
            extr_loss_value, extr_r_loss_value, extr_t_loss_value = self.absolute_camera_loss(output_dict, data)
        
        ## Normalization loss to regulate the value range of depth maps and camera coordinate.
        normalize_loss_value = self.normalize_loss(output_dict, data)
        
        ## Geometric matching loss based on triagonal minimization
        if self.enable_alternate_geometric_opt:
            self.opt_flag = not self.opt_flag
        detach_camera = self.enable_alternate_geometric_opt and not self.opt_flag or self.cfg.geometric.get("detach_camera", False)
        detach_depth  = self.enable_alternate_geometric_opt and self.opt_flag
        geometric_loss_value, normal_loss_value, dist_loss_value, matching_info = \
            self.geometric_loss(output_dict, data, detach_camera=detach_camera, detach_depth=detach_depth, use_scaled_points=False, with_sq=with_sq)
        if self.norm_dist_loss:
            self.scale_points(output_dict, data, S_opt_local)
            _, _, dist_loss_value, _ = \
                self.geometric_loss(output_dict, data, detach_camera=detach_camera, detach_depth=detach_depth, use_scaled_points=True, with_sq=with_sq)
        
        total_loss = (w_ds * ds_loss_value + 
                      w_geometric * geometric_loss_value + 
                      w_normal * normal_loss_value + 
                      w_dist * dist_loss_value + 
                      w_point * point_loss + 
                      w_intr * intr_loss_value + 
                      w_extr * extr_loss_value + 
                      w_normalize * normalize_loss_value) * loss_scale
        
        ### Return loss information
        loss_dict = {
            "total":         total_loss, 
            "match_cosine":  geometric_loss_value, 
            "match_normal":  normal_loss_value, 
            "match_dist":    dist_loss_value, 
            "intr_fov":      intr_loss_value, 
            "extr_rel":      extr_loss_value, 
            "extr_r":        extr_r_loss_value, 
            "extr_t":        extr_t_loss_value, 
            "normalize":     normalize_loss_value, 
            "matching_info": matching_info, 
        }
        loss_dict.update(depth_loss_details)
        if data.get("input_points_local", None) is not None:
            if not self.cfg.point.get("is_absolute", False):
                loss_dict.update({
                    "point_local":  point_loss_details["local_pts_loss"], 
                    "point_global": point_loss_details["global_pts_loss"], 
                })
            else:
                loss_dict.update(point_loss_details)
        return loss_dict