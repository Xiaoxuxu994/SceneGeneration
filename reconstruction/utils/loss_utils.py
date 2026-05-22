import torch
import torch.nn.functional as F
from einops import rearrange

from .platform import DEVICE_NAME
from utils.common import WeightedMAD, calc_extr_inv, intr2ray, localpoint2global, mat_mul_vec_by_conv, pad_one, normalize_trans  #, get_norm_map
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


def calc_pixel_grad_map(x: torch.Tensor, w: torch.Tensor, eps=1e-6) -> torch.Tensor:
    assert x.shape[-3] == 1, "The input of pixel gradient calculation should be a depth map."
    H, W = x.shape[-2:]
    step = 1
    maps, weights = [], []
    while step < H and step < W:
        _x = x[..., ::step, ::step]
        hori_grad = _x[..., :-1, :-1] - _x[..., :-1, 1:] # ..., 1, H-1, W-1
        vert_grad = _x[..., :-1, :-1] - _x[..., 1:, :-1] # ..., 1, H-1, W-1
        norm_grad = (hori_grad.square() + vert_grad.square() + eps).sqrt() + eps # ..., 1, H-1, W-1
        maps.append(torch.cat([hori_grad / norm_grad, vert_grad / norm_grad], dim=-3)) # Concatinate by channel.
        _w = w[..., ::step, ::step]
        total_grad_w = _w[..., :-1, :-1] * _w[..., :-1, 1:] * _w[..., 1:, :-1]
        weights.append(total_grad_w)
        step *= 2
    return maps, weights


def depth_shape_loss(x: torch.Tensor, y: torch.Tensor, 
                     conf_map: torch.Tensor=None, alpha=0.05, calc_grad_loss=True, norm_depth=True) -> torch.Tensor:
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
    
    if norm_depth:
        weighted_mad = WeightedMAD()
        x = weighted_mad(x.reshape(*x.shape[:-3], -1), 
                        conf_map.reshape(*x.shape[:-3], -1).detach() if conf_map is not None else None).reshape(*x.shape)
        y = weighted_mad(y.reshape(*y.shape[:-3], -1), 
                        conf_map.reshape(*x.shape[:-3], -1).detach() if conf_map is not None else None).reshape(*y.shape)
    scale_invariant_loss_map = (x - y).abs().mean(dim=-3, keepdim=True)
    if conf_map is None:
        scale_invariant_loss_value = scale_invariant_loss_map.mean()
    else:
        scale_invariant_loss_value = (conf_map * scale_invariant_loss_map - alpha * torch.log(conf_map)).mean()
    
    if calc_grad_loss:
        x_grad_maps, grad_weights = calc_pixel_grad_map(x, conf_map)
        y_grad_maps, grad_weights = calc_pixel_grad_map(y, conf_map)
        depth_grad_loss_maps = [(x_grad_map - y_grad_map).abs().mean(dim=-3, keepdim=True) * weight / (weight.sum() + 1e-8)
                                for x_grad_map, y_grad_map, weight in zip(x_grad_maps, y_grad_maps, grad_weights)]
        depth_grad_loss_value = torch.cat([loss_map.sum()[None] for loss_map in depth_grad_loss_maps]).mean()
        depth_shape_loss_value = (scale_invariant_loss_value + depth_grad_loss_value) / 2
    else:
        depth_shape_loss_value = scale_invariant_loss_value
    
    return depth_shape_loss_value #, s, t


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
    
    dot_product = (vec_mat * cent_mat).sum(dim=-1)
    norm_vec    = vec_mat.norm(dim=-1, p=2)
    norm_cent   = cent_mat.norm(dim=-1, p=2)
    cossim      = dot_product / (norm_vec * norm_cent + 1e-8)
    
    return cossim # b v v k


def matching_error_3D(points: torch.Tensor, 
                      extrs: torch.Tensor, 
                      matching_info: dict,  view_per_batch: int):
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
    ## Currently we filter invalid pairs in the dataloader. So these two lines can be commented out. 
    valid_batch_mask = mask.sum(dim=-1) > 16 # b
    cossim_loss = (cossim_loss * valid_batch_mask).sum() / (valid_batch_mask.sum() + 1e-8) # tensor
    
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
    # points = points.permute(0, 3, 1, 2) # b 3 h w
    # norms = get_norm_map(points) # b 3 h w    
    
    # assert len(points) % view_per_batch == 0
    # tracks = matching_info["matches"] # b*v 2 k 2
    # norms = samp_points_by_pixel(norms, tracks.reshape(-1, *tracks.shape[-2:]), mode="nearest") # bv k c
    # norms = norms.reshape(-1, view_per_batch, *norms.shape[-2:]) # b v k c
    # norm_loss = (1 - (norms[..., 0, :, :] * norms[..., 1, :, :]).sum(dim=-1).mean()) / 2
    
    # return norm_loss
    return points.new_zeros(())


def matching_error_3D_dist(points: torch.Tensor, 
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
    assert len(points) % view_per_batch == 0
    
    tracks = matching_info["matches"] # b*v 2 k 2
    vis = matching_info["visibilities"] # b*v 2 k
    mask = torch.all(vis, dim=1) # b*v k
    points = samp_points_by_pixel(points, tracks.reshape(-1, *tracks.shape[-2:])) # bv k c
    points = points.reshape(-1, view_per_batch, *points.shape[-2:]) # bv 2 k c
    # dist_loss = (points[..., 0, :, :] - points[..., 1, :, :]).square().sum(dim=-1) # bv k
    points_diff = points[..., 0, :, :] - points[..., 1, :, :]
    # dist_loss = points_diff.abs().sum(dim=-1) + points_diff.square().sum(dim=-1) # bv k
    dist_loss = points_diff.abs().sum(dim=-1) # bv k
    dist_loss = ((dist_loss * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)).mean()
    
    return dist_loss.mean()


# image metrics, use reduce=True for training loss, and reduce=False for per-image eval logging
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.image.lpips import _NoTrainLpips

LPIPS_NET = _NoTrainLpips()

def psnr(x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    val = F.mse_loss(x, y, reduction='none').mean((-3, -2, -1)).log10_().mul_(-10)
    return val.mean() if reduce else val

def ssim(x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    val = structural_similarity_index_measure(
        x.flatten(0, 1), y.flatten(0, 1), data_range=1, reduction='none')
    return val.mean() if reduce else val.view(x.shape[:2])

@torch.autocast(torch.accelerator.current_accelerator().type, enabled=False)
def lpips(x: torch.Tensor, y: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    if next(LPIPS_NET.parameters()).device != x.device: LPIPS_NET.to(x.device)
    val = LPIPS_NET.forward(x.flatten(0, 1).float(), y.flatten(0, 1).float(), normalize=True)
    return val.mean() if reduce else val.view(x.shape[:2])


@torch.autocast(torch.accelerator.current_accelerator().type, enabled=False)
def compute_render_metrics(render: torch.Tensor, images_gt: torch.Tensor) -> dict[str, torch.Tensor]:
    render_rgb = render[:, :, :3].detach().clamp(0, 1)
    return {
        'psnr': psnr(render_rgb, images_gt),
        'ssim': ssim(render_rgb, images_gt),
        'lpips': lpips(render_rgb, images_gt),
    }



def translation_error_se3(local_points: torch.Tensor,
                          extrs_w2c: torch.Tensor,
                          matching_info: dict,
                          view_per_batch: int = 2,
                          conf_map: torch.Tensor | None = None,
                          eps: float = 1e-6,
                          min_points: int = 16,
                          clip_err: float = 5.0,
                          robust_delta: float = 1.0):

    assert view_per_batch == 2

    # local_points -> (B*V, 3, H, W)
    lp = local_points.permute(0, 3, 1, 2).contiguous()
    BV, _, H, W = lp.shape
    assert BV % 2 == 0
    B = BV // 2

    tracks = matching_info["matches"]        # (B,2,K,2)
    vis    = matching_info["visibilities"]   # (B,2,K)
    mask   = torch.all(vis, dim=1)           # (B,K)

    # ---- sample Xc0, Xc1 from local pointmaps ----
    pts = samp_points_by_pixel(lp, tracks.reshape(-1, *tracks.shape[-2:]))  # (B*2,K,3)
    pts = pts.reshape(B, 2, *pts.shape[-2:])                                 # (B,2,K,3)
    X0, X1 = pts[:, 0], pts[:, 1]                                            # (B,K,3)

    # ---- compute relative pose T10 = w2c1 * inv(w2c0) ----
    extrs = extrs_w2c.reshape(B, 2, 4, 4)
    T0 = extrs[:, 0]  # (B,4,4) w2c0
    T1 = extrs[:, 1]  # (B,4,4) w2c1
    T0_inv = calc_extr_inv(T0)  # (B,4,4) c2w0

    T10 = T1 @ T0_inv           # (B,4,4) cam0 -> cam1
    R10 = T10[:, :3, :3]        # (B,3,3)
    t10 = T10[:, :3, 3]         # (B,3)

    mask = mask & (X0[..., 2] > 1e-4) & (X1[..., 2] > 1e-4) # Xc0, Xc1 本来 in camera coords -> just check z>0

    X1_hat = torch.einsum("bij,bkj->bki", R10, X0) + t10[:, None, :]  # (B,K,3)  predict X1_hat = R10*X0 + t10
    resid = X1_hat - X1                                               # (B,K,3)

    # ---- depth normalization ----
    scale = 0.5 * (X0.norm(dim=-1) + X1.norm(dim=-1)) + 1e-3          # (B,K)     # scale = mean depth magnitude
    resid_n = resid / scale[..., None]                                # (B,K,3)

    # per-point error smooth_l1_loss
    err = F.smooth_l1_loss(resid_n, torch.zeros_like(resid_n),
                           beta=robust_delta, reduction="none").sum(dim=-1)  # (B,K)

    if clip_err is not None:
        err = err.clamp(max=clip_err)

    if conf_map is not None:
        # conf_map: (B*V, H, W) -> sample to (B,2,K)
        conf_chw = conf_map.unsqueeze(1)  # (B*V,1,H,W)
        conf = samp_points_by_pixel(conf_chw, tracks.reshape(-1, *tracks.shape[-2:]))  # (B*2,K,1)
        conf = conf.squeeze(-1).reshape(B, 2, -1)                                      # (B,2,K)
        w = (conf[:, 0] * conf[:, 1]).clamp(0.0, 1.0).detach()                         # detach
    else:
        w = torch.ones_like(err)

    denom = (w * mask).sum(dim=-1).add(eps)
    loss_per_batch = (err * w * mask).sum(dim=-1) / denom  # (B,)

    valid_batch = mask.sum(dim=-1) > min_points
    loss = (loss_per_batch * valid_batch).sum() / (valid_batch.sum().add(eps))
    return loss


def project_uv(X_bk3: torch.Tensor, K_b33: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    X_bk3: (B,K,3) camera coords
    K_b33: (B,3,3)
    return: (B,K,2) pixel coords
    """
    x = torch.einsum("bij,bkj->bki", K_b33, X_bk3)  # (B,K,3)
    u = x[..., 0] / (x[..., 2] + eps)
    v = x[..., 1] / (x[..., 2] + eps)
    return torch.stack([u, v], dim=-1)

def sparse_rigid_ray_flow_loss_from_pairs(
    paired_local: torch.Tensor,   # (BP,2,H,W,3) camera coords XYZ
    paired_extrs: torch.Tensor,   # (BP,2,4,4)   w2c
    paired_rays: torch.Tensor,    # (BP,2,H,W,3) unit ray dirs
    matching_info: dict,          # matches/visibilities: (BP,2,K,2)/(BP,2,K)
    paired_conf: torch.Tensor | None = None,  # (BP,2,H,W) or None
    eps: float = 1e-6,
    min_points: int = 16,
    clip_cos: float = 0.999999,
):

    BP, _, H, W, _ = paired_local.shape
    tracks = matching_info["matches"]        # (BP,2,K,2)
    vis    = matching_info["visibilities"]   # (BP,2,K)
    mask   = torch.all(vis > 0, dim=1)       # (BP,K)

    uv0 = tracks[:, 0]  # (BP,K,2)
    uv1 = tracks[:, 1]  # (BP,K,2)

    # --- sample X0 from view0 local pointmap ---
    lp0 = paired_local[:, 0].permute(0, 3, 1, 2).contiguous()  # (BP,3,H,W)
    X0 = samp_points_by_pixel(lp0, uv0)                        # (BP,K,3)

    # --- relative pose T10 = w2c1 @ inv(w2c0) ---
    T0 = paired_extrs[:, 0].float()
    T1 = paired_extrs[:, 1].float()
    T10 = T1 @ calc_extr_inv(T0)
    R10 = T10[:, :3, :3]  # (BP,3,3)
    t10 = T10[:, :3, 3]   # (BP,3)

    # --- transform to cam1 ---
    X1_hat = torch.einsum("bij,bkj->bki", R10, X0.float()) + t10[:, None, :]  # (BP,K,3)

    # --- cheirality ---
    valid = mask & (X0[..., 2] > 1e-4) & (X1_hat[..., 2] > 1e-4)

    # predicted ray in cam1
    r_hat = X1_hat / (X1_hat.norm(dim=-1, keepdim=True) + eps)  # (BP,K,3)

    # observed ray from ray-map at uv1
    rays1 = paired_rays[:, 1].permute(0, 3, 1, 2).contiguous()   # (BP,3,H,W)
    r_obs = samp_points_by_pixel(rays1, uv1)                     # (BP,K,3)
    r_obs = r_obs / (r_obs.norm(dim=-1, keepdim=True) + eps)

    # cosine distance (avoid acos)
    cos = (r_hat * r_obs).sum(dim=-1).clamp(-clip_cos, clip_cos)  # (BP,K)
    per = (1.0 - cos)                                             # (BP,K)

    # conf weighting (optional)
    if paired_conf is not None:
        c0_map = paired_conf[:, 0].unsqueeze(1).contiguous()  # (BP,1,H,W)
        c1_map = paired_conf[:, 1].unsqueeze(1).contiguous()
        c0 = samp_points_by_pixel(c0_map, uv0).squeeze(-1)    # (BP,K)
        c1 = samp_points_by_pixel(c1_map, uv1).squeeze(-1)

        c0n = (c0 * 0.5).clamp(0.0, 1.0)
        c1n = (c1 * 0.5).clamp(0.0, 1.0)
        w = (c0n * c1n).clamp_min(1e-3).detach()
    else:
        w = torch.ones_like(per)

    w = w * valid.float()
    denom = w.sum(dim=-1).clamp_min(1.0)
    loss_per = (per * w).sum(dim=-1) / denom

    valid_batch = valid.sum(dim=-1) > min_points
    loss = (loss_per * valid_batch).sum() / (valid_batch.sum().clamp_min(1.0))
    return loss


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

        self.w_ds_def        = self.cfg.depth_shape.w if self.cfg.get("depth_shape", None) is not None else 0.0
        self.w_geometric_def = self.cfg.geometric.w if self.cfg.get("geometric", None) is not None else 0.0
        self.w_intr_def      = self.cfg.camera_intr.w if self.cfg.get("camera_intr", None) is not None else 0.0
        self.w_extr_def      = self.cfg.camera_extr.w if self.cfg.get("camera_extr", None) is not None else 0.0
        self.w_glbpoint_def  = self.cfg.point_glb.w if self.cfg.get("point_glb", None) is not None else 0.0
        self.w_locpoint_def  = self.cfg.point_loc.w if self.cfg.get("point_loc", None) is not None else 0.0
        self.w_normalize_def = self.cfg.normalize.w if self.cfg.get("normalize", None) is not None else 0.0
        self.w_normal_def    = self.cfg.normal.w if self.cfg.get("normal", None) is not None else 0.0
        self.w_dist_def      = self.cfg.dist.w if self.cfg.get("dist", None) is not None else 0.0
        self.w_translate_def = self.cfg.translate.w if self.cfg.get("translate", None) is not None else 0.0
        self.w_flow_def      = self.cfg.flow.w if self.cfg.get("flow", None) is not None else 0.0
    
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
    
    def point_loss(self, output_dict, data):
        def weighted_mean(x: torch.Tensor, w: torch.Tensor=None, dim=None, keepdim: bool=False, eps: float=1e-7) -> torch.Tensor:
            if w is None:
                return x.mean(dim=dim, keepdim=keepdim)
            else:
                w = w.to(x.dtype)
                return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)
        
        pred_local_pts = rearrange(output_dict['local_points'], "b v c h w -> b v h w c")
        gt_local_pts = data['input_points_local'] # b v h w 3
        valid_masks = data['input_depth_masks'] > 1e-2 # b v 1 h w
        details = dict()
        locpts_loss, glbpts_loss = 0.0, 0.0

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
        pred_trans_to_firstF = torch.cat([output_dict['camera_extrs'][:, :1] for _ in range(N)], dim=1) # b v 4 4
        pred_global_pts = mat_mul_vec_by_conv(pad_one(pred_global_pts, dim=-1), pred_trans_to_firstF)[..., :3] # b v h w 3
        pred_global_pts = pred_global_pts * S_opt_local.view(B, 1, 1, 1, 1)
        global_pts_loss = self.point_criteria(pred_global_pts[valid_masks].float(), 
                                              gt_pts[valid_masks].float()) * weights_[valid_masks].float()[..., None]
        # final_loss += global_pts_loss.mean()
        details['global_pts_loss'] = global_pts_loss.mean()

        return locpts_loss, glbpts_loss, details, S_opt_local
    
    def ds_loss(self, output_dict, data, enable_conf=False, lower=None):
        depths = output_dict["depths"]
        if self.dl_detach_depth:
            depths = depths.detach()
        input_depths = data["input_depths"].detach()
        input_masks  = torch.ones_like(output_dict["depths_conf"]) if data["input_depth_masks"] is None else data["input_depth_masks"]
        output_dict["depths_conf"] = output_dict["depths_conf"] * input_masks.clamp(min=lower)
        ds_loss = depth_shape_loss(depths, input_depths, 
            conf_map=output_dict["depths_conf"] if enable_conf else None, 
            alpha=self.conf_alpha, norm_depth=self.norm_depth)
        return ds_loss
    
    # def geometric_loss(self, output_dict, data, detach_camera=False, detach_depth=False):
    #     _, _, _, H, W = output_dict["depths"].shape
        
    #     points_src = output_dict["local_points_detach_xy"]
    #     extrs_src  = GradBalancerForCamera.apply(output_dict["camera_extrs"]) if output_dict.get("camera_extrs", None) is not None \
    #         else data["input_viewmats"]
    #     if detach_depth:
    #         points_src = points_src.detach()
    #     if detach_camera:
    #         extrs_src = extrs_src.detach()
    #     points_src = localpoint2global(rearrange(points_src, "b v c h w -> b v h w c"), extrs_src) # b v h w 3
        
    #     tracks0 = data["input_tracks0"] # B P K 2
    #     tracks1 = data["input_tracks1"] # B P K 2
    #     visibs0 = data["input_visibs0"] # B P K
    #     visibs1 = data["input_visibs1"] # B P K
    #     matching_info = dict(
    #         matches=torch.stack([rearrange(tracks0, "b p k c -> (b p) k c"), rearrange(tracks1, "b p k c -> (b p) k c")], dim=1), # (b p) 2 k c
    #         visibilities=torch.stack([rearrange(visibs0, "b p k -> (b p) k"), rearrange(visibs1, "b p k -> (b p) k")], dim=1), # (b p) 2 k
    #         matched_views0=data["input_tracks_view_id0"], 
    #         matched_views1=data["input_tracks_view_id1"], 
    #     )
        
    #     paired_points, paired_extrs = [], []
    #     for points, extrs, pairs_x, pairs_y in zip(
    #         points_src, extrs_src, data["input_tracks_view_id0"], data["input_tracks_view_id1"]):
    #         paired_points.append(torch.stack([points[pairs_x], points[pairs_y]], dim=1)) # <- P 2 H W 3
    #         paired_extrs.append(torch.stack([extrs[pairs_x], extrs[pairs_y]], dim=1))    # <- P 2 4 4
    #     paired_points = torch.stack(paired_points).reshape(-1, 2, H, W, 3)
    #     paired_extrs = torch.stack(paired_extrs).reshape(-1, 2, 4, 4)
        
    #     matching_error = matching_error_3D(points=paired_points.reshape(-1, H, W, 3), 
    #                                        extrs=paired_extrs.reshape(-1, 4, 4), 
    #                                        matching_info=matching_info, 
    #                                        view_per_batch=2)
    #     matching_error_norm = matching_error_3D_normal(points=paired_points.reshape(-1, H, W, 3), 
    #                                                    matching_info=matching_info, 
    #                                                    view_per_batch=2)
    #     matching_error_dist = matching_error_3D_dist(points=paired_points.reshape(-1, H, W, 3), 
    #                                                  matching_info=matching_info, 
    #                                                  view_per_batch=2)
        
    #     return matching_error, matching_error_norm, matching_error_dist, [matching_info, ]

    def geometric_loss(self, output_dict, data, detach_camera=False, detach_depth=False, w_translate=0., w_flow=0.):
        _, _, _, H, W = output_dict["depths"].shape
        points_src = output_dict["local_points_detach_xy"]
        extrs_src  = GradBalancerForCamera.apply(output_dict["camera_extrs"]) if output_dict.get("camera_extrs", None) is not None \
            else data["input_viewmats"]
        if detach_depth:
            points_src = points_src.detach()
        if detach_camera:
            extrs_src = extrs_src.detach()
        cam_rays = output_dict["cam_rays"]  # (B,V,3,H,W)
        cam_rays_hw3 = rearrange(cam_rays, "b v c h w -> b v h w c").contiguous()

        # local -> (B,V,H,W,3)
        local_pts_hw3 = rearrange(points_src, "b v c h w -> b v h w c").contiguous()

        # world points (B,V,H,W,3)
        world_pts = localpoint2global(local_pts_hw3, extrs_src)
        
        # -------- matching info: flatten (B,P,...) -> (B*P,...) --------
        tracks0 = data["input_tracks0"]  # (B,P,K,2)
        tracks1 = data["input_tracks1"]  # (B,P,K,2)
        visibs0 = data["input_visibs0"]  # (B,P,K)
        visibs1 = data["input_visibs1"]  # (B,P,K)
        view_id0 = data["input_tracks_view_id0"]  # (B,P)  long ideally
        view_id1 = data["input_tracks_view_id1"]  # (B,P)
        matching_info = dict(
            matches=torch.stack([rearrange(tracks0, "b p k c -> (b p) k c"), rearrange(tracks1, "b p k c -> (b p) k c")], dim=1),  # (BP,2,K,2)
            visibilities=torch.stack([rearrange(visibs0, "b p k -> (b p) k"), rearrange(visibs1, "b p k -> (b p) k")], dim=1),  # (BP,2,K)
            matched_views0=view_id0,
            matched_views1=view_id1,
        )

        conf_src = output_dict["depths_conf"].detach()
        if conf_src.dim() == 5:
            conf_src = conf_src.squeeze(2)  # (B,V,H,W)
        # now conf_src: (B,V,H,W)
        # else: keep None        

        paired_world = []
        paired_local = []
        paired_extrs = []
        paired_rays = []
        paired_conf  = []  

        # zip over batch dimension
        for wp_b, lp_b, extrs_b, rays_b, conf_b, pairs_x_b, pairs_y_b in zip(
            world_pts, local_pts_hw3, extrs_src, cam_rays_hw3,
            conf_src if conf_src is not None else [None] * world_pts.shape[0],
            view_id0, view_id1
        ):
            # wp_b: (V,H,W,3)
            # lp_b: (V,H,W,3)
            # extrs_b: (V,4,4)
            # Ks_b: (V,3,3)
            # pairs_x_b / pairs_y_b: (P,)  view indices

            # ---- cast indices to long + same device ----
            if not torch.is_tensor(pairs_x_b):
                pairs_x_b = torch.as_tensor(pairs_x_b)
            if not torch.is_tensor(pairs_y_b):
                pairs_y_b = torch.as_tensor(pairs_y_b)
            pairs_x_b = pairs_x_b.to(device=extrs_b.device, dtype=torch.long)
            pairs_y_b = pairs_y_b.to(device=extrs_b.device, dtype=torch.long)

            # ---- gather P pairs: output shapes ----
            # each: (P,2,H,W,3) or (P,2,4,4) or (P,2,3,3)
            paired_world.append(torch.stack([wp_b[pairs_x_b], wp_b[pairs_y_b]], dim=1))
            paired_local.append(torch.stack([lp_b[pairs_x_b], lp_b[pairs_y_b]], dim=1))
            paired_extrs.append(torch.stack([extrs_b[pairs_x_b], extrs_b[pairs_y_b]], dim=1))
            paired_rays.append(torch.stack([rays_b[pairs_x_b], rays_b[pairs_y_b]], dim=1))  # (P,2,H,W,3)

            if conf_b is not None:
                # conf_b: (V,H,W) -> (P,2,H,W)
                paired_conf.append(torch.stack([conf_b[pairs_x_b], conf_b[pairs_y_b]], dim=1))

        # stack across batch: (B,P,...) -> reshape to (BP,...)
        paired_world = torch.stack(paired_world).reshape(-1, 2, H, W, 3).contiguous()  # (BP,2,H,W,3)
        paired_local = torch.stack(paired_local).reshape(-1, 2, H, W, 3).contiguous()
        paired_extrs = torch.stack(paired_extrs).reshape(-1, 2, 4, 4).contiguous()     # (BP,2,4,4)
        paired_rays = torch.stack(paired_rays).reshape(-1, 2, H, W, 3).contiguous()
        if conf_src is not None:
            paired_conf = torch.stack(paired_conf).reshape(-1, 2, H, W).contiguous()   # (BP,2,H,W)
        else:
            paired_conf = None

        # ---------- 1) angle  and dist loss ----------
        matching_error = matching_error_3D(
            points=paired_world.reshape(-1, H, W, 3),
            extrs=paired_extrs.reshape(-1, 4, 4),
            matching_info=matching_info,
            view_per_batch=2
        )
        # matching_error_dist = matching_error_3D_dist(points=paired_world.reshape(-1, H, W, 3), 
        #                                              extrs=paired_extrs.reshape(-1, 4, 4), 
        #                                              matching_info=matching_info, 
        #                                              view_per_batch=2)        
        
        # matching_error_norm = matching_error_3D_normal(points=paired_world.reshape(-1, H, W, 3), 
        #                                                matching_info=matching_info, 
        #                                                view_per_batch=2)
        matching_error_norm = torch.tensor([0.0]).to(matching_error.device)
        matching_error_dist = matching_error_3D_dist(points=paired_world.reshape(-1, H, W, 3), 
                                                     matching_info=matching_info, 
                                                     view_per_batch=2) 
        

        # ---------- 2) SE3 translation consistency (3D domain) ----------
        if w_translate > 0:
            translation_error = translation_error_se3(
                local_points=paired_local.reshape(-1, H, W, 3),
                extrs_w2c=paired_extrs.reshape(-1, 4, 4),
                matching_info=matching_info,
                view_per_batch=2,
                conf_map=None,      # 这里先不加，跑稳后再加
                clip_err=5.0,
                robust_delta=1.0
            )
        else: translation_error = matching_error.new_zeros(())

        # ---------- 3) sparse rigid flow loss (image domain) ----------
        if w_flow > 0:
            ray_flow_loss = sparse_rigid_ray_flow_loss_from_pairs(
                paired_local=paired_local,         # (BP,2,H,W,3)
                paired_extrs=paired_extrs,         # (BP,2,4,4)
                paired_rays=paired_rays,           # (BP,2,H,W,3)
                matching_info=matching_info,       # (BP,2,K,2)/(BP,2,K)
                paired_conf=paired_conf,           # (BP,2,H,W) 
            )
        else: ray_flow_loss = matching_error.new_zeros(())

        return matching_error, matching_error_norm, matching_error_dist, translation_error, ray_flow_loss, [matching_info] 

    
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
        # L_t = huber_loss(t_rel_pr, t_rel_gt, delta=0.1)
        L_t = (t_rel_pr - t_rel_gt).abs().mean()
        
        return L_r + 100 * L_t, L_r, L_t
        
    
    def intr_loss(self, output_dict, data):
        ## Constrain point maps with intrinsics.
        pr_local_points = output_dict["local_points_detach_z"].clone() # b v 3 h w
        _, _, _, H, W = pr_local_points.shape
        pr_ray_dirs = pr_local_points / (pr_local_points.norm(dim=-3, keepdim=True) + 1e-12) # b v 3 h w
        gt_ray_dirs = intr2ray(data["input_Ks"], H, W) # b v 3 h w
        gt_ray_dirs = gt_ray_dirs / (gt_ray_dirs.norm(dim=-3, keepdim=True) + 1e-12) # b v 3 h w
        ray_dirs_dot = (1 - (pr_ray_dirs * gt_ray_dirs).sum(dim=-3).mean()) / 2
        return ray_dirs_dot
    
    
    def normalize_loss(self, output_dict):
        ## Access local points
        pr_local_points = rearrange(output_dict["local_points_detach_xy"], "b v c h w -> b v h w c") # b v h w c
        # pr_local_points = rearrange(output_dict["local_points_detach_z"], "b v c h w -> b v h w c") # b v h w c
        B, N = pr_local_points.shape[:2]
        ## Transform local points to global points
        pr_global_points = localpoint2global(pr_local_points, output_dict["camera_extrs"]) # b v h w c
        # ## Transform global points to the coordinate of first frame
        # pred_trans_to_firstF = torch.cat([output_dict['camera_extrs'][:, :1] for _ in range(N)], dim=1) # b v 4 4
        # pr_global_points = mat_mul_vec_by_conv(pad_one(pr_global_points, dim=-1), pred_trans_to_firstF)[..., :3] # b v h w 3
        ## Calculate the overall scale.
        point_scale = pr_global_points.norm(dim=-1).reshape(B, -1).mean(dim=-1) # B
        return (point_scale - 1).abs().mean()
    

    def render_loss(self, render: torch.Tensor, images_gt: torch.Tensor, depth_gt: torch.Tensor, w_render: dict) -> torch.Tensor:
        loss = render.new_zeros(())
        images, depth = render[:, :, :3], render[:, :, 3]
        HW = render.shape[-2:]
        if images_gt.shape[-2:] != HW:
            images_gt = F.interpolate(images_gt.flatten(0, 1), HW, mode='bilinear').unflatten(0, images_gt.shape[:2])
        if depth_gt.shape[-2:] != HW:
            depth_gt = F.interpolate(depth_gt, HW)
        if w_render.get(k := 'w_l1'): loss = loss + w_render[k] * F.l1_loss(images, images_gt)
        if w_render.get(k := 'w_mse'): loss = loss + w_render[k] * F.mse_loss(images, images_gt)
        if w_render.get(k := 'w_ssim'): loss = loss + w_render[k] * (1 - ssim(images, images_gt))
        if w_render.get(k := 'w_lpips'): loss = loss + w_render[k] * lpips(images, images_gt)
        if w_render.get(k := 'w_depth'): loss = loss + w_render[k] * F.l1_loss(depth, depth_gt)
        return loss


    def _set_stage_steps(self, step_begin=0, step_begin_stage1=0, step_begin_stage2=0, step_begin_stage3=0):
        # self.step_begin = step_begin
        # self.step_begin_stage1 = self.step_begin + step_begin_stage1
        # self.step_begin_stage2 = self.step_begin + step_begin_stage2
        # self.step_begin_stage3 = self.step_begin + step_begin_stage3
        self.step_begin_stage1 = step_begin_stage1
        self.step_begin_stage2 = step_begin_stage2
        self.step_begin_stage3 = step_begin_stage3        
    
    def forward(self, output_dict, data, step):
        ### Compute loss weight
        w_ds        = self.w_ds_def
        w_intr      = self.w_intr_def
        # w_extr      = self.w_extr_def
        # w_point     = self.w_point_def
        w_normalize = self.w_normalize_def
        w_render    = self.cfg.get("render", None)
        
        ### Compute loss value
        ## Depth shape loss
        ds_loss_value = self.ds_loss(output_dict, data, enable_conf=self.enable_conf, lower=1e-4)
        ## Geometric matching loss based on triagonal minimization
        detach_camera = self.enable_alternate_geometric_opt and not self.opt_flag or self.cfg.geometric.get("detach_camera", False)
        detach_depth  = self.enable_alternate_geometric_opt and self.opt_flag
        geometric_loss_value, normal_loss_value, dist_loss_value, translation_loss_value, flow_loss_value, matching_info = self.geometric_loss(output_dict, data, 
                                                                                                                            detach_camera=detach_camera, 
                                                                                                                            detach_depth=detach_depth,
                                                                                                                            w_translate=self.w_translate_def, w_flow=self.w_flow_def)
        
        # ---------------------------------------------------------------------------------------------------
        has_sv_data = (
            data.get("input_points_local") is not None
            and data.get("input_points") is not None
        )
        has_nv_data = (
            data.get("input_points_local") is None
            and data.get("input_points") is None
        )

        if not (has_sv_data or has_nv_data):
            raise ValueError(
                f"Inconsistent supervised data state at step {step}: "
                f"input_points_local is {'not None' if data.get('input_points_local') is not None else 'None'}, "
                f"input_points is {'not None' if data.get('input_points') is not None else 'None'}"
            )

        if self.step_begin_stage1 <= step < self.step_begin_stage2:
            # 纯监督阶段：必须有监督数据
            assert has_sv_data, f"Supervised stage requires supervised data, but got none at step {step}"
            w_extr, w_locpoint, w_glbpoint = self.w_extr_def, self.w_locpoint_def, self.w_glbpoint_def
            w_geometric, w_dist, w_normal, w_translate, w_flow = 0.0, 0.0, 0.0, 0.0, 0.0

            mode = "stage1-svData_svLoss"
            locpts_loss, glbpts_loss, point_loss_details, S_opt_local = self.point_loss(output_dict, data)

        elif self.step_begin_stage2 <= step < self.step_begin_stage3:
            # 混合阶段：必须有监督数据
            assert has_sv_data, f"Combined stage requires supervised data, but got none at step {step}"
            w_extr, w_locpoint, w_glbpoint = self.w_extr_def, self.w_locpoint_def, self.w_glbpoint_def
            w_geometric, w_dist, w_normal, w_translate, w_flow = self.w_geometric_def, self.w_dist_def, self.w_normal_def, self.w_translate_def, self.w_flow_def
            mode = "stage2-svData_svLoss_nvLoss"
            locpts_loss, glbpts_loss, point_loss_details, S_opt_local = self.point_loss(output_dict, data)

        elif step >= self.step_begin_stage3:
            # 无监督阶段：允许两种数据来源
            if has_nv_data:
                w_extr, w_locpoint, w_glbpoint = 0.0, 0.0, 0.0
                w_geometric, w_dist, w_normal, w_translate, w_flow = self.w_geometric_def, self.w_dist_def, self.w_normal_def, self.w_translate_def, self.w_flow_def
                mode = "stage3-nvData_nvLoss"
                locpts_loss, glbpts_loss, S_opt_local = torch.tensor(0.0), torch.tensor(0.0), None
            else:
                w_extr, w_locpoint, w_glbpoint = self.w_extr_def, self.w_locpoint_def, self.w_glbpoint_def
                w_geometric, w_dist, w_normal, w_translate, w_flow = self.w_geometric_def, self.w_dist_def, self.w_normal_def, self.w_translate_def, self.w_flow_def
                mode = "stage3-svData_svLoss_nvLoss"
                locpts_loss, glbpts_loss, point_loss_details, S_opt_local = self.point_loss(output_dict, data)

        # if data.get("input_points_local", None) is not None:
        #     point_loss, point_loss_details, S_opt_local = self.point_loss(output_dict, data)
        #     if w_point == 0.0 and w_extr > 0:
        #         S_opt_local = None
        
        else:
            raise NotImeplementedError(f'check step: {step}')
        # ---------------------------------------------------------------------------------------------------        
        ## Extrinsic loss, disabled for self-sup learning
        extr_loss_value, extr_r_loss_value, extr_t_loss_value = self.relative_camera_loss(output_dict, data, S_opt_local)            

        if self.enable_alternate_geometric_opt:
            self.opt_flag = not self.opt_flag
        ## Intrinsic loss
        intr_loss_value = self.intr_loss(output_dict, data)
        ## Normalization loss to regulate the value range of depth maps and camera coordinate.
        normalize_loss_value = self.normalize_loss(output_dict)
        
        total_loss = (w_ds * ds_loss_value + 
                      w_geometric * geometric_loss_value +
                      w_dist * dist_loss_value +  
                      w_normal * normal_loss_value + 
                      w_translate * translation_loss_value +
                      w_flow * flow_loss_value +
                      w_locpoint * locpts_loss + 
                      w_glbpoint * glbpts_loss + 
                      w_intr * intr_loss_value + 
                      w_extr * extr_loss_value + 
                      w_normalize * normalize_loss_value)

        if w_render is not None:
            images_gt = data["input_images"]
            depth_gt = (data["input_depths"] if data["input_depths"] is not None \
                else output_dict["depths"])[:, :, 0]
            render = output_dict["render"]
            render_loss_value = self.render_loss(render, images_gt, depth_gt, w_render)
            total_loss += render_loss_value
        
        ### Return loss information
        loss_dict = {
            "total":         total_loss, 
            "depth_shape":   ds_loss_value, 
            "match_cosine":  geometric_loss_value, 
            "match_normal":  normal_loss_value, 
            "match_dist":    dist_loss_value, 
            "translation":   translation_loss_value,
            "flow":          flow_loss_value,
            "intr_fov":      intr_loss_value, 
            "extr_rel":      extr_loss_value, 
            "extr_r":        extr_r_loss_value, 
            "extr_t":        extr_t_loss_value, 
            "normalize":     normalize_loss_value, 
            "matching_info": matching_info, 
            "mode":          mode,
        }
        if data.get("input_points_local", None) is not None:
            loss_dict.update({
                "point_local":  point_loss_details["local_pts_loss"], 
                "point_global": point_loss_details["global_pts_loss"], 
            })
        if w_render is not None:
            loss_dict.update(render_loss=render_loss_value)
        return loss_dict