import torch
import numpy as np
import open3d as o3d
from einops import rearrange

from utils.platform import DEVICE_NAME


def cross_prod(x: torch.Tensor, y: torch.Tensor):
    """
    Perform cross product between x and y, where the vector dimension is supposed to be on the last axis.
    """
    assert x.shape[-1] == 3 and y.shape[-1] == 3
    z = torch.stack([
        x[..., 1] * y[..., 2] - x[..., 2] * y[..., 1], 
        x[..., 2] * y[..., 0] - x[..., 0] * y[..., 2], 
        x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0], 
    ], dim=-1)
    return z


def concat_data_by_view(data):
    images = torch.cat([data["input_images"], data["output_images"]], dim=1)
    plucker_maps = torch.cat([data["input_cams_plucker"], data["output_cams_plucker"]], dim=1)
    cameras = []
    for input_cams, output_cams in zip(data["input_cams"], data["output_cams"]):
        cameras.append(input_cams + output_cams)
    return images, cameras, plucker_maps

def calc_intr_inv(intr: torch.Tensor):
    assert intr.shape[-2] == 3 and intr.shape[-1] == 3
    ret = torch.zeros_like(intr)
    ret[..., 0, 0] = 1 / intr[..., 0, 0]
    ret[..., 1, 1] = 1 / intr[..., 1, 1]
    ret[..., 0, 2] = -intr[..., 0, 2] / intr[..., 0, 0]
    ret[..., 1, 2] = -intr[..., 1, 2] / intr[..., 1, 1]
    ret[..., 2, 2] = ret[..., 2, 2] + 1.0
    return ret

def calc_extr_inv(extr: torch.Tensor):
    assert extr.shape[-2] == 4 and extr.shape[-1] == 4
    rot   = extr[..., :3, :3]
    trans = extr[..., :3, 3:]
    ret = torch.zeros_like(extr)
    ret[..., :3, :3] = rot.transpose(-2, -1)
    ret[..., :3, 3:] = torch.matmul(rot.transpose(-2, -1), -trans)
    ret[..., 3, 3] = ret[..., 3, 3] + 1.0
    return ret

def norm_camera_by_pcd(cameras: torch.Tensor, point_clouds: torch.Tensor):
    """Normalize camera translation by point cloud coordinate.

    Args:
        cameras (torch.Tensor): Cameras to be normalized. Tensor of shape (B, V, 4, 4).
        point_clouds (torch.Tensor): Point cloud to draw scaling factor from. Tensor of shape (B, V, H, W, 3)

    Returns:
        torch.Tensor: Cameras with normalized translation.
    """
    assert cameras.ndim == 4 and point_clouds.ndim == 5
    B = cameras.shape[0]
    point_clouds = point_clouds.reshape(B, -1, 3) # b vhw 3
    norm_factor = point_clouds.norm(dim=-1).mean(dim=-1)[..., None, None].detach() # b 1 1
    cameras_normed = cameras.clone()
    cameras_normed[..., :3, -1] = cameras[..., :3, -1] / norm_factor # b v 3
    return cameras_normed, norm_factor

class WeightedMAD(torch.nn.Module):
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def weighted_median(self, x: torch.Tensor, weights: torch.Tensor, q: float=0.5):
        """Calculate the weighted median.
        
        Args:
            x (torch.Tensor): Tensor of shape [..., D].
            weights (torch.Tensor) Tensor of shape [..., D].
            q (float): Target pencertage of median.
        """
        
        orig_shape = x.shape
        flat_x = x.reshape(-1, orig_shape[-1])
        flat_weights = weights.reshape(-1, orig_shape[-1])
        
        sorted_x, sorted_idx = flat_x.sort(dim=-1)
        sorted_weights = torch.gather(flat_weights, -1, sorted_idx)
        
        cum_weights = sorted_weights.cumsum(dim=-1)
        total_weights = cum_weights[..., -1].unsqueeze(-1)
        
        target = q * total_weights
        med_idx = (cum_weights <= target).sum(dim=-1, keepdim=True).long()
        
        lower_bound = torch.gather(sorted_x, -1, med_idx.clamp(max=orig_shape[-1] - 1))
        upper_bound = torch.gather(sorted_x, -1, (med_idx + 1).clamp(max=orig_shape[-1] - 1))
        
        median = (lower_bound + upper_bound) / 2
        return median.reshape(*orig_shape[:-1], 1)
    
    def forward(self, x: torch.Tensor, weights: torch.Tensor=None) -> torch.Tensor:
        """A apply weighted MAD to x.

        Args:
            x (torch.Tensor): Tensor of shape [..., D].
            weights (torch.Tensor, optional): Tensor of shape [..., D]. Defaults to None.

        Returns:
            torch.Tensor: MAD normalized x.
        """
        if weights is None:
            weights = torch.ones_like(x)
        assert x.shape == weights.shape
        
        t_w = self.weighted_median(x, weights) # [..., 1]
        s_w = (torch.abs(x - t_w) * weights).sum(dim=-1, keepdim=True) / (weights.sum(dim=-1, keepdim=True) + self.eps) # [..., 1]
        
        normalized_x = (x - t_w) / (s_w + self.eps) # [..., D]
        
        return normalized_x


def normalize_trans(extr: torch.Tensor, detach_norm_factor: bool=False, return_norm_factor=False):
    """Normalize tranlation of extrinsics by their camera coordinate.

    Args:
        extr (torch.Tensor): Extrinsics. Tensor of shape (..., v, 4, 4)

    Returns:
        torch.Tensor: Normalized extrinsics. Tensor of shape (..., v, 4, 4)
    """
    # t: Tensor of shape (..., v, 4, 4)
    t = extr[..., :3, 3] # ... v 3
    t_magnitude = t.norm(dim=-1).mean(dim=-1)[..., None, None] # ... 1 1
    if detach_norm_factor:
        t_magnitude = t_magnitude.detach()
    t = t / (t_magnitude + 1e-12)
    extr = extr.clone()
    extr[..., :3, 3] = t
    if not return_norm_factor:
        return extr
    return extr, t_magnitude[..., 0, 0] # (..., v, 4, 4), (..., )


def normalize_points(points: torch.Tensor, detach_norm_factor: bool=False, return_norm_factor=False):
    """Normalize points by their coordinate.

    Args:
        points (torch.Tensor): Points to be normalized. Tensor of shape (..., v, h, w, 3)

    Returns:
        torch.Tensor: Normalized extrinsics. Tensor of shape (..., v, h, w, 3)
    """
    p_magnitude = points.norm(dim=-1).mean(dim=-1).mean(dim=-1).mean(dim=-1)[..., None, None, None, None] # ..., 1, 1, 1, 1
    if detach_norm_factor:
        p_magnitude = p_magnitude.detach()
    points = points / (p_magnitude + 1e-12)
    if not return_norm_factor:
        return points
    return points, p_magnitude[..., 0, 0, 0, 0] # (..., v, h, w, 3), (..., )


def mat_mul_vec_by_conv(vec: torch.Tensor, mat: torch.Tensor):
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
    

def pad_one(x: torch.Tensor, dim: int):
    shape = torch.tensor(x.shape)
    shape[dim] = 1
    return torch.cat([x, torch.ones(*shape, dtype=x.dtype, device=x.device)], dim=dim)


def intr2ray(intrs: torch.Tensor, H: int, W: int):
    """Transform normalized intrinsic matrices into ray maps.

    Args:
        intrs (torch.Tensor): Intrinsics, normalized. A tensor of shape (B, V, 3, 3) or (B, 3, 3)
        H (int): Height of the output ray map.
        W (int): Width of the output ray map.

    Returns:
        torch.Tensor: Tensor of shape (B, V, 3, H, W) or shape (B, 3, H, W)
    """
    assert intrs.ndim == 4 or intrs.ndim == 3
    assert intrs.shape[-2] == 3 and intrs.shape[-1] == 3
    intrs_shape = intrs.shape
    intrs = intrs.reshape(-1, 3, 3)
    
    intrs_inv = calc_intr_inv(intrs)
    grids = torch.cartesian_prod((torch.arange(W, device=intrs.device) + 0.5) / W, (torch.arange(H, device=intrs.device) + 0.5) / H)
    grids = grids.reshape(1, W, H, 2).permute(0, 3, 2, 1).expand(intrs.shape[0], -1, -1, -1) # bv 2 h w
    pixel_rays = mat_mul_vec_by_conv(pad_one(grids, dim=-3), intrs_inv) # bv 3 h w
    return pixel_rays.reshape(*intrs_shape[:-2], *pixel_rays.shape[-3:])


def fov2intr(fovs: torch.Tensor):
    """Transform fovs into intrinsics.

    Args:
        fovs (torch.Tensor): Field of views. Tensor of shape (..., 2).
    
    Returns:
        torch.Tensor: Intrinsics. Tensor of shape (..., 3, 3).
    """
    assert (fovs.ndim == 3 or fovs.ndim == 2) and fovs.shape[-1] == 2
    focals = 0.5 / torch.tan(fovs / 2) # ..., 2
    intrs = torch.zeros((*focals.shape[:-1], 3, 3), dtype=focals.dtype, device=focals.device)
    intrs[..., 0, 2] = 0.5
    intrs[..., 1, 2] = 0.5
    intrs[..., 0, 0] = focals[..., 0]
    intrs[..., 1, 1] = focals[..., 1]
    intrs[..., 2, 2] = 1.0
    return intrs


def depth2pointmap(depths: torch.Tensor, extrs: torch.Tensor, intrs: torch.Tensor, device=DEVICE_NAME, is_intr_normalized=True, depth_is_z=True):
    """Back-project depth maps into 3D point maps with given cameras. 

    Args:
        depths (torch.Tensor): Depth maps, tensor of shape (B, 1, H, W).
        extrs (torch.Tensor): Extrinsics (world-to-camera), tensor of shape (B, 4, 4).
        intrs (torch.Tensor): Intrinsics (not normalized, assuming c corresponds to the image center), tensor of shape (B, 3, 3).
        device (str): The default device to perform operations on.
        is_intr_normalized (bool): If the intrinsics are normalized or not. 
        depth_is_z (bool): If the input depths maps are z-depth. True if they are z-depth while False if they are r-depth.
    
    Returns:
        torch.Tensor: The back-projected point maps, tensor of shape (B, H, W, 3).
    """
    B, C, H, W = depths.shape
    assert depths.ndim == 4 and C == 1
    
    if not is_intr_normalized:
        intr_normalizer = torch.tensor([W, H, 1], dtype=torch.float32, device=device).reshape(1, 3, 1)
        intrs = intrs / intr_normalizer
    extrs_inv = calc_extr_inv(extrs)
    intrs_inv = calc_intr_inv(intrs)
    
    grids = torch.cartesian_prod((torch.arange(W, device=device) + 0.5) / W, (torch.arange(H, device=device) + 0.5) / H)
    grids = grids.reshape(1, W, H, 2).permute(0, 3, 2, 1).expand(B, -1, -1, -1) # b 2 h w
    pixel_vecs = mat_mul_vec_by_conv(pad_one(grids, dim=1), intrs_inv)
    if not depth_is_z:
        pixel_vecs = pixel_vecs / (pixel_vecs.norm(dim=1, keepdim=True) + 1e-8)
    points_local = pixel_vecs * depths # b 3 h w
    points_world = mat_mul_vec_by_conv(pad_one(points_local, dim=1), extrs_inv)
    points_world = points_world[:, :3, ...] / points_world[:, 3:, ...] # b 3 h w
    points_world = rearrange(points_world, "b c h w -> b h w c")
    
    return points_world


def localpoint2global(points: torch.Tensor, extrs: torch.Tensor):
    """Align local point map under the global coordinate.

    Args:
        points (torch.Tensor): Local point map, tensor of shape (..., h, w, 3).
        extrs (torch.Tensor): World-to-camera transform, tensor of shape (..., 4, 4)
    """
    points_world_homo = mat_mul_vec_by_conv(pad_one(points, dim=-1), calc_extr_inv(extrs))
    points_world = points_world_homo[..., :3] / points_world_homo[..., -1:]
    return points_world


def globalpoint2local(points: torch.Tensor, extrs: torch.Tensor):
    """Align local point map under the global coordinate.

    Args:
        points (torch.Tensor): Global point map, tensor of shape (..., h, w, 3).
        extrs (torch.Tensor): World-to-camera transform, tensor of shape (..., 4, 4)
    """
    points_local_homo = mat_mul_vec_by_conv(pad_one(points, dim=-1), extrs)
    points_local = points_local_homo[..., :3] / points_local_homo[..., -1:]
    return points_local


def trans_geo_to_first_frame_and_norm(point_maps: torch.Tensor, extrs: torch.Tensor, depths: torch.Tensor=None, normalize=True):
    """Transformer point maps and camera extrinsics coordinate to the first frame.

    Args:
        point_maps (torch.Tensor): Points in world coordinate. Tensor of shape (B, V, H, W, 3)
        extrs (torch.Tensor): Camera extrinsics (w2c). Tensor of shape (B, V, 4, 4)
        depths (torch.Tensor): Depth maps. Tensor of shape (B, V, 1, H, W)
    """
    def normalize_trans(point_maps: torch.Tensor, extr: torch.Tensor, depths: torch.Tensor=None):
        # t: Tensor of shape (..., v, 4, 4)
        norm_magnitude = point_maps.norm(dim=-1).mean(-1).mean(-1).mean(-1) # ...
        t = extr[..., :3, 3] / norm_magnitude[..., None, None] # ... v 3
        extr = extr.clone()
        extr[..., :3, 3] = t
        point_maps = point_maps / norm_magnitude[..., None, None, None, None]
        if depths is not None:
            depths = depths / norm_magnitude[..., None, None, None, None]
        return point_maps, extr, depths
    V = point_maps.shape[-4]
    homo_point_maps = pad_one(point_maps, dim=-1)
    trans_mat = torch.cat([extrs[..., :1, :, :] for _ in range(V)], dim=-3) # ... v 4 4
    homo_point_maps = mat_mul_vec_by_conv(homo_point_maps, trans_mat)
    point_maps = homo_point_maps[..., :3]
    extrs = extrs @ calc_extr_inv(trans_mat) # c0-to-world @ world-to-c -> c0-to-c
    if normalize:
        point_maps, extrs, depths = normalize_trans(point_maps, extrs, depths)
    return point_maps, extrs, depths


def get_norm_map(points: torch.Tensor, normalize=True):
    """Compute normals from the given points

    Args:
        points (torch.Tensor): Given point maps, tensor of shape (..., 3, H, W) or (..., H, W, 3)
        normalize (bool): Whether to normalize the output normal maps.
    Returns:
        norms (torch.Tensor): The calculated normal maps, tensor of shape (..., 3, H, W) or (..., H, W, 3)
    """
    assert points.ndim == 4 and points.shape[-3] == 3 or points.shape[-1] == 3
    raw_shape = points.shape
    points = points.reshape(-1, *points.shape[-3:]) # b h w 3 / b 3 h w
    flag = False
    if points.shape[-3] == 3:
        flag = True
        points = rearrange(points, "b c h w -> b h w c")
    norms_00 = cross_prod(points[..., :-1, :-1, :] - points[..., 1:, :-1, :], points[..., :-1, :-1, :] - points[..., :-1, 1:, :])       # b h-1 w-1 3
    norms_01 = cross_prod(points[..., :-1, -1:, :] - points[..., :-1, -2:-1, :], points[..., :-1, -1:, :] - points[..., 1:, -1:, :])    # b h-1 1 3
    norms_10 = cross_prod(points[..., -1:, :-1, :] - points[..., -1:, 1:, :], points[..., -1:, :-1, :] - points[..., -2:-1, :-1, :])    # b 1 w-1 3
    norms_11 = cross_prod(points[..., -1:, -1:, :] - points[..., -2:-1, -1:, :], points[..., -1:, -1:, :] - points[..., -1:, -2:-1, :]) # b 1 1 3
    norms = torch.cat([torch.cat([norms_00, norms_01], dim=-2), 
                       torch.cat([norms_10, norms_11], dim=-2), ], dim=-3) # b h w 3
    if normalize:
        norms = norms / (norms.norm(dim=-1, keepdim=True) + 1e-8)
    if flag:
        norms = rearrange(norms, "b h w c -> b c h w")
    norms = norms.reshape(*raw_shape)
    return norms