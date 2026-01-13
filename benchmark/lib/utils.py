# benchmark/models/utils.py
from __future__ import annotations

# benchmark/datasets/utils.py
from __future__ import annotations
from pathlib import Path
from typing import Any, List, Tuple
from PIL import Image

import numpy as np
import torch

from lib.dataset_util import depth_to_world_coords_points
from lib.cropping import resize_image_depth_and_intrinsic
from lib.normalization import normalize_camera_extrinsics_and_points_batch

import time
from functools import wraps

# for recording the execution time
# use it as a simple decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"'{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def save_npy(root: Path, name: str, x) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(str(root / name), to_numpy(x))

def filter_depth(depthmaps, far=10.0, near=0.1):
    # depthmaps: np.ndarray
    depthmaps[depthmaps > far] = 0.0
    depthmaps[depthmaps < near] = 0.0
    depth_masks = depthmaps != 0.0
    return depthmaps, depth_masks

def build_gt_geometry_data(
        image_pils: List[Image.Image], 
        depths: torch.Tensor, 
        raw_w2c: torch.Tensor, 
        raw_K: torch.Tensor, 
        output_width: int,
        depth_far: float=10.0,
        depth_near: float=1e-3,):
    
    S = depths.shape[0]
    new_depthmaps, extrinsics_3x4, cam_points, world_points = [], [], [], []
    new_intrinsics = []
    new_images = []

    for i in range(S):
        depth_map = depths[i].numpy()
        extrinsic = raw_w2c[i, :3].numpy()   # (3,4)
        intrinsic = raw_K[i].numpy()       # (3,3)

        img_rs, depth_map, intrinsic = resize_image_depth_and_intrinsic(
            image=image_pils[i],
            depth_map=depth_map,
            intrinsic=intrinsic,
            output_width=output_width
        )

        # save resized rgb (same H,W as depth_map)
        if img_rs.mode != "RGB":
            img_rs = img_rs.convert("RGB")
        new_images.append(np.asarray(img_rs, dtype=np.uint8))  # (H,W,3)

        world_pts, cam_pts, _ = depth_to_world_coords_points(depth_map, extrinsic, intrinsic)

        new_depthmaps.append(depth_map)
        new_intrinsics.append(intrinsic)
        extrinsics_3x4.append(extrinsic)
        cam_points.append(cam_pts)
        world_points.append(world_pts)

    new_depthmaps = np.stack(new_depthmaps)
    new_intrinsics = np.stack(new_intrinsics)
    extrinsics_3x4 = np.stack(extrinsics_3x4)
    cam_points = np.stack(cam_points)
    world_points = np.stack(world_points)
    new_images = np.stack(new_images, axis=0)

    new_depthmaps, depth_masks = filter_depth(new_depthmaps,far=depth_far,near=depth_near)

    new_extrinsics, new_cam_points, new_world_points, new_depthmaps = normalize_camera_extrinsics_and_points_batch(
        extrinsics=torch.from_numpy(extrinsics_3x4).unsqueeze(0),
        cam_points=torch.from_numpy(cam_points).unsqueeze(0),
        world_points=torch.from_numpy(world_points).unsqueeze(0),
        depths=torch.from_numpy(new_depthmaps).unsqueeze(0),
        point_masks=torch.from_numpy(depth_masks).unsqueeze(0),
    )

    return new_intrinsics, new_extrinsics, new_cam_points, new_world_points, new_depthmaps, depth_masks, new_images

def strip_ddp_prefix(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    ks = list(state_dict.keys())
    if all(k.startswith("module.") for k in ks):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def calc_intr_inv(intr: torch.Tensor):
    assert intr.shape[-2] == 3 and intr.shape[-1] == 3
    ret = torch.zeros_like(intr)
    ret[..., 0, 0] = 1 / intr[..., 0, 0]
    ret[..., 1, 1] = 1 / intr[..., 1, 1]
    ret[..., 0, 2] = -intr[..., 0, 2] / intr[..., 0, 0]
    ret[..., 1, 2] = -intr[..., 1, 2] / intr[..., 1, 1]
    ret[..., 2, 2] = ret[..., 2, 2] + 1.0
    return ret

def rdepth_to_zdepth(r_depths: torch.Tensor, intrs: torch.Tensor):
    """
    Transform ray depth into z depth.
    Args:
        r_depths (torch.Tensor): Tensor of shape (N, 1, H, W).
        intrs (torch.Tensor): Tensor of shape (N, 3, 3), normalized.
    """
    H, W = r_depths.shape[-2:]
    image_grid = torch.cartesian_prod((torch.arange(H) + 0.5) / H, (torch.arange(W) + 0.5) / W).reshape(H, W, 2).flip(dims=[-1])
    
    pixel_vec = torch.cat([image_grid, torch.ones_like(image_grid[..., :1])], dim=-1)[None] # 1 H W 3
    pixel_vec = pixel_vec.to(intrs.device)
    pixel_vec = (calc_intr_inv(intrs)[:, None, None] @ pixel_vec[..., None])[..., 0]
    pixel_vec = pixel_vec / (pixel_vec.norm(dim=-1, keepdim=True) + 1e-12) # N H W 3
    cam_coord = pixel_vec * r_depths.permute(0, 2, 3, 1)
    z_depth = cam_coord[..., -1:].permute(0, 3, 1, 2)
        
    return z_depth

def fovs_to_K_norm(fovs_pred: torch.Tensor, fov_in_degrees: bool = False):
    """
    fovs_pred: (B,V,2)  last dim = (fov_x, fov_y)
    return:    (B,V,3,3) normalized intrinsics (cx=cy=0.5)
    """
    if fov_in_degrees:
        fovs_pred = torch.deg2rad(fovs_pred)

    B, V = fovs_pred.shape[:2]
    focal_pred = 0.5 / torch.tan(fovs_pred / 2)  # (B,V,2)

    eye3 = torch.eye(3, device=fovs_pred.device, dtype=fovs_pred.dtype)
    intrs = eye3[None, None].expand(B, V, -1, -1).clone()         # (B,V,3,3)
    intrs[..., :2, :] = intrs[..., :2, :] * focal_pred[..., None] # row0*=fx_norm, row1*=fy_norm
    intrs[..., 0, 2] = 0.5
    intrs[..., 1, 2] = 0.5
    return intrs


def K_denormalize(K_norm: torch.Tensor, W: int, H: int):
    """
    把归一化内参 K_norm 反归一化成像素内参 K_pix
    约定：x_norm = x_pix / W, y_norm = y_pix / H
    """
    K_pix = K_norm.clone()
    K_pix[..., 0, 0] = K_norm[..., 0, 0] * W
    K_pix[..., 1, 1] = K_norm[..., 1, 1] * H
    K_pix[..., 0, 2] = K_norm[..., 0, 2] * W
    K_pix[..., 1, 2] = K_norm[..., 1, 2] * H
    K_pix[..., 2, 2] = 1.0
    return K_pix

# def camera_fovs_to_K(camera_fovs: torch.Tensor, W: int, H: int,
#                      cx: float | None = None, cy: float | None = None,
#                      fov_in_degrees: bool = False) -> torch.Tensor:
#     """
#     camera_fovs: (B, V, 2) or (..., 2), last dim = (fov_x, fov_y)
#     returns:     (B, V, 3, 3) or (..., 3, 3)
#     """
#     assert camera_fovs.shape[-1] == 2, camera_fovs.shape

#     fov = camera_fovs
#     if fov_in_degrees:
#         fov = torch.deg2rad(fov)

#     fov_x = fov[..., 0]
#     fov_y = fov[..., 1]

#     fx = (0.5 * W) / torch.tan(0.5 * fov_x)
#     fy = (0.5 * H) / torch.tan(0.5 * fov_y)

#     if cx is None:
#         cx = (W - 1) * 0.5
#     if cy is None:
#         cy = (H - 1) * 0.5

#     K = torch.zeros((*camera_fovs.shape[:-1], 3, 3),
#                     device=camera_fovs.device, dtype=camera_fovs.dtype)
#     K[..., 0, 0] = fx
#     K[..., 1, 1] = fy
#     K[..., 0, 2] = cx
#     K[..., 1, 2] = cy
#     K[..., 2, 2] = 1.0
#     return K