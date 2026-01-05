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

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x

def save_npy(root: Path, name: str, x) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(str(root / name), to_numpy(x))

def filter_depth(depthmaps, far=10.0, near=1e-3):
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