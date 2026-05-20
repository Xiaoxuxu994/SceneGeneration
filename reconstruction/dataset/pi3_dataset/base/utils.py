# adapted from https://github.com/YufeiWang777/LRRU/blob/main/dataloaders/NNfill.py

import cv2
import numpy as np
from scipy.ndimage.measurements import label
from .transforms import *
import math
from torch.utils.data.dataloader import default_collate
import random
import os.path as osp
import h5py
import re

from ...reliev3r_dataset.utils import common as reliev3r_data_cm
from utils import common as cm

def tensor2pil(img):
    if -1 <= img.min() <= 0:
        img_ = inverse_ImgNorm(img) 
    elif img.min() < -1:
        img_ = inverse_CustomNorm(img)
    else:
        img_ = img

    return PIL.Image.fromarray((img_.detach().cpu().numpy()*255).astype(np.uint8).transpose(1, 2, 0))


PATTERN_IDS = {
    'random': 0,
    'velodyne': 1,
    'sfm': 2,
    'denoise': 3,         
    'superresolution': 4,
    'allzero': 5,
    'identity': 6
}

def tensor2numpy(img):
    if -1 <= img.min() <= 0:
        img_ = inverse_ImgNorm(img) 
    elif img.min() < -1:
        img_ = inverse_CustomNorm(img)
    else:
        img_ = img
    return (img_.detach().cpu().numpy()*255).astype(np.uint8).transpose(1, 2, 0)

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_13 = np.ones((13, 13), np.uint8)
FULL_KERNEL_25 = np.ones((25, 25), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_9 = np.asarray(
    [
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
    ], dtype=np.uint8)

DIAMOND_KERNEL_13 = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """
    # depth_map = np.squeeze(depth_map, axis=-1)

    thres=0.0001
    depth_map_original = np.copy(depth_map)

    # Invert
    valid_pixels = (depth_map > thres)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < thres)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > thres, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

    # Large Fill
    empty_pixels = depth_map < thres
    dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = depth_map.astype('float32')  # Cast a float64 image to float32
    depth_map = cv2.medianBlur(depth_map, 5)
    depth_map = depth_map.astype('float64')  # Cast a float32 image to float64
    #
    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = depth_map.astype('float32')
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
        depth_map = depth_map.astype('float64')
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > thres)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > thres)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # fill zero value
    mask = (depth_map <= thres)
    if np.sum(mask) != 0:
        labeled_array, num_features = label(mask)
        for i in range(num_features):
            index = i + 1
            m = (labeled_array == index)
            m_dilate1 = cv2.dilate(1.0*m, FULL_KERNEL_7)
            m_dilate2 = cv2.dilate(1.0*m, FULL_KERNEL_13)
            m_diff = m_dilate2 - m_dilate1
            if np.sum(m_diff>0) > 0:
                v = np.mean(depth_map[m_diff>0])
            else:
                v = np.mean(depth_map)
                
            depth_map = np.ma.array(depth_map, mask=m_dilate1, fill_value=v)
            depth_map = depth_map.filled()
            depth_map = np.array(depth_map)
    else:
        depth_map = depth_map

    # if originally has value, keep it
    depth_map[depth_map_original > thres] = depth_map_original[depth_map_original > thres]

    return depth_map


def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"

def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (torch.bool, np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        if 'normal' in view:
            assert view['normal'].shape == (height, width, 3)
            view['normal'] = view['normal'].swapaxes(0, 1)

        if 'far_mask' in view:
            assert view['far_mask'].shape == (height, width)
            view['far_mask'] = view['far_mask'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)

        # transpose x and y pixels
        view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]

def transform_data_to_reliev3r_format(views, grid_size=100):
    name_mapping = {
        "img":               "input_images", 
        "depthmap":          "input_depths", 
        "valid_mask":        "input_depth_masks", 
        "camera_pose":       "input_extrs", 
        "camera_intrinsics": "input_intrs", 
        "dataset":           "dataset_name", 
        "pts3d":             "input_points", 
    }
    view_num = len(views)
    matched_view_i, matched_view_j = [], []
    for i in range(view_num):
        for j in range(view_num):
            if i != j:
                matched_view_i.append(i)
                matched_view_j.append(j)
    matched_view_i, matched_view_j = np.array(matched_view_i), np.array(matched_view_j)
    batch_data = []
    batch_size = views[0]["img"].shape[0]
    for b in range(batch_size):
        data = dict()
        for view in views:
            ### Mapping dictinary names
            for k, v in name_mapping.items():
                if data.get(v, None) is None:
                    data[v] = [view[k][b]] if k != "dataset" else view[k][b]
                else:
                    data[v] = (data[v] + [view[k][b]]) if k != "dataset" else view[k][b]
        for k in data.keys():
            if k == "dataset_name":
                continue
            data[k] = torch.stack(data[k], dim=0)
        data["input_extrs"] = cm.calc_extr_inv(data["input_extrs"]) # c2w -> w2c
        data["input_depths"] = data["input_depths"].unsqueeze(1)
        data["input_depth_masks"] = data["input_depth_masks"].unsqueeze(1)
        data["input_depth_masks"] = data["input_depth_masks"] & (~data["input_depths"].isinf()) & (~data["input_depths"].isnan())
        data["input_depths"][~data["input_depth_masks"]] = data["input_depths"][data["input_depth_masks"]].max() * 1.05
        data["input_depth_masks"] = data["input_depth_masks"].to(data["input_depths"].dtype)
        data["input_intrs"][..., 0:1, :] /= data["input_images"].shape[-1]
        data["input_intrs"][..., 1:2, :] /= data["input_images"].shape[-2]
        
        ### Calculate matching with GT for Reliev3R supervision
        tracks0, tracks1, visibs = reliev3r_data_cm.exhaustive_geo2match(
            data["input_depths"], data["input_depth_masks"], 
            cm.calc_extr_inv(data["input_extrs"]), data["input_intrs"], 
            eps=1e-2, samples_to_keep=10000, grid_size=grid_size)
        data.update(dict(input_tracks_view_id0=torch.from_numpy(matched_view_i), # v-1
                        input_tracks_view_id1=torch.from_numpy(matched_view_j), # v-1
                        input_tracks0=torch.stack([tracks0[reliev3r_data_cm.get_tracking_idx_by_pair(i, j, len(views), True)] \
                                                    for i, j in zip(matched_view_i, matched_view_j)]), # v-1 k 2
                        input_tracks1=torch.stack([tracks1[reliev3r_data_cm.get_tracking_idx_by_pair(i, j, len(views), True)] \
                                                    for i, j in zip(matched_view_i, matched_view_j)]), # v-1 k 2
                        input_visibs0=torch.stack([visibs[reliev3r_data_cm.get_tracking_idx_by_pair(i, j, len(views), True)] \
                                                    for i, j in zip(matched_view_i, matched_view_j)]), # v-1 k
                        input_visibs1=torch.stack([visibs[reliev3r_data_cm.get_tracking_idx_by_pair(i, j, len(views), True)] \
                                                    for i, j in zip(matched_view_i, matched_view_j)]), # v-1 k
                        ))
        ### Transfer z-depth to r-depth
        # data.update(dict(input_depths=reliev3r_data_cm.zdepth_to_rdepth(data["input_depths"], data["input_intrs"])))
        ### Normalize pointmaps, cameras, depths simultaneously
        data["input_points"], data["input_extrs"], data["input_depths"] = \
            cm.trans_geo_to_first_frame_and_norm(data["input_points"], data["input_extrs"], data["input_depths"])
        data["input_points_local"] = cm.globalpoint2local(data["input_points"], data["input_extrs"])
        data["data_type"] = "synthetic"
        
        batch_data.append(data)
    batch_data = reliev3r_data_cm.customized_collate_fn(batch_data)
    return batch_data

def unified_collate_fn(batch):
    if isinstance(batch[0], dict):
        batch = [batch]
    views_num = len(batch[0])
    all_keys = batch[0][0].keys()

    batched_data = [{key: [] for key in all_keys} for _ in range(views_num)]
    for sample in batch:
        for i in range(views_num):
            for key in all_keys:
                batched_data[i][key].append(sample[i].get(key, None))

    for i in range(views_num):
        for key, data in batched_data[i].items():
            try:
                batched_data[i][key] = default_collate(data)
            except Exception:
                batched_data[i][key] = data

    batched_data = transform_data_to_reliev3r_format(batched_data)

    return batched_data

def add_noise(dep, input_noise, generator=None):
    # add noise
    # the noise can be "0.1" (fixed probablity) or "0.0~0.1" (uniform in the range)
    if input_noise != "0.0":
        if generator is None:
            generator = np.random.default_rng(None)

        if '~' in input_noise:
            noise_prob_low, noise_prob_high = input_noise.split('~')
            noise_prob_low, noise_prob_high = float(noise_prob_low), float(noise_prob_high)
        else:
            noise_prob_low, noise_prob_high = float(input_noise), float(input_noise)

        noise_prob = generator.uniform(noise_prob_low, noise_prob_high)
        noise_mask = torch.tensor(generator.binomial(n=1, p=noise_prob, size=dep.shape))
        depth_min, depth_max = np.percentile(dep, 10), np.percentile(dep, 90)
        noise_values = torch.tensor(generator.uniform(depth_min, depth_max, size=dep.shape)).float()

        dep[noise_mask == 1] = noise_values[noise_mask == 1]

    return dep


def _find_all_resolutions(
    aspect_ratio_range,
    pixel_count_range,
    patch_size
):
    """
    Find all possible resolutions that satisfy the given constraints.

    This function performs an exhaustive search instead of random sampling.

    Args:
        aspect_ratio_range (tuple): The range of aspect ratios (width / height).
        pixel_count_range (tuple): The range of total pixel counts (width * height).
        patch_size (int): The width and height must be divisible by this value.

    Returns:
        list of (int, int): A list of all valid (width, height) tuples.
    """
    min_ar, max_ar = aspect_ratio_range
    min_pixels, max_pixels = pixel_count_range
    
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")

    valid_resolutions = set()

    min_w_bound = int(math.sqrt(min_pixels * min_ar))
    max_w_bound = int(math.sqrt(max_pixels * max_ar))

    start_w = (min_w_bound // patch_size) * patch_size
    end_w = ((max_w_bound + patch_size - 1) // patch_size) * patch_size
    
    for width in range(start_w, end_w + 1, patch_size):
        if width == 0: continue

        h_from_ar_min = width / max_ar
        h_from_ar_max = width / min_ar
        
        h_from_pixels_min = min_pixels / width
        h_from_pixels_max = max_pixels / width

        min_h = max(h_from_ar_min, h_from_pixels_min)
        max_h = min(h_from_ar_max, h_from_pixels_max)

        start_h = math.ceil(min_h / patch_size) * patch_size
        end_h = math.floor(max_h / patch_size) * patch_size

        for height in range(start_h, end_h + 1, patch_size):
            if height == 0: continue
            pixels = width * height
            aspect_ratio = width / height
            if (min_pixels <= pixels <= max_pixels) and (min_ar <= aspect_ratio <= max_ar):
                valid_resolutions.add((width, height))

    return list(valid_resolutions)

def sample_resolutions(aspect_ratio_range=(0.5, 2.0), pixel_count_range=(250000, 500000), patch_size=1, num_resolutions=5, seed=None, base_resolution=[]):
    """
    Sample a list of random resolutions based on aspect ratio, pixel count constraints, 
    and ensure the width and height are divisible by patch_size.

    Args:
        aspect_ratio_range (tuple): The range of aspect ratios (width / height), e.g., (0.5, 2.0).
        pixel_count_range (tuple): The range of total pixel counts (width * height), e.g., (250000, 500000).
        patch_size (int): Ensure the output width and height are divisible by this value.
        num_resolutions (int): The number of resolutions to sample.

    Returns:
        list of (int, int): A list of (width, height) tuples representing the sampled resolutions.
    """
    rng = np.random.default_rng(seed=seed)
    resolutions = set(base_resolution)  # Use a set to ensure uniqueness

    if num_resolutions == -1:
        all_res = _find_all_resolutions(aspect_ratio_range, pixel_count_range, patch_size)
        resolutions.update(all_res)
        return sorted(list(resolutions))

    while len(resolutions) < num_resolutions:
        # Randomly sample an aspect ratio within the given range
        aspect_ratio = rng.uniform(*aspect_ratio_range)

        # Randomly sample a total pixel count within the given range
        pixel_count = rng.uniform(*pixel_count_range)

        # Compute height and width based on the sampled aspect ratio and pixel count
        height = math.sqrt(pixel_count / aspect_ratio)
        width = aspect_ratio * height

        # Round height and width to the nearest integers that are divisible by patch_size
        height = int(round(height / patch_size) * patch_size)
        width = int(round(width / patch_size) * patch_size)

        # Add the resolution to the set (duplicates are automatically ignored)
        resolutions.add((width, height))

    return list(resolutions)


def rodrigues_to_rotation_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of axis-angle vectors to rotation matrices using Rodrigues' formula.

    Args:
        axis_angle (torch.Tensor): A tensor of axis-angle vectors with shape (..., 3).

    Returns:
        torch.Tensor: A tensor of rotation matrices with shape (..., 3, 3).
    """
    # Extract angle and axis
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)

    # Handle the zero-angle case (which would result in a NaN axis)
    # If angle is close to zero, rotation is identity.
    is_zero_angle = angle.squeeze(-1) < 1e-8
    
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    # Create the skew-symmetric matrix K from the axis vector
    # K = [[ 0, -z,  y],
    #      [ z,  0, -x],
    #      [-y,  x,  0]]
    zeros = torch.zeros_like(angle)
    k_x, k_y, k_z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    
    # Manually construct the skew-symmetric matrix K for batch processing
    K = torch.cat([
        zeros, -k_z,  k_y,
         k_z,  zeros, -k_x,
        -k_y,  k_x,  zeros
    ], dim=-1).view(*axis.shape[:-1], 3, 3)

    # Rodrigues' rotation formula: R = I + sin(θ)K + (1 - cos(θ))K^2
    identity = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(*axis.shape[:-1], 3, 3)
    K_sq = torch.matmul(K, K)
    
    R = identity + sin_angle.unsqueeze(-1) * K + (1 - cos_angle.unsqueeze(-1)) * K_sq

    # For zero-angle cases, explicitly set to identity matrix
    R[is_zero_angle] = torch.eye(3, device=axis.device, dtype=axis.dtype)

    return R

def add_pose_noise_torch(poses_c2w: torch.Tensor,
                         translation_noise_std: float = 0.01,
                         rotation_noise_std_deg: float = 1.0) -> torch.Tensor:
    """
    Adds Gaussian noise to a batch of camera poses represented as PyTorch tensors.

    Noise is applied independently to the translation and rotation components
    of each pose in the batch.
    - Translation noise: Adds zero-mean Gaussian noise.
    - Rotation noise: Generates a small random rotation and left-multiplies
      it with the original rotation.

    Args:
        poses_c2w (torch.Tensor): A tensor of camera-to-world pose matrices
                                  with shape (B, N, 4, 4), where B is the batch
                                  size and N is the number of views/frames.
        translation_noise_std (float): The standard deviation of the translation
                                       noise in meters. Default: 0.01 (1cm).
        rotation_noise_std_deg (float): The standard deviation of the rotation
                                        noise in degrees. This standard deviation
                                        is used for each component of the random
                                        rotation vector. Default: 1.0.

    Returns:
        torch.Tensor: A new tensor of noisy pose matrices with the same shape
                      and device as the input.
    """
    if poses_c2w.ndim != 4 or poses_c2w.shape[-2:] != (4, 4):
        raise ValueError("Input poses tensor must have shape (B, N, 4, 4)")

    B, N = poses_c2w.shape[0], poses_c2w.shape[1]
    device = poses_c2w.device
    dtype = poses_c2w.dtype

    # Use .clone() to create a new tensor that is differentiable
    poses_noisy = poses_c2w.clone()

    # 1. Add translation noise
    translation_noise = torch.randn(B, N, 3, device=device, dtype=dtype) * translation_noise_std
    poses_noisy[..., :3, 3] += translation_noise

    # 2. Add rotation noise
    # Convert rotation noise standard deviation from degrees to radians
    rotation_noise_std_rad = math.radians(rotation_noise_std_deg)

    # Generate BxN random rotation vectors (axis-angle representation)
    rotation_vectors = torch.randn(B, N, 3, device=device, dtype=dtype) * rotation_noise_std_rad

    # Convert the rotation vectors to rotation matrices
    noise_rotation_matrices = rodrigues_to_rotation_matrix(rotation_vectors)

    # Extract original rotation matrices
    original_rotation_matrices = poses_c2w[..., :3, :3]

    # Compose the noise rotation with the original rotation (left-multiplication)
    # R_noisy = R_noise @ R_original
    noisy_rotation_matrices = torch.matmul(noise_rotation_matrices, original_rotation_matrices)

    # Place the new rotation matrices back into the pose tensor
    poses_noisy[..., :3, :3] = noisy_rotation_matrices

    return poses_noisy

def add_randomized_pose_noise_torch(
    poses_c2w: torch.Tensor,
    trans_std_range: tuple[float, float] = (0.0, 0.05),
    rot_std_deg_range: tuple[float, float] = (0.0, 2.0)
) -> torch.Tensor:
    """
    Adds pose noise with a randomly sampled standard deviation.

    Args:
        poses_c2w (torch.Tensor): The input poses tensor.
        trans_std_range (tuple[float, float]): The [min, max] range to sample
            the translation_noise_std from (in meters).
        rot_std_deg_range (tuple[float, float]): The [min, max] range to sample
            the rotation_noise_std_deg from (in degrees).

    Returns:
        torch.Tensor: The new tensor with randomized noise.
    """
    # 1. Sample the standard deviation from the given range
    random_trans_std = random.uniform(trans_std_range[0], trans_std_range[1])
    random_rot_std_deg = random.uniform(rot_std_deg_range[0], rot_std_deg_range[1])

    # 2. Call the original function with the sampled std values
    return add_pose_noise_torch(
        poses_c2w,
        translation_noise_std=random_trans_std,
        rotation_noise_std_deg=random_rot_std_deg
    )

def normalize_poses(poses: torch.Tensor,
                    only_translation: bool = False,
                    augment: bool = False,
                    rotation_range: float = 15.0,
                    translation_range: float = 0.1,
                    scale_range: tuple = (0.9, 1.1)):
    """
    Normalizes a batch of poses and optionally applies data augmentation.

    Args:
        poses (torch.Tensor): A tensor of poses with shape B x N x 4 x 4.
        augment (bool, optional): If True, applies random rotation, translation,
                                  and scaling. Defaults to False.
        rotation_range (float, optional): The range in degrees for random
                                          rotations. Defaults to 15.0.
        translation_range (float, optional): The range for random
                                             translations. Defaults to 0.1.
        scale_range (tuple, optional): The range for random scaling.
                                       Defaults to (0.9, 1.1).

    Returns:
        torch.Tensor: The normalized (and optionally augmented) poses.
    """
    B, N, _, _ = poses.shape
    device = poses.device

    # Isolate rotation and translation
    rotations = poses[:, :, :3, :3]
    translations = poses[:, :, :3, 3]

    # --- Normalization ---
    # Center the poses by subtracting the centroid of the translations
    centroid = torch.mean(translations, dim=1, keepdim=True)
    translations_centered = translations - centroid

    # Compute the scale factor as the mean norm of the centered translations
    if only_translation:
        scale = torch.ones((B, 1), device=device)
    else:
        scale = torch.mean(torch.linalg.norm(translations_centered, dim=2), dim=1, keepdim=True)
        # Add a small epsilon to avoid division by zero
        scale = scale + 1e-8

    # Scale the centered translations
    translations_normalized = translations_centered / scale.unsqueeze(-1)

    # Re-assemble the normalized poses
    normalized_poses = torch.clone(poses)
    normalized_poses[:, :, :3, 3] = translations_normalized

    if augment:
        # --- Data Augmentation ---
        # 1. Random Rotation
        # Generate random rotation angles
        angle_deg = (torch.rand(B, device=device) - 0.5) * 2 * rotation_range
        angle_rad = torch.deg2rad(angle_deg)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # Generate a random rotation axis (here, we use the y-axis for simplicity,
        # but a random axis can also be generated)
        # Random rotation around the Y-axis
        random_rot_mats = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
        random_rot_mats[:, 0, 0] = cos_a
        random_rot_mats[:, 0, 2] = sin_a
        random_rot_mats[:, 2, 0] = -sin_a
        random_rot_mats[:, 2, 2] = cos_a

        # Apply the random rotation to the normalized poses
        augmented_rotations = torch.matmul(random_rot_mats.unsqueeze(1), normalized_poses[:, :, :3, :3])
        augmented_translations = torch.matmul(random_rot_mats.unsqueeze(1), normalized_poses[:, :, :3, 3].unsqueeze(-1)).squeeze(-1)

        # 2. Random Translation
        random_translations = (torch.rand(B, 1, 3, device=device) - 0.5) * 2 * translation_range
        augmented_translations += random_translations

        # 3. Random Scaling
        random_scales = torch.rand(B, 1, 1, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0]
        augmented_translations *= random_scales

        scale /= random_scales[..., 0]

        # Assemble the final augmented poses
        augmented_poses = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        augmented_poses[:, :, :3, :3] = augmented_rotations
        augmented_poses[:, :, :3, 3] = augmented_translations

    return normalized_poses, scale


def read_any_file(file_name, read_dtype=np.float16, return_dtype=None):
    """
    Reads any kind file either from a local path or a remote URL.

    Args:
        file_name (str): The file name.
        dtype (np.dtype): data type for read binary data
        return_dtype (np.dtype): data type for return data, use only when necessary

    Returns:
        np.Array: file
    """
    obs_file = file_name.startswith("obs://")

    ext = osp.splitext(file_name)[-1].lower()
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        with open(file_name, "rb") as f:
            return np.array(Image.open(f), return_dtype)

    elif ext == ".bin" or ext == ".raw":
        return np.frombuffer(open(file_name, "rb").read(), read_dtype)

    elif ext == ".tiff" or ext == ".exr":
        if not obs_file:
            return np.asarray(
                cv2.imread(
                    file_name,
                    (cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR),
                ),
                return_dtype,
            )
        else:
            return np.asarray(
                cv2.imdecode(
                    np.frombuffer(open(file_name, "rb").read(), read_dtype),
                    (cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR),
                ),
                return_dtype,
            )

    elif ext == ".hdf5" or ext == ".h5":
        if not obs_file:
            result = h5py.File(file_name, "r")
            result = result.get("depth", result.get("dataset"))[:]
        else:
            with open(file_name, "rb") as f:
                result = h5py.File(f, "r")
                result = result.get("depth", result.get("dataset"))[:]
        return np.asarray(result, return_dtype)

    elif ext == ".npy" or ext == ".npz":
        return np.load(file_name)

    elif ext == ".pfm":
        return np.asarray(read_pfm_file(file_name))

    raise NotImplementedError(f"Unsupported File: {file_name}")

def read_pfm_file(file_path):
    with open(file_path, "rb") as file:
        header = file.readline().decode("UTF-8").strip()

        if header == "PF":
            is_color = True
        elif header == "Pf":
            is_color = False
        else:
            raise ValueError("The provided file is not a valid PFM file.")

        dimensions = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("UTF-8"))
        if dimensions:
            img_width, img_height = map(int, dimensions.groups())
        else:
            raise ValueError("Invalid PFM header format.")

        endian_scale = float(file.readline().decode("UTF-8").strip())
        if endian_scale < 0:
            dtype = "<f"  # little-endian
        else:
            dtype = ">f"  # big-endian

        data_buffer = file.read()
        img_data = np.frombuffer(data_buffer, dtype=dtype)

        if is_color:
            img_data = np.reshape(img_data, (img_height, img_width, 3))
        else:
            img_data = np.reshape(img_data, (img_height, img_width))

        img_data = cv2.flip(img_data, 0)

    return img_data

def parse_colmap_to_dict(images_txt_path, cameras_txt_path):
    """
    Parse COLMAP format images.txt and cameras.txt files and return a dictionary
    mapping image names (without .jpg) to (intrinsics, extrinsics) tuples.

    Args:
        images_txt_path (str): Path to COLMAP images.txt file
        cameras_txt_path (str): Path to COLMAP cameras.txt file

    Returns:
        dict: {image_name_without_jpg: (intrinsics_3x3, extrinsics_3x4)}
    """
    # First, parse cameras.txt to get intrinsics for each camera ID
    cameras = {}
    with open(cameras_txt_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]

            # Create intrinsic matrix based on camera model
            if model == "SIMPLE_PINHOLE":
                # SIMPLE_PINHOLE: f, cx, cy
                f, cx, cy = params
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            elif model == "PINHOLE":
                # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = params
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            elif model == "SIMPLE_RADIAL":
                # SIMPLE_RADIAL: f, cx, cy, k1
                f, cx, cy, k1 = params
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            else:
                raise ValueError(f"Unsupported camera model: {model}")

            cameras[camera_id] = K

    # Parse images.txt to get extrinsics and map to camera IDs
    result_dict = {}
    with open(images_txt_path) as f:
        lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Skip comments and empty lines
            if line.startswith("#") or not line:
                i += 1
                continue

            # Parse image line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            parts = line.split()
            if len(parts) >= 10:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                image_name = parts[9]

                # Convert quaternion to rotation matrix
                # Normalize quaternion
                q_norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                qw, qx, qy, qz = qw / q_norm, qx / q_norm, qy / q_norm, qz / q_norm

                # Quaternion to rotation matrix
                R = np.array(
                    [
                        [
                            1 - 2 * (qy * qy + qz * qz),
                            2 * (qx * qy - qw * qz),
                            2 * (qx * qz + qw * qy),
                        ],
                        [
                            2 * (qx * qy + qw * qz),
                            1 - 2 * (qx * qx + qz * qz),
                            2 * (qy * qz - qw * qx),
                        ],
                        [
                            2 * (qx * qz - qw * qy),
                            2 * (qy * qz + qw * qx),
                            1 - 2 * (qx * qx + qy * qy),
                        ],
                    ]
                )

                # Translation vector
                t = np.array([[tx], [ty], [tz]])

                # Create extrinsics matrix [R|t]
                extrinsics = np.hstack([R, t])

                # Get intrinsics from camera ID
                intrinsics = cameras[camera_id]

                # Remove .jpg/.JPG extension from image name
                name_without_ext = image_name.replace(".jpg", "").replace(".JPG", "")

                result_dict[name_without_ext] = (intrinsics, extrinsics)

            # Skip the next line (POINTS2D data)
            i += 2

    return result_dict