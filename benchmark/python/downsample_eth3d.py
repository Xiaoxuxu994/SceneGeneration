import os
import os.path as osp

import json
import imageio

import torch
from torchvision import transforms as TF

import numpy as np

import cv2

from pathlib import Path
from PIL import Image

from typing import List

ROOT = Path("./data/ETH3D/multi_view_training_dslr/")
W, H = 600, 400

def get_sequence_list(root=ROOT):
    return sorted([d for d in root.iterdir() if d.is_dir()])

def make_path(sequence_list: List[Path]):
    src_paths, tar_paths = {}, {}
    for seq in sequence_list:
        seq_name = seq.name
        src_paths[seq_name] = {"rgb": [], "depth": []}
        src_rgb_dir = seq / "images/dslr_images_undistorted"
        src_depth_dir = seq / "gt_depth/undistorted"
        tar_paths[seq_name] = {"rgb": [], "depth": []}
        tar_rgb_dir = seq / f"images/dslr_images_undistorted_{W}x{H}"
        tar_depth_dir = seq / f"gt_depth/undistorted_{W}x{H}"
        # tar_rgb_dir.mkdir(exist_ok=True)
        # tar_depth_dir.mkdir(exist_ok=True)
        for src_rgb_path in sorted(src_rgb_dir.iterdir()):
            rgb_name = src_rgb_path.name
            src_paths[seq_name]["rgb"].append(str(src_rgb_path))
            tar_paths[seq_name]["rgb"].append(str(tar_rgb_dir / rgb_name))
            
        for src_depth_path in sorted(src_depth_dir.iterdir()):
            depth_name = src_depth_path.name
            src_paths[seq_name]["depth"].append(str(src_depth_path))
            tar_paths[seq_name]["depth"].append(str(tar_depth_dir / depth_name))
            
    return src_paths, tar_paths

def downsample_rgb(sequence_list: List[Path]):
    for seq in sequence_list:
        src_dir = seq / "images/dslr_images_undistorted"
        tar_dir = seq / f"images/dslr_images_undistorted_{W}x{H}"
        tar_dir.mkdir(exist_ok=True)
        for src_path in sorted(src_dir.iterdir()):
            src_name = src_path.name
            tar_path = tar_dir / src_name
            
            img = Image.open(src_path).convert("RGB")
            img = img.resize((W, H))
            img.save(tar_path)
            
def resize_sparse_depth(depth_path, target_width=600, target_height=400):
    # 1. Load image (using cv2.IMREAD_UNCHANGED to preserve 16-bit or float data)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 2. Create a binary mask where depth is valid (1.0) and invalid (0.0)
    valid_mask = (depth > 0).astype(np.float32)
    
    # depth_path = Path(depth_path)
    # mask_dir = depth_path.parent.parent / "valid_mask"
    # mask_dir.mkdir(exist_ok=True, parents=True)
    # mask_path = mask_dir / "{}.png".format(depth_path.stem)
    
    # from PIL import Image
    # img = valid_mask.astype(np.uint8) * 255
    # img = Image.fromarray(img, mode="L")
    # img.save(mask_path)

    # 3. Downsample both the depth map and the mask using AREA interpolation.
    # cv2.INTER_AREA is ideal for shrinking images as it integrates pixel values
    # over the entire pixel area, preventing aliasing.
    target_size = (target_width, target_height)
    downscaled_depth_sum = cv2.resize(depth, target_size, interpolation=cv2.INTER_AREA)
    downscaled_mask_sum = cv2.resize(
        valid_mask, target_size, interpolation=cv2.INTER_AREA
    )

    # 4. Divide the downscaled depth by the downscaled mask.
    # This effectively computes the average of ONLY the valid pixels that
    # fell into each target pixel's receptive field.
    resized_depth = np.divide(
        downscaled_depth_sum,
        downscaled_mask_sum,
        out=np.zeros_like(downscaled_depth_sum),
        where=downscaled_mask_sum > 0,
    )

    return resized_depth
            
def downsample_depth(sequence_list: List[Path]):
    for seq in sequence_list:
        src_dir = seq / "gt_depth/undistorted"
        tar_dir = seq / f"gt_depth/undistorted_{W}x{H}"
        tar_dir.mkdir(exist_ok=True)
        for src_path in sorted(src_dir.iterdir()):
            src_name = src_path.name
            tar_path = tar_dir / src_name
            
            # depth = imageio.v2.imread(src_path).astype(np.float32)
            # depth = np.array(depth).astype(np.float32)
            # depth = torch.from_numpy(depth).unsqueeze(0)
            
            # # resize to match rgb
            # depth = TF.functional.resize(
            #     depth,
            #     size=(H, W), 
            #     interpolation=TF.InterpolationMode.NEAREST
            # )
            # depth = depth[0].numpy().astype(np.uint16)
            
            depth = resize_sparse_depth(str(src_path), W, H).astype(np.uint16)
            
            img = Image.new("I", depth.T.shape)
            img.frombytes(depth.tobytes(), 'raw', "I;16")
            img.save(tar_path)

if __name__ == "__main__":
    sequence_list = get_sequence_list()
    # src_paths, tar_paths = make_path(sequence_list)
    # with open("original_paths.json", "w") as f:
    #     json.dump(src_paths, f, indent=4)
        
    # with open("downsampled_paths.json", "w") as f:
    #     json.dump(tar_paths, f, indent=4)
    # downsample_rgb(sequence_list)
    downsample_depth(sequence_list)
    