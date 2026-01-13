# benchmark/datasets/dtu.py
from __future__ import annotations
import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import cv2
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images

from .base import BaseDataset, SequenceBatch


# -----------------------------
# copy from pi3
# -----------------------------
def load_cam_mvsnet(words, interval_scale=1):
    """read camera txt file"""
    '''
    cam[1][3][0]: depth_min
    cam[1][3][1]: depth_interval（乘上 interval_scale）
    cam[1][3][2]: num_depth（深度平面数，常见 192）
    cam[1][3][3]: depth_max（有时直接给，有时由 min + interval * num 计算）
    '''

    cam = np.zeros((2, 4, 4))
    # words = file.read().split()
    words = words.split()
    # read extrinsic
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = 192
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    extrinsic = cam[0].astype(np.float32)
    intrinsic = cam[1].astype(np.float32)

    return intrinsic, extrinsic


# -----------------------------
# dataset
# -----------------------------
class DTUDataset(BaseDataset):
    """
    DTU:
      DTU/spann3r_version/dtu_test_mvsnet_release/scanxxx
        images/00000000.jpg
        depths/00000000.npy
        binary_masks/00000000.png
        cams/00000000_cam.txt
        00x_pcd.ply
        pair.txt

    Notes:
      - Depth:
          depth npy is in mm -> /1000 to meters,
          apply binary mask,
          depth related param & extrinsic mm -> /1000 to meters
    """
    DEPTH_SCALE = 1000.0  # mm -> m

    TEST_SCANS = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]
    VALID_SCANS = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
    TRAIN_SCANS = [
        2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
        45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
        74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
        121, 122, 123, 124, 125, 126, 127, 128
    ]

    def __init__(
        self,
        root: str | Path,
        split: str = "test",
        interval_scale: float = 1.0,
        depth_scale: float = DEPTH_SCALE,
    ):
        super().__init__(root)
        self.split = split
        self.interval_scale = float(interval_scale)
        self.depth_scale = depth_scale

    def list_sequences(self, split: str) -> List[str]:
        split = split.lower()
        if split == "test":
            seq_numbers = self.TEST_SCANS
        elif split in ("val", "valid", "validation"):
            seq_numbers = self.VALID_SCANS
        elif split == "train":
            seq_numbers = self.TRAIN_SCANS
        else:
            raise ValueError(f"Invalid split: {split} (use train/valid/test)")
        return [f"scan{n}" for n in seq_numbers]

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.root, seq)

        def list_suffix(prefix: str, suffix: str):
            return sorted([d for d in os.listdir(os.path.join(seq_path,prefix)) if d.endswith(suffix)])

        rgbs = list_suffix(prefix="images",suffix=".jpg")
        depths = list_suffix(prefix="depths",suffix=".npy")
        poses = list_suffix(prefix="cams",suffix="_cam.txt")

        if not (len(rgbs) == len(depths) == len(poses)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={len(poses)}")
        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.root, seq)
        images = [osp.join(seq_dir, "images" ,f"{i:08d}.jpg") for i in frame_ids]
        depths = [osp.join(seq_dir, "depths", f"{i:08d}.npy") for i in frame_ids]
        poses = [osp.join(seq_dir, "cams", f"{i:08d}_cam.txt") for i in frame_ids]
        masks = [osp.join(seq_dir, "binary_masks", f"{i:08d}.png") for i in frame_ids]
        return images, poses, depths, masks

    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p).convert("RGB") for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu

    def _load_depth(self, depth_paths: List[str], mask_paths: List[str]) -> torch.Tensor:
        kernel = np.ones((10, 10), np.uint8) # Define the erosion kernel as pi3
 
        depths = []
        for dp, mp in zip(depth_paths, mask_paths):
            if (not osp.exists(dp)) or (not osp.exists(mp)):
                raise FileNotFoundError(f"Missing depth/mask:\n{dp}\n{mp}")

            depth = np.load(dp).astype(np.float32)
            depth /= self.depth_scale # mm->m
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED) / 255.0
            mask = mask.astype(np.float32)

            mask[mask > 0.5] = 1.0
            mask[mask < 0.5] = 0.0

            # align mask to depth resolution
            mask = cv2.resize(mask, 
                              (depth.shape[1], depth.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
            mask = cv2.erode(mask, kernel, iterations=1)
            depth = depth * mask.astype(np.float32)
            depths.append(depth)

        return torch.from_numpy(np.stack(depths, axis=0))  # (V,H,W)
        
    def _load_w2c_K(self, pose_paths: List[str]) -> torch.Tensor:
        w2c_list, K_list = [], []
        for p in pose_paths:
            if not osp.exists(p):
                raise FileNotFoundError(f"Missing cam file: {p}")
            cam_txt = open(p, "r").read()
            intr_4x4, extr_4x4 = load_cam_mvsnet(cam_txt)
            extr_4x4[:3, 3] /= self.depth_scale # mm->m
            intr_4x4[3, [0, 1, 3]] /= self.depth_scale
            w2c_list.append(extr_4x4.astype(np.float64))
            K_list.append(intr_4x4[:3, :3].astype(np.float64))
        return torch.from_numpy(np.stack(w2c_list, axis=0)),torch.from_numpy(np.stack(K_list, axis=0)) # (V,4,4), # (V,3,3) 
    
    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths, mask_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)               # (V,3,*,*) per VGGT
        depths = self._load_depth(depth_paths, mask_paths)                 # (V,H,W) original depth res
        raw_w2c,raw_K = self._load_w2c_K(pose_paths)                       # (V,4,4),(V,3,3)

        return SequenceBatch(
            seq=seq,
            frame_ids=frame_ids,
            image_paths=image_paths,
            pose_paths=pose_paths,
            depth_paths=depth_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            raw_w2c=raw_w2c,
            raw_K=raw_K,
            meta={
                "dataset": "dtu",
            },
        )
