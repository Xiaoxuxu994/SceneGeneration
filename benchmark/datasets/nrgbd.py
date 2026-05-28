# benchmark/datasets/nrgbd.py
from __future__ import annotations
import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import imageio
from PIL import Image
from torchvision import transforms as TF
import cv2

from lib.load_fn import load_and_preprocess_images
from lib.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch, DatasetConfig


def _read_poses_txt_as_4x4(pose_txt: str) -> np.ndarray:
    """Read poses.txt where every 4 lines form a 4x4 matrix. Return (N,4,4) float64."""
    with open(pose_txt, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) % 4 != 0:
        raise ValueError(f"poses.txt lines not multiple of 4: {pose_txt}, got {len(lines)} lines")

    mats = []
    for i in range(0, len(lines), 4):
        block = lines[i:i+4]
        mat = [[float(x) for x in ln.split()] for ln in block]
        mats.append(mat)
    arr = np.array(mats, dtype=np.float64)  # (N,4,4)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"Bad pose shape from {pose_txt}: {arr.shape}")
    return arr


class NRGBDDataset(BaseDataset):
    """
    NRGBD:
      NRGBD/data/<seq>/
        images/img{id}.png
        depth/depth{id}.png
        poses.txt                 # each 4 lines is a 4x4 pose

    Notes:
      - Pose conversion:
          poses.txt gives c2w in GL convention (per your script)
          apply gl->cv: pose[:, :, 1:3] *= -1
          then w2c = inverse(c2w) via closed_form_inverse_se3
      - Depth:
          depth png is in mm -> /1000 to meters
          >10m or <1e-3m treated invalid -> 0
    """

    # # NOTE this is hard-coded and kept identical to pi3
    # # please check https://github.com/yyfz/Pi3/blob/evaluation/datasets/nrgbd.py
    # FX = 554.2562584220408
    # FY = 554.2562584220408
    # CX = 320.0
    # CY = 240.0
    # DEPTH_SCALE = 1000.0  # mm -> m
    # NUM_TEST = 10

    # def __init__(
    #     self,
    #     root: str | Path,
    #     num_seq: int = NUM_TEST,
    #     fx: float = FX, fy: float = FY, cx: float = CX, cy: float = CY,
    #     depth_scale: float = DEPTH_SCALE,
    # ):
    #     super().__init__(root, num_seq)
    #     self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
    #     self.depth_scale = depth_scale

    #     self.sequence_list = sorted([d for d in os.listdir(self.config.data_root) if osp.isdir(osp.join(self.config.data_root, d))])

    #     # cache: dict[seq] -> w2c_all (N,4,4) float64
    #     self._w2c_cache = {}
    #     for seq in self.sequence_list:
    #         seq_dir = osp.join(self.config.data_root, seq)
    #         pose_txt = osp.join(seq_dir, "poses.txt")
    #         if not osp.exists(pose_txt):
    #             raise FileNotFoundError(f"Missing poses.txt: {pose_txt}")

    #         c2w = _read_poses_txt_as_4x4(pose_txt)  # (N,4,4)
    #         # gl -> cv (match your script): flip Y/Z axis columns
    #         c2w[:, :, 1:3] *= -1.0
    #         w2c = closed_form_inverse_se3(torch.from_numpy(c2w)).cpu().numpy().astype(np.float64)  # (N,4,4)
    #         self._w2c_cache[seq] = w2c
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        self.sequence_list = sorted([d for d in os.listdir(self.config.data_root) if osp.isdir(osp.join(self.config.data_root, d))])

        # cache: dict[seq] -> w2c_all (N,4,4) float64
        self._w2c_cache = {}
        for seq in self.sequence_list:
            seq_dir = osp.join(self.config.data_root, seq)
            pose_txt = osp.join(seq_dir, "poses.txt")
            if not osp.exists(pose_txt):
                raise FileNotFoundError(f"Missing poses.txt: {pose_txt}")

            c2w = _read_poses_txt_as_4x4(pose_txt)  # (N,4,4)
            # gl -> cv (match your script): flip Y/Z axis columns
            c2w[:, :, 1:3] *= -1.0
            w2c = closed_form_inverse_se3(torch.from_numpy(c2w)).cpu().numpy().astype(np.float64)  # (N,4,4)
            self._w2c_cache[seq] = w2c

    def list_sequences(self, split: str) -> List[str]:
        # No official split; keep signature consistent
        return self.sequence_list

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.config.data_root, seq)

        def list_suffix(prefix: str, suffix: str):
            return sorted([d for d in os.listdir(os.path.join(seq_path,prefix)) if d.endswith(suffix)])

        rgbs = list_suffix(prefix="images",suffix=".png")
        depths = list_suffix(prefix="depth",suffix=".png")
        pose_txt = osp.join(seq_path, "poses.txt")
        poses = _read_poses_txt_as_4x4(pose_txt) # c2w
        n_pose = poses.shape[0]

        if not (len(rgbs) == len(depths) == n_pose):
            raise ValueError(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={n_pose}")

        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.config.data_root, seq)
        images = [osp.join(seq_dir, "images", f"img{i}.png") for i in frame_ids]
        depths = [osp.join(seq_dir, "depth", f"depth{i}.png") for i in frame_ids]
        # poses are in one file; keep per-frame list for SequenceBatch compatibility
        pose_txt = osp.join(seq_dir, "poses.txt")
        poses = [pose_txt for _ in frame_ids]
        return images, poses, depths

    # def _load_rgb(self, image_paths: List[str]):
    #     image_pils = [Image.open(p).convert("RGB") for p in image_paths]
    #     images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
    #     return image_pils, images_cpu
    
    def _load_rgb(self, image_paths: List[str], 
            target_width=518, target_height=518, border=28):
        """default RGBs in ASE have black corners
            cropping input ASE RGBs from 518x518 to 462x462
        """
        
        # crop input images
        image_pils, images_cpu = [], []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            
            # resize to origin size
            img = img.resize((self.config.rgb_origin_W, self.config.rgb_origin_H), Image.Resampling.BILINEAR)
            
            # border cropping
            if self.config.crop_left + self.config.crop_right + \
                self.config.crop_top + self.config.crop_bottom > 0:
                (left, top, right, bottom) = (
                    int(self.config.crop_left), 
                    int(self.config.crop_top), 
                    int(self.config.rgb_origin_W - self.config.crop_right), 
                    int(self.config.rgb_origin_H - self.config.crop_bottom)
                )
                img = img.crop((left, top, right, bottom))
                
            # resize to target size
            actual_W, actual_H = img.size
            if self.config.rgb_resize_W != actual_W or self.config.rgb_resize_H != actual_H:
                img = img.resize(
                    (self.config.rgb_resize_W, self.config.rgb_resize_H), 
                    Image.Resampling.BILINEAR
                )
    
            image_pils.append(img)
            img_tensor = TF.ToTensor()(img)
            images_cpu.append(img_tensor)
            
        images_cpu = torch.stack(images_cpu)
        
        return image_pils, images_cpu

    # def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
    #     d = np.stack([imageio.v2.imread(p).astype(np.float32) for p in depth_paths])  # (V,H,W)
    #     d = d / self.depth_scale  # mm -> m
    #     return torch.from_numpy(d)  # (V,H,W) meters
    
    # def _load_depth(self, depth_paths: List[str], 
    #         target_width=518, target_height=392) -> torch.Tensor:
        
    #     depths = []
    #     for depth_path in depth_paths:
    #         # loading with PIL has value issues
    #         depth = imageio.v2.imread(depth_path).astype(np.float32)
            
    #         # resize to 518x518
    #         depth = cv2.resize(depth, (target_width, target_height), interpolation = cv2.INTER_NEAREST)
            
    #         # to tensor
    #         depth = np.array(depth).astype(np.float32)
    #         depths.append(depth)
            
    #     depths = np.stack(depths)
    #     depths = depths / self.depth_scale  # mm -> m
    #     depths = torch.from_numpy(depths)
            
    #     return depths # (V,392,518) meters
    
    def _load_depth(self, depth_paths: List[str], 
            target_width=518, target_height=392) -> torch.Tensor:
        
        depths = []
        for depth_path in depth_paths:
            # loading with PIL has value issues
            depth = imageio.v2.imread(depth_path).astype(np.float32)
            depth = np.array(depth).astype(np.float32)
            depths.append(depth)
            
        # to tensor
        depths = np.stack(depths)
        depths = torch.from_numpy(depths)
        depths = depths / self.config.depth_scale  # mm -> m
        
        # resize to match rgb
        depths = TF.functional.resize(
            depths,
            size=(self.config.rgb_origin_H, self.config.rgb_origin_W), 
            interpolation=TF.InterpolationMode.NEAREST
        )
        
        # border cropping
        if self.config.crop_left + self.config.crop_right + \
            self.config.crop_top + self.config.crop_bottom > 0:
            (left, top, right, bottom) = (
                int(self.config.crop_left), 
                int(self.config.crop_top), 
                int(self.config.rgb_origin_W - self.config.crop_right), 
                int(self.config.rgb_origin_H - self.config.crop_bottom)
            )
            # NOTE the depth is (V, H, W)
            depths = depths[:, top:bottom, left:right]
            
        # resize to target size
        _, actual_H, actual_W = depths.shape
        if self.config.rgb_resize_W != actual_W or self.config.rgb_resize_H != actual_H:
            depths = TF.functional.resize(
                depths,
                size=(self.config.rgb_resize_H, self.config.rgb_resize_W), 
                interpolation=TF.InterpolationMode.NEAREST
            )
            
        return depths

    def _load_w2c(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        w2c_all = self._w2c_cache[seq]  # (N,4,4)
        w2c_sel = w2c_all[np.array(frame_ids, dtype=np.int64)]
        return torch.from_numpy(w2c_sel)  # (V,4,4)

    def _load_K(self, num_frames: int) -> torch.Tensor:
        norm_K = np.array(
            [[self.config.fx/(self.config.cx*2), 0.0, 0.5],
            [0.0, self.config.fy/(self.config.cy*2), 0.5],
            [0.0, 0.0, 1.0]], 
            dtype=np.float32
        )
        norm_K = torch.from_numpy(norm_K).view(1, 3, 3).repeat(num_frames, 1, 1)
        
        pix_K = self._denormalize_K(norm_K, self.config.rgb_origin_W, self.config.rgb_origin_H)
        
        if self.config.crop_left + self.config.crop_top > 0:
            pix_K[..., 0, 2] -= self.config.crop_left
            pix_K[..., 1, 2] -= self.config.crop_top
            
        # normalize again
        (actual_H, actual_W) = (
            self.config.rgb_origin_H - self.config.crop_top - self.config.crop_bottom,
            self.config.rgb_origin_W - self.config.crop_left - self.config.crop_right,
        )
        norm_K = self._normalize_K(pix_K, actual_H, actual_W)
            
        if self.config.rgb_resize_W != actual_W or self.config.rgb_resize_H != actual_H:
            pix_K = self._denormalize_K(norm_K, self.config.rgb_resize_W, self.config.rgb_resize_H)
        
        return norm_K, pix_K

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)                 # (V,H,W) meters (no resize)
        raw_w2c = self._load_w2c(seq, frame_ids)               # (V,4,4)
        norm_K, pix_K = self._load_K(num_frames=len(frame_ids))        # (V,3,3)

        return SequenceBatch(
            seq=seq,
            frame_ids=frame_ids,
            image_paths=image_paths,
            pose_paths=pose_paths,
            depth_paths=depth_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            tracks={},
            raw_w2c=raw_w2c,
            raw_K=pix_K,
            norm_K=norm_K,
            meta={
                "dataset": "nrgbd",
                "FXFY_CXCY": (self.config.fx, self.config.fy, self.config.cx, self.config.cy),
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )
