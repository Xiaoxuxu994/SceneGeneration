# benchmark/datasets/sevenscenes.py
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

from lib.load_fn import load_and_preprocess_images
from lib.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch, DatasetConfig

class SevenScenesDataset(BaseDataset):
    """
    7Scenes:
      7Scenes/data/xxx/
        - seq-xx
            - RGB:   frame-000000.color.png
            - Depth: frame-000000.depth.proj.png  (uint16/float from png)  depth(mm)
            - Pose:  frame-000000.pose.txt        (c2w 4x4)
            - K:     固定 (fx,fy,cx,cy) 
        - TestSplit.txt
        - TrainSplit.txt
    """

    # NOTE this is hard-coded and kept identical to pi3
    # please check https://github.com/yyfz/Pi3/blob/evaluation/datasets/sevenscenes.py#L130
    FX, FY, CX, CY = 525.0, 525.0, 320.0, 240.0
    DEPTH_SCALE = 1000.0     # mm -> m
    INVALID_DEPTH_RAW = 60000.0  
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def list_sequences(self, split: str) -> List[str]:
        split_txt = f"{split.capitalize()}Split.txt"
        scenes = sorted([d for d in os.listdir(self.config.data_root) if osp.isdir(osp.join(self.config.data_root, d))])

        seqs = []
        for scene in scenes:
            txt = osp.join(self.config.data_root, scene, split_txt)
            if not osp.exists(txt):
                continue
            with open(txt, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            for raw in lines:
                if raw.startswith("sequence"):
                    sid = int(raw[len("sequence"):])
                    seqs.append(f"{scene}/seq-{sid:02d}")
                else:
                    seqs.append(f"{scene}/{raw}")
        return seqs

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.config.data_root, seq)

        def list_suffix(suffix: str):
            return sorted([d for d in os.listdir(seq_path) if d.endswith(suffix)])

        rgbs = list_suffix(".color.png")
        depths = list_suffix(".depth.proj.png")
        poses = list_suffix(".pose.txt")

        if not (len(rgbs) == len(depths) == len(poses)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={len(poses)}")
        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.config.data_root, seq)
        images = [osp.join(seq_dir, f"frame-{i:06d}.color.png") for i in frame_ids]
        poses  = [osp.join(seq_dir, f"frame-{i:06d}.pose.txt")  for i in frame_ids]
        depths = [osp.join(seq_dir, f"frame-{i:06d}.depth.proj.png") for i in frame_ids]
        return images, poses, depths

    # ------- dataset specific loaders -------
    def _load_rgb(self, image_paths: List[str]):
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

    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        d = np.stack([imageio.v2.imread(p).astype(np.float32) for p in depth_paths])  # (V,H,W)
        d[d >= self.INVALID_DEPTH_RAW] = 0.0
        d = d / self.config.depth_scale
        depths = torch.from_numpy(d)
        
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

    def _load_w2c(self, pose_paths: List[str]) -> torch.Tensor:
        # pose.txt is c2w (4x4)
        c2w = np.stack([np.loadtxt(p) for p in pose_paths])  # (V,4,4)
        c2w = torch.from_numpy(c2w)
        w2c = closed_form_inverse_se3(c2w)  # (V,4,4)
        return w2c

    def _load_K(self, num_frames: int) -> torch.Tensor:
        K = np.array([[self.config.fx, 0.0, self.config.cx],
                      [0.0, self.config.fy, self.config.cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        pix_K = torch.from_numpy(K).view(1, 3, 3).repeat(num_frames, 1, 1)
        
        pix_W, pix_H = pix_K[0, 0, 2] * 2, pix_K[0, 1, 2] * 2
        norm_K = self._normalize_K(pix_K, pix_H, pix_W)
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
        depths = self._load_depth(depth_paths)               # (V,H,W) meters
        raw_w2c = self._load_w2c(pose_paths)                 # (V,4,4)
        norm_K, pix_K = self._load_K(num_frames=len(frame_ids))      # (V,3,3)

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
                "dataset": "7scenes",
                "FXFY_CXCY": (self.config.fx, self.config.fy, self.config.cx, self.config.cy),
                "DEPTH_SCALE": self.config.depth_scale
            },
        )
