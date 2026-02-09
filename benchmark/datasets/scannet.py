# benchmark/datasets/scannet_posed_images.py
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

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch

class ScanNetDataset(BaseDataset):
    """
    ScanNet:
      scannet/posed_images/
        <scene_id>/
          color/{id}.jpg
          depth/{id}.png
          pose/{id}.txt
          intrinsic/intrinsic_color.txt

    Notes:
      - depth.png is uint16 in millimeters -> /1000 -> meters
      - intrinsic: intrinsic/intrinsic_depth.txt (use its top-left 3x3) 
        * intrinsic should be align with depth, resize_image_depth_and_intrinsic use depth HW
        * for scannet, image res != depth res
      - pose/{id}.txt is typically c2w (4x4). return w2c via inverse (same as 7scenes).
    """

    DEPTH_SCALE = 1000.0  # mm -> m

    def __init__(self, root: str | Path,
                 depth_scale: float = DEPTH_SCALE,):
        super().__init__(root)
        self.depth_scale = depth_scale

    def list_sequences(self, split: str = "all") -> List[str]:
        return sorted([d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))])

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.root, seq)

        def list_suffix(prefix: str, suffix: str):
            return sorted([d for d in os.listdir(os.path.join(seq_path,prefix)) if d.endswith(suffix)])

        rgbs = list_suffix(prefix="color",suffix=".jpg")
        depths = list_suffix(prefix="depth",suffix=".png")
        poses = list_suffix(prefix="pose",suffix=".txt")

        if not (len(rgbs) == len(depths) == len(poses)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={len(poses)}")
        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.root, seq)
        images = [osp.join(seq_dir, "color", f"{i}.jpg") for i in frame_ids]
        depths = [osp.join(seq_dir, "depth", f"{i}.png") for i in frame_ids]
        poses  = [osp.join(seq_dir, "pose",  f"{i}.txt") for i in frame_ids]
        return images, poses, depths

    def _load_rgb(self, image_paths: List[str], target_width=518, target_height=392, border=15):
        # image_pils = [Image.open(p).convert("RGB") for p in image_paths]
        # images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        
        # crop input images
        image_pils, images_cpu = [], []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            left, top, right, bottom = int(border), int(border), int(width-border), int(height-border)
            img = img.crop((left, top, right, bottom))        
            img_rs = img.resize((target_width, target_height), Image.Resampling.BICUBIC)
            image_pils.append(img_rs)
            img_tensor = TF.ToTensor()(img_rs)
            images_cpu.append(img_tensor)
            
        images_cpu = torch.stack(images_cpu)
        
        return image_pils, images_cpu

    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        d = np.stack([imageio.v2.imread(p).astype(np.float32) for p in depth_paths])  # (V,H,W)
        d = d / self.depth_scale  # mm -> m
        return torch.from_numpy(d)  # (V,H,W) meters

    def _load_w2c(self, pose_paths: List[str]) -> torch.Tensor:
        # pose.txt is c2w (4x4)
        c2w = np.stack([np.loadtxt(p) for p in pose_paths])  # (V,4,4)
        c2w = torch.from_numpy(c2w)
        w2c = closed_form_inverse_se3(c2w)  # (V,4,4)
        return w2c

    def _load_K(self, seq: str, num_frames: int) -> torch.Tensor:
        intr_path = osp.join(self.root, seq, "intrinsic", "intrinsic_depth.txt")
        intr = np.loadtxt(intr_path).astype(np.float32)
        K = intr[:3, :3].astype(np.float32)
        return torch.from_numpy(K).view(1, 3, 3).repeat(num_frames, 1, 1)

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)                     # (V,H,W) meters, no resize
        raw_w2c = self._load_w2c(pose_paths)                       # (V,4,4)
        raw_K = self._load_K(seq, num_frames=len(frame_ids))       # (V,3,3)

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
                "dataset": "scannet",
                "DEPTH_SCALE": self.DEPTH_SCALE,
            },
        )
