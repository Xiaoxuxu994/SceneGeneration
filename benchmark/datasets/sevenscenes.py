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

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch

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

    def __init__(self, root: str | Path,
                 fx: float = FX, fy: float = FY, cx: float = CX, cy: float = CY,
                 depth_scale: float = DEPTH_SCALE,
                 invalid_depth_raw: float = INVALID_DEPTH_RAW):
        super().__init__(root)
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.depth_scale = depth_scale
        self.invalid_depth_raw = invalid_depth_raw

    def list_sequences(self, split: str) -> List[str]:
        split_txt = f"{split.capitalize()}Split.txt"
        scenes = sorted([d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))])

        seqs = []
        for scene in scenes:
            txt = osp.join(self.root, scene, split_txt)
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
        seq_path = osp.join(self.root, seq)

        def list_suffix(suffix: str):
            return sorted([d for d in os.listdir(seq_path) if d.endswith(suffix)])

        rgbs = list_suffix(".color.png")
        depths = list_suffix(".depth.proj.png")
        poses = list_suffix(".pose.txt")

        if not (len(rgbs) == len(depths) == len(poses)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={len(poses)}")
        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.root, seq)
        images = [osp.join(seq_dir, f"frame-{i:06d}.color.png") for i in frame_ids]
        poses  = [osp.join(seq_dir, f"frame-{i:06d}.pose.txt")  for i in frame_ids]
        depths = [osp.join(seq_dir, f"frame-{i:06d}.depth.proj.png") for i in frame_ids]
        return images, poses, depths

    # ------- dataset specific loaders -------
    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p) for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu

    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        d = np.stack([imageio.v2.imread(p).astype(np.float32) for p in depth_paths])  # (V,H,W)
        d[d >= self.invalid_depth_raw] = 0.0
        d = d / self.depth_scale
        return torch.from_numpy(d)

    def _load_w2c(self, pose_paths: List[str]) -> torch.Tensor:
        # pose.txt is c2w (4x4)
        c2w = np.stack([np.loadtxt(p) for p in pose_paths])  # (V,4,4)
        c2w = torch.from_numpy(c2w)
        w2c = closed_form_inverse_se3(c2w)  # (V,4,4)
        return w2c

    def _load_K(self, num_frames: int) -> torch.Tensor:
        K = np.array([[self.fx, 0.0, self.cx],
                      [0.0, self.fy, self.cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        K = torch.from_numpy(K).view(1, 3, 3).repeat(num_frames, 1, 1)
        return K

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)               # (V,H,W) meters
        raw_w2c = self._load_w2c(pose_paths)                 # (V,4,4)
        raw_K = self._load_K(num_frames=len(frame_ids))      # (V,3,3)

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
                "dataset": "7scenes",
                "FXFY_CXCY": (self.FX, self.FY, self.CX, self.CY),
                "DEPTH_SCALE": self.depth_scale,},
        )
