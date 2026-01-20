# benchmark/datasets/ase.py
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

class ASEDataset(BaseDataset):
    """
    ASE:
      NRGBD/<seq>/
        undistort_rgb/vignette{id}.jpg # id is zero-padded 7 digits
        undistort_depth/depth{id}.png  # id is zero-padded 7 digits
        trajectory.csv                 # each line is a 3-dim T & 4-dim quaterion
        
    or simply input the sample sequence list:
        [
            {
                "input_images": <path>,    # read this
                "input_depths": <path>,    # read this
                "input_depth_masks": None,
                "input_extrs": np.array,
                "input_intrs": np.array,
            }
        ]
    """
    
    DEPTH_SCALE = 1000.0  # mm -> m
    NUM_TEST = 10
    
    def __init__(self, root: str | Path, depth_scale: float = DEPTH_SCALE):
        super().__init__(root)
        
        self.depth_scale = depth_scale
        
        if root.endswith(".npz"):
            self.sample_sequences = np.load(root, allow_pickle=True)["data"][-self.NUM_TEST:]
        else:
            raise NotImplementedError
        
    def list_sequences(self, split):
        return [str(i) for i in range(len(self.sample_sequences))]
    
    def count_seq_len(self, seq):
        seq = int(seq)
        rgbs = self.sample_sequences[seq]["input_images"]
        depths = self.sample_sequences[seq]["input_depths"]
        poses = self.sample_sequences[seq]["input_extrs"]
        
        if not (len(rgbs) == len(depths) == len(poses)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={len(poses)}")
        return len(rgbs)
    
    def make_paths(self, seq, frame_ids):
        seq = int(seq)
        images = [self.sample_sequences[seq]["input_images"][i].replace("efs-mi-east4-3", "/efs-m3ga") for i in frame_ids]
        depths = [self.sample_sequences[seq]["input_depths"][i].replace("efs-mi-east4-3", "/efs-m3ga") for i in frame_ids]
        
        return images, depths
    
    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p).convert("RGB") for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu
    
    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        d = np.stack([imageio.v2.imread(p).astype(np.float32) for p in depth_paths])  # (V,H,W)
        d = d / self.depth_scale  # mm -> m
        return torch.from_numpy(d)  # (V,H,W) meters
    
    def _load_w2c(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        seq = int(seq)
        poses = np.stack([self.sample_sequences[seq]["input_extrs"][i] for i in frame_ids])
        
        return torch.from_numpy(poses)
    
    def _load_K(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        seq = int(seq)
        K = np.stack([self.sample_sequences[seq]["input_intrs"][i] for i in frame_ids])
    
        return torch.from_numpy(K)
    
    def load_sequence(self, seq, frame_ids) -> SequenceBatch:
        image_paths, depth_paths = self.make_paths(seq, frame_ids)
        
        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)     # (V,H,W) meters, no resize
        raw_w2c = self._load_w2c(seq, frame_ids)   # (V,4,4)
        raw_K = self._load_K(seq, frame_ids)       # (V,3,3)
        
        return SequenceBatch(
            seq=seq,
            frame_ids=frame_ids,
            image_paths=image_paths,
            pose_paths=image_paths,
            depth_paths=depth_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            raw_w2c=raw_w2c,
            raw_K=raw_K,
            meta={
                "dataset": "ase",
                "DEPTH_SCALE": self.DEPTH_SCALE,
            },
        )