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

class NewDL3DVDataset(BaseDataset):
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
    
    DEPTH_SCALE = 1000.0  # mm -> m, NOTE there's no GT depth
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
        seq_root = Path(self.sample_sequences[seq].replace("efs-mi-east4-3", "efs-m3ga"))
        rgbs = seq_root / "rgb"
        
        return len(list(rgbs.iterdir()))
    
    def make_paths(self, seq, frame_ids):
        seq = int(seq)
        seq_root = Path(self.sample_sequences[seq].replace("efs-mi-east4-3", "efs-m3ga"))
        rgbs = list(sorted((seq_root / "rgb").iterdir()))
        images = [rgbs[i] for i in frame_ids]
        cameras = str(seq_root / "{}.npy".format(str(seq_root.name)))
        
        return images, cameras
    
    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p).convert("RGB") for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu
    
    def _load_depth(self, image_pils: List[Image.Image]) -> torch.Tensor:
        W, H = image_pils[0].size
        V = len(image_pils)
        return torch.zeros((V, H, W), dtype=torch.float32)
    
    def _load_w2c_and_K(self, camera_paths: List[str], frame_ids: List[int]) -> torch.Tensor:
        cam_info = np.load(camera_paths, allow_pickle=True).item()
        poses = np.stack([cam_info["extr_mat"][i] for i in frame_ids]).astype(np.float)
        K = np.stack([cam_info["intr_mat"][i] for i in frame_ids]).astype(np.float)
        
        return torch.from_numpy(poses), torch.from_numpy(K)
    
    def load_sequence(self, seq, frame_ids) -> SequenceBatch:
        image_paths, camera_paths = self.make_paths(seq, frame_ids)
        
        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(image_pils)     # (V,H,W) meters, no resize
        raw_w2c, raw_K = self._load_w2c_and_K(camera_paths, frame_ids)   # (V,4,4)
        
        return SequenceBatch(
            seq=seq,
            frame_ids=frame_ids,
            image_paths=image_paths,
            pose_paths=image_paths,
            depth_paths=image_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            raw_w2c=raw_w2c,
            raw_K=raw_K,
            meta={
                "dataset": "dl3dv_new",
                "DEPTH_SCALE": self.DEPTH_SCALE,
            },
        )