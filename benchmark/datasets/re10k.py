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

from lib.load_fn import load_and_preprocess_images
from lib.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch

class RE10KDataset(BaseDataset):
    """
    input the sample sequence list:
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
    
    def __init__(self, root: str | Path, num_seq: int = NUM_TEST, depth_scale: float = DEPTH_SCALE):
        super().__init__(root, num_seq)
        
        self.depth_scale = depth_scale
        
        if root.endswith(".npz"):
            self.sample_sequences = np.load(root, allow_pickle=True)["data"][-self.num_seq:]
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
    
    def _load_rgb(self, image_paths: List[str], target_width=518, target_height=294):
        image_pils = [Image.open(p).convert("RGB").resize((target_width, target_height), Image.Resampling.BICUBIC) \
            for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu
    
    def _load_depth(self, images_cpu: torch.Tesnor) -> torch.Tensor:
        V,C,H,W = images_cpu.shape
        return torch.zeros((V, H, W), dtype=torch.float32)
    
    def _load_w2c_and_K(self, camera_paths: List[str], frame_ids: List[int], 
            images_cpu: torch.Tensor) -> torch.Tensor:
        
        cam_info = np.load(camera_paths, allow_pickle=True).item()
        
        # load extrinsics
        # NOTE the original data is OpenCV's camera-to-world
        poses = np.stack([cam_info["extr_mat"][i] for i in frame_ids]).astype(float)
        poses = torch.from_numpy(poses) # 4,4
        poses = closed_form_inverse_se3(poses) # convert to world-to-camera
        
        # denormalize intrinsics
        V,C,H,W = images_cpu.shape
        K = []
        for i in frame_ids:
            intr = np.array(cam_info["intr_mat"][i]).astype(float) # normalized intrinsics
            intr = torch.from_numpy(intr)
            
            intr_denorm = intr.clone()
            intr_denorm[0, 0] = intr[0, 0] / (intr[0, 2] * 2) * W
            intr_denorm[0, 2] = intr[0, 2] / (intr[0, 2] * 2) * W
            intr_denorm[1, 1] = intr[1, 1] / (intr[1, 2] * 2) * H
            intr_denorm[1, 2] = intr[1, 2] / (intr[1, 2] * 2) * H
            
            K.append(intr_denorm)
            
        K = torch.stack(K)
        
        return poses, K
    
    def load_sequence(self, seq, frame_ids) -> SequenceBatch:
        image_paths, camera_paths = self.make_paths(seq, frame_ids)
        
        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(images_cpu)     # (V,H,W) meters, no resize
        raw_w2c, raw_K = self._load_w2c_and_K(camera_paths, frame_ids, images_cpu)   # (V,4,4)
        
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
                "dataset": "re10k",
                "has_gt_depth": False,
            },
        )