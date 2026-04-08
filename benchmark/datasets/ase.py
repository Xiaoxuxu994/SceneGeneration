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
from torchvision import transforms as TF
import cv2

from .base import BaseDataset, SequenceBatch

from lib.utils import rdepth_to_zdepth
from lib.matching import *

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
    NUM_TEST = 50
    RGB_WIDTH = 518
    RGB_HEIGHT = 518
    CROP_BORDER = 28
    
    def __init__(self, root: str | Path, num_seq: int = NUM_TEST, depth_scale: float = DEPTH_SCALE):
        super().__init__(root, num_seq)
        
        self.depth_scale = depth_scale
        
        if root.endswith(".npz"):
            self.sample_sequences = np.load(root, allow_pickle=True)
        else:
            raise NotImplementedError
        
    def list_sequences(self, split):
        return self.sample_sequences.files[-self.num_seq:]
    
    def count_seq_len(self, seq):
        sequence = self.sample_sequences[seq].item()
        rgbs = sequence["input_images"]
        depths = sequence["input_depths"]
        poses = sequence["input_extrs"]
        
        if not (len(rgbs) == len(depths) == len(poses)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={len(poses)}")
        return len(rgbs)
    
    def make_paths(self, seq, frame_ids):
        sequence = self.sample_sequences[seq].item()
        images = [sequence["input_images"][i].replace("efs-mi-east4-3", "/efs-m3ga") for i in frame_ids]
        depths = [sequence["input_depths"][i].replace("efs-mi-east4-3", "/efs-m3ga") for i in frame_ids]
        
        return images, depths
    
    def _load_rgb(self, image_paths: List[str], 
            target_width=518, target_height=518, border=28):
        """default RGBs in ASE have black corners
            cropping input ASE RGBs from 518x518 to 462x462
        """
        
        # crop input images
        image_pils, images_cpu = [], []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            
            # resize to 518x518
            img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
            
            # crop to 462x462
            left, top, right, bottom = int(border), int(border), int(target_width-border), int(target_height-border)
            img = img.crop((left, top, right, bottom))
            
            # resize back to 518x518
            img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)
    
            image_pils.append(img)
            img_tensor = TF.ToTensor()(img)
            images_cpu.append(img_tensor)
            
        images_cpu = torch.stack(images_cpu)
        
        return image_pils, images_cpu
    
    def _load_depth(self, depth_paths: List[str], K_norm: torch.Tensor, 
            target_width=518, target_height=518, border=28, restore_size=True) -> torch.Tensor:
        """default RGBs in ASE have black corners
            cropping GT depths to match RGBs from 518x518 to 462x462
        """
        
        depths = []
        for depth_path in depth_paths:
            # loading with PIL has value issues
            depth = imageio.v2.imread(depth_path).astype(np.float32)
            
            # resize to 518x518
            depth = cv2.resize(depth, (target_width, target_height), interpolation = cv2.INTER_NEAREST)
            
            # to tensor
            depth = np.array(depth).astype(np.float32)
            depths.append(depth)
            
        depths = np.stack(depths)
        depths = depths / self.depth_scale  # mm -> m
        depths = torch.from_numpy(depths)
        depths = rdepth_to_zdepth(depths.unsqueeze(1), K_norm).squeeze(1) # 518x518
        
        # crop to 462x462
        left, top, right, bottom = int(border), int(border), int(target_width-border), int(target_height-border)
        depths = depths[:, left:right, top:bottom]
        
        # resize back to 518x518
        depths = TF.functional.resize(
            depths,
            size=(target_height, target_width), 
            interpolation=TF.InterpolationMode.NEAREST
        )
            
        return depths # (V,518,518) meters
    
    def _load_w2c(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        sequence = self.sample_sequences[seq].item()
        poses = np.stack([sequence["input_extrs"][i] for i in frame_ids])
        
        return torch.from_numpy(poses)
    
    def _load_K(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        sequence = self.sample_sequences[seq].item()
        K = np.stack([sequence["input_intrs"][i] for i in frame_ids])
    
        return torch.from_numpy(K).float()
    
    def load_sequence(self, seq, frame_ids) -> SequenceBatch:
        image_paths, depth_paths = self.make_paths(seq, frame_ids)
        
        image_pils, images_cpu = self._load_rgb(image_paths) # (V,518,518)
        raw_w2c = self._load_w2c(seq, frame_ids)             # (V,4,4)
        raw_K = self._load_K(seq, frame_ids)                 # (V,3,3), normalized
        depths = self._load_depth(depth_paths, raw_K)        # (V,518,518) meters, no resize
        
        raw_K = self._denormalize_K(raw_K, self.RGB_WIDTH, self.RGB_HEIGHT)
        
        # from lib.geometry import depth_to_cam_coords_points
        # local_points = depth_to_cam_coords_points(depths.numpy()[0], raw_K.numpy()[0])
        
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pts = local_points.reshape(-1, 3)
        # pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        # c = images_cpu.numpy()[0].transpose((1, 2, 0)).reshape(-1, 3)
        # pcd.colors = o3d.utility.Vector3dVector(c.astype(np.float32))
        # o3d.io.write_point_cloud("temp.ply", pcd)
        # exit()
        
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