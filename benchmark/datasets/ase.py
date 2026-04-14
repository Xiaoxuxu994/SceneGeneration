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

from .base import BaseDataset, SequenceBatch, DatasetConfig

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
    
    # DEPTH_SCALE = 1000.0  # mm -> m
    # NUM_TEST = 50
    # RGB_WIDTH = 518
    # RGB_HEIGHT = 518
    # CROP_BORDER = 28
    
    # def __init__(self, root: str | Path, num_seq: int = NUM_TEST, depth_scale: float = DEPTH_SCALE):
    #     super().__init__(root, num_seq)
        
    #     self.depth_scale = depth_scale
        
    #     if root.endswith(".npz"):
    #         self.sample_sequences = np.load(root, allow_pickle=True)
    #     else:
    #         raise NotImplementedError
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        
        if self.config.data_root.endswith(".npz"):
            self.sample_sequences = np.load(self.config.data_root, allow_pickle=True)
        else:
            raise NotImplementedError
        
    def list_sequences(self, split):
        return self.sample_sequences.files[-self.config.num_seq:]
    
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
    
    def _load_depth(self, depth_paths: List[str], seq: str, frame_ids: List[int]) -> torch.Tensor:
        """default RGBs in ASE have black corners
            cropping GT depths to match RGBs from 518x518 to 462x462
        """
        
        sequence = self.sample_sequences[seq].item()
        norm_K = np.stack([sequence["input_intrs"][i] for i in frame_ids])
        norm_K = torch.from_numpy(norm_K).float()
        
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
        depths = rdepth_to_zdepth(depths.unsqueeze(1), norm_K).squeeze(1) # 518x518
        
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
            
        return depths, norm_K # (V,518,518) meters
    
    def _load_w2c(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        sequence = self.sample_sequences[seq].item()
        poses = np.stack([sequence["input_extrs"][i] for i in frame_ids])
        
        return torch.from_numpy(poses)
    
    def _load_K(self, norm_K: torch.Tensor) -> torch.Tensor:    
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
    
    def load_sequence(self, seq, frame_ids) -> SequenceBatch:
        image_paths, depth_paths = self.make_paths(seq, frame_ids)
        
        image_pils, images_cpu = self._load_rgb(image_paths)  # (V,518,518)
        depths, norm_K = self._load_depth(depth_paths, seq, frame_ids)        # (V,518,518) meters, no resize
        raw_w2c = self._load_w2c(seq, frame_ids)              # (V,4,4)
        norm_K, pix_K = self._load_K(norm_K)          # (V,3,3), normalized
        
        # from lib.geometry import depth_to_cam_coords_points
        # local_points = depth_to_cam_coords_points(depths.numpy()[0], pix_K.numpy()[0])
        
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
            raw_K=pix_K,
            norm_K=norm_K,
            meta={
                "dataset": "ase",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )