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
import json
import pickle

from .base import BaseDataset, SequenceBatch, DatasetConfig

from lib.utils import rdepth_to_zdepth
from lib.matching import *

class KITTIDataset(BaseDataset):
    """
    KITTI/val/2011_09_30_drive_0016_sync:
        color/<seq>/data
            0000000000.png
        depth/<seq>/
            0000000000.png
    """
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        
        with open(f"{self.config.data_root}/sequence_list.pkl", "rb") as f:
            self.sample_sequences = pickle.load(f)
        
        self.seq2data = self._build_seq2data()
        
    def _build_seq2data(self):
        seq2data = {}
        for d in self.sample_sequences:
            seq_id = d["_name"]
            for image_path, extrs, intrs in zip(d["images"], d["poses"], d["intrinsics"]):
                *_, seq_dir, seq_name, _, image_name = image_path.split('/')
                image_name = image_name.split('.')[0]
                
                if seq_id not in seq2data: seq2data[seq_id] = []
                
                color_path = f"{self.config.data_root}/{seq_dir}/color/{seq_name}/data/{image_name}.png"
                depth_path = f"{self.config.data_root}/{seq_dir}/depth/{seq_name}/{image_name}.png"
                if not os.path.exists(color_path) or not os.path.exists(depth_path): continue
                item = {
                    "color": color_path,
                    "depth": depth_path,
                    "extr": extrs,
                    "intr": intrs
                }
                seq2data[seq_id].append(item)
                
        return seq2data
        
    def list_sequences(self, split):
        return [k for k in self.seq2data.keys()][:self.config.num_seq]
    
    def count_seq_len(self, seq):
        rgbs = [d["color"] for d in self.seq2data[seq]]
        depths = [d["depth"] for d in self.seq2data[seq]]
        extrs = [d["extr"] for d in self.seq2data[seq]]
        intrs = [d["intr"] for d in self.seq2data[seq]]
        
        if not (len(rgbs) == len(depths) == len(extrs) == len(intrs)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} extrinsics={len(extrs)} intrinsics={len(intrs)}")
        return len(rgbs)
    
    def make_paths(self, seq, frame_ids):
        images = [self.seq2data[seq][f]["color"] for f in frame_ids]
        depths = [self.seq2data[seq][f]["depth"] for f in frame_ids]
        
        return images, depths
    
    def _load_rgb(self, image_paths: List[str]):
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
    
    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        """default RGBs in ASE have black corners
            cropping GT depths to match RGBs from 518x518 to 462x462
        """
        
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
            
        return depths # (V,518,518) meters
    
    def _load_w2c(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        w2c = [self.seq2data[seq][f]["extr"] for f in frame_ids]
        w2c = np.stack(w2c)
        w2c = torch.from_numpy(w2c).double()
        
        return w2c
    
    def _load_K(self, seq: str, frame_ids: List[int]) -> torch.Tensor:    
        pix_K = [self.seq2data[seq][f]["intr"] for f in frame_ids]
        pix_K = np.stack(pix_K)
        pix_K = torch.from_numpy(pix_K).double()
        
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
    
    def load_sequence(self, seq, frame_ids) -> SequenceBatch:
        image_paths, depth_paths = self.make_paths(seq, frame_ids)
        
        image_pils, images_cpu = self._load_rgb(image_paths)  # (V,518,518)
        depths = self._load_depth(depth_paths)        # (V,518,518) meters, no resize
        raw_w2c = self._load_w2c(seq, frame_ids)              # (V,4,4)
        norm_K, pix_K = self._load_K(seq, frame_ids)          # (V,3,3), normalized
        
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
            tracks={},
            raw_w2c=raw_w2c,
            raw_K=pix_K,
            norm_K=norm_K,
            meta={
                "dataset": "ase",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )