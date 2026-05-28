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
import cv2

from lib.load_fn import load_and_preprocess_images
from lib.geometry import closed_form_inverse_se3
from lib.sintel_io import depth_read, cam_read

from .base import BaseDataset, SequenceBatch, DatasetConfig

class SintelDataset(BaseDataset):
    """
    MPI-Sintel Dataset Loader.
    
    Expected Directory Structure:
      <config.data_root>/
        └── training/
            ├── final/          (RGB images: pass 'final')
            │     └── <scene_id>/frame_{id:04d}.png
            ├── depth/          (Depth maps)
            │     └── <scene_id>/frame_{id:04d}.dpt
            └── camdata_left/   (Camera parameters)
                  └── <scene_id>/frame_{id:04d}.cam
    """

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        self.rgb_root = osp.join(self.config.data_root, "training", "final")
        self.depth_root = osp.join(self.config.data_root, "training", "depth")
        self.cam_root = osp.join(self.config.data_root, "training", "camdata_left")

    def list_sequences(self, split: str = "all") -> List[str]:
        # return sorted([d for d in os.listdir(self.rgb_root) if osp.isdir(osp.join(self.rgb_root, d))])[-self.config.num_seq:]
        
        target_seqs = [
        "alley_2", "cave_2", "cave_4", "market_5",
        "market_6", "shaman_3", "sleeping_1", "sleeping_2", "temple_2",
        ]
        
        all_seqs = sorted([d for d in os.listdir(self.rgb_root) if osp.isdir(osp.join(self.rgb_root, d))])
        filtered_seqs = [seq for seq in target_seqs if seq in all_seqs]
        return filtered_seqs

    def count_seq_len(self, seq: str) -> int:
        def count_files(base_dir: str, extension: str) -> int:
            target_dir = osp.join(base_dir, seq)
            if not osp.exists(target_dir):
                return 0
            return len([d for d in os.listdir(target_dir) if d.endswith(extension)])

        len_rgb = count_files(self.rgb_root, ".png")
        len_depth = count_files(self.depth_root, ".dpt")
        len_cam = count_files(self.cam_root, ".cam")

        if not (len_rgb == len_depth == len_cam):
            print(f"Warning: {seq} mismatched! rgb={len_rgb}, depth={len_depth}, cam={len_cam}")
            
        return len_rgb

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.config.data_root, seq)

        images = [osp.join(self.rgb_root, seq, f"frame_{i + 1:04d}.png") for i in frame_ids]
        depths = [osp.join(self.depth_root, seq, f"frame_{i + 1:04d}.dpt") for i in frame_ids]
        poses   = [osp.join(self.cam_root, seq, f"frame_{i + 1:04d}.cam") for i in frame_ids]
        
        return images, poses, depths
    
    def _load_rgb(self, image_paths: List[str], 
            target_width=518, target_height=392, border=0):
        """
            Sintel RGBs normally don't have black borders, so default border=0.
        """
        
        # crop input images
        image_pils, images_cpu = [], []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")
            
            # crop borders 15px
            width, height = img.size
            left, top, right, bottom = int(border), int(border), int(width-border), int(height-border)
            img = img.crop((left, top, right, bottom))
            
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
    
    
    def _load_depth(self, depth_paths: List[str], 
            target_width=518, target_height=392, border=15) -> torch.Tensor:
        
        depths = []
        for depth_path in depth_paths:
            # loading with PIL has value issues
            depth = depth_read(depth_path)
            depths.append(depth)
            
        depths = np.stack(depths)
        depths = depths / self.config.depth_scale  # mm -> m
        depths = torch.from_numpy(depths)
        
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
        w2c_list = []
        for cam_path in pose_paths:
            _, N = cam_read(cam_path)
            
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :4] = N
            w2c_list.append(w2c)
            
        w2c = torch.from_numpy(np.stack(w2c_list))  # (V, 4, 4)
        return w2c
    
    def _load_K(self, seq: str, num_frames: int,
            target_width=1024, target_height=436) -> torch.Tensor:
        cam_dir = osp.join(self.cam_root, seq)
        cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith('.cam')])
        first_cam_path = osp.join(cam_dir, cam_files[0])
        
        # 读取内参 M
        M, _ = cam_read(first_cam_path)
        pix_K = M.astype(np.float32)
        pix_K = torch.from_numpy(pix_K)
        
        # 复制到与帧数匹配的维度 (V, 3, 3)
        pix_K = pix_K.view(1, 3, 3).repeat(num_frames, 1, 1)
        
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
        
        return norm_K.float(), pix_K.float()

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)                     # (V,H,W) meters, no resize
        raw_w2c = self._load_w2c(pose_paths)                       # (V,4,4)
        norm_K, pix_K = self._load_K(seq, num_frames=len(frame_ids))       # (V,3,3)

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
                "dataset": "sintel",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )
