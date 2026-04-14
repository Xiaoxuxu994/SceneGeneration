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

from .base import BaseDataset, SequenceBatch, DatasetConfig

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

    # DEPTH_SCALE = 1000.0  # mm -> m
    # NUM_TEST = 10

    # def __init__(self, root: str | Path, num_seq: int = NUM_TEST,
    #              depth_scale: float = DEPTH_SCALE):
    #     super().__init__(root, num_seq)
    #     self.depth_scale = depth_scale

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def list_sequences(self, split: str = "all") -> List[str]:
        return sorted([d for d in os.listdir(self.config.data_root) if osp.isdir(osp.join(self.config.data_root, d))])[-self.config.num_seq:]

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.config.data_root, seq)

        def list_suffix(prefix: str, suffix: str):
            return sorted([d for d in os.listdir(os.path.join(seq_path,prefix)) if d.endswith(suffix)])

        rgbs = list_suffix(prefix="color",suffix=".jpg")
        depths = list_suffix(prefix="depth",suffix=".png")
        poses = list_suffix(prefix="pose",suffix=".txt")

        if not (len(rgbs) == len(depths) == len(poses)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={len(poses)}")
        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.config.data_root, seq)
        images = [osp.join(seq_dir, "color", f"{i}.jpg") for i in frame_ids]
        depths = [osp.join(seq_dir, "depth", f"{i}.png") for i in frame_ids]
        poses  = [osp.join(seq_dir, "pose",  f"{i}.txt") for i in frame_ids]
        return images, poses, depths

    # def _load_rgb(self, image_paths: List[str], target_width=518, target_height=392, border=15):
    #     # crop input images
    #     image_pils, images_cpu = [], []
    #     for image_path in image_paths:
    #         img = Image.open(image_path).convert("RGB")
            
    #         # crop borders 15px
    #         width, height = img.size
    #         left, top, right, bottom = int(border), int(border), int(width-border), int(height-border)
    #         img = img.crop((left, top, right, bottom))
            
    #         # resize to 392x518
    #         img_rs = img.resize((target_width, target_height), Image.Resampling.BICUBIC)
    #         image_pils.append(img_rs)
    #         img_tensor = TF.ToTensor()(img_rs)
    #         images_cpu.append(img_tensor)
            
    #     images_cpu = torch.stack(images_cpu)
        
    #     return image_pils, images_cpu
    
    def _load_rgb(self, image_paths: List[str], 
            target_width=518, target_height=392, border=15):
        """default RGBs in ASE have black corners
            cropping input ASE RGBs from 518x518 to 462x462
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
    
    # def _load_depth(self, depth_paths: List[str], 
    #         target_width=518, target_height=392, border=15) -> torch.Tensor:
        
    #     depths = []
    #     for depth_path in depth_paths:
    #         # loading with PIL has value issues
    #         depth = imageio.v2.imread(depth_path).astype(np.float32)
            
    #         # resize to 392x518
    #         depth = cv2.resize(depth, (target_width, target_height), interpolation = cv2.INTER_NEAREST)

    #         # to tensor
    #         depth = np.array(depth).astype(np.float32)
    #         depths.append(depth)
            
    #     depths = np.stack(depths)
    #     depths = depths / self.depth_scale  # mm -> m
    #     depths = torch.from_numpy(depths)
        
    #     return depths
    
    def _load_depth(self, depth_paths: List[str], 
            target_width=518, target_height=392, border=15) -> torch.Tensor:
        
        depths = []
        for depth_path in depth_paths:
            # loading with PIL has value issues
            depth = imageio.v2.imread(depth_path).astype(np.float32)
            depth = np.array(depth).astype(np.float32)
            depths.append(depth)
            
        depths = np.stack(depths)
        depths = depths / self.depth_scale  # mm -> m
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
        # pose.txt is c2w (4x4)
        c2w = np.stack([np.loadtxt(p) for p in pose_paths])  # (V,4,4)
        c2w = torch.from_numpy(c2w)
        w2c = closed_form_inverse_se3(c2w)  # (V,4,4)
        return w2c

    # def _load_K(self, seq: str, num_frames: int,
    #         target_width=518, target_height=392) -> torch.Tensor:
    #     intr_path = osp.join(self.config.data_root, seq, "intrinsic", "intrinsic_depth.txt")
    #     intr = np.loadtxt(intr_path).astype(np.float32)
    #     K = intr[:3, :3].astype(np.float32)
    #     K = torch.from_numpy(K)
        
    #     K_rs = K.clone()
    #     K_rs[0, 0] = K[0, 0] / (K[0, 2] * 2) * target_width
    #     K_rs[0, 2] = K[0, 2] / (K[0, 2] * 2) * target_width
    #     K_rs[1, 1] = K[1, 1] / (K[1, 2] * 2) * target_height
    #     K_rs[1, 2] = K[1, 2] / (K[1, 2] * 2) * target_height
        
    #     return K_rs.view(1, 3, 3).repeat(num_frames, 1, 1)
    
    def _load_K(self, seq: str, num_frames: int,
            target_width=518, target_height=392) -> torch.Tensor:
        intr_path = osp.join(self.config.data_root, seq, "intrinsic", "intrinsic_depth.txt")
        intr = np.loadtxt(intr_path).astype(np.float32)
        pix_K = intr[:3, :3].astype(np.float32)
        pix_K = torch.from_numpy(pix_K)
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
        
        return norm_K, pix_K

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)                     # (V,H,W) meters, no resize
        raw_w2c = self._load_w2c(pose_paths)                       # (V,4,4)
        norm_K, pix_K = self._load_K(seq, num_frames=len(frame_ids))       # (V,3,3)

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
            pose_paths=pose_paths,
            depth_paths=depth_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            raw_w2c=raw_w2c,
            raw_K=pix_K,
            norm_K=norm_K,
            meta={
                "dataset": "scannet",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )
