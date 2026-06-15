# benchmark/datasets/ase.py
from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple
from glob import glob

import numpy as np
import torch
import imageio
from PIL import Image
from torchvision import transforms as TF
import cv2

from lib.load_fn import load_and_preprocess_images
from lib.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch, DatasetConfig

class SekaiDataset(BaseDataset):
    """
    Sekai_root:
        - <seq>
            - rgb/:  frame_000000.jpg
            - xxx.npy: 
                data.item() -> {
                    "extr_mat": (V, 4, 4) OpenCV W2C
                    "intr_mat": (V, 3, 3)
                }
    """
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        
    def list_sequences(self, split) -> List[str]:
        return sorted([d for d in os.listdir(self.config.data_root) if osp.isdir(osp.join(self.config.data_root, d))])[-self.config.num_seq:]
    
    def count_seq_len(self, seq):
        rgb_dir = Path(self.config.data_root) / seq / "rgb"
        
        return len(list(rgb_dir.iterdir()))
    
    def make_paths(self, seq, frame_ids):
        seq_root = Path(self.config.data_root) / seq
        images = [seq_root / "rgb" / "frame_{}.jpg".format(str(i).zfill(5)) for i in frame_ids]
        cameras = glob(str(seq_root / "*.npy"))[0]
        tracks = glob(str(seq_root / "romav2" / "*.npz"))[0]
        
        return images, cameras, tracks
    
    def _load_rgb(self, image_paths: List[str]):
        # image_pils = [Image.open(p).convert("RGB").resize((target_width, target_height), Image.Resampling.BICUBIC) \
        #               for p in image_paths]
        # images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        
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
    
    def _load_depth(self, images_cpu: torch.Tesnor) -> torch.Tensor:
        V,C,H,W = images_cpu.shape
        return torch.zeros((V, H, W), dtype=torch.float32)
    
    def _load_w2c_and_K(self, camera_paths: List[str], frame_ids: List[int]) -> torch.Tensor:
        
        cam_info = np.load(camera_paths, allow_pickle=True).item()
        
        # load extrinsics
        # w2c
        poses = np.stack([cam_info["extr_mat"][i] for i in frame_ids]).astype(float)
        poses = torch.from_numpy(poses) # 4,4
        
        norm_K = np.stack([cam_info["intr_mat"][i] for i in frame_ids]).astype(float)
        norm_K = torch.from_numpy(norm_K).float()
        
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
        
        return poses, pix_K, norm_K
    
    def _load_track(self, track_paths, device="cpu", dtype=torch.float32, dtype_visib=torch.bool):
        track_info = np.load(track_paths, allow_pickle=True)["data"]
        track0 = torch.from_numpy(np.concatenate([matching_info["tracks"][0] for matching_info in track_info], axis=0)) # n k 2
        track1 = torch.from_numpy(np.concatenate([matching_info["tracks"][1] for matching_info in track_info], axis=0)) # n k 2
        visib0 = torch.from_numpy(np.concatenate([matching_info["visibility"][0] for matching_info in track_info], axis=0)) # n k
        visib1 = torch.from_numpy(np.concatenate([matching_info["visibility"][1] for matching_info in track_info], axis=0)) # n k
        
        return dict(
            track0=track0.to(device=device, dtype=dtype), 
            track1=track1.to(device=device, dtype=dtype), 
            visib0=visib0.to(device=device, dtype=dtype_visib),
            visib1=visib1.to(device=device, dtype=dtype_visib), 
        )
    
    def load_sequence(self, seq, frame_ids) -> SequenceBatch:
        image_paths, camera_paths, track_paths = self.make_paths(seq, frame_ids)
        
        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(images_cpu)     # (V,H,W) meters, no resize
        raw_w2c, pix_K, norm_K = self._load_w2c_and_K(camera_paths, frame_ids)   # (V,4,4)
        tracks = self._load_track(track_paths)
        
        return SequenceBatch(
            seq=seq,
            frame_ids=frame_ids,
            image_paths=image_paths,
            pose_paths=image_paths,
            depth_paths=image_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            tracks=tracks,
            raw_w2c=raw_w2c,
            raw_K=pix_K,
            norm_K=norm_K,
            meta={
                "dataset": "sekai",
                "has_gt_depth": False,
            },
        )