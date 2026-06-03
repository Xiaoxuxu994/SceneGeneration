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

class SpatialVIDDataset(BaseDataset):
    """
    simply input the sample sequence list:
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
    
    # DEPTH_SCALE = 1000.0  # mm -> m, NOTE there's no GT depth
    # NUM_TEST = 10
    
    # def __init__(self, root: str | Path, num_seq: int = NUM_TEST, depth_scale: float = DEPTH_SCALE):
    #     super().__init__(root, num_seq)
        
    #     self.depth_scale = depth_scale
        
    #     if root.endswith(".npz"):
    #         self.sample_sequences = np.load(root, allow_pickle=True)["data"][-self.num_seq:]
    #     else:
    #         raise NotImplementedError
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        
        if self.config.data_root.endswith(".npz"):
            self.sample_sequences = np.load(self.config.data_root, allow_pickle=True)["data"][-self.config.num_seq:]
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
        tracks = glob(str(seq_root / "romav2" / "*.npz"))[0]
        
        return images, cameras, tracks
    
    def _load_rgb(self, image_paths: List[str], target_width=518, target_height=294):
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
                "dataset": "spatialvid",
                "has_gt_depth": False,
            },
        )