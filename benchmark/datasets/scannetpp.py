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
import json

from lib.load_fn import load_and_preprocess_images
from lib.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch, DatasetConfig

class ScanNetPPDataset(BaseDataset):
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

    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        
        self.sequence_list = sorted([d for d in os.listdir(self.config.data_root) if osp.isdir(osp.join(self.config.data_root, d))])[-self.config.num_seq:]
        
        # build lookup dict for frame IDs -> image names
        self.frame_id2name = {}
        for seq in self.sequence_list:
            seq_path = osp.join(self.config.data_root, seq)
            rgbs = self._list_suffix(seq_path, prefix="image", suffix=".JPG")
            names = [s.split('/')[-1].split('.')[0] for s in rgbs]
            self.frame_id2name[seq] = {id_: name for id_, name in enumerate(names)}

    def _list_suffix(self, seq_path: str, prefix: str, suffix: str):
        return sorted([d for d in os.listdir(os.path.join(seq_path,prefix)) if d.endswith(suffix)])

    def list_sequences(self, split: str = "all") -> List[str]:
        return self.sequence_list

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.config.data_root, seq)

        rgbs = self._list_suffix(seq_path, prefix="image",suffix=".JPG")
        depths = self._list_suffix(seq_path, prefix="depth",suffix=".png")

        if not (len(rgbs) == len(depths)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)}")
        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.config.data_root, seq)
        names = [self.frame_id2name[seq][i] for i in frame_ids]
        images = [osp.join(seq_dir, "image", f"{self.frame_id2name[seq][i]}.JPG") for i in frame_ids]
        depths = [osp.join(seq_dir, "depth", f"{self.frame_id2name[seq][i]}.png") for i in frame_ids]
        cameras  = osp.join(seq_dir, "transforms.json")
        return names, images, depths, cameras
    
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
        
        depths = []
        for depth_path in depth_paths:
            # loading with PIL has value issues
            depth = imageio.v2.imread(depth_path).astype(np.float32)
            depth = np.array(depth).astype(np.float32)
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

    def _load_w2c(self, image_names: List[str], cameras: dict) -> torch.Tensor:
        c2w = []
        for name in image_names:
            raw_c2w = cameras[name]["extr"]
            cur_c2w = raw_c2w.copy()
            cur_c2w[2, :] *= -1.0
            cur_c2w = cur_c2w[[1, 0, 2, 3], :]
            cur_c2w[:3, 1:3] *= -1.0
            c2w.append(cur_c2w)
        c2w = np.stack(c2w)
        c2w = torch.from_numpy(c2w)
        w2c = closed_form_inverse_se3(c2w)
        
        return w2c
    
    def _load_K(self, image_names: List[str], cameras: dict) -> torch.Tensor:
        norm_K = []
        for image_name in image_names:
            fx, fy, cx, cy = cameras[image_name]["intr"]
            cur_W, cur_H = cx * 2, cy * 2
            cur_K = np.array([
                [fx/cur_W, 0, cx/cur_W],
                [0, fy/cur_H, cy/cur_H],
                [0, 0, 1]
            ]).astype(np.float32)
            cur_K = torch.from_numpy(cur_K)
            norm_K.append(cur_K)
        norm_K = torch.stack(norm_K)
        pix_K = self._denormalize_K(norm_K, self.config.rgb_origin_W, self.config.rgb_origin_H)
        
        # normalize again
        (actual_H, actual_W) = (
            self.config.rgb_origin_H - self.config.crop_top - self.config.crop_bottom,
            self.config.rgb_origin_W - self.config.crop_left - self.config.crop_right,
        )
        norm_K = self._normalize_K(pix_K, actual_H, actual_W)
        
        if self.config.rgb_resize_W != actual_W or self.config.rgb_resize_H != actual_H:
            pix_K = self._denormalize_K(norm_K, self.config.rgb_resize_W, self.config.rgb_resize_H)
        
        return norm_K, pix_K

    def _load_camera(self, camera_path: str) -> dict:
        name2camera = {}
        with open(camera_path) as f:
            camera_json = json.load(f)
            for data in camera_json["frames"]:
                name = data["file_path"].split('.')[0]
                name2camera[name] = {
                    "intr": np.array([
                        camera_json["fl_x"],
                        camera_json["fl_y"],
                        camera_json["cx"],
                        camera_json["cy"]
                    ]).astype(np.float32),
                    "extr": np.array(data["transform_matrix"]).astype(np.float32)
                }
                
        return name2camera

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_names, image_paths, depth_paths, camera_path = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)                     # (V,H,W) meters, no resize
        cameras = self._load_camera(camera_path)
        raw_w2c = self._load_w2c(image_names, cameras)                       # (V,4,4)
        norm_K, pix_K = self._load_K(image_names, cameras)       # (V,3,3)

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
            pose_paths=camera_path,
            depth_paths=depth_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths.double(),
            tracks={},
            raw_w2c=raw_w2c.double(),
            raw_K=pix_K.double(),
            norm_K=norm_K.double(),
            meta={
                "dataset": "scannetpp",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )
