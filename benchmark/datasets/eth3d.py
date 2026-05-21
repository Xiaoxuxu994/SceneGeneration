# benchmark/datasets/dtu.py
from __future__ import annotations
import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import cv2
import imageio
from PIL import Image

from torchvision import transforms as TF
from scipy.spatial.transform import Rotation as R

from lib.load_fn import load_and_preprocess_images

from .base import BaseDataset, SequenceBatch, DatasetConfig

# -----------------------------
# dataset
# -----------------------------
class ETH3DDataset(BaseDataset):
    """
    ETH3D:
      ETH3D/multi_view_training_dslr/xxx/
        dslr_calibration_undistorted_600x400/
          cameras.txt
          images.txt
          points3D.txt
        gt_depth/
          undistorted_600x400/
            DSC_xxxx.JPG
        images/
          dslr_images_undistorted_600x400/
            DSC_xxxx.JPG
    """
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        
        self.sequence_list = sorted([d for d in os.listdir(self.config.data_root) if osp.isdir(osp.join(self.config.data_root, d))])
    
        # build lookup dict for frame IDs -> image names
        self.frame_id2name = {}
        for seq in self.sequence_list:
            seq_path = osp.join(self.config.data_root, seq)
            rgbs = self._list_suffix(seq_path, prefix="images/dslr_images_undistorted_600x400", suffix=".JPG")
            names = [s.split('/')[-1].split('.')[0] for s in rgbs]
            self.frame_id2name[seq] = {id_: name for id_, name in enumerate(names)}

    def _list_suffix(self, seq_path: str, prefix: str, suffix: str):
        return sorted([d for d in os.listdir(os.path.join(seq_path, prefix)) if d.endswith(suffix)])
    
    def list_sequences(self, split: str) -> List[str]:
        return self.sequence_list

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.config.data_root, seq)

        rgbs = self._list_suffix(seq_path, prefix="images/dslr_images_undistorted_600x400", suffix=".JPG")
        depths = self._list_suffix(seq_path, prefix="gt_depth/undistorted_600x400", suffix=".png")

        if not (len(rgbs) == len(depths)):
            print(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)}")
        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str], str, str]:
        seq_dir = osp.join(self.config.data_root, seq)
        names = [f"{self.frame_id2name[seq][i]}" for i in frame_ids]
        images = [osp.join(seq_dir, "images/dslr_images_undistorted_600x400" ,f"{self.frame_id2name[seq][i]}.JPG") for i in frame_ids]
        depths = [osp.join(seq_dir, "gt_depth/undistorted_600x400", f"{self.frame_id2name[seq][i]}.png") for i in frame_ids]
        metas = osp.join(seq_dir, "dslr_calibration_undistorted", "images.txt")
        cameras = osp.join(seq_dir, "dslr_calibration_undistorted", "cameras.txt")
        return names, images, depths, metas, cameras

    # def _load_rgb(self, image_paths: List[str]):
    #     image_pils = [Image.open(p).convert("RGB") for p in image_paths]
    #     images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
    #     return image_pils, images_cpu
    
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
            depth = depth / self.config.depth_scale  # mm -> m
            depth = torch.from_numpy(depth).unsqueeze(0)
            
            # resize to match rgb
            depth = TF.functional.resize(
                depth,
                size=(self.config.rgb_origin_H, self.config.rgb_origin_W), 
                interpolation=TF.InterpolationMode.NEAREST
            )
            depths.append(depth)
        depths = torch.concat(depths)
        
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
        
    def _load_view_meta(self, meta_path: str, camera_path: str) -> Tuple[dict, dict]:
        # load view meta
        view_meta = {}
        with open(meta_path) as f:
            for i, line in enumerate(f.readlines()):
                if i > 3 and i % 2 == 0 and len(line) > 0:
                    _, qw, qx, qy, qz, tx, ty, tz, camera_id, name = line.rstrip().split(' ')
                    # e.g. name = "dslr_images_undistorted/DSC_0323.JPG"
                    name = name.split('/')[1].split('.')[0]
                    view_meta[name] = {
                        "trans": [
                            float(qw), 
                            float(qx), 
                            float(qy), 
                            float(qz), 
                            float(tx), 
                            float(ty), 
                            float(tz)
                        ], # world to camera
                        "camera_id": camera_id
                    }
                    
        # load cameras
        cameras = {}
        with open(camera_path) as f:
            for i, line in enumerate(f.readlines()):
                if i > 2: 
                    camera_id, _, width, height, fx, fy, cx, cy = line.rstrip().split(' ')
                    cameras[camera_id] = {
                        "width": width,
                        "height": height,
                        "params": [
                            float(fx), 
                            float(fy), 
                            float(cx), 
                            float(cy)
                        ]
                    }
                
        return view_meta, cameras
        
    def _load_w2c(self, image_names: List[str], view_metas: dict) -> torch.Tensor:
        w2c = []
        for image_name in image_names:
            qw, qx, qy, qz, tx, ty, tz = view_metas[image_name]["trans"]
            q = [qx, qy, qz, qw] 
            t = [tx, ty, tz]
            
            rot_matrix = R.from_quat(q).as_matrix()
            
            cur_w2c = np.eye(4)
            cur_w2c[:3, :3] = rot_matrix
            cur_w2c[:3, 3] = t
            
            w2c.append(cur_w2c)
        w2c = np.stack(w2c)
        w2c = torch.from_numpy(w2c)
        
        return w2c
    
    def _load_K(self, image_names: List[str], view_metas: dict, cameras: dict) -> torch.Tensor:
        norm_K = []
        for image_name in image_names:
            camera_id = view_metas[image_name]["camera_id"]
            fx, fy, cx, cy = cameras[camera_id]["params"]
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
    
    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_names, image_paths, depth_paths, meta_path, camera_path = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)               # (V,3,*,*) per VGGT
        depths = self._load_depth(depth_paths)                 # (V,H,W) original depth res
        view_metas, cameras = self._load_view_meta(meta_path, camera_path)
        raw_w2c = self._load_w2c(image_names, view_metas)
        norm_K, pix_K = self._load_K(image_names, view_metas, cameras)                       # (V,4,4),(V,3,3)

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
            pose_paths=meta_path,
            depth_paths=depth_paths,
            image_pils=image_pils,
            images_cpu=images_cpu,
            depths=depths,
            tracks={},
            raw_w2c=raw_w2c,
            raw_K=pix_K,
            norm_K=norm_K,
            meta={
                "dataset": "eth3d",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )
