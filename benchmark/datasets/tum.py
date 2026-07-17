from __future__ import annotations

import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms as TF
from torchvision.transforms import functional as F

from .base import BaseDataset, DatasetConfig, SequenceBatch


class TumDataset(BaseDataset):
    """
    TUM RGB-D Dynamic Dataset loader.

    Expected Directory Structure:
      <config.data_root>/
        └── <scene_id>/
            ├── rgb_90/             (preprocessed RGB images)
            ├── depth/              (original registered depth maps)
            ├── depth.txt           (depth timestamp to filename map)
            └── groundtruth_90.txt  (TUM format poses)
    """

    DEFAULT_INTRINSICS = (535.4, 539.2, 320.1, 247.6)
    MAX_DEPTH_TIME_DIFF = 0.04

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        self.dataset_root = self.config.data_root
        self.rgb_dirname = "rgb_90"
        self.pose_filename = "groundtruth_90.txt"
        self.depth_index_filename = "depth.txt"

    def list_sequences(self, split: str = "all") -> List[str]:
        return sorted(
            d for d in os.listdir(self.dataset_root)
            if osp.isdir(osp.join(self.dataset_root, d))
        )

    def count_seq_len(self, seq: str) -> int:
        rgb_dir = osp.join(self.dataset_root, seq, self.rgb_dirname)
        pose_path = osp.join(self.dataset_root, seq, self.pose_filename)

        len_rgb = len([d for d in os.listdir(rgb_dir) if d.endswith(".png")]) if osp.exists(rgb_dir) else 0
        len_pose = len(np.loadtxt(pose_path, ndmin=2)) if osp.exists(pose_path) else 0

        if len_rgb != len_pose:
            print(f"Warning: {seq} mismatched! rgb={len_rgb}, pose={len_pose}")

        return len_rgb

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.dataset_root, seq)
        image_files = sorted(os.listdir(osp.join(seq_dir, self.rgb_dirname)))
        pose_path = osp.join(seq_dir, self.pose_filename)
        depth_index = self._load_depth_index(seq_dir)

        images = [osp.join(seq_dir, self.rgb_dirname, image_files[i]) for i in frame_ids]
        poses = [f"{pose_path}::{i}" for i in frame_ids]
        depths = [self._match_depth_path(seq_dir, image_files[i], depth_index) for i in frame_ids]

        return images, poses, depths

    def _load_depth_index(self, seq_dir: str) -> Dict[float, str]:
        depth_index = {}
        with open(osp.join(seq_dir, self.depth_index_filename), "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                timestamp, rel_path = line.split()[:2]
                depth_index[float(timestamp)] = rel_path
        return depth_index

    def _match_depth_path(self, seq_dir: str, image_file: str, depth_index: Dict[float, str]) -> str:
        rgb_timestamp = float(osp.splitext(image_file)[0])
        depth_timestamp = min(depth_index.keys(), key=lambda t: abs(t - rgb_timestamp))
        if abs(depth_timestamp - rgb_timestamp) >= self.MAX_DEPTH_TIME_DIFF:
            raise ValueError(
                f"No matched TUM depth for {image_file}: nearest diff={abs(depth_timestamp - rgb_timestamp):.6f}"
            )
        return osp.join(seq_dir, depth_index[depth_timestamp])

    def _load_rgb(self, image_paths: List[str], target_width=518, target_height=392, border=0):
        image_pils, images_cpu = [], []
        for image_path in image_paths:
            img = Image.open(image_path).convert("RGB")

            width, height = img.size
            left, top, right, bottom = int(border), int(border), int(width - border), int(height - border)
            img = img.crop((left, top, right, bottom))
            img = img.resize((self.config.rgb_origin_W, self.config.rgb_origin_H), Image.Resampling.BILINEAR)

            if self.config.crop_left + self.config.crop_right + self.config.crop_top + self.config.crop_bottom > 0:
                left, top, right, bottom = (
                    int(self.config.crop_left),
                    int(self.config.crop_top),
                    int(self.config.rgb_origin_W - self.config.crop_right),
                    int(self.config.rgb_origin_H - self.config.crop_bottom),
                )
                img = img.crop((left, top, right, bottom))

            actual_W, actual_H = img.size
            if self.config.rgb_resize_W != actual_W or self.config.rgb_resize_H != actual_H:
                img = img.resize((self.config.rgb_resize_W, self.config.rgb_resize_H), Image.Resampling.BILINEAR)

            image_pils.append(img)
            images_cpu.append(TF.ToTensor()(img))

        images_cpu = torch.stack(images_cpu)
        return image_pils, images_cpu

    def _load_depth(self, depth_paths: List[str], target_width=518, target_height=392, border=0) -> torch.Tensor:
        depths = []
        for depth_path in depth_paths:
            depth_png = np.asarray(Image.open(depth_path))
            depth = depth_png.astype(np.float32) / self.config.depth_scale
            depth[depth_png == 0] = -1.0
            depths.append(depth)

        depths = torch.from_numpy(np.stack(depths))

        depths = F.resize(
            depths,
            size=[self.config.rgb_origin_H, self.config.rgb_origin_W],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        if self.config.crop_left + self.config.crop_right + self.config.crop_top + self.config.crop_bottom > 0:
            left, top, right, bottom = (
                int(self.config.crop_left),
                int(self.config.crop_top),
                int(self.config.rgb_origin_W - self.config.crop_right),
                int(self.config.rgb_origin_H - self.config.crop_bottom),
            )
            depths = depths[:, top:bottom, left:right]

        _, actual_H, actual_W = depths.shape
        if self.config.rgb_resize_W != actual_W or self.config.rgb_resize_H != actual_H:
            depths = F.resize(
                depths,
                size=[self.config.rgb_resize_H, self.config.rgb_resize_W],
                interpolation=TF.InterpolationMode.NEAREST,
            )

        return depths

    def _load_w2c(self, pose_paths: List[str]) -> torch.Tensor:
        pose_cache = {}
        w2c_list = []
        for pose_ref in pose_paths:
            pose_path, frame_id = pose_ref.rsplit("::", 1)
            if pose_path not in pose_cache:
                pose_cache[pose_path] = np.loadtxt(pose_path, ndmin=2)
            row = pose_cache[pose_path][int(frame_id)]

            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
            c2w[:3, 3] = row[1:4]
            w2c_list.append(np.linalg.inv(c2w))

        w2c = torch.from_numpy(np.stack(w2c_list))
        return w2c

    def _load_K(self, seq: str, num_frames: int, target_width=640, target_height=480) -> Tuple[torch.Tensor, torch.Tensor]:
        fx, fy, cx, cy = self.DEFAULT_INTRINSICS
        pix_K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 3, 3).repeat(num_frames, 1, 1)

        norm_K = self._normalize_K(pix_K, target_height, target_width)
        pix_K = self._denormalize_K(norm_K, self.config.rgb_origin_W, self.config.rgb_origin_H)

        if self.config.crop_left + self.config.crop_top > 0:
            pix_K[..., 0, 2] -= self.config.crop_left
            pix_K[..., 1, 2] -= self.config.crop_top

        actual_H, actual_W = (
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
        depths = self._load_depth(depth_paths)
        raw_w2c = self._load_w2c(pose_paths)
        norm_K, pix_K = self._load_K(seq, num_frames=len(frame_ids))

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
                "dataset": "tum",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )