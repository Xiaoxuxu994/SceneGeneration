from __future__ import annotations

import os
import os.path as osp
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF
from torchvision.transforms import functional as F

from .base import BaseDataset, DatasetConfig, SequenceBatch


class HiRoomDataset(BaseDataset):
    """
    HiRoom Dataset loader.

    Expected Directory Structure:
      <config.data_root>/
        ├── data/20241230
        │   └── <scene_id>/
        │       ├── image/           (.jpg RGB images)
        │       ├── depth/           (uint16 depth PNGs, raw / 65535 * 100 meters)
        │       ├── pose/            (.npy world-to-camera matrices)
        │       ├── cam_K.npy        (shared 3x3 intrinsics)
        │       └── aliasing_mask/   (invalid-region masks)
        └── selected_scene_list_val.txt
    """

    scene_list_filename = "selected_scene_list_val.txt"
    image_dirname = "image"
    depth_dirname = "depth"
    pose_dirname = "pose"
    aliasing_mask_dirname = "aliasing_mask"
    intrinsics_filename = "cam_K.npy"

    def __init__(self, config: DatasetConfig):
        super().__init__(config)

        self.hiroom_root = osp.normpath(self.config.data_root)
        self.dataset_root = osp.join(self.hiroom_root, "data")
        self.scene_list_path = osp.join(self.hiroom_root, self.scene_list_filename)

    def list_sequences(self, split: str = "all") -> List[str]:
        with open(self.scene_list_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def count_seq_len(self, seq: str) -> int:
        scene_dir = osp.join(self.dataset_root, seq)
        image_files = self._list_frame_files(osp.join(scene_dir, self.image_dirname), ".jpg")
        depth_files = self._list_frame_files(osp.join(scene_dir, self.depth_dirname), ".png")
        pose_files = self._list_frame_files(osp.join(scene_dir, self.pose_dirname), ".npy")

        len_image = len(image_files)
        len_depth = len(depth_files)
        len_pose = len(pose_files)
        if not (len_image == len_depth == len_pose):
            print(f"Warning: {seq} mismatched! image={len_image}, depth={len_depth}, pose={len_pose}")

        return len_image

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        scene_dir = osp.join(self.dataset_root, seq)
        image_files = self._list_frame_files(osp.join(scene_dir, self.image_dirname), ".jpg")
        depth_files = self._list_frame_files(osp.join(scene_dir, self.depth_dirname), ".png")
        pose_files = self._list_frame_files(osp.join(scene_dir, self.pose_dirname), ".npy")

        images = [image_files[i] for i in frame_ids]
        poses = [pose_files[i] for i in frame_ids]
        depths = [depth_files[i] for i in frame_ids]

        return images, poses, depths

    def _list_frame_files(self, directory: str, extension: str) -> List[str]:
        return [
            osp.join(directory, filename)
            for filename in sorted(
                [f for f in os.listdir(directory) if f.endswith(extension)],
                key=lambda name: int(osp.splitext(name)[0]),
            )
        ]

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
            aliasing_mask_path = self._aliasing_mask_path(depth_path)
            if osp.exists(aliasing_mask_path):
                aliasing_mask = np.asarray(Image.open(aliasing_mask_path)) > 0
                depth[aliasing_mask] = -1.0
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

    def _aliasing_mask_path(self, depth_path: str) -> str:
        scene_dir = osp.dirname(osp.dirname(depth_path))
        frame_name = osp.basename(depth_path)
        return osp.join(scene_dir, self.aliasing_mask_dirname, frame_name)

    def _load_w2c(self, pose_paths: List[str]) -> torch.Tensor:
        w2c_list = []
        for pose_path in pose_paths:
            w2c = np.load(pose_path).astype(np.float64)
            w2c_list.append(w2c)
        w2c = torch.from_numpy(np.stack(w2c_list))
        return w2c

    def _load_K(self, seq: str, num_frames: int, target_width=1024, target_height=1024) -> Tuple[torch.Tensor, torch.Tensor]:
        scene_dir = osp.join(self.dataset_root, seq)
        pix_K = np.load(osp.join(scene_dir, self.intrinsics_filename)).astype(np.float32)
        pix_K = torch.from_numpy(pix_K).view(1, 3, 3).repeat(num_frames, 1, 1)

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
                "dataset": "hiroom",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )