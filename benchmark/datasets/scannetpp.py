# benchmark/datasets/scannetpp_posed_images.py
from __future__ import annotations

import os
import os.path as osp
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import imageio
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch


class ScanNetPPDataset(BaseDataset):
    """
    ScanNet++ (your rendered posed images):

      <root>/
        <scene_id>/
          image/<filename>.JPG
          depth/<filename_stem>.png     # uint16 in millimeters
          transforms.json              # contains fl_x, fl_y, cx, cy, w, h, frames[...]

    Assumptions (per your message):
      - transforms.json frame["file_path"] matches the real filename in image/
      - depth filename is <stem>.png (same stem as image)
      - transform_matrix is c2w, but needs your fixed conversion (swap xy etc.)
    """

    DEPTH_SCALE = 1000.0  # mm -> m

    def __init__(self, root: str | Path, depth_scale: float = DEPTH_SCALE):
        super().__init__(root)
        self.depth_scale = float(depth_scale)

        # cache parsed transforms per scene
        self._cache: Dict[str, Dict[str, Any]] = {}

    def list_sequences(self, split: str = "all") -> List[str]:
        return sorted([d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))])

    def count_seq_len(self, seq: str) -> int:
        info = self._load_transforms(seq)
        frames = info["frames"]
        return len(frames)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        """
        Keep same signature as ScanNetDataset: (images, poses, depths)

        poses: for scannetpp we only have transforms.json, so we return it repeated (debug/log only).
        actual pose matrix is loaded from transforms.json inside _load_w2c.
        """
        info = self._load_transforms(seq)
        frames = info["frames"]

        seq_dir = osp.join(self.root, seq)
        img_dir = osp.join(seq_dir, "image")
        depth_dir = osp.join(seq_dir, "depth")
        trans_path = osp.join(seq_dir, "transforms.json")

        images = []
        depths = []
        poses = []

        for i in frame_ids:
            fp = frames[i]["file_path"]
            name = osp.basename(fp)  # matches actual image filename
            img_path = osp.join(img_dir, name)
            depth_path = osp.join(depth_dir, f"{osp.splitext(name)[0]}.png")

            images.append(img_path)
            depths.append(depth_path)
            poses.append(trans_path)

        return images, poses, depths

    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p).convert("RGB") for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu

    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        d = np.stack([imageio.v2.imread(p).astype(np.float32) for p in depth_paths])  # (V,H,W)
        d = d / self.depth_scale  # mm -> m
        return torch.from_numpy(d)

    def _load_transforms(self, seq: str) -> Dict[str, Any]:
        if seq in self._cache:
            return self._cache[seq]

        trans_path = osp.join(self.root, seq, "transforms.json")
        if not osp.isfile(trans_path):
            raise FileNotFoundError(trans_path)

        with open(trans_path, "r") as f:
            info: Dict[str, Any] = json.load(f)

        # build filename -> raw c2w mapping for fast lookup
        frames = info.get("frames", [])
        name_to_T: Dict[str, np.ndarray] = {}
        for fr in frames:
            name = osp.basename(fr["file_path"])
            T_raw = np.array(fr["transform_matrix"], dtype=np.float32).reshape(4, 4)
            name_to_T[name] = T_raw

        pack = {
            "fx": float(info["fl_x"]),
            "fy": float(info["fl_y"]),
            "cx": float(info["cx"]),
            "cy": float(info["cy"]),
            "W": int(info["w"]),
            "H": int(info["h"]),
            "frames": frames,
            "name_to_T": name_to_T,
        }
        self._cache[seq] = pack
        return pack

    @staticmethod
    def _fix_c2w(T_raw: np.ndarray) -> np.ndarray:
        """
        EXACTLY your snippet:
          c2w[2,:] *= -1
          c2w = c2w[[1,0,2,3],:]
          c2w[:3,1:3] *= -1
        """
        T = T_raw.astype(np.float32).copy()
        T[2, :] *= -1.0
        T = T[[1, 0, 2, 3], :]
        T[:3, 1:3] *= -1.0
        return T

    def _load_w2c(self, seq: str, image_paths: List[str]) -> torch.Tensor:
        info = self._load_transforms(seq)
        name_to_T: Dict[str, np.ndarray] = info["name_to_T"]

        c2w_list = []
        for p in image_paths:
            name = osp.basename(p)
            if name not in name_to_T:
                raise KeyError(f"[{seq}] cannot find pose for {name} in transforms.json")

            T_raw = name_to_T[name]
            c2w_fixed = self._fix_c2w(T_raw)
            c2w_list.append(c2w_fixed)

        c2w = torch.from_numpy(np.stack(c2w_list, axis=0).astype(np.float64))  # (V,4,4)
        w2c = closed_form_inverse_se3(c2w)  # (V,4,4)
        return w2c

    def _load_K(self, seq: str, num_frames: int) -> torch.Tensor:
        info = self._load_transforms(seq)
        fx, fy, cx, cy = info["fx"], info["fy"], info["cx"], info["cy"]
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return torch.from_numpy(K).view(1, 3, 3).repeat(num_frames, 1, 1)

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)                      # (V,H,W) meters
        raw_w2c = self._load_w2c(seq, image_paths=image_paths)      # (V,4,4)
        raw_K = self._load_K(seq, num_frames=len(frame_ids))        # (V,3,3)

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
            raw_K=raw_K,
            meta={
                "dataset": "scannetpp",
                "DEPTH_SCALE": self.DEPTH_SCALE,
            },
        )
