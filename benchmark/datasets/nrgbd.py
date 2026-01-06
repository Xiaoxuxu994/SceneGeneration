# benchmark/datasets/nrgbd.py
from __future__ import annotations
import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import imageio
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch


def _read_poses_txt_as_4x4(pose_txt: str) -> np.ndarray:
    """Read poses.txt where every 4 lines form a 4x4 matrix. Return (N,4,4) float64."""
    with open(pose_txt, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if len(lines) % 4 != 0:
        raise ValueError(f"poses.txt lines not multiple of 4: {pose_txt}, got {len(lines)} lines")

    mats = []
    for i in range(0, len(lines), 4):
        block = lines[i:i+4]
        mat = [[float(x) for x in ln.split()] for ln in block]
        mats.append(mat)
    arr = np.array(mats, dtype=np.float64)  # (N,4,4)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"Bad pose shape from {pose_txt}: {arr.shape}")
    return arr


class NRGBDDataset(BaseDataset):
    """
    NRGBD:
      NRGBD/data/<seq>/
        images/img{id}.png
        depth/depth{id}.png
        poses.txt                 # each 4 lines is a 4x4 pose

    Notes:
      - Pose conversion:
          poses.txt gives c2w in GL convention (per your script)
          apply gl->cv: pose[:, :, 1:3] *= -1
          then w2c = inverse(c2w) via closed_form_inverse_se3
      - Depth:
          depth png is in mm -> /1000 to meters
          >10m or <1e-3m treated invalid -> 0
    """

    # NOTE this is hard-coded and kept identical to pi3
    # please check https://github.com/yyfz/Pi3/blob/evaluation/datasets/nrgbd.py
    FX = 554.2562584220408
    FY = 554.2562584220408
    CX = 320.0
    CY = 240.0
    DEPTH_SCALE = 1000.0  # mm -> m

    def __init__(
        self,
        root: str | Path,
        fx: float = FX, fy: float = FY, cx: float = CX, cy: float = CY,
        depth_scale: float = DEPTH_SCALE,
    ):
        super().__init__(root)
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.depth_scale = depth_scale

        self.sequence_list = sorted([d for d in os.listdir(self.root) if osp.isdir(osp.join(self.root, d))])

        # cache: dict[seq] -> w2c_all (N,4,4) float64
        self._w2c_cache = {}
        for seq in self.sequence_list:
            seq_dir = osp.join(self.root, seq)
            pose_txt = osp.join(seq_dir, "poses.txt")
            if not osp.exists(pose_txt):
                raise FileNotFoundError(f"Missing poses.txt: {pose_txt}")

            c2w = _read_poses_txt_as_4x4(pose_txt)  # (N,4,4)
            # gl -> cv (match your script): flip Y/Z axis columns
            c2w[:, :, 1:3] *= -1.0
            w2c = closed_form_inverse_se3(torch.from_numpy(c2w)).cpu().numpy().astype(np.float64)  # (N,4,4)
            self._w2c_cache[seq] = w2c

    def list_sequences(self, split: str) -> List[str]:
        # No official split; keep signature consistent
        return self.sequence_list

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.root, seq)

        def list_suffix(prefix: str, suffix: str):
            return sorted([d for d in os.listdir(os.path.join(seq_path,prefix)) if d.endswith(suffix)])

        rgbs = list_suffix(prefix="images",suffix=".png")
        depths = list_suffix(prefix="depth",suffix=".png")
        pose_txt = osp.join(seq_path, "poses.txt")
        poses = _read_poses_txt_as_4x4(pose_txt) # c2w
        n_pose = poses.shape[0]

        if not (len(rgbs) == len(depths) == n_pose):
            raise ValueError(f"{seq} mismatched: rgb={len(rgbs)} depth={len(depths)} pose={n_pose}")

        return len(rgbs)

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.root, seq)
        images = [osp.join(seq_dir, "images", f"img{i}.png") for i in frame_ids]
        depths = [osp.join(seq_dir, "depth", f"depth{i}.png") for i in frame_ids]
        # poses are in one file; keep per-frame list for SequenceBatch compatibility
        pose_txt = osp.join(seq_dir, "poses.txt")
        poses = [pose_txt for _ in frame_ids]
        return images, poses, depths

    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p).convert("RGB") for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu

    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        d = np.stack([imageio.v2.imread(p).astype(np.float32) for p in depth_paths])  # (V,H,W)
        d = d / self.depth_scale  # mm -> m
        return torch.from_numpy(d)  # (V,H,W) meters

    def _load_w2c(self, seq: str, frame_ids: List[int]) -> torch.Tensor:
        w2c_all = self._w2c_cache[seq]  # (N,4,4)
        w2c_sel = w2c_all[np.array(frame_ids, dtype=np.int64)]
        return torch.from_numpy(w2c_sel)  # (V,4,4)

    def _load_K(self, num_frames: int) -> torch.Tensor:
        K = np.array([[self.fx, 0.0, self.cx],
                      [0.0, self.fy, self.cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        K = torch.from_numpy(K).view(1, 3, 3).repeat(num_frames, 1, 1)
        return K

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)                 # (V,H,W) meters (no resize)
        raw_w2c = self._load_w2c(seq, frame_ids)               # (V,4,4)
        raw_K = self._load_K(num_frames=len(frame_ids))        # (V,3,3)

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
                "dataset": "nrgbd",
                "FXFY_CXCY": (self.FX, self.FY, self.CX, self.CY),
                "DEPTH_SCALE": self.depth_scale,
            },
        )
