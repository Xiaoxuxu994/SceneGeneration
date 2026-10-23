# benchmark/datasets/scannetpp_r3.py
from __future__ import annotations

import os
import os.path as osp
import pickle
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from lib.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch, DatasetConfig
from .scannetpp import ScanNetPPDataset


class ScanNetPPR3Dataset(ScanNetPPDataset):
    """
    ScanNet++ DSLR (undistorted) in the reliev3r layout used by the SceneGeneration training dataloader
    (datasets/scannetpp_dataset.py), with the same train / held-out split:
      <data_root>/
        <scene_id>/
          images/<name>.jpg
          depth/<name>.png            uint16, millimetres
          scene_metadata.npz          images: frame names, trajectories: (N,4,4) cam->world (OpenCV),
                                      intrinsics: (N,3,3) at image resolution
        sequence_list_v2.pkl          optional; written by the training dataset (scene list + frame lists)

    Notes:
      - sequences: scenes sorted by name that have images/ and depth/ and >= MIN_NUM_IMAGES frames; split "test"
        = the last (1 - SPLIT_RATIO) of them (training validation), split "train" = the rest. When
        sequence_list_v2.pkl exists it is used as is, exactly as in training.
      - frame order (frame id i = i-th frame) = training order: DSC counter, wrapped 09999 -> 00001 tails last.
      - image / depth resize and crop are inherited from ScanNetPPDataset (same config fields).
      - poses / intrinsics come from scene_metadata.npz instead of transforms.json; returns w2c.
    """

    SPLIT_RATIO = 0.9          # SceneGeneration data/scannetpp_only.yaml: split_ratio
    MIN_NUM_IMAGES = 24        # SceneGeneration ScannetppDataset default
    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, config: DatasetConfig):
        BaseDataset.__init__(self, config)   # ScanNetPPDataset.__init__ reads test.txt of the v2 layout

        all_sequences, self.frames = self._scan_scenes(self.config.data_root)
        n_train = int(len(all_sequences) * self.SPLIT_RATIO)
        self.splits = {"train": all_sequences[:n_train], "test": all_sequences[n_train:], "all": all_sequences}
        self.sequence_list = self.splits["test"]
        self.frame_id2name = {seq: dict(enumerate(self.frames[seq])) for seq in all_sequences}
        self._meta_cache = {}
        print(f"[ScanNetPP-R3] {len(all_sequences)} scenes in {self.config.data_root}: "
              f"{len(self.splits['train'])} train / {len(self.splits['test'])} test (split_ratio {self.SPLIT_RATIO})")

    # ---------------- scene list / frame order (as in SceneGeneration ScannetppDataset) ----------------
    @staticmethod
    def _frame_order(frames):
        def num(s):
            try:
                return int(s[-5:])
            except (ValueError, TypeError):
                return 0

        nums = [num(f) for f in frames]
        wrapped = bool(nums) and min(nums) < 1000 and max(nums) > 9000
        key = (lambda f: num(f) + (10000 if num(f) < 5000 else 0)) if wrapped else num
        return sorted(frames, key=key)

    def _scan_scenes(self, root):
        pkl = osp.join(root, "sequence_list_v2.pkl")
        if osp.exists(pkl):
            with open(pkl, "rb") as f:
                sequence_list, data_store = pickle.load(f)
        else:
            sequence_list, data_store = [], {}
            for scene in sorted(d for d in os.listdir(root) if osp.isdir(osp.join(root, d))):
                rgb_dir, depth_dir = osp.join(root, scene, "images"), osp.join(root, scene, "depth")
                if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir)):
                    continue
                rgb = [osp.splitext(f)[0] for f in os.listdir(rgb_dir) if f.lower().endswith(self.IMAGE_EXTS)]
                dep = [osp.splitext(f)[0] for f in os.listdir(depth_dir) if f.lower().endswith(".png")]
                frames = self._frame_order(list(set(rgb) & set(dep)))
                if len(frames) < self.MIN_NUM_IMAGES:
                    continue
                data_store[scene] = frames
                sequence_list.append(scene)
        seqs = [s for s in sequence_list if len(data_store[s]) >= self.MIN_NUM_IMAGES]
        return seqs, {s: list(data_store[s]) for s in seqs}

    def list_sequences(self, split: str = "test") -> List[str]:
        return self.splits.get(split, self.splits["test"])

    def count_seq_len(self, seq: str) -> int:
        return len(self.frames[seq])

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str], str]:
        seq_dir = osp.join(self.config.data_root, seq)
        names = [self.frame_id2name[seq][i] for i in frame_ids]
        images = [osp.join(seq_dir, "images", f"{n}.jpg") for n in names]
        depths = [osp.join(seq_dir, "depth", f"{n}.png") for n in names]
        cameras = osp.join(seq_dir, "scene_metadata.npz")
        return names, images, depths, cameras

    # ---------------- cameras from scene_metadata.npz ----------------
    def _load_camera(self, camera_path: str) -> dict:
        if camera_path not in self._meta_cache:
            m = np.load(camera_path)
            self._meta_cache = {camera_path: {
                "rows": {str(n): i for i, n in enumerate(m["images"])},
                "c2w": m["trajectories"].astype(np.float64),
                "K": m["intrinsics"].astype(np.float64),
            }}
        return self._meta_cache[camera_path]

    def _load_w2c(self, image_names: List[str], cameras: dict) -> torch.Tensor:
        c2w = np.stack([cameras["c2w"][cameras["rows"][n]] for n in image_names])
        return closed_form_inverse_se3(torch.from_numpy(c2w))

    def _load_K(self, image_names: List[str], cameras: dict, image_hw) -> Tuple[torch.Tensor, torch.Tensor]:
        H0, W0 = image_hw
        K = torch.from_numpy(np.stack([cameras["K"][cameras["rows"][n]] for n in image_names]))
        norm_K = self._normalize_K(K, H0, W0)
        pix_K = self._denormalize_K(norm_K, self.config.rgb_origin_W, self.config.rgb_origin_H)

        if self.config.crop_left + self.config.crop_top > 0:
            pix_K[..., 0, 2] -= self.config.crop_left
            pix_K[..., 1, 2] -= self.config.crop_top

        (actual_H, actual_W) = (
            self.config.rgb_origin_H - self.config.crop_top - self.config.crop_bottom,
            self.config.rgb_origin_W - self.config.crop_left - self.config.crop_right,
        )
        norm_K = self._normalize_K(pix_K, actual_H, actual_W)

        if self.config.rgb_resize_W != actual_W or self.config.rgb_resize_H != actual_H:
            pix_K = self._denormalize_K(norm_K, self.config.rgb_resize_W, self.config.rgb_resize_H)

        return norm_K, pix_K

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_names, image_paths, depth_paths, camera_path = self.make_paths(seq, frame_ids)

        with Image.open(image_paths[0]) as im:
            W0, H0 = im.size                                       # K in scene_metadata is at this resolution
        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)                     # (V,H,W) meters
        cameras = self._load_camera(camera_path)
        raw_w2c = self._load_w2c(image_names, cameras)             # (V,4,4)
        norm_K, pix_K = self._load_K(image_names, cameras, (H0, W0))

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
                "dataset": "scannetpp_r3",
                "DEPTH_SCALE": self.config.depth_scale,
            },
        )
