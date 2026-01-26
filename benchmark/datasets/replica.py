# benchmark/datasets/replica.py
from __future__ import annotations

import os
import os.path as osp
import json
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import torch
import imageio
from PIL import Image

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch


class ReplicaDataset(BaseDataset):
    """
    Replica (common layout you showed):

      Replica/
        cam_params.json
        office0/
          frame000000.jpg
          depth000000.png
          ...
          traj.txt (optional; if missing, fall back to Replica/traj.txt)
        office1/
        ...
        room2/

    Notes:
      - RGB:  frame%06d.jpg
      - Depth: depth%06d.png  (often uint16 in mm; we divide by depth_scale -> meters)
      - Pose: traj.txt
          * default: one pose per line, 16 floats => 4x4 row-major (c2w)
          * we then invert to w2c for output raw_w2c (consistent with SevenScenesDataset)
    """
    DEPTH_SCALE = 1000.0  # mm -> m (common)
    INVALID_DEPTH_RAW = 60000.0 

    def __init__(
        self,
        root: str | Path,
        depth_scale: float = DEPTH_SCALE,
        invalid_depth_raw: float = INVALID_DEPTH_RAW,
    ):
        super().__init__(root)

        self.depth_scale = depth_scale
        self.invalid_depth_raw = invalid_depth_raw

        # runtime context (for selecting poses by frame_ids without changing BaseDataset's call pattern)
        self._ctx_frame_ids: Optional[List[int]] = None

        # try load cam params from root/cam_params.json
        self._cam_params_path = osp.join(self.root, "cam_params.json")
        self._cam_params: Optional[Dict[str, Any]] = self._try_load_cam_params(self._cam_params_path)
        self._apply_cam_params_if_present()

    # -------------------------
    # split / sequences
    # -------------------------
    def list_sequences(self, split: str) -> List[str]:
        """
        Replica often doesn't have TrainSplit/TestSplit.
        We support:
          - sequence_list_{split}.txt if exists
          - sequence_list.txt if exists
          - otherwise: list scene dirs under root (office*, room*, etc.), excluding known non-scene items
        """
        split = (split or "all").lower()

        cand_txts = [
            osp.join(self.root, f"sequence_list_{split}.txt"),
            osp.join(self.root, "sequence_list.txt"),
        ]
        for txt in cand_txts:
            if osp.exists(txt):
                with open(txt, "r") as f:
                    lines = [ln.strip() for ln in f.readlines() if ln.strip()]
                # allow either "office0" or "office0/..." formats; keep as-is
                return lines

        # fallback: list subdirs
        ignore = {"results", "__pycache__"}
        seqs = []
        for d in sorted(os.listdir(self.root)):
            p = osp.join(self.root, d)
            if not osp.isdir(p):
                continue
            if d in ignore:
                continue
            # skip hidden/system dirs
            if d.startswith("."):
                continue
            seqs.append(d)
        return seqs

    def count_seq_len(self, seq: str) -> int:
        seq_path = osp.join(self.root, seq, 'results')

        def list_glob(prefix: str, suffix: str):
            return sorted([fn for fn in os.listdir(seq_path) if fn.startswith(prefix) and fn.endswith(suffix)])

        rgbs = list_glob("frame", ".jpg")
        depths = list_glob("depth", ".png")

        traj_path = osp.join(self.root, seq, "traj.txt")
        traj_n = None
        if traj_path and osp.exists(traj_path):
            try:
                traj = self._load_traj_file(traj_path)  # (N,4,4)
                traj_n = int(traj.shape[0])
            except Exception:
                traj_n = None

        if len(rgbs) != len(depths):
            print(f"[ReplicaDataset] {seq} mismatched: rgb={len(rgbs)} depth={len(depths)}")

        n = len(rgbs)
        if traj_n is not None and traj_n != n:
            print(f"[ReplicaDataset] {seq} mismatched: frames={n} traj={traj_n} (will index by frame_ids)")

        return n

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        seq_dir = osp.join(self.root, seq)
        images = [osp.join(seq_dir, 'results', f"frame{i:06d}.jpg") for i in frame_ids]
        depths = [osp.join(seq_dir, 'results', f"depth{i:06d}.png") for i in frame_ids]

        traj_path = osp.join(self.root, seq, "traj.txt")
        poses = [traj_path for _ in frame_ids]
        return images, poses, depths

    # -------------------------
    # dataset specific loaders
    # -------------------------
    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p) for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu

    def _load_depth(self, depth_paths: List[str]) -> torch.Tensor:
        # depth png can be uint16 or uint32 depending on export; imageio handles both
        d = np.stack([imageio.v2.imread(p) for p in depth_paths], axis=0)  # (V,H,W)
        d = d.astype(np.float32)

        # invalid handling
        if self.invalid_depth_raw > 0:
            d[d >= self.invalid_depth_raw] = 0.0
        d[d <= 0.0] = 0.0

        # scale to meters
        d = d / float(self.depth_scale)
        return torch.from_numpy(d)

    def _load_w2c(self, pose_paths: List[str]) -> torch.Tensor:
        """
        pose_paths are repeated traj.txt paths.
        We use self._ctx_frame_ids (set in load_sequence) to pick corresponding rows.
        """
        if len(pose_paths) == 0:
            raise ValueError("[ReplicaDataset] empty pose_paths")

        traj_path = pose_paths[0]
        arr = np.loadtxt(traj_path)
        traj = arr.reshape(-1, 4, 4).astype(np.float32)

        if self._ctx_frame_ids is None:
            # fallback: take first V poses
            sel = traj[: len(pose_paths)]
        else:
            idx = np.array(self._ctx_frame_ids, dtype=np.int64)
            if idx.max(initial=0) >= traj.shape[0]:
                raise IndexError(
                    f"[ReplicaDataset] frame_id out of range: max(frame_ids)={int(idx.max())} >= traj_len={traj.shape[0]} "
                    f"(traj_path={traj_path})"
                )
            sel = traj[idx]  # (V,4,4)

        c2w = torch.from_numpy(sel.astype(np.float64))
        w2c = closed_form_inverse_se3(c2w)
        return w2c

    def _load_K(self, num_frames: int) -> torch.Tensor:
        K = np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        K = torch.from_numpy(K).view(1, 3, 3).repeat(num_frames, 1, 1)
        return K

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(depth_paths)          # (V,H,W) meters

        # set context for pose indexing
        self._ctx_frame_ids = frame_ids
        try:
            raw_w2c = self._load_w2c(pose_paths)        # (V,4,4)
        finally:
            self._ctx_frame_ids = None

        raw_K = self._load_K(num_frames=len(frame_ids)) # (V,3,3)

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
                "dataset": "replica",
                "DEPTH_SCALE": float(self.depth_scale),
            },
        )

    # -------------------------
    # helpers
    # -------------------------
    def _try_load_cam_params(self, path: str) -> Optional[Dict[str, Any]]:
        if not osp.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ReplicaDataset] failed to load cam_params.json: {path} err={e}")
            return None

    def _apply_cam_params_if_present(self):
        if not self._cam_params:
            return

        # allow nesting like {"camera": {...}} or {"intrinsics": {...}}
        cand_dict = self._cam_params['camera']
        self.fx = cand_dict["fx"]
        self.fy = cand_dict["fy"]
        self.cx = cand_dict["cx"]
        self.cy = cand_dict["cy"]
        self.depth_scale = cand_dict["scale"]

    def _load_traj_file(self, traj_path: str) -> np.ndarray:
        """
        Returns (N,4,4) float32.
        Supports:
          - N x 16 (one pose per line)
          - (4N) x 4 (4 lines per pose)
          - N x 12 (3x4 per pose) -> append [0,0,0,1]
          - single 4x4
        """
        arr = np.loadtxt(traj_path)

        # single pose special cases

        return arr.reshape(-1, 4, 4).astype(np.float32)