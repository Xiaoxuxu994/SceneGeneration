# benchmark/datasets/dl3dv.py
from __future__ import annotations

import os
import os.path as osp
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image, ImageFile

from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3

from .base import BaseDataset, SequenceBatch

def _extract_int(s: str) -> Optional[int]:
    stem = Path(s.replace("\\", "/")).stem
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def _resolve_image_path(seq_dir: Path, file_path: str, image_dir: str = "images_8") -> str:
    fp = file_path.replace("\\", "/")
    if fp.startswith("./"):
        fp = fp[2:]

    base = Path(fp).name
    p = seq_dir / image_dir / base
    if p.exists() and p.is_file():
        return str(p)

    if Path(base).suffix == "":
        for suf in [".png", ".jpg", ".jpeg"]:
            pp = seq_dir / image_dir / (base + suf)
            if pp.exists():
                return str(pp)

    raise FileNotFoundError(f"[DL3DV] cannot resolve image: {file_path} -> {p}")

def _read_cameras_from_transforms(transform_dict: Dict[str, Any]) -> np.ndarray:
    cam_w = float(transform_dict["w"])
    cam_h = float(transform_dict["h"])

    K_norm = np.eye(3, dtype=np.float32)
    K_norm[0, 0] = float(transform_dict["fl_x"]) / cam_w
    K_norm[1, 1] = float(transform_dict.get("fl_y", transform_dict["fl_x"])) / cam_h
    K_norm[0, 2] = float(transform_dict.get("cx", cam_w * 0.5)) / cam_w
    K_norm[1, 2] = float(transform_dict.get("cy", cam_h * 0.5)) / cam_h
    return K_norm


class DL3DVDataset(BaseDataset):
    """
    DL3DV:
      root/<scene_id>/
        images_8/frame_00000.png ...
        transforms.json

    Notes:
      - frames[*].transform_matrix assumed to be c2w (4x4)
      - we always return raw_w2c = inverse(c2w)
      - no GT depth: depths are zeros
    """

    def __init__(
        self,
        root: str | Path,
        image_dir: str = "images_8",
        transforms_name: str = "transforms.json",
    ):
        super().__init__(root)
        self.image_dir = image_dir
        self.transforms_name = transforms_name
        self._cache: Dict[str, Dict[str, Any]] = {}

    def list_sequences(self, split: str = "all") -> List[str]:
        seqs: List[str] = []
        for d in sorted(os.listdir(self.root)):
            seq_dir = Path(self.root) / d
            if not seq_dir.is_dir():
                continue
            if not (seq_dir / self.transforms_name).exists():
                continue
            if not (seq_dir / self.image_dir).is_dir():
                continue
            seqs.append(d)
        return seqs

    def count_seq_len(self, seq: str) -> int:
        return int(self._get_meta(seq)["num_frames"])

    def make_paths(self, seq: str, frame_ids: List[int]) -> Tuple[List[str], List[str], List[str]]:
        meta = self._get_meta(seq)
        img_paths_all: List[str] = meta["image_paths"]
        tf_path: str = meta["transforms_path"]

        images = [img_paths_all[i] for i in frame_ids]
        poses = [f"{tf_path}#{i}" for i in frame_ids]   # transforms.json#idx
        depths = ["" for _ in frame_ids]                # no GT depth
        return images, poses, depths

    # -------- loaders (aligned with 7scenes naming) --------
    def _load_rgb(self, image_paths: List[str]):
        image_pils = [Image.open(p).convert("RGB") for p in image_paths]
        images_cpu = load_and_preprocess_images(image_paths)  # (V,3,H,W) CPU
        return image_pils, images_cpu

    def _load_depth(self, image_pils: List[Image.Image]) -> torch.Tensor:
        W, H = image_pils[0].size
        V = len(image_pils)
        return torch.zeros((V, H, W), dtype=torch.float32)

    def _load_w2c(self, pose_paths: List[str]) -> torch.Tensor:
        if len(pose_paths) == 0:
            return torch.zeros((0, 4, 4), dtype=torch.float64)

        tf_path, _ = pose_paths[0].split("#", 1)
        seq = Path(tf_path).parent.name  # root/<scene_id>/transforms.json

        meta = self._get_meta(seq)
        c2w_all: np.ndarray = meta["c2w"]  # (N,4,4)

        idxs = [int(s.split("#", 1)[1]) for s in pose_paths]
        c2w = torch.from_numpy(c2w_all[idxs].astype(np.float64))  # (V,4,4)

        # always inverse: c2w -> w2c
        w2c = closed_form_inverse_se3(c2w)
        return w2c

    def _load_K(self, seq: str, num_frames: int) -> torch.Tensor:
        K: np.ndarray = self._get_meta(seq)["K"]
        return torch.from_numpy(K.astype(np.float32)).view(1, 3, 3).repeat(num_frames, 1, 1)

    def load_sequence(self, seq: str, frame_ids: List[int]) -> SequenceBatch:
        image_paths, pose_paths, depth_paths = self.make_paths(seq, frame_ids)

        image_pils, images_cpu = self._load_rgb(image_paths)
        depths = self._load_depth(image_pils)
        raw_w2c = self._load_w2c(pose_paths)
        raw_K = self._load_K(seq, num_frames=len(frame_ids))

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
                "dataset": "dl3dv",
                "has_gt_depth": False,
            },
        )

    def _get_meta(self, seq: str) -> Dict[str, Any]:
        if seq in self._cache:
            return self._cache[seq]

        seq_dir = Path(self.root) / seq
        tf_path = seq_dir / self.transforms_name
        if not tf_path.exists():
            raise FileNotFoundError(f"[DL3DV] transforms not found: {tf_path}")

        with open(tf_path, "r") as f:
            data = json.load(f)

        frames = data.get("frames", [])
        if not isinstance(frames, list) or len(frames) == 0:
            raise ValueError(f"[DL3DV] invalid frames in {tf_path}")

        items = []
        for i, fr in enumerate(frames):
            fp = fr.get("file_path", "")
            if not fp:
                raise ValueError(f"[DL3DV] missing frame.file_path in {tf_path}")

            imgp = _resolve_image_path(seq_dir, fp, image_dir=self.image_dir)

            T = np.array(fr.get("transform_matrix", None), dtype=np.float32)
            if T.shape != (4, 4):
                raise ValueError(f"[DL3DV] invalid transform_matrix shape {T.shape} for {fp}")

            n = _extract_int(fp)
            if n is None:
                n = i
            items.append((n, imgp, T))

        items.sort(key=lambda x: x[0])
        image_paths = [x[1] for x in items]
        c2w = np.stack([x[2] for x in items], axis=0).astype(np.float32)

        # image size (real image)
        im0 = Image.open(image_paths[0])
        W, H = im0.size

        # ---- apply transform ----
        extr_trans_mat = np.array(
            [[
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]],
            dtype=np.float32,
        )  # (1,4,4)

        c2w = (extr_trans_mat @ c2w).astype(np.float32)
        c2w[:, 0:3, 1:3] *= -1.0
        
        # --- 平移归一化（会抹掉绝对尺度） ---
        t = c2w[:, :3, 3]                       # (V,3)
        den = np.linalg.norm(t, axis=1).mean()  # 标量
        den = max(float(den), 1e-8)
        c2w[:, :3, 3] = t / den

        # ---- intrinsics: normalize by json w/h, then scale by real W/H ----
        K_norm = _read_cameras_from_transforms(data)
        K = K_norm.copy()
        K[0, :] *= float(W)
        K[1, :] *= float(H)

        meta = {
            "transforms_path": str(tf_path),
            "num_frames": len(image_paths),
            "image_paths": image_paths,
            "c2w": c2w,
            "K": K,          # pixel K after scaling
            "w": W,
            "h": H,
        }
        self._cache[seq] = meta
        return meta
