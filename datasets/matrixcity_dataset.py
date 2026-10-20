import sys
sys.path.append('.')

import os
import os.path as osp
from glob import glob
import json
import hashlib
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class MatrixCityDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        use_normal=False,
        **kwargs,
    ):
        """
        https://github.com/city-super/MatrixCity
        基于这个脚本更新：
        https://github.com/city-super/MatrixCity/blob/main/scripts/rgbd2pc.py
        """
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "MatrixCity"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.use_normal = use_normal

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)
        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(
            cache_dir,
            f"{self.dataset_label}_{root_hash}_cache.pkl"
        )

        cache_path = self.cache_file or cache_path

        if not fio.exists(cache_path):
            print(f"[Dataset] Constructing cache {cache_path}..")
            self._scan_scenes()
            self._save_cache(
                keys=["all_sequences", "scene_to_path", "scene_meta"],
                save_path=cache_path
            )
            print(f"[Dataset] Cache is saved {cache_path}..")
        else:
            print(f"[Dataset] Loading cache from {cache_path}..")
            res = self._load_cache(cache_path)
            if res:
                self.cache_loaded = True
                print(f"[Dataset] Cache is loaded.")

        self.sequences = self._split_sequences(self.all_sequences)

        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]

        if self.verbose:
            print(f"[{self.dataset_label}] Sequences: {self.sequences}")

        print(
            f"[{self.dataset_label}] Found {len(self.sequences)} unique videos in {self.data_root} ({self.mode})",
            flush=True,
        )

    def __len__(self):
        return len(self.sequences)

    # =========================
    # scan / cache
    # =========================
    def _scan_scenes(self):
        scene_names = [
            x for x in fio.listdir(self.data_root)
            if fio.isdir(osp.join(self.data_root, x))
        ]
        scene_names = sorted(scene_names)

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for scene in tqdm(scene_names, desc="Building MatrixCity cache"):
            scene_path = osp.join(self.data_root, scene)
            rgb_dir = osp.join(scene_path, "rgb")
            depth_dir = osp.join(scene_path, "depth")
            normal_dir = osp.join(scene_path, "normal")
            tf_path = osp.join(scene_path, "transforms_1.json")

            if not (fio.isdir(rgb_dir) and fio.isdir(depth_dir) and fio.isfile(tf_path)):
                continue

            if self.use_normal and not fio.isdir(normal_dir):
                continue

            try:
                with fio.open(tf_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                frames = meta["frames"]
            except Exception as e:
                print(f"[{self.dataset_label}] Skip scene {scene}, failed to load transforms file: {e}")
                continue

            frame_ids = []
            frame_meta = {}

            for i, fr in enumerate(frames):
                try:
                    rel_path = fr["file_path"].replace("\\", "/")
                    stem = osp.splitext(osp.basename(rel_path))[0]

                    rgb_path = osp.join(scene_path, "rgb", stem + ".png")
                    depth_path = osp.join(scene_path, "depth", stem + ".exr")
                    normal_path = osp.join(scene_path, "normal", stem + ".exr")

                    if not fio.isfile(rgb_path) or not fio.isfile(depth_path):
                        continue
                    if self.use_normal and not fio.isfile(normal_path):
                        continue

                    fl_x = fr.get("fl_x", meta["fl_x"])
                    fl_y = fr.get("fl_y", meta.get("fl_y", meta["fl_x"]))
                    cx = fr.get("cx", meta.get("cx", meta["w"] / 2))
                    cy = fr.get("cy", meta.get("cy", meta["h"] / 2))

                    K = np.array([
                        [fl_x, 0, cx],
                        [0, fl_y, cy],
                        [0, 0, 1],
                    ], dtype=np.float32)

                    T = np.array(fr["transform_matrix"], dtype=np.float32)
                    if T.shape != (4, 4):
                        continue

                    fid = str(i)
                    frame_ids.append(fid)

                    rec = {
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "intrinsic": K,
                        "transform_matrix": T,
                    }
                    if self.use_normal:
                        rec["normal_path"] = normal_path

                    frame_meta[fid] = rec
                except Exception:
                    continue

            if len(frame_ids) < self.min_num_images:
                continue

            all_sequences.append(scene)
            scene_to_path[scene] = str(scene_path)
            scene_meta[scene] = {
                "frame_ids": frame_ids,
                "frame_meta": frame_meta,
            }

        self.all_sequences = sorted(all_sequences)
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

    # =========================
    # split
    # =========================
    def _split_sequences(self, all_sequences):
        split_idx = int(len(all_sequences) * self.split_ratio)
        if self.mode == "train":
            return all_sequences[:split_idx]
        return all_sequences[split_idx:]

    # =========================
    # camera helpers
    # =========================
    def _read_exr_depth(self, path):
        """
        读取单通道 EXR depth，和参考脚本保持一致。
        """
        depth = fio.cv2_imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to read EXR depth: {path}")

        if depth.ndim == 3:
            depth = depth[..., 0]
        elif depth.ndim != 2:
            raise ValueError(f"Unexpected depth shape {depth.shape} @ {path}")

        depth = depth.astype(np.float32)
        invalid_mask = (depth == 65504.0)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[invalid_mask] = 0.0
        depth[depth <= 0] = 0.0
        return depth

    def _read_exr_normal(self, path):
        """
        读取3通道 EXR normal。
        """
        normal = fio.cv2_imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if normal is None:
            raise ValueError(f"Failed to read EXR normal: {path}")

        normal = normal.astype(np.float32)
        normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        return normal

    # =========================
    # sample
    # =========================
    # re-implement in basedataset

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, scene_name, fid, frame_meta, resolution, rng):
        rec = frame_meta[fid]

        rgb_image = fio.pil_open(rec["rgb_path"]).convert("RGB")

        depthmap = self._read_exr_depth(rec["depth_path"])
        depthmap = depthmap / 100.0   # 与参考脚本有出入，这里/100, 原始脚本/10000 ->100m
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depthmap[depthmap <= 0] = 0.0
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        normal = None
        if self.use_normal:
            normal = self._read_exr_normal(rec["normal_path"]).astype(np.float32)
            if normal.ndim == 2:
                normal = normal[..., None]
            if normal.shape[-1] > 3:
                normal = normal[..., :3]
            normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.use_normal:
            rgb_image, depthmap, intrinsic_, normal = self._crop_resize_if_necessary(
                rgb_image,
                depthmap,
                rec["intrinsic"].copy(),
                resolution,
                rng=rng,
                info=rec["rgb_path"],
                normal=normal,
                far_mask=None,
            )
        else:
            rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
                rgb_image,
                depthmap,
                rec["intrinsic"].copy(),
                resolution,
                rng=rng,
                info=rec["rgb_path"],
            )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depthmap[depthmap <= 0] = 0.0

        # 和参考脚本对齐：
        # 它是在 cam_coords 上翻 y/z；
        # 等价到 pose 上，就是翻 c2w 的第 1、2 列
        camera_pose = rec["transform_matrix"].copy().astype(np.float32)
        camera_pose[:, 1:3] *= -1

        out = dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose,   # c2w, aligned with reference script
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=scene_name,
            instance=str(fid),
        )

        if self.use_normal:
            out["normal"] = normal.astype(np.float32)

        return out

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        if self.mode == "train":
            index = index % len(self.sequences)

        scene = self.sequences[index]
        scene_path = self.scene_to_path[scene]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frame_ids"], rng)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    scene_name=scene,
                    fid=fid,
                    frame_meta=meta["frame_meta"],
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.matrixcity_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(MatrixCityDataset))
