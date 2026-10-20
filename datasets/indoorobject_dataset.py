import sys
sys.path.append(".")

import os
# 必须在 import cv2 前设置，否则 OpenCV 可能打不开 EXR
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import os.path as osp
import json
import hashlib
from glob import glob

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


class IndoorObjectRenderDataset(BaseDataset):
    """
    IndoorObjectRender dataset.

    Strict expected structure:

    data_root/
      01/
        01_Furniture/
          benches/
            Benches_005/
              rgb/
                00000.png
                00001.png
                ...
              depth/
                00000.exr
                00001.exr
                ...
              colmap/
              transforms.json

          beds/
            Beds_xxx/
              rgb/
              depth/
              transforms.json

        02_Home_Decoration/
          ...
    """

    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "IndoorObjectRender"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)

        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = osp.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")

        cache_path = self.cache_file or cache_path

        if not fio.exists(cache_path):
            print(f"[Dataset] Constructing cache {cache_path}..")
            self._scan_scenes()
            self._save_cache(
                keys=["all_sequences", "scene_to_path", "scene_meta"],
                save_path=cache_path,
            )
            print(f"[Dataset] Cache is saved {cache_path}..")
        else:
            print(f"[Dataset] Loading cache from {cache_path}..")
            res = self._load_cache(cache_path)
            if res:
                self.cache_loaded = True
                print("[Dataset] Cache is loaded.")

        self.sequences = self._split_sequences(self.all_sequences)

        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]

        print(
            f"[{self.dataset_label}] Found {len(self.sequences)} unique videos "
            f"in {self.data_root} ({self.mode})",
            flush=True,
        )

        if self.verbose:
            print(f"[{self.dataset_label}] Example sequences:")
            for s in self.sequences[:10]:
                print("  ", s)

    def __len__(self):
        return len(self.sequences)

    # ============================================================
    # scan / cache
    # ============================================================
    def _scan_scenes(self):
        """
        Strict scan pattern:

            data_root/*/*/*/*

        Example:

            IndoorObjectRender/01/01_Furniture/benches/Benches_005
        """
        object_dirs = sorted(
            fio.glob(osp.join(self.data_root, "*", "*", "*", "*")),
            key=self._sequence_sort_key,
        )

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        print(f"[IndoorObjectRender] Scanning object dirs: {len(object_dirs)}")

        for object_dir in tqdm(object_dirs, desc="Building IndoorObjectRender cache"):
            if not fio.isdir(object_dir):
                continue

            rgb_dir = osp.join(object_dir, "rgb")
            depth_dir = osp.join(object_dir, "depth")
            transform_file = osp.join(object_dir, "transforms.json")

            if not fio.isdir(rgb_dir):
                if self.verbose:
                    print(f"[IndoorObjectRender] Skip no rgb dir: {object_dir}")
                continue

            if not fio.isdir(depth_dir):
                if self.verbose:
                    print(f"[IndoorObjectRender] Skip no depth dir: {object_dir}")
                continue

            if not fio.isfile(transform_file):
                if self.verbose:
                    print(f"[IndoorObjectRender] Skip no transforms.json: {transform_file}")
                continue

            try:
                transform_data = self._load_transform_json(transform_file)
                frame_meta = self._build_frame_meta(transform_data)
            except Exception as e:
                if self.verbose:
                    print(
                        f"[IndoorObjectRender] Skip bad transforms.json: "
                        f"{transform_file}, err={e}"
                    )
                continue

            rgb_ids = {
                osp.splitext(osp.basename(p))[0]
                for p in fio.glob(osp.join(rgb_dir, "*.png"))
            }

            depth_ids = {
                osp.splitext(osp.basename(p))[0]
                for p in fio.glob(osp.join(depth_dir, "*.exr"))
            }

            json_ids = set(frame_meta.keys())

            frame_ids = sorted(
                list(rgb_ids & depth_ids & json_ids),
                key=self._frame_sort_key,
            )

            if len(frame_ids) < self.min_num_images:
                if self.verbose:
                    print(
                        f"[IndoorObjectRender] Skip insufficient frames: {object_dir}, "
                        f"rgb={len(rgb_ids)}, depth={len(depth_ids)}, "
                        f"json={len(json_ids)}, valid={len(frame_ids)}"
                    )
                continue

            # 例如：01/01_Furniture/benches/Benches_005
            seq_name = osp.relpath(object_dir, self.data_root)

            all_sequences.append(seq_name)
            scene_to_path[seq_name] = object_dir
            scene_meta[seq_name] = {
                "frame_ids": frame_ids,
                "transform_file": transform_file,
            }

        self.all_sequences = sorted(all_sequences, key=self._sequence_name_sort_key)
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

        print(f"[IndoorObjectRender] Total valid sequences: {len(self.all_sequences)}")

    def _load_transform_json(self, transform_file):
        with fio.open(transform_file, "r") as f:
            data = json.load(f)

        assert "frames" in data, f"No frames in {transform_file}"
        return data

    def _build_frame_meta(self, transform_data):
        frame_meta = {}

        for fr in transform_data["frames"]:
            file_path = fr["file_path"]
            fid = osp.splitext(osp.basename(file_path))[0]
            frame_meta[fid] = fr

        return frame_meta

    # ============================================================
    # split
    # ============================================================
    def _split_sequences(self, all_sequences):
        split_idx = int(len(all_sequences) * self.split_ratio)

        if self.mode == "train":
            return all_sequences[:split_idx]
        return all_sequences[split_idx:]

    # ============================================================
    # IO
    # ============================================================
    def _load_rgb(self, path):
        return fio.pil_open(path).convert("RGB")

    def _load_depth_exr(self, path):
        depth = fio.cv2_imread(path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise RuntimeError(f"Failed to read EXR depth: {path}")

        if depth.ndim == 3:
            depth = depth[..., 0]

        depth = depth.astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0

        return depth

    # ============================================================
    # camera
    # ============================================================
    def _build_intrinsic(self, transform_data, image_size):
        """
        Build pixel-space OpenCV intrinsic K.

        Expected JSON fields:
            fl_x, fl_y, cx, cy, w, h

        If actual image size differs from json w/h, K is scaled.
        """
        W0, H0 = image_size

        json_w = float(transform_data.get("w", W0))
        json_h = float(transform_data.get("h", H0))

        fl_x = 599.9999771118164
        fl_y = 599.9999771118164 ## hardcode get from guoxing

        # if "fl_x" in transform_data:
        #     fl_x = float(transform_data["fl_x"])
        # else:
        #     angle_x = float(transform_data["camera_angle_x"])
        #     fl_x = 0.5 * json_w / np.tan(0.5 * angle_x)

        # if "fl_y" in transform_data:
        #     fl_y = float(transform_data["fl_y"])
        # else:
        #     if "camera_angle_y" in transform_data:
        #         angle_y = float(transform_data["camera_angle_y"])
        #         fl_y = 0.5 * json_h / np.tan(0.5 * angle_y)
        #     else:
        #         fl_y = fl_x

        cx = float(transform_data.get("cx", json_w * 0.5))
        cy = float(transform_data.get("cy", json_h * 0.5))

        sx = float(W0) / json_w
        sy = float(H0) / json_h

        intrinsic = np.array(
            [
                [fl_x * sx, 0.0, cx * sx],
                [0.0, fl_y * sy, cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        return intrinsic

    def _build_pose(self, transform_matrix):
        """
        Input transform_matrix is assumed to be c2w in Blender / NeRF / OpenGL convention.

        Hard-coded conversion to OpenCV camera convention:

            c2w_cv = c2w_gl @ diag(1, -1, -1, 1)

        Equivalent:
            pose[:3, 1:3] *= -1
        """
        extr_trans_mat = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        pose = np.asarray(transform_matrix, dtype=np.float32)

        pose = pose.copy()
        pose = extr_trans_mat @ np.array(pose)
        pose[:3, 1:3] *= -1.0 #c2w
        # camera_pose = np.linalg.inv(pose).astype(np.float32)
        assert pose.shape == (4, 4), f"Bad pose shape: {pose.shape}"

        return pose.astype(np.float32)

    # ============================================================
    # load one view
    # ============================================================
    def _load_one_view(
        self,
        seq_path,
        seq_name,
        fid,
        transform_data,
        frame_meta,
        resolution,
        rng,
    ):
        rgb_path = osp.join(seq_path, "rgb", f"{fid}.png")
        depth_path = osp.join(seq_path, "depth", f"{fid}.exr")

        rgb_image = self._load_rgb(rgb_path)
        depthmap = self._load_depth_exr(depth_path)

        W0, H0 = rgb_image.size
        intrinsic = self._build_intrinsic(transform_data, image_size=(W0, H0))

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=rgb_path,
        )

        frame_info = frame_meta[fid]
        camera_pose = self._build_pose(frame_info["transform_matrix"])

        depthmap = np.nan_to_num(
            depthmap,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        # depthmap = threshold_depth_map(
        #     depthmap,
        #     min_percentile=-1,
        #     max_percentile=98,
        # )

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=seq_name,
            instance=str(fid),
        )

    # ============================================================
    # main interface
    # ============================================================
    def _get_views(self, index, resolution, rng):
        if self.mode == "train":
            index = index % len(self.sequences)

        seq_name = self.sequences[index]
        seq_path = self.scene_to_path[seq_name]
        meta = self.scene_meta[seq_name]

        idxs = self._sample_frame_ids(meta["frame_ids"], rng)

        self.this_views_info = {
            "scene": seq_name,
            "idxs": idxs,
        }

        transform_data = self._load_transform_json(meta["transform_file"])
        frame_meta = self._build_frame_meta(transform_data)

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    seq_path=seq_path,
                    seq_name=seq_name,
                    fid=fid,
                    transform_data=transform_data,
                    frame_meta=frame_meta,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views

    # ============================================================
    # sort helpers
    # ============================================================
    @staticmethod
    def _frame_sort_key(x):
        try:
            return int(x)
        except Exception:
            return x

    @staticmethod
    def _natural_key(s):
        """
        Natural-ish key for stable sorting.

        Example:
            01 -> 1
            Benches_005 -> ["Benches_", 5]
        """
        import re

        parts = re.split(r"(\d+)", str(s))
        key = []
        for p in parts:
            if p.isdigit():
                key.append(int(p))
            else:
                key.append(p.lower())
        return key

    @classmethod
    def _sequence_sort_key(cls, object_dir):
        """
        Sort full path:

            data_root / 01 / 01_Furniture / benches / Benches_005
        """
        parts = osp.normpath(object_dir).split(os.sep)

        # use last 4 levels after data_root
        if len(parts) >= 4:
            return tuple(cls._natural_key(p) for p in parts[-4:])
        return tuple(cls._natural_key(p) for p in parts)

    @classmethod
    def _sequence_name_sort_key(cls, seq_name):
        """
        Sort relative path:

            01/01_Furniture/benches/Benches_005
        """
        parts = osp.normpath(seq_name).split(os.sep)
        return tuple(cls._natural_key(p) for p in parts)


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.indoorobject_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(IndoorObjectRenderDataset))
