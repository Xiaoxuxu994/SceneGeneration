import sys
sys.path.append('.')

import os
import os.path as osp
import pickle
import random
import hashlib
from glob import glob
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class KubricDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        mask_background=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "Kubric"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.mask_background = mask_background

        #
        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # 新增：只缓存当前 scene 的 segmentation，避免重复读
        self._cached_seg_scene = None
        self._cached_segmentations = None

        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)

        # 改回原来的轻量 cache
        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")

        cache_path = self.cache_file or cache_path

        if not fio.exists(cache_path):
            print(f"[Dataset] Constructing cache {cache_path}..")
            self._scan_scenes()
            self._save_cache(keys=["all_sequences", "scene_to_path", "scene_meta"],save_path=cache_path)
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

        for scene in tqdm(scene_names, desc="Building Kubric cache"):
            scene_path = osp.join(self.data_root, scene)
            rgb_dir = os.path.join(scene_path , "frames")
            depth_dir = os.path.join(scene_path , "depths")
            cam_file = os.path.join(scene_path , f"{scene}.npy")

            if not (fio.isdir(rgb_dir) and fio.isdir(depth_dir) and fio.isfile(cam_file)):
                continue

            try:
                cam_data = fio.np_load(str(cam_file), allow_pickle=True).item()["camera"]
            except Exception as e:
                print(f"[Kubric] Skip scene {scene}, failed to load camera file: {e}")
                continue

            rgb_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in fio.glob(osp.join(str(rgb_dir), "*.png"))
            }
            depth_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in fio.glob(osp.join(str(depth_dir), "*.npy"))
            }

            frame_ids = sorted(list(rgb_ids & depth_ids), key=lambda x: int(x))

            if len(frame_ids) < self.min_num_images:
                continue

            all_sequences.append(scene)
            scene_to_path[scene] = str(scene_path)
            scene_meta[scene] = {
                "frame_ids": frame_ids,
                "camera_data": cam_data,
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
    def _build_intrinsic(self, H, W, focal_length, sensor_width):
        """
        Follow the original Kubric official camera convention:
            f_x = focal_length / sensor_width
            f_y = focal_length / sensor_width * H / W
            p_x = 0.5
            p_y = 0.5

        Then convert to pixel-space OpenCV intrinsics.
        """
        f_x = focal_length / sensor_width
        f_y = focal_length / sensor_width * H / W
        p_x = 0.5
        p_y = 0.5

        intrinsic = np.array(
            [
                [f_x * W, 0.0, p_x * W],
                [0.0, f_y * H, p_y * H],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return intrinsic

    def _build_camera_pose_c2w_opencv(self, position, quaternion_wxyz):
        """
        Kubric camera quaternion is in [w, x, y, z].
        scipy expects [x, y, z, w].

        Original code path:
            1. build matrix_world (camera-to-world) in Kubric/OpenGL convention
            2. flip local camera y/z axes to convert GL -> CV
            3. if needed, invert to get w2c

        Here we directly return c2w in OpenCV convention.
        """
        quat_xyzw = np.concatenate([quaternion_wxyz[1:], quaternion_wxyz[0:1]])
        rot = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)

        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[:3, :3] = rot
        camera_pose[:3, 3] = np.asarray(position, dtype=np.float32)

        # GL -> CV on camera coordinates:
        # equivalent to multiplying c2w by diag(1, -1, -1, 1) on the right
        camera_pose[:3, 1:3] *= -1.0

        return camera_pose

    def _get_segmentation_mask(self, segmentations, idx):
        """
        segmentations[idx]: (H, W, 1) or (H, W)
        foreground: seg > 0
        """
        seg = segmentations[idx]
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[..., 0]
        fg_mask = (seg > 0)
        return fg_mask

    # 新增：按 scene 懒加载 segmentation
    def _get_segmentations_for_scene(self, scene_name, scene_path):
        if self._cached_seg_scene == scene_name and self._cached_segmentations is not None:
            return self._cached_segmentations

        scene_npy_path = osp.join(scene_path, f"{scene_name}.npy")
        full_data = fio.np_load(scene_npy_path, allow_pickle=True).item()
        segmentations = full_data["segmentations"]

        self._cached_seg_scene = scene_name
        self._cached_segmentations = segmentations
        return segmentations

    # =========================
    # sample
    # =========================
    # re-implement in basedataset

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, scene_name, fid, camera_data, segmentations, resolution, rng):
        impath = osp.join(scene_path, "frames", f"{fid}.png")
        depthpath = osp.join(scene_path, "depths", f"{fid}.npy")

        rgb_image = fio.pil_open(impath).convert("RGB")
        depthmap = fio.np_load(depthpath).astype(np.float32)

        # 清理 depth
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

        idx = int(fid)

        # 用 segmentation 构造前景 mask，只 mask depth，RGB 保持不变
        if self.mask_background and segmentations is not None:
            fg_mask = self._get_segmentation_mask(segmentations, idx)
            depthmap *= fg_mask.astype(np.float32)

        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        W0, H0 = rgb_image.size

        focal_length = float(camera_data["focal_length"])
        sensor_width = float(camera_data["sensor_width"])
        intrinsic = self._build_intrinsic(H0, W0, focal_length, sensor_width)

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=impath,
        )

        position = camera_data["positions"][idx]
        quaternion = camera_data["quaternions"][idx]  # [w, x, y, z]
        camera_pose = self._build_camera_pose_c2w_opencv(position, quaternion)

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose.astype(np.float32),          # c2w
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=scene_name,
            instance=str(fid),
        )

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

        # 只在需要时加载当前 scene 的 segmentation
        segmentations = None
        if self.mask_background:
            segmentations = self._get_segmentations_for_scene(scene, scene_path)

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    scene_name=scene,
                    fid=fid,
                    camera_data=meta["camera_data"],
                    segmentations=segmentations,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.kubric_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(KubricDataset))
