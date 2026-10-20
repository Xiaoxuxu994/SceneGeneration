import sys
sys.path.append('.')

import os
import os.path as osp
import hashlib

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as SciR

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class TartanAirV2Dataset(BaseDataset):
    def __init__(
        self,
        data_root=None,                 # root containing P000, P001, ...
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        camera_name="lcam_front",
        **kwargs,
    ):
        super().__init__(**kwargs)
        '''
            https://tartanair.org/modalities.html
        '''
        assert data_root is not None

        self.dataset_label = "TartanAirV2"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.camera_name = camera_name

        self.is_metric_scale = True

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # official raw pinhole intrinsics
        self.intrinsic = np.array(
            [[320.0, 0.0, 320.0],
             [0.0, 320.0, 320.0],
             [0.0,   0.0,   1.0]],
            dtype=np.float32,
        )

        # convert OpenCV camera coords (x right, y down, z forward)
        # to TartanAir NED camera coords (x forward, y right, z down)
        self.T_ned_from_cv = np.array(
            [[0.0, 0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)
        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_{self.camera_name}_cache.pkl")

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
        traj_names = sorted([
            x for x in fio.listdir(self.data_root)
            if fio.isdir(osp.join(self.data_root, x))
        ])

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for traj in tqdm(traj_names, desc="Building TartanAirV2 cache"):
            traj_path = osp.join(self.data_root, traj)

            rgb_dir = osp.join(traj_path, f"image_{self.camera_name}")
            depth_dir = osp.join(traj_path, f"depth_{self.camera_name}")
            pose_path = osp.join(traj_path, f"pose_{self.camera_name}.txt")

            if not (fio.isdir(rgb_dir) and fio.isdir(depth_dir) and fio.isfile(pose_path)):
                continue

            try:
                poses = fio.loadtxt(pose_path).astype(np.float32)
                poses = np.atleast_2d(poses)
            except Exception as e:
                print(f"[TartanAirV2] Skip {traj}, pose load failed: {e}")
                continue

            rgb_files = sorted([f for f in fio.listdir(rgb_dir) if f.endswith(".png")])
            depth_files = sorted([f for f in fio.listdir(depth_dir) if f.endswith(".png")])

            rgb_ids = {f.split("_", 1)[0] for f in rgb_files}
            depth_ids = {f.split("_", 1)[0] for f in depth_files}
            frame_ids = sorted(list(rgb_ids & depth_ids), key=int)

            if len(frame_ids) < self.min_num_images:
                continue

            valid_frame_ids = []
            camera_info = {}
            file_info = {}

            for fid in frame_ids:
                idx = int(fid)
                if idx >= len(poses):
                    continue

                img_path = osp.join(rgb_dir, f"{fid}_{self.camera_name}.png")
                depth_path = osp.join(depth_dir, f"{fid}_{self.camera_name}_depth.png")

                if not (fio.isfile(img_path) and fio.isfile(depth_path)):
                    continue

                camera_pose = self._pose_row_to_camera_pose(poses[idx])

                valid_frame_ids.append(fid)
                camera_info[fid] = {
                    "camera_pose": camera_pose.astype(np.float32),          # c2w in your pipeline convention
                    "camera_intrinsics": self.intrinsic.astype(np.float32),
                }
                file_info[fid] = {
                    "img_path": img_path,
                    "depth_path": depth_path,
                }

            if len(valid_frame_ids) < self.min_num_images:
                continue

            all_sequences.append(traj)
            scene_to_path[traj] = traj_path
            scene_meta[traj] = {
                "frame_ids": valid_frame_ids,
                "camera_info": camera_info,
                "file_info": file_info,
            }

        self.all_sequences = sorted(all_sequences)
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

    def _pose_row_to_camera_pose(self, pose_row):
        """
        pose_row: [tx, ty, tz, qx, qy, qz, qw]
        Official docs say poses are in NED frame.
        We build c2w_ned first, then convert the camera local axes to OpenCV style.
        """
        position = pose_row[:3]
        quaternion = pose_row[3:7]

        rot = SciR.from_quat(quaternion)
        c2w_ned = np.eye(4, dtype=np.float32)
        c2w_ned[:3, :3] = rot.as_matrix().astype(np.float32)
        c2w_ned[:3, 3] = position.astype(np.float32)

        # keep world in dataset frame, convert local camera basis from cv -> ned
        camera_pose = c2w_ned @ self.T_ned_from_cv
        return camera_pose.astype(np.float32)

    # =========================
    # split
    # =========================
    def _split_sequences(self, all_sequences):
        split_idx = int(len(all_sequences) * self.split_ratio)
        if self.mode == "train":
            return all_sequences[:split_idx]
        return all_sequences[split_idx:]

    # =========================
    # sample
    # =========================
    def _sample_frame_ids(self, frame_ids, rng):
        n = len(frame_ids)
        assert n > 0

        if self.frame_num > 16 and rng.random() < self.random_sample_thres:
            pos = rng.choice(n, size=self.frame_num, replace=(n < self.frame_num))
            return [frame_ids[int(i)] for i in pos]

        ref = int(rng.integers(0, n))
        max_dist = int(self.max_distance / 8 * self.frame_num)

        start = max(0, ref - max_dist)
        end = min(n, ref + max_dist + 1)
        pool = np.arange(start, end)

        extra = rng.choice(pool, size=self.frame_num - 1, replace=(len(pool) < self.frame_num - 1))
        pos = [ref] + list(extra)
        return [frame_ids[int(i)] for i in pos]

    # =========================
    # helpers
    # =========================
    def _read_rgb(self, path):
        return fio.pil_open(path).convert("RGB")

    def _read_depth(self, path):
        """
        Official decoding:
            depth_rgba = fio.cv2_imread(depthpath, cv2.IMREAD_UNCHANGED)
            depth = depth_rgba.view('<f4').squeeze(axis=-1)
        """
        depth_rgba = fio.cv2_imread(path, cv2.IMREAD_UNCHANGED)
        if depth_rgba is None:
            raise ValueError(f"Failed to read depth png: {path}")

        if depth_rgba.ndim != 3 or depth_rgba.shape[2] != 4:
            raise ValueError(f"Unexpected decoded depth shape {depth_rgba.shape} @ {path}")

        depth = depth_rgba.view("<f4").squeeze(axis=-1).astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth <= 0)] = 0.0
        depth = threshold_depth_map(depth, min_percentile=-1, max_percentile=98)
        return depth.astype(np.float32)

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, traj_name, fid, cam_info, file_info, resolution, rng):
        img_path = file_info["img_path"]
        depth_path = file_info["depth_path"]

        rgb_image = self._read_rgb(img_path)
        depthmap = self._read_depth(depth_path)
        intrinsics = cam_info["camera_intrinsics"].copy()

        rgb_image, depthmap, intrinsics_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsics.copy(),
            resolution,
            rng=rng,
            info=img_path,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depthmap[(depthmap <= 0)] = 0.0

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=cam_info["camera_pose"].astype(np.float32),   # c2w
            camera_intrinsics=intrinsics_.astype(np.float32),
            dataset=self.dataset_label,
            label=traj_name,
            instance=fid,
        )

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        if self.mode == "train":
            index = index % len(self.sequences)

        traj = self.sequences[index]
        meta = self.scene_meta[traj]

        idxs = self._sample_frame_ids(meta["frame_ids"], rng)

        self.this_views_info = {
            "scene": traj,
            "idxs": idxs,
        }

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    traj_name=traj,
                    fid=fid,
                    cam_info=meta["camera_info"][fid],
                    file_info=meta["file_info"][fid],
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.tartanairv2_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(TartanAirV2Dataset))
