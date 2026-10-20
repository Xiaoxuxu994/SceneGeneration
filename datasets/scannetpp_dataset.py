import sys
sys.path.append('.')

import os
import os.path as osp
from glob import glob

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import hashlib
import pickle

FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import rdepth_to_zdepth, threshold_depth_map
from datasets.base.utils import read_any_file, parse_colmap_to_dict

def pos_quat2SE(quat_data):
    from scipy.spatial.transform import Rotation as R

    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.eye(4, dtype=np.float32)
    SE[:3, :3] = SO.astype(np.float32)
    SE[:3, 3] = np.asarray(quat_data[:3], dtype=np.float32)
    return SE


def get_trans_device_from_camera():
    trans_device_from_camera = np.array(
        [[ 0.99606003, -0.04388682,  0.07706079, -0.00753010],
         [ 0.08210934,  0.78468796, -0.61442889, -0.01090855],
         [-0.03350334,  0.61833547,  0.78519983, -0.00359806],
         [ 0.        ,  0.        ,  0.        ,  1.        ]],
        dtype=np.float32
    )
    rot90 = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]], dtype=np.float32)
    trans_device_from_camera[:3, :3] = trans_device_from_camera[:3, :3] @ rot90
    return trans_device_from_camera


class ScannetppDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        min_num_images: int = 24,
        split_ratio=0.9,
        seq_num=-1,
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "Scannetpp"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.ScanNetPP_DIR = data_root
        self.trans_device_from_camera = get_trans_device_from_camera()

        #
        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # 1) load/save cache
        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)
        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")
        cache_path = self.cache_file or cache_path
        if not fio.exists(cache_path):
            print(f'[Dataset] Constructing cache {cache_path}..')
            self._scan_scenes()
            self._save_cache(keys=["all_sequences", "scene_to_path", "scene_meta"],save_path=cache_path)
            print(f'[Dataset] Cache is saved {cache_path}..')
        else:
            print(f'[Dataset] Loading cache from {cache_path}..')
            res = self._load_cache(cache_path)
            if res:
                self.cache_loaded = True
                print(f'[Dataset] Cache is loaded.')

        # 2) train/val split
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
        sequence_list_path = f"{self.ScanNetPP_DIR}/sequence_list.pkl"
        if fio.exists(sequence_list_path):
            with fio.open(sequence_list_path, "rb") as f:
                sequence_list, data_store = pickle.load(f)
        else:

            def dsc_sort_key(s):
                try:
                    return int(s[-5:])
                except (ValueError, TypeError):
                    return 0

            data_store = {}
            sequence_list = []

            scenes = sorted(fio.listdir(self.ScanNetPP_DIR))

            for scene in scenes:
                rgb_dir = osp.join(self.ScanNetPP_DIR, scene, "images")
                depth_dir = osp.join(self.ScanNetPP_DIR, scene, "depth")

                rgb_files = [os.path.splitext(f)[0] for f in fio.listdir(rgb_dir)]
                depth_files = [os.path.splitext(f)[0] for f in fio.listdir(depth_dir)]

                frames = sorted(list(set(rgb_files) & set(depth_files)), key=dsc_sort_key)
                if len(frames) < self.min_num_images:
                    continue

                data_store[scene] = frames
                sequence_list.append(scene)

            with fio.open(sequence_list_path, "wb") as f:
                pickle.dump((sequence_list, data_store), f)

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}
        for scene_name in sequence_list:
            frames = data_store[scene_name]
            if len(frames) >= self.min_num_images:
                all_sequences.append(scene_name)
                scene_to_path[scene_name] = osp.join(self.ScanNetPP_DIR, scene_name)
                scene_meta[scene_name] = {
                    "frames_id": list(range(len(frames))),
                    "frames_name": frames
                }
        self.all_sequences = all_sequences
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
    def _build_intrinsic(self, H, W):
        return np.array(
            [
                [0.42278125 * W, 0.0, 0.5 * W],
                [0.0, 0.42278125 * H, 0.5 * H],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    # =========================
    # sample
    # =========================
    # re-implement in basedataset

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, frame_name, fid, extrinsic, intrinsic, resolution, rng):
        impath = osp.join(scene_path, "images", f"{frame_name}.jpg")
        depthpath = osp.join(scene_path, "depth", f"{frame_name}.png")

        rgb_image = read_any_file(impath)
        depthmap = read_any_file(depthpath)
        depthmap = depthmap * 0.001

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=impath,
        )
        
        depthmap = threshold_depth_map(depthmap, max_depth=80)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=extrinsic.astype(np.float32),
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=osp.basename(scene_path),
            instance=str(fid),
        )


    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        scene_path = self.scene_to_path[scene]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frames_id"], rng)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        pose_path = osp.join(scene_path, "scene_metadata.npz")
        poses = read_any_file(pose_path)

        poses_trajectories = poses["trajectories"]
        poses_intrinsics = poses["intrinsics"]

        frames = meta["frames_name"]
        frames = np.array(frames)
        idxs = np.array(idxs)
        full_images = poses["images"]
        select_images = frames[idxs]
        sorter = np.argsort(full_images)
        # searchsorted 带 sorter 时返回的是排序后数组里的位置，要经 sorter 映射回原始索引；
        # images 本来有序时两者相同（当前数据就是这样），无序时旧写法会让位姿和内参错配。
        pose_idxs = sorter[np.searchsorted(full_images, select_images, sorter=sorter)]

        views = []
        for fid, pid in zip(idxs, pose_idxs):
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    frame_name=frames[fid],
                    fid=fid,
                    extrinsic=poses_trajectories[pid],
                    intrinsic=poses_intrinsics[pid],
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.scannetpp_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(ScannetppDataset))
