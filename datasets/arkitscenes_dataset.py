import sys
sys.path.append('.')

import os
import os.path as osp

import torch
import numpy as np
import hashlib
import math
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map
from datasets.base.utils import read_any_file

from scipy.spatial.transform import Rotation as R
import cv2

class ARKitScenesDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        tol=0.0005,
        conf_min=-1,        # >= 0 时读 ARKit 自带的 confidence（0/1/2），只保留 confidence >= conf_min 的像素（作为 extra_valid）
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "ARKitScenes"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.conf_min = conf_min
        self._conf_missing_warned = False
        self.tol = tol
        self.ARKITSCENES_DIR = data_root

        #
        self.all_sequences = []
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
        all_sequences = []
        scene_to_path = {}
        scene_meta = {}
        for subdir in fio.listdir(self.ARKITSCENES_DIR):
            subdir_path = osp.join(self.ARKITSCENES_DIR, subdir)
            if not fio.isdir(subdir_path):
                continue

            for scene_id in tqdm(
                sorted(fio.listdir(subdir_path)),
                desc=f"{subdir}",
                leave=False
            ):
                intri_path = osp.join(self.ARKITSCENES_DIR, subdir, scene_id, "vga_wide_intrinsics")
                rgb_path = osp.join(self.ARKITSCENES_DIR, subdir, scene_id, "vga_wide")
                depth_path = osp.join(self.ARKITSCENES_DIR, subdir, scene_id, "lowres_depth")
                pose_path = osp.join(self.ARKITSCENES_DIR, subdir, scene_id, "lowres_wide.traj") 
                if fio.exists(intri_path) and fio.exists(rgb_path) and fio.exists(depth_path) and fio.exists(pose_path):
                    # get all timestamp of intrisic/rgb/depth/traj aand sorted
                    intri_timestamp_list = sorted([s.split("_")[1].replace(".pincam", "") for s in fio.listdir(intri_path)], key=lambda s: float(s))
                    rgb_timestamp_list = sorted([s.split("_")[1].replace(".png", "") for s in fio.listdir(rgb_path)], key=lambda s: float(s))
                    depth_timestamp_list = sorted([s.split("_")[1].replace(".png", "") for s in fio.listdir(depth_path)], key=lambda s: float(s))   
                    traj_timestamp_list = []
                    with fio.open(pose_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()  # 6790.13168783 1.212956130924529 -1.2455780157270364 1.2588744491484622 0.0613927 0.0737052 0.00398328
                            if not parts:
                                continue
                            traj_timestamp_list.append(parts[0])
                    traj_timestamp_list = sorted(traj_timestamp_list, key=lambda s: float(s))
                    # get closest timestamp in intrisic/rgb/depth of timestamp in traj
                    closest_in_intri = closest_in_b_for_each_a(traj_timestamp_list, intri_timestamp_list)
                    closest_in_rgb = closest_in_b_for_each_a(traj_timestamp_list, rgb_timestamp_list)
                    closest_in_depth = closest_in_b_for_each_a(traj_timestamp_list, depth_timestamp_list)
                    # get final effective timestamp frames
                    timestamp_list = []
                    for res_intri, res_rgb, res_depth in zip(closest_in_intri, closest_in_rgb, closest_in_depth):
                        if res_intri[2] < self.tol and res_rgb[2] < self.tol and res_depth[2] < self.tol:
                            timestamp_list.append((res_intri[0], res_intri[1], res_rgb[1], res_depth[1])) 
                    
                    # print(subdir, scene_id, len(timestamp_list))
                    if len(timestamp_list) > self.min_num_images:
                        key = f"{subdir}/{scene_id}"
                        all_sequences.append(key)
                        scene_to_path[key] = key
                        scene_meta[key] = {
                            "frames_id": list(range(len(timestamp_list))),
                            "frames_timestamp": timestamp_list,
                        }
        # print(all_sequences)
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
    # sample
    # =========================
    # re-implement in basedataset

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, scene_id, timestamp_group, pose, fid, resolution, rng):
        intri_path = osp.join(self.ARKITSCENES_DIR, scene_path, "vga_wide_intrinsics", f"{scene_id}_{timestamp_group[1]}.pincam")
        image_path = osp.join(self.ARKITSCENES_DIR, scene_path, "vga_wide", f"{scene_id}_{timestamp_group[2]}.png")
        depth_path = osp.join(self.ARKITSCENES_DIR, scene_path, "lowres_depth", f"{scene_id}_{timestamp_group[3]}.png")
        rgb_image = read_any_file(image_path)
        h,w = rgb_image.shape[:2]
        depthmap = np.array(fio.pil_open(depth_path).resize((w, h), resample=Image.NEAREST))/1000
        with fio.open(intri_path, "r") as f:
            line = f.read().strip()
            w, h, fx, fy, cx, cy = map(float, line.split())
        intrinsic = np.eye(3,dtype=np.float32)
        intrinsic[0,0] = fx
        intrinsic[1,1] = fy
        intrinsic[0,2] = cx
        intrinsic[1,2] = cy

        t = np.array(pose[3:])
        rvec = np.array(pose[:3])
        R, _ = cv2.Rodrigues(rvec)
        T_cw = np.eye(4)
        T_cw[:3,:3] = R
        T_cw[:3,3] = t
        extrinsic = np.linalg.inv(T_cw)

        conf = None
        if self.conf_min >= 0:
            conf_path = osp.join(self.ARKITSCENES_DIR, scene_path, "confidence", f"{scene_id}_{timestamp_group[3]}.png")
            if fio.exists(conf_path):
                conf = np.array(fio.pil_open(conf_path).resize((rgb_image.shape[1], rgb_image.shape[0]), resample=Image.NEAREST)).astype(np.uint8)
            elif not self._conf_missing_warned:
                print(f"[{self.dataset_label}] confidence not found (e.g. {conf_path}); conf_min ignored for such frames", flush=True)
                self._conf_missing_warned = True

        out = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=image_path,
            far_mask=conf,
        )
        rgb_image, depthmap, intrinsic_ = out[:3]

        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        view = dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=extrinsic.astype(np.float32),
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=scene_path,
            instance=str(fid),
        )
        if conf is not None:
            view['extra_valid'] = out[3] >= self.conf_min
        return view

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        scene_path = self.scene_to_path[scene]
        scene_id = scene_path.split('/')[1]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frames_id"], rng)
        
        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        pose_dict = {}
        pose_path = osp.join(self.ARKITSCENES_DIR, scene, "lowres_wide.traj") 
        with fio.open(pose_path, "r") as f:
            for line in f:
                parts = line.strip().split()  # 6790.13168783 1.212956130924529 -1.2455780157270364 1.2588744491484622 0.0613927 0.0737052 0.00398328
                if not parts:
                    continue
                timestamp = parts[0]
                pose_dict[timestamp] = [float(x) for x in parts[1:]] # tx,ty,tz,rx,ry,rz

        views = []
        for fid in idxs:
            timestamp_group = meta['frames_timestamp'][fid]
            # print(timestamp_group)
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    scene_id=scene_id,
                    timestamp_group=timestamp_group,
                    pose=pose_dict[timestamp_group[0]],
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views

# find the closest element in list b of each element of list a
def closest_in_b_for_each_a(a:list[str], b:list[str]):
    res = []
    j = 0
    for x in a:
        while j+1 < len(b) and abs(float(b[j+1]) - float(x)) <= abs(float(b[j]) - float(x)):
            j += 1
        res.append((x, b[j], abs(float(b[j]) - float(x))))
    return res


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.arkitscenes_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(ARKitScenesDataset))
