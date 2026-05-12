import sys
sys.path.append('.')

import os
import os.path as osp
import hashlib
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class Matterport3DDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,               
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        '''
            https://github.com/niessner/Matterport/blob/master/data_organization.md
            camera is c2w, OpenGL
            image & depth keep no flip, origin is bottom left, flip intrinsic only
        '''
        assert data_root is not None

        self.dataset_label = "Matterport3D"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        cache_dir = "/efs-mi-east4-2/z00533370/proj_3dgs/3drecon/data/dataset_cache"
        os.makedirs(cache_dir, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")

        if not os.path.exists(cache_path):
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
        scene_names = sorted([
            x for x in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, x))
        ])

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for scene in tqdm(scene_names, desc="Building Matterport3D cache"):
            outer_scene_path = osp.join(self.data_root, scene)
            inner_scene_path = osp.join(outer_scene_path, scene)
            scene_path = inner_scene_path if osp.isdir(inner_scene_path) else outer_scene_path

            cam_param_dir = osp.join(scene_path, "undistorted_camera_parameters")
            color_dir = osp.join(scene_path, "undistorted_color_images")
            depth_dir = osp.join(scene_path, "undistorted_depth_images")

            if not (osp.isdir(cam_param_dir) and osp.isdir(color_dir) and osp.isdir(depth_dir)):
                continue

            conf_files = sorted(glob(osp.join(cam_param_dir, "*.conf")))
            if len(conf_files) == 0:
                continue

            conf_path = conf_files[0]

            try:
                frame_ids, camera_info, file_info = self._parse_conf(scene_path, conf_path)
            except Exception as e:
                print(f"[Matterport3D] Skip scene {scene}, failed to parse {conf_path}: {e}")
                continue

            if len(frame_ids) < self.min_num_images:
                continue

            all_sequences.append(scene)
            scene_to_path[scene] = scene_path
            scene_meta[scene] = {
                "frame_ids": frame_ids,
                "camera_info": camera_info,
                "file_info": file_info,
            }

        self.all_sequences = sorted(all_sequences)
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

    def _parse_conf(self, scene_path, conf_path):
        frame_ids = []
        camera_info = {}
        file_info = {}

        color_dirname = None
        depth_dirname = None
        current_K = None

        with open(conf_path, "r") as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]

        for line in lines:
            parts = line.split()
            key = parts[0]

            if key == "color_directory":
                color_dirname = parts[1]

            elif key == "depth_directory":
                depth_dirname = parts[1]

            elif key == "intrinsics_matrix":
                vals = list(map(float, parts[1:10]))
                current_K = np.array(vals, dtype=np.float32).reshape(3, 3)

            elif key == "scan":
                assert color_dirname is not None
                assert depth_dirname is not None
                assert current_K is not None

                depth_name = parts[1]
                color_name = parts[2]
                pose_vals = list(map(float, parts[3:19]))
                pose = np.array(pose_vals, dtype=np.float32).reshape(4, 4)

                # Official docs:
                # c2w matrix that takes canonical camera coords to world coords.
                # canonical camera: x right, y up, view direction (0,0,-1)
                # convert camera local frame to OpenCV style: x right, y down, z forward
                pose[:, 1:3] *= -1

                fid = osp.splitext(color_name)[0]

                img_path = osp.join(scene_path, color_dirname, color_name)
                depth_path = osp.join(scene_path, depth_dirname, depth_name)

                if not (osp.isfile(img_path) and osp.isfile(depth_path)):
                    continue

                frame_ids.append(fid)
                camera_info[fid] = {
                    "camera_pose": pose.astype(np.float32),         # c2w
                    "camera_intrinsics": current_K.astype(np.float32),
                }
                file_info[fid] = {
                    "img_path": img_path,
                    "depth_path": depth_path,
                }

        return frame_ids, camera_info, file_info

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

        # if self.frame_num > 16 and rng.random() < self.random_sample_thres:
        #     pos = rng.choice(n, size=self.frame_num, replace=(n < self.frame_num))
        #     return [frame_ids[int(i)] for i in pos]

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

        return Image.open(path).convert("RGB")

    def _read_depth(self, path):
        # Official docs:
        # 16-bit PNG, z-direction depth, 0.25 mm per unit -> divide by 4000
        depth = np.array(Image.open(path)).astype(np.float32) / 4000.0
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth <= 0)] = 0.0
        return depth.astype(np.float32)

    def _adjust_intrinsics_for_flip(self, K, H):
        K = K.copy()
        # flip_vertical:
        K[1, 2] = (H - 1) - K[1, 2]
        return K

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_name, fid, cam_info, file_info, resolution, rng):
        img_path = file_info["img_path"]
        depth_path = file_info["depth_path"]

        rgb_image = self._read_rgb(img_path)
        depthmap = self._read_depth(depth_path)

        W0, H0 = rgb_image.size
        intrinsics = self._adjust_intrinsics_for_flip(
            cam_info["camera_intrinsics"].copy(),
            H0,
        )

        # Matterport undistorted depth is already z-depth in meters
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
        depthmap = threshold_depth_map(depthmap, max_percentile=98)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=cam_info["camera_pose"].astype(np.float32),
            camera_intrinsics=intrinsics_.astype(np.float32),
            dataset=self.dataset_label,
            label=scene_name,
            instance=fid,
        )

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        if self.mode == "train":
            index = index % len(self.sequences)

        scene = self.sequences[index]
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
                    scene_name=scene,
                    fid=fid,
                    cam_info=meta["camera_info"][fid],
                    file_info=meta["file_info"][fid],
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/Matterport3D_decode"
    out_dir = "output_debug/Matterport3D/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = Matterport3DDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=128,
        max_distance=8,
        min_num_images=8,
        split_ratio=1.0,
        verbose=False,
        shuffle=False,
    )

    print(f"len(dataset) = {len(dataset)}")

    views = dataset[0]
    print(f"[dataset] number of views = {len(views)}")
    for i, v in enumerate(views):
        print(
            f"[dataset view {i}] "
            f"img={tuple(v['img'].shape) if torch.is_tensor(v['img']) else v['img'].size}, "
            f"depth={v['depthmap'].shape}, "
            f"depthmin={v['depthmap'].min()}, "
            f"depthmax={v['depthmap'].max()}, "
            f"pts3d={v['pts3d'].shape}, "
            f"valid={v['valid_mask'].sum()}"
        )

    visualize_gt(views, out_dir, prefix="Matterport3D_GT")