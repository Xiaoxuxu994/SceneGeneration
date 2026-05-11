import sys
sys.path.append('.')

import os
import os.path as osp
import hashlib
from glob import glob

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .base.base_dataset import BaseDataset, CACHE_DIR
from .base.transforms import *


def xyzqxqyqxqw_to_c2w(xyzqxqyqxqw):
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)

    # NOTE: convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw,     2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw,     1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw,     2 * qy * qz + 2 * qx * qw,     1 - 2 * qx * qx - 2 * qy * qy]
    ], dtype=np.float32)
    c2w[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return c2w


class TarTanAirDataset(BaseDataset):
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
        assert data_root is not None

        self.dataset_label = "TarTanAir"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images

        # TartanAir fixed intrinsics
        fx = 320.0
        fy = 320.0
        cx = 320.0
        cy = 240.0
        self.intrinsics = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=np.float32,
        )

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        os.makedirs(CACHE_DIR, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(
            CACHE_DIR,
            f"{self.dataset_label}_{root_hash}_cache.pkl"
        )

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
        assert osp.isdir(self.data_root), f"[{self.dataset_label}] data_root not found: {self.data_root}"

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        top_scene_names = sorted([
            x for x in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, x))
        ])

        for top_scene in tqdm(top_scene_names, desc=f"Building {self.dataset_label} cache"):
            top_scene_root = osp.join(self.data_root, top_scene)

            for difficulty in ["Easy", "Hard"]:
                difficulty_root = osp.join(top_scene_root, difficulty)
                if not osp.isdir(difficulty_root):
                    continue

                sub_scene_names = sorted([
                    x for x in os.listdir(difficulty_root)
                    if osp.isdir(osp.join(difficulty_root, x))
                ])

                for sub_scene in sub_scene_names:
                    scene = (top_scene, difficulty, sub_scene)
                    scene_path = osp.join(difficulty_root, sub_scene)

                    rgb_dir = osp.join(scene_path, "image_left")
                    depth_dir = osp.join(scene_path, "depth_left")
                    cam_file = osp.join(scene_path, "pose_left.txt")

                    if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir) and osp.isfile(cam_file)):
                        if self.verbose:
                            print(f"[{self.dataset_label}] Skip invalid scene: {scene_path}")
                        continue

                    try:
                        caminfo = np.loadtxt(cam_file).astype(np.float32)
                        if caminfo.ndim == 1:
                            caminfo = caminfo[None, :]
                    except Exception as e:
                        print(f"[{self.dataset_label}] Skip scene {scene}, failed to load pose file: {e}")
                        continue

                    rgb_ids = {
                        osp.basename(x).replace("_left.png", "")
                        for x in glob(osp.join(rgb_dir, "*_left.png"))
                    }
                    depth_ids = {
                        osp.basename(x).replace("_left_depth.npy", "")
                        for x in glob(osp.join(depth_dir, "*_left_depth.npy"))
                    }

                    inter_ids = rgb_ids & depth_ids
                    numeric_ids = [x for x in inter_ids if str(x).isdigit()]
                    frame_ids = sorted(numeric_ids, key=lambda x: int(x))

                    num_cam_frames = len(caminfo)
                    frame_ids = [
                        fid for fid in frame_ids
                        if int(fid) < num_cam_frames
                    ]

                    if len(frame_ids) < self.min_num_images:
                        if self.verbose:
                            print(
                                f"[{self.dataset_label}] Skip short scene {scene}, "
                                f"valid_frames={len(frame_ids)} < min_num_images={self.min_num_images}"
                            )
                        continue

                    all_sequences.append(scene)
                    scene_to_path[scene] = scene_path
                    scene_meta[scene] = {
                        "frame_ids": frame_ids,
                        "camera_data": caminfo,
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
    # load one view
    # =========================
    def _load_one_view(self, scene_path, scene_name, fid, camera_data, resolution, rng):
        idx = int(fid)

        impath = osp.join(scene_path, "image_left", f"{idx:06d}_left.png")
        depthpath = osp.join(scene_path, "depth_left", f"{idx:06d}_left_depth.npy")

        rgb_image = np.array(Image.open(impath).convert("RGB"))
        depthmap = np.load(depthpath).astype(np.float32)

        depthmap[depthmap > 80] = -1
        depthmap = np.nan_to_num(depthmap, nan=-1.0, posinf=-1.0, neginf=-1.0).astype(np.float32)

        camera_pose = xyzqxqyqxqw_to_c2w(camera_data[idx]).astype(np.float32)

        rgb_image, depthmap, intrinsics_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            self.intrinsics.copy(),
            resolution,
            rng=rng,
            info=impath,
        )

        return dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=intrinsics_.astype(np.float32),
            dataset=self.dataset_label,
            label=f"{scene_name[0]}_{scene_name[1]}_{scene_name[2]}",
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

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    scene_name=scene,
                    fid=fid,
                    camera_data=meta["camera_data"],
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt
    data_root = "/efs-m3ga/c50041401/datasets/reliev3r_synth_data/TartanAir/"
    out_dir = "output_debug/TarTanAir/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = TarTanAirDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=24,
        min_num_images=24,
        split_ratio=0.9,
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

    visualize_gt(views, out_dir, prefix="Tartanair_GT")