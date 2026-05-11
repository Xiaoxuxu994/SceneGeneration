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
import json

FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .base.base_dataset import BaseDataset, CACHE_DIR
from .base.transforms import *
from .sample_utils.common import rdepth_to_zdepth, threshold_depth_map
from .base.utils import read_any_file

CATEGORIES = [
    "bottle",
    "cup",
    "tooth_brush",
    "book",
    "box",
    "knife",
    "remote_control",
    "razor",
    "keyboard",
    "bowl",
    "scissor",
    "kettle",
    "bucket",
    "pliers",
    "ball",
    "mouse",
    "handbag",
    "cellphone",
    "microwave",
    "hat",
    "clock",
    "shoe",
    "flower_pot",
    "detergent",
    "backpack",
    "chair",
    "TV",
    "pineapple",
    "potato",
    "cucumber",
    "apple",
    "banana",
    "pear",
    "tomato",
    "peach",
    "orange",
    "carrot",
    "donut",
    "cake",
    "stuffed_toy",
    "train",
    "truck",
    "boat",
    "bus",
    "plane",
    "car",
]

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


class WildrgbdDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        debug=False,
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "Wildrgbd"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.debug = debug

        self.WILDRGBD_DIR = data_root
        self.trans_device_from_camera = get_trans_device_from_camera()

        #
        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # 1) load/save cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(CACHE_DIR, f"{self.dataset_label}_{root_hash}_cache.pkl")
        if not os.path.exists(cache_path):
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
        from tqdm import tqdm

        category = sorted(CATEGORIES)
        if self.debug:
            category = ["backpack"]

        self.invalid_sequence = []
        self.seqlen = None

        sequence_list_path = f"{self.WILDRGBD_DIR}/{self.mode}_sequence_list.pkl"

        if osp.exists(sequence_list_path):
            print(f"[WildRGBD] Loading sequence list from {sequence_list_path}")
            with open(sequence_list_path, "rb") as f:
                sequence_list, scene_len = pickle.load(f)

        else:
            print(f"[WildRGBD] Scanning scenes under {self.WILDRGBD_DIR}")

            scene_len = {}
            sequence_list = []

            category_list = set(os.listdir(self.WILDRGBD_DIR))

            all_scenes = []
            for c in tqdm(category, desc="[WildRGBD] Collecting categories"):
                if c not in category_list:
                    continue

                scene_dir = osp.join(self.WILDRGBD_DIR, c, "scenes")
                if not osp.isdir(scene_dir):
                    continue

                for each in os.listdir(scene_dir):
                    abs_path = osp.join(scene_dir, each)
                    if osp.isdir(abs_path):
                        all_scenes.append((c, each, abs_path))

            print(f"[WildRGBD] Found {len(all_scenes)} candidate scenes")

            for c, each, abs_path in tqdm(all_scenes, desc="[WildRGBD] Validating scenes"):
                rgb_dir = osp.join(abs_path, "rgb")
                depth_dir = osp.join(abs_path, "depth")
                masks_dir = osp.join(abs_path, "masks")
                cam_pose_path = osp.join(abs_path, "cam_poses.txt")

                if not (
                    osp.isdir(rgb_dir)
                    and osp.isdir(depth_dir)
                    and osp.isdir(masks_dir)
                    and osp.isfile(cam_pose_path)
                ):
                    self.invalid_sequence.append(abs_path)
                    continue

                try:
                    num_rgb = len(os.listdir(rgb_dir))
                    num_depth = len(os.listdir(depth_dir))
                    num_masks = len(os.listdir(masks_dir))

                    with open(cam_pose_path, encoding="utf-8") as cam_pose_txt:
                        num_cam = sum(1 for _ in cam_pose_txt)

                except Exception:
                    self.invalid_sequence.append(abs_path)
                    continue

                if not (num_rgb == num_depth == num_masks == num_cam):
                    self.invalid_sequence.append(abs_path)
                    continue

                scene_name = abs_path.removeprefix(self.WILDRGBD_DIR).lstrip("/")
                sequence_list.append(scene_name)
                scene_len[scene_name] = num_cam

            print(
                f"[WildRGBD] Valid scenes: {len(sequence_list)} | "
                f"Invalid scenes: {len(self.invalid_sequence)}"
            )

            with open(sequence_list_path, "wb") as f:
                pickle.dump((sequence_list, scene_len), f)

            print(f"[WildRGBD] Saved sequence list to {sequence_list_path}")

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for scene_name in tqdm(sequence_list, desc="[WildRGBD] Building metadata"):
            all_sequences.append(scene_name)
            scene_to_path[scene_name] = osp.join(self.WILDRGBD_DIR, scene_name)
            scene_meta[scene_name] = {
                "frames_id": list(range(scene_len[scene_name])),
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
    def _load_one_view(self, scene_path, fid, extrinsic, intrinsic, resolution, rng):
        file = f"{fid:05d}.png"
        image_path = osp.join(scene_path, "rgb", file)
        depth_map_path = osp.join(scene_path, "depth", file)
        mvs_mask_path = osp.join(scene_path, "masks", file)

        rgb_image = read_any_file(image_path)
        depthmap = read_any_file(depth_map_path)
        mvs_mask = read_any_file(mvs_mask_path) > 128

        # cull the inf values in the original depth array
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap = depthmap / 1000

        depthmap[~mvs_mask] = 0

        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        assert rgb_image.shape[:2] == depthmap.shape, f"Image and depth shape mismatch: {rgb_image.shape[:2]} vs {depthmap.shape}"

        extrinsic = np.concatenate([extrinsic, np.array([[0,0,0,1]], dtype=extrinsic.dtype)], axis=0)
        extrinsic = np.linalg.inv(extrinsic)

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=image_path,
        )

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

        camera_path = osp.join(scene_path, "cam_poses.txt")
        extrinsics_list = []
        with open(camera_path) as file:
            for line in file:
                line = line.strip()
                elements = line.split()
                matrix_values = elements[1:13]

                c2w_matrix = np.array([float(x) for x in matrix_values]).reshape(3, 4)
                R_c2w = c2w_matrix[:, :3]
                t_c2w = c2w_matrix[:, 3]
                R_w2c = R_c2w.T
                t_w2c = -R_w2c @ t_c2w

                w2c_matrix = np.column_stack([R_w2c, t_w2c])
                extrinsics_list.append(w2c_matrix)

        intrinsics_path = osp.join(scene_path, "metadata")

        intri_opencv_original = np.zeros((3, 3))
        with open(intrinsics_path) as f:
            data = json.load(f)
            intrinsics_value = data["K"]
            i = 0
            for col in range(3):
                for row in range(3):
                    intri_opencv_original[row][col] = intrinsics_value[i]
                    i += 1

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    fid=fid,
                    extrinsic=extrinsics_list[fid],
                    intrinsic=intri_opencv_original.copy(),
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views

if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt
    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/Wildrgbd"
    out_dir = "output_debug/Wildrgbd/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = WildrgbdDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=24,
        max_distance=8,
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

    """Visualize gt data. """
    visualize_gt(views, out_dir, prefix="Wildrgbd_GT")