import sys
sys.path.append('.')

import os
import os.path as osp
import json
import re
import math
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
# from pytorch3d.renderer import FoVPerspectiveCameras
# from pytorch3d.transforms import euler_angles_to_matrix

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


EULER_X_OFFSET_RADS = math.radians(90.0)

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c],
    ], dtype=np.float32)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float32)


def euler_xzy_to_matrix(x, z, y):
    """
    Replacement of pytorch3d.transforms.euler_angles_to_matrix(..., "XZY")
    using row-major matrices.

    Equivalent to:
        R = Rx(x) @ Rz(z) @ Ry(y)
    """
    return rot_x(x) @ rot_z(z) @ rot_y(y)


def fov_to_intrinsics_from_p3d_style(fov_rads, H, W):
    """
    Approximate FoVPerspectiveCameras(..., aspect_ratio=W/H).compute_projection_matrix(...)
    converted to OpenCV intrinsics in the same style as your current code.

    Assuming `fov_rads` is vertical FOV:
        fy = H / (2 * tan(fov/2))
        fx = fy
        cx = W / 2
        cy = H / 2
    """
    fy = H / (2.0 * np.tan(fov_rads / 2.0))
    fx = fy
    cx = W / 2.0
    cy = H / 2.0

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1 ],
    ], dtype=np.float32)
    return K

def get_sort_key(filename: str):
    """
    Sort by (point_id, view_id), e.g.
    point_10_view_1_domain_depth_zbuffer.png -> (10, 1)
    """
    match = re.search(r"point_(\d+)_view_(\d+)", filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (float("inf"), float("inf"))


def parse_point_view(filename: str):
    """
    Parse point_id and view_id from filename.
    """
    match = re.search(r"point_(\d+)_view_(\d+)", filename)
    if match:
        return match.group(1), match.group(2)
    return None

def habitat_meta_to_pose_and_intrinsics(point_info, H, W):
    """
    Convert Habitat metadata to:
      - camera_pose (c2w, 4x4)
      - camera_intrinsics (opencv K, 3x3)

    Pure numpy replacement of the original pytorch3d-based implementation.
    """
    location = point_info["camera_location"]
    rotation = point_info["camera_rotation_final"]
    fov = point_info["field_of_view_rads"]

    ex, ey, ez = rotation

    # Match original logic:
    # euler_angles_to_matrix([(ex - offset, -ey, -ez)], "XZY")
    R = euler_xzy_to_matrix(
        ex - EULER_X_OFFSET_RADS,
        -ey,
        -ez,
    )

    Tx, Ty, Tz = location
    T = np.array([-Tx, Tz, Ty], dtype=np.float32)  # [3]

    # Keep the same variable naming/semantics as original code
    # Original:
    # R_inv = R.transpose(1, 2)
    # T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
    R_inv = R.T
    T_inv = -(R @ T.reshape(3, 1)).reshape(3)

    # Replacement of FoVPerspectiveCameras(...).compute_projection_matrix(...)
    camera_intrinsics = fov_to_intrinsics_from_p3d_style(fov, H, W)

    # Keep the same OpenCV world->cam conversion logic
    convert_mat = np.array(
        [[-1, 0, 0],
         [ 0,-1, 0],
         [ 0, 0, 1]],
        dtype=np.float32,
    )

    # Original code:
    # R_w2c = convert_mat @ R_inv.squeeze(0).cpu().numpy().T
    # T_w2c = convert_mat @ T_inv.squeeze(0).cpu().numpy()
    R_w2c = convert_mat @ R_inv.T
    T_w2c = convert_mat @ T_inv

    # Convert world->cam extrinsic to c2w pose
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = R_w2c.T.astype(np.float32)
    camera_pose[:3, 3] = (-R_w2c.T @ T_w2c).astype(np.float32)

    return camera_pose, camera_intrinsics


class HabitatDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        z_far=127.99,
        pass_z_far=False,   # True：把配置里的 z_far 传给 BaseDataset 生效（旧行为里它被这里截下、实际不起作用）
        **kwargs,
    ):
        super().__init__(**(dict(kwargs, z_far=z_far) if pass_z_far else kwargs))
        assert data_root is not None

        self.dataset_label = "Habitat"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images

        # expected layout:
        # data_root/
        #   rgb/habitat/<scene_name>/point_x_view_y_domain_rgb.png
        #   depth_zbuffer/habitat/<scene_name>/point_x_view_y_domain_depth_zbuffer.png
        #   point_info/habitat/<scene_name>/point_x_view_y_domain_fixatedpose.json
        self.rgb_root = osp.join(self.data_root, "rgb", "habitat")
        self.depth_root = osp.join(self.data_root, "depth_zbuffer", "habitat")
        self.point_info_root = osp.join(self.data_root, "point_info", "habitat")

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # 1) load/save cache
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
        depth_root = Path(self.depth_root)
        rgb_root = Path(self.rgb_root)
        point_info_root = Path(self.point_info_root)

        if not depth_root.exists():
            raise FileNotFoundError(f"Depth root not found: {depth_root}")
        if not rgb_root.exists():
            raise FileNotFoundError(f"RGB root not found: {rgb_root}")
        if not point_info_root.exists():
            raise FileNotFoundError(f"Point-info root not found: {point_info_root}")

        scene_names = sorted([p.name for p in depth_root.iterdir() if p.is_dir()])

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for scene_name in tqdm(scene_names, desc="Building Habitat cache"):
            depth_scene_dir = depth_root / scene_name
            rgb_scene_dir = rgb_root / scene_name
            point_scene_dir = point_info_root / scene_name

            if not (rgb_scene_dir.is_dir() and point_scene_dir.is_dir()):
                continue

            grouped_views = defaultdict(list)

            depth_files = sorted(
                depth_scene_dir.glob("*_depth_zbuffer.png"),
                key=lambda x: get_sort_key(x.name),
            )

            for depth_file in depth_files:
                base = depth_file.name.replace("_depth_zbuffer.png", "")
                rgb_file = rgb_scene_dir / f"{base}_rgb.png"
                point_file = point_scene_dir / f"{base}_fixatedpose.json"

                if not (rgb_file.exists() and point_file.exists()):
                    continue

                parsed = parse_point_view(depth_file.name)
                if parsed is None:
                    continue

                point_id, view_id = parsed
                seq_key = f"{scene_name}/{point_id}"
                grouped_views[seq_key].append(view_id)

            for seq_key, view_ids in grouped_views.items():
                # sort views numerically
                view_ids = sorted(view_ids, key=lambda x: int(x) if str(x).isdigit() else x)

                if len(view_ids) < self.min_num_images:
                    continue

                scene_name_, point_id_ = seq_key.split("/")
                all_sequences.append(seq_key)
                scene_to_path[seq_key] = scene_name_
                scene_meta[seq_key] = {
                    "scene_name": scene_name_,
                    "point_id": point_id_,
                    "view_ids": view_ids,
                }

        def seq_sort_key(x):
            scene_name, point_id = x.split("/")
            point_ord = int(point_id) if point_id.isdigit() else point_id
            return (scene_name, point_ord)

        self.all_sequences = sorted(all_sequences, key=seq_sort_key)
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
    def _load_one_view(self, seq_key, view_id, resolution, rng):
        meta = self.scene_meta[seq_key]
        scene_name = meta["scene_name"]
        point_id = meta["point_id"]

        base = f"point_{point_id}_view_{view_id}_domain"

        impath = osp.join(self.rgb_root, scene_name, f"{base}_rgb.png")
        depthpath = osp.join(self.depth_root, scene_name, f"{base}_depth_zbuffer.png")
        pointpath = osp.join(self.point_info_root, scene_name, f"{base}_fixatedpose.json")

        if not osp.isfile(impath):
            raise FileNotFoundError(impath)
        if not osp.isfile(depthpath):
            raise FileNotFoundError(depthpath)
        if not osp.isfile(pointpath):
            raise FileNotFoundError(pointpath)

        rgb_image = Image.open(impath).convert("RGB")
        depthmap = np.array(Image.open(depthpath)).astype(np.float32) / 512.0

        # Habitat original logic: threshold max depth 127.99
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap[(depthmap <= 0.0) | (depthmap > 127.99)] = 0.0

        with open(pointpath, "r") as f:
            point_info = json.load(f)

        W0, H0 = rgb_image.size
        if depthmap.shape != (H0, W0):
            raise ValueError(
                f"Shape mismatch: image {(H0, W0)} vs depth {depthmap.shape}, file={impath}"
            )

        camera_pose, camera_intrinsics = habitat_meta_to_pose_and_intrinsics(
            point_info=point_info,
            H=H0,
            W=W0,
        )

        rgb_image, depthmap, camera_intrinsics = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            camera_intrinsics.copy(),
            resolution,
            rng=rng,
            info=impath,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),         # c2w
            camera_intrinsics=camera_intrinsics.astype(np.float32),
            dataset=self.dataset_label,
            label=seq_key,
            instance=str(view_id),
        )

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        seq_key = self.sequences[index]
        meta = self.scene_meta[seq_key]

        idxs = self._sample_frame_ids(meta["view_ids"], rng)

        self.this_views_info = {
            "scene": seq_key,
            "idxs": idxs,
        }

        views = []
        for view_id in idxs:
            views.append(
                self._load_one_view(
                    seq_key=seq_key,
                    view_id=view_id,
                    resolution=resolution,
                    rng=rng,
                )
            )
        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/habitat"
    out_dir = "output_debug/Habitat/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = HabitatDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=24,
        split_ratio=1.0,
        min_num_images=24,
        z_far=127.99,
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

    visualize_gt(views, out_dir, prefix="Habitat_GT")