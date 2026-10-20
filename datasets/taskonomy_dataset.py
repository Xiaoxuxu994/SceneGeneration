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
from collections import defaultdict
import pickle
from pathlib import Path
import re
import json
import math

FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import rdepth_to_zdepth, threshold_depth_map
from datasets.base.utils import read_any_file

EULER_X_OFFSET_RADS = math.radians(90.0)

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

def compute_projection_matrix(
        znear, zfar, fov, aspect_ratio, degrees: bool
    ):
        """
        Compute the calibration matrix K of shape (N, 4, 4)

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            fov: field of view angle of the camera.
            aspect_ratio: aspect ratio of the image pixels.
                1.0 indicates square pixels.
            degrees: bool, set to True if fov is specified in degrees.

        Returns:
            numpy.ndarray of the calibration matrix with shape (4, 4)
        """
        K = np.zeros((4, 4), dtype=np.float32)
        if degrees:
            fov = (np.pi / 180) * fov

        tanHalfFov = np.tan((fov / 2))
        max_y = tanHalfFov * znear
        min_y = -max_y
        max_x = max_y * aspect_ratio
        min_x = -max_x

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space positive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0

        K[0, 0] = 2.0 * znear / (max_x - min_x)
        K[1, 1] = 2.0 * znear / (max_y - min_y)
        K[0, 2] = (max_x + min_x) / (max_x - min_x)
        K[1, 2] = (max_y + min_y) / (max_y - min_y)
        K[3, 2] = z_sign

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane.
        K[2, 2] = z_sign * zfar / (zfar - znear)
        K[2, 3] = -(zfar * znear) / (zfar - znear)

        return K

def _axis_angle_rotation(axis: str, angle) -> np.ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (3, 3).
    """

    cos = math.cos(angle)
    sin = math.sin(angle)
    one = 1
    zero = 0

    if axis == "X":
        R_flat = np.array([one, zero, zero, zero, cos, -sin, zero, sin, cos]).reshape(3, 3) 
    elif axis == "Y":
        R_flat = np.array([cos, zero, sin, zero, one, zero, -sin, zero, cos]).reshape(3, 3) 
    elif axis == "Z":
        R_flat = np.array([cos, -sin, zero, sin, cos, zero, zero, zero, one]).reshape(3, 3) 
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return R_flat

def euler_angles_to_matrix(euler_angles: np.ndarray, convention: str) -> np.ndarray:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (3, 3).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    
    n = len(convention)
    matrices = [
        _axis_angle_rotation(convention[i], euler_angles[i])
        for i in range(n)
    ]
    return matrices[0] @ matrices[1] @ matrices[2]


class TaskonomyDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        sample_method=None,
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "Taskonomy"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.sample_method = sample_method

        self.TASKONOMY_DIR = data_root
        self.trans_device_from_camera = get_trans_device_from_camera()

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
            self._save_cache(keys=["all_sequences", "scene_meta"],save_path=cache_path)
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
        sequence_list_path = f"{self.TASKONOMY_DIR}/sequence_list.pkl"
        if fio.exists(sequence_list_path):
            with fio.open(sequence_list_path, "rb") as f:
                sequence_list = pickle.load(f)
        else:
            sequence_list = defaultdict(list)

            base_dir = Path(self.TASKONOMY_DIR)
            depth_base_path = base_dir / "depth_zbuffer"
            rgb_base_path = base_dir / "rgb"
            point_info_base_path = base_dir / "point_info"
            scene_paths = sorted(depth_base_path.iterdir())

            for scene_path in tqdm(scene_paths, desc="Scanning Taskonomy scenes"):
                if not scene_path.is_dir():
                    continue

                scene_name = scene_path.name
                
                depth_scene_path = scene_path / scene_name
                rgb_scene_path = rgb_base_path / scene_name / scene_name
                point_info_scene_path = point_info_base_path / scene_name / scene_name

                if not (rgb_scene_path.is_dir() and point_info_scene_path.is_dir()):
                    print(f"Warning: The '{scene_name}' is missing the corresponding rgb or point_info and has been skipped.")
                    continue

                depth_bases = {f.name.replace("_depth_zbuffer.png", "") for f in depth_scene_path.glob("*.png")}
                rgb_bases = {f.name.replace("_rgb.png", "") for f in rgb_scene_path.glob("*.png")}
                point_info_bases = {f.name.replace("_point_info.json", "") for f in point_info_scene_path.glob("*.json")}

                common_base_names = depth_bases.intersection(rgb_bases, point_info_bases)

                if not common_base_names:
                    continue

                intersection_filenames = [f"{base}_depth_zbuffer.png" for base in common_base_names]

                sorted_filenames = sorted(intersection_filenames, key=get_sort_key)

                if sorted_filenames:
                    for sorted_filename in sorted_filenames:
                        parts = sorted_filename.split("_")
                        key = f"{scene_name}/{parts[1]}"
                        value = parts[3]
                        sequence_list[key].append(value)

            with fio.open(sequence_list_path, "wb") as f:
                pickle.dump(sequence_list, f)
        
        all_sequences = []
        scene_meta = {}
        for scene_name, sequence in sequence_list.items():
            all_sequences.append(scene_name)
            scene_meta[scene_name] = {
                "frames_id": list(range(len(sequence))),
                "views_id": sequence
            }
        self.all_sequences = all_sequences
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
    def _load_one_view(self, scene_name, vid, resolution, rng):
        scene_point = scene_name.split("/")
        base_name = f"{scene_point[0]}/point_{scene_point[1]}_view_{vid}_domain_"

        ############ Obtaining image ############
        image_name = base_name + "rgb.png"
        image_path = osp.join(self.TASKONOMY_DIR, "rgb", scene_point[0], image_name)
        rgb_image = read_any_file(image_path)

        ############ Obtaining depth map  ############
        depth_name = base_name + "depth_zbuffer.png"
        depth_path = osp.join(self.TASKONOMY_DIR, "depth_zbuffer", scene_point[0], depth_name)
        depthmap = read_any_file(depth_path) / 512.0
        depthmap = threshold_depth_map(depthmap, max_depth=127.99)

        if rgb_image.shape[:2] != depthmap.shape:
            print(f"shape of image: {rgb_image.shape}")
            print(f"shape of depth: {depthmap.shape}")
            return None
        original_size = depthmap.shape

        ############ Obtain Camera Extrinsics and intrinsics ############
        point_name = base_name + "point_info.json"
        point_path = osp.join(self.TASKONOMY_DIR, "point_info", scene_point[0], point_name)
        with fio.open(point_path) as f:
            point_info = json.load(f)

        location = point_info["camera_location"]
        rotation = point_info["camera_rotation_final"]
        fov = point_info["field_of_view_rads"]

        # Recover cam -> world
        ex, ey, ez = rotation
        angles = np.array([ex - EULER_X_OFFSET_RADS, -ey, -ez], dtype=np.float64)
        R = euler_angles_to_matrix(angles, "XZY")
        Tx, Ty, Tz = location
        T = np.array([-Tx, Tz, Ty], dtype=np.float64)

        # P3D expects world -> cam
        T_inv = -R @ T
        # T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
        # T_inv = T
        # R_inv = R
        proj_K = compute_projection_matrix(znear=0.001, zfar=512.0, fov=fov, aspect_ratio=1.0, degrees=False)

        # Opencv world--> cam
        h, w = original_size
        fx, fy, cx, cy = proj_K[0, 0] * w / 2, proj_K[1, 1] * h / 2, (w - proj_K[0, 2] * w) / 2, (proj_K[1, 2] * h + h) / 2
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ R.astype(np.float32)
        T = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ T_inv.astype(np.float32)
        # extrinsic = np.hstack((R, T.reshape(3, 1)))
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T
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
            label=osp.basename(scene_name), 
            instance=str(vid),
        )

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        scene_name = self.sequences[index]
        scene_point = scene_name.split("/")
        meta = self.scene_meta[scene_name]

        if self.sample_method is None:
            idxs = self._sample_frame_ids(meta["frames_id"], rng)
        elif self.sample_method=='pose_sim':
            with fio.open(osp.join(self.TASKONOMY_DIR, "../covisibility", f"{scene_point[0]}_{scene_point[1]}.pkl"), "rb") as f:
                ranking, dists = pickle.load(f)
            random_id = np.random.randint(0, ranking.shape[0])
            idxs = ranking[random_id, :self.frame_num]
            np.random.shuffle(idxs)
        else:
            raise NotImplementedError("Not implemented sample method '{}'".format(self.sample_method))

        self.this_views_info = {
            "scene": scene_name,
            "idxs": idxs,
        }

        views = []
        for fid in idxs:
            vid = meta["views_id"][fid]
            views.append(
                self._load_one_view(
                    scene_name=scene_name,
                    vid=vid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views

def get_sort_key(filename: str) -> tuple[int, int]:
    """
    Extracts point and view numbers from a filename for sorting.

    This function uses a regular expression to find numbers after "point_"
    and "view_". It returns them as a tuple of integers, e.g., (10, 1),
    which allows Python's `sorted()` to sort by point, then by view.

    Args:
        filename: The filename string (e.g., "point_10_view_1_domain_depth_zbuffer.png").

    Returns:
        A tuple containing the point and view numbers as integers.
    """
    match = re.search(r"point_(\d+)_view_(\d+)", filename)
    if match:
        point_num = int(match.group(1))
        view_num = int(match.group(2))
        return (point_num, view_num)
    # Return a "large" tuple for any filenames that don't match the pattern
    # so they are sorted at the end.
    return (float("inf"), float("inf"))


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.taskonomy_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(TaskonomyDataset))
