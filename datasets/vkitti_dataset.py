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

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class VKittiDataset(BaseDataset):
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

        self.dataset_label = "VirtualKitti"
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
        cache_path = os.path.join(
            cache_dir,
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

        # 正确结构:
        # data_root/Scene01/rain/frames/rgb/Camera_0
        rgb_cam_dirs = sorted(glob(osp.join(self.data_root, "*", "*", "frames", "rgb", "Camera_*")))

        for rgb_cam_dir in tqdm(rgb_cam_dirs, desc=f"Building {self.dataset_label} cache"):
            cam_name = osp.basename(rgb_cam_dir)                        # Camera_0
            rgb_root = rgb_cam_dir                                      # .../frames/rgb/Camera_0
            frames_rgb_dir = osp.dirname(rgb_root)                      # .../frames/rgb
            frames_dir = osp.dirname(frames_rgb_dir)                    # .../frames
            scene_path = osp.dirname(frames_dir)                        # .../Scene01/rain

            rel_scene_path = osp.relpath(scene_path, self.data_root)
            path_parts = rel_scene_path.split(os.sep)
            if len(path_parts) != 2:
                if self.verbose:
                    print(f"[{self.dataset_label}] Skip malformed scene path: {rel_scene_path}")
                continue

            depth_root = osp.join(frames_dir, "depth", cam_name)
            if not osp.isdir(depth_root):
                if self.verbose:
                    print(f"[{self.dataset_label}] Skip invalid scene (missing depth dir): {depth_root}")
                continue

            scene_name = (path_parts[0], path_parts[1], cam_name)       # ("Scene01", "rain", "Camera_0")

            extri_file = osp.join(scene_path, "extrinsic.txt")
            intri_file = osp.join(scene_path, "intrinsic.txt")

            if not (osp.isfile(extri_file) and osp.isfile(intri_file)):
                if self.verbose:
                    print(f"[{self.dataset_label}] Skip invalid scene (missing camera txt): {scene_path}")
                continue

            try:
                extrinsics_all = np.loadtxt(extri_file, delimiter=" ", skiprows=1).astype(np.float32)
                intrinsics_all = np.loadtxt(intri_file, delimiter=" ", skiprows=1).astype(np.float32)
            except Exception as e:
                print(f"[{self.dataset_label}] Skip scene {scene_name}, failed to load camera txt: {e}")
                continue

            try:
                camera_id = int(cam_name.split("_")[-1])   # Camera_0 -> 0
            except Exception:
                print(f"[{self.dataset_label}] Skip scene {scene_name}, invalid camera name: {cam_name}")
                continue

            extrinsics_cam = extrinsics_all[extrinsics_all[:, 1] == camera_id]
            intrinsics_cam = intrinsics_all[intrinsics_all[:, 1] == camera_id]

            if len(extrinsics_cam) == 0 or len(intrinsics_cam) == 0:
                if self.verbose:
                    print(f"[{self.dataset_label}] Skip scene {scene_name}, no camera params for camera_id={camera_id}")
                continue

            rgb_ids = {
                osp.basename(x).replace("rgb_", "").replace(".jpg", "")
                for x in glob(osp.join(rgb_root, "rgb_*.jpg"))
            }
            depth_ids = {
                osp.basename(x).replace("depth_", "").replace(".png", "")
                for x in glob(osp.join(depth_root, "depth_*.png"))
            }

            inter_ids = rgb_ids & depth_ids
            numeric_ids = [x for x in inter_ids if str(x).isdigit()]
            frame_ids = sorted(numeric_ids, key=lambda x: int(x))

            max_cam_len = min(len(extrinsics_cam), len(intrinsics_cam))
            frame_ids = [fid for fid in frame_ids if int(fid) < max_cam_len]

            if len(frame_ids) < self.min_num_images:
                if self.verbose:
                    print(
                        f"[{self.dataset_label}] Skip short scene {scene_name}, "
                        f"valid_frames={len(frame_ids)} < min_num_images={self.min_num_images}"
                    )
                continue

            all_sequences.append(scene_name)
            scene_to_path[scene_name] = scene_path
            scene_meta[scene_name] = {
                "frame_ids": frame_ids,
                "extrinsics": extrinsics_cam,
                "intrinsics": intrinsics_cam,
                "camera_id": camera_id,
                "camera_name": cam_name,
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
    # re-implement in basedataset

    # =========================
    # camera helpers
    # =========================
    def _build_intrinsic_from_row(self, intrinsic_row):
        """
        按原始 VKitti 逻辑：
        intri_opencv[0,0] = row[-4]
        intri_opencv[1,1] = row[-3]
        intri_opencv[0,2] = row[-2]
        intri_opencv[1,2] = row[-1]
        """
        intri = np.eye(3, dtype=np.float32)
        intri[0, 0] = intrinsic_row[-4]
        intri[1, 1] = intrinsic_row[-3]
        intri[0, 2] = intrinsic_row[-2]
        intri[1, 2] = intrinsic_row[-1]
        return intri

    def _build_camera_pose_from_extri(self, extrinsic_row):
        """
        原代码 world2cam 形式, 转成 c2w 4x4 再返回。
        """
        w2c = extrinsic_row[2:].reshape(4, 4).astype(np.float32)
        c2w = np.linalg.inv(w2c).astype(np.float32)
        return c2w

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, scene_name, fid, scene_meta, resolution, rng):
        idx = int(fid)
        cam_name = scene_meta["camera_name"]

        impath = osp.join(scene_path, "frames", "rgb", cam_name, f"rgb_{idx:05d}.jpg")
        depthpath = osp.join(scene_path, "frames", "depth", cam_name, f"depth_{idx:05d}.png")

        rgb_image = np.array(Image.open(impath).convert("RGB"))
        depthmap = np.array(Image.open(depthpath)).astype(np.float32)

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap = depthmap / 100.0
        depthmap = threshold_depth_map(
            depthmap,
            min_percentile=-1,
            max_percentile=98,
        ).astype(np.float32)

        intrinsic = self._build_intrinsic_from_row(scene_meta["intrinsics"][idx])
        camera_pose = self._build_camera_pose_from_extri(scene_meta["extrinsics"][idx])

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=impath,
        )

        return dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=intrinsic_.astype(np.float32),
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
                    scene_meta=meta,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt
    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/vkitti2"
    out_dir = "output_debug/vKitti2/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = VKittiDataset(
        data_root=data_root,
        mode="train",
        resolution=[[512, 336]],
        frame_num=8,
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
    visualize_gt(views, out_dir, prefix="vKitti_GT")