import sys
sys.path.append('.')

import os
import os.path as osp
import json
import pickle
import hashlib
from glob import glob

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
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


class MVSSynthDataset(BaseDataset):
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

        self.dataset_label = "MVS"
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
    # split / scan / cache
    # =========================
    def _scan_scenes(self):
        scene_names = [
            x for x in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, x))
        ]
        scene_names = sorted(scene_names)

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for scene in tqdm(scene_names, desc="Building MVS cache"):
            scene_path = osp.join(self.data_root, scene)
            rgb_dir = osp.join(scene_path, "images")
            depth_dir = osp.join(scene_path, "depths")
            pose_dir = osp.join(scene_path, "poses")

            if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir) and osp.isdir(pose_dir)):
                continue

            rgb_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in glob(osp.join(rgb_dir, "*.png"))
            }
            depth_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in glob(osp.join(depth_dir, "*.exr"))
            }
            pose_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in glob(osp.join(pose_dir, "*.json"))
            }

            frame_ids = sorted(list(rgb_ids & depth_ids & pose_ids), key=lambda x: int(x))

            if len(frame_ids) < self.min_num_images:
                continue

            all_sequences.append(scene)
            scene_to_path[scene] = scene_path
            scene_meta[scene] = {
                "frame_ids": frame_ids,
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
    # helpers
    # =========================
    def _read_exr_depth(self, path):
        """
        读取单通道 EXR depth。
        """
        depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to read EXR depth: {path}")

        if depth.ndim == 3:
            # 有些 exr 会读成 HxWx3，这里取第一通道
            depth = depth[..., 0]

        depth = depth.astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth

    def _build_intrinsic_hardcode_pixel(self, data, W, H):
        # fx = 771.89826242
        fx = 771.89826242 / (2*640) * W
        fy = float(data["f_y"])
        cx = 0.5 * W
        cy = 0.5 * H

        intrinsic = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return intrinsic

    # =========================
    # sample
    # =========================
    # re-implement in basedataset

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, scene_name, fid, resolution, rng):
        image_path = osp.join(scene_path, "images", f"{fid}.png")
        depth_path = osp.join(scene_path, "depths", f"{fid}.exr")
        pose_path = osp.join(scene_path, "poses", f"{fid}.json")

        rgb_image = Image.open(image_path).convert("RGB")
        depthmap = self._read_exr_depth(depth_path)

        with open(pose_path, "r") as f:
            data = json.load(f)

        W0, H0 = rgb_image.size
        intrinsic = self._build_intrinsic_hardcode_pixel(data, W0, H0)

        # # # =====================================================
        # # # MVS-Synth: follow the scale factor of MoGe
        # # # https://github.com/microsoft/MoGe/issues/123
        # # # Pi3 training should not be sensitive to this
        # # # =====================================================
        extrinsic_w2c = np.array(data["extrinsic"], dtype=np.float32)
        depthmap = depthmap / 10.0
        extrinsic_w2c[:3, 3] /= 10.0
        camera_pose = np.linalg.inv(extrinsic_w2c).astype(np.float32)

        # 清理 depth
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=95)

        W0, H0 = rgb_image.size
        assert depthmap.shape == (H0, W0), f"Image/depth mismatch: {(H0, W0)} vs {depthmap.shape}"

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=image_path,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose.astype(np.float32),         # c2w
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

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    scene_name=scene,
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/MVS-Synth/GTAV_540/"
    out_dir = "output_debug/MVS_hardcodeK_yy3/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = MVSSynthDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=8,
        min_num_images=24,
        z_far=160,
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

    visualize_gt(views, out_dir, prefix="MVS_GT")