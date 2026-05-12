import sys
sys.path.append('.')

import os
import os.path as osp
import json
import hashlib

# 必须在 import cv2 前设置
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import rdepth_to_zdepth, threshold_depth_map


class OmniObject3DNewDataset(BaseDataset):
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

        self.dataset_label = "OmniObject3DNew"
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

        for scene in tqdm(scene_names, desc="Building OmniObject3D-New cache"):
            scene_path = osp.join(self.data_root, scene)
            render_path = osp.join(scene_path, "render")
            image_dir = osp.join(render_path, "images")
            depth_dir = osp.join(render_path, "depths")
            trans_path = osp.join(render_path, "transforms.json")

            if not (
                osp.isdir(render_path)
                and osp.isdir(image_dir)
                and osp.isdir(depth_dir)
                and osp.isfile(trans_path)
            ):
                continue

            try:
                with open(trans_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[OmniObject3D-New] Skip scene {scene}, failed to read transforms.json: {e}")
                continue

            camera_angle_x = float(meta["camera_angle_x"])
            frames = meta["frames"]

            frame_ids = []
            camera_info = {}
            file_info = {}

            for fr in frames:
                fid = osp.basename(fr["file_path"])   # e.g. r_0

                img_path = osp.join(image_dir, f"{fid}.png")
                depth_path = osp.join(depth_dir, f"{fid}_depth.exr")

                if not (osp.isfile(img_path) and osp.isfile(depth_path)):
                    continue

                img = imageio.imread(img_path)
                H, W = img.shape[:2]

                focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
                intrinsics = np.array(
                    [
                        [focal, 0.0, W / 2.0],
                        [0.0, focal, H / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

                P = np.array(fr["transform_matrix"], dtype=np.float32)
                pose = np.linalg.inv(P)
                pose[:, 1:3] *= -1

                frame_ids.append(fid)
                camera_info[fid] = {
                    "camera_pose": pose.astype(np.float32),
                    "camera_intrinsics": intrinsics.astype(np.float32),
                }
                file_info[fid] = {
                    "img_path": img_path,
                    "depth_path": depth_path,
                }

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
        img = imageio.imread(path)

        alpha_mask = None
        if img.shape[-1] == 4:
            alpha_mask = img[..., 3] > 0
            rgb_channels = img[..., :3]
            white_background = np.full_like(rgb_channels, 255)
            img = np.where(alpha_mask[..., None], rgb_channels, white_background)
        else:
            img = img[..., :3]

        img = img.astype(np.uint8)
        return Image.fromarray(img), alpha_mask

    def _read_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read depth: {path}")

        # 3 channels are identical, pick one
        depth = depth[..., 0].astype(np.float32)

        # invalid background in EXR
        depth[depth >= 65504.0] = 0.0

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth <= 0] = 0.0
        return depth.astype(np.float32)

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_name, fid, cam_info, file_info, resolution, rng):
        img_path = file_info["img_path"]
        depth_path = file_info["depth_path"]

        rgb_image, alpha_mask = self._read_rgb(img_path)
        depthmap = self._read_depth(depth_path)
        intrinsics = cam_info["camera_intrinsics"].copy()

        # 如果有 alpha，把背景深度直接清掉
        if alpha_mask is not None:
            depthmap[~alpha_mask] = 0.0

        rgb_image, depthmap, intrinsics_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsics.copy(),
            resolution,
            rng=rng,
            info=img_path,
        )

        # OmniObject3D: ray-depth -> z-depth
        depth_t = torch.from_numpy(depthmap).float()[None, None]
        intr_t = torch.from_numpy(intrinsics_).float()[None]
        depthmap = rdepth_to_zdepth(depth_t, intr_t)[0, 0].cpu().numpy()

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap[depthmap <= 0] = 0.0
        depthmap = threshold_depth_map(depthmap, max_percentile=98)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=cam_info["camera_pose"].astype(np.float32),   # c2w
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

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/Omniobject3d-New/blender_renders_decode"
    out_dir = "output_debug/OmniObject3DNew/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = OmniObject3DNewDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=24,
        min_num_images=8,
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

    visualize_gt(views, out_dir, prefix="OmniObject3D_New_GT")