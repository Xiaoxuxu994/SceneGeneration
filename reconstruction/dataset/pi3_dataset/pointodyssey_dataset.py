import sys
sys.path.append('.')

import os
import os.path as osp
import json
import hashlib

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .base.base_dataset import BaseDataset, CACHE_DIR
from .base.transforms import *


def _stem(x):
    return osp.splitext(osp.basename(str(x)))[0]


def _as_array(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype)


class PointOdysseyWAIDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "PointOdyssey"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images

        self.is_metric_scale = True

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        os.makedirs(CACHE_DIR, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(CACHE_DIR, f"{self.dataset_label}_{root_hash}_cache.pkl")

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
        scene_names = [
            x for x in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, x))
        ]
        scene_names = sorted(scene_names)

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for scene in tqdm(scene_names, desc="Building PointOdyssey WAI cache"):
            scene_path = osp.join(self.data_root, scene)
            meta_path = osp.join(scene_path, "scene_meta.json")

            if not osp.isfile(meta_path):
                continue

            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[PointOdyssey] Skip scene {scene}, failed to read scene_meta.json: {e}")
                continue

            frame_ids, camera_info, file_info = self._parse_scene_meta(scene_path, meta)

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

    def _parse_scene_meta(self, scene_path, meta):
        """
        兼容两种常见格式：
        1) newer style:
           {
             "frames": [
               {
                 "frame_name": "...",
                 "image": "images/xxx.png",
                 "depth": "depths/xxx.npy",
                 "transform_matrix": [...],
                 "fl_x": ..., "fl_y": ..., "cx": ..., "cy": ..., "h": ..., "w": ...
               },
               ...
             ]
           }

        2) older style:
           {
             "frame_names": {
               "xxx.png": {...},
               ...
             }
           }
        """
        frame_ids = []
        camera_info = {}
        file_info = {}

        # ---- style 1: frames list ----
        if "frames" in meta and isinstance(meta["frames"], list):
            for fr in meta["frames"]:
                fid = fr.get("frame_name", None)
                if fid is None:
                    # fallback: try image path stem
                    if "image" in fr:
                        fid = _stem(fr["image"])
                    elif "file_path" in fr:
                        fid = _stem(fr["file_path"])
                    else:
                        continue

                image_rel = fr.get("image", fr.get("file_path", None))
                depth_rel = fr.get("depth", None)

                K = self._extract_intrinsics_from_frame(fr)
                pose = self._extract_pose_from_frame(fr)

                if K is None or pose is None:
                    continue

                image_path = self._resolve_file(scene_path, image_rel, "images", fid, [".png", ".jpg", ".jpeg"])
                depth_path = self._resolve_file(scene_path, depth_rel, "depths", fid, [".npy", ".exr", ".png"])

                if image_path is None or depth_path is None:
                    continue

                frame_ids.append(fid)
                camera_info[fid] = {
                    "camera_pose": pose.astype(np.float32),         # c2w
                    "camera_intrinsics": K.astype(np.float32),
                }
                file_info[fid] = {
                    "img_path": image_path,
                    "depth_path": depth_path,
                }

            return frame_ids, camera_info, file_info

        return [], {}, {}

    def _extract_intrinsics_from_frame(self, fr):
        # common explicit style
        if all(k in fr for k in ["fl_x", "fl_y", "cx", "cy"]):
            fx = float(fr["fl_x"])
            fy = float(fr["fl_y"])
            cx = float(fr["cx"])
            cy = float(fr["cy"])
            return np.array(
                [
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

        # matrix style
        for key in ["intrinsics", "camera_intrinsics", "K"]:
            if key in fr:
                arr = _as_array(fr[key], np.float32)
                if arr.size == 9:
                    return arr.reshape(3, 3)

        return None

    def _extract_pose_from_frame(self, fr):
        # WAI / common processed format
        for key in ["transform_matrix", "camera_pose", "c2w", "cam2world", "extrinsics", "pose"]:
            if key in fr:
                arr = _as_array(fr[key], np.float32)
                if arr.size == 16:
                    return arr.reshape(4, 4)

        # optional traj path
        traj_rel = fr.get("traj", None)
        if traj_rel is not None:
            return None

        return None

    def _resolve_file(self, scene_path, rel_path, default_subdir, fid, exts):
        # 1) scene_meta relative path
        if rel_path is not None:
            p = osp.join(scene_path, rel_path)
            if osp.isfile(p):
                return p

        # 2) default_subdir/fid.ext
        for ext in exts:
            p = osp.join(scene_path, default_subdir, f"{fid}{ext}")
            if osp.isfile(p):
                return p

        # 3) default_subdir/stem(rel_path).ext
        if rel_path is not None:
            rel_stem = _stem(rel_path)
            for ext in exts:
                p = osp.join(scene_path, default_subdir, f"{rel_stem}{ext}")
                if osp.isfile(p):
                    return p

        return None

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
    # depth helpers
    # =========================
    def _read_depth(self, path):
        ext = osp.splitext(path)[1].lower()

        if ext == ".npy":
            depth = np.load(path).astype(np.float32)
        else:
            depth = iio.imread(path).astype(np.float32)

        depth = np.asarray(depth, dtype=np.float32)

        # 兼容 EXR 被读成 H x W x C
        if depth.ndim == 3:
            if depth.shape[2] == 1:
                depth = depth[..., 0]
            elif depth.shape[2] == 3:
                depth = depth[..., 0]
            else:
                raise ValueError(f"Unexpected depth shape {depth.shape} @ {path}")

        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"Unexpected depth shape {depth.shape} @ {path}")

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth <= 0)] = 0.0
        return depth.astype(np.float32)

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_name, fid, cam_info, file_info, resolution, rng):
        img_path = file_info["img_path"]
        depth_path = file_info["depth_path"]

        rgb_image = Image.open(img_path).convert("RGB")
        depthmap = self._read_depth(depth_path)
        intrinsics = cam_info["camera_intrinsics"].copy()

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

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/pointodyssey"
    out_dir = "output_debug/PointOdyssey/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = PointOdysseyWAIDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=24,
        min_num_images=8,
        split_ratio=1.0,
        verbose=False,
        shuffle=False,
        z_far=200.0,
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

    visualize_gt(views, out_dir, prefix="PointOdyssey_GT")