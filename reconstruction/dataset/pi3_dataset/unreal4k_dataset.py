import sys
sys.path.append('.')

import os
import os.path as osp
import json
import hashlib

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
from .sample_utils.common import threshold_depth_map


def parse_frame_name(frame_name):
    # "00000_cam0" -> ("00000", 0)
    stem, cam = frame_name.split("_cam")
    return stem, int(cam)


def compute_baseline(T0, T1):
    # baseline 取相对位姿在 x 方向的平移
    T_01 = np.linalg.inv(T0) @ T1
    return float(abs(T_01[0, 3]))


class Unreal4KDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        view_idx=0,   # 0 -> Image0/Disp0, 1 -> Image1/Disp1
        **kwargs,
    ):
        '''
        https://github.com/CUT3R/CUT3R/blob/main/datasets_preprocess/preprocess_unreal4k.py
        '''
        super().__init__(**kwargs)
        assert data_root is not None
        assert view_idx in [0, 1]

        self.dataset_label = "Unreal4k"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.view_idx = view_idx

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        os.makedirs(CACHE_DIR, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(
            CACHE_DIR, f"{self.dataset_label}_{root_hash}_cache.pkl"
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
        scene_names = [
            x for x in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, x)) and x.isdigit()
        ]
        scene_names = sorted(scene_names, key=int)

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for scene in tqdm(scene_names, desc="Building UnrealStereo4K cache"):
            scene_path = osp.join(self.data_root, scene)
            meta_path = osp.join(scene_path, "scene_meta.json")

            image_dir = osp.join(scene_path, f"Image{self.view_idx}")
            disp_dir = osp.join(scene_path, f"Disp{self.view_idx}")

            if not (osp.isfile(meta_path) and osp.isdir(image_dir) and osp.isdir(disp_dir)):
                continue

            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                continue

            frames = meta.get("frames", [])
            if len(frames) == 0:
                continue

            # 先把所有 cam0/cam1 的相机信息读出来
            frame_dict = {}
            for fr in frames:
                frame_name = fr["frame_name"]
                stem, cam = parse_frame_name(frame_name)

                T = np.asarray(fr["transform_matrix"], dtype=np.float32)

                frame_dict[(stem, cam)] = {
                    "camera_pose": T.astype(np.float32),
                    "fl_x": float(fr["fl_x"]),
                    "fl_y": float(fr["fl_y"]),
                    "cx": float(fr["cx"]),
                    "cy": float(fr["cy"]),
                    "h_meta": int(fr["h"]),
                    "w_meta": int(fr["w"]),
                }

            frame_ids = []
            camera_info = {}

            other_cam = 1 - self.view_idx
            stems = sorted(list(set([k[0] for k in frame_dict.keys()])), key=int)

            for stem in stems:
                if (stem, self.view_idx) not in frame_dict:
                    continue
                if (stem, other_cam) not in frame_dict:
                    continue

                img_path = osp.join(scene_path, f"Image{self.view_idx}", f"{stem}.jpg")
                disp_path = osp.join(scene_path, f"Disp{self.view_idx}", f"{stem}.npy")
                if not (osp.isfile(img_path) and osp.isfile(disp_path)):
                    continue

                cam_self = frame_dict[(stem, self.view_idx)]
                cam_other = frame_dict[(stem, other_cam)]

                baseline = compute_baseline(
                    cam_self["camera_pose"],
                    cam_other["camera_pose"],
                )

                frame_ids.append(stem)
                camera_info[stem] = {
                    "camera_pose": cam_self["camera_pose"],
                    "fl_x": cam_self["fl_x"],
                    "fl_y": cam_self["fl_y"],
                    "cx": cam_self["cx"],
                    "cy": cam_self["cy"],
                    "h_meta": cam_self["h_meta"],
                    "w_meta": cam_self["w_meta"],
                    "baseline": baseline,
                }

            if len(frame_ids) == 0:
                continue

            all_sequences.append(scene)
            scene_to_path[scene] = scene_path
            scene_meta[scene] = {
                "frame_ids": frame_ids,
                "camera_info": camera_info,
            }

        self.all_sequences = sorted(all_sequences, key=int)
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
    def _build_intrinsic(self, cam_info, H, W):
        # json 里的 K 对应原始 3840x2160，要先缩放到当前 jpg 分辨率
        sx = float(W) / float(cam_info["w_meta"])
        sy = float(H) / float(cam_info["h_meta"])

        return np.array(
            [
                [cam_info["fl_x"] * sx, 0.0, cam_info["cx"] * sx],
                [0.0, cam_info["fl_y"] * sy, cam_info["cy"] * sy],
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
    def _load_one_view(self, scene_path, fid, cam_info, resolution, rng):
        img_path = osp.join(scene_path, f"Image{self.view_idx}", f"{fid}.jpg")
        disp_path = osp.join(scene_path, f"Disp{self.view_idx}", f"{fid}.npy")

        rgb_image = Image.open(img_path).convert("RGB")
        disparity = np.load(disp_path).astype(np.float32)
        disparity = np.squeeze(disparity)
        
        disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
        valid = np.isfinite(disparity) & (disparity > 0.1)
        
        W0, H0 = rgb_image.size
        assert disparity.shape == (H0, W0), f"shape mismatch: img={rgb_image.size}, disp={disparity.shape}"

        intrinsic = self._build_intrinsic(cam_info, H0, W0)
        baseline = cam_info["baseline"]

        depthmap = np.zeros_like(disparity, dtype=np.float32)
        depthmap[valid] = baseline * intrinsic[0, 0] / disparity[valid]

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=img_path,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap[depthmap < 0] = 0.0
        depthmap = threshold_depth_map(depthmap, max_percentile=98)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=cam_info["camera_pose"].astype(np.float32),
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
                    fid=fid,
                    cam_info=meta["camera_info"][fid],
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/efs-m3ga/z00533370/reliev3r_data/UnrealStereo4K_480P_decode/"
    out_dir = "output_debug/UnrealStereo4K/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = Unreal4KDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=24,
        split_ratio=0.9,
        z_far=200,
        verbose=False,
        shuffle=False,
        view_idx=0,
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

    visualize_gt(views, out_dir, prefix="UnrealStereo4K_GT")