import sys
sys.path.append('.')

import os
import os.path as osp
import hashlib
import json

import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class SpatialVIDDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=8,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=16,
        pose_is_w2c=True,
        **kwargs,
    ):
        '''
            SpatialVID-HQ（https://huggingface.co/datasets/SpatialVID/SpatialVID-HQ），真实视频 + MegaSaM 类标注。
            data_root：tools/preprocess_spatialvid.py 的输出目录（本地或 obs://）：
                <data_root>/<group>/_index.json                  {clip: 帧数}
                <data_root>/<group>/<clip>/frame_<k:05d>.jpg     按官方规则（每 int(fps/5) 帧一帧）抽出的 RGB，584x328
                <data_root>/<group>/<clip>/depth_<k:05d>.npy     float16 米制深度，0 = 无效（负值、天空、动态区域已置 0）
                <data_root>/<group>/<clip>/cams.pkl              {"<k:05d>": (K 3x3 像素, poses.npy 构造的 4x4)}
            pose_is_w2c：官方 quat_to_mat.py 把 poses.npy 构造成 w2c，这里求逆得到 c2w；可视化若显示约定相反，改成 False。
            帧率约 5fps，max_distance=8 约等于 1.6 秒内取帧。
        '''
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "SpatialVID"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.pose_is_w2c = pose_is_w2c

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)
        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")
        cache_path = self.cache_file or cache_path
        if not fio.exists(cache_path):
            print(f"[Dataset] Constructing cache {cache_path}..")
            self._scan_scenes()
            self._save_cache(
                keys=["all_sequences", "scene_to_path", "scene_meta"],
                save_path=cache_path
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
        # 只读每个 group 的 _index.json（预处理写完一个 group 才写它），不逐个 clip 列目录
        groups = sorted(g for g in fio.listdir(self.data_root) if g.startswith("group_"))
        all_sequences, scene_to_path, scene_meta = [], {}, {}
        for group in groups:
            index_path = osp.join(self.data_root, group, "_index.json")
            if not fio.exists(index_path):
                print(f"[{self.dataset_label}] Skip {group}: no _index.json (preprocessing not finished)")
                continue
            index = json.loads(fio.read_bytes(index_path).decode())
            for clip, n in sorted(index.items()):
                if n < self.min_num_images:
                    continue
                scene = f"{group}/{clip}"
                all_sequences.append(scene)
                scene_to_path[scene] = osp.join(self.data_root, group, clip)
                scene_meta[scene] = {"frame_ids": [f"{k:05d}" for k in range(n)]}

        if len(all_sequences) == 0:
            raise RuntimeError(f"[{self.dataset_label}] No preprocessed clips found under {self.data_root}")

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
        else:
            return all_sequences[split_idx:]

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, clip_dir, scene_name, fid, cams, resolution, rng):
        rgb_image = fio.pil_open(osp.join(clip_dir, f"frame_{fid}.jpg")).convert("RGB")
        depthmap = fio.np_load(osp.join(clip_dir, f"depth_{fid}.npy")).astype(np.float32)
        intrinsic, pose = (np.asarray(x, dtype=np.float32) for x in cams[fid])
        camera_pose = np.linalg.inv(pose.astype(np.float64)).astype(np.float32) if self.pose_is_w2c else pose

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=f"{clip_dir}:{fid}",
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose.astype(np.float32),
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
        clip_dir = self.scene_to_path[scene]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frame_ids"], rng)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        cams = fio.load_pickle(osp.join(clip_dir, "cams.pkl"))
        return [
            self._load_one_view(clip_dir, scene, fid, cams, resolution, rng)
            for fid in idxs
        ]


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.spatialvid_dataset --n 3 [--data_root ... --cache_dir ... --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(SpatialVIDDataset))
