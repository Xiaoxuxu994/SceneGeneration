import sys
sys.path.append('.')

import os
import os.path as osp
import hashlib
from glob import glob
from pathlib import Path

import h5py
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from io import BytesIO

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class GTASfMDataset(BaseDataset):
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
        '''
            https://wang-kx.github.io/pdf/flow_motion_depth_sup.pdf
            z_far=200
        '''
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "GTA-sfm"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # lazy h5 handle
        self._h5_handles = {}

        cache_dir = "/efs-mi-east4-2/z00533370/proj_3dgs/3drecon/data/dataset_cache"
        os.makedirs(cache_dir, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")
        if not os.path.exists(cache_path):
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
        h5_files = []
        for subdir in ["train", "test"]:
            h5_files.extend(sorted(glob(osp.join(self.data_root, subdir, "*.hdf5"))))

        if len(h5_files) == 0:
            raise RuntimeError(
                f"[{self.dataset_label}] No .hdf5 files found in "
                f"{osp.join(self.data_root, 'train')} / {osp.join(self.data_root, 'test')} / {self.data_root}"
            )

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for h5_path in tqdm(h5_files, desc=f"Building {self.dataset_label} cache"):
            # 为了避免 train/test 下重名文件冲突，把相对路径编码进 scene name
            rel_path = osp.relpath(h5_path, self.data_root)
            scene = rel_path.replace(os.sep, "_")
            if scene.endswith(".hdf5"):
                scene = scene[:-5]

            try:
                with h5py.File(h5_path, "r") as f:
                    layout = self._inspect_h5_layout(f)
                    frame_ids = layout["frame_ids"]

                if len(frame_ids) < self.min_num_images:
                    continue

                all_sequences.append(scene)
                scene_to_path[scene] = str(h5_path)
                scene_meta[scene] = {
                    "frame_ids": frame_ids,
                }

            except Exception as e:
                print(f"[{self.dataset_label}] Skip file {h5_path}, reason: {e}")
                continue

        self.all_sequences = sorted(all_sequences)
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

    def _inspect_h5_layout(self, f):
        keys = list(f.keys())

        def collect_ids(prefix):
            ids = set()
            for k in keys:
                if k.startswith(prefix):
                    suffix = k[len(prefix):]
                    if suffix.isdigit():
                        ids.add(suffix)
            return ids

        k_ids = collect_ids("K_")
        depth_ids = collect_ids("depth_")
        image_ids = collect_ids("image_")
        pose_ids = collect_ids("pose_")

        frame_ids = sorted(
            list(k_ids & depth_ids & image_ids & pose_ids),
            key=lambda x: int(x)
        )

        if len(frame_ids) == 0:
            raise RuntimeError(
                f"[{self.dataset_label}] No valid frame intersection found. "
                f"#K={len(k_ids)}, #depth={len(depth_ids)}, "
                f"#image={len(image_ids)}, #pose={len(pose_ids)}"
            )

        return {
            "frame_ids": frame_ids,
        }
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
    # h5 helpers
    # =========================
    def _get_h5_handle(self, h5_path):
        """
        lazy open per worker/process
        """
        if h5_path not in self._h5_handles:
            self._h5_handles[h5_path] = h5py.File(h5_path, "r")
        return self._h5_handles[h5_path]

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
    def _load_one_view(self, h5_path, scene_name, fid, resolution, rng):
        f = self._get_h5_handle(h5_path)

        image_raw = np.asarray(f[f"image_{fid}"])
        depthmap = np.asarray(f[f"depth_{fid}"]).astype(np.float32)
        intrinsic = np.asarray(f[f"K_{fid}"]).astype(np.float32)
        camera_pose = np.asarray(f[f"pose_{fid}"]).astype(np.float32)

        image_bytes = image_raw.reshape(-1).astype(np.uint8).tobytes()
        rgb_image = Image.open(BytesIO(image_bytes)).convert("RGB")


        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=f"{h5_path}:{fid}",
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
        h5_path = self.scene_to_path[scene]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frame_ids"], rng)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
            "h5_path": h5_path,
        }

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    h5_path=h5_path,
                    scene_name=scene,
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/GTA-SfM"
    # data_root = "/efs-m3ga/h00548084/relive3r_raw/GTA-SfM/"

    out_dir = "output_debug/GTASfM/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = GTASfMDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=24,
        max_distance=8,
        z_far=200,
        min_num_images=24,
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

    visualize_gt(views, out_dir, prefix="GTASfM_GT")