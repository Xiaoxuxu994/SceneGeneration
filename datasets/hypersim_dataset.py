import sys
sys.path.append('.')

import os
import os.path as osp
import re
import hashlib
from glob import glob

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


def read_hdf5(path, return_dtype=None):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        assert len(keys) > 0, f"No dataset found in {path}"
        arr = f[keys[0]][()]
    if return_dtype is not None:
        arr = arr.astype(return_dtype)
    return arr


def ensure_4x4(extr):
    extr = np.asarray(extr, dtype=np.float32)
    if extr.shape == (4, 4):
        return extr
    if extr.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = extr
        return out
    raise ValueError(f"Unsupported extrinsic shape: {extr.shape}")


class HypersimDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "Hypersim"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio

        self.metadata_file = osp.join(self.data_root, "scene_metadata.csv")

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        cache_dir = "/efs-mi-east4-2/z00533370/proj_3dgs/3drecon/data/dataset_cache"
        os.makedirs(cache_dir, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")

        if not os.path.exists(cache_path):
            print(f"[Dataset] Constructing cache {cache_path} ..")
            self._scan_scenes()
            self._save_cache(keys=["all_sequences", "scene_to_path", "scene_meta"], save_path=cache_path)
            print(f"[Dataset] Cache saved to {cache_path}")
        else:
            print(f"[Dataset] Loading cache from {cache_path} ..")
            res = self._load_cache(cache_path)
            if res:
                self.cache_loaded = True
                print(f"[Dataset] Cache loaded.")

        self.metadata_df = pd.read_csv(self.metadata_file)

        self.sequences = self._split_sequences(self.all_sequences)

        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]

        if self.verbose:
            print(f"[{self.dataset_label}] Sequences: {self.sequences}")

        print(
            f"[{self.dataset_label}] Found {len(self.sequences)} unique scenes in {self.data_root} ({self.mode})",
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

        for scene in tqdm(scene_names, desc="Building Hypersim cache"):
            scene_path = osp.join(self.data_root, scene)

            final_dirs = glob(osp.join(scene_path, "scene_*_final_hdf5"))

            for final_dir in final_dirs:
                final_name = osp.basename(final_dir)  # e.g. scene_cam_00_final_hdf5
                if not final_name.startswith("scene_") or not final_name.endswith("_final_hdf5"):
                    continue

                camera_name = final_name[len("scene_"):-len("_final_hdf5")]   # cam_00

                rgb_dir = osp.join(scene_path, f"scene_{camera_name}_final_hdf5")
                depth_dir = osp.join(scene_path, f"scene_{camera_name}_geometry_hdf5")
                extr_path = osp.join(scene_path, f"{camera_name}_extrinsics.hdf5")

                if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir) and osp.isfile(extr_path)):
                    continue

                rgb_ids = {
                    m.group(1)
                    for x in glob(osp.join(rgb_dir, "frame.*.color.hdf5"))
                    for m in [re.search(r"frame\.(\d+)\.color\.hdf5$", osp.basename(x))]
                    if m is not None
                }

                depth_ids = {
                    m.group(1)
                    for x in glob(osp.join(depth_dir, "frame.*.depth_meters.hdf5"))
                    for m in [re.search(r"frame\.(\d+)\.depth_meters\.hdf5$", osp.basename(x))]
                    if m is not None
                }

                frame_ids = sorted(rgb_ids & depth_ids, key=lambda x: int(x))

                if len(frame_ids) == 0:
                    continue

                seq_name = f"{scene}__{camera_name}"
                all_sequences.append(seq_name)
                scene_to_path[seq_name] = scene_path
                scene_meta[seq_name] = {
                    "scene_name": scene,
                    "camera_name": camera_name,
                    "frame_ids": frame_ids,
                    "extrinsics_path": extr_path,
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
    def _build_intrinsics(self, scene_name):
        row = self.metadata_df[self.metadata_df["scene_name"] == scene_name].iloc[0]

        K = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                K[i, j] = row[f"M_cam_from_uv_{i}{j}"]
        return K

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, scene_name, camera_name, fid, scene_extrinsics, resolution, rng):
        rgb_path = osp.join(
            scene_path,
            f"scene_{camera_name}_final_hdf5",
            f"frame.{fid}.color.hdf5",
        )
        depth_path = osp.join(
            scene_path,
            f"scene_{camera_name}_geometry_hdf5",
            f"frame.{fid}.depth_meters.hdf5",
        )

        image = read_hdf5(rgb_path)
        depthmap = read_hdf5(depth_path, return_dtype=np.float32)

        intrinsic = self._build_intrinsics(scene_name)

        frame_idx = int(fid)
        extrinsic_w2c = ensure_4x4(scene_extrinsics[frame_idx])

        # c2w
        camera_pose = np.linalg.inv(extrinsic_w2c).astype(np.float32)

        image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=rgb_path,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        return dict(
            img=image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),   # c2w
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=f"{scene_name}_{camera_name}",
            instance=str(fid),
        )

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        seq_name = self.sequences[index]
        scene_path = self.scene_to_path[seq_name]
        meta = self.scene_meta[seq_name]

        idxs = self._sample_frame_ids(meta["frame_ids"], rng)
        scene_extrinsics = read_hdf5(meta["extrinsics_path"])

        self.this_views_info = {
            "scene": meta["scene_name"],
            "camera": meta["camera_name"],
            "idxs": idxs,
        }

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    scene_name=meta["scene_name"],
                    camera_name=meta["camera_name"],
                    fid=fid,
                    scene_extrinsics=scene_extrinsics,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/Hypersim"
    out_dir = "output_debug/Hypersim/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = HypersimDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=24,
        max_distance=24,
        split_ratio=0.9,
        verbose=False,
        shuffle=False,
    )

    print(f"len(dataset) = {len(dataset)}")

    views = dataset[0]
    print(f"[dataset] number of views = {len(views)}")
    for i, v in enumerate(views):
        img_shape = tuple(v["img"].shape) if hasattr(v["img"], "shape") else v["img"].size
        print(
            f"[dataset view {i}] "
            f"img={img_shape}, "
            f"depth={v['depthmap'].shape}, "
            f"depthmin={v['depthmap'].min()}, "
            f"depthmax={v['depthmap'].max()}, "
            f"pts3d={v['pts3d'].shape}, "
            f"valid={v['valid_mask'].sum()}"
        )

    visualize_gt(views, out_dir, prefix="Hypersim_GT")