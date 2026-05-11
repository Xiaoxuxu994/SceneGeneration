import sys
sys.path.append('.')

import os
import os.path as osp
from glob import glob
import re
import pickle
import hashlib

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


def _load_pose(path, ret_44=False):
    """
    Read BlendedMVS camera file.
    Format:
      - first 4x4: extrinsic, world2cam
      - then 3x3: intrinsic
    """
    with open(path, "r") as f:
        RT = np.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)  # world2cam
        assert RT.shape == (4, 4), f"Invalid extrinsic shape in {path}: {RT.shape}"

        K = np.loadtxt(f, skiprows=2, max_rows=3, dtype=np.float32)
        assert K.shape == (3, 3), f"Invalid intrinsic shape in {path}: {K.shape}"

    if ret_44:
        return K, RT
    return K, RT[:3, :]


def read_pfm(path):
    """
    Minimal PFM reader.
    """
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")

        color = (header == "PF")

        dim_line = f.readline().decode("utf-8").strip()
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("utf-8").strip()

        m = re.match(r"^(\d+)\s+(\d+)$", dim_line)
        if m is None:
            raise ValueError(f"Malformed PFM header in {path}: {dim_line}")
        width, height = map(int, m.groups())

        scale = float(f.readline().decode("utf-8").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")

        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data.astype(np.float32)


class BlendedMVSDataset(BaseDataset):
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

        self.dataset_label = "BlendedMVS"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        os.makedirs(CACHE_DIR, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(CACHE_DIR, f"{self.dataset_label}_{root_hash}_cache.pkl")
        if not os.path.exists(cache_path):
            print(f"[Dataset] Constructing cache {cache_path} ..")
            self._scan_scenes()
            self._save_cache(keys=["all_sequences", "scene_to_path", "scene_meta"],save_path=cache_path)
            print(f"[Dataset] Cache saved to {cache_path}")
        else:
            print(f"[Dataset] Loading cache from {cache_path} ..")
            res = self._load_cache(cache_path)
            if res:
                self.cache_loaded = True
                print(f"[Dataset] Cache loaded.")

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

        for scene in tqdm(scene_names, desc="Building BlendedMVS cache"):
            scene_path = osp.join(self.data_root, scene)
            rgb_dir = osp.join(scene_path, "blended_images")
            depth_dir = osp.join(scene_path, "rendered_depth_maps")
            cam_dir = osp.join(scene_path, "cams")

            if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir) and osp.isdir(cam_dir)):
                continue

            rgb_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in glob(osp.join(rgb_dir, "*.jpg"))
                if not osp.splitext(osp.basename(x))[0].endswith("_masked")
            }
            depth_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in glob(osp.join(depth_dir, "*.pfm"))
            }
            cam_ids = {
                osp.basename(x).replace("_cam.txt", "")
                for x in glob(osp.join(cam_dir, "*_cam.txt"))
            }

            frame_ids = sorted(
                rgb_ids & depth_ids & cam_ids,
                key=self._safe_frame_sort_key
            )

            if len(frame_ids) == 0:
                continue

            all_sequences.append(scene)
            scene_to_path[scene] = scene_path
            scene_meta[scene] = {
                "frame_ids": frame_ids,
            }

        self.all_sequences = all_sequences
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

    def _safe_frame_sort_key(self, x):
        nums = re.findall(r"\d+", x)
        if len(nums) > 0:
            return int(nums[0])
        return x

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
    # load one view
    # =========================
    def _load_one_view(self, scene_path, fid, resolution, rng):
        impath = osp.join(scene_path, "blended_images", f"{fid}.jpg")
        depthpath = osp.join(scene_path, "rendered_depth_maps", f"{fid}.pfm")
        campath = osp.join(scene_path, "cams", f"{fid}_cam.txt")

        rgb_image = Image.open(impath).convert("RGB")
        depthmap = read_pfm(depthpath).astype(np.float32)

        if depthmap.ndim == 3:
            depthmap = depthmap[..., 0]

        intrinsic, extrinsic_w2c = _load_pose(campath, ret_44=True)

        # c2w
        camera_pose = np.linalg.inv(extrinsic_w2c).astype(np.float32)

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=impath,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),   # c2w
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
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/BlendedMVS"
    out_dir = "output_debug/BlendedMVS/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = BlendedMVSDataset(
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

    visualize_gt(views, out_dir, prefix="BlendedMVS_GT")