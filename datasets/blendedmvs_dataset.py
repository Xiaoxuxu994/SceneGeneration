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

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


def _load_pose(path, ret_44=False):
    """
    Read BlendedMVS camera file.
    Format:
      - first 4x4: extrinsic, world2cam
      - then 3x3: intrinsic
    """
    with fio.open(path, "r") as f:
        RT = fio.loadtxt(f, skiprows=1, max_rows=4, dtype=np.float32)  # world2cam
        assert RT.shape == (4, 4), f"Invalid extrinsic shape in {path}: {RT.shape}"

        K = fio.loadtxt(f, skiprows=2, max_rows=3, dtype=np.float32)
        assert K.shape == (3, 3), f"Invalid intrinsic shape in {path}: {K.shape}"

    if ret_44:
        return K, RT
    return K, RT[:3, :]


def read_pfm(path):
    """
    Minimal PFM reader.
    """
    with fio.open(path, "rb") as f:
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
        data = fio.fromfile(f, endian + "f")

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
        dataset_label="BlendedMVS",
        scene_filter_file=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = dataset_label
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.scene_filter_file = scene_filter_file
        self.scene_filter_set = self._load_scene_filter_set(scene_filter_file)

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)
        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")
        cache_path = self.cache_file or cache_path
        if not fio.exists(cache_path):
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

        # Apply optional scene filtering before train/test split.
        # This keeps the split ratio meaningful for the filtered subset.
        self._apply_scene_filter()

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
    # optional scene filtering
    # =========================
    def _load_scene_filter_set(self, scene_filter_file):
        """
        Load a text file that contains one BlendedMVS scene id per line.

        Example lines:
            5a0d4d4...
            5a0d4d4...  # comments are allowed after whitespace

        Empty lines and lines starting with '#' are ignored.
        """
        if scene_filter_file is None or str(scene_filter_file).strip() == "":
            return None

        if isinstance(scene_filter_file, (list, tuple)):
            files = [str(x) for x in scene_filter_file]
        else:
            files = [str(scene_filter_file)]

        keep = set()
        for path in files:
            if not fio.exists(path):
                raise FileNotFoundError(f"scene_filter_file does not exist: {path}")

            with fio.open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Use the first whitespace-separated token as scene id.
                    scene_id = line.split()[0]
                    if scene_id:
                        keep.add(scene_id)

        if len(keep) == 0:
            raise ValueError(f"scene_filter_file is empty: {scene_filter_file}")

        return keep

    def _scene_id_from_sequence(self, sequence):
        """Return the top-level BlendedMVS scene id from a sequence key."""
        if isinstance(sequence, (tuple, list)):
            sequence = sequence[0]
        sequence = str(sequence)
        return sequence.replace("\\", "/").split("/")[0]

    def _apply_scene_filter(self):
        if self.scene_filter_set is None:
            return

        old_len = len(self.all_sequences)
        self.all_sequences = [
            s for s in self.all_sequences
            if self._scene_id_from_sequence(s) in self.scene_filter_set
        ]

        print(
            f"[{self.dataset_label}] Applied scene_filter_file={self.scene_filter_file}: "
            f"{old_len} -> {len(self.all_sequences)} scenes "
            f"(keep_list={len(self.scene_filter_set)})",
            flush=True,
        )

        if len(self.all_sequences) == 0:
            raise RuntimeError(
                f"[{self.dataset_label}] scene filter removed all scenes. "
                f"Please check scene ids in {self.scene_filter_file} and data_root={self.data_root}."
            )

    # =========================
    # scan / cache
    # =========================
    def _scan_scenes(self):
        scene_names = [
            x for x in fio.listdir(self.data_root)
            if fio.isdir(osp.join(self.data_root, x))
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

            if not (fio.isdir(rgb_dir) and fio.isdir(depth_dir) and fio.isdir(cam_dir)):
                continue

            rgb_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in fio.glob(osp.join(rgb_dir, "*.jpg"))
                if not osp.splitext(osp.basename(x))[0].endswith("_masked")
            }
            depth_ids = {
                osp.splitext(osp.basename(x))[0]
                for x in fio.glob(osp.join(depth_dir, "*.pfm"))
            }
            cam_ids = {
                osp.basename(x).replace("_cam.txt", "")
                for x in fio.glob(osp.join(cam_dir, "*_cam.txt"))
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
    # re-implement in basedataset

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, fid, resolution, rng):
        impath = osp.join(scene_path, "blended_images", f"{fid}.jpg")
        depthpath = osp.join(scene_path, "rendered_depth_maps", f"{fid}.pfm")
        campath = osp.join(scene_path, "cams", f"{fid}_cam.txt")

        rgb_image = fio.pil_open(impath).convert("RGB")
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


class BlendedMVSObjectDataset(BlendedMVSDataset):
    """
    BlendedMVS subset for DTU-like object/sculpture scenes.

    Pass scene_filter_file pointing to blendedmvs_small_sculpture.txt.
    The cache is still built from the same data_root; filtering is applied before
    train/test split, so this class can be used as a separate datamix entry.
    """
    def __init__(
        self,
        scene_filter_file=None,
        dataset_label="BlendedMVSObject",
        **kwargs,
    ):
        if scene_filter_file is None:
            raise ValueError(
                "BlendedMVSObjectDataset requires scene_filter_file, "
                "e.g. /path/to/blendedmvs_small_sculpture.txt"
            )
        super().__init__(
            scene_filter_file=scene_filter_file,
            dataset_label=dataset_label,
            **kwargs,
        )


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.blendedmvs_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(BlendedMVSDataset))
