import sys
sys.path.append(".")

import os
import os.path as osp
from glob import glob
import hashlib

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


# ============================================================
# Hard-coded settings
# ============================================================
CAMERA_YAML_NAME = "camera.yaml"
TRJ_NAME = "trj_0.txt"
YAW_DIR_NAME = "yaw_amplitude_0522"

# trj_0.txt is assumed to contain one OpenCV c2w 4x4 matrix per simulator step.
# Frame id is used directly as trajectory line index:
#   00000000 -> line 0
#   00000004 -> line 4
#   00000099 -> line 99


# ============================================================
# Camera / trajectory helpers
# ============================================================
def _load_yaml_file(path):
    if yaml is None:
        raise ImportError(
            "PyYAML is required to read camera.yaml. "
            "Please install it with: pip install pyyaml"
        )
    with fio.open(path, "r") as f:
        return yaml.safe_load(f)


def load_camera_yaml(camera_yaml_path):
    """
    Load OpenCV-style camera.yaml.

    Returns:
        K: np.ndarray [3, 3]
        image_size: tuple[int, int] = (W, H)
    """
    data = _load_yaml_file(camera_yaml_path)

    K = np.asarray(data["K"]["data"], dtype=np.float32).reshape(3, 3)
    image_size = data.get("Imagesize", [1280, 720])
    base_W = int(image_size[0])
    base_H = int(image_size[1])

    return K.astype(np.float32), (base_W, base_H)


def load_trj_poses(trj_path):
    """
    Load trj_0.txt.

    Each non-empty line contains 16 floats, row-major 4x4 matrix.
    The matrix is assumed to be OpenCV camera-to-world c2w directly.
    """
    poses = []

    with fio.open(trj_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            vals = [float(x) for x in line.split()]
            if len(vals) != 16:
                raise ValueError(
                    f"Bad trajectory line {line_idx} in {trj_path}: "
                    f"expected 16 floats, got {len(vals)}"
                )

            pose = np.asarray(vals, dtype=np.float32).reshape(4, 4)
            poses.append(pose.astype(np.float32))

    if len(poses) == 0:
        raise ValueError(f"Empty trajectory file: {trj_path}")

    return poses


# ============================================================
# Dataset
# ============================================================
class SAGE10KDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=100,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=8,
        **kwargs,
    ):
        """
        Expected folder structure:

            data_root/
              layout_000f8048/
                yaw_amplitude_0522/    # hard-coded
                  route_1/
                    camera.yaml
                    trj_0.txt
                    vis/
                      color/
                        00000000_color_vis.png
                        00000001_color_vis.png
                        ...
                      depth/
                        00000000_depth.tif
                        00000001_depth.tif
                        ...
                  route_2/
                    ...

        Intrinsics:
            loaded from route_xxx/camera.yaml.

        Extrinsics:
            loaded from route_xxx/trj_0.txt.
            Default assumption: each line is OpenCV camera-to-world c2w.
            Frame id is used directly as trajectory line index:
                00000000 -> line 0
                00000004 -> line 4
                00000099 -> line 99

        Cache:
            scene_meta only stores frame_ids. Camera intrinsics and poses are
            loaded from camera.yaml / trj_0.txt at runtime, same as the minimal
            MayaIndoor trj dataset style.
        """
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "SAGE10K"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)
        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = osp.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")

        cache_path = self.cache_file or cache_path

        if not fio.exists(cache_path):
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
                print("[Dataset] Cache is loaded.")

        self.sequences = self._split_sequences(self.all_sequences)

        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]

        if self.verbose:
            print(f"[{self.dataset_label}] First sequences:")
            for s in self.sequences[:20]:
                print(" ", s, "->", self.scene_to_path[s])

        print(
            f"[{self.dataset_label}] Found {len(self.sequences)} routes "
            f"in {self.data_root} ({self.mode})",
            flush=True,
        )

    def __len__(self):
        return len(self.sequences)

    # ============================================================
    # path helpers
    # ============================================================
    def _is_valid_route(self, route_path):
        color_dir = osp.join(route_path, "vis", "color")
        depth_dir = osp.join(route_path, "vis", "depth")
        camera_yaml_path = osp.join(route_path, CAMERA_YAML_NAME)
        trj_path = osp.join(route_path, TRJ_NAME)
        return (
            fio.isdir(color_dir)
            and fio.isdir(depth_dir)
            and fio.isfile(camera_yaml_path)
            and fio.isfile(trj_path)
        )

    def _make_sequence_name(self, route_path):
        rel = osp.relpath(route_path, self.data_root)
        rel = rel.replace("\\", "/")
        return rel.replace("/", "__")

    def _parse_frame_id(self, color_path):
        """
        Example:
            00000000_color_vis.png -> 00000000
            00000097_color_vis.png -> 00000097
        """
        name = osp.basename(color_path)
        return name.split("_")[0]

    def _get_color_path(self, route_path, fid):
        return osp.join(route_path, "vis", "color", f"{fid}_color_vis.png")

    def _find_depth_path(self, route_path, fid):
        candidates = [
            osp.join(route_path, "vis", "depth", f"{fid}_depth.tif"),
            osp.join(route_path, "vis", "depth", f"{fid}_depth.tiff"),
            osp.join(route_path, "vis", "depth", f"{fid}_depth.png"),
            osp.join(route_path, "vis", "depth", f"{fid}_depth_vis.png"),
        ]
        for p in candidates:
            if fio.isfile(p):
                return p
        return None

    def _get_camera_yaml_path(self, route_path):
        return osp.join(route_path, CAMERA_YAML_NAME)

    def _get_trj_path(self, route_path):
        return osp.join(route_path, TRJ_NAME)

    @staticmethod
    def _route_sort_key(name):
        try:
            return int(name.split("_")[-1])
        except Exception:
            return name

    # ============================================================
    # scan / cache
    # ============================================================
    def _scan_scenes(self):
        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        layout_names = sorted(
            [
                x for x in fio.listdir(self.data_root)
                if fio.isdir(osp.join(self.data_root, x)) and x.startswith("layout_")
            ]
        )

        print(f"[{self.dataset_label}] scanning layouts: {len(layout_names)}, yaw_dir={YAW_DIR_NAME}")

        for layout_name in tqdm(layout_names, desc=f"Building {self.dataset_label} cache"):
            layout_path = osp.join(self.data_root, layout_name)

            yaw_path = osp.join(layout_path, YAW_DIR_NAME)
            if not fio.isdir(yaw_path):
                continue

            route_names = sorted(
                [
                    x for x in fio.listdir(yaw_path)
                    if fio.isdir(osp.join(yaw_path, x)) and x.startswith("route_")
                ],
                key=self._route_sort_key,
            )

            for route_name in route_names:
                route_path = osp.join(yaw_path, route_name)

                if not self._is_valid_route(route_path):
                    continue

                trj_path = self._get_trj_path(route_path)

                try:
                    num_trj_poses = len(load_trj_poses(trj_path))
                except Exception as e:
                    print(f"[WARN] skip {route_path}, failed to load {TRJ_NAME}: {e}")
                    continue

                color_dir = osp.join(route_path, "vis", "color")
                color_files = sorted(fio.glob(osp.join(color_dir, "*_color_vis.png")))
                if len(color_files) == 0:
                    continue

                frame_ids = []
                for color_path in color_files:
                    fid = self._parse_frame_id(color_path)

                    if not fid.isdigit():
                        continue

                    depth_path = self._find_depth_path(route_path, fid)
                    if depth_path is None:
                        continue

                    # trj_0.txt is one matrix per original simulator step.
                    # Use frame id directly as trajectory line index.
                    pose_idx = int(fid)
                    if pose_idx >= num_trj_poses:
                        print(
                            f"[WARN] skip frame {fid} in {route_path}, "
                            f"pose_idx={pose_idx}, num_trj_poses={num_trj_poses}"
                        )
                        continue

                    frame_ids.append(fid)

                frame_ids = sorted(frame_ids, key=lambda x: int(x))

                if len(frame_ids) < self.min_num_images:
                    continue

                seq_name = self._make_sequence_name(route_path)
                all_sequences.append(seq_name)
                scene_to_path[seq_name] = route_path
                scene_meta[seq_name] = {
                    "frame_ids": frame_ids,
                }

                if self.verbose:
                    print(
                        f"[DEBUG trj mapping] route={route_path}, "
                        f"num_images={len(frame_ids)}, "
                        f"num_trj_poses={num_trj_poses}, "
                        f"index=fid, "
                        f"first_fid={frame_ids[0]}, last_fid={frame_ids[-1]}"
                    )

        self.all_sequences = sorted(all_sequences)
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

        print(
            f"[{self.dataset_label}] scanned {len(self.all_sequences)} valid routes "
            f"from {self.data_root}",
            flush=True,
        )

    # ============================================================
    # split
    # ============================================================
    def _split_sequences(self, all_sequences):
        if len(all_sequences) <= 1:
            return all_sequences

        split_idx = int(len(all_sequences) * self.split_ratio)

        if self.mode == "train":
            return all_sequences[:split_idx]

        return all_sequences[split_idx:]

    # ============================================================
    # camera / depth
    # ============================================================
    def _build_intrinsic_from_yaml(self, K_yaml, image_size, H, W):
        """
        Scale K from camera.yaml resolution to actual RGB image resolution.

        camera.yaml:
            Imagesize: [base_W, base_H]
        PIL:
            rgb_image.size -> (W, H)
        """
        base_W, base_H = image_size

        sx = float(W) / float(base_W)
        sy = float(H) / float(base_H)

        K = np.asarray(K_yaml, dtype=np.float32).copy()
        K[0, :] *= sx
        K[1, :] *= sy
        K[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return K.astype(np.float32)

    def _load_depth(self, depth_path):
        depth_img = fio.pil_open(depth_path)
        depth = np.array(depth_img)

        if depth.ndim == 3:
            depth = depth[..., :3].astype(np.float32).mean(axis=-1)
        else:
            depth = depth.astype(np.float32)

        # uint16 mm -> meter
        # depth = depth * 0.001
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[depth < 0] = 0.0

        return depth.astype(np.float32)

    # =========================
    # sample
    # =========================
    # re-implement in BaseDataset

    # ============================================================
    # load one view
    # ============================================================
    def _load_one_view(
        self,
        scene,
        fid,
        resolution,
        rng,
        K_yaml,
        image_size,
        trj_poses,
    ):
        route_path = self.scene_to_path[scene]

        color_path = self._get_color_path(route_path, fid)
        depth_path = self._find_depth_path(route_path, fid)

        if depth_path is None:
            raise FileNotFoundError(f"Depth not found for fid={fid}, route={route_path}")

        rgb_image = fio.pil_open(color_path).convert("RGB")
        depthmap = self._load_depth(depth_path)

        W0, H0 = rgb_image.size
        intrinsic = self._build_intrinsic_from_yaml(
            K_yaml,
            image_size,
            H0,
            W0,
        )

        # Default: trj_0.txt is OpenCV c2w. Use fid as trajectory line index.
        camera_pose = trj_poses[int(fid)]

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=color_path,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=scene,
            instance=str(fid),
        )

    # ============================================================
    # main interface
    # ============================================================
    def _get_views(self, index, resolution, rng):
        if self.mode == "train":
            index = index % len(self.sequences)

        scene = self.sequences[index]
        route_path = self.scene_to_path[scene]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frame_ids"], rng)

        camera_yaml_path = self._get_camera_yaml_path(route_path)
        trj_path = self._get_trj_path(route_path)
        K_yaml, image_size = load_camera_yaml(camera_yaml_path)
        trj_poses = load_trj_poses(trj_path)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
            "route_path": route_path,
            "camera_yaml_path": camera_yaml_path,
            "trj_path": trj_path,
            "trj_index_mode": "fid",
        }

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene=scene,
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                    K_yaml=K_yaml,
                    image_size=image_size,
                    trj_poses=trj_poses,
                )
            )

        return views


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.sage10k_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(SAGE10KDataset))
