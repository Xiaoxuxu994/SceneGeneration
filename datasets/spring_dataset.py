import sys
sys.path.append('.')

import os
import os.path as osp
import hashlib
from glob import glob

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *


class SpringDataset(BaseDataset):
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
            https://github.com/CUT3R/CUT3R/blob/main/datasets_preprocess/preprocess_spring.py
        '''
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "Spring"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images

        # fixed settings
        self.baseline = 0.065

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
        scene_names = [
            x for x in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, x))
        ]
        scene_names = sorted(scene_names)

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        frame_dirname = "frame_left"
        disp_dirname = "disp1_left"
        frame_prefix = "frame_left_"
        disp_prefix = "disp1_left_"

        for scene in tqdm(scene_names, desc="Building Spring cache"):
            scene_path = osp.join(self.data_root, scene)
            frame_dir = osp.join(scene_path, frame_dirname)
            disp_dir = osp.join(scene_path, disp_dirname)
            cam_dir = osp.join(scene_path, "cam_data")
            intr_path = osp.join(cam_dir, "intrinsics.txt")
            extr_path = osp.join(cam_dir, "extrinsics.txt")

            if not (
                osp.isdir(frame_dir)
                and osp.isdir(disp_dir)
                and osp.isfile(intr_path)
                and osp.isfile(extr_path)
            ):
                continue

            img_files = sorted(glob(osp.join(frame_dir, "*.png")))
            if len(img_files) < self.min_num_images:
                continue

            ordered_frame_ids = []
            valid_frame_ids = []

            for impath in img_files:
                img_name = osp.basename(impath)
                stem = osp.splitext(img_name)[0]  # frame_left_0001

                if not stem.startswith(frame_prefix):
                    continue

                fid = stem.replace(frame_prefix, "")
                if not fid.isdigit():
                    continue

                ordered_frame_ids.append(fid)

                disppath = osp.join(disp_dir, f"{disp_prefix}{fid}.dsp5")
                if osp.isfile(disppath):
                    valid_frame_ids.append(fid)

            if len(valid_frame_ids) < self.min_num_images:
                continue

            try:
                intr_rows = np.loadtxt(intr_path)
                extr_rows = np.loadtxt(extr_path)

                intr_rows = np.atleast_2d(intr_rows).astype(np.float32)
                extr_rows = np.atleast_2d(extr_rows).astype(np.float32)

                if len(intr_rows) < len(ordered_frame_ids):
                    raise ValueError(
                        f"intrinsics rows ({len(intr_rows)}) < frames ({len(ordered_frame_ids)})"
                    )
                if len(extr_rows) < len(ordered_frame_ids):
                    raise ValueError(
                        f"extrinsics rows ({len(extr_rows)}) < frames ({len(ordered_frame_ids)})"
                    )
            except Exception as e:
                print(f"[Spring] Skip scene {scene}, failed to parse camera files: {e}")
                continue

            valid_frame_ids_set = set(valid_frame_ids)
            camera_info = {}

            for i, fid in enumerate(ordered_frame_ids):
                if fid not in valid_frame_ids_set:
                    continue

                fx, fy, cx, cy = intr_rows[i][:4]
                K = np.array(
                    [
                        [fx, 0.0, cx],
                        [0.0, fy, cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )

                pose_raw = extr_rows[i][:16].reshape(4, 4).astype(np.float32)
                camera_pose = np.linalg.inv(pose_raw).astype(np.float32)

                camera_info[fid] = {
                    "camera_pose": camera_pose,
                    "camera_intrinsics": K,
                    "fx_baseline": float(fx) * self.baseline,
                }

            valid_frame_ids = [fid for fid in valid_frame_ids if fid in camera_info]
            if len(valid_frame_ids) < self.min_num_images:
                continue

            all_sequences.append(scene)
            scene_to_path[scene] = scene_path
            scene_meta[scene] = {
                "frame_ids": valid_frame_ids,
                "camera_info": camera_info,
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
    # re-implement in basedataset

    # =========================
    # disparity/depth helpers
    # =========================
    def _read_disparity_dsp5(self, path):
        with h5py.File(path, "r") as f:
            if "disparity" in f:
                disp = f["disparity"][()]
            else:
                keys = list(f.keys())
                if len(keys) == 0:
                    raise ValueError(f"No dataset found in dsp5 file: {path}")
                disp = f[keys[0]][()]

        disp = np.asarray(disp, dtype=np.float32)
        disp = np.squeeze(disp)
        disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)
        return disp

    def _prepare_disparity(self, disparity, rgb_h, rgb_w):
        disparity = np.asarray(disparity, dtype=np.float32)
        if disparity.ndim != 2:
            raise ValueError(f"Unexpected disparity ndim={disparity.ndim}, shape={disparity.shape}")

        dh, dw = disparity.shape

        # align CUT3R:
        # if GT disparity is 2x image resolution, downsample by [::2, ::2]
        if (dh, dw) == (rgb_h * 2, rgb_w * 2):
            disparity = disparity[::2, ::2]
        elif (dh, dw) == (rgb_h, rgb_w):
            pass
        else:
            raise ValueError(
                f"Unexpected disparity shape {(dh, dw)} for image shape {(rgb_h, rgb_w)}"
            )

        disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)
        return disparity.astype(np.float32)

    def _disparity_to_depth(self, disparity, fx_baseline):
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid = disparity != 0
        depth[valid] = fx_baseline / disparity[valid]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, scene_name, fid, cam_info, resolution, rng):
        frame_prefix = "frame_left_"
        disp_prefix = "disp1_left_"

        impath = osp.join(scene_path, "frame_left", f"{frame_prefix}{fid}.png")
        disppath = osp.join(scene_path, "disp1_left", f"{disp_prefix}{fid}.dsp5")

        rgb_image = Image.open(impath).convert("RGB")
        rgb_np = np.array(rgb_image)

        disparity = self._read_disparity_dsp5(disppath)
        disparity = self._prepare_disparity(
            disparity,
            rgb_h=rgb_np.shape[0],
            rgb_w=rgb_np.shape[1],
        )

        intrinsics = cam_info["camera_intrinsics"].copy()
        depthmap = self._disparity_to_depth(disparity, cam_info["fx_baseline"])

        if self.verbose:
            print("scene:", scene_name, "fid:", fid)
            print("instance:", f"{frame_prefix}{fid}.png")
            print("K:", intrinsics)
            print("camera_pose_t:", cam_info["camera_pose"][:3, 3])
            print("disp min/max:", disparity.min(), disparity.max())
            print("depth min/max:", depthmap.min(), depthmap.max())

        rgb_image, depthmap, intrinsics_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsics.copy(),
            resolution,
            rng=rng,
            info=impath,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        depthmap[depthmap < 0] = 0.0

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=cam_info["camera_pose"].astype(np.float32),
            camera_intrinsics=intrinsics_.astype(np.float32),
            dataset=self.dataset_label,
            label=scene_name,
            instance=f"{frame_prefix}{fid}.png",
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
                    cam_info=meta["camera_info"][fid],
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/efs-m3ga/z00533370/reliev3r_data/Spring_decode/spring/train/"
    out_dir = "output_debug/Spring/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = SpringDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=24,
        max_distance=8,
        min_num_images=24,
        split_ratio=0.9,
        z_far=120,
        verbose=False,
        shuffle=False,
    )

    print(f"len(dataset) = {len(dataset)}")

    test_idx = min(20, len(dataset) - 1)
    views = dataset[test_idx]
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

    visualize_gt(views, out_dir, prefix="Spring_GT")