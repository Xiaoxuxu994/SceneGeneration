import sys
sys.path.append('.')

import os
import os.path as osp
from glob import glob
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

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class ReplicaDataset(BaseDataset):
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

        self.dataset_label = "Replica"
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
        """
        Replica directory example:
            data_root/
                cam_params.json
                room0/
                    traj.txt
                    results/
                        frame000000.jpg
                        depth000000.png
                room1/
                    ...
        """
        cam_params_path = osp.join(self.data_root, "cam_params.json")
        if not osp.isfile(cam_params_path):
            raise FileNotFoundError(f"Missing shared camera params: {cam_params_path}")

        with open(cam_params_path, "r") as f:
            cam_params = json.load(f)

        camera_intrinsic = np.array(
            [
                cam_params["camera"]["fx"],
                cam_params["camera"]["fy"],
                cam_params["camera"]["cx"],
                cam_params["camera"]["cy"],
            ],
            dtype=np.float32,
        )

        scene_names = [
            x for x in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, x))
        ]
        scene_names = sorted(scene_names)

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        for scene in tqdm(scene_names, desc="Building Replica cache"):
            scene_path = osp.join(self.data_root, scene)
            results_dir = osp.join(scene_path, "results")
            traj_file = osp.join(scene_path, "traj.txt")

            if not (osp.isdir(results_dir) and osp.isfile(traj_file)):
                continue

            try:
                camera_parameters = np.loadtxt(traj_file, delimiter=" ", skiprows=0).astype(np.float32)
            except Exception as e:
                print(f"[Replica] Skip {scene}, failed to load traj: {e}")
                continue

            if camera_parameters.ndim == 1:
                camera_parameters = camera_parameters[None, :]

            if camera_parameters.shape[-1] != 16:
                print(f"[Replica] Skip {scene}, traj shape invalid: {camera_parameters.shape}")
                continue

            rgb_ids = {
                osp.splitext(osp.basename(x))[0].replace("frame", "")
                for x in glob(osp.join(results_dir, "frame*.jpg"))
            }
            depth_ids = {
                osp.splitext(osp.basename(x))[0].replace("depth", "")
                for x in glob(osp.join(results_dir, "depth*.png"))
            }

            frame_ids = sorted(rgb_ids & depth_ids, key=int)
            frame_ids = [
                fid for fid in frame_ids
                if fid.isdigit() and int(fid) < len(camera_parameters)
            ]

            if len(frame_ids) < self.min_num_images:
                continue

            all_sequences.append(scene)
            scene_to_path[scene] = scene_path
            scene_meta[scene] = {
                "frame_ids": frame_ids,
                "camera_parameters": camera_parameters,   # each row is 4x4 c2w
                "camera_intrinsic": camera_intrinsic,     # [fx, fy, cx, cy]
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
    # camera helpers
    # =========================
    def _build_intrinsic(self, camera_intrinsic):
        fx, fy, cx, cy = camera_intrinsic
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

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
    def _load_one_view(self, scene_path, scene_name, fid, camera_parameters, camera_intrinsic, resolution, rng):
        image_idx = int(fid)

        image_filepath = osp.join(scene_path, "results", f"frame{image_idx:06d}.jpg")
        depth_filepath = osp.join(scene_path, "results", f"depth{image_idx:06d}.png")

        rgb_image = Image.open(image_filepath).convert("RGB")
        depthmap = np.array(Image.open(depth_filepath)).astype(np.float32)

        # Replica depth conversion
        depthmap = depthmap / 6553.5
        depthmap = threshold_depth_map(
            depthmap,
            min_percentile=-1,
            max_percentile=98,
        )

        intrinsic = self._build_intrinsic(camera_intrinsic)

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=image_filepath,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        camera_pose = camera_parameters[image_idx].reshape(4, 4).astype(np.float32)

        return dict(
            img=rgb_image,
            depthmap=depthmap,
            camera_pose=camera_pose.astype(np.float32),   # c2w
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=scene_name,
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
                    scene_name=scene,
                    fid=fid,
                    camera_parameters=meta["camera_parameters"],
                    camera_intrinsic=meta["camera_intrinsic"],
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/efs-m3ga/z00533370/reliev3r_data/Replica"
    out_dir = "output_debug/Replica/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = ReplicaDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=24,
        split_ratio=0.9,
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

    visualize_gt(views, out_dir, prefix="Replica_GT")