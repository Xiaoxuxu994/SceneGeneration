import sys
sys.path.append('.')

import os
import os.path as osp
import hashlib

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as SciR

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map


class TartanGroundDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,           
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        use_diff=True,
        use_omni=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "TartanGround"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.use_diff = use_diff
        self.use_omni = use_omni

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
        '''
            data_root
                └── scene_name
                        ├── Data_diff
                        │     └── Pxxxx
                        └── Data_omni
                            └── Pxxxx
        '''
        # camera_modes = [
        #     "lcam_front", "lcam_back", "lcam_left", "lcam_right", "lcam_top", "lcam_bottom",
        #     "rcam_front", "rcam_back", "rcam_left", "rcam_right", "rcam_top", "rcam_bottom",
        # ]

        camera_modes = [
            "lcam_front", "lcam_back", "lcam_left", "lcam_right", 
            "rcam_front", "rcam_back", "rcam_left", "rcam_right",
        ]

        traj_roots = []
        if self.use_diff:
            traj_roots.append("Data_diff")
        if self.use_omni:
            traj_roots.append("Data_omni")

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        scene_names = sorted([
            x for x in os.listdir(self.data_root)
            if osp.isdir(osp.join(self.data_root, x))
        ])

        for scene_name in tqdm(scene_names, desc="Building TartanGround cache"):
            # if scene_name != 'AbandonedCable':
            #     continue
            scene_root = osp.join(self.data_root, scene_name)

            for traj_root in traj_roots:

                # if traj_root != "Data_diff":
                #     continue

                traj_root_path = osp.join(scene_root, traj_root)
                if not osp.isdir(traj_root_path):
                    continue

                traj_names = sorted([
                    x for x in os.listdir(traj_root_path)
                    if osp.isdir(osp.join(traj_root_path, x))
                ])

                for traj_name in traj_names:
                    # if traj_name != "P1000":
                    #     continue
                    traj_path = osp.join(traj_root_path, traj_name)

                    for camera in camera_modes:
                        # if camera != "lcam_front":
                        #     continue
                        rgb_dir = osp.join(traj_path, f"image_{camera}")
                        depth_dir = osp.join(traj_path, f"depth_{camera}")
                        pose_path = osp.join(traj_path, f"pose_{camera}.txt")

                        if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir) and osp.isfile(pose_path)):
                            continue

                        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
                        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".png")])

                        rgb_ids = {f.split("_", 1)[0] for f in rgb_files}
                        depth_ids = {f.split("_", 1)[0] for f in depth_files}
                        frame_ids = sorted(list(rgb_ids & depth_ids), key=int)

                        if len(frame_ids) < self.min_num_images:
                            continue

                        try:
                            poses = np.loadtxt(pose_path).astype(np.float32)
                            poses = np.atleast_2d(poses)
                        except Exception as e:
                            print(f"[TartanGround] Skip {scene_name}/{traj_root}/{traj_name}/{camera}, pose load failed: {e}")
                            continue

                        intrinsic = np.array(
                            [[320.0, 0.0, 320.0],
                            [0.0, 320.0, 320.0],
                            [0.0,   0.0,   1.0]],
                            dtype=np.float32,
                        )

                        valid_frame_ids = []
                        camera_info = {}
                        file_info = {}

                        for fid in frame_ids:
                            idx = int(fid)
                            if idx >= len(poses):
                                continue

                            img_path = osp.join(rgb_dir, f"{fid}_{camera}.png")
                            depth_path = osp.join(depth_dir, f"{fid}_{camera}_depth.png")

                            if not (osp.isfile(img_path) and osp.isfile(depth_path)):
                                continue

                            camera_pose = self._convert_pose_to_camera_pose(poses[idx])

                            valid_frame_ids.append(fid)
                            camera_info[fid] = {
                                "camera_pose": camera_pose.astype(np.float32),
                                "camera_intrinsics": intrinsic.astype(np.float32),
                            }
                            file_info[fid] = {
                                "img_path": img_path,
                                "depth_path": depth_path,
                            }

                        if len(valid_frame_ids) < self.min_num_images:
                            continue

                        seq_name = f"{scene_name}/{traj_root}/{traj_name}/{camera}"

                        all_sequences.append(seq_name)
                        scene_to_path[seq_name] = traj_path
                        scene_meta[seq_name] = {
                            "frame_ids": valid_frame_ids,
                            "camera_info": camera_info,
                            "file_info": file_info,
                        }

        self.all_sequences = sorted(all_sequences)
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

        print(f"[TartanGround] Built {len(self.all_sequences)} sequences")
        if len(self.all_sequences) > 0:
            print(f"[TartanGround] First few sequences: {self.all_sequences[:5]}")

    def _convert_pose_to_camera_pose(self, pose_row):
        """
        Follow the old reference implementation exactly:

        source pose row:
            [tx, ty, tz, qx, qy, qz, qw]

        reference:
            c2w = ...
            w2c = inv(c2w)
            w2c = w2c[[1, 2, 0, 3]][:3]
            extri_opencv = w2c

        Here we convert back into 4x4 c2w for current pipeline.
        """
        position = pose_row[:3]
        quaternion = pose_row[3:7]

        rot = SciR.from_quat(quaternion)
        c2w_src = np.eye(4, dtype=np.float32)
        c2w_src[:3, :3] = rot.as_matrix().astype(np.float32)
        c2w_src[:3, 3] = position.astype(np.float32)

        w2c_src = np.linalg.inv(c2w_src)

        w2c_cv = np.eye(4, dtype=np.float32)
        w2c_cv[:3, :] = w2c_src[[1, 2, 0, 3]][:3]

        camera_pose = np.linalg.inv(w2c_cv).astype(np.float32)
        return camera_pose

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
    def _read_rgb(self, path):
        return Image.open(path).convert("RGB")

    def _read_depth(self, path):
        """
        Follow old implementation:
            depth_map = depth[..., [2,1,0,3]].view(np.float32).squeeze(-1)
        """
        depth_rgba = np.array(Image.open(path))
        depth_rgba = np.asarray(depth_rgba)

        assert depth_rgba.ndim == 3 and depth_rgba.shape[2] == 4, \
            f"Unexpected depth png shape {depth_rgba.shape} @ {path}"

        depth = np.ascontiguousarray(depth_rgba[..., [2, 1, 0, 3]]).view(np.float32).squeeze(-1)
        depth = depth.astype(np.float32)

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth[(depth <= 0)] = 0.0
        depth = threshold_depth_map(depth, max_percentile=98)
        return depth.astype(np.float32)

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_name, fid, cam_info, file_info, resolution, rng):
        img_path = file_info["img_path"]
        depth_path = file_info["depth_path"]

        rgb_image = self._read_rgb(img_path)
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
            camera_pose=cam_info["camera_pose"].astype(np.float32),
            camera_intrinsics=intrinsics_.astype(np.float32),
            dataset=self.dataset_label,
            label=scene_name.replace("/", "_"),
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

    data_root = "/efs-m3ga/z00533370/reliev3r_data/TartanGround/"
    out_dir = "output_debug/TartanGround/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = TartanGroundDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=24,
        min_num_images=24,
        split_ratio=1.0,
        z_far=80,
        verbose=False,
        shuffle=False,
        use_diff=True,
        use_omni=True,
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

    visualize_gt(views, out_dir, prefix="TartanGround_GT")