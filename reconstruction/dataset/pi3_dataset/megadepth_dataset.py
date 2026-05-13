import sys
sys.path.append('.')

import os
import os.path as osp
from glob import glob

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import hashlib
from collections import defaultdict
import pickle


FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .base.base_dataset import BaseDataset, CACHE_DIR
from .base.transforms import *
from .sample_utils.common import rdepth_to_zdepth, threshold_depth_map
from .base.utils import read_any_file, parse_colmap_to_dict

def pos_quat2SE(quat_data):
    from scipy.spatial.transform import Rotation as R

    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.eye(4, dtype=np.float32)
    SE[:3, :3] = SO.astype(np.float32)
    SE[:3, 3] = np.asarray(quat_data[:3], dtype=np.float32)
    return SE


def get_trans_device_from_camera():
    trans_device_from_camera = np.array(
        [[ 0.99606003, -0.04388682,  0.07706079, -0.00753010],
         [ 0.08210934,  0.78468796, -0.61442889, -0.01090855],
         [-0.03350334,  0.61833547,  0.78519983, -0.00359806],
         [ 0.        ,  0.        ,  0.        ,  1.        ]],
        dtype=np.float32
    )
    rot90 = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]], dtype=np.float32)
    trans_device_from_camera[:3, :3] = trans_device_from_camera[:3, :3] @ rot90
    return trans_device_from_camera


class MegaDepthDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        sample_method=None,
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "MegaDepth"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.sample_method = sample_method

        self.MEGADEPTH_DIR = data_root + "/Undistorted_SfM"
        self.MEGADEPTH_SPLIT_DIR = data_root + "/final_list"
        self.trans_device_from_camera = get_trans_device_from_camera()

        #
        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # 1) load/save cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(CACHE_DIR, f"{self.dataset_label}_{root_hash}_cache.pkl")
        if not os.path.exists(cache_path):
            print(f'[Dataset] Constructing cache {cache_path}..')
            self._scan_scenes()
            self._save_cache(keys=["all_sequences", "scene_to_path", "scene_meta"],save_path=cache_path)
            print(f'[Dataset] Cache is saved {cache_path}..')
        else:
            print(f'[Dataset] Loading cache from {cache_path}..')
            res = self._load_cache(cache_path)
            if res:
                self.cache_loaded = True
                print(f'[Dataset] Cache is loaded.')

        # 2) train/val split
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
        sequence_list = defaultdict(list)
        sequence_list_path = f"{self.MEGADEPTH_SPLIT_DIR}/sequence_list.pkl"
        if osp.exists(sequence_list_path):
            with open(sequence_list_path, "rb") as f:
                sequence_list = pickle.load(f)
        else:
            split_name = "train_val" if self.mode == "train" else "test"
            orientations = ["landscape", "portrait"]
            for orientation in orientations:
                split_path = osp.join(
                    self.MEGADEPTH_SPLIT_DIR,
                    f"{split_name}_list",
                    orientation,
                    "imgs_MD.p",
                )
                with open(split_path, "rb") as f:
                    img_paths = pickle.load(f)
                for img_path in img_paths:
                    scene, dense_num, _, img_name = img_path.split("/")
                    img_name = osp.splitext(img_name)[0]

                    # checking if image path exists
                    img_path = osp.join(self.MEGADEPTH_DIR, scene, "images", f"{img_name}.jpg")
                    depth_path = osp.join(self.MEGADEPTH_DIR, scene, "depths", f"{img_name}.h5")

                    if not osp.exists(img_path) or not osp.exists(depth_path):
                        continue

                    sequence_list[scene].append(img_name)

            with open(sequence_list_path, "wb") as f:
                pickle.dump(sequence_list, f)

        all_sequences = []
        scene_to_path = {}
        scene_meta = {}
        for scene_name, sequence in sequence_list.items():
            if len(sequence) >= self.min_num_images:
                all_sequences.append(scene_name)
                scene_to_path[scene_name] = osp.join(self.MEGADEPTH_DIR, scene_name)
                scene_meta[scene_name] = {
                    "frames_id": list(range(len(sequence))),
                    "frames_name": sequence
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
    def _build_intrinsic(self, H, W):
        return np.array(
            [
                [0.42278125 * W, 0.0, 0.5 * W],
                [0.0, 0.42278125 * H, 0.5 * H],
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
    def _load_one_view(self, scene_path, frames_name, fid, intri_extri_dict, resolution, rng):
        image_name = frames_name[fid]
        impath = osp.join(scene_path, "images", f"{image_name}.jpg")
        depthpath = osp.join(scene_path, "depths", f"{image_name}.h5")

        rgb_image = read_any_file(impath)
        depthmap = read_any_file(depthpath)
        # print(rgb_image.shape, depthmap.shape)
        intrinsic, extrinsic = intri_extri_dict[image_name]
        extrinsic = np.concatenate([extrinsic, np.array([[0,0,0,1]], dtype=extrinsic.dtype)], axis=0)
        extrinsic = np.linalg.inv(extrinsic)
        
        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=impath,
        )
        
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)
        # print(intrinsic_)
        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=extrinsic.astype(np.float32),
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

        if self.sample_method is None:
            idxs = self._sample_frame_ids(meta["frames_id"], rng)
        elif self.sample_method=='pose_sim':
            with open(osp.join(self.MEGADEPTH_DIR, "../covisibility", f"{scene}.pkl"), "rb") as f:
                ranking, dists = pickle.load(f)
            random_id = np.random.randint(0, ranking.shape[0])
            idxs = ranking[random_id, :self.frame_num]
            np.random.shuffle(idxs)
        else:
            raise NotImplementedError("Not implemented sample method '{}'".format(self.sample_method))

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        # Loading extriniscs and intrinsics data
        images_txt_path = osp.join(scene_path, "sparse-txt", "images.txt")
        cameras_txt_path = osp.join(scene_path, "sparse-txt", "cameras.txt")
        intri_extri_dict = parse_colmap_to_dict(images_txt_path, cameras_txt_path)

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    frames_name=meta["frames_name"],
                    fid=fid,
                    intri_extri_dict=intri_extri_dict,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views

if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt
    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/MegaDepth"
    out_dir = "output_debug/MegaDepth/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = MegaDepthDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=24,
        max_distance=8,
        split_ratio=0.9,
        verbose=False,
        shuffle=False,
    )

    print(f"len(dataset) = {len(dataset)}")

    views = dataset[1]
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

    """Visualize gt data. """
    visualize_gt(views, out_dir, prefix="MegaDepth_GT")