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
from pathlib import Path
import re
import json
import math
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import rdepth_to_zdepth, threshold_depth_map
from datasets.base.utils import read_any_file, parse_colmap_to_dict

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


class ScanNetDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        sample_method=None,
        min_num_images=24,
        load_segment=False,
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "ScanNet"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.sample_method = sample_method
        self.min_num_images = min_num_images
        self.load_segment = load_segment

        self.SCANNET_DIR = data_root
        self.trans_device_from_camera = get_trans_device_from_camera()

        #
        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # 1) load/save cache
        cache_dir = "/efs-mi-east4-2/z00533370/proj_3dgs/3drecon/data/dataset_cache"
        os.makedirs(cache_dir, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")
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
        sequence_list_path = f"{self.SCANNET_DIR}/sequence_list.pkl"
        if osp.exists(sequence_list_path):
            with open(sequence_list_path, "rb") as f:
                sequence_list, data_store = pickle.load(f)
        else:
            data_store = {}
            sequence_list = []
            subdirs = os.listdir(self.SCANNET_DIR)
            scenes = sorted([osp.join(subdir, d) for subdir in subdirs for d in os.listdir(osp.join(self.SCANNET_DIR, subdir)) if osp.isdir(osp.join(self.SCANNET_DIR, subdir, d))])
            for scene in tqdm(scenes, desc="Building Scannet cache"):
                scene_path = osp.join(self.SCANNET_DIR, scene)
                color_dir = osp.join(scene_path, "color")
                depth_dir = osp.join(scene_path, "depth")
                segment_dir = osp.join(scene_path, "label")
                cam_dir = osp.join(scene_path, "cam")
                pose_dir = osp.join(scene_path, "pose")
                intrinsic_path = osp.join(scene_path, "intrinsics", "intrinsic_depth.txt")
                use_cam = osp.exists(cam_dir)
                if not (osp.exists(color_dir) and osp.exists(depth_dir)):
                    continue

                def sort_by_number(file_name):
                    return int(file_name.split(".")[0])

                if use_cam:
                    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg")], key=sort_by_number)
                    cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(".npz")], key=sort_by_number)
                    if len(color_files) < self.min_num_images or len(cam_files) < self.min_num_images:
                        continue
                    frame_metadata = []
                    for fname in color_files:
                        frame_id = osp.splitext(fname)[0]
                        color_path = osp.join(color_dir, fname)
                        depth_path = osp.join(depth_dir, f"{frame_id}.png")
                        segment_path = osp.join(segment_dir, f"{frame_id}.png")
                        cam_path = osp.join(cam_dir, f"{int(frame_id):05d}.npz")
                        if not (
                            osp.exists(color_path)
                            and osp.exists(depth_path)
                            and osp.exists(segment_path)
                            and osp.exists(cam_path)
                        ):
                            continue
                        frame_metadata.append(
                            {
                                "color": osp.join(*color_path.split(os.sep)[-4:]),
                                "depth": osp.join(*depth_path.split(os.sep)[-4:]),
                                "segment": osp.join(*segment_path.split(os.sep)[-4:]),
                                "cam": osp.join(*cam_path.split(os.sep)[-4:]),
                                "frame_id": osp.join(*frame_id.split(os.sep)[-4:]),
                                "format": "npz",
                            }
                        )
                    if len(frame_metadata) >= self.min_num_images:
                        data_store[scene] = frame_metadata
                        sequence_list.append(scene)
                else:
                    if not (osp.exists(pose_dir) and osp.exists(intrinsic_path)):
                        continue
                    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".jpg")], key=sort_by_number)
                    if len(color_files) < self.min_num_images:
                        continue
                    frame_metadata = []
                    for fname in color_files:
                        frame_id = osp.splitext(fname)[0]
                        color_path = osp.join(color_dir, fname)
                        depth_path = osp.join(depth_dir, f"{frame_id}.png")
                        segment_path = osp.join(segment_dir, f"{frame_id}.png")
                        pose_path = osp.join(pose_dir, f"{frame_id}.txt")
                        if not (
                            osp.exists(color_path)
                            and osp.exists(depth_path)
                            # and osp.exists(segment_path)
                            and osp.exists(pose_path)
                        ):
                            continue
                        # filter out images without pose annotation
                        pose_anno = np.loadtxt(open(pose_path))
                        if np.isnan(pose_anno).any() or np.isinf(pose_anno).any():
                            continue
                        frame_metadata.append(
                            {
                                "color": osp.join(*color_path.split(os.sep)[-4:]),
                                "depth": osp.join(*depth_path.split(os.sep)[-4:]),
                                "segment": osp.join(*segment_path.split(os.sep)[-4:]),
                                "pose": osp.join(*pose_path.split(os.sep)[-4:]),
                                "frame_id": osp.join(*frame_id.split(os.sep)[-4:]),
                                "intrinsic": osp.join(*intrinsic_path.split(os.sep)[-4:]),
                                "format": "raw",
                            }
                        )
                    if len(frame_metadata) >= self.min_num_images:
                        data_store[scene] = frame_metadata
                        sequence_list.append(scene)

                with open(sequence_list_path, "wb") as f:
                    pickle.dump((sequence_list, data_store), f)

        if self.load_segment:
            self.classes = 40 + 1
            with open(self.SCANNET_DIR + "/scannetv2-labels.combined.tsv") as _file:
                df = pd.read_csv(_file, delimiter="\t")
                id_to_nyu40id = dict(zip(df["id"], df["nyu40id"]))
                id_to_nyu40id[0] = 0
                self.reid = np.vectorize(lambda id_: id_to_nyu40id.get(id_, -1))
        
        all_sequences = []
        scene_to_path = {}
        scene_meta = {}
        
        for scene_name in sequence_list:
            all_sequences.append(scene_name)
            scene_to_path[scene_name] = osp.join(self.SCANNET_DIR, scene_name)
            frame_meta = data_store[scene_name]
            scene_meta[scene_name] = {
                "frames_id": list(range(len(frame_meta))),
                "meta_info": frame_meta
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
    # re-implement in basedataset

    # =========================
    # load one view
    # =========================
    def _load_one_view(self, scene_path, fid, frame, resolution, rng):
        
        # rgb_image = read_any_file(osp.join(self.SCANNET_DIR, frame["color"]))
        rgb_image = np.array(Image.open(osp.join(self.SCANNET_DIR, frame["color"])).resize((640, 480), resample=lanczos))
        # rgb_image = np.array(Image.open(osp.join(self.SCANNET_DIR, frame["color"])).resize((600, 400), resample=lanczos))
        depthmap = read_any_file(osp.join(self.SCANNET_DIR, frame["depth"]))

        depthmap = depthmap * 0.001
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        if self.load_segment:
            segment_map = read_any_file(osp.join(self.SCANNET_DIR, frame["segment"]))
            segment_map = self.reid(segment_map)
        else:
            segment_map = None


        if frame.get("format") == "npz":
            with open(osp.join(self.SCANNET_DIR, frame["cam"]), "rb") as f:
                cam_data = np.load(f)
            intrinsic = cam_data["intrinsics"][:3, :3]
            extrinsic = cam_data["pose"][:3, :]
        else:
            with open(osp.join(self.SCANNET_DIR, frame["intrinsic"])) as f:
                intrinsic = np.loadtxt(f)[:3, :3]
            with open(osp.join(self.SCANNET_DIR, frame["pose"])) as f:
                extrinsic = np.loadtxt(f)[:3, :]

        # extri_opencv is cam-to-world, transform it to world-to-cam
        extrinsic[:3, :3] = extrinsic[:3, :3].transpose(-1, -2)
        extrinsic[:3, 3:] = -np.matmul(extrinsic[:3, :3], extrinsic[:3, 3:])
        extrinsic = np.concatenate([extrinsic, np.array([[0,0,0,1]], dtype=extrinsic.dtype)], axis=0)
        extrinsic = np.linalg.inv(extrinsic)

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution=(512,336),
            rng=rng,
            info=osp.join(self.SCANNET_DIR, frame["color"]),
        )

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
        scene= self.sequences[index]
        scene_path = self.scene_to_path[scene]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frames_id"], rng)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        views = []
        for fid in idxs:
            frame = meta["meta_info"][fid]
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    fid=fid,
                    frame=frame,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views

if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt
    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/ScanNet"
    out_dir = "output_debug/ScanNet/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = ScanNetDataset(
        data_root=data_root,
        mode="train",
        resolution=[[518, 336]],
        frame_num=8,
        max_distance=24,
        split_ratio=0.9,
        random_sample_thres=0.05,
        stride_sample_thres=0.30,
        stride_candidates=[20, 22, 25, 28, 30],
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

    """Visualize gt data. """
    visualize_gt(views, out_dir, prefix="ScanNet_GT")