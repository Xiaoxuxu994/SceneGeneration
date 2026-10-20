import sys
sys.path.append('.')

import os
import os.path as osp

import torch
import numpy as np
import hashlib
import json
import gzip
import glob
from tqdm import tqdm

SEEN_CATEGORIES = [
    "apple",
    "backpack",
    "ball", 
    "banana",
    "baseballbat",
    # "baseballglove",
    "bench",
    "bicycle",
    "book", 
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    "couch", 
    "cup",
    "donut",
    "frisbee", 
    "hairdryer",
    "handbag",
    "hotdog", 
    "hydrant",
    "keyboard",
    "kite", 
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    "remote",
    "sandwich",
    "skateboard",
    "stopsign",
    "suitcase",
    "teddybear",
    "toaster",
    "toilet",
    "toybus",
    "toyplane",
    "toytrain",
    "toytruck",
    "tv",
    "umbrella",
    "vase",
    "wineglass",
]

FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map
from datasets.base.utils import read_any_file

def opencv_from_cameras_projection(R, T, focal, p0, image_size):
    R = R[None, :, :]
    T = T[None, :]
    focal = focal[None, :]
    p0 = p0[None, :]
    image_size = image_size[None, :]

    R_pytorch3d = R.copy()
    T_pytorch3d = T.copy()
    focal_pytorch3d = focal
    p0_pytorch3d = p0
    T_pytorch3d[:, :2] *= -1
    R_pytorch3d[:, :, :2] *= -1
    tvec = T_pytorch3d
    R = R_pytorch3d.transpose(0, 2, 1)

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size[:, ::-1]

    # NDC to screen conversion.
    scale = np.min(image_size_wh, axis=1, keepdims=True) / 2.0
    scale = np.repeat(scale, 2, axis=1)
    c0 = image_size_wh / 2.0

    principal_point = -p0_pytorch3d * scale + c0
    focal_length = focal_pytorch3d * scale

    camera_matrix = np.zeros_like(R)
    camera_matrix[:, :2, 2] = principal_point
    camera_matrix[:, 2, 2] = 1.0
    camera_matrix[:, 0, 0] = focal_length[:, 0]
    camera_matrix[:, 1, 1] = focal_length[:, 1]
    return R[0], tvec[0], camera_matrix[0]

class CO3DV2Dataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        cam_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        sample_method=None,
        load_depth=True,
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "CO3DV2"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.sample_method = sample_method
        self.load_depth = load_depth
        self.CO3DV2_DIR = data_root
        #
        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        # 1) load/save cache
        cache_dir = self.cache_dir
        fio.makedirs(cache_dir)
        root_hash = hashlib.md5(fio.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = os.path.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")
        cache_path = self.cache_file or cache_path
        if not fio.exists(cache_path):
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
        anno = {}
        subdir = sorted(fio.listdir(self.CO3DV2_DIR)) 
        for subname in tqdm(subdir, desc="Building CO3DV2 cache"):
            obj, idx = subname.split('_')
            # anno["apple"]=anno_obj[(110_13048_23163, frame000199)]="viewpoint"
            if obj in SEEN_CATEGORIES and idx == "000": # first 000
                anno_path = osp.join(self.CO3DV2_DIR, subname, obj, "frame_annotations.jgz")
                anno_obj = {}
                with gzip.open(fio.open(anno_path, 'rb'), 'rt', encoding='utf-8') as f:
                    frame_annotations = json.load(f)
                for frame_anno in frame_annotations:
                    key = (frame_anno["sequence_name"], frame_anno["image"]["path"].split('/')[-1].split('.')[0])
                    anno_obj[key] = frame_anno["viewpoint"]
                anno[obj] = anno_obj
            if obj in SEEN_CATEGORIES and idx != "000" and obj in anno: # 001->00X
                subscene = fio.listdir(osp.join(self.CO3DV2_DIR, subname, obj))
                for scene in subscene:
                    depthmask_path = osp.join(self.CO3DV2_DIR, subname, obj, scene, 'depth_masks')
                    depth_path = osp.join(self.CO3DV2_DIR, subname, obj, scene, 'depths')
                    image_path = osp.join(self.CO3DV2_DIR, subname, obj, scene, 'images')
                    mask_path = osp.join(self.CO3DV2_DIR, subname, obj, scene, 'masks')
                    if fio.exists(depthmask_path)  and fio.exists(depth_path) and fio.exists(image_path) and fio.exists(mask_path):
                        file_list = fio.listdir(image_path)
                        frames_info = {}
                        valid_file_list = []
                        for file in file_list: 
                            prefix = file.split('.')[0]
                            if fio.exists(osp.join(self.CO3DV2_DIR, subname, obj, scene, 'depth_masks', prefix+'.png')) and \
                               fio.exists(osp.join(self.CO3DV2_DIR, subname, obj, scene, 'depths', prefix+'.jpg.geometric.png')) and \
                               fio.exists(osp.join(self.CO3DV2_DIR, subname, obj, scene, 'masks', prefix+'.png')) and \
                                (scene, prefix) in anno[obj]:
                                frames_info[prefix] = anno[obj][(scene, prefix)]
                                valid_file_list.append(file)
                        if len(valid_file_list) > self.min_num_images:
                            key = f"{subname}/{obj}/{scene}" # for example, key="apple_001/apple/110_13048_23163"
                            self.all_sequences.append(key)
                            self.scene_to_path[key] = key
                            self.scene_meta[key] = {
                                "frames_id": list(range(len(valid_file_list))),
                                "frames_name": [f.split('.')[0] for f in sorted(valid_file_list)],
                                "frames_info": frames_info
                            }

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
    def _load_one_view(self, scene_path, prefix, frames_info, fid, resolution, rng):
        # scene_path: apple_001/apple/110_13048_23163
        # prefix frame000001
        image_path = osp.join(self.CO3DV2_DIR, scene_path, 'images', prefix+'.jpg')
        rgb_image = read_any_file(image_path)
        H, W, _ = rgb_image.shape
        
        if self.load_depth:
            depth_path = osp.join(self.CO3DV2_DIR, scene_path, 'depths', prefix+'.jpg.geometric.png')
            depthmap = read_any_file(depth_path)
            depthmap = (
                np.frombuffer(np.array(depthmap, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depthmap.shape[0], depthmap.shape[1]))
            )
            depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

            # depth mask     
            depth_mask_path = osp.join(self.CO3DV2_DIR, scene_path, 'depth_masks', prefix+'.png')
            depth_mask = read_any_file(depth_mask_path)
            depth_mask = (depth_mask.astype(np.float32) / 255.0) > 0.1

            # mask
            mask_path = osp.join(self.CO3DV2_DIR, scene_path, 'masks', prefix+'.png')
            mask = read_any_file(mask_path)
            mask = (mask.astype(np.float32) / 255.0) > 0.1

            depthmap *= mask
  
            # depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)
        else:
            depthmap = None

        # load camera params
        camera_pose = np.eye(4)
        R = np.array(frames_info['R'])
        T = np.array(frames_info['T'])
        # camera_pose[:3, :3] = R.T
        # camera_pose[:3, 3] = - R.T @ T

        image_size = np.array([rgb_image.shape[0], rgb_image.shape[1]]) # HW
        focal_length = np.array(frames_info['focal_length'])
        principal_point = np.array(frames_info['principal_point'])
        # K = convert_ndc_to_pinhole(focal_length, principal_point, image_size)

        R, tvec, camera_intrinsics = opencv_from_cameras_projection(R, T, focal_length, principal_point, image_size)
        camera_pose[:3, :3] = R
        camera_pose[:3, 3] = tvec
        camera_pose = np.linalg.inv(camera_pose)

        # 自带的 depth_mask 跟深度一起裁剪缩放（最近邻），作为 extra_valid 返回；只在 use_dataset_masks=True 时生效
        dmask = depth_mask.astype(np.uint8) if (self.load_depth and depth_mask.shape == depthmap.shape) else None
        out = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            camera_intrinsics.copy(),
            resolution,
            rng=rng,
            info=image_path,
            far_mask=dmask,
        )
        rgb_image, depthmap, intrinsics = out[:3]
        extra_valid = (out[3] > 0) if dmask is not None else None

        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        view = dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=camera_pose.astype(np.float32),
            camera_intrinsics=intrinsics.astype(np.float32),
            dataset=self.dataset_label,
            label= scene_path, # 'apple_001/apple/110_13048_23163'
            instance=str(fid),
        )
        if extra_valid is not None:
            view['extra_valid'] = extra_valid
        return view

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        scene_path = self.scene_to_path[scene]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frames_id"], rng)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        views = []
        for fid in idxs:
            prefix = meta['frames_name'][fid] # frame000200
            frames_info = meta['frames_info'][prefix] # viewpoint
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    prefix=prefix,
                    frames_info=frames_info,
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.co3dv2_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(CO3DV2Dataset))
