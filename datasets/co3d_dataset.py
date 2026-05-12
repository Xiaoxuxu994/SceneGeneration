import sys
sys.path.append('.')

import os
import os.path as osp

import torch
import numpy as np
import hashlib
import pickle
import json
import gzip
from pathlib import Path
from tqdm import tqdm

SEEN_CATEGORIES = [
    "apple",
    "backpack",
    # "ball", # not in original vggt categories
    "banana",
    "baseballbat",
    "baseballglove",
    "bench",
    "bicycle",
    # "book", # not in original vggt categories
    "bottle",
    "bowl",
    "broccoli",
    "cake",
    "car",
    "carrot",
    "cellphone",
    "chair",
    # "couch", # not in original vggt categories
    "cup",
    "donut",
    # "frisbee", # not in original vggt categories
    "hairdryer",
    "handbag",
    # "hotdog". # not in original vggt categories
    "hydrant",
    "keyboard",
    # "kite", # not in original vggt categories
    "laptop",
    "microwave",
    "motorcycle",
    "mouse",
    "orange",
    "parkingmeter",
    "pizza",
    "plant",
    # "remote", # not in original vggt categories
    # "sandwich", # not in original vggt categories
    # "skateboard", # not in original vggt categories
    "stopsign",
    # "suitcase", # not in original vggt categories
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
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map
from datasets.base.utils import read_any_file

class CO3DDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
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

        self.dataset_label = "CO3D"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.sample_method = sample_method
        self.load_depth = load_depth
        self.CO3D_DIR = data_root

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
        split_name_list = ['train', 'test']
        all_sequences = []
        scene_to_path = {}
        scene_meta = {}
        category = sorted(SEEN_CATEGORIES)
        for c in tqdm(category, desc="Scanning categories"):
            for split_name in tqdm(split_name_list, desc=f"{c} splits", leave=False):
                annotation_file = osp.join(self.CO3D_DIR, "anno", f"{c}_{split_name}.jgz")

                try:
                    with gzip.open(annotation_file, "r") as fin:
                        annotation = json.loads(fin.read())
                except FileNotFoundError:
                    print(f"Annotation file not found: {annotation_file}")
                    continue

                for seq_name, seq_data in annotation.items():
                    seq_path = osp.join(self.CO3D_DIR, c, seq_name)

                    if not osp.exists(seq_path):
                        continue
                    if len(seq_data) >= self.min_num_images:
                        key = f"{c}/{seq_name}"
                        all_sequences.append(key)
                        scene_to_path[key] = key
                        scene_meta[key] = {
                            "frames_id": list(range(len(seq_data))),
                            "frames_data": seq_data
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
    # load one view
    # =========================
    def _load_one_view(self, scene_path, anno, fid, resolution, rng):
        filepath = anno["filepath"]

        image_path = osp.join(self.CO3D_DIR, filepath)
        # print(f"image_path: {image_path}")
        rgb_image = read_any_file(image_path)

        if self.load_depth:
            depth_path = image_path.replace("/images", "/depths") + ".geometric.png"
            depthmap = read_any_file(depth_path)
            depthmap = (
                np.frombuffer(np.array(depthmap, dtype=np.uint16), dtype=np.float16)
                .astype(np.float32)
                .reshape((depthmap.shape[0], depthmap.shape[1]))
            )

            mvs_mask_path = image_path.replace("/images", "/depth_masks").replace(".jpg", ".png")
            mvs_mask = read_any_file(mvs_mask_path) == True
            depthmap[~mvs_mask] = 0

            # depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)
        else:
            depthmap = None

        extrinsic = np.array(anno["extri"])
        intrinsic = np.array(anno["intri"])
        
        extrinsic = np.concatenate([extrinsic, np.array([[0,0,0,1]], dtype=extrinsic.dtype)], axis=0)
        extrinsic = np.linalg.inv(extrinsic)

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=image_path,
        )

        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=extrinsic.astype(np.float32),
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label= scene_path, # 'baseballbat/135_15553_31011'
            instance=str(fid),
        )

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
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    anno=meta['frames_data'][fid],
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views

if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt
    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/CO3D"
    out_dir = "output_debug/CO3D/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = CO3DDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=24,
        max_distance=8,
        split_ratio=1,
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
    visualize_gt(views, out_dir, prefix="CO3D_GT")