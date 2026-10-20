import sys
sys.path.append('.')

import os
import os.path as osp
import torch
import numpy as np
import hashlib
import json
from tqdm import tqdm

FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map
from datasets.base.utils import read_any_file

class DL3DVDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=24,
        split_ratio=0.9,
        seq_num=-1,
        min_num_images=24,
        sample_method=None,
        conf_thres=0.25,
        depth_thres=1e-4,
        apply_trans=False,
        rgb_root='/efs-mi-east4-2/datasets/DL3DV/',   # 原始 RGB 目录（原先写死在代码里）
        depth_interp='lanczos',     # 'nearest'：深度缩放到 RGB 尺寸时用最近邻（lanczos 会在深度边缘造出飞点）
        pct_twice=True,             # False：98 分位只在裁剪缩放后截一次（旧行为是前后各截一次）
        **kwargs,
    ):

        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "DL3DV"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.sample_method = sample_method
        self.conf_thres = conf_thres
        self.depth_thres = depth_thres
        self.apply_trans = apply_trans
        self.rgb_root = rgb_root
        assert depth_interp in ('lanczos', 'nearest'), depth_interp
        self.depth_interp = depth_interp
        self.pct_twice = pct_twice
        self.DL3DV_DIR = data_root

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
        def get_intrinsic(cam_info):
            intrinsic = np.eye(3, dtype=np.float32)
            intrinsic[0,0] = float(cam_info["fl_x"])
            intrinsic[0,2] = float(cam_info["cx"])
            intrinsic[1,1] = float(cam_info["fl_y"])
            intrinsic[1,2] = float(cam_info["cy"])
            return intrinsic
        scene_list = os.listdir(self.DL3DV_DIR)
        
        for scene_name in tqdm(scene_list, desc=f"Building {self.dataset_label} cache"):
            cam_file = osp.join(self.DL3DV_DIR, scene_name, "scene_meta.json")
            if osp.exists(cam_file):
                with open(cam_file, 'r') as f:
                    cam_info = json.load(f)
                if cam_info["scene_name"] == scene_name:
                    frame_metadata = {}
                    frame_metadata["intrinsic"] = get_intrinsic(cam_info)
                    # choose valid view info with exisitence of rgb&depth file path and transformation matrix
                    frames = []
                    for frame_info in cam_info["frames"]:
                        # rgb_path = osp.join(self.DL3DV_DIR, scene_name, frame_info["file_path"].replace("images","images_use").replace(".jpg", ".png")) # for example
                        scene_seg = scene_name.split('-') # 9K-id
                        rgb_path = osp.join(self.rgb_root, scene_seg[0], scene_seg[1], 'images_8', osp.basename(frame_info["file_path"])).replace('.jpg', '.png')
                        depth_path = osp.join(self.DL3DV_DIR, scene_name, frame_info["mvsanywhere_depth"])
                        depthconf_path = osp.join(self.DL3DV_DIR, scene_name, frame_info["mvsanywhere_depth_confidence"])
        
                        if osp.exists(rgb_path) and osp.exists(depth_path) and osp.exists(depthconf_path):
                            frames.append(frame_info)

                    frame_metadata["rgb_depth_extrinsic"] = frames
                    frame_metadata["_applied_transform"] = cam_info["_applied_transform"]
                    frame_metadata["_applied_transforms"] = cam_info["_applied_transforms"]["opengl2opencv"]

                    if len(frames) >= self.min_num_images: # check view num > minimum required
                        self.all_sequences.append(scene_name)
                        self.scene_to_path[scene_name] = osp.join(self.DL3DV_DIR, scene_name)
                        self.scene_meta[scene_name] = {
                            "frames_id": list(range(len(frames))),
                            "frames_meta": frame_metadata
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
    def _load_one_view(self, scene_path, scene_name, frame_info, intrinsic, transmat, fid, resolution, rng):
        # rgb_image = read_any_file(osp.join(scene_path, frame_info["file_path"]).replace("images","images_use").replace(".jpg", ".png")) # for example
        scene_seg = scene_name.split('-') # 9K-id
        rgb_file = osp.join(self.rgb_root, scene_seg[0], scene_seg[1], 'images_8', osp.basename(frame_info["file_path"])).replace('.jpg', '.png')
        rgb_image = read_any_file(rgb_file)
        depthmap = read_any_file(osp.join(scene_path, frame_info["mvsanywhere_depth"]), return_dtype=np.float32)

        h, w, d = rgb_image.shape
        interp = cv2.INTER_NEAREST if self.depth_interp == 'nearest' else cv2.INTER_LANCZOS4
        depthmap = cv2.resize(depthmap, (w, h), interpolation=interp)
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        depthmap[(depthmap <= 0)] = 0.0
        depthconfmap = read_any_file(osp.join(scene_path, frame_info["mvsanywhere_depth_confidence"]), return_dtype=np.float32)
        if depthconfmap.shape[:2] != (h, w):
            # 置信度图要和深度一样先对齐到 RGB 尺寸，否则后面按 RGB 坐标裁剪时会错位
            depthconfmap = cv2.resize(depthconfmap, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.pct_twice:
            depthmap = threshold_depth_map(depthmap, min_percentile=0, max_percentile=98)
        extrinsic = np.array(frame_info["transform_matrix"])
        if self.apply_trans:
            extrinsic = transmat[1] @ transmat[0] @ extrinsic
        # extrinsic = np.linalg.inv(extrinsic)

        rgb_image, depthmap, intrinsic_, depthconfmap = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=osp.join(scene_path, frame_info["file_path"]),
            far_mask=depthconfmap,
        )
        depthmap = threshold_depth_map(depthmap, min_percentile=0, max_percentile=98)
        mask = (depthconfmap > self.conf_thres) & (depthmap> self.depth_thres)

        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=extrinsic.astype(np.float32),
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=osp.basename(scene_path),
            instance=str(fid),
            depth_mask=mask.astype(np.int32),
            extra_valid=mask,           # use_dataset_masks=True 时并入 valid_mask
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

        intrinsic = meta["frames_meta"]["intrinsic"]
        transmat1 = np.array(meta["frames_meta"]["_applied_transform"])
        transmat2 = np.array(meta["frames_meta"]["_applied_transforms"])
        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene_path=scene_path,
                    scene_name=scene,
                    frame_info=meta["frames_meta"]["rgb_depth_extrinsic"][fid],
                    intrinsic=intrinsic,
                    transmat=[transmat1,transmat2],
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views
    
if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt
    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/DL3DV_WAI"
    out_dir = "output_debug/DL3DV/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = DL3DVDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=64,
        split_ratio=1,
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
            f"valid={v['valid_mask'].sum()}",
        )

    """Visualize gt data. """
    visualize_gt(views, out_dir, prefix="DL3DV_GT")