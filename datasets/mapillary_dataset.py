import sys
sys.path.append('.')

import os
import os.path as osp

import torch
import numpy as np
import hashlib
import pickle
import json
from pyquaternion import Quaternion

FILE_DIR = os.path.dirname(os.path.abspath(__file__))        
PROJECT_ROOT = os.path.dirname(FILE_DIR)                     
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets.base.base_dataset import BaseDataset
from datasets.base import fileio as fio
from datasets.base.transforms import *
from datasets.sample_utils.common import threshold_depth_map
from datasets.base.utils import read_any_file

class MapillaryDataset(BaseDataset):
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

        self.dataset_label = "Mapillary"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio
        self.min_num_images = min_num_images
        self.sample_method = sample_method
        self.subdir = ['train', 'test', 'val']
        self.MAPILLARY_DIR = data_root

        self.all_sequences = []
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
            self._save_cache(keys=["all_sequences", "scene_meta"],save_path=cache_path)
            print(f'[Dataset] Cache is saved {cache_path}..')
        else:
            print(f'[Dataset] Loading cache from {cache_path}..')
            res = self._load_cache(cache_path)
            if res:
                self.cache_loaded = True
                print(f'[Dataset] Cache is loaded.')

        # 2) load all intrisics & extrinsics
        self.ego_pose= {}
        self.calibrated_sensor = {}
        for subdir in fio.listdir(self.MAPILLARY_DIR):
            if subdir in self.subdir:
                pose_dir = f"{self.MAPILLARY_DIR}/{subdir}/ego_pose.json"
                with fio.open(pose_dir) as file:
                    self.ego_pose[subdir] = json.load(file)
                instrinsic_dir = f"{self.MAPILLARY_DIR}/{subdir}/calibrated_sensor.json"
                with fio.open(instrinsic_dir) as file:
                    self.calibrated_sensor[subdir] = json.load(file)
                


        # 3) train/val split
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
    def _load_json_table(self, subdir, table_name):
        table_path = osp.join(self.MAPILLARY_DIR, subdir, f"{table_name}.json")
        with fio.open(table_path, "r") as f:
            return json.load(f)
    
    def _scan_scenes(self):
        """
        Load sequence metadata for Mapillary without relying on metropolis.py.
        It directly reads raw json tables and builds the minimal reverse index:
            sample["data"][channel] = sample_data_token
        """

        cameras = ["CAM_FRONT", "CAM_LEFT", "CAM_RIGHT", "CAM_BACK"]
        sequence_list = []
        data_store = {}

        for subdir in fio.listdir(self.MAPILLARY_DIR):
            if subdir not in self.subdir:
                continue

            # directly load raw tables
            scenes = self._load_json_table(subdir, "scene")
            samples = self._load_json_table(subdir, "sample")
            sample_data = self._load_json_table(subdir, "sample_data")
            sensors = self._load_json_table(subdir, "sensor")
            calibrated_sensors = self._load_json_table(subdir, "calibrated_sensor")

            # token -> record
            sample_by_token = {x["token"]: x for x in samples}
            sample_data_by_token = {x["token"]: x for x in sample_data}
            sensor_by_token = {x["token"]: x for x in sensors}
            calibrated_sensor_by_token = {x["token"]: x for x in calibrated_sensors}

            # build reverse index: sample["data"][channel] = sample_data_token
            for s in samples:
                s.setdefault("data", {})

            for sd in sample_data:
                sample_token = sd["sample_token"]
                calibrated_sensor_token = sd["calibrated_sensor_token"]

                if calibrated_sensor_token not in calibrated_sensor_by_token:
                    continue

                cs_record = calibrated_sensor_by_token[calibrated_sensor_token]
                sensor_token = cs_record["sensor_token"]

                if sensor_token not in sensor_by_token:
                    continue

                sensor_record = sensor_by_token[sensor_token]
                channel = sensor_record["channel"]

                # 顺手补回去，后面也能直接用
                sd["channel"] = channel
                sd["sensor_modality"] = sensor_record.get("modality", None)

                if sample_token in sample_by_token:
                    sample_by_token[sample_token]["data"][channel] = sd["token"]

            for cur_camera in cameras:
                for scene_info in scenes:
                    frame_metadata = []
                    cur_scene_token = scene_info["token"]

                    next_sample_token = scene_info["first_sample_token"]
                    while next_sample_token:
                        current_sample = sample_by_token[next_sample_token]

                        if cur_camera not in current_sample["data"]:
                            break

                        camera_info = sample_data_by_token[current_sample["data"][cur_camera]]
                        ego_pose_token = camera_info["ego_pose_token"]

                        if self.mode != "test" and ego_pose_token == "dummy_ego_pose":
                            next_sample_token = current_sample["next_sample"]
                            continue

                        camera_file = os.path.join(self.MAPILLARY_DIR, camera_info["filename"])
                        camera_height = camera_info["height"]
                        camera_width = camera_info["width"]

                        depth_cam = "DEPTH_" + cur_camera[4:]
                        if depth_cam not in current_sample["data"]:
                            next_sample_token = current_sample["next_sample"]
                            continue

                        calibrated_sensor_token = camera_info["calibrated_sensor_token"]
                        depth_info = sample_data_by_token[current_sample["data"][depth_cam]]
                        depth_file = os.path.join(self.MAPILLARY_DIR, depth_info["filename"])

                        if not (fio.exists(camera_file) and fio.exists(depth_file)):
                            next_sample_token = current_sample["next_sample"]
                            continue

                        frame_metadata.append(
                            {
                                "color": camera_info["filename"],
                                "depth": depth_info["filename"],
                                "pose": ego_pose_token,
                                "intrinsic": calibrated_sensor_token,
                                "height": camera_height,
                                "width": camera_width,
                                "format": "raw",
                            }
                        )

                        next_sample_token = current_sample["next_sample"]

                    scene_name = subdir + "_" + cur_scene_token + "_" + cur_camera
                    if len(frame_metadata) >= self.min_num_images:
                        data_store[scene_name] = frame_metadata
                        sequence_list.append(scene_name)

        all_sequences = []
        scene_meta = {}
        for scene_name in sequence_list:
            frame_metadata = data_store[scene_name]
            if len(frame_metadata) >= self.min_num_images:
                all_sequences.append(scene_name)
                scene_meta[scene_name] = {
                    "frames_id": list(range(len(frame_metadata))),
                    "frames_meta": frame_metadata,
                }

        self.all_sequences = all_sequences
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
    # load one view
    # =========================
    def _load_one_view(self, subdir, anno, fid, resolution, rng):
        if anno["depth"] == None:
            print("no depth data")
        rgb_image = read_any_file(osp.join(self.MAPILLARY_DIR, anno["color"]))

        depthmap = read_any_file(osp.join(self.MAPILLARY_DIR, anno["depth"]), return_dtype=np.float32) / 256.0
        depthmap = threshold_depth_map(depthmap, min_percentile=-1, max_percentile=98)

        index_intrinsic = next(
            (i for i, d in enumerate(self.calibrated_sensor[subdir]) if d.get("token") == anno["intrinsic"]), None
        )
        cs_record = self.calibrated_sensor[subdir][index_intrinsic]

        index_pose = next((i for i, d in enumerate(self.ego_pose[subdir]) if d.get("token") == anno["pose"]), None)
        pose_record = self.ego_pose[subdir][index_pose]

        intrinsic = np.array(cs_record["camera_intrinsic"])

        sens_to_ego = self.transform_matrix(
            rotation=Quaternion(cs_record["rotation"]),
            translation=np.array(cs_record["translation"]),
        )
        ego_to_world = self.transform_matrix(
            rotation=Quaternion(pose_record["rotation"]),
            translation=np.array(pose_record["translation"]),
        )
        extrinsic = np.dot(ego_to_world, sens_to_ego)
        
        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=osp.join(self.MAPILLARY_DIR, anno["color"]),
        )
        
        return dict(
            img=rgb_image,
            depthmap=depthmap.astype(np.float32),
            camera_pose=extrinsic.astype(np.float32),
            camera_intrinsics=intrinsic_.astype(np.float32),
            dataset=self.dataset_label,
            label=osp.basename(osp.join(self.MAPILLARY_DIR, anno["color"])),
            instance=str(fid),
        )

    # =========================
    # main interface
    # =========================
    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        meta = self.scene_meta[scene]
        subdir = scene.split('_')[0]
        idxs = self._sample_frame_ids(meta["frames_id"], rng)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
        }

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    subdir=subdir,
                    anno=meta['frames_meta'][fid],
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views

    def transform_matrix(
        self,
        translation: np.ndarray = np.array([0, 0, 0]),
        rotation: Quaternion = Quaternion([1, 0, 0, 0]),
        inverse: bool = False,
    ) -> np.ndarray:
        """Convert pose to transformation matrix.

        Args:
            translation: Translation in x, y, z.
            rotation: Rotation in quaternions (w ri rj rk).
            inverse: Whether to compute inverse transform matrix.

        Returns:
            Transformation matrix.
        """
        tm = np.eye(4)

        if inverse:
            # pyre-fixme[16]: `Quaternion` has no attribute `rotation_matrix`.
            rot_inv = rotation.rotation_matrix.T
            trans = np.transpose(-np.array(translation))
            tm[:3, :3] = rot_inv
            tm[:3, 3] = rot_inv.dot(trans)
        else:
            tm[:3, :3] = rotation.rotation_matrix
            tm[:3, 3] = np.transpose(np.array(translation))

        return tm
    


if __name__ == "__main__":
    # 单独调试（在仓库根目录）：python -m datasets.mapillary_dataset --n 3 [--data_root ... --cache_dir ... --remap SRC=DST --vis output_debug/x]
    # 参数说明见 datasets/base/debug.py
    from datasets.base.debug import debug_main
    sys.exit(debug_main(MapillaryDataset))
