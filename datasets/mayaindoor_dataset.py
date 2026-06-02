import sys
sys.path.append(".")

import os
import os.path as osp
from glob import glob
import hashlib
import math
import xml.etree.ElementTree as ET

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


# ============================================================
# XML pose helpers
# ============================================================
def euler_to_rotmat(roll, pitch, yaw, degrees=True):
    """
    Convert roll / pitch / yaw to rotation matrix.

    当前使用:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    XML 里:
        <waypoint pitch="..." roll="..." x="..." y="..." yaw="..." z="..."/>
    """
    if degrees:
        roll = math.radians(float(roll))
        pitch = math.radians(float(pitch))
        yaw = math.radians(float(yaw))

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float32,
    )

    Ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float32,
    )

    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return (Rz @ Ry @ Rx).astype(np.float32)


def parse_route_xml_poses(xml_path):
    """
    Parse route_xxx.xml.

    XML format:
        <route id="route_95" ...>
            <waypoint pitch="0.0" roll="84.0" x="..." y="..." yaw="..." z="..."/>
            ...
        </route>

    Return:
        poses_sensor_c2w: list of [4, 4] np.ndarray

    注意:
        这里返回的是 simulator sensor/local coordinate 的 c2w。
        后面可选再转成 OpenCV camera coordinate。
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    poses = []

    for wp in root.findall(".//waypoint"):
        x = float(wp.attrib["x"])
        y = float(wp.attrib["y"])
        z = float(wp.attrib["z"])

        roll = float(wp.attrib.get("roll", 0.0))
        pitch = float(wp.attrib.get("pitch", 0.0))
        yaw = float(wp.attrib.get("yaw", 0.0))

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = euler_to_rotmat(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            degrees=True,
        )
        pose[:3, 3] = np.array([x, y, z], dtype=np.float32)

        poses.append(pose)

    return poses


def get_sensor_from_opencv_camera():
    """
    如果 XML pose 表示的是类似 CARLA / Unreal 的 sensor pose，
    local coordinate 常见约定是:
        x: forward
        y: right
        z: up

    OpenCV camera coordinate:
        x: right
        y: down
        z: forward

    则:
        p_sensor = T_sensor_from_cv @ p_cv

    即:
        sensor_x = cv_z
        sensor_y = cv_x
        sensor_z = -cv_y

    如果后面 visualize_gt 发现方向不对，可以关掉 use_opencv_camera 或调整这里。
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return T


# ============================================================
# Dataset
# ============================================================
class MayaIndoorDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=100.0,
        split_ratio=0.9,
        seq_num=-1,
        use_opencv_camera=True,
        cache_dir="/efs-mi-east4-2/z00533370/proj_3dgs/3drecon/data/dataset_cache",
        **kwargs,
    ):
        """
        Expected folder structure:

            data_root/
              SceneA/
                results/
                  route_95/
                    route_95.xml
                    vis/
                      color/
                        00000000_color_vis.png
                        00000004_color_vis.png
                        ...
                      depth/
                        00000000_depth_vis.png
                        00000004_depth_vis.png
                        ...
                  route_96/
                    ...
              SceneB/
                results/
                  route_xxx/
                    ...

        Args:
            data_root:
                /efs-m3ga/h00548084/relive3r_raw/Maya/Modern/Indoor

            depth_scale:
                如果 depth png 是 uint16 mm，则用 0.001。
                如果 depth 已经是 meter，则改成 1.0。
                如果 depth 是 RGB 可视化图，这个 depth 不适合正式训练。

            fov:
                XML 里没有内参，所以这里用水平 FOV 估计 K。
                如果后面找到真实 FOV，就改这个值。

            use_opencv_camera:
                True: 把 simulator camera local coordinate 转成 OpenCV camera coordinate。
                False: 直接使用 XML pose。
        """
        super().__init__(**kwargs)
        assert data_root is not None

        self.dataset_label = "MayaIndoor"
        self.data_root = data_root
        self.verbose = verbose
        self.max_distance = max_distance
        self.split_ratio = split_ratio

        self.use_opencv_camera = use_opencv_camera

        self.sensor_from_opencv = get_sensor_from_opencv_camera()

        self.all_sequences = []
        self.scene_to_path = {}
        self.scene_meta = {}

        os.makedirs(cache_dir, exist_ok=True)
        root_hash = hashlib.md5(osp.abspath(self.data_root).encode()).hexdigest()[:8]
        cache_path = osp.join(cache_dir, f"{self.dataset_label}_{root_hash}_cache.pkl")

        if not osp.exists(cache_path):
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
            f"[{self.dataset_label}] Found {len(self.sequences)} routes in {self.data_root} ({self.mode})",
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
        return osp.isdir(color_dir) and osp.isdir(depth_dir)

    def _make_sequence_name(self, route_path):
        """
        Use relative path as unique sequence name.

        Example:
            HuoCheCheXiangNei/results/route_95
        ->  HuoCheCheXiangNei__results__route_95
        """
        rel = osp.relpath(route_path, self.data_root)
        rel = rel.replace("\\", "/")
        return rel.replace("/", "__")

    def _parse_frame_id(self, color_path):
        """
        Example:
            00000000_color_vis.png -> 00000000
            00000004_color_vis.png -> 00000004
        """
        name = osp.basename(color_path)
        return name.split("_")[0]

    def _get_color_path(self, route_path, fid):
        return osp.join(route_path, "vis", "color", f"{fid}_color_vis.png")

    def _get_depth_path(self, route_path, fid):
        return osp.join(route_path, "vis", "depth", f"{fid}_depth.png")

    def _get_xml_path(self, route_path):
        """
        Prefer:
            route_path/route_xxx.xml

        Also support:
            parent_dir/route_xxx.xml
        """
        route_name = osp.basename(route_path)

        candidates = [
            osp.join(route_path, f"{route_name}.xml"),
            osp.join(osp.dirname(route_path), f"{route_name}.xml"),
        ]

        for p in candidates:
            if osp.isfile(p):
                return p

        return None

    def _route_sort_key(self, name):
        """
        route_95 -> 95
        """
        try:
            return int(name.split("_")[-1])
        except Exception:
            return name

    # ============================================================
    # scan / cache
    # ============================================================
    def _scan_scenes(self):
        """
        Fixed scanning structure:

            data_root/
              SceneA/
                results/
                  route_xxx/
                    route_xxx.xml
                    vis/color/
                    vis/depth/
                results_xxx/
                  route_xxx/
              SceneB/
                results/
                  route_xxx/

        Cache:
            self.all_sequences:
                list of sequence names

            self.scene_to_path:
                seq_name -> route_path

            self.scene_meta:
                seq_name -> {
                    "frame_ids": [...],
                    "fid_to_idx": {...},
                    "xml_path": ...,
                    "poses_c2w": [...]
                }
        """
        all_sequences = []
        scene_to_path = {}
        scene_meta = {}

        scene_names = sorted(
            [
                x for x in os.listdir(self.data_root)
                if osp.isdir(osp.join(self.data_root, x))
            ]
        )

        print(f"[{self.dataset_label}] scanning scenes: {len(scene_names)}")

        for scene_name in tqdm(scene_names, desc=f"Building {self.dataset_label} cache"):
            scene_path = osp.join(self.data_root, scene_name)

            result_names = sorted(
                [
                    x for x in os.listdir(scene_path)
                    if osp.isdir(osp.join(scene_path, x)) and x.startswith("result")
                ]
            )

            for result_name in result_names:
                result_path = osp.join(scene_path, result_name)

                route_names = sorted(
                    [
                        x for x in os.listdir(result_path)
                        if osp.isdir(osp.join(result_path, x)) and x.startswith("route_")
                    ],
                    key=self._route_sort_key,
                )

                for route_name in route_names:
                    route_path = osp.join(result_path, route_name)

                    if not self._is_valid_route(route_path):
                        continue

                    xml_path = self._get_xml_path(route_path)
                    if xml_path is None:
                        print(f"[WARN] skip {route_path}, no route xml found")
                        continue

                    try:
                        all_xml_poses = parse_route_xml_poses(xml_path)
                    except Exception as e:
                        print(f"[WARN] skip {route_path}, failed to parse xml: {e}")
                        continue

                    if len(all_xml_poses) == 0:
                        print(f"[WARN] skip {route_path}, empty xml poses")
                        continue

                    color_dir = osp.join(route_path, "vis", "color")
                    color_files = sorted(glob(osp.join(color_dir, "*_color_vis.png")))

                    frame_ids = []
                    poses_c2w = []

                    for color_path in color_files:
                        fid = self._parse_frame_id(color_path)
                        depth_path = self._get_depth_path(route_path, fid)

                        if not osp.isfile(depth_path):
                            continue

                        if not fid.isdigit():
                            continue

                        # Key rule:
                        #   00000000 -> waypoint[0]
                        #   00000004 -> waypoint[4]
                        #   00000192 -> waypoint[192]
                        pose_idx = int(fid)

                        if pose_idx >= len(all_xml_poses):
                            print(
                                f"[WARN] skip frame {fid} in {route_path}, "
                                f"pose_idx={pose_idx}, num_poses={len(all_xml_poses)}"
                            )
                            continue

                        pose_sensor_c2w = all_xml_poses[pose_idx]

                        if self.use_opencv_camera:
                            # c2w_opencv = c2w_sensor @ sensor_from_opencv
                            pose_c2w = pose_sensor_c2w @ self.sensor_from_opencv
                        else:
                            pose_c2w = pose_sensor_c2w

                        frame_ids.append(fid)
                        poses_c2w.append(pose_c2w.astype(np.float32))

                    frame_ids = sorted(frame_ids, key=lambda x: int(x))

                    if len(frame_ids) == 0:
                        continue

                    # 重新按照排序后的 frame_ids 对齐 pose
                    # 因为上面 color_files 通常已经 sorted，但这里保险起见重排一次
                    fid_to_pose = {
                        fid: pose for fid, pose in zip(frame_ids, poses_c2w)
                    }
                    poses_c2w = [fid_to_pose[fid] for fid in frame_ids]

                    seq_name = self._make_sequence_name(route_path)

                    all_sequences.append(seq_name)
                    scene_to_path[seq_name] = route_path
                    scene_meta[seq_name] = {
                        "frame_ids": frame_ids,
                        "fid_to_idx": {fid: i for i, fid in enumerate(frame_ids)},
                        "xml_path": xml_path,
                        "poses_c2w": poses_c2w,
                        "num_xml_poses": len(all_xml_poses),
                    }

        self.all_sequences = sorted(all_sequences)
        self.scene_to_path = scene_to_path
        self.scene_meta = scene_meta

        print(
            f"[{self.dataset_label}] scanned {len(self.all_sequences)} valid routes from {self.data_root}",
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
    def _build_intrinsic(self, H, W):
        """
        Hard-code render config intrinsic.

        Original render config:
            width  = 1280
            height = 720
            fx = 762.7
            fy = 762.7
            cx = 640.0
            cy = 360.0

        If image size is not 1280x720, scale K accordingly.
        """
        base_W = 1280.0
        base_H = 720.0

        fx = 762.7
        fy = 762.7
        cx = 640.0
        cy = 360.0

        sx = float(W) / base_W
        sy = float(H) / base_H

        K = np.array(
            [
                [fx * sx, 0.0, cx * sx],
                [0.0, fy * sy, cy * sy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        return K
    
    def _load_depth(self, depth_path):
        """
        简化版 depth 读取。

        如果 depth png 是 uint16 mm:
            depth_scale=0.001

        如果 depth 已经是 meter:
            depth_scale=1.0

        如果 depth 是 RGB 可视化图:
            这里会转灰度，但这个 depth 不适合正式训练。
        """
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)

        if depth.ndim == 3:
            depth = depth[..., :3].astype(np.float32).mean(axis=-1)
        else:
            depth = depth.astype(np.float32)

        depth = depth * 0.001
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        return depth.astype(np.float32)

    # ============================================================
    # load one view
    # ============================================================
    def _load_one_view(self, scene, fid, resolution, rng):
        route_path = self.scene_to_path[scene]
        meta = self.scene_meta[scene]

        color_path = self._get_color_path(route_path, fid)
        depth_path = self._get_depth_path(route_path, fid)

        rgb_image = Image.open(color_path).convert("RGB")
        depthmap = self._load_depth(depth_path)

        W0, H0 = rgb_image.size
        intrinsic = self._build_intrinsic(H0, W0)

        frame_idx = meta["fid_to_idx"][fid]
        camera_pose = meta["poses_c2w"][frame_idx]

        rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
            rgb_image,
            depthmap,
            intrinsic.copy(),
            resolution,
            rng=rng,
            info=color_path,
        )

        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

        if self.max_distance is not None and self.max_distance > 0:
            depthmap = np.where(depthmap > self.max_distance, 0.0, depthmap)

        depthmap = threshold_depth_map(depthmap, max_percentile=98)

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
        scene = self.sequences[index]
        meta = self.scene_meta[scene]

        idxs = self._sample_frame_ids(meta["frame_ids"], rng)

        self.this_views_info = {
            "scene": scene,
            "idxs": idxs,
            "route_path": self.scene_to_path[scene],
            "xml_path": meta.get("xml_path", None),
            "num_xml_poses": meta.get("num_xml_poses", None),
        }

        views = []
        for fid in idxs:
            views.append(
                self._load_one_view(
                    scene=scene,
                    fid=fid,
                    resolution=resolution,
                    rng=rng,
                )
            )

        return views


if __name__ == "__main__":
    from datasets.sample_utils.vis_util import visualize_gt

    data_root = "/mnt/nvme3/z00533370/dataset/sample_data/Modern/Indoor/"

    out_dir = "output_debug/MayaIndoor/"
    os.makedirs(out_dir, exist_ok=True)

    dataset = MayaIndoorDataset(
        data_root=data_root,
        mode="train",
        resolution=[[224, 224]],
        frame_num=8,
        max_distance=100.0,
        split_ratio=0.9,
        verbose=True,
        shuffle=False,
        # True: 把 simulator camera 坐标转成 OpenCV camera 坐标。
        # 如果 visualize_gt 发现方向明显不对，可以先改成 False 对比。
        use_opencv_camera=False,
    )

    print(f"len(dataset) = {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Please check data_root and folder structure.")

    views = dataset[0]

    print(f"[dataset] number of views = {len(views)}")
    print(f"[dataset] this_views_info = {dataset.this_views_info}")

    for i, v in enumerate(views):
        img_shape = tuple(v["img"].shape) if torch.is_tensor(v["img"]) else v["img"].size

        print(
            f"[dataset view {i}] "
            f"img={img_shape}, "
            f"depth={v['depthmap'].shape}, "
            f"depthmin={float(v['depthmap'].min())}, "
            f"depthmax={float(v['depthmap'].max())}, "
            f"pose_t={v['camera_pose'][:3, 3]}, "
            f"K={v['camera_intrinsics']}, "
            f"pts3d={v['pts3d'].shape if 'pts3d' in v else None}, "
            f"valid={v['valid_mask'].sum() if 'valid_mask' in v else None}, "
            f"label={v['label']}, "
            f"instance={v['instance']}"
        )

    visualize_gt(views, out_dir, prefix="MayaIndoor_GT")