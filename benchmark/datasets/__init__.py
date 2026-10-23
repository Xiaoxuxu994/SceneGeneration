# benchmark/datasets/__init__.py
from .base import DatasetConfig
from .sevenscenes import SevenScenesDataset
from .dtu import DTUDataset
from .nrgbd import NRGBDDataset
from .scannet import ScanNetDataset
from .scannetpp import ScanNetPPDataset
from .scannetpp_r3 import ScanNetPPR3Dataset
# from .dl3dv import DL3DVDataset
from .ase import ASEDataset
from .dl3dv import NewDL3DVDataset
from .re10k import RE10KDataset
from .dtu import DTUDataset
from .eth3d import ETH3DDataset
from .kitti import KITTIDataset
from .sintel import SintelDataset
from .bonn import BonnDataset
from .tum import TumDataset
from .spatialvid import SpatialVIDDataset
from .hiroom import HiRoomDataset
from .sekai import SekaiDataset

from pathlib import Path

DATASET_REGISTRY = {
    "7scenes": SevenScenesDataset,
    "dtu": DTUDataset,
    "eth3d": ETH3DDataset,
    "nrgbd": NRGBDDataset,
    "scannet": ScanNetDataset,
    "scannetpp": ScanNetPPDataset,
    "scannetpp_r3": ScanNetPPR3Dataset, # SceneGeneration (reliev3r) layout, training held-out split
    # "dl3dv": DL3DVDataset, # no depth gt
    "dl3dv": NewDL3DVDataset, # no depth gt
    "ase": ASEDataset, # currently only support sampled sequences as a .npz file
    "re10k": RE10KDataset, # currently only support sampled sequences as a .npz file
    "kitti": KITTIDataset,
    "sintel": SintelDataset,
    "bonn": BonnDataset,
    "tum": TumDataset,
    "spatialvid": SpatialVIDDataset,
    "hiroom": HiRoomDataset,
    "sekai": SekaiDataset
}

def get_dataset(name: str, config: DatasetConfig):
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset={name}, available={list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](config)

def get_dataset_config(args: dict):
    return DatasetConfig(
        data_root=args.get("data_root", "."),
        num_seq=args.get("num_seq", 10),
        
        rgb_origin_W=args.get("rgb_origin_W", 518),
        rgb_origin_H=args.get("rgb_origin_H", 518),
        
        crop_left=args.get("crop_left", 0),
        crop_right=args.get("crop_right", 0),
        crop_top=args.get("crop_top", 0),
        crop_bottom=args.get("crop_bottom", 0),
        
        rgb_resize_W=args.get("rgb_resize_W", 0),
        rgb_resize_H=args.get("rgb_resize_H", 0),
        
        depth_scale=args.get("depth_scale", 1000.0),
        depth_min=args.get("depth_min", 0.1),
        depth_max=args.get("depth_max", 10.0),
        
        fx=args.get("fx", 1.0),
        fy=args.get("fy", 1.0),
        cx=args.get("cx", 1.0),
        cy=args.get("cy", 1.0)
    )
