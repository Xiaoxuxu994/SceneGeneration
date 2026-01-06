# benchmark/datasets/__init__.py
from .sevenscenes import SevenScenesDataset
from .dtu import DTUDataset
from .nrgbd import NRGBDDataset
from .scannet import ScanNetDataset
from .dl3dv import DL3DVDataset

DATASET_REGISTRY = {
    "7scenes": SevenScenesDataset,
    "dtu": DTUDataset,
    "nrgbd": NRGBDDataset,
    "scannet": ScanNetDataset,
    "dl3dv": DL3DVDataset, # no depth gt
}

def get_dataset(name: str, root: str):
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset={name}, available={list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](root)
