# benchmark/datasets/__init__.py
from .sevenscenes import SevenScenesDataset
from .dtu import DTUDataset
from .nrgbd import NRGBDDataset
from .scannet import ScanNetDataset
# from .dl3dv import DL3DVDataset
from .ase import ASEDataset
from .dl3dv import NewDL3DVDataset
from .re10k import RE10KDataset

DATASET_REGISTRY = {
    "7scenes": SevenScenesDataset,
    "dtu": DTUDataset,
    "nrgbd": NRGBDDataset,
    "scannet": ScanNetDataset,
    # "dl3dv": DL3DVDataset, # no depth gt
    "dl3dv": NewDL3DVDataset, # no depth gt
    "ase": ASEDataset, # currently only support sampled sequences as a .npz file
    "re10k": RE10KDataset # currently only support sampled sequences as a .npz file
}

def get_dataset(name: str, root: str, num_seq: int):
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset={name}, available={list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](root, num_seq)
