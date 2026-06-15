import bisect
import numpy as np
import torch

from dotted_dict import DottedDict
from torch.utils.data import ConcatDataset, DataLoader

from utils import PLATFORM, DEVICE_NAME

## Synthetic dataset
from .ase_dataset import ASE_Dataset
from .blendedmvs_dataset import BlendedMVS_Dataset
from .hypersim_dataset import HyperSim_Dataset
from .mvssynth_dataset import MVSSynth_Dataset
from .replica_dataset import Replica_Dataset
from .tartanair_dataset import TartanAir_Dataset
from .vkitti_dataset import VKitti_Dataset

## Realistic dataset
from .co3dv2_dataset import CO3Dv2_Dataset
from .dl3dv_dataset import DL3DV_Dataset
from .mapillary_dataset import Mapillary_Dataset
from .re10k_dataset import Re10K_Dataset
from .scannet_dataset import ScanNet_Dataset
from .scannetv2_dataset import ScanNetv2_Dataset
from .scannetpp_dataset import ScanNetpp_Dataset
from .wildrgbd_dataset import WildRGBD_Dataset
from .spatialvid_dataset import SpatialVID_Dataset
from .sekai_dataset import Sekai_Dataset

from ..utils.common import customized_collate_fn_placeholder

DATASET_DICT = {
    "ase":        ASE_Dataset,        # indoor
    "blendedmvs": BlendedMVS_Dataset, # outdoor
    "hypersim":   HyperSim_Dataset,   # indoor
    "mvssynth":   MVSSynth_Dataset,   # outdoor
    "replica":    Replica_Dataset,    # indoor
    "tartanair":  TartanAir_Dataset,  # indoor/outdoor
    "vkitti":     VKitti_Dataset,     # outdoor
    "co3dv2":     CO3Dv2_Dataset,     # object
    "dl3dv":      DL3DV_Dataset,      # indoor/outdoor
    "mapillary":  Mapillary_Dataset, 
    "re10k":      Re10K_Dataset,      # indoor/outdoor
    "scannet":    ScanNet_Dataset,    # indoor
    "scannetv2":  ScanNetv2_Dataset, 
    "scannetpp":  ScanNetpp_Dataset,  # outdoor
    "wildrgbd":   WildRGBD_Dataset, 
    "spatialvid": SpatialVID_Dataset, 
    "sekai":      Sekai_Dataset, 
}

EPOCH_SCALE = 100000
DATASET_SAMPLES_PER_EPOCH_TRAIN = {
    "ase":        100 * EPOCH_SCALE, 
    "blendedmvs": 1   * EPOCH_SCALE, 
    "hypersim":   1   * EPOCH_SCALE, 
    "mvssynth":   1   * EPOCH_SCALE, 
    "replica":    1   * EPOCH_SCALE, 
    "tartanair":  10  * EPOCH_SCALE, 
    "vkitti":     1   * EPOCH_SCALE, 
    "co3dv2":     25  * EPOCH_SCALE, 
    "dl3dv":      25  * EPOCH_SCALE, 
    "mapillary":  1   * EPOCH_SCALE, 
    "re10k":      25  * EPOCH_SCALE, 
    "scannet":    20  * EPOCH_SCALE, 
    "scannetv2":  20  * EPOCH_SCALE, 
    "scannetpp":  20  * EPOCH_SCALE, 
    "wildrgbd":   25  * EPOCH_SCALE, 
    "spatialvid": 100 * EPOCH_SCALE, 
    "sekai":      100 * EPOCH_SCALE, 
}
DATASET_SAMPLES_PER_EPOCH_EVAL = {
    "ase":        1 * EPOCH_SCALE, 
    "blendedmvs": 1 * EPOCH_SCALE, 
    "hypersim":   1 * EPOCH_SCALE, 
    "mvssynth":   1 * EPOCH_SCALE, 
    "replica":    1 * EPOCH_SCALE, 
    "tartanair":  1 * EPOCH_SCALE, 
    "vkitti":     1 * EPOCH_SCALE, 
    "co3dv2":     1 * EPOCH_SCALE, 
    "dl3dv":      1 * EPOCH_SCALE, 
    "mapillary":  1 * EPOCH_SCALE, 
    "re10k":      1 * EPOCH_SCALE, 
    "scannet":    1 * EPOCH_SCALE, 
    "scannetv2":  1 * EPOCH_SCALE, 
    "scannetpp":  1 * EPOCH_SCALE, 
    "wildrgbd":   1 * EPOCH_SCALE, 
    "spatialvid": 1 * EPOCH_SCALE, 
    "sekai":      1 * EPOCH_SCALE, 
}


def create_mixed_dataloader(cfg: DottedDict, mode="train", batch_size=1, num_workers=8):
    datasets = []
    dataset_batch_samples = []
    for dataset_name, dataset_cfg in cfg.items():
        dataset = DATASET_DICT[dataset_name](**dict(dataset_cfg), is_train=mode=="train", batch_size=batch_size)
        if len(dataset) > 0:
            datasets.append(dataset)
            dataset_batch_samples.append(DATASET_SAMPLES_PER_EPOCH_TRAIN[dataset_name] if mode == "train" else \
                                         DATASET_SAMPLES_PER_EPOCH_EVAL[dataset_name])
    mixed_dataset = CustomizedConcatDataset(datasets, dataset_batch_samples)
    dataloader = DataLoader(mixed_dataset, 
                            batch_size=1,  ## We implement batch function in Dataset class to ensure intra-batch image shape consistency. 
                            collate_fn=customized_collate_fn_placeholder, 
                            sampler=torch.utils.data.DistributedSampler(mixed_dataset, 
                                                                        shuffle=True, 
                                                                        drop_last=True) \
                                    if mode == "train" else torch.utils.data.RandomSampler(mixed_dataset), 
                            num_workers=num_workers, 
                            persistent_workers=True, 
                            multiprocessing_context="fork", 
                            )
    return dataloader


def load_data_to_device(data):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device=DEVICE_NAME)
    return data


class CustomizedConcatDataset(ConcatDataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    
    Customzied to balance sample ratio in each sample. 

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    def __init__(self, datasets, samples_per_dataset):
        super().__init__(datasets)
        assert len(datasets) == len(samples_per_dataset)
        self.samples_per_dataset = torch.tensor(samples_per_dataset)
        self.cumulative_sizes = torch.cumsum(self.samples_per_dataset, dim=0).numpy()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = np.random.randint(low=0, high=len(self.datasets[dataset_idx]))
        return self.datasets[dataset_idx][sample_idx]
        