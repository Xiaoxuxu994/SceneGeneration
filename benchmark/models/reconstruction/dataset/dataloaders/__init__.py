import bisect
import numpy as np
import torch
import torch.distributed as dist

from dotted_dict import DottedDict
from functools import partial
from torch.utils.data import ConcatDataset, DataLoader

from utils import PLATFORM, DEVICE_NAME
# if PLATFORM == "ASCEND":
#     import torch_npu
#     import torch_npu.npu as torch_device_lib
# else:
#     import torch.cuda as torch_device_lib

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
from .re10k_dataset import Re10K_Dataset
from .scannet_dataset import ScanNet_Dataset

from ..utils.common import customized_collate_fn_placeholder, worker_init_fn

DATASET_DICT = {
    "ase":        ASE_Dataset, 
    "blendedmvs": BlendedMVS_Dataset, 
    "hypersim":   HyperSim_Dataset, 
    "mvssynth":   MVSSynth_Dataset, 
    "replica":    Replica_Dataset, 
    "tartanair":  TartanAir_Dataset, 
    "vkitti":     VKitti_Dataset, 
    "co3dv2":     CO3Dv2_Dataset, 
    "dl3dv":      DL3DV_Dataset, 
    "re10k":      Re10K_Dataset, 
    "scannet":    ScanNet_Dataset, 
}

DATASET_SAMPLES_PER_EPOCH = {
    "ase":        100000, 
    "blendedmvs": 100000, 
    "hypersim":   100000, 
    "mvssynth":   100000, 
    "replica":    100000, 
    "tartanair":  100000, 
    "vkitti":     100000, 
    "co3dv2":     100000, 
    "dl3dv":      100000, 
    "re10k":      100000, 
    "scannet":    100000, 
}


def create_mixed_dataloader(cfg: DottedDict, data_type="train", batch_size=1, rank=None):
    assert rank is not None
    datasets = []
    dataset_batch_samples = []
    for dataset_name, dataset_cfg in cfg.items():
        dataset = DATASET_DICT[dataset_name](**dict(dataset_cfg), is_train=data_type=="train", batch_size=batch_size)
        if len(dataset) > 0:
            datasets.append(dataset)
            dataset_batch_samples.append(DATASET_SAMPLES_PER_EPOCH[dataset_name])
    mixed_dataset = CustomizedConcatDataset(datasets, dataset_batch_samples)
    device_id = "{}:{}".format(DEVICE_NAME, rank)
    dataloader = DataLoader(mixed_dataset, 
                            batch_size=1,  ## We implement batch function in Dataset class to ensure intra-batch image shape consistency. 
                            collate_fn=customized_collate_fn_placeholder, 
                            sampler=torch.utils.data.DistributedSampler(mixed_dataset, 
                                                                        shuffle=True, 
                                                                        drop_last=True) \
                                    if data_type == "train" else None, 
                            num_workers=4, 
                            persistent_workers=True,
                            multiprocessing_context="spawn", 
                            worker_init_fn=partial(worker_init_fn, device_id=device_id), 
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
        