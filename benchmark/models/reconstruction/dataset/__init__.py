import torch
import torch.distributed as dist

from dotted_dict import DottedDict
from functools import partial
from torch.utils.data import ConcatDataset, DataLoader

from .dl3dv import DL3DV_Dataset, DL3DV_collate_fn
from .dl3dv_wai import DL3DV_WAI_Dataset, DL3DV_WAI_collate_fn
from .dl3dv_cut3r import DL3DV_CUT3R_Dataset, DL3DV_CUT3R_collate_fn
from utils import PLATFORM, DEVICE_NAME
if PLATFORM == "ASCEND":
    import torch_npu
    import torch_npu.npu as torch_device_lib
else:
    import torch.cuda as torch_device_lib


DATASET_DICT = {
    "dl3dv": DL3DV_Dataset, 
    "dl3dv_wai": DL3DV_WAI_Dataset, 
    "dl3dv_cut3r": DL3DV_CUT3R_Dataset, 
}
COLLATE_FN_DICT = {
    "dl3dv": DL3DV_collate_fn, 
    "dl3dv_wai": DL3DV_WAI_collate_fn, 
    "dl3dv_cut3r": DL3DV_CUT3R_collate_fn, 
}

def worker_init_fn(worker_id, device_id):
    torch_device_lib.set_device(device_id)

def create_mixed_dataloader(cfg: DottedDict, data_type="train", batch_size=1, rank=None):
    assert rank is not None
    datasets = []
    for dataset_name, dataset_cfg in cfg.items():
        dataset = DATASET_DICT[dataset_name](dataset_cfg, data_type=data_type)
        datasets.append(dataset)
    mixed_dataset = ConcatDataset(datasets)
    device_id = "{}:{}".format(DEVICE_NAME, rank)
    dataloader = DataLoader(mixed_dataset, 
                            batch_size=batch_size, 
                            collate_fn=DL3DV_WAI_collate_fn, 
                            sampler=torch.utils.data.DistributedSampler(mixed_dataset, 
                                                                        shuffle=True, 
                                                                        drop_last=True) \
                                    if data_type == "train" else None, 
                            num_workers=2, 
                            persistent_workers=True,
                            multiprocessing_context="spawn", 
                            worker_init_fn=partial(worker_init_fn, device_id=device_id), 
                            )
    return dataloader

def load_data_to_device(data):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device=DEVICE_NAME)
    if data.get("matching_infos", None) is not None:
        for key in data["matching_infos"].keys():
            for key2 in data["matching_infos"][key].keys():
                if isinstance(data["matching_infos"][key][key2], torch.Tensor):
                    data["matching_infos"][key][key2] = data["matching_infos"][key][key2].to(device=DEVICE_NAME)
    return data