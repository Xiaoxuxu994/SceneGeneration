# %%
import os
import numpy as np
import random
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch._C._jit_set_profiling_mode(False)
from torch.utils.data import DataLoader
from tqdm import tqdm

from opensora_local.utils.ckpt_utils import load_checkpoint_ckpt
from opensora_local.utils.config_utils import parse_configs

## src code
from src.registry import MODELS, DATASETS, build_module
from src.dataset.dataset_build_scannet_v2 import DatasetForScanNetV2

def main():
    cfg = parse_configs(training=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.npu.manual_seed(cfg.seed)
    torch.npu.manual_seed_all(cfg.seed)

    if cfg.accurate_reproduce:
        torch.use_deterministic_algorithms(True) # 实验精确复现需开启
        os.environ['HCCL_DETERMINISTIC'] = 'True'  # 'HCCL_DETERMINISTIC' is a deterministic switch in ops level, set it to 'True' to enable ops level deterministic, set it to 'False' to disable ops level deterministic.
        
    print("====== cfg: ",cfg)
    device = "npu:0"
    dtype = torch.float32
    
    # build DatasetForScanNetV2
    dataset = build_module(cfg.dataset, DATASETS)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)
    num_val = len(dataloader)

    # model ckpt loading
    ckpt_path = os.path.join(cfg.root, cfg.ckpt_name)

    # model ckpt load
    print("====== cfg.model: ", cfg.model)
    model = build_module(cfg.model, MODELS)
    model = model.to(device, dtype)
    print("====== Loading checkpoint......")
    load_checkpoint_ckpt(model, ckpt_path, device)
    model.eval()

    for input_dict in tqdm(dataloader):
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key][0].to(device)
        with torch.no_grad():
            latent, kl = model.encode(input_dict)
        
        # === save and print ===
        scene_name = input_dict['scene_id'][0]+'.pth'
        torch.save(latent, os.path.join(cfg.outputs, scene_name))
        print(f"{scene_name} kl = {kl.item()}")
if __name__ == "__main__":
    main()