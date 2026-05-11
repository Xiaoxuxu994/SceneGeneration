import os
import json
from utils import PLATFORM
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

from argparse import ArgumentParser

import torch
if PLATFORM == "ASCEND":
    import torch_npu
    import torch_npu.npu as torch_device_lib
else:
    import torch.cuda as torch_device_lib
import torch.distributed as dist

import numpy as np

from utils.config_loader import LoadConfig
from utils.loss_utils import LossFunction
from utils.platform import DEVICE_NAME
from dataset import create_mixed_dataloader, load_data_to_device

from model.blocks.matching_model import MatchingModel


ENABLE_TIME_PROFILING = False

# def clean_up():
#     dist.destroy_process_group()

def model_output_cheater(data):
    B, V, C, H, W = data["input_images"].shape
    return {
        "mono_depths_0":      torch.zeros((B, V, 1, H, W), device=DEVICE_NAME), 
        "depths_00":          torch.zeros((B, V, 1, H, W), device=DEVICE_NAME), 
        "scaled_depths_00":   torch.zeros((B, V, 1, H, W), device=DEVICE_NAME), 
        "pred_depth_scale_0": None, 
        "pred_depth_trans_0": None, 
        "depths_00_conf":     torch.ones((B, V, 1, H, W), device=DEVICE_NAME), 
    }

def run(rank, cfg):
    print("[ INFO ] Start DDP on rank {}".format(rank))
    
    torch_device_lib.set_device(rank)
    # dist.init_process_group("nccl" if PLATFORM == "NVIDIA" else "hccl")
    
    # Specify global rank for training output.
    global_rank = rank + int(os.environ.get("VC_TASK_INDEX", 0)) * int(os.environ.get("MA_NUM_GPUS", 0))
    
    """Load data, create data loader. Avoid possible conflicts of access to data list from multiple gpus.
    """
    if rank == 0:
        dataloader = create_mixed_dataloader(cfg.datasets, "eval", cfg.training.batch_size, rank=rank)
        data_iter  = iter(dataloader)
    # dist.barrier()
    if rank != 0:
        dataloader = create_mixed_dataloader(cfg.datasets, "eval", cfg.training.batch_size, rank=rank)
        data_iter  = iter(dataloader)
    # dist.barrier()
    
    """Create loss functions.
    """
    matching_model = MatchingModel(cfg.model.matching_model).eval().to(device=DEVICE_NAME) \
        if cfg.loss.geometric.w > 0.0 else None
    # loss_func = create_loss_func(cfg.loss, matching_model)
    loss_func = LossFunction(cfg.loss, matching_model)
    
    """Training loop.
    """
    step_start = torch_device_lib.Event(enable_timing=True)
    step_end   = torch_device_lib.Event(enable_timing=True)
    
    total_time = 0.0
    error_count = 0
    step = 0
    
    step_start.record()
    for data in dataloader:
    # while step < cfg.training.iterations:
        step += 1
        # dist.barrier()
        if data is None:
            error_count += 1
            continue
        else:
            """Cheat the program with tensor of the same shape as network prediction.
            """
            data = load_data_to_device(data)
            pred = model_output_cheater(data)

            loss_dict = loss_func(pred, data)
            
            matching_info_export_path = os.path.join(cfg.datasets.dl3dv.data_root, "matching_preprocess")
            os.makedirs(matching_info_export_path, exist_ok=True)
            matching_infos = loss_dict["matching_info"]
            batch_size = len(data["data_idxs"])
            for i, data_idx in enumerate(data["data_idxs"]):
                
                ### If there are corner samples not processed by multi-processing, address them.
                if os.path.exists(os.path.join(matching_info_export_path, "{:09d}.npy".format(data_idx))):
                    continue
                
                ### Preprocess the matching information from loss function and export them.
                export_matching_info = {}
                for idx, layer_matching_info in enumerate(matching_infos):
                    matching_info_length = len(layer_matching_info["matches"])
                    assert matching_info_length % batch_size == 0
                    matching_info_length_per_batch = matching_info_length // batch_size
                    export_matching_info.update({
                        "layer{}".format(idx): {
                            "type": layer_matching_info["type"], 
                            "matches": layer_matching_info["matches"][matching_info_length_per_batch * i: 
                                                                    matching_info_length_per_batch * (i + 1)].cpu().numpy(), 
                            "visibilities": layer_matching_info["visibilities"][matching_info_length_per_batch * i: 
                                                                                matching_info_length_per_batch * (i + 1)].cpu().numpy(), 
                        }
                    })
                np.save(os.path.join(matching_info_export_path, "{:09d}.npy".format(data_idx)), export_matching_info)
        
        step_end.record()
        # dist.barrier()
        step_time = step_start.elapsed_time(step_end) / 1000
        total_time += step_time / 3600
        process_rate = step / len(dataloader) + 1e-6
        if rank == 0:
            print("[ INFO - device{} ] Process {}/{}, {:.2f} seconds/iter, {:.2f} hours left, error count: {}.".format(
                rank, step, len(dataloader), step_time, total_time / process_rate * (1 - process_rate), error_count))
        
        del pred
        del loss_dict
        # del export_matching_info
        torch_device_lib.empty_cache()
        step_start.record()
    
    # clean_up()


def main(cfg):
    run(0, cfg)

if __name__ == '__main__':
    parser = ArgumentParser("Argument parser for feed-forward training.")
    parser.add_argument("--config", type=str, required=True, help="Path to .yaml format config file.")
    parser.add_argument("--data_url", type=str, default=None, help="Input path for ModelArts training.")
    parser.add_argument("--train_url", type=str, default=None, help="Output path for ModelArts training.")
    parser.add_argument("--iterations", type=int, default=None, help="Training iterations, overwriting the configuration file.")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size, overwriting the configuration file.")
    parser.add_argument("--ckpt_interval", type=int, default=None, help="Checkpoint save interval, overwriting the configuration file.")
    parser.add_argument("--verbose_interval", type=int, default=None, help="Prediction verbose interval, overwriting the configuration file.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint.")
    parser.add_argument("--eval", action="store_true", help="Skip training and evaluate in every step.")
    args = parser.parse_args()
    
    cfg = LoadConfig(args.config)
    
    if args.data_url is not None:
        cfg.datasets.dl3dv.data_root = args.data_url
    if args.train_url is not None:
        cfg.output_dir = args.train_url
    if args.iterations is not None:
        cfg.training.iterations = args.iterations
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.ckpt_interval is not None:
        cfg.training.ckpt_interval = args.ckpt_interval
    if args.verbose_interval is not None:
        cfg.training.verbose_interval = args.verbose_interval
    if args.ckpt_path is not None:
        cfg.ckpt_path = args.ckpt_path
    cfg.eval = args.eval
    
    main(cfg)