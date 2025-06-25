
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import os
import torch
import numpy as np
import random
import torch_npu

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def create_tensorboard_writer(exp_dir):
    if dist.get_rank() == 0:  # real logger
        tensorboard_dir = f"{exp_dir}/tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_dir)
    else:
        writer = None
    return writer


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True) # 实验精确复现需开启
    # os.environ['HCCL_DETERMINISTIC'] = 'True'  # 'HCCL_DETERMINISTIC' is a deterministic switch in ops level, set it to 'True' to enable ops level deterministic, set it to 'False' to disable ops level deterministic.

def extend_seed_all(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['HCCL_DETERMINISTIC'] = 'True'  # 'HCCL_DETERMINISTIC' is a deterministic switch in ops level, set it to 'True' to enable ops level deterministic, set it to 'False' to disable ops level deterministic.
    os.environ['CLOSE_MATMUL_K_SHIFT'] = '1'  # 'CLOSE_MATMUL_K_SHIFT' is a switch of matmul K-axis shift, set it to '1' to close matmul K-axis shift, set it to '0' to enable matmul K-axis shift.
    os.environ['PYTHONHASHSEED'] = str(seed)  # 'PYTHONHASHSEED' refers to python hash seed, use a string of non-negative integer to specify the seed.
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch_npu.npu.manual_seed_all(seed)
    torch_npu.npu.manual_seed(seed)

def save_checkpoint(model, optimizer, lr_scheduler, scaler, epoch, step, global_step, batch_size, save_dir: str):
    if dist.get_rank() == 0:
        checkpoint = {}
        checkpoint["model"] = model.state_dict() # ddp model.module
        checkpoint["optimizer"] = optimizer.state_dict()
        checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["step"] = step
        checkpoint["global_step"] = global_step
        checkpoint["sample_start_index"] = step * batch_size
        torch.save(checkpoint, os.path.join(save_dir, f"ckpt-epoch{epoch}-global_step{global_step}.pth"))

def load_checkpoint(model, optimizer, lr_scheduler, scaler, path: str):
    checkpoint = torch.load(path, map_location=f"npu:{dist.get_rank()}")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["epoch"], checkpoint["step"] + 1, checkpoint["sample_start_index"] + 1