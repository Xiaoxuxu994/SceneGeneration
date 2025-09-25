import functools
import json
import logging
import operator
import os
from typing import Tuple, Optional, Set

try:
    import colossalai
except ImportError:
    colossalai = None
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import is_initialized
from opensora_local.acceleration.parallel_states import (
    get_data_parallel_group,
    set_data_parallel_group,
    set_sequence_parallel_group,
)

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets.utils import download_url

import deepspeed
import deepspeed.comm as dist

from collections import OrderedDict

pretrained_models = {
    "DiT-XL-2-512x512.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt": "https://huggingface.co/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth": "https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth",
    "OpenSora-v1-16x256x256.pth": "https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-16x256x256.pth",
    "OpenSora-v1-HQ-16x256x256.pth": "https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x256x256.pth",
    "OpenSora-v1-HQ-16x512x512.pth": "https://huggingface.co/hpcai-tech/Open-Sora/resolve/main/OpenSora-v1-HQ-16x512x512.pth",
}


def reparameter(ckpt, name=None):
    if "DiT" in name:
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    elif "Latte" in name:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    elif "PixArt" in name:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    return ckpt


def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        model = download_model(model_name)
        model = reparameter(model, model_name)
        return model
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "pos_embed_temporal" in checkpoint:
            del checkpoint["pos_embed_temporal"]
        if "pos_embed" in checkpoint:
            del checkpoint["pos_embed"]
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f"pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        web_path = pretrained_models[model_name]
        download_url(web_path, "pretrained_models", model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


# def load_from_sharded_state_dict(model, ckpt_path):
#     ckpt_io = GeneralCheckpointIO()
#     ckpt_io.load_model(model, os.path.join(ckpt_path, "model"))

def model_sharding(model: torch.nn.Module):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def remove_padding(tensor: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
    return tensor[: functools.reduce(operator.mul, original_shape)]


def model_gathering(model: torch.nn.Module, model_shape_dict: dict):
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    for name, param in model.named_parameters():
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if int(global_rank) == 0:
            all_params = torch.cat(all_params)
            param.data = remove_padding(all_params, model_shape_dict[name]).view(model_shape_dict[name])
    dist.barrier()


def record_model_param_shape(model: torch.nn.Module) -> dict:
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape

def deepspeed_save(model, epoch, step, batch_size, global_step, save_dir=''):
    model.save_checkpoint(
        save_dir=save_dir,
        client_state={
            "epoch": epoch,
            "step": step,
            "start_idx": step*batch_size,
            "global_step": global_step
        }    
    )

def deepspeed_load(model, load_dir=''):
    load_path, client_state = model.load_checkpoint(load_dir=load_dir)
    if load_path:
        epoch, step, idx = client_state["epoch"], client_state["step"], client_state["start_idx"]
    else:
        print('[WARNING] not find checkpoint, start training from beginning')
        epoch = step = idx = 0
    return epoch, step, idx


def save(
    booster,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    global_step: int,
    batch_size: int,
    coordinator=None,
    save_dir='',
):

    save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
    running_states = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "sample_start_index": step * batch_size,
    }
    if booster is None:
        # import pdb; pdb.set_trace()
        if is_initialized():
            rank = dist.get_rank()
            if rank == 0:
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, os.path.join(save_dir, f'ckpt{epoch}.pth'))
        else:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(save_dir, f'ckpt{epoch}.pth'))
    else:
        booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
        # ema is not boosted, so we don't need to use booster.save_model

        booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096)
        if lr_scheduler is not None:
            booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    
    if coordinator is None or coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))
    if coordinator is not None:
        dist.barrier()


def load(
    booster, model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler, load_dir: str
) -> Tuple[int, int, int]:
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    epoch = running_states["epoch"]
    if booster is None:
        if is_initialized():
            rank = dist.get_rank(get_data_parallel_group())
            map_location = {'npu:%d'% 0: "npu:%d" % rank}
            pts = torch.load(os.path.join(load_dir, f'ckpt{epoch}.pth'), map_location=map_location)
            # pts = torch.load(os.path.join(load_dir, 'model_ckpt.pt'), map_location=map_location)

            model.module.load_state_dict(pts["model"])
            optimizer.load_state_dict(pts["optimizer"])
        else:
            pts = torch.load(os.path.join(load_dir, f'ckpt{epoch}.pth'), map_location='cpu')
            # pts = torch.load(os.path.join(load_dir, 'model_ckpt.pt'), map_location='cpu')
            model.load_state_dict(pts["model"])
            optimizer.load_state_dict(pts["optimizer"])
    else:
        booster.load_model(model, os.path.join(load_dir, "model"))
        booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
        if lr_scheduler is not None:
            booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
        
        dist.barrier()
    return running_states["epoch"], running_states["step"], running_states["sample_start_index"]


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if (is_initialized() and dist.get_rank() == 0):  # real logger
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

def remap_prefix(sd, src_prefix="decoders4.", dst_prefix="decoders.0."):
    """
    仅当键以 src_prefix 开头时才改名，避免误替换中间子串。
    注意：通常目标前缀应写成 'decoders.0.'（末尾带点），这样能无缝接上后续子键。
    """
    new_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith(src_prefix):
            new_k = dst_prefix + k[len(src_prefix):]
        else:
            new_k = k
        if new_k in new_sd:
            raise RuntimeError(f"Key collision after remap: '{new_k}' 已存在（请检查映射是否合理）")
        new_sd[new_k] = v
    return new_sd


def load_checkpoint(model, ckpt_path, save_as_pt=True, strict=True,src_prefix=None, dst_prefix=None):
    """
    Load model checkpoint with flexible support for .pt/.pth files and sharded directories.

    Args:
        model (torch.nn.Module): Target model to load weights into.
        ckpt_path (str): Path to checkpoint file (.pt/.pth) or directory (sharded checkpoint).
        save_as_pt (bool): Whether to save a `.pt` version after loading from sharded checkpoint.
        strict (bool): Whether to strictly enforce key match when loading state_dict.
    """
    def add_module_prefix(state_dict):
        return {'module.' + k: v for k, v in state_dict.items() if not k.startswith('module.')}

    if ckpt_path.endswith(('.pt', '.pth')):
        state_dict = find_model(ckpt_path)

        # Handle wrapper key (e.g., 'model')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        # Adjust module prefix
        model_keys = list(model.state_dict().keys())
        model_uses_module = model_keys[0].startswith('module.')
        ckpt_uses_module = next(iter(state_dict)).startswith('module.')

        if model_uses_module and not ckpt_uses_module:
            state_dict = add_module_prefix(state_dict)

        # Load
        if 'module' in state_dict.keys():
            if src_prefix != None:
                state_dict['module'] = remap_prefix(state_dict['module'], src_prefix=src_prefix, dst_prefix=dst_prefix)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict['module'], strict=strict)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            
        print(f"[Checkpoint Load] Missing keys: {missing_keys}")
        print(f"[Checkpoint Load] Unexpected keys: {unexpected_keys}")

    elif os.path.isdir(ckpt_path):
        load_from_sharded_state_dict(model, ckpt_path)

        if save_as_pt:
            save_path = os.path.join(ckpt_path, "model_ckpt.pt")
            torch.save(model.state_dict(), save_path)
            print(f"[Checkpoint Saved] Converted checkpoint saved to {save_path}")

    else:
        raise ValueError(f"[Error] Invalid checkpoint path: {ckpt_path}")
    
def get_loaded_param_names(ckpt_path, src_prefix, dst_prefix, model) -> Set[str]:
    
    obj = torch.load(ckpt_path, map_location="cpu")
    sd =  obj.get("module", obj)
    if not isinstance(sd, dict):
        sd = obj  

    mapped = set()
    for k in sd.keys():
        k_m = k
        if src_prefix and k.startswith(src_prefix):
            k_m = dst_prefix + k[len(src_prefix):]
        mapped.add(k_m)

    param_names = {n for n, _ in model.named_parameters()}
    return mapped & param_names