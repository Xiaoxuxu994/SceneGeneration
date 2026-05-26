import torch
import torch.nn as nn
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)



def format_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def create_optimizer(model: nn.Module, weight_decay, learning_rate) -> torch.optim.Optimizer:
    """Create optimizer. 
    
    Reference to https://github.com/Haian-Jin/LVSM/blob/main/utils/training_utils.py#L33.

    Args:
        model (nn.Module): _description_
        weight_decay (_type_): _description_
        learning_rate (_type_): _description_
    """
    all_param_dict = {name: param for name, param in model.named_parameters()}
    optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}
    
    decay_params, decay_params_small_lr, camera_params, nodecay_params = [], [], [], []
    for name, param in optimized_param_dict.items():
        if param.dim() == 1 or getattr(param, '_no_weight_decay', False):
            nodecay_params.append(param)
        else:
            if name.startswith("depth_aligner.image_tokenizer") or name.startswith("depth_aligner.encoder"):
                decay_params_small_lr.append(param)
            # elif name.startswith("depth_aligner.camera_head") or name.startswith("depth_aligner.camera_decoder"):
            #     camera_params.append(param)
            else:
                decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': decay_params_small_lr, 'weight_decay': weight_decay, 'lr': learning_rate * 1e-2}, 
        # {'params': camera_params, 'weight_decay': weight_decay, 'lr': learning_rate * 0.2}, 
        {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}, 
    ]
    
    # specified_param_group_dicts = {
    #     learning_rate * 1e-2: ["depth_aligner.image_tokenizer", "depth_aligner.encoder"], 
    #     learning_rate * 5e-1: ["depth_aligner.decoder", ], 
    # }
    # params_collected = {}
    # for name, param in optimized_param_dict.items():
    #     registered = False
    #     for lr, name_list in specified_param_group_dicts.items():
    #         for module_name in name_list:
    #             if name.startswith(module_name):
    #                 if params_collected.get(lr, None) is None:
    #                     params_collected[lr] = []
    #                 params_collected[lr].append(param)
    #                 registered = True
    #                 break
    #         if registered:
    #             break
    #     if not registered:
    #         if params_collected.get(learning_rate, None) is None:
    #             params_collected[learning_rate] = []
    #         params_collected[learning_rate].append(param)
    # optim_groups = []
    # for lr, params in params_collected.items():
    #     optim_groups.append({'params': params, 'weight_decay': weight_decay, 'lr': lr})
    
    # use fused AdamW optimizer by default. 
    # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    optimizer = torch.optim.AdamW(optim_groups)
    
    def get_module_name(name):
        parts = name.split('.')
        if len(parts) > 2 and parts[0] == 'module':
            return parts[1] + '.' + parts[2]
        return parts[0]  # Fallback to first part if no 'module.' prefix
    print(f'Optimizer: AdamW, learning rate: {learning_rate}, weight decay: {weight_decay}')
    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in optimized_param_dict.values())
    optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
    frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))
    
    print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')        
    print(f'Optimized parameters: {optim_module_names}')
    print(f'Frozen parameters: {frozen_module_names}')
    
    return optimizer


def create_lr_scheduler(optimizer, param_update_steps, warm_up_steps, scheduler_type='cosine'):
    """Bind LR scheduler for optimizer. Reference to https://github.com/Haian-Jin/LVSM/blob/main/utils/training_utils.py#L72.

    Args:
        optimizer (torch.data.optimizer): Optimizer to bind scheduler for.
        param_update_steps (int): Total training steps.
        warm_up_steps (int): Warm up steps.
    
    Returns:
        The lr scheduler binded to the optimizer. 
    """
    if scheduler_type == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, param_update_steps)
    elif scheduler_type == 'constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warm_up_steps)
    else:
        raise ValueError(f'Invalid scheduler type: {scheduler_type}')
    return scheduler
