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


def create_optimizer(model: nn.Module, weight_decay, learning_rate, encoder_lr_ratio=None, decoder_lr_ratio=None) -> torch.optim.Optimizer:
    """Create optimizer. 
    
    Reference to https://github.com/Haian-Jin/LVSM/blob/main/utils/training_utils.py#L33.

    Args:
        model (nn.Module): _description_
        weight_decay (_type_): _description_
        learning_rate (_type_): _description_
    """
    if encoder_lr_ratio == 0:
        model.encoder.eval().requires_grad_(False)
    if decoder_lr_ratio == 0:
        model.register_token.requires_grad_(False)
        model.decoder.eval().requires_grad_(False)
    if hasattr(model, 'conf_decoder'): model.conf_decoder.eval().requires_grad_(False)
    if hasattr(model, 'conf_head'): model.conf_head.eval().requires_grad_(False)

    def nodecay(name, param) -> bool:
        return param.dim() == 1 or getattr(param, '_no_weight_decay', False) \
            or any(n in name for n in ('pos_embed', 'cls_token', 'register'))

    encoder_params, encoder_nodecay_params = [], []
    decoder_params, decoder_nodecay_params = [], []
    downstream_params, downstream_nodecay_params = [], []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith('encoder.'):
                if nodecay(name, param): encoder_nodecay_params.append(param)
                else: encoder_params.append(param)
            elif name == 'register_token' or name.startswith('decoder.'):
                if nodecay(name, param): decoder_nodecay_params.append(param)
                else: decoder_params.append(param)
            else:
                if nodecay(name, param): downstream_nodecay_params.append(param)
                else: downstream_params.append(param)

    if encoder_lr_ratio is None: encoder_lr_ratio = 1.
    if decoder_lr_ratio is None: decoder_lr_ratio = 1.

    optim_groups = [
        {'params': encoder_params, 'weight_decay': weight_decay, 'lr': learning_rate * encoder_lr_ratio},
        {'params': encoder_nodecay_params, 'weight_decay': 0., 'lr': learning_rate * encoder_lr_ratio},
        {'params': decoder_params, 'weight_decay': weight_decay, 'lr': learning_rate * decoder_lr_ratio},
        {'params': decoder_nodecay_params, 'weight_decay': 0., 'lr': learning_rate * decoder_lr_ratio},
        {'params': downstream_params, 'weight_decay': weight_decay, 'lr': learning_rate},
        {'params': downstream_nodecay_params, 'weight_decay': 0., 'lr': learning_rate},
    ]
    optim_groups = [group for group in optim_groups if group['params']]

    optimizer = torch.optim.AdamW(optim_groups)
    
    def get_module_name(name):
        parts = name.split('.')
        if len(parts) > 2 and parts[0] == 'module':
            return parts[1] + '.' + parts[2]
        return parts[0]  # Fallback to first part if no 'module.' prefix
    print(f'Optimizer: AdamW, learning rate: {learning_rate}, weight decay: {weight_decay}')
    # Number of parameters
    all_param_dict = {name: param for name, param in model.named_parameters()}
    optimized_param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in optimized_param_dict.values())
    optim_module_names = sorted(set(get_module_name(name) for name in optimized_param_dict.keys()))
    frozen_module_names = sorted(set(get_module_name(name) for name in set(all_param_dict.keys()) - set(optimized_param_dict.keys())))
    
    print(f'Total parameters: {format_number(total_params)}, Trainable parameters: {format_number(trainable_params)}')        
    print(f'Optimized parameters: {optim_module_names}')
    print(f'Frozen parameters: {frozen_module_names}')
    for group in optim_groups:
        print(f'Optimizer group: {len(group["params"])} tensors with lr={group["lr"]:.2} and decay={group["weight_decay"]}')

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
