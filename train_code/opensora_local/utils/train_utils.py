'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-10-28 13:03:00
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-10-30 10:48:27
FilePath: /proj_xdgs/data02/z84301856/devel/xDGS/proj_xdgs/opensora_vae/opensora_local/utils/train_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from collections import OrderedDict

import torch


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())    
    
    for name, param in model_params.items():
        # name = name.replace('module.module.', 'module.')
        name = name.replace('module.', '')
        # if name == "pos_embed":
        if "pos_embed" in name:
            continue
        if param.requires_grad == False:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            # if param.data.dtype != torch.float32:
            #     param_id = id(param)
            #     # master_param = optimizer._param_store.working_to_master_param[param_id]
            #     master_param = optimizer.working_to_master_param[param_id] # NOTE: 修改了
            #     param_data = master_param.data
            # else:
            #     param_data = param.data
            # # print(name, param_data.shape)

            param_id = id(param)
            # master_param = optimizer._param_store.working_to_master_param[param_id]
            master_param = optimizer.working_to_master_param[param_id] # NOTE: 配套 pip install colossalai==0.4.5
            param_data = master_param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
