import os
import torch

def find_model(model_name):
  # Load a custom checkpoint:
    assert os.path.isfile(model_name), f"Could not find checkpoint at {model_name}"
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "pos_embed_temporal" in checkpoint:
        del checkpoint["pos_embed_temporal"]
    if "pos_embed" in checkpoint:
        del checkpoint["pos_embed"]
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint

def load_checkpoint(model, ckpt_path, strict=True): 
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path)
        if 'module' in list(model.state_dict().items())[0][0]:
            contains_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
            if not contains_module_prefix:
                new_state_dict = {}
                for key, value in state_dict.items():
                    if 'module' not in key:
                        new_key = 'module.' + key
                        new_state_dict[new_key] = value
                state_dict = new_state_dict
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
