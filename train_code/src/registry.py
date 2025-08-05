from copy import deepcopy

import torch.nn as nn
from mmengine.registry import Registry

def build_dataset_kwargs(cfg):
    kwargs = {}
    if hasattr(cfg, "vs_res"):
        kwargs["vs_res"] = cfg.vs_res
    if hasattr(cfg, "query_sample_rate"):
        kwargs["query_sample_rate"] = cfg.query_sample_rate
    if hasattr(cfg, "max_query_number"):
        kwargs["max_query_number"] = cfg.max_query_number
    if getattr(cfg, "query_class_balance_sample", False):
        kwargs["query_class_balance_sample"] = True
        kwargs["pos_neg_ratio"] = getattr(cfg, "pos_neg_ratio", 0.5)
        kwargs["max_pos_keep"] = getattr(cfg, "max_pos_keep", 100000)
    if hasattr(cfg, "datatype"):
        kwargs["datatype"] = cfg.datatype
    if getattr(cfg, "transform", False):
        kwargs["transform"] = cfg.transform
    return kwargs

def build_module(module, builder, **kwargs):
    """Build module from config or return the module itself.

    Args:
        module (Union[dict, nn.Module]): The module to build.
        builder (Registry): The registry to build module.
        *args, **kwargs: Arguments passed to build function.

    Returns:
        Any: The built module.
    """
    if isinstance(module, dict):
        cfg = deepcopy(module)
        for k, v in kwargs.items():
            cfg[k] = v
        return builder.build(cfg)
    elif isinstance(module, nn.Module):
        return module
    elif module is None:
        return None
    else:
        raise TypeError(f"Only support dict and nn.Module, but got {type(module)}.")

MODELS = Registry(
    "model",
    locations=["src.models"],
    scope="src"
)

SCHEDULERS = Registry(
    "scheduler",
    locations=["src.schedulers"],
    scope="src"
)

DATASETS = Registry(
    "dataset",
    locations=["src.dataset"],
    scope="src"
)

TRANSFORMS = Registry(
    "transform",
    locations=["src.dataset.transform"],
    scope="src"
)